# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for tools/real_postgres_guard.py — the gate on the real-Postgres CI job.

The job runs the gated real-substrate tests against a real Postgres server and
hands the guard pytest's JUnit report. A green pytest exit proves little there:
every gated test skips when its environment variable is missing, so a job that
lost its database would pass with eight skips. The guard instead requires each
named test to have passed, allows exactly two named skips for their stated
reasons, and fails on anything else.

The fixture XML below copies the shapes pytest 9 writes: a skip in setup, a
blank ``pytest.skip()`` (message "Skipped"), a failure, a setup error, a
teardown error after a pass (one testcase with an ``<error>``), a failure plus
a teardown error (the same testcase reported twice), an xfail
(``type="pytest.xfail"``) and a module that skips itself at collection
(``classname=""``, no type). Every case asserts the exit code AND the printed
reason, and "cannot tell" (exit 2) never comes back as a pass.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL_PATH = _REPO_ROOT / "tools" / "real_postgres_guard.py"

_ROW = "tests.adapters.test_coherent_row"
_TIERS = "tests.conformance.test_tier_honesty"
# Literal copies of the guard's tables: changing either is a reviewed decision,
# so it has to be made twice.
MUST_PASS = frozenset(
    {
        (_ROW, "test_it_concurrent_lost_update_one_winner_then_converge"),
        (_ROW, "test_it_trigger_authority_ignores_client_new_version"),
        (_ROW, "test_it_foreign_write_moves_version_agent_cas_fails"),
        (_ROW, "test_it_unknown_write_reconciles_on_token_identity"),
        (_ROW, "test_it_xmin_wraparound_is_the_missed_conflict_hazard"),
        (_TIERS, "test_real_postgres_native_cas_conformance"),
    }
)
LIMITED_ROLE = (_ROW, "test_it_limited_role_cannot_disable_the_trigger")
REAL_S3 = (_TIERS, "test_real_s3_native_cas_conformance")
ALLOWED_SKIP = {LIMITED_ROLE: "CCS_TEST_PG_LIMITED_DSN", REAL_S3: "CCS_REAL_S3_BUCKET"}
SKIP_MESSAGES = {
    LIMITED_ROLE: "set CCS_TEST_PG_LIMITED_DSN (non-owner coherence role) to run this test",
    REAL_S3: "set CCS_REAL_S3_BUCKET (a real, non-Moto bucket) to run the real S3 arm",
}
FOREIGN_WRITE = (_ROW, "test_it_foreign_write_moves_version_agent_cas_fails")
DSN_SKIP = "set CCS_TEST_PG_DSN (owner DSN) to run real Postgres tests"

PASSED = ""
FAILURE = '<failure message="assert 1 == 2">E   assert 1 == 2</failure>'
SETUP_ERROR = '<error message="failed on setup with &quot;RuntimeError: boom&quot;">E   RuntimeError</error>'
TEARDOWN_ERROR = '<error message="failed on teardown with &quot;RuntimeError: boom&quot;">E   RuntimeError</error>'
XFAIL = '<skipped type="pytest.xfail" message="known" />'


def _skip(message: str) -> str:
    return f'<skipped type="pytest.skip" message="{message}">tests/x.py:1: {message}</skipped>'


def _untyped_skip(message: str) -> str:
    return f'<skipped message="{message}">tests/x.py:1: {message}</skipped>'


Case = tuple[tuple[str, str], str]


def _healthy() -> list[Case]:
    """Six passes and the two allowed skips, as the job's real run reports them."""
    passes = [(key, PASSED) for key in sorted(MUST_PASS)]
    return passes + [(key, _skip(SKIP_MESSAGES[key])) for key in sorted(ALLOWED_SKIP)]


def _replace(cases: list[Case], key: tuple[str, str], child: str) -> list[Case]:
    return [(case_key, child if case_key == key else case_child) for case_key, case_child in cases]


def _without(cases: list[Case], key: tuple[str, str]) -> list[Case]:
    return [case for case in cases if case[0] != key]


def _write_report(path: Path, cases: list[Case]) -> Path:
    body = "".join(
        f'<testcase classname="{classname}" name="{name}" time="0.001">{child}</testcase>'
        for (classname, name), child in cases
    )
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests">'
        f'<testsuite name="pytest" tests="{len(cases)}">{body}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return path


def _import_guard():
    spec = importlib.util.spec_from_file_location("real_postgres_guard", _TOOL_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def guard():
    return _import_guard()


def _judge(guard, tmp_path: Path, cases: list[Case], capsys) -> tuple[int, list[str]]:
    report = _write_report(tmp_path / "junit.xml", cases)
    exit_code = guard.main(["--junit-xml", str(report)])
    return exit_code, capsys.readouterr().out.splitlines()


def _node(key: tuple[str, str]) -> str:
    return f"{key[0]}::{key[1]}"


def test_the_tables_are_the_pinned_six_and_two(guard):
    assert len(guard.MUST_PASS) == 6
    assert len(guard.ALLOWED_SKIP) == 2
    assert guard.MUST_PASS == MUST_PASS
    assert dict(guard.ALLOWED_SKIP) == ALLOWED_SKIP


def test_six_passes_and_two_expected_skips_pass(guard, tmp_path, capsys):
    exit_code, lines = _judge(guard, tmp_path, _healthy(), capsys)
    assert exit_code == 0
    assert lines[0].startswith("PASS")
    assert "6 passed" in lines[0]
    assert "2 skipped" in lines[0]


@pytest.mark.parametrize("key", sorted(MUST_PASS))
def test_a_must_pass_test_that_skipped_fails_naming_it(guard, tmp_path, capsys, key):
    cases = _replace(_healthy(), key, _skip(DSN_SKIP))
    exit_code, lines = _judge(guard, tmp_path, cases, capsys)
    assert exit_code == 1
    assert lines[0].startswith("FAIL")
    assert any(_node(key) in line and "skipped" in line and DSN_SKIP in line for line in lines)


@pytest.mark.parametrize(
    ("child", "outcome"),
    [(FAILURE, "failed"), (SETUP_ERROR, "error"), (TEARDOWN_ERROR, "error"), (XFAIL, "xfail")],
    ids=["failure", "setup-error", "teardown-error-after-pass", "xfail"],
)
def test_a_must_pass_test_that_did_not_pass_fails_naming_it(guard, tmp_path, capsys, child, outcome):
    cases = _replace(_healthy(), FOREIGN_WRITE, child)
    exit_code, lines = _judge(guard, tmp_path, cases, capsys)
    assert exit_code == 1
    assert lines[0].startswith("FAIL")
    assert any(_node(FOREIGN_WRITE) in line and outcome in line for line in lines)


def test_a_failure_plus_a_teardown_error_fails_on_both_reports(guard, tmp_path, capsys):
    cases = _replace(_healthy(), FOREIGN_WRITE, FAILURE) + [(FOREIGN_WRITE, TEARDOWN_ERROR)]
    exit_code, lines = _judge(guard, tmp_path, cases, capsys)
    assert exit_code == 1
    assert any(_node(FOREIGN_WRITE) in line and "reported 2 times" in line for line in lines)
    assert any(_node(FOREIGN_WRITE) in line and "failed" in line for line in lines)


def test_a_test_reported_twice_fails_even_when_both_reports_passed(guard, tmp_path, capsys):
    cases = _healthy() + [(FOREIGN_WRITE, PASSED)]
    exit_code, lines = _judge(guard, tmp_path, cases, capsys)
    assert exit_code == 1
    assert any(_node(FOREIGN_WRITE) in line and "reported 2 times" in line for line in lines)


@pytest.mark.parametrize("key", sorted(MUST_PASS | set(ALLOWED_SKIP)))
def test_a_table_entry_missing_from_the_report_fails_naming_it(guard, tmp_path, capsys, key):
    exit_code, lines = _judge(guard, tmp_path, _without(_healthy(), key), capsys)
    assert exit_code == 1
    assert lines[0].startswith("FAIL")
    assert any(_node(key) in line and "missing" in line for line in lines)


@pytest.mark.parametrize("key", sorted(ALLOWED_SKIP))
@pytest.mark.parametrize(
    ("child", "outcome"),
    [
        (PASSED, "passed"),
        (FAILURE, "failed"),
        (_skip("set SOMETHING_ELSE to run this test"), "SOMETHING_ELSE"),
        (_skip("Skipped"), "Skipped"),
    ],
    ids=["passed", "failed", "other-reason", "blank-skip"],
)
def test_an_allowed_skip_that_did_not_skip_for_its_reason_fails_naming_it(
    guard, tmp_path, capsys, key, child, outcome
):
    exit_code, lines = _judge(guard, tmp_path, _replace(_healthy(), key, child), capsys)
    assert exit_code == 1
    assert lines[0].startswith("FAIL")
    assert any(_node(key) in line and outcome in line and ALLOWED_SKIP[key] in line for line in lines)


@pytest.mark.parametrize(
    "child",
    [
        f'<skipped type="pytest.xfail" message="{SKIP_MESSAGES[REAL_S3]}" />',
        _untyped_skip(SKIP_MESSAGES[REAL_S3]),
    ],
    ids=["xfail", "untyped"],
)
def test_an_allowed_skip_with_its_reason_but_not_a_plain_skip_fails(guard, tmp_path, capsys, child):
    exit_code, lines = _judge(guard, tmp_path, _replace(_healthy(), REAL_S3, child), capsys)
    assert exit_code == 1
    assert any(_node(REAL_S3) in line for line in lines[1:])


def test_an_unclassified_test_fails_naming_it(guard, tmp_path, capsys):
    newcomer = (_ROW, "test_it_a_new_gated_test")
    exit_code, lines = _judge(guard, tmp_path, _healthy() + [(newcomer, PASSED)], capsys)
    assert exit_code == 1
    assert any(_node(newcomer) in line and "unclassified" in line for line in lines)


def test_a_module_that_skipped_itself_at_collection_fails_as_unclassified(guard, tmp_path, capsys):
    module = ("", "tests.adapters.test_coherent_row")
    cases = _healthy() + [(module, _untyped_skip("collection skipped"))]
    exit_code, lines = _judge(guard, tmp_path, cases, capsys)
    assert exit_code == 1
    assert any("tests.adapters.test_coherent_row" in line and "unclassified" in line for line in lines)


def test_named_tests_are_capped_with_a_count_of_the_rest(guard, tmp_path, capsys):
    extras = [((_ROW, f"test_extra_{index:02d}"), PASSED) for index in range(25)]
    exit_code, lines = _judge(guard, tmp_path, _healthy() + extras, capsys)
    assert exit_code == 1
    assert sum("test_extra_" in line for line in lines) == guard.MAX_NAMED
    assert any(f"and {25 - guard.MAX_NAMED} more" in line for line in lines)


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (None, "no JUnit report"),
        ("<testsuites><testsuite>", "unreadable"),
        ('<?xml version="1.0"?><testsuites><testsuite name="pytest" tests="0"/></testsuites>', "no testcases"),
    ],
    ids=["no-file", "unparseable", "zero-testcases"],
)
def test_a_report_that_proves_nothing_cannot_tell(guard, tmp_path, capsys, content, reason):
    report = tmp_path / "junit.xml"
    if content is not None:
        report.write_text(content, encoding="utf-8")
    exit_code = guard.main(["--junit-xml", str(report)])
    lines = capsys.readouterr().out.splitlines()
    assert exit_code == 2
    assert lines[0].startswith("CANNOT TELL")
    assert reason in lines[0]


@pytest.mark.parametrize(
    ("cases", "expected"),
    [
        (_healthy(), 0),
        (_replace(_healthy(), FOREIGN_WRITE, FAILURE), 1),
        (None, 2),
    ],
    ids=["pass", "fail", "cannot-tell"],
)
def test_the_cli_on_a_bare_interpreter_returns_the_same_codes(tmp_path, cases, expected):
    report = tmp_path / "junit.xml"
    if cases is not None:
        _write_report(report, cases)
    # -I -S: no site-packages and no user paths, as on the runner's system python3.
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(_TOOL_PATH), "--junit-xml", str(report)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == expected, result.stdout + result.stderr
    assert result.stdout.splitlines()[0].split(":")[0] in {"PASS", "FAIL", "CANNOT TELL"}
