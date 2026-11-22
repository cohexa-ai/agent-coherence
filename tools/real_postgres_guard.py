# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Gate: prove the real-Postgres CI job ran its gated tests rather than skipped them.

Every real-substrate test skips itself when its environment variable is unset,
so a pytest exit of 0 proves nothing on its own: a job that lost its database
would report eight skips and stay green. This gate reads the job's JUnit report
and requires each test in MUST_PASS to have passed, allows each test in
ALLOWED_SKIP to skip only for its stated reason, and fails on anything else.

Tests are keyed on the report's (classname, name) pair, which is unique per
test function in pytest's JUnit output (it carries no file path).

Verdicts (the first printed line names it):
    exit 2  CANNOT TELL  no report at the path, the report is unreadable, or it
                         holds no testcases
    exit 1  FAIL         a MUST_PASS test failed, errored (in setup or teardown
                         too), skipped or xfailed; an ALLOWED_SKIP test passed,
                         failed, or skipped for another reason (a blank
                         ``pytest.skip()`` reads "Skipped"); a test reported
                         more than once; a test in neither table (a module that
                         skipped itself at collection lands here, with an empty
                         classname); a table entry missing from the report
    exit 0  PASS         otherwise

Adding a gated test file to the job means classifying each of its tests here.
A test that cannot run in the job as written moves to ALLOWED_SKIP only with an
explicit skip gated on an environment variable the job does not set, whose
reason names that variable; a test that merely fails stays in MUST_PASS and
keeps the job red.

Usage:
    python3 tools/real_postgres_guard.py --junit-xml PATH
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CANNOT_TELL = 2
# Enough to start debugging; a report from the wrong run can name many tests.
MAX_NAMED = 20

_ROW = "tests.adapters.test_coherent_row"
_TIERS = "tests.conformance.test_tier_honesty"
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
# Each value is a fragment the skip's message must contain: the variable whose
# absence is the only accepted reason to skip.
ALLOWED_SKIP = MappingProxyType(
    {
        (_ROW, "test_it_limited_role_cannot_disable_the_trigger"): "CCS_TEST_PG_LIMITED_DSN",
        (_TIERS, "test_real_s3_native_cas_conformance"): "CCS_REAL_S3_BUCKET",
    }
)
PASSED = "passed"
SKIPPED = "skipped"
_SKIP_OUTCOMES = {"pytest.skip": SKIPPED, "pytest.xfail": "xfailed", "": "untyped skip"}


class Report(NamedTuple):
    """One ``<testcase>`` element: which test, what happened, and pytest's message for it."""

    key: tuple[str, str]
    outcome: str
    message: str


class CannotTell(Exception):
    """The report proves nothing either way."""


def to_report(case: ET.Element) -> Report:
    key = (case.get("classname", ""), case.get("name", ""))
    # A failure or error outranks a skip in the same testcase.
    for tag, outcome in (("failure", "failed"), ("error", "errored")):
        child = case.find(tag)
        if child is not None:
            return Report(key, outcome, child.get("message", ""))
    skipped = case.find("skipped")
    if skipped is None:
        return Report(key, PASSED, "")
    skip_type = skipped.get("type", "")
    return Report(key, _SKIP_OUTCOMES.get(skip_type, f"skip of type {skip_type}"), skipped.get("message", ""))


def read_reports(path: Path) -> list[Report]:
    try:
        root = ET.parse(path).getroot()
    except FileNotFoundError as exc:
        raise CannotTell(f"no JUnit report at {path}") from exc
    except (OSError, ET.ParseError) as exc:
        raise CannotTell(f"JUnit report unreadable ({type(exc).__name__}) at {path}") from exc
    reports = [to_report(case) for case in root.iter("testcase")]
    if not reports:
        raise CannotTell(f"no testcases in {path}")
    return reports


def node_name(key: tuple[str, str]) -> str:
    classname, name = key
    return f"{classname}::{name}" if classname else name


def judge_one(report: Report) -> str | None:
    """The problem with one testcase, or None when it did what its table requires."""
    node = node_name(report.key)
    seen = f"{report.outcome}: {report.message}" if report.message else report.outcome
    if report.key in MUST_PASS:
        return None if report.outcome == PASSED else f"{node}: must pass, but {seen}"
    reason = ALLOWED_SKIP.get(report.key)
    if reason is None:
        return f"{node}: unclassified ({seen}); add it to MUST_PASS or ALLOWED_SKIP in this tool"
    if report.outcome == SKIPPED and reason in report.message:
        return None
    return f"{node}: may skip only for {reason}, but {seen}"


def judge(reports: Sequence[Report]) -> list[str]:
    """Every problem with the run, in report order; empty when the run proves what it must."""
    counts = Counter(report.key for report in reports)
    expected = MUST_PASS | ALLOWED_SKIP.keys()
    findings = [f"{node_name(key)}: reported {count} times" for key, count in counts.items() if count > 1]
    findings += [f"{node_name(key)}: missing from the report" for key in sorted(expected - counts.keys())]
    return findings + [problem for problem in map(judge_one, reports) if problem is not None]


def capped(findings: Sequence[str]) -> list[str]:
    lines = [f"  {finding}" for finding in findings[:MAX_NAMED]]
    if len(findings) > MAX_NAMED:
        lines.append(f"  ... and {len(findings) - MAX_NAMED} more")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prove the real-Postgres job ran its gated tests.")
    parser.add_argument("--junit-xml", type=Path, required=True, help="pytest's JUnit report from the job.")
    args = parser.parse_args(argv)
    try:
        reports = read_reports(args.junit_xml)
    except CannotTell as exc:
        print(f"CANNOT TELL: {exc}")
        return EXIT_CANNOT_TELL
    findings = judge(reports)
    if findings:
        print(f"FAIL: {len(findings)} problem(s) with the gated tests")
        print("\n".join(capped(findings)))
        return EXIT_FAIL
    print(f"PASS: {len(MUST_PASS)} passed, {len(ALLOWED_SKIP)} skipped for their stated reasons")
    return EXIT_PASS


if __name__ == "__main__":
    raise SystemExit(main())
