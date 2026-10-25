# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Unit 7 tests — ``ccs-diagnose`` CLI integration.

Coverage map:

* End-to-end pipeline against ``build_graph_no_store`` (the load-bearing
  integration test).
* Argparse / flag handling — required-flag rules, CSV parsing,
  invalid-choice rejection.
* Subcommand-style flags (``--show-payload``, ``--reset-token``).
* Trust-posture flag stubs (``--dry-run`` etc.).
* Defensive / edge cases (graph errors, empty buffers, custom token cost).
* Determinism — repeated runs produce byte-identical artefacts.

Tests prefer direct ``main(argv)`` calls for speed; one subprocess test
verifies the ``python -m ccs.cli.diagnose`` entrypoint actually runs.
"""

from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from ccs.diagnose import CCS_DIAGNOSE_LOG_SCHEMA_VERSION

GRAPH_PATH = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "langgraph_planner"
    / "main.py"
)
GRAPH_FACTORY = f"{GRAPH_PATH}:build_graph_no_store"


def _have_langgraph() -> bool:
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _have_langgraph(),
    reason="ccs-diagnose CLI tests require the [diagnose] extra (langgraph + jinja2).",
)


# -------------------------------------------------------------------- #
# Helpers
# -------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _isolate_consent(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ensure CLI tests never touch the user's real consent file.

    Sets ``XDG_CONFIG_HOME`` to a per-test temp dir and ``CI=1`` so the
    consent resolver never prompts (Unit 8 makes the resolver
    non-interactive when ``CI`` is truthy).
    """
    cfg_dir = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    monkeypatch.setenv("CI", "1")
    # Defensive: clear kill switches so the few tests that exercise the
    # actual flow don't see an inherited DO_NOT_TRACK from the developer's
    # shell.
    for name in ("DO_NOT_TRACK", "DISABLE_TELEMETRY", "CCS_DIAGNOSE_NO_TELEMETRY"):
        monkeypatch.delenv(name, raising=False)


def _invoke(argv: list[str]) -> tuple[int, str, str]:
    """Call ``main(argv)`` and capture stdout/stderr."""
    # Late import so the skip marker can short-circuit when langgraph is
    # missing.
    from ccs.cli.diagnose import main

    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            code = main(argv)
    except SystemExit as exc:
        code = int(exc.code) if exc.code is not None else 0
    return code, out_buf.getvalue(), err_buf.getvalue()


def _basic_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--graph",
        GRAPH_FACTORY,
        "--output-html",
        str(tmp_path / "r.html"),
        "--output-json",
        str(tmp_path / "r.json"),
        *extra,
    ]


# -------------------------------------------------------------------- #
# 1-6: End-to-end pipeline
# -------------------------------------------------------------------- #


def test_pipeline_emits_html_and_json(tmp_path: Path) -> None:
    code, stdout, _ = _invoke(_basic_argv(tmp_path))
    assert code == 0, stdout
    html_path = tmp_path / "r.html"
    json_path = tmp_path / "r.json"
    assert html_path.exists()
    assert json_path.exists()
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html.lower()
    assert "Your write pattern" in html
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == CCS_DIAGNOSE_LOG_SCHEMA_VERSION
    assert "verdict" in payload
    assert "report" in payload
    # Terminal summary printed.
    assert "Your write pattern" in stdout
    assert str(html_path) in stdout


def test_pipeline_with_volume_renders_cost(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--volume", "50"))
    assert code == 0
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    # When --volume is set, the cost-KPI path renders. Even when token
    # estimates are unavailable the renderer surfaces an explanatory
    # block; assert the cost-side wiring fired.
    assert "cost" in html.lower()


def test_pipeline_strict_propagates_to_report(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--strict"))
    assert code == 0
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert payload["report"]["strict_mode"] is True


def test_pipeline_lead_pain_type_cost_no_volume_renders_unmeasurable(
    tmp_path: Path,
) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--lead-pain-type", "cost"))
    assert code == 0
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    # Cost-unmeasurable copy fires when value_token_estimates is missing.
    assert "Rework cost" in html or "cost cannot be measured" in html.lower()


def test_pipeline_warm_lead_renders_warm_cta(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--warm-lead"))
    assert code == 0
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    # Warm-lead variant: no soft-ask, has 2 seed questions.
    assert "30-min" not in html
    # The seed-question helper produces two questions ending with '?'.
    assert html.count("?") >= 2


def test_pipeline_no_json_suppresses_json_output(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--no-json"))
    assert code == 0
    assert (tmp_path / "r.html").exists()
    assert not (tmp_path / "r.json").exists()


# -------------------------------------------------------------------- #
# 7-11: Argparse / flag handling
# -------------------------------------------------------------------- #


def test_no_flags_errors_on_missing_graph() -> None:
    code, _, stderr = _invoke([])
    assert code == 2
    assert "--graph" in stderr


def test_nonexistent_module_path_errors(tmp_path: Path) -> None:
    code, _, stderr = _invoke(
        [
            "--graph",
            "nonexistent.py:foo",
            "--output-html",
            str(tmp_path / "r.html"),
            "--no-json",
        ]
    )
    assert code == 1
    assert "graph file not found" in stderr.lower()


def test_nonexistent_function_errors(tmp_path: Path) -> None:
    code, _, stderr = _invoke(
        [
            "--graph",
            f"{GRAPH_PATH}:does_not_exist",
            "--output-html",
            str(tmp_path / "r.html"),
            "--no-json",
        ]
    )
    assert code == 1
    assert "does_not_exist" in stderr


def test_csv_flags_parse_to_tuples(tmp_path: Path) -> None:
    # Smoke test: pipeline accepts CSV ignore/track without crashing.
    code, _, _ = _invoke(
        _basic_argv(tmp_path, "--ignore", "foo, bar", "--track", "baz")
    )
    assert code == 0
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    # Override-only names that don't appear in events are dropped by the
    # classifier's resolver — the report still serialises cleanly.
    assert "verdict" in payload


def test_invalid_lead_pain_type_rejected(tmp_path: Path) -> None:
    code, _, stderr = _invoke(_basic_argv(tmp_path, "--lead-pain-type", "wrong"))
    assert code == 2
    assert "invalid choice" in stderr.lower() or "wrong" in stderr.lower()


# -------------------------------------------------------------------- #
# 12-16: Subcommand-style flags
# -------------------------------------------------------------------- #


def test_show_payload_prints_payload(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path))
    assert code == 0
    json_path = tmp_path / "r.json"

    code, stdout, _ = _invoke(["--show-payload", str(json_path)])
    assert code == 0
    payload = json.loads(stdout)
    assert payload["schema_version"] == CCS_DIAGNOSE_LOG_SCHEMA_VERSION
    assert "verdict_bucket" in payload
    assert "coverage" in payload
    assert payload["installation_token"] is None  # Unit 8 stub


def test_show_payload_with_graph_warns_and_succeeds(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path))
    assert code == 0
    json_path = tmp_path / "r.json"

    code, stdout, stderr = _invoke(
        ["--show-payload", str(json_path), "--graph", "ignored.py:bar"]
    )
    assert code == 0
    assert "warning: --graph ignored when --show-payload is set" in stderr
    assert json.loads(stdout)["schema_version"] == CCS_DIAGNOSE_LOG_SCHEMA_VERSION


def test_show_payload_missing_file_errors(tmp_path: Path) -> None:
    code, _, stderr = _invoke(["--show-payload", str(tmp_path / "nope.json")])
    # I/O error: file not found.
    assert code == 5
    assert "cannot read" in stderr.lower()


def test_show_payload_schema_mismatch_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"random": "stuff"}), encoding="utf-8")
    code, _, stderr = _invoke(["--show-payload", str(bad)])
    # Schema mismatch: dedicated exit code (#25).
    assert code == 4
    assert "schema version mismatch" in stderr.lower()


def test_reset_token_prints_new_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unit 8: --reset-token regenerates consent.json with a fresh UUID4."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    code, stdout, _ = _invoke(["--reset-token"])
    assert code == 0
    assert "installation token regenerated:" in stdout
    consent_file = tmp_path / "ccs-diagnose" / "consent.json"
    assert consent_file.exists()
    consent_data = json.loads(consent_file.read_text(encoding="utf-8"))
    assert consent_data["granted"] is True
    assert consent_data["policy_version"] == 1
    # UUID4 string survives round-trip.
    import uuid as _uuid
    _uuid.UUID(consent_data["installation_token"])


# -------------------------------------------------------------------- #
# 17-20: Trust-posture flag stubs
# -------------------------------------------------------------------- #


def test_dry_run_prints_telemetry_payload(tmp_path: Path) -> None:
    code, stdout, stderr = _invoke(_basic_argv(tmp_path, "--dry-run"))
    assert code == 0
    # Dry-run routes the machine-readable JSON payload to stdout and the
    # human-readable summary to stderr so callers can pipe stdout straight
    # into ``jq`` etc.
    payload = json.loads(stdout.strip())
    assert payload["schema_version"] == CCS_DIAGNOSE_LOG_SCHEMA_VERSION
    assert payload["installation_token"] is None
    # The summary text always mentions the user's write pattern.
    assert "Your write pattern" in stderr


def test_no_network_is_noop_in_v0(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--no-network"))
    assert code == 0
    assert (tmp_path / "r.html").exists()


def test_no_telemetry_is_noop_in_v0(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--no-telemetry"))
    assert code == 0
    assert (tmp_path / "r.html").exists()


def test_calibration_record_skipped_without_consent(tmp_path: Path) -> None:
    """In CI (consent denied), --calibration-record prints a skip message."""
    cal = tmp_path / "cal.jsonl"
    code, stdout, _ = _invoke(
        _basic_argv(tmp_path, "--calibration-record", str(cal))
    )
    assert code == 0
    assert "calibration write skipped" in stdout
    # No file written when consent is denied.
    assert not cal.exists()


# -------------------------------------------------------------------- #
# 21-25: Defensive / edge cases
# -------------------------------------------------------------------- #


def test_volume_zero_is_treated_as_no_volume(tmp_path: Path) -> None:
    code, _, _ = _invoke(_basic_argv(tmp_path, "--volume", "0"))
    assert code == 0
    payload = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    # auto + volume==0 → auditability path; no annualised cost ridden.
    # (volume=0.0 still drives the detection cost extrapolation to 0.)
    assert payload["report"]["rework_cost_annualized"] == 0.0


def test_custom_token_cost_respected(tmp_path: Path) -> None:
    code, _, _ = _invoke(
        _basic_argv(
            tmp_path, "--volume", "50", "--cost-per-1k-tokens", "0.001"
        )
    )
    assert code == 0


def test_graph_invoke_failure_surfaces_warning(tmp_path: Path) -> None:
    """Graph that can't be invoked: pipeline finishes with insufficient verdict."""
    bad_module = tmp_path / "bad_graph.py"
    bad_module.write_text(
        """
def build_bad():
    class _Graph:
        def invoke(self, state, config=None):
            raise RuntimeError('boom')
    return _Graph()
""",
        encoding="utf-8",
    )
    out_json = tmp_path / "r.json"
    with pytest.warns(Warning, match="graph invoke failed"):
        code, stdout, _ = _invoke(
            [
                "--graph",
                f"{bad_module}:build_bad",
                "--output-html",
                str(tmp_path / "r.html"),
                "--output-json",
                str(out_json),
            ]
        )
    # The user still gets a (partial) report; exit 0 is the chosen
    # behaviour (graceful degradation).
    assert code == 0
    assert "Your write pattern" in stdout
    # The warning was also routed into the callback buffer (and onward to
    # the JSON report's verdict reason) so downstream stages have a clean
    # signal rather than only a Python warning.
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["verdict"]["bucket"] == "insufficient"


def test_writers_by_key_consistency_silent_for_insufficient_verdict(
    tmp_path: Path,
) -> None:
    """Insufficient verdicts have empty writers_by_key by design — no warning."""
    bad_module = tmp_path / "bad_graph.py"
    bad_module.write_text(
        """
def build_bad():
    class _Graph:
        def invoke(self, state, config=None):
            raise RuntimeError('boom')
    return _Graph()
""",
        encoding="utf-8",
    )
    with pytest.warns(Warning, match="graph invoke failed"):
        code, _, stderr = _invoke(
            [
                "--graph",
                f"{bad_module}:build_bad",
                "--output-html",
                str(tmp_path / "r.html"),
                "--no-json",
            ]
        )
    assert code == 0
    assert "writers_by_key inconsistency" not in stderr


def test_volume_rejects_inf(tmp_path: Path) -> None:
    code, _, stderr = _invoke(_basic_argv(tmp_path, "--volume", "inf"))
    assert code == 2
    assert "finite" in stderr.lower() or "non-negative" in stderr.lower()


def test_volume_rejects_nan(tmp_path: Path) -> None:
    code, _, stderr = _invoke(_basic_argv(tmp_path, "--volume", "nan"))
    assert code == 2
    assert "finite" in stderr.lower() or "non-negative" in stderr.lower()


def test_volume_rejects_negative(tmp_path: Path) -> None:
    code, _, stderr = _invoke(_basic_argv(tmp_path, "--volume", "-1"))
    assert code == 2
    assert "non-negative" in stderr.lower()


def test_report_json_has_no_duplicate_schema_version(tmp_path: Path) -> None:
    """``schema_version`` is at the top level only — never nested under report."""
    code, _, _ = _invoke(_basic_argv(tmp_path))
    assert code == 0
    loaded = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert loaded["schema_version"] == CCS_DIAGNOSE_LOG_SCHEMA_VERSION
    assert "schema_version" not in loaded["report"]


def test_html_output_path_unwritable_returns_nonzero(tmp_path: Path) -> None:
    """A read-only output dir surfaces a clean error and exits non-zero."""
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        code, _, stderr = _invoke(
            [
                "--graph",
                GRAPH_FACTORY,
                "--output-html",
                str(readonly / "r.html"),
                "--no-json",
            ]
        )
    finally:
        readonly.chmod(0o700)
    # I/O error: HTML write failed; --no-json + html_written=False ⇒ exit 5.
    assert code == 5
    assert "failed to write HTML report" in stderr


def test_writers_by_key_consistent_for_clean_run(tmp_path: Path) -> None:
    """No inconsistency warning fires for a clean run against the substrate."""
    code, _, stderr = _invoke(_basic_argv(tmp_path))
    assert code == 0
    assert "writers_by_key inconsistency" not in stderr


def test_state_file_is_loaded_when_provided(tmp_path: Path) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"plan": {}, "log": [], "iteration": 0}),
        encoding="utf-8",
    )
    code, _, _ = _invoke(
        _basic_argv(tmp_path, "--state-file", str(state_path))
    )
    assert code == 0


# -------------------------------------------------------------------- #
# 26: Determinism
# -------------------------------------------------------------------- #


def test_repeat_runs_produce_identical_html(tmp_path: Path) -> None:
    out_a = tmp_path / "a.html"
    out_b = tmp_path / "b.html"
    code, _, _ = _invoke(
        [
            "--graph",
            GRAPH_FACTORY,
            "--output-html",
            str(out_a),
            "--no-json",
        ]
    )
    assert code == 0
    code, _, _ = _invoke(
        [
            "--graph",
            GRAPH_FACTORY,
            "--output-html",
            str(out_b),
            "--no-json",
        ]
    )
    assert code == 0
    assert out_a.read_bytes() == out_b.read_bytes()


# -------------------------------------------------------------------- #
# Subprocess form — verifies the ``python -m`` entrypoint
# -------------------------------------------------------------------- #


def test_python_m_entrypoint(tmp_path: Path) -> None:
    """One subprocess test pins the ``python -m ccs.cli.diagnose`` entrypoint."""
    out_path = tmp_path / "r.html"
    json_path = tmp_path / "r.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ccs.cli.diagnose",
            "--graph",
            GRAPH_FACTORY,
            "--output-html",
            str(out_path),
            "--output-json",
            str(json_path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    assert json_path.exists()


def test_python_m_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "ccs.cli.diagnose", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0
    assert "ccs-diagnose" in result.stdout


# -------------------------------------------------------------------- #
# Cluster 1 — Security: URL scheme allowlist on book-a-call / contact-email
# -------------------------------------------------------------------- #


def test_book_a_call_url_javascript_scheme_rejected(tmp_path: Path) -> None:
    """``--book-a-call-url 'javascript:alert(1)'`` exits with usage error."""
    code, _, stderr = _invoke(
        _basic_argv(tmp_path, "--book-a-call-url", "javascript:alert(1)")
    )
    # Validation rejects the scheme; usage error code (#25).
    assert code == 2
    assert "book_a_call_url" in stderr or "javascript" in stderr.lower()


def test_contact_email_javascript_scheme_rejected(tmp_path: Path) -> None:
    """``--contact-email 'javascript:alert(1)'`` exits with usage error."""
    code, _, stderr = _invoke(
        _basic_argv(tmp_path, "--contact-email", "javascript:alert(1)")
    )
    assert code == 2
    assert "contact_email" in stderr or "javascript" in stderr.lower()


# -------------------------------------------------------------------- #
# Cluster 1 — Security: --show-payload schema-version allowlist (#3)
# -------------------------------------------------------------------- #


def test_show_payload_accepts_v1_schema(tmp_path: Path) -> None:
    """``ccs.diagnose.v1`` must round-trip through --show-payload."""
    bad = tmp_path / "v1.json"
    bad.write_text(
        json.dumps(
            {
                "schema_version": "ccs.diagnose.v1",
                "verdict": {
                    "bucket": "single_writer",
                    "confidence": "high",
                    "coverage": {
                        "tick_count": 1,
                        "read_count": 1,
                        "write_count": 1,
                        "artifact_count": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    code, stdout, _ = _invoke(["--show-payload", str(bad)])
    assert code == 0
    payload = json.loads(stdout)
    # Top-level schema_version reflects what the file declared.
    assert payload["schema_version"] == "ccs.diagnose.v1"


# -------------------------------------------------------------------- #
# Cluster 2 — Agent readiness: --yes / --non-interactive (#5)
# -------------------------------------------------------------------- #


def test_yes_flag_skips_prompt_with_no_consent_file(tmp_path: Path) -> None:
    """--yes runs the pipeline as denied when no consent.json exists."""
    code, stdout, _ = _invoke(_basic_argv(tmp_path, "--yes", "--dry-run"))
    assert code == 0
    payload = json.loads(stdout.strip())
    assert payload["installation_token"] is None


def test_yes_flag_uses_persisted_consent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When consent.json exists at the current policy version, --yes reuses it."""
    cfg_dir = tmp_path / "xdg"
    cfg_dir.mkdir()
    consent_dir = cfg_dir / "ccs-diagnose"
    consent_dir.mkdir()
    import uuid as _uuid

    token = _uuid.uuid4()
    (consent_dir / "consent.json").write_text(
        json.dumps(
            {
                "granted": True,
                "policy_version": 1,
                "installation_token": str(token),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(cfg_dir))
    code, stdout, _ = _invoke(_basic_argv(tmp_path, "--yes", "--dry-run"))
    assert code == 0
    payload = json.loads(stdout.strip())
    assert payload["installation_token"] == str(token)


def test_yes_flag_alias_non_interactive(tmp_path: Path) -> None:
    """--non-interactive is an alias for --yes."""
    code, stdout, _ = _invoke(
        _basic_argv(tmp_path, "--non-interactive", "--dry-run")
    )
    assert code == 0
    payload = json.loads(stdout.strip())
    assert payload["installation_token"] is None


# -------------------------------------------------------------------- #
# Cluster 2 — Agent readiness: --output-json - sentinel (#6)
# -------------------------------------------------------------------- #


def test_output_json_dash_writes_to_stdout(tmp_path: Path) -> None:
    """``--output-json -`` writes the JSON payload to stdout."""
    code, stdout, _ = _invoke(
        [
            "--graph",
            GRAPH_FACTORY,
            "--output-html",
            str(tmp_path / "r.html"),
            "--output-json",
            "-",
        ]
    )
    assert code == 0
    payload = json.loads(stdout.strip())
    assert payload["schema_version"] == CCS_DIAGNOSE_LOG_SCHEMA_VERSION
    assert "verdict" in payload
    assert "report" in payload


def test_output_json_dash_routes_summary_to_stderr(tmp_path: Path) -> None:
    """In stdout-mode, the human summary is on stderr (not mixed with JSON)."""
    code, stdout, stderr = _invoke(
        [
            "--graph",
            GRAPH_FACTORY,
            "--output-html",
            str(tmp_path / "r.html"),
            "--output-json",
            "-",
        ]
    )
    assert code == 0
    # JSON parse must NOT see the summary text mixed in.
    json.loads(stdout.strip())
    assert "Your write pattern" in stderr
    assert "Your write pattern" not in stdout


def test_output_json_dash_skips_json_file_write(tmp_path: Path) -> None:
    """``--output-json -`` does not create diagnose_report.json on disk."""
    code, _, _ = _invoke(
        [
            "--graph",
            GRAPH_FACTORY,
            "--output-html",
            str(tmp_path / "r.html"),
            "--output-json",
            "-",
        ]
    )
    assert code == 0
    # The default JSON path lives in the cwd; ensure no surprising file
    # snuck out under the literal name ``-``.
    assert not (tmp_path / "-").exists()
    # And no default-named file landed beside the html in tmp_path.
    assert not (tmp_path / "diagnose_report.json").exists()


# -------------------------------------------------------------------- #
# Cluster 3 — Distinct exit codes per failure category (#25)
# -------------------------------------------------------------------- #


def test_show_payload_oversize_returns_exit_5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Oversized --show-payload input returns EXIT_IO_ERROR (5)."""
    big = tmp_path / "big.json"
    big.write_text(json.dumps({"schema_version": "ccs.diagnose.v0-preview"}))
    # Force-stub the size guard rather than write 10MB of text.
    import ccs.cli.diagnose as cli_mod

    monkeypatch.setattr(cli_mod, "_MAX_INPUT_FILE_SIZE", 1)
    code, _, stderr = _invoke(["--show-payload", str(big)])
    assert code == 5
    assert "exceeds" in stderr.lower()


def test_invalid_json_in_show_payload_returns_exit_1(tmp_path: Path) -> None:
    """A malformed JSON file under --show-payload returns generic error 1."""
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json{", encoding="utf-8")
    code, _, _ = _invoke(["--show-payload", str(bad)])
    assert code == 1


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="no SIGPIPE on this platform")
@pytest.mark.parametrize("handler_name", ["ignored", "default", "custom"])
@pytest.mark.parametrize("path", ["help", "usage", "parse_error", "io_error", "pipeline"])
def test_main_preserves_sigpipe(handler_name: str, path: str, tmp_path: Path) -> None:
    """Assert before fixture teardown, which restores SIGPIPE between tests."""
    def custom_handler(signum, frame):
        pass

    handler = {"ignored": signal.SIG_IGN, "default": signal.SIG_DFL, "custom": custom_handler}[handler_name]
    argv, expected = {
        "help": (["--help"], 0),
        "usage": ([], 2),
        "parse_error": (["--unknown-option"], 2),
        "io_error": (["--show-payload", str(tmp_path / "missing.json")], 5),
        "pipeline": (_basic_argv(tmp_path), 0),
    }[path]
    saved = signal.signal(signal.SIGPIPE, handler)
    try:
        code, _, _ = _invoke(argv)
        assert code == expected
        assert signal.getsignal(signal.SIGPIPE) == handler
    finally:
        signal.signal(signal.SIGPIPE, saved)


def test_main_can_run_in_a_worker_thread() -> None:
    """Embedded help must not attempt a main-thread-only signal operation."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        code, stdout, stderr = executor.submit(_invoke, ["--help"]).result(timeout=15)
    assert code == 0
    assert "ccs-diagnose" in stdout
    assert stderr == ""


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="no SIGPIPE on this platform")
def test_main_leaves_later_broken_pipe_catchable() -> None:
    """A real write after an embedded call must not kill its host process."""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
            import contextlib
            import io
            import os
            from ccs.cli.diagnose import main

            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    main(["--help"])
                except SystemExit as exc:
                    assert exc.code == 0
            read_fd, write_fd = os.pipe()
            os.close(read_fd)
            try:
                os.write(write_fd, b"later write")
            except BrokenPipeError:
                print("host survived")
            finally:
                os.close(write_fd)
            """)],
        capture_output=True, text=True, timeout=15, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "host survived\n"


@pytest.mark.parametrize("has_sigpipe", [False, True])
@pytest.mark.parametrize("outcome", [0, 5, "exception"])
def test_console_main_delegates_without_changing_errors(
    has_sigpipe: bool, outcome: int | str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from ccs.cli import diagnose

    calls = []
    fake_signal = SimpleNamespace(SIG_DFL=0, signal=lambda *args: calls.append(args))
    if has_sigpipe:
        fake_signal.SIGPIPE = 13
    monkeypatch.setattr(diagnose, "signal", fake_signal)
    argv = ["--help"]
    error = RuntimeError("original failure")

    def embedded_main(received, *, _graph_context):
        assert _graph_context is diagnose._graph_sigpipe_guard
        assert received is argv
        assert calls == ([(13, 0)] if has_sigpipe else [])
        if outcome == "exception":
            raise error
        return outcome

    monkeypatch.setattr(diagnose, "main", embedded_main)
    if outcome == "exception":
        with pytest.raises(RuntimeError) as exc:
            diagnose.console_main(argv)
        assert exc.value is error
    else:
        assert diagnose.console_main(argv) == outcome


@pytest.mark.skipif(not hasattr(signal, "SIGPIPE"), reason="no SIGPIPE on this platform")
@pytest.mark.parametrize("unbuffered", [False, True])
@pytest.mark.parametrize("path", ["help", "small_output", "usage_error"])
def test_python_m_closed_stdout_is_quiet(
    unbuffered: bool, path: str,
) -> None:
    # Close the reader before launch so this cannot race against a fast writer.
    # reset-token's small output also covers the buffered shutdown flush.
    argv, expected = {
        "help": (["--help"], -signal.SIGPIPE),
        "small_output": (["--reset-token"], -signal.SIGPIPE),
        "usage_error": ([], 2),
    }[path]
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    try:
        proc = subprocess.run(
            [sys.executable, *(["-u"] if unbuffered else []), "-m", "ccs.cli.diagnose", *argv],
            env={**os.environ, "PYTHONUNBUFFERED": "1" if unbuffered else ""},
            stdout=write_fd, stderr=subprocess.PIPE, text=True, timeout=15, check=False,
        )
    finally:
        os.close(write_fd)
    assert proc.returncode == expected, proc.stderr
    if path == "usage_error":
        assert "--graph is required" in proc.stderr
    else:
        assert proc.stderr == ""
