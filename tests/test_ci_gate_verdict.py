# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The verdicts behind the required checks that gate split work.

``Tests (Python 3.x)`` and ``TLA+ Model Check`` are gate jobs over the jobs
that do the work, and they run under ``always()``. A gate that turned green on
anything but the contract's pass rows would let a PR merge with shards that
failed, never ran, or were cancelled — and GitHub reports a cancelled
``always()`` gate as failed only if the gate itself decides so. Each table
below is the whole contract: every combination is listed, and only the pass
rows pass.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "ci_gate_verdict.py"

JOB_RESULTS = ("success", "failure", "cancelled", "skipped")
COVERAGE_EXITS = {"pass": "0", "fail": "1", "cannot tell": "2"}
EVENTS = ("pull_request", "push", "workflow_dispatch", "schedule")


def _load():
    spec = importlib.util.spec_from_file_location("ci_gate_verdict", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def verdict():
    return _load()


@pytest.mark.parametrize("shards", JOB_RESULTS)
@pytest.mark.parametrize("coverage", sorted(COVERAGE_EXITS))
def test_the_tests_gate_passes_only_succeeded_shards_with_proven_coverage(verdict, shards, coverage) -> None:
    """Twelve combinations, one pass: shards that succeeded AND a coverage
    check that proved every test ran exactly once. A failed, cancelled or
    skipped shard, or coverage that failed or could not tell, keeps the
    required check red."""
    passed, _reason = verdict.tests_verdict(shards, COVERAGE_EXITS[coverage])
    assert passed == (shards == "success" and coverage == "pass")


@pytest.mark.parametrize("coverage_exit", ["", "3", "-1", "137", "nan"])
def test_the_tests_gate_reads_any_other_coverage_exit_as_cannot_tell(verdict, coverage_exit) -> None:
    """A coverage step that crashed or never ran leaves no 0; only an exact 0
    is proof."""
    passed, reason = verdict.tests_verdict("success", coverage_exit)
    assert not passed
    assert "cannot tell" in reason


@pytest.mark.parametrize("bins", JOB_RESULTS)
@pytest.mark.parametrize("event", EVENTS)
@pytest.mark.parametrize("plan", JOB_RESULTS)
@pytest.mark.parametrize("tla_needed", ["true", "false", "", "False", "no"])
def test_the_tla_gate_passes_only_its_two_contract_rows(verdict, bins, event, plan, tla_needed) -> None:
    """Bins that succeeded pass on any event. Bins skipped pass only on a pull
    request whose successful plan said, in exactly ``false``, that no TLA+
    input changed. Bins skipped on any other event, or with an empty,
    differently-spelled or unsuccessful plan output, fail: a skip there means
    the model check silently did not run."""
    passed, _reason = verdict.tla_verdict(bins, event, plan, tla_needed)
    expected = bins == "success" or (
        bins == "skipped" and event == "pull_request" and plan == "success" and tla_needed == "false"
    )
    assert passed == expected


@pytest.mark.parametrize(
    ("argv", "exit_code"),
    [
        (["tests", "--shards", "success", "--coverage-exit", "0"], 0),
        (["tests", "--shards", "success", "--coverage-exit", "2"], 1),
        (["tests", "--shards", "cancelled", "--coverage-exit", "0"], 1),
        (["tla", "--bins", "success", "--event", "push", "--plan", "success", "--tla-needed", ""], 0),
        (["tla", "--bins", "skipped", "--event", "pull_request", "--plan", "success", "--tla-needed", "false"], 0),
        (["tla", "--bins", "skipped", "--event", "push", "--plan", "success", "--tla-needed", "false"], 1),
    ],
)
def test_the_command_line_exits_on_the_verdict(argv, exit_code) -> None:
    """The gate job runs the script on the runner's system Python and relies
    on its exit code alone, and the reason it prints is what a reader of a
    red gate sees."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *argv], capture_output=True, text=True, timeout=30, check=False
    )
    assert proc.returncode == exit_code, proc.stdout + proc.stderr
    assert proc.stdout.startswith("PASS" if exit_code == 0 else "FAIL")
