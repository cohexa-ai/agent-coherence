# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The test gate's job-summary counts, summed across shards per lane.

They are what a reader compares with a single-process run when the layout
changes, so a lane's totals must add every shard's report and keep the lanes
apart; an unreadable report is said so rather than counted as zero.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("summarize_junit", REPO_ROOT / "tools" / "summarize_junit.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(path: Path, tests: int, failures: int, errors: int, skipped: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<testsuites><testsuite name="pytest" tests="{tests}" failures="{failures}" '
        f'errors="{errors}" skipped="{skipped}"></testsuite></testsuites>'
    )


def test_each_lane_sums_every_shards_report(tmp_path: Path, capsys) -> None:
    """Two shards' parallel reports add up, and the serial lane is counted on its own."""
    _report(tmp_path / "shard-1" / "junit-parallel.xml", 10, 1, 0, 2)
    _report(tmp_path / "shard-2" / "junit-parallel.xml", 7, 0, 1, 3)
    _report(tmp_path / "shard-1" / "junit-serial.xml", 4, 0, 0, 1)

    assert _load().main([str(tmp_path)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines == [
        "parallel: 17 tests, 1 failures, 1 errors, 5 skipped (from 2 JUnit report(s))",
        "serial: 4 tests, 0 failures, 0 errors, 1 skipped (from 1 JUnit report(s))",
    ]


def test_an_unreadable_report_is_named_not_counted_as_zero(tmp_path: Path, capsys) -> None:
    """A truncated upload must not read as a lane that ran nothing."""
    _report(tmp_path / "shard-1" / "junit-serial.xml", 4, 0, 0, 1)
    (tmp_path / "shard-1" / "junit-parallel.xml").write_text("<testsuites><testsuite")

    assert _load().main([str(tmp_path)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "parallel: JUnit reports unreadable (ParseError)"
    assert lines[1].startswith("serial: 4 tests")
