# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The suite's two lanes and the node-id lists that prove a split run.

CI runs each Python version as several shards, and each shard as two lanes:
a parallel lane on four workers, then a serial lane with ``-n 0`` for tests
marked ``serial`` (multi-process tests that are not safe beside other
workers). A gate then proves from the node-id lists every shard writes that
the shards together ran every collected test exactly once.

Each guard below protects one way that proof could pass while checking less:
a lane that drops or doubles tests, a lane that lets an excluded marker back
in, a writer that sees a different list than the run, or a serial module that
silently joins the parallel lane. The checks run nested ``pytest`` over a few
named files, not the whole suite.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Frozen on purpose (tests/CLAUDE.md): derived from the code under test, a
# module that gained or lost its serial mark would move this expectation too.
EXPECTED_SERIAL_MODULES = frozenset(
    {
        "tests/conformance/test_cross_process_coordinator.py",
        "tests/conformance/test_durability_regime.py",
        "tests/conformance/test_process_harness.py",
        "tests/test_claude_code_hook_client.py",
        "tests/test_claude_code_lifecycle.py",
        "tests/test_run_local_tests.py",
    }
)
EXPECTED_SERIAL_MODULE_COUNT = 6

SERIAL_ONLY_FILE = "tests/conformance/test_process_harness.py"
MIXED_FILE = "tests/test_claude_code_hook_client.py"  # one serial test among parallel ones
PARALLEL_ONLY_FILE = "tests/test_bash_path_detector.py"
MIXED_FILES = (SERIAL_ONLY_FILE, MIXED_FILE, PARALLEL_ONLY_FILE)
DURATIONS = "--durations-path=tests/.test_durations"
# Each carries a marker the default `addopts` deselects.
EXCLUDED_MARKER_FILES = (
    "tests/conformance/test_tier_honesty.py",
    "tests/protocol_corpus/test_strict_mode_corpus.py",
    "tests/integration/test_warn_mode_behavior_change.py",
    "tests/integration/test_strict_mode_launch_gate.py",
)
EXCLUDED_MARKERS = (
    "real_substrate",
    "live_api",
    "protocol_corpus",
    "launch_gate",
    "launch_gate_pilot",
    "launch_gate_strict",
    "launch_gate_pilot_matrix",
)
SERIAL_MARK = re.compile(r"\bpytest\.mark\.serial\b")


def _pytest(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *args]
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=300, check=False)


def _collected(*args: str) -> list[str]:
    """Node ids a nested ``--collect-only`` run prints, in collection order."""
    proc = _pytest("--collect-only", *args)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return [line for line in proc.stdout.splitlines() if "::" in line]


def _written(directory: Path, name: str) -> list[str]:
    return (directory / f"{name}.txt").read_text().splitlines()


def test_the_two_lanes_partition_the_plain_collection() -> None:
    """The parallel lane holds every non-serial test and the serial lane only
    serial ones, and together they are exactly today's collection: a lane
    that dropped a test would let the coverage proof pass over a smaller set."""
    plain = _collected(*MIXED_FILES)
    parallel = _collected("--test-lane", "parallel", *MIXED_FILES)
    serial = _collected("--test-lane", "serial", *MIXED_FILES)

    assert serial, "control: the named files contain serial tests"
    assert parallel, "control: the named files contain parallel tests"
    assert not set(parallel) & set(serial)
    assert sorted(parallel + serial) == sorted(plain)
    serial_only = _collected("-m", "serial", *MIXED_FILES)
    assert sorted(serial) == sorted(serial_only)


@pytest.mark.parametrize("lane", ["parallel", "serial"])
def test_two_groups_of_a_lane_cover_it_once(lane: str, tmp_path: Path) -> None:
    """pytest-split's two groups of a lane add up to the lane with no test in
    both, and each group writes the lane's full list and its own split."""
    lists = {group: tmp_path / f"group-{group}" for group in (1, 2)}
    for group, directory in lists.items():
        _collected(
            "--test-lane", lane, "--splits", "2", "--group", str(group), DURATIONS,
            f"--node-id-dir={directory}", *MIXED_FILES,
        )
    full = _collected("--test-lane", lane, *MIXED_FILES)
    splits = [_written(lists[group], f"{lane}-split") for group in (1, 2)]

    assert _written(lists[1], f"{lane}-full") == _written(lists[2], f"{lane}-full") == full
    assert not set(splits[0]) & set(splits[1])
    assert sorted(splits[0] + splits[1]) == sorted(full)


@pytest.mark.parametrize("lane", ["parallel", "serial"])
def test_a_lane_keeps_the_excluded_markers_out(lane: str, tmp_path: Path) -> None:
    """A lane is a filter on top of ``addopts``, never a replacement for it:
    a lane chosen with ``-m`` would drop the exclusions and run tests that
    need credentials, a live CLI or real backends."""
    files = (*EXCLUDED_MARKER_FILES, SERIAL_ONLY_FILE, MIXED_FILE)
    _collected("--test-lane", lane, f"--node-id-dir={tmp_path}", *files)
    written = set(_written(tmp_path, f"{lane}-full"))
    assert written, f"control: the {lane} lane has tests in these files"
    excluded = set(_collected("-m", " or ".join(EXCLUDED_MARKERS), *files))
    assert excluded, "control: excluded tests exist in these files"
    assert not written & excluded, f"excluded tests reached the {lane} lane: {sorted(written & excluded)[:5]}"


def test_no_lane_leaves_the_corpus_jobs_selection_alone(tmp_path: Path) -> None:
    """The corpus job runs ``pytest -m protocol_corpus`` with no lane; the lane
    machinery must not change what that selects."""
    selected = _collected("-m", "protocol_corpus", f"--node-id-dir={tmp_path}", *EXCLUDED_MARKER_FILES)
    assert selected
    assert _written(tmp_path, "all-full") == selected


def test_xdist_workers_write_the_same_lists_once(tmp_path: Path) -> None:
    """Under xdist every worker collects; only one may write, and what it
    writes must be the list a single process would write. A second writer
    fails the run, because each list is created exclusively."""
    files = ("tests/test_bash_path_detector.py", "tests/test_release_readiness.py")
    single, workers = tmp_path / "n0", tmp_path / "n2"
    _collected("--test-lane", "parallel", f"--node-id-dir={single}", *files)
    proc = _pytest("-n", "2", "--test-lane", "parallel", f"--node-id-dir={workers}", *files)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sorted(p.name for p in workers.iterdir()) == ["parallel-full.txt", "parallel-ran.txt", "parallel-split.txt"]
    assert _written(workers, "parallel-full") == _written(single, "parallel-full")
    assert _written(workers, "parallel-split") == _written(single, "parallel-split")
    assert sorted(_written(workers, "parallel-ran")) == sorted(_written(single, "parallel-split"))


def test_an_empty_split_passes_and_writes_an_empty_list(tmp_path: Path) -> None:
    """A shard whose split selects nothing for a lane exits 0, not pytest's 5,
    and still writes its (empty) split list, so the gate can see it ran."""
    one_serial_test = "tests/test_claude_code_hook_client.py::test_racing_hook_processes_for_a_new_session_bind_one_principal"
    runs = {
        group: _pytest(
            "--collect-only", "--test-lane", "serial", "--splits", "2", "--group", str(group), DURATIONS,
            f"--node-id-dir={tmp_path / ('group-' + str(group))}", one_serial_test,
        )
        for group in (1, 2)
    }
    assert {group: proc.returncode for group, proc in runs.items()} == {1: 0, 2: 0}
    splits = {group: _written(tmp_path / f"group-{group}", "serial-split") for group in (1, 2)}
    assert sorted(len(split) for split in splits.values()) == [0, 1]


def test_the_serial_modules_are_the_frozen_set() -> None:
    """A module that gains or loses a ``serial`` mark changes which lane runs
    it; that has to be a deliberate edit here, not a side effect."""
    this_file = Path(__file__).resolve()
    marked = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in (REPO_ROOT / "tests").rglob("*.py")
        if path.resolve() != this_file and SERIAL_MARK.search(path.read_text())
    }
    assert marked == EXPECTED_SERIAL_MODULES
    assert len(EXPECTED_SERIAL_MODULES) == EXPECTED_SERIAL_MODULE_COUNT


def test_the_committed_durations_are_a_map_of_node_ids_to_seconds() -> None:
    """CI balances its shards from ``tests/.test_durations``; a file pytest-split
    cannot read makes it fall back to an even count split without failing,
    which quietly unbalances the shards."""
    durations = json.loads((REPO_ROOT / "tests" / ".test_durations").read_text())
    assert isinstance(durations, dict) and durations
    assert all(isinstance(node_id, str) and "::" in node_id for node_id in durations)
    assert all(isinstance(seconds, (int, float)) and seconds >= 0 for seconds in durations.values())


def test_lists_can_go_to_a_directory_that_already_exists(tmp_path: Path) -> None:
    """The serial lane writes into the directory the parallel lane created.
    Given as ``--node-id-dir DIR``, an existing DIR is taken for a test path
    while pytest is still finding its conftest files, so this conftest never
    loads and the option is rejected; the ``--node-id-dir=DIR`` form is safe."""
    _collected("--test-lane", "parallel", f"--node-id-dir={tmp_path}", MIXED_FILE)
    _collected("--test-lane", "serial", f"--node-id-dir={tmp_path}", MIXED_FILE)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "parallel-full.txt", "parallel-split.txt", "serial-full.txt", "serial-split.txt",
    ]


def test_a_worker_lost_in_collection_fails_the_coverage_check(tmp_path: Path) -> None:
    """xdist 3.8 exits 0 with "no tests ran" when a worker dies during
    collection, after ``gw0`` has written a split full of tests. The ran list
    is what lets the coverage check see that nothing ran."""
    plugin_dir = tmp_path / "plugin"
    plugin_dir.mkdir()
    (plugin_dir / "lose_a_worker.py").write_text(textwrap.dedent("""\
        import os

        def pytest_collection_modifyitems(config, items):
            workerinput = getattr(config, "workerinput", None)
            if workerinput is not None and workerinput["workerid"] != "gw0":
                os._exit(1)
    """))
    shard = tmp_path / "ids" / "shard-1"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(plugin_dir), os.environ.get("PYTHONPATH", "")])}
    cmd = [
        sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "lose_a_worker",
        "-n", "2", "--max-worker-restart=0", "--test-lane", "parallel", "--splits", "1", "--group", "1",
        DURATIONS, f"--node-id-dir={shard}", PARALLEL_ONLY_FILE,
    ]
    run = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300, check=False)
    assert "no tests ran" in run.stdout, "control: the worker was lost before anything ran"
    assert _written(shard, "parallel-split"), "control: the split still lists the tests"

    check = subprocess.run(
        [sys.executable, "tools/check_shard_coverage.py", f"--root={tmp_path / 'ids'}", "--shards", "1"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    assert check.returncode != 0
    assert "parallel: FAIL" in check.stdout and "not run:" in check.stdout, check.stdout


def test_the_serial_lane_refuses_xdist_workers() -> None:
    """The serial lane exists to keep racing multi-process tests off shared
    workers; a CI command that passed ``-n 4`` to both lanes would otherwise
    run them side by side and turn their timing into flakes."""
    proc = _pytest("--collect-only", "-n", "2", "--test-lane", "serial", SERIAL_ONLY_FILE)
    assert proc.returncode == pytest.ExitCode.USAGE_ERROR
    assert "--test-lane serial runs with -n 0" in proc.stderr
