# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for tools/check_shard_coverage.py — the per-version shard coverage gate.

The suite runs split across several CI shards, each with a parallel (xdist) and
a serial lane. Nothing else notices when the split goes wrong: a test that lands
in no shard silently stops running, and one that lands in two runs twice. The
gate reads the node-id lists every shard wrote and proves the partition, so each
case writes tiny lists into ``tmp_path`` in the layout the conftest writer
produces and calls ``main`` directly. ``tools/`` is not an importable package,
so the module is loaded by file path.

Every case asserts the exit code AND the printed reason: a gate that goes red
for the wrong reason sends whoever reads the log after the wrong bug. "Cannot
tell" (exit 2) is pinned apart from "fail" (exit 1) and never comes back as a
pass (exit 0).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

LANES = ("parallel", "serial")
PARALLEL = ["tests/a.py::test_one", "tests/a.py::test_two", "tests/b.py::TestB::test_three"]
SERIAL = ["tests/s.py::test_serial_one", "tests/s.py::test_serial_two"]
FULL = {"parallel": PARALLEL, "serial": SERIAL}


def _import_check_shard_coverage():
    spec = importlib.util.spec_from_file_location(
        "check_shard_coverage", _REPO_ROOT / "tools" / "check_shard_coverage.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _import_check_shard_coverage()


def _write_ids(path: Path, node_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{node_id}\n" for node_id in node_ids), encoding="utf-8")


def _build(
    root: Path, splits: dict[str, dict[str, list[str]]], full: dict[str, list[str]] | None = None
) -> None:
    """Write each shard's full list (the whole lane, same in every shard), its split list, and a ran list
    equal to its split (the shard ran what it selected)."""
    full = full or FULL
    for shard, split_by_lane in splits.items():
        for lane in LANES:
            _write_ids(root / shard / f"{lane}-full.txt", full[lane])
            _write_ids(root / shard / f"{lane}-split.txt", split_by_lane[lane])
            _write_ids(root / shard / f"{lane}-ran.txt", split_by_lane[lane])


def _partition() -> dict[str, dict[str, list[str]]]:
    """Two shards that between them select each lane's full list exactly once."""
    return {
        "shard-1": {"parallel": PARALLEL[:2], "serial": SERIAL[:1]},
        "shard-2": {"parallel": PARALLEL[2:], "serial": SERIAL[1:]},
    }


def _run(tool, root: Path, shards: int = 2) -> int:
    return tool.main(["--root", str(root), "--shards", str(shards)])


def _listed_under(out: str, heading: str) -> list[str]:
    """The entries printed (4-space indent) under the first category heading starting with ``heading``."""
    lines = out.splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith(f"  {heading}")]
    assert starts, f"no {heading!r} section in:\n{out}"
    entries: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if not line.startswith("    "):
            break
        entries.append(line[4:])
    return entries


# --- pass ---------------------------------------------------------------------------------------


def test_main_with_exact_partition_passes_and_prints_lane_counts(tool, tmp_path, capsys):
    """A correct split must pass, or the gate blocks every healthy run; the count line feeds the job summary."""
    _build(tmp_path, _partition())

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "parallel: 3 tests, 2 shards, each run once" in out
    assert "serial: 2 tests, 2 shards, each run once" in out


def test_main_ignores_files_next_to_shard_directories(tool, tmp_path, capsys):
    """A stray file in the root (an artifact manifest, a log) must not count as a shard and flip a pass to cannot-tell."""
    _build(tmp_path, _partition())
    (tmp_path / "manifest.txt").write_text("not a shard\n", encoding="utf-8")

    exit_code = _run(tool, tmp_path)

    assert exit_code == 0
    assert "parallel: 3 tests, 2 shards, each run once" in capsys.readouterr().out


def test_main_with_empty_split_in_one_shard_passes(tool, tmp_path, capsys):
    """A shard whose split selected nothing for a lane is legitimate when the other shard holds the whole lane."""
    splits = _partition()
    splits["shard-1"]["parallel"] = []
    splits["shard-2"]["parallel"] = list(PARALLEL)
    _build(tmp_path, splits)

    exit_code = _run(tool, tmp_path)

    assert exit_code == 0
    assert "parallel: 3 tests, 2 shards, each run once" in capsys.readouterr().out


def test_main_keeps_parametrized_id_with_spaces_and_brackets_whole(tool, tmp_path, capsys):
    """Splitting a node id on spaces would miscount the lane and report fragments no one can rerun."""
    odd_id = "tests/x.py::test_a[cat plan.md && echo x]"
    full = {"parallel": [*PARALLEL, odd_id], "serial": SERIAL}
    splits = _partition()
    splits["shard-1"]["parallel"] = [*PARALLEL[:2], odd_id]
    _build(tmp_path / "covered", splits, full)
    _build(tmp_path / "missed", _partition(), full)

    covered_exit = _run(tool, tmp_path / "covered")
    covered_out = capsys.readouterr().out
    missed_exit = _run(tool, tmp_path / "missed")
    missed_out = capsys.readouterr().out

    assert covered_exit == 0
    assert "parallel: 4 tests, 2 shards, each run once" in covered_out
    assert missed_exit == 1
    assert _listed_under(missed_out, "missed") == [odd_id]


# --- fail ---------------------------------------------------------------------------------------


def test_main_with_node_id_in_two_shards_fails_as_duplicate(tool, tmp_path, capsys):
    """A test selected by two shards runs twice, so per-shard counts no longer prove anything about coverage."""
    splits = _partition()
    splits["shard-2"]["parallel"] = PARALLEL[1:]
    _build(tmp_path, splits)

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "parallel: FAIL" in out
    assert _listed_under(out, "duplicate") == [f"{PARALLEL[1]} (shard-1, shard-2)"]


def test_main_with_node_id_twice_in_one_split_fails_as_duplicate(tool, tmp_path, capsys):
    """A split list naming a test twice is a broken split, even though the union still looks complete."""
    splits = _partition()
    splits["shard-1"]["parallel"] = [*PARALLEL[:2], PARALLEL[0]]
    _build(tmp_path, splits)

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert _listed_under(out, "duplicate") == [f"{PARALLEL[0]} (shard-1, shard-1)"]


def test_main_with_collected_id_in_no_split_fails_as_missed(tool, tmp_path, capsys):
    """The failure the gate exists for: a collected test no shard ran, which no test result would ever show."""
    splits = _partition()
    splits["shard-2"]["parallel"] = []
    _build(tmp_path, splits)

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert _listed_under(out, "missed") == [PARALLEL[2]]


def test_main_with_split_id_absent_from_full_list_fails_as_unexpected(tool, tmp_path, capsys):
    """A split selecting an id the lane never collected means the lists describe different runs."""
    ghost = "tests/ghost.py::test_gone"
    splits = _partition()
    splits["shard-2"]["serial"] = [*SERIAL[1:], ghost]
    _build(tmp_path, splits)

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "serial: FAIL" in out
    assert _listed_under(out, "unexpected") == [ghost]


def test_main_with_full_lists_differing_between_shards_fails_as_non_deterministic(tool, tmp_path, capsys):
    """Shards that collected different suites split different universes, so their splits cannot be compared."""
    _build(tmp_path, _partition())
    _write_ids(tmp_path / "shard-2" / "parallel-full.txt", PARALLEL[:2])

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    heading = next(line for line in out.splitlines() if "non-deterministic collection" in line)
    assert exit_code == 1
    assert "shard-1" in heading and "shard-2" in heading
    assert _listed_under(out, "non-deterministic collection") == [PARALLEL[2]]


def test_main_caps_named_ids_at_twenty_per_category(tool, tmp_path, capsys):
    """An uncapped list of thousands of missed ids buries the reason the gate failed."""
    many = [f"tests/many.py::test_{index:02d}" for index in range(25)]
    splits = {name: {"parallel": [], "serial": lanes["serial"]} for name, lanes in _partition().items()}
    _build(tmp_path, splits, {"parallel": many, "serial": SERIAL})

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert _listed_under(out, "missed") == [*many[:20], "... and 5 more"]


# --- cannot tell --------------------------------------------------------------------------------


def _remove_serial_split(root: Path) -> None:
    (root / "shard-2" / "serial-split.txt").unlink()


def _corrupt_parallel_full(root: Path) -> None:
    (root / "shard-1" / "parallel-full.txt").write_bytes(b"tests/a.py::test_one\n\xff\xfe\n")


def _directory_for_parallel_split(root: Path) -> None:
    target = root / "shard-1" / "parallel-split.txt"
    target.unlink()
    target.mkdir()


def _empty_parallel_full(root: Path) -> None:
    for shard in ("shard-1", "shard-2"):
        _write_ids(root / shard / "parallel-full.txt", [])


def _remove_parallel_ran(root: Path) -> None:
    (root / "shard-1" / "parallel-ran.txt").unlink()


def _leave_unchanged(root: Path) -> None:
    del root


def _add_third_shard(root: Path) -> None:
    _build(root, {"shard-3": {"parallel": [], "serial": []}})


@pytest.mark.parametrize(
    ("mutate", "shards", "reasons"),
    [
        pytest.param(
            _remove_serial_split,
            2,
            ("serial: CANNOT TELL", "missing or unreadable: shard-2/serial-split.txt"),
            id="missing-list",
        ),
        pytest.param(
            _corrupt_parallel_full,
            2,
            ("parallel: CANNOT TELL", "missing or unreadable: shard-1/parallel-full.txt"),
            id="invalid-utf8",
        ),
        pytest.param(
            _directory_for_parallel_split,
            2,
            ("parallel: CANNOT TELL", "missing or unreadable: shard-1/parallel-split.txt"),
            id="list-is-dir",
        ),
        pytest.param(
            _empty_parallel_full,
            2,
            ("parallel: CANNOT TELL", "full list is empty in every shard"),
            id="empty-full-list",
        ),
        pytest.param(
            _remove_parallel_ran,
            2,
            ("parallel: CANNOT TELL", "missing or unreadable: shard-1/parallel-ran.txt"),
            id="missing-ran-list",
        ),
        pytest.param(_leave_unchanged, 3, ("CANNOT TELL", "found 2 shard directories"), id="fewer-shards"),
        pytest.param(_add_third_shard, 2, ("CANNOT TELL", "found 3 shard directories"), id="more-shards"),
    ],
)
def test_main_when_coverage_cannot_be_proved_exits_cannot_tell(tool, tmp_path, capsys, mutate, shards, reasons):
    """Evidence that is missing, corrupt, empty or from the wrong shard set proves nothing; calling it a pass
    waves through a run whose split was never checked, and each case here would pass if the check were skipped."""
    _build(tmp_path, _partition())
    mutate(tmp_path)

    exit_code = _run(tool, tmp_path, shards)

    out = capsys.readouterr().out
    assert exit_code == 2
    assert all(reason in out for reason in reasons), out


def test_main_with_one_lane_failing_and_one_cannot_tell_reports_both_and_exits_fail(tool, tmp_path, capsys):
    """A cannot-tell lane must not hide a definite failure in the other lane, nor be dropped from the report."""
    splits = _partition()
    splits["shard-2"]["parallel"] = PARALLEL[1:]
    _build(tmp_path, splits)
    _remove_serial_split(tmp_path)

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "parallel: FAIL" in out
    assert "serial: CANNOT TELL" in out
    assert "missing or unreadable: shard-2/serial-split.txt" in out


def test_main_with_no_shard_directory_at_all_exits_cannot_tell(tool, tmp_path, capsys):
    """When no shard uploaded its lists, the gate's root directory never exists; that is missing evidence, so
    it must read as cannot tell, not crash with a traceback the gate would misreport."""
    exit_code = _run(tool, tmp_path / "never-downloaded")

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "found 0 shard directories" in out


def test_main_with_a_selected_test_that_never_ran_fails_as_not_run(tool, tmp_path, capsys):
    """xdist can exit 0 with "no tests ran" after a worker dies during collection, leaving a full split list; a
    gate that read only the selection lists would pass a shard that ran nothing."""
    _build(tmp_path, _partition())
    _write_ids(tmp_path / "shard-2" / "parallel-ran.txt", [])

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert _listed_under(out, "not run") == [f"{PARALLEL[2]} (shard-2)"]


def test_main_with_a_test_run_but_not_selected_fails_as_ran_unselected(tool, tmp_path, capsys):
    """A result for a test the shard never selected means the lists describe a different run than the one that
    happened, so none of them can be trusted as evidence."""
    _build(tmp_path, _partition())
    _write_ids(tmp_path / "shard-1" / "serial-ran.txt", [SERIAL[0], SERIAL[1]])

    exit_code = _run(tool, tmp_path)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert _listed_under(out, "ran unselected") == [f"{SERIAL[1]} (shard-1)"]
