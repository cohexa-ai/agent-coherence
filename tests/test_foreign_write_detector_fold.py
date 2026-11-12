# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The git foreign-write check on a disk that folds names.

Git drops a literal pathspec whose case differs from the index name before
printing anything, so a row stored as ``PLAN.MD`` while the index holds
``plan.md`` is never reported by a poll that asks for ``PLAN.MD``. On a
folding profile the pass therefore reads git's index first, maps each covered
row to git's own spelling by fold key, and polls with that. On ``none`` the
pass is today's.

Tests that need a real folding disk classify the test's own temporary disk with
the coordinator's start-time probe and skip when it does not fold that way.
The rest build the registry's fold index directly and hold on any disk.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import unicodedata
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ccs.adapters.claude_code import foreign_write_detector as detector
from ccs.adapters.claude_code.foreign_write_detector import (
    _fold_key_of,
    _repository_is_absent,
    run_detection_pass,
)
from ccs.adapters.claude_code.fs_profile import FoldProbeInput, probe_fold_profile
from ccs.adapters.claude_code.policy import TrackedArtifactPolicy
from ccs.coordinator.fold_merge import index_key_of
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.path_fold import FOLD_LENGTH_LIMIT, FoldProfile
from ccs.core.substrate import sha256_hex
from ccs.diagnose.foreign_writes import NOT_COVERABLE, read_foreign_write_report

DETECTOR_LOGGER = "ccs.adapters.claude_code.foreign_write_detector"
NFD_NAME = "café.md"


class _Coordinator:
    """The attributes the pass reads off the real coordinator server."""

    def __init__(self, root: Path, registry: SqliteArtifactRegistry, policy: TrackedArtifactPolicy) -> None:
        self.coordinator_root = root
        self.registry = registry
        self.policy = policy


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    return result.stdout.decode("utf-8")


def _tick(coordinator: _Coordinator, now_unix: float, faults: set[str] | None = None) -> int:
    return run_detection_pass(
        coordinator,
        now_unix=now_unix,
        window_sec=10.0,
        poll_budget_sec=30.0,
        stat_cache={},
        reported_faults=faults if faults is not None else set(),
        tick_clock={},
        max_gap_sec=1_000_000.0,
    )


def _committed_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """A git repository holding ``files``, written under the names given and
    committed, so the index carries whatever spelling git makes of them."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@example.com")
    _git(root, "config", "user.name", "t")
    for name, content in files.items():
        (root / name).write_text(content)
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "seed")
    return root


def _disk_profile(root: Path) -> FoldProfile:
    """The profile the coordinator's start-time probe gives ``root``'s disk."""
    coherence_dir = root / ".coherence"
    coherence_dir.mkdir(exist_ok=True)
    result = probe_fold_profile(
        FoldProbeInput(root=root, coherence_dir=coherence_dir, lock_state="held", log=lambda _line: None)
    )
    return result.profile


def _require_disk(root: Path, *profiles: FoldProfile) -> FoldProfile:
    profile = _disk_profile(root)
    if profile not in profiles:
        pytest.skip(f"needs a disk that folds as {profiles}, this one folds as {profile}")
    return profile


@pytest.fixture
def coordinator_for() -> Iterator[Callable[[Path, tuple[str, ...]], _Coordinator]]:
    """Opens a coordinator over a root with a policy tracking exactly the
    patterns given; every registry it opens is closed at teardown."""
    opened: list[SqliteArtifactRegistry] = []

    def make(root: Path, patterns: tuple[str, ...]) -> _Coordinator:
        registry = SqliteArtifactRegistry(root / ".coherence" / "state.db")
        opened.append(registry)
        policy = TrackedArtifactPolicy(coordinator_root=root, tracked_patterns=patterns)
        return _Coordinator(root, registry, policy)

    yield make
    for registry in opened:
        registry.close()


def _register(coordinator: _Coordinator, name: str, content: str):
    return coordinator.registry.resolve_or_register(name, sha256_hex(content.encode()))


def _foreign_count(coordinator: _Coordinator, artifact_id) -> int:
    return coordinator.registry.foreign_write_totals().get(artifact_id, {}).get("foreign", 0)


def _covered_counts(coordinator: _Coordinator) -> list[int]:
    return [run.covered_count for run in coordinator.registry.detection_runs()]


@pytest.fixture
def git_calls(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Every argv the pass hands ``subprocess.run``, in order, still run for real."""
    calls: list[list[str]] = []
    real_run = subprocess.run

    def recording(argv, **kwargs):
        calls.append([str(arg) for arg in argv])
        return real_run(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording)
    return calls


def _subcommand(argv: list[str]) -> str:
    return next(arg for arg in argv if arg in ("status", "ls-files"))


def _pathspecs(argv: list[str]) -> list[str]:
    return argv[argv.index("--") + 1:] if "--" in argv else []


# ---------------------------------------------------------------------------
# A row stored under a spelling git does not use
# ---------------------------------------------------------------------------


def test_a_row_stored_in_another_case_than_the_index_has_its_outside_edit_counted(
    tmp_path: Path, coordinator_for
) -> None:
    """A merge survivor keeps the disk's spelling, and a case-only rename on a
    caseless disk leaves git's index on the old one. Polling the stored name
    asks git about a path it drops unprinted; polling git's name sees the edit."""
    root = _committed_repo(tmp_path, {"plan.md": "v1\n"})
    profile = _require_disk(root, "caseless")
    coordinator = coordinator_for(root, ("PLAN.MD",))
    coordinator.registry.build_fold_index(profile)
    artifact = _register(coordinator, "PLAN.MD", "v1\n")
    assert coordinator.registry.artifact_names_under_prefix("") == ["PLAN.MD"], "premise: stored as typed"

    (root / "plan.md").write_text("an outside edit\n")
    _tick(coordinator, now_unix=100.0)

    assert _foreign_count(coordinator, artifact) == 1
    assert _covered_counts(coordinator) == [1]


def test_a_row_stored_in_nfd_whose_index_name_is_nfc_has_its_outside_edit_counted(
    tmp_path: Path, coordinator_for
) -> None:
    """A file created under a decomposed name keeps that spelling on disk,
    while git precomposes it in the index. The row is stored as the disk
    spells it, so only a fold maps it to git's name."""
    root = _committed_repo(tmp_path, {NFD_NAME: "v1\n"})
    profile = _require_disk(root, "caseless", "nfd")
    nfc_name = unicodedata.normalize("NFC", NFD_NAME)
    assert _git(root, "ls-files", "-z").split("\0")[0] == nfc_name, "premise: git indexed it precomposed"
    coordinator = coordinator_for(root, ("*.md",))
    coordinator.registry.build_fold_index(profile)
    artifact = _register(coordinator, NFD_NAME, "v1\n")

    (root / NFD_NAME).write_text("an outside edit\n")
    _tick(coordinator, now_unix=100.0)

    assert _foreign_count(coordinator, artifact) == 1
    assert _covered_counts(coordinator) == [1]


def test_a_row_tracked_by_a_pattern_in_git_spelling_is_covered_in_any_case(
    tmp_path: Path, coordinator_for
) -> None:
    """The policy meets the stored name the way it meets a request on this
    disk, so ``plan.md`` tracks a row the disk stores as ``PLAN.MD``."""
    root = _committed_repo(tmp_path, {"plan.md": "v1\n"})
    profile = _require_disk(root, "caseless")
    coordinator = coordinator_for(root, ("plan.md",))
    coordinator.registry.build_fold_index(profile)
    artifact = _register(coordinator, "PLAN.MD", "v1\n")

    (root / "plan.md").write_text("an outside edit\n")
    _tick(coordinator, now_unix=100.0)

    assert _foreign_count(coordinator, artifact) == 1


def test_a_row_maps_to_a_skip_worktree_index_name_too(tmp_path: Path, coordinator_for) -> None:
    """A change staged before ``--skip-worktree`` was set is still reported by
    the poll, so the map must reach every index name, whatever its tag: mapped
    to the visible names alone, this row would be polled by its own spelling,
    which git drops."""
    root = _committed_repo(tmp_path, {"draft.md": "v1\n"})
    profile = _require_disk(root, "caseless")
    (root / "draft.md").write_text("staged then hidden\n")
    _git(root, "add", "draft.md")
    _git(root, "update-index", "--skip-worktree", "draft.md")
    coordinator = coordinator_for(root, ("DRAFT.MD",))
    coordinator.registry.build_fold_index(profile)
    artifact = _register(coordinator, "DRAFT.MD", "v1\n")

    _tick(coordinator, now_unix=100.0)

    assert _foreign_count(coordinator, artifact) == 1


# ---------------------------------------------------------------------------
# What a folding pass asks git, and what it claims
# ---------------------------------------------------------------------------


def test_a_folding_pass_reads_the_index_first_and_polls_with_git_spelling(
    tmp_path: Path, coordinator_for, git_calls: list[list[str]]
) -> None:
    root = _committed_repo(tmp_path, {"notes.md": "v1\n"})
    coordinator = coordinator_for(root, ("NOTES.MD",))
    coordinator.registry.build_fold_index("caseless")
    _register(coordinator, "NOTES.MD", "v1\n")
    git_calls.clear()

    _tick(coordinator, now_unix=100.0)

    assert [_subcommand(argv) for argv in git_calls] == ["ls-files", "status"]
    assert _pathspecs(git_calls[1]) == ["notes.md"]


@pytest.mark.parametrize("index_profile", [None, "none"])
def test_a_pass_that_folds_nothing_polls_the_stored_names_first_as_today(
    tmp_path: Path, coordinator_for, git_calls: list[list[str]], index_profile
) -> None:
    """Exact names (no index) and ``none`` keep today's calls: the poll first,
    with the stored names, then the index read."""
    root = _committed_repo(tmp_path, {"notes.md": "v1\n"})
    coordinator = coordinator_for(root, ("NOTES.MD",))
    if index_profile is not None:
        coordinator.registry.build_fold_index(index_profile)
    _register(coordinator, "NOTES.MD", "v1\n")
    git_calls.clear()

    _tick(coordinator, now_unix=100.0)

    assert [_subcommand(argv) for argv in git_calls] == ["status", "ls-files"]
    assert _pathspecs(git_calls[0]) == ["NOTES.MD"]


def test_a_covered_row_that_maps_to_no_git_name_is_not_counted_visible(
    tmp_path: Path, coordinator_for
) -> None:
    root = _committed_repo(tmp_path, {"notes.md": "v1\n"})
    (root / "draft.md").write_text("never added to git\n")
    coordinator = coordinator_for(root, ("notes.md", "draft.md"))
    coordinator.registry.build_fold_index("caseless")
    _register(coordinator, "notes.md", "v1\n")
    _register(coordinator, "draft.md", "v1\n")

    _tick(coordinator, now_unix=100.0)

    assert _covered_counts(coordinator) == [1]


def test_a_staged_deletion_the_poll_reports_stays_watched_on_a_folding_pass(
    tmp_path: Path, coordinator_for
) -> None:
    """A path removed with ``git rm --cached`` is gone from the index yet the
    poll still reports it under its own name, as it does today; a row with no
    index name is polled by its own spelling so that report still reaches it."""
    root = _committed_repo(tmp_path, {"notes.md": "v1\n"})
    coordinator = coordinator_for(root, ("notes.md",))
    coordinator.registry.build_fold_index("caseless")
    artifact = _register(coordinator, "notes.md", "v1\n")
    _git(root, "rm", "--cached", "-q", "notes.md")
    (root / "notes.md").write_text("an outside edit\n")

    _tick(coordinator, now_unix=100.0)

    assert _foreign_count(coordinator, artifact) == 1
    assert _covered_counts(coordinator) == [1]


def test_a_folding_pass_in_a_workspace_with_no_repository_is_noted_without_a_traceback(
    tmp_path: Path, coordinator_for, caplog: pytest.LogCaptureFixture
) -> None:
    """The index read now runs first, so it meets a missing repository first,
    and must say so the way the poll does: one note, no traceback per tick."""
    root = tmp_path / "bare"
    root.mkdir()
    (root / "notes.md").write_text("v1\n")
    assert _repository_is_absent(root), "fixture is inside a repository"
    coordinator = coordinator_for(root, ("notes.md",))
    coordinator.registry.build_fold_index("caseless")
    _register(coordinator, "notes.md", "v1\n")
    faults: set[str] = set()

    with caplog.at_level(logging.DEBUG, logger=DETECTOR_LOGGER):
        for tick in range(2):
            _tick(coordinator, now_unix=100.0 + tick, faults=faults)

    assert [note.reason for note in coordinator.registry.detection_uncoverable()] == ["no-git-work-tree"]
    assert faults == {"not-a-git-repository"}
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    report = read_foreign_write_report(root / ".coherence" / "state.db")
    assert report.state == NOT_COVERABLE


def test_a_corrupt_repository_met_by_the_index_read_keeps_its_traceback(
    tmp_path: Path, coordinator_for, caplog: pytest.LogCaptureFixture
) -> None:
    """Git says "not a git repository" of a gutted ``.git`` too; only a walk
    that finds no ``.git`` earns the quiet note."""
    root = _committed_repo(tmp_path, {"notes.md": "v1\n"})
    coordinator = coordinator_for(root, ("notes.md",))
    coordinator.registry.build_fold_index("caseless")
    _register(coordinator, "notes.md", "v1\n")
    shutil.rmtree(root / ".git" / "objects")
    shutil.rmtree(root / ".git" / "refs")

    with caplog.at_level(logging.DEBUG, logger=DETECTOR_LOGGER):
        _tick(coordinator, now_unix=100.0)

    assert coordinator.registry.detection_uncoverable() == []
    assert [record.levelno for record in caplog.records if record.exc_info] == [logging.ERROR]


# ---------------------------------------------------------------------------
# The key a git name is mapped by
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("profile", ["caseless", "nfd"])
def test_the_git_name_key_is_the_registry_key(profile) -> None:
    """Every index name is keyed each tick, so ASCII names take a short cut;
    it must give exactly the key the registry stores rows under."""
    ascii_names = [f"Dir{chr(c)}x/Name{chr(c)}Y.MD" for c in range(32, 127) if chr(c) != "/"]
    names = [
        *ascii_names,
        "./Docs//Plan.MD",
        "A" * (FOLD_LENGTH_LIMIT + 1),
        "Straße/PLAN.md",
        "Kelvin.md",
        NFD_NAME,
    ]
    assert {name: _fold_key_of(name, profile) for name in names} == {
        name: index_key_of(name, profile) for name in names
    }


def test_an_ascii_index_name_is_keyed_without_the_table_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    """The table fold costs about 50 microseconds a name, so keying every index
    name of a large repository with it would spend the tick's whole budget."""
    folded: list[str] = []

    def counting(name: str, profile: FoldProfile) -> str:
        folded.append(name)
        return index_key_of(name, profile)

    monkeypatch.setattr(detector, "index_key_of", counting)
    _fold_key_of("src/Package/Module.PY", "caseless")
    _fold_key_of("docs/Plan.md", "nfd")
    assert folded == []
    _fold_key_of(NFD_NAME, "caseless")
    assert folded == [NFD_NAME]
