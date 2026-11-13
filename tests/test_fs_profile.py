# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The workspace disk's fold profile, found once per coordinator start.

The classifier is driven with synthetic answer tables, one row per vector, so
a profile that stopped reading any one vector would show here. The probe runs
on real disks: this machine's temporary directory, and on macOS a fresh
case-sensitive APFS disk image. AGENT_COHERENCE_EXPECT_FOLD_PROFILE names the
profile a CI job expects of its runner's disk, so that job fails, never
skips, when the probe answers otherwise. The plugin's Node coordinator runs
the same scenarios against its own probe.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from ccs.adapters.claude_code import lifecycle
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.fs_profile import (
    PROBE_DIR_PREFIX,
    FoldProbeInput,
    FoldProfileResult,
    LockState,
    classify_vector_answers,
    fold_profile_log_line,
    probe_fold_profile,
)

KNOWN_ROWS = (("caseless", "11111111000"), ("nfd", "01010001000"), ("none", "00000000000"))

#: The answers measured on disks no profile models; each must fold nothing.
MEASURED_UNMODELED_ROWS = (
    ("HFS+", "11000001010"),
    ("HFSX", "01000001000"),
    ("exFAT", "11011110000"),
    ("FAT32", "11000000000"),
)

HAS_HDIUTIL = sys.platform == "darwin" and shutil.which("hdiutil") is not None
IS_ROOT = hasattr(os, "geteuid") and os.geteuid() == 0


def expected_tmp_disk_profile() -> str | None:
    """The profile this runner's temporary directory must classify as, or
    None when no one says."""
    expected = os.environ.get("AGENT_COHERENCE_EXPECT_FOLD_PROFILE", "")
    if expected:
        return expected
    if sys.platform == "darwin":
        return "caseless"
    if sys.platform.startswith("linux"):
        return "none"
    return None


TMP_DISK_PROFILE = expected_tmp_disk_profile()
needs_expected_profile = pytest.mark.skipif(
    TMP_DISK_PROFILE is None, reason="no expected fold profile for this platform"
)


def answers_of(pattern: str) -> list[bool]:
    return [answer == "1" for answer in pattern]


def pattern_of(answers: list[bool]) -> str:
    return "".join("1" if answer else "0" for answer in answers)


@dataclass
class Workspace:
    root: Path
    coherence_dir: Path
    logged: list[str] = field(default_factory=list)

    def probe_input(self, lock_state: LockState) -> FoldProbeInput:
        return FoldProbeInput(
            root=self.root, coherence_dir=self.coherence_dir, lock_state=lock_state, log=self.logged.append
        )


def make_workspace_in(base: str) -> Workspace:
    if not base or not os.path.isabs(base) or base == "/":
        raise RuntimeError(f"refusing to use {base!r}")
    root = Path(tempfile.mkdtemp(prefix="fs-profile-", dir=base))
    coherence_dir = root / ".coherence"
    coherence_dir.mkdir(mode=0o700)
    return Workspace(root=root, coherence_dir=coherence_dir)


@pytest.fixture
def workspace() -> Iterator[Workspace]:
    made = make_workspace_in(tempfile.gettempdir())
    try:
        yield made
    finally:
        shutil.rmtree(made.root)


PAST_NS = 981_173_106_000_000_000  # 2001-02-03T04:05:06Z


def backdate(directory: Path) -> None:
    """Backdate ``directory`` so any entry created or removed in it shows as a new mtime."""
    os.utime(directory, ns=(PAST_NS, PAST_NS))


def touched_since_backdate(directory: Path) -> bool:
    return os.lstat(directory).st_mtime_ns != PAST_NS


def snapshot(directory: Path) -> list[str]:
    """Every path under ``directory`` with its kind and, for a file, its
    content; links are listed, never followed."""
    lines = []
    for entry in sorted(os.scandir(directory), key=lambda entry: entry.name):
        info = entry.stat(follow_symlinks=False)
        if stat.S_ISREG(info.st_mode):
            lines.append(f"{entry.name} file {base64.b64encode(Path(entry.path).read_bytes()).decode()}")
        elif stat.S_ISDIR(info.st_mode):
            lines.append(f"{entry.name} dir {info.st_ino}")
            lines.extend(f"{entry.name}/{line}" for line in snapshot(Path(entry.path)))
        else:
            lines.append(f"{entry.name} other {info.st_ino}")
    return lines


def probe_entries(coherence_dir: Path) -> list[str]:
    return [name for name in os.listdir(coherence_dir) if name.startswith(PROBE_DIR_PREFIX)]


def create_file(path: Path) -> None:
    os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))


CLASSIFIED = "classified"
CANNOT_TELL = "cannot_tell"


def cannot_tell(reason: str) -> FoldProfileResult:
    return FoldProfileResult(profile="none", status=CANNOT_TELL, reason=reason)


# --- the classifier ---------------------------------------------------------


@pytest.mark.parametrize(("profile", "pattern"), KNOWN_ROWS)
def test_the_known_answers_classify_the_disk(profile: str, pattern: str) -> None:
    assert classify_vector_answers(answers_of(pattern)) == FoldProfileResult(profile=profile, status=CLASSIFIED)


@pytest.mark.parametrize("vector", range(1, 12))
def test_a_vector_answering_otherwise_than_every_known_profile_is_cannot_tell(vector: int) -> None:
    for _profile, pattern in KNOWN_ROWS:
        answers = answers_of(pattern)
        answers[vector - 1] = not answers[vector - 1]
        assert classify_vector_answers(answers) == cannot_tell(f"answers {pattern_of(answers)} match no known profile")


def test_the_answers_measured_on_hfs_hfsx_exfat_and_fat32_are_cannot_tell() -> None:
    for disk, pattern in MEASURED_UNMODELED_ROWS:
        assert classify_vector_answers(answers_of(pattern)).status == CANNOT_TELL, disk


def test_the_log_line_names_the_profile_or_why_the_disk_could_not_be_classified() -> None:
    assert fold_profile_log_line(FoldProfileResult(profile="caseless", status=CLASSIFIED)) == (
        "workspace disk fold profile: caseless"
    )
    assert fold_profile_log_line(cannot_tell("unlocked")) == (
        "workspace disk fold profile: none (cannot tell: unlocked)"
    )


# --- the probe on real disks --------------------------------------------------


@needs_expected_profile
def test_the_real_workspace_disk_classifies_as_this_platform_expects(workspace: Workspace) -> None:
    assert probe_fold_profile(workspace.probe_input("held")) == FoldProfileResult(
        profile=TMP_DISK_PROFILE, status=CLASSIFIED
    )
    assert os.listdir(workspace.coherence_dir) == []
    assert workspace.logged == []


def hdiutil(*args: str) -> None:
    run = subprocess.run(["hdiutil", *args], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, f"hdiutil {' '.join(args)}: {run.stderr}"


@pytest.fixture
def case_sensitive_apfs() -> Iterator[str]:
    """A fresh case-sensitive APFS volume mounted under a temporary directory."""
    scratch = tempfile.mkdtemp(prefix="fs-profile-image-")
    if not scratch or not os.path.isabs(scratch):
        raise RuntimeError(f"refusing to use {scratch!r}")
    mount_point = os.path.join(scratch, "mnt")
    try:
        os.mkdir(mount_point)
        image = os.path.join(scratch, "cs")
        hdiutil("create", "-quiet", "-type", "SPARSE", "-fs", "Case-sensitive APFS", "-size", "32m", image)
        hdiutil("attach", "-quiet", "-nobrowse", "-mountpoint", mount_point, f"{image}.sparseimage")
        try:
            yield mount_point
        finally:
            hdiutil("detach", "-quiet", "-force", mount_point)
    finally:
        shutil.rmtree(scratch)


def assert_case_link_reads_as_link(base: str, expected: str) -> None:
    """On a disk that keeps the two spellings apart, a committed link must
    answer vector 1 as 0."""
    linked = make_workspace_in(base)
    try:
        os.symlink(".coherence", linked.root / ".COHERENCE")
        assert probe_fold_profile(linked.probe_input("held")) == FoldProfileResult(profile=expected, status=CLASSIFIED)
    finally:
        shutil.rmtree(linked.root)


@pytest.mark.skipif(not HAS_HDIUTIL, reason="needs hdiutil (macOS)")
def test_a_case_sensitive_apfs_disk_image_classifies_as_nfd_and_a_coherence_link_there_reads_as_a_link(
    case_sensitive_apfs: str,
) -> None:
    on_image = make_workspace_in(case_sensitive_apfs)
    try:
        assert probe_fold_profile(on_image.probe_input("held")) == FoldProfileResult(profile="nfd", status=CLASSIFIED)
        assert os.listdir(on_image.coherence_dir) == []
    finally:
        shutil.rmtree(on_image.root)
    assert_case_link_reads_as_link(case_sensitive_apfs, "nfd")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the case-sensitive disk here is the image above")
def test_a_coherence_link_on_a_case_sensitive_temporary_disk_reads_as_a_link() -> None:
    assert_case_link_reads_as_link(tempfile.gettempdir(), "none")


# --- cannot-tell starts ------------------------------------------------------


def test_a_coherence_dir_on_another_device_is_cannot_tell_and_creates_no_probe_file(workspace: Workspace) -> None:
    backdate(workspace.coherence_dir)
    probe = replace(
        workspace.probe_input("held"),
        device_of=lambda path: 2 if path == workspace.coherence_dir else 1,
    )
    assert probe_fold_profile(probe) == cannot_tell("coherence dir on another device")
    assert not touched_since_backdate(workspace.coherence_dir)


def test_an_unlocked_start_is_cannot_tell_runs_no_vector_and_sweeps_nothing(workspace: Workspace) -> None:
    leftover = workspace.coherence_dir / f"{PROBE_DIR_PREFIX}1-0badc0de"
    leftover.mkdir(mode=0o700)
    create_file(leftover / "p\u00e9")
    before = snapshot(workspace.coherence_dir)
    backdate(workspace.coherence_dir)
    assert probe_fold_profile(workspace.probe_input("unlocked")) == cannot_tell("unlocked")
    assert not touched_since_backdate(workspace.coherence_dir)
    assert snapshot(workspace.coherence_dir) == before


@pytest.mark.skipif(IS_ROOT, reason="root creates files in a read-only directory")
def test_a_read_only_coherence_dir_is_cannot_tell_and_leaves_nothing_behind(workspace: Workspace) -> None:
    workspace.coherence_dir.chmod(0o500)
    try:
        result = probe_fold_profile(workspace.probe_input("held"))
    finally:
        workspace.coherence_dir.chmod(0o700)
    assert (result.profile, result.status) == ("none", CANNOT_TELL)
    assert result.reason is not None and result.reason.endswith("could not be created: EACCES")
    assert os.listdir(workspace.coherence_dir) == []


# --- the sweep ---------------------------------------------------------------


@needs_expected_profile
def test_probe_files_a_crashed_start_left_are_swept_and_the_disk_is_then_classified(workspace: Workspace) -> None:
    crashed = workspace.coherence_dir / f"{PROBE_DIR_PREFIX}4242-0badc0de"
    crashed.mkdir(mode=0o700)
    create_file(crashed / "p\u00e9")
    create_file(crashed / "p\u00df")
    (workspace.coherence_dir / f"{PROBE_DIR_PREFIX}4243-feedf00d").mkdir(mode=0o700)
    assert probe_fold_profile(workspace.probe_input("held")) == FoldProfileResult(
        profile=TMP_DISK_PROFILE, status=CLASSIFIED
    )
    assert os.listdir(workspace.coherence_dir) == []
    assert workspace.logged == []


@dataclass(frozen=True)
class Planted:
    outside: Path
    planted_link: Path
    planted: Path


def plant_under_probe_prefix(workspace: Workspace) -> Planted:
    """A link to a directory outside, and a stale directory holding a nested
    directory, a link and a file."""
    outside = workspace.root / "outside"
    outside.mkdir()
    (outside / "kept.md").write_text("kept\n", encoding="utf-8")
    planted_link = workspace.coherence_dir / f"{PROBE_DIR_PREFIX}link"
    os.symlink(outside, planted_link)
    planted = workspace.coherence_dir / f"{PROBE_DIR_PREFIX}4242-0badc0de"
    planted.mkdir(mode=0o700)
    (planted / "nested").mkdir()
    (planted / "nested" / "kept.md").write_text("kept\n", encoding="utf-8")
    os.symlink(outside / "kept.md", planted / "p-link")
    create_file(planted / "p\u00e9")
    return Planted(outside=outside, planted_link=planted_link, planted=planted)


def test_a_link_or_nested_directory_under_the_probe_prefix_is_logged_left_alone_and_makes_the_start_cannot_tell(
    workspace: Workspace,
) -> None:
    planted = plant_under_probe_prefix(workspace)
    outside_before = snapshot(planted.outside)
    backdate(workspace.coherence_dir)

    assert probe_fold_profile(workspace.probe_input("held")) == cannot_tell("stale probe entries left in place")

    assert planted.planted_link.is_symlink()
    assert (planted.planted / "p-link").is_symlink()
    assert (planted.planted / "nested" / "kept.md").read_text(encoding="utf-8") == "kept\n"
    assert not os.path.lexists(planted.planted / "p\u00e9")
    assert snapshot(planted.outside) == outside_before
    assert not touched_since_backdate(workspace.coherence_dir)
    for left in (planted.planted_link, planted.planted / "nested", planted.planted / "p-link"):
        assert any(str(left) in line for line in workspace.logged), f"no log line names {left}: {workspace.logged}"


@pytest.mark.skipif(IS_ROOT, reason="root lists any directory")
def test_a_stale_probe_directory_the_sweep_cannot_list_is_logged_left_alone_and_cannot_tell(
    workspace: Workspace,
) -> None:
    unlistable = workspace.coherence_dir / f"{PROBE_DIR_PREFIX}4242-0badc0de"
    unlistable.mkdir(mode=0o700)
    unlistable.chmod(0o000)
    try:
        result = probe_fold_profile(workspace.probe_input("held"))
    finally:
        unlistable.chmod(0o700)
    assert result == cannot_tell("stale probe entries left in place")
    assert unlistable.is_dir()
    assert workspace.logged == [f"left {unlistable} in place: EACCES"]


@needs_expected_profile
def test_a_start_without_the_lock_beside_a_probing_start_sweeps_nothing_and_the_probing_start_classifies(
    workspace: Workspace,
) -> None:
    # The probing start's directory as it stands between creating vector 2's
    # file and looking up its other spelling.
    probing = workspace.coherence_dir / f"{PROBE_DIR_PREFIX}4242-0badc0de"
    probing.mkdir(mode=0o700)
    created = probing / "p\u00e9"
    create_file(created)
    created_inode = os.lstat(created).st_ino

    assert probe_fold_profile(workspace.probe_input("unlocked")).reason == "unlocked"

    assert os.lstat(created).st_ino == created_inode
    other_spelling = probing / "pe\u0301"
    reaches_created = os.path.lexists(other_spelling) and os.lstat(other_spelling).st_ino == created_inode
    assert reaches_created == (TMP_DISK_PROFILE != "none")
    shutil.rmtree(probing)
    assert probe_fold_profile(workspace.probe_input("held")) == FoldProfileResult(
        profile=TMP_DISK_PROFILE, status=CLASSIFIED
    )


def test_a_link_at_a_vector_name_is_never_followed(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    target = workspace.root / "outside.md"
    make_directory = os.mkdir

    def make_then_plant(path: Path, mode: int = 0o777) -> None:
        make_directory(path, mode)
        if Path(path).name.startswith(PROBE_DIR_PREFIX):
            os.symlink(target, Path(path) / "p\u00e9")

    with monkeypatch.context() as patched:
        patched.setattr(os, "mkdir", make_then_plant)
        result = probe_fold_profile(workspace.probe_input("held"))
    assert (result.profile, result.status) == ("none", CANNOT_TELL)
    assert not os.path.lexists(target)


def test_the_probe_leaves_the_workspace_and_the_store_as_they_were(workspace: Workspace) -> None:
    (workspace.root / "docs").mkdir()
    (workspace.root / "docs" / "plan.md").write_text("plan\n", encoding="utf-8")
    (workspace.root / "README.md").write_text("readme\n", encoding="utf-8")
    (workspace.coherence_dir / "state.db").write_bytes(bytes([0, 1, 2, 3, 255]))
    backdate(workspace.root)
    before = snapshot(workspace.root)
    probe_fold_profile(workspace.probe_input("held"))
    assert snapshot(workspace.root) == before
    assert not touched_since_backdate(workspace.root)


# --- the coordinator start ---------------------------------------------------

PROFILE_LINE = "workspace disk fold profile"


class _EventLog(logging.Handler):
    """Collects the coordinator's log messages in one list with other events,
    so a test can read their order."""

    def __init__(self, events: list[str]) -> None:
        super().__init__(level=logging.DEBUG)
        self._events = events

    def emit(self, record: logging.LogRecord) -> None:
        self._events.append(record.getMessage())


@pytest.fixture
def coordinator_events() -> Iterator[list[str]]:
    events: list[str] = []
    handler = _EventLog(events)
    server_logger = logging.getLogger("ccs.adapters.claude_code.coordinator_server")
    saved_level = server_logger.level
    server_logger.addHandler(handler)
    server_logger.setLevel(logging.DEBUG)
    try:
        yield events
    finally:
        server_logger.removeHandler(handler)
        server_logger.setLevel(saved_level)


@needs_expected_profile
def test_a_coordinator_start_logs_one_profile_line_before_the_port_is_written(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, coordinator_events: list[str]
) -> None:
    (workspace.root / ".git").mkdir()
    write_pidfile = lifecycle._write_pidfile

    def recording_write(fd: int, pid: int, port: int) -> None:
        coordinator_events.append("port written")
        write_pidfile(fd, pid, port)

    monkeypatch.setattr(lifecycle, "_write_pidfile", recording_write)
    handle = lifecycle.ensure_coordinator_handle(workspace.root)
    try:
        assert handle is not None and handle.outcome is lifecycle.SpawnOutcome.BOUND
        profile_lines = [event for event in coordinator_events if PROFILE_LINE in event]
        assert profile_lines == [f"{PROFILE_LINE}: {TMP_DISK_PROFILE}"]
        assert coordinator_events.index(profile_lines[0]) < coordinator_events.index("port written")
        assert probe_entries(workspace.coherence_dir) == []
    finally:
        lifecycle.stop_coordinator(workspace.root)


@needs_expected_profile
def test_a_coordinator_keeps_the_disk_profile_it_found(workspace: Workspace) -> None:
    with CoordinatorHTTPServer(workspace.root) as server:
        assert server.fold_profile == FoldProfileResult(profile=TMP_DISK_PROFILE, status=CLASSIFIED)
    assert probe_entries(workspace.coherence_dir) == []


@pytest.mark.usefixtures("two_writer_store")
def test_a_coordinator_whose_store_opened_without_the_writer_lock_is_cannot_tell_unlocked(
    workspace: Workspace, coordinator_events: list[str]
) -> None:
    leftover = workspace.coherence_dir / f"{PROBE_DIR_PREFIX}1-0badc0de"
    leftover.mkdir(mode=0o700)
    with CoordinatorHTTPServer(workspace.root) as server:
        assert server.fold_profile == cannot_tell("unlocked")
    assert [event for event in coordinator_events if PROFILE_LINE in event] == [
        f"{PROFILE_LINE}: none (cannot tell: unlocked)"
    ]
    assert leftover.is_dir()
    leftover.rmdir()
