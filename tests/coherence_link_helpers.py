# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Links planted in a workspace's ``.coherence`` and what they name, shared by
the start-refusal tests (#498): every surface that starts, joins or prepares a
coordinator must refuse a link there and leave the file or directory it names
as it was."""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

#: Names a link in ``.coherence`` may take: what a start writes, reads or
#: creates beside the store, the policy files and their lock, the logs, a
#: caller-principal file, and a name no code reads (the check scans every
#: entry, not a list of known names).
LINK_NAMES = (
    ".gitignore",
    "server.pid",
    "hook.secret",
    "state.db",
    "state.db-wal",
    "state.db-shm",
    "state.db-journal",
    "workspace.db",
    "tracked.yaml",
    "ignored.yaml",
    "strict_mode.yaml",
    "tracked.yaml.lock",
    "audit.log",
    "session-audit.log",
    "caller-principal-" + "0" * 32 + ".nonce",
    "notes.txt",
)

#: What a planted link names: a regular file, nothing, a FIFO or a directory.
LINK_SHAPES = ("existing", "dangling", "to_fifo", "to_dir")

#: The bytes a planted link's existing target holds.
TARGET_BYTES = b"keep\n"


def plant_link(coherence_dir: Path, name: str, shape: str) -> Path:
    """Make ``coherence_dir / name`` a symlink of ``shape`` (one of
    :data:`LINK_SHAPES`) and return its target. The target sits in the
    workspace, outside ``.coherence``, in a directory of its own, so a test
    can tell whether anything created a file beside it. A ``to_fifo`` target
    blocks whoever opens it: the test releases it."""
    coherence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    targets = coherence_dir.parent / "link-targets"
    targets.mkdir(exist_ok=True)
    target = targets / name
    if shape == "existing":
        target.write_bytes(TARGET_BYTES)
    elif shape == "to_fifo":
        os.mkfifo(target)
    elif shape == "to_dir":
        target.mkdir()
    elif shape != "dangling":
        raise ValueError(f"unknown link shape {shape!r}")
    (coherence_dir / name).symlink_to(target)
    return target


def release_fifo(fifo: Path) -> None:
    """Give anything blocked opening ``fifo`` a writer, then remove it. A
    FIFO test calls this in ``finally``, so a regression that opens the FIFO
    fails its deadline instead of hanging the run."""
    try:
        fd = os.open(fifo, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return
    try:
        fifo.unlink(missing_ok=True)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class TargetSnapshot:
    """What is at a path, read without following a link or opening anything
    but a regular file: ``content`` is a regular file's bytes or a
    directory's sorted names, ``link`` a symlink's ``readlink``."""

    exists: bool
    mode: int | None = None
    content: bytes | tuple[str, ...] | None = None
    link: str | None = None


def target_snapshot(path: Path) -> TargetSnapshot:
    """Snapshot ``path`` so a test can assert it is unchanged; a FIFO's
    content is never read, so the snapshot cannot block."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return TargetSnapshot(exists=False)
    if stat.S_ISLNK(mode):
        return TargetSnapshot(True, mode, link=os.readlink(path))
    if stat.S_ISREG(mode):
        return TargetSnapshot(True, mode, content=path.read_bytes())
    if stat.S_ISDIR(mode):
        return TargetSnapshot(True, mode, content=tuple(sorted(os.listdir(path))))
    return TargetSnapshot(True, mode)
