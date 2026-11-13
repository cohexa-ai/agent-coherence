# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The start check on a workspace's ``.coherence`` (#498), and the refusal
texts every surface prints for it.

``.coherence`` must be absent or a real directory, and a mount counts as one.
No entry directly inside it may be a symlink, FIFO, socket or device. A
cloned repository can commit a link there; a start, a join or a port probe
would then read or write the file or directory the link names, and a FIFO
blocks the open that meets it. No coordinator creates any of these, so
refusing them costs nothing.

:func:`check_coherence_dir` runs ``lstat`` on ``.coherence`` and lists it,
taking each entry's type from the listing. It opens no file, follows no link
and runs no git command, so a caller can run it before anything else it does
under ``.coherence``. A link above ``.coherence`` (a linked home or temp
directory) is outside the rule.

The texts are wire text: the Node coordinator and the plugin shims print the
same strings, so a reword here is a reword there. Paths are kept raw; a
printer escapes them.

Standard library and ``ccs.core.exceptions`` only.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from ccs.core.exceptions import (
    COHERENCE_LINK_KIND,
    COHERENCE_NOT_A_DIRECTORY_KIND,
    COHERENCE_SPECIAL_ENTRY_KIND,
    CoherenceDirRefused,
)

_STORE_NAME_PREFIXES = ("state.db", "workspace.db")
_STORE_NOTE = " (move state.db together with its -wal and -shm)"


def check_coherence_dir(coordinator_root: Path) -> None:
    """Raise :class:`~ccs.core.exceptions.CoherenceDirRefused` when
    ``coordinator_root``'s ``.coherence`` is a symlink or not a directory, or
    holds a symlink, FIFO, socket or device directly inside it.

    Several offending entries are named by the first in sorted (byte) order
    plus a count, links before the rest. An absent ``.coherence`` passes (a
    start creates a real one), and so does one this user cannot look at or
    list: lifecycle repairs the mode of its own directory and checks again,
    and any other later open fails the same way.
    """
    coherence_dir = Path(coordinator_root) / ".coherence"
    try:
        mode = os.lstat(coherence_dir).st_mode
    except OSError:
        return
    if stat.S_ISLNK(mode):
        raise _refused(COHERENCE_LINK_KIND, coherence_dir, dir_link_refusal(coherence_dir))
    if not stat.S_ISDIR(mode):
        raise _refused(COHERENCE_NOT_A_DIRECTORY_KIND, coherence_dir, not_a_directory_refusal(coherence_dir))
    try:
        links, specials = _offending_entries(coherence_dir)
    except OSError:
        return
    if links:
        raise entry_link_refused(links[0], more=len(links) - 1)
    if specials:
        raise _refused(
            COHERENCE_SPECIAL_ENTRY_KIND, specials[0], special_entry_refusal(specials[0], more=len(specials) - 1)
        )


def entry_link_refused(path: Path, more: int = 0) -> CoherenceDirRefused:
    """The refusal of a symlink directly inside ``.coherence``, as the start
    check raises it; ``more`` counts the other links there."""
    return _refused(COHERENCE_LINK_KIND, path, entry_link_refusal(path, more))


def _refused(kind: str, path: Path, message: str) -> CoherenceDirRefused:
    return CoherenceDirRefused(kind, str(path), message)


def _offending_entries(coherence_dir: Path) -> tuple[list[Path], list[Path]]:
    """The links and the FIFOs, sockets and devices directly inside
    ``coherence_dir``, each sorted by the name's bytes, the order the Node
    coordinator sorts in. The type comes from the listing (``d_type``), so an
    entry costs no syscall: the ``caller-principal-*`` files are never
    pruned, and the directory grows with every session."""
    links: list[str] = []
    specials: list[str] = []
    with os.scandir(coherence_dir) as entries:
        for entry in entries:
            if entry.is_symlink():
                links.append(entry.name)
            elif _is_special(entry):
                specials.append(entry.name)
    return _sorted_paths(coherence_dir, links), _sorted_paths(coherence_dir, specials)


def _is_special(entry: os.DirEntry[str]) -> bool:
    if entry.is_file(follow_symlinks=False) or entry.is_dir(follow_symlinks=False):
        return False
    # Both also answer False for an entry removed mid-scan (a policy *.tmp
    # renamed into place); only one still there is special.
    try:
        entry.stat(follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _sorted_paths(coherence_dir: Path, names: list[str]) -> list[Path]:
    return [coherence_dir / name for name in sorted(names, key=os.fsencode)]


def _more_text(more: int, one: str, many: str) -> str:
    if more < 1:
        return ""
    if more == 1:
        return f" 1 more entry in .coherence {one} too."
    return f" {more} more entries in .coherence {many} too."


def dir_link_refusal(coherence_dir: Path) -> str:
    """The refusal of a ``.coherence`` that is itself a symlink."""
    return (
        f"refusing {coherence_dir}: it is a symlink, and .coherence must be a real directory. "
        "Replace the link with a real directory and copy in the files you want to keep"
        f"{_STORE_NOTE}, or mount a directory at {coherence_dir}."
    )


def entry_link_refusal(path: Path, more: int = 0) -> str:
    """The refusal of a symlink directly inside ``.coherence``; ``more``
    counts the other links there."""
    store_note = _STORE_NOTE if path.name.startswith(_STORE_NAME_PREFIXES) else ""
    return (
        f"refusing {path}: it is a symlink, and .coherence must be a real directory with no link "
        f"directly inside it. Replace the link with a copy of the file it names{store_note}, "
        f"or mount a directory at {path.parent}.{_more_text(more, 'is a link', 'are links')}"
    )


def state_file_link_refusal(link_path: Path, coherence_dir: Path) -> str:
    """The refusal for a :func:`~ccs.adapters.claude_code.regular_file.read_state_file`
    ``LINK`` answer: ``link_path`` is the answer's path, ``coherence_dir``
    itself when the state directory is the link."""
    if link_path == coherence_dir:
        return dir_link_refusal(coherence_dir)
    return entry_link_refusal(link_path)


def not_a_directory_refusal(coherence_dir: Path) -> str:
    """The refusal of a ``.coherence`` that exists and is not a directory."""
    return (
        f"refusing {coherence_dir}: it is not a directory. Move it aside; a start then creates "
        ".coherence as a real directory."
    )


def special_entry_refusal(path: Path, more: int = 0) -> str:
    """The refusal of a FIFO, socket or device directly inside
    ``.coherence``; ``more`` counts the others there."""
    more_text = _more_text(more, "is a FIFO, socket or device", "are FIFOs, sockets or devices")
    return (
        f"refusing {path}: it is a FIFO, socket or device, which no coordinator creates, and "
        f"opening it can block a start. Remove it, then start again.{more_text}"
    )


def policy_link_refusal(name: str) -> str:
    """The policy routes' and merges' refusal of a linked policy file.
    ``name`` is the file as the routes print it: ``.coherence/tracked.yaml``."""
    return (
        f"{name} is a symlink; nothing was written. Replace it with a copy of the file it names, "
        "or mount a directory at .coherence."
    )


def lock_link_refusal(name: str) -> str:
    """The policy routes' refusal of a linked lock sidecar. ``name`` is the
    policy file's, as for :func:`policy_link_refusal`; the sidecar is
    ``<name>.lock``."""
    return f"{name}.lock is a symlink; nothing was written. Remove the link."
