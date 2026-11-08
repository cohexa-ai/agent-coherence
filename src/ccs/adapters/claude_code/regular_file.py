# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Open or read a path only when it is a regular file, never through a link
(#498).

One reader for every read that must not follow a link, block on a FIFO or
read a device: the state files a client reads under ``.coherence`` without a
start, the store check's copy and lock probe, and the workspace-file hash.
Every open adds ``O_NOFOLLOW``, ``O_NONBLOCK`` and ``O_CLOEXEC``, looked up on
each call because a platform (or a test) may lack them, and then checks the
descriptor with ``fstat``. Standard library only, so every module of the
package can import it.

Two rules keep it safe to call from anywhere in the library:

- A link is told by ``lstat``, never by the errno of the failed open: macOS
  answers ENOTDIR, not ELOOP, for some links, and the BSDs use other codes.
- Data is read through :class:`io.FileIO`, never ``io.open``, ``open`` or
  ``Path.read_*``: the volume's all-glob shim patches those and routes a
  managed path back into the volume. ``FileIO`` also closes with the C
  ``close``, which lifecycle tests that count ``os.close`` calls by descriptor
  number never see.

``O_NOFOLLOW`` guards the last component only. :func:`read_state_file` also
looks at ``.coherence`` itself, so a read never goes through a linked state
directory into the one it names.
"""

from __future__ import annotations

import errno
import io
import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_READ_CHUNK_BYTES = 1 << 20


class ReadStatus(str, Enum):
    """What :func:`read_regular_file` found at the path."""

    READ = "read"
    ABSENT = "absent"
    LINK = "link"
    NOT_REGULAR = "not_regular"
    TOO_LARGE = "too_large"
    UNREADABLE = "unreadable"


@dataclass(frozen=True)
class RegularFileRead:
    """One read's answer. ``data`` is set for ``READ`` only. ``error`` keeps
    the ``OSError`` behind ``ABSENT``, ``LINK``, ``NOT_REGULAR`` and
    ``UNREADABLE``, so a caller can rebuild the text it printed before; it is
    ``None`` for ``READ`` and ``TOO_LARGE``. ``path`` is the path refused:
    the state directory itself when it is the link."""

    status: ReadStatus
    path: Path
    data: bytes | None = None
    error: OSError | None = None


class LinkRefused(OSError):
    """The path is a symlink, so it was not opened. Errno ELOOP."""


class NotRegularFile(OSError):
    """The path opened but is not a regular file, so it was closed unread.
    Errno EISDIR for a directory, EINVAL for a FIFO, socket or device."""


def _is_link(path: Path) -> bool:
    try:
        return stat.S_ISLNK(os.lstat(path).st_mode)
    except OSError:
        return False


def _link_refused(path: Path) -> LinkRefused:
    return LinkRefused(errno.ELOOP, os.strerror(errno.ELOOP), str(path))


def _open_checked(path: Path, flags: int, mode: int) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and _is_link(path):
        # No O_NOFOLLOW here: the lstat is the only guard, with a check-then-open
        # race that no coordinator meets, because none runs on such a platform.
        raise _link_refused(path)
    extra = nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags | extra, mode)
    except OSError as exc:
        if _is_link(path):
            raise _link_refused(path) from exc
        raise
    try:
        info = os.fstat(fd)
    except OSError:
        os.close(fd)
        raise
    if not stat.S_ISREG(info.st_mode):
        # os.close, not FileIO: FileIO refuses a directory's descriptor and
        # leaves it open.
        os.close(fd)
        code = errno.EISDIR if stat.S_ISDIR(info.st_mode) else errno.EINVAL
        raise NotRegularFile(code, f"{path} is not a regular file")
    return fd, info


def open_regular(path: Path, flags: int = os.O_RDONLY, mode: int = 0o600) -> int:
    """``os.open(path, flags, mode)`` for a regular file only, and return the
    descriptor, which the caller closes.

    Raises :class:`LinkRefused` for a symlink at ``path`` (dangling or not;
    with ``O_CREAT`` nothing is created at its target), :class:`NotRegularFile`
    for anything else that is not a regular file, and the open's own
    ``OSError`` otherwise (``FileNotFoundError`` for an absent path). Never
    blocks on a FIFO. ``O_NONBLOCK`` changes nothing for a regular file, nor
    for ``flock``.
    """
    fd, _ = _open_checked(path, flags, mode)
    return fd


def read_regular_file(path: Path, *, max_bytes: int | None = None) -> RegularFileRead:
    """Read ``path`` whole when it is a regular file of at most ``max_bytes``
    bytes (no cap when ``None``); otherwise say what it is. Never raises an
    ``OSError``. A file that grows past the cap while it is read is
    ``TOO_LARGE`` too: the read stops one byte past the cap."""
    try:
        fd, info = _open_checked(path, os.O_RDONLY, 0o600)
    except LinkRefused as exc:
        return RegularFileRead(ReadStatus.LINK, path, error=exc)
    except NotRegularFile as exc:
        return RegularFileRead(ReadStatus.NOT_REGULAR, path, error=exc)
    except FileNotFoundError as exc:
        return RegularFileRead(ReadStatus.ABSENT, path, error=exc)
    except OSError as exc:
        return RegularFileRead(ReadStatus.UNREADABLE, path, error=exc)
    with io.FileIO(fd, "rb", closefd=True) as handle:
        if max_bytes is not None and info.st_size > max_bytes:
            return RegularFileRead(ReadStatus.TOO_LARGE, path)
        try:
            data = _read_up_to(handle, None if max_bytes is None else max_bytes + 1)
        except OSError as exc:
            return RegularFileRead(ReadStatus.UNREADABLE, path, error=exc)
    if max_bytes is not None and len(data) > max_bytes:
        return RegularFileRead(ReadStatus.TOO_LARGE, path)
    return RegularFileRead(ReadStatus.READ, path, data)


def _read_up_to(handle: io.FileIO, limit: int | None) -> bytes:
    # FileIO.read(n) is one read(2), so a short read is not the end of the file.
    chunks: list[bytes] = []
    total = 0
    while limit is None or total < limit:
        size = _READ_CHUNK_BYTES if limit is None else min(_READ_CHUNK_BYTES, limit - total)
        chunk = handle.read(size)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def read_state_file(coherence_dir: Path, name: str, *, max_bytes: int | None = None) -> RegularFileRead:
    """:func:`read_regular_file` for ``coherence_dir / name``, refusing first
    when ``coherence_dir`` itself is a symlink: ``LINK`` with ``path`` set to
    ``coherence_dir``, and nothing opened. ``O_NOFOLLOW`` on the file alone
    would read the file of the directory the link names."""
    if _is_link(coherence_dir):
        return RegularFileRead(ReadStatus.LINK, coherence_dir, error=_link_refused(coherence_dir))
    return read_regular_file(coherence_dir / name, max_bytes=max_bytes)
