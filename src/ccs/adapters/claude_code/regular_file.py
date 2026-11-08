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
  ``close``.
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
    ``UNREADABLE``; it is ``None`` for ``READ`` and ``TOO_LARGE``. ``path`` is
    the path refused: the state directory itself when it is the link."""

    status: ReadStatus
    path: Path
    data: bytes | None = None
    error: OSError | None = None


class LinkRefused(OSError):
    """The path is a symlink, so it was not opened. Errno ELOOP."""


class NotRegularFile(OSError):
    """The path opened but is not a regular file, so it was closed unread.
    Errno EISDIR for a directory, EINVAL for a FIFO, socket or device."""


def _link_refused(path: Path) -> LinkRefused:
    return LinkRefused(errno.ELOOP, os.strerror(errno.ELOOP), str(path))


def _open_checked(path: Path, flags: int, mode: int) -> tuple[int, os.stat_result]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow and os.path.islink(path):
        # No O_NOFOLLOW here: the lstat is the only guard, with a check-then-open
        # race that no coordinator meets, because none runs on such a platform.
        raise _link_refused(path)
    extra = nofollow | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(path, flags | extra, mode)
    except OSError as exc:
        if os.path.islink(path):
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
            data = _read_up_to(handle, None if max_bytes is None else max_bytes + 1, first=info.st_size + 1)
        except OSError as exc:
            return RegularFileRead(ReadStatus.UNREADABLE, path, error=exc)
    if max_bytes is not None and len(data) > max_bytes:
        return RegularFileRead(ReadStatus.TOO_LARGE, path)
    return RegularFileRead(ReadStatus.READ, path, data)


def _read_up_to(handle: io.FileIO, limit: int | None, first: int) -> bytes:
    # FileIO.read(n) is one read(2), so a short read is not the end of the file.
    # The first read is sized to the file: one that arrives whole is not copied
    # by the join, and a small file does not allocate a whole chunk.
    chunks: list[bytes] = []
    total = 0
    size = first
    while limit is None or total < limit:
        chunk = handle.read(size if limit is None else min(size, limit - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        size = _READ_CHUNK_BYTES
    return b"".join(chunks)


def read_state_file(coherence_dir: Path, name: str, *, max_bytes: int | None = None) -> RegularFileRead:
    """:func:`read_regular_file` for ``coherence_dir / name``, refusing first
    when ``coherence_dir`` itself is a symlink: ``LINK`` with ``path`` set to
    ``coherence_dir``, and nothing opened. ``O_NOFOLLOW`` on the file alone
    would read the file of the directory the link names."""
    if os.path.islink(coherence_dir):
        return RegularFileRead(ReadStatus.LINK, coherence_dir, error=_link_refused(coherence_dir))
    return read_regular_file(coherence_dir / name, max_bytes=max_bytes)
