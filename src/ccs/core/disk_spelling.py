# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Two resolvers for an absolute path: one that follows links and keeps the
case the path was typed in, and one that answers with the disk's own
spelling.

- :func:`resolve_links` follows every link, as ``Path.resolve()`` does: on
  macOS ``/var`` still becomes ``/private/var`` (it is a link), and every
  other name keeps the case it was typed in.
- :func:`native_spelling` also returns each name as the disk stores it, so
  ``REPO/PLAN.MD`` on a case-insensitive APFS volume comes back as
  ``Repo/plan.md``. On macOS that takes libc ``realpath(3)``, which
  ``os.path.realpath`` does not call; elsewhere the two resolvers agree.

Both resolve the deepest ancestor that exists and keep the rest as typed, so
a file about to be created has a spelling too. An entry that exists yet does
not resolve, a dangling or looping link, raises :class:`LinkDoesNotResolve`
rather than being keyed by the link's own path: that would give its file a
second identity once the target exists.

Standard library only, in ``ccs.core`` so the coordinator and the hook
clients can both import it.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import stat
import sys
from collections.abc import Callable

Resolver = Callable[[str], str]

#: Darwin's PATH_MAX: realpath(3) writes at most this many bytes.
_DARWIN_PATH_MAX = 1024


class LinkDoesNotResolve(OSError):
    """An entry on the path exists yet does not resolve: a dangling or
    looping link. ``filename`` is that entry."""


def _follow_links(path: str) -> str:
    return os.path.realpath(path, strict=True)


def _load_darwin_realpath() -> Callable[[bytes, ctypes.Array[ctypes.c_char]], bytes | None]:
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    # The symbol C programs (and Node's realpathSync.native) link against on macOS.
    function = getattr(libc, "realpath$DARWIN_EXTSN", None) or libc.realpath
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    function.restype = ctypes.c_char_p
    return function


def _darwin_realpath(path: str) -> str:
    resolved = ctypes.create_string_buffer(_DARWIN_PATH_MAX)
    if _DARWIN_REALPATH(os.fsencode(path), resolved) is None:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), path)
    return os.fsdecode(resolved.value)


if sys.platform == "darwin":
    _DARWIN_REALPATH = _load_darwin_realpath()
    _NATIVE_RESOLVER: Resolver = _darwin_realpath
else:
    _NATIVE_RESOLVER = _follow_links


#: The lstat errnos that mean no entry is reachable at this level, so the walk
#: goes up to the parent: ELOOP when a looping link sits above it, which the
#: walk then reports as the link that does not resolve.
_ABSENT_ENTRY_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP})


def _lstat_or_none(path: str) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except OSError as exc:
        if exc.errno in _ABSENT_ENTRY_ERRNOS:
            return None
        raise


def _resolve_deepest_existing(absolute: str, resolver: Resolver) -> str:
    try:
        return resolver(absolute)
    except OSError as exc:
        unresolved = exc
    entry = _lstat_or_none(absolute)
    if entry is not None and stat.S_ISLNK(entry.st_mode):
        raise LinkDoesNotResolve(unresolved.errno, "link does not resolve", absolute) from unresolved
    if entry is not None:
        raise unresolved
    parent, name = os.path.split(absolute)
    return os.path.join(_resolve_deepest_existing(parent, resolver), name)


def _require_absolute(path: str | os.PathLike[str]) -> str:
    text = os.fspath(path)
    if not os.path.isabs(text):
        raise ValueError(f"not an absolute path: {text!r}")
    return text


def resolve_links(path: str | os.PathLike[str]) -> str:
    """``path`` with every link followed and the typed case kept; a tail that
    does not exist yet is appended as typed. Raises
    :class:`LinkDoesNotResolve` for a dangling or looping link on the path,
    and ``ValueError`` for a relative path."""
    return _resolve_deepest_existing(_require_absolute(path), _follow_links)


def native_spelling(path: str | os.PathLike[str]) -> str:
    """``path`` as the disk spells it: links followed and, on macOS, each
    existing name in the case and form the disk stores; a tail that does not
    exist yet is appended as typed. Raises as :func:`resolve_links` does."""
    return _resolve_deepest_existing(_require_absolute(path), _NATIVE_RESOLVER)
