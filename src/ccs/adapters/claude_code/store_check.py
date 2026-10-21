# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Ask, BEFORE a coordinator start, what that start would do to the
workspace's ``.coherence/state.db`` (#345).

A coordinator start opens the store with the registry's writer open, which
creates the schema in a fresh store, opens a current one unchanged, migrates
an older one in place — after which a coordinator built for the old schema
refuses it — or refuses the store. A host that ships a coordinator wants to
know which of these a start will do before it starts one, so it can ask for
consent to a migration, or name a refusal, instead of finding out after.
:func:`check_store` answers that with a :class:`StoreCheck`, and never creates
or changes anything under the workspace while it looks.

Why it copies and dry-runs instead of reading the store (OBSERVED, #345):

- A store whose last writer died before a checkpoint keeps its truth in a
  hot ``-wal`` or ``-journal``: its db header can say "fresh" while the WAL
  holds an older build's schema, so a header or ``immutable=1`` read answers
  "nothing to migrate" for a store the start migrates.
- A ``mode=ro`` or read-only registry open replays that WAL in place, leaving
  ``-wal``/``-shm`` behind or rewriting ``-shm`` — it changes the workspace.
- A read-only classifier raises on the version before it checks
  ``registry_meta``, so a v7 store with a corrupt meta reads "will migrate"
  while the start migrates it to 8 and then fails.

So the check copies ``state.db`` plus any non-empty ``-wal``/``-journal``
(never ``-shm``: SQLite rebuilds the wal-index from the WAL) into a private
0700 temp directory, runs the very writer open a start reaches — the
``SqliteArtifactRegistry`` that ``coordinator_server`` constructs, via
:func:`~ccs.coordinator.sqlite_registry.trial_writer_open` — on the COPY, and
removes the copy. A symlinked ``state.db`` is read as SQLite reads it: the
target, with the sidecars beside the TARGET. The dry run is the spawn by
construction on every store the check copies. An absent store is answered
from a stat (``fresh``), and a sidecar holding data beside an absent or empty
store is flagged (``orphan_sidecar``) rather than dry-run: a start deletes
that sidecar unread.

The answer is relative to a Python coordinator of THIS build, and valid only
as of the check: nothing holds a lock between the check and a later start.

Outcomes (:class:`StoreCheckOutcome`):

- ``current`` — the store is at this build's schema; a start opens it as is.
- ``fresh`` — no store, a zero-byte one, or ``user_version`` 0: a start
  creates the schema. No consent needed: there is no schema to lose.
- ``will_migrate`` — the store is at an older schema (``found_version``); a
  start migrates it, and a coordinator built for that schema refuses it from
  then on. The case a host asks consent for.
- ``newer`` — the store is past this build's schema; a start refuses it.
- ``foreign`` — the sibling Node coordinator's ledger wrote it; a start
  refuses it. A LIVE Node coordinator also reads ``foreign``: it takes no
  workspace lock, and a Python start refuses its store before it could join.
- ``unusable`` — a start fails on the store itself (``reason`` names how).
- ``live`` — a Python coordinator holds the workspace lock: a start joins it
  and migrates nothing while it runs. The store is not read, so this says
  nothing about its schema: an older build's coordinator left running after
  an upgrade keeps the store at that build's schema, and the first start of
  this build after it exits migrates it.
- ``cannot_tell`` — the check could not observe the answer (``reason``).
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

# The check's SQLite calls live in the registry module (the sqlite3 module
# stays out of adapters, cli and mcp; tests/coordinator/test_caller_principal.py
# sweeps them); their failures arrive here typed as StoreSqliteError.
from ccs.coordinator.sqlite_registry import (
    SCHEMA_USER_VERSION,
    CrossRuntimeSchemaError,
    SchemaVersionError,
    StoreSqliteError,
    read_copy_user_version,
    trial_writer_open,
)
from ccs.core.exceptions import SCHEMA_NEWER_REASON

# fcntl is POSIX-only, like the lifecycle that owns the workspace lock: without
# it no coordinator starts at all, so the check has nothing to predict.
try:
    import fcntl  # type: ignore[import-not-found]
    _FCNTL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only on Windows
    fcntl = None  # type: ignore[assignment]
    _FCNTL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Check reasons (#345)
# ---------------------------------------------------------------------------
#
# ``StoreCheck.reason`` for ``live`` and ``cannot_tell``. The other outcomes
# carry the store-open reason the dry run raised (``SchemaVersionError.reason``
# — one of ``ccs.core.exceptions.SCHEMA_VERSION_REASONS``) or ``sqlite_<code>``
# for a SQLite failure on the store itself. Wire-stable: a host matches
# ``reason == CONSTANT``; add, do not mutate.
COORDINATOR_RUNNING_REASON = "coordinator_running"
"""A Python coordinator holds the workspace lock (``live``)."""

STORE_CHANGED_REASON = "store_changed"
"""The store or a sidecar changed during every copy attempt: some writer that
takes no workspace lock (a Node coordinator, a script) is writing it."""

ORPHAN_SIDECAR_REASON = "orphan_sidecar"
"""A non-empty ``-wal`` or ``-journal`` sits beside NO ``state.db``, or beside
an empty one. A start deletes it unread (SQLite discards the sidecar of a
database with no pages) and creates an empty store, and the sidecar may hold
the only copy of a removed store's pages; the check flags that instead of
calling the start a plain create."""

STORE_UNREADABLE_REASON = "store_unreadable"
"""This process cannot read the store as a regular file: it lacks a
permission the coordinator's user may have, or the path names a FIFO or a
device, which the check refuses to copy."""

CHECK_ENVIRONMENT_FAILED_REASON = "check_environment_failed"
"""The check's own side failed (no usable temp dir, a full or failing temp
disk), which says nothing about the store."""

LOCK_PROBE_FAILED_REASON = "lock_probe_failed"
"""The workspace lock could not be probed, so a running coordinator cannot be
ruled out."""

PLATFORM_UNSUPPORTED_REASON = "platform_unsupported"
"""No ``fcntl`` on this platform: no coordinator starts here."""

STORE_CHECK_REASONS: frozenset[str] = frozenset(
    {
        COORDINATOR_RUNNING_REASON,
        STORE_CHANGED_REASON,
        ORPHAN_SIDECAR_REASON,
        STORE_UNREADABLE_REASON,
        CHECK_ENVIRONMENT_FAILED_REASON,
        LOCK_PROBE_FAILED_REASON,
        PLATFORM_UNSUPPORTED_REASON,
    }
)

#: Copies taken before a store that changes under every one reads
#: ``store_changed``. One retry absorbs a single unlucky write; a store still
#: changing after three is being written continuously.
_COPY_ATTEMPTS = 3

#: The sidecars a start's open reads. Never ``-shm``: it is a cache of the WAL
#: that SQLite rebuilds when it is absent, and a copied one could be stale.
_SIDECAR_SUFFIXES = ("-wal", "-journal")

_PRIVATE_DIR_PREFIX = "ccs-store-check-"
_COPY_CHUNK_BYTES = 1 << 20

#: flock's "someone else holds a conflicting lock" errnos.
_LOCK_CONTENDED_ERRNOS = frozenset({errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES})


class StoreCheckOutcome(str, Enum):
    """What a coordinator start would do to the store (see the module doc)."""

    CURRENT = "current"
    FRESH = "fresh"
    WILL_MIGRATE = "will_migrate"
    NEWER = "newer"
    FOREIGN = "foreign"
    UNUSABLE = "unusable"
    LIVE = "live"
    CANNOT_TELL = "cannot_tell"


@dataclass(frozen=True)
class StoreCheck:
    """One answer from :func:`check_store`.

    ``found_version`` is the store's raw ``user_version`` as a start would
    read it (WAL and journal applied): past ``expected_version`` under
    ``newer``, and negative under ``unusable``/``schema_unrecognized``. It is
    ``None`` for an absent store, for ``foreign`` (a Node ledger number is not
    comparable, #294), for ``unusable`` when the copy's ``user_version``
    cannot be read (``sqlite_notadb`` and similar), and for ``live`` and
    ``cannot_tell``. ``expected_version`` is the schema this build serves.
    ``reason`` is ``None`` for ``current``, ``fresh`` and ``will_migrate``.
    ``message`` is one line of prose for a human. Branch on ``outcome`` and
    ``reason``: never on ``message``, nor on ``found_version`` alone.
    """

    outcome: StoreCheckOutcome
    db_path: Path
    found_version: int | None
    expected_version: int
    reason: str | None
    message: str

    @property
    def will_migrate(self) -> bool:
        """True exactly when a start migrates the store in place."""
        return self.outcome is StoreCheckOutcome.WILL_MIGRATE


class _CheckStopped(Exception):
    """Ends the check with ``cannot_tell(reason)``: what failed is the check's
    view of the store, not the store."""

    def __init__(self, reason: str, error: OSError) -> None:
        super().__init__(reason)
        self.reason = reason
        self.error = error


class _SourceChanged(Exception):
    """A file the copy needed vanished mid-copy: retry like a fence mismatch."""


def check_store(coordinator_root: Path) -> StoreCheck:
    """Say what a coordinator start for ``coordinator_root`` would do to its
    ``.coherence/state.db``, without creating or changing anything under the
    root (no ``.coherence/``, no sidecar, no lock file).

    1. Probe the workspace lock (``server.pid``) with a SHARED, non-blocking
       flock released at once: held means a start joins the running
       coordinator (``live``). Shared, so two concurrent checks never read each
       other as a coordinator. A start that lands inside a probe (microseconds,
       once up front and once per copy) finds the lock held and joins whatever
       port ``server.pid`` names, as it does while any other start holds the
       lock; after a crash, that port is a dead one.
    2. Resolve ``state.db`` as SQLite does (a symlink to its target) and name
       every file read below from the target, where SQLite keeps the sidecars.
       A sidecar holding data beside an absent or empty store is
       ``cannot_tell(orphan_sidecar)``; otherwise no store is ``fresh``.
    3. Stat the store and its sidecars, copy them (regular files only), stat
       again and re-probe the lock. A held lock is ``live``; a changed stat,
       or a link that now resolves elsewhere, retries the copy, up to
       :data:`_COPY_ATTEMPTS` in all, then ``cannot_tell(store_changed)``.
    4. Read the copy's ``user_version`` and run the writer open on the copy.
    """
    coherence_dir = Path(coordinator_root) / ".coherence"
    db_path = coherence_dir / "state.db"
    pid_file = coherence_dir / "server.pid"
    if not _FCNTL_AVAILABLE:
        return _cannot_tell(
            db_path, PLATFORM_UNSUPPORTED_REASON,
            "this platform has no fcntl, so no coordinator starts here",
        )
    try:
        if _coordinator_holds_lock(pid_file):
            return _live(db_path, pid_file)
        uncopied = _answer_from_stat(db_path)
        if uncopied is not None:
            return uncopied
        for _attempt in range(_COPY_ATTEMPTS):
            answer = _check_one_copy(db_path, pid_file)
            if answer is not None:
                return answer
    except _CheckStopped as stopped:
        return _cannot_tell(db_path, stopped.reason, f"{stopped.error}")
    return _cannot_tell(
        db_path, STORE_CHANGED_REASON,
        f"{db_path} changed during each of {_COPY_ATTEMPTS} copies; a writer that "
        "takes no workspace lock (a Node coordinator, a script) may be using it",
    )


# ---------------------------------------------------------------------------
# Lock probe
# ---------------------------------------------------------------------------


def _coordinator_holds_lock(pid_file: Path) -> bool:
    """True when another open file holds the workspace flock exclusively.

    Opens ``server.pid`` read-only WITHOUT ``O_CREAT`` (an absent file is a
    free lock, and the check must not create one), without following a
    symlink, and as a regular file only (a FIFO there would block the open).
    flock locks belong to the open file, so the probe sees a coordinator
    running in this very process too. Raises :class:`_CheckStopped` when the
    probe itself fails.
    """
    with _oserror_means(LOCK_PROBE_FAILED_REASON):
        try:
            fd = _open_regular(pid_file, os.O_NOFOLLOW)
        except FileNotFoundError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in _LOCK_CONTENDED_ERRNOS:
                    return True
                raise
            fcntl.flock(fd, fcntl.LOCK_UN)
            return False
        finally:
            os.close(fd)


# ---------------------------------------------------------------------------
# Fence and copy
# ---------------------------------------------------------------------------


@contextmanager
def _oserror_means(reason: str) -> Iterator[None]:
    """Turn an ``OSError`` raised in the block into ``cannot_tell(reason)``."""
    try:
        yield
    except OSError as exc:
        raise _CheckStopped(reason, exc) from exc


def _sidecar(db_path: Path, suffix: str) -> Path:
    return db_path.with_name(db_path.name + suffix)


def _resolved(db_path: Path) -> Path:
    """``db_path`` with every symlink resolved, as SQLite's unix VFS resolves
    it: SQLite names a symlinked store's ``-wal`` and ``-journal`` from the
    TARGET, so those are the sidecars a start reads, never the (empty) ones
    the registry's pre-create leaves beside the link."""
    with _oserror_means(STORE_UNREADABLE_REASON):
        return Path(os.path.realpath(db_path))


def _open_regular(path: Path, extra_flags: int = 0) -> int:
    """Open ``path`` read-only and return the descriptor, or raise
    ``OSError(EINVAL)`` when it is not a regular file. Non-blocking, because
    opening a FIFO that has no writer blocks forever; checked on the
    descriptor, because a device such as ``/dev/zero`` never reaches EOF and
    copying it would fill the temp disk. ``O_NONBLOCK`` changes nothing for a
    regular file, nor for flock."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | extra_flags)
    try:
        mode = os.fstat(fd).st_mode
    except OSError:
        os.close(fd)
        raise
    if not stat.S_ISREG(mode):
        os.close(fd)
        raise OSError(errno.EINVAL, f"{path} is not a regular file")
    return fd


def _stat_key(path: Path) -> tuple[int, int, int, int, int] | None:
    """The fields a write moves, or None for an absent file. Any other
    ``OSError`` reads as an unreadable store."""
    with _oserror_means(STORE_UNREADABLE_REASON):
        try:
            st = os.stat(path)
        except FileNotFoundError:
            return None
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _fence(store_file: Path) -> tuple[tuple[int, int, int, int, int] | None, ...]:
    """One stat key per file a start reads: the store, then each sidecar."""
    return tuple(
        _stat_key(path)
        for path in (store_file, *(_sidecar(store_file, s) for s in _SIDECAR_SUFFIXES))
    )


def _answer_from_stat(db_path: Path) -> StoreCheck | None:
    """The answer for a store a start never reads, or None when there is one
    to copy. SQLite deletes the sidecar of a database with no pages unread,
    so a sidecar holding data beside an absent OR empty store is an orphan a
    start discards; an absent store with no such sidecar is a plain create."""
    store_file = _resolved(db_path)
    store_key = _stat_key(store_file)
    if store_key is not None and store_key[2] > 0:
        return None
    for suffix in _SIDECAR_SUFFIXES:
        sidecar = _sidecar(store_file, suffix)
        sidecar_key = _stat_key(sidecar)
        if sidecar_key is not None and sidecar_key[2] > 0:
            state = "does not exist" if store_key is None else "is empty"
            return _cannot_tell(
                db_path, ORPHAN_SIDECAR_REASON,
                f"{sidecar} holds data but {db_path} {state}; a start deletes it "
                f"unread and creates an empty store at registry schema "
                f"{SCHEMA_USER_VERSION}, so move it aside first if it matters",
            )
    if store_key is not None:
        return None  # empty: the dry run creates the schema in it, as a start does
    return _result(
        StoreCheckOutcome.FRESH, db_path, None, None,
        f"no store at {db_path}; a start creates it at registry schema "
        f"{SCHEMA_USER_VERSION}",
    )


def _check_one_copy(db_path: Path, pid_file: Path) -> StoreCheck | None:
    """One fenced copy and dry run, or None when the store changed under the
    copy (the caller retries)."""
    store_file = _resolved(db_path)
    before = _fence(store_file)
    if before[0] is None:
        return None  # deleted since the presence check: changed
    with _oserror_means(CHECK_ENVIRONMENT_FAILED_REASON):
        private_dir = Path(tempfile.mkdtemp(prefix=_PRIVATE_DIR_PREFIX))
    try:
        try:
            copy = _copy_store(store_file, private_dir / db_path.name, before)
        except _SourceChanged:
            return None
        # The lock first: a coordinator that started during the copy explains
        # any change, and a start would join it.
        if _coordinator_holds_lock(pid_file):
            return _live(db_path, pid_file)
        # A link retargeted during the copy changed which store a start reads.
        if _resolved(db_path) != store_file or _fence(store_file) != before:
            return None
        return _dry_run(db_path, copy)
    finally:
        shutil.rmtree(private_dir, ignore_errors=True)


def _copy_store(
    store_file: Path, copy: Path, fence: tuple[tuple[int, int, int, int, int] | None, ...]
) -> Path:
    """Copy the store to ``copy``, plus each sidecar the fence saw non-empty
    to the matching name beside it; return ``copy``."""
    _copy_file(store_file, copy)
    for suffix, key in zip(_SIDECAR_SUFFIXES, fence[1:]):
        if key is not None and key[2] > 0:
            _copy_file(_sidecar(store_file, suffix), _sidecar(copy, suffix))
    return copy


def _copy_file(source: Path, target: Path) -> None:
    """Copy one regular file, the target created exclusively at 0600 (the
    store may hold retained content; ``mkdtemp`` already made the directory
    0700). Source failures (a source that is not a regular file included)
    read as an unreadable store, target failures as the check's environment,
    and a vanished source as a change."""
    try:
        source_file = os.fdopen(_open_regular(source), "rb")
    except FileNotFoundError as exc:
        raise _SourceChanged from exc
    except OSError as exc:
        raise _CheckStopped(STORE_UNREADABLE_REASON, exc) from exc
    with source_file, _oserror_means(CHECK_ENVIRONMENT_FAILED_REASON):
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, "wb") as target_file:
            while chunk := _read_source(source_file):
                target_file.write(chunk)


def _read_source(source_file: BinaryIO) -> bytes:
    # Raised as _CheckStopped (not OSError) so the caller's environment
    # mapping cannot relabel a source read failure as the check's own.
    try:
        return source_file.read(_COPY_CHUNK_BYTES)
    except OSError as exc:
        raise _CheckStopped(STORE_UNREADABLE_REASON, exc) from exc


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _dry_run(db_path: Path, copy: Path) -> StoreCheck:
    """Classify the copy by what the spawn's writer open does to it.

    ``found`` is read first, on the copy, as the start would see it (WAL and
    journal applied), so a refusal still reports the version the store was
    at — a v7 store with a corrupt meta is UNUSABLE at 7, although the open
    raised after migrating the copy to 8.
    """
    try:
        found = read_copy_user_version(copy)
    except StoreSqliteError as exc:
        return _from_sqlite_error(db_path, None, exc)
    try:
        trial_writer_open(copy)
    except CrossRuntimeSchemaError as exc:
        return _result(
            StoreCheckOutcome.FOREIGN, db_path, None, exc.reason,
            f"{db_path} was written under the sibling Node coordinator's ledger; "
            "a Python coordinator start refuses it (the Node backend owns it)",
        )
    except SchemaVersionError as exc:
        outcome = (
            StoreCheckOutcome.NEWER if exc.reason == SCHEMA_NEWER_REASON
            else StoreCheckOutcome.UNUSABLE
        )
        return _result(outcome, db_path, found, exc.reason, _about_store(exc, copy, db_path))
    except StoreSqliteError as exc:
        return _from_sqlite_error(db_path, found, exc)
    except OSError as exc:
        return _cannot_tell(db_path, CHECK_ENVIRONMENT_FAILED_REASON, f"{exc}")
    return _opened(db_path, found)


def _opened(db_path: Path, found: int) -> StoreCheck:
    """The writer open succeeded on a copy that was at ``found``."""
    if found == 0:
        return _result(
            StoreCheckOutcome.FRESH, db_path, 0, None,
            f"{db_path} holds no registry schema yet; a start creates schema "
            f"{SCHEMA_USER_VERSION} in it",
        )
    if found == SCHEMA_USER_VERSION:
        return _result(
            StoreCheckOutcome.CURRENT, db_path, found, None,
            f"{db_path} is at registry schema {found}; a start opens it unchanged",
        )
    # The writer open admits only 0..SCHEMA_USER_VERSION, so this is 1..9.
    return _result(
        StoreCheckOutcome.WILL_MIGRATE, db_path, found, None,
        f"{db_path} is at registry schema {found}; a start migrates it to "
        f"{SCHEMA_USER_VERSION}, and a coordinator that expects schema {found} "
        "refuses it from then on",
    )


def _from_sqlite_error(db_path: Path, found: int | None, exc: StoreSqliteError) -> StoreCheck:
    """A SQLite failure on the copy: the check's environment when its primary
    code is opener-side (temp disk, memory, the copy's own locking), else the
    store's (``sqlite_<code>``)."""
    if exc.opener_side:
        return _cannot_tell(db_path, CHECK_ENVIRONMENT_FAILED_REASON, f"{exc}")
    return _result(
        StoreCheckOutcome.UNUSABLE, db_path, found, exc.reason,
        f"a start cannot open {db_path}: {_one_line(str(exc))}",
    )


def _about_store(exc: SchemaVersionError, copy: Path, db_path: Path) -> str:
    """The refusal's own message, naming the workspace store, not the copy."""
    return _one_line(str(exc).replace(str(copy), str(db_path)))


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _live(db_path: Path, pid_file: Path) -> StoreCheck:
    return _result(
        StoreCheckOutcome.LIVE, db_path, None, COORDINATOR_RUNNING_REASON,
        f"a coordinator holds {pid_file}; a start joins it and migrates nothing "
        "while it runs. The store was not read: stop the coordinator and check "
        "again to learn what a start after it would do",
    )


def _cannot_tell(db_path: Path, reason: str, detail: str) -> StoreCheck:
    return _result(
        StoreCheckOutcome.CANNOT_TELL, db_path, None, reason,
        f"cannot tell what a start would do to {db_path} ({reason}): {_one_line(detail)}",
    )


def _result(
    outcome: StoreCheckOutcome,
    db_path: Path,
    found: int | None,
    reason: str | None,
    message: str,
) -> StoreCheck:
    return StoreCheck(
        outcome=outcome,
        db_path=db_path,
        found_version=found,
        expected_version=SCHEMA_USER_VERSION,
        reason=reason,
        message=message,
    )
