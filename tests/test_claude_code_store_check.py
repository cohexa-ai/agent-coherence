# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The store check: what a coordinator start would do to ``state.db`` (#345).

:func:`check_store` must answer exactly what the spawn's writer open would do
— create, open unchanged, migrate, or refuse — or say it cannot tell, and it
must never create or change anything under the workspace while it looks.
Every shape below is therefore asserted three ways: the outcome, an unchanged
(name, size, sha256) listing of the workspace and its parent, and AGREEMENT
with a real start on a separate copy of the same files: the start's own
``CoordinatorHTTPServer`` built on it, so the check's open is compared with
the server's, not with itself. The temp directory the check copies into is
redirected to an empty one, which must be empty again after.

The hot-WAL and hot-journal shapes need a writer killed mid-flight; those
tests start a child process, SIGKILL only that child, and reap it.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from ccs.adapters.claude_code import store_check
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.adapters.claude_code.store_check import StoreCheck, StoreCheckOutcome, check_store
from ccs.coordinator import sqlite_registry
from ccs.coordinator.sqlite_registry import (
    SCHEMA_USER_VERSION,
    CrossRuntimeSchemaError,
    SchemaVersionError,
    SqliteArtifactRegistry,
)
from tests.coordinator.test_checkpoint_registration_schema import (
    _revert_to_v7_shape,
    _revert_to_v9_shape,
)
from tests.test_sqlite_registry import (
    _build_raw_v1_db,
    _forge_node_v3_db,
    _stamp_schema_runtime,
)

#: FROZEN duplicates of the outcome values a host hard-codes (#345).
_CURRENT = "current"
_FRESH = "fresh"
_WILL_MIGRATE = "will_migrate"
_NEWER = "newer"
_FOREIGN = "foreign"
_UNUSABLE = "unusable"
_LIVE = "live"
_CANNOT_TELL = "cannot_tell"

#: FROZEN duplicates of the check's own reasons (#345).
_COORDINATOR_RUNNING = "coordinator_running"
_STORE_CHANGED = "store_changed"
_ORPHAN_SIDECAR = "orphan_sidecar"
_STORE_UNREADABLE = "store_unreadable"
_CHECK_ENVIRONMENT_FAILED = "check_environment_failed"
_LOCK_PROBE_FAILED = "lock_probe_failed"
_PLATFORM_UNSUPPORTED = "platform_unsupported"

#: FROZEN duplicate of the prefix of the check's private copy directory.
_PRIVATE_DIR_PREFIX = "ccs-store-check-"

#: FROZEN duplicates of the store-open reasons a check passes through.
_SCHEMA_NEWER = "schema_newer"
_SCHEMA_UNRECOGNIZED = "schema_unrecognized"
_REGISTRY_META_INCOMPLETE = "registry_meta_incomplete"
_CROSS_RUNTIME = "cross_runtime_schema"
_SQLITE_NOTADB = "sqlite_notadb"


# ----------------------------------------------------------------------
# Shapes
# ----------------------------------------------------------------------


def _db(root: Path) -> Path:
    return root / ".coherence" / "state.db"


def _sql(db: Path, *statements: str) -> None:
    conn = sqlite3.connect(str(db))
    try:
        for statement in statements:
            conn.execute(statement)
        conn.commit()
    finally:
        conn.close()


def _coherence_dir(root: Path) -> Path:
    directory = root / ".coherence"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def _current(root: Path) -> Path:
    db = _db(root)
    SqliteArtifactRegistry(db).close()
    return db


def _build(shape: str, root: Path) -> None:
    """Lay one store shape down under ``root``; every connection is closed,
    so a WAL store is checkpointed and carries no sidecars."""
    root.mkdir(parents=True, exist_ok=True)
    if shape == "absent":
        return
    if shape == "zero_byte":
        (_coherence_dir(root) / "state.db").write_bytes(b"")
    elif shape == "text":
        (_coherence_dir(root) / "state.db").write_text("hello, not sqlite\n" * 300)
    elif shape == "v0_with_table":
        _sql(_coherence_dir(root) / "state.db", "CREATE TABLE stray (x)")
    elif shape == "raw_v1":
        _build_raw_v1_db(_coherence_dir(root) / "state.db", fence=True, notices=True)
    elif shape == "node_v3":
        _forge_node_v3_db(_coherence_dir(root) / "state.db")
    elif shape in ("v10", "v10_hash_dir"):
        _current(root)
    elif shape == "v9":
        _revert_to_v9_shape(_current(root))
    elif shape == "v7":
        _revert_to_v7_shape(_current(root))
    elif shape == "v11":
        _sql(_current(root), "PRAGMA user_version = 11")
    elif shape == "uv_negative":
        _sql(_current(root), "PRAGMA user_version = -1")
    elif shape == "node_stamp":
        _stamp_schema_runtime(_current(root), "node")
    elif shape == "corrupt_meta":
        _sql(_current(root), "DELETE FROM registry_meta WHERE key = 'instance_id'")
    elif shape == "v7_corrupt_meta":
        db = _current(root)
        _revert_to_v7_shape(db)
        _sql(db, "DELETE FROM registry_meta WHERE key = 'coordinator_epoch'")
    elif shape in ("seq_text", "seq_null"):
        value = "'abc'" if shape == "seq_text" else "NULL"
        _sql(
            _current(root),
            f"UPDATE registry_meta SET value = {value} WHERE key = 'sequence_number'",
        )
    else:  # pragma: no cover - a typo in a parametrize table
        raise AssertionError(f"unknown shape {shape!r}")


def _tree(directory: Path) -> list[tuple[str, object, object]]:
    """(name, size, sha256) for every file, and every directory, under
    ``directory`` — a created ``.coherence/`` or a stray ``ws`` file beside a
    ``ws#hash`` workspace shows up as a difference."""
    entries: list[tuple[str, object, object]] = []
    for path in sorted(directory.rglob("*")):
        name = str(path.relative_to(directory))
        if path.is_symlink():
            entries.append((name, "symlink", os.readlink(path)))
        elif path.is_dir():
            entries.append((name, "dir", None))
        elif not path.is_file():
            entries.append((name, "special", None))  # a FIFO: reading it would block
        else:
            entries.append(
                (name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
            )
    return entries


def _raw_user_version(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _spawn_truth(root: Path, scratch: Path) -> tuple[str, int | None]:
    """What a coordinator start would do to a SEPARATE copy of the
    workspace's ``.coherence/`` (every file, sidecars included); see
    :func:`_start_outcome`."""
    twin = scratch / "twin"
    source = root / ".coherence"
    if source.exists():
        shutil.copytree(source, twin / ".coherence", symlinks=True)
    return _start_outcome(twin)


def _start_outcome(root: Path) -> tuple[str, int | None]:
    """What a coordinator start on ``root`` does, observed: the start's own
    ``CoordinatorHTTPServer`` built on it (never served), classified by what
    its open did, with the schema the start says it migrated from. The real
    server, not the registry the check calls, so a step the server adds
    before or around its open shows up as a disagreement."""
    root.mkdir(parents=True, exist_ok=True)
    db = root / ".coherence" / "state.db"
    try:
        before = _raw_user_version(db) if db.exists() else 0
    except sqlite3.DatabaseError:
        before = None  # not a database: the start must refuse it too
    try:
        server = CoordinatorHTTPServer(root, port=0)
    except CrossRuntimeSchemaError:
        return _FOREIGN, None
    except SchemaVersionError:
        return (_NEWER if (before or 0) > SCHEMA_USER_VERSION else _UNUSABLE), None
    except sqlite3.DatabaseError:
        return _UNUSABLE, None
    try:
        migrated = server.registry.migrated_from_version
    finally:
        server.shutdown()
    if migrated is not None:
        return _WILL_MIGRATE, migrated
    if before == 0:
        return _FRESH, None
    # Opened unchanged: current only at this build's schema; anything else is
    # a start the check's vocabulary does not describe, so it never agrees.
    return (_CURRENT if before == SCHEMA_USER_VERSION else f"opened_at_{before}"), None


def _truth(outcome: str, found: int | None) -> tuple[str, int | None]:
    """The :func:`_start_outcome` that agrees with a check answering
    ``outcome`` at ``found``: a start that migrates says it migrated from
    exactly the version the check found."""
    return outcome, (found if outcome == _WILL_MIGRATE else None)


@pytest.fixture
def check_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory standing in for the system temp dir, so a test can
    assert that every copy the check made is gone afterwards."""
    directory = tmp_path / "check-tmp"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


@pytest.fixture
def copies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every private directory the check made, in order."""
    made: list[str] = []
    real_mkdtemp = tempfile.mkdtemp

    def recording_mkdtemp(*args, **kwargs):
        made.append(real_mkdtemp(*args, **kwargs))
        return made[-1]

    monkeypatch.setattr(tempfile, "mkdtemp", recording_mkdtemp)
    return made


@pytest.fixture
def parent(tmp_path: Path) -> Path:
    directory = tmp_path / "parent"
    directory.mkdir()
    return directory


def _assert_untouched_check(
    root: Path, parent: Path, check_tmp: Path
) -> StoreCheck:
    """Run the check and assert it changed nothing in the workspace, its
    parent or the temp dir."""
    before = _tree(parent)
    result = check_store(root)
    assert _tree(parent) == before
    assert list(check_tmp.iterdir()) == []
    assert result.db_path == _db(root)
    assert result.expected_version == SCHEMA_USER_VERSION
    return result


# ----------------------------------------------------------------------
# Cold shapes
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "outcome", "found", "reason"),
    [
        ("absent", _FRESH, None, None),
        ("zero_byte", _FRESH, 0, None),
        ("v0_with_table", _FRESH, 0, None),
        ("v10", _CURRENT, SCHEMA_USER_VERSION, None),
        ("v10_hash_dir", _CURRENT, SCHEMA_USER_VERSION, None),
        ("v9", _WILL_MIGRATE, 9, None),
        ("v7", _WILL_MIGRATE, 7, None),
        ("raw_v1", _WILL_MIGRATE, 1, None),
        ("v11", _NEWER, 11, _SCHEMA_NEWER),
        ("node_stamp", _FOREIGN, None, _CROSS_RUNTIME),
        ("node_v3", _FOREIGN, None, _CROSS_RUNTIME),
        ("corrupt_meta", _UNUSABLE, SCHEMA_USER_VERSION, _REGISTRY_META_INCOMPLETE),
        ("v7_corrupt_meta", _UNUSABLE, 7, _REGISTRY_META_INCOMPLETE),
        ("seq_text", _UNUSABLE, SCHEMA_USER_VERSION, _REGISTRY_META_INCOMPLETE),
        ("seq_null", _UNUSABLE, SCHEMA_USER_VERSION, _REGISTRY_META_INCOMPLETE),
        ("uv_negative", _UNUSABLE, -1, _SCHEMA_UNRECOGNIZED),
        ("text", _UNUSABLE, None, _SQLITE_NOTADB),
    ],
)
def test_check_store_answers_what_the_spawn_does_and_touches_nothing(
    shape: str, outcome: str, found: int | None, reason: str | None,
    parent: Path, check_tmp: Path, copies: list[str], tmp_path: Path,
) -> None:
    """Each cold shape gets the outcome a coordinator start would produce —
    the same one a writer open on a separate copy produces — with the store's
    schema version and the refusal reason, and the check leaves the
    workspace, its parent and the temp dir as it found them. ``ws#hash``
    pins the plain-path open: a ``file:`` URI would cut the path at ``#`` and
    create a stray ``ws`` file. ``v7_corrupt_meta`` pins the dry run: a
    read-only classifier calls it "will migrate", but the spawn migrates it to
    8 and then fails. ``seq_text`` and ``seq_null`` hold a ``registry_meta``
    value a start cannot parse: unusable like a missing key, never an
    exception (which the CLI would turn into exit 1, "not a git
    repository"). Prevents a host acting on an answer the spawn would
    contradict, the check itself changing the store it reports on, or a
    refusal pointing the operator at the deleted private copy instead of the
    workspace store."""
    root = parent / ("ws#hash" if shape == "v10_hash_dir" else "ws")
    _build(shape, root)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.found_version, result.reason) == (outcome, found, reason)
    assert result.will_migrate is (outcome == _WILL_MIGRATE)
    # The message names the workspace store, never the deleted private copy.
    assert str(_db(root)) in result.message
    assert _PRIVATE_DIR_PREFIX not in result.message
    assert _spawn_truth(root, tmp_path / "truth") == _truth(outcome, found)
    # Only a store that exists is copied; an absent one is answered from a stat.
    assert len(copies) == (0 if shape == "absent" else 1)


def test_a_negative_user_version_reads_unusable_with_a_backup_remedy(
    parent: Path, check_tmp: Path
) -> None:
    """The check passes the start's own refusal text through, so a store no
    build stamped reads ``unusable`` with advice to restore it from a backup.
    Prevents ``--check-store`` printing "upgrade" advice for a newer store
    beside an outcome that says the store is broken."""
    root = parent / "ws"
    _build("uv_negative", root)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.reason) == (_UNUSABLE, _SCHEMA_UNRECOGNIZED)
    # The path is masked so a temp directory's name cannot satisfy or trip a
    # phrase check.
    message = result.message.replace(str(_db(root)), "<db>").lower()
    assert "backup" in message, result.message
    for wrong_advice in ("newer", "upgrade"):
        assert wrong_advice not in message, result.message


def test_the_outcome_vocabulary_is_exactly_the_eight_frozen_values() -> None:
    """The outcome enum carries exactly the eight values a host hard-codes,
    and each member is its own string. Prevents a renamed or added outcome
    reaching a host that maps them to actions."""
    assert {member.value for member in StoreCheckOutcome} == {
        _CURRENT, _FRESH, _WILL_MIGRATE, _NEWER, _FOREIGN, _UNUSABLE, _LIVE, _CANNOT_TELL,
    }
    assert all(member == member.value for member in StoreCheckOutcome)


# ----------------------------------------------------------------------
# Hot shapes: a writer SIGKILLed before it checkpointed
# ----------------------------------------------------------------------

#: The child writer. It changes the store, says READY, then blocks so the
#: test can SIGKILL it before any checkpoint, close or rollback runs.
_CHILD_WRITER = r"""
import sqlite3
import sys
import time

mode, db = sys.argv[1], sys.argv[2]
keep = []
if mode == "migrate":
    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
    keep.append(SqliteArtifactRegistry(db))
elif mode == "fresh_v9":
    # An older (v9) build's fresh create: the schema, rewound to v9's shape,
    # all of it still in the WAL.
    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
    keep.append(SqliteArtifactRegistry(db))
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("ALTER TABLE workspace_checkpoints DROP COLUMN receiver")
    conn.execute("ALTER TABLE workspace_checkpoints DROP COLUMN registered_by")
    conn.execute("PRAGMA user_version = 9")
    keep.append(conn)
elif mode == "uv11":
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("PRAGMA user_version = 11")
    keep.append(conn)
elif mode == "spill":
    # Rollback-journal store: an uncommitted transaction large enough to
    # spill dirty pages into the db file, leaving a hot -journal behind.
    conn = sqlite3.connect(db, isolation_level=None)
    conn.execute("PRAGMA cache_size=2")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("PRAGMA user_version = 7")
    conn.execute("CREATE TABLE spill (b BLOB)")
    for _ in range(400):
        conn.execute("INSERT INTO spill VALUES (randomblob(3000))")
    keep.append(conn)
else:
    raise SystemExit(f"unknown mode {mode}")
print("READY", flush=True)
time.sleep(600)
"""


def _kill_writer_mid_flight(mode: str, db: Path) -> None:
    """Run the child writer in ``mode`` against ``db``, SIGKILL it once it
    reports READY, and reap it. Signals only the child this call started."""
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_WRITER, mode, str(db)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        ready = child.stdout.readline()
        assert ready.strip() == b"READY", (ready, child.stderr.read() if child.poll() else b"")
    finally:
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
        child.stdout.close()
        child.stderr.close()


def _header_user_version(db: Path) -> int | None:
    """The user_version in the db FILE's header (offset 60), ignoring any
    WAL — what a raw-header or ``immutable=1`` read would report."""
    header = db.read_bytes()[:64]
    return int.from_bytes(header[60:64], "big", signed=True) if len(header) == 64 else None


def _sidecar(db: Path, suffix: str) -> Path:
    return db.with_name(db.name + suffix)


def _hot_migrated_in_wal(root: Path) -> None:
    _build("v7", root)
    _kill_writer_mid_flight("migrate", _db(root))
    assert _header_user_version(_db(root)) == 7  # the migration lives only in the WAL
    assert _sidecar(_db(root), "-wal").stat().st_size > 0


def _hot_fresh_v9(root: Path) -> None:
    _coherence_dir(root)
    _kill_writer_mid_flight("fresh_v9", _db(root))
    assert _header_user_version(_db(root)) in (0, None)  # the db file reads "fresh"
    assert _sidecar(_db(root), "-wal").stat().st_size > 0


def _hot_uv11_in_wal(root: Path) -> None:
    _build("v10", root)
    _kill_writer_mid_flight("uv11", _db(root))
    assert _header_user_version(_db(root)) == SCHEMA_USER_VERSION
    assert _sidecar(_db(root), "-wal").stat().st_size > 0


def _hot_journal_on_v1(root: Path) -> None:
    _build("raw_v1", root)
    _kill_writer_mid_flight("spill", _db(root))
    assert _sidecar(_db(root), "-journal").stat().st_size > 0


# A SIGKILLed child writer per case: run with -n 0 (tests/conftest.py).
@pytest.mark.serial
@pytest.mark.parametrize(
    ("build", "outcome", "found"),
    [
        (_hot_migrated_in_wal, _CURRENT, SCHEMA_USER_VERSION),
        (_hot_fresh_v9, _WILL_MIGRATE, 9),
        (_hot_uv11_in_wal, _NEWER, 11),
        (_hot_journal_on_v1, _WILL_MIGRATE, 1),
    ],
    ids=["v7-migrated-only-in-wal", "fresh-v9-create", "uv11-only-in-wal", "hot-journal-v1"],
)
def test_check_store_reads_through_a_hot_wal_or_journal_like_the_spawn(
    build, outcome: str, found: int,
    parent: Path, check_tmp: Path, tmp_path: Path,
) -> None:
    """A store whose last writer was SIGKILLed holds its truth in a hot
    ``-wal`` or ``-journal``, not in the db file's header. The check copies
    the sidecars with the db, so it answers what the spawn will see: a v7
    store already migrated in its WAL is current; an older build's fresh
    create (header 0, WAL 9) will migrate; a WAL-only ``user_version`` 11 is
    newer; a hot journal rolls back to v1, which will migrate. Prevents the
    exact false answer #345 is about — "fresh, nothing to migrate" for a
    store the spawn migrates — and the check replaying the original's WAL or
    journal in place."""
    root = parent / "ws"
    build(root)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.found_version) == (outcome, found)
    assert _spawn_truth(root, tmp_path / "truth") == _truth(outcome, found)


def _store_elsewhere(parent: Path, name: str, shape: str) -> Path:
    """``parent/elsewhere/<name>``, laid down as ``shape`` (or left for a
    writer to create when ``shape`` is ``absent``)."""
    target = parent / "elsewhere" / name
    target.parent.mkdir(exist_ok=True)
    if shape != "absent":
        built = parent / "built"
        _build(shape, built)
        _db(built).rename(target)
        shutil.rmtree(built)
    return target


def _point_store_at(root: Path, name: str) -> None:
    """Make ``root``'s ``state.db`` a RELATIVE symlink to
    ``parent/elsewhere/<name>``: it still resolves inside a twin of the
    whole parent."""
    _db(root).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _db(root).unlink(missing_ok=True)
    _db(root).symlink_to(Path("..", "..", "elsewhere", name))


def _link_store_elsewhere(parent: Path, root: Path, shape: str) -> Path:
    target = _store_elsewhere(parent, "real.db", shape)
    _point_store_at(root, "real.db")
    return target


def _hot_v7_migrated_in_target_wal(parent: Path, root: Path) -> Path:
    target = _link_store_elsewhere(parent, root, "v7")
    _kill_writer_mid_flight("migrate", _db(root))
    assert _header_user_version(target) == 7  # the migration lives only in the WAL
    return target


def _hot_fresh_v9_in_target_wal(parent: Path, root: Path) -> Path:
    target = _link_store_elsewhere(parent, root, "absent")
    _kill_writer_mid_flight("fresh_v9", _db(root))
    assert _header_user_version(target) in (0, None)  # the db file reads "fresh"
    return target


# A SIGKILLed child writer per case: run with -n 0 (tests/conftest.py).
@pytest.mark.serial
@pytest.mark.parametrize(
    ("build", "outcome", "found"),
    [
        (_hot_v7_migrated_in_target_wal, _CURRENT, SCHEMA_USER_VERSION),
        (_hot_fresh_v9_in_target_wal, _WILL_MIGRATE, 9),
    ],
    ids=["v7-migrated-only-in-wal", "fresh-v9-create"],
)
def test_a_symlinked_store_is_read_with_the_sidecars_beside_its_target(
    build, outcome: str, found: int,
    parent: Path, check_tmp: Path, tmp_path: Path,
) -> None:
    """SQLite resolves a symlinked ``state.db`` and keeps its ``-wal`` and
    ``-journal`` beside the TARGET; the registry's own pre-create leaves only
    empty ones beside the link. The check reads the sidecars SQLite reads, so
    a symlinked store with a hot WAL gets the answer a start produces.
    Prevents "fresh, nothing to migrate" for a symlinked store a start
    migrates, and "will migrate" for one a start opens unchanged."""
    root = parent / "ws"
    target = build(parent, root)
    assert _sidecar(target, "-wal").stat().st_size > 0
    link_wal = _sidecar(_db(root), "-wal")
    assert not link_wal.exists() or link_wal.stat().st_size == 0
    # The truth from a twin of the WHOLE parent, so the relative link
    # resolves inside the twin and the start never touches the original.
    shutil.copytree(parent, tmp_path / "twin", symlinks=True)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.found_version) == (outcome, found)
    assert _start_outcome(tmp_path / "twin" / "ws") == _truth(outcome, found)


def test_a_store_link_retargeted_during_the_copy_is_copied_again(
    parent: Path, check_tmp: Path, copies: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``state.db`` symlink pointed at another store while the check copies
    the old target leaves a copy of a store a start no longer reads, though
    the old target itself never changed; the check copies again and answers
    for the store the link names now. Prevents CURRENT from the old v10
    target for a v7 store a start migrates."""
    root = parent / "ws"
    _link_store_elsewhere(parent, root, "v10")
    _store_elsewhere(parent, "other.db", "v7")
    real_copy = store_check._copy_store

    def copy_then_retarget(*args, **kwargs):
        copied = real_copy(*args, **kwargs)
        if len(copies) == 1:
            _point_store_at(root, "other.db")
        return copied

    monkeypatch.setattr(store_check, "_copy_store", copy_then_retarget)

    result = check_store(root)

    assert (result.outcome.value, result.found_version) == (_WILL_MIGRATE, 7)
    assert len(copies) == 2
    assert list(check_tmp.iterdir()) == []


# ----------------------------------------------------------------------
# Live coordinator and lock probe
# ----------------------------------------------------------------------


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=10,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
        spawn_self_probe_attempts=20,
    )


# Starts a real coordinator: run with -n 0 (tests/conftest.py).
@pytest.mark.serial
def test_a_running_coordinator_reads_live_and_its_store_current_once_stopped(
    parent: Path, check_tmp: Path, copies: list[str], fast_cfg: LifecycleConfig
) -> None:
    """While a Python coordinator holds the workspace lock, a start joins it
    and migrates nothing, so the answer is LIVE — even on a v7 store whose
    db header still reads 7 — and the check copies nothing. Once the
    coordinator stops, the store it migrated reads CURRENT. Prevents a host
    asking for consent to a migration this build's running coordinator
    already did, and the check copying a store a live writer is changing.
    LIVE claims no schema: an older build's coordinator would keep the store
    at 7, and the check would read WILL_MIGRATE only once it stopped."""
    root = parent / "ws"
    _build("v7", root)
    assert ensure_coordinator(root, config=fast_cfg) > 0
    try:
        live = check_store(root)
    finally:
        stopped = stop_coordinator(root)

    assert stopped
    assert (live.outcome.value, live.reason, live.found_version) == (
        _LIVE, _COORDINATOR_RUNNING, None,
    )
    assert copies == []
    after = check_store(root)
    assert (after.outcome.value, after.found_version) == (_CURRENT, SCHEMA_USER_VERSION)
    assert list(check_tmp.iterdir()) == []


def _pid_file(root: Path) -> Path:
    pid_file = _coherence_dir(root) / "server.pid"
    os.close(os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o600))
    return pid_file


def test_a_shared_lock_another_check_holds_is_not_a_live_coordinator(
    parent: Path, check_tmp: Path
) -> None:
    """The probe asks for a SHARED lock, so a reader holding one — another
    check mid-copy — does not read as a coordinator. Prevents two concurrent
    checks each answering LIVE because of the other."""
    root = parent / "ws"
    _build("v10", root)
    holder = os.open(_pid_file(root), os.O_RDONLY)
    try:
        fcntl.flock(holder, fcntl.LOCK_SH | fcntl.LOCK_NB)
        result = _assert_untouched_check(root, parent, check_tmp)
    finally:
        os.close(holder)

    assert result.outcome.value == _CURRENT


def test_a_coordinator_that_binds_while_the_check_copies_reads_live(
    parent: Path, check_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lock is probed again after the copy: a coordinator that took the
    workspace lock while the check copied makes the answer LIVE, not an
    answer read from a copy that writer may already have changed. Prevents a
    WILL_MIGRATE for a store a just-started coordinator is migrating."""
    root = parent / "ws"
    _build("v7", root)
    pid_file = _pid_file(root)
    holder = os.open(pid_file, os.O_RDWR)
    real_copy = store_check._copy_store

    def copy_then_bind(*args, **kwargs):
        copied = real_copy(*args, **kwargs)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)  # as a winning spawn does
        return copied

    monkeypatch.setattr(store_check, "_copy_store", copy_then_bind)
    try:
        result = check_store(root)
    finally:
        os.close(holder)

    assert (result.outcome.value, result.reason) == (_LIVE, _COORDINATOR_RUNNING)
    assert list(check_tmp.iterdir()) == []


# ----------------------------------------------------------------------
# Cannot tell
# ----------------------------------------------------------------------


def _write_to_source_after_copy(monkeypatch: pytest.MonkeyPatch, root: Path, *, times: int) -> list[int]:
    """Make the first ``times`` copies race a real write to the source store,
    landing between the check's two stats. Returns the copy-call log."""
    calls: list[int] = []
    real_copy = store_check._copy_store

    def copy_then_write(*args, **kwargs):
        copied = real_copy(*args, **kwargs)
        calls.append(len(calls))
        if len(calls) <= times:
            _sql(
                _db(root),
                "INSERT OR REPLACE INTO registry_meta (key, value) "
                f"VALUES ('test_concurrent_write', '{len(calls)}')",
            )
        return copied

    monkeypatch.setattr(store_check, "_copy_store", copy_then_write)
    return calls


def test_a_store_that_keeps_changing_while_copied_reads_cannot_tell(
    parent: Path, check_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that lands between the stat before the copy and the stat after
    it makes that copy untrustworthy; after three such attempts the check
    says it cannot tell, and removes every copy. Prevents an answer read from
    a torn copy of a store some unlocked writer (a Node coordinator, a
    script) is changing."""
    root = parent / "ws"
    _build("v10", root)
    calls = _write_to_source_after_copy(monkeypatch, root, times=3)

    result = check_store(root)

    assert (result.outcome.value, result.reason, result.found_version) == (
        _CANNOT_TELL, _STORE_CHANGED, None,
    )
    assert len(calls) == 3
    assert list(check_tmp.iterdir()) == []


def test_a_store_that_changes_during_one_copy_only_is_copied_again_and_answered(
    parent: Path, check_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One torn copy is retried, not reported: the second copy, taken with no
    write in between, answers. Prevents a single unlucky write turning every
    check into "cannot tell"."""
    root = parent / "ws"
    _build("v10", root)
    calls = _write_to_source_after_copy(monkeypatch, root, times=1)

    result = check_store(root)

    assert result.outcome.value == _CURRENT
    assert len(calls) == 2
    assert list(check_tmp.iterdir()) == []


@pytest.mark.parametrize("store", ["absent", "zero_byte"])
@pytest.mark.parametrize("suffix", ["-wal", "-journal"])
def test_a_sidecar_with_no_store_beside_it_reads_cannot_tell(
    suffix: str, store: str, parent: Path, check_tmp: Path
) -> None:
    """A non-empty ``-wal`` or ``-journal`` beside no ``state.db``, or beside
    an empty one, is an orphan: SQLite deletes a sidecar of a database with
    no pages, unread, so a start discards it and creates an empty store. The
    check cannot call that a plain create, and answers both shapes alike.
    Prevents FRESH for a start that silently discards a sidecar that may hold
    a removed store's only copy."""
    root = parent / "ws"
    _build(store, root)
    (_coherence_dir(root) / f"state.db{suffix}").write_bytes(b"\x00" * 4096)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.reason, result.found_version) == (
        _CANNOT_TELL, _ORPHAN_SIDECAR, None,
    )


def test_an_empty_wal_with_no_store_beside_it_is_still_fresh(
    parent: Path, check_tmp: Path
) -> None:
    """A zero-byte ``-wal`` (what a read-only open leaves behind) holds no
    frames, so it is no orphan and the start is a plain create. Prevents a
    harmless leftover blocking every fresh start."""
    root = parent / "ws"
    (_coherence_dir(root) / "state.db-wal").write_bytes(b"")

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.found_version) == (_FRESH, None)


def test_a_platform_without_fcntl_reads_cannot_tell(
    parent: Path, check_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``fcntl`` the check cannot see the workspace lock, and the
    spawn does not run at all. Prevents an answer about a start that cannot
    happen on this platform."""
    root = parent / "ws"
    _build("v7", root)
    monkeypatch.setattr(store_check, "_FCNTL_AVAILABLE", False)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.reason) == (_CANNOT_TELL, _PLATFORM_UNSUPPORTED)


def test_a_pid_file_the_probe_cannot_open_reads_cannot_tell(
    parent: Path, check_tmp: Path
) -> None:
    """A ``server.pid`` the probe cannot open (here a symlink, which it
    refuses to follow) leaves the lock unknown, so the check cannot rule a
    coordinator out. Prevents an answer that ignores a coordinator it could
    not see."""
    root = parent / "ws"
    _build("v10", root)
    target = parent / "elsewhere.pid"
    target.write_text("")
    (root / ".coherence" / "server.pid").symlink_to(target)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.reason) == (_CANNOT_TELL, _LOCK_PROBE_FAILED)


@pytest.mark.parametrize("code", [errno.ENOLCK, errno.EOPNOTSUPP], ids=["enolck", "eopnotsupp"])
def test_a_lock_probe_that_fails_without_contention_reads_cannot_tell(
    code: int, parent: Path, check_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A flock that fails for a reason other than contention (a filesystem
    without flock, no lock records left) leaves the lock unknown. On a v7
    store, so a probe failure read as "held" shows up as LIVE instead of
    WILL_MIGRATE. Prevents LIVE ("a start joins it and migrates nothing")
    for a store a start would in fact migrate without consent."""
    root = parent / "ws"
    _build("v7", root)
    _pid_file(root)

    def failing_flock(fd: int, operation: int) -> None:
        raise OSError(code, os.strerror(code))

    monkeypatch.setattr(store_check.fcntl, "flock", failing_flock)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.reason) == (_CANNOT_TELL, _LOCK_PROBE_FAILED)


#: The check in a child process, so a check that blocks fails its test at a
#: deadline instead of hanging the run.
_CHILD_CHECK = r"""
import sys
from pathlib import Path

from ccs.adapters.claude_code.store_check import check_store

result = check_store(Path(sys.argv[1]))
print(result.outcome.value, result.reason, flush=True)
"""

#: Seconds a child check may take: an interpreter start and a copy of a
#: small store, with room for a loaded machine. A check that blocks on open
#: never finishes, so any finite deadline tells the two apart.
_CHILD_CHECK_DEADLINE_SEC = 30


def _check_in_child(root: Path, parent: Path, check_tmp: Path) -> tuple[str, str]:
    """Run the check on ``root`` in a child whose temp dir is ``check_tmp``,
    assert it changed nothing in ``parent`` or the temp dir, and return its
    (outcome, reason). Kills only the child this call started."""
    before = _tree(parent)
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_CHECK, str(root)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path), "TMPDIR": str(check_tmp)},
    )
    try:
        out, err = child.communicate(timeout=_CHILD_CHECK_DEADLINE_SEC)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        pytest.fail(f"the check blocked for {_CHILD_CHECK_DEADLINE_SEC}s")
    assert child.returncode == 0, err
    assert _tree(parent) == before
    assert list(check_tmp.iterdir()) == []
    outcome, reason = out.split()
    return outcome, reason


# A child process per case: run with -n 0 (tests/conftest.py).
@pytest.mark.serial
@pytest.mark.parametrize(
    ("fifo", "reason"),
    [("state.db", _STORE_UNREADABLE), ("server.pid", _LOCK_PROBE_FAILED)],
    ids=["store", "pid-file"],
)
def test_a_fifo_in_place_of_the_store_or_pid_file_reads_cannot_tell_at_once(
    fifo: str, reason: str, parent: Path, check_tmp: Path
) -> None:
    """A FIFO at ``state.db`` or ``server.pid`` has no writer, so a plain
    open of it blocks forever; the check opens without blocking and refuses
    anything that is not a regular file. Prevents a pre-flight check that
    leaves a host waiting forever (a start fails at once on such a pid
    file)."""
    root = parent / "ws"
    if fifo == "server.pid":
        _build("v10", root)
    os.mkfifo(_coherence_dir(root) / fifo, 0o600)

    assert _check_in_child(root, parent, check_tmp) == (_CANNOT_TELL, reason)


def test_a_store_linked_to_a_device_reads_cannot_tell(
    parent: Path, check_tmp: Path
) -> None:
    """A ``state.db`` that is a symlink to a device is no store: a git
    checkout can carry one, and copying ``/dev/zero`` would never end and
    fill the temp disk. ``/dev/null`` stands in for it here, so a regression
    answers wrongly instead of filling the disk. Prevents a pre-flight check
    that copies a device, or answers about one as if it were a store."""
    root = parent / "ws"
    (_coherence_dir(root) / "state.db").symlink_to(os.devnull)

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.reason) == (_CANNOT_TELL, _STORE_UNREADABLE)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_a_store_the_check_cannot_read_reads_cannot_tell(
    parent: Path, check_tmp: Path
) -> None:
    """A store this user cannot read cannot be copied; that says nothing
    about what the coordinator's own user would see. Prevents UNUSABLE for a
    store that may be fine."""
    root = parent / "ws"
    _build("v10", root)
    before = _tree(parent)
    _db(root).chmod(0o000)
    try:
        result = check_store(root)
    finally:
        _db(root).chmod(0o600)

    assert (result.outcome.value, result.reason) == (_CANNOT_TELL, _STORE_UNREADABLE)
    assert _tree(parent) == before
    assert list(check_tmp.iterdir()) == []


def test_a_temp_dir_the_check_cannot_use_reads_cannot_tell(
    parent: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure on the check's own side (no usable temp dir) is the check's
    problem, not the store's. Prevents UNUSABLE for a store never read."""
    root = parent / "ws"
    _build("v10", root)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))

    result = check_store(root)

    assert (result.outcome.value, result.reason) == (_CANNOT_TELL, _CHECK_ENVIRONMENT_FAILED)


def _raising(error: BaseException):
    def open_registry(*args, **kwargs):
        raise error
    return open_registry


def _sqlite_error(name: str | None) -> sqlite3.OperationalError:
    """An OperationalError carrying the code SQLite would attach to it (none
    for ``None``: an error SQLite did not raise itself)."""
    error = sqlite3.OperationalError("injected")
    if name is not None:
        error.sqlite_errorcode = getattr(sqlite3, name)
        error.sqlite_errorname = name
    return error


@pytest.mark.parametrize(
    ("error", "outcome", "reason"),
    [
        (_sqlite_error("SQLITE_FULL"), _CANNOT_TELL, _CHECK_ENVIRONMENT_FAILED),
        (_sqlite_error("SQLITE_IOERR"), _CANNOT_TELL, _CHECK_ENVIRONMENT_FAILED),
        (_sqlite_error("SQLITE_IOERR_SHORT_READ"), _CANNOT_TELL, _CHECK_ENVIRONMENT_FAILED),
        (_sqlite_error("SQLITE_CANTOPEN"), _CANNOT_TELL, _CHECK_ENVIRONMENT_FAILED),
        (OSError(errno.ENOSPC, "no space"), _CANNOT_TELL, _CHECK_ENVIRONMENT_FAILED),
        (_sqlite_error("SQLITE_CORRUPT"), _UNUSABLE, "sqlite_corrupt"),
        (_sqlite_error("SQLITE_CORRUPT_INDEX"), _UNUSABLE, "sqlite_corrupt_index"),
        (_sqlite_error(None), _UNUSABLE, "sqlite_error"),
    ],
    ids=["full", "ioerr", "ioerr-extended", "cantopen", "oserror", "corrupt",
         "corrupt-extended", "no-code"],
)
def test_a_dry_run_failure_is_the_environments_or_the_stores_by_its_sqlite_code(
    error: BaseException, outcome: str, reason: str,
    parent: Path, check_tmp: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dry run that fails for a reason of the check's own (a full or
    failing temp disk, a copy it cannot open — judged by the PRIMARY code, so
    an extended I/O code counts) cannot tell; one that fails on the store's
    content (corruption) is UNUSABLE under its full code name. Prevents a
    full temp disk reporting the workspace store as broken."""
    root = parent / "ws"
    _build("v10", root)
    monkeypatch.setattr(sqlite_registry, "SqliteArtifactRegistry", _raising(error))

    result = _assert_untouched_check(root, parent, check_tmp)

    assert (result.outcome.value, result.reason) == (outcome, reason)


def test_an_unexpected_dry_run_error_propagates(
    parent: Path, check_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error the check does not classify is a bug, so it propagates — and
    the copy is still removed. Prevents a defect reading as a verdict about
    the store."""
    root = parent / "ws"
    _build("v10", root)
    monkeypatch.setattr(sqlite_registry, "SqliteArtifactRegistry", _raising(ValueError("bug")))

    with pytest.raises(ValueError, match="bug"):
        check_store(root)

    assert list(check_tmp.iterdir()) == []


def test_the_result_is_frozen() -> None:
    """A result is a value a host can pass around without it changing under
    it. Prevents a caller mutating the outcome another caller reads."""
    result = StoreCheck(
        outcome=StoreCheckOutcome.CURRENT, db_path=Path("x"), found_version=10,
        expected_version=10, reason=None, message="m",
    )
    with pytest.raises(AttributeError):
        result.outcome = StoreCheckOutcome.LIVE  # type: ignore[misc]
