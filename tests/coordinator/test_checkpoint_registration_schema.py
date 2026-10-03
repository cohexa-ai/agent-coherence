# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Registry schema v9 — the checkpoint receiver and registration claim (#191).

The v8 -> v9 step adds two nullable columns to ``workspace_checkpoints``
(``receiver``, ``registered_by``). Pinned here, in the house migration shape:

- a fresh db lands both columns at the current stamp;
- a v8 db migrates in place and keeps its manifests, which come back with no
  receiver and no claim (the pre-#191 behaviour) and can then be claimed;
- the re-stamp trap, fifth arming: a v7-origin walk lands the v8 table AND the
  v9 columns at the v9 stamp (fails if ``_migrate_v7_to_v8`` stamps the
  constant again);
- a failure before the v9 stamp leaves a bootable v8;
- a v9 stamp without the columns is refused as foreign-or-corrupt.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.coordinator.registry_protocol import CheckpointMember
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import (
    SCHEMA_USER_VERSION,
    CrossRuntimeSchemaError,
    SqliteArtifactRegistry,
)
from ccs.core.substrate import sha256_hex
from ccs.core.types import WorkspaceRestoreWrite

FP = sha256_hex(b"captured")


def _user_version(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _checkpoint_columns(db: Path) -> set[str]:
    conn = sqlite3.connect(str(db))
    try:
        return {
            row[1]
            for row in conn.execute("PRAGMA table_info(workspace_checkpoints)")
        }
    finally:
        conn.close()


def _revert_to_v8_shape(db: Path) -> None:
    """A current db rewound to what a v8 build produced: no #191 columns."""
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("ALTER TABLE workspace_checkpoints DROP COLUMN receiver")
        conn.execute("ALTER TABLE workspace_checkpoints DROP COLUMN registered_by")
        conn.execute("PRAGMA user_version = 8")
        conn.commit()
    finally:
        conn.close()


def _revert_to_v7_shape(db: Path) -> None:
    _revert_to_v8_shape(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("DROP TABLE IF EXISTS caller_principals")
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    finally:
        conn.close()


def _mint(reg: SqliteArtifactRegistry) -> str:
    return CoordinatorService(reg).create_workspace_checkpoint(
        name="pre-upgrade",
        owner=uuid4(),
        members=[
            CheckpointMember(
                member_path="notes/plan.md",
                artifact_id=None,
                native_token="7",
                fingerprint=FP,
                captured_at=1.0,
            )
        ],
        window_min=1.0,
        window_max=1.0,
    ).checkpoint_id


def test_fresh_db_has_the_columns_at_the_v9_stamp(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    with SqliteArtifactRegistry(db):
        pass
    assert SCHEMA_USER_VERSION == 9
    assert _user_version(db) == 9
    assert {"receiver", "registered_by"} <= _checkpoint_columns(db)


def test_v8_db_migrates_and_its_manifests_are_unclaimed(tmp_path: Path) -> None:
    db = tmp_path / "v8.db"
    with SqliteArtifactRegistry(db) as reg:
        checkpoint_id = _mint(reg)
    _revert_to_v8_shape(db)
    assert _user_version(db) == 8
    assert "registered_by" not in _checkpoint_columns(db)

    with SqliteArtifactRegistry(db) as reg:
        record = reg.get_checkpoint(checkpoint_id)
        assert record is not None and record.name == "pre-upgrade"
        assert record.receiver is None and record.registered_by is None
        controller = uuid4()
        result = CoordinatorService(reg).register_workspace_restore(
            checkpoint_id=checkpoint_id,
            controller=controller,
            writes=(WorkspaceRestoreWrite("notes/plan.md", FP),),
        )
        assert result.retry_of_own_registration is False
        assert reg.get_checkpoint(checkpoint_id).registered_by == controller
    assert _user_version(db) == 9


def test_v7_origin_walk_lands_table_and_columns_at_v9(tmp_path: Path) -> None:
    """THE RE-STAMP TRAP, fifth arming: ``_migrate_v7_to_v8`` must stamp its
    own literal 8, or a v7-origin db is stamped 9 without the v9 columns and
    the chained v8->v9 loser-guard no-ops."""
    db = tmp_path / "v7.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v7_shape(db)
    assert _user_version(db) == 7

    with SqliteArtifactRegistry(db):
        pass

    assert _user_version(db) == 9
    assert {"receiver", "registered_by"} <= _checkpoint_columns(db)


def test_a_crash_before_the_v9_stamp_leaves_a_bootable_v8(tmp_path: Path) -> None:
    db = tmp_path / "crash.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v8_shape(db)

    class _Crash(RuntimeError):
        pass

    class _CrashingConn:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql: str, *args):
            if sql.strip() == f"PRAGMA user_version = {SCHEMA_USER_VERSION}":
                raise _Crash("simulated kill before the stamp")
            return self._inner.execute(sql, *args)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    reg = SqliteArtifactRegistry.__new__(SqliteArtifactRegistry)
    real = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    try:
        reg._conn = _CrashingConn(real)  # noqa: SLF001 — the seam under test
        reg._db_path = db  # noqa: SLF001
        with pytest.raises(_Crash):
            reg._migrate_v8_to_v9(None)  # noqa: SLF001
    finally:
        real.close()

    assert _user_version(db) == 8
    assert "registered_by" not in _checkpoint_columns(db)
    with SqliteArtifactRegistry(db):
        pass
    assert _user_version(db) == 9
    assert {"receiver", "registered_by"} <= _checkpoint_columns(db)


def test_a_v9_stamp_without_the_columns_is_refused(tmp_path: Path) -> None:
    genuine, forged = tmp_path / "genuine.db", tmp_path / "forged.db"
    for db in (genuine, forged):
        with SqliteArtifactRegistry(db):
            pass
    conn = sqlite3.connect(str(forged))
    try:
        conn.execute("ALTER TABLE workspace_checkpoints DROP COLUMN registered_by")
        conn.commit()
    finally:
        conn.close()
    assert _user_version(forged) == 9

    with SqliteArtifactRegistry(genuine):
        pass
    with pytest.raises(CrossRuntimeSchemaError, match="registered_by"):
        SqliteArtifactRegistry(forged)
