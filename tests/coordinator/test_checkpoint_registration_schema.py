# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Registry schema v10 — the checkpoint receiver and registration claim (#191).

The v9 -> v10 step adds two nullable columns to ``workspace_checkpoints``
(``receiver``, ``registered_by``). v9 is the grant handoff's
``transfer_records`` table (#185), so a store a v9 build wrote carries that
table and neither column. Pinned here, in the house migration shape:

- a fresh db lands both columns at the current stamp;
- a v9 db (the transfer table, no #191 columns) is NOT refused as foreign: it
  migrates in place and keeps its manifests, which come back with no receiver
  and no claim (the pre-#191 behaviour) and can then be claimed;
- the re-stamp trap, sixth arming: a v8-origin walk lands the v9 table AND the
  v10 columns at the v10 stamp (fails if ``_migrate_v8_to_v9`` stamps the
  constant again), and a v7-origin walk lands all three steps;
- a failure before the v10 stamp leaves a bootable v9;
- a v10 stamp without either column is refused as foreign-or-corrupt.
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


def _tables(db: Path) -> set[str]:
    conn = sqlite3.connect(str(db))
    try:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def _revert_to_v9_shape(db: Path) -> None:
    """A current db rewound to what a v9 build produced: the transfer table
    (#185) and no #191 columns."""
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("ALTER TABLE workspace_checkpoints DROP COLUMN receiver")
        conn.execute("ALTER TABLE workspace_checkpoints DROP COLUMN registered_by")
        conn.execute("PRAGMA user_version = 9")
        conn.commit()
    finally:
        conn.close()


def _revert_to_v8_shape(db: Path) -> None:
    """Rewound one step further: the v9 transfer table absent too."""
    _revert_to_v9_shape(db)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("DROP TABLE IF EXISTS transfer_records")
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


def test_fresh_db_has_the_columns_at_the_v10_stamp(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    with SqliteArtifactRegistry(db):
        pass
    assert SCHEMA_USER_VERSION == 10
    assert _user_version(db) == 10
    assert {"receiver", "registered_by"} <= _checkpoint_columns(db)


def test_a_v9_store_migrates_and_its_manifests_are_unclaimed(tmp_path: Path) -> None:
    """A store a v9 build wrote has the transfer table and neither #191
    column. It is a genuine Python store one step behind, so the open must
    migrate it, not refuse it as foreign: a probe that read v9 as "must carry
    the columns" would leave every coordinator and CLI that ran a v9 build
    unable to start."""
    db = tmp_path / "v9.db"
    with SqliteArtifactRegistry(db) as reg:
        checkpoint_id = _mint(reg)
    _revert_to_v9_shape(db)
    assert _user_version(db) == 9
    assert "transfer_records" in _tables(db)
    assert not {"receiver", "registered_by"} & _checkpoint_columns(db)

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
    assert _user_version(db) == 10
    assert "transfer_records" in _tables(db)


def test_v8_origin_walk_lands_table_and_columns_at_v10(tmp_path: Path) -> None:
    """THE RE-STAMP TRAP, sixth arming: ``_migrate_v8_to_v9`` must stamp its
    own literal 9, or a v8-origin db is stamped 10 without the v10 columns and
    the chained v9->v10 loser-guard no-ops."""
    db = tmp_path / "v8.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v8_shape(db)
    assert _user_version(db) == 8

    with SqliteArtifactRegistry(db):
        pass

    assert _user_version(db) == 10
    assert "transfer_records" in _tables(db)
    assert {"receiver", "registered_by"} <= _checkpoint_columns(db)


def test_v7_origin_walk_lands_both_tables_and_columns_at_v10(tmp_path: Path) -> None:
    """The fifth arming still holds under the sixth: ``_migrate_v7_to_v8``
    stamps its own literal 8, so a v7-origin walk runs all three steps."""
    db = tmp_path / "v7.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v7_shape(db)
    assert _user_version(db) == 7

    with SqliteArtifactRegistry(db):
        pass

    assert _user_version(db) == 10
    assert {"caller_principals", "transfer_records"} <= _tables(db)
    assert {"receiver", "registered_by"} <= _checkpoint_columns(db)


def test_a_crash_before_the_v10_stamp_leaves_a_bootable_v9(tmp_path: Path) -> None:
    db = tmp_path / "crash.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v9_shape(db)

    class _Crash(RuntimeError):
        pass

    class _CrashingConn:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql: str, *args):
            if sql.strip() == "PRAGMA user_version = 10":
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
            reg._migrate_v9_to_v10(None)  # noqa: SLF001
    finally:
        real.close()

    assert _user_version(db) == 9
    assert "registered_by" not in _checkpoint_columns(db)
    with SqliteArtifactRegistry(db):
        pass
    assert _user_version(db) == 10
    assert {"receiver", "registered_by"} <= _checkpoint_columns(db)


@pytest.mark.parametrize("column", ["receiver", "registered_by"])
def test_a_v10_stamp_without_a_column_is_refused(tmp_path: Path, column: str) -> None:
    """Each column is probed on its own: a v10 store missing either one is
    foreign-or-corrupt, with a control that a genuine v10 opens."""
    genuine, forged = tmp_path / "genuine.db", tmp_path / "forged.db"
    for db in (genuine, forged):
        with SqliteArtifactRegistry(db):
            pass
    conn = sqlite3.connect(str(forged))
    try:
        conn.execute(f"ALTER TABLE workspace_checkpoints DROP COLUMN {column}")
        conn.commit()
    finally:
        conn.close()
    assert _user_version(forged) == 10

    with SqliteArtifactRegistry(genuine):
        pass
    with pytest.raises(CrossRuntimeSchemaError, match="receiver and registered_by"):
        SqliteArtifactRegistry(forged)
