# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Registry v6 — the ``last_observed_version`` comparand (SB-10, R6/R7, KTD3/KTD9).

SB-10 plan (docs/plans/2026-08-24-1750-feat-sb10-compaction-reemission-plan.md),
durable-comparand unit. After a Claude Code compaction a session must re-learn
which cached tracked-artifact views are stale; the comparand is the artifact
version whose BYTES an agent last observed, recorded per (agent, artifact)
atomically with every non-INVALID ``agent_states`` upsert. Covers, per the
unit's test scenarios:

- migration: a fresh db lands v6 with the ``agent_states.last_observed_version``
  column; a v5 db upgrades (pre-v6 grants read NULL — never a 0-sentinel);
  upgraded and fresh ``agent_states`` shapes are IDENTICAL;
- **the v4 -> 5 -> 6 walk regression** (the re-stamp trap, KTD9): with the
  constant at 6, ``_migrate_v4_to_v5`` must guard/stamp the intermediate
  ``_V5_USER_VERSION`` literal — stamping the constant would mark a v4-origin
  db ``user_version=6`` WITHOUT the column, and the chained v5->v6 step's
  loser-guard would no-op. The walk test fails if the literal is ever reverted;
- a SIGKILL-simulation mid v5->v6 (ALTER executed, raise before the in-txn
  stamp) leaves a bootable v5 that a clean reopen re-migrates;
- the foreign-ledger guard still refuses a Node-fingerprint db
  (``agent_states.deadline_tick`` at ``user_version>=3``) and does NOT
  false-positive on a genuine Python v6;
- recording semantics on BOTH registries (the parametrized arms are the parity
  run): S/E/M upserts record the artifact's current version; a transition to
  INVALID preserves the prior recorded value; a never-observed pair stays
  NULL/absent (accessor returns None);
- the commit paths advance the WRITER's recorded value to the NEW version in
  the same transaction (``commit_cas`` / ``commit_all`` / the pessimistic
  ``service.commit``), while invalidated peers keep their pre-commit value.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.sqlite_registry import (
    SCHEMA_USER_VERSION,
    CrossRuntimeSchemaError,
    SqliteArtifactRegistry,
)
from ccs.core.states import MESIState
from ccs.core.types import Artifact, CommitAllEntry, ConflictDetail, MultiCommitResult


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


# Both-arms fixture: every recording scenario runs against the in-memory AND
# the sqlite registry — the parametrization IS the parity harness for this
# surface (the test_fencing.py precedent for per-agent bookkeeping).
@pytest.fixture(params=["in_memory", "sqlite"])
def registry(
    request: pytest.FixtureRequest, tmp_path: Path
) -> "Iterator[ArtifactRegistry | SqliteArtifactRegistry]":
    if request.param == "in_memory":
        yield ArtifactRegistry()
    else:
        reg = SqliteArtifactRegistry(tmp_path / "parity-arm.db")
        yield reg
        reg.close()


def _register(reg) -> Artifact:
    art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
    reg.register_artifact(art, content="seed-body")
    return art


# ---------------------------------------------------------------------------
# Raw-db helpers (probes + shape reverts; plain sqlite3, never the registry)
# ---------------------------------------------------------------------------


def _user_version(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()


def _agent_states_shape(db_path: Path) -> list[tuple]:
    """Full ``PRAGMA table_info(agent_states)`` rows — (cid, name, type,
    notnull, dflt_value, pk) — so the fresh-vs-upgraded comparison pins the
    column ORDER and declared types, not just the name set."""
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA table_info(agent_states)").fetchall()
    finally:
        conn.close()


def _revert_to_v5_shape(db_path: Path) -> None:
    """Mutate a current (v6) db on disk back to the v5 shape:
    ``last_observed_version`` ABSENT, stamped ``user_version=5`` (what a pre-SB-10
    build produced). DROP COLUMN needs sqlite >= 3.35 (2021) — every supported
    interpreter's bundled sqlite clears that."""
    _revert_to_v6_shape(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("ALTER TABLE agent_states DROP COLUMN last_observed_version")
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
    finally:
        conn.close()


def _revert_to_v4_shape(db_path: Path) -> None:
    """Mutate a current (v6) db on disk back to the v4 shape (the re-stamp
    trap's migration origin): v6 column + v5 workspace tables ABSENT, stamped
    ``user_version=4``."""
    _revert_to_v5_shape(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS workspace_checkpoint_members")
        conn.execute("DROP TABLE IF EXISTS workspace_checkpoints")
        conn.execute("DROP INDEX IF EXISTS idx_workspace_checkpoints_name")
        conn.execute("PRAGMA user_version = 4")
        conn.commit()
    finally:
        conn.close()


def _revert_to_v6_shape(db_path: Path) -> None:
    """Mutate a current (v8) db on disk back to the v6 shape: the
    ``idx_agent_states_agent`` index and the v8 ``caller_principals`` table
    ABSENT, stamped ``user_version=6`` (what an SB-10-era build produced
    before the index migration)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS caller_principals")
        conn.execute("DROP INDEX IF EXISTS idx_agent_states_agent")
        conn.execute("PRAGMA user_version = 6")
        conn.commit()
    finally:
        conn.close()


def _indexes(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
            ).fetchall()
        }
    finally:
        conn.close()


def _forge_node_v3_db(db_path: Path) -> None:
    """Forge a sibling-Node-ledger v3 db: the shared v1 DDL + the Node ledger's
    v3 marker (``agent_states.deadline_tick``) + ``user_version=3``. Compact
    mirror of the fuller forge in ``tests/test_sqlite_registry.py``."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "CREATE TABLE artifacts (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
            "version INTEGER NOT NULL, content_hash TEXT NOT NULL, size_tokens INTEGER, "
            "last_writer_id TEXT, updated_at REAL NOT NULL)"
        )
        conn.execute(
            "CREATE TABLE agent_states (artifact_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
            "state TEXT NOT NULL, PRIMARY KEY (artifact_id, agent_id))"
        )
        conn.execute("CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute("ALTER TABLE agent_states ADD COLUMN deadline_tick INTEGER")
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Migration — fresh init, v5 upgrade, the re-stamp trap, crash atomicity
# ---------------------------------------------------------------------------


def test_fresh_v6_init_has_column(db_path: Path) -> None:
    """A fresh db is created at v6 directly: ``last_observed_version`` inline in
    the ``agent_states`` DDL, ``user_version=6`` — no migration shim ever runs."""
    with SqliteArtifactRegistry(db_path) as reg:
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
    assert SCHEMA_USER_VERSION == 9
    cols = {row[1] for row in _agent_states_shape(db_path)}
    assert "last_observed_version" in cols


def test_v5_db_upgrades_to_v6_and_prior_rows_read_null(db_path: Path) -> None:
    """A v5 db migrates on open; a grant recorded BEFORE v6 existed has no
    observation on file, so the accessor reports None (never a 0-sentinel) and
    the pre-migration data survives untouched."""
    agent = uuid4()
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        art = _register(reg)
        reg.set_agent_state(art.id, agent, MESIState.EXCLUSIVE, trigger="write", tick=1)
    _revert_to_v5_shape(db_path)
    assert _user_version(db_path) == 5

    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        # Pre-migration data intact.
        got = reg.get_artifact(art.id)
        assert got is not None and got.version == 1 and got.content_hash == "h"
        assert reg.get_agent_state(art.id, agent) is MESIState.EXCLUSIVE
        assert reg.get_content_at_version(art.id, 1) == "seed-body"
        # The pre-v6 grant was never observed under the new comparand: NULL.
        assert reg.last_observed_version_for(art.id, agent) is None


def test_upgraded_and_fresh_agent_states_shapes_identical(
    tmp_path: Path,
) -> None:
    """A v5->v6-migrated ``agent_states`` and a fresh-v6 one are byte-identical
    per ``PRAGMA table_info`` (names, order, types) — the ALTER appends the
    column exactly where the fresh DDL declares it."""
    fresh = tmp_path / "fresh.db"
    upgraded = tmp_path / "upgraded.db"
    with SqliteArtifactRegistry(fresh):
        pass
    with SqliteArtifactRegistry(upgraded):
        pass
    _revert_to_v5_shape(upgraded)
    with SqliteArtifactRegistry(upgraded) as reg:
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
    assert _agent_states_shape(upgraded) == _agent_states_shape(fresh)


def test_v4_to_v5_to_v6_walk_lands_v6_column_at_v6_stamp(db_path: Path) -> None:
    """THE RE-STAMP TRAP REGRESSION (KTD9). A v4-origin db must walk 4 -> 5 -> 6
    and end with the v5 DDL AND the v6 column present AT the v6 stamp.

    Two literal conversions are load-bearing, one per constant bump. If
    ``_migrate_v4_to_v5`` still stamped ``SCHEMA_USER_VERSION``, a v4-origin
    db would be stamped past v6 WITHOUT the column; if ``_migrate_v5_to_v6``
    still stamped the constant (correct while it was 6), a v5-origin db would
    be stamped 7 WITHOUT the index and the chained v6->v7 loser-guard would
    no-op. This test fails if EITHER the ``_V5_USER_VERSION`` or the
    ``_V6_USER_VERSION`` literal conversion is ever reverted."""
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        art = _register(reg)
    _revert_to_v4_shape(db_path)
    assert _user_version(db_path) == 4
    assert "workspace_checkpoints" not in _tables(db_path)

    agent = uuid4()
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        # The intermediate v5 surface landed too (the chain did not skip a step).
        tables = {
            r[0]
            for r in reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        idx = {
            r[0]
            for r in reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_agent_states_agent" in idx
        assert "workspace_checkpoints" in tables
        assert "workspace_checkpoint_members" in tables
        # The column is USABLE, not just present: a grant records through it.
        reg.set_agent_state(art.id, agent, MESIState.SHARED, trigger="fetch", tick=1)
        assert reg.last_observed_version_for(art.id, agent) == 1
    # And the migrated db re-opens write-free at v6 (the trap's failure mode
    # surfaced on the SECOND open).
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        assert reg.last_observed_version_for(art.id, agent) == 1


def test_sigkill_mid_v5_to_v6_migration_leaves_bootable_v5(
    db_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single-transaction proof: crash the migration AFTER its ALTER executed
    but BEFORE the in-txn stamp lands. The rollback must discard the ALTER (one
    ``BEGIN IMMEDIATE`` wraps DDL + stamp), leaving a bootable v5 db that a
    clean reopen re-migrates without erroring on a duplicate column."""
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        art = _register(reg)
    _revert_to_v5_shape(db_path)

    real_connect = sqlite3.connect
    opened: list[sqlite3.Connection] = []

    class _CrashBeforeStamp:
        """Connection proxy that simulates a SIGKILL at the v6 stamp: the ALTER
        before it has executed inside the open BEGIN IMMEDIATE."""

        def __init__(self, conn: sqlite3.Connection) -> None:
            object.__setattr__(self, "_conn", conn)

        def execute(self, sql: str, *args: object) -> object:
            if sql.strip().startswith("PRAGMA user_version = 6"):
                raise sqlite3.OperationalError("simulated SIGKILL before the v6 stamp")
            return self._conn.execute(sql, *args)

        def __getattr__(self, name: str) -> object:
            return getattr(self._conn, name)

    def crashing_connect(*args: object, **kwargs: object) -> _CrashBeforeStamp:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(conn)
        return _CrashBeforeStamp(conn)

    with monkeypatch.context() as mp:
        mp.setattr(
            "ccs.coordinator.sqlite_registry.sqlite3.connect", crashing_connect
        )
        with pytest.raises(sqlite3.OperationalError, match="simulated SIGKILL"):
            SqliteArtifactRegistry(db_path)
    for conn in opened:
        conn.close()

    # Pre-state bootable and UNCHANGED: still v5, and the ALTER that ran before
    # the crash did NOT survive (it was inside the rolled-back txn).
    assert _user_version(db_path) == 5
    assert "last_observed_version" not in {r[1] for r in _agent_states_shape(db_path)}

    # A clean reopen completes the migration (idempotent), data intact.
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        got = reg.get_artifact(art.id)
        assert got is not None and got.content_hash == "h"
        assert reg.last_observed_version_for(art.id, uuid4()) is None


# ---------------------------------------------------------------------------
# Foreign-lineage refusal (probes intact; no false positive on Python v6)
# ---------------------------------------------------------------------------


def test_node_v3_db_still_refused(db_path: Path) -> None:
    """The Node-ledger marker (``agent_states.deadline_tick`` at
    ``user_version>=3``) still refuses after the v6 bump — KTD9 obliged NO new
    probe, and the existing chain was not rearranged."""
    _forge_node_v3_db(db_path)
    with pytest.raises(CrossRuntimeSchemaError):
        SqliteArtifactRegistry(db_path)
    assert _user_version(db_path) == 3  # untouched


def test_genuine_v6_db_reopens_fine_both_paths(db_path: Path) -> None:
    """No false positive: a Python v6 db (which carries ``last_observed_version``,
    never ``deadline_tick``) passes the guard on the writer AND read-only paths."""
    with SqliteArtifactRegistry(db_path) as reg:
        _register(reg)
    with SqliteArtifactRegistry(db_path) as rw:
        assert rw._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
    with SqliteArtifactRegistry(db_path, read_only=True) as ro:
        assert ro.instance_id


# ---------------------------------------------------------------------------
# Recording semantics — BOTH registries (the parametrized arms are the parity run)
# ---------------------------------------------------------------------------


def test_parity_grant_records_current_version(registry) -> None:
    reg = registry
    art = _register(reg)
    a, b, c = uuid4(), uuid4(), uuid4()
    reg.set_agent_state(art.id, a, MESIState.SHARED, trigger="fetch", tick=1)
    assert reg.last_observed_version_for(art.id, a) == 1
    reg.set_agent_state(art.id, b, MESIState.EXCLUSIVE, trigger="write", tick=2)
    assert reg.last_observed_version_for(art.id, b) == 1
    reg.set_agent_state(art.id, b, MESIState.MODIFIED, trigger="write", tick=3)
    assert reg.last_observed_version_for(art.id, b) == 1
    # A never-observed pair stays absent — None, never a 0-sentinel.
    assert reg.last_observed_version_for(art.id, c) is None


def test_parity_invalid_transition_preserves_prior_value(registry) -> None:
    """R7: INVALID never overwrites the comparand — the stored value is the last
    version whose bytes the agent actually saw, exactly what the stale flag
    compares against after a compaction."""
    reg = registry
    art = _register(reg)
    a, b = uuid4(), uuid4()
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=9)
    assert reg.last_observed_version_for(art.id, a) == 1
    # The non-reclaim invalidation leg preserves too.
    reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=10)
    reg.set_agent_state(art.id, b, MESIState.INVALID, trigger="peer_invalidation", tick=11)
    assert reg.last_observed_version_for(art.id, b) == 1


def test_parity_invalid_on_fresh_pair_records_nothing(registry) -> None:
    """An INVALID upsert on a never-observed pair must not mint a value (the
    row/entry may exist for other bookkeeping; the comparand stays NULL)."""
    reg = registry
    art = _register(reg)
    a = uuid4()
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="peer_invalidation", tick=1)
    assert reg.last_observed_version_for(art.id, a) is None


def test_parity_refetch_after_version_move_records_new_version(registry) -> None:
    reg = registry
    art = _register(reg)
    a, writer = uuid4(), uuid4()
    reg.set_agent_state(art.id, a, MESIState.SHARED, trigger="fetch", tick=1)
    assert reg.last_observed_version_for(art.id, a) == 1
    res = reg.commit_cas(art.id, writer, expected_version=1, content_hash="h2")
    assert not isinstance(res, ConflictDetail)  # WIN -> version 2
    # The invalidated reader still shows the version it last SAW ...
    assert reg.last_observed_version_for(art.id, a) == 1
    # ... until it re-fetches the new bytes.
    reg.set_agent_state(art.id, a, MESIState.SHARED, trigger="fetch", tick=2)
    assert reg.last_observed_version_for(art.id, a) == 2


def test_parity_unobserved_grant_preserves_the_prior_value(registry) -> None:
    """``observed=False`` grants the state but certifies no read: the reader
    that last saw v1 stays at v1 while holding SHARED on v2. Both directions
    are pinned -- the default on the very same transition still records v2 --
    because asserting only the preserve side cannot tell a working flag from
    a registry that stopped recording altogether."""
    reg = registry
    art = _register(reg)
    a, writer = uuid4(), uuid4()
    reg.set_agent_state(art.id, a, MESIState.SHARED, trigger="fetch", tick=1)
    res = reg.commit_cas(art.id, writer, expected_version=1, content_hash="h2")
    assert not isinstance(res, ConflictDetail)  # WIN -> version 2, a INVALID
    assert reg.get_agent_state(art.id, a) == MESIState.INVALID

    reg.set_agent_state(
        art.id, a, MESIState.SHARED, trigger="post_stale_bash", tick=2, observed=False,
    )
    assert reg.get_agent_state(art.id, a) == MESIState.SHARED
    assert reg.last_observed_version_for(art.id, a) == 1

    reg.set_agent_state(art.id, a, MESIState.SHARED, trigger="post_stale_bash", tick=3)
    assert reg.last_observed_version_for(art.id, a) == 2


def test_parity_unobserved_grant_on_a_fresh_pair_records_nothing(registry) -> None:
    """A never-observed pair granted without a read stays never-observed --
    None, never a 0-sentinel and never the current version."""
    reg = registry
    art = _register(reg)
    a = uuid4()
    reg.set_agent_state(
        art.id, a, MESIState.SHARED, trigger="first_bash_read", tick=1, observed=False,
    )
    assert reg.get_agent_state(art.id, a) == MESIState.SHARED
    assert reg.last_observed_version_for(art.id, a) is None


def test_parity_commit_cas_advances_writer_same_transaction(registry) -> None:
    """KTD4 first layer: the WIN that bumps the version also advances the
    committing agent's comparand to the NEW version — atomically (sqlite: the
    same BEGIN IMMEDIATE; in-memory: the same GIL-atomic apply)."""
    reg = registry
    art = _register(reg)
    writer, peer = uuid4(), uuid4()
    reg.set_agent_state(art.id, writer, MESIState.SHARED, trigger="fetch", tick=1)
    reg.set_agent_state(art.id, peer, MESIState.SHARED, trigger="fetch", tick=1)
    res = reg.commit_cas(art.id, writer, expected_version=1, content_hash="h2")
    assert not isinstance(res, ConflictDetail)
    assert reg.get_artifact(art.id).version == 2
    assert reg.last_observed_version_for(art.id, writer) == 2
    # The invalidated peer keeps its pre-commit observation.
    assert reg.get_agent_state(art.id, peer) is MESIState.INVALID
    assert reg.last_observed_version_for(art.id, peer) == 1


def test_parity_commit_cas_fresh_writer_records_new_version(registry) -> None:
    """An OCC writer with NO prior row (never fetched) ends SHARED at the new
    version it just produced — the INSERT leg of the committer upsert records
    too, not only the UPDATE leg."""
    reg = registry
    art = _register(reg)
    writer = uuid4()
    res = reg.commit_cas(art.id, writer, expected_version=1, content_hash="h2")
    assert not isinstance(res, ConflictDetail)
    assert reg.last_observed_version_for(art.id, writer) == 2


def test_parity_commit_all_advances_writer_per_member(registry) -> None:
    reg = registry
    art_a = Artifact(id=uuid4(), name="a.md", version=1, content_hash="ha")
    art_b = Artifact(id=uuid4(), name="b.md", version=1, content_hash="hb")
    reg.register_artifact(art_a, content="A")
    reg.register_artifact(art_b, content="B")
    writer, peer = uuid4(), uuid4()
    reg.set_agent_state(art_b.id, peer, MESIState.SHARED, trigger="fetch", tick=1)
    res = reg.commit_all(
        writer,
        {
            art_a.id: CommitAllEntry(expected_version=1, content_hash="ha2"),
            art_b.id: CommitAllEntry(expected_version=1, content_hash="hb2"),
        },
        tick=2,
    )
    assert isinstance(res, MultiCommitResult)
    assert reg.last_observed_version_for(art_a.id, writer) == 2
    assert reg.last_observed_version_for(art_b.id, writer) == 2
    # The invalidated peer keeps its pre-commit observation on its member.
    assert reg.last_observed_version_for(art_b.id, peer) == 1


def test_parity_service_commit_advances_writer(registry) -> None:
    """The pessimistic commit path (M/E holder via ``service.commit``): the
    version bump and the committer's MODIFIED upsert land the NEW version in the
    comparand, while the invalidated peer keeps the version it last saw."""
    from ccs.coordinator.service import CoordinatorService

    reg = registry
    art = _register(reg)
    service = CoordinatorService(reg)
    writer, peer = uuid4(), uuid4()
    reg.set_agent_state(art.id, peer, MESIState.SHARED, trigger="fetch", tick=1)
    service.write(agent_id=writer, artifact_id=art.id, issued_at_tick=2)
    assert reg.last_observed_version_for(art.id, writer) == 1  # acquire at v1
    service.commit(
        agent_id=writer, artifact_id=art.id, content="new-body", issued_at_tick=3
    )
    assert reg.get_artifact(art.id).version == 2
    assert reg.last_observed_version_for(art.id, writer) == 2
    assert reg.get_agent_state(art.id, peer) is MESIState.INVALID
    assert reg.last_observed_version_for(art.id, peer) == 1

# ---------------------------------------------------------------------------
# Migration v7 — the agent_states(agent_id) index (SB-10 perf follow-up)
# ---------------------------------------------------------------------------


def test_fresh_v7_init_has_agent_states_agent_index(db_path: Path) -> None:
    """A fresh db is created at the head directly: the
    ``idx_agent_states_agent`` index inline in the fresh DDL, stamped
    ``SCHEMA_USER_VERSION`` — no migration shim."""
    with SqliteArtifactRegistry(db_path) as reg:
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
    assert "idx_agent_states_agent" in _indexes(db_path)


def test_v6_db_upgrades_to_v7_and_gains_the_index(db_path: Path) -> None:
    """A v6 db (SB-10-era: column present, index absent) migrates on open:
    index created, the chain walks on to the head stamp, pre-migration data
    intact."""
    agent = uuid4()
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        art = _register(reg)
        reg.set_agent_state(art.id, agent, MESIState.SHARED, trigger="fetch", tick=1)
    _revert_to_v6_shape(db_path)
    assert _user_version(db_path) == 6
    assert "idx_agent_states_agent" not in _indexes(db_path)

    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        assert reg.get_agent_state(art.id, agent) is MESIState.SHARED
        assert reg.last_observed_version_for(art.id, agent) == 1
    assert "idx_agent_states_agent" in _indexes(db_path)


def test_upgraded_and_fresh_index_sets_identical(tmp_path: Path) -> None:
    """A v6->v7-migrated db and a fresh-v7 db expose the same idx_* set — the
    migration creates exactly what the fresh DDL declares."""
    fresh = tmp_path / "fresh.db"
    upgraded = tmp_path / "upgraded.db"
    with SqliteArtifactRegistry(fresh):
        pass
    with SqliteArtifactRegistry(upgraded):
        pass
    _revert_to_v6_shape(upgraded)
    with SqliteArtifactRegistry(upgraded):
        pass
    assert _indexes(upgraded) == _indexes(fresh)

