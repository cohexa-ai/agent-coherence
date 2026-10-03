# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for SqliteArtifactRegistry (plan Unit 1).

Scope is the plugin's hot-path methods + plugin-only extensions
(resolve_or_register, artifacts_held_by_agent). Full ArtifactRegistry
parity test against tests/test_coordinator.py is NOT a v0.1 goal — that
test suite exercises content-fetch semantics that this storage layer
deliberately does not preserve (KTD-13 contract divergence).
"""

from __future__ import annotations

import os
import sqlite3
import stat
import threading
import warnings
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.retention import RetentionPolicy
from ccs.coordinator.sqlite_registry import (
    _ARTIFACT_VERSIONS_DDL,
    _DB_FILE_MODE,
    CCS_STATE_LOG_SCHEMA_VERSION,
    SCHEMA_USER_VERSION,
    CrossRuntimeSchemaError,
    MissingDatabaseError,
    ReadOnlyMutationError,
    SchemaVersionError,
    SqliteArtifactRegistry,
    StoreNeedsRecoveryError,
)
from ccs.core.exceptions import CROSS_RUNTIME_SCHEMA_REASON
from ccs.core.states import MESIState, TransientState
from ccs.core.types import Artifact, CasCorruption, ConflictDetail


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


def _make_artifact(name: str = "plan.md", version: int = 1, content_hash: str = "h1") -> Artifact:
    return Artifact(id=uuid4(), name=name, version=version, content_hash=content_hash)


# --------------------------------------------------------------------
# Schema lifecycle
# --------------------------------------------------------------------


def test_fresh_init_creates_v1_schema(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        # Tables exist
        rows = reg._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = {r[0] for r in rows}
        assert {"artifacts", "agent_states", "heartbeats", "registry_meta"}.issubset(names)
        # user_version set
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        # registry_meta seeded
        meta = dict(reg._conn.execute("SELECT key, value FROM registry_meta").fetchall())
        assert "instance_id" in meta and "sequence_number" in meta
        assert meta["sequence_number"] == "0"


def test_rehydration_preserves_instance_id_and_seq(db_path: Path) -> None:
    # Write seed data with a state_log that emits a few entries.
    emitted: list[dict] = []
    instance_id_pinned = str(uuid4())
    with SqliteArtifactRegistry(
        db_path,
        state_log=emitted.append,
        instance_id=instance_id_pinned,
    ) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="ignored-per-KTD-13")
        agent = uuid4()
        reg.set_agent_state(art.id, agent, MESIState.SHARED, trigger="read", tick=100)
        reg.set_agent_state(art.id, agent, MESIState.EXCLUSIVE, trigger="write", tick=101)
        assert reg._seq == 2
        assert emitted[-1]["sequence_number"] == 2
        first_artifact_id = art.id

    # Re-open
    with SqliteArtifactRegistry(db_path) as reg2:
        assert reg2._instance_id == instance_id_pinned
        assert reg2._seq == 2  # persisted across restart
        assert reg2.has_artifact(first_artifact_id)
        assert reg2.get_agent_state(first_artifact_id, agent) == MESIState.EXCLUSIVE


def test_state_log_without_instance_id_raises(db_path: Path) -> None:
    with pytest.raises(ValueError, match="instance_id must be provided"):
        SqliteArtifactRegistry(db_path, state_log=lambda e: None)


def test_unexpected_schema_version_raises(db_path: Path) -> None:
    # Seed db with v1, then mutate user_version to a future value.
    with SqliteArtifactRegistry(db_path):
        pass
    import sqlite3
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    conn.execute("PRAGMA user_version = 99")
    conn.close()

    with pytest.raises(SchemaVersionError, match="unexpected schema version 99"):
        SqliteArtifactRegistry(db_path)


# --------------------------------------------------------------------
# Artifact CRUD
# --------------------------------------------------------------------


def test_register_and_get_artifact(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact(name="plan.md", version=1, content_hash="abc123")
        reg.register_artifact(art, content="ignored")
        assert reg.has_artifact(art.id)
        fetched = reg.get_artifact(art.id)
        assert fetched is not None
        assert fetched.name == "plan.md"
        assert fetched.version == 1
        assert fetched.content_hash == "abc123"


def test_get_content_returns_empty_for_known_none_for_unknown(db_path: Path) -> None:
    """KTD-13 contract divergence."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="real-content-that-will-be-discarded")
        assert reg.get_content(art.id) == b""
        assert reg.get_content(uuid4()) is None


def test_artifact_ids_returns_all(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        a1 = _make_artifact(name="plan.md")
        a2 = _make_artifact(name="spec.md")
        reg.register_artifact(a1, content="")
        reg.register_artifact(a2, content="")
        ids = reg.artifact_ids()
        assert set(ids) == {a1.id, a2.id}


def test_set_artifact_and_content_updates_metadata(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact(version=1, content_hash="v1hash")
        reg.register_artifact(art, content="")
        new_art = Artifact(id=art.id, name=art.name, version=2, content_hash="v2hash")
        writer = uuid4()
        reg.set_artifact_and_content(art.id, new_art, content="ignored", last_writer=writer)
        fetched = reg.get_artifact(art.id)
        assert fetched.version == 2
        assert fetched.content_hash == "v2hash"


def test_remove_artifact_cascades_to_agent_states(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        agent = uuid4()
        reg.set_agent_state(art.id, agent, MESIState.SHARED, tick=1)
        reg.remove_artifact(art.id)
        assert not reg.has_artifact(art.id)
        # State for the removed artifact should be gone (FK cascade)
        assert reg.get_agent_state(art.id, agent) is None


# --------------------------------------------------------------------
# Agent state map
# --------------------------------------------------------------------


def test_set_agent_state_cycles(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        agent = uuid4()

        # INVALID -> SHARED
        reg.set_agent_state(art.id, agent, MESIState.SHARED, tick=10)
        assert reg.get_agent_state(art.id, agent) == MESIState.SHARED
        # SHARED -> EXCLUSIVE (M∪E acquire — sets granted_at_tick)
        reg.set_agent_state(art.id, agent, MESIState.EXCLUSIVE, tick=20)
        assert reg.granted_at_tick(agent, art.id) == 20
        # EXCLUSIVE -> MODIFIED (M↔E preserve grant tick)
        reg.set_agent_state(art.id, agent, MESIState.MODIFIED, tick=25)
        assert reg.granted_at_tick(agent, art.id) == 20  # preserved
        # MODIFIED -> INVALID (drops the slot)
        reg.set_agent_state(art.id, agent, MESIState.INVALID, tick=30)
        assert reg.granted_at_tick(agent, art.id) is None


def test_set_agent_state_on_unknown_artifact_raises(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        with pytest.raises(KeyError):
            reg.set_agent_state(uuid4(), uuid4(), MESIState.SHARED, tick=1)


def test_get_state_map_returns_per_agent_view(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        a1, a2 = uuid4(), uuid4()
        reg.set_agent_state(art.id, a1, MESIState.SHARED, tick=1)
        reg.set_agent_state(art.id, a2, MESIState.EXCLUSIVE, tick=2)
        state_map = reg.get_state_map(art.id)
        assert state_map == {a1: MESIState.SHARED, a2: MESIState.EXCLUSIVE}


def test_valid_holders_excludes_invalid(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        a1, a2 = uuid4(), uuid4()
        reg.set_agent_state(art.id, a1, MESIState.SHARED, tick=1)
        reg.set_agent_state(art.id, a2, MESIState.SHARED, tick=2)
        reg.set_agent_state(art.id, a2, MESIState.INVALID, tick=3)
        holders = reg.valid_holders(art.id)
        assert holders == [a1]


# --------------------------------------------------------------------
# state_log mutation-then-log + sequence rollback
# --------------------------------------------------------------------


def test_state_log_emitted_on_each_state_change(db_path: Path) -> None:
    emitted: list[dict] = []
    with SqliteArtifactRegistry(
        db_path, state_log=emitted.append, instance_id="instance-1"
    ) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        agent = uuid4()
        reg.set_agent_state(art.id, agent, MESIState.SHARED, trigger="t1", tick=10)
        reg.set_agent_state(art.id, agent, MESIState.EXCLUSIVE, trigger="t2", tick=11)
        assert len(emitted) == 2
        assert emitted[0]["sequence_number"] == 1
        assert emitted[0]["from_state"] == "INVALID"
        assert emitted[0]["to_state"] == "SHARED"
        assert emitted[0]["trigger"] == "t1"
        assert emitted[0]["schema_version"] == CCS_STATE_LOG_SCHEMA_VERSION
        assert emitted[0]["instance_id"] == "instance-1"
        assert emitted[1]["sequence_number"] == 2
        assert emitted[1]["from_state"] == "SHARED"
        assert emitted[1]["to_state"] == "EXCLUSIVE"


def test_state_log_raise_rolls_back_seq_and_state(db_path: Path) -> None:
    """Mutation-then-log invariant: callback raise → ROLLBACK + _seq decrement."""
    raise_on_seq = 2
    emitted: list[dict] = []

    def hooky(entry: dict) -> None:
        if entry["sequence_number"] == raise_on_seq:
            raise RuntimeError("simulated callback failure")
        emitted.append(entry)

    with SqliteArtifactRegistry(
        db_path, state_log=hooky, instance_id="instance-1"
    ) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        agent = uuid4()
        # First transition emits seq=1 successfully.
        reg.set_agent_state(art.id, agent, MESIState.SHARED, tick=10)
        assert reg._seq == 1
        # Second transition would emit seq=2; callback raises.
        with pytest.raises(RuntimeError, match="simulated"):
            reg.set_agent_state(art.id, agent, MESIState.EXCLUSIVE, tick=11)
        # _seq must have rolled back to 1.
        assert reg._seq == 1
        # Agent state must remain SHARED (the rolled-back transition).
        assert reg.get_agent_state(art.id, agent) == MESIState.SHARED
        # registry_meta sequence_number persisted is still 1 (not 2).
        meta = dict(reg._conn.execute(
            "SELECT key, value FROM registry_meta WHERE key='sequence_number'"
        ).fetchall())
        assert meta["sequence_number"] == "1"


# --------------------------------------------------------------------
# Heartbeat monotonicity
# --------------------------------------------------------------------


def test_record_heartbeat_monotonic(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        agent = uuid4()
        reg.record_heartbeat(agent, now_tick=100)
        reg.record_heartbeat(agent, now_tick=50)  # earlier; should be ignored
        reg.record_heartbeat(agent, now_tick=200)
        assert reg.last_heartbeat_tick(agent) == 200


# --------------------------------------------------------------------
# Plugin extensions
# --------------------------------------------------------------------


def test_resolve_or_register_first_observation_creates_v1(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id = reg.resolve_or_register("plan.md", content_hash="hash-v1")
        art = reg.get_artifact(artifact_id)
        assert art is not None
        assert art.name == "plan.md"
        assert art.version == 1
        assert art.content_hash == "hash-v1"


def test_resolve_or_register_idempotent(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        id1 = reg.resolve_or_register("plan.md", content_hash="any-hash")
        id2 = reg.resolve_or_register("plan.md", content_hash="different-hash")
        assert id1 == id2
        # Content hash stays the original v1 — resolve_or_register does NOT update.
        assert reg.get_artifact(id1).content_hash == "any-hash"


def test_resolve_or_register_race_safe(db_path: Path) -> None:
    """N threads concurrently resolve the same fresh path → all return same id."""
    with SqliteArtifactRegistry(db_path) as reg:
        results: list[UUID] = []
        barrier = threading.Barrier(10)

        def call() -> None:
            barrier.wait()
            results.append(reg.resolve_or_register("plan.md", content_hash="h"))

        threads = [threading.Thread(target=call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10
        assert len(set(results)) == 1, f"expected single id, got {set(results)}"


def test_artifacts_held_by_agent_filters_by_state(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        a_held = _make_artifact(name="plan.md")
        a_invalid = _make_artifact(name="other.md")
        reg.register_artifact(a_held, content="")
        reg.register_artifact(a_invalid, content="")
        agent = uuid4()
        reg.set_agent_state(a_held.id, agent, MESIState.EXCLUSIVE, tick=1)
        reg.set_agent_state(a_invalid.id, agent, MESIState.INVALID, tick=2)

        held = reg.artifacts_held_by_agent(
            agent, {MESIState.EXCLUSIVE, MESIState.MODIFIED}
        )
        assert held == [a_held.id]


def test_artifacts_held_by_agent_empty_states_returns_empty(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        assert reg.artifacts_held_by_agent(uuid4(), set()) == []


def test_lookup_artifact_id_by_name_roundtrip(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id = reg.resolve_or_register("plan.md", content_hash="h")
        assert reg.lookup_artifact_id_by_name("plan.md") == artifact_id
        assert reg.lookup_artifact_id_by_name("does-not-exist.md") is None


# --------------------------------------------------------------------
# Transient state
# --------------------------------------------------------------------


def test_transient_state_set_get_clear(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        agent = uuid4()
        reg.set_agent_transient(art.id, agent, TransientState.ISG, entered_tick=5)
        assert reg.get_agent_transient(art.id, agent) == TransientState.ISG
        assert reg.get_transient_tick(art.id, agent) == 5
        reg.clear_agent_transient(art.id, agent)
        assert reg.get_agent_transient(art.id, agent) is None


# --------------------------------------------------------------------
# Cross-thread concurrency (ThreadingHTTPServer simulation)
# --------------------------------------------------------------------


def test_concurrent_state_changes_no_deadlock(db_path: Path) -> None:
    """10 threads racing on set_agent_state for distinct (agent, artifact)
    pairs should all complete without deadlock; final states are observable."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        agents = [uuid4() for _ in range(10)]

        def worker(agent_id: UUID, t: int) -> None:
            reg.set_agent_state(art.id, agent_id, MESIState.SHARED, tick=t)

        threads = [
            threading.Thread(target=worker, args=(a, i + 1))
            for i, a in enumerate(agents)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        state_map = reg.get_state_map(art.id)
        assert len(state_map) == 10
        assert all(s == MESIState.SHARED for s in state_map.values())


# ----------------------------------------------------------------------
# COR-04 — resolve_or_register re-fetch race produces informative error
# ----------------------------------------------------------------------


def test_cor04_resolve_or_register_post_rollback_delete_raises_runtime_error(
    tmp_path
):
    """COR-04: when a concurrent remove_artifact deletes the winning
    racer's row between our ROLLBACK and re-fetch, the function must
    raise an informative RuntimeError explaining the race rather than
    re-raising the original sqlite3.IntegrityError with no context.

    Wraps reg._conn in a proxy that forces INSERT to raise IntegrityError
    and the post-ROLLBACK SELECT to return None.
    """
    import sqlite3

    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry

    reg = SqliteArtifactRegistry(tmp_path / "state.db")
    try:
        real_conn = reg._conn
        call_state = {"insert_seen": False, "post_rollback_select_seen": False}

        class _ConnProxy:
            def execute(self, sql, *args):
                sql_lower = sql.strip().lower()
                if sql_lower.startswith("insert into artifacts"):
                    call_state["insert_seen"] = True
                    raise sqlite3.IntegrityError("simulated UNIQUE collision")
                if (
                    call_state["insert_seen"]
                    and not call_state["post_rollback_select_seen"]
                    and sql_lower.startswith("select id from artifacts where name")
                ):
                    call_state["post_rollback_select_seen"] = True

                    class _Empty:
                        def fetchone(self_inner):
                            return None

                    return _Empty()
                return real_conn.execute(sql, *args)

            def __getattr__(self, name):
                return getattr(real_conn, name)

        reg._conn = _ConnProxy()
        try:
            import pytest as _pytest
            with _pytest.raises(RuntimeError) as excinfo:
                reg.resolve_or_register("plan.md", content_hash="abc")
            assert "lost INSERT race" in str(excinfo.value)
            assert "plan.md" in str(excinfo.value)
            assert isinstance(excinfo.value.__cause__, sqlite3.IntegrityError)
        finally:
            reg._conn = real_conn
    finally:
        reg.close()


# --------------------------------------------------------------------
# OCC commit-CAS (plan Unit 2)
# --------------------------------------------------------------------


def _seed_artifact_with_shared_writer(
    reg: SqliteArtifactRegistry, *, version: int = 1
) -> tuple[UUID, UUID]:
    """Register an artifact at ``version`` and put one agent in SHARED (the OCC
    writer's pre-commit state). Returns ``(artifact_id, agent_id)``."""
    art = _make_artifact(version=version, content_hash="h-init")
    reg.register_artifact(art, content="")
    agent = uuid4()
    reg.set_agent_state(art.id, agent, MESIState.SHARED, tick=1)
    return art.id, agent


def test_commit_cas_happy_bumps_version_and_shares(db_path: Path) -> None:
    """expected == current, no other E/M holder → version N+1, agent SHARED."""
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id, agent = _seed_artifact_with_shared_writer(reg, version=1)
        result = reg.commit_cas(
            artifact_id, agent, expected_version=1, content_hash="h-new", tick=7
        )
        assert isinstance(result, tuple)
        updated, invalidated = result
        assert updated.version == 2
        assert updated.content_hash == "h-new"
        assert invalidated == []
        # An OCC writer holds no grant — it ends SHARED (not MODIFIED), and
        # SHARED is not an M∪E acquire so no granted_at_tick slot is set.
        assert reg.get_agent_state(artifact_id, agent) == MESIState.SHARED
        assert reg.granted_at_tick(agent, artifact_id) is None
        # last_writer recorded on the artifact row.
        assert reg.last_writer_for(artifact_id) == agent


def test_commit_cas_invalidates_non_invalid_peers(db_path: Path) -> None:
    """WIN invalidates every non-INVALID peer; returns their ids for signals."""
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id, agent = _seed_artifact_with_shared_writer(reg, version=1)
        peer_shared = uuid4()
        peer_invalid = uuid4()
        reg.set_agent_state(artifact_id, peer_shared, MESIState.SHARED, tick=2)
        reg.set_agent_state(artifact_id, peer_invalid, MESIState.SHARED, tick=2)
        reg.set_agent_state(artifact_id, peer_invalid, MESIState.INVALID, tick=3)

        updated, invalidated = reg.commit_cas(
            artifact_id, agent, expected_version=1, content_hash="h-new", tick=9
        )
        assert updated.version == 2
        # Only the still-valid peer is invalidated; the already-INVALID one is not.
        assert invalidated == [peer_shared]
        assert reg.get_agent_state(artifact_id, peer_shared) == MESIState.INVALID
        assert reg.get_agent_state(artifact_id, agent) == MESIState.SHARED


def test_commit_cas_version_mismatch_no_mutation(db_path: Path) -> None:
    """expected < current → version_mismatch ConflictDetail, NO mutation."""
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id, _ = _seed_artifact_with_shared_writer(reg, version=5)
        loser = uuid4()
        reg.set_agent_state(artifact_id, loser, MESIState.SHARED, tick=2)
        result = reg.commit_cas(
            artifact_id, loser, expected_version=3, content_hash="h-stale", tick=10
        )
        assert result == ConflictDetail("version_mismatch", 5)
        # No mutation: version unchanged, content_hash unchanged, state SHARED.
        art = reg.get_artifact(artifact_id)
        assert art.version == 5
        assert art.content_hash == "h-init"
        assert reg.get_agent_state(artifact_id, loser) == MESIState.SHARED


def test_commit_cas_other_holder_no_mutation(db_path: Path) -> None:
    """version matches but another agent holds E/M → other_holder, NO mutation."""
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id, occ_writer = _seed_artifact_with_shared_writer(reg, version=1)
        pessimistic = uuid4()
        reg.set_agent_state(artifact_id, pessimistic, MESIState.EXCLUSIVE, tick=2)
        result = reg.commit_cas(
            artifact_id, occ_writer, expected_version=1, content_hash="h-new", tick=11
        )
        assert result == ConflictDetail("other_holder", 1)
        assert reg.get_artifact(artifact_id).version == 1
        assert reg.get_agent_state(artifact_id, occ_writer) == MESIState.SHARED
        # The pessimistic holder is untouched.
        assert reg.get_agent_state(artifact_id, pessimistic) == MESIState.EXCLUSIVE


def test_commit_cas_modified_peer_also_trips_other_holder(db_path: Path) -> None:
    """MODIFIED (not just EXCLUSIVE) peer trips other_holder."""
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id, occ_writer = _seed_artifact_with_shared_writer(reg, version=1)
        holder = uuid4()
        reg.set_agent_state(artifact_id, holder, MESIState.EXCLUSIVE, tick=2)
        reg.set_agent_state(artifact_id, holder, MESIState.MODIFIED, tick=3)
        result = reg.commit_cas(
            artifact_id, occ_writer, expected_version=1, content_hash="x", tick=4
        )
        assert result == ConflictDetail("other_holder", 1)


def test_commit_cas_expected_greater_returns_corruption(db_path: Path) -> None:
    """expected > current → CasCorruption sentinel (service maps to CoherenceError)."""
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id, agent = _seed_artifact_with_shared_writer(reg, version=2)
        result = reg.commit_cas(
            artifact_id, agent, expected_version=9, content_hash="x", tick=1
        )
        assert isinstance(result, CasCorruption)
        assert result.current_version == 2
        # Corruption is distinct from a conflict.
        assert not isinstance(result, ConflictDetail)
        # No mutation.
        assert reg.get_artifact(artifact_id).version == 2


def test_commit_cas_version_check_precedes_holder_check(db_path: Path) -> None:
    """The version branch fires before the holder branch → a stale loser gets
    version_mismatch (not other_holder). The winner ends SHARED (an OCC writer
    holds no grant), so the version check is the sole discriminator here."""
    with SqliteArtifactRegistry(db_path) as reg:
        artifact_id, winner = _seed_artifact_with_shared_writer(reg, version=1)
        loser = uuid4()
        reg.set_agent_state(artifact_id, loser, MESIState.SHARED, tick=2)
        # Winner commits: version 1→2, winner now SHARED (no grant).
        reg.commit_cas(artifact_id, winner, expected_version=1, content_hash="w", tick=5)
        assert reg.get_agent_state(artifact_id, winner) == MESIState.SHARED
        # Loser retries at stale expected_version=1; version_mismatch must win
        # (not other_holder), proving branch order.
        result = reg.commit_cas(
            artifact_id, loser, expected_version=1, content_hash="l", tick=6
        )
        assert result == ConflictDetail("version_mismatch", 2)


def test_commit_cas_unknown_artifact_raises_keyerror(db_path: Path) -> None:
    with SqliteArtifactRegistry(db_path) as reg:
        with pytest.raises(KeyError):
            reg.commit_cas(uuid4(), uuid4(), expected_version=1, content_hash="x")


def test_commit_cas_emits_state_log_for_committer_and_peers(db_path: Path) -> None:
    """WIN emits one state_log entry per transition (peers→INVALID, committer→M),
    with monotonic sequence numbers continuing from prior emissions."""
    emitted: list[dict] = []
    with SqliteArtifactRegistry(
        db_path, state_log=emitted.append, instance_id="inst-cas"
    ) as reg:
        art = _make_artifact(version=1, content_hash="h-init")
        reg.register_artifact(art, content="")
        agent = uuid4()
        peer = uuid4()
        reg.set_agent_state(art.id, agent, MESIState.SHARED, tick=1)  # seq 1
        reg.set_agent_state(art.id, peer, MESIState.SHARED, tick=1)  # seq 2
        assert reg._seq == 2
        emitted.clear()

        reg.commit_cas(art.id, agent, expected_version=1, content_hash="h-new", tick=5)
        # One peer invalidation + one committer SHARED = 2 new entries.
        assert len(emitted) == 2
        transitions = {(e["from_state"], e["to_state"]) for e in emitted}
        assert ("SHARED", "INVALID") in transitions
        # The committer ends SHARED (OCC writer holds no grant): SHARED→SHARED.
        assert ("SHARED", "SHARED") in transitions
        seqs = sorted(e["sequence_number"] for e in emitted)
        assert seqs == [3, 4]
        assert reg._seq == 4
        # The committer entry carries the new content_hash; the invalidation does
        # not. (The committer is the SHARED→SHARED entry, not the peer→INVALID.)
        committer_entry = next(
            e for e in emitted if (e["from_state"], e["to_state"]) == ("SHARED", "SHARED")
        )
        assert committer_entry["content_hash"] == "h-new"
        assert committer_entry["version"] == 2
        inv_entry = next(e for e in emitted if e["to_state"] == "INVALID")
        assert inv_entry["content_hash"] is None
        # sequence_number persisted to registry_meta.
        meta = dict(reg._conn.execute(
            "SELECT key, value FROM registry_meta WHERE key='sequence_number'"
        ).fetchall())
        assert meta["sequence_number"] == "4"


def test_commit_cas_state_log_raise_rolls_back_seq_and_state(db_path: Path) -> None:
    """Atomicity oracle (mirrors test_state_log_raise_rolls_back_seq_and_state):
    a state_log raise mid-CAS → ROLLBACK leaves version + _seq + agent state
    untouched, and registry_meta.sequence_number unchanged."""
    # Raise on the committer's SHARED emission (the committer now ends SHARED —
    # an OCC writer holds no grant). The peer invalidation is emitted first
    # (to_state INVALID), the committer emission is the one that raises — forcing
    # a rollback after the UPDATE + peer mutation already ran inside the
    # transaction. SHARED uniquely targets the committer (the peer goes →INVALID).
    raise_on_to_state = "SHARED"
    emitted: list[dict] = []

    def hooky(entry: dict) -> None:
        if entry["to_state"] == raise_on_to_state and entry["trigger"] == "commit_cas":
            raise RuntimeError("simulated callback failure mid-CAS")
        emitted.append(entry)

    with SqliteArtifactRegistry(
        db_path, state_log=hooky, instance_id="inst-cas"
    ) as reg:
        art = _make_artifact(version=1, content_hash="h-init")
        reg.register_artifact(art, content="")
        agent = uuid4()
        peer = uuid4()
        reg.set_agent_state(art.id, agent, MESIState.SHARED, tick=1)  # seq 1
        reg.set_agent_state(art.id, peer, MESIState.SHARED, tick=1)  # seq 2
        assert reg._seq == 2

        with pytest.raises(RuntimeError, match="simulated callback failure mid-CAS"):
            reg.commit_cas(art.id, agent, expected_version=1, content_hash="h-new", tick=5)

        # Everything rolled back: version still 1, content_hash still h-init.
        art_now = reg.get_artifact(art.id)
        assert art_now.version == 1
        assert art_now.content_hash == "h-init"
        # Agent state preserved (committer still SHARED, peer still SHARED — the
        # peer's INVALID mutation was rolled back too).
        assert reg.get_agent_state(art.id, agent) == MESIState.SHARED
        assert reg.get_agent_state(art.id, peer) == MESIState.SHARED
        # _seq rolled back to 2 (both the peer emission AND the failed committer
        # reservation are undone).
        assert reg._seq == 2
        meta = dict(reg._conn.execute(
            "SELECT key, value FROM registry_meta WHERE key='sequence_number'"
        ).fetchall())
        assert meta["sequence_number"] == "2"


def test_commit_cas_concurrent_writers_exactly_one_wins(db_path: Path) -> None:
    """Two barrier-synced OCC writers (both SHARED, same expected_version) race
    on commit_cas → exactly one commits (version N+1, not N+2), the loser gets a
    ConflictDetail. No silent clobber (R5 atomicity via BEGIN IMMEDIATE)."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact(version=1, content_hash="h-init")
        reg.register_artifact(art, content="")
        writers = [uuid4() for _ in range(8)]
        for w in writers:
            reg.set_agent_state(art.id, w, MESIState.SHARED, tick=1)

        results: dict[UUID, object] = {}
        barrier = threading.Barrier(len(writers))

        def attempt(writer_id: UUID) -> None:
            barrier.wait()
            results[writer_id] = reg.commit_cas(
                art.id, writer_id, expected_version=1,
                content_hash=f"h-{writer_id.hex[:6]}", tick=5,
            )

        threads = [threading.Thread(target=attempt, args=(w,)) for w in writers]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        wins = [r for r in results.values() if isinstance(r, tuple)]
        conflicts = [r for r in results.values() if isinstance(r, ConflictDetail)]
        assert len(wins) == 1, f"expected exactly one win, got {len(wins)}"
        assert len(conflicts) == len(writers) - 1
        # Final version advanced by exactly one — no lost update / no clobber.
        assert reg.get_artifact(art.id).version == 2
        # Every loser saw version_mismatch (two OCC writers are both SHARED, so
        # the loser is elected by the version check, never other_holder).
        assert all(c.reason == "version_mismatch" for c in conflicts)
        assert all(c.current_version == 2 for c in conflicts)


# ----------------------------------------------------------------------
# Read-generation fence (Piece #2) — Unit 2: additive schema, no migration
# ----------------------------------------------------------------------


def _build_pre_fence_db(db_path: Path, art_id: str, ag_id: str) -> None:
    """Create a v1 state.db with the schema as it existed BEFORE the
    read-generation fence (no owner_generation / read_generation columns, no
    coordinator_epoch), with one artifact + one M-grant, user_version = 1."""
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE artifacts (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
            "version INTEGER NOT NULL, content_hash TEXT NOT NULL, size_tokens INTEGER, "
            "last_writer_id TEXT, updated_at REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX idx_artifacts_name ON artifacts(name)")
        conn.execute(
            "CREATE TABLE agent_states (artifact_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
            "state TEXT NOT NULL, transient_state TEXT, transient_tick INTEGER, "
            "granted_at_tick INTEGER, last_reclaim_trigger TEXT, last_reclaim_tick INTEGER, "
            "PRIMARY KEY (artifact_id, agent_id), "
            "FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE)"
        )
        conn.execute(
            "CREATE TABLE heartbeats (agent_id TEXT PRIMARY KEY, last_tick INTEGER NOT NULL)"
        )
        conn.execute("CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO registry_meta (key, value) VALUES (?, ?), (?, ?)",
            ("instance_id", "fixed-instance", "sequence_number", "7"),
        )
        conn.execute(
            "INSERT INTO artifacts (id, name, version, content_hash, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (art_id, "plan.md", 3, "deadbeef", 1.0),
        )
        conn.execute(
            "INSERT INTO agent_states (artifact_id, agent_id, state) VALUES (?, ?, ?)",
            (art_id, ag_id, "M"),
        )
        conn.execute("PRAGMA user_version = 1")
        conn.execute("COMMIT")
    finally:
        conn.close()


def test_fresh_db_has_fence_schema(db_path: Path) -> None:
    """A freshly initialized db carries the fence columns + coordinator_epoch
    inline in the v2 schema AND the v3 ``session_pins`` table (a fresh db is
    created at the latest schema directly)."""
    with SqliteArtifactRegistry(db_path) as reg:
        art_cols = {r[1] for r in reg._conn.execute("PRAGMA table_info(artifacts)").fetchall()}
        state_cols = {
            r[1] for r in reg._conn.execute("PRAGMA table_info(agent_states)").fetchall()
        }
        assert "owner_generation" in art_cols
        assert "read_generation" in state_cols
        assert reg._coordinator_epoch
        tables = {
            r[0] for r in reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        # The v3 session-pin table (SB-17 Unit 2) is present on a fresh db.
        assert "session_pins" in tables
        # The v5 workspace-checkpoint tables (WV Unit 2) are present on a fresh
        # db too — the fresh apply always lands the COMPLETE current schema.
        assert "workspace_checkpoints" in tables
        assert "workspace_checkpoint_members" in tables
        # The v6 observed-version comparand column (SB-10) is inline as well.
        assert "last_observed_version" in state_cols
        # The v7 agent_id index (SB-10 perf follow-up) is inline as well.
        indexes = {
            r[0] for r in reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_agent_states_agent" in indexes
        # The v8 caller-principal table (its own table, never session_meta).
        assert "caller_principals" in tables
        # A fresh db is created directly at the current head.
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        assert SCHEMA_USER_VERSION == 9


def test_pre_fence_db_upgrades_in_place_additively(db_path: Path) -> None:
    """A pre-fence v1 db migrates all the way to v3 on open (chained
    v1->v2->v3): it gains the fence columns + coordinator_epoch (the subsumed
    fence shim), the v2 artifact_versions table, AND the v3 session_pins table,
    with user_version stamped 3. Existing rows backfill to owner_generation = 0 /
    read_generation = NULL, and prior meta is preserved. Re-open is the
    write-free v3 path; the epoch is stable."""
    art_id, ag_id = str(uuid4()), str(uuid4())
    _build_pre_fence_db(db_path, art_id, ag_id)

    with SqliteArtifactRegistry(db_path) as reg:
        art_cols = {r[1] for r in reg._conn.execute("PRAGMA table_info(artifacts)").fetchall()}
        state_cols = {
            r[1] for r in reg._conn.execute("PRAGMA table_info(agent_states)").fetchall()
        }
        assert "owner_generation" in art_cols
        assert "read_generation" in state_cols
        # The migration also created the v2 retention table.
        tables = {
            r[0] for r in reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "artifact_versions" in tables
        # The chained v2->v3 step also created the session_pins table.
        assert "session_pins" in tables
        # Existing rows backfill: artifact -> 0, grant -> NULL (absent operand).
        assert (
            reg._conn.execute(
                "SELECT owner_generation FROM artifacts WHERE id = ?", (art_id,)
            ).fetchone()[0]
            == 0
        )
        assert (
            reg._conn.execute(
                "SELECT read_generation FROM agent_states "
                "WHERE artifact_id = ? AND agent_id = ?",
                (art_id, ag_id),
            ).fetchone()[0]
            is None
        )
        # Epoch seeded + loaded; user_version stamped to current by the chained migration.
        assert reg._coordinator_epoch
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        # Prior meta preserved (no clobber of instance_id / sequence_number).
        assert reg._instance_id == "fixed-instance"
        assert reg._seq == 7
        epoch1 = reg._coordinator_epoch

    # Idempotent re-open: no error, columns intact, epoch stable.
    with SqliteArtifactRegistry(db_path) as reg2:
        assert reg2._coordinator_epoch == epoch1
        art_cols2 = {
            r[1] for r in reg2._conn.execute("PRAGMA table_info(artifacts)").fetchall()
        }
        assert "owner_generation" in art_cols2


def test_owner_generation_bumps_on_reclaim_only(db_path: Path) -> None:
    """A sweep reclamation (M/E -> INVALID via a reclaim trigger) bumps the
    artifact's owner_generation monotonically; a peer-invalidation or a clean
    downgrade does NOT bump (version-CAS covers those)."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
        reg.register_artifact(art, content="ignored")
        a, b = uuid4(), uuid4()
        assert reg.get_owner_generation(art.id) == 0

        reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
        reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
        assert reg.get_owner_generation(art.id) == 1  # reclaim bumps

        reg.set_agent_state(art.id, b, MESIState.EXCLUSIVE, trigger="write", tick=11)
        reg.set_agent_state(art.id, b, MESIState.INVALID, trigger="reclaim_max_hold", tick=20)
        assert reg.get_owner_generation(art.id) == 2  # monotonic across reclaims

        # Peer-invalidation (non-reclaim trigger) does NOT bump.
        reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=21)
        reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="peer_invalidation", tick=22)
        assert reg.get_owner_generation(art.id) == 2

        # Clean downgrade E -> SHARED does NOT bump.
        reg.set_agent_state(art.id, b, MESIState.EXCLUSIVE, trigger="write", tick=23)
        reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="downgrade", tick=24)
        assert reg.get_owner_generation(art.id) == 2


def test_read_generation_captured_at_claim(db_path: Path) -> None:
    """read_generation is captured at the claim-establishing point: an E/M
    acquire (incl. acquire-WITHOUT-a-prior-read -- the P0 fix) and a fetch read.
    A never-claimed agent has None (absent operand). A reclaim does NOT refresh
    the captured value; the next claim captures the new generation."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
        reg.register_artifact(art, content="ignored")
        a, b, c = uuid4(), uuid4(), uuid4()

        # Never-claimed agent: absent operand.
        assert reg.get_read_generation(art.id, c) is None

        # Pessimistic acquire WITHOUT a prior fetch (the P0 fix) captures gen 0.
        reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
        assert reg.get_read_generation(art.id, a) == 0
        # A fetch read captures too.
        reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=2)
        assert reg.get_read_generation(art.id, b) == 0

        # Reclaim a -> owner_generation bumps to 1, but a's captured value is
        # PRESERVED (the reclaim does not refresh it) -- this is what makes a's
        # later commit fail the generation guard.
        reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
        assert reg.get_owner_generation(art.id) == 1
        assert reg.get_read_generation(art.id, a) == 0

        # A fresh re-acquire by a captures the NEW generation (1).
        reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=11)
        assert reg.get_read_generation(art.id, a) == 1


def test_commit_cas_fence_rejects_superseded_reader(db_path: Path) -> None:
    """OCC commit_cas returns ConflictDetail('stale_read_generation') when the
    committer's read_generation was superseded by a reclaim -- version
    unchanged, no other holder, so version-CAS cannot catch it (the headline
    reclaim-zombie case). A re-fetch (fresh read_generation) then wins; a
    never-fetched committer is rejected as an absent operand."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
        reg.register_artifact(art, content="ignored")
        a, c = uuid4(), uuid4()

        # A acquires E (captures read_generation=0), then is reclaimed
        # (owner_generation -> 1; A's read_generation preserved at 0).
        reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
        reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
        assert reg.get_owner_generation(art.id) == 1

        # A commits via OCC: version still matches (1), no other holder, but the
        # generation fence rejects the superseded read -- no phantom bump.
        res = reg.commit_cas(art.id, a, expected_version=1, content_hash="new")
        assert isinstance(res, ConflictDetail)
        assert res.reason == "stale_read_generation"
        assert reg.get_artifact(art.id).version == 1

        # A re-fetches -> fresh read_generation=1 -> the commit now wins.
        reg.set_agent_state(art.id, a, MESIState.SHARED, trigger="fetch", tick=11)
        res2 = reg.commit_cas(art.id, a, expected_version=1, content_hash="new2")
        assert not isinstance(res2, ConflictDetail)
        assert reg.get_artifact(art.id).version == 2

        # A never-claimed committer has no captured claim to supersede, so the
        # fence does NOT reject it -- version-CAS alone arbitrates (here the
        # version matches, so it wins). The fence only rejects a captured-then-
        # superseded read.
        res3 = reg.commit_cas(art.id, c, expected_version=2, content_hash="x")
        assert not isinstance(res3, ConflictDetail)
        assert reg.get_artifact(art.id).version == 3


def test_set_artifact_and_content_fence_rejects_stale_committer(db_path: Path) -> None:
    """The pessimistic guarded primitive (set_artifact_and_content with
    fence_agent_id) rejects a committer whose captured read_generation was
    superseded by a reclaim, atomically with the version persist -- closing the
    commit() race. A fresh committer persists; committer=None is unguarded."""
    from ccs.core.exceptions import StaleReadGeneration

    with SqliteArtifactRegistry(db_path) as reg:
        art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
        reg.register_artifact(art, content="ignored")
        a, b = uuid4(), uuid4()

        # a acquires E (read_generation=0), then is reclaimed (owner_generation
        # -> 1; a's read_generation preserved at 0).
        reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
        reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)

        # a's guarded persist is rejected; the version does NOT advance.
        stale = Artifact(id=art.id, name="plan.md", version=2, content_hash="stale")
        with pytest.raises(StaleReadGeneration):
            reg.set_artifact_and_content(art.id, stale, "x", last_writer=a, fence_agent_id=a)
        assert reg.get_artifact(art.id).version == 1

        # b acquires E AFTER the reclaim -> captures the current generation (1)
        # -> its guarded persist succeeds.
        reg.set_agent_state(art.id, b, MESIState.EXCLUSIVE, trigger="write", tick=11)
        ok = Artifact(id=art.id, name="plan.md", version=2, content_hash="ok")
        reg.set_artifact_and_content(art.id, ok, "x", last_writer=b, fence_agent_id=b)
        assert reg.get_artifact(art.id).version == 2

        # committer=None (external source churn) is unguarded -- persists even
        # though no agent established a claim.
        src = Artifact(id=art.id, name="plan.md", version=3, content_hash="src")
        reg.set_artifact_and_content(art.id, src, "x")
        assert reg.get_artifact(art.id).version == 3


def test_fence_columns_present_epoch_absent_recovers(db_path: Path) -> None:
    """A v1 db whose fence COLUMNS exist but whose coordinator_epoch was never
    seeded (one wild v1 variant) migrates cleanly (chained v1->v2->v3): the
    migration's table_info guards skip the duplicate column ALTERs and the
    INSERT OR IGNORE seeds the missing epoch, while stamping user_version to 3."""
    import sqlite3

    # hex ids: the registry's lookups key on UUID.hex (the pre-fence builder
    # stores ids verbatim, so dashed ids would miss the hex-keyed lookup).
    art_id, ag_id = uuid4().hex, uuid4().hex
    _build_pre_fence_db(db_path, art_id, ag_id)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "ALTER TABLE artifacts ADD COLUMN owner_generation INTEGER NOT NULL DEFAULT 0"
    )
    conn.execute("ALTER TABLE agent_states ADD COLUMN read_generation INTEGER")
    conn.commit()
    conn.close()

    with SqliteArtifactRegistry(db_path) as reg:
        assert reg._coordinator_epoch  # seeded on this open
        assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        assert reg.get_owner_generation(UUID(art_id)) == 0


# ======================================================================
# Durable version retention (plan item N v1, Unit 3) — R2, R3, R4
# ======================================================================


_K8 = RetentionPolicy(max_versions=8)


def _retain_artifact(reg, *, version: int = 1, content: str = "v1") -> Artifact:
    art = Artifact(id=uuid4(), name="plan.md", version=version, content_hash="h")
    reg.register_artifact(art, content=content)
    return art


def _build_raw_v1_db(
    db_path: Path, *, fence: bool, notices: bool
) -> tuple[str, str]:
    """Construct a RAW v1 ``state.db`` BY HAND (not via the current code path,
    which now emits v2) for one cell of the wild-v1 2x2 matrix.

    ``fence`` toggles the read-generation fence columns + the ``coordinator_epoch``
    seed (the fence shim shipped 2026-06-09 with no version bump, so most wild
    dbs PRE-DATE it). ``notices`` toggles the ``pending_notices`` table (also a
    no-bump shim). Seeds one artifact at version 3 + one M-grant so post-migration
    queries have rows to check. Returns ``(artifact_id_hex, agent_id_hex)``.
    """
    art_id, ag_id = uuid4().hex, uuid4().hex
    fence_art = ", owner_generation INTEGER NOT NULL DEFAULT 0" if fence else ""
    fence_state = ", read_generation INTEGER" if fence else ""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            f"CREATE TABLE artifacts (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
            f"version INTEGER NOT NULL{fence_art}, content_hash TEXT NOT NULL, "
            f"size_tokens INTEGER, last_writer_id TEXT, updated_at REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX idx_artifacts_name ON artifacts(name)")
        conn.execute(
            f"CREATE TABLE agent_states (artifact_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
            f"state TEXT NOT NULL, transient_state TEXT, transient_tick INTEGER, "
            f"granted_at_tick INTEGER, last_reclaim_trigger TEXT, last_reclaim_tick INTEGER"
            f"{fence_state}, PRIMARY KEY (artifact_id, agent_id), "
            f"FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE)"
        )
        conn.execute(
            "CREATE TABLE heartbeats (agent_id TEXT PRIMARY KEY, last_tick INTEGER NOT NULL)"
        )
        conn.execute("CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT)")
        meta = [("instance_id", "wild-v1"), ("sequence_number", "4")]
        if fence:
            meta.append(("coordinator_epoch", "epoch-pre-migration"))
        conn.executemany("INSERT INTO registry_meta (key, value) VALUES (?, ?)", meta)
        if notices:
            conn.execute(
                "CREATE TABLE pending_notices (agent_id TEXT NOT NULL, "
                "artifact_id TEXT NOT NULL, preempter_agent_id TEXT NOT NULL, "
                "preempted_at_unix_ts REAL NOT NULL, PRIMARY KEY (agent_id, artifact_id), "
                "FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE)"
            )
        if fence:
            conn.execute(
                "INSERT INTO artifacts (id, name, version, owner_generation, "
                "content_hash, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (art_id, "plan.md", 3, 0, "deadbeef", 1.0),
            )
        else:
            conn.execute(
                "INSERT INTO artifacts (id, name, version, content_hash, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (art_id, "plan.md", 3, "deadbeef", 1.0),
            )
        conn.execute(
            "INSERT INTO agent_states (artifact_id, agent_id, state) VALUES (?, ?, ?)",
            (art_id, ag_id, "M"),
        )
        conn.execute("PRAGMA user_version = 1")
        conn.execute("COMMIT")
    finally:
        conn.close()
    return art_id, ag_id


class TestDurableRetentionRoundTrip:
    """R2/R3: retained versions survive a fresh process; types preserved."""

    def test_round_trip_across_fresh_instance_str_and_bytes(self, db_path: Path) -> None:
        # Capture across all three points with both str and bytes, then reopen a
        # NEW registry instance (= fresh process) and read the durable rows.
        pol = RetentionPolicy(max_versions=4)
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=pol
        ) as reg:
            art = _retain_artifact(reg, version=1, content="str-v1")  # register
            n2 = Artifact(id=art.id, name="plan.md", version=2, content_hash="h")
            reg.set_artifact_and_content(art.id, n2, "str-v2")  # pessimistic
            w = uuid4()
            reg.set_agent_state(art.id, w, MESIState.SHARED, tick=1)
            reg.commit_cas(
                art.id, w, expected_version=2, content_hash="h3",
                content=b"\x00\x01bytes-v3", tick=2,
            )  # commit_cas WIN, bytes
            art_id = art.id

        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=pol
        ) as reg2:
            v1 = reg2.get_content_at_version(art_id, 1)
            v2 = reg2.get_content_at_version(art_id, 2)
            v3 = reg2.get_content_at_version(art_id, 3)
            # str rows round-trip TEXT -> str; bytes row round-trips BLOB -> bytes.
            assert v1 == "str-v1" and isinstance(v1, str)
            assert v2 == "str-v2" and isinstance(v2, str)
            assert v3 == b"\x00\x01bytes-v3" and isinstance(v3, bytes)

    def test_retention_off_returns_none_and_stores_nothing(self, db_path: Path) -> None:
        # Default (retain_versions=False): no capture, get_content_at_version None.
        with SqliteArtifactRegistry(db_path) as reg:
            art = _retain_artifact(reg, content="discarded")
            assert reg.get_content_at_version(art.id, 1) is None
            assert reg._conn.execute(
                "SELECT COUNT(*) FROM artifact_versions"
            ).fetchone()[0] == 0
            # The persisted marker says retention was never enabled.
            assert reg.retention_meta() == (False, None)

    def test_commit_cas_content_none_skips_capture(self, db_path: Path) -> None:
        # Mirror of the in-memory fix: a content=None WIN retains NO row for the
        # new version (no stale-old-body poisoning).
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            art = _retain_artifact(reg, content="seed-v1")
            w = uuid4()
            reg.set_agent_state(art.id, w, MESIState.SHARED, tick=1)
            updated, _ = reg.commit_cas(
                art.id, w, expected_version=1, content_hash="hN", content=None, tick=2
            )
            assert updated.version == 2
            assert reg.get_content_at_version(art.id, 2) is None
            assert reg.get_content_at_version(art.id, 1) == "seed-v1"

    def test_t_expiry_backdated_row_rejects_then_sweeps(self, db_path: Path) -> None:
        # Sqlite-arm-isolated T-expiry (the parity suite diagnoses divergence,
        # not the sqlite SQL mechanics): backdate captured_at DIRECTLY in
        # artifact_versions, assert read_at_version logically rejects
        # not_retained, then assert the next capture PHYSICALLY removes the row
        # (deletion piggybacks on capture-time GC).
        from ccs.coordinator.service import CoordinatorService
        from ccs.core.exceptions import NOT_RETAINED_REASON
        from ccs.core.types import VersionedReadRejection

        pol = RetentionPolicy(max_versions=100, max_age_seconds=600.0)
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=pol
        ) as reg:
            art = _retain_artifact(reg, content="c1")
            for v in (2, 3):
                nx = Artifact(id=art.id, name="plan.md", version=v, content_hash="h")
                reg.set_artifact_and_content(art.id, nx, f"c{v}")
            # Backdate v2 far past the 600s horizon (raw SQL on purpose: this
            # pins the sqlite captured_at-float comparison path in isolation).
            reg._conn.execute(
                "UPDATE artifact_versions SET captured_at = 1.0 "
                "WHERE artifact_id = ? AND version = 2",
                (art.id.hex,),
            )
            svc = CoordinatorService(reg)
            out = svc.read_at_version(art.id, 2)
            assert isinstance(out, VersionedReadRejection)
            assert out.reason == NOT_RETAINED_REASON
            # Logical-at-read: the row is still physically present.
            assert reg._conn.execute(
                "SELECT COUNT(*) FROM artifact_versions "
                "WHERE artifact_id = ? AND version = 2",
                (art.id.hex,),
            ).fetchone()[0] == 1
            # The next capture's inline GC physically removes the aged row.
            nx4 = Artifact(id=art.id, name="plan.md", version=4, content_hash="h")
            reg.set_artifact_and_content(art.id, nx4, "c4")
            assert reg._conn.execute(
                "SELECT COUNT(*) FROM artifact_versions "
                "WHERE artifact_id = ? AND version = 2",
                (art.id.hex,),
            ).fetchone()[0] == 0
            # Fresh rows are untouched by the sweep.
            assert reg.get_content_at_version(art.id, 3) == "c3"

    def test_unbounded_marker_persisted_when_policy_none(self, db_path: Path) -> None:
        # retain_versions=True + policy=None -> enabled with NULL axes, so a
        # resolver can tell retention-on-unbounded from retention-never-enabled.
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=None
        ) as reg:
            art = _retain_artifact(reg, content="c1")
            for v in (2, 3, 4):
                nx = Artifact(id=art.id, name="plan.md", version=v, content_hash="h")
                reg.set_artifact_and_content(art.id, nx, f"c{v}")
            # Unbounded: every version retained.
            assert reg.get_content_at_version(art.id, 1) == "c1"
            assert reg.get_content_at_version(art.id, 4) == "c4"
            assert reg.retention_meta() == (True, None)


class TestWildV1Migration:
    """R2: the four wild v1 variants ({±fence cols} x {±pending_notices}) each
    migrate (chained v1->v2->v3) in one open with data intact + complete schema
    (artifact_versions from v2, session_pins from v3)."""

    @pytest.mark.parametrize("fence", [True, False])
    @pytest.mark.parametrize("notices", [True, False])
    def test_wild_variant_migrates_to_v2(
        self, db_path: Path, fence: bool, notices: bool
    ) -> None:
        art_id, ag_id = _build_raw_v1_db(db_path, fence=fence, notices=notices)
        with SqliteArtifactRegistry(db_path) as reg:
            # Chained migration stamps the current version; complete schema guaranteed.
            assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
            tables = {
                r[0] for r in reg._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            assert {
                "artifact_versions", "pending_notices", "session_pins"
            }.issubset(tables)
            art_cols = {
                r[1] for r in reg._conn.execute(
                    "PRAGMA table_info(artifacts)"
                ).fetchall()
            }
            state_cols = {
                r[1] for r in reg._conn.execute(
                    "PRAGMA table_info(agent_states)"
                ).fetchall()
            }
            assert "owner_generation" in art_cols
            assert "read_generation" in state_cols
            # Pre-existing data intact.
            art = reg.get_artifact(UUID(art_id))
            assert art is not None and art.version == 3
            assert reg.instance_id == "wild-v1"
            assert reg._seq == 4
            # Fence queries work even on a PREVIOUSLY-pre-fence variant.
            assert reg.get_owner_generation(UUID(art_id)) == 0
            assert reg.get_read_generation(UUID(art_id), UUID(ag_id)) is None
            # A fence-present v1 keeps its epoch; a fence-absent one is seeded.
            if fence:
                assert reg._coordinator_epoch == "epoch-pre-migration"
            else:
                assert reg._coordinator_epoch  # freshly seeded hex

    def test_migrated_v1_supports_durable_capture(self, db_path: Path) -> None:
        # After migrating a pre-fence v1, retention works on the upgraded store.
        art_id, _ = _build_raw_v1_db(db_path, fence=False, notices=False)
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            nx = Artifact(id=UUID(art_id), name="plan.md", version=4, content_hash="h")
            reg.set_artifact_and_content(UUID(art_id), nx, "post-migration-body")
            assert reg.get_content_at_version(UUID(art_id), 4) == "post-migration-body"

    def test_half_migrated_db_completes_cleanly(self, db_path: Path) -> None:
        # Characterization: a db with the v2 table already present but
        # user_version still 1 (a crashed prior migration whose stamp never
        # committed) must complete via the chained v1->v2->v3 path to v3, never
        # "table already exists".
        _build_raw_v1_db(db_path, fence=True, notices=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(_ARTIFACT_VERSIONS_DDL)  # table present; user_version still 1
        conn.commit()
        conn.close()
        # Open via the normal path: must not raise OperationalError.
        with SqliteArtifactRegistry(db_path) as reg:
            assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION

    def test_half_migrated_db_missing_stamp_recovers(self, db_path: Path) -> None:
        # The classic learnings case generalized: all v2 tables present but
        # user_version left at 1. The migration's IF-NOT-EXISTS / table_info
        # guards make it idempotent — clean completion through to v3, no error.
        _build_raw_v1_db(db_path, fence=True, notices=True)
        conn = sqlite3.connect(str(db_path))
        conn.execute(_ARTIFACT_VERSIONS_DDL)
        conn.commit()
        conn.close()
        reg = SqliteArtifactRegistry(db_path)  # must not raise
        try:
            assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        finally:
            reg.close()

    def test_concurrent_loser_re_checks_in_txn_and_runs_no_ddl(
        self, db_path: Path, monkeypatch
    ) -> None:
        # Two processes can both read user_version==1 in the dispatch (outside
        # any txn) and both enter _migrate_v1_to_v2; they serialize on BEGIN
        # IMMEDIATE. Simulate the LOSER deterministically: the db is already
        # migrated (the "winner" ran), but the dispatch decision was made from
        # the stale pre-lock read — force it down the migration path and
        # assert the in-txn user_version re-check no-ops with ZERO DDL.
        art_id, _ = _build_raw_v1_db(db_path, fence=True, notices=True)
        with SqliteArtifactRegistry(db_path) as winner:
            assert winner._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION

        statements: list[str] = []

        def loser_initialize(self, instance_id):
            # The loser's stale dispatch: call the v1->v2 migration directly
            # against the already-migrated (v3) db, tracing every statement. The
            # v1->v2 loser-guard (>= _V2_USER_VERSION) sees v3 and short-circuits
            # with zero DDL before any ALTER/CREATE could re-run.
            self._conn.set_trace_callback(statements.append)
            try:
                self._migrate_v1_to_v2(instance_id)
            finally:
                self._conn.set_trace_callback(None)

        monkeypatch.setattr(
            SqliteArtifactRegistry, "_initialize_schema", loser_initialize
        )
        with SqliteArtifactRegistry(db_path) as loser:
            # The loser rehydrated normally (meta loaded, data intact) ...
            assert loser.instance_id == "wild-v1"
            assert loser.get_artifact(UUID(art_id)) is not None
            assert loser._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        # ... and performed NO DDL: the in-txn re-check exited before any
        # ALTER/CREATE could re-run against the migrated schema.
        ddl = [
            s for s in statements
            if s.lstrip().upper().startswith(("ALTER", "CREATE"))
        ]
        assert ddl == [], f"concurrent loser re-ran DDL: {ddl}"


# ----------------------------------------------------------------------
# Cross-runtime fail-closed guard (sibling Node coordinator ledger)
# ----------------------------------------------------------------------


def _assert_no_delete_advice(message: str) -> None:
    """The anti-delete wording rule (SchemaVersionError docstring): a store-open
    failure message must never advise deleting state.db — the file holds
    durable retained content and, for a foreign-ledger db, the sibling
    runtime's live state. The Node coordinator's own KTD-D message DOES say
    ``rm <workspace>/.coherence/state.db``; assert that advice shape never
    leaks into this side's cross-runtime message. A strict substring ban is
    safe here because the message is written without the verbs entirely
    (unlike SchemaVersionError's, which negates them: 'rather than removing').
    """
    lowered = message.lower()
    for fragment in ("rm ", "delet", "remov", "discard", "unlink"):
        assert fragment not in lowered, (
            f"cross-runtime message contains {fragment!r} (reads as delete "
            f"advice): {message}"
        )


def _forge_node_v3_db(db_path: Path) -> None:
    """Forge what the sibling Node coordinator's ledger produces at ITS v3:
    Node v1 is a byte-for-byte mirror of this repo's v1 DDL (pending_notices
    included, no fence columns), Node v2 adds no schema objects, Node v3 is
    ``ALTER TABLE agent_states ADD COLUMN deadline_tick`` + user_version=3."""
    _build_raw_v1_db(db_path, fence=False, notices=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("ALTER TABLE agent_states ADD COLUMN deadline_tick INTEGER")
        conn.execute("PRAGMA user_version = 3")
        conn.commit()
    finally:
        conn.close()


def _forge_node_v2_db(db_path: Path) -> None:
    """Forge a Node-v2 db: this repo's v1 DDL with user_version=2 and NO
    artifact_versions table (the Node ledger's v2 adds no schema objects; a
    Python-v2 db ALWAYS has artifact_versions)."""
    _build_raw_v1_db(db_path, fence=False, notices=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA user_version = 2")
        conn.commit()
    finally:
        conn.close()


def _stamp_schema_runtime(db_path: Path, value: str) -> None:
    """Hand-stamp registry_meta.schema_runtime (forging the Node side's mirror
    stamp, or clearing ours via DELETE elsewhere)."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT OR REPLACE INTO registry_meta (key, value) "
            "VALUES ('schema_runtime', ?)",
            (value,),
        )
        conn.commit()
    finally:
        conn.close()


def _user_version(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


class TestCrossRuntimeGuard:
    """Fail-closed open against a sibling-Node-ledger state.db (both open
    paths). The Node coordinator shares the db path but its ledger assigns
    different meanings to user_version 2/3, so the guard must fire BEFORE any
    migration or rehydrate write — extends the wild-v1 migration and
    read-only open coverage above."""

    @pytest.mark.parametrize("read_only", [False, True])
    def test_node_v3_db_raises_cross_runtime(
        self, db_path: Path, read_only: bool
    ) -> None:
        _forge_node_v3_db(db_path)
        with pytest.raises(CrossRuntimeSchemaError) as ei:
            SqliteArtifactRegistry(db_path, read_only=read_only)
        # Typed signal: match the wire-stable constant with ==, never the text.
        assert ei.value.reason == CROSS_RUNTIME_SCHEMA_REASON
        msg = str(ei.value)
        # The supported backend-switch path is named; deletion never advised.
        assert "agent-coherence-coordinator --prepare-for-migration" in msg
        _assert_no_delete_advice(msg)
        # Fail-closed means NO migration/write happened: still the Node shape.
        assert _user_version(db_path) == 3

    @pytest.mark.parametrize("read_only", [False, True])
    def test_foreign_v2_db_raises_cross_runtime(
        self, db_path: Path, read_only: bool
    ) -> None:
        _forge_node_v2_db(db_path)
        with pytest.raises(CrossRuntimeSchemaError) as ei:
            SqliteArtifactRegistry(db_path, read_only=read_only)
        assert ei.value.reason == CROSS_RUNTIME_SCHEMA_REASON
        _assert_no_delete_advice(str(ei.value))
        assert _user_version(db_path) == 2

    @pytest.mark.parametrize("read_only", [False, True])
    @pytest.mark.parametrize("forge_version", [1, 2])
    def test_schema_runtime_node_stamp_raises_regardless_of_version(
        self, db_path: Path, read_only: bool, forge_version: int
    ) -> None:
        # The explicit stamp is the STRONGEST marker: it must reject even a
        # db whose structure would otherwise pass — a genuine v1 (normally
        # migratable) and a structurally perfect Python v2.
        if forge_version == 1:
            _build_raw_v1_db(db_path, fence=False, notices=True)
        else:
            with SqliteArtifactRegistry(db_path):
                pass
        _stamp_schema_runtime(db_path, "node")
        before = _user_version(db_path)
        with pytest.raises(CrossRuntimeSchemaError) as ei:
            SqliteArtifactRegistry(db_path, read_only=read_only)
        assert ei.value.reason == CROSS_RUNTIME_SCHEMA_REASON
        _assert_no_delete_advice(str(ei.value))
        # NO migration ran on the stamped-foreign db (v1 stays v1).
        assert _user_version(db_path) == before

    def test_genuine_v1_migrates_and_stamps_python(self, db_path: Path) -> None:
        # The documented residual risk, accepted by design: a v1 db is
        # byte-identical between the two runtimes, so it is NOT blocked — the
        # normal chained v1->v2->v3 migration proceeds and stamps OUR lineage
        # marker.
        art_id, _ = _build_raw_v1_db(db_path, fence=False, notices=False)
        with SqliteArtifactRegistry(db_path) as reg:
            assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
            assert reg.get_artifact(UUID(art_id)) is not None
            stamp = reg._conn.execute(
                "SELECT value FROM registry_meta WHERE key = 'schema_runtime'"
            ).fetchone()
            assert stamp is not None and stamp[0] == "python"

    def test_fresh_db_stamps_python(self, db_path: Path) -> None:
        with SqliteArtifactRegistry(db_path) as reg:
            stamp = reg._conn.execute(
                "SELECT value FROM registry_meta WHERE key = 'schema_runtime'"
            ).fetchone()
            assert stamp is not None and stamp[0] == "python"

    def test_genuine_python_v2_reopens_fine_both_paths(self, db_path: Path) -> None:
        # A db this build created (stamped python) passes the guard on the
        # writer AND read-only paths.
        with SqliteArtifactRegistry(db_path) as reg:
            _retain_artifact(reg, content="x")
        with SqliteArtifactRegistry(db_path) as rw:
            assert rw._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
        with SqliteArtifactRegistry(db_path, read_only=True) as ro:
            assert ro.instance_id

    def test_pre_stamp_python_v2_opens_fine(self, db_path: Path) -> None:
        # Back-compat: a Python db written BEFORE the schema_runtime marker
        # shipped has no stamp at all. Absence is NOT foreign (the steady-state
        # v3 open path is write-free, so the stamp is never backfilled either).
        # The structural session_pins probe (present on a Python-v3 db) keeps the
        # un-stamped db classified as native, never foreign.
        with SqliteArtifactRegistry(db_path):
            pass
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DELETE FROM registry_meta WHERE key = 'schema_runtime'")
            conn.commit()
        finally:
            conn.close()
        with SqliteArtifactRegistry(db_path) as reg:
            assert reg._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
            # Still un-stamped: the steady-state v3 open performs no writes.
            assert reg._conn.execute(
                "SELECT value FROM registry_meta WHERE key = 'schema_runtime'"
            ).fetchone() is None


class TestRetentionPolicyAcrossRuns:
    """R2/R4: K-lowered-between-runs (no open-time GC) + disable != purge."""

    def test_k_lowered_between_runs_persists_until_next_capture(
        self, db_path: Path
    ) -> None:
        art_id = None
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=RetentionPolicy(max_versions=10)
        ) as reg:
            art = _retain_artifact(reg, content="c1")
            art_id = art.id
            for v in range(2, 6):
                nx = Artifact(id=art_id, name="plan.md", version=v, content_hash="h")
                reg.set_artifact_and_content(art_id, nx, f"c{v}")
            assert _versions(reg, art_id) == [1, 2, 3, 4, 5]

        # Reopen with a lower K: rows persist (NO open-time GC pass). The
        # policy change is deliberate here — acknowledge the mismatch warning.
        with pytest.warns(RuntimeWarning, match="retention policy mismatch"):
            reg = SqliteArtifactRegistry(
                db_path, retain_versions=True, retention_policy=RetentionPolicy(max_versions=2)
            )
        with reg:
            assert _versions(reg, art_id) == [1, 2, 3, 4, 5]
            # Next capture for THIS artifact prunes to the new K.
            nx = Artifact(id=art_id, name="plan.md", version=6, content_hash="h")
            reg.set_artifact_and_content(art_id, nx, "c6")
            assert _versions(reg, art_id) == [5, 6]

    def test_same_policy_reopen_performs_no_write(self, db_path: Path) -> None:
        # Write-free steady-state: a reopen with the SAME policy must not
        # UPSERT the (unchanged) retention keys — the v2 open path stays
        # write-free end-to-end. total_changes counts every row this
        # connection inserted/updated/deleted; zero == no write happened.
        pol = RetentionPolicy(max_versions=4, max_age_seconds=60.0)
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=pol
        ) as reg:
            _retain_artifact(reg, content="seed")

        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=pol
        ) as reg2:
            assert reg2._conn.total_changes == 0, (
                "same-policy reopen wrote to the store (the retention-key "
                "UPSERT must be skipped when persisted values already match)"
            )
            # The cached meta still reflects the persisted policy.
            assert reg2.retention_meta() == (True, pol)

        # A CHANGED policy does write (the skip is equality-gated, not
        # blanket) — and warns about the mismatch before overwriting.
        with pytest.warns(RuntimeWarning, match="retention policy mismatch"):
            reg3 = SqliteArtifactRegistry(
                db_path, retain_versions=True,
                retention_policy=RetentionPolicy(max_versions=2),
            )
        with reg3:
            assert reg3._conn.total_changes > 0
            assert reg3.retention_meta() == (True, RetentionPolicy(max_versions=2))

    def test_disable_retention_preserves_existing_rows(self, db_path: Path) -> None:
        # disable != purge: reopening with retain_versions=False over existing
        # rows preserves them (capture stops; rows stay readable).
        art_id = None
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            art = _retain_artifact(reg, content="kept")
            art_id = art.id

        with SqliteArtifactRegistry(db_path, retain_versions=False) as reg:
            assert reg.get_content_at_version(art_id, 1) == "kept"
            # The marker now records retention OFF, but the rows survive.
            assert reg.retention_meta() == (False, None)
            # A further write captures NOTHING (retention off).
            nx = Artifact(id=art_id, name="plan.md", version=2, content_hash="h")
            reg.set_artifact_and_content(art_id, nx, "not-captured")
            assert reg.get_content_at_version(art_id, 2) is None
            assert reg.get_content_at_version(art_id, 1) == "kept"

    def test_policy_mismatch_reopen_warns_naming_both_policies(
        self, db_path: Path
    ) -> None:
        # A writer whose constructor policy differs from the persisted one
        # warns ONCE before overwriting — the single-writer-embedder
        # assumption made operator-visible. Both policies appear in the text.
        with SqliteArtifactRegistry(
            db_path,
            retain_versions=True,
            retention_policy=RetentionPolicy(max_versions=8, max_age_seconds=60.0),
        ):
            pass
        with pytest.warns(
            RuntimeWarning,
            match=(
                r"retention policy mismatch.*"
                r"max_versions=8 / max_age_seconds=60\.0.*"
                r"max_versions=2 / max_age_seconds=None.*"
                r"SINGLE writer-embedder"
            ),
        ):
            reg = SqliteArtifactRegistry(
                db_path,
                retain_versions=True,
                retention_policy=RetentionPolicy(max_versions=2),
            )
        try:
            # Warn-then-overwrite: the constructor policy DID land.
            assert reg.retention_meta() == (True, RetentionPolicy(max_versions=2))
        finally:
            reg.close()

    def test_policy_first_persist_match_and_read_only_stay_silent(
        self, db_path: Path
    ) -> None:
        # No warning on: first persist (nothing persisted yet), a matching
        # reopen (skip-if-unchanged), and a read-only open (never persists) —
        # escalate any RuntimeWarning to an error to prove silence.
        pol = RetentionPolicy(max_versions=4, max_age_seconds=60.0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with SqliteArtifactRegistry(
                db_path, retain_versions=True, retention_policy=pol
            ):
                pass
            with SqliteArtifactRegistry(
                db_path, retain_versions=True, retention_policy=pol
            ):
                pass
            # Read-only with a DIVERGENT constructor policy: still silent —
            # a reader never persists, so there is no overwrite to warn about.
            with SqliteArtifactRegistry(
                db_path,
                read_only=True,
                retain_versions=True,
                retention_policy=RetentionPolicy(max_versions=1),
            ):
                pass


class TestRetentionCrashAtomicity:
    """R2: a BaseException mid-transaction leaves no phantom history AND no
    version bump — capture and commit are one atomic unit."""

    def test_commit_cas_win_log_raise_rolls_back_history_and_version(
        self, db_path: Path
    ) -> None:
        # A state_log raise during the committer's WIN log entry must roll back
        # the whole txn: no version move, no captured next_version row.
        def boom(entry):
            if entry["to_state"] == "SHARED" and entry["trigger"] == "commit_cas":
                raise RuntimeError("boom in committer log")

        iid = str(uuid4())
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8,
            state_log=boom, instance_id=iid,
        ) as reg:
            art = _retain_artifact(reg, content="seed-v1")
            w = uuid4()
            reg.set_agent_state(art.id, w, MESIState.SHARED, tick=1)
            with pytest.raises(RuntimeError):
                reg.commit_cas(
                    art.id, w, expected_version=1, content_hash="hN",
                    content="cas-v2", tick=2,
                )
            # No version bump, no phantom history, _seq consistent.
            assert reg.get_artifact(art.id).version == 1
            assert reg.get_content_at_version(art.id, 2) is None
            assert reg.get_content_at_version(art.id, 1) == "seed-v1"

    def test_fence_reject_leaves_no_capture(self, db_path: Path) -> None:
        # The pessimistic fence rejects a superseded committer BEFORE the
        # capture, so no phantom history row for the rejected version.
        from ccs.core.exceptions import StaleReadGeneration

        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            art = _retain_artifact(reg, content="seed")
            a = uuid4()
            reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
            reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=2)
            stale = Artifact(id=art.id, name="plan.md", version=2, content_hash="stale")
            with pytest.raises(StaleReadGeneration):
                reg.set_artifact_and_content(
                    art.id, stale, "should-not-store", last_writer=a, fence_agent_id=a
                )
            assert reg.get_content_at_version(art.id, 2) is None


class TestReadOnlyMode:
    """R2: read-only open never creates / migrates / mutates; typed failures."""

    def test_missing_path_raises_and_creates_no_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "absent.db"
        with pytest.raises(MissingDatabaseError):
            SqliteArtifactRegistry(missing, read_only=True)
        assert not missing.exists()  # NO fresh db materialized

    def test_v1_db_raises_version_error_no_migration(self, db_path: Path) -> None:
        _build_raw_v1_db(db_path, fence=True, notices=True)
        with pytest.raises(SchemaVersionError):
            SqliteArtifactRegistry(db_path, read_only=True)
        # The read-only open performed NO migration.
        conn = sqlite3.connect(str(db_path))
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        finally:
            conn.close()

    def test_read_only_reads_durable_rows(self, db_path: Path) -> None:
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            art = _retain_artifact(reg, content="ro-body")
            art_id = art.id
            epoch = reg.coordinator_epoch
        with SqliteArtifactRegistry(db_path, read_only=True) as ro:
            assert ro.get_content_at_version(art_id, 1) == "ro-body"
            assert ro.coordinator_epoch == epoch  # same epoch the writer minted
            assert ro.retention_meta()[0] is True

    def test_read_only_mutator_raises(self, db_path: Path) -> None:
        with SqliteArtifactRegistry(db_path) as reg:
            _retain_artifact(reg, content="x")
        with SqliteArtifactRegistry(db_path, read_only=True) as ro:
            with pytest.raises(ReadOnlyMutationError):
                ro.register_artifact(
                    Artifact(id=uuid4(), name="y", version=1, content_hash="h"), content="x"
                )
            with pytest.raises(ReadOnlyMutationError):
                ro.set_agent_state(uuid4(), uuid4(), MESIState.SHARED)
            with pytest.raises(ReadOnlyMutationError):
                ro.remove_artifact(uuid4())

    def test_read_only_needs_recovery_fault_injected(
        self, db_path: Path, monkeypatch
    ) -> None:
        # A real post-crash hot-WAL state cannot be deterministically created in
        # pytest, so inject SQLITE_READONLY_RECOVERY on the ro connection's first
        # user_version read (the point a hot-WAL ro open fails).
        with SqliteArtifactRegistry(db_path) as reg:
            _retain_artifact(reg, content="x")

        from ccs.coordinator import sqlite_registry as sr

        class _RecoveryProxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if "user_version" in sql:
                    raise sqlite3.OperationalError(
                        "attempt to write a readonly database "
                        "(SQLITE_READONLY_RECOVERY)"
                    )
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _RecoveryProxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        with pytest.raises(StoreNeedsRecoveryError):
            SqliteArtifactRegistry(db_path, read_only=True)

    @pytest.mark.parametrize(
        ("build_store", "fault_message", "expected_error"),
        [
            pytest.param("v1", None, SchemaVersionError, id="schema-version-mismatch"),
            pytest.param(
                "v2",
                "attempt to write a readonly database (SQLITE_READONLY_RECOVERY)",
                StoreNeedsRecoveryError,
                id="needs-recovery",
            ),
            pytest.param(
                "v2", "database is locked", StoreNeedsRecoveryError, id="busy"
            ),
        ],
    )
    def test_read_only_open_failure_closes_connection(
        self, db_path: Path, monkeypatch, build_store, fault_message, expected_error
    ) -> None:
        # A typed failure AFTER sqlite3.connect succeeded must close the
        # connection before raising — the caller never receives a registry
        # handle, so a leaked conn would hold the db open for the process
        # lifetime (and pin the WAL on platforms that block unlink).
        if build_store == "v1":
            _build_raw_v1_db(db_path, fence=True, notices=True)
        else:
            with SqliteArtifactRegistry(db_path) as reg:
                _retain_artifact(reg, content="x")

        from ccs.coordinator import sqlite_registry as sr

        closes: list[bool] = []

        class _Proxy:
            def __init__(self, real):
                self._real = real

            def close(self):
                closes.append(True)
                return self._real.close()

            def execute(self, sql, *a, **k):
                if fault_message is not None and "user_version" in sql:
                    raise sqlite3.OperationalError(fault_message)
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _Proxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        with pytest.raises(expected_error):
            SqliteArtifactRegistry(db_path, read_only=True)
        assert closes == [True], "typed read-only open failure leaked the connection"

    def test_read_only_busy_open_carries_busy_reason(
        self, db_path: Path, monkeypatch
    ) -> None:
        # The typed signal (not the message) is the routing contract: a locked
        # store classifies reason == STORE_SIGNAL_BUSY at the registry's single
        # classification point.
        from ccs.core.exceptions import STORE_SIGNAL_BUSY

        with SqliteArtifactRegistry(db_path) as reg:
            _retain_artifact(reg, content="x")

        from ccs.coordinator import sqlite_registry as sr

        class _BusyProxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if "user_version" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _BusyProxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        with pytest.raises(StoreNeedsRecoveryError) as excinfo:
            SqliteArtifactRegistry(db_path, read_only=True)
        assert excinfo.value.reason == STORE_SIGNAL_BUSY


class TestRetentionRemoveAndEpoch:
    """R4: delete cascades history; re-register mints new UUID; epoch reset."""

    def test_remove_artifact_cascades_history(self, db_path: Path) -> None:
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            art = _retain_artifact(reg, content="seed")
            for v in (2, 3):
                nx = Artifact(id=art.id, name="plan.md", version=v, content_hash="h")
                reg.set_artifact_and_content(art.id, nx, f"c{v}")
            assert reg._conn.execute(
                "SELECT COUNT(*) FROM artifact_versions WHERE artifact_id = ?",
                (art.id.hex,),
            ).fetchone()[0] == 3
            reg.remove_artifact(art.id)
            # ON DELETE CASCADE dropped the retained rows.
            assert reg._conn.execute(
                "SELECT COUNT(*) FROM artifact_versions WHERE artifact_id = ?",
                (art.id.hex,),
            ).fetchone()[0] == 0
            assert reg.get_content_at_version(art.id, 1) is None

    def test_delete_then_reregister_same_name_fresh_history(self, db_path: Path) -> None:
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            art = _retain_artifact(reg, content="old")
            old_id = art.id
            reg.remove_artifact(old_id)
            # Re-register the SAME name mints a new UUID with fresh history.
            art2 = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
            assert art2.id != old_id
            reg.register_artifact(art2, content="new")
            assert reg.get_content_at_version(art2.id, 1) == "new"
            assert reg.get_content_at_version(old_id, 1) is None  # old id gone

    def test_epoch_reset_drops_rows(self, db_path: Path) -> None:
        # Epoch reset = delete-and-recreate the db; retained rows do not survive.
        art_id = None
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            art = _retain_artifact(reg, content="pre-reset")
            art_id = art.id
            epoch1 = reg.coordinator_epoch
        # Simulate the operator purge: remove the db (+ sidecars) and recreate.
        db_path.unlink()
        for sidecar in (db_path.with_name(db_path.name + "-wal"),
                        db_path.with_name(db_path.name + "-shm")):
            if sidecar.exists():
                sidecar.unlink()
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            assert reg.coordinator_epoch != epoch1  # fresh epoch
            assert reg.get_content_at_version(art_id, 1) is None  # rows gone


class TestRetentionFileModes:
    """R2: state.db + sidecars are 0600, race-free; migration tightens + warns."""

    def test_fresh_db_is_0600_immediately(self, db_path: Path) -> None:
        with SqliteArtifactRegistry(db_path) as reg:
            mode = stat.S_IMODE(db_path.stat().st_mode)
            assert mode == _DB_FILE_MODE, f"db mode {oct(mode)} != 0600"

    def test_sidecars_are_0600_after_writes(self, db_path: Path) -> None:
        with SqliteArtifactRegistry(
            db_path, retain_versions=True, retention_policy=_K8
        ) as reg:
            # Force WAL frames so the sidecars materialize.
            _retain_artifact(reg, content="x")
            wal = db_path.with_name(db_path.name + "-wal")
            shm = db_path.with_name(db_path.name + "-shm")
            for sidecar in (wal, shm):
                if sidecar.exists():
                    assert stat.S_IMODE(sidecar.stat().st_mode) == _DB_FILE_MODE, (
                        f"{sidecar.name} mode "
                        f"{oct(stat.S_IMODE(sidecar.stat().st_mode))} != 0600"
                    )

    def test_umask_window_db_and_sidecars_0600_from_creation(
        self, db_path: Path, monkeypatch
    ) -> None:
        # The no-umask-window claim, pinned: under a permissive umask (0o022)
        # all three files (db + -wal + -shm) must ALREADY exist at 0600 when
        # sqlite3.connect first sees the path — i.e. they were pre-created by
        # os.open(..., 0o600), never materialized by sqlite under the umask.
        # A regression to open()+chmod (or to post-connect pre-creation) makes
        # the at-connect snapshot miss files or show a broader mode.
        from ccs.coordinator import sqlite_registry as sr

        wal = db_path.with_name(db_path.name + "-wal")
        shm = db_path.with_name(db_path.name + "-shm")
        modes_at_connect: dict[str, int] = {}
        real_connect = sqlite3.connect

        def spying_connect(*a, **k):
            if not modes_at_connect:  # first (writer) connect only
                for p in (db_path, wal, shm):
                    if p.exists():
                        modes_at_connect[p.name] = stat.S_IMODE(p.stat().st_mode)
            return real_connect(*a, **k)

        monkeypatch.setattr(sr.sqlite3, "connect", spying_connect)
        old_umask = os.umask(0o022)
        try:
            with SqliteArtifactRegistry(
                db_path, retain_versions=True, retention_policy=_K8
            ) as reg:
                _retain_artifact(reg, content="x")  # force WAL frames
        finally:
            os.umask(old_umask)
        # All three existed at 0600 BEFORE sqlite touched the path ...
        assert modes_at_connect == {
            db_path.name: 0o600,
            wal.name: 0o600,
            shm.name: 0o600,
        }, f"files at connect time: {modes_at_connect}"
        # ... and end at 0600 after real writes under the permissive umask.
        for p in (db_path, wal, shm):
            if p.exists():
                assert stat.S_IMODE(p.stat().st_mode) == _DB_FILE_MODE

    def test_migration_tightens_broadened_db_and_warns_once(
        self, db_path: Path, caplog
    ) -> None:
        # A pre-existing v1 db whose mode an operator broadened (0644) is
        # tightened to 0600 by the migration, with a one-time warning.
        _build_raw_v1_db(db_path, fence=True, notices=True)
        os.chmod(db_path, 0o644)
        import logging

        with caplog.at_level(logging.WARNING, logger="ccs.coordinator.sqlite_registry"):
            with SqliteArtifactRegistry(db_path) as reg:
                pass
        assert stat.S_IMODE(db_path.stat().st_mode) == _DB_FILE_MODE
        warnings = [
            r for r in caplog.records if "v1->v2 migration tightened" in r.message
        ]
        assert len(warnings) == 1, "expected exactly one posture-change warning"


def _versions(reg, artifact_id) -> list[int]:
    """Sorted retained versions for an artifact (sqlite test helper)."""
    return sorted(
        r[0] for r in reg._conn.execute(
            "SELECT version FROM artifact_versions WHERE artifact_id = ?",
            (artifact_id.hex,),
        ).fetchall()
    )


class TestV3ToV4Migration:
    """The durable-owner fix added ``session_meta`` + ``idx_session_pins_artifact``
    to the schema. Earlier commits of THIS branch already stamped
    ``user_version=3`` with ``session_pins`` but WITHOUT ``session_meta``; such a
    db MUST migrate to v4 on open (gaining both) rather than crash on the first
    session op with 'no such table: session_meta'. Regression for the re-review
    P0: F3/F6 added tables to v3 in-place without bumping the schema version."""

    def _revert_to_prefix_v3(self, db_path: Path) -> None:
        """Mutate a current (v4) db on disk back to the pre-fix v3 shape:
        ``session_pins`` present, ``session_meta`` + the artifact index ABSENT,
        stamped ``user_version=3`` (what an earlier branch commit produced)."""
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DROP TABLE IF EXISTS session_meta")
            conn.execute("DROP INDEX IF EXISTS idx_session_pins_artifact")
            conn.execute("PRAGMA user_version = 3")
            conn.commit()
        finally:
            conn.close()

    def test_existing_v3_db_migrates_to_v4_and_session_ops_work(
        self, db_path: Path
    ) -> None:
        with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
            art = _retain_artifact(reg, content="seed")
        self._revert_to_prefix_v3(db_path)
        # Sanity: the reverted db is a genuine pre-fix v3 (session_pins, no meta).
        probe = sqlite3.connect(str(db_path))
        try:
            assert probe.execute("PRAGMA user_version").fetchone()[0] == 3
            assert probe.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_meta'"
            ).fetchone() is None
            assert probe.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_pins'"
            ).fetchone() is not None
        finally:
            probe.close()
        # Reopen with the current build → _migrate_v3_to_v4 fires (no crash).
        with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
            assert reg._conn.execute(
                "PRAGMA user_version"
            ).fetchone()[0] == SCHEMA_USER_VERSION
            assert reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='session_meta'"
            ).fetchone() is not None
            assert reg._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name='idx_session_pins_artifact'"
            ).fetchone() is not None
            # The session op that would have crashed pre-migration now works.
            owner = uuid4()
            cut = reg.capture_version_vector(
                [art.id], "tok-migrated", owner=owner, created_at_tick=5
            )
            assert cut == {art.id: art.version}
            assert reg.get_session_meta("tok-migrated") == (owner, 5)
            assert reg.session_count() == 1

    def test_v2_db_chains_through_to_v4_with_session_meta(self, db_path: Path) -> None:
        # A v2 db (artifact_versions, no session_pins/meta) chains 2 -> 3 -> 4 and
        # ends with the FULL current schema (both session tables + the index).
        with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
            _retain_artifact(reg, content="seed")
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("DROP TABLE IF EXISTS session_meta")
            conn.execute("DROP INDEX IF EXISTS idx_session_pins_artifact")
            conn.execute("DROP TABLE IF EXISTS session_pins")
            conn.execute("PRAGMA user_version = 2")
            conn.commit()
        finally:
            conn.close()
        with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
            assert reg._conn.execute(
                "PRAGMA user_version"
            ).fetchone()[0] == SCHEMA_USER_VERSION
            for table in ("session_pins", "session_meta"):
                assert reg._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone() is not None, table


# --------------------------------------------------------------------
# status_snapshot agent scoping (SB-10 review)
# --------------------------------------------------------------------


def test_status_snapshot_unscoped_reads_every_agent(db_path: Path) -> None:
    """The /status caller passes no scope and keeps the whole-ledger view."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        a1, a2 = uuid4(), uuid4()
        reg.set_agent_state(art.id, a1, MESIState.SHARED, tick=1)
        reg.set_agent_state(art.id, a2, MESIState.EXCLUSIVE, tick=2)

        artifact_by_id, state_by_artifact = reg.status_snapshot()
        assert set(artifact_by_id) == {art.id}
        assert state_by_artifact[art.id] == {
            a1: MESIState.SHARED,
            a2: MESIState.EXCLUSIVE,
        }


def test_status_snapshot_carries_last_writer_and_updated_at(db_path: Path) -> None:
    """#199 §2: the writer rides the batched artifact row so /status never
    falls back to per-artifact ``last_writer_for`` calls."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        artifact_by_id, _ = reg.status_snapshot()
        assert artifact_by_id[art.id]["last_writer_id"] is None
        assert artifact_by_id[art.id]["updated_at"] == reg.get_artifact_updated_at(art.id)

        writer = uuid4()
        new_art = Artifact(id=art.id, name=art.name, version=2, content_hash="h2")
        reg.set_artifact_and_content(art.id, new_art, content="", last_writer=writer)
        artifact_by_id, _ = reg.status_snapshot()
        assert artifact_by_id[art.id]["last_writer_id"] == writer == reg.last_writer_for(art.id)
        assert artifact_by_id[art.id]["updated_at"] == reg.get_artifact_updated_at(art.id)


def test_status_snapshot_scoped_reads_only_the_named_agents(db_path: Path) -> None:
    """SB-10 review: the session-start builder runs on a HOOK path, twice per
    compaction, under the registry lock — and can only ever render its own
    session's agents. Nothing garbage-collects ``agent_states``, so an
    unscoped read would drag the workspace's entire coordination history
    through that lock to discard all but a handful of rows."""
    with SqliteArtifactRegistry(db_path) as reg:
        mine = _make_artifact(name="mine.md")
        theirs = _make_artifact(name="theirs.md", content_hash="h2")
        reg.register_artifact(mine, content="")
        reg.register_artifact(theirs, content="")
        ours, foreign = uuid4(), uuid4()
        reg.set_agent_state(mine.id, ours, MESIState.SHARED, tick=1)
        reg.set_agent_state(mine.id, foreign, MESIState.EXCLUSIVE, tick=2)
        reg.set_agent_state(theirs.id, foreign, MESIState.SHARED, tick=3)

        artifact_by_id, state_by_artifact = reg.status_snapshot(agent_ids=[ours])
        # Artifact metadata stays workspace-wide on purpose: SB-10's R8
        # breadcrumb asks whether the WORKSPACE holds state, not this session.
        assert set(artifact_by_id) == {mine.id, theirs.id}
        assert state_by_artifact[mine.id] == {ours: MESIState.SHARED}
        assert state_by_artifact[theirs.id] == {}


def test_status_snapshot_scoped_to_no_agents_reads_no_state(db_path: Path) -> None:
    """An empty scope is 'no agents', never 'every agent' — the degenerate
    case must not silently widen back to a full scan."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact()
        reg.register_artifact(art, content="")
        reg.set_agent_state(art.id, uuid4(), MESIState.SHARED, tick=1)

        artifact_by_id, state_by_artifact = reg.status_snapshot(agent_ids=[])
        assert set(artifact_by_id) == {art.id}
        assert state_by_artifact == {art.id: {}}


def test_status_snapshot_scoped_query_is_index_backed(db_path: Path) -> None:
    """SB-10 perf follow-up: the scoped state read must SEARCH via
    ``idx_agent_states_agent``, never SCAN. Without the index the scope only
    trims per-row materialization — the page walk over a table nothing
    garbage-collects still runs on the hook path, and dropping the index in a
    later migration would regress it silently (every query stays correct)."""
    with SqliteArtifactRegistry(db_path) as reg:
        plan = " | ".join(
            r[3]
            for r in reg._conn.execute(
                "EXPLAIN QUERY PLAN SELECT artifact_id, agent_id, state "
                "FROM agent_states WHERE agent_id IN (?, ?)",
                ("a" * 32, "b" * 32),
            ).fetchall()
        )
    assert "idx_agent_states_agent" in plan, plan
    assert "SCAN agent_states" not in plan, plan
