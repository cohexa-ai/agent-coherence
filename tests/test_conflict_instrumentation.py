# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""U5 — conflict-outcome instrumentation (guarantee-ladder plan, R-4 / KTD-6/9).

The typed deny outcomes (``version_mismatch`` / ``other_holder`` /
``stale_read_generation``) are counted where they are constructed — inside the
registries, in the same transaction/branch that returns them — persisted in the
registry's own store so counts survive coordinator idle-shutdown respawns
(KTD-9), and readable offline for the 30-day report (KTD-6). No user-facing
warning exists anywhere on this path (the demand probes' verdict: instrument,
never warn); zero is a reportable result. Attribution is by agent identity
only — no host identifier reaches the commit path today, and the report says
so rather than inventing one.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.states import MESIState
from ccs.core.types import Artifact, CommitAllEntry, ConflictDetail, MultiCommitConflict
from ccs.diagnose.conflict_counters import read_conflict_totals


def _mk_artifact() -> Artifact:
    return Artifact(name=f"{uuid4().hex}.md", version=1, content_hash="seed")


def _seed(reg, artifact: Artifact, *agents: UUID) -> None:
    reg.register_artifact(artifact, "seed-content")
    for agent in agents:
        reg.set_agent_state(artifact.id, agent, MESIState.SHARED, trigger="fetch", tick=0)


# ---------------------------------------------------------------------------
# Counting at the construction site
# ---------------------------------------------------------------------------


def test_version_mismatch_increments_exactly_one_cell(registry) -> None:
    art = _mk_artifact()
    agent = uuid4()
    _seed(registry, art, agent)
    result = registry.commit_cas(
        art.id, agent, expected_version=0, content_hash="h"
    )
    assert isinstance(result, ConflictDetail) and result.reason == "version_mismatch"
    assert registry.conflict_outcome_totals() == {(art.id, agent, "version_mismatch"): 1}


def test_stale_read_generation_counted_distinctly(registry) -> None:
    art = _mk_artifact()
    zombie, peer = uuid4(), uuid4()
    _seed(registry, art, zombie, peer)
    # Zombie acquires EXCLUSIVE (captures read_generation), then the sweep
    # reclaims the grant (owner_generation bumps; the version does NOT move).
    registry.set_agent_state(art.id, zombie, MESIState.EXCLUSIVE, trigger="write", tick=1)
    registry.set_agent_state(
        art.id, zombie, MESIState.INVALID, trigger="reclaim_heartbeat", tick=2
    )
    result = registry.commit_cas(
        art.id, zombie, expected_version=art.version, content_hash="h"
    )
    assert isinstance(result, ConflictDetail) and result.reason == "stale_read_generation"
    assert registry.conflict_outcome_totals() == {
        (art.id, zombie, "stale_read_generation"): 1
    }


def test_other_holder_counted_distinctly(registry) -> None:
    art = _mk_artifact()
    holder, committer = uuid4(), uuid4()
    _seed(registry, art, holder, committer)
    registry.set_agent_state(art.id, holder, MESIState.EXCLUSIVE, trigger="write", tick=1)
    result = registry.commit_cas(art.id, committer, expected_version=art.version, content_hash="h")
    assert isinstance(result, ConflictDetail) and result.reason == "other_holder"
    assert registry.conflict_outcome_totals() == {(art.id, committer, "other_holder"): 1}


def test_win_counts_nothing_and_zero_is_the_reported_result(registry) -> None:
    art = _mk_artifact()
    agent = uuid4()
    _seed(registry, art, agent)
    result = registry.commit_cas(art.id, agent, expected_version=1, content_hash="h")
    assert not isinstance(result, ConflictDetail)
    assert registry.conflict_outcome_totals() == {}


def test_counting_leaves_the_deny_and_arbitration_state_byte_stable(registry) -> None:
    """The deny returned to the caller is unchanged, and the counter write
    bumps neither version nor owner_generation."""
    art = _mk_artifact()
    agent = uuid4()
    _seed(registry, art, agent)
    before_gen = registry.get_owner_generation(art.id)
    result = registry.commit_cas(art.id, agent, expected_version=0, content_hash="h")
    assert result == ConflictDetail("version_mismatch", 1)
    assert registry.get_artifact(art.id).version == 1
    assert registry.get_owner_generation(art.id) == before_gen


def test_callbacks_fire_with_typed_fields(registry) -> None:
    seen: list[tuple] = []
    registry.conflict_callbacks.append(lambda a, g, r: seen.append((a, g, r)))
    art = _mk_artifact()
    agent = uuid4()
    _seed(registry, art, agent)
    registry.commit_cas(art.id, agent, expected_version=0, content_hash="h")
    assert seen == [(art.id, agent, "version_mismatch")]


def test_raising_callback_does_not_alter_the_outcome(registry) -> None:
    def _boom(a: UUID, g: UUID, r: str) -> None:
        raise RuntimeError("observer crashed")

    registry.conflict_callbacks.append(_boom)
    art = _mk_artifact()
    agent = uuid4()
    _seed(registry, art, agent)
    result = registry.commit_cas(art.id, agent, expected_version=0, content_hash="h")
    assert isinstance(result, ConflictDetail) and result.reason == "version_mismatch"
    assert registry.conflict_outcome_totals() == {(art.id, agent, "version_mismatch"): 1}


def test_commit_all_conflicts_are_counted_per_failing_member(registry) -> None:
    a1, a2 = _mk_artifact(), _mk_artifact()
    agent = uuid4()
    _seed(registry, a1, agent)
    _seed(registry, a2, agent)
    result = registry.commit_all(
        agent,
        {
            a1.id: CommitAllEntry(expected_version=0, content_hash="h1"),
            a2.id: CommitAllEntry(expected_version=1, content_hash="h2"),
        },
    )
    assert isinstance(result, MultiCommitConflict)
    totals = registry.conflict_outcome_totals()
    assert totals.get((a1.id, agent, "version_mismatch")) == 1
    assert (a2.id, agent, "version_mismatch") not in totals


def test_repeat_conflicts_accumulate(registry) -> None:
    art = _mk_artifact()
    agent = uuid4()
    _seed(registry, art, agent)
    for _ in range(3):
        registry.commit_cas(art.id, agent, expected_version=0, content_hash="h")
    assert registry.conflict_outcome_totals() == {(art.id, agent, "version_mismatch"): 3}


# ---------------------------------------------------------------------------
# Persistence across respawn (the finding that forced KTD-9's amendment)
# ---------------------------------------------------------------------------


def test_sqlite_counts_survive_registry_reopen(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    art = _mk_artifact()
    agent = uuid4()
    _seed(reg, art, agent)
    reg.commit_cas(art.id, agent, expected_version=0, content_hash="h")
    reg.close()
    reopened = SqliteArtifactRegistry(db)
    try:
        assert reopened.conflict_outcome_totals() == {
            (art.id, agent, "version_mismatch"): 1
        }
    finally:
        reopened.close()


def test_offline_reader_reads_a_closed_db(tmp_path: Path) -> None:
    """KTD-6's report path: aggregate offline, across coordinator restarts,
    without importing the coordinator (interface-layer reader)."""
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    art = _mk_artifact()
    agent = uuid4()
    _seed(reg, art, agent)
    for _ in range(3):
        reg.commit_cas(art.id, agent, expected_version=0, content_hash="h")
    reg.close()
    assert read_conflict_totals(db) == {(art.id.hex, agent.hex, "version_mismatch"): 3}


def test_offline_reader_missing_file_raises_not_zero(tmp_path: Path) -> None:
    """A report against a nonexistent store is a caller error, never zero."""
    with pytest.raises(FileNotFoundError):
        read_conflict_totals(tmp_path / "nope.db")


def test_offline_reader_reports_zero_for_a_conflict_free_db(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    assert read_conflict_totals(db) == {}


def test_offline_reader_tolerates_a_pre_instrumentation_db(tmp_path: Path) -> None:
    """A db that predates the instrumentation has no conflict_counters table at
    all — created here with raw sqlite, bypassing the registry writer whose
    open would create it. That reads as zero recorded, not an error."""
    import sqlite3

    db = tmp_path / "old.db"
    sqlite3.connect(db).close()
    assert read_conflict_totals(db) == {}


# ---------------------------------------------------------------------------
# BaseException pass-through and read-only handles (regressions for the
# in_transaction-guarded handler and the tolerant read-only totals path)
# ---------------------------------------------------------------------------


def test_base_exception_in_callback_propagates_clean_after_commit(tmp_path: Path) -> None:
    """An observer letting a BaseException through must not be shadowed by a
    'cannot rollback - no transaction is active' error: the deny's COMMIT lands
    before callbacks fire, so the in_transaction guard skips the moot ROLLBACK
    and the original SystemExit propagates with the counter row durable."""
    reg = SqliteArtifactRegistry(tmp_path / "state.db")
    try:
        art = _mk_artifact()
        agent = uuid4()
        _seed(reg, art, agent)

        def _exit(a: UUID, g: UUID, r: str) -> None:
            raise SystemExit("observer")

        reg.conflict_callbacks.append(_exit)
        with pytest.raises(SystemExit):
            reg.commit_cas(art.id, agent, expected_version=0, content_hash="h")
        # Same handle stays usable: the COMMIT preceded the raise.
        reg.conflict_callbacks.clear()
        assert reg.conflict_outcome_totals() == {(art.id, agent, "version_mismatch"): 1}
    finally:
        reg.close()


def test_base_exception_in_callback_propagates_clean_after_commit_all(
    tmp_path: Path,
) -> None:
    reg = SqliteArtifactRegistry(tmp_path / "state.db")
    try:
        art = _mk_artifact()
        agent = uuid4()
        _seed(reg, art, agent)

        def _exit(a: UUID, g: UUID, r: str) -> None:
            raise SystemExit("observer")

        reg.conflict_callbacks.append(_exit)
        with pytest.raises(SystemExit):
            reg.commit_all(
                agent, {art.id: CommitAllEntry(expected_version=0, content_hash="h")}
            )
        reg.conflict_callbacks.clear()
        assert reg.conflict_outcome_totals() == {(art.id, agent, "version_mismatch"): 1}
    finally:
        reg.close()


def test_read_only_handle_tolerates_pre_instrumentation_db(tmp_path: Path) -> None:
    """A read-only open of a pre-U5 db (valid coordinator schema, no
    conflict_counters table — simulated by dropping the table the writer
    created, keeping the schema stamp valid) reports zero, not an error."""
    import sqlite3

    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE conflict_counters")
    conn.commit()
    conn.close()
    ro = SqliteArtifactRegistry(db, read_only=True)
    try:
        assert ro.conflict_outcome_totals() == {}
    finally:
        ro.close()


def test_read_only_handle_serves_persisted_counts(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    art = _mk_artifact()
    agent = uuid4()
    _seed(reg, art, agent)
    reg.commit_cas(art.id, agent, expected_version=0, content_hash="h")
    reg.close()
    ro = SqliteArtifactRegistry(db, read_only=True)
    try:
        assert ro.conflict_outcome_totals() == {(art.id, agent, "version_mismatch"): 1}
    finally:
        ro.close()
