# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``forget_agent``: a registry removes the records of an identity that has
ended (#525), on both registries.

Forgetting removes the identity's per-file states, its heartbeat and its
queued notices, so ``/status`` and a peer's writes stop paying for it. It
deletes nothing while the identity holds a write grant, and it keeps a row
that records a sweep reclaim, which ``/status`` reports for a while after.
Deleting a row is only safe for an identity that can never act again: the
read-generation fence admits a commit from an identity with no row (pinned
here), which is why only an explicit end may trigger it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.sqlite_registry import ReadOnlyMutationError, SqliteArtifactRegistry
from ccs.core.states import MESIState, TransientState
from ccs.core.types import Artifact

_RECLAIM_TRIGGER = "reclaim_heartbeat"

# What ``_records_of`` reads back on a file the identity has no row on.
_NO_RECORD = (None, None, None, None, None, None)


def _register(registry: object, name: str = "plan.md") -> UUID:
    artifact_id = uuid4()
    registry.register_artifact(  # type: ignore[attr-defined]
        Artifact(id=artifact_id, name=name, version=1, content_hash="h-init"), content=""
    )
    return artifact_id


def _rows_of(registry: object, agent: UUID, artifacts: list[UUID]) -> dict[UUID, MESIState | None]:
    return {artifact_id: registry.get_agent_state(artifact_id, agent) for artifact_id in artifacts}  # type: ignore[attr-defined]


def _records_of(registry: object, agent: UUID, artifacts: list[UUID]) -> dict[UUID, tuple[object, ...]]:
    """Everything a registry keeps for one identity on each file."""
    return {
        artifact_id: (
            registry.get_agent_state(artifact_id, agent),  # type: ignore[attr-defined]
            registry.get_agent_transient(artifact_id, agent),  # type: ignore[attr-defined]
            registry.get_transient_tick(artifact_id, agent),  # type: ignore[attr-defined]
            registry.get_read_generation(artifact_id, agent),  # type: ignore[attr-defined]
            registry.last_observed_version_for(artifact_id, agent),  # type: ignore[attr-defined]
            registry.get_last_reclamation(agent, artifact_id),  # type: ignore[attr-defined]
        )
        for artifact_id in artifacts
    }


def _queues_notices(registry: object) -> bool:
    return hasattr(registry, "record_preemption_notice")


def _queue_notice(registry: object, victim: UUID, file_and_preempter: tuple[UUID, UUID]) -> None:
    """Queue a notice to ``victim`` that the preempter took the file from it."""
    artifact_id, preempter = file_and_preempter
    registry.record_preemption_notice(  # type: ignore[attr-defined]
        victim_agent_id=victim, artifact_id=artifact_id, preempter_agent_id=preempter, preempted_at_unix_ts=1.0
    )


def test_forgetting_an_ended_identity_removes_its_rows_and_heartbeat(registry: object) -> None:
    """A reader, a copy made stale and a copy left mid-transition all go, with
    every value recorded on them and the heartbeat; another identity on the
    same files keeps its rows; a second call removes nothing.

    Prevents an ended identity's rows staying listed and costing peers' writes."""
    shared, stale, moving = _register(registry, "a.md"), _register(registry, "b.md"), _register(registry, "c.md")
    ended, peer = uuid4(), uuid4()
    registry.set_agent_state(shared, ended, MESIState.SHARED, trigger="fetch", tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(stale, ended, MESIState.INVALID, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(moving, ended, MESIState.INVALID, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_transient(moving, ended, TransientState.SIA, entered_tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(shared, peer, MESIState.SHARED, tick=1)  # type: ignore[attr-defined]
    registry.record_heartbeat(ended, 5)  # type: ignore[attr-defined]

    removed = registry.forget_agent(ended)  # type: ignore[attr-defined]

    assert removed == 3
    files = [shared, stale, moving]
    assert _records_of(registry, ended, files) == {f: _NO_RECORD for f in files}
    assert registry.get_transient_map(moving) == {}  # type: ignore[attr-defined]
    assert registry.get_state_map(shared) == {peer: MESIState.SHARED}  # type: ignore[attr-defined]
    assert registry.last_heartbeat_tick(ended) is None  # type: ignore[attr-defined]
    assert registry.forget_agent(ended) == 0  # type: ignore[attr-defined]


def test_forgetting_removes_the_identitys_queued_notices(registry: object) -> None:
    """Prevents a forgotten identity's notices waiting for a stop that never comes."""
    if not _queues_notices(registry):
        pytest.skip("only the durable registry queues notices")
    artifact_id, ended, preempter = _register(registry), uuid4(), uuid4()
    registry.set_agent_state(artifact_id, ended, MESIState.INVALID, tick=1)  # type: ignore[attr-defined]
    _queue_notice(registry, ended, (artifact_id, preempter))

    registry.forget_agent(ended)  # type: ignore[attr-defined]

    assert registry.pop_pending_notices(ended) == []  # type: ignore[attr-defined]


def test_forgetting_leaves_every_other_identitys_records(registry: object) -> None:
    """A peer's rows, the values recorded on them, its heartbeat and its
    queued notices stay as they were, a notice the ended identity caused
    included: that notice belongs to the peer it names as victim.

    Prevents a forget that reaches past the identity it was given."""
    read, taken = _register(registry, "read.md"), _register(registry, "taken.md")
    ended, peer = uuid4(), uuid4()
    registry.set_agent_state(read, ended, MESIState.SHARED, trigger="fetch", tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(read, peer, MESIState.SHARED, trigger="fetch", tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(taken, peer, MESIState.EXCLUSIVE, tick=2)  # type: ignore[attr-defined]
    registry.set_agent_state(taken, peer, MESIState.INVALID, trigger="write", tick=3)  # type: ignore[attr-defined]
    registry.set_agent_transient(taken, peer, TransientState.EIA, entered_tick=3)  # type: ignore[attr-defined]
    registry.record_heartbeat(peer, 7)  # type: ignore[attr-defined]
    if _queues_notices(registry):
        _queue_notice(registry, peer, (taken, ended))
    before = _records_of(registry, peer, [read, taken])

    registry.forget_agent(ended)  # type: ignore[attr-defined]

    assert _records_of(registry, peer, [read, taken]) == before
    assert registry.last_heartbeat_tick(peer) == 7  # type: ignore[attr-defined]
    if _queues_notices(registry):
        assert registry.peek_preemption_notice(peer, taken) == (ended, 1.0)  # type: ignore[attr-defined]


def test_a_transient_with_no_state_counts_as_one_row(registry: object) -> None:
    """A copy can enter a transition before any state is set on it. Sqlite
    stores that as an INVALID row carrying the transient, the in-memory
    registry as the transient alone; both forget it as one row.

    Prevents the two registries reporting different counts for one identity."""
    artifact_id, ended = _register(registry), uuid4()
    registry.set_agent_transient(artifact_id, ended, TransientState.SIA, entered_tick=1)  # type: ignore[attr-defined]

    assert registry.forget_agent(ended) == 1  # type: ignore[attr-defined]

    assert registry.get_transient_map(artifact_id) == {}  # type: ignore[attr-defined]
    assert registry.get_transient_tick(artifact_id, ended) is None  # type: ignore[attr-defined]


def test_a_peers_write_grants_do_not_stop_the_forget(registry: object) -> None:
    """Only the identity's own write grant refuses the forget. A peer holding
    EXCLUSIVE on another file and MODIFIED on a file both have rows on keeps
    both grants, and the ended identity's row goes.

    Prevents a forget refused, or a peer's grant moved, by a grant that is
    not the ended identity's."""
    own, both = _register(registry, "a.md"), _register(registry, "b.md")
    ended, peer = uuid4(), uuid4()
    registry.set_agent_state(own, peer, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(both, ended, MESIState.INVALID, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(both, peer, MESIState.MODIFIED, tick=2)  # type: ignore[attr-defined]
    peer_before = _records_of(registry, peer, [own, both])

    assert registry.forget_agent(ended) == 1  # type: ignore[attr-defined]

    assert _records_of(registry, ended, [own, both]) == {own: _NO_RECORD, both: _NO_RECORD}
    assert registry.get_state_map(own) == {peer: MESIState.EXCLUSIVE}  # type: ignore[attr-defined]
    assert registry.get_state_map(both) == {peer: MESIState.MODIFIED}  # type: ignore[attr-defined]
    assert _records_of(registry, peer, [own, both]) == peer_before


def test_a_row_that_records_a_reclaim_stays(registry: object) -> None:
    """The reclaim record lives on the row, and ``/status`` reports it for a
    while after the holder is gone, so forgetting the holder keeps that row.

    Prevents a sweep reclaim vanishing from ``/status`` as the holder ends."""
    reclaimed, plain = _register(registry, "a.md"), _register(registry, "b.md")
    ended = uuid4()
    registry.set_agent_state(reclaimed, ended, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(reclaimed, ended, MESIState.INVALID, trigger=_RECLAIM_TRIGGER, tick=2)  # type: ignore[attr-defined]
    registry.set_agent_state(plain, ended, MESIState.SHARED, tick=2)  # type: ignore[attr-defined]

    assert registry.forget_agent(ended) == 1  # type: ignore[attr-defined]

    assert _rows_of(registry, ended, [reclaimed, plain]) == {reclaimed: MESIState.INVALID, plain: None}
    assert registry.get_last_reclamation(ended, reclaimed) == (_RECLAIM_TRIGGER, 2)  # type: ignore[attr-defined]


def test_a_reclaim_row_the_identity_read_again_stays_as_invalid(registry: object) -> None:
    """A holder that lost its grant to the sweep and read the file again is
    SHARED on a row that still carries the reclaim record. Forgetting it keeps
    the row for the record and leaves it INVALID: an ended identity reads
    nothing, so it must stop counting as a reader of that file.

    Prevents an ended identity staying listed as a reader through its
    reclaim row."""
    reclaimed, peer_file = _register(registry, "a.md"), _register(registry, "b.md")
    ended, peer = uuid4(), uuid4()
    registry.set_agent_state(reclaimed, ended, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(reclaimed, ended, MESIState.INVALID, trigger=_RECLAIM_TRIGGER, tick=2)  # type: ignore[attr-defined]
    registry.set_agent_state(reclaimed, peer, MESIState.SHARED, trigger="fetch", tick=3)  # type: ignore[attr-defined]
    registry.set_agent_state(reclaimed, ended, MESIState.SHARED, trigger="fetch", tick=4)  # type: ignore[attr-defined]
    registry.set_agent_state(peer_file, peer, MESIState.SHARED, trigger="fetch", tick=4)  # type: ignore[attr-defined]
    assert registry.get_last_reclamation(ended, reclaimed) == (_RECLAIM_TRIGGER, 2), (  # type: ignore[attr-defined]
        "precondition: a read after a reclaim keeps the reclaim record"
    )

    assert registry.forget_agent(ended) == 0  # type: ignore[attr-defined]

    assert registry.get_agent_state(reclaimed, ended) == MESIState.INVALID  # type: ignore[attr-defined]
    assert registry.get_last_reclamation(ended, reclaimed) == (_RECLAIM_TRIGGER, 2)  # type: ignore[attr-defined]
    assert registry.valid_holders(reclaimed) == [peer]  # type: ignore[attr-defined]
    assert registry.get_agent_state(reclaimed, peer) == MESIState.SHARED  # type: ignore[attr-defined]


@pytest.mark.parametrize("held", [MESIState.EXCLUSIVE, MESIState.MODIFIED])
def test_an_identity_that_holds_a_write_grant_is_not_forgotten(registry: object, held: MESIState) -> None:
    """A write grant is what peers are refused on and what the sweep reclaims,
    so its holder's records all stay, the other files' included.

    Prevents a forget that hides a held grant from its peers."""
    writing, reading = _register(registry, "a.md"), _register(registry, "b.md")
    holder, preempter = uuid4(), uuid4()
    registry.set_agent_state(writing, holder, held, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(reading, holder, MESIState.SHARED, tick=1)  # type: ignore[attr-defined]
    registry.record_heartbeat(holder, 3)  # type: ignore[attr-defined]
    if _queues_notices(registry):
        _queue_notice(registry, holder, (reading, preempter))
    before = _records_of(registry, holder, [writing, reading])

    with pytest.raises(ValueError, match="write grant"):
        registry.forget_agent(holder)  # type: ignore[attr-defined]

    assert _records_of(registry, holder, [writing, reading]) == before
    assert _rows_of(registry, holder, [writing, reading]) == {writing: held, reading: MESIState.SHARED}
    assert registry.last_heartbeat_tick(holder) == 3  # type: ignore[attr-defined]
    if _queues_notices(registry):
        assert registry.peek_preemption_notice(holder, reading) == (preempter, 1.0)  # type: ignore[attr-defined]


def test_a_forgotten_identitys_commit_is_admitted_on_version_alone(registry: object) -> None:
    """Pins why only a provable end may forget: a holder that released its
    grant keeps the read generation it captured, which the release made
    stale, so its compare-and-swap is refused; once its row is gone the fence
    has nothing to refuse it on, and the version check alone admits it."""
    artifact_id, holder = _register(registry), uuid4()
    registry.set_agent_state(artifact_id, holder, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_state(artifact_id, holder, MESIState.SHARED, trigger="invalidate", tick=2)  # type: ignore[attr-defined]
    refused = registry.commit_cas(artifact_id, holder, expected_version=1, content_hash="h-2", tick=3)  # type: ignore[attr-defined]
    assert not isinstance(refused, tuple), f"control: the released holder's commit is refused, got {refused!r}"

    registry.forget_agent(holder)  # type: ignore[attr-defined]
    admitted = registry.commit_cas(artifact_id, holder, expected_version=1, content_hash="h-2", tick=4)  # type: ignore[attr-defined]

    assert isinstance(admitted, tuple), f"the forgotten identity's commit was refused: {admitted!r}"


# ---------------------------------------------------------------------------
# The durable registry only: its transaction and its read-only mode.
# ---------------------------------------------------------------------------


class _SimulatedCrash(Exception):
    """What :class:`_CrashAtStatement` raises in place of the statement it stops."""


class _CrashAtStatement:
    """Connection proxy that simulates a crash at one statement: every other
    statement runs on the real connection, so the ones before it have run
    inside the forget's open transaction."""

    def __init__(self, conn: sqlite3.Connection, statement: str) -> None:
        object.__setattr__(self, "_conn", conn)
        object.__setattr__(self, "_statement", statement)

    def execute(self, sql: str, *args: object) -> object:
        if sql.strip().startswith(self._statement):
            raise _SimulatedCrash(f"simulated crash at {self._statement!r}")
        return self._conn.execute(sql, *args)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


def test_a_forget_that_fails_partway_changes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The forget's statements share one transaction. A failure after its
    first delete, here at the heartbeat delete, rolls back that delete and the
    update to the reclaim row too: the SHARED row, the reclaim row the
    identity read again and the heartbeat all stay as they were.

    Prevents a forget that fails partway leaving an identity half forgotten."""
    with SqliteArtifactRegistry(tmp_path / "forget.db") as reg:
        shared, reclaimed, ended = _register(reg, "a.md"), _register(reg, "b.md"), uuid4()
        reg.set_agent_state(shared, ended, MESIState.SHARED, trigger="fetch", tick=1)
        reg.set_agent_state(reclaimed, ended, MESIState.EXCLUSIVE, tick=1)
        reg.set_agent_state(reclaimed, ended, MESIState.INVALID, trigger=_RECLAIM_TRIGGER, tick=2)
        reg.set_agent_state(reclaimed, ended, MESIState.SHARED, trigger="fetch", tick=3)
        reg.record_heartbeat(ended, 5)
        before = (_records_of(reg, ended, [shared, reclaimed]), reg.last_heartbeat_tick(ended))
        assert before[0][reclaimed][0] == MESIState.SHARED, "precondition: the reclaim row was read again"

        with monkeypatch.context() as patched:
            patched.setattr(reg, "_conn", _CrashAtStatement(reg._conn, "DELETE FROM heartbeats"))
            with pytest.raises(_SimulatedCrash):
                reg.forget_agent(ended)

        assert (_records_of(reg, ended, [shared, reclaimed]), reg.last_heartbeat_tick(ended)) == before


def test_a_read_only_registry_refuses_to_forget(tmp_path: Path) -> None:
    """Prevents a read-only view of the store deleting an identity's rows."""
    db_path = tmp_path / "forget.db"
    with SqliteArtifactRegistry(db_path) as writer:
        artifact_id, ended = _register(writer), uuid4()
        writer.set_agent_state(artifact_id, ended, MESIState.SHARED, tick=1)

    with SqliteArtifactRegistry(db_path, read_only=True) as reader:
        with pytest.raises(ReadOnlyMutationError):
            reader.forget_agent(ended)

        assert reader.get_agent_state(artifact_id, ended) == MESIState.SHARED
