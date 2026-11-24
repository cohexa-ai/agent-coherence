# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The registry's basis read and its commit under a basis, on both registries.

``basis_rows`` is how a caller records and later re-reads the versions and
registrations of the inputs it acted on; ``commit_cas_if_basis`` commits one
target only while every input is still the version it read. Most tests run on
the shared ``registry`` fixture, so both registries answer the same scenarios.
The registration tests are per registry because each mints it differently: the
in-memory registry a token per registration, SQLite the row id.

These are registry-level: what the service resolves, holds and answers around
them is tested with the service. The conformance kit certifies the lookup, the
moved-basis answer and the compare-inside-the-transaction window for an outside
backend; this file adds what a shipped registry also owes: purity, the
unchanged-basis commit matching ``commit_cas`` outcome for outcome, the
statement cap, and the in-memory lock held from the compare to the commit.
"""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import BASIS_MAX_BOUND_IDS, BasisMoved
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry, row_registration
from ccs.core.basis import canonical_registration
from ccs.core.exceptions import OTHER_HOLDER_REASON, STALE_READ_GENERATION_REASON
from ccs.core.states import MESIState, TransientState
from ccs.core.types import Artifact, ConflictDetail

# How long a peer writer is given to land in the window under test; the effect
# gate's window tests use the same figure.
_WINDOW_WAIT = 0.5
_JOIN_TIMEOUT = 10.0


def _register(reg, name: str, *, version: int = 1, artifact_id: UUID | None = None) -> UUID:
    artifact_id = artifact_id or uuid4()
    reg.register_artifact(
        Artifact(id=artifact_id, name=name, version=version, content_hash=f"h-{name}"), content=""
    )
    return artifact_id


def _registration(reg, name: str) -> str:
    (row,) = reg.basis_rows((), (name,))
    return row.registration


def _peer_commit(reg, artifact_id: UUID, *, expected: int) -> None:
    """A compare-and-swap by a fresh writer that holds no grant."""
    won = reg.commit_cas(
        artifact_id, uuid4(), expected_version=expected, content_hash=f"peer-{expected}", tick=2
    )
    assert isinstance(won, tuple), f"precondition: the peer's commit must win, got {won!r}"


def _recorder(reg) -> list[tuple[UUID, UUID, str]]:
    fired: list[tuple[UUID, UUID, str]] = []
    reg.conflict_callbacks.append(lambda *event: fired.append(event))
    return fired


def _everything_on(reg, artifact_id: UUID, agents: tuple[UUID, ...]) -> tuple[object, ...]:
    """Every value a basis read must leave alone, for one artifact."""
    artifact = reg.get_artifact(artifact_id)
    return (
        sorted(reg.artifact_ids()),
        artifact.version,
        artifact.content_hash,
        reg.get_owner_generation(artifact_id),
        reg.get_state_map(artifact_id),
        reg.get_transient_map(artifact_id),
        tuple(reg.get_read_generation(artifact_id, agent) for agent in agents),
        tuple(reg.last_observed_version_for(artifact_id, agent) for agent in agents),
        reg.conflict_outcome_totals(),
        reg.session_count(),
    )


# ---------------------------------------------------------------------------
# registrations
# ---------------------------------------------------------------------------


def test_in_memory_registering_the_same_id_again_mints_a_new_registration() -> None:
    """CCSStore re-registers a deleted key under the same deterministic id, so
    the id alone cannot tell the new record from the one a caller read. Every
    registration mints a new token, whether it replaces a live record or
    follows a removal, and an old token stops resolving."""
    reg = ArtifactRegistry()
    key = _register(reg, "notes:key")
    first = _registration(reg, "notes:key")
    _peer_commit(reg, key, expected=1)

    _register(reg, "notes:key", artifact_id=key)  # replaces the live record
    second = _registration(reg, "notes:key")
    reg.remove_artifact(key)
    _register(reg, "notes:key", artifact_id=key)
    third = _registration(reg, "notes:key")

    assert len({first, second, third}) == 3
    assert all(canonical_registration(token) == token for token in (first, second, third))
    assert reg.basis_rows((first, second), ()) == []
    (row,) = reg.basis_rows((third,), ())
    assert (row.artifact_id, row.version) == (key, 1)


def test_sqlite_registration_is_the_row_id_and_a_reregistered_path_gets_a_new_one(
    tmp_path: Path,
) -> None:
    """On SQLite the registration is the row id in canonical dashed form, and
    the store mints a new id for a path registered again after a delete."""
    with SqliteArtifactRegistry(tmp_path / "basis.db") as reg:
        first_id = reg.resolve_or_register("docs/brief.md", "h1")
        assert _registration(reg, "docs/brief.md") == str(first_id) == row_registration(first_id)

        reg.remove_artifact(first_id)
        second_id = reg.resolve_or_register("docs/brief.md", "h2")

        assert second_id != first_id
        assert _registration(reg, "docs/brief.md") == str(second_id)
        assert reg.basis_rows((str(first_id),), ()) == []


# ---------------------------------------------------------------------------
# basis_rows is read-only
# ---------------------------------------------------------------------------


def test_basis_rows_changes_nothing(registry) -> None:
    """A basis read neither grants, captures a read generation, counts, opens a
    session nor mints a row, whatever it is asked for."""
    target = _register(registry, "approvals.md")
    reader, reclaimed, pending = uuid4(), uuid4(), uuid4()
    registry.set_agent_state(target, reclaimed, MESIState.EXCLUSIVE, tick=1)
    registry.set_agent_state(target, reclaimed, MESIState.INVALID, trigger="reclaim_heartbeat", tick=2)
    registry.set_agent_state(target, reader, MESIState.SHARED, trigger="fetch", tick=3)
    registry.set_agent_transient(target, pending, TransientState.ISG, entered_tick=4)
    registry.capture_version_vector([target], "session-1", owner=reader, created_at_tick=4)
    agents = (reader, reclaimed, pending)
    before = _everything_on(registry, target, agents)

    registry.basis_rows((_registration(registry, "approvals.md"), str(uuid4())), ("approvals.md", "never.md"))

    assert _everything_on(registry, target, agents) == before


# ---------------------------------------------------------------------------
# a moved basis writes nothing
# ---------------------------------------------------------------------------


def test_a_moved_basis_writes_nothing_and_counts_no_conflict(registry) -> None:
    """A basis move is not a compare-and-swap refusal: no version bump, no
    state change, no conflict count and no conflict callback. The store is
    left as it was, so the same commit on the current basis then wins."""
    target = _register(registry, "approvals.md")
    member = _register(registry, "brief.md")
    agent, reader = uuid4(), uuid4()
    registry.set_agent_state(target, reader, MESIState.SHARED, trigger="fetch", tick=1)
    registry.set_agent_state(target, agent, MESIState.SHARED, tick=1)
    _peer_commit(registry, member, expected=1)
    fired = _recorder(registry)
    before = _everything_on(registry, target, (agent, reader))

    result = registry.commit_cas_if_basis(
        {member: 1, target: 1}, target, agent, expected_version=1, content_hash="h-mine", tick=5
    )

    assert isinstance(result, BasisMoved), result
    assert dict(result.moved) == {member: 2}
    assert _everything_on(registry, target, (agent, reader)) == before
    assert fired == []

    won = registry.commit_cas_if_basis(
        {member: 2, target: 1}, target, agent, expected_version=1, content_hash="h-mine", tick=6
    )
    assert isinstance(won, tuple) and won[0].version == 2, won


@pytest.mark.parametrize("target_listed", [True, False], ids=["target_in_basis", "target_outside_basis"])
@pytest.mark.parametrize("change", ["moved", "vanished"])
def test_a_changed_target_is_a_basis_move(registry, target_listed: bool, change: str) -> None:
    """The target's own expected version is compared with the basis, listed or
    not: a moved target is never a ``version_mismatch`` and a vanished one never
    a ``KeyError``, and neither is counted as a conflict."""
    target = _register(registry, "approvals.md")
    member = _register(registry, "brief.md")
    if change == "moved":
        _peer_commit(registry, target, expected=1)
        current = 2
    else:
        registry.remove_artifact(target)
        current = None
    basis = {member: 1, target: 1} if target_listed else {member: 1}

    result = registry.commit_cas_if_basis(
        basis, target, uuid4(), expected_version=1, content_hash="h-mine", tick=5
    )

    assert isinstance(result, BasisMoved), result
    assert dict(result.moved) == {target: current}
    assert registry.conflict_outcome_totals() == {}


# ---------------------------------------------------------------------------
# an unchanged basis commits exactly as commit_cas does
# ---------------------------------------------------------------------------


def test_an_unchanged_basis_wins_as_commit_cas_does(registry) -> None:
    """The win is commit_cas's win on the target: next version, the new hash,
    readers invalidated, the committer SHARED at the version it wrote. No other
    member is rewritten."""
    target = _register(registry, "approvals.md")
    member = _register(registry, "brief.md", version=4)
    agent, reader = uuid4(), uuid4()
    registry.set_agent_state(target, reader, MESIState.SHARED, tick=1)

    result = registry.commit_cas_if_basis(
        {member: 4}, target, agent, expected_version=1, content_hash="h-approved", tick=3
    )

    updated, invalidated = result
    assert (updated.id, updated.name, updated.version, updated.content_hash) == (
        target, "approvals.md", 2, "h-approved",
    )
    assert invalidated == [reader]
    assert registry.get_state_map(target) == {reader: MESIState.INVALID, agent: MESIState.SHARED}
    assert registry.last_observed_version_for(target, agent) == 2
    member_now = registry.get_artifact(member)
    assert (member_now.version, member_now.content_hash) == (4, "h-brief.md")


def _hold_by_a_peer(registry, target: UUID, agent: UUID) -> ConflictDetail:
    registry.set_agent_state(target, uuid4(), MESIState.MODIFIED, tick=1)
    return ConflictDetail(OTHER_HOLDER_REASON, 1)


def _reclaim_the_committer(registry, target: UUID, agent: UUID) -> ConflictDetail:
    registry.set_agent_state(target, agent, MESIState.EXCLUSIVE, tick=1)
    registry.set_agent_state(target, agent, MESIState.INVALID, trigger="reclaim_heartbeat", tick=2)
    registry.set_agent_state(target, agent, MESIState.SHARED, trigger="peer_regrant", tick=3)
    return ConflictDetail(STALE_READ_GENERATION_REASON, 1, captured_generation=0, owner_generation=1)


@pytest.mark.parametrize(
    "arrange", [_hold_by_a_peer, _reclaim_the_committer], ids=["other_holder", "stale_read_generation"]
)
def test_an_unchanged_basis_is_refused_as_commit_cas_refuses(registry, arrange) -> None:
    """On an unchanged basis the target's own refusal is commit_cas's: the same
    typed detail with its generation pair, counted once, its callback fired
    once, and nothing written."""
    target = _register(registry, "approvals.md")
    member = _register(registry, "brief.md")
    agent = uuid4()
    expected = arrange(registry, target, agent)
    fired = _recorder(registry)

    result = registry.commit_cas_if_basis(
        {member: 1}, target, agent, expected_version=1, content_hash="h-mine", tick=5
    )

    assert result == expected
    assert registry.conflict_outcome_totals() == {(target, agent, expected.reason): 1}
    assert fired == [(target, agent, expected.reason)]
    assert registry.get_artifact(target).version == 1


# ---------------------------------------------------------------------------
# members and arguments
# ---------------------------------------------------------------------------


def test_a_record_written_only_by_compare_and_swap_works_as_a_member(registry) -> None:
    """A member no agent was ever granted, moved only by compare-and-swap,
    reads, holds and admits like any other."""
    member = _register(registry, "brief.md")
    target = _register(registry, "approvals.md")
    _peer_commit(registry, member, expected=1)
    (row,) = registry.basis_rows((), ("brief.md",))
    assert (row.artifact_id, row.version) == (member, 2)

    won = registry.commit_cas_if_basis(
        {member: 2}, target, uuid4(), expected_version=1, content_hash="h-mine", tick=5
    )
    assert isinstance(won, tuple) and won[0].version == 2, won

    _peer_commit(registry, member, expected=2)
    moved = registry.commit_cas_if_basis(
        {member: 2}, target, uuid4(), expected_version=2, content_hash="h-mine", tick=6
    )
    assert isinstance(moved, BasisMoved) and dict(moved.moved) == {member: 3}, moved


def test_a_target_listed_at_another_version_is_refused(registry) -> None:
    """One row never carries two comparands: a basis that lists the target at
    a version other than its expected version is a malformed call."""
    target = _register(registry, "approvals.md")
    member = _register(registry, "brief.md")

    with pytest.raises(ValueError, match="target"):
        registry.commit_cas_if_basis(
            {member: 1, target: 2}, target, uuid4(), expected_version=1, content_hash="h", tick=5
        )
    assert registry.get_artifact(target).version == 1


def test_more_ids_than_one_statement_binds_are_refused(registry) -> None:
    """A basis is read in one statement, so a request that would not fit one
    is refused whole rather than split across reads; at the cap it is served."""
    target = _register(registry, "approvals.md")
    agent = uuid4()
    over = {uuid4(): 1 for _ in range(BASIS_MAX_BOUND_IDS)}  # with the target, one too many

    with pytest.raises(ValueError, match=str(BASIS_MAX_BOUND_IDS)):
        registry.commit_cas_if_basis(over, target, agent, expected_version=1, content_hash="h", tick=5)
    with pytest.raises(ValueError, match=str(BASIS_MAX_BOUND_IDS)):
        registry.basis_rows([str(uuid4()) for _ in range(BASIS_MAX_BOUND_IDS)], ["approvals.md"])

    at_cap = {uuid4(): 1 for _ in range(BASIS_MAX_BOUND_IDS - 1)}
    result = registry.commit_cas_if_basis(at_cap, target, agent, expected_version=1, content_hash="h", tick=5)
    assert isinstance(result, BasisMoved) and dict(result.moved) == dict.fromkeys(at_cap)
    rows = registry.basis_rows([str(uuid4()) for _ in range(BASIS_MAX_BOUND_IDS - 1)], ["approvals.md"])
    assert [row.artifact_id for row in rows] == [target]
    assert registry.get_artifact(target).version == 1


# ---------------------------------------------------------------------------
# the in-memory lock is held from the compare through the commit
# ---------------------------------------------------------------------------


def test_in_memory_peer_cannot_land_between_the_basis_compare_and_the_commit() -> None:
    """The in-memory commit compares, then hands the target to ``commit_cas``
    inside the same hold of the registry lock. A peer on the same registry,
    started the moment ``commit_cas`` is entered, must still be waiting a full
    window later; the target wins on the basis it was compared against and the
    peer lands afterwards. A lock released between the two lets it land first."""
    reg = ArtifactRegistry()
    member = _register(reg, "brief.md")
    target = _register(reg, "approvals.md")
    real_commit_cas = reg.commit_cas
    peer_outcome: list[object] = []
    peer = threading.Thread(
        target=lambda: peer_outcome.append(
            real_commit_cas(member, uuid4(), expected_version=1, content_hash="h-peer", tick=6)
        ),
        daemon=True,
    )
    landed_before_the_commit: list[bool] = []

    def commit_cas_after_starting_the_peer(*args, **kwargs):
        if not landed_before_the_commit:
            peer.start()
            peer.join(timeout=_WINDOW_WAIT)
            landed_before_the_commit.append(not peer.is_alive())
        return real_commit_cas(*args, **kwargs)

    reg.commit_cas = commit_cas_after_starting_the_peer
    result = reg.commit_cas_if_basis(
        {member: 1}, target, uuid4(), expected_version=1, content_hash="h-mine", tick=5
    )
    peer.join(timeout=_JOIN_TIMEOUT)

    assert landed_before_the_commit == [False]
    assert isinstance(result, tuple) and result[0].version == 2, result
    assert not peer.is_alive()
    assert isinstance(peer_outcome[0], tuple), peer_outcome
    assert reg.get_artifact(member).version == 2
