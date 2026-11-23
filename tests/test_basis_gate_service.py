# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The service's basis check, its commit under a basis, and the recording read.

A caller reads several inputs, records each one's path, version and
registration (``basis_member``), and acts later. ``check_basis`` answers, at one
instant, whether every input is still what it read, naming every one that moved
and why; ``commit_if_basis`` commits one target only if so, comparing the basis
and the target inside one hold of the service and registry locks. Both resolve
members by lookup only and change nothing they do not commit.

Every scenario runs on both registries unless it names one: the in-memory
registry mints a registration token per register, SQLite uses the row id, so a
delete followed by a register reads as replaced on each. Threads that race the
hold start at a chosen point and are joined with a bound; a test that could
deadlock builds its own registry and leaves it open when a thread is stuck, so
it fails instead of hanging.
"""

from __future__ import annotations

import statistics
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import (
    BASIS_MAX_MEMBERS,
    BasisCommitResult,
    CoordinatorService,
)
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.basis import (
    ANSWER_HOLD,
    ANSWER_PROCEED,
    RECOVERY_REDERIVE,
    BasisMember,
    CheckedMember,
    CommittedTarget,
    MovedMember,
    TargetRefusal,
)
from ccs.core.exceptions import (
    BASIS_MEMBER_REPLACED,
    BASIS_MEMBER_UNMANAGED,
    HOLD_INPUT_VANISHED,
    HOLD_VERSION_MOVED,
    OCC_IS_SHARED_OR_INVALID_ONLY_REASON,
    OTHER_HOLDER_REASON,
    CoherenceError,
    GiverFenced,
    OccCallerTransientError,
    RegistryLockTimeout,
    WatchdogAbandoned,
)
from ccs.core.hashing import compute_content_hash
from ccs.core.states import MESIState, TransientState
from ccs.core.types import Artifact, FetchRequest
from tests.test_service_handoff import _hand_off, _label

# How long a peer started inside a hold is given to land; the effect gate's
# window tests use the same figure.
_WINDOW_WAIT = 0.5
_JOIN_TIMEOUT = 10.0
# A thread that holds a lock for a test releases it on its own after this, so
# a mutant that waits without a deadline fails rather than hangs.
_SELF_RELEASE = 3.0


@pytest.fixture(params=["memory", "sqlite"])
def service(request: pytest.FixtureRequest, tmp_path: Path):
    """The service over each registry."""
    if request.param == "memory":
        yield CoordinatorService(ArtifactRegistry())
        return
    registry = SqliteArtifactRegistry(tmp_path / "basis-gate-service.db")
    yield CoordinatorService(registry)
    registry.close()


@pytest.fixture
def sqlite_service(tmp_path: Path):
    registry = SqliteArtifactRegistry(tmp_path / "basis-gate-hook.db")
    yield CoordinatorService(registry)
    registry.close()


def _new_registry(kind: str, tmp_path: Path):
    if kind == "memory":
        return ArtifactRegistry()
    return SqliteArtifactRegistry(tmp_path / f"basis-gate-{kind}.db")


def _close(registry) -> None:
    if isinstance(registry, SqliteArtifactRegistry):
        registry.close()


def _artifact(service: CoordinatorService, name: str, body: str = "v1") -> UUID:
    return service.register_artifact(
        name=name, content=body, content_hash=compute_content_hash(body)
    ).id


def _record(service: CoordinatorService, *artifact_ids: UUID) -> list[BasisMember]:
    return [service.basis_member(artifact_id).member for artifact_id in artifact_ids]


def _version(service: CoordinatorService, artifact_id: UUID) -> int | None:
    artifact = service.registry.get_artifact(artifact_id)
    return None if artifact is None else artifact.version


def _peer_commit(service: CoordinatorService, artifact_id: UUID, body: str = "peer") -> int:
    """A compare-and-swap by a fresh writer that holds no grant."""
    current = _version(service, artifact_id)
    won = service.commit_cas(
        agent_id=uuid4(),
        artifact_id=artifact_id,
        expected_version=current,
        content_hash=compute_content_hash(body),
        content=body,
    )
    assert isinstance(won, tuple), f"precondition: the peer's commit must win, got {won!r}"
    return won[0].version


def _registry_peer(registry, artifact_id: UUID) -> None:
    """A peer commit straight on the registry, as a writer on another thread."""
    current = registry.get_artifact(artifact_id).version
    registry.commit_cas(
        artifact_id, uuid4(), expected_version=current, content_hash="peer-thread", tick=3
    )


def _commit(
    service: CoordinatorService,
    basis: list[BasisMember],
    target: BasisMember,
    *,
    body: str = "result",
    agent_id: UUID | None = None,
    **kwargs,
) -> BasisCommitResult:
    return service.commit_if_basis(
        basis, target, target.version, body, agent_id=agent_id or uuid4(), **kwargs
    )


def _snapshot(service: CoordinatorService, ids: list[UUID], agents: list[UUID]) -> dict:
    """Everything a check must leave as it was, read without side effects."""
    registry = service.registry
    per_artifact = {}
    for artifact_id in ids:
        artifact = registry.get_artifact(artifact_id)
        per_artifact[artifact_id] = (
            artifact.version,
            artifact.content_hash,
            registry.get_owner_generation(artifact_id),
            registry.get_state_map(artifact_id),
            registry.get_transient_map(artifact_id),
            tuple(registry.get_read_generation(artifact_id, agent) for agent in agents),
        )
    return {
        "artifacts": per_artifact,
        "ids": sorted(registry.artifact_ids()),
        "sessions": registry.session_count(),
    }


# ---------------------------------------------------------------------------
# check_basis: answers
# ---------------------------------------------------------------------------


def test_two_moved_members_are_both_named_in_request_order(service) -> None:
    spec = _artifact(service, "spec.md")
    fixture = _artifact(service, "fixture.json")
    notes = _artifact(service, "notes.md")
    for _ in range(3):
        _peer_commit(service, spec)
    _peer_commit(service, fixture)
    basis = _record(service, fixture, notes, spec)
    assert [m.version for m in basis] == [2, 1, 4]

    _peer_commit(service, spec, "spec v5")
    _peer_commit(service, fixture, "fixture v3")
    answer = service.check_basis(basis)

    assert answer.answer == ANSWER_HOLD
    assert answer.moved == (
        MovedMember("fixture.json", 2, 3, HOLD_VERSION_MOVED),
        MovedMember("spec.md", 4, 5, HOLD_VERSION_MOVED),
    )
    assert answer.recovery == RECOVERY_REDERIVE


def test_an_unchanged_basis_proceeds_with_every_expected_and_current_version(service) -> None:
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    _peer_commit(service, brief)
    basis = _record(service, plan, brief)

    answer = service.check_basis(basis)

    assert answer.answer == ANSWER_PROCEED
    assert answer.checked == (CheckedMember("plan.md", 1, 1), CheckedMember("brief.md", 2, 2))
    assert answer.moved == ()


def test_a_byte_identical_resave_moves_the_version_and_holds(service) -> None:
    spec = _artifact(service, "spec.md", "same bytes")
    basis = _record(service, spec)

    _peer_commit(service, spec, "same bytes")

    assert service.check_basis(basis).moved == (
        MovedMember("spec.md", 1, 2, HOLD_VERSION_MOVED),
    )


def test_an_unknown_member_holds_input_vanished_and_mints_nothing(service) -> None:
    plan = _artifact(service, "plan.md")
    ghost = BasisMember("ghost.md", 1, registration=str(uuid4()))
    ids_before = sorted(service.registry.artifact_ids())

    answer = service.check_basis([*_record(service, plan), ghost])

    assert answer.moved == (MovedMember("ghost.md", 1, None, HOLD_INPUT_VANISHED),)
    assert sorted(service.registry.artifact_ids()) == ids_before


def test_a_deleted_and_reregistered_member_holds_replaced(service) -> None:
    plan = _artifact(service, "plan.md")
    basis = _record(service, plan)

    service.delete(agent_id=uuid4(), artifact_id=plan)
    again = _artifact(service, "plan.md")

    assert _version(service, again) == 1
    assert service.check_basis(basis).moved == (
        MovedMember("plan.md", 1, None, BASIS_MEMBER_REPLACED),
    )


def test_a_member_whose_version_dropped_holds_replaced(service) -> None:
    plan = _artifact(service, "plan.md")
    _peer_commit(service, plan)
    basis = _record(service, plan)

    service.registry.set_artifact_and_content(
        plan, Artifact(id=plan, name="plan.md", version=1, content_hash="h"), ""
    )

    assert service.check_basis(basis).moved == (
        MovedMember("plan.md", 2, 1, BASIS_MEMBER_REPLACED),
    )


def test_a_registration_whose_record_carries_another_name_holds_replaced(service) -> None:
    plan = _artifact(service, "plan.md")
    (recorded,) = _record(service, plan)
    renamed = BasisMember("other.md", recorded.version, recorded.registration)

    assert service.check_basis([renamed]).moved == (
        MovedMember("other.md", 1, None, BASIS_MEMBER_REPLACED),
    )


def test_unmanaged_cannot_occur_in_process(service) -> None:
    """In process nothing is out of policy: every state a member can be in
    answers one of the other reasons, never unmanaged."""
    moved = _artifact(service, "moved.md")
    replaced = _artifact(service, "replaced.md")
    unchanged = _artifact(service, "unchanged.md")
    basis = [
        *_record(service, moved, replaced, unchanged),
        BasisMember("vanished.md", 1, registration=str(uuid4())),
    ]
    _peer_commit(service, moved)
    service.delete(agent_id=uuid4(), artifact_id=replaced)
    _artifact(service, "replaced.md")

    reasons = [member.reason for member in service.check_basis(basis).moved]

    assert reasons == [HOLD_VERSION_MOVED, BASIS_MEMBER_REPLACED, HOLD_INPUT_VANISHED]
    assert BASIS_MEMBER_UNMANAGED not in reasons


def test_a_ccsstore_delete_and_reput_at_the_same_version_holds_replaced() -> None:
    """CCSStore re-registers a deleted key under the same id and reaches the
    same version again; only the registration tells the two records apart."""
    pytest.importorskip("langgraph.store.base")
    from ccs.adapters.ccsstore import CCSStore
    from ccs.core.identity import artifact_uuid

    store = CCSStore()
    namespace = ("planner", "shared")
    store.put(namespace, "notes", {"text": "draft"})
    key = artifact_uuid("shared", "notes")
    service = store.core.coordinator
    basis = _record(service, key)

    store.delete(namespace, "notes")
    store.put(namespace, "notes", {"text": "draft"})

    assert _version(service, key) == basis[0].version
    assert service.check_basis(basis).moved == (
        MovedMember("shared:notes", basis[0].version, None, BASIS_MEMBER_REPLACED),
    )


def test_a_record_written_only_by_compare_and_swap_works_as_a_member(service) -> None:
    plan = _artifact(service, "plan.md")
    _peer_commit(service, plan)
    assert service.registry.get_state_map(plan) != {}  # the peer ends SHARED, no grant
    basis = _record(service, plan)

    assert service.check_basis(basis).answer == ANSWER_PROCEED
    _peer_commit(service, plan, "again")
    assert service.check_basis(basis).moved == (
        MovedMember("plan.md", 2, 3, HOLD_VERSION_MOVED),
    )


def test_a_reread_and_another_agents_fetch_leave_the_basis_proceeding(service) -> None:
    plan = _artifact(service, "plan.md")
    caller, other = uuid4(), uuid4()
    basis = _record(service, plan)

    service.basis_member(plan)
    service.fetch(FetchRequest(artifact_id=plan, requesting_agent_id=caller, requested_at_tick=1))
    service.fetch(FetchRequest(artifact_id=plan, requesting_agent_id=other, requested_at_tick=2))

    assert service.check_basis(basis).answer == ANSWER_PROCEED


def test_a_check_changes_nothing(service) -> None:
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    reader, writer, sleeper = uuid4(), uuid4(), uuid4()
    service.fetch(FetchRequest(artifact_id=plan, requesting_agent_id=reader, requested_at_tick=1))
    service.write(agent_id=writer, artifact_id=brief, caller=writer)
    service.registry.set_agent_transient(plan, sleeper, TransientState.SIA, entered_tick=2)
    service.begin_session(read_set=[plan], owner=uuid4())
    proceeding = _record(service, plan, brief)
    holding = [*proceeding, BasisMember("ghost.md", 1, registration=str(uuid4()))]
    before = _snapshot(service, [plan, brief], [reader, writer, sleeper])

    assert service.check_basis(proceeding).answer == ANSWER_PROCEED
    assert service.check_basis(holding).answer == ANSWER_HOLD

    assert _snapshot(service, [plan, brief], [reader, writer, sleeper]) == before


def test_a_peer_commit_forced_into_the_check_lands_after_it(service) -> None:
    """The peer starts the moment the members' rows are read and cannot land
    until the check has answered from those rows."""
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    basis = _record(service, plan, brief)
    registry = service.registry
    peer = threading.Thread(target=_registry_peer, args=(registry, brief), daemon=True)
    landed_inside: list[bool] = []
    real_basis_rows = registry.basis_rows

    def rows_then_peer(*args, **kwargs):
        rows = real_basis_rows(*args, **kwargs)
        if not landed_inside:
            peer.start()
            peer.join(timeout=_WINDOW_WAIT)
            landed_inside.append(not peer.is_alive())
        return rows

    registry.basis_rows = rows_then_peer
    answer = service.check_basis(basis)
    peer.join(timeout=_JOIN_TIMEOUT)

    assert landed_inside == [False]
    assert answer.answer == ANSWER_PROCEED
    assert not peer.is_alive()
    assert _version(service, brief) == 2


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("empty", "at least one member"),
        ("no registration", "registration it was read under"),
        ("duplicate path", "each path and each registration once"),
        ("duplicate registration", "each path and each registration once"),
        ("too many", f"at most {BASIS_MAX_MEMBERS} members"),
    ],
)
def test_a_malformed_basis_raises(service, case: str, message: str) -> None:
    plan = _artifact(service, "plan.md")
    (member,) = _record(service, plan)
    basis = {
        "empty": [],
        "no registration": [BasisMember("plan.md", 1)],
        "duplicate path": [member, BasisMember("plan.md", 1, registration=str(uuid4()))],
        "duplicate registration": [member, BasisMember("copy.md", 1, member.registration)],
        "too many": [
            BasisMember(f"m{i}.md", 1, registration=str(uuid4()))
            for i in range(BASIS_MAX_MEMBERS + 1)
        ],
    }[case]

    with pytest.raises(ValueError, match=message):
        service.check_basis(basis)


def test_the_member_cap_admits_exactly_its_limit(service) -> None:
    basis = [
        BasisMember(f"m{i}.md", 1, registration=str(uuid4())) for i in range(BASIS_MAX_MEMBERS)
    ]

    assert len(service.check_basis(basis).moved) == BASIS_MAX_MEMBERS


# ---------------------------------------------------------------------------
# basis_member: the recording read
# ---------------------------------------------------------------------------


def test_basis_member_grants_nothing_and_describes_its_bytes(service) -> None:
    plan = _artifact(service, "plan.md", "plan body")
    (row,) = [r for r in service.registry.basis_rows((), ("plan.md",)) if r.artifact_id == plan]
    states_before = service.registry.get_state_map(plan)

    read = service.basis_member(plan)

    assert read.member == BasisMember("plan.md", 1, row.registration)
    assert read.content_hash == compute_content_hash("plan body")
    in_memory = isinstance(service.registry, ArtifactRegistry)
    assert read.content == ("plan body" if in_memory else None)
    assert service.registry.get_state_map(plan) == states_before


def test_a_peer_commit_forced_into_basis_member_cannot_split_it(service) -> None:
    plan = _artifact(service, "plan.md", "first body")
    registry = service.registry
    peer = threading.Thread(target=_peer_commit, args=(service, plan, "second body"), daemon=True)
    landed_inside: list[bool] = []
    real_get_artifact = registry.get_artifact

    def get_then_peer(artifact_id):
        artifact = real_get_artifact(artifact_id)
        if not landed_inside:
            peer.start()
            peer.join(timeout=_WINDOW_WAIT)
            landed_inside.append(not peer.is_alive())
        return artifact

    registry.get_artifact = get_then_peer
    read = service.basis_member(plan)
    registry.get_artifact = real_get_artifact
    peer.join(timeout=_JOIN_TIMEOUT)

    assert landed_inside == [False]
    assert (read.member.version, read.content_hash) == (1, compute_content_hash("first body"))
    in_memory = isinstance(registry, ArtifactRegistry)
    assert read.content == ("first body" if in_memory else None)
    assert _version(service, plan) == 2


def test_after_a_hash_only_commit_basis_member_returns_the_hash_without_bytes(service) -> None:
    plan = _artifact(service, "plan.md", "old body")
    basis = _record(service, plan)
    new_hash = compute_content_hash("new body")

    won = service.commit_if_basis(
        basis, basis[0], 1, agent_id=uuid4(), content_hash=new_hash
    )
    read = service.basis_member(plan)

    assert won.answer.answer == ANSWER_PROCEED
    assert (read.member.version, read.content_hash, read.content) == (2, new_hash, None)


# ---------------------------------------------------------------------------
# commit_if_basis: answers
# ---------------------------------------------------------------------------


def test_a_win_moves_only_the_target_and_invalidates_peers_only_on_it(service) -> None:
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    reader = uuid4()
    for artifact_id in (plan, brief):
        service.registry.set_agent_state(artifact_id, reader, MESIState.SHARED, tick=1)
    basis = _record(service, plan, brief)
    committer = uuid4()

    result = _commit(service, basis, basis[0], body="plan v2", agent_id=committer)

    assert result.answer.answer == ANSWER_PROCEED
    assert result.answer.committed == CommittedTarget("plan.md", 2)
    assert (result.artifact.id, result.artifact.version) == (plan, 2)
    assert result.artifact.content_hash == compute_content_hash("plan v2")
    assert [(s.artifact_id, s.victim_agent_id, s.issuer_agent_id) for s in result.signals] == [
        (plan, reader, committer)
    ]
    assert _version(service, brief) == 1
    assert service.registry.get_agent_state(brief, reader) is MESIState.SHARED
    assert service.registry.get_agent_state(plan, reader) is MESIState.INVALID


def test_a_moved_member_holds_the_commit_and_nothing_is_written(service) -> None:
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    basis = _record(service, plan, brief)
    _peer_commit(service, brief)
    before = _snapshot(service, [plan, brief], [])

    result = _commit(service, basis, basis[0])

    assert result.answer.answer == ANSWER_HOLD
    assert result.answer.moved == (MovedMember("brief.md", 1, 2, HOLD_VERSION_MOVED),)
    assert result.answer.recovery == RECOVERY_REDERIVE
    assert result.artifact is None and result.signals == ()
    assert _snapshot(service, [plan, brief], []) == before


def test_a_target_outside_the_basis_is_committed_when_everything_held(service) -> None:
    brief = _artifact(service, "brief.md")
    approvals = _artifact(service, "approvals.json")
    basis = _record(service, brief)
    (target,) = _record(service, approvals)

    result = _commit(service, basis, target)

    assert result.answer.committed == CommittedTarget("approvals.json", 2)
    assert (_version(service, approvals), _version(service, brief)) == (2, 1)


def test_a_target_outside_the_basis_that_moved_holds_with_a_member_reason(service) -> None:
    brief = _artifact(service, "brief.md")
    approvals = _artifact(service, "approvals.json")
    basis = _record(service, brief)
    (target,) = _record(service, approvals)
    _peer_commit(service, approvals)

    result = _commit(service, basis, target)

    assert result.answer.moved == (MovedMember("approvals.json", 1, 2, HOLD_VERSION_MOVED),)
    assert result.answer.recovery == RECOVERY_REDERIVE
    assert _version(service, approvals) == 2


def test_a_peer_write_grant_on_the_target_holds_with_other_holder(service) -> None:
    plan = _artifact(service, "plan.md")
    basis = _record(service, plan)
    holder = uuid4()
    service.write(agent_id=holder, artifact_id=plan, caller=holder)

    result = _commit(service, basis, basis[0])

    assert result.answer.answer == ANSWER_HOLD
    assert result.answer.moved == ()
    assert result.answer.recovery is None
    assert result.answer.target == TargetRefusal("plan.md", OTHER_HOLDER_REASON, 1)
    assert _version(service, plan) == 1


def test_a_caller_that_recorded_through_fetch_gets_the_m_or_e_refusal(service) -> None:
    plan = _artifact(service, "plan.md")
    caller = uuid4()
    service.fetch(FetchRequest(artifact_id=plan, requesting_agent_id=caller, requested_at_tick=1))
    assert service.registry.get_agent_state(plan, caller) is MESIState.EXCLUSIVE
    basis = _record(service, plan)

    with pytest.raises(CoherenceError, match=OCC_IS_SHARED_OR_INVALID_ONLY_REASON):
        _commit(service, basis, basis[0], agent_id=caller)
    assert _version(service, plan) == 1


def test_a_classification_hold_wins_over_a_transient_caller(service) -> None:
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    caller = uuid4()
    service.registry.set_agent_transient(plan, caller, TransientState.SIA, entered_tick=1)
    basis = _record(service, plan, brief)

    with pytest.raises(OccCallerTransientError):
        _commit(service, basis, basis[0], agent_id=caller)
    _peer_commit(service, brief)
    held = _commit(service, basis, basis[0], agent_id=caller)

    assert held.answer.moved == (MovedMember("brief.md", 1, 2, HOLD_VERSION_MOVED),)
    assert _version(service, plan) == 1


def test_a_handed_off_giver_is_fenced_and_the_successors_win_labels_the_handoff(
    service,
) -> None:
    handoff = _hand_off(service)
    basis = _record(service, handoff.path)

    with pytest.raises(GiverFenced):
        _commit(service, basis, basis[0], agent_id=handoff.giver_read, caller=handoff.giver)
    won = _commit(service, basis, basis[0], agent_id=uuid4(), caller=handoff.successor)

    assert won.answer.answer == ANSWER_PROCEED
    assert _label(service, handoff.path)[0] == "completed"


_TARGET_DISAGREES = "the target names a member but disagrees"


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("target disagrees with its member's version", _TARGET_DISAGREES),
        ("target disagrees with its member's registration", _TARGET_DISAGREES),
        ("target names a member's registration under another path", _TARGET_DISAGREES),
        ("target without registration", "the target needs the registration"),
        ("expected version is not the target's", "but expected at version 2"),
        ("neither content nor hash", "needs content or content_hash"),
        ("content and hash disagree", "content_hash is not the hash of content"),
    ],
)
def test_a_malformed_commit_raises(service, case: str, message: str) -> None:
    plan = _artifact(service, "plan.md")
    approvals = _artifact(service, "approvals.json")
    (member,) = _record(service, plan)
    (outside,) = _record(service, approvals)
    target, expected = member, member.version
    content, content_hash = "body", None
    if case == "target disagrees with its member's version":
        target = BasisMember("plan.md", 2, member.registration)
        expected = 2
    elif case == "target disagrees with its member's registration":
        target = BasisMember("plan.md", 1, str(uuid4()))
    elif case == "target names a member's registration under another path":
        target = BasisMember("approvals.json", 1, member.registration)
    elif case == "target without registration":
        target = BasisMember("approvals.json", 1)
    elif case == "expected version is not the target's":
        target, expected = outside, 2
    elif case == "neither content nor hash":
        content = None
    elif case == "content and hash disagree":
        content_hash = compute_content_hash("another body")

    with pytest.raises(ValueError, match=message):
        service.commit_if_basis(
            [member], target, expected, content, agent_id=uuid4(), content_hash=content_hash
        )
    assert (_version(service, plan), _version(service, approvals)) == (1, 1)


def test_a_matching_content_and_hash_are_accepted(service) -> None:
    plan = _artifact(service, "plan.md")
    basis = _record(service, plan)

    result = service.commit_if_basis(
        basis, basis[0], 1, "body", agent_id=uuid4(), content_hash=compute_content_hash("body")
    )

    assert result.answer.committed == CommittedTarget("plan.md", 2)


# ---------------------------------------------------------------------------
# commit_if_basis: one hold
# ---------------------------------------------------------------------------


def test_the_basis_compare_and_the_commit_share_one_outermost_hold(service) -> None:
    """The rows the classification reads and the registry's commit under the
    basis run inside one hold of the registry guard, with no outermost hold
    ending between them."""
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    basis = _record(service, plan, brief)
    registry = service.registry
    depth, holds_ended = [0], [0]
    seen: dict[str, tuple[int, int]] = {}
    real_guard = registry.abort_guard
    real_rows = registry.basis_rows
    real_commit = registry.commit_cas_if_basis

    @contextmanager
    def counting_guard(*args, **kwargs):
        with real_guard(*args, **kwargs):
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1
                if depth[0] == 0:
                    holds_ended[0] += 1

    def recording_rows(*args, **kwargs):
        seen.setdefault("classify", (depth[0], holds_ended[0]))
        return real_rows(*args, **kwargs)

    def recording_commit(*args, **kwargs):
        seen.setdefault("commit", (depth[0], holds_ended[0]))
        return real_commit(*args, **kwargs)

    registry.abort_guard = counting_guard
    registry.basis_rows = recording_rows
    registry.commit_cas_if_basis = recording_commit

    result = _commit(service, basis, basis[0])

    assert result.answer.answer == ANSWER_PROCEED
    assert seen["classify"][0] > 0 and seen["commit"][0] > 0
    assert seen["commit"][1] == seen["classify"][1]


def test_a_peer_forced_between_classification_and_commit_lands_after_it(service) -> None:
    """A peer writing a non-target member the moment the classification has
    read the rows waits until the commit has landed: the target was written
    once, from a basis that was still current."""
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    basis = _record(service, plan, brief)
    registry = service.registry
    peer = threading.Thread(target=_registry_peer, args=(registry, brief), daemon=True)
    landed_inside: list[bool] = []
    real_rows = registry.basis_rows

    def rows_then_peer(*args, **kwargs):
        rows = real_rows(*args, **kwargs)
        if not landed_inside:
            peer.start()
            peer.join(timeout=_WINDOW_WAIT)
            landed_inside.append(not peer.is_alive())
        return rows

    registry.basis_rows = rows_then_peer
    result = _commit(service, basis, basis[0])
    peer.join(timeout=_JOIN_TIMEOUT)

    assert landed_inside == [False]
    assert result.answer.committed == CommittedTarget("plan.md", 2)
    assert not peer.is_alive()
    assert (_version(service, plan), _version(service, brief)) == (2, 2)


def test_an_unguarded_split_commits_on_a_moved_basis(service) -> None:
    """The loss the commit form prevents: a check, then a separate
    compare-and-swap, with a peer between them, writes the target although an
    input it was derived from moved."""
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    basis = _record(service, plan, brief)

    assert service.check_basis(basis).answer == ANSWER_PROCEED
    _peer_commit(service, brief)
    won = service.commit_cas(
        agent_id=uuid4(),
        artifact_id=plan,
        expected_version=basis[0].version,
        content_hash=compute_content_hash("from brief v1"),
    )

    assert isinstance(won, tuple)
    assert (_version(service, plan), _version(service, brief)) == (2, 2)


def test_commit_if_basis_holds_an_input_moved_before_it(service) -> None:
    plan = _artifact(service, "plan.md")
    brief = _artifact(service, "brief.md")
    basis = _record(service, plan, brief)
    _peer_commit(service, brief)

    result = _commit(service, basis, basis[0], body="from brief v1")

    assert result.answer.moved == (MovedMember("brief.md", 1, 2, HOLD_VERSION_MOVED),)
    assert _version(service, plan) == 1


@pytest.mark.usefixtures("two_writer_store")
def test_a_move_inside_the_commit_transaction_is_labelled_by_the_classifier(
    tmp_path: Path,
) -> None:
    """A writer on a second handle to the same store is outside the service's
    hold. It moves a member after the classification read the rows; the
    registry's compare inside the commit transaction catches it, and the hold
    names the member with its reason."""
    registry = SqliteArtifactRegistry(tmp_path / "two-handles.db")
    other = SqliteArtifactRegistry(tmp_path / "two-handles.db")
    try:
        service = CoordinatorService(registry)
        plan = _artifact(service, "plan.md")
        brief = _artifact(service, "brief.md")
        basis = _record(service, plan, brief)
        real_commit = registry.commit_cas_if_basis

        def other_handle_then_commit(*args, **kwargs):
            _registry_peer(other, brief)
            return real_commit(*args, **kwargs)

        registry.commit_cas_if_basis = other_handle_then_commit
        result = _commit(service, basis, basis[0])

        assert result.answer.answer == ANSWER_HOLD
        assert result.answer.moved == (MovedMember("brief.md", 1, 2, HOLD_VERSION_MOVED),)
        assert result.answer.recovery == RECOVERY_REDERIVE
        assert _version(service, plan) == 1
    finally:
        other.close()
        registry.close()


# ---------------------------------------------------------------------------
# locks: deadline, abort, order
# ---------------------------------------------------------------------------


def _hold_in_thread(enter) -> tuple[threading.Event, threading.Thread]:
    """Run ``enter`` (a context manager factory) on a daemon thread until the
    returned event is set, or ``_SELF_RELEASE`` passes."""
    held, release = threading.Event(), threading.Event()

    def hold() -> None:
        with enter():
            held.set()
            release.wait(_SELF_RELEASE)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(_JOIN_TIMEOUT)
    return release, thread


def _session_lock_is_free(service: CoordinatorService) -> bool:
    """Whether another thread can take the service's session lock now."""
    taken: list[bool] = []

    def probe() -> None:
        acquired = service._session_lock.acquire(timeout=1.0)
        taken.append(acquired)
        if acquired:
            service._session_lock.release()

    thread = threading.Thread(target=probe, daemon=True)
    thread.start()
    thread.join(_JOIN_TIMEOUT)
    return taken == [True]


def test_a_session_lock_held_past_the_deadline_raises_naming_the_session_lock(service) -> None:
    plan = _artifact(service, "plan.md")
    basis = _record(service, plan)
    release, thread = _hold_in_thread(lambda: service._session_lock)
    try:
        with pytest.raises(RegistryLockTimeout, match="session lock"):
            service.check_basis(basis, deadline=time.monotonic() + 0.2)
        with pytest.raises(RegistryLockTimeout, match="session lock"):
            _commit(service, basis, basis[0], deadline=time.monotonic() + 0.2)
    finally:
        release.set()
        thread.join(_JOIN_TIMEOUT)
    assert _version(service, plan) == 1


def test_a_stalled_registry_raises_its_timeout_and_frees_the_session_lock(service) -> None:
    plan = _artifact(service, "plan.md")
    basis = _record(service, plan)
    release, thread = _hold_in_thread(service.registry.abort_guard)
    try:
        with pytest.raises(RegistryLockTimeout, match="registry lock"):
            service.check_basis(basis, deadline=time.monotonic() + 0.2)
        assert _session_lock_is_free(service)
    finally:
        release.set()
        thread.join(_JOIN_TIMEOUT)


def test_an_abort_set_before_the_guard_is_won_leaves_nothing_done(service) -> None:
    plan = _artifact(service, "plan.md")
    basis = _record(service, plan)
    before = _snapshot(service, [plan], [])
    abort = threading.Event()
    abort.set()

    with pytest.raises(WatchdogAbandoned):
        _commit(service, basis, basis[0], abort=abort)

    assert _snapshot(service, [plan], []) == before
    assert _session_lock_is_free(service)


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_a_check_racing_begin_session_never_deadlocks(tmp_path: Path, kind: str) -> None:
    """The check takes the service lock before the registry lock, as
    ``begin_session`` does. A begin started once the check holds the registry
    lock waits for the service lock and finishes after the check; taken the
    other way round, each would wait on the other."""
    registry = _new_registry(kind, tmp_path)
    deadlocked = False
    try:
        service = CoordinatorService(registry)
        plan = _artifact(service, "plan.md")
        basis = _record(service, plan)
        begun: list[object] = []
        begin = threading.Thread(
            target=lambda: begun.append(service.begin_session(read_set=[plan], owner=uuid4())),
            daemon=True,
        )
        real_guard = registry.abort_guard

        @contextmanager
        def guard_then_begin(*args, **kwargs):
            with real_guard(*args, **kwargs):
                if begin.ident is None:
                    begin.start()
                    begin.join(timeout=_WINDOW_WAIT)
                yield

        registry.abort_guard = guard_then_begin
        answers: list[object] = []
        check = threading.Thread(
            target=lambda: answers.append(service.check_basis(basis)), daemon=True
        )
        check.start()
        check.join(_JOIN_TIMEOUT)
        begin.join(_JOIN_TIMEOUT)
        deadlocked = check.is_alive() or begin.is_alive()

        assert not deadlocked, "the basis check deadlocked against begin_session"
        assert answers[0].answer == ANSWER_PROCEED
        assert len(begun) == 1
    finally:
        # A stuck thread still holds the registry lock, which close() takes.
        if not deadlocked:
            _close(registry)


# ---------------------------------------------------------------------------
# the resolve hook (the coordinator's request door builds one)
# ---------------------------------------------------------------------------


def _hook(registry, untracked: frozenset[str] = frozenset()):
    """A lookup-only resolver like the request door's: the row a path reaches,
    and whether the path is tracked."""

    def resolve(member: BasisMember) -> tuple[UUID | None, bool]:
        return registry.lookup_artifact_id_by_name(member.path), member.path not in untracked

    return resolve


def test_a_resolve_hook_answers_unmanaged_vanished_and_replaced(sqlite_service) -> None:
    service = sqlite_service
    registry = service.registry
    plan = registry.resolve_or_register("plan.md", "h1")
    registry.resolve_or_register("build/out.o", "h2")
    old_brief = registry.resolve_or_register("brief.md", "h3")
    stale_registration = str(old_brief)
    registry.remove_artifact(old_brief)
    registry.resolve_or_register("brief.md", "h3")
    ids_before = sorted(registry.artifact_ids())
    basis = [
        BasisMember("plan.md", 1, str(plan)),
        BasisMember("build/out.o", 1),
        BasisMember("ghost.md", 1),
        BasisMember("brief.md", 1, stale_registration),
    ]

    answer = service.check_basis(basis, resolve=_hook(registry, frozenset({"build/out.o"})))

    assert answer.moved == (
        MovedMember("build/out.o", 1, 1, BASIS_MEMBER_UNMANAGED),
        MovedMember("ghost.md", 1, None, HOLD_INPUT_VANISHED),
        MovedMember("brief.md", 1, 1, BASIS_MEMBER_REPLACED),
    )
    assert sorted(registry.artifact_ids()) == ids_before


def test_a_resolve_hook_target_without_registration_is_compared_by_version(
    sqlite_service,
) -> None:
    service = sqlite_service
    registry = service.registry
    registry.resolve_or_register("brief.md", "h1")
    approvals = registry.resolve_or_register("approvals.json", "h2")
    basis = [BasisMember("brief.md", 1)]
    target = BasisMember("approvals.json", 1)

    won = service.commit_if_basis(
        basis, target, 1, agent_id=uuid4(), content_hash="h3", resolve=_hook(registry)
    )
    again = service.commit_if_basis(
        basis, target, 1, agent_id=uuid4(), content_hash="h4", resolve=_hook(registry)
    )

    assert won.answer.committed == CommittedTarget("approvals.json", 2)
    assert again.answer.moved == (MovedMember("approvals.json", 1, 2, HOLD_VERSION_MOVED),)
    assert _version(service, approvals) == 2


# ---------------------------------------------------------------------------
# lock-hold time
# ---------------------------------------------------------------------------

_TIMING_RUNS = 5
_HOLD_CEILING_SEC = 0.5


@contextmanager
def _timed_outermost_holds(registry, holds: list[float]):
    real_guard = registry.abort_guard
    depth, started = [0], [0.0]

    @contextmanager
    def timed_guard(*args, **kwargs):
        with real_guard(*args, **kwargs):
            if depth[0] == 0:
                started[0] = time.perf_counter()
            depth[0] += 1
            try:
                yield
            finally:
                depth[0] -= 1
                if depth[0] == 0:
                    holds.append(time.perf_counter() - started[0])

    registry.abort_guard = timed_guard
    try:
        yield
    finally:
        registry.abort_guard = real_guard


def _one_hold(registry, call) -> float:
    holds: list[float] = []
    with _timed_outermost_holds(registry, holds):
        call()
    assert len(holds) == 1, f"expected one outermost hold, saw {len(holds)}"
    return holds[0]


@pytest.mark.parametrize("kind", ["memory", "sqlite"])
def test_lock_hold_time_at_1_8_and_64_members(tmp_path: Path, kind: str) -> None:
    """The outermost guard hold of a check and of a commit, measured and
    reported, never gated beyond a ceiling well under the handler deadline."""
    registry = _new_registry(kind, tmp_path)
    try:
        service = CoordinatorService(registry)
        report: list[str] = []
        for size in (1, 8, 64):
            ids = [_artifact(service, f"{size}/m{i}.md") for i in range(size)]
            checks, commits = [], []
            for _ in range(_TIMING_RUNS):
                basis = _record(service, *ids)
                checks.append(_one_hold(registry, lambda b=basis: service.check_basis(b)))
                commits.append(_one_hold(registry, lambda b=basis: _commit(service, b, b[0])))
            assert _version(service, ids[0]) == 1 + _TIMING_RUNS
            for name, figures in (("check", checks), ("commit", commits)):
                assert min(figures) < _HOLD_CEILING_SEC, (name, size, figures)
                report.append(
                    f"{name} {size}: min {min(figures) * 1000:.2f} ms, "
                    f"median {statistics.median(figures) * 1000:.2f} ms"
                )
        warnings.warn(f"basis lock hold on {kind}: " + "; ".join(report), RuntimeWarning)
    finally:
        _close(registry)
