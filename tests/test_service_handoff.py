# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The targeted grant handoff, driven through the service (#185).

A session done with a path hands it to a named successor. The registry keeps
the transfer record and decides each grant; this module drives the
service on top of it: the transfer, accept, decline and withdraw operations,
the giver fence every write path consults, and the completion or overtake a
successor's or a bystander's admitted write records.

Every scenario runs against BOTH registries through this module's own
``service`` fixture. Identities follow the HTTP shape: a session-level identity
(what the fence and the record key on) and the composite agent ids its
incarnations and subagents write under. The service never derives one from the
other -- the route passes the session-level identity as ``caller``.

Scenarios:

- the measured lost update (read, transfer, re-mint, compare-and-swap at the
  transfer version) is refused, while the plain optimistic writer with no
  record still wins -- and would not under an absent-record predicate;
- the fence on the pessimistic acquire, the pessimistic commit, the batch
  commit and both snapshot-session wrappers;
- what ends the fence (a decline, a withdraw, a version move; never an
  accept);
- the transfer through the service, re-sends, supersession and re-handing a
  path after a handoff ended;
- a watchdog abort at hold entry lands nothing, one set after the hold is
  taken lands every path;
- completion and overtake on an admitted acquire or win, never on an ended
  record, and a failed label write never turns a win into a failure;
- a session stop leaves the record as it was.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import TransferRecord
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import CoherenceError, GiverFenced, WatchdogAbandoned
from ccs.core.states import MESIState
from ccs.core.types import (
    Artifact,
    CommitAllEntry,
    ConflictDetail,
    FetchRequest,
    MultiCommitResult,
    TransferGrantOutcome,
    TransferVerbOutcome,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


class _Log:
    """A state log that records entries and can run a hook on each one."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.hook: Any = None

    def __call__(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        if self.hook is not None:
            self.hook()


@pytest.fixture(params=["memory", "sqlite"])
def service(request: pytest.FixtureRequest, tmp_path: Path):
    """The service over each registry. Named ``service``, not ``registry``, so it
    never shadows the tree's shared registry fixture."""
    if request.param == "memory":
        yield CoordinatorService(ArtifactRegistry())
        return
    registry = SqliteArtifactRegistry(tmp_path / "service-handoff.db")
    yield CoordinatorService(registry)
    registry.close()


@pytest.fixture(params=["memory", "sqlite"])
def logged_service(request: pytest.FixtureRequest, tmp_path: Path):
    """The service over each registry with a state log wired in."""
    log = _Log()
    if request.param == "memory":
        yield CoordinatorService(ArtifactRegistry(state_log=log, instance_id="handoff")), log
        return
    registry = SqliteArtifactRegistry(
        tmp_path / "service-handoff-logged.db", state_log=log, instance_id="handoff"
    )
    yield CoordinatorService(registry), log
    registry.close()


@dataclass(frozen=True)
class _Handoff:
    """One live handoff of ``path`` at version 1.

    ``giver`` and ``successor`` are session-level identities; ``giver_read`` is
    the composite that held the giver's claim and moved INVALID."""

    path: UUID
    giver: UUID
    giver_read: UUID
    successor: UUID


def _hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()


def _path(service: CoordinatorService, name: str = "plan.md") -> UUID:
    """A tracked path at version 1."""
    return service.register_artifact(name=name, content="v1").id


def _read(service: CoordinatorService, path: UUID, composite: UUID) -> None:
    """A standing SHARED read, recorded the way the pre-read route records one."""
    service.registry.set_agent_state(path, composite, MESIState.SHARED, trigger="fetch", tick=1)


def _hold(
    service: CoordinatorService, path: UUID, composite: UUID, session: UUID, shape: MESIState
) -> None:
    """Give ``composite`` (an agent of ``session``) a claim in ``shape``."""
    if shape is MESIState.SHARED:
        _read(service, path, composite)
        return
    service.write(agent_id=composite, artifact_id=path, caller=session)


def _transfer(
    service: CoordinatorService,
    giver: UUID,
    successor: UUID,
    holders: dict[UUID, UUID],
    **kwargs: Any,
) -> list[TransferGrantOutcome]:
    kwargs.setdefault("successor_known", True)
    return service.transfer(giver=giver, successor=successor, holders=holders, **kwargs)


def _hand_off(
    service: CoordinatorService, *, shape: MESIState = MESIState.SHARED, name: str = "plan.md"
) -> _Handoff:
    path = _path(service, name)
    giver, giver_read, successor = uuid4(), uuid4(), uuid4()
    _hold(service, path, giver_read, giver, shape)
    (outcome,) = _transfer(service, giver, successor, {path: giver_read})
    assert outcome.transferred, outcome
    return _Handoff(path=path, giver=giver, giver_read=giver_read, successor=successor)


def _cas(
    service: CoordinatorService,
    path: UUID,
    composite: UUID,
    *,
    expected: int = 1,
    caller: UUID | None = None,
    body: str = "edit",
) -> Any:
    return service.commit_cas(
        agent_id=composite,
        artifact_id=path,
        expected_version=expected,
        content_hash=_hash(body),
        caller=caller,
    )


def _label(service: CoordinatorService, path: UUID) -> tuple[str, UUID | None, bool]:
    """The record's (status, counterparty, live)."""
    record, live = service.registry.get_transfer_record(path)
    return record.status, record.counterparty, live


def _assert_fenced(exc: GiverFenced, handoff: _Handoff) -> None:
    assert type(exc) is GiverFenced
    assert (exc.reason, exc.artifact_id, exc.successor, exc.version_at_transfer) == (
        "handed_off",
        handoff.path,
        handoff.successor,
        1,
    )


def _absent_record_form(
    self: CoordinatorService, artifact_id: UUID, identity: UUID
) -> TransferRecord | None:
    """The rejected giver predicate: it refuses when NO record exists, the
    shape the read-generation fence would take without its ``is not None``."""
    read = self.registry.get_transfer_record(artifact_id)
    if read is None:
        return TransferRecord(
            artifact_id=artifact_id,
            giver=identity,
            holder=identity,
            successor=UUID(int=0),
            version_at_transfer=self.registry.get_artifact(artifact_id).version,
            hold_shape=MESIState.SHARED,
            cause="handoff",
            superseded_successor=None,
            status="pending",
            counterparty=None,
            created_at=0.0,
            updated_at=0.0,
        )
    record, live = read
    return record if live and record.giver == identity else None


# ---------------------------------------------------------------------------
# The giver fence on every write path
# ---------------------------------------------------------------------------


def test_a_reminted_givers_cas_at_the_transfer_version_is_refused(service) -> None:
    """The measured sequence: A reads, transfers to B, its client
    re-mints, and the fresh incarnation commits at the transfer version with A's
    session-level identity. Today that wins and overwrites the file; it must be
    refused with the giver reason naming B and v1, with nothing moved. Fails if
    the fence keys on the composite, which a re-mint changes."""
    handoff = _hand_off(service)
    before = service.registry.get_artifact(handoff.path)

    with pytest.raises(GiverFenced) as caught:
        _cas(service, handoff.path, uuid4(), caller=handoff.giver, body="late-from-A")

    _assert_fenced(caught.value, handoff)
    assert service.registry.get_artifact(handoff.path) == before
    assert service.registry.get_artifact(handoff.path).version == 1


def test_a_giver_subagent_holding_its_own_read_is_refused_alike(service) -> None:
    """A subagent of the giver's session that read the path itself keeps a
    SHARED row the transfer did not move (only the presented composite moves).
    Its compare-and-swap, sent with the parent's session-level identity, is
    refused like the parent's. Fails if the fence reads the committer's own row."""
    path = _path(service)
    giver, giver_read, giver_subagent, successor = uuid4(), uuid4(), uuid4(), uuid4()
    _read(service, path, giver_read)
    _read(service, path, giver_subagent)
    _transfer(service, giver, successor, {path: giver_read})
    assert service.registry.get_agent_state(path, giver_subagent) is MESIState.SHARED

    with pytest.raises(GiverFenced) as caught:
        _cas(service, path, giver_subagent, caller=giver)

    _assert_fenced(caught.value, _Handoff(path, giver, giver_read, successor))
    assert service.registry.get_artifact(path).version == 1


def _plain_cas(service: CoordinatorService) -> tuple[UUID, Any]:
    """The plain optimistic writer: SHARED after two fetches, no record anywhere."""
    path = _path(service)
    writer, peer = uuid4(), uuid4()
    service.fetch(FetchRequest(artifact_id=path, requesting_agent_id=writer, requested_at_tick=1))
    service.fetch(FetchRequest(artifact_id=path, requesting_agent_id=peer, requested_at_tick=2))
    assert service.registry.get_agent_state(path, writer) is MESIState.SHARED
    return path, _cas(service, path, writer)


def test_the_plain_cas_with_no_record_wins(service) -> None:
    """The shipped present-record predicate: a path never handed off
    fences nobody, so the plain optimistic writer wins at v1 and lands v2. Fails
    if the predicate refuses on an absent record."""
    path, result = _plain_cas(service)

    assert isinstance(result, tuple), result
    assert result[0].version == 2
    assert service.registry.get_transfer_record(path) is None


def test_an_absent_record_predicate_would_refuse_the_plain_cas(
    service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mutation: with the giver member rewritten to refuse when no
    record exists, the same plain scenario answers the giver reason. Proves the
    compare-and-swap consults the member, and that the scenario above has no
    record for an absent-record check to trip on."""
    monkeypatch.setattr(CoordinatorService, "live_handoff_given_by", _absent_record_form)

    with pytest.raises(GiverFenced):
        _plain_cas(service)


def test_a_library_cas_with_no_caller_identity_is_admitted(service) -> None:
    """The library default: with no caller identity the composite itself is
    compared, and a library agent's own id never equals a session-level giver,
    so an in-process writer is not fenced by an HTTP giver's record. Fails if
    the fence refuses every writer on a path with a live record."""
    handoff = _hand_off(service)
    library_agent = uuid4()
    _read(service, handoff.path, library_agent)

    result = _cas(service, handoff.path, library_agent)

    assert isinstance(result, tuple), result
    assert result[0].version == 2


def test_a_library_giver_with_no_caller_identity_is_refused(service) -> None:
    """The other half of the library default: a library giver names its own agent id
    as giver, so the composite compared when no identity is passed IS the giver
    and its compare-and-swap is fenced."""
    path = _path(service)
    library_giver, successor = uuid4(), uuid4()
    _read(service, path, library_giver)
    _transfer(service, library_giver, successor, {path: library_giver})

    with pytest.raises(GiverFenced):
        _cas(service, path, library_giver)


def test_the_givers_pessimistic_acquire_is_refused_and_leaves_the_successor_alone(
    service,
) -> None:
    """The giver fence on the pessimistic acquire: the giver's re-acquire is refused before
    it invalidates anyone, so the successor's SHARED row and its read generation
    are untouched and its compare-and-swap at the transfer version wins rather
    than meeting the giver as another holder."""
    handoff = _hand_off(service)
    successor_read = uuid4()
    _read(service, handoff.path, successor_read)
    read_generation = service.registry.get_read_generation(handoff.path, successor_read)

    with pytest.raises(GiverFenced) as caught:
        service.write(agent_id=uuid4(), artifact_id=handoff.path, caller=handoff.giver)

    _assert_fenced(caught.value, handoff)
    assert service.registry.get_agent_state(handoff.path, successor_read) is MESIState.SHARED
    assert service.registry.get_read_generation(handoff.path, successor_read) == read_generation
    result = _cas(service, handoff.path, successor_read, caller=handoff.successor)
    assert isinstance(result, tuple), result
    assert result[0].version == 2


def test_the_givers_pessimistic_commit_is_refused_ahead_of_the_not_allowed_reason(
    service,
) -> None:
    """The giver fence on the pessimistic commit: the typed giver refusal comes before the
    not-EXCLUSIVE-or-MODIFIED branch, so a giver whose grant was handed off is
    never told commit_not_allowed and never shown a reclaim slot it did not
    earn, even when one is recorded for its composite."""
    handoff = _hand_off(service, shape=MESIState.EXCLUSIVE)
    service.registry.record_last_reclamation(
        handoff.giver_read, handoff.path, "reclaim_heartbeat", 7
    )

    with pytest.raises(GiverFenced) as caught:
        service.commit(
            agent_id=handoff.giver_read,
            artifact_id=handoff.path,
            content="late",
            content_hash=_hash("late"),
            caller=handoff.giver,
        )

    _assert_fenced(caught.value, handoff)
    assert "reclaimed_by" not in str(caught.value)
    assert "commit_not_allowed" not in str(caught.value)
    assert service.registry.get_artifact(handoff.path).version == 1


def test_the_givers_batch_commit_including_the_handed_path_lands_nothing(service) -> None:
    """The giver fence on the batch commit: one handed path refuses the whole batch with
    the giver reason, and the path that was not handed off -- listed first --
    does not land either."""
    handoff = _hand_off(service, name="plan.md")
    other = _path(service, "notes.md")
    remint = uuid4()

    with pytest.raises(GiverFenced) as caught:
        service.commit_all(
            agent_id=remint,
            writes={
                other: CommitAllEntry(expected_version=1, content_hash=_hash("o")),
                handoff.path: CommitAllEntry(expected_version=1, content_hash=_hash("p")),
            },
            caller=handoff.giver,
        )

    _assert_fenced(caught.value, handoff)
    assert service.registry.get_artifact(other).version == 1
    assert service.registry.get_artifact(handoff.path).version == 1


def test_a_snapshot_session_commit_by_an_in_process_giver_is_refused(service) -> None:
    """The giver fence on the snapshot-session wrappers: they gain no argument and forward
    the owner they already validate as the caller identity, so an in-process
    giver's session commit, its session batch commit and the effect gate's
    atomic commit are refused with the giver reason (raised, never a new hold
    arm) and nothing lands. Fails if the wrappers compare only the committer id
    they mint from the session token."""
    path, other = _path(service, "plan.md"), _path(service, "notes.md")
    owner, successor = uuid4(), uuid4()
    _read(service, path, owner)
    _transfer(service, owner, successor, {path: owner})
    token = service.begin_session(read_set=[path, other], owner=owner).session_token

    with pytest.raises(GiverFenced):
        service.session_commit(token, path, "late", caller=owner)
    with pytest.raises(GiverFenced):
        service.session_commit_all(
            token, {other: ("o", None), path: ("p", None)}, caller=owner
        )
    with pytest.raises(GiverFenced):
        service.effect_gate(
            read_set=[path], owner=owner, decide=lambda view: None, commit=(path, "g")
        )

    assert service.registry.get_artifact(path).version == 1
    assert service.registry.get_artifact(other).version == 1


# ---------------------------------------------------------------------------
# What ends the fence: a decline, a withdraw or a version move
# ---------------------------------------------------------------------------


def test_an_accept_keeps_the_giver_refused(service) -> None:
    """The successor's accept without a write marks the record
    completed and leaves the fence standing -- completion alone never lifts it,
    or the lost update re-opens through the successor's accept."""
    handoff = _hand_off(service)

    answer = service.accept_transfer(artifact_id=handoff.path, caller=handoff.successor)

    assert answer == TransferVerbOutcome(
        artifact_id=handoff.path, ok=True, status="completed"
    )
    assert _label(service, handoff.path) == ("completed", None, True)
    assert service.live_handoff_given_by(handoff.path, handoff.giver).status == "completed"
    with pytest.raises(GiverFenced):
        _cas(service, handoff.path, uuid4(), caller=handoff.giver)


def test_a_decline_lifts_the_fence_on_the_next_commit(service) -> None:
    """After the successor declines, the giver's very next
    compare-and-swap is decided by the ordinary rules and wins."""
    handoff = _hand_off(service)

    answer = service.decline_transfer(artifact_id=handoff.path, caller=handoff.successor)

    assert answer == TransferVerbOutcome(artifact_id=handoff.path, ok=True, status="declined")
    result = _cas(service, handoff.path, uuid4(), caller=handoff.giver)
    assert isinstance(result, tuple), result
    assert result[0].version == 2


def test_after_the_successors_write_the_giver_meets_a_version_mismatch(service) -> None:
    """Once the successor's commit moved the version, the giver's
    compare-and-swap at the transfer version answers the existing version
    mismatch, not the giver reason. Fails if the fence ignores liveness."""
    handoff = _hand_off(service)
    won = _cas(service, handoff.path, uuid4(), caller=handoff.successor)
    assert isinstance(won, tuple), won

    result = _cas(service, handoff.path, uuid4(), caller=handoff.giver)

    assert result == ConflictDetail("version_mismatch", 2)


def test_a_withdraw_lifts_the_fence_for_a_read_shape_giver(service) -> None:
    """The giver's withdraw ends the record and its next
    compare-and-swap is admitted by the ordinary rules: a SHARED-shape handoff
    moved no epoch, so even the composite that read before it wins."""
    handoff = _hand_off(service)

    answer = service.withdraw_transfer(artifact_id=handoff.path, caller=handoff.giver)

    assert answer == TransferVerbOutcome(artifact_id=handoff.path, ok=True, status="withdrawn")
    assert _label(service, handoff.path) == ("withdrawn", None, False)
    result = _cas(service, handoff.path, handoff.giver_read, caller=handoff.giver)
    assert isinstance(result, tuple), result


def test_a_write_shape_giver_stays_stale_until_a_fresh_generation(service) -> None:
    """A withdrawn EXCLUSIVE-shape handoff still moved the epoch, so
    the giver's old composite is refused as a stale read generation, and a mere
    pre-read (which never re-captures a generation) does not change that; a
    re-minted incarnation, which carries no stale generation, wins."""
    handoff = _hand_off(service, shape=MESIState.EXCLUSIVE)
    service.withdraw_transfer(artifact_id=handoff.path, caller=handoff.giver)

    stale = _cas(service, handoff.path, handoff.giver_read, caller=handoff.giver)
    service.registry.set_agent_state(
        handoff.path, handoff.giver_read, MESIState.SHARED, trigger="post_stale_read", tick=3
    )
    still_stale = _cas(service, handoff.path, handoff.giver_read, caller=handoff.giver)
    reminted = _cas(service, handoff.path, uuid4(), caller=handoff.giver)

    assert stale == ConflictDetail("stale_read_generation", 1)
    assert still_stale == ConflictDetail("stale_read_generation", 1)
    assert isinstance(reminted, tuple), reminted


def test_a_withdraw_by_another_session_is_refused(service) -> None:
    """Only the giver withdraws; anyone else is refused as not the
    giver, carrying the record's status, and the record is left live."""
    handoff = _hand_off(service)

    answer = service.withdraw_transfer(artifact_id=handoff.path, caller=uuid4())

    assert answer == TransferVerbOutcome(
        artifact_id=handoff.path, ok=False, reason="handoff_not_giver", status="pending"
    )
    assert _label(service, handoff.path) == ("pending", None, True)


@pytest.mark.parametrize("ending", ["withdraw", "decline"])
def test_after_a_bystanders_acquire_a_withdraw_or_decline_still_lifts_the_fence(
    service, ending: str
) -> None:
    """A bystander's pessimistic acquire with no commit leaves the
    record overtaken and LIVE (the version did not move), so the giver is still
    fenced; the giver's withdraw and the successor's decline both apply to an
    overtaken record and lift the fence, after which the giver meets the ordinary
    other-holder answer rather than the giver reason."""
    handoff = _hand_off(service)
    bystander, bystander_inc = uuid4(), uuid4()
    service.write(agent_id=bystander_inc, artifact_id=handoff.path, caller=bystander)
    assert _label(service, handoff.path) == ("overtaken", bystander, True)

    if ending == "withdraw":
        answer = service.withdraw_transfer(artifact_id=handoff.path, caller=handoff.giver)
    else:
        answer = service.decline_transfer(artifact_id=handoff.path, caller=handoff.successor)

    assert answer.ok is True
    assert service.live_handoff_given_by(handoff.path, handoff.giver) is None
    result = _cas(service, handoff.path, uuid4(), caller=handoff.giver)
    assert result == ConflictDetail("other_holder", 1)


# ---------------------------------------------------------------------------
# Accept and decline: the successor's verbs
# ---------------------------------------------------------------------------


def test_accept_and_decline_by_anyone_but_the_successor_are_refused(service) -> None:
    """A session that is not the live record's successor -- the giver
    included -- is refused carrying the record's status, and nothing changes."""
    handoff = _hand_off(service)

    accepts = service.accept_transfer(artifact_id=handoff.path, caller=handoff.giver)
    declines = service.decline_transfer(artifact_id=handoff.path, caller=uuid4())

    refused = TransferVerbOutcome(
        artifact_id=handoff.path, ok=False, reason="handoff_not_successor", status="pending"
    )
    assert (accepts, declines) == (refused, refused)
    assert _label(service, handoff.path) == ("pending", None, True)


def test_a_verb_on_a_record_that_is_not_live_is_refused_with_its_status(service) -> None:
    """After a decline the record has ended, so a second decline, an
    accept and a withdraw are each refused as not live carrying the declined
    status, and the record is unchanged."""
    handoff = _hand_off(service)
    service.decline_transfer(artifact_id=handoff.path, caller=handoff.successor)
    before = service.registry.get_transfer_record(handoff.path)

    answers = [
        service.decline_transfer(artifact_id=handoff.path, caller=handoff.successor),
        service.accept_transfer(artifact_id=handoff.path, caller=handoff.successor),
        service.withdraw_transfer(artifact_id=handoff.path, caller=handoff.giver),
    ]

    not_live = TransferVerbOutcome(
        artifact_id=handoff.path, ok=False, reason="handoff_not_live", status="declined"
    )
    assert answers == [not_live, not_live, not_live]
    assert service.registry.get_transfer_record(handoff.path) == before


def test_a_verb_on_a_path_with_no_record_is_not_live_with_no_status(service) -> None:
    """A path never handed off has nothing to accept, decline or withdraw: the
    answer is not live and names no status."""
    path = _path(service)
    caller = uuid4()

    answers = [
        service.accept_transfer(artifact_id=path, caller=caller),
        service.decline_transfer(artifact_id=path, caller=caller),
        service.withdraw_transfer(artifact_id=path, caller=caller),
    ]

    none_live = TransferVerbOutcome(artifact_id=path, ok=False, reason="handoff_not_live")
    assert answers == [none_live, none_live, none_live]
    assert service.registry.get_transfer_record(path) is None


def test_an_accept_of_a_completed_record_succeeds_and_changes_nothing(service) -> None:
    """Accepting a record already completed answers success and writes
    nothing -- not even its updated time."""
    handoff = _hand_off(service)
    service.accept_transfer(artifact_id=handoff.path, caller=handoff.successor)
    before = service.registry.get_transfer_record(handoff.path)

    answer = service.accept_transfer(artifact_id=handoff.path, caller=handoff.successor)

    assert answer == TransferVerbOutcome(artifact_id=handoff.path, ok=True, status="completed")
    assert service.registry.get_transfer_record(handoff.path) == before


def test_an_accept_after_a_bystanders_acquire_answers_overtaken(service) -> None:
    """Once a bystander acquired, the successor's accept changes
    nothing and answers the overtaken status with that bystander as
    counterparty; an accept never replaces an overtaken label."""
    handoff = _hand_off(service)
    bystander = uuid4()
    service.write(agent_id=uuid4(), artifact_id=handoff.path, caller=bystander)

    answer = service.accept_transfer(artifact_id=handoff.path, caller=handoff.successor)

    assert answer == TransferVerbOutcome(
        artifact_id=handoff.path, ok=True, status="overtaken", counterparty=bystander
    )
    assert _label(service, handoff.path) == ("overtaken", bystander, True)


# ---------------------------------------------------------------------------
# Completion and overtake on an admitted acquire or win
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("writer", ["successor", "third session"])
def test_a_cas_win_completes_or_overtakes_by_the_caller_identity(
    service, writer: str
) -> None:
    """A win by a fresh composite carrying the successor's
    session-level identity marks the record completed; the same win carrying a
    third session's identity marks it overtaken with that identity as
    counterparty. Fails if the label compares the composite agent id, which a
    successor's re-minted incarnation never matches."""
    handoff = _hand_off(service)
    identity = handoff.successor if writer == "successor" else uuid4()

    result = _cas(service, handoff.path, uuid4(), caller=identity)

    assert isinstance(result, tuple), result
    expected = ("completed", None) if writer == "successor" else ("overtaken", identity)
    assert _label(service, handoff.path) == (*expected, False)


def test_the_successors_acquire_completes_the_record_and_a_read_does_not(service) -> None:
    """A read never completes a handoff; the successor's pessimistic
    acquire does, and the record stays live (the version did not move), so the
    giver stays fenced."""
    handoff = _hand_off(service)
    successor_inc = uuid4()
    service.fetch(
        FetchRequest(artifact_id=handoff.path, requesting_agent_id=successor_inc, requested_at_tick=2)
    )
    assert _label(service, handoff.path) == ("pending", None, True)

    service.write(agent_id=successor_inc, artifact_id=handoff.path, caller=handoff.successor)

    assert _label(service, handoff.path) == ("completed", None, True)
    with pytest.raises(GiverFenced):
        _cas(service, handoff.path, uuid4(), caller=handoff.giver)


@pytest.mark.parametrize("writer", ["successor", "third session"])
def test_a_batch_commit_win_completes_or_overtakes(service, writer: str) -> None:
    """On the batch commit, the win labels each handed member by the
    caller identity, exactly as a single compare-and-swap would."""
    handoff = _hand_off(service, name="plan.md")
    other = _path(service, "notes.md")
    identity = handoff.successor if writer == "successor" else uuid4()

    result = service.commit_all(
        agent_id=uuid4(),
        writes={
            other: CommitAllEntry(expected_version=1, content_hash=_hash("o")),
            handoff.path: CommitAllEntry(expected_version=1, content_hash=_hash("p")),
        },
        caller=identity,
    )

    assert isinstance(result, tuple) and isinstance(result[0], MultiCommitResult), result
    expected = ("completed", None) if writer == "successor" else ("overtaken", identity)
    assert _label(service, handoff.path) == (*expected, False)
    assert service.registry.get_transfer_record(other) is None


def test_a_successors_snapshot_session_win_completes_the_record(service) -> None:
    """The session commit forwards its owner as the caller identity, so a
    successor's snapshot-session win completes the record instead of being
    recorded as an overtake by the committer id minted from the token."""
    handoff = _hand_off(service)
    token = service.begin_session(read_set=[handoff.path], owner=handoff.successor).session_token

    result = service.session_commit(token, handoff.path, "next", caller=handoff.successor)

    assert isinstance(result, tuple), result
    assert _label(service, handoff.path) == ("completed", None, False)


def test_a_bystanders_win_after_an_accept_overtakes_the_record(service) -> None:
    """The latest admitted write names the label, so a bystander's win
    after the successor's accept replaces completed with overtaken, naming the
    bystander -- the giver's next touch must not report completed."""
    handoff = _hand_off(service)
    service.accept_transfer(artifact_id=handoff.path, caller=handoff.successor)
    bystander = uuid4()

    result = _cas(service, handoff.path, uuid4(), caller=bystander)

    assert isinstance(result, tuple), result
    assert _label(service, handoff.path) == ("overtaken", bystander, False)


def test_the_successors_acquire_after_an_overtake_completes_the_record(service) -> None:
    """Overtaken and completed replace each other on a live record -- the
    successor's acquire after a bystander's names it completed again."""
    handoff = _hand_off(service)
    service.write(agent_id=uuid4(), artifact_id=handoff.path, caller=uuid4())

    service.write(agent_id=uuid4(), artifact_id=handoff.path, caller=handoff.successor)

    assert _label(service, handoff.path) == ("completed", None, True)


def test_a_win_after_the_record_ended_never_relabels_it(service) -> None:
    """Only a record live when the win was admitted is labelled. After
    the successor's win ended it, a bystander's later win leaves it completed;
    after a withdraw, the giver's own later win leaves it withdrawn."""
    ended_by_win = _hand_off(service, name="plan.md")
    _cas(service, ended_by_win.path, uuid4(), caller=ended_by_win.successor)
    ended_by_withdraw = _hand_off(service, name="notes.md")
    service.withdraw_transfer(artifact_id=ended_by_withdraw.path, caller=ended_by_withdraw.giver)

    later = _cas(service, ended_by_win.path, uuid4(), expected=2, caller=uuid4())
    own = _cas(service, ended_by_withdraw.path, uuid4(), caller=ended_by_withdraw.giver)

    assert isinstance(later, tuple) and isinstance(own, tuple), (later, own)
    assert _label(service, ended_by_win.path) == ("completed", None, False)
    assert _label(service, ended_by_withdraw.path) == ("withdrawn", None, False)


def test_a_label_write_that_raises_after_a_win_leaves_the_win_answered(
    service, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The label write after a win is best-effort -- the win already
    landed, so a raise there must not turn it into a failure the client would
    retry. The record then reads not live by its version, still pending: the
    outcome of a write whose writer was not recorded."""
    handoff = _hand_off(service)

    def refuse(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected label-write failure")

    monkeypatch.setattr(service.registry, "set_transfer_status", refuse)

    result = _cas(service, handoff.path, uuid4(), caller=handoff.successor)

    assert isinstance(result, tuple), result
    assert result[0].version == 2
    assert _label(service, handoff.path) == ("pending", None, False)


# ---------------------------------------------------------------------------
# The transfer through the service
# ---------------------------------------------------------------------------


def test_an_empty_transfer_is_refused_before_it_reaches_the_registry(service) -> None:
    """Success means every named grant transferred, so a request naming no
    path must not answer a vacuous success."""
    with pytest.raises(CoherenceError, match="at least one path"):
        service.transfer(giver=uuid4(), successor=uuid4(), holders={}, successor_known=True)


def test_a_mixed_transfer_answers_per_grant(service) -> None:
    """Each grant is answered on its own -- the held
    path transferred with the session-level ids, the path with no confirmed
    version refused with no record -- so the top-level answer is false and the
    refused grant is left exactly as it was."""
    held = _path(service, "a.md")
    fresh = uuid4()
    service.registry.register_artifact(Artifact(id=fresh, name="new.md", version=0), "")
    giver, giver_read, successor = uuid4(), uuid4(), uuid4()
    _read(service, held, giver_read)
    _read(service, fresh, giver_read)

    outcomes = _transfer(service, giver, successor, {held: giver_read, fresh: giver_read})

    assert outcomes == [
        TransferGrantOutcome(
            artifact_id=held,
            transferred=True,
            giver=giver,
            successor=successor,
            version_at_transfer=1,
            hold_shape=MESIState.SHARED,
            status="pending",
        ),
        TransferGrantOutcome(
            artifact_id=fresh, transferred=False, reason="handoff_version_unconfirmed"
        ),
    ]
    assert not all(outcome.transferred for outcome in outcomes)
    assert service.registry.get_transfer_record(fresh) is None
    assert service.registry.get_agent_state(fresh, giver_read) is MESIState.SHARED
    assert service.registry.get_agent_state(held, giver_read) is MESIState.INVALID


def test_a_transfer_to_the_callers_own_session_is_refused_through_the_service(service) -> None:
    """A successor that normalises to the caller's own session -- its
    own subagent included -- is refused as self and the read is left alone."""
    path = _path(service)
    giver, giver_read = uuid4(), uuid4()
    _read(service, path, giver_read)

    (outcome,) = _transfer(service, giver, giver, {path: giver_read})

    assert outcome == TransferGrantOutcome(
        artifact_id=path, transferred=False, reason="handoff_to_self"
    )
    assert service.registry.get_agent_state(path, giver_read) is MESIState.SHARED


def test_an_exact_resend_answers_the_current_status_with_no_second_record(
    service,
) -> None:
    """A re-send of the same tuple -- whose presented composite is
    already INVALID, with no read in between -- is answered transferred at the
    same version with the record's current status and writes no second record;
    once the successor has declined it is refused carrying the declined status."""
    handoff = _hand_off(service)
    first = service.registry.get_transfer_record(handoff.path)

    (resent,) = _transfer(service, handoff.giver, handoff.successor, {handoff.path: handoff.giver_read})
    after_resend = service.registry.get_transfer_record(handoff.path)
    service.accept_transfer(artifact_id=handoff.path, caller=handoff.successor)
    (after_accept,) = _transfer(
        service, handoff.giver, handoff.successor, {handoff.path: handoff.giver_read}
    )
    service.decline_transfer(artifact_id=handoff.path, caller=handoff.successor)
    (after_decline,) = _transfer(
        service, handoff.giver, handoff.successor, {handoff.path: handoff.giver_read}
    )

    assert (resent.transferred, resent.version_at_transfer, resent.status) == (True, 1, "pending")
    assert after_resend == first
    assert (after_accept.transferred, after_accept.status) == (True, "completed")
    assert (after_decline.transferred, after_decline.reason, after_decline.status) == (
        False,
        "handoff_ended",
        "declined",
    )


def test_a_giver_transfers_by_presenting_the_incarnation_that_read(service) -> None:
    """A volume that read under one incarnation and re-minted for a
    compare-and-swap on another path hands the read path on by presenting the
    incarnation that holds the read; the incarnation it now writes under holds
    nothing there."""
    path, other = _path(service, "plan.md"), _path(service, "other.md")
    giver, read_incarnation, write_incarnation, successor = uuid4(), uuid4(), uuid4(), uuid4()
    _read(service, path, read_incarnation)
    assert isinstance(_cas(service, other, write_incarnation, caller=giver), tuple)

    (wrong,) = _transfer(service, giver, successor, {path: write_incarnation})
    (right,) = _transfer(service, giver, successor, {path: read_incarnation})

    assert (wrong.transferred, wrong.reason) == (False, "handoff_not_held")
    assert (right.transferred, right.version_at_transfer) == (True, 1)


def test_another_session_cannot_hand_on_a_path_with_a_live_record(service) -> None:
    """C's transfer of a path A handed to B is refused naming the
    pending pair and leaves A->B as it was; A's own re-transfer to D supersedes
    it, the replacing record's cause records the supersession, and A's late
    re-send of the A->B tuple answers the superseded status."""
    handoff = _hand_off(service)
    other_session, other_read, other_successor = uuid4(), uuid4(), uuid4()
    _read(service, handoff.path, other_read)
    before = service.registry.get_transfer_record(handoff.path)

    (by_other,) = _transfer(service, other_session, other_successor, {handoff.path: other_read})
    unchanged = service.registry.get_transfer_record(handoff.path)
    (superseding,) = _transfer(
        service, handoff.giver, other_successor, {handoff.path: handoff.giver_read}
    )
    (late,) = _transfer(service, handoff.giver, handoff.successor, {handoff.path: handoff.giver_read})

    assert (by_other.reason, by_other.giver, by_other.successor) == (
        "handoff_in_flight",
        handoff.giver,
        handoff.successor,
    )
    assert unchanged == before
    assert superseding.transferred is True
    record, live = service.registry.get_transfer_record(handoff.path)
    assert (record.successor, record.cause, record.superseded_successor, live) == (
        other_successor,
        "supersession",
        handoff.successor,
        True,
    )
    assert (late.reason, late.status) == ("handoff_ended", "superseded")


def test_a_write_shape_supersession_moves_no_row_and_bumps_nothing(logged_service) -> None:
    """``decide_transfer_grant`` arm (c): a hook-path EXCLUSIVE giver, whose
    row is already INVALID under the handoff trigger, re-transfers to a new
    successor while the first record is live. The old successor reads superseded through the replacing
    record's cause, and no second INVALID move or epoch bump is recorded."""
    service, log = logged_service
    handoff = _hand_off(service, shape=MESIState.EXCLUSIVE)
    generation = service.registry.get_owner_generation(handoff.path)
    logged = len(log.entries)
    second_successor = uuid4()

    (outcome,) = _transfer(
        service, handoff.giver, second_successor, {handoff.path: handoff.giver_read}
    )
    (old_tuple,) = _transfer(
        service, handoff.giver, handoff.successor, {handoff.path: handoff.giver_read}
    )

    assert outcome.transferred is True
    assert old_tuple.status == "superseded"
    record, live = service.registry.get_transfer_record(handoff.path)
    assert (record.successor, record.superseded_successor, record.hold_shape, live) == (
        second_successor,
        handoff.successor,
        MESIState.EXCLUSIVE,
        True,
    )
    assert service.registry.get_owner_generation(handoff.path) == generation
    assert log.entries[logged:] == []


@pytest.mark.parametrize("next_successor", ["same", "different"])
def test_a_write_shape_giver_hands_a_path_on_again_after_its_handoff_ended(
    logged_service, next_successor: str
) -> None:
    """``decide_transfer_grant`` arm (e): once the successor's write ended
    the first handoff, the giver re-acquires, commits and transfers again. The ended record is treated
    as absent, so the new grant moves INVALID under the handoff trigger with
    exactly one epoch bump and the new record is a plain handoff; the new
    successor's compare-and-swap at the new transfer version then wins rather
    than meeting the giver's MODIFIED row as another holder."""
    service, log = logged_service
    handoff = _hand_off(service, shape=MESIState.EXCLUSIVE)
    assert isinstance(_cas(service, handoff.path, uuid4(), caller=handoff.successor), tuple)
    service.write(agent_id=handoff.giver_read, artifact_id=handoff.path, caller=handoff.giver)
    service.commit(
        agent_id=handoff.giver_read,
        artifact_id=handoff.path,
        content="v3",
        content_hash=_hash("v3"),
        caller=handoff.giver,
    )
    successor = handoff.successor if next_successor == "same" else uuid4()
    generation = service.registry.get_owner_generation(handoff.path)
    logged = len(log.entries)

    (outcome,) = _transfer(service, handoff.giver, successor, {handoff.path: handoff.giver_read})

    assert (outcome.transferred, outcome.version_at_transfer, outcome.hold_shape) == (
        True,
        3,
        MESIState.MODIFIED,
    )
    assert [(e["agent_id"], e["from_state"], e["trigger"]) for e in log.entries[logged:]] == [
        (str(handoff.giver_read), "MODIFIED", "handoff")
    ]
    assert service.registry.get_owner_generation(handoff.path) == generation + 1
    record, live = service.registry.get_transfer_record(handoff.path)
    assert (record.cause, record.superseded_successor, record.status, live) == (
        "handoff",
        None,
        "pending",
        True,
    )
    result = _cas(service, handoff.path, uuid4(), expected=3, caller=successor)
    assert isinstance(result, tuple), result


def test_a_resend_while_holding_nothing_after_the_handoff_ended_is_not_held(
    service,
) -> None:
    """Once the successor's write moved the version, the giver's re-send
    of the same request -- presenting a composite that holds nothing -- is
    refused as not held, and the ended record is left as it was."""
    handoff = _hand_off(service)
    _cas(service, handoff.path, uuid4(), caller=handoff.successor)
    ended = service.registry.get_transfer_record(handoff.path)

    (outcome,) = _transfer(service, handoff.giver, handoff.successor, {handoff.path: handoff.giver_read})

    assert (outcome.transferred, outcome.reason) == (False, "handoff_not_held")
    assert service.registry.get_transfer_record(handoff.path) == ended


@pytest.mark.parametrize("ending", ["decline", "withdraw"])
@pytest.mark.parametrize("shape", [MESIState.SHARED, MESIState.EXCLUSIVE])
def test_after_a_decline_or_withdraw_a_new_successor_meets_the_ordinary_checks(
    service, ending: str, shape: MESIState
) -> None:
    """``decide_transfer_grant`` arm (e): a record ended by decline or withdraw at an unmoved
    version, re-sent to a DIFFERENT successor, is treated as absent: refused as
    not held while the giver holds nothing, then admitted with the INVALID move
    (and an epoch bump for a write shape) once the giver re-read or re-acquired."""
    handoff = _hand_off(service, shape=shape)
    if ending == "decline":
        service.decline_transfer(artifact_id=handoff.path, caller=handoff.successor)
    else:
        service.withdraw_transfer(artifact_id=handoff.path, caller=handoff.giver)
    ended = service.registry.get_transfer_record(handoff.path)
    new_successor = uuid4()

    (refused,) = _transfer(service, handoff.giver, new_successor, {handoff.path: handoff.giver_read})
    assert (refused.transferred, refused.reason) == (False, "handoff_not_held")
    assert service.registry.get_transfer_record(handoff.path) == ended

    _hold(service, handoff.path, handoff.giver_read, handoff.giver, shape)
    generation = service.registry.get_owner_generation(handoff.path)
    (admitted,) = _transfer(service, handoff.giver, new_successor, {handoff.path: handoff.giver_read})

    assert (admitted.transferred, admitted.hold_shape) == (True, shape)
    assert service.registry.get_agent_state(handoff.path, handoff.giver_read) is MESIState.INVALID
    bump = 1 if shape is MESIState.EXCLUSIVE else 0
    assert service.registry.get_owner_generation(handoff.path) == generation + bump
    assert _label(service, handoff.path) == ("pending", None, True)


def test_a_live_record_refuses_a_retransfer_to_self_unknown_or_past_a_write_holder(
    service,
) -> None:
    """During a live handoff the giver's re-transfer to its own
    session (as its subagent normalises), to an unknown id, or while another
    session holds a write grant on the path is refused for that reason, and the
    live record is unchanged each time."""
    handoff = _hand_off(service)
    holders = {handoff.path: handoff.giver_read}
    before = service.registry.get_transfer_record(handoff.path)

    (to_self,) = _transfer(service, handoff.giver, handoff.giver, holders)
    (to_unknown,) = _transfer(service, handoff.giver, uuid4(), holders, successor_known=False)
    assert service.registry.get_transfer_record(handoff.path) == before
    service.write(agent_id=uuid4(), artifact_id=handoff.path, caller=uuid4())
    overtaken = service.registry.get_transfer_record(handoff.path)
    (past_writer,) = _transfer(service, handoff.giver, uuid4(), holders)

    assert [o.reason for o in (to_self, to_unknown, past_writer)] == [
        "handoff_to_self",
        "handoff_successor_unknown",
        "handoff_other_holder",
    ]
    assert service.registry.get_transfer_record(handoff.path) == overtaken


def test_a_resend_after_a_fresh_read_leaves_that_read_and_keeps_the_fence(
    service,
) -> None:
    """A volume that re-read the path since the transfer presents its
    fresh read on the re-send; the re-send moves no row, so that SHARED read
    stays SHARED, and the session is still fenced by identity."""
    handoff = _hand_off(service)
    fresh_read = uuid4()
    _read(service, handoff.path, fresh_read)

    (outcome,) = _transfer(service, handoff.giver, handoff.successor, {handoff.path: fresh_read})

    assert (outcome.transferred, outcome.status) == (True, "pending")
    assert service.registry.get_agent_state(handoff.path, fresh_read) is MESIState.SHARED
    with pytest.raises(GiverFenced):
        _cas(service, handoff.path, fresh_read, caller=handoff.giver)


@pytest.mark.parametrize(
    ("shape", "bump"), [(MESIState.EXCLUSIVE, 1), (MESIState.SHARED, 0)]
)
def test_a_write_shape_transfer_bumps_the_epoch_and_a_read_shape_does_not(
    service, shape: MESIState, bump: int
) -> None:
    """Handing off an EXCLUSIVE grant moves the ownership epoch as a release
    does; handing off a standing read moves nothing, so bystanders that read
    the path are not fenced as though a writer was reclaimed."""
    path = _path(service)
    giver, giver_read = uuid4(), uuid4()
    _hold(service, path, giver_read, giver, shape)
    generation = service.registry.get_owner_generation(path)

    _transfer(service, giver, uuid4(), {path: giver_read})

    assert service.registry.get_owner_generation(path) == generation + bump


def test_a_watchdog_abort_while_waiting_for_the_lock_lands_nothing(service) -> None:
    """The watchdog fires while a two-path transfer is still waiting
    for the registry lock, so the abort checked at hold entry refuses the whole
    request -- no record and no grant change on either path."""
    first, second = _path(service, "a.md"), _path(service, "b.md")
    giver, giver_read, successor = uuid4(), uuid4(), uuid4()
    _read(service, first, giver_read)
    _read(service, second, giver_read)
    abort = threading.Event()
    raised: list[BaseException] = []

    def run() -> None:
        try:
            _transfer(service, giver, successor, {first: giver_read, second: giver_read}, abort=abort)
        except WatchdogAbandoned as exc:
            raised.append(exc)

    with service.registry.abort_guard():
        worker = threading.Thread(target=run)
        worker.start()
        abort.set()
    worker.join(timeout=10)

    assert len(raised) == 1
    for path in (first, second):
        assert service.registry.get_transfer_record(path) is None
        assert service.registry.get_agent_state(path, giver_read) is MESIState.SHARED


def test_an_abort_set_after_the_hold_is_taken_lands_every_path(logged_service) -> None:
    """An abort that arrives once the transfer holds the lock is not
    re-checked mid-sequence, so both paths land together (the client, already
    answered unconfirmed, reads both records next) -- never one without the
    other."""
    service, log = logged_service
    first, second = _path(service, "a.md"), _path(service, "b.md")
    giver, giver_read, successor = uuid4(), uuid4(), uuid4()
    _read(service, first, giver_read)
    _read(service, second, giver_read)
    abort = threading.Event()
    log.hook = abort.set

    outcomes = _transfer(service, giver, successor, {first: giver_read, second: giver_read}, abort=abort)

    assert abort.is_set()
    assert [o.transferred for o in outcomes] == [True, True]
    for path in (first, second):
        assert _label(service, path) == ("pending", None, True)
        assert service.registry.get_agent_state(path, giver_read) is MESIState.INVALID


# ---------------------------------------------------------------------------
# A session stop and the giver's failed edit change no record
# ---------------------------------------------------------------------------


def test_a_session_stop_from_either_party_leaves_the_record(service) -> None:
    """A session stop releases the session's EXCLUSIVE and
    MODIFIED grants as today and never withdraws or declines. The successor's
    stop after acquiring the handed path, and the giver's stop releasing its
    write grant elsewhere, leave the record as it was."""
    handoff = _hand_off(service, name="plan.md")
    elsewhere = _path(service, "notes.md")
    giver_inc, successor_inc = uuid4(), uuid4()
    service.write(agent_id=giver_inc, artifact_id=elsewhere, caller=handoff.giver)
    service.write(agent_id=successor_inc, artifact_id=handoff.path, caller=handoff.successor)
    before = service.registry.get_transfer_record(handoff.path)

    for composite, path in ((successor_inc, handoff.path), (giver_inc, elsewhere)):
        service.invalidate(
            agent_id=composite,
            artifact_id=path,
            new_version=service.registry.get_artifact(path).version,
            issuer_agent_id=composite,
            issued_at_tick=5,
        )

    assert service.registry.get_transfer_record(handoff.path) == before
    assert _label(service, handoff.path) == ("completed", None, True)


def test_the_giver_member_answers_what_a_failed_edit_reports(service) -> None:
    """A post-edit reporting failure from the giver of a live record is
    answered not held and handed to the successor at the transfer version, and
    changes nothing. The read-only member names exactly that for the giver's
    session-level identity -- and nothing for the successor or a bystander, whose
    failed edits release as before -- and reading it changes nothing."""
    handoff = _hand_off(service)
    before = service.registry.get_transfer_record(handoff.path)

    record = service.live_handoff_given_by(handoff.path, handoff.giver)

    assert (record.successor, record.version_at_transfer) == (handoff.successor, 1)
    assert service.live_handoff_given_by(handoff.path, handoff.successor) is None
    assert service.live_handoff_given_by(handoff.path, uuid4()) is None
    assert service.registry.get_transfer_record(handoff.path) == before
    assert service.registry.get_agent_state(handoff.path, handoff.giver_read) is MESIState.INVALID
