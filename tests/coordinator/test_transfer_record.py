# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The transfer record: the registry half of the targeted grant handoff (#185).

A session done with a path hands it to a named successor. The registry keeps one
transfer record per artifact and exposes four members: the composite transfer
(decide every path, then apply the admitted ones, in one lock hold or one
transaction), the read (the record together with whether it is live), the
status write, and the eviction of records that are no longer live.

Every behavioural scenario runs against BOTH registries through the shared
``registry`` fixture's two arms; the state-log scenarios build their own pair,
because the shared fixture wires no state log. The schema step, durability and
the read-only handle are sqlite facts and sit in their own section.

Scenarios:

- a transfer stores the giver's session-level identity, the composite that held
  the claim and the version at transfer, and moves that composite INVALID;
- the giver's second transfer of a live record supersedes it, recording the
  superseded successor; a late re-send of the superseded tuple answers the
  superseded status and never supersedes back; another giver is refused naming
  the pending pair;
- an exact re-send of a live record answers its current status with no row
  move; one ended by decline or withdraw at an unmoved version answers that
  status; any other record naming the caller is treated as absent;
- the ordinary checks: not held (winning over a foreign write holder), an
  unconfirmed version, a foreign write holder, a live record by another giver,
  self, and an unknown successor;
- a refused grant is left exactly as it was, a mixed request commits its
  admitted paths, and a raise mid-apply rolls every path back;
- the INVALID move is logged under the handoff trigger and moves the epoch for
  a write shape only;
- liveness follows the version and the ended statuses, never the label alone,
  and is judged in one place per registry; eviction is its negation plus an age;
- the record goes with its artifact, survives a reopen, serves a read-only
  handle, and lands as schema v9 on fresh and migrated stores alike.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import TransferRecord, TransferRequest
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import (
    SCHEMA_USER_VERSION,
    CrossRuntimeSchemaError,
    ReadOnlyMutationError,
    SqliteArtifactRegistry,
)
from ccs.core.states import MESIState
from ccs.core.types import Artifact, TransferGrantOutcome

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifact(registry: Any, *, version: int = 3, name: str = "plan.md") -> UUID:
    art = Artifact(id=uuid4(), name=name, version=version, content_hash="h0")
    registry.register_artifact(art, "")
    return art.id


def _hold(registry: Any, artifact_id: UUID, agent_id: UUID, state: MESIState) -> None:
    """Give ``agent_id`` a claim in ``state``: a read for SHARED, an acquire for
    EXCLUSIVE, an acquire then a local write for MODIFIED."""
    if state is MESIState.SHARED:
        registry.set_agent_state(artifact_id, agent_id, state, trigger="fetch", tick=1)
        return
    registry.set_agent_state(
        artifact_id, agent_id, MESIState.EXCLUSIVE, trigger="write", tick=1
    )
    if state is MESIState.MODIFIED:
        registry.set_agent_state(
            artifact_id, agent_id, MESIState.MODIFIED, trigger="write", tick=1
        )


def _transfer(
    registry: Any,
    giver: UUID,
    successor: UUID,
    holders: dict[UUID, UUID],
    *,
    known: bool = True,
    now: float = 1000.0,
) -> list[TransferGrantOutcome]:
    request = TransferRequest(
        giver=giver, successor=successor, holders=holders, successor_known=known
    )
    return registry.transfer_grants(request, tick=2, now_unix=now)


def _refused(artifact_id: UUID, reason: str, **fields: Any) -> TransferGrantOutcome:
    return TransferGrantOutcome(
        artifact_id=artifact_id, transferred=False, reason=reason, **fields
    )


def _move_version(registry: Any, artifact_id: UUID, expected: int) -> None:
    """A bystander's compare-and-swap win: the version moves, no label is written."""
    result = registry.commit_cas(
        artifact_id, uuid4(), expected_version=expected, content_hash="h-moved"
    )
    assert isinstance(result, tuple), result


class _Log:
    """A state log that records entries and can be told to raise on one call."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []
        self.calls = 0
        self.raise_on_call: int | None = None

    def __call__(self, entry: dict[str, Any]) -> None:
        self.calls += 1
        if self.calls == self.raise_on_call:
            raise RuntimeError("injected state-log failure")
        self.entries.append(entry)


@pytest.fixture(params=["memory", "sqlite"])
def logged(request: pytest.FixtureRequest, tmp_path: Path):
    """Both registries wired to a state log (the shared fixture wires none)."""
    log = _Log()
    if request.param == "memory":
        yield ArtifactRegistry(state_log=log, instance_id="transfer-record"), log
    else:
        reg = SqliteArtifactRegistry(
            tmp_path / "logged.db", state_log=log, instance_id="transfer-record"
        )
        yield reg, log
        reg.close()


# ---------------------------------------------------------------------------
# A transfer stores the record and moves the presented composite INVALID
# ---------------------------------------------------------------------------


def test_a_transfer_records_the_session_level_giver_and_the_holding_composite(
    registry,
) -> None:
    """The record carries the giver's session-level identity, which the fence
    keys on, AND the composite that held the claim, which is the row that moved.
    Fails if the composite is stored as giver (a re-minted incarnation of the
    same session would then escape the fence) or if the answer omits the pair,
    version or hold shape a client reports."""
    art = _artifact(registry, version=3)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)

    (outcome,) = _transfer(registry, giver, successor, {art: holder}, now=1000.0)

    assert outcome == TransferGrantOutcome(
        artifact_id=art,
        transferred=True,
        giver=giver,
        successor=successor,
        version_at_transfer=3,
        hold_shape=MESIState.SHARED,
        status="pending",
    )
    assert registry.get_transfer_record(art) == (
        TransferRecord(
            artifact_id=art,
            giver=giver,
            holder=holder,
            successor=successor,
            version_at_transfer=3,
            hold_shape=MESIState.SHARED,
            cause="handoff",
            superseded_successor=None,
            status="pending",
            counterparty=None,
            created_at=1000.0,
            updated_at=1000.0,
        ),
        True,
    )
    assert registry.get_agent_state(art, holder) is MESIState.INVALID
    assert registry.get_artifact(art).version == 3


def test_get_answers_none_for_a_path_never_handed_off(registry) -> None:
    """No record is not a record: a path never handed off, and an unknown
    artifact, both read as None rather than raising or inventing one."""
    art = _artifact(registry)
    assert registry.get_transfer_record(art) is None
    assert registry.get_transfer_record(uuid4()) is None


@pytest.mark.parametrize(
    ("shape", "bump"),
    [(MESIState.EXCLUSIVE, 1), (MESIState.MODIFIED, 1), (MESIState.SHARED, 0)],
)
def test_a_write_shape_handoff_moves_the_epoch_and_a_read_shape_does_not(
    registry, shape: MESIState, bump: int
) -> None:
    """The handoff trigger joins the epoch-bump set, through the composite
    member: the INVALID move under the handoff trigger revokes a write claim
    from EXCLUSIVE or MODIFIED, so the giver's late commit at the unchanged
    version must meet the fence; from SHARED it revoked nothing, and a bump
    would fence every bystander that read the path."""
    art = _artifact(registry)
    giver, holder = uuid4(), uuid4()
    _hold(registry, art, holder, shape)
    before = registry.get_owner_generation(art)

    (outcome,) = _transfer(registry, giver, uuid4(), {art: holder})

    assert outcome.transferred is True
    assert outcome.hold_shape is shape
    assert registry.get_owner_generation(art) == before + bump
    assert registry.get_agent_state(art, holder) is MESIState.INVALID
    assert registry.granted_at_tick(holder, art) is None


def test_the_invalid_move_is_logged_under_the_handoff_trigger(logged) -> None:
    """One state-log entry per admitted path, under ``handoff`` -- never
    ``invalidate`` -- so the log tells a handoff from a release."""
    registry, log = logged
    art = _artifact(registry, version=3)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.EXCLUSIVE)
    before = len(log.entries)

    _transfer(registry, uuid4(), uuid4(), {art: holder})

    entries = log.entries[before:]
    assert [
        (e["agent_id"], e["from_state"], e["to_state"], e["trigger"], e["version"])
        for e in entries
    ] == [(str(holder), "EXCLUSIVE", "INVALID", "handoff", 3)]


# ---------------------------------------------------------------------------
# Supersession and re-sends
# ---------------------------------------------------------------------------


def test_the_givers_second_transfer_supersedes_its_live_record(registry) -> None:
    """A live record by the same giver is replaced, not duplicated: the
    replacing record keeps the stored holder, version and hold shape, and its
    cause records which successor it superseded. Fails if a second record is
    kept or the cause cannot name the superseded tuple."""
    art = _artifact(registry, version=3)
    giver, holder, first, second = uuid4(), uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, first, {art: holder}, now=1000.0)

    (outcome,) = _transfer(registry, giver, second, {art: uuid4()}, now=2000.0)

    assert outcome == TransferGrantOutcome(
        artifact_id=art,
        transferred=True,
        giver=giver,
        successor=second,
        version_at_transfer=3,
        hold_shape=MESIState.SHARED,
        status="pending",
    )
    assert registry.get_transfer_record(art) == (
        TransferRecord(
            artifact_id=art,
            giver=giver,
            holder=holder,
            successor=second,
            version_at_transfer=3,
            hold_shape=MESIState.SHARED,
            cause="supersession",
            superseded_successor=first,
            status="pending",
            counterparty=None,
            created_at=2000.0,
            updated_at=2000.0,
        ),
        True,
    )


def test_a_late_resend_of_the_superseded_tuple_never_supersedes_back(registry) -> None:
    """The giver's re-send of the tuple its later transfer superseded
    answers the superseded status and changes nothing. Fails if the re-send
    is read as a fresh supersession, which would hand the path back to the
    successor the giver already replaced."""
    art = _artifact(registry, version=3)
    giver, holder, first, second = uuid4(), uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, first, {art: holder}, now=1000.0)
    _transfer(registry, giver, second, {art: holder}, now=2000.0)
    current = registry.get_transfer_record(art)

    (outcome,) = _transfer(registry, giver, first, {art: holder}, now=3000.0)

    assert outcome == _refused(
        art,
        "handoff_ended",
        giver=giver,
        successor=first,
        version_at_transfer=3,
        hold_shape=MESIState.SHARED,
        status="superseded",
    )
    assert registry.get_transfer_record(art) == current


def test_another_givers_transfer_of_a_live_path_is_refused_naming_the_pair(
    registry,
) -> None:
    """Only the giver of a live handoff may transfer the path again.
    Another session's transfer is refused naming the pending pair, its own grant
    stays as it was, and the live record is unchanged."""
    art = _artifact(registry)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    other_giver, other_holder = uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _hold(registry, art, other_holder, MESIState.SHARED)
    _transfer(registry, giver, successor, {art: holder})
    current = registry.get_transfer_record(art)

    (outcome,) = _transfer(registry, other_giver, uuid4(), {art: other_holder})

    assert outcome == _refused(
        art, "handoff_in_flight", giver=giver, successor=successor
    )
    assert registry.get_transfer_record(art) == current
    assert registry.get_agent_state(art, other_holder) is MESIState.SHARED


@pytest.mark.parametrize("label", ["pending", "completed", "overtaken"])
def test_an_exact_resend_of_a_live_record_answers_its_status_and_moves_nothing(
    registry, label: str
) -> None:
    """A lost answer re-sent finds the live record and answers transferred
    with its current label. Nothing moves -- a fresh read the giver's session
    took since the transfer stays SHARED, the epoch stays, and the record is
    not rewritten -- so a re-send can never undo the giver's own re-read."""
    art = _artifact(registry, version=3)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.EXCLUSIVE)
    _transfer(registry, giver, successor, {art: holder}, now=1000.0)
    if label != "pending":
        registry.set_transfer_status(art, label, counterparty=None, now_unix=1500.0)
    _hold(registry, art, holder, MESIState.SHARED)
    current = registry.get_transfer_record(art)
    generation = registry.get_owner_generation(art)

    (outcome,) = _transfer(registry, giver, successor, {art: holder}, now=2000.0)

    assert outcome == TransferGrantOutcome(
        artifact_id=art,
        transferred=True,
        giver=giver,
        successor=successor,
        version_at_transfer=3,
        hold_shape=MESIState.EXCLUSIVE,
        status=label,
    )
    assert registry.get_transfer_record(art) == current
    assert registry.get_agent_state(art, holder) is MESIState.SHARED
    assert registry.get_owner_generation(art) == generation


def test_an_exact_resend_with_no_intervening_read_is_still_answered_transferred(
    registry,
) -> None:
    """The presented composite is already INVALID on a re-send that follows the
    transfer directly; the live record, not the hold check, answers it."""
    art = _artifact(registry, version=3)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, successor, {art: holder})

    (outcome,) = _transfer(registry, giver, successor, {art: holder})

    assert outcome.transferred is True
    assert outcome.version_at_transfer == 3


@pytest.mark.parametrize("ended", ["declined", "withdrawn"])
def test_an_exact_resend_after_decline_or_withdraw_answers_that_status(
    registry, ended: str
) -> None:
    """Once the record ended at an unmoved version, the re-sent tuple
    is refused carrying that status and nothing is written."""
    art = _artifact(registry, version=3)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, successor, {art: holder})
    registry.set_transfer_status(art, ended, counterparty=None, now_unix=1500.0)
    current = registry.get_transfer_record(art)

    (outcome,) = _transfer(registry, giver, successor, {art: holder})

    assert outcome == _refused(
        art,
        "handoff_ended",
        giver=giver,
        successor=successor,
        version_at_transfer=3,
        hold_shape=MESIState.SHARED,
        status=ended,
    )
    assert registry.get_transfer_record(art) == current


@pytest.mark.parametrize("ended", ["declined", "withdrawn"])
def test_an_ended_record_does_not_stand_in_for_the_hold_check(
    registry, ended: str
) -> None:
    """Arm (e): an ended record naming the caller is treated as absent, so a
    transfer to a different successor meets the ordinary hold check on the
    presented composite. Fails if an ended record is matched as though it were
    live -- the giver would hand on a path it no longer holds."""
    art = _artifact(registry)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, successor, {art: holder})
    registry.set_transfer_status(art, ended, counterparty=None, now_unix=1500.0)
    current = registry.get_transfer_record(art)

    (outcome,) = _transfer(registry, giver, uuid4(), {art: holder})

    assert outcome == _refused(art, "handoff_not_held")
    assert registry.get_transfer_record(art) == current


def test_a_giver_that_re_read_hands_the_path_on_again_after_a_withdraw(
    registry,
) -> None:
    """Arm (e) admitted: after a withdraw the ordinary checks decide, and a giver
    holding a fresh read is admitted. The ended record is replaced by a plain
    handoff, not a supersession, and the fresh read moves INVALID."""
    art = _artifact(registry, version=3)
    giver, holder, first, second = uuid4(), uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, first, {art: holder}, now=1000.0)
    registry.set_transfer_status(art, "withdrawn", counterparty=None, now_unix=1500.0)
    _hold(registry, art, holder, MESIState.SHARED)

    (outcome,) = _transfer(registry, giver, second, {art: holder}, now=2000.0)

    assert outcome.transferred is True
    record, live = registry.get_transfer_record(art)
    assert live is True
    assert (record.successor, record.cause, record.superseded_successor) == (
        second,
        "handoff",
        None,
    )
    assert (record.status, record.created_at) == ("pending", 2000.0)
    assert registry.get_agent_state(art, holder) is MESIState.INVALID


def test_a_resend_after_the_version_moved_answers_not_held(registry) -> None:
    """Once the version moved the record is not live, so the
    re-sent tuple is decided by the hold check and the giver holds nothing."""
    art = _artifact(registry, version=3)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, successor, {art: holder})
    _move_version(registry, art, expected=3)
    current = registry.get_transfer_record(art)

    (outcome,) = _transfer(registry, giver, successor, {art: holder})

    assert outcome == _refused(art, "handoff_not_held")
    assert registry.get_transfer_record(art) == current


# ---------------------------------------------------------------------------
# A refused supersession leaves the live record unchanged
# ---------------------------------------------------------------------------


def test_a_supersession_still_meets_the_self_unknown_and_write_holder_refusals(
    registry,
) -> None:
    """The stored holder stands in for the hold check only: the giver cannot
    hand a live path on to itself, to an id nobody knows, or while another
    session holds a write grant on it, and each refusal leaves the live record
    exactly as it was."""
    art = _artifact(registry)
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, giver, successor, {art: holder})
    current = registry.get_transfer_record(art)

    (to_self,) = _transfer(registry, giver, giver, {art: holder})
    (to_unknown,) = _transfer(registry, giver, uuid4(), {art: holder}, known=False)
    bystander = uuid4()
    _hold(registry, art, bystander, MESIState.EXCLUSIVE)
    (while_held,) = _transfer(registry, giver, uuid4(), {art: holder})

    assert to_self == _refused(art, "handoff_to_self")
    assert to_unknown == _refused(art, "handoff_successor_unknown")
    assert while_held == _refused(art, "handoff_other_holder")
    assert registry.get_transfer_record(art) == current


def test_a_supersession_moves_no_row_and_logs_nothing(logged) -> None:
    """There is no second INVALID move: the stored holder's row already moved,
    so a supersession writes the record alone -- no state-log entry, no epoch."""
    registry, log = logged
    art = _artifact(registry)
    giver, holder = uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.EXCLUSIVE)
    _transfer(registry, giver, uuid4(), {art: holder})
    generation = registry.get_owner_generation(art)
    before = len(log.entries)

    (outcome,) = _transfer(registry, giver, uuid4(), {art: holder})
    (resend,) = _transfer(registry, giver, outcome.successor, {art: holder})

    assert outcome.transferred is True
    assert resend.transferred is True
    assert log.entries[before:] == []
    assert registry.get_owner_generation(art) == generation


# ---------------------------------------------------------------------------
# The ordinary checks: the hold, the version, a foreign write holder, the successor
# ---------------------------------------------------------------------------


def test_a_path_the_presented_composite_does_not_hold_is_refused(registry) -> None:
    """A composite with no grant on the path has nothing to hand on, and
    nothing is written for it."""
    art = _artifact(registry)
    (outcome,) = _transfer(registry, uuid4(), uuid4(), {art: uuid4()})
    assert outcome == _refused(art, "handoff_not_held")
    assert registry.get_transfer_record(art) is None


def test_an_unknown_artifact_is_refused_as_not_held(registry) -> None:
    """An artifact the registry never saw is refused per grant, not raised:
    a raise would abort every other path of the request."""
    art = uuid4()
    (outcome,) = _transfer(registry, uuid4(), uuid4(), {art: uuid4()})
    assert outcome == _refused(art, "handoff_not_held")


def test_not_held_wins_over_a_foreign_write_holder(registry) -> None:
    """The not-held reason's precedence: when both apply, the caller is told it holds nothing,
    not that someone else holds the path."""
    art = _artifact(registry)
    _hold(registry, art, uuid4(), MESIState.EXCLUSIVE)
    (outcome,) = _transfer(registry, uuid4(), uuid4(), {art: uuid4()})
    assert outcome == _refused(art, "handoff_not_held")


def test_a_read_under_a_foreign_write_holder_is_refused(registry) -> None:
    """The caller's standing read is not the write authority to hand on
    while another session holds the path EXCLUSIVE; its read is left alone."""
    art = _artifact(registry)
    holder, bystander = uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _hold(registry, art, bystander, MESIState.EXCLUSIVE)

    (outcome,) = _transfer(registry, uuid4(), uuid4(), {art: holder})

    assert outcome == _refused(art, "handoff_other_holder")
    assert registry.get_agent_state(art, holder) is MESIState.SHARED
    assert registry.get_transfer_record(art) is None


def test_an_unconfirmed_version_is_refused_and_writes_no_record(registry) -> None:
    """With no confirmed version there is nothing to fence on."""
    art = _artifact(registry, version=0)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.EXCLUSIVE)

    (outcome,) = _transfer(registry, uuid4(), uuid4(), {art: holder})

    assert outcome == _refused(art, "handoff_version_unconfirmed")
    assert registry.get_transfer_record(art) is None
    assert registry.get_agent_state(art, holder) is MESIState.EXCLUSIVE


def test_a_transfer_to_the_callers_own_session_is_refused(registry) -> None:
    """A handoff is between sessions, so naming the caller's own
    session-level identity is refused and the read is left as it was."""
    art = _artifact(registry)
    giver, holder = uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)

    (outcome,) = _transfer(registry, giver, giver, {art: holder})

    assert outcome == _refused(art, "handoff_to_self")
    assert registry.get_agent_state(art, holder) is MESIState.SHARED


def test_an_unresolved_successor_with_no_binding_or_grant_row_is_unknown(
    registry,
) -> None:
    """An id the caller could not resolve, with no bound principal and
    no grant row, is unknown -- never fenced on, never handed to."""
    art = _artifact(registry)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)

    (outcome,) = _transfer(registry, uuid4(), uuid4(), {art: holder}, known=False)

    assert outcome == _refused(art, "handoff_successor_unknown")
    assert registry.get_agent_state(art, holder) is MESIState.SHARED


def test_a_principal_binding_or_a_grant_row_makes_a_successor_known(registry) -> None:
    """The member's own two ways to know a successor: a bound principal (a session-level
    identity) and, for library callers with no name map, a grant row."""
    bound, with_row = uuid4(), uuid4()
    registry.bind_caller_principal(bound, "principal-for-the-test", "nonce-for-the-test")
    _hold(registry, _artifact(registry, name="other.md"), with_row, MESIState.SHARED)
    first, second = _artifact(registry, name="a.md"), _artifact(registry, name="b.md")
    holder = uuid4()
    _hold(registry, first, holder, MESIState.SHARED)
    _hold(registry, second, holder, MESIState.SHARED)

    (to_bound,) = _transfer(registry, uuid4(), bound, {first: holder}, known=False)
    (to_row,) = _transfer(registry, uuid4(), with_row, {second: holder}, known=False)

    assert to_bound.transferred is True
    assert to_row.transferred is True


# ---------------------------------------------------------------------------
# Multi-path: per-grant answers, all-or-nothing apply
# ---------------------------------------------------------------------------


def test_a_mixed_request_commits_the_admitted_path_and_leaves_the_refused_one(
    registry,
) -> None:
    """Each grant is answered on its own; the admitted path's record and
    INVALID move land while the refused path's grant is left exactly as it was."""
    held, contested = _artifact(registry, name="a.md"), _artifact(registry, name="b.md")
    giver, holder, successor = uuid4(), uuid4(), uuid4()
    _hold(registry, held, holder, MESIState.EXCLUSIVE)
    _hold(registry, contested, holder, MESIState.SHARED)
    _hold(registry, contested, uuid4(), MESIState.EXCLUSIVE)

    outcomes = _transfer(registry, giver, successor, {held: holder, contested: holder})

    assert [(o.artifact_id, o.transferred, o.reason) for o in outcomes] == [
        (held, True, None),
        (contested, False, "handoff_other_holder"),
    ]
    assert registry.get_transfer_record(held)[0].successor == successor
    assert registry.get_agent_state(held, holder) is MESIState.INVALID
    assert registry.get_transfer_record(contested) is None
    assert registry.get_agent_state(contested, holder) is MESIState.SHARED


def test_a_raise_mid_apply_rolls_every_path_back(logged) -> None:
    """All-or-nothing: the second path's state-log emit raises, and neither
    path keeps a record, an INVALID move or an epoch bump. The sequence number
    the failed emits reserved is released, so the next entry leaves no gap."""
    registry, log = logged
    first, second = _artifact(registry, name="a.md"), _artifact(registry, name="b.md")
    holder = uuid4()
    _hold(registry, first, holder, MESIState.EXCLUSIVE)
    _hold(registry, second, holder, MESIState.EXCLUSIVE)
    generations = (registry.get_owner_generation(first), registry.get_owner_generation(second))
    sequence = log.entries[-1]["sequence_number"]
    log.raise_on_call = log.calls + 2

    with pytest.raises(RuntimeError, match="injected"):
        _transfer(registry, uuid4(), uuid4(), {first: holder, second: holder})

    for art in (first, second):
        assert registry.get_transfer_record(art) is None
        assert registry.get_agent_state(art, holder) is MESIState.EXCLUSIVE
    assert (
        registry.get_owner_generation(first),
        registry.get_owner_generation(second),
    ) == generations
    registry.set_agent_state(first, uuid4(), MESIState.SHARED, trigger="fetch", tick=9)
    assert log.entries[-1]["sequence_number"] == sequence + 1


# ---------------------------------------------------------------------------
# Liveness and the status write
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "live"),
    [
        ("pending", True),
        ("completed", True),
        ("overtaken", True),
        ("declined", False),
        ("withdrawn", False),
    ],
)
def test_liveness_at_the_transfer_version_follows_the_ended_statuses(
    registry, label: str, live: bool
) -> None:
    """At an unmoved version only a decline or a withdraw ends a record; a
    completed or overtaken one still fences the giver."""
    art = _artifact(registry)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder})
    registry.set_transfer_status(art, label, counterparty=None, now_unix=1500.0)

    record, is_live = registry.get_transfer_record(art)

    assert record.status == label
    assert is_live is live


@pytest.mark.parametrize("label", ["pending", "completed", "overtaken"])
def test_a_record_whose_version_moved_is_not_live_whatever_its_label(
    registry, label: str
) -> None:
    """A version can move with no status write (a crash or a failed label
    write after a win), and such a record must read as not live. Fails if
    liveness is read off the stored status alone."""
    art = _artifact(registry, version=3)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder})
    registry.set_transfer_status(art, label, counterparty=None, now_unix=1500.0)

    _move_version(registry, art, expected=3)

    record, live = registry.get_transfer_record(art)
    assert record.status == label
    assert live is False


def test_the_status_write_sets_status_counterparty_and_updated_at(registry) -> None:
    """The write is unconditional (the service decides liveness): it lands on a
    record that is no longer live, and each write replaces the counterparty."""
    art = _artifact(registry, version=3)
    holder, bystander = uuid4(), uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder}, now=1000.0)

    registry.set_transfer_status(art, "overtaken", counterparty=bystander, now_unix=1100.0)
    overtaken = registry.get_transfer_record(art)[0]
    _move_version(registry, art, expected=3)
    registry.set_transfer_status(art, "completed", counterparty=None, now_unix=1200.0)
    completed = registry.get_transfer_record(art)[0]

    assert (overtaken.status, overtaken.counterparty, overtaken.updated_at) == (
        "overtaken",
        bystander,
        1100.0,
    )
    assert (completed.status, completed.counterparty, completed.updated_at) == (
        "completed",
        None,
        1200.0,
    )
    assert completed.created_at == 1000.0


@pytest.mark.parametrize("status", ["superseded", "lapsed"])
def test_the_status_write_refuses_a_status_no_row_may_carry(
    registry, status: str
) -> None:
    """``superseded`` never persists (the replacing record's cause carries it),
    and a value outside the vocabulary is not a status; neither is written."""
    art = _artifact(registry)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder})
    current = registry.get_transfer_record(art)

    with pytest.raises(ValueError):
        registry.set_transfer_status(art, status, counterparty=None, now_unix=1500.0)

    assert registry.get_transfer_record(art) == current


def test_a_transfer_deciding_a_cause_outside_the_vocabulary_stores_nothing(
    registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cause is a closed vocabulary at write time: a decision carrying any
    other cause (here a future edit to the handoff constant) raises before the
    registry applies anything, so no record is stored and the holder keeps its
    claim, on both backends."""
    import ccs.coordinator.registry_protocol as protocol

    art = _artifact(registry)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    monkeypatch.setattr(protocol, "TRANSFER_CAUSE_HANDOFF", "reclaim")

    with pytest.raises(ValueError, match="transfer cause 'reclaim' cannot be stored"):
        _transfer(registry, uuid4(), uuid4(), {art: holder})

    assert registry.get_transfer_record(art) is None
    assert registry.get_agent_state(art, holder) == MESIState.SHARED


def test_a_stored_cause_this_build_does_not_know_is_read_back(tmp_path: Path) -> None:
    """A cause is add-only, so a row a later build wrote with a cause this one
    does not know reads back as it is: the record, its liveness, the /status
    projection and eviction all still work. Fails if the vocabulary is checked
    on read-back, where one such row broke every /status and every sweep's
    eviction pass."""
    db_path = tmp_path / "later-cause.db"
    with SqliteArtifactRegistry(db_path) as registry:
        art = _artifact(registry)
        holder = uuid4()
        _hold(registry, art, holder, MESIState.SHARED)
        _transfer(registry, uuid4(), uuid4(), {art: holder})
    conn = sqlite3.connect(str(db_path))
    with conn:
        conn.execute("UPDATE transfer_records SET cause = 'reclaim'")
    conn.close()

    with SqliteArtifactRegistry(db_path) as registry:
        record, live = registry.get_transfer_record(art)
        assert (record.cause, live) == ("reclaim", True)
        registry.status_snapshot(include_transfers=True)
        assert registry.evict_transfer_records(max_age_sec=0.0, now_unix=10.0**12) == 0
        registry.set_transfer_status(art, "declined", counterparty=None, now_unix=1.0)
        assert registry.evict_transfer_records(max_age_sec=0.0, now_unix=10.0**12) == 1


def test_the_status_write_on_a_path_with_no_record_raises(registry) -> None:
    """A label write with no record is a caller bug; both registries raise
    rather than one inventing a row and the other doing nothing."""
    art = _artifact(registry)
    with pytest.raises(KeyError):
        registry.set_transfer_status(art, "completed", counterparty=None, now_unix=1.0)


# ---------------------------------------------------------------------------
# Eviction is the negation of liveness, plus an age
# ---------------------------------------------------------------------------


def test_eviction_removes_an_ended_record_once_older_than_the_age(registry) -> None:
    """The record's own update is the later stamp here (written ahead of the
    artifact's registration clock), so the age runs from it on both backends."""
    art = _artifact(registry)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    ended_at = time.time() + 10_000.0
    _transfer(registry, uuid4(), uuid4(), {art: holder}, now=ended_at)
    registry.set_transfer_status(art, "declined", counterparty=None, now_unix=ended_at)

    young = registry.evict_transfer_records(max_age_sec=100.0, now_unix=ended_at + 50.0)
    assert young == 0
    assert registry.get_transfer_record(art) is not None

    old = registry.evict_transfer_records(max_age_sec=100.0, now_unix=ended_at + 101.0)
    assert old == 1
    assert registry.get_transfer_record(art) is None


@pytest.mark.parametrize("label", ["pending", "completed", "overtaken"])
def test_eviction_never_removes_a_live_record_whatever_its_age(
    registry, label: str
) -> None:
    """No timer ever touches a live record -- removing one would lift
    the giver's fence without the successor, the giver or a write ending it."""
    art = _artifact(registry)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder}, now=1000.0)
    registry.set_transfer_status(art, label, counterparty=None, now_unix=1000.0)

    evicted = registry.evict_transfer_records(max_age_sec=0.0, now_unix=10.0**12)

    assert evicted == 0
    assert registry.get_transfer_record(art)[1] is True


def test_eviction_removes_a_pending_record_whose_version_moved(registry) -> None:
    """The ending nobody labelled: the version moved and the record still says
    pending. It is not live, so eviction takes it once it is old enough."""
    art = _artifact(registry, version=3)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder}, now=1000.0)
    _move_version(registry, art, expected=3)

    evicted = registry.evict_transfer_records(max_age_sec=100.0, now_unix=10.0**12)

    assert evicted == 1
    assert registry.get_transfer_record(art) is None


def test_sqlite_eviction_ages_from_the_artifacts_last_update_when_later(
    tmp_path: Path,
) -> None:
    """A version-move ending is not evicted before the giver's next touch
    can report it, so on sqlite the age runs from the later of the record's
    update and the artifact's. The move below stamps the artifact at the real
    clock while the record was written at t=1000."""
    with SqliteArtifactRegistry(tmp_path / "age.db") as registry:
        art = _artifact(registry, version=3)
        holder = uuid4()
        _hold(registry, art, holder, MESIState.SHARED)
        _transfer(registry, uuid4(), uuid4(), {art: holder}, now=1000.0)
        _move_version(registry, art, expected=3)
        moved_at = registry.get_artifact_updated_at(art)
        assert moved_at is not None and moved_at > 1000.0 + 100.0

        assert registry.evict_transfer_records(max_age_sec=100.0, now_unix=1101.0) == 0
        assert registry.get_transfer_record(art) is not None
        assert (
            registry.evict_transfer_records(max_age_sec=100.0, now_unix=moved_at + 101.0)
            == 1
        )
        assert registry.get_transfer_record(art) is None


def test_memory_eviction_ages_from_the_records_own_update_alone() -> None:
    """The in-memory side of the test above, stated so the divergence is a
    decision rather than drift: the slot keeps no wall-clock stamp of the
    artifact's last update, so the same version-move ending is evicted as
    soon as the RECORD is older than the age."""
    registry = ArtifactRegistry()
    art = _artifact(registry, version=3)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder}, now=1000.0)
    _move_version(registry, art, expected=3)

    assert registry.evict_transfer_records(max_age_sec=100.0, now_unix=1101.0) == 1
    assert registry.get_transfer_record(art) is None


# ---------------------------------------------------------------------------
# The record goes with its artifact
# ---------------------------------------------------------------------------


def test_the_library_delete_verb_drops_the_record(registry) -> None:
    """The record goes with its artifact on both backends (the sqlite
    cascade, the in-memory slot), so a re-registered name never inherits a
    handoff of the path it replaced."""
    art = _artifact(registry)
    holder = uuid4()
    _hold(registry, art, holder, MESIState.SHARED)
    _transfer(registry, uuid4(), uuid4(), {art: holder})

    CoordinatorService(registry).delete(agent_id=uuid4(), artifact_id=art)

    assert registry.get_transfer_record(art) is None


# ---------------------------------------------------------------------------
# sqlite: durability, the read-only handle, schema v9
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
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def _transfer_table_shape(db_path: Path) -> tuple[list[tuple], list[tuple]]:
    conn = sqlite3.connect(str(db_path))
    try:
        return (
            conn.execute("PRAGMA table_info(transfer_records)").fetchall(),
            conn.execute("PRAGMA foreign_key_list(transfer_records)").fetchall(),
        )
    finally:
        conn.close()


def _transfer_row_count(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM transfer_records").fetchone()[0]
    finally:
        conn.close()


def _revert_to_v8_shape(db_path: Path) -> None:
    """A current db rewound to what a v8 build produced: no transfer table."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS transfer_records")
        conn.execute("PRAGMA user_version = 8")
        conn.commit()
    finally:
        conn.close()


def _revert_to_v7_shape(db_path: Path) -> None:
    """Rewound one step further: the v8 caller-principal table absent too."""
    _revert_to_v8_shape(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS caller_principals")
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    finally:
        conn.close()


def test_a_record_survives_close_and_reopen(tmp_path: Path) -> None:
    """The record is durable on sqlite: a coordinator restart must not lift
    a fence the record still owes the giver."""
    db = tmp_path / "durable.db"
    with SqliteArtifactRegistry(db) as registry:
        art = _artifact(registry)
        holder = uuid4()
        _hold(registry, art, holder, MESIState.EXCLUSIVE)
        _transfer(registry, uuid4(), uuid4(), {art: holder})
        stored = registry.get_transfer_record(art)

    with SqliteArtifactRegistry(db) as reopened:
        assert reopened.get_transfer_record(art) == stored


def test_a_read_only_handle_serves_the_record_and_refuses_the_mutators(
    tmp_path: Path,
) -> None:
    """The read is a lock-only select, so it serves a read-only open; the
    composite transfer, the status write and the eviction are mutators and
    raise before touching the store."""
    db = tmp_path / "ro.db"
    with SqliteArtifactRegistry(db) as registry:
        art = _artifact(registry)
        holder = uuid4()
        _hold(registry, art, holder, MESIState.SHARED)
        _transfer(registry, uuid4(), uuid4(), {art: holder})
        stored = registry.get_transfer_record(art)

    with SqliteArtifactRegistry(db, read_only=True) as read_only:
        assert read_only.get_transfer_record(art) == stored
        with pytest.raises(ReadOnlyMutationError):
            _transfer(read_only, uuid4(), uuid4(), {art: holder})
        with pytest.raises(ReadOnlyMutationError):
            read_only.set_transfer_status(art, "declined", counterparty=None, now_unix=1.0)
        with pytest.raises(ReadOnlyMutationError):
            read_only.evict_transfer_records(max_age_sec=0.0, now_unix=10.0**12)
        assert read_only.get_transfer_record(art) == stored


def test_fresh_db_is_created_at_v9_with_the_transfer_table(tmp_path: Path) -> None:
    """A fresh store lands the table at the head version directly; no
    migration step ever runs against it."""
    db = tmp_path / "fresh.db"
    with SqliteArtifactRegistry(db):
        pass
    assert SCHEMA_USER_VERSION == 9
    assert _user_version(db) == 9
    assert "transfer_records" in _tables(db)


def test_a_migrated_transfer_table_equals_a_fresh_one(tmp_path: Path) -> None:
    """Fresh and migrated stores must be identical: the same columns, types,
    defaults, key and cascade. Fails if either path drifts from the other."""
    fresh, migrated = tmp_path / "fresh.db", tmp_path / "migrated.db"
    with SqliteArtifactRegistry(fresh):
        pass
    with SqliteArtifactRegistry(migrated):
        pass
    _revert_to_v8_shape(migrated)
    assert _user_version(migrated) == 8
    assert "transfer_records" not in _tables(migrated)

    with SqliteArtifactRegistry(migrated):
        pass

    assert _user_version(migrated) == 9
    assert _transfer_table_shape(migrated) == _transfer_table_shape(fresh)
    columns, foreign_keys = _transfer_table_shape(fresh)
    assert columns and foreign_keys  # non-empty: the probe sees the table


def test_a_migrated_table_enforces_the_cascade(tmp_path: Path) -> None:
    """The migrated table carries the cascade too, and the writer's
    foreign keys enforce it: deleting the artifact leaves no orphan row."""
    db = tmp_path / "cascade.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v8_shape(db)
    with SqliteArtifactRegistry(db) as registry:
        art = _artifact(registry)
        holder = uuid4()
        _hold(registry, art, holder, MESIState.SHARED)
        _transfer(registry, uuid4(), uuid4(), {art: holder})
        registry.remove_artifact(art)
    assert _transfer_row_count(db) == 0


def test_v7_origin_walk_lands_the_transfer_table_at_the_v9_stamp(tmp_path: Path) -> None:
    """THE RE-STAMP TRAP, fifth arming. ``_migrate_v7_to_v8`` was the final step
    and stamped the constant; with the constant at 9 it must stamp its own
    literal 8, or a v7-origin db is stamped 9 WITHOUT the transfer table and
    the chained v8->v9 loser-guard no-ops."""
    db = tmp_path / "v7.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v7_shape(db)
    assert _user_version(db) == 7

    with SqliteArtifactRegistry(db):
        pass

    assert _user_version(db) == 9
    assert {"caller_principals", "transfer_records"} <= _tables(db)


def test_a_crash_before_the_v9_stamp_leaves_a_bootable_v8(tmp_path: Path) -> None:
    """The DDL and the stamp share one transaction: a failure between them
    rolls the table back with the stamp, and a clean reopen re-migrates."""
    db = tmp_path / "crash.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v8_shape(db)

    class _Crash(RuntimeError):
        pass

    class _CrashingConn:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql: str, *args: Any) -> Any:
            if sql.strip() == "PRAGMA user_version = 9":
                raise _Crash("simulated kill before the stamp")
            return self._inner.execute(sql, *args)

        def __getattr__(self, name: str) -> Any:
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
    assert "transfer_records" not in _tables(db)
    with SqliteArtifactRegistry(db):
        pass
    assert _user_version(db) == 9
    assert "transfer_records" in _tables(db)


def test_a_v9_stamp_without_the_transfer_table_is_refused(tmp_path: Path) -> None:
    """No ledger this build recognizes stamps 9 without the table, so the open
    fails closed -- with a control that a genuine v9 opens."""
    genuine, forged = tmp_path / "genuine.db", tmp_path / "forged.db"
    for db in (genuine, forged):
        with SqliteArtifactRegistry(db):
            pass
    conn = sqlite3.connect(str(forged))
    try:
        conn.execute("DROP TABLE transfer_records")
        conn.commit()
    finally:
        conn.close()
    assert _user_version(forged) == 9

    with SqliteArtifactRegistry(genuine):
        pass
    with pytest.raises(CrossRuntimeSchemaError, match="transfer_records"):
        SqliteArtifactRegistry(forged)
