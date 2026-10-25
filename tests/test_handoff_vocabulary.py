"""The targeted-handoff vocabulary (Cohexa-ai/agent-coherence#185).

Every name pinned here is a WIRE value: a client, the MCP mapper and the
protocol corpus branch on ``reason == CONSTANT`` / ``status == CONSTANT``, never
on a substring of a message. So each constant is pinned to a hand-written
literal (add, never rename), the closed sets are pinned to literal sets with
their cardinality, and no value may collide with another reason or status a
client could receive in the same body -- a collision routes one answer into
another's recovery.
"""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from uuid import uuid4

import pytest

from ccs.core.exceptions import (
    CALLER_PRINCIPAL_REASONS,
    CAS_EXHAUSTED_REASON,
    COMMIT_PREEMPTED_REASON,
    COMMIT_UNCONFIRMED_REASON,
    COORDINATOR_FAILED_REASON,
    COORDINATOR_UNAVAILABLE_REASON,
    GIVER_FENCED_REASON,
    HANDOFF_ACCEPT_UNCONFIRMED_REASON,
    HANDOFF_DECLINE_UNCONFIRMED_REASON,
    HANDOFF_ENDED_REASON,
    HANDOFF_IN_FLIGHT_REASON,
    HANDOFF_NOT_GIVER_REASON,
    HANDOFF_NOT_HELD_REASON,
    HANDOFF_NOT_LIVE_REASON,
    HANDOFF_NOT_SUCCESSOR_REASON,
    HANDOFF_OTHER_HOLDER_REASON,
    HANDOFF_REASONS,
    HANDOFF_SELF_REASON,
    HANDOFF_SUCCESSOR_MALFORMED_REASON,
    HANDOFF_SUCCESSOR_UNKNOWN_REASON,
    HANDOFF_TRANSFER_REFUSAL_REASONS,
    HANDOFF_TRANSFER_UNCONFIRMED_REASON,
    HANDOFF_UNCONFIRMED_REASONS,
    HANDOFF_VERSION_UNCONFIRMED_REASON,
    HANDOFF_WITHDRAW_UNCONFIRMED_REASON,
    HOLD_REASONS,
    INTERNAL_CONCURRENCY_REASON,
    OCC_CALLER_TRANSIENT_REASON,
    OTHER_HOLDER_REASON,
    SPLIT_PAIR_REASON,
    STALE_READ_GENERATION_REASON,
    STALE_VIEW_REASON,
    UNAVAILABLE_REASON,
    VERSION_MISMATCH_REASON,
    VIEW_WEDGED_REASON,
    CoherenceError,
    CoordinatorFailed,
    CoordinatorUnreachable,
    GiverFenced,
    OtherHolder,
    StaleView,
)
from ccs.core.states import MESIState
from ccs.core.types import (
    TRANSFER_STATUS_COMPLETED,
    TRANSFER_STATUS_DECLINED,
    TRANSFER_STATUS_OVERTAKEN,
    TRANSFER_STATUS_PENDING,
    TRANSFER_STATUS_SUPERSEDED,
    TRANSFER_STATUS_WITHDRAWN,
    TRANSFER_STATUSES,
    TransferGrantOutcome,
)

# (constant, the literal it must carry). A list of pairs, not a dict keyed on the
# constant: a dict would silently merge two constants sharing a value.
_REASON_LITERALS = [
    (GIVER_FENCED_REASON, "handed_off"),
    (HANDOFF_SELF_REASON, "handoff_to_self"),
    (HANDOFF_SUCCESSOR_UNKNOWN_REASON, "handoff_successor_unknown"),
    (HANDOFF_SUCCESSOR_MALFORMED_REASON, "handoff_successor_malformed"),
    (HANDOFF_NOT_HELD_REASON, "handoff_not_held"),
    (HANDOFF_VERSION_UNCONFIRMED_REASON, "handoff_version_unconfirmed"),
    (HANDOFF_IN_FLIGHT_REASON, "handoff_in_flight"),
    (HANDOFF_OTHER_HOLDER_REASON, "handoff_other_holder"),
    (HANDOFF_ENDED_REASON, "handoff_ended"),
    (HANDOFF_NOT_SUCCESSOR_REASON, "handoff_not_successor"),
    (HANDOFF_NOT_GIVER_REASON, "handoff_not_giver"),
    (HANDOFF_NOT_LIVE_REASON, "handoff_not_live"),
    (HANDOFF_TRANSFER_UNCONFIRMED_REASON, "handoff_transfer_unconfirmed"),
    (HANDOFF_ACCEPT_UNCONFIRMED_REASON, "handoff_accept_unconfirmed"),
    (HANDOFF_DECLINE_UNCONFIRMED_REASON, "handoff_decline_unconfirmed"),
    (HANDOFF_WITHDRAW_UNCONFIRMED_REASON, "handoff_withdraw_unconfirmed"),
]


def test_every_handoff_reason_is_pinned_and_distinct() -> None:
    """Each reason is its own wire value. Two
    constants sharing one would make, say, an unknown successor read as a
    malformed one, or an unconfirmed transfer read as a definite refusal that
    changed nothing."""
    for constant, literal in _REASON_LITERALS:
        assert constant == literal
    values = [constant for constant, _ in _REASON_LITERALS]
    assert len(set(values)) == len(values) == 16
    assert HANDOFF_REASONS == {literal for _, literal in _REASON_LITERALS}
    assert len(HANDOFF_REASONS) == 16


def test_transfer_refusals_and_unconfirmed_answers_are_separate_closed_sets() -> None:
    """A per-grant transfer refusal changed nothing for that grant; an
    unconfirmed answer means the outcome is unknown. A client that
    reads one as the other either re-sends over a transfer that landed or
    treats a refused grant as handed off, so the two sets never overlap."""
    assert HANDOFF_TRANSFER_REFUSAL_REASONS == {
        "handoff_to_self",
        "handoff_successor_unknown",
        "handoff_successor_malformed",
        "handoff_not_held",
        "handoff_version_unconfirmed",
        "handoff_in_flight",
        "handoff_other_holder",
        "handoff_ended",
    }
    assert len(HANDOFF_TRANSFER_REFUSAL_REASONS) == 8
    assert HANDOFF_UNCONFIRMED_REASONS == {
        "handoff_transfer_unconfirmed",
        "handoff_accept_unconfirmed",
        "handoff_decline_unconfirmed",
        "handoff_withdraw_unconfirmed",
    }
    assert len(HANDOFF_UNCONFIRMED_REASONS) == 4
    assert HANDOFF_TRANSFER_REFUSAL_REASONS.isdisjoint(HANDOFF_UNCONFIRMED_REASONS)
    assert HANDOFF_TRANSFER_REFUSAL_REASONS < HANDOFF_REASONS
    assert HANDOFF_UNCONFIRMED_REASONS < HANDOFF_REASONS


def test_handoff_reasons_collide_with_no_existing_reason_or_status() -> None:
    """The giver reason rides the commit routes beside the existing refusals,
    and the handoff bodies carry a ``status`` beside the ``reason``. A shared
    value would send the giver into another refusal's recovery -- a reacquire
    or a re-merge that the fence refuses again -- and the giver reason is
    deliberately not a hold: a hold invites the retry this fence forbids."""
    existing = (
        HOLD_REASONS
        | CALLER_PRINCIPAL_REASONS
        | {
            VERSION_MISMATCH_REASON,
            OTHER_HOLDER_REASON,
            STALE_READ_GENERATION_REASON,
            OCC_CALLER_TRANSIENT_REASON,
            STALE_VIEW_REASON,
            COMMIT_PREEMPTED_REASON,
            VIEW_WEDGED_REASON,
            COMMIT_UNCONFIRMED_REASON,
            CAS_EXHAUSTED_REASON,
            INTERNAL_CONCURRENCY_REASON,
            COORDINATOR_UNAVAILABLE_REASON,
            COORDINATOR_FAILED_REASON,
            SPLIT_PAIR_REASON,
            UNAVAILABLE_REASON,
        }
    )
    assert HANDOFF_REASONS.isdisjoint(existing)
    assert HANDOFF_REASONS.isdisjoint(TRANSFER_STATUSES)


def test_giver_fenced_carries_its_reason_successor_and_version() -> None:
    """The giver's refusal is one typed terminal whose
    class-level reason equals the wire constant, so a consumer classifies it by
    type or ``.reason`` and never by the message, and it carries the successor
    and the version at transfer the giver reports to its user or host."""
    successor = uuid4()
    with pytest.raises(GiverFenced) as caught:
        raise GiverFenced("notes/plan.md", successor, 7)
    err = caught.value
    assert GiverFenced.reason == "handed_off"
    assert err.reason == GIVER_FENCED_REASON
    assert err.successor == successor
    assert err.version_at_transfer == 7
    assert err.artifact_id == "notes/plan.md"
    assert isinstance(err, CoherenceError)
    assert str(successor) in str(err)
    assert "version_at_transfer=7" in str(err)


def test_coordinator_unreachable_carries_its_reason_and_crosses_a_process_boundary() -> None:
    """The refusal for no usable answer from the coordinator is a typed
    terminal in the coherence family, so a strict caller catches it with the
    rest and a consumer classifies it by type or ``.reason``, never by the
    message. Its class-level reason is the existing wire value the MCP
    unattached refusal already answers, not a new one. It carries the message
    only, so a child process can raise it and the parent unpickle it whole."""
    err = CoordinatorUnreachable("coordinator gone")
    assert CoordinatorUnreachable.reason == "coordinator_unavailable"
    assert err.reason == COORDINATOR_UNAVAILABLE_REASON
    assert isinstance(err, CoherenceError)

    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is CoordinatorUnreachable
    assert restored.args == ("coordinator gone",)
    assert restored.reason == "coordinator_unavailable"


def test_coordinator_failed_carries_its_reason_and_crosses_a_process_boundary() -> None:
    """The read error for the coordinator's failure envelope (#259) is a typed
    terminal in the coherence family, never a stale view (a caller reads a
    ``StaleView`` from a versioned read as the split pair), with its own wire
    value. It carries the message only, so a child process can raise it and the
    parent unpickle it whole."""
    err = CoordinatorFailed("coordinator answered the read with a failure")
    assert CoordinatorFailed.reason == "coordinator_failed"
    assert err.reason == COORDINATOR_FAILED_REASON
    assert isinstance(err, CoherenceError)
    assert not isinstance(err, StaleView)

    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is CoordinatorFailed
    assert restored.args == ("coordinator answered the read with a failure",)
    assert SPLIT_PAIR_REASON == "split_pair"


def test_unavailable_reason_is_its_own_wire_value() -> None:
    """An opted-in pre-edit the coordinator could not decide in time is refused
    with this reason, and a client keys its refusal on the value. It stays apart
    from ``coordinator_unavailable``, which means the coordinator gave no usable
    answer at all: this answer comes from a coordinator that is alive and busy."""
    assert UNAVAILABLE_REASON == "unavailable"
    assert UNAVAILABLE_REASON != COORDINATOR_UNAVAILABLE_REASON


def test_other_holder_reason_is_its_own_wire_value() -> None:
    """An opted-in pre-edit refused because another agent holds the file
    answers this reason, the value ``post-edit-cas`` already returns when a
    commit meets a holder. It is not the handoff route's
    ``handoff_other_holder``, which refuses a transfer, and not an effect-gate
    hold cause, which an effect check answers and a client matches by
    membership in ``HOLD_REASONS``."""
    assert OTHER_HOLDER_REASON == "other_holder"
    assert OTHER_HOLDER_REASON != HANDOFF_OTHER_HOLDER_REASON
    assert OTHER_HOLDER_REASON not in HOLD_REASONS


def test_other_holder_carries_its_reason_artifact_and_holder() -> None:
    """The refusal is one typed error whose class-level reason equals the wire
    constant, so a consumer classifies it by type or ``.reason`` and never by
    the message, and it carries the holder the caller backs off from."""
    holder = uuid4()
    with pytest.raises(OtherHolder) as caught:
        raise OtherHolder("notes/plan.md", holder)
    err = caught.value
    assert OtherHolder.reason == "other_holder"
    assert err.reason == OTHER_HOLDER_REASON
    assert err.artifact_id == "notes/plan.md"
    assert err.holder_agent_id == holder
    assert isinstance(err, CoherenceError)
    assert str(holder) in str(err)


@pytest.mark.parametrize(
    ("artifact_id", "holder"),
    [
        (uuid4(), uuid4()),
        ("notes/plan.md", "0123456789abcdef0123456789abcdef"),
        (None, None),
    ],
    ids=["uuid-holder", "wire-string-holder", "no-holder"],
)
def test_other_holder_crosses_a_process_boundary(artifact_id: object, holder: object) -> None:
    """The service raises it in process, and a client raises the same class
    from the wire answer, so a worker process that raises it must hand the
    parent the typed refusal, not a ``TypeError`` from unpickling."""
    err = OtherHolder(artifact_id, holder)
    restored = pickle.loads(pickle.dumps(err))
    assert type(restored) is OtherHolder
    assert restored.reason == OTHER_HOLDER_REASON
    assert restored.artifact_id == artifact_id
    assert restored.holder_agent_id == holder
    assert str(restored) == str(err)


def test_transfer_statuses_are_the_six_closed_values() -> None:
    """A record's status is exactly one of six values, a closed set the
    registries validate before a write and the corpus covers row by row."""
    for constant, literal in [
        (TRANSFER_STATUS_PENDING, "pending"),
        (TRANSFER_STATUS_COMPLETED, "completed"),
        (TRANSFER_STATUS_DECLINED, "declined"),
        (TRANSFER_STATUS_WITHDRAWN, "withdrawn"),
        (TRANSFER_STATUS_OVERTAKEN, "overtaken"),
        (TRANSFER_STATUS_SUPERSEDED, "superseded"),
    ]:
        assert constant == literal
    assert TRANSFER_STATUSES == {
        "pending", "completed", "declined", "withdrawn", "overtaken", "superseded",
    }
    assert len(TRANSFER_STATUSES) == 6


def test_transfer_grant_outcome_is_frozen_and_names_giver_and_successor_by_keyword() -> None:
    """The per-grant transfer outcome is a value, never mutated after the
    answer is built. Its giver and successor are two ids of one type side by
    side; a positional swap would report the giver as the successor, so it is
    keyword-only."""
    giver, successor = uuid4(), uuid4()
    grant = TransferGrantOutcome(
        artifact_id=uuid4(),
        transferred=True,
        giver=giver,
        successor=successor,
        version_at_transfer=3,
        hold_shape=MESIState.EXCLUSIVE,
        status=TRANSFER_STATUS_PENDING,
    )
    with pytest.raises(FrozenInstanceError):
        grant.successor = giver  # type: ignore[misc]
    with pytest.raises(TypeError):
        TransferGrantOutcome(uuid4(), True)  # type: ignore[misc]
