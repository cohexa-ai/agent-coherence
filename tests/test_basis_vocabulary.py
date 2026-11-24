"""The basis check's vocabulary: closed sets, member validation, the classifier
and the answers' wire forms.

Every reason and recovery string is a WIRE value, so each is pinned to a
hand-written literal beside the code's own constant: a rename or a silent
addition fails here rather than un-matching a consumer's branch.
"""

from __future__ import annotations

import pickle
from uuid import UUID

import pytest

from ccs.core import exceptions as exc
from ccs.core.basis import (
    BASIS_RECOVERY,
    BASIS_TARGET_REFUSAL_REASONS,
    RECOVERY_CLASSES,
    BasisCheckAnswer,
    BasisCommitAnswer,
    BasisMember,
    CheckedMember,
    CommittedTarget,
    MovedMember,
    ResolvedRow,
    TargetRefusal,
    classify_member,
)
from ccs.core.exceptions import (
    BASIS_HOLD_REASONS,
    BASIS_MEMBER_REPLACED,
    BASIS_MEMBER_UNMANAGED,
    HOLD_REASONS,
    BasisCheckAbsent,
    CoherenceError,
)

REG_A = "6f1c2b9e-3d4a-4f5b-8c7d-9e0a1b2c3d4e"
REG_B = "0a1b2c3d-4e5f-4a6b-8c7d-8e9f0a1b2c3d"

_FROZEN_BASIS_HOLD_REASONS = frozenset(
    {
        "version_moved",
        "input_vanished",
        "version_unconfirmed",
        "basis_member_unmanaged",
        "basis_member_replaced",
    }
)
_FROZEN_TARGET_REFUSALS = frozenset(
    {
        "other_holder",
        "stale_read_generation",
        "caller_in_transient_state",
        "handed_off",
        "occ_is_shared_or_invalid_only",
    }
)


# --------------------------------------------------------------------- sets


def test_basis_hold_reasons_equal_the_frozen_five() -> None:
    assert BASIS_HOLD_REASONS == _FROZEN_BASIS_HOLD_REASONS
    assert len(BASIS_HOLD_REASONS) == 5


def test_the_shared_reasons_are_the_fence_constants() -> None:
    assert exc.HOLD_VERSION_MOVED in BASIS_HOLD_REASONS
    assert exc.HOLD_INPUT_VANISHED in BASIS_HOLD_REASONS
    assert exc.HOLD_VERSION_UNCONFIRMED in BASIS_HOLD_REASONS
    assert BASIS_HOLD_REASONS - HOLD_REASONS == {
        BASIS_MEMBER_UNMANAGED,
        BASIS_MEMBER_REPLACED,
    }


def test_the_fence_hold_set_gains_neither_new_string() -> None:
    assert BASIS_MEMBER_UNMANAGED not in HOLD_REASONS
    assert BASIS_MEMBER_REPLACED not in HOLD_REASONS
    assert len(HOLD_REASONS) == 9


def test_recovery_classes_are_exactly_rederive_and_retry() -> None:
    assert RECOVERY_CLASSES == {"rederive", "retry"}


def test_the_recovery_map_is_total_and_only_unconfirmed_retries() -> None:
    assert set(BASIS_RECOVERY) == set(BASIS_HOLD_REASONS)
    assert {r for r, c in BASIS_RECOVERY.items() if c == "retry"} == {"version_unconfirmed"}
    assert {r for r, c in BASIS_RECOVERY.items() if c == "rederive"} == (
        _FROZEN_BASIS_HOLD_REASONS - {"version_unconfirmed"}
    )


def test_target_refusal_tokens_equal_the_frozen_set_apart_from_basis_reasons() -> None:
    assert BASIS_TARGET_REFUSAL_REASONS == _FROZEN_TARGET_REFUSALS
    assert len(BASIS_TARGET_REFUSAL_REASONS) == 5
    assert BASIS_TARGET_REFUSAL_REASONS.isdisjoint(BASIS_HOLD_REASONS)
    assert exc.GIVER_FENCED_REASON in BASIS_TARGET_REFUSAL_REASONS


# ------------------------------------------------------------------- member


def test_member_accepts_no_registration_and_defaults_to_a_file() -> None:
    member = BasisMember("spec.md", 4)
    assert member.registration is None
    assert member.name_kind == "file"


@pytest.mark.parametrize("version", [0, -1, 2**63, True, 1.0, "3", None])
def test_member_rejects_a_version_that_is_not_a_confirmed_one(version: object) -> None:
    with pytest.raises(ValueError):
        BasisMember("spec.md", version)  # type: ignore[arg-type]


def test_member_accepts_the_largest_version() -> None:
    assert BasisMember("spec.md", 2**63 - 1).version == 2**63 - 1


@pytest.mark.parametrize("path", ["", None, 7])
def test_member_rejects_an_empty_or_non_string_path(path: object) -> None:
    with pytest.raises(ValueError):
        BasisMember(path, 1)  # type: ignore[arg-type]


@pytest.mark.parametrize("registration", ["not-a-uuid", "", 5, b"x"])
def test_member_rejects_a_registration_that_is_not_a_uuid(registration: object) -> None:
    with pytest.raises(ValueError):
        BasisMember("spec.md", 1, registration)  # type: ignore[arg-type]


def test_member_rejects_an_unknown_name_kind() -> None:
    with pytest.raises(ValueError):
        BasisMember("spec.md", 1, None, "directory")


def test_member_canonicalises_every_registration_spelling_to_one_string() -> None:
    spellings = [REG_A, REG_A.replace("-", ""), REG_A.upper(), UUID(REG_A)]
    assert {BasisMember("a", 1, s).registration for s in spellings} == {REG_A}  # type: ignore[arg-type]


def test_member_rebuilds_from_its_own_wire_fields() -> None:
    member = BasisMember("cfg", 3, REG_A, "key")
    assert BasisMember(**member.to_wire()) == member
    assert BasisMember.from_wire(member.to_wire()) == member


# --------------------------------------------------------------- classifier


def _classify(
    *,
    recorded: int = 4,
    registration: str | None = REG_A,
    tracked: bool = True,
    row: ResolvedRow | None = ResolvedRow(4, REG_A),
    name_held: bool = False,
) -> str | None:
    return classify_member(
        recorded_version=recorded,
        recorded_registration=registration,
        is_tracked=tracked,
        row=row,
        name_held_by_other_record=name_held,
    )


def test_an_unchanged_member_classifies_as_none() -> None:
    assert _classify() is None


def test_a_member_without_a_recorded_registration_compares_by_version_only() -> None:
    assert _classify(registration=None, row=ResolvedRow(4, REG_B)) is None
    assert _classify(registration=None, row=ResolvedRow(5, REG_B)) == "version_moved"


def test_an_untracked_path_is_unmanaged() -> None:
    assert _classify(tracked=False) == "basis_member_unmanaged"


def test_a_different_registration_is_replaced() -> None:
    assert _classify(row=ResolvedRow(4, REG_B)) == "basis_member_replaced"


def test_a_lower_current_version_is_replaced() -> None:
    assert _classify(recorded=4, row=ResolvedRow(2, REG_A)) == "basis_member_replaced"


def test_no_row_is_vanished() -> None:
    assert _classify(row=None) == "input_vanished"


def test_no_record_for_the_registration_but_the_name_held_elsewhere_is_replaced() -> None:
    assert _classify(row=None, name_held=True) == "basis_member_replaced"


def test_a_higher_version_is_moved() -> None:
    assert _classify(row=ResolvedRow(5, REG_A)) == "version_moved"


def test_untracked_beats_a_vanished_row() -> None:
    assert _classify(tracked=False, row=None) == "basis_member_unmanaged"


def test_untracked_beats_a_replaced_row() -> None:
    assert _classify(tracked=False, row=ResolvedRow(1, REG_B)) == "basis_member_unmanaged"


def test_a_different_registration_beats_a_moved_version() -> None:
    assert _classify(row=ResolvedRow(9, REG_B)) == "basis_member_replaced"


def test_replaced_beats_vanished_when_the_name_is_held() -> None:
    assert _classify(row=None, name_held=True) != "input_vanished"


# ------------------------------------------------------------------ answers


def _moved(path: str = "spec.md", reason: str = "version_moved") -> MovedMember:
    return MovedMember(path, 4, None if reason == "input_vanished" else 5, reason)


def test_a_proceed_check_answer_round_trips() -> None:
    answer = BasisCheckAnswer("proceed", checked=(CheckedMember("a", 4, 4), CheckedMember("b", 2, 2)))
    assert answer.to_wire() == {
        "answer": "proceed",
        "checked": {"a": {"expected": 4, "current": 4}, "b": {"expected": 2, "current": 2}},
    }
    assert BasisCheckAnswer.from_wire(answer.to_wire()) == answer


def test_a_hold_check_answer_keeps_every_moved_member_in_order() -> None:
    answer = BasisCheckAnswer(
        "hold",
        moved=(_moved("a"), _moved("b", "input_vanished")),
        recovery="rederive",
    )
    parsed = BasisCheckAnswer.from_wire(answer.to_wire())
    assert parsed == answer
    assert [m.path for m in parsed.moved] == ["a", "b"]
    assert parsed.moved[1].current is None


def test_a_hold_with_no_member_carries_its_reason_and_retry_class() -> None:
    answer = BasisCheckAnswer("hold", reason="version_unconfirmed", recovery="retry")
    assert answer.to_wire() == {
        "answer": "hold",
        "moved": [],
        "reason": "version_unconfirmed",
        "recovery": "retry",
    }
    assert BasisCheckAnswer.from_wire(answer.to_wire()) == answer


@pytest.mark.parametrize("answer", ["maybe", None])
def test_an_answer_name_outside_proceed_and_hold_is_refused(answer: object) -> None:
    with pytest.raises(ValueError):
        BasisCheckAnswer.from_wire({"answer": answer})
    with pytest.raises(ValueError):
        BasisCommitAnswer.from_wire({"answer": answer})


def test_a_moved_member_with_a_reason_outside_the_set_is_refused() -> None:
    with pytest.raises(ValueError):
        MovedMember("a", 1, 2, "split_pair")


def test_a_won_commit_round_trips_with_the_handoff_key() -> None:
    answer = BasisCommitAnswer(
        "proceed",
        committed=CommittedTarget("out.md", 7),
        handoff={"status": "live"},
    )
    assert answer.to_wire() == {
        "answer": "proceed",
        "committed": {"path": "out.md", "version": 7},
        "handoff": {"status": "live"},
    }
    assert BasisCommitAnswer.from_wire(answer.to_wire()) == answer


def test_a_basis_moved_commit_hold_round_trips_with_rederive() -> None:
    answer = BasisCommitAnswer("hold", moved=(_moved(),), recovery="rederive")
    assert BasisCommitAnswer.from_wire(answer.to_wire()) == answer
    assert "target" not in answer.to_wire()


def test_a_target_refusal_hold_has_empty_moved_and_no_recovery() -> None:
    refusal = TargetRefusal(
        "out.md",
        "stale_read_generation",
        6,
        captured_generation=2,
        owner_generation=3,
    )
    answer = BasisCommitAnswer("hold", target=refusal)
    wire = answer.to_wire()
    assert wire["moved"] == []
    assert "recovery" not in wire
    assert wire["target"] == {
        "path": "out.md",
        "reason": "stale_read_generation",
        "current_version": 6,
        "captured_generation": 2,
        "owner_generation": 3,
    }
    assert BasisCommitAnswer.from_wire(wire) == answer


def test_a_fenced_giver_target_block_carries_successor_and_version_at_transfer() -> None:
    refusal = TargetRefusal(
        "out.md", "handed_off", 6, successor="s-2", version_at_transfer=5
    )
    assert BasisCommitAnswer.from_wire(BasisCommitAnswer("hold", target=refusal).to_wire()) == (
        BasisCommitAnswer("hold", target=refusal)
    )


def test_a_target_refusal_outside_the_closed_set_is_refused() -> None:
    with pytest.raises(ValueError):
        TargetRefusal("out.md", "version_moved", 6)


def test_a_half_generation_pair_is_refused() -> None:
    with pytest.raises(ValueError):
        TargetRefusal("out.md", "other_holder", 6, captured_generation=1)


def test_an_outcome_unknown_hold_round_trips_without_a_recovery_class() -> None:
    answer = BasisCommitAnswer(
        "hold", reason="commit_unconfirmed", degraded=True, held_by="agent-1"
    )
    wire = answer.to_wire()
    assert wire == {
        "answer": "hold",
        "moved": [],
        "reason": "commit_unconfirmed",
        "degraded": True,
        "held_by": "agent-1",
    }
    assert BasisCommitAnswer.from_wire(wire) == answer


def test_a_nothing_ran_commit_hold_is_unconfirmed_with_retry() -> None:
    answer = BasisCommitAnswer("hold", reason="version_unconfirmed", recovery="retry")
    assert BasisCommitAnswer.from_wire(answer.to_wire()) == answer


# -------------------------------------------------------------- absent error


def test_basis_check_absent_pickles_whole_and_is_a_coherence_error() -> None:
    error = BasisCheckAbsent("no basis routes on this coordinator")
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, CoherenceError)
    assert type(restored) is BasisCheckAbsent
    assert str(restored) == "no basis routes on this coordinator"
