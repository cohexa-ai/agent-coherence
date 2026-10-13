# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the effect-fence classification (``ccs.core.fence``).

This is the branch table that decides whether an irreversible effect may still
fire. It used to live inside the in-process wrapper's message builder, where a
second surface answering the same question had no way to reach it — and a
second implementation of a safety rule is the drift this module exists to
remove. These tests pin the rule at its new home, by typed REASON IDENTIFIER
rather than by message text, and they exercise input shapes the in-process path
can never produce (a refused read carrying a real integer generation) because
those are exactly the shapes a coordinator route hands it.

Core-layer tests live FLAT in this repo (``test_invariants.py``,
``test_states.py``, ``test_types.py``); this file follows them.
"""

from __future__ import annotations

import itertools

import pytest

from ccs.core.exceptions import (
    HOLD_CONTENT_CLAIM_ABSENT,
    HOLD_GENERATION_UNCONFIRMED,
    HOLD_GRANT_PREEMPTED,
    HOLD_GRANT_RECLAIMED,
    HOLD_INPUT_VANISHED,
    HOLD_READ_DENIED,
    HOLD_REASONS,
    HOLD_VERSION_MOVED,
    HOLD_VERSION_UNCONFIRMED,
)
from ccs.core.fence import (
    all_legs_confirmed,
    classify_hold,
    confirmed_generation,
    coordinator_holds_content_claim,
)
from ccs.core.types import FenceComparands

# The FROZEN duplicate of the published vocabulary, written out as literals on
# purpose (tests/CLAUDE.md): a set derived from the code under test moves its
# own goalposts, so the edit that breaks the wire contract would also fix the
# expectation and this guard would report green. Cardinality is pinned
# separately so a same-size add+remove cannot slip past set equality.
EXPECTED_HOLD_REASONS = {
    "input_vanished",
    "version_unconfirmed",
    "version_moved",
    "read_denied",
    "content_claim_absent",
    "generation_unconfirmed",
    "grant_reclaimed",
    "grant_preempted",
}
EXPECTED_HOLD_REASON_COUNT = 8

# A clean, all-legs-confirmed input. Every test below states its case as a
# DELTA from this, so what each one is actually varying is visible at a glance
# and an unrelated field can never silently carry the verdict.
_CLEAN = FenceComparands(
    expected_version=5,
    current_version=5,
    expected_generation=7,
    current_generation=7,
    read_refused=False,
    grant_did_not_stand=False,
    content_claim_present=True,
)


def _with(**overrides) -> FenceComparands:
    """A copy of the clean input with named fields replaced."""
    from dataclasses import replace

    return replace(_CLEAN, **overrides)


# --- the seven shipped reasons, by identifier ------------------------------


@pytest.mark.parametrize(
    "overrides, expected_reason",
    [
        # The input is gone: nothing left to compare a comparand against.
        ({"current_version": None, "current_generation": None}, HOLD_INPUT_VANISHED),
        # A 0 on EITHER side is the "could not resolve" sentinel.
        ({"expected_version": 0}, HOLD_VERSION_UNCONFIRMED),
        ({"current_version": 0}, HOLD_VERSION_UNCONFIRMED),
        # The value the decision was derived from changed.
        ({"current_version": 6}, HOLD_VERSION_MOVED),
        # The coordinator refused the re-validate read (strict-mode deny).
        ({"read_refused": True, "current_generation": None}, HOLD_READ_DENIED),
        # The residual bucket: a generation the coordinator never confirmed.
        ({"expected_generation": None}, HOLD_GENERATION_UNCONFIRMED),
        ({"current_generation": None}, HOLD_GENERATION_UNCONFIRMED),
        # Both generations confirmed and they differ: a sweep reclaimed the
        # grant the decision was read under, version untouched.
        ({"current_generation": 8}, HOLD_GRANT_RECLAIMED),
        # Both comparands confirmed AND unchanged, but the re-validate read was
        # served without a standing grant.
        ({"grant_did_not_stand": True}, HOLD_GRANT_PREEMPTED),
    ],
)
def test_each_shipped_reason_is_produced_by_its_trigger(overrides, expected_reason) -> None:
    """Every reason the shipped in-process fence could produce is still produced
    by the input combination that triggered it, named by IDENTIFIER.

    Prevents: a silent re-labelling during the move out of the message builder.
    An agent branches on ``hold_cause``, so a reason that quietly becomes a
    different reason changes which recovery the agent runs — reacquire-and-retry
    against an operator escalation — while every message-text assertion in the
    suite still passes.
    """
    assert classify_hold(_with(**overrides)) == expected_reason


def test_zero_version_and_absent_generation_are_sentinels_not_values() -> None:
    """A zero version and an absent generation each answer under their OWN
    unconfirmed reason instead of comparing equal to their counterpart.

    Prevents the SENTINEL-⇒-UNCONFIRMED-⇒-HOLD invariant collapsing: two zeros
    compare equal and two ``None``s compare equal, so a table that compared
    before it checked for sentinels would turn "the coordinator confirmed
    nothing" into "I know it is unchanged" — an admit on exactly the input this
    layer exists to refuse.
    """
    both_zero = _with(expected_version=0, current_version=0)
    assert both_zero.expected_version == both_zero.current_version
    assert classify_hold(both_zero) == HOLD_VERSION_UNCONFIRMED

    both_absent = _with(expected_generation=None, current_generation=None)
    assert both_absent.expected_generation == both_absent.current_generation
    assert classify_hold(both_absent) == HOLD_GENERATION_UNCONFIRMED


def test_grant_preemption_holds_with_neither_comparand_moved() -> None:
    """A peer's write-claim preemption HOLDs even though both comparands are
    confirmed and unchanged.

    Prevents the pair-only fence returning: a pessimistic write-acquire ends the
    caller's grant without a commit (version unmoved) and ``trigger="write"``
    sits outside ``EPOCH_BUMP_TRIGGERS`` (generation unmoved), so the comparand
    pair is structurally blind here and only the grant-state leg sees it.
    """
    preempted = _with(grant_did_not_stand=True)
    assert preempted.current_version == preempted.expected_version
    assert preempted.current_generation == preempted.expected_generation
    assert classify_hold(preempted) == HOLD_GRANT_PREEMPTED


# --- the refused-read leg, on the route's input shape ----------------------


def test_refused_read_holds_even_when_both_comparands_match() -> None:
    """A refused read reports the refused-read reason on the ROUTE's input
    shape: a real integer generation on both sides, both comparands matching.

    Prevents the leg's old conjunction with an unconfirmed generation coming
    back. In process that conjunction was always satisfied — a strict deny
    carries no generation at all — so it was invisible there. A caller holding
    the registry's real generation (a coordinator route) falls straight past a
    conjunctive leg and, with matching comparands, reaches PROCEED: it would
    fire an irreversible effect on a view the coordinator had just refused to
    serve.
    """
    refused = _with(read_refused=True)
    assert isinstance(refused.current_generation, int)
    assert refused.current_generation == refused.expected_generation
    assert refused.current_version == refused.expected_version
    assert classify_hold(refused) == HOLD_READ_DENIED


# --- the no-claim reason, and its control ----------------------------------


@pytest.mark.parametrize(
    "recorded_content_hash, reported_generation",
    [
        # The empty-string form: what most code paths record when a first
        # observation carried no caller hash. The coordinator's hash_differs
        # predicate needs a TRUTHY hash on both sides, so this form never
        # demoted the generation — the fence ADMITTED and the effect fired
        # against a value nothing backs.
        ("", 7),
        (None, 7),
        # The all-``f`` launch-gate sentinel: no real SHA-256 matches it, so it
        # DOES trip hash_differs on the stale path and arrives with its
        # generation already demoted to None. Without its own leg it lands in
        # the residual bucket and is byte-identical on the wire to a genuinely
        # degraded read.
        ("f" * 64, None),
    ],
)
def test_no_content_claim_answers_under_its_own_reason(
    recorded_content_hash, reported_generation
) -> None:
    """A coordinator holding NO content claim HOLDs under its own reason — for
    both forms that seed it, whichever generation they arrive with.

    Prevents two distinct regressions at once. The empty form is a behaviour
    CHANGE: it admitted before, because a cannot-tell was allowed to collapse
    into clean. The all-``f`` form is a mis-labelling: it held, but under the
    residual reason, with the same recovery hint and retryability as a degraded
    read — and the recoveries differ (re-read your bytes vs. check the daemon
    and call an operator).
    """
    claim = coordinator_holds_content_claim(recorded_content_hash)
    assert claim is False
    comparands = _with(
        content_claim_present=claim,
        current_generation=reported_generation,
        expected_generation=reported_generation,
    )
    assert classify_hold(comparands) == HOLD_CONTENT_CLAIM_ABSENT


def test_degraded_read_still_answers_under_the_residual_reason() -> None:
    """THE CONTROL for the test above: a genuinely degraded read — the
    coordinator claims content, it simply confirmed no generation — still
    answers under the RESIDUAL reason, so the two stay distinguishable by
    identifier.

    Without this arm a test asserting only the new reason cannot see the two
    collapsing back into one: a mutation that answered the no-claim case with
    the residual reason, or that routed every unconfirmed generation to the new
    one, would leave the no-claim assertion green.
    """
    degraded = _with(current_generation=None, content_claim_present=True)
    assert classify_hold(degraded) == HOLD_GENERATION_UNCONFIRMED
    assert HOLD_GENERATION_UNCONFIRMED != HOLD_CONTENT_CLAIM_ABSENT


def test_content_claim_normaliser_accepts_a_real_recorded_hash() -> None:
    """A real recorded hash IS a claim — the normaliser must not refuse every
    hash and turn the new leg into a blanket deny that holds every effect."""
    assert coordinator_holds_content_claim("a" * 64) is True
    assert classify_hold(_with(content_claim_present=True)) is None


# --- the generation-demotion normaliser (shared with the client) -----------


def test_confirmed_generation_demotes_a_real_generation_on_a_hash_mismatch() -> None:
    """A differing content hash demotes an otherwise-REAL integer generation to
    unconfirmed.

    Prevents the warn-mode fail-open path firing an effect on superseded bytes:
    the coordinator allows the read, so the version comparand re-validates
    clean, and only the authority comparand can say "I cannot vouch that these
    bytes are the content at this version".
    """
    assert confirmed_generation(7, content_hash_differs=False) == 7
    assert confirmed_generation(7, content_hash_differs=True) is None


def test_confirmed_generation_keeps_zero_and_refuses_non_integers() -> None:
    """``0`` is a REAL generation and survives; ``None``, a bool and a string
    are not generations and normalise to the unconfirmed sentinel.

    ``True`` is the one that bites: ``isinstance(True, int)`` is True, so a
    bare int check would accept a JSON ``true`` as generation 1 and compare it
    as a value.
    """
    assert confirmed_generation(0, content_hash_differs=False) == 0
    assert confirmed_generation(None, content_hash_differs=False) is None
    assert confirmed_generation(True, content_hash_differs=False) is None
    assert confirmed_generation("7", content_hash_differs=False) is None


# --- proceed is affirmative, and the drift guard ---------------------------


def test_proceed_requires_the_affirmative_predicate() -> None:
    """Proceed is returned only for an input the all-legs-confirmed predicate
    accepts, and that predicate rejects every HOLD input.

    Prevents proceed becoming a fall-through: a fence that admits by running
    off the end of its guards starts admitting silently the moment an input
    arrives that no guard covers — and widening the input space with a second
    caller is precisely what this module is for.
    """
    assert all_legs_confirmed(_CLEAN) is True
    assert classify_hold(_CLEAN) is None
    for overrides in (
        {"current_version": None},
        {"expected_version": 0},
        {"current_version": 6},
        {"read_refused": True},
        {"content_claim_present": False},
        {"current_generation": None},
        {"current_generation": 8},
        {"grant_did_not_stand": True},
    ):
        assert all_legs_confirmed(_with(**overrides)) is False, overrides


def test_fall_through_fails_loud_instead_of_proceeding(monkeypatch) -> None:
    """A state that matches no HOLD leg AND fails the proceed predicate RAISES.

    That state is unreachable by construction today — the branch table and the
    predicate are exact complements — so the drift it guards against has to be
    SIMULATED, by making the predicate disagree with the table. That is the
    point: the guard exists for the version of this file where someone narrows
    a leg, and on that day the classifier must fail loud rather than hand a
    caller a silent proceed for an input nothing examined.
    """
    monkeypatch.setattr("ccs.core.fence.all_legs_confirmed", lambda comparands: False)
    with pytest.raises(AssertionError) as exc:
        classify_hold(_CLEAN)
    assert "drifted" in str(exc.value)


def test_proceed_and_the_predicate_agree_across_the_whole_input_space() -> None:
    """Over EVERY input combination, "classify_hold returned proceed" and "the
    affirmative predicate holds" are the same answer.

    The complement of the simulated-drift test above: that one proves the guard
    fires, this one proves the guard is not papering over a real disagreement
    that already exists somewhere in the input space.
    """
    for comparands in _exhaustive_inputs():
        assert (classify_hold(comparands) is None) == all_legs_confirmed(comparands), (
            comparands
        )


# --- the published vocabulary ----------------------------------------------


def _exhaustive_inputs():
    """Every combination of the classifier's inputs, with each field taking one
    value per equivalence class: a sentinel, a real value, and (for the
    comparands) a DIFFERENT real value so "moved" is reachable."""
    for (
        expected_version,
        current_version,
        expected_generation,
        current_generation,
        read_refused,
        grant_did_not_stand,
        content_claim_present,
    ) in itertools.product(
        (0, 5),
        (None, 0, 5, 6),
        (None, 7),
        (None, 7, 8),
        (False, True),
        (False, True),
        (False, True),
    ):
        yield FenceComparands(
            expected_version=expected_version,
            current_version=current_version,
            expected_generation=expected_generation,
            current_generation=current_generation,
            read_refused=read_refused,
            grant_did_not_stand=grant_did_not_stand,
            content_claim_present=content_claim_present,
        )


def test_every_reason_the_classifier_returns_is_published() -> None:
    """Asserted over the classifier's WHOLE output range, not a sampled few:
    every reason it can return is a member of the published set, and every
    published reason is reachable.

    Prevents both halves of a vocabulary drift. A reason the classifier returns
    but the set omits is a value no consumer's membership test will match, so a
    real HOLD reads as an unrecognised one; a reason the set publishes but
    nothing can produce is a documented recovery path that never fires.
    """
    produced = {
        reason
        for reason in (classify_hold(c) for c in _exhaustive_inputs())
        if reason is not None
    }
    assert produced <= HOLD_REASONS, produced - HOLD_REASONS
    assert produced == HOLD_REASONS


def test_published_vocabulary_matches_its_frozen_duplicate() -> None:
    """The published set equals a set written out by hand here, and has exactly
    the pinned cardinality.

    A derived expectation would move with the code it checks. The cardinality
    pin is separate because a same-size add-and-remove passes set equality
    against a stale literal — and a RENAME is a wire break, never an addition.
    """
    assert HOLD_REASONS == EXPECTED_HOLD_REASONS
    assert len(HOLD_REASONS) == EXPECTED_HOLD_REASON_COUNT


# --- the value object ------------------------------------------------------


@pytest.mark.parametrize(
    "omitted",
    ["read_refused", "grant_did_not_stand", "content_claim_present"],
)
def test_comparands_cannot_be_built_without_grant_state(omitted) -> None:
    """Omitting a grant-state field fails LOUDLY at construction, naming the
    field it missed.

    Prevents a default standing in for an answer nobody supplied. A field that
    defaults to "nothing was wrong" turns a caller's oversight into an admit on
    the one path where an admit IS the lost update — and the failure would be
    invisible, since the object would construct and the fence would proceed.
    """
    fields = {
        "expected_version": 5,
        "current_version": 5,
        "expected_generation": 7,
        "current_generation": 7,
        "read_refused": False,
        "grant_did_not_stand": False,
        "content_claim_present": True,
    }
    del fields[omitted]
    with pytest.raises(TypeError) as exc:
        FenceComparands(**fields)
    assert omitted in str(exc.value)


def test_comparands_are_keyword_only_and_frozen() -> None:
    """The comparands cannot be passed positionally, and cannot be mutated
    after construction.

    Keyword-only is what makes a transposition UNEXPRESSIBLE rather than merely
    unlikely: two ints, two optional ints and three bools in a row type-check
    in either order, and swapping the expected and current sides of a comparand
    is a silent admit. Frozen keeps the object the classifier examined the same
    one the message was formatted from.
    """
    with pytest.raises(TypeError):
        FenceComparands(5, 5, 7, 7, False, False, True)  # type: ignore[misc]
    with pytest.raises(Exception):
        _CLEAN.expected_version = 9  # type: ignore[misc]


def test_a_refused_read_outranks_an_absent_content_claim() -> None:
    """Pins leg 4 over leg 5 -- the one precedence pair the docstring's "each
    leg's precedence is pinned by a test" claim did not actually cover.

    Both predicates are reachable AT ONCE on the coordinator route: an
    artifact seeded with no content claim (the launch-gate all-``f`` form, or
    a first observation that carried no caller hash) whose holder is then
    invalidated by a peer under strict mode arrives with ``read_refused`` and
    ``content_claim_present=False`` both set. Without this test the two legs
    could be swapped and nothing would go red -- and their order is published
    contract, so a swap changes the reason a real, reachable state answers
    with. The refusal wins because it is the immediate blocker: the
    coordinator declined to serve the read at all, so there is no served view
    whose content claim could be evaluated.
    """
    both = _with(read_refused=True, content_claim_present=False)
    assert classify_hold(both) == HOLD_READ_DENIED

    # Controls. Without these the assertion above passes for the wrong
    # reason -- a table that always answered ``read_denied`` would satisfy it.
    only_refused = _with(read_refused=True, content_claim_present=True)
    assert classify_hold(only_refused) == HOLD_READ_DENIED
    only_no_claim = _with(read_refused=False, content_claim_present=False)
    assert classify_hold(only_no_claim) == HOLD_CONTENT_CLAIM_ABSENT
