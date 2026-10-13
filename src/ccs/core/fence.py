# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The effect-fence classification — ONE branch table, every surface calls it.

"May this irreversible effect still fire, and if not, WHY" is a safety rule.
It is answered here, once, as a pure function over plain values, so that the
in-process wrapper (``adapters.effect_gate``) and the coordinator route that
answers the same question over HTTP cannot drift apart. A second
implementation of a safety rule is the failure this module exists to remove,
not to add.

Core is the right home: the classification takes only plain values (two version
comparands, two generation comparands, three booleans) and the application
layer already answers a weaker version of this question, so an adapters home
would bar that caller forever.

Shape: :func:`classify_hold` RETURNS a typed reason or ``None`` (proceed), it
never raises for a HOLD — the sibling ``invariants`` module raises because a
violated invariant is a fault, whereas a HOLD is an ordinary, expected answer
the caller must branch on. The one raise here is the drift guard, and it fires
only on a state no leg claimed and the affirmative proceed predicate rejected.
"""

from __future__ import annotations

from .exceptions import (
    HOLD_CONTENT_CLAIM_ABSENT,
    HOLD_GENERATION_UNCONFIRMED,
    HOLD_GRANT_PREEMPTED,
    HOLD_GRANT_RECLAIMED,
    HOLD_INPUT_VANISHED,
    HOLD_READ_DENIED,
    HOLD_VERSION_MOVED,
    HOLD_VERSION_UNCONFIRMED,
)
from .types import FenceComparands

# A recorded content hash of all ``f`` is the launch-gate scenario seed: no real
# SHA-256 matches it, so it is a placeholder standing in for "we had to write
# something", never a claim about bytes. The empty string and ``None`` are the
# other two no-claim seeds (a first observation that carried no caller hash
# surfaces as ``None`` on the Artifact). All three mean the SAME thing to a
# caller — the coordinator cannot identify this artifact's content — so they
# resolve to one answer rather than three.
#
# The all-f literal also exists as ``_F_SENTINEL_CONTENT_HASH`` in the
# coordinator server, where the foreign-write detector imports it for its
# canonical-hash no-claim check. The coordinator's own hash-differs suppression
# no longer reads it -- that path asks this module instead -- so the two copies
# now serve different mechanisms. Deduping them would need core to import from
# an adapter, which the layering forbids; grep both if either moves.
_NO_CLAIM_CONTENT_HASHES: frozenset[str] = frozenset({"", "f" * 64})


def coordinator_holds_content_claim(recorded_content_hash: str | None) -> bool:
    """Return whether the coordinator records a real content claim.

    ``False`` for ``None``, the empty string and the all-``f`` sentinel — three
    spellings of "nothing was recorded". This is INPUT NORMALISATION, not a
    verdict: it feeds ``FenceComparands.content_claim_present`` so the branch
    table can answer under its own reason. Deciding it inside a route handler
    is what would make it a second copy of the rule.
    """
    if recorded_content_hash is None:
        return False
    return recorded_content_hash not in _NO_CLAIM_CONTENT_HASHES


def confirmed_generation(
    reported_generation: int | None, *, content_hash_differs: bool
) -> int | None:
    """Normalise a reported ownership generation to a CONFIRMED one, or ``None``.

    Two demotions, both fail-closed:

    - a non-int (or a ``bool``, which ``isinstance(x, int)`` would otherwise
      wave through) is not a generation at all;
    - a REAL integer generation is demoted to ``None`` when the coordinator
      reports the caller's content hash differs from the content it records at
      that version. The generation is the AUTHORITY comparand — "is the grant
      these bytes were read under still standing" — and a hash mismatch means
      the coordinator cannot vouch that the bytes in hand ARE the content at
      that version. In warn mode such a read is a fail-open allow, so the
      version comparand alone would re-validate clean at an effect boundary and
      fire a decision derived from superseded bytes.

    ``None`` is never coerced to ``0``: ``0`` is a REAL generation, and the
    sentinel exists precisely so that comparing it LOSES.
    """
    if content_hash_differs:
        return None
    if isinstance(reported_generation, int) and not isinstance(reported_generation, bool):
        return reported_generation
    return None


def all_legs_confirmed(comparands: FenceComparands) -> bool:
    """Return whether EVERY leg of the fence affirmatively cleared.

    This is the proceed predicate, stated positively and independently of the
    branch table. A fence that proceeds by falling off the end of its guards
    silently starts admitting the moment someone adds an input the guards do
    not cover — and adding a second caller with a wider input space is exactly
    what this module is for. Proceeding therefore requires saying YES to all of
    it: the input exists, both comparands are confirmed (no sentinels) and
    unmoved, the read was not refused, the coordinator claims the content, and
    the grant stood.
    """
    return (
        comparands.current_version is not None
        and comparands.expected_version != 0
        and comparands.current_version != 0
        and comparands.current_version == comparands.expected_version
        and not comparands.read_refused
        and comparands.content_claim_present
        and comparands.expected_generation is not None
        and comparands.current_generation is not None
        and comparands.current_generation == comparands.expected_generation
        and not comparands.grant_did_not_stand
    )


def classify_hold(comparands: FenceComparands) -> str | None:
    """Return the typed HOLD reason, or ``None`` when the effect may proceed.

    The branch order is load-bearing and each leg's precedence is pinned by a
    test; changing it changes which reason a caller branches on:

    1. ``input_vanished`` — nothing left to compare against.
    2. ``version_unconfirmed`` — a ``0`` on either side is the "could not
       resolve" sentinel; comparing sentinels as values is how "I do not know"
       becomes "I know it is unchanged".
    3. ``version_moved`` — the value the decision was derived from changed.
    4. ``read_denied`` — the coordinator REFUSED the re-validate read. Fires on
       the refusal ALONE. It once required an unconfirmed generation as well,
       which held only because an in-process strict deny carries no generation;
       a caller that holds the registry's real generation (the coordinator
       route) would have fallen past this leg and either mislabelled its HOLD
       or, with matching comparands, proceeded on a view the coordinator had
       just refused to serve.
    5. ``content_claim_absent`` — the coordinator records no content hash at
       all, so it cannot vouch that these bytes are the content at that
       version. Ordered ABOVE the residual bucket deliberately: both no-claim
       forms must land here, and the all-``f`` form arrives with its generation
       already demoted to ``None``, so a lower position would let it fall into
       ``generation_unconfirmed`` and become byte-identical on the wire to a
       genuinely degraded read — the same reason, hint and retryability for two
       cases whose recovery differs (re-read your bytes vs. call an operator).
    6. ``generation_unconfirmed`` — the residual bucket: a degraded read, an
       unconfirmable out-of-band edit, or a coordinator too old to report
       generations.
    7. ``grant_reclaimed`` — both generations confirmed and they differ: a
       sweep reclaimed the grant the decision was read under, while the version
       never moved.
    8. ``grant_preempted`` — both comparands confirmed AND unchanged, yet the
       re-validate read was served without a standing grant.

    Raises:
        AssertionError: the state matched no leg AND failed the affirmative
            proceed predicate — the two have drifted. Fail loud rather than
            return a silent proceed, which is what a bare fall-through would
            become the moment a leg stops covering its case.
    """
    if comparands.current_version is None:
        return HOLD_INPUT_VANISHED
    if comparands.expected_version == 0 or comparands.current_version == 0:
        return HOLD_VERSION_UNCONFIRMED
    if comparands.current_version != comparands.expected_version:
        return HOLD_VERSION_MOVED
    if comparands.read_refused:
        return HOLD_READ_DENIED
    if not comparands.content_claim_present:
        return HOLD_CONTENT_CLAIM_ABSENT
    if comparands.expected_generation is None or comparands.current_generation is None:
        return HOLD_GENERATION_UNCONFIRMED
    if comparands.current_generation != comparands.expected_generation:
        return HOLD_GRANT_RECLAIMED
    if comparands.grant_did_not_stand:
        return HOLD_GRANT_PREEMPTED
    if all_legs_confirmed(comparands):
        return None
    raise AssertionError(
        "internal: classify_hold() fell past every HOLD leg with a state the "
        f"proceed predicate rejects ({comparands!r}) -- the branch table and "
        "all_legs_confirmed() have drifted; refusing to invent a proceed"
    )
