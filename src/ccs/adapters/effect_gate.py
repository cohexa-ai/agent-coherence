# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Effect-ordering gate wrapper (EO-5) -- the builder-facing surface.

``gate()`` is a plain-Python drop-in over a :class:`CoherentVolume` handle:
capture an input's decision-time ``(version, owner_generation)`` pair, run the
caller's decision, re-read the pair at the effect boundary, and fire the
escaping effect only if BOTH are unchanged; else HOLD (raise
:class:`~ccs.core.exceptions.StaleView`) before the effect runs. The two
comparands answer different questions: the version answers "is the value still
the one ``decide`` saw", the ownership generation answers "is the grant it was
read under still standing" -- a sweep reclamation of a stalled holder advances
the generation WITHOUT a version move, which a version-only check cannot see
(the same distinction the read-generation fence draws at the commit seam). One
revocation moves neither comparand: a peer's pessimistic write-acquire ends
the caller's grant with no commit (version unmoved) and no epoch bump
(``trigger="write"`` is deliberately outside ``EPOCH_BUMP_TRIGGERS``), so the
re-validate additionally requires the read that answers it to be served under
a STANDING grant -- a stale-status re-read HOLDs. That re-read travels
``verify_only`` so the coordinator does NOT re-grant the preempted holder on
it; the HOLD is level-triggered, so a bare re-check re-HOLDs until
``reacquire()`` re-mints a live grant rather than silently re-arming the
preempted decision on retry.
It reuses the shipped ``CoherentVolume`` optimistic-concurrency primitives and
never reimplements the coordinator gate.

Honest scope:

- **Escaping effects only.** A pure *write* effect uses
  :meth:`CoherentVolume.write_cas_at` directly -- that is the atomic, no-window
  path; wrapping it here would add nothing over the shipped CAS.
- **Ordering, not rollback** (the gate fires pre-effect and never rolls back).
  For an escaping effect there is a residual re-validate -> fire window this
  layer cannot close (the effect escapes), so the gate *narrows* a stale fire, it
  does not *eliminate* it.
- **Single-host, cooperative opt-in, deny is pull-not-push, correctness first.**
- **Single-artifact read-set.** Gating on several mutually-consistent inputs is
  the shipped in-process ``CoordinatorService.effect_gate`` (a coherent cut), not
  this wrapper.

The HOLD is *raised* (a drop-in guard). Recover with ``volume.reacquire(path)``
for fresh bytes, then re-decide and re-gate.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, TypeVar

from ccs.core.exceptions import (
    HOLD_CONTENT_CLAIM_ABSENT,
    HOLD_GENERATION_UNCONFIRMED,
    HOLD_GRANT_PREEMPTED,
    HOLD_GRANT_RECLAIMED,
    HOLD_INPUT_VANISHED,
    HOLD_READ_DENIED,
    HOLD_VERSION_MOVED,
    HOLD_VERSION_UNCONFIRMED,
    InvariantViolationError,
    StaleView,
)
from ccs.core.fence import classify_hold
from ccs.core.types import FenceComparands

if TYPE_CHECKING:
    from ccs.adapters.coherent_volume import CoherentVolume

_Decision = TypeVar("_Decision")
_Result = TypeVar("_Result")


def gate(
    volume: "CoherentVolume",
    path: str | os.PathLike[str],
    *,
    decide: Callable[[bytes], _Decision],
    effect: Callable[[_Decision], _Result],
) -> _Result:
    """Fire ``effect`` only if ``path`` is unchanged from the version ``decide``
    saw AND the grant it was read under still stands; otherwise raise
    :class:`~ccs.core.exceptions.StaleView` before firing.

    Steps:

    1. Capture ``(bytes, version, owner_generation)`` from ONE
       ``read_with_version_generation`` (the value and both comparands come from
       the same read -- the split-comparand discipline that keeps a
       stale-derived decision from firing).
    2. ``decision = decide(bytes)``.
    3. Re-read the current ``(version, owner_generation)`` at the effect
       boundary. If either moved, the file vanished, or the coordinator could
       not confirm either comparand (a degraded read surfaces version ``0``; an
       older / denying / degraded coordinator surfaces generation ``None``),
       the gate HOLDs: it raises ``StaleView`` carrying
       ``expected_version`` / ``current_version`` and ``expected_generation`` /
       ``current_generation``, and the effect NEVER runs on unconfirmed or
       stale input. The generation leg is what catches the reclaim case: a
       sweep reclaimed this holder's grant (its decision is a zombie's) while
       the bytes -- and so the version -- never moved. The re-read must also
       itself be served under a STANDING grant: a peer's pessimistic
       write-acquire preempts this holder while moving NEITHER comparand (no
       commit yet, and ``trigger="write"`` does not bump the epoch), so a
       stale-status re-read HOLDs even with the pair unchanged.
    4. Otherwise fire ``effect(decision)`` and return its result.

    Escaping effects only, single-host, ordering-not-rollback. The
    re-validate -> fire window is unclosable for an escaping effect. For a write
    effect use ``volume.write_cas_at`` directly.

    Args:
        volume: a ``CoherentVolume`` attached to the coordinator that tracks
            ``path``.
        path: the workspace-relative managed artifact whose
            ``(version, owner_generation)`` pair gates the effect (the
            single-artifact read-set).
        decide: ``(bytes) -> decision`` -- reads the captured bytes and returns a
            decision threaded to ``effect``.
        effect: ``(decision) -> result`` -- the escaping side effect, fired only
            if the input is unchanged at the re-validate point.

    Returns:
        The value ``effect`` returned.

    Raises:
        StaleView: the input moved, vanished, or lost its grant between capture
            and fire, or the capture read was refused before ``decide`` ran
            because the bytes on disk are not the content at the coordinator's
            version (a peer's commit still reaching disk, an out-of-band edit,
            or a commit whose disk write failed). The effect did not run.
            Recover via ``volume.reacquire(path)`` then re-decide. A refusal
            from a peer's commit clears within seconds, so retry that for a
            few seconds; if it outlasts that, re-reading will not clear it:
            ``write()`` the bytes ``volume.reacquire(path)`` returned (or a
            merge of them) to record them, then re-decide.
        InvariantViolationError: ``volume`` cannot report the grant state of
            its reads, so the fence cannot answer its third leg for it (see
            :func:`_require_grant_state`). A ``CoherentVolume`` always can; a
            duck-typed stand-in must declare both flags. Not a ``StaleView``,
            because no re-read can clear it -- but still under
            ``CoherenceError``, so an existing handler catches it.
    """
    if not callable(decide):
        raise TypeError("gate() requires a callable decide=")
    if not callable(effect):
        raise TypeError("gate() requires a callable effect=")

    data, expected_version, expected_generation = volume.read_with_version_generation(
        path
    )
    decision = decide(data)
    check_fence(
        volume,
        path,
        expected_version=expected_version,
        expected_generation=expected_generation,
    )

    # Re-validate passed: fire. The residual re-validate -> fire window is
    # unclosable for an escaping effect; the gate gates pre-fire and never rolls
    # back.
    return effect(decision)


# The human half of a HOLD, keyed by the typed reason ``ccs.core.fence``
# already decided. Pure formatting: every entry reads values off the comparands
# and NONE of them re-tests a condition, so the message can never disagree with
# the ``hold_cause`` an agent branches on. The prose is byte-stable by house
# rule (a model's retry loop measurably worsens when deny bytes change between
# attempts), so drift rides the ``expected_*``/``current_*`` attributes.
_HOLD_DETAILS: dict[str, Callable[[FenceComparands], str]] = {
    HOLD_INPUT_VANISHED: lambda c: "vanished",
    HOLD_VERSION_UNCONFIRMED: (
        lambda c: "could not be confirmed (coordinator degraded or unresolved)"
    ),
    HOLD_VERSION_MOVED: lambda c: f"moved to v{c.current_version}",
    # The coordinator REFUSED the re-read (strict mode). A distinct, recoverable
    # answer -- and on the strict path it is how a sweep reclaim actually
    # reaches the client, so folding it into the residual bucket would hide the
    # very case this fence exists for.
    HOLD_READ_DENIED: (
        lambda c: "was denied at re-read by the coordinator (view is INVALID)"
    ),
    # The coordinator records no content hash for this artifact at all, so it
    # cannot vouch that the bytes in hand are the content at the version it
    # reports. Recovery is a fresh read, NOT an operator -- which is exactly why
    # this does not share the residual bucket's prose.
    HOLD_CONTENT_CLAIM_ABSENT: (
        lambda c: (
            "has no recorded content claim at the coordinator (it cannot "
            "vouch that these bytes are the content at this version)"
        )
    ),
    HOLD_GENERATION_UNCONFIRMED: (
        lambda c: (
            "has no confirmed ownership generation (degraded read, an "
            "unconfirmable out-of-band edit, or a coordinator that does not "
            "report generations)"
        )
    ),
    # Version unchanged, both generations confirmed: the grant was reclaimed out
    # from under the decision -- the failure class a version-only check cannot
    # see.
    HOLD_GRANT_RECLAIMED: (
        lambda c: (
            f"had its grant reclaimed (ownership generation "
            f"g{c.expected_generation} -> g{c.current_generation}, version unchanged)"
        )
    ),
    # Both comparands unchanged AND confirmed, yet the re-validate read was
    # served WITHOUT a standing grant: the caller's grant did not stand at the
    # re-validate. The usual cause is a peer write-claim acquire that preempted
    # the caller between capture and fire -- no commit has landed (version
    # unmoved) and trigger="write" deliberately does not bump the ownership
    # epoch, so the pair is structurally blind here; only the grant-state answer
    # on the re-read itself sees it. A grantless re-read with NO peer lands here
    # too (a re-minted identity gating with an earlier read's comparands), so
    # the detail says "typically" rather than asserting a peer the fence never
    # observed.
    HOLD_GRANT_PREEMPTED: (
        lambda c: (
            f"was not under a standing grant at re-validate (version and "
            f"ownership generation g{c.expected_generation} unchanged -- "
            f"typically a peer write-claim preemption)"
        )
    ),
}


def _held(
    path: str | os.PathLike[str],
    comparands: FenceComparands,
    reason: str,
) -> StaleView:
    """Format the HOLD exception for a reason :func:`classify_hold` ALREADY
    decided, carrying the drift and the path.

    A formatter, not a decision: the branch table lives in ``ccs.core.fence``
    so the coordinator route and this wrapper answer with one rule. The only
    branch left here is the lookup miss, which fails LOUD — a missing entry
    would otherwise let a newly minted reason reach a caller with no message
    at all, or (worse, on a ``.get(...) or ""`` shape) an empty one that reads
    like nothing is wrong.
    """
    detail_of = _HOLD_DETAILS.get(reason)
    if detail_of is None:
        raise AssertionError(
            f"internal: no HOLD message for reason {reason!r} -- "
            "ccs.core.fence.classify_hold returned a reason effect_gate cannot "
            "render; the vocabulary and its formatter have drifted"
        )
    # A real CoherentVolume rejects a non-PathLike path before gate() runs, but
    # volume is duck-typed at runtime -- never let fspath() mask the HOLD.
    try:
        target = os.fspath(path)
    except TypeError:
        target = str(path)
    exc = StaleView(
        f"effect held: {target} {detail_of(comparands)} since it was read at "
        f"v{comparands.expected_version}; effect not fired (reacquire and re-decide)"
    )
    exc.expected_version = comparands.expected_version
    exc.current_version = comparands.current_version
    exc.expected_generation = comparands.expected_generation
    exc.current_generation = comparands.current_generation
    exc.hold_cause = reason
    return exc


#: The two answers the fence's third leg is MADE of: whether the coordinator
#: REFUSED the re-validate read, and whether it served that read without a
#: standing grant. A volume REPORTS both on its last read and the fence cannot
#: derive either from the ``(version, owner_generation)`` pair -- which is
#: precisely why a peer's write-claim preemption is invisible without them.
_GRANT_STATE_FLAGS: tuple[str, str] = ("_last_read_denied", "_last_read_stale")


def _require_grant_state(volume: "CoherentVolume") -> None:
    """Refuse a volume that cannot report the grant state of its last read.

    These two were once read through a defaulting accessor, justified as
    compatibility for duck-typed volumes that predated the flags. That
    allowance is WITHDRAWN: an absent flag is indistinguishable from an
    affirmative "nothing was wrong", so it did not degrade the fence, it
    silently DELETED a leg -- the one that catches a peer's pessimistic
    write-acquire, which moves NEITHER comparand. ``CoherentVolume`` declares
    both in its class body, so every real instance carries them from
    construction and nothing shipped reaches this raise.

    Typed INSIDE the coherence hierarchy on purpose, and that -- not the
    requirement itself -- is the compatibility that matters here: a bare
    ``AttributeError`` from reading the flag directly, or the ``TypeError`` a
    missing :class:`~ccs.core.types.FenceComparands` keyword would raise, is
    catchable as neither :class:`~ccs.core.exceptions.StaleView` nor its base
    :class:`~ccs.core.exceptions.CoherenceError`, so a caller that catches a
    HOLD today (``ccs.mcp.server`` catches the base around :func:`check_fence`)
    would CRASH where it used to hold. It is deliberately NOT a ``StaleView``
    either: re-reading cannot supply a flag the volume never declares, so the
    retryable ``stale_view`` recovery would send a cooperating agent into a
    reacquire loop that can never clear.

    Raises:
        InvariantViolationError: the volume reports neither flag, or only one
            of the two -- a half-equipped volume keeps the deny leg and loses
            the preemption leg, which is the same fail-open wearing one flag.
    """
    missing = [name for name in _GRANT_STATE_FLAGS if not hasattr(volume, name)]
    if missing:
        raise InvariantViolationError(
            f"{type(volume).__name__} cannot report the grant state of its "
            f"reads (missing {', '.join(missing)}), so the effect fence cannot "
            "see a write-claim preemption -- which moves neither the version "
            "nor the ownership generation. Declare both flags and set them on "
            "every read_with_version_generation()"
        )


def check_fence(
    volume: "CoherentVolume",
    path: str | os.PathLike[str],
    *,
    expected_version: int,
    expected_generation: int | None,
) -> None:
    """Re-validate a comparand the caller captured earlier; raise
    :class:`~ccs.core.exceptions.StaleView` (a HOLD) unless the input is
    provably unchanged AND still under the grant it was read from.

    This is the pull-based half of :func:`gate` -- the same check, for callers
    whose decision step happens somewhere this process cannot reach with a
    callable: an agent that read through one tool call, reasoned, and is about
    to dispatch an irreversible effect through another (the ``swg_gate`` MCP
    tool is exactly that shape). Such a caller holds ``(expected_version,
    expected_generation)`` from the earlier read and pulls a verdict here
    immediately before dispatching.

    Returns None when the effect may proceed; raises otherwise -- including when
    either comparand is UNCONFIRMED, since firing on something the coordinator
    never confirmed is precisely what this layer exists to prevent, and when
    the re-read itself was served WITHOUT a standing grant (a stale-status
    re-grant), since a peer's write-claim preemption moves neither comparand. Same honest
    boundary as :func:`gate`: the verdict is true as of THIS check, and the
    caller's dispatch still follows it.

    A volume that cannot report the grant state of its reads is refused before
    the re-validate read even runs (:func:`_require_grant_state`) -- typed
    inside the coherence hierarchy, so a caller that catches a HOLD catches
    this too rather than crashing on a bare ``AttributeError``.
    """

    # PRECONDITION, checked before the re-validate read rather than around it:
    # the third leg is built from answers only the volume can give, so a volume
    # that cannot give them fails LOUD here instead of gating on a fabricated
    # "nothing was wrong". Unconditional, so no read path can slip past it.
    _require_grant_state(volume)

    current_version: int | None
    current_generation: int | None
    try:
        # observe=False: this read exists only to COMPARE comparands and its
        # bytes are discarded, so it must not advance the volume's foreign-edit
        # baseline. Otherwise checking freshness would quietly absolve an
        # out-of-band edit the caller never saw, and the caller's next write —
        # which the foreign-edit guard would have denied — would clobber it.
        _, current_version, current_generation = volume.read_with_version_generation(
            path, observe=False
        )
        # Read DIRECTLY, no default: _require_grant_state() above proved both
        # flags exist, and a defaulting accessor is what silently admitted a
        # volume that answers neither question.
        denied = bool(volume._last_read_denied)
        # The third leg of the fence: the pair answers "did the value or the
        # epoch move", but a peer's pessimistic write-acquire ends the caller's
        # grant while moving NEITHER (no commit yet, and trigger="write" is
        # outside EPOCH_BUMP_TRIGGERS -- the epoch is per-artifact and cannot
        # even see an S-holder's preemption). The re-validate read itself
        # carries the answer: a stale-status response (warn re-grant or deny)
        # means the grant the decision was read under did NOT stand at this
        # check. A volume that does not report it never reaches here -- the
        # pair-only fallback that allowance produced WAS this leg's absence.
        lapsed = bool(volume._last_read_stale)
    except FileNotFoundError:
        # The input is GONE: the vanish sentinels, and no read to have been
        # refused or served grantless (the stale ``_last_read_*`` flags describe
        # some EARLIER read, never this one). Reported as observations, not as a
        # verdict -- classify_hold still names the HOLD.
        current_version = None
        current_generation = None
        denied = False
        lapsed = False

    # ONE classification, shared with the coordinator route: HOLD unless the
    # coordinator CONFIRMED an unchanged (version, generation) pair under a
    # standing grant. Version 0 is the "could not resolve" sentinel (an
    # older/degraded coordinator, or a degrade-mode volume whose read did not
    # fail closed); generation None is its sibling (an older coordinator that
    # predates this release's generation reporting, a strict-mode deny, or a
    # degraded read). Firing on either would act on input the coordinator never
    # confirmed. Treating an unconfirmed comparand as a HOLD keeps the gate
    # fail-closed by construction, independent of the volume's on_error mode --
    # and it is why this wrapper against a pre-fence coordinator HOLDs loudly
    # instead of silently reverting to the generation-blind check.
    comparands = FenceComparands(
        expected_version=expected_version,
        current_version=current_version,
        expected_generation=expected_generation,
        current_generation=current_generation,
        read_refused=denied,
        grant_did_not_stand=lapsed,
        # HONEST LIMIT, in-process: the pre-read wire carries ``hash_differs``
        # (a comparison) and never the coordinator's RECORDED hash, so this
        # client cannot distinguish "the claim matches" from "there is no claim
        # at all" -- both arrive as hash_differs=False. Passing True preserves
        # the shipped verdict rather than HOLDing every gate on a blind spot;
        # the coordinator route, which reads the recorded hash straight off the
        # registry, calls ``coordinator_holds_content_claim`` and passes the
        # real answer. Closing this leg in-process needs an additive wire field,
        # not a client-side guess.
        content_claim_present=True,
    )
    reason = classify_hold(comparands)
    if reason is not None:
        raise _held(path, comparands, reason)

