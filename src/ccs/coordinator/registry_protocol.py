# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Structural Protocols for the coordinator's artifact registries.

This module holds the registry CONTRACT the service layer depends on, extracted
as a pure refactor (zero behavior change). Before this extraction the two
registries (:class:`ccs.coordinator.registry.ArtifactRegistry` in-memory and
:class:`ccs.coordinator.sqlite_registry.SqliteArtifactRegistry` durable) were
duck-typed with NO shared interface; their parity was asserted piecemeal in
tests.

The two Protocols below name that shared surface explicitly (the authoritative
method set is the one pinned by ``tests/test_registry_protocol_parity.py``):

- :class:`RegistryBase` — the methods :class:`CoordinatorService` (the service
  layer) depends on. ``ArtifactRegistry`` and ``SqliteArtifactRegistry`` both
  satisfy it.
- :class:`SqliteExtended` — ``RegistryBase`` plus the SQLite-backed methods
  ``coordinator_server.py`` depends on (preemption notices, prefix lookups,
  ``resolve_or_register``, ``status_snapshot``, connection ``close``). Only
  ``SqliteArtifactRegistry`` satisfies it today.

Both are :func:`~typing.runtime_checkable` so ``isinstance`` (structural
presence-of-methods only) and the parity test can verify conformance. The
registries do NOT inherit these Protocols at runtime — conformance is structural,
backed by a ``TYPE_CHECKING``-guarded static assertion in each registry module
and by ``tests/test_registry_protocol_parity.py``.

To avoid an import cycle, this module imports ONLY domain types — never the
registry classes themselves (the registries import this module's Protocols under
``TYPE_CHECKING`` for the conformance assertion).
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from threading import Event
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeAlias,
    runtime_checkable,
)
from uuid import UUID

from ccs.core.exceptions import (
    HANDOFF_ENDED_REASON,
    HANDOFF_IN_FLIGHT_REASON,
    HANDOFF_NOT_HELD_REASON,
    HANDOFF_OTHER_HOLDER_REASON,
    HANDOFF_SELF_REASON,
    HANDOFF_SUCCESSOR_UNKNOWN_REASON,
    HANDOFF_VERSION_UNCONFIRMED_REASON,
)
from ccs.core.states import MESIState, TransientState
from ccs.core.types import (
    TRANSFER_STATUS_DECLINED,
    TRANSFER_STATUS_PENDING,
    TRANSFER_STATUS_SUPERSEDED,
    TRANSFER_STATUS_WITHDRAWN,
    TRANSFER_STATUSES,
    Artifact,
    CasCorruption,
    CommitAllEntry,
    ConflictDetail,
    MultiCommitConflict,
    MultiCommitResult,
    TransferGrantOutcome,
    VersionedReadRejection,
)

from .retention import RetentionPolicy

# Contract return types — DEFINED here in the contract module and re-exported by
# the concrete registries (registry.py / sqlite_registry.py import them back).
# They ARE part of the contract (the return types of its methods); this module
# deduplicated them here (each registry previously defined its own identical copy).
ReclamationSlot: TypeAlias = tuple[str, int]  # (trigger, tick)
# WIN = (updated_artifact, invalidated_agent_ids); loss = ConflictDetail;
# impossible state = CasCorruption. None is raised by the registry.
CasResult: TypeAlias = "tuple[Artifact, list[UUID]] | ConflictDetail | CasCorruption"
# Atomic multi-artifact publish (SB-18 / commit_all): WIN = MultiCommitResult
# (per-member new versions + the aggregated invalidated set); any member blocked =
# MultiCommitConflict (per-member ConflictDetail); any member corrupt = CasCorruption.
# All-or-nothing — never a partial batch. None is raised by the registry.
MultiCasResult: TypeAlias = "MultiCommitResult | MultiCommitConflict | CasCorruption"
# Snapshot consistent-cut capture: WIN = the pinned cut
# {artifact_id: version}; a read_set with an unknown id = VersionedReadRejection,
# NO pins inserted. Neither is raised by the registry.
CaptureResult: TypeAlias = "dict[UUID, int] | VersionedReadRejection"


# Foreign-write detection outcomes (detection-substrate plan R5). Exactly three
# classifications and nothing finer: a disk content the coordinator holds
# (``mediated``), one it does not and cannot explain (``foreign``), and one it
# declines to call foreign because a recent mediated commit to that artifact is
# witnessed inside the benign commit-to-disk lag window (``lag_suppressed``, R13
# — its own reported outcome so an operator can size the window's admitted
# false-negative, never folded into ``mediated``).
FOREIGN_WRITE_OUTCOMES: tuple[str, str, str] = ("foreign", "mediated", "lag_suppressed")


@dataclass(frozen=True)
class DetectionRun:
    """One coordinator run's observed detection interval (plan R8, R12 / KTD6).

    A cumulative tick count cannot tell a reader whether the detector observed a
    PARTICULAR span — a coordinator down for the middle of a session still
    presents a large count and a recent last tick. So each run records its own
    interval, and a coverage question is answered by intersecting the session's
    span with these intervals rather than by trusting a total.

    ``run_id`` is opaque and per-writer-open. An empty run list is the
    not-instrumented signal: the tables exist on every writer open whether or
    not the sweep thread was created, so the TABLE cannot carry that meaning and
    the ROW must.
    """

    run_id: str
    first_tick_unix: float
    last_tick_unix: float
    tick_count: int
    covered_count: int = 0
    """How many artifacts were in scope during this interval. A run that
    watched nothing and a run that watched five hundred and found nothing are
    different answers to "was this period clean?", and a tick count alone
    cannot tell them apart."""


@dataclass(frozen=True)
class UncoverableRun:
    """One coordinator run that had nothing it could ever watch.

    Deliberately NOT a :class:`DetectionRun` with a zero tick count. A detection
    run is an interval a coverage claim may be checked against; this is the
    absence of any such interval, plus the reason. Folding it into the same type
    would put a row that observed nothing into the sequence
    ``ForeignWriteReport.covers`` walks, and a workspace nothing ever polled
    would start answering coverage questions.

    ``reason`` is an opaque stable token, not a message: it is written by the
    detector and read by the offline report across a process and a release
    boundary, which is the one place a human-readable string turns into a
    parsing hazard.
    """

    run_id: str
    reason: str
    observed_at_unix: float


@dataclass(frozen=True)
class CheckpointRecord:
    """One workspace-checkpoint manifest header (WV plan Unit 2 / R1, R9).

    The contract row both registries store and return — DEFINED here (the
    contract module) like the other contract return types. Frozen: a manifest
    record is a fact; updates go through the targeted registry mutators
    (:meth:`RegistryBase.set_checkpoint_restore_status` /
    :meth:`RegistryBase.adjust_checkpoint_pin_refcount`), never in-place edits.

    ``owner`` is REQUIRED metadata (fail-closed: ``create_checkpoint`` raises on
    an absent owner — an ownerless manifest is unrepresentable). ``window_min`` /
    ``window_max`` are the skew-declared cut window's endpoints (monotonic
    seconds as captured by the engine). ``restore_status`` +
    ``restore_updated_at`` are the checkpoint-level restore-progress fields
    (durable because restore is crash-resumable); the vocabulary is the service
    layer's — the registry stores the string. ``pin_refcount`` is the Unit-6 GC
    pin bookkeeping (never negative; adjusted only through the registry).
    """

    checkpoint_id: str
    name: str
    owner: UUID
    created_at: float
    created_at_tick: int
    window_min: float
    window_max: float
    restore_status: str = "none"
    restore_updated_at: float | None = None
    pin_refcount: int = 0


@dataclass(frozen=True)
class CheckpointMember:
    """One member row of a workspace-checkpoint manifest (WV plan Unit 2 / R1).

    Fixed-width capture facts only — tokens, fingerprints, flags, tiers,
    timestamps; NEVER content bytes. ``member_path`` keys the member within its
    checkpoint; ``artifact_id`` is the optional coordinator artifact ref (an S3
    member has none). ``native_token`` is the member's substrate CAS token as
    captured (opaque here). ``absent`` records absent-at-capture (ABSENT is a
    fact distinct from empty); ``dirty_during_window`` is the torn-cut flag.
    ``arbitration_tier`` (``native-cas`` / ``no-arbiter``) and ``restore_tier``
    (``restorable`` / ``restorable-unpinned`` / ``forward_only``) default to the
    WEAKEST claims — honesty is the default, upgrades are explicit.
    ``restore_outcome`` + ``deleted_at_restore`` are the per-member
    restore-progress columns: the typed terminal outcome a crash-resumable
    restore records, and the manifest-side delete record (``commit_all`` has no
    delete semantics, so delete legs are recorded here per the plan's
    restore-registration design).
    """

    member_path: str
    artifact_id: UUID | None
    native_token: str | None
    fingerprint: str | None
    captured_at: float
    absent: bool = False
    dirty_during_window: bool = False
    arbitration_tier: str = "no-arbiter"
    restore_tier: str = "forward_only"
    pin_state: str = "unpinned"
    restore_outcome: str | None = None
    deleted_at_restore: float | None = None

# Shared registry trigger constants — DEFINED here (canonical) and re-exported by
# both registries, which previously each kept an identical copy pinned equal by
# the dual-registry parity test.
#
# RECLAIM_TRIGGERS: the coordinator-side EVICTION triggers (the stable-grant
# sweep's reclaim_heartbeat / reclaim_max_hold + the transient-timeout fail-safe
# "timeout"). A SUBSET of the bump-gating set: an M/E -> INVALID transition
# carrying one of these bumps the artifact's owner_generation (the
# read-generation fence), because the claim was revoked WITHOUT a version move,
# which version-CAS cannot see. EPOCH_BUMP_TRIGGERS immediately below is the
# full bump set -- it adds the voluntary "invalidate" release and the
# HANDOFF_TRIGGER, which each end a claim at an unchanged version the same way
# without being a reclaim. The peer-invalidation triggers
# never bump, for different reasons each: "commit" moves the version, so
# version-CAS already catches a stale write; "write" (a peer's pessimistic
# acquire) moves NOTHING, and is kept out anyway -- see the note under
# EPOCH_BUMP_TRIGGERS for where that revocation IS fenced.
RECLAIM_TRIGGERS: frozenset[str] = frozenset(
    {"reclaim_heartbeat", "reclaim_max_hold", "timeout"}
)
# HANDOFF_TRIGGER: the trigger a transfer moves its giver INVALID under (the
# targeted grant handoff, #185). Its own value, distinct from "invalidate", so
# the state log records a deliberate handoff as a handoff rather than as a
# release, and deliberately NOT a RECLAIM_TRIGGER: a handoff is the holder's own
# act, never an eviction. Wire-stable: renaming it without EPOCH_BUMP_TRIGGERS
# leaves the bump keyed on a string nothing emits.
HANDOFF_TRIGGER: str = "handoff"
# EPOCH_BUMP_TRIGGERS: the triggers whose M/E -> INVALID transition MOVES the
# ownership epoch. The rule is not "the sweep did it" but "a write-claim was
# revoked WITHOUT the version moving" -- the one condition version-CAS is
# structurally blind to. That is every RECLAIM_TRIGGER plus "invalidate", the
# voluntary release (service.invalidate: a post-edit failure, a session-stop
# release, an operator drain), which likewise ends a claim at an unchanged
# version. Leaving it out let a release SUPPRESS the fence: the holder is
# already INVALID, so the sweep has no M/E grant left to reclaim and the epoch
# never moves at all -- the identical end-state a sweep reclaim fences was
# silently admitted (NoSilentRevoke, formal/tla/Fencing.tla).
#
# HANDOFF_TRIGGER joins for the same reason: a giver that hands a path off from
# EXCLUSIVE or MODIFIED ends its write claim at an unchanged version exactly as
# a release does, so its late commit must meet the fence. Because the bump keys
# on leaving M/E, a giver handing off a standing SHARED read never moves the
# epoch -- it revoked no write claim, and a bump there would fence every
# bystander that read the path as though a writer had been reclaimed.
#
# The peer-invalidation triggers ("write" / "commit") stay OUT, unchanged.
# "commit" moves the version, so version-CAS already arbitrates a stale write.
# "write" -- a peer's pessimistic acquire -- revokes the holder's claim while
# moving NEITHER the version (no commit yet) NOR, per this exclusion, the
# epoch. It stays out because the per-ARTIFACT epoch is the wrong fence for
# preemption: bumping here would fence every bystander's read-generation at
# the next CAS for a revocation that version-CAS will arbitrate anyway once
# the acquirer commits, and it still could not cover a SHARED holder's
# preemption (bumps key on M/E revocations only). The preempted holder is
# fenced elsewhere, per operation class: its commit is refused by the M/E
# state check + version-CAS, and its ESCAPING EFFECT -- which has no commit to
# arbitrate -- is HELD by the effect gate's standing-grant re-check
# (adapters.effect_gate.check_fence: a stale-status re-validate read HOLDs
# even with the (version, generation) pair unchanged).
EPOCH_BUMP_TRIGGERS: frozenset[str] = RECLAIM_TRIGGERS | frozenset(
    {"invalidate", HANDOFF_TRIGGER}
)
# CLAIM_CAPTURE_TRIGGERS: triggers under which a transition MAY be a genuine
# content read for read-generation capture (the E/M-acquire capture is keyed
# on the state transition, not the trigger). Membership is necessary, not
# sufficient: service.fetch() emits "fetch" on two legs -- the requester's own
# read (I/S -> S/E) and the downgrade of an M/E peer to SHARED -- and only the
# first is a claim. The capture site therefore also requires that the agent is
# not leaving M/E (and is not being granted INVALID); a peer's downgrade never
# refreshes the ex-holder's captured generation. Renaming the trigger without
# updating this would silently disable capture on reads.
CLAIM_CAPTURE_TRIGGERS: frozenset[str] = frozenset({"fetch"})


# ---------------------------------------------------------------------------
# The transfer record (targeted grant handoff, #185)
# ---------------------------------------------------------------------------
#
# One record per artifact, kept by both registries and exposed through four
# base members: ``transfer_grants`` (the composite transfer), the read
# ``get_transfer_record``, the status write ``set_transfer_status`` and the
# eviction ``evict_transfer_records``. The decision order and the liveness
# predicate below are shared by both registries, which share no base class:
# each gathers a path's facts in its own critical section (one lock hold, or one
# ``BEGIN IMMEDIATE``) and decides with the same function, so the two backends
# cannot answer one request differently.

# Why a record exists, as a closed vocabulary (stored; add, never rename). A
# plain transfer is a ``handoff``; the giver's re-transfer of its own live
# record is a ``supersession``, and that record also names the successor it
# superseded, so a late re-send of the superseded tuple is recognised
# rather than read as a fresh supersession. The cause is its own column, never
# an id encoded into a string, so a later cause (#195's reclaim causes) adds a
# value without re-parsing the stored ones.
TRANSFER_CAUSE_HANDOFF = "handoff"
TRANSFER_CAUSE_SUPERSESSION = "supersession"
TRANSFER_CAUSES: frozenset[str] = frozenset(
    {TRANSFER_CAUSE_HANDOFF, TRANSFER_CAUSE_SUPERSESSION}
)

# The statuses that end a record whatever its version. Every other label
# leaves a record live while the artifact's version equals the version at
# transfer -- an overtaken record included.
TRANSFER_ENDED_STATUSES: frozenset[str] = frozenset(
    {TRANSFER_STATUS_DECLINED, TRANSFER_STATUS_WITHDRAWN}
)

# The statuses a stored record may carry. ``superseded`` never persists: the
# replacing record's cause carries it, and a stored superseded row would read as
# live (it is neither declined nor withdrawn) and fence its giver for nothing.
TRANSFER_STORED_STATUSES: frozenset[str] = TRANSFER_STATUSES - {
    TRANSFER_STATUS_SUPERSEDED
}


def require_storable_transfer_status(status: str) -> None:
    """Raise ``ValueError`` for a status no stored record may carry (anything
    outside :data:`TRANSFER_STORED_STATUSES`, ``superseded`` included). Both
    registries' ``set_transfer_status`` call it before touching any state, so
    the closed vocabulary is checked, and its refusal worded, in one place."""
    if status not in TRANSFER_STORED_STATUSES:
        raise ValueError(
            f"transfer status {status!r} cannot be stored; expected one of "
            f"{sorted(TRANSFER_STORED_STATUSES)}"
        )


# A claim the presented composite can hand on: a write grant or a standing read.
_HELD_STATES: frozenset[MESIState] = frozenset(
    {MESIState.EXCLUSIVE, MESIState.MODIFIED, MESIState.SHARED}
)


@dataclass(frozen=True, kw_only=True)
class TransferRecord:
    """One path's transfer record, as both registries store and return it.

    ``kw_only`` because four fields are ids of one type side by side; a
    positional swap would record the handoff pointed the wrong way.

    - ``giver`` -- the caller's session-level identity: what the fence keys on,
      so a re-minted incarnation or a subagent of the same session is fenced
      alike.
    - ``holder`` -- the composite that held the claim and moved INVALID.
    - ``successor`` -- the successor's session-level identity.
    - ``version_at_transfer`` -- the artifact version the handoff is fenced on.
    - ``hold_shape`` -- the grant given up: EXCLUSIVE, MODIFIED or SHARED.
    - ``cause`` -- one of :data:`TRANSFER_CAUSES`; ``superseded_successor`` is
      set exactly when the cause is a supersession.
    - ``status`` -- one of :data:`TRANSFER_STORED_STATUSES`. A label, not
      liveness: the read answers liveness beside the record.
    - ``counterparty`` -- the bystander an overtaken label names.
    - ``created_at`` / ``updated_at`` -- wall-clock unix seconds, since the
      coordinator's ticks reset on a restart.
    """

    artifact_id: UUID
    giver: UUID
    holder: UUID
    successor: UUID
    version_at_transfer: int
    hold_shape: MESIState
    cause: str
    superseded_successor: UUID | None
    status: str
    counterparty: UUID | None
    created_at: float
    updated_at: float

    def __post_init__(self) -> None:
        # Checked on every construction, so a record built for a write and the
        # sqlite row read back into one are both held to the closed vocabulary.
        if self.cause not in TRANSFER_CAUSES:
            raise ValueError(
                f"transfer cause {self.cause!r} is not one of {sorted(TRANSFER_CAUSES)}"
            )


@dataclass(frozen=True, kw_only=True)
class TransferRequest:
    """One transfer request, as :meth:`RegistryBase.transfer_grants` decides it.

    - ``giver`` -- the caller's session-level identity. The registry never
      derives it: the route passes the identity it derived from the session id,
      a library caller its own agent id.
    - ``successor`` -- the successor's session-level identity, already
      normalised by the caller (the registry never resolves a composite).
    - ``holders`` -- artifact id to the composite presented as holding the
      claim on that path (a volume presents the incarnation that holds
      its read or write there). Keyed by artifact, so a path is decided once.
    - ``successor_known`` -- whether the caller already resolved the successor
      through a name map the registry cannot see (the HTTP route's live map).
      The registry also counts a bound principal and a grant row as known.
    """

    giver: UUID
    successor: UUID
    holders: Mapping[UUID, UUID]
    successor_known: bool = False


def transfer_record_live(record: TransferRecord, current_version: int) -> bool:
    """A record's liveness, the one predicate both registries read it from.

    Live while the artifact's version still equals the version at transfer and
    the record was neither declined nor withdrawn, whatever its label otherwise
    says. The version leg is what a status-only reading misses: a version can
    move with no status write (a crash, or a failed label write, between a win
    and its completion), and such a record must read as not live everywhere at
    once. Each registry supplies ``current_version`` from inside the same hold
    or transaction that read the record."""
    return (
        current_version == record.version_at_transfer
        and record.status not in TRANSFER_ENDED_STATUSES
    )


@dataclass(frozen=True, kw_only=True)
class TransferPathView:
    """What one path looks like to the composite transfer, read by a registry
    inside the hold or transaction that will also apply the decision.

    ``current_version`` is None when the artifact is absent; ``record`` and
    ``live`` come from the registry's liveness helper; ``holder_state`` is the
    presented composite's grant (None when it has no row);
    ``other_write_holder`` is whether any OTHER agent holds the path EXCLUSIVE
    or MODIFIED."""

    artifact_id: UUID
    holder: UUID
    current_version: int | None
    record: TransferRecord | None
    live: bool
    holder_state: MESIState | None
    other_write_holder: bool


@dataclass(frozen=True)
class TransferDecision:
    """One path's decision: the per-grant answer, the record to store (None:
    nothing is written), and whether the presented composite moves INVALID
    under the handoff trigger (a supersession writes the record alone)."""

    outcome: TransferGrantOutcome
    write: TransferRecord | None = None
    move_holder: bool = False


def decide_transfer_grant(
    request: TransferRequest,
    view: TransferPathView,
    *,
    successor_known: bool,
    now_unix: float,
) -> TransferDecision:
    """Decide one path of a transfer in the order below. Pure: it reads the view
    and writes nothing, so a registry decides EVERY path before it applies any.

    First the path's record is matched against the caller as giver:

    - (a) live, same successor: answered transferred with the record's status;
      nothing moves, so a re-send writes no second record.
    - (b) live, the successor this record's supersession replaced: refused as
      ended with the superseded status, so a late re-send never supersedes back.
    - (c) live, any other successor: supersedes. The stored holder stands in for
      the hold check only; the foreign write-holder, self and unknown refusals
      still run, and a refused supersession changes nothing.
    - (d) ended by decline or withdraw, same successor, unmoved version: refused
      as ended with that status.
    - (e) any other record naming the caller is treated as absent.

    Then, with no record naming the caller, the ordinary checks run on the
    presented composite, in order: hold, unconfirmed version, foreign write
    holder (not held already won), another giver's live record, self, unknown.
    """
    record = view.record
    if record is not None and record.giver == request.giver:
        decided = _decide_on_own_record(
            request, view, record, successor_known=successor_known, now_unix=now_unix
        )
        if decided is not None:
            return decided
        record = None  # arm (e)
    return _decide_ordinary(
        request, view, record, successor_known=successor_known, now_unix=now_unix
    )


def _decide_on_own_record(
    request: TransferRequest,
    view: TransferPathView,
    record: TransferRecord,
    *,
    successor_known: bool,
    now_unix: float,
) -> TransferDecision | None:
    """Arms (a) to (d); None is arm (e)."""
    if view.live:
        if record.successor == request.successor:
            return TransferDecision(_transferred(record))
        if record.superseded_successor == request.successor:
            return TransferDecision(
                _ended(record, successor=request.successor, status=TRANSFER_STATUS_SUPERSEDED)
            )
        return _supersede(request, view, record, successor_known=successor_known, now_unix=now_unix)
    if (
        record.status in TRANSFER_ENDED_STATUSES
        and record.successor == request.successor
        and view.current_version == record.version_at_transfer
    ):
        return TransferDecision(
            _ended(record, successor=record.successor, status=record.status)
        )
    return None


def _supersede(
    request: TransferRequest,
    view: TransferPathView,
    record: TransferRecord,
    *,
    successor_known: bool,
    now_unix: float,
) -> TransferDecision:
    """Arm (c). No INVALID move: the stored holder's row moved at the first
    transfer, and the hold shape and version at transfer carry over."""
    refusal = _write_holder_refusal(view) or _successor_refusal(
        request, view, successor_known
    )
    if refusal is not None:
        return TransferDecision(refusal)
    write = TransferRecord(
        artifact_id=view.artifact_id,
        giver=request.giver,
        holder=record.holder,
        successor=request.successor,
        version_at_transfer=record.version_at_transfer,
        hold_shape=record.hold_shape,
        cause=TRANSFER_CAUSE_SUPERSESSION,
        superseded_successor=record.successor,
        status=TRANSFER_STATUS_PENDING,
        counterparty=None,
        created_at=now_unix,
        updated_at=now_unix,
    )
    return TransferDecision(_transferred(write), write=write)


def _decide_ordinary(
    request: TransferRequest,
    view: TransferPathView,
    record: TransferRecord | None,
    *,
    successor_known: bool,
    now_unix: float,
) -> TransferDecision:
    """The checks on the presented composite, then a plain handoff with the
    INVALID move; an ended record on the path is replaced."""
    holder_state, version = view.holder_state, view.current_version
    if holder_state is None or holder_state not in _HELD_STATES:
        return TransferDecision(_refused(view, HANDOFF_NOT_HELD_REASON))
    if version is None or version < 1:
        return TransferDecision(_refused(view, HANDOFF_VERSION_UNCONFIRMED_REASON))
    refusal = _write_holder_refusal(view)
    if refusal is None and record is not None and view.live:
        refusal = _refused(
            view, HANDOFF_IN_FLIGHT_REASON, giver=record.giver, successor=record.successor
        )
    refusal = refusal or _successor_refusal(request, view, successor_known)
    if refusal is not None:
        return TransferDecision(refusal)
    write = TransferRecord(
        artifact_id=view.artifact_id,
        giver=request.giver,
        holder=view.holder,
        successor=request.successor,
        version_at_transfer=version,
        hold_shape=holder_state,
        cause=TRANSFER_CAUSE_HANDOFF,
        superseded_successor=None,
        status=TRANSFER_STATUS_PENDING,
        counterparty=None,
        created_at=now_unix,
        updated_at=now_unix,
    )
    return TransferDecision(_transferred(write), write=write, move_holder=True)


def _write_holder_refusal(view: TransferPathView) -> TransferGrantOutcome | None:
    if view.other_write_holder:
        return _refused(view, HANDOFF_OTHER_HOLDER_REASON)
    return None


def _successor_refusal(
    request: TransferRequest, view: TransferPathView, successor_known: bool
) -> TransferGrantOutcome | None:
    """Self, then unknown. Self compares session-level identities only: the
    caller normalised both, and the registry never derives one."""
    if request.successor == request.giver:
        return _refused(view, HANDOFF_SELF_REASON)
    if not successor_known:
        return _refused(view, HANDOFF_SUCCESSOR_UNKNOWN_REASON)
    return None


def _refused(
    view: TransferPathView,
    reason: str,
    *,
    giver: UUID | None = None,
    successor: UUID | None = None,
) -> TransferGrantOutcome:
    return TransferGrantOutcome(
        artifact_id=view.artifact_id,
        transferred=False,
        reason=reason,
        giver=giver,
        successor=successor,
    )


def _transferred(record: TransferRecord) -> TransferGrantOutcome:
    return TransferGrantOutcome(
        artifact_id=record.artifact_id,
        transferred=True,
        giver=record.giver,
        successor=record.successor,
        version_at_transfer=record.version_at_transfer,
        hold_shape=record.hold_shape,
        status=record.status,
    )


def _ended(record: TransferRecord, *, successor: UUID, status: str) -> TransferGrantOutcome:
    return TransferGrantOutcome(
        artifact_id=record.artifact_id,
        transferred=False,
        reason=HANDOFF_ENDED_REASON,
        giver=record.giver,
        successor=successor,
        version_at_transfer=record.version_at_transfer,
        hold_shape=record.hold_shape,
        status=status,
    )


@runtime_checkable
class RegistryBase(Protocol):
    """The registry contract the service layer (:class:`CoordinatorService`)
    depends on — the methods shared by both the in-memory and SQLite-backed
    registries.

    Extracted as a pure refactor (no behavior change): it names the
    previously-implicit duck-type both registries already satisfied. The
    in-memory :class:`~ccs.coordinator.registry.ArtifactRegistry` is the
    canonical shape; the durable
    :class:`~ccs.coordinator.sqlite_registry.SqliteArtifactRegistry` mirrors it.

    Note on :meth:`get_content`: the in-memory registry returns ``Optional[str]``
    while the SQLite registry returns ``Optional[bytes]`` (it returns ``b""`` for
    known artifacts). The honest union return type is therefore
    ``str | bytes | None``.
    """

    def abort_guard(self, abort: "Event | None" = None) -> AbstractContextManager[None]:
        ...

    def adjust_checkpoint_pin_refcount(self, checkpoint_id: str, delta: int) -> int:
        """Atomically add ``delta`` to a checkpoint's pin refcount and return the
        new value. Raises ``KeyError`` for an unknown checkpoint and
        ``ValueError`` if the result would go negative (a release without a
        matching pin is a bookkeeping bug, fail-closed)."""
        ...

    def all_session_meta(self) -> "dict[str, tuple[UUID, int]]":
        ...

    def artifact_ids(self) -> list[UUID]:
        ...

    def bind_caller_principal(
        self, identity: UUID, principal: str, mint_nonce: str
    ) -> tuple[str, str]:
        """First-claim-wins bind of a caller principal to ``identity``: insert
        ``(principal, mint_nonce)`` only if ``identity`` has no binding yet, and
        return the BOUND pair either way — the caller's own on a first claim,
        the earlier claimant's otherwise. Never rebinds. One atomic step, so two
        concurrent first claims bind ONE principal. Deliberately its own store,
        never the session-meta one: the session sweep, the session cap and
        session release must not see it (a principal outlives both a grant and a
        snapshot session)."""
        ...

    def capture_version_vector(
        self,
        read_set: "Iterable[UUID]",
        session_token: str,
        *,
        owner: "UUID | None" = None,
        created_at_tick: int | None = None,
    ) -> CaptureResult:
        ...

    def clear_agent_transient(self, artifact_id: UUID, agent_id: UUID) -> None:
        ...

    def commit_cas(
        self,
        artifact_id: UUID,
        agent_id: UUID,
        *,
        expected_version: int,
        content_hash: str,
        size_tokens: int | None = None,
        content: bytes | str | None = None,
        tick: int = 0,
        trigger: str = "commit_cas",
    ) -> CasResult:
        ...

    def commit_all(
        self,
        agent_id: UUID,
        writes: Mapping[UUID, CommitAllEntry],
        *,
        tick: int = 0,
        trigger: str = "commit_all",
    ) -> MultiCasResult:
        """Atomic multi-artifact publish (SB-18 / commit_all): commit ``writes``
        all-or-nothing — every member advances to its next version or none do and
        the batch is HELD. A genuinely new atomic multi-row op, never a loop of
        :meth:`commit_cas`. Implemented on BOTH backends with identical outcomes
        (parity)."""
        ...

    def create_checkpoint(
        self,
        checkpoint: CheckpointRecord,
        members: Sequence[CheckpointMember],
    ) -> None:
        """Persist a checkpoint manifest — the header row (owner metadata
        INCLUDED, same transaction) plus every member row — atomically: all rows
        land or none do. Raises ``ValueError`` on an absent owner (fail-closed:
        an ownerless manifest is never persisted), on a duplicate
        ``checkpoint_id``, and on duplicate member paths within the manifest."""
        ...

    def evict_transfer_records(
        self, *, max_age_sec: float, now_unix: float | None = None
    ) -> int:
        """Delete every transfer record that is NOT live and older than
        ``max_age_sec``, and return how many went. Liveness is the read's
        (:func:`transfer_record_live`), so a live record is never evicted
        whatever its age. The age runs from the record's
        ``updated_at`` -- on sqlite from the later of that and the artifact's
        own last update, so a version-move ending survives until the giver's
        next touch can report it. ``now_unix`` defaults to the wall clock."""
        ...

    @property
    def coordinator_epoch(self) -> str:
        """Fence token identifying this coordinator incarnation. A ``@property``
        on both registries; read by :class:`CoordinatorService` on the read-fence
        and session paths (``read_at_version`` / ``begin_session`` /
        ``session_read`` / ``session_commit``). Declared here so a backend typed
        against ``RegistryBase`` cannot omit it and pass ``isinstance`` yet fail
        at the first fence read."""
        ...

    def get_agent_state(self, artifact_id: UUID, agent_id: UUID) -> MESIState | None:
        ...

    def get_agent_transient(self, artifact_id: UUID, agent_id: UUID) -> TransientState | None:
        ...

    def get_artifact(self, artifact_id: UUID) -> Optional[Artifact]:
        ...

    def get_caller_principal(self, identity: UUID) -> str | None:
        """Return the caller principal bound to ``identity``, or ``None`` when
        the identity is unclaimed. The durable tier the service's validator
        falls back to when its in-process cache misses."""
        ...

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        ...

    def get_checkpoint_members(self, checkpoint_id: str) -> list[CheckpointMember]:
        """Return the manifest's member rows ordered by ``member_path`` (empty
        list for an unknown checkpoint — the header getter tells known from
        unknown)."""
        ...

    def get_content(self, artifact_id: UUID) -> str | bytes | None:
        ...

    def get_content_at_version(self, artifact_id: UUID, version: int) -> str | bytes | None:
        ...

    def get_last_reclamation(
        self, agent_id: UUID, artifact_id: UUID
    ) -> ReclamationSlot | None:
        ...

    def get_owner_generation(self, artifact_id: UUID) -> int:
        ...

    def get_read_generation(self, artifact_id: UUID, agent_id: UUID) -> int | None:
        ...

    def get_session_cut(self, session_token: str) -> dict[UUID, int] | None:
        ...

    def get_session_meta(self, session_token: str) -> "tuple[UUID, int] | None":
        ...

    def get_state_map(self, artifact_id: UUID) -> dict[UUID, MESIState]:
        ...

    def get_transfer_record(
        self, artifact_id: UUID
    ) -> "tuple[TransferRecord, bool] | None":
        """Return the artifact's transfer record together with whether it is
        live, or None when the path has no record. Both halves come from ONE
        read under the registry lock (sqlite: one SELECT joining the
        artifact's version), so a concurrent version move cannot tear them:
        this is the one place the service reads liveness from. A
        plain read, so it serves a read-only open too."""
        ...

    def get_transient_map(self, artifact_id: UUID) -> dict[UUID, TransientState]:
        ...

    def get_transient_tick(self, artifact_id: UUID, agent_id: UUID) -> int | None:
        ...

    def get_artifact_and_generation(
        self, artifact_id: UUID
    ) -> "tuple[Artifact, int] | None":
        """Return ``(artifact, owner_generation)`` from ONE snapshot, or None if
        the artifact is absent. The pair MUST have coexisted at a single
        instant: a backend serving it as two independent reads lets a concurrent
        sweep reclamation (which bumps the generation WITHOUT a version move)
        tear the pair, silently reopening the reclaim-zombie EFFECT hole
        downstream (see ``adapters.effect_gate``). Any caller needing a
        version and its ownership epoch together must use this, never two
        separate accessors.
        """
        ...

    def get_version_record(
        self, artifact_id: UUID, version: int
    ) -> tuple[str | bytes, float] | None:
        ...

    def granted_at_tick(self, agent_id: UUID, artifact_id: UUID) -> int | None:
        ...

    def has_artifact(self, artifact_id: UUID) -> bool:
        ...

    def last_heartbeat_tick(self, agent_id: UUID) -> int | None:
        ...

    def last_observed_version_for(self, artifact_id: UUID, agent_id: UUID) -> int | None:
        """Return the artifact version whose bytes this agent last observed
        (SB-10: recorded atomically with every non-INVALID grant/commit upsert),
        or None when the pair was never observed. Absence semantics are part of
        the contract: never a 0-sentinel, and a transition to INVALID preserves
        the prior recorded value — this is the durable comparand the
        post-compaction stale flag is computed from."""
        ...

    def list_checkpoints(self) -> list[CheckpointRecord]:
        """Return every checkpoint header, ordered by ``(created_at,
        checkpoint_id)`` (deterministic for the CLI ``list`` verb)."""
        ...

    def record_heartbeat(self, agent_id: UUID, now_tick: int) -> None:
        ...

    def record_last_reclamation(
        self, agent_id: UUID, artifact_id: UUID, trigger: str, tick: int
    ) -> None:
        ...

    def register_artifact(self, artifact: Artifact, content: str) -> None:
        ...

    def release_session(self, session_token: str) -> None:
        ...

    def remove_artifact(self, artifact_id: UUID) -> None:
        ...

    def retention_meta(self) -> tuple[bool, RetentionPolicy | None]:
        ...

    def session_count(self) -> int:
        ...

    def set_agent_state(
        self,
        artifact_id: UUID,
        agent_id: UUID,
        state: MESIState,
        *,
        trigger: str = "unknown",
        tick: int = 0,
        content_hash: str | None = None,
        observed: bool = True,
    ) -> None:
        """Set one agent's MESI state on one artifact.

        ``observed`` says whether this transition certifies that the agent now
        holds the artifact's CURRENT bytes. Default ``True``: a non-INVALID
        target records the current version as its ``last_observed_version``.
        ``False`` is a grant that certifies no read -- the Claude Code
        pre-bash / pre-grep re-grant issued alongside a DENIED command, which
        never ran -- and leaves the recorded value exactly as a transition to
        INVALID does: the prior value kept, a never-observed pair still None.
        It never affects the state written, the grant tick, the epoch or the
        read-generation capture.
        """
        ...

    def set_agent_transient(
        self,
        artifact_id: UUID,
        agent_id: UUID,
        transient_state: TransientState,
        *,
        entered_tick: int,
    ) -> None:
        ...

    def set_checkpoint_member_pin(
        self,
        checkpoint_id: str,
        member_path: str,
        *,
        pin_state: str,
        restore_tier: str | None = None,
    ) -> None:
        """Update a member's pin state; ``restore_tier`` (when given) rewrites
        the member's restore tier in the same step — the Unit-6 loud tier
        downgrade (``restorable`` -> ``restorable-unpinned`` on a failed pin
        leg). ``restore_tier=None`` leaves the tier untouched. Raises
        ``KeyError`` for an unknown (checkpoint, member) pair."""
        ...

    def set_checkpoint_member_restore(
        self,
        checkpoint_id: str,
        member_path: str,
        *,
        restore_outcome: str | None,
        deleted_at_restore: float | None = None,
    ) -> None:
        """Record a member's restore progress: BOTH columns are written to the
        given values (a full member-restore-state write — a new restore run's
        first write for a member resets any prior run's delete record). Raises
        ``KeyError`` for an unknown (checkpoint, member) pair."""
        ...

    def set_checkpoint_restore_status(
        self, checkpoint_id: str, status: str, *, updated_at: float
    ) -> None:
        """Update the checkpoint-level restore status + its ``updated_at``
        stamp. Raises ``KeyError`` for an unknown checkpoint."""
        ...

    def set_artifact_and_content(
        self,
        artifact_id: UUID,
        artifact: Artifact,
        content: str,
        *,
        last_writer: Optional[UUID] = None,
        fence_agent_id: Optional[UUID] = None,
    ) -> None:
        ...

    def set_transfer_status(
        self,
        artifact_id: UUID,
        status: str,
        *,
        counterparty: UUID | None = None,
        now_unix: float | None = None,
    ) -> None:
        """Write a record's label: ``status``, ``counterparty`` (None clears
        it) and ``updated_at``, all three. Unconditional -- it never checks
        liveness, because a win's label is written after the version already
        moved; the service decides from the record it read under the same
        hold. Raises ``ValueError`` for a status outside
        :data:`TRANSFER_STORED_STATUSES` (``superseded`` included) and
        ``KeyError`` when the path has no record; either way nothing is
        written."""
        ...

    def transfer_grants(
        self,
        request: TransferRequest,
        *,
        tick: int = 0,
        now_unix: float | None = None,
    ) -> list[TransferGrantOutcome]:
        """The composite transfer: decide every path of ``request`` with
        :func:`decide_transfer_grant`, then apply the admitted subset -- the
        record upsert, and for a plain handoff the presented composite's
        INVALID move under ``HANDOFF_TRIGGER`` (moving ``owner_generation``
        when it leaves EXCLUSIVE or MODIFIED) with one state-log entry per
        path -- all inside ONE lock hold and ONE transaction. A refused path is
        left untouched while the admitted ones commit; any raise mid-apply
        rolls every path back. Returns one outcome per path, in request
        order. The registry never derives an identity: the giver and the
        successor arrive session-level, the holders as presented."""
        ...

    def valid_holders(self, artifact_id: UUID) -> list[UUID]:
        ...


@runtime_checkable
class ForeignWriteDetection(Protocol):
    """The detection instrument's own surface, deliberately NOT part of
    :class:`SqliteExtended`.

    ``SqliteExtended`` is ``runtime_checkable`` and ``CoordinatorService`` uses
    an ``isinstance`` against it to choose between the durable
    ``resolve_or_register`` and a slower lock-guarded mint. A structural check
    passes only when EVERY member is present, so folding instrumentation into
    that protocol would silently demote a backend that implements the whole
    coordination surface but not this one — a behaviour change no author of
    such a backend could see coming.

    The separation also matches how :mod:`ccs.coordinator.backend_contract`
    classifies these members: observability exhaust the atomic write boundary
    never reads. A backend may implement this surface independently, and a
    backend that does not is simply un-instrumented.
    """

    def artifacts_with_detection_edge(self) -> set[UUID]:
        ...

    def clear_detection_edges(self, artifact_ids: list[UUID]) -> None:
        ...

    def close_detection_run(self) -> None:
        ...

    def detection_runs(self) -> list[DetectionRun]:
        ...

    def detection_uncoverable(self) -> list[UncoverableRun]:
        ...

    def foreign_write_totals(self) -> dict[UUID, dict[str, int]]:
        ...

    def record_detection_uncoverable(self, reason: str, now_unix: float) -> None:
        ...

    def record_detection_tick(self, now_unix: float, *, covered_count: int = 0) -> None:
        ...

    def record_foreign_write(
        self, artifact_id: UUID, outcome: str, disk_hash: str
    ) -> bool:
        ...


@runtime_checkable
class SqliteExtended(RegistryBase, Protocol):
    """The extended registry surface ``coordinator_server.py`` depends on —
    :class:`RegistryBase` plus the methods that only the SQLite-backed
    registry (:class:`~ccs.coordinator.sqlite_registry.SqliteArtifactRegistry`)
    provides today.

    These cover the durable-store-only concerns: connection ``close``, durable
    name/prefix lookups, the preemption-notice surface (record/peek/pop/evict),
    ``resolve_or_register`` first-observation seeding, and the ``status_snapshot``
    batch. Extracted as a pure refactor (no behavior change).
    """

    def artifact_names_under_prefix(self, prefix: str) -> list[str]:
        ...

    def artifacts_held_by_agent(
        self, agent_id: UUID, states: Iterable[MESIState]
    ) -> list[UUID]:
        ...

    def close(self) -> None:
        ...

    def evict_stale_notices(
        self, *, max_age_sec: float, now_unix: Optional[float] = None
    ) -> int:
        ...

    def get_artifact_updated_at(self, artifact_id: UUID) -> Optional[float]:
        ...

    def last_writer_for(self, artifact_id: UUID) -> Optional[UUID]:
        ...

    def lookup_artifact_id_by_name(self, parent_rel_path: str) -> UUID | None:
        ...

    def peek_preemption_notice(
        self, agent_id: UUID, artifact_id: UUID
    ) -> Optional[tuple[UUID, float]]:
        ...

    def pop_pending_notices(
        self, agent_id: UUID, *, consume_limit: int | None = None
    ) -> list[tuple[UUID, UUID, float]]:
        ...

    def pop_preemption_notice(
        self, agent_id: UUID, artifact_id: UUID
    ) -> Optional[tuple[UUID, float]]:
        ...

    def record_preemption_notice(
        self,
        *,
        victim_agent_id: UUID,
        artifact_id: UUID,
        preempter_agent_id: UUID,
        preempted_at_unix_ts: float,
    ) -> None:
        ...

    def resolve_or_register(
        self,
        parent_rel_path: str,
        content_hash: str,
        *,
        initial_owner: Optional[UUID] = None,
    ) -> UUID:
        ...

    def status_snapshot(
        self,
        *,
        agent_ids: Iterable[UUID] | None = None,
        include_transfers: bool = False,
    ) -> (
        tuple[
            dict[UUID, dict[str, Any]],
            dict[UUID, dict[UUID, MESIState]],
        ]
        | tuple[
            dict[UUID, dict[str, Any]],
            dict[UUID, dict[UUID, MESIState]],
            dict[UUID, tuple[TransferRecord, bool]],
        ]
    ):
        """The artifact rows and the per-artifact state maps, read under ONE
        lock hold; ``agent_ids`` scopes the state half to the named agents.

        ``include_transfers`` (keyword-only, off by default) adds a third
        element, ``{artifact_id: (record, live)}`` for every artifact that has
        a transfer record, read inside the same hold and judged by the same
        liveness helper :meth:`RegistryBase.get_transfer_record` uses,
        so ``/status`` renders each record beside the version it was judged
        against. Without it the answer is the two-element tuple, unchanged."""
        ...
