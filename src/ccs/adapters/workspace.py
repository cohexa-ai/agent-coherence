# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""WorkspaceVersioner — the Workspace-Versioning capture engine (WV Unit 3).

A workspace is a set of heterogeneous MEMBERS: files on a
:class:`~ccs.adapters.coherent_volume.CoherentVolume` (or any source speaking
its ``read_with_version`` surface), S3 objects on a
:class:`~ccs.adapters.coherent_object.CoherentObject`, and declared
forward-only members (actions/effects with no state to capture).
:meth:`WorkspaceVersioner.checkpoint` takes a **skew-declared cut** over them:

- **Per-member capture** — one read per member yields the restore POINTER
  (``native_token``: the S3 versionId / the coordinator content-state version
  for a file), a fixed-width fingerprint, and a
  :func:`~ccs.core.clock.monotonic_seconds` timestamp. The CAS comparand (the
  S3 ETag / the live coordinator version at restore time) is NEVER manifested
  — it is re-read live when a restore leg runs (the F4 pointer-vs-comparand
  split).
- **ABSENT is a fact, not an empty body** — a missing member is recorded
  ``absent=True`` with no token and no fingerprint, distinct from a present
  empty member (whose fingerprint is ``sha256(b"")``).
- **Honest tiers per member** — derived through
  :func:`~ccs.core.substrate.derive_restore_tier`, never asserted: a versioned
  S3 member is ``restorable`` (the substrate offers history + a per-version
  pin; the Unit-6 pin leg establishes the hold and downgrades LOUDLY on
  failure), an unversioned S3 member is ``forward_only`` (the typed
  :class:`~ccs.adapters.coherent_object.VersionPointerUnconfirmed` refusal IS
  the discovery — no pre-probe), a file member is ``restorable-unpinned``
  (coordinator retention holds history; the retention pin is Unit 6), and a
  member whose pointer cannot be confirmed is ``forward_only`` (an unconfirmed
  pointer never lands in a manifest — the Sentinel rule).
- **Torn-cut detection** — after the capture window closes, EVERY captured
  member is re-read once; any observed movement (token, fingerprint, or
  presence) marks that member ``dirty_during_window=True``. Conservative by
  design: the verification read runs after ``window_max``, so a write landing
  between the window close and the verify read still flags — dirty means "not
  verified quiescent across the window", never "verified torn". Disclosed
  residual (the tail window): a foreign write landing AFTER the verification
  read but BEFORE the manifest persists is NOT flagged — the cut itself stays
  internally consistent (every token and fingerprint comes from the capture
  reads) and restore never trusts the flag (every leg re-reads its live
  comparand, and RECORDS what that read saw — see the restore ``observation``
  below), so the residual under-flags the window; it never corrupts the
  manifest (the safe direction).
- **The window** — ``[window_min, window_max]`` is the min/max of the members'
  capture timestamps (whole-second wall-clock ticks; the skew is DECLARED, not
  hidden).
- **One registration** — the manifest persists through the coordinator
  service's ``create_workspace_checkpoint`` (the Unit-2 registry API: header +
  owner + every member in ONE transaction). Any persist failure raises the
  typed :class:`CheckpointPersistFailed`; the single-transaction registration
  means a raise leaves NO partial manifest.

v1 limitation (typed, capture-time): a file member whose bytes are not UTF-8
text is refused with :class:`BinaryFileMemberRefused` BEFORE anything persists
— the file restore leg rides the snapshot-session text wire (the same
constraint ``CoherentVolume.atomic_publish`` enforces), so capturing a member
that could never be restored would be a silent over-claim.

**Restore (WV Unit 4 / R3)** — :meth:`WorkspaceVersioner.restore` drives one
conditional leg per DURABLE member row under a TERMINATION CONTRACT:

- **Every member reaches exactly one terminal outcome** from the closed
  :data:`~ccs.core.exceptions.RESTORE_MEMBER_OUTCOMES` vocabulary — success
  (``restored`` / ``converged``), absorbing (``conflict`` / ``target_lost`` /
  ``forward_only_skipped``), or hold (``held_unconfirmed``). A restore that
  starts always CONCLUDES with a frozen per-member report; per-member failures
  are absorbed into the report, never raised — including a source's own
  :class:`StructuralMemberRefused` (a member the source declares undrivable
  through its declared surface), which is re-drive-proof by construction and
  so must never wedge the OTHER members' restore.
- **Bounded re-drive** — each contended leg re-drives at most
  :data:`MAX_RESTORE_LEG_REDRIVES` times (the ``MAX_CAS_REACQUIRES`` twin);
  sustained foreign-writer contention exhausts the budget into the absorbing
  ``conflict``, never a livelock.
- **Per-member honesty** — the S3 leg is NATIVE-CAS (an If-Match put; the
  substrate arbitrates a racing foreign writer); the file leg is
  **no-arbiter**: a version-checked CAS whose foreign-edit signal is
  adapter-local DETECTION only, and every file outcome is labeled so — never
  presented as substrate arbitration (the cross-host carve-out).
- **Crash-resumable from durable state** — progress rides the registry
  (checkpoint ``restore_status``: ``none`` → ``in_progress`` → ``concluded``;
  per-member ``restore_outcome`` rows). A fresh engine restoring an
  ``in_progress`` checkpoint RESUMES: members with a terminal outcome are
  skipped (reported ``resumed_from_prior_run``), the rest are re-driven
  idempotently — a member whose live state already matches the manifest
  concludes ``converged`` WITHOUT a write, so a crash between a landed leg and
  its durable outcome record can never double-apply.
- **Delete legs restore the ABSENT fact** — a member captured ABSENT that
  exists live is deleted (S3: an unconditional-latest ``delete``, minting a
  marker on a versioned bucket; the pre-delete race window is a documented
  residual).
- **What restore does NOT promise, and what it reports instead** — restore is
  not a merge and nothing on this path refuses a write: a member whose content
  moved after the capture is put back OVER, and that later content is gone.
  What the engine promises is that the run SAYS so. Each leg's comparand read
  is also recorded, as a :class:`RestoreObservation` on that member's outcome
  (``state`` / ``pointer`` / ``fingerprint``, from the closed
  :data:`~ccs.core.exceptions.RESTORE_OBSERVATION_STATES` vocabulary):
  ``observed_differs`` (a live state was read, it differed, and the write
  discarded it — the ONLY state carrying the pointer to the version
  overwritten and a digest of the content overwritten), ``no_live_state``
  (create-on-absent; nothing discarded), ``present_not_comparable`` (the
  delete leg's probe established live state EXISTED and destroyed it without
  reading a comparand to name it by), ``no_write_attempted`` (no write
  decision was reached — the default), and ``not_recorded`` (this run holds no
  observation at all; never clean). The values ride the read each leg was
  ALREADY making — no second substrate call — so the split-comparand rule
  stands: on the S3 leg the ETag remains the comparand and is never recorded,
  the versionId is the pointer.

**Registration (WV Unit 5 / R4–R5)** — after every member is terminal and
before ``concluded``, :meth:`WorkspaceVersioner._registration_seam` registers
the restore with the coordinator, split per member class (the plan's
restore-registration design, grounded in what the coordinator stores):

- **written FILE members** (``restored``, no delete record, ``no-arbiter``
  tier) register through ``CoordinatorService.register_workspace_restore`` →
  ONE all-or-nothing ``commit_all`` batch (hash-only — fingerprints, never
  bytes; the controller is the versioner's owner, an S/I caller per D4).
  Registered peers holding the artifact are invalidated atomically and the
  invalidation signals surface on the report; a superseded controller's late
  apply is rejected by the read-generation fence (``stale_read_generation``,
  never retried). The registration is a FORWARD commit carrying old bytes:
  versions strictly increase, never decrement.
- **written S3 members** are registered MANIFEST-SIDE ONLY: the coordinator
  holds no artifact identity for a BYO-substrate member (the substrate owns
  identity; their manifest ``artifact_id`` is NULL by design) — their durable
  ``restore_outcome`` row IS the registration.
- **deleted members** are recorded manifest-side (``deleted_at_restore``,
  written by the delete leg) — ``commit_all`` has no delete semantics and the
  registration adds NO commit interaction for them.
- **an empty commit write-set** (every member skipped/absorbed/converged/
  deleted/S3) concludes via the status update alone — ``commit_all`` is NEVER
  called (it raises on an empty set).

Idempotent under crash-resume by TWO durable mechanisms: the checkpoint-level
``registered`` status (written between a committed/empty registration and
``concluded`` — a resume finding it skips the step), and the service-side
hash filter (an artifact already carrying the manifest fingerprint is skipped,
so a crash between the ``commit_all`` landing and the marker write still
re-answers without a second version bump).

**Pins (WV Unit 6 / R9)** — "``restorable`` means restorable, or says
``restorable-unpinned`` loudly". :meth:`WorkspaceVersioner.checkpoint` runs
the pin legs by default (``pin=True``) right after the manifest persists —
folding the pin into the capture verb is the ergonomic fail-closed default: a
checkpoint whose S3 members claim ``restorable`` must not depend on a second
call the operator can forget. The honesty contract is the durable PAIR
``(restore_tier, pin_state)``: ``restorable`` is BACKED only by
``pin_state="held"`` (:data:`~ccs.core.exceptions.PIN_STATES`); every
consumer must render (``restorable``, ``unpinned``) — the ``pin=False``
capture-only shape, or a run that died before a member's pin leg — as
claimed-but-not-yet-backed. :meth:`WorkspaceVersioner.pin_checkpoint`
re-drives pins idempotently (only ``unpinned`` members are attempted).

- **S3 member (tier ``restorable``)** — the pin is the substrate's own:
  ``set_legal_hold`` on the manifested versionId. Established → ``held`` (+1
  on the checkpoint's pin refcount); the typed
  :class:`~ccs.adapters.coherent_object.LegalHoldUnavailable` (no Object Lock
  configuration) → the LOUD durable downgrade to ``restorable-unpinned`` with
  ``pin_state="pin_unavailable"`` — the tier and the pin state land in ONE
  registry write, never silently; a version already gone at pin time
  (``KeyError``) downgrades the same way (restore will say ``target_lost``).
- **File member (tier ``restorable-unpinned``)** — v1's VERIFICATION pin over
  the coordinator's declared retention: the captured version's retained bytes
  are read through the :class:`FileContentResolver` seam and checked against
  the captured fingerprint. Verified → ``held`` (+1) with the tier kept at
  ``restorable-unpinned`` (coordinator retention offers no per-version hold
  in v1 — a bounded K/T policy may still evict, which restore reports
  ``target_lost``; the R13 declared-retention disclosure documents this).
  Retention off / the version not retained / a fingerprint mismatch → the
  LOUD downgrade to ``forward_only`` with ``pin_state="pin_unavailable"``
  (the captured state is ALREADY unreachable — describing it as any flavor of
  restorable would be the silent-decay lie). A versioner with no resolver
  skips file pin legs entirely (nothing above ``restorable-unpinned`` was
  ever claimed, so nothing needs backing).
- **Forward-only / absent / unconfirmed members** — never pinned, untouched.
- **Release is PUBLIC at the checkpoint level** (:meth:`WorkspaceVersioner.
  release_checkpoint`, over the ``_release_checkpoint_pins`` engine); there is
  still no checkpoint-DELETE verb, and no CLI/HTTP route for either half.
  Refcount-aware across
  checkpoints: before dropping an S3 legal hold the release scans every OTHER
  checkpoint's members for another ``held`` pin of the same ``(member_path,
  native_token)`` — a shared hold survives until the LAST holder releases.
  Each released member is recorded ``pin_state="released"`` (tier downgraded
  to ``restorable-unpinned`` where it was ``restorable``) BEFORE the substrate
  hold is dropped — a crash between the record and the drop leaves an
  over-retained version (safe direction), never an unbacked ``restorable``
  claim. A SHARING peer's own release still drops it; where this checkpoint
  was the sole holder nothing in the product does, and the recovery is
  ``CoherentObject.release_legal_hold`` on the binding, by version.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)
from uuid import UUID

from ccs.adapters.coherent_object import (
    CREATE_IF_ABSENT,
    CoherentObject,
    LegalHoldUnavailable,
    VersionedCasWritten,
    VersionPointerUnconfirmed,
)
from ccs.adapters.coherent_volume import denied_read_backoff_sec
from ccs.adapters.substrate import CasConflict, ReconcileVerdict
from ccs.coordinator.registry_protocol import CheckpointMember, CheckpointRecord
from ccs.core.clock import monotonic_seconds
from ccs.core.exceptions import (
    CHECKPOINT_ALREADY_REGISTERED_REASON,
    CHECKPOINT_NOT_THE_RECEIVER_REASON,
    PIN_STATE_HELD,
    PIN_STATE_RELEASED,
    PIN_STATE_UNAVAILABLE,
    PIN_STATE_UNPINNED,
    RESTORE_MEMBER_OUTCOMES,
    RESTORE_OBSERVATION_DIFFERS,
    RESTORE_OBSERVATION_NO_LIVE_STATE,
    RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
    RESTORE_OBSERVATION_NOT_RECORDED,
    RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE,
    RESTORE_OBSERVATION_STATES,
    RESTORE_OBSERVATION_STATES_WITHOUT_COMPARAND,
    RESTORE_OUTCOME_CONFLICT,
    RESTORE_OUTCOME_CONVERGED,
    RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED,
    RESTORE_OUTCOME_HELD_UNCONFIRMED,
    RESTORE_OUTCOME_RESTORED,
    RESTORE_OUTCOME_TARGET_LOST,
    RESTORE_OUTCOMES_PROVING_NO_WRITE,
    RESTORE_STATUS_CONCLUDED,
    RESTORE_STATUS_IN_PROGRESS,
    RESTORE_STATUS_REGISTERED,
    RESTORE_STATUSES,
    STALE_READ_GENERATION_REASON,
    WORKSPACE_REGISTRATION_COMMITTED,
    WORKSPACE_REGISTRATION_EMPTY,
    WORKSPACE_REGISTRATION_PRIOR_RUN,
    WORKSPACE_REGISTRATION_REFUSED,
    CasRetriesExhausted,
    CasVersionConflict,
    CheckpointRegistrationRefused,
    CheckpointUnknown,
    CoherenceError,
    CommitUnconfirmed,
    OccCallerTransientError,
    StaleView,
    ViewWedged,
)
from ccs.core.substrate import ArbitrationTier, RestoreTier, derive_restore_tier
from ccs.core.substrate import sha256_hex as _sha256_hex
from ccs.core.types import WorkspaceRegistrationResult, WorkspaceRestoreWrite

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from typing import Callable

__all__ = [
    "BINARY_FILE_MEMBER_REASON",
    "BinaryFileMemberRefused",
    "CHECKPOINT_NOT_PERSISTED_REASON",
    "CheckpointPersistFailed",
    "CheckpointPersistence",
    "CheckpointPinStore",
    "CheckpointRestoreStore",
    "FileContentResolver",
    "FileMemberSource",
    "FileRestoreTarget",
    "MAX_RESTORE_LEG_REDRIVES",
    "MemberRestoreOutcome",
    "OBSERVATION_NO_WRITE_ATTEMPTED",
    "OBSERVATION_NOT_RECORDED",
    "RestoreObservation",
    "RestoreRegistration",
    "STRUCTURAL_MEMBER_REFUSAL_REASON",
    "StructuralMemberRefused",
    "WorkspaceCheckpoint",
    "WorkspaceRestoreReport",
    "WorkspaceVersioner",
]

# Per-leg re-drive budget for a contended restore leg (WV Unit 4 / R3): total
# leg iterations allowed is MAX_RESTORE_LEG_REDRIVES + 1 (the initial attempt
# plus the re-drives) — the twin of ``coherent_volume.MAX_CAS_REACQUIRES`` and
# ``coherent_object.MAX_RETRYABLE_PUT_ATTEMPTS`` (one budget discipline across
# the family). Exhaustion is the absorbing ``conflict`` outcome, never a raise
# and never a livelock: a restore leg racing a sustained foreign writer loses
# honestly and the restore still concludes.
MAX_RESTORE_LEG_REDRIVES: Final[int] = 8


# --- typed-reason vocabulary (identity-matched constants; add, never rename) ----

# A file member's bytes are not UTF-8 text: the file restore leg rides the
# snapshot-session TEXT wire, so the member could never be restored — refuse at
# capture (typed), never silently capture an unrestorable member.
BINARY_FILE_MEMBER_REASON: Final[str] = "binary_file_member_unsupported"

# The checkpoint manifest was NOT persisted: the coordinator/registry raised
# during the single-transaction registration, so no header and no member row
# landed (no partial manifest — the registration is all-or-nothing).
CHECKPOINT_NOT_PERSISTED_REASON: Final[str] = "checkpoint_not_persisted"

# A member source REFUSES this member structurally: its declared surface will
# never yield or accept the member's state, whatever the caller does. The
# engine-owned base reason; a concrete source subclasses the exception and
# narrows the reason to its own vocabulary (the CLI's containment refusal).
STRUCTURAL_MEMBER_REFUSAL_REASON: Final[str] = "structural_member_refused"


class BinaryFileMemberRefused(CoherenceError):
    """A file member holds non-UTF-8 bytes — the typed capture-time refusal.

    The v1 file restore leg rides the snapshot-session text wire
    (``CoherentVolume.atomic_publish`` enforces the same UTF-8 constraint), so
    a binary member can never be restored; capturing it would over-claim.
    Raised BEFORE anything persists. Carries
    :data:`BINARY_FILE_MEMBER_REASON`, matched by identity.
    """

    reason = BINARY_FILE_MEMBER_REASON
    #: The member the refusal names (the operator removes it or accepts that
    #: this workspace cannot be checkpointed in v1).
    member_path: str = ""

    def __init__(self, message: str, *, member_path: str) -> None:
        super().__init__(message)
        self.member_path = member_path


class StructuralMemberRefused(CoherenceError):
    """A member source STRUCTURALLY refuses this member — the engine-owned base.

    The seam-level twin of :class:`BinaryFileMemberRefused`: both name a member
    the engine can never drive. The difference is who decides. The binary
    refusal is the ENGINE's own capture-time content limitation; this one is
    raised BY a :class:`FileMemberSource` / :class:`FileRestoreTarget`
    implementation about the member's own shape — the CLI's working-tree bridge
    refuses a member path that fails containment validation (a symlink
    component, a hardlinked regular file with an external co-owner, a
    ``.coherence`` self-target, a non-regular leaf), and any other source may
    refuse for its own structural reason.

    Sources raise a SUBCLASS carrying their own vocabulary (the CLI's
    ``MemberPathRefused`` narrows :attr:`reason` and stays a ``ValueError`` for
    its own arg-validation handling); the engine catches only this base, so it
    never imports the interface module that defines the subclass.

    The engine's two legs treat it differently, by design:

    - **capture** — it propagates. A member that can never be driven must never
      land in a manifest, so the operator sees a hard typed refusal and nothing
      persists.
    - **restore** — :meth:`WorkspaceVersioner._drive_member_absorbing` ABSORBS
      it into the absorbing ``target_lost`` (the same "unreachable through its
      declared surface" meaning the ``OSError`` family already maps there): a
      re-drive can never succeed, so raising would wedge ``restore_status`` at
      ``in_progress`` durably and break the termination contract for every
      OTHER member of the checkpoint.

    Carries :data:`STRUCTURAL_MEMBER_REFUSAL_REASON` unless a subclass narrows
    it, and (by convention, like the sibling refusals) names the member in
    :attr:`member_path`.
    """

    reason = STRUCTURAL_MEMBER_REFUSAL_REASON
    #: The member the refusal names. Subclasses set it in their constructor.
    member_path: str = ""


class CheckpointPersistFailed(CoherenceError):
    """The manifest registration raised — the checkpoint was NOT persisted.

    The Unit-2 registration is a single transaction (header + owner + every
    member land together or not at all), so this failure guarantees NO partial
    manifest. The operational cause chains as ``__cause__``. Carries
    :data:`CHECKPOINT_NOT_PERSISTED_REASON`, matched by identity.
    """

    reason = CHECKPOINT_NOT_PERSISTED_REASON


# --- the member-source seams (structural; CoherentVolume / CoordinatorService
# --- satisfy them without importing this module) --------------------------------


@runtime_checkable
class FileMemberSource(Protocol):
    """The file-member read surface: ``CoherentVolume.read_with_version``.

    One call returns ``(bytes, coordinator_version)`` — the bytes and the
    coordinator's authoritative version from the SAME read path, so the
    fingerprint and the restore pointer always describe one observation.
    Raises ``FileNotFoundError`` for an absent member (the ABSENT fact).
    A version ``< 1`` means the pointer could not be resolved (no coordinator
    / degraded) — the capture treats it as UNCONFIRMED (never manifested).
    May raise :class:`~ccs.core.exceptions.StaleView` when the bytes on disk
    cannot be paired with a version (``CoherentVolume`` refuses that pair):
    capture records the member UNCONFIRMED too, with no digest, and flags it
    ``dirty_during_window``; a restore leg re-drives within its budget and then
    concludes ``conflict`` without writing.
    May raise :class:`StructuralMemberRefused` (or a subclass) to declare THIS
    MEMBER undrivable through this surface: capture propagates it as a hard
    typed refusal, restore absorbs it into that member's ``target_lost``.
    """

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        ...


@runtime_checkable
class CheckpointPersistence(Protocol):
    """The persist seam: ``CoordinatorService.create_workspace_checkpoint``.

    ONE call registers the whole manifest (header + owner + members) in one
    transaction via the Unit-2 registry API and returns the minted header.
    A checkpoint taken with a ``receiver`` (#191) passes it as one more
    keyword, ``receiver=``; one without passes nothing, so a seam that
    predates the keyword keeps serving receiver-less checkpoints.
    """

    def create_workspace_checkpoint(
        self,
        *,
        name: str,
        owner: UUID,
        members: Sequence[CheckpointMember],
        window_min: float,
        window_max: float,
        issued_at_tick: int = 0,
    ) -> CheckpointRecord:
        ...


@runtime_checkable
class CheckpointRestoreStore(Protocol):
    """The restore engine's durable-store seam (WV Units 4–5 / R3–R5):
    ``CoordinatorService``'s workspace-checkpoint read + progress +
    registration surface.

    Restore is driven FROM these durable reads and records progress THROUGH
    these durable writes — never from an in-memory capture return — so a fresh
    engine (post-crash) resumes from exactly the state the registry holds.
    :meth:`register_workspace_restore` is the Unit-5 coordinator-registration
    half (written file members → one all-or-nothing hash-only ``commit_all``).
    :meth:`workspace_member_registered` is its READ-ONLY rebuild seam: a
    re-restore of an already-``concluded`` checkpoint re-derives the
    registration answer from the coordinator's registered state (a
    refused-terminal registration must never re-read as ``None``-means-fine)
    — it never resolves, mints, or mutates anything.
    :meth:`WorkspaceVersioner.restore` verifies its service speaks this surface
    BEFORE touching any state (fail-fast on a capture-only service).
    """

    def get_workspace_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        ...

    def get_workspace_checkpoint_members(
        self, checkpoint_id: str
    ) -> "list[CheckpointMember]":
        ...

    def set_workspace_checkpoint_restore_status(
        self, checkpoint_id: str, status: str, *, updated_at: float
    ) -> None:
        ...

    def set_workspace_checkpoint_member_restore(
        self,
        checkpoint_id: str,
        member_path: str,
        *,
        restore_outcome: str | None,
        deleted_at_restore: float | None = None,
    ) -> None:
        ...

    def register_workspace_restore(
        self,
        *,
        checkpoint_id: str,
        controller: UUID,
        writes: Sequence[WorkspaceRestoreWrite],
        issued_at_tick: int = 0,
    ) -> WorkspaceRegistrationResult:
        ...

    def workspace_member_registered(self, member_path: str, fingerprint: str) -> bool:
        ...


@runtime_checkable
class CheckpointPinStore(Protocol):
    """The pin engine's durable-store seam (WV Unit 6 / R9):
    ``CoordinatorService``'s checkpoint read + pin-state + refcount surface.

    Pin legs are driven FROM the durable member rows and record their answer
    THROUGH ``set_workspace_checkpoint_member_pin`` — the pin state and any
    loud tier downgrade land in ONE registry write.
    ``list_workspace_checkpoints`` feeds the checkpoint release's
    cross-checkpoint scan (a legal hold shared by another checkpoint's
    ``held`` member must survive this checkpoint's release).
    :meth:`WorkspaceVersioner.checkpoint` verifies its service speaks this
    surface BEFORE capturing anything when pins will be needed (fail-fast on a
    capture-only seam; pass ``pin=False`` there).
    """

    def get_workspace_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        ...

    def get_workspace_checkpoint_members(
        self, checkpoint_id: str
    ) -> "list[CheckpointMember]":
        ...

    def list_workspace_checkpoints(self) -> "list[CheckpointRecord]":
        ...

    def set_workspace_checkpoint_member_pin(
        self,
        checkpoint_id: str,
        member_path: str,
        *,
        pin_state: str,
        restore_tier: str | None = None,
    ) -> None:
        ...

    def adjust_workspace_checkpoint_pin_refcount(
        self, checkpoint_id: str, delta: int
    ) -> int:
        ...


@runtime_checkable
class FileContentResolver(Protocol):
    """The file-member HISTORY seam: the captured bytes for ``(path, version)``.

    The restore engine reads a file member's pinned bytes through THIS Protocol
    only. Unit 5 wires the real resolution (coordinator retention's
    ``get_content_at_version`` + member-path→artifact-id mapping); Unit-4 tests
    inject a fake. Raises :class:`KeyError` when the version is not retained —
    the engine maps it to the absorbing ``target_lost`` outcome (an expired
    retention window is the file twin of an expired S3 pin).
    """

    def content_at(self, member_path: str, version: int) -> bytes:
        ...


@runtime_checkable
class FileRestoreTarget(FileMemberSource, Protocol):
    """The file-member RESTORE surface: the capture read plus the single-shot
    version-checked CAS write — ``CoherentVolume`` speaks both natively.

    The write leg is **no-arbiter**: ``write_cas_at`` commits iff the
    coordinator's current version equals ``expected_version``, which DETECTS a
    foreign edit adapter-locally (typed
    :class:`~ccs.core.exceptions.CasVersionConflict`) — it is never substrate
    arbitration, and no restore outcome may present it as such (the cross-host
    carve-out). A confirmed win lands ``new_content`` and advances the version
    deterministically to ``expected_version + 1``. Like the read leg it may
    raise :class:`StructuralMemberRefused` to refuse the member outright (no
    write lands); the restore absorbs that into ``target_lost``.
    """

    def write_cas_at(
        self, path: str, expected_version: int, new_content: bytes
    ) -> None:
        ...


# --- declared members (registration-time facts) ---------------------------------


@dataclass(frozen=True)
class _FileMember:
    member_path: str
    source: FileMemberSource


@dataclass(frozen=True)
class _ObjectMember:
    member_path: str
    binding: CoherentObject
    key: str


@dataclass(frozen=True)
class _ForwardOnlyMember:
    member_path: str


# --- the capture result ---------------------------------------------------------


@dataclass(frozen=True)
class WorkspaceCheckpoint:
    """One persisted checkpoint: the minted header + the member rows as stored.

    ``record.checkpoint_id`` keys later restore/status calls (Units 4–5);
    ``members`` are the exact :class:`CheckpointMember` rows the registry
    holds — capture facts only, never content.
    """

    record: CheckpointRecord
    members: tuple[CheckpointMember, ...]


@dataclass(frozen=True)
class _Observation:
    """One member's live observation — capture pass and verify pass share it.

    ``token`` is the restore pointer (or ``None`` when absent/unconfirmed);
    ``fingerprint`` is the fixed-width content digest (``None`` when absent, or
    when the read returned no bytes); ``restorable``/``pinnable`` feed
    :func:`derive_restore_tier` on the capture pass (the verify pass compares
    only presence/token/fingerprint). ``confirmed`` is ``False`` when the source
    refused to pair the member's bytes with a version, so neither a pointer nor
    a digest was observed.
    """

    absent: bool
    token: str | None
    fingerprint: str | None
    versioned: bool = False
    pinnable: bool = False
    confirmed: bool = True


# --- the restore report (frozen facts; durably mirrored in the registry) --------


@dataclass(frozen=True)
class RestoreObservation:
    """What ONE restore leg saw of the live state immediately before it wrote.

    An additive REPORT value carried beside the member outcome — the outcome
    vocabulary is unchanged, so a member restored over content committed after
    the capture still concludes ``restored``. ``state`` is one of the closed
    :data:`~ccs.core.exceptions.RESTORE_OBSERVATION_STATES`, matched by IDENTITY
    (never a substring of any ``detail`` line). ``pointer`` (the live restore
    pointer the leg read) and ``fingerprint`` (that state's content digest) are
    lifted from the SAME read that produced the leg's CAS comparand, so no
    second read is ever issued to populate them. Construction enforces one half
    of that: every state that read no comparand is refused a pointer and a
    fingerprint. It does not require them on ``observed_differs``, because a
    source may land a write without naming a version, so the report treats both
    as optional there.

    NEVER a comparand: nothing downstream may seed a CAS, an If-Match or a
    pinned read from a value recorded here. It describes what the leg
    overwrote, not what the next write may overwrite.
    """

    state: str
    pointer: str | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        # Fail closed on an unclassifiable state: every consumer branches on a
        # named state, so an unrecognised one would fall through every arm and
        # be read as clean — the one answer an honesty field must never give.
        if self.state not in RESTORE_OBSERVATION_STATES:
            raise ValueError(
                f"unknown restore observation state {self.state!r}; the "
                f"vocabulary is {sorted(RESTORE_OBSERVATION_STATES)} "
                "(fail-closed: an unclassifiable state would read as clean)"
            )
        if self.state in RESTORE_OBSERVATION_STATES_WITHOUT_COMPARAND and (
            self.pointer is not None or self.fingerprint is not None
        ):
            raise ValueError(
                f"restore observation {self.state!r} observed no comparand, so "
                "it carries no pointer and no fingerprint (naming one would "
                "claim content the leg never compared)"
            )


# The two observations no leg's read produces, as shared immutable singletons
# (frozen, so one instance is safe as a default). They are kept DISTINCT on
# purpose: ``no_write_attempted`` is a fact this run established (the member
# converged, was skipped, or was absorbed before any write), while
# ``not_recorded`` is the absence of a fact — this run drove no leg, or drove
# one whose outcome it could not learn.
OBSERVATION_NO_WRITE_ATTEMPTED: Final[RestoreObservation] = RestoreObservation(
    state=RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
)
OBSERVATION_NOT_RECORDED: Final[RestoreObservation] = RestoreObservation(
    state=RESTORE_OBSERVATION_NOT_RECORDED
)


@dataclass(frozen=True)
class MemberRestoreOutcome:
    """One member's TERMINAL restore outcome — the termination contract's unit.

    ``outcome`` is one of the closed
    :data:`~ccs.core.exceptions.RESTORE_MEMBER_OUTCOMES` (matched by identity —
    ``detail`` is the human line and is never matched). ``attempts`` counts the
    budgeted leg iterations consumed (0 for skips and pre-leg absorbing
    outcomes). ``new_native_token`` is the restore pointer a LANDED write
    minted (S3 versionId / the file's new coordinator version) — the Unit-5
    registration seam consumes it; ``None`` on every non-write path (a delete
    is recorded via ``deleted_at_restore``, its marker id is never a content
    pointer). ``resumed_from_prior_run`` marks a member whose terminal outcome
    a crashed run already recorded durably: reported, never re-driven.

    ``observation`` is what the leg SAW of the live state before writing (see
    :class:`RestoreObservation`) — additive, run-local, and never a comparand.
    It defaults to the no-write-attempted state, which is the truth at every
    converged, skipped and pre-write absorbing site: a member that never
    reached a write decision must not be reported as one whose observation was
    lost. ``not_recorded`` is set explicitly at the two sites that hold no
    observation: the rebuild of a terminal with no leg having run
    (:meth:`WorkspaceVersioner._outcome_from_durable_row`), and an object write
    whose outcome was lost and then reconciled (see
    :meth:`WorkspaceVersioner._observation_of_overwritten`).
    """

    member_path: str
    outcome: str
    attempts: int
    detail: str
    new_native_token: str | None = None
    deleted_at_restore: float | None = None
    resumed_from_prior_run: bool = False
    observation: RestoreObservation = OBSERVATION_NO_WRITE_ATTEMPTED


@dataclass(frozen=True)
class RestoreRegistration:
    """The registration step's terminal answer, per member class (WV Unit 5).

    ``status`` is one of the closed
    :data:`~ccs.core.exceptions.WORKSPACE_REGISTRATION_STATUSES` (identity-
    matched): ``committed`` (the all-or-nothing ``commit_all`` landed —
    ``registered`` maps each file member path to its NEW coordinator version),
    ``empty_write_set`` (nothing needed a commit; ``commit_all`` was never
    called), ``registered_by_prior_run`` (the durable ``registered`` marker —
    a crashed run already completed the step; nothing re-registered), or
    ``refused`` (the batch was HELD: NOTHING registered, ``refused`` maps each
    failing member path to its typed conflict reason — identity-matched wire
    constants; ``stale_read_generation`` is the fence rejecting a superseded
    controller's late apply, never retried).

    The per-member-class honesty surfaces: ``substrate_registered`` names the
    written S3 members whose registration is MANIFEST-SIDE by design (the
    substrate owns their identity — no coordinator artifact row is ever
    forced); ``deleted_recorded`` names the delete-leg members whose
    registration is their durable ``deleted_at_restore`` record; ``skipped``
    names file members already registered at the manifest fingerprint (the
    exactly-once filter). ``invalidated_peers`` counts the invalidation
    signals the commit returned (broadcast-after-commit: registered peers
    holding a member were atomically invalidated by the registry).

    Run-local detail, durable REBUILD: durable rows retain outcomes and the
    ``registered`` status marker, not this detail — a report rebuilt from
    durable rows of an already-``concluded`` checkpoint carries a REBUILT
    answer instead (:meth:`WorkspaceVersioner._registration_from_durable_state`
    re-reads the coordinator's registered state): ``empty_write_set`` /
    ``registered_by_prior_run`` where the durable truth backs them, or
    ``refused`` (with an empty per-member reason map — reasons are not durably
    retained; the identity-matched ``status`` is the signal) where written
    members are not at their manifest fingerprints. A refused-terminal
    registration must never re-read as ``None``-means-fine.
    """

    status: str
    detail: str
    registered: Mapping[str, int] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    substrate_registered: tuple[str, ...] = ()
    deleted_recorded: tuple[str, ...] = ()
    refused: Mapping[str, str] = field(default_factory=dict)
    invalidated_peers: int = 0
    attempts: int = 0


@dataclass(frozen=True)
class WorkspaceRestoreReport:
    """The complete per-member terminal report one restore run concludes with.

    Returned to the caller AND durably mirrored (each member's ``outcome`` in
    its registry row; ``status`` — always ``concluded`` — on the checkpoint
    header), so the same report is reconstructible from durable state alone.
    ``registration`` is the Unit-5 registration answer for THIS run, or —
    on a report rebuilt from durable rows (a re-restore of a ``concluded``
    checkpoint) — the answer REBUILT from the coordinator's registered state
    (never ``None`` on that path: a refused-terminal registration stays
    visible so downstream surfaces can exit non-zero).
    """

    checkpoint_id: str
    status: str
    members: tuple[MemberRestoreOutcome, ...]
    registration: RestoreRegistration | None = None

    @property
    def members_by_path(self) -> "dict[str, MemberRestoreOutcome]":
        """The report keyed by member path (paths are unique per manifest)."""
        return {m.member_path: m for m in self.members}


@dataclass(frozen=True)
class _ObjectLiveView:
    """What ONE versioned live read of an object member yielded.

    All three fields come from the SAME ``get_object`` response (SPLIT-COMPARAND,
    extended to the triple), and they are kept NAMED rather than positional
    because two of them are same-typed strings that must never be swapped:
    ``comparand`` is the ETag and the ONLY value that may arbitrate a write,
    while ``pointer`` is the versionId, which addresses a version and arbitrates
    nothing. ``fingerprint`` is the content digest the converged check already
    computes.

    ``pointer``/``fingerprint`` are ``None`` exactly on the create-on-absent
    path, where ``comparand`` is the :data:`CREATE_IF_ABSENT` sentinel and the
    read found no live state at all. A ``pointer`` present here is always a REAL
    versionId — the ``"null"``/absent case raises ``VersionPointerUnconfirmed``
    upstream rather than reaching this type — so no sentinel guard is needed on
    it, and none may be invented (a sentinel must never seed a comparand).
    """

    comparand: str
    pointer: str | None = None
    fingerprint: str | None = None


class _LegBudget:
    """One restore leg's bounded re-drive budget (the ``MAX_CAS_REACQUIRES`` twin).

    :meth:`try_consume` admits at most ``limit + 1`` iterations (the initial
    attempt plus ``limit`` re-drives) and then answers ``None`` forever — the
    caller's absorbing-``conflict`` signal. The increment AND the decision run
    in ONE ``threading.RLock`` critical section (the GIL-TOCTOU discipline: a
    check-then-increment split across the lock could admit an extra attempt
    when paths interleave at a bytecode boundary). The counter is RUN-LOCAL by
    design — the durable truth is the member's terminal ``restore_outcome``,
    so a crash-resumed run re-arms a fresh budget for a member that never
    reached one: each run is individually bounded, and no run can livelock.
    """

    def __init__(self, limit: int) -> None:
        self._lock = threading.RLock()
        self._limit = limit
        self._attempts = 0

    def try_consume(self) -> int | None:
        """Admit one leg iteration: its 1-based number, or ``None`` when spent."""
        with self._lock:
            if self._attempts >= self._limit + 1:
                return None
            self._attempts += 1
            return self._attempts

    @property
    def attempts(self) -> int:
        """Iterations consumed so far (read under the same lock)."""
        with self._lock:
            return self._attempts


# --- the engine -----------------------------------------------------------------


class WorkspaceVersioner:
    """Skew-declared checkpoint capture over heterogeneous workspace members.

    Declare members first (:meth:`add_file_member` / :meth:`add_object_member`
    / :meth:`add_forward_only_member` — member paths must be unique), then
    :meth:`checkpoint` captures, verifies, and persists ONE manifest through
    the coordinator service. Thread-safe: registration and capture serialize
    on one lock (a checkpoint is a single logical operation; two overlapping
    captures of one versioner would interleave their windows).

    ``clock`` defaults to :func:`~ccs.core.clock.monotonic_seconds` — the ONE
    coordinator tick basis; injectable for deterministic tests only.
    """

    def __init__(
        self,
        *,
        service: CheckpointPersistence,
        owner: UUID,
        clock: "Callable[[], int]" = monotonic_seconds,
        file_resolver: FileContentResolver | None = None,
    ) -> None:
        if owner is None:
            raise ValueError(
                "WorkspaceVersioner needs an owner: an ownerless manifest is "
                "unrepresentable (fail-closed; the registry enforces it too)"
            )
        self._service = service
        self._owner = owner
        self._clock = clock
        # The file-member history seam (restore only): pinned bytes for
        # (path, version). Capture never needs it; restore pre-flight requires
        # it for every actionable file member — fail-fast, before any status
        # write (Unit 5 wires the real coordinator-retention resolver).
        self._file_resolver = file_resolver
        self._lock = threading.Lock()
        self._members: list[_FileMember | _ObjectMember | _ForwardOnlyMember] = []

    # --- member registration --------------------------------------------------

    def add_file_member(self, source: FileMemberSource, path: str) -> None:
        """Declare one file member: ``path`` read through ``source``
        (a :class:`CoherentVolume` or anything speaking its
        ``read_with_version`` surface). ``path`` is the member's manifest key.
        """
        with self._lock:
            self._register(_FileMember(member_path=self._require_new_path(path), source=source))

    def add_object_member(
        self,
        binding: CoherentObject,
        key: str,
        *,
        member_path: str | None = None,
    ) -> None:
        """Declare one S3 object member: ``key`` read through ``binding``.

        ``member_path`` keys the member in the manifest; it is REQUIRED to be
        explicit or defaulted to ``s3://<key>`` — the bucket lives inside the
        binding, and the manifest key only needs to be unique + stable within
        this workspace.
        """
        path = member_path if member_path is not None else f"s3://{key}"
        with self._lock:
            self._register(
                _ObjectMember(
                    member_path=self._require_new_path(path), binding=binding, key=key
                )
            )

    def add_forward_only_member(self, member_path: str) -> None:
        """Declare one forward-only member (an action/effect surface).

        Enumerated in every manifest — the checkpoint DESCRIBES it so a
        restore can say "skipped, forward-only" per member — but never
        token-captured: there is no state to capture and nothing to compare.
        """
        with self._lock:
            self._register(_ForwardOnlyMember(member_path=self._require_new_path(member_path)))

    # --- the capture ----------------------------------------------------------

    def checkpoint(
        self, name: str, *, pin: bool = True, receiver: UUID | None = None
    ) -> WorkspaceCheckpoint:
        """Capture a named checkpoint: cut → verify → persist (one registration)
        → pin (the Unit-6 GC pin legs, on by default).

        ``receiver`` (#191, optional) names the one controller allowed to
        register a restore of this checkpoint — another versioner's ``owner``
        when the checkpoint is a handoff. ``None`` (the default) lets any
        controller restore it, as before; this versioner's ``owner`` is
        recorded as provenance either way and authorizes nothing.

        ``pin=True`` (the fail-closed default) runs the pin legs right after
        the manifest persists — module docstring, "Pins": an S3 hold lands
        (``pin_state="held"``) or the tier downgrades LOUDLY in the same
        registry write; a file member's retention pin is verified or the
        member downgrades loudly; the returned member rows are re-read from
        the registry so they carry the durable pin outcome. ``pin=False``
        captures only (a capture-only service seam, or a caller deliberately
        deferring to :meth:`pin_checkpoint`) — the persisted pair
        (``restorable``, ``unpinned``) is the documented
        claimed-but-not-yet-backed shape, never a guarantee.

        Raises :class:`BinaryFileMemberRefused` (typed, BEFORE any persist) for
        a non-UTF-8 file member, and :class:`CheckpointPersistFailed` when the
        coordinator/registry raises during the single-transaction registration
        (no partial manifest either way). ``ValueError`` for an empty member
        set or a blank name (nothing to checkpoint is a caller bug, not a
        cut); ``TypeError`` — BEFORE any capture read — when ``pin=True``
        needs pin legs this service cannot record (fail-fast, never a silent
        unpinned pass).
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("checkpoint needs a non-empty name")
        with self._lock:
            if not self._members:
                raise ValueError(
                    "checkpoint needs at least one declared member (an empty "
                    "workspace manifest would describe nothing)"
                )
            pin_store = self._require_pin_store_for_checkpoint() if pin else None
            rows = self._capture_all(name)
            rows = self._verify_window(rows)
            result = self._persist(name, rows, receiver=receiver)
            if pin_store is None:
                return result
            checkpoint_id = result.record.checkpoint_id
            self._pin_members(pin_store, checkpoint_id)
            # Return the DURABLE truth: the rows re-read from the registry
            # carry each member's pin outcome (held / the loud downgrade).
            return WorkspaceCheckpoint(
                record=result.record,
                members=tuple(
                    pin_store.get_workspace_checkpoint_members(checkpoint_id)
                ),
            )

    def pin_checkpoint(self, checkpoint_id: str) -> "tuple[CheckpointMember, ...]":
        """Re-drive the pin legs for a persisted checkpoint (idempotent).

        The retry half of the fold-in default: a run whose pin legs died
        mid-way (or a ``pin=False`` capture) is completed here. Only members
        still ``pin_state="unpinned"`` are attempted — ``held`` is already
        backed, ``pin_unavailable`` is a structural refusal (re-attempting
        cannot change the bucket's Object Lock configuration or resurrect a
        gone version), and ``released`` was deliberately dropped by the
        release path (``WorkspaceVersioner.release_checkpoint``, or the
        engine beneath it). Returns the refreshed durable member rows.

        Raises the typed :class:`~ccs.core.exceptions.CheckpointUnknown` for
        an unknown id, ``TypeError`` for a service without the pin surface,
        and ``ValueError`` (pre-flight, before any write) when a pin-eligible
        S3 member row has no declared binding on this versioner.
        """
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("pin_checkpoint needs a non-empty checkpoint id")
        store = self._require_pin_store()
        with self._lock:
            if store.get_workspace_checkpoint(checkpoint_id) is None:
                raise CheckpointUnknown(checkpoint_id)
            self._pin_members(store, checkpoint_id)
            return tuple(store.get_workspace_checkpoint_members(checkpoint_id))

    def release_checkpoint(self, checkpoint_id: str) -> "tuple[CheckpointMember, ...]":
        """Release the pins this checkpoint holds — ONE-WAY (idempotent).

        Every hold this checkpoint placed, and that no other checkpoint in
        this registry still relies on, is DROPPED; every ``held`` row (an S3
        legal hold and a file member's verification pin alike) becomes
        ``pin_state="released"``, which is TERMINAL —
        :meth:`pin_checkpoint` deliberately skips a released row, so nothing
        re-establishes the hold, and from that instant the captured version is
        eligible for lifecycle expiry AND for a version-targeted delete. The
        MANIFEST survives: this releases pins, it does not delete the
        checkpoint (there is no checkpoint-delete verb); the members stay
        described, tiered honestly ``restorable-unpinned``.

        PRECONDITION for S3 members: re-declare each object member on THIS
        versioner with :meth:`add_object_member` first — through the SAME
        binding and key that placed the hold. The pre-flight only checks that
        an object member is declared at that member path, never that it is the
        one that pinned the version, so a mismatched binding or key records
        the row ``released`` WITHOUT dropping the hold, silently (the drop's
        ``KeyError`` reads as "the version is gone, the hold is moot") —
        leaving an un-expirable version no later call here will free.

        A hold SHARED with another checkpoint survives until the LAST holder
        releases: S3's hold is a flag, not a counter, and the cross-checkpoint
        ``(member_path, native_token)`` scan is the counter. Two bounds on it. The
        scan walks THIS versioner's registry only, so a holder recorded
        elsewhere is invisible to it. And it only ever counts holds this
        system placed: ``set_legal_hold`` writes ON unconditionally and
        records nothing about whether a hold was already there, so a hold set
        by a person or another tool is indistinguishable from one of ours and
        this verb will drop it.

        Separately from those bounds, the window between the last re-check and
        the drop is covered by CONVERGING rather than by checking harder:
        after the drop this re-reads and puts the hold back when a peer
        claimed it in the gap. A failure to put it back is logged.

        Idempotent means REPEAT calls, not concurrent ones: this instance's
        lock does not serialize a second versioner or process releasing the
        same checkpoint, and two that both read the same ``held`` rows will
        both decrement the pin refcount — the registry fails closed with
        ``ValueError`` on whichever decrement would take the count below zero.
        That raise lands after the member's row is already ``released`` and
        after its substrate drop has been attempted, so the hold is already
        gone and what is left is refcount drift, not a stranded hold. The
        call still fails; treat it as "some members were released".

        Idempotent is NOT self-healing: the release RECORDS ``released``
        before it drops the substrate hold, so a crash or an untyped substrate
        error between the two strands a live hold on an already-terminal row.
        The recovery is
        :meth:`~ccs.adapters.coherent_object.CoherentObject.release_legal_hold`
        on the binding, by version. The remaining residuals (two processes
        releasing concurrently and both over-retaining; the refcount's
        separate registry write) are documented on
        :meth:`_release_checkpoint_pins`, the engine this delegates to.

        Raises ``ValueError`` from three places, and only the first two are
        before any write — do NOT read a ``ValueError`` as "nothing was
        released": a blank/non-string id (BEFORE any store access); a ``held``
        S3 row with no declared binding (pre-flight, before any write); and the
        refcount decrement failing closed under a concurrent release, which
        fires after that member is already recorded ``released`` AND after its
        hold has been dropped — so it aborts the rest of the checkpoint's
        members, leaving them ``held`` for a later call (see above).
        Also raises the typed
        :class:`~ccs.core.exceptions.CheckpointUnknown` for an unknown id, and
        ``TypeError`` for a service without the pin surface.
        """
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("release_checkpoint needs a non-empty checkpoint id")
        return self._release_checkpoint_pins(checkpoint_id)

    # --- capture pass ---------------------------------------------------------

    def _capture_all(self, name: str) -> list[CheckpointMember]:
        """Phase 1 — one capture read per member, in registration order."""
        rows: list[CheckpointMember] = []
        for member in self._members:
            captured_at = float(self._clock())
            if isinstance(member, _ForwardOnlyMember):
                # Enumerated, never token-captured: the weakest-claim defaults
                # (no-arbiter / forward_only) are already the member's truth.
                rows.append(
                    CheckpointMember(
                        member_path=member.member_path,
                        artifact_id=None,
                        native_token=None,
                        fingerprint=None,
                        captured_at=captured_at,
                    )
                )
                continue
            observed = self._observe(member, refuse_binary=True)
            rows.append(
                CheckpointMember(
                    member_path=member.member_path,
                    artifact_id=None,
                    native_token=observed.token,
                    fingerprint=observed.fingerprint,
                    captured_at=captured_at,
                    absent=observed.absent,
                    arbitration_tier=self._arbitration_tier(member),
                    restore_tier=derive_restore_tier(
                        versioned=observed.versioned, pinnable=observed.pinnable
                    ).value,
                )
            )
        return rows

    # --- verification pass (torn-cut detection) -------------------------------

    def _verify_window(self, rows: list[CheckpointMember]) -> list[CheckpointMember]:
        """Phase 2 — after the window closes, re-read every captured member.

        Any movement (presence, token, or fingerprint) marks THAT member
        ``dirty_during_window=True``. Forward-only members carry no state and
        are never verified. Conservative: a write landing after ``window_max``
        but before this re-read still flags (dirty = "not verified quiescent").

        Disclosed residual (the tail window): the flag covers movement
        observed between the capture read and THIS verification read — a
        foreign write landing AFTER the verify read but BEFORE the manifest
        persists is NOT flagged. Safe direction: the cut stays internally
        consistent (every manifested token/fingerprint comes from the capture
        reads, never from the tail), and restore never trusts the flag —
        every leg re-reads its live comparand — so the residual under-flags
        the window; it cannot corrupt the manifest or a restore.
        """
        verified: list[CheckpointMember] = []
        member_by_path = {m.member_path: m for m in self._members}
        for row in rows:
            member = member_by_path[row.member_path]
            if isinstance(member, _ForwardOnlyMember):
                verified.append(row)
                continue
            live = self._observe(member, refuse_binary=False)
            # A re-read the source refused observed nothing, so the member was
            # not verified quiescent, whatever the capture pass recorded.
            dirty = (
                not live.confirmed
                or live.absent != row.absent
                or live.token != row.native_token
                or live.fingerprint != row.fingerprint
            )
            if dirty:
                # Frozen rows are replaced, never mutated (the manifest-record
                # discipline); only the torn-cut flag moves.
                row = replace(row, dirty_during_window=True)
            verified.append(row)
        return verified

    # --- one member observation (shared by both passes) -----------------------

    def _observe(
        self, member: _FileMember | _ObjectMember, *, refuse_binary: bool
    ) -> _Observation:
        if isinstance(member, _FileMember):
            return self._observe_file(member, refuse_binary=refuse_binary)
        return self._observe_object(member)

    def _observe_file(self, member: _FileMember, *, refuse_binary: bool) -> _Observation:
        try:
            data, version = member.source.read_with_version(member.member_path)
        except FileNotFoundError:
            return _Observation(absent=True, token=None, fingerprint=None)
        except StaleView:
            # The source refused to pair the bytes on disk with a version: they
            # are not the content the coordinator records (a peer's commit still
            # reaching disk, or an out-of-band edit). No pointer can be
            # manifested, which is the unconfirmed-pointer case below (the
            # Sentinel rule): present, never above forward_only, and described
            # without a digest because the read returned no bytes.
            return _Observation(absent=False, token=None, fingerprint=None, confirmed=False)
        if refuse_binary:
            self._require_utf8_text(member.member_path, data)
        if version < 1:
            # Pointer UNCONFIRMED (no coordinator / degraded resolution): the
            # Sentinel rule — an unconfirmed pointer never lands in a manifest,
            # and the member can never claim a restore tier above forward_only.
            return _Observation(
                absent=False, token=None, fingerprint=_sha256_hex(data), versioned=False
            )
        # File members: coordinator retention holds the version history
        # (versioned=True) but no retention pin exists yet (Unit 6), so the
        # honest tier is restorable-unpinned — derive, never assert.
        return _Observation(
            absent=False,
            token=str(version),
            fingerprint=_sha256_hex(data),
            versioned=True,
            pinnable=False,
        )

    def _observe_object(self, member: _ObjectMember) -> _Observation:
        try:
            read = member.binding.read_versioned(member.key)
        except KeyError:
            return _Observation(absent=True, token=None, fingerprint=None)
        except VersionPointerUnconfirmed:
            # Unversioned bucket — the typed refusal IS the discovery (no
            # pre-probe). The member is still DESCRIBED (fingerprint from a
            # second consistent read) but holds no pointer and can never be
            # restorable: derive_restore_tier(versioned=False) → forward_only.
            try:
                data, _etag = member.binding.read(member.key)
            except KeyError:
                return _Observation(absent=True, token=None, fingerprint=None)
            return _Observation(
                absent=False, token=None, fingerprint=_sha256_hex(data), versioned=False
            )
        # Versioned bucket: the versionId is the manifest pointer (the ETag —
        # the CAS comparand — is deliberately NOT manifested; a restore leg
        # re-reads it live). S3 offers a per-version pin (Object Lock legal
        # hold), so pinnable=True → restorable; the Unit-6 pin leg establishes
        # the hold at capture and downgrades LOUDLY (restorable-unpinned via
        # set_checkpoint_member_pin) where the bucket offers no lock.
        return _Observation(
            absent=False,
            token=read.version_id,
            fingerprint=_sha256_hex(read.data),
            versioned=True,
            pinnable=True,
        )

    # --- persist (one registration) -------------------------------------------

    def _persist(
        self,
        name: str,
        rows: list[CheckpointMember],
        *,
        receiver: UUID | None = None,
    ) -> WorkspaceCheckpoint:
        window_min = min(row.captured_at for row in rows)
        window_max = max(row.captured_at for row in rows)
        # Passed only when set, so a persist seam predating #191 keeps working
        # for every checkpoint that names no receiver.
        extra: dict[str, Any] = {} if receiver is None else {"receiver": receiver}
        try:
            record = self._service.create_workspace_checkpoint(
                name=name,
                owner=self._owner,
                members=rows,
                window_min=window_min,
                window_max=window_max,
                issued_at_tick=int(self._clock()),
                **extra,
            )
        except Exception as exc:
            raise CheckpointPersistFailed(
                f"checkpoint {name!r} was NOT persisted: the coordinator "
                f"registration raised ({type(exc).__name__}). The registration "
                "is a single transaction, so no partial manifest exists — "
                "retry once the coordinator is reachable."
            ) from exc
        return WorkspaceCheckpoint(record=record, members=tuple(rows))

    # --- the pin legs (WV Unit 6 / R9) ----------------------------------------

    def _require_pin_store(self) -> CheckpointPinStore:
        if not isinstance(self._service, CheckpointPinStore):
            raise TypeError(
                "pin legs need a service speaking the CheckpointPinStore "
                "surface (checkpoint read + durable pin state + refcounts); "
                f"this service ({type(self._service).__name__}) does not — a "
                "capture-only seam cannot record a pin outcome, and an "
                "unrecorded pin would be the silent-decay lie"
            )
        return self._service

    def _require_pin_store_for_checkpoint(self) -> CheckpointPinStore | None:
        """The fold-in's fail-fast gate, BEFORE any capture read.

        A service speaking the pin surface is used as-is. One that does not is
        tolerated ONLY when no declared member could need a pin leg (no S3
        member, and no file member unless a resolver was injected) — anything
        else raises ``TypeError`` here rather than persisting a manifest whose
        ``restorable`` claim nothing can back (pass ``pin=False`` explicitly
        for a capture-only run; the unpinned pair is then the caller's
        documented choice, not an accident).
        """
        if isinstance(self._service, CheckpointPinStore):
            return self._service
        needs_pins = any(isinstance(m, _ObjectMember) for m in self._members) or (
            self._file_resolver is not None
            and any(isinstance(m, _FileMember) for m in self._members)
        )
        if needs_pins:
            self._require_pin_store()  # raises the descriptive TypeError
        return None

    def _pin_members(self, store: CheckpointPinStore, checkpoint_id: str) -> None:
        """Drive one pin leg per eligible DURABLE member row (idempotent).

        Eligible = present, pointer manifested, and still
        ``pin_state="unpinned"`` — ``held`` is already backed,
        ``pin_unavailable`` is structural, ``released`` was deliberate. Each
        leg ends in exactly one durable answer: ``held`` (+1 refcount) or the
        loud ``pin_unavailable`` downgrade (pin state and tier in ONE registry
        write). Operational failures outside the typed surface propagate — the
        member then still reads (``restorable``, ``unpinned``) =
        claimed-but-not-backed, and :meth:`pin_checkpoint` re-drives it.
        """
        rows = store.get_workspace_checkpoint_members(checkpoint_id)
        declared = {m.member_path: m for m in self._members}
        eligible = [row for row in rows if self._pin_eligible(row)]
        problems = [
            f"{row.member_path}: no declared S3 object member binding on this "
            "versioner (re-declare the member before pin_checkpoint)"
            for row in eligible
            if row.arbitration_tier == ArbitrationTier.NATIVE_CAS.value
            and not isinstance(declared.get(row.member_path), _ObjectMember)
        ]
        if problems:
            raise ValueError(
                "pin pre-flight failed (nothing was pinned): " + "; ".join(problems)
            )
        for row in eligible:
            if row.arbitration_tier == ArbitrationTier.NATIVE_CAS.value:
                member = declared[row.member_path]
                assert isinstance(member, _ObjectMember)  # pre-flight guaranteed
                self._pin_object_leg(store, checkpoint_id, row, member)
            else:
                self._pin_file_leg(store, checkpoint_id, row)

    def _pin_eligible(self, row: CheckpointMember) -> bool:
        """One pin-attempt admission rule for both member kinds."""
        if row.absent or row.native_token is None:
            return False  # nothing to pin: no captured state / no pointer
        if row.pin_state != PIN_STATE_UNPINNED:
            return False  # held is backed; unavailable/released are terminal
        if (
            row.arbitration_tier == ArbitrationTier.NATIVE_CAS.value
            and row.restore_tier == RestoreTier.RESTORABLE.value
        ):
            return True  # the S3 legal-hold leg
        return (
            row.arbitration_tier == ArbitrationTier.NO_ARBITER.value
            and row.restore_tier == RestoreTier.RESTORABLE_UNPINNED.value
            and self._file_resolver is not None
        )  # the file verification leg (no resolver -> nothing claimed to back)

    def _pin_object_leg(
        self,
        store: CheckpointPinStore,
        checkpoint_id: str,
        row: CheckpointMember,
        member: _ObjectMember,
    ) -> None:
        """S3: establish the substrate's own per-version pin (legal hold ON)."""
        assert row.native_token is not None  # eligibility guaranteed
        try:
            member.binding.set_legal_hold(member.key, version_id=row.native_token)
        except LegalHoldUnavailable:
            # No Object Lock configuration: the pin can NEVER be established on
            # this bucket — the loud, durable downgrade (tier + pin state in
            # one write), never a silent restorable claim.
            store.set_workspace_checkpoint_member_pin(
                checkpoint_id,
                row.member_path,
                pin_state=PIN_STATE_UNAVAILABLE,
                restore_tier=RestoreTier.RESTORABLE_UNPINNED.value,
            )
            return
        except (KeyError, VersionPointerUnconfirmed):
            # The pinned version is already gone (raced lifecycle/delete) or
            # the pointer axis broke: nothing left to hold — downgrade loudly;
            # a later restore reports target_lost for it.
            store.set_workspace_checkpoint_member_pin(
                checkpoint_id,
                row.member_path,
                pin_state=PIN_STATE_UNAVAILABLE,
                restore_tier=RestoreTier.RESTORABLE_UNPINNED.value,
            )
            return
        store.set_workspace_checkpoint_member_pin(
            checkpoint_id, row.member_path, pin_state=PIN_STATE_HELD
        )
        store.adjust_workspace_checkpoint_pin_refcount(checkpoint_id, +1)

    def _pin_file_leg(
        self, store: CheckpointPinStore, checkpoint_id: str, row: CheckpointMember
    ) -> None:
        """File: v1's VERIFICATION pin over the coordinator's declared retention.

        The captured version's retained bytes are read through the resolver
        seam and checked against the captured fingerprint. Verified → ``held``
        with the tier kept at ``restorable-unpinned`` (coordinator retention
        offers no per-version hold in v1 — the R13 disclosure); retention off
        / the version not retained / a fingerprint mismatch → the captured
        state is ALREADY unreachable, so the member downgrades LOUDLY to
        ``forward_only`` (a restore can only describe it and skip).
        """
        resolver = self._file_resolver
        assert resolver is not None and row.native_token is not None  # eligibility
        try:
            retained = resolver.content_at(row.member_path, int(row.native_token))
        except KeyError:
            store.set_workspace_checkpoint_member_pin(
                checkpoint_id,
                row.member_path,
                pin_state=PIN_STATE_UNAVAILABLE,
                restore_tier=RestoreTier.FORWARD_ONLY.value,
            )
            return
        if _sha256_hex(retained) != row.fingerprint:
            store.set_workspace_checkpoint_member_pin(
                checkpoint_id,
                row.member_path,
                pin_state=PIN_STATE_UNAVAILABLE,
                restore_tier=RestoreTier.FORWARD_ONLY.value,
            )
            return
        store.set_workspace_checkpoint_member_pin(
            checkpoint_id, row.member_path, pin_state=PIN_STATE_HELD
        )
        store.adjust_workspace_checkpoint_pin_refcount(checkpoint_id, +1)

    def _release_checkpoint_pins(
        self, checkpoint_id: str
    ) -> "tuple[CheckpointMember, ...]":
        """Release every pin this checkpoint holds (idempotent).

        The engine behind the public :meth:`release_checkpoint`, which adds
        the blank/non-string id guard and the caller-facing docstring. Still
        underscore-private and still with NO CLI or HTTP route (the bindings
        carry credentials); checkpoint DELETION remains absent, and the GC
        drop-half is v2.

        Per ``held`` member, in the fail-closed order: (1) record
        ``pin_state="released"`` — downgrading a ``restorable`` tier to
        ``restorable-unpinned`` in the SAME write, so no instant leaves an
        unbacked ``restorable`` claim; (2) for an S3 member, drop the legal
        hold ONLY when no OTHER checkpoint still holds a ``held`` pin on the
        same ``(member_path,
        native_token)`` — the cross-checkpoint scan: a shared hold survives
        until the LAST holder releases (S3's hold is a flag, not a counter;
        the scan is the counter). The scan runs TWICE, fresh both times: once
        as the drop's admission check and once as the LAST-INSTANT re-check
        immediately before the substrate ``release_legal_hold`` call — a
        concurrent ``pin_checkpoint`` (another versioner/process; this lock
        does not serialize it) that recorded ``held`` on the same identity
        since the first scan is seen there and the drop is SKIPPED, failing
        toward over-retention (safe: the new holder's own release drops the
        hold last-out); and (3) decrement the checkpoint's pin refcount —
        LAST, because that write fails closed below zero under a concurrent
        release, and ordered before the drop it aborted with the row already
        terminal and the hold still ON.

        A crash between (1) and (2) leaves an over-retained hold — the safe
        direction. Where a sharing peer exists its own release still drops it
        (its scan sees this row as ``released``); where this checkpoint was
        the SOLE holder nothing in the product drops it, and the recovery is
        :meth:`~ccs.adapters.coherent_object.CoherentObject.release_legal_hold`
        on the binding, by version — the released row keeps its
        ``native_token`` for exactly that.

        Documented residuals: two checkpoints releasing CONCURRENTLY from
        different processes can each see the other still ``held`` and both
        skip the drop (over-retention, never data loss); a concurrent pin
        whose ``held`` row lands AFTER the last-instant re-check but BEFORE
        the drop is no longer lost outright: the drop CONVERGES, re-reading
        afterwards and re-placing the hold when a peer claimed it in that gap
        (``set_legal_hold`` is idempotent, and re-placing one nobody needs is
        over-retention, the safe direction). The residual is now the narrower
        case where that re-place itself fails, which is logged; closing even
        that needs a registry-side transactional claim spanning
        check-and-drop, outside v1's registry surface. The refcount write is a
        separate registry write from the pin record and lands AFTER the drop,
        so neither a crash nor a fail-closed refcount between them can strand
        a hold — what is left is benign bookkeeping drift.
        Cross-checkpoint identity is ``(member_path, native_token)`` — the
        bucket lives binding-side (real S3 version ids are unique; a
        cross-bucket collision is a documented residual of the fake's shape,
        not the engine's).

        Returns the refreshed durable rows. Raises
        :class:`~ccs.core.exceptions.CheckpointUnknown` for an unknown id,
        ``TypeError`` for a service without the pin surface, ``ValueError``
        (pre-flight, before any write) when a held S3 row has no declared
        binding.
        """
        store = self._require_pin_store()
        with self._lock:
            if store.get_workspace_checkpoint(checkpoint_id) is None:
                raise CheckpointUnknown(checkpoint_id)
            rows = store.get_workspace_checkpoint_members(checkpoint_id)
            declared = {m.member_path: m for m in self._members}
            held = [row for row in rows if row.pin_state == PIN_STATE_HELD]
            problems = [
                f"{row.member_path}: no declared S3 object member binding on "
                "this versioner (re-declare the member before releasing)"
                for row in held
                if row.arbitration_tier == ArbitrationTier.NATIVE_CAS.value
                and not isinstance(declared.get(row.member_path), _ObjectMember)
            ]
            if problems:
                raise ValueError(
                    "release pre-flight failed (nothing was released): "
                    + "; ".join(problems)
                )
            for row in held:
                downgrade = (
                    RestoreTier.RESTORABLE_UNPINNED.value
                    if row.restore_tier == RestoreTier.RESTORABLE.value
                    else None
                )
                store.set_workspace_checkpoint_member_pin(
                    checkpoint_id,
                    row.member_path,
                    pin_state=PIN_STATE_RELEASED,
                    restore_tier=downgrade,
                )
                if (
                    row.arbitration_tier != ArbitrationTier.NATIVE_CAS.value
                    or row.native_token is None
                ):
                    pass  # the file verification pin has no substrate half
                elif self._pin_shared_elsewhere(store, checkpoint_id, row):
                    pass  # another checkpoint's hold: survives (last-out drops)
                else:
                    member = declared[row.member_path]
                    assert isinstance(member, _ObjectMember)  # pre-flight guaranteed
                    if self._pin_shared_elsewhere(store, checkpoint_id, row):
                        # LAST-INSTANT re-check (fresh registry read): a
                        # concurrent pin_checkpoint recorded ``held`` on this
                        # (member_path, native_token) since the scan above —
                        # dropping now would strip the hold the sibling just
                        # claimed (under-retention, the UNSAFE direction).
                        # Skip: over-retention is the safe direction, and the
                        # new holder's own release drops it last-out.
                        pass
                    else:
                        try:
                            member.binding.release_legal_hold(
                                member.key, version_id=row.native_token
                            )
                        except (KeyError, LegalHoldUnavailable) as exc:
                            # The hold is moot — the version (or the lock
                            # configuration) is gone; the release's goal (no
                            # dangling hold) already stands, and the record
                            # above is the truth. EXCEPT when the member was
                            # re-declared through a binding or key that does
                            # not address the pinned version: the pre-flight
                            # cannot tell those apart (it only asks that SOME
                            # object member is declared at the path), the hold
                            # is then still ON, and the caller gets no
                            # exception and no changed row. This line is the
                            # only signal either way.
                            logger.warning(
                                "checkpoint pin release could not reach the "
                                "held version, treating the hold as moot: "
                                "checkpoint=%s member=%s version=%s cause=%r "
                                "— if this member was re-declared through a "
                                "different binding or key, the hold is STILL ON",
                                checkpoint_id,
                                row.member_path,
                                row.native_token,
                                exc,
                            )
                        except Exception:
                            # Record-before-drop already made this row
                            # terminal, so no later release_checkpoint or
                            # pin_checkpoint walks it: this line is the only
                            # record of WHICH version was stranded, and that
                            # version id is exactly what the documented
                            # recovery needs. Re-raise — absorbing it here
                            # would hide the strand behind a clean return.
                            logger.error(
                                "checkpoint pin release STRANDED a live hold: "
                                "checkpoint=%s member=%s version=%s — the row "
                                "is already terminal, so recover out of band "
                                "with CoherentObject.release_legal_hold on the "
                                "binding, by version",
                                checkpoint_id,
                                row.member_path,
                                row.native_token,
                            )
                            raise
                        else:
                            self._reclaim_hold_taken_during_drop(
                                store, checkpoint_id, row, member
                            )
                # The refcount decrement is a SEPARATE registry write and it
                # lands AFTER the substrate attempt on purpose. It fails CLOSED
                # below zero, which is exactly what a second releaser running
                # concurrently causes — and ordered BEFORE the drop, that raise
                # aborted with this row already terminal and its hold still ON,
                # permanently, because nothing walks a released row again.
                # Ordered here the same raise leaves only refcount drift, which
                # this engine already documents as benign. Record-before-drop,
                # the load-bearing order, is unchanged; only bookkeeping moved.
                store.adjust_workspace_checkpoint_pin_refcount(checkpoint_id, -1)
            return tuple(store.get_workspace_checkpoint_members(checkpoint_id))

    def _reclaim_hold_taken_during_drop(
        self,
        store: CheckpointPinStore,
        checkpoint_id: str,
        row: CheckpointMember,
        member: "_ObjectMember",
    ) -> None:
        """Put back a hold a peer claimed inside the drop window (CONVERGE).

        The last-instant re-check is a registry read and the drop is a separate
        substrate call, so a concurrent ``pin_checkpoint`` can record ``held``
        between them and lose the hold it just took — under-retention, the one
        direction this engine refuses everywhere else. No re-check can close
        that gap; re-reading AFTER the drop can. Every outcome here is
        acceptable: the peer's claim gets backed again, or this fails and the
        result is exactly what it was without the call (plus a record), or it
        re-places a hold nobody needs, which is over-retention — the safe
        direction. ``set_legal_hold`` is idempotent, so re-placing a live hold
        is a no-op.

        Deliberately does not raise. The release itself already succeeded, and
        the state without this call is the state with it failing, so raising
        would turn a narrowed race into a hard error.
        """
        if not self._pin_shared_elsewhere(store, checkpoint_id, row):
            return
        try:
            member.binding.set_legal_hold(member.key, version_id=row.native_token)
        except Exception:
            logger.error(
                "checkpoint pin release dropped a hold a concurrent pin had "
                "just claimed, and could not put it back: checkpoint=%s "
                "member=%s version=%s — the peer checkpoint reads ``held`` "
                "with nothing behind it; re-pin it or treat it as "
                "restorable-unpinned",
                checkpoint_id,
                row.member_path,
                row.native_token,
            )
        else:
            logger.warning(
                "checkpoint pin release re-placed a hold a concurrent pin "
                "claimed inside the drop window: checkpoint=%s member=%s "
                "version=%s",
                checkpoint_id,
                row.member_path,
                row.native_token,
            )

    @staticmethod
    def _pin_shared_elsewhere(
        store: CheckpointPinStore, checkpoint_id: str, row: CheckpointMember
    ) -> bool:
        """True iff ANOTHER checkpoint holds a ``held`` pin on the same
        ``(member_path, native_token)`` — the cross-checkpoint refcount scan
        (cheap: checkpoints are few and the read is registry-local)."""
        for record in store.list_workspace_checkpoints():
            if record.checkpoint_id == checkpoint_id:
                continue
            for other in store.get_workspace_checkpoint_members(record.checkpoint_id):
                if (
                    other.pin_state == PIN_STATE_HELD
                    and other.member_path == row.member_path
                    and other.native_token == row.native_token
                ):
                    return True
        return False

    # --- the restore (WV Unit 4 / R3) -----------------------------------------

    def restore(self, checkpoint_id: str) -> WorkspaceRestoreReport:
        """Restore a checkpoint: one conditional leg per durable member row,
        under the TERMINATION CONTRACT (module docstring, "Restore").

        A restore that starts always CONCLUDES: every member reaches exactly
        one terminal outcome from the closed
        :data:`~ccs.core.exceptions.RESTORE_MEMBER_OUTCOMES` vocabulary, the
        checkpoint's ``restore_status`` moves ``in_progress`` →
        (``registered`` on a committed/empty registration step) →
        ``concluded``, and the frozen :class:`WorkspaceRestoreReport` —
        including the Unit-5 ``registration`` answer — is returned AND durably
        mirrored. Per-member failures are ABSORBED into the report — the only
        raises are pre-flight, before any status write: the typed
        :class:`~ccs.core.exceptions.CheckpointUnknown` for an unknown id, the
        typed :class:`~ccs.core.exceptions.CheckpointRegistrationRefused`
        (#191) when the checkpoint names a receiver that is not this
        versioner's ``owner`` (``not_the_receiver``) or another controller
        already registered it (``already_registered``),
        ``TypeError`` for a service that lacks the restore surface, and
        ``ValueError`` for missing member bindings / a missing file resolver
        (caller misconfiguration must never mint an ``in_progress`` record it
        cannot drive).

        Termination-contract boundary (the unanticipated-exception classes): a
        DETERMINISTIC member/environment pathology escaping a leg — the OSError
        family (the member path replaced by a directory, permission denied)
        minus the transient transport shapes, plus ``UnicodeDecodeError`` —
        and a :class:`StructuralMemberRefused` (the source declaring the member
        undrivable through its own surface) are ABSORBED into that member's
        terminal ``target_lost`` (re-driving can never succeed; see
        :meth:`_drive_member_absorbing`), so the restore still concludes.
        Everything else (``ConnectionError`` / ``TimeoutError`` transport
        blips, registry raises, genuine bugs) PROPAGATES by design — the
        crash-resume path: ``restore_status`` stays ``in_progress`` durably
        and a later ``restore()`` resumes it idempotently.

        Crash-resume: a checkpoint found ``in_progress`` is RESUMED — members
        whose durable ``restore_outcome`` is already terminal are skipped
        (reported ``resumed_from_prior_run``), the rest re-driven idempotently
        (a live state already matching the manifest concludes ``converged``
        with NO write — no double-apply). A ``concluded`` checkpoint returns
        its report rebuilt from durable rows, driving nothing — including a
        REBUILT registration answer (re-read from the coordinator's registered
        state, never ``None``): a registration that concluded ``refused`` (the
        fence rejecting a superseded controller, or budget exhaustion) is
        TERMINAL — the re-restore REPORTS the refusal; it never re-attempts
        the registration and never calls ``commit_all`` on this path.

        Disclosed single-controller assumption: two CONCURRENT restores of
        DIFFERENT checkpoints over an overlapping member are not
        cross-serialized — each leg CASes only against the live state it
        read, so both runs can honestly report ``restored`` for the member
        while only the LAST write survives live. The engine serializes its
        OWN runs (one lock per versioner); cross-controller ordering is the
        operator's contract in v1.
        """
        if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
            raise ValueError("restore needs a non-empty checkpoint id")
        store = self._require_restore_store()
        with self._lock:
            return self._restore_locked(store, checkpoint_id)

    def _restore_locked(
        self, store: CheckpointRestoreStore, checkpoint_id: str
    ) -> WorkspaceRestoreReport:
        record = store.get_workspace_checkpoint(checkpoint_id)
        if record is None:
            raise CheckpointUnknown(checkpoint_id)
        self._require_registrable_by_owner(record)
        rows = store.get_workspace_checkpoint_members(checkpoint_id)
        self._require_known_restore_state(record, rows)
        if record.restore_status == RESTORE_STATUS_CONCLUDED:
            # Idempotent re-conclude: the durable rows ARE the report. The
            # run-local registration detail is not durably retained, so the
            # registration answer is REBUILT from the coordinator's registered
            # state — a refused-terminal registration must surface here, never
            # a silent registration=None the caller reads as fine.
            return self._report_from_durable_rows(store, checkpoint_id, rows)
        # The durable idempotency marker (Unit 5): a crashed run that already
        # completed its registration step must never re-register on resume.
        prior_registration = record.restore_status == RESTORE_STATUS_REGISTERED
        pending = [row for row in rows if row.restore_outcome is None]
        self._require_drivable(pending)
        store.set_workspace_checkpoint_restore_status(
            checkpoint_id, RESTORE_STATUS_IN_PROGRESS, updated_at=float(self._clock())
        )
        outcomes: list[MemberRestoreOutcome] = []
        for row in rows:
            if row.restore_outcome is not None:
                # Terminal from a prior (crashed) run: skip, never re-drive.
                outcomes.append(self._outcome_from_durable_row(row))
                continue
            outcome = self._drive_member_absorbing(row)
            # Durable BEFORE the next member: a crash after this write leaves a
            # terminal row the resuming run skips; a crash before it leaves a
            # pending row the resuming run re-drives idempotently.
            store.set_workspace_checkpoint_member_restore(
                checkpoint_id,
                row.member_path,
                restore_outcome=outcome.outcome,
                deleted_at_restore=outcome.deleted_at_restore,
            )
            outcomes.append(outcome)
        registration = self._registration_seam(
            store, checkpoint_id, rows, outcomes, prior_run=prior_registration
        )
        if registration.status in (
            WORKSPACE_REGISTRATION_COMMITTED,
            WORKSPACE_REGISTRATION_EMPTY,
        ):
            # The durable marker: a crash between here and ``concluded`` makes
            # the resuming run skip the (already answered) registration step.
            # A REFUSED registration skips the marker — NOT as an invitation to
            # retry: REFUSED is TERMINAL (the very next write stamps
            # ``concluded``, which is absorbing — a re-restore REBUILDS and
            # REPORTS the refusal from durable state, never re-attempts it).
            # The skipped marker matters only in the narrow crash window
            # BEFORE the ``concluded`` write lands: that resume re-runs the
            # seam once more, where the fence rejects a superseded controller
            # again (never retried past it). PRIOR_RUN already holds the
            # marker.
            store.set_workspace_checkpoint_restore_status(
                checkpoint_id,
                RESTORE_STATUS_REGISTERED,
                updated_at=float(self._clock()),
            )
        store.set_workspace_checkpoint_restore_status(
            checkpoint_id, RESTORE_STATUS_CONCLUDED, updated_at=float(self._clock())
        )
        return WorkspaceRestoreReport(
            checkpoint_id=checkpoint_id,
            status=RESTORE_STATUS_CONCLUDED,
            members=tuple(outcomes),
            registration=registration,
        )

    def _registration_seam(
        self,
        store: CheckpointRestoreStore,
        checkpoint_id: str,
        rows: Sequence[CheckpointMember],
        outcomes: Sequence[MemberRestoreOutcome],
        *,
        prior_run: bool,
    ) -> RestoreRegistration:
        """THE UNIT-5 REGISTRATION STEP — after all-terminal, before ``concluded``.

        Classifies every terminal outcome into the plan's registration split
        (module docstring, "Registration") and drives the coordinator half:

        - written FILE members (``restored``, no delete record, ``no-arbiter``
          tier) → ONE ``register_workspace_restore`` call → one all-or-nothing
          hash-only ``commit_all``;
        - written S3 members → ``substrate_registered`` (manifest-side by
          design — no coordinator artifact identity is ever forced);
        - delete legs → ``deleted_recorded`` (their durable
          ``deleted_at_restore`` record IS the registration);
        - empty commit write-set → still ONE ``register_workspace_restore``
          call (#191: registering nothing claims the checkpoint all the
          same), answered typed EMPTY; ``commit_all`` never runs.

        Bounded re-drive (the leg-budget twin): a HELD batch whose reasons are
        all retry-eligible (``version_mismatch`` — a live registered writer
        moved a member between the comparand read and the batch —
        or ``other_holder``) re-drives from fresh comparands at most
        :data:`MAX_RESTORE_LEG_REDRIVES` times; ``stale_read_generation``
        (the fence: this controller was superseded by a sweep reclamation) is
        TERMINAL on sight — a superseded controller's late apply must never
        retry its way past the fence. Exhaustion and fence rejections conclude
        as ``refused`` (nothing registered, all-or-nothing); the restore still
        CONCLUDES — the refusal is reported, never raised.
        """
        writes: list[WorkspaceRestoreWrite] = []
        substrate_registered: list[str] = []
        deleted_recorded: list[str] = []
        row_by_path = {row.member_path: row for row in rows}
        for outcome in outcomes:
            if outcome.deleted_at_restore is not None:
                deleted_recorded.append(outcome.member_path)
                continue
            if outcome.outcome != RESTORE_OUTCOME_RESTORED:
                continue
            row = row_by_path[outcome.member_path]
            if row.arbitration_tier == ArbitrationTier.NATIVE_CAS.value:
                substrate_registered.append(outcome.member_path)
                continue
            if row.fingerprint is None:  # defensive: a written member has one
                continue
            writes.append(
                WorkspaceRestoreWrite(
                    member_path=outcome.member_path, fingerprint=row.fingerprint
                )
            )
        if prior_run:
            return RestoreRegistration(
                status=WORKSPACE_REGISTRATION_PRIOR_RUN,
                detail=(
                    "the durable 'registered' marker is set: a prior (crashed) "
                    "run already completed the registration step — nothing "
                    "re-registered (exactly-once)"
                ),
                substrate_registered=tuple(substrate_registered),
                deleted_recorded=tuple(deleted_recorded),
            )
        # An empty write-set still goes to the service (#191): registering
        # nothing claims the checkpoint all the same, so another owner's
        # restore of it is refused already_registered whether or not this run
        # had bytes to write. The service answers empty_write_set without
        # calling commit_all.
        return self._drive_registration(
            store,
            checkpoint_id,
            writes,
            substrate_registered=tuple(substrate_registered),
            deleted_recorded=tuple(deleted_recorded),
        )

    def _drive_registration(
        self,
        store: CheckpointRestoreStore,
        checkpoint_id: str,
        writes: "list[WorkspaceRestoreWrite]",
        *,
        substrate_registered: tuple[str, ...],
        deleted_recorded: tuple[str, ...],
    ) -> RestoreRegistration:
        """The bounded registration re-drive loop (see ``_registration_seam``)."""
        budget = _LegBudget(MAX_RESTORE_LEG_REDRIVES)
        last_refused: "dict[str, str]" = {}
        while True:
            if budget.try_consume() is None:
                return RestoreRegistration(
                    status=WORKSPACE_REGISTRATION_REFUSED,
                    detail=(
                        f"registration re-drive budget exhausted "
                        f"({budget.attempts} attempts) under sustained "
                        "registered-writer contention — NOTHING registered "
                        "(all-or-nothing)"
                    ),
                    substrate_registered=substrate_registered,
                    deleted_recorded=deleted_recorded,
                    refused=dict(last_refused),
                    attempts=budget.attempts,
                )
            try:
                result = store.register_workspace_restore(
                    checkpoint_id=checkpoint_id,
                    controller=self._owner,
                    writes=tuple(writes),
                    issued_at_tick=int(self._clock()),
                )
            except OccCallerTransientError:
                # The controller was invalidated mid-flight (a peer's commit
                # left it mid-transient): retry-eligible, budget-bounded.
                continue
            except CheckpointRegistrationRefused as exc:
                # #191, past the pre-flight: a concurrent controller claimed
                # the checkpoint first, or the write-set disagrees with the
                # manifest. Terminal — re-driving cannot change either — and
                # nothing was registered. The restore still CONCLUDES.
                paths = exc.member_paths or tuple(w.member_path for w in writes)
                return RestoreRegistration(
                    status=WORKSPACE_REGISTRATION_REFUSED,
                    detail=f"{exc} — not retried",
                    substrate_registered=substrate_registered,
                    deleted_recorded=deleted_recorded,
                    refused={path: exc.reason for path in paths},
                    attempts=budget.attempts,
                )
            if result.status in (
                WORKSPACE_REGISTRATION_COMMITTED,
                WORKSPACE_REGISTRATION_EMPTY,
            ):
                return RestoreRegistration(
                    status=result.status,
                    detail=result.detail,
                    registered=dict(result.versions),
                    skipped=result.skipped,
                    substrate_registered=substrate_registered,
                    deleted_recorded=deleted_recorded,
                    invalidated_peers=len(result.signals),
                    attempts=budget.attempts,
                )
            # REFUSED (all-or-nothing HELD; nothing mutated). Reasons are
            # identity-matched wire constants from ConflictDetail.
            last_refused = {
                path: conflict.reason for path, conflict in result.refused.items()
            }
            if STALE_READ_GENERATION_REASON in last_refused.values():
                # The read-generation fence: THIS controller was superseded by
                # a sweep reclamation — its late apply is rejected and must
                # never be retried into landing (the fence's whole point).
                return RestoreRegistration(
                    status=WORKSPACE_REGISTRATION_REFUSED,
                    detail=(
                        "the read-generation fence rejected this superseded "
                        "controller's late apply — NOTHING registered "
                        "(all-or-nothing); not retried"
                    ),
                    substrate_registered=substrate_registered,
                    deleted_recorded=deleted_recorded,
                    refused=dict(last_refused),
                    attempts=budget.attempts,
                )
            # version_mismatch / other_holder: re-drive from fresh comparands.

    # --- restore pre-flight (fail-fast, before any status write) --------------

    def _require_registrable_by_owner(self, record: CheckpointRecord) -> None:
        """Refuse a restore this versioner's registration could never land
        (#191) — BEFORE any status write or member leg, so a refused restore
        writes no bytes.

        The two controller refusals of
        ``CoordinatorService.register_workspace_restore``, read from the
        header: the checkpoint names a receiver that is not this versioner's
        ``owner`` (``not_the_receiver``), or another controller already
        registered it (``already_registered``, naming nobody). A concurrent
        restore that claims the checkpoint between this read and this run's
        registration is caught by the service's atomic claim instead, and
        concludes ``refused`` (see :meth:`_drive_registration`).
        """
        if record.receiver is not None and record.receiver != self._owner:
            raise CheckpointRegistrationRefused(
                record.checkpoint_id, CHECKPOINT_NOT_THE_RECEIVER_REASON
            )
        if record.registered_by is not None and record.registered_by != self._owner:
            raise CheckpointRegistrationRefused(
                record.checkpoint_id, CHECKPOINT_ALREADY_REGISTERED_REASON
            )

    def _require_restore_store(self) -> CheckpointRestoreStore:
        if not isinstance(self._service, CheckpointRestoreStore):
            raise TypeError(
                "restore needs a service speaking the CheckpointRestoreStore "
                "surface (checkpoint read + durable restore progress + the "
                "Unit-5 registration); this service "
                f"({type(self._service).__name__}) does not — a capture-only "
                "seam cannot record a crash-resumable, registrable restore"
            )
        return self._service

    @staticmethod
    def _require_known_restore_state(
        record: CheckpointRecord, rows: Sequence[CheckpointMember]
    ) -> None:
        """Fail closed on durable state outside the closed vocabularies.

        An unknown status/outcome string would silently break crash-resume
        (which classifies terminality by identity), so it is refused loudly
        rather than guessed at.
        """
        if record.restore_status not in RESTORE_STATUSES:
            raise CoherenceError(
                f"checkpoint {record.checkpoint_id!r} carries unknown "
                f"restore_status {record.restore_status!r} (closed vocabulary: "
                f"{sorted(RESTORE_STATUSES)}) — refusing to drive legs from "
                "unclassifiable durable state"
            )
        for row in rows:
            if (
                row.restore_outcome is not None
                and row.restore_outcome not in RESTORE_MEMBER_OUTCOMES
            ):
                raise CoherenceError(
                    f"member {row.member_path!r} carries unknown restore_outcome "
                    f"{row.restore_outcome!r} — refusing to classify it as "
                    "terminal or pending (fail-closed)"
                )
        if record.restore_status in (
            RESTORE_STATUS_CONCLUDED,
            RESTORE_STATUS_REGISTERED,
        ) and any(row.restore_outcome is None for row in rows):
            raise CoherenceError(
                f"checkpoint {record.checkpoint_id!r} is "
                f"{record.restore_status!r} but holds outcome-less members — "
                "inconsistent durable state (both statuses require a terminal "
                "outcome for EVERY member: registration runs only after "
                "all-terminal, and a concluded restore records every outcome)"
            )

    def _require_drivable(self, pending: Sequence[CheckpointMember]) -> None:
        """Every leg this run will drive must be reachable BEFORE ``in_progress``.

        A missing binding or resolver is caller misconfiguration: raising here
        (ValueError) keeps an undrivable restore from minting progress state.
        Skip legs (declared forward-only / capture-refused tiers) need nothing.
        """
        declared = {m.member_path: m for m in self._members}
        problems: list[str] = []
        for row in pending:
            if not row.absent and row.restore_tier == RestoreTier.FORWARD_ONLY.value:
                continue  # forward_only_skipped needs no binding
            member = declared.get(row.member_path)
            if member is None or isinstance(member, _ForwardOnlyMember):
                problems.append(
                    f"{row.member_path}: no declared file/object member binding "
                    "(re-declare the member on this versioner before restore)"
                )
                continue
            if isinstance(member, _FileMember) and not row.absent:
                if self._file_resolver is None:
                    problems.append(
                        f"{row.member_path}: file restore needs a "
                        "FileContentResolver (pass file_resolver= at construction)"
                    )
                if not isinstance(member.source, FileRestoreTarget):
                    problems.append(
                        f"{row.member_path}: the member's source lacks the "
                        "write_cas_at restore leg (FileRestoreTarget surface)"
                    )
        if problems:
            raise ValueError(
                "restore pre-flight failed (nothing was started): "
                + "; ".join(problems)
            )

    # --- one member's leg (terminal outcome, always) --------------------------

    def _drive_member_absorbing(self, row: CheckpointMember) -> MemberRestoreOutcome:
        """Drive one member's leg, absorbing DETERMINISTIC environment failures.

        The termination-contract boundary for exceptions the legs did not
        anticipate. Three classes:

        - **Deterministic member/environment pathology** — the ``OSError``
          family (the member path replaced by a directory → ``IsADirectoryError``,
          permission denial → ``PermissionError``, …) and ``UnicodeDecodeError``
          (the text wire meeting bytes it cannot carry). Re-driving can NEVER
          succeed, so letting it raise would leave ``restore_status=in_progress``
          durably with no report and every resume re-raising forever — the
          termination contract would hold only for anticipated failures.
          Absorbed HERE into the absorbing ``target_lost``: the member's state
          is unreachable THROUGH ITS DECLARED SURFACE — the same "the restore
          target cannot be reached" meaning ``target_lost`` already carries for
          an expired pin or unretained version. ``conflict`` is deliberately
          NOT used: it is arbitration/divergence-shaped (a foreign writer
          losing a CAS, a detected live divergence) and an EISDIR/EACCES has
          neither an arbiter nor a diverging writer.
        - **Structural member refusal** — :class:`StructuralMemberRefused`, the
          source declaring THIS MEMBER undrivable through its own surface (the
          CLI bridge's containment refusal: a symlink component, a hardlinked
          co-owner, a non-regular leaf). Absorbed for exactly the reason above
          and into the same ``target_lost``: a refusal is by construction
          re-drive-proof, so raising it would wedge the whole checkpoint —
          including the members that restored fine — at ``in_progress`` forever
          with no report. Catching the ENGINE-owned base (never the interface
          subclass) keeps the seam one-directional.
        - **Everything else** — ``ConnectionError`` / ``TimeoutError`` (OSError
          subclasses, but TRANSIENT transport shapes: a redrive can succeed
          once the transport heals), registry raises, genuine bugs — PROPAGATES
          by design: the crash-resume path. ``restore_status`` stays
          ``in_progress`` durably and a later :meth:`restore` resumes
          idempotently.
        """
        try:
            return self._drive_member(row)
        except (ConnectionError, TimeoutError):
            # Transient transport shapes: crash-resume, never terminalized.
            raise
        except (OSError, UnicodeDecodeError, StructuralMemberRefused) as exc:
            lead = (
                "structural member refusal absorbed"
                if isinstance(exc, StructuralMemberRefused)
                else "deterministic member-environment failure absorbed"
            )
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_TARGET_LOST,
                attempts=0,
                detail=(
                    f"{lead} "
                    f"({type(exc).__name__}: {exc}) — the member is unreachable "
                    "through its declared surface and a re-drive cannot "
                    "succeed; the restore still concludes (termination "
                    "contract)"
                ),
                # This boundary catches a failure raised from ANYWHERE inside
                # the leg, including after the live member was truncated and
                # partially rewritten (the file target truncates before it
                # writes). It cannot tell that from a failure raised before the
                # leg read anything, so it must not report that no write was
                # attempted — the one answer that would read as clean over
                # bytes this run may have destroyed.
                observation=OBSERVATION_NOT_RECORDED,
            )

    def _drive_member(self, row: CheckpointMember) -> MemberRestoreOutcome:
        # ABSENT-fact FIRST: an absent-at-capture member restores to ABSENCE
        # via a delete leg. Its restore_tier is forward_only only because
        # there was no STATE to tier (derive_restore_tier's absent default) —
        # the tier speaks to state restorability, and absence needs no
        # history; checking the tier first would dead-code every delete leg.
        if row.absent:
            return self._drive_absent_member(row, self._require_binding(row.member_path))
        if row.restore_tier == RestoreTier.FORWARD_ONLY.value:
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED,
                attempts=0,
                detail=(
                    "forward-only member: enumerated and skipped — no captured "
                    "state to bring back (declared action surface, or a member "
                    "whose pointer could not be confirmed at capture)"
                ),
            )
        member = self._require_binding(row.member_path)
        if isinstance(member, _ObjectMember):
            return self._drive_object_leg(row, member)
        assert isinstance(member, _FileMember)  # pre-flight guaranteed
        return self._drive_file_leg(row, member)

    def _require_binding(self, member_path: str) -> "_FileMember | _ObjectMember":
        member = next(
            (m for m in self._members if m.member_path == member_path), None
        )
        if member is None or isinstance(member, _ForwardOnlyMember):
            # Pre-flight already refused this shape; kept as a hard invariant.
            raise CoherenceError(
                f"no declared binding for member {member_path!r} at leg time"
            )
        return member

    # --- delete legs (the ABSENT fact restored) -------------------------------

    def _drive_absent_member(
        self, row: CheckpointMember, member: "_FileMember | _ObjectMember"
    ) -> MemberRestoreOutcome:
        if isinstance(member, _ObjectMember):
            return self._drive_object_delete_leg(row, member)
        return self._drive_file_absent_leg(row, member)

    def _drive_object_delete_leg(
        self, row: CheckpointMember, member: _ObjectMember
    ) -> MemberRestoreOutcome:
        try:
            # Presence probe via the plain read (works on versioned AND
            # unversioned buckets — read_versioned would refuse the latter).
            member.binding.read(member.key)
        except KeyError:
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_CONVERGED,
                attempts=0,
                detail=(
                    "manifest records ABSENT and the object is already absent "
                    "live — no delete issued"
                ),
            )
        # Unconditional-latest delete: between the presence probe and this
        # request a foreign writer may land a version the delete then covers.
        # A DOCUMENTED residual (plan Unit 4): on a versioned bucket the
        # minted marker preserves that write as a noncurrent version
        # (recoverable); on an unversioned bucket the window is the member's
        # declared no-history reality. S3 offers no If-Match DELETE to close it.
        deletion = member.binding.delete(member.key)
        kind = (
            "delete marker minted (history survives)"
            if deletion.delete_marker
            else "permanent unversioned delete"
        )
        return MemberRestoreOutcome(
            member_path=row.member_path,
            outcome=RESTORE_OUTCOME_RESTORED,
            attempts=1,
            detail=(
                f"live object deleted to match the manifest's ABSENT fact — "
                f"{kind}; unconditional-latest (the pre-delete race window is "
                "a documented residual)"
            ),
            deleted_at_restore=float(self._clock()),
            # The state ALONE (R1's carve-out): the probe above established that
            # live state existed and this leg destroyed it, but it compared no
            # content and holds no versionId — the plain read is what works on
            # an unversioned bucket, and a second call to fetch a pointer is
            # forbidden (SPLIT-COMPARAND), so the honest answer is presence
            # without evidence. Never ``observed_differs``: that state asserts a
            # comparison with the capture that this leg never ran.
            observation=RestoreObservation(state=RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE),
        )

    def _drive_file_absent_leg(
        self, row: CheckpointMember, member: _FileMember
    ) -> MemberRestoreOutcome:
        try:
            member.source.read_with_version(row.member_path)
        except FileNotFoundError:
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_CONVERGED,
                attempts=0,
                detail=(
                    "manifest records ABSENT and the file is already absent "
                    "live — nothing to delete"
                ),
            )
        except StaleView:
            # A refusal is about bytes that exist on disk but cannot be paired
            # with a version, so the file IS present live: the divergence below.
            pass
        # v1 residual: the file seam offers no delete leg, so a live file the
        # manifest records ABSENT is a divergence this engine cannot converge
        # — absorbed as conflict (detection only; no-arbiter), never a raise
        # and never a silent skip.
        return MemberRestoreOutcome(
            member_path=row.member_path,
            outcome=RESTORE_OUTCOME_CONFLICT,
            attempts=0,
            detail=(
                "manifest records ABSENT but the file exists live; the v1 file "
                "leg has no delete surface (no-arbiter: adapter-local detection "
                "only) — divergence not converged"
            ),
        )

    # --- the S3 CAS leg (native-cas: the substrate arbitrates) ----------------

    def _drive_object_leg(
        self, row: CheckpointMember, member: _ObjectMember
    ) -> MemberRestoreOutcome:
        if row.native_token is None:
            # A present, restorable-tiered member always manifests a real
            # pointer (capture's Sentinel rule); its absence means the pinned
            # target is unreachable — absorbing, the run must still conclude.
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_TARGET_LOST,
                attempts=0,
                detail="no manifested restore pointer — the pinned target is unreachable",
            )
        budget = _LegBudget(MAX_RESTORE_LEG_REDRIVES)
        # Pinned bytes resolve LAZILY, after the converged check: a member
        # whose live state already matches the manifest concludes ``converged``
        # even when its pin has since expired (the token-identity rule — a
        # crash-resumed, already-landed leg must never report target_lost).
        pinned_bytes: bytes | None = None
        while True:
            if budget.try_consume() is None:
                return MemberRestoreOutcome(
                    member_path=row.member_path,
                    outcome=RESTORE_OUTCOME_CONFLICT,
                    attempts=budget.attempts,
                    detail=(
                        f"re-drive budget exhausted ({budget.attempts} attempts) "
                        "under sustained live-writer contention — no write "
                        "landed (native-CAS: every If-Match attempt lost its race)"
                    ),
                )
            view = self._object_live_view(row, member, budget)
            if isinstance(view, MemberRestoreOutcome):
                return view
            if view.fingerprint == row.fingerprint:
                return MemberRestoreOutcome(
                    member_path=row.member_path,
                    outcome=RESTORE_OUTCOME_CONVERGED,
                    attempts=budget.attempts,
                    detail=(
                        "live object already byte-identical to the manifest — "
                        "no write issued (authorship not claimed)"
                    ),
                )
            if pinned_bytes is None:
                resolved = self._resolve_pinned_object(row, member)
                if isinstance(resolved, MemberRestoreOutcome):
                    return resolved
                pinned_bytes = resolved
            # The WHOLE view is handed on, re-read every iteration and never
            # hoisted: a leg that re-drives under contention must report the
            # state the winning attempt overwrote, not the first read's (KTD7).
            outcome = self._object_cas_attempt(row, member, pinned_bytes, view, budget)
            if outcome is not None:
                return outcome

    def _object_live_view(
        self, row: CheckpointMember, member: _ObjectMember, budget: _LegBudget
    ) -> "MemberRestoreOutcome | _ObjectLiveView":
        """One live read → a :class:`_ObjectLiveView`, or a terminal outcome.

        The (bytes, ETag, versionId) triple comes from ONE response (the
        split-comparand rule); the versionId is only ever the POINTER, never the
        comparand (the F4 split). It is KEPT rather than discarded because it is
        the only handle that still resolves the state a restore is about to
        overwrite, and the write arms record it (R1/KTD6) — carrying it out of
        the read that already returned it is what keeps that report free of a
        second call. Live-absent yields the explicit :data:`CREATE_IF_ABSENT`
        comparand with no pointer and no digest — the create leg loses to any
        concurrent re-creation, never overwrites one.
        """
        try:
            live = member.binding.read_versioned(member.key)
        except KeyError:
            return _ObjectLiveView(comparand=CREATE_IF_ABSENT)
        except VersionPointerUnconfirmed:
            # The live pointer axis broke mid-restore (versioning suspended
            # since capture): what a write would mint can no longer be
            # confirmed — UNCONFIRMED → HOLD, never best-effort.
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_HELD_UNCONFIRMED,
                attempts=budget.attempts,
                detail=(
                    "live version pointer unconfirmed (bucket versioning "
                    "suspended since capture) — HELD, never best-effort"
                ),
            )
        return _ObjectLiveView(
            comparand=live.etag,
            pointer=live.version_id,
            fingerprint=_sha256_hex(live.data),
        )

    def _resolve_pinned_object(
        self, row: CheckpointMember, member: _ObjectMember
    ) -> "bytes | MemberRestoreOutcome":
        """The pinned target's bytes (an immutable S3 version), or target_lost."""
        assert row.native_token is not None  # guarded by the leg entry
        try:
            pinned = member.binding.read_pinned(member.key, version_id=row.native_token)
        except KeyError:
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_TARGET_LOST,
                attempts=0,
                detail=(
                    f"pinned version {row.native_token!r} no longer resolves "
                    "(expired or raced pin) — the restore target is gone"
                ),
            )
        return pinned.data

    @staticmethod
    def _observation_of_overwritten(
        view: _ObjectLiveView, *, write_confirmed: bool
    ) -> RestoreObservation:
        """What an object CAS arm DESTROYED, from the read that fed its comparand.

        ``write_confirmed`` says whether a landing was established, and it only
        ever matters on the live path: what a leg overwrote is settled by its
        READ, not by whose put won. A live read that found the member absent
        proves no writer — this run or a peer — could have destroyed content
        that was not there, so the create-on-absent path answers
        ``no_live_state`` either way and never fires a divergence gate.

        Branching on the CREATE_IF_ABSENT comparand rather than on a missing
        pointer is the fail-closed direction: were a live read ever to yield a
        real ETag without a pointer, this reports a divergence with the digest
        it has, instead of reporting that nothing was there to lose.
        """
        if view.comparand == CREATE_IF_ABSENT:
            return RestoreObservation(state=RESTORE_OBSERVATION_NO_LIVE_STATE)
        if not write_confirmed:
            return OBSERVATION_NOT_RECORDED
        return RestoreObservation(
            state=RESTORE_OBSERVATION_DIFFERS,
            # The versionId, never the ETag: the comparand arbitrates the write
            # and addresses nothing, while this pointer is what still resolves
            # the bytes the operator just lost (KTD5).
            pointer=view.pointer,
            fingerprint=view.fingerprint,
        )

    def _object_cas_attempt(
        self,
        row: CheckpointMember,
        member: _ObjectMember,
        pinned_bytes: bytes,
        view: _ObjectLiveView,
        budget: _LegBudget,
    ) -> MemberRestoreOutcome | None:
        """One conditional put under the freshly read comparand; ``None`` = re-drive.

        ``row.fingerprint`` doubles as the intended hash: a pinned S3 version
        is immutable, so its bytes always hash to the captured fingerprint.

        ``view`` is the caller's CURRENT read, passed per call rather than held
        by the loop, so a re-drive re-reads it and the landed arms report the
        iteration that WON (KTD7). Only its ``comparand`` reaches the substrate;
        its pointer and digest reach the report and nothing else.
        """
        binding, key = member.binding, member.key
        intended_hash = row.fingerprint or _sha256_hex(pinned_bytes)
        try:
            result = binding.cas_write_versioned(
                key, expected_token=view.comparand, new_bytes=pinned_bytes
            )
        except CasRetriesExhausted:
            # The binding's OWN 409-transient budget (this leg's in-binding
            # twin) exhausted: terminal, absorbing — no write landed.
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_CONFLICT,
                attempts=budget.attempts,
                detail=(
                    "the binding's conditional-put transient budget exhausted "
                    "— no write landed"
                ),
            )
        except VersionPointerUnconfirmed:
            # The put LANDED durably (its ETag was captured) but minted no
            # pointer — the bucket lost versioning mid-restore. The bytes are
            # back; the new state simply cannot be pinned or registered (the
            # Unit-5 seam receives no token for it).
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_RESTORED,
                attempts=budget.attempts,
                detail=(
                    "pinned bytes landed but the write minted no version "
                    "pointer (bucket unversioned mid-restore) — the restored "
                    "state cannot be re-pinned"
                ),
                # What is missing here is the pointer this write MINTED, not the
                # one it DESTROYED: the durable landing is confirmed, and the
                # state it replaced was read, compared and overwritten like any
                # other landed arm. Silence would leave the least recoverable
                # member the least described one.
                observation=self._observation_of_overwritten(view, write_confirmed=True),
            )
        if isinstance(result, VersionedCasWritten):
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_RESTORED,
                attempts=budget.attempts,
                detail=(
                    "pinned bytes landed via the native-CAS If-Match put "
                    f"(attempt {budget.attempts})"
                ),
                new_native_token=result.version_id,
                # Two versionIds sit three lines apart and mean opposite things:
                # ``result.version_id`` is the state the operator now HAS, the
                # observation names the state this put DESTROYED. A REPORT value
                # only — nothing may seed a later comparand from it.
                observation=self._observation_of_overwritten(view, write_confirmed=True),
            )
        if isinstance(result, CasConflict):
            # A foreign writer moved the comparand (412 / raced delete): the
            # substrate arbitrated and this attempt lost — re-drive from a
            # fresh live read, bounded by the leg budget.
            return None
        # CasUnknown: ONE reconciliation read decides; still-unknown → HOLD.
        decision = binding.reconcile_after_unknown(
            key, expected_token=view.comparand, intended_hash=intended_hash
        )
        if decision.verdict is ReconcileVerdict.CONVERGE:
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_CONVERGED,
                attempts=budget.attempts,
                detail=(
                    "unconfirmed write reconciled CONVERGE: the live object is "
                    "byte-identical to the manifest (authorship not claimed)"
                ),
                # ``write_confirmed=False``: a put was issued and its outcome
                # was lost, so on the live path this run cannot say whether IT
                # discarded the divergent state or a peer converged it first —
                # and R2's five states hold no "attempted, outcome unknowable".
                # ``not_recorded`` is the one that does not lie: its consumer
                # contract is "an observation the run never made, never read as
                # clean", which is exactly this. The default would under-claim
                # on the QUIETEST terminal in the vocabulary — the file leg's
                # unconfirmed arm can afford that only because it lands on
                # ``held_unconfirmed``, which is already loud.
                observation=self._observation_of_overwritten(view, write_confirmed=False),
            )
        if decision.verdict in (ReconcileVerdict.RE_DRIVE, ReconcileVerdict.CONFLICT):
            # Knowledge either way — an unmoved comparand (retry the intent)
            # or a real peer write (re-derive from fresh state): both re-drive
            # under the leg budget.
            return None
        # HOLD (and any future verdict): the outcome stayed unconfirmable
        # after the reconciliation read — HELD, never best-effort. (On the
        # create path a HOLD can also mean "still absent": re-driving blind
        # would risk a second landing of a two-world put, so HOLD wins.)
        return MemberRestoreOutcome(
            member_path=row.member_path,
            outcome=RESTORE_OUTCOME_HELD_UNCONFIRMED,
            attempts=budget.attempts,
            detail=(
                "write outcome still UNCONFIRMED after the reconciliation "
                "read — HELD, never best-effort"
            ),
            # Same ambiguity as the CONVERGE arm above and the same answer: a
            # put was issued and its outcome is unknowable, so this run cannot
            # say what it destroyed. Defaulting here would report a leg that
            # read divergent content and wrote exactly like one that never
            # reached a write decision.
            observation=self._observation_of_overwritten(view, write_confirmed=False),
        )

    # --- the file CAS leg (no-arbiter: detection-guarded ONLY) ----------------

    def _drive_file_leg(
        self, row: CheckpointMember, member: _FileMember
    ) -> MemberRestoreOutcome:
        budget = _LegBudget(MAX_RESTORE_LEG_REDRIVES)
        # Lazily resolved, after the converged check (token-identity first —
        # an already-matching live state concludes ``converged`` even when
        # retention has since expired; the resolver is consulted only when a
        # write is actually needed).
        pinned: bytes | None = None
        refusals = 0  # refused reads so far in this leg, which set the next wait
        while True:
            if budget.try_consume() is None:
                return MemberRestoreOutcome(
                    member_path=row.member_path,
                    outcome=RESTORE_OUTCOME_CONFLICT,
                    attempts=budget.attempts,
                    detail=(
                        f"re-drive budget exhausted ({budget.attempts} attempts) "
                        "under live-editor contention — no write landed "
                        "(no-arbiter: detection-guarded only, never substrate "
                        "arbitration)"
                    ),
                )
            try:
                live_bytes, live_version = member.source.read_with_version(row.member_path)
            except StaleView:
                # The source refused to pair the live bytes with a version (a
                # peer's commit still reaching disk, or an out-of-band edit), so
                # there is no comparand to CAS from. Re-drive: the transient
                # clears once the disk write lands, and a lasting refusal spends
                # the budget into conflict with no write, so the run concludes.
                # WAIT between refused reads, on the schedule write_cas uses for
                # the same transient: back-to-back reads spend the budget before
                # a peer's disk write can land, which bounds the leg by read
                # latency instead of by time.
                refusals += 1
                time.sleep(denied_read_backoff_sec(refusals))
                continue
            except FileNotFoundError:
                # v1 residual: recreation needs the coordinator artifact
                # identity, which lands with Unit-5 registration — absorbed,
                # never silent.
                return MemberRestoreOutcome(
                    member_path=row.member_path,
                    outcome=RESTORE_OUTCOME_CONFLICT,
                    attempts=budget.attempts,
                    detail=(
                        "member present in the manifest but absent live; the "
                        "v1 file leg cannot recreate it (artifact registration "
                        "lands in Unit 5) — divergence not converged (no-arbiter)"
                    ),
                )
            # The converged check's digest is the leg's own observation of the
            # live content, and it is computed from the SAME read that produced
            # the CAS comparand — keeping it is what lets the write arm report
            # what it overwrote without a second read (R1, KTD5). Recomputed
            # every iteration on purpose: the loop re-reads, so a hoisted
            # digest would name a state a later attempt never saw (KTD7).
            live_fingerprint = _sha256_hex(live_bytes)
            if live_fingerprint == row.fingerprint:
                return MemberRestoreOutcome(
                    member_path=row.member_path,
                    outcome=RESTORE_OUTCOME_CONVERGED,
                    attempts=budget.attempts,
                    detail=(
                        "live file already byte-identical to the manifest — "
                        "no write issued"
                    ),
                )
            if pinned is None:
                resolved = self._resolve_pinned_file(row)
                if isinstance(resolved, MemberRestoreOutcome):
                    return resolved
                pinned = resolved
            outcome = self._file_cas_attempt(
                row, member, pinned, live_version, live_fingerprint, budget
            )
            if outcome is not None:
                return outcome

    def _resolve_pinned_file(
        self, row: CheckpointMember
    ) -> "bytes | MemberRestoreOutcome":
        """The retained bytes for the manifested version, or target_lost.

        The fingerprint cross-check is load-bearing: unlike an immutable S3
        version, a resolver is a SEAM — bytes that do not hash to the captured
        fingerprint are not the captured state, and restoring them would be a
        silent wrong-content restore.
        """
        resolver = self._file_resolver
        assert resolver is not None  # pre-flight guaranteed
        try:
            pinned = resolver.content_at(row.member_path, int(row.native_token or 0))
        except KeyError:
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_TARGET_LOST,
                attempts=0,
                detail=(
                    f"retained version {row.native_token} no longer resolves "
                    "(retention expired) — the restore target is gone"
                ),
            )
        if _sha256_hex(pinned) != row.fingerprint:
            return MemberRestoreOutcome(
                member_path=row.member_path,
                outcome=RESTORE_OUTCOME_TARGET_LOST,
                attempts=0,
                detail=(
                    "retained bytes do not match the captured fingerprint — "
                    "the restore target is gone"
                ),
            )
        return bytes(pinned)

    def _file_cas_attempt(
        self,
        row: CheckpointMember,
        member: _FileMember,
        pinned: bytes,
        live_version: int,
        live_fingerprint: str,
        budget: _LegBudget,
    ) -> MemberRestoreOutcome | None:
        """One version-checked CAS write; ``None`` means re-drive.

        Detection-guarded ONLY: the CAS detects a foreign edit adapter-locally
        (typed :class:`~ccs.core.exceptions.CasVersionConflict`) — nothing here
        is, or may ever be labeled, substrate arbitration (no-arbiter).

        ``live_version``/``live_fingerprint`` are the caller's CURRENT read,
        passed per call rather than held by the loop, so a re-drive re-reads
        both and the landed arm reports the iteration that WON (KTD7). The
        version fills two roles at once here — the CAS comparand and the
        observation's pointer — while the fingerprint is the leg's independent
        evidence that the content differed (a source may rewrite bytes without
        advancing its version, which the CAS alone cannot see).
        """
        path = row.member_path
        source = member.source
        assert isinstance(source, FileRestoreTarget)  # pre-flight guaranteed
        try:
            source.write_cas_at(path, live_version, pinned)
        except CasVersionConflict:
            # DETECTION fired: a foreign edit moved the version between the
            # read and the CAS — re-drive from a fresh read (leg-budgeted).
            return None
        except ViewWedged:
            # This arm DID read a live comparand, but the observation states
            # what the leg OVERWROTE and a wedged view landed nothing — so
            # ``no_write_attempted`` (the default) is the truthful answer, and
            # ``observed_differs`` would report a loss that did not occur. The
            # member still concludes ``conflict``, which is the field an
            # operator acts on.
            return MemberRestoreOutcome(
                member_path=path,
                outcome=RESTORE_OUTCOME_CONFLICT,
                attempts=budget.attempts,
                detail=(
                    "the comparand view stayed strict-denied (wedged) — no "
                    "write landed (no-arbiter: detection-guarded only)"
                ),
            )
        except CommitUnconfirmed:
            # NOT the default, and the wedged arm above is why: that one never
            # issued a write, this one did and cannot learn the outcome. The
            # split matters because the target decides the order. A CAS-first
            # binding may never have touched the bytes; the file target this
            # CLI ships writes them to disk FIRST and raises here only when the
            # ledger commit is refused, so on that path the live content is
            # already gone. The engine cannot tell the two apart through the
            # seam, so it reports what it knows: no observation. Claiming a
            # discarded version would over-claim on the first shape; claiming
            # no write was attempted would read as clean over the second.
            return MemberRestoreOutcome(
                member_path=path,
                outcome=RESTORE_OUTCOME_HELD_UNCONFIRMED,
                attempts=budget.attempts,
                detail=(
                    "the version-CAS commit could not be confirmed (transport "
                    "failed mid-commit) — HELD, never best-effort"
                ),
                observation=OBSERVATION_NOT_RECORDED,
            )
        return MemberRestoreOutcome(
            member_path=path,
            outcome=RESTORE_OUTCOME_RESTORED,
            attempts=budget.attempts,
            # write_cas_at advances deterministically to expected+1 on a
            # confirmed win — the pointer the Unit-5 registration consumes.
            new_native_token=str(live_version + 1),
            detail=(
                "pinned bytes landed via the detection-guarded version-CAS "
                f"(attempt {budget.attempts}; no-arbiter: adapter-local "
                "detection, never substrate arbitration)"
            ),
            # The ONE arm with a confirmed write behind it, and the leg reached
            # it only past the converged short-circuit — so the live content
            # provably differed from the capture. Both halves name the state
            # just OVERWRITTEN (``live_version``), never the one just minted
            # (``live_version + 1``, carried separately above): conflating them
            # would hand an operator the pointer of the state they still have.
            # A REPORT value only — nothing may seed a later comparand from it.
            observation=RestoreObservation(
                state=RESTORE_OBSERVATION_DIFFERS,
                pointer=str(live_version),
                fingerprint=live_fingerprint,
            ),
        )

    # --- restore report reconstruction (durable rows → report) ----------------

    @staticmethod
    def _outcome_from_durable_row(row: CheckpointMember) -> MemberRestoreOutcome:
        assert row.restore_outcome is not None  # callers filtered / validated
        return MemberRestoreOutcome(
            member_path=row.member_path,
            outcome=row.restore_outcome,
            attempts=0,
            detail=(
                "terminal outcome recorded by a prior run (resumed: reported, "
                "not re-driven)"
            ),
            deleted_at_restore=row.deleted_at_restore,
            resumed_from_prior_run=True,
            # No leg ran in THIS run and the observation is run-local, so what
            # a prior run overwrote is unrecoverable — but the durable outcome
            # is not silent. A row that concluded in
            # RESTORE_OUTCOMES_PROVING_NO_WRITE proves no write landed, by any
            # run, so no-write-attempted is established rather than assumed and
            # the honest answer is the quiet one. Reporting not-recorded for
            # those would make a re-restore of a workspace nothing ever touched
            # fail under the operator's gate, on its second identical run.
            observation=(
                OBSERVATION_NO_WRITE_ATTEMPTED
                if row.restore_outcome in RESTORE_OUTCOMES_PROVING_NO_WRITE
                else OBSERVATION_NOT_RECORDED
            ),
        )

    def _report_from_durable_rows(
        self,
        store: CheckpointRestoreStore,
        checkpoint_id: str,
        rows: Sequence[CheckpointMember],
    ) -> WorkspaceRestoreReport:
        return WorkspaceRestoreReport(
            checkpoint_id=checkpoint_id,
            status=RESTORE_STATUS_CONCLUDED,
            members=tuple(self._outcome_from_durable_row(row) for row in rows),
            registration=self._registration_from_durable_state(store, rows),
        )

    def _registration_from_durable_state(
        self, store: CheckpointRestoreStore, rows: Sequence[CheckpointMember]
    ) -> RestoreRegistration:
        """Rebuild the registration ANSWER for an already-concluded checkpoint.

        The run-local :class:`RestoreRegistration` detail (per-member refusal
        reasons, invalidation counts, attempts) is not durably retained — but
        the registration's EFFECT is: a landed registration leaves every
        written file member's coordinator artifact at the manifest fingerprint
        (the exact predicate the service-side idempotency filter matches).
        This rebuild re-classifies the durable member rows the way
        :meth:`_registration_seam` classifies run-local outcomes, then
        re-READS that effect through the store's
        ``workspace_member_registered`` seam — pure reads; nothing resolves,
        mints, commits, or re-attempts (REFUSED is terminal; ``concluded`` is
        absorbing):

        - no written file members → ``empty_write_set`` (nothing ever needed
          a commit — the delete-only / all-converged / S3-only shapes);
        - every written file member registered at its manifest fingerprint →
          ``registered_by_prior_run`` (the registration landed before this
          checkpoint concluded; nothing re-registered);
        - anything else → ``refused``: the registration concluded REFUSED
          (fence or exhaustion — terminal either way). The un-registered
          members are named in the detail; durable rows retain no per-member
          conflict reason, so the ``refused`` map stays empty and the
          identity-matched ``status`` is the signal downstream surfaces exit
          non-zero on.

        Disclosed residual: a foreign commit landing AFTER a committed
        registration moves the artifact off the manifest fingerprint, so a
        much-later rebuild can read a landed registration as ``refused`` —
        the fail-closed direction (a loud false alarm an operator can
        inspect), never the silent ``None``-means-fine this rebuild replaces.
        """
        substrate_registered: list[str] = []
        deleted_recorded: list[str] = []
        writes: list[WorkspaceRestoreWrite] = []
        for row in rows:
            if row.deleted_at_restore is not None:
                deleted_recorded.append(row.member_path)
                continue
            if row.restore_outcome != RESTORE_OUTCOME_RESTORED:
                continue
            if row.arbitration_tier == ArbitrationTier.NATIVE_CAS.value:
                substrate_registered.append(row.member_path)
                continue
            if row.fingerprint is None:  # defensive: a written member has one
                continue
            writes.append(
                WorkspaceRestoreWrite(
                    member_path=row.member_path, fingerprint=row.fingerprint
                )
            )
        if not writes:
            return RestoreRegistration(
                status=WORKSPACE_REGISTRATION_EMPTY,
                detail=(
                    "rebuilt from durable rows: no written file members — the "
                    "commit write-set was empty and commit_all was never called"
                ),
                substrate_registered=tuple(substrate_registered),
                deleted_recorded=tuple(deleted_recorded),
            )
        unregistered = [
            write.member_path
            for write in writes
            if not store.workspace_member_registered(
                write.member_path, write.fingerprint
            )
        ]
        if not unregistered:
            return RestoreRegistration(
                status=WORKSPACE_REGISTRATION_PRIOR_RUN,
                detail=(
                    "rebuilt from durable state: every written file member is "
                    "registered at its manifest fingerprint — the registration "
                    "landed before this checkpoint concluded (nothing "
                    "re-registered)"
                ),
                substrate_registered=tuple(substrate_registered),
                deleted_recorded=tuple(deleted_recorded),
            )
        return RestoreRegistration(
            status=WORKSPACE_REGISTRATION_REFUSED,
            detail=(
                "rebuilt from durable state: written file member(s) "
                f"{sorted(unregistered)} are NOT registered at their manifest "
                "fingerprints — the registration concluded REFUSED (fence or "
                "exhaustion; TERMINAL — concluded is absorbing, nothing is "
                "re-attempted). Per-member conflict reasons are not durably "
                "retained; the status is the signal."
            ),
            substrate_registered=tuple(substrate_registered),
            deleted_recorded=tuple(deleted_recorded),
        )

    # --- internals ------------------------------------------------------------

    def _register(self, member: _FileMember | _ObjectMember | _ForwardOnlyMember) -> None:
        self._members.append(member)

    def _require_new_path(self, member_path: str) -> str:
        if not isinstance(member_path, str) or not member_path.strip():
            raise ValueError("member path must be a non-empty string")
        if any(member_path == m.member_path for m in self._members):
            raise ValueError(
                f"duplicate member path {member_path!r}: member paths key the "
                "manifest and must be unique within a workspace"
            )
        return member_path

    @staticmethod
    def _arbitration_tier(member: _FileMember | _ObjectMember) -> str:
        """Who arbitrates a foreign writer racing this member's restore leg.

        S3 members: the substrate itself (an If-Match put — native CAS). File
        members: NOBODY — the volume can only DETECT a foreign edit
        adapter-locally, and the manifest must never present detection as
        substrate arbitration (the cross-host carve-out).
        """
        if isinstance(member, _ObjectMember):
            return ArbitrationTier.NATIVE_CAS.value
        return ArbitrationTier.NO_ARBITER.value

    @staticmethod
    def _require_utf8_text(member_path: str, data: bytes) -> None:
        try:
            data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise BinaryFileMemberRefused(
                f"file member {member_path!r} holds non-UTF-8 bytes and cannot "
                "be checkpointed (v1 limitation: the file restore leg rides the "
                f"UTF-8 snapshot-session wire; {exc}). Nothing was persisted.",
                member_path=member_path,
            ) from None
