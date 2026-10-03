# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""In-memory artifact registry for coherence coordination."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)
from uuid import UUID, uuid4

from ccs.core.exceptions import (
    STALE_READ_GENERATION_REASON,
    UNKNOWN_ARTIFACT_REASON,
    StaleReadGeneration,
    WatchdogAbandoned,
)
from ccs.core.states import MESIState, TransientState
from ccs.core.types import (
    Artifact,
    CasCorruption,
    CommitAllEntry,
    ConflictDetail,
    MultiCommitConflict,
    MultiCommitResult,
    VersionedReadRejection,
)

# The registry contract return types (ReclamationSlot / CasResult / CaptureResult)
# live in the Protocol module (deduplicated there); re-exported here to keep
# `ccs.coordinator.registry.CasResult` a stable public import path.
from .registry_protocol import (
    CLAIM_CAPTURE_TRIGGERS,
    EPOCH_BUMP_TRIGGERS,
    FOREIGN_WRITE_OUTCOMES,
    RECLAIM_TRIGGERS,  # noqa: F401 — re-exported; see the parity test
    CaptureResult,
    CasResult,
    CheckpointMember,
    CheckpointRecord,
    DetectionRun,
    ReclamationSlot,
    UncoverableRun,
)
from .retention import RetentionPolicy, collectible_versions

CCS_STATE_LOG_SCHEMA_VERSION = "ccs.state_log.v2"

logger = logging.getLogger(__name__)

_M_OR_E_STATES: frozenset[MESIState] = frozenset({MESIState.MODIFIED, MESIState.EXCLUSIVE})


@dataclass
class ArtifactRecord:
    """Internal registry record for one artifact."""

    artifact: Artifact
    content: str
    state_by_agent: dict[UUID, MESIState] = field(default_factory=dict)
    transient_by_agent: dict[UUID, TransientState] = field(default_factory=dict)
    transient_tick_by_agent: dict[UUID, int] = field(default_factory=dict)
    last_writer: Optional[UUID] = None
    # Retained version snapshots, keyed by artifact version. The value is the
    # captured BODY. The annotation admits ``bytes`` because the ``commit_cas``
    # WIN path stores ``record.content``, which is ``bytes`` on the in-process
    # library path (``AgentRuntime.write_cas`` threads bytes through). The old
    # ``dict[int, str]`` annotation was wrong (a ``type: ignore`` papered over
    # it at the WIN site); corrected here as part of the value-shape change for
    # bounded retention. ``version_captured_at`` is a PARALLEL dict of capture
    # wall-clock timestamps (``time.time()``), one per retained version — kept
    # separate (rather than tupling the value) so the v0.5 pinned suites that
    # read ``version_history[v]`` as the body stay valid byte-for-byte. The two
    # dicts are always mutated together (capture and GC) so their key sets match.
    version_history: dict[int, bytes | str] = field(default_factory=dict)
    version_captured_at: dict[int, float] = field(default_factory=dict)
    granted_at_tick_by_agent: dict[UUID, int] = field(default_factory=dict)
    last_reclamation_by_agent: dict[UUID, ReclamationSlot] = field(default_factory=dict)
    # Read-generation fence (single-host Piece #2). owner_generation is the
    # per-artifact ownership epoch, bumped on every sweep reclamation;
    # read_generation_by_agent[ag] is the owner_generation an agent captured
    # when it last established its write-claim (a genuine read OR an E/M
    # acquire). An ABSENT key (== None) means the agent never established a
    # fence claim -- a plain OCC writer whose lost-update protection is
    # version-CAS, not the fence -- so the commit guard ADMITS it (version-CAS,
    # checked first, arbitrates); only a PRESENT-and-superseded read_generation
    # (captured < current owner_generation) is rejected. The counter only grows
    # (resets to 0 per construction in-memory).
    owner_generation: int = 0
    read_generation_by_agent: dict[UUID, int] = field(default_factory=dict)
    # SB-10 (compaction re-emission, R6/R7): the artifact version whose BYTES
    # an agent last observed, recorded atomically with every non-INVALID grant
    # transition (set_agent_state) and advanced by the commit WIN paths
    # (commit_cas / commit_all / the pessimistic service.commit's MODIFIED
    # upsert). An ABSENT key means never-observed (surfaced as None, never a
    # 0-sentinel); a transition to INVALID deliberately preserves the prior
    # value -- it is the durable comparand the post-compaction stale flag is
    # computed from. Mirrors the sqlite agent_states.last_observed_version
    # column (nullable INTEGER, schema v6).
    last_observed_version_by_agent: dict[UUID, int] = field(default_factory=dict)


class ArtifactRegistry:
    """Canonical in-memory artifact directory and payload store.

    Concurrency contract: thread-safe, same as :class:`SqliteArtifactRegistry`.
    Every public method that touches mutable registry state serializes on one
    reentrant ``self._lock`` (the widened successor of the old capture-only
    lock). The only exemptions are accessors of fields immutable since
    construction, each named at its declaration site. Callers composing
    MULTI-call sequences (check-then-act across method boundaries) still need
    :meth:`abort_guard` to hold the lock across the whole sequence -- per-call
    serialization alone cannot make a decision and its write atomic.
    """

    @contextmanager
    def abort_guard(self, abort: "threading.Event | None" = None) -> Iterator[None]:
        """Hold ``self._lock`` across the caller's whole mutation, failing
        closed if the handler watchdog already timed out (finding A6). The
        same guarantee, word for word, as
        :meth:`SqliteArtifactRegistry.abort_guard`, so
        :class:`CoordinatorService` composes check-then-act sequences
        identically over either registry.

        The lock is what makes the guard mean something across MULTIPLE
        registry calls: each public method serializes itself, but a service
        sequence that reads, decides, then writes needs the decision and the
        write inside one hold. The RLock is reentrant, so the nested registry
        calls under the guard acquire freely.

        The abort check runs the instant the lock is won: by then the watchdog
        may have fired, returned ``degraded: true``, and SET ``abort`` -- the
        late "phantom grant" aborts before it lands. ``abort=None`` (every
        non-watchdog caller) is a plain lock acquire with no behavioural
        change.
        """
        with self._lock:
            if abort is not None and abort.is_set():
                raise WatchdogAbandoned(
                    "handler watchdog timed out before this mutation ran; "
                    "aborting before it lands (A6)."
                )
            yield

    def __init__(
        self,
        *,
        state_log: Callable[[dict[str, Any]], None] | None = None,
        agent_names: dict[UUID, str] | None = None,
        instance_id: str | None = None,
        retain_versions: bool = False,
        retention_policy: RetentionPolicy | None = None,
    ) -> None:
        # KTD7 -- ``state_log`` (and every callback this registry accepts) now
        # runs while ``self._lock`` is HELD: it must never block on other
        # threads and may call back only into this registry's own (reentrant)
        # methods. The sqlite registry has always fired it inside its lock;
        # this makes the obligation symmetric.
        if state_log is not None and instance_id is None:
            raise ValueError(
                "instance_id must be provided when state_log is set; "
                "pass instance_id=str(uuid4()) or route through CCSStore which manages it automatically"
            )
        self._records: dict[UUID, ArtifactRecord] = {}
        self._heartbeat_by_agent: dict[UUID, int] = {}
        self._state_log = state_log
        self._agent_names = agent_names
        self._instance_id: str = instance_id if instance_id is not None else str(uuid4())
        # coordinator_epoch: seeded for the CROSS-HOST fence follow-on
        # (demand-gated; a client-carried token would compare it). NOT used in
        # any guard yet -- the single-host fence keys only on owner_generation,
        # and a wiped/recreated store also wipes the read_generation rows, so
        # there is no pre-wipe claim for an epoch to fail. Kept so the durable
        # store identity exists from day one.
        self._coordinator_epoch: str = uuid4().hex
        self._seq: int = 0
        # Retention is active iff ``retain_versions`` is True. The attribute is
        # kept TRUTHY and named ``_retain_versions`` because the recorder test
        # (tests/test_replay_recorder.py) asserts on this private name.
        self._retain_versions = retain_versions
        # ``retention_policy=None`` with ``retain_versions=True`` == today's
        # UNBOUNDED semantics (no GC) — this is the back-compat contract that
        # keeps the four pinned v0.5/recorder suites green. A policy is an
        # explicit opt-in to BOUNDED retention: GC runs only when this is set.
        self._retention_policy = retention_policy
        # Snapshot session pin store (SB-17 / TX-1, Unit 2 / R1, R4):
        # ``{session_token: {artifact_id: pinned_version}}``. The version a live
        # session pins is exempt from the inline retention GC (the exemptions
        # seam) until ``release_session`` drops it. Process-scoped only —
        # in-memory pins do NOT survive a restart (R6; restart-survival is
        # sqlite-only), which the parity harness asserts rather than masks.
        self._session_pins: dict[str, dict[UUID, int]] = {}
        # Durable owner-binding MIRROR (SB-17 / TX-1, R13/R6/R14):
        # ``{session_token: (owner, created_at_tick)}``. Symmetric with
        # :attr:`SqliteArtifactRegistry`'s ``session_meta`` table for API parity,
        # but PROCESS-SCOPED like the pins — a fresh in-memory instance (the
        # "restart") has none, so an in-memory session never survives a restart
        # (the asserted divergence). Lets the sweep enumerate sessions uniformly
        # across both registries via :meth:`all_session_meta`.
        self._session_meta: dict[str, tuple[UUID, int]] = {}
        # Workspace-checkpoint manifest store (WV plan Unit 2 / R1, R9):
        # ``{checkpoint_id: header}`` + ``{checkpoint_id: {member_path: member}}``.
        # The in-memory mirror of the sqlite ``workspace_checkpoints`` /
        # ``workspace_checkpoint_members`` tables for API parity. PROCESS-SCOPED
        # like the session pins — an in-memory manifest does NOT survive a
        # restart (restart-durability is sqlite-only; the parity harness asserts
        # the divergence rather than masking it). Guarded by ``_lock``
        # (the manifest create is a multi-row atomic insert, same class of
        # critical section as the session-pin capture).
        self._checkpoints: dict[str, CheckpointRecord] = {}
        self._checkpoint_members: dict[str, dict[str, CheckpointMember]] = {}
        # Caller-principal bindings (caller-principal plan, U4): ``{identity:
        # (principal, mint_nonce)}``, first claim wins, never rebound. The
        # in-memory mirror of the sqlite ``caller_principals`` table — its OWN
        # store, never ``_session_meta``, so the session sweep, cap and release
        # cannot see it. PROCESS-SCOPED like everything here (a fresh instance
        # has no bindings — the declared restart loss). Guarded by ``_lock``.
        self._caller_principals: dict[UUID, tuple[str, str]] = {}
        # Conflict-outcome instrumentation (guarantee-ladder U5 / R-4): counts
        # keyed by (artifact_id, agent_id, reason) for the three typed deny
        # reasons, incremented at the SAME branch that constructs the returned
        # ConflictDetail — never at the service layer, so wire and library
        # callers are counted identically. Observability only: NOT part of the
        # RegistryBase coordination contract (the protocol-parity guard pins
        # the Protocol surface, and this deliberately stays off it). Process-
        # scoped here; durable in the sqlite registry (KTD-9). Callbacks fire
        # after the deny is decided, each guarded — an observer raise must
        # never turn a typed deny into an exception. (Merge note: authored
        # against the old lock-free contract with a dedicated counter lock;
        # under the registry-wide lock below the deny branches already hold
        # self._lock, so the counter serializes there like all other state.)
        self._conflict_counts: dict[tuple[UUID, UUID, str], int] = {}
        # Foreign-write detection state (plan U2). ``_detection_last_hash`` is
        # KTD14's edge gate — the disk content last counted per artifact, held
        # here and never on the canonical hash a safety check reads.
        self._detection_counts: dict[UUID, dict[str, int]] = {}
        self._detection_last: dict[UUID, tuple[str, str]] = {}
        self._detection_run_id: str = uuid4().hex
        self._detection_run: DetectionRun | None = None
        self._detection_runs_closed: list[DetectionRun] = []
        self._detection_uncoverable: dict[str, UncoverableRun] = {}
        self.conflict_callbacks: list[Callable[[UUID, UUID, str], None]] = []
        # THE registry lock. Started life as a capture-only lock (the
        # multi-artifact session-pin critical sections); widened to serialize
        # EVERY public method that touches mutable registry state, because the
        # old "lock-free by contract, GIL per-access atomicity" policy only
        # ever covered a single dict access -- and the service layer composes
        # multi-access sequences (the fence check-then-persist, the sweep's
        # read-then-reclaim) that the GIL does not hold together. Reentrant so
        # a caller holding it via abort_guard can compose registry calls into
        # one atomic sequence, and so the internal cross-method calls
        # (commit paths -> set_agent_state) re-enter freely. Uncontended
        # RLock acquire measured ~168 ns; the widest method makes ~16 internal
        # acquires ~= 2.7 us per commit cycle -- noise against the sqlite arm.
        self._lock = threading.RLock()

    def _capture_version(
        self, record: ArtifactRecord, version: int, content: bytes | str
    ) -> None:
        """Snapshot ``content`` under ``version`` and run inline GC (R1, R3, R4).

        The single retention apply path shared by all three capture points
        (``register_artifact``, ``set_artifact_and_content``, the ``commit_cas``
        WIN). Stores the body and its capture timestamp, then — ONLY when a
        bounded policy is set — drops the versions :func:`collectible_versions`
        marks (the current version always survives; unbounded mode skips GC
        entirely, preserving today's semantics).

        Lock-free posture (registry contract): the version-history dict ops are
        plain GIL-atomic; a concurrent reader of ``get_content_at_version`` sees
        a consistent value at any single dict access, no ``BEGIN IMMEDIATE``.
        Caveat (Unit 2): the exemptions read below re-enters ``_lock``
        via ``_live_pins_for_artifact``. This method is ALWAYS invoked from
        within an already-held ``_lock`` (the version-move apply in
        ``commit_cas`` / ``set_artifact_and_content`` / ``register_artifact``),
        so the RLock re-entry is harmless and the dict ops stay GIL-atomic.
        """
        captured_at = time.time()  # one wall-clock read: stamp == GC reference.
        record.version_history[version] = content
        record.version_captured_at[version] = captured_at
        if self._retention_policy is None:
            return  # unbounded mode (retain_versions=True, no policy): no GC.
        # Exemptions seam (Unit 2 — the first GC producer to populate it): a
        # version pinned by ANY live snapshot session for this artifact is held
        # back from collection until its session is released (R4). Without this,
        # a bounded K/T policy could collect a pinned version out from under a
        # live cut, breaking the consistent-read guarantee.
        for dropped in collectible_versions(
            record.version_captured_at,
            current_version=version,
            policy=self._retention_policy,
            now=captured_at,
            exemptions=self._live_pins_for_artifact(record.artifact.id),
        ):
            record.version_history.pop(dropped, None)
            record.version_captured_at.pop(dropped, None)

    def _live_pins_for_artifact(self, artifact_id: UUID) -> set[int]:
        """Return the versions pinned by LIVE snapshot sessions for ``artifact_id``
        — the GC exemption set the inline retention GC honors (Unit 2 / R4).

        Mirrors ``Snapshot.tla``'s ``PinnedVersions(art)``. Taken under the
        capture lock so a concurrent ``capture_version_vector`` / ``release_session``
        sees a consistent pin store; the returned set is a plain snapshot the GC
        loop consumes after the lock is released. A session pins at most one
        version per artifact (the cut is a map), so the union is naturally
        deduplicated by the set."""
        with self._lock:
            return {
                pins[artifact_id]
                for pins in self._session_pins.values()
                if artifact_id in pins
            }

    def register_artifact(self, artifact: Artifact, content: str) -> None:
        """Insert artifact record into registry.

        The version-establishing mutation is performed under ``_lock``
        (Unit 2): a concurrent ``capture_version_vector`` reads N artifact
        versions + inserts the pin set under that same lock, so registering /
        moving a version cannot interleave a multi-artifact capture and tear its
        cut (read skew)."""
        with self._lock:
            record = ArtifactRecord(artifact=artifact, content=content)
            if self._retain_versions:
                self._capture_version(record, artifact.version, content)
            self._records[artifact.id] = record

    def has_artifact(self, artifact_id: UUID) -> bool:
        """Return whether an artifact exists in registry."""
        with self._lock:
            return artifact_id in self._records

    def artifact_ids(self) -> list[UUID]:
        """Return all known artifact ids."""
        with self._lock:
            return list(self._records.keys())

    def get_artifact(self, artifact_id: UUID) -> Optional[Artifact]:
        """Return artifact metadata if present."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.artifact if record else None

    def get_content(self, artifact_id: UUID) -> Optional[str]:
        """Return artifact content if present."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.content if record else None

    @property
    def coordinator_epoch(self) -> str:
        """The registry's coordinator epoch (read-generation fence follow-on).

        Public accessor mirroring :attr:`SqliteArtifactRegistry.coordinator_epoch`
        so the two registries satisfy ONE duck-type (Unit 4 / R5): every
        ``read_at_version`` answer and rejection stamps this, and an optional
        ``expected_epoch`` is compared against it. The in-memory epoch is
        **per-construction** (minted fresh in ``__init__``, NOT persisted) — a
        new ``ArtifactRegistry`` gets a new epoch, so it cannot mismatch across a
        restart the way the durable sqlite epoch can; it exists for surface
        parity and for the same-process ``expected_epoch`` guard.

        Lock exemption: reads ``_coordinator_epoch``, assigned once in
        ``__init__`` and never mutated."""
        return self._coordinator_epoch

    @property
    def instance_id(self) -> str:
        """The registry's instance identity (minted per construction unless the
        caller supplied one).

        Public read-only accessor mirroring
        :attr:`SqliteArtifactRegistry.instance_id` so the two registries share
        one identity duck-type and consumers never reach into the private
        field.

        Lock exemption: reads ``_instance_id``, assigned once in ``__init__``
        and never mutated."""
        return self._instance_id

    def retention_meta(self) -> tuple[bool, RetentionPolicy | None]:
        """Return ``(retention_enabled, policy_or_None)`` (Unit 4 / R5 duck-type).

        Mirrors :meth:`SqliteArtifactRegistry.retention_meta` so ``read_at_version``
        derives ``retention_off`` and the T-expiry axis identically on both
        registries. In-memory there is no persisted meta — the answer is the
        live constructor state: ``_retain_versions`` is the enabled marker and
        ``_retention_policy`` the bound (``None`` ⇒ unbounded when enabled).
        ``retain_versions=False`` ⇒ ``(False, None)`` (retention never on).

        Lock exemption: reads ``_retain_versions`` and ``_retention_policy``,
        both assigned once in ``__init__`` and never mutated."""
        if not self._retain_versions:
            return False, None
        return True, self._retention_policy

    def get_owner_generation(self, artifact_id: UUID) -> int:
        """Return the artifact's ownership epoch (read-generation fence)."""
        with self._lock:
            return self._records[artifact_id].owner_generation

    def get_read_generation(self, artifact_id: UUID, agent_id: UUID) -> int | None:
        """Return the generation an agent captured at its last claim, or None if
        it never established a fence claim (a plain OCC writer that version-CAS,
        not the fence, arbitrates)."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.read_generation_by_agent.get(agent_id) if record else None

    def last_observed_version_for(self, artifact_id: UUID, agent_id: UUID) -> int | None:
        """Return the artifact version whose bytes this agent last observed
        (SB-10 R6/R7: recorded atomically with every non-INVALID grant/commit
        transition), or None when the pair was never observed — absence is an
        absent key, never a 0-sentinel, and a transition to INVALID preserved
        the prior recorded value. The post-compaction staleness comparand."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.last_observed_version_by_agent.get(agent_id) if record else None

    def get_artifact_and_generation(
        self, artifact_id: UUID
    ) -> tuple[Artifact, int] | None:
        """Return ``(artifact, owner_generation)`` as one pair-consistent
        snapshot, or None when the artifact is absent.

        Under the widened registry lock this method is serialized like every
        other, so the seqlock below is redundant -- kept deliberately, because
        its no-ABA argument is the only written record of WHY the pair read is
        safe, and it costs one extra field read. The pair is stitched with a
        seqlock on ``owner_generation``: within the
        life of one record the generation only ever increments (reclaims bump
        it; nothing decrements or resets it), so if it reads equal on both sides
        of the artifact read, the returned pair coexisted at the instant the
        artifact was read — a concurrent sweep bump retries rather than tearing
        the pair. ``record.artifact`` is a frozen dataclass swapped wholesale on
        a version move, so reading it is itself one atomic access.

        Scope of that no-ABA property: it holds for a LIVE record, which is what
        the seqlock needs. It is NOT an identity guarantee across the record's
        lifetime — deleting an artifact and re-registering the same name mints a
        fresh record back at ``(version=1, owner_generation=0)``, so a comparand
        pair captured before such a cycle can numerically match one read after
        it. Callers comparing pairs across a window (see
        ``adapters.effect_gate.gate``) inherit that boundary; it is unchanged
        from the version-only comparand that preceded the generation leg. Note
        the cycle needs an actual delete: no HTTP route, MCP tool, or CLI verb
        wires ``CoordinatorService.delete``, so none of the surfaces the gate
        runs over can drive it — but the in-process ``CCSStore.delete()`` can,
        so it is a reachable boundary, not an impossible one."""
        with self._lock:
            record = self._records.get(artifact_id)
            if record is None:
                return None
            while True:
                generation = record.owner_generation
                artifact = record.artifact
                if record.owner_generation == generation:
                    return artifact, generation

    def set_artifact_and_content(
        self,
        artifact_id: UUID,
        artifact: Artifact,
        content: str,
        *,
        last_writer: Optional[UUID] = None,
        fence_agent_id: Optional[UUID] = None,
    ) -> None:
        """Replace artifact metadata/content for an existing record.

        Read-generation fence (pessimistic ``commit()`` path): when
        ``fence_agent_id`` is given, reject -- atomically (GIL) with the persist
        -- if that committer's captured read_generation was superseded by a
        sweep reclamation (the race the service's earlier get_agent_state check
        misses). A ``None`` fence_agent_id (source-churn) is unguarded.
        """
        # One hold covers the fence check AND the persist: deciding on
        # read_generation outside the lock and writing inside it would be the
        # same check-then-act gap this widening exists to close. Also
        # serializes with a concurrent multi-artifact capture so the cut
        # cannot observe this artifact moved while a peer in the same
        # read-set is not (read skew).
        with self._lock:
            record = self._records[artifact_id]
            if fence_agent_id is not None:
                read_gen = record.read_generation_by_agent.get(fence_agent_id)
                if read_gen is not None and read_gen < record.owner_generation:
                    raise StaleReadGeneration(
                        f"{STALE_READ_GENERATION_REASON} agent={fence_agent_id} "
                        f"artifact={artifact_id} read_gen={read_gen} "
                        f"owner_gen={record.owner_generation}"
                    )
            if self._retain_versions:
                self._capture_version(record, artifact.version, content)
            record.artifact = artifact
            record.content = content
            record.last_writer = last_writer

    def get_content_at_version(self, artifact_id: UUID, version: int) -> str | bytes | None:
        """Return content for a specific version, if retained.

        Returns the body with its original Python type (``bytes`` when the
        ``commit_cas`` WIN threaded bytes — matching the corrected
        ``version_history`` annotation); ``None`` when not retained. Mirrors
        :meth:`SqliteArtifactRegistry.get_content_at_version`'s widened type."""
        with self._lock:
            record = self._records.get(artifact_id)
            if record is None:
                return None
            return record.version_history.get(version)

    def get_version_record(
        self, artifact_id: UUID, version: int
    ) -> tuple[str | bytes, float] | None:
        """Return ``(content, captured_at)`` for a retained version, or ``None``.

        The single accessor Unit 4's ``read_at_version`` needs: the body AND its
        wall-clock capture timestamp (for ``VersionedContent.captured_at`` and the
        T-expiry check) in ONE call. Mirrors
        :meth:`SqliteArtifactRegistry.get_version_record` so the two registries
        share one duck-type.

        Single-scope (R5 atomicity): ``version_history`` and
        ``version_captured_at`` are always mutated together (capture and GC), so
        their key sets match; reading the body first and missing means the row is
        absent. Each dict read is GIL-atomic, but the PAIR is not: a concurrent
        inline GC can drop THIS version between the two reads, so the stamp is
        read defensively (``.get``) and an absent stamp after a present body is
        reported as absent — the row was collected mid-read. ``None`` when the
        artifact or the version is absent (never captured, K-evicted, or
        T-expired-then-swept)."""
        with self._lock:
            record = self._records.get(artifact_id)
            if record is None:
                return None
            content = record.version_history.get(version)
            if content is None:
                return None
            # One .get() per dict, no bare getitem: the GIL makes each dict access
            # atomic but NOT the pair — a concurrent inline GC (a peer thread's
            # capture under a bounded policy) can drop THIS version between the two
            # reads, and a getitem here would KeyError out of read_at_version's
            # never-raise contract. An absent stamp after a present body simply
            # means the row was GC'd mid-read ⇒ report it as not retained.
            captured_at = record.version_captured_at.get(version)
            if captured_at is None:
                return None
            return content, captured_at

    def get_state_map(self, artifact_id: UUID) -> dict[UUID, MESIState]:
        """Return copy of per-agent MESI states for an artifact."""
        with self._lock:
            record = self._records.get(artifact_id)
            return dict(record.state_by_agent) if record else {}

    def get_agent_state(self, artifact_id: UUID, agent_id: UUID) -> MESIState | None:
        """Return MESI state for one agent/artifact pair if present.

        Absent-artifact tolerance (this and every read accessor except the
        deliberately KeyError-raising ``get_owner_generation``): answers match
        sqlite's empty SELECT -- None/{}/[] -- because under the thread-safety
        contract a delete may land between a sweep's snapshot and its per-pair
        hold, and the hold's re-read must skip the vanished pair, not crash
        the walk."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.state_by_agent.get(agent_id) if record else None

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
        """Set MESI state for one agent/artifact pair.

        ``observed=False``: the grant certifies no read, so the recorded
        ``last_observed_version`` is left as it was (see the Protocol)."""
        with self._lock:
            record = self._records[artifact_id]
            from_state = record.state_by_agent.get(agent_id, MESIState.INVALID)
            record.state_by_agent[agent_id] = state

            # Crash-recovery bookkeeping (no log emit, no serialization). Runs
            # unconditionally and BEFORE the log emit so a state_log raise cannot
            # leave state_by_agent and granted_at_tick_by_agent inconsistent
            # (review fix COR-01 / REL-01: previously, a failed log emit left an
            # M∪E entry without a granted_at_tick slot, defeating max-hold reclaim).
            # Keeping the hot path branch-free preserves R5 byte-identity (these
            # are dict mutations only, never serialized) and avoids subtle
            # flag-on/flag-off divergence.
            new_in_me = state in _M_OR_E_STATES
            prev_in_me = from_state in _M_OR_E_STATES
            if new_in_me:
                if not prev_in_me:
                    # Set granted_at_tick on M∪E acquire only; M↔E transitions preserve the
                    # original grant tick (the agent has continuously held some M∪E grant).
                    record.granted_at_tick_by_agent[agent_id] = tick
                    # Slot clears on M∪E acquire ONLY (not on SHARED) — preserves the
                    # checkpoint-restore diagnostic across SHARED re-fetches.
                    record.last_reclamation_by_agent.pop(agent_id, None)
            elif prev_in_me:
                record.granted_at_tick_by_agent.pop(agent_id, None)
                # Read-generation fence: revoking this M/E grant WITHOUT moving the
                # version bumps the artifact's ownership epoch, atomically (GIL)
                # with the INVALID transition, so a commit by the ex-holder (or any
                # pre-revocation holder) fails the generation check. Both the sweep
                # reclaims and the voluntary release ("invalidate") qualify — see
                # EPOCH_BUMP_TRIGGERS. The peer invalidations ("write" /
                # "commit") do not -- a "commit" moves the version and a
                # "write" preemption is fenced by grant-state at the effect
                # gate and by the M/E state check at commit, never by the
                # epoch (see registry_protocol.EPOCH_BUMP_TRIGGERS).
                if trigger in EPOCH_BUMP_TRIGGERS:
                    record.owner_generation += 1

            # Read-generation fence: capture the current ownership epoch into the
            # agent's read_generation ONLY on the agent's own claim -- an E/M
            # acquire (including a pessimistic acquire with no prior content
            # read) or a genuine fetch read into S/E. Atomic (GIL) with the grant.
            # Two guards on the fetch leg. INVALID: no current fetch path grants
            # INVALID, but a cache-miss-INVALID fetch must not mint a fresh claim
            # for an unfenced zombie. Not-leaving-M/E: service.fetch tags its
            # peer downgrade (M/E -> S) "fetch" too, and leaving M/E is a loss of
            # authority, never a claim -- refreshing here would re-arm a
            # superseded value and un-stick a stale_read_generation rejection.
            # service.fetch no longer rewrites an already-SHARED peer, so an
            # S -> S "fetch" reaching this predicate is always the agent's own
            # re-read, which must keep capturing (the reclaimed-reader recovery
            # path).
            if (new_in_me and not prev_in_me) or (
                trigger in CLAIM_CAPTURE_TRIGGERS
                and state != MESIState.INVALID
                and not prev_in_me
            ):
                record.read_generation_by_agent[agent_id] = record.owner_generation

            # SB-10 R6/R7: record the version whose bytes this agent now holds,
            # GIL-atomic with the state write above (parity: the sqlite side writes
            # it inside the same BEGIN IMMEDIATE as the upsert). Non-INVALID
            # targets only -- a transition TO INVALID preserves the prior recorded
            # value (the last version actually observed, the post-compaction
            # staleness comparand) and a never-observed agent keeps no key (absent
            # == None, never a 0-sentinel). A grant the caller says certifies no
            # read (``observed=False``: a denied command's re-grant) preserves
            # the same way -- a SHARED holder can then sit BELOW the current
            # version, which every reader treats as "behind", the safe side.
            if state != MESIState.INVALID and observed:
                record.last_observed_version_by_agent[agent_id] = record.artifact.version

            if self._state_log is not None:
                self._seq += 1
                entry = {
                    "tick": tick,
                    "artifact_id": str(artifact_id),
                    "agent_id": str(agent_id),
                    "agent_name": self._agent_names.get(agent_id) if self._agent_names is not None else None,
                    "from_state": from_state.name,
                    "to_state": state.name,
                    "trigger": trigger,
                    "version": record.artifact.version,
                    "content_hash": content_hash,
                    "sequence_number": self._seq,
                    "instance_id": self._instance_id,
                    "schema_version": CCS_STATE_LOG_SCHEMA_VERSION,
                }
                try:
                    self._state_log(entry)
                except Exception:
                    # Sequence number is reserved on success, not on attempt.
                    # Roll back so the next successful emission does not create a phantom gap.
                    self._seq -= 1
                    raise

    def _note_conflict(self, artifact_id: UUID, agent_id: UUID, reason: str) -> None:
        """Count one typed deny at its construction site and notify observers.

        Called ONLY where a ``ConflictDetail`` is actually returned to the
        caller (never for :class:`CasCorruption`, never mid-``commit_all``
        before the aggregate is decided). The compound get-then-set serializes
        on the registry-wide ``self._lock`` (already held at every deny
        branch; reentrant). KTD7 applies to ``conflict_callbacks``: they fire
        while that lock is HELD, must never block on other threads, and each
        is exception-guarded so an observer crash cannot alter the typed
        outcome already decided."""
        key = (artifact_id, agent_id, reason)
        with self._lock:
            self._conflict_counts[key] = self._conflict_counts.get(key, 0) + 1
        for callback in self.conflict_callbacks:
            try:
                callback(artifact_id, agent_id, reason)
            except Exception:  # noqa: BLE001 — observer isolation by design
                logger.exception("conflict callback raised; deny outcome unaffected")

    def conflict_outcome_totals(self) -> dict[tuple[UUID, UUID, str], int]:
        """Return the per-(artifact, agent, reason) typed-deny counts.

        Zero conflicts → an empty dict — zero is a reportable result, not an
        error. Process-scoped for this registry (parity divergence asserted,
        not masked: durability is the sqlite registry's job)."""
        with self._lock:
            return dict(self._conflict_counts)

    # ------------------------------------------------------------------
    # Foreign-write detection instrumentation (detection-substrate plan U2).
    # Behavioural parity with the sqlite registry; durability is deliberately
    # NOT claimed here, exactly as the conflict counters above declare.
    # ------------------------------------------------------------------

    def record_detection_tick(self, now_unix: float, *, covered_count: int = 0) -> None:
        """Record that the detector observed one tick in this interval.

        See :meth:`SqliteArtifactRegistry.record_detection_tick`. Only a tick
        whose poll actually succeeded reaches here — an advancing count over a
        broken poll would read as a quiet month."""
        with self._lock:
            run = self._detection_run
            if run is None:
                self._detection_run = DetectionRun(
                    run_id=self._detection_run_id,
                    first_tick_unix=now_unix,
                    last_tick_unix=now_unix,
                    tick_count=1,
                    covered_count=covered_count,
                )
            else:
                self._detection_run = DetectionRun(
                    run_id=run.run_id,
                    first_tick_unix=run.first_tick_unix,
                    last_tick_unix=now_unix,
                    tick_count=run.tick_count + 1,
                    covered_count=max(run.covered_count, covered_count),
                )

    def close_detection_run(self) -> None:
        """End the current observed interval and open a fresh one.

        See :meth:`SqliteArtifactRegistry.close_detection_run` — a failed tick
        must not let a later success extend an interval across the outage."""
        with self._lock:
            if self._detection_run is not None:
                self._detection_runs_closed.append(self._detection_run)
                self._detection_run = None
            self._detection_run_id = uuid4().hex

    def record_detection_uncoverable(self, reason: str, now_unix: float) -> None:
        """Record that this run found nothing it could ever watch here.

        See :meth:`SqliteArtifactRegistry.record_detection_uncoverable` for why
        this is a third fact rather than a tick row with a zero count."""
        with self._lock:
            self._detection_uncoverable[self._detection_run_id] = UncoverableRun(
                run_id=self._detection_run_id,
                reason=reason,
                observed_at_unix=now_unix,
            )

    def detection_uncoverable(self) -> list[UncoverableRun]:
        """Runs that found no watchable workspace; empty is a real answer."""
        with self._lock:
            return sorted(
                self._detection_uncoverable.values(),
                key=lambda run: run.observed_at_unix,
            )

    def record_foreign_write(
        self, artifact_id: UUID, outcome: str, disk_hash: str
    ) -> bool:
        """Count one newly observed on-disk content, edge-gated on that content.

        See :meth:`SqliteArtifactRegistry.record_foreign_write` for why the gate
        keys on content rather than on the tick, and why the suppressing value
        lives here rather than on the canonical hash."""
        if outcome not in FOREIGN_WRITE_OUTCOMES:
            raise ValueError(
                f"unknown foreign-write outcome {outcome!r}; "
                f"expected one of {FOREIGN_WRITE_OUTCOMES}"
            )
        # Keyed by the outcome name directly — this backend stores a dict, so
        # it has no column names to bind and needs no equivalent of the sqlite
        # registry's outcome-to-column map.
        with self._lock:
            if self._detection_last.get(artifact_id) == (disk_hash, outcome):
                return False
            self._detection_last[artifact_id] = (disk_hash, outcome)
            counts = self._detection_counts.setdefault(artifact_id, {})
            counts[outcome] = counts.get(outcome, 0) + 1
        return True

    def artifacts_with_detection_edge(self) -> set[UUID]:
        """Artifacts currently holding an edge-gate value."""
        with self._lock:
            return set(self._detection_last)

    def clear_detection_edges(self, artifact_ids: list[UUID]) -> None:
        """Re-arm detection for artifacts that are no longer diverging."""
        with self._lock:
            for artifact_id in artifact_ids:
                self._detection_last.pop(artifact_id, None)

    def foreign_write_totals(self) -> dict[UUID, dict[str, int]]:
        """Return per-artifact detection counts; zero detections is an empty
        mapping, a reportable result rather than an error."""
        with self._lock:
            return {art: dict(counts) for art, counts in self._detection_counts.items()}

    def detection_runs(self) -> list[DetectionRun]:
        """Return this run's observed interval, or an empty list when the
        detector never ticked — the not-instrumented signal."""
        with self._lock:
            runs = list(self._detection_runs_closed)
            if self._detection_run is not None:
                runs.append(self._detection_run)
            return runs

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
        """In-memory optimistic-concurrency compare-and-swap (plan Unit 2,
        parity with :meth:`SqliteArtifactRegistry.commit_cas`). Written fresh
        from the contract — there is no ``resolve_or_register`` precedent here
        and the two registries share no base class. One method-wide ``_lock``
        hold stands in for the sqlite ``BEGIN IMMEDIATE``: the arbitration
        legs and the apply are a single serialized step (U4).

        Same 3-outcome discrimination, version check BEFORE the holder check:

        - ``expected_version > current`` → :class:`CasCorruption` (no mutation).
        - ``expected_version < current`` → ``ConflictDetail("version_mismatch")``
          (no mutation).
        - version matches but another agent holds M/E → ``ConflictDetail(
          "other_holder")`` (no mutation).
        - else → WIN: version → ``current + 1``, committer S/I → SHARED (an OCC
          writer holds no grant — SHARED keeps its next commit_cas repeatable;
          MODIFIED would trip the service D4 precondition), peers → INVALID;
          returns ``(updated_artifact, invalidated_agent_ids)``.

        The state-log emit follows the same mutation-then-log + ``_seq``-rollback
        invariant as :meth:`set_agent_state`. To keep that invariant under a
        callback raise during peer/committer logging, the in-memory mutations
        are computed into a staging plan and applied only after all log entries
        emit successfully — so a raise leaves ``state_by_agent`` /
        ``granted_at_tick`` untouched (matching the sqlite ROLLBACK).

        ``content`` is the winning body. When provided (the in-process library
        path threads it from ``AgentRuntime.write_cas``) the WIN updates
        ``record.content`` to the NEW body, so a peer re-fetching after the win
        reads the winner's content at the new version — not the stale pre-CAS
        body, and (when versions are retained) captures that NEW body under
        ``next_version``. ``None`` (the cross-process / sqlite path, which stores
        no content) leaves the prior content-coherence behaviour unchanged AND
        skips the version capture (it is not retained under ``next_version``);
        previously the None path retained the stale OLD body under the new
        version, a latent history-poisoning bug now fixed.
        """
        with self._lock:
            record = self._records.get(artifact_id)
            if record is None:
                raise KeyError(f"artifact {artifact_id} not in registry")
            current = record.artifact.version

            if expected_version > current:
                return CasCorruption(current_version=current)
            if expected_version < current:
                detail = ConflictDetail("version_mismatch", current)
                self._note_conflict(artifact_id, agent_id, detail.reason)
                return detail
            other_holder = any(
                peer_id != agent_id and state in _M_OR_E_STATES
                for peer_id, state in record.state_by_agent.items()
            )
            if other_holder:
                detail = ConflictDetail("other_holder", current)
                self._note_conflict(artifact_id, agent_id, detail.reason)
                return detail

            # Read-generation fence: reject a committer whose CAPTURED read-claim
            # was superseded by a sweep reclamation. A reclaimed M/E holder kept its
            # stale read_generation (captured at acquire), so it is caught here even
            # though the version is unchanged and no peer holds M/E -- exactly what
            # version-CAS cannot catch. An ABSENT read_generation means the committer
            # never established a fence claim: a plain OCC writer whose lost-update
            # protection is version-CAS (checked above), so it is admitted. Strict->;
            # equality admits. Server-side; no commit_cas signature change.
            read_gen = record.read_generation_by_agent.get(agent_id)
            if read_gen is not None and read_gen < record.owner_generation:
                detail = ConflictDetail("stale_read_generation", current)
                self._note_conflict(artifact_id, agent_id, detail.reason)
                return detail

            # ---- WIN ----
            next_version = current + 1
            committer_from = record.state_by_agent.get(agent_id, MESIState.INVALID)
            peers = [
                (peer_id, state)
                for peer_id, state in record.state_by_agent.items()
                if peer_id != agent_id and state != MESIState.INVALID
            ]

            # Emit all state_log entries FIRST (peers then committer), reserving
            # _seq per emission. If any raises, _emit_state_log has already
            # decremented its own reservation; we decrement the ones that already
            # succeeded in this call and re-raise — nothing has mutated yet, so the
            # registry stays consistent (mutation-then-log parity).
            emitted_here = 0
            try:
                for peer_id, peer_from in peers:
                    emitted_here += self._emit_state_log(
                        artifact_id=artifact_id,
                        agent_id=peer_id,
                        from_state=peer_from,
                        to_state=MESIState.INVALID,
                        trigger=trigger,
                        tick=tick,
                        version=next_version,
                        content_hash=None,
                    )
                emitted_here += self._emit_state_log(
                    artifact_id=artifact_id,
                    agent_id=agent_id,
                    from_state=committer_from,
                    to_state=MESIState.SHARED,
                    trigger=trigger,
                    tick=tick,
                    version=next_version,
                    content_hash=content_hash,
                )
            except Exception:
                self._seq -= emitted_here
                raise

            # All logs emitted — now apply the mutations (cannot fail).
            updated = Artifact(
                id=artifact_id,
                name=record.artifact.name,
                version=next_version,
                content_hash=content_hash,
                size_tokens=size_tokens if size_tokens is not None else record.artifact.size_tokens,
                depends_on=record.artifact.depends_on,
            )
            # The whole method runs under ONE _lock hold (Unit 2 -> U4): the
            # arbitration legs above (version, holder, fence) and the apply
            # below are a single check-then-act -- deciding on a version read
            # outside the hold and applying inside it was the same class of gap
            # as the sweep race. A concurrent multi-artifact capture reads N
            # versions under this same lock, so the WIN cannot tear a cut
            # (read skew) either. The RLock re-enters for _capture_version.
            # Content coherence: when the caller threaded the winning body,
            # advance record.content to it so a peer re-fetch reads the NEW
            # content at the new version (without this, version + content_hash
            # bump but the body stays stale). content is None on the cross-process
            # / sqlite path (no content stored) — keep the prior body unchanged.
            if content is not None:
                record.content = content
            # Retention capture joins this APPLIED block (after every state_log
            # emit succeeded) so a callback raise above cannot leave phantom
            # history — parity with the stage-then-apply discipline for the MESI
            # mutations. content=None SKIPS capture entirely (R-fix): the old code
            # retained the OLD body under the NEW version when content was None —
            # a latent history-poisoning bug only observable through retention
            # reads. Now an unsupplied body simply means "no snapshot for this
            # version"; a later read of next_version misses (None).
            if self._retain_versions and content is not None:
                self._capture_version(record, next_version, content)
            record.artifact = updated
            record.last_writer = agent_id

            invalidated: list[UUID] = []
            for peer_id, peer_from in peers:
                record.state_by_agent[peer_id] = MESIState.INVALID
                if peer_from in _M_OR_E_STATES:
                    record.granted_at_tick_by_agent.pop(peer_id, None)
                invalidated.append(peer_id)

            # Committer S/I → SHARED, NOT MODIFIED: an OCC writer is optimistic and
            # holds no grant, so SHARED is the honest end-state and keeps the same
            # agent's next commit_cas repeatable (a sticky MODIFIED would trip the
            # service D4 "M/E callers use commit()" precondition). SHARED is not in
            # M∪E, so this is not an acquire — do NOT set granted_at_tick and do NOT
            # clear the reclaim slot (mirror set_agent_state's non-M/E-from-non-M/E
            # path, which leaves both untouched).
            record.state_by_agent[agent_id] = MESIState.SHARED
            # SB-10 R6/R7 (KTD4 first layer): the WIN advances the WRITER's
            # last_observed_version to the version it just produced; the
            # invalidated peers above keep their prior values.
            record.last_observed_version_by_agent[agent_id] = next_version

            return updated, invalidated

    def commit_all(
        self,
        agent_id: UUID,
        writes: "Mapping[UUID, CommitAllEntry]",
        *,
        tick: int = 0,
        trigger: str = "commit_all",
    ) -> "MultiCommitResult | MultiCommitConflict | CasCorruption":
        """In-memory ATOMIC multi-artifact publish (SB-18 / commit_all, Unit 2).

        All-or-nothing: either EVERY member of ``writes`` advances to its next
        version, or ZERO do and the batch is HELD. This is a genuinely new atomic
        registry op — **NOT a loop of** :meth:`commit_cas` — structured as a single
        CHECK-all → STAGE → APPLY-total → (caller) BROADCAST-after-commit critical
        section, so no peer ever observes a partial batch (``NoPartialPublish``,
        ``AtomicPublish.tla``; parity with
        :meth:`SqliteArtifactRegistry.commit_all`).

        Outcomes (all typed returns, never a raise for a normal conflict):

        - Any member corrupt (``expected_version > current``) → :class:`CasCorruption`
          (the whole batch aborts; the service maps it to ``CoherenceError``).
        - Else any member a retry-eligible conflict → :class:`MultiCommitConflict`
          naming EVERY failing member's own :class:`ConflictDetail`
          (``version_mismatch`` / ``other_holder`` / ``stale_read_generation``).
        - Else WIN → :class:`MultiCommitResult` (``versions`` per member + the
          aggregated ``invalidated`` peer set for post-commit broadcast).

        CHECK is pure reads; STAGE computes every mutation; the state-log emit and
        the APPLY both roll back in full on any raise (snapshot-then-restore), so a
        mid-apply exception can never leave a torn batch.
        """
        with self._lock:
            if not writes:
                raise ValueError("commit_all requires a non-empty write-set")

            # ---- CHECK: every member, pure reads, no mutation (all-or-nothing) ----
            conflicts: dict[UUID, ConflictDetail] = {}
            for art_id, entry in writes.items():
                record = self._records.get(art_id)
                if record is None:
                    raise KeyError(f"artifact {art_id} not in registry")
                current = record.artifact.version
                if entry.expected_version > current:
                    # Corruption aborts the whole batch (all-or-nothing); non-retryable.
                    return CasCorruption(current_version=current)
                if entry.expected_version < current:
                    conflicts[art_id] = ConflictDetail("version_mismatch", current)
                    continue
                other_holder = any(
                    peer_id != agent_id and state in _M_OR_E_STATES
                    for peer_id, state in record.state_by_agent.items()
                )
                if other_holder:
                    conflicts[art_id] = ConflictDetail("other_holder", current)
                    continue
                read_gen = record.read_generation_by_agent.get(agent_id)
                if read_gen is not None and read_gen < record.owner_generation:
                    conflicts[art_id] = ConflictDetail("stale_read_generation", current)
                    continue

            if conflicts:
                # Count each failing member only now that the aggregate deny is
                # decided — a mid-CHECK KeyError raise counts nothing.
                for art_id, detail in conflicts.items():
                    self._note_conflict(art_id, agent_id, detail.reason)
                return MultiCommitConflict(per_artifact=dict(conflicts))

            # ---- STAGE: compute all mutations (nothing applied yet) ----
            staged = []
            for art_id, entry in writes.items():
                record = self._records[art_id]
                next_version = record.artifact.version + 1
                committer_from = record.state_by_agent.get(agent_id, MESIState.INVALID)
                peers = [
                    (peer_id, state)
                    for peer_id, state in record.state_by_agent.items()
                    if peer_id != agent_id and state != MESIState.INVALID
                ]
                updated = Artifact(
                    id=art_id,
                    name=record.artifact.name,
                    version=next_version,
                    content_hash=entry.content_hash,
                    size_tokens=(
                        entry.size_tokens
                        if entry.size_tokens is not None
                        else record.artifact.size_tokens
                    ),
                    depends_on=record.artifact.depends_on,
                )
                staged.append((art_id, record, entry, next_version, committer_from, peers, updated))

            # ---- EMIT all state_log entries FIRST, across the whole batch, with a
            # running _seq rollback total so a raise leaves the registry unmutated. ----
            emitted_here = 0
            try:
                for art_id, record, entry, next_version, committer_from, peers, updated in staged:
                    for peer_id, peer_from in peers:
                        emitted_here += self._emit_state_log(
                            artifact_id=art_id,
                            agent_id=peer_id,
                            from_state=peer_from,
                            to_state=MESIState.INVALID,
                            trigger=trigger,
                            tick=tick,
                            version=next_version,
                            content_hash=None,
                        )
                    emitted_here += self._emit_state_log(
                        artifact_id=art_id,
                        agent_id=agent_id,
                        from_state=committer_from,
                        to_state=MESIState.SHARED,
                        trigger=trigger,
                        tick=tick,
                        version=next_version,
                        content_hash=entry.content_hash,
                    )
            except Exception:
                self._seq -= emitted_here
                raise

            # ---- APPLY (total): snapshot every affected record, apply all N, and
            # RESTORE ALL on any raise — a mid-apply exception can never leave a
            # partial batch (the plan's total-apply hardening). The method-wide
            # _lock hold (U4) already covers this block AND the CHECK legs above:
            # check-all and apply-total are one serialized step, as the contract
            # says of the sqlite BEGIN IMMEDIATE. ----
            invalidated: dict[UUID, list[UUID]] = {}
            versions: dict[UUID, int] = {}
            snapshots = [
                (
                    record,
                    record.artifact,
                    record.content,
                    dict(record.state_by_agent),
                    dict(record.granted_at_tick_by_agent),
                    record.last_writer,
                    # Retention history is mutated in-place by _capture_version
                    # (add + GC-pop); snapshot it too so a mid-apply raise restores
                    # it, matching sqlite's ROLLBACK (which undoes the version-
                    # history table). Cheap dict copies; retention is the only
                    # other per-record state the apply touches.
                    dict(record.version_history),
                    dict(record.version_captured_at),
                    # SB-10: the apply advances the committer's observed-version
                    # slot per member; a mid-apply raise must restore it with
                    # the rest (sqlite's ROLLBACK undoes the column write).
                    dict(record.last_observed_version_by_agent),
                )
                for _art, record, *_rest in staged
            ]
            try:
                for art_id, record, entry, next_version, committer_from, peers, updated in staged:
                    if entry.content is not None:
                        record.content = entry.content
                    if self._retain_versions and entry.content is not None:
                        self._capture_version(record, next_version, entry.content)
                    record.artifact = updated
                    record.last_writer = agent_id
                    member_invalidated: list[UUID] = []
                    for peer_id, peer_from in peers:
                        record.state_by_agent[peer_id] = MESIState.INVALID
                        if peer_from in _M_OR_E_STATES:
                            record.granted_at_tick_by_agent.pop(peer_id, None)
                        member_invalidated.append(peer_id)
                    if member_invalidated:
                        invalidated[art_id] = member_invalidated
                    record.state_by_agent[agent_id] = MESIState.SHARED
                    # SB-10 R6/R7 (KTD4 first layer): each member's WIN advances
                    # the WRITER's observed version to that member's
                    # next_version; invalidated peers keep theirs.
                    record.last_observed_version_by_agent[agent_id] = next_version
                    versions[art_id] = next_version
            except Exception:
                # Total apply: restore every mutated record, roll back the logs.
                for (rec, art_obj, content, state_by, granted, last_w, ver_hist, ver_at, observed) in snapshots:
                    rec.artifact = art_obj
                    rec.content = content
                    rec.state_by_agent = state_by
                    rec.granted_at_tick_by_agent = granted
                    rec.last_writer = last_w
                    rec.version_history = ver_hist
                    rec.version_captured_at = ver_at
                    rec.last_observed_version_by_agent = observed
                self._seq -= emitted_here
                raise

            # BROADCAST is the caller's (service) responsibility, AFTER this returns —
            # the per-artifact invalidations are published to the event bus only post-commit.
            return MultiCommitResult(
                versions=versions,
                invalidated={art: tuple(peers) for art, peers in invalidated.items()},
            )

    def capture_version_vector(
        self,
        read_set: "Iterable[UUID]",
        session_token: str,
        *,
        owner: "UUID | None" = None,
        created_at_tick: int | None = None,
    ) -> CaptureResult:
        """Atomically pin a consistent multi-artifact CUT (SB-17 / TX-1, Unit 2 /
        R1). Written fresh from the contract — parity with
        :meth:`SqliteArtifactRegistry.capture_version_vector` (the two registries
        share no base class), divergent ONLY on restart-survival (in-memory pins
        are process-scoped; the parity harness asserts the divergence).

        Captures ``{artifact_id: current_version}`` for every id in ``read_set``
        at ONE linearization point and records the pins under ``session_token``
        so the inline retention GC exempts those versions (R4). Non-mutating: it
        mints NO MESI grant and captures NO ``read_generation`` (it never calls
        ``set_agent_state`` / the fence-capture path) — a reader is not an owner.

        Atomicity (Unit 2): GIL per-access atomicity is per single dict
        access, which is insufficient across N artifacts. The multi-artifact
        version read AND the pin insert run under ``_lock`` so a peer
        ``commit_cas`` cannot interleave the reads and yield a torn
        (read-skewed) cut.

        Unknown-id validation (security, F7): the captured row-count must equal
        ``len(read_set)``. Any missing id → a typed
        :class:`~ccs.core.types.VersionedReadRejection` reusing the
        ``unknown_artifact`` reason, with NO pins inserted (no partial cut, no
        existence-probe oracle — the rejection is decided before any pin write).
        The first missing id (sorted for determinism) names the rejection.

        Args:
            read_set: The artifact ids to pin into the cut. An empty read_set
                pins an empty cut (a degenerate-but-valid session).
            session_token: The server-minted session identity the pins are keyed
                under (owner-binding is the service layer's concern; the registry
                only keys the pin store by this token).

        Returns:
            The pinned cut ``{artifact_id: version}`` on success, else a
            :class:`VersionedReadRejection` (``unknown_artifact``) — no pins
            inserted.
        """
        ids = list(read_set)
        with self._lock:
            # ONE linearization point: read every current version while holding
            # the capture lock. Every version-moving write — commit_cas WIN,
            # set_artifact_and_content, register_artifact — ALSO takes
            # _lock around its apply, so a peer write is serialized
            # entirely before or after this whole capture, never partially
            # visible within the cut. The atomicity depends on BOTH sides taking
            # the lock; do NOT remove it from either side.
            cut: dict[UUID, int] = {}
            missing: list[UUID] = []
            for artifact_id in ids:
                record = self._records.get(artifact_id)
                if record is None:
                    missing.append(artifact_id)
                    continue
                cut[artifact_id] = record.artifact.version
            # F7: validate row-count == len(read_set) BEFORE any pin insert. A
            # missing id rejects the WHOLE capture (no partial cut). Decided
            # inside the lock but it inserts nothing on this branch.
            if missing:
                return VersionedReadRejection(
                    reason=UNKNOWN_ARTIFACT_REASON,
                    artifact_id=sorted(missing, key=lambda a: a.int)[0],
                    requested_version=0,
                    current_version=None,
                    coordinator_epoch=self._coordinator_epoch,
                )
            # Insert the pins atomically with the read (same lock hold) so the
            # exemptions seam sees the full cut the instant it becomes live.
            self._session_pins[session_token] = dict(cut)
            # Mirror the durable owner-binding (R13/R6/R14) so the sweep can
            # enumerate sessions uniformly. Recorded even for an empty cut, and
            # only when the caller supplies an owner (direct test captures pass
            # none and create no session-meta entry).
            if owner is not None:
                self._session_meta[session_token] = (owner, int(created_at_tick or 0))
            return cut

    def release_session(self, session_token: str) -> None:
        """Drop a session's pins AND its owner-binding mirror so its pinned
        versions become collectible again (Unit 2 / R4). Idempotent — releasing
        an unknown/already-released token is a no-op (no raise), mirroring the
        sqlite ``DELETE`` semantics."""
        with self._lock:
            self._session_pins.pop(session_token, None)
            self._session_meta.pop(session_token, None)

    def get_session_meta(self, session_token: str) -> "tuple[UUID, int] | None":
        """Return ``(owner, created_at_tick)`` for ``session_token`` or ``None``
        (SB-17 / TX-1, R13/R6/R14). Parity with
        :meth:`SqliteArtifactRegistry.get_session_meta`; process-scoped here, so a
        fresh in-memory instance always returns ``None`` (in-memory sessions do
        not survive a restart — the asserted divergence)."""
        with self._lock:
            return self._session_meta.get(session_token)

    def all_session_meta(self) -> "dict[str, tuple[UUID, int]]":
        """Return ``{session_token: (owner, created_at_tick)}`` for every live
        session (SB-17 / TX-1, R6/R14). Parity with
        :meth:`SqliteArtifactRegistry.all_session_meta`; the sweep enumerates this
        UNION'd with its in-memory token set (identical here, since both are
        process-scoped)."""
        with self._lock:
            return dict(self._session_meta)

    def session_count(self) -> int:
        """Return the number of live sessions (SB-17 / TX-1, R14). Parity with
        :meth:`SqliteArtifactRegistry.session_count`; process-scoped, so a fresh
        instance is 0 (no durable survivors)."""
        with self._lock:
            return len(self._session_meta)

    def get_session_cut(self, session_token: str) -> dict[UUID, int] | None:
        """Return the pinned cut ``{artifact_id: version}`` for ``session_token``,
        or ``None`` if the token has no live cut (SB-17 / TX-1, Unit 3 / R2).

        The single accessor ``session_read`` needs to (a) tell a known token from
        an unknown/released one (``None`` ⇒ ``session_not_found``) and (b) read
        the per-artifact pinned version for the serve. Returns a COPY so a caller
        cannot mutate the live pin store. Parity with
        :meth:`SqliteArtifactRegistry.get_session_cut` (the two registries share
        no base class); divergent only on restart-survival (in-memory pins are
        process-scoped — a restart drops them and a post-restart token reads as
        ``None``, the Unit-5 fail-closed concern). Taken under ``_lock``
        so a concurrent capture/release sees a consistent pin store."""
        with self._lock:
            pins = self._session_pins.get(session_token)
            return dict(pins) if pins is not None else None

    # ------------------------------------------------------------------
    # Caller-principal bindings (caller-principal plan, U4)
    # ------------------------------------------------------------------

    def bind_caller_principal(
        self, identity: UUID, principal: str, mint_nonce: str
    ) -> tuple[str, str]:
        """First-claim-wins bind; returns the BOUND ``(principal, mint_nonce)``
        pair. Parity with :meth:`SqliteArtifactRegistry.bind_caller_principal`:
        the check and the insert share one ``_lock`` hold, so two concurrent
        first claims bind one principal."""
        with self._lock:
            return self._caller_principals.setdefault(identity, (principal, mint_nonce))

    def get_caller_principal(self, identity: UUID) -> str | None:
        """The principal bound to ``identity``, or ``None`` when unclaimed."""
        with self._lock:
            bound = self._caller_principals.get(identity)
        return bound[0] if bound is not None else None

    # ------------------------------------------------------------------
    # Workspace-checkpoint manifest store (WV plan Unit 2 / R1, R9)
    # ------------------------------------------------------------------

    def create_checkpoint(
        self,
        checkpoint: CheckpointRecord,
        members: Sequence[CheckpointMember],
    ) -> None:
        """Persist a checkpoint manifest atomically under ``_lock`` —
        the header (owner metadata included, same critical section: the
        in-memory mirror of the sqlite same-transaction rule) plus every member,
        or nothing. Parity with
        :meth:`SqliteArtifactRegistry.create_checkpoint`; divergent only on
        restart-survival (in-memory manifests are process-scoped).

        Fail-closed owner contract: an absent owner raises ``ValueError`` and
        nothing is stored. A duplicate ``checkpoint_id`` or duplicate member
        path raises ``ValueError`` with nothing stored (validated before the
        first insert — no partial manifest)."""
        if checkpoint.owner is None:
            raise ValueError(
                "create_checkpoint requires an owner: a checkpoint manifest "
                "without owner metadata is unrepresentable (fail-closed; the "
                "restore path owner-validates against it)"
            )
        with self._lock:
            if checkpoint.checkpoint_id in self._checkpoints:
                raise ValueError(
                    f"create_checkpoint: duplicate checkpoint_id "
                    f"{checkpoint.checkpoint_id!r}"
                )
            member_by_path: dict[str, CheckpointMember] = {}
            for member in members:
                if member.member_path in member_by_path:
                    raise ValueError(
                        f"create_checkpoint: duplicate member path "
                        f"{member.member_path!r} in the manifest for checkpoint "
                        f"{checkpoint.checkpoint_id!r}"
                    )
                member_by_path[member.member_path] = member
            # Both dicts are written inside ONE lock hold, after every
            # validation passed — all rows land or none do (the in-memory
            # equivalent of the sqlite single transaction).
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
            self._checkpoint_members[checkpoint.checkpoint_id] = member_by_path

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        """Return the checkpoint header, or ``None`` when unknown."""
        with self._lock:
            return self._checkpoints.get(checkpoint_id)

    def list_checkpoints(self) -> list[CheckpointRecord]:
        """Return every checkpoint header, ordered by ``(created_at,
        checkpoint_id)`` (deterministic for the CLI ``list`` verb)."""
        with self._lock:
            return sorted(
                self._checkpoints.values(),
                key=lambda c: (c.created_at, c.checkpoint_id),
            )

    def get_checkpoint_members(self, checkpoint_id: str) -> list[CheckpointMember]:
        """Return the manifest's member rows ordered by ``member_path`` (empty
        list for an unknown checkpoint — :meth:`get_checkpoint` tells known from
        unknown)."""
        with self._lock:
            member_by_path = self._checkpoint_members.get(checkpoint_id, {})
            return [member_by_path[path] for path in sorted(member_by_path)]

    def set_checkpoint_restore_status(
        self, checkpoint_id: str, status: str, *, updated_at: float
    ) -> None:
        """Update the checkpoint-level restore status + its ``updated_at`` stamp.
        Raises ``KeyError`` for an unknown checkpoint (fail fast, no silent
        no-op). Frozen records are replaced, never mutated in place."""
        with self._lock:
            record = self._checkpoints.get(checkpoint_id)
            if record is None:
                raise KeyError(f"checkpoint {checkpoint_id!r} not in registry")
            self._checkpoints[checkpoint_id] = replace(
                record, restore_status=status, restore_updated_at=updated_at
            )

    def set_checkpoint_member_restore(
        self,
        checkpoint_id: str,
        member_path: str,
        *,
        restore_outcome: str | None,
        deleted_at_restore: float | None = None,
    ) -> None:
        """Record a member's restore progress: BOTH fields are written to the
        given values (a full member-restore-state write — a new restore run's
        first write for a member resets any prior run's delete record). Raises
        ``KeyError`` for an unknown (checkpoint, member) pair."""
        with self._lock:
            member = self._checkpoint_members.get(checkpoint_id, {}).get(member_path)
            if member is None:
                raise KeyError(
                    f"member {member_path!r} of checkpoint {checkpoint_id!r} "
                    f"not in registry"
                )
            self._checkpoint_members[checkpoint_id][member_path] = replace(
                member,
                restore_outcome=restore_outcome,
                deleted_at_restore=deleted_at_restore,
            )

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
        downgrade. ``restore_tier=None`` leaves the tier untouched. Raises
        ``KeyError`` for an unknown (checkpoint, member) pair."""
        with self._lock:
            member = self._checkpoint_members.get(checkpoint_id, {}).get(member_path)
            if member is None:
                raise KeyError(
                    f"member {member_path!r} of checkpoint {checkpoint_id!r} "
                    f"not in registry"
                )
            self._checkpoint_members[checkpoint_id][member_path] = replace(
                member,
                pin_state=pin_state,
                restore_tier=(
                    member.restore_tier if restore_tier is None else restore_tier
                ),
            )

    def adjust_checkpoint_pin_refcount(self, checkpoint_id: str, delta: int) -> int:
        """Atomically add ``delta`` to a checkpoint's pin refcount and return the
        new value (read + validate + write in one lock hold — two concurrent
        adjustments cannot lose an update). Raises ``KeyError`` for an unknown
        checkpoint and ``ValueError`` if the result would go negative (a release
        without a matching pin is a bookkeeping bug, fail-closed)."""
        with self._lock:
            record = self._checkpoints.get(checkpoint_id)
            if record is None:
                raise KeyError(f"checkpoint {checkpoint_id!r} not in registry")
            new_count = record.pin_refcount + delta
            if new_count < 0:
                raise ValueError(
                    f"pin refcount for checkpoint {checkpoint_id!r} would go "
                    f"negative ({record.pin_refcount} + {delta}); a release "
                    f"without a matching pin is a bookkeeping bug (fail-closed)"
                )
            self._checkpoints[checkpoint_id] = replace(
                record, pin_refcount=new_count
            )
            return new_count

    def _emit_state_log(
        self,
        *,
        artifact_id: UUID,
        agent_id: UUID,
        from_state: MESIState,
        to_state: MESIState,
        trigger: str,
        tick: int,
        version: int,
        content_hash: str | None,
    ) -> int:
        """Emit one ``state_log`` entry for the inlined CAS region. Returns 1 if
        ``_seq`` was bumped (0 if no state_log configured). On callback raise,
        decrements its own reservation and re-raises (mutation-then-log parity
        with :meth:`set_agent_state`)."""
        if self._state_log is None:
            return 0
        self._seq += 1
        entry = {
            "tick": tick,
            "artifact_id": str(artifact_id),
            "agent_id": str(agent_id),
            "agent_name": self._agent_names.get(agent_id) if self._agent_names is not None else None,
            "from_state": from_state.name,
            "to_state": to_state.name,
            "trigger": trigger,
            "version": version,
            "content_hash": content_hash,
            "sequence_number": self._seq,
            "instance_id": self._instance_id,
            "schema_version": CCS_STATE_LOG_SCHEMA_VERSION,
        }
        try:
            self._state_log(entry)
        except Exception:
            self._seq -= 1
            raise
        return 1

    def record_heartbeat(self, agent_id: UUID, now_tick: int) -> None:
        """Record an agent's heartbeat tick using max(prev, incoming) (R12 monotonicity)."""
        with self._lock:
            prev = self._heartbeat_by_agent.get(agent_id)
            if prev is None or now_tick > prev:
                self._heartbeat_by_agent[agent_id] = now_tick

    def last_heartbeat_tick(self, agent_id: UUID) -> int | None:
        """Return the last recorded heartbeat tick for an agent, if any."""
        with self._lock:
            return self._heartbeat_by_agent.get(agent_id)

    def record_last_reclamation(
        self, agent_id: UUID, artifact_id: UUID, trigger: str, tick: int
    ) -> None:
        """Record the most recent reclamation slot for an (agent, artifact) pair."""
        with self._lock:
            self._records[artifact_id].last_reclamation_by_agent[agent_id] = (trigger, tick)

    def get_last_reclamation(
        self, agent_id: UUID, artifact_id: UUID
    ) -> ReclamationSlot | None:
        """Return the most recent reclamation slot for an (agent, artifact) pair, if any."""
        with self._lock:
            record = self._records.get(artifact_id)
            if record is None:
                return None
            return record.last_reclamation_by_agent.get(agent_id)

    def invalid_reclamations(self) -> dict[UUID, dict[UUID, ReclamationSlot]]:
        """Reclamation slots of the pairs that are INVALID right now (#195).

        See the Protocol docstring; one lock hold, so the state and the slot
        of each pair are read together."""
        out: dict[UUID, dict[UUID, ReclamationSlot]] = {}
        with self._lock:
            for artifact_id, record in self._records.items():
                for agent_id, slot in record.last_reclamation_by_agent.items():
                    if record.state_by_agent.get(agent_id) == MESIState.INVALID:
                        out.setdefault(artifact_id, {})[agent_id] = slot
        return out

    def granted_at_tick(self, agent_id: UUID, artifact_id: UUID) -> int | None:
        """Return the tick at which agent acquired its current M/E grant on artifact, if any."""
        with self._lock:
            record = self._records.get(artifact_id)
            if record is None:
                return None
            return record.granted_at_tick_by_agent.get(agent_id)

    def get_agent_transient(self, artifact_id: UUID, agent_id: UUID) -> TransientState | None:
        """Return transient state for one agent/artifact pair if present."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.transient_by_agent.get(agent_id) if record else None

    def set_agent_transient(
        self,
        artifact_id: UUID,
        agent_id: UUID,
        transient_state: TransientState,
        *,
        entered_tick: int,
    ) -> None:
        """Set transient state and entry tick for one agent/artifact pair."""
        with self._lock:
            self._records[artifact_id].transient_by_agent[agent_id] = transient_state
            self._records[artifact_id].transient_tick_by_agent[agent_id] = entered_tick

    def clear_agent_transient(self, artifact_id: UUID, agent_id: UUID) -> None:
        """Clear transient state and timestamp for one agent/artifact pair."""
        with self._lock:
            self._records[artifact_id].transient_by_agent.pop(agent_id, None)
            self._records[artifact_id].transient_tick_by_agent.pop(agent_id, None)

    def get_transient_map(self, artifact_id: UUID) -> dict[UUID, TransientState]:
        """Return copy of per-agent transient states for an artifact."""
        with self._lock:
            record = self._records.get(artifact_id)
            return dict(record.transient_by_agent) if record else {}

    def get_transient_tick(self, artifact_id: UUID, agent_id: UUID) -> int | None:
        """Return tick when agent entered transient state if present."""
        with self._lock:
            record = self._records.get(artifact_id)
            return record.transient_tick_by_agent.get(agent_id) if record else None

    def remove_artifact(self, artifact_id: UUID) -> None:
        """Remove artifact record and all associated state from registry."""
        with self._lock:
            self._records.pop(artifact_id, None)

    def valid_holders(self, artifact_id: UUID) -> list[UUID]:
        """Return agents that currently hold non-invalid entries."""
        with self._lock:
            record = self._records.get(artifact_id)
            if record is None:
                return []
            return [
                agent_id
                for agent_id, state in record.state_by_agent.items()
                if state != MESIState.INVALID
            ]


if TYPE_CHECKING:
    # Static conformance assertion (Phase 1, zero runtime change): a type checker
    # rejects this if ``ArtifactRegistry`` ever drifts from the ``RegistryBase``
    # contract the service layer depends on. Structural (no runtime inheritance);
    # no import cycle (registry_protocol imports only domain types).
    from .registry_protocol import RegistryBase

    def _conforms(r: ArtifactRegistry) -> RegistryBase:
        return r
