# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The generalized backend atomic-boundary contract — importable vocabulary.

This module formalizes what a networked registry backend must provide to
re-home the coordinator's atomic write-arbitration boundary out of a single
process, so N stateless coordinators can front one managed HA backend. It is
**pure vocabulary**: enums, frozen dataclasses, and string constants with rich
docstrings — **zero networked code, zero I/O, zero behavior change** to anything
shipped. Nothing here connects to, reads from, or writes to any store.

It follows the :mod:`ccs.coordinator.registry_protocol` precedent (a module that
owns Protocols + shared types the two registries re-export). Where that module
names the registry SURFACE (the 62-member ``RegistryBase`` + ``SqliteExtended``
Protocols; the 9 detection members sit on their own ``ForeignWriteDetection``),
this module names the CONTRACT that surface must satisfy for a
backend to host the atomic boundary: which members participate in the
single-writer atomic step (:data:`MEMBER_CLASSIFICATION`), what that atomic step
IS (:data:`R9_ATOMIC_BOUNDARY`), the conformance tiers (:class:`Tier`), the
statelessness re-home inventory (:data:`STATELESSNESS_INVENTORY`), and the
liveness / epoch / content / credential / identity obligations (R18, R12, R13,
R14, R15).

**Non-goal (R2 trip-wire — never own the durable store).** The competitive
niche survives only if we never own the durable store or log. A conforming
backend is **operated infrastructure we depend on, NEVER a store we ship**. This
contract names the atomic primitive a backend must expose; it does not, and must
not, grow into a store the project builds and operates. That trip-wire binds the
contract itself, not just its consumers.

**Build gating.** This module is the BUILDABLE, gate-independent contract layer.
Writing it does NOT move Gate B. Any networked backend *implementation*, any
routed production deployment, and the HA topology itself remain Gate-B-gated.
R14/R15 appear here as documented OBLIGATION records with typed placeholders,
never implementations — their code is backend-connect logic (Gate-B).

Layer discipline: this lives in ``ccs.coordinator`` (the application layer). It
imports ONLY from ``ccs.core`` and the sibling ``registry_protocol`` /
``retention`` modules — never from ``bus`` / ``transport`` / ``simulation``
(infrastructure) or any interface namespace (``output`` / ``cli`` / ``adapters``
/ ``validation`` / ``diagnose`` / ``replay`` / ``mcp``). ``tools/check_architecture.py``
enforces this.

Requirement trace (see
``docs/brainstorms/2026-07-06-tls-posture-and-networked-backend-contract-requirements.md``):
R2 (non-goal, above), R8 (member mapping), R9 (atomic boundary), R10 (tiers),
R11 (statelessness inventory), R12 (epoch monotonicity), R13 (content posture),
R14/R15 (documented obligations), R18 (liveness source).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# R8 — member classification enum + map
# ---------------------------------------------------------------------------


class MemberClass(Enum):
    """How a ``RegistryBase`` / ``SqliteExtended`` member relates to the R9
    single-writer atomic boundary, when the boundary is re-homed into a networked
    backend (R8). Exactly three classes — a value outside this enum is
    unrepresentable (:data:`MEMBER_CLASSIFICATION` is typed by it), which is the
    drift guard against an ad-hoc string slipping into the map.
    """

    ATOMIC_CLASS = "atomic_class"
    """The member participates in the single-writer atomic boundary (R9): the
    version-CAS **+** grant arbitration (``other_holder`` over ``state_by_agent``)
    **+** read-generation fence, arbitrated in ONE backend atomic step, OR the
    same-lock liveness sweep that shares that serialization. A Tier-1 backend must
    host every ATOMIC_CLASS member such that they compose into one atomic RMW (the
    sweep may keep separate-sweep semantics under an equivalent serialization
    guarantee — see :data:`R9_ATOMIC_BOUNDARY`). Authored from the ``service.py``
    call sites, NOT assumed."""

    INDEPENDENT = "independent"
    """The member is a mutation, but NOT part of the R9 boundary: it may be
    individually atomic in the backend without composing into the single-writer
    RMW. Examples: the grant-holder heartbeat record (liveness input, read by the
    sweep but written on its own), a session's durable-cut release, artifact
    registration/removal, the preemption-notice surface. A backend must make each
    individually durable but need not fold them into the boundary."""

    READ_ONLY = "read_only"
    """The member only READS registry state (never mutates). It serves the atomic
    boundary's inputs and the coordinator's read paths but is not itself part of
    the atomic mutation. A backend serves these as consistent reads; they carry no
    single-writer obligation of their own."""


@dataclass(frozen=True)
class MemberContract:
    """One ``RegistryBase`` / ``SqliteExtended`` member's classification (R8).

    Frozen so the map is immutable vocabulary. ``rationale`` records WHY the
    member lands in ``member_class`` — authored from the ``service.py`` call site,
    so a reviewer can check the classification against the code rather than trust
    it. ``surface`` records whether the member is on the base contract (every
    backend) or the SQLite-extended contract (the durable-store-only surface the
    HTTP server depends on).
    """

    name: str
    member_class: MemberClass
    surface: str  # "base" | "sqlite_extended" | "detection"
    rationale: str


# The 71 members of RegistryBase (48 methods + 1 property), SqliteExtended
# (+13 methods) and ForeignWriteDetection (+9 methods), classified against the CoordinatorService call sites. The
# ATOMIC_CLASS members are the ones the service touches INSIDE its atomic
# mutation paths (``write`` / ``commit`` / ``commit_cas`` under ``abort_guard``;
# ``invalidate``; the same-lock ``enforce_stable_grant_timeouts`` sweep; the
# session capture). INDEPENDENT members are individually-durable mutations that
# do not compose into the R9 RMW. READ_ONLY members never mutate.
#
# The ``coordinator_epoch`` PROPERTY is included explicitly (property-omission
# teeth — ``@runtime_checkable`` cannot see properties, per the #133 lesson; a
# map that classified only methods would silently drop it, the exact gap that
# lesson exists to prevent).
_MEMBER_CONTRACTS: tuple[MemberContract, ...] = (
    # ---- ATOMIC_CLASS (base) ----------------------------------------------
    MemberContract(
        "commit_cas",
        MemberClass.ATOMIC_CLASS,
        "base",
        "THE atomic boundary. The version-CAS + grant arbitration (other_holder "
        "over state_by_agent) + read-generation fence all live INSIDE "
        "registry.commit_cas and return a CasResult in one serialized step "
        "(sqlite: BEGIN IMMEDIATE). service.commit_cas wraps it in abort_guard "
        "and maps the three outcomes; the arbitration itself is atomic here.",
    ),
    MemberContract(
        "commit_all",
        MemberClass.ATOMIC_CLASS,
        "base",
        "THE N-artifact atomic boundary (SB-18). commit_all runs CHECK-all "
        "(version-CAS + other_holder + fence per member) then applies EVERY member "
        "or NONE, in one serialized step (sqlite: one BEGIN IMMEDIATE; in-memory: "
        "one _lock hold with snapshot-restore). All-or-nothing -- a partial "
        "batch is never observable (NoPartialPublish). Returns a MultiCasResult; "
        "the aggregated invalidation broadcasts only after the commit. A genuinely "
        "new atomic op, never a loop of commit_cas.",
    ),
    MemberContract(
        "set_artifact_and_content",
        MemberClass.ATOMIC_CLASS,
        "base",
        "The pessimistic-commit version bump + content persist, carrying "
        "fence_agent_id so the read-generation fence check fires ATOMICALLY with "
        "the version move (service._commit_impl). A StaleReadGeneration raise "
        "here is the fence rejecting a reclaimed committer in the write's race "
        "window — part of the boundary, not a plain write.",
    ),
    MemberContract(
        "set_agent_state",
        MemberClass.ATOMIC_CLASS,
        "base",
        "The MESI grant transition — the grant-arbitration leg of the boundary. "
        "Called inside every atomic mutation path (write/commit peer-invalidation "
        "+ grant, invalidate, and the same-lock sweep's M/E->INVALID reclaim). "
        "An M/E->INVALID transition that revokes the write claim WITHOUT moving "
        "the version bumps owner_generation for the fence — the sweep reclaims "
        "AND the voluntary invalidate release (EPOCH_BUMP_TRIGGERS); the "
        "version-moving peer invalidations do not, because version-CAS already "
        "arbitrates those. Single-writer is checked after each such transition. "
        "The read_generation capture rides the same transition and is keyed on "
        "the agent's OWN claim: an I/S->M/E acquire, or a genuine read the agent "
        "requested (CLAIM_CAPTURE_TRIGGERS plus the not-previously-in-M/E "
        "guard); a transition that takes the agent OUT of M/E captures nothing, "
        "so a peer's fetch that downgrades a holder never refreshes its "
        "captured generation.",
    ),
    MemberContract(
        "set_agent_transient",
        MemberClass.ATOMIC_CLASS,
        "base",
        "Sets the ISG/IED/EIA/SIA/MWB/MSA transient marking an in-flight grant "
        "arbitration. Written under abort_guard inside write/commit (the "
        "acquire/downgrade legs) and gates the sweep (a pair mid-transient is "
        "skipped). Part of the grant-arbitration state the boundary serializes.",
    ),
    MemberContract(
        "clear_agent_transient",
        MemberClass.ATOMIC_CLASS,
        "base",
        "Clears the arbitration transient at the end of a grant transition, "
        "under abort_guard inside write/commit/invalidate. The completing half of "
        "the transient leg of the boundary.",
    ),
    MemberContract(
        "capture_version_vector",
        MemberClass.ATOMIC_CLASS,
        "base",
        "Atomically pins a consistent multi-artifact cut AND persists the durable "
        "session_meta owner-binding at ONE linearization point (begin_session). "
        "The snapshot-session surface is part of the boundary at Tier-1 (R9): the "
        "cut + durable owner are captured together or not at all.",
    ),
    MemberContract(
        "abort_guard",
        MemberClass.ATOMIC_CLASS,
        "base",
        "The context manager that wraps EVERY atomic mutation (write / commit / "
        "commit_cas / invalidate). It is the boundary's serialization + fail-shut "
        "seam: a watchdog-aborted mutation fails closed at the write lock instead "
        "of landing as a phantom write. A backend re-homing the boundary must "
        "provide the equivalent atomic-or-abort envelope.",
    ),
    MemberContract(
        "record_last_reclamation",
        MemberClass.INDEPENDENT,
        "base",
        "Records the (trigger, tick) reclamation slot during the same-lock "
        "enforce_stable_grant_timeouts sweep, in its OWN transaction — a serialized "
        "follow-on AFTER the M/E->INVALID reclaim, NOT one atomic RMW with it "
        "(verified: set_agent_state and record_last_reclamation each open their own "
        "BEGIN IMMEDIATE, called consecutively by the sweep). Read back by commit "
        "only to explain a reclaimed-grant failure — a diagnostic write a backend "
        "makes individually durable, not part of the single-writer boundary "
        "(cf. the heartbeat record, likewise INDEPENDENT).",
    ),
    # ---- ATOMIC_CLASS reads consumed INSIDE the boundary -------------------
    MemberContract(
        "get_state_map",
        MemberClass.ATOMIC_CLASS,
        "base",
        "Reads state_by_agent — the grant-arbitration operand. Read INSIDE the "
        "atomic mutation loop of write/commit (to invalidate peers) and by the "
        "same-lock sweep to enumerate M/E holders and re-check single-writer. Its "
        "consistency with the concurrent grant transitions is what the boundary "
        "serializes; a backend must serve it inside the same atomic step.",
    ),
    # ---- INDEPENDENT (base) -----------------------------------------------
    MemberContract(
        "register_artifact",
        MemberClass.INDEPENDENT,
        "base",
        "Seeds a new artifact + initial content (register_artifact). A durable "
        "insert, individually atomic, but not part of a single-writer RMW — an "
        "artifact has no competing writer at first observation.",
    ),
    MemberContract(
        "remove_artifact",
        MemberClass.INDEPENDENT,
        "base",
        "Deletes an artifact (delete). Individually durable; not part of the "
        "version-CAS/grant boundary (delete does not arbitrate a writer).",
    ),
    MemberContract(
        "record_heartbeat",
        MemberClass.INDEPENDENT,
        "base",
        "Records a grant-holder's liveness tick (max(prev, incoming)). A liveness "
        "INPUT the same-lock sweep reads, but written on its own outside any "
        "atomic mutation — individually durable, not part of the boundary. Its "
        "authoritative-source obligation is R18, not atomicity.",
    ),
    MemberContract(
        "release_session",
        MemberClass.INDEPENDENT,
        "base",
        "Drops a session's durable pins + session_meta owner-binding (reap / "
        "release / effect-gate teardown). Individually durable and idempotent; "
        "not part of the write-arbitration RMW.",
    ),
    MemberContract(
        "create_checkpoint",
        MemberClass.INDEPENDENT,
        "base",
        "Persists a workspace-checkpoint manifest (WV Unit 2): the header row + "
        "owner metadata + every member row in ONE registry transaction (sqlite: "
        "one BEGIN IMMEDIATE; in-memory: one _lock hold with validate-"
        "before-first-insert) — individually atomic, all rows or none. NOT part "
        "of the R9 single-writer RMW: the manifest records capture facts, it "
        "arbitrates no writer (the capture engine's version reads ride the "
        "session machinery; restore registration rides commit_all).",
    ),
    MemberContract(
        "set_checkpoint_restore_status",
        MemberClass.INDEPENDENT,
        "base",
        "Updates the checkpoint-level restore-progress status (crash-resumable "
        "restore's durable marker). Individually durable; not part of the "
        "write-arbitration RMW — the arbitration of restore WRITES happens in "
        "commit_all / the member legs, never in this bookkeeping update.",
    ),
    MemberContract(
        "set_checkpoint_member_restore",
        MemberClass.INDEPENDENT,
        "base",
        "Records one member's typed restore outcome + the manifest-side "
        "deleted_at_restore record (commit_all has no delete semantics, so "
        "delete legs are recorded here per the restore-registration design). "
        "Individually durable bookkeeping; arbitration lives in the member leg.",
    ),
    MemberContract(
        "set_checkpoint_member_pin",
        MemberClass.INDEPENDENT,
        "base",
        "Updates one member's pin state, optionally rewriting its restore tier "
        "in the same step (the Unit-6 loud restorable -> restorable-unpinned "
        "downgrade). Individually durable; not part of the R9 boundary.",
    ),
    MemberContract(
        "adjust_checkpoint_pin_refcount",
        MemberClass.INDEPENDENT,
        "base",
        "Atomic read-validate-write refcount adjustment (never below zero, "
        "fail-closed) for the Unit-6 GC pin legs. Individually atomic in the "
        "backend (one BEGIN IMMEDIATE / one lock hold) but not part of the "
        "single-writer RMW — it arbitrates pin bookkeeping, not artifact writes.",
    ),
    MemberContract(
        "bind_caller_principal",
        MemberClass.INDEPENDENT,
        "base",
        "First-claim-wins insert-if-absent of an identity's caller principal + "
        "mint nonce, returning the BOUND pair (service.claim_caller_principal). "
        "A backend must make the insert and the read-back one atomic step so two "
        "concurrent first claims bind ONE principal — but it is not part of the "
        "R9 RMW: it decides which caller may name an identity, never a write. "
        "Its own store, never the session-meta one, so the session sweep, cap "
        "and release cannot reach it.",
    ),
    # ---- READ_ONLY (base) --------------------------------------------------
    MemberContract(
        "coordinator_epoch",
        MemberClass.READ_ONLY,
        "base",
        "PROPERTY (not a method) — included explicitly per the #133 "
        "property-omission lesson. Read (never written by the service) on the "
        "read-fence and session paths to stamp results with the coordinator "
        "incarnation. READ_ONLY as a registry SURFACE; its re-home disposition is "
        "the strongest MUST_REHOME (R12) — the classification and the "
        "statelessness disposition are orthogonal axes.",
    ),
    MemberContract(
        "get_artifact",
        MemberClass.READ_ONLY,
        "base",
        "Reads artifact metadata + current version. The version operand every CAS "
        "reason is decided against, but the READ itself is non-mutating "
        "(_require_artifact, read_at_version, session paths, effect-gate "
        "re-validate).",
    ),
    MemberContract(
        "get_content",
        MemberClass.READ_ONLY,
        "base",
        "Reads current canonical body (fetch). Non-mutating serve.",
    ),
    MemberContract(
        "get_content_at_version",
        MemberClass.READ_ONLY,
        "base",
        "Reads a retained historical body by version. Non-mutating serve (only "
        "present under declared retention, R13).",
    ),
    MemberContract(
        "get_version_record",
        MemberClass.READ_ONLY,
        "base",
        "Reads a (body, captured_at) history row (read_at_version / session_read). "
        "Non-mutating; immutable once captured.",
    ),
    MemberContract(
        "get_agent_state",
        MemberClass.READ_ONLY,
        "base",
        "Reads one agent's MESI state as a PRECONDITION check before an atomic "
        "mutation (commit's M/E holder check; commit_cas's shared-or-invalid "
        "gate). The read is non-mutating; the atomicity lives in the subsequent "
        "commit_cas / set_artifact_and_content step, not here.",
    ),
    MemberContract(
        "get_agent_transient",
        MemberClass.READ_ONLY,
        "base",
        "Reads one agent's transient (commit_cas precondition; the sweep's "
        "skip-if-transient guard). Non-mutating read of arbitration state.",
    ),
    MemberContract(
        "get_transient_map",
        MemberClass.READ_ONLY,
        "base",
        "Reads all transients for an artifact (enforce_transient_timeouts scan). "
        "Non-mutating.",
    ),
    MemberContract(
        "get_transient_tick",
        MemberClass.READ_ONLY,
        "base",
        "Reads when a transient was entered (transient-timeout age check). "
        "Non-mutating.",
    ),
    MemberContract(
        "get_last_reclamation",
        MemberClass.READ_ONLY,
        "base",
        "Reads the (trigger, tick) reclamation slot to EXPLAIN a reclaimed-grant "
        "commit failure (commit). Non-mutating; the WRITE side "
        "(record_last_reclamation) is ATOMIC_CLASS.",
    ),
    MemberContract(
        "invalid_reclamations",
        MemberClass.READ_ONLY,
        "base",
        "Batch read of the (trigger, tick) slots of the pairs currently INVALID, "
        "for the /status operator tier (#195): tells an observer a sweep reclaim "
        "from a voluntary release. Non-mutating; outside the boundary.",
    ),
    MemberContract(
        "get_owner_generation",
        MemberClass.READ_ONLY,
        "base",
        "Reads the per-artifact fence counter. Non-mutating; the BUMP happens "
        "inside the atomic reclaim (set_agent_state with a RECLAIM_TRIGGER).",
    ),
    MemberContract(
        "get_read_generation",
        MemberClass.READ_ONLY,
        "base",
        "Reads an agent's captured read-generation (the fence operand). "
        "Non-mutating; the fence CHECK against owner_generation is atomic inside "
        "commit_cas / set_artifact_and_content.",
    ),
    MemberContract(
        "last_observed_version_for",
        MemberClass.READ_ONLY,
        "base",
        "Reads the artifact version whose bytes an agent last observed (the "
        "SB-10 post-compaction staleness comparand). Non-mutating; the RECORD "
        "side rides the atomic non-INVALID upserts (set_agent_state and the "
        "commit_cas / commit_all WIN paths), which are already classified.",
    ),
    MemberContract(
        "get_artifact_and_generation",
        MemberClass.READ_ONLY,
        "base",
        "Reads an artifact together with the per-artifact fence counter as ONE "
        "snapshot. Non-mutating, but carries a consistency obligation beyond "
        "plain READ_ONLY: the two values must have coexisted, since a backend "
        "serving them as two reads lets a sweep reclaim tear the pair and "
        "silently reopens the reclaim-zombie EFFECT hole in adapters.effect_gate.",
    ),
    MemberContract(
        "granted_at_tick",
        MemberClass.READ_ONLY,
        "base",
        "Reads when a grant was granted (the sweep's max-hold age check). "
        "Non-mutating liveness input.",
    ),
    MemberContract(
        "last_heartbeat_tick",
        MemberClass.READ_ONLY,
        "base",
        "Reads a grant-holder's last heartbeat (the sweep's staleness check). "
        "Non-mutating liveness input; its authoritative source is R18.",
    ),
    MemberContract(
        "has_artifact",
        MemberClass.READ_ONLY,
        "base",
        "Existence check (invalidate / delete short-circuit). Non-mutating.",
    ),
    MemberContract(
        "artifact_ids",
        MemberClass.READ_ONLY,
        "base",
        "Enumerates artifact ids (the sweep + transient-timeout scans). "
        "Non-mutating.",
    ),
    MemberContract(
        "valid_holders",
        MemberClass.READ_ONLY,
        "base",
        "Reads the non-INVALID holders of an artifact. Non-mutating serve.",
    ),
    MemberContract(
        "retention_meta",
        MemberClass.READ_ONLY,
        "base",
        "Reads (retain_versions, policy) — the deployment branch indicator (R13). "
        "Non-mutating.",
    ),
    MemberContract(
        "get_session_cut",
        MemberClass.READ_ONLY,
        "base",
        "Reads a session's pinned cut (session_read / session_commit / owner "
        "validation). Non-mutating; the cut was written atomically by "
        "capture_version_vector.",
    ),
    MemberContract(
        "get_checkpoint",
        MemberClass.READ_ONLY,
        "base",
        "Reads one workspace-checkpoint header (status / restore verbs). "
        "Non-mutating; the manifest was written atomically by create_checkpoint.",
    ),
    MemberContract(
        "get_checkpoint_members",
        MemberClass.READ_ONLY,
        "base",
        "Reads a manifest's member rows, ordered by member_path (restore / "
        "status enumeration). Non-mutating.",
    ),
    MemberContract(
        "list_checkpoints",
        MemberClass.READ_ONLY,
        "base",
        "Enumerates checkpoint headers (the CLI list verb). Non-mutating.",
    ),
    MemberContract(
        "get_session_meta",
        MemberClass.READ_ONLY,
        "base",
        "Reads one session's durable (owner, created_at_tick) — the durable "
        "fallback the service-level owner/lease maps fall back to across a "
        "restart. Non-mutating.",
    ),
    MemberContract(
        "all_session_meta",
        MemberClass.READ_ONLY,
        "base",
        "Reads ALL durable session metas (the liveness sweep enumerates the union "
        "of in-memory + durable sessions). Non-mutating.",
    ),
    MemberContract(
        "session_count",
        MemberClass.READ_ONLY,
        "base",
        "Counts live durable sessions for the begin_session max-sessions cap. "
        "Non-mutating; the cap decision is made under the SERVICE-level "
        "_session_lock, but this registry read is a plain count.",
    ),
    MemberContract(
        "get_caller_principal",
        MemberClass.READ_ONLY,
        "base",
        "Reads the caller principal bound to an identity — the durable tier "
        "service.validate_caller_principal falls back to when its in-process "
        "cache misses (after a restart). Non-mutating.",
    ),
    # ---- INDEPENDENT / READ_ONLY (sqlite_extended) ------------------------
    MemberContract(
        "resolve_or_register",
        MemberClass.INDEPENDENT,
        "sqlite_extended",
        "First-observation seeding of a durable artifact by path+hash. An "
        "individually-atomic upsert; not a single-writer RMW.",
    ),
    MemberContract(
        "record_preemption_notice",
        MemberClass.INDEPENDENT,
        "sqlite_extended",
        "Records an adapter-side preemption notice for a sweep-reclaimed victim. "
        "A durable side-record threaded through the sweep's on_reclaim callback; "
        "best-effort and adapter-only, NOT part of the boundary (a notice failure "
        "must not stop the sweep).",
    ),
    MemberContract(
        "pop_preemption_notice",
        MemberClass.INDEPENDENT,
        "sqlite_extended",
        "Consumes one preemption notice (adapter enrichment path). A durable "
        "read-and-delete; individually atomic, not part of the boundary.",
    ),
    MemberContract(
        "pop_pending_notices",
        MemberClass.INDEPENDENT,
        "sqlite_extended",
        "Reads an agent's pending notices and consumes the ones the caller can "
        "deliver (``consume_limit``). Durable read-and-delete; not part of the "
        "boundary.",
    ),
    MemberContract(
        "evict_stale_notices",
        MemberClass.INDEPENDENT,
        "sqlite_extended",
        "Prunes aged preemption notices. Durable maintenance delete; not part of "
        "the boundary.",
    ),
    MemberContract(
        "close",
        MemberClass.INDEPENDENT,
        "sqlite_extended",
        "Closes the durable connection. Lifecycle, not a registry mutation; a "
        "networked backend's equivalent is disconnect, outside the boundary.",
    ),
    MemberContract(
        "peek_preemption_notice",
        MemberClass.READ_ONLY,
        "sqlite_extended",
        "Reads (without consuming) a victim's preemption notice. Non-mutating.",
    ),
    MemberContract(
        "artifact_names_under_prefix",
        MemberClass.READ_ONLY,
        "sqlite_extended",
        "Durable prefix listing of artifact names. Non-mutating serve.",
    ),
    MemberContract(
        "artifacts_held_by_agent",
        MemberClass.READ_ONLY,
        "sqlite_extended",
        "Lists artifacts an agent holds in given MESI states. Non-mutating serve.",
    ),
    MemberContract(
        "get_artifact_updated_at",
        MemberClass.READ_ONLY,
        "sqlite_extended",
        "Reads an artifact's last-updated wall time. Non-mutating.",
    ),
    MemberContract(
        "last_writer_for",
        MemberClass.READ_ONLY,
        "sqlite_extended",
        "Reads the recorded last writer of an artifact. Non-mutating.",
    ),
    MemberContract(
        "lookup_artifact_id_by_name",
        MemberClass.READ_ONLY,
        "sqlite_extended",
        "Resolves a durable name to an artifact id. Non-mutating.",
    ),
    MemberContract(
        "status_snapshot",
        MemberClass.READ_ONLY,
        "sqlite_extended",
        "Batch read of the artifact + state maps for the /status surface, "
        "optionally scoping the state half to named agents. Non-mutating.",
    ),
    MemberContract(
        "record_foreign_write",
        MemberClass.INDEPENDENT,
        "detection",
        "Counts one newly observed on-disk content for an artifact, gated on the "
        "disk hash last counted for it. A mutation, but observability exhaust "
        "rather than coordination state: it decides nothing, and the boundary "
        "never reads it. A backend makes the read-compare-upsert individually "
        "atomic so two ticks cannot double-count one content; it must NOT fold "
        "it into the single-writer RMW, where it would put a detector's write "
        "on the arbitration path.",
    ),
    MemberContract(
        "record_detection_tick",
        MemberClass.INDEPENDENT,
        "detection",
        "Records that the detector observed one tick in this coordinator run. "
        "Individually durable liveness exhaust, outside the boundary — the same "
        "posture as the grant-holder heartbeat, and read only by the offline "
        "report that distinguishes a true zero from an instrument that never ran.",
    ),
    MemberContract(
        "foreign_write_totals",
        MemberClass.READ_ONLY,
        "detection",
        "Reads per-artifact detection counts. Non-mutating.",
    ),
    MemberContract(
        "detection_runs",
        MemberClass.READ_ONLY,
        "detection",
        "Reads the observed per-run detection intervals a coverage claim is "
        "checked against. Non-mutating.",
    ),
    MemberContract(
        "record_detection_uncoverable",
        MemberClass.INDEPENDENT,
        "detection",
        "Records that a coordinator run found no workspace it could ever poll — "
        "no git work tree — so the offline report can separate that from an "
        "instrument that broke and from one that never ran. Written once per "
        "became-uncoverable transition, outside the boundary; a backend need "
        "only make it individually durable. It must NOT be recorded as a "
        "detection tick: a tick asserts the detector looked, and a coverage "
        "claim is read off those intervals.",
    ),
    MemberContract(
        "detection_uncoverable",
        MemberClass.READ_ONLY,
        "detection",
        "Reads the runs that found no watchable workspace. Non-mutating.",
    ),
    MemberContract(
        "close_detection_run",
        MemberClass.INDEPENDENT,
        "detection",
        "Ends the current observed interval after a failed tick so a later "
        "success cannot extend it across the outage. Process-local bookkeeping "
        "outside the boundary; a backend need only ensure the next tick opens a "
        "distinct interval.",
    ),
    MemberContract(
        "clear_detection_edges",
        MemberClass.INDEPENDENT,
        "detection",
        "Re-arms detection for artifacts no longer diverging, clearing only the "
        "value that suppresses a repeat and never a count. Individually "
        "durable, outside the boundary.",
    ),
    MemberContract(
        "artifacts_with_detection_edge",
        MemberClass.READ_ONLY,
        "detection",
        "Reads which artifacts currently hold an edge-gate value. Non-mutating.",
    ),
)

MEMBER_CLASSIFICATION: dict[str, MemberContract] = {
    contract.name: contract for contract in _MEMBER_CONTRACTS
}
"""Every ``RegistryBase`` + ``SqliteExtended`` member → its :class:`MemberContract`
(R8). Keyed by member name. The key set must equal the 71-member Protocol surface
exactly — :mod:`tests.test_backend_contract` fails if ``registry_protocol.py``
gains or loses a member without a matching update here (bidirectional drift
guard). Includes the ``coordinator_epoch`` property (property-omission teeth)."""


# ---------------------------------------------------------------------------
# R9 — the atomic boundary definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AtomicBoundary:
    """The R9 single-writer atomic boundary — the extended P0 tuple.

    A frozen documentation record, not runtime behavior. It names, in prose a
    backend implementer targets, the ONE atomic operation class a Tier-1 backend
    must host and the reference semantics it must reproduce.
    """

    tuple_elements: tuple[str, ...]
    """The three legs that MUST be arbitrated together in one atomic step:
    version-CAS, grant arbitration, and the read-generation fence. Single-writer
    enforcement is NOT the version compare alone — it is all three composing."""

    fence_admit_on_absent: str
    """The fence's load-bearing admit-on-absent asymmetry, stated EXACTLY (it has
    drifted once before — the fence-parity lesson)."""

    reference_semantics: str
    """What "atomic" references: the registry's single-process serialization AS A
    WHOLE — atomic mutations PLUS the same-lock liveness sweep."""

    liveness_eviction_note: str
    """Precision on where liveness eviction sits relative to the atomic RMW."""


R9_ATOMIC_BOUNDARY = AtomicBoundary(
    tuple_elements=(
        "version-CAS (the artifact version compare-and-swap: a write wins only if "
        "the expected version still matches current)",
        "grant arbitration (other_holder over state_by_agent: no pessimistic peer "
        "may hold MODIFIED/EXCLUSIVE when an OCC writer commits; the MESI grant "
        "transition is part of the same step)",
        "read-generation fence (owner_generation vs the committer's captured "
        "read_generation: reject a writer whose M/E grant was revoked without "
        "the version moving — by the sweep OR by a voluntary invalidate release "
        "— since version-CAS alone cannot see that; the captured operand is "
        "written only at the committer's own claim-establishing transitions, "
        "never by a peer's fetch)",
    ),
    # Reproduced verbatim from the shipped guard semantics + the fence-parity
    # lesson (docs/solutions/.../fence-admit-on-absent-...). Do NOT paraphrase:
    # an ABSENT read_generation is ADMITTED, and version-CAS arbitrates; only a
    # PRESENT-and-superseded read_generation is rejected.
    fence_admit_on_absent=(
        "An ABSENT read_generation is ADMITTED — version-CAS arbitrates it (a "
        "plain OCC writer that never established a fence claim; NoLostUpdate, the "
        "sibling invariant, is its protection). A PRESENT read_generation equal to "
        "owner_generation WINS (the version bumps). ONLY a PRESENT read_generation "
        "that is superseded (< owner_generation) is REJECTED as "
        "stale_read_generation. A reclaim-zombie is NEVER absent — it captured its "
        "read_generation atomically at the I/S->M/E acquire, the sweep preserved "
        "that value while bumping owner_generation, and no later transition that "
        "leaves its authority unchanged or reduces it refreshes the value: a "
        "voluntary release preserves it, a peer's fetch that downgrades it to "
        "SHARED captures nothing, and a peer's fetch never rewrites an "
        "already-SHARED holder, which is why the superseded value SURVIVES until "
        "the zombie's OWN re-acquire (I/S->M/E) or its OWN genuine read "
        "re-captures it — so admit-on-absent "
        "removes no fence coverage. The `is not None` predicate is load-bearing, "
        "not defensive: it separates the plain OCC writer (admit) from the "
        "reclaim-zombie (reject)."
    ),
    reference_semantics=(
        "The registry's single-process serialization AS A WHOLE: the atomic "
        "mutations (version-CAS + grant arbitration + fence, under one sqlite "
        "BEGIN IMMEDIATE transaction / one process lock) TOGETHER WITH the "
        "same-lock enforce_stable_grant_timeouts sweep. A backend re-homing the "
        "boundary reproduces this whole serialization, not merely the version "
        "compare."
    ),
    liveness_eviction_note=(
        "Liveness EVICTION is a SEPARATE same-lock sweep "
        "(enforce_stable_grant_timeouts), NOT read inside commit_cas's "
        "transaction. It reclaims stale M/E grants (heartbeat / max-hold) and "
        "bumps owner_generation atomically with the M/E->INVALID transition — as "
        "does the voluntary invalidate release, so a release can never SUPPRESS "
        "the epoch move by getting there before the sweep — both "
        "serialized under the SAME registry lock as the atomic mutations. Per "
        "tier, a backend states whether liveness folds into the atomic RMW "
        "(Tier-1 may) or preserves the separate-sweep semantics WITH an equivalent "
        "serialization guarantee."
    ),
)


# ---------------------------------------------------------------------------
# R10 — conformance tiers
# ---------------------------------------------------------------------------


class Tier(Enum):
    """A backend's declared conformance tier (R10). Exactly two tiers.

    A backend declares a tier and the conformance kit verifies it. The kit builds
    Tier-1 verification NOW; Tier-2 is DEFINED here but its conformance machinery
    is a stated scope boundary — NOT built until a real Tier-2 candidate backend
    exists (speculative test infrastructure for an implementation class with zero
    candidates is not built).
    """

    TIER_1 = "tier_1_full_tuple"
    """FULL-TUPLE. The backend expresses the R9 boundary as ONE atomic RMW (a
    multi-key transaction or atomic script). Guarantee parity: full SingleWriter
    + NoStaleApply (+ session fail-closed lifecycle) with the shipped registry.
    The snapshot-session surface (pins + session_meta ownership + session leases)
    is REQUIRED at Tier-1 — it is shipped core; a backend without it cannot claim
    a stateless coordinator. QUALIFIES for the stateless-N-replica HA claim."""

    TIER_2 = "tier_2_lease_decomposed"
    """LEASE-DECOMPOSED. The backend provides per-artifact atomic version-CAS PLUS
    a lease/fencing-token primitive; the grant arbitration is re-derived rather
    than hosted as one RMW. Residual windows are documented per residual and
    test-demonstrated. NO NoStaleApply parity claim. Session support is an OPEN
    research question (the admit-on-absent fence + session-scoped uuid5 committer
    identity may impose ordering a lease primitive cannot express). Does NOT
    qualify for the HA claim. DEFINED here; its conformance-kit machinery is NOT
    built (scope boundary)."""


@dataclass(frozen=True)
class TierDeclaration:
    """A tier's obligations, parity, and HA qualification (R10). Frozen
    documentation vocabulary — the human-readable expansion of :class:`Tier`."""

    tier: Tier
    backend_must_provide: str
    guarantee_parity: str
    ha_qualifies: bool
    session_support: str


TIER_DECLARATIONS: dict[Tier, TierDeclaration] = {
    Tier.TIER_1: TierDeclaration(
        tier=Tier.TIER_1,
        backend_must_provide=(
            "The R9 boundary as ONE atomic RMW (multi-key transaction / atomic "
            "script): version-CAS + grant arbitration + read-generation fence "
            "arbitrated together, plus the same-lock liveness sweep semantics and "
            "the durable snapshot-session surface."
        ),
        guarantee_parity=(
            "Full SingleWriter + NoStaleApply parity with the shipped registry, "
            "plus the session fail-closed lifecycle."
        ),
        ha_qualifies=True,
        session_support=(
            "REQUIRED. Pins + session_meta ownership + session leases must re-home; "
            "a backend without them cannot claim a stateless coordinator."
        ),
    ),
    Tier.TIER_2: TierDeclaration(
        tier=Tier.TIER_2,
        backend_must_provide=(
            "Per-artifact atomic version-CAS + a lease/fencing-token primitive; "
            "grant arbitration re-derived from the lease rather than hosted as one "
            "RMW."
        ),
        guarantee_parity=(
            "Documented residual windows (named per residual, test-demonstrated). "
            "NO NoStaleApply parity claim."
        ),
        ha_qualifies=False,
        session_support=(
            "OPEN research question — the admit-on-absent fence and the "
            "session-scoped uuid5 committer identity may impose ordering a lease "
            "primitive cannot express; Tier-2 may exclude sessions or grow its "
            "residual list. Its conformance-kit machinery is NOT built here."
        ),
    ),
}


# ---------------------------------------------------------------------------
# R11 — statelessness inventory
# ---------------------------------------------------------------------------


class Disposition(Enum):
    """A coordinator state item's re-home disposition on failover (R11)."""

    MUST_REHOME = "must_rehome"
    """The item MUST live in the shared backend for the coordinator to be
    stateless. If it does not re-home, a failover loses correctness or liveness."""

    DERIVABLE = "derivable"
    """The item can be left in-process because it is re-derivable from re-homed
    backend state after a failover (rebuilt on demand, no durable copy needed)."""

    SAFELY_LOST = "safely_lost"
    """The item can be left in-process and simply dropped on failover with no
    correctness consequence (at worst a benign efficiency/attribution loss).
    ``coordinator_epoch`` is NEVER eligible for this disposition (R12)."""


@dataclass(frozen=True)
class StateItem:
    """One coordinator state item + its R11 re-home disposition. Frozen."""

    name: str
    disposition: Disposition
    location: str  # "durable_registry" | "service_level" | "service_with_durable_fallback"
    consequence: str


STATELESSNESS_INVENTORY: tuple[StateItem, ...] = (
    StateItem(
        "artifacts / versions / content_hashes",
        Disposition.MUST_REHOME,
        "durable_registry",
        "The version operand of every CAS. If not re-homed, a failover cannot "
        "arbitrate writes — every version-CAS is decided against wrong state.",
    ),
    StateItem(
        "MESI grants (state_by_agent)",
        Disposition.MUST_REHOME,
        "durable_registry",
        "The grant-arbitration operand (other_holder). If not re-homed, "
        "single-writer cannot be enforced across coordinators — two could both "
        "grant EXCLUSIVE.",
    ),
    StateItem(
        "fence generations (owner_generation / read_generation)",
        Disposition.MUST_REHOME,
        "durable_registry",
        "The read-generation fence operands. If not re-homed, a reclaim-zombie "
        "write cannot be rejected — NoStaleApply is lost.",
    ),
    StateItem(
        "grant heartbeats",
        Disposition.MUST_REHOME,
        "durable_registry",
        "The liveness input the sweep reads. If not re-homed, one coordinator's "
        "heartbeats are invisible to another's sweep — grants mis-reclaimed or "
        "never reclaimed. Its VALUES re-home, but an authoritative time SOURCE is "
        "a separate obligation (R18).",
    ),
    StateItem(
        "session pins (consistent cuts)",
        Disposition.MUST_REHOME,
        "durable_registry",
        "The pinned cut a session reads/commits against. If not re-homed, a "
        "failover silently invalidates every live session (fail-closed, but an "
        "availability cliff the HA claim must own).",
    ),
    StateItem(
        "session_meta (durable owner + created_at_tick)",
        Disposition.MUST_REHOME,
        "service_with_durable_fallback",
        "The durable session owner-binding (sqlite schema v4, the #132 "
        "make-identity-survive-restart precedent). The SERVICE-level owner/lease "
        "maps are the fast tier; this durable table is the fallback. If not "
        "re-homed, a survived session cannot be owner-validated across a failover "
        "without opening an owner-isolation hole (R13).",
    ),
    StateItem(
        "session leases (heartbeat lease + creation tick + reaped tombstone)",
        Disposition.MUST_REHOME,
        "service_with_durable_fallback",
        "The session-liveness lease and absolute-age ceiling inputs. Today these "
        "live in SERVICE-level maps (_session_heartbeats / _session_created / "
        "_reaped_tombstone) under service._session_lock, with the durable "
        "session_meta as the restart fallback (a post-restart heartbeat rehydrates "
        "the in-memory lease from the durable owner+created tick). For a stateless "
        "coordinator the lease + ceiling must re-home too, else a failover drops "
        "them and the session's GC-starvation bounds go unenforced across replicas.",
    ),
    StateItem(
        "retention rows (version bodies, when retention declared)",
        Disposition.MUST_REHOME,
        "durable_registry",
        "Present ONLY under a declared retain-versions capability (R13). When "
        "declared they must re-home for a failover to serve pinned/historical "
        "bytes; with retention OFF there are no bodies to re-home (hash-only "
        "baseline).",
    ),
    StateItem(
        "coordinator_epoch",
        Disposition.MUST_REHOME,
        "durable_registry",
        "The fence token identifying the coordinator incarnation. NEVER eligible "
        "for SAFELY_LOST (R12): losing it invalidates ALL client-carried tokens "
        "(session tokens, versioned reads) at once. Always MUST_REHOME. NOTE: the "
        "shipped LOCAL epoch is an opaque uuid4 string; the backend epoch must be "
        "a monotonic-increasing integer (R12) — that str->int migration is "
        "deferred to a re-home, out of this contract's scope.",
    ),
    StateItem(
        "caller-principal bindings (identity -> principal + mint nonce)",
        Disposition.MUST_REHOME,
        "service_with_durable_fallback",
        "The first-claim-wins binding a request's caller principal is checked "
        "against (sqlite schema v8, its own table). The SERVICE-level map is a "
        "positive-only cache in front of the durable table. If not re-homed, a "
        "failover forgets which identities are bound, so the next claimant of a "
        "bound identity would be issued a fresh principal and first-claim-wins "
        "would silently become last-failover-wins.",
    ),
    StateItem(
        "reaped-session tombstone (attribution only)",
        Disposition.SAFELY_LOST,
        "service_level",
        "Precise 'definitely reaped' attribution for a dead token. Its LOSS is "
        "benign: an in-shape token still classifies session_invalidated via the "
        "shape predicate, so the fail-closed guarantee never depends on it — only "
        "the sharper attribution is lost. (The lease + creation-tick portions of "
        "the session-lease item above are MUST_REHOME; this attribution-only "
        "residue is the safely-lost part.)",
    ),
)
"""Every coordinator state item + its R11 re-home disposition. ``coordinator_epoch``
is MUST_REHOME and asserted never SAFELY_LOST by
:mod:`tests.test_backend_contract`. Session owner/lease maps are SERVICE-level
(``service._session_lock``) with a durable ``session_meta`` fallback — noted in
each item's ``location`` / ``consequence``."""


# ---------------------------------------------------------------------------
# R18 — liveness / clock source declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LivenessSourceObligation:
    """R18 — a conforming backend (or the coordinator above it) MUST supply an
    authoritative time/lease source. Frozen documentation record."""

    requirement: str
    conforming_sources: tuple[str, ...]
    non_conforming: str
    shipped_source: str


R18_LIVENESS_SOURCE = LivenessSourceObligation(
    requirement=(
        "Heartbeat and lease timing must ride an AUTHORITATIVE time/lease source. "
        "N stateless coordinators have no shared tick base, so this is a "
        "backend-selection criterion alongside the atomic CAS."
    ),
    conforming_sources=(
        "a backend-authoritative monotonic clock or sequence",
        "a backend-native lease/TTL primitive (exactly what a Tier-2 lease "
        "primitive provides; Tier-1 must state its source explicitly)",
    ),
    non_conforming=(
        "Storing tick VALUES without an authoritative source does NOT conform — "
        "that is the silent mis-evict / never-reclaim failure this requirement "
        "names (the plan's second P0)."
    ),
    shipped_source=(
        "The shipped registries' source is CALLER-SUPPLIED LOGICAL TICKS under a "
        "SINGLE coordinator. Fine single-host (one process is the authority); it "
        "does NOT survive N coordinators — there is no shared tick base, so it "
        "does not conform for the HA claim until a re-home supplies an "
        "authoritative source."
    ),
)


# ---------------------------------------------------------------------------
# R12 — epoch monotonicity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochObligation:
    """R12 — the backend coordinator_epoch contract. Frozen documentation."""

    backend_contract: str
    shipped_local: str
    migration_scope: str


R12_EPOCH = EpochObligation(
    backend_contract=(
        "When promoted to the shared backend, coordinator_epoch is a "
        "MONOTONICALLY-INCREASING INTEGER across restarts and failovers — NEVER a "
        "fresh random id (that would invalidate all client-carried session tokens "
        "and versioned reads at once). A deposed coordinator is provably stale by "
        "comparison."
    ),
    shipped_local=(
        "The shipped LOCAL epoch is an opaque uuid4().hex string, seeded in "
        "registry_meta. It stays as-is (an incarnation label, not a monotonic "
        "counter) until a re-home."
    ),
    migration_scope=(
        "The str->monotonic-int migration (and its client-carried-token "
        "compatibility across the cutover) is a Gate-B re-home concern, OUT of "
        "this contract's scope / deferred."
    ),
)


# ---------------------------------------------------------------------------
# R13 — content posture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentPosture:
    """R13 — what a backend holds. Frozen documentation record."""

    baseline: str
    retention_capability: str
    disclosure_note: str


R13_CONTENT_POSTURE = ContentPosture(
    baseline=(
        "HASH-ONLY. The backend holds coherence metadata only: content_hash + "
        "version + the R9/R11 state. It NEVER holds artifact bodies at baseline."
    ),
    retention_capability=(
        "RETENTION is an explicit, DECLARED opt-in capability. A backend that "
        "declares it stores version bodies with the SAME gating as "
        "SqliteArtifactRegistry(retain_versions=True) — mirroring shipped "
        "semantics (KTD-13 held), parity with the reference backend."
    ),
    disclosure_note=(
        "With retention ON, artifact BODIES cross the network and rest in the "
        "backend — the disclosure / tenancy surface WIDENS and the deployment's "
        "security posture MUST say so."
    ),
)


# ---------------------------------------------------------------------------
# R14 / R15 — documented obligations only (NO implementation — Gate-B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackendConnectObligation:
    """A backend-connect security obligation stated as documentation ONLY (R14 /
    R15). Its CODE is Gate-B backend-connect logic (credential read, fingerprint
    verification, rotation) — NOT implemented here. This is a typed placeholder
    that names the obligation so a conformance discussion has vocabulary, without
    any networked code."""

    requirement_id: str  # "R14" | "R15"
    obligation: str
    discipline: str
    open_at_planning: str


R14_CREDENTIAL = BackendConnectObligation(
    requirement_id="R14",
    obligation=(
        "The backend credential (DSN / password) arrives via a 0600 credential "
        "FILE — never inline env or config literal."
    ),
    discipline=(
        "Mirror the CCS_REMOTE_SECRET_FILE reader's O_NOFOLLOW open + mode-check + "
        "single-read discipline (a trust anchor is the same attack class as the "
        "bearer file: a symlink swap between check and read must not be possible)."
    ),
    open_at_planning=(
        "The rotation path (file-replace + re-read-on-reconnect vs "
        "restart-required, and the dual-credential overlap window if live) MUST be "
        "specified AT PLANNING — TBD, not decided here."
    ),
)


R15_BACKEND_IDENTITY = BackendConnectObligation(
    requirement_id="R15",
    obligation=(
        "The coordinator validates the backend target (allowlist) and verifies a "
        "schema/contract FINGERPRINT on FIRST connect AND re-verifies on ANY "
        "reconnect or failover. Mismatch is a FAIL-CLOSED refusal."
    ),
    discipline=(
        "A promoted replica or a re-pointed DSN with a foreign fingerprint is the "
        "SAME attacker-controlled-backend surface (SSRF) as a bad first connect — "
        "so re-verify every reconnect/failover, never just the first."
    ),
    open_at_planning=(
        "The fingerprint content (schema version + epoch + contract revision?) and "
        "where the allowlist lives are open at planning."
    ),
)
