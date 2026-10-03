# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Domain exception hierarchy for coherence protocol operations."""

from __future__ import annotations

# Wire-stable reason string for the retry-eligible OCC precondition where a peer
# invalidated the caller in the window BETWEEN its fresh read and its CAS. Defined
# ONCE here so the coordinator server's response mapping
# (``_handle_post_edit_cas``) and the CoherentVolume client matcher
# (``_classify_cas_response``) reference the SAME literal and cannot drift — a
# reword on one side would otherwise silently break the other's retry routing.
OCC_CALLER_TRANSIENT_REASON = "caller_in_transient_state"

# Read-generation fence (Piece #2): the reason a commit is rejected because the
# committer's read_generation is older than the artifact's current
# owner_generation -- its captured claim was superseded by a sweep reclamation.
# Retry-eligible (reacquire + fresh read + re-commit). Shared by the
# ConflictDetail reason (OCC path) and the StaleReadGeneration exception
# (pessimistic path) so the two surfaces cannot drift.
STALE_READ_GENERATION_REASON = "stale_read_generation"

# Effect-gate HOLD classes. A HOLD says "the effect did not fire"; the cause
# says WHY. Kept as typed constants (never substring-matched on the message) so
# an agent or operator can branch.
#
# Honest limits of the discriminator: VERSION_MOVED, GRANT_RECLAIMED,
# GRANT_PREEMPTED, INPUT_VANISHED, READ_DENIED and CONTENT_CLAIM_ABSENT are
# recoverable — reacquire, re-decide, re-gate. GENERATION_UNCONFIRMED is the
# residual bucket and is NOT a clean
# permanent/transient signal: a degraded read, an unconfirmable foreign edit,
# and a coordinator that predates generation reporting all land there. Treat it
# as "reacquire and re-gate first"; a HOLD that survives a SUCCESSFUL reacquire
# is the one that needs an operator (check the daemon's version). Splitting
# READ_DENIED out is what keeps the common strict-mode reclaim — which reaches
# the client as a deny, not as a missing field — from hiding in that bucket, and
# splitting CONTENT_CLAIM_ABSENT out keeps a coordinator that simply holds no
# content claim — whose fix is a re-read, not an operator — from hiding there too.
HOLD_VERSION_MOVED = "version_moved"
HOLD_GRANT_RECLAIMED = "grant_reclaimed"
# The grant the decision was read under did not STAND at the re-validate read
# while BOTH comparands stayed put — a peer's write-claim acquire preempts a
# holder without moving the version (no commit yet) or the ownership epoch
# (trigger="write" is outside EPOCH_BUMP_TRIGGERS; see registry_protocol).
HOLD_GRANT_PREEMPTED = "grant_preempted"
HOLD_INPUT_VANISHED = "input_vanished"
HOLD_VERSION_UNCONFIRMED = "version_unconfirmed"
HOLD_READ_DENIED = "read_denied"
# The coordinator records NO content claim for this artifact at all — the
# recorded hash is absent, empty, or the all-``f`` launch-gate sentinel — so it
# cannot vouch that the bytes the decision was derived from ARE the content at
# the version it reports. Split OUT of the residual bucket because the caller's
# recovery differs: re-read your bytes (the coordinator will record a claim on
# the next observation), NOT "check the daemon's version and call an operator".
# Before this reason existed the empty-recorded-hash form did not HOLD at all:
# ``hash_differs`` needs a TRUTHY recorded hash on both sides, so the generation
# was never demoted and the effect fired against a value nothing backs.
HOLD_CONTENT_CLAIM_ABSENT = "content_claim_absent"
HOLD_GENERATION_UNCONFIRMED = "generation_unconfirmed"

# The closed, published set every consumer matches against
# (``hold_cause in HOLD_REASONS``). Same discipline as
# :data:`READ_AT_VERSION_REASONS` below: the EXACT string values are the WIRE
# CONTRACT, so a reason may be ADDED but is NEVER renamed or repurposed — a
# rename silently un-matches every consumer's branch and downgrades a HOLD it
# no longer recognises. Membership is the only legal test; never a substring of
# the human message (the typed-signal-not-substring house rule).
HOLD_REASONS: frozenset[str] = frozenset(
    {
        HOLD_VERSION_MOVED,
        HOLD_GRANT_RECLAIMED,
        HOLD_GRANT_PREEMPTED,
        HOLD_INPUT_VANISHED,
        HOLD_VERSION_UNCONFIRMED,
        HOLD_READ_DENIED,
        HOLD_CONTENT_CLAIM_ABSENT,
        HOLD_GENERATION_UNCONFIRMED,
    }
)

# ---------------------------------------------------------------------------
# read-at-version rejection vocabulary (plan item N v1, Unit 4 / R5)
# ---------------------------------------------------------------------------
#
# The EXACT string values below are the WIRE CONTRACT: ``read_at_version``
# returns a :class:`~ccs.core.types.VersionedReadRejection` whose ``reason`` is
# ONE of these, and every consumer (the Unit 6 replay resolver, SB-17 later)
# matches with ``reason == CONSTANT`` against :data:`READ_AT_VERSION_REASONS` —
# NEVER a substring of any human message (the typed-signal-not-substring house
# rule, ``docs/solutions/best-practices/typed-signal-not-substring-...``).
# Renaming a value here is a wire break; add, do not mutate.
#
# Six reasons, not seven: ``never_retained`` and ``beyond_horizon`` deliberately
# COLLAPSE into a single ``not_retained``. Once a retained history carries gaps
# (a T-expiry cleanup, or an off->on retention toggle), "never captured" and
# "captured then collected/expired" are UNDECIDABLE from persisted state, so a
# wire-stable constant must not encode a forensic distinction the store cannot
# honestly make (plan Key Decisions: "Rejection vocabulary (6 reasons ...)").
RETENTION_OFF_REASON = "retention_off"
"""Retention was never enabled for this store (``retention_meta()[0]`` False).

Distinct from ``not_retained``: the artifact_versions surface exists on every
v2 sqlite db, so table-presence cannot tell retention-on-unbounded from
retention-never-enabled — the persisted ``retention_enabled`` marker does."""

UNKNOWN_ARTIFACT_REASON = "unknown_artifact"
"""The artifact id is unknown to the registry (``get_artifact`` is None).

Deleted == never-existed: a post-delete read also lands here (delete cascades
the history rows), per the plan's deliberate collapse."""

NOT_RETAINED_REASON = "not_retained"
"""``1 <= version < current`` but no servable retained row exists — it was
never captured (e.g. ``commit_cas(content=None)``), K-evicted, or T-expired.
The single, deliberately-merged reason (see the module note above)."""

EPOCH_MISMATCH_REASON = "epoch_mismatch"
"""An ``expected_epoch`` was supplied and != the registry's
``coordinator_epoch`` — the store was reset (delete-and-recreate) since the
caller captured the epoch, so its retained history is from a different epoch."""

CURRENT_VERSION_REASON = "current_version"
"""``version == current``. The history surface serves HISTORY ONLY; current
content is read via the protocol fetch path (``artifacts`` stores hashes, not
bodies). A read-only consumer cannot obtain current bytes through any surface —
by design (plan Key Decisions: consequence of ``current_version``)."""

FUTURE_VERSION_REASON = "future_version"
"""``version > current``. Preserves the diagnostic ``commit_cas`` keeps via
:class:`~ccs.core.types.CasCorruption`: a requested version ABOVE the current
one suggests a second coordinator writing the same store."""

# The closed set every consumer matches against (``reason in
# READ_AT_VERSION_REASONS``). ``version < 1`` is NOT here — it is a ``ValueError``
# (caller misuse, house style), never a rejection reason.
READ_AT_VERSION_REASONS: frozenset[str] = frozenset(
    {
        RETENTION_OFF_REASON,
        UNKNOWN_ARTIFACT_REASON,
        NOT_RETAINED_REASON,
        EPOCH_MISMATCH_REASON,
        CURRENT_VERSION_REASON,
        FUTURE_VERSION_REASON,
    }
)

# ---------------------------------------------------------------------------
# session.read rejection reasons (SB-17 / TX-1, Unit 3 / R2)
# ---------------------------------------------------------------------------
#
# Wire-stable, ADDITIVE constants carried by
# :class:`~ccs.core.types.SessionReadRejection.reason`. ADDITIVE-only (R7): a NEW
# closed set, never folded into ``READ_AT_VERSION_REASONS`` — ``session_read`` is
# a distinct surface from the bare ``read_at_version`` history read. Consumers
# match ``reason == CONSTANT`` against :data:`SESSION_READ_REASONS`, never on a
# human message (the typed-signal-not-substring house rule). Unit 5 ADDED the
# heartbeat-liveness ``session_invalidated`` reason (a reaped / restart-wiped
# token lands there; a never-opened/malformed token stays ``session_not_found``).
SESSION_NOT_FOUND_REASON = "session_not_found"
"""The ``session_token`` has no pinned cut — unknown, never opened, or released
(``release_session``). A coordinator restart that wiped an in-memory session also
lands here in Unit 3 (the durable Unit-5 liveness/restart taxonomy is later)."""

SESSION_ARTIFACT_NOT_IN_CUT_REASON = "artifact_not_in_cut"
"""The token is a live session but the artifact was NOT in its captured read-set.
Reading an un-pinned artifact mid-session is out of scope and is REJECTED here,
never served from live HEAD (the no-fall-through guarantee)."""

# ---------------------------------------------------------------------------
# session liveness / fail-closed reason (SB-17 / TX-1, Unit 5 / R4)
# ---------------------------------------------------------------------------
#
# ADDITIVE (R7): the heartbeat-liveness fail-closed reason. A session whose pins
# are UNAVAILABLE — reaped by the session-liveness sweep (stale heartbeat),
# GC-raced, or wiped by an in-memory coordinator restart — fails CLOSED with
# this reason, NEVER a live-HEAD fall-through. It is deliberately DISTINCT from
# ``session_not_found``:
#
#   * ``session_invalidated`` — "your session DIED; re-establish it." The token
#     WAS (or structurally still looks like) a real server-minted session, but
#     its cut is gone. Returned for a token that is (a) in the bounded reaped-
#     tombstone, or (b) shaped like a server-minted token (the
#     ``looks_like_session_token`` format predicate) yet has no live cut — the
#     post-restart-unknown case MUST land here so a previously-valid token is
#     never served live HEAD as if still pinned.
#   * ``session_not_found`` — a genuinely-never-opened / malformed token (does
#     not match the server-minted format and is not in the tombstone). Kept
#     reachable additively so a clearly-bogus token is still distinguishable.
#
# Both are fail-closed (typed rejection, never live HEAD); the split only sharpens
# the operator/agent signal ("re-establish" vs "this was never a session"). Wire-
# stable: ADD, never rename ``session_not_found``; consumers match
# ``reason == CONSTANT``, never a substring.
SESSION_INVALIDATED_REASON = "session_invalidated"
"""A live session's pins are unavailable — reaped (stale heartbeat), GC-raced, or
wiped by an in-memory restart. Fails CLOSED (never live HEAD). Distinct from
``session_not_found`` (a never-opened/malformed token): a token that was, or
still structurally looks like, a real server-minted session lands HERE so it is
never mistaken for a fresh empty session and served live HEAD."""

# The closed set every ``session_read`` consumer matches against. ADDITIVE-only:
# disjoint from ``READ_AT_VERSION_REASONS`` (a separate surface, R7).
SESSION_READ_REASONS: frozenset[str] = frozenset(
    {
        SESSION_NOT_FOUND_REASON,
        SESSION_ARTIFACT_NOT_IN_CUT_REASON,
        SESSION_INVALIDATED_REASON,
    }
)

# ---------------------------------------------------------------------------
# session.commit rejection reasons (SB-17 / TX-1, Unit 4 / R3)
# ---------------------------------------------------------------------------
#
# Wire-stable, ADDITIVE constants carried by
# :class:`~ccs.core.types.SessionCommitRejection.reason` — the typed VALIDATION
# rejection of :meth:`CoordinatorService.session_commit` (the token has no pin /
# the artifact is not in the cut), NEVER a silent fall-through. The OCC OUTCOMES
# are NOT here: a lost-race surfaces as the shipped :class:`ConflictDetail`
# (returned unchanged) and corruption as a raised ``CoherenceError`` (the
# ``commit_cas`` taxonomy, preserved byte-for-byte). This set covers ONLY the
# pre-commit validation gate. The two reasons are SHARED with ``session_read``
# (the same token/pin checks), so they REUSE the Unit-3 literals rather than
# minting parallel ones — one token-validation vocabulary across the session
# surface. ADDITIVE-only (R7): a NEW closed set, disjoint from
# ``READ_AT_VERSION_REASONS``; consumers match ``reason == CONSTANT``, never a
# substring. Unit 5 ADDED ``session_invalidated`` (heartbeat-liveness): a reaped /
# restart-wiped token lands there; a never-opened/malformed one stays
# ``session_not_found``.
SESSION_COMMIT_REASONS: frozenset[str] = frozenset(
    {
        SESSION_NOT_FOUND_REASON,
        SESSION_ARTIFACT_NOT_IN_CUT_REASON,
        SESSION_INVALIDATED_REASON,
    }
)

# ---------------------------------------------------------------------------
# begin_session resource-cap rejection reasons (SB-17 / TX-1, Unit 7 / R14)
# ---------------------------------------------------------------------------
#
# ADDITIVE (R7): the resource-bound rejections of
# :meth:`CoordinatorService.begin_session`. ``begin_session`` already returns a
# typed :class:`~ccs.core.types.VersionedReadRejection` for an unknown id
# (``unknown_artifact``); these two reasons EXTEND that typed-return surface for
# the R14 caps that bound the snapshot blast radius (security-calibrated DEFAULTS,
# not post-hoc tuning):
#
#   * ``session_cap_exceeded`` — opening this session would exceed
#     ``max_sessions`` concurrent sessions (the many-sessions GC-starvation DoS
#     bound, threat-model #2). No token minted, no cut pinned.
#   * ``read_set_too_large`` — the ``read_set`` cardinality exceeds
#     ``max_read_set_cardinality`` (the enormous-single-read-set GC-starvation
#     bound, threat-model #2). No token minted, no cut pinned.
#
# Both are PRE-CAPTURE rejections (fail BEFORE any token mint / pin insert), so a
# rejected ``begin_session`` leaves NO half-open session. They reuse the
# ``VersionedReadRejection`` carrier (the same typed return ``begin_session``
# already produces) rather than minting a parallel rejection type. Wire-stable:
# ADD, never rename; consumers match ``reason == CONSTANT``, never a substring.
SESSION_CAP_EXCEEDED_REASON = "session_cap_exceeded"
"""``begin_session`` would exceed ``max_sessions`` concurrent sessions (R14). No
token minted, no cut pinned — the caller retries after a session is released or
reaped. Bounds the many-sessions GC-starvation DoS (threat-model #2)."""

SESSION_READ_SET_TOO_LARGE_REASON = "read_set_too_large"
"""``begin_session``'s ``read_set`` exceeds ``max_read_set_cardinality`` (R14). No
token minted, no cut pinned. Bounds the enormous-single-read-set GC-starvation
DoS (threat-model #2)."""

# The closed set every ``begin_session`` cap-rejection consumer matches against.
# ADDITIVE-only (R7): disjoint from ``READ_AT_VERSION_REASONS`` /
# ``SESSION_READ_REASONS`` / ``SESSION_COMMIT_REASONS``. ``begin_session`` may
# ALSO return ``unknown_artifact`` (the Unit-2 capture rejection), which stays in
# ``READ_AT_VERSION_REASONS``; these are the NET-NEW cap reasons only.
SESSION_BEGIN_CAP_REASONS: frozenset[str] = frozenset(
    {
        SESSION_CAP_EXCEEDED_REASON,
        SESSION_READ_SET_TOO_LARGE_REASON,
    }
)

# ---------------------------------------------------------------------------
# read-only store-open classification signals (Unit 6 hardening)
# ---------------------------------------------------------------------------
#
# Machine-readable signals carried by ``StoreNeedsRecoveryError.reason``
# (``ccs.coordinator.sqlite_registry``). INTERNAL routing values, not the wire
# contract: the CLI's JSON ``reason`` slugs (``needs_recovery`` / ``db_busy`` /
# ``db_corrupt``) stay owned by the resolver error classes. sqlite renders its
# operational errors as prose, so SOME substring matching is unavoidable — it
# happens in exactly ONE place (``classify_sqlite_operational_signal``), and
# every consumer branches on ``exc.reason == CONSTANT`` from here, never on a
# substring of the human message (the typed-signal-not-substring house rule,
# ``docs/solutions/best-practices/typed-signal-not-substring-...``).
STORE_SIGNAL_WAL_RECOVERY = "wal_recovery"
"""A hot WAL a read-only connection cannot replay (SQLITE_READONLY_RECOVERY).
Remedy: re-open once with the embedder (read-write) to checkpoint the WAL."""

STORE_SIGNAL_BUSY = "busy"
"""The store is locked by a concurrent writer (SQLITE_BUSY); retry shortly."""

STORE_SIGNAL_UNREADABLE = "unreadable"
"""The catch-all: the read-only connection could not read the store for a
reason that is neither a recognized recovery state nor a lock. The operator
remedy matches ``wal_recovery`` (re-open with the embedder), so consumers may
fold this into their needs-recovery surface."""

STORE_OPEN_SIGNALS: frozenset[str] = frozenset(
    {STORE_SIGNAL_WAL_RECOVERY, STORE_SIGNAL_BUSY, STORE_SIGNAL_UNREADABLE}
)

# ---------------------------------------------------------------------------
# cross-runtime store-open guard (sibling Node coordinator hazard)
# ---------------------------------------------------------------------------
#
# The sibling Node coordinator (the agent-coherence-plugin repo) shares the
# SAME ``<workspace>/.coherence/state.db`` path but maintains its OWN migration
# ledger: its v2 adds no schema objects (pending_notices validation) and its v3
# is ``ALTER TABLE agent_states ADD COLUMN deadline_tick`` — while THIS repo's
# v2 adds ``artifact_versions`` and its v3 (SB-17 / TX-1, Unit 2) adds
# ``session_pins``. So the two ledgers assign DIFFERENT meanings to
# ``PRAGMA user_version`` 2 AND 3 on the same file: a Node coordinator opening a
# Python-v3 db (or vice-versa) must DETECT and REJECT rather than silently
# misread the schema. The detection lives in
# ``SqliteArtifactRegistry._reject_foreign_ledger_db`` (the ``schema_runtime``
# lineage stamp + structural ``artifact_versions``/``session_pins`` /
# ``deadline_tick`` probes); the Node side mirrors it.
# ``CrossRuntimeSchemaError`` (defined in
# ``ccs.coordinator.sqlite_registry`` because it subclasses the
# coordinator-layer ``SchemaVersionError`` to keep existing catch-sites
# compatible; core must not import upward) carries THIS wire-stable reason so
# every consumer — and the Node side's mirror check — matches
# ``exc.reason == CONSTANT``, never a substring of the human message (the
# typed-signal-not-substring house rule,
# ``docs/solutions/best-practices/typed-signal-not-substring-...``).
# Renaming the value is a wire break; add, do not mutate.
CROSS_RUNTIME_SCHEMA_REASON = "cross_runtime_schema"

# ---------------------------------------------------------------------------
# caller-principal refusal reasons (coordinator caller principal, U4)
# ---------------------------------------------------------------------------
#
# The coordinator authenticates the WORKSPACE (one bearer secret), not the
# caller: a request's acting identity is a caller-asserted ``session_id``. A
# caller principal is a coordinator-issued value bound to one acting identity on
# the first claim of that identity, so a request naming the identity can be
# checked against it. The point is accident-resistance and attributability under
# the same-OS-user cooperative-trust model — bearer possession stays full
# authority by design — never a boundary against a process that can read
# ``.coherence/``.
#
# Wire-stable and ADDITIVE, matched by ``reason == CONSTANT`` (the
# typed-signal-not-substring house rule). Deliberately DISJOINT from
# :data:`HOLD_REASONS`: a principal refusal is a client error, never a hold — a
# hold invites a retry, and no retry supplies a principal the caller never had.
CALLER_PRINCIPAL_ABSENT_REASON = "caller_principal_absent"
"""The request names an identity but presents no caller principal. On the wire
a route sends it only for an identity that is BOUND: a request presenting none
for an identity nobody has claimed is a client predating the principal, and is
admitted (KTD15)."""

CALLER_PRINCIPAL_FOREIGN_REASON = "caller_principal_foreign"
"""The request presents a caller principal that is not the one bound to the
identity it names — minted for another identity, never minted, or presented
for an identity nobody has claimed."""

CALLER_PRINCIPAL_CLAIMED_REASON = "caller_principal_claimed"
"""A mint claim named an identity that is already bound, and did not present
the mint nonce the binding was made with. The first claim of an identity wins;
only a retry of THAT claim (same nonce) re-obtains its principal."""

CALLER_PRINCIPAL_REASONS: frozenset[str] = frozenset(
    {
        CALLER_PRINCIPAL_ABSENT_REASON,
        CALLER_PRINCIPAL_FOREIGN_REASON,
        CALLER_PRINCIPAL_CLAIMED_REASON,
    }
)

CALLER_PRINCIPAL_REFUSAL_REASONS: frozenset[str] = frozenset(
    {CALLER_PRINCIPAL_ABSENT_REASON, CALLER_PRINCIPAL_FOREIGN_REASON}
)
"""The ``reason`` a route's principal refusal carries: HTTP 400 with the body
``{"error": <prose>, "reason": <one of these>}``. A client classifies the
refusal by membership here, never by a substring of ``error``. A refused request
changed nothing — the gate runs before any mutation — so a client may retry it
once it holds the right principal. ``caller_principal_claimed`` is not one of
these: it answers a mint claim, in an HTTP 200 ``{ok: false}`` body."""

# ---------------------------------------------------------------------------
# MCP-C deny vocabulary (stale-write-guard-fs, 2026-06-18 plan, Unit 1)
# ---------------------------------------------------------------------------
#
# Each fail-closed coherence terminal carries a typed ``.reason`` CONSTANT on its
# class (below). The MCP deny mapper (``ccs.mcp.deny``) classifies by exception
# TYPE and reads ``.reason`` — NEVER a substring of the message, which stays the
# byte-stable coordinator ``permissionDecisionReason`` prose (the model's retry
# loop depends on that stability, auto-memory
# ``project_cc_strict_mode_retry_hazard``; the typed-signal-not-substring house
# rule). Renaming a value is a wire break; add, do not mutate.
STALE_VIEW_REASON = "stale_view"
COMMIT_PREEMPTED_REASON = "commit_preempted"
VIEW_WEDGED_REASON = "view_wedged"
COMMIT_UNCONFIRMED_REASON = "commit_unconfirmed"
#: The prefix of the reason in the coordinator's failure envelope, HTTP 200
#: ``{"ok": false, "reason": "internal: <Type>"}``: what it answers when a
#: route's work, or the caller-principal gate's store read, raised. The answer
#: decides nothing, so a client reads the request's outcome as unknown. The
#: exception's type follows the prefix, which is why this one reason is matched
#: as a prefix; the prefix itself is fixed.
HANDLER_FAILURE_REASON_PREFIX = "internal: "
CAS_EXHAUSTED_REASON = "cas_exhausted"
INTERNAL_CONCURRENCY_REASON = "internal_concurrency_error"
# Option-A single-shot CAS (MCP-C Unit 5): the caller's expected_version no
# longer matches the coordinator's current version. Typed-conflict, NOT
# auto-merge — the agent re-reads at current_version, re-merges, and retries.
VERSION_MISMATCH_REASON = "version_mismatch"
# Type B — synthesized by the mapper (no adapter raise-site) when the volume is
# unattached / coordinator transport failed; the write is NOT version-committed.
COORDINATOR_UNAVAILABLE_REASON = "coordinator_unavailable"


class CoherenceError(Exception):
    """Base error for coherence domain failures."""


class StaleReadGeneration(CoherenceError):
    """The read-generation fence rejected a commit: the committer's
    read_generation is older than the artifact's current owner_generation --
    its captured ownership/read-claim was superseded by a sweep reclamation.

    Raised on the pessimistic ``commit()`` path; the OCC
    ``commit_cas`` path returns a :class:`ConflictDetail` with the same reason
    instead. Retry-eligible: ``reacquire()`` + fresh read + re-commit. Carries
    ``STALE_READ_GENERATION_REASON`` so the client classifier matches it
    exactly (never on the human message)."""


class OccCallerTransientError(CoherenceError):
    """Retry-eligible OCC precondition: the caller is mid-transient at CAS time.

    Raised by ``CoordinatorService.commit_cas`` when a peer invalidated the
    caller between its fresh read and its commit-CAS — the registry left an
    invalidation transient that ``commit_cas`` rejects as a precondition. This
    is a LOST RACE, not corruption: a fresh identity (via ``reacquire()``) has
    no transient, so the client may retry.

    A dedicated type so the wire reason (:data:`OCC_CALLER_TRANSIENT_REASON`,
    surfaced by the coordinator server) is decoupled from the human-readable
    message — a reword of the message can no longer break the client's
    substring-free retry classification. The M/E-rejection and
    artifact-not-found branches of ``commit_cas`` stay plain
    :class:`CoherenceError`; only the transient precondition is retry-eligible.
    """


class SessionInvalidated(CoherenceError):
    """A snapshot session's pins are unavailable — fail-closed (SB-17 / TX-1,
    Unit 5 / R4). The session-liveness sweep reaped it (stale heartbeat), a GC
    race dropped a pinned body, or an in-memory coordinator restart wiped the
    pin store. The session can no longer serve its consistent cut, so any
    ``session_read`` / ``session_commit`` against it MUST fail closed — NEVER a
    live-HEAD fall-through.

    Carries :data:`SESSION_INVALIDATED_REASON` so a consumer classifies it by
    type / ``.reason`` (the typed-signal-not-substring house rule), distinct from
    a generic :class:`CoherenceError`. The service-layer ``session_read`` /
    ``session_commit`` surface returns the typed REJECTION
    (:class:`~ccs.core.types.SessionReadRejection` /
    :class:`~ccs.core.types.SessionCommitRejection`) carrying this same reason
    rather than raising, mirroring the ``ConflictDetail`` discipline; this
    exception is the raise-form for callers (e.g. an effect-gate, Unit 6) that
    want a hard failure on a dead session.

    Recovery is to OPEN A NEW SESSION (re-establish the cut + re-read), exactly
    like a ``version_mismatch`` HELD — the dead cut is not retry-eligible in
    place."""

    reason = SESSION_INVALIDATED_REASON


class CallerPrincipalRefused(CoherenceError):
    """A request's caller principal does not establish the identity it names,
    or a mint claim does not become an already-bound identity.

    ``reason`` is one of :data:`CALLER_PRINCIPAL_REASONS`; consumers branch on
    it, never on the message. The message never carries a principal or a mint
    nonce: it is written to logs and response ``detail`` fields, and a principal
    appears on exactly one response — the mint response that issues it.

    ``settled`` says whether the refusal is the session's settled state.
    ``True`` (the default): claiming again with the session's own nonce
    cannot cure it — the session is bound under another nonce, or the claim
    handed back the very principal that was refused — so every later request
    from this session meets the same answer, and only a new session claims
    its own. ``False``: the request was refused, but the recovery claim's
    ANSWER was lost (a transport blip, a watchdog-degraded claim), so nothing
    is known about the session's standing; the client claims again with the
    same nonce by itself — a long-lived volume, and a one-shot hook client
    holding no stored principal, before the next request; the substrate
    session, and a hook client whose stored principal is stale, when that
    request is refused — and the request may then be admitted. A consumer that tells an
    agent what to do next branches on this, never on the message: read as
    settled, a lost answer sends the agent to a new session for a state its
    next call cures.
    """

    def __init__(self, reason: str, message: str, *, settled: bool = True) -> None:
        super().__init__(message)
        self.reason = reason
        self.settled = settled


class CoherenceDegradedWarning(UserWarning):
    """Emitted once per adapter instance when a coherence error degrades to fallback.

    Canonical home so every adapter (CCSStore, OpenAIAgentsAdapter, ...) emits and
    catches the *same* class — ``from ccs.adapters import CoherenceDegradedWarning``
    must match whatever any adapter raises.
    """


class CoherenceTopologyWarning(UserWarning):
    """Emitted when an adapter is used in a topology its coherence model can't fully cover.

    Example: an OpenAI Agents run that combines a server-side ``conversation_id`` with
    multi-agent handoffs, where the SDK disables ``input_filter`` / nested handoff
    history — so handoff-history coherence is unavailable. Surfaced once, never silent.
    """


class InvalidTransitionError(CoherenceError):
    """Raised when the MESI transition table rejects a state transition."""

    def __init__(self, from_state: str, to_state: str, trigger: str):
        super().__init__(f"invalid_transition from={from_state} to={to_state} trigger={trigger}")
        self.from_state = from_state
        self.to_state = to_state
        self.trigger = trigger


class InvariantViolationError(CoherenceError):
    """Raised when a runtime invariant check fails."""


class CasRetriesExhausted(CoherenceError):
    """Raised when an optimistic-concurrency commit-CAS retry loop is exhausted.

    The typed terminal failure for the OCC write path (plan Unit 5, R6 /
    R-OCC-6). When ``AgentRuntime.write_cas`` or ``CoherentVolume.write_cas`` has
    retried ``commit_cas`` the allowed number of times and every attempt was
    refused (``ConflictDetail``), the loop surfaces THIS rather than silently
    dropping the write. A ``CasRetriesExhausted`` therefore means *no mutation
    landed for this caller* — the cache is left at the latest observed
    (refreshed) version, never corrupted with an unconfirmed write.

    :attr:`last_conflict_reason` is the coordinator's reason for the LAST
    refusal, when the raise site has it; earlier attempts may have been refused
    for other reasons. ``version_mismatch`` is a lost race. ``other_holder`` is
    not: another agent holds the grant at an unchanged version. Against the
    cross-process coordinator (``CoherentVolume``) nothing the caller does
    releases it. The holder does: a Claude Code session at its turn end, a
    ``CoherentVolume`` at its own next ``write_cas`` of that file or any
    ``write_cas`` retry, ``write_cas_at``, ``atomic_publish`` or
    ``reacquire()`` (a ``session-stop`` naming only that volume's
    ``session_id`` releases nothing). Otherwise the coordinator's reclaim
    does. In process (``AgentRuntime``) the loop's own re-fetch
    downgrades the holder, so ``other_holder`` there clears on the next attempt
    unless the holder re-acquires.

    A subclass of :class:`CoherenceError` so the deny-always-raises consumers
    (CoherentVolume / CCSStore strict mode) already treat it as a hard failure.

    Carries :data:`CAS_EXHAUSTED_REASON` (MCP-C Unit 1) so the deny mapper
    classifies it by type; the wire reason ``cas_exhausted`` is deliberately
    distinct from the ``cas_retries_exhausted`` prose token in the message.
    """

    reason = CAS_EXHAUSTED_REASON
    #: Class default, so a subclass that bypasses ``__init__``
    #: (``ConditionalPutRetriesExhausted``) still has the attribute.
    last_conflict_reason: str | None = None

    def __init__(
        self,
        artifact_id: object,
        attempts: int,
        last_current_version: int,
        *,
        last_conflict_reason: str | None = None,
    ) -> None:
        # isinstance, not truthiness alone: the reason arrives from a JSON body,
        # and an exception constructor must never raise (see CasVersionConflict).
        if isinstance(last_conflict_reason, str) and last_conflict_reason:
            self.last_conflict_reason = last_conflict_reason
        if self.last_conflict_reason is None:
            # No reason known (a raise site that does not pass one): the text
            # this message has always had.
            outcome = "(no write landed — every commit_cas attempt lost the race)"
        else:
            # .get, never []: a reason the advice table has not learned yet still
            # produces a usable terminal.
            outcome = (
                f"last_reason={self.last_conflict_reason} "
                f"{_CAS_EXHAUSTED_ADVICE.get(self.last_conflict_reason, _CAS_CONFLICT_ADVICE_FALLBACK)}"
            )
        super().__init__(
            f"cas_retries_exhausted artifact={artifact_id} attempts={attempts} "
            f"last_current_version={last_current_version} {outcome}"
        )
        self.artifact_id = artifact_id
        self.attempts = attempts
        self.last_current_version = last_current_version


class StaleView(CoherenceError):
    """Pre-edit deny (MCP-C Unit 1): this instance's view is INVALID — a peer
    committed a newer version, so the coordinator denied the write BEFORE any
    disk mutation. Recoverable: ``reacquire()`` for fresh bytes, then write FROM
    them. Carries :data:`STALE_VIEW_REASON`; the message stays the verbatim
    coordinator ``permissionDecisionReason`` (matched by type, not substring).

    ``expected_version`` / ``current_version`` and ``expected_generation`` /
    ``current_generation`` carry the captured-vs-current drift when raised by
    ``adapters.effect_gate.gate()`` — the version pair answers "did the value
    move", the generation pair answers "was the grant it was read under
    reclaimed" (a sweep reclamation advances the generation WITHOUT a version
    move). On every other raise path (the coordinator deny sites) all four stay
    ``None``, so a generic ``except StaleView`` handler reads them uniformly."""

    reason = STALE_VIEW_REASON
    #: Version drift set by gate()'s HOLD; ``None`` on coordinator-raised instances.
    expected_version: int | None = None
    current_version: int | None = None
    #: On a HELD batch publish, the first conflicting member's own refusal
    #: reason (``version_mismatch`` / ``other_holder`` / ...). ``None``
    #: everywhere else, so a generic handler reads it uniformly.
    member_reason: str | None = None
    #: WHICH hold class fired, as a typed value (see the ``HOLD_*`` constants).
    #: ``None`` on coordinator-raised instances. Matched on this, never on the
    #: human message — and it is the difference between a HOLD ``reacquire()``
    #: clears and one it never can (``HOLD_GENERATION_UNCONFIRMED`` against a
    #: coordinator that does not report generations is an operator fix).
    hold_cause: str | None = None
    #: Ownership-epoch drift set by gate()'s HOLD; ``None`` on coordinator-raised
    #: instances AND when the coordinator never confirmed a generation.
    expected_generation: int | None = None
    current_generation: int | None = None


class CommitPreempted(CoherenceError):
    """Post-edit deny (MCP-C Unit 1): the EXCLUSIVE grant was preempted /
    sweep-reclaimed AFTER the atomic disk write but BEFORE the commit landed, so
    the bytes may already be on disk *un-versioned* (disk ahead of the
    coordinator version). NOT only a concurrent edge — a lone sequential writer's
    grant can age out mid-write (crash-recovery sweep, default-on). Recover by
    re-reading fresh bytes and reconciling the pending buffer (agent-driven; v1
    has no server reconcile primitive). Carries :data:`COMMIT_PREEMPTED_REASON`."""

    reason = COMMIT_PREEMPTED_REASON


class ViewWedged(CoherenceError):
    """OCC comparand wedged (MCP-C Unit 1): a ``write_cas`` comparand read stayed
    strict-denied across the bounded reacquire streak — the view never cleared to
    a usable state. Not retry-eligible in-loop: wait or escalate. Carries
    :data:`VIEW_WEDGED_REASON`."""

    reason = VIEW_WEDGED_REASON


class RemoteAuthFailed(CoherenceError):
    """Remote-coordinator bearer auth rejected (cross-host demo, R2): the
    coordinator returned ``401`` for the supplied secret. A misconfiguration —
    the remote ``CCS_REMOTE_SECRET_FILE`` does not match the coordinator's
    ``hook.secret`` — NOT an infra hiccup. It fails LOUD and CLOSED, typed
    distinctly from a watchdog-timeout degrade or a stale-view deny, so a remote
    client never silently degrades past a wrong secret."""


class InsecureTransportRefused(CoherenceError):
    """Refused to send a bearer token to a NON-loopback coordinator over plaintext
    HTTP without an explicit operator acknowledgement (Phase-1.5 client-side guard).

    The remote transport is always ``http://`` — encryption (WireGuard / a TLS
    terminating proxy) is provided out-of-band by the operator, so there is no
    in-band TLS signal. Rather than silently leak the bearer to a routed host, the
    client fails LOUD and CLOSED: set ``CCS_REMOTE_INSECURE=1`` to acknowledge the
    link is secured out-of-band, or point at a loopback host. This *reduces* the
    silent-plaintext footgun; it does not *guarantee* the link is encrypted (the
    ack is an operator assertion). The guard is evaluated once, when the endpoint is
    minted (per process / instance) — not re-checked per request. Carries the
    offending ``host``."""

    def __init__(self, host: str) -> None:
        super().__init__(
            f"refusing to send a bearer token to non-loopback host {host!r} over "
            "plaintext HTTP; set CCS_REMOTE_INSECURE=1 to acknowledge the link is "
            "secured out-of-band (e.g. WireGuard / a TLS-terminating proxy), or use "
            "a loopback coordinator"
        )
        self.host = host


class TlsVerificationFailed(CoherenceError):
    """The coordinator's TLS certificate did not verify against the trusted CA
    (Unit-1 client-side ``https://`` guard).

    Raised when a verified-TLS request fails certificate validation — a wrong /
    untrusted signing CA, a hostname/IP-SAN mismatch (e.g. a DNS-only cert on an
    IP-literal endpoint), an expired cert, or any other
    :class:`ssl.SSLCertVerificationError`. The client *always* verifies (there is
    no insecure ``https`` mode), so this fails LOUD and CLOSED: the bearer is NOT
    retried over plaintext and no request body reaches the peer — the handshake
    dies before the HTTP exchange. Distinct from :class:`CoordinatorUnavailable`
    (a network hiccup that a caller might treat as transient); a verification
    failure is a trust decision, never a degrade. Carries the offending
    ``host``."""

    def __init__(self, host: str, detail: str = "") -> None:
        suffix = f": {detail}" if detail else ""
        super().__init__(
            f"TLS certificate verification failed for coordinator host {host!r}"
            f"{suffix}; the bearer was NOT sent — check the server certificate and "
            "the trusted CA (CCS_REMOTE_CA_FILE), then retry"
        )
        self.host = host
        self.detail = detail


class TlsConfigError(CoherenceError):
    """The TLS client configuration is unusable (Unit-1 factory guard).

    Raised at SSL-context build time for an operator misconfiguration of the
    CA-trust anchor: ``CCS_REMOTE_CA_FILE`` names a path that is missing,
    unreadable, a symlink (refused via ``O_NOFOLLOW`` — a swapped trust anchor is
    the same attack class as a swapped bearer file), group/world-*writable*, or
    not valid PEM. Fails CLOSED with an actionable message naming the path rather
    than surfacing a raw :class:`ssl.SSLError` / :class:`OSError`. Also raised if
    the built context ever fails its own hardening invariant
    (``check_hostname`` + ``CERT_REQUIRED``) — that guards against a future edit
    silently weakening verification. Carries the CA ``path`` (when applicable)
    and a human ``reason``."""

    def __init__(self, reason: str, *, path: str | None = None) -> None:
        super().__init__(reason)
        self.path = path
        self.reason = reason


class RedirectRefused(CoherenceError):
    """The coordinator responded with a 3xx redirect; the client refuses to
    follow it (Unit-1 no-redirects-ever policy).

    The client talks to ONE fixed, operator-configured coordinator endpoint — no
    redirect is ever legitimate. A bare ``urlopen`` would auto-follow and
    urllib's default redirect handler *copies the Authorization header onto the
    new hop* before application code can intervene, so the bearer would ride an
    attacker-chosen URL. The client therefore refuses ANY 3xx (both ``http`` and
    ``https`` endpoints) before a second request is made — the bearer never
    leaves the configured endpoint. Carries the ``status``. The client never
    passes the ``Location`` it was sent as ``location``, only a placeholder:
    the ``Location`` is the redirector's text, and may echo a principal or a
    mint nonce it was sent."""

    def __init__(self, location: str, status: int | None = None) -> None:
        status_part = f" ({status})" if status is not None else ""
        super().__init__(
            f"coordinator returned a redirect{status_part} to {location!r}; refusing "
            "to follow — the coordinator is one fixed endpoint and following a "
            "redirect would leak the bearer onto another host"
        )
        self.location = location
        self.status = status


class CommitUnconfirmed(CoherenceError):
    """OCC commit unconfirmed (MCP-C Unit 1): the coordinator transport failed
    mid-commit, a false-negative ack — NOT a confirmed loss. The write may or may
    not have landed; reconcile by re-reading, then retry only if absent. Carries
    :data:`COMMIT_UNCONFIRMED_REASON`."""

    reason = COMMIT_UNCONFIRMED_REASON


class PublishMaterializationError(CoherenceError):
    """An ``atomic_publish`` batch COMMITTED at the coordinator but then failed to
    materialize to disk. The coordinator has already advanced every member's
    version and content hash — this is NOT retryable as a fresh publish (a retry
    would version-mismatch). ``landed`` names the members whose new bytes reached
    disk; ``not_landed`` names the members still holding their old bytes. When
    ``landed`` is non-empty the on-disk set is TORN relative to the coordinator;
    when it is empty the disk is uniformly stale (coordinator ahead of disk).
    Recover by re-materializing each ``not_landed`` member with ``write()`` of
    the bytes this publish committed for it. Until those bytes reach disk, a
    version-checked read of that member (``read_with_version``) raises
    ``StaleView``: its disk bytes are not the content at the coordinator's
    version, so no version is returned for them."""

    def __init__(
        self,
        landed: "tuple[str, ...]",
        not_landed: "tuple[str, ...]",
        cause: BaseException | None = None,
    ) -> None:
        detail = f" ({cause})" if cause is not None else ""
        super().__init__(
            f"atomic_publish committed at the coordinator but disk materialization "
            f"failed{detail}: landed={list(landed)} not_landed={list(not_landed)}. "
            "The coordinator is ahead of disk — re-materialize each not_landed "
            "member with write() of the bytes this publish committed for it (a "
            "versioned read of it is refused until then); do NOT retry the "
            "publish (it would version-mismatch)."
        )
        self.landed = landed
        self.not_landed = not_landed


class InternalConcurrencyError(CoherenceError):
    """The single-op guard fired (MCP-C Unit 1): one CoherentVolume instance was
    used concurrently from another thread — a SERVER misuse bug (the MCP server
    serializes tool access), never an agent-recoverable deny. Mapped to
    :data:`INTERNAL_CONCURRENCY_REASON`, distinct from ``stale_view`` so it is
    never relayed to the model as a retryable view."""

    reason = INTERNAL_CONCURRENCY_REASON


#: Per-reason recovery advice for :class:`CasVersionConflict`'s message. The
#: four reasons do not share a recovery: only a genuine lost race is fixed by
#: re-reading and re-merging.
_CAS_CONFLICT_ADVICE: dict[str, str] = {
    VERSION_MISMATCH_REASON: (
        "(no write landed; re-read at current and re-merge)"
    ),
    "other_holder": (
        "(no write landed; a peer holds the grant and the version has NOT "
        "moved — re-reading returns the same bytes, so back off and retry "
        "rather than re-merging)"
    ),
    STALE_READ_GENERATION_REASON: (
        "(no write landed; the claim this read was taken under was reclaimed "
        "— reacquire and re-read, a merge alone does not help)"
    ),
    OCC_CALLER_TRANSIENT_REASON: (
        "(no write landed; a peer invalidated this session between its read "
        "and its CAS — recover with a fresh identity)"
    ),
}

#: Used when the coordinator reports a reason this table does not know.
_CAS_CONFLICT_ADVICE_FALLBACK = "(no write landed; re-read before retrying)"

#: Per-reason text for :class:`CasRetriesExhausted`, keyed on the LAST refusal.
#: It differs from the single-refusal advice above: the retry loop has already
#: re-read and re-minted, so it says what the last refusal means for a caller
#: deciding whether to try again. Only ``version_mismatch`` is a lost race;
#: ``caller_in_transient_state`` also arrives when a peer's ``write()`` has
#: acquired the file and still holds it, with the version unmoved.
_CAS_EXHAUSTED_ADVICE: dict[str, str] = {
    VERSION_MISMATCH_REASON: (
        "(no write landed — the last commit_cas attempt lost the race: a peer "
        "committed first)"
    ),
    "other_holder": (
        "(no write landed — another agent still holds the grant at an unchanged "
        "version; retry after it releases, not in a loop)"
    ),
    STALE_READ_GENERATION_REASON: (
        "(no write landed — the grant the last read was taken under was "
        "reclaimed; reacquire and re-read)"
    ),
    OCC_CALLER_TRANSIENT_REASON: (
        "(no write landed — a peer's write() invalidated this caller between its "
        "last read and its CAS; that peer may still hold the grant)"
    ),
}


class CasVersionConflict(CoherenceError):
    """Option-A CAS rejected (MCP-C Unit 5). Typed-conflict, NOT auto-merge: NO
    write landed, and both versions are carried so the client can recover
    without another read.

    WHY it was rejected is :attr:`reason`, and the four values do not share a
    recovery. ``version_mismatch`` is the lost race the class is named for — a
    peer committed in between, so re-read at ``current_version``, re-merge and
    retry. ``other_holder`` means a pessimistic peer holds the grant with the
    version UNCHANGED, so a re-read returns the same bytes and only backing off
    makes progress. ``stale_read_generation`` means the claim this read was
    taken under was reclaimed by the sweep: reacquire, then re-read.
    ``caller_in_transient_state`` means a peer invalidated this session between
    its read and its CAS: recover with a fresh identity.

    ``current_version`` is authoritative when the coordinator itself reported it
    (a bump-conflict). On a substrate-side CAS loss — where the coordinator was
    not re-read — it is best-effort (the version observed at pre-read time); a
    caller keys recovery on ``reacquire()`` (which re-reads fresh), not on the
    numeric.
    """

    #: Class default, kept so ``except`` clauses and consumers that read the
    #: class attribute keep working. An instance raised from a wire refusal
    #: SHADOWS it with the coordinator's own reason.
    reason = VERSION_MISMATCH_REASON

    def __init__(
        self,
        artifact_id: object,
        expected_version: int,
        current_version: int,
        *,
        reason: str | None = None,
    ) -> None:
        """``reason`` is the coordinator's refusal reason when the raise site
        has one. The coordinator distinguishes four — ``version_mismatch``,
        ``other_holder``, ``stale_read_generation`` and
        ``caller_in_transient_state`` — and each needs different recovery, so
        collapsing them into the class default misdirected the caller. The
        worst case is ``other_holder``: the version has NOT moved, so
        "re-read at current and re-merge" produces a byte-identical CAS that
        fails identically until the holder releases.
        """
        # isinstance, not truthiness alone: ``reason`` arrives from a JSON
        # response body, and a non-string there (a list, a dict) would make the
        # advice lookup below raise TypeError INSIDE an exception constructor,
        # turning a recoverable conflict into an unhandled error at the raise
        # site. A constructor must never raise.
        self.reason = reason if isinstance(reason, str) and reason else VERSION_MISMATCH_REASON
        super().__init__(
            f"{self.reason} artifact={artifact_id} expected={expected_version} "
            f"current={current_version} "
            # .get, never []: an exception constructor must not raise. A reason
            # this table has not learned yet still produces a usable terminal.
            f"{_CAS_CONFLICT_ADVICE.get(self.reason, _CAS_CONFLICT_ADVICE_FALLBACK)}"
        )
        self.artifact_id = artifact_id
        self.expected_version = expected_version
        self.current_version = current_version


class ScenarioValidationError(CoherenceError):
    """Raised when scenario configuration does not match schema expectations."""

    def __init__(self, path: str, message: str):
        super().__init__(f"scenario={path}: {message}")
        self.path = path


# ---------------------------------------------------------------------------
# Coherence-manifest load/validation errors (BYO-substrate, Unit 2 / R8, R9)
# ---------------------------------------------------------------------------
#
# The manifest is a named TRUST BOUNDARY: it wires artifact identities to
# substrate connection targets and credential references supplied by the
# builder. Every failure below is caught at LOAD/VALIDATE time — before any
# substrate connection is attempted — so a bad target or an inline literal
# secret can never reach a driver. Discipline mirrors the shipped
# plaintext-bearer guard (``ccs.cli._coherence_client._guard_plaintext_bearer``):
# the message names the HOST and the POSTURE only, NEVER a resolved credential
# value or a DSN that could carry one.


class ManifestError(CoherenceError):
    """A coherence manifest is malformed or fails validation (Unit 2 / R8, R9).

    The base for every load/validate rejection: an unknown top-level key, a
    missing/unknown tier, a forward-only artifact declaring a version_source
    (rejected via the descriptor's own validation), and the security subclasses
    below. Raised at config time, never mid-run."""


class SubstrateCredentialRefused(ManifestError):
    """A manifest connection credential is a suspected inline literal, not a
    reference form (Unit 2 / R8).

    Credentials must be one of the allowlisted reference forms
    (``secret-file:PATH``, ``secret:URI``, ``env:VARNAME``, ``aws-default``) so
    the secret value never lives in the manifest. An inline DSN, a bare key, an
    unknown prefix, or an inline password inside a DSN target is refused here.
    The message states the CATEGORY of violation only — it NEVER echoes the
    offending value, which may itself be the secret."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            f"refusing a suspected inline secret in the manifest connection: {detail}. "
            "Credentials must be a reference form: secret-file:PATH (preferred), "
            "aws-default (zero-secret), secret:URI (allowlisted providers), or "
            "env:VARNAME (least-preferred)"
        )
        self.detail = detail


class SubstrateTargetDenied(ManifestError):
    """A manifest connection target resolves to a denied address (Unit 2 / R8, SSRF).

    The deny runs on the ``getaddrinfo``-RESOLVED address(es), not the literal
    string, so a DNS-rebind hostname and decimal/octal/hex IPv4 encodings are
    all classified. Hard-denied: the cloud metadata / link-local class
    (169.254.0.0/16, 100.64.0.0/10, metadata.google.internal, fd00:ec2::254,
    fe80::/10), unwrapped through any mapped/compat IPv6 wrapper. RFC-1918/4193
    private ranges are denied unless the per-manifest opt-in
    (``CCS_SUBSTRATE_ALLOW_PRIVATE``) is set. Carries ``host`` and a ``posture``
    describing the denied range — never a credential."""

    def __init__(self, host: str, posture: str) -> None:
        super().__init__(
            f"manifest connection target {host!r} is denied: {posture}"
        )
        self.host = host
        self.posture = posture


class SubstrateInsecureTransport(ManifestError):
    """A plaintext-credential substrate config points at a routable host without
    an ack (Unit 2 / R8 — the substrate analog of the plaintext-bearer guard).

    A Postgres DSN with ``sslmode=disable``/unset (libpq silently downgrades to
    plaintext) or a boto3 ``endpoint_url=http://`` to a non-loopback host ships
    the credential in cleartext. Refused unless the operator sets the DISTINCT
    ack ``CCS_SUBSTRATE_INSECURE`` (deliberately NOT ``CCS_REMOTE_INSECURE`` /
    ``CCS_REMOTE_COORDINATOR`` — relaxing coordinator transport must not relax
    substrate egress). Carries ``host`` and ``posture``; never the credential."""

    def __init__(self, host: str, posture: str) -> None:
        super().__init__(
            f"refusing a plaintext substrate credential to routable host {host!r} "
            f"({posture}); set CCS_SUBSTRATE_INSECURE to acknowledge an "
            "out-of-band-secured link, or use TLS (postgres sslmode=require/"
            "verify-full, or an https S3 endpoint_url)"
        )
        self.host = host
        self.posture = posture


# ---------------------------------------------------------------------------
# Workspace-Versioning restore vocabulary (WV plan Unit 4 / R3)
# ---------------------------------------------------------------------------
#
# The restore engine (``ccs.adapters.workspace.WorkspaceVersioner.restore``)
# concludes with a per-member TERMINAL outcome for EVERY manifest member — the
# termination contract: bounded re-drive, absorbing outcomes, a complete typed
# report. Each outcome below is a wire-stable constant matched by IDENTITY
# (add, never rename); modality is part of the contract:
#
# - success  — ``restored`` (the leg landed the pinned state) /
#   ``converged`` (the live state already matched the manifest, so NO write was
#   issued — authorship is never claimed, mirroring ReconcileVerdict.CONVERGE);
# - absorbing — ``conflict`` (live-writer contention absorbed the bounded
#   re-drive budget, or a divergence the leg cannot converge), ``target_lost``
#   (the manifested restore pointer no longer resolves — expired/raced pin),
#   ``forward_only_skipped`` (nothing to restore; enumerated, never silent);
# - hold — ``held_unconfirmed`` (a write's outcome stayed UNKNOWN after the
#   binding's reconciliation read: HOLD, never best-effort — coordinator state
#   must never advance on it).
#
# Every outcome is terminal: a restore never leaves a member outcome-less once
# it concludes, and a crash-resumed run skips members already terminal.
RESTORE_OUTCOME_RESTORED = "restored"
RESTORE_OUTCOME_CONVERGED = "converged"
RESTORE_OUTCOME_CONFLICT = "conflict"
RESTORE_OUTCOME_HELD_UNCONFIRMED = "held_unconfirmed"
RESTORE_OUTCOME_TARGET_LOST = "target_lost"
RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED = "forward_only_skipped"

# The modality split, machine-usable (the report and Unit-5 registration branch
# on these): success outcomes may carry a freshly minted pointer; absorbing and
# hold outcomes never do.
RESTORE_SUCCESS_OUTCOMES: frozenset[str] = frozenset(
    {RESTORE_OUTCOME_RESTORED, RESTORE_OUTCOME_CONVERGED}
)
RESTORE_ABSORBING_OUTCOMES: frozenset[str] = frozenset(
    {
        RESTORE_OUTCOME_CONFLICT,
        RESTORE_OUTCOME_TARGET_LOST,
        RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED,
    }
)
RESTORE_HOLD_OUTCOMES: frozenset[str] = frozenset({RESTORE_OUTCOME_HELD_UNCONFIRMED})

# The closed set of member terminal outcomes. Consumers match membership /
# ``outcome == CONSTANT`` — never a substring of any human detail line.
RESTORE_MEMBER_OUTCOMES: frozenset[str] = (
    RESTORE_SUCCESS_OUTCOMES | RESTORE_ABSORBING_OUTCOMES | RESTORE_HOLD_OUTCOMES
)

# The outcomes that PROVE no write landed, readable from a durable row alone.
# ``converged`` short-circuits before every leg's write; ``forward_only_skipped``
# is enumerated and never driven; and every ``conflict`` construction site is
# either pre-write or says "no write landed" in its own detail (a budget
# exhausted, a wedged view, an absent member the v1 leg cannot recreate).
#
# ``target_lost`` is DELIBERATELY absent: the absorbing boundary catches an
# OSError raised from anywhere in a leg, including one raised after the live
# file was truncated and partially rewritten, so that outcome cannot vouch for
# the bytes. ``restored`` and ``held_unconfirmed`` wrote, or may have.
RESTORE_OUTCOMES_PROVING_NO_WRITE: frozenset[str] = frozenset(
    {
        RESTORE_OUTCOME_CONVERGED,
        RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED,
        RESTORE_OUTCOME_CONFLICT,
    }
)

# ---------------------------------------------------------------------------
# Workspace-Versioning restore OBSERVATION vocabulary (restore-divergence
# signal / R1-R3)
# ---------------------------------------------------------------------------
#
# What a restore leg SAW of the live state immediately before it wrote, carried
# BESIDE the member outcome (``MemberRestoreOutcome.observation``) and never
# folded into it: a member restored over a peer's committed content still
# concludes ``restored``. Wire-stable constants matched by IDENTITY (add, never
# rename); no token here is a substring of another, because a consumer must
# never be able to classify one state by matching a fragment of a second.
#
# - ``observed_differs`` — the leg read a live state, compared it with the
#   capture, and they differ: the write discarded content committed after the
#   capture. This is the ONLY state that carries a pointer and a fingerprint,
#   both lifted from the read that produced the leg's CAS comparand.
# - ``no_live_state`` — the leg wrote onto nothing (create-on-absent); nothing
#   was discarded, so this state is not a divergence.
# - ``present_not_comparable`` — the leg established that live state EXISTED
#   but read no comparand for it (the delete leg's presence probe verifies no
#   content), so it reports the state alone: honest about what it destroyed,
#   silent about what that content was.
# - ``no_write_attempted`` — the member reached its terminal without a write
#   decision: converged, enumerated and skipped, or absorbed before any write
#   was issued (a wedged view, an exhausted re-drive budget). The DEFAULT,
#   because it is the truth at every such site. It is NOT the answer for an arm
#   that issued a write and could not learn the outcome — see ``not_recorded``.
# - ``not_recorded`` — this run holds no observation for the member, in either
#   of two ways. (1) No leg ran: a member resumed from a prior run, or a
#   concluded restore rebuilt from its durable rows, where the row's outcome
#   does not itself prove the member never wrote (see
#   RESTORE_OUTCOMES_PROVING_NO_WRITE — the observation is run-local by
#   decision, so a prior run's is unrecoverable). (2) A leg ran and its write
#   outcome was lost: an unconfirmed commit, a reconciled unknown write, or a
#   failure absorbed from anywhere inside the leg. Such a leg cannot say
#   whether IT destroyed live state or never reached it. An observation the run
#   never made is its own answer, never a clean one: consumers MUST NOT read it
#   as clean.
RESTORE_OBSERVATION_DIFFERS = "observed_differs"
RESTORE_OBSERVATION_NO_LIVE_STATE = "no_live_state"
RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE = "present_not_comparable"
RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED = "no_write_attempted"
RESTORE_OBSERVATION_NOT_RECORDED = "not_recorded"

# The closed set of observation states. DELIBERATELY disjoint from
# RESTORE_MEMBER_OUTCOMES: the observation is an additive report value and the
# outcome vocabulary is unchanged, which is what keeps the coordinator's
# fail-closed check and the cross-implementation kit green with no edits.
RESTORE_OBSERVATION_STATES: frozenset[str] = frozenset(
    {
        RESTORE_OBSERVATION_DIFFERS,
        RESTORE_OBSERVATION_NO_LIVE_STATE,
        RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE,
        RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
        RESTORE_OBSERVATION_NOT_RECORDED,
    }
)

# The states that observed NO comparand, so they can never name a pointer or a
# fingerprint: three never read one, and the delete leg's probe read presence
# only. Enforced at construction — a state carrying values it never observed
# would tell an operator the run compared content it never saw.
RESTORE_OBSERVATION_STATES_WITHOUT_COMPARAND: frozenset[str] = (
    RESTORE_OBSERVATION_STATES - {RESTORE_OBSERVATION_DIFFERS}
)

# Checkpoint-level restore status (the ``CheckpointRecord.restore_status``
# vocabulary — the registry stores the string, THIS is its meaning). Kept
# deliberately small: ``none`` (never restored) → ``in_progress`` (a restore
# run is driving legs; a checkpoint FOUND in this state by a fresh engine is a
# crashed run and is RESUMED, never refused) → ``registered`` (WV Unit 5: every
# member is terminal AND the coordinator-registration step ran to its answer —
# the durable idempotency marker; a resumed run finding this status skips
# registration, so a crash between the registration commit and ``concluded``
# can never double-register) → ``concluded`` (every member holds a terminal
# outcome; the report is reconstructible from durable rows). ``registered`` is
# written ONLY on a committed/empty registration — a REFUSED registration
# concludes without it so a later resume may re-attempt (the refusal may have
# been transient contention; the fence rejects a superseded controller again).
RESTORE_STATUS_NONE = "none"
RESTORE_STATUS_IN_PROGRESS = "in_progress"
RESTORE_STATUS_REGISTERED = "registered"
RESTORE_STATUS_CONCLUDED = "concluded"
RESTORE_STATUSES: frozenset[str] = frozenset(
    {
        RESTORE_STATUS_NONE,
        RESTORE_STATUS_IN_PROGRESS,
        RESTORE_STATUS_REGISTERED,
        RESTORE_STATUS_CONCLUDED,
    }
)

# ---------------------------------------------------------------------------
# Workspace-Versioning member PIN-STATE vocabulary (WV plan Unit 6 / R9)
# ---------------------------------------------------------------------------
#
# The ``CheckpointMember.pin_state`` vocabulary (the registry stores the
# string; THIS is its meaning). Wire-stable constants matched by IDENTITY
# (add, never rename). The honesty contract is the PAIR (restore_tier,
# pin_state): tier ``restorable`` is BACKED only when pin_state is ``held`` —
# every consumer (the Unit-8 ``status`` surface included) must render
# (``restorable``, ``unpinned``) as claimed-but-not-yet-backed, never as a
# guarantee. A pin that cannot be established never leaves tier ``restorable``
# (the fail-closed rule: the failed attempt writes the loud tier downgrade in
# the SAME registry write as its pin state).
#
# - ``unpinned`` — no pin established (the capture-time default; also the
#   honest durable state when pin orchestration was skipped via ``pin=False``
#   or died before this member's leg ran).
# - ``held`` — the pin the member's substrate offers is ESTABLISHED. S3: the
#   Object Lock legal hold is ON for the manifested versionId (a lifecycle
#   expiry and a version-targeted delete are refused by the substrate). File:
#   the captured version was VERIFIED retained-and-readable under the
#   coordinator's declared retention at pin time — v1's verification pin: the
#   tier stays ``restorable-unpinned`` because coordinator retention offers no
#   per-version hold (a bounded K/T policy may still evict; restore then
#   reports ``target_lost`` loudly — the R13 declared-retention disclosure).
# - ``pin_unavailable`` — the pin attempt RAN and could not be established
#   (no Object Lock configuration, the pinned version already gone, retention
#   off / the version not retained, or retained bytes that no longer match the
#   captured fingerprint). Always written together with the loud tier
#   downgrade — never a silent pass.
# - ``released`` — a previously ``held`` pin was dropped by the release path
#   (``WorkspaceVersioner.release_checkpoint``, or the engine beneath it);
#   TERMINAL — ``pin_checkpoint`` will not re-drive it, and there is still no
#   public checkpoint-DELETE verb in v1;
#   written together with the tier downgrade where the tier was ``restorable``
#   (the claim is no longer backed).
PIN_STATE_UNPINNED = "unpinned"
PIN_STATE_HELD = "held"
PIN_STATE_UNAVAILABLE = "pin_unavailable"
PIN_STATE_RELEASED = "released"
PIN_STATES: frozenset[str] = frozenset(
    {
        PIN_STATE_UNPINNED,
        PIN_STATE_HELD,
        PIN_STATE_UNAVAILABLE,
        PIN_STATE_RELEASED,
    }
)

# ---------------------------------------------------------------------------
# Workspace-Versioning restore REGISTRATION vocabulary (WV plan Unit 5 / R4-R5)
# ---------------------------------------------------------------------------
#
# The terminal answer of the coordinator-registration step that runs after
# every member holds a terminal outcome and before the restore concludes.
# Wire-stable constants matched by IDENTITY (add, never rename). The per-
# member-class registration split (the plan's restore-registration design):
#
# - WRITTEN file members (``restored``, no delete record, ``no-arbiter``
#   tier) register through ``commit_all`` — all-or-nothing, hash-only
#   (fingerprints, never content bytes), an S/I controller per D4;
# - WRITTEN S3 members are registered MANIFEST-SIDE ONLY: the coordinator
#   holds no artifact identity for a BYO substrate member (the substrate owns
#   identity — ``workspace_checkpoint_members.artifact_id`` is NULL for them
#   by design), so their registration IS the durable outcome row;
# - DELETED members are recorded manifest-side (``deleted_at_restore`` —
#   ``commit_all`` has no delete semantics);
# - an EMPTY commit write-set NEVER calls ``commit_all`` (it raises on empty).
WORKSPACE_REGISTRATION_COMMITTED = "committed"
WORKSPACE_REGISTRATION_EMPTY = "empty_write_set"
WORKSPACE_REGISTRATION_PRIOR_RUN = "registered_by_prior_run"
WORKSPACE_REGISTRATION_REFUSED = "refused"
WORKSPACE_REGISTRATION_STATUSES: frozenset[str] = frozenset(
    {
        WORKSPACE_REGISTRATION_COMMITTED,
        WORKSPACE_REGISTRATION_EMPTY,
        WORKSPACE_REGISTRATION_PRIOR_RUN,
        WORKSPACE_REGISTRATION_REFUSED,
    }
)

# Pre-flight restore refusal: the checkpoint id names no persisted manifest.
# One of the two typed restore pre-flight exceptions (the other is
# ``CheckpointRegistrationRefused`` below) — a restore that starts always
# CONCLUDES with a report (failures are per-member absorbing outcomes).
CHECKPOINT_UNKNOWN_REASON = "checkpoint_unknown"


class CheckpointUnknown(CoherenceError):
    """``restore(checkpoint_id)`` pre-flight refusal: no such checkpoint.

    Raised BEFORE any status/progress write — an unknown id must never mint an
    ``in_progress`` record. Carries :data:`CHECKPOINT_UNKNOWN_REASON`, matched
    by identity (the typed-signal-not-substring house rule), plus the offending
    ``checkpoint_id``.
    """

    reason = CHECKPOINT_UNKNOWN_REASON

    def __init__(self, checkpoint_id: str) -> None:
        super().__init__(
            f"checkpoint {checkpoint_id!r} is unknown: no persisted manifest — "
            "nothing was restored (list checkpoints and retry with a known id)"
        )
        self.checkpoint_id = checkpoint_id


# Restore-REGISTRATION pre-flight refusals (#191). Each is a typed reason a
# client branches on by identity, never by parsing prose; every one is raised
# BEFORE any artifact is resolved or minted and before ``commit_all`` runs, so
# a refused registration changes nothing.
#
# - ``not_a_checkpoint_member`` — a write names a path the checkpoint's
#   manifest does not describe (the stale-id case: a write-set registered
#   against a checkpoint from another round);
# - ``fingerprint_mismatch`` — the path IS a member, but the write's
#   fingerprint is not the one the manifest captured for it (a restore
#   registers the captured bytes, never a caller-chosen hash);
# - ``not_the_receiver`` — the checkpoint named a receiver at creation and the
#   registering controller is not it;
# - ``already_registered`` — another controller already registered this
#   checkpoint. The refusal names nobody: the caller learns only that it was
#   not first. A retry by the controller that DID register is not refused; its
#   result carries ``retry_of_own_registration=True`` instead.
CHECKPOINT_NOT_A_MEMBER_REASON = "not_a_checkpoint_member"
CHECKPOINT_FINGERPRINT_MISMATCH_REASON = "fingerprint_mismatch"
CHECKPOINT_NOT_THE_RECEIVER_REASON = "not_the_receiver"
CHECKPOINT_ALREADY_REGISTERED_REASON = "already_registered"
CHECKPOINT_REGISTRATION_REFUSAL_REASONS: frozenset[str] = frozenset(
    {
        CHECKPOINT_NOT_A_MEMBER_REASON,
        CHECKPOINT_FINGERPRINT_MISMATCH_REASON,
        CHECKPOINT_NOT_THE_RECEIVER_REASON,
        CHECKPOINT_ALREADY_REGISTERED_REASON,
    }
)


class CheckpointRegistrationRefused(CoherenceError):
    """A restore registration refused before anything was resolved (#191).

    ``reason`` is one of :data:`CHECKPOINT_REGISTRATION_REFUSAL_REASONS`,
    matched by identity (the typed-signal-not-substring house rule): the
    instance carries the module constant itself, never an equal copy.
    ``member_paths`` names the offending writes for the two membership reasons
    (the caller's own paths, nothing it did not send) and is empty for the two
    controller reasons. Neither controller reason names another controller.

    Raised by ``CoordinatorService.register_workspace_restore`` and, for the
    two controller reasons, by ``WorkspaceVersioner.restore`` as a pre-flight
    refusal before any status write or member leg.
    """

    def __init__(
        self,
        checkpoint_id: str,
        reason: str,
        *,
        member_paths: "tuple[str, ...]" = (),
    ) -> None:
        canonical = {r: r for r in CHECKPOINT_REGISTRATION_REFUSAL_REASONS}
        if reason not in canonical:
            raise ValueError(
                f"unknown checkpoint registration refusal reason {reason!r}"
            )
        reason = canonical[reason]
        paths = list(member_paths)
        detail = {
            CHECKPOINT_NOT_A_MEMBER_REASON: (
                "the write-set names paths the checkpoint's manifest does not "
                f"describe: {paths!r}"
            ),
            CHECKPOINT_FINGERPRINT_MISMATCH_REASON: (
                "the write-set's fingerprints differ from the ones the manifest "
                f"captured for: {paths!r}"
            ),
            CHECKPOINT_NOT_THE_RECEIVER_REASON: (
                "the checkpoint names a receiver and this controller is not it"
            ),
            CHECKPOINT_ALREADY_REGISTERED_REASON: (
                "another controller already registered this checkpoint"
            ),
        }[reason]
        super().__init__(
            f"restore registration of checkpoint {checkpoint_id!r} refused "
            f"({reason}): {detail} — nothing was registered"
        )
        self.checkpoint_id = checkpoint_id
        self.reason = reason
        self.member_paths = tuple(member_paths)


class WatchdogAbandoned(RuntimeError):
    """A handler's 4s watchdog fired, so its still-running work was told to abort
    before it could mutate the registry (finding A6).

    Raised by the registries' ``abort_guard`` when the per-request abort
    :class:`threading.Event` is already set at the moment the mutation wins the
    registry write lock — i.e. the handler timed out (and the client already got
    ``degraded: true``) while this work was blocked on that lock. Aborting there
    is what stops the late "phantom grant" / grant-revocation from landing.

    Deliberately NOT a :class:`CoherenceError`: it never reaches a client. The
    only caller that ever sets the abort Event is the handler watchdog, which
    has already responded; this exception surfaces solely inside the abandoned
    pool future, where ``_on_watchdog_future_done_after_timeout`` treats it as a
    clean no-op (no phantom state landed). Every non-watchdog caller
    (CoherentVolume, CCSStore, the CLI) passes ``abort=None`` and never sees it.
    """
