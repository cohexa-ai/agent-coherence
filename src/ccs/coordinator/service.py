# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Coordinator service implementing core artifact coherence operations."""

from __future__ import annotations

import hmac
import logging
import secrets
import string
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from ccs.coordinator.retention import collectible_versions
from ccs.core.exceptions import (
    CALLER_PRINCIPAL_ABSENT_REASON,
    CALLER_PRINCIPAL_CLAIMED_REASON,
    CALLER_PRINCIPAL_FOREIGN_REASON,
    CURRENT_VERSION_REASON,
    EPOCH_MISMATCH_REASON,
    FUTURE_VERSION_REASON,
    NOT_RETAINED_REASON,
    OCC_CALLER_TRANSIENT_REASON,
    PIN_STATES,
    RESTORE_MEMBER_OUTCOMES,
    RESTORE_STATUSES,
    RETENTION_OFF_REASON,
    SESSION_ARTIFACT_NOT_IN_CUT_REASON,
    SESSION_CAP_EXCEEDED_REASON,
    SESSION_INVALIDATED_REASON,
    SESSION_NOT_FOUND_REASON,
    SESSION_READ_SET_TOO_LARGE_REASON,
    UNKNOWN_ARTIFACT_REASON,
    WORKSPACE_REGISTRATION_COMMITTED,
    WORKSPACE_REGISTRATION_EMPTY,
    WORKSPACE_REGISTRATION_REFUSED,
    CallerPrincipalRefused,
    CheckpointUnknown,
    CoherenceError,
    OccCallerTransientError,
    SessionInvalidated,
    StaleReadGeneration,
)
from ccs.core.hashing import compute_content_hash
from ccs.core.invariants import check_monotonic_version, check_single_writer
from ccs.core.states import MESIState, TransientState
from ccs.core.substrate import RestoreTier
from ccs.core.types import (
    Artifact,
    CasCorruption,
    CommitAllEntry,
    ConflictDetail,
    DataPlaneDeferredRead,
    EffectFired,
    EffectHeld,
    FetchRequest,
    FetchResponse,
    InvalidationSignal,
    MultiCommitConflict,
    MultiCommitResult,
    SessionCommitRejection,
    SessionReadRejection,
    SnapshotSession,
    VersionedContent,
    VersionedReadRejection,
    WorkspaceRegistrationResult,
    WorkspaceRestoreWrite,
)

from .registry_protocol import (
    CheckpointMember,
    CheckpointRecord,
    RegistryBase,
    SqliteExtended,
)

# The write-claim states, as one membership set (mirrors the registries' _M_OR_E_STATES).
_M_OR_E_STATES: frozenset[MESIState] = frozenset({MESIState.MODIFIED, MESIState.EXCLUSIVE})

logger = logging.getLogger(__name__)


# v0.9.0 transitional first-use warning — see
# docs/plans/2026-05-28-001-feat-c-flip-crash-recovery-default-on-plan.md.
# The v0.8.3 deprecation cycle (falsy sentinel + DeprecationWarning) is gone
# now that the default has flipped to ``enabled=True``. A single transitional
# ``RuntimeWarning`` fires once per process on the first ``CrashRecoveryConfig``
# construction so jump-upgraders (v0.8.2 -> v0.9.0) who skipped the v0.8.3
# cycle still get a migration heads-up. Removed entirely in v0.10.0.
#
# Test-isolation contract: ``_V090_FIRST_USE_WARNED`` is module-level mutable
# state that persists across instances and pytest test functions in the same
# process. Tests that assert on the warning MUST reset it to ``False`` before
# construction (via ``service._V090_FIRST_USE_WARNED = False`` or the
# ``reset_v090_first_use_flag`` fixture in ``tests/test_coordinator.py``).
#
# ``_V090_FIRST_USE_LOCK`` makes the check-then-set atomic so two threads
# constructing configs concurrently cannot both emit. The GIL made the naive
# check incidentally safe; free-threaded Python 3.13+ removes it (mirrors the
# v0.8.3 ADV-02 lock). The warning is emitted OUTSIDE the lock because warning
# filters can run arbitrary user code.
_V090_FIRST_USE_WARNED: bool = False
_V090_FIRST_USE_LOCK = threading.Lock()

_V090_FIRST_USE_MESSAGE = (
    "CrashRecoveryConfig default changed in v0.9.0: enabled=True is now the "
    "default (was False in v0.8.x), so crash recovery runs by default. Pass "
    "CrashRecoveryConfig(enabled=True) to keep the new behavior or "
    "CrashRecoveryConfig(enabled=False) to opt out. This notice fires once per "
    "process on the first construction regardless of the value passed; to "
    "suppress it, filter RuntimeWarning from the 'ccs.coordinator.service' "
    "logger. See CHANGELOG.md (section: [0.9.0]) at "
    "https://github.com/Cohexa-ai/agent-coherence/blob/main/CHANGELOG.md "
    "for migration details."
)


@dataclass(frozen=True)
class CrashRecoveryConfig:
    """Configuration knobs for the stable-grant reclamation sweep.

    The sweep ships **enabled by default** as of v0.9.0 (R10 — the default
    flipped from ``False`` to ``True``, so bare ``CrashRecoveryConfig()`` now
    activates crash recovery). A one-shot transitional ``RuntimeWarning`` fires
    on the first construction per process to flag the change for jump-upgraders
    who skipped the v0.8.3 deprecation cycle. Pass
    ``CrashRecoveryConfig(enabled=False)`` to restore v0.8.x behavior.

    Attributes:
        enabled: Master flag. When ``True`` (v0.9.0 default), the sweep
            reclaims stale M∪E grants. Pass ``enabled=False`` to opt out.
        heartbeat_timeout_ticks: Sweep reclaims any M∪E grant whose holder
            has not heartbeated within this many ticks.
        max_hold_ticks: Sweep reclaims any M∪E grant held for at least this
            many ticks regardless of heartbeat. Must be ``>`` the longest
            inspectable strategy lease TTL when ``enabled=True`` (R11).
    """

    enabled: bool = True
    heartbeat_timeout_ticks: int = 120
    max_hold_ticks: int = 900

    def __post_init__(self) -> None:
        """Emit the one-shot v0.9.0 transitional first-use warning.

        With the v0.8.3 sentinel removed, bare ``CrashRecoveryConfig()`` and
        explicit ``CrashRecoveryConfig(enabled=True)`` are indistinguishable,
        so the warning fires once per process on the FIRST construction
        regardless of how ``enabled`` was supplied — catching jump-upgraders
        (v0.8.2 -> v0.9.0) who skipped the v0.8.3 cycle. Removed in v0.10.0.
        """
        global _V090_FIRST_USE_WARNED
        # Claim the one-shot emission atomically under the lock, then emit
        # OUTSIDE it: warning filters can run arbitrary user code, and we never
        # hold the lock across that (mirrors the v0.8.3 ADV-02 discipline).
        should_emit = False
        with _V090_FIRST_USE_LOCK:
            if not _V090_FIRST_USE_WARNED:
                _V090_FIRST_USE_WARNED = True
                should_emit = True
        if should_emit:
            # stacklevel=3: warn() -> __post_init__ -> __init__ -> caller.
            warnings.warn(_V090_FIRST_USE_MESSAGE, RuntimeWarning, stacklevel=3)


def validate_crash_recovery_config(
    crash_recovery: CrashRecoveryConfig,
    strategy: object,
) -> None:
    """Fail-fast composition check (R11).

    When the sweep is enabled, ``max_hold_ticks`` must exceed the strategy's
    inspectable lease TTL strictly. Equal is rejected because a sweep at the
    TTL boundary races the strategy's own refresh logic.

    Strategies without an introspectable ``ttl_ticks`` attribute (lazy,
    eager, access-count, broadcast) cannot be statically validated against
    the rule; we emit a ``RuntimeWarning`` so a custom strategy with a
    non-inspectable TTL is at least surfaced, but do not refuse startup.
    """
    if not crash_recovery.enabled:
        return

    ttl = getattr(strategy, "ttl_ticks", None)

    # Built-in non-lease strategies (lazy, eager, access_count, broadcast) and
    # any custom strategy without a ttl_ticks attribute cannot be statically
    # validated against R11. Silent-accept matches the spec's design choice
    # for the common case.
    if ttl is None:
        return

    # Integer ttl (including 0 and negatives): apply the rule. Review fix
    # ADV-02 / ADV-03: previously this branch required ttl > 0, which silently
    # dropped ttl=0 into the "non-integer" warning path with misleading text.
    # Now any int ttl is checked, and the warn-on-non-integer branch below
    # only fires for genuinely non-integer attributes (string, float, etc.).
    if isinstance(ttl, int):
        if crash_recovery.max_hold_ticks <= ttl:
            raise ValueError(
                f"crash_recovery composition violation: "
                f"max_hold_ticks={crash_recovery.max_hold_ticks} must be > "
                f"strategy.ttl_ticks={ttl} "
                f"(strategy={type(strategy).__name__}); "
                f"sweep at lease TTL boundary races strategy refresh."
            )
        return

    warnings.warn(
        f"crash_recovery: strategy {type(strategy).__name__} exposes a "
        f"non-integer ttl_ticks={ttl!r}; composition rule (R11) cannot be "
        f"statically verified.",
        RuntimeWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class SessionCapsConfig:
    """Resource bounds for snapshot sessions (SB-17 / TX-1, Unit 7 / R14).

    Security-calibrated DEFAULTS — placeholder-but-sane, NOT post-hoc tuning.
    Each cap bounds a distinct snapshot-session DoS surface from the plan's
    threat model; the defaults are the spec's stated values (R14). A frozen
    dataclass mirroring :class:`CrashRecoveryConfig`'s shape so a deployment can
    tighten the bounds (or a test set tiny ones) without touching the sweep.

    Attributes:
        max_sessions: Maximum CONCURRENT live sessions. ``begin_session`` rejects
            opening an (N+1)th session with ``session_cap_exceeded`` (no token
            minted, no cut pinned). Bounds the many-sessions GC-starvation DoS
            (an attacker opening unboundedly many sessions to hold versions back
            from GC; threat-model #2). Default 256.
        max_read_set_cardinality: Maximum artifacts in a single ``read_set``.
            ``begin_session`` rejects a larger read-set with ``read_set_too_large``
            (no token minted, no cut pinned). Bounds the enormous-single-read-set
            GC-starvation DoS (threat-model #2). Default 64.
        absolute_age_ticks: HARD age ceiling, in logical ticks, SEPARATE from the
            heartbeat lease. A session older than this (``current_tick -
            created_at_tick >= absolute_age_ticks``) is reaped by
            :meth:`CoordinatorService.enforce_session_liveness` EVEN WHEN ITS
            HEARTBEAT IS LIVE — a live heartbeat must NOT exempt it. Bounds the
            heartbeat-spoofing-past-the-ceiling DoS (an attacker keeping a stale
            cut alive indefinitely via heartbeats; threat-model #3). The plan's
            ``absolute_age_seconds=3600`` expressed in the coordinator's logical
            tick unit (the runtime is tick-driven, no wall clock); default 3600.
    """

    max_sessions: int = 256
    max_read_set_cardinality: int = 64
    absolute_age_ticks: int = 3600

    def __post_init__(self) -> None:
        # Fail fast on a nonsensical cap (caller misuse), mirroring the
        # house "validate at the boundary" rule. A cap < 1 would reject every
        # session / read-set or never reap — both are configuration bugs.
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        if self.max_read_set_cardinality < 1:
            raise ValueError("max_read_set_cardinality must be >= 1")
        if self.absolute_age_ticks < 1:
            raise ValueError("absolute_age_ticks must be >= 1")


# Snapshot session token entropy (SB-17 / TX-1, Unit 2 / R13). 32 bytes →
# ~43 url-safe chars; unguessable, server-minted, never client-supplied. The
# timing-safe compare + per-call validation that consume it are Unit 7.
_SESSION_TOKEN_BYTES = 32

# Placeholder artifact id for a ``begin_session`` cap rejection (Unit 7 / R14):
# a cap rejection (``session_cap_exceeded`` / ``read_set_too_large``) is about the
# WHOLE call, not any single artifact, and must leak NO member id. The
# ``VersionedReadRejection`` carrier requires an ``artifact_id`` field, so the
# nil UUID stands in — the consumer matches on ``reason``, never this field.
_NIL_UUID = UUID(int=0)


# Server-minted session-token SHAPE (SB-17 / TX-1, Unit 5 / R4). A
# ``secrets.token_urlsafe(32)`` is ALWAYS exactly 43 characters drawn from the
# URL-safe base64 alphabet ([A-Za-z0-9_-], no padding). The session-liveness
# fail-closed taxonomy uses this as the structural discriminator: a token of
# this shape that has NO live cut was, or still looks like, a real session
# (reaped / GC-raced / restart-wiped) → ``session_invalidated`` (fail closed,
# never live HEAD). A token NOT of this shape is genuinely never-opened /
# malformed → ``session_not_found``. The check is structural only (it cannot
# prove a token was ever minted — fail-closed is the safe default for an
# in-shape token), and it is NOT an authentication boundary (the unguessable
# entropy + the Unit-7 owner-binding are). It exists ONLY to make the dead-vs-
# never-opened reason split honest and survive an in-memory restart that wipes
# every service-layer map including the tombstone.
_SESSION_TOKEN_LEN = 43
_SESSION_TOKEN_ALPHABET = frozenset(string.ascii_letters + string.digits + "-_")

# Bounded reaped-session tombstone cap (SB-17 / TX-1, Unit 5 / R4). The
# session-liveness sweep records a reaped token here so a subsequent
# ``session_read`` / ``session_commit`` attributes it precisely to
# ``session_invalidated`` ("your session died") rather than relying on the shape
# predicate alone. Capped + oldest-evicted so a long-lived coordinator cannot
# accumulate unbounded tombstones (a reaper-amplification DoS). Eviction is
# benign: an evicted token still classifies ``session_invalidated`` via the
# shape predicate (every server-minted token is in-shape), so dropping the
# tombstone entry only loses the precise "definitely reaped" attribution, never
# the fail-closed guarantee.
_REAPED_TOMBSTONE_CAP = 4096


def looks_like_session_token(token: str) -> bool:
    """Structural predicate: does ``token`` have the server-minted SHAPE?

    ``True`` iff it is exactly :data:`_SESSION_TOKEN_LEN` characters, all in the
    URL-safe base64 alphabet — the shape every ``begin_session`` token has. Used
    ONLY by the Unit-5 fail-closed taxonomy to split a dead/restart-wiped session
    (in-shape, no cut → ``session_invalidated``) from a genuinely-never-opened or
    malformed token (out-of-shape → ``session_not_found``). NOT an auth check; a
    well-formed token that was never minted still classifies invalidated, which
    is the safe (fail-closed) default — it is never served live HEAD either way.
    """
    return (
        len(token) == _SESSION_TOKEN_LEN
        and all(ch in _SESSION_TOKEN_ALPHABET for ch in token)
    )


# Caller-principal entropy (caller-principal plan, U4): same draw as a session
# token — 32 bytes, 43 url-safe characters. Minted here, never client-supplied.
_CALLER_PRINCIPAL_BYTES = 32

# Mint-nonce shape bounds (KTD11). The nonce is CLIENT-generated and persisted by
# the claimant before it calls the mint; the floor keeps a buggy client's "1"
# from standing in for a retry proof, the ceiling bounds what the store holds.
_MINT_NONCE_MIN_LEN = 16
_MINT_NONCE_MAX_LEN = 128


def mint_nonce_problem(mint_nonce: object) -> str | None:
    """Why ``mint_nonce`` cannot key a caller-principal claim, or ``None``.

    A usable nonce is a string of :data:`_MINT_NONCE_MIN_LEN` to
    :data:`_MINT_NONCE_MAX_LEN` url-safe base64 characters — the shape
    ``secrets.token_urlsafe`` produces. A claim with no usable nonce is refused
    outright, even for an unclaimed identity: a binding made without one could
    never be re-established after a lost mint response (R20)."""
    if not isinstance(mint_nonce, str):
        return "mint_nonce is required (a client-generated url-safe string)"
    if not _MINT_NONCE_MIN_LEN <= len(mint_nonce) <= _MINT_NONCE_MAX_LEN:
        return (
            f"mint_nonce must be {_MINT_NONCE_MIN_LEN}-{_MINT_NONCE_MAX_LEN} "
            f"characters long"
        )
    if not all(ch in _SESSION_TOKEN_ALPHABET for ch in mint_nonce):
        return "mint_nonce must use only url-safe base64 characters [A-Za-z0-9_-]"
    return None


def _same_secret(expected: str, presented: str) -> bool:
    """Timing-safe string equality (:func:`hmac.compare_digest` over UTF-8,
    so a non-ASCII presented value compares instead of raising)."""
    return hmac.compare_digest(expected.encode("utf-8"), presented.encode("utf-8"))


class CallerPrincipalUncached(Exception):
    """A ``cached_only`` caller-principal lookup that the service's in-process
    cache cannot answer.

    Not a refusal, and never on the wire: it tells the caller that the answer
    needs a read of the registry's durable store, so the coordinator's
    admission gate can make that read under its handler watchdog instead of on
    the request thread (caller-principal plan, U6). Repeating the same call
    without ``cached_only`` resolves it. Carries no principal."""


class SessionView:
    """A read-only view of a live snapshot session's PINNED cut, handed to the
    ``effect_gate`` ``decide`` callback (SB-17 / TX-1, Unit 6 / EO-5).

    The decision step reads the consistent cut — NOT live HEAD — through this thin
    wrapper over :meth:`CoordinatorService.session_read`, so the caller's decision
    is computed against exactly the pinned versions the gate will re-validate. It
    is read-only by construction (it exposes no write/commit), and it FAILS CLOSED:
    a ``session_read`` that returns a dead-session rejection
    (``session_invalidated`` / ``session_not_found``) raises
    :class:`SessionInvalidated` rather than letting the decision proceed on an
    invalid cut. A ``DataPlaneDeferredRead`` (the eager / ``content=None`` branch,
    where the coordinator holds no body) is returned as-is — the caller fetches the
    pinned bytes from the data plane; the gate still re-validates by VERSION, which
    is byte-source-independent.
    """

    def __init__(
        self,
        service: "CoordinatorService",
        session_token: str,
        owner: UUID,
    ) -> None:
        self._service = service
        self._token = session_token
        # The session's OWNER — threaded through so the view's pinned reads pass
        # the per-call owner validation (Unit 7 / R13). The view reads the
        # session's OWN cut, so the caller IS the owner by construction.
        self._owner = owner

    def read(
        self, artifact_id: UUID
    ) -> VersionedContent | DataPlaneDeferredRead:
        """Read ``artifact_id`` at its pinned version. Fails closed on a dead
        session or an un-pinned artifact (a typed rejection → raised
        :class:`SessionInvalidated` / ``CoherenceError``) so a decision never runs
        on an invalid cut."""
        result = self._service.session_read(
            self._token, artifact_id, caller=self._owner
        )
        if isinstance(result, SessionReadRejection):
            if result.reason == SESSION_ARTIFACT_NOT_IN_CUT_REASON:
                # Caller misuse: reading outside the pinned read-set. Not a dead
                # session — a hard error, never a live-HEAD fall-through.
                raise CoherenceError(
                    f"effect_gate decide read an un-pinned artifact "
                    f"{artifact_id}: {result.reason}"
                )
            # session_invalidated / session_not_found → fail closed.
            raise SessionInvalidated(
                f"effect_gate decide read failed closed: {result.reason} "
                f"(artifact={artifact_id})"
            )
        return result


class CoordinatorService:
    """Control-plane service for artifact read/write/commit synchronization."""

    def __init__(
        self,
        registry: RegistryBase,
        *,
        session_caps: SessionCapsConfig | None = None,
    ):
        self.registry = registry
        # Snapshot session resource bounds (SB-17 / TX-1, Unit 7 / R14). Defaults
        # to the security-calibrated :class:`SessionCapsConfig` (max_sessions,
        # max_read_set_cardinality, absolute_age_ticks); a deployment or test may
        # pass tighter caps. Constructor param so the bounds are configurable
        # without touching the sweep (mirrors how ``crash_recovery`` knobs are
        # threaded into ``enforce_stable_grant_timeouts``).
        self.session_caps = session_caps or SessionCapsConfig()
        # Snapshot session owner-binding (SB-17 / TX-1, Unit 2 / R13):
        # ``{session_token: owner_id}``. Populated at ``begin_session`` mint; the
        # per-CALL read/commit validation that READS it (a foreign owner →
        # SessionInvalidated, timing-safe compare) is in
        # :meth:`_validate_session_owner`; :meth:`record_session_heartbeat`
        # owner-binds the heartbeat path. This in-memory map is the FAST tier;
        # ``begin_session`` ALSO persists the owner durably (the registry's
        # ``session_meta``). A restart wipes this map but the durable owner
        # survives, so :meth:`_validate_session_owner` falls back to it: a survived
        # sqlite session is still owner-validated (R6 + R13 both hold across a
        # restart — finding F3), while an in-memory-registry session has no durable
        # owner and so fails closed post-restart (the asserted divergence).
        #
        # Free-threading discipline (no-GIL-reliance, finding F4): the snapshot
        # session maps below (``_session_owners`` / ``_session_created`` /
        # ``_session_heartbeats`` / ``_reaped_tombstone``) are read AND mutated
        # across concurrent HTTP handler threads (begin / read / commit /
        # heartbeat) plus the sweep. The begin_session cap-check + insert is a
        # compound check-then-act that a bare GIL does not make atomic (two
        # concurrent begins could both pass the cap and overshoot it), and the
        # sweep iterates while handlers mutate. This RLock serializes EVERY
        # session-map access so correctness never depends on the GIL. Re-entrant
        # because the sweep holds it across ``_reap_session`` (which re-takes it
        # via ``_drop_session_state``). Lock ORDER is always service-lock →
        # registry-lock (the registry never calls back into the service), so the
        # ``with self._session_lock:`` blocks that wrap ``registry`` calls
        # (capture / release / get_session_meta) cannot deadlock.
        self._session_lock = threading.RLock()
        self._session_owners: dict[str, UUID] = {}
        # Snapshot session CREATION tick (SB-17 / TX-1, Unit 7 / R14):
        # ``{session_token: created_at_tick}``. The absolute-age ceiling reads
        # THIS — NOT the heartbeat lease — so a session older than
        # ``session_caps.absolute_age_ticks`` is reaped EVEN with a fresh
        # heartbeat (a live heartbeat must not exempt the hard ceiling,
        # threat-model #3). Seeded at ``begin_session`` alongside the lease;
        # dropped on reap/release with the owner + lease.
        self._session_created: dict[str, int] = {}
        # Session heartbeat lease (SB-17 / TX-1, Unit 5 / R4):
        # ``{session_token: last_heartbeat_tick}``. Keyed by the server-minted
        # SESSION TOKEN, NOT the MESI ``agent_id`` — a snapshot session is NOT a
        # MESI agent and holds no grant, so the grant sweep (which walks M∪E
        # holders) can never see it. The session-liveness sweep is a NEW axis
        # over THIS map. Seeded at ``begin_session`` with the creation tick so a
        # never-yet-heartbeated session still carries a lease baseline (mirrors a
        # grant's ``granted_at_tick``), rather than being reaped on the first
        # sweep. Service-scoped: an in-memory restart drops the lease (and the
        # pins), so a post-restart token has no cut and fails closed.
        self._session_heartbeats: dict[str, int] = {}
        # Bounded reaped-session tombstone (Unit 5 / R4): recently-reaped tokens,
        # capped + oldest-evicted (FIFO). Lets ``session_read`` / ``session_commit``
        # attribute a reaped token precisely to ``session_invalidated``. Eviction
        # is benign — an evicted in-shape token still classifies invalidated via
        # ``looks_like_session_token`` (the shape predicate), so the fail-closed
        # guarantee never depends on the tombstone, only the precise attribution.
        self._reaped_tombstone: "OrderedDict[str, None]" = OrderedDict()
        # Workspace-registration first-observation mint (WV Unit 5): the
        # IN-MEMORY registry has no ``resolve_or_register``, so the fallback in
        # :meth:`_resolve_workspace_member_artifact` is a name scan followed by
        # a mint — a compound check-then-act with no UNIQUE constraint behind
        # it. This lock serializes that scan+mint so two concurrent first
        # observations of one member_path mint ONE artifact (parity with the
        # sqlite path's atomic ``resolve_or_register``; free-threading
        # discipline F4 — correctness never rides the GIL). Lock order stays
        # service-lock → registry-lock (the registry never calls back into the
        # service), and this lock is never held together with
        # ``_session_lock``.
        self._workspace_mint_lock = threading.Lock()
        # Caller-principal FAST tier (caller-principal plan, U4 and U6), in front
        # of the registry's ``caller_principals`` store (the durable tier):
        # ``{identity: principal}`` for a BOUND identity, and ``{identity: None}``
        # for one the durable store answered UNBOUND — the negative tier, which
        # is what lets a client that never claims (an older one, KTD15) be
        # admitted by the require-class routes without a registry read on every
        # request. Both are safe to keep because a binding is never rebound and
        # this service is the only writer of bindings in a coordinator process
        # (it binds only in :meth:`claim_caller_principal`): a bound entry never
        # goes stale, and an UNBOUND one goes stale only when this service binds
        # the identity — which replaces it (see :meth:`_record_bind_attempt`).
        # A second writer of bindings into the same store would not be seen
        # here; there is none. ``_caller_principal_binds`` counts finished bind
        # attempts, so a lookup whose durable read raced one never caches the
        # UNBOUND answer it read before that bind. Deliberately separate from
        # every session map above and from ``_session_lock``: a principal's
        # lifetime is independent of any snapshot session (R4). Its own lock,
        # never held across a registry call (lock order unchanged: service →
        # registry).
        self._caller_principal_lock = threading.Lock()
        self._caller_principals: dict[UUID, str | None] = {}
        self._caller_principal_binds = 0

    def register_artifact(
        self,
        *,
        name: str,
        content: str,
        initial_owner: UUID | None = None,
        size_tokens: int | None = None,
        content_hash: str | None = None,
        depends_on: tuple[UUID, ...] = (),
    ) -> Artifact:
        """Register a new artifact and optionally assign an initial owner.

        The register and the initial EXCLUSIVE grant run under one
        :meth:`registry.abort_guard` hold (U5): without it a concurrent sweep
        or peer fetch can interleave between them and act on an artifact that
        exists but whose owner grant has not landed yet."""
        artifact = Artifact(
            name=name,
            version=1,
            content_hash=content_hash,
            size_tokens=size_tokens,
            depends_on=depends_on,
        )
        with self.registry.abort_guard():
            self.registry.register_artifact(artifact, content)
            if initial_owner is not None:
                self.registry.set_agent_state(
                    artifact.id, initial_owner, MESIState.EXCLUSIVE, trigger="register", tick=0
                )
        return artifact

    def fetch(self, request: FetchRequest) -> FetchResponse:
        """Fetch canonical artifact payload and grant requester state.

        The whole read-decide-grant sequence runs under one
        :meth:`registry.abort_guard` hold (U5): the grant decision (EXCLUSIVE
        vs SHARED) is made from the state-map snapshot, so a peer write
        landing between that read and the grant write could otherwise leave
        two holders in a write state — the check-then-act shape every other
        mutation path already guards."""
        with self.registry.abort_guard():
            artifact = self._require_artifact(request.artifact_id)
            content = self.registry.get_content(request.artifact_id)
            if content is None:
                raise CoherenceError(f"artifact_content_missing artifact={request.artifact_id}")

            state_map = self.registry.get_state_map(request.artifact_id)
            other_holders = [
                agent_id
                for agent_id, state in state_map.items()
                if agent_id != request.requesting_agent_id and state != MESIState.INVALID
            ]

            grant = MESIState.EXCLUSIVE if not other_holders else MESIState.SHARED
            self.registry.set_agent_transient(
                request.artifact_id,
                request.requesting_agent_id,
                TransientState.IED if grant == MESIState.EXCLUSIVE else TransientState.ISG,
                entered_tick=request.requested_at_tick,
            )
            if other_holders:
                # Multiple readers must stay coherent: downgrade the M/E holder, if
                # any. A peer already in SHARED is left alone — rewriting it would
                # re-capture its read_generation on a read it never made and
                # re-stamp an observation it never had ("fetch" is the capture
                # trigger, and set_agent_state records both on every non-INVALID
                # write). other_holders itself stays wide: the EXCLUSIVE-vs-SHARED
                # grant above must still see every non-INVALID peer.
                for agent_id in other_holders:
                    if state_map[agent_id] not in _M_OR_E_STATES:
                        continue
                    self.registry.set_agent_state(
                        request.artifact_id, agent_id, MESIState.SHARED, trigger="fetch", tick=request.requested_at_tick
                    )

            self.registry.set_agent_state(
                request.artifact_id, request.requesting_agent_id, grant, trigger="fetch", tick=request.requested_at_tick
            )
            self.registry.clear_agent_transient(request.artifact_id, request.requesting_agent_id)
            self._validate_single_writer(request.artifact_id)

        return FetchResponse(
            artifact_id=request.artifact_id,
            version=artifact.version,
            content=content,
            state_grant=grant,
        )

    def read_at_version(
        self,
        artifact_id: UUID,
        version: int,
        expected_epoch: str | None = None,
    ) -> VersionedContent | VersionedReadRejection:
        """Read a specific RETAINED version off-protocol (plan item N v1 / R5–R7).

        The first-class read-at-version surface: returns the body the registry
        committed at ``version`` as a :class:`~ccs.core.types.VersionedContent`,
        or a typed :class:`~ccs.core.types.VersionedReadRejection` carrying one of
        the six wire-stable reasons in :data:`ccs.core.exceptions.READ_AT_VERSION_REASONS`.
        It is a typed RETURN, never an exception (the ``ConflictDetail``
        discipline) — except ``version < 1``, which is caller misuse and raises
        ``ValueError`` (house style; not a wire reason).

        **Protocol non-interaction by construction (R6, R7):** this method calls
        NONE of ``set_agent_state`` / ``set_agent_transient`` / grant / transient
        / invalidation code. Read-generation capture lives ONLY inside
        ``set_agent_state`` (``registry.py`` ``CLAIM_CAPTURE_TRIGGERS``), so not
        calling it means the read CANNOT touch ``read_generation`` (R6 fence
        non-capture) or any MESI state / invalidation membership (R7). The whole
        point is verifiable by reading this body: it only ever READS the registry
        (``get_artifact`` / ``retention_meta`` / ``coordinator_epoch`` /
        ``get_version_record``) and constructs frozen return values.

        Discrimination order (first match wins), computed against ONE snapshot of
        the current version so a racing commit cannot mislabel a reason:

        1. ``version < 1`` → ``ValueError`` (caller misuse).
        2. artifact unknown → ``unknown_artifact`` (``current_version=None``).
        3. retention not enabled for the store → ``retention_off``.
        4. ``expected_epoch`` supplied and != the store epoch → ``epoch_mismatch``.
        5. ``version == current`` → ``current_version`` (history surface serves
           history only; current bytes are read via the protocol fetch path).
        6. ``version > current`` → ``future_version`` (hints at a 2nd coordinator).
        7. ``1 <= version < current`` → fetch the row. Present AND not T-expired →
           ``VersionedContent``; absent OR T-expired → ``not_retained``.

        Single-scope atomicity: ``current`` is read ONCE (the linearization
        point) and every reason is decided relative to that snapshot. History
        rows below ``current`` are immutable (a version is captured once and only
        ever DROPPED by GC, never rewritten), so the step-7 row fetch is safe
        against a commit that races in after the ``current`` read — that commit
        only captures the NEW (higher) version and never touches the requested
        ``version < current`` row. The worst a race yields is the old-current
        served as history (correct bytes) or ``current_version`` (the value that
        WAS current at the read point) — never wrong bytes or a mislabeled reason.

        T-expiry is LOGICAL at read (R-fix: a read is non-mutating, so an
        age-collectible row is reported ``not_retained`` but NOT physically
        deleted here — physical deletion piggybacks on the next capture). It
        reuses the one GC seam :func:`collectible_versions` against the persisted
        policy so the read-side expiry rule matches the write-side eviction rule.

        Args:
            artifact_id: The artifact to read.
            version: The 1-based version to read (``< 1`` raises ``ValueError``).
            expected_epoch: If given, the read rejects ``epoch_mismatch`` unless
                it equals the store's ``coordinator_epoch`` (the store was reset
                since the caller captured the epoch).

        Returns:
            :class:`VersionedContent` on a hit, else :class:`VersionedReadRejection`.

        Raises:
            ValueError: ``version < 1`` (caller misuse — not a wire reason).
        """
        if version < 1:
            raise ValueError(
                f"read_at_version: version must be >= 1 (got {version}); "
                f"versions are 1-based. A sub-1 version is caller misuse, not a "
                f"retained-history miss (which is the not_retained reason)."
            )

        epoch = self.registry.coordinator_epoch

        # (2) Unknown artifact — no current version exists for it. Read the
        # artifact ONCE; ``current`` from this same metadata object is the
        # linearization snapshot every reason below is decided against.
        artifact = self.registry.get_artifact(artifact_id)
        if artifact is None:
            return VersionedReadRejection(
                reason=UNKNOWN_ARTIFACT_REASON,
                artifact_id=artifact_id,
                requested_version=version,
                current_version=None,
                coordinator_epoch=epoch,
            )
        current = artifact.version

        # (3) Retention never enabled for this store (store-derived: persisted
        # meta on sqlite, live ctor state in-memory). The artifact_versions
        # surface exists on every v2 db, so this marker — not table presence —
        # distinguishes retention-off from a mere history gap.
        retention_enabled, policy = self.registry.retention_meta()
        if not retention_enabled:
            return self._reject(
                RETENTION_OFF_REASON, artifact_id, version, current, epoch
            )

        # (4) Epoch guard — a stale expected_epoch means the store was reset
        # (delete-and-recreate) since the caller captured it, so its retained
        # history is from a different incarnation.
        if expected_epoch is not None and expected_epoch != epoch:
            return self._reject(
                EPOCH_MISMATCH_REASON, artifact_id, version, current, epoch
            )

        # (5) Current version — history surface serves HISTORY ONLY. Current
        # content is read via the protocol fetch path (artifacts store hashes,
        # not bodies), never here, by design.
        if version == current:
            return self._reject(
                CURRENT_VERSION_REASON, artifact_id, version, current, epoch
            )

        # (6) Future version — above current suggests a second coordinator
        # writing the same store (the diagnostic commit_cas keeps via CasCorruption).
        if version > current:
            return self._reject(
                FUTURE_VERSION_REASON, artifact_id, version, current, epoch
            )

        # (7) 1 <= version < current: a genuine history request. Fetch body +
        # capture timestamp in ONE scoped accessor (single SELECT/sqlite, GIL-
        # atomic pair/in-memory). Absent ⇒ not_retained (never captured, K-
        # evicted, or already T-swept).
        record = self.registry.get_version_record(artifact_id, version)
        if record is None:
            return self._reject(
                NOT_RETAINED_REASON, artifact_id, version, current, epoch
            )
        content, captured_at = record

        # Logical T-expiry (R-fix: non-mutating read). Reuse the single GC seam
        # against the persisted policy: an age-collectible row reports
        # not_retained without being physically deleted (deletion piggybacks on
        # the next capture). ``current`` is exempt in collectible_versions, but
        # we already excluded version==current above, so this only ages the
        # requested historical row. ``policy is None`` ⇒ unbounded ⇒ no T axis.
        #
        # Unit 3 (DONE): the read-serve allowance — a live-session pin suppresses
        # this read-side logical T-expiry so a pinned-but-age-collectible row is
        # still SERVED — lives in ``session_read`` (the session-scoped path), NOT
        # here. The bare ``read_at_version`` deliberately keeps ages-out semantics
        # (no live session, no allowance); ``session_read`` passes the pinned
        # version as its OWN ``exemptions`` to this same seam. (The GC-HOLD
        # exemption — distinct — was wired by Unit 2 at the GC producers.)
        if policy is not None and version in collectible_versions(
            {version: captured_at},
            current_version=current,
            policy=policy,
            now=time.time(),
        ):
            return self._reject(
                NOT_RETAINED_REASON, artifact_id, version, current, epoch
            )

        return VersionedContent(
            artifact_id=artifact_id,
            version=version,
            content=content,
            captured_at=captured_at,
            coordinator_epoch=epoch,
        )

    @staticmethod
    def _reject(
        reason: str,
        artifact_id: UUID,
        requested_version: int,
        current_version: int | None,
        epoch: str,
    ) -> VersionedReadRejection:
        """Build a :class:`VersionedReadRejection` (no body material, by type)."""
        return VersionedReadRejection(
            reason=reason,
            artifact_id=artifact_id,
            requested_version=requested_version,
            current_version=current_version,
            coordinator_epoch=epoch,
        )

    def begin_session(
        self,
        *,
        read_set: Iterable[UUID],
        owner: UUID,
        created_at_tick: int = 0,
    ) -> SnapshotSession | VersionedReadRejection:
        """Open a consistent multi-artifact snapshot session (SB-17 / TX-1,
        Unit 2 / R1, R13). Pins a coherent CUT of ``read_set`` at one
        linearization point and returns an inspectable
        :class:`~ccs.core.types.SnapshotSession`.

        Orchestration (Unit 2 scope):

        1. Mint a server-minted ``session_token`` (``secrets.token_urlsafe`` —
           unguessable, never client-supplied; R9/R13) and bind it to ``owner``
           (the creating caller's MESI agent/process identity) in
           ``_session_owners``. (Unit 2 mints + owner-binds AT CREATION ONLY;
           per-call token validation, timing-safe compare, caps, the
           heartbeat-lease, and the absolute-age ceiling are LATER units 5/7.)
        2. Call the registry's atomic ``capture_version_vector`` — the cut is
           captured and pinned in one linearization point, non-mutating (no MESI
           grant, no ``read_generation``). An unknown id in ``read_set`` returns
           a typed :class:`VersionedReadRejection` (``unknown_artifact``) with NO
           pins inserted; the token binding is then dropped (no half-open session
           for a rejected cut).
        3. Read ``retain_versions`` from the store (the deployment-branch
           indicator) and return the :class:`SnapshotSession` with the
           INSPECTABLE cut (R11).

        Byte handling is explicitly NOT here (Unit 3): this captures the
        version-MAP only and records ``retain_versions``; the eager-vs-lazy serve
        is resolved at the serve layer. For ``content=None`` / cross-process the
        bytes live in the data plane, not the coordinator.

        Args:
            read_set: The artifact ids to pin into the consistent cut.
            owner: The creating caller's identity (the MESI agent/process label),
                bound to the minted token for R13 owner-binding.

        Returns:
            A :class:`SnapshotSession` on success, else a
            :class:`VersionedReadRejection` (``unknown_artifact``) — no session
            opened, no pins held.
        """
        # Resource caps (Unit 7 / R14) — enforced BEFORE any token mint or pin
        # insert, so a rejected ``begin_session`` leaves NO half-open session
        # (no owner binding, no lease, no pins). Typed RETURNS (the
        # ``VersionedReadRejection`` carrier ``begin_session`` already produces),
        # never an exception — the caps are a bounded-blast-radius surface, not a
        # crash. ``read_set`` is materialized ONCE here (it may be a one-shot
        # iterable) so the cardinality check and the capture see the same set.
        read_set = list(read_set)
        epoch = self.registry.coordinator_epoch
        if len(read_set) > self.session_caps.max_read_set_cardinality:
            # The enormous-single-read-set GC-starvation bound (threat-model #2).
            # No artifact id is leaked (artifact_id=None-equivalent uses a nil
            # UUID); the rejection is about the CARDINALITY, not any member.
            return VersionedReadRejection(
                reason=SESSION_READ_SET_TOO_LARGE_REASON,
                artifact_id=_NIL_UUID,
                requested_version=len(read_set),
                current_version=self.session_caps.max_read_set_cardinality,
                coordinator_epoch=epoch,
            )
        # The cap-check + token mint + owner-bind + capture + seed run under
        # ``_session_lock`` as ONE critical section (finding F4): the
        # ``max_sessions`` check and the owner insert must be atomic, else two
        # concurrent begins both pass the bare-GIL check and overshoot the cap
        # (the GC-starvation bound the cap exists to enforce). Holding the lock
        # across the capture also serializes begin against the sweep — both
        # acquire ``_session_lock`` first, then the registry lock, so there is no
        # deadlock and a begin never races a reap of the same maps.
        with self._session_lock:
            # The many-sessions GC-starvation bound (threat-model #2). Count ALL
            # live sessions via the durable session_meta (one row per open session
            # — in-memory OR a durable-only restart survivor), NOT just the
            # in-memory ``_session_owners``: post-restart the in-memory map is
            # empty while durable survivors still hold pins, so counting only it
            # would let the pinned-cut count transiently reach ~2x max_sessions and
            # evade the bound (finding). An (N+1)th is rejected until a session is
            # released or the sweep reaps a stale one.
            live_sessions = self.registry.session_count()
            if live_sessions >= self.session_caps.max_sessions:
                return VersionedReadRejection(
                    reason=SESSION_CAP_EXCEEDED_REASON,
                    artifact_id=_NIL_UUID,
                    requested_version=live_sessions,
                    current_version=self.session_caps.max_sessions,
                    coordinator_epoch=epoch,
                )

            session_token = secrets.token_urlsafe(_SESSION_TOKEN_BYTES)
            # Owner-bind at mint (R13). Recorded BEFORE the capture so the Unit-7
            # per-call validator never sees a pinned-but-unowned token; dropped
            # again if the capture rejects. The INVERSE window — between this line
            # and the capture below, the token is owned but PINLESS — is handled
            # by :meth:`_validate_session_owner`, which treats an owned-but-pinless
            # token as NOT-yet-live (fail closed), never a valid empty session.
            self._session_owners[session_token] = owner
            # Pass ``owner`` + ``created_at_tick`` so the registry persists the
            # durable owner-binding (``session_meta``) ATOMICALLY with the pins
            # (R13/R6/R14): a post-restart read falls back to the durable owner —
            # the legitimate owner is still served (R6) and a leaked-token foreign
            # caller is still rejected (R13) — and the durable creation tick lets
            # the absolute-age ceiling bound a survived session (finding F3).
            result = self.registry.capture_version_vector(
                read_set, session_token, owner=owner, created_at_tick=created_at_tick
            )
            if isinstance(result, VersionedReadRejection):
                # No cut pinned (unknown id) ⇒ no session. The capture txn wrote
                # neither pins nor durable meta (atomic), so only the in-memory
                # owner binding needs dropping — the rejected token cannot linger
                # as a half-open session.
                self._session_owners.pop(session_token, None)
                return result
            # Seed the heartbeat lease at the creation tick (Unit 5 / R4): a
            # session that never heartbeats still carries a baseline so it is not
            # reaped on the very first sweep — the lease starts now, exactly like a
            # grant's ``granted_at_tick``. Recorded only on a SUCCESSFUL capture.
            self._session_heartbeats[session_token] = created_at_tick
            # Record the CREATION tick for the absolute-age ceiling (Unit 7 / R14).
            # The ceiling reads THIS, not the heartbeat lease, so a live-heartbeat
            # session past the ceiling is still reaped (threat-model #3).
            self._session_created[session_token] = created_at_tick
        retain_versions, _policy = self.registry.retention_meta()
        return SnapshotSession(
            session_token=session_token,
            cut=result,
            coordinator_epoch=self.registry.coordinator_epoch,
            retain_versions=retain_versions,
        )

    def session_read(
        self,
        session_token: str,
        artifact_id: UUID,
        *,
        caller: UUID,
    ) -> VersionedContent | DataPlaneDeferredRead | SessionReadRejection:
        """Serve an artifact's PINNED version from a live snapshot session — the
        non-mutating read from the consistent cut (SB-17 / TX-1, Unit 3 / R2).

        A NEW service path, NOT an extension of ``read_at_version`` (which is the
        bare history read with its own frozen 6-reason contract and its
        deliberate ``version == current`` REJECTION). The two surfaces have
        OPPOSITE rules for the current version — bare ``read_at_version`` rejects
        it; ``session_read`` SERVES it — and a different validation gate (a live
        pin, not raw retention state). Threading session-awareness through
        ``read_at_version`` would entangle those contracts and muddy the
        ``current_version`` rejection a pinned test pins; a separate path keeps
        each surface honest. ``begin_session`` is the coherence event that earns
        the pinned-version serve (including ``version == current``), so the
        allowance is scoped to a valid live session here.

        **Non-mutating (R2, the shipped invariant):** like ``read_at_version``,
        this calls NONE of ``set_agent_state`` / ``set_agent_transient`` / grant /
        invalidation code. It only READS the registry (``get_session_cut`` /
        ``get_artifact`` / ``retention_meta`` / ``get_version_record`` /
        ``coordinator_epoch``) and builds frozen returns, so it mints NO MESI
        grant and captures NO ``read_generation`` (read-gen capture lives ONLY in
        ``set_agent_state``). A reader is not an owner.

        Bytes source — the deployment-dependent rule resolved at the serve layer
        (KTD), keyed off the session's branch (``retain_versions``, recorded at
        ``begin_session``):

        - **LAZY (``retain_versions=True``)** — the coordinator HAS bodies in
          history. Serve the PINNED version's body from ``get_version_record``
          (the retained-history accessor): the current version's body is captured
          into history at commit, so it serves BOTH ``pinned == current`` AND
          ``pinned < current`` uniformly. The TRANSITION is automatic — once a
          peer commits past the pin, ``current`` advances but the pinned row
          persists in history, so the SAME ``get_version_record(pinned)`` keeps
          serving the pinned bytes (re-read ``current`` each call, never cache
          the branch). **Read-serve allowance (the Unit-3 obligation):** the
          pinned version is passed as its OWN ``exemptions`` to the T-expiry
          ``collectible_versions`` seam, so a pinned-but-age-collectible row is
          STILL SERVED (distinct from the GC-hold the Unit-2 exemptions seam
          already provides at the GC producers — this lifts the read-side LOGICAL
          T-expiry that ``read_at_version`` would apply). A genuinely absent body
          (``content=None`` committed even under retain=True, or a GC race)
          degrades to the data-plane-deferred result, never a crash or wrong
          bytes.
        - **EAGER (``retain_versions=False`` / ``content=None`` ICP)** — the
          coordinator holds NO body for the pinned version (bodies live in the
          CoherentVolume data plane). Return a typed
          :class:`~ccs.core.types.DataPlaneDeferredRead` carrying the pinned
          version + epoch (+ ``content_hash`` when known) — the honest "ask the
          data plane for the bytes" signal. The actual eager byte serve is
          **Unit 6 (CoherentVolume)**; this method never reads the data plane.

        Validation: the caller must be the session OWNER (Unit 7 / R13,
        timing-safe — see ``caller`` below) AND the token must have a live pin for
        ``artifact_id``. An unknown/released token → ``session_not_found``; a live
        token whose cut lacks ``artifact_id`` → ``artifact_not_in_cut`` — both
        typed :class:`~ccs.core.types.SessionReadRejection`, NEVER a live-HEAD
        fall-through. A FOREIGN caller or an OWNED-BUT-PINLESS token RAISES
        :class:`SessionInvalidated` (Unit 7, validated BEFORE the pin lookup); the
        heartbeat-liveness ``session_invalidated`` axis is Unit 5.

        Args:
            session_token: The server-minted session identity from
                ``begin_session``.
            artifact_id: The artifact to read at its pinned version.
            caller: The CALLER'S identity (the MESI agent/process label). Must be
                the session's bound owner — validated timing-safe
                (:func:`hmac.compare_digest`) against the owner bound at
                ``begin_session``; a foreign caller fails closed
                (:class:`SessionInvalidated`). Required (R13): a sibling MUST NOT
                read another's cut.

        Returns:
            :class:`VersionedContent` (coordinator-held pinned bytes),
            :class:`DataPlaneDeferredRead` (bytes live in the data plane), or a
            :class:`SessionReadRejection` (no valid pin) — all typed RETURNS,
            never an exception.
        """
        epoch = self.registry.coordinator_epoch

        # Per-call OWNER-binding validation (Unit 7 / R13) — read the cut and the
        # owner binding CONSISTENTLY (one cut read, passed to the validator), then
        # fail closed BEFORE acting on the pin: a FOREIGN caller or an
        # OWNED-BUT-PINLESS token raises :class:`SessionInvalidated` (a sibling
        # MUST NOT read another's cut; an owned-but-pinless token is not-yet-live).
        # A token with NO owner binding falls through to the cut-absent liveness
        # taxonomy below (still fail-closed, never live HEAD).
        cut = self.registry.get_session_cut(session_token)
        self._validate_session_owner(session_token, caller, cut)
        if cut is None:
            # FAIL CLOSED (Unit 5 / R4): no live cut for this token — it was
            # reaped by the session-liveness sweep, GC-raced, wiped by an
            # in-memory restart, released, or never opened. NEVER a live-HEAD
            # fall-through. ``_classify_no_cut_reason`` splits the wire-stable
            # taxonomy: ``session_invalidated`` for a reaped / restart-wiped /
            # in-shape token ("re-establish your session"), ``session_not_found``
            # for a genuinely never-opened / malformed token. Both are typed
            # rejections; the split only sharpens the signal.
            return SessionReadRejection(
                reason=self._classify_no_cut_reason(session_token),
                artifact_id=artifact_id,
                coordinator_epoch=epoch,
            )
        if artifact_id not in cut:
            # A live session, but this artifact was not pinned. Reject — NEVER
            # serve live HEAD for an un-pinned artifact (out of scope: a session
            # wanting fresh data starts a new session).
            return SessionReadRejection(
                reason=SESSION_ARTIFACT_NOT_IN_CUT_REASON,
                artifact_id=artifact_id,
                coordinator_epoch=epoch,
            )
        pinned = cut[artifact_id]

        # Read ``current`` ONCE (the per-call linearization snapshot) so the
        # branch routing and the deferred-hash hint are decided against one view.
        # The artifact may have been deleted out from under a live pin (the
        # session_pins table deliberately has NO cascade FK); that fail-closed
        # path is Unit 5 (``SessionInvalidated``). Until then a missing artifact
        # under a live pin degrades to data-plane-deferred (never wrong bytes).
        artifact = self.registry.get_artifact(artifact_id)
        current = artifact.version if artifact is not None else None
        # The pinned version's hash is knowable only when it is STILL current
        # (the artifacts table holds the current hash only); a superseded pin's
        # hash is not separately retained on the coordinator.
        pinned_hash = (
            artifact.content_hash
            if artifact is not None and current == pinned
            else None
        )

        retain_versions, policy = self.registry.retention_meta()
        if not retain_versions:
            # EAGER branch: the coordinator never retained a body for the pinned
            # version — the canonical bytes live in the data plane. Honest typed
            # deferral (pinned coordinates only, NO bytes); the data-plane serve
            # is Unit 6.
            return DataPlaneDeferredRead(
                artifact_id=artifact_id,
                version=pinned,
                content_hash=pinned_hash,
                coordinator_epoch=epoch,
            )

        # LAZY branch: serve the pinned version's body from retained history
        # (the current version's body is captured into history at commit, so
        # this serves both pinned==current and pinned<current). A genuinely
        # absent body (content=None under retain=True, or a GC race) is NOT a
        # crash — degrade to the data-plane-deferred signal.
        record = self.registry.get_version_record(artifact_id, pinned)
        if record is None:
            return DataPlaneDeferredRead(
                artifact_id=artifact_id,
                version=pinned,
                content_hash=pinned_hash,
                coordinator_epoch=epoch,
            )
        content, captured_at = record

        # Read-serve allowance (the Unit-3 obligation): the pinned version is its
        # OWN exemption to the read-side LOGICAL T-expiry, so a pinned-but-age-
        # collectible row is STILL served. Because ``pinned`` is always in
        # ``exemptions``, ``collectible_versions`` can never mark it — the call is
        # kept (rather than skipped) to make the allowance explicit and to age
        # NOTHING else here. ``policy is None`` ⇒ unbounded ⇒ no T axis. This is
        # the read-serve counterpart to the Unit-2 GC-hold ``exemptions`` seam.
        if policy is not None and current is not None:
            _served_despite_age = pinned not in collectible_versions(
                {pinned: captured_at},
                current_version=current,
                policy=policy,
                now=time.time(),
                exemptions={pinned},
            )
            # Invariant by construction: a self-exempt version is never
            # collectible. Asserting documents intent without a runtime branch.
            assert _served_despite_age, (
                "pinned version unexpectedly collectible despite self-exemption "
                "(the read-serve allowance regressed)"
            )

        return VersionedContent(
            artifact_id=artifact_id,
            version=pinned,
            content=content,
            captured_at=captured_at,
            coordinator_epoch=epoch,
        )

    def session_commit(
        self,
        session_token: str,
        artifact_id: UUID,
        content: bytes | str,
        *,
        caller: UUID,
        size_tokens: int | None = None,
        issued_at_tick: int = 0,
        abort: threading.Event | None = None,
    ) -> tuple[Artifact, list[InvalidationSignal]] | ConflictDetail | SessionCommitRejection:
        """Validate one artifact's commit against its PINNED version via the
        shipped ``commit_cas`` — the single-artifact OCC commit from a snapshot
        session (SB-17 / TX-1, Unit 4 / R3).

        The commit is arbitrated against the cut's pinned base: ``expected_version``
        is ``cut[artifact_id]`` (the version captured at ``begin_session``), so a
        commit WINS only if no peer moved the artifact since the cut was pinned.
        This reuses the shipped ``commit_cas`` arbitration VERBATIM — no
        re-implemented OCC, single-shot (NEVER the auto-rederive ``write_cas``
        loop, whose split-comparand hazard a pinned base is precisely meant to
        avoid).

        **The admit-on-absent load-bearing path (R3, the reconciled fence).** The
        commit rides a SESSION-SCOPED committer identity derived deterministically
        from the ``session_token`` (``uuid5``), NOT the owner's MESI ``agent_id``.
        The reason is the read-generation fence: ``commit_cas`` ADMITS a committer
        with NO captured ``read_generation`` (version-CAS then arbitrates) and
        REJECTS one whose PRESENT ``read_generation`` was superseded by a sweep
        reclamation. The owner's MESI agent could be carrying such a superseded
        ``read_generation`` from unrelated prior MESI activity — committing under
        it would spuriously fail with ``stale_read_generation`` on a perfectly
        healthy session. The session-derived identity has never established a fence
        claim (no ``read_generation`` row), so admit-on-absent holds and the
        pinned-base version-CAS is the sole arbiter. It is deterministic (stable
        across a session's calls) and collision-free against real agent ids (a
        ``uuid5`` over a 32-byte server-minted token namespace).

        Validation: the caller must be the session OWNER (Unit 7 / R13,
        timing-safe — see ``caller`` below) AND the token must have a live pin for
        ``artifact_id``. An unknown/released token → ``session_not_found``; a live
        token whose cut lacks ``artifact_id`` → ``artifact_not_in_cut`` — both a
        typed :class:`~ccs.core.types.SessionCommitRejection`, NEVER a silent
        fall-through to a live-HEAD commit. A FOREIGN caller or an
        OWNED-BUT-PINLESS token RAISES :class:`SessionInvalidated` (Unit 7,
        validated BEFORE the pin lookup); the heartbeat-liveness
        ``session_invalidated`` axis is Unit 5. (The R14 caps are enforced at
        ``begin_session``, not here — a committed session already passed them.)

        Outcome mapping (mirrors the shipped ``commit_cas`` orchestration exactly):

        - WIN → ``(updated_artifact, invalidation_signals)``: the artifact moved
          to ``pinned + 1`` and ``commit_cas`` ALREADY invalidated the peers
          atomically — this method emits NO additional invalidation signal.
        - :class:`ConflictDetail` (``version_mismatch`` / ``other_holder`` /
          ``stale_read_generation``) → RETURNED UNCHANGED (HELD, retry-eligible;
          nothing mutated, so no invalidation is emitted). Recover via a NEW
          session + re-read + re-commit.
        - corruption (``expected_version > current``) → ``commit_cas`` maps the
          registry's :class:`CasCorruption` sentinel to a RAISED ``CoherenceError``
          (non-retryable); ``session_commit`` lets it propagate.

        **"Exactly one validated commit" (R11) is naturally enforced — no explicit
        single-use machinery.** After a WIN the artifact advanced to ``pinned + 1``
        but the cut still pins ``pinned``; a SECOND ``session_commit`` at the same
        pin therefore version-mismatches (``expected_version < current``) and is
        HELD. The pin is not consumed or rewritten here (that would foreclose the
        SB-18 multi-commit shape, R11) — staleness does the enforcing.

        Args:
            session_token: The server-minted session identity from
                ``begin_session``.
            artifact_id: The pinned artifact to commit. Must be in the cut.
            caller: The CALLER'S identity (the MESI agent/process label). Must be
                the session's bound owner — validated timing-safe
                (:func:`hmac.compare_digest`) against the owner bound at
                ``begin_session``; a foreign caller fails closed
                (:class:`SessionInvalidated`). Required (R13): a sibling MUST NOT
                commit into another's cut.
            content: The new body. ``content_hash`` is derived from it
                (``compute_content_hash``); the body is threaded to ``commit_cas``
                so the in-memory path advances ``record.content`` on a WIN (the
                cross-process / ``content=None`` path keeps no body — see
                ``commit_cas``).
            size_tokens: Optional token count to persist with the commit.
            issued_at_tick: Logical tick for the commit (threaded to ``commit_cas``).
            abort: Optional watchdog abort Event (A6). Threaded into ``commit_cas``'s
                ``abort_guard`` so a watchdog-timed-out commit fails closed at the
                registry write lock instead of landing as a phantom write after the
                caller already got a degraded response. ``None`` for in-process
                callers with no watchdog.

        Returns:
            ``(updated_artifact, signals)`` on a WIN, a :class:`ConflictDetail` on a
            retry-eligible lost race, or a :class:`SessionCommitRejection` on a
            validation failure — all typed RETURNS. Corruption RAISES
            ``CoherenceError`` (via ``commit_cas``); a missing artifact under a
            live pin also raises there (the fail-closed ``SessionInvalidated`` for
            that race is Unit 5).
        """
        epoch = self.registry.coordinator_epoch

        # Per-call OWNER-binding validation (Unit 7 / R13) — read the cut and the
        # owner binding CONSISTENTLY, then fail closed BEFORE acting on the pin: a
        # FOREIGN caller or an OWNED-BUT-PINLESS token raises
        # :class:`SessionInvalidated` (a sibling MUST NOT commit into another's
        # cut; an owned-but-pinless token is not-yet-live). A token with NO owner
        # binding falls through to the cut-absent liveness taxonomy below (still
        # fail-closed, never a live-HEAD commit).
        cut = self.registry.get_session_cut(session_token)
        self._validate_session_owner(session_token, caller, cut)
        if cut is None:
            # FAIL CLOSED (Unit 5 / R4): no live cut for this token — reaped,
            # GC-raced, restart-wiped, released, or never opened. NEVER a silent
            # fall-through to a live-HEAD commit. Same wire-stable taxonomy as
            # ``session_read``: ``session_invalidated`` for a reaped / restart-
            # wiped / in-shape token, ``session_not_found`` for a never-opened /
            # malformed one.
            return SessionCommitRejection(
                reason=self._classify_no_cut_reason(session_token),
                artifact_id=artifact_id,
                coordinator_epoch=epoch,
            )
        if artifact_id not in cut:
            # A live session, but this artifact was not pinned. Reject — NEVER
            # commit live HEAD for an un-pinned artifact (a session commits only
            # against what it pinned).
            return SessionCommitRejection(
                reason=SESSION_ARTIFACT_NOT_IN_CUT_REASON,
                artifact_id=artifact_id,
                coordinator_epoch=epoch,
            )

        expected_version = cut[artifact_id]
        committer_id = self._session_committer_id(session_token)
        content_hash = compute_content_hash(content)

        # Reuse the shipped service ``commit_cas`` orchestration VERBATIM: it owns
        # the CasCorruption-sentinel -> raised CoherenceError mapping, returns a
        # ConflictDetail unchanged (no mutation, no invalidation), and builds the
        # InvalidationSignal list on a WIN. Single-shot — there is no retry loop.
        # ``committer_id`` is fence-claimless (admit-on-absent), so the pinned
        # ``expected_version`` is the sole arbiter.
        #
        # ``abort`` is threaded into ``commit_cas``'s ``abort_guard`` (A6): if the
        # handler watchdog timed out and the caller already received a degraded
        # ``commit_unconfirmed`` response, the guard fails this commit closed THE
        # INSTANT it wins the registry write lock — so a watchdog-late commit never
        # lands as a phantom write behind the client's back. Every other mutating
        # path (write / write_cas) threads ``abort`` the same way; ``session_commit``
        # had silently dropped it before this fix.
        return self.commit_cas(
            agent_id=committer_id,
            artifact_id=artifact_id,
            expected_version=expected_version,
            content_hash=content_hash,
            issued_at_tick=issued_at_tick,
            size_tokens=size_tokens,
            content=content,
            abort=abort,
        )

    def _validate_session_owner(
        self,
        session_token: str,
        caller: UUID,
        cut: Mapping[UUID, int] | None,
    ) -> None:
        """Per-call OWNER-binding validation for ``session_read`` / ``session_commit``
        (SB-17 / TX-1, Unit 7 / R13). Fails CLOSED — raises
        :class:`SessionInvalidated` — for a FOREIGN caller (an owner-isolation
        violation). Returns for every non-foreign case; the cut-absent fail-closed
        taxonomy (including the OWNED-BUT-PINLESS case) is left to the caller's
        existing ``cut is None`` path so a single fail-closed shape governs.

        Called BEFORE the pin lookup is acted on, with the cut already read by the
        caller (so owner-binding and pins are read consistently within one call —
        no second registry round-trip that could race the first). A SIBLING agent
        MUST NOT read or commit another session's cut, even with a leaked token:
        cross-agent access is OUT (R13).

        The owner comparison is TIMING-SAFE: it uses :func:`hmac.compare_digest`
        over the stable 16-byte ``UUID.bytes`` encoding (the SAME shape as the
        Unit-5 heartbeat owner-check), NEVER ``==`` / ``in``, so a foreign caller
        learns nothing from response timing about how much of the owner id matched.

        Owner resolution has TWO tiers (finding F3, R6/R13): the in-memory binding
        (fast, service-scoped) and — when that is absent — the DURABLE owner from
        ``registry.get_session_meta`` (persisted alongside the pins, survives a
        restart). The durable fallback is what lets a sqlite session survive a
        coordinator restart (R6) WITHOUT opening an owner-isolation hole: the
        in-memory binding is gone post-restart, but the durable owner still
        authorizes the legitimate owner and rejects a leaked-token foreigner (R13).

        Outcomes:

        - **Owner resolved, caller matches** — RETURN (the happy path; the caller
          proceeds to its pin lookup / serve / commit).
        - **Foreign caller** (owner resolved, caller mismatches) — RAISE
          ``SessionInvalidated``: a sibling cannot read/commit another's cut, even
          with a leaked token. The ONE genuinely exceptional case — an isolation
          breach, raised so it is never confused with a benign not-found. Holds
          whether the owner came from the in-memory binding OR the durable
          fallback (so the post-restart isolation guarantee is identical).
        - **No owner anywhere, but a cut IS present** — RAISE
          ``SessionInvalidated``. A pinned-but-ownerless token: there is no owner
          to authorize the caller, so serving the cut would bypass R13. Should not
          occur given the atomic pins+meta capture, but kept as a defensive
          fail-closed (never serve a cut we cannot authorize).
        - **No owner anywhere, no cut** — RETURN. The token was never opened, was
          released/reaped, or an in-memory-registry restart wiped BOTH the binding
          and the (process-scoped) pins. Not an isolation failure (no owner, no
          cut); the caller's ``cut is None`` path classifies it into the
          wire-stable liveness taxonomy (``session_invalidated`` /
          ``session_not_found``). This is also the owned-but-pinless case
          (begin_session mint→capture window, or the degenerate empty-read-set
          sqlite session whose ``get_session_cut`` is ``None``): owner resolves but
          ``cut is None``, so the match path returns and the cut-absent path fails
          closed — preserving the shipped Unit-3 sqlite-empty-session contract.
        """
        with self._session_lock:
            bound_owner = self._session_owners.get(session_token)
        if bound_owner is None:
            # Fall back to the DURABLE owner (R6/R13): on sqlite the owner survives
            # a restart even though the in-memory map was wiped, so a survived
            # session is still owner-validated. On the in-memory registry this is
            # always None post-restart (process-scoped), so an in-memory restart
            # still fails closed via the cut-absent path below.
            durable = self.registry.get_session_meta(session_token)
            bound_owner = durable[0] if durable is not None else None
        if bound_owner is None:
            # No owner in EITHER tier. If a cut is somehow still present (a
            # pins-without-meta orphan — should not happen given the atomic
            # capture, but defensive), FAIL CLOSED: there is no owner to authorize
            # the caller. Otherwise let the caller's cut-absent path classify it.
            if cut is not None:
                raise SessionInvalidated(
                    "session has pins but no owner-binding: no owner remains to "
                    "authorize the caller (fail closed, R13)"
                )
            return
        # Timing-safe owner-binding compare (R13): stable 16-byte UUID encoding,
        # never ``==``. A foreign caller is rejected fail-closed and cannot probe
        # id-match progress via response timing.
        if not hmac.compare_digest(bound_owner.bytes, caller.bytes):
            raise SessionInvalidated(
                "session owner mismatch: the caller is not the session's owner "
                "(cross-agent session access is out of scope, R13)"
            )

    # ------------------------------------------------------------------
    # Caller principal — mint gate + validator (caller-principal plan, U4)
    # ------------------------------------------------------------------
    #
    # The coordinator authenticates the WORKSPACE (bearer), not the caller: a
    # request's acting identity is caller-asserted. A caller principal is a value
    # this service mints and binds to ONE identity on that identity's first claim,
    # so a request naming the identity can be checked against it. Accident-
    # resistance and attributability under same-OS-user cooperative trust — not a
    # boundary against a process that can read ``.coherence/``.
    #
    # Three lifetimes stay independent (R4): the binding lives in its own registry
    # store, which the grant sweep, the session-liveness sweep, the session cap
    # and session release never read or delete.

    def claim_caller_principal(
        self, *, identity: UUID, mint_nonce: str, abort: threading.Event | None = None
    ) -> str:
        """Bind a caller principal to ``identity`` on its first claim, or hand
        the SAME principal back to a retry of that claim (R1, R2, R20 / KTD11).

        The claimant generates ``mint_nonce`` and persists it BEFORE calling;
        the nonce is stored beside the binding. A later claim presenting the
        same nonce is the same claimant recovering a lost response and gets the
        bound principal again; a claim presenting any other nonce does not
        become the identity and raises :class:`CallerPrincipalRefused`
        (``caller_principal_claimed``). A missing or malformed nonce raises
        ``ValueError`` before anything is bound.

        The bind is ONE registry step (insert-if-absent + read-back), so two
        concurrent first claims bind one principal and the loser is refused.
        ``abort`` is the watchdog Event (A6): a timed-out claim fails closed at
        the registry lock; one that lands anyway is recovered by its nonce.

        Returns the principal. This return value is the one place a principal
        leaves the service; nothing here logs it or puts it in an error."""
        problem = mint_nonce_problem(mint_nonce)
        if problem is not None:
            raise ValueError(problem)
        candidate = secrets.token_urlsafe(_CALLER_PRINCIPAL_BYTES)
        try:
            with self.registry.abort_guard(abort):
                principal, bound_nonce = self.registry.bind_caller_principal(
                    identity, candidate, mint_nonce
                )
        except BaseException:
            self._record_bind_attempt(identity, None)
            raise
        self._record_bind_attempt(identity, principal)
        if not _same_secret(bound_nonce, mint_nonce):
            raise CallerPrincipalRefused(
                CALLER_PRINCIPAL_CLAIMED_REASON,
                "the identity is already bound to a caller principal under a "
                "different mint nonce; a claim never rebinds it, and only a retry "
                "of the original claim (presenting its nonce) re-obtains it",
            )
        return principal

    def _record_bind_attempt(self, identity: UUID, bound_principal: str | None) -> None:
        """Bring the cache up to date with a finished bind attempt on
        ``identity`` — the negative tier's one way to go stale.

        ``bound_principal`` is what the store now binds to ``identity`` (this
        claim's or an earlier one's); it REPLACES whatever is cached, above all
        a cached UNBOUND answer, so a claimed identity's absent-principal
        request is refused from the first request after the claim. ``None``
        means the attempt raised, which does not prove nothing landed: a cached
        UNBOUND answer is dropped, and the next lookup reads the store. The
        attempt is counted either way (see ``_caller_principal_binds``)."""
        with self._caller_principal_lock:
            self._caller_principal_binds += 1
            if bound_principal is not None:
                self._caller_principals[identity] = bound_principal
            elif identity in self._caller_principals and self._caller_principals[identity] is None:
                del self._caller_principals[identity]

    def validate_caller_principal(
        self, *, identity: UUID, principal: object, cached_only: bool = False
    ) -> None:
        """Check that ``principal`` is the one bound to ``identity`` (R1).

        Returns on a match. Raises :class:`CallerPrincipalRefused` with
        ``caller_principal_absent`` when no principal is presented, and with
        ``caller_principal_foreign`` when one is presented but does not match —
        minted for another identity, never minted, or presented for an identity
        nobody has claimed. The comparison is :func:`hmac.compare_digest` over
        the encoded values, the owner-validation pattern.

        Resolution has two tiers, like :meth:`_validate_session_owner`: the
        in-process cache, then the registry's durable store — so a binding made
        before a coordinator restart still validates after it (sqlite).
        ``cached_only`` stops at the first tier: when the cache cannot answer it
        raises :class:`CallerPrincipalUncached` instead of reading the store."""
        if principal is None or principal == "":
            raise CallerPrincipalRefused(
                CALLER_PRINCIPAL_ABSENT_REASON,
                "the request names an identity but presents no caller principal",
            )
        bound = self._bound_caller_principal(identity, cached_only=cached_only)
        if bound is None or not isinstance(principal, str) or not _same_secret(
            bound, principal
        ):
            raise CallerPrincipalRefused(
                CALLER_PRINCIPAL_FOREIGN_REASON,
                "the presented caller principal is not the one bound to the "
                "identity the request names",
            )

    def is_caller_principal_bound(self, identity: UUID, *, cached_only: bool = False) -> bool:
        """Whether ``identity`` has ever been claimed — a principal is bound to
        it, in the cache or the durable store.

        The fact the require-class routes branch on when a request presents NO
        principal (caller-principal plan, U6 / R16): an identity nobody has
        claimed belongs to a caller that predates the principal and is admitted
        as before, while a bound one is refused as absent. A False answer from
        the store is cached (the negative tier), and this service's own claim of
        the identity replaces it, so an identity bound after a False answer
        reads True at once. ``cached_only`` raises
        :class:`CallerPrincipalUncached` where the store would be read. The
        principal itself never leaves the service through here."""
        return self._bound_caller_principal(identity, cached_only=cached_only) is not None

    def _bound_caller_principal(
        self, identity: UUID, *, cached_only: bool = False
    ) -> str | None:
        """The principal bound to ``identity``, or ``None`` when it is unbound:
        the cache, else the durable store, whose answer is cached either way —
        a bound one because bindings are never rebound, an UNBOUND one until
        this service binds the identity (:meth:`_record_bind_attempt`)."""
        with self._caller_principal_lock:
            if identity in self._caller_principals:
                return self._caller_principals[identity]
            if cached_only:
                raise CallerPrincipalUncached(
                    "the caller-principal cache holds no answer for this identity"
                )
            binds_before = self._caller_principal_binds
        durable = self.registry.get_caller_principal(identity)
        with self._caller_principal_lock:
            if durable is not None:
                self._caller_principals[identity] = durable
            elif self._caller_principal_binds == binds_before:
                # setdefault, not assignment: a lookup that read the store after
                # a bind may already have cached the bound principal, and this
                # older UNBOUND answer must not replace it.
                self._caller_principals.setdefault(identity, None)
            return self._caller_principals.get(identity, durable)

    @staticmethod
    def _session_committer_id(session_token: str) -> UUID:
        """Derive the SESSION-SCOPED committer identity for ``session_commit``.

        A deterministic ``uuid5`` over the server-minted ``session_token`` (under
        the URL namespace). Stable across a session's commits and collision-free
        against real MESI agent ids; crucially it has NEVER established a
        read-generation fence claim, so ``commit_cas`` ADMITS it on absence and
        the pinned-base version-CAS arbitrates (the R3 load-bearing path). NOT the
        owner's MESI ``agent_id``, which could carry a superseded
        ``read_generation`` that would spuriously trip the fence.
        """
        return uuid5(NAMESPACE_URL, session_token)

    # ------------------------------------------------------------------
    # Effect-gate wrapper — "fire E iff read-set R unchanged" (Unit 6 / EO-5)
    # ------------------------------------------------------------------

    def effect_gate(
        self,
        *,
        read_set: Iterable[UUID],
        owner: UUID,
        decide: Callable[["SessionView"], object],
        effect: Callable[[object], object] | None = None,
        commit: tuple[UUID, bytes | str] | None = None,
        created_at_tick: int = 0,
        issued_at_tick: int = 0,
        release_on_exit: bool = True,
    ) -> EffectFired | EffectHeld:
        """Fire an effect IFF the whole read-set is still unchanged — the
        ergonomic EO-5 surface (SB-17 / TX-1, Unit 6 = EO-4 = SB-17 user surface).

        One call composes the shipped session primitives end to end:

        1. **PIN** — ``begin_session(read_set)`` captures a consistent cut at one
           linearization point (Unit 2).
        2. **DECIDE** — the caller's ``decide`` callback reads the PINNED cut (via
           a :class:`SessionView` over ``session_read``, Unit 3) and computes a
           decision (the value it returns is threaded to an escaping effect).
        3. **RE-VALIDATE** — at the effect boundary, re-read each read-set
           member's CURRENT version (``registry.get_artifact(id).version``) and
           compare to the pin. If EVERY member matches → fire; if ANY moved (or
           vanished) → **HELD** (:class:`EffectHeld`), never fire on stale input.
        4. **FIRE** — per mode (below).

        Two effect modes (exactly one of ``effect`` / ``commit`` — passing both,
        or neither, is caller misuse → ``ValueError``):

        - **ATOMIC** (``commit=(artifact_id, content)``) — the effect IS an
          artifact write, routed through :meth:`session_commit` so the shipped
          ``commit_cas`` arbitrates AT the pinned base in the SAME step. There is
          NO re-validate→fire window: "unchanged" and "commit" are one atomic
          arbitration. The pre-fire re-validate still runs as a fast HELD short
          circuit (it spares the CAS when a peer already moved the target), but
          the AUTHORITATIVE guard is ``commit_cas`` — even if a peer commits in
          the instant between the re-validate and the CAS, the CAS at the pinned
          ``expected_version`` loses cleanly and returns :class:`ConflictDetail`
          (surfaced as :class:`EffectHeld` with ``conflict`` set). This is the
          STRONG guarantee.
        - **ESCAPING** (``effect=callable``) — the effect is a non-commit side
          effect (deploy / charge / click). The gate re-validates, then fires the
          callable. **The guarantee is "the read-set was unchanged AS OF the
          re-validate point", NOT "as of the fire point".** A peer can commit a
          read-set member in the residual RE-VALIDATE→FIRE window — after the
          check passed but before the callable runs — and the gate will STILL
          fire (it gates pre-fire and never rolls back, EO-7). This window is
          unclosable for escaping effects and is NOT claimed away. Use ATOMIC
          mode when the effect is an artifact write and you need the window
          closed; use ESCAPING for genuine side effects, accepting the bound.

        **Fail-closed (in-process typed path).** This gate is a pure in-process
        coordinator method — it composes the typed session results directly and
        is NOT the HTTP/CoherentVolume path, so there is no 200-body deny/degrade
        to translate. A dead session at ANY step (``session_read`` /
        ``session_commit`` returning ``session_invalidated``/``session_not_found``,
        or the cut vanishing at re-validate) RAISES :class:`SessionInvalidated`
        and NEVER fires. (If this gate were ever rebuilt over the
        ``coherent_volume`` HTTP surface, BOTH the ``ok:false`` deny body and the
        ``degraded:true`` 200 body would map to a raise — never proceed
        best-effort on a degrade, learnings #3 ``coordinator-invalidation-not-
        mutex``. The in-process typed path enforces the same fail-closed shape by
        construction.)

        Reason classification uses ``reason == CONSTANT`` against the wire-stable
        session reason sets, never a substring of a human message (the
        typed-signal-not-substring house rule).

        Args:
            read_set: The artifacts to pin into the consistent cut.
            owner: The creating caller's identity (bound to the minted session).
            decide: Callback ``(view) -> decision``; reads the pinned cut via
                ``view.read(artifact_id)`` and returns a decision value. For an
                escaping effect the decision is passed to ``effect``.
            effect: ESCAPING mode — a side-effect callable ``(decision) -> result``
                fired only if re-validate passes. Mutually exclusive with ``commit``.
            commit: ATOMIC mode — ``(artifact_id, content)`` committed via
                ``session_commit`` at the pinned base. Mutually exclusive with ``effect``.
            created_at_tick: Logical tick for ``begin_session`` (heartbeat seed).
            issued_at_tick: Logical tick for the ATOMIC ``session_commit``.
            release_on_exit: If ``True`` (default), the session's pins are released
                after the gate resolves (fire or HELD), so a one-shot gate does not
                leak a pin until the liveness sweep reaps it. Set ``False`` to keep
                the session live for further use.

        Returns:
            :class:`EffectFired` if the read-set was unchanged (as of re-validate,
            or atomically for ``commit``) and the effect fired, else
            :class:`EffectHeld` (drift detected pre-fire, or an atomic OCC loss).

        Raises:
            ValueError: neither or both of ``effect`` / ``commit`` supplied.
            SessionInvalidated: the session died mid-gate (fail-closed; no fire).
        """
        if (effect is None) == (commit is None):
            raise ValueError(
                "effect_gate requires exactly one of effect= (escaping mode) or "
                "commit= (atomic mode); got "
                + ("both" if effect is not None else "neither")
            )

        read_set = list(read_set)
        session = self.begin_session(
            read_set=read_set, owner=owner, created_at_tick=created_at_tick
        )
        if isinstance(session, VersionedReadRejection):
            # An unknown id in the read-set never opened a session; surface it as
            # a fail-closed raise (no cut to gate against, never a silent fire).
            raise SessionInvalidated(
                f"effect_gate could not pin the read-set: {session.reason} "
                f"(artifact={session.artifact_id})"
            )

        token = session.session_token
        try:
            # DECIDE — the caller reads the PINNED cut and computes its decision.
            view = SessionView(self, token, owner)
            decision = decide(view)

            if commit is not None:
                return self._effect_gate_atomic(
                    session=session,
                    owner=owner,
                    commit=commit,
                    issued_at_tick=issued_at_tick,
                )
            assert effect is not None  # narrowed by the XOR check above
            return self._effect_gate_escaping(
                session=session,
                effect=effect,
                decision=decision,
            )
        finally:
            if release_on_exit:
                # Best-effort cleanup: drop the pins + durable owner-binding so a
                # one-shot gate does not hold versions back from GC until the
                # liveness sweep. The shared teardown (idempotent on the registry,
                # lock-aware on the in-memory maps) — the SAME path the sweep reaps
                # through, so the token cannot linger in either store.
                self._drop_session_state(token)

    def _revalidate_cut(
        self, cut: Mapping[UUID, int]
    ) -> dict[UUID, "tuple[int, int | None]"]:
        """Re-read each pinned member's CURRENT version and return the DRIFT map.

        For every ``(artifact_id, pinned)`` in ``cut``, read the live current
        version (``registry.get_artifact(id).version``); a vanished artifact reads
        ``None`` (deleted / GC-raced under the pin). Returns ``{artifact_id:
        (pinned, current)}`` for ONLY the members whose ``current != pinned`` (or
        whose artifact vanished) — an empty dict means the whole vector is
        unchanged. This is the pre-fire re-validate vector compare; it is
        non-mutating (a read-only ``get_artifact`` per member, no grant, no
        ``read_generation``)."""
        moved: dict[UUID, "tuple[int, int | None]"] = {}
        for artifact_id, pinned in cut.items():
            artifact = self.registry.get_artifact(artifact_id)
            current = artifact.version if artifact is not None else None
            if current != pinned:
                moved[artifact_id] = (pinned, current)
        return moved

    def _effect_gate_escaping(
        self,
        *,
        session: SnapshotSession,
        effect: Callable[[object], object],
        decision: object,
    ) -> EffectFired | EffectHeld:
        """ESCAPING mode — re-validate the vector, then fire the side-effect
        callable. Documents the residual re-validate→fire window (EO-7): the
        callable fires on a vector proven unchanged AT the re-validate point, not
        at the fire point — a peer commit in the window is NOT caught and the
        effect is NOT rolled back. Use ATOMIC mode to close the window."""
        moved = self._revalidate_cut(session.cut)
        if moved:
            # Drift detected BEFORE firing — HELD, the side effect never runs.
            return EffectHeld(
                moved=moved,
                coordinator_epoch=session.coordinator_epoch,
            )

        # The vector was unchanged AS OF this point. Everything below this line is
        # the unclosable RE-VALIDATE→FIRE WINDOW for an escaping effect: a peer can
        # commit a read-set member here, after the check passed, before the
        # callable runs. The gate gates PRE-FIRE and never rolls back (EO-7), so
        # the guarantee is "unchanged as of re-validate", NOT "as of fire". This
        # is intrinsic to a non-commit side effect (deploy/charge/click cannot be
        # version-CAS'd); ATOMIC mode closes it by riding ``commit_cas``.
        result = effect(decision)
        return EffectFired(
            revalidated_cut=dict(session.cut),
            coordinator_epoch=session.coordinator_epoch,
            result=result,
        )

    def _effect_gate_atomic(
        self,
        *,
        session: SnapshotSession,
        owner: UUID,
        commit: tuple[UUID, bytes | str],
        issued_at_tick: int,
    ) -> EffectFired | EffectHeld:
        """ATOMIC mode — route the write through ``session_commit`` so the shipped
        ``commit_cas`` arbitrates at the pinned base in the SAME step. NO
        re-validate→fire window: the CAS at ``expected_version=pin`` is the
        authoritative guard. A pre-CAS re-validate runs as a fast HELD short
        circuit (sparing the CAS when the target already moved), but even without
        it the CAS would lose cleanly on a raced peer commit."""
        artifact_id, content = commit
        # Fast pre-CAS re-validate: if the COMMIT TARGET already moved, hold
        # before attempting the CAS. (The CAS itself is still authoritative for
        # the no-window guarantee; this only short-circuits the common case and
        # surfaces a uniform drift map with the escaping mode.)
        moved = self._revalidate_cut(session.cut)
        if artifact_id in moved:
            return EffectHeld(
                moved=moved,
                coordinator_epoch=session.coordinator_epoch,
            )

        # Atomic arbitration at the pinned base. ``session_commit`` reuses the
        # shipped ``commit_cas`` verbatim (admit-on-absent → version-CAS), maps
        # the CasCorruption sentinel to a raised CoherenceError, and returns the
        # ConflictDetail unchanged on a lost race. A SessionCommitRejection here
        # means the cut died mid-gate (token/pin gone) → fail closed.
        outcome = self.session_commit(
            session.session_token,
            artifact_id,
            content,
            caller=owner,
            issued_at_tick=issued_at_tick,
        )
        if isinstance(outcome, SessionCommitRejection):
            # Fail-closed: the session is gone (reaped / restart-wiped / released)
            # — never a silent non-fire. Raise the typed dead-session error.
            raise SessionInvalidated(
                f"effect_gate atomic commit failed closed: {outcome.reason} "
                f"(artifact={outcome.artifact_id})"
            )
        if isinstance(outcome, ConflictDetail):
            # Lost the OCC race AT the pinned base — HELD, nothing mutated. Carry
            # the shipped ConflictDetail through verbatim (the same taxonomy a
            # bare session_commit surfaces). Re-read the post-conflict drift so
            # ``moved`` reflects what actually changed under the commit target.
            post = self._revalidate_cut(session.cut)
            return EffectHeld(
                moved=post,
                coordinator_epoch=session.coordinator_epoch,
                conflict=outcome,
            )
        # WIN — (updated_artifact, signals); the commit landed atomically at the
        # pinned base with NO window.
        return EffectFired(
            revalidated_cut=dict(session.cut),
            coordinator_epoch=session.coordinator_epoch,
            commit=outcome,
        )

    # ------------------------------------------------------------------
    # Session pin lifetime — heartbeat lease + liveness sweep (Unit 5 / R4)
    # ------------------------------------------------------------------

    def record_session_heartbeat(
        self, *, session_token: str, owner: UUID, now_tick: int
    ) -> bool:
        """Refresh a snapshot session's heartbeat LEASE (SB-17 / TX-1, Unit 5 /
        R4). Keyed by the server-minted SESSION TOKEN, owner-bound.

        A session is NOT a MESI agent — this does NOT key by ``agent_id`` and does
        NOT reuse :meth:`record_heartbeat` (the grant-holder heartbeat). It keeps
        the session's own lease alive so the session-liveness sweep
        (:meth:`enforce_session_liveness`) does not reap its pins while the owner
        is still working. Monotonic like the grant heartbeat: ``max(prev,
        incoming)``, so a stale/replayed lower tick never moves the lease back.

        **Owner-bound (R13, security).** The caller must be the session's OWNER —
        the identity bound at ``begin_session``. A FOREIGN caller must NOT be able
        to keep another agent's session alive (that would let an attacker pin
        versions against GC indefinitely under someone else's session). The owner
        check is TIMING-SAFE: it compares the bound owner and the supplied
        ``owner`` via :func:`hmac.compare_digest` over their stable 16-byte
        big-endian encoding, so a foreign caller learns nothing from response
        timing about how much of the id matched. A mismatch is rejected and the
        lease is NOT refreshed.

        Heartbeating an unknown / released token is a typed NO-OP: it returns
        ``False`` (never a crash, never a resurrection — a genuinely dead session
        cannot be revived; re-establish it via ``begin_session``). It does NOT
        create a lease for a token with no owner anywhere, so a heartbeat cannot
        conjure a live lease for a cut-less token.

        **Restart rehydration (R6).** A sqlite session that SURVIVED a restart has
        durable pins + a durable owner but no in-memory lease. Its owner's first
        post-restart heartbeat falls back to the durable owner, and on a match
        REHYDRATES the in-memory owner + creation tick + lease — so the sweep
        manages it normally again. An in-memory-registry session has no durable
        meta, so its post-restart heartbeat stays a no-op (the asserted
        process-scoped divergence).

        Args:
            session_token: The server-minted token from ``begin_session``.
            owner: The caller's identity; must match the bound owner (timing-safe).
            now_tick: The heartbeat tick (``>= 0``). Applied as ``max(prev,
                incoming)``.

        Returns:
            ``True`` if the lease was refreshed (known token, owner matched),
            else ``False`` (unknown/released/wiped token, or a foreign caller —
            indistinguishable to the caller by design: a foreign caller is not
            told whether the token exists).
        """
        if now_tick < 0:
            raise ValueError("now_tick must be >= 0")
        with self._session_lock:
            bound_owner = self._session_owners.get(session_token)
            rehydrate_created: int | None = None
            if bound_owner is None:
                # Fall back to the DURABLE owner (R6): a sqlite session that
                # survived a restart has no in-memory binding, but its owner can
                # still refresh the lease — and doing so REHYDRATES the in-memory
                # state (below) so the sweep lifecycle-manages it normally again. A
                # foreign caller still cannot keep it alive: the timing-safe check
                # runs on the durable owner too. An unknown/released token (or any
                # in-memory-registry token post-restart) has no durable meta →
                # typed no-op.
                durable = self.registry.get_session_meta(session_token)
                if durable is None:
                    return False
                bound_owner, rehydrate_created = durable[0], durable[1]
            # Timing-safe owner-binding check (R13): compare over the stable
            # 16-byte big-endian UUID encoding. A foreign caller cannot keep
            # another's session alive AND cannot probe id-match progress via timing.
            if not hmac.compare_digest(bound_owner.bytes, owner.bytes):
                return False
            if rehydrate_created is not None:
                # Durable-only session (post-restart): adopt it back into the
                # in-memory maps so subsequent sweeps see + manage it. The creation
                # tick is the ORIGINAL durable one, so the absolute-age ceiling is
                # measured from first creation — a restart never resets it (R14).
                self._session_owners[session_token] = bound_owner
                self._session_created[session_token] = rehydrate_created
            prev = self._session_heartbeats.get(session_token)
            if prev is None or now_tick > prev:
                self._session_heartbeats[session_token] = now_tick
        return True

    def enforce_session_liveness(
        self,
        *,
        current_tick: int,
        heartbeat_timeout_ticks: int,
    ) -> int:
        """Reap snapshot sessions whose heartbeat lease has gone stale — the NEW
        session-liveness sweep AXIS (SB-17 / TX-1, Unit 5 / R4).

        Distinct from :meth:`enforce_stable_grant_timeouts`: that sweep walks
        only M∪E GRANT-HOLDERS, and a snapshot session holds NO grant, so it is
        invisible to the grant sweep. This sweep ENUMERATES SESSIONS (the UNION of
        the in-memory owner-bound token set AND the durable ``session_meta`` token
        set, so a restart-survived sqlite session is still bounded) and reaps any
        whose lease is stale, reusing the SAME heartbeat-staleness predicate SHAPE
        as the grant sweep (``current - last_hb >= timeout``, with ``>=`` matching
        ADV-02) and the same ``CrashRecoveryConfig`` knob
        (``heartbeat_timeout_ticks``). A durable-only session (post-restart, no
        in-memory lease) is bounded by the absolute-age ceiling ALONE — applying
        heartbeat-staleness to a leaseless survivor would reap it on the first
        post-restart sweep and defeat R6.

        **TWO reap conditions, OR'd (a session is reaped if EITHER fires):**

        1. **Heartbeat staleness** (the Unit-5 lease) — ``current_tick - last_hb
           >= heartbeat_timeout_ticks``. A slow-but-LIVE session heartbeated
           within the window survives indefinitely; this is a PREDICATE on the
           lease, not a hard TTL.
        2. **Absolute-age ceiling** (Unit 7 / R14, threat-model #3) —
           ``current_tick - created_at_tick >= session_caps.absolute_age_ticks``.
           A HARD ceiling SEPARATE from the heartbeat lease: a session older than
           the ceiling is reaped **EVEN WHEN ITS HEARTBEAT IS LIVE** — a live
           heartbeat MUST NOT exempt it. This bounds the heartbeat-spoofing DoS
           (an attacker keeping a stale cut pinned indefinitely via heartbeats).
           The age is measured from ``created_at_tick`` (seeded at
           ``begin_session``), NOT the last heartbeat, so heartbeating never
           resets it. This is the ONLY place the ceiling is enforced; it does NOT
           reuse the grant sweep's ``max_hold_ticks`` (that bounds GRANTS, this
           bounds SESSIONS).

        Reaping a session: :meth:`registry.release_session` drops its pins AND its
        durable owner-binding (so its pinned versions become collectible again and
        no orphaned owner remains), then its in-memory owner binding, heartbeat
        lease, and creation tick are cleared and the token is recorded in the
        bounded reaped tombstone. After reaping, a ``session_read`` /
        ``session_commit`` on that token fails closed with ``session_invalidated``
        (the cut is gone).

        Args:
            current_tick: The sweep's logical clock (wall-clock ticks via
                ``monotonic_seconds``, the SAME basis the heartbeat handlers seed
                from — finding F1). Both reap conditions are measured against it.
            heartbeat_timeout_ticks: Reap any session whose lease is at least
                this many ticks stale (or that somehow has no lease — defensive).
                The absolute-age ceiling (``session_caps.absolute_age_ticks``) is
                applied INDEPENDENTLY — a fresh heartbeat does not exempt it.

        Returns:
            The number of sessions reaped (by EITHER condition).
        """
        if heartbeat_timeout_ticks < 1:
            raise ValueError("heartbeat_timeout_ticks must be >= 1")

        absolute_age_ticks = self.session_caps.absolute_age_ticks
        reaped = 0
        with self._session_lock:
            # Enumerate the UNION of in-memory sessions and DURABLE sessions
            # (R6/F3). A sqlite session that survived a restart has durable meta
            # but NO in-memory state, so an in-memory-only walk would never see it
            # — yet its pins must still be bounded (the age ceiling) and eventually
            # reaped (else they starve GC). ``durable`` maps token → (owner,
            # created_at_tick); the in-memory set is authoritative for the lease.
            # The token set is a snapshot (reaping mutates the maps under the loop).
            durable = self.registry.all_session_meta()
            tokens = set(self._session_owners) | set(durable)
            for session_token in tokens:
                if session_token in self._session_owners:
                    # IN-MEMORY session: heartbeat staleness (Condition 1) applies.
                    # Same predicate SHAPE as the grant sweep (``>=`` per ADV-02);
                    # a missing lease is treated as stale, defensively.
                    last_hb = self._session_heartbeats.get(session_token)
                    stale = (
                        last_hb is None
                        or (current_tick - last_hb) >= heartbeat_timeout_ticks
                    )
                    created = self._session_created.get(session_token)
                else:
                    # DURABLE-ONLY (post-restart, not yet rehydrated): there is NO
                    # in-memory lease, so heartbeat-staleness does NOT apply —
                    # applying it would reap a just-survived session on the first
                    # post-restart sweep and defeat R6. It is bounded ONLY by the
                    # absolute-age ceiling, measured from the DURABLE creation tick.
                    # (The owner can rehydrate it to full in-memory management via a
                    # heartbeat; until then the ceiling is its sole bound.)
                    stale = False
                    created = durable[session_token][1]
                # Condition 2 — absolute-age ceiling (Unit 7 / R14, threat-model
                # #3): measured from the CREATION tick, INDEPENDENT of the
                # heartbeat, so a live-heartbeat session past the ceiling is STILL
                # reaped. A missing creation tick is treated as over-age,
                # defensively. With wall-clock ticks (finding F1) the durable
                # creation tick stays comparable to ``current_tick`` across a
                # restart, so the ceiling correctly bounds a survived session.
                over_age = (
                    created is None
                    or (current_tick - created) >= absolute_age_ticks
                )
                if not (stale or over_age):
                    continue
                self._reap_session(session_token)
                reaped += 1
        return reaped

    def _drop_session_state(self, session_token: str) -> None:
        """Release a session's durable pins+owner-binding AND clear its in-memory
        lease/owner/creation state (SB-17 / TX-1, Unit 5 / R4). The ONE place this
        teardown lives — shared by :meth:`_reap_session` (sweep) and the
        ``effect_gate`` release-on-exit path (finding: dedup the two identical
        cleanup blocks). Order is fixed: release the durable store FIRST (the
        registry is the authoritative pin store, idempotent on an unknown token, so
        a crash mid-teardown leaves no live-pin-without-owner leak on sqlite), then
        drop the in-memory maps. Holds ``_session_lock`` (re-entrant — the sweep
        already holds it) so the in-memory drops never race a concurrent handler."""
        self.registry.release_session(session_token)
        with self._session_lock:
            self._session_heartbeats.pop(session_token, None)
            self._session_owners.pop(session_token, None)
            self._session_created.pop(session_token, None)

    def _reap_session(self, session_token: str) -> None:
        """Drop a session's pins + lease + owner binding and tombstone the token
        (Unit 5 / R4). Reaping = drop the session state (durable + in-memory) then
        record the token in the bounded tombstone so a subsequent read/commit
        attributes it precisely to ``session_invalidated``."""
        self._drop_session_state(session_token)
        self._tombstone_token(session_token)

    def _tombstone_token(self, session_token: str) -> None:
        """Record a reaped token in the bounded FIFO tombstone (Unit 5 / R4).
        Capped at :data:`_REAPED_TOMBSTONE_CAP`; the oldest entry is evicted when
        full. Eviction is benign — an evicted in-shape token still classifies
        ``session_invalidated`` via :func:`looks_like_session_token`, so the
        fail-closed guarantee never depends on tombstone residency."""
        # Move-to-end keeps the most-recently-reaped tokens; popitem(last=False)
        # evicts the oldest (FIFO) when over the cap. Under ``_session_lock``
        # (re-entrant — the sweep already holds it): the insert+move+evict is a
        # compound op a bare GIL would not make atomic against a concurrent
        # classify read (finding F4).
        with self._session_lock:
            self._reaped_tombstone[session_token] = None
            self._reaped_tombstone.move_to_end(session_token)
            while len(self._reaped_tombstone) > _REAPED_TOMBSTONE_CAP:
                self._reaped_tombstone.popitem(last=False)

    def _classify_no_cut_reason(self, session_token: str) -> str:
        """Classify a token that has NO live cut into the wire-stable fail-closed
        reason (Unit 5 / R4). FAIL CLOSED in every branch — this only sharpens
        the SIGNAL, never serves live HEAD.

        - In the reaped tombstone → ``session_invalidated`` (definitely reaped
          this process: "your session died, re-establish it").
        - Shaped like a server-minted token (``looks_like_session_token``) →
          ``session_invalidated``. This is the POST-RESTART-UNKNOWN safety case:
          an in-memory restart wiped ``_session_pins`` (and the tombstone), so a
          previously-valid token now has no cut — but it WAS a real session, so it
          must fail closed as invalidated, NEVER served live HEAD as if pinned. A
          well-formed-but-never-minted token also lands here (fail-closed is the
          safe default for an in-shape token; it is rejected either way).
        - Otherwise (out-of-shape / malformed) → ``session_not_found``: a
          genuinely never-opened token. Kept reachable additively so a clearly
          bogus token is still distinguishable.
        """
        with self._session_lock:
            in_tombstone = session_token in self._reaped_tombstone
        if in_tombstone:
            return SESSION_INVALIDATED_REASON
        if looks_like_session_token(session_token):
            return SESSION_INVALIDATED_REASON
        return SESSION_NOT_FOUND_REASON

    def write(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        issued_at_tick: int = 0,
        abort: threading.Event | None = None,
    ) -> list[InvalidationSignal]:
        """Request write ownership by invalidating peers and granting EXCLUSIVE.

        Wrapped in :meth:`registry.abort_guard` (finding A6): if the handler
        watchdog already timed out, the grant aborts before it lands rather than
        leaving a phantom EXCLUSIVE the agent never saw (and silently
        invalidating its peers)."""
        with self.registry.abort_guard(abort):
            return self._write_impl(
                agent_id=agent_id, artifact_id=artifact_id, issued_at_tick=issued_at_tick
            )

    def _write_impl(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        issued_at_tick: int = 0,
    ) -> list[InvalidationSignal]:
        artifact = self._require_artifact(artifact_id)
        self.registry.set_agent_transient(
            artifact_id,
            agent_id,
            TransientState.IED,
            entered_tick=issued_at_tick,
        )
        signals: list[InvalidationSignal] = []
        for peer_id, state in self.registry.get_state_map(artifact_id).items():
            if peer_id == agent_id or state == MESIState.INVALID:
                continue
            transient = _invalidation_transient_for_state(state)
            if transient is not None:
                self.registry.set_agent_transient(
                    artifact_id,
                    peer_id,
                    transient,
                    entered_tick=issued_at_tick,
                )
            self.registry.set_agent_state(artifact_id, peer_id, MESIState.INVALID, trigger="write", tick=issued_at_tick)
            signals.append(
                InvalidationSignal(
                    artifact_id=artifact_id,
                    new_version=artifact.version,
                    issued_at_tick=issued_at_tick,
                    issuer_agent_id=agent_id,
                )
            )

        self.registry.set_agent_state(artifact_id, agent_id, MESIState.EXCLUSIVE, trigger="write", tick=issued_at_tick)
        self.registry.clear_agent_transient(artifact_id, agent_id)
        self._validate_single_writer(artifact_id)
        return signals

    def upgrade(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        issued_at_tick: int = 0,
    ) -> list[InvalidationSignal]:
        """Upgrade a shared holder to exclusive owner (alias of write request)."""
        return self.write(agent_id=agent_id, artifact_id=artifact_id, issued_at_tick=issued_at_tick)

    def commit(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        content: str,
        issued_at_tick: int = 0,
        content_hash: str | None = None,
        size_tokens: int | None = None,
        abort: threading.Event | None = None,
    ) -> tuple[Artifact, list[InvalidationSignal]]:
        """Commit modified content under the A6 abort guard (see _commit_impl)."""
        with self.registry.abort_guard(abort):
            return self._commit_impl(
                agent_id=agent_id,
                artifact_id=artifact_id,
                content=content,
                issued_at_tick=issued_at_tick,
                content_hash=content_hash,
                size_tokens=size_tokens,
            )

    def _commit_impl(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        content: str,
        issued_at_tick: int = 0,
        content_hash: str | None = None,
        size_tokens: int | None = None,
    ) -> tuple[Artifact, list[InvalidationSignal]]:
        """Commit modified content, increment version, and invalidate peers.

        Raises:
            CoherenceError: the committer does not hold M/E (e.g. its grant was
                already reclaimed — the error names the reclaim trigger/tick).
            StaleReadGeneration: the read-generation fence fired — a sweep
                reclaimed this committer in the race window between the state
                check above and the version persist (``ccs.core.exceptions``).
                Retry-eligible: ``reacquire()`` / re-acquire, take a fresh read,
                and re-commit; there is no built-in retry loop on this path
                (unlike ``write_cas``).
        """
        artifact = self._require_artifact(artifact_id)
        agent_state = self.registry.get_agent_state(artifact_id, agent_id)
        if agent_state not in {MESIState.EXCLUSIVE, MESIState.MODIFIED}:
            reclamation = self.registry.get_last_reclamation(agent_id, artifact_id)
            if reclamation is not None:
                trigger, reclaimed_at_tick = reclamation
                raise CoherenceError(
                    f"commit_not_allowed agent={agent_id} artifact={artifact_id} "
                    f"state={agent_state} reclaimed_by={trigger} at_tick={reclaimed_at_tick}"
                )
            raise CoherenceError(
                f"commit_not_allowed agent={agent_id} artifact={artifact_id} state={agent_state}"
            )

        self.registry.set_agent_transient(
            artifact_id,
            agent_id,
            TransientState.MWB,
            entered_tick=issued_at_tick,
        )
        next_version = artifact.version + 1
        check_monotonic_version(artifact.version, next_version)
        updated = Artifact(
            id=artifact.id,
            name=artifact.name,
            version=next_version,
            content_hash=content_hash if content_hash is not None else artifact.content_hash,
            size_tokens=size_tokens if size_tokens is not None else artifact.size_tokens,
            depends_on=artifact.depends_on,
        )
        try:
            self.registry.set_artifact_and_content(
                artifact_id,
                updated,
                content,
                last_writer=agent_id,
                # Read-generation fence: reject atomically with the version bump
                # if a sweep reclaimed this committer in the race window between
                # the get_agent_state check above and here.
                fence_agent_id=agent_id,
            )
        except StaleReadGeneration:
            # The MWB transient set above must not outlive a fence reject:
            # a stuck MWB blocks this agent's next commit_cas (transient
            # precondition) and makes the stable-grant sweep skip the pair
            # until the transient timeout. The reclaim already dropped the
            # grant, so clearing the transient is the only cleanup needed.
            self.registry.clear_agent_transient(artifact_id, agent_id)
            raise

        signals: list[InvalidationSignal] = []
        for peer_id, state in self.registry.get_state_map(artifact_id).items():
            if peer_id == agent_id or state == MESIState.INVALID:
                continue
            transient = _invalidation_transient_for_state(state)
            if transient is not None:
                self.registry.set_agent_transient(
                    artifact_id,
                    peer_id,
                    transient,
                    entered_tick=issued_at_tick,
                )
            self.registry.set_agent_state(
                artifact_id, peer_id, MESIState.INVALID, trigger="commit", tick=issued_at_tick
            )
            signals.append(
                InvalidationSignal(
                    artifact_id=artifact_id,
                    new_version=next_version,
                    issued_at_tick=issued_at_tick,
                    issuer_agent_id=agent_id,
                )
            )
        commit_hash = content_hash if content_hash is not None else compute_content_hash(content)
        self.registry.set_agent_state(
            artifact_id, agent_id, MESIState.MODIFIED,
            trigger="commit", tick=issued_at_tick, content_hash=commit_hash,
        )
        self.registry.clear_agent_transient(artifact_id, agent_id)
        self._validate_single_writer(artifact_id)
        return updated, signals

    def commit_cas(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        expected_version: int,
        content_hash: str,
        issued_at_tick: int = 0,
        size_tokens: int | None = None,
        content: bytes | str | None = None,
        abort: threading.Event | None = None,
    ) -> tuple[Artifact, list[InvalidationSignal]] | ConflictDetail:
        """Optimistic-concurrency commit under the A6 abort guard (see _commit_cas_impl)."""
        with self.registry.abort_guard(abort):
            return self._commit_cas_impl(
                agent_id=agent_id,
                artifact_id=artifact_id,
                expected_version=expected_version,
                content_hash=content_hash,
                issued_at_tick=issued_at_tick,
                size_tokens=size_tokens,
                content=content,
            )

    def _commit_cas_impl(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        expected_version: int,
        content_hash: str,
        issued_at_tick: int = 0,
        size_tokens: int | None = None,
        content: bytes | str | None = None,
    ) -> tuple[Artifact, list[InvalidationSignal]] | ConflictDetail:
        """Optimistic-concurrency commit via an atomic version-checked CAS.

        The OCC counterpart to :meth:`commit` (plan Unit 3, R1–R4/R6). Unlike
        ``commit`` — which requires the caller to already hold EXCLUSIVE/MODIFIED
        from a pessimistic ``write()`` acquire — ``commit_cas`` lets a SHARED (or
        INVALID) caller commit *only if* ``expected_version`` still matches the
        registry's current version and no *pessimistic* peer holds M/E. The
        winner is elected by the registry's serialized ``BEGIN IMMEDIATE``, not a
        lock on the acquire, so two concurrent OCC writers cannot both land the
        same version.

        This method owns the **D4 precondition layer** the registry deliberately
        omits (the registry only does the version/holder CAS):

        - artifact must exist (``_require_artifact`` → ``CoherenceError``);
        - the caller must NOT be mid-transient (``get_agent_transient`` is
          ``None``, else ``CoherenceError``);
        - the caller's MESI state must be SHARED or INVALID — a MODIFIED/EXCLUSIVE
          holder is an *acquired* pessimistic writer and must use plain
          :meth:`commit` (rejected with a ``CoherenceError`` pointing there).

        Three-outcome discrimination of the registry result (plan R2):

        - :class:`CasCorruption` (``expected_version > current``) → raise
          ``CoherenceError`` — corruption / a second coordinator on the store,
          non-retryable.
        - :class:`ConflictDetail` (``version_mismatch`` / ``other_holder`` /
          ``stale_read_generation``) → returned UNCHANGED, with **no mutation
          and no invalidation signals** (it is a typed return, never an
          exception). ``stale_read_generation`` is the read-generation fence:
          the committer's captured claim was superseded by a sweep reclamation;
          retry-eligible via reacquire + fresh read.
        - WIN ``(updated_artifact, invalidated_ids)`` → the registry has already
          done the peer-invalidation + committer S/I→SHARED transition
          atomically (the OCC writer holds no grant, so it ends SHARED — which
          keeps a subsequent commit_cas by the same caller eligible past the D4
          precondition below); this method only builds the matching
          :class:`InvalidationSignal` list (mirroring ``commit``'s shape),
          re-validates single-writer, and returns ``(updated, signals)``.

        ``content`` is the winning body, threaded to the registry so the
        in-memory (library) path advances ``record.content`` on a WIN — a peer
        re-fetch then reads the winner's NEW content, not the stale pre-CAS body.
        The cross-process / sqlite path passes ``None`` (it stores no content).

        Returns:
            ``(updated_artifact, signals)`` on a winning commit, or a
            :class:`ConflictDetail` on a retry-eligible conflict.

        Raises:
            OccCallerTransientError: the caller is mid-transient (a peer
                invalidated it between read and CAS) — a retry-eligible subclass
                of ``CoherenceError`` carrying the stable wire reason
                :data:`~ccs.core.exceptions.OCC_CALLER_TRANSIENT_REASON`.
            CoherenceError: artifact missing, caller in M/E (use ``commit``), or
                the registry reported corruption (all non-retryable).
        """
        artifact = self._require_artifact(artifact_id)

        if self.registry.get_agent_transient(artifact_id, agent_id) is not None:
            # Retry-eligible: a peer invalidated the caller between its read and
            # this CAS, leaving an invalidation transient. Typed so the wire
            # reason stays stable independent of this human message (AC2).
            raise OccCallerTransientError(
                f"commit_cas_not_allowed agent={agent_id} artifact={artifact_id} "
                f"reason={OCC_CALLER_TRANSIENT_REASON}"
            )

        agent_state = self.registry.get_agent_state(artifact_id, agent_id)
        if agent_state in {MESIState.EXCLUSIVE, MESIState.MODIFIED}:
            raise CoherenceError(
                f"commit_cas_not_allowed agent={agent_id} artifact={artifact_id} "
                f"state={agent_state} reason=occ_is_shared_or_invalid_only "
                f"(use commit() for an EXCLUSIVE/MODIFIED holder)"
            )

        result = self.registry.commit_cas(
            artifact_id,
            agent_id,
            expected_version=expected_version,
            content_hash=content_hash,
            size_tokens=size_tokens,
            content=content,
            tick=issued_at_tick,
        )

        if isinstance(result, CasCorruption):
            raise CoherenceError(
                f"commit_cas_corruption agent={agent_id} artifact={artifact_id} "
                f"expected_version={expected_version} "
                f"current_version={result.current_version} "
                f"(expected > current — corruption or multi-coordinator violation)"
            )
        if isinstance(result, ConflictDetail):
            # Typed retry-eligible conflict: no mutation happened in the
            # registry, so emit no invalidation signals and surface it as-is.
            return result

        updated, invalidated_ids = result
        # Defense-in-depth: the CAS computed N+1 atomically; assert it did not
        # regress (NOT the concurrency guard — that was the version check).
        check_monotonic_version(artifact.version, updated.version)
        signals = [
            InvalidationSignal(
                artifact_id=artifact_id,
                new_version=updated.version,
                issued_at_tick=issued_at_tick,
                issuer_agent_id=agent_id,
            )
            for _ in invalidated_ids
        ]
        self._validate_single_writer(artifact_id)
        return updated, signals

    def commit_all(
        self,
        *,
        agent_id: UUID,
        writes: Mapping[UUID, CommitAllEntry],
        issued_at_tick: int = 0,
        abort: threading.Event | None = None,
    ) -> tuple[MultiCommitResult, list[InvalidationSignal]] | MultiCommitConflict:
        """Atomic multi-artifact publish (SB-18) under the A6 abort guard.

        The batch analog of :meth:`commit_cas` — all-or-nothing over the caller's
        write-set. A single check-point at the registry write lock (abort_guard);
        the registry does CHECK-all / total-apply, and this method owns the D4
        precondition layer + the ``InvalidationSignal`` construction. The signals
        are RETURNED for the caller to publish to the event bus AFTER the commit
        (broadcast-after-commit — never mid-batch).
        """
        with self.registry.abort_guard(abort):
            return self._commit_all_impl(
                agent_id=agent_id, writes=writes, issued_at_tick=issued_at_tick
            )

    def _commit_all_impl(
        self,
        *,
        agent_id: UUID,
        writes: Mapping[UUID, CommitAllEntry],
        issued_at_tick: int = 0,
    ) -> tuple[MultiCommitResult, list[InvalidationSignal]] | MultiCommitConflict:
        if not writes:
            raise CoherenceError("commit_all requires a non-empty write-set")
        # D4 precondition layer (per member, all-or-nothing): each artifact must
        # exist, the caller must not be mid-transient, and must be SHARED/INVALID
        # (an M/E holder is a pessimistic writer — use commit()). One failing member
        # refuses the WHOLE batch before any mutation.
        for artifact_id in writes:
            self._require_artifact(artifact_id)
            if self.registry.get_agent_transient(artifact_id, agent_id) is not None:
                raise OccCallerTransientError(
                    f"commit_all_not_allowed agent={agent_id} artifact={artifact_id} "
                    f"reason={OCC_CALLER_TRANSIENT_REASON}"
                )
            state = self.registry.get_agent_state(artifact_id, agent_id)
            if state in {MESIState.EXCLUSIVE, MESIState.MODIFIED}:
                raise CoherenceError(
                    f"commit_all_not_allowed agent={agent_id} artifact={artifact_id} "
                    f"state={state} reason=occ_is_shared_or_invalid_only "
                    f"(use commit() for an EXCLUSIVE/MODIFIED holder)"
                )

        result = self.registry.commit_all(agent_id, writes, tick=issued_at_tick)

        if isinstance(result, CasCorruption):
            raise CoherenceError(
                f"commit_all_corruption agent={agent_id} "
                f"current_version={result.current_version} (a member's "
                f"expected_version > current — corruption or multi-coordinator violation)"
            )
        if isinstance(result, MultiCommitConflict):
            # Typed all-or-nothing HELD: nothing mutated, no invalidation signals.
            return result

        # WIN: one InvalidationSignal per (member artifact, invalidated peer),
        # published to the event bus AFTER the commit (broadcast-after-commit).
        signals: list[InvalidationSignal] = []
        for art_id, new_version in result.versions.items():
            for _peer in result.invalidated.get(art_id, ()):
                signals.append(
                    InvalidationSignal(
                        artifact_id=art_id,
                        new_version=new_version,
                        issued_at_tick=issued_at_tick,
                        issuer_agent_id=agent_id,
                    )
                )
            self._validate_single_writer(art_id)
        return result, signals

    def session_commit_all(
        self,
        session_token: str,
        writes: Mapping[UUID, "tuple[bytes | str, int | None]"],
        *,
        caller: UUID,
        issued_at_tick: int = 0,
        abort: threading.Event | None = None,
    ) -> (
        tuple[MultiCommitResult, list[InvalidationSignal]]
        | MultiCommitConflict
        | SessionCommitRejection
    ):
        """Atomic multi-artifact publish against a session's PINNED cut (SB-18, R5).

        The thin session convenience over :meth:`commit_all`: each member's
        ``expected_version`` is sourced from ``cut[artifact_id]`` (the version
        pinned at ``begin_session``), so the comparands all come from ONE
        serialization point — the only fractured-read-safe path for size > 1. It
        adds NO atomicity logic; the commit rides ``commit_all`` under the same
        session-derived, fence-claimless committer identity as ``session_commit``
        (admit-on-absent). ``writes`` maps each artifact to ``(content, size_tokens)``.

        Validation mirrors ``session_commit`` (all-or-nothing): a foreign caller or
        owned-but-pinless token raises ``SessionInvalidated``; a missing cut →
        ``session_not_found`` / ``session_invalidated``; a member NOT in the cut →
        ``artifact_not_in_cut`` (the whole batch is rejected). Outcome maps
        ``commit_all`` verbatim (WIN → ``(result, signals)``; HELD →
        ``MultiCommitConflict``; corruption RAISES).
        """
        if not writes:
            raise CoherenceError("session_commit_all requires a non-empty write-set")
        epoch = self.registry.coordinator_epoch
        cut = self.registry.get_session_cut(session_token)
        self._validate_session_owner(session_token, caller, cut)
        if cut is None:
            return SessionCommitRejection(
                reason=self._classify_no_cut_reason(session_token),
                artifact_id=next(iter(writes)),
                coordinator_epoch=epoch,
            )
        for artifact_id in writes:
            if artifact_id not in cut:
                return SessionCommitRejection(
                    reason=SESSION_ARTIFACT_NOT_IN_CUT_REASON,
                    artifact_id=artifact_id,
                    coordinator_epoch=epoch,
                )

        committer_id = self._session_committer_id(session_token)
        batch = {
            artifact_id: CommitAllEntry(
                expected_version=cut[artifact_id],
                content_hash=compute_content_hash(content),
                size_tokens=size_tokens,
                content=content,
            )
            for artifact_id, (content, size_tokens) in writes.items()
        }
        return self.commit_all(
            agent_id=committer_id, writes=batch, issued_at_tick=issued_at_tick, abort=abort
        )

    def create_workspace_checkpoint(
        self,
        *,
        name: str,
        owner: UUID,
        members: "Sequence[CheckpointMember]",
        window_min: float,
        window_max: float,
        issued_at_tick: int = 0,
        abort: threading.Event | None = None,
    ) -> CheckpointRecord:
        """Persist ONE workspace-checkpoint manifest (WV plan Unit 3 / R1–R2).

        The single registration point for the capture engine
        (``ccs.adapters.workspace.WorkspaceVersioner``): mints the
        ``checkpoint_id`` SERVER-SIDE (a caller never names its own id) and
        hands the header + every member row to the Unit-2 registry API
        (:meth:`~ccs.coordinator.registry_protocol.RegistryBase.create_checkpoint`),
        which lands them in ONE transaction — owner metadata included, all rows
        or none. A raise from the registry therefore leaves NO partial
        manifest; the adapter maps it to its typed
        ``CheckpointPersistFailed``.

        Validation fails closed BEFORE minting anything: a blank name, an
        absent owner, an empty member set, or an inverted window raise
        ``ValueError`` (an empty manifest describes nothing; the window
        endpoints come from the capture pass and can never invert unless the
        caller is buggy).

        ``abort`` threads into :meth:`registry.abort_guard` (the A6
        session-commit lesson — every mutating path threads it): a
        watchdog-timed-out ``/workspace/checkpoint`` request fails closed at
        the registry write lock instead of landing a manifest AFTER the client
        already received the degraded ``checkpoint_unconfirmed`` response.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("create_workspace_checkpoint requires a non-empty name")
        if owner is None:
            raise ValueError(
                "create_workspace_checkpoint requires an owner: an ownerless "
                "manifest is unrepresentable (fail-closed)"
            )
        member_rows = list(members)
        if not member_rows:
            raise ValueError(
                "create_workspace_checkpoint requires at least one member row "
                "(an empty manifest describes nothing)"
            )
        if window_max < window_min:
            raise ValueError(
                f"inverted capture window: window_max={window_max!r} < "
                f"window_min={window_min!r}"
            )
        record = CheckpointRecord(
            checkpoint_id=str(uuid4()),
            name=name,
            owner=owner,
            created_at=float(issued_at_tick),
            created_at_tick=issued_at_tick,
            window_min=float(window_min),
            window_max=float(window_max),
        )
        with self.registry.abort_guard(abort):
            self.registry.create_checkpoint(record, member_rows)
        return record

    def get_workspace_checkpoint(self, checkpoint_id: str) -> CheckpointRecord | None:
        """The checkpoint header (durable), or ``None`` when unknown.

        The restore engine's pre-flight read (WV plan Unit 4 / R3): ``None``
        maps to its typed ``CheckpointUnknown`` refusal — the service never
        raises for an unknown id on the read side (the header getter tells
        known from unknown; the member getter below returns ``[]`` either way).
        """
        return self.registry.get_checkpoint(checkpoint_id)

    def get_workspace_checkpoint_members(
        self, checkpoint_id: str
    ) -> "list[CheckpointMember]":
        """The manifest's member rows (durable), ordered by ``member_path``.

        THE restore engine's member source (WV plan Unit 4 / R3): legs are
        driven from these durable rows — never from an in-memory capture
        return — so a fresh engine can resume a crashed restore from exactly
        the state the registry holds. Empty list for an unknown checkpoint.
        """
        return self.registry.get_checkpoint_members(checkpoint_id)

    def set_workspace_checkpoint_restore_status(
        self,
        checkpoint_id: str,
        status: str,
        *,
        updated_at: float,
        abort: threading.Event | None = None,
    ) -> None:
        """Update the checkpoint-level restore status (durable, crash-visible).

        The vocabulary is the SERVICE layer's (the registry stores the string):
        only :data:`~ccs.core.exceptions.RESTORE_STATUSES` pass — an unknown
        status is a caller bug and fails closed BEFORE any write (a misspelled
        status would silently orphan a crash-resume that matches by identity).
        ``abort`` threads into :meth:`registry.abort_guard` (the A6 lesson:
        every mutating path threads it). Raises ``KeyError`` for an unknown
        checkpoint.
        """
        if status not in RESTORE_STATUSES:
            raise ValueError(
                f"unknown restore status {status!r}: the closed vocabulary is "
                f"{sorted(RESTORE_STATUSES)} (fail-closed — an unknown status "
                "would orphan crash-resume, which matches by identity)"
            )
        with self.registry.abort_guard(abort):
            self.registry.set_checkpoint_restore_status(
                checkpoint_id, status, updated_at=updated_at
            )

    def set_workspace_checkpoint_member_restore(
        self,
        checkpoint_id: str,
        member_path: str,
        *,
        restore_outcome: str | None,
        deleted_at_restore: float | None = None,
        abort: threading.Event | None = None,
    ) -> None:
        """Record one member's durable restore progress (both columns written).

        ``restore_outcome`` must be one of the closed
        :data:`~ccs.core.exceptions.RESTORE_MEMBER_OUTCOMES` (or ``None`` — the
        explicit reset a new run may write); anything else fails closed BEFORE
        the write, because crash-resume decides skip-vs-redrive by matching
        this value against the closed set — an unvetted string would make a
        non-terminal member look terminal. ``abort`` threads into
        :meth:`registry.abort_guard`. Raises ``KeyError`` for an unknown
        (checkpoint, member) pair.
        """
        if restore_outcome is not None and restore_outcome not in RESTORE_MEMBER_OUTCOMES:
            raise ValueError(
                f"unknown restore outcome {restore_outcome!r}: the closed "
                f"vocabulary is {sorted(RESTORE_MEMBER_OUTCOMES)} (fail-closed "
                "— crash-resume classifies terminality by identity against it)"
            )
        with self.registry.abort_guard(abort):
            self.registry.set_checkpoint_member_restore(
                checkpoint_id,
                member_path,
                restore_outcome=restore_outcome,
                deleted_at_restore=deleted_at_restore,
            )

    def list_workspace_checkpoints(self) -> "list[CheckpointRecord]":
        """Every checkpoint header, ordered ``(created_at, checkpoint_id)``.

        The pin engine's cross-checkpoint read (WV plan Unit 6 / R9): before
        a checkpoint release drops an S3 legal hold, it scans the OTHER
        checkpoints' members for another ``held`` pin of the same
        ``(member_path, native_token)`` — a shared hold must survive the
        first checkpoint's release. Also the Unit-8 ``list`` verb's source.
        """
        return self.registry.list_checkpoints()

    def set_workspace_checkpoint_member_pin(
        self,
        checkpoint_id: str,
        member_path: str,
        *,
        pin_state: str,
        restore_tier: str | None = None,
        abort: threading.Event | None = None,
    ) -> None:
        """Record one member's durable pin state (WV plan Unit 6 / R9).

        ``pin_state`` must be one of the closed
        :data:`~ccs.core.exceptions.PIN_STATES`; ``restore_tier`` (when given)
        rewrites the member's restore tier IN THE SAME registry write — the
        loud tier downgrade a failed/released pin must land together with its
        pin state (two writes could crash apart and leave a ``restorable``
        member with no backing pin). Both vocabularies fail closed BEFORE any
        write. ``abort`` threads into :meth:`registry.abort_guard` (the A6
        lesson: every mutating path threads it). Raises ``KeyError`` for an
        unknown (checkpoint, member) pair.
        """
        if pin_state not in PIN_STATES:
            raise ValueError(
                f"unknown pin state {pin_state!r}: the closed vocabulary is "
                f"{sorted(PIN_STATES)} (fail-closed — the pair "
                "(restore_tier, pin_state) is the honesty surface and an "
                "unvetted string would make it unreadable)"
            )
        if restore_tier is not None and restore_tier not in {
            tier.value for tier in RestoreTier
        }:
            raise ValueError(
                f"unknown restore tier {restore_tier!r}: the closed vocabulary "
                f"is {sorted(tier.value for tier in RestoreTier)} (fail-closed)"
            )
        with self.registry.abort_guard(abort):
            self.registry.set_checkpoint_member_pin(
                checkpoint_id,
                member_path,
                pin_state=pin_state,
                restore_tier=restore_tier,
            )

    def adjust_workspace_checkpoint_pin_refcount(
        self,
        checkpoint_id: str,
        delta: int,
        *,
        abort: threading.Event | None = None,
    ) -> int:
        """Atomically adjust a checkpoint's pin refcount; the new value returned.

        The Unit-6 bookkeeping counter: +1 per pin the checkpoint establishes,
        -1 per pin the checkpoint release drops. The registry refuses a result
        below zero (``ValueError`` — a release without a matching pin is a
        bookkeeping bug, fail-closed). ``abort`` threads into
        :meth:`registry.abort_guard`. Raises ``KeyError`` for an unknown
        checkpoint.
        """
        with self.registry.abort_guard(abort):
            return self.registry.adjust_checkpoint_pin_refcount(checkpoint_id, delta)

    def register_workspace_restore(
        self,
        *,
        checkpoint_id: str,
        controller: UUID,
        writes: "Sequence[WorkspaceRestoreWrite]",
        issued_at_tick: int = 0,
        abort: threading.Event | None = None,
    ) -> WorkspaceRegistrationResult:
        """Register a restore run's WRITTEN file members with the coordinator —
        all-or-nothing (WV plan Unit 5 / R4–R5).

        The coordinator half of the plan's restore-registration split. The
        caller (the restore engine's registration seam) hands the written FILE
        members only; the other member classes never reach this method by
        design:

        - **S3 members** — the coordinator holds no artifact identity for a
          BYO-substrate member (the substrate owns identity; their
          ``artifact_id`` is NULL in the manifest by design), so their
          registration is the manifest-side outcome row alone;
        - **deleted members** — recorded manifest-side
          (``deleted_at_restore``); ``commit_all`` has no delete semantics;
        - **an empty ``writes``** — answers the typed ``empty_write_set``
          result; ``commit_all`` is NEVER called (it raises on an empty set).

        Per write, hash-only (never content bytes into the coordinator):

        1. ``member_path`` resolves to the coordinator artifact — via the
           durable registry's ``resolve_or_register`` (first observation mints
           the artifact at version 1 carrying the FINGERPRINT as its content
           hash: for a path the coordinator never saw, the mint IS the
           registration), or a name-scan + hash-only first-observation seed on
           the in-memory registry (content ``""`` — the coordinator stays
           metadata-only; parity: both backends mint at version 1 with the
           fingerprint).
        2. Members whose artifact ALREADY carries the fingerprint are
           ``skipped`` — the idempotency filter: a crash-resumed registration
           whose prior run's ``commit_all`` landed re-answers without a second
           version bump (exactly-once), and a member whose leg rode a
           coordinator-connected volume (its ``write_cas_at`` already
           committed) is never double-registered.
        3. The rest form ONE :meth:`commit_all` batch — ``expected_version``
           read here per member, ``content_hash`` = the fingerprint,
           ``content=None`` — under the D4 preconditions (the CONTROLLER must
           be S/I and not mid-transient; those raises propagate as caller
           bugs / retry-eligible signals, matching ``commit_all``). WIN →
           ``committed`` with per-path NEW versions (defense-in-depth
           ``check_monotonic_version`` against the pre-read version: restore
           registers FORWARD, old bytes at a NEW version, never a decrement)
           plus the invalidation signals for broadcast-after-commit (peers
           holding S on a registered member are invalidated atomically by the
           registry; the signals let a server surface it). HELD →
           ``refused`` mapping each failing member path to its own typed
           :class:`ConflictDetail` — including ``stale_read_generation``, the
           fence rejecting a superseded controller's late apply. Nothing
           mutates on a refusal (all-or-nothing).

        The per-member ``expected_version`` reads are sequential (no pinned
        cut): a racing commit between the read and the batch simply HELDs the
        batch ``version_mismatch`` — the CAS arbitrates; the caller re-drives
        bounded. ``abort`` threads into ``commit_all`` → ``abort_guard`` (the
        A6 lesson: every mutating path threads it).

        Raises:
            CheckpointUnknown: ``checkpoint_id`` names no persisted manifest.
            ValueError: blank/duplicate member paths or a blank fingerprint
                (caller bugs, refused before any resolution).
        """
        if self.registry.get_checkpoint(checkpoint_id) is None:
            raise CheckpointUnknown(checkpoint_id)
        entries = list(writes)
        seen: set[str] = set()
        for write in entries:
            if not write.member_path or not write.member_path.strip():
                raise ValueError("register_workspace_restore: blank member_path")
            if not write.fingerprint or not write.fingerprint.strip():
                raise ValueError(
                    f"register_workspace_restore: member {write.member_path!r} "
                    "carries no fingerprint (a written member always has one)"
                )
            if write.member_path in seen:
                raise ValueError(
                    f"register_workspace_restore: duplicate member_path "
                    f"{write.member_path!r} in the write-set"
                )
            seen.add(write.member_path)
        if not entries:
            return WorkspaceRegistrationResult(
                checkpoint_id=checkpoint_id,
                status=WORKSPACE_REGISTRATION_EMPTY,
                detail=(
                    "empty write-set: no written file members to register — "
                    "commit_all was never called (it raises on an empty set)"
                ),
            )

        skipped: list[str] = []
        batch: dict[UUID, CommitAllEntry] = {}
        path_by_artifact: dict[UUID, str] = {}
        pre_versions: dict[UUID, int] = {}
        for write in entries:
            artifact_id = self._resolve_workspace_member_artifact(
                write.member_path, write.fingerprint
            )
            artifact = self.registry.get_artifact(artifact_id)
            if artifact is None:  # pragma: no cover - resolve just minted/found it
                raise CoherenceError(
                    f"register_workspace_restore: artifact {artifact_id} for "
                    f"member {write.member_path!r} vanished mid-registration"
                )
            if artifact.content_hash == write.fingerprint:
                # Already registered at the manifest fingerprint (a prior run's
                # landed commit, a coordinator-connected leg, or the first-
                # observation mint above): exactly-once, no second bump.
                skipped.append(write.member_path)
                continue
            path_by_artifact[artifact_id] = write.member_path
            pre_versions[artifact_id] = artifact.version
            batch[artifact_id] = CommitAllEntry(
                expected_version=artifact.version,
                content_hash=write.fingerprint,
            )

        if not batch:
            return WorkspaceRegistrationResult(
                checkpoint_id=checkpoint_id,
                status=WORKSPACE_REGISTRATION_EMPTY,
                detail=(
                    "every written member is already registered at its manifest "
                    "fingerprint — commit_all was never called"
                ),
                skipped=tuple(skipped),
            )

        out = self.commit_all(
            agent_id=controller,
            writes=batch,
            issued_at_tick=issued_at_tick,
            abort=abort,
        )
        if isinstance(out, MultiCommitConflict):
            # All-or-nothing HELD: nothing mutated, no signals. Typed per-path
            # reasons — stale_read_generation is the fence rejecting a
            # superseded controller's late apply.
            return WorkspaceRegistrationResult(
                checkpoint_id=checkpoint_id,
                status=WORKSPACE_REGISTRATION_REFUSED,
                detail=(
                    "commit_all HELD the registration batch (all-or-nothing: "
                    "no member registered); per-member typed reasons attached"
                ),
                skipped=tuple(skipped),
                refused={
                    path_by_artifact[art_id]: conflict
                    for art_id, conflict in out.per_artifact.items()
                },
            )
        result, signals = out
        versions: dict[str, int] = {}
        for art_id, new_version in result.versions.items():
            # Defense-in-depth (the commit_cas mirror): the CAS computed N+1
            # atomically; assert the registration never regressed a version —
            # restore is a FORWARD commit carrying old bytes.
            check_monotonic_version(pre_versions[art_id], new_version)
            versions[path_by_artifact[art_id]] = new_version
        return WorkspaceRegistrationResult(
            checkpoint_id=checkpoint_id,
            status=WORKSPACE_REGISTRATION_COMMITTED,
            detail=(
                f"registered {len(versions)} written file member(s) via one "
                "all-or-nothing commit_all (hash-only; peers invalidated "
                "atomically, signals returned for broadcast-after-commit)"
            ),
            versions=versions,
            skipped=tuple(skipped),
            signals=tuple(signals),
        )

    def _resolve_workspace_member_artifact(
        self, member_path: str, fingerprint: str
    ) -> UUID:
        """Resolve a file member path to its coordinator artifact id, hash-only.

        Durable registry: :meth:`SqliteExtended.resolve_or_register` (one
        ``BEGIN IMMEDIATE``; a first observation mints the artifact at version
        1 carrying the fingerprint). In-memory registry (no
        ``resolve_or_register`` — it is SqliteExtended-only): a name scan over
        the registered artifacts, with the SAME first-observation semantics on
        a miss — an ``Artifact(name, version=1, content_hash=fingerprint)``
        seeded with EMPTY content (hash-only: member bytes never enter the
        coordinator; parity with the sqlite mint, which stores no body either).

        The in-memory scan+mint is a compound check-then-act, so it runs under
        ``_workspace_mint_lock`` — two concurrent first observations of one
        ``member_path`` serialize and mint ONE artifact, keeping the parity
        claim with sqlite's UNIQUE-backed ``resolve_or_register`` honest
        instead of GIL-dependent.
        """
        if isinstance(self.registry, SqliteExtended):
            return self.registry.resolve_or_register(member_path, fingerprint)
        with self._workspace_mint_lock:
            for artifact_id in self.registry.artifact_ids():
                artifact = self.registry.get_artifact(artifact_id)
                if artifact is not None and artifact.name == member_path:
                    return artifact_id
            minted = Artifact(name=member_path, version=1, content_hash=fingerprint)
            self.registry.register_artifact(minted, "")
            return minted.id

    def workspace_member_registered(self, member_path: str, fingerprint: str) -> bool:
        """READ-ONLY: is ``member_path`` currently registered at ``fingerprint``?

        True iff a coordinator artifact named ``member_path`` exists AND its
        current ``content_hash`` equals ``fingerprint`` — the same predicate
        :meth:`register_workspace_restore`'s idempotency filter applies,
        exposed as a pure read so the restore engine can REBUILD the
        registration answer for an already-``concluded`` checkpoint (WV Unit
        5: a refused-terminal registration must never re-read as
        ``None``-means-fine on a re-restore). Never resolves, never mints,
        never mutates: an unknown path answers ``False`` (nothing registered).
        A plain name scan on BOTH registry backends — sqlite's
        ``resolve_or_register`` is mint-on-miss and therefore unusable for a
        read — matching the first-match semantics of
        :meth:`_resolve_workspace_member_artifact`'s in-memory scan.
        """
        for artifact_id in self.registry.artifact_ids():
            artifact = self.registry.get_artifact(artifact_id)
            if artifact is not None and artifact.name == member_path:
                return artifact.content_hash == fingerprint
        return False

    def invalidate(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        new_version: int,
        issuer_agent_id: UUID,
        issued_at_tick: int,
        abort: threading.Event | None = None,
    ) -> InvalidationSignal | None:
        """Apply invalidation for one agent under the A6 abort guard and the
        NoZombieRevoke pin.

        Wrapped in :meth:`registry.abort_guard` (finding A6): a late
        session-stop release whose handler already timed out aborts here rather
        than revoking a grant the registry has since handed to another session.
        That guard is a *time* predicate on the local watchdog, so it covers
        only the callers that thread an ``abort`` Event; a PEER-issued
        invalidation delivered late through the event bus threads none.

        The version pin covers the rest: an invalidation whose target has since
        observed a version at least as new as the one announced is dropped as
        obsolete, and returns ``None`` (see
        :meth:`_revoke_is_superseded` for the two guards that keep the drop
        conservative). Self-issued releases are never pinned.
        """
        with self.registry.abort_guard(abort):
            return self._invalidate_impl(
                agent_id=agent_id,
                artifact_id=artifact_id,
                new_version=new_version,
                issuer_agent_id=issuer_agent_id,
                issued_at_tick=issued_at_tick,
            )

    def _invalidate_impl(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        new_version: int,
        issuer_agent_id: UUID,
        issued_at_tick: int,
    ) -> InvalidationSignal | None:
        if not self.registry.has_artifact(artifact_id):
            return None
        if self._revoke_is_superseded(
            agent_id=agent_id,
            artifact_id=artifact_id,
            new_version=new_version,
            issuer_agent_id=issuer_agent_id,
        ):
            return None
        self.registry.set_agent_state(
            artifact_id, agent_id, MESIState.INVALID, trigger="invalidate", tick=issued_at_tick
        )
        self.registry.clear_agent_transient(artifact_id, agent_id)
        return InvalidationSignal(
            artifact_id=artifact_id,
            new_version=new_version,
            issued_at_tick=issued_at_tick,
            issuer_agent_id=issuer_agent_id,
        )

    def _revoke_is_superseded(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        new_version: int,
        issuer_agent_id: UUID,
    ) -> bool:
        """NoZombieRevoke: is this PEER-issued invalidation obsolete for its
        target? (``NoZombieRevoke``; see ``formal/tla/ZombieRevoke.tla``.)

        An invalidation announces "the artifact reached ``new_version``; the
        copy you hold is behind it". It is minted inside the issuer's registry
        lock and applied later, outside it — so between mint and apply the
        target may have been reclaimed, re-acquired, and re-read. Applying the
        stale signal then revokes a grant established AFTER the signal was
        issued: Temporal's late-release-Signal shape, which they close with
        signal pinning.

        The pin is the target's own ``last_observed_version`` (SB-10 R6/R7) —
        recorded atomically with every non-INVALID grant, preserved across a
        transition to INVALID. A target that has already observed a version at
        least as new as the one announced is not behind this signal, so the
        signal cannot be authority over its claim.

        Two guards keep the drop conservative, because wrongly DROPPING an
        invalidation is far worse than wrongly applying one (it would leave a
        genuinely stale copy marked valid — the stale-read → write hole this
        layer exists to close):

        1. **Self-issued releases are always honoured.** ``issuer_agent_id ==
           agent_id`` is an agent giving back its OWN claim (a post-edit
           failure, a session-stop release, an operator drain), never a
           cross-agent revoke. It is not pinned.
        2. **Only a target currently holding a claim is pinned.** An already-
           INVALID target is pinned by nothing: applying is a state no-op, and
           the call still has to run — ``_write_impl`` / ``_commit_impl``
           invalidate peers directly and leave their SIA/EIA transient set for
           the bus-delivered invalidation to clear. Skipping that would strand
           the transient and stall both the sweep and the peer's next
           ``commit_cas``.

        Admit-on-absent, matching the commit-path fence: a target with NO
        recorded observation is never dropped — absence is not evidence of
        freshness.
        """
        if issuer_agent_id == agent_id:
            return False
        if self.registry.get_agent_state(artifact_id, agent_id) == MESIState.INVALID:
            return False
        observed = self.registry.last_observed_version_for(artifact_id, agent_id)
        return observed is not None and observed >= new_version

    def delete(
        self,
        *,
        agent_id: UUID,
        artifact_id: UUID,
        issued_at_tick: int = 0,
    ) -> list[InvalidationSignal]:
        """Remove artifact and emit invalidation signals to all non-INVALID holders.

        Does not require the caller to hold EXCLUSIVE or MODIFIED state first.
        Returns [] when the artifact is absent (silent no-op for the caller).

        The existence check, the holder enumeration and the removal run under
        one :meth:`registry.abort_guard` hold (U5), so the emitted signals
        match the holders that actually existed at the instant of removal — a
        grant established between the enumeration and the remove can neither
        be missed nor phantom-signalled.
        """
        with self.registry.abort_guard():
            if not self.registry.has_artifact(artifact_id):
                return []
            artifact = self._require_artifact(artifact_id)
            signals = [
                InvalidationSignal(
                    artifact_id=artifact_id,
                    new_version=artifact.version,
                    issued_at_tick=issued_at_tick,
                    issuer_agent_id=agent_id,
                )
                for holder_id, state in self.registry.get_state_map(artifact_id).items()
                if state != MESIState.INVALID
            ]
            self.registry.remove_artifact(artifact_id)
        return signals

    def record_heartbeat(self, *, agent_id: UUID, now_tick: int) -> None:
        """Record an agent's heartbeat tick (R12: max(prev, incoming))."""
        if now_tick < 0:
            raise ValueError("now_tick must be >= 0")
        self.registry.record_heartbeat(agent_id, now_tick)

    def enforce_transient_timeouts(self, *, current_tick: int, timeout_ticks: int) -> int:
        """Force expired transient entries to INVALID as fail-safe recovery."""
        if timeout_ticks < 1:
            raise ValueError("timeout_ticks must be >= 1")

        expired = 0
        for artifact_id in self.registry.artifact_ids():
            for agent_id in list(self.registry.get_transient_map(artifact_id)):
                # Per-pair hold (U5 / KTD2): the walk's snapshot is stale by
                # the time a pair is visited, so the decision re-reads INSIDE
                # the hold and the write lands in the same hold. Held per pair,
                # not across the walk — a long scan must not block unrelated
                # reads for its whole duration.
                with self.registry.abort_guard():
                    if self.registry.get_agent_transient(artifact_id, agent_id) is None:
                        # Cleared since the snapshot (the grant completed);
                        # evicting now would destroy a settled entry.
                        continue
                    entered = self.registry.get_transient_tick(artifact_id, agent_id)
                    if entered is None:
                        continue
                    if (current_tick - entered) < timeout_ticks:
                        continue

                    # Conservative fail-safe: transient timeout always forces local invalidation.
                    self.registry.set_agent_state(
                        artifact_id, agent_id, MESIState.INVALID, trigger="timeout", tick=current_tick
                    )
                    self.registry.clear_agent_transient(artifact_id, agent_id)
                    expired += 1

        return expired

    def enforce_stable_grant_timeouts(
        self,
        *,
        current_tick: int,
        heartbeat_timeout_ticks: int,
        max_hold_ticks: int,
        on_reclaim: Optional[Callable[[UUID, UUID, str], None]] = None,
    ) -> int:
        """Reclaim stale stable-state (M∪E) grants whose holders are gone or over-held.

        Trigger order (first match wins):
          1. ``reclaim_heartbeat`` — agent's last heartbeat is older than
             ``heartbeat_timeout_ticks``, or the agent has never heartbeated.
          2. ``reclaim_max_hold`` — agent's grant is at least ``max_hold_ticks``
             old. Skipped if ``granted_at_tick`` is missing (defensive).

        Pairs with a non-empty transient slot are skipped so the transient sweep
        (which must run first) owns those entries — preserves R4 sweep ordering.

        ADV-004: ``on_reclaim`` is a per-reclamation callback the adapter
        uses to record a preemption notice for the victim agent — so when
        the victim's post-edit later arrives and fails CoherenceError, the
        F4 enrichment path can pop the notice and emit a "reclaimed by
        sweep" message rather than a generic error with no context.
        Library code remains preemption-notice-agnostic; the registry
        method is adapter-only (SqliteArtifactRegistry).

        KTD7 — ``on_reclaim`` runs while the per-pair registry hold is HELD:
        it must never block on other threads and may reach only into this
        registry's own (reentrant) methods. Both shipped callbacks qualify (a
        list append; ``record_preemption_notice``).

        Returns the number of grants reclaimed.
        """
        if heartbeat_timeout_ticks < 1:
            raise ValueError("heartbeat_timeout_ticks must be >= 1")
        if max_hold_ticks < 1:
            raise ValueError("max_hold_ticks must be >= 1")

        m_or_e = {MESIState.MODIFIED, MESIState.EXCLUSIVE}
        snapshot: list[tuple[UUID, UUID, MESIState]] = [
            (artifact_id, agent_id, mesi)
            for artifact_id in self.registry.artifact_ids()
            for agent_id, mesi in self.registry.get_state_map(artifact_id).items()
            if mesi in m_or_e
        ]

        reclaimed = 0
        for artifact_id, agent_id, _mesi in snapshot:
            # Per-pair hold (U5 / KTD2): everything from the state re-read to
            # the reclaiming write happens inside ONE registry hold, so the
            # decision cannot outlive the state it was made from -- the window
            # that let a reclaim land on a grant renewed after the reads, and
            # let the walk observe the mid-renewal "M/E holder with no
            # granted_at slot" state this method used to log as impossible.
            # Held per pair, not across the walk: a long sweep must not block
            # an unrelated artifact's reads for the whole scan.
            with self.registry.abort_guard():
                # The snapshot is stale by now; re-read the pair's state and
                # skip anything that is no longer a stable-state grant.
                if self.registry.get_agent_state(artifact_id, agent_id) not in m_or_e:
                    continue
                # Agents that entered transient since the snapshot are owned by
                # the transient sweep, not this one (R4).
                if self.registry.get_agent_transient(artifact_id, agent_id) is not None:
                    continue

                last_hb = self.registry.last_heartbeat_tick(agent_id)
                # Heartbeat uses `>=` to match max-hold's `>=` (review fix ADV-02).
                # An effective timeout of exactly heartbeat_timeout_ticks means a
                # grant is reclaimed when (current_tick - last_hb) reaches the
                # threshold, not the tick after. Matches the 'at least this many
                # ticks since last heartbeat' framing in CrashRecoveryConfig docs.
                heartbeat_stale = last_hb is None or (current_tick - last_hb) >= heartbeat_timeout_ticks

                if heartbeat_stale:
                    trigger = "reclaim_heartbeat"
                else:
                    granted_at = self.registry.granted_at_tick(agent_id, artifact_id)
                    if granted_at is None:
                        # M∪E holder without granted_at inside the hold — a real
                        # inconsistency now, not a mid-renewal read: skip and log
                        # so operators can investigate.
                        logger.warning(
                            "sweep: M/E holder has no granted_at slot; skipping max-hold check "
                            "agent=%s artifact=%s",
                            agent_id, artifact_id,
                        )
                        continue
                    if (current_tick - granted_at) >= max_hold_ticks:
                        trigger = "reclaim_max_hold"
                    else:
                        continue

                self.registry.set_agent_state(
                    artifact_id,
                    agent_id,
                    MESIState.INVALID,
                    trigger=trigger,
                    tick=current_tick,
                    content_hash=None,
                )
                self.registry.record_last_reclamation(agent_id, artifact_id, trigger, current_tick)
                # The callback fires once the reclaim is durable and BEFORE the
                # single-writer check: a check that raises still leaves this
                # pair reclaimed (each pair commits on its own), and a reclaim
                # the registry records must not go without its notice, count
                # or log line (#195).
                if on_reclaim is not None:
                    # Best-effort: a notice-recording failure must not stop the sweep
                    # (the reclamation itself already landed in the registry).
                    try:
                        on_reclaim(artifact_id, agent_id, trigger)
                    except Exception:  # noqa: BLE001 — telemetry surface, best-effort
                        logger.exception(
                            "on_reclaim callback raised for agent=%s artifact=%s trigger=%s",
                            agent_id, artifact_id, trigger,
                        )
                self._validate_single_writer(artifact_id)
                reclaimed += 1

        return reclaimed

    def _validate_single_writer(self, artifact_id: UUID) -> None:
        check_single_writer(self.registry.get_state_map(artifact_id))

    def _require_artifact(self, artifact_id: UUID) -> Artifact:
        artifact = self.registry.get_artifact(artifact_id)
        if artifact is None:
            raise CoherenceError(f"artifact_not_found artifact={artifact_id}")
        return artifact


def _invalidation_transient_for_state(state: MESIState) -> TransientState | None:
    if state == MESIState.SHARED:
        return TransientState.SIA
    if state == MESIState.EXCLUSIVE:
        return TransientState.EIA
    if state == MESIState.MODIFIED:
        return TransientState.MSA
    return None
