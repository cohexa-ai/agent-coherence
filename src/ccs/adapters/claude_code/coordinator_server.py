# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Local HTTP coordinator for the Claude Code coherence plugin (Unit 4).

stdlib HTTP server (KTD-4) that wraps :class:`CoordinatorService` driven
by :class:`SqliteArtifactRegistry`. Exposes seven endpoints behind
shared-secret Bearer auth + Host-header check (KTD-12):

- ``POST /hooks/pre-read``      — stale-read check; KTD-9 first-observation
- ``POST /hooks/pre-edit``      — acquire EXCLUSIVE; KTD-1 single-writer
- ``POST /hooks/post-edit``     — commit on success, release on failure
- ``POST /hooks/post-edit-cas`` — OCC commit (Unit 6); version-checked CAS,
  NO pre-edit acquire (the OCC writer stays S/I); fail-closed degrade
- ``POST /hooks/session-stop``  — KTD-11 release on end-of-turn
- ``POST /hooks/session-start`` — SB-10 post-compaction re-grounding payload
- ``POST /policy/track``        — Unit 6 CLI hot-add to tracked.yaml
- ``POST /policy/untrack``      — Unit 6 CLI hot-add to ignored.yaml
- ``GET  /status``              — Unit 6 status CLI
- ``POST /principal/claim``     — caller-principal mint (plan U4; Python-only)

Every route that names an acting identity (``session_id``) has a caller-
principal posture in :data:`_CALLER_PRINCIPAL_POSTURE` (plan U6): require-class
routes refuse a request naming a CLAIMED identity without the principal minted
for it, accept-class routes admit and count one, every class refuses a foreign
principal, the mint is neither.

Every handler:
- Verifies ``Authorization: Bearer <secret>`` (constant-time)
- Verifies ``Host`` header is localhost / 127.0.0.1 (DNS-rebind guard)
- Records the calling session's heartbeat (KTD-2)
- Runs the coordinator call under a 4s ThreadPoolExecutor timeout
  (handler-side watchdog — keeps us under the 5s hook timeout even when
  SQLite contention exceeds busy_timeout=2000); a caller-principal gate that
  must read the registry runs under the same deadline, before the call
- Converts ``CoherenceError`` to 200 ``{ok: false, reason}`` (NOT 500 — we
  want hooks to proceed gracefully on protocol violations, not block)
- Logs request/response at DEBUG, errors at WARNING

Lifecycle (spawn, port-file, idle-shutdown, sweep) lives in :mod:`lifecycle`
(Unit 5) and consumes ``CoordinatorHTTPServer.from_root(...)``.
"""

from __future__ import annotations

import base64
import http.server
import json
import logging
import os
import re
import socketserver
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable
from uuid import NAMESPACE_URL, UUID, uuid5

from ccs.adapters.claude_code import audit_log as _audit_log
from ccs.adapters.claude_code import hook_payloads as _payloads
from ccs.adapters.claude_code import session_audit_log as _session_audit
from ccs.adapters.claude_code.auth import (
    CALLER_PRINCIPAL_HEADER,
    assert_serve_transport_acknowledged,
    build_host_allowlist,
    ensure_secret,
    verify_bearer,
    verify_host,
)
from ccs.adapters.claude_code.bash_path_detector import detect_tracked_paths
from ccs.adapters.claude_code.policy import UNTRACK_STRICT_PATH_REASON, TrackedArtifactPolicy
from ccs.coordinator.registry_protocol import CheckpointMember
from ccs.coordinator.service import (
    CallerPrincipalUncached,
    CoordinatorService,
    mint_nonce_problem,
)
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.clock import monotonic_seconds
from ccs.core.exceptions import (
    CALLER_PRINCIPAL_ABSENT_REASON,
    CALLER_PRINCIPAL_FOREIGN_REASON,
    CHECKPOINT_UNKNOWN_REASON,
    HANDLER_FAILURE_REASON_PREFIX,
    HOLD_REASONS,
    HOLD_VERSION_UNCONFIRMED,
    OCC_CALLER_TRANSIENT_REASON,
    RESTORE_MEMBER_OUTCOMES,
    RESTORE_STATUSES,
    SESSION_INVALIDATED_REASON,
    STALE_READ_GENERATION_REASON,
    CallerPrincipalRefused,
    CheckpointUnknown,
    CoherenceError,
    OccCallerTransientError,
    SessionInvalidated,
    StaleReadGeneration,
    WatchdogAbandoned,
)
from ccs.core.fence import (
    classify_hold,
    confirmed_generation,
    coordinator_holds_content_claim,
)
from ccs.core.states import MESIState
from ccs.core.substrate import ArbitrationTier, RestoreTier
from ccs.core.types import (
    Artifact,
    ConflictDetail,
    DataPlaneDeferredRead,
    FenceComparands,
    MultiCommitConflict,
    SessionCommitRejection,
    SessionReadRejection,
    SnapshotSession,
    VersionedContent,
    VersionedReadRejection,
    WorkspaceRestoreWrite,
)

logger = logging.getLogger(__name__)


# Finding #30 — typed interface for the `req` parameter passed to all
# endpoint handlers. The concrete implementation lives inside the
# _make_handler_class closure as a BaseHTTPRequestHandler subclass; the
# Protocol lets handlers declare the interface they need without coupling
# to the concrete class or breaking the closure structure.
@runtime_checkable
class _RequestProtocol(Protocol):
    """Minimal interface every endpoint handler expects from `req`."""

    headers: Any  # http.client.HTTPMessage (Mapping-like)
    path: str
    #: ``(method, route path)`` the dispatcher matched in ``_ROUTES``, set
    #: before the handler runs; keys the caller-principal posture lookup.
    _route_key: tuple[str, str]
    #: ``time.monotonic()`` instant this request's handler watchdog expires, or
    #: ``None`` while nothing has started it. The dispatcher resets it; the
    #: caller-principal gate starts it when it must read the registry, and the
    #: work body then waits only for what is left (:func:`_watchdog_deadline`).
    _watchdog_deadline: float | None

    def _read_json(self) -> dict | None:
        """Read + parse the request body as JSON. Returns None on error."""
        ...

    def _json(self, status: int, body: dict) -> None:
        """Write a JSON response with the given HTTP status code."""
        ...


HANDLER_TIMEOUT_SEC = 4.0

# v0.2 Unit 4: window for the "Read strict-deny → Bash cat strict-deny on
# same (session, path)" route-around detector. Phase 0 saw the model
# retry-loop and route-around behaviors complete within ~10s; 30s
# accommodates slow runners + provides a margin for the operator-visible
# strict_mode_routed_around_via_bash_total counter to remain accurate.
STRICT_DENY_ROUTE_AROUND_WINDOW_SEC = 30.0

# v0.1.1 KTD-G concurrency limits per plugin docs/known-issues/
# 2026-05-17-watchdog-races.md A7 fix. Watchdog pool size × 2 is the
# upper bound on both (i) work queue depth before we reject with 503,
# and (ii) concurrent HTTP handler threads. Two layers because:
#   - The queue-depth gate (item 1) catches submit-time overflow.
#   - The handler semaphore (item 2) caps thread creation upstream of
#     the watchdog pool, preventing a session that's slow-rolling N
#     overlapping requests from starving the watchdog pool's queue.
# Both are independently effective; running both is defense in depth.
_WATCHDOG_POOL_SIZE = 4
WATCHDOG_QUEUE_LIMIT = _WATCHDOG_POOL_SIZE * 2
HANDLER_CONCURRENCY_LIMIT = _WATCHDOG_POOL_SIZE * 2
"""Each endpoint's coordinator call is bounded to 4s by the watchdog;
leaves 1s of margin under the 5s Claude Code hook timeout (KTD-12 / Unit 4)."""

IN_FLIGHT_DRAIN_TIMEOUT_SEC = 5.0
"""KTD-I (Unit 5 L2): ``shutdown()`` waits up to this many seconds for
in-flight handlers to complete before closing the SQLite registry. After
the deadline, any still-running handlers are abandoned — they may raise
``sqlite3.ProgrammingError`` and return HTTP 500 to their clients. Better
a 500 than a wedged shutdown (silent hang vs observable failure)."""

# ADV-004: a sentinel preempter UUID the stable-grant sweep uses when
# recording a preemption notice for an agent whose M/E grant it just
# reclaimed. The F4 enrichment in ``_handle_post_edit`` compares against
# this constant to distinguish "your grant was reclaimed by the sweep"
# from "your grant was preempted by another session" — both surface via
# the same notice table but communicate distinct failure modes to the
# model. UUID derived from a stable namespace string so it stays the
# same across processes, restarts, and instances.
SWEEP_RECLAMATION_PREEMPTER_ID: UUID = uuid5(
    NAMESPACE_URL, "ccs-coordinator-sweep:stable-grant-reclamation"
)

MAX_POLICY_PATHS_PER_REQUEST = 20
"""Cap on the number of paths /policy/track and /policy/untrack accept
in one request body (security-lens P1)."""

MAX_POLICY_YAML_BYTES = 64 * 1024
"""Cap on the resulting tracked.yaml / ignored.yaml file size (security-lens P1)."""

MAX_PATH_LEN = 1024
"""Cap on inbound path length to defend against memory/DoS and prose-injection
attacks. 1024 covers nested-deep paths in any realistic project."""

MAX_REQUEST_BODY_BYTES = 64 * 1024
"""R21 (Unit 6): hard cap on the Content-Length the HTTP server is willing
to read into memory. Matches MAX_POLICY_YAML_BYTES — generous for the
~1 KB hook payloads we actually expect, tight enough that a hostile or
buggy client cannot OOM the coordinator with a single oversized body.
Validated BEFORE rfile.read so we never allocate the offending buffer."""

MAX_SESSION_READ_SET_PATHS = 64
"""SB-17 Unit 8 (R14 wire mirror): cap on the number of read-set paths
``POST /session/begin`` accepts in one request, mirroring the service's
``SessionCapsConfig.max_read_set_cardinality`` default. A boundary reject
(loud 400) keeps a hostile client from forcing the path→id resolution loop
to walk an unbounded list before the service's own cap fires. The service
cap remains authoritative; this is defense-in-depth at the wire."""


def _resolve_coordinator_version() -> str:
    """KTD-J (Unit 8): surface the package version via /status so operators
    pasting status output into bug reports always include which build
    they're running. Reads ``ccs.__version__`` lazily so test fixtures
    that import this module before ``ccs`` is fully loaded don't crash."""
    try:
        from ccs import __version__ as _v
        return _v
    except Exception:  # pragma: no cover — defensive against import order
        return "unknown"


_COORDINATOR_VERSION: str = _resolve_coordinator_version()

_SESSION_ID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
                            re.IGNORECASE)
"""UUID4 shape. CC v2.1.131 fixtures (cc_hook_stdin/) confirm session_ids
are always UUIDs. Rejecting non-UUIDs closes the unbounded-agent_names
abuse vector (Adv #13)."""

_CONTENT_HASH_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
"""SHA-256 hex shape. Rejecting malformed hashes closes the
caller-supplied-hash abuse vector (Adv #6) where a bearer-holding client
could mint v1 with attacker-chosen hash strings."""


# ----------------------------------------------------------------------
# Session → agent UUID derivation (matches src/ccs/adapters/base.py:82)
# ----------------------------------------------------------------------


def session_to_agent_id(session_id: str, subagent_id: str | None = None) -> UUID:
    """Deterministic UUID for a Claude Code session, matching the convention
    in :class:`CoherenceAdapterCore` so other adapters and the in-process
    library see the same agent identity.

    SB-25 composite identity: a subagent's hook payload carries the PARENT
    session_id plus a distinct ``agent_id`` — folding it into the uuid5 name
    makes each subagent a first-class coherence peer (correct ``last_writer``
    attribution + sibling-collision detection). ``subagent_id`` absent/empty
    ⇒ the original derivation, byte-identical (main-thread behavior
    unchanged). The fold string is mirrored byte-for-byte by the Node
    backend (``agent_id.ts``) — a one-char divergence would silently fork
    the shared ``agent_states`` rows.
    """
    if subagent_id:
        return uuid5(
            NAMESPACE_URL,
            f"ccs-agent:claude-session-{session_id}:subagent-{subagent_id}",
        )
    return uuid5(NAMESPACE_URL, f"ccs-agent:claude-session-{session_id}")


def session_to_agent_name(session_id: str, subagent_id: str | None = None) -> str:
    """Human-readable agent name for state_log and status display. SB-25:
    a subagent identity renders as ``claude-session-<sid>:subagent-<aid>``
    so /status keeps the parent linkage visible."""
    if subagent_id:
        return f"claude-session-{session_id}:subagent-{subagent_id}"
    return f"claude-session-{session_id}"


def caller_principal_identity(session_id: str) -> UUID:
    """The identity a caller principal binds to: the SESSION component of the
    acting identity, i.e. the parent session's agent id.

    A subagent carries its parent's ``session_id``, so it presents its parent's
    principal: the principal's unit of identity is the session, and it cannot
    separate a subagent from its parent (the subagent component stays
    caller-asserted and unverified — plan R3). Keyed by the uuid5 rather than
    the raw ``session_id`` so the store holds no raw session identifier."""
    return session_to_agent_id(session_id)


@dataclass(frozen=True)
class PresentedCaller:
    """The acting identity as ONE request presents it.

    ``session_id`` is body-supplied and caller-asserted — validated for UUID
    shape only; the coordinator authenticates the workspace bearer, not the
    caller. ``subagent_id`` is caller-asserted and unverified. ``principal`` is
    the caller principal the request carries, or ``None``. This value is what
    lets a route check the SESSION component of the acting identity against the
    principal the coordinator minted for it (plan U4 / R1, R3)."""

    session_id: str
    subagent_id: str | None
    principal: str | None

    def attributed_agent_id(
        self, service: CoordinatorService, *, cached_only: bool = False
    ) -> UUID:
        """The composite agent id a write by this caller is attributed to
        (``artifacts.last_writer_id``), with its SESSION component verified
        against the caller principal first.

        ``cached_only`` decides from the service's in-process binding cache and
        raises :class:`~ccs.coordinator.service.CallerPrincipalUncached` where
        the durable store would have to be read — the admission gate's way of
        keeping that read off the request thread (:func:`_admit_caller`).

        Raises :class:`~ccs.core.exceptions.CallerPrincipalRefused` when a
        principal is presented and does not match the session's binding
        (``caller_principal_foreign`` — including one presented for a session
        nobody claimed), or when none is presented for a session that IS bound
        (``caller_principal_absent``), so a refused caller never reaches a
        write. A caller presenting none for a session nobody ever claimed is a
        client that predates the principal: it is attributed under the plain
        composite, as before the principal existed (plan U6 / R16) — refusing
        it would only turn its writes silently unrecorded, since the shipped
        hook client reads any non-2xx as an allow. The composite over session
        and subagent is unchanged — two subagents of one session keep distinct
        writer ids, which the self-commit-lag comparison depends on."""
        identity = caller_principal_identity(self.session_id)
        if self.principal is None and not service.is_caller_principal_bound(
            identity, cached_only=cached_only
        ):
            return session_to_agent_id(self.session_id, self.subagent_id)
        service.validate_caller_principal(
            identity=identity, principal=self.principal, cached_only=cached_only
        )
        return session_to_agent_id(self.session_id, self.subagent_id)


# ----------------------------------------------------------------------
# Caller-principal route posture (caller-principal plan, U6)
# ----------------------------------------------------------------------
#
# Every route that reads a ``session_id`` is classified by the harm an ABSENT
# principal admits on it (KTD3) — not by retryability: a principal is an
# authorization operand, and no downstream invariant catches a request that
# acted as someone else. What the principal buys is accident-resistance and
# attributability under the same-OS-user cooperative trust model; on the hook
# surface it is stored under ``.coherence/``, so there it is convention-
# enforcement and a detectable unbound caller, never caller separation (KTD5).
#
# Enforcement is THIS coordinator's only: the sibling Node coordinator issues
# no principals, answers 404 on ``/principal/claim`` and ignores the header
# (KTD12). The protocol corpus pins that asymmetry.


class PrincipalPosture(Enum):
    """How a route treats the caller principal of the identity it names."""

    #: Refuse a foreign principal, and an absent one for an identity that is
    #: BOUND (HTTP 400 naming the header, with a typed ``reason``; never a
    #: hold — KTD9). An absent principal for an identity nobody ever claimed is
    #: an older client's: admitted and counted, as before the principal
    #: existed (R16, KTD15).
    REQUIRE = "require"
    #: Admit an absent principal, bound identity or not, and count it (a local
    #: diagnostic, not a flip trigger — KTD4); refuse a foreign one, so a
    #: presented principal is always CHECKED, never ignored.
    ACCEPT = "accept"
    #: The route that issues principals. It cannot require one (that would be
    #: circular); its gate is the mint nonce (KTD11). The header is not read.
    MINT = "mint"


@dataclass(frozen=True)
class RoutePrincipalPosture:
    """One route's posture and the harm that places it there."""

    posture: PrincipalPosture
    harm: str


_REQUIRE = PrincipalPosture.REQUIRE
_ACCEPT = PrincipalPosture.ACCEPT

_CALLER_PRINCIPAL_POSTURE: dict[tuple[str, str], RoutePrincipalPosture] = {
    # -- the nine hook routes ------------------------------------------------
    ("POST", "/hooks/pre-read"): RoutePrincipalPosture(_ACCEPT, (
        "a read: registers or refreshes the named identity's SHARED view and "
        "records no attribution; strict mode never re-grants on a stale read, "
        "so a strict-mode write route still catches a stale writer. Accepted "
        "behaviour for an absent principal, whether the identity is unbound or "
        "bound: the read drains the named identity's pending notices into its "
        "own response, and in warn mode a stale read re-grants the identity "
        "SHARED and uses up its stale warning, so a caller naming a bound peer "
        "without its principal takes the peer's notices and warning and the "
        "peer never sees them")),
    ("POST", "/hooks/effect-fence"): RoutePrincipalPosture(_REQUIRE, (
        "gates an escaping effect on the named identity's grant standing: a "
        "caller naming a claimed peer without its principal would be answered "
        "about the PEER's grant and told an irreversible effect may fire")),
    ("POST", "/hooks/pre-edit"): RoutePrincipalPosture(_REQUIRE, (
        "acquires EXCLUSIVE under the named identity, preempting its holders, "
        "and drains its pending notices: a caller without a claimed identity's "
        "principal could take a grant it can then neither commit nor release "
        "(post-edit, either success value, and session-stop refuse it), so the "
        "grant would stand until the max-hold sweep while optimistic peers are "
        "refused other_holder. The trade-off, accepted: a claimed session whose "
        "client cannot present its principal (and cannot recover it by "
        "re-claiming with its stored nonce) has its pre-edit refused, and a "
        "hook client then proceeds uncoordinated, as on any other refusal: the "
        "edit gets no grant, no strict-mode deny and no invalidation of its "
        "peers. That was chosen over admitting an acquire the caller could "
        "neither commit nor release. The class does not stop preemption "
        "itself: a fresh, principal-bearing identity preempts identically "
        "until the acquire-or-fail contract lands")),
    ("POST", "/hooks/post-edit"): RoutePrincipalPosture(_REQUIRE, (
        "commits a version and records artifacts.last_writer_id under the "
        "named identity (or, on success:false, releases its grant): a caller "
        "without a claimed peer's principal could write under the peer's name "
        "or drop the peer's grant")),
    ("POST", "/hooks/post-edit-cas"): RoutePrincipalPosture(_REQUIRE, (
        "the optimistic commit: bumps a version and records last_writer_id "
        "under the named identity")),
    ("POST", "/hooks/session-stop"): RoutePrincipalPosture(_REQUIRE, (
        "releases every EXCLUSIVE/MODIFIED grant the named identity holds and "
        "drains its notices: a caller without a claimed peer's principal could "
        "end the peer's work")),
    ("POST", "/hooks/session-start"): RoutePrincipalPosture(_ACCEPT, (
        "re-grounding of the named session; names peers only by agent id (R7). "
        "It shows the named identity's pending notices without draining them "
        "and writes only advisory state: the session's display name and, when "
        "the re-grounding is non-empty, its compact-pending flag, which the "
        "named identity's next admit delivers. Accepted behaviour for an "
        "absent principal: a caller naming a bound peer without its principal "
        "can arm that delivery; nothing a later check depends on changes")),
    ("POST", "/hooks/pre-bash"): RoutePrincipalPosture(_ACCEPT, (
        "a read, as pre-read: stale-read detection for paths a shell command "
        "names. Accepted behaviour for an absent principal, as pre-read's: it "
        "drains the named identity's pending notices into its own response, so "
        "a caller naming a bound peer without its principal takes the peer's "
        "notices")),
    ("POST", "/hooks/pre-grep"): RoutePrincipalPosture(_ACCEPT, (
        "a read, as pre-read: stale-read detection for a search root. Accepted "
        "behaviour for an absent principal, as pre-read's: it drains the named "
        "identity's pending notices into its own response, so a caller naming "
        "a bound peer without its principal takes the peer's notices")),
    # -- the five snapshot-session routes (KTD3's own exemption) -------------
    ("POST", "/session/begin"): RoutePrincipalPosture(_ACCEPT, (
        "opens a snapshot session owned by the named identity; the session "
        "token it returns is server-minted and reaches only this caller, and "
        "every later session route is gated by that token's owner binding")),
    ("POST", "/session/read"): RoutePrincipalPosture(_ACCEPT, (
        "gated by the server-minted session token's owner binding, which the "
        "caller cannot assert; an absent principal admits nothing more")),
    ("POST", "/session/commit"): RoutePrincipalPosture(_ACCEPT, (
        "writes attribution, but only through a server-minted session token "
        "bound to its owner; an absent principal admits nothing more")),
    ("POST", "/session/commit_all"): RoutePrincipalPosture(_ACCEPT, (
        "as /session/commit, over a write-set")),
    ("POST", "/session/heartbeat"): RoutePrincipalPosture(_ACCEPT, (
        "keeps a snapshot session alive; the token's owner binding already "
        "refuses a request naming another session")),
    # -- the four workspace checkpoint-and-restore routes --------------------
    ("POST", "/workspace/checkpoint"): RoutePrincipalPosture(_REQUIRE, (
        "records durable checkpoint-manifest ownership under the named "
        "identity: a caller without a claimed identity's principal could "
        "record a manifest as that identity's")),
    ("POST", "/workspace/restore/status"): RoutePrincipalPosture(_REQUIRE, (
        "writes durable restore progress that crash-resume acts on. The route "
        "validates the identity and never consults it, so the check refuses "
        "only a request naming a CLAIMED identity without its principal; a "
        "caller naming an unclaimed identity is admitted, so this class does "
        "not limit who moves a restore's state. Kept require (not the field "
        "dropped) to match its register sibling; see the plan's U6 choice")),
    ("POST", "/workspace/restore/member"): RoutePrincipalPosture(_REQUIRE, (
        "writes a member's durable restore terminal; as /workspace/restore/"
        "status, the identity is validated but not consulted")),
    ("POST", "/workspace/restore/register"): RoutePrincipalPosture(_REQUIRE, (
        "a version-bumping write registered under a controller identity "
        "derived from the named session: a caller without a claimed session's "
        "principal could register a restore as that session's")),
    # -- the mint itself -----------------------------------------------------
    ("POST", "/principal/claim"): RoutePrincipalPosture(PrincipalPosture.MINT, (
        "issues principals; requiring one would be circular, and its gate is "
        "the mint nonce (first claim wins, a retry must present the same nonce)")),
}
"""Route -> caller-principal posture, over EVERY route that reads a
``session_id`` and no other. The require-class harm is what an absent
principal admits for an identity a client has CLAIMED; a request naming an
identity nobody claimed (a client predating the principal) is admitted and
counted on every class, as it was before. An accept-class entry states what an
absent principal is nonetheless ALLOWED to do there — the accepted behaviour,
not only the harm it rules out. The lookup in :func:`_admit_caller` has no
default: a route missing here fails loudly (500) rather than falling to a
residual class, and a test pins the key set to the routes that read one."""

# Sent only for a session that IS bound; a session nobody claimed is admitted
# without a principal (KTD15).
_CALLER_PRINCIPAL_ABSENT_ERROR = (
    f"missing {CALLER_PRINCIPAL_HEADER} header: this route requires the caller "
    f"principal that POST /principal/claim bound to the session_id it names "
    f"({CALLER_PRINCIPAL_ABSENT_REASON})"
)
# Sent on every class for a presented principal that is not the one bound to the
# named session — including one presented for a session nobody claimed, where
# nothing is bound and so no presented value can match.
_CALLER_PRINCIPAL_FOREIGN_ERROR = (
    f"the {CALLER_PRINCIPAL_HEADER} header is not the caller principal bound to "
    f"the session_id this request names ({CALLER_PRINCIPAL_FOREIGN_REASON})"
)
_CALLER_PRINCIPAL_ERRORS: dict[str, str] = {
    CALLER_PRINCIPAL_ABSENT_REASON: _CALLER_PRINCIPAL_ABSENT_ERROR,
    CALLER_PRINCIPAL_FOREIGN_REASON: _CALLER_PRINCIPAL_FOREIGN_ERROR,
}
"""The ``error`` text of a principal refusal, keyed by its reason. The 400 body
is ``{"error": <text>, "reason": <reason>}``: ``error`` keeps the key every
non-200 body carries (the CLI reads it), and ``reason`` is the typed key a
client classifies the refusal by — by equality against
:data:`~ccs.core.exceptions.CALLER_PRINCIPAL_REFUSAL_REASONS`, never by a
substring of ``error``. Each text names the header and ends with its reason, so
an absent principal and a foreign one stay distinguishable from either key (a
phase is diagnosable from the refusal alone). Neither carries a principal."""


def _admit_caller(
    req: _RequestProtocol,
    coordinator: CoordinatorHTTPServer,
    presented: PresentedCaller,
) -> UUID | None:
    """Apply the dispatched route's posture to ``presented``.

    Returns the composite agent id the request acts as, or ``None`` after
    writing the HTTP 400 refusal. On a require-class route, and whenever a
    principal IS presented, the id comes from
    :meth:`PresentedCaller.attributed_agent_id`, which owns the rule: a
    presented principal must be the bound one; none may be presented only for
    an identity nobody has claimed. An accept-class request presenting none is
    admitted under the unverified derivation. Every admission without a
    principal — accept-class, or require-class on an unbound identity — is
    counted (a local diagnostic, KTD4); every refusal is counted separately and
    logged, by :func:`_refuse_caller`.

    The gate runs BEFORE any handler mutation — the display-name registration,
    the heartbeat, the re-grounding flag, a notice drain, a grant — so a
    refused request has changed nothing, and a client may retry it once it
    holds the right principal.

    It never waits longer than the handler watchdog: see
    :func:`_attributed_within_deadline`. A gate that cannot decide in time
    writes the route's degraded envelope (past a full watchdog queue, the 503)
    and returns ``None`` too — neither admitted nor refused, so it is counted
    by neither principal counter."""
    posture = _CALLER_PRINCIPAL_POSTURE[req._route_key].posture
    unverified = session_to_agent_id(presented.session_id, presented.subagent_id)
    if posture is PrincipalPosture.MINT:
        return unverified
    if posture is PrincipalPosture.ACCEPT and presented.principal is None:
        coordinator.increment_caller_principal_absent()
        return unverified
    try:
        agent_id = _attributed_within_deadline(req, coordinator, presented)
    except CallerPrincipalRefused as exc:
        _refuse_caller(req, coordinator, exc.reason)
        return None
    if agent_id is None:
        return None
    if presented.principal is None:
        # Require-class, admitted without a principal: the identity is unbound.
        coordinator.increment_caller_principal_absent()
    return agent_id


def _attributed_within_deadline(
    req: _RequestProtocol,
    coordinator: CoordinatorHTTPServer,
    presented: PresentedCaller,
) -> UUID | None:
    """:meth:`PresentedCaller.attributed_agent_id`, decided without waiting
    past the handler watchdog.

    The service's binding cache answers on this thread, touching no registry:
    a bound identity, and — after one read of the store — an unbound one
    (the negative tier, so a client that never claims stops costing a registry
    read per request). Only a cache miss reads the durable store, and that read
    waits on the registry lock, so it runs in the watchdog pool and STARTS this
    request's watchdog deadline: the work body later gets only what the lookup
    left of it (:func:`_watchdog_deadline`), and gate plus body stay inside one
    ``HANDLER_TIMEOUT_SEC``. Were the read made here instead, a client that
    never claims (KTD15) would wait on registry contention with no deadline on
    every require-class route — pre-edit included.

    Returns the agent id, or ``None`` after writing the answer itself, one of
    three: the watchdog pool's queue-overflow 503; when the lookup is not done
    by the deadline, exactly the route's degraded envelope
    (:data:`_CALLER_GATE_DEGRADED_RESPONSE`), counted as a watchdog timeout
    like a timed-out work body; and when the lookup RAISED anything but a
    refusal — the registry read failing once ``busy_timeout`` elapsed, say —
    exactly the route's internal-error envelope, logged as a failed work body
    is (:func:`_answer_handler_error`). That last arm is what keeps the gate
    and the body on one error contract: were the raise left to the
    dispatcher's 500, a degrade-mode ``CoherentVolume`` would read the answer
    as "no coordinator" and write with no grant, where the typed 200 raises in
    both of its modes. Raises :class:`CallerPrincipalRefused` for a refusal
    decided within the deadline, which stays an HTTP 400."""
    service = coordinator.service
    try:
        return presented.attributed_agent_id(service, cached_only=True)
    except CallerPrincipalUncached:
        pass
    if _watchdog_queue_overflowed(req, coordinator):
        return None
    deadline = time.monotonic() + HANDLER_TIMEOUT_SEC
    req._watchdog_deadline = deadline
    method, path = req._route_key
    try:
        return coordinator.run_admission_lookup(
            lambda: presented.attributed_agent_id(service), deadline=deadline
        )
    except CallerPrincipalRefused:
        raise
    except FuturesTimeout:
        coordinator.increment_watchdog_timeout()
        logger.warning(
            "caller-principal lookup on %s %s timed out after %ss; degrading",
            method, path, HANDLER_TIMEOUT_SEC,
        )
        req._json(200, _CALLER_GATE_DEGRADED_RESPONSE[req._route_key])
        return None
    except Exception as exc:
        # Ordered after the refusal and the timeout: both are Exceptions too
        # (the futures timeout is the builtin TimeoutError since 3.11). The
        # lookup only reads, so nothing landed; the request was neither
        # admitted nor refused, so neither principal counter moves.
        _answer_handler_error(
            req, coordinator, exc, failed=f"caller-principal lookup on {method} {path}"
        )
        return None


def _refuse_caller(
    req: _RequestProtocol, coordinator: CoordinatorHTTPServer, reason: str
) -> None:
    """Answer a principal refusal: HTTP 400 ``{"error", "reason"}``.

    Counted and logged at INFO with the route and the reason BEFORE the answer
    goes out, so an operator can see a session being refused — the hook client
    exits 0 on a refusal and Claude Code shows no stderr, so nothing else
    would. Only the reason is read: neither the presented principal nor the
    bound one reaches the log, the counter or the body (R5)."""
    coordinator.increment_caller_principal_refused()
    method, path = req._route_key
    logger.info("caller principal refused on %s %s (%s)", method, path, reason)
    req._json(400, {"error": _CALLER_PRINCIPAL_ERRORS[reason], "reason": reason})


def _presented_caller(
    req: _RequestProtocol, session_id: str, subagent_id: str | None = None
) -> PresentedCaller:
    """The acting identity this request presents: the body's ``session_id``
    and subagent id, and the header's principal (an empty header is absent)."""
    principal = req.headers.get(CALLER_PRINCIPAL_HEADER) or None
    return PresentedCaller(session_id=session_id, subagent_id=subagent_id, principal=principal)


# ``monotonic_seconds`` moved to ``ccs.core.clock`` (WV plan Unit 3) so the
# workspace capture engine shares the coordinator's ONE wall-clock tick basis
# without importing this server module. Re-exported here under the same name —
# every existing import site (``lifecycle.py``'s lazy import, tests) is
# unchanged. See the ``ccs.core.clock.monotonic_seconds`` docstring for the
# wall-clock-not-monotonic rationale (the F1 sweep bug).


# ----------------------------------------------------------------------
# Boundary validators (Adv-review hardening A2 + A3 + A8)
# ----------------------------------------------------------------------


_SUBAGENT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def read_subagent_id(body: dict) -> str | None:
    """SB-25: optional subagent identity from the hook request body.

    Accepts the documented snake_case ``agent_id`` with a defensive fallback
    to the transcript-observed camelCase ``agentId`` (exact wire casing is
    pinned by the R6 live capture; until then both are honored). Additive +
    backward-compatible: absent, non-string, or out-of-shape values resolve
    to ``None`` — the parent identity — never a 400.
    """
    raw = _raw_subagent_id_value(body)
    # fullmatch (not match): Python's `$` also matches just before a trailing
    # newline, so `.match("abc\n")` would ACCEPT it — but JS's `$` does not,
    # so a trailing-newline id would fork the composite agent_id across
    # backends. fullmatch closes that byte-parity gap.
    if isinstance(raw, str) and _SUBAGENT_ID_RE.fullmatch(raw):
        return raw
    return None


def _raw_subagent_id_value(body: dict) -> object:
    """Raw subagent-id value, snake_case preferred (SB-25). Byte-parity with
    the Node reader: an explicitly-present ``agent_id`` (even null) is NOT
    overridden by ``agentId`` — only an ABSENT ``agent_id`` key falls back."""
    return body.get("agent_id", body.get("agentId"))


def has_subagent_id_field(body: dict) -> bool:
    """True iff the body carries a present, non-null, non-empty subagent-id
    value of ANY type — regardless of whether it passes :data:`_SUBAGENT_ID_RE`.
    Lets the destructive session-stop path distinguish "absent → legitimate
    parent stop" from "present-but-malformed → refuse, never degrade to
    releasing the parent's grants" (the P1 subagent-stop safety fix).

    Covering non-string types (int/list/dict/bool) matters: a present
    ``agent_id: 42`` must be REFUSED, not treated as absent and degraded to the
    parent identity. Read paths don't carry this guard — a malformed id
    degrading to parent attribution there is benign."""
    raw = _raw_subagent_id_value(body)
    return raw is not None and raw != ""


def validate_session_id(
    session_id: Any,
) -> tuple[Literal["MISSING", "MALFORMED"], str] | None:
    """Return (error_kind, reason) if invalid, None if valid. UUID-shape required.

    AC-51 / finding #51: structured error kind lets callers branch on MISSING vs
    MALFORMED without string-prefix matching (``err.startswith(...)``).
    """
    if not isinstance(session_id, str):
        return ("MISSING", "missing session_id")
    if not _SESSION_ID_RE.match(session_id):
        return ("MALFORMED", "session_id must be a UUID (8-4-4-4-12 hex with hyphens)")
    return None


def validate_path(path: Any) -> str | None:
    """Server-side authoritative path check. Return reason if invalid,
    None if valid. The coordinator stores paths as parent-repo-relative
    — KTD-7 normalization happens client-side (the hook script must
    compute repo-relative from CC's absolute tool_input.file_path).
    The boundary validation here is defense-in-depth against bad hook
    clients AND prose-injection abuse (Adv #4 + Adv #11).

    M-02 layer-distinction note: this is the STRICTER server-side check
    (rejects non-string types, leading backslash, control characters,
    paths longer than MAX_PATH_LEN). The CLI-side counterpart is
    :func:`ccs.cli._coherence_client.validate_relative_path` — lighter
    (string-typed-input assumed, just empty/leading-slash/`..` checks),
    runs before the HTTP request is built for fast operator feedback.
    This function is the authoritative gate; do NOT remove either —
    they live at different layers of the trust boundary.
    """
    if not isinstance(path, str):
        return "path must be a string"
    if not path:
        return "path is empty"
    if len(path) > MAX_PATH_LEN:
        return f"path exceeds {MAX_PATH_LEN} chars"
    if path.startswith("/"):
        return "path must be relative (no leading /)"
    if path.startswith("\\"):
        return "path must be relative (no leading \\)"
    # ANY control character rejected — guards against newline injection into
    # additionalContext prose (Adv #11) and other terminal-control mischief.
    if any(ord(c) < 0x20 or ord(c) == 0x7F for c in path):
        return "path contains control characters"
    # .. traversal at any segment boundary
    parts = path.replace("\\", "/").split("/")
    if ".." in parts:
        return "path contains '..' traversal"
    return None


def validate_content_hash(content_hash: Any, *, required: bool) -> str | None:
    """Return reason if invalid, None if valid. content_hash is OPTIONAL on
    pre-read (the caller may not have it yet) but REQUIRED on post-edit
    (the caller just wrote the file and computed it). Empty string is
    always rejected to avoid the silent-record-known-wrong-hash anti-pattern."""
    if content_hash is None and not required:
        return None
    if not isinstance(content_hash, str):
        return "content_hash must be a string"
    if not content_hash:
        return "content_hash is empty (omit the field instead if unknown)"
    if not _CONTENT_HASH_RE.match(content_hash):
        return "content_hash must be 64 hex characters (sha-256)"
    return None


# Sentinel recorded hash carrying no real content claim: launch-gate
# scenarios inject artifact rows with "f"*64 (no real SHA-256 matches it,
# guaranteeing hash_differs on the stale path). The other no-claim seed —
# "" from KTD-9 first observations without a caller hash — surfaces as
# ``None`` on the Artifact, so truthiness checks already exclude it. The
# fresh-SHARED mismatch signal must not fire against either.
_F_SENTINEL_CONTENT_HASH = "f" * 64

# Survivor #6 v1 (R2): a SHARED-holder hash mismatch is treated as the benign
# commit→disk-write lag (not a foreign edit) only when THIS session's own
# commit is within this window of the read. Generous on purpose — the flush
# after a commit_cas WIN is normally sub-second, so a value well above that
# avoids false-denies during a slow flush, at the cost of a narrow
# false-negative (a foreign edit within the window of a same-session commit).
# The fresh_shared_hash_mismatch counter remains the regression guard.
_SHARED_FOREIGN_DENY_LAG_WINDOW_SEC = 5.0


# ----------------------------------------------------------------------
# Server shell
# ----------------------------------------------------------------------


class CoordinatorHTTPServer:
    """Holds the wired-up coordinator state and a stdlib ThreadingHTTPServer.

    Caller (Unit 5 ``lifecycle.ensure_coordinator``) instantiates this then
    calls :meth:`serve_in_thread` to start a daemon-thread serving loop. The
    ``port`` attribute is populated once the OS picks one (port=0 binding)."""

    def __init__(
        self,
        coordinator_root: Path,
        *,
        port: int = 0,  # 0 = OS picks
        bind_host: str = "127.0.0.1",
        agent_names: dict[UUID, str] | None = None,
        state_log: Callable[[dict[str, Any]], None] | None = None,
        instance_id: str | None = None,
    ) -> None:
        self.coordinator_root = Path(coordinator_root).resolve()
        self.bind_host = bind_host
        # Host-header allowlist: loopback always, plus a validated non-loopback
        # bind (cross-host demo). Validates bind_host here at construction — a
        # disallowed bind (wildcard, loopback alias, link-local, CGNAT, public, or
        # non-loopback without the CCS_REMOTE_COORDINATOR opt-in) raises.
        self.host_allowlist = build_host_allowlist(bind_host)
        # Unit 3 / R5: fail-closed transport symmetry with the client's #135
        # plaintext-bearer guard. A routed (non-loopback) bind serves the bearer
        # secret on the wire, so it must either assert a TLS-terminating front
        # (CCS_TLS_TERMINATED) or explicitly acknowledge the insecure link
        # (CCS_SERVE_INSECURE), else construction raises ValueError. Loopback binds
        # read neither env and are byte-unchanged. Runs before the socket bind so a
        # routed coordinator without an ack never opens a listening socket.
        assert_serve_transport_acknowledged(bind_host)
        # Monotonic reference points (NOT wall-clock): idle/uptime deltas must
        # survive NTP steps and suspend/resume (finding L5). time.time() stays
        # reserved for operator-facing absolute timestamps elsewhere.
        self._started_at = time.monotonic()
        self._last_request_at = self._started_at
        self._shutting_down = False
        # ADV-001 (fix): "migration draining" is a halfway state between
        # serving and shutting_down. While true, new pre-edit requests are
        # rejected (would mint an EXCLUSIVE that the agent can never
        # post-edit since shutdown is imminent). All other endpoints —
        # pre-read, post-edit, session-stop, status — continue to be
        # served so in-flight pre-edit→post-edit chains can complete
        # naturally. Set by /admin/prepare-for-migration; cleared only
        # at process exit (no rollback path).
        self._migration_draining = False
        self._agent_names: dict[UUID, str] = dict(agent_names or {})
        # R10 (Unit 6): explicit lock around _agent_names mutation. CPython's
        # GIL makes single-key dict assignment effectively atomic today, but
        # the project standard is "don't rely on GIL" so the contract holds
        # on PyPy / future free-threading builds and so the snapshot pattern
        # at _handle_status (list(items()) under the lock) sees a consistent
        # view. threading.Lock per pattern (NOT RLock — no nested acquisition).
        self._agent_names_lock = threading.Lock()
        # v0.1.1 KTD-G item 3: surface watchdog/concurrency degradation
        # rather than letting it stay silent. Counters are read by
        # _handle_status; incremented in _run_or_degrade (timeouts +
        # queue overflows) and _ThreadingHTTPServer.process_request
        # (handler concurrency overflows).
        #
        # REL-03 (free-threading-safe): under CPython's traditional
        # build, ``x += 1`` on an int attribute is effectively atomic
        # because the GIL serializes bytecode execution. Under Python
        # 3.13+ free-threaded builds (PEP 703) and on PyPy, that
        # guarantee is gone — concurrent threads can read-modify-write
        # the same counter and tear increments. These three counters
        # are operator-facing reliability signals (a missed bump means
        # an under-reported degradation event in a bug report), so
        # protect their mutation with a lock. Product-signal counters
        # (intra_task_acquire_release_total, stale_warning_*_total)
        # stay GIL-reliant per the reviewer's recommendation: they're
        # advisory ratios, not absolute counts.
        self._reliability_counter_lock = threading.Lock()
        self._watchdog_timeouts_total: int = 0
        self._watchdog_queue_overflows_total: int = 0
        # P1 #6: silent 401 surface. If hook.secret is deleted out from
        # under a running coordinator (operator misclick, accidental
        # rm in .coherence), every subsequent hook request from any
        # session 401s. The client treats 401 as "no coordinator
        # available" and degrades silently — agents lose the coherence
        # layer with zero operator signal. Counter + WARNING log
        # surface the symptom; 60s dedupe avoids spamming the log when
        # a real burst hits. ``self._last_401_warn_at`` is the
        # monotonic timestamp of the last warning emission.
        self._auth_401_total: int = 0
        self._last_401_warn_at: float = 0.0
        self._auth_401_warn_lock = threading.Lock()
        # P1 #5 detection-only: when ``run_with_watchdog`` raises
        # FuturesTimeout, the handler returns degraded — but the
        # underlying ``work()`` future is left running in the pool
        # (cancel_futures=False). If it eventually completes
        # successfully, any state it mutated (EXCLUSIVE grant from
        # ``service.write``, for instance) lands in the registry AFTER
        # the agent received a degraded response — a phantom grant the
        # agent will never post-edit. We can't cancel the future
        # without invasive cancel-token plumbing through service.write,
        # but we CAN detect it: every timed-out future gets a
        # done_callback that bumps this counter + logs CRITICAL if the
        # future eventually finished without exception. Operators see
        # the symptom via ``/status?detail=metrics`` even when the
        # cause is rare (4s deadline + 1.5s busy_timeout = real-world
        # only hits with a wedged SQLite or contended drive).
        self._watchdog_late_completion_total: int = 0
        # A6: a watchdog-timed-out future that aborted cleanly at the registry
        # write lock (abort_guard) before mutating — the mitigation working.
        # Distinct from late_completion (which is an UNmitigated phantom).
        self._watchdog_late_aborts_total: int = 0

        # KTD-I (Unit 5 L2) — in-flight handler counter. Incremented at
        # dispatch entry via :meth:`acquire_handler_slot`, decremented in
        # the handler's finally via :meth:`release_handler_slot`.
        # :meth:`shutdown` blocks on the counter reaching zero for up to
        # IN_FLIGHT_DRAIN_TIMEOUT_SEC before closing the SQLite registry,
        # so a handler mid-write doesn't see ProgrammingError on a closed
        # connection (silent hang → observable 500 at worst).
        self._in_flight_lock = threading.Lock()
        self._in_flight_zero = threading.Condition(self._in_flight_lock)
        self._in_flight = 0
        self._in_flight_drain_timed_out = False

        # KTD-H/I/L3 (Unit 5 L3) — cold-start timing populated by the
        # lifecycle winner path after self-probe completes. Telemetry
        # surface for the future /status endpoint (Unit 8). 0.0 until the
        # lifecycle module sets it; remains 0.0 when the server is
        # constructed directly in tests.
        self.cold_start_duration_ms: float = 0.0

        # KTD-J (Unit 8) — telemetry counters. CACHE, not persistent
        # state: reset to 0 on coordinator respawn (do NOT persist in
        # state.db per the plan rationale). Plain ints — CPython's
        # GIL guarantees atomicity for ``+= 1``; the contract is
        # "advisory, not auditable", so a missed increment on a
        # future free-threading build is acceptable.
        #
        # Per-endpoint request counters — drive operator visibility
        # into which hooks fire and how often. Surfaced via
        # /status?detail=full and /status?detail=metrics.
        self._endpoint_counters: dict[str, int] = {
            "pre_read_total": 0,
            "pre_edit_total": 0,
            "post_edit_total": 0,
            "post_edit_cas_total": 0,
            "session_stop_total": 0,
            "session_start_total": 0,
            "pre_bash_total": 0,
            "pre_grep_total": 0,
            "policy_track_total": 0,
            "policy_untrack_total": 0,
            "status_total": 0,
            # Registered in _ENDPOINT_COUNTER_NAMES *and* here, in one edit:
            # increment_endpoint_counter ignores an unknown name by contract,
            # so a route named there but missing here is counted nowhere while
            # still looking wired up at both call sites.
            "effect_fence_total": 0,
            # Same rule as effect_fence_total above: registered in
            # _ENDPOINT_COUNTER_NAMES too, or the mint is counted nowhere.
            "principal_claim_total": 0,
        }
        self._endpoint_counters_lock = threading.Lock()

        # KTD-J product-signal counters. These shape v0.2 / hosted-tier
        # decisions, so each has documented economic meaning.
        #
        # intra_task_acquire_release_total: how often a session acquired
        # EXCLUSIVE and released within the same dispatch chain. Sizes
        # the hosted-tier upsell argument (signal that fine-grained
        # write protection is exercised, not idle).
        #
        # stale_warning_emitted_total: how often /hooks/pre-read or
        # /hooks/pre-bash returned a stale-summary response. Denominator
        # for the operator-computed re-read rate.
        #
        # stale_warning_reread_total: how often the agent re-read after
        # a stale warning (heuristic: same session re-reads same path
        # within HANDLER_TIMEOUT_SEC × 4 of receiving a stale warning).
        # Numerator for re-read rate; operator computes the ratio at
        # query time.
        self._intra_task_acquire_release_total: int = 0
        self._stale_warning_emitted_total: int = 0
        self._stale_warning_reread_total: int = 0

        # v0.2 Unit 4 (KTD-V minimal + KTD-J extension): strict-mode
        # telemetry counters surfaced via /status?detail=metrics. Three
        # counters:
        # - strict_mode_denials_total: bumped on every strict-mode deny
        #   emission across all 4 handlers (Read / Edit/Write / Bash /
        #   Grep). Operator-visible measure of strict mode's active
        #   workload.
        # - strict_mode_routed_around_via_bash_total: bumped in
        #   _handle_pre_bash when a strict-deny fires AND a prior Read
        #   strict-deny was logged for the same (session, path) within
        #   STRICT_DENY_ROUTE_AROUND_WINDOW_SEC (default 30). Measures
        #   the Phase 0 H4 routing pattern in live operation.
        # - audit_log_mode_drift_total: bumped when audit.log mode
        #   differs from 0o600 at append time. Operator chmod drift
        #   surfaces here without refusing the append.
        self._strict_mode_denials_total: int = 0
        self._strict_mode_routed_around_via_bash_total: int = 0
        self._audit_log_mode_drift_total: int = 0

        # Defense-in-depth signal (PR #108 follow-up, 2026-06-11):
        # fresh_shared_hash_mismatch_total counts pre-reads where a
        # SHARED holder supplied a content_hash that mismatches the
        # recorded (non-sentinel) artifact hash. A peer commit would
        # have left the session INVALID, so a mismatch here implies an
        # out-of-band write or the commit→disk-write lag observed via a
        # warn-mode re-grant. The fresh response stays an allow (the
        # plugin path is fail-open); this counter sizes the
        # false-positive rate before any strict-mode deny knob is added.
        self._fresh_shared_hash_mismatch_total: int = 0
        # effect_fence_holds_total: how often the /hooks/effect-fence verdict
        # answered HOLD. The denominator is the endpoint counter
        # ``effect_fence_total``; the ratio is what tells an operator whether
        # their agents are gating on state the coordinator can confirm.
        self._effect_fence_holds_total: int = 0
        # caller_principal_absent_total: requests admitted without a caller
        # principal — on an accept-class route, or on a require-class route for
        # an identity nobody claimed (plan U6 / KTD4). A LOCAL diagnostic an
        # operator reads on one workspace — "some caller here is not yet sending
        # a principal". Never a trigger for making a route require one: like
        # every counter here it is process-local, resets on respawn and on idle
        # shutdown, so a zero proves nothing about the callers out there.
        self._caller_principal_absent_total: int = 0
        # caller_principal_refused_total: requests a route REFUSED for their
        # caller principal — an absent one naming a bound identity, or a
        # foreign one on any class (plan U6). The operator-visible trace of a
        # session being refused: the endpoint counters count attempts, so a
        # refused request looks like any other there. Process-local like
        # every counter here.
        self._caller_principal_refused_total: int = 0
        # Survivor #6 v1 (R2) observability: how often a SHARED-holder hash
        # mismatch on a strict path was SUPPRESSED as the benign
        # commit→disk-write lag (this session's own recent commit) rather than
        # denied as a foreign edit. Pairs with strict_mode_denials_total to let
        # an operator size the lag-window (5s) false-negative exposure.
        self._shared_foreign_lag_suppressed_total: int = 0
        # Per-(session, path) "recent strict-deny" memory for route-around
        # detection. Bounded by the registry's own (session, artifact)
        # cardinality at worst — same upper bound as _stale_warned_pairs.
        # GC happens lazily on every check_strict_deny_route_around call.
        self._recent_strict_denies: dict[tuple[str, str], float] = {}
        self._recent_strict_denies_lock = threading.Lock()

        # KTD-J re-read detection: a stale warning marks an (agent, artifact)
        # pair as "warned"; the next pre-read on that pair consumes the
        # marker and bumps :attr:`_stale_warning_reread_total`. The set
        # cannot grow without bound — any subsequent pre-read clears the
        # entry, and a worst-case scenario of one entry per active
        # (agent, artifact) pair is the same upper bound as the registry's
        # own state map.
        self._stale_warned_pairs: set[tuple[UUID, UUID]] = set()
        self._stale_warned_pairs_lock = threading.Lock()

        # SB-10 (KTD5): compact-pending flags, keyed by session_id. Set by
        # /hooks/session-start when it emitted a non-empty re-grounding
        # payload; consumed by the next qualifying parent allow via
        # _deliver_pending_reground (SB-10 U4) and expired on parent Stop.
        # PROCESS-LOCAL on purpose — a restarted coordinator loses the flag,
        # an accepted degradation (the restart is a bigger re-grounding event
        # than a compaction).
        self._compact_pending: set[str] = set()
        self._compact_pending_lock = threading.Lock()

        # Wire storage + coordinator service.
        db_path = self.coordinator_root / ".coherence" / "state.db"
        self.registry = SqliteArtifactRegistry(
            db_path,
            state_log=state_log,
            agent_names=self._agent_names,
            instance_id=instance_id,
        )
        self.service = CoordinatorService(self.registry)
        self.policy = TrackedArtifactPolicy.load(self.coordinator_root)

        # Shared secret (auth) — generated on first spawn, persisted across restarts.
        self.secret = ensure_secret(self.coordinator_root)

        # Handler watchdog executor — bounded to a small pool, the work is
        # SQLite-bound and we want timeouts not parallelism. Size matches
        # _WATCHDOG_POOL_SIZE; KTD-G concurrency limits derive from this.
        self._watchdog = ThreadPoolExecutor(
            max_workers=_WATCHDOG_POOL_SIZE,
            thread_name_prefix="coord-wd",
        )

        # ThreadingHTTPServer — handlers see this instance via .server.coordinator.
        # KTD-G item 2: concurrency_limit caps handler threads at
        # HANDLER_CONCURRENCY_LIMIT (= pool_size × 2); requests above the
        # limit get a synchronous 503 without spawning a handler thread.
        handler_cls = _make_handler_class(self)
        self._server = _ThreadingHTTPServer(
            (bind_host, port),
            handler_cls,
            concurrency_limit=HANDLER_CONCURRENCY_LIMIT,
        )
        self.port = self._server.server_port
        self._serve_thread: threading.Thread | None = None

    def serve_in_thread(self) -> None:
        """Start the serving loop in a daemon thread."""
        if self._serve_thread is not None:
            return
        self._serve_thread = threading.Thread(
            target=self._server.serve_forever, name="coord-http", daemon=True
        )
        self._serve_thread.start()

    def shutdown(self) -> None:
        """Stop the server, drain in-flight handlers, close storage.

        KTD-I (Unit 5 L2): waits up to ``IN_FLIGHT_DRAIN_TIMEOUT_SEC`` for
        the in-flight counter to reach zero before closing the SQLite
        registry. Sets ``shutting_down`` BEFORE the drain so new dispatch
        attempts 503 immediately and don't replenish the counter. After
        the deadline, closes regardless — handlers still mid-write may
        raise ``sqlite3.ProgrammingError`` (becoming HTTP 500 to clients),
        which is observable. The alternative — wedging shutdown waiting
        for a stuck handler — is silent and worse.

        COR-07: actual wall-clock shutdown time can EXCEED
        ``IN_FLIGHT_DRAIN_TIMEOUT_SEC`` when watchdog timeouts have
        fired. The in-flight counter decrements when the handler
        thread returns (after FuturesTimeout), but the corresponding
        watchdog-pool future may still be running. The subsequent
        ``self._watchdog.shutdown(wait=True, cancel_futures=False)``
        waits for those orphaned futures to complete. Worst-case
        addition: one extra ``HANDLER_TIMEOUT_SEC`` (4s) per orphaned
        future. ``cancel_futures=True`` would shorten shutdown but
        risk aborting a SQLite write mid-transaction; documented
        trade-off, not a bug.
        """
        # REL-05 / finding #42: protect the check-then-set with a lock so
        # concurrent callers (idle thread + stop_coordinator) cannot both
        # observe _shutting_down=False and both enter the shutdown body.
        # _in_flight_lock is the right granularity: shutdown is what drives
        # the drain, so no new lock is needed.
        with self._in_flight_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        try:
            # http.server.HTTPServer.shutdown() blocks on an Event set by
            # serve_forever's exit. If serve_in_thread was never called,
            # serve_forever never ran, and the event was never set —
            # shutdown() would wait forever. Guard against that so unit
            # tests that construct a server purely for state manipulation
            # can still tear it down cleanly.
            #
            # COR-01: order is shutdown → drain → server_close → registry.close.
            # _server.shutdown() stops the serve_forever accept loop but does
            # NOT close in-flight handler threads (they own their accepted
            # sockets and finish writing on their own). Drain those slots
            # BEFORE server_close + registry.close, so the registry remains
            # open while a handler is still mid-transaction. Closing the
            # listening socket is independent of accepted-connection sockets
            # but keeping it open through the drain matches the canonical
            # shutdown ordering and removes a refactor footgun.
            if self._serve_thread is not None:
                self._server.shutdown()
        finally:
            self._drain_in_flight(IN_FLIGHT_DRAIN_TIMEOUT_SEC)
            try:
                self._server.server_close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                logger.exception("server_close raised during shutdown")
            self._watchdog.shutdown(wait=True, cancel_futures=False)
            self.registry.close()

    def acquire_handler_slot(self) -> bool:
        """KTD-I L2: atomic shutting_down check + counter increment.

        Returns False if shutdown has started between the dispatcher's
        outer ``shutting_down`` check and this call (race window of a few
        microseconds). Returns True iff the slot was acquired and the
        caller MUST pair with :meth:`release_handler_slot`."""
        with self._in_flight_lock:
            if self._shutting_down:
                return False
            self._in_flight += 1
            return True

    def release_handler_slot(self) -> None:
        """KTD-I L2: decrement the counter and notify drain waiters when
        it reaches zero. Safe to call from any handler thread's finally."""
        with self._in_flight_lock:
            self._in_flight -= 1
            if self._in_flight <= 0:
                self._in_flight = 0  # defensive — never go negative
                self._in_flight_zero.notify_all()

    def _drain_in_flight(self, timeout_sec: float) -> None:
        """Wait up to ``timeout_sec`` for in-flight handlers to complete.

        Sets :attr:`_in_flight_drain_timed_out` if the deadline elapses
        with handlers still running, so operators can observe the event
        via the eventual /status surface (deferred to Unit 8)."""
        if timeout_sec <= 0:
            return
        deadline = time.monotonic() + timeout_sec
        with self._in_flight_lock:
            while self._in_flight > 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._in_flight_drain_timed_out = True
                    logger.warning(
                        "shutdown drain timed out after %.1fs with %d handler(s) still in-flight; "
                        "closing registry anyway (KTD-I — observable 500 > wedged shutdown)",
                        timeout_sec, self._in_flight,
                    )
                    return
                self._in_flight_zero.wait(timeout=remaining)

    def __enter__(self) -> "CoordinatorHTTPServer":
        self.serve_in_thread()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Server-side state & metrics surfaced to handlers
    # ------------------------------------------------------------------

    def mark_request(self) -> None:
        # Monotonic — see __init__ (L5). Feeds idle_seconds, not any wire timestamp.
        self._last_request_at = time.monotonic()

    @property
    def uptime_s(self) -> float:
        """Monotonic seconds since server start (NTP-/suspend-safe duration)."""
        return time.monotonic() - self._started_at

    @property
    def last_request_at(self) -> float:
        """Monotonic reference point of the most recent request (or server
        start if none yet). NOT a wall-clock timestamp — do not emit on the
        wire or convert to ISO (finding L5).

        P3 ce-review fix #39 (kieran-python): the idle-shutdown loop in
        lifecycle.py previously reached into the private
        ``_last_request_at`` with a ``# type: ignore[attr-defined]``.
        This public property removes the cross-module private access."""
        return self._last_request_at

    @property
    def idle_seconds(self) -> float:
        """Monotonic seconds since the most recent request (NTP-/suspend-safe)."""
        return time.monotonic() - self._last_request_at

    @property
    def migration_draining(self) -> bool:
        """ADV-001: True between the prepare-for-migration trigger and the
        scheduled shutdown. The dispatcher uses this to reject NEW write
        initiations (pre-edit) while still serving in-flight chains'
        completions (post-edit) and all read/observability endpoints."""
        return self._migration_draining

    @property
    def shutting_down(self) -> bool:
        return self._shutting_down

    def register_session(
        self, session_id: str, subagent_id: str | None = None
    ) -> UUID:
        """Idempotent session registration. Returns the deterministic agent UUID.

        R10 (Unit 6): mutation is wrapped in ``_agent_names_lock`` so the
        check-then-set is atomic w.r.t. concurrent registrations AND so
        :meth:`agent_name_for` snapshots see a consistent dict — relying
        on the GIL is forbidden by the project standard."""
        agent_id = session_to_agent_id(session_id, subagent_id)
        with self._agent_names_lock:
            if agent_id not in self._agent_names:
                self._agent_names[agent_id] = session_to_agent_name(
                    session_id, subagent_id
                )
        return agent_id

    def agent_names_snapshot(self) -> list[tuple[UUID, str]]:
        """R10 (Unit 6): return a stable list snapshot of (agent_id, name)
        pairs taken under the lock — callers iterate over the snapshot
        rather than the live dict so a concurrent register_session cannot
        invalidate the iteration (RuntimeError: dictionary changed size)."""
        with self._agent_names_lock:
            return list(self._agent_names.items())

    def agent_name_for(self, agent_id: UUID) -> str | None:
        """R10 (Unit 6): single-key read under the lock. Returns None if
        the agent has never been registered."""
        with self._agent_names_lock:
            return self._agent_names.get(agent_id)

    def agents_for_session(self, session_id: str) -> list[tuple[UUID, str | None]]:
        """SB-10 U2: the session's coherence agents as ``(agent_id,
        subagent_name)`` pairs — the parent first (``subagent_name=None``,
        derivable via uuid5 WITHOUT prior registration), then registered
        subagents sorted by agent name (KTD8 group order).

        Enumeration is a prefix scan of the in-memory ``_agent_names`` map
        using the deterministic SB-25 naming scheme
        (``claude-session-<sid>:subagent-<name>``) — neither registry has a
        session→subagents accessor, and the adapter-owned map is the SB-25
        source of truth for registration. Session ids are fixed-shape UUIDs
        (A3 validation), so one session's prefix can never be a prefix of
        another's. Restart-empty maps are an accepted degradation (KTD5):
        subagents re-enter on their next hook call."""
        subagent_prefix = f"claude-session-{session_id}:subagent-"
        with self._agent_names_lock:
            subagents = [
                (agent_id, name[len(subagent_prefix):])
                for agent_id, name in self._agent_names.items()
                if name.startswith(subagent_prefix)
            ]
        # Same-prefix names sort identically by full name or by suffix; the
        # suffix is what the Subagent group prefix renders.
        subagents.sort(key=lambda pair: pair[1])
        parent: list[tuple[UUID, str | None]] = [
            (session_to_agent_id(session_id), None)
        ]
        return parent + subagents

    def increment_endpoint_counter(self, name: str) -> None:
        """KTD-J (Unit 8): bump a per-endpoint counter. Names match the
        keys in ``_endpoint_counters`` (e.g., ``pre_read_total``). Unknown
        names are silently ignored — counters are advisory; a typo in a
        future endpoint dispatch must not crash the request."""
        with self._endpoint_counters_lock:
            if name in self._endpoint_counters:
                self._endpoint_counters[name] += 1

    def endpoint_counters_snapshot(self) -> dict[str, int]:
        """KTD-J (Unit 8): stable snapshot of per-endpoint counters.
        Taken under the lock so concurrent increments cannot tear the
        view."""
        with self._endpoint_counters_lock:
            return dict(self._endpoint_counters)

    def increment_intra_task_acquire_release(self) -> None:
        """KTD-J product-signal counter. Increment when a session's
        EXCLUSIVE grant is released within the same dispatch chain that
        acquired it (post-edit on the same artifact as the pre-edit).
        Documents how often fine-grained write protection actually fires
        — feeds the v0.2 / hosted-tier upsell case."""
        # CPython GIL guarantees atomicity of += on a plain int; no
        # explicit lock needed for this advisory counter.
        self._intra_task_acquire_release_total += 1

    def increment_stale_warning_emitted(self) -> None:
        """KTD-J: denominator counter for operator-computed re-read rate."""
        self._stale_warning_emitted_total += 1

    def increment_stale_warning_reread(self) -> None:
        """KTD-J: numerator counter for operator-computed re-read rate."""
        self._stale_warning_reread_total += 1

    # Finding #31 — infrastructure counters now use the same public-method
    # pattern as product-signal counters. _run_or_degrade calls these
    # instead of reaching into private attributes directly.

    def increment_watchdog_timeout(self) -> None:
        """M-03 / finding #31: increment the watchdog-timeout operator counter.
        Mirrors the increment_* pattern for product-signal counters.
        REL-03: locked so free-threading Py 3.13+ and PyPy don't tear."""
        with self._reliability_counter_lock:
            self._watchdog_timeouts_total += 1

    def increment_watchdog_queue_overflow(self) -> None:
        """M-03 / finding #31: increment the watchdog queue-overflow counter.
        REL-03: locked."""
        with self._reliability_counter_lock:
            self._watchdog_queue_overflows_total += 1

    def increment_strict_mode_denial(self) -> None:
        """v0.2 Unit 4 (KTD-V minimal): increment the strict-mode denial
        counter. Called from every strict-deny branch in the 4 PreToolUse
        handlers. CPython GIL serializes ``x += 1`` on int attrs; counter
        is advisory so we don't lock for free-threading Py 3.13+ either —
        an undercount in a pathological race is acceptable given the
        denials-only nature of the metric."""
        self._strict_mode_denials_total += 1

    def increment_strict_mode_routed_around_via_bash(self) -> None:
        """v0.2 Unit 4: bumped only in _handle_pre_bash when the strict-
        deny fires AND a prior Read strict-deny was logged for the same
        (session, path) within the route-around window. Measures the
        Phase 0 H4 routing pattern live; the ratio against
        strict_mode_denials_total tells the operator whether their model
        is exhibiting the bypass behavior in production."""
        self._strict_mode_routed_around_via_bash_total += 1

    def increment_audit_log_mode_drift(self) -> None:
        """v0.2 Unit 4: bumped when audit_log.append_strict_deny detects
        mode drift on the existing audit.log (operator chmod changed
        away from 0o600). The append still proceeds — the counter is the
        operator-visible signal to fix the mode."""
        self._audit_log_mode_drift_total += 1

    def increment_fresh_shared_hash_mismatch(self) -> None:
        """PR #108 follow-up: bumped when pre-read's fresh-SHARED branch
        observes a caller content_hash that mismatches the recorded
        non-sentinel artifact hash. The response stays fresh/allow — the
        counter (plus the additive ``hash_differs`` response field) is
        the observable. Advisory counter, same GIL-atomicity contract as
        :meth:`increment_strict_mode_denial`."""
        self._fresh_shared_hash_mismatch_total += 1

    def increment_effect_fence_hold(self) -> None:
        """Bumped when ``/hooks/effect-fence`` answers HOLD rather than
        proceed. Counts VERDICTS THE HANDLER REACHED: a watchdog-degraded hold
        is returned by ``_run_or_degrade`` after the work body was abandoned
        and is counted by ``watchdog_timeouts_total`` instead, and a 400 never
        reached a verdict at all.

        Paired with ``effect_fence_total`` it is the operator's hold RATE — how
        often the fence actually stopped something. A counter that also ticked
        on a proceed would report a healthy workspace and a wedged one
        identically. Same GIL-atomicity contract as
        :meth:`increment_strict_mode_denial`."""
        self._effect_fence_holds_total += 1

    def increment_caller_principal_absent(self) -> None:
        """Bumped when a route ADMITS a request that names an identity but
        presents no caller principal (plan U6): any accept-class route, and a
        require-class route naming an identity nobody has claimed (a client
        that predates the principal). A refusal is not counted here (it
        answered 400 and is counted by :meth:`increment_caller_principal_refused`),
        nor is a foreign principal (refused on every class). A local
        diagnostic, not a flip trigger (KTD4). Same GIL-atomicity contract as
        :meth:`increment_strict_mode_denial`."""
        self._caller_principal_absent_total += 1

    def increment_caller_principal_refused(self) -> None:
        """Bumped when a route REFUSES a request for its caller principal:
        absent for a bound identity, or foreign on any class (plan U6). Never
        bumped by an admission. Surfaced at every /status tier beside
        ``caller_principal_absent_total``. Same GIL-atomicity contract as
        :meth:`increment_strict_mode_denial`."""
        self._caller_principal_refused_total += 1

    def increment_shared_foreign_lag_suppressed(self) -> None:
        """Survivor #6 v1 (R2): bumped when a SHARED-holder hash mismatch on a
        strict path is suppressed as the benign commit→disk-write lag (this
        session's own recent commit) instead of denied. Lets an operator size
        the lag-window false-negative rate. Same GIL-atomicity contract as
        :meth:`increment_strict_mode_denial`."""
        self._shared_foreign_lag_suppressed_total += 1

    def record_strict_deny(self, session_id: str, path: str) -> None:
        """v0.2 Unit 4 route-around tracker — store the (session, path)
        pair with monotonic timestamp so a subsequent pre-bash deny on
        the same pair within STRICT_DENY_ROUTE_AROUND_WINDOW_SEC can be
        recognized as a route-around."""
        with self._recent_strict_denies_lock:
            self._recent_strict_denies[(session_id, path)] = time.monotonic()

    def check_strict_deny_route_around(self, session_id: str, path: str) -> bool:
        """v0.2 Unit 4: return True iff a prior strict-deny on (session,
        path) was recorded within STRICT_DENY_ROUTE_AROUND_WINDOW_SEC.
        Lazy GC: entries older than the window are evicted on every call
        so the dict's worst-case footprint stays bounded by active
        (session × path) cardinality within the window."""
        now = time.monotonic()
        cutoff = now - STRICT_DENY_ROUTE_AROUND_WINDOW_SEC
        with self._recent_strict_denies_lock:
            # Lazy GC.
            stale_keys = [
                k for k, ts in self._recent_strict_denies.items() if ts < cutoff
            ]
            for k in stale_keys:
                del self._recent_strict_denies[k]
            return (session_id, path) in self._recent_strict_denies

    def increment_watchdog_late_completion(self) -> None:
        """P1 #5: a watchdog-timed-out future eventually completed
        successfully — any state it mutated (e.g., an EXCLUSIVE grant
        from ``service.write``) is now in the registry without the
        agent's knowledge, since the handler had already returned a
        degraded response. Operator-visible via
        ``/status?detail=metrics`` so a phantom-grant cluster is
        diagnosable from a bug report."""
        self._watchdog_late_completion_total += 1

    def increment_watchdog_late_abort(self) -> None:
        """A6: a watchdog-timed-out future aborted at the registry write lock
        (``abort_guard``) before mutating — the late-completion mitigation
        working as intended. No phantom state landed. Operator-visible via
        ``/status?detail=metrics`` so the abort rate is distinguishable from
        the unmitigated ``watchdog_late_completion_total`` residual."""
        self._watchdog_late_aborts_total += 1

    def record_401(self) -> None:
        """P1 #6: bump ``auth_401_total`` and (deduped to once per 60s)
        emit a WARNING log explaining the most common cause — operator
        deleted ``hook.secret`` while the coordinator was running, so
        every hook request from every session now 401s and the client
        treats it as a coordinator-unavailable degrade. Without this
        signal an operator sees no symptom except "coherence stopped
        working" with no log line to point at. We deliberately do NOT
        shut down the coordinator on 401 — the secret may be restored,
        or this may be a single bad request rather than a system
        misconfig."""
        self._auth_401_total += 1
        now = time.monotonic()
        with self._auth_401_warn_lock:
            if now - self._last_401_warn_at < 60.0:
                return
            self._last_401_warn_at = now
        logger.warning(
            "auth: 401 on request — bearer mismatch or hook.secret missing. "
            "If this is the first 401 after a healthy period, check that "
            "%s/.coherence/hook.secret exists and matches the client's "
            "bearer. Subsequent 401s within 60s suppressed; total: %d.",
            self.coordinator_root,
            self._auth_401_total,
        )

    def counters_snapshot(self) -> dict[str, Any]:
        """M-03 / finding #31: stable snapshot of ALL coordinator counters
        (per-endpoint + product-signal + infrastructure + watchdog).

        ``_handle_status`` uses this instead of reaching into private attrs,
        giving a single source-of-truth for the counter set.
        """
        return {
            "watchdog_timeouts_total": self._watchdog_timeouts_total,
            "watchdog_queue_overflows_total": self._watchdog_queue_overflows_total,
            "watchdog_late_completion_total": self._watchdog_late_completion_total,
            "watchdog_late_aborts_total": self._watchdog_late_aborts_total,
            "handler_concurrency_overflows_total": (
                self._server.handler_concurrency_overflows_total
                if self._server is not None else 0
            ),
            "in_flight_drain_timed_out": self._in_flight_drain_timed_out,
            "cold_start_duration_ms": self.cold_start_duration_ms,
            "endpoint_counters": self.endpoint_counters_snapshot(),
            "intra_task_acquire_release_total": self._intra_task_acquire_release_total,
            "stale_warning_emitted_total": self._stale_warning_emitted_total,
            "strict_mode_denials_total": self._strict_mode_denials_total,
            "strict_mode_routed_around_via_bash_total": self._strict_mode_routed_around_via_bash_total,
            "audit_log_mode_drift_total": self._audit_log_mode_drift_total,
            "stale_warning_reread_total": self._stale_warning_reread_total,
            "fresh_shared_hash_mismatch_total": self._fresh_shared_hash_mismatch_total,
            "shared_foreign_lag_suppressed_total": self._shared_foreign_lag_suppressed_total,
            "effect_fence_holds_total": self._effect_fence_holds_total,
            "caller_principal_absent_total": self._caller_principal_absent_total,
            "caller_principal_refused_total": self._caller_principal_refused_total,
            "auth_401_total": self._auth_401_total,
        }

    def mark_stale_warned(self, agent_id: UUID, artifact_id: UUID) -> None:
        """KTD-J: stamp an (agent, artifact) pair as having received a
        stale warning. The next pre-read on the same pair consumes the
        marker via :meth:`consume_stale_marker` and bumps the re-read
        counter."""
        with self._stale_warned_pairs_lock:
            self._stale_warned_pairs.add((agent_id, artifact_id))

    def consume_stale_marker(self, agent_id: UUID, artifact_id: UUID) -> bool:
        """KTD-J: returns True (and clears the marker) if a stale warning
        had been emitted for this (agent, artifact) pair since the last
        consumption. Returns False otherwise. Used by pre-read entry to
        bump the re-read counter exactly once per warning cycle."""
        with self._stale_warned_pairs_lock:
            pair = (agent_id, artifact_id)
            if pair in self._stale_warned_pairs:
                self._stale_warned_pairs.remove(pair)
                return True
            return False

    def mark_compact_pending(self, session_id: str) -> None:
        """SB-10 (KTD5): flag the session for deferred re-grounding delivery.
        Set only by /hooks/session-start after it built a NON-empty payload —
        an empty session must leave the deferred path unarmed (R5).
        Idempotent: a second compaction before delivery simply re-marks."""
        with self._compact_pending_lock:
            self._compact_pending.add(session_id)

    def consume_compact_pending(self, session_id: str) -> bool:
        """SB-10 (KTD5): ATOMIC test-and-clear of the compact-pending flag.

        The deferred-delivery unit's at-most-once contract (R2) hangs on
        this primitive: two concurrent qualifying admits must never both
        see True, so the check and the clear happen under one lock hold —
        a separate has/clear pair would reopen the TOCTOU window."""
        with self._compact_pending_lock:
            if session_id in self._compact_pending:
                self._compact_pending.remove(session_id)
                return True
            return False

    def has_compact_pending(self, session_id: str) -> bool:
        """SB-10 U4 (KTD6): NON-consuming advisory peek at the flag — the
        cheap process-local dict lookup the admit handlers hoist above
        their untracked fast-path exits, so no-flag traffic keeps today's
        exact response bytes and never touches the registry. Advisory
        only: the answer can go stale the instant the lock is released;
        the authoritative at-most-once decision stays with the locked
        test-and-clear in :meth:`consume_compact_pending` at the
        allow-attach seam."""
        with self._compact_pending_lock:
            return session_id in self._compact_pending

    def expire_compact_pending(self, session_id: str) -> None:
        """SB-10 (KTD5): drop an unconsumed flag without delivering. Wired
        to the parent Stop by the deferred-delivery unit (R2: the flag's
        lifetime ends at turn end; coordinator restart clears implicitly)."""
        with self._compact_pending_lock:
            self._compact_pending.discard(session_id)

    def run_with_watchdog(
        self,
        fn: Callable[[], Any],
        abort: threading.Event | None = None,
        *,
        deadline: float | None = None,
    ) -> Any:
        """Run a callable under the 4s handler-side timeout. Raises
        :class:`FuturesTimeout` on timeout (caller decides degradation).

        ``deadline`` is the request's watchdog deadline when something earlier
        in the request already started it (a ``time.monotonic()`` instant, see
        :func:`_watchdog_deadline`): ``fn`` then gets only the time left, and
        none at all — it is not submitted — once the deadline has passed.
        ``None`` gives ``fn`` the whole ``HANDLER_TIMEOUT_SEC``.

        A6 mitigation: when the future times out, ``cancel_futures`` is not
        set so the underlying work keeps running in the pool. If the caller
        passed an ``abort`` Event (the mutating handlers do, threaded into the
        ``service.*`` calls inside ``fn``), we SET it here — the work then
        fails closed via ``registry.abort_guard`` the instant it wins the
        registry write lock, so its mutation never lands after we returned a
        degraded response. A ``fn`` that ignores ``abort`` (the read-only
        handlers) is unaffected.

        Either way we attach a done_callback for the residual: a future that
        slips past the abort window and completes successfully bumps
        ``watchdog_late_completion_total`` + logs CRITICAL; one that aborts
        cleanly bumps ``watchdog_late_aborts_total`` so operators can see the
        mitigation working.
        """
        timeout = HANDLER_TIMEOUT_SEC if deadline is None else deadline - time.monotonic()
        if timeout <= 0:
            raise FuturesTimeout()
        future = self._watchdog.submit(fn)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeout:
            if abort is not None:
                abort.set()
            future.add_done_callback(self._on_watchdog_future_done_after_timeout)
            raise

    def run_admission_lookup(self, fn: Callable[[], Any], *, deadline: float) -> Any:
        """Run the caller-principal gate's durable-store lookup in the watchdog
        pool, waiting until ``deadline`` (a ``time.monotonic()`` instant) at
        most. Raises :class:`FuturesTimeout` when it is not done by then, and
        re-raises whatever ``fn`` raised — a refusal included.

        Unlike :meth:`run_with_watchdog` there is no abort and no late-completion
        accounting: the lookup only READS, so a late one lands no coordinator
        state (it can only fill the service's binding cache), and counting it as
        a late completion would report a phantom mutation that never happened."""
        future = self._watchdog.submit(fn)
        return future.result(timeout=max(0.0, deadline - time.monotonic()))

    def _on_watchdog_future_done_after_timeout(self, future: Any) -> None:
        """Callback wired by :meth:`run_with_watchdog` when its future
        timed out. Fires later (microseconds to many seconds) when the
        underlying work actually completes. We only count + log when
        the late completion produced state — i.e., the future finished
        without raising. A future that ultimately raised was a no-op
        in the registry; nothing phantom there."""
        if future.cancelled():
            return
        try:
            future.result(timeout=0)
        except WatchdogAbandoned:
            # A6 mitigation fired: the late work aborted at the registry write
            # lock before mutating. Clean no-op (no phantom state landed) —
            # counted separately so operators can see the guard working,
            # distinct from an unmitigated late completion.
            self.increment_watchdog_late_abort()
            return
        except Exception:
            # Late failure — no phantom state landed in the registry.
            return
        self.increment_watchdog_late_completion()
        logger.critical(
            "watchdog late completion: a handler future timed out but the "
            "underlying work completed successfully afterwards. Any state "
            "it mutated (e.g., an EXCLUSIVE grant) is in the registry "
            "without the agent's knowledge. Check /status?detail=metrics "
            "for watchdog_late_completion_total and consider running "
            "agent-coherence-status --detail=full to inspect orphaned "
            "M/E grants. Counter total now: %d.",
            self._watchdog_late_completion_total,
        )


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """ThreadingMixIn + HTTPServer — concurrent hook handling per request,
    with v0.1.1 KTD-G item 2 handler concurrency semaphore.

    Per plugin docs/known-issues/2026-05-17-watchdog-races.md A7: without
    an upper bound on concurrent handler threads, a same-secret client
    issuing slow-rolling overlapping requests can saturate the watchdog
    pool's _work_queue. KTD-G item 2 caps thread creation upstream of
    the watchdog pool by acquiring a BoundedSemaphore (limit =
    HANDLER_CONCURRENCY_LIMIT = pool_size × 2) BEFORE spawning the
    handler thread. Excess requests receive HTTP 503 synchronously
    without spawning a thread.

    The semaphore is bounded so over-release surfaces as ValueError —
    catches the bug where a handler exit path forgets the release
    rather than silently allowing extra concurrent handlers.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        RequestHandlerClass: type,
        *,
        concurrency_limit: int,
    ) -> None:
        super().__init__(server_address, RequestHandlerClass)
        self._concurrency_sem = threading.BoundedSemaphore(concurrency_limit)
        # KTD-G item 3: surfaced in /status. REL-03 (free-threading-safe):
        # increment under a lock so concurrent process_request calls
        # don't tear the counter on Py 3.13+ free-threaded builds or
        # PyPy. The lock is on the hot path — but only the over-limit
        # case fires it (cold path), so the overhead is negligible.
        self.handler_concurrency_overflows_total: int = 0
        self._overflow_counter_lock = threading.Lock()

    def server_bind(self) -> None:
        """Bind without the stdlib's reverse-DNS lookup.

        Regression note: http.server.HTTPServer.server_bind calls
        socket.getfqdn() to populate self.server_name, which blocks on
        the host resolver — measured at 35 s on a macOS workstation
        with a stalled resolver, blowing the coordinator spawn's 10 s
        reachability window (_spawn_detached in cli/coherence_coordinator.py).
        We bind 127.0.0.1 only, so the FQDN is purely cosmetic: bind at
        the TCPServer layer and fill server_name/server_port from the
        bound address ourselves. Never call socket.getfqdn here.
        """
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0] or "localhost"
        self.server_port = self.server_address[1]

    def process_request(self, request: Any, client_address: Any) -> None:
        """Override ThreadingMixIn.process_request to gate handler spawn
        on the concurrency semaphore. If at limit, send 503 directly
        without spawning a thread."""
        if not self._concurrency_sem.acquire(blocking=False):
            with self._overflow_counter_lock:
                self.handler_concurrency_overflows_total += 1
            self._send_concurrency_503(request)
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            # If thread spawn fails (rare), release so we don't leak a slot.
            self._concurrency_sem.release()
            raise

    def process_request_thread(self, request: Any, client_address: Any) -> None:
        """Override to release the concurrency semaphore in the handler
        thread's finally block, so the slot is freed when the handler
        completes (NOT when process_request returns — that happens
        immediately after thread spawn)."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._concurrency_sem.release()

    @staticmethod
    def _send_concurrency_503(request: Any) -> None:
        """Send a minimal 503 response without going through the full
        BaseHTTPRequestHandler pipeline (which would spawn a thread).
        Conforms to KTD-B.3 C1: single-key {error: lowercase phrase}.
        """
        body = b'{"error":"handler concurrency exceeded"}'
        response = (
            b"HTTP/1.1 503 Service Unavailable\r\n"
            b"Content-Type: application/json; charset=utf-8\r\n"
            b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            + body
        )
        try:
            request.sendall(response)
        except (OSError, BrokenPipeError):
            # Client gave up before we could respond; nothing to recover.
            pass


# ----------------------------------------------------------------------
# Request handler factory
# ----------------------------------------------------------------------


def _make_handler_class(coordinator: CoordinatorHTTPServer) -> type:
    """Build a BaseHTTPRequestHandler subclass closed over the coordinator
    instance. This is the stdlib idiom for handing per-server state to
    handlers without subclassing the server class."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        # Use HTTP/1.1 with Connection: close per response (keep-alive is
        # not worth the complexity at hook-call rate).
        protocol_version = "HTTP/1.0"

        # ----------------------------------------------------------
        # stdlib hook overrides
        # ----------------------------------------------------------

        def log_message(self, fmt: str, *args: Any) -> None:
            # Silence stdout/stderr — we have our own logger.
            logger.debug("http %s - " + fmt, self.address_string(), *args)

        # ----------------------------------------------------------
        # Routing
        # ----------------------------------------------------------

        def do_POST(self) -> None:
            self._dispatch("POST")

        def do_GET(self) -> None:
            self._dispatch("GET")

        def _dispatch(self, method: str) -> None:
            # KTD-I L2: acquire_handler_slot does the atomic shutting_down
            # check + counter increment. If shutdown started between this
            # call and the dispatcher entry, the slot is denied and we 503
            # without touching the SQLite registry (which may already be
            # mid-close).
            if not coordinator.acquire_handler_slot():
                self._json(503, {"error": "coordinator shutting down"})
                return
            try:
                coordinator.mark_request()

                # Auth + Host check on every endpoint
                if not verify_host(self.headers.get("Host"), coordinator.host_allowlist):
                    self._json(403, {"error": "host header not allowlisted"})
                    logger.warning(
                        "rejected request with bad Host: %r", self.headers.get("Host")
                    )
                    return
                if not verify_bearer(self.headers.get("Authorization"), coordinator.secret):
                    coordinator.record_401()
                    self._json(401, {"error": "missing or invalid bearer token"})
                    return

                # Route. R12 (Unit 6): query string is intentionally
                # separated from the route key so /status?detail=full
                # dispatches to the same handler as /status, with the
                # handler reading the detail parameter for tier gating.
                raw_path = self.path
                if "?" in raw_path:
                    route_path, query = raw_path.split("?", 1)
                else:
                    route_path, query = raw_path, ""
                self._query_string = query  # consumed by status handler
                try:
                    handler = _ROUTES.get((method, route_path))
                    if handler is None:
                        self._json(404, {"error": f"unknown route {method} {route_path}"})
                        return
                    # ADV-001: while the coordinator is draining for migration,
                    # reject NEW write-initiation requests (pre-edit) with a
                    # structured error the agent can see. Existing in-flight
                    # pre-edit→post-edit chains are allowed to complete (the
                    # post-edit endpoint is NOT in this set), and all read +
                    # observability endpoints continue to serve. Without this
                    # gate, a pre-edit landing mid-migration mints an
                    # EXCLUSIVE grant that gets immediately invalidated by
                    # the migration handler, and the agent's matching
                    # post-edit hits a dead coordinator (silent failure).
                    if coordinator.migration_draining and (method, route_path) in _MIGRATION_REJECTED_ROUTES:
                        self._json(503, {
                            "error": (
                                "coordinator is draining for backend migration; "
                                "this write was rejected. Retry after the migration "
                                "completes and the coordinator restarts."
                            ),
                        })
                        return
                    # KTD-J (Unit 8): bump the per-endpoint counter BEFORE
                    # invoking the handler so timeouts/exceptions still
                    # show up in operator-visible counters. Contract:
                    # counts attempted requests, not successful ones.
                    counter_name = _ENDPOINT_COUNTER_NAMES.get((method, route_path))
                    if counter_name is not None:
                        coordinator.increment_endpoint_counter(counter_name)
                    # The route this request dispatched to, for the handler's
                    # caller-principal posture lookup (_admit_caller).
                    self._route_key = (method, route_path)
                    # No watchdog deadline yet: the caller-principal gate starts
                    # one only when it has to read the registry.
                    self._watchdog_deadline = None
                    handler(self, coordinator)
                except Exception as exc:
                    logger.exception("unhandled error in handler for %s %s", method, route_path)
                    self._json(500, {"error": f"internal: {type(exc).__name__}"})
            finally:
                coordinator.release_handler_slot()

        # ----------------------------------------------------------
        # Helpers
        # ----------------------------------------------------------

        def _read_json(self) -> dict | None:
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(400, {"error": "invalid Content-Length"})
                return None
            # ADV-005 (defensive): reject Content-Length:0 or missing with
            # an explicit 400 rather than silently returning {}. Every POST
            # endpoint expects a body with required fields; a missing body
            # used to fall through to per-field validate_* errors ("missing
            # session_id" etc.) which mask the actual cause. Loud-at-the-
            # right-layer fails fast for hook-client serialization bugs.
            if n <= 0:
                self._json(400, {"error": "missing or empty body (Content-Length:0)"})
                return None
            # R21 (Unit 6): cap the body BEFORE rfile.read so a hostile or
            # buggy client cannot allocate an oversized buffer in the
            # coordinator process. Validates Content-Length only — chunked
            # transfer encoding is not supported by http.server in any
            # case, so a missing/zero Content-Length already short-circuits
            # above.
            if n > MAX_REQUEST_BODY_BYTES:
                self._json(
                    413,
                    {
                        "error": (
                            f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes "
                            f"(Content-Length={n})"
                        )
                    },
                )
                return None
            raw = self.rfile.read(n)
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._json(400, {"error": "invalid json"})
                return None
            if not isinstance(obj, dict):
                self._json(400, {"error": "body must be a JSON object"})
                return None
            return obj

        def _json(self, status: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

    return _Handler


# ----------------------------------------------------------------------
# Endpoint implementations
# ----------------------------------------------------------------------


def _handle_pre_read(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/pre-read — stale-read check + KTD-9 first-observation seeding.

    Notice-drain contract (COR-03 — fragile but correct; documented here
    so future refactors don't break the two-path discipline):

    A pre-read response can shape one of three ways:

    1. **Fresh (first observation OR already-seen valid grant)** —
       returns ``{"status": "fresh", ...}`` (no ``hookSpecificOutput``;
       additive keys like Unit 6 ``version`` ride along). The
       ``work_with_notice_surfacing`` wrapper at the bottom of
       this handler pops pending notices on this exact shape and
       attaches ``hookSpecificOutput.additionalContext`` onto the
       work() payload — additive keys must survive — if any
       notices were pending. Fresh response → wrapper drains.

    2. **Stale (peer commit invalidated us)** — the stale branch
       inside ``work()`` builds a ``hookSpecificOutput`` envelope
       AND pops + prepends pending notices itself (line ~890). The
       wrapper sees ``hookSpecificOutput`` present and SKIPS its own
       notice drain — single-consumer semantics, no double-pop.

    3. **Fresh with already-drained notices** — if ``work()`` itself
       drained (the stale path's behaviour), the wrapper's
       ``status == 'fresh' and 'hookSpecificOutput' not in result``
       check is False (hookSpecificOutput present) so no re-pop.

    The KTD-J ``consume_stale_marker`` call (re-read counter) fires
    BEFORE the fresh-path short-circuit so a re-read that returns
    fresh still bumps ``stale_warning_reread_total``. Moving the
    marker check after the short-circuit would silently break the
    counter; do not refactor without preserving this ordering.
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    path = body.get("path", "")
    content_hash = body.get("content_hash") or None
    # Effect-gate opt-in: a truthy ``want_owner_generation`` asks the fresh and
    # warn-stale responses to carry the pair-consistent
    # ``(version, owner_generation)``. Absent (every shipped client), the
    # response shapes below are byte-unchanged.
    want_generation = bool(body.get("want_owner_generation"))
    # Effect-gate verification read: the fence re-reads only to COMPARE
    # comparands and discards the body. Such a read must not mutate coordinator
    # state -- above all it must not re-grant SHARED to a preempted holder,
    # which would heal the grant loss the fence exists to detect and let a bare
    # re-check ADMIT (the write-preemption blind spot). It also must not consume
    # the victim's preemption notices or bump stale-warning telemetry the fence
    # throws away. Absent (every non-fence client), behavior is byte-unchanged.
    verify_only = bool(body.get("verify_only"))

    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    path_err = validate_path(path)
    if path_err:
        # Mirror the prior "missing or empty path" message shape for empty/missing
        # to keep client-side error-handling stable.
        msg = "missing or empty path" if path_err in ("path is empty", "path must be a string") else path_err
        req._json(400, {"error": msg})
        return
    err = validate_content_hash(content_hash, required=False)
    if err:
        req._json(400, {"error": err})
        return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))) is None:
        return
    # Tracked-policy gate: untracked paths fast-path to {fresh} without
    # touching SQLite (R8 false-positive budget protection). SB-10 U4
    # (KTD6): the advisory compact-pending peek — a process-local dict
    # lookup, never a registry touch — is hoisted above this exit so a
    # pending re-grounding payload still reaches an untracked admit; with
    # no flag pending, response bytes and the no-registry behavior are
    # exactly today's.
    if not coordinator.policy.is_tracked(path):
        _fast_path_json(req, coordinator, session_id, body, {"status": "fresh"})
        return

    agent_id = coordinator.register_session(session_id, read_subagent_id(body))
    now = monotonic_seconds()

    def work() -> dict:
        coordinator.service.record_heartbeat(agent_id=agent_id, now_tick=now)
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)

        if artifact_id is None:
            # First observation per KTD-9 — seed v1 with the on-disk hash if
            # the caller supplied one, else use a sentinel.
            seed_hash = content_hash or ""
            artifact_id = coordinator.registry.resolve_or_register(
                path, content_hash=seed_hash
            )
            # Grant SHARED to the first reader so subsequent reads see
            # themselves as known-fresh.
            coordinator.registry.set_agent_state(
                artifact_id, agent_id, MESIState.SHARED,
                trigger="first_read", tick=now, content_hash=seed_hash,
            )
            # Unit 6: surface the version so an OCC writer can source
            # ``expected_version`` for a later ``post-edit-cas`` (additive —
            # status-based clients ignore it). First observation seeds v1.
            seeded_pair = coordinator.registry.get_artifact_and_generation(
                artifact_id
            )
            seeded_version = seeded_pair[0].version if seeded_pair else 1
            first: dict[str, Any] = {"status": "fresh", "version": seeded_version}
            # The generation rides the SAME snapshot as the version it is
            # reported beside (opt-in only, so shipped shapes stay
            # byte-identical) — never a second read a peer commit could
            # overtake.
            if want_generation and seeded_pair is not None:
                first["owner_generation"] = seeded_pair[1]
            return first

        # KTD-J (Unit 8): if a prior pre-read on this exact (agent,
        # artifact) pair emitted a stale warning, count THIS call as the
        # re-read. Increment whether the re-read returns fresh or stale —
        # the agent attempted the read either way. A verify_only fence read is
        # NOT a genuine agent re-read: skip it so the marker survives for the
        # agent's next real read and the counter is not skewed.
        if not verify_only and coordinator.consume_stale_marker(agent_id, artifact_id):
            coordinator.increment_stale_warning_reread()

        # ONE pair-atomic read backs BOTH the branch classification (version,
        # content hash) and the ownership generation reported alongside it, so
        # the comparand pair a caller receives is the very snapshot this
        # response was classified from. Reading them separately is what let a
        # peer commit slip between the two and hand back a version newer than
        # the bytes and grant the response answers for.
        artifact_pair = coordinator.registry.get_artifact_and_generation(artifact_id)
        artifact = artifact_pair[0] if artifact_pair else None
        owner_generation = artifact_pair[1] if artifact_pair else None
        agent_state = coordinator.registry.get_agent_state(artifact_id, agent_id)

        # ONE clock read for the whole decision, handed IN rather than taken
        # inside, so the lag gate and every timestamp stamped from this verdict
        # share an instant — no intra-block skew, and the deny reason stays
        # stable within the call (KTD-P).
        _now = _payloads.now_unix()
        # THE classification, not a second copy of it. Everything below keys off
        # a field of this value; the mutations the handler owes each branch —
        # the two hash-mismatch counters, the strict-deny emission, the SHARED
        # re-grant, the stale-warned marker and the notice drain — stay here,
        # because the verdict must stay a pure query for the surfaces that ask
        # it WITHOUT granting anything (``/hooks/effect-fence``). The handler
        # decides what to DO; the decision only says which way.
        decision = decide_tracked_read(
            coordinator,
            path=path,
            artifact_id=artifact_id,
            agent_id=agent_id,
            artifact=artifact,
            owner_generation=owner_generation,
            agent_state=agent_state,
            caller_content_hash=content_hash,
            now_unix=_now,
        )

        if decision.holds_valid_grant:
            # Reader has a valid grant on the current version. Unit 6:
            # include the version so an OCC writer can source
            # ``expected_version`` (additive — status clients ignore it).
            fresh: dict[str, Any] = {"status": "fresh", "version": decision.version}
            # Defense-in-depth (PR #108 follow-up): a SHARED holder whose
            # supplied disk hash mismatches the recorded content is
            # anomalous — a peer commit would have left it INVALID — so a
            # mismatch implies an out-of-band write or the commit→disk-
            # write lag observed through a warn-mode re-grant. Surface it
            # additively: the key appears ONLY when firing (exact-shape
            # status clients are untouched) and the response stays an
            # allow — the plugin path is fail-open by design; a
            # strict-mode deny knob waits on this counter proving a
            # ~zero false-positive rate. Sentinel recorded hashes carry
            # no content claim and must not fire — ``fresh_hash_differs``
            # is the arm-specific predicate that already excludes them
            # (``None``, ``""`` and the all-``f`` launch-gate sentinel);
            # the STALE arm's ``stale_hash_differs`` deliberately keeps
            # them in, so reading either field on both arms would move the
            # strict-deny gate.
            if decision.fresh_hash_differs:
                coordinator.increment_fresh_shared_hash_mismatch()
                # Survivor #6 v1: promote the SHARED-holder mismatch from a
                # fail-open allow to a strict-mode deny. A still-SHARED reader
                # proves no peer commit since its grant, so the mismatch is this
                # session's own disk diverging from the canonical: either its
                # own un-flushed recent commit (the benign commit→disk-write
                # lag, R2 — suppress) or a foreign out-of-band edit (deny).
                # reacquire() forces the fresh re-read; KTD-T leaves the grant
                # untouched so retries re-deny byte-stably (no INVALID needed).
                # Warn mode reaches neither field: the verdict consults the
                # strict-mode gate before either can be set.
                if decision.commit_lag_suppressed:
                    # Benign commit→disk-write lag (R2): count the
                    # suppression so operators can size the lag-window
                    # false-negative rate, then fall through to the
                    # warn-mode hash_differs allow.
                    coordinator.increment_shared_foreign_lag_suppressed()
                elif decision.outcome == "denied":
                    shared_summary: _payloads.StaleSummary = {
                        "path": path,
                        "current_version": decision.version,
                        # A SHARED holder was granted on the current
                        # version; that is the version it last saw — the
                        # verdict resolved that per-arm already.
                        "prior_version_seen_by_session": decision.prior_version_seen,
                        "last_writer_session_id": (
                            _last_writer_for(coordinator, artifact_id)
                            or "<unknown>"
                        ),
                        "last_writer_at_unix_ts": (
                            _last_writer_unix_ts(coordinator, artifact_id)
                            or _now
                        ),
                        "warning_generated_at_unix_ts": _now,
                        # Flat True, not a re-read of the predicate: this arm
                        # denies only ON a mismatch, so the enclosing branch
                        # has already established it.
                        "hash_differs": True,
                    }
                    return _emit_pre_read_strict_deny(
                        coordinator,
                        agent_id=agent_id,
                        session_id=session_id,
                        artifact_id=artifact_id,
                        path=path,
                        summary=shared_summary,
                        source="pre_read_shared_hash_deny",
                    )
                fresh["hash_differs"] = True
            # The read is allowed. A SHARED grant that a denied Bash / Grep
            # command re-armed certifies no read, so this one is the read to
            # record. A verify_only fence read discards its bytes and must not.
            if not verify_only and agent_state == MESIState.SHARED:
                _record_held_read(
                    coordinator, artifact_id, agent_id, trigger="held_read", tick=now,
                )
            if want_generation and decision.owner_generation is not None:
                fresh["owner_generation"] = decision.owner_generation
            return fresh

        # Stale: either first time this session sees the artifact OR they
        # were invalidated by a peer commit. ``prior_version_seen`` is the
        # arm-resolved value (one below current for an INVALIDated session,
        # ``None`` for one with no prior grant), and ``stale_hash_differs`` is
        # this arm's predicate — the caller's bytes against whatever the
        # registry recorded, no-claim seeds INCLUDED, which is exactly how the
        # launch-gate scenarios reach their deny.
        #
        # last_writer_at_unix_ts is REAL — from the artifact's updated_at
        # in the registry (semantically honest, A5). warning_generated_at
        # is now() to guarantee per-invocation variation (A5 + structural
        # defense for v0.2 strict-mode flip).
        last_writer_id = _last_writer_for(coordinator, artifact_id)
        last_writer_ts = _last_writer_unix_ts(coordinator, artifact_id) or _payloads.now_unix()
        summary: _payloads.StaleSummary = {
            "path": path,
            "current_version": decision.version,
            "prior_version_seen_by_session": decision.prior_version_seen,
            "last_writer_session_id": last_writer_id or "<unknown>",
            "last_writer_at_unix_ts": last_writer_ts,
            "warning_generated_at_unix_ts": _payloads.now_unix(),
            "hash_differs": decision.stale_hash_differs,
        }

        # v0.2 KTD-O / KTD-P: strict-mode deny branch. The gate itself lives in
        # ``decide_tracked_read`` (the fence route answers off the same rule);
        # what stays here is the EMISSION, which records the deny, bumps the
        # counter and writes the audit row.
        #
        # Gate (refined 2026-05-24 per launch-gate finding): strict-deny
        # fires when the artifact is in strict mode AND the session
        # demonstrably lacks a fresh view of the current content. Two
        # branches:
        #
        # 1. ``agent_state == INVALID`` — true preemption. Session
        #    previously held SHARED on an older version; a peer commit
        #    invalidated it. The session's context still carries the
        #    stale beliefs. Strict-deny forces explicit re-read.
        #
        # 2. ``agent_state is None AND hash_differs`` — session has no
        #    prior grant on the artifact AND the content the session
        #    just hashed at the disk does not match what the registry
        #    last recorded as the canonical content. This catches the
        #    "agent has stale beliefs from somewhere else (CLAUDE.md
        #    at session start, an earlier subagent's prose summary, a
        #    file on disk that's been peer-updated)" pattern. If hashes
        #    MATCH, the session is observing the same bytes the
        #    registry recorded — no stale content to act on — falls
        #    through to warn-mode allow.
        #
        # The hash_differs branch was added because the original "any
        # None on existing artifact = deny" rule broke the unit-test
        # setup pattern (warn-mode-style multi-session setup where
        # both sessions read the same registered artifact). The hash
        # check disambiguates "session sees identical bytes" (safe)
        # from "session sees different bytes" (must re-acknowledge).
        # Multi-model launch-gate scenarios always trigger via the
        # hash-differs branch because the synthetic SQLite injection
        # uses a sentinel hash ("f" * 64) that no real SHA-256
        # matches.
        #
        # KTD-Q: this is the Read surface only. Pre-edit reverts to
        # INVALID-only because pre-edit has no caller content_hash to
        # apply the hash_differs disambiguation. Pre-bash + pre-grep
        # capture None|INVALID via the loop's `state is not None and
        # != INVALID: continue` filter — same effective gate as the
        # original Unit 2 design, applied per-detected-path.
        #
        # KTD-T: do NOT re-grant SHARED on deny. The MESI state stays
        # INVALID (or None) across the model's retry loop so every
        # retry produces the same byte-stable deny reason text. Per
        # Phase 0 finding the model exits the retry loop after 2-5
        # attempts and routes to alternative behavior; the deny IS the
        # signal. Re-granting would let the second retry get fresh and
        # silently downgrade the operator's hard guardrail.
        if decision.outcome == "denied":
            return _emit_pre_read_strict_deny(
                coordinator,
                agent_id=agent_id,
                session_id=session_id,
                artifact_id=artifact_id,
                path=path,
                summary=summary,
                source="pre_read_strict_deny",
            )

        # Re-grant SHARED so this read doesn't re-fire stale on every call.
        # A verify_only fence read SKIPS the re-grant: healing the caller's
        # grant here would let its next bare re-check take the fresh branch and
        # ADMIT an effect whose grant a peer already revoked (the write-
        # preemption blind spot). Leaving it INVALID makes the HOLD
        # level-triggered — every bare re-check re-HOLDs until a real
        # reacquire() re-mints a live grant — matching the strict leg (KTD-T
        # never re-grants on a deny). The client discards this read's bytes.
        if not verify_only:
            coordinator.registry.set_agent_state(
                artifact_id, agent_id, MESIState.SHARED,
                trigger="post_stale_read", tick=now, content_hash=content_hash,
            )
        resp = _payloads.build_stale_response(summary)
        if want_generation and decision.owner_generation is not None:
            # The effect gate re-validates through THIS path after a sweep
            # reclaim (the zombie's re-read is warn-stale with the version
            # unchanged) — the attached generation is what lets it see the epoch
            # moved even though the version did not. It rides the SAME snapshot
            # the verdict classified from, which is what produced
            # summary.current_version, so the pair cannot disagree.
            resp["owner_generation"] = decision.owner_generation
        if not verify_only:
            # KTD-J (Unit 8): bump the stale-warning emission counter +
            # mark the pair so a follow-up pre-read counts as a re-read. A
            # fence verification read is not an agent-visible stale warning, so
            # skip both — the fence discards this response body.
            coordinator.increment_stale_warning_emitted()
            coordinator.mark_stale_warned(agent_id, artifact_id)
            # A1: if THIS session has pending preemption notices, prepend them
            # to the additionalContext so X learns about Y's revocation
            # alongside the stale-read warning. A verify_only read must NOT pop
            # them — the fence discards the body, so a destructive pop here
            # would drop the notice the agent needs on its next real read.
            notices = coordinator.registry.pop_pending_notices(
                agent_id, consume_limit=_PREEMPTION_PROSE_VERBATIM_CAP
            )
            if notices:
                notice_text = _build_preemption_text(coordinator, notices)
                resp["hookSpecificOutput"]["additionalContext"] = (
                    notice_text + "\n\n" + resp["hookSpecificOutput"]["additionalContext"]
                )
        return resp

    # Wrap _run_or_degrade so we can also pop notices for the FRESH-response
    # path (work() returned {status: "fresh"} without going through stale logic).
    #
    # COR-08: graceful-degradation note — if work() raises before this
    # wrapper can drain notices, pop_pending_notices never runs and the
    # notice stays in the DB. That's intentional: the notice will surface
    # on the next successful pre-read for the same (agent, artifact) OR
    # on session-stop OR via the F2 sweep eviction at
    # notice_evict_max_age_sec. A transient SQLite error delays — but
    # does not lose — the notice surface.
    def work_with_notice_surfacing() -> dict:
        result = work()
        # A verify_only fence read discards this body, so a destructive notice
        # pop here would silently drop the preemption notice the agent needs on
        # its next real read — skip the drain for it (the stale branch skips its
        # own pop for the same reason).
        if (
            not verify_only
            and result.get("status") == "fresh"
            and "hookSpecificOutput" not in result
        ):
            notices = coordinator.registry.pop_pending_notices(
                agent_id, consume_limit=_PREEMPTION_PROSE_VERBATIM_CAP
            )
            if notices:
                notice_text = _build_preemption_text(coordinator, notices)
                # Spread work()'s payload so additive fresh-path keys
                # (Unit 6 ``version``) survive the notice attachment —
                # rebuilding a literal dict here drops them.
                result = {
                    **result,
                    "hookSpecificOutput": _payloads.emit_allow(
                        source="pre_read_fresh_with_notice",
                        additional_context=notice_text,
                    ),
                }
        # SB-10 U4 (KTD6): the deferred re-grounding seam sits AFTER the
        # deny decision inside work() — a strict deny returns untouched and
        # keeps the flag pending — and AFTER the notice drain above, so
        # notices render before the re-grounding block. The abort token
        # rides along so a timed-out request never consumes the
        # compact-pending flag.
        return _deliver_pending_reground(
            coordinator, session_id, body, result, abort=abort
        )

    # A6: pre-read never threaded an abort into work() — its SHARED
    # re-grants are idempotent, so a late one is harmless. The token is
    # introduced here solely for the re-grounding delivery, whose pop is
    # NOT idempotent: a zombie that outlives the degraded response must not
    # claim the flag out from under the next live admit.
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work_with_notice_surfacing, abort=abort)


# ----------------------------------------------------------------------
# The effect fence, as a route
# ----------------------------------------------------------------------
#
# "May this irreversible effect still fire, and if not, WHY" asked over HTTP,
# for ONE artifact, from the five things only the caller knows: which session
# it is, which artifact it named, the version and the ownership generation it
# captured at read time, and the content hash of the bytes it actually holds.
#
# ON A ROUTE OF ITS OWN, not a flag on ``/hooks/pre-read``: that response
# already varies by freshness, and hanging a safety verdict off a shape with
# two forms invites the same misreading that produced the problem. A fence
# answer has exactly two forms and one vocabulary.
#
# THE LEG THIS SURFACE EXISTS FOR. The in-process wrapper
# (``adapters.effect_gate``) passes ``content_claim_present=True``
# unconditionally and says so at the site: the pre-read WIRE carries
# ``hash_differs`` -- a comparison -- and never the coordinator's RECORDED
# hash, so "the claim matches" and "there is no claim at all" arrive there as
# one value. This route reads the registry, so it can tell them apart, and
# ``coordinator_holds_content_claim`` is what it asks. Passing ``True`` here
# would silently reproduce the very admit this route was built to close, on
# the one surface able to close it.
#
# PURE QUERY (R7). Nothing below grants, re-grants, registers, heartbeats,
# marks, pops or records. The pre-read path does all six around the same
# decision -- it registers the session BEFORE its work body (and session-start
# reads that map to decide whether a session was ever seen), records a
# heartbeat as the work body's first statement (the sweep reclaims grants on
# heartbeat staleness), re-grants SHARED on the stale arm, fires five
# mutations on a strict deny, and consumes the pending re-grounding flag on
# its untracked fast path. A verdict that did any of it would heal exactly the
# grant loss it is being asked about, turning a level-triggered HOLD into an
# edge-triggered one. This codebase has recorded three instances of a read on
# the safety path healing what it checks; this comment is here so the fence
# does not become the fourth.

_EFFECT_FENCE_VERDICT_PROCEED = "proceed"
"""The ONE affirmative answer: every leg of the fence confirmed."""

_EFFECT_FENCE_VERDICT_HOLD = "hold"
"""The other answer. Always carries ``reason``, drawn from
:data:`~ccs.core.exceptions.HOLD_REASONS`."""

_EFFECT_FENCE_VERSION_FIELD = "expected_version"
_EFFECT_FENCE_GENERATION_FIELD = "expected_generation"
_EFFECT_FENCE_HASH_FIELD = "content_hash"

_EFFECT_FENCE_DEGRADED_RESPONSE: dict = {
    "verdict": _EFFECT_FENCE_VERDICT_HOLD,
    "reason": HOLD_VERSION_UNCONFIRMED,
    "degraded": True,
    "held_by": "watchdog_timeout",
}
"""Watchdog-timeout envelope for ``/hooks/effect-fence``.

``_run_or_degrade``'s default is :data:`_DEFAULT_DEGRADED_RESPONSE` --
``{"status": "fresh", "degraded": true, ...}``, a PROCEED shape. That is the
right answer for a pre-read whose contract is fresh/stale and a catastrophic
one for a safety verdict: a handler that timed out ran no comparison at all,
so "fresh" would be a claim about state nothing looked at. The precedent is
:data:`_OCC_DEGRADED_RESPONSE`, which reads as failure for the same reason.

``version_unconfirmed`` is the honest reason rather than a new vocabulary
entry: ``FenceComparands`` already documents version ``0`` as the "could not
resolve" sentinel produced by *a degraded or pre-fence coordinator*, and a
watchdog timeout is precisely a coordinator that resolved nothing."""

_EFFECT_FENCE_INTERNAL_HOLD_RESPONSE: dict = {
    "verdict": _EFFECT_FENCE_VERDICT_HOLD,
    "reason": HOLD_VERSION_UNCONFIRMED,
    "degraded": True,
    "held_by": "handler_error",
}
"""Handler-exception envelope for ``/hooks/effect-fence``.

``_run_or_degrade``'s OTHER non-normal arm turns a handler exception into HTTP
200 ``{"ok": false, "reason": "internal: <Type>"}`` -- a shape carrying no
verdict and a ``reason`` from no published vocabulary, and one that NO
``degraded_response`` argument overrides. A client branching on ``verdict``
reads it as neither answer. The route therefore catches inside its own work
body and returns this, so the exception arm is a hold like every other
unconfirmed state. ``held_by`` is what distinguishes the two degraded arms for
an operator; a client needs only ``verdict``."""


def _read_expected_version(body: dict) -> tuple[int | None, str | None]:
    """Parse ``expected_version``. Returns ``(value, error)``.

    The comparand is parsed as an integer and NEVER coerced: no ``or 0``, no
    ``int(x or 0)``. A string ``"2"`` silently becoming ``2`` is a comparand
    the caller never captured being compared as though it had been, and ``0``
    is the "could not resolve" sentinel -- manufacturing one turns "I do not
    know" into a value the branch table then compares.

    ``bool`` is excluded explicitly because it IS an ``int`` subclass:
    ``isinstance(True, int)`` is ``True``, so a bare type check accepts
    ``True`` as version 1.
    """
    if _EFFECT_FENCE_VERSION_FIELD not in body:
        return None, f"missing {_EFFECT_FENCE_VERSION_FIELD}"
    raw = body[_EFFECT_FENCE_VERSION_FIELD]
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, f"{_EFFECT_FENCE_VERSION_FIELD} must be an integer"
    return raw, None


def _read_expected_generation(body: dict) -> tuple[int | None, str | None]:
    """Parse ``expected_generation``. Returns ``(value, error)``.

    OMITTED and NULL are deliberately different answers, and the distinction is
    the whole point of the field:

    - the key ABSENT is a client that never captured the comparand. A hold
      would invite a retry, and no number of identical retries can supply a
      value the caller does not have -- so it is a 400 naming the field.
    - the key present as ``null`` is a captured FACT: "the coordinator I read
      from confirmed no generation" (one too old to report them, a strict deny,
      a degraded read). That is representable in the fence vocabulary, so it
      flows through as the sentinel and lands on ``generation_unconfirmed``.

    ``None`` is never coerced to ``0``, which is a REAL generation.
    """
    if _EFFECT_FENCE_GENERATION_FIELD not in body:
        return None, f"missing {_EFFECT_FENCE_GENERATION_FIELD}"
    raw = body[_EFFECT_FENCE_GENERATION_FIELD]
    if raw is None:
        return None, None
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None, f"{_EFFECT_FENCE_GENERATION_FIELD} must be an integer or null"
    return raw, None


def _effect_fence_no_record(
    expected_version: int, expected_generation: int | None,
) -> FenceComparands:
    """Comparands for "the coordinator holds no record of this artifact".

    Three ways to get here, one answer: the path is not tracked, it is tracked
    but was never observed, or the row vanished between the lookup and the
    read. ``current_version=None`` is the VANISH sentinel, so the branch table
    answers ``input_vanished`` on its first leg.

    Reached through the same classification as every other state rather than by
    returning a literal reason: the untracked case is exactly where pre-read
    answers with a fresh SHAPE carrying no version -- which a client reads as
    the zero sentinel -- and where the handler's first-observation branch
    answers by MUTATING (registering the row, seeding v1, granting SHARED). A
    verdict may do neither, so it holds on the absence instead of minting the
    record it was asked to check.
    """
    return FenceComparands(
        expected_version=expected_version,
        current_version=None,
        expected_generation=expected_generation,
        # No row means no generation and no claim; the read was not refused
        # (there was nothing to refuse) and no grant stood.
        current_generation=None,
        read_refused=False,
        grant_did_not_stand=True,
        content_claim_present=False,
    )


def _effect_fence_comparands(
    coordinator: CoordinatorHTTPServer,
    *,
    path: str,
    agent_id: UUID,
    expected_version: int,
    expected_generation: int | None,
    caller_content_hash: str,
    now_unix: float,
) -> FenceComparands:
    """Assemble the seven fence inputs from the registry and the request.

    Every registry call here is a READ through a non-mutating accessor:
    ``lookup_artifact_id_by_name`` (never ``resolve_or_register``, which mints
    a row), the pair-atomic ``get_artifact_and_generation``, and
    ``get_agent_state``. ``decide_tracked_read`` is the extracted pre-read
    classification with the handler's mutations left behind.
    """
    # The untracked exit is HELD, not fast-pathed. pre-read's twin returns
    # ``{"status": "fresh"}`` with no version here; copying that would turn
    # "the coordinator knows nothing about this file" into "the coordinator
    # agrees nothing moved".
    if not coordinator.policy.is_tracked(path):
        return _effect_fence_no_record(expected_version, expected_generation)
    artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
    if artifact_id is None:
        return _effect_fence_no_record(expected_version, expected_generation)
    # ONE pair-atomic read backs the version, the canonical content hash and
    # the generation reported beside them -- reading them separately is what
    # lets a peer commit slip between the two and pair a fresh version with
    # bytes it never described.
    pair = coordinator.registry.get_artifact_and_generation(artifact_id)
    if pair is None:
        return _effect_fence_no_record(expected_version, expected_generation)
    artifact, owner_generation = pair

    decision = decide_tracked_read(
        coordinator,
        path=path,
        artifact_id=artifact_id,
        agent_id=agent_id,
        artifact=artifact,
        owner_generation=owner_generation,
        agent_state=coordinator.registry.get_agent_state(artifact_id, agent_id),
        caller_content_hash=caller_content_hash,
        # ONE instant for the whole request: the lag gate inside the verdict
        # and anything else stamped from it must share a clock read (KTD-P).
        now_unix=now_unix,
    )

    # The mismatch the coordinator WOULD HAVE REPORTED on the wire for this
    # read, which is what the demotion rule is defined against: the fresh shape
    # carries ``fresh_hash_differs`` (no-claim recorded hashes excluded, so a
    # still-SHARED holder is never denied against content the coordinator never
    # claimed) and the stale shape carries ``stale_hash_differs`` (they are
    # kept IN). THE TWO ARE NOT ONE PREDICATE -- picking either one for both
    # arms moves the demotion in whichever direction the survivor points.
    hash_differs = (
        decision.fresh_hash_differs if decision.holds_valid_grant
        else decision.stale_hash_differs
    )

    return FenceComparands(
        expected_version=expected_version,
        current_version=decision.version,
        expected_generation=expected_generation,
        # The demotion normaliser, not a local rule: a real integer generation
        # is worth nothing as an AUTHORITY comparand when the coordinator
        # cannot vouch that the bytes in hand are the content at that version.
        current_generation=confirmed_generation(
            decision.owner_generation, content_hash_differs=hash_differs,
        ),
        # A strict-mode deny is a REFUSED re-validate read. It holds on the
        # refusal alone -- this route is precisely the caller the fence
        # docstring warns about, one holding the registry's real generation,
        # which would otherwise fall past that leg and either mislabel its hold
        # or proceed on a view the coordinator had just refused to serve.
        read_refused=decision.outcome == "denied",
        # A peer's pessimistic write-acquire ends a holder's grant while moving
        # NEITHER comparand, so this is the only leg that can see it.
        grant_did_not_stand=not decision.holds_valid_grant,
        # THE LEG ONLY THIS SURFACE CAN ANSWER. ``None``, ``""`` and the all-f
        # launch-gate sentinel are three spellings of "nothing was recorded";
        # the core normaliser owns which is which. Hard-coding ``True`` here
        # (the honest in-process limit) would admit every no-claim artifact.
        content_claim_present=coordinator_holds_content_claim(artifact.content_hash),
    )


def _effect_fence_verdict(
    coordinator: CoordinatorHTTPServer,
    *,
    path: str,
    agent_id: UUID,
    expected_version: int,
    expected_generation: int | None,
    caller_content_hash: str,
    now_unix: float,
) -> dict:
    """Classify, then serialise. ONE classification, shared with the in-process
    wrapper -- a second implementation of a safety rule is what
    ``ccs.core.fence`` exists to remove.

    Raises:
        AssertionError: ``classify_hold`` produced a reason outside the
            published set. The vocabulary is declared protocol (R3), so an
            unlisted reason is a drift fault, not a verdict; raising sends it
            into this route's own hold envelope rather than teaching a client a
            word no published set contains. Written as an explicit raise rather
            than ``assert`` so ``python -O`` cannot strip the check.
    """
    comparands = _effect_fence_comparands(
        coordinator,
        path=path,
        agent_id=agent_id,
        expected_version=expected_version,
        expected_generation=expected_generation,
        caller_content_hash=caller_content_hash,
        now_unix=now_unix,
    )
    reason = classify_hold(comparands)
    if reason is None:
        return {"verdict": _EFFECT_FENCE_VERDICT_PROCEED}
    if reason not in HOLD_REASONS:
        raise AssertionError(
            f"internal: classify_hold() returned {reason!r}, which is not a "
            f"member of HOLD_REASONS -- the published reason vocabulary and "
            f"the branch table have drifted; refusing to put it on the wire"
        )
    return {"verdict": _EFFECT_FENCE_VERDICT_HOLD, "reason": reason}


def _handle_effect_fence(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/effect-fence — may this irreversible effect still fire?

    Request (every field required; one artifact per call)::

        {"session_id": "<uuid>",          # grant standing is PER SESSION
         "path": "<repo-relative path>",
         "expected_version": <int>,       # the caller's captured comparands
         "expected_generation": <int|null>,
         "content_hash": "<64 hex>",      # the bytes the caller actually holds
         "agent_id": "<subagent id>"}     # optional, as on every hook route

    Response, always HTTP 200 for a protocol outcome::

        {"verdict": "proceed"}
        {"verdict": "hold", "reason": "<one of HOLD_REASONS>"}

    and HTTP 400 ``{"error": "<what to send>"}`` for a request that never
    reached a verdict. A MALFORMED REQUEST IS NEVER A HOLD: a hold invites a
    retry, and no number of retries supplies a comparand the caller never
    captured. The sharpest case is why it matters -- a caller resolving to an
    agent the coordinator has never seen legitimately produces a hold that no
    identical retry can clear, so if an incomplete request also answered
    "hold", the fixable and the unfixable would be one answer on the wire.

    THE PYTHON COORDINATOR ONLY. The sibling Node backend does not implement
    this route and answers 404 for it -- an unimplemented route is a visible
    gap, whereas two coordinators disagreeing about whether an effect may fire
    is a silent one.

    Stays OUT of ``_MIGRATION_REJECTED_ROUTES``: it initiates no write, and a
    drained fence would degrade to a hold for readers that have nothing to do
    with the migration.
    """
    body = req._read_json()
    if body is None:
        return

    session_id = body.get("session_id")
    path = body.get("path", "")

    # Validated in the order the sibling hook routes use, through the SAME
    # server-side gates: identity, then path, then the comparands, then the
    # hash. Skipping the path gate here would let an absolute or traversing
    # path fall through to the policy check, which answers False and therefore
    # HOLDS -- a 200 that looks like a safe answer while the boundary check
    # never ran.
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    path_err = validate_path(path)
    if path_err:
        # Mirror the sibling routes' message shape for empty/missing.
        msg = "missing or empty path" if path_err in ("path is empty", "path must be a string") else path_err
        req._json(400, {"error": msg})
        return

    expected_version, version_err = _read_expected_version(body)
    if version_err:
        req._json(400, {"error": version_err})
        return
    expected_generation, generation_err = _read_expected_generation(body)
    if generation_err:
        req._json(400, {"error": generation_err})
        return

    # REQUIRED here, unlike on pre-read where the caller may not have it yet.
    # Without the caller's own content hash the generation demotion can never
    # fire, and the fence silently loses the one leg that catches a decision
    # derived from superseded bytes. Named as missing in its own words so a
    # caller can tell WHICH field it forgot.
    if body.get(_EFFECT_FENCE_HASH_FIELD) is None:
        req._json(400, {"error": f"missing {_EFFECT_FENCE_HASH_FIELD}"})
        return
    hash_err = validate_content_hash(body[_EFFECT_FENCE_HASH_FIELD], required=True)
    if hash_err:
        req._json(400, {"error": hash_err})
        return
    content_hash: str = body[_EFFECT_FENCE_HASH_FIELD]

    # DERIVED, never registered. ``register_session`` is a mutation the
    # pre-read path performs before its work body, and session-start reads the
    # resulting map to decide whether a session was ever seen -- so registering
    # here would let a verdict answer a LATER question about its own caller.
    # The derivation is deterministic, so an unknown session resolves to an
    # agent with no grant and holds, which is the correct answer.
    # A malformed subagent id is REFUSED here, not degraded to the parent.
    # ``read_subagent_id`` resolves an out-of-shape value to ``None`` -- the
    # parent identity -- and the session-stop guard's comment scopes that
    # allowance to "read paths (where degrading to parent attribution is
    # benign)". This is not one: grant standing is the single leg computed
    # PER AGENT, so answering about the parent would tell a preempted
    # subagent to proceed while the parent still holds SHARED. Every other
    # malformed field on this route answers 400; so does this one.
    if read_subagent_id(body) is None and has_subagent_id_field(body):
        req._json(400, {"error": "agent_id must be 1-64 chars of [A-Za-z0-9_-]"})
        return
    if _admit_caller(req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))) is None:
        return
    agent_id = session_to_agent_id(session_id, read_subagent_id(body))

    def work() -> dict:
        try:
            verdict = _effect_fence_verdict(
                coordinator,
                path=path,
                agent_id=agent_id,
                expected_version=expected_version,
                expected_generation=expected_generation,
                caller_content_hash=content_hash,
                # ONE clock read for the whole request.
                now_unix=_payloads.now_unix(),
            )
        except Exception as exc:
            # The wrapper's exception arm answers ``{"ok": false, "reason":
            # "internal: ..."}`` and no parameter overrides it, so the route
            # owns this arm itself. See _EFFECT_FENCE_INTERNAL_HOLD_RESPONSE.
            logger.exception("effect fence verdict failed: %s", exc)
            verdict = dict(_EFFECT_FENCE_INTERNAL_HOLD_RESPONSE)
        if verdict["verdict"] == _EFFECT_FENCE_VERDICT_HOLD:
            # Advisory, and the ONE thing a fence call moves. It counts
            # verdicts the handler REACHED: a watchdog-degraded hold is
            # returned by the wrapper after this body was abandoned, and is
            # counted by ``watchdog_timeouts_total`` instead.
            coordinator.increment_effect_fence_hold()
        return verdict

    # No abort Event: the work body performs no REGISTRY write, so a late
    # completion cannot land coordinator state after the degraded response.
    # It can still land the advisory hold counter above, which is deliberate
    # -- that counter's contract is "verdicts the handler REACHED", and a
    # late completion did reach one. The degraded hold the caller actually
    # received is counted by ``watchdog_timeouts_total``.
    _run_or_degrade(
        req, coordinator, work, degraded_response=_EFFECT_FENCE_DEGRADED_RESPONSE,
    )


def _handle_pre_edit(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/pre-edit — acquire EXCLUSIVE (KTD-1) + KTD-9 collision surfacing."""
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    path = body.get("path", "")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    path_err = validate_path(path)
    if path_err:
        msg = "missing or empty path" if path_err in ("path is empty", "path must be a string") else path_err
        req._json(400, {"error": msg})
        return
    if _admit_caller(req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))) is None:
        return
    # SB-10 U4 (KTD6): advisory peek hoisted above the untracked exit —
    # see the pre-read twin for the no-flag byte/behavior guarantee.
    if not coordinator.policy.is_tracked(path):
        _fast_path_json(req, coordinator, session_id, body, {"ok": True})
        return

    agent_id = coordinator.register_session(session_id, read_subagent_id(body))
    now = monotonic_seconds()

    def work() -> dict:
        coordinator.service.record_heartbeat(agent_id=agent_id, now_tick=now)
        # Seed the artifact row if this is the first Edit on a fresh path.
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
        if artifact_id is None:
            artifact_id = coordinator.registry.resolve_or_register(path, content_hash="")

        # v0.2 KTD-O / KTD-P: strict-mode stale-edit deny branch. If the
        # artifact is opted into strict mode AND the editor's prior state
        # is INVALID (preempted by a peer commit), deny before acquiring
        # EXCLUSIVE. The editor must re-read first to take a fresh
        # SHARED grant.
        #
        # Semantic guard (matches pre-read): strict-deny fires only on
        # editor_state == INVALID. A first-time editor (state is None on
        # an existing artifact) falls through to the normal acquire flow
        # — they have not acted on stale state, they're establishing a
        # new write claim. The strict-mode intent is "must re-read after
        # preemption," not "must read before any write."
        #
        # This is the Edit/Write surface of KTD-Q (the hooks.json matcher
        # routes both tools through /hooks/pre-edit).
        if coordinator.policy.is_strict_mode(path):
            artifact = coordinator.registry.get_artifact(artifact_id)
            editor_state = coordinator.registry.get_agent_state(artifact_id, agent_id)
            # Reverted to INVALID-only 2026-05-24 per the launch-gate
            # finding. Pre-edit has no caller content_hash so it cannot
            # apply the hash_differs disambiguation that pre-read uses
            # (see pre-read gate above). Treating None as stale here
            # would deny first-time editors on any tracked-strict
            # artifact even when their working content is current.
            # The original Unit 2 design — INVALID-only — is the right
            # gate for the Edit/Write surface: pre-edit strict-deny
            # fires after a session has been explicitly preempted.
            editor_stale = editor_state == MESIState.INVALID
            if artifact is not None and artifact.version > 0 and editor_stale:
                last_writer_id = _last_writer_for(coordinator, artifact_id)
                last_writer_ts = (
                    _last_writer_unix_ts(coordinator, artifact_id) or _payloads.now_unix()
                )
                summary: _payloads.StaleSummary = {
                    "path": path,
                    "current_version": artifact.version,
                    "prior_version_seen_by_session": _prior_version_observed(
                        coordinator, artifact_id, agent_id,
                        agent_state=editor_state,
                        current_version=artifact.version,
                    ),
                    "last_writer_session_id": last_writer_id or "<unknown>",
                    "last_writer_at_unix_ts": last_writer_ts,
                    "warning_generated_at_unix_ts": _payloads.now_unix(),
                    "hash_differs": False,  # pre-edit doesn't carry content_hash
                }
                coordinator.increment_stale_warning_emitted()
                # v0.2 Unit 4 telemetry.
                coordinator.increment_strict_mode_denial()
                coordinator.record_strict_deny(session_id, path)
                if not _audit_log.append_strict_deny(
                    coordinator.coordinator_root,
                    agent_id=session_id, path=path, tool="Edit",
                ):
                    coordinator.increment_audit_log_mode_drift()
                return {
                    "ok": False,
                    "hookSpecificOutput": _payloads.emit_strict_deny(
                        source="pre_edit_strict_deny", summary=summary,
                    ),
                    "status": "stale",
                    "summary": summary,
                }

        # A1: snapshot peers in M∪E BEFORE write so we can record preemption
        # notices for victims after the side-effecting invalidation.
        peers_in_me = _peers_in_me_excluding(coordinator, artifact_id, agent_id)
        # Detect collision BEFORE acquiring: is any other session in M∪E?
        holder_id, holder_ts = _exclusive_holder(coordinator, artifact_id, exclude_agent=agent_id)

        # Acquire EXCLUSIVE — this invalidates peers (KTD-1 single-writer).
        try:
            coordinator.service.write(agent_id=agent_id, artifact_id=artifact_id, issued_at_tick=now, abort=abort)
        except CoherenceError as exc:
            return {"ok": False, "reason": str(exc)}

        # A1: record preemption notices for the agents whose M∪E grants we
        # just silently revoked via the write() side effect. The victims
        # will see these on their next pre-read / pre-edit hook.
        for victim_id in peers_in_me:
            coordinator.registry.record_preemption_notice(
                victim_agent_id=victim_id,
                artifact_id=artifact_id,
                preempter_agent_id=agent_id,
                preempted_at_unix_ts=_payloads.now_unix(),
            )

        # Pop any notices for THIS session (the caller of pre-edit) and
        # merge into the response (A1: surface on the victim's next hook
        # of any kind).
        notices = coordinator.registry.pop_pending_notices(
            agent_id, consume_limit=_PREEMPTION_PROSE_VERBATIM_CAP
        )
        notice_text = _build_preemption_text(coordinator, notices) if notices else None

        if holder_id is not None:
            # Existing collision surfacing path. If we also have preemption
            # notices for this session, prepend them.
            # R7: name the incumbent by the agent id the registry holds it
            # under, not by the session id that id was derived from.
            resp = _payloads.build_collision_response(
                holder_session_id=holder_id.hex,
                holder_acquired_at_unix_ts=float(holder_ts or _payloads.now_unix()),
                path=path,
            )
            if notice_text:
                resp["hookSpecificOutput"]["additionalContext"] = (
                    notice_text + "\n\n" + resp["hookSpecificOutput"]["additionalContext"]
                )
            return resp

        if notice_text:
            # No collision, but THIS session was preempted previously —
            # promote {ok: true} into a hookSpecificOutput.
            return {
                "ok": True,
                "hookSpecificOutput": _payloads.emit_allow(
                    source="pre_edit_notice_only",
                    additional_context=notice_text,
                ),
            }
        return {"ok": True}

    def work_with_reground() -> dict:
        # SB-10 U4 (KTD6): allow-attach seam — after every deny decision
        # inside work(); notices are already merged into the result's
        # additionalContext, so the re-grounding block always lands last.
        # The abort token rides along so a timed-out request never
        # consumes the compact-pending flag.
        return _deliver_pending_reground(
            coordinator, session_id, body, work(), abort=abort
        )

    # On watchdog timeout this route ADMITS. _PRE_EDIT_DEGRADED_RESPONSE is
    # {ok: true, degraded: true}: no hookSpecificOutput, so the hook client
    # relays no deny and no advisory and the edit proceeds. The answer is
    # chosen without the work body's decision, so it admits the strict-mode
    # stale editor the body above would have denied. See the constant for the
    # rule a future acquire-or-fail refusal must keep.
    # A6: abort threaded into service.write so a late acquire aborts at the
    # registry lock instead of granting a phantom EXCLUSIVE (and silently
    # invalidating peers) the agent never saw. The abort covers the acquire
    # only: a late body that takes the strict-deny branch never reaches it,
    # so it completes and records a deny (counter, route-around marker,
    # audit line) for an edit the caller was already told to proceed with.
    abort = threading.Event()
    _run_or_degrade(
        req, coordinator, work_with_reground,
        degraded_response=_PRE_EDIT_DEGRADED_RESPONSE, abort=abort,
    )


def _handle_post_edit(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/post-edit — commit on success, release on failure (KTD-1)."""
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    path = body.get("path", "")
    content_hash = body.get("content_hash")  # required when success=true
    success = bool(body.get("success", True))
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    path_err = validate_path(path)
    if path_err:
        msg = "missing or empty path" if path_err in ("path is empty", "path must be a string") else path_err
        req._json(400, {"error": msg})
        return
    # content_hash is required only on success — if the tool succeeded, the
    # hook script computed it from the worktree's post-write state.
    err = validate_content_hash(content_hash, required=bool(success))
    if err:
        req._json(400, {"error": err})
        return
    # Require-class: the write is attributed to the id _admit_caller returns —
    # PresentedCaller.attributed_agent_id, the session component checked
    # against its caller principal (plan U6 / R3).
    agent_id = _admit_caller(
        req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))
    )
    if agent_id is None:
        return
    if not coordinator.policy.is_tracked(path):
        req._json(200, {"ok": True})
        return

    # Records the display name for /status; the attributed id is agent_id above.
    coordinator.register_session(session_id, read_subagent_id(body))
    now = monotonic_seconds()

    def work() -> dict:
        coordinator.service.record_heartbeat(agent_id=agent_id, now_tick=now)
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
        if artifact_id is None:
            # No prior pre-edit / pre-read; nothing to commit against.
            return {"ok": True, "note": "untracked-at-commit"}

        if not success:
            # Tool failure path — release the EXCLUSIVE grant without bumping version.
            artifact = coordinator.registry.get_artifact(artifact_id)
            if artifact is not None:
                try:
                    coordinator.service.invalidate(
                        agent_id=agent_id,
                        artifact_id=artifact_id,
                        new_version=artifact.version,
                        issuer_agent_id=agent_id,
                        issued_at_tick=now,
                        abort=abort,
                    )
                except CoherenceError as exc:
                    return {"ok": False, "reason": str(exc)}
            return {"ok": True, "released": True}

        # Success path — commit and bump version.
        try:
            coordinator.service.commit(
                agent_id=agent_id,
                artifact_id=artifact_id,
                content="",  # KTD-13 — registry stores only the hash
                issued_at_tick=now,
                content_hash=content_hash,
                abort=abort,
            )
        except StaleReadGeneration:
            # Read-generation fence: a sweep reclaimed this committer in the race
            # window between its grant and this commit. Surface the STABLE machine
            # reason (not str(exc)) so the client classifier matches exactly. The
            # reclaim already released the grant, so there is nothing to release
            # here. The {ok: false} body reads as a definite reject (fail-closed).
            current_artifact = coordinator.registry.get_artifact(artifact_id)
            body: dict = {"ok": False, "reason": STALE_READ_GENERATION_REASON}
            if current_artifact is not None:
                body["current_version"] = current_artifact.version
            return body
        except CoherenceError as exc:
            # A1 + F4: if this commit failed because the grant was preempted
            # silently, enrich the reason with the preemption context so
            # the caller (and any stream-json telemetry) sees WHO took the
            # grant and WHEN — not just the generic CoherenceError text.
            #
            # F4 (P2): consume the notice here (single-consumer semantics) so
            # the next pre-event for this (agent, artifact) does NOT re-emit
            # the same preemption prose. The subagent flagged this as a
            # double-emit hazard — the post-edit-failure response IS the
            # surfacing channel for this specific case.
            popped = coordinator.registry.pop_preemption_notice(agent_id, artifact_id)
            if popped is not None:
                preempter_id, preempted_at = popped
                # ADV-004: distinguish sweep reclamation from peer preemption.
                # The sweep uses SWEEP_RECLAMATION_PREEMPTER_ID; matching it
                # means "your heartbeat went stale (or you held the grant past
                # max-hold) and the coordinator pulled the grant back" — a
                # different failure mode than "another session committed".
                if preempter_id == SWEEP_RECLAMATION_PREEMPTER_ID:
                    reason = (
                        f"commit_not_allowed: your M/E grant on {path} was "
                        f"reclaimed by the coordinator sweep (heartbeat "
                        f"timeout or max-hold ceiling) at "
                        f"{_iso_utc(preempted_at)}. Your edit landed in your "
                        f"local worktree but the coordinator's version was "
                        f"not bumped. Re-fetch the latest via pre-read and "
                        f"retry. Underlying coordinator error: {exc}"
                    )
                    return {"ok": False, "reason": reason, "reclaimed": True}
                # R7: the popped notice carries the preempter's agent id.
                reason = (
                    f"commit_not_allowed: your EXCLUSIVE grant on {path} was "
                    f"preempted by agent {_payloads.short_session_id(preempter_id.hex)} at "
                    f"{_iso_utc(preempted_at)}. Your edit landed in your local "
                    f"worktree but will not be reflected in the coordinator's "
                    f"version. Underlying coordinator error: {exc}"
                )
                return {"ok": False, "reason": reason, "preempted": True}
            return {"ok": False, "reason": str(exc)}
        # KTD-J (Unit 8): successful commit means an EXCLUSIVE grant was
        # acquired (pre-edit) and released (post-edit) within this same
        # turn → sizes the hosted-tier upsell case.
        coordinator.increment_intra_task_acquire_release()
        return {"ok": True}

    # AC-05: post-edit's wire contract is {ok: bool}; ok-shape degraded
    # envelope keeps clients reading result.get("ok") safe on timeout.
    # A6: abort threaded into invalidate/commit so a late post-edit aborts at
    # the registry lock instead of landing a phantom version bump/invalidation.
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work, degraded_response=_OK_DEGRADED_RESPONSE, abort=abort)


def _handle_post_edit_cas(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/post-edit-cas — OCC commit (plan Unit 6, R1–R3 + fail-closed).

    The optimistic-concurrency counterpart to ``/hooks/post-edit``. The hook
    flow is ``pre-read`` (→ SHARED, returns the artifact ``version``) →
    [agent edits] → ``post-edit-cas`` carrying that ``expected_version`` —
    it **never takes the** ``/hooks/pre-edit`` **EXCLUSIVE acquire** (the OCC
    writer stays S/I; the winner is elected by ``service.commit_cas``'s
    serialized version check, not a lock on the acquire). The pessimistic
    ``pre-edit`` → ``post-edit`` flow is untouched for strict-deny callers.

    Three outcomes map to STABLE, byte-consistent bodies (plan R2):

    - WIN → ``{ok: true, version: N+1}``.
    - :class:`~ccs.core.types.ConflictDetail` → ``{ok: false,
      reason: "version_mismatch"|"other_holder"|"stale_read_generation",
      current_version: N}`` — a clean typed conflict, NOT a degrade. The client
      re-reads + retries. ``stale_read_generation`` is the read-generation
      fence: the caller's captured claim was superseded by a sweep reclamation.
    - retry-eligible transient precondition
      (:class:`~ccs.core.exceptions.OccCallerTransientError`: a peer
      invalidated the caller between its read and this CAS) → ``{ok: false,
      reason: "caller_in_transient_state", current_version: N}`` carrying the
      STABLE :data:`~ccs.core.exceptions.OCC_CALLER_TRANSIENT_REASON` so the
      client (``CoherentVolume._classify_cas_response``) routes it to
      reacquire + retry independent of the exception's human message (AC2).
    - corruption / non-retryable precondition failure (``CoherenceError`` from
      ``commit_cas``: ``expected > current``, artifact missing, or caller in
      M/E) → ``{ok: false, reason: <verbatim>}`` the client raises on.

    Fail-closed degrade: on a watchdog timeout this endpoint returns the
    DISTINCT :data:`_OCC_DEGRADED_RESPONSE` (``{ok: false, degraded: true,
    reason: "commit_unconfirmed"}``), NOT the ``{ok: true, …}``
    :data:`_OK_DEGRADED_RESPONSE` the pessimistic post-edit uses — a
    timed-out CAS returning ``ok: true`` would let the client assume its
    write landed (the load-bearing fix; see :data:`_OCC_DEGRADED_RESPONSE`).
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    path = body.get("path", "")
    content_hash = body.get("content_hash")  # always required on the OCC commit
    expected_version = body.get("expected_version")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    path_err = validate_path(path)
    if path_err:
        msg = "missing or empty path" if path_err in ("path is empty", "path must be a string") else path_err
        req._json(400, {"error": msg})
        return
    # The OCC commit always carries the bytes it just wrote — there is no
    # "release on failure" sub-mode here (that lives on the pessimistic
    # post-edit), so content_hash is unconditionally required.
    hash_err = validate_content_hash(content_hash, required=True)
    if hash_err:
        req._json(400, {"error": hash_err})
        return
    # expected_version is the OCC discriminator — a non-negative int the
    # client read from pre-read. Reject anything else at the boundary so a
    # malformed client cannot drive the CAS with a garbage comparand.
    if not isinstance(expected_version, int) or isinstance(expected_version, bool) or expected_version < 0:
        req._json(400, {"error": "expected_version must be a non-negative integer"})
        return
    # Require-class: the write is attributed to the id _admit_caller returns —
    # PresentedCaller.attributed_agent_id, the session component checked
    # against its caller principal (plan U6 / R3).
    agent_id = _admit_caller(
        req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))
    )
    if agent_id is None:
        return
    if not coordinator.policy.is_tracked(path):
        req._json(200, {"ok": True})
        return

    # Records the display name for /status; the attributed id is agent_id above.
    coordinator.register_session(session_id, read_subagent_id(body))
    now = monotonic_seconds()

    def work() -> dict:
        coordinator.service.record_heartbeat(agent_id=agent_id, now_tick=now)
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
        if artifact_id is None:
            # No prior pre-read seeded the artifact; nothing to CAS against.
            # Mirror post-edit's untracked-at-commit fast path.
            return {"ok": True, "note": "untracked-at-commit"}

        # OCC commit — version-checked CAS. Does NOT take EXCLUSIVE: the
        # caller is S (from pre-read) or I (preempted); commit_cas rejects an
        # M/E caller (D4) by raising CoherenceError.
        try:
            result = coordinator.service.commit_cas(
                agent_id=agent_id,
                artifact_id=artifact_id,
                expected_version=expected_version,
                content_hash=content_hash,
                issued_at_tick=now,
                abort=abort,
            )
        except OccCallerTransientError:
            # Retry-eligible (AC2): a peer invalidated the caller between its
            # read and this CAS. Surface a STABLE machine reason decoupled from
            # the exception's human message so the client's retry classifier
            # (CoherentVolume._classify_cas_response) matches exactly. Include
            # current_version when readily available (the caller is INVALID, so
            # the artifact still exists) so the client can advance its comparand.
            current_artifact = coordinator.registry.get_artifact(artifact_id)
            body: dict = {"ok": False, "reason": OCC_CALLER_TRANSIENT_REASON}
            if current_artifact is not None:
                body["current_version"] = current_artifact.version
            return body
        except CoherenceError as exc:
            # Corruption (expected > current) or a non-retryable precondition
            # (caller in M/E) — the client raises on the {ok: false, reason}
            # body verbatim.
            return {"ok": False, "reason": str(exc)}

        if isinstance(result, ConflictDetail):
            # Retry-eligible typed conflict: NO mutation happened. Byte-stable
            # body the client maps to reacquire() + retry.
            return {
                "ok": False,
                "reason": result.reason,
                "current_version": result.current_version,
            }

        updated, _signals = result
        # WIN: commit_cas already did the peer-invalidation + S/I→SHARED
        # transition atomically. A successful OCC commit acquired no separate
        # EXCLUSIVE grant (it ends SHARED, not MODIFIED), but it IS a write that
        # bumped the version, so it exercises fine-grained write protection —
        # mirror post-edit's signal.
        coordinator.increment_intra_task_acquire_release()
        return {"ok": True, "version": updated.version}

    # Fail-closed: the OCC commit degrades to a body that reads as FAILURE.
    # A timed-out commit_cas whose future the watchdog does NOT cancel may
    # still run to completion later. A6: the abort token (below) makes the
    # dominant case — work blocked on the registry write lock — abort before
    # it lands (tracked via watchdog_late_aborts_total). The residual is bounded
    # and honest: in the contended case the late CAS sees the advanced
    # version → version_mismatch, no mutation; in the uncontended case it
    # lands N+1 AFTER the client gave up — a phantom/duplicate version bump
    # from the SAME edit, NOT a lost write (NoLostUpdate still holds). Full
    # prevention (a per-attempt fencing/idempotency token) needs surface
    # beyond v1 (no new reservation / no migration) and is DEFERRED to the
    # cross-host follow-on. v1's only obligation: the client must never
    # mistake a degraded CAS for success — hence _OCC_DEGRADED_RESPONSE
    # ({ok: false, …}), never _OK_DEGRADED_RESPONSE.
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work, degraded_response=_OCC_DEGRADED_RESPONSE, abort=abort)


def _handle_session_stop(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/session-stop — release uncommitted EXCLUSIVE grants (KTD-11)."""
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if _admit_caller(req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))) is None:
        return

    subagent_id = read_subagent_id(body)
    # P1 subagent-stop safety: session-stop RELEASES grants, so a malformed
    # agent_id must NOT silently degrade to the parent identity — that would
    # release the PARENT's live grants. Absent agent_id = a legitimate parent
    # Stop (release the parent's own grants, unchanged). Present-but-invalid =
    # refuse (no-op), since the caller intended a scoped subagent release the
    # server can't resolve. The read paths (where degrading to parent
    # attribution is benign) deliberately do NOT carry this guard.
    if subagent_id is None and has_subagent_id_field(body):
        req._json(200, {"ok": True, "released_artifacts": []})
        return

    # SB-10 U4 (R2): the deferred re-grounding flag lives at most one
    # parent turn. A PARENT Stop — the wire carries NO agent_id field;
    # SubagentStop always carries one (SB-25) — ends that turn, so an
    # undelivered flag expires here instead of leaking into a later turn.
    # A SubagentStop (or a malformed subagent id, which also presents the
    # field and returned above) leaves the flag untouched: the parent turn
    # is still in flight and its next admit may yet deliver.
    if not has_subagent_id_field(body):
        coordinator.expire_compact_pending(session_id)

    agent_id = coordinator.register_session(session_id, subagent_id)
    now = monotonic_seconds()

    def work() -> dict:
        coordinator.service.record_heartbeat(agent_id=agent_id, now_tick=now)
        held = coordinator.registry.artifacts_held_by_agent(
            agent_id, {MESIState.EXCLUSIVE, MESIState.MODIFIED}
        )
        released: list[str] = []
        for artifact_id in held:
            artifact = coordinator.registry.get_artifact(artifact_id)
            if artifact is None:
                continue
            try:
                coordinator.service.invalidate(
                    agent_id=agent_id,
                    artifact_id=artifact_id,
                    new_version=artifact.version,
                    issuer_agent_id=agent_id,
                    issued_at_tick=now,
                    abort=abort,
                )
                released.append(artifact.name)
            except CoherenceError as exc:
                logger.warning("session-stop release failed for %s: %s", artifact_id, exc)

        # F1 (P0): pop any pending preemption notices for the ending session.
        # The phpmac canonical case: X was preempted, but X's next action was
        # a Bash/Grep (not a tracked file op), or the turn just ended — so
        # no pre-read / pre-edit / post-edit fires to drain the notice queue.
        # Without this drain, notices orphan indefinitely (or until F2 evict).
        # We surface them in the response body (telemetry-visible via
        # stream-json `--include-hook-events`) AND, opportunistically, as
        # `additionalContext` so any post-Stop processing or human-readable
        # log still carries the signal.
        pending = coordinator.registry.pop_pending_notices(agent_id)
        notices_payload: list[dict] = []
        for art_id, preempter_id, ts in pending:
            art = coordinator.registry.get_artifact(art_id)
            # R7: this was the most reachable of the two paths — a structured
            # field carrying the preempter's session id in FULL, not just its
            # prose prefix. The keys are RENAMED rather than aliased, against
            # the module's usual additive-evolution rule: keeping
            # ``preempter_session_id`` alive would either go on emitting the
            # session id or hand a consumer that parses by exact shape an
            # agent id under a name promising a session id, which is the
            # worse of the two. The value is never absent now — the notice
            # row carries the agent id, so there is no unnamed-preempter arm
            # left to sentinel.
            notices_payload.append({
                "path": art.name if art else "<unknown-artifact>",
                "preempter_agent_id": preempter_id.hex,
                "preempter_agent_short": preempter_id.hex[:8],
                "preempted_at_unix_ts": ts,
                "preempted_at_iso": _iso_utc(ts),
            })

        response: dict = {"ok": True, "released_artifacts": released}
        if notices_payload:
            response["notices"] = notices_payload
            # Render prose for stream-json consumers / human inspection.
            response["hookSpecificOutput"] = {
                "hookEventName": "Stop",
                # Unbounded drain + full structured array in this same
                # response: the overflow is not deferred anywhere.
                "additionalContext": _build_preemption_text(
                    coordinator, pending, overflow_deferred=False
                ),
            }
        return response

    # AC-05: session-stop's wire contract is {ok: bool}; ok-shape degraded
    # envelope keeps clients reading result.get("ok") safe on timeout.
    # A6: abort threaded into the per-artifact invalidate so a late release
    # aborts before revoking a grant the registry handed to the next session.
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work, degraded_response=_OK_DEGRADED_RESPONSE, abort=abort)


def _handle_session_start(
    req: _RequestProtocol, coordinator: CoordinatorHTTPServer
) -> None:
    """POST /hooks/session-start — SB-10 U2 post-compaction re-grounding.

    Returns the re-grounding ``additionalContext`` payload for a compacted
    session and arms the process-local compact-pending flag for deferred
    delivery on the next qualifying admit (consumed by
    ``_deliver_pending_reground`` at the allow-attach seam, SB-10 U4).
    The ``source == "compact"`` gate lives client-side (U3, the
    hook-client ladder): this endpoint trusts its caller and treats every
    request as a compact event (R1).

    Shape notes:
    - Empty session → literal ``{}`` and NO flag (R5): the deferred path
      must stay unarmed when there is nothing to deliver.
    - Optional ``agent_id`` resolves per SB-25 like the read paths, but the
      payload and the flag are SESSION-scoped either way — the parent agent
      id is derivable without prior registration, and the payload always
      covers the parent plus every registered subagent.
    - No heartbeat and no observation recording: the endpoint is read-only
      toward the registry (R6 — its own snapshot must never count as the
      session "seeing" bytes; KTD-2 liveness resumes with the session's
      next admit).
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    subagent_id = read_subagent_id(body)
    if _admit_caller(req, coordinator, _presented_caller(req, session_id, subagent_id)) is None:
        return

    # R8 breadcrumb precondition: sample seen-ness BEFORE registration
    # erases it — registration below makes this session "seen" for every
    # later request, which is exactly what keeps the breadcrumb a
    # once-per-rotation signal rather than log spam.
    never_seen = coordinator.agent_name_for(session_to_agent_id(session_id)) is None
    coordinator.register_session(session_id)
    if subagent_id is not None:
        coordinator.register_session(session_id, subagent_id)

    abort = threading.Event()

    def work() -> dict:
        text, workspace_has_state = _build_session_start_context(
            coordinator, session_id, abort=abort
        )
        if never_seen and workspace_has_state:
            # R8: a compact event for a session this coordinator never saw,
            # while the workspace demonstrably holds coordination state,
            # is the signature of a silent session-id rotation (or a
            # coordinator restart) — debug-level so it is an observable,
            # not an alarm.
            logger.debug(
                "session-start for never-seen session %s while the workspace "
                "holds coordination state — possible session-id rotation or "
                "coordinator restart (SB-10 R8)",
                session_id,
            )
        if text is None:
            return {}
        # Non-empty payload → arm deferred delivery (KTD5). Ordering matters:
        # the flag is set only AFTER a successful build, so a degraded or
        # failed request leaves the deferred path unarmed.
        coordinator.mark_compact_pending(session_id)
        return {
            "hookSpecificOutput": _payloads.emit_session_start(
                additional_context=text
            )
        }

    # A6 pattern: abort threads into the builder's abort_guard so a
    # watchdog-timed-out request fails closed at the registry lock instead
    # of arming the flag after the client already saw the degraded `{}`.
    _run_or_degrade(
        req,
        coordinator,
        work,
        degraded_response=_SESSION_START_DEGRADED_RESPONSE,
        abort=abort,
    )


def _handle_policy_track(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /policy/track — Unit 6 CLI add to tracked.yaml.

    P2 ce-review fixes:
    - #4 (security YAML injection): every path passes validate_path() which
      rejects control chars (newlines), absolute paths, and ../ traversal
      before being appended to tracked.yaml. Without this, a bearer-holding
      caller could POST {"paths":["real.md\\n- injected.yaml"]} and inject
      additional patterns.
    - #11 (correctness 500→400): _append_policy_yaml's ValueError on YAML
      cap overflow is caught and returned as HTTP 400 instead of falling
      through the catch-all as a 500.
    """
    body = req._read_json()
    if body is None:
        return
    paths = body.get("paths")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        req._json(400, {"error": "paths must be a list of strings"})
        return
    if len(paths) > MAX_POLICY_PATHS_PER_REQUEST:
        req._json(400, {"error": f"max {MAX_POLICY_PATHS_PER_REQUEST} paths per request"})
        return
    # Pre-validate each path: filter out malformed entries (path traversal,
    # absolute paths, control chars including the newlines that previously
    # allowed YAML injection). Invalid paths join the response's `rejected`
    # list — preserves partial-accept semantics while defending against
    # injection into tracked.yaml.
    safe_paths: list[str] = []
    pre_rejected: list[dict] = []
    for p in paths:
        v_err = validate_path(p)
        if v_err is not None:
            pre_rejected.append({"path": p, "reason": v_err})
        else:
            safe_paths.append(p)
    yaml_path = coordinator.coordinator_root / ".coherence" / "tracked.yaml"
    try:
        added, rejected = _append_policy_yaml(yaml_path, safe_paths)
    except ValueError as exc:
        req._json(400, {"error": str(exc)})
        return
    # Reload the live policy so subsequent hook calls see the additions.
    #
    # COR-05: this is an atomic-swap-via-local-variable pattern. The RHS
    # evaluates fully (policy.reloaded() returns a new object)
    # before the attribute assignment fires. Single PyObject* write is
    # atomic on CPython, and even on free-threading builds the per-object
    # lock makes the swap visible to other threads as a single edge.
    # Handlers reading coordinator.policy bind it to a local at entry
    # (see pre-read / pre-edit / pre-bash / pre-grep) so a mid-handler
    # swap can't change which policy object the handler reasons about.
    #
    # #261: ``reloaded`` never narrows strict enforcement mid-run (a strict or
    # user-added pattern hand-removed from disk is kept until restart).
    new_policy = coordinator.policy.reloaded()
    coordinator.policy = new_policy
    req._json(200, {
        "ok": True, "added": added, "rejected": rejected + pre_rejected,
    })


def _untrack_strict_error(refused: list[dict]) -> str:
    named = "; ".join(
        f"{r['path']} (strict: {', '.join(r['strict_patterns'])})" for r in refused
    )
    return (
        f"refusing to untrack paths the coordinator enforces in strict mode: {named}. "
        "A strict path stays enforced while the coordinator runs; to untrack it, "
        "remove its entry from .coherence/strict_mode.yaml and restart the "
        f"coordinator ({UNTRACK_STRICT_PATH_REASON})"
    )


def _handle_policy_untrack(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /policy/untrack — Unit 6 CLI add to ignored.yaml.

    Same hardening as /policy/track: per-path validate_path call + ValueError
    → HTTP 400 mapping.

    #261: an entry (path or glob) that covers a path the live policy puts in
    strict mode is refused, and the whole request with it — HTTP 409
    ``{"ok": false, "error": <text>, "reason": "untrack_strict_path",
    "refused": [{"path", "strict_patterns"}], "rejected": [...]}`` — with
    nothing written to ignored.yaml. Untracking a strict path after a
    ``CoherentVolume`` (or any CAS writer) attached put it on the untracked
    fast path, where a CAS commit at any expected version is accepted: a
    silent lost update. Strict wins over ignore in the policy anyway
    (:meth:`TrackedArtifactPolicy.is_tracked`), so the refusal is what tells
    the operator the entry would not take effect; the way to stop enforcing a
    strict path is a coordinator restart without its strict entry.
    """
    body = req._read_json()
    if body is None:
        return
    paths = body.get("paths")
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        req._json(400, {"error": "paths must be a list of strings"})
        return
    if len(paths) > MAX_POLICY_PATHS_PER_REQUEST:
        req._json(400, {"error": f"max {MAX_POLICY_PATHS_PER_REQUEST} paths per request"})
        return
    # Same defense-in-depth + partial-accept as /policy/track.
    # AC-06 / finding #27: collect pre_rejected so the response is symmetric
    # with /policy/track's {ok, removed, rejected} shape. Previously, invalid
    # paths were silently dropped with no rejected field in the response.
    safe_paths: list[str] = []
    pre_rejected: list[dict] = []
    for p in paths:
        v_err = validate_path(p)
        if v_err is None:
            safe_paths.append(p)
        else:
            pre_rejected.append({"path": p, "reason": v_err})
    policy = coordinator.policy
    refused: list[dict] = []
    for p in safe_paths:
        covering = policy.strict_patterns_covering(p)
        if covering:
            refused.append({"path": p, "strict_patterns": list(covering)})
    if refused:
        req._json(409, {
            "ok": False,
            "error": _untrack_strict_error(refused),
            "reason": UNTRACK_STRICT_PATH_REASON,
            "refused": refused,
            "rejected": pre_rejected,
        })
        return
    yaml_path = coordinator.coordinator_root / ".coherence" / "ignored.yaml"
    try:
        added, yaml_rejected = _append_policy_yaml(yaml_path, safe_paths)
    except ValueError as exc:
        req._json(400, {"error": str(exc)})
        return
    coordinator.policy = coordinator.policy.reloaded()
    req._json(200, {"ok": True, "removed": added, "rejected": yaml_rejected + pre_rejected})


def _record_held_read(
    coordinator: CoordinatorHTTPServer,
    artifact_id: UUID,
    agent_id: UUID,
    *,
    trigger: str,
    tick: int,
) -> None:
    """Record a read by a session that already holds SHARED on the path.

    A held grant usually certifies the current bytes already, which makes this
    a no-op. After a strict Bash / Grep deny it does not: that re-grant records
    no observation (see ``_apply_bash_grep_regrants``), so a retried Bash
    command -- or the Read the session takes instead -- finds the grant held,
    answers fresh, and is the session's first read at this version. Left
    unrecorded, the baseline stays at the version before the deny (never
    observed, on a first touch), and every message computed from it errs
    toward "nothing moved": a peer's commit loses its post-compaction flag and
    a later grant handover is reported as a write the session already read.

    The row is re-read under the registry lock because the caller classified
    the path before its deny decision: a peer's pre-edit or commit landing
    since leaves the row INVALID, and writing SHARED over it would clear a
    stale flag the session has not seen. Only SHARED is credited, so an E/M
    holder's grant is never touched, and the SHARED-to-SHARED write uses a
    trigger outside ``CLAIM_CAPTURE_TRIGGERS``, so the one thing it moves is
    ``last_observed_version``.
    """
    registry = coordinator.registry
    with registry.abort_guard():
        if registry.get_agent_state(artifact_id, agent_id) != MESIState.SHARED:
            return
        artifact = registry.get_artifact(artifact_id)
        observed = registry.last_observed_version_for(artifact_id, agent_id)
        if artifact is None or (observed is not None and observed >= artifact.version):
            return
        registry.set_agent_state(
            artifact_id, agent_id, MESIState.SHARED, trigger=trigger, tick=tick,
        )


def _apply_bash_grep_regrants(
    coordinator: CoordinatorHTTPServer,
    agent_id: UUID,
    regrants: list[tuple[UUID, str]],
    *,
    held: list[tuple[UUID, str]],
    command_runs: bool,
    tick: int,
) -> None:
    """Grant SHARED on every path a Bash / Grep command named, AFTER the deny
    decision, and record an observation only if the command will run.

    The grant is the same either way: strict pre-bash / pre-grep deny once and
    re-arm the session (Node's ``pre_bash.ts`` pins the same contract), so a
    retry of the command goes through. What depends on the decision is the
    OBSERVATION. ``set_agent_state`` records the current version as the
    agent's ``last_observed_version`` for any non-INVALID grant, and a denied
    command never ran -- crediting it told a session that a later grant
    handover left it at "the version you last saw", a version it was refused.
    An allowed command does run and does read the current bytes, exactly like
    pre-read's ``post_stale_read`` re-grant, so it must keep recording.

    ``held`` lists the paths the session already held SHARED going in. They
    need no grant, but a Bash command that runs reads them too, and the retry of
    a denied command is exactly that run -- see ``_record_held_read``. Grep
    passes none: its path set is every tracked file under its root, not what it
    showed, so crediting a held file would advance a baseline the session was
    just told to re-read.

    Keyed on the COMMAND's outcome, not the trigger and not the path's own
    strictness: a warn-only path inside a denied command was not read either.
    """
    for artifact_id, trigger in regrants:
        coordinator.registry.set_agent_state(
            artifact_id, agent_id, MESIState.SHARED,
            trigger=trigger, tick=tick, observed=command_runs,
        )
    if command_runs:
        for artifact_id, trigger in held:
            _record_held_read(coordinator, artifact_id, agent_id, trigger=trigger, tick=tick)


def _handle_pre_bash(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/pre-bash — KTD-N H4 mitigation.

    The v0.2 Phase 0 falsifiability experiment (see
    ``docs/probes/2026-05-19-ktd-e-falsifiability/REPORT.md``) confirmed
    that when a stale-read warning fires on Read, the model retries 2-5
    times then routes around via `Bash cat plan.md` — bypassing the
    coherence layer entirely if Bash is unhooked. KTD-N closes that gap
    for v0.1.1's warn mode (without this, marketplace cohort sees silent
    stale-read misses on the common Bash routing pattern).

    Detects tracked-artifact READS in the Bash command via
    ``bash_path_detector.detect_tracked_paths``. For each detected path,
    runs the same stale-vs-fresh logic as ``/hooks/pre-read``. False
    negatives are acceptable (adversarial obfuscation, command
    substitution, etc. are OUT of scope per KTD-N).

    Request: ``{session_id, command}``.
    Response:
      - ``{status: "fresh"}`` if no tracked paths detected (fast path)
      - ``{status: "fresh"}`` if all detected paths are fresh
      - ``{status: "stale", hookSpecificOutput: {...}, stale_paths: [...]}``
        if any detected path is stale; ``additionalContext`` lists the
        affected paths and prepends any pending preemption notices
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    command = body.get("command")

    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if not isinstance(command, str) or not command.strip():
        req._json(400, {"error": "missing or empty command"})
        return
    if len(command) > 16384:
        # Bash commands beyond 16K are pathological; reject rather than
        # spend CPU on the regex pipeline. Matches MAX_REQUEST_BODY_BYTES
        # spirit (KTD-K item 4 / R21 — defense in depth).
        req._json(413, {"error": "command too long"})
        return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))) is None:
        return
    # Detect tracked paths the command would read. is_tracked is the
    # policy gate — handler never touches SQLite for an untracked workspace.
    # SB-10 U4 (KTD6): advisory peek hoisted above the zero-tracked-reads
    # exit — see the pre-read twin for the no-flag byte/behavior guarantee.
    tracked_paths = detect_tracked_paths(command, coordinator.policy.is_tracked)
    if not tracked_paths:
        _fast_path_json(req, coordinator, session_id, body, {"status": "fresh"})
        return

    agent_id = coordinator.register_session(session_id, read_subagent_id(body))
    now = monotonic_seconds()

    def work() -> dict:
        coordinator.service.record_heartbeat(agent_id=agent_id, now_tick=now)
        stale_summaries: list[dict] = []
        # v0.2 KTD-Q: track the first strict + stale path encountered so we
        # can emit a strict-mode deny on the whole bash command. ANY strict-
        # stale match in the path set triggers deny per the plan's edge-case
        # contract (cat a.md b.md where a.md is strict → deny).
        strict_stale_first: dict | None = None
        # The SHARED grants this command earns, and the SHARED grants it
        # already holds, both applied only once the deny decision is known --
        # see _apply_bash_grep_regrants.
        regrants: list[tuple[UUID, str]] = []
        held: list[tuple[UUID, str]] = []
        for path in tracked_paths:
            artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
            if artifact_id is None:
                # First observation per KTD-9 — seed v1 + grant SHARED so
                # subsequent reads see fresh. ``detect_tracked_paths``
                # deduplicates, so deferring the grant cannot make a later
                # iteration read this path as stale.
                artifact_id = coordinator.registry.resolve_or_register(
                    path, content_hash=""
                )
                regrants.append((artifact_id, "first_bash_read"))
                continue
            agent_state = coordinator.registry.get_agent_state(artifact_id, agent_id)
            if agent_state is not None and agent_state != MESIState.INVALID:
                if agent_state == MESIState.SHARED:
                    held.append((artifact_id, "held_bash_read"))
                continue  # fresh on this path
            # Stale. Record summary; re-grant SHARED to suppress repeat fires.
            artifact = coordinator.registry.get_artifact(artifact_id)
            stale_summaries.append({
                "path": path,
                "current_version": artifact.version,
            })
            if (
                strict_stale_first is None
                and coordinator.policy.is_strict_mode(path)
            ):
                last_writer_id = _last_writer_for(coordinator, artifact_id)
                last_writer_ts = (
                    _last_writer_unix_ts(coordinator, artifact_id) or _payloads.now_unix()
                )
                strict_stale_first = {
                    "path": path,
                    "current_version": artifact.version,
                    "prior_version_seen_by_session": _prior_version_observed(
                        coordinator, artifact_id, agent_id,
                        agent_state=agent_state,
                        current_version=artifact.version,
                    ),
                    "last_writer_session_id": last_writer_id or "<unknown>",
                    "last_writer_at_unix_ts": last_writer_ts,
                    "warning_generated_at_unix_ts": _payloads.now_unix(),
                    "hash_differs": False,
                }
            regrants.append((artifact_id, "post_stale_bash"))

        _apply_bash_grep_regrants(
            coordinator, agent_id, regrants,
            held=held, command_runs=strict_stale_first is None, tick=now,
        )

        # v0.2 KTD-Q strict-mode deny short-circuit. If any detected path in
        # the bash command is strict + stale, deny the whole command. The
        # reason references the first strict-stale path; multi-path bash
        # commands with multiple strict-stale paths re-deny on retry with
        # the next path's reason as the model resolves them one by one
        # (bounded by the model's own retry-loop per Phase 0 finding).
        if strict_stale_first is not None:
            coordinator.increment_stale_warning_emitted()
            # v0.2 Unit 4 telemetry: count the deny, check for route-around,
            # append audit-log line.
            coordinator.increment_strict_mode_denial()
            denied_path = strict_stale_first["path"]
            if coordinator.check_strict_deny_route_around(session_id, denied_path):
                # H4 routing pattern observed in live operation — Read got
                # strict-denied earlier on this (session, path), and now
                # Bash is hitting the same artifact within the window.
                coordinator.increment_strict_mode_routed_around_via_bash()
            coordinator.record_strict_deny(session_id, denied_path)
            if not _audit_log.append_strict_deny(
                coordinator.coordinator_root,
                agent_id=session_id, path=denied_path, tool="Bash",
            ):
                coordinator.increment_audit_log_mode_drift()
            summary_typed: _payloads.StaleSummary = strict_stale_first  # type: ignore[assignment]
            return {
                "hookSpecificOutput": _payloads.emit_strict_deny(
                    source="pre_bash_strict_deny", summary=summary_typed,
                ),
                "status": "stale",
                "stale_paths": [s["path"] for s in stale_summaries],
            }

        notices = coordinator.registry.pop_pending_notices(
            agent_id, consume_limit=_PREEMPTION_PROSE_VERBATIM_CAP
        )

        if not stale_summaries and not notices:
            return {"status": "fresh"}

        # Build merged additionalContext: notices first (most-urgent),
        # then bash-multipath stale warning.
        parts: list[str] = []
        if notices:
            parts.append(_build_preemption_text(coordinator, notices))
        if stale_summaries:
            paths_str = ", ".join(
                f"{s['path']} (current v{s['current_version']})"
                for s in stale_summaries
            )
            parts.append(
                f"⚠ Bash command reads tracked artifacts that have been "
                f"updated since your session's last fresh read: {paths_str}. "
                f"The command will still execute (v0.1.1 is warn-only), but "
                f"consider re-reading via the Read tool before relying on "
                f"the output as ground truth."
            )

        resp: dict[str, Any] = {
            "hookSpecificOutput": _payloads.emit_allow(
                source="pre_bash_stale_warn",
                additional_context="\n\n".join(parts),
            ),
        }
        if stale_summaries:
            resp["status"] = "stale"
            resp["stale_paths"] = [s["path"] for s in stale_summaries]
            # KTD-J (Unit 8): one increment per stale RESPONSE, regardless
            # of how many paths the response summarizes.
            coordinator.increment_stale_warning_emitted()
        else:
            resp["status"] = "fresh"
        return resp

    def work_with_reground() -> dict:
        # SB-10 U4 (KTD6): allow-attach seam — after the strict-deny
        # decision inside work(); notices/stale prose render first. The
        # abort token rides along so a timed-out request never consumes
        # the compact-pending flag.
        return _deliver_pending_reground(
            coordinator, session_id, body, work(), abort=abort
        )

    # A6: pre-bash never threaded an abort into work() — its SHARED
    # re-grants are idempotent. The token is introduced here solely for the
    # re-grounding delivery, whose pop is NOT idempotent: a zombie that
    # outlives the degraded response must not claim the flag out from under
    # the next live admit.
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work_with_reground, abort=abort)


def _handle_pre_grep(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /hooks/pre-grep — KTD-N H4 mitigation, Grep variant.

    Same threat model as ``/hooks/pre-bash`` but for the Grep tool:
    when the model uses Grep over a directory containing tracked
    artifacts, surface stale-read warnings for any artifacts the
    session has not freshened since peer commits.

    Request: ``{session_id, search_root}`` where ``search_root`` is
    the parent-repo-relative path Grep is scanning (== Grep tool's
    ``path`` arg, empty string for workspace root).

    Response shape mirrors /hooks/pre-bash.
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    search_root = body.get("search_root", "")

    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    # search_root may be "" (workspace root). If non-empty, apply path validator.
    if search_root != "":
        v = validate_path(search_root)
        if v is not None:
            req._json(400, {"error": v})
            return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id, read_subagent_id(body))) is None:
        return
    # Find registry-known tracked artifacts under the search root.
    # SB-10 U4 (KTD6): advisory peek hoisted above the zero-tracked-
    # artifacts exit so a pending payload still reaches this admit.
    tracked_paths = coordinator.registry.artifact_names_under_prefix(search_root)
    if not tracked_paths:
        _fast_path_json(req, coordinator, session_id, body, {"status": "fresh"})
        return

    agent_id = coordinator.register_session(session_id, read_subagent_id(body))
    now = monotonic_seconds()

    def work() -> dict:
        coordinator.service.record_heartbeat(agent_id=agent_id, now_tick=now)
        stale_summaries: list[dict] = []
        # v0.2 KTD-Q: track the first strict + stale path encountered so we
        # can emit strict-mode deny on the whole grep command. Mirrors pre-bash.
        strict_stale_first: dict | None = None
        regrants: list[tuple[UUID, str]] = []
        for path in tracked_paths:
            artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
            if artifact_id is None:
                continue  # registry-listed but raced away; skip
            agent_state = coordinator.registry.get_agent_state(artifact_id, agent_id)
            if agent_state is not None and agent_state != MESIState.INVALID:
                continue
            artifact = coordinator.registry.get_artifact(artifact_id)
            stale_summaries.append({
                "path": path,
                "current_version": artifact.version,
            })
            if (
                strict_stale_first is None
                and coordinator.policy.is_strict_mode(path)
            ):
                last_writer_id = _last_writer_for(coordinator, artifact_id)
                last_writer_ts = (
                    _last_writer_unix_ts(coordinator, artifact_id) or _payloads.now_unix()
                )
                strict_stale_first = {
                    "path": path,
                    "current_version": artifact.version,
                    "prior_version_seen_by_session": _prior_version_observed(
                        coordinator, artifact_id, agent_id,
                        agent_state=agent_state,
                        current_version=artifact.version,
                    ),
                    "last_writer_session_id": last_writer_id or "<unknown>",
                    "last_writer_at_unix_ts": last_writer_ts,
                    "warning_generated_at_unix_ts": _payloads.now_unix(),
                    "hash_differs": False,
                }
            regrants.append((artifact_id, "post_stale_grep"))

        # No held paths: a Grep over a root does not read every file under it
        # -- see _apply_bash_grep_regrants.
        _apply_bash_grep_regrants(
            coordinator, agent_id, regrants,
            held=[], command_runs=strict_stale_first is None, tick=now,
        )

        # v0.2 KTD-Q strict-mode deny short-circuit. Same shape as pre-bash.
        if strict_stale_first is not None:
            coordinator.increment_stale_warning_emitted()
            # v0.2 Unit 4 telemetry — Grep does NOT contribute to the
            # route-around-via-bash counter (that's bash-specific by the
            # plan's KTD-J extension; Grep is a separate H4 surface).
            coordinator.increment_strict_mode_denial()
            denied_path = strict_stale_first["path"]
            coordinator.record_strict_deny(session_id, denied_path)
            if not _audit_log.append_strict_deny(
                coordinator.coordinator_root,
                agent_id=session_id, path=denied_path, tool="Grep",
            ):
                coordinator.increment_audit_log_mode_drift()
            summary_typed: _payloads.StaleSummary = strict_stale_first  # type: ignore[assignment]
            return {
                "hookSpecificOutput": _payloads.emit_strict_deny(
                    source="pre_grep_strict_deny", summary=summary_typed,
                ),
                "status": "stale",
                "stale_paths": [s["path"] for s in stale_summaries],
            }

        notices = coordinator.registry.pop_pending_notices(
            agent_id, consume_limit=_PREEMPTION_PROSE_VERBATIM_CAP
        )

        if not stale_summaries and not notices:
            return {"status": "fresh"}

        parts: list[str] = []
        if notices:
            parts.append(_build_preemption_text(coordinator, notices))
        if stale_summaries:
            paths_str = ", ".join(
                f"{s['path']} (current v{s['current_version']})"
                for s in stale_summaries
            )
            parts.append(
                f"⚠ Grep search over tracked artifacts your session has "
                f"not freshened since peer commits: {paths_str}. The "
                f"results may reflect outdated content. Consider re-reading "
                f"via Read before acting on Grep output."
            )

        resp: dict[str, Any] = {
            "hookSpecificOutput": _payloads.emit_allow(
                source="pre_grep_stale_warn",
                additional_context="\n\n".join(parts),
            ),
        }
        if stale_summaries:
            resp["status"] = "stale"
            resp["stale_paths"] = [s["path"] for s in stale_summaries]
            # KTD-J (Unit 8): one increment per stale pre-grep response.
            coordinator.increment_stale_warning_emitted()
        else:
            resp["status"] = "fresh"
        return resp

    def work_with_reground() -> dict:
        # SB-10 U4 (KTD6): allow-attach seam — after the strict-deny
        # decision inside work(); notices/stale prose render first. The
        # abort token rides along so a timed-out request never consumes
        # the compact-pending flag.
        return _deliver_pending_reground(
            coordinator, session_id, body, work(), abort=abort
        )

    # A6: pre-grep never threaded an abort into work() — its SHARED
    # re-grants are idempotent. The token is introduced here solely for the
    # re-grounding delivery, whose pop is NOT idempotent: a zombie that
    # outlives the degraded response must not claim the flag out from under
    # the next live admit.
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work_with_reground, abort=abort)


# ----------------------------------------------------------------------
# Snapshot-session endpoints (SB-17 / TX-1, Unit 8 — R7 / R9 / R10a)
# ----------------------------------------------------------------------
#
# Four ADDITIVE routes — registered in the central ``_ROUTES`` table so they
# ride the SAME ``verify_bearer`` + ``verify_host`` seam every other endpoint
# uses (the dispatcher applies auth before any handler runs; there is NO
# parallel router). All four derive the CALLER/OWNER identity from the request
# body's ``session_id`` via :func:`session_to_agent_id` — the same derivation the
# hook endpoints use. That ``session_id`` is caller-asserted and validated for
# UUID shape only: the dispatcher authenticates the workspace bearer, not the
# caller, so the owner is whichever identity the request NAMES. (A caller
# principal, minted by ``POST /principal/claim``, is what a route can check the
# named identity against — plan U4; which routes require it is U6.) What the
# lock guarantees is narrower: no OTHER client-supplied identity field sets the
# owner. This is the R9 server-capture boundary lock:
#
#   * ``begin_session`` captures the cut SERVER-SIDE; the client cannot supply a
#     cut / pinned versions.
#   * ``/session/read`` and ``/session/commit`` carry ONLY the ``session_token``
#     (+ artifact path / content). Any client-supplied ``cut`` / ``pinned_*`` /
#     ``owner`` / ``caller`` field is IGNORED — the server reads the pinned cut
#     from the registry by token and derives the owner from the request's
#     ``session_id``. A forged / replayed token CANNOT forge or bypass the server-side
#     capture: a token with no live cut fails closed (``session_invalidated`` /
#     ``session_not_found``), and a foreign caller raises ``SessionInvalidated``.
#   * The day a client legitimately carries the cut is CROSS-HOST — out of scope
#     here; the boundary-lock test FAILS if anyone wires a client-carried cut in.


def _session_owner_from_request(
    coordinator: CoordinatorHTTPServer, session_id: str
) -> UUID:
    """Derive the session OWNER/CALLER identity from the request's
    ``session_id`` (R9 / R13), via the SAME :func:`session_to_agent_id`
    derivation the hook endpoints use through ``register_session``.

    The ``session_id`` is body-supplied and caller-asserted — validated for UUID
    shape only, never authenticated: the bearer authenticates the workspace,
    not the caller, so this returns whichever identity the request names. The
    guarantee is narrower than "the identity comes from auth": no client-supplied
    ``owner``/``caller`` field is ever consulted. Checking the named identity
    against its caller principal is :meth:`PresentedCaller.attributed_agent_id`
    (plan U4); which routes require that check is U6. Also registers the session
    so status/name lookups stay consistent with the hook surface."""
    return coordinator.register_session(session_id)


def _resolve_read_set(
    coordinator: CoordinatorHTTPServer, paths: list[str]
) -> list[UUID]:
    """Resolve repo-relative read-set paths to artifact UUIDs, seeding a v1
    row (KTD-9 first-observation, mirroring ``/hooks/pre-read``) for any path
    not yet known. Non-mutating w.r.t. coherence STATE for already-known
    artifacts (it only looks them up); a first-observation seed registers the
    artifact row exactly as pre-read does. The returned id order matches the
    input path order so the begin handler can render a path-keyed cut."""
    ids: list[UUID] = []
    for path in paths:
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
        if artifact_id is None:
            artifact_id = coordinator.registry.resolve_or_register(path, content_hash="")
        ids.append(artifact_id)
    return ids


def _handle_session_begin(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /session/begin — open a consistent multi-artifact snapshot session.

    Request: ``{session_id, read_set: [<repo-rel path>, ...]}``.

    The CALLER/OWNER is derived from the caller-asserted ``session_id``
    (R9/R13) — the client does NOT supply a separate owner field. The server
    resolves each read-set PATH to an artifact id (seeding first-observations
    like pre-read), captures the cut
    via ``service.begin_session``, and returns the server-minted
    ``session_token`` plus the INSPECTABLE path-keyed cut. The bytes are never
    captured here (version-map only); ``retain_versions`` tells the client which
    serve branch a later ``/session/read`` resolves.

    Responses:
      - WIN → ``{ok: true, session_token, cut: {path: version}, coordinator_epoch,
        retain_versions}``
      - typed rejection (unknown id / caps) → ``{ok: false, reason, ...}``
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    read_set = body.get("read_set")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if not isinstance(read_set, list) or not all(isinstance(p, str) for p in read_set):
        req._json(400, {"error": "read_set must be a list of repo-relative path strings"})
        return
    if len(read_set) > MAX_SESSION_READ_SET_PATHS:
        req._json(400, {"error": f"read_set exceeds {MAX_SESSION_READ_SET_PATHS} paths"})
        return
    for path in read_set:
        path_err = validate_path(path)
        if path_err:
            req._json(400, {"error": path_err})
            return

    # OWNER derived from the caller-asserted session_id, never an owner field (R9/R13).
    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    owner = _session_owner_from_request(coordinator, session_id)
    now = monotonic_seconds()

    def work() -> dict:
        read_set_ids = _resolve_read_set(coordinator, read_set)
        # Keep a path↔id map so the response renders the cut path-keyed (the
        # client speaks paths; artifact UUIDs stay server-internal).
        id_to_path = {aid: path for aid, path in zip(read_set_ids, read_set)}
        result = coordinator.service.begin_session(
            read_set=read_set_ids, owner=owner, created_at_tick=now,
        )
        if isinstance(result, VersionedReadRejection):
            # Typed cap / unknown-id rejection — NO session opened, no pins.
            return {"ok": False, "reason": result.reason}
        assert isinstance(result, SnapshotSession)
        # Content-free audit (R10a): begin event with ids + pinned versions.
        _session_audit.append_session_begin(
            coordinator.coordinator_root,
            session_token=result.session_token,
            cut=result.cut,
        )
        cut_by_path = {
            id_to_path.get(aid, str(aid)): version
            for aid, version in result.cut.items()
        }
        return {
            "ok": True,
            "session_token": result.session_token,
            "cut": cut_by_path,
            "coordinator_epoch": result.coordinator_epoch,
            "retain_versions": result.retain_versions,
        }

    _run_or_degrade(req, coordinator, work, degraded_response=_OK_DEGRADED_RESPONSE)


def _session_read_content_fields(content: bytes | str) -> dict:
    """Render a pinned body for the /session/read response so the client can
    verify the hash CLIENT-SIDE (finding F5).

    A lossy ``decode("utf-8", "replace")`` corrupted non-UTF-8 bodies — the
    replacement chars made the client's hash unable to match the pinned
    ``content_hash``. So: serve valid-UTF-8 (incl. all text) as a plain
    ``content`` string (the unchanged, byte-stable wire shape), and EXACT-encode
    non-UTF-8 bytes as base64 under an explicit ``content_encoding: "base64"``
    flag (``content_b64``) so the round-trip is lossless. ``str`` content (the
    in-memory registry's stored text) serves directly."""
    if isinstance(content, bytes):
        try:
            return {"content": content.decode("utf-8")}
        except UnicodeDecodeError:
            return {
                "content_b64": base64.b64encode(content).decode("ascii"),
                "content_encoding": "base64",
            }
    return {"content": content}


def _handle_session_read(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /session/read — serve an artifact's PINNED version from a session.

    Request: ``{session_id, session_token, path}``.

    R9 boundary lock: the request carries ONLY the ``session_token`` + ``path``
    — NO client-supplied cut / pinned version / owner. The server reads the
    pinned version from the registry by token and derives the caller from the
    caller-asserted ``session_id``. A forged/replayed token or a foreign caller
    cannot forge or bypass the server-side capture.

    Responses:
      - coordinator-held pinned bytes (LAZY branch) → ``{ok: true, served:
        "content", version, content, coordinator_epoch}``. Non-UTF-8 bytes are
        served losslessly as ``{..., content_b64, content_encoding: "base64"}``
        instead of a (lossy) ``content`` string, so the client hash round-trips.
      - bytes live in the data plane (EAGER branch) → ``{ok: true, served:
        "data_plane_deferred", version, content_hash, coordinator_epoch}``
      - typed rejection (no valid pin) → ``{ok: false, reason, coordinator_epoch}``
      - foreign caller / dead session → ``{ok: false, reason: "session_invalidated"}``
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    session_token = body.get("session_token")
    path = body.get("path", "")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if not isinstance(session_token, str) or not session_token:
        req._json(400, {"error": "missing or invalid session_token"})
        return
    path_err = validate_path(path)
    if path_err:
        req._json(400, {"error": path_err})
        return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    caller = _session_owner_from_request(coordinator, session_id)

    def work() -> dict:
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
        if artifact_id is None:
            # Unknown path → cannot be in any cut. Mirror the service's
            # not-in-cut signal rather than seeding a row a read can't serve.
            return {"ok": False, "reason": "artifact_not_in_cut"}
        try:
            result = coordinator.service.session_read(
                session_token, artifact_id, caller=caller,
            )
        except SessionInvalidated as exc:
            # Foreign caller / fenced-off session — fail closed. Content-free
            # invalidate audit (R10a).
            _session_audit.append_session_invalidate(
                coordinator.coordinator_root,
                session_token=session_token, reason=exc.reason,
            )
            return {"ok": False, "reason": exc.reason}
        if isinstance(result, VersionedContent):
            resp = {
                "ok": True,
                "served": "content",
                "version": result.version,
                "coordinator_epoch": result.coordinator_epoch,
            }
            resp.update(_session_read_content_fields(result.content))
            return resp
        if isinstance(result, DataPlaneDeferredRead):
            return {
                "ok": True,
                "served": "data_plane_deferred",
                "version": result.version,
                "content_hash": result.content_hash,
                "coordinator_epoch": result.coordinator_epoch,
            }
        assert isinstance(result, SessionReadRejection)
        if result.reason == SESSION_INVALIDATED_REASON:
            _session_audit.append_session_invalidate(
                coordinator.coordinator_root,
                session_token=session_token, reason=result.reason,
            )
        return {
            "ok": False,
            "reason": result.reason,
            "coordinator_epoch": result.coordinator_epoch,
        }

    _run_or_degrade(req, coordinator, work, degraded_response=_OK_DEGRADED_RESPONSE)


def _handle_session_commit(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /session/commit — single-artifact OCC commit against the pinned base.

    Request: ``{session_id, session_token, path, content}``.

    R9 boundary lock: the request carries ONLY ``session_token`` + ``path`` +
    ``content`` — NO client-supplied cut / pinned version / expected_version /
    owner. The pinned ``expected_version`` is read SERVER-SIDE from the cut
    (``service.session_commit`` sources it from ``cut[artifact_id]``); a client
    cannot drive the CAS with a forged comparand. The caller is derived from the
    caller-asserted ``session_id``.

    Responses (preserving the shipped ``commit_cas`` taxonomy):
      - WIN → ``{ok: true, version: N+1, coordinator_epoch}``
      - retry-eligible ``ConflictDetail`` (HELD) → ``{ok: false, reason, current_version}``
      - typed validation rejection → ``{ok: false, reason, coordinator_epoch}``
      - corruption / foreign caller / dead session → ``{ok: false, reason}`` (raised
        ``CoherenceError`` / ``SessionInvalidated`` mapped to a fail-closed body)
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    session_token = body.get("session_token")
    path = body.get("path", "")
    content = body.get("content")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if not isinstance(session_token, str) or not session_token:
        req._json(400, {"error": "missing or invalid session_token"})
        return
    path_err = validate_path(path)
    if path_err:
        req._json(400, {"error": path_err})
        return
    if not isinstance(content, str):
        req._json(400, {"error": "content must be a string"})
        return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    caller = _session_owner_from_request(coordinator, session_id)
    now = monotonic_seconds()

    def work() -> dict:
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
        if artifact_id is None:
            return {"ok": False, "reason": "artifact_not_in_cut"}
        # Read the pinned base BEFORE the commit so the content-free audit can
        # record (pinned_version, committed_version) on a WIN. Server-side only.
        cut = coordinator.registry.get_session_cut(session_token)
        pinned_version = cut.get(artifact_id) if cut else None
        try:
            result = coordinator.service.session_commit(
                session_token, artifact_id, content, caller=caller, issued_at_tick=now,
                abort=abort,
            )
        except SessionInvalidated as exc:
            _session_audit.append_session_invalidate(
                coordinator.coordinator_root,
                session_token=session_token, reason=exc.reason,
            )
            return {"ok": False, "reason": exc.reason}
        except CoherenceError as exc:
            # Corruption (expected_version > current) — non-retryable. The
            # client raises on this {ok: false, reason} body.
            return {"ok": False, "reason": str(exc)}
        if isinstance(result, ConflictDetail):
            # HELD: nothing mutated. Byte-stable typed conflict the client maps
            # to "open a new session + re-read + retry".
            return {
                "ok": False,
                "reason": result.reason,
                "current_version": result.current_version,
            }
        if isinstance(result, SessionCommitRejection):
            if result.reason == SESSION_INVALIDATED_REASON:
                _session_audit.append_session_invalidate(
                    coordinator.coordinator_root,
                    session_token=session_token, reason=result.reason,
                )
            return {
                "ok": False,
                "reason": result.reason,
                "coordinator_epoch": result.coordinator_epoch,
            }
        updated, _signals = result
        # Content-free commit audit (R10a): ids + versions, no body / no hash.
        _session_audit.append_session_commit(
            coordinator.coordinator_root,
            session_token=session_token,
            artifact_id=artifact_id,
            pinned_version=pinned_version if pinned_version is not None else updated.version - 1,
            committed_version=updated.version,
        )
        return {
            "ok": True,
            "version": updated.version,
            "coordinator_epoch": coordinator.registry.coordinator_epoch,
        }

    # Fail-closed degrade: a timed-out session.commit must NOT read as success
    # (mirror the OCC commit endpoint — a degraded commit is a definite reject).
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work, degraded_response=_OCC_DEGRADED_RESPONSE, abort=abort)


def _handle_session_commit_all(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /session/commit_all — atomic multi-artifact publish against the cut.

    Request: ``{session_id, session_token, writes: [{path, content}, ...]}``.

    The batch analog of ``/session/commit``: all-or-nothing over the write-set —
    either every member advances or none do, and a torn batch is never
    observable. Same R9 boundary lock: the request carries ONLY ``session_token``
    + a list of ``{path, content}`` — NO client-supplied cut / expected_version /
    owner. Each member's ``expected_version`` is read SERVER-SIDE from the pinned
    cut (``service.session_commit_all`` sources it from ``cut[artifact_id]``), so
    a client cannot drive any member's CAS with a forged comparand. The caller is
    derived from the caller-asserted ``session_id``.

    Responses (the batch generalization of ``/session/commit``):
      - WIN → ``{ok: true, versions: {path: N+1, ...}, coordinator_epoch}``
      - retry-eligible batch conflict (HELD, nothing mutated) → ``{ok: false,
        reason: "conflict", per_artifact: {path: {reason, current_version}}}``
      - typed session rejection → ``{ok: false, reason, coordinator_epoch}``
      - corruption / foreign caller / dead session → ``{ok: false, reason}``
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    session_token = body.get("session_token")
    writes = body.get("writes")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if not isinstance(session_token, str) or not session_token:
        req._json(400, {"error": "missing or invalid session_token"})
        return
    if not isinstance(writes, list) or not writes:
        req._json(400, {"error": "writes must be a non-empty list of {path, content}"})
        return
    if len(writes) > MAX_SESSION_READ_SET_PATHS:
        req._json(400, {"error": f"writes exceeds {MAX_SESSION_READ_SET_PATHS} paths"})
        return
    paths: list[str] = []
    contents: list[str] = []
    for item in writes:
        if not isinstance(item, dict):
            req._json(400, {"error": "each write must be a {path, content} object"})
            return
        path = item.get("path", "")
        content = item.get("content")
        path_err = validate_path(path)
        if path_err:
            req._json(400, {"error": path_err})
            return
        if not isinstance(content, str):
            req._json(400, {"error": "content must be a string"})
            return
        paths.append(path)
        contents.append(content)
    # A duplicate path would collapse in the id-keyed map — reject it up front so
    # a silently-dropped member can't turn an all-or-nothing batch into a partial.
    if len(set(paths)) != len(paths):
        req._json(400, {"error": "writes contains duplicate paths"})
        return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    caller = _session_owner_from_request(coordinator, session_id)
    now = monotonic_seconds()

    def work() -> dict:
        # Resolve every path -> artifact id. An unknown path can be in no cut, so
        # the WHOLE batch is refused (all-or-nothing) — mirror session_commit's
        # not-in-cut signal rather than seeding a row a commit can't source.
        path_by_id: dict[UUID, str] = {}
        writes_map: dict[UUID, tuple[str, int | None]] = {}
        for path, content in zip(paths, contents):
            artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
            if artifact_id is None:
                return {"ok": False, "reason": "artifact_not_in_cut"}
            path_by_id[artifact_id] = path
            writes_map[artifact_id] = (content, None)
        # Pinned bases read BEFORE the commit for the content-free WIN audit.
        cut = coordinator.registry.get_session_cut(session_token)
        try:
            result = coordinator.service.session_commit_all(
                session_token, writes_map, caller=caller, issued_at_tick=now,
                abort=abort,
            )
        except SessionInvalidated as exc:
            _session_audit.append_session_invalidate(
                coordinator.coordinator_root,
                session_token=session_token, reason=exc.reason,
            )
            return {"ok": False, "reason": exc.reason}
        except CoherenceError as exc:
            # Corruption (a member's expected_version > current) — non-retryable.
            return {"ok": False, "reason": str(exc)}
        if isinstance(result, MultiCommitConflict):
            # HELD: nothing mutated. Byte-stable typed per-member conflicts the
            # client maps to "open a new session + re-read + retry".
            return {
                "ok": False,
                "reason": "conflict",
                "per_artifact": {
                    path_by_id[aid]: {
                        "reason": detail.reason,
                        "current_version": detail.current_version,
                    }
                    for aid, detail in result.per_artifact.items()
                },
            }
        if isinstance(result, SessionCommitRejection):
            if result.reason == SESSION_INVALIDATED_REASON:
                _session_audit.append_session_invalidate(
                    coordinator.coordinator_root,
                    session_token=session_token, reason=result.reason,
                )
            return {
                "ok": False,
                "reason": result.reason,
                "coordinator_epoch": result.coordinator_epoch,
            }
        updated, _signals = result
        # Content-free per-member commit audit (R10a): ids + versions, no body.
        versions_by_path: dict[str, int] = {}
        for aid, new_version in updated.versions.items():
            pinned = cut.get(aid) if cut else None
            _session_audit.append_session_commit(
                coordinator.coordinator_root,
                session_token=session_token,
                artifact_id=aid,
                pinned_version=pinned if pinned is not None else new_version - 1,
                committed_version=new_version,
            )
            versions_by_path[path_by_id[aid]] = new_version
        return {
            "ok": True,
            "versions": versions_by_path,
            "coordinator_epoch": coordinator.registry.coordinator_epoch,
        }

    # Fail-closed degrade: a timed-out batch commit must NOT read as success —
    # same posture as /session/commit (a degraded commit is a definite reject).
    abort = threading.Event()
    _run_or_degrade(req, coordinator, work, degraded_response=_OCC_DEGRADED_RESPONSE, abort=abort)


def _handle_session_heartbeat(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /session/heartbeat — refresh a session's heartbeat lease (R4).

    Request: ``{session_id, session_token}``.

    The OWNER is derived from the caller-asserted ``session_id`` — a request
    naming a different session cannot keep another agent's session alive (the
    service enforces the timing-safe owner-binding and returns False on
    mismatch; the response does NOT reveal whether the token exists).
    ``{ok: true, refreshed: bool}``.
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    session_token = body.get("session_token")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if not isinstance(session_token, str) or not session_token:
        req._json(400, {"error": "missing or invalid session_token"})
        return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    owner = _session_owner_from_request(coordinator, session_id)
    now = monotonic_seconds()

    def work() -> dict:
        refreshed = coordinator.service.record_session_heartbeat(
            session_token=session_token, owner=owner, now_tick=now,
        )
        return {"ok": True, "refreshed": refreshed}

    _run_or_degrade(req, coordinator, work, degraded_response=_OK_DEGRADED_RESPONSE)


# ----------------------------------------------------------------------
# Caller-principal mint (coordinator caller principal plan, U4)
# ----------------------------------------------------------------------
#
# The coordinator authenticates the WORKSPACE bearer, not the caller: the
# acting identity every route reads is the body-supplied ``session_id``. The
# mint binds a coordinator-issued caller principal to that identity on its
# first claim, so a route can later check a request naming the identity
# against it. Accident-resistance and attributability under same-OS-user
# cooperative trust: a process that can read ``.coherence/`` can read whatever
# a hook client stores there. Which routes REQUIRE the principal is U6's
# posture table, not this route's concern.

_PRINCIPAL_CLAIM_DEGRADED_RESPONSE: dict = {
    "ok": False,
    "degraded": True,
    "reason": "claim_unconfirmed",
}
"""Fail-closed degrade envelope for ``POST /principal/claim``: a timed-out claim
must NOT read as success and carries no principal. The watchdog abort fails a
late bind closed at the registry lock; if one lands anyway, the claimant
recovers it by retrying with the SAME mint nonce (R20)."""


def _handle_principal_claim(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /principal/claim — bind a caller principal to the acting identity
    on its first claim, or hand it back to a retry of that claim (U4 / R1, R2,
    R5, R20).

    Request: ``{session_id, mint_nonce}``. ``mint_nonce`` is generated by the
    claimant and persisted BEFORE this call (KTD11); it is the only thing that
    tells a retry of a claim whose response was lost from a second claimant.

    Responses:
      - bound → ``{ok: true, principal}`` — the ONLY response a principal ever
        crosses the wire in (R5). A retry presenting the same nonce gets the
        same principal.
      - identity already bound under a different nonce → ``{ok: false, reason:
        "caller_principal_claimed", detail}`` — the claimant does not become
        the identity (R2). Never a hold.
      - malformed ``session_id`` / ``mint_nonce`` → HTTP 400 naming the field.
      - watchdog degrade → :data:`_PRINCIPAL_CLAIM_DEGRADED_RESPONSE`.
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    mint_nonce = body.get("mint_nonce")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    nonce_err = mint_nonce_problem(mint_nonce)
    if nonce_err:
        req._json(400, {"error": nonce_err})
        return
    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    identity = caller_principal_identity(session_id)

    def work() -> dict:
        try:
            principal = coordinator.service.claim_caller_principal(
                identity=identity, mint_nonce=mint_nonce, abort=abort,
            )
        except CallerPrincipalRefused as exc:
            return _typed_reason_response(exc)
        return {"ok": True, "principal": principal}

    abort = threading.Event()
    _run_or_degrade(
        req, coordinator, work, degraded_response=_PRINCIPAL_CLAIM_DEGRADED_RESPONSE, abort=abort
    )


# ----------------------------------------------------------------------
# Workspace-checkpoint endpoints (WV plan Unit 3 — R1/R2/R8)
# ----------------------------------------------------------------------
#
# Registered in the central ``_ROUTES`` table so they ride the SAME
# ``verify_bearer`` + ``verify_host`` seam as every other endpoint (no parallel
# router). The OWNER is derived from the caller-asserted ``session_id`` (the
# R9/R13 boundary lock — a client-supplied ``owner`` field is ignored), and the
# ``checkpoint_id`` is minted SERVER-SIDE by the service.
# The member rows themselves are CLIENT-captured facts (tokens, fingerprints,
# timestamps — the capture engine runs client-side against ITS substrates, the
# coordinator never sees the bytes); the boundary validates their SHAPE
# fail-closed (paths, 64-hex fingerprints, bounded opaque tokens, closed tier
# vocabularies) so a hostile client cannot store unbounded or malformed rows.

MAX_CHECKPOINT_MEMBERS = MAX_SESSION_READ_SET_PATHS
"""Cap on member rows per ``POST /workspace/checkpoint`` — the checkpoint
manifest analog of the session read-set cap (same defense-in-depth posture)."""

MAX_CHECKPOINT_NAME_LEN = 256
"""Cap on the checkpoint name (an operator label, not content)."""

MAX_NATIVE_TOKEN_LEN = 256
"""Cap on one member's opaque restore pointer — mirrors the never-ship-a-store
opaque-text bound (``ccs.core.substrate._MAX_OPAQUE_TEXT_LEN``): a token longer
than this is a content-proportional shadow wearing a token's name."""

_CHECKPOINT_TIER_VALUES: frozenset[str] = frozenset(t.value for t in RestoreTier)
_CHECKPOINT_ARBITRATION_VALUES: frozenset[str] = frozenset(
    t.value for t in ArbitrationTier
)

_CHECKPOINT_DEGRADED_RESPONSE: dict = {
    "ok": False,
    "degraded": True,
    "reason": "checkpoint_unconfirmed",
}
"""Fail-closed degrade envelope for ``POST /workspace/checkpoint`` (the
``_OCC_DEGRADED_RESPONSE`` posture): a watchdog-timed-out registration must
NOT read as success — the manifest may or may not have landed, and the abort
Event threaded into the service's ``abort_guard`` fails the late write closed
at the registry lock (the A6 session-commit lesson)."""


def _typed_reason_response(exc: BaseException) -> dict:
    """The stable-reason reject envelope for the workspace routes.

    A typed exception's identity-stable ``reason`` token goes on the wire (the
    ``/workspace/restore/register`` house pattern — clients classify by token
    identity, never by substring-matching prose), and the human prose rides in
    a separate ``detail`` field so nothing is lost. An exception carrying no
    ``reason`` falls back to its prose as before.
    """
    reason = getattr(exc, "reason", None)
    if isinstance(reason, str) and reason:
        return {"ok": False, "reason": reason, "detail": str(exc)}
    return {"ok": False, "reason": str(exc)}


def _unknown_checkpoint_response(exc: KeyError) -> dict:
    """The registry's unknown-checkpoint/member ``KeyError`` mapped to the SAME
    stable token ``/workspace/restore/register`` emits for that failure class
    (:data:`~ccs.core.exceptions.CHECKPOINT_UNKNOWN_REASON`), prose preserved
    in ``detail`` (``KeyError`` str() wraps its arg in quotes, so unwrap)."""
    detail = str(exc.args[0]) if exc.args else str(exc)
    return {"ok": False, "reason": CHECKPOINT_UNKNOWN_REASON, "detail": detail}


def _parse_checkpoint_member(item: object) -> "CheckpointMember | str":
    """Validate ONE wire member object into a :class:`CheckpointMember`, or
    return the boundary-rejection reason string (fail-closed shape gate)."""
    if not isinstance(item, dict):
        return "each member must be a JSON object"
    member_path = item.get("member_path", "")
    path_err = validate_path(member_path)
    if path_err:
        return f"member_path: {path_err}"
    native_token = item.get("native_token")
    if native_token is not None and (
        not isinstance(native_token, str)
        or not native_token
        or len(native_token) > MAX_NATIVE_TOKEN_LEN
    ):
        return (
            f"native_token must be null or a non-empty string of at most "
            f"{MAX_NATIVE_TOKEN_LEN} chars"
        )
    fingerprint = item.get("fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str) or not _CONTENT_HASH_RE.match(fingerprint)
    ):
        return "fingerprint must be null or a 64-hex sha-256 digest"
    captured_at = item.get("captured_at")
    if isinstance(captured_at, bool) or not isinstance(captured_at, (int, float)):
        return "captured_at must be a number (monotonic_seconds basis)"
    absent = item.get("absent", False)
    dirty = item.get("dirty_during_window", False)
    if not isinstance(absent, bool) or not isinstance(dirty, bool):
        return "absent and dirty_during_window must be booleans"
    arbitration_tier = item.get("arbitration_tier", ArbitrationTier.NO_ARBITER.value)
    if arbitration_tier not in _CHECKPOINT_ARBITRATION_VALUES:
        return (
            f"arbitration_tier must be one of "
            f"{sorted(_CHECKPOINT_ARBITRATION_VALUES)}"
        )
    restore_tier = item.get("restore_tier", RestoreTier.FORWARD_ONLY.value)
    if restore_tier not in _CHECKPOINT_TIER_VALUES:
        return f"restore_tier must be one of {sorted(_CHECKPOINT_TIER_VALUES)}"
    return CheckpointMember(
        member_path=member_path,
        artifact_id=None,
        native_token=native_token,
        fingerprint=fingerprint,
        captured_at=float(captured_at),
        absent=absent,
        dirty_during_window=dirty,
        arbitration_tier=arbitration_tier,
        restore_tier=restore_tier,
    )


def _handle_workspace_checkpoint(
    req: _RequestProtocol, coordinator: CoordinatorHTTPServer
) -> None:
    """POST /workspace/checkpoint — persist one captured checkpoint manifest.

    Request: ``{session_id, name, window_min, window_max, members: [{
    member_path, native_token?, fingerprint?, captured_at, absent?,
    dirty_during_window?, arbitration_tier?, restore_tier?}, ...]}``.

    The OWNER is derived from the caller-asserted ``session_id`` (R9/R13); the
    ``checkpoint_id`` is minted server-side. The registration is ONE registry
    transaction (header + owner + every member — the Unit-2 API), so a typed
    failure means NO partial manifest.

    NOT in ``_MIGRATION_REJECTED_ROUTES`` deliberately: a checkpoint create is
    durable metadata with no version bump and no MESI grant — like the policy
    mutations, it either lands durably (surviving the restart) or fails typed;
    there is no stranded-write hazard for the drain gate to prevent.

    Responses:
      - WIN → ``{ok: true, checkpoint_id, name, window_min, window_max,
        coordinator_epoch}``
      - validation / registry rejection → ``{ok: false, reason}``; a typed
        ``CoherenceError`` carries its identity-stable ``reason`` token with
        the prose in ``detail``
      - watchdog degrade → ``{ok: false, degraded: true, reason:
        "checkpoint_unconfirmed"}`` (fail-closed; see
        :data:`_CHECKPOINT_DEGRADED_RESPONSE`)
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    name = body.get("name")
    window_min = body.get("window_min")
    window_max = body.get("window_max")
    members = body.get("members")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    if not isinstance(name, str) or not name.strip():
        req._json(400, {"error": "name must be a non-empty string"})
        return
    if len(name) > MAX_CHECKPOINT_NAME_LEN:
        req._json(400, {"error": f"name exceeds {MAX_CHECKPOINT_NAME_LEN} chars"})
        return
    for label, value in (("window_min", window_min), ("window_max", window_max)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            req._json(400, {"error": f"{label} must be a number"})
            return
    if window_max < window_min:
        req._json(400, {"error": "window_max must be >= window_min"})
        return
    if not isinstance(members, list) or not members:
        req._json(400, {"error": "members must be a non-empty list of member objects"})
        return
    if len(members) > MAX_CHECKPOINT_MEMBERS:
        req._json(400, {"error": f"members exceeds {MAX_CHECKPOINT_MEMBERS} rows"})
        return
    member_rows: list[CheckpointMember] = []
    for item in members:
        parsed = _parse_checkpoint_member(item)
        if isinstance(parsed, str):
            req._json(400, {"error": parsed})
            return
        member_rows.append(parsed)
    # A duplicate member path would reject registry-side mid-transaction;
    # reject it loud at the boundary instead (mirror the commit_all guard).
    paths = [m.member_path for m in member_rows]
    if len(set(paths)) != len(paths):
        req._json(400, {"error": "members contains duplicate member_path values"})
        return

    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    owner = _session_owner_from_request(coordinator, session_id)
    now = monotonic_seconds()

    def work() -> dict:
        try:
            record = coordinator.service.create_workspace_checkpoint(
                name=name,
                owner=owner,
                members=member_rows,
                window_min=float(window_min),
                window_max=float(window_max),
                issued_at_tick=now,
                abort=abort,
            )
        except ValueError as exc:
            # Service/registry validation (defense-in-depth behind the
            # boundary checks above) — typed reject, nothing persisted.
            return {"ok": False, "reason": str(exc)}
        except CoherenceError as exc:
            # Typed domain failure: identity-stable reason token on the wire
            # (the restore/register posture), prose preserved in "detail".
            return _typed_reason_response(exc)
        return {
            "ok": True,
            "checkpoint_id": record.checkpoint_id,
            "name": record.name,
            "window_min": record.window_min,
            "window_max": record.window_max,
            "coordinator_epoch": coordinator.registry.coordinator_epoch,
        }

    # Fail-closed degrade + abort threading (the A6 session-commit lesson):
    # a timed-out registration must NOT read as success, and the abort Event
    # fails the late write closed at the registry lock.
    abort = threading.Event()
    _run_or_degrade(
        req, coordinator, work, degraded_response=_CHECKPOINT_DEGRADED_RESPONSE, abort=abort
    )


def _render_checkpoint_member(member: CheckpointMember) -> dict:
    return {
        "member_path": member.member_path,
        "native_token": member.native_token,
        "fingerprint": member.fingerprint,
        "captured_at": member.captured_at,
        "absent": member.absent,
        "dirty_during_window": member.dirty_during_window,
        "arbitration_tier": member.arbitration_tier,
        "restore_tier": member.restore_tier,
        "pin_state": member.pin_state,
        "restore_outcome": member.restore_outcome,
        "deleted_at_restore": member.deleted_at_restore,
    }


def _handle_workspace_checkpoints(
    req: _RequestProtocol, coordinator: CoordinatorHTTPServer
) -> None:
    """GET /workspace/checkpoints — list every checkpoint manifest.

    Non-mutating observability (the /status posture): reads the registry's
    checkpoint store directly and renders each header with its member rows —
    including the honesty surfaces a consumer must see per member
    (``restore_tier`` / ``arbitration_tier`` / ``absent`` /
    ``dirty_during_window`` / ``pin_state``). Deterministic order:
    ``(created_at, checkpoint_id)`` for headers, ``member_path`` for members
    (the registry contract).

    Degrades FAIL-CLOSED (``ok: false``) — a timed-out list must not read as
    "no checkpoints exist".
    """

    def work() -> dict:
        checkpoints = []
        for record in coordinator.registry.list_checkpoints():
            members = coordinator.registry.get_checkpoint_members(record.checkpoint_id)
            checkpoints.append(
                {
                    "checkpoint_id": record.checkpoint_id,
                    "name": record.name,
                    "owner": str(record.owner),
                    "created_at": record.created_at,
                    "created_at_tick": record.created_at_tick,
                    "window_min": record.window_min,
                    "window_max": record.window_max,
                    "restore_status": record.restore_status,
                    "restore_updated_at": record.restore_updated_at,
                    "pin_refcount": record.pin_refcount,
                    "members": [_render_checkpoint_member(m) for m in members],
                }
            )
        return {
            "ok": True,
            "checkpoints": checkpoints,
            "coordinator_epoch": coordinator.registry.coordinator_epoch,
        }

    _run_or_degrade(
        req,
        coordinator,
        work,
        degraded_response={
            "ok": False,
            "degraded": True,
            "reason": "checkpoint_list_unconfirmed",
        },
    )


MAX_CHECKPOINT_ID_LEN = 128
"""Cap on a wire ``checkpoint_id`` (server-minted uuid4 strings are 36 chars;
the headroom absorbs future id shapes without admitting content-sized text)."""

_RESTORE_PROGRESS_DEGRADED_RESPONSE: dict = {
    "ok": False,
    "degraded": True,
    "reason": "restore_progress_unconfirmed",
}
"""Fail-closed degrade envelope for the restore progress routes (WV Unit 5):
a watchdog-timed-out status/outcome write must NOT read as recorded — the
abort Event threaded into the service's ``abort_guard`` fails the late write
closed at the registry lock (the A6 lesson)."""

_RESTORE_REGISTER_DEGRADED_RESPONSE: dict = {
    "ok": False,
    "degraded": True,
    "reason": "restore_registration_unconfirmed",
}
"""Fail-closed degrade envelope for ``POST /workspace/restore/register``: a
timed-out registration must read as FAILURE (the ``_OCC_DEGRADED_RESPONSE``
posture — it is a version-bumping commit path), never as success."""


def _validate_checkpoint_id(checkpoint_id: Any) -> str | None:
    """Boundary shape check for a wire checkpoint id (reason, or None if ok)."""
    if not isinstance(checkpoint_id, str) or not checkpoint_id.strip():
        return "checkpoint_id must be a non-empty string"
    if len(checkpoint_id) > MAX_CHECKPOINT_ID_LEN:
        return f"checkpoint_id exceeds {MAX_CHECKPOINT_ID_LEN} chars"
    return None


def _handle_workspace_restore_status(
    req: _RequestProtocol, coordinator: CoordinatorHTTPServer
) -> None:
    """POST /workspace/restore/status — record checkpoint-level restore status.

    Request: ``{session_id, checkpoint_id, status}`` where ``status`` is one of
    the closed :data:`~ccs.core.exceptions.RESTORE_STATUSES` (boundary-gated
    here AND service-validated — fail-closed twice; crash-resume classifies by
    identity, so an unvetted string must never land). The remote half of the
    restore engine's ``CheckpointRestoreStore`` seam (the capture path's
    mirror: durable metadata write, abort-threaded, degrade fail-closed).

    NOT in ``_MIGRATION_REJECTED_ROUTES``: like the checkpoint create, this is
    durable metadata with no version bump — it lands durably or fails typed.
    An unknown checkpoint answers ``{ok: false, reason: "checkpoint_unknown",
    detail}`` — the register route's stable token, matched by identity.
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    checkpoint_id = body.get("checkpoint_id")
    status = body.get("status")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    cid_err = _validate_checkpoint_id(checkpoint_id)
    if cid_err:
        req._json(400, {"error": cid_err})
        return
    if status not in RESTORE_STATUSES:
        req._json(400, {"error": f"status must be one of {sorted(RESTORE_STATUSES)}"})
        return
    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    now = monotonic_seconds()

    def work() -> dict:
        try:
            coordinator.service.set_workspace_checkpoint_restore_status(
                checkpoint_id, status, updated_at=float(now), abort=abort
            )
        except KeyError as exc:
            # Unknown checkpoint — the register route's stable token, not prose.
            return _unknown_checkpoint_response(exc)
        except ValueError as exc:
            return {"ok": False, "reason": str(exc)}
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "restore_status": status,
            "coordinator_epoch": coordinator.registry.coordinator_epoch,
        }

    abort = threading.Event()
    _run_or_degrade(
        req,
        coordinator,
        work,
        degraded_response=_RESTORE_PROGRESS_DEGRADED_RESPONSE,
        abort=abort,
    )


def _handle_workspace_restore_member(
    req: _RequestProtocol, coordinator: CoordinatorHTTPServer
) -> None:
    """POST /workspace/restore/member — record one member's terminal outcome.

    Request: ``{session_id, checkpoint_id, member_path, restore_outcome,
    deleted_at_restore?}``. ``restore_outcome`` is ``null`` (the explicit
    reset) or one of the closed
    :data:`~ccs.core.exceptions.RESTORE_MEMBER_OUTCOMES` (boundary-gated AND
    service-validated); ``deleted_at_restore`` is the manifest-side delete
    record — per the plan's registration split, delete legs register HERE,
    never through ``commit_all`` (which has no delete semantics). An unknown
    (checkpoint, member) pair answers ``{ok: false, reason:
    "checkpoint_unknown", detail}`` — the stable token, prose in ``detail``.
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    checkpoint_id = body.get("checkpoint_id")
    member_path = body.get("member_path")
    restore_outcome = body.get("restore_outcome")
    deleted_at_restore = body.get("deleted_at_restore")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    cid_err = _validate_checkpoint_id(checkpoint_id)
    if cid_err:
        req._json(400, {"error": cid_err})
        return
    path_err = validate_path(member_path)
    if path_err:
        req._json(400, {"error": f"member_path: {path_err}"})
        return
    if restore_outcome is not None and restore_outcome not in RESTORE_MEMBER_OUTCOMES:
        req._json(
            400,
            {
                "error": (
                    f"restore_outcome must be null or one of "
                    f"{sorted(RESTORE_MEMBER_OUTCOMES)}"
                )
            },
        )
        return
    if deleted_at_restore is not None and (
        isinstance(deleted_at_restore, bool)
        or not isinstance(deleted_at_restore, (int, float))
    ):
        req._json(400, {"error": "deleted_at_restore must be null or a number"})
        return
    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return

    def work() -> dict:
        try:
            coordinator.service.set_workspace_checkpoint_member_restore(
                checkpoint_id,
                member_path,
                restore_outcome=restore_outcome,
                deleted_at_restore=(
                    float(deleted_at_restore) if deleted_at_restore is not None else None
                ),
                abort=abort,
            )
        except KeyError as exc:
            # Unknown (checkpoint, member) pair — same stable token as the
            # register route's unknown-checkpoint class; "detail" says which.
            return _unknown_checkpoint_response(exc)
        except ValueError as exc:
            return {"ok": False, "reason": str(exc)}
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "member_path": member_path,
            "restore_outcome": restore_outcome,
            "coordinator_epoch": coordinator.registry.coordinator_epoch,
        }

    abort = threading.Event()
    _run_or_degrade(
        req,
        coordinator,
        work,
        degraded_response=_RESTORE_PROGRESS_DEGRADED_RESPONSE,
        abort=abort,
    )


def _handle_workspace_restore_register(
    req: _RequestProtocol, coordinator: CoordinatorHTTPServer
) -> None:
    """POST /workspace/restore/register — the Unit-5 coordinator registration.

    Request: ``{session_id, checkpoint_id, writes: [{member_path,
    fingerprint}, ...]}`` — the restore run's WRITTEN file members, hash-only
    (fingerprints, never content bytes; the boundary enforces 64-hex). The
    CONTROLLER identity derives from the caller-asserted ``session_id``
    (R9/R13), never a separate client field. One
    ``register_workspace_restore`` call → at most one all-or-nothing
    ``commit_all``; an empty ``writes`` answers the typed ``empty_write_set``
    (``commit_all`` never called, by contract).

    Registered peers holding a member are invalidated ATOMICALLY by the
    registry commit; peers learn through the protocol (their next read is
    strict-denied), so — matching the post-edit-cas house pattern — the
    response surfaces the invalidation COUNT rather than re-broadcasting
    signals.

    In ``_MIGRATION_REJECTED_ROUTES``: this is a version-bumping write
    initiation (the post-edit-cas strand hazard applies mid-drain).

    Responses:
      - WIN → ``{ok: true, status, detail, versions: {path: v}, skipped,
        refused: {path: reason}, invalidated, coordinator_epoch}`` (``status``
        from the closed WORKSPACE_REGISTRATION set; ``refused`` non-empty only
        on ``status == "refused"`` — nothing mutated then, all-or-nothing)
      - unknown checkpoint → ``{ok: false, reason: "checkpoint_unknown"}``
      - watchdog degrade → fail-closed
        :data:`_RESTORE_REGISTER_DEGRADED_RESPONSE`
    """
    body = req._read_json()
    if body is None:
        return
    session_id = body.get("session_id")
    checkpoint_id = body.get("checkpoint_id")
    writes = body.get("writes")
    sid_err = validate_session_id(session_id)
    if sid_err:
        req._json(400, {"error": sid_err[1]})
        return
    cid_err = _validate_checkpoint_id(checkpoint_id)
    if cid_err:
        req._json(400, {"error": cid_err})
        return
    if not isinstance(writes, list):
        req._json(400, {"error": "writes must be a list of {member_path, fingerprint}"})
        return
    if len(writes) > MAX_CHECKPOINT_MEMBERS:
        req._json(400, {"error": f"writes exceeds {MAX_CHECKPOINT_MEMBERS} rows"})
        return
    entries: list[WorkspaceRestoreWrite] = []
    seen_paths: set[str] = set()
    for item in writes:
        if not isinstance(item, dict):
            req._json(400, {"error": "each write must be a JSON object"})
            return
        member_path = item.get("member_path", "")
        path_err = validate_path(member_path)
        if path_err:
            req._json(400, {"error": f"member_path: {path_err}"})
            return
        fingerprint = item.get("fingerprint")
        if not isinstance(fingerprint, str) or not _CONTENT_HASH_RE.match(fingerprint):
            req._json(400, {"error": "fingerprint must be a 64-hex sha-256 digest"})
            return
        if member_path in seen_paths:
            req._json(400, {"error": "writes contains duplicate member_path values"})
            return
        seen_paths.add(member_path)
        entries.append(
            WorkspaceRestoreWrite(member_path=member_path, fingerprint=fingerprint)
        )

    # CONTROLLER derived from the caller-asserted session_id, never a separate field (R9/R13).
    if _admit_caller(req, coordinator, _presented_caller(req, session_id)) is None:
        return
    controller = _session_owner_from_request(coordinator, session_id)
    now = monotonic_seconds()

    def work() -> dict:
        try:
            result = coordinator.service.register_workspace_restore(
                checkpoint_id=checkpoint_id,
                controller=controller,
                writes=entries,
                issued_at_tick=now,
                abort=abort,
            )
        except CheckpointUnknown as exc:
            return {"ok": False, "reason": exc.reason}
        except ValueError as exc:
            return {"ok": False, "reason": str(exc)}
        except OccCallerTransientError:
            # Retry-eligible: the controller is mid-transient (a peer's commit
            # invalidated it between read and CAS). Byte-stable reason.
            return {"ok": False, "reason": OCC_CALLER_TRANSIENT_REASON}
        except CoherenceError as exc:
            return {"ok": False, "reason": str(exc)}
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "status": result.status,
            "detail": result.detail,
            "versions": dict(result.versions),
            "skipped": list(result.skipped),
            "refused": {
                path: conflict.reason for path, conflict in result.refused.items()
            },
            "invalidated": len(result.signals),
            "coordinator_epoch": coordinator.registry.coordinator_epoch,
        }

    abort = threading.Event()
    _run_or_degrade(
        req,
        coordinator,
        work,
        degraded_response=_RESTORE_REGISTER_DEGRADED_RESPONSE,
        abort=abort,
    )


def _handle_status(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """GET /status — drives the agent-coherence-status console script.

    R12 (Unit 6): three-tier disclosure model.

    | tier      | query                | extra requirement              |
    |-----------|----------------------|--------------------------------|
    | minimal   | (none) or detail=minimal | none                       |
    | metrics   | ?detail=metrics      | none — telemetry block only    |
    | full      | ?detail=full         | ``Coherence-Local-Operator: true`` header opt-in |

    ``minimal`` is the default and includes no absolute paths (workspace
    root is reported as a sentinel ``.``) and no session names (R6:
    ``agent_name`` embeds the raw session id, so it is null below the
    operator tier; the non-reversible ``agent_id`` and the per-artifact
    ``states`` map stay). ``metrics`` returns only the
    counter block — useful for operators scraping /status into a
    dashboard without leaking workspace state; it carries no sessions at
    all and is unaffected by R6. ``full`` is the legacy
    everything-block plus absolute ``coordinator_root`` and ``coordinator_pid``,
    gated by the explicit ``Coherence-Local-Operator: true`` header so a
    same-user adversary (Adversary 1 in auth.py) cannot trivially grab
    the operator's home-directory path. Bearer auth is enforced by the
    dispatcher; the header is a SECOND factor specifically for the
    elevated tier.

    AC-07 — metrics-tier stability contract (operator-facing):

      Fields PRESENT in the metrics tier are stable within a major
      version. ``coordinator_uptime_seconds``, ``coordinator_backend``,
      ``coordinator_version``, ``watchdog_timeouts_total``,
      ``watchdog_queue_overflows_total``,
      ``handler_concurrency_overflows_total``,
      ``in_flight_drain_timed_out``, ``cold_start_duration_ms``,
      ``endpoint_counters``, ``intra_task_acquire_release_total``,
      ``stale_warning_emitted_total``, ``stale_warning_reread_total``.

      Fields may be ADDED in minor versions (additive change is
      non-breaking for dashboards using selective key access).

      Fields are REMOVED only in major versions and only after at
      least one minor-version release where the field is emitted
      ALONGSIDE its replacement as a deprecated alias (see AC-02
      for the ``coordinator_uptime_s`` → ``coordinator_uptime_seconds``
      precedent — alias ships through v0.1.x; remove at v0.2.0).

      Fields EXPLICITLY OMITTED from metrics tier vs. full tier:
      ``tracked_artifacts``, ``sessions``, ``policy_summary``,
      ``coordinator_root``, ``coordinator_pid``. Operators wanting
      these for a dashboard must call ``?detail=full`` with the
      ``Coherence-Local-Operator: true`` header.
    """
    detail = _parse_detail_query(getattr(req, "_query_string", ""))
    if detail == "full":
        if req.headers.get("Coherence-Local-Operator", "").lower() != "true":
            req._json(403, {
                "error": (
                    "detail=full requires the Coherence-Local-Operator: true "
                    "opt-in header in addition to the Bearer secret (R12)."
                ),
            })
            return

    # Counter block — present at every tier so the telemetry-only
    # consumer (?detail=metrics) doesn't pay for the artifact/session
    # walk. KTD-J (Unit 8) adds per-endpoint + product-signal counters
    # alongside the existing watchdog/concurrency counters.
    # M-03 / finding #31: use counters_snapshot() instead of reaching into
    # private attrs directly — single source of truth for the counter set.
    # AC-02 cross-backend parity: KTD-J naming convention locks the
    # full-word ``_seconds`` suffix for duration fields. Node emits
    # ``coordinator_uptime_seconds``; Python now matches. The old
    # ``coordinator_uptime_s`` field is emitted ALONGSIDE the new one
    # for one release as a backward-compat alias (consumers can detect
    # which field to read by checking which is present, or just read
    # the canonical name). Deprecation note in docs/metrics.md (TODO).
    _uptime = coordinator.uptime_s
    counters = {
        "coordinator_uptime_seconds": _uptime,
        "coordinator_uptime_s": _uptime,  # AC-02: deprecated alias, removed in v0.2
        "coordinator_backend": "python",
        "coordinator_version": _COORDINATOR_VERSION,
        **coordinator.counters_snapshot(),
    }
    if detail == "metrics":
        req._json(200, {"detail": "metrics", **counters})
        return

    # PERF-1: single batched snapshot — replaces 2N SELECTs (one
    # get_artifact + one get_state_map per artifact) with 2 SELECTs total
    # held under one registry lock so the view is consistent.
    artifact_by_id, state_by_artifact = coordinator.registry.status_snapshot()
    named_agents = coordinator.agent_names_snapshot()

    tracked: list[dict] = []
    for artifact_id, meta in artifact_by_id.items():
        entry: dict[str, Any] = {
            "path": meta["name"],
            "version": meta["version"],
            "id": str(artifact_id),
        }
        if detail == "full":
            # #199 §2: writer attribution is workspace-state disclosure (it
            # names which session wrote what), so it rides the operator-gated
            # tier only — the minimal-tier question is #198's to settle.
            # Read from the same status_snapshot row, not per-artifact
            # last_writer_for calls, which would reopen the PERF-1 N+1.
            # ``last_writer_agent_id`` is the registry's own handle: it joins
            # against ``sessions[].agent_id`` (whose ``agent_name`` this tier
            # already shows) and survives a restart. No session id is derived
            # from it: the only way back from an agent id to a session is the
            # process-local name map, which the ``sessions[]`` join already
            # consults, and which is empty for a writer from before a restart.
            # The timestamp is null when nothing has committed: ``updated_at``
            # then stamps the first observation, not a write.
            writer = meta.get("last_writer_id")
            entry["last_writer_agent_id"] = str(writer) if writer else None
            entry["last_writer_at_unix_ts"] = meta.get("updated_at") if writer else None
        tracked.append(entry)

    # The HOLDER set comes from the registry; a NAME comes from the adapter.
    # They are not the same set. ``_agent_names`` is process-local, seeded
    # empty, and written only by ``register_session`` on hook traffic — so
    # walking it as the outer loop dropped every grant that outlived the
    # coordinator process that issued it, while the durable ``agent_states``
    # row kept arbitrating (a peer's pre-edit still collides against a holder
    # the payload never named). Build from the registry, then label.
    states_by_agent: dict[UUID, dict[str, str]] = {}
    for artifact_id, meta in artifact_by_id.items():
        for agent_id, state in state_by_artifact[artifact_id].items():
            if state == MESIState.INVALID:
                continue
            states_by_agent.setdefault(agent_id, {})[meta["name"]] = state.name

    # R6: ``agent_name`` renders the raw session id verbatim
    # (``session_to_agent_name`` → ``claude-session-<sid>``, and the SB-25
    # subagent form carries it too), so every tier that listed a session row
    # republished that session's identifier alongside its per-artifact state.
    # Below the operator tier the row keeps ``agent_id`` — already a uuid5 of
    # the session id and documented at ``session_to_agent_id`` as not
    # reversible — and drops the name. Nothing new is derived here: a second
    # derivation would duplicate the one the Node backend is parity-pinned to.
    # The null-name row is the shape the unnamed-holder branch below already
    # emits and the status CLI already renders.
    name_visible = detail == "full"
    sessions: list[dict] = []
    named_ids: set[UUID] = set()
    for agent_id, name in named_agents:
        named_ids.add(agent_id)
        sessions.append({
            "agent_name": name if name_visible else None,
            "agent_id": str(agent_id),
            "states": states_by_agent.get(agent_id, {}),
        })
    # ``agent_name`` is null rather than a guess: ``session_to_agent_id`` is a
    # uuid5 of the session id, so the session id is NOT recoverable from the
    # row. An entry keyed on the raw agent id is the honest answer, and it is
    # strictly more than the empty list these holders used to render as.
    for agent_id in sorted(states_by_agent.keys() - named_ids, key=str):
        sessions.append({
            "agent_name": None,
            "agent_id": str(agent_id),
            "states": states_by_agent[agent_id],
        })

    # The pattern lists are workspace-relative paths and globs. Publishing
    # them at the minimal/metrics tiers would expose the operator's directory
    # layout to non-operator callers — ask for them only at the full tier;
    # the counts are in the summary at every tier.
    policy_summary = coordinator.policy.summary(include_patterns=detail == "full")
    base = {
        "detail": detail,
        "tracked_artifacts": tracked,
        "sessions": sessions,
        "policy_summary": policy_summary,
        # P1 #7: coordinator_pid is in the minimal tier too. Process IDs
        # are public on POSIX (anyone with `ps` sees them) so this is
        # not a disclosure beyond the trust boundary the threat model
        # already accepts. Operators use this field to verify "is the
        # coordinator I think is running actually mine" — restoring it
        # closes the regression Unit 6 R12 introduced when it moved pid
        # behind the operator-header gate. The contract is also
        # documented in CLAUDE.md and used by status-rendering CLIs.
        "coordinator_pid": os.getpid(),
        **counters,
    }
    if detail == "full":
        # Full tier still adds the absolute workspace root — that DOES
        # leak $HOME / directory layout and stays gated behind the
        # Coherence-Local-Operator: true header.
        base["coordinator_root"] = str(coordinator.coordinator_root)
    else:
        # Minimal: replace absolute workspace path with sentinel "." so the
        # default tier never leaks $HOME or directory layout.
        base["coordinator_root"] = "."
    req._json(200, base)


def _parse_detail_query(query: str) -> str:
    """R12 (Unit 6): map a raw ``?detail=...`` query string to one of
    ``{minimal, metrics, full}``. Unknown values fall back to ``minimal``
    so a typo never exposes more than the default tier."""
    if not query:
        return "minimal"
    for part in query.split("&"):
        if "=" not in part:
            continue
        key, _, value = part.partition("=")
        if key.strip() == "detail":
            v = value.strip().lower()
            if v in ("minimal", "metrics", "full"):
                return v
            return "minimal"
    return "minimal"


MIGRATION_DRAIN_TIMEOUT_SEC = 5.0
"""ADV-001: how long the migration handler waits for in-flight non-pre-edit
handlers (post-edit, pre-read, etc.) to complete before invalidating
remaining grants + scheduling shutdown. Same magnitude as
``IN_FLIGHT_DRAIN_TIMEOUT_SEC`` since the drain semantics are the same;
keeping them as separate constants documents intent."""


def _handle_prepare_for_migration(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> None:
    """POST /admin/prepare-for-migration — drain → release-all-grants → shutdown.

    Unit 8 (Decision 1, locked 2026-05-18): operator runs
    ``agent-coherence-coordinator --prepare-for-migration`` before
    switching the Python/Node backend.

    ADV-001 (fix): the prior implementation invalidated grants
    synchronously then scheduled shutdown 100ms later. That race let a
    pre-edit landing at T=50ms mint an EXCLUSIVE grant that the
    invalidation step at T=51ms revoked — and the agent's matching
    post-edit at T=150ms hit a dead coordinator (silent failure).

    New sequence:

    1. Flip ``coordinator._migration_draining = True``. Dispatcher
       starts rejecting NEW pre-edit requests with HTTP 503 + a
       structured "migration in progress" error visible to the model.
       Other endpoints (post-edit, pre-read, session-stop, /status,
       policy mutations) continue to serve so in-flight chains can
       finish naturally.
    2. Background thread waits up to ``MIGRATION_DRAIN_TIMEOUT_SEC``
       for the in-flight handler counter to reach zero. In-flight
       pre-edit→post-edit pairs complete normally during this window.
    3. After drain, snapshot every (agent, artifact) pair still in
       MODIFIED or EXCLUSIVE state — these are orphaned grants from
       sessions that pre-edited but never post-edited (already broken).
       Invalidate each so the new backend doesn't inherit them.
    4. Schedule ``coordinator.shutdown()`` ~100ms later (kernel send
       buffer flush window for the response).

    Returns immediately with ``{ok:true, draining:true,
    drain_timeout_ms}``. The CLI polls /status until the coordinator
    becomes TCP-unreachable; counts/errors land in the coordinator log
    rather than the HTTP response (the response goes out before the
    drain completes).

    Requires the same elevated-tier signal as /status?detail=full:
    Bearer + ``Coherence-Local-Operator: true`` header.

    Security note (SEC-01): the ``Coherence-Local-Operator: true``
    header value is a static, well-known string embedded in public
    source — it does NOT constitute a second factor against Adversary
    1 (same OS user who can read the 0600 hook.secret file). This
    endpoint is a DoS surface within the Adversary 1 boundary: a
    same-UID process with hook.secret access can force coordinator
    shutdown. Accepted per the v0.1 threat model. The header serves as
    an explicit opt-in signal for operator-automation tooling, not as
    a security gate.
    """
    if req.headers.get("Coherence-Local-Operator", "").lower() != "true":
        req._json(403, {
            "error": (
                "prepare-for-migration requires the Coherence-Local-Operator: "
                "true opt-in header (operator-only endpoint)."
            ),
        })
        return

    # Idempotent: a second call while already draining returns the same
    # accepted-but-already-running envelope.
    if coordinator.migration_draining:
        req._json(200, {
            "ok": True,
            "draining": True,
            "already_in_progress": True,
        })
        return

    coordinator._migration_draining = True
    SHUTDOWN_DELAY_MS = 100

    def _drain_invalidate_and_shutdown() -> None:
        """Background sequence: drain in-flight handlers (during which
        pre-edit→post-edit pairs complete naturally), invalidate any
        remaining M/E grants (orphans from sessions that pre-edited
        but never post-edited), schedule shutdown.
        """
        # Step 1: wait for in-flight handlers to drain. The current
        # handler holds one slot itself, so account for that by
        # comparing against 1 rather than 0. After this handler
        # returns, the counter drops to its true in-flight value
        # which the second pass observes.
        deadline = time.monotonic() + MIGRATION_DRAIN_TIMEOUT_SEC
        while time.monotonic() < deadline:
            with coordinator._in_flight_lock:
                # Strictly less than 2 means: just this handler still
                # in flight (1), or fewer (handler already returned).
                if coordinator._in_flight <= 1:
                    break
            time.sleep(0.020)

        # Step 2: invalidate any remaining M/E grants (orphans).
        now = monotonic_seconds()
        released = 0
        errors: list[dict[str, str]] = []
        for artifact_id in list(coordinator.registry.artifact_ids()):
            state_map = coordinator.registry.get_state_map(artifact_id)
            for agent_id, state in list(state_map.items()):
                if state not in (MESIState.MODIFIED, MESIState.EXCLUSIVE):
                    continue
                artifact = coordinator.registry.get_artifact(artifact_id)
                if artifact is None:
                    continue
                try:
                    coordinator.service.invalidate(
                        agent_id=agent_id,
                        artifact_id=artifact_id,
                        new_version=artifact.version,
                        issuer_agent_id=agent_id,
                        issued_at_tick=now,
                    )
                    released += 1
                except CoherenceError as exc:
                    errors.append({
                        "artifact_id": str(artifact_id),
                        "agent_id": str(agent_id),
                        "reason": str(exc),
                    })
        logger.info(
            "prepare-for-migration drained: released=%d errors=%d",
            released, len(errors),
        )
        if errors:
            for e in errors:
                logger.warning("prepare-for-migration invalidate error: %s", e)

        # Step 3: schedule shutdown ~100ms later so any further status
        # polls from the CLI see the draining state once before TCP
        # becomes unreachable.
        time.sleep(SHUTDOWN_DELAY_MS / 1000.0)
        try:
            coordinator.shutdown()
        except Exception:  # pragma: no cover — best-effort cleanup
            logger.exception("scheduled shutdown after prepare-for-migration failed")

    threading.Thread(
        target=_drain_invalidate_and_shutdown,
        name="prepare-for-migration-drain",
        daemon=True,
    ).start()

    req._json(200, {
        "ok": True,
        "draining": True,
        "drain_timeout_ms": int(MIGRATION_DRAIN_TIMEOUT_SEC * 1000),
        "shutdown_scheduled_in_ms": int(MIGRATION_DRAIN_TIMEOUT_SEC * 1000) + SHUTDOWN_DELAY_MS,
    })


_ROUTES: dict[tuple[str, str], Callable] = {
    ("POST", "/hooks/pre-read"): _handle_pre_read,
    # The effect fence: "may this irreversible effect still fire, and if not,
    # WHY". A route of its own rather than a flag on pre-read, whose response
    # already varies by freshness. Registered HERE so it rides the one
    # dispatcher seam (Host -> Bearer -> migration gate -> endpoint counter ->
    # handler) like every other route; read-only toward the registry, so it
    # stays out of _MIGRATION_REJECTED_ROUTES.
    ("POST", "/hooks/effect-fence"): _handle_effect_fence,
    ("POST", "/hooks/pre-edit"): _handle_pre_edit,
    ("POST", "/hooks/post-edit"): _handle_post_edit,
    # Unit 6: OCC commit. Version-checked CAS that bypasses the pre-edit
    # EXCLUSIVE acquire (the OCC writer stays S/I).
    ("POST", "/hooks/post-edit-cas"): _handle_post_edit_cas,
    ("POST", "/hooks/session-stop"): _handle_session_stop,
    # SB-10 U2 — post-compaction re-grounding. Registered HERE so it rides
    # the central verify_bearer + verify_host seam (KTD7: no operator
    # header, no parallel router). Read-only toward the registry; stays out
    # of _MIGRATION_REJECTED_ROUTES because it initiates no writes.
    ("POST", "/hooks/session-start"): _handle_session_start,
    # v0.1.1 KTD-N — H4 mitigation: catch model routing-around-via-Bash/Grep
    # to bypass the Read-only stale-read warning. Per the v0.2 Phase 0
    # falsifiability experiment, the model retries Read 2-5 times then
    # routes via `bash cat plan.md`; unhooked Bash means silent stale miss.
    ("POST", "/hooks/pre-bash"): _handle_pre_bash,
    ("POST", "/hooks/pre-grep"): _handle_pre_grep,
    ("POST", "/policy/track"): _handle_policy_track,
    ("POST", "/policy/untrack"): _handle_policy_untrack,
    # SB-17 / TX-1 Unit 8 — snapshot-session endpoints. Registered HERE so they
    # ride the central verify_bearer + verify_host seam the dispatcher applies
    # to every route (NO parallel router; R9 / Unit 8 endpoint-auth obligation).
    ("POST", "/session/begin"): _handle_session_begin,
    ("POST", "/session/read"): _handle_session_read,
    ("POST", "/session/commit"): _handle_session_commit,
    ("POST", "/session/commit_all"): _handle_session_commit_all,
    ("POST", "/session/heartbeat"): _handle_session_heartbeat,
    # Caller-principal plan U4 — the mint. Registered HERE so it rides the one
    # dispatcher seam (Host -> Bearer -> migration gate -> counter -> handler).
    # Out of _MIGRATION_REJECTED_ROUTES: a binding is durable metadata with no
    # version bump and no grant (the checkpoint-create rationale), and a
    # claimant that loses its response to a restart recovers it by nonce.
    # Python-only: the sibling Node coordinator answers 404 here; U6 declares
    # that asymmetry with a corpus fixture rather than leaving it silent.
    ("POST", "/principal/claim"): _handle_principal_claim,
    # WV plan Unit 3 — workspace-checkpoint endpoints. Registered HERE so they
    # ride the central verify_bearer + verify_host seam like every route. The
    # POST threads a watchdog abort Event into the service's abort_guard (the
    # A6 session-commit lesson: every new mutating route threads abort).
    ("POST", "/workspace/checkpoint"): _handle_workspace_checkpoint,
    ("GET", "/workspace/checkpoints"): _handle_workspace_checkpoints,
    # WV plan Unit 5 — restore progress + registration endpoints (the remote
    # half of the restore engine's CheckpointRestoreStore seam). Every
    # mutating handler threads a watchdog abort Event (the A6 lesson).
    ("POST", "/workspace/restore/status"): _handle_workspace_restore_status,
    ("POST", "/workspace/restore/member"): _handle_workspace_restore_member,
    ("POST", "/workspace/restore/register"): _handle_workspace_restore_register,
    ("GET", "/status"): _handle_status,
    ("POST", "/admin/prepare-for-migration"): _handle_prepare_for_migration,
}


# ADV-001: routes the dispatcher rejects with 503 while
# ``coordinator.migration_draining`` is True. Only NEW write-initiation
# requests belong here — post-edit must still serve (so in-flight chains
# can complete), policy mutations must still serve (operator may be
# clearing tracked artifacts as part of migration prep), and all reads
# + observability always serve. pre-bash / pre-grep don't initiate
# writes, so they stay out of this set.
_MIGRATION_REJECTED_ROUTES: set[tuple[str, str]] = {
    ("POST", "/hooks/pre-edit"),
    # Unit 6: the OCC commit is a self-contained NEW write initiation — it
    # has no prior pre-edit chain to complete (it bypasses the acquire), so
    # it belongs here with pre-edit. Letting it through mid-migration would
    # bump a version the imminent shutdown then strands; the client retries
    # after the coordinator restarts.
    ("POST", "/hooks/post-edit-cas"),
    # A snapshot session.commit is a version-bumping OCC write at the pinned base
    # — same hazard as post-edit-cas: letting it land mid-migration bumps a
    # version the imminent shutdown then strands. Reject it draining; the client
    # opens a fresh session + re-reads + re-commits after the restart. (begin /
    # read / heartbeat are non-mutating and stay out of this set.)
    ("POST", "/session/commit"),
    # session.commit_all is the batch OCC write — same strand hazard as
    # /session/commit, multiplied across the write-set. Reject it draining.
    ("POST", "/session/commit_all"),
    # WV Unit 5: the restore registration is a version-bumping write
    # initiation (its commit_all rides the same strand hazard). The restore
    # PROGRESS routes (/workspace/restore/status, /workspace/restore/member)
    # stay out: durable metadata with no version bump — the checkpoint-create
    # rationale.
    ("POST", "/workspace/restore/register"),
}


# KTD-J (Unit 8): route → counter-name lookup. Used by ``_dispatch`` to
# bump per-endpoint counters BEFORE invoking the handler (contract:
# counts attempted requests, not successful ones, so a timeout or
# exception still shows up in operator-visible /status output).
_ENDPOINT_COUNTER_NAMES: dict[tuple[str, str], str] = {
    ("POST", "/hooks/pre-read"): "pre_read_total",
    ("POST", "/hooks/effect-fence"): "effect_fence_total",
    ("POST", "/hooks/pre-edit"): "pre_edit_total",
    ("POST", "/hooks/post-edit"): "post_edit_total",
    ("POST", "/hooks/post-edit-cas"): "post_edit_cas_total",
    ("POST", "/hooks/session-stop"): "session_stop_total",
    ("POST", "/hooks/session-start"): "session_start_total",
    ("POST", "/hooks/pre-bash"): "pre_bash_total",
    ("POST", "/hooks/pre-grep"): "pre_grep_total",
    ("POST", "/policy/track"): "policy_track_total",
    ("POST", "/policy/untrack"): "policy_untrack_total",
    ("GET", "/status"): "status_total",
    ("POST", "/principal/claim"): "principal_claim_total",
    # /admin/prepare-for-migration intentionally not counted — it
    # initiates shutdown, so counting it would never be observable via
    # subsequent /status calls (coordinator is already down).
}


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


_DEFAULT_DEGRADED_RESPONSE: dict = {
    "status": "fresh",
    "degraded": True,
    # A7: a degraded read must NOT silently masquerade as a verified-fresh read.
    # The handler timed out, so the staleness check never ran — surface an
    # advisory the model actually sees (the hook client passes hookSpecificOutput
    # straight through) instead of an empty allow. ``status`` stays "fresh" for
    # wire back-compat (clients branch on it); the advisory + ``degraded`` flag
    # are the honest signal that freshness is UNVERIFIED, not confirmed.
    "hookSpecificOutput": _payloads.emit_allow(
        source="watchdog_degraded_read",
        additional_context=(
            "⚠ Coherence could not verify this file's freshness — the coordinator "
            "staleness check timed out under load. Proceeding WITHOUT a stale-read "
            "guarantee: if this file is shared with other agents or sessions, "
            "re-read it before relying on its contents."
        ),
    ),
}
"""AC-05 / A7: pre-read / pre-bash / pre-grep degrade to a fresh-shape envelope
because their wire contract uses ``{status: "fresh"|"stale"}``. A7: the envelope
now carries a ``hookSpecificOutput`` advisory so a watchdog-degraded read is
visible to the model rather than silently passing as a confirmed fresh read.
Endpoints whose contract is ``{ok: bool}`` pass an ok-shape envelope instead
(:data:`_OK_DEGRADED_RESPONSE`; pre-edit its own
:data:`_PRE_EDIT_DEGRADED_RESPONSE`) so the client doesn't see ``None`` from
``result.get("ok")``."""

_OK_DEGRADED_RESPONSE: dict = {"ok": True, "degraded": True}
"""AC-05: degraded envelope for {ok: bool}-shape endpoints (post-edit,
session-stop, and the snapshot-session begin/read/heartbeat routes). Pairs with
``_DEFAULT_DEGRADED_RESPONSE``. Pre-edit has its own
:data:`_PRE_EDIT_DEGRADED_RESPONSE`, equal in value today."""

_PRE_EDIT_DEGRADED_RESPONSE: dict = {"ok": True, "degraded": True}
"""What ``/hooks/pre-edit`` answers when the watchdog times out: an ADMIT.

The rule this constant carries: a timed-out acquire must not answer in a shape
that admits an edit the route would refuse when it is not degraded. The
acquire-or-fail refusal asked for in Cohexa-ai/agent-coherence#196 (an opt-in
answering ``{"ok": false, "reason": "other_holder"}`` instead of displacing a
live holder) is not built; :data:`_ACQUIRE_OR_FAIL_REFUSAL_REASONS` is where it
will register its reasons, and a guard test fails while that set is non-empty
and this envelope is still ``ok: true``. A timed-out request never looked for a
holder: admitting it lets the caller edit over a holder the refusal would have
reported, and answering ``other_holder`` would report a holder nobody observed.

The rule is already broken by a refusal that exists today. Undegraded, a
strict-mode editor whose grant a peer invalidated gets the strict deny; on a
timeout it gets this envelope, which carries no ``hookSpecificOutput``, so the
hook client relays no deny and no advisory -- unlike the read path's
:data:`_DEFAULT_DEGRADED_RESPONSE`, which at least says freshness was not
checked. That degrade-to-admit is shipped behaviour (a hook must not block a
tool call on coordinator load) and this constant does not change it. Library
callers are stricter: ``CoherentVolume`` with ``on_error="strict"`` (its
default) raises on ``degraded: true``.

Its own name, rather than a share of :data:`_OK_DEGRADED_RESPONSE`, so that
changing what a timed-out acquire answers cannot silently change what a
timed-out post-edit or session-stop answers."""

_ACQUIRE_OR_FAIL_REFUSAL_REASONS: frozenset[str] = frozenset()
"""Reasons an opted-in acquire-or-fail ``/hooks/pre-edit`` may refuse with.

Empty: that refusal is not built (see :data:`_PRE_EDIT_DEGRADED_RESPONSE`). The
change that builds it adds its reasons here -- ``other_holder`` first -- and
that same change must give a timed-out opted-in request a non-admitting
answer, or the guard test on this pair goes red. The strict-mode deny is not
listed: it is not opt-in and its reason travels in ``hookSpecificOutput``."""

_SESSION_START_DEGRADED_RESPONSE: dict = {}
"""SB-10 U2 (KTD7): degrade envelope for ``/hooks/session-start`` — the
empty payload. Re-grounding is advisory (KD3): a watchdog timeout must look
exactly like "nothing to say", never block, and never claim state it could
not read. The compact-pending flag is armed INSIDE the timed work only
after a successful non-empty build (with the A6 abort threaded into the
builder's ``abort_guard``), so a degraded response also means the deferred
path stays unarmed."""

_OCC_DEGRADED_RESPONSE: dict = {
    "ok": False,
    "degraded": True,
    "reason": "commit_unconfirmed",
}
"""Plan Unit 6 (the load-bearing fix): degrade envelope for the OCC commit
endpoint (``/hooks/post-edit-cas``). Unlike the pessimistic endpoints — which
degrade to ``_OK_DEGRADED_RESPONSE`` (``{ok: true, …}``) so a generic client
reading ``result.get("ok")`` proceeds — a degraded/timed-out ``commit_cas``
MUST read as **failure**: returning ``ok: true`` would let the client assume
its write landed when the CAS never confirmed (and may still be running in the
watchdog pool — the late-completion residual). ``ok: false`` forces the
fail-closed client to raise. The residual is a phantom version bump, NOT a
lost update — full fencing is deferred to the cross-host follow-on (see
``_handle_post_edit_cas``)."""

_CALLER_GATE_DEGRADED_RESPONSE: dict[tuple[str, str], dict] = {
    ("POST", "/hooks/pre-read"): _DEFAULT_DEGRADED_RESPONSE,
    ("POST", "/hooks/effect-fence"): _EFFECT_FENCE_DEGRADED_RESPONSE,
    ("POST", "/hooks/pre-edit"): _PRE_EDIT_DEGRADED_RESPONSE,
    ("POST", "/hooks/post-edit"): _OK_DEGRADED_RESPONSE,
    ("POST", "/hooks/post-edit-cas"): _OCC_DEGRADED_RESPONSE,
    ("POST", "/hooks/session-stop"): _OK_DEGRADED_RESPONSE,
    ("POST", "/hooks/session-start"): _SESSION_START_DEGRADED_RESPONSE,
    ("POST", "/hooks/pre-bash"): _DEFAULT_DEGRADED_RESPONSE,
    ("POST", "/hooks/pre-grep"): _DEFAULT_DEGRADED_RESPONSE,
    ("POST", "/session/begin"): _OK_DEGRADED_RESPONSE,
    ("POST", "/session/read"): _OK_DEGRADED_RESPONSE,
    ("POST", "/session/commit"): _OCC_DEGRADED_RESPONSE,
    ("POST", "/session/commit_all"): _OCC_DEGRADED_RESPONSE,
    ("POST", "/session/heartbeat"): _OK_DEGRADED_RESPONSE,
    ("POST", "/workspace/checkpoint"): _CHECKPOINT_DEGRADED_RESPONSE,
    ("POST", "/workspace/restore/status"): _RESTORE_PROGRESS_DEGRADED_RESPONSE,
    ("POST", "/workspace/restore/member"): _RESTORE_PROGRESS_DEGRADED_RESPONSE,
    ("POST", "/workspace/restore/register"): _RESTORE_REGISTER_DEGRADED_RESPONSE,
}
"""What a route answers when its caller-principal gate cannot decide within the
handler watchdog: exactly what its handler passes ``_run_or_degrade`` for a
timed-out work body. The request was neither admitted nor refused, so it gets
the route's ordinary degraded answer — never a principal refusal (nothing was
refused) and never an admission (nothing was checked). Keyed like
:data:`_CALLER_PRINCIPAL_POSTURE`, over every route whose gate can read the
registry, i.e. all but the mint; no default, so a missing route fails loudly. A
test compares each entry with the route's own timed-out work body, so the two
cannot drift."""


def _watchdog_deadline(req: _RequestProtocol) -> float | None:
    """The request's watchdog deadline, if its caller-principal gate started
    one (``None`` — the full ``HANDLER_TIMEOUT_SEC`` — otherwise)."""
    return getattr(req, "_watchdog_deadline", None)


def _watchdog_queue_overflowed(req: _RequestProtocol, coordinator: CoordinatorHTTPServer) -> bool:
    """KTD-G item 1: the queue-depth gate in front of the watchdog pool. When
    the pool's work queue is past :data:`WATCHDOG_QUEUE_LIMIT`, count it, write
    the 503 and return True — the caller submits nothing. Shared by
    :func:`_run_or_degrade` and the caller-principal gate's registry lookup, so
    neither path queues work the other would have refused."""
    # A defensive try because ThreadPoolExecutor's _work_queue attribute is
    # technically private — guard against future stdlib changes.
    try:
        qsize = coordinator._watchdog._work_queue.qsize()  # type: ignore[attr-defined]
    except AttributeError:
        qsize = 0
    if qsize <= WATCHDOG_QUEUE_LIMIT:
        return False
    coordinator.increment_watchdog_queue_overflow()  # finding #31
    logger.warning(
        "watchdog queue at %d items (limit %d); rejecting with 503",
        qsize,
        WATCHDOG_QUEUE_LIMIT,
    )
    req._json(503, {"error": "watchdog queue overloaded"})
    return True


def _run_or_degrade(
    req: _RequestProtocol,
    coordinator: CoordinatorHTTPServer,
    work: Callable[[], dict],
    *,
    degraded_response: dict | None = None,
    abort: threading.Event | None = None,
) -> None:
    """Run ``work`` under the handler-side watchdog. On timeout, log WARNING
    and return 200 with ``degraded_response`` (or the default fresh-shape
    envelope) so the user's tool call proceeds.

    AC-05: callers from ``{ok: bool}``-shape endpoints (post-edit,
    session-stop) pass ``degraded_response=_OK_DEGRADED_RESPONSE`` -- and
    pre-edit its own :data:`_PRE_EDIT_DEGRADED_RESPONSE` -- so clients
    reading ``result.get("ok")`` see ``True`` rather than ``None``. Callers
    from ``{status: ...}``-shape endpoints (pre-read, pre-bash, pre-grep)
    accept the default fresh-shape envelope.

    v0.1.1 KTD-G:
      - Item 1: queue-depth gate. Reject with HTTP 503 if the watchdog
        ThreadPoolExecutor's _work_queue is past WATCHDOG_QUEUE_LIMIT
        items. Prevents the silent-degradation cascade documented in
        plugin docs/known-issues/2026-05-17-watchdog-races.md A7 where
        queued tasks wait long enough in the executor queue that they
        race the future's timeout on submit-side.
      - Item 3: increment ``_watchdog_timeouts_total`` on FuturesTimeout
        so silent degradation becomes observable via /status.

    Item 2 (handler concurrency semaphore) lives in
    _ThreadingHTTPServer.process_request — gates BEFORE this function
    is reached.

    The watchdog deadline is the REQUEST's: when the caller-principal gate
    already started it with a registry lookup, ``work`` gets only what is left
    (:func:`_watchdog_deadline`).
    """
    if _watchdog_queue_overflowed(req, coordinator):
        return

    try:
        result = coordinator.run_with_watchdog(
            work, abort=abort, deadline=_watchdog_deadline(req)
        )
    except FuturesTimeout:
        # KTD-G item 3: surface watchdog degradation via /status counter.
        coordinator.increment_watchdog_timeout()  # finding #31
        logger.warning("handler watchdog timeout after %ss; degrading", HANDLER_TIMEOUT_SEC)
        req._json(200, degraded_response if degraded_response is not None else _DEFAULT_DEGRADED_RESPONSE)
        return
    except Exception as exc:
        _answer_handler_error(req, coordinator, exc, failed="handler work")
        return
    req._json(200, result)


_EFFECT_FENCE_ROUTE: tuple[str, str] = ("POST", "/hooks/effect-fence")
"""The one route whose work body owns its own exception arm (a hold, see
:data:`_EFFECT_FENCE_INTERNAL_HOLD_RESPONSE`), keyed as ``_ROUTES`` keys it."""


def _answer_handler_error(
    req: _RequestProtocol,
    coordinator: CoordinatorHTTPServer,
    exc: BaseException,
    *,
    failed: str,
) -> None:
    """The ONE exception arm for a request's work, whether the caller-principal
    gate's registry lookup raised or the handler's work body did: log the
    failure with its traceback and answer HTTP 200 with the route's
    internal-error envelope — ``{"ok": false, "reason": "internal: <Type>"}``,
    a shape the library types as a refused request (``StaleView`` on pre-edit)
    and no client reads as an absent coordinator.

    Why one arm and not two: every client reads a non-200 as "no coordinator
    available" and degrades, so a raise the gate let reach the dispatcher's
    500 would have a degrade-mode ``CoherentVolume`` write with no grant and
    no peer invalidation, while the same raise inside the work body answered
    the typed 200 it raises ``StaleView`` on. The effect fence is the one
    route whose body catches inside itself and answers a HOLD rather than
    this shape (a client branching on ``verdict`` must read an answer), so on
    that route the gate answers, and counts, the same hold the body would.
    Nothing else is counted: a raise is not a watchdog timeout, and a gate
    that decided nothing is neither an admission nor a refusal.

    ``failed`` names what raised, for the log. Only the exception's own text
    reaches the log and only its TYPE reaches the body; the lookup's operands
    (a presented principal) are in neither."""
    logger.exception("%s failed: %s", failed, exc)
    if req._route_key == _EFFECT_FENCE_ROUTE:
        coordinator.increment_effect_fence_hold()
        req._json(200, dict(_EFFECT_FENCE_INTERNAL_HOLD_RESPONSE))
        return
    req._json(200, {"ok": False, "reason": f"{HANDLER_FAILURE_REASON_PREFIX}{type(exc).__name__}"})


def _exclusive_holder(
    coordinator: CoordinatorHTTPServer,
    artifact_id: UUID,
    *,
    exclude_agent: UUID,
) -> tuple[UUID | None, int | None]:
    """Return (agent_id, granted_at_tick) of any current M∪E holder OTHER
    than the given agent. Used for KTD-9 collision detection in pre-edit."""
    state_map = coordinator.registry.get_state_map(artifact_id)
    for other_id, state in state_map.items():
        if other_id == exclude_agent:
            continue
        if state in (MESIState.EXCLUSIVE, MESIState.MODIFIED):
            granted_at = coordinator.registry.granted_at_tick(other_id, artifact_id)
            return other_id, granted_at
    return None, None


def _peers_in_me_excluding(
    coordinator: CoordinatorHTTPServer,
    artifact_id: UUID,
    agent_id: UUID,
) -> list[UUID]:
    """A1: return the list of agents currently in MODIFIED or EXCLUSIVE on
    the given artifact, EXCLUDING the given agent. Used to snapshot victims
    BEFORE service.write side-effects invalidate them, so the plugin can
    record preemption notices for each."""
    state_map = coordinator.registry.get_state_map(artifact_id)
    return [
        peer_id
        for peer_id, state in state_map.items()
        if peer_id != agent_id and state in (MESIState.EXCLUSIVE, MESIState.MODIFIED)
    ]


def _iso_utc(unix_ts: float) -> str:
    """Format a unix timestamp as an ISO 8601 UTC string for prose."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


#: F3 hardening — render up to this many notices verbatim; coalesce the
#: rest into a single "Plus K more …" line that points at the status surface.
#: Chosen so the rendered prose stays comfortably under Claude Code's 10KB
#: additionalContext cap even with the prepended stale-read warning (each
#: verbatim line ≈ 250 bytes; budget of 3 × 250 + header/footer keeps total
#: well under 4KB before any stale-read prepend).
_PREEMPTION_PROSE_VERBATIM_CAP = 3

#: SB-10 U2 (R5) — same cap pattern for the post-compaction re-grounding
#: payload: at most this many artifact lines render verbatim; the rest
#: coalesce into one overflow line pointing at the status surface. Keeps the
#: payload constant-size regardless of how many artifacts a session touched,
#: comfortably under Claude Code's 10KB additionalContext ceiling even when
#: the preemption-notice block (its own cap above) is prepended.
_SESSION_START_ARTIFACT_VERBATIM_CAP = 3


def _build_preemption_text(
    coordinator: CoordinatorHTTPServer,
    notices: list[tuple[UUID, UUID, float]],
    *,
    overflow_deferred: bool = True,
) -> str:
    """A1 + F3: render pending preemption notices as additionalContext prose.

    notices: list of (artifact_id, preempter_agent_id, preempted_at_unix_ts).
    Variance per invocation comes from the timestamps (real preemption time)
    + the session-id prefixes.

    F3 hardening: render newest-first up to ``_PREEMPTION_PROSE_VERBATIM_CAP``
    notices in full. If more remain, coalesce them into a single overflow line.
    This bounds the prose to a constant-size block regardless of N and sidesteps
    Claude Code's 10KB additionalContext cap.

    On the four admit paths the overflow is a DEFERRAL, not a truncation:
    they pass that same cap as ``pop_pending_notices(consume_limit=...)``, so
    only the rendered rows are consumed and the rest surface on the session's
    next tracked-file operation (or age out via ``evict_stale_notices``). The
    overflow line used to name ``/agent-coherence status`` / ``GET /status`` as
    the place to read the rest — neither has ever carried notice data, and the
    rows had already been deleted by the call that under-reported them.

    ``overflow_deferred=False`` is for ``session-stop``, whose drain is
    UNBOUNDED and whose response returns every notice in its ``notices`` array.
    Nothing is queued there — the rows are gone and a stopping session has no
    next operation — so repeating the deferral promise would be false twice
    over. It names the response's own array instead, which is where they are.
    """
    # Sort newest first — the most recent preemption is the most informative
    # signal for the agent's next decision. The registry already returns rows
    # newest-first (ts DESC, artifact_id DESC) and this sort is STABLE, so the
    # verbatim slice below picks exactly the rows the drain consumed; the rest
    # are still queued. Changing either ordering without the other re-opens the
    # drop (rows deleted that this never renders).
    sorted_notices = sorted(notices, key=lambda n: n[2], reverse=True)
    verbatim = sorted_notices[:_PREEMPTION_PROSE_VERBATIM_CAP]
    overflow = sorted_notices[_PREEMPTION_PROSE_VERBATIM_CAP:]

    lines: list[str] = ["⚠ Coordinator notice: your EXCLUSIVE grant was preempted:"]
    for artifact_id, preempter_id, ts in verbatim:
        artifact = coordinator.registry.get_artifact(artifact_id)
        path = artifact.name if artifact else "<unknown-artifact>"
        # ADV-004: render sweep-reclaimed notices with a clear cause rather
        # than the awkward truncated sentinel UUID.
        if preempter_id == SWEEP_RECLAMATION_PREEMPTER_ID:
            lines.append(
                f"  • {path} — reclaimed by the coordinator sweep "
                f"(heartbeat timeout or max-hold ceiling) at {_iso_utc(ts)}. "
                f"Any local edit you made to this file will land in your "
                f"worktree but is NOT reflected in the coordinator's version. "
                f"Re-fetch via pre-read and retry."
            )
            continue
        # R7: the notice row already carries the preempter's agent id, which
        # is what the registry stores and what /status publishes. Rendering
        # it directly also removes the restart degradation the reverse
        # lookup had — the agent-name map is process-local and starts empty,
        # so a preempter that survived a restart used to render "<unknown>".
        lines.append(
            f"  • {path} — preempted/revoked by agent "
            f"{_payloads.short_session_id(preempter_id.hex)} "
            f"at {_iso_utc(ts)}. Any local edit you made to this file will land "
            f"in your worktree but is NOT reflected in the coordinator's version."
        )
    if overflow:
        # Say where the unrendered notices actually ARE, which differs by
        # caller. The bounded admit paths leave them in the table, so they
        # surface on the next tracked-file operation. session-stop consumed
        # them all and hands them back structurally in the same response —
        # claiming they are "still queued" there would be false twice over
        # (deleted rows, and no next operation for a stopping session). The
        # line used to point at /status, which carries no notice data at any
        # disclosure tier, while the rows it named had just been deleted.
        if overflow_deferred:
            where = "still queued — they surface on your next tracked-file operation."
        else:
            where = "listed in full in the `notices` array of this response."
        lines.append(
            f"  • Plus {len(overflow)} more preemptions since your last "
            f"activity, {where}"
        )
    lines.append(
        "Re-read affected files before continuing if you need the latest "
        "coordinator-tracked version, or proceed knowing your edits remain "
        "local-only until you re-acquire and commit."
    )
    return "\n".join(lines)


def _build_session_start_context(
    coordinator: CoordinatorHTTPServer,
    session_id: str,
    *,
    abort: threading.Event | None = None,
) -> tuple[str | None, bool]:
    """SB-10 U2: render the post-compaction re-grounding prose for a session.

    Returns ``(additional_context, workspace_has_state)`` where the context is
    ``None`` when the session holds no coordination state (R5's no-op arm) and
    ``workspace_has_state`` feeds the R8 breadcrumb decision in the handler.

    KTD2: the payload is rebuilt from registry truth on EVERY delivery, from
    ONE consistent read pass — the whole registry walk happens under a single
    ``abort_guard`` hold (a plain re-entrant lock acquire for the inner read
    calls), so a peer commit cannot tear the view between the snapshot and
    the per-row last-observed / last-writer reads. The pass is read-only:
    R6 forbids this endpoint from recording observations, and the pending
    preemption notices are PEEKED, never popped — consumption ownership
    stays with the admit-endpoint drains.

    Rendering (KTD8, byte-mirrored by the Node coordinator):
    - header first; the existing preemption-notice prose (if any) next;
      then artifact lines — the parent agent's first, then each registered
      subagent's under a ``Subagent {name}:`` prefix, groups sorted by name,
      artifacts sorted by path within a group;
    - a held E/M/S row renders the event-anchored grant line with the
      CURRENT version; any other row (INVALID) renders the stale line only
      when the version advanced past a RECORDED last-observed value and the
      last writer is not this very agent (R7: never-observed admits, own
      edits are exempt — KTD4's second layer — and no 0-sentinel compares);
    - at most ``_SESSION_START_ARTIFACT_VERBATIM_CAP`` artifact lines render
      verbatim, the rest coalesce into the overflow line (R5);
    - the self-qualifying closing line is always last. No timestamps.
    """
    # Agent enumeration takes only the agent-names lock; done BEFORE the
    # registry lock so the two locks never nest in names→registry order.
    agents = coordinator.agents_for_session(session_id)

    with coordinator.registry.abort_guard(abort):
        # Scoped to THIS session's agents (SB-10 review): the walk below can
        # only render rows belonging to `agents`, this runs on a hook path
        # twice per compaction, and nothing garbage-collects `agent_states` —
        # so an unscoped read would pull the workspace's whole history through
        # the registry lock only to discard it. The artifact half stays
        # workspace-wide: R8's breadcrumb asks about the WORKSPACE.
        artifact_by_id, state_by_artifact = coordinator.registry.status_snapshot(
            agent_ids=[agent_id for agent_id, _ in agents]
        )
        workspace_has_state = bool(artifact_by_id)
        # KTD8: artifacts sorted by path (ASCII-lexicographic — identical to
        # the Node backend's default string sort for the path charset).
        sorted_artifacts = sorted(
            artifact_by_id.items(), key=lambda item: item[1]["name"]
        )
        notices: list[tuple[UUID, UUID, float]] = []
        groups: list[tuple[str | None, list[str]]] = []
        # last_writer depends only on the artifact; cache it across agents so
        # the loop stays O(pairs-with-state) single SELECTs, not O(A x P).
        last_writer_cache: dict[UUID, UUID | None] = {}
        for agent_id, subagent_name in agents:
            lines: list[str] = []
            for artifact_id, meta in sorted_artifacts:
                state = state_by_artifact.get(artifact_id, {}).get(agent_id)
                if state is None:
                    # A pending notice implies the victim held a grant, so its
                    # state row exists (rows are never deleted; preemption
                    # transitions them to INVALID) -- stateless pairs cannot
                    # carry a notice and are skipped before any per-pair SQL.
                    continue
                pending = coordinator.registry.peek_preemption_notice(
                    agent_id, artifact_id
                )
                if pending is not None:
                    notices.append((artifact_id, pending[0], pending[1]))
                path = meta["name"]
                current = meta["version"]
                if state is not MESIState.INVALID:
                    lines.append(
                        _payloads.SESSION_START_GRANT_LINE_TEMPLATE.format(
                            state=state.name, path=path, version=current
                        )
                    )
                    continue
                last = coordinator.registry.last_observed_version_for(
                    artifact_id, agent_id
                )
                stale = last is not None and current > last
                if stale:
                    if artifact_id not in last_writer_cache:
                        last_writer_cache[artifact_id] = (
                            coordinator.registry.last_writer_for(artifact_id)
                        )
                    stale = last_writer_cache[artifact_id] != agent_id
                if stale:
                    lines.append(
                        _payloads.SESSION_START_STALE_LINE_TEMPLATE.format(
                            path=path, current=current, last=last
                        )
                    )
                else:
                    lines.append(
                        _payloads.SESSION_START_TOUCHED_LINE_TEMPLATE.format(
                            path=path, current=current
                        )
                    )
            if lines:
                groups.append((subagent_name, lines))
        if not groups and not notices:
            return None, workspace_has_state
        # Rendered inside the guard: _build_preemption_text re-reads artifact
        # names from the registry, and those reads belong to the same
        # consistent pass as the snapshot the notices came from.
        notice_text = (
            _build_preemption_text(coordinator, notices) if notices else None
        )

    rendered: list[str] = [_payloads.SESSION_START_HEADER]
    if notice_text is not None:
        rendered.append(notice_text)
    total_lines = sum(len(lines) for _, lines in groups)
    budget = _SESSION_START_ARTIFACT_VERBATIM_CAP
    for subagent_name, lines in groups:
        if budget <= 0:
            break
        take = lines[:budget]
        if subagent_name is not None:
            rendered.append(
                _payloads.SESSION_START_SUBAGENT_PREFIX_TEMPLATE.format(
                    name=subagent_name
                )
            )
        rendered.extend(take)
        budget -= len(take)
    overflow = total_lines - _SESSION_START_ARTIFACT_VERBATIM_CAP
    if overflow > 0:
        rendered.append(
            _payloads.SESSION_START_OVERFLOW_LINE_TEMPLATE.format(count=overflow)
        )
    rendered.append(_payloads.SESSION_START_CLOSING_LINE)
    return "\n".join(rendered), workspace_has_state


def _reground_qualifies(result: dict) -> bool:
    """SB-10 U4 (R8): does this admit response qualify to carry — and
    therefore consume — the deferred re-grounding payload?

    The payload attaches ONLY to allow envelopes. A response already
    carrying a ``hookSpecificOutput`` qualifies iff its decision is
    ``allow`` — a strict-mode deny must stay byte-identical (KTD-P) and
    must NOT consume: the flag survives for the next qualifying admit
    (AE3). An envelope-less response qualifies iff it is an admit body —
    ``{ok: true}`` (pre-edit) or ``{status: "fresh"}`` (pre-read /
    pre-bash / pre-grep). A service-refusal body (``{ok: false, ...}``)
    is neither a deny envelope nor an admit: no attach, no consume."""
    hso = result.get("hookSpecificOutput")
    if hso is not None:
        return hso.get("permissionDecision") == "allow"
    return result.get("ok") is True or result.get("status") == "fresh"


def _claim_reground_context(
    coordinator: CoordinatorHTTPServer,
    session_id: str,
    body: dict,
    *,
    abort: threading.Event | None = None,
) -> str | None:
    """SB-10 U4: atomically claim the session's pending re-grounding
    delivery and rebuild its prose.

    Returns the payload text when THIS call wins the claim; ``None`` when
    the request carries a subagent identity (R8: a parent request carries
    NO agent_id field on the wire — a request presenting one must neither
    consume nor attach), when the caller's watchdog ``abort`` is already
    set (a doomed request must not consume the flag — the abort check runs
    BEFORE the pop so the delivery survives for the next live admit), when
    no flag is pending, when a concurrent admit won the pop (R2's
    at-most-once hangs on ``consume_compact_pending``'s locked
    test-and-clear), or when the rebuild came back empty (state drained
    since the compact event — nothing left worth re-grounding).

    KTD2 rebuild-at-delivery: the prose is rebuilt from registry truth via
    ``_build_session_start_context`` at the moment of attach, so the
    deferred payload says what a fresh /hooks/session-start would say NOW
    — never a snapshot cached at compact time. The session-scoped build
    covers the parent agent plus every registered subagent, and carries
    the R5 verbatim cap, so the delivery budget matches the direct path.

    Failure containment: re-grounding is advisory (KD3). A rebuild error
    after a won claim must not turn an otherwise-successful admit into an
    internal-error body, so the build is guarded and the delivery is
    forfeited (R2 permits at-most-once → zero) rather than propagated."""
    if has_subagent_id_field(body):
        return None
    if abort is not None and abort.is_set():
        return None
    if not coordinator.consume_compact_pending(session_id):
        return None
    try:
        text, _ = _build_session_start_context(coordinator, session_id, abort=abort)
        return text
    except Exception:  # advisory delivery must never break the admit it rides
        logger.exception(
            "deferred re-grounding rebuild failed for session %s; "
            "delivery forfeited",
            session_id,
        )
        return None


def _attach_reground(result: dict, text: str) -> dict:
    """SB-10 U4: merge the claimed re-grounding prose into a qualifying
    admit response. An existing envelope keeps its text AND its existing
    permission decision, and gets the block appended AFTER it (notices and
    stale warnings render first — KTD6 ordering); a bare admit body gains
    a CONTEXT-ONLY PreToolUse envelope. The deferred path rides the
    PreToolUse shape — never the SessionStart ``hookSpecificOutput``
    shape."""
    hso = result.get("hookSpecificOutput")
    if hso is None:
        # WHY context-only rather than emit_allow: re-grounding is
        # advisory (KD3), and an advisory payload must NEVER widen a
        # permission decision. A bare body here is the untracked
        # fast-path admit — a tool call Claude Code would ordinarily
        # prompt the user about. Stamping permissionDecision "allow" on
        # it just to carry prose auto-approved that call once per
        # compaction, purely as a side effect of delivery.
        # Empirical basis: a PreToolUse hookSpecificOutput carrying only
        # hookEventName + additionalContext IS rendered to the model —
        # A/B capture against the installed Claude Code CLI 2.1.233 on
        # 2026-08-25 showed the marker-primed model quoting the injected
        # line verbatim with and without permissionDecision. The allow
        # bought nothing the context-only envelope does not already give.
        return {
            **result,
            "hookSpecificOutput": _payloads.emit_pretooluse_context(
                additional_context=text,
            ),
        }
    existing = hso.get("additionalContext")
    merged = dict(hso)
    merged["additionalContext"] = (
        text if existing is None else existing + "\n\n" + text
    )
    return {**result, "hookSpecificOutput": merged}


def _fast_path_json(
    req: _RequestProtocol,
    coordinator: CoordinatorHTTPServer,
    session_id: str,
    body: dict,
    base: dict,
) -> None:
    """SB-10 U4 (KTD6): shared exit for the four hoisted fast paths. The
    advisory peek keeps no-flag traffic registry-free with the exact
    pre-SB-10 response bytes; a pending flag routes through the deferred
    re-ground attach, which claims it atomically at the allow seam.

    The delivery rebuild walks the registry, so it runs under the same
    handler-side watchdog as the tracked seams (a slow/contended rebuild
    must not hang the hook past its budget). On timeout the response is
    the bare ``base`` — byte-identical to the no-flag fast path — and the
    abort token (A6 pattern, set by ``run_with_watchdog`` on timeout)
    stops the zombie delivery from consuming the flag or landing a
    payload nobody will read; a zombie that already won the pop forfeits
    the delivery (R2 permits at-most-once → zero)."""
    if coordinator.has_compact_pending(session_id):
        abort = threading.Event()

        def deliver() -> dict:
            return _deliver_pending_reground(
                coordinator, session_id, body, base, abort=abort
            )

        try:
            base = coordinator.run_with_watchdog(
                deliver, abort=abort, deadline=_watchdog_deadline(req)
            )
        except FuturesTimeout:
            # KTD-G item 3 mirror of _run_or_degrade: surface the
            # degradation via the /status counter, then fall through to
            # the bare fast-path body.
            coordinator.increment_watchdog_timeout()
            logger.warning(
                "deferred re-grounding delivery timed out after %ss; "
                "degrading to the bare fast-path response",
                HANDLER_TIMEOUT_SEC,
            )
    req._json(200, base)


def _deliver_pending_reground(
    coordinator: CoordinatorHTTPServer,
    session_id: str,
    body: dict,
    result: dict,
    *,
    abort: threading.Event | None = None,
) -> dict:
    """SB-10 U4 (KTD6): the allow-attach seam shared by the four admit
    surfaces (pre-read, pre-edit, pre-bash, pre-grep). Runs AFTER any deny
    decision: a non-qualifying result is returned untouched — the same
    object, so deny bodies stay byte-identical — and only then does the
    atomic pop race; exactly one concurrent qualifying admit attaches.

    ``abort`` is the caller's per-request watchdog token (A6 pattern):
    threaded into the claim so a timed-out request neither consumes the
    flag (checked before the pop) nor holds the registry walk hostage
    (the rebuild aborts at the registry lock)."""
    if not _reground_qualifies(result):
        return result
    text = _claim_reground_context(coordinator, session_id, body, abort=abort)
    if text is None:
        return result
    return _attach_reground(result, text)


def _last_writer_for(coordinator: CoordinatorHTTPServer, artifact_id: UUID) -> str | None:
    """Return the AGENT id (hex) of the artifact's last writer, if any.

    COR-09 (fixed): authoritative source is ``artifacts.last_writer_id``
    in the registry — set by the commit path on successful post-edit and
    NOT touched by reads or invalidations. Falls back to the state-map
    only when the registry doesn't have a recorded writer (e.g.,
    artifact created via first-observation seeding with no subsequent
    successful commit). The previous state-map fallback could attribute
    the write to the very session receiving the warning (agent appears
    in state_map as SHARED, becomes the "first known agent").

    R7: this used to run ``session_to_agent_id`` backwards through the
    adapter's agent-name map and return the recovered session id, so the
    stale/deny response handed to one session named a peer by that peer's
    session id. It returns the registry's own handle instead. Nothing new is
    derived — the agent id is a uuid5 of the session id that ``/status``
    already emits, and a second derivation here would duplicate the one the
    Node backend is parity-pinned to.

    The hex form (32 lowercase chars, no hyphens) rather than ``str(uuid)``:
    it is byte-identical to what Node's ``sessionToAgentId`` returns, and the
    value lands in a corpus field compared verbatim across both backends.
    """
    # COR-09 primary path: authoritative last_writer_id from the registry.
    committed_writer = coordinator.registry.last_writer_for(artifact_id)
    if committed_writer is not None:
        return committed_writer.hex
    # Fallback for artifacts that exist but never had a successful commit
    # (first-observation seeding with no post-edit yet). Best signal: an
    # agent currently in MODIFIED state (mid-commit, in case the write
    # is still in-flight).
    state_map = coordinator.registry.get_state_map(artifact_id)
    for agent_id, state in state_map.items():
        if state == MESIState.MODIFIED:
            return agent_id.hex
    # No committed writer + no MODIFIED holder → genuinely unknown.
    # Return None rather than the misleading "first state-map entry"
    # which could be the querying agent itself.
    return None


def _prior_version_observed(
    coordinator: CoordinatorHTTPServer,
    artifact_id: UUID,
    agent_id: UUID,
    *,
    agent_state: MESIState | None,
    current_version: int,
) -> int | None:
    """The version this agent last ACTUALLY observed, for the stale summary.

    ``None`` for an agent that has never held a grant on the artifact — there
    is no observation to report, and a 0-sentinel would read as "saw v0".

    R8. Every stale summary used to compute this as ``current_version - 1``,
    which is only correct if the invalidation came from a commit. When a peer
    merely TAKES the grant (Cohexa-ai/agent-coherence#196) the version does
    not move, so the inference reported a version the session never saw and
    made an unchanged version look changed — which is also what hid the false
    "was updated by" claim, because ``current > prior`` held unconditionally.

    ``agent_states.last_observed_version`` is the durable per-agent record
    (SB-10 R6/R7), written GIL-atomically with every non-INVALID grant and
    deliberately PRESERVED across the transition to INVALID: the last version
    whose bytes the agent actually held. The Node backend keeps the same
    column and reads it the same way.

    The old inference survives as the fallback for a NULL value, which two
    rows produce: a row that predates the column (a v5 database migrated
    forward leaves it NULL), and a row whose only grant certified no read
    (``observed=False``: the agent's first grant on the path came from a
    DENIED Bash/Grep command, see ``_apply_bash_grep_regrants``). That
    degrades to the old behaviour rather than to a worse one: ``current - 1``
    is always below current, so the summary takes the write wording and asks
    for a re-read.
    It cannot silently become the normal path, because both cases need an
    INVALID agent that never recorded an observation.
    """
    if agent_state != MESIState.INVALID:
        return None
    observed = coordinator.registry.last_observed_version_for(artifact_id, agent_id)
    if observed is not None:
        return observed
    return current_version - 1 if current_version > 0 else 0


def _last_writer_unix_ts(
    coordinator: CoordinatorHTTPServer, artifact_id: UUID
) -> float | None:
    """Return the REAL wall-clock time the artifact was last written, from
    `artifacts.updated_at` in the registry. None if the artifact is unknown.

    P2 ce-review fix #16: uses the public ``get_artifact_updated_at()``
    accessor on SqliteArtifactRegistry instead of reaching into ``_conn``
    + ``_lock`` directly. Layer violation closed."""
    return coordinator.registry.get_artifact_updated_at(artifact_id)


def _emit_pre_read_strict_deny(
    coordinator: CoordinatorHTTPServer,
    *,
    agent_id: UUID,
    session_id: str,
    artifact_id: UUID,
    path: str,
    summary: _payloads.StaleSummary,
    source: str,
) -> dict:
    """Single source of truth for emitting a pre-read strict-mode deny, used by
    both deny arms: the INVALID/None-grant path and the Survivor #6
    SHARED-holder foreign-edit path. Bumps the stale-warning + strict-denial
    telemetry, marks the (agent, artifact) pair for KTD-J re-read detection,
    records the (session, path) for pre-bash/pre-grep route-around, appends the
    audit row (counting mode drift on failure), and returns the byte-stable
    deny payload. Keeping both arms here means their telemetry and KTD-T
    byte-stable deny text cannot drift apart."""
    coordinator.increment_stale_warning_emitted()
    coordinator.mark_stale_warned(agent_id, artifact_id)
    coordinator.increment_strict_mode_denial()
    coordinator.record_strict_deny(session_id, path)
    if not _audit_log.append_strict_deny(
        coordinator.coordinator_root, agent_id=session_id, path=path, tool="Read",
    ):
        coordinator.increment_audit_log_mode_drift()
    return {
        "hookSpecificOutput": _payloads.emit_strict_deny(
            source=source, summary=summary,
        ),
        "status": "stale",
        "summary": summary,
    }


def _is_recent_self_commit_lag(
    coordinator: CoordinatorHTTPServer,
    artifact_id: UUID,
    caller_agent_id: UUID,
    *,
    now_unix: float,
) -> bool:
    """Survivor #6 v1 (R2): is a SHARED-holder hash mismatch the benign
    commit→disk-write lag rather than a foreign edit?

    A still-SHARED reader proves no peer commit landed since its grant (a peer
    commit invalidates every non-INVALID peer), so the canonical hash is the
    reader's own granted-version hash and a disk-hash mismatch is the CALLER's
    own disk diverging. That is the legitimate lag iff the CALLER is the
    artifact's last committer AND that commit is recent — the registry advanced
    the canonical hash (e.g. a commit_cas WIN that leaves the writer SHARED) but
    the agent has not yet flushed the new bytes to disk. Any other last_writer,
    no last_writer, or a stale self-commit (something rewrote disk since) is a
    genuine out-of-band edit and must be denied.
    """
    # Compare the RAW committed-writer agent id against the CALLER's composite
    # agent id — matching the Node backend (`last_writer_id === agentId`). An
    # earlier version compared ATTRIBUTION strings via `_agent_id_to_session`,
    # but a subagent's attribution is its BARE agent_id, which COLLIDES across
    # two different parent sessions that present the same agent_id string — that
    # would suppress a genuine foreign-edit deny (adversarial review 2026-07-17)
    # and diverge from Node. The raw composite id is unique per (session,
    # subagent) and cannot collide. `_last_writer_for` is still used for the
    # DISPLAY attribution (the [:8] prose naming the writer), not this gate.
    if coordinator.registry.last_writer_for(artifact_id) != caller_agent_id:
        return False
    updated_at = _last_writer_unix_ts(coordinator, artifact_id)
    if updated_at is None:
        return False
    return (now_unix - updated_at) <= _SHARED_FOREIGN_DENY_LAG_WINDOW_SEC


# ----------------------------------------------------------------------
# The tracked-artifact pre-read decision, as a value
# ----------------------------------------------------------------------


@dataclass(frozen=True, kw_only=True)
class TrackedReadDecision:
    """The fresh / stale / denied verdict for an ALREADY-TRACKED artifact.

    ``_handle_pre_read`` answers this question today by interleaving the
    decision with the mutations it triggers — the fresh arm bumps two counters
    and can return a strict deny; the stale arm re-grants SHARED, marks the
    session stale-warned and drains notices — so a surface that must answer
    the SAME question WITHOUT granting anything cannot reach the answer by
    calling the handler. This type is that answer with the mutations left
    behind, so two surfaces branch off one classification rather than off two
    copies of a safety rule.

    Every field keys a branch a caller still has to take:

    ``outcome``
        which response to build: the fresh payload, the stale envelope, or
        the byte-stable strict deny.
    ``version``
        the fresh payload's ``version`` and the stale summary's
        ``current_version`` — both from the SAME pair-atomic snapshot the
        caller read, never a second read a peer commit could overtake.
    ``owner_generation``
        the effect-gate opt-in pair member. ``None`` means the snapshot
        carried none and the key is omitted rather than coerced to ``0``.
    ``fresh_hash_differs``
        the FRESH arm's mismatch: bumps ``fresh_shared_hash_mismatch_total``
        and sets ``hash_differs`` on the fresh payload.
    ``stale_hash_differs``
        the STALE arm's mismatch: the summary's ``hash_differs``, and half
        the no-prior-grant deny gate.
    ``commit_lag_suppressed``
        the fresh arm's deny was WITHHELD as the benign commit→disk-write lag
        (Survivor #6 R2); bumps ``shared_foreign_lag_suppressed_total``.
    ``holds_valid_grant``
        which arm produced the verdict, and therefore which ``source`` a deny
        carries (``pre_read_shared_hash_deny`` vs ``pre_read_strict_deny``) —
        ``denied`` alone cannot tell the two deny arms apart.
    ``prior_version_seen``
        the summary's ``prior_version_seen_by_session``, already resolved for
        the arm that ran: the current version for a still-granted holder, the
        registry's recorded ``last_observed_version`` for an INVALIDated one
        (R8 — see :func:`_prior_version_observed`; it used to be inferred as
        one below current, which is wrong whenever a peer took the grant
        without committing), ``None`` for a session with no prior grant.

    THE TWO HASH PREDICATES ARE NOT ONE PREDICATE, and folding them into one
    field silently moves the strict-deny gate in whichever direction the
    survivor points. ``fresh_hash_differs`` excludes the no-claim recorded
    hashes (``None``, ``""``, the all-``f`` launch-gate sentinel) because a
    still-SHARED holder must never be denied against content the coordinator
    never claimed. ``stale_hash_differs`` keeps them IN, because the
    launch-gate scenarios reach their deny THROUGH the stale arm and the
    sentinel is precisely what makes their hashes differ.
    """

    outcome: Literal["fresh", "stale", "denied"]
    version: int
    owner_generation: int | None
    fresh_hash_differs: bool
    stale_hash_differs: bool
    commit_lag_suppressed: bool
    holds_valid_grant: bool
    prior_version_seen: int | None


def decide_tracked_read(
    coordinator: CoordinatorHTTPServer,
    *,
    path: str,
    artifact_id: UUID,
    agent_id: UUID,
    artifact: Artifact,
    owner_generation: int | None,
    agent_state: MESIState | None,
    caller_content_hash: str | None,
    now_unix: float,
) -> TrackedReadDecision:
    """Classify a pre-read of an already-tracked artifact. PURE QUERY.

    Takes the values the caller has ALREADY read — the pair-atomic
    ``(artifact, owner_generation)`` snapshot, this agent's MESI state, the
    hash the caller offered — and reads nothing further except
    ``policy.is_strict_mode``, the INVALIDated arm's ``last_observed_version``
    lookup, and, on the one arm that consults it, the lag gate's two registry
    lookups. All three are READS. It grants nothing, re-grants nothing,
    advances no observation baseline, bumps no counter, records no heartbeat,
    marks no stale pair, records no deny and pops no notice. That is the whole
    point: a read on the safety path must not mutate or heal what it checks,
    so a verdict surface can ANSWER without also committing to the effects
    ``_handle_pre_read`` performs around the same decision — a verdict that
    healed a grant would turn a level-triggered HOLD into an edge-triggered
    one.

    ``now_unix`` is the CALLER's single clock read (KTD-P): the lag gate and
    every summary timestamp stamped from this verdict must share one instant,
    so the clock is an argument rather than a call inside.

    The classification mirrors ``_handle_pre_read``'s tracked branch exactly.
    The untracked and first-observation branches are deliberately NOT covered:
    each reaches its answer BY mutating (registering the artifact, seeding v1,
    granting SHARED), so neither has a side-effect-free form to extract.

    Args:
        artifact: the artifact from the caller's snapshot. Non-optional —
            a vanished row is the caller's branch to take, as it is today.
    """
    # A grant that is present and not INVALID is the fresh arm's admission
    # ticket: the holder was granted on the current version, and a peer commit
    # would have invalidated it.
    holds_valid_grant = agent_state is not None and agent_state != MESIState.INVALID

    # The FRESH arm's mismatch predicate. ``coordinator_holds_content_claim``
    # is the core fence's own no-claim normalisation — the same three spellings
    # of "nothing was recorded" (``None``, ``""``, the all-``f`` sentinel) that
    # the in-process fence refuses to treat as a content claim — so this
    # surface asks the shared rule instead of restating it against a local copy
    # of the sentinel literal.
    fresh_hash_differs = bool(
        caller_content_hash
        and coordinator_holds_content_claim(artifact.content_hash)
        and caller_content_hash != artifact.content_hash
    )
    # The STALE arm's mismatch predicate: the same comparison MINUS the
    # no-claim exclusion. Kept separate on purpose (see TrackedReadDecision).
    stale_hash_differs = bool(
        caller_content_hash
        and artifact.content_hash
        and caller_content_hash != artifact.content_hash
    )

    if holds_valid_grant:
        # Survivor #6 v1: a still-SHARED holder whose disk hash mismatches the
        # canonical content is this session's own disk diverging — either its
        # own un-flushed recent commit (benign lag, suppress) or a foreign
        # out-of-band edit (deny). Warn mode never denies here.
        commit_lag_suppressed = False
        outcome: Literal["fresh", "stale", "denied"] = "fresh"
        if fresh_hash_differs and coordinator.policy.is_strict_mode(path):
            if _is_recent_self_commit_lag(
                coordinator, artifact_id, agent_id, now_unix=now_unix,
            ):
                commit_lag_suppressed = True
            else:
                outcome = "denied"
        return TrackedReadDecision(
            outcome=outcome,
            version=artifact.version,
            owner_generation=owner_generation,
            fresh_hash_differs=fresh_hash_differs,
            stale_hash_differs=stale_hash_differs,
            commit_lag_suppressed=commit_lag_suppressed,
            holds_valid_grant=True,
            # A SHARED holder was granted on the current version; that is the
            # version it last saw.
            prior_version_seen=artifact.version,
        )

    # No valid grant: either a peer commit INVALIDated this session, or it has
    # never seen this artifact.
    prior_version_seen = _prior_version_observed(
        coordinator, artifact_id, agent_id,
        agent_state=agent_state, current_version=artifact.version,
    )

    # v0.2 KTD-O / KTD-P deny gate, unchanged: strict mode AND the session
    # demonstrably lacks a fresh view — a true preemption (INVALID), or no
    # prior grant at all while the bytes it just hashed differ from the
    # canonical. Matching hashes mean the session is observing what the
    # registry recorded, so there is nothing stale to act on: warn-mode allow.
    denied = coordinator.policy.is_strict_mode(path) and (
        agent_state == MESIState.INVALID
        or (agent_state is None and stale_hash_differs)
    )
    return TrackedReadDecision(
        outcome="denied" if denied else "stale",
        version=artifact.version,
        owner_generation=owner_generation,
        fresh_hash_differs=fresh_hash_differs,
        stale_hash_differs=stale_hash_differs,
        # The lag gate belongs to the fresh arm only: it answers "is this
        # holder's OWN un-flushed commit", and a caller with no standing grant
        # has no such commit to be waiting on.
        commit_lag_suppressed=False,
        holds_valid_grant=False,
        prior_version_seen=prior_version_seen,
    )


# R7: ``_agent_id_to_session`` used to live here — it reversed
# ``session_to_agent_id`` through the agent-name map so the five renderers
# below could print a session id. Every one of them now prints the agent id
# the registry already holds, and the helper is DELETED rather than left
# unused: kept around, the next renderer that wants a friendlier label reaches
# for it and re-introduces the mapping. SB-25 attribution is unaffected — a
# subagent's composite agent id is distinct per (session, subagent), so the
# short form still names the actual writer, and the parent linkage stays
# visible via ``agent_name`` on /status's operator tier.


def _parse_yaml_pattern_lines(text: str) -> set[str]:
    """Extract pattern strings already present in a YAML list file.

    Used to detect already-present entries before appending, keeping
    /policy/track idempotent. Uses yaml.safe_load so quoted values
    (``- "plan.md"``) and inline comments are handled correctly.
    Returns empty set on empty input or malformed YAML — the caller
    re-appends, which is safe (MAX_POLICY_YAML_BYTES caps growth)."""
    if not text.strip():
        return set()
    import yaml as _yaml
    try:
        parsed = _yaml.safe_load(text)
    except _yaml.YAMLError:
        return set()
    if not isinstance(parsed, list):
        return set()
    return {item for item in parsed if isinstance(item, str)}


def _append_policy_yaml(yaml_path: Path, new_paths: list[str]) -> tuple[list[str], list[dict]]:
    """Append valid patterns to a YAML file. Returns (added, rejected).
    Honors MAX_POLICY_YAML_BYTES — raises ValueError if the resulting file
    would exceed the cap.

    R14 (Unit 6): the read-modify-write is wrapped in an ``fcntl.flock``
    exclusive lock on a sidecar ``<yaml_path>.lock`` file so two concurrent
    /policy/track or /policy/untrack requests cannot interleave their
    reads-and-writes and corrupt the YAML (e.g., both read the same
    pre-state, both compute "previous + my_paths", second write loses the
    first writer's additions). fcntl is POSIX-only; on the deferred
    Windows path this is a no-op (lifecycle already disables the
    coordinator on Windows per _FCNTL_AVAILABLE).

    COR-06: callers pre-validate via ``validate_path`` (or equivalent)
    and pass only safe paths. The local re-check below is a defensive
    second pass that catches accidentally-bypassed validation BUT in
    the normal flow the ``rejected`` list it returns from this branch
    is always empty (everything passes the caller's check already).
    Kept as defense-in-depth — removing it would couple this helper to
    the caller's validation discipline, which is a tighter contract
    than the function's current "self-contained validate-and-write"
    behaviour. Tests should assert callers reject before reaching here.
    """
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    # Validate each path the same way TrackedArtifactPolicy does. This
    # validation is pure (no I/O) so it lives outside the lock window.
    # Defense-in-depth per COR-06: callers pre-validate, but this loop
    # ensures the YAML write is never reached with traversal patterns.
    added: list[str] = []
    rejected: list[dict] = []
    for p in new_paths:
        if not p:
            rejected.append({"path": p, "reason": "empty"})
            continue
        if p.startswith("/"):
            rejected.append({"path": p, "reason": "absolute path"})
            continue
        if ".." in Path(p).parts:
            rejected.append({"path": p, "reason": "contains '..'"})
            continue
        added.append(p)

    if not added:
        return added, rejected

    lock_path = yaml_path.with_suffix(yaml_path.suffix + ".lock")
    try:
        import fcntl as _fcntl
        lock_fd: int | None = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        try:
            _fcntl.flock(lock_fd, _fcntl.LOCK_EX)
            try:
                existing = yaml_path.read_text()
            except FileNotFoundError:
                existing = ""
            already_present = _parse_yaml_pattern_lines(existing)
            truly_new = [p for p in added if p not in already_present]
            if not truly_new:
                return [], rejected
            new_lines = "\n".join(f"- {p}" for p in truly_new)
            new_content = (
                (existing.rstrip("\n") + "\n" + new_lines + "\n") if existing
                else (new_lines + "\n")
            )
            if len(new_content.encode("utf-8")) > MAX_POLICY_YAML_BYTES:
                raise ValueError(
                    f"policy YAML cap of {MAX_POLICY_YAML_BYTES} bytes would be exceeded"
                )
            yaml_path.write_text(new_content)
            return truly_new, rejected
        finally:
            try:
                _fcntl.flock(lock_fd, _fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(lock_fd)
            except OSError:
                pass
    except ImportError:
        # Windows fallback: no fcntl. Lifecycle already disables the
        # coordinator on Windows, but degrade defensively if reached.
        try:
            existing = yaml_path.read_text()
        except FileNotFoundError:
            existing = ""
        already_present = _parse_yaml_pattern_lines(existing)
        truly_new = [p for p in added if p not in already_present]
        if not truly_new:
            return [], rejected
        new_lines = "\n".join(f"- {p}" for p in truly_new)
        new_content = (
            (existing.rstrip("\n") + "\n" + new_lines + "\n") if existing
            else (new_lines + "\n")
        )
        if len(new_content.encode("utf-8")) > MAX_POLICY_YAML_BYTES:
            raise ValueError(
                f"policy YAML cap of {MAX_POLICY_YAML_BYTES} bytes would be exceeded"
            )
        yaml_path.write_text(new_content)
        return truly_new, rejected
