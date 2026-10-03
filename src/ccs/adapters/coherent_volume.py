# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""CoherentVolume — the data-plane coherent-workspace appliance (v1).

v1 prevents the **sequential stale-read→write lost update** for a single-host
agent fleet sharing files in a workspace — the OpenViktor cron shape (agent A
reads v1, agent B reads v1, A commits v2, B's stale write is denied → B
re-reads). It does **not** serialize concurrent racing writers, attach to a
coordinator it did not spawn, or detect an agent that re-reads fresh bytes then
writes a buffer computed from older bytes — those are explicit v1.1 / honest-
boundary cases (see ``docs/plans/2026-06-03-001-feat-data-plane-coherent-workspace-plan.md``
and ``docs/solutions/best-practices/coordinator-invalidation-not-mutex-honest-coherence-claims-2026-06-04.md``).

**Architecture.** CoherentVolume is a thin *out-of-process coordinator client*,
not a wrapper over the in-process :class:`~ccs.adapters.base.CoherenceAdapterCore`.
The teeth that make invalidation enforceable — the strict-mode ``INVALID``-deny —
live in the coordinator HTTP server (``ccs.adapters.claude_code``), so v1 reuses
that shipped path: it writes the policy YAML, spawns/attaches the local-HTTP
coordinator over SQLite-WAL, and routes reads/writes through the ``/hooks/*``
endpoints. Content stays on the real filesystem; the coordinator holds MESI
state + content-hash + version only.

The façade scaffolding (Unit 1) is spawn-with-strict enablement, per-instance
identity (fork-safe), and the fail-closed degrade contract. The sequential
read/write/reacquire contract (Unit 2) builds on it: :meth:`CoherentVolume.read`
registers a SHARED view, :meth:`CoherentVolume.write` acquires EXCLUSIVE (or
fails closed on a stale-view deny), and :meth:`CoherentVolume.reacquire`
recovers from the sticky strict deny. The ``install()`` shim (Unit 3) is next.
"""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import logging
import os
import secrets
import threading
import time
import urllib.error
import uuid
import warnings
import weakref
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NamedTuple

import yaml

from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    connect_or_spawn,
    read_port_from_file,
    tcp_probe,
)
from ccs.adapters.claude_code.policy import matches_any
from ccs.cli._coherence_client import (
    NODE_BACKEND,
    PRINCIPAL_REFUSED_AGAIN,
    CoordinatorEndpoint,
    CoordinatorUnavailable,
    PrincipalClaim,
    PrincipalRecovery,
    RemoteCoordinatorConfig,
    caller_principal_headers,
    claim_caller_principal,
    coordinator_backend,
    decide_principal_recovery,
    principal_refusal_message,
    principal_refusal_reason,
    resolve_endpoint,
)
from ccs.cli._coherence_client import (
    get as _coordinator_get,
)
from ccs.cli._coherence_client import (
    post as _coordinator_post,
)
from ccs.core.exceptions import (
    CALLER_PRINCIPAL_CLAIMED_REASON,
    COMMIT_UNCONFIRMED_REASON,
    OCC_CALLER_TRANSIENT_REASON,
    STALE_READ_GENERATION_REASON,
    CallerPrincipalRefused,
    CasRetriesExhausted,
    CasVersionConflict,
    CoherenceDegradedWarning,
    CoherenceError,
    CommitPreempted,
    CommitUnconfirmed,
    InternalConcurrencyError,
    PublishMaterializationError,
    RedirectRefused,
    RemoteAuthFailed,
    StaleView,
    ViewWedged,
)
from ccs.core.fence import confirmed_generation

logger = logging.getLogger(__name__)

__all__ = ["CoherentVolume", "coherent_workspace", "install", "uninstall"]


class _ReadResult(NamedTuple):
    """What one coordinator-mediated read yields. Named rather than a bare
    tuple because two of its fields are int-shaped and adjacent
    (``version`` and ``owner_generation``), so a positional transposition would
    type-check in one direction while silently swapping a value comparand for
    an authority one."""

    data: bytes
    version: int
    stale_denied: bool
    owner_generation: int | None
    #: The pre-read classified this instance as WITHOUT a standing grant on the
    #: current version (the ``status == "stale"`` shape — a warn-mode re-grant
    #: or a strict deny). The effect fence reads this: a caller whose grant did
    #: not stand at re-validate must HOLD even when the ``(version,
    #: owner_generation)`` pair is unchanged, because a peer's write-claim
    #: acquire preempts a holder while moving NEITHER comparand.
    stale_status: bool
    #: The coordinator reported that ``data`` is NOT the content it records at
    #: ``version`` (``hash_differs``, on either response shape). Together with
    #: ``stale_denied`` this names the split pair a read inside a peer's
    #: commit→disk window produces: old bytes under the new version. ``False``
    #: also covers a coordinator holding no recorded hash to compare against;
    #: the wire does not tell the two apart.
    content_differs: bool

    @property
    def split_pair(self) -> bool:
        """The coordinator refused this read AND said ``data`` is not its content
        at ``version``: a pair no caller may CAS from, so the public reads refuse
        it and the read does not advance the foreign-edit baseline (a path's
        first read still records one; see :meth:`_read_with_version`)."""
        return self.stale_denied and self.content_differs


class _Sent(NamedTuple):
    """What one POST from :meth:`CoherentVolume._send` yielded: the 2xx body,
    or the typed reason of a caller-principal refusal left to the caller's
    recovery (R20). Both ``None`` after any other failure, which already went
    through ``on_error`` (degrade mode only — strict raised)."""

    body: dict | None
    principal_refusal: str | None


# Plan Unit 6 (R6): client-side bound on the OCC re-mint→re-commit loop in
# :meth:`CoherentVolume.write_cas`. Mirrors ``SyncStrategy.max_cas_retries``
# (the in-process library knob) — the HTTP path runs its own bounded loop via
# ``_remint()`` + a fresh hash-checked read rather than the
# ``AgentRuntime``/``SyncStrategy`` loop (which governs only the in-process
# library path). Bounds TWO independent failure modes, both fail-closed:
#
# - Total commit (CAS POST) attempts is ``MAX_CAS_REACQUIRES + 1`` (initial +
#   retries); on exhaustion ``write_cas`` raises
#   :class:`~ccs.core.exceptions.CasRetriesExhausted` (a typed terminal, never
#   a silent drop). A stale-denied comparand read never POSTs, so it does NOT
#   consume this budget.
# - CONSECUTIVE stale-denied comparand reads are bounded at
#   ``MAX_CAS_REACQUIRES + 1``, WITH a capped-exponential wait between re-reads
#   (the transient clears on a peer's progress, so the poll yields to it — see
#   ``DENIED_READ_BACKOFF_BASE_SEC``); on exhaustion ``write_cas`` raises
#   :class:`~ccs.core.exceptions.CoherenceError` (a view that never clears —
#   wedged coordinator / perpetually lagging disk — must not spin). A clean
#   read resets the streak.
MAX_CAS_REACQUIRES = 8

# A stale-denied comparand read has two causes. One is LOCAL and clears on the
# ``_remint()`` alone (this instance is ``INVALID``). The other is a TRANSIENT that
# clears only on ANOTHER writer's progress: the window between a peer's confirmed
# CAS and its ``_atomic_write`` landing on disk. Because the second cause is the one
# a streak is made of, the denied-read retry must WAIT, not spin.
# Polling it with no delay made the streak bound a proxy for CPU scheduling luck
# rather than for a wedged view: measured, 9 undelayed re-reads burn through in
# ~37ms of wall clock, so a peer descheduled inside that window (routine on a
# loaded 2-vCPU CI runner) wedged the loser even though the view would have
# cleared moments later. Worse, the undelayed loser COMPETES for CPU with
# the very peer whose disk write unblocks it. A capped-exponential wait between
# denied re-reads yields to that peer and denominates the bound in time, so the
# streak still fails closed on a genuinely never-clearing view (foreign edit,
# wedged coordinator) without going red on scheduler jitter.
# The schedule below bounds ONE streak (8 waits, 212ms). A clean read resets the
# streak, so a single ``write_cas`` call that alternates streaks with lost CAS races
# can traverse up to ``MAX_CAS_REACQUIRES + 1`` of them — the per-call wait is
# bounded, but by ~1.9s in aggregate, not by one schedule.
DENIED_READ_BACKOFF_BASE_SEC = 0.002
DENIED_READ_BACKOFF_CAP_SEC = 0.05

PRINCIPAL_CLAIM_NOT_ATTEMPTED = "not_attempted"
"""What :attr:`CoherentVolume.principal_claim_outcome` reads while no claim
has been made for the session: before attach, after an attach that reached no
coordinator, and in a forked child until it re-attaches. The fifth arm beside
the four a claim itself can settle (``PrincipalClaim.outcome``)."""

PrincipalClaimOutcome = Literal["bound", "unsupported", "refused", "unconfirmed", "not_attempted"]


def denied_read_backoff_sec(refusals: int) -> float:
    """The wait before the next read after ``refusals`` consecutive refused
    reads (counted from 1): capped-exponential from the base to the cap.

    ``write_cas`` and ``WorkspaceVersioner``'s restore leg both wait on this
    schedule for the same transient, a peer's commit still reaching disk. Each
    caller keeps its own count: ``write_cas`` resets its streak on a clean read,
    and a restore leg counts refusals across its whole budget."""
    return min(
        DENIED_READ_BACKOFF_BASE_SEC * (2 ** (refusals - 1)),
        DENIED_READ_BACKOFF_CAP_SEC,
    )


# SB-23 content-CAS deny message. Byte-stable (no path/hash interpolation) so a
# model's retry loop sees identical text each attempt (KTD-P), and distinct from
# the coordinator's INVALID-deny prose so the two are not conflated.
_STALE_WRITE_DENY_REASON = (
    "stale write blocked: the file on disk changed since this view was read "
    "(out-of-band edit). reacquire() and rebuild the write from the fresh bytes."
)

# SB-18 atomic-publish HELD message. Byte-stable (no path/version interpolation)
# so a model's retry loop sees identical text each attempt (KTD-P). The specific
# member drift travels on StaleView.expected_version / .current_version instead.
_PUBLISH_HELD_REASON = (
    "atomic publish held: a peer committed a member of this write-set since it "
    "was read, so the batch was NOT published (all-or-nothing — no file was "
    "written). reacquire() and rebuild the publish from the fresh versions."
)

# What a multi-member atomic_publish raises (CommitUnconfirmed) when the batch
# commit is not confirmed, or is answered with nothing this client can
# classify: whether it landed at the coordinator is unknown, and CAS-first
# means no member touched disk. Built from constants — no coordinator text.
_PUBLISH_UNCONFIRMED_MESSAGE = (
    "atomic_publish commit was not confirmed (the coordinator answered that "
    "its commit is unconfirmed); whether the batch landed at the coordinator "
    "is unknown, and no file was written. Re-read every member, and retry only "
    "if the publish is absent."
)
_PUBLISH_UNCLASSIFIABLE_MESSAGE = (
    "atomic_publish commit was answered with no outcome this client can "
    "classify (not a win, and no reason); whether the batch landed at the "
    "coordinator is unknown, and no file was written. Re-read every member, "
    "and retry only if the publish is absent."
)


# What write()'s grant request (pre-edit) had already done when its commit is
# refused for the caller principal after the bytes reached disk. On a path the
# coordinator tracks, an admitted EXCLUSIVE acquire invalidates every peer
# holding a copy (KTD-1), which the refusal of the commit cannot undo; on a
# path it does not track, the request takes its fast path and does neither.
# Both answer a bare {"ok": true}, and nothing the volume holds says which
# (its managed globs do not: a coordinator that predates #261 lets an
# ignored.yaml or a runtime /policy/untrack untrack a managed path; a current
# one keeps a strict path tracked for its lifetime, but the volume cannot tell
# which it attached to), so the text states both cases.
# A degraded or lost answer leaves the grant and the peers unknown.
_GRANT_REQUEST_ADMITTED = (
    "If the coordinator tracked the path when it admitted this write's grant "
    "request, that request invalidated every peer it had recorded as holding a "
    "copy and took a grant this volume has not released; if it did not track "
    "the path, the request took no grant and invalidated no peer."
)
_GRANT_REQUEST_UNCONFIRMED = (
    "The answer to this write's grant request was not confirmed, so whether it "
    "invalidated any peer, and whether it took a grant this volume now holds, "
    "is not known."
)


def _unclassifiable_cas_message(rel: str) -> str:
    """What an OCC commit reports when its answer is not a win and carries no
    string reason. The coordinator's own non-win answers always carry one, so
    this is what a proxy or gateway in front of it could send; nothing says
    whether the commit landed, so it is the unknown outcome
    (:class:`~ccs.core.exceptions.CommitUnconfirmed`) — and CAS-first means the
    bytes never touched disk. Built from constants: no coordinator text."""
    return (
        f"OCC commit of {rel} was answered with no outcome this client can "
        "classify (not a win, and no reason); whether it landed at the "
        "coordinator is unknown, and the write did not touch disk. Re-read, "
        "and retry only if it is absent."
    )


def _unrecorded_write_state(rel: str, *, wrote: bool, grant_confirmed: bool) -> str:
    """What :meth:`CoherentVolume.write` reports, ahead of the refusal's own
    text, when its commit (post-edit) is refused for the caller principal
    after its grant request: the state the write left — the file (``wrote``:
    whether this call put the bytes there, or they were already on disk),
    the coordinator's record of this write's commit, the peers and the
    grant, each as far as this client knows it. Built from constants and the
    path — no coordinator text, so no principal or nonce.

    It says the commit was not recorded, not that no version moved: the
    grant request registers a path the coordinator has never seen (no
    version, then 1), and a peer can commit between the grant and the
    refused commit."""
    if wrote:
        disk = f"This write put its bytes on disk at {rel}."
    else:
        disk = f"{rel} already held this write's bytes on disk, so this write left the file as it was."
    grant = _GRANT_REQUEST_ADMITTED if grant_confirmed else _GRANT_REQUEST_UNCONFIRMED
    return f"{disk} The coordinator did not record this write's commit. {grant}"


# Split-pair read refusal. Byte-stable (no path/version interpolation) so a
# model's retry loop sees identical text each attempt (KTD-P).
_SPLIT_READ_DENY_REASON = (
    "stale read refused: the bytes on disk are not the content the coordinator "
    "records at its current version, so no version is returned for them. A "
    "peer's commit still reaching disk clears on its own: reacquire() and read "
    "again, retrying with backoff for a few seconds. Only a refusal that "
    "outlasts that means the file was changed outside the coordinator (an "
    "out-of-band edit, or a commit whose disk write failed): then write() the "
    "bytes reacquire() returned, or your merge of them, to record them, and read "
    "again. A write made while a peer's commit is still reaching disk is "
    "overwritten when that commit lands."
)

# Live volumes a forked child must reset (CoherentVolume._after_fork). One
# process-wide handler walks this set because os.register_at_fork cannot
# unregister: a per-instance registration holds its volume, even one whose
# constructor raised, for the life of the process. Weak, so a volume its caller
# drops is collected and a later fork no longer touches it.
_FORK_RESET_VOLUMES: weakref.WeakSet[CoherentVolume] = weakref.WeakSet()


def _reset_volumes_after_fork() -> None:
    # One volume's failure must not skip the rest: each volume used to have its
    # own handler, and CPython reports a failing handler and still runs the next.
    # Reset every volume, then raise what failed so the child still reports it.
    failures: list[BaseException] = []
    for volume in _FORK_RESET_VOLUMES:
        try:
            volume._after_fork()
        except BaseException as exc:
            failures.append(exc)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        # CPython's default unraisable hook prints only a group's own message,
        # not its sub-exceptions, so the message names every failure.
        details = "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
        raise BaseExceptionGroup(f"resetting volumes after fork failed: {details}", failures)


# Guarded like lifecycle's fcntl import: registering at import on a platform
# without fork would make merely importing the adapter fail.
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_volumes_after_fork)


@dataclass(frozen=True)
class ManagedGlobEnforcement:
    """What the coordinator's published policy says about a volume's managed
    globs: three answers, never collapsed into two.

    ``enforced`` are the globs the coordinator carries in both its strict set
    and its tracked set. ``unenforced`` are the declared globs it does not carry
    that way. (On a coordinator that carries #261 an ignore entry never
    untracks a strict path — strict wins — so an ignore covering a managed glob
    under another spelling is harmless there; a literal ignore of the glob is
    still reported as not enforced, which is conservative for an older one.)
    ``unavailable`` is set when the sets could not be read at all — an older
    coordinator whose summary publishes only counts, the Node coordinator, which
    does not serve the operator view, or a ``/status`` that failed — and then
    both lists are empty and say nothing: "cannot tell" is not "not enforced",
    and it is never "enforced".
    """

    enforced: tuple[str, ...]
    unenforced: tuple[str, ...]
    unavailable: str | None

    @property
    def confirmed(self) -> bool:
        """True only when the sets were read and every declared glob is enforced."""
        return self.unavailable is None and not self.unenforced


class CoherentVolume:
    """Coherent shared workspace for a single-host agent fleet (v1).

    .. warning::

       **An instance is NOT thread-safe — one operation at a time, one instance
       per thread (A5).** ``read``/``write``/``write_cas`` read the coordinator
       identity lock-free while a re-mint (``reacquire``, the OCC retry) or
       ``_after_fork`` rotates it, so overlapping calls on a SINGLE instance
       from different threads could split an in-flight CAS across identities.
       This is misuse, and it is made LOUD: a public op
       that detects another op already in flight on the same instance raises
       :class:`~ccs.core.exceptions.CoherenceError` rather than corrupting
       silently. Each thread (and each forked child) must own its OWN instance —
       per-instance identity is exactly what makes distinct writers distinct. The
       guard is re-entrant for the same thread, so the internal
       ``write_cas`` → ``reacquire`` → ``read`` nesting is unaffected.

    The appliance **spawns** the coordinator for ``root`` and enables strict
    mode on the ``managed`` globs (by writing ``.coherence/tracked.yaml`` and
    ``.coherence/strict_mode.yaml`` before spawn — the coordinator loads policy
    once at startup, so enablement must precede the spawn). Strict mode is what
    gives invalidation teeth: a write from an ``INVALID`` holder is denied, and
    the façade surfaces that deny *fail-closed* so the caller re-reads.

    ``on_error``:

    - ``"strict"`` (default): any coherence failure — a coordinator that is
      unavailable, or one already running that we cannot enable strict on —
      raises :class:`~ccs.core.exceptions.CoherenceError`. Fail-closed.
    - ``"degrade"``: the same conditions warn once
      (:class:`~ccs.core.exceptions.CoherenceDegradedWarning`) and the appliance
      operates best-effort (coherence may be off). Mirrors the other adapters.

    ``on_stale_read`` (read surface):

    - ``"allow"`` (default): a strict-mode foreign-edit / stale-view deny on
      :meth:`read` is swallowed — read returns the current on-disk bytes
      (back-compatible; the coordinator still detects, denies, counts, and
      audits the foreign edit).
    - ``"raise"``: that same deny surfaces as
      :class:`~ccs.core.exceptions.StaleView` so the caller can abort or
      :meth:`reacquire` instead of silently acting on the foreign bytes.
      Independent of ``on_error`` (which governs only infra failures);
      :meth:`reacquire`'s recovery read always bypasses it.

    ``on_stale_write`` (write surface — SB-23 content-CAS):

    - ``"raise"`` (default): at :meth:`write`, if the file on disk changed since
      this instance last read/wrote it (a foreign / out-of-band edit), the write
      is denied with :class:`~ccs.core.exceptions.StaleView` rather than silently
      clobbering it — :meth:`reacquire` and rebuild from the fresh bytes. The safe
      default *guards*, because a write is a data-loss surface (the inverse of
      ``on_stale_read``, where returning bytes is always safe).
    - ``"allow"``: the foreign edit is not raised; the write proceeds and clobbers
      (pre-SB-23 behavior). Adapter-local: fires for a **managed (strict) path**
      whenever a per-path baseline exists, independent of ``on_error`` / coordinator
      attachment. An unmanaged or tracked-but-non-strict path is exempt — there a
      coordinated peer change can hit the disk without an INVALID deny, so a
      mismatch is not necessarily foreign.

    **Fleet requirement (hard, v1).** Every instance coordinating a given
    workspace MUST declare the **same** ``managed`` globs: an attaching instance
    adds no globs to a running coordinator's policy (only the track and untrack
    commands change it). At attach each instance checks every glob it declared
    against the glob sets the coordinator publishes in its operator view
    (:meth:`managed_glob_enforcement`) and fails closed — strict raises,
    degrade warns and runs detached — when any is not enforced (not tracked,
    not strict, or ignored), naming the globs, or when the sets cannot be read
    at all. The check runs once, at attach, and its answer holds for the
    coordinator's lifetime: the coordinator refuses to untrack a strict path
    (typed reason ``untrack_strict_path``), an ignore pattern never overrides a
    strict one, and its hot reloads never drop a strict or tracked pattern
    (#261); a strict path stops being enforced only when the coordinator
    restarts without its strict entry. Before this check existed a sibling whose globs differed from the
    spawner's passed on the spawner's pattern *count*, and its stale writes
    landed with no signal (#190).
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        managed: tuple[str, ...] = (),
        on_error: Literal["strict", "degrade"] = "strict",
        on_stale_read: Literal["allow", "raise"] = "allow",
        on_stale_write: Literal["allow", "raise"] = "raise",
        config: LifecycleConfig | None = None,
        bind_host: str = "127.0.0.1",
        remote_endpoint: CoordinatorEndpoint | None = None,
    ) -> None:
        if on_error not in ("strict", "degrade"):
            raise ValueError(f"on_error must be 'strict' or 'degrade', got {on_error!r}")
        if on_stale_read not in ("allow", "raise"):
            raise ValueError(
                f"on_stale_read must be 'allow' or 'raise', got {on_stale_read!r}"
            )
        if on_stale_write not in ("allow", "raise"):
            raise ValueError(
                f"on_stale_write must be 'allow' or 'raise', got {on_stale_write!r}"
            )
        if remote_endpoint is not None:
            # Cross-host demo path (R0b). Defense-in-depth: remote mode is gated by
            # the default-OFF flag, is strict-only (an unreachable coordinator must
            # fail closed, never degrade-open), and is coordinator-version-only —
            # no managed globs, because strict mode is load-once-at-spawn and cannot
            # be enabled on a coordinator this client did not spawn.
            if not RemoteCoordinatorConfig.from_env().enabled:
                raise ValueError(
                    "remote_endpoint requires the CCS_REMOTE_COORDINATOR env flag"
                )
            if on_error != "strict":
                raise ValueError(
                    "remote coordinator mode is strict-only (on_error must be 'strict')"
                )
            if managed:
                raise ValueError(
                    "remote coordinator mode is coordinator-version-only; managed globs "
                    "are not supported cross-host in v1"
                )

        self._root = Path(root).resolve()
        # Managed globs are coordinator-policy patterns (parent-repo-relative,
        # same grammar as tracked.yaml). They must be tracked AND strict for the
        # INVALID-deny to fire (is_strict_mode ⊂ is_tracked).
        self._managed: tuple[str, ...] = tuple(managed)
        self._on_error = on_error
        self._on_stale_read = on_stale_read
        self._on_stale_write = on_stale_write
        self._config = config
        self._bind_host = bind_host
        # Cross-host demo: when set, attach connects to THIS endpoint and never
        # spawns a local coordinator (R0b / plan C1). Preserved across fork so the
        # child re-attaches to the same remote coordinator under a fresh identity.
        self._remote_endpoint = remote_endpoint

        self._lock = threading.Lock()
        # A5: single-instance concurrency guard, SEPARATE from self._lock (which
        # protects identity mutation in reacquire and the degradation count;
        # _after_fork replaces both locks instead of taking them). A non-reentrant
        # threading.Lock here would deadlock with the reacquire-within-write_cas
        # path; instead we track the owning thread + a re-entry depth so the SAME
        # thread's nested internal calls (write_cas → reacquire → read) pass while
        # an OVERLAPPING call from a DIFFERENT thread raises. _guard_meta_lock is
        # held only for the microsecond check-then-set, never across an operation.
        self._guard_meta_lock = threading.Lock()
        self._guard_owner_ident: int | None = None
        self._guard_depth = 0
        self._degradation_count = 0
        self._endpoint: CoordinatorEndpoint | None = None
        # Set by the fork child-handler so the next read/write re-attaches (the
        # child cannot re-attach inside the fork handler — see _ensure_attached).
        self._needs_reattach = False
        # Per-path commit hashes for the Unit 2 write no-op-skip; declared here so
        # the fork handler and reacquire() can reset it.
        self._last_committed_hash: dict[str, str] = {}
        # SB-23 content-CAS baseline: the hash this instance last OBSERVED on disk
        # per path (seeded on read, advanced on its own writes). SEPARATE from
        # _last_committed_hash (own-write hashes for the no-op-skip fast path). At
        # write time a current disk hash that differs from this baseline is a
        # foreign / out-of-band edit. Cleared on remint/fork alongside the sibling.
        self._last_observed_hash: dict[str, str] = {}
        # Incarnations that write() may have left holding EXCLUSIVE/MODIFIED and
        # that no confirmed release has cleared yet, each with the paths it wrote.
        # The re-mint that abandons one releases it (see _remint); only write()
        # adds to it, because it is the only path that takes a grant a later
        # optimistic commit is refused on. The paths decide whether write_cas
        # must rotate first: a held grant refuses a commit of that file only.
        self._grant_incarnations: dict[str, set[str]] = {}

        self._mint_identity()
        self._attach()
        # A forked child must not share the parent's identity (single-writer
        # would conflate them) or its cached endpoint/connection. Joined only
        # once construction succeeded, so a constructor that raised leaves
        # nothing behind for a forked child to reset.
        _FORK_RESET_VOLUMES.add(self)

    # --- identity -----------------------------------------------------------

    def _mint_identity(self) -> None:
        # A v4 UUID string satisfies the coordinator's session_id regex
        # (``_SESSION_ID_RE``). Per-instance (not per-process): two volumes in
        # one process are distinct writers. The session id is stable for the
        # volume's lifetime; what a re-mint changes is the incarnation, which
        # every request carries in the subagent field, so the coordinator's
        # grant-row key ``session_to_agent_id(session_id, incarnation)`` is new
        # per attempt while the session id is not.
        self._session_id = str(uuid.uuid4())
        self._incarnation = self._new_incarnation()
        self._new_principal_claim()

    def _new_principal_claim(self) -> None:
        # The caller principal is bound to the SESSION, so it is obtained once per
        # session — at attach, and in a forked child for the child's own session —
        # and never at _remint, which keeps the session (plan KTD14). A long-lived
        # caller holds the nonce and the principal in memory and never looks a
        # principal up by the identity it names, so it cannot present another
        # identity's by accident (KTD5). That is accident-resistance, not
        # unreadability: the coordinator keeps every principal it issued in
        # ``.coherence/state.db``, which any process of the same OS user can read.
        # The nonce lives as long as the session: it is what a later claim
        # presents to re-obtain the binding after a lost answer or a refused
        # principal (R20). ``_principal`` is ``None`` until the claim at attach,
        # and for good against a coordinator that issues none.
        self._mint_nonce = secrets.token_urlsafe(32)
        self._principal: str | None = None
        # The last claim's outcome (the PrincipalClaim vocabulary): "unconfirmed"
        # is claimed again, same nonce, before the next request; "refused" (the
        # session is bound under another nonce) is never claimed again.
        self._claim_outcome: str | None = None

    @staticmethod
    def _new_incarnation() -> str:
        # 32 lowercase hex characters, inside the coordinator's subagent-id shape
        # (``^[A-Za-z0-9_-]{1,64}$``). The shape is load-bearing: the coordinator
        # reads an out-of-shape value as "no subagent", which would put every
        # attempt on the ONE parent row and bring back the sticky deny a re-mint
        # exists to shed.
        return uuid.uuid4().hex

    def _after_fork(self) -> None:
        # Runs in the child after fork, where the forking thread is the only
        # thread. A lock another parent thread held at the fork stays held here
        # with nothing left to release it, and an operation it had in flight
        # never finishes, so replace the locks instead of taking them and drop
        # that operation's claim on the single-op guard.
        self._lock = threading.Lock()
        self._guard_meta_lock = threading.Lock()
        self._guard_owner_ident = None
        self._guard_depth = 0
        # Re-mint identity (a forked child is a new writer, so it gets its own
        # session id, not just a new incarnation), drop the inherited endpoint
        # (its connection/secret context belongs to the parent), and clear
        # per-path beliefs the child has not established itself. Releases
        # NOTHING, and forgets the grants write() recorded: those belong to the
        # parent's identity, which is still live in the parent.
        self._session_id = str(uuid.uuid4())
        self._incarnation = self._new_incarnation()
        # The parent's principal is bound to the parent's session: discard it
        # (and its nonce); the child claims for its own session when it
        # re-attaches.
        self._new_principal_claim()
        self._grant_incarnations.clear()
        self._endpoint = None
        self._needs_reattach = True
        self._last_committed_hash.clear()
        self._last_observed_hash.clear()  # SB-23: child re-seeds its own baselines

    def _ensure_attached(self) -> None:
        """Lazily re-attach after a fork dropped the endpoint.

        ``_after_fork`` re-mints identity and clears the endpoint but cannot
        re-attach in the fork handler (the coordinator-client context is the
        parent's). The child's first ``read``/``write`` re-attaches here,
        sibling-attaching to the coordinator under the child's fresh identity.
        A no-op outside the post-fork window. Under ``on_error="strict"`` a
        failed attempt keeps that window open: every ``read``,
        ``read_with_version``, ``read_with_version_generation``, ``write``,
        ``write_cas``, ``write_cas_at`` and ``atomic_publish`` retries until one
        attaches.
        """
        if self._endpoint is None and self._needs_reattach:
            # Cleared BEFORE the attempt: _attach reads .coherence/ files, and
            # under the install() shim with a managed glob that matches them each
            # of those reads re-enters here. The cleared flag stops the recursion.
            self._needs_reattach = False
            degradations_before = self._degradation_count
            try:
                self._attach()
            except BaseException as exc:
                # Detach in both modes: a failure can escape after the endpoint
                # was resolved (an interrupt during the strict check), and an
                # endpoint left set would route later ops through a coordinator
                # never confirmed to enforce our paths. Strict re-arms, so the
                # next read or write retries the re-attach instead of taking the
                # unattached branch — which skips the coordinator and its
                # invalidations for the child's whole life (the retry runs only
                # while the endpoint is None). Degrade keeps its one attempt, as
                # at construction, and so runs best-effort from here on: record
                # that, since an error _attach did not route through
                # _fail_closed_or_degrade would otherwise leave is_degraded False.
                self._endpoint = None
                if self._on_error == "strict":
                    self._needs_reattach = True
                elif self._degradation_count == degradations_before:
                    # Unchanged count: _attach did not record this failure itself
                    # (a handled one whose warning a caller escalated to an error
                    # escapes here already counted). Suppress our own escalated
                    # warning so the failure propagates as itself — an interrupt
                    # must not come back as an Exception.
                    with contextlib.suppress(CoherenceDegradedWarning):
                        self._record_degraded(f"re-attach after fork failed: {exc!r}")
                raise

    @property
    def session_id(self) -> str:
        """The per-instance coordinator session id (a v4 UUID string).

        Stable for the volume's lifetime: :meth:`reacquire` and the optimistic
        retries shed stale coordinator state without changing it. A forked child
        gets its own."""
        return self._session_id

    @property
    def is_attached(self) -> bool:
        """True if a coordinator endpoint was resolved (strict-mode owner)."""
        return self._endpoint is not None

    @property
    def principal_claim_outcome(self) -> PrincipalClaimOutcome:
        """What this session's last caller-principal claim settled — the
        ``PrincipalClaim`` vocabulary, plus :data:`PRINCIPAL_CLAIM_NOT_ATTEMPTED`
        while no claim has been made for the session.

        ``bound``: the session holds its principal and presents it on every
        request. ``unsupported``: the coordinator issues none, so none is
        presented. ``unconfirmed``: the last claim's answer was lost; the same
        nonce is claimed again before the next request. ``refused``: the
        session is bound under another nonce, nothing is claimed again, and
        every request a require-class route (or, for a presented principal, any
        route) refuses raises :class:`~ccs.core.exceptions.CallerPrincipalRefused`
        — durable for this session, so a reader can tell "this session lost
        coordination" from a transient ahead of its next write. Read by the
        MCP ``swg_status``. Never the principal or the nonce."""
        return self._claim_outcome or PRINCIPAL_CLAIM_NOT_ATTEMPTED

    # --- single-instance concurrency guard (A5) -----------------------------

    @contextlib.contextmanager
    def _single_op_guard(self) -> Iterator[None]:
        """Reject overlapping use of ONE instance across threads (A5).

        A :class:`CoherentVolume` instance is single-threaded by contract: one
        operation at a time. Concurrent ``read``/``write``/``write_cas`` on the
        same instance from different threads could split an in-flight CAS across
        identities (a re-mint or ``_after_fork`` rotates the coordinator identity
        while the lock-free op path reads it). This guard makes that misuse LOUD
        rather than silently corrupting: a second thread entering while another
        holds the guard raises ``InternalConcurrencyError`` (a ``CoherenceError``
        subclass).

        Re-entrant for the SAME thread so internal nesting works (``write_cas``
        calls :meth:`reacquire`, which calls :meth:`read`): the owning thread
        bumps a depth counter instead of self-deadlocking. A separate
        non-reentrant ``threading.Lock`` would deadlock that path, and silently
        serializing instead of raising would hide the misuse + risk deadlock with
        ``self._lock`` — so we detect-and-raise, never block.
        """
        ident = threading.get_ident()
        with self._guard_meta_lock:
            if self._guard_owner_ident is not None and self._guard_owner_ident != ident:
                # A server-misuse bug (the MCP server serializes tool access), not
                # an agent-recoverable deny — typed so the mapper never relays it
                # as a retryable stale view.
                raise InternalConcurrencyError(
                    "CoherentVolume is single-threaded; concurrent use detected. "
                    "One operation at a time per instance — use one instance per "
                    "thread (an in-flight read/write/write_cas can re-mint identity "
                    "via reacquire, so overlapping ops on one instance could split a "
                    "CAS across identities)."
                )
            self._guard_owner_ident = ident
            self._guard_depth += 1
        try:
            yield
        finally:
            with self._guard_meta_lock:
                self._guard_depth -= 1
                if self._guard_depth <= 0:
                    self._guard_depth = 0
                    self._guard_owner_ident = None

    # --- attach dispatch ----------------------------------------------------

    def _attach(self) -> None:
        """Dispatch attach: local spawn-with-strict, or remote connect-only.

        Remote mode (the cross-host demo) NEVER spawns a local coordinator."""
        if self._remote_endpoint is not None:
            self._attach_remote()
        else:
            self._attach_with_strict()

    def _attach_remote(self) -> None:
        """Connect-only attach to a REMOTE coordinator — never spawn (plan C1).

        The endpoint was supplied explicitly, so we set it directly and read
        nothing from the local ``.coherence/`` directory. We do NOT call
        ``connect_or_spawn``: a remote client with no local ``server.pid`` would
        otherwise spawn a second, LOCAL coordinator and silently split this writer
        and its peer onto different coordinators — the demo would "work" for the
        wrong reason. A remote coordinator we cannot reach fails closed (remote
        mode is strict-only). Response-body semantics (deny / degrade / 401) are
        enforced on the actual read/write ops (R2)."""
        endpoint = self._remote_endpoint
        try:
            _coordinator_get(endpoint, "/status")
        except CoordinatorUnavailable as exc:
            # Transport failure = unreachable -> fail closed (strict raises, so the
            # lines below are unreachable today). They are kept defensively: if a
            # future non-strict remote mode ever lets _fail_closed_or_degrade return
            # instead of raise, we must stay DETACHED, never attached to a dead endpoint.
            self._fail_closed_or_degrade(f"remote coordinator unreachable: {exc}")
            self._endpoint = None
            return
        except urllib.error.HTTPError as exc:
            # R2: a 401 at attach is a wrong/missing secret — fail loud + closed.
            if exc.code == 401:
                raise RemoteAuthFailed(
                    "remote coordinator rejected the bearer token (401) at attach; "
                    "the remote secret (CCS_REMOTE_SECRET_FILE) does not match the "
                    "coordinator's hook.secret"
                ) from exc
            # Any other non-2xx at attach (403 Host mismatch, 503 draining, ...)
            # fails CLOSED here rather than deferring a misconfig to the first op.
            self._fail_closed_or_degrade(
                f"remote coordinator returned HTTP {exc.code} at attach"
            )
            self._endpoint = None
            return
        self._endpoint = endpoint
        self._claim_principal()

    # --- spawn-with-strict --------------------------------------------------

    def _attach_with_strict(self) -> None:
        coherence_dir = self._root / ".coherence"
        pid_file = coherence_dir / "server.pid"
        cfg = self._config or LifecycleConfig()

        # Strict mode is load-once-at-startup, so only the process that SPAWNS
        # the coordinator can enable it. Write our policy YAML only when no
        # coordinator is running yet (we are about to spawn it). If one is
        # already running we must NOT mutate its policy files: our write cannot
        # take effect (load-once), and clobbering a foreign coordinator's policy
        # is a side effect on someone else's workspace. We verify enforcement
        # after attaching instead.
        pre_port = read_port_from_file(pid_file)
        pre_existing = pre_port is not None and tcp_probe(
            pre_port, cfg, bind_host=self._bind_host
        )
        if not pre_existing:
            self._write_policy_yaml()

        port = connect_or_spawn(self._root, config=cfg, bind_host=self._bind_host)
        if port == -1:
            self._fail_closed_or_degrade(
                "coordinator unavailable (could not spawn or attach for this workspace)"
            )
            self._endpoint = None
            return

        try:
            self._endpoint = resolve_endpoint(self._root)
        except CoordinatorUnavailable as exc:
            self._fail_closed_or_degrade(str(exc))
            self._endpoint = None
            return

        # Verify that the coordinator enforces strict mode for EVERY managed
        # glob, from the glob sets it publishes in its operator view. One
        # post-attach check covers the cases:
        #   * we spawned the coordinator      -> it loaded our YAML -> enforced
        #   * a sibling spawned it with the SAME globs -> the fleet case: enforced
        #   * a sibling spawned it with OTHER globs, a foreign coordinator
        #     (e.g. a Claude Code session), or a policy that ignores our globs
        #     -> our globs are not enforced and an attach adds none -> fail
        #     closed, naming them. The comparison is literal and taken once;
        #     the coordinator keeps the answer true for its lifetime (#261:
        #     strict wins over ignore, /policy/untrack refuses a strict path,
        #     a reload never drops a strict pattern).
        #   * the sets cannot be read (an older coordinator publishing counts
        #     only, the Node coordinator, a failed /status) -> cannot tell ->
        #     fail closed, saying so; never read as enforced.
        # A pattern COUNT is not a check: the spawner's one pattern satisfied
        # every later attacher, whose untracked paths then answered every hook
        # as if enforced, so its stale writes landed with no signal (#190).
        if self._managed:
            enforcement = self.managed_glob_enforcement()
            if not enforcement.confirmed:
                # Detach BEFORE failing, in both modes. Do NOT keep a live
                # endpoint to a coordinator that does not enforce our paths —
                # that would route reads/writes through it while is_attached
                # reported True. Degrade falls through detached, mirroring the
                # other two degrade branches; strict raises detached rather
                # than relying on the caller to drop the endpoint.
                self._endpoint = None
                self._fail_closed_or_degrade(self._enforcement_failure(enforcement))
                return
        self._claim_principal()

    def _enforcement_failure(self, enforcement: ManagedGlobEnforcement) -> str:
        """The message for an attach the per-glob check did not confirm: which
        globs are not enforced, or why enforcement could not be told."""
        tail = (
            " Every volume on a workspace must declare the globs the coordinator was "
            "started with; an attaching volume adds none to a running coordinator's "
            "policy. Stop it, or use a dedicated workspace root. In degrade mode the "
            "volume operates best-effort with coherence enforcement off."
        )
        if enforcement.unavailable is not None:
            globs = ", ".join(self._managed)
            return (
                f"enforcement of managed glob(s) {globs} cannot be confirmed: "
                f"{enforcement.unavailable}; failing closed." + tail
            )
        globs = ", ".join(enforcement.unenforced)
        return (
            "attached to a coordinator whose policy does not enforce strict mode for "
            f"managed glob(s) {globs}: they are not in its strict set, not tracked, "
            "or ignored."
            + tail
        )

    def _claim_principal(self) -> None:
        """Obtain this session's caller principal: once per session, at attach.

        A coordinator that answers 404 issues none (the sibling Node coordinator,
        or an older Python one) and the volume proceeds without a header, as
        before. A claim REFUSED because the session is already bound under
        another nonce routes through ``on_error`` as the typed
        :class:`~ccs.core.exceptions.CallerPrincipalRefused` and is never made
        again: a retry would present the same nonce and meet the same refusal,
        and a new nonce would be a second claimant (KTD11). The one repeat is a
        strict forked child: a refusal during its lazy re-attach leaves it
        detached (:meth:`_ensure_attached`), so each later operation re-runs the
        re-attach, presents the SAME nonce and raises again; it never runs
        without a principal. An UNCONFIRMED claim
        may have bound anyway (R20): it routes through ``on_error`` too — strict
        raises, degrade warns once — and the SAME nonce is claimed again before
        this volume's next request (:meth:`_settle_unconfirmed_claim`). Nothing
        is re-minted."""
        if self._endpoint is None:
            return
        claim = claim_caller_principal(self._endpoint, self._session_id, self._mint_nonce)
        self._adopt_claim(claim)
        if claim.outcome == "refused":
            self._refuse_principal(
                CALLER_PRINCIPAL_CLAIMED_REASON,
                f"coordinator caller principal not obtained (refused: {claim.detail})",
            )
        elif claim.outcome == "unconfirmed":
            self._fail_closed_or_degrade(
                f"coordinator caller principal not obtained (unconfirmed: {claim.detail}); "
                "the same mint nonce is claimed again before the next request"
            )

    def _adopt_claim(self, claim: PrincipalClaim) -> None:
        """Record ``claim``'s outcome, and the principal it settles: the bound
        one, or none when the coordinator issues none."""
        self._claim_outcome = claim.outcome
        if claim.outcome == "bound":
            self._principal = claim.principal
        elif claim.outcome == "unsupported":
            self._principal = None

    def _settle_unconfirmed_claim(self) -> None:
        """Claim again, presenting the SAME nonce, while this session's last
        claim is unconfirmed — before every request, until an answer settles it.

        A bind that landed while its answer was lost leaves the session bound
        and this volume without the principal every require-class route now
        demands; this is the recovery R20 promises, and it adds no binding (the
        nonce is the first claim's). Still unconfirmed: the request goes out
        without a principal and reports its own outcome. Refused now: routes
        through ``on_error`` as at attach."""
        if self._claim_outcome != "unconfirmed" or self._endpoint is None:
            return
        claim = claim_caller_principal(self._endpoint, self._session_id, self._mint_nonce)
        self._adopt_claim(claim)
        if claim.outcome == "refused":
            self._refuse_principal(
                CALLER_PRINCIPAL_CLAIMED_REASON,
                f"coordinator caller principal not obtained (refused: {claim.detail})",
            )

    def _refuse_principal(self, reason: str, message: str) -> None:
        """A CLAIM that did not bind, through ``on_error``: strict raises the
        typed :class:`~ccs.core.exceptions.CallerPrincipalRefused` carrying
        ``reason``; degrade warns once and counts, and the volume goes on
        without a principal — the routes that admit none still serve it, and a
        request one of them refuses raises from :meth:`_post` in both modes.
        ``message`` never carries a principal or a nonce."""
        if self._on_error == "strict":
            raise CallerPrincipalRefused(reason, message)
        self._record_degraded(message)

    def _write_policy_yaml(self) -> None:
        """Enable strict mode on the managed globs before the coordinator spawns.

        Writes the managed globs to both ``tracked.yaml`` and ``strict_mode.yaml``
        (a path must be tracked AND strict for the deny to fire). Idempotent —
        merges with any existing entries. No-op when ``managed`` is empty.
        """
        if not self._managed:
            return
        coherence_dir = self._root / ".coherence"
        # This write runs BEFORE the lifecycle creates the directory, so on a
        # fresh workspace it is the creator. The lifecycle requires 0700 here
        # (state.db, hook.secret and the pidfile live inside) and re-tightens
        # anything looser with a warning — a default-mode mkdir would earn every
        # brand-new workspace that warning for a directory we made a moment
        # earlier. An existing directory is left as-is (mkdir never chmods);
        # re-tightening stays the lifecycle's job.
        coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._merge_yaml_list(coherence_dir / "tracked.yaml", self._managed)
        self._merge_yaml_list(coherence_dir / "strict_mode.yaml", self._managed)

    @staticmethod
    def _merge_yaml_list(path: Path, globs: tuple[str, ...]) -> None:
        existing: list[str] = []
        if path.is_file():
            try:
                loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (yaml.YAMLError, OSError):
                loaded = None
            if isinstance(loaded, list):
                existing = [x for x in loaded if isinstance(x, str)]
        merged = sorted(set(existing) | set(globs))
        if merged == sorted(existing):
            return  # nothing new — avoid a needless rewrite
        # Atomic write so a concurrent coordinator spawn never reads a partial file.
        # UUID-suffix the temp name (like _atomic_write) so a predictable ``.tmp``
        # path can't be pre-placed as a symlink by a local same-uid process (the
        # write_text would otherwise follow it).
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(yaml.safe_dump(merged, default_flow_style=False), encoding="utf-8")
        os.replace(tmp, path)

    # --- strict-mode verification (used by tests + callers) -----------------

    def managed_glob_enforcement(self) -> ManagedGlobEnforcement:
        """Check every managed glob against the glob sets the coordinator
        publishes in its operator view (``/status?detail=full``, the same view
        the status command reads), by the coordinator's own rule: a path is
        enforced when it is tracked (a default or user-added pattern), matches
        a strict pattern, and matches no ignored pattern. So a glob is enforced
        when the coordinator carries it in its strict set and its tracked set
        and not in its ignored set. The globs are compared as the strings this
        volume wrote to the policy files, which is what a sibling that declared
        the same globs wrote too; a coordinator that enforces the same paths
        under another spelling, or ignores them under a covering pattern, is
        not matched, so the answer is as literal as the comparison.

        The attach checks this answer once, and the coordinator keeps it true
        while it runs (#261): ``/policy/untrack`` refuses, with the typed reason
        ``untrack_strict_path`` and nothing written, an entry covering a path it
        holds in strict mode; an ignored pattern never takes a strict path off
        the tracked set, whether it was on disk at spawn or arrived later, and
        however broadly it is spelled; and the hot reload behind
        ``/policy/track`` and ``/policy/untrack`` never drops a strict or
        tracked pattern. A strict path stops being enforced only when the
        coordinator restarts without its strict entry, and a volume attaching
        to that coordinator is checked again. (A coordinator that predates the
        fix lets an untrack or a covering ignore put a managed path on the
        untracked fast path — a read reports version 0 and a CAS commit is
        accepted at any expected version — with no signal unless a caller asks
        this method again.)

        The view is read for this check only. When it cannot be read, or carries
        no glob sets, the result is ``unavailable`` with the reason, and nothing
        is concluded about the globs: the attach fails closed on it, and never
        takes a count for confirmation.
        """
        globs = self._managed
        if self._endpoint is None:
            return ManagedGlobEnforcement((), (), "the volume is not attached")
        try:
            status = _coordinator_get(
                self._endpoint,
                "/status?detail=full",
                extra_headers={"Coherence-Local-Operator": "true"},
            )
        except urllib.error.HTTPError as exc:
            # The Node coordinator answers 501 to the operator view and
            # publishes no strict patterns at any tier; name it when the pid
            # file says so. Only the status code is relayed, never the body.
            who = (
                "the Node coordinator does not serve the operator view"
                if coordinator_backend(self._root) == NODE_BACKEND
                else "the operator view was refused"
            )
            return ManagedGlobEnforcement((), (), f"{who} (HTTP {exc.code})")
        except Exception as exc:  # any failure to read the view is "cannot tell"
            return ManagedGlobEnforcement(
                (), (), f"{type(exc).__name__} reading the operator view"
            )
        summary = status.get("policy_summary") if isinstance(status, dict) else None
        sets: dict[str, set[str]] = {}
        for key in (
            "tracked_patterns", "user_added_patterns", "ignored_patterns", "strict_mode_patterns",
        ):
            value = summary.get(key) if isinstance(summary, dict) else None
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                return ManagedGlobEnforcement(
                    (), (),
                    "the coordinator publishes no glob sets in its operator view "
                    "(an older coordinator, whose summary carries only counts)",
                )
            sets[key] = set(value)
        tracked = sets["tracked_patterns"] | sets["user_added_patterns"]
        ignored = sets["ignored_patterns"]
        enforced = tuple(
            g for g in globs
            if g in sets["strict_mode_patterns"] and g in tracked and g not in ignored
        )
        unenforced = tuple(g for g in globs if g not in enforced)
        return ManagedGlobEnforcement(enforced, unenforced, None)

    def strict_mode_active(self) -> bool:
        """True if the attached coordinator reports any strict-mode patterns.

        A COARSE check: it reads ``strict_mode_pattern_count > 0`` from the
        default-tier ``/status`` summary, so it says nothing about WHICH patterns
        are strict. The attach check uses :meth:`managed_glob_enforcement`, which
        compares this volume's own globs against the published sets. Returns
        False if unattached or if the status surface is unavailable.
        """
        if self._endpoint is None:
            return False
        count = self._policy_summary_value("strict_mode_pattern_count")
        return isinstance(count, int) and count > 0

    def _policy_summary_value(self, key: str) -> object | None:
        if self._endpoint is None:
            return None
        try:
            status = _coordinator_get(self._endpoint, "/status")
        except Exception:  # status is best-effort; any failure → unknown
            return None
        if not isinstance(status, dict):
            return None
        summary = status.get("policy_summary")
        if isinstance(summary, dict) and key in summary:
            return summary[key]
        return status.get(key)

    def coordinator_status(self) -> dict | None:
        """The coordinator ``/status`` document, or ``None`` if unattached or the
        status surface is unreachable.

        Best-effort (any failure → ``None``). The MCP ``swg_status`` tool uses it
        to split ``off`` (reachable, no strict patterns) from ``unknown``
        (unreachable) — a distinction :meth:`strict_mode_active` collapses to
        ``False`` — and to read the tracked-artifact versions for ``per_path``.
        """
        if self._endpoint is None:
            return None
        try:
            status = _coordinator_get(self._endpoint, "/status")
        except Exception as exc:  # best-effort; any failure → unknown
            logger.debug("coordinator_status probe failed: %s", exc)
            return None
        return status if isinstance(status, dict) else None

    # --- read / write / reacquire contract (Unit 2) -------------------------

    def read(self, path: str | os.PathLike[str]) -> bytes:
        """Read a workspace file and register a SHARED view of it.

        Always returns the current on-disk bytes. The ``pre-read`` call records
        this instance as a SHARED reader@hash on the coordinator — that is what
        makes a *later* peer write invalidate this instance (the sequential
        stale-read→write guard). Under the default ``on_stale_read="allow"`` a
        stale-read response never makes ``read()`` raise (returning current
        bytes is always safe); under ``on_stale_read="raise"`` a strict-mode
        foreign-edit / stale-view deny raises
        :class:`~ccs.core.exceptions.StaleView` instead — recover via
        :meth:`reacquire`.

        The strict deny is **sticky** (KTD-T): if this instance is already
        ``INVALID`` on a strict path, ``pre-read`` does NOT re-grant SHARED, so
        ``read()`` returns the fresh bytes yet the instance stays ``INVALID``
        and a subsequent :meth:`write` is still denied. Recovery is
        :meth:`reacquire`, not a bare ``read()``.

        Raises ``FileNotFoundError`` for a missing file (no phantom artifact is
        seeded). Under ``on_error="strict"`` a coordinator-infrastructure
        failure (unavailable coordinator, watchdog timeout, a coordinator-side
        error answering the pre-read) raises ``CoherenceError`` — a read whose
        coherence cannot be registered fails closed.

        **Not thread-safe (A5).** Overlapping use of this instance from another
        thread raises ``CoherenceError`` (the single-op guard) — one instance per
        thread.
        """
        with self._single_op_guard():
            return self._read_impl(path)

    def _read_impl(
        self, path: str | os.PathLike[str], *, _enforce_stale: bool = True
    ) -> bytes:
        self._ensure_attached()
        abs_path, rel = self._to_relative(path)
        # Stat before registering so a missing file raises rather than seeding a
        # phantom artifact in the coordinator registry.
        if not abs_path.is_file():
            raise FileNotFoundError(f"no such file in workspace: {rel}")
        data = self._read_file_bytes(abs_path)  # empty file -> b"" -> sha256(b"")
        content_hash = self._sha256_bytes(data)
        # SB-23: a path this volume has no baseline for takes these bytes as its
        # first one even if the read is refused below. Without it a refused first
        # read leaves a later write unchecked, and a peer commit that reached disk
        # after the refusal is overwritten. setdefault never replaces an earlier
        # observation, so the refused bytes cannot absolve an edit made after one.
        self._last_observed_hash.setdefault(rel, content_hash)
        if self._endpoint is not None:
            resp = self._post(
                "/hooks/pre-read",
                {
                    "session_id": self._session_id,
                    "path": rel,
                    "content_hash": content_hash,
                },
            )
            # A stale / strict-deny response is expected and changes nothing here
            # (read returns current bytes; INVALID stays sticky). Two answers
            # are infra failures that take the unanswered-request seam: a
            # watchdog timeout, and the coordinator's failure envelope.
            if isinstance(resp, dict) and resp.get("degraded"):
                self._fail_closed_or_degrade(
                    f"coordinator watchdog timeout during read of {rel}"
                )
            elif isinstance(resp, dict) and resp.get("ok") is False:
                self._fail_closed_or_degrade(self._pre_read_failure(resp, rel))
            elif _enforce_stale and self._on_stale_read == "raise":
                # PH-A read-surface instance (opt-in): surface a strict
                # foreign-edit / stale-view deny as StaleView so the caller can
                # abort or reacquire(), instead of silently returning current
                # bytes. Default ("allow") keeps the back-compat swallow.
                # reacquire()'s recovery read passes _enforce_stale=False so
                # recovery is never blocked.
                hook_output = (
                    resp.get("hookSpecificOutput") if isinstance(resp, dict) else None
                )
                if (
                    isinstance(hook_output, dict)
                    and hook_output.get("permissionDecision") == "deny"
                ):
                    raise StaleView(self._deny_reason(resp))
        # SB-23: advance the foreign-edit baseline only HERE, where the bytes
        # reach the caller. Every raise above leaves the caller without them: a
        # request refused for its caller principal (raised out of _post in both
        # on_error modes), a strict-mode transport or watchdog failure, a
        # StaleView. Advancing on any of those would absolve an out-of-band edit
        # the caller never saw, and its next write() would clobber it instead of
        # being denied. (A path with no baseline yet was already given one by the
        # setdefault above.)
        self._last_observed_hash[rel] = content_hash
        return data

    def write(self, path: str | os.PathLike[str], data: bytes | bytearray) -> None:
        """Write bytes to a workspace file under the single-writer guard.

        Prevents the **sequential** stale-read→write lost update: if a peer
        committed a newer version since this instance last read, this instance
        is ``INVALID`` and the coordinator denies the write (``pre-edit``). The
        deny is surfaced as ``StaleView`` (pre-edit) or ``CommitPreempted``
        (post-edit preempt) — both ``CoherenceError`` subclasses — carrying the
        coordinator's byte-stable reason VERBATIM. **A deny always raises, in both
        ``on_error`` modes** — the deny is enforcement working, not an
        infrastructure failure; recover via :meth:`reacquire` and write from the
        fresh bytes.

        Also prevents the **foreign-edit clobber** (content-CAS, write surface):
        on a managed (strict) path, if the file on disk changed out-of-band since
        this instance last read/wrote it, the write is denied with
        :class:`~ccs.core.exceptions.StaleView` rather than silently overwriting
        it. Governed by ``on_stale_write`` (default ``"raise"`` — a write is a
        data-loss surface, so the safe default GUARDS, the inverse of
        ``on_stale_read``; ``"allow"`` opts out and restores the prior clobber).
        Fires adapter-locally, independent of ``on_error``. Recover via
        :meth:`reacquire`. (Catch as ``ccs.core.exceptions.StaleView``.) A caller
        that loops ``reacquire`` + ``write`` against a fast out-of-band editor must
        bound its OWN retries — there is no built-in cap here (unlike
        :meth:`write_cas`'s ``MAX_CAS_REACQUIRES``).

        Does NOT prevent concurrent racing writers (this is
        single-writer-by-invalidation, not a mutex) nor a caller that ignores
        :meth:`reacquire`'s fresh bytes. ``on_error`` governs only
        coherence-infrastructure failures (unavailable coordinator, watchdog
        timeout): strict raises, degrade warns once and writes best-effort.

        **Not thread-safe (A5).** Overlapping use of this instance from another
        thread raises ``CoherenceError`` (the single-op guard) — one instance per
        thread.
        """
        with self._single_op_guard():
            self._write_impl(path, data)

    def _write_impl(self, path: str | os.PathLike[str], data: bytes | bytearray) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError("CoherentVolume.write expects bytes")
        data = bytes(data)
        self._ensure_attached()
        abs_path, rel = self._to_relative(path)

        if self._endpoint is None:
            # Unattached / degraded coordinator (strict already raised at
            # construction). Degrade mode: best-effort write, no coordinator
            # enforcement — but the SB-23 content-CAS is adapter-local, so it still
            # fires (gated by on_stale_write, independent of on_error).
            self._check_foreign_edit(rel, self._disk_hash(abs_path))
            self._atomic_write(abs_path, data)
            self._record_own_write(rel, self._sha256_bytes(data))
            return

        # Record the grant BEFORE the acquire is sent: an acquire whose answer is
        # lost, a degraded answer, and every failure after the grant (the disk
        # write, the post-edit POST) can each leave this incarnation holding
        # EXCLUSIVE with write() raising. The re-mint that abandons the incarnation
        # releases it (see _remint). Only a stale deny or a principal refusal
        # proves no grant was taken: an ok:false with an "internal:" reason can
        # follow an acquire that landed. Either withdraws only what this call
        # recorded — the same incarnation may still hold a grant from an earlier
        # write() on another path. A file the coordinator does not track stays
        # recorded too: its answer is the same bare ok:true as an acquire, so the
        # volume cannot tell that no grant was taken.
        written = self._grant_incarnations.setdefault(self._incarnation, set())
        recorded_before = rel in written
        written.add(rel)

        def withdraw_record() -> None:
            if recorded_before:
                return
            written.discard(rel)
            if not written:
                self._grant_incarnations.pop(self._incarnation, None)

        # pre-edit: acquire EXCLUSIVE, or be denied because we are INVALID. A
        # principal refusal proves no grant was taken too — the coordinator
        # refuses before any mutation — so it withdraws the record like a deny.
        try:
            resp = self._post("/hooks/pre-edit", {"session_id": self._session_id, "path": rel})
        except CallerPrincipalRefused:
            withdraw_record()
            raise
        if isinstance(resp, dict) and resp.get("ok") is False and resp.get("status") == "stale":
            withdraw_record()
        if resp is not None:
            self._check_grant(resp, rel, phase="pre-edit")
        # Past the deny check, a dict answer that is not watchdog-degraded is
        # an admitted acquire; a degraded or lost one (degrade mode only) may
        # or may not have taken the grant.
        grant_confirmed = isinstance(resp, dict) and not resp.get("degraded")

        # EXCLUSIVE grant is now held. ANY failure before the post-edit commit
        # must release it via the coordinator's tool-failure path (success:false),
        # not only an OSError from the atomic write: an unexpected error while
        # hashing the data or the on-disk file (e.g. MemoryError on a very large
        # file) would otherwise orphan the grant until the crash-recovery sweep.
        try:
            new_hash = self._sha256_bytes(data)
            # SB-23 content-CAS: re-hash disk ONCE under the held grant, deny a write
            # whose target diverged on disk since this instance last observed it (a
            # foreign edit), then reuse the same disk hash for the no-op-skip below
            # (hoisted out of the former short-circuit — one disk read, not two).
            disk_hash = self._disk_hash(abs_path)
            self._check_foreign_edit(rel, disk_hash)
            # No-op skip: skip the os.replace (and its fsync) only when the file
            # ALREADY holds these bytes. _last_committed_hash is a cheap fast-path
            # gate; the CURRENT on-disk hash is the authority (see _disk_hash).
            already_on_disk = (
                self._last_committed_hash.get(rel) == new_hash
                and disk_hash == new_hash
            )
            if not already_on_disk:
                self._atomic_write(abs_path, data)
            # SB-23: advance the observed baseline as soon as the bytes are on disk
            # — BEFORE the post-edit POST — so a post-edit preempt cannot leave a
            # stale baseline that self-false-denies this instance's next write. This
            # is DELIBERATELY split from the _last_committed_hash advance below (after
            # the post-edit POST); the other own-write sites advance both caches
            # together, so do not "fix" this into a pair — the ordering is load-bearing.
            self._last_observed_hash[rel] = new_hash
        except Exception:
            # Release the grant so it is not orphaned until the sweep, then
            # re-raise the original error. Best-effort release — a release failure
            # must not mask it. It names the incarnation that holds the grant: a
            # release without one addresses the session's parent row, which holds
            # nothing, and releases nothing.
            with contextlib.suppress(Exception):
                _coordinator_post(
                    self._endpoint,
                    "/hooks/post-edit",
                    {
                        "session_id": self._session_id,
                        "agent_id": self._incarnation,
                        "path": rel,
                        "success": False,
                    },
                    extra_headers=caller_principal_headers(self._principal),
                )
            raise

        redirect_status: int | None = None
        try:
            post_resp = self._post(
                "/hooks/post-edit",
                {
                    "session_id": self._session_id,
                    "path": rel,
                    "success": True,
                    "content_hash": new_hash,
                },
            )
        except RedirectRefused as redirect:
            # A redirected commit is a failed commit like any other status
            # outside 2xx — refused, never followed — so, the bytes being on
            # disk, it takes on_error's path below, as a 5xx or a lost answer
            # does, by its status alone, and not the typed refusal of a request
            # that changed nothing.
            post_resp, redirect_status = None, redirect.status
        except CallerPrincipalRefused as refusal:
            # Refused AFTER the grant request: the refusal itself changed
            # nothing at the coordinator, but this write() had already put its
            # bytes on disk (unless they were there) and — on a tracked path —
            # invalidated the peers through its admitted grant request, so the
            # error says so rather than reading like a request that had no
            # effect. The grant stays recorded (nothing released it for this
            # incarnation); the observed baseline already names the bytes on
            # disk, which are this instance's own.
            state = _unrecorded_write_state(
                rel, wrote=not already_on_disk, grant_confirmed=grant_confirmed
            )
            # The re-raise keeps what the refusal typed: its reason, and
            # whether it is settled (a lost recovery answer is not).
            raise CallerPrincipalRefused(
                refusal.reason, f"{state} {refusal}", settled=refusal.settled
            ) from None
        if redirect_status is not None:
            self._fail_closed_or_degrade(
                f"coordinator request to /hooks/post-edit failed: HTTP {redirect_status}, "
                "a redirect, which this client never follows"
            )
        if post_resp is not None:
            # ok:false here means the grant was preempted / sweep-reclaimed mid-
            # write (a concurrent-writer case v1 does not claim to serialize);
            # fail closed so the caller knows the commit did not register.
            self._check_grant(post_resp, rel, phase="post-edit")
        self._last_committed_hash[rel] = new_hash

    def reacquire(self, path: str | os.PathLike[str]) -> bytes:
        """Recover from a sticky strict deny and return the current bytes.

        The strict deny is sticky: once ``INVALID``, a bare :meth:`read` does
        NOT clear it (KTD-T). Recovery requires a fresh coordinator identity AND
        a fresh read under that identity, atomically: the new identity carries
        no ``INVALID`` state, and the mandatory read registers it as
        SHARED@current. The fresh identity is a new per-attempt incarnation
        under the same :attr:`session_id`, which does not change.

        **The forced read is non-optional.** Re-minting identity *without*
        reading would merely rename the stale-buffer hole — the next write would
        be granted while the caller still holds pre-reacquire bytes. The caller
        MUST write from the bytes this method returns (or a later :meth:`read`),
        never from a buffer computed before ``reacquire()``; v1 cannot catch a
        caller that ignores them (the one fundamental, OCC-proof boundary).

        Re-minting the identity resets this instance's view of *every* path, not
        just ``path`` — other tracked paths should be re-read before writing.

        Always returns the current bytes without raising, even under
        ``on_stale_read="raise"`` — recovery must never be blocked by the deny
        that triggered it.
        """
        # The re-mint runs under the guard with the read: it can send a request
        # (the release of an abandoned grant), and every request this instance
        # sends is one operation's.
        with self._single_op_guard():
            self._remint()  # fresh identity -> no INVALID; has committed nothing
            # MANDATORY fresh read under the new identity -> SHARED@current.
            # Recovery bypasses on_stale_read='raise' (_enforce_stale=False) so a
            # reacquire after a foreign edit returns the current bytes rather than
            # re-raising and blocking recovery.
            return self._read_impl(path, _enforce_stale=False)

    def _remint(self) -> None:
        """Re-mint coordinator identity (shed ``INVALID`` + any invalidation
        transient) WITHOUT a read — the OCC :meth:`write_cas` retry primitive.

        Distinct from :meth:`reacquire` (re-mint **and** a read): reacquire's
        read registers the fresh identity as ``SHARED``, after which the
        ``write_cas`` loop's next :meth:`_read_with_version` hits the
        coordinator's *fresh-SHARED* pre-read branch, which returns the version
        WITHOUT re-checking the disk bytes' hash. That pairs a possibly-stale
        on-disk read with a fresh version — and because the commit-CAS checks
        only the *version*, a ``make_content()`` over the stale bytes would WIN
        and silently drop a peer's update (an on-disk lost update; the
        protocol-level ``NoLostUpdate`` still holds — the version-bump count is
        right — but the persisted value is wrong). Re-minting WITHOUT a read
        keeps the loop's next ``_read_with_version`` a **None-state, HASH-CHECKED**
        read (warn on hash match, deny on mismatch), so its ``(bytes, version)``
        comparand pair is always validated.

        What rotates is the per-attempt incarnation, not :attr:`session_id`: the
        coordinator keys a grant row on ``session_to_agent_id(session_id,
        incarnation)``, so a new incarnation is a new row with no ``INVALID``,
        no invalidation transient and no read generation — which is all the
        shedding above needs.

        The abandoned incarnation is a different agent to the coordinator, so a
        grant it still holds is FOREIGN to this volume: an EXCLUSIVE/MODIFIED
        grant left by :meth:`write` would refuse this volume's next optimistic
        commit as ``other_holder`` with equal versions, and every retry re-mints
        into the same refusal. So every incarnation :meth:`write` recorded is
        released here, one ``session-stop`` naming it, and leaves the record only
        on a confirmed release. An incarnation that did only reads and
        optimistic commits holds at most SHARED/INVALID, blocks nothing, and is
        never released — no request is spent on it.
        """
        with self._lock:
            self._incarnation = self._new_incarnation()  # fresh row -> no INVALID
            self._last_committed_hash.clear()  # the new identity has committed nothing
            abandoned = sorted(self._grant_incarnations)
            # SB-23: do NOT clear _last_observed_hash here. The observed-disk
            # baselines are DISK-scoped (what disk looked like when this instance
            # last read/wrote a path), not identity-scoped — a re-mint changes the
            # coordinator identity but not the disk. Clearing them would blind the
            # foreign-edit guard on every OTHER path after a write_cas conflict
            # remint (the CAS path is re-seeded once the loop's next clean read
            # reaches make_content anyway). _after_fork still clears them — a
            # forked child is a new process that must re-establish its own view.
        # The release runs OUTSIDE self._lock. That lock is a plain (non-
        # reentrant) threading.Lock guarding identity mutation, and a failed
        # request in degrade mode re-enters it (_post -> _fail_closed_or_degrade
        # -> _record_degraded takes self._lock), so holding it here would
        # deadlock on the first failed release.
        self._release_abandoned_grants(abandoned)

    def _release_abandoned_grants(self, incarnations: list[str]) -> None:
        """Release every grant each abandoned incarnation holds: one
        ``session-stop`` naming it in ``agent_id``, which releases all of that
        row's EXCLUSIVE/MODIFIED grants (every path it wrote) and nothing else.

        An incarnation leaves the record only on a confirmed release — ``ok:
        true`` without ``degraded`` (a watchdog-degraded stop answers ``ok:
        true`` whether or not the release ran). Anything else keeps it, so the
        next re-mint retries it, and the failure routes through ``on_error``
        like every other coordinator request: strict raises, degrade warns once.
        The pass stops at the first failure — the rest would meet the same
        coordinator — so one re-mint spends at most one failed request.
        """
        if self._endpoint is None:
            return  # nothing to send it to; the record waits for an attached re-mint
        for incarnation in incarnations:
            resp = self._post(
                "/hooks/session-stop",
                {"session_id": self._session_id, "agent_id": incarnation},
            )
            if isinstance(resp, dict) and resp.get("degraded"):
                self._fail_closed_or_degrade(
                    "coordinator watchdog timeout releasing an abandoned write grant"
                )
                return
            if not (isinstance(resp, dict) and resp.get("ok") is True):
                # None follows a degrade-mode transport failure _post already
                # warned about; any other answer is not a shape session-stop has.
                if resp is not None:
                    logger.warning(
                        "release of an abandoned write grant was not confirmed; "
                        "it will be retried at the next re-mint: %r",
                        resp,
                    )
                return
            self._grant_incarnations.pop(incarnation, None)

    def write_cas(
        self,
        path: str | os.PathLike[str],
        make_content: Callable[[bytes], bytes | bytearray],
    ) -> None:
        """Optimistically commit a write that BYPASSES the EXCLUSIVE acquire.

        The OCC counterpart to :meth:`write` (plan Unit 6). Unlike ``write`` —
        which takes EXCLUSIVE via ``pre-edit`` before committing — ``write_cas``
        **never acquires EXCLUSIVE**: it reads (→ SHARED), derives the bytes via
        ``make_content(current_bytes)``, and commits through the coordinator's
        ``/hooks/post-edit-cas`` version-checked CAS. The winner is elected by
        the coordinator's serialized commit, so two concurrent OCC writers
        cannot both land the same version — the loser is told ``version_mismatch``
        and retries. The pessimistic ``write()`` path is untouched.

        Conflict recovery: on a ``version_mismatch`` / ``other_holder`` conflict
        (or a stale-read deny) the loop re-mints identity (:meth:`_remint` —
        sheds ``INVALID``, but does NOT do reacquire's read; that read would route
        the next comparand read through the coordinator's unchecked *fresh-SHARED*
        branch and re-open the lost update), then on the next attempt does ONE
        hash-checked fresh read whose ``(bytes, version)`` pair is validated,
        re-derives the bytes from that view via ``make_content``, and retries —
        bounded by :data:`MAX_CAS_REACQUIRES`. On exhaustion it raises
        :class:`~ccs.core.exceptions.CasRetriesExhausted` (a typed terminal,
        NEVER a silent drop), whose ``last_conflict_reason`` names the refusal
        that exhausted the budget. The first attempt also re-mints when this
        volume's own ``write()`` may still hold ``path``, and only then. Any re-mint,
        like :meth:`reacquire`'s, resets this instance's view of every path: a
        refusal a peer's commit left on another path is gone, so re-read other
        tracked paths before a plain ``write()`` of them.

        **Contention bound.** The commit budget is :data:`MAX_CAS_REACQUIRES`
        (=8 → 9 CAS attempts); each lost race (a peer winning the version)
        costs one. So does each ``other_holder`` refusal, which is not a race:
        another agent still holds the grant at an unchanged version. A
        pessimistic ``write()`` keeps that grant until the coordinator reclaims it
        (after ``grant_heartbeat_timeout_sec`` without a coordinator call, 600 s
        by default, or ``grant_max_hold_sec`` of holding, 1800 s by default) or
        the holder releases it, which a ``CoherentVolume`` holder does only at
        its own next re-mint: a ``write_cas`` of that file or any ``write_cas``
        retry, ``write_cas_at``, ``atomic_publish`` or :meth:`reacquire` (the
        re-mint releases what its ``write()`` left held; see :meth:`_remint`). A
        ``session-stop`` naming only its :attr:`session_id` releases nothing,
        because each attempt holds its grants under its own incarnation. A
        peer's reads and CAS attempts never release it, and a peer's ``write()``
        only takes it over.
        So the loop does not wait between those attempts:
        no wait that fits in one call could outlast the grant, it would only make
        the terminal slower. A caller that gets ``last_conflict_reason ==
        "other_holder"`` retries after the holder releases, not in a loop.
        Under very high single-host contention — more concurrent
        writers racing the SAME key than the budget — a writer can exhaust it
        and raise :class:`~ccs.core.exceptions.CasRetriesExhausted`. A
        stale-denied comparand read (the transient window where a peer's commit
        landed but its disk write hasn't) does NOT consume the commit budget;
        instead CONSECUTIVE denied reads are separately bounded (also at
        ``MAX_CAS_REACQUIRES + 1``) and raise ``ViewWedged`` (a ``CoherenceError``
        subclass) if the view never clears. That poll WAITS between re-reads
        (capped-exponential, :data:`DENIED_READ_BACKOFF_BASE_SEC` →
        :data:`DENIED_READ_BACKOFF_CAP_SEC`) rather than spinning: the transient
        clears on a PEER's progress, so the waiting writer must yield the CPU to
        it instead of racing it — an undelayed poll made this bound a proxy for
        scheduler luck (a measured ~37ms of wall clock) rather than for a wedged
        view. That schedule bounds ONE streak (212ms); a clean read resets the
        streak, so a call that alternates streaks with lost races can traverse up
        to ``MAX_CAS_REACQUIRES + 1`` of them (~1.9s in aggregate).
        Both terminals are the honest fail-closed outcome, **never** a silent
        lost update: the invariant this method guarantees is
        *final == start + every applied delta, OR a typed raise* — a successful
        return always means the update landed.

        ``make_content`` is invoked once per attempt with the freshly-read
        current bytes and returns the bytes to commit — re-deriving the caller's
        intent against the latest state is what makes the retry an *update*
        rather than a stale overwrite. A caller that ignores its ``bytes``
        argument and returns a buffer computed from older bytes defeats the
        guard (the one fundamental OCC-proof boundary; same as :meth:`reacquire`).

        **A deny ALWAYS raises, in both ``on_error`` modes** — including the
        coordinator's fail-closed degrade body
        (``{ok: false, degraded: true, reason: "commit_unconfirmed"}``): a
        degraded CAS is unconfirmed, so it must read as failure (the client must
        never assume the write landed). The SAME fail-closed rule covers a
        mid-commit transport failure that ``degrade`` mode swallowed (``_post``
        returned ``None`` after a version was read): an unconfirmed CAS must
        never write its bytes to disk, so it raises in both modes rather than
        best-effort writing (unconfirmed bytes on disk would re-open the
        lost-update). ``on_error`` still governs whether an infra failure raises
        vs. warns *inside* ``_post`` and the genuinely-unattached
        (``_endpoint is None``) degrade branch above — but never lets an
        unconfirmed OCC commit silently land.

        **Not thread-safe (A5).** Overlapping use of this instance from another
        thread raises ``CoherenceError`` (the single-op guard). The internal
        re-mint + re-read retries run on the SAME thread under the guard already
        held by this call — one instance per thread.
        """
        with self._single_op_guard():
            self._write_cas_impl(path, make_content)

    def _write_cas_impl(
        self,
        path: str | os.PathLike[str],
        make_content: Callable[[bytes], bytes | bytearray],
    ) -> None:
        self._ensure_attached()
        _abs_path, rel = self._to_relative(path)

        if self._endpoint is None:
            # Unattached / degraded coordinator (strict already raised at
            # construction). Degrade mode: best-effort write, no enforcement —
            # there is no version to CAS against, so derive from current bytes.
            current = self._current_bytes_or_empty(_abs_path)
            data = bytes(make_content(current))
            self._atomic_write(_abs_path, data)
            self._record_own_write(rel, self._sha256_bytes(data))
            return

        if any(rel in paths for paths in self._grant_incarnations.values()):
            # This volume's write() may still hold THIS file: under the current
            # incarnation, or under an abandoned one whose release was not
            # confirmed. The loop's comparand read must run as a None-state
            # identity (see below), and commit_cas refuses an E/M caller on the
            # file it commits, so the current incarnation's grant would refuse
            # the first attempt outright and an abandoned one's would refuse it
            # as other_holder, with no peer anywhere. Rotate first; the re-mint
            # releases every recorded incarnation. Only this file warrants it: a
            # grant on another file refuses nothing here, and a rotation drops the
            # refusal a peer's commit left on every other file, which is what
            # stops a later write() of one of them from stale bytes. An
            # optimistic-only write_cas still re-mints only on retry.
            self._remint()

        last_current_version = -1
        max_attempts = MAX_CAS_REACQUIRES + 1  # commit (CAS POST) budget
        cas_attempts = 0
        denied_streak = 0  # CONSECUTIVE stale-denied comparand reads
        while True:
            # Fresh read each attempt. _read_with_version returns the bytes, the
            # coordinator's authoritative version (the OCC comparand), and whether
            # the read was a strict-deny (stale_denied — this instance is INVALID,
            # or a re-minted identity whose disk read does not match the recorded
            # content). The comparand read ALWAYS runs under a None-state identity
            # (the first read's, or a _remint()ed one), so the coordinator
            # HASH-CHECKS it: warn (re-grant SHARED) on a hash match, deny on a
            # mismatch. That is exactly what makes (current_bytes,
            # expected_version) a VALIDATED pair — current_bytes is the content
            # the coordinator records at expected_version, so make_content()
            # derives the successor from the right state. (Re-minting WITHOUT a
            # read is the point: reacquire()'s read would leave the identity
            # SHARED and route the next comparand read through the coordinator's
            # fresh-SHARED branch, which returns the version WITHOUT a hash
            # check — see _remint() / KTD-LU.)
            result = self._read_with_version(rel, advance_baseline=False)
            current_bytes = result.data
            expected_version = result.version
            stale_denied = result.stale_denied
            if stale_denied:
                # Cannot CAS from this view: INVALID, or the disk lags a just-
                # committed version whose peer write has not landed yet
                # (hash_differs under strict). Re-mint identity (sheds INVALID +
                # the invalidation transient) and RE-READ — once the peer's disk
                # write lands, the hash-checked None-state read yields a validated
                # comparand pair. A denied read never POSTs a commit, so it does
                # NOT consume the CAS budget (max_attempts counts *commit
                # attempts*, keeping MAX_CAS_REACQUIRES' documented semantic);
                # instead the CONSECUTIVE-denied streak is bounded so a
                # never-clearing view (wedged coordinator, perpetually lagging
                # disk, a starving foreign writer) still fails closed instead of
                # spinning. A clean read resets the streak.
                last_current_version = max(last_current_version, expected_version)
                denied_streak += 1
                if denied_streak > MAX_CAS_REACQUIRES:
                    raise ViewWedged(
                        f"OCC comparand read of {rel} stayed strict-denied across "
                        f"{denied_streak} consecutive reads under re-minted "
                        "identities; cannot establish a clean (bytes, version) "
                        "view to CAS from — the on-disk content may be lagging "
                        "peer commits or the coordinator may be wedged. No write "
                        "landed (fail-closed)."
                    )
                self._remint()
                # WAIT, don't spin: yield the CPU to the peer whose disk write
                # clears this view (see DENIED_READ_BACKOFF_BASE_SEC).
                time.sleep(denied_read_backoff_sec(denied_streak))
                continue
            denied_streak = 0

            # SB-23: a clean read's bytes reach the caller HERE, through
            # make_content, so they become the foreign-edit baseline — and only
            # here: a denied read (above) was never handed to it, and seeding
            # from one would absolve the out-of-band edit that got it denied.
            self._last_observed_hash[rel] = self._sha256_bytes(current_bytes)
            data = bytes(make_content(current_bytes))
            new_hash = self._sha256_bytes(data)

            # CAS FIRST, write to disk only on WIN. An OCC writer is S/I — it
            # holds no EXCLUSIVE grant, so unconfirmed bytes must NEVER touch
            # disk (a denied/degraded CAS landing on disk would be the very
            # lost-update this guards). Contrast the pessimistic write(), which
            # writes between pre-edit (EXCLUSIVE held) and post-edit.
            cas_attempts += 1
            resp = self._post(
                "/hooks/post-edit-cas",
                {
                    "session_id": self._session_id,
                    "path": rel,
                    "success": True,
                    "content_hash": new_hash,
                    "expected_version": expected_version,
                },
            )
            if resp is None:
                # Degrade mode swallowed a mid-operation transport/infra failure
                # in _post and returned None — the CAS was NOT confirmed. An OCC
                # writer holds NO grant (S/I), so unconfirmed bytes must NEVER
                # touch disk: writing them would re-open the very lost-update this
                # guards. Fail closed in BOTH on_error modes — identical to how
                # the {ok:false, degraded:true, commit_unconfirmed} degrade BODY
                # is handled below ("raise"). on_error governs only whether the
                # infra failure already warned (degrade) or raised (strict) inside
                # _post; either way an unconfirmed CAS must read as failure.
                raise CommitUnconfirmed(
                    f"OCC commit of {rel} could not be confirmed (coordinator "
                    "transport failed mid-commit); the write did not land. "
                    "reacquire() and retry from the fresh bytes."
                )

            outcome = self._classify_cas_response(resp)
            if outcome == "win":
                # SB-23 scope (v1): write_cas is VERSION-CAS, not content-CAS, and
                # does NOT raise StaleView. It is not unguarded, though: on a strict
                # path the read-side hash deny makes the comparand read
                # (_read_with_version) fail closed on a foreign edit, so write_cas
                # WEDGES (ViewWedged) rather than clobbering. SB-23's content-CAS
                # guards only plain write(). _record_own_write still advances the
                # observed baseline so a LATER plain write() on this path is consistent.
                self._atomic_write(_abs_path, data)  # confirmed → persist
                self._record_own_write(rel, new_hash)
                return
            if outcome == "conflict":
                last_current_version = self._cas_current_version(resp, last_current_version)
                if cas_attempts >= max_attempts:
                    # Every allowed commit attempt was refused — typed terminal,
                    # never a silent drop. (last_current_version is the latest
                    # version the loser observed.) The last refusal's reason
                    # tells a lost race (version_mismatch) from a grant another
                    # agent holds (other_holder, or caller_in_transient_state
                    # when that agent's write() landed mid-attempt).
                    raise CasRetriesExhausted(
                        artifact_id=rel,
                        attempts=cas_attempts,
                        last_current_version=last_current_version,
                        last_conflict_reason=resp.get("reason"),
                    )
                # Re-mint (NOT reacquire) before the next attempt so the next
                # comparand read is a hash-checked None-state read that sees the
                # winner's state — reacquire()'s read would route it through the
                # unchecked fresh-SHARED branch and could pair stale bytes with a
                # fresh version (the on-disk lost update; see _remint() / KTD-LU).
                self._remint()
                continue
            # outcome == "raise": a deny, corruption, or the fail-closed
            # commit_unconfirmed degrade body — ALWAYS raises in both modes.
            # Nothing was written to disk (the CAS did not win). Type the terminal
            # so the MCP deny mapper classifies by exception type, not a substring
            # of the (verbatim) coordinator prose:
            #  - commit_unconfirmed degrade body → CommitUnconfirmed (re-read, retry
            #    only if absent — the false-negative-ack window);
            #  - a permission-style deny → StaleView (recoverable by reacquire);
            #  - anything else (corruption: commit_cas_corruption / expected>current)
            #    → plain CoherenceError → the mapper fails it closed as internal_error;
            #  - no string reason at all → the outcome is unknown → CommitUnconfirmed.
            if resp.get("reason") == COMMIT_UNCONFIRMED_REASON:
                raise CommitUnconfirmed(self._deny_reason(resp))
            if not isinstance(resp.get("reason"), str):
                raise CommitUnconfirmed(_unclassifiable_cas_message(rel))
            hook_output = resp.get("hookSpecificOutput")
            if isinstance(hook_output, dict) and hook_output.get("permissionDecisionReason"):
                raise StaleView(self._deny_reason(resp))
            raise CoherenceError(self._deny_reason(resp))

    def write_cas_at(
        self,
        path: str | os.PathLike[str],
        expected_version: int,
        new_content: bytes | bytearray,
    ) -> None:
        """Single-shot, version-checked CAS (Option A — the MCP ``swg_write_cas``).

        Commit ``new_content`` IFF the coordinator's current version ==
        ``expected_version``. Unlike :meth:`write_cas` (which auto-retries by
        re-deriving from the latest bytes), this does NOT retry and does NOT
        re-derive: a stale ``expected_version`` raises
        :class:`~ccs.core.exceptions.CasVersionConflict` carrying the current
        version — the AGENT re-reads, re-merges, and retries
        (typed-conflict, not auto-merge). This is the correct primitive for an
        agent that already merged against a specific version it read: it never
        commits content derived from version V over a peer's later V+1 (the
        split-comparand lost update; ``coherent-volume-write-cas-split-comparand``).

        The CAS commits against the AGENT's ``expected_version``, NOT a re-read
        one, so a peer winning the version between the comparand read and the
        commit is rejected as a conflict — never a silent overwrite. ``new_content``
        touches disk ONLY on a confirmed win.
        """
        if not isinstance(new_content, (bytes, bytearray)):
            raise TypeError(f"new_content must be bytes, got {type(new_content).__name__}")
        with self._single_op_guard():
            self._write_cas_at_impl(path, int(expected_version), bytes(new_content))

    def _write_cas_at_impl(
        self, path: str | os.PathLike[str], expected_version: int, new_content: bytes
    ) -> int:
        self._ensure_attached()
        abs_path, rel = self._to_relative(path)
        if self._endpoint is None:
            # Strict-only construction prevents this for the server; fail closed
            # rather than best-effort writing an unversioned blob.
            raise CoherenceError(
                f"cannot CAS {rel}: coordinator endpoint unavailable (fail-closed)"
            )
        # Re-mint first / read second: a hash-checked None-state read establishes
        # a VALIDATED (bytes, version) comparand under a fresh identity (do NOT
        # re-create the split-comparand hole — KTD-LU). The read's bytes are
        # discarded (the caller already holds new_content), so they must not
        # ADVANCE the foreign-edit baseline: that let a caller whose CAS lost
        # fall back to write() with older content and land it over the peer's
        # commit. It stays an observing read otherwise, and both halves matter:
        # on a path never read it records the first baseline before its request
        # (so a lost, refused or failed CAS still leaves write() one to check),
        # and it registers the fresh identity SHARED, so a peer commit that wins
        # before our CAS invalidates it and the write() fallback is refused at
        # the coordinator even while the peer's bytes are still reaching disk,
        # when the disk alone still matches the baseline. A win records its own
        # bytes below.
        self._remint()
        result = self._read_with_version(rel, advance_baseline=False)
        current_version = result.version
        stale_denied = result.stale_denied
        if stale_denied:
            # The comparand view is INVALID / the disk lags a just-landed commit;
            # the agent must reacquire / re-read before it can CAS.
            raise ViewWedged(
                f"OCC comparand read of {rel} is strict-denied; cannot establish a "
                "clean (bytes, version) view to CAS from. reacquire() / re-read first."
            )
        if current_version != expected_version:
            # Stale expected_version → typed conflict; NO write, NO server retry.
            raise CasVersionConflict(rel, expected_version, current_version)
        new_hash = self._sha256_bytes(new_content)
        resp = self._post(
            "/hooks/post-edit-cas",
            {
                "session_id": self._session_id,
                "path": rel,
                "success": True,
                "content_hash": new_hash,
                "expected_version": expected_version,
            },
        )
        if resp is None:
            raise CommitUnconfirmed(
                f"OCC commit of {rel} could not be confirmed (coordinator transport "
                "failed mid-commit); the write did not land. re-read and retry."
            )
        outcome = self._classify_cas_response(resp)
        if outcome == "win":
            self._atomic_write(abs_path, new_content)  # confirmed → persist
            self._record_own_write(rel, new_hash)
            # The version-CAS committed against `expected_version`, so the new
            # version is deterministically expected+1 (atomic_publish surfaces it).
            return expected_version + 1
        if outcome == "conflict":
            # A peer won the version between our read and the CAS, OR a
            # pessimistic peer holds the grant, OR the claim this read was taken
            # under was reclaimed, OR a peer invalidated us mid-window. All four
            # are conflicts; they are NOT the same conflict, so the coordinator's
            # own reason travels with the terminal rather than being relabelled.
            raise CasVersionConflict(
                rel,
                expected_version,
                self._cas_current_version(resp, current_version),
                reason=resp.get("reason"),
            )
        # outcome == "raise": corruption or the commit_unconfirmed degrade body,
        # or an answer with no string reason, whose outcome is unknown.
        if resp.get("reason") == COMMIT_UNCONFIRMED_REASON:
            raise CommitUnconfirmed(self._deny_reason(resp))
        if not isinstance(resp.get("reason"), str):
            raise CommitUnconfirmed(_unclassifiable_cas_message(rel))
        raise CoherenceError(self._deny_reason(resp))

    def atomic_publish(
        self,
        writes: Sequence[tuple[str | os.PathLike[str], int, bytes | bytearray | str]],
    ) -> dict[str, int]:
        """Publish a SET of artifacts ALL-OR-NOTHING (SB-18 batch OCC).

        Commit every ``(path, expected_version, content)`` as ONE unit IFF every
        member is still at its ``expected_version``; otherwise NOTHING commits and
        NO file is written. The multi-artifact generalization of
        :meth:`write_cas_at` — the primitive for an agent that merged a coherent
        SET of files against specific versions and must land them together (a plan
        + its manifest; a config split across files) with no torn intermediate
        state ever observable. Returns ``{path: new_version}`` for every member on
        success.

        Outcomes:
          - every member at its ``expected_version`` → the batch COMMITS at the
            coordinator as one unit, then every file is materialized to disk; new
            versions returned.
          - a member is already behind at capture →
            :class:`~ccs.core.exceptions.CasVersionConflict` (the caller's view
            was stale before the publish began); NOTHING committed, NO file written.
          - a peer commits a member in the capture→commit window →
            :class:`~ccs.core.exceptions.StaleView` (HELD; recover via
            :meth:`reacquire` + re-decide + retry); NOTHING committed, NO file written.
          - the commit is answered as unconfirmed, or with no outcome this
            client can classify →
            :class:`~ccs.core.exceptions.CommitUnconfirmed` in both ``on_error``
            modes; whether the batch committed is unknown, NO file written —
            re-read, and retry only if the publish is absent.

        **Atomicity boundary (read this).** The all-or-nothing guarantee is at the
        COORDINATOR commit: either every member's version advances as one unit or
        none does — a torn *commit* is never reachable (``NoPartialPublish``). Disk
        materialization happens AFTER that commit and is *best-effort*: every
        member is first staged to a durable tmp, then renamed into place, so a disk
        fault (ENOSPC, EACCES) fails BEFORE any rename with disk uniformly old, and
        a rename failing partway raises
        :class:`~ccs.core.exceptions.PublishMaterializationError` naming exactly
        which members landed. This shrinks — but a process crash between two
        renames cannot fully eliminate — the multi-file disk window (no POSIX
        multi-file atomic rename exists). On a ``PublishMaterializationError`` the
        coordinator is ahead of disk; recover by re-materializing each
        ``not_landed`` member with :meth:`write` of the bytes this publish
        committed for it (until then :meth:`read_with_version` refuses that
        member, since its disk bytes are not the content at the current
        version). Never retry the publish — it would version-mismatch.

        **Foreign-edit boundary (read this too).** The staleness this API
        detects is VERSION drift at the coordinator — and only volume-mediated
        writes advance versions. An out-of-band disk edit (a human in an
        editor, a formatter, a script writing the file directly) advances
        nothing, so a MULTI-member publish cannot see it: the session path
        never re-reads disk between the caller's read and materialization, the
        batch commits, and the foreign bytes are silently overwritten. The
        SB-23 content-CAS (``on_stale_write``) does NOT run on this path —
        the caller's own read seeds the foreign-edit baseline, but no
        publish step consults it. This is a deliberate, regression-pinned
        boundary (see test_atomic_publish.py's foreign-edit-boundary section),
        not a gap in the version check. Contrast: plain :meth:`write` denies
        the same edit with ``StaleView``, and a SINGLE-member publish takes the
        standalone CAS path, whose hash-checked comparand read fails closed
        (``ViewWedged``) on a managed path. Publish only write-sets whose every
        contending writer routes through a volume; for a file that humans or
        out-of-band tools also edit, use :meth:`write` — its foreign-edit
        guard covers exactly that case.

        **Sizing.** A single-member publish takes the standalone CAS path
        (:meth:`write_cas_at`'s primitive). A multi-member publish opens a
        consistent snapshot session over the write-set so the comparands are
        pinned at ONE linearization point (fractured-read-safe — no member is read
        across a peer commit), then commits them atomically. The session path adds
        a session-open round-trip, so its capture→commit window is WIDER than a
        single CAS's — a lost race there is HELD (``StaleView``), never a silent
        torn commit. A multi-member set requires UTF-8 text content (the session
        commit wire is text); a single-member publish accepts arbitrary bytes.

        **Single-host, cooperative** (like the rest of the volume): recovery from
        a HELD publish is ``reacquire`` + re-read + retry; the caller must write
        from freshly re-read bytes, never a buffer computed before the hold.
        """
        entries = self._normalize_publish_writes(writes)
        with self._single_op_guard():
            if len(entries) == 1:
                abs_path, rel, expected_version, disk_bytes = entries[0]
                new_version = self._write_cas_at_impl(rel, expected_version, disk_bytes)
                return {rel: new_version}
            return self._atomic_publish_session_impl(entries)

    def _normalize_publish_writes(
        self,
        writes: Sequence[tuple[str | os.PathLike[str], int, bytes | bytearray | str]],
    ) -> list[tuple[Path, str, int, bytes]]:
        """Validate + resolve the write-set to ``[(abs_path, rel, expected_version,
        disk_bytes), ...]``. Rejects an empty set, a bad content type, or a
        DUPLICATE path — a duplicate would silently collapse a member out of an
        all-or-nothing batch, so it fails loud before any coordinator I/O."""
        if not writes:
            raise ValueError("atomic_publish requires a non-empty write-set")
        resolved: list[tuple[Path, str, int, bytes]] = []
        seen: set[str] = set()
        for path, expected_version, content in writes:
            if isinstance(content, str):
                disk_bytes = content.encode("utf-8")
            elif isinstance(content, (bytes, bytearray)):
                disk_bytes = bytes(content)
            else:
                raise TypeError(
                    f"atomic_publish content for {path!r} must be bytes or str, "
                    f"got {type(content).__name__}"
                )
            abs_path, rel = self._to_relative(path)
            if rel in seen:
                raise ValueError(f"atomic_publish write-set has a duplicate path: {rel}")
            seen.add(rel)
            resolved.append((abs_path, rel, int(expected_version), disk_bytes))
        return resolved

    def _atomic_publish_session_impl(
        self, entries: list[tuple[Path, str, int, bytes]]
    ) -> dict[str, int]:
        """Multi-member publish via a consistent snapshot session (size > 1).

        Re-mint (split-comparand-safe), pin the write-set at ONE point via
        ``/session/begin``, verify each pinned version still equals the caller's
        ``expected_version`` (else the view was stale before the publish →
        ``CasVersionConflict``, nothing committed), then ``/session/commit_all``
        atomically. On a confirmed WIN write EVERY file (no same-bytes skip, so
        disk can't diverge from the coordinator's recorded hash); on a HELD
        conflict (a peer raced the window) write NONE and raise ``StaleView``;
        on an answer that does not confirm the commit — unconfirmed, or with
        no reason this client can classify — write NONE and raise
        ``CommitUnconfirmed`` in both ``on_error`` modes.
        """
        self._ensure_attached()
        if self._endpoint is None:
            raise CoherenceError(
                "cannot atomic_publish: coordinator endpoint unavailable (fail-closed)"
            )
        # Decode to the text wire BEFORE any side effect, so a non-UTF-8 member
        # fails with zero coordinator I/O rather than mid-session.
        try:
            wire = [
                {"path": rel, "content": disk_bytes.decode("utf-8")}
                for _abs, rel, _ev, disk_bytes in entries
            ]
        except UnicodeDecodeError as exc:
            raise ValueError(
                "atomic_publish of a multi-file set requires UTF-8 text content "
                "(the snapshot-session commit wire is text); a non-UTF-8 member "
                f"cannot be batch-published ({exc})."
            ) from None

        # Re-mint first: pin the cut under a fresh, non-INVALID identity — never
        # reacquire()-then-trust (KTD-LU split-comparand hole).
        self._remint()
        begin = self._post(
            "/session/begin",
            {"session_id": self._session_id, "read_set": [rel for _a, rel, _e, _b in entries]},
        )
        if begin is None:
            raise CommitUnconfirmed(
                "atomic_publish could not open a snapshot session (coordinator "
                "transport failed); nothing was published. retry."
            )
        if begin.get("degraded"):
            self._fail_closed_or_degrade(
                "coordinator watchdog timeout opening the publish session"
            )
        if not begin.get("ok"):
            raise CoherenceError(
                f"atomic_publish could not pin the write-set: {self._deny_reason(begin)}"
            )
        cut = begin.get("cut") or {}
        # Fractured-read-safe staleness gate: every member's pinned version must
        # still equal the caller's expected_version. A mismatch means a peer moved
        # that member before the session opened → the caller's view is stale; hold
        # the WHOLE publish (nothing committed, no file written).
        for _abs, rel, expected_version, _b in entries:
            pinned = cut.get(rel)
            if pinned != expected_version:
                raise CasVersionConflict(
                    rel, expected_version, pinned if isinstance(pinned, int) else -1
                )

        commit = self._post(
            "/session/commit_all",
            {
                "session_id": self._session_id,
                "session_token": begin.get("session_token"),
                "writes": wire,
            },
        )
        if commit is None:
            raise CommitUnconfirmed(
                "atomic_publish commit could not be confirmed (coordinator transport "
                "failed mid-commit); nothing landed. re-read and retry."
            )
        # An unconfirmed commit — the watchdog's degrade envelope, or a non-win
        # carrying its reason — raises in BOTH on_error modes, as on the
        # single-commit paths: degrade mode must not soften it into the HELD
        # conflict below, which would say nothing was published of a batch
        # that may have landed.
        if commit.get("degraded"):
            raise CommitUnconfirmed(_PUBLISH_UNCONFIRMED_MESSAGE)
        if commit.get("ok") is True:
            return self._materialize_publish(entries, commit.get("versions") or {})
        reason = commit.get("reason")
        if reason == COMMIT_UNCONFIRMED_REASON:
            raise CommitUnconfirmed(_PUBLISH_UNCONFIRMED_MESSAGE)
        # Non-WIN with no string reason: the coordinator's own non-win answers
        # always carry one, so this is what a proxy or gateway could send, and
        # it says nothing about whether the batch landed — the unknown, as on
        # the single-commit paths; never the HELD conflict (a definite "nothing
        # committed"), and never a TypeError from the substring test below.
        if not isinstance(reason, str):
            raise CommitUnconfirmed(_PUBLISH_UNCLASSIFIABLE_MESSAGE)
        # Non-WIN. A retry-eligible batch conflict (peer raced the window) is a
        # StaleView; a NON-retryable corruption reason must not masquerade as one
        # (mirror the size-1 CAS path, which raises CoherenceError on corruption).
        # Corruption is unreachable via the pinned cut (which guarantees
        # expected <= current), so this is defense-in-depth against a future
        # comparand source, not a currently-reachable branch.
        if "corruption" in reason:
            raise CoherenceError(
                f"atomic_publish rejected (non-retryable): {reason}"
            )
        # A batch conflict or a session-lifecycle rejection: recover via a fresh
        # read/session, which a StaleView (reacquire + re-decide) drives. Nothing
        # was committed and NO file is written.
        raise self._publish_hold(commit, entries)

    def _materialize_publish(
        self, entries: list[tuple[Path, str, int, bytes]], versions: dict
    ) -> dict[str, int]:
        """Materialize a coordinator-WON batch to disk with the smallest torn
        window: stage EVERY member's bytes to a durable tmp FIRST, then
        ``os.replace`` them all in a tight loop. Staging every tmp before any
        replace means a staging failure (e.g. ENOSPC) leaves disk UNIFORMLY OLD —
        the coordinator is ahead of disk but not torn. A replace failing partway
        tears the set (an earlier member's rename landed, a later one didn't);
        either failure raises a typed :class:`PublishMaterializationError` naming
        exactly which members landed vs not, so the caller can reconcile — never a
        bare ``OSError`` implying nothing published. This shrinks, but does not
        eliminate, the multi-file window: a process crash between two renames can
        still tear (no POSIX multi-file atomic rename exists); the coordinator
        commit is the atomic boundary, disk is best-effort materialization."""
        rels = [rel for _a, rel, _e, _b in entries]
        # ---- Stage every tmp first. A failure here means NO member is replaced. ----
        staged: list[tuple[Path, Path, str, int, bytes]] = []
        try:
            for abs_path, rel, expected_version, disk_bytes in entries:
                tmp = self._stage_tmp(abs_path, disk_bytes)
                staged.append((tmp, abs_path, rel, expected_version, disk_bytes))
        except OSError as exc:
            for tmp, *_ in staged:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
            raise PublishMaterializationError(
                landed=(), not_landed=tuple(rels), cause=exc
            ) from exc
        # ---- Commit every staged tmp (tight rename loop, near-atomic per file). ----
        written: dict[str, int] = {}
        landed: list[str] = []
        for index, (tmp, abs_path, rel, expected_version, disk_bytes) in enumerate(staged):
            try:
                self._replace_tmp(tmp, abs_path)
            except OSError as exc:
                not_landed = [r for (_t, _a, r, _e, _b) in staged[index:]]
                for (_t, _a, _r, _e, _b) in staged[index + 1:]:
                    with contextlib.suppress(OSError):
                        os.unlink(_t)
                raise PublishMaterializationError(
                    landed=tuple(landed), not_landed=tuple(not_landed), cause=exc
                ) from exc
            self._record_own_write(rel, self._sha256_bytes(disk_bytes))
            resolved_version = versions.get(rel)
            written[rel] = (
                resolved_version
                if isinstance(resolved_version, int)
                else expected_version + 1
            )
            landed.append(rel)
        return written

    def _publish_hold(
        self, commit: dict, entries: list[tuple[Path, str, int, bytes]]
    ) -> StaleView:
        """Build the ``StaleView`` for a HELD ``session/commit_all`` — carrying the
        first conflicting member's captured-vs-current drift (like ``gate()``) so a
        generic ``except StaleView`` reads ``expected_version`` / ``current_version``
        uniformly. Nothing is mutated on this path."""
        per_artifact = commit.get("per_artifact")
        expected_by_rel = {rel: ev for _a, rel, ev, _b in entries}
        held = StaleView(_PUBLISH_HELD_REASON)
        if isinstance(per_artifact, dict) and per_artifact:
            rel, detail = next(iter(per_artifact.items()))
            held.expected_version = expected_by_rel.get(rel)
            if isinstance(detail, dict):
                current = detail.get("current_version")
                held.current_version = current if isinstance(current, int) else None
                # The per-member refusal reason: same four-way distinction the
                # single-artifact CAS carries. The HOLD's own ``reason`` stays
                # the batch-level constant, so this rides alongside it.
                #
                # Allowlisted against the SAME set the single-artifact path
                # matches on, not merely type-checked. ``per_artifact`` is
                # coordinator-supplied JSON, and this string is destined for
                # prose a model reads; an isinstance check alone would let an
                # unrecognized reason through to whatever first renders it.
                # isinstance BEFORE the membership test: ``in`` against a
                # frozenset raises TypeError on an unhashable value, and this
                # body is JSON the coordinator supplied.
                member_reason = detail.get("reason")
                if (
                    isinstance(member_reason, str)
                    and member_reason in self._CAS_RETRY_REASONS
                ):
                    held.member_reason = member_reason
        return held

    # --- coordinator I/O helpers --------------------------------------------

    def _to_relative(self, path: str | os.PathLike[str]) -> tuple[Path, str]:
        """Resolve ``path`` to ``(absolute, workspace-relative-posix)``. Raises
        ``CoherenceError`` if it escapes the workspace root."""
        abs_path = Path(path)
        if not abs_path.is_absolute():
            abs_path = self._root / abs_path
        abs_path = abs_path.resolve()
        try:
            rel = abs_path.relative_to(self._root)
        except ValueError as exc:
            raise CoherenceError(
                f"path is outside the CoherentVolume root {self._root}: {abs_path}"
            ) from exc
        return abs_path, rel.as_posix()

    @staticmethod
    def _sha256_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _read_file_bytes(abs_path: Path) -> bytes:
        """Read a file via raw ``os.read`` (NOT ``builtins.open``), so the
        volume's own I/O is never self-intercepted by the ``install()``
        open()-shim (which patches ``builtins.open`` / ``io.open`` only — a
        ``read_bytes`` here would recurse back into the shim)."""
        fd = os.open(abs_path, os.O_RDONLY)
        try:
            chunks: list[bytes] = []
            while True:
                chunk = os.read(fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        finally:
            os.close(fd)

    @staticmethod
    def _disk_hash(abs_path: Path) -> str | None:
        """SHA-256 of the CURRENT on-disk bytes, or ``None`` if the file is
        absent/unreadable.

        The write no-op-skip uses this to confirm the file ALREADY holds the
        bytes being committed before skipping the ``os.replace``. A per-instance
        cached hash (``_last_committed_hash``) can be stale across a peer commit:
        on a tracked-but-non-strict glob the coordinator re-grants the write
        without a deny, so the cache still reads this instance's OWN last hash
        while disk holds the peer's bytes. The cache is therefore only a cheap
        fast-path gate; the on-disk hash is the authority for whether a write is
        truly a no-op. Reads via :meth:`_read_file_bytes` (raw ``os.read``) so it
        is never self-intercepted by the ``install()`` open()-shim."""
        try:
            return CoherentVolume._sha256_bytes(CoherentVolume._read_file_bytes(abs_path))
        except OSError:
            return None

    def _check_foreign_edit(self, rel: str, disk_hash: str | None) -> None:
        """SB-23 content-CAS predicate. Given the CURRENT on-disk hash for ``rel``,
        raise :class:`StaleView` under ``on_stale_write="raise"`` if the disk diverged
        from the baseline this instance last observed (a foreign / out-of-band edit),
        so the caller :meth:`reacquire`s instead of clobbering the foreign bytes.
        ``on_stale_write="allow"`` is a no-op (proceed and clobber — pre-SB-23
        behavior). A missing baseline (never-read path) or an absent file
        (``disk_hash is None`` — a foreign delete) also no-ops.

        Gated on a **managed (strict) path** — ``rel`` matched against this volume's
        managed globs. The check is only sound there: on a strict path a peer commit
        would have DENIED at pre-edit, so reaching this check means no peer commit
        landed and a disk mismatch is genuinely foreign. On a path this volume does
        NOT manage (untracked, or tracked-but-non-strict) pre-edit re-grants over a
        peer write, so a disk mismatch could be that coordinated change — not foreign
        — and must NOT be denied (the no-op-skip handles it). The caller fetches
        ``disk_hash`` once via :meth:`_disk_hash` and reuses it for the no-op-skip."""
        baseline = self._last_observed_hash.get(rel)
        diverged = (
            baseline is not None and disk_hash is not None and disk_hash != baseline
        )
        # matches_any (the glob scan) is evaluated last — the cheap checks short-circuit.
        if diverged and self._on_stale_write == "raise" and matches_any(rel, self._managed):
            raise StaleView(_STALE_WRITE_DENY_REASON)

    def _record_own_write(self, rel: str, content_hash: str) -> None:
        """This instance just put ``content_hash`` bytes on disk for ``rel`` — advance
        BOTH per-path caches together: ``_last_committed_hash`` (the no-op-skip
        fast-path gate) and ``_last_observed_hash`` (the SB-23 foreign-edit baseline).
        Use at the own-write sites where the two advance in lockstep. The main
        :meth:`write` path advances them SEPARATELY (observed before the post-edit
        POST, committed after — see :meth:`_write_impl`), so it does NOT use this."""
        self._last_committed_hash[rel] = content_hash
        self._last_observed_hash[rel] = content_hash

    def _stage_tmp(self, abs_path: Path, data: bytes) -> Path:
        """Durably write ``data`` to a unique ``.tmp`` beside ``abs_path`` (mkdir +
        raw ``os.write`` + ``fsync``) and return the tmp path; the caller commits
        it with :meth:`_replace_tmp`. Uses raw ``os.write`` (NOT ``builtins.open``)
        so the volume's own I/O is never self-intercepted by the ``install()``
        open()-shim. Cleans up the tmp and re-raises on any ``OSError`` — the tmp
        is not left orphaned, and nothing has touched ``abs_path`` yet."""
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = abs_path.with_name(f"{abs_path.name}.{self._session_id}.tmp")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                view = memoryview(data)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return tmp

    def _replace_tmp(self, tmp: Path, abs_path: Path) -> None:
        """Atomically move a staged tmp onto ``abs_path`` via ``os.replace``. On an
        ``OSError`` the tmp is removed and the error re-raised (``abs_path`` keeps
        its old bytes for THIS member — the rename is near-atomic, so this is the
        smallest failure surface)."""
        try:
            os.replace(tmp, abs_path)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def _atomic_write(self, abs_path: Path, data: bytes) -> None:
        """Durable, atomic single-file write: stage a tmp then ``os.replace`` it.
        The composition of :meth:`_stage_tmp` + :meth:`_replace_tmp` used by the
        single-file write paths."""
        tmp = self._stage_tmp(abs_path, data)
        self._replace_tmp(tmp, abs_path)

    def _post(self, endpoint_path: str, payload: dict) -> dict | None:
        """POST to the coordinator. Transport errors and non-2xx HTTP responses
        route through ``on_error`` (strict raises, degrade warns + returns
        ``None``). Otherwise returns the parsed 200 body.

        Every request names the current incarnation in the subagent field
        (``agent_id``), which the coordinator folds into the grant-row key; a
        body that already names one — the release of an abandoned incarnation —
        keeps its own. Every request presents the session's caller principal
        (header), which a require-class route checks the session against; an
        incarnation is a subagent component, so it shares the session's.

        A request refused for its principal (a typed ``caller_principal_*``
        reason) is recovered as R20 describes: the SAME nonce claims again, and
        a principal that differs from the one presented is adopted and the
        request retried exactly ONCE — safe, because a refused request mutated
        nothing. A 404 on that claim retries once without the header. A refusal
        the claim cannot cure raises the typed
        :class:`~ccs.core.exceptions.CallerPrincipalRefused` in BOTH ``on_error``
        modes: it is the coordinator's definite answer, not an infrastructure
        failure, so degrade mode does not soften it into a ``None`` that the
        caller would read as an unanswered request (a degraded read, a
        ``CommitUnconfirmed``, bytes written to disk unrecorded)."""
        payload = {"agent_id": self._incarnation, **payload}
        self._settle_unconfirmed_claim()
        sent = self._send(endpoint_path, payload)
        if sent.principal_refusal is None:
            return sent.body
        reason = sent.principal_refusal
        recovery = self._recover_principal()
        detail = recovery.detail
        if recovery.action == "retry":
            sent = self._send(endpoint_path, payload)
            if sent.principal_refusal is None:
                return sent.body
            reason, detail = sent.principal_refusal, PRINCIPAL_REFUSED_AGAIN
        # ``settled`` is False only when the recovery claim's answer was lost:
        # the volume adopted ``unconfirmed``, and its next request claims again
        # with the same nonce by itself (_settle_unconfirmed_claim) — so this
        # refusal is not the session's settled state, and a consumer must not
        # report it as one. A retry refused again follows a confirmed claim.
        raise CallerPrincipalRefused(
            reason, principal_refusal_message(reason, detail), settled=recovery.settled
        )

    def _recover_principal(self) -> PrincipalRecovery:
        """Claim again with the session's SAME nonce after a principal refusal
        and adopt what that settles (:func:`decide_principal_recovery` says
        whether to retry). Nothing is claimed once the session is known to be
        bound under another nonce: that never changes, so a permanently refused
        session costs no round trip per request."""
        if self._claim_outcome == "refused" or self._endpoint is None:
            return PrincipalRecovery(
                "stop", None,
                detail=(
                    "the session is bound under a different mint nonce "
                    f"({CALLER_PRINCIPAL_CLAIMED_REASON}); not claiming again"
                ),
            )
        presented = self._principal
        claim = claim_caller_principal(self._endpoint, self._session_id, self._mint_nonce)
        self._adopt_claim(claim)
        return decide_principal_recovery(claim, presented)

    def _send(self, endpoint_path: str, payload: dict) -> _Sent:
        """One POST presenting the current principal. A caller-principal
        refusal is returned for :meth:`_post` to recover; every other failure
        routes through ``on_error`` here.

        Every raise for a rejected request happens OUTSIDE the ``except``
        block, so the ``HTTPError`` never rides the raised error's chain: its
        text is the status line's reason phrase, which is the coordinator's,
        and a traceback or ``logger.exception`` would print it. What this
        client reports of a rejected request is its status code."""
        try:
            body = _coordinator_post(
                self._endpoint,
                endpoint_path,
                payload,
                extra_headers=caller_principal_headers(self._principal),
            )
        except urllib.error.HTTPError as exc:
            status, reason = exc.code, principal_refusal_reason(exc)
        except CoordinatorUnavailable as exc:
            self._fail_closed_or_degrade(
                f"coordinator request to {endpoint_path} failed: {exc}"
            )
            return _Sent(None, None)  # reached only in degrade mode
        else:
            return _Sent(body, None)
        # R2: a remote 401 is a wrong/missing secret — fail LOUD and CLOSED
        # with a distinct type, never the generic degrade path (which a
        # degrade-mode local client could swallow).
        if self._remote_endpoint is not None and status == 401:
            raise RemoteAuthFailed(
                "remote coordinator rejected the bearer token (401); the remote "
                "secret (CCS_REMOTE_SECRET_FILE) does not match the coordinator's "
                "hook.secret"
            )
        if reason is not None:
            return _Sent(None, reason)
        self._fail_closed_or_degrade(
            f"coordinator request to {endpoint_path} failed: HTTP {status}"
        )
        return _Sent(None, None)  # reached only in degrade mode

    def read_with_version(self, path: str | os.PathLike[str]) -> tuple[bytes, int]:
        """Read current bytes + the coordinator's authoritative version.

        The version is the OCC comparand an Option-A writer (the MCP
        ``swg_write_cas`` tool) passes back as ``expected_version``. Like
        :meth:`read`, a sticky-INVALID instance still returns fresh bytes and the
        version reflects the peer's commit; the instance stays INVALID until
        :meth:`reacquire` re-mints. ``FileNotFoundError`` for a missing file.

        Raises :class:`~ccs.core.exceptions.StaleView` instead of returning a
        pair the coordinator denied because the bytes are not its content at
        that version (see :meth:`_refuse_split_pair`).
        """
        with self._single_op_guard():
            result = self._read_with_version(path)
            self._refuse_split_pair(result)
            return result.data, result.version

    @staticmethod
    def _refuse_split_pair(result: _ReadResult) -> None:
        """Raise ``StaleView`` rather than hand out a split ``(bytes, version)``.

        A read that lands between a peer's confirmed CAS and that peer's disk
        write sees the OLD bytes while the coordinator already reports the NEW
        version. Strict mode denies it with ``hash_differs``, but the pair used
        to be returned anyway, and a caller that derived from those bytes and
        CASed at that version won once the peer's disk write landed: the peer's
        update was overwritten. ``write_cas_at`` cannot catch this, because its
        own comparand read runs after the disk write and compares only
        versions. The same deny also fires on an out-of-band edit; that pair is
        just as unsound.

        A deny WITHOUT ``hash_differs`` is kept: that is the sticky-INVALID read
        of bytes that match the version (the pair is sound; the instance still
        has to reacquire before a plain :meth:`write`). An admitted pair is
        returned unchanged even when it carries ``hash_differs``. On a
        tracked-but-not-strict path a mismatch is the normal state (the
        cross-host mode keeps each host's bytes on its own disk). On a strict
        path the coordinator admits a mismatch for this instance while it is
        the artifact's most recent committer, within the coordinator's short
        commit-lag window, whatever the cause: its own disk write not yet
        landed or failed, or an out-of-band edit made after it landed. Those
        split pairs are returned, not refused.
        """
        if result.split_pair:
            raise StaleView(_SPLIT_READ_DENY_REASON)

    #: Whether the most recent :meth:`read_with_version_generation` was refused
    #: by the coordinator (strict-mode deny). Read by the effect fence to name
    #: the HOLD cause precisely; per-instance and overwritten each call.
    _last_read_denied: bool = False

    #: Whether the most recent :meth:`read_with_version_generation` came back
    #: with the STALE status shape — this instance had no standing grant on the
    #: current version at that read (a warn-mode stale re-grant, or a deny).
    #: Read by the effect fence: a grant that did not stand at re-validate
    #: HOLDs even when the ``(version, owner_generation)`` pair is unchanged,
    #: the drift a peer's write-claim preemption leaves behind (no commit, no
    #: epoch bump). Per-instance and overwritten each call, like its sibling.
    _last_read_stale: bool = False

    def read_with_version_generation(
        self, path: str | os.PathLike[str], *, observe: bool = True
    ) -> tuple[bytes, int, int | None]:
        """Read current bytes + the coordinator's authoritative
        ``(version, owner_generation)`` pair.

        The generation-bearing sibling of :meth:`read_with_version`, driven by
        ``adapters.effect_gate.gate()``: the version answers "is the value still
        the one the decision saw", the ownership generation answers "is the
        grant it was read under still standing" — a sweep reclamation of a
        stalled holder advances the generation WITHOUT a version move, which a
        version-only comparand cannot see. The pair is a single coordinator-side
        snapshot (never torn by a concurrent reclaim). ``owner_generation`` is
        ``None`` when the coordinator did not confirm one — an older
        coordinator, a strict-mode deny, or a degraded read — and callers must
        treat ``None`` as UNCONFIRMED (the gate HOLDs on it), never as "no
        movement". ``FileNotFoundError`` for a missing file.

        ``observe=False`` marks a VERIFICATION read whose bytes the caller
        discards (the effect fence re-reading to compare comparands). It leaves
        the foreign-edit baseline where it was, so checking freshness cannot
        absolve an out-of-band edit the caller never actually saw.

        An observing read raises :class:`~ccs.core.exceptions.StaleView` instead
        of returning a split pair (see :meth:`_refuse_split_pair`). A
        verification read still returns: its bytes are discarded, and the fence
        classifies it from ``_last_read_denied``, which is set either way.
        """
        with self._single_op_guard():
            result = self._read_with_version(path, observe=observe)
            # A strict-mode deny is reported as a REFUSED read, not merely as a
            # missing generation: they need different answers (a deny clears
            # with reacquire; a coordinator that cannot report generations at
            # all does not), and on the strict path a sweep reclaim reaches the
            # client precisely AS a deny.
            self._last_read_denied = result.stale_denied
            # A stale-status read (warn re-grant OR deny) means the grant this
            # instance was previously read under did NOT stand at this read —
            # the re-grant, if any, is a NEW grant. The effect fence keys the
            # write-preemption HOLD on this: a peer's write-claim acquire ends
            # the caller's grant while moving neither comparand.
            self._last_read_stale = result.stale_status
            if observe:
                self._refuse_split_pair(result)
            return result.data, result.version, result.owner_generation

    def _read_with_version(
        self, rel: str, *, observe: bool = True, advance_baseline: bool = True
    ) -> _ReadResult:
        """OCC helper: register a SHARED view and return
        ``(bytes, version, stale_denied, owner_generation, stale_status,
        content_differs)``
        (a :class:`_ReadResult`); ``stale_status`` is True when the coordinator
        classified this read as stale (a warn re-grant or a deny), the
        standing-grant signal the effect fence keys the ``grant_preempted``
        HOLD on.

        ``advance_baseline=False`` is for a CAS comparand read, whose bytes are
        not handed to the caller as such: :meth:`write_cas` advances the
        foreign-edit baseline itself, only once a clean read's bytes reach
        ``make_content``, and :meth:`write_cas_at` never does (its caller
        supplies the content). The one exception is a path this instance has
        never observed: there the comparand read, like every observing read,
        records the first baseline before its request, and it never replaces
        one. It changes nothing on the wire, unlike ``observe=False``.

        Mirrors :meth:`read` (same pre-read call + same fail-closed degrade
        handling) but also surfaces the coordinator's authoritative ``version``
        — the comparand ``write_cas`` passes as ``expected_version`` — and
        whether this instance is INVALID (a sticky strict-deny: the pre-read
        returned a ``stale`` response and did NOT re-grant SHARED, KTD-T).
        ``read`` itself stays ``bytes``-returning (a public contract); this is
        the internal version-aware variant ``write_cas`` drives.

        A fresh pre-read carries ``{"status": "fresh", "version": N}``; a
        warn-mode stale read and a strict-deny both carry ``status == "stale"``
        with the version under ``summary.current_version``. ``stale_denied`` is
        True on the strict-deny case — an INVALID writer cannot CAS until it
        :meth:`reacquire`s. If the version cannot be resolved (older coordinator
        / degraded status surface) ``0`` is used — a CAS against
        ``expected_version=0`` on a non-empty artifact loses cleanly
        (``version_mismatch``), so the fallback never causes a silent overwrite.
        ``owner_generation`` rides the same fail-closed discipline: ``None``
        when the coordinator did not confirm one (older coordinator, deny,
        degraded), which generation-aware callers treat as UNCONFIRMED.
        """
        self._ensure_attached()
        abs_path, _rel = self._to_relative(rel)
        if not abs_path.is_file():
            raise FileNotFoundError(f"no such file in workspace: {_rel}")
        data = self._read_file_bytes(abs_path)
        content_hash = self._sha256_bytes(data)
        # A path's FIRST observing read records what the disk held, before the
        # response can refuse or fail it (a strict deny, a request refused for
        # its caller principal, a strict transport or watchdog failure). It
        # never replaces a baseline, so it cannot absolve an edit made after an
        # earlier read; it only keeps a path whose first read was refused — or
        # whose only observing read was a CAS comparand read that lost, was
        # refused or failed — from having no baseline at all, which would let a
        # later write() land over a peer commit or an out-of-band edit unchecked.
        if observe:
            self._last_observed_hash.setdefault(_rel, content_hash)
        version = 0
        stale_denied = False
        stale_status = False
        content_differs = False
        owner_generation: int | None = None
        if self._endpoint is not None:
            resp = self._post(
                "/hooks/pre-read",
                {
                    "session_id": self._session_id,
                    "path": _rel,
                    "content_hash": content_hash,
                    # Opt-in: ask for the pair-consistent (version,
                    # owner_generation). Older coordinators ignore the flag and
                    # answer without the key (generation stays None).
                    "want_owner_generation": True,
                    # A verification read (``observe=False`` -- the effect
                    # fence re-reading only to compare comparands) must not
                    # MUTATE coordinator grant state: re-granting SHARED here
                    # would heal the very grant loss the fence is checking, so
                    # the caller stays whatever it was (INVALID after a
                    # preemption), and a bare re-check re-HOLDs instead of
                    # admitting. The client-side ``observe`` already keeps the
                    # foreign-edit baseline still; this carries the same
                    # "verification is not observation" rule to the server.
                    # Older coordinators ignore it (edge-triggered fallback,
                    # still fail-closed on the first check).
                    "verify_only": not observe,
                },
            )
            if isinstance(resp, dict):
                if resp.get("degraded"):
                    self._fail_closed_or_degrade(
                        f"coordinator watchdog timeout during read of {_rel}"
                    )
                elif resp.get("ok") is False:
                    # Degrade mode falls through with the fail-closed
                    # comparands: no version key → 0, no generation → None.
                    self._fail_closed_or_degrade(self._pre_read_failure(resp, _rel))
                version = self._pre_read_version(resp)
                owner_generation = self._pre_read_owner_generation(resp)
                # A strict-deny (INVALID, NOT re-granted — KTD-T) is the only
                # pre-read outcome that leaves this instance unable to CAS: it
                # stays INVALID AND keeps the invalidation transient the peer
                # commit set, which commit_cas rejects as a precondition. A
                # warn-mode stale read (permissionDecision == "allow") re-grants
                # SHARED, so it is NOT treated as denied here. The distinguisher
                # is permissionDecision == "deny" (set only by emit_strict_deny).
                hook_output = resp.get("hookSpecificOutput")
                if isinstance(hook_output, dict):
                    stale_denied = hook_output.get("permissionDecision") == "deny"
                # Any stale-status response — warn re-grant or deny alike —
                # means this instance's prior grant did not stand at this read.
                stale_status = resp.get("status") == "stale"
                content_differs = self._pre_read_hash_differs(resp)
        result = _ReadResult(
            data, version, stale_denied, owner_generation, stale_status, content_differs
        )
        # SB-23: the OCC read path also advances the foreign-edit baseline — but
        # ONLY when the caller actually OBSERVES these bytes: here, where they
        # are returned to it (a raise above — a request refused for its caller
        # principal, a strict-mode transport or watchdog failure — leaves the
        # caller without them). A verification read (``observe=False``, used by
        # the effect fence) reads the file to compare comparands and then
        # DISCARDS the bytes; a CAS comparand read (``advance_baseline=False``)
        # hands them on only if it is clean, and then write_cas advances the
        # baseline itself (write_cas_at never does: its caller supplies the
        # content). A split pair reaches no caller either (the public reads
        # refuse it and the OCC loops discard it), so it is decided AFTER the
        # response, not before: seeding first let a refused read of an
        # out-of-band edit clear the way for a write over that edit. Advancing
        # the baseline in any of these cases would silently absolve a foreign
        # edit the caller never saw, so the next write would clobber it instead
        # of denying. A fail-closed check must not have a fail-open side
        # effect. The first-read seed above still applies to every observing
        # read.
        if observe and advance_baseline and not result.split_pair:
            self._last_observed_hash[_rel] = content_hash
        return result

    @staticmethod
    def _pre_read_failure(resp: dict, rel: str) -> str:
        """The message for a pre-read the coordinator answered with its
        failure envelope, HTTP 200 ``{"ok": false, "reason": "internal:
        <Type>"}`` — what it answers when the caller-principal gate's store
        read or the pre-read work body raised. The gate's raise registered
        nothing; a body raise may have, since the handler's registry calls are
        separate statements and one can fail after another has recorded a
        SHARED grant. Either way the client cannot know the read's standing,
        so the answer is an unanswered request with a name: it goes through
        ``_fail_closed_or_degrade`` exactly as a transport failure does
        (strict raises, degrade warns and counts), never through the deny
        arm (a deny is enforcement working and the read stays registered)
        and never through as a registered read. Before this arm existed a
        strict volume returned the bytes with no view recorded, so a peer's
        later commit invalidated nothing and this instance's next write
        landed over it. The pre-read handler itself never answers an ``ok``
        key, so the family is exactly the envelope.

        Only the envelope's ``reason`` is named — the coordinator's, relayed
        verbatim as ``_deny_reason`` relays a deny's, and carrying only an
        exception's type — never the rest of the body."""
        reason = resp.get("reason")
        named = reason if isinstance(reason, str) and reason else "no reason given"
        return (
            f"coordinator answered the read of {rel} with a failure ({named}); "
            "the read is not confirmed as registered"
        )

    @staticmethod
    def _pre_read_version(resp: dict) -> int:
        """Extract the coordinator version from a pre-read response (fresh or
        stale shape). Returns 0 when absent (older coordinator / degraded)."""
        v = resp.get("version")
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        summary = resp.get("summary")
        if isinstance(summary, dict):
            cv = summary.get("current_version")
            if isinstance(cv, int) and not isinstance(cv, bool):
                return cv
        return 0

    @staticmethod
    def _pre_read_owner_generation(resp: dict) -> int | None:
        """Extract the ownership generation from a pre-read response. ``None``
        when absent or malformed — the fail-closed sentinel for "the coordinator
        never confirmed a generation" (older coordinator, strict deny,
        degraded); never coerced to 0, which is a REAL generation.

        ``None`` ALSO when the coordinator reports the caller's content hash
        differs from the content it records at that version (``hash_differs``,
        on either the fresh or the stale shape). The generation is the
        *authority* comparand — "is the grant these bytes were read under still
        standing" — and a hash mismatch means the coordinator cannot vouch that
        the bytes in hand ARE the content at that version. In warn mode such a
        read is a fail-open allow (a re-grant, or a `hash_differs` fresh), so the
        version comparand alone would re-validate clean at an effect boundary and
        fire a decision derived from superseded bytes. Reporting the generation
        as UNCONFIRMED makes the effect gate HOLD instead; the plain
        ``read``/``read_with_version`` paths are unaffected.

        Only the WIRE DECODING is local (which key on which response shape);
        the demotion rule itself lives in ``ccs.core.fence`` because the
        coordinator route reaching the same fence must apply the identical
        rule, and a second copy of a safety rule is the drift this split
        exists to remove."""
        return confirmed_generation(
            resp.get("owner_generation"),
            content_hash_differs=CoherentVolume._pre_read_hash_differs(resp),
        )

    @staticmethod
    def _pre_read_hash_differs(resp: dict) -> bool:
        """True when the coordinator flagged the caller's content hash as
        differing from the content it records — top-level on the fresh shape,
        under ``summary`` on the stale shape."""
        if resp.get("hash_differs") is True:
            return True
        summary = resp.get("summary")
        return isinstance(summary, dict) and summary.get("hash_differs") is True

    def _current_bytes_or_empty(self, abs_path: Path) -> bytes:
        """Degrade-path read of the on-disk bytes (b\"\" if the file is absent),
        so ``make_content`` always gets a defined current view."""
        if not abs_path.is_file():
            return b""
        return self._read_file_bytes(abs_path)

    # Retry-eligible OCC conflict reasons matched EXACTLY against the wire
    # ``reason``: the typed ``ConflictDetail`` reasons plus
    # ``caller_in_transient_state`` (AC2 — a peer invalidated us mid-window).
    # The transient literal is the SHARED constant the coordinator server emits,
    # so a reword on either side can't drift the retry classification.
    _CAS_RETRY_REASONS: frozenset[str] = frozenset(
        {
            "version_mismatch",
            "other_holder",
            OCC_CALLER_TRANSIENT_REASON,
            # Read-generation fence: a reclaimed reader's OCC commit_cas returns
            # ConflictDetail("stale_read_generation"); retry via reacquire +
            # fresh read (the next fetch captures the current generation).
            STALE_READ_GENERATION_REASON,
        }
    )

    def _classify_cas_response(self, resp: dict) -> Literal["win", "conflict", "raise"]:
        """Map a ``/hooks/post-edit-cas`` 200 body to an OCC outcome.

        - ``ok: true``  → ``"win"`` (the CAS committed; version bumped).
        - ``ok: false`` with a retry-eligible reason → ``"conflict"`` (reacquire
          + retry). Matched EXACTLY against :attr:`_CAS_RETRY_REASONS`: the
          typed ``ConflictDetail`` reasons (``version_mismatch`` /
          ``other_holder`` / ``stale_read_generation`` — the read-generation
          fence: the caller's captured claim was superseded by a sweep
          reclamation; a reacquire + fresh read mints a current claim)
          AND ``caller_in_transient_state`` — when a peer
          invalidates this instance in the window BETWEEN its fresh read and its
          CAS, the coordinator leaves an invalidation transient that
          ``commit_cas`` rejects as a precondition; that is a lost race, not
          corruption, so reacquire + retry (a fresh identity has no transient).
          The transient reason is the shared
          :data:`~ccs.core.exceptions.OCC_CALLER_TRANSIENT_REASON` the server
          emits, so the match is exact (no brittle substring) and cannot drift.
        - ``ok: false`` otherwise — true corruption (``commit_cas_corruption``,
          ``expected > current``) OR the fail-closed ``{degraded: true,
          reason: "commit_unconfirmed"}`` body → ``"raise"``. The degrade body
          deliberately reads as failure so the client never mistakes an
          unconfirmed CAS for a landed write.
        """
        if resp.get("ok") is True:
            return "win"
        reason = resp.get("reason")
        # Only a string can be a member: a list or an object here (never the
        # coordinator's own answer) is not retry-eligible, not a TypeError
        # from the frozenset membership test.
        if isinstance(reason, str) and reason in self._CAS_RETRY_REASONS:
            return "conflict"
        return "raise"

    @staticmethod
    def _cas_current_version(resp: dict, fallback: int) -> int:
        cv = resp.get("current_version")
        if isinstance(cv, int) and not isinstance(cv, bool):
            return cv
        return fallback

    def _check_grant(self, resp: dict, rel: str, *, phase: str) -> None:
        """Map a coordinator pre-/post-edit response to the fail-closed contract.

        A deny (``ok: false`` — strict-deny or collision/preempt) ALWAYS raises
        ``CoherenceError`` with the reason verbatim, in both ``on_error`` modes:
        the deny is the enforcement signal. A watchdog-timeout degrade
        (``degraded: true``) is an infra failure and routes through ``on_error``.
        """
        if not isinstance(resp, dict):
            return
        if resp.get("ok") is False:
            # Type the deny by phase so the MCP mapper classifies by exception
            # type, not a substring of the (verbatim) coordinator prose. A
            # pre-edit INVALID deny is a recoverable stale view; a post-edit
            # ok:false is a mid-write preempt/reclaim — the atomic write (:523)
            # already hit disk un-versioned, so it is a distinct terminal that
            # needs reconcile, not a bare retry.
            if phase == "post-edit":
                raise CommitPreempted(self._deny_reason(resp))
            raise StaleView(self._deny_reason(resp))
        if resp.get("degraded"):
            self._fail_closed_or_degrade(
                f"coordinator watchdog timeout on {phase} for {rel}"
            )

    @staticmethod
    def _deny_reason(resp: dict) -> str:
        """Extract the coordinator's deny reason VERBATIM (regenerating it
        worsens model retries — auto memory: project_cc_strict_mode_retry_hazard)."""
        hook_output = resp.get("hookSpecificOutput")
        if isinstance(hook_output, dict):
            reason = hook_output.get("permissionDecisionReason")
            if isinstance(reason, str) and reason:
                return reason
        reason = resp.get("reason")
        if isinstance(reason, str) and reason:
            return reason
        return (
            "coherence coordinator denied the write (stale view); "
            "reacquire() and write from the fresh bytes"
        )

    # --- degrade contract ---------------------------------------------------

    @property
    def is_degraded(self) -> bool:
        return self._degradation_count > 0

    @property
    def degradation_count(self) -> int:
        return self._degradation_count

    def _fail_closed_or_degrade(self, message: str) -> None:
        """Strict → raise (fail-closed); degrade → warn once + count."""
        if self._on_error == "strict":
            raise CoherenceError(message)
        self._record_degraded(message)

    def _record_degraded(self, message: str) -> None:
        with self._lock:
            first = self._degradation_count == 0
            self._degradation_count += 1
        if first:
            # Log before warning: a caller that escalates the warning to an error
            # makes warn() raise, and the log line must not depend on it.
            logger.warning("CoherentVolume degraded under on_error='degrade': %s", message)
            warnings.warn(
                f"CoherentVolume degraded: {message}",
                CoherenceDegradedWarning,
                stacklevel=3,
            )


# ----------------------------------------------------------------------
# install() — opt-in builtins.open / io.open shim (Unit 3, demo-grade)
# ----------------------------------------------------------------------
#
# Routes opens of *managed* paths through a process-singleton CoherentVolume so
# existing code gets coherence without changing its open()/pathlib calls. This
# is a DEMO-GRADE convenience layer; the explicit CoherentVolume read/write/
# reacquire API is the supported primitive. Coverage matrix (single-host, one
# process + its forks):
#
#   COORDINATED   builtins.open(p, "r"|"rb"|"w"|"wb") for a managed path, and
#                 pathlib Path.open / read_text / write_text / read_bytes /
#                 write_bytes — they call io.open, which we patch ALONGSIDE
#                 builtins.open (patching builtins.open alone does NOT catch
#                 pathlib; verified empirically).
#   NOT COVERED   os.open / os.write (raw fds), subprocess / shell redirection
#                 ("echo >> p"), mmap, C-level writes, and append / update /
#                 exclusive modes ("a", "r+", "x") — all delegate to the
#                 original open() unchanged.
#
# Recovery: a managed read() through the shim registers a SHARED view, so a peer
# commit makes the next shim'd write fail closed (raises out of close()). The
# deny is STICKY — a bare re-open for read does NOT clear it; recovery uses the
# explicit volume.reacquire() (the shim deliberately keeps open() semantics
# simple rather than auto-refetching, which would defeat the guard).


class _CommitOnCloseMixin:
    """In-memory write buffer that commits through the volume on ``close()``.

    A stale-view deny raises out of ``close()`` (fail-closed), so a
    ``with open(p, "w") as f: ...`` block surfaces the lost-update guard. The
    bytes never touch disk except via the volume's atomic write.
    """

    _volume: CoherentVolume
    _rel: str
    _encoding: str | None  # None => binary mode
    _committed: bool

    def __exit__(self, exc_type, exc_val, exc_tb):  # type: ignore[override]
        # If the ``with`` body raised, DISCARD the buffered write rather than
        # commit a partial/abandoned buffer. No coordinator grant was acquired
        # (the EXCLUSIVE acquire happens inside volume.write, only on a clean
        # commit in close()), so there is nothing to release — skipping the
        # commit is sufficient and leaks nothing.
        if exc_type is not None:
            self._committed = True
        return super().__exit__(exc_type, exc_val, exc_tb)  # type: ignore[misc]

    def close(self) -> None:  # type: ignore[override]
        if self._committed or self.closed:  # type: ignore[attr-defined]
            super().close()  # type: ignore[misc]
            return
        self._committed = True
        try:
            payload = self.getvalue()  # type: ignore[attr-defined]
            if self._encoding is not None:
                payload = payload.encode(self._encoding)
            self._volume.write(self._rel, payload)
        finally:
            super().close()  # type: ignore[misc]


class _CoherentBytesWriter(_CommitOnCloseMixin, io.BytesIO):
    def __init__(self, volume: CoherentVolume, rel: str) -> None:
        io.BytesIO.__init__(self)
        self._volume = volume
        self._rel = rel
        self._encoding = None
        self._committed = False


class _CoherentTextWriter(_CommitOnCloseMixin, io.StringIO):
    def __init__(self, volume: CoherentVolume, rel: str, encoding: str) -> None:
        io.StringIO.__init__(self)
        self._volume = volume
        self._rel = rel
        self._encoding = encoding
        self._committed = False


class _ShimState:
    """Process-global state for the installed shim (one workspace per process)."""

    def __init__(self, volume: CoherentVolume, original_open: Callable) -> None:
        self.volume = volume
        self.original_open = original_open


_shim_state: _ShimState | None = None


def _managed_rel(volume: CoherentVolume, file: object) -> str | None:
    """Return the workspace-relative posix path if ``file`` is a managed path
    under the volume root, else ``None`` (→ delegate to the original open()).

    Never raises: an fd int, a bytes path, an outside-root or unresolvable path
    all fall back to ``None`` so the original open() handles them unchanged.
    """
    if isinstance(file, int):  # raw fd — not a path
        return None
    try:
        raw = os.fspath(file)
    except TypeError:
        return None
    if not isinstance(raw, str):  # bytes path — demo-grade: delegate
        return None
    try:
        # Resolve like builtins.open (relative → against CWD), then require it be
        # under the volume root.
        rel = Path(raw).resolve().relative_to(volume._root).as_posix()
    except (ValueError, OSError):
        return None
    # Use the coordinator's own glob matcher so the shim's notion of "managed" is
    # exactly the coordinator's tracked set (handles ``**`` segment globs).
    return rel if matches_any(rel, volume._managed) else None


def _make_open_wrapper(volume: CoherentVolume, original_open: Callable) -> Callable:
    def coherent_open(file, mode="r", *args, **kwargs):
        rel = _managed_rel(volume, file)
        if rel is None:
            return original_open(file, mode, *args, **kwargs)
        # Demo-grade: only plain read / truncating-write are mediated. Append,
        # update ("+"), and exclusive ("x") modes delegate unchanged.
        if "+" in mode or "a" in mode or "x" in mode:
            return original_open(file, mode, *args, **kwargs)
        if "w" in mode:
            if "b" in mode:
                return _CoherentBytesWriter(volume, rel)
            return _CoherentTextWriter(volume, rel, kwargs.get("encoding") or "utf-8")
        if "r" in mode:
            # Register the SHARED view (raises FileNotFoundError if missing, like
            # open()), then return a real handle over the same on-disk bytes.
            volume.read(rel)
            return original_open(file, mode, *args, **kwargs)
        return original_open(file, mode, *args, **kwargs)

    return coherent_open


def install(
    root: str | os.PathLike[str],
    *,
    managed: tuple[str, ...] = (),
    on_error: Literal["strict", "degrade"] = "strict",
    on_stale_read: Literal["allow", "raise"] = "allow",
    on_stale_write: Literal["allow", "raise"] = "raise",
    config: LifecycleConfig | None = None,
    bind_host: str = "127.0.0.1",
) -> CoherentVolume:
    """Patch ``builtins.open`` + ``io.open`` to route managed-path opens through
    a process-singleton :class:`CoherentVolume`. Idempotent — a second call
    returns the already-installed volume (one workspace per process in v1).
    Reverse with :func:`uninstall`, or use the :func:`coherent_workspace` context
    manager. Opt-in and demo-grade — see the coverage matrix above.
    """
    global _shim_state
    if _shim_state is not None:
        return _shim_state.volume
    volume = CoherentVolume(
        root, managed=managed, on_error=on_error, on_stale_read=on_stale_read,
        on_stale_write=on_stale_write, config=config, bind_host=bind_host,
    )
    original_open = builtins.open  # is io.open (same object) at install time
    wrapper = _make_open_wrapper(volume, original_open)
    builtins.open = wrapper  # type: ignore[assignment]
    io.open = wrapper  # type: ignore[assignment]  # pathlib routes here, not builtins.open
    _shim_state = _ShimState(volume, original_open)
    return volume


def uninstall() -> None:
    """Restore the original ``builtins.open`` / ``io.open``. Idempotent."""
    global _shim_state
    if _shim_state is None:
        return
    builtins.open = _shim_state.original_open  # type: ignore[assignment]
    io.open = _shim_state.original_open  # type: ignore[assignment]
    _shim_state = None


@contextlib.contextmanager
def coherent_workspace(
    root: str | os.PathLike[str],
    *,
    managed: tuple[str, ...] = (),
    on_error: Literal["strict", "degrade"] = "strict",
    on_stale_read: Literal["allow", "raise"] = "allow",
    on_stale_write: Literal["allow", "raise"] = "raise",
    config: LifecycleConfig | None = None,
    bind_host: str = "127.0.0.1",
) -> Iterator[CoherentVolume]:
    """Context manager wrapping :func:`install`/:func:`uninstall`; yields the
    process-singleton :class:`CoherentVolume`. Guarantees the ``open()`` patch is
    reversed on exit even if the body raises.

    Reentrant-safe: if a shim is already installed (e.g. a nested
    ``coherent_workspace``), this yields the existing volume and does NOT
    uninstall on exit — the outer context owns the patch.
    """
    already_installed = _shim_state is not None
    volume = install(
        root, managed=managed, on_error=on_error, on_stale_read=on_stale_read,
        on_stale_write=on_stale_write, config=config, bind_host=bind_host,
    )
    try:
        yield volume
    finally:
        if not already_installed:
            uninstall()
