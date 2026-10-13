# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The bytes-I/O contract a bring-your-own-substrate binding implements.

This module is the CONTRACT half that lives with its callers: every
implementer and consumer of the byte-level substrate surface is an adapter,
so the Protocol sits in the adapters layer while the capability descriptor,
tier vocabulary, and never-ship-a-store floor live in
:mod:`ccs.core.substrate` (importable by the conformance kit without pulling
in any I/O).

Conformance is structural, mirroring the registry-contract discipline: the
Protocol is :func:`~typing.runtime_checkable` so ``isinstance`` verifies
presence of the surface, each binding adds a ``TYPE_CHECKING``-guarded static
assertion, and the descriptor-parametrized conformance suite is the parity
test. Bindings never inherit the Protocol.

Shared contract vocabulary (type aliases and the typed compare-and-set
outcomes) is DEFINED here and imported by bindings, so no two bindings can
drift apart on what a win, a conflict, or an unknown outcome means.
"""

from __future__ import annotations

import os
import secrets
import urllib.error
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol, TypeAlias, runtime_checkable

import yaml

from ccs.cli._coherence_client import (
    PRINCIPAL_REFUSED_AGAIN,
    CoordinatorEndpoint,
    CoordinatorUnavailable,
    caller_principal_headers,
    claim_caller_principal,
    decide_principal_recovery,
    principal_refusal_message,
    principal_refusal_reason,
    resolve_endpoint,
)
from ccs.cli._coherence_client import (
    post as _coordinator_post,
)
from ccs.core.exceptions import (
    CALLER_PRINCIPAL_CLAIMED_REASON,
    COMMIT_UNCONFIRMED_REASON,
    HANDLER_FAILURE_REASON_PREFIX,
    OCC_CALLER_TRANSIENT_REASON,
    STALE_READ_GENERATION_REASON,
    VERSION_MISMATCH_REASON,
    CallerPrincipalRefused,
    CasVersionConflict,
    CoherenceError,
    CommitUnconfirmed,
    RedirectRefused,
    StaleView,
    TlsVerificationFailed,
    ViewWedged,
)
from ccs.core.substrate import CapabilityDescriptor
from ccs.core.substrate import sha256_hex as _sha256_hex

if TYPE_CHECKING:
    from ccs.adapters.claude_code.lifecycle import LifecycleConfig

# The substrate-minted version token: an object ETag, a row-version rendered
# opaque, etc. Two properties are contractual: it comes from the SAME read (or
# write response) as the bytes it vouches for, and it is never portable across
# artifact refs — identical bytes elsewhere can mint an identical token, so a
# token for one ref must never arbitrate a write to another.
SubstrateToken: TypeAlias = str


@dataclass(frozen=True)
class CasWritten:
    """WIN: the substrate accepted the compare-and-set write.

    ``token`` is the token the substrate minted FOR THIS WRITE, captured from
    the write response — never recomputed client-side (a recomputed token can
    silently vouch for bytes the substrate never acknowledged).
    """

    token: SubstrateToken


@dataclass(frozen=True)
class CasConflict:
    """LOSS: the substrate rejected the compare-and-set — the token moved.

    No write landed. Recovery is a fresh read (bytes and token from ONE read),
    re-derive, retry; the stale comparand must never be re-driven.
    """


@dataclass(frozen=True)
class CasUnknown:
    """UNKNOWN, not failed: the write's outcome could not be confirmed.

    A transport failure after the request left the client means the write may
    or may not be durable. Treat it as unconfirmed: reconcile by re-reading the
    token before doing anything else, never blindly re-drive the write, and
    never advance coordinator state on it.
    """


# The three-way compare-and-set outcome. A typed return, never an exception:
# conflict and unknown are expected protocol states a caller must branch on.
CasWriteResult: TypeAlias = "CasWritten | CasConflict | CasUnknown"


# --- the UNIFIED reconciliation seam ---------------------------------------
#
# A binding's ``reconcile_after_unknown`` re-reads the substrate after an
# UNKNOWN write and returns one of these verdicts. Both native-CAS bindings
# share this ONE vocabulary so the cross-agent commit dispatches uniformly —
# a Postgres row and an S3 object speak the same reconciliation language even
# though each derives its verdict from a different token shape.


class ReconcileVerdict(Enum):
    """What the cross-agent commit must do after an unconfirmed substrate write.

    The union across both native-CAS bindings. A binding uses only the arms its
    token shape can honestly reach — Postgres (a write-counting version) uses
    ``CONVERGE`` / ``RE_DERIVE`` / ``HOLD``; S3 (a content-derived ETag) uses
    ``CONVERGE`` / ``RE_DRIVE`` / ``CONFLICT`` / ``HOLD`` — but the caller
    dispatches on the ONE enum regardless of which binding produced it.

    - ``CONVERGE`` — the re-read proves the intended bytes landed under this
      writer's identity (Postgres: version ``expected + 1`` AND the intended
      hash; S3: the token moved AND the observed bytes hash to the intended).
      Adopt the observed ``(bytes, token)`` and STILL drive the coordinator
      bump (:attr:`ReconcileDecision.bump_fires`). Attribution is not claimed —
      a byte-identical concurrent peer is arbitrated by the coordinator's
      version-CAS; the write is "converged", never "landed".
    - ``RE_DRIVE`` — the token is UNMOVED, so the write has not landed. Re-drive
      the identical bytes ONCE, and ONLY under the held token
      (:attr:`ReconcileDecision.re_drive_token`): the possibly-in-flight ghost
      and the re-drive carry the same precondition, so at most one lands.
    - ``RE_DERIVE`` — the write did not land as mine (Postgres: version moved but
      not to ``expected + 1``, or the bytes differ). Re-read fresh, rebuild the
      intended bytes against the current state, and retry.
    - ``CONFLICT`` — the token MOVED and the observed bytes DIFFER: a real peer
      write. Reacquire and re-decide; never re-drive a superseded write.
    - ``HOLD`` — the outcome is unconfirmable (the row/object is absent, or the
      token is an unusable sentinel). Reacquire and re-decide; never a match
      against ``sha256(b"")`` and never an auto re-create. Coordinator state
      must never advance on a HOLD.
    """

    CONVERGE = "converge"
    RE_DERIVE = "re_derive"
    RE_DRIVE = "re_drive"
    CONFLICT = "conflict"
    HOLD = "hold"


# Verdict wording is a closed table (never free text) so a converged write is
# always surfaced as "converged", never "landed" — attribution is disclaimed on
# every convergence path.
_VERDICT_SUMMARY: dict[ReconcileVerdict, str] = {
    ReconcileVerdict.CONVERGE: (
        "converged: the re-read is byte-identical to the intended write, so "
        "adopt the observed token as the comparand and STILL fire the "
        "coordinator bump. Attribution is not claimed — this is convergence, "
        "not confirmation"
    ),
    ReconcileVerdict.RE_DERIVE: (
        "did not land as mine: the token moved but not under my identity (or "
        "the bytes differ) — re-read fresh, rebuild the intended bytes, and retry"
    ),
    ReconcileVerdict.RE_DRIVE: (
        "not confirmed yet: the token is unmoved, so re-drive the identical "
        "bytes under the held token (at most one of the in-flight ghost and the "
        "re-drive wins)"
    ),
    ReconcileVerdict.CONFLICT: (
        "conflict: the token moved and the bytes differ — reacquire and "
        "re-decide; never re-drive a superseded write"
    ),
    ReconcileVerdict.HOLD: (
        "unconfirmed: the row/object or its token is absent — reacquire and "
        "re-decide; never auto re-create (a delete is itself an update)"
    ),
}


@dataclass(frozen=True)
class ReconcileDecision:
    """A binding's verdict after an unconfirmed write, plus what the re-read saw.

    ``observed_bytes`` / ``observed_token`` carry the consistent pair from the
    single reconciliation read (both ``None`` on a ``HOLD``, where the operand
    was absent). ``CONVERGE`` is the ONLY verdict on which the caller settles
    the substrate leg without re-deriving — and even then "did MY write land" is
    left to the coordinator version-CAS, never to a bare content match.

    The three derived properties are what let the cross-agent commit dispatch on
    the ONE decision type regardless of which binding produced it.
    """

    verdict: ReconcileVerdict
    observed_bytes: bytes | None
    observed_token: "SubstrateToken | None"

    @property
    def bump_fires(self) -> bool:
        """Whether the caller must STILL drive the coordinator bump.

        True on ``CONVERGE`` ONLY. Refusing to bump on a converged write (the
        "never converge" mistake) strands the coordinator behind the substrate
        and wedges every peer; firing it on any other verdict would advance the
        version for a write that did not land.
        """
        return self.verdict is ReconcileVerdict.CONVERGE

    @property
    def re_drive_token(self) -> "SubstrateToken | None":
        """The precondition token to re-drive under — set only on ``RE_DRIVE``.

        Equal to the held token (the substrate token is unmoved). Re-driving
        under any other precondition is forbidden; every other verdict is
        ``None``.
        """
        return self.observed_token if self.verdict is ReconcileVerdict.RE_DRIVE else None

    @property
    def summary(self) -> str:
        """Human-facing wording. A converged write says "converged", never
        "landed" — attribution is not claimed."""
        return _VERDICT_SUMMARY[self.verdict]


@runtime_checkable
class CoherenceSubstrate(Protocol):
    """The byte-level surface a substrate binding exposes to the coherence layer.

    Implementations wrap one substrate (a Postgres row, an S3 object) and keep
    ALL content substrate-side; the coherence layer above only ever sees the
    bytes in transit plus the opaque token. A binding for an action backend
    (the ``forward_only`` tier) does not implement this Protocol — an action
    surface has no bytes to read back and no token to compare; it declares a
    descriptor only.
    """

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """The binding's declared capability tier and guarantee metadata.

        Part of the Protocol so a binding cannot satisfy ``isinstance`` while
        omitting its honesty declaration — the conformance suite reads the
        tier from here.
        """
        ...

    def read(self, artifact_ref: str) -> tuple[bytes, SubstrateToken]:
        """Return ``(bytes, token)`` observed by ONE substrate read.

        Both values must come from the same read operation: a token fetched by
        a second call can vouch for bytes it never described, which silently
        loses a concurrent update.
        """
        ...

    def cas_write(
        self,
        artifact_ref: str,
        *,
        expected_token: SubstrateToken,
        new_bytes: bytes,
    ) -> CasWriteResult:
        """Conditionally write ``new_bytes`` keyed on ``expected_token``.

        The comparison is the SUBSTRATE'S atomic conditional write, never a
        client-side check-then-write. Returns a typed outcome: a win carries
        the newly minted token; a conflict means no write landed; an unknown
        outcome means unconfirmed and must be reconciled by re-read.
        """
        ...


# The two-part commit ordering (substrate CAS first, coordinator bump second) is
# load-bearing, and it lives in ONE place: CoordinatedSubstrate below. Bumping
# the coordinator first would advance the version and invalidate peers for a
# write that can still lose the substrate compare-and-set — a phantom
# invalidation, indistinguishable from a real advance and therefore
# unrecoverable. Substrate-first is self-healing: if the coordinator leg is lost,
# the next read reconciles because the token moved.


# ---------------------------------------------------------------------------
# The cross-agent layer (Unit 5): coordinator-mediated pull invalidation + the
# divergence contract, over the SHIPPED coordinator HTTP surface. No coordinator
# file is modified and no route is added — binding reads ride the shipped
# ``/hooks/pre-read`` (SHARED grant) and version bumps ride ``/hooks/post-edit-cas``
# (content_hash ONLY — the never-ship-a-store commit path). The read-generation
# fence is NEITHER wired NOR claimed here: v1 OCC writers sit on the fence's
# admit-on-absent path by design, the same posture as ``CoherentVolume.write_cas``.
# ---------------------------------------------------------------------------


@runtime_checkable
class ReconcilingSubstrate(CoherenceSubstrate, Protocol):
    """A :class:`CoherenceSubstrate` that can also reconcile an unknown write.

    The cross-agent commit needs one method beyond the bytes-I/O surface: after
    a substrate ``CasUnknown`` it asks the binding to re-read and decide (per its
    own token shape) which unified :class:`ReconcileDecision` applies. Both v1
    native-CAS bindings satisfy this structurally.
    """

    def reconcile_after_unknown(
        self,
        artifact_ref: str,
        *,
        expected_token: SubstrateToken,
        intended_hash: str,
    ) -> ReconcileDecision:
        """Re-read the substrate and return the unified reconciliation verdict."""
        ...


def _as_int(value: object, fallback: int) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else fallback


def _maybe_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _pre_read_version(resp: dict) -> int:
    """The coordinator's authoritative version from a pre-read body (fresh or
    stale shape). ``0`` when absent — a CAS against ``expected_version=0`` on a
    seeded artifact loses cleanly, so the fallback never causes a silent write."""
    version = resp.get("version")
    if isinstance(version, int) and not isinstance(version, bool):
        return version
    summary = resp.get("summary")
    if isinstance(summary, dict):
        current = summary.get("current_version")
        if isinstance(current, int) and not isinstance(current, bool):
            return current
    return 0


def _pre_read_denied(resp: dict) -> bool:
    """True on a strict-mode deny (the caller is INVALID, or a fresh reader whose
    bytes diverge from the coordinator's record) — matched on the typed
    ``permissionDecision == "deny"`` signal, never a substring."""
    hook_output = resp.get("hookSpecificOutput")
    return isinstance(hook_output, dict) and hook_output.get("permissionDecision") == "deny"


def _pre_read_prior_seen(resp: dict) -> int | None:
    summary = resp.get("summary")
    if isinstance(summary, dict):
        return _maybe_int(summary.get("prior_version_seen_by_session"))
    return None


def _pre_read_hash_differs(resp: dict) -> bool:
    top = resp.get("hash_differs")
    if isinstance(top, bool):
        return top
    summary = resp.get("summary")
    if isinstance(summary, dict):
        return bool(summary.get("hash_differs"))
    return False


def _deny_is_peer_invalidation(version: int, prior_seen: int | None) -> bool:
    """A deny is a PEER-invalidation (StaleView) when the caller last saw a
    strictly older version — a peer's commit moved the coordinator ahead of it.
    Otherwise the substrate moved out of band of the coordinator
    (coordinator-behind), which is a wedged view (ViewWedged)."""
    return prior_seen is not None and prior_seen < version


def _merge_yaml_list(path: Path, globs: tuple[str, ...]) -> None:
    """Idempotently merge ``globs`` into a coordinator policy YAML list file."""
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
        return
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(yaml.safe_dump(merged, default_flow_style=False), encoding="utf-8")
    os.replace(tmp, path)


def _write_strict_policy(coherence_dir: Path, managed: tuple[str, ...]) -> None:
    """Enable strict mode on the managed artifact globs BEFORE the coordinator
    spawns (policy is load-once-at-startup): a path must be tracked AND strict
    for a peer commit to invalidate this reader's cached view."""
    if not managed:
        return
    # Runs BEFORE the lifecycle creates the directory, so on a fresh workspace
    # this is the creator: match the 0700 the lifecycle requires, or every new
    # workspace spawns with a "tightened existing .coherence directory" warning
    # about a directory this call made a moment earlier. An existing directory
    # is left as-is (mkdir never chmods); re-tightening is the lifecycle's job.
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("tracked.yaml", "strict_mode.yaml"):
        _merge_yaml_list(coherence_dir / name, managed)


@dataclass(frozen=True)
class PreReadResult:
    """What a coordinator pre-read told this session about an artifact."""

    version: int
    stale_denied: bool
    hash_differs: bool
    prior_version_seen: int | None


@dataclass(frozen=True)
class CoordinatorWin:
    """The coordinator committed the version bump; peers are now invalidated."""

    version: int


@dataclass(frozen=True)
class CoordinatorConflict:
    """The coordinator rejected the bump — a peer moved the version first."""

    current_version: int | None


CoordinatorCommit: TypeAlias = "CoordinatorWin | CoordinatorConflict"

# The coordinator commit reasons that mean "a peer won this version" (re-read +
# re-decide), matched EXACTLY against the wire reason — never a substring.
_RETRYABLE_COMMIT_REASONS: frozenset[str] = frozenset(
    {VERSION_MISMATCH_REASON, OCC_CALLER_TRANSIENT_REASON, STALE_READ_GENERATION_REASON}
)


def _classify_commit(resp: dict, expected_version: int) -> CoordinatorCommit:
    """Map a ``/hooks/post-edit-cas`` 200 body to win / conflict, raising the
    typed unknown on the fail-closed degrade body (deny AND degrade both raise —
    an unconfirmed CAS must never read as success).

    A non-win answer with no string ``reason`` is the unknown too. The
    coordinator's own non-win answers always carry one, so it is what a proxy
    or gateway in front of it could send, and it says nothing about whether
    the bump landed. This is the commit leg, reached only after the substrate
    write landed, so it must take the existing unknown path — reconcile by
    re-reading, never re-drive — and never escape as a ``TypeError`` from the
    membership test (an unhashable reason) or read as a rejection."""
    if resp.get("ok") is True:
        return CoordinatorWin(version=_as_int(resp.get("version"), expected_version + 1))
    reason = resp.get("reason")
    if resp.get("degraded") or reason == COMMIT_UNCONFIRMED_REASON:
        raise CommitUnconfirmed(
            "coordinator commit_cas was unconfirmed (degraded); reconcile by "
            "re-reading before retrying — never blind re-drive"
        )
    if not isinstance(reason, str):
        raise CommitUnconfirmed(
            "coordinator commit_cas was answered with no outcome this client can "
            "classify (not a win, and no reason); reconcile by re-reading before "
            "retrying — never blind re-drive"
        )
    if reason in _RETRYABLE_COMMIT_REASONS:
        return CoordinatorConflict(current_version=_maybe_int(resp.get("current_version")))
    if reason.startswith(HANDLER_FAILURE_REASON_PREFIX):
        # The coordinator's failure envelope: the bump's work, or the
        # caller-principal gate's store read, raised. It decides nothing, and
        # a body that raised may have recorded the bump first, so the outcome
        # is unknown, never a rejection: read as one, a caller would retry a
        # write that already landed on the substrate.
        raise CommitUnconfirmed(
            f"coordinator commit_cas was answered with a failure ({reason}); whether "
            "the bump landed is unknown — reconcile by re-reading before retrying, "
            "never blind re-drive"
        )
    raise CoherenceError(f"coordinator commit_cas rejected (fail-closed): {reason}")


class _Sent(NamedTuple):
    """What one POST from :meth:`SubstrateCoordinatorSession._send` yielded: the
    2xx body, or the typed reason of a caller-principal refusal left to the
    caller's recovery (R20). Every other failure raised."""

    body: dict | None
    principal_refusal: str | None


class SubstrateCoordinatorSession:
    """A single agent's client to the SHIPPED coordinator, for a substrate binding.

    Spawns-or-attaches the local-HTTP coordinator for ``root`` (reusing the
    shipped lifecycle) and rides its ``/hooks/pre-read`` + ``/hooks/post-edit-cas``
    routes under ONE stable ``session_id`` (a v4 UUID) used on BOTH legs — the
    same-agent identity a bump-after-read requires. :meth:`reacquire` mints a
    fresh identity (shedding a sticky INVALID); a fresh identity is minted
    NOWHERE else.

    Every coordinator call is fail-closed: a transport error, a non-2xx, or a
    ``{degraded: true}`` body RAISES (deny AND degrade both map to a raise) — the
    client never silently degrades open. A request refused for its caller
    principal is first recovered with the session's own mint nonce (R20); what
    that cannot cure raises the typed
    :class:`~ccs.core.exceptions.CallerPrincipalRefused` on either leg. Two
    sessions over one ``root`` are two distinct agents; the first spawns, the
    rest attach.
    """

    def __init__(
        self,
        root: "str | os.PathLike[str]",
        *,
        managed: tuple[str, ...],
        config: LifecycleConfig | None = None,
        bind_host: str = "127.0.0.1",
    ) -> None:
        # Deferred so importing a binding never pulls in the coordinator server.
        from ccs.adapters.claude_code.lifecycle import (  # noqa: PLC0415
            LifecycleConfig,
            connect_or_spawn,
            read_port_from_file,
            tcp_probe,
        )

        self._root = Path(root).resolve()
        self._managed = tuple(managed)
        cfg = config or LifecycleConfig()
        coherence_dir = self._root / ".coherence"
        pid_file = coherence_dir / "server.pid"
        pre_existing = (port := read_port_from_file(pid_file)) is not None and tcp_probe(
            port, cfg, bind_host=bind_host
        )
        # Only the SPAWNER writes policy (load-once); a sibling must not clobber it.
        if not pre_existing:
            _write_strict_policy(coherence_dir, self._managed)
        if connect_or_spawn(self._root, config=cfg, bind_host=bind_host) == -1:
            raise CoherenceError(
                "substrate coordinator unavailable (could not spawn or attach; fail-closed)"
            )
        try:
            self._endpoint: CoordinatorEndpoint = resolve_endpoint(self._root)
        except CoordinatorUnavailable as exc:
            raise CoherenceError(
                f"substrate coordinator endpoint unresolved (fail-closed): {exc}"
            ) from exc
        session_id, mint_nonce = str(uuid.uuid4()), secrets.token_urlsafe(32)
        principal = self._claim_principal(session_id, mint_nonce)
        self._session_id, self._mint_nonce, self._principal = session_id, mint_nonce, principal

    @property
    def session_id(self) -> str:
        """The stable per-agent coordinator identity (a v4 UUID)."""
        return self._session_id

    @property
    def root(self) -> Path:
        """The coordinator workspace root (for ``stop_coordinator``)."""
        return self._root

    def reacquire(self) -> None:
        """Mint a fresh identity, shedding a sticky INVALID. The ONLY place a new
        identity is minted — read and commit always share one identity between
        them. A new identity is a new session, so it claims its own caller
        principal (the commit route requires one) under its own new nonce.

        The claim runs for the NEW session id BEFORE anything is swapped: only a
        claim that settles replaces the session id, nonce and principal
        together. A claim that raises leaves the previous three exactly as they
        were, so the session keeps working under its old identity rather than
        naming a new session while presenting the old one's principal."""
        session_id, mint_nonce = str(uuid.uuid4()), secrets.token_urlsafe(32)
        principal = self._claim_principal(session_id, mint_nonce)
        self._session_id, self._mint_nonce, self._principal = session_id, mint_nonce, principal

    def _claim_principal(self, session_id: str, mint_nonce: str) -> str | None:
        """Claim ``session_id``'s caller principal presenting ``mint_nonce``. The
        principal and the nonce are held in memory for the session's life: a
        long-lived caller never looks a principal up by the identity it names
        (KTD5) — accident-resistance, not unreadability, since the coordinator
        stores every principal in ``.coherence/state.db``. ``None`` when the
        coordinator issues none (404). A refused or unconfirmed claim RAISES,
        like every other coordinator failure here (fail-closed); nothing is
        re-minted."""
        claim = claim_caller_principal(self._endpoint, session_id, mint_nonce)
        if claim.outcome == "bound":
            return claim.principal
        if claim.outcome == "unsupported":
            return None
        message = (
            f"coordinator caller principal not obtained ({claim.outcome}: "
            f"{claim.detail}; fail-closed)"
        )
        if claim.outcome == "refused":
            raise CallerPrincipalRefused(CALLER_PRINCIPAL_CLAIMED_REASON, message)
        raise CoherenceError(message)

    def pre_read(
        self,
        artifact_ref: str,
        content_hash: str | None,
        *,
        unknown: type[CoherenceError] = CoherenceError,
    ) -> PreReadResult:
        """Register a SHARED view and return the coordinator's version + deny
        state. This is what makes a later peer commit invalidate this reader
        (pull invalidation, surfaced at THIS reader's next binding-mediated act).

        A failure raises ``unknown`` — a plain ``CoherenceError`` for a read,
        ``CommitUnconfirmed`` for a pre-read a commit sends after its
        substrate write landed.
        """
        payload: dict[str, object] = {"session_id": self._session_id, "path": artifact_ref}
        if content_hash is not None:
            payload["content_hash"] = content_hash
        resp = self._post("/hooks/pre-read", payload, unknown=unknown)
        if resp.get("degraded"):
            raise unknown("coordinator watchdog timeout on pre-read (fail-closed)")
        if resp.get("ok") is False:
            # The coordinator's failure envelope, HTTP 200 ``{"ok": false,
            # "reason": "internal: <Type>"}``: its caller-principal gate's store
            # read or the pre-read work body raised. It decides nothing (no
            # version, no deny, no hash comparison), so it is ``unknown`` like
            # an unanswered request, never a clean pre-read: read as one, a
            # commit skipped the deny and bumped from version 0, and a
            # converged commit's hash check read "no difference". The pre-read
            # handler never answers an ``ok`` key otherwise. Only the
            # envelope's reason is named: the exception's type, nothing else.
            reason = resp.get("reason")
            named = reason if isinstance(reason, str) and reason else "no reason given"
            raise unknown(
                f"coordinator answered the pre-read of {artifact_ref!r} with a failure "
                f"({named}); not confirmed (fail-closed)"
            )
        return PreReadResult(
            version=_pre_read_version(resp),
            stale_denied=_pre_read_denied(resp),
            hash_differs=_pre_read_hash_differs(resp),
            prior_version_seen=_pre_read_prior_seen(resp),
        )

    def commit_cas(
        self, artifact_ref: str, *, expected_version: int, content_hash: str
    ) -> CoordinatorCommit:
        """Drive the version bump — ``content_hash`` ONLY, content is NEVER sent
        (the never-ship-a-store commit path). Raises :class:`CommitUnconfirmed`
        on a transport/degrade UNKNOWN (a false-negative ack), returns win /
        conflict otherwise."""
        payload = {
            "session_id": self._session_id,
            "path": artifact_ref,
            "success": True,
            "content_hash": content_hash,
            "expected_version": expected_version,
        }
        resp = self._post("/hooks/post-edit-cas", payload, unknown=CommitUnconfirmed)
        return _classify_commit(resp, expected_version)

    def coordinator_hash_matches(
        self,
        artifact_ref: str,
        content_hash: str,
        *,
        unknown: type[CoherenceError] = CoherenceError,
    ) -> bool:
        """True iff the coordinator's recorded hash equals ``content_hash`` — the
        token-identity signal that a byte-identical peer already carried this
        writer's intended content to the coordinator (so the bump is complete).
        A failure to tell raises ``unknown`` (see :meth:`pre_read`)."""
        return not self.pre_read(artifact_ref, content_hash, unknown=unknown).hash_differs

    def _post(
        self, endpoint_path: str, payload: dict, *, unknown: type[CoherenceError]
    ) -> dict:
        """POST with fail-closed classification. A transport error / non-2xx /
        non-dict body raises ``unknown`` (``CoherenceError`` for a read leg,
        ``CommitUnconfirmed`` for the commit leg) — never a silent degrade-open.

        A refusal of the caller principal is DEFINITE — the coordinator refused
        before any mutation — so it is never ``unknown``. It is recovered as
        R20 describes: the session's SAME nonce claims again, and a principal
        that differs from the one presented is adopted and the request retried
        exactly ONCE (a 404 on the claim retries once without the header).
        What that cannot cure raises
        :class:`~ccs.core.exceptions.CallerPrincipalRefused` with the wire
        reason, on either leg."""
        sent = self._send(endpoint_path, payload, unknown=unknown)
        if sent.principal_refusal is None:
            return sent.body  # type: ignore[return-value]
        reason = sent.principal_refusal
        recovery = decide_principal_recovery(
            claim_caller_principal(self._endpoint, self._session_id, self._mint_nonce),
            self._principal,
        )
        if recovery.action == "retry":
            self._principal = recovery.principal
            sent = self._send(endpoint_path, payload, unknown=unknown)
            if sent.principal_refusal is None:
                return sent.body  # type: ignore[return-value]
            reason, detail = sent.principal_refusal, PRINCIPAL_REFUSED_AGAIN
        else:
            detail = recovery.detail
        # ``settled`` is False only when the recovery claim's answer was lost:
        # nothing was adopted, and this session's next refused request claims
        # again with the retained nonce by itself — so the refusal is not the
        # session's settled state, and a consumer must not report it as one.
        raise CallerPrincipalRefused(
            reason, principal_refusal_message(reason, detail), settled=recovery.settled
        )

    def _send(
        self, endpoint_path: str, payload: dict, *, unknown: type[CoherenceError]
    ) -> _Sent:
        """One POST presenting the current principal. A caller-principal refusal
        is returned for :meth:`_post` to recover; anything else that is not a
        dict body raises ``unknown``.

        A rejected request raises OUTSIDE the ``except`` block, so the
        ``HTTPError`` is not on the raised error's chain: its text is the
        status line's reason phrase — the coordinator's — which a traceback
        would print. This client reports a rejected request by its status
        code only.

        A redirect is a non-2xx like any other, so it raises ``unknown`` too:
        refused and never followed, but whatever answered may have passed the
        request on, so on the commit leg — reached only after the substrate
        write landed — whether the bump landed is not known. Reported by its
        status alone, outside the ``except`` block, so the ``RedirectRefused``
        is not on the raised error's chain either: the refusal withholds the
        ``Location`` today, and this error does not depend on it doing so."""
        redirected = False
        try:
            resp = _coordinator_post(
                self._endpoint,
                endpoint_path,
                payload,
                extra_headers=caller_principal_headers(self._principal),
            )
        except urllib.error.HTTPError as exc:
            status, reason = exc.code, principal_refusal_reason(exc)
        except RedirectRefused as exc:
            status, reason, redirected = exc.status, None, True
        except CoordinatorUnavailable as exc:
            raise unknown(
                f"coordinator {endpoint_path} failed (fail-closed): {exc}"
            ) from exc
        else:
            if not isinstance(resp, dict):
                raise unknown(
                    f"coordinator {endpoint_path} returned a non-dict body (fail-closed)"
                )
            return _Sent(resp, None)
        if redirected:
            raise unknown(
                f"coordinator {endpoint_path} failed (fail-closed): HTTP {status}, "
                "a redirect, which this client never follows"
            )
        if reason is not None:
            return _Sent(None, reason)
        raise unknown(f"coordinator {endpoint_path} failed (fail-closed): HTTP {status}")


@dataclass(frozen=True)
class CommitResult:
    """The outcome of a landed cross-agent commit.

    ``converged`` is True when the write landed via reconciliation of an unknown
    (attribution is not claimed) rather than a clean coordinator win. ``noop`` is
    True when the commit was byte-identical to the last-observed bytes and so
    touched neither the substrate nor the coordinator — the version did not
    advance and no peer was invalidated.
    """

    version: int
    converged: bool
    noop: bool = False

    @property
    def summary(self) -> str:
        """"unchanged" for a byte-identical no-op, "converged" for a reconciled
        write, "committed" for a clean win — never "landed", since a converged
        write's attribution is disclaimed and a no-op landed nothing."""
        if self.noop:
            return "unchanged"
        return "converged" if self.converged else "committed"


@dataclass(frozen=True)
class _PendingCommit:
    """The client-held intent carried across the two commit legs + reconciliation."""

    artifact_ref: str
    expected_token: "SubstrateToken"
    new_bytes: bytes
    expected_version: int
    intended_hash: str


class CoordinatedSubstrate:
    """A substrate binding made coordinator-mediated (the Unit-5 public surface).

    Composes ONE :class:`ReconcilingSubstrate` binding (a Postgres row, an S3
    object, or a scripted fake) with ONE :class:`SubstrateCoordinatorSession`.
    Two instances over the same coordinator ``root`` — each with its own session
    — are two agents sharing one artifact.

    The value over the substrate's bare CAS: a peer's commit invalidates this
    instance's cached read BEFORE it acts (pull invalidation, surfaced as the
    uniform typed :class:`~ccs.core.exceptions.StaleView`), with the same
    typed-conflict vocabulary over a row, an object, a file, or a store key. The
    read-generation fence is NOT part of v1 — OCC writers ride admit-on-absent +
    version-CAS, stated honestly.

    Commit ordering is load-bearing: the substrate CAS runs FIRST, the
    coordinator bump SECOND. A coordinator-bump-first path does not exist — the
    bump is reached only through a confirmed (or reconciled-converged) substrate
    write, so a write that fails the substrate CAS can never invalidate a peer.
    """

    def __init__(
        self,
        substrate: ReconcilingSubstrate,
        coordinator: SubstrateCoordinatorSession,
    ) -> None:
        # never-ship-a-store, made load-bearing: a binding that declares it sends
        # content to the coordinator is refused at composition, so the content-free
        # commit path (commit_cas sends content_hash only) cannot be undercut by a
        # binding that shadows the body coordinator-side.
        if getattr(substrate, "SENDS_CONTENT_TO_COORDINATOR", False):
            raise CoherenceError(
                "substrate binding declares SENDS_CONTENT_TO_COORDINATOR=True; the "
                "coordinator holds a content_hash only (never-ship-a-store) — refused"
            )
        self._substrate = substrate
        self._coordinator = coordinator
        # Per-artifact last-observed content hash (seeded on read), so a commit's
        # pre-read carries the hash the coordinator recorded — a fresh-SHARED
        # holder is never falsely foreign-denied by its own intended hash.
        self._observed_hash: dict[str, str] = {}

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """The binding's honest capability declaration (passthrough)."""
        return self._substrate.descriptor

    @property
    def session_id(self) -> str:
        """This agent's stable coordinator identity."""
        return self._coordinator.session_id

    def read(
        self, artifact_ref: str, *, on_stale: str = "allow"
    ) -> tuple[bytes, "SubstrateToken"]:
        """Read ``(bytes, token)`` from the substrate and register a SHARED view.

        Under ``on_stale="allow"`` (default, R10) a peer-invalidation deny returns
        the current bytes (the instance stays sticky-INVALID); under ``"raise"``
        it surfaces :class:`~ccs.core.exceptions.StaleView`. A coordinator-behind
        deny (the substrate moved out of band) always raises
        :class:`~ccs.core.exceptions.ViewWedged` — the view is wedged, not merely
        stale.
        """
        if on_stale not in ("allow", "raise"):
            raise ValueError(f"on_stale must be 'allow' or 'raise', got {on_stale!r}")
        observed_bytes, token = self._substrate.read(artifact_ref)
        content_hash = _sha256_hex(observed_bytes)
        pre = self._coordinator.pre_read(artifact_ref, content_hash)
        self._observed_hash[artifact_ref] = content_hash
        self._raise_on_deny(artifact_ref, pre, on_stale=on_stale)
        return observed_bytes, token

    def reacquire(self, artifact_ref: str) -> tuple[bytes, "SubstrateToken"]:
        """Shed a sticky deny: mint a fresh identity, then re-read under it
        (SHARED@current). The caller MUST rebuild any pending write from these
        fresh bytes."""
        self._coordinator.reacquire()
        observed_bytes, token = self._substrate.read(artifact_ref)
        content_hash = _sha256_hex(observed_bytes)
        self._coordinator.pre_read(artifact_ref, content_hash)
        self._observed_hash[artifact_ref] = content_hash
        return observed_bytes, token

    def commit(
        self, artifact_ref: str, *, expected_token: "SubstrateToken", new_bytes: bytes
    ) -> CommitResult:
        """Two-part commit: pull-invalidation check, substrate CAS, coordinator bump.

        Step 1 is a coordinator pre-read: if a peer committed since this instance
        read, the instance is INVALID and this raises
        :class:`~ccs.core.exceptions.StaleView` BEFORE the substrate is touched
        (the invalidation headline; a bare CAS never surfaces this). Otherwise the
        substrate CAS runs FIRST and only a confirmed win reaches the coordinator
        bump. Divergence (substrate/coordinator UNKNOWN) is reconciled by
        token-identity — never a blind re-drive, never a stranded converged bump.
        """
        pending = _PendingCommit(
            artifact_ref=artifact_ref,
            expected_token=expected_token,
            new_bytes=new_bytes,
            expected_version=0,
            intended_hash=_sha256_hex(new_bytes),
        )
        pre = self._coordinator.pre_read(artifact_ref, self._observed_hash.get(artifact_ref))
        # A write is a data-loss surface — a deny ALWAYS raises before acting.
        self._raise_on_deny(artifact_ref, pre, on_stale="raise")
        # No-op guard: committing the exact bytes this instance last observed
        # changes nothing. Skip BOTH legs so a byte-identical rewrite never mints a
        # phantom version advance that would invalidate every peer (Open Q C). The
        # deny check above still ran, so a stale no-op surfaces rather than passing.
        if self._observed_hash.get(artifact_ref) == pending.intended_hash:
            return CommitResult(version=pre.version, converged=False, noop=True)
        pending = _replace_expected_version(pending, pre.version)
        outcome = self._substrate.cas_write(
            artifact_ref, expected_token=expected_token, new_bytes=new_bytes
        )
        return self._after_substrate_write(pending, outcome)

    # --- internals ------------------------------------------------------------

    def _raise_on_deny(self, artifact_ref: str, pre: PreReadResult, *, on_stale: str) -> None:
        if not pre.stale_denied:
            return
        if _deny_is_peer_invalidation(pre.version, pre.prior_version_seen):
            if on_stale == "raise":
                raise StaleView(
                    f"a peer commit invalidated the cached view of {artifact_ref!r} "
                    "before this act; reacquire() and act on the fresh state"
                )
            return
        # Coordinator-behind: the substrate token moved but the coordinator did
        # not — a foreign out-of-band write, or a writer crash between the two
        # commit legs. Unbounded in v1 (a peer on an already-cached read is
        # unprotected until its next binding-mediated read); recover via reacquire.
        raise ViewWedged(
            f"the substrate for {artifact_ref!r} moved out of band of the coordinator "
            "(coordinator-behind); reacquire() and re-decide"
        )

    def _after_substrate_write(
        self, pending: _PendingCommit, outcome: CasWriteResult
    ) -> CommitResult:
        if isinstance(outcome, CasWritten):
            return self._drive_bump(pending)
        if isinstance(outcome, CasConflict):
            # No coordinator leg — no write landed; a foreign writer moved the token.
            # current_version is best-effort (no coordinator re-read); recover via
            # reacquire(), not the numeric (see CasVersionConflict).
            raise CasVersionConflict(
                pending.artifact_ref, pending.expected_version, pending.expected_version
            )
        return self._reconcile_unknown(pending)

    def _drive_bump(self, pending: _PendingCommit, *, converged: bool = False) -> CommitResult:
        """The coordinator leg, reached only after the substrate write landed.

        A caller-principal refusal here is reported through the bump leg's
        unknown, not raised bare: the refusal is definite (the refused request
        changed nothing), but the substrate already holds the new bytes, so
        the commit is in exactly the state Case 1 leaves — and the refusal's
        own recovery, resending a request that "changed nothing", would
        re-drive a landed write. ``CommitUnconfirmed`` carries the recovery
        that is right (re-read; retry only if absent; never re-drive), its
        message says what happened, and the typed refusal is its cause. A TLS
        verification failure — definite too: the request was never sent — is
        reported the same way.

        Each message names what this write did not do, never the version: on
        a converged write a byte-identical peer's bump may have advanced it,
        and the refused request is then the pre-read that checks for it. Nor
        does a converged write claim it "landed": attribution is disclaimed
        (see :class:`ReconcileVerdict`)."""
        landed = _landed_clause(pending.artifact_ref, converged=converged)
        try:
            return self._bump(pending, converged=converged)
        except CallerPrincipalRefused as refusal:
            raise CommitUnconfirmed(
                f"{landed}, but the coordinator refused the caller principal of a "
                f"request this commit sent ({refusal.reason}), so it recorded no bump "
                "from this write, and this write invalidated no peer. Not re-driven; "
                "re-read before retrying."
            ) from refusal
        except TlsVerificationFailed as failure:
            raise CommitUnconfirmed(
                f"{landed}, but a request this commit made to the coordinator failed "
                "TLS certificate verification, so it was never sent, and the "
                "coordinator recorded no bump from this write. Not re-driven; re-read "
                "before retrying."
            ) from failure

    def _bump(self, pending: _PendingCommit, *, converged: bool) -> CommitResult:
        try:
            result = self._coordinator.commit_cas(
                pending.artifact_ref,
                expected_version=pending.expected_version,
                content_hash=pending.intended_hash,
            )
        except CommitUnconfirmed:
            # Case 1: the substrate write is durable but the coordinator leg is
            # unknown. NEVER blind re-drive (it would conflict against the moved
            # token). Surface the false-negative ack — the caller re-reads.
            raise
        if isinstance(result, CoordinatorWin):
            self._observed_hash[pending.artifact_ref] = pending.intended_hash
            return CommitResult(version=result.version, converged=converged)
        return self._resolve_bump_conflict(pending, result, converged=converged)

    def _resolve_bump_conflict(
        self, pending: _PendingCommit, result: CoordinatorConflict, *, converged: bool
    ) -> CommitResult:
        # A converged write whose bump lost to a byte-identical peer is COMPLETE
        # if the coordinator already holds the intended hash — no re-drive, no
        # second bump (that would be a phantom advance). The substrate write has
        # landed, so a check that cannot tell is the bump leg's unknown.
        if converged and self._coordinator.coordinator_hash_matches(
            pending.artifact_ref, pending.intended_hash, unknown=CommitUnconfirmed
        ):
            self._observed_hash[pending.artifact_ref] = pending.intended_hash
            return CommitResult(
                version=result.current_version or pending.expected_version, converged=True
            )
        raise CasVersionConflict(
            pending.artifact_ref,
            pending.expected_version,
            result.current_version or pending.expected_version,
        )

    def _reconcile_unknown(self, pending: _PendingCommit) -> CommitResult:
        decision = self._substrate.reconcile_after_unknown(
            pending.artifact_ref,
            expected_token=pending.expected_token,
            intended_hash=pending.intended_hash,
        )
        if decision.verdict is ReconcileVerdict.HOLD:
            raise ViewWedged(
                f"unconfirmed substrate write to {pending.artifact_ref!r}: the operand "
                "is absent — reacquire() and re-decide (never auto re-create)"
            )
        if decision.bump_fires:  # CONVERGE — the intended bytes landed; drive the bump.
            return self._drive_bump(pending, converged=True)
        if decision.re_drive_token is not None:  # RE_DRIVE — token unmoved, retry once.
            return self._re_drive(pending, decision.re_drive_token)
        # RE_DERIVE / CONFLICT — the write did not land as mine; reacquire + re-decide.
        # current_version is best-effort (no coordinator re-read); recover via
        # reacquire(), not the numeric (see CasVersionConflict).
        raise CasVersionConflict(
            pending.artifact_ref, pending.expected_version, pending.expected_version
        )

    def _re_drive(self, pending: _PendingCommit, token: "SubstrateToken") -> CommitResult:
        retry = self._substrate.cas_write(
            pending.artifact_ref, expected_token=token, new_bytes=pending.new_bytes
        )
        if isinstance(retry, CasWritten):
            return self._drive_bump(pending)
        if isinstance(retry, CasConflict):
            # The token moved between the reconcile read and this re-drive. Either
            # MY own in-flight ghost put landed (converge — complete the bump) or a
            # peer superseded me (a real conflict). Reconcile ONCE more to tell them
            # apart rather than raising a misleading conflict on my own write. Only
            # CONVERGE is acted on here; any other verdict is surfaced as a conflict,
            # so this cannot loop back into another re-drive.
            decision = self._substrate.reconcile_after_unknown(
                pending.artifact_ref,
                expected_token=pending.expected_token,
                intended_hash=pending.intended_hash,
            )
            if decision.bump_fires:  # CONVERGE — my ghost carried my bytes; complete it.
                return self._drive_bump(pending, converged=True)
            # current_version is best-effort here — no coordinator re-read; the
            # caller recovers via reacquire(), not the numeric (see CasVersionConflict).
            raise CasVersionConflict(
                pending.artifact_ref, pending.expected_version, pending.expected_version
            )
        # A SECOND unknown — surface rather than loop unbounded (fail-closed).
        raise CommitUnconfirmed(
            f"re-drive of {pending.artifact_ref!r} is again unconfirmed; reconcile by "
            "re-reading before retrying"
        )


def _landed_clause(artifact_ref: str, *, converged: bool) -> str:
    """How a commit's report names the substrate write once it is on the
    substrate: "landed" for a confirmed write, and for a converged one only
    that the substrate holds the intended bytes — a byte-identical peer may
    have put them there."""
    if converged:
        return (
            f"the substrate holds the bytes this write intended for {artifact_ref!r} "
            "(converged: which write put them there is not claimed)"
        )
    return f"the substrate write to {artifact_ref!r} landed"


def _replace_expected_version(pending: _PendingCommit, version: int) -> _PendingCommit:
    return _PendingCommit(
        artifact_ref=pending.artifact_ref,
        expected_token=pending.expected_token,
        new_bytes=pending.new_bytes,
        expected_version=version,
        intended_hash=pending.intended_hash,
    )
