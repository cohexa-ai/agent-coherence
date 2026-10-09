# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Shared HTTP client helpers for the four agent-coherence-* console scripts.

The coordinator binds to 127.0.0.1 with shared-secret Bearer auth
(KTD-12). Each console script needs:

1. Resolve the workspace root.
2. Read the port from ``<root>/.coherence/server.pid``.
3. Read the bearer secret from ``<root>/.coherence/hook.secret``.
4. Make an authenticated request with a short timeout.

Failures degrade gracefully — these scripts run interactively and should
print a one-line human message + exit 2 rather than dump a stack trace.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import logging
import os
import ssl
import sys
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ccs.adapters.claude_code.auth import (
    CALLER_PRINCIPAL_HEADER,
    MintNonceUnavailable,
    ensure_mint_nonce,
    load_caller_principal,
    load_mint_nonce,
    store_caller_principal,
)
from ccs.adapters.claude_code.coordinator_server import (
    caller_principal_identity,
    validate_session_id,
)
from ccs.adapters.claude_code.lifecycle import read_port_from_file as _read_port_from_file
from ccs.core.exceptions import (
    CALLER_PRINCIPAL_CLAIMED_REASON,
    CALLER_PRINCIPAL_REASONS,
    CALLER_PRINCIPAL_REFUSAL_REASONS,
    CallerPrincipalRefused,
    InsecureTransportRefused,
    RedirectRefused,
    TlsConfigError,
    TlsVerificationFailed,
)

logger = logging.getLogger(__name__)

#: HTTP timeout for CLI requests. The coordinator's per-request watchdog is
#: 4s; we add headroom for connection setup. CLI users are interactive so a
#: 6s ceiling is preferable to retrying.
CLI_HTTP_TIMEOUT_SEC = 6.0


def err(message: str) -> None:
    """Write a diagnostic / error line to stderr.

    P2 ce-review fix #15 (cli-readiness): error output must go to stderr so
    agents composing workflows can read machine-parseable data from stdout
    (e.g., ``port=$(agent-coherence-coordinator)``) without prose pollution.
    Success output stays on stdout via plain ``print()``.
    """
    print(message, file=sys.stderr, flush=True)


def escape_nonprintable(value: object) -> str:
    """``value`` as text safe to print to a terminal: each character
    :meth:`str.isprintable` accepts is kept, and any other is written as
    ``repr`` writes it, without the quotes (ESC as ``\\x1b``, LF as ``\\n``,
    U+202E as ``\\u202e``). A non-string is stringified first.

    Every string a console script prints from a coordinator answer goes
    through this (#245): an answer could otherwise drive the terminal with an
    escape sequence, reorder a line with a bidi control, or crash the print on
    a lone surrogate. Printable text, the ASCII space included, prints
    unchanged, so a plain path's line stays byte-identical; other spaces and
    joiners print escaped."""
    text = value if isinstance(value, str) else str(value)
    return "".join(c if c.isprintable() else repr(c)[1:-1] for c in text)


def validate_relative_path(p: str) -> str | None:
    """Client-side path pre-check before sending to the coordinator.

    Reject absolute paths, ``..`` traversal, and empty input. Returns
    None on valid, a reason string on invalid.

    M-02 layer-distinction note: this is the CLI-side check (light;
    designed for friendly operator error messages before the request
    is built). The server-side check is
    :func:`ccs.adapters.claude_code.coordinator_server.validate_path`
    — STRICTER (also rejects backslash-leading paths, control characters,
    paths longer than MAX_PATH_LEN, non-string types). The server-side
    check is the authoritative gate; this CLI check is for fast feedback
    without a coordinator round-trip. Do NOT remove either — they live
    at different layers of the trust boundary.

    P2 ce-review fix #6 (maintainability): consolidates the
    ``_validate_path`` helper that previously existed byte-for-byte in
    both coherence_track.py and coherence_untrack.py — divergence risk
    eliminated. Both scripts now import this single source of truth.

    Note: this is a PURE-STRING validator. Callers that want to accept
    absolute paths inside the workspace (operator-UX path, exposed via
    ``/agent-coherence:track`` skill template that passes ``$ARGUMENTS``
    verbatim) should call :func:`normalize_workspace_path` instead, which
    handles absolute-vs-relative + workspace-containment before delegating
    to this function."""
    if not p:
        return "empty"
    if p.startswith("/"):
        return "path must be relative (no leading '/')"
    if ".." in Path(p).parts:
        return "path must not contain '..' traversal"
    return None


def normalize_workspace_path(p: str, root: Path) -> tuple[str, str | None]:
    """Normalize a CLI path argument to workspace-relative form.

    Returns ``(normalized_path, None)`` if the path is valid; returns
    ``(original_path, reason_string)`` if invalid. Accepts both relative
    and absolute paths:

    - **Empty** → ``("", "empty")``
    - **Relative** (e.g., ``"docs/plan.md"``) → validated as-is via
      :func:`validate_relative_path`; returned unchanged on success.
    - **Absolute and inside workspace root** (e.g.,
      ``"/Users/x/repo/docs/plan.md"`` with ``root=/Users/x/repo``) →
      stripped to workspace-relative (``"docs/plan.md"``); re-validated
      for ``..`` traversal defense.
    - **Absolute and outside workspace root** (e.g., ``"/etc/passwd"``)
      → rejected with ``"path outside workspace root"``.

    This helper exists because the Claude Code plugin's
    ``/agent-coherence:track`` skill template substitutes ``$ARGUMENTS``
    verbatim — operators routinely type absolute paths (autocomplete from
    their shell or IDE). Pre-2026-05-26 the CLI rejected those outright;
    this helper normalizes them so the operator UX matches the skill UX.

    The normalized form is what gets written to ``tracked.yaml`` /
    ``ignored.yaml`` — absolute paths must NEVER leak into those files
    because they're per-machine / per-worktree and would break cross-host
    state sharing if the coordinator-backed state.db is ever migrated.

    M-02 trust-boundary note: the server-side validator
    (:func:`ccs.adapters.claude_code.coordinator_server.validate_path`)
    still independently rejects absolute paths in the coordinator
    request body. This client-side normalization happens BEFORE the
    request is built — by the time the request hits the wire, the path
    is workspace-relative. The server check remains the authoritative
    gate against malformed direct-HTTP calls that bypass this CLI.
    """
    if not p:
        return p, "empty"
    if Path(p).is_absolute():
        try:
            normalized = str(Path(p).resolve().relative_to(root.resolve()))
        except ValueError:
            return p, "path outside workspace root"
        # Re-validate the normalized form against the pure-string rules
        # (catches e.g. a resolved path that still contains '..' — defensive)
        reason = validate_relative_path(normalized)
        return (normalized, None) if reason is None else (p, reason)
    reason = validate_relative_path(p)
    return (p, None) if reason is None else (p, reason)


@dataclass(frozen=True)
class CoordinatorEndpoint:
    """Resolved (host, port, bearer_token) for a coordinator.

    ``host`` defaults to loopback so the local path is byte-unchanged; the
    cross-host demo (gated by :class:`RemoteCoordinatorConfig`) supplies a
    routable host via :func:`resolve_remote_endpoint`.

    ``scheme`` defaults to ``"http"`` (the loopback path is byte-unchanged);
    ``"https"`` selects verified TLS — :func:`_execute` builds a hardened
    context via :func:`build_tls_context` (there is no insecure ``https`` mode).
    ``ca_file`` optionally names an exclusive private-CA bundle for that
    context; it is validated and read once at request time (see
    :func:`build_tls_context`).
    """

    port: int
    bearer: str
    host: str = "127.0.0.1"
    scheme: str = "http"
    ca_file: str | None = None

    @property
    def base_url(self) -> str:
        # Bracket an IPv6 literal so the authority parses: http://[::1]:8080, not
        # the ambiguous http://::1:8080 (urllib reads the last colon as the port
        # separator). A hostname / IPv4 raises ValueError -> no brackets. The
        # bracketing branch fires for BOTH schemes (https://[::1]:8443 too).
        authority = self.host
        try:
            if ipaddress.ip_address(self.host).version == 6:
                authority = f"[{self.host}]"
        except ValueError:
            pass
        return f"{self.scheme}://{authority}:{self.port}"


class CoordinatorUnavailable(Exception):
    """Coordinator is not running or its auth surface is missing.

    Carries a human-readable message the console script prints verbatim.
    """


def _read_ca_bundle(ca_file: str) -> str:
    """Read a private-CA PEM bundle with the same discipline as ``_read_secret``.

    A CA bundle is a *trust anchor* — a symlink swap or a writable file between
    check and use is the same attack class as a swapped bearer file, so we open
    with ``O_NOFOLLOW`` (refuse symlinks), reject a group/world-*writable* file
    (the ``0o022`` bits — readable is fine, certs are public), and read ONCE via
    the fd (no path re-open → no TOCTOU). The bytes are handed to the SSL context
    as ``cadata`` so the path is never re-opened.

    Any failure (missing / unreadable / symlink / loose perms) is normalized to
    a :class:`~ccs.core.exceptions.TlsConfigError` naming the path — never a raw
    ``OSError`` / ``ssl.SSLError`` leaking out.
    """
    try:
        fd = os.open(ca_file, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        # Missing, or a symlink (ELOOP), or otherwise unopenable — fail closed.
        raise TlsConfigError(
            f"CCS_REMOTE_CA_FILE {ca_file!r} could not be opened "
            f"({exc.strerror or exc}); it must be a regular, readable PEM file "
            "(symlinks are refused)",
            path=ca_file,
        ) from exc
    try:
        with os.fdopen(fd, encoding="utf-8") as handle:
            mode = os.fstat(handle.fileno()).st_mode
            # Only the WRITABLE bits are the attack (a swapped trust anchor);
            # a public cert may be group/world-readable.
            if mode & 0o022:
                raise TlsConfigError(
                    f"CCS_REMOTE_CA_FILE {ca_file!r} is group/world-writable "
                    f"(mode {mode & 0o777:o}); tighten it so the trust anchor "
                    "cannot be swapped (e.g. 0644)",
                    path=ca_file,
                )
            return handle.read()
    except UnicodeDecodeError as exc:
        raise TlsConfigError(
            f"CCS_REMOTE_CA_FILE {ca_file!r} is not a text PEM file ({exc})",
            path=ca_file,
        ) from exc
    except OSError as exc:
        raise TlsConfigError(
            f"CCS_REMOTE_CA_FILE {ca_file!r} could not be read ({exc.strerror or exc})",
            path=ca_file,
        ) from exc


def build_tls_context(ca_file: str | None = None) -> ssl.SSLContext:
    """Build the ONE verified-TLS context for the coordinator client.

    This is the single mTLS-forward-compat choke point: a later mTLS phase adds
    ``load_cert_chain`` config keys here and nowhere else. There is deliberately
    NO cert-verification off-switch — ``CERT_NONE`` / ``check_hostname=False`` is
    unrepresentable through any parameter (the footgun rule).

    - ``create_default_context`` (secure defaults on 3.11+: ``CERT_REQUIRED`` +
      ``check_hostname`` + ``VERIFY_X509_STRICT``); with ``cadata`` from an
      exclusive private-CA bundle when ``ca_file`` is given, else the system
      trust store.
    - The TLS floor is pinned to 1.2 explicitly (do not rely on the default).
    - ``OP_NO_RENEGOTIATION`` is intentionally NOT set (mTLS forward-compat).
    - Hardening invariant asserted: if ``check_hostname`` or ``CERT_REQUIRED``
      were ever weakened by a future edit, this raises :class:`TlsConfigError`
      rather than silently shipping an insecure client.
    """
    cadata = _read_ca_bundle(ca_file) if ca_file else None
    try:
        ctx = ssl.create_default_context(cadata=cadata)
    except ssl.SSLError as exc:
        # e.g. cadata present but not valid PEM — normalize to a typed config error.
        raise TlsConfigError(
            f"CCS_REMOTE_CA_FILE {ca_file!r} does not contain a valid PEM certificate "
            f"({exc})",
            path=ca_file,
        ) from exc

    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    # IP-literal endpoints: OpenSSL matches IP SANs natively; disabling the
    # legacy CN fallback keeps verification to SAN-only (no effect on the SAN
    # match, tightens the no-SAN case). Harmless where the attribute is absent.
    if hasattr(ctx, "hostname_checks_common_name"):
        ctx.hostname_checks_common_name = False

    # Invariant: verification MUST be enforced. This is the assertion the guard's
    # positive signal (Unit 2) relies on — https means enforced verification.
    if not (ctx.check_hostname and ctx.verify_mode == ssl.CERT_REQUIRED):
        raise TlsConfigError(
            "internal error: the TLS context is not enforcing certificate "
            "verification (check_hostname/CERT_REQUIRED invariant violated)"
        )
    return ctx


def resolve_endpoint(coordinator_root: Path) -> CoordinatorEndpoint:
    """Read port + secret from ``<root>/.coherence/`` or raise
    :class:`CoordinatorUnavailable` with an operator-friendly message."""
    coherence_dir = coordinator_root / ".coherence"
    pid_file = coherence_dir / "server.pid"
    secret_file = coherence_dir / "hook.secret"

    port = _read_port_from_file(pid_file)
    if port is None:
        raise CoordinatorUnavailable(
            "no coordinator running for this workspace "
            f"(no port in {pid_file}); start one with `agent-coherence-coordinator`"
        )

    if not secret_file.is_file():
        raise CoordinatorUnavailable(
            "coordinator authentication unavailable "
            f"(missing {secret_file}); restart with `agent-coherence-coordinator`"
        )

    try:
        bearer = secret_file.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise CoordinatorUnavailable(
            f"could not read {secret_file}: {exc}"
        ) from exc

    if not bearer:
        raise CoordinatorUnavailable(
            f"{secret_file} is empty; restart with `agent-coherence-coordinator`"
        )

    return CoordinatorEndpoint(port=port, bearer=bearer)


#: Truthy env values that enable cross-host remote mode (mirrors the
#: telemetry kill-switch parser). Everything else (incl. "" and "0") is OFF.
_REMOTE_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})


def _is_loopback_transport_host(host: str) -> bool:
    """True for hosts that may receive a bearer over plaintext HTTP WITHOUT an ack.

    Broader than the Host-allowlist :func:`~ccs.adapters.claude_code.auth.is_loopback_host`:
    covers ``127.0.0.0/8`` and ``::1`` (via :mod:`ipaddress`) plus ``"localhost"``.
    IPv4-mapped IPv6 forms (``::ffff:127.0.0.1``) are treated as NON-loopback and
    require the ack: ``is_loopback`` for the mapped form varies across CPython patch
    releases, so we classify it deterministically as non-loopback rather than depend
    on the stdlib version (fail-closed either way — genuine local dev uses
    ``127.0.0.1`` / ``::1``). A non-IP hostname (a name we cannot classify) is
    likewise NON-loopback, so the guard fails closed on it.
    """
    if host == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # IPv4-mapped IPv6 (::ffff:a.b.c.d): fail closed deterministically (see above).
    if getattr(ip, "ipv4_mapped", None) is not None:
        return False
    return ip.is_loopback


def _guard_plaintext_bearer(host: str, env: Mapping[str, str], scheme: str = "http") -> None:
    """Fail closed on a plaintext bearer to a non-loopback host (Phase-1.5 guard).

    ``scheme == "https"`` is the verified-TLS positive signal (Unit 2): in THIS
    client ``https`` ALWAYS means enforced certificate verification
    (:func:`build_tls_context` has no insecure-https mode — the footgun rule), so
    the bearer rides an in-band-trusted link. The guard passes at mint with NO ack
    and NO warning, short-circuiting BEFORE the ack branch — the ack is irrelevant
    when verification is enforced (an ``https`` endpoint with ``CCS_REMOTE_INSECURE``
    also set emits no plaintext warning). This retires the permanent-
    ``CCS_REMOTE_INSECURE=1`` wart for TLS-fronted deployments. If that "https ⇒
    verified" invariant ever weakened, this positive signal would regress — it is
    asserted in :func:`build_tls_context`.

    The ``http`` path is byte-unchanged. The remote transport there is plaintext
    (encryption is operator-provided out-of-band — WireGuard or a TLS-terminating
    proxy), so there is no in-band TLS signal. For a non-loopback host the operator
    must set ``CCS_REMOTE_INSECURE`` (truthy) to acknowledge the link is secured, or
    the bearer is never sent (:class:`InsecureTransportRefused`). Reduces-not-
    eliminates: it removes the SILENT plaintext-bearer footgun, not the operator's
    duty to secure the link.
    """
    if scheme == "https":
        return
    if _is_loopback_transport_host(host):
        return
    if env.get("CCS_REMOTE_INSECURE", "").strip().lower() in _REMOTE_TRUTHY_ENV_VALUES:
        # Names host/posture only — never the bearer/secret value.
        logger.warning(
            "sending a bearer to non-loopback host %r over plaintext HTTP "
            "(CCS_REMOTE_INSECURE acknowledged — ensure the link is encrypted)",
            host,
        )
        return
    raise InsecureTransportRefused(host)


def resolve_remote_endpoint(
    host: str,
    port: int,
    secret: str,
    *,
    scheme: str = "http",
    ca_file: str | None = None,
    env: Mapping[str, str] | None = None,
) -> CoordinatorEndpoint:
    """Build an endpoint for a REMOTE coordinator (cross-host demo).

    Unlike :func:`resolve_endpoint`, this reads nothing from the local
    ``.coherence/`` directory — host, port, and the bearer secret are supplied
    by the caller (typically via :meth:`RemoteCoordinatorConfig.from_env`).
    Gated by :class:`RemoteCoordinatorConfig`.

    ``scheme`` (default ``"http"``) and ``ca_file`` (default ``None``) are
    additive: they thread the verified-TLS surface onto the endpoint without
    changing any existing caller. ``scheme="https"`` selects the verified-TLS
    request path in :func:`_execute`.

    Fail-closed transport guard: an ``https`` endpoint (verified TLS — there is no
    insecure ``https`` mode) passes at mint with NO ack; for a plaintext ``http``
    NON-loopback host the bearer is only sent when ``CCS_REMOTE_INSECURE`` (read
    from ``env``, default ``os.environ``) is truthy — otherwise
    :class:`InsecureTransportRefused` is raised (the ack acknowledges an
    out-of-band-secured link). Loopback ``http`` is byte-unchanged. See
    :func:`_guard_plaintext_bearer` for the verified-TLS positive signal (Unit 2).
    """
    if not host:
        raise CoordinatorUnavailable("remote coordinator host is empty")
    if not secret:
        raise CoordinatorUnavailable("remote coordinator bearer secret is empty")
    _guard_plaintext_bearer(host, os.environ if env is None else env, scheme)
    return CoordinatorEndpoint(
        port=port, bearer=secret, host=host, scheme=scheme, ca_file=ca_file
    )


@dataclass(frozen=True)
class RemoteCoordinatorConfig:
    """Default-OFF gate for cross-host remote-coordinator mode.

    Absent the ``CCS_REMOTE_COORDINATOR`` env flag, :meth:`from_env` returns a
    disabled config and every existing loopback-only behavior is byte-unchanged
    — the cross-host relaxation never reaches local users.

    Secret channel: the bearer is read from a FILE whose path is given by
    ``CCS_REMOTE_SECRET_FILE`` — never inline in an env var, which would leak in
    ``ps`` / ``docker inspect`` (the R7 security pass owns this). The file
    mirrors the local ``hook.secret`` (mode 0600, mounted into the remote
    container).
    """

    enabled: bool
    host: str | None = None
    port: int | None = None
    secret: str | None = None
    #: ``"https"`` when ``CCS_REMOTE_TLS`` is truthy, else ``"http"`` (default).
    #: Selects verified TLS on the resolved endpoint (Unit 1).
    scheme: str = "http"
    #: Optional private-CA bundle PATH from ``CCS_REMOTE_CA_FILE`` (file-not-inline,
    #: mirroring ``CCS_REMOTE_SECRET_FILE``). Validated/read at request time.
    ca_file: str | None = None

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> RemoteCoordinatorConfig:
        """Parse the cross-host flag from the environment (default OFF)."""
        env = os.environ if env is None else env
        flag = env.get("CCS_REMOTE_COORDINATOR", "").strip().lower()
        if flag not in _REMOTE_TRUTHY_ENV_VALUES:
            return cls(enabled=False)
        host = (env.get("CCS_REMOTE_HOST") or "").strip() or None
        port_raw = (env.get("CCS_REMOTE_PORT") or "").strip()
        port = int(port_raw) if port_raw.isdigit() else None
        if port is not None and not (1 <= port <= 65535):
            port = None  # out of TCP range -> treat as unset (fail closed downstream)
        tls = env.get("CCS_REMOTE_TLS", "").strip().lower() in _REMOTE_TRUTHY_ENV_VALUES
        ca_file = (env.get("CCS_REMOTE_CA_FILE") or "").strip() or None
        return cls(
            enabled=True,
            host=host,
            port=port,
            secret=cls._read_secret(env),
            scheme="https" if tls else "http",
            ca_file=ca_file,
        )

    @staticmethod
    def _read_secret(env: dict[str, str]) -> str | None:
        """Read the bearer from ``CCS_REMOTE_SECRET_FILE`` (not an inline env var).

        Hardened: refuses to follow a symlinked secret file (``O_NOFOLLOW`` — an
        attacker able to set the env var could otherwise repoint it at any
        readable file), and warns if the file is group/world-accessible (``0600``
        expected, like the local ``hook.secret``). Fails closed (returns ``None``)
        on any error.
        """
        secret_path = (env.get("CCS_REMOTE_SECRET_FILE") or "").strip()
        if not secret_path:
            return None
        try:
            fd = os.open(secret_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except OSError:
            return None  # missing, or a symlink (ELOOP) — fail closed
        try:
            with os.fdopen(fd, encoding="utf-8") as handle:
                mode = os.fstat(handle.fileno()).st_mode
                if mode & 0o077:
                    logger.warning(
                        "CCS_REMOTE_SECRET_FILE %s is group/world-accessible (mode %o); "
                        "tighten it to 0600",
                        secret_path,
                        mode & 0o777,
                    )
                secret = handle.read().strip()
        except OSError:
            return None
        return secret or None


def get(
    endpoint: CoordinatorEndpoint,
    path: str,
    *,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Authenticated GET. Raises :class:`CoordinatorUnavailable` on network
    error; raises :class:`urllib.error.HTTPError` for non-2xx so the caller
    can format status codes explicitly.

    R12 (Unit 6): ``extra_headers`` lets local-operator CLIs (e.g.,
    ``agent-coherence-status``) add ``Coherence-Local-Operator: true``
    for the elevated ``/status?detail=full`` tier without hard-coding
    that header here."""
    headers: dict[str, str] = {
        "Authorization": f"Bearer {endpoint.bearer}",
        "Host": endpoint.host,
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url=f"{endpoint.base_url}{path}",
        method="GET",
        headers=headers,
    )
    # Carry the CA bundle to _execute (only consulted for https requests).
    req._ccs_ca_file = endpoint.ca_file  # type: ignore[attr-defined]
    return _execute(req)


def post(
    endpoint: CoordinatorEndpoint,
    path: str,
    body: dict[str, Any],
    *,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Authenticated POST with JSON body.

    M-04 / finding #28: ``extra_headers`` mirrors the pattern on ``get()``
    so callers can add e.g. ``Coherence-Local-Operator: true`` without
    reimplementing the urllib transport layer.
    """
    payload = json.dumps(body).encode("utf-8")
    headers: dict[str, str] = {
        "Authorization": f"Bearer {endpoint.bearer}",
        "Host": endpoint.host,
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)
    req = urllib.request.Request(
        url=f"{endpoint.base_url}{path}",
        data=payload,
        method="POST",
        headers=headers,
    )
    # Carry the CA bundle to _execute (only consulted for https requests).
    req._ccs_ca_file = endpoint.ca_file  # type: ignore[attr-defined]
    return _execute(req)


# ---------------------------------------------------------------------------
# Caller principal — obtaining one and presenting it (caller-principal plan, U5)
# ---------------------------------------------------------------------------

PRINCIPAL_CLAIM_ROUTE = "/principal/claim"

CLAIM_UNCONFIRMED_REASON = "claim_unconfirmed"
"""The ``reason`` of the coordinator's watchdog-degraded claim envelope — a
twin of ``coordinator_server._PRINCIPAL_CLAIM_DEGRADED_RESPONSE``, pinned equal
by a test."""

REPORTABLE_CLAIM_REASONS: frozenset[str] = frozenset(
    {*CALLER_PRINCIPAL_REASONS, CLAIM_UNCONFIRMED_REASON}
)
"""The claim and refusal reasons a client repeats in what it reports. Anything
else the coordinator put in a reason field is reported as
:data:`UNRECOGNISED_REASON`."""

UNRECOGNISED_REASON = "unrecognised"


def reportable_reason(value: object) -> str:
    """``value`` when it is a reason in :data:`REPORTABLE_CLAIM_REASONS`, else
    :data:`UNRECOGNISED_REASON`. What a client reports on its claim and
    recovery paths is built from constants and these tokens, never from
    coordinator-supplied text, so a coordinator that echoed a nonce or a
    principal into a reason field cannot get it into a message."""
    if isinstance(value, str) and value in REPORTABLE_CLAIM_REASONS:
        return value
    return UNRECOGNISED_REASON


@dataclass(frozen=True)
class PrincipalClaim:
    """What one ``POST /principal/claim`` established.

    - ``bound``: ``principal`` is the value bound to the session — minted now,
      or handed back to a retry presenting the binding's own nonce (R20).
    - ``unsupported``: the coordinator answered 404 — it issues no principals
      (the sibling Node coordinator, or an older Python one). Proceed without
      one; there is nothing to present.
    - ``refused``: the session is already bound under a DIFFERENT mint nonce.
      The caller does not become that identity and must not try to: deleting a
      stored nonce and claiming again would reopen the gate first-claim-wins
      closes (KTD11).
    - ``unconfirmed``: anything else — a watchdog-degraded claim, a transport
      failure, an unexpected answer. A claim that landed anyway is recovered by
      the next claim presenting the SAME nonce: the one-shot client's next
      invocation, or the long-lived client's claim before its next request
      (R20).

    ``detail`` is diagnostic prose built from constants, known reason tokens,
    an HTTP status code or this client's own transport message — never from
    the coordinator's answer — so it carries no principal or nonce."""

    outcome: Literal["bound", "unsupported", "refused", "unconfirmed"]
    principal: str | None = None
    detail: str = ""


def claim_caller_principal(
    endpoint: CoordinatorEndpoint, session_id: str, mint_nonce: str
) -> PrincipalClaim:
    """Claim ``session_id``'s caller principal, presenting ``mint_nonce``.

    Transport-shaped failures — an unreachable coordinator, and a malformed
    answer such as a non-HTTP status line or a truncated body — come back as
    ``unconfirmed`` rather than raising. So does a redirect: it is refused,
    never followed, and only a 2xx answer carries the claim contract, so like
    any other status outside it the claim is ``unconfirmed`` — the Node client
    decides it the same way, and a one-shot client then sends its request
    without a principal instead of dropping it. Its ``detail`` names the
    status, never the ``Location``. A TLS verification failure, the other
    typed trust refusal, still raises exactly as it does from :func:`post`.
    The coordinator's ``reason`` is repeated in ``detail`` only when it is a
    known token (:func:`reportable_reason`)."""
    try:
        body = post(
            endpoint,
            PRINCIPAL_CLAIM_ROUTE,
            {"session_id": session_id, "mint_nonce": mint_nonce},
        )
    except RedirectRefused as exc:
        return PrincipalClaim(
            "unconfirmed", detail=f"claim answered HTTP {exc.status}, a redirect; not followed"
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return PrincipalClaim("unsupported", detail="the coordinator issues no caller principals")
        # The status and the reason as a known token, as for a 2xx answer and
        # as the Node client reports it. Only a 2xx body carries the claim
        # contract, so a ``caller_principal_claimed`` here is reported, not
        # taken for a refusal (the Node client decides it the same way).
        error_body = http_status_from_error(exc)
        reason = reportable_reason(error_body.get("reason") if isinstance(error_body, dict) else None)
        return PrincipalClaim("unconfirmed", detail=f"claim answered HTTP {exc.code}, reason={reason}")
    except CoordinatorUnavailable as exc:
        return PrincipalClaim("unconfirmed", detail=str(exc))
    principal = body.get("principal") if isinstance(body, dict) else None
    if isinstance(body, dict) and body.get("ok") is True and isinstance(principal, str) and principal:
        return PrincipalClaim("bound", principal=principal)
    reason = reportable_reason(body.get("reason") if isinstance(body, dict) else None)
    if reason == CALLER_PRINCIPAL_CLAIMED_REASON:
        return PrincipalClaim("refused", detail=CALLER_PRINCIPAL_CLAIMED_REASON)
    return PrincipalClaim("unconfirmed", detail=f"claim not confirmed (reason={reason})")


NODE_BACKEND = "node"


def coordinator_backend(coordinator_root: Path) -> str | None:
    """The ``backend=<name>`` a coordinator recorded on the third line of
    ``.coherence/server.pid``, or ``None`` when there is none.

    The Node coordinator writes ``<pid>\\n<port>\\nbackend=node\\n``; the
    Python coordinator writes ``<pid>\\n<port>\\n`` with no backend line. Only
    the port line is load-bearing for reaching a coordinator — this is read
    solely to skip work a Node coordinator could never answer."""
    try:
        lines = (coordinator_root / ".coherence" / "server.pid").read_text(
            encoding="utf-8"
        ).splitlines()
    except (OSError, UnicodeDecodeError):
        return None
    if len(lines) < 3 or not lines[2].startswith("backend="):
        return None
    return lines[2][len("backend="):].strip() or None


def caller_principal_headers(principal: str | None) -> dict[str, str] | None:
    """The extra header presenting ``principal``, or ``None`` for none."""
    return {CALLER_PRINCIPAL_HEADER: principal} if principal else None


def claims_for_session(session_id: str) -> bool:
    """Whether a client claims (or re-claims) a caller principal for
    ``session_id``: the coordinator's session-id shape check, held to the
    WHOLE string.

    :func:`validate_session_id` matches with ``$``, which in Python also
    admits one trailing newline; the Node client's check (a JS ``$``) does
    not. A Python client claiming for ``"<uuid>\\n"`` would bind a session the
    Node client never presents a principal for, so the two would stop sharing
    one binding per session. The shape is fixed-width hex and hyphens, so a
    trailing newline is the only string ``match`` admits that ``fullmatch``
    refuses — the same gap ``read_subagent_id`` closes with ``fullmatch``."""
    return validate_session_id(session_id) is None and not session_id.endswith("\n")


def principal_refusal_reason(exc: urllib.error.HTTPError) -> str | None:
    """The typed reason when ``exc`` is a caller-principal refusal — HTTP 400
    whose body carries ``reason`` in
    :data:`~ccs.core.exceptions.CALLER_PRINCIPAL_REFUSAL_REASONS`, matched by
    exact membership, never by a substring of the prose — else ``None``.
    Only a string can be a member: a list or an object in that field (a proxy
    or gateway in front of the coordinator) is an ordinary rejected request,
    not a ``TypeError`` from the membership test. Reads the error body (only
    for a 400)."""
    if exc.code != 400:
        return None
    body = http_status_from_error(exc)
    reason = body.get("reason") if isinstance(body, dict) else None
    if isinstance(reason, str) and reason in CALLER_PRINCIPAL_REFUSAL_REASONS:
        return reason
    return None


@dataclass(frozen=True)
class PrincipalRecovery:
    """What claiming again with the HELD mint nonce decided after a request was
    refused for its caller principal — the recovery every client runs (R20).

    - ``retry``: send the refused request ONCE more presenting ``principal``;
      ``None`` when the claim answered 404 (the coordinator issues none now).
      Safe because a refused request mutated nothing.
    - ``stop``: report the refusal; claiming cannot cure it. ``claim`` says
      why — it returned the very principal that was refused (the refusal is
      not about staleness), the session is bound under ANOTHER nonce
      (``refused``: never re-mint), or it did not confirm (``unconfirmed``);
      ``None`` when no nonce was held, so nothing was claimed.

    ``detail`` is diagnostic prose and never carries a principal or nonce."""

    action: Literal["retry", "stop"]
    claim: PrincipalClaim | None
    principal: str | None = None
    detail: str = ""

    @property
    def settled(self) -> bool:
        """Whether a refusal this recovery could not cure is the session's
        settled state — what the raised
        :class:`~ccs.core.exceptions.CallerPrincipalRefused` carries as
        ``settled``. ``False`` only when the claim did NOT confirm: its answer
        was lost, so the session's standing is unknown and the next claim
        with the same nonce may still cure the refusal. Every other stop is
        settled: the session is bound under another nonce, the claim handed
        back the refused principal, or no nonce was held to claim with; and a
        request refused again after a confirmed claim is settled too."""
        return self.claim is None or self.claim.outcome != "unconfirmed"


def decide_principal_recovery(claim: PrincipalClaim, presented: str | None) -> PrincipalRecovery:
    """Map the claim a refused client made with its held nonce to the next step:
    a bound principal that differs from ``presented`` (or anything, when none
    was presented) is retried; the same one, a first-claim refusal, or an
    unconfirmed claim stops; a 404 retries without a header."""
    if claim.outcome == "bound" and claim.principal != presented:
        return PrincipalRecovery("retry", claim, principal=claim.principal)
    if claim.outcome == "bound":
        return PrincipalRecovery(
            "stop", claim, detail="claiming with the held nonce returned the principal that was refused"
        )
    if claim.outcome == "unsupported":
        return PrincipalRecovery("retry", claim, principal=None)
    if claim.outcome == "refused":
        return PrincipalRecovery(
            "stop", claim,
            detail=(
                "the session is bound under a different mint nonce "
                f"({CALLER_PRINCIPAL_CLAIMED_REASON}); not re-minting"
            ),
        )
    return PrincipalRecovery("stop", claim, detail=f"the claim was not confirmed ({claim.detail})")


PRINCIPAL_REFUSED_AGAIN = "refused again after claiming with the held nonce; not retrying"
"""What a client reports when the one retry recovery allows is refused too."""


def principal_refusal_message(reason: str, detail: str) -> str:
    """The text of a caller-principal refusal a client reports: the typed
    reason and what recovery found — built from constants, never from the
    coordinator's prose, so no principal or nonce can reach it."""
    return f"coordinator refused the caller principal ({reason}): {detail}"


def recover_stored_principal(
    endpoint: CoordinatorEndpoint,
    coordinator_root: Path,
    session_id: str,
    presented: str | None,
    report: Callable[[str], None] = err,
) -> PrincipalRecovery:
    """Run the recovery for a ONE-SHOT client whose request naming
    ``session_id`` was refused for its principal: claim again with the nonce
    STORED for the session — never a new one, and the nonce file is never
    touched — and, when that yields a principal to retry with, replace the
    stored principal with it (write-then-rename, ``0600``). Without a stored
    nonce there is nothing to prove a retry with, so nothing is claimed."""
    if not claims_for_session(session_id):
        return PrincipalRecovery("stop", None, detail="the session id is malformed")
    key = caller_principal_identity(session_id).hex
    nonce = load_mint_nonce(coordinator_root, key)
    if nonce is None:
        return PrincipalRecovery("stop", None, detail="no stored mint nonce to claim again with")
    recovery = decide_principal_recovery(
        claim_caller_principal(endpoint, session_id, nonce), presented
    )
    if recovery.action == "retry" and recovery.principal is not None:
        try:
            store_caller_principal(coordinator_root, key, recovery.principal)
        except OSError as exc:
            # The retry still presents it; the next invocation recovers again.
            report(f"caller principal not stored ({type(exc).__name__}); it will be re-obtained")
    return recovery


def post_with_stored_principal(
    endpoint: CoordinatorEndpoint,
    coordinator_root: Path,
    path: str,
    payload: dict[str, Any],
    report: Callable[[str], None] = err,
    *,
    send: Callable[..., dict[str, Any]] = post,
) -> dict[str, Any]:
    """POST ``payload`` for a ONE-SHOT client, presenting the stored principal
    of the session it names (:func:`obtain_stored_principal`).

    A refusal carrying a typed principal reason runs
    :func:`recover_stored_principal` and, when that says so, retries the
    request exactly ONCE. A refusal recovery cannot cure — or a second refusal
    of the retry — raises :class:`~ccs.core.exceptions.CallerPrincipalRefused`
    carrying the wire reason. Any other non-2xx re-raises its ``HTTPError``
    (its body already read); transport failures raise as from :func:`post`.
    ``send`` is the transport (:func:`post`; a caller's own seam)."""
    session_id = payload["session_id"]
    principal = obtain_stored_principal(endpoint, coordinator_root, session_id, report=report)
    answer, reason = _send_presenting(send, endpoint, path, payload, principal)
    if reason is None:
        return answer
    recovery = recover_stored_principal(endpoint, coordinator_root, session_id, principal, report)
    if recovery.action == "stop":
        # A one-shot client's next invocation claims again with the stored
        # nonce by itself, so a stop on an unconfirmed claim is not settled.
        raise CallerPrincipalRefused(
            reason, principal_refusal_message(reason, recovery.detail), settled=recovery.settled
        )
    answer, reason = _send_presenting(send, endpoint, path, payload, recovery.principal)
    if reason is None:
        return answer
    raise CallerPrincipalRefused(reason, principal_refusal_message(reason, PRINCIPAL_REFUSED_AGAIN))


def _send_presenting(
    send: Callable[..., dict[str, Any]],
    endpoint: CoordinatorEndpoint,
    path: str,
    payload: dict[str, Any],
    principal: str | None,
) -> tuple[Any, str | None]:
    """One send presenting ``principal``: ``(answer, None)``, or ``(None,
    reason)`` when the coordinator refused the principal with a typed reason.
    Any other ``HTTPError`` re-raises.

    The refusal is RETURNED, so the caller raises its
    :class:`~ccs.core.exceptions.CallerPrincipalRefused` outside this
    ``except`` block and the ``HTTPError`` — whose text is the status line's
    reason phrase, the coordinator's — never rides that error's chain."""
    try:
        answer = send(endpoint, path, payload, extra_headers=caller_principal_headers(principal))
    except urllib.error.HTTPError as exc:
        reason = principal_refusal_reason(exc)
        if reason is None:
            raise
        return None, reason
    return answer, None


def obtain_stored_principal(
    endpoint: CoordinatorEndpoint,
    coordinator_root: Path,
    session_id: str,
    report: Callable[[str], None] = err,
) -> str | None:
    """The caller principal a ONE-SHOT client (one process per invocation)
    presents for ``session_id``, persisted under ``.coherence/``.

    Keyed by the PARENT session's derived id, so a subagent's hook presents its
    parent's principal (the principal's unit of identity is the session). The
    stored principal is used if present; otherwise the mint nonce is read or
    exclusively created FIRST, then claimed, then the bound principal stored.
    Two processes racing on a new session share one nonce and so receive one
    principal. Returns ``None`` — send no header — when the coordinator issues
    none, when the claim is unconfirmed (a later claim retries with the same
    nonce), or when the claim is refused. A refusal is reported through
    ``report`` and nothing is deleted or re-minted. A stored principal the
    coordinator later refuses is recovered by :func:`post_with_stored_principal`.

    Against a coordinator whose pid file says ``backend=node`` nothing is
    claimed, read or created: the Node coordinator issues no principals, and a
    one-shot client would otherwise pay a 404 round trip on every hook event.
    A pid file without that line (the Python coordinator's own format) is
    claimed against, and an older Python coordinator's 404 still means "send
    no header".

    What this buys is convention-enforcement and a detectable unbound caller:
    the files are readable by any process that can read ``.coherence/``, so
    this is not separation between callers of the same OS user (KTD5)."""
    if not claims_for_session(session_id):
        return None
    if coordinator_backend(coordinator_root) == NODE_BACKEND:
        return None
    key = caller_principal_identity(session_id).hex
    stored = load_caller_principal(coordinator_root, key)
    if stored is not None:
        return stored
    try:
        nonce = ensure_mint_nonce(coordinator_root, key)
    except (OSError, MintNonceUnavailable) as exc:
        report(f"caller principal unavailable: no usable mint nonce ({exc})")
        return None
    claim = claim_caller_principal(endpoint, session_id, nonce)
    if claim.outcome == "bound" and claim.principal is not None:
        try:
            store_caller_principal(coordinator_root, key, claim.principal)
        except OSError as exc:
            # The claim stands; the next invocation re-obtains it by its nonce.
            report(f"caller principal not stored ({exc}); it will be re-obtained")
        return claim.principal
    if claim.outcome == "refused":
        report(
            "caller principal refused: this session is already bound under a "
            "different mint nonce; proceeding without a principal and NOT "
            "re-minting (routes that require one will refuse this session)"
        )
    return None


REDIRECT_LOCATION_WITHHELD = "(withheld)"
"""The ``location`` every :class:`~ccs.core.exceptions.RedirectRefused` carries,
in place of the one the coordinator sent: a redirect is reported by its
status alone."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse ANY 3xx instead of following it.

    The coordinator is one fixed, operator-configured endpoint — no redirect is
    ever legitimate. Critically, urllib's default ``HTTPRedirectHandler`` COPIES
    the ``Authorization`` header onto the redirected hop *before* returning, so a
    post-hoc check on the final response cannot protect the bearer. Refusing here,
    before the second request is issued, ensures the bearer never leaves the
    configured endpoint.

    Every 3xx code enters through :meth:`_refuse`, which names the status
    only — never the ``Location``, and nothing parses it first. The
    ``Location`` is the coordinator's text, and whatever answers may echo
    into it what it was sent: a principal header, or a mint nonce. A request
    that carries neither cannot rule that out, because the session may have
    sent its nonce to the same endpoint earlier, from this process or
    another. Quoted, the echo would reach the refusal's message and from
    there logs and tool results; and the stdlib's own parse raises a
    ``ValueError`` that quotes a malformed ``Location`` (a bracketed host
    that is not an address).
    """

    def _refuse(self, req, fp, code, msg, headers):  # noqa: ANN001, ANN202 - stdlib signature
        raise RedirectRefused(REDIRECT_LOCATION_WITHHELD, status=code)

    # All five codes, 308 included: each stdlib ``http_error_30x`` starts by
    # parsing the Location, so any one left to it reopens the gap above.
    http_error_301 = http_error_302 = http_error_303 = _refuse
    http_error_307 = http_error_308 = _refuse

    def redirect_request(  # type: ignore[override]
        self, req, fp, code, msg, headers, newurl
    ):  # noqa: ANN001, ANN201 - matches the stdlib handler signature
        # Unreached while every code above refuses first; kept so a redirect
        # path added later still refuses — by the status alone.
        raise RedirectRefused(REDIRECT_LOCATION_WITHHELD, status=code)


def _build_opener(context: ssl.SSLContext | None) -> urllib.request.OpenerDirector:
    """A private opener that goes straight to the endpoint and refuses every 3xx.

    Assembled by hand, not with ``urllib.request.build_opener``: that adds a
    default ``HTTPSHandler`` to every opener, and on Python 3.12+ its constructor
    builds a default SSL context, loading the whole system CA store (~13 ms of
    CPU) even for plain ``http://`` to loopback. The handlers below are the
    ``build_opener`` defaults an http(s) request reaches, with
    ``_NoRedirectHandler`` in place of ``HTTPRedirectHandler``.
    ``HTTPErrorProcessor`` is what routes a 3xx to that handler, so it must stay.
    The ftp/file/data handlers are left out: every URL here is ``base_url + path``
    and no redirect is followed.

    There is deliberately no ``ProxyHandler``: no coordinator request goes
    through a proxy, loopback or remote. The default one reads ``http_proxy``
    and ``https_proxy`` (on macOS and Windows, the system settings when those are
    unset) and, with ``no_proxy`` unset, proxies loopback too, sending the bearer
    to the proxy. A remote endpoint is the one host the operator configured and
    secured the link to; a proxy is a hop that neither ``CCS_REMOTE_INSECURE``
    nor https verification covers.

    An ``HTTPSHandler`` is added only for ``context``, the verified-TLS context
    from :func:`build_tls_context`. Without one there is no https handler at all,
    so an https request cannot fall back to a default context: it fails as an
    unknown URL type.
    """
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.UnknownHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        _NoRedirectHandler(),
        urllib.request.HTTPErrorProcessor(),
    ]
    if context is not None:
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.OpenerDirector()
    for handler in handlers:
        opener.add_handler(handler)
    return opener


_SHARED_OPENERS_LOCK = threading.Lock()
#: Keyed by ``system_tls`` (see :func:`_get_shared_opener`).
_shared_openers: dict[bool, urllib.request.OpenerDirector] = {}


def _reset_shared_openers_lock_in_child() -> None:
    # A fork while another thread is building an opener copies this lock held,
    # and no thread in the child will ever release it: the child's first request
    # would block forever. Openers are stored only once fully built, so the child
    # keeps them and builds any that were still in flight.
    global _SHARED_OPENERS_LOCK
    _SHARED_OPENERS_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_shared_openers_lock_in_child)


def _get_shared_opener(*, system_tls: bool) -> urllib.request.OpenerDirector:
    """A process-wide opener, built on first use: plain http, or with
    ``system_tls`` https verified against the system trust store.

    Shared across threads: each of its handlers keeps no per-request state
    (``HTTPHandler`` opens a fresh connection per request, the rest only read or
    annotate the request), which is also why the stdlib's ``urlopen`` shares one
    module-level opener. Never add a stateful handler such as a cookie processor
    here, because every endpoint and thread in the process would share its state.

    The system-trust context is shared too (an OpenSSL client context caches no
    TLS sessions, so none carries over between connections), so the trust store
    is not parsed again on every request. It is loaded when this opener is
    built. Until the process restarts, a certificate later removed from the
    system bundle (or from ``SSL_CERT_FILE``) stays trusted, and a bundle that
    was missing at that moment stays missing.
    """
    opener = _shared_openers.get(system_tls)
    if opener is None:
        with _SHARED_OPENERS_LOCK:
            opener = _shared_openers.get(system_tls)
            if opener is None:
                context = build_tls_context() if system_tls else None
                opener = _shared_openers[system_tls] = _build_opener(context)
    return opener


def _execute(req: urllib.request.Request) -> dict[str, Any]:
    if req.type == "https":
        # build_tls_context may raise TlsConfigError (typed, fail-closed) — that
        # is a config bug, not a transient network failure, so it propagates.
        ca_file = getattr(req, "_ccs_ca_file", None)
        if ca_file:
            # Built per request, never shared: build_tls_context re-validates
            # and re-reads the CA bundle each time, so a swapped, loosened or
            # rotated trust anchor is caught by the very next request.
            opener = _build_opener(build_tls_context(ca_file))
        else:
            opener = _get_shared_opener(system_tls=True)
    else:
        opener = _get_shared_opener(system_tls=False)
    malformed: str | None = None
    try:
        with opener.open(req, timeout=CLI_HTTP_TIMEOUT_SEC) as resp:
            raw = resp.read()
    except urllib.error.HTTPError:
        # Caller handles status-code-specific paths.
        raise
    except RedirectRefused:
        # Typed refusal from _NoRedirectHandler — fail closed, do not degrade.
        raise
    except urllib.error.URLError as exc:
        # A TLS certificate-verification failure surfaces here (SSLError wrapped
        # in URLError). It is a TRUST decision, not a transient hiccup: map it to
        # the typed refusal so the bearer is never retried over plaintext, and
        # keep it distinct from CoordinatorUnavailable (which callers may treat
        # as retryable). Every other URLError stays CoordinatorUnavailable.
        if isinstance(exc.reason, ssl.SSLCertVerificationError):
            raise TlsVerificationFailed(
                _host_of(req), str(exc.reason)
            ) from exc
        raise CoordinatorUnavailable(
            f"could not reach coordinator at {req.full_url}: {exc.reason}"
        ) from exc
    except (OSError, TimeoutError) as exc:
        raise CoordinatorUnavailable(
            f"network error talking to coordinator: {exc}"
        ) from exc
    except http.client.HTTPException as exc:
        # A malformed answer — a status line that is not HTTP (BadStatusLine),
        # a body cut short (IncompleteRead), an overlong header line — is
        # transport-shaped like the failures above, so it is reported as one:
        # urllib does not wrap these, and they would otherwise escape every
        # caller's transport handling untyped. Only the exception TYPE is
        # named: a BadStatusLine's text IS the line the coordinator sent, and
        # an IncompleteRead holds the partial body. The CoordinatorUnavailable
        # is raised below, outside this block, so neither rides its chain.
        malformed = type(exc).__name__
    if malformed is not None:
        raise CoordinatorUnavailable(f"coordinator sent a malformed HTTP response ({malformed})")

    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CoordinatorUnavailable(
            f"coordinator returned non-JSON response: {exc}"
        ) from exc


def _host_of(req: urllib.request.Request) -> str:
    """Best-effort host for a TLS-verification error message (never the bearer).

    Prefer the explicit ``Host`` header (a bare host, set by ``get``/``post`` from
    ``endpoint.host``) over ``req.host`` (which carries ``host:port``) so the
    typed refusal names the clean host the operator configured.
    """
    return req.get_header("Host") or getattr(req, "host", "") or req.full_url


def http_status_from_error(exc: urllib.error.HTTPError) -> dict[str, Any] | None:
    """Best-effort JSON decode of an HTTPError body, for one-line user output:
    ``None`` unless the body is a JSON object."""
    try:
        raw = exc.read()
    except Exception:
        return None
    if not raw:
        return None
    try:
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return body if isinstance(body, dict) else None


def http_error_line(code: int, body: dict[str, Any] | None) -> str:
    """The one line a console script prints for an HTTP error answer:
    ``HTTP {code}: {error}`` when the body's ``error`` is a string, escaped
    (:func:`escape_nonprintable`), else ``HTTP {code}``. Never the status
    line's reason phrase: that is the server's text too. ``body`` is
    :func:`http_status_from_error`'s, read once by the caller."""
    error = body.get("error") if body is not None else None
    return f"HTTP {code}: {escape_nonprintable(error)}" if isinstance(error, str) else f"HTTP {code}"
