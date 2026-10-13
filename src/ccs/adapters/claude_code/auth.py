# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Shared-secret authentication for the local HTTP coordinator (KTD-12).

Without this, any same-user process or browser tab can corrupt MESI state
via direct POST or DNS rebinding (browser pages can resolve their own
domain to 127.0.0.1 and bypass same-origin). Suppression of stale-read
warnings is the most direct attack — it would nullify the product's value
silently. ~50 lines of stdlib, no dependency cost.

VERIFIED 2026-05-14 (brainstorm §13.8, §13.9):
- HTTP hook handler accepts arbitrary Authorization headers (axios/1.13.6)
- Bearer secret is REDACTED from `--include-hook-events` debug streams

Threat model:
- Adversary 1: another process running as the same OS user (e.g., a malicious
  npm package, a compromised dev tool). Mitigated by hook.secret being mode
  0600 — only this user can read it.
- Adversary 2: browser tab visiting an attacker page that does DNS rebinding
  to resolve attacker.com → 127.0.0.1. Mitigated by Host-header check
  (browser sends Host: attacker.com, server rejects).
- NOT mitigated: malicious code running with the same UID that ALSO has
  filesystem read of `.coherence/hook.secret`. v0.1 accepts this — it's the
  same trust boundary as the user's shell history, SSH agent socket, etc.
"""

from __future__ import annotations

import contextlib
import hmac
import ipaddress
import logging
import os
import secrets
import time
from pathlib import Path

logger = logging.getLogger(__name__)


SECRET_FILENAME = "hook.secret"
"""The file name inside <coordinator-root>/.coherence/ that holds the
hex-encoded shared secret. Mode 0600 — owner-read-only."""

SECRET_BYTES = 32
"""32 bytes of random entropy → 64-char hex token in the Authorization header."""

ENSURE_SECRET_MAX_RETRIES = 5
"""R11 (Unit 6): bound on the empty-file recovery loop in :func:`ensure_secret`.
If we observe 'file exists but is empty' more than this many times in a row,
something pathological is happening (a racer that creates but never writes,
a misbehaving editor, disk-full mid-write); fail closed rather than risk
clobbering valid secrets via O_TRUNC."""

ENSURE_SECRET_RETRY_SLEEP_SEC = 0.020
"""R11 (Unit 6): brief sleep between empty-file recovery attempts so a
racer that has the file open but hasn't flushed its write yet gets a
chance to make progress before we re-poll."""

_BEARER_PREFIX = "Bearer "
_HOST_ALLOWLIST: frozenset[str] = frozenset({"localhost", "127.0.0.1"})

#: Truthy env values that opt into cross-host mode (mirrors the client flag).
_REMOTE_TRUTHY_ENV_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on"})

#: Explicit private-range networks a cross-host coordinator may bind to. We
#: range-check explicitly because ``ipaddress.is_private`` returns True for BOTH
#: ``0.0.0.0`` and loopback aliases (e.g. 127.0.0.2) — admitting either would
#: defeat the bind guard. Loopback, link-local (169.254/16) and CGNAT (100.64/10)
#: are deliberately excluded here (loopback is handled separately).
_PRIVATE_V4_NETS = (
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
)
_PRIVATE_V6_NET = ipaddress.ip_network("fc00::/7")


def _remote_flag_enabled() -> bool:
    """True when CCS_REMOTE_COORDINATOR opts into cross-host mode (default OFF)."""
    return (
        os.environ.get("CCS_REMOTE_COORDINATOR", "").strip().lower()
        in _REMOTE_TRUTHY_ENV_VALUES
    )


def is_loopback_host(host: str) -> bool:
    """True for the always-allowed loopback names (localhost / 127.0.0.1)."""
    return host in _HOST_ALLOWLIST


def _is_private_range(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if ip.version == 4:
        return any(ip in net for net in _PRIVATE_V4_NETS)
    return ip in _PRIVATE_V6_NET


def validate_bind_host(host: str) -> None:
    """Validate a coordinator bind address (raise ``ValueError`` if disallowed).

    Loopback (localhost / 127.0.0.1) is always allowed. Any other address
    requires the cross-host opt-in (``CCS_REMOTE_COORDINATOR``) AND must be an
    explicit RFC-1918/4193 private-range IP. ``0.0.0.0`` / wildcard, loopback
    aliases (127.0.0.2), link-local (169.254/16), CGNAT (100.64/10) and public
    addresses are all rejected — binding to any of them would expose the
    coordinator beyond the intended private network.
    """
    if is_loopback_host(host):
        return
    if not _remote_flag_enabled():
        raise ValueError(
            f"refusing to bind the coordinator to {host!r} beyond loopback; set "
            "CCS_REMOTE_COORDINATOR to opt into cross-host mode"
        )
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(f"bind host {host!r} is not a valid IP address") from exc
    if ip.is_unspecified:
        raise ValueError(f"refusing to bind the coordinator to the wildcard address {host!r}")
    if not _is_private_range(ip):
        raise ValueError(
            f"bind host {host!r} is not an RFC-1918/4193 private-range address "
            "(loopback aliases, link-local, CGNAT and public addresses are rejected)"
        )


#: Server-side transport-posture env vars (Unit 3 / R5). ``CCS_TLS_TERMINATED``
#: asserts a TLS-terminating front sits ahead of the coordinator (posture INFO);
#: ``CCS_SERVE_INSECURE`` is the explicit plaintext-link acknowledgement (WARNING),
#: the mirror of the client's ``CCS_REMOTE_INSECURE``. Both parse against the same
#: truthy set as ``CCS_REMOTE_COORDINATOR`` (:data:`_REMOTE_TRUTHY_ENV_VALUES`).
_TLS_TERMINATED_ENV = "CCS_TLS_TERMINATED"
_SERVE_INSECURE_ENV = "CCS_SERVE_INSECURE"


def _serve_env_enabled(name: str) -> bool:
    """True when the named transport-posture env var is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in _REMOTE_TRUTHY_ENV_VALUES


def assert_serve_transport_acknowledged(bind_host: str) -> None:
    """Fail-closed guard: refuse to serve on a routed bind without a transport ack.

    Symmetry with the client's ``CCS_REMOTE_INSECURE`` plaintext-bearer guard
    (#135): the coordinator authenticates hook requests with a bearer secret, so
    serving them on a routed (non-loopback) bind over plaintext exposes the
    bearer on the wire. This guard refuses to construct such a server unless the
    operator either asserts a TLS-terminating front is present
    (``CCS_TLS_TERMINATED``) or explicitly acknowledges the insecure link
    (``CCS_SERVE_INSECURE``).

    Loopback binds read NEITHER env and return immediately (byte-unchanged; no
    log). The bind host must already have passed :func:`validate_bind_host`
    (non-loopback ⇒ CCS_REMOTE_COORDINATOR-gated private-range IP); this guard
    reuses that loopback classification rather than re-deriving one, so the
    client and server never drift.

    Precedence: if BOTH envs are set the TLS assertion wins the log line (a
    single INFO posture line, no double-warn). Raises :class:`ValueError` (the
    established construction-time bind-rejection idiom of
    :func:`build_host_allowlist`) naming both envs when neither is set.
    """
    if is_loopback_host(bind_host):
        return
    if _serve_env_enabled(_TLS_TERMINATED_ENV):
        # Assertion wins even when CCS_SERVE_INSECURE is also set (no double-warn).
        # Names the bind host + posture only — there is no secret in scope here.
        logger.info(
            "serving the coordinator on routed bind %r; %s asserted "
            "(a TLS-terminating front is expected ahead of the coordinator)",
            bind_host,
            _TLS_TERMINATED_ENV,
        )
        return
    if _serve_env_enabled(_SERVE_INSECURE_ENV):
        logger.warning(
            "serving the coordinator bearer on routed bind %r over plaintext HTTP "
            "(%s acknowledged — ensure the link is encrypted out-of-band)",
            bind_host,
            _SERVE_INSECURE_ENV,
        )
        return
    raise ValueError(
        f"refusing to serve the coordinator bearer on routed bind {bind_host!r} "
        f"over plaintext HTTP; set {_TLS_TERMINATED_ENV} to assert a "
        f"TLS-terminating front, or {_SERVE_INSECURE_ENV} to acknowledge an "
        "out-of-band-secured link"
    )


def build_host_allowlist(bind_host: str) -> frozenset[str]:
    """The Host-header allowlist for a coordinator bound to ``bind_host``.

    Always includes loopback; for a validated non-loopback bind it also admits
    that exact host. Validates ``bind_host`` (raises on a disallowed bind), so a
    coordinator constructed with a bad bind fails loud at construction.

    This function stays a pure address→allowlist derivation (no env-driven
    transport-posture side effects). The routed-bind transport acknowledgement is
    a distinct concern enforced by :func:`assert_serve_transport_acknowledged` at
    the coordinator construction path — keeping the two separable lets callers
    that only need the allowlist derive it without a live transport-posture env.
    """
    validate_bind_host(bind_host)
    if is_loopback_host(bind_host):
        return _HOST_ALLOWLIST
    return _HOST_ALLOWLIST | {bind_host}


class EnsureSecretError(RuntimeError):
    """R11 (Unit 6): ensure_secret could not converge — the file exists
    but stays empty across ENSURE_SECRET_MAX_RETRIES attempts. The
    coordinator startup path should treat this as fatal; the alternative
    (O_TRUNC re-write of a file another process may have just populated)
    risks clobbering a concurrent racer's valid secret and giving two
    spawn-side processes different secrets for the same workspace."""


def ensure_secret(coordinator_root: Path) -> str:
    """Generate-and-persist the shared secret if missing; otherwise load it.

    Idempotent: safe to call from every coordinator spawn. Returns the
    hex-encoded secret. Raises ``OSError`` if the ``.coherence`` directory
    cannot be created (graceful-degradation should happen at the
    lifecycle layer, not here). Raises :class:`EnsureSecretError` if a
    ``hook.secret`` file exists but stays empty across
    ``ENSURE_SECRET_MAX_RETRIES`` attempts — see R11 for the rationale
    on failing closed instead of falling back to O_TRUNC.

    R11 (Unit 6): the empty-file recovery branch is now a bounded
    O_EXCL retry loop instead of the prior O_TRUNC re-write. The
    O_TRUNC path could clobber a concurrent spawn's valid secret in
    the narrow window between O_EXCL-create and write — both processes
    would then walk away with DIFFERENT secrets for the same workspace,
    which silently breaks all peer hooks until one coordinator restarts.
    """
    coherence_dir = coordinator_root / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret_path = coherence_dir / SECRET_FILENAME

    for attempt in range(ENSURE_SECRET_MAX_RETRIES):
        # Fast path: file exists and is populated.
        # SEC-02 / finding #40: wrap read_text() in try/except so that a
        # same-UID concurrent unlink between is_file() and read_text()
        # (TOCTOU) causes a retry instead of a propagated FileNotFoundError
        # that would crash coordinator startup.
        if secret_path.is_file():
            try:
                token = secret_path.read_text().strip()
            except (FileNotFoundError, OSError):
                continue
            if token:
                return token

        # Try the atomic O_EXCL create. Either we win (write our secret)
        # or someone else owns the file (re-read on next iteration).
        new_token = secrets.token_hex(SECRET_BYTES)
        try:
            fd = os.open(
                str(secret_path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError:
            # Another process owns the file. Brief sleep so they can
            # finish writing, then loop back to the fast-path re-read.
            if attempt + 1 < ENSURE_SECRET_MAX_RETRIES:
                time.sleep(ENSURE_SECRET_RETRY_SLEEP_SEC)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(new_token + "\n")
        logger.info("generated shared secret at %s", secret_path)
        return new_token

    # All retries observed the file existing but staying empty. We do
    # NOT O_TRUNC over it — that would risk overwriting a concurrent
    # racer's just-written secret. Fail closed; coordinator startup
    # aborts with an actionable error.
    raise EnsureSecretError(
        f"hook.secret at {secret_path} exists but stayed empty across "
        f"{ENSURE_SECRET_MAX_RETRIES} attempts; refusing to O_TRUNC over a "
        f"file another process may be writing concurrently. Manually "
        f"remove {secret_path} if it is genuinely stale."
    )


def load_secret(coordinator_root: Path) -> str | None:
    """Load the secret if it exists. Returns None if the file is missing.
    Used by hook clients (CLI scripts, console-script entry points) that
    should NEVER create the secret — only the coordinator-spawn path does."""
    secret_path = coordinator_root / ".coherence" / SECRET_FILENAME
    if not secret_path.is_file():
        return None
    # SEC-02 / finding #40: same TOCTOU guard as ensure_secret — a concurrent
    # unlink between is_file() and read_text() returns None instead of raising.
    try:
        token = secret_path.read_text().strip()
    except (FileNotFoundError, OSError):
        return None
    return token or None


# ---------------------------------------------------------------------------
# Caller principal — the wire header and the one-shot client's stored values
# (coordinator caller principal plan, U5)
# ---------------------------------------------------------------------------
#
# The bearer above authenticates the WORKSPACE. A caller principal is a value
# the coordinator mints and binds to ONE acting identity on that identity's
# first claim (``POST /principal/claim``); a request naming the identity
# presents it in :data:`CALLER_PRINCIPAL_HEADER`. What it buys depends on the
# caller (plan KTD5):
#
# - a LONG-LIVED caller (``CoherentVolume``, a direct client) mints once and
#   holds the principal in memory for its lifetime: it never looks a principal
#   up by the identity it names, so it cannot present another identity's by
#   accident. That is accident-resistance, not unreadability — the coordinator
#   stores every principal it issued in ``.coherence/state.db``, which any
#   process of the same OS user can read;
# - a ONE-SHOT caller (the hook client: one process per hook event) can keep it
#   only on disk, in this ``0700`` directory, keyed by the identity it claims.
#   Any process that can read ``.coherence/`` can read it for any identity.
#   There the principal buys convention-enforcement and a detectable unbound
#   caller — never separation between callers of the same OS user.
#
# The nonce file is created once, exclusively, and never replaced or removed:
# it is what proves a later claim is a retry of the first (R20). The principal
# file is a cache of what a claim presenting that nonce returned, so it IS
# replaced — atomically, and only with such a claim's answer — when the
# coordinator no longer holds the value it caches (its store was reset) or the
# file was torn.

CALLER_PRINCIPAL_HEADER = "Coherence-Caller-Principal"
"""Request header carrying the caller principal. Named in the style of
``Coherence-Local-Operator``; ignored by a coordinator that issues none (the
sibling Node coordinator ignores any header it does not read)."""

CALLER_PRINCIPAL_FILE_PREFIX = "caller-principal-"
"""Stored-value files are ``<prefix><identity hex>.nonce`` and
``<prefix><identity hex>.principal`` inside ``.coherence/``. The identity is
the PARENT session's derived agent id (32 lowercase hex), never the raw session
id — the same key the sibling Node hook client uses, so both clients share one
binding per session."""

_PRINCIPAL_VALUE_LEN = 43
"""``secrets.token_urlsafe(32)`` length — the shape of both a mint nonce this
module generates and a principal the coordinator mints. A read that is not
exactly this shape is a file another process has created but not finished
writing, and is treated as not-yet-there, never as a value."""

_PRINCIPAL_VALUE_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)

_IDENTITY_KEY_ALPHABET = frozenset("0123456789abcdef")


TORN_FILE_GRACE_SEC = 2.0
"""How long an existing but incomplete nonce file is treated as a racer's write
still in progress. The exclusive create and the write are two steps, so a
reader can see the file between them; a YOUNG torn file is waited on (the
bounded ``ENSURE_SECRET_*`` retry) so a loser adopts the winner's nonce instead
of treating the half-written file as final. An OLDER one is treated as
abandoned — its writer was killed, or ran out of space — and reported at once
rather than charging every later hook of the session the whole bounded wait.
The rule is safe whichever it really is: after the wait the outcome of an
unreadable nonce is the same as without it (no principal this invocation, and
the file is never overwritten), so the grace decides only whether to wait.
Sized at 25 times the bounded wait (4 x 20 ms)."""


_NONCE_REMEDIATION = (
    "The session runs without a principal until it is fixed: remove {path} by "
    "hand if no hook of this session is running."
)
"""The operator step a :class:`MintNonceUnavailable` for an ABANDONED write
names, as the ``hook.secret`` error does: the file is never repaired
automatically, and the Node client prints the same guidance (parity)."""

_NONCE_STILL_BEING_WRITTEN = (
    "This invocation proceeds without a principal; a later one adopts the nonce "
    "if that write lands, or reports {path} as an interrupted write once it has "
    "stayed incomplete for {grace:g} s."
)
"""What a :class:`MintNonceUnavailable` for a YOUNG file says instead: its writer
may still be alive and finish, so it names no remedy and does not say the
session runs without a principal until someone acts. Once the file is older
than :data:`TORN_FILE_GRACE_SEC` the next invocation reports it as abandoned,
with :data:`_NONCE_REMEDIATION`."""


class MintNonceUnavailable(RuntimeError):
    """The mint-nonce file exists but never became readable within the
    bounded wait — a racer that created it and never wrote it. Like
    :class:`EnsureSecretError`, never repaired by truncating it: truncating
    could replace a nonce another process already claimed with."""


def _principal_file(coordinator_root: Path, identity_key: str, suffix: str) -> Path:
    if len(identity_key) != 32 or not set(identity_key) <= _IDENTITY_KEY_ALPHABET:
        raise ValueError("identity_key must be 32 lowercase hex characters")
    return coordinator_root / ".coherence" / f"{CALLER_PRINCIPAL_FILE_PREFIX}{identity_key}{suffix}"


def _read_principal_value(path: Path) -> str | None:
    """The stored value, or ``None`` when absent or not (yet) well-formed."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        return None
    if len(value) != _PRINCIPAL_VALUE_LEN or not set(value) <= _PRINCIPAL_VALUE_ALPHABET:
        return None
    return value


def _is_past_grace(path: Path) -> bool:
    """Whether an existing ``path`` was last written longer ago than
    :data:`TORN_FILE_GRACE_SEC`. A file that vanished, or whose clock reads in
    the future, is not past it — waiting is the safe default."""
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age > TORN_FILE_GRACE_SEC


def _create_exclusive(path: Path, value: str) -> bool:
    """Create ``path`` holding ``value`` with ``O_CREAT|O_EXCL`` at ``0600``
    (the ``hook.secret`` discipline). ``False`` if it already existed; an
    existing file is never truncated or rewritten."""
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        handle.write(value + "\n")
    return True


def ensure_mint_nonce(coordinator_root: Path, identity_key: str) -> str:
    """The mint nonce for ``identity_key``, generating and persisting it first
    if no process has (plan KTD11: persisted BEFORE the claim is sent).

    Two hook processes racing on a new session get ONE nonce: the loser of the
    exclusive create adopts the winner's value, so both claims present the same
    nonce and the coordinator hands both the same principal. Never creates
    ``.coherence/`` (a hook client never does; ``OSError`` propagates when it
    is missing). Raises :class:`MintNonceUnavailable` if an existing file stays
    unreadable across the bounded retry — at once, without waiting, when the
    file is older than :data:`TORN_FILE_GRACE_SEC` and so treated as an
    abandoned write rather than one still in progress."""
    path = _principal_file(coordinator_root, identity_key, ".nonce")
    for attempt in range(ENSURE_SECRET_MAX_RETRIES):
        existing = _read_principal_value(path)
        if existing is not None:
            return existing
        candidate = secrets.token_urlsafe(32)
        if _create_exclusive(path, candidate):
            return candidate
        if _is_past_grace(path):
            raise MintNonceUnavailable(
                f"{path} exists but holds no complete nonce (an interrupted "
                f"write); not overwriting it. {_NONCE_REMEDIATION.format(path=path)}"
            )
        if attempt + 1 < ENSURE_SECRET_MAX_RETRIES:
            time.sleep(ENSURE_SECRET_RETRY_SLEEP_SEC)
    raise MintNonceUnavailable(
        f"{path} exists but its write was still in progress across "
        f"{ENSURE_SECRET_MAX_RETRIES} attempts; not overwriting it. "
        f"{_NONCE_STILL_BEING_WRITTEN.format(path=path, grace=TORN_FILE_GRACE_SEC)}"
    )


def load_mint_nonce(coordinator_root: Path, identity_key: str) -> str | None:
    """The mint nonce stored for ``identity_key``, or ``None``. Never creates
    one: recovering a refused principal re-claims only with the nonce the
    session already persisted — a nonce generated at recovery time would be a
    second claimant, which first-claim-wins refuses (KTD11)."""
    return _read_principal_value(_principal_file(coordinator_root, identity_key, ".nonce"))


def load_caller_principal(coordinator_root: Path, identity_key: str) -> str | None:
    """The principal stored for ``identity_key``, or ``None``."""
    return _read_principal_value(_principal_file(coordinator_root, identity_key, ".principal"))


def store_caller_principal(coordinator_root: Path, identity_key: str, principal: str) -> None:
    """Persist ``principal`` for ``identity_key``: written to a private
    temporary file in the same directory (``O_CREAT|O_EXCL``, ``0600``) and
    renamed over the stored file, so a reader sees the old value or the new
    one, never a torn one.

    Callers pass only what a claim presenting the session's STORED nonce
    returned — the value the coordinator binds to that nonce — so replacing
    the file is never a re-mint: it repairs a torn file, and it follows a
    binding store that was reset (R20). Concurrent writers hold the same value
    (the binding for that nonce), so the order of their renames does not
    matter; a value a reset made stale in between is replaced again by the
    next recovery."""
    path = _principal_file(coordinator_root, identity_key, ".principal")
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(str(temporary), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(principal + "\n")
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise


def verify_bearer(authorization_header: str | None, expected_secret: str) -> bool:
    """Constant-time comparison of an Authorization header against the
    expected secret. Returns True only when the header is present, well-
    formed (``Bearer <token>``), and the token matches exactly.

    Constant-time prevents timing oracles on the token bytes — relevant
    because the server's response time is observable from another process
    on the same machine."""
    if not authorization_header:
        return False
    if not authorization_header.startswith(_BEARER_PREFIX):
        return False
    presented = authorization_header[len(_BEARER_PREFIX):]
    return hmac.compare_digest(presented, expected_secret)


def _host_from_header(host_header: str) -> str:
    """Extract the host from a Host header, stripping any ``:port`` suffix.

    Handles bracketed IPv6 literals (``[fc00::1]:8080`` → ``fc00::1``) as well as
    IPv4/hostname forms (``127.0.0.1:54321`` → ``127.0.0.1``). Returns the raw
    header unchanged when it is malformed — no closing ``]``, or junk between
    ``]`` and the ``:port`` — so it matches no allowlist entry and verify_host
    fails closed. Deliberately does NOT trim whitespace/control characters: a real
    Host header arrives already-clean from http.server, and tolerating padding
    here would admit values like ``localhost\\r`` that the exact check must reject.
    """
    if host_header.startswith("["):
        end = host_header.find("]")
        if end == -1:
            return host_header  # no closing bracket -> malformed -> fail closed
        rest = host_header[end + 1 :]
        if rest and not rest.startswith(":"):
            return host_header  # junk after "]" before the port -> fail closed
        return host_header[1:end]
    return host_header.split(":", 1)[0]


def verify_host(host_header: str | None, allowlist: frozenset[str] = _HOST_ALLOWLIST) -> bool:
    """Reject Host headers not in ``allowlist`` (default: localhost/127.0.0.1).

    Block DNS rebinding: an attacker page at attacker.com resolves
    attacker.com → 127.0.0.1, browser sends ``Host: attacker.com``, server
    must reject. The cross-host coordinator passes a wider allowlist
    ({loopback, validated bind_host}); every other host still 403s. Allows the
    allowlisted names with or without a port suffix, including bracketed IPv6.
    """
    if not host_header:
        return False
    hostname = _host_from_header(host_header)
    if hostname in allowlist:
        return True
    # IP literals: match on the normalized address so equivalent spellings
    # (e.g. fc00::1 vs fc00:0:0:0:0:0:0:1) resolve to the same allowlist entry.
    # Non-IP names (localhost, attacker.com) only match via the exact check
    # above — this never widens admission to a host not already in the allowlist.
    try:
        candidate = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    # IPv4-mapped IPv6 (::ffff:127.0.0.1) and scope-id forms are rejected here
    # by construction: ipaddress compares unequal across IPv4Address/IPv6Address
    # and across differing scope ids. Do NOT add an .ipv4_mapped unwrap — that
    # would let a mapped Host alias an allowlisted IPv4 entry.
    for entry in allowlist:
        try:
            if ipaddress.ip_address(entry) == candidate:
                return True
        except ValueError:
            continue
    return False
