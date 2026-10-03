# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Cross-implementation protocol corpus harness (plan Unit 7).

Loads JSON fixtures from ``fixtures/<mode>/*.json`` and routes each fixture's
``request`` to BOTH the Python in-thread coordinator AND the Node subprocess
coordinator, then asserts the responses are JSON-equivalent after normalization.

The harness is the security-relevant surface here: the normalization rules
("what we ignore in the diff") decide whether a real Python ↔ Node drift gets
caught or false-passes. Each rule has an inline comment explaining why the
field is ignored — review the rules in a focused PR per the Unit 7 verification
checklist.

Fixture schema (see ``fixtures/warn_mode/*.json``)::

    {
      "name": "<unique-fixture-name>",
      "description": "<optional human-readable>",
      "setup": {
        "tracked": ["<gitignore-glob>", ...],
        "files":   {"<path>": "<contents>", ...},
        "policy":  {<optional pre-test policy.yaml overrides>}
      },
      "request": {
        "method":  "POST" | "GET",        # default POST
        "path":    "/health" | ...,
        "body":    {<JSON body>}           # for POST
        "headers": {<optional overrides>}  # auth headers added automatically
      },
      "expected": {
        "status": <int>,
        "body":   {<normalized expected JSON>}
      },
      "backends": ["python", "node"],     # default both
      "preserve_identity": ["<key>", ...] # opt out of UUID scrubbing (R15)
    }

Two things a fixture can express since the identity unit (R15), both of which
the harness could not carry before and which are described in full at
``normalize_response`` and ``apply_preflight_requests``:

- **A minted value.** A preflight request may declare
  ``"capture": {"<name>": "<dotted field path>"}``; any LATER request (preflight
  or main) names the bound value as ``"${<name>}"``. Preflight responses used to
  be discarded, so no fixture could reach a value the coordinator mints at run
  time. Substitution applies to REQUESTS only — never to an expected body (R5).
- **Which identity a response names.** ``"preserve_identity": ["<key>", ...]``
  compares those keys' string values verbatim instead of scrubbing them to
  ``<UUID>``. The scrub stays the default everywhere else: it is what lets one
  fixture set run against two implementations that mint different ids.

A value the coordinator mints and a client only echoes (``session_token``,
``coordinator_epoch``) is scrubbed BY NAME to ``MINTED_SENTINEL`` (#193): the
key stays in the diff, its per-run value does not, and ``preserve_identity``
cannot un-hide it.

A principal is subject to R5 and is NOT expressible in an expectation: it
normalizes to ``PRINCIPAL_SENTINEL``, ``preserve_identity`` cannot un-hide it,
and ``build_fixture`` refuses a fixture whose expected body carries one.

- **A value only one backend issues.** A preflight may declare
  ``"backends": [...]``; it then runs only on those backends. The caller
  principal is the case: only the Python coordinator mints one, and a
  both-backend fixture that exercises a require-class route must present it
  there while the Node coordinator (which answers 404 on the claim and ignores
  the header, plan KTD12) gets none. A capture made by such a preflight may be
  referenced ONLY as the whole value of a request header; on a backend where
  the preflight did not run, that header is omitted — exactly what a conforming
  client does after a 404 on the claim. Any other reference is refused at load.

Spawn isolation: each fixture gets a fresh ``tmp_path`` workspace. The Python
coordinator runs in-thread (no subprocess overhead). The Node coordinator runs
as ``node <plugin-dist>/coordinator.js`` with ``AGENT_COHERENCE_WORKSPACE``
pointing at the tmp workspace. Both coordinators bind ephemeral ports.

Plugin coordinator discovery: ``AGENT_COHERENCE_PLUGIN_DIST_PATH`` env var
(absolute path to ``dist/coordinator.js``); only while it is UNSET, fallback
``../agent-coherence-plugin/dist/coordinator.js`` relative to the library repo root, then
``~/projects/agent-coherence-plugin/dist/coordinator.js``. Set to a path that does not exist, it
raises instead of falling back. If it is unset and neither fallback exists, the resolver returns
``None``: the parity corpora xfail their Node rows with a clear reason, and the asymmetry corpora
(caller principal, effect fence) fail them."""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_ROOT = Path(__file__).resolve().parent / "fixtures"
HARNESS_TIMEOUT_SEC = 10.0
NODE_SPAWN_TIMEOUT_SEC = 8.0
NODE_SHUTDOWN_TIMEOUT_SEC = 5.0
#: Names the Node coordinator entry point to run; see :func:`resolve_node_dist_path`.
NODE_DIST_PATH_ENV = "AGENT_COHERENCE_PLUGIN_DIST_PATH"


# ----------------------------------------------------------------------
# Normalization rules
# ----------------------------------------------------------------------

# Field names whose VALUE is non-deterministic across coordinator instances
# and should be replaced with a sentinel before diffing. Comments explain the
# rationale for each — these are the "what we ignore" rules and any addition
# weakens the harness's catch surface, so changes go in a dedicated PR.
_TS_SENTINEL = "<TS>"
#: Public because a fixture-facing rule is written in terms of them: a corpus
#: module has to be able to say "this value must NOT have been scrubbed", and
#: an identity opt-in that preserves a key whose asserted value is already the
#: sentinel preserves nothing.
UUID_SENTINEL = "<UUID>"
IGNORED_SENTINEL = "<IGNORED>"
#: R5: a principal appears on exactly ONE response — the mint response that
#: issues it — and never in a log, a status tier, another response body, or a
#: fixture's expected body. It gets a sentinel DISTINCT from ``UUID_SENTINEL``
#: so the rule is enforceable by name: collapsed into the UUID sentinel, a
#: principal field would be indistinguishable from any other identifier and
#: ``_validate_expected_body`` could not refuse one.
PRINCIPAL_SENTINEL = "<PRINCIPAL>"
#: A value the coordinator MINTS per run and a client only ever echoes back:
#: a snapshot ``session_token`` and the per-boot ``coordinator_epoch``. Its own
#: sentinel, not ``UUID_SENTINEL``: neither value is UUID-shaped (a 43-char
#: URL-safe base64 token; a 32-char undashed hex epoch), so ``_UUID_RE`` never
#: touched them, and naming the class keeps "this field is present and was
#: minted" distinguishable from "this field is an identity" in an expected body.
MINTED_SENTINEL = "<MINTED>"
_PID_SENTINEL = "<PID>"
_UPTIME_SENTINEL = "<UPTIME>"
_PORT_SENTINEL = "<PORT>"
_HASH_SENTINEL = "<SHA256>"

# Top-level / nested key names whose value is normalized regardless of type.
# Listed explicitly so a new "started_at_ms" field doesn't silently slip
# through normalization just because it looks ISO-8601-ish.
_TIMESTAMP_KEYS: frozenset[str] = frozenset({
    # Wall-clock fields surfaced by /status default + metrics tiers.
    "ts",
    "started_at",
    "last_request_at",
    "last_401_warn_at",
    "started_at_ms",        # Node-side field; Python emits float seconds
    "last_completed_ms",
    "first_observation_ts",
    "last_seen_at",
})

_UPTIME_KEYS: frozenset[str] = frozenset({
    "coordinator_uptime_seconds",
    # AC-02 deprecated alias — Python emits both during the v0.1.x window so a
    # consumer migrating from the old name still gets a value. Removed in v0.2.
    "coordinator_uptime_s",
    "uptime_seconds",
    "uptime_ms",
    "process_uptime_seconds",
})

_PID_KEYS: frozenset[str] = frozenset({
    "pid",
    "coordinator_pid",
    "process_pid",
    "worker_pid",
})

_PORT_KEYS: frozenset[str] = frozenset({
    "port",
    "coordinator_port",
    "listen_port",
})

# Fields whose value is a UUID4 — we keep them present (the wire shape carries
# the field, both coordinators emit a value) but normalize the value itself.
_UUID_KEYS: frozenset[str] = frozenset({
    "instance_id",
    "agent",
    "agent_id",
    "session_id",
    "request_id",
})

# Caller-principal fields. A FROZEN vocabulary, deliberately not derived from
# the route: a principal field name this set does not know is compared
# literally, so it would flake per run AND would slip past the R5 check on
# expected bodies. A route that grows a new principal field name adds it HERE
# in the same diff.
_PRINCIPAL_KEYS: frozenset[str] = frozenset({
    "principal",
    "caller_principal",
    "principal_id",
})

# Minted-per-run fields (#193 ask 2). Before these were listed, every fixture
# asserting a /session/* or /workspace/* response needed a per-fixture
# ``ignore_keys`` entry for each — a hole the fixture had to open by hand, and
# one ``ignore_keys`` widens to ANY value, absent included as ``<IGNORED>``.
# Scrubbing by name keeps the KEY compared: a response that drops
# ``coordinator_epoch`` (``/session/heartbeat`` never carries one) still
# differs from one that carries it.
_MINTED_KEYS: frozenset[str] = frozenset({
    # Server-minted snapshot-session capability (``/session/begin``); 32
    # random bytes, URL-safe base64. Echoed by the client, never inspected.
    "session_token",
    # Per-coordinator-boot epoch (32-char hex) on every session and workspace
    # response except the heartbeat; a client compares it for CHANGE, so its
    # value is never portable across runs or runtimes.
    "coordinator_epoch",
})

# Content-hash fields. SHA-256 hex is deterministic for identical bytes, so we
# *don't* normalize these by default — drift in content_hash IS a real wire
# regression. Sentinel reserved for fixtures that need to ignore content_hash
# (e.g., where the hash depends on a timestamp embedded in the file).
_HASH_KEYS: frozenset[str] = frozenset()  # opt-in per fixture via "ignore_keys"

# UUIDv4 string regex — used to scrub UUID-shaped values that appear in string
# positions (error messages, log fragments) even when the key name isn't in
# _UUID_KEYS. Permissive: any 8-4-4-4-12 hex sequence.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)

# ISO-8601 timestamp regex — used to scrub timestamp-shaped values in string
# positions (error messages with "at 2026-05-23T...").
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)


# ``${name}`` — the only way a fixture names a value the coordinator minted at
# run time. Deliberately NOT a bare ``$name``: a reference has to be
# unmistakable on sight in a JSON fixture, and an accidental match against
# prose in a description or an error string would silently rewrite it.
_CAPTURE_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class FixtureContractError(ValueError):
    """A fixture breaks one of the corpus's own rules.

    Raised rather than worked around, because every rule it enforces exists to
    stop a fixture that READS as an assertion from being a vacuous one: a
    principal in an expected body (normalized away and therefore equal to any
    other principal), an identity opt-in naming a key the fixture never
    asserts, a capture of a field the mint response does not carry."""


class FixtureSubstitutionError(FixtureContractError):
    """A ``${...}`` reference could not be resolved at run time.

    The whole point of raising: an unresolved reference left on the wire as a
    literal string is a well-formed request carrying a value the coordinator
    never minted, so it draws a REJECTION — and a fixture whose expectation
    happened to be that rejection would pass while proving nothing about
    substitution having happened."""


def normalize_response(
    value: Any,
    *,
    ignore_keys: Optional[frozenset[str]] = None,
    optional_keys: Optional[frozenset[str]] = None,
    preserve_identity: Optional[frozenset[str]] = None,
) -> Any:
    """Replace non-deterministic field values with stable sentinels.

    Recurses into dicts and lists. Scalars are returned as-is unless the
    enclosing key is in one of the normalization sets.

    ``ignore_keys`` — per-fixture opt-in set of additional key names whose
    values should be replaced with ``"<IGNORED>"``. Use sparingly; each entry
    is a hole in the catch surface.

    ``optional_keys`` — per-fixture opt-in set of key names DROPPED entirely
    from the normalized dict (on both actual and expected), for backend-specific
    optional extension fields whose shared baseline both backends must still
    match (e.g. a Python-only OCC ``version`` field on a pre-read response that
    the Node coordinator does not emit). **Scoped to the TOP-LEVEL response body
    only** (AC3) — a nested key that happens to share a name with an optional_key
    is still compared, so a real nested divergence is never masked.

    ``preserve_identity`` — per-fixture opt-in set of key names whose STRING
    value is compared verbatim: no UUID sentinel, no string-position UUID/ISO
    scrubbing. It exists because the default below scrubs every 8-4-4-4-12 hex
    value unconditionally, which is right for portability — one corpus runs
    against two implementations that mint different ids — and wrong for a
    fixture whose whole content is WHICH identity a response names. Without the
    opt-in, "attributed to A" and "attributed to B" normalize to the same bytes
    and both pass (KTD7).

    Three deliberate limits. (1) It applies at ANY depth, because the fields
    that name an identity are nested (``summary.last_writer_session_id``).
    (2) It applies to ``str`` values ONLY — a declared key holding a dict or a
    list falls through to ordinary walking, so a declaration can never freeze a
    subtree and take a timestamp along with it. (3) It does NOT outrank R5: a
    key that is also a principal field still normalizes to
    ``PRINCIPAL_SENTINEL``.

    A fixture declaring this must be run by a module that PASSES it through —
    ``run_scenario`` does so for the actual side, and the asserting module must
    do the same for the expected side. A module that does not know the field
    compares a preserved actual against a scrubbed expected, which fails red,
    not green; ``build_fixture`` keeps that from being a surprise by refusing a
    declaration the fixture's own expected body does not carry."""
    ignore_keys = ignore_keys or frozenset()
    optional_keys = optional_keys or frozenset()
    preserve_identity = preserve_identity or frozenset()

    def _walk(v: Any, *, depth: int) -> Any:
        if isinstance(v, dict):
            # optional_keys are dropped at the ROOT dict only (depth == 0): they
            # describe top-level backend-specific extension fields, not arbitrary
            # nested occurrences. A nested same-named key (depth > 0) is kept and
            # compared so a nested divergence still fails the diff.
            drop = optional_keys if depth == 0 else frozenset()
            return {
                k: _walk_keyed(v[k], k, depth=depth)
                for k in sorted(v.keys())
                if k not in drop
            }
        if isinstance(v, list):
            return [_walk(item, depth=depth + 1) for item in v]
        if isinstance(v, str):
            # String-position scrubbing: UUID and ISO-8601 substrings in
            # error messages / log fragments.
            scrubbed = _UUID_RE.sub(UUID_SENTINEL, v)
            scrubbed = _ISO_TS_RE.sub(_TS_SENTINEL, scrubbed)
            return scrubbed
        return v

    def _walk_keyed(v: Any, key: str, *, depth: int) -> Any:
        # Per-fixture ignore set wins over everything else.
        if key in ignore_keys:
            return IGNORED_SENTINEL
        # R5 next, BEFORE the identity opt-in: a principal is never comparable,
        # so a fixture must not be able to un-hide one by declaring its key.
        if key in _PRINCIPAL_KEYS:
            return PRINCIPAL_SENTINEL
        # The identity opt-in: a declared key's STRING value survives verbatim.
        # Non-string values fall through so a declaration cannot freeze a
        # subtree (and smuggle a timestamp through with it).
        # A minted value is never comparable either (it differs on every run),
        # so it too outranks the opt-in; ``_validate_preserve_identity`` refuses
        # a declaration over one at load. Only a non-empty STRING scrubs: a
        # null / "" / non-string under a minted key was not minted, so it falls
        # through and is compared literally (an expected ``<MINTED>`` then
        # fails against it).
        if key in _MINTED_KEYS and isinstance(v, str) and v:
            return MINTED_SENTINEL
        if key in preserve_identity and isinstance(v, str):
            return v
        # Key-driven normalization fires regardless of value type so we don't
        # care whether the coordinator emits int or float for an uptime.
        if key in _TIMESTAMP_KEYS:
            return _TS_SENTINEL
        if key in _UPTIME_KEYS:
            return _UPTIME_SENTINEL
        if key in _PID_KEYS:
            return _PID_SENTINEL
        if key in _PORT_KEYS:
            return _PORT_SENTINEL
        if key in _UUID_KEYS:
            return UUID_SENTINEL
        if key in _HASH_KEYS:
            return _HASH_SENTINEL
        # Descend into the value; nested dicts are depth + 1 (so optional_keys no
        # longer apply below the root).
        return _walk(v, depth=depth + 1)

    return _walk(value, depth=0)


# ----------------------------------------------------------------------
# Backend identifiers
# ----------------------------------------------------------------------


BACKEND_PYTHON = "python"
BACKEND_NODE = "node"
ALL_BACKENDS = (BACKEND_PYTHON, BACKEND_NODE)


# ----------------------------------------------------------------------
# Fixture loader
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Fixture:
    """A single protocol-corpus scenario."""

    name: str
    description: str
    path: Path
    setup: dict[str, Any]
    request: dict[str, Any]
    expected: dict[str, Any]
    backends: tuple[str, ...]
    ignore_keys: frozenset[str]
    optional_keys: frozenset[str]
    #: Key names whose identity this fixture asserts verbatim. Empty for every
    #: fixture that predates the identity capability, which is the portability
    #: default (see ``normalize_response``).
    preserve_identity: frozenset[str] = frozenset()


def _iter_keyed(value: Any) -> Iterator[tuple[str, Any]]:
    """Yield every ``(key, value)`` pair in a JSON tree, at any depth."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _iter_keyed(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_keyed(item)


def _iter_strings(value: Any) -> Iterator[str]:
    """Yield every string anywhere in a JSON tree, keys included."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _iter_strings(child)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, str):
        yield value


def _validate_expected_body(name: str, body: Any) -> None:
    """R5, enforced at load: no principal and no minted value in an expectation.

    A principal normalizes to ``PRINCIPAL_SENTINEL``, so a hard-coded principal
    in an expected body would be replaced by that sentinel and would then
    compare equal to ANY principal — an assertion that reads as pinning an
    identity and pins nothing. Refusing the fixture is the difference between
    "the harness normalized it" and "the harness told someone". The sentinel
    itself is admissible: it asserts the field is present without asserting
    whose principal it is.

    The ``${...}`` half closes the only other route a minted value could take
    into an expectation. Substitution is deliberately never applied to expected
    bodies, so a reference there would survive to the diff as a literal — but
    saying so at load is what makes the omission a decision rather than an
    oversight."""
    for key, value in _iter_keyed(body):
        if key in _PRINCIPAL_KEYS and value != PRINCIPAL_SENTINEL:
            raise FixtureContractError(
                f"{name}: expected body carries a principal under {key!r} "
                f"({value!r}). R5 — a principal appears on exactly one "
                f"response, the mint response that issues it, and never in an "
                f"expected body. Assert {PRINCIPAL_SENTINEL!r} to pin that the "
                f"FIELD is present without pinning whose principal it is."
            )
    for text in _iter_strings(body):
        ref = _CAPTURE_REF_RE.search(text)
        if ref:
            raise FixtureContractError(
                f"{name}: expected body references the captured value "
                f"{ref.group(1)!r}. Captures are substituted into REQUESTS "
                f"only — an expected body naming a minted value would be "
                f"asserting the harness against itself."
            )


def _validate_capture_references(name: str, setup: dict, request: dict) -> None:
    """Every ``${...}`` must be resolvable from a capture declared EARLIER.

    Checked at load so the failure lands at collection, where it names the
    fixture, rather than mid-run where it names a request. The run-time guard in
    ``_substitute`` stays regardless: a fixture mutated in memory (which is how
    the corpus tests drive their controls) never passes through here.

    A capture made by a backend-scoped preflight is SCOPED: it is not issued on
    the other backends, so it may appear only as the whole value of a request
    header (omitted where it was not issued). Anywhere else — a body field, a
    path, part of a longer string — there is no honest value to send in its
    place, and the fixture is refused."""
    declared: set[str] = set()
    scoped: set[str] = set()
    stages: list[tuple[str, Any]] = [
        (f"preflight #{i}", req)
        for i, req in enumerate(setup.get("preflight_requests") or [])
    ]
    stages.append(("request", request))
    for label, stage in stages:
        for text in _iter_strings(stage):
            for ref in _CAPTURE_REF_RE.findall(text):
                if ref not in declared:
                    raise FixtureContractError(
                        f"{name}: {label} references {ref!r}, which no earlier "
                        f"preflight captures. Declared so far: "
                        f"{sorted(declared) or '<none>'}."
                    )
        header_refs = {
            m.group(1)
            for value in (stage.get("headers") or {}).values()
            if isinstance(value, str) and (m := _CAPTURE_REF_RE.fullmatch(value))
        }
        without_headers = {k: v for k, v in stage.items() if k != "headers"}
        for text in _iter_strings(without_headers):
            for ref in _CAPTURE_REF_RE.findall(text):
                if ref in scoped:
                    raise FixtureContractError(
                        f"{name}: {label} references the backend-scoped capture "
                        f"{ref!r} outside a request header. It is not issued on "
                        f"every backend, so it may only be a whole header value."
                    )
        for value in (stage.get("headers") or {}).values():
            for ref in _CAPTURE_REF_RE.findall(value if isinstance(value, str) else ""):
                if ref in scoped and ref not in header_refs:
                    raise FixtureContractError(
                        f"{name}: {label} embeds the backend-scoped capture "
                        f"{ref!r} in a longer header value; it must be the whole "
                        f"value so the header can be omitted where it is not issued."
                    )
        stage_backends = stage.get("backends")
        if stage_backends is not None:
            if label == "request":
                raise FixtureContractError(
                    f"{name}: the main request cannot be backend-scoped; scope the "
                    f"fixture with its top-level 'backends' instead."
                )
            unknown = [b for b in stage_backends if b not in ALL_BACKENDS]
            if not stage_backends or unknown:
                raise FixtureContractError(
                    f"{name}: {label} declares backends {stage_backends!r}; it "
                    f"must be a non-empty subset of {ALL_BACKENDS}."
                )
        for captured in stage.get("capture") or {}:
            if captured in declared:
                raise FixtureContractError(
                    f"{name}: {label} re-captures {captured!r}; a capture name "
                    f"is bound once, or a later request silently reads the "
                    f"wrong mint."
                )
            declared.add(captured)
            if stage_backends is not None and set(stage_backends) != set(ALL_BACKENDS):
                scoped.add(captured)


def _validate_preserve_identity(
    name: str,
    preserve_identity: frozenset[str],
    ignore_keys: frozenset[str],
    body: Any,
) -> None:
    """Refuse a declaration the comparison would quietly override.

    Refused: a key ``ignore_keys`` discards first; a principal key (R5 outranks
    the opt-in); a minted key (``_MINTED_KEYS`` outrank it too — the value
    differs per run); a key the fixture's expected body never carries; a key
    with no non-sentinel STRING occurrence; and a key with ANY occurrence whose
    asserted value is a sentinel. Each would leave a fixture looking like it
    pinned an identity while the comparison it produces is the scrubbed one.

    Every occurrence is checked, not the last one: a key repeated across rows
    (``receiver`` per checkpoint, ``agent_id`` per session) whose first row
    asserts ``<UUID>`` and whose last asserts an identity is refused, where a
    dict built over the walk would have kept only the last and accepted it. A
    non-string occurrence (``null`` for a checkpoint that names no receiver) is
    admitted beside a string one: the opt-in does not apply to it, and the
    default compares ``null`` literally anyway.

    One shape is ACCEPTED although it does nothing: a declared key whose every
    value the default scrub leaves unchanged (an undashed hex id, which
    ``_UUID_RE`` never matches). Declared, it states the contract for the field;
    deleted, the comparison is the same bytes — it is verbatim either way, so
    the declaration cannot turn a failing comparison into a passing one, which
    is the class of fixture this check exists to refuse.
    ``harness_identity/02`` is that case and its description says so."""
    both = sorted(preserve_identity & ignore_keys)
    if both:
        raise FixtureContractError(
            f"{name}: {both} are declared in BOTH preserve_identity and "
            f"ignore_keys. ignore_keys wins, so the identity is discarded, not "
            f"preserved."
        )
    principals = sorted(preserve_identity & _PRINCIPAL_KEYS)
    if principals:
        raise FixtureContractError(
            f"{name}: {principals} name a principal field. R5 outranks the "
            f"identity opt-in — a principal is never comparable."
        )
    minted = sorted(preserve_identity & _MINTED_KEYS)
    if minted:
        raise FixtureContractError(
            f"{name}: {minted} name a minted field. Its value differs on every "
            f"run, so it always normalizes to {MINTED_SENTINEL!r} — there is no "
            f"identity to preserve."
        )
    asserted: dict[str, list[Any]] = {}
    for key, value in _iter_keyed(body):
        asserted.setdefault(key, []).append(value)
    sentinels = {
        UUID_SENTINEL, PRINCIPAL_SENTINEL, IGNORED_SENTINEL, _TS_SENTINEL,
        MINTED_SENTINEL,
    }
    for key in sorted(preserve_identity):
        if key not in asserted:
            raise FixtureContractError(
                f"{name}: preserve_identity names {key!r}, which this "
                f"fixture's expected body does not carry. A declaration over a "
                f"key nothing asserts preserves nothing."
            )
        values = asserted[key]
        pinned = [v for v in values if isinstance(v, str)]
        if not pinned:
            raise FixtureContractError(
                f"{name}: preserve_identity names {key!r}, whose asserted "
                f"values are {values!r}. The opt-in applies to string values only."
            )
        hidden = [v for v in pinned if v in sentinels]
        if hidden:
            raise FixtureContractError(
                f"{name}: preserve_identity names {key!r}, and "
                f"{len(hidden)} of its {len(values)} asserted occurrence(s) is "
                f"the sentinel {hidden[0]!r}. Preserving a sentinel preserves "
                f"nothing — pin the identity itself."
            )


def build_fixture(data: dict[str, Any], path: Path) -> Fixture:
    """Validate one fixture's raw JSON and build the dataclass.

    Split out of ``load_fixtures`` so the contract rules are reachable from a
    test with a dict in hand: every rule here refuses a fixture that would
    otherwise run and report green, and a rule nothing can exercise is the same
    class of problem it exists to catch."""
    name = data["name"]
    backends_raw = data.get("backends") or list(ALL_BACKENDS)
    unknown = [b for b in backends_raw if b not in ALL_BACKENDS]
    if unknown:
        raise FixtureContractError(
            f"{path.name}: unknown backends {unknown!r}; allowed={ALL_BACKENDS}"
        )
    setup = data.get("setup", {})
    request = data["request"]
    expected = data["expected"]
    ignore_keys = frozenset(data.get("ignore_keys", []))
    preserve_identity = frozenset(data.get("preserve_identity", []))

    _validate_expected_body(name, expected.get("body"))
    _validate_capture_references(name, setup, request)
    _validate_preserve_identity(name, preserve_identity, ignore_keys, expected.get("body"))

    return Fixture(
        name=name,
        description=data.get("description", ""),
        path=path,
        setup=setup,
        request=request,
        expected=expected,
        backends=tuple(backends_raw),
        ignore_keys=ignore_keys,
        optional_keys=frozenset(data.get("optional_keys", [])),
        preserve_identity=preserve_identity,
    )


def load_fixtures(mode: str = "warn_mode") -> list[Fixture]:
    """Load all JSON fixtures under ``fixtures/<mode>/`` in sorted order."""
    mode_root = FIXTURES_ROOT / mode
    if not mode_root.exists():
        return []
    fixtures: list[Fixture] = []
    for path in sorted(mode_root.glob("*.json")):
        with path.open() as fh:
            data = json.load(fh)
        fixtures.append(build_fixture(data, path))
    return fixtures


# ----------------------------------------------------------------------
# Workspace setup
# ----------------------------------------------------------------------


def apply_setup(workspace: Path, setup: dict[str, Any]) -> None:
    """Materialize a fixture's ``setup`` block onto a fresh tmp workspace.

    - ``files``: writes named files relative to the workspace root.
    - ``tracked``: writes ``.coherence/tracked.yaml`` with one path per line.
    - ``ignored``: writes ``.coherence/ignored.yaml`` with one path per line.
    - ``strict_mode``: writes ``.coherence/strict_mode.yaml`` with one path
      per line (Unit 7b — strict-mode-fixture support; the Node coordinator
      doesn't ship strict_mode.yaml loading in v0.2 so strict fixtures are
      python-only).
    - ``preflight_requests``: list of request dicts fired BEFORE the main
      test request; responses ignored. Used to drive multi-step setup like
      "session A reads → session B preempts → A re-reads (the test request)".
      Schema mirrors the top-level ``request``: ``{method, path, body, headers}``."""
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(exist_ok=True, mode=0o700)

    files = setup.get("files") or {}
    for rel_path, contents in files.items():
        target = workspace / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)

    # YAML loaders for tracked/ignored/strict_mode all accept a top-level
    # list of patterns. Write the simplest valid shape that
    # TrackedArtifactPolicy.load accepts.
    for setup_key, yaml_file in (
        ("tracked", "tracked.yaml"),
        ("ignored", "ignored.yaml"),
        ("strict_mode", "strict_mode.yaml"),
    ):
        patterns = setup.get(setup_key) or []
        if patterns:
            (coherence_dir / yaml_file).write_text(
                "\n".join(f"- {p}" for p in patterns) + "\n"
            )


class _NotIssued:
    """A capture whose preflight was scoped away from this backend."""

    def __repr__(self) -> str:
        return "<not issued on this backend>"


NOT_ISSUED = _NotIssued()


def _omit_headers_not_issued(request: dict, captures: dict[str, Any]) -> dict:
    """Drop every request header whose whole value names a capture this backend
    never issued (see the module docstring). Everything else is left for
    ``_substitute``, which refuses a not-issued value anywhere else."""
    headers = request.get("headers")
    if not headers:
        return request
    kept = {
        k: v for k, v in headers.items()
        if not (
            isinstance(v, str)
            and (m := _CAPTURE_REF_RE.fullmatch(v))
            and captures.get(m.group(1)) is NOT_ISSUED
        )
    }
    return {**request, "headers": kept}


def _substitute(value: Any, captures: dict[str, Any], *, where: str) -> Any:
    """Replace every ``${name}`` in a request tree with its captured value.

    A reference that IS the whole string keeps the captured value's JSON type
    (an int stays an int); a reference embedded in a longer string interpolates
    its text. An unresolved reference raises — it is never left on the wire as
    a literal, because a literal is a well-formed request carrying a value the
    coordinator never minted, and the rejection it draws can accidentally BE
    what a fixture expects."""
    if isinstance(value, dict):
        return {k: _substitute(v, captures, where=where) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, captures, where=where) for v in value]
    if not isinstance(value, str):
        return value

    def _resolve(ref: str) -> Any:
        if ref not in captures:
            raise FixtureSubstitutionError(
                f"{where}: unresolved reference ${{{ref}}}. Captured so far: "
                f"{sorted(captures) or '<none>'}. Refusing to send the literal "
                f"— a never-minted value draws a rejection that can silently "
                f"satisfy the fixture's own expectation."
            )
        if captures[ref] is NOT_ISSUED:
            raise FixtureSubstitutionError(
                f"{where}: ${{{ref}}} was not issued on this backend and is "
                f"referenced outside a whole header value; there is no honest "
                f"value to send in its place."
            )
        return captures[ref]

    whole = _CAPTURE_REF_RE.fullmatch(value)
    if whole:
        return _resolve(whole.group(1))
    return _CAPTURE_REF_RE.sub(lambda m: str(_resolve(m.group(1))), value)


def _record_captures(
    spec: dict[str, str],
    body: Any,
    captures: dict[str, Any],
    *,
    where: str,
) -> None:
    """Bind each declared name to a field of the response that just landed.

    The field path is dotted. An absent field and a ``null`` value both raise
    rather than binding ``None``: a null substitutes as a JSON null, the
    coordinator refuses it, and the fixture ends up asserting a rejection it
    never meant to drive — the same vacuous pass an unresolved reference
    would produce, one level up."""
    for name, field_path in spec.items():
        if name in captures:
            raise FixtureContractError(
                f"{where}: re-captures {name!r}; a capture name is bound once."
            )
        cursor: Any = body
        for part in field_path.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                raise FixtureContractError(
                    f"{where}: capture {name!r} reads {field_path!r}, but the "
                    f"response has no {part!r}: {body!r}"
                )
            cursor = cursor[part]
        if cursor is None:
            raise FixtureContractError(
                f"{where}: capture {name!r} read {field_path!r} as null. A null "
                f"would go on the wire as a value the coordinator never minted."
            )
        captures[name] = cursor


def apply_preflight_requests(
    backend: CoordinatorBackend,
    preflight: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fire each preflight request against ``backend``, returning its captures.

    Preflight responses used to be discarded outright, which is why no fixture
    could express a value the coordinator MINTS at run time (KTD7). They are
    still not asserted — a preflight is setup, not a claim — but a request may
    now declare ``capture: {<name>: <dotted field path>}``, and each later
    request (preflight or main) may name the bound value as ``${<name>}``.

    Status-code mismatches in preflight are reported as RuntimeError so a
    broken setup surfaces early instead of corrupting the main assertion."""
    captures: dict[str, Any] = {}
    for i, req in enumerate(preflight):
        where = f"preflight #{i} ({req.get('method', 'POST')} {req.get('path')!r})"
        scope = req.get("backends")
        if scope is not None and backend.backend_id not in scope:
            # Scoped away from this backend: it never runs here, and what it
            # would have captured is recorded as not issued.
            for name in req.get("capture") or {}:
                captures[name] = NOT_ISSUED
            continue
        resolved = _substitute(
            _omit_headers_not_issued(req, captures), captures, where=where
        )
        resolved.pop("backends", None)
        status, body = execute_request(backend, resolved)
        # Allow 200, 400 (preflight that expects validation errors), 404.
        # Anything in the 500s indicates a coordinator bug — fail loud.
        if status >= 500:
            raise RuntimeError(
                f"Preflight request #{i} ({req.get('method', 'POST')} "
                f"{req.get('path')!r}) returned 5xx: status={status}, body={body!r}"
            )
        spec = req.get("capture")
        if spec:
            _record_captures(spec, body, captures, where=where)
    return captures


# ----------------------------------------------------------------------
# Coordinator backends
# ----------------------------------------------------------------------


class CoordinatorBackend:
    """Abstract base: spawn, expose (port, secret), shutdown."""

    backend_id: str

    def start(self, workspace: Path) -> None:
        raise NotImplementedError

    def url(self, path: str) -> str:
        raise NotImplementedError

    def secret(self) -> str:
        raise NotImplementedError

    def shutdown(self) -> None:
        raise NotImplementedError


class PythonCoordinator(CoordinatorBackend):
    """In-process Python coordinator via ``CoordinatorHTTPServer.serve_in_thread``."""

    backend_id = BACKEND_PYTHON

    def __init__(self) -> None:
        self._server = None
        self._secret: Optional[str] = None
        self._port: Optional[int] = None

    def start(self, workspace: Path) -> None:
        # Defer the import so test discovery doesn't pay the cost.
        from ccs.adapters.claude_code.auth import load_secret
        from ccs.adapters.claude_code.coordinator_server import (
            CoordinatorHTTPServer,
        )

        server = CoordinatorHTTPServer(workspace, port=0, instance_id="protocol-corpus-py")
        server.serve_in_thread()
        # Tiny grace window — matches the existing
        # tests/test_claude_code_coordinator_server.py pattern.
        time.sleep(0.05)
        secret = load_secret(server.coordinator_root)
        assert secret is not None, "Python coordinator failed to write hook.secret"
        self._server = server
        self._secret = secret
        self._port = server.port

    def url(self, path: str) -> str:
        assert self._port is not None
        return f"http://127.0.0.1:{self._port}{path}"

    def secret(self) -> str:
        assert self._secret is not None
        return self._secret

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.shutdown()


class NodeCoordinator(CoordinatorBackend):
    """Out-of-process Node coordinator spawned as ``node dist/coordinator.js``."""

    backend_id = BACKEND_NODE

    def __init__(self, dist_path: Path) -> None:
        if not dist_path.exists():
            raise FileNotFoundError(
                f"Node coordinator entry point not found: {dist_path}. "
                f"Build the plugin (cd <plugin-repo> && npm ci && npm run build) "
                f"or set AGENT_COHERENCE_PLUGIN_DIST_PATH to the absolute path."
            )
        self._dist_path = dist_path
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._port: Optional[int] = None
        self._secret: Optional[str] = None

    def start(self, workspace: Path) -> None:
        env = dict(os.environ)
        env["AGENT_COHERENCE_WORKSPACE"] = str(workspace)
        # Suppress harmless verbose logging if the operator wants quieter
        # test output (Node coordinator logs to stderr; harness captures
        # both streams and surfaces them only on failure).
        self._proc = subprocess.Popen(
            ["node", str(self._dist_path)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        port, secret = _wait_for_coordinator(
            workspace,
            timeout_sec=NODE_SPAWN_TIMEOUT_SEC,
            proc=self._proc,
            backend_label="node",
        )
        self._port = port
        self._secret = secret

    def url(self, path: str) -> str:
        assert self._port is not None
        return f"http://127.0.0.1:{self._port}{path}"

    def secret(self) -> str:
        assert self._secret is not None
        return self._secret

    def shutdown(self) -> None:
        if self._proc is None:
            return
        # SIGTERM → graceful close per coordinator.ts's shutdown handler.
        # Fall back to SIGKILL after NODE_SHUTDOWN_TIMEOUT_SEC.
        self._proc.terminate()
        try:
            self._proc.wait(timeout=NODE_SHUTDOWN_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=NODE_SHUTDOWN_TIMEOUT_SEC)


def _wait_for_coordinator(
    workspace: Path,
    *,
    timeout_sec: float,
    proc: subprocess.Popen[bytes],
    backend_label: str,
) -> tuple[int, str]:
    """Poll for ``.coherence/server.pid`` + ``hook.secret`` to materialize.

    Surfaces subprocess crash early (returns nonzero before the pid file lands)
    with the captured stderr — a silent timeout would otherwise hide the real
    diagnostic from a failed Node build."""
    pid_file = workspace / ".coherence" / "server.pid"
    secret_file = workspace / ".coherence" / "hook.secret"
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rc = proc.poll()
        if rc is not None:
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            stdout = proc.stdout.read().decode("utf-8", errors="replace") if proc.stdout else ""
            raise RuntimeError(
                f"{backend_label} coordinator exited with rc={rc} before writing server.pid.\n"
                f"stdout:\n{stdout}\nstderr:\n{stderr}"
            )
        if pid_file.exists() and secret_file.exists():
            try:
                lines = pid_file.read_text().splitlines()
                if len(lines) >= 2:
                    port = int(lines[1])
                    secret = secret_file.read_text().strip()
                    if secret:
                        # Probe the actual socket once before returning so we
                        # don't race the listen() callback on slow CI.
                        if _socket_open("127.0.0.1", port):
                            return port, secret
            except (OSError, ValueError):
                pass  # Partial write — retry on next tick.
        time.sleep(0.05)
    raise TimeoutError(
        f"{backend_label} coordinator did not write server.pid within {timeout_sec}s; "
        f"workspace={workspace}"
    )


def _socket_open(host: str, port: int) -> bool:
    """Check whether the coordinator's listen socket is accepting yet."""
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def resolve_node_dist_path() -> Optional[Path]:
    """Locate the Node coordinator entry point.

    Resolution order:
    1. ``AGENT_COHERENCE_PLUGIN_DIST_PATH`` env var (absolute path)
    2. ``../agent-coherence-plugin/dist/coordinator.js`` relative to library repo root
    3. ``~/projects/agent-coherence-plugin/dist/coordinator.js``

    The fallbacks apply only while the variable is UNSET. Set to a path that
    does not exist (or set empty), it raises ``FileNotFoundError`` naming the
    variable: an explicit path is the one a run was asked to test, and falling
    back from a mistyped one would run the Node rows against whatever dist the
    fallback checkout holds and report green. Every corpus module resolves at
    import, so the raise is a collection error in each of them and none of
    their rows is collected -- not a failure in only the rows that remembered
    to check.

    Returns ``None`` if the variable is unset and no fallback exists — the
    parity corpora then xfail their Node rows with a clear reason, and the
    asymmetry corpora fail them, rather than hanging."""
    explicit = os.environ.get(NODE_DIST_PATH_ENV)
    if explicit is not None:
        p = Path(explicit).expanduser().resolve() if explicit else None
        if p is None or not p.exists():
            raise FileNotFoundError(
                f"{NODE_DIST_PATH_ENV} is set to {explicit!r}, which is not an "
                f"existing path. An explicit dist is never replaced by a fallback "
                f"checkout: point it at a built dist/coordinator.js, or unset it "
                f"to use ../agent-coherence-plugin or ~/projects/agent-coherence-plugin."
            )
        return p
    sibling = (REPO_ROOT.parent / "agent-coherence-plugin" / "dist" / "coordinator.js").resolve()
    if sibling.exists():
        return sibling
    home_fallback = (Path.home() / "projects" / "agent-coherence-plugin" / "dist" / "coordinator.js").resolve()
    if home_fallback.exists():
        return home_fallback
    return None


# ----------------------------------------------------------------------
# Request execution
# ----------------------------------------------------------------------


def execute_request(
    backend: CoordinatorBackend,
    request: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Issue a fixture's request against ``backend``. Returns (status, body_dict).

    Adds the Authorization + Host headers automatically. Body is JSON-encoded
    for POST; ignored for GET."""
    method = request.get("method", "POST").upper()
    path = request["path"]
    body = request.get("body")

    headers = {
        "Authorization": f"Bearer {backend.secret()}",
        "Host": "127.0.0.1",
        "Content-Type": "application/json",
    }
    headers.update(request.get("headers", {}))

    data = json.dumps(body).encode("utf-8") if body is not None else b""
    req = urlrequest.Request(
        backend.url(path),
        data=data if method == "POST" else None,
        method=method,
        headers=headers,
    )
    try:
        with urlrequest.urlopen(req, timeout=HARNESS_TIMEOUT_SEC) as resp:
            raw = resp.read().decode("utf-8")
            parsed = json.loads(raw) if raw else {}
            return resp.status, parsed
    except urlerror.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp else ""
        parsed = json.loads(raw) if raw else {}
        return exc.code, parsed


# ----------------------------------------------------------------------
# Top-level scenario runner
# ----------------------------------------------------------------------


@contextmanager
def coordinator_running(
    backend_id: str,
    workspace: Path,
    node_dist_path: Optional[Path] = None,
) -> Iterator[CoordinatorBackend]:
    """Context manager that spawns a coordinator + tears it down cleanly."""
    if backend_id == BACKEND_PYTHON:
        backend: CoordinatorBackend = PythonCoordinator()
    elif backend_id == BACKEND_NODE:
        if node_dist_path is None:
            raise RuntimeError(
                "Node backend requested but no plugin dist path resolved. "
                "Set AGENT_COHERENCE_PLUGIN_DIST_PATH or build the plugin checkout."
            )
        backend = NodeCoordinator(node_dist_path)
    else:
        raise ValueError(f"Unknown backend: {backend_id!r}")
    backend.start(workspace)
    try:
        yield backend
    finally:
        backend.shutdown()


def run_scenario(
    fixture: Fixture,
    backend_id: str,
    workspace: Path,
    node_dist_path: Optional[Path] = None,
) -> tuple[int, dict[str, Any]]:
    """End-to-end: setup workspace → spawn backend → preflight requests →
    main request → normalize → return.

    Returns the normalized (status, body) for assertion by the caller. The
    preflight_requests list in fixture.setup is fired against the same live
    coordinator BEFORE the main request — used by strict-mode fixtures to
    drive the multi-step "A reads → B preempts → A re-reads (test)" setup
    without requiring a separate fixture per step."""
    apply_setup(workspace, fixture.setup)
    with coordinator_running(backend_id, workspace, node_dist_path) as backend:
        preflight = fixture.setup.get("preflight_requests") or []
        captures = apply_preflight_requests(backend, preflight) if preflight else {}
        # The main request may name any value a preflight captured. Applied to
        # the REQUEST only — never to the expected body (R5).
        request = _substitute(
            _omit_headers_not_issued(fixture.request, captures),
            captures,
            where=f"{fixture.name} request",
        )
        status, body = execute_request(backend, request)
    return status, normalize_response(
        body,
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
        preserve_identity=fixture.preserve_identity,
    )
