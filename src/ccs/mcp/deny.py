# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""MCP-C deny-contract mapper (stale-write-guard-fs, Unit 1).

Translates every fail-closed coherence terminal into a *non-ignorable* MCP tool
result — ``CallToolResult(isError=True, structuredContent={...})`` — so a client
relays the deny to the model as a business-logic error it self-corrects on,
NEVER as a success. Degrade/deny are HTTP-200 bodies inside the coordinator, so a
mapping bug here silently reintroduces the lost update: this is the load-bearing
first deliverable, built and tested before any tool binding.

Two input types (plan §"Deny reason-flow"):

- **Type A** — typed adapter exceptions, matched by EXACT ``type(exc)`` and read
  via the ``.reason`` constant carried on the exception class — or, for the two
  types that carry the coordinator's reason per INSTANCE (``CasVersionConflict``,
  ``CallerPrincipalRefused``), by that reason's exact membership in its
  vocabulary. The exception
  *message* stays the verbatim coordinator ``permissionDecisionReason`` (it is
  surfaced as ``detail``; the model's retry loop depends on that byte-stability,
  auto-memory ``project_cc_strict_mode_retry_hazard``). An unrecognized
  ``CoherenceError`` (a CAS corruption raise, a path escape) fails closed as
  ``internal_error`` — the mapper never lets an exception become a success.
- **Type B** — synthesized here (NO adapter raise-site) when the volume is
  unattached / the coordinator transport failed → ``coordinator_unavailable``.

The ``reason`` token is NEVER substring-matched off the message (the
typed-signal-not-substring house rule).
"""

from __future__ import annotations

from dataclasses import dataclass

from mcp.types import CallToolResult, TextContent

from ccs.core.exceptions import (
    CALLER_PRINCIPAL_REASONS,
    CAS_EXHAUSTED_REASON,
    COORDINATOR_UNAVAILABLE_REASON,
    OCC_CALLER_TRANSIENT_REASON,
    STALE_READ_GENERATION_REASON,
    VERSION_MISMATCH_REASON,
    CallerPrincipalRefused,
    CasRetriesExhausted,
    CasVersionConflict,
    CommitPreempted,
    CommitUnconfirmed,
    GiverFenced,
    InternalConcurrencyError,
    StaleView,
    ViewWedged,
)


@dataclass(frozen=True)
class _Terminal:
    """The wire shape for one deny terminal: a typed ``reason``, a ``recover``
    verb the agent can act on, and whether a bare retry is safe. ``next_step``
    is fixed text for a terminal whose verb needs words beside the verbatim
    detail; it is carried as ``next_step`` and as a second text item."""

    reason: str
    recover: str
    retryable: bool
    next_step: str | None = None


# The giver of a live handoff (#185) wrote a path it handed off. No reacquire,
# re-read or re-merge clears the fence -- it is keyed on the session -- so the
# verb is a stop, and the words say to report. They never name withdraw: the
# MCP giver's own withdraw tool is one call away, and a refusal that names the
# act that lifts it invites the agent to take it. Withdraw is the user's or
# host's call, and its tool says so.
GIVER_FENCED_RECOVER = "stop_and_report"
GIVER_FENCED_NEXT_STEP = (
    "Stop: this session handed this path off, so it can no longer write it, and "
    "no retry, reacquire or re-read changes that. Do not write it by any other "
    "route. Report to your user or host that the path was handed off to the "
    "successor named here, at the version named here."
)


# Type A — matched by EXACT exception type (so a future subclass cannot silently
# inherit a mapping). ``reason`` mirrors the constant carried on the exception
# class itself; recover/retryable are this layer's contract.
_TERMINALS: dict[type, _Terminal] = {
    StaleView: _Terminal(StaleView.reason, "reacquire", True),
    CommitPreempted: _Terminal(CommitPreempted.reason, "reacquire_and_reconcile", False),
    ViewWedged: _Terminal(ViewWedged.reason, "wait_or_escalate", False),
    CommitUnconfirmed: _Terminal(CommitUnconfirmed.reason, "read_then_retry", False),
    CasRetriesExhausted: _Terminal(CasRetriesExhausted.reason, "stop", False),
    InternalConcurrencyError: _Terminal(InternalConcurrencyError.reason, "none", False),
    # Both write routes raise it (swg_write's pre-edit, swg_write_cas's commit).
    GiverFenced: _Terminal(
        GiverFenced.reason, GIVER_FENCED_RECOVER, False, GIVER_FENCED_NEXT_STEP
    ),
}

# A CAS refusal is four different terminals wearing one exception type. The
# recover verb is what the agent acts on, so it is keyed on the coordinator's
# reason rather than on the exception class. ``version_mismatch`` keeps its
# existing contract exactly; the other three used to inherit it and send the
# caller to re-merge, which cannot make progress on any of them.
_CAS_TERMINALS: dict[str, _Terminal] = {
    VERSION_MISMATCH_REASON: _Terminal(
        VERSION_MISMATCH_REASON, "read_then_merge", False
    ),
    # The holder will release; a bounded backoff is the recovery, not a merge.
    "other_holder": _Terminal("other_holder", "wait_and_retry", True),
    STALE_READ_GENERATION_REASON: _Terminal(
        STALE_READ_GENERATION_REASON, "reacquire_and_reread", False
    ),
    OCC_CALLER_TRANSIENT_REASON: _Terminal(
        OCC_CALLER_TRANSIENT_REASON, "reacquire", False
    ),
}

# A caller-principal refusal is three terminals wearing one exception type,
# keyed like the CAS refusals on the reason the instance carries (one of
# ``CALLER_PRINCIPAL_REASONS``, matched by exact membership). One recover verb
# fits all three when the refusal is SETTLED (``exc.settled``, the default):
# the coordinator's answer is definite and durable for this server session —
# the volume presents the one principal it holds, and once its session is
# known to be bound under another nonce it claims nothing again — so no tool
# call in the session can regain coordination; a new server session claims its
# own. ``retryable`` is false for the same reason. Without this branch every
# later swg_* call read ``internal_error`` / ``none``, the shape of a
# coordinator bug, for a state that is typed and the session's own.
PRINCIPAL_REFUSED_RECOVER = "restart_session"

# The refusal that is NOT settled: the request was refused, but the recovery
# claim's answer was lost, so the volume adopted ``unconfirmed`` and its very
# next request claims again with the same nonce by itself — the next tool call
# may simply succeed, and ``swg_status`` reads ``principal_claim: unconfirmed``
# meanwhile. The verb is the existing one for "the state clears on its own,
# call again" (``other_holder`` uses it), ``retryable`` true. Answering this arm
# ``restart_session`` over-claimed durability: it sent the agent to a new
# session for a state the next call cures.
PRINCIPAL_UNSETTLED_RECOVER = "wait_and_retry"

# Fallback for any unrecognized exception (an unexpected ``CoherenceError`` such
# as a CAS corruption raise or a path escape): fail closed as a generic internal
# error — never a success, never a recoverable ``stale_view``.
_UNRECOGNIZED = _Terminal("internal_error", "none", False)


def _result(terminal: _Terminal, detail: str, extra: dict | None = None) -> CallToolResult:
    """Build the non-ignorable tool result. ``detail`` is the verbatim deny prose,
    carried in BOTH ``structuredContent`` (structured clients) and the text
    content (clients that surface only the text channel still see the deny).
    ``extra`` merges terminal-specific fields (e.g. the CAS versions)."""
    structured = {
        "reason": terminal.reason,
        "recover": terminal.recover,
        "retryable": terminal.retryable,
        "detail": detail,
    }
    content = [TextContent(type="text", text=detail)]
    if terminal.next_step is not None:
        structured["next_step"] = terminal.next_step
        content.append(TextContent(type="text", text=terminal.next_step))
    if extra:
        structured.update(extra)
    return CallToolResult(isError=True, content=content, structuredContent=structured)


def deny_result(exc: BaseException) -> CallToolResult:
    """Map a coherence terminal (Type A) to a non-ignorable ``isError`` result,
    preserving the coordinator deny text verbatim in ``detail``."""
    if isinstance(exc, CasVersionConflict):
        # Surface both versions so the agent can re-read at current_version and
        # re-CAS without another round-trip. Typed-conflict, NOT auto-merge.
        # ``exc.reason`` is the coordinator's own refusal reason (the class
        # default when the raise site had none), never substring-matched off
        # the message. An unknown reason falls back to the version_mismatch
        # contract rather than to internal_error — it is still a real conflict.
        terminal = _CAS_TERMINALS.get(
            exc.reason, _CAS_TERMINALS[VERSION_MISMATCH_REASON]
        )
        return _result(
            terminal,
            str(exc),
            {"expected_version": exc.expected_version, "current_version": exc.current_version},
        )
    if type(exc) is CallerPrincipalRefused and exc.reason in CALLER_PRINCIPAL_REASONS:
        # Exact type, like ``_TERMINALS``; the reason by exact membership, so
        # a string outside the vocabulary never rides to the model as a
        # ``reason`` (it falls to ``_UNRECOGNIZED`` below). The message is the
        # client's own prose, built from constants: no principal, no nonce.
        # Whether the refusal is settled is the typed attribute, never prose.
        if exc.settled:
            return _result(_Terminal(exc.reason, PRINCIPAL_REFUSED_RECOVER, False), str(exc))
        return _result(_Terminal(exc.reason, PRINCIPAL_UNSETTLED_RECOVER, True), str(exc))
    terminal = _TERMINALS.get(type(exc), _UNRECOGNIZED)
    # The giver terminal also carries who the path went to and at which version
    # as values the agent can report (the detail says both in prose). Only
    # through its own row: without one it fails closed like any unknown type.
    extra = (
        _giver_fields(exc)
        if isinstance(exc, GiverFenced) and terminal is not _UNRECOGNIZED
        else None
    )
    return _result(terminal, str(exc), extra)


def _giver_fields(exc: GiverFenced) -> dict:
    """The successor (the wire string a client raises with) and the version
    at transfer, each ``None`` when the refusal did not carry it typed."""
    successor = exc.successor if isinstance(exc.successor, str) else None
    version = exc.version_at_transfer
    if isinstance(version, bool) or not isinstance(version, int):
        version = None
    return {"successor": successor, "version_at_transfer": version}


def coordinator_unavailable_result(detail: str) -> CallToolResult:
    """Type B (synthesized): no adapter raised, but the volume is unattached or
    the coordinator transport failed. Fail closed — the write is NOT
    version-committed."""
    return _result(
        _Terminal(COORDINATOR_UNAVAILABLE_REASON, "retry_later", False),
        detail,
    )


def cas_exhausted_result(detail: str) -> CallToolResult:
    """Synthesized: the per-session conflict counter bounded a COOPERATING agent
    (too many consecutive version_mismatch conflicts on one path). ``retryable``
    is false so a cooperating agent stops. The no-livelock bound is scoped to
    cooperating agents — a fresh session or one that ignores ``retryable:false``
    resets it (coordinator-side fencing → v1.1)."""
    return _result(_Terminal(CAS_EXHAUSTED_REASON, "stop", False), detail)
