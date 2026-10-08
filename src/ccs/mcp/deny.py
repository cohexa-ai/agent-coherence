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
- **Type C** — a handoff tool's typed refusal (#185), a value the volume
  returns rather than raises: mapped by exact reason through
  :data:`HANDOFF_REFUSALS`, each row not retryable and carrying a fixed
  ``next_step``; an unknown reason fails closed as ``internal_error``.

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
    HANDOFF_ENDED_REASON,
    HANDOFF_IN_FLIGHT_REASON,
    HANDOFF_NOT_GIVER_REASON,
    HANDOFF_NOT_HELD_REASON,
    HANDOFF_NOT_LIVE_REASON,
    HANDOFF_NOT_SUCCESSOR_REASON,
    HANDOFF_OTHER_HOLDER_REASON,
    HANDOFF_SELF_REASON,
    HANDOFF_SUCCESSOR_MALFORMED_REASON,
    HANDOFF_SUCCESSOR_UNKNOWN_REASON,
    HANDOFF_TRANSFER_REFUSAL_REASONS,
    HANDOFF_VERSION_UNCONFIRMED_REASON,
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

# A handoff verb whose answer did not settle its outcome (#276). It may have
# landed, and a transfer sent again after the path was read or written can be a
# NEW handoff at the new version, so the four handoff tools send the agent to
# the path's handoff record, never to the generic read-then-retry. Each verb
# has its own next step: what "it landed" looks like differs per verb, and only
# a transfer is unsafe to repeat (a repeated accept, decline or withdraw that
# already landed changes nothing).
HANDOFF_UNCONFIRMED_RECOVER = "check_handoff"
_LOOK_FIRST = (
    "It may have landed. Look at the path's handoff first (the handoff key of "
    "swg_read, or swg_status). "
)
HANDOFF_UNCONFIRMED_NEXT_STEPS: dict[str, str] = {
    # The version qualifier tells this transfer from an earlier round's ended
    # record between the same two sessions.
    "transfer": (
        "Do not repeat this call yet. " + _LOOK_FIRST + "A handoff from you to that "
        "successor made at the version you held (its version_at_transfer), live or "
        "ended, means your transfer landed: do not transfer again, and do not "
        "withdraw to start over. Transfer again only if no such handoff shows and "
        "the path is still at the version you held; if it was written since, or "
        "shows another session's handoff, ask your user first."
    ),
    "accept": (
        _LOOK_FIRST + "Status completed means your accept landed. If the handoff "
        "is still pending and names you as successor, call swg_accept again: a "
        "repeat changes nothing once it has landed."
    ),
    "decline": (
        _LOOK_FIRST + "Status declined means your decline landed. If the handoff "
        "is still live and names you as successor, call swg_decline again: a "
        "repeat changes nothing once it has landed."
    ),
    "withdraw": (
        _LOOK_FIRST + "Status withdrawn means your withdraw landed. If the handoff "
        "is still live and names you as giver, call swg_withdraw again, still only "
        "on your user's or host's instruction: a repeat changes nothing once it "
        "has landed."
    ),
}


# A handoff tool's refusal: a typed value the volume returns (Type C), not a
# raise. Nothing changed, so none is retryable with the same call, and every
# row has a fixed next step, because the refusals that a timer would clear do
# not exist (a live handoff and a foreign write hold have none) and the obvious
# moves are the dangerous ones: re-sending a transfer that already landed hands
# the path on a second time, and writing a path to end a handoff overtakes it.
# The successor rows say how to name a session, the not-held row sends the
# agent to the record first (a landed transfer answers exactly this), and every
# other row stops and reports.
FIX_SUCCESSOR_RECOVER = "fix_successor"
_FIX_SUCCESSOR_TAIL = (
    "Do not guess, and do not retry the same id. If you do not have the right "
    "one, ask your user or host."
)
_TWO_SESSIONS = (
    "this model's other session (a model with both the Claude Code hooks and this "
    "server is two sessions); or the server session this one replaced (a "
    "restarted server is a new session and cannot act for the old one)"
)
HANDOFF_REFUSALS: dict[str, _Terminal] = {
    HANDOFF_SELF_REASON: _Terminal(HANDOFF_SELF_REASON, FIX_SUCCESSOR_RECOVER, False, (
        "Nothing was handed off and your claims are as they were: the successor you "
        "named is this session itself (the session_agent_id this session's own "
        "swg_status reports, or an id that resolves to this session). Name the other "
        "session by the session_agent_id that session's own swg_status reports. If "
        "you do not have it, ask your user or host, and do not guess one."
    )),
    HANDOFF_SUCCESSOR_UNKNOWN_REASON: _Terminal(HANDOFF_SUCCESSOR_UNKNOWN_REASON, FIX_SUCCESSOR_RECOVER, False, (
        "Nothing was handed off and your claims are as they were: the coordinator "
        "knows no session by that id. Use the session_agent_id that the successor's "
        "own swg_status reports while it shows principal_claim: bound, not a "
        "session_id and not a subagent's id. " + _FIX_SUCCESSOR_TAIL
    )),
    HANDOFF_SUCCESSOR_MALFORMED_REASON: _Terminal(HANDOFF_SUCCESSOR_MALFORMED_REASON, FIX_SUCCESSOR_RECOVER, False, (
        "Nothing was handed off and your claims are as they were: the successor is "
        "not an agent id. A session_agent_id is a UUID, hyphenated or 32 hex digits, "
        "taken from the successor's own swg_status. Do not pass a name or a shortened "
        "id. " + _FIX_SUCCESSOR_TAIL
    )),
    HANDOFF_NOT_HELD_REASON: _Terminal(HANDOFF_NOT_HELD_REASON, HANDOFF_UNCONFIRMED_RECOVER, False, (
        "Nothing changed: this session holds no claim on this path, and a transfer of "
        "it that already landed gets exactly this answer. Look at the path's handoff "
        "in swg_status first: it lists every record, while swg_read's handoff key is "
        "best-effort, so a read without one proves nothing. A handoff from you to that "
        "successor made at the version you held (its version_at_transfer), live or "
        "ended, means your transfer landed: do not transfer again, and do not withdraw "
        "to start over. Ask your user or host first if any other handoff shows, if "
        "swg_status reports the path past the version you last held, or if you have no "
        "version of your own to compare. Only if no handoff shows and swg_status "
        "reports the path at the version you last held: swg_reacquire the path to take "
        "a claim (a plain swg_read after a lost claim takes none), then transfer it "
        "again."
    )),
    HANDOFF_VERSION_UNCONFIRMED_REASON: _Terminal(HANDOFF_VERSION_UNCONFIRMED_REASON, GIVER_FENCED_RECOVER, False, (
        "Nothing changed and this session still holds its claim: the coordinator has "
        "no confirmed version for this path, so there is no version to fence a handoff "
        "on, and a retry, re-read or reacquire does not give it one. Do not write the "
        "path just to give it a version. Report to your user or host that this path "
        "could not be handed off."
    )),
    HANDOFF_IN_FLIGHT_REASON: _Terminal(HANDOFF_IN_FLIGHT_REASON, GIVER_FENCED_RECOVER, False, (
        "Nothing changed: another session's handoff of this path is live (its giver "
        "and successor are named here), and until it ends only its giver can hand the "
        "path on. It has no timer, so a retry gets the same answer until someone acts. "
        "Do not end it yourself: do not write the path, and do not call another "
        "handoff tool to make room for this transfer. Report to your user or host that "
        "the path is in another session's handoff."
    )),
    HANDOFF_OTHER_HOLDER_REASON: _Terminal(HANDOFF_OTHER_HOLDER_REASON, GIVER_FENCED_RECOVER, False, (
        "Nothing changed: another session holds this path for writing, so this "
        "transfer cannot hand it on now. That session may keep its hold for a long "
        "time, and its next write ends any claim you have on the path, so retrying "
        "will not get you through. Do not retry in a loop, and do not write the path "
        "to clear it. Report to your user or host that another session holds the "
        "path. If you are told to hand it on later, look at its handoff and version "
        "in swg_status first."
    )),
    HANDOFF_ENDED_REASON: _Terminal(HANDOFF_ENDED_REASON, GIVER_FENCED_RECOVER, False, (
        "Nothing changed: your earlier handoff of this path to this successor, at this "
        "version, has ended, as its status says: declined (the successor refused it), "
        "withdrawn, or superseded (you have since handed the path to another "
        "session). Sending it again gets the same answer. Do not send it again, do not "
        "hand the path to another session, and do not write it to get past this. "
        "Report the status to your user or host and ask what to do next."
    )),
    HANDOFF_NOT_SUCCESSOR_REASON: _Terminal(HANDOFF_NOT_SUCCESSOR_REASON, GIVER_FENCED_RECOVER, False, (
        "Nothing changed: this session is not the successor this live handoff names, "
        "so it cannot accept or decline it, and a retry gets the same answer. The "
        "handoff may name another session; " + _TWO_SESSIONS + ". Writing the path "
        "would overtake the handoff, not take it, so do not write it or transfer it to "
        "take the handoff over. Report to your user or host which session the handoff "
        "names as successor (swg_status shows it)."
    )),
    HANDOFF_NOT_GIVER_REASON: _Terminal(HANDOFF_NOT_GIVER_REASON, GIVER_FENCED_RECOVER, False, (
        "Nothing changed: this session is not the giver this live handoff names, so "
        "it cannot withdraw it, and a retry gets the same answer. The giver may be "
        "another session; " + _TWO_SESSIONS + ". Do not end the handoff another way: "
        "do not write the path, decline it, or call another handoff tool. Report to "
        "your user or host that only the giving session can withdraw it (swg_status "
        "names it)."
    )),
    HANDOFF_NOT_LIVE_REASON: _Terminal(HANDOFF_NOT_LIVE_REASON, GIVER_FENCED_RECOVER, False, (
        "Nothing changed: there is no live handoff of this path to act on, so do not "
        "repeat the call; repeating it gets the same answer. The status says how the "
        "handoff ended: declined, withdrawn, or completed or overtaken once a write "
        "moved the version. No status means the path has no handoff record. If the "
        "status is the one this call sets (completed for swg_accept, declined for "
        "swg_decline, withdrawn for swg_withdraw) and swg_status names this session as "
        "that handoff's successor (its giver, for swg_withdraw), an earlier call or "
        "write of yours already did it. Do not write the path as if it had been handed "
        "to you unless that handoff completed with this session as its successor. "
        "Report the status to your user or host."
    )),
}
"""Every handoff refusal reason a tool answers, by exact reason."""

_HANDOFF_VERB_REFUSALS: frozenset[str] = frozenset(
    {HANDOFF_NOT_SUCCESSOR_REASON, HANDOFF_NOT_GIVER_REASON, HANDOFF_NOT_LIVE_REASON}
)

# Which refused grant of a partly refused transfer speaks for the whole
# result: the most restrictive verb first, so an agent that obeys only the top
# level never takes a step riskier than some refused grant allows. A tie goes
# to the first refused grant in request order.
_RECOVER_RANK: dict[str, int] = {
    GIVER_FENCED_RECOVER: 0,
    "none": 0,  # an unrecognized grant reason fails closed with the stops
    HANDOFF_UNCONFIRMED_RECOVER: 1,
    FIX_SUCCESSOR_RECOVER: 2,
}
HANDOFF_PARTIAL_TRANSFER_LEAD = (
    "Not every path was handed off. A path listed as transferred was handed off: "
    "never send it again."
)


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


def handoff_deny_result(exc: BaseException, verb: str) -> CallToolResult:
    """:func:`deny_result` for the four handoff tools, except that an
    unconfirmed ``verb`` answers ``check_handoff`` with that verb's next step
    rather than ``read_then_retry`` (exact type, like ``_TERMINALS``)."""
    if type(exc) is CommitUnconfirmed:
        terminal = _Terminal(
            CommitUnconfirmed.reason,
            HANDOFF_UNCONFIRMED_RECOVER,
            False,
            HANDOFF_UNCONFIRMED_NEXT_STEPS[verb],
        )
        return _result(terminal, str(exc))
    return deny_result(exc)


def handoff_verb_refusal_result(reason: object, detail: str, extra: dict) -> CallToolResult:
    """A refused accept, decline or withdraw: ``reason`` mapped through
    :data:`HANDOFF_REFUSALS` by exact membership in the three verb refusals,
    and anything else failing closed as ``internal_error``. ``extra`` carries
    the result's own fields (path, ok, status, counterparty); a ``reason`` in
    it is dropped, so an unvetted string cannot reach the top level."""
    terminal = _refusal_terminal(reason, _HANDOFF_VERB_REFUSALS)
    return _result(terminal, detail, {k: v for k, v in extra.items() if k != "reason"})


def _refusal_terminal(reason: object, vocabulary: frozenset[str]) -> _Terminal:
    """``reason``'s row when it is one of ``vocabulary``, the refusals the
    calling tool can answer; anything else, a vocabulary reason with no row
    included, fails closed."""
    if isinstance(reason, str) and reason in vocabulary:
        return HANDOFF_REFUSALS.get(reason, _UNRECOGNIZED)
    return _UNRECOGNIZED


def handoff_transfer_refusal_result(grants: list[dict], detail: str) -> CallToolResult:
    """A transfer with at least one refused grant. Each refused grant keeps
    every field it was answered with and adds its row's ``recover``,
    ``retryable`` and ``next_step``; the top level speaks for the most
    restrictive refused grant (:data:`_RECOVER_RANK`), and when any grant
    transferred its next step opens by saying never to send those again. The
    other refused reasons' next steps follow as text items, labelled by
    reason when more than one reason was refused."""
    rows = [
        None if grant.get("transferred")
        else _refusal_terminal(grant.get("reason"), HANDOFF_TRANSFER_REFUSAL_REASONS)
        for grant in grants
    ]
    entries = [grant if row is None else _with_row(grant, row) for grant, row in zip(grants, rows)]
    refused = [row for row in rows if row is not None]
    chosen = min(refused, key=_rank)
    lead = HANDOFF_PARTIAL_TRANSFER_LEAD if len(refused) < len(grants) else None
    next_step = " ".join(step for step in (lead, chosen.next_step) if step) or None
    result = _result(
        _Terminal(chosen.reason, chosen.recover, False, next_step), detail,
        {"ok": False, "grants": entries},
    )
    distinct = sorted({row.reason: row for row in refused}.values(), key=_rank)
    for row in distinct:
        if row is not chosen and row.next_step is not None:
            text = f"[{row.reason}] {row.next_step}" if len(distinct) > 1 else row.next_step
            result.content.append(TextContent(type="text", text=text))
    return result


def _rank(row: _Terminal) -> int:
    """A row's place in :data:`_RECOVER_RANK`; a verb with no place ranks with
    the stops, the most restrictive."""
    return _RECOVER_RANK.get(row.recover, 0)


def _with_row(grant: dict, row: _Terminal) -> dict:
    entry = {**grant, "recover": row.recover, "retryable": row.retryable}
    if row.next_step is not None:
        entry["next_step"] = row.next_step
    return entry


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
