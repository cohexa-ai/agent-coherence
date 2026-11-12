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
  vocabulary. The split-pair read refusal also carries a reason per instance
  (``split_pair``, #264), but it is the client's own name for one kind of
  ``StaleView``, not a coordinator reason, so it is answered by the class
  constant like every other ``StaleView``. The exception
  *message* stays the verbatim coordinator ``permissionDecisionReason`` (it is
  surfaced as ``detail``; the model's retry loop depends on that byte-stability,
  auto-memory ``project_cc_strict_mode_retry_hazard``). An unrecognized
  ``CoherenceError`` (a CAS corruption raise, a path escape) fails closed as
  ``internal_error`` — the mapper never lets an exception become a success.
- **Type B** — synthesized here, with no exception, when the volume is
  unattached → ``coordinator_unavailable`` with ``retry_later``. The same
  reason also arrives as Type A: an attached strict volume raises
  ``CoordinatorUnreachable`` when its coordinator gives no usable answer, and
  that row stops and reports instead.
- **Type C** — a handoff tool's typed refusal (#185), a value the volume
  returns rather than raises: mapped by exact reason through
  :data:`HANDOFF_REFUSALS`, each row not retryable and carrying a fixed
  ``next_step``; an unknown reason fails closed as ``internal_error``.

Some tools map through their own function first. ``swg_write`` goes through
:func:`write_deny_result`, whose rows differ where a write may leave a grant
standing or its bytes on disk, or meets a refuse-mode answer; ``swg_release``
through :func:`release_deny_result`, whose unconfirmed release is safe to
send again; and the handoff tools through :func:`handoff_deny_result`.
Anything else falls to :func:`deny_result`.

The ``reason`` token is NEVER substring-matched off the message (the
typed-signal-not-substring house rule).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
    CoherenceDirRefused,
    CoherenceError,
    CommitPreempted,
    CommitUnconfirmed,
    CoordinatorFailed,
    CoordinatorUnreachable,
    GiverFenced,
    InternalConcurrencyError,
    LinkOutOfManaged,
    OtherHolder,
    RedirectRefused,
    StaleView,
    ViewWedged,
    _generation_pair_fields,
)

if TYPE_CHECKING:
    from ccs.adapters.coherent_volume import OnHeld


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

# An attached strict volume got no usable answer from its coordinator. The
# volume keeps the endpoint it attached to and never reaches a coordinator
# started after this one died, so no call in this server session gets through:
# the verb is the house stop, not the unattached refusal's ``retry_later``.
# Like the giver's words, these never name a retry, since a refusal that names
# the act that lifts it invites the agent to take it. The restart is the user's
# or host's to do, so the words say to report it.
COORDINATOR_UNREACHABLE_RECOVER = GIVER_FENCED_RECOVER
COORDINATOR_UNREACHABLE_NEXT_STEP = (
    "Stop: this server got no usable answer from its coordinator, which is gone "
    "or wedged, and the server cannot recover from that by itself. Do not write "
    "the managed files by any other route. Report to your user or host that the "
    "coordinator is gone and that they must restart this MCP server, stopping "
    "the workspace's coordinator process first if the server will not start."
)

# The coordinator answered a read, or the volume's release of an earlier write
# grant before a CAS or a reacquire (#401), with its failure envelope (#259) or
# an HTTP 5xx (#433); or a commit was refused by the volume's own earlier
# grant, which that release did not clear (#434). None of these says whether
# its cause clears by itself (the envelope names only an exception's type, a
# status names nothing), so a busy store looks the same as a fault that repeats
# on every call. The verb is the house stop, with one retry named in the words:
# an unbounded ``wait_and_retry`` would loop forever on the fault. ``retryable``
# stays false so a client that retries on that flag alone does not loop either.
# swg_write's own answers of this type have their own rows (below).
COORDINATOR_FAILED_RECOVER = GIVER_FENCED_RECOVER
COORDINATOR_FAILED_NEXT_STEP = (
    "The coordinator failed while answering this call, so nothing was confirmed. "
    "Make the same call once more after a few seconds: a busy coordinator store "
    "clears by itself. If it fails the same way again, stop and report to your "
    "user or host that the coordinator is failing; do not keep retrying."
)

# A person must replace the workspace's .coherence (#498), so the verb is the
# house stop. The row maps the refusal for library callers of deny_result: the
# server itself refuses to start before any tool runs, and its tools answer only
# from an attached volume, so no tool call carries this refusal.
COHERENCE_DIR_REFUSED_RECOVER = GIVER_FENCED_RECOVER
COHERENCE_DIR_REFUSED_NEXT_STEP = (
    "Stop: nothing was written. This workspace's .coherence is a symlink or not a "
    "directory, or holds a symlink, FIFO, socket or device directly inside it. Do not "
    "write the managed files by any other route. Report to your user or host that a "
    "person must replace it with a real directory, or mount one at .coherence, and "
    "then restart this MCP server."
)


# A refuse-mode write another agent holds, from a volume that opted in. Only
# write() raises it. swg_write resends the same bytes, and once the path comes
# free a bare retry lands them over whatever the holder committed (measured), so
# the flag is false and the verb puts the wait before a reacquire that rebuilds
# from the holder's bytes. The words give the wait in order. Holds go first:
# two agents that keep theirs while each waits on the other wait out the
# 30-minute ceiling, then overwrite each other blind (measured). The poll is
# swg_status, which changes nothing, because a refused write refreshes the
# waiter's other holds. The 30 minutes end the poll because the Node
# coordinator takes back a grant that old only when a write asks for the path.
# The server never waits itself, and the words give no total bound: the
# agent's own bound ends the wait, and sizing one needs a real caller.
OTHER_HOLDER_RECOVER = "wait_then_reacquire"
OTHER_HOLDER_NEXT_STEP = (
    "Nothing was written and nothing changed hands: another agent holds this path for "
    "writing, and holder_agent_id names it. Do not send this write again as it is: once "
    "the path comes free, these bytes would land over what the holder committed. Do this "
    "in order. (1) Before you wait, call swg_release on every path swg_status lists in "
    "held, including any an earlier attempt of yours holds; if a release answers "
    "release_unconfirmed twice, or held is null because swg_status cannot tell who holds "
    "each path, call swg_reacquire on this path instead, which drops every hold you "
    "have. Two agents that "
    "keep their holds while each waits for the other wait 30 minutes and then overwrite "
    "each other's work. (2) Poll swg_status, which changes nothing, and not swg_write, "
    "whose refusal keeps your other holds alive: wait at least 4 seconds before each "
    "poll and double the wait each time up to 60 seconds, with jitter, until swg_status "
    "lists this path, with held not null, and no longer shows it held_by another_agent, "
    "or 30 minutes have passed (the "
    "Claude Code plugin's coordinator takes back a "
    "grant that old only when a write asks for the path). (3) Then call swg_reacquire on "
    "the path, rebuild your content from the bytes it returns, and call swg_write once; "
    "if it is refused other_holder again, go back to step 2. A live holder can keep a "
    "path for up to 30 minutes, and the next holder starts a new clock. A holder that "
    "stopped clears in about 10 minutes on this package's coordinator and in up to 30 "
    "minutes on the Claude Code plugin's. The holder may be you: if you edited this file "
    "with your own file tools this turn, it may be your own Claude Code hooks session, "
    "which lets go when your turn ends, so waiting inside this turn does not clear it; "
    "if this server restarted in the last 30 minutes, it may be this server's previous "
    "run. If your own bound on the wait runs out, stop and report to your user or host "
    "that another agent holds the path."
)

# A refuse-mode write through a link whose name is managed to a file that is
# not, refused before anything was sent. Only write() raises it. The detail
# names the routes around the refusal (the target's own path, a volume that
# may displace), so the words forbid them and send the agent to report.
LINK_OUT_OF_MANAGED_RECOVER = GIVER_FENCED_RECOVER
LINK_OUT_OF_MANAGED_NEXT_STEP = (
    "Stop: nothing was sent or written. This path goes through a link whose name is "
    "among the managed files to a file outside them, where this server cannot keep the "
    "file from an agent that holds it. Do not write that file by any other route: not "
    "by its own path through this server, not through another server, and not with your "
    "own file tools or a shell. Report to your user or host that this link leads out of "
    "the managed files (the detail names the link and its target)."
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
    OtherHolder: _Terminal(OtherHolder.reason, OTHER_HOLDER_RECOVER, False, OTHER_HOLDER_NEXT_STEP),
    LinkOutOfManaged: _Terminal(
        LinkOutOfManaged.reason,
        LINK_OUT_OF_MANAGED_RECOVER,
        False,
        LINK_OUT_OF_MANAGED_NEXT_STEP,
    ),
    CoordinatorUnreachable: _Terminal(
        CoordinatorUnreachable.reason,
        COORDINATOR_UNREACHABLE_RECOVER,
        False,
        COORDINATOR_UNREACHABLE_NEXT_STEP,
    ),
    CoordinatorFailed: _Terminal(
        CoordinatorFailed.reason,
        COORDINATOR_FAILED_RECOVER,
        False,
        COORDINATOR_FAILED_NEXT_STEP,
    ),
    CoherenceDirRefused: _Terminal(
        CoherenceDirRefused.reason,
        COHERENCE_DIR_REFUSED_RECOVER,
        False,
        COHERENCE_DIR_REFUSED_NEXT_STEP,
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
# How a giver tells that a transfer it is unsure of landed. One copy, used by
# every text that gives the rule (the unconfirmed transfer, the not-held
# refusal, the transfer tool's description): a tightening that missed a copy
# would leave an agent on the old rule, and that failure is a second handoff.
HANDOFF_LANDED_TRANSFER = (
    "A handoff from you to that successor made at the version you held (its "
    "version_at_transfer), live or ended, means your transfer landed: do not "
    "transfer again, and do not withdraw to start over. "
)
HANDOFF_UNCONFIRMED_NEXT_STEPS: dict[str, str] = {
    # The version qualifier tells this transfer from an earlier round's ended
    # record between the same two sessions.
    "transfer": (
        "Do not repeat this call yet. " + _LOOK_FIRST + HANDOFF_LANDED_TRANSFER
        + "Transfer again only if no such handoff shows and "
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
        "best-effort, so a read without one proves nothing. " + HANDOFF_LANDED_TRANSFER
        + "Ask your user or host first if any other handoff shows, if "
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
    # The holder will release; a bounded backoff is the recovery, not a merge,
    # and a bare retry is safe because each CAS re-derives from the version it
    # reads. A refuse-mode pre-edit's refusal has its own row, by exception
    # type, whose resent bytes would not re-derive.
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

# The refusal that is NOT settled: the request was refused, and the recovery
# claim was answered without binding (a redirect, an HTTP status, an
# unconfirmed body), so the volume adopted ``unconfirmed`` and its very next
# request claims again with the same nonce by itself — the next tool call may
# simply succeed, and ``swg_status`` reads ``principal_claim: unconfirmed``
# meanwhile. The verb is the existing one for "the state clears on its own,
# call again" (a CAS commit's ``other_holder`` uses it), ``retryable`` true.
# Answering this arm ``restart_session`` over-claimed durability: it sent the
# agent to a new session for a state the next call cures. A recovery claim
# that gets no answer at all raises ``CoordinatorUnreachable`` on the strict
# volume and takes that row's stop instead.
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
            {
                "expected_version": exc.expected_version,
                "current_version": exc.current_version,
                **_generation_pair_fields(exc),
            },
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
    extra: dict | None = None
    if isinstance(exc, GiverFenced) and terminal is not _UNRECOGNIZED:
        extra = _giver_fields(exc)
    elif isinstance(exc, OtherHolder) and terminal is not _UNRECOGNIZED:
        extra = _other_holder_fields(exc)
    elif type(exc) in (CasRetriesExhausted, CommitPreempted):
        # The two commit terminals add the refusal's lost token beside their
        # unchanged recover verb. Diagnostic only: it changes no recovery.
        extra = _generation_pair_fields(exc)
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


# --- swg_write's own rows -----------------------------------------------------
#
# A write differs from the tools that share the table above: it may leave a
# grant standing or its bytes on disk, and only it meets the refuse-mode
# answers. Its rows keep every detail verbatim, and their words name the tool
# for each library method a detail names, never the method.

# Every swg_write answer after which a write grant may still stand closes with
# this. It is safe on each: a release of a path with no recorded grant sends
# nothing (measured).
WRITE_RELEASE_SENTENCE = (
    "If you stop here instead, call swg_release on the path first: it gives back any "
    "write grant this write may have left standing, which could otherwise keep another "
    "agent waiting."
)

# A refusal by this session's own earlier attempt, whose release did not land.
# write() starts no new attempt, so the same call gets the same answer. A
# reacquire, or a release of the path, sends that release again: each was
# measured clearing it on both coordinators. The row names the reacquire,
# which also returns the bytes to rebuild from; a stopping agent takes the
# release, which keeps its other holds.
WRITE_OWN_GRANT_RECOVER = "reacquire"
WRITE_OWN_GRANT_NEXT_STEP = (
    "Nothing was written: an earlier attempt of this server's own session still holds "
    "this path, because its release was not confirmed, and the coordinator refused this "
    "write because of it. No other agent holds the path, so do not wait for one, and do "
    "not send the same write again, which gets the same answer: call swg_reacquire on the "
    "path, which sends that release again, rebuild your content from the bytes it "
    "returns, and call swg_write once. " + WRITE_RELEASE_SENTENCE
)

# The coordinator answered the opted-in grant request with neither an admit nor
# a refusal. It is running, so no restart is named: the stop and its one retry
# are coordinator_failed's, and the request may have taken the grant.
WRITE_ANSWERED_UNUSABLE_NEXT_STEP = (
    "Nothing was written: the coordinator answered this write's grant request with "
    "neither an admit nor a refusal. It is running, so this server needs nothing done to "
    "it, but the request may have taken the write grant. Make the same swg_write once "
    "more after a few seconds. If it is answered the same way again, stop and report to "
    "your user or host that the coordinator answers grant requests with neither an admit "
    "nor a refusal; do not keep retrying. " + WRITE_RELEASE_SENTENCE
)

# The coordinator is gone. A commit that never left put its bytes on disk
# unrecorded, and its detail names the write that records them; a grant this
# run took stands until the coordinator takes it back.
WRITE_GONE_NEXT_STEP = (
    COORDINATOR_UNREACHABLE_NEXT_STEP + " If the detail says this write's bytes are on "
    "disk, then once the server is back call swg_reacquire on the path first and compare "
    "the bytes it returns with the ones this write sent. If they differ, another agent "
    "has written the path since: rebuild your change from the returned bytes and "
    "swg_write that. If they match, swg_write them, which records them. Never write "
    "your bytes again without that check: a restarted server has no record of what you "
    "read, so nothing else stops the write landing over another agent's commit. A "
    "write grant this server holds stays until the coordinator takes it back: "
    "swg_release cannot reach a coordinator that is gone, and a restarted server cannot "
    "release this one's grants."
)
# A restarted refuse-mode server meets those grants as another agent's: the
# Python coordinator sweeps a grant whose holder went quiet 10 minutes ago, and
# the Node one takes back a 30-minute-old grant when a write asks for it.
WRITE_GONE_REFUSE_NEXT_STEP = (
    WRITE_GONE_NEXT_STEP + " Because this server refuses to take a file another agent "
    "holds, the restarted server's writes to paths this one held can be refused "
    "other_holder until then: for about 10 minutes on this package's coordinator, and up "
    "to 30 minutes on the Claude Code plugin's."
)

# A commit that may have landed, its bytes on disk. A read refuses them until a
# write of the path is recorded, so the verb is the detail's: write again.
# ``retryable`` stays false, as on coordinator_failed, so a client that retries
# on that flag alone does not loop on a coordinator that keeps failing.
# swg_write_cas keeps read_then_retry: its unconfirmed bytes never reached disk.
WRITE_AGAIN_RECOVER = "write_again"
WRITE_AGAIN_NEXT_STEP = (
    "The write's bytes are on disk, but the coordinator may not have recorded them, and "
    "reading the path first does not help: a read refuses these bytes until a write of "
    "them is recorded. So call swg_write on the path once more, with the same content or "
    "a change built from it, which records it either way. If that write fails the same "
    "way, stop and report to your user or host that the coordinator is not confirming "
    "writes; do not keep retrying. " + WRITE_RELEASE_SENTENCE
)

# A stale write keeps the shared verb and flag: the same call is refused again,
# so the flag keeps its meaning. The words stop the two moves that defeat it.
WRITE_STALE_VIEW_NEXT_STEP = (
    "Nothing was written: the file changed since this server last read or wrote it, "
    "through another agent's commit or an edit made outside this server. To write it, "
    "call swg_reacquire on the path, rebuild your content from the bytes it returns, and "
    "swg_write that. Never send the content you just sent again: it was built on the old "
    "file and would overwrite the change. Do not re-read the file with your own file tools "
    "instead: only swg_reacquire's bytes are the version this server checks your next "
    "write against. " + WRITE_RELEASE_SENTENCE
)

# A write failure no row recognises (a 4xx, a degraded answer, an admit with a
# collision warning): nothing confirms what it did, and its grant may stand. A
# collision's detail sends a library caller to a new volume; the words forbid
# that route and name the tools that give the grant back.
WRITE_UNRECOGNIZED_NEXT_STEP = (
    "Stop: this write failed in a way this server has no recovery for, and nothing "
    "confirms what it did. Do not send it again, and do not write the path by any other "
    "route, through another server included. First call swg_release on the path, which "
    "gives back any write grant this write may have left standing; if it answers "
    "release_unconfirmed twice, call swg_reacquire on the path instead. Then report the "
    "detail to your user or host."
)

_WRITE_OWN_GRANT = _Terminal(
    CoordinatorFailed.reason, WRITE_OWN_GRANT_RECOVER, False, WRITE_OWN_GRANT_NEXT_STEP
)
_WRITE_COORDINATOR_FAILED = _Terminal(
    CoordinatorFailed.reason,
    COORDINATOR_FAILED_RECOVER,
    False,
    COORDINATOR_FAILED_NEXT_STEP + " " + WRITE_RELEASE_SENTENCE,
)
_WRITE_ANSWERED_UNUSABLE = _Terminal(
    CoordinatorUnreachable.reason, COORDINATOR_UNREACHABLE_RECOVER, False, WRITE_ANSWERED_UNUSABLE_NEXT_STEP
)
_WRITE_GONE = _Terminal(
    CoordinatorUnreachable.reason, COORDINATOR_UNREACHABLE_RECOVER, False, WRITE_GONE_NEXT_STEP
)
_WRITE_GONE_REFUSE = _Terminal(
    CoordinatorUnreachable.reason, COORDINATOR_UNREACHABLE_RECOVER, False, WRITE_GONE_REFUSE_NEXT_STEP
)
# By exact type, like ``_TERMINALS``.
_WRITE_TERMINALS: dict[type, _Terminal] = {
    CommitUnconfirmed: _Terminal(CommitUnconfirmed.reason, WRITE_AGAIN_RECOVER, False, WRITE_AGAIN_NEXT_STEP),
    StaleView: _Terminal(StaleView.reason, "reacquire", True, WRITE_STALE_VIEW_NEXT_STEP),
}
_WRITE_UNRECOGNIZED = _Terminal(
    _UNRECOGNIZED.reason, _UNRECOGNIZED.recover, False, WRITE_UNRECOGNIZED_NEXT_STEP
)


def write_deny_result(exc: BaseException, on_held: OnHeld) -> CallToolResult:
    """:func:`deny_result` for ``swg_write``: its own row where one differs,
    else the shared row, with a next step added to the generic
    ``internal_error``. ``on_held`` is the server's mode, ``"refuse"`` or
    ``"displace"``; only the row for a coordinator that is gone reads it."""
    terminal = _write_terminal(exc, on_held)
    if terminal is not None:
        return _result(terminal, str(exc))
    result = deny_result(exc)
    shared = result.structuredContent
    if isinstance(shared, dict) and shared.get("reason") == _UNRECOGNIZED.reason:
        return _result(_WRITE_UNRECOGNIZED, str(exc))
    return result


def _write_terminal(exc: BaseException, on_held: OnHeld) -> _Terminal | None:
    """swg_write's own row for ``exc``, or ``None`` where the shared row
    answers. Exact type, like ``_TERMINALS``; between two rows of one type the
    attribute that only write() sets decides, never the message."""
    if type(exc) is CoordinatorFailed:
        return _WRITE_OWN_GRANT if exc.needs_reacquire is True else _WRITE_COORDINATOR_FAILED
    if type(exc) is CoordinatorUnreachable:
        if exc.answered is True:
            return _WRITE_ANSWERED_UNUSABLE
        return _WRITE_GONE_REFUSE if on_held == "refuse" else _WRITE_GONE
    return _WRITE_TERMINALS.get(type(exc))


# --- swg_release's own rows ---------------------------------------------------
#
# A release the coordinator did not confirm: no usable answer, its failure
# envelope, an HTTP 5xx or 4xx, a watchdog-degraded answer, an answer that
# reports the grant still held, or a redirect this client refuses to follow.
# The volume keeps the path recorded in each case, so a repeat sends the
# release again, and a release of a grant already given back changes nothing:
# the flag is true and the verb is the repeat. A coordinator that keeps failing
# answers every repeat the same, so the words bound the repeats and say what
# to report, never a restart: the coordinator answered, or may answer the next.
RELEASE_UNCONFIRMED_REASON = "release_unconfirmed"
RELEASE_AGAIN_RECOVER = "release_again"
RELEASE_UNCONFIRMED_NEXT_STEP = (
    "The coordinator did not confirm this release, so this session's write grant on the "
    "path may still stand; this session's other holds are as they were. Calling "
    "swg_release on the path again is safe: it sends the release again, and a release of a "
    "grant already given back changes nothing. Try it at most twice more. If it still is "
    "not confirmed, stop and report to your user or host that this session's write grant "
    "on the path may still stand, which can keep other agents that write the path waiting. "
    "swg_reacquire on the path is the step that drops every hold this session has, this "
    "one included."
)
_RELEASE_UNCONFIRMED = _Terminal(
    RELEASE_UNCONFIRMED_REASON, RELEASE_AGAIN_RECOVER, True, RELEASE_UNCONFIRMED_NEXT_STEP
)
# By exact type, like ``_TERMINALS``: the plain ``CoherenceError`` is a 4xx, a
# degraded answer or a grant reported still held, and a subclass with a meaning
# of its own must not inherit the repeat.
_RELEASE_UNCONFIRMED_TYPES: frozenset[type] = frozenset(
    {CoordinatorUnreachable, CoordinatorFailed, CoherenceError, RedirectRefused}
)


def release_deny_result(exc: BaseException) -> CallToolResult:
    """``swg_release``'s answer to a release that raised: the
    ``release_unconfirmed`` row for each exact type a release the coordinator
    did not confirm raises, the shared rows for a caller-principal refusal,
    and ``internal_error`` for any other type."""
    if type(exc) in _RELEASE_UNCONFIRMED_TYPES:
        return _result(_RELEASE_UNCONFIRMED, str(exc))
    if type(exc) is CallerPrincipalRefused:
        return deny_result(exc)
    return _result(_UNRECOGNIZED, str(exc))


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


def _other_holder_fields(exc: OtherHolder) -> dict:
    """The holder as the wire string a client raises with, ``None`` otherwise: an
    in-process refusal carries a ``UUID``, which ``json.dumps`` refuses."""
    holder = exc.holder_agent_id
    return {"holder_agent_id": holder if isinstance(holder, str) else None}


def coordinator_unavailable_result(detail: str) -> CallToolResult:
    """Type B (synthesized): no adapter raised, because the volume is
    unattached. Fail closed — the write is NOT version-committed. A coordinator
    that fails an attached strict volume arrives typed instead, as
    ``CoordinatorUnreachable`` through its own row."""
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
