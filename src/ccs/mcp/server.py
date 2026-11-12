# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The ``stale-write-guard-fs`` MCP server: a stdio FastMCP binding over CoherentVolume.

**stdio invariant:** NEVER write to stdout — the MCP JSON-RPC stream owns fd 1.
All logging goes to stderr; the coordinator subprocess's stdout is redirected by
``connect_or_spawn``. **Serialization:** access to the single shared volume is
guarded by an ``asyncio.Lock`` and runs on the event-loop thread (no thread
offload), so FastMCP's coroutine dispatch cannot interleave two volume ops — the
volume's A5 thread-guard only sees *different threads*, not coroutine interleave,
so we serialize in code rather than rely on it.

Tool logic lives in sync ``_do_*`` helpers (real ``volume`` + ``config`` in,
``CallToolResult`` out); the async tool wrappers are thin (lock + delegate), so
the contract is testable against a real coordinator without a FastMCP client.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from ccs.adapters.claude_code.lifecycle import stop_coordinator
from ccs.adapters.claude_code.state_dir import check_coherence_dir
from ccs.adapters.coherent_volume import (
    CoherentVolume,
    HandoffTransferResult,
    HandoffVerbResult,
)
from ccs.adapters.effect_gate import check_fence
from ccs.cli._coherence_client import err, escape_nonprintable
from ccs.coordinator.fold_merge import index_key_of
from ccs.core.exceptions import (
    HOLD_INPUT_VANISHED,
    CasVersionConflict,
    CoherenceDirRefused,
    CoherenceError,
    HandoffPathsInvalid,
    RefuseModeUnsupported,
)
from ccs.mcp.deny import (
    HANDOFF_LANDED_TRANSFER,
    cas_exhausted_result,
    coordinator_unavailable_result,
    deny_result,
    handoff_deny_result,
    handoff_transfer_refusal_result,
    handoff_verb_refusal_result,
    release_deny_result,
    write_deny_result,
)
from ccs.mcp.session import SessionConfig, build_volume
from ccs.mcp.status import (
    build_status,
    fold_profile_of,
    handoff_from_status,
    status_text,
)
from ccs.mcp.uri import UriValidationError, validate_uri

logger = logging.getLogger(__name__)

SERVER_NAME = "stale-write-guard-fs"

# The honesty ceiling (origin §5 / R5): the server-level instructions, the
# per-tool descriptions, and the deny ``structuredContent`` together bound what
# may be claimed. Annotations are untrusted hints; this prose is not.
INSTRUCTIONS = """\
stale-write-guard-fs guards a SINGLE-HOST workspace against silent lost updates
when two or more agents share a mutable file. It enforces VERSION LINEAGE via a
local coherence coordinator; it does NOT merge content for you.

Three guarantees, all single-host and fail-closed:
  1. Sequential stale-overwrite — swg_write is DENIED (reason=stale_view) if the
     file changed since you read it: either a peer committed a newer version OR it
     was edited out-of-band (an editor, another tool, a shell script). Recover with
     swg_reacquire, then write FROM the fresh bytes it returns.
  2. Concurrent same-key lost-update — swg_write_cas(path, expected_version,
     new_content) rejects a stale compare-and-set as a TYPED CONFLICT (not an
     auto-merge): you read, merge, and retry; the server never merges for you.
  3. Effect fired on a superseded read — before ANY irreversible external
     action you decided from a swg_read (webhook, deploy, opened PR, posted
     message), call swg_gate(path, expected_version, expected_generation) with
     the pair that read returned. It DENIES (reason=stale_view) if the value
     moved OR the grant you read under was reclaimed while you were thinking —
     a reclaim leaves the version untouched, so the version alone cannot see
     it — OR a peer's write-claim preempted your grant, which leaves BOTH
     comparands untouched: the gate also re-checks that your grant still
     stands. On a deny, do not take the action: swg_reacquire, re-read,
     re-decide.
     The verdict is true as of that call; the dispatch after it is still yours.

WRITE HOLDS, in both modes: swg_write takes a write grant on the path it writes
and keeps it after the write. swg_release(path) gives back that one path and
keeps your other holds; swg_reacquire and swg_write_cas, of any path, drop every
hold this session has, and so does this server's normal exit. When you finish
writing a path, swg_release it: until you do, other opted-in agents that write
it wait, for up to 30 minutes. swg_status lists your holds in held (null means
it cannot tell, never that you hold nothing).

REFUSE MODE is the operator's setting, SWG_ON_HELD=refuse; swg_status reports
the mode as on_held. In it, a swg_write of a managed path that another agent
holds is refused with reason=other_holder, recover=wait_then_reacquire,
retryable=false: nothing was written, nothing changed hands, and its next_step
gives the wait. Any writer's grant counts as a hold, but only opted-in writers
honour a hold, such as a refuse-mode server writing one of its managed paths:
the Claude Code plugin's hooks, displace-mode servers (the default) and older
volumes take the file anyway, and the holder then sees stale_view on its next
swg_write. A hold is not a lock.

WHO HOLDS A PATH: holder ids are per attempt and never equal your
session_agent_id, so tell your own holds by held_by in swg_status
(this_attempt, earlier_attempt or another_agent), never by comparing ids.
Every caller of one server process is one agent to the coordinator, so
this_attempt can be another caller sharing this server.

HANDOFF between sessions: swg_transfer(paths, successor) hands this session's
claim on each path (the write grant from swg_write, or the standing read from
swg_read) to another session, the successor, named by the session_agent_id the
successor's own swg_status reports.
A transfer FENCES THE GIVER and DOES NOT RESERVE THE PATH: other sessions keep
reading and writing it by the ordinary rules, and the handoff only labels what
they do. While the handoff is live, this session's swg_write and swg_write_cas
on the path are DENIED with reason=handed_off, retryable=false,
recover=stop_and_report: stop and report to your user or host that the path
was handed off. The fence covers this MCP session only: a write you make to
the path through any other route (your own file tools, a shell) is another
session's write and is not refused, so do not write a path you handed off by
any route. The successor sees the handoff in the handoff key of its
swg_read and swg_status, and takes it with swg_accept or a write, or refuses it
with swg_decline. swg_withdraw is taken only on the user's or host's explicit
instruction, never as the recovery for a handed_off deny.

REFUSALS are error results whose structuredContent carries reason, recover,
retryable and detail, and often next_step: when next_step is present it is the
procedure, so follow it. retryable=true means only that sending the identical
call again is safe; it is not the next step. retryable=false means do not send
it again unless next_step says to. The recover verbs:
  reacquire: swg_reacquire the path, then write FROM the bytes it returns.
  wait_then_reacquire: another agent holds the path; release your holds, poll
    swg_status, then swg_reacquire and swg_write once (next_step has the steps).
  write_again: the bytes swg_write put on disk may be unrecorded; swg_write the
    path once more, which records them.
  release_again: the release was not confirmed; swg_release the path again,
    at most twice more.
  reacquire_and_reconcile: the write lost its grant mid-write and its bytes may
    be on disk unrecorded; swg_reacquire, reconcile your change with the bytes
    it returns, and write that.
  read_then_merge: swg_read the path, merge into what it returns, then
    swg_write_cas at its version.
  reacquire_and_reread: swg_reacquire the path, then swg_read it for a fresh
    version.
  read_then_retry: the commit may have landed; swg_read the path and retry only
    if your change is not there.
  wait_and_retry: wait a few seconds, then send the same call again.
  wait_or_escalate: wait a while before trying again; if it persists, report it.
  retry_later: this server has no coordinator attached; try later, and report
    it if it persists.
  check_handoff: look at the path's handoff in swg_status before you act.
  fix_successor: name the successor by the session_agent_id its own swg_status
    reports.
  fix_path, check_path, use_text, reread: fix the path; check that the file
    exists; the file is not UTF-8 text; swg_read again and pass both comparands
    it returns.
  restart_session: this server session lost coordination; report it, since
    only a new server session regains it.
  stop, stop_and_report, none: stop, do not write the path by any other route,
    and report the detail to your user or host.

OUT OF GUARANTEE (do not rely on this server for): writers on DIFFERENT hosts or
across a synced/network mount; divergent-history reconciliation; semantic/content
correctness; any server-enforced auto-merge. These are NOT detected in v1 — a
heterogeneous multi-host setup looks identical to a guarded one (swg_status
reports heterogeneous_scope_detectable=false).

TRUST BOUNDARY: the server enforces that your write descends from a version you
read; it CANNOT verify you derived your content from the bytes you read. Same-uid
local processes can reach the coordinator directly, bypassing this server — the
model is single-uid, single-host.

v1 guards UTF-8 TEXT artifacts (config, notes, memory, code); a non-text file is
reported, not silently mangled (binary support → v1.1).
"""

# Appended to every tool description so the honesty floor travels with each tool
# (a client that surfaces only descriptions still sees the scope).
_SCOPE_CLAUSE = (
    " SINGLE-HOST only. Out of guarantee and NOT detected in v1: writers on "
    "different hosts or across a synced/network mount, divergent-history "
    "reconciliation, semantic correctness, server-enforced auto-merge."
)

_READ_DESC = (
    "Read a workspace text file under coherence tracking. Returns {content, "
    "version, owner_generation}. The version is the comparand you pass to "
    "swg_write_cas; KEEP BOTH version and owner_generation and pass them to "
    "swg_gate before any irreversible external action you decide from this "
    "read (owner_generation=null means this coordinator does not report "
    "generations, so swg_gate will hold). When the path has a handoff record "
    "the result also carries handoff: the record as it concerns this session "
    "(role giver, successor or bystander, the two session_agent_ids, "
    "version_at_transfer, hold_shape, status, live); handoff_unknown=true "
    "instead means the read was denied and the record could not be fetched. "
    "The coordinator attaches the record best-effort, so a result with "
    "neither does not prove there is no record; swg_status lists every "
    "record. A "
    "sticky-INVALID view returns fresh bytes but stays INVALID — use "
    "swg_reacquire to recover before writing. If the bytes on disk are not the "
    "content at the current version, the read is DENIED with reason=stale_view "
    "and no version. A peer's commit still reaching disk clears on its own: "
    "swg_reacquire, then swg_read again, retrying for a few seconds. If it "
    "stays denied past that, the bytes on disk were never recorded (an "
    "out-of-band edit, a commit whose disk write failed, or a write whose commit "
    "did not reach the coordinator); reading again does not clear it: swg_write "
    "the content swg_reacquire returned (or your merge of it), then swg_read. A "
    "write made sooner can be overwritten by a peer commit still reaching "
    "disk." + _SCOPE_CLAUSE
)
_GATE_DESC = (
    "Verify a file is STILL unchanged and still under the same grant. Pass BOTH "
    "comparands from your earlier swg_read — expected_version AND "
    "expected_generation (its owner_generation). Call this "
    "immediately BEFORE any irreversible external action you decided from that "
    "read (sending a webhook, opening a PR, running a deploy, posting a "
    "message). Returns decision=proceed, or DENIES with reason=stale_view if the "
    "file moved OR the grant you read it under was reclaimed (the version alone "
    "cannot see a reclaim) OR a peer write-claim preempted your grant (which "
    "moves neither comparand — the gate re-checks that the grant still stands) "
    "OR either comparand is unconfirmed. On a deny: do NOT "
    "take the action — swg_reacquire, re-read, re-decide." + _SCOPE_CLAUSE
)
_WRITE_DESC = (
    "Write a workspace text file (acquire -> write -> commit). In either mode the "
    "write takes a write grant on the path and keeps it after the write. When you "
    "finish writing a path, swg_release it: until you do, other opted-in agents that "
    "write it wait, for up to 30 minutes. On a refuse-mode server (SWG_ON_HELD=refuse) "
    "a success also carries opted_in: true when the write asked first, false when the "
    "path is not among the managed files, so the write took the file from any agent "
    "that held it. A refusal carries reason, recover and retryable; follow its "
    "next_step when it has one. DENIED with reason=stale_view if the file changed "
    "since you read it — a peer commit OR an out-of-band edit (another tool/editor): "
    "recover with swg_reacquire, then write FROM its bytes, never the content you "
    "just sent. The other reasons: other_holder (refuse mode only), another agent "
    "holds the path and nothing was written (recover=wait_then_reacquire); "
    "commit_preempted, a mid-write preempt (disk may hold un-versioned bytes; "
    "reacquire_and_reconcile); commit_unconfirmed, the bytes are on disk but may be "
    "unrecorded (recover=write_again); link_out_of_managed (refuse mode only), the "
    "path goes through a link to a file outside the managed files; handed_off, this "
    "session handed the path off; coordinator_failed, the coordinator failed while "
    "answering (recover=reacquire when an earlier attempt of this session still "
    "holds the path); coordinator_unavailable, it gave no usable answer or is not "
    "attached; caller_principal_absent, caller_principal_foreign or "
    "caller_principal_claimed, this session's caller principal was refused "
    "(swg_status shows principal_claim); internal_error, a failure with no "
    "recovery; invalid_path or io_error, a bad path or a disk failure." + _SCOPE_CLAUSE
)
_REACQUIRE_DESC = (
    "Recover from a stale_view deny: re-mint identity and return the CURRENT "
    "bytes. You MUST write FROM these exact bytes — the server enforces version "
    "lineage, NOT that your content was derived from what you read. It is also the "
    "step a swg_write refusal names with recover=reacquire or wait_then_reacquire. "
    "The re-mint drops every hold this session has: the write grant on every path, "
    "not only this one, so other agents can then write those paths. To give back "
    "one path and keep the others, use swg_release." + _SCOPE_CLAUSE
)
_RELEASE_DESC = (
    "Give back this session's write grant on one path and keep every other hold. "
    "swg_write takes a write grant on the path it writes and keeps it after the "
    "write. When you finish writing a path, swg_release it: until you do, other "
    "opted-in agents that write it wait, for up to 30 minutes. Answers {ok, path, "
    "released}. released=true means this session gave a grant back; it does not "
    "mean the path is free (another agent may hold it), and this session keeps a "
    "read at the version it held, so its next swg_write of the path needs no "
    "swg_reacquire unless the file changed since. released=false means this "
    "session had no write grant recorded on the path and nothing was sent, as on "
    "a repeat. A release the coordinator did not confirm answers "
    "reason=release_unconfirmed, recover=release_again, retryable=true: calling "
    "swg_release again is safe, at most twice more; if it still is not confirmed, "
    "report that this session's write grant on the path may still stand "
    "(swg_reacquire drops every hold, this one included)." + _SCOPE_CLAUSE
)
_STATUS_DESC = (
    "Report coherence state: coordinator on|off|unknown (unknown is NOT off), "
    "per-path enforced|not_registered (with handoff: the path's handoff record "
    "-- giver, successor, version_at_transfer, hold_shape, status, live -- when "
    "it has one; per_path is null, not {}, when the coordinator's /status "
    "answer carries no list of tracked paths, as a degraded answer does when "
    "its registry was busy: retry shortly and do not treat it as nothing "
    "tracked; the text result then says per_path=unavailable. With "
    "coordinator=unknown, per_path is {} and says nothing about what is "
    "tracked), "
    "the write holder on each per_path entry someone holds for writing: "
    "holder_agent_id (the holder's attempt id, in the 32-hex spelling an "
    "other_holder refusal names; ids are per attempt, so it never equals "
    "session_agent_id) and held_by (this_attempt; earlier_attempt, an earlier "
    "attempt of this session whose grant still stands, which you release; or "
    "another_agent), neither key on an entry nobody holds, and this_attempt "
    "covering another caller sharing this server, since every caller of one "
    "server process is one agent to the coordinator; "
    "on_held (this server's mode: refuse when the operator set SWG_ON_HELD=refuse, "
    "else displace), held (the sorted paths this session holds for writing, "
    "either attempt; null when the coordinator cannot say who holds each path: "
    "null means cannot tell, never that nothing is held; the text result gives "
    "on_held, held, and a line per path held_by another_agent or earlier_attempt), "
    "is_attached/is_degraded/session_id, session_agent_id (this "
    "session's id as a handoff successor: the value another session passes to "
    "swg_transfer to hand this session a path; it does not change when this "
    "session reacquires, and it names this session as a successor only while "
    "principal_claim is bound), "
    "principal_claim (this session's caller-principal state: bound; unsupported "
    "= the coordinator issues none; unconfirmed = the last claim's answer was "
    "lost and the next call claims again by itself; refused = the session is "
    "bound under another nonce, so every later swg_read/swg_write/swg_gate "
    "answers the typed caller_principal_* deny with recover=restart_session "
    "while the coordinator stays on; not_attempted = nothing claimed yet), "
    "caller_principal_absent_total and caller_principal_refused_total (the "
    "coordinator's counters; null, never 0, when it is unreachable or does not "
    "report them), registry_schema_version (the schema version of the "
    "coordinator's store, an integer; an older release refuses a store at a "
    "newer version; null when the coordinator is unreachable "
    "(coordinator=unknown) or its answer carries no such field, as a "
    "coordinator built before the field does; report it to the user rather "
    "than act on it), and heterogeneous_scope_detectable=false (a multi-host or "
    "differently-scoped setup is NOT distinguishable in v1)." + _SCOPE_CLAUSE
)

_WRITE_CAS_DESC = (
    "Concurrent same-key write via compare-and-set. You read (swg_read returns the "
    "version comparand; swg_reacquire does NOT — it is for swg_write recovery), "
    "MERGE, then call swg_write_cas(path, expected_version, "
    "new_content). Stale-write-rejected: if a peer committed since your read, the "
    "CAS is a TYPED CONFLICT (reason=version_mismatch, current_version returned) "
    "— NOT an auto-merge; re-read at current_version, re-merge, and retry. A "
    "win on a path with a live handoff carries handoff: outcome=completed when "
    "this session is its successor, overtaken (with counterparty) otherwise. The "
    "per-session conflict counter bounds only a COOPERATING agent (one session, "
    "stops on retryable=false); it is NOT livelock-proof against a fresh session "
    "or one that ignores retryable=false. A call that reaches the coordinator drops "
    "every hold this session has: it starts a new attempt, which gives back the write grant swg_write took "
    "on every path, as swg_reacquire does." + _SCOPE_CLAUSE
)

# Appended to each of the four handoff tools' descriptions: what a transfer
# does to the giver and what it does not do to anyone else.
_HANDOFF_CLAUSE = (
    " A transfer fences the giver and does not reserve the path: other sessions "
    "keep reading and writing it by the ordinary rules, and the handoff only "
    "labels what they do."
)

_TRANSFER_DESC = (
    "Hand this session's claim on one or more paths -- the write grant from "
    "swg_write, or the standing read from swg_read -- to another session, the "
    "successor. Name the successor by the session_agent_id that the "
    "successor's OWN swg_status reports. Answers one grant per path, in order: "
    "transferred=true with the record (giver, successor, version_at_transfer, "
    "hold_shape, status=pending), or transferred=false with a typed reason and "
    "nothing changed (handoff_not_held: this session holds no claim on the "
    "path; handoff_successor_unknown: the coordinator does not know that id; "
    "handoff_in_flight: another session's handoff of the path is live). The "
    "result is an error unless every grant transferred, and each refused grant "
    "carries recover and retryable=false and a fixed next_step: fix_successor "
    "for a successor id that names no other session, check_handoff for "
    "handoff_not_held (a transfer that already landed answers it too), "
    "stop_and_report otherwise; the top level speaks for the most restrictive "
    "refused grant. While a handoff is "
    "live this session's swg_write and swg_write_cas on its path are denied "
    "with reason=handed_off. The fence covers this MCP session only: a write "
    "you make through any other route (your own file tools, a shell) is "
    "another session's and is not refused, so write a path you handed off by "
    "no route. If the answer is commit_unconfirmed "
    "(recover=check_handoff) the transfer may have landed: look at the path's "
    "handoff (swg_read or swg_status) first. " + HANDOFF_LANDED_TRANSFER
    + "Transfer again only if no such handoff shows; if the path "
    "was written since or shows another session's handoff, ask your user. Once the successor has written the path, a "
    "second transfer is a new handoff."
    + _HANDOFF_CLAUSE + _SCOPE_CLAUSE
)
_ACCEPT_DESC = (
    "As the successor, accept the live handoff of a path without writing it: "
    "a pending handoff becomes completed (a write of the path completes it "
    "too). The giver stays fenced until a write moves the version. If another "
    "session has acquired the path and nothing has written it since, the "
    "handoff is overtaken but live: the accept is taken but changes nothing, "
    "answers status=overtaken and names that session as counterparty. Once a "
    "write has moved the version the handoff has ended, and the accept is "
    "refused handoff_not_live with the status saying how. Refused "
    "with handoff_not_successor or handoff_not_live, as an error that changes "
    "nothing (recover=stop_and_report, retryable=false, and a fixed next_step)."
    + _HANDOFF_CLAUSE + _SCOPE_CLAUSE
)
_DECLINE_DESC = (
    "As the successor, decline the live handoff of a path: the handoff ends "
    "and its giver's fence lifts. Refused with handoff_not_successor or "
    "handoff_not_live, as an error that changes nothing (recover=stop_and_report, "
    "retryable=false, and a fixed next_step)."
    + _HANDOFF_CLAUSE + _SCOPE_CLAUSE
)
_WITHDRAW_DESC = (
    "As the giver, withdraw this session's live handoff of a path: the handoff "
    "ends and this session's fence on the path lifts. Take it only on the "
    "user's or host's explicit instruction. It is never the recovery for the "
    "handed_off refusal: when a write of a path you handed off is refused, stop "
    "and report to your user or host. Refused with handoff_not_giver or "
    "handoff_not_live, as an error that changes nothing, whose next_step says "
    "to stop and report." + _HANDOFF_CLAUSE + _SCOPE_CLAUSE
)

_REACQUIRE_NOTE = "write FROM these exact bytes — the server enforces version lineage, not content derivation"

# The per-session bound on consecutive CAS conflicts for one path (the
# cooperating-agent livelock guard). Mirrors the adapter's MAX_CAS_REACQUIRES=8.
MAX_CAS_CONFLICTS = 8


@dataclass
class ServerContext:
    """Lifespan-owned session state shared by every tool.

    ``lock`` serializes access to ``volume`` — every tool acquires it for the
    duration of its volume interaction (see module docstring).
    """

    volume: CoherentVolume
    config: SessionConfig
    lock: asyncio.Lock
    # file's fold key → consecutive CAS-conflict count, the cooperating-agent
    # livelock bound for swg_write_cas (reset on a win; survives only within one
    # stdio session).
    cas_conflicts: dict[str, int] = field(default_factory=dict)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
    """Own the coordinator for the server's lifetime.

    Enter → construct the strict-only volume (self-spawns/attaches; raises
    fail-closed if it can't). Exit → close the volume, which gives back its
    write grants (a failed close is logged), then ``stop_coordinator``: an
    in-process coordinator must still answer the close. The coordinator is
    stopped whatever the close does. Only a normal end runs this; nothing
    handles a signal. Does NOT call ``connect_or_spawn`` — construction
    already did, so calling it again would double-spawn.
    """
    config = SessionConfig.from_env()
    volume = build_volume(config)  # blocking, fail-closed
    logger.info("stale-write-guard-fs attached: root=%s managed=%s", config.root, config.managed)
    ctx = ServerContext(volume=volume, config=config, lock=asyncio.Lock())
    try:
        yield ctx
    finally:
        try:
            async with ctx.lock:
                try:
                    volume.close()
                except CoherenceError as exc:
                    logger.warning("stale-write-guard-fs could not close its volume: %s", exc)
        finally:
            stop_coordinator(config.root)
            logger.info("stale-write-guard-fs coordinator stopped: root=%s", config.root)


# --- result builders ---------------------------------------------------------


def _ok_result(structured: dict, text: str) -> CallToolResult:
    return CallToolResult(
        isError=False,
        content=[TextContent(type="text", text=text)],
        structuredContent=structured,
    )


def _client_error_result(reason: str, recover: str, detail: str) -> CallToolResult:
    """A non-deny client/input error (invalid path, missing file, binary). Still
    a non-ignorable isError, but NOT a coherence deny (never ``stale_view``)."""
    return CallToolResult(
        isError=True,
        content=[TextContent(type="text", text=detail)],
        structuredContent={
            "reason": reason,
            "recover": recover,
            "retryable": False,
            "detail": detail,
        },
    )


def _decode_text(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --- tool logic (sync, testable: real volume + config in, CallToolResult out) -


def _do_read(volume: CoherentVolume, config: SessionConfig, path: str) -> CallToolResult:
    try:
        key = validate_uri(path, root=config.root)
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    if not volume.is_attached:
        return coordinator_unavailable_result(f"coordinator unattached; cannot read {path}")
    try:
        data, version, owner_generation = volume.read_with_version_generation(key)
    except FileNotFoundError as exc:
        return _client_error_result("file_not_found", "check_path", str(exc))
    except CoherenceError as exc:
        return deny_result(exc)
    except OSError as exc:  # disk/permission failure → fail closed, never escape to FastMCP
        return _client_error_result("io_error", "none", str(exc))
    text = _decode_text(data)
    if text is None:
        return _client_error_result("binary_unsupported", "use_text", f"{key} is not UTF-8 text (v1 guards text only)")
    # owner_generation rides alongside the version as the SECOND comparand:
    # the version answers "is this value still current", the generation answers
    # "is the grant it was read under still standing". Both are what swg_gate
    # re-checks before an irreversible external action. ``null`` means the
    # coordinator could not confirm one (older daemon, deny, or degraded) — a
    # later swg_gate HOLDs on it rather than firing blind.
    structured = {
        "content": text,
        "version": version,
        "owner_generation": owner_generation,
        "encoding": "utf-8",
    }
    # The read's provenance (#185): the path's transfer record projected for
    # this session, only when the path has one, so a read of a path with no
    # record answers exactly what it did before. A strict deny never carries
    # the key -- its bytes stay the corpus's -- and the giver's own re-read of
    # a path it handed off is one, so after a deny the record comes from
    # /status; an omitted key must never read as "no record" when the
    # coordinator was not asked. An admitted read's key is best-effort on the
    # coordinator too, so its absence there is not proof of no record either;
    # the tool text says so.
    handoff = volume.read_handoff(key)
    if handoff is None and volume.last_read_denied:
        known, handoff = handoff_from_status(volume, key)
        if not known:
            structured["handoff_unknown"] = True
    if handoff is not None:
        structured["handoff"] = handoff
    return _ok_result(structured, text)


def _do_gate(
    volume: CoherentVolume,
    config: SessionConfig,
    path: str,
    expected_version: int,
    expected_generation: int | None,
) -> CallToolResult:
    """Re-validate a comparand from an earlier ``swg_read`` immediately before
    the agent takes an irreversible external action.

    This is the pull-based fence for the tool surface. ``gate()`` cannot be
    exposed as a tool — its ``decide``/``effect`` are Python callables, and an
    MCP agent's decision and effect live outside this process, between tool
    calls. So the agent holds the ``(version, owner_generation)`` pair from its
    read and asks HERE, one call before dispatching. Same guarantee, same
    honest boundary: the verdict is true as of this check, and the dispatch
    that follows is still the agent's own step."""
    try:
        key = validate_uri(path, root=config.root)
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    if not volume.is_attached:
        return coordinator_unavailable_result(
            f"coordinator unattached; cannot verify {path} before an effect"
        )
    if expected_generation is None:
        # Distinct from a coherence deny: the agent supplied no authority
        # comparand, so there is nothing to verify. Returning the retryable
        # stale_view deny here would send a cooperating agent into an
        # unbounded reacquire loop that can never clear, since re-reading
        # cannot supply a comparand it failed to carry. Name the real problem.
        return _client_error_result(
            "missing_comparand",
            "reread",
            "swg_gate needs the owner_generation from your swg_read alongside "
            "expected_version; call swg_read and pass BOTH comparands back. "
            "If that read returned owner_generation=null, this coordinator "
            "does not report generations — restart it on a current version "
            "(re-reading will not help).",
        )
    try:
        check_fence(
            volume,
            key,
            expected_version=expected_version,
            expected_generation=expected_generation,
        )
    except CoherenceError as exc:
        cause = getattr(exc, "hold_cause", None)
        if cause == HOLD_INPUT_VANISHED:
            # The input is GONE. check_fence raises this as a StaleView (a
            # HOLD), but routing it through the deny mapping would advertise
            # recover="reacquire", retryable=true — telling the agent to
            # reacquire a file that does not exist, which swg_reacquire then
            # refuses. Answer with the same shape swg_read gives for a missing
            # file, so the advice matches reality.
            return _client_error_result("file_not_found", "check_path", str(exc))
        # Every other HOLD is a StaleView, so it maps through the SAME deny path
        # (and the same reason vocabulary) as every other refusal on this
        # surface — the agent already knows how to read it. Never a soft
        # "false", and retry-then-escalate is the correct advice for all of
        # them (see the hold_cause table in the guide).
        result = deny_result(exc)
        if cause is not None and isinstance(result.structuredContent, dict):
            # WHY it held, typed: the agent branches on a value, not on prose.
            result.structuredContent["hold_cause"] = cause
        return result
    except OSError as exc:
        return _client_error_result("io_error", "none", str(exc))
    return _ok_result(
        {
            "decision": "proceed",
            "path": key,
            "version": expected_version,
            "owner_generation": expected_generation,
        },
        f"{key} is unchanged at v{expected_version} under the same grant; proceed",
    )


def _do_write(volume: CoherentVolume, config: SessionConfig, path: str, content: str) -> CallToolResult:
    try:
        key = validate_uri(path, root=config.root)
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    # is_attached re-check BEFORE the write — a mid-session endpoint loss must
    # fail closed here, never reach the adapter's best-effort unversioned write.
    if not volume.is_attached:
        return coordinator_unavailable_result(f"coordinator unattached; refusing write of {key}")
    try:
        # Judged before the write, by the volume's own rule for the path, so a
        # refuse-mode answer says whether this write asked only if nobody held
        # the file. A displace-mode server asks nothing and answers as before.
        opted_in = volume.asks_if_unheld(key) if config.on_held == "refuse" else None
        volume.write(key, content.encode("utf-8"))
    except CoherenceError as exc:
        return write_deny_result(exc, config.on_held)
    except OSError as exc:  # disk/permission failure → fail closed, never escape to FastMCP
        return _client_error_result("io_error", "none", str(exc))
    if opted_in is None:
        return _ok_result({"ok": True, "path": key}, f"wrote {key}")
    text = f"wrote {key}" if opted_in else (
        f"wrote {key} without asking first: it is not among the managed files, so this "
        "write takes the file from any agent that holds it"
    )
    return _ok_result({"ok": True, "path": key, "opted_in": opted_in}, text)


def _do_reacquire(volume: CoherentVolume, config: SessionConfig, path: str) -> CallToolResult:
    try:
        key = validate_uri(path, root=config.root)
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    if not volume.is_attached:
        return coordinator_unavailable_result(f"coordinator unattached; cannot reacquire {key}")
    try:
        data = volume.reacquire(key)
    except FileNotFoundError as exc:
        return _client_error_result("file_not_found", "check_path", str(exc))
    except CoherenceError as exc:
        return deny_result(exc)
    except OSError as exc:  # disk/permission failure → fail closed, never escape to FastMCP
        return _client_error_result("io_error", "none", str(exc))
    text = _decode_text(data)
    if text is None:
        return _client_error_result("binary_unsupported", "use_text", f"{key} is not UTF-8 text (v1 guards text only)")
    return _ok_result({"content": text, "encoding": "utf-8", "note": _REACQUIRE_NOTE}, text)


def _do_release(volume: CoherentVolume, config: SessionConfig, path: str) -> CallToolResult:
    try:
        key = validate_uri(path, root=config.root)
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    # A volume with no endpoint sends nothing and still answers True for a
    # grant it recorded, which would report a release no coordinator saw.
    if not volume.is_attached:
        return coordinator_unavailable_result(f"coordinator unattached; cannot release {key}")
    try:
        released = volume.release(key)
    except CoherenceError as exc:
        return release_deny_result(exc)
    return _ok_result({"ok": True, "path": key, "released": released}, _release_text(key, released))


def _release_text(key: str, released: bool) -> str:
    """What a client that shows only the text channel learns of a release:
    whether a grant went back, and never that the path is free."""
    if not released:
        return f"released nothing on {key}: this session has no write grant recorded there, so nothing was sent"
    return (
        f"released this session's write grant on {key}. The path is not necessarily free: "
        "another agent may hold it. This session keeps a read of it at the version it held."
    )


def _do_status(volume: CoherentVolume, config: SessionConfig) -> CallToolResult:
    status = build_status(volume, config)
    # A client that shows only the text channel must still see what an agent
    # acts on: the mode, its own holds, and who else holds a path.
    return _ok_result(status, status_text(status))


def _do_write_cas(
    volume: CoherentVolume,
    config: SessionConfig,
    conflicts: dict[str, int],
    path: str,
    expected_version: int,
    new_content: str,
) -> CallToolResult:
    try:
        key = validate_uri(path, root=config.root)
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    if not volume.is_attached:
        return coordinator_unavailable_result(f"coordinator unattached; refusing CAS of {key}")
    try:
        won = volume.write_cas_at(key, expected_version, new_content.encode("utf-8"))
    except CasVersionConflict as exc:
        # Bound a COOPERATING agent's retry loop: too many consecutive conflicts
        # on one file in this session, under any of its spellings → tell it to
        # stop (retryable=false).
        streak = _conflict_streak_key(volume, key)
        count = conflicts.get(streak, 0) + 1
        if count > MAX_CAS_CONFLICTS:
            conflicts.pop(streak, None)
            return cas_exhausted_result(
                f"{key}: {count} consecutive CAS conflicts in this session — stop "
                "(the cooperating-agent livelock bound; coordinator-side fencing is v1.1)"
            )
        conflicts[streak] = count
        return deny_result(exc)
    except FileNotFoundError as exc:
        return _client_error_result("file_not_found", "check_path", str(exc))
    except CoherenceError as exc:
        return deny_result(exc)
    except OSError as exc:  # disk/permission failure → fail closed, never escape to FastMCP
        return _client_error_result("io_error", "none", str(exc))
    if conflicts:  # a win resets the file's conflict streak; with none open, no /status read
        conflicts.pop(_conflict_streak_key(volume, key), None)
    structured: dict = {"ok": True, "path": key}
    if won.handoff is not None:
        # What the win did to a live handoff of the path (#185): completed it
        # (this session is the successor) or overtook it (a bystander).
        structured["handoff"] = _present(asdict(won.handoff))
    return _ok_result(structured, f"committed {key}")


def _conflict_streak_key(volume: CoherentVolume, key: str) -> str:
    """The name ``key``'s consecutive CAS conflicts are counted under: its
    fold key by the profile the coordinator's default ``/status`` tier
    advertises, read each time it is needed, so two spellings the
    coordinator keys as one file share one streak. A coordinator restarted
    on a disk it classifies differently is then followed, not the profile
    of an earlier start."""
    return index_key_of(key, fold_profile_of(volume.coordinator_status()))


def _present(fields: dict) -> dict:
    """``fields`` without the ones the answer did not carry, so a result has
    the shape of the coordinator's answer (a refused grant is ``{path,
    transferred, reason}``, to which the error result then adds its
    ``recover``, ``retryable`` and ``next_step``). ``fields`` is a volume
    result's ``asdict``: its
    keys are the result's fields in declaration order, and every value is a
    str, int, bool or ``None``, so ``False`` survives and only ``None`` goes."""
    return {key: value for key, value in fields.items() if value is not None}


def _transfer_result(result: HandoffTransferResult) -> CallToolResult:
    grants = [_present(asdict(grant)) for grant in result.grants]
    lines = [
        f"{grant.path}: transferred to {grant.successor} at v{grant.version_at_transfer} "
        f"({grant.status})"
        if grant.transferred
        else f"{grant.path}: not transferred ({grant.reason})"
        for grant in result.grants
    ]
    if result.ok:
        return _ok_result({"ok": True, "grants": grants}, "\n".join(lines))
    # Not every grant moved: a non-ignorable error, so a partly refused
    # transfer never reads as done, carrying a recover verb like every other.
    return handoff_transfer_refusal_result(grants, "\n".join(lines))


def _verb_result(verb: str, result: HandoffVerbResult) -> CallToolResult:
    structured = _present(asdict(result))
    if not result.ok:
        return handoff_verb_refusal_result(
            result.reason, f"{verb} {result.path}: refused ({result.reason}); nothing changed", structured
        )
    return _ok_result(structured, f"{verb} {result.path}: taken (status={result.status})")


def _do_transfer(
    volume: CoherentVolume, config: SessionConfig, paths: list[str], successor: str
) -> CallToolResult:
    try:
        keys = [validate_uri(path, root=config.root) for path in paths]
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    # The agent's own input mistake, answered like any bad path rather than
    # sent on for the coordinator's HTTP 400, which reads as an internal error.
    if not keys or len(set(keys)) != len(keys):
        return _client_error_result(
            "invalid_path", "fix_path", "paths must name at least one path, each once"
        )
    if not volume.is_attached:
        return coordinator_unavailable_result(f"coordinator unattached; cannot transfer {', '.join(keys)}")
    try:
        result = volume.transfer(keys, successor=successor)
    except HandoffPathsInvalid as exc:  # two spellings the volume resolves to one file
        return _client_error_result("invalid_path", "fix_path", str(exc))
    except CoherenceError as exc:
        return handoff_deny_result(exc, "transfer")
    return _transfer_result(result)


def _do_handoff_verb(
    volume: CoherentVolume,
    config: SessionConfig,
    verb: str,
    path: str,
) -> CallToolResult:
    """Accept, decline or withdraw ``path``'s handoff as this session. A refusal
    is the volume's typed value; only an answer that does not settle the
    outcome raises, and maps through the deny table like every terminal."""
    try:
        key = validate_uri(path, root=config.root)
    except UriValidationError as exc:
        return _client_error_result("invalid_path", "fix_path", str(exc))
    if not volume.is_attached:
        return coordinator_unavailable_result(f"coordinator unattached; cannot {verb} {key}")
    act = {"accept": volume.accept, "decline": volume.decline, "withdraw": volume.withdraw}[verb]
    try:
        result = act(key)
    except CoherenceError as exc:
        return handoff_deny_result(exc, verb)
    return _verb_result(verb, result)


def _do_accept(volume: CoherentVolume, config: SessionConfig, path: str) -> CallToolResult:
    return _do_handoff_verb(volume, config, "accept", path)


def _do_decline(volume: CoherentVolume, config: SessionConfig, path: str) -> CallToolResult:
    return _do_handoff_verb(volume, config, "decline", path)


def _do_withdraw(volume: CoherentVolume, config: SessionConfig, path: str) -> CallToolResult:
    return _do_handoff_verb(volume, config, "withdraw", path)


# --- registration ------------------------------------------------------------


def _server_context(ctx: Context) -> ServerContext:
    return ctx.request_context.lifespan_context


def register_tools(server: FastMCP) -> None:
    """Register the sequential ``swg_*`` tools under the serialization lock.

    ``swg_write_cas`` is the concurrent regime; ``swg_gate`` is the
    pull-based effect fence agents call before an irreversible dispatch;
    ``swg_release`` gives back this session's write grant on one path and
    keeps every other hold (#504);
    ``swg_transfer``/``swg_accept``/``swg_decline``/``swg_withdraw`` hand this
    session's own claims to another session (#185).
    """

    @server.tool(
        name="swg_read",
        description=_READ_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),  # pre-read mutates coordinator MESI state
        structured_output=False,
    )
    async def swg_read(path: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_read(sctx.volume, sctx.config, path)

    @server.tool(
        name="swg_write",
        description=_WRITE_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_write(path: str, content: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_write(sctx.volume, sctx.config, path, content)

    @server.tool(
        name="swg_reacquire",
        description=_REACQUIRE_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_reacquire(path: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_reacquire(sctx.volume, sctx.config, path)

    @server.tool(
        name="swg_release",
        description=_RELEASE_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_release(path: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_release(sctx.volume, sctx.config, path)

    @server.tool(
        name="swg_status",
        description=_STATUS_DESC,
        annotations=ToolAnnotations(readOnlyHint=True),
        structured_output=False,
    )
    async def swg_status(ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_status(sctx.volume, sctx.config)

    @server.tool(
        name="swg_gate",
        description=_GATE_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),  # pre-read mutates coordinator MESI state
        structured_output=False,
    )
    async def swg_gate(
        path: str,
        expected_version: int,
        expected_generation: int | None,
        ctx: Context,
    ) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_gate(
                sctx.volume, sctx.config, path, expected_version, expected_generation
            )

    @server.tool(
        name="swg_write_cas",
        description=_WRITE_CAS_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_write_cas(path: str, expected_version: int, new_content: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_write_cas(sctx.volume, sctx.config, sctx.cas_conflicts, path, expected_version, new_content)

    @server.tool(
        name="swg_transfer",
        description=_TRANSFER_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_transfer(paths: list[str], successor: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_transfer(sctx.volume, sctx.config, paths, successor)

    @server.tool(
        name="swg_accept",
        description=_ACCEPT_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_accept(path: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_accept(sctx.volume, sctx.config, path)

    @server.tool(
        name="swg_decline",
        description=_DECLINE_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_decline(path: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_decline(sctx.volume, sctx.config, path)

    @server.tool(
        name="swg_withdraw",
        description=_WITHDRAW_DESC,
        annotations=ToolAnnotations(readOnlyHint=False),
        structured_output=False,
    )
    async def swg_withdraw(path: str, ctx: Context) -> CallToolResult:
        sctx = _server_context(ctx)
        async with sctx.lock:
            return _do_withdraw(sctx.volume, sctx.config, path)


def build_server() -> FastMCP:
    """Build the FastMCP server (lifespan wired, tools registered)."""
    server = FastMCP(name=SERVER_NAME, instructions=INSTRUCTIONS, lifespan=lifespan)
    register_tools(server)
    return server


def _configure_stderr_logging() -> None:
    """Route all logging to stderr — stdout is the JSON-RPC channel (stdio invariant)."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)


def _exit_if_coherence_dir_refused(root: Path) -> None:
    """Exit 2 on one stderr line when a start would refuse ``root``'s
    ``.coherence`` (#498), as the other console scripts do, before the server
    serves. The lifespan's own refusal stays the backstop for a link made after
    this check."""
    try:
        check_coherence_dir(root)
    except CoherenceDirRefused as exc:
        # A cloned repository chose the name in this text, and a host runs the
        # server unattended: escaped, it cannot drive a terminal or split a log line.
        err(f"{SERVER_NAME}: {escape_nonprintable(str(exc))}")
        raise SystemExit(2) from None


def _config_or_exit() -> SessionConfig:
    """The spawn environment's config, or exit 2 on one stderr line, before
    anything starts, when it names a value the server cannot honour (an
    ``SWG_ON_HELD`` other than ``refuse`` or ``displace``). The message shows
    the value as ``repr`` writes it, so it stays one line."""
    try:
        return SessionConfig.from_env()
    except ValueError as exc:
        err(f"{SERVER_NAME}: {exc}")
        raise SystemExit(2) from None


#: How an operator gets a refuse-mode server to start, after the volume's own
#: reason it could not.
_REFUSE_MODE_FIX = (
    "Run a coordinator that can refuse (agent-coherence after 0.15.0, or the Claude Code "
    "plugin after 0.6.0), or unset SWG_ON_HELD to serve in displace mode."
)


def _serve_or_exit(server: FastMCP) -> None:
    """Serve over stdio. A refuse-mode start the coordinator cannot honour,
    raised by the lifespan's attach before anything is served, ends on one
    stderr line and exit 2, as a refused ``.coherence`` does."""
    try:
        server.run(transport="stdio")
    except* RefuseModeUnsupported as refused:
        # The volume's reason can quote what the coordinator answered.
        reason = escape_nonprintable(str(refused.exceptions[0]))
        err(f"{SERVER_NAME}: refusing to start with SWG_ON_HELD=refuse: {reason} {_REFUSE_MODE_FIX}")
        raise SystemExit(2) from None


def main() -> None:
    """Console-script entrypoint: run the server over stdio."""
    _configure_stderr_logging()
    _exit_if_coherence_dir_refused(_config_or_exit().root)
    _serve_or_exit(build_server())


if __name__ == "__main__":
    main()
