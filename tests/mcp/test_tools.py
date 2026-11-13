"""Unit 4 — swg_read / swg_write / swg_reacquire against a real coordinator.

Tests drive the sync ``_do_*`` helpers (the tool logic) with real
``CoherentVolume`` instances + a real loopback coordinator, so the contract is
exercised end-to-end without a FastMCP client. The headline is the sequential
deny→reacquire→write loop; the rest is fail-closed + edge coverage + the honesty
surface (SC4).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import re
import sys
import time
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import get_args

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

import ccs.mcp.deny as deny
import ccs.mcp.server as server_module
import ccs.mcp.status as status_module
from ccs.adapters.claude_code.coordinator_server import session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import (
    CALLER_PRINCIPAL_REASONS,
    HANDOFF_TRANSFER_REFUSAL_REASONS,
    CallerPrincipalRefused,
    CasRetriesExhausted,
    CasVersionConflict,
    CoherenceDirRefused,
    CoherenceError,
    GiverFenced,
    InternalConcurrencyError,
    LinkOutOfManaged,
    OtherHolder,
    RedirectRefused,
    ViewWedged,
)
from ccs.mcp.deny import (
    cas_exhausted_result,
    coordinator_unavailable_result,
    deny_result,
    handoff_deny_result,
    handoff_transfer_refusal_result,
    handoff_verb_refusal_result,
    release_deny_result,
    write_deny_result,
)
from ccs.mcp.server import (
    _READ_DESC,
    _STATUS_DESC,
    _WRITE_DESC,
    INSTRUCTIONS,
    _do_reacquire,
    _do_read,
    _do_write,
    _do_write_cas,
)
from ccs.mcp.session import OnHeld, SessionConfig

_SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def _seed(tmp_path: Path, rel: str = "data/shared.txt", content: bytes = b"v1") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _vol(tmp_path: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=cfg)


def _config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(root=tmp_path.resolve(), managed=("data/**",))


# --- happy + the headline sequential loop ------------------------------------


def test_read_then_write_happy(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    target = _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        read = _do_read(vol, config, "data/shared.txt")
        assert read.isError is False
        assert read.structuredContent["content"] == "v1"
        assert isinstance(read.structuredContent["version"], int)

        wrote = _do_write(vol, config, "data/shared.txt", "v2")
        assert wrote.isError is False
        assert target.read_bytes() == b"v2"
    finally:
        stop_coordinator(tmp_path)


def test_sequential_deny_reacquire_write_loop(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """read v → peer commits v+1 → swg_write(stale_view) → swg_reacquire(v+1) →
    swg_write(from those bytes) ok. The lost update is denied, not silent."""
    target = _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    vol_a = _vol(tmp_path, fast_cfg)
    vol_b = _vol(tmp_path, fast_cfg)
    try:
        assert _do_read(vol_a, config, "data/shared.txt").structuredContent["content"] == "v1"

        # Peer B commits v2 → A goes INVALID.
        _do_read(vol_b, config, "data/shared.txt")
        assert _do_write(vol_b, config, "data/shared.txt", "v2-from-b").isError is False

        # A's stale write is DENIED with the recoverable stale_view terminal.
        denied = _do_write(vol_a, config, "data/shared.txt", "v2-from-a")
        assert denied.isError is True
        assert denied.structuredContent["reason"] == "stale_view"
        assert denied.structuredContent["recover"] == "reacquire"
        assert denied.structuredContent["retryable"] is True
        assert target.read_bytes() == b"v2-from-b"  # A's stale write did NOT land

        # A reacquires → fresh bytes, then writes FROM them → ok.
        reacq = _do_reacquire(vol_a, config, "data/shared.txt")
        assert reacq.isError is False
        assert reacq.structuredContent["content"] == "v2-from-b"
        assert "version lineage" in reacq.structuredContent["note"]

        wrote = _do_write(vol_a, config, "data/shared.txt", "v3-from-a")
        assert wrote.isError is False
        assert target.read_bytes() == b"v3-from-a"
    finally:
        stop_coordinator(tmp_path)


# --- fail-closed -------------------------------------------------------------


def test_write_unattached_fails_closed_no_disk_write(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A mid-session endpoint loss must fail closed BEFORE the adapter's
    best-effort unversioned write — coordinator_unavailable, disk untouched."""
    target = _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        vol._endpoint = None  # simulate a post-construction endpoint loss
        result = _do_write(vol, config, "data/shared.txt", "should-not-land")
        assert result.isError is True
        assert result.structuredContent["reason"] == "coordinator_unavailable"
        assert target.read_bytes() == b"v1"  # NO disk write
    finally:
        stop_coordinator(tmp_path)


# --- edges -------------------------------------------------------------------


def test_read_invalid_path_is_non_deny_error(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        result = _do_read(vol, config, "../escape")
        assert result.isError is True
        assert result.structuredContent["reason"] == "invalid_path"
        assert result.structuredContent["reason"] != "stale_view"
    finally:
        stop_coordinator(tmp_path)


def test_read_missing_file_is_not_stale_view(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    _seed(tmp_path, content=b"v1")  # creates data/ but not nope.txt
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        result = _do_read(vol, config, "data/nope.txt")
        assert result.isError is True
        assert result.structuredContent["reason"] == "file_not_found"
        assert result.structuredContent["reason"] != "stale_view"
    finally:
        stop_coordinator(tmp_path)


def test_read_empty_file_is_valid_view(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    _seed(tmp_path, rel="data/empty.txt", content=b"")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        result = _do_read(vol, config, "data/empty.txt")
        assert result.isError is False
        assert result.structuredContent["content"] == ""
    finally:
        stop_coordinator(tmp_path)


def test_read_binary_file_unsupported(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    _seed(tmp_path, rel="data/bin.dat", content=b"\xff\xfe\x00\x01")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        result = _do_read(vol, config, "data/bin.dat")
        assert result.isError is True
        assert result.structuredContent["reason"] == "binary_unsupported"
    finally:
        stop_coordinator(tmp_path)


# --- honesty surface (SC4) ---------------------------------------------------


def test_tool_descriptions_state_the_scope() -> None:
    for desc in (_READ_DESC, _WRITE_DESC, _STATUS_DESC, *_registered_descriptions().values()):
        assert "SINGLE-HOST" in desc
        assert "auto-merge" in desc
    assert "heterogeneous_scope_detectable=false" in _STATUS_DESC


#: FROZEN duplicates of what the registered ``swg_status`` description must
#: name (never imported from the code under test): the session's
#: ``principal_claim`` with its five values, and the two coordinator counters.
_PRINCIPAL_CLAIM_VALUES = ("bound", "unsupported", "unconfirmed", "refused", "not_attempted")
_PRINCIPAL_COUNTERS = ("caller_principal_absent_total", "caller_principal_refused_total")


def test_the_registered_status_description_names_the_principal_claim_and_counters() -> None:
    """The description the model reads for ``swg_status`` — the registered
    tool's, as a client lists it — names ``principal_claim``, each of its
    five values, what ``refused`` means for every later tool call, and the
    two caller-principal counters as ``null`` when unreported. A model that
    only sees the description cannot otherwise tell a session that lost
    coordination from a coordinator that is off."""
    import asyncio

    from ccs.mcp.server import build_server

    tools = {tool.name: tool for tool in asyncio.run(build_server().list_tools())}
    description = tools["swg_status"].description or ""

    assert "principal_claim" in description
    for value in _PRINCIPAL_CLAIM_VALUES:
        assert value in description, value
    assert "refused" in description and "restart_session" in description
    for counter in _PRINCIPAL_COUNTERS:
        assert counter in description, counter
    assert "null" in description


#: FROZEN duplicate of the store-format field ``swg_status`` forwards (#294).
_SCHEMA_FIELD = "registry_schema_version"


def test_the_registered_status_description_names_the_registry_schema_version() -> None:
    """The registered ``swg_status`` description names
    ``registry_schema_version`` and, in the same clause, says it is ``null``
    when the coordinator does not report one. Prevents a model reading a null
    schema as a store with no version, or never learning the field exists."""
    description = _registered_descriptions()["swg_status"]

    assert _SCHEMA_FIELD in description
    clause = description.split(_SCHEMA_FIELD, 1)[1].split(")", 1)[0]
    assert "null" in clause, clause


#: FROZEN duplicate of the four handoff tools' registered names (#185).
_HANDOFF_TOOLS = ("swg_transfer", "swg_accept", "swg_decline", "swg_withdraw")


def _registered_descriptions() -> dict[str, str]:
    """Each tool's description as a client lists it from the built server."""
    import asyncio

    from ccs.mcp.server import build_server

    return {
        tool.name: tool.description or ""
        for tool in asyncio.run(build_server().list_tools())
    }


def test_every_handoff_tool_and_the_instructions_say_a_transfer_fences_the_giver_and_reserves_nothing() -> None:
    """A model that reads only one handoff tool's description, or only the
    server instructions, must learn both halves: a transfer fences the giver
    (its own later writes are refused) and does not reserve the path (other
    sessions keep writing it). Read as a reservation, a successor would wait
    on a path a bystander can still overwrite."""
    descriptions = _registered_descriptions()

    for name in _HANDOFF_TOOLS:
        text = descriptions[name].lower()
        assert "fences the giver" in text, name
        assert "does not reserve the path" in text, name
        assert "single-host" in text, name
    instructions = INSTRUCTIONS.lower()
    assert "fences the giver" in instructions
    assert "does not reserve the path" in instructions
    assert "handed_off" in instructions and "stop_and_report" in instructions


def test_the_transfer_tool_and_the_instructions_say_the_fence_covers_this_session_only() -> None:
    """The fence is on this MCP session's writes. The same model writing the
    path through its own file tools or a shell is another session, and nothing
    refuses it, so the giver must learn before its first refused write, not
    after, that it may write a handed-off path by no route at all."""
    transfer = _registered_descriptions()["swg_transfer"].lower()
    instructions = INSTRUCTIONS.lower()

    for text in (transfer, instructions):
        assert "the fence covers this mcp session only" in text
        assert "is not refused" in text


def test_the_accept_tool_says_an_overtaken_handoff_answers_overtaken() -> None:
    """An accept of a handoff a bystander overtook is taken but changes
    nothing and answers ``status=overtaken`` with the bystander as
    ``counterparty``: not an error, so a successor reading only the
    description must learn that an accept is not always a completion."""
    text = _registered_descriptions()["swg_accept"]

    assert "status=overtaken" in text
    assert "counterparty" in text
    assert "nothing has written it since" in text  # only an acquire leaves it live
    assert "refused handoff_not_live" in text


def test_the_transfer_tool_says_how_a_giver_tells_its_transfer_landed() -> None:
    """A giver unsure whether an earlier transfer landed (its context was
    compacted, or the session restarted) may hold no result's next step, only
    the tool's description, so the description carries the same rule: a
    handoff at the version it held means the transfer landed, and transferring
    again or withdrawing to start over hands the path on a second time."""
    text = _registered_descriptions()["swg_transfer"]

    assert "version_at_transfer" in text
    assert "means your transfer landed" in text
    assert "do not transfer again, and do not withdraw to start over" in text


def test_the_transfer_tool_names_the_successor_by_the_id_its_own_status_tool_reports() -> None:
    """The giver names the successor by the value the successor's
    OWN status tool reports, and the status tool says that value names the
    session as a successor only while its principal claim is bound (the
    coordinator knows a session-level id through its principal binding)."""
    descriptions = _registered_descriptions()

    assert "session_agent_id" in descriptions["swg_transfer"]
    assert "swg_status" in descriptions["swg_transfer"]
    assert "session_agent_id" in descriptions["swg_status"]
    assert "only while principal_claim is bound" in descriptions["swg_status"]


def test_the_withdraw_tool_is_taken_only_on_instruction_and_is_never_the_givers_recovery() -> None:
    """The giver's withdraw lifts its own fence, so it is one call away from
    a fenced giver. Its description says it is taken only on the user's or
    host's explicit instruction and is never the recovery for the
    ``handed_off`` refusal, beside the non-reservation statement; and, the
    negative pin, it names no way back to writing: no recover verb, no
    retry, no write tool to call after it."""
    text = _registered_descriptions()["swg_withdraw"]

    assert "only on the user's or host's explicit instruction" in text
    assert "never the recovery for the handed_off refusal" in text
    assert "does not reserve the path" in text
    assert "recover=" not in text
    assert "retry" not in text.lower()
    assert "swg_write" not in text


def test_the_read_and_cas_descriptions_name_the_handoff_key() -> None:
    """The read result carries the path's handoff record and a CAS win
    what it did to a live handoff; the descriptions say so, so a model knows
    the key it may find."""
    descriptions = _registered_descriptions()

    assert "handoff" in descriptions["swg_read"]
    assert "handoff" in descriptions["swg_write_cas"]


def test_instructions_state_forbidden_and_trust_boundary() -> None:
    text = INSTRUCTIONS.lower()
    assert "different hosts" in text
    assert "auto-merge" in text
    assert "trust boundary" in text
    assert "single-uid" in text


# --- what the server tells an agent up front matches what it answers ----------

#: FROZEN: the tools a client lists, eleven, ``swg_release`` among them.
_TOOL_NAMES = (
    "swg_read", "swg_write", "swg_reacquire", "swg_release", "swg_status", "swg_gate",
    "swg_write_cas", "swg_transfer", "swg_accept", "swg_decline", "swg_withdraw",
)
#: FROZEN: the refusal types the shared table answers that a swg_write never
#: meets: a compare-and-set's own three, a start's refusal of ``.coherence``
#: (no tool call carries it), and a second thread on the volume, which the
#: server's lock rules out.
_NOT_FROM_A_WRITE = (CasVersionConflict, CasRetriesExhausted, ViewWedged, CoherenceDirRefused, InternalConcurrencyError)
#: The tools whose refusals map through the shared table alone.
_SHARED_TABLE_TOOLS = ("swg_read", "swg_gate", "swg_reacquire", "swg_write_cas")
#: Constructor arguments for the refusal types that take more than a message.
_REFUSAL_ARGS: dict[type, tuple[object, ...]] = {
    GiverFenced: ("data/x.md", "6b0f3c1e-4a5d-5b7e-9c2f-1d3e5a7b9c0d", 3),
    OtherHolder: ("data/x.md", "5d0f3c9a8b7e4d21a6c3b2e1f0d9c8b7"),
    LinkOutOfManaged: ("data/l/x.md", "data/l", "other/x.md"),
    CasRetriesExhausted: ("data/x.md", 8, 3),
    CoherenceDirRefused: ("coherence_link", "/ws/.coherence", "detail"),
    RedirectRefused: ("http://elsewhere/", 307),
}


def _names(text: str, token: str) -> bool:
    """Whether ``text`` names ``token`` as a whole word: ``swg_reacquire`` does
    not name ``reacquire``, nor ``stop_and_report`` ``stop``."""
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", text) is not None


def _flat(text: str) -> str:
    """``text`` with each run of whitespace one space, so a phrase is found
    across the instructions' line breaks."""
    return " ".join(text.split())


def _instances(kind: type) -> list[BaseException]:
    """A refusal of ``kind``, and one more per flag the class declares, set
    the other way: a mapper may answer a flagged refusal with its own row."""
    args = _REFUSAL_ARGS.get(kind, ("detail",))
    flags = {name: not value for name, value in vars(kind).items() if isinstance(value, bool)}
    return [kind(*args), *(kind(*args, **{name: value}) for name, value in flags.items())]


def _refusals(kinds: Iterable[type]) -> list[BaseException]:
    """Every refusal of ``kinds``, each compare-and-set refusal by the reason
    it carries, each caller-principal refusal settled or not, and a plain
    ``CoherenceError``, which no row recognises."""
    found = [exc for kind in kinds for exc in _instances(kind)]
    found += [CasVersionConflict("data/x.md", 1, 2, reason=reason) for reason in deny._CAS_TERMINALS]
    found += [
        CallerPrincipalRefused(reason, "detail", settled=settled)
        for reason in sorted(CALLER_PRINCIPAL_REASONS)
        for settled in (True, False)
    ]
    return [*found, CoherenceError("detail")]


def _handoff_answers(every: list[BaseException]) -> dict[str, list[CallToolResult]]:
    """The four handoff tools' refusals: their mapper over ``every``, and the
    typed refusals each returns as a value."""
    refused_grants = [
        handoff_transfer_refusal_result([{"path": "data/x.md", "transferred": False, "reason": reason}], "d")
        for reason in sorted(HANDOFF_TRANSFER_REFUSAL_REASONS)
    ]
    answers = {"swg_transfer": [*(handoff_deny_result(exc, "transfer") for exc in every), *refused_grants]}
    for verb in sorted(deny.HANDOFF_UNCONFIRMED_NEXT_STEPS.keys() - {"transfer"}):
        refused = [handoff_verb_refusal_result(reason, "d", {}) for reason in sorted(deny.HANDOFF_REFUSALS)]
        answers[f"swg_{verb}"] = [*(handoff_deny_result(exc, verb) for exc in every), *refused]
    return answers


def _answers() -> dict[str, list[dict]]:
    """Each tool's possible refusals, from the mapper its code calls, over
    every refusal type the deny tables answer (more than each tool's volume
    call raises), plus the unattached refusal every tool can give. Taken from
    the tables, so a new row is checked without a change here."""
    kinds = dict.fromkeys([*deny._TERMINALS, *deny._WRITE_TERMINALS, *deny._RELEASE_UNCONFIRMED_TYPES])
    every = _refusals(kinds)
    writes = [exc for exc in every if type(exc) not in _NOT_FROM_A_WRITE]
    results = {
        "swg_write": [write_deny_result(exc, mode) for exc in writes for mode in get_args(OnHeld)],
        "swg_release": [release_deny_result(exc) for exc in every],
        **{tool: [deny_result(exc) for exc in every] for tool in _SHARED_TABLE_TOOLS},
        **_handoff_answers(every),
    }
    results["swg_write_cas"].append(cas_exhausted_result("detail"))
    unattached = coordinator_unavailable_result("detail")
    return {tool: [r.structuredContent for r in [*found, unattached]] for tool, found in results.items()}


def _client_errors(function: str | None = None) -> set[tuple[str, str]]:
    """Each ``(reason, recover)`` the server's code answers an input or disk
    error with, read from its ``_client_error_result`` calls: in ``function``
    when one is named, else anywhere in the module."""
    tree = ast.parse(inspect.getsource(server_module))
    scopes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == function]
    return {
        (call.args[0].value, call.args[1].value)
        for scope in (scopes if function else [tree])
        for call in ast.walk(scope)
        if isinstance(call, ast.Call) and getattr(call.func, "id", None) == "_client_error_result"
    }


def test_the_write_description_names_every_reason_swg_write_can_answer() -> None:
    """Every ``reason`` ``_do_write`` can return -- each swg_write row, each
    shared row a write reaches, the unattached refusal and its input and disk
    errors, all taken from the code -- is named in the registered swg_write
    description, as is ``opted_in``, the key a refuse-mode server's success
    adds. Prevents an agent meeting a refusal (``other_holder``, say) of which
    the tool it called says nothing."""
    reasons = {answer["reason"] for answer in _answers()["swg_write"]}
    reasons |= {reason for reason, _ in _client_errors("_do_write")}
    description = _registered_descriptions()["swg_write"]

    assert {"other_holder", "link_out_of_managed", "invalid_path"} <= reasons  # the derivation reached them
    assert [reason for reason in sorted(reasons) if not _names(description, reason)] == []
    assert _names(description, "opted_in")


def test_every_recover_verb_is_explained_in_the_instructions_or_the_tool_that_answers_it() -> None:
    """Every ``recover`` value any row can return -- swg_write's rows,
    swg_release's, the shared rows, the handoff rows and the input errors,
    taken from the code -- is named in the server instructions or in the
    description of the tool that returns it; an input error's, which several
    tools return, in the instructions. Prevents a verb (``write_again``,
    ``wait_then_reacquire``) that the agent is told to act on and nothing
    explains."""
    descriptions = _registered_descriptions()
    verbs = {(tool, answer["recover"]) for tool, answers in _answers().items() for answer in answers}
    unexplained = sorted(
        (tool, verb) for tool, verb in verbs
        if not (_names(INSTRUCTIONS, verb) or _names(descriptions[tool], verb))
    )
    unexplained += sorted({("input error", verb) for _, verb in _client_errors() if not _names(INSTRUCTIONS, verb)})

    assert {("swg_write", "write_again"), ("swg_release", "release_again")} <= verbs  # the derivation reached them
    assert unexplained == []


def test_reacquire_and_write_cas_say_they_drop_every_hold() -> None:
    """Both re-mint this session's attempt, which gives back every write grant
    it holds, not only the path named (measured). Prevents a holder that calls
    either and believes its other paths still keep peers out."""
    descriptions = _registered_descriptions()

    for tool in ("swg_reacquire", "swg_write_cas"):
        assert "drops every hold this session has" in _flat(descriptions[tool]), tool


def test_the_instructions_state_refuse_mode_retryable_and_who_honours_a_hold() -> None:
    """The instructions name the operator's switch, the release tool, what
    ``retryable`` means and does not, and the writers that take a held file
    anyway, with what the holder then sees. Prevents an agent reading a hold
    as a lock, or ``retryable: true`` as the step to take."""
    text = _flat(INSTRUCTIONS)

    assert _names(text, "SWG_ON_HELD") and _names(text, "swg_release")
    assert "retryable=true means only that sending the identical call again is safe" in text
    assert "it is not the next step" in text
    assert "only opted-in writers honour a hold" in text
    for writer in ("the Claude Code plugin's hooks", "displace-mode servers", "older volumes"):
        assert writer in text, writer
    assert "the holder then sees stale_view" in text


def test_the_instructions_give_the_hold_lifecycle_and_the_identity_note() -> None:
    """What takes, keeps and drops a hold, and how an agent tells its own
    holds: by ``held_by``, never by comparing ids, since holder ids are per
    attempt; and every caller of one server is one agent to the coordinator.
    Prevents an agent waiting on its own earlier attempt as on a stranger."""
    text = _flat(INSTRUCTIONS)

    assert "swg_write takes a write grant on the path it writes and keeps it after the write" in text
    assert "swg_reacquire and swg_write_cas, of any path, drop every hold this session has" in text
    assert "so does this server's normal exit" in text
    assert "holder ids are per attempt and never equal your session_agent_id" in text
    assert _names(text, "held_by")
    assert "Every caller of one server process is one agent to the coordinator" in text


def test_the_instructions_and_the_write_tool_tell_a_writer_to_release_a_finished_path_and_why() -> None:
    """The holder's duty, in both modes, where an agent reads it before its
    first write: release a path once finished, because other opted-in agents
    that write it wait until then, for up to 30 minutes. Prevents a holder
    that keeps every path it ever wrote while peers wait."""
    for text in (INSTRUCTIONS, _registered_descriptions()["swg_write"]):
        flat = _flat(text)
        assert "When you finish writing a path, swg_release it" in flat
        assert "until you do, other opted-in agents that write it wait, for up to 30 minutes" in flat


def test_the_status_description_gives_the_mode_the_holds_and_each_holder() -> None:
    """The registered swg_status description names ``on_held`` with its two
    values, ``held`` with null as cannot tell, and the per-path
    ``holder_agent_id`` and ``held_by`` with each value the code gives it.
    Prevents a null ``held`` read as "nothing held", the state in which a
    waiting agent would skip its own releases."""
    text = _flat(_registered_descriptions()["swg_status"])

    for token in ("on_held", *get_args(OnHeld), "held", "holder_agent_id", "held_by"):
        assert _names(text, token), token
    for value in sorted(set(status_module._HELD_BY.values())):
        assert _names(text, value), value
    assert "null means cannot tell, never that nothing is held" in text


async def _listed_over_stdio(workspace: Path, errlog: Path) -> list[str]:
    """The tool names a stdio client lists from the server, started as a host
    starts it, with its stderr in ``errlog``."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ccs.mcp.server"],
        env={"PYTHONPATH": str(_SRC), "SWG_ROOT": str(workspace), "SWG_MANAGED": "data/**"},
        cwd=str(workspace),
    )
    with errlog.open("w") as stream:
        async with stdio_client(params, errlog=stream) as (read, write), ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=60)
            listed = await asyncio.wait_for(session.list_tools(), timeout=60)
    return [tool.name for tool in listed.tools]


def test_a_stdio_client_lists_eleven_tools_with_swg_release_among_them(tmp_path: Path) -> None:
    """``tools/list`` over a real stdio session lists the eleven tools,
    ``swg_release`` among them. Prevents a tool the in-process listing has but
    the transport a host uses does not serve."""
    workspace = tmp_path / "ws"
    _seed(workspace)
    errlog = tmp_path / "server.stderr"

    names = asyncio.run(_listed_over_stdio(workspace, errlog))

    assert sorted(names) == sorted(_TOOL_NAMES), errlog.read_text()
    assert len(names) == 11 and "swg_release" in names


# --- fail-closed: unattached read/reacquire + IO errors ----------------------


def test_read_unattached_fails_closed(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        vol._endpoint = None
        result = _do_read(vol, config, "data/shared.txt")
        assert result.isError is True
        assert result.structuredContent["reason"] == "coordinator_unavailable"
    finally:
        stop_coordinator(tmp_path)


def test_reacquire_unattached_fails_closed(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        vol._endpoint = None
        result = _do_reacquire(vol, config, "data/shared.txt")
        assert result.isError is True
        assert result.structuredContent["reason"] == "coordinator_unavailable"
    finally:
        stop_coordinator(tmp_path)


def test_write_os_error_fails_closed(tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch) -> None:
    """A non-CoherenceError OSError (disk full / permission) from the underlying
    write must fail closed as a tool error, never escape to FastMCP."""
    _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        monkeypatch.setattr(vol, "write", lambda *a, **k: (_ for _ in ()).throw(PermissionError("disk")))
        result = _do_write(vol, config, "data/shared.txt", "x")
        assert result.isError is True
        assert result.structuredContent["reason"] == "io_error"
    finally:
        stop_coordinator(tmp_path)


# --- the lost token on commit refusals ----------------------------------------

_PAIR_KEYS = ("captured_generation", "owner_generation")


def _holds_exclusive(vol: CoherentVolume, holder: str, path: str) -> bool:
    status = vol.coordinator_status()
    assert status is not None, "the coordinator must be reachable to read its grant rows"
    rows = [row["states"] for row in status["sessions"] if row["agent_id"] == holder]
    return bool(rows) and rows[0].get(path) == "EXCLUSIVE"


def test_write_reclaimed_before_its_commit_is_commit_preempted_naming_the_lost_token(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch
) -> None:
    """swg_write's write is paused after its bytes reach disk until the sweep
    reclaims its grant (it sends nothing meanwhile, and ``/status`` heartbeats
    no one). The refusal keeps its recovery and adds the pair: the claim the
    pre-edit recorded at generation 0, against the reclaim's 1."""
    _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    # Whole-second ticks: a 2 s timeout reclaims 1-2 s after the grant, never
    # before the paused write has seen it held.
    vol = _vol(tmp_path, replace(fast_cfg, grant_heartbeat_timeout_sec=2))
    try:
        _do_read(vol, config, "data/shared.txt")
        holder = str(session_to_agent_id(vol.session_id, vol._incarnation))
        real_atomic_write = vol._atomic_write

        def write_then_wait_for_the_reclaim(abs_path: Path, data: bytes) -> None:
            real_atomic_write(abs_path, data)
            assert _holds_exclusive(vol, holder, "data/shared.txt"), "the write took no grant to lose"
            deadline = time.monotonic() + 30.0
            while _holds_exclusive(vol, holder, "data/shared.txt"):
                assert time.monotonic() < deadline, "the sweep never reclaimed the paused write's grant"
                time.sleep(0.05)

        monkeypatch.setattr(vol, "_atomic_write", write_then_wait_for_the_reclaim)
        result = _do_write(vol, config, "data/shared.txt", "v2")

        assert result.isError is True
        sc = result.structuredContent
        assert (sc["reason"], sc["recover"], sc["retryable"]) == (
            "commit_preempted", "reacquire_and_reconcile", False,
        )
        assert (sc["captured_generation"], sc["owner_generation"]) == (0, 1)
        assert "reclaimed by the coordinator sweep" in sc["detail"]
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_refusal_carries_no_pair_keys(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """swg_write_cas commits from a fresh incarnation whose read records no
    claim, so its refusal from the coordinator carries neither key, never a
    null. A holds the grant after its write, so B is refused other_holder."""
    _seed(tmp_path, content=b"v1")
    config = _config(tmp_path)
    vol_a, vol_b = _vol(tmp_path, fast_cfg), _vol(tmp_path, fast_cfg)
    try:
        _do_read(vol_a, config, "data/shared.txt")
        assert _do_write(vol_a, config, "data/shared.txt", "v2-from-a").isError is False

        result = _do_write_cas(vol_b, config, {}, "data/shared.txt", 2, "v3-from-b")

        assert result.isError is True
        sc = result.structuredContent
        assert (sc["reason"], sc["current_version"]) == ("other_holder", 2)
        assert not set(_PAIR_KEYS) & set(sc)
    finally:
        stop_coordinator(tmp_path)
