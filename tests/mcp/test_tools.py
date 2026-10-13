"""Unit 4 — swg_read / swg_write / swg_reacquire against a real coordinator.

Tests drive the sync ``_do_*`` helpers (the tool logic) with real
``CoherentVolume`` instances + a real loopback coordinator, so the contract is
exercised end-to-end without a FastMCP client. The headline is the sequential
deny→reacquire→write loop; the rest is fail-closed + edge coverage + the honesty
surface (SC4).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.mcp.server import (
    _READ_DESC,
    _STATUS_DESC,
    _WRITE_DESC,
    INSTRUCTIONS,
    _do_reacquire,
    _do_read,
    _do_write,
)
from ccs.mcp.session import SessionConfig


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
    for desc in (_READ_DESC, _WRITE_DESC, _STATUS_DESC):
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
