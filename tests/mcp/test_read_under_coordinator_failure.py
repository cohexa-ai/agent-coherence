# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""MCP tool calls when the coordinator fails: an answered failure, and death.

**An answered failure.** Two coordinator-side failures answer a pre-read as
HTTP 200 ``{ok: false, reason: "internal: <Type>"}``: the caller-principal
gate's store read raising (a bound session whose binding is not cached — what
a coordinator restart leaves — under a locked registry) and the pre-read work
body raising. Neither confirms the read was registered (a body raise can fail
after part of it was), so the server's strict volume must fail it closed and
the tool call must answer a non-ignorable deny, never content: a read the
agent takes for registered is one a peer's later commit invalidates nothing
for, and the agent's next write lands over it.

**Death.** Once the coordinator the server's strict volume attached to is
gone, no request reaches it, and the volume keeps that dead endpoint: it never
re-attaches, so it never reaches a coordinator started later. Every tool call
must answer ``coordinator_unavailable`` with the stop verb, not
``internal_error`` (the shape of a bug in the server) and not the unattached
refusal's ``retry_later`` (no retry in this server session can succeed).

What the death tests do not prove: they run on one machine over loopback, and
the coordinator is shut down in this process, so a refused connect is the only
failure driven (no network partition, no coordinator that hangs, no process
killed by a signal). The Claude Code hook client is not driven and still fails
open when its coordinator is gone. The server's attach leg, which still spawns
or attaches a coordinator at startup, is unchanged and not driven. The Node
client is untouched.

Driven through the sync ``_do_*`` helpers against a coordinator running in
THIS process, so its registry can be made to fail and its listening socket
closed, as the volume's own tests do.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest
from mcp.types import CallToolResult

from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.mcp.server import _do_read, _do_write, _do_write_cas
from ccs.mcp.session import SessionConfig

#: FROZEN duplicate of the deny reason the mapper gives a coherence failure it
#: recognises no typed terminal for, which an answered failure envelope still
#: gets. A request that gets no answer at all gets ``_UNREACHABLE_REASON``
#: instead (never imported from the code under test).
_INTERNAL_ERROR = "internal_error"
#: FROZEN duplicates of the coordinator-unreachable terminal: its reason and
#: the house stop verb, never the unattached refusal's ``retry_later``.
_UNREACHABLE_REASON = "coordinator_unavailable"
_STOP_AND_REPORT = "stop_and_report"


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


def _serve_in_process(tmp_path: Path, instance_id: str):
    """A strict coordinator for ``data/**`` running in THIS process, reachable
    through the usual pid file, so a test can reach into its registry. Policy
    is loaded once at construction, so the YAML is written first."""
    from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer

    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    for name in ("tracked.yaml", "strict_mode.yaml"):
        (coherence / name).write_text("- data/**\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id=instance_id)
    server.serve_in_thread()
    (coherence / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
    return server


def _fail_the_pre_read(server, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """Make the coordinator answer the next pre-read with its failure envelope
    from one of its two arms: ``gate`` — the caller-principal gate's store
    read raises (a cold cache, as after a restart, and a locked registry);
    ``body`` — the pre-read work body raises (the watchdog call itself)."""

    def raising(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    if fault == "gate":
        with server.service._caller_principal_lock:
            server.service._caller_principals.clear()
        monkeypatch.setattr(server.registry, "get_caller_principal", raising)
    else:
        monkeypatch.setattr(server, "run_with_watchdog", raising)


@pytest.mark.parametrize("fault", ["gate", "body"])
def test_a_read_the_coordinator_answers_with_a_failure_is_a_deny_never_content(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """``swg_read`` answers ``isError`` with the fail-closed deny for a
    failure it has no typed terminal for — ``internal_error``, ``retryable:
    false`` — and no content, version or generation; the coordinator holds no
    view for the path; once the coordinator recovers the same call answers the
    content, registered. The envelope is an answer, so it is not
    ``coordinator_unavailable``, which is for a request that got none.

    Prevents the tool answering content with ``version: 0`` and no view
    registered: the agent reads that as a successful read and decides from
    it, and the server's own status reports nothing degraded."""
    rel = "data/shared.txt"
    _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = _serve_in_process(tmp_path, f"swg-read-{fault}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert vol.is_attached and vol.principal_claim_outcome == "bound", "control: bound"
        with monkeypatch.context() as faulted:
            _fail_the_pre_read(server, faulted, fault)
            result = _do_read(vol, config, rel)

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert sc["reason"] == _INTERNAL_ERROR
        assert sc["retryable"] is False
        assert "content" not in sc and "version" not in sc and "owner_generation" not in sc
        assert server.registry.lookup_artifact_id_by_name(rel) is None, "the read registered"
        assert vol.is_degraded is False, "the server's strict volume degraded"

        recovered = _do_read(vol, config, rel)
        assert recovered.isError is False, "control: the coordinator recovered"
        assert recovered.structuredContent["content"] == "v1"
        assert server.registry.lookup_artifact_id_by_name(rel) is not None
    finally:
        server.shutdown()


def _assert_stop_and_report(result: CallToolResult) -> None:
    """A non-ignorable ``coordinator_unavailable`` deny with the stop verb, not
    retryable, carrying its fixed next step, and no content or version."""
    assert result.isError is True, result.structuredContent
    sc = result.structuredContent
    assert sc["reason"] == _UNREACHABLE_REASON, sc
    assert sc["recover"] == _STOP_AND_REPORT, sc
    assert sc["retryable"] is False
    assert sc["next_step"], "the stop carries no next step"
    assert "content" not in sc and "version" not in sc and "owner_generation" not in sc


def _attach_then_shut_down(
    tmp_path: Path, fast_cfg: LifecycleConfig, instance_id: str
) -> tuple[CoherentVolume, SessionConfig, int]:
    """Attach the server's strict volume to a live coordinator and read the
    seeded path once (control: the coordinator answered), then shut the
    coordinator down. Its listening socket closes, so every later request is
    refused at connect. Returns the volume, the session config and the version
    the live read answered."""
    _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = _serve_in_process(tmp_path, instance_id)
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert vol.is_attached and vol.principal_claim_outcome == "bound", "control: bound"
        live = _do_read(vol, config, "data/shared.txt")
        assert live.isError is False, "control: the live coordinator answers"
    finally:
        server.shutdown()
    return vol, config, live.structuredContent["version"]


def test_after_the_coordinator_dies_read_then_write_each_stop_and_report_and_the_volume_stays_attached(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """AE10. After the coordinator dies, ``swg_read`` then ``swg_write`` each
    answer ``coordinator_unavailable`` with ``stop_and_report``, not
    retryable, and the volume stays attached between and after the calls, so
    the second call is refused by the same typed stop as the first rather than
    by the unattached ``retry_later`` refusal. The write leaves the file's
    bytes as they were.

    Prevents ``internal_error``, which an agent reads as a bug in the server,
    and ``retry_later``, which sends it to retry a session that can never
    reach a coordinator again. Without the mapper's row both calls answer
    ``internal_error``.

    Not proven here: one machine and loopback only, a refused connect only
    (no partition), the hook client (still fails open), the server's attach
    leg (unchanged), the Node client (untouched)."""
    rel = "data/shared.txt"
    vol, config, _version = _attach_then_shut_down(tmp_path, fast_cfg, "swg-dead-read-write")

    read = _do_read(vol, config, rel)
    _assert_stop_and_report(read)
    assert vol.is_attached is True, "the read detached the volume"

    write = _do_write(vol, config, rel, "v2")
    _assert_stop_and_report(write)
    assert vol.is_attached is True, "the write detached the volume"
    assert vol.is_degraded is False, "the server's strict volume degraded"
    assert (tmp_path / rel).read_bytes() == b"v1", "the refused write reached disk"


def test_after_the_coordinator_dies_a_cas_stops_and_reports_from_its_comparand_read(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """``swg_write_cas`` at the version the live read answered, after the
    coordinator dies, answers ``coordinator_unavailable`` with
    ``stop_and_report``. The refusal comes from the comparand read, the first
    request the CAS sends: this volume never wrote, so its re-mint releases
    nothing, and the detail (the volume's verbatim message) names the pre-read
    route. No conflict is counted, the volume stays attached, and nothing
    reaches disk.

    Not proven here: as for the read-then-write test above."""
    rel = "data/shared.txt"
    vol, config, version = _attach_then_shut_down(tmp_path, fast_cfg, "swg-dead-cas")
    conflicts: dict[str, int] = {}

    result = _do_write_cas(vol, config, conflicts, rel, version, "v2")

    _assert_stop_and_report(result)
    assert "/hooks/pre-read" in result.structuredContent["detail"], "not the comparand read"
    assert conflicts == {}, "a transport failure counted as a CAS conflict"
    assert vol.is_attached is True, "the CAS detached the volume"
    assert (tmp_path / rel).read_bytes() == b"v1", "the refused CAS reached disk"
