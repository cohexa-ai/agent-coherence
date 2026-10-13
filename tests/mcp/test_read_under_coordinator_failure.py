# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``swg_read`` when the coordinator answers the pre-read with a failure.

Two coordinator-side failures answer a pre-read as HTTP 200 ``{ok: false,
reason: "internal: <Type>"}``: the caller-principal gate's store read raising
(a bound session whose binding is not cached — what a coordinator restart
leaves — under a locked registry) and the pre-read work body raising. Neither
confirms the read was registered (a body raise can fail after part of it was),
so the server's strict volume must fail it closed and the
tool call must answer a non-ignorable deny, never content: a read the agent
takes for registered is one a peer's later commit invalidates nothing for,
and the agent's next write lands over it. Driven through the sync ``_do_*``
helpers against a coordinator running in THIS process so its registry can be
made to fail, as the volume's own tests do.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.mcp.server import _do_read
from ccs.mcp.session import SessionConfig

#: FROZEN duplicate of the deny reason the mapper gives a coherence failure it
#: recognises no typed terminal for — the same answer an unanswered request
#: gets (never imported from the code under test).
_INTERNAL_ERROR = "internal_error"


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
    """``swg_read`` answers ``isError`` with the typed deny an unanswered
    request gets — ``internal_error``, ``retryable: false`` — and no content,
    version or generation; the coordinator holds no view for the path; once
    the coordinator recovers the same call answers the content, registered.

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
