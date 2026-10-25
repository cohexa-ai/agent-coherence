# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A coordinator running in the test's own process, what its registry
records, and a status answer put in place of one of its answers, shared by
the volume, substrate and MCP tests that reach into that registry."""

from __future__ import annotations

import io
import json
import os
import urllib.error
from collections.abc import Callable
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.core.states import MESIState

#: What a status answer's reason phrase and body say. The volume reports a
#: status answer by its code alone, so none of this may reach what it raises
#: or warns.
ANSWERED_TEXT = "text-the-coordinator-answered-with"


def serve_in_process(tmp_path: Path, instance_id: str, tracked: str = "data/**") -> CoordinatorHTTPServer:
    """A strict coordinator for ``tracked`` running in THIS process, reachable
    through the usual pid file, so a test can reach into its registry. Policy
    is loaded once at construction, so the YAML is written first; a volume or
    a substrate session then attaches to it as to any spawned one."""
    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    for name in ("tracked.yaml", "strict_mode.yaml"):
        (coherence / name).write_text(f"- {tracked}\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id=instance_id)
    server.serve_in_thread()
    (coherence / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
    return server


def recorded_version(server: CoordinatorHTTPServer, rel: str) -> int:
    """The version the coordinator's registry records for ``rel``."""
    artifact_id = server.registry.lookup_artifact_id_by_name(rel)
    assert artifact_id is not None, f"control: {rel} is registered"
    return server.registry.get_artifact(artifact_id).version


def write_holders(server: CoordinatorHTTPServer, rel: str) -> set[str]:
    """Every grant row the coordinator keeps EXCLUSIVE or MODIFIED on ``rel``."""
    artifact_id = server.registry.lookup_artifact_id_by_name(rel)
    return {
        str(agent) for agent, state in server.registry.get_state_map(artifact_id).items()
        if state in (MESIState.EXCLUSIVE, MESIState.MODIFIED)
    }


def answer_with_status(
    monkeypatch: pytest.MonkeyPatch, picks: Callable[[str, dict], bool], status: int,
    *, forwarded: bool = True, calls: list[tuple[str, dict, dict]] | None = None,
) -> None:
    """Answer each request ``picks(route, payload)`` picks with HTTP ``status``,
    through the client's ``HTTPError`` carrying :data:`ANSWERED_TEXT`; send
    every other request to the coordinator. ``forwarded``: the coordinator got
    the request and answered it first, as behind a gateway, so the work behind
    the status ran; otherwise the request never reached it. ``calls``, when
    given, records every request as ``(route, payload, keyword arguments)``."""
    real_post = coherent_volume_module._coordinator_post
    body = json.dumps({"error": f"internal: RuntimeError ({ANSWERED_TEXT})"}).encode()

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if calls is not None:
            calls.append((path, dict(payload), dict(kwargs)))
        if not picks(path, payload):
            return real_post(endpoint, path, payload, **kwargs)
        if forwarded:
            real_post(endpoint, path, payload, **kwargs)
        # A fresh body per answer: a reader that drains one must not leave the
        # next answer empty.
        raise urllib.error.HTTPError(path, status, ANSWERED_TEXT, {}, io.BytesIO(body))  # type: ignore[arg-type]

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", post)
