# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A grant after a stale read clears the pending-invalidation marker (#523).

A peer's pre-edit leaves each reader INVALID with a pending-invalidation marker
(SIA), which the transient sweep expires to INVALID a minute later if nothing
acknowledged it. In warn mode a session's stale Read, Bash command or Grep
search is allowed and grants the session SHARED again, and that read is the
acknowledgement. The grant used to keep the marker, so the sweep made the
reader INVALID a minute after its re-read, and its next Read was warned stale
with no new version. The grant now clears the marker, as the coordinator's own
fetch and a holder's release of its write grant do.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    CoordinatorHTTPServer,
    _apply_bash_grep_regrants,
    _regrant_unless_overtaken,
    session_to_agent_id,
)
from ccs.coordinator.service import CoordinatorService
from ccs.core.clock import monotonic_seconds
from ccs.core.states import MESIState, TransientState
from ccs.core.types import Artifact
from tests.test_claude_code_coordinator_server import _Client, _hash, _sid

_PATH = "plan.md"  # tracked and warn-mode on a fresh coordinator
_TRANSIENT_TIMEOUT = 60  # the lifecycle's default transient timeout


@dataclass(frozen=True)
class _Live:
    """A coordinator serving in this process and a client for it."""

    server: CoordinatorHTTPServer
    client: _Client


@pytest.fixture
def live(tmp_path: Path) -> Iterator[_Live]:
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="regrant")
    server.serve_in_thread()
    try:
        yield _Live(server, _Client("127.0.0.1", server.port, load_secret(server.coordinator_root)))
    finally:
        server.shutdown()


def _marker(server: CoordinatorHTTPServer, sid: str) -> tuple[MESIState | None, TransientState | None]:
    """The session's state on ``_PATH`` and the marker its row carries."""
    registry = server.registry
    artifact_id = registry.lookup_artifact_id_by_name(_PATH)
    agent_id = session_to_agent_id(sid)
    return registry.get_agent_state(artifact_id, agent_id), registry.get_agent_transient(artifact_id, agent_id)


def _read(http: _Client, sid: str, content_hash: str) -> dict:
    status, answer = http.post("/hooks/pre-read", {"session_id": sid, "path": _PATH, "content_hash": content_hash})
    assert status == 200, answer
    return answer


def _peer_makes_reader_stale(http: _Client, flow: str, reader: str, peer: str) -> str:
    """Leave ``reader`` INVALID with the peer write's marker, as ``flow`` says.
    Returns the hash of the bytes now recorded."""
    if flow == "after-own-stop":
        # The reader edited, committed and ended its turn first (#513): a Stop
        # leaves it SHARED, so the peer's pre-edit marks it like any reader.
        for route, body in (
            ("/hooks/pre-edit", {}),
            ("/hooks/post-edit", {"success": True, "content_hash": _hash("own")}),
            ("/hooks/session-stop", {}),
        ):
            assert http.post(route, {"session_id": reader, "path": _PATH, **body})[0] == 200
        recorded = _hash("own")
    else:
        _read(http, reader, _hash("v1"))
        recorded = _hash("v1")
    assert http.post("/hooks/pre-edit", {"session_id": peer, "path": _PATH})[1].get("ok") is True
    if flow == "peer-committed":
        answer = http.post("/hooks/post-edit", {
            "session_id": peer, "path": _PATH, "success": True, "content_hash": _hash("v2"),
        })[1]
        assert answer == {"ok": True}, answer
        recorded = _hash("v2")
    return recorded


def _reread(http: _Client, route: str, reader: str, recorded: str) -> None:
    """The reader's warn-mode read through ``route``: stale, and allowed."""
    if route == "read":
        answer = _read(http, reader, recorded)
    elif route == "bash":
        answer = http.post("/hooks/pre-bash", {"session_id": reader, "command": f"cat {_PATH}"})[1]
    else:
        answer = http.post("/hooks/pre-grep", {"session_id": reader, "search_root": ""})[1]
    assert answer.get("status") == "stale", answer


@pytest.mark.parametrize("route", ["read", "bash", "grep"])
@pytest.mark.parametrize("flow", ["peer-acquired", "peer-committed", "after-own-stop"])
def test_a_warn_mode_reread_stays_fresh_after_the_transient_sweep(live: _Live, route: str, flow: str) -> None:
    """The reader's stale re-read grants it SHARED and acknowledges the peer's
    write, so far past the transient timeout the sweep expires nothing, the
    reader is still SHARED, and its next Read is fresh. It used to keep the
    marker, so the sweep made it INVALID and that Read was warned stale with no
    new version."""
    reader, peer = _sid(f"reader-{route}-{flow}"), _sid(f"peer-{route}-{flow}")
    recorded = _peer_makes_reader_stale(live.client, flow, reader, peer)
    assert _marker(live.server, reader) == (MESIState.INVALID, TransientState.SIA)

    _reread(live.client, route, reader, recorded)
    assert _marker(live.server, reader) == (MESIState.SHARED, None)

    far_past = monotonic_seconds() + _TRANSIENT_TIMEOUT + 1
    assert live.server.service.enforce_transient_timeouts(
        current_tick=far_past, timeout_ticks=_TRANSIENT_TIMEOUT,
    ) == 0
    assert _marker(live.server, reader) == (MESIState.SHARED, None)
    assert _read(live.client, reader, recorded).get("status") == "fresh"


def test_a_strict_bash_retry_a_minute_after_its_deny_still_goes_through(tmp_path: Path) -> None:
    """A strict Bash command over a file another session's pre-edit made stale
    is denied once and grants the read, so its retry goes through. That grant
    acknowledges the marker too, so the retry still goes through after the
    transient sweep; it used to be denied again once the sweep ran."""
    coherence = tmp_path / ".coherence"
    coherence.mkdir()
    (coherence / "strict_mode.yaml").write_text(f"- {_PATH}\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="regrant-strict")
    server.serve_in_thread()
    try:
        http = _Client("127.0.0.1", server.port, load_secret(server.coordinator_root))
        reader, peer = _sid("strict-reader"), _sid("strict-peer")
        command = {"session_id": reader, "command": f"cat {_PATH}"}
        _peer_makes_reader_stale(http, "peer-acquired", reader, peer)

        denied = http.post("/hooks/pre-bash", command)[1]
        assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", denied
        assert _marker(server, reader) == (MESIState.SHARED, None)

        far_past = monotonic_seconds() + _TRANSIENT_TIMEOUT + 1
        assert server.service.enforce_transient_timeouts(current_tick=far_past, timeout_ticks=_TRANSIENT_TIMEOUT) == 0
        assert http.post("/hooks/pre-bash", command)[1].get("status") == "fresh"
    finally:
        server.shutdown()


# --- the grant itself, on both registries (the shared ``registry`` fixture) ------


def _grant(registry: object, grant: str, ids: tuple[uuid.UUID, uuid.UUID]) -> None:
    """Grant SHARED after a stale read, the way the ``grant`` route does."""
    artifact_id, reader = ids
    server = SimpleNamespace(registry=registry)  # the grant reads only the registry
    if grant == "read":
        _regrant_unless_overtaken(server, artifact_id, reader, classified_version=1, tick=2, content_hash=None)
    else:
        command_runs = grant == "bash-or-grep-ran"
        _apply_bash_grep_regrants(
            server, reader, [(artifact_id, "post_stale_bash")],
            held=[], command_runs=command_runs, tick=2,
        )


@pytest.mark.parametrize("grant", ["read", "bash-or-grep-ran", "bash-or-grep-denied"])
def test_the_grant_after_a_stale_read_clears_the_marker_on_either_registry(registry: object, grant: str) -> None:
    """Whichever route grants it, the reader ends SHARED with no marker, so the
    transient sweep has nothing to expire. A strict Bash or Grep deny grants
    the read too (the retry goes through), so it clears the marker as well."""
    artifact_id, reader = uuid.uuid4(), uuid.uuid4()
    registry.register_artifact(  # type: ignore[attr-defined]
        Artifact(id=artifact_id, name=_PATH, version=1, content_hash=_hash("v1")), content="",
    )
    registry.set_agent_state(artifact_id, reader, MESIState.INVALID, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_transient(artifact_id, reader, TransientState.SIA, entered_tick=1)  # type: ignore[attr-defined]

    _grant(registry, grant, (artifact_id, reader))

    assert registry.get_agent_state(artifact_id, reader) == MESIState.SHARED  # type: ignore[attr-defined]
    assert registry.get_agent_transient(artifact_id, reader) is None  # type: ignore[attr-defined]
    service = CoordinatorService(registry)  # type: ignore[arg-type]
    assert service.enforce_transient_timeouts(current_tick=10**6, timeout_ticks=_TRANSIENT_TIMEOUT) == 0
    assert registry.get_agent_state(artifact_id, reader) == MESIState.SHARED  # type: ignore[attr-defined]
