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
from contextlib import contextmanager
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


def _marked_reader(registry: object) -> tuple[uuid.UUID, uuid.UUID]:
    """A reader a peer's write left INVALID with its marker, on ``plan.md`` at
    v1 with the bytes ``v1``."""
    artifact_id, reader = uuid.uuid4(), uuid.uuid4()
    registry.register_artifact(  # type: ignore[attr-defined]
        Artifact(id=artifact_id, name=_PATH, version=1, content_hash=_hash("v1")), content="",
    )
    registry.set_agent_state(artifact_id, reader, MESIState.INVALID, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_transient(artifact_id, reader, TransientState.SIA, entered_tick=1)  # type: ignore[attr-defined]
    return artifact_id, reader


def _grant(registry: object, grant: str, ids: tuple[uuid.UUID, uuid.UUID]) -> None:
    """Grant SHARED after a stale read, the way the ``grant`` route does. The
    read route reads the recorded bytes."""
    artifact_id, reader = ids
    server = SimpleNamespace(registry=registry)  # the grant reads only the registry
    if grant == "read":
        _regrant_unless_overtaken(
            server, artifact_id, reader, classified_version=1, tick=2, content_hash=_hash("v1"),
        )
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
    artifact_id, reader = _marked_reader(registry)

    _grant(registry, grant, (artifact_id, reader))

    assert registry.get_agent_state(artifact_id, reader) == MESIState.SHARED  # type: ignore[attr-defined]
    assert registry.get_agent_transient(artifact_id, reader) is None  # type: ignore[attr-defined]
    service = CoordinatorService(registry)  # type: ignore[arg-type]
    assert service.enforce_transient_timeouts(current_tick=10**6, timeout_ticks=_TRANSIENT_TIMEOUT) == 0
    assert registry.get_agent_state(artifact_id, reader) == MESIState.SHARED  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("classified_version", "content_hash"),
    [(0, _hash("v1")), (1, _hash("older")), (1, None)],
    ids=["overtaken", "other-bytes", "no-hash"],
)
def test_a_read_that_did_not_see_the_recorded_bytes_keeps_the_marker_on_either_registry(
    registry: object, classified_version: int, content_hash: str | None,
) -> None:
    """Only a read of the recorded bytes acknowledges the peer's write. A read
    overtaken by a commit stays INVALID; a read of other bytes, or one that
    sent no hash, is granted SHARED but keeps the marker, so an optimistic
    commit built from it is still refused as mid-transition."""
    artifact_id, reader = _marked_reader(registry)

    _regrant_unless_overtaken(
        SimpleNamespace(registry=registry), artifact_id, reader,
        classified_version=classified_version, tick=2, content_hash=content_hash,
    )

    state = MESIState.INVALID if classified_version == 0 else MESIState.SHARED
    assert registry.get_agent_state(artifact_id, reader) == state  # type: ignore[attr-defined]
    assert registry.get_agent_transient(artifact_id, reader) == TransientState.SIA  # type: ignore[attr-defined]


def test_a_read_of_a_file_with_no_recorded_content_clears_the_marker_on_either_registry(registry: object) -> None:
    """A file registered with no content (first seen through Bash or an edit)
    has had no commit since, so no read of it can predate one: any read of it,
    hashed or not, acknowledges the peer's write."""
    artifact_id, reader = uuid.uuid4(), uuid.uuid4()
    registry.register_artifact(  # type: ignore[attr-defined]
        Artifact(id=artifact_id, name=_PATH, version=1, content_hash=""), content="",
    )
    registry.set_agent_state(artifact_id, reader, MESIState.INVALID, tick=1)  # type: ignore[attr-defined]
    registry.set_agent_transient(artifact_id, reader, TransientState.SIA, entered_tick=1)  # type: ignore[attr-defined]

    _regrant_unless_overtaken(
        SimpleNamespace(registry=registry), artifact_id, reader,
        classified_version=1, tick=2, content_hash=_hash("on-disk"),
    )

    assert registry.get_agent_state(artifact_id, reader) == MESIState.SHARED  # type: ignore[attr-defined]
    assert registry.get_agent_transient(artifact_id, reader) is None  # type: ignore[attr-defined]


class _CommitAsTheGrantLocks:
    """The registry, with a peer's optimistic commit landing just before the
    grant takes the registry lock."""

    def __init__(self, registry: object, artifact_id: uuid.UUID) -> None:
        self._registry = registry
        self._artifact_id = artifact_id

    def __getattr__(self, name: str) -> object:
        return getattr(self._registry, name)

    @contextmanager
    def abort_guard(self, *args: object, **kwargs: object) -> Iterator[None]:
        landed = self._registry.commit_cas(  # type: ignore[attr-defined]
            self._artifact_id, uuid.uuid4(), expected_version=1, content_hash=_hash("v2"), tick=2,
        )
        assert isinstance(landed, tuple), landed
        with self._registry.abort_guard(*args, **kwargs):  # type: ignore[attr-defined]
            yield


def test_a_commit_landing_as_the_grant_takes_the_lock_keeps_the_marker_on_either_registry(
    registry: object,
) -> None:
    """The overtaken check and the clear run under the registry lock, so a
    commit that lands just before it leaves the reader INVALID with its marker,
    not INVALID with the marker cleared by a check made before the commit."""
    artifact_id, reader = _marked_reader(registry)

    _regrant_unless_overtaken(
        SimpleNamespace(registry=_CommitAsTheGrantLocks(registry, artifact_id)), artifact_id, reader,
        classified_version=1, tick=3, content_hash=_hash("v1"),
    )

    assert registry.get_agent_state(artifact_id, reader) == MESIState.INVALID  # type: ignore[attr-defined]
    assert registry.get_agent_transient(artifact_id, reader) == TransientState.SIA  # type: ignore[attr-defined]


# --- an optimistic commit after the re-read ---------------------------------------


def _cas(http: _Client, sid: str, expected_version: int) -> dict:
    return http.post("/hooks/post-edit-cas", {
        "session_id": sid, "path": _PATH, "content_hash": _hash("cas"), "expected_version": expected_version,
    })[1]


@pytest.mark.parametrize("peer_lets_go", [True, False], ids=["peer-stopped", "peer-holds"])
def test_a_cas_after_the_reread_is_arbitrated_by_version_not_refused_as_mid_transition(
    live: _Live, peer_lets_go: bool,
) -> None:
    """After the re-read of the peer's committed bytes, the reader's optimistic
    commit meets the version check and the holder check, not the marker: at the
    version it re-read it lands once the peer has let go and is refused
    ``other_holder`` while the peer still holds the file; at the version from
    before the peer's write it is refused ``version_mismatch``. It used to be
    refused ``caller_in_transient_state`` until the sweep ran."""
    reader, peer = _sid(f"cas-reader-{peer_lets_go}"), _sid(f"cas-peer-{peer_lets_go}")
    recorded = _peer_makes_reader_stale(live.client, "peer-committed", reader, peer)
    if peer_lets_go:
        assert live.client.post("/hooks/session-stop", {"session_id": peer})[0] == 200
    _reread(live.client, "read", reader, recorded)

    assert _cas(live.client, reader, 1).get("reason") == "version_mismatch"
    if peer_lets_go:
        assert _cas(live.client, reader, 2) == {"ok": True, "version": 3}
    else:
        assert _cas(live.client, reader, 2).get("reason") == "other_holder"


def test_a_reread_of_older_bytes_keeps_the_marker_and_its_cas_is_refused(live: _Live) -> None:
    """A read whose bytes were taken before the peer's commit -- a volume reads
    the file before it asks the coordinator -- is granted SHARED at the new
    version but does not acknowledge the peer's write, so its optimistic commit
    is refused ``caller_in_transient_state`` and its writer reads again instead
    of overwriting the peer's change."""
    reader, peer = _sid("older-bytes-reader"), _sid("older-bytes-peer")
    _peer_makes_reader_stale(live.client, "peer-committed", reader, peer)
    assert live.client.post("/hooks/session-stop", {"session_id": peer})[0] == 200

    _reread(live.client, "read", reader, _hash("v1"))
    assert _marker(live.server, reader) == (MESIState.SHARED, TransientState.SIA)
    assert _cas(live.client, reader, 2).get("reason") == "caller_in_transient_state"
