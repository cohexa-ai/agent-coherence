# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A route the coordinator does not serve names itself.

A coordinator answers HTTP 404 only from its router, for a route it does not
serve; the Node coordinator serves no snapshot sessions and no handoffs. A 404
on one of those routes therefore means nothing ran, and the same request gets
the same answer. The volume raises ``CoherenceError`` naming the route, in both
``on_error`` modes, never ``CommitUnconfirmed``, which says the work may have
landed, and the answer does not count as a degradation. Each test runs a real
coordinator in this process with the route taken out of its routing table, so
the 404 is its router's own.
"""

from __future__ import annotations

import io
import urllib.error
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

import ccs.adapters.claude_code.coordinator_server as coordinator_server_module
import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable
from ccs.core.exceptions import CoherenceDegradedWarning, CoherenceError, CommitUnconfirmed
from tests.coordinator_helpers import recorded_version, serve_in_process

_MANAGED = ("data/**",)
_MEMBERS = ("data/a.txt", "data/b.txt")
#: FROZEN duplicate of what a request to a route the coordinator does not serve raises.
_NOT_SERVED = (
    "the coordinator does not serve {route} (HTTP 404), so nothing ran and a retry "
    "gets the same answer. The Node coordinator has no snapshot sessions or handoffs."
)


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``data/**`` running in this process, with every
    member seeded ``old``."""
    (tmp_path / "data").mkdir()
    for rel in _MEMBERS:
        (tmp_path / rel).write_bytes(b"old")
    running = serve_in_process(tmp_path, "route-not-served")
    yield running
    running.shutdown()


def _unserve(monkeypatch: pytest.MonkeyPatch, route: str) -> None:
    """Take ``route`` out of the coordinator's routing table, so its router
    answers it 404, as for any route the coordinator does not serve."""
    monkeypatch.delitem(coordinator_server_module._ROUTES, ("POST", route))


def _volume(root: Path, cfg: LifecycleConfig, on_error: str) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, on_error=on_error, config=cfg)  # type: ignore[arg-type]


def _read_members(volume: CoherentVolume) -> dict[str, int]:
    return {rel: volume.read_with_version(rel)[1] for rel in _MEMBERS}


def _publish_new(volume: CoherentVolume, versions: dict[str, int]) -> None:
    volume.atomic_publish([(rel, version, b"new") for rel, version in versions.items()])


@pytest.mark.parametrize("route", ["/session/begin", "/session/commit_all"])
@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_publish_on_a_coordinator_without_snapshot_sessions_names_the_route(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch, on_error: str, route: str,
) -> None:
    """A publish of two files opens a snapshot session and commits through it.
    A coordinator that serves neither route answers 404, so the publish raises
    ``CoherenceError`` naming the route, never ``CommitUnconfirmed``; nothing
    is published or written, the volume stays attached and is not counted
    degraded, and a retry raises the same.

    Prevents a degrade caller being told a publish may have landed, and to
    retry it, when it cannot land on that coordinator, and the answer marking
    the volume degraded for good."""
    volume = _volume(tmp_path, fast_cfg, on_error)
    versions = _read_members(volume)
    _unserve(monkeypatch, route)

    for _attempt in range(2):
        with pytest.raises(CoherenceError) as raised:
            _publish_new(volume, versions)
        assert type(raised.value) is CoherenceError, repr(raised.value)
        assert str(raised.value) == _NOT_SERVED.format(route=route)

    assert {rel: recorded_version(server, rel) for rel in _MEMBERS} == versions, "a member was published"
    assert [(tmp_path / rel).read_bytes() for rel in _MEMBERS] == [b"old", b"old"], "a member was written"
    assert volume.is_attached and not volume.is_degraded and volume.degradation_count == 0


@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_handoff_on_a_coordinator_without_handoffs_names_the_route(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch, on_error: str, verb: str,
) -> None:
    """A coordinator without handoffs answers each handoff route 404, so every
    verb raises ``CoherenceError`` naming its route and never says the handoff
    may have landed, in both modes, and the volume is not counted degraded.

    Prevents a caller told to look for a handoff that cannot have happened."""
    giver, successor = _volume(tmp_path, fast_cfg, on_error), _volume(tmp_path, fast_cfg, on_error)
    giver.read(_MEMBERS[0])
    _unserve(monkeypatch, f"/handoff/{verb}")

    with pytest.raises(CoherenceError) as raised:
        if verb == "transfer":
            giver.transfer(_MEMBERS[0], successor=str(session_to_agent_id(successor.session_id)))
        else:
            getattr(giver, verb)(_MEMBERS[0])

    assert type(raised.value) is CoherenceError, repr(raised.value)
    assert str(raised.value) == _NOT_SERVED.format(route=f"/handoff/{verb}")
    assert "landed" not in str(raised.value)
    assert giver.is_attached and not giver.is_degraded


def _fail_the_commit(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """Deliver the batch commit to the coordinator, then fail its answer:
    ``http-503`` answers it 503, ``dropped`` loses it as a dropped connection
    does. Every other request goes through as sent."""
    real_post = coherent_volume_module._coordinator_post

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        answer = real_post(endpoint, path, payload, **kwargs)
        if path != "/session/commit_all":
            return answer
        if failure == "dropped":
            raise CoordinatorUnavailable("simulated: the connection dropped before the answer")
        raise urllib.error.HTTPError(path, 503, "scripted", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", post)


@pytest.mark.parametrize("failure", ["http-503", "dropped"])
@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_batch_commit_with_no_answer_is_still_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch, on_error: str, failure: str,
) -> None:
    """Control: only a 404 names a route the coordinator does not serve. A
    batch commit the coordinator ran and answered 503, or whose connection
    dropped before the answer, may have landed, so the publish raises
    ``CommitUnconfirmed`` as before and writes no member."""
    volume = _volume(tmp_path, fast_cfg, on_error)
    versions = _read_members(volume)
    _fail_the_commit(monkeypatch, failure)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", CoherenceDegradedWarning)
        with pytest.raises(CommitUnconfirmed):
            _publish_new(volume, versions)

    assert [(tmp_path / rel).read_bytes() for rel in _MEMBERS] == [b"old", b"old"], "a member was written"


def test_a_single_file_publish_needs_no_snapshot_session(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: a publish of one file takes the compare-and-swap path, so a
    coordinator without snapshot sessions still lands it, as before."""
    for route in ("/session/begin", "/session/commit_all"):
        _unserve(monkeypatch, route)
    volume = _volume(tmp_path, fast_cfg, "strict")
    rel = _MEMBERS[0]
    version = volume.read_with_version(rel)[1]

    assert volume.atomic_publish([(rel, version, b"new")]) == {rel: version + 1}
    assert (tmp_path / rel).read_bytes() == b"new"
    assert recorded_version(server, rel) == version + 1


def test_a_404_on_any_other_route_keeps_on_errors_path(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: only the snapshot-session and handoff routes name themselves.
    A degrade volume whose read is answered 404 warns, counts a degradation
    and returns the bytes on disk, as before."""
    volume = _volume(tmp_path, fast_cfg, "degrade")
    _unserve(monkeypatch, "/hooks/pre-read")

    with pytest.warns(CoherenceDegradedWarning, match="HTTP 404"):
        assert volume.read(_MEMBERS[0]) == b"old"

    assert volume.is_degraded and volume.degradation_count == 1
