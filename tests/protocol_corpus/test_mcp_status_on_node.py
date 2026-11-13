# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``swg_status``'s write holders against the real Node coordinator.

The Node coordinator lists each write holder in its default ``/status`` tier
under the 32-hex id its refusals name; the Python coordinator lists it
hyphenated, which the tests under ``tests/mcp`` cover. Each test here starts
the Node coordinator with the harness, on a workspace whose policy files hold
the volume's glob, and builds the ``swg_status`` payload from real volumes
there; nothing in the volume or on the wire to Node is stubbed, except one
release that must not arrive, to leave an earlier attempt's grant standing.

Only ``ccs.mcp.status`` and ``ccs.mcp.session`` are imported from the server
package, never the server or deny modules: the job that builds the plugin
installs no MCP SDK.

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``, in the
job that builds the paired plugin branch. Without a resolvable plugin dist
every test xfails, as the Node rows of the parity corpora do."""

from __future__ import annotations

import io
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import UUID

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import CoordinatorFailed, OtherHolder
from ccs.mcp.session import SessionConfig
from ccs.mcp.status import build_status, status_text
from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    apply_setup,
    coordinator_running,
    resolve_node_dist_path,
)

pytestmark = pytest.mark.protocol_corpus

_NODE_DIST_PATH = resolve_node_dist_path()
_MANAGED = ("data/**",)
_MINE = "data/mine.md"
_STRANDED = "data/stranded.md"
_THEIRS = "data/theirs.md"
_SESSION_STOP = "/hooks/session-stop"
_FAST = LifecycleConfig(
    idle_shutdown_sec=0,
    port_file_retry_attempts=20,
    port_file_retry_interval_sec=0.05,
    connect_retry_attempts=10,
    connect_retry_interval_sec=0.05,
)

VolumeFactory = Callable[..., CoherentVolume]


@pytest.fixture
def node_root(tmp_path: Path) -> Iterator[Path]:
    """The Node coordinator running on ``tmp_path``, with ``data/**`` in its
    tracked and strict policy files."""
    if _NODE_DIST_PATH is None:
        pytest.xfail(
            "Node coordinator dist not resolvable. Build the plugin checkout "
            "(npm ci && npm run build) or set AGENT_COHERENCE_PLUGIN_DIST_PATH."
        )
    apply_setup(tmp_path, {"tracked": list(_MANAGED), "strict_mode": list(_MANAGED)})
    (tmp_path / "data").mkdir()
    with coordinator_running(BACKEND_NODE, tmp_path, _NODE_DIST_PATH):
        yield tmp_path


@pytest.fixture
def open_volume(node_root: Path) -> Iterator[VolumeFactory]:
    """Build volumes on the Node workspace; each is closed before the
    coordinator stops."""
    built: list[CoherentVolume] = []

    def build(on_held: str = "displace") -> CoherentVolume:
        volume = CoherentVolume(node_root, managed=_MANAGED, config=_FAST, on_held=on_held)
        built.append(volume)
        return volume

    yield build
    for volume in reversed(built):
        volume.close()


def _hex(agent_id: str) -> str:
    """An agent id in the spelling a refusal names a holder by."""
    return UUID(agent_id).hex


def _strand_its_grants(volume: CoherentVolume, monkeypatch: pytest.MonkeyPatch) -> str:
    """Start a new attempt with a ``reacquire()`` whose release never reaches
    the coordinator, so every grant of the attempt before it stands. Returns
    that earlier attempt's id."""
    earlier = volume.agent_id
    real_post = coherent_volume_module._coordinator_post

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path == _SESSION_STOP:
            raise urllib.error.HTTPError(path, 503, "scripted", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", post)
    with pytest.raises(CoordinatorFailed):
        volume.reacquire(_STRANDED)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)
    assert volume.agent_id != earlier, "control: the reacquire started a new attempt"
    return earlier


def _node_write_holders(volume: CoherentVolume, rel: str) -> set[str]:
    """Every id Node's ``/status`` lists holding ``rel`` for writing, as listed."""
    status = volume.coordinator_status()
    assert status is not None and status["backend"] == "node", status
    return {
        session["agent_id"] for session in status["sessions"]
        if session["states"].get(rel) in ("EXCLUSIVE", "MODIFIED")
    }


def test_status_names_this_attempt_an_earlier_one_and_another_agent_on_node(
    open_volume: VolumeFactory, node_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On Node, a path this attempt wrote, one an earlier attempt still holds
    and one another volume holds each name their holder in 32 hex, the other
    volume's byte for byte as the refusal named it, and only this session's
    two are held.

    Prevents an agent that cannot match a Node refusal's holder to a status
    entry, or that waits on its own earlier grant there."""
    volume, peer = open_volume(on_held="refuse"), open_volume()
    volume.write(_STRANDED, b"first")
    earlier = _strand_its_grants(volume, monkeypatch)
    volume.write(_MINE, b"mine")
    peer.write(_THEIRS, b"theirs")
    with pytest.raises(OtherHolder) as refused:
        volume.write(_THEIRS, b"mine")

    status = build_status(volume, SessionConfig(root=node_root, managed=_MANAGED, on_held="refuse"))

    holders = {
        path: (entry.get("holder_agent_id"), entry.get("held_by")) for path, entry in status["per_path"].items()
    }
    assert holders == {
        _MINE: (_hex(volume.agent_id), "this_attempt"),
        _STRANDED: (_hex(earlier), "earlier_attempt"),
        _THEIRS: (refused.value.holder_agent_id, "another_agent"),
    }
    assert _node_write_holders(volume, _THEIRS) == {refused.value.holder_agent_id}, "Node lists 32 hex"
    assert status["on_held"] == "refuse"
    assert status["held"] == [_MINE, _STRANDED]
    assert status_text(status).splitlines()[1:] == [
        'on_held=refuse held=["data/mine.md", "data/stranded.md"]',
        f'path="data/stranded.md" held_by=earlier_attempt holder_agent_id={_hex(earlier)}',
        f'path="data/theirs.md" held_by=another_agent holder_agent_id={refused.value.holder_agent_id}',
    ]


def test_a_release_of_an_earlier_attempts_path_clears_its_grant_on_node(
    open_volume: VolumeFactory, node_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``release()`` of a path an earlier attempt still holds gives that grant
    back on Node: it answers ``True``, Node lists no write holder for the path,
    and the status entry has no holder and is no longer held.

    Prevents telling an agent to release an earlier attempt's path on Node,
    where only a reacquire would clear it."""
    volume = open_volume(on_held="refuse")
    volume.write(_STRANDED, b"first")
    earlier = _strand_its_grants(volume, monkeypatch)
    assert _node_write_holders(volume, _STRANDED) == {_hex(earlier)}, "control: the earlier grant stands"

    assert volume.release(_STRANDED) is True

    assert _node_write_holders(volume, _STRANDED) == set()
    status = build_status(volume, SessionConfig(root=node_root, managed=_MANAGED, on_held="refuse"))
    assert sorted(status["per_path"][_STRANDED]) == ["status", "version"], "no holder keys"
    assert status["held"] == []
