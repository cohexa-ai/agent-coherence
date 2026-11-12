# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A real ``CoherentVolume`` against the real Node coordinator (#518).

Every other volume test drives the Python coordinator, so nothing else shows
that a volume attaches to the plugin's Node coordinator and is coordinated
there. Each test here starts that coordinator with the harness, on a workspace
whose ``tracked.yaml`` and ``strict_mode.yaml`` hold the volume's glob, as a
volume needs it (a volume never starts the Node coordinator), and builds the
volumes on that workspace; nothing in the volume or on the wire is stubbed.

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``, in the
job that builds the paired plugin branch. Without a resolvable plugin dist
every test xfails, as the Node rows of the parity corpora do."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import UUID

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume, ManagedGlobEnforcement
from ccs.core.exceptions import CoherenceError, OtherHolder
from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    apply_setup,
    coordinator_running,
    resolve_node_dist_path,
    workspace_fold_profile,
)

pytestmark = pytest.mark.protocol_corpus

_NODE_DIST_PATH = resolve_node_dist_path()
_MANAGED = ("data/**",)
_HELD = "data/x.md"
_MEMBERS = ("data/a.md", "data/b.md")
#: FROZEN duplicate of what a request to a route the coordinator does not serve raises.
_NOT_SERVED = (
    "the coordinator does not serve {route} (HTTP 404), so nothing ran and a retry "
    "gets the same answer. The Node coordinator has no snapshot sessions or handoffs."
)
_FAST = LifecycleConfig(
    idle_shutdown_sec=0,
    port_file_retry_attempts=20,
    port_file_retry_interval_sec=0.05,
    connect_retry_attempts=10,
    connect_retry_interval_sec=0.05,
)

VolumeFactory = Callable[..., CoherentVolume]


@pytest.fixture
def disk_profile(tmp_path: Path) -> str:
    """What the disk under this test's workspace folds, classified before the
    Node coordinator opens the workspace, as every fold row is."""
    return workspace_fold_profile(tmp_path)


@pytest.fixture
def node_root(tmp_path: Path, disk_profile: str) -> Iterator[Path]:
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

    def build(on_held: str = "displace", on_error: str = "strict") -> CoherentVolume:
        volume = CoherentVolume(node_root, managed=_MANAGED, config=_FAST, on_held=on_held, on_error=on_error)
        built.append(volume)
        return volume

    yield build
    for volume in reversed(built):
        volume.close()


def _write_holders(volume: CoherentVolume, rel: str) -> set[UUID]:
    """Every session the coordinator's ``/status`` shows holding ``rel`` for writing."""
    status = volume.coordinator_status()
    assert status is not None and status["backend"] == "node", status
    return {
        UUID(session["agent_id"]) for session in status["sessions"]
        if session["states"].get(rel) in ("EXCLUSIVE", "MODIFIED")
    }


def _recorded_version(volume: CoherentVolume, rel: str) -> int:
    """The version the coordinator's ``/status`` lists for ``rel``."""
    status = volume.coordinator_status()
    assert status is not None and status["backend"] == "node", status
    (version,) = [a["version"] for a in status["tracked_artifacts"] if a["path"] == rel]
    return version


@pytest.mark.parametrize(
    ("on_held", "on_error"),
    [("refuse", "strict"), ("displace", "strict"), ("displace", "degrade")],
)
def test_a_volume_attaches_to_node_with_its_glob_enforced(
    open_volume: VolumeFactory, on_held: str, on_error: str
) -> None:
    """A volume with ``managed=("data/**",)`` attaches to the Node coordinator
    and its glob check reads the glob as enforced, from Node's operator view.

    Prevents a volume that cannot confirm its globs on Node: one that raises at
    construction, or runs detached with coherence off in degrade mode."""
    volume = open_volume(on_held=on_held, on_error=on_error)

    assert volume.is_attached and not volume.is_degraded
    assert volume.coordinator_status()["backend"] == "node"
    assert volume.managed_glob_enforcement() == ManagedGlobEnforcement(_MANAGED, (), None)


def test_a_refuse_volume_is_refused_a_held_file_and_lands_after_the_release(
    open_volume: VolumeFactory, node_root: Path
) -> None:
    """While another volume holds ``data/x.md``, a refuse-mode volume's write
    raises ``OtherHolder`` naming the holder, and the holder keeps the file
    and its bytes. Once the holder releases it, the same write lands.

    Prevents a refuse-mode write on Node that takes a file from its holder, or
    one refused before it is sent because Node issues no caller principal."""
    holder, refused = open_volume(), open_volume(on_held="refuse")
    holder.write(_HELD, b"from-the-holder")

    with pytest.raises(OtherHolder) as raised:
        refused.write(_HELD, b"from-the-refused")

    assert UUID(raised.value.holder_agent_id) == UUID(holder.agent_id)
    assert (node_root / _HELD).read_bytes() == b"from-the-holder"
    assert _write_holders(holder, _HELD) == {UUID(holder.agent_id)}

    holder.release(_HELD)
    refused.write(_HELD, b"from-the-refused")
    assert (node_root / _HELD).read_bytes() == b"from-the-refused"
    assert _write_holders(refused, _HELD) == {UUID(refused.agent_id)}


def test_a_first_read_gives_the_version_a_cas_write_lands_on(
    open_volume: VolumeFactory, node_root: Path
) -> None:
    """A volume's first read of a file the coordinator has not seen reports
    the file's version as the coordinator records it, and ``write_cas_at``
    with that version lands.

    Prevents a version-checked write on Node that is refused as a conflict
    because the read reported no version, or one that is not the file's."""
    (node_root / "data/new.md").write_bytes(b"first")
    volume = open_volume()

    content, version = volume.read_with_version("data/new.md")
    assert content == b"first"
    assert version == _recorded_version(volume, "data/new.md")

    result = volume.write_cas_at("data/new.md", version, b"second")
    assert result.version == version + 1 == _recorded_version(volume, "data/new.md")
    assert (node_root / "data/new.md").read_bytes() == b"second"


def test_a_two_file_publish_names_the_route_and_writes_nothing(
    open_volume: VolumeFactory, node_root: Path
) -> None:
    """A degrade volume's publish of two files raises ``CoherenceError`` naming
    ``/session/begin``, which the Node coordinator does not serve; nothing is
    written, and a retry raises the same.

    Prevents a degrade caller told its publish may have landed, and to retry
    it, when it cannot land on Node."""
    for rel in _MEMBERS:
        (node_root / rel).write_bytes(b"old")
    volume = open_volume(on_error="degrade")
    writes = [(rel, volume.read_with_version(rel)[1], b"new") for rel in _MEMBERS]

    for _attempt in range(2):
        with pytest.raises(CoherenceError) as raised:
            volume.atomic_publish(writes)
        assert type(raised.value) is CoherenceError, repr(raised.value)
        assert str(raised.value) == _NOT_SERVED.format(route="/session/begin")

    assert [(node_root / rel).read_bytes() for rel in _MEMBERS] == [b"old", b"old"]
    assert volume.is_attached and not volume.is_degraded


def test_another_case_of_a_held_file_is_refused_where_the_disk_folds_case(
    open_volume: VolumeFactory, node_root: Path, disk_profile: str
) -> None:
    """On a case-insensitive disk, a refuse-mode write of ``DATA/X.md`` while
    another volume holds ``data/x.md`` is refused like a write of its disk
    spelling: the volume and the Node coordinator both read it as under
    ``data/**`` and as that one held file.

    Prevents a second spelling that one side reads as outside the managed glob,
    and so writes over the holder unchecked."""
    if disk_profile != "caseless":
        pytest.skip(
            f"the disk under this workspace is not case-insensitive (fold profile {disk_profile}), "
            "so DATA/X.md is not shown to be another spelling of data/x.md"
        )
    holder, refused = open_volume(), open_volume(on_held="refuse")
    holder.write(_HELD, b"from-the-holder")

    with pytest.raises(OtherHolder) as raised:
        refused.write("DATA/X.md", b"from-the-refused")

    assert UUID(raised.value.holder_agent_id) == UUID(holder.agent_id)
    assert (node_root / _HELD).read_bytes() == b"from-the-holder"
    assert _write_holders(holder, _HELD) == {UUID(holder.agent_id)}
