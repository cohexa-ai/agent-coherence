"""Unit 2 (R0b): CoherentVolume connect-only remote mode — never spawns.

The plan's C1 fix: a remote client must attach to the supplied endpoint and
NEVER fall through to ``connect_or_spawn`` (which would spawn a second, local
coordinator and silently split peers onto different coordinators).
"""

from __future__ import annotations

import errno
import socket
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import resolve_endpoint, resolve_remote_endpoint
from ccs.core.exceptions import CoherenceError


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


def _refusing_port() -> int:
    """A loopback port that refuses connections and that no other test can take.

    A port freed after ``bind(("127.0.0.1", 0))`` can be handed to another
    test worker's port-0 bind before the connect, and the connect then reaches
    a stranger. Ports below 1024 are never handed out for port 0, and port 4
    is unassigned, so no standard service listens on it; the assert says so
    if something does. A socket bound without listening is no substitute: on
    macOS a connect to it hangs until its timeout instead of being refused."""
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", 4)) == errno.ECONNREFUSED, "port 4 accepts connections here"
    return 4


def test_remote_mode_connects_without_spawning(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1 happy path: a remote volume attaches to an existing coordinator and
    does NOT spawn its own (no server.pid under its root)."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()

    # root A spawns the coordinator (stands in for the "remote" one).
    vol_a = CoherentVolume(root_a, config=fast_cfg)
    try:
        ep_a = resolve_endpoint(root_a)
        remote = resolve_remote_endpoint("127.0.0.1", ep_a.port, ep_a.bearer)
        vol_b = CoherentVolume(
            root_b, config=fast_cfg, on_error="strict", remote_endpoint=remote
        )
        assert vol_b.is_attached
        assert vol_b._endpoint is not None and vol_b._endpoint.port == ep_a.port
        # The C1 invariant: root B never spawned its own coordinator.
        assert not (root_b / ".coherence" / "server.pid").exists()
    finally:
        stop_coordinator(root_a)


def test_remote_unreachable_fails_closed_no_spawn(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """C1 unreachable: a dead remote endpoint fails CLOSED (strict raises) and
    still does NOT fall through to spawning a local coordinator."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    remote = resolve_remote_endpoint("127.0.0.1", _refusing_port(), "deadbeef")
    with pytest.raises(CoherenceError):
        CoherentVolume(tmp_path, config=fast_cfg, on_error="strict", remote_endpoint=remote)
    assert not (tmp_path / ".coherence" / "server.pid").exists()


def test_remote_endpoint_requires_flag(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CCS_REMOTE_COORDINATOR", raising=False)
    remote = resolve_remote_endpoint("127.0.0.1", 8080, "deadbeef")
    with pytest.raises(ValueError, match="CCS_REMOTE_COORDINATOR"):
        CoherentVolume(tmp_path, config=fast_cfg, on_error="strict", remote_endpoint=remote)


def test_remote_mode_rejects_degrade(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    remote = resolve_remote_endpoint("127.0.0.1", 8080, "deadbeef")
    with pytest.raises(ValueError, match="strict-only"):
        CoherentVolume(tmp_path, config=fast_cfg, on_error="degrade", remote_endpoint=remote)


def test_remote_mode_rejects_managed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    remote = resolve_remote_endpoint("127.0.0.1", 8080, "deadbeef")
    with pytest.raises(ValueError, match="version-only|managed"):
        CoherentVolume(
            tmp_path,
            config=fast_cfg,
            on_error="strict",
            managed=("x/**",),
            remote_endpoint=remote,
        )
