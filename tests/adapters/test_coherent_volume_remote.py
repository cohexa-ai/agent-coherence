"""Unit 2 (R0b): CoherentVolume connect-only remote mode — never spawns.

The plan's C1 fix: a remote client must attach to the supplied endpoint and
NEVER fall through to ``connect_or_spawn`` (which would spawn a second, local
coordinator and silently split peers onto different coordinators).
"""

from __future__ import annotations

from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorEndpoint, resolve_endpoint, resolve_remote_endpoint
from ccs.core.exceptions import CoordinatorUnreachable
from tests.loopback_helpers import refused_endpoint, refusing_port


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
    """C1 unreachable: a dead remote endpoint fails CLOSED (strict raises the
    typed ``CoordinatorUnreachable``, AE5) and still does NOT fall through to
    spawning a local coordinator.

    Does not prove: anything off loopback, or a partition."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    remote = resolve_remote_endpoint("127.0.0.1", refusing_port(), "deadbeef")
    with pytest.raises(CoordinatorUnreachable, match="remote coordinator unreachable"):
        CoherentVolume(tmp_path, config=fast_cfg, on_error="strict", remote_endpoint=remote)
    assert not (tmp_path / ".coherence" / "server.pid").exists()


def test_remote_claim_with_no_answer_is_unreachable(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AE5, remote attach: the coordinator answers the attach-time ``/status``
    but the caller-principal claim goes to a port that refuses, so the real
    client reports it got no answer. The strict attach raises
    ``CoordinatorUnreachable``, not the plain error an answered claim keeps.

    Does not prove: anything off loopback, or a partition."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()
    vol_a = CoherentVolume(root_a, config=fast_cfg)  # spawns the "remote" coordinator
    real_claim = coherent_volume_module.claim_caller_principal

    def claim_refused(endpoint: CoordinatorEndpoint, session_id: str, nonce: str) -> object:
        return real_claim(refused_endpoint(endpoint), session_id, nonce)

    monkeypatch.setattr(coherent_volume_module, "claim_caller_principal", claim_refused)
    try:
        ep_a = resolve_endpoint(root_a)
        remote = resolve_remote_endpoint("127.0.0.1", ep_a.port, ep_a.bearer)
        with pytest.raises(CoordinatorUnreachable, match="caller principal not obtained"):
            CoherentVolume(root_b, config=fast_cfg, on_error="strict", remote_endpoint=remote)
        assert vol_a.is_attached, "control: the coordinator itself is up"
        assert not (root_b / ".coherence" / "server.pid").exists()
    finally:
        stop_coordinator(root_a)


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
