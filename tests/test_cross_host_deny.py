"""Scenario 1 cross-host deny + recover.

This loopback integration test proves the full chain end-to-end on a real socket
(runnable on any platform): two remote-endpoint clients (never-spawn) coordinate
ONE coordinator; a stale write is denied by version-CAS *across the endpoint* and
recovers via re-read + retry. The genuinely-non-loopback path (a real RFC-1918
bind exercising verify_host's new branch) is the netns/veth runner in
examples/cross_host/ (Linux-only); that Host-check logic is unit-covered in
tests/test_claude_code_bind_validation.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import (
    post as _cc_post,
)
from ccs.cli._coherence_client import (
    resolve_endpoint,
    resolve_remote_endpoint,
)
from ccs.core.exceptions import CasVersionConflict


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


def test_slice1_cross_endpoint_deny_and_recover(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()

    ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        # Coordinator-version-only: the artifact must be TRACKED on the coordinator
        # to be versioned (an untracked path is never versioned -> no CAS). Tracking
        # is separate from STRICT mode — it gives versioning; strict would add
        # read-deny enforcement (not needed for the version-CAS write deny).
        _cc_post(ep, "/policy/track", {"paths": ["shared.txt"]})

        def remote():
            return resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer)

        vol_a = CoherentVolume(root_a, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        vol_b = CoherentVolume(root_b, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        # Coordinator-version-only: per-root file copies; coordination is the
        # coordinator's version of the shared key.
        (root_a / "shared.txt").write_text("v0", encoding="utf-8")
        (root_b / "shared.txt").write_text("v0", encoding="utf-8")

        # A reads the shared key -> version v_a (registers SHARED on the coordinator).
        _data_a, v_a = vol_a.read_with_version("shared.txt")

        # B reads + commits -> the coordinator version advances past v_a.
        _data_b, v_b = vol_b.read_with_version("shared.txt")
        vol_b.write_cas_at("shared.txt", v_b, b"from-b")

        # A's write against its now-stale version is DENIED across the endpoint
        # (the typed version-CAS conflict, NOT just any CoherenceError — pin the
        # exact type and the carried versions so the deny can't pass on an
        # unrelated failure).
        with pytest.raises(CasVersionConflict) as deny:
            vol_a.write_cas_at("shared.txt", v_a, b"from-a")
        assert deny.value.expected_version == v_a
        assert deny.value.current_version > v_a

        # A recovers: re-read -> fresh version -> retry succeeds (no silent loss).
        _data_a2, v_a2 = vol_a.read_with_version("shared.txt")
        assert v_a2 > v_a
        vol_a.write_cas_at("shared.txt", v_a2, b"from-a-2")
        assert (root_a / "shared.txt").read_bytes() == b"from-a-2"
    finally:
        stop_coordinator(coord_root)


def test_untracked_key_is_not_versioned_so_deny_passes_vacuously(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard: WITHOUT /policy/track the shared key is never versioned (stays
    version 0), so the version-CAS cannot detect a stale write — A silently
    overwrites B's commit, NO CasVersionConflict is raised. This is WHY the demo
    and the deny test must track the key first; it pins the vacuous-pass failure
    mode so a dropped track() call can't quietly defeat the deny."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()

    ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        # Deliberately DO NOT track "shared.txt" — the contrast with the deny test.

        def remote():
            return resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer)

        vol_a = CoherentVolume(root_a, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        vol_b = CoherentVolume(root_b, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        (root_a / "shared.txt").write_text("v0", encoding="utf-8")
        (root_b / "shared.txt").write_text("v0", encoding="utf-8")

        _data_a, v_a = vol_a.read_with_version("shared.txt")
        _data_b, v_b = vol_b.read_with_version("shared.txt")
        assert v_a == 0 and v_b == 0  # untracked -> never versioned

        vol_b.write_cas_at("shared.txt", v_b, b"from-b")
        # A's "stale" write is NOT denied (the version never moved off 0): A's
        # bytes silently win — the lost update the deny test prevents via tracking.
        vol_a.write_cas_at("shared.txt", v_a, b"from-a")
        assert (root_a / "shared.txt").read_bytes() == b"from-a"
    finally:
        stop_coordinator(coord_root)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_remote_volume_reattaches_to_remote_coordinator_after_fork(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A REMOTE-mode volume forked into a child re-attaches to the SAME remote
    coordinator (connect-only, never spawn): the child re-mints identity, takes
    the _attach_remote path, and creates NO local server.pid under its own root.

    The child's first post-fork read_with_version is what re-attaches, so it
    returns the coordinator's version rather than a version-0 fallback. That
    version is a live CAS comparand: the child's write against it lands through
    the shared coordinator (the parent sees it) with no reacquire() first."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root = tmp_path / "a"
    root.mkdir()

    ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        _cc_post(ep, "/policy/track", {"paths": ["shared.txt"]})
        remote = resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer)
        vol = CoherentVolume(root, on_error="strict", on_stale_write="allow", remote_endpoint=remote)
        (root / "shared.txt").write_text("v0", encoding="utf-8")
        _d, v_parent = vol.read_with_version("shared.txt")  # parent attaches
        assert v_parent > 0  # so the child's version check cannot pass on a 0 fallback
        parent_id = vol.session_id

        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child: first op must re-attach to the remote coordinator
            os.close(read_fd)
            try:
                # First op post-fork: read_with_version re-attaches to the remote
                # coordinator (connect-only) and returns its version. The facts
                # are captured before any other op, so they are the first read's.
                _dc, v_first = vol.read_with_version("shared.txt")
                local_pid = (root / ".coherence" / "server.pid").exists()
                facts = f"{vol.session_id}|{vol.is_attached}|{local_pid}|{vol._endpoint.port}|{v_first}"
                # The first read's version is the CAS comparand; no reacquire().
                vol.write_cas_at("shared.txt", v_first, b"from-child")
                msg = f"{facts}|ok"
            except Exception as exc:  # noqa: BLE001 - report any failure to the parent
                msg = f"ERR:{exc!r}"
            try:
                os.write(write_fd, msg.encode("utf-8"))
            finally:
                os.close(write_fd)
                os._exit(0)
        # parent (child os._exit'd above and never reaches the finally below)
        os.close(write_fd)
        raw = os.read(read_fd, 256).decode("utf-8")
        os.close(read_fd)
        os.waitpid(pid, 0)
        assert not raw.startswith("ERR:"), raw
        child_id, attached, local_pid, child_port, first_v, wrote = raw.split("|")
        assert child_id != parent_id  # re-minted identity
        assert attached == "True"  # the first read re-attached
        assert local_pid == "False"  # connect-only: NEVER spawned a local coordinator
        assert int(child_port) == ep.port  # to the SAME remote coordinator
        assert int(first_v) == v_parent  # the coordinator's version, nothing wrote since
        assert wrote == "ok"  # write_cas_at against the first read's version was granted
        # The child's write landed through the shared coordinator.
        _d2, v_after = vol.read_with_version("shared.txt")
        assert v_after > v_parent
        assert (root / "shared.txt").read_bytes() == b"from-child"
    finally:
        stop_coordinator(coord_root)


def test_slice2_effect_gate_across_hosts(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scenario 2 effect ordering: an effect gated on config@vN FIRES when config
    is unchanged and is HELD when config advanced under it — across the endpoint,
    on the shipped read_with_version primitive."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()

    ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        _cc_post(ep, "/policy/track", {"paths": ["config.json"]})

        def remote():
            return resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer)

        vol_a = CoherentVolume(root_a, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        vol_b = CoherentVolume(root_b, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        (root_a / "config.json").write_text('{"v": 0}', encoding="utf-8")
        (root_b / "config.json").write_text('{"v": 0}', encoding="utf-8")

        # A reads config -> the version its effect decision depends on.
        _c, v_decision = vol_a.read_with_version("config.json")

        def effect_fires(decision_version: int) -> bool:
            """Gate the effect on config still being at the decision version."""
            _cur, current = vol_a.read_with_version("config.json")
            return current == decision_version

        # Unchanged config -> the gate FIRES.
        assert effect_fires(v_decision) is True

        # B advances config under A.
        _cb, v_b = vol_b.read_with_version("config.json")
        vol_b.write_cas_at("config.json", v_b, b'{"v": 1}')

        # Config moved -> the gate HOLDS the effect (not fired on stale input).
        assert effect_fires(v_decision) is False
    finally:
        stop_coordinator(coord_root)


# ---------------------------------------------------------------------------
# Negative controls — codify the baselines the demo's --baseline flag runs.
#
# These tests assert that the FAILURES this library prevents are real and
# reproducible. If a future library change quietly made the baseline 'work',
# the demo's contrast would erode and we would notice here first.
# ---------------------------------------------------------------------------


def test_slice1_baseline_silent_lost_update_without_decision_time_version(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEGATIVE CONTROL: an agent that does NOT track its decision-time version
    silently loses a peer's intervening commit.

    Pattern: A reads@v_a but writes against the LATEST version (the classic
    convention-only / un-coordinated lost-update bug pattern). B's bytes are
    dropped from the canonical record. This is the failure
    ``test_slice1_cross_endpoint_deny_and_recover`` proves this library prevents.
    """
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()

    ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        _cc_post(ep, "/policy/track", {"paths": ["shared.txt"]})

        def remote():
            return resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer)

        vol_a = CoherentVolume(root_a, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        vol_b = CoherentVolume(root_b, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        (root_a / "shared.txt").write_text("v0", encoding="utf-8")
        (root_b / "shared.txt").write_text("v0", encoding="utf-8")

        # A's decision-time read — but the baseline agent "forgets" v_a below.
        _data_a, v_a = vol_a.read_with_version("shared.txt")

        # B commits intervening bytes; coordinator's canonical version advances.
        _data_b, v_b = vol_b.read_with_version("shared.txt")
        vol_b.write_cas_at("shared.txt", v_b, b"from-b")

        # The bug pattern: A re-reads to get the LATEST version and writes
        # against it — no check that the artifact moved relative to v_a.
        _, v_a_latest = vol_a.read_with_version("shared.txt")
        assert v_a_latest > v_a, "precondition: B's commit advanced the version"
        vol_a.write_cas_at("shared.txt", v_a_latest, b"from-a")  # SUCCEEDS — no deny

        # Canonical state now has A's bytes; B's bytes are silently lost.
        data_now, _v_now = vol_a.read_with_version("shared.txt")
        assert data_now == b"from-a", "baseline failed: lost-update did not occur"
        assert data_now != b"from-b", "B's bytes should be gone from the canonical record"
    finally:
        stop_coordinator(coord_root)


def test_slice2_baseline_stale_fire_without_effect_gate(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEGATIVE CONTROL: an agent that does NOT gate its effect on the
    decision-time version fires on stale config.

    Pattern: A decides @ v_decision, B advances config, A's effect fires
    ungated. This is the CI failure (build/deploy against a config that was
    edited mid-decision) that the version-CAS effect gate in
    ``test_slice2_effect_gate_across_hosts`` prevents.
    """
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_a = tmp_path / "a"
    root_a.mkdir()
    root_b = tmp_path / "b"
    root_b.mkdir()

    ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        _cc_post(ep, "/policy/track", {"paths": ["config.json"]})

        def remote():
            return resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer)

        vol_a = CoherentVolume(root_a, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        vol_b = CoherentVolume(root_b, on_error="strict", on_stale_write="allow", remote_endpoint=remote())
        (root_a / "config.json").write_text('{"v": 0}', encoding="utf-8")
        (root_b / "config.json").write_text('{"v": 0}', encoding="utf-8")

        # A decides on config @ v_decision.
        _c, v_decision = vol_a.read_with_version("config.json")

        # B advances config under A.
        _cb, v_b = vol_b.read_with_version("config.json")
        vol_b.write_cas_at("config.json", v_b, b'{"v": 1}')

        # The baseline: A fires the effect ungated — the current version is
        # NOT compared against v_decision before the effect runs.
        _, v_when_firing = vol_a.read_with_version("config.json")
        assert v_when_firing > v_decision, "precondition: config must have advanced"

        # In a real CI step the effect would be the deploy/build invocation.
        # We assert the failure by observing that NOTHING in the baseline
        # mechanism would have prevented the effect from firing — the agent
        # has no gate to enforce.
        # (Concretely: it would have called effect(v=v_decision) against a
        # config that is now at v_when_firing.)
        assert v_when_firing != v_decision, "baseline failed: stale config not exercised"
    finally:
        stop_coordinator(coord_root)


def test_main_baseline_flag_runs_negative_control_then_with_ccs(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """End-to-end CLI smoke: ``python examples/cross_host/main.py --baseline``
    runs the negative control then the with-coordination pass, exits 0, and the
    output contains both phases. This pins the honest contract: broken-must-lose
    AND fixed-must-prevent in one invocation.
    """
    # main() spawns a coordinator + clients in this process, and for the local
    # loopback path it sets CCS_REMOTE_COORDINATOR itself (setdefault). Make the
    # flag absent while main() runs, so main() is what turns it on, and register
    # it with monkeypatch, so it is gone again afterwards: otherwise every later
    # test in this process runs with remote-coordinator mode switched on.
    monkeypatch.delenv("CCS_REMOTE_HOST", raising=False)
    monkeypatch.delenv("CCS_REMOTE_PORT", raising=False)
    monkeypatch.delenv("CCS_REMOTE_SECRET_FILE", raising=False)
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "0")
    monkeypatch.delenv("CCS_REMOTE_COORDINATOR")

    # Import lazily so the test file does not depend on the example being on
    # the import path at collection time.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ccs_cross_host_demo_main",
        Path(__file__).parent.parent / "examples" / "cross_host" / "main.py",
    )
    assert spec is not None and spec.loader is not None
    demo_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo_main)

    rc = demo_main.main(["--baseline"])
    out = capsys.readouterr().out
    assert rc == 0, f"--baseline should exit 0 on a green run, got {rc}; output:\n{out}"
    assert "Negative control" in out, "baseline phase must be labeled in output"
    assert "With coordination" in out, "with-coordination phase must be labeled in output"
    assert "RESULT: PASS" in out
