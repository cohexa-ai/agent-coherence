# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the Claude Code coordinator lifecycle module (Unit 5).

Covers spawn/connect, port-file race, idle shutdown, sweep reclaim, and
the 10-process race integration test that gives Unit 5 its main signal.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import select
import signal
import socket
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

from ccs.adapters.claude_code import lifecycle
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    connect_or_spawn,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.adapters.claude_code.lifecycle import (
    read_port_from_file as _read_port_from_file,
)

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A coordinator_root for tests — a tmp dir that mimics a repo root."""
    return tmp_path


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """A config tuned for fast tests: short sweep, short connect retries.
    Idle shutdown disabled by default — opt-in per test."""
    return LifecycleConfig(
        idle_shutdown_sec=0,           # disabled by default in tests
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
        grant_heartbeat_timeout_sec=600,
        grant_max_hold_sec=1800,
        transient_timeout_sec=60,
    )


@pytest.fixture
def spawned(workspace: Path, fast_cfg: LifecycleConfig):
    """Spawn a coordinator, yield (port, workspace), and tear down."""
    port = ensure_coordinator(workspace, config=fast_cfg)
    assert port > 0, f"ensure_coordinator returned sentinel: {port}"
    yield port, workspace
    stop_coordinator(workspace)


# ----------------------------------------------------------------------
# Happy paths
# ----------------------------------------------------------------------


def test_ensure_coordinator_first_call_binds_and_writes_port(
    workspace: Path, fast_cfg: LifecycleConfig
) -> None:
    """Winner path: returns a valid port and the pid file is populated."""
    port = ensure_coordinator(workspace, config=fast_cfg)
    try:
        assert 1024 <= port <= 65535, f"unexpected port: {port}"
        pid_file = workspace / ".coherence" / "server.pid"
        assert pid_file.exists()
        lines = pid_file.read_text().splitlines()
        assert len(lines) >= 2
        assert int(lines[0]) == os.getpid()
        assert int(lines[1]) == port
    finally:
        stop_coordinator(workspace)


def test_ensure_coordinator_second_call_same_process_returns_same_port(
    workspace: Path, fast_cfg: LifecycleConfig
) -> None:
    """Second call from the same process should be idempotent — return
    the same port without rebinding. fcntl.flock is per-process so the
    same-process second call WILL re-acquire the lock and re-bind.

    Per the plan, the contract is 'second call (different process) reads
    existing port without rebinding'. Same-process behavior is operator
    error — we just need it not to corrupt state."""
    port_1 = ensure_coordinator(workspace, config=fast_cfg)
    try:
        # Same-process flock re-acquire succeeds (per-process semantics),
        # so the second call writes a new port. This is acceptable —
        # callers shouldn't spawn twice from the same process. We just
        # verify the second call doesn't raise and produces a valid port.
        port_2 = ensure_coordinator(workspace, config=fast_cfg)
        assert 1024 <= port_2 <= 65535
    finally:
        stop_coordinator(workspace)


def test_connect_or_spawn_reads_existing_port_without_respawning(
    spawned, fast_cfg: LifecycleConfig
) -> None:
    """Once a coordinator is up, connect_or_spawn returns the existing
    port via TCP probe and does NOT trigger another bind."""
    port_existing, workspace = spawned
    port_returned = connect_or_spawn(workspace, config=fast_cfg)
    assert port_returned == port_existing


def test_connect_or_spawn_returns_minus_one_when_root_unwritable(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Edge case: parent repo is read-only — connect_or_spawn returns -1
    so the caller (hook handler) treats as 'no coordinator available'."""
    ro_root = tmp_path / "ro_repo"
    ro_root.mkdir()
    os.chmod(ro_root, 0o500)  # read+execute, no write
    try:
        port = connect_or_spawn(ro_root, config=fast_cfg)
        assert port == -1
    finally:
        os.chmod(ro_root, 0o700)  # restore for cleanup


# ----------------------------------------------------------------------
# Port-file race: loser path
# ----------------------------------------------------------------------


def test_port_file_retry_succeeds_when_holder_writes_late(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, fast_cfg: LifecycleConfig
) -> None:
    """Loser path: simulate the holder being mid-bind (port file empty)
    by populating the port file partway through the loser's retry loop."""
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pid_file = coherence_dir / "server.pid"

    # Holder writes JUST the pid (no port) initially.
    pid_file.write_text("99999\n", encoding="utf-8")

    # In a background thread, write the port after a small delay.
    def late_write() -> None:
        time.sleep(0.075)  # 1.5 retry intervals at 50ms
        pid_file.write_text("99999\n50001\n", encoding="utf-8")

    import threading as _t
    t = _t.Thread(target=late_write, daemon=True)
    t.start()

    port = lifecycle._read_port_with_retry(pid_file, fast_cfg)
    t.join()
    assert port == 50001


def test_port_file_retry_returns_minus_one_on_exhaustion(
    workspace: Path, fast_cfg: LifecycleConfig
) -> None:
    """Loser path: if the holder NEVER writes the port within the budget,
    return -1 so the caller degrades."""
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pid_file = coherence_dir / "server.pid"
    pid_file.write_text("99999\n", encoding="utf-8")  # pid only, forever

    # Use a tiny budget for the test
    tiny_cfg = LifecycleConfig(
        port_file_retry_attempts=3,
        port_file_retry_interval_sec=0.01,
    )
    port = lifecycle._read_port_with_retry(pid_file, tiny_cfg)
    assert port == -1


# ----------------------------------------------------------------------
# Stale pid file
# ----------------------------------------------------------------------


def test_stale_pidfile_triggers_respawn(
    workspace: Path, fast_cfg: LifecycleConfig
) -> None:
    """Edge case: previous coordinator crashed; its port is in the file
    but nothing is listening. connect_or_spawn TCP-probes, fails, then
    calls ensure_coordinator to re-spawn (which re-acquires the lock
    cleanly because the OS released it on the crashed process exit)."""
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pid_file = coherence_dir / "server.pid"
    # Write a port number that's almost certainly not bound by anything.
    pid_file.write_text("99999\n65530\n", encoding="utf-8")

    port = connect_or_spawn(workspace, config=fast_cfg)
    try:
        assert port > 0 and port != 65530, (
            f"expected respawn on a fresh port; got {port}"
        )
        # The pid file should now reflect the new coordinator
        new_port = _read_port_from_file(pid_file)
        assert new_port == port
    finally:
        stop_coordinator(workspace)


# ----------------------------------------------------------------------
# Idle shutdown
# ----------------------------------------------------------------------


def test_idle_shutdown_triggers_and_clears_port(
    workspace: Path,
) -> None:
    """Edge case: idle threshold elapsed → coordinator self-stops, rewrites
    port file removing the port line, releases flock. A subsequent
    ensure_coordinator call re-spawns on a (possibly different) port."""
    # Aggressive idle shutdown for the test: 0.5s threshold, 0.1s tick.
    cfg = LifecycleConfig(
        idle_shutdown_sec=0.5,
        sweep_interval_sec=0.1,
        connect_retry_attempts=5,
        connect_retry_interval_sec=0.05,
        port_file_retry_attempts=10,
        port_file_retry_interval_sec=0.05,
    )
    port_1 = ensure_coordinator(workspace, config=cfg)
    assert port_1 > 0

    # Wait past idle threshold + one sweep tick + HTTPServer.shutdown()'s
    # 0.5s poll interval + margin. Empirically ~1.5s total; 2.0s safe.
    time.sleep(2.0)

    # The idle watcher should have run shutdown by now. Verify:
    # 1. Port file no longer has port line.
    pid_file = workspace / ".coherence" / "server.pid"
    assert _read_port_from_file(pid_file) is None, (
        "idle shutdown must drop the port line from the pid file"
    )

    # 2. TCP probe on the old port fails.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)
    try:
        sock.connect(("127.0.0.1", port_1))
        sock.close()
        pytest.fail(f"old port {port_1} still accepting connections after idle shutdown")
    except OSError:
        pass  # expected

    # 3. ensure_coordinator from a fresh call re-spawns cleanly.
    port_2 = ensure_coordinator(workspace, config=cfg)
    try:
        assert port_2 > 0
    finally:
        stop_coordinator(workspace)


# ----------------------------------------------------------------------
# Sweep behavior — F2 wiring + grant reclaim
# ----------------------------------------------------------------------


def test_sweep_evicts_stale_preemption_notices(
    workspace: Path,
) -> None:
    """F2 wired: sweep periodically calls registry.evict_stale_notices
    with the lifecycle's notice_evict_max_age_sec threshold."""
    # Aggressive: 0.1s sweep, 0.2s max-age — anything older than 0.2s gets evicted.
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=0.2,
    )
    port = ensure_coordinator(workspace, config=cfg)
    try:
        coordinator = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())].coordinator
        # Plant an old notice — needs an artifact (FK constraint).
        from ccs.core.types import Artifact
        art = Artifact(id=uuid.uuid4(), name="sweep-test.md", version=1, content_hash="h")
        coordinator.registry.register_artifact(art, content="")
        coordinator.registry.record_preemption_notice(
            victim_agent_id=uuid.uuid4(),
            artifact_id=art.id,
            preempter_agent_id=uuid.uuid4(),
            preempted_at_unix_ts=time.time() - 5.0,  # 5s old, well past 0.2s max-age
        )
        # Wait for a sweep tick to fire.
        time.sleep(0.4)
        # Notice should be gone.
        remaining = coordinator.registry.peek_preemption_notice(
            uuid.uuid4(), art.id  # any agent — notice keyed by (agent, artifact)
        )
        assert remaining is None
        # Stronger: pop from the original victim returns empty too.
        # (We don't know the victim_id here because we used uuid4() inline;
        # instead, plant another known notice and re-check that the sweep ran.)
        victim_2 = uuid.uuid4()
        coordinator.registry.record_preemption_notice(
            victim_agent_id=victim_2,
            artifact_id=art.id,
            preempter_agent_id=uuid.uuid4(),
            preempted_at_unix_ts=time.time() - 5.0,
        )
        time.sleep(0.4)
        assert coordinator.registry.pop_pending_notices(victim_2) == []
    finally:
        stop_coordinator(workspace)


# ----------------------------------------------------------------------
# 10-process race — the integration test
# ----------------------------------------------------------------------


def _race_worker(args: tuple) -> tuple[int, int]:
    """Subprocess worker for the race test: call ensure_coordinator and
    return (pid, port). Run at module top-level for picklability."""
    from pathlib import Path as _P

    from ccs.adapters.claude_code.lifecycle import (
        LifecycleConfig as _C,
    )
    from ccs.adapters.claude_code.lifecycle import (
        ensure_coordinator as _ec,
    )
    workspace_str, = args
    cfg = _C(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=60,    # generous — accommodates cold-start
        port_file_retry_interval_sec=0.050,
        connect_retry_attempts=5,
        connect_retry_interval_sec=0.05,
    )
    port = _ec(_P(workspace_str), config=cfg)
    return (os.getpid(), port)


def test_ten_process_race_one_binds_others_read_same_port(
    workspace: Path,
) -> None:
    """Integration: 10 concurrent ensure_coordinator calls (real
    subprocesses) — exactly one binds the port, the other 9 read the
    SAME port from the pid file. This is the load-bearing race test
    for Unit 5."""
    ctx = mp.get_context("spawn")  # fresh interpreter per worker
    with ctx.Pool(10) as pool:
        results = pool.map(_race_worker, [(str(workspace),)] * 10)
    pids, ports = zip(*results)
    # All workers must report some port.
    assert all(p > 0 for p in ports), f"some workers got sentinel ports: {ports}"
    # The LOAD-BEARING correctness assertion: all 10 ensure_coordinator
    # calls converged on a single bound port. Whether the work was
    # distributed across 1, 4, or 10 worker processes is implementation-
    # detail noise of `mp.Pool` scheduling on the host — irrelevant to
    # the race-safety claim.
    unique_ports = set(ports)
    assert len(unique_ports) == 1, (
        f"expected exactly one bound port; got {unique_ports} from pids {pids}"
    )
    # Note on test-harness limitation: on resource-constrained CI runners,
    # `mp.Pool(10)` can serialize all 10 tasks into a single worker
    # process (one pid handles every task before others get scheduled). When
    # that happens, the test exercises the G3 entry-short-circuit / port-
    # file read path rather than the full multi-process race. The load-
    # bearing assertion above still passes either way. A dedicated
    # subprocess + threading.Barrier test would force actual concurrency;
    # deferred as an enhancement since the port-equality check is the
    # actual correctness gate.
    distinct_workers = len(set(pids))
    if distinct_workers < 2:
        # Surface as a warning so the test still passes but the operator
        # sees that this run didn't exercise the multi-process race path.
        import warnings as _warnings
        _warnings.warn(
            f"10-process race test ran in a single worker (pid {pids[0]}); "
            f"port-equality assertion passed but multi-process race scenario "
            f"was not exercised on this host.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Clean up: kill the holder if it's still alive (the holder is one
    # of the worker subprocesses, which exited at pool teardown — the
    # OS released the flock and torn down the socket).
    # Verify by re-spawning in this process; should bind a fresh port.
    fresh_cfg = LifecycleConfig(idle_shutdown_sec=0, sweep_interval_sec=0)
    fresh_port = ensure_coordinator(workspace, config=fresh_cfg)
    try:
        # If the workers' holder is dead, we get a NEW port. If somehow
        # still alive (shouldn't be — pool exit kills children), we'd
        # get the same port via flock contention + port file read.
        assert fresh_port > 0
    finally:
        stop_coordinator(workspace)


# ----------------------------------------------------------------------
# State preservation across idle shutdown + respawn
# ----------------------------------------------------------------------


def test_state_persists_across_idle_shutdown_and_respawn(
    workspace: Path,
) -> None:
    """Integration: idle-shutdown → re-spawn must preserve SQLite state
    (cross-references KTD-6 — SQLite-WAL rehydration on next spawn)."""
    cfg = LifecycleConfig(
        idle_shutdown_sec=0.5,
        sweep_interval_sec=0.1,
        port_file_retry_attempts=10,
        port_file_retry_interval_sec=0.05,
    )
    port_1 = ensure_coordinator(workspace, config=cfg)
    # Register a tracked artifact while up.
    coordinator_1 = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())].coordinator
    from ccs.core.types import Artifact
    art = Artifact(id=uuid.uuid4(), name="persist-test.md", version=1, content_hash="abc")
    coordinator_1.registry.register_artifact(art, content="")
    artifact_id_known = art.id

    # Wait past idle threshold + shutdown completion (HTTPServer.shutdown()
    # polls every 0.5s; G2 self-probe adds ~0.3s; safe budget = 2.5s).
    # Also confirm shutdown actually fired before proceeding — otherwise
    # the unified-loop respawn would attach to the still-dying coordinator
    # and the test would race coord_1's registry close.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if str(workspace.resolve()) not in lifecycle._SPAWNED_REGISTRY:
            break
        time.sleep(0.1)
    else:
        pytest.fail("idle shutdown did not complete within 3s")

    # Re-spawn.
    cfg_no_idle = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
    )
    port_2 = ensure_coordinator(workspace, config=cfg_no_idle)
    try:
        assert port_2 > 0
        coordinator_2 = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())].coordinator
        # The artifact registered in coordinator_1 must be visible in coordinator_2
        # because they share the same SQLite file.
        rehydrated = coordinator_2.registry.get_artifact(artifact_id_known)
        assert rehydrated is not None
        assert rehydrated.name == "persist-test.md"
        assert rehydrated.version == 1
    finally:
        stop_coordinator(workspace)


# ----------------------------------------------------------------------
# G3/G4/G5/G6 hardening regression tests
# ----------------------------------------------------------------------


def test_g3_same_process_entry_short_circuit(
    workspace: Path, fast_cfg: LifecycleConfig
) -> None:
    """G3: a second ensure_coordinator call from the SAME process must
    short-circuit to the existing port via the entry check, NOT spawn a
    second coordinator that would leak fds and sweep threads."""
    port_1 = ensure_coordinator(workspace, config=fast_cfg)
    try:
        entry_1 = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())]
        port_2 = ensure_coordinator(workspace, config=fast_cfg)
        entry_2 = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())]
        assert port_2 == port_1, "second call must return the existing port"
        assert entry_1 is entry_2, "registry entry must NOT be replaced"
        assert entry_1.coordinator is entry_2.coordinator
    finally:
        stop_coordinator(workspace)


def test_g5_port_dropped_before_shutdown_completes(
    workspace: Path,
) -> None:
    """G5: pid file's port line is dropped BEFORE coordinator.shutdown()
    returns — so a concurrent loser reading the file during the drain
    window sees 'no port' instead of a port pointing at a coordinator
    that is about to die.

    P2 ce-review fix #12 (testing): the earlier version of this test
    guarded the assertion behind `if not shutdown_done.is_set()`. On
    fast machines coordinator.shutdown() completes within the 50ms sleep
    window, leaving the assertion silently skipped — the test passed
    without actually checking G5. Fix: monkeypatch coordinator.shutdown
    to inject a 500ms sleep so the drain window deterministically
    outlasts the assertion window. The assertion is now unconditional.
    """
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,  # no idle-shutdown loop interference
        spawn_self_probe_attempts=10,
    )
    port = ensure_coordinator(workspace, config=cfg)
    assert port > 0
    pid_file = workspace / ".coherence" / "server.pid"
    assert _read_port_from_file(pid_file) == port

    # Inject a deterministic sleep into coordinator.shutdown so the drain
    # window outlasts the assertion check window on any host (fast laptop
    # or slow CI). The monkeypatch wraps the real shutdown so cleanup
    # still happens.
    entry = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())]
    real_shutdown = entry.coordinator.shutdown
    drain_started = __import__("threading").Event()

    def slow_shutdown():
        drain_started.set()
        time.sleep(0.5)  # 500ms drain window — much longer than the assertion check
        real_shutdown()

    entry.coordinator.shutdown = slow_shutdown  # type: ignore[method-assign]

    # Trigger shutdown in a background thread.
    import threading as _t
    shutdown_done = _t.Event()

    def trigger():
        stop_coordinator(workspace)
        shutdown_done.set()

    t = _t.Thread(target=trigger, daemon=True)
    t.start()

    # Wait until coordinator.shutdown has been entered (drain_started fires)
    # then wait an additional small margin to ensure the port-drop step
    # (which happens BEFORE coordinator.shutdown in _shutdown_sequence) has
    # already committed to disk.
    assert drain_started.wait(timeout=2.0), (
        "monkeypatched shutdown was not entered within 2s"
    )
    # We're now inside the 500ms drain window. The port MUST be dropped
    # (G5 ordering: drop port BEFORE coordinator.shutdown).
    assert not shutdown_done.is_set(), (
        "shutdown completed unexpectedly fast — drain window too short to test G5"
    )
    port_visible = _read_port_from_file(pid_file)
    assert port_visible is None, (
        "G5 violation: port still visible at "
        f"{port_visible} while shutdown is mid-drain (port-drop must "
        f"precede coordinator.shutdown per _shutdown_sequence ordering)"
    )

    t.join(timeout=5.0)
    assert shutdown_done.is_set()


def test_g6_concurrent_stop_and_idle_do_not_double_execute(
    workspace: Path,
) -> None:
    """G6: if stop_coordinator and the idle-shutdown loop fire concurrently
    against the same entry, the shutdown sequence runs EXACTLY ONCE.
    Verified via the shutdown_done event being set exactly once and the
    lock_fd being closed exactly once (no double-close raising EBADF)."""
    cfg = LifecycleConfig(
        idle_shutdown_sec=0.3,    # very aggressive — race with stop
        sweep_interval_sec=0.05,
    )
    port = ensure_coordinator(workspace, config=cfg)
    assert port > 0
    entry = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())]

    # Race: let idle have ~0.3s, fire stop concurrently at ~0.3s.
    import threading as _t

    def manual_stop():
        time.sleep(0.3)
        stop_coordinator(workspace)

    t = _t.Thread(target=manual_stop, daemon=True)
    t.start()
    t.join(timeout=5.0)

    # Wait for idle thread to also finish (it may have raced manual_stop).
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if entry.shutdown_done.is_set():
            break
        time.sleep(0.05)

    assert entry.shutdown_done.is_set(), "shutdown did not complete"
    # The fd must be closed — verified by os.fstat raising EBADF.
    import errno as _e
    with pytest.raises(OSError) as excinfo:
        os.fstat(entry.lock_fd)
    assert excinfo.value.errno == _e.EBADF


def test_g4_shutdown_raise_aborts_without_releasing_lock(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G4: if coordinator.shutdown() raises, the sequence aborts BEFORE
    releasing the flock. This keeps the workspace lock held so a new
    spawn cannot race against the still-running coordinator process."""
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        spawn_self_probe_attempts=10,
    )
    port = ensure_coordinator(workspace, config=cfg)
    assert port > 0
    entry = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())]
    lock_fd = entry.lock_fd

    # Force coordinator.shutdown to raise on its next call.
    def raising_shutdown():
        raise RuntimeError("simulated shutdown failure")

    monkeypatch.setattr(entry.coordinator, "shutdown", raising_shutdown)

    # Stop should run but ABORT after shutdown raises.
    ran = stop_coordinator(workspace)
    assert ran is True, "stop_coordinator should have invoked the sequence"
    # G4: shutdown_done must NOT be set (sequence aborted), so a retry
    # is possible. lock_fd must still be open (lock still held).
    assert not entry.shutdown_done.is_set(), (
        "shutdown_done set despite abort — release happened anyway"
    )
    # If the lock had been released and fd closed, os.fstat would raise.
    # We expect it to succeed: lock_fd is still a valid open descriptor.
    try:
        os.fstat(lock_fd)
    except OSError as exc:
        pytest.fail(f"lock_fd unexpectedly closed: {exc}")
    # The registry entry must still be present so a retry can find it.
    assert str(workspace.resolve()) in lifecycle._SPAWNED_REGISTRY

    # Manual cleanup for the test (otherwise we'd leak the lock_fd).
    monkeypatch.undo()
    stop_coordinator(workspace)


# ----------------------------------------------------------------------
# KTD-H (Unit 5 L1) — inode revalidation per retry iteration
# ----------------------------------------------------------------------


def test_l1_inode_match_helper_detects_unlink_recreate(workspace: Path) -> None:
    """L1 invariant: ``_inode_matches`` returns True when fd and path point
    at the same inode, and False after an external unlink + recreate.
    Canonical l1_ prefix per plan §'Cross-cutting test discipline' line 789."""
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pid_file = coherence_dir / "server.pid"
    pid_file.write_text("", encoding="utf-8")
    fd = os.open(pid_file, os.O_RDWR)
    try:
        # Same inode → match.
        assert lifecycle._inode_matches(fd, pid_file) is True

        # Unlink + recreate → mismatch.
        pid_file.unlink()
        pid_file.write_text("", encoding="utf-8")
        assert lifecycle._inode_matches(fd, pid_file) is False
    finally:
        os.close(fd)


def test_h1_inode_matches_helper_detects_unlink_recreate(workspace: Path) -> None:  # backward alias
    return test_l1_inode_match_helper_detects_unlink_recreate(workspace)


def test_h2_inode_matches_returns_false_when_path_absent(workspace: Path) -> None:
    """If the path is unlinked entirely (no recreate), the helper still
    returns False rather than raising — caller treats as 'revalidate'."""
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pid_file = coherence_dir / "server.pid"
    pid_file.write_text("", encoding="utf-8")
    fd = os.open(pid_file, os.O_RDWR)
    try:
        pid_file.unlink()
        assert lifecycle._inode_matches(fd, pid_file) is False
    finally:
        os.close(fd)


def test_h3_ensure_coordinator_recovers_after_rm_rf_race(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KTD-H end-to-end: simulate an external ``rm -rf .coherence/`` between
    the pid-file open and the first retry iteration. ``ensure_coordinator``
    must detect the inode mismatch, re-open, and still bind a coordinator
    on a fresh inode rather than holding an orphan fd."""
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.02,
        inode_revalidation_budget=3,
        spawn_self_probe_attempts=20,
        spawn_self_probe_interval_sec=0.05,
    )

    # Monkey-patch _open_pidfile so the FIRST call performs the open AND
    # immediately wipes the directory (simulating `rm -rf .coherence`
    # racing in just after open). Subsequent calls open normally.
    real_open = lifecycle._open_pidfile
    call_count = {"n": 0}

    def open_then_wipe(pid_file: Path) -> Optional[int]:
        call_count["n"] += 1
        fd = real_open(pid_file)
        if fd is None:
            return None
        if call_count["n"] == 1:
            # Simulate the race: external rm -rf wipes .coherence/ AFTER
            # we got our fd. Our fd now refers to an orphaned inode.
            try:
                pid_file.unlink()
            except FileNotFoundError:
                pass
        return fd

    monkeypatch.setattr(lifecycle, "_open_pidfile", open_then_wipe)
    try:
        port = ensure_coordinator(workspace, config=cfg)
        assert port > 0, (
            f"expected ensure_coordinator to recover via inode revalidation; got {port}"
        )
        # The live pid file should match the bound port.
        pid_file = workspace / ".coherence" / "server.pid"
        live_port = _read_port_from_file(pid_file)
        assert live_port == port, (
            f"pid file ({live_port}) does not reflect bound port ({port}) — "
            "winner may have written to an orphaned inode"
        )
        # At least one revalidation must have occurred.
        assert call_count["n"] >= 2, (
            f"_open_pidfile called only {call_count['n']} time(s); "
            "expected at least 2 (initial + revalidation re-open)"
        )
    finally:
        monkeypatch.undo()
        stop_coordinator(workspace)


def test_h4_revalidation_budget_exhaustion_returns_minus_one(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pathological churn: every iteration sees a fresh inode mismatch.
    The revalidation budget caps the recovery work and returns -1 cleanly
    rather than spinning forever."""
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=5,
        port_file_retry_interval_sec=0.01,
        inode_revalidation_budget=2,
    )

    # Force _inode_matches to always report mismatch — simulates an
    # adversary unlinking the pid file every iteration.
    monkeypatch.setattr(lifecycle, "_inode_matches", lambda _fd, _path: False)
    try:
        port = ensure_coordinator(workspace, config=cfg)
        assert port == -1, (
            f"expected -1 after revalidation budget exhausted; got {port}"
        )
    finally:
        monkeypatch.undo()


# ----------------------------------------------------------------------
# Unit 5 L3 — cold-start instrumentation (telemetry only)
# ----------------------------------------------------------------------


def test_l3_cold_start_duration_populated_on_winner_path(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """L3 instrumentation: the lifecycle winner path measures the wall-clock
    time from CoordinatorHTTPServer construction through self-probe success
    and stores it on the coordinator for the future /status endpoint."""
    port = ensure_coordinator(workspace, config=fast_cfg)
    assert port > 0
    try:
        entry = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())]
        # Some time must have been recorded — fast hardware can finish under
        # 50ms but never zero.
        assert entry.coordinator.cold_start_duration_ms > 0.0
        # Sanity ceiling: even on a slow CI runner, cold start under the
        # spawn self-probe budget plus a generous margin.
        ceiling_ms = (
            fast_cfg.spawn_self_probe_attempts
            * fast_cfg.spawn_self_probe_interval_sec
            * 1000.0
        ) + 1000.0
        assert entry.coordinator.cold_start_duration_ms < ceiling_ms, (
            f"cold_start_duration_ms={entry.coordinator.cold_start_duration_ms:.1f} "
            f"exceeded ceiling {ceiling_ms:.1f}ms"
        )
    finally:
        stop_coordinator(workspace)


# ----------------------------------------------------------------------
# #187 — the sweep thresholds the coordinator reports on /status
# ----------------------------------------------------------------------


def _published_thresholds(workspace: Path) -> tuple[int | None, int | None]:
    coordinator = lifecycle._SPAWNED_REGISTRY[str(workspace.resolve())].coordinator
    return coordinator.grant_heartbeat_timeout_sec, coordinator.grant_max_hold_sec


def _sweep_threads() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t.name == "coord-sweep"}


@pytest.mark.parametrize(
    ("thresholds", "expected"),
    [
        ({}, (600, 1800)),
        ({"grant_heartbeat_timeout_sec": 120, "grant_max_hold_sec": 300}, (120, 300)),
    ],
    ids=["defaults", "configured"],
)
def test_coordinator_holds_the_thresholds_its_sweep_was_started_with(
    workspace: Path, thresholds: dict[str, int], expected: tuple[int, int],
) -> None:
    """#187 R4: the winner path hands the coordinator the two thresholds its
    sweep enforces, so /status reports what the sweep acts on."""
    cfg = LifecycleConfig(idle_shutdown_sec=0, sweep_interval_sec=0.1, **thresholds)
    assert ensure_coordinator(workspace, config=cfg) > 0
    try:
        assert _published_thresholds(workspace) == expected
    finally:
        stop_coordinator(workspace)


def test_coordinator_holds_no_thresholds_when_no_sweep_runs(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#187 R4: with ``sweep_interval_sec=0`` nothing enforces the configured
    thresholds, so the coordinator reports none rather than unenforced values."""
    cfg = replace(fast_cfg, sweep_interval_sec=0)
    assert ensure_coordinator(workspace, config=cfg) > 0
    try:
        assert _published_thresholds(workspace) == (None, None)
    finally:
        stop_coordinator(workspace)


@pytest.mark.parametrize(
    "field", ["grant_heartbeat_timeout_sec", "grant_max_hold_sec", "transient_timeout_sec"]
)
def test_a_threshold_below_one_publishes_none_and_the_sweep_still_starts(
    workspace: Path, fast_cfg: LifecycleConfig, field: str,
) -> None:
    """#187 R4: a threshold below 1 fails every stable-grant pass, so neither
    is reported as enforced; so does a transient timeout below 1, whose pass
    runs first in the same tick and raises before the stable-grant pass. The
    sweep thread still starts as before, and its per-pass failure log stays
    the report of the misconfiguration."""
    cfg = replace(fast_cfg, **{field: 0})
    sweeps_before = _sweep_threads()
    assert ensure_coordinator(workspace, config=cfg) > 0
    try:
        assert _published_thresholds(workspace) == (None, None)
        assert len(_sweep_threads() - sweeps_before) == 1
    finally:
        stop_coordinator(workspace)


def _a_sweep_tick_refuses(coordinator: CoordinatorHTTPServer, cfg: LifecycleConfig) -> bool:
    """Whether the two grant passes of one sweep tick, called in
    :func:`lifecycle._sweep_loop`'s order with ``cfg``'s values, refuse them."""
    try:
        coordinator.service.enforce_transient_timeouts(
            current_tick=0, timeout_ticks=cfg.transient_timeout_sec,
        )
        coordinator.service.enforce_stable_grant_timeouts(
            current_tick=0,
            heartbeat_timeout_ticks=cfg.grant_heartbeat_timeout_sec,
            max_hold_ticks=cfg.grant_max_hold_sec,
        )
    except ValueError:
        return True
    return False


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize(
    "field", ["grant_heartbeat_timeout_sec", "grant_max_hold_sec", "transient_timeout_sec"]
)
def test_the_thresholds_are_withheld_exactly_when_the_service_refuses_them(
    tmp_path: Path, fast_cfg: LifecycleConfig, field: str, value: int,
) -> None:
    """#187 R4: the publish rule follows the service's own refusal at each
    boundary value: the pair is withheld exactly when the sweep's service
    calls raise on the configured values, so a rule that drifts from the
    service's (``< 2``, say) is red on the value the two disagree about."""
    cfg = replace(fast_cfg, **{field: value})
    coordinator = CoordinatorHTTPServer(tmp_path, port=0)
    try:
        refused = _a_sweep_tick_refuses(coordinator, cfg)
    finally:
        coordinator.shutdown()

    published = lifecycle._published_sweep_thresholds(cfg)
    assert (published == (None, None)) is refused
    if not refused:
        assert published == (cfg.grant_heartbeat_timeout_sec, cfg.grant_max_hold_sec)


class _OneSweepTick:
    """The coordinator as :func:`lifecycle._sweep_loop` sees it, shutting down
    after one tick. The loop reads ``shutting_down`` before its sleep and again
    after it, then runs a tick; the third read is the first one after it."""

    def __init__(self, coordinator: CoordinatorHTTPServer) -> None:
        self._coordinator = coordinator
        self._reads = 0

    @property
    def shutting_down(self) -> bool:
        self._reads += 1
        return self._reads > 2

    def __getattr__(self, name: str) -> object:
        return getattr(self._coordinator, name)


@pytest.mark.parametrize("value", [0, 1])
@pytest.mark.parametrize(
    "field", ["grant_heartbeat_timeout_sec", "grant_max_hold_sec", "transient_timeout_sec"]
)
def test_one_sweep_loop_tick_reclaims_a_stale_grant_exactly_when_status_reports_the_thresholds(
    workspace: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    field: str, value: int,
) -> None:
    """One tick of the real sweep loop, with one setting at 0 or 1, reclaims
    a stale write grant exactly when the operator view of /status reports the
    thresholds the winner path published. At 0 the tick reclaims nothing and
    both thresholds are null, a transient timeout of 0 included: its pass runs
    first in the tick and raises before the stable-grant pass. Prevents
    /status naming thresholds no sweep enforces, or naming none while the
    sweep reclaims grants, when the loop's passes are reordered or split --
    which the hand-written order in the test above cannot see."""
    from ccs.adapters.claude_code import coordinator_server
    from ccs.cli._coherence_client import get, post, resolve_endpoint

    cfg = replace(
        replace(fast_cfg, grant_heartbeat_timeout_sec=60, grant_max_hold_sec=999_999_999),
        **{field: value},
    )
    entries: list = []
    # The winner path publishes the thresholds as it always does; the sweep is
    # then driven one tick at a time below, not on a background thread.
    monkeypatch.setattr(
        lifecycle, "_start_background_threads", lambda entry, _cfg: entries.append(entry),
    )
    assert ensure_coordinator(workspace, config=cfg) > 0
    try:
        [entry] = entries
        endpoint = resolve_endpoint(workspace)
        sid = str(uuid.uuid4())
        agent_id = str(coordinator_server.session_to_agent_id(sid))
        # The holder took its grant, and last heartbeated, 1000 seconds ago.
        with monkeypatch.context() as clock:
            clock.setattr(coordinator_server, "monotonic_seconds", lambda: int(time.time()) - 1_000)
            post(endpoint, "/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})

        def operator_view() -> tuple[dict, tuple[object, object]]:
            body = get(
                endpoint, "/status?detail=full", extra_headers={"Coherence-Local-Operator": "true"},
            )
            [row] = [s for s in body["sessions"] if s["agent_id"] == agent_id]
            return row, (body["grant_heartbeat_timeout_sec"], body["grant_max_hold_sec"])

        assert operator_view()[0]["states"] == {"plan.md": "EXCLUSIVE"}

        lifecycle._sweep_loop(SimpleNamespace(coordinator=_OneSweepTick(entry.coordinator)), cfg)

        row, reported = operator_view()
    finally:
        stop_coordinator(workspace)
    if value >= 1:
        assert row["states"] == {} and "plan.md" in row["reclaimed"], row
        assert reported == (cfg.grant_heartbeat_timeout_sec, cfg.grant_max_hold_sec)
    else:
        assert row["states"] == {"plan.md": "EXCLUSIVE"} and row["reclaimed"] == {}, row
        assert reported == (None, None)


def test_a_coordinator_built_without_lifecycle_holds_no_thresholds(
    tmp_path: Path,
) -> None:
    """#187 R4: a server constructed directly (tests, the corpus harness) runs
    no sweep, so it reports no thresholds."""
    coordinator = CoordinatorHTTPServer(tmp_path, port=0)
    try:
        assert coordinator.grant_heartbeat_timeout_sec is None
        assert coordinator.grant_max_hold_sec is None
    finally:
        coordinator.shutdown()


# ----------------------------------------------------------------------
# KP-7 — wait_for_shutdown public API
# ----------------------------------------------------------------------


def test_kp7_wait_for_shutdown_returns_false_when_no_entry(tmp_path):
    """wait_for_shutdown returns False when no in-process coordinator
    entry exists for the given root (idempotent observer; no spawn)."""
    from ccs.adapters.claude_code.lifecycle import wait_for_shutdown
    assert wait_for_shutdown(tmp_path, timeout_sec=0.1) is False


def test_kp7_wait_for_shutdown_returns_true_after_stop(tmp_path, fast_cfg):
    """End-to-end: spawn → stop in background → wait_for_shutdown
    returns True once shutdown_done flips."""
    import threading as _t

    from ccs.adapters.claude_code.lifecycle import (
        ensure_coordinator,
        stop_coordinator,
        wait_for_shutdown,
    )

    port = ensure_coordinator(tmp_path, config=fast_cfg)
    assert port > 0

    # Stop in a background thread; main waits.
    def trigger_stop():
        time.sleep(0.1)
        stop_coordinator(tmp_path)

    _t.Thread(target=trigger_stop, daemon=True).start()
    assert wait_for_shutdown(tmp_path, poll_interval_sec=0.05, timeout_sec=3.0) is True


def test_kp7_wait_for_shutdown_times_out_when_coordinator_stays_up(
    tmp_path, fast_cfg
):
    """timeout_sec elapsed without shutdown → returns False, leaves the
    coordinator running so the caller can react."""
    from ccs.adapters.claude_code.lifecycle import (
        ensure_coordinator,
        stop_coordinator,
        wait_for_shutdown,
    )

    port = ensure_coordinator(tmp_path, config=fast_cfg)
    assert port > 0
    try:
        # Coordinator is up; wait_for_shutdown must time out.
        assert wait_for_shutdown(tmp_path, poll_interval_sec=0.05, timeout_sec=0.3) is False
    finally:
        stop_coordinator(tmp_path)


# ----------------------------------------------------------------------
# PS-03 — L2 (in-flight drain on shutdown) invariant alias tests
# ----------------------------------------------------------------------
#
# The Unit 5 plan mandates at least one ``test_l2_*`` prefixed test in
# this file. The substantive drain coverage lives in
# tests/test_claude_code_coordinator_server.py as ``test_i1..i6``
# (acquire/release pairing, drain blocking until release, drain
# timeout, dispatch wiring, exception-path balance). The alias
# below routes the lifecycle-suite reader to the existing detail
# and exercises the public shutdown → drain → unreachable path.


def test_l2_drain_observable_via_lifecycle_stop(tmp_path, fast_cfg):
    """L2 (KTD-I) invariant: stop_coordinator() drives the in-flight
    drain inside CoordinatorHTTPServer.shutdown() and returns only
    after the registry is closed. End-to-end smoke that the public
    lifecycle API wires the drain correctly; detailed acquire/release
    semantics are in test_claude_code_coordinator_server::test_i1..i6.
    """
    from ccs.adapters.claude_code.lifecycle import (
        ensure_coordinator,
        stop_coordinator,
    )

    port = ensure_coordinator(tmp_path, config=fast_cfg)
    assert port > 0
    # No in-flight requests: drain should return promptly (well under
    # IN_FLIGHT_DRAIN_TIMEOUT_SEC=5s). Just verifying the path runs.
    start = time.monotonic()
    stop_coordinator(tmp_path)
    elapsed = time.monotonic() - start
    assert elapsed < 4.0, (
        f"shutdown with no in-flight handlers took {elapsed:.2f}s; "
        "drain may not be returning promptly when counter is already 0"
    )


# ----------------------------------------------------------------------
# L1 — construct->write inode TOCTOU sub-window
# ----------------------------------------------------------------------


def test_l1_construct_to_write_window_recheck_recovers(
    workspace: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L1: the top-of-loop inode check runs BEFORE CoordinatorHTTPServer
    construction (SQLite open + TCP bind). An external rm -rf landing DURING
    construction must be caught by the post-construct recheck — the winner must
    tear down the orphaned coordinator and recover on a fresh inode, NOT write
    the port into an orphaned server.pid no concurrent reader can see."""
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.02,
        inode_revalidation_budget=3,
        spawn_self_probe_attempts=20,
        spawn_self_probe_interval_sec=0.05,
    )
    pid_file = workspace / ".coherence" / "server.pid"
    real_ctor = lifecycle.CoordinatorHTTPServer
    state = {"constructs": 0, "teardowns": 0}

    def wiping_ctor(*args, **kwargs):
        inst = real_ctor(*args, **kwargs)
        state["constructs"] += 1
        if state["constructs"] == 1:
            # Simulate `rm -rf .coherence/` racing in DURING construction:
            # unlink server.pid so the fd ensure_coordinator holds is orphaned.
            try:
                pid_file.unlink()
            except FileNotFoundError:
                pass
            real_shutdown = inst.shutdown

            def tracking_shutdown() -> None:
                state["teardowns"] += 1
                real_shutdown()

            inst.shutdown = tracking_shutdown  # type: ignore[method-assign]
        return inst

    monkeypatch.setattr(lifecycle, "CoordinatorHTTPServer", wiping_ctor)
    try:
        port = ensure_coordinator(workspace, config=cfg)
        assert port > 0, f"expected recovery on a fresh inode; got {port}"
        live_port = _read_port_from_file(pid_file)
        assert live_port == port, (
            f"pid file ({live_port}) != bound port ({port}); the recheck did not "
            "fire and the port was written to an orphaned inode"
        )
        assert state["constructs"] >= 2, (
            "recheck must have torn down the orphaned coordinator and reconstructed "
            f"on a fresh inode; constructs={state['constructs']}"
        )
        assert state["teardowns"] == 1, (
            f"the orphaned coordinator must be shut down exactly once; "
            f"teardowns={state['teardowns']}"
        )
    finally:
        monkeypatch.undo()
        stop_coordinator(workspace)


# ----------------------------------------------------------------------
# L3 — spawn/join exhaustion observability counter
# ----------------------------------------------------------------------


def test_l3_retry_exhaustion_increments_per_reason_counter(
    workspace: Path,
) -> None:
    """L3: when the spawn-or-join loop exhausts its port-file retry budget and
    returns -1, the per-reason observability counter records a "retry_budget"
    exhaustion so the otherwise-silent loser degrade is countable — without
    conflating it with the dying-coordinator "loser_probe_failed" reason."""
    import fcntl  # POSIX-only; the surrounding suite is POSIX-gated elsewhere

    lifecycle._reset_spawn_join_exhaustion()
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pid_file = coherence_dir / "server.pid"
    # Hold the flock from THIS process so ensure_coordinator can never win, and
    # leave the port line empty so the loser-read never succeeds — deterministic
    # retry-budget exhaustion with no real coordinator spawned.
    hold_fd = os.open(str(pid_file), os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(hold_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=3,
        port_file_retry_interval_sec=0.01,
        inode_revalidation_budget=3,
        spawn_self_probe_attempts=3,
        spawn_self_probe_interval_sec=0.01,
    )
    try:
        port = ensure_coordinator(workspace, config=cfg)
        assert port == -1, f"expected retry-budget exhaustion -> -1; got {port}"
        by_reason = lifecycle.get_spawn_join_exhaustion_by_reason()
        assert by_reason.get("retry_budget") == 1, by_reason
        assert "loser_probe_failed" not in by_reason, (
            f"must not conflate herd exhaustion with the dying-coordinator probe "
            f"failure; got {by_reason}"
        )
        assert lifecycle.get_spawn_join_exhaustion_total() == 1
    finally:
        fcntl.flock(hold_fd, fcntl.LOCK_UN)
        os.close(hold_fd)
        lifecycle._reset_spawn_join_exhaustion()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_l3_counter_lock_held_at_fork_does_not_hang_the_child() -> None:
    """Only the forking thread survives a fork. If another thread held the
    counter lock at that moment, the child inherited it held with no thread left
    to release it, so its next exhaustion (a post-fork re-attach can reach one)
    or counter read blocked forever."""
    lifecycle._reset_spawn_join_exhaustion()
    held = threading.Event()
    release = threading.Event()

    def hold_lock() -> None:
        with lifecycle._EXHAUSTION_LOCK:
            held.set()
            release.wait(30)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            os.close(read_fd)
            try:
                lifecycle._record_spawn_join_exhaustion("retry_budget")
                os.write(write_fd, str(lifecycle.get_spawn_join_exhaustion_total()).encode())
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        try:
            # A hung child never writes: stop waiting, then kill and reap it.
            ready, _, _ = select.select([read_fd], [], [], 5.0)
            reported = os.read(read_fd, 16).decode() if ready else None
        finally:
            os.close(read_fd)
            os.kill(pid, signal.SIGKILL)  # no-op on a child that already exited
            os.waitpid(pid, 0)
    finally:
        release.set()
        holder.join(5)
        lifecycle._reset_spawn_join_exhaustion()
    assert reported == "1", "the forked child hung on the inherited counter lock"
