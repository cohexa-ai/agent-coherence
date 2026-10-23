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
import subprocess
import sys
import threading
import time
import uuid
import warnings
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

import ccs
from ccs.adapters.claude_code import lifecycle
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import (
    CoordinatorHandle,
    LifecycleConfig,
    SpawnOutcome,
    connect_or_spawn,
    ensure_coordinator,
    ensure_coordinator_handle,
    stop_coordinator,
)
from ccs.adapters.claude_code.lifecycle import (
    read_port_from_file as _read_port_from_file,
)

#: FROZEN duplicates of the ``SpawnOutcome`` values (#348), written out as
#: literals on purpose: the enum is a ``str`` Enum and a host compares the
#: outcome to these strings, so a rename is a breaking change. A set derived
#: from the enum under test would move its own goalposts. Cardinality is
#: pinned separately so a same-size add+remove cannot slip past set equality.
_BOUND = "bound"
_JOINED_SAME_PROCESS = "joined_same_process"
_JOINED_OTHER_PROCESS = "joined_other_process"
_EXPECTED_SPAWN_OUTCOMES = {_BOUND, _JOINED_SAME_PROCESS, _JOINED_OTHER_PROCESS}
_EXPECTED_SPAWN_OUTCOME_COUNT = 3

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
    """Second call from the same process is idempotent: it returns the
    same port without rebinding.

    It answers from the G3 short-circuit (this process's registry entry for
    the root). It would not re-bind without that entry either: ``flock``
    locks belong to the open file description, not to the process, so the
    second call's own open of server.pid finds the lock held, reads the port
    the first call wrote and joins it (#348 reports that join as
    JOINED_SAME_PROCESS; see the ``ensure_coordinator_handle`` tests)."""
    port_1 = ensure_coordinator(workspace, config=fast_cfg)
    try:
        port_2 = ensure_coordinator(workspace, config=fast_cfg)
        assert port_2 == port_1
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


#: The start barrier of the ten-process race, handed to each pool process by
#: :func:`_race_worker_init` (a multiprocessing barrier reaches a worker only
#: through process creation, never through a task's arguments).
_RACE_BARRIER = None


def _race_worker_init(barrier) -> None:
    """Pool initializer: give this worker process the shared start barrier."""
    global _RACE_BARRIER
    _RACE_BARRIER = barrier


def _race_worker(args: tuple) -> tuple[int, int, str | None, int | None]:
    """Subprocess worker for the race test: call ensure_coordinator_handle
    and return (pid, port, outcome value, binder_pid). Run at module
    top-level for picklability.

    It waits on the start barrier first. A worker blocked there cannot take
    a second task, so the ten calls run in ten distinct processes and none
    of them can answer from the binder's own registry entry (#348)."""
    from pathlib import Path as _P

    from ccs.adapters.claude_code.lifecycle import (
        LifecycleConfig as _C,
    )
    from ccs.adapters.claude_code.lifecycle import (
        ensure_coordinator_handle as _ech,
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
    _RACE_BARRIER.wait(timeout=60)
    handle = _ech(_P(workspace_str), config=cfg)
    if handle is None:
        return (os.getpid(), -1, None, None)
    return (os.getpid(), handle.port, handle.outcome.value, handle.binder_pid)


# Ten real processes race for one port: run with -n 0 (tests/conftest.py).
@pytest.mark.serial
def test_ten_process_race_one_binds_others_read_same_port(
    workspace: Path,
) -> None:
    """Integration: 10 concurrent ensure_coordinator calls (real
    subprocesses) — exactly one binds the port, the other 9 read the
    SAME port from the pid file. This is the load-bearing race test
    for Unit 5.

    #348: the one bind reports ``bound`` and the nine joins report
    ``joined_other_process``, each naming the binder's pid. Prevents a
    joining process from believing it started the coordinator, and so
    stopping, or waiting on, one it does not own."""
    ctx = mp.get_context("spawn")  # fresh interpreter per worker
    barrier = ctx.Barrier(10)
    with ctx.Pool(10, initializer=_race_worker_init, initargs=(barrier,)) as pool:
        results = pool.map(_race_worker, [(str(workspace),)] * 10)
    pids, ports, outcomes, binder_pids = zip(*results)
    # All workers must report some port.
    assert all(p > 0 for p in ports), f"some workers got sentinel ports: {ports}"
    # The LOAD-BEARING correctness assertion: all 10 ensure_coordinator
    # calls converged on a single bound port.
    unique_ports = set(ports)
    assert len(unique_ports) == 1, (
        f"expected exactly one bound port; got {unique_ports} from pids {pids}"
    )
    # The start barrier holds every worker until all ten have a task, so
    # the ten calls ran in ten processes (the barrier would have broken
    # otherwise) and each loser joined through the pid file.
    assert len(set(pids)) == 10, pids
    assert sorted(outcomes) == [_BOUND] + [_JOINED_OTHER_PROCESS] * 9, outcomes
    bound_pid = pids[outcomes.index(_BOUND)]
    assert set(binder_pids) == {bound_pid}, (bound_pid, binder_pids)
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


# ----------------------------------------------------------------------
# #348 — ensure_coordinator_handle: did this call start the coordinator?
# ----------------------------------------------------------------------


def test_spawn_outcome_values_are_the_published_vocabulary() -> None:
    """#348: a host compares the outcome to these strings (the enum is a
    ``str`` Enum). Prevents a rename or an added outcome from reaching a
    host unannounced."""
    assert {outcome.value for outcome in SpawnOutcome} == _EXPECTED_SPAWN_OUTCOMES
    assert len(SpawnOutcome) == _EXPECTED_SPAWN_OUTCOME_COUNT
    assert all(isinstance(outcome, str) for outcome in SpawnOutcome)


def test_handle_first_call_on_a_fresh_root_reports_bound(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#348: the call that binds the coordinator says so: BOUND, with this
    process as the binder and the port the pid file publishes.

    Prevents a host from having to infer "I started it" from process state
    read around the call, which cannot tell a bind from a join."""
    handle = ensure_coordinator_handle(workspace, config=fast_cfg)
    try:
        assert isinstance(handle, CoordinatorHandle)
        assert isinstance(handle.outcome, SpawnOutcome)
        assert handle.outcome == _BOUND
        assert handle.spawned_by_this_call is True
        assert handle.binder_pid == os.getpid()
        assert handle.port == _read_port_from_file(workspace / ".coherence" / "server.pid")
        with pytest.raises(FrozenInstanceError):
            handle.port = 1  # type: ignore[misc]
    finally:
        stop_coordinator(workspace)


def test_handle_second_call_in_this_process_reports_joined_same_process(
    tmp_path: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#348: a later call in the process that bound the coordinator, under
    the root's own path or an alias of it, joins it and says so:
    JOINED_SAME_PROCESS on the same port, this process as the binder, and
    ``spawned_by_this_call`` False.

    Prevents a second caller in the binding process from being told it
    started the coordinator, and so stopping it under the first caller.
    The alias is a symlink: ``root / "."`` is no alias, since pathlib drops
    the dot before the call sees it."""
    root = tmp_path / "ws"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    first = ensure_coordinator_handle(root, config=fast_cfg)
    try:
        assert first is not None and first.outcome == _BOUND
        for path in (root, alias):
            again = ensure_coordinator_handle(path, config=fast_cfg)
            assert again is not None
            assert again.outcome == _JOINED_SAME_PROCESS, path
            assert again.spawned_by_this_call is False
            assert (again.port, again.binder_pid) == (first.port, os.getpid())
    finally:
        stop_coordinator(root)


#: The other process of the cross-process join test: binds a coordinator for
#: the root in argv[1], prints "<pid> <port> <outcome>", holds it until its
#: stdin closes, then stops it and exits.
_CHILD_BINDER = """\
import os
import sys
from pathlib import Path

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, ensure_coordinator_handle, stop_coordinator

root = Path(sys.argv[1])
handle = ensure_coordinator_handle(root, config=LifecycleConfig(idle_shutdown_sec=0, sweep_interval_sec=0))
print(os.getpid(), handle.port, handle.outcome.value, flush=True)
sys.stdin.read()
stop_coordinator(root)
"""


def _child_env() -> dict[str, str]:
    """This environment with this tree's ``src`` first on PYTHONPATH, so a
    child interpreter runs the code under test, not another checkout's."""
    tree_src = str(Path(ccs.__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [tree_src, existing]))}


@pytest.mark.serial
def test_handle_join_of_a_coordinator_another_process_bound_names_that_process(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#348: a call that joins a coordinator another process bound reports
    JOINED_OTHER_PROCESS with that process's pid, and this process cannot
    stop it (``stop_coordinator`` returns False).

    Prevents a host from taking a coordinator it joined for its own: it
    would wait on, or try to stop, a coordinator whose lifetime belongs to
    another process."""
    err_path = workspace / "binder.err"
    with open(err_path, "w", encoding="utf-8") as err:
        child = subprocess.Popen(
            [sys.executable, "-c", _CHILD_BINDER, str(workspace)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True, env=_child_env(),
        )
    try:
        ready, _, _ = select.select([child.stdout], [], [], 60.0)
        line = child.stdout.readline() if ready else ""
        assert line, f"the binder never reported its bind: {err_path.read_text()}"
        child_pid, child_port, child_outcome = line.split()
        assert (int(child_pid), child_outcome) == (child.pid, _BOUND)

        handle = ensure_coordinator_handle(workspace, config=fast_cfg)

        assert handle is not None
        assert handle.outcome == _JOINED_OTHER_PROCESS
        assert handle.spawned_by_this_call is False
        assert handle.binder_pid == child.pid
        assert handle.port == int(child_port)
        assert stop_coordinator(workspace) is False
    finally:
        child.stdin.close()  # the binder stops its coordinator and exits
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()  # our own child, wedged
            child.wait()
    assert child.returncode == 0, err_path.read_text()


_RACE_THREADS = 8
_RACE_ROOTS = 10


def _thread_race_cfg(fast_cfg: LifecycleConfig) -> LifecycleConfig:
    # No background threads, and a tight 5ms port-file poll so a losing
    # thread reads the port as soon as it is visible: inside the window the
    # bind marker exists to cover. 1000 polls leave a 5s cold-start budget.
    return replace(
        fast_cfg, sweep_interval_sec=0, port_file_retry_attempts=1000, port_file_retry_interval_sec=0.005,
    )


def _race_in_threads(root: Path, cfg: LifecycleConfig) -> list[CoordinatorHandle | None]:
    """Call ensure_coordinator_handle from ``_RACE_THREADS`` threads released
    together by one barrier, and return every answer. A raise fails the test."""
    barrier = threading.Barrier(_RACE_THREADS)
    answers: list[CoordinatorHandle | None] = []
    errors: list[BaseException] = []

    def call() -> None:
        try:
            barrier.wait(timeout=30)
            answers.append(ensure_coordinator_handle(root, config=cfg))
        except BaseException as exc:  # reported on the test's own thread below
            errors.append(exc)

    workers = [threading.Thread(target=call) for _ in range(_RACE_THREADS)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=60)
    assert not errors, errors
    return answers


def _assert_one_bound_and_the_rest_joined_this_process(answers: list[CoordinatorHandle | None]) -> None:
    assert len(answers) == _RACE_THREADS and None not in answers, answers
    assert len({answer.port for answer in answers}) == 1, answers
    assert sorted(answer.outcome for answer in answers) == (
        [_BOUND] + [_JOINED_SAME_PROCESS] * (_RACE_THREADS - 1)
    ), answers
    assert {answer.binder_pid for answer in answers} == {os.getpid()}, answers


def _count_port_file_joins(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record each pid-file read that found a port, i.e. each join that came
    through the pid file rather than this process's registry entry (G3).
    Observation only: the reads return what they always would."""
    seen: list[int] = []
    real_read = lifecycle._read_pidfile

    def counting_read(pid_file: Path) -> tuple[int | None, int | None]:
        pid, port = real_read(pid_file)
        if port is not None:
            seen.append(port)
        return pid, port

    monkeypatch.setattr(lifecycle, "_read_pidfile", counting_read)
    return seen


def _warn_unless_a_thread_joined_through_the_pid_file(port_file_joins: list[int]) -> None:
    if not port_file_joins:
        warnings.warn(
            "no thread joined through the pid file on this host: every joiner took the "
            "registry short-circuit, so the same-process bind marker was not exercised",
            RuntimeWarning,
            stacklevel=3,
        )


@pytest.mark.serial
def test_handle_thread_race_in_one_process_reports_one_bound_and_the_rest_joined_same_process(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#348: eight threads of one process race on each of ten fresh roots.
    Exactly one call per root binds and reports BOUND; the seven others join
    the same port and report JOINED_SAME_PROCESS, whether they joined
    through the pid file or through this process's registry entry.

    Prevents the answer from being inferred from state read before and after
    the call: a thread that joins sees the same "a coordinator is now
    registered in this process" as the thread that bound it, so such a read
    credits every racing thread with the spawn, and each would then stop a
    coordinator the others still use."""
    cfg = _thread_race_cfg(fast_cfg)
    port_file_joins = _count_port_file_joins(monkeypatch)
    previous_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for index in range(_RACE_ROOTS):
            root = tmp_path / f"ws{index}"
            root.mkdir()
            try:
                answers = _race_in_threads(root, cfg)
            finally:
                stop_coordinator(root)
            _assert_one_bound_and_the_rest_joined_this_process(answers)
    finally:
        sys.setswitchinterval(previous_interval)
    _warn_unless_a_thread_joined_through_the_pid_file(port_file_joins)


@pytest.mark.serial
def test_handle_same_process_join_inside_the_pid_file_write_window_reports_joined_same_process(
    workspace: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#348: the winner's pid-file write makes the port readable before the
    write (with its fsync) returns, and the registry insert comes after
    that. A slow disk, held open here for 0.3s, lets every losing thread of
    this process read the port inside that window; each must still report
    JOINED_SAME_PROCESS.

    Prevents the in-process bind marker from being recorded after the
    pid-file write: a joiner that read the port first would report another
    process's coordinator, so a host would disown a coordinator its own
    process runs. The plain race above reaches this window only by chance;
    this test holds it open."""
    real_write = lifecycle._write_pidfile

    def slow_write(fd: int, pid: int, port: int) -> None:
        real_write(fd, pid, port)
        time.sleep(0.3)  # the port is readable; the write has not returned

    monkeypatch.setattr(lifecycle, "_write_pidfile", slow_write)
    port_file_joins = _count_port_file_joins(monkeypatch)
    try:
        answers = _race_in_threads(workspace, _thread_race_cfg(fast_cfg))
    finally:
        stop_coordinator(workspace)
    _assert_one_bound_and_the_rest_joined_this_process(answers)
    _warn_unless_a_thread_joined_through_the_pid_file(port_file_joins)


def _leave_a_dead_binders_pid_file(root: Path) -> tuple[int, int]:
    """Write server.pid as a binder that exited without ``stop_coordinator``
    leaves it (SIGTERM, crash, reboot): its pid and port, with nothing
    holding the flock or listening on the port. Returns ``(pid, port)``."""
    exited = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True, text=True, check=True,
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        free_port = sock.getsockname()[1]
    coherence_dir = root / ".coherence"
    coherence_dir.mkdir(mode=0o700)
    dead_pid = int(exited.stdout)
    (coherence_dir / "server.pid").write_text(f"{dead_pid}\n{free_port}\n", encoding="utf-8")
    return dead_pid, free_port


@pytest.mark.serial
def test_handle_race_over_a_dead_binders_pid_file_reports_one_bound_and_the_rest_joined_same_process(
    workspace: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#348: a binder that exited without ``stop_coordinator`` leaves its pid
    and port in server.pid. Threads of one process race over that file, and
    each losing thread reads it only once the winner's build, held open
    here for 0.3s, has begun: one reports BOUND, and every other joins the
    new coordinator and reports JOINED_SAME_PROCESS on its port.

    Prevents the winner from leaving the dead binder's lines in place until
    its own pid-file write: a thread that lost the flock meanwhile would join
    a port nothing listens on, credited to a pid that no longer runs (or that
    another program now carries), so a host would disown the coordinator its
    own process runs. The reads wait for the build so the race cannot land
    them in the instant between the winner's flock and its clearing of the
    file, which this does not close."""
    dead_pid, dead_port = _leave_a_dead_binders_pid_file(workspace)
    building = threading.Event()
    real_server = lifecycle.CoordinatorHTTPServer
    real_read = lifecycle._read_pidfile

    def slow_server(*args: object, **kwargs: object) -> CoordinatorHTTPServer:
        building.set()
        time.sleep(0.3)  # the winner holds the flock and has not built yet
        return real_server(*args, **kwargs)

    def read_once_the_build_runs(pid_file: Path) -> tuple[int | None, int | None]:
        assert building.wait(timeout=30), "no call started a build"
        return real_read(pid_file)

    monkeypatch.setattr(lifecycle, "CoordinatorHTTPServer", slow_server)
    monkeypatch.setattr(lifecycle, "_read_pidfile", read_once_the_build_runs)
    try:
        answers = _race_in_threads(workspace, _thread_race_cfg(fast_cfg))
    finally:
        stop_coordinator(workspace)
    assert all(
        answer is None or (answer.binder_pid, answer.port) != (dead_pid, dead_port)
        for answer in answers
    ), answers
    _assert_one_bound_and_the_rest_joined_this_process(answers)


def _a_store_the_start_refuses(root: Path) -> None:
    """A store a later build wrote: this build's store open refuses it."""
    import sqlite3

    from ccs.coordinator.sqlite_registry import SCHEMA_USER_VERSION, SqliteArtifactRegistry

    db = root / ".coherence" / "state.db"
    SqliteArtifactRegistry(db).close()
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(f"PRAGMA user_version = {SCHEMA_USER_VERSION + 1}")
    finally:
        conn.close()


def test_a_start_over_a_dead_binders_pid_file_that_the_store_refuses_leaves_it_empty(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#348: a start clears a dead binder's lines before it builds, so when
    the build then raises (here: a store a later build wrote) the pid file
    is left empty, as on a fresh root.

    Prevents both a later joiner reading the dead port and a pid-file reader
    taking this live process, which runs no coordinator, for one: liveness
    readers count a live pid on line 1 as a coordinator even with no port."""
    from ccs.coordinator.sqlite_registry import SchemaVersionError

    _leave_a_dead_binders_pid_file(workspace)
    _a_store_the_start_refuses(workspace)

    with pytest.raises(SchemaVersionError):
        ensure_coordinator_handle(workspace, config=fast_cfg)

    assert (workspace / ".coherence" / "server.pid").read_text(encoding="utf-8") == ""
    assert str(workspace.resolve()) not in lifecycle._SPAWNED_REGISTRY


def test_a_start_the_store_refuses_leaves_a_node_coordinators_pid_file_as_it_was(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#348: the plugin's Node coordinator writes ``backend=node`` on a third
    line of server.pid and holds no flock, so a start takes the flock while
    it runs. A start whose build then raises (a Node store is always
    refused; here a store a later build wrote stands in) leaves its lines in
    place.

    Prevents a failed Python start from blanking a running Node
    coordinator's port: its hook clients would find no coordinator and could
    start a second one on the same store."""
    from ccs.coordinator.sqlite_registry import SchemaVersionError

    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(mode=0o700)
    node_lines = f"{os.getpid()}\n50001\nbackend=node\n"
    (coherence_dir / "server.pid").write_text(node_lines, encoding="utf-8")
    _a_store_the_start_refuses(workspace)

    with pytest.raises(SchemaVersionError):
        ensure_coordinator_handle(workspace, config=fast_cfg)

    assert (coherence_dir / "server.pid").read_text(encoding="utf-8") == node_lines


@pytest.mark.serial
@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() beside the server's threads
def test_handle_in_a_forked_child_reports_the_parents_coordinator_as_another_process(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#348: a child forked after its parent bound the coordinator inherits
    the parent's registry entry (and its bind markers), and its call answers
    from that entry. It must report JOINED_OTHER_PROCESS with the parent's
    pid.

    Prevents a forked worker from taking its parent's coordinator for one it
    started: it would stop it on exit, under the parent and every session
    using it."""
    parent = ensure_coordinator_handle(workspace, config=fast_cfg)
    try:
        assert parent is not None and parent.outcome == _BOUND
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child: report, then leave without stopping anything
            os.close(read_fd)
            try:
                child = ensure_coordinator_handle(workspace, config=fast_cfg)
                report = "none" if child is None else f"{child.outcome.value} {child.binder_pid} {child.port}"
                os.write(write_fd, report.encode())
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        try:
            ready, _, _ = select.select([read_fd], [], [], 10.0)
            reported = os.read(read_fd, 256).decode() if ready else None
        finally:
            os.close(read_fd)
            os.kill(pid, signal.SIGKILL)  # our own child; a no-op once it has exited
            os.waitpid(pid, 0)
    finally:
        stop_coordinator(workspace)
    assert reported == f"{_JOINED_OTHER_PROCESS} {os.getpid()} {parent.port}"


@pytest.mark.parametrize("stopped_by", ["idle_shutdown", "stop_coordinator"])
def test_handle_after_the_coordinator_stopped_reports_bound_again(
    workspace: Path, fast_cfg: LifecycleConfig, stopped_by: str,
) -> None:
    """#348: the handle records what the call did, not whether the
    coordinator still runs. Once the coordinator this process bound has
    stopped (idle shutdown or ``stop_coordinator``), the next call binds a
    new one and reports BOUND, never a join of the stopped one.

    Prevents an answer cached per root or per process: after an idle
    shutdown the host would be told it joined a coordinator that no longer
    exists, and would not own the one that replaced it."""
    idle = stopped_by == "idle_shutdown"
    first = ensure_coordinator_handle(workspace, config=replace(fast_cfg, idle_shutdown_sec=0.5) if idle else fast_cfg)
    assert first is not None and first.outcome == _BOUND
    key = str(workspace.resolve())
    first_coordinator = lifecycle._SPAWNED_REGISTRY[key].coordinator
    if idle:
        assert lifecycle.wait_for_shutdown(workspace, poll_interval_sec=0.05, timeout_sec=10.0) is True
        # The idle thread drops the registry entry just after shutdown_done:
        # wait for that, so the drop cannot land on the next call's entry.
        deadline = time.monotonic() + 5.0
        while key in lifecycle._SPAWNED_REGISTRY and time.monotonic() < deadline:
            time.sleep(0.01)
    else:
        assert stop_coordinator(workspace) is True
    second = ensure_coordinator_handle(workspace, config=fast_cfg)
    try:
        assert second is not None
        assert second.outcome == _BOUND
        assert second.spawned_by_this_call is True
        assert second.binder_pid == os.getpid()
        assert second.port == _read_port_from_file(workspace / ".coherence" / "server.pid")
        assert lifecycle._SPAWNED_REGISTRY[key].coordinator is not first_coordinator
    finally:
        stop_coordinator(workspace)


@pytest.mark.parametrize("stopped_by", ["idle_shutdown", "stop_coordinator"])
def test_a_stopped_coordinators_bind_marker_is_dropped(
    workspace: Path, fast_cfg: LifecycleConfig, stopped_by: str,
) -> None:
    """#348: once the coordinator this process bound has shut down, its bind
    marker is gone.

    Prevents a host that starts and stops coordinators for many workspaces
    from keeping one marker per start for its whole life. Shutdown drops the
    port from the pid file before it finishes, so no later read of the pid
    file can name that port and need the marker."""
    idle = stopped_by == "idle_shutdown"
    handle = ensure_coordinator_handle(workspace, config=replace(fast_cfg, idle_shutdown_sec=0.5) if idle else fast_cfg)
    try:
        assert handle is not None and handle.outcome == _BOUND
        marker = (str(workspace.resolve()), handle.port)
        assert marker in lifecycle._BOUND_IN_THIS_PROCESS
        if idle:
            assert lifecycle.wait_for_shutdown(workspace, poll_interval_sec=0.05, timeout_sec=10.0) is True
        else:
            assert stop_coordinator(workspace) is True
        assert marker not in lifecycle._BOUND_IN_THIS_PROCESS
    finally:
        stop_coordinator(workspace)


def test_a_start_that_fails_after_marking_its_bind_drops_the_marker(
    workspace: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#348: a start that raises after it marked its bind leaves no marker.

    Prevents each failed start from adding a marker that nothing removes."""
    built: list[CoordinatorHTTPServer] = []

    def recording_server(*args, **kwargs) -> CoordinatorHTTPServer:
        built.append(CoordinatorHTTPServer(*args, **kwargs))
        return built[-1]

    def failing_write(fd: int, pid: int, port: int) -> None:
        raise RuntimeError("pid file write failed")

    monkeypatch.setattr(lifecycle, "CoordinatorHTTPServer", recording_server)
    monkeypatch.setattr(lifecycle, "_write_pidfile", failing_write)
    try:
        with pytest.raises(RuntimeError, match="pid file write failed"):
            ensure_coordinator_handle(workspace, config=fast_cfg)
        assert built, "the start failed before it built a coordinator, so it never marked a bind"
        assert (str(workspace.resolve()), built[0].port) not in lifecycle._BOUND_IN_THIS_PROCESS
    finally:
        for coordinator in built:
            coordinator.shutdown()


def test_a_start_that_fails_after_registering_keeps_the_marker(
    workspace: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#348: a start that raises after its coordinator is registered keeps
    the marker, since that coordinator keeps serving under a pid file that
    names its port; its own shutdown drops the marker.

    Prevents the failure path from dropping the marker of a coordinator that
    still serves, which would report a join of it from this process as
    another process's."""
    def failing_threads(entry: object, cfg: LifecycleConfig) -> None:
        raise RuntimeError("background threads failed")

    monkeypatch.setattr(lifecycle, "_start_background_threads", failing_threads)
    with pytest.raises(RuntimeError, match="background threads failed"):
        ensure_coordinator_handle(workspace, config=fast_cfg)
    key = str(workspace.resolve())
    entry = lifecycle._SPAWNED_REGISTRY[key]
    try:
        assert entry.bound_marker == (key, entry.coordinator.port)
        assert entry.bound_marker in lifecycle._BOUND_IN_THIS_PROCESS
    finally:
        # Not stop_coordinator: the failed start already closed the entry's
        # lock fd, and a stop would write to that fd number.
        entry.coordinator.shutdown()
        lifecycle._SPAWNED_REGISTRY.pop(key, None)
        lifecycle._BOUND_IN_THIS_PROCESS.discard(entry.bound_marker)


def _hold_the_flock_with_no_port(root: Path) -> int:
    """Hold server.pid's flock through a separate open in this process, with
    no port line: every call then loses the flock and never reads a port."""
    import fcntl  # POSIX-only; the surrounding suite is POSIX-gated elsewhere

    coherence_dir = root / ".coherence"
    coherence_dir.mkdir(mode=0o700)
    fd = os.open(coherence_dir / "server.pid", os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@pytest.mark.parametrize("cause", ["no_fcntl", "read_only_root", "join_retry_budget_exhausted"])
def test_handle_is_none_exactly_where_ensure_coordinator_returns_minus_one(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, cause: str,
) -> None:
    """#348: where :func:`ensure_coordinator` degrades to -1 (here: no fcntl,
    a read-only repo, and a join whose retry budget runs out), the handle
    form returns None rather than raising, and the int form still returns a
    plain ``int`` -1.

    Prevents the handle form from raising where hooks and hosts rely on a
    quiet degrade, and the int form from changing what ``== -1`` sees."""
    root = tmp_path / "ws"
    root.mkdir()
    cfg = replace(fast_cfg, port_file_retry_attempts=3, port_file_retry_interval_sec=0.01)
    hold_fd = None
    if cause == "no_fcntl":
        monkeypatch.setattr(lifecycle, "_FCNTL_AVAILABLE", False)
    elif cause == "read_only_root":
        os.chmod(root, 0o500)  # .coherence/ cannot be created
    else:
        hold_fd = _hold_the_flock_with_no_port(root)
    try:
        assert ensure_coordinator_handle(root, config=cfg) is None
        port = ensure_coordinator(root, config=cfg)
        assert type(port) is int and port == -1
    finally:
        os.chmod(root, 0o700)
        if hold_fd is not None:
            os.close(hold_fd)  # the only open of it: closing releases the flock
        lifecycle._reset_spawn_join_exhaustion()


def test_ensure_coordinator_still_returns_a_plain_int_beside_the_handle(
    workspace: Path, fast_cfg: LifecycleConfig,
) -> None:
    """#348 adds the handle as a sibling and leaves :func:`ensure_coordinator`
    returning an ``int``. Prevents the return type from changing under its
    callers: a handle object there would make every ``port == -1`` check
    False, so a failed spawn would read as a coordinator to connect to."""
    port = ensure_coordinator(workspace, config=fast_cfg)
    try:
        assert type(port) is int and port > 0
        handle = ensure_coordinator_handle(workspace, config=fast_cfg)
        assert handle is not None
        assert (handle.port, handle.outcome) == (port, _JOINED_SAME_PROCESS)
    finally:
        stop_coordinator(workspace)


def test_connect_or_spawn_spawns_through_the_module_level_ensure_coordinator(
    workspace: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#348 moved the spawn body into ensure_coordinator_handle; connect_or_spawn
    must still reach it through the module-level ``ensure_coordinator`` name.
    Prevents the cross-process conformance suite's spawn counter, which wraps
    that name, from silently counting zero spawns."""
    calls: list[Path] = []
    real_ensure = lifecycle.ensure_coordinator

    def counted(coordinator_root: Path, **kwargs: object) -> int:
        calls.append(coordinator_root)
        return real_ensure(coordinator_root, **kwargs)

    monkeypatch.setattr(lifecycle, "ensure_coordinator", counted)
    port = connect_or_spawn(workspace, config=fast_cfg)
    try:
        assert port > 0
        assert calls == [workspace]
    finally:
        stop_coordinator(workspace)


@pytest.mark.parametrize(
    ("binder_pid_offset", "marker", "expected"),
    [
        (0, ("/ws", 50001), _JOINED_SAME_PROCESS),
        (0, None, _JOINED_OTHER_PROCESS),
        (1, ("/ws", 50001), _JOINED_OTHER_PROCESS),
        (None, ("/ws", 50001), _JOINED_OTHER_PROCESS),
        (0, ("/ws", 50002), _JOINED_OTHER_PROCESS),
        (0, ("/other", 50001), _JOINED_OTHER_PROCESS),
    ],
    ids=[
        "pid_and_marker_match",
        "pid_match_without_marker",  # a binder in another PID namespace carrying this pid
        "marker_with_pid_mismatch",  # a forked child: it inherits the marker, not the pid
        "unparsable_pid",
        "marker_for_another_port",  # an earlier bind of this root, since stopped
        "marker_for_another_root",
    ],
)
def test_classify_join_reports_same_process_only_on_this_pid_and_this_processs_bind_marker(
    monkeypatch: pytest.MonkeyPatch,
    binder_pid_offset: int | None,
    marker: tuple[str, int] | None,
    expected: str,
) -> None:
    """#348: a pid-file join is JOINED_SAME_PROCESS only when the pid file
    names this process AND this process marked that root and port when it
    bound them.

    Prevents either half from deciding alone: a pid match alone misreads a
    binder in another PID namespace sharing the volume, and a marker alone
    misreads the parent's coordinator in a forked child, which inherits the
    marker set."""
    monkeypatch.setattr(lifecycle, "_BOUND_IN_THIS_PROCESS", set() if marker is None else {marker})
    binder_pid = None if binder_pid_offset is None else os.getpid() + binder_pid_offset
    assert lifecycle._classify_join("/ws", binder_pid, 50001) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("4242\n50001\n", (4242, 50001)),
        ("4242\n", (4242, None)),  # mid-bind, or after shutdown dropped the port
        ("", (None, None)),
        ("4242\nx\n", (4242, None)),
        ("4242\n70000\n", (4242, None)),  # out of range: the port check is unchanged
        ("x\n50001\n", (None, 50001)),  # joinable, but not attributable
        ("0\n50001\n", (None, 50001)),
        (None, (None, None)),  # no pid file
    ],
)
def test_read_pidfile_takes_pid_and_port_from_one_read(
    tmp_path: Path, content: str | None, expected: tuple[int | None, int | None],
) -> None:
    """#348: a join is attributed to the pid on the line above the port it
    returns, taken from the same read; a malformed pid line costs the
    attribution, never the join. ``read_port_from_file`` answers exactly the
    port half, as it always has.

    Prevents a second read from pairing a port with the pid of a different
    bind: the file is rewritten on every bind and every shutdown."""
    pid_file = tmp_path / "server.pid"
    if content is not None:
        pid_file.write_text(content, encoding="utf-8")
    assert lifecycle._read_pidfile(pid_file) == expected
    assert _read_port_from_file(pid_file) == expected[1]


# ----------------------------------------------------------------------
# #345: a spawn that migrates the store says so, once
# ----------------------------------------------------------------------

#: FROZEN duplicate of the migration line's opening words (#345).
_MIGRATED_LINE = "coordinator migrated "
_LIFECYCLE_LOGGER = "ccs.adapters.claude_code.lifecycle"


def _migration_records(caplog: pytest.LogCaptureFixture) -> list[tuple[str, str]]:
    """``(logger name, message)`` of every migration line captured, from ANY
    logger: a copy of the line on the registry's logger, which check_store's
    dry run reaches, must count too."""
    return [
        (record.name, record.getMessage())
        for record in caplog.records
        if record.getMessage().startswith(_MIGRATED_LINE)
    ]


def _v7_store(workspace: Path) -> Path:
    """A store the current build migrates forward: written at the current
    schema, then put back into the version-7 shape."""
    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
    from tests.coordinator.test_checkpoint_registration_schema import _revert_to_v7_shape

    coherence = workspace / ".coherence"
    coherence.mkdir(mode=0o700, parents=True, exist_ok=True)
    db = coherence / "state.db"
    SqliteArtifactRegistry(db).close()
    _revert_to_v7_shape(db)
    return db


def test_a_spawn_that_migrates_the_store_logs_both_versions_once(
    workspace: Path, fast_cfg: LifecycleConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """A spawn on a version-7 store migrates it to the current schema and
    logs exactly one INFO line naming the store and both versions (#345),
    on the lifecycle logger and on no other.

    Prevents a one-way migration happening silently: once it runs, a
    coordinator that expects the old schema refuses the store, and before
    this line the spawn logged only that it started. Captured at the root,
    so a second copy on another logger, or the line moving off the
    lifecycle logger, fails here too."""
    from ccs.coordinator.sqlite_registry import SCHEMA_USER_VERSION

    db = _v7_store(workspace)
    caplog.set_level("INFO")
    try:
        assert ensure_coordinator(workspace, config=fast_cfg) > 0
    finally:
        stop_coordinator(workspace)

    records = _migration_records(caplog)
    assert [name for name, _ in records] == [_LIFECYCLE_LOGGER], records
    message = records[0][1]
    assert f"from registry schema 7 to {SCHEMA_USER_VERSION}" in message
    assert str(db) in message


@pytest.mark.parametrize("store", ["fresh", "current"])
def test_a_spawn_that_migrates_nothing_logs_no_migration(
    workspace: Path, fast_cfg: LifecycleConfig, caplog: pytest.LogCaptureFixture, store: str
) -> None:
    """A spawn that creates the store, or opens one already at the current
    schema, migrates nothing and logs no migration line."""
    if store == "current":
        from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry

        coherence = workspace / ".coherence"
        coherence.mkdir(mode=0o700, parents=True, exist_ok=True)
        SqliteArtifactRegistry(coherence / "state.db").close()
    caplog.set_level("INFO")
    try:
        assert ensure_coordinator(workspace, config=fast_cfg) > 0
    finally:
        stop_coordinator(workspace)

    assert _migration_records(caplog) == []


def test_checking_a_store_that_would_migrate_logs_no_migration(
    workspace: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """check_store dry-runs the migration on a copy; it reports
    ``will_migrate`` and logs no migration line on any logger, so the line
    means a store really moved.

    Prevents the line moving into, or being copied into, the registry's
    migration path: the dry run reaches that path, never the lifecycle
    module, so a filter on the lifecycle logger would not see it."""
    from ccs.adapters.claude_code.store_check import check_store

    _v7_store(workspace)
    caplog.set_level("DEBUG")

    result = check_store(workspace)

    assert result.outcome.value == "will_migrate"
    assert _migration_records(caplog) == []
