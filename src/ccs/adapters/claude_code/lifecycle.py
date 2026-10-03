# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Race-safe lazy spawn, idle shutdown, and background sweep for the
Claude Code coordinator HTTP server (Unit 5 per the v0.1 plan).

The plugin's hook scripts call :func:`connect_or_spawn` on every hook
event. The first call from any session in a workspace lazily spawns the
coordinator process; subsequent calls (from the same or other sessions)
read the existing port file and skip spawn.

Key correctness properties:
- **Single binder per workspace.** `fcntl.flock(server.pid, LOCK_EX|LOCK_NB)`
  ensures exactly one process at a time owns the coordinator. POSIX-only;
  Windows fallback (KTD-5) is deferred to v0.1.1.
- **No port-file TOCTOU.** The holder binds the ``ThreadingHTTPServer``
  FIRST (port=0 lets the OS pick), reads ``server.server_port``, writes
  ``<pid>\\n<port>\\n`` to ``server.pid``, fsyncs, THEN starts the serving
  loop and the sweep thread. Losers' bounded retry reads the port once
  it appears.
- **Race-safe idle shutdown.** Shutdown rewrites the port file to drop
  the port line BEFORE releasing the flock, so a concurrent
  :func:`ensure_coordinator` that acquires the flock right after release
  sees a port-less file and re-spawns cleanly instead of returning a
  dead port.
"""

from __future__ import annotations

import errno
import logging
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer

logger = logging.getLogger(__name__)


# L3 observability (companion to the deferred budget retune): the spawn-or-join
# loop can exhaust its retry/inode budget under a cold-start thundering herd and
# return -1, degrading the caller. Each exhaustion already logs a warning, but a
# per-process, per-reason aggregate lets an operator measure how often — and
# why — without grepping logs (the real-world p99 data the port_file_retry
# budget comment is waiting for before any deterministic retune). Telemetry
# only; never changes control flow. Per-reason (not one total) so herd
# exhaustion ("inode_budget"/"retry_budget") stays distinguishable from the
# dying-coordinator probe failure ("loser_probe_failed") — a different failure
# mode that would otherwise pollute the retune signal.
_EXHAUSTION_LOCK = threading.Lock()
_spawn_join_exhaustion_by_reason: dict[str, int] = {}


def _record_spawn_join_exhaustion(reason: str) -> None:
    """Increment the process-lifetime exhaustion counter for ``reason`` (one of
    ``"inode_budget"``, ``"retry_budget"``, ``"loser_probe_failed"``)."""
    with _EXHAUSTION_LOCK:
        _spawn_join_exhaustion_by_reason[reason] = (
            _spawn_join_exhaustion_by_reason.get(reason, 0) + 1
        )


def get_spawn_join_exhaustion_total() -> int:
    """Process-lifetime count of spawn/join exhaustion -1 degrades (all reasons).

    Surfaces the otherwise-invisible loser-degrade path (L3). Use
    :func:`get_spawn_join_exhaustion_by_reason` to separate herd exhaustion
    (``inode_budget``/``retry_budget``) from the dying-coordinator probe failure
    (``loser_probe_failed``)."""
    with _EXHAUSTION_LOCK:
        return sum(_spawn_join_exhaustion_by_reason.values())


def get_spawn_join_exhaustion_by_reason() -> dict[str, int]:
    """Per-reason snapshot of the spawn/join exhaustion counter (L3)."""
    with _EXHAUSTION_LOCK:
        return dict(_spawn_join_exhaustion_by_reason)


def _reset_spawn_join_exhaustion() -> None:  # pragma: no cover - test hook
    """Test-only: clear the counter so tests don't see cross-test bleed."""
    with _EXHAUSTION_LOCK:
        _spawn_join_exhaustion_by_reason.clear()


def _reset_exhaustion_lock_in_child() -> None:
    # A fork while another thread is counting an exhaustion copies this lock
    # held, and no thread in the child will ever release it: the child's next
    # exhaustion (a post-fork re-attach can reach one) or counter read would
    # block forever. Each count is a single dict store, so the child keeps them.
    global _EXHAUSTION_LOCK
    _EXHAUSTION_LOCK = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_exhaustion_lock_in_child)


# G8 fix (subagent finding #8): fcntl is POSIX-only. On Windows, import-time
# failure would crash hook handlers with a stack trace on every hook event.
# Guard the import and provide a stub that degrades gracefully — hook handlers
# already treat -1 as "no coordinator available" and no-op.
try:
    import fcntl  # type: ignore[import-not-found]
    _FCNTL_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised only on Windows
    fcntl = None  # type: ignore[assignment]
    _FCNTL_AVAILABLE = False
    logger.warning(
        "fcntl not available (platform=%s); agent-coherence coordinator is disabled. "
        "Use WSL2 on Windows. Native Windows support tracked in v0.1.1.",
        sys.platform,
    )


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LifecycleConfig:
    """Tunables for the spawn / sweep / shutdown loops.

    Defaults are chosen for a single-developer interactive session on
    macOS; the 10-process race test in :mod:`test_claude_code_lifecycle`
    measures real cold-start p99 and confirms the port-file retry budget
    covers it.
    """

    #: Wall-clock seconds of inactivity before the coordinator self-stops.
    #: 0 disables idle shutdown (tests, long-running benchmarks).
    idle_shutdown_sec: float = 900.0

    #: How often the idle-shutdown watcher and grant-timeout sweep tick.
    #: 0 disables the sweep entirely.
    sweep_interval_sec: float = 5.0

    #: F2 hardening — orphan preemption notices older than this are
    #: evicted by the sweep. 30 min default: long enough for a model to
    #: pause and come back, short enough to bound state on a dead session.
    notice_evict_max_age_sec: float = 1800.0

    #: Loser's bounded retry when reading the port file. Bumped from 30 to
    #: 60 (G9) so a 30-process thundering herd with cold Python imports
    #: and SQLite WAL setup has time to settle before losers degrade.
    #: Full mitigation requires measuring real-world p99 cold-start —
    #: deferred to v0.1.1 (docs/known-issues).
    port_file_retry_attempts: int = 60
    port_file_retry_interval_sec: float = 0.050  # × 60 = 3000ms budget

    #: Hook handler's TCP connect retry before falling back to spawn.
    connect_retry_attempts: int = 3
    connect_retry_interval_sec: float = 0.100

    #: G2 fix — the spawn-side self-probe budget is independent and much
    #: larger because the spawning process knows it just bound the socket
    #: and can afford to wait for serve_forever to reach accept(). Cold
    #: start can take well over 300ms on a slow disk.
    spawn_self_probe_attempts: int = 50  # × 100ms = 5000ms budget
    spawn_self_probe_interval_sec: float = 0.100

    #: Grant-timeout sweep thresholds. With v0.1's CrashRecoveryConfig
    #: shipping disabled-by-default, these effectively define the sweep's
    #: safety net for genuinely orphaned grants. Generous so a thinking
    #: session is never reclaimed under interactive load.
    grant_heartbeat_timeout_sec: int = 600
    grant_max_hold_sec: int = 1800

    #: Transient-state timeout (fail-safe for unfinished M↔E protocol steps).
    transient_timeout_sec: int = 60

    #: KTD-H — Unit 5 L1: cap on inode re-opens during one ``ensure_coordinator``
    #: call. Each external ``rm -rf .coherence/ && recreate`` consumes one
    #: revalidation. Churn beyond this budget indicates a runaway external
    #: process; the coordinator returns -1 and lets the hook degrade.
    inode_revalidation_budget: int = 5


_DEFAULT_CONFIG = LifecycleConfig()


# ----------------------------------------------------------------------
# Public API: spawn / connect / shutdown
# ----------------------------------------------------------------------


def ensure_coordinator(
    coordinator_root: Path,
    *,
    config: LifecycleConfig | None = None,
    bind_host: str = "127.0.0.1",
) -> int:
    """Lazy-spawn entry point.

    Acquires the fcntl exclusive lock on ``<root>/.coherence/server.pid``.
    If acquired, binds the HTTP server, writes the port file, starts
    serving in a daemon thread, and starts the sweep + idle-shutdown
    threads. If not acquired, reads the existing port file (with bounded
    retry for the brief window where the holder hasn't written it yet).

    Returns the port the coordinator is bound to. Returns ``-1`` if the
    parent repo is read-only and the coordinator cannot be spawned — the
    caller (hook handler) treats this as "no coordinator available" and
    degrades gracefully.
    """
    if not _FCNTL_AVAILABLE:
        return -1

    cfg = config or _DEFAULT_CONFIG
    coherence_dir = _ensure_coherence_dir(coordinator_root)
    if coherence_dir is None:
        return -1

    # G3 entry short-circuit: if this process already spawned a coordinator
    # for this workspace and it's still healthy, return its port directly.
    # Prevents fd / sweep-thread leaks from accidental re-entrant calls.
    resolved_key = str(coordinator_root.resolve())
    existing = _SPAWNED_REGISTRY.get(resolved_key)
    if existing is not None and not existing.shutdown_done.is_set():
        existing_port = existing.coordinator.port
        if tcp_probe(existing_port, cfg, bind_host=bind_host):
            return existing_port
        # Existing entry not actually reachable — fall through to respawn.
        logger.warning(
            "ensure_coordinator: existing entry for %s port=%d not reachable; respawning",
            resolved_key, existing_port,
        )

    pid_file = coherence_dir / "server.pid"
    fd = _open_pidfile(pid_file)
    if fd is None:
        return -1

    # Unified spawn-or-join loop (G1 fix per Unit 5 §5.654 — the
    # idle-shutdown-vs-spawn race). On each attempt:
    #   0. KTD-H (Unit 5 L1): revalidate fd's inode against the on-disk
    #      path. If an external process unlinked `.coherence/` and
    #      recreated it mid-retry, our fd points at the orphaned inode
    #      and any flock/write happens invisibly. Re-open and restart
    #      the retry counter on mismatch (bounded by the revalidation
    #      budget to defend against pathological churn).
    #   1. Try to acquire the flock (non-blocking). If acquired, we're
    #      the winner — bind, write port, serve, return.
    #   2. If contended, try to read a valid port from the file. If
    #      present, we're a clean loser — return the holder's port.
    #   3. Otherwise the holder is either mid-bind (will write the port
    #      shortly) OR mid-shutdown (will release the lock shortly).
    #      Sleep one interval and retry both checks.
    #
    # This handles both the cold-start thundering herd (holder is
    # mid-bind; port appears) and the idle-shutdown race (holder is
    # mid-shutdown; flock releases) without baking in an order, plus
    # the rm -rf race that KTD-H closes.
    revalidations_remaining = cfg.inode_revalidation_budget
    attempt = 0
    while attempt < cfg.port_file_retry_attempts:
        # KTD-H: per-iteration inode revalidation. Cheap (~50μs) — st_dev
        # + st_ino comparison detects unlink-and-recreate races before the
        # flock attempt commits us to an orphan.
        if not _inode_matches(fd, pid_file):
            if revalidations_remaining <= 0:
                logger.warning(
                    "ensure_coordinator: inode revalidation budget exhausted "
                    "(%d revalidations); external churn on .coherence/ — giving up",
                    cfg.inode_revalidation_budget,
                )
                _record_spawn_join_exhaustion("inode_budget")  # L3 observability
                _close_quiet(fd)
                return -1
            revalidations_remaining -= 1
            logger.info(
                "ensure_coordinator: server.pid inode mismatch (rm -rf race?); "
                "re-opening (%d revalidations remaining)",
                revalidations_remaining,
            )
            _close_quiet(fd)
            # Re-create the .coherence dir in case the rm -rf removed it too.
            new_coherence_dir = _ensure_coherence_dir(coordinator_root)
            if new_coherence_dir is None:
                return -1
            coherence_dir = new_coherence_dir
            pid_file = coherence_dir / "server.pid"
            new_fd = _open_pidfile(pid_file)
            if new_fd is None:
                return -1
            fd = new_fd
            attempt = 0  # KTD-H: restart the retry counter on re-open
            continue

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EWOULDBLOCK, errno.EACCES, errno.EAGAIN):
                _close_quiet(fd)
                raise
            # Contended — try to read the port.
            port = read_port_from_file(pid_file)
            if port is not None:
                _close_quiet(fd)
                return port
            # Port file empty (holder mid-bind or mid-shutdown). Wait
            # and try the whole loop again — on the next iteration we
            # may either see the port populated OR acquire the released
            # lock ourselves.
            time.sleep(cfg.port_file_retry_interval_sec)
            attempt += 1
            continue

        # Winner — we hold the lock. Bind, write port, serve.
        # KTD-H/I/L3 (Unit 5 L3): time the winner path so operators have a
        # signal for cold-start regressions. Telemetry-only — no default
        # behavior change. Surfaced via the coordinator's
        # ``cold_start_duration_ms`` attribute for the future /status
        # endpoint (Unit 8).
        cold_start_start = time.monotonic()
        try:
            coordinator = CoordinatorHTTPServer(coordinator_root, port=0, bind_host=bind_host)
            port = coordinator.port
            # KTD-H Unit 5 L1: close the construct->write TOCTOU window. The
            # top-of-loop inode check ran BEFORE this construction, which opens
            # state.db (+WAL) and binds the TCP socket (>300ms cold). An external
            # `rm -rf .coherence/ && recreate` landing DURING construction would
            # leave us about to write the port into an ORPHANED server.pid that no
            # concurrent reader can see — losers read the recreated, port-less file
            # and degrade ("spawned a coordinator no one can reach"). Re-validate
            # the inode now, before committing any bytes. On mismatch: tear down
            # the just-bound coordinator to free the socket (shutdown() is safe
            # pre-serve — it skips _server.shutdown when no serve thread exists,
            # then server_close + registry.close), release the flock, and restart
            # the loop; the top-of-loop revalidation re-opens on the fresh inode
            # and consumes one revalidation budget.
            if not _inode_matches(fd, pid_file):
                logger.info(
                    "ensure_coordinator: server.pid inode changed during coordinator "
                    "construction (rm -rf race in the construct->write window); "
                    "tearing down and revalidating"
                )
                coordinator.shutdown()
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                continue
            _write_pidfile(fd, os.getpid(), port)
            coordinator.serve_in_thread()
            entry = _SpawnedEntry(
                coordinator=coordinator,
                lock_fd=fd,
                coherence_dir=coherence_dir,
                shutdown_lock=threading.Lock(),
                shutdown_done=threading.Event(),
            )
            _SPAWNED_REGISTRY[resolved_key] = entry
            _start_background_threads(entry, cfg)
            # G2 fix: self-probe with generous spawn-side budget before
            # returning so the caller is guaranteed a coordinator that
            # is actually accepting. Cold-start (Python interpreter +
            # SQLite WAL rehydration) can take well past the loser-side
            # connect_retry budget.
            probe_ok = _self_probe(port, cfg, bind_host=bind_host)
            cold_start_ms = (time.monotonic() - cold_start_start) * 1000.0
            coordinator.cold_start_duration_ms = cold_start_ms
            if not probe_ok:
                logger.warning(
                    "coordinator bound port=%d but self-probe exhausted after %dms; returning anyway",
                    port,
                    int(cfg.spawn_self_probe_attempts * cfg.spawn_self_probe_interval_sec * 1000),
                )
            logger.info(
                "coordinator spawned: pid=%d port=%d root=%s cold_start=%.1fms",
                os.getpid(), port, coordinator_root, cold_start_ms,
            )
            return port
        except Exception:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                _close_quiet(fd)
            raise

    # Retry budget exhausted without acquiring the lock or seeing a
    # populated port. Operationally this means the holder is wedged
    # between flock-hold and port-write for the full retry budget —
    # which would itself be a bug in this module — OR the holder is
    # mid-shutdown for longer than the retry budget. Either way, return
    # -1 so the caller degrades gracefully.
    _close_quiet(fd)
    logger.warning(
        "ensure_coordinator: %d attempts exhausted without acquiring lock or reading port",
        cfg.port_file_retry_attempts,
    )
    _record_spawn_join_exhaustion("retry_budget")  # L3 observability
    return -1


def connect_or_spawn(
    coordinator_root: Path,
    *,
    config: LifecycleConfig | None = None,
    bind_host: str = "127.0.0.1",
) -> int:
    """Hook-handler entry point.

    Read the port file → TCP-probe the port → on connect failure, call
    :func:`ensure_coordinator` once and retry the probe.
    """
    if not _FCNTL_AVAILABLE:
        return -1

    cfg = config or _DEFAULT_CONFIG
    pid_file = coordinator_root / ".coherence" / "server.pid"
    port = read_port_from_file(pid_file)
    if port is not None and tcp_probe(port, cfg, bind_host=bind_host):
        return port

    # Stale or absent — spawn (or join existing holder via fcntl race).
    port = ensure_coordinator(coordinator_root, config=cfg, bind_host=bind_host)
    if port == -1:
        return -1
    # ensure_coordinator already self-probes on the spawn path; here we
    # only re-probe if the caller landed in the loser-read path (where
    # the just-read port may belong to a coordinator mid-shutdown).
    if not tcp_probe(port, cfg, bind_host=bind_host):
        logger.warning("coordinator spawned at port=%d but TCP probe failed", port)
        # L3 observability: distinct reason — this is a stale/dying coordinator
        # (re-probe of an already-read port failed), NOT herd budget exhaustion.
        _record_spawn_join_exhaustion("loser_probe_failed")
        return -1
    return port


def stop_coordinator(coordinator_root: Path) -> bool:
    """Race-safe in-process shutdown.

    Returns True if a coordinator was running in *this* process and was
    cleanly stopped (the caller's invocation actually executed the
    sequence). Returns False if no such coordinator exists locally OR
    if idle-shutdown already completed it (caller's intent is fulfilled
    either way; the return distinguishes who did the work).
    """
    key = str(Path(coordinator_root).resolve())
    entry = _SPAWNED_REGISTRY.get(key)
    if entry is None:
        return False
    ran = _shutdown_sequence(entry)
    # Only pop the registry if the shutdown actually completed successfully.
    # If shutdown_sequence aborted (e.g. coordinator.shutdown raised) we leave
    # the entry in place so a subsequent stop_coordinator call can retry.
    if entry.shutdown_done.is_set():
        _SPAWNED_REGISTRY.pop(key, None)
    return ran


def wait_for_shutdown(
    coordinator_root: Path,
    *,
    poll_interval_sec: float = 1.0,
    timeout_sec: float | None = None,
) -> bool:
    """KP-7 public API. Block until the coordinator at ``coordinator_root``
    has reached shutdown_done (idle-shutdown completed, stop_coordinator
    called, or shutdown raised). Returns True if shutdown completed, False
    if ``timeout_sec`` elapsed first or no in-process coordinator entry
    exists for this root.

    Designed for the ``agent-coherence-coordinator --_daemonized`` worker
    that needs to keep the main thread alive until daemon threads have
    cleanly torn down. Replaces direct reach-into ``_SPAWNED_REGISTRY``
    + ``entry.shutdown_done.is_set()`` polling.

    KeyboardInterrupt during the wait raises through to the caller so a
    SIGINT can trigger the caller's own ``stop_coordinator`` shutdown
    path (the wait itself never initiates shutdown — it only observes).
    """
    key = str(Path(coordinator_root).resolve())
    entry = _SPAWNED_REGISTRY.get(key)
    if entry is None:
        return False
    deadline: float | None = None
    if timeout_sec is not None:
        deadline = time.monotonic() + timeout_sec
    while not entry.shutdown_done.is_set():
        if deadline is not None and time.monotonic() >= deadline:
            return False
        time.sleep(poll_interval_sec)
    return True


# ----------------------------------------------------------------------
# Internals — pid file, port file, sockets
# ----------------------------------------------------------------------


#: REL-06: cap on consecutive G4 aborts (coordinator.shutdown() raises)
#: before the lifecycle gives up on a clean teardown and force-closes the
#: flock so a fresh ensure_coordinator can re-spawn. The current G4
#: behaviour holds the flock indefinitely on every abort — that protects
#: in-flight handler state but means a wedged shutdown blocks all future
#: spawns for the workspace. Capping at 3 attempts (≈3 sweep ticks) gives
#: the coordinator a real chance to drain before we trade safety for
#: liveness.
_MAX_SHUTDOWN_RETRIES_BEFORE_ESCALATION = 3


@dataclass
class _SpawnedEntry:
    """Per-spawn state kept by the spawn-side process.

    The shutdown_lock + shutdown_done pair (G6 fix per subagent finding #6)
    mutexes the shutdown sequence so concurrent triggers (stop_coordinator +
    _idle_shutdown_loop) cannot interleave pid-file writes or double-close
    the lock_fd. The first caller acquires the lock, runs the sequence,
    sets shutdown_done; subsequent callers acquire the lock, see done=True,
    return immediately.

    REL-06: ``shutdown_abort_count`` tracks consecutive G4 aborts so the
    idle loop can escalate after a bounded number of failures rather than
    holding the flock indefinitely on a wedged shutdown.
    """

    coordinator: CoordinatorHTTPServer
    lock_fd: int
    coherence_dir: Path
    shutdown_lock: threading.Lock
    shutdown_done: threading.Event
    shutdown_abort_count: int = 0


#: Maps coordinator_root → _SpawnedEntry. Only the spawn-side ever populates
#: this; loser-side and other-process paths don't have a Coordinator instance
#: to manage.
_SPAWNED_REGISTRY: dict[str, _SpawnedEntry] = {}


def _ensure_coherence_dir(coordinator_root: Path) -> Path | None:
    """Create ``<root>/.coherence/`` with mode 0700, re-tightening an existing
    dir to 0700 if it pre-exists at a looser mode. Returns the dir path, or None
    if the parent repo is read-only.

    Also writes ``.coherence/.gitignore`` containing ``*`` per KTD-13 so
    a careless ``git add .`` doesn't accidentally commit the SQLite
    state.db (containing MESI state + agent UUIDs), the hook.secret
    (a credential), or the server.pid file. The README claims these are
    auto-gitignored — this is the implementation.
    """
    coherence_dir = coordinator_root / ".coherence"
    try:
        coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError as exc:
        logger.warning(
            "cannot create .coherence directory under %s: %s — coordinator disabled",
            coordinator_root, exc,
        )
        return None
    # ``mkdir(mode=0o700)`` does NOT chmod an ALREADY-EXISTING directory, so a
    # ``.coherence/`` that pre-existed at a looser mode (e.g. 0755 from a careless
    # ``mkdir`` or an older tool) would keep it — leaving the SQLite state.db,
    # the hook.secret credential, and the pidfile group/world-accessible.
    # Re-assert the 0700 guarantee unconditionally (idempotent); warn once when
    # we actually had to tighten an existing dir from a more-permissive mode.
    try:
        prior_mode = coherence_dir.stat().st_mode & 0o777
        if prior_mode != 0o700:
            os.chmod(coherence_dir, 0o700)
            if prior_mode & 0o077:
                logger.warning(
                    "tightened existing .coherence directory %s from mode %o to "
                    "0700 (it exposed group/world bits; the SQLite state, the "
                    "hook.secret credential and the pidfile must not be readable "
                    "outside the owner)",
                    coherence_dir, prior_mode,
                )
    except OSError as exc:
        logger.warning(
            "could not re-assert 0700 on .coherence directory %s: %s",
            coherence_dir, exc,
        )
    # Write .gitignore (idempotent — only write if missing to avoid
    # clobbering any operator customization)
    gitignore = coherence_dir / ".gitignore"
    if not gitignore.exists():
        try:
            gitignore.write_text("*\n")
        except OSError as exc:
            logger.warning(
                "could not write %s: %s — workspace data risks being committed",
                gitignore, exc,
            )
    return coherence_dir


def _open_pidfile(pid_file: Path) -> int | None:
    """Open (creating if needed) the pid file with mode 0600 for fcntl use."""
    try:
        fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o600)
    except OSError as exc:
        logger.warning("cannot open pid file %s: %s", pid_file, exc)
        return None
    return fd


def _close_quiet(fd: int) -> None:
    """Best-effort close. Swallows OSError so cleanup paths can chain
    without separate try/except blocks — closing an already-bad fd just
    means the underlying file was already gone."""
    try:
        os.close(fd)
    except OSError:
        pass


def _inode_matches(fd: int, path: Path) -> bool:
    """KTD-H Unit 5 L1 helper. Returns True iff the file currently at
    ``path`` shares (st_dev, st_ino) with the open ``fd``. Used to detect
    the unlink-and-recreate race where an external ``rm -rf .coherence/``
    leaves our fd orphaned on a no-longer-reachable inode.

    Returns False on any stat/fstat failure or mismatch — caller treats
    that as "revalidate" rather than trying to disambiguate. Cost is
    roughly two syscalls (~50μs) per call, executed once per retry
    iteration."""
    try:
        fd_stat = os.fstat(fd)
    except OSError:
        return False
    try:
        path_stat = os.stat(path)
    except OSError:
        # Path was unlinked, or its parent dir was. Definitely mismatch.
        return False
    return (fd_stat.st_dev, fd_stat.st_ino) == (path_stat.st_dev, path_stat.st_ino)


def _write_pidfile(fd: int, pid: int, port: int) -> None:
    """Replace the pid file's contents with ``<pid>\\n<port>\\n`` and fsync.
    The fd must hold the exclusive flock."""
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    payload = f"{pid}\n{port}\n".encode("utf-8")
    written = 0
    while written < len(payload):
        n = os.write(fd, payload[written:])
        if n == 0:  # defensive — write should always make progress
            break
        written += n
    os.fsync(fd)


def _rewrite_pidfile_drop_port(fd: int, pid: int) -> None:
    """Idle-shutdown step: rewrite the pid file with just ``<pid>\\n``.
    Callers must hold the flock. A subsequent ensure_coordinator call
    that acquires the lock will see an empty port and re-spawn."""
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, f"{pid}\n".encode("utf-8"))
    os.fsync(fd)


def read_port_from_file(pid_file: Path) -> int | None:
    """Read the port line from the pid file. Returns None if absent,
    empty, malformed, or the file doesn't exist.

    Public API (promoted from the private ``read_port_from_file`` per
    P2 ce-review fix #17 / maintainability + kieran-python). CLI scripts
    and external callers should use this function rather than reaching
    into the underscore-prefixed name."""
    try:
        text = pid_file.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    lines = text.splitlines()
    if len(lines) < 2:
        return None
    try:
        port = int(lines[1].strip())
    except ValueError:
        return None
    if not (1 <= port <= 65535):
        return None
    return port


def _read_port_with_retry(pid_file: Path, cfg: LifecycleConfig) -> int:
    """Bounded retry for the brief window where the holder has the lock
    but hasn't written the port yet. Returns -1 if the retry exhausts.

    P2 ce-review fix #20 (maintainability + reliability): kept as a
    TEST-ONLY utility. Production callers use the unified spawn-or-join
    loop in :func:`ensure_coordinator` which handles the same retry
    pattern alongside flock acquisition (G1 fix). Do not call from
    production code paths — the inlined version covers both port-read
    and lock-acquire retries simultaneously."""
    for _ in range(cfg.port_file_retry_attempts):
        port = read_port_from_file(pid_file)
        if port is not None:
            return port
        time.sleep(cfg.port_file_retry_interval_sec)
    logger.warning(
        "loser path: port file %s never populated within %dms",
        pid_file,
        int(cfg.port_file_retry_attempts * cfg.port_file_retry_interval_sec * 1000),
    )
    return -1


def tcp_probe(port: int, cfg: LifecycleConfig, *, bind_host: str = "127.0.0.1") -> bool:
    """Loser-side / generic TCP probe with the connect_retry budget. Used
    by hook-handler-style callers that have already paid a port-read.

    Public API (promoted from the private ``tcp_probe`` per P2 ce-review
    fix #17). The underscore-prefixed alias is retained for backward
    compatibility with internal callers."""
    return _probe_with_budget(
        port, bind_host, cfg.connect_retry_attempts, cfg.connect_retry_interval_sec
    )


def _self_probe(port: int, cfg: LifecycleConfig, *, bind_host: str = "127.0.0.1") -> bool:
    """Spawn-side self-probe with the much larger spawn budget. G2 fix:
    the spawning process knows it just bound the socket and can afford to
    wait for the daemon thread to reach serve_forever's accept loop."""
    return _probe_with_budget(
        port, bind_host, cfg.spawn_self_probe_attempts, cfg.spawn_self_probe_interval_sec
    )


def _probe_with_budget(port: int, bind_host: str, attempts: int, interval_sec: float) -> bool:
    """Shared TCP-probe implementation. residual[3] fix: bind_host is no
    longer hardcoded to 127.0.0.1 — important if a caller ever opts into
    a non-loopback bind."""
    for _ in range(attempts):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.250)
        try:
            sock.connect((bind_host, port))
            return True
        except OSError:
            pass
        finally:
            sock.close()
        time.sleep(interval_sec)
    return False


# ----------------------------------------------------------------------
# Background threads — sweep + idle shutdown
# ----------------------------------------------------------------------


def _start_background_threads(entry: _SpawnedEntry, cfg: LifecycleConfig) -> None:
    """Start the sweep + idle-shutdown daemon threads."""
    if cfg.sweep_interval_sec > 0:
        sweep_thread = threading.Thread(
            target=_sweep_loop,
            args=(entry, cfg),
            name="coord-sweep",
            daemon=True,
        )
        sweep_thread.start()
    if cfg.idle_shutdown_sec > 0:
        idle_thread = threading.Thread(
            target=_idle_shutdown_loop,
            args=(entry, cfg),
            name="coord-idle",
            daemon=True,
        )
        idle_thread.start()


def _report_sweep_reclaims(
    coordinator: CoordinatorHTTPServer,
    reclaims: list[tuple[UUID, UUID, str]],
    now_tick: int,
) -> None:
    """Count and log each reclaim the sweep landed (#195). Best-effort per
    entry: a failure for one never skips the rest, and never raises."""
    for artifact_id, agent_id, trigger in reclaims:
        try:
            coordinator.record_sweep_reclaim(trigger)
        except Exception:  # noqa: BLE001 — observability, best-effort
            logger.exception("sweep reclaim counter update failed")
        try:
            artifact = coordinator.registry.get_artifact(artifact_id)
            label = artifact.name if artifact is not None else str(artifact_id)
        except Exception:  # noqa: BLE001 — a log label, best-effort
            label = str(artifact_id)
        # The agent id is the non-reversible uuid5 /status already publishes
        # (R6): no session id, no session name, in the log line either.
        logger.warning(
            "sweep reclaimed grant: trigger=%s tick=%d agent_id=%s artifact=%s",
            trigger,
            now_tick,
            agent_id,
            label,
        )


def _sweep_stable_grants(
    coordinator: CoordinatorHTTPServer, cfg: LifecycleConfig, now_tick: int
) -> int:
    """One stable-grant sweep pass, with its reclaims made observable (#195).

    ADV-004: each reclaim records a preemption notice for the victim under
    ``SWEEP_RECLAMATION_PREEMPTER_ID``, so its eventual post-edit gets the
    "reclaimed by coordinator sweep" error instead of a bare CoherenceError.

    #195: each reclaim is also logged and counted
    (``sweep_reclaims_total`` / ``sweep_reclaims_by_trigger``), so a tick that
    pulled a live grant no longer reads as a quiet tick. Both happen AFTER
    ``enforce_stable_grant_timeouts`` returns: its ``on_reclaim`` runs inside
    the per-pair registry hold (KTD7 — no blocking on other threads), so the
    callback only records the notice and appends to a local list; the log
    handler's lock and the counter lock are taken outside the hold. WARNING,
    not INFO: the coordinator configures no logging, and Python's last-resort
    handler emits WARNING and above only, so an INFO line (like the
    reaped/evicted ones beside this) is dropped even by a host that keeps the
    coordinator's stderr; and a reclaim pulls a grant from a possibly live
    holder. (The detached ``agent-coherence-coordinator`` child sends stderr
    to /dev/null, so there the counter and ``/status`` are the surface.)
    The observability step is best-effort: a failure there is logged and never
    costs the tick its remaining passes. It runs in a ``finally``, so reclaims
    that landed before a mid-walk exception are still counted and logged.

    Returns the reclaimed count, as ``enforce_stable_grant_timeouts`` does.
    """
    from ccs.adapters.claude_code.coordinator_server import (
        SWEEP_RECLAMATION_PREEMPTER_ID,
    )

    reclaims: list[tuple[UUID, UUID, str]] = []

    def _on_reclaim(artifact_id: UUID, agent_id: UUID, trigger: str) -> None:
        # Append FIRST: a notice-write failure below is caught and logged by
        # the service, and must not lose the reclaim from the count.
        reclaims.append((artifact_id, agent_id, trigger))
        # Float wall-clock time (the notice timestamp is operator-visible via
        # the F4 prose) rather than the integer sweep tick.
        coordinator.registry.record_preemption_notice(
            victim_agent_id=agent_id,
            artifact_id=artifact_id,
            preempter_agent_id=SWEEP_RECLAMATION_PREEMPTER_ID,
            preempted_at_unix_ts=time.time(),
        )

    # try/finally: the walk commits each pair on its own, so a later pair
    # raising (a store error mid-walk) leaves the earlier reclaims durable.
    # They are counted and logged on the way out all the same; otherwise the
    # tick reads as quiet in exactly the case an operator most needs it not
    # to. The exception still propagates to the caller unchanged.
    try:
        reclaimed = coordinator.service.enforce_stable_grant_timeouts(
            current_tick=now_tick,
            heartbeat_timeout_ticks=cfg.grant_heartbeat_timeout_sec,
            max_hold_ticks=cfg.grant_max_hold_sec,
            on_reclaim=_on_reclaim,
        )
    finally:
        _report_sweep_reclaims(coordinator, reclaims, now_tick)
    return reclaimed


def _sweep_loop(entry: _SpawnedEntry, cfg: LifecycleConfig) -> None:
    """Periodic sweep: transient → stable grant → notice eviction.

    Order matters per R4: transient sweep first so the stable sweep does
    not race entries that are mid-protocol. F2 notice eviction last —
    it's a pure storage reclaim and doesn't interact with grants.

    ADV-004: stable-grant reclamation records a preemption notice for
    the reclaimed victim so the victim's eventual post-edit gets an
    enriched "reclaimed by coordinator sweep" error instead of a
    generic CoherenceError with no context. The sentinel preempter
    UUID is ``SWEEP_RECLAMATION_PREEMPTER_ID`` (imported lazily to
    avoid an import cycle at module load). See :func:`_sweep_stable_grants`.
    """
    # Lazy import — coordinator_server imports lifecycle's
    # CoordinatorHTTPServer; importing back at module load would cycle.
    from ccs.adapters.claude_code.coordinator_server import (
        _SHARED_FOREIGN_DENY_LAG_WINDOW_SEC,
        monotonic_seconds,
    )
    from ccs.adapters.claude_code.foreign_write_detector import run_detection_pass

    coordinator = entry.coordinator
    # Held across ticks for this coordinator: which files the detector has
    # already read, by size and modification time. It carries no coordination
    # state and no safety comparand — only a hint about what is worth
    # re-reading, so a stale entry costs one extra read and never a wrong count.
    detection_stat_cache: dict[str, tuple[tuple[int, int], str]] = {}
    # Also held across ticks for this coordinator: which permanent poll faults
    # have already been reported. A workspace that is not a git repository can
    # never be polled, and logging that once per sweep interval for the life of
    # the coordinator buries the failures an operator can act on.
    detection_reported_faults: set[str] = set()
    # And when detection last recorded a tick, so a sweep that stalls — a
    # suspended host, or the safety passes above stuck on the store — ends the
    # observed interval instead of being interpolated across. The pass cannot
    # see the interval in which it did not run; the distance back to the last
    # tick is the only evidence it left.
    detection_tick_clock: dict[str, float] = {}

    while not coordinator.shutting_down:
        time.sleep(cfg.sweep_interval_sec)
        if coordinator.shutting_down:
            break
        # The sweep tick MUST share the basis the heartbeat handlers seed from
        # (``monotonic_seconds`` == ``int(time.time())``, wall-clock seconds): the
        # grant heartbeat handlers, the session heartbeat, and ``begin_session``'s
        # ``created_at_tick`` are ALL seeded from ``monotonic_seconds()``. Using
        # ``int(time.monotonic())`` here (a boot-relative clock ~1.7e9 below
        # wall-clock) made ``current_tick - last_hb`` permanently negative, so the
        # transient, grant, AND session sweeps never fired over the HTTP transport
        # (heartbeat timeouts + the absolute-age ceiling silently inoperative).
        # Wall-clock also survives a restart — the basis ``deadline_tick`` and the
        # session ceiling need; monotonic resets on reboot and would break those.
        now_tick = monotonic_seconds()
        try:
            coordinator.service.enforce_transient_timeouts(
                current_tick=now_tick,
                timeout_ticks=cfg.transient_timeout_sec,
            )
            _sweep_stable_grants(coordinator, cfg, now_tick)
            # SB-17 / TX-1 Unit 5 / R4: the session-liveness sweep — a SEPARATE
            # axis from the grant sweep above (a snapshot session holds no MESI
            # grant, so the grant sweep can never see it). Reaps a session whose
            # heartbeat lease has gone stale, dropping its pins so its pinned
            # versions become collectible again; afterward a session_read /
            # session_commit on that token fails closed with session_invalidated.
            # Reuses the SAME staleness predicate + the grant heartbeat timeout
            # knob. Best-effort like the rest of the sweep (the outer try guards
            # it). A slow-but-live session that keeps heartbeating is NOT reaped.
            # A REMOTE client refreshes its lease via the HTTP
            # POST /session/heartbeat endpoint; an abandoned session is reaped
            # fail-closed (session_invalidated, never wrong bytes) once the lease
            # goes stale or the absolute-age ceiling fires, with durable-only
            # post-restart sessions bounded by that ceiling.
            reaped_sessions = coordinator.service.enforce_session_liveness(
                current_tick=now_tick,
                heartbeat_timeout_ticks=cfg.grant_heartbeat_timeout_sec,
            )
            if reaped_sessions:
                logger.info(
                    "sweep reaped %d stale snapshot session(s)", reaped_sessions
                )
            evicted = coordinator.registry.evict_stale_notices(
                max_age_sec=cfg.notice_evict_max_age_sec,
            )
            if evicted:
                logger.info("sweep evicted %d stale preemption notice(s)", evicted)
        except Exception as exc:
            # Sweep is best-effort — never crash the coordinator.
            logger.exception("sweep tick failed: %s", exc)
        # Foreign-write detection: a fifth pass, deliberately OUTSIDE the try
        # above rather than appended inside it. The four passes share one
        # best-effort guard, so a detection failure inside it would cost the
        # tick's reclamation work; and running detection only when all four
        # succeeded would make the instrument's own liveness depend on theirs.
        # Two separate guards keep the failure domains apart in both
        # directions. ``run_detection_pass`` raises nothing by contract.
        #
        # The poll gets one sweep interval as its whole budget, not per batch:
        # detection runs in this loop, so an instrument that overruns delays the
        # next tick's grant reclamation. Exhausting it fails the poll honestly,
        # which leaves the tick unrecorded and the gap visible.
        #
        # The window is the shipped benign commit-to-disk lag plus one tick.
        # They are otherwise both 5.0s and the comparison is inclusive, so a
        # mediated commit first observed on the next tick would sit exactly on
        # the boundary and jitter would decide whether it read as foreign.
        run_detection_pass(
            coordinator,
            now_unix=now_tick,
            window_sec=_SHARED_FOREIGN_DENY_LAG_WINDOW_SEC + cfg.sweep_interval_sec,
            poll_budget_sec=cfg.sweep_interval_sec,
            stat_cache=detection_stat_cache,
            reported_faults=detection_reported_faults,
            tick_clock=detection_tick_clock,
            # Three sweep intervals of slack: one tick that arrives late
            # because the four passes ahead of it ran long must not fragment a
            # healthy run, but a stall long enough to hide a write must not be
            # read as continuously observed either.
            max_gap_sec=cfg.sweep_interval_sec * 3,
        )


def _idle_shutdown_loop(entry: _SpawnedEntry, cfg: LifecycleConfig) -> None:
    """Monotonic idle watcher. When ``idle_seconds >= idle_shutdown_sec``,
    runs the race-safe shutdown sequence. Finding L5: ``idle_seconds`` is now a
    monotonic delta, so a wall-clock (NTP / suspend-resume) step no longer
    misfires or defers idle shutdown.

    P2 ce-review fix #7 (reliability): retry shutdown on the next tick if
    G4 abort path fired (coordinator.shutdown raised) — previously the
    loop exited permanently on the first attempt, leaving flock held
    forever and the idle thread dead.
    """
    coordinator = entry.coordinator
    while not coordinator.shutting_down:
        time.sleep(cfg.sweep_interval_sec)
        if coordinator.shutting_down:
            break
        # P3 ce-review fix #39: use the public idle_seconds property (was
        # private _last_request_at + type: ignore).
        idle_for = coordinator.idle_seconds
        if idle_for >= cfg.idle_shutdown_sec:
            logger.info(
                "coordinator idle for %.0fs (>= %ss threshold) — shutting down",
                idle_for, cfg.idle_shutdown_sec,
            )
            _shutdown_sequence(entry)
            if entry.shutdown_done.is_set():
                _SPAWNED_REGISTRY.pop(str(coordinator.coordinator_root), None)
                return
            # G4 abort: shutdown_done NOT set — loop continues so the next
            # sweep tick will retry. Stable-grant reclamation
            # (max_hold_ticks) remains the long-term safety net but we
            # should still try again ourselves rather than wedging the
            # idle thread.
            logger.warning(
                "shutdown aborted (G4); will retry on next sweep tick"
            )


def _shutdown_sequence(entry: _SpawnedEntry) -> bool:
    """Race-safe, mutexed shutdown.

    Concurrent triggers (stop_coordinator + idle-shutdown thread) are
    serialized by ``entry.shutdown_lock``. The first caller runs the
    sequence; subsequent callers see ``shutdown_done`` set and return
    immediately without touching pid file or fd.

    Ordering (revised per subagent findings G4 + G5):
      1. Drop the port from the pid file FIRST. This closes the cascade
         window in G5: loser readers immediately see "no port" instead
         of a port pointing at a coordinator that's about to die.
      2. Set the coordinator's shutting_down flag (handlers 503).
      3. Run coordinator.shutdown() — blocks on serve_forever exit and
         drains in-flight handlers. If THIS raises (G4), we ABORT: the
         lock stays held, the pid file stays port-empty, and the next
         spawn-side caller will retry. Stable-grant reclamation
         (max_hold_ticks) provides the long-term safety net.
      4. Release the flock.
      5. Close the fd.

    Returns True if this caller's invocation actually ran the sequence
    (whether it completed cleanly OR aborted on G4). Returns False only
    when shutdown_done was already set on entry (no-op).

    To check whether shutdown ACTUALLY COMPLETED (vs aborted via G4),
    check ``entry.shutdown_done.is_set()`` after the call — that's the
    truth source. The return value is "did THIS caller execute the
    sequence body" so concurrent callers can distinguish "I did the
    work" from "someone else already did it".
    """
    with entry.shutdown_lock:
        if entry.shutdown_done.is_set():
            return False

        coordinator = entry.coordinator
        lock_fd = entry.lock_fd

        # Step 1 (G5): drop port FIRST so loser readers don't get a
        # stale-but-live port during the shutdown drain window.
        try:
            _rewrite_pidfile_drop_port(lock_fd, os.getpid())
        except OSError as exc:
            logger.warning(
                "could not drop port from pid file during shutdown: %s — "
                "loser readers may still see stale port",
                exc,
            )

        # Step 2 + 3: shut down the HTTP server. If this raises, abort the
        # sequence — leave the lock held so no new coordinator can spawn
        # while in-flight handlers are still running against a partially
        # torn-down state.
        try:
            coordinator.shutdown()
        except Exception as exc:
            entry.shutdown_abort_count += 1
            # REL-06: after N consecutive G4 aborts the operator's only
            # recovery path was "kill the process and let stable-grant
            # reclamation eventually clear the M/E slots, then ensure
            # a fresh spawn". That's a load-bearing safety net but it's
            # ~1800s in the worst case (max_hold_ticks default). Once
            # we've exhausted our patience for the coordinator to drain
            # cleanly, ESCALATE: release the flock + mark shutdown_done
            # so a fresh ensure_coordinator can re-spawn immediately.
            # Logged at CRITICAL with the escalation reason — operator
            # can correlate with any in-flight handler state in their
            # diagnostic surface.
            if entry.shutdown_abort_count >= _MAX_SHUTDOWN_RETRIES_BEFORE_ESCALATION:
                logger.critical(
                    "coordinator.shutdown failed %d consecutive times "
                    "(>= escalation threshold %d). ESCALATING: releasing "
                    "flock + marking shutdown_done so a fresh spawn can "
                    "proceed. In-flight handler state may be inconsistent. "
                    "Underlying error: %s",
                    entry.shutdown_abort_count,
                    _MAX_SHUTDOWN_RETRIES_BEFORE_ESCALATION,
                    exc,
                )
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
                _close_quiet(lock_fd)
                entry.shutdown_done.set()
                return True
            logger.critical(
                "coordinator.shutdown failed (attempt %d/%d); aborting "
                "shutdown sequence with lock still held. Stable-grant "
                "reclamation is the recovery path. Underlying error: %s",
                entry.shutdown_abort_count,
                _MAX_SHUTDOWN_RETRIES_BEFORE_ESCALATION,
                exc,
            )
            # shutdown_done remains UNSET so a retry is possible. Return
            # True because this invocation DID run the sequence body
            # (whose outcome was abort). Caller can inspect
            # entry.shutdown_done to distinguish complete vs aborted.
            return True

        # Step 4 + 5: release the flock + close fd. Wrap each in try/except
        # so a failure at this stage (rare — lock already validly held)
        # doesn't crash but still marks the sequence done.
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        except OSError as exc:
            logger.warning("could not release flock during shutdown: %s", exc)
        try:
            os.close(lock_fd)
        except OSError:
            pass

        entry.shutdown_done.set()
        return True


# ----------------------------------------------------------------------
# Backward-compat aliases removed (finding #46: all internal callers now
# import the public names directly via `read_port_from_file as read_port_from_file`
# or `tcp_probe as tcp_probe`). No alias needed.
