# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Unit 1 tests for CoherentVolume: spawn-with-strict, identity, fail-closed.

These exercise the façade scaffolding only (construction, strict-mode
enablement, per-instance + fork-safe identity, and the on_error contract).
The read/write contract (Unit 2) and the install() shim (Unit 3) are tested
separately.
"""

from __future__ import annotations

import builtins
import contextlib
import gc
import http.server
import io
import logging
import os
import select
import signal
import subprocess
import threading
import time
import warnings
import weakref
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

import ccs.adapters.claude_code.coordinator_server as coordinator_server_module
import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.coordinator_server import (
    read_subagent_id,
    session_to_agent_id,
)
from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.adapters.coherent_volume import (
    DENIED_READ_BACKOFF_BASE_SEC,
    DENIED_READ_BACKOFF_CAP_SEC,
    MAX_CAS_REACQUIRES,
    CoherentVolume,
    ManagedGlobEnforcement,
    coherent_workspace,
    denied_read_backoff_sec,
    install,
    uninstall,
)
from ccs.cli._coherence_client import CoordinatorEndpoint, CoordinatorUnavailable
from ccs.core.exceptions import (
    CasRetriesExhausted,
    CasVersionConflict,
    CoherenceDegradedWarning,
    CoherenceError,
    StaleView,
    ViewWedged,
)


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def test_construct_spawns_with_strict_enabled(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """Constructing with managed globs spawns a coordinator that actually
    reports strict mode (verified on the coordinator via /status, not just
    the façade's intent)."""
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol.is_attached
        assert vol.strict_mode_active() is True
        assert not vol.is_degraded
    finally:
        stop_coordinator(tmp_path)


def test_unmanaged_paths_get_no_strict(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """With no managed globs there is no strict-mode opt-in — documents why
    the managed set is what gives invalidation teeth."""
    vol = CoherentVolume(tmp_path, managed=(), config=fast_cfg)
    try:
        assert vol.is_attached
        assert vol.strict_mode_active() is False
    finally:
        stop_coordinator(tmp_path)


def test_fresh_workspace_coherence_dir_is_0700_without_tighten_warning(
    tmp_path: Path, fast_cfg: LifecycleConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """The pre-spawn policy write creates ``.coherence/`` itself, ahead of the
    lifecycle. It must create it at the 0700 the lifecycle requires; otherwise
    every brand-new workspace spawns with a "tightened existing .coherence
    directory" warning that blames the operator for a directory the volume
    created a moment earlier. umask is pinned to 022 so a permissive default
    ``mkdir`` is observable regardless of the host's setting."""
    prior_umask = os.umask(0o022)
    try:
        caplog.set_level(logging.WARNING, logger="ccs.adapters.claude_code.lifecycle")
        vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
        assert vol.is_attached
        assert ((tmp_path / ".coherence").stat().st_mode & 0o777) == 0o700
        tightened = [
            r.getMessage()
            for r in caplog.records
            if "tightened existing .coherence directory" in r.getMessage()
        ]
        assert tightened == []
    finally:
        os.umask(prior_umask)
        stop_coordinator(tmp_path)


def test_per_instance_identity_is_distinct(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """Two CoherentVolume instances are distinct writers (distinct session ids),
    even in the same process."""
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()
    vol_a = CoherentVolume(ws_a, managed=("data/**",), config=fast_cfg)
    vol_b = CoherentVolume(ws_b, managed=("data/**",), config=fast_cfg)
    try:
        assert vol_a.session_id != vol_b.session_id
        # Each session id is a v4-shaped UUID string.
        assert len(vol_a.session_id) == 36 and vol_a.session_id.count("-") == 4
    finally:
        stop_coordinator(ws_a)
        stop_coordinator(ws_b)


def test_after_fork_remints_identity_and_drops_endpoint(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The fork child-handler re-mints identity and drops the cached endpoint
    (so a forked worker is not conflated with its parent as one writer)."""
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        parent_id = vol.session_id
        assert vol.is_attached
        vol._after_fork()  # simulate the child-side handler directly
        assert vol.session_id != parent_id
        assert vol._endpoint is None
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_real_fork_child_has_distinct_identity(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """An actual os.fork() child re-mints identity via os.register_at_fork."""
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    parent_id = vol.session_id
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        try:
            os.write(write_fd, vol.session_id.encode("utf-8"))
        finally:
            os.close(write_fd)
            os._exit(0)
    # parent
    os.close(write_fd)
    try:
        child_id = os.read(read_fd, 64).decode("utf-8")
        os.close(read_fd)
        os.waitpid(pid, 0)
        assert child_id != parent_id
        assert len(child_id) == 36
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_real_fork_resets_every_live_volume(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """One fork handler walks every live volume: with two alive at fork time,
    the child re-mints both, not only the first one the weak set yields."""
    ws_a = tmp_path / "a"
    ws_b = tmp_path / "b"
    ws_a.mkdir()
    ws_b.mkdir()
    vol_a = CoherentVolume(ws_a, managed=("data/**",), config=fast_cfg)
    vol_b = CoherentVolume(ws_b, managed=("data/**",), config=fast_cfg)
    parent_ids = (vol_a.session_id, vol_b.session_id)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        try:
            os.write(write_fd, f"{vol_a.session_id} {vol_b.session_id}".encode("utf-8"))
        finally:
            os.close(write_fd)
            os._exit(0)
    # parent
    os.close(write_fd)
    try:
        child_ids = tuple(os.read(read_fd, 128).decode("utf-8").split())
        os.close(read_fd)
        os.waitpid(pid, 0)
        assert len(child_ids) == 2
        assert child_ids[0] != parent_ids[0]
        assert child_ids[1] != parent_ids[1]
        assert child_ids[0] != child_ids[1]
    finally:
        stop_coordinator(ws_a)
        stop_coordinator(ws_b)


def test_fork_reset_continues_past_a_failing_volume(monkeypatch: pytest.MonkeyPatch) -> None:
    """A volume whose reset raises must not leave the rest unreset: each volume
    once had its own fork handler, and CPython runs every handler even after one
    raises. The shared handler resets them all, then raises what failed so the
    child still reports it. CPython's default unraisable hook prints only a
    group's own message, so that message must name each failure. Every stub
    raises, so a loop that stops at the first failure resets exactly one,
    whichever the weak set yields first."""
    reset_calls: list[str] = []

    class _FailingReset:
        def __init__(self, name: str) -> None:
            self.name = name

        def _after_fork(self) -> None:
            reset_calls.append(self.name)
            raise RuntimeError(f"reset {self.name} failed")

    stubs = [_FailingReset("a"), _FailingReset("b")]
    monkeypatch.setattr(coherent_volume_module, "_FORK_RESET_VOLUMES", weakref.WeakSet(stubs))

    with pytest.raises(BaseExceptionGroup) as excinfo:
        coherent_volume_module._reset_volumes_after_fork()

    assert sorted(reset_calls) == ["a", "b"]
    assert sorted(str(exc) for exc in excinfo.value.exceptions) == ["reset a failed", "reset b failed"]
    printed = str(excinfo.value)
    assert "RuntimeError: reset a failed" in printed
    assert "RuntimeError: reset b failed" in printed


def test_fork_reset_raises_nothing_when_every_volume_resets(monkeypatch: pytest.MonkeyPatch) -> None:
    """The handler runs in every forked child of a process that imported the
    adapter, so a clean pass must raise nothing: anything it raises is printed as
    an ignored exception in that child."""
    reset_calls: list[str] = []

    class _Reset:
        def __init__(self, name: str) -> None:
            self.name = name

        def _after_fork(self) -> None:
            reset_calls.append(self.name)

    stubs = [_Reset("a"), _Reset("b")]
    monkeypatch.setattr(coherent_volume_module, "_FORK_RESET_VOLUMES", weakref.WeakSet(stubs))

    coherent_volume_module._reset_volumes_after_fork()

    assert sorted(reset_calls) == ["a", "b"]


@contextlib.contextmanager
def _held_by_another_thread(enter: Callable[[], contextlib.AbstractContextManager[object]]) -> Iterator[None]:
    """Keep ``enter()`` entered on a second thread for the duration of the block."""
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with enter():
            held.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert held.wait(5)
    try:
        yield
    finally:
        release.set()
        holder.join(5)


def _report_from_fork_child(child: Callable[[], str], timeout: float = 5.0) -> str | None:
    """Fork, run ``child`` in the child, and return what it reported: ``""`` if it
    raised, ``None`` if the child produced nothing within ``timeout`` (a hang). The
    child is killed and reaped either way, so a hung child cannot stall the suite."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        os.close(read_fd)
        try:
            os.write(write_fd, child().encode("utf-8"))
        finally:
            os.close(write_fd)
            os._exit(0)
    os.close(write_fd)
    try:
        ready, _, _ = select.select([read_fd], [], [], timeout)
        return os.read(read_fd, 256).decode("utf-8") if ready else None
    finally:
        os.close(read_fd)
        os.kill(pid, signal.SIGKILL)  # no-op on a child that already exited
        os.waitpid(pid, 0)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
@pytest.mark.parametrize("held", ["lock", "single_op_guard", "guard_meta_lock"])
def test_fork_while_another_thread_holds_volume_state_leaves_child_usable(
    tmp_path: Path, fast_cfg: LifecycleConfig, held: str
) -> None:
    """Only the forking thread survives a fork, so a lock or guard another parent
    thread held at that moment stays held in the child with nothing left to
    release it. The child must neither hang, in the fork handler or on a later
    ``_lock`` or ``_guard_meta_lock``, nor refuse every operation as concurrent
    use by a thread that no longer exists (the single-op guard)."""
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    enter = {
        "lock": lambda: vol._lock,
        "single_op_guard": vol._single_op_guard,
        "guard_meta_lock": lambda: vol._guard_meta_lock,
    }[held]

    def child() -> str:
        # A real read and write re-attach under the child's identity through the
        # single-op guard; taking _lock covers reacquire and the degradation path.
        if vol.read("data/seed.txt") != b"seed":
            return ""
        vol.write("data/seed.txt", b"child")
        with vol._lock:
            return vol.session_id

    try:
        vol.write("data/seed.txt", b"seed")
        parent_id = vol.session_id
        with _held_by_another_thread(enter):
            reported = _report_from_fork_child(child)
        assert reported is not None, "the forked child hung"
        assert reported not in ("", parent_id)
    finally:
        stop_coordinator(tmp_path)


def _raise_once(real: Callable[..., object], error: BaseException) -> Callable[..., object]:
    """Wrap ``real`` so its next call raises ``error`` and later calls pass through."""
    pending_failures = [error]

    def flaky(*args: object, **kwargs: object) -> object:
        if pending_failures:
            raise pending_failures.pop()
        return real(*args, **kwargs)

    return flaky


def _fail_first_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the next ``resolve_endpoint`` raise once, as when ``server.pid`` or
    ``hook.secret`` is briefly missing during a coordinator restart."""
    unavailable = CoordinatorUnavailable("server.pid missing (coordinator restarting)")
    flaky_resolve = _raise_once(coherent_volume_module.resolve_endpoint, unavailable)
    monkeypatch.setattr(coherent_volume_module, "resolve_endpoint", flaky_resolve)


class _Interrupted(BaseException):
    """Stands in for an interrupt (KeyboardInterrupt, SystemExit) mid-attach."""


def _coordinator_version(vol: CoherentVolume, rel: str) -> int | None:
    """The coordinator's recorded version of ``rel`` (None if untracked)."""
    status = vol.coordinator_status() or {}
    versions = {a.get("path"): a.get("version") for a in status.get("tracked_artifacts", [])}
    return versions.get(rel)


@pytest.mark.parametrize(
    ("failing", "error", "raised"),
    [
        ("resolve_endpoint", CoordinatorUnavailable("server.pid missing"), CoherenceError),
        ("connect_or_spawn", OSError("simulated spawn failure"), OSError),
    ],
    ids=["coordinator-unavailable", "unhandled-error"],
)
def test_failed_reattach_after_fork_is_retried_in_strict_mode(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
    error: BaseException,
    raised: type[BaseException],
) -> None:
    """A forked child whose first re-attach fails must retry it on the next
    read or write, whatever the failure raised — the handled CoherenceError or
    an error from the coordinator spawn the fail-closed path never sees.
    Dropping the pending re-attach when the attempt failed left a strict child
    detached for life: every later op took the unattached branch, so an
    ordinary write landed on disk without an error while the coordinator
    recorded nothing and peers were never invalidated."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        vol.write("data/shared.txt", b"parent")
        version_before = _coordinator_version(vol, "data/shared.txt")
        assert version_before is not None

        vol._after_fork()  # simulate the child-side fork handler
        real = getattr(coherent_volume_module, failing)
        monkeypatch.setattr(coherent_volume_module, failing, _raise_once(real, error))

        with pytest.raises(raised):
            vol.read("data/shared.txt")  # re-attach fails -> strict raises
        assert not vol.is_attached

        assert vol.read("data/shared.txt") == b"parent"  # retries the re-attach
        assert vol.is_attached

        vol.write("data/shared.txt", b"child")
        assert _coordinator_version(vol, "data/shared.txt") > version_before
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize(
    "operation",
    [
        lambda vol: vol.read("data/shared.txt"),
        lambda vol: vol.read_with_version("data/shared.txt"),
        lambda vol: vol.read_with_version_generation("data/shared.txt"),
        lambda vol: vol.write("data/shared.txt", b"child"),
        lambda vol: vol.write_cas("data/shared.txt", lambda current: current + b"+child"),
    ],
    ids=["read", "read_with_version", "read_with_version_generation", "write", "write_cas"],
)
def test_strict_child_fails_closed_on_every_op_while_reattach_fails(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: Callable[[CoherentVolume], object],
) -> None:
    """While the re-attach keeps failing, each read (plain or versioned), write
    and write_cas of a strict forked child raises and nothing lands on disk —
    none of them may fall through to the unattached branch, first call or
    retry."""
    target = _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        vol._after_fork()

        def unavailable(*_args: object) -> None:
            raise CoordinatorUnavailable("server.pid missing (coordinator restarting)")

        monkeypatch.setattr(coherent_volume_module, "resolve_endpoint", unavailable)

        for _ in range(2):
            with pytest.raises(CoherenceError):
                operation(vol)
        assert target.read_bytes() == b"v1"
        assert not vol.is_attached
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("method", ["read_with_version", "read_with_version_generation"])
def test_versioned_read_reattaches_after_fork(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """The versioned reads are reads too. A forked child's first one must
    re-attach and register the view — failing closed while it cannot — not
    return version 0 with the coordinator never asked."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        vol.write("data/shared.txt", b"parent")
        coordinator_version = _coordinator_version(vol, "data/shared.txt")
        assert coordinator_version  # non-zero, so the version-0 fallback cannot pass

        vol._after_fork()
        _fail_first_resolve(monkeypatch)
        versioned_read = getattr(vol, method)

        with pytest.raises(CoherenceError):
            versioned_read("data/shared.txt")
        data, version, *_ = versioned_read("data/shared.txt")
        assert vol.is_attached
        assert (data, version) == (b"parent", coordinator_version)
    finally:
        stop_coordinator(tmp_path)


def test_reattach_to_non_strict_coordinator_after_fork_keeps_failing_closed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forked child that re-attaches to a coordinator not enforcing its managed
    paths must fail closed on every op, not just the first. The strict check
    raised with the endpoint to that coordinator still set, so the next op saw
    an endpoint, skipped the re-attach, and ran through it unenforced."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        vol._after_fork()
        monkeypatch.setattr(
            vol, "managed_glob_enforcement",
            lambda: ManagedGlobEnforcement((), ("data/**",), None),
        )

        for _ in range(2):
            with pytest.raises(CoherenceError):
                vol.read("data/shared.txt")
            assert not vol.is_attached

        monkeypatch.undo()  # the coordinator enforces the managed paths again
        assert vol.read("data/shared.txt") == b"v1"
        assert vol.is_attached
    finally:
        stop_coordinator(tmp_path)


def test_interrupted_reattach_after_fork_leaves_the_child_detached(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupt that escapes the re-attach after the endpoint was resolved —
    here during the strict-enforcement check — must not leave that endpoint set.
    The retry runs only while the endpoint is None, so a kept endpoint would
    carry every later op through a coordinator whose enforcement was never
    checked."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        vol._after_fork()
        flaky_check = _raise_once(vol.managed_glob_enforcement, _Interrupted())
        monkeypatch.setattr(vol, "managed_glob_enforcement", flaky_check)

        with pytest.raises(_Interrupted):
            vol.read("data/shared.txt")
        assert not vol.is_attached

        assert vol.read("data/shared.txt") == b"v1"  # retries, checks, attaches
        assert vol.is_attached
    finally:
        stop_coordinator(tmp_path)


def test_failed_reattach_after_fork_stays_best_effort_in_degrade_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under on_error='degrade' the same failed re-attach warns and the op runs
    best-effort, as a failed attach at construction does — it does not raise."""
    target = _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        vol._after_fork()
        _fail_first_resolve(monkeypatch)

        with pytest.warns(CoherenceDegradedWarning):
            assert vol.read("data/shared.txt") == b"v1"
        vol.write("data/shared.txt", b"child")
        assert target.read_bytes() == b"child"
        assert vol.is_degraded
        assert not vol.is_attached  # one attempt: the write did not re-attach
    finally:
        stop_coordinator(tmp_path)


def test_unexpected_reattach_error_after_fork_is_not_retried_in_degrade_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degrade keeps its one attempt even when the re-attach fails with an error
    the degrade path does not handle: that op raises it, and later ops run
    best-effort instead of retrying (and re-raising) on every call. Running
    best-effort for life, the child must warn and count like the handled
    failures — re-raising alone left is_degraded False while enforcement was off."""
    target = _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        vol._after_fork()
        attempts: list[object] = []

        def failing_connect_or_spawn(*args: object, **kwargs: object) -> None:
            attempts.append(args)
            _raise_oserror()

        monkeypatch.setattr(coherent_volume_module, "connect_or_spawn", failing_connect_or_spawn)

        with pytest.warns(CoherenceDegradedWarning, match="OSError"):
            with pytest.raises(OSError):
                vol.read("data/shared.txt")
        assert vol.is_degraded

        vol.write("data/shared.txt", b"child")  # best-effort, no second attempt
        assert target.read_bytes() == b"child"
        assert len(attempts) == 1
        assert not vol.is_attached
    finally:
        stop_coordinator(tmp_path)


def _fail_spawn(vol: CoherentVolume, monkeypatch: pytest.MonkeyPatch) -> None:
    """An error the degrade path does not handle, before the child connects."""
    monkeypatch.setattr(coherent_volume_module, "connect_or_spawn", _raise_oserror)


def _interrupt_strict_check(vol: CoherentVolume, monkeypatch: pytest.MonkeyPatch) -> None:
    """An interrupt after the child connected, during the strict-mode check."""
    flaky_check = _raise_once(vol.managed_glob_enforcement, _Interrupted())
    monkeypatch.setattr(vol, "managed_glob_enforcement", flaky_check)


def _fail_resolve(vol: CoherentVolume, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failure the degrade path handles: the coordinator briefly unreachable."""
    _fail_first_resolve(monkeypatch)


@pytest.mark.parametrize(
    ("inject_failure", "raised"),
    [(_fail_spawn, OSError), (_interrupt_strict_check, _Interrupted)],
    ids=["unhandled-error", "interrupt"],
)
def test_failed_reattach_after_fork_never_reports_degraded_in_strict_mode(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    inject_failure: Callable[[CoherentVolume, pytest.MonkeyPatch], None],
    raised: type[BaseException],
) -> None:
    """Strict fails closed on these failures and retries on the next op, so it is
    never running best-effort and must not report itself degraded."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        vol._after_fork()
        inject_failure(vol, monkeypatch)

        with pytest.raises(raised):
            vol.read("data/shared.txt")
        assert not vol.is_degraded

        monkeypatch.undo()
        assert vol.read("data/shared.txt") == b"v1"  # retries the re-attach
        assert vol.is_attached
        assert not vol.is_degraded
    finally:
        stop_coordinator(tmp_path)


def test_interrupted_reattach_after_fork_is_detached_and_degraded_in_degrade_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degrade reports the escaped interrupt as degraded, so it must also drop the
    endpoint resolved before it. Keeping it would report degraded while later
    ops ran through a coordinator whose enforcement was never checked."""
    target = _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        vol._after_fork()
        flaky_check = _raise_once(vol.managed_glob_enforcement, _Interrupted())
        monkeypatch.setattr(vol, "managed_glob_enforcement", flaky_check)

        with pytest.warns(CoherenceDegradedWarning):
            with pytest.raises(_Interrupted):
                vol.read("data/shared.txt")
        assert vol.is_degraded
        assert not vol.is_attached

        vol.write("data/shared.txt", b"child")  # best-effort, no second attempt
        assert target.read_bytes() == b"child"
        assert not vol.is_attached
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize(
    ("inject_failure", "raised"),
    [
        (_fail_spawn, OSError),
        (_interrupt_strict_check, _Interrupted),
        (_fail_resolve, CoherenceDegradedWarning),
    ],
    ids=["unhandled-error", "interrupt", "coordinator-unavailable"],
)
def test_failed_degrade_reattach_counts_once_and_keeps_its_error_when_warnings_are_errors(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    inject_failure: Callable[[CoherentVolume, pytest.MonkeyPatch], None],
    raised: type[BaseException],
) -> None:
    """With CoherenceDegradedWarning escalated to an error, a failed re-attach
    still counts once and raises what it raised before. An unhandled error or
    interrupt propagates as itself, not as the warning (which ``except
    Exception`` would catch in place of an interrupt). A handled failure already
    raised the escalated warning from inside the attempt and must not be
    counted a second time on the way out."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        vol._after_fork()
        inject_failure(vol, monkeypatch)

        with warnings.catch_warnings():
            warnings.simplefilter("error", CoherenceDegradedWarning)
            with pytest.raises(raised):
                vol.read("data/shared.txt")
        assert vol.degradation_count == 1
        assert not vol.is_attached
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize(
    ("inject_failure", "raised"),
    [
        (_fail_spawn, OSError),
        (_interrupt_strict_check, _Interrupted),
        (_fail_resolve, CoherenceDegradedWarning),
    ],
    ids=["unhandled-error", "interrupt", "coordinator-unavailable"],
)
def test_failed_degrade_reattach_still_logs_when_warnings_are_errors(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    inject_failure: Callable[[CoherentVolume, pytest.MonkeyPatch], None],
    raised: type[BaseException],
) -> None:
    """The degradation log line must not depend on the warning being shown. An
    escalated warning raises out of ``warnings.warn``; for an unhandled error or
    interrupt it is then suppressed so the original propagates, and for a handled
    failure it propagates as the error itself. Either way the degradation is
    counted, so the log is the only record left for an operator who filters
    warnings into errors."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        vol._after_fork()
        inject_failure(vol, monkeypatch)
        caplog.set_level(logging.WARNING, logger="ccs.adapters.coherent_volume")

        with warnings.catch_warnings():
            warnings.simplefilter("error", CoherenceDegradedWarning)
            with pytest.raises(raised):
                vol.read("data/shared.txt")
        degraded_logs = [
            r
            for r in caplog.records
            if r.name == "ccs.adapters.coherent_volume"
            and r.levelno == logging.WARNING
            and "CoherentVolume degraded" in r.getMessage()
        ]
        assert len(degraded_logs) == 1
    finally:
        stop_coordinator(tmp_path)


def test_foreign_coordinator_strict_raises(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A coordinator already running (not spawned by the appliance) cannot have
    strict mode enabled on it (load-once policy); strict mode fails closed."""
    port = ensure_coordinator(tmp_path, config=fast_cfg)
    assert port > 0
    try:
        with pytest.raises(CoherenceError):
            CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    finally:
        stop_coordinator(tmp_path)


def test_foreign_coordinator_degrade_warns(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """Under on_error='degrade' the same foreign-coordinator condition warns
    once and operates best-effort rather than raising."""
    port = ensure_coordinator(tmp_path, config=fast_cfg)
    assert port > 0
    try:
        with pytest.warns(CoherenceDegradedWarning):
            vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
        assert vol.is_degraded
    finally:
        stop_coordinator(tmp_path)


def _fail_strict_construction(root: Path, cfg: LifecycleConfig) -> list[CoherentVolume]:
    """Construct a strict volume over a coordinator it did not spawn, so ``_attach``
    raises, and return the half-built instance captured just before it did."""
    half_built: list[CoherentVolume] = []

    class _CapturingVolume(CoherentVolume):
        def _attach(self) -> None:
            half_built.append(self)
            super()._attach()

    with pytest.raises(CoherenceError):
        _CapturingVolume(root, managed=("data/**",), on_error="strict", config=cfg)
    return half_built


def test_volume_is_collected_once_unreferenced(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """The fork handler must not own the volume: os.register_at_fork cannot be
    undone, so registering a bound method there kept every volume alive (and
    reset in every forked child) for the life of the process."""
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol_ref = weakref.ref(vol)
        del vol
        gc.collect()
        assert vol_ref() is None
    finally:
        stop_coordinator(tmp_path)


def test_failed_construction_is_collected(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A constructor that raised hands the caller nothing, so nothing may keep
    the half-built instance alive either."""
    ensure_coordinator(tmp_path, config=fast_cfg)
    try:
        vol_ref = weakref.ref(_fail_strict_construction(tmp_path, fast_cfg).pop())
        gc.collect()
        assert vol_ref() is None
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_failed_construction_is_not_reset_in_fork_child(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A construction that raised leaves nothing registered for fork: even while
    something still references the half-built instance, a forked child does not
    re-mint its identity. Being collectable alone would not show this — a weak
    registration made before ``_attach`` and never withdrawn still resets it."""
    ensure_coordinator(tmp_path, config=fast_cfg)
    try:
        (half_built,) = _fail_strict_construction(tmp_path, fast_cfg)
        parent_id = half_built.session_id
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            os.close(read_fd)
            try:
                os.write(write_fd, half_built.session_id.encode("utf-8"))
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        child_id = os.read(read_fd, 64).decode("utf-8")
        os.close(read_fd)
        os.waitpid(pid, 0)
        assert child_id == parent_id
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# Unit 2 — sequential enforce-on-INVALID read/write/reacquire contract.
#
# The teeth: a write from a holder that a peer commit invalidated is DENIED
# (fail-closed). These tests use a FIXED stale buffer — bytes computed from the
# view read BEFORE the peer commit, never re-read — i.e. the OpenViktor cron
# lost-update shape. A refetch-safe "re-read then write" arm would pass even if
# the deny were broken (it would silently re-fetch fresh bytes), so it proves
# nothing; only the fixed-stale-buffer shape actually exercises the deny. See
# docs/solutions/best-practices/
#   coordinator-invalidation-not-mutex-honest-coherence-claims-2026-06-04.md.
# ---------------------------------------------------------------------------


def _seed(tmp_path: Path, rel: str = "data/shared.txt", content: bytes = b"v1") -> Path:
    """Create a tracked file under the workspace; return its absolute path."""
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _pair(tmp_path: Path, cfg: LifecycleConfig) -> tuple[CoherentVolume, CoherentVolume]:
    """Two volumes sharing one workspace + coordinator: A spawns it (writing the
    strict policy), B sibling-attaches to the strict coordinator A spawned."""
    vol_a = CoherentVolume(tmp_path, managed=("data/**",), config=cfg)
    vol_b = CoherentVolume(tmp_path, managed=("data/**",), config=cfg)
    return vol_a, vol_b


def _track_only(tmp_path: Path, glob: str = "data/**") -> None:
    """Mark a glob TRACKED but NOT strict before the coordinator spawns.

    Writes ``.coherence/tracked.yaml`` (so a peer commit invalidates a SHARED
    view) while deliberately leaving ``strict_mode.yaml`` absent (so the re-grant
    is warn-mode — never denied). ``managed=()`` volumes then attach to this
    coordinator without the strict-mode requirement that ``managed`` globs carry.
    Reuses the coordinator's own ``_merge_yaml_list`` writer so the fixture tracks
    the real tracked.yaml format instead of duplicating it.
    """
    coherence_dir = tmp_path / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True)
    CoherentVolume._merge_yaml_list(coherence_dir / "tracked.yaml", (glob,))


def _agent_id(vol: CoherentVolume) -> str:
    """The coordinator's grant-row key for the volume's CURRENT attempt: the
    session id folded with the per-attempt incarnation, derived the way the
    coordinator derives it from the request body."""
    return str(session_to_agent_id(vol.session_id, vol._incarnation))


def _held(vol: CoherentVolume, agent_id: str) -> dict[str, str]:
    """What the COORDINATOR says ``agent_id`` holds, ``{path: state}``, read from
    ``/status`` — which lists every non-INVALID row, so ``{}`` means the row holds
    nothing (released, or never taken)."""
    status = vol.coordinator_status()
    assert status is not None, "the coordinator must be reachable to read its grant rows"
    for session in status["sessions"]:
        if session["agent_id"] == agent_id:
            return dict(session["states"])
    return {}


def _end_turn(vol: CoherentVolume) -> None:
    """Release the volume's grants the way an agent's turn end does: a
    session-stop naming the CURRENT incarnation. A stop without it addresses the
    session's parent row, which holds nothing, and releases nothing. The stop is
    require-class, so it presents the volume's session principal."""
    from ccs.cli._coherence_client import caller_principal_headers
    from ccs.cli._coherence_client import post as _cpost

    _cpost(
        vol._endpoint,
        "/hooks/session-stop",
        {"session_id": vol.session_id, "agent_id": vol._incarnation},
        extra_headers=caller_principal_headers(vol._principal),
    )
    assert _held(vol, _agent_id(vol)) == {}, "the turn-end stop released nothing"


def test_sibling_volume_attaches_to_strict_coordinator(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The fleet case: two volumes on one workspace both attach with strict
    enforced. The second must NOT trip the foreign-coordinator guard — a sibling
    appliance enabled strict, so attaching (rather than failing closed) is
    correct. A truly foreign coordinator without strict still fails closed
    (covered by test_foreign_coordinator_strict_raises)."""
    _seed(tmp_path)
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        assert vol_a.is_attached and vol_b.is_attached
        assert vol_a.strict_mode_active() and vol_b.strict_mode_active()
        assert vol_a.session_id != vol_b.session_id
        assert not vol_a.is_degraded and not vol_b.is_degraded
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_sibling_with_different_managed_globs_is_refused_not_silently_unenforced(
    tmp_path: Path, fast_cfg: LifecycleConfig, on_error: str
) -> None:
    """A volume that attaches to a coordinator a sibling spawned with DIFFERENT
    managed globs is refused: strict raises, degrade warns and runs detached.
    The message names the globs the coordinator does not enforce.

    Prevents the silent lost update in #190: the sibling passed the attach
    check because the coordinator's /status reported only a COUNT of strict
    patterns, and the spawner's one pattern satisfied it. Its own paths were
    untracked, so its reads registered nothing, read_with_version answered 0,
    and a CAS at any expected_version was accepted; nothing raised and
    is_degraded stayed False."""
    for rel, content in (("a/plan.md", b"v0\n"), ("b/notes.md", b"v0\n")):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_bytes(content)
    spawner = CoherentVolume(tmp_path, managed=("a/**",), config=fast_cfg)
    try:
        if on_error == "strict":
            with pytest.raises(CoherenceError) as raised:
                CoherentVolume(tmp_path, managed=("b/**",), on_error="strict", config=fast_cfg)
            message = str(raised.value)
        else:
            with pytest.warns(CoherenceDegradedWarning) as warned:
                sibling = CoherentVolume(
                    tmp_path, managed=("b/**",), on_error="degrade", config=fast_cfg
                )
            assert sibling.is_degraded and not sibling.is_attached, (
                "a degraded sibling kept a live endpoint to a coordinator that does "
                "not enforce its paths")
            message = str(warned[0].message)
        assert "b/**" in message, message
        assert "does not enforce strict mode for" in message, message
        # Control: the spawner's own glob is enforced by the same coordinator.
        assert spawner.is_attached and spawner.strict_mode_active()
        _data, version = spawner.read_with_version("a/plan.md")
        assert version >= 1, "control: the spawner's path is version-tracked"
    finally:
        stop_coordinator(tmp_path)


def test_a_sibling_with_the_same_managed_globs_still_attaches_enforced(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The per-glob check must not refuse the supported fleet: a sibling
    declaring exactly the spawner's globs attaches, and its paths are
    version-tracked by the shared coordinator."""
    _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        assert vol_a.is_attached and vol_b.is_attached
        assert not vol_a.is_degraded and not vol_b.is_degraded
        vol_a.write("data/shared.txt", b"v2")
        _data, version = vol_b.read_with_version("data/shared.txt")
        assert version >= 1, "the sibling's path is version-tracked"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("policy", ["tracked-not-strict", "strict-not-tracked"])
def test_a_managed_glob_the_coordinator_carries_in_only_one_set_is_not_enforced(
    tmp_path: Path, fast_cfg: LifecycleConfig, policy: str
) -> None:
    """Strict mode is an intersection on the coordinator: a path is enforced
    only when it is tracked AND matches a strict pattern. A running
    coordinator that carries this volume's glob in just one of the two sets
    does not enforce it, and the volume is refused by name in strict mode.

    Prevents the per-glob check testing only half of that rule: a glob found
    in the tracked set alone (an operator tracked it but never opted it into
    strict mode), or in the strict set alone (a hand-written strict_mode.yaml
    with no tracked.yaml entry), read as enforced."""
    _seed(tmp_path)
    coherence_dir = tmp_path / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True)
    only = "tracked.yaml" if policy == "tracked-not-strict" else "strict_mode.yaml"
    CoherentVolume._merge_yaml_list(coherence_dir / only, ("data/**",))
    assert ensure_coordinator(tmp_path, config=fast_cfg) > 0  # loads that policy
    try:
        with pytest.raises(CoherenceError) as raised:
            CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert "does not enforce strict mode for managed glob(s) data/**" in str(raised.value)
    finally:
        stop_coordinator(tmp_path)


def test_a_managed_glob_the_coordinator_ignores_is_not_enforced(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The coordinator enforces a path only when it is tracked, strict, and
    matches no ignored pattern, and its hooks answer an ignored path as
    untracked: a read reports version 0 and a CAS commit is accepted at any
    expected version. A volume whose managed glob the coordinator's policy
    ignores is therefore refused at attach, by name, like one it does not
    track.

    Prevents the attach check confirming a glob the coordinator ignores: with
    ``ignored.yaml`` carrying the glob before the spawn, the check answered
    enforced while a stale CAS write landed with nothing raised."""
    _seed(tmp_path)
    coherence_dir = tmp_path / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True)
    CoherentVolume._merge_yaml_list(coherence_dir / "ignored.yaml", ("data/**",))
    try:
        with pytest.raises(CoherenceError) as raised:
            CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        message = str(raised.value)
        assert "does not enforce strict mode for managed glob(s) data/**" in message, message
        assert "ignored" in message, message
    finally:
        stop_coordinator(tmp_path)


# How the operator view fails to answer, and the reason the volume must name.
_UNCONFIRMABLE_VIEWS = {
    "absent": "publishes no glob sets",
    "refused": "the operator view was refused (HTTP 501)",
    "refused-node": "the Node coordinator does not serve the operator view (HTTP 501)",
    "transport": "URLError reading the operator view",
}


@pytest.mark.parametrize("sets", list(_UNCONFIRMABLE_VIEWS))
def test_a_coordinator_that_does_not_publish_its_glob_sets_cannot_confirm_enforcement(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, sets: str
) -> None:
    """When the operator view cannot be read — an older Python coordinator
    whose summary carries only counts, a coordinator that refuses the view,
    the Node coordinator (which answers 501, and is named from the pid file's
    backend line), or a transport failure during that one GET — the volume
    cannot tell whether its globs are enforced. It fails closed, detached, and
    says which: "cannot tell" is never reported as "not enforced", nor as
    enforced.

    Prevents a count, nothing at all, or a raised transport error being read
    as confirmation; the generic exception arm is what keeps a coordinator
    dying mid-attach from escaping the constructor as a raw error."""
    import urllib.error

    _seed(tmp_path, content=b"v1")
    real_get = coherent_volume_module._coordinator_get
    if sets == "refused-node":
        monkeypatch.setattr(
            coherent_volume_module, "coordinator_backend",
            lambda root: coherent_volume_module.NODE_BACKEND,
        )

    def operator_view_unavailable(endpoint, path, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if "detail=full" not in path:
            return real_get(endpoint, path, **kwargs)
        if sets in ("refused", "refused-node"):
            raise urllib.error.HTTPError(path, 501, "Not Implemented", {}, None)  # type: ignore[arg-type]
        if sets == "transport":
            raise urllib.error.URLError("connection refused")
        doc = real_get(endpoint, path, **kwargs)
        summary = dict(doc["policy_summary"])
        for key in ("tracked_patterns", "strict_mode_patterns", "ignored_patterns"):
            summary.pop(key, None)
        return {**doc, "policy_summary": summary}

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", operator_view_unavailable)
    try:
        with pytest.raises(CoherenceError) as raised:
            CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        message = str(raised.value)
        assert "cannot be confirmed" in message, message
        assert _UNCONFIRMABLE_VIEWS[sets] in message, message
        assert "does not enforce strict mode for" not in message, message
        assert "data/**" in message, message

        monkeypatch.setattr(coherent_volume_module, "_coordinator_get", real_get)
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert vol.is_attached, "control: with the sets published, the same volume attaches"
    finally:
        stop_coordinator(tmp_path)


def test_an_unconfirmable_enforcement_answer_runs_detached_in_degrade_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under ``on_error="degrade"`` an operator view that carries no glob sets
    warns once that enforcement cannot be confirmed and leaves the volume
    detached and degraded — the same outcome as an unenforced glob, so
    "cannot tell" never keeps a live endpoint to a coordinator that may not
    enforce the paths.

    Prevents the degrade branch of the cannot-tell answer going unpinned: the
    unenforced branch is pinned in both modes, this one was only in strict."""
    _seed(tmp_path, content=b"v1")
    real_get = coherent_volume_module._coordinator_get

    def counts_only(endpoint, path, **kwargs):  # noqa: ANN001, ANN003, ANN202
        doc = real_get(endpoint, path, **kwargs)
        if "detail=full" not in path:
            return doc
        summary = {k: v for k, v in doc["policy_summary"].items() if not k.endswith("_patterns")}
        return {**doc, "policy_summary": summary}

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", counts_only)
    try:
        with pytest.warns(CoherenceDegradedWarning) as warned:
            vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
        assert vol.is_degraded and not vol.is_attached, (
            "a degraded volume kept a live endpoint to a coordinator whose enforcement "
            "could not be confirmed")
        message = str(warned[0].message)
        assert "cannot be confirmed" in message and "data/**" in message, message
    finally:
        stop_coordinator(tmp_path)


def test_fixed_stale_buffer_write_is_denied(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """THE TEETH. A reads v1, B reads v1, A commits v2 (B -> INVALID), then B
    writes a buffer it computed from v1 WITHOUT re-reading -> the coordinator
    denies the write and write() raises. The stale bytes never land."""
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        assert vol_a.read("data/shared.txt") == b"v1"
        b_view = vol_b.read("data/shared.txt")
        assert b_view == b"v1"
        # B captures a write derived from its v1 view (the lost-update shape).
        b_stale_buffer = b_view + b"\nappended-by-B"

        vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID
        assert target.read_bytes() == b"v2-from-A"

        with pytest.raises(CoherenceError):
            vol_b.write("data/shared.txt", b_stale_buffer)  # DENIED

        # The deny actually protected the file — the stale write did not land.
        assert target.read_bytes() == b"v2-from-A"
    finally:
        stop_coordinator(tmp_path)


def test_strict_deny_is_sticky_bare_read_does_not_recover(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """KTD-T: once INVALID, a bare read() returns fresh bytes but does NOT clear
    INVALID — a subsequent write is still denied, with byte-stable deny text
    across retries (a bare re-read is more robust than an auto-refetch would
    be; recovery requires reacquire())."""
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        vol_a.read("data/shared.txt")
        vol_b.read("data/shared.txt")
        vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID

        # Bare re-read returns the current bytes ...
        assert vol_b.read("data/shared.txt") == b"v2-from-A"
        # ... but does NOT clear INVALID: the write is still denied.
        with pytest.raises(CoherenceError) as first:
            vol_b.write("data/shared.txt", b"v3-attempt-1")
        with pytest.raises(CoherenceError) as second:
            vol_b.write("data/shared.txt", b"v3-attempt-2")
        # Byte-stable deny reason across retries (KTD-P — the model's retry loop
        # relies on this; regenerating it worsens retries).
        assert str(first.value) == str(second.value)
        assert target.read_bytes() == b"v2-from-A"
    finally:
        stop_coordinator(tmp_path)


def test_reacquire_recovers_then_write_succeeds(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """RECOVERY: reacquire() re-mints identity AND does a mandatory fresh read
    (atomically), clearing INVALID. A write from the returned fresh bytes then
    succeeds — no lost update.

    What sheds the sticky INVALID is a FRESH COORDINATOR ROW, and the row is keyed
    on the session id folded with the per-attempt incarnation — so the row key
    must change while the session id stays put. A reacquire that left the key
    unchanged would land the read on the INVALID row and never clear it."""
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        vol_a.read("data/shared.txt")
        vol_b.read("data/shared.txt")
        vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID

        with pytest.raises(CoherenceError):
            vol_b.write("data/shared.txt", b"stale")  # denied

        old_session = vol_b.session_id
        old_row = _agent_id(vol_b)
        fresh = vol_b.reacquire("data/shared.txt")
        assert fresh == b"v2-from-A"  # mandatory fresh read returns current bytes
        assert _agent_id(vol_b) != old_row  # a fresh coordinator row ...
        assert vol_b.session_id == old_session  # ... under the same session id
        assert _held(vol_b, _agent_id(vol_b)) == {"data/shared.txt": "SHARED"}

        # Write rebased on the fresh bytes -> granted.
        vol_b.write("data/shared.txt", fresh + b"\nrebased-by-B")
        assert target.read_bytes() == b"v2-from-A\nrebased-by-B"
    finally:
        stop_coordinator(tmp_path)


def test_first_time_writer_is_not_denied(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Negative control / boundary: strict mode denies an INVALID (preempted)
    writer, NOT a first-time writer. A write to a path this instance never read
    is granted — the strict intent is 'must re-read after preemption', not
    'must read before any write'."""
    _seed(tmp_path)  # ensure data/ exists so the managed glob spawns a coordinator
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write("data/brand-new.txt", b"hello")  # never read -> granted
        assert (tmp_path / "data/brand-new.txt").read_bytes() == b"hello"
    finally:
        stop_coordinator(tmp_path)


def test_reacquire_then_ignoring_fresh_bytes_is_not_caught(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """HONEST BOUNDARY (documented, not a bug): after reacquire() returns fresh
    bytes, a caller that IGNORES them and writes a buffer computed earlier is
    NOT caught — no layer (OCC included) catches 'wrote from a buffer older than
    the read'. v1's honest scope is 'write from the bytes read()/reacquire()
    returned'. This pins the ceiling so a future reader doesn't mistake it for a
    regression."""
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        vol_a.read("data/shared.txt")
        b_v1_view = vol_b.read("data/shared.txt")
        vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID
        vol_b.reacquire("data/shared.txt")  # B current again — but ignores the result
        # B writes a buffer derived from the STALE v1 view -> NOT caught.
        vol_b.write("data/shared.txt", b_v1_view + b"\nignored-reacquire")
        assert target.read_bytes() == b"v1\nignored-reacquire"
    finally:
        stop_coordinator(tmp_path)


def test_read_missing_file_raises_filenotfound(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """read() stats before registering: a missing file raises FileNotFoundError
    and seeds no phantom artifact in the coordinator."""
    _seed(tmp_path)  # ensure data/ exists so a coordinator spawns
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        with pytest.raises(FileNotFoundError):
            vol.read("data/missing.txt")
    finally:
        stop_coordinator(tmp_path)


def test_read_empty_file_then_write(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """An empty file reads as b'' (sha256(b'')), and a subsequent write from the
    same instance is granted (no spurious deny on the empty-hash seed)."""
    _seed(tmp_path, rel="data/empty.txt", content=b"")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol.read("data/empty.txt") == b""
        vol.write("data/empty.txt", b"now-full")
        assert (tmp_path / "data/empty.txt").read_bytes() == b"now-full"
    finally:
        stop_coordinator(tmp_path)


def test_identical_rewrite_skips_filesystem_write(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-op skip: rewriting the exact bytes this instance last committed, while
    holding a fresh grant, skips the os.replace (no filesystem churn) but still
    finalizes the coordinator grant (so the EXCLUSIVE grant is not leaked)."""
    import ccs.adapters.coherent_volume as cv_mod

    _seed(tmp_path, rel="data/x.txt", content=b"orig")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write("data/x.txt", b"committed")  # establishes last_committed_hash
        assert (tmp_path / "data/x.txt").read_bytes() == b"committed"

        calls = {"n": 0}
        real_replace = cv_mod.os.replace

        def counting_replace(src: object, dst: object) -> None:
            calls["n"] += 1
            real_replace(src, dst)

        monkeypatch.setattr(cv_mod.os, "replace", counting_replace)
        vol.write("data/x.txt", b"committed")  # identical bytes -> no-op skip
        assert calls["n"] == 0  # os.replace NOT called the second time
        assert (tmp_path / "data/x.txt").read_bytes() == b"committed"
    finally:
        stop_coordinator(tmp_path)


def test_no_op_skip_gated_on_disk_not_stale_cache(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Regression (pre-existing since CoherentVolume v1): the write no-op-skip
    must check the CURRENT on-disk bytes, not a per-instance cached hash a peer
    commit left stale.

    On a tracked-but-NON-strict glob, pre-edit RE-GRANTS (no strict deny), so the
    skip's own check is the only thing between a stale belief and a silent skip.
    A writes C (cache := H(C)); peer B overwrites with D (A is invalidated but,
    non-strict, not denied); A writes C again. A's cache still reads
    H(C) == H(C), so a cache-TRUSTING skip leaves B's D on disk while post-edit
    commits H(C) — disk/coordinator divergence, and the next reader gets D under a
    coordinator hash of C. The skip must fire only when the file ACTUALLY holds
    the bytes, so A's rewrite lands C.
    """
    rel = "data/shared.txt"
    _track_only(tmp_path)  # tracked but non-strict, before any coordinator spawns
    target = _seed(tmp_path, rel=rel, content=b"v0")

    vol_a = CoherentVolume(tmp_path, managed=(), config=fast_cfg)
    vol_b = CoherentVolume(tmp_path, managed=(), config=fast_cfg)
    try:
        # Warn-mode setup sanity: both attached, and the coordinator is NOT strict
        # for the path (so the later re-grant is not denied — the bug's precondition).
        assert vol_a.is_attached and vol_b.is_attached
        assert not vol_a.strict_mode_active()

        c_bytes = b"content-from-A"
        d_bytes = b"content-from-B-overwrite"

        vol_a.read(rel)
        vol_a.write(rel, c_bytes)  # A commits C: cache := H(C), disk == C
        assert target.read_bytes() == c_bytes

        vol_b.read(rel)  # B SHARED@C
        vol_b.write(rel, d_bytes)  # B commits D: A -> INVALID, disk == D
        assert target.read_bytes() == d_bytes

        # A rewrites the SAME bytes it last committed. Non-strict -> pre-edit
        # re-grants; A's cache still reads H(C). A cache-trusting no-op skip would
        # leave B's D on disk (the bug); the disk-gated skip rewrites C.
        vol_a.write(rel, c_bytes)
        assert target.read_bytes() == c_bytes  # A's intent on disk, not B's stale D
    finally:
        stop_coordinator(tmp_path)


def test_no_op_skip_not_taken_when_disk_file_missing(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-op skip is gated on the on-disk hash, so a cache hit ALONE does not
    skip the write. If the file is gone at write time (``_disk_hash`` -> None,
    and None != new_hash), the write proceeds and recreates it. Pins the
    ``_disk_hash`` missing-file branch the divergence fix relies on."""
    import ccs.adapters.coherent_volume as cv_mod

    _seed(tmp_path, rel="data/x.txt", content=b"orig")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write("data/x.txt", b"committed")  # cache := H("committed"), disk holds it
        (tmp_path / "data/x.txt").unlink()  # disk now diverges from the cached belief

        calls = {"n": 0}
        real_replace = cv_mod.os.replace

        def counting_replace(src: object, dst: object) -> None:
            calls["n"] += 1
            real_replace(src, dst)

        monkeypatch.setattr(cv_mod.os, "replace", counting_replace)
        vol.write("data/x.txt", b"committed")  # same bytes, but the file is GONE
        assert calls["n"] == 1  # skip NOT taken — the write recreated the file
        assert (tmp_path / "data/x.txt").read_bytes() == b"committed"
    finally:
        stop_coordinator(tmp_path)


def test_deny_raises_even_in_degrade_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A coordinator deny is enforcement WORKING, not an infrastructure failure,
    so write() raises on deny in BOTH on_error modes. on_error governs only
    infra failures (unavailable coordinator, watchdog timeout) — not the deny."""
    target = _seed(tmp_path, content=b"v1")
    vol_a = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    vol_b = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        vol_a.read("data/shared.txt")
        vol_b.read("data/shared.txt")
        vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID
        with pytest.raises(CoherenceError):
            vol_b.write("data/shared.txt", b"stale")  # deny still raises in degrade mode
        assert not vol_b.is_degraded  # the deny did not register as infra degradation
        assert target.read_bytes() == b"v2-from-A"
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# Unit 6 — CoherentVolume.write_cas (OCC write path, bypasses the acquire).
#
# write_cas reads (→ SHARED) → derives bytes via make_content(current) →
# commits through /hooks/post-edit-cas. On a version conflict it reacquire()s
# (re-mint + fresh read) and retries, bounded by MAX_CAS_REACQUIRES; on
# exhaustion it raises CasRetriesExhausted. Deny — including the fail-closed
# {ok:false, degraded:true, commit_unconfirmed} body — ALWAYS raises.
# ---------------------------------------------------------------------------


def test_write_cas_first_writer_commits(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A single OCC writer reads then write_cas-commits cleanly (version bumps
    on the coordinator; bytes land on disk)."""
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write_cas("data/shared.txt", lambda cur: cur + b"\nappended")
        assert target.read_bytes() == b"v1\nappended"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_conflict_reacquires_and_converges(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """THE OCC RECOVERY: A reads v1, B reads v1, A commits v2 (B → INVALID).
    A's turn then ends (session-stop releases A's grant — the realistic
    "the other agent finished" case). B.write_cas finds itself INVALID,
    reacquire()s (re-mint + fresh read of A's v2 bytes), re-derives via
    make_content against the fresh view, and commits → converges. No lost
    update: B's commit is an UPDATE rebased on A's v2, not a stale clobber.

    (A's grant must clear before B's OCC commit can land: an OCC writer is S/I
    and never invalidates a peer's MODIFIED grant, so a lingering pessimistic
    holder yields ``other_holder`` until the grant is released or times out —
    the OCC-vs-pessimistic coexistence bound. Here A releases via session-stop.)
    """
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        vol_a.read("data/shared.txt")
        vol_b.read("data/shared.txt")  # B is SHARED@v1

        vol_a.write("data/shared.txt", b"v2-from-A")  # B → INVALID, version → 2
        assert target.read_bytes() == b"v2-from-A"
        # A's turn ends — release its grant so the OCC writer is not blocked by
        # other_holder against A's lingering MODIFIED.
        _end_turn(vol_a)

        seen: list[bytes] = []

        def make(current: bytes) -> bytes:
            # Records the bytes each attempt derives from — proves the retry
            # re-reads A's v2 (not B's stale v1 buffer).
            seen.append(current)
            return current + b"\nrebased-by-B"

        vol_b.write_cas("data/shared.txt", make)
        # Converged on top of A's bytes — the lost update did NOT happen.
        assert target.read_bytes() == b"v2-from-A\nrebased-by-B"
        # The winning attempt derived from A's v2 bytes (re-read via reacquire),
        # never from the original stale v1.
        assert b"v2-from-A" in seen[-1]
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_exhaustion_raises_typed_terminal(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Bounded progress (R6): if every attempt loses the race, write_cas raises
    CasRetriesExhausted (a typed terminal) rather than silently dropping the
    write. Simulated by a peer that commits a fresh version on EVERY attempt,
    between B's read and B's CAS, so every refusal is
    ``caller_in_transient_state`` (the peer's pre-edit invalidated B)."""
    import ccs.adapters.coherent_volume as cv_mod

    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    # Shrink the bound so the test is fast + deterministic.
    original_max = cv_mod.MAX_CAS_REACQUIRES
    cv_mod.MAX_CAS_REACQUIRES = 2
    try:
        vol_a.read("data/shared.txt")
        counter = {"n": 1}

        def make(current: bytes) -> bytes:
            # On every B attempt, A commits a NEW version between B's read and
            # B's CAS → A's pre-edit invalidates B → caller_in_transient_state.
            counter["n"] += 1
            vol_a.reacquire("data/shared.txt")
            vol_a.write("data/shared.txt", f"vA-{counter['n']}".encode())
            return current + b"\nB-attempt"

        with pytest.raises(CasRetriesExhausted) as exc:
            vol_b.write_cas("data/shared.txt", make)
        # The terminal records the artifact + that no write landed for B.
        assert exc.value.attempts == cv_mod.MAX_CAS_REACQUIRES + 1
        assert exc.value.last_conflict_reason == "caller_in_transient_state"
        assert "may still hold the grant" in str(exc.value)
        # B's stale buffer never clobbered A's latest.
        assert b"B-attempt" not in target.read_bytes()
    finally:
        cv_mod.MAX_CAS_REACQUIRES = original_max
        stop_coordinator(tmp_path)


def test_write_cas_exhausted_by_a_held_grant_names_other_holder(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pessimistic write() leaves its writer MODIFIED until the coordinator
    reclaims the grant or the holder's session is stopped, so a peer's write_cas
    is refused ``other_holder`` at an unchanged version on every remaining
    attempt. Each refusal costs one unit of budget, the loop does not wait
    between them (no in-call wait could outlast the grant), and the terminal
    names the LAST refusal: here A writes during B's first attempt, so the first
    refusal is ``caller_in_transient_state`` and the other eight are
    ``other_holder``."""
    import ccs.adapters.coherent_volume as cv_mod

    sleeps: list[float] = []
    monkeypatch.setattr(cv_mod, "time", SimpleNamespace(sleep=sleeps.append))
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        vol_a.read("data/shared.txt")
        calls = {"n": 0}

        def make(current: bytes) -> bytes:
            calls["n"] += 1
            if calls["n"] == 1:
                vol_a.write("data/shared.txt", b"v2-from-A")  # A now holds MODIFIED
            return current + b"\nB"

        with pytest.raises(CasRetriesExhausted) as exc:
            vol_b.write_cas("data/shared.txt", make)
        assert exc.value.attempts == cv_mod.MAX_CAS_REACQUIRES + 1
        assert exc.value.last_current_version == 2
        assert exc.value.last_conflict_reason == "other_holder"
        assert "last_reason=other_holder" in str(exc.value)
        assert "lost the race" not in str(exc.value)
        assert sleeps == []
        assert target.read_bytes() == b"v2-from-A"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_deny_raises_in_both_on_error_modes(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A non-retry-eligible deny (e.g. corruption: expected_version > current)
    ALWAYS raises CoherenceError — in BOTH strict and degrade on_error modes —
    and never silently succeeds. write_cas sources expected_version from its own
    read, so we force corruption by stubbing _read_with_version to over-report
    the version (expected > current → the coordinator returns an error body)."""
    for mode in ("strict", "degrade"):
        target = _seed(tmp_path, content=b"v1")
        vol = CoherentVolume(
            tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg
        )
        try:
            # Seed the artifact on the coordinator (v1 + SHARED) via a real read
            # so the CAS has a real version to compare against.
            assert vol.read("data/shared.txt") == b"v1"
            # Force expected_version far above current → corruption body
            # ({ok:false, reason:commit_cas_corruption...}) which must raise.
            # Not stale, so no reacquire.
            vol._read_with_version = lambda rel, **_kw: coherent_volume_module._ReadResult(  # type: ignore[assignment]
                data=b"v1",
                version=999,
                stale_denied=False,
                owner_generation=0,
                stale_status=False,
                content_differs=False,
            )
            with pytest.raises(CoherenceError):
                vol.write_cas("data/shared.txt", lambda cur: b"should-not-land")
            assert not vol.is_degraded, (
                "a deny is enforcement working, not infra degradation"
            )
            # The unconfirmed write never landed.
            assert target.read_bytes() == b"v1"
        finally:
            stop_coordinator(tmp_path)


def test_write_cas_degrade_body_raises_in_both_modes(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The fail-closed degrade body ({ok:false, degraded:true,
    reason:'commit_unconfirmed'}) must raise in BOTH on_error modes — a degraded
    CAS is unconfirmed, so the client must never assume the write landed.
    Simulated by stubbing the coordinator POST to return that body."""
    for mode in ("strict", "degrade"):
        target = _seed(tmp_path, content=b"v1")
        vol = CoherentVolume(
            tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg
        )
        try:
            real_post = vol._post

            def fake_post(endpoint_path: str, payload: dict, _real=real_post):
                if endpoint_path == "/hooks/post-edit-cas":
                    return {"ok": False, "degraded": True, "reason": "commit_unconfirmed"}
                return _real(endpoint_path, payload)

            vol._post = fake_post  # type: ignore[assignment]
            with pytest.raises(CoherenceError):
                vol.write_cas("data/shared.txt", lambda cur: b"unconfirmed")
            # commit_unconfirmed is a hard failure, not infra degradation, so the
            # deny path does NOT bump the degradation counter.
            assert not vol.is_degraded
            assert target.read_bytes() == b"v1"
        finally:
            stop_coordinator(tmp_path)


def test_write_cas_make_content_sees_current_bytes(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """make_content is invoked with the freshly-read current bytes so the
    caller re-derives intent against the latest state (the OCC update contract,
    same boundary as reacquire())."""
    _seed(tmp_path, content=b"hello")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        captured: list[bytes] = []
        vol.write_cas("data/shared.txt", lambda cur: captured.append(cur) or (cur + b"!"))
        assert captured == [b"hello"]
        assert (tmp_path / "data/shared.txt").read_bytes() == b"hello!"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_degrade_none_response_fails_closed_no_disk_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """FIX 1: in degrade mode, a mid-commit transport failure that ``_post``
    swallowed (returns None for /hooks/post-edit-cas AFTER a version was read)
    must FAIL CLOSED — raise CoherenceError and NOT write the unconfirmed bytes
    to disk. An OCC writer holds no grant, so unconfirmed bytes touching disk
    would re-open the lost update the feature prevents. Before the fix this path
    best-effort _atomic_write'd and returned success."""
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg
    )
    try:
        # Seed a real read so the CAS has a real version comparand (pre-read must
        # succeed; only the commit POST is forced to None).
        assert vol.read("data/shared.txt") == b"v1"
        real_post = vol._post

        def fake_post(endpoint_path: str, payload: dict, _real=real_post):
            # Simulate a degrade-swallowed transport failure on the OCC commit
            # only — every other call (pre-read) behaves normally.
            if endpoint_path == "/hooks/post-edit-cas":
                return None
            return _real(endpoint_path, payload)

        vol._post = fake_post  # type: ignore[assignment]
        with pytest.raises(CoherenceError):
            vol.write_cas("data/shared.txt", lambda cur: b"unconfirmed-bytes")
        # The unconfirmed bytes NEVER touched disk (the whole point of the fix).
        assert target.read_bytes() == b"v1"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_repeatable_for_same_volume(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """FIX 3 (cross-process): the same volume can write_cas the same path TWICE
    back-to-back — both win (version bumps each time) with no D4 'use commit()'
    rejection, because a winning commit_cas ends the committer SHARED on the
    coordinator (an OCC writer holds no grant). Before the fix the first win left
    the agent MODIFIED and the second write_cas hard-failed the D4 precondition."""
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write_cas("data/shared.txt", lambda cur: cur + b"\nfirst")
        assert target.read_bytes() == b"v1\nfirst"
        # Second OCC write by the SAME volume must also land (no D4 rejection).
        vol.write_cas("data/shared.txt", lambda cur: cur + b"\nsecond")
        assert target.read_bytes() == b"v1\nfirst\nsecond"
    finally:
        stop_coordinator(tmp_path)


def test_classify_cas_response_transient_is_conflict(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """AC2 (unit): the stable wire reason 'caller_in_transient_state' classifies
    as a retry-eligible 'conflict' via an EXACT match (no brittle substring) so a
    reword of the coordinator's human message can't break retry routing."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol._classify_cas_response(
            {"ok": False, "reason": "caller_in_transient_state"}
        ) == "conflict"
        # The legacy human message ("commit_cas_not_allowed ... reason=...") is
        # no longer matched — only the exact stable reason routes to conflict.
        assert vol._classify_cas_response(
            {"ok": False, "reason": "commit_cas_not_allowed agent=x reason=caller_in_transient_state"}
        ) == "raise"
        # Sanity: the typed ConflictDetail reasons still classify as conflict.
        assert vol._classify_cas_response(
            {"ok": False, "reason": "version_mismatch", "current_version": 2}
        ) == "conflict"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_transient_reason_reacquires_and_converges(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """AC2 (end-to-end client): a CAS that comes back with the stable transient
    reason 'caller_in_transient_state' is treated as a CONFLICT — write_cas
    reacquires (re-mint + fresh read) and retries to convergence, NOT raise.
    Stubbed so the FIRST OCC commit returns the transient body and the next
    passes through to the real coordinator (which wins)."""
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        # A real read so the artifact + version exist for the eventual real CAS.
        assert vol.read("data/shared.txt") == b"v1"
        real_post = vol._post
        cas_calls = {"n": 0}

        def fake_post(endpoint_path: str, payload: dict, _real=real_post):
            if endpoint_path == "/hooks/post-edit-cas":
                cas_calls["n"] += 1
                if cas_calls["n"] == 1:
                    # First attempt: a peer invalidated us mid-window. Stable
                    # retry-eligible reason — the client must reacquire + retry.
                    return {
                        "ok": False,
                        "reason": "caller_in_transient_state",
                        "current_version": 1,
                    }
            return _real(endpoint_path, payload)

        vol._post = fake_post  # type: ignore[assignment]
        vol.write_cas("data/shared.txt", lambda cur: cur + b"\nrebased")
        # Converged (did NOT raise): the retry landed the rebased bytes.
        assert target.read_bytes() == b"v1\nrebased"
        assert cas_calls["n"] >= 2  # first transient-conflict, then a real win
        # A retry-eligible conflict is not infra degradation.
        assert not vol.is_degraded
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# A5 — single-instance concurrency guard. One instance is single-threaded by
# contract; overlapping use across threads raises (loud misuse) rather than
# splitting an in-flight CAS across re-minted identities. The guard is re-entrant
# for the same thread so the internal write_cas → reacquire → read nesting works.
# ---------------------------------------------------------------------------


def test_overlapping_use_from_another_thread_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A5: while one thread holds an op in flight on an instance, a second
    thread calling a public op on the SAME instance raises CoherenceError —
    concurrent use is detected, not silently allowed."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    held = threading.Event()
    release = threading.Event()

    def holder() -> None:
        # Hold the guard the way an in-flight op does (same mechanism the public
        # ops use), then block so the main thread's op truly overlaps.
        with vol._single_op_guard():
            held.set()
            release.wait(timeout=5)

    t = threading.Thread(target=holder)
    t.start()
    try:
        assert held.wait(timeout=5), "holder thread never acquired the guard"
        # A different thread (this one) calling a public op while the guard is
        # held elsewhere must raise — overlapping single-instance use.
        with pytest.raises(CoherenceError, match="single-threaded"):
            vol.read("data/shared.txt")
        with pytest.raises(CoherenceError, match="single-threaded"):
            vol.write("data/shared.txt", b"nope")
        with pytest.raises(CoherenceError, match="single-threaded"):
            vol.write_cas("data/shared.txt", lambda cur: b"nope")
    finally:
        release.set()
        t.join(timeout=5)
        stop_coordinator(tmp_path)


def test_guard_released_after_op_allows_subsequent_ops(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A5: the guard is released in a finally, so normal SEQUENTIAL use is
    unaffected — back-to-back read/write/write_cas (and the internal
    reacquire-within-write_cas path) all succeed. Also asserts the guard owner is
    cleared after each op so the instance is reusable."""
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        # Sequential reads + write on one instance: guard taken + released each
        # time, never tripping itself.
        assert vol_a.read("data/shared.txt") == b"v1"
        assert vol_b.read("data/shared.txt") == b"v1"
        vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID
        assert vol_a._guard_owner_ident is None  # released after the op
        # A's turn ends — release its MODIFIED grant so B's OCC commit is not
        # blocked by other_holder (the OCC-vs-pessimistic coexistence bound).
        _end_turn(vol_a)

        # The internal reacquire-within-write_cas path: B is INVALID, so
        # write_cas must reacquire() (which calls read()) on the SAME thread —
        # re-entering the guard, not deadlocking or tripping it — and converge.
        vol_b.write_cas("data/shared.txt", lambda cur: cur + b"\nrebased-by-B")
        assert target.read_bytes() == b"v2-from-A\nrebased-by-B"
        assert vol_b._guard_owner_ident is None  # released after write_cas too

        # And a plain direct reacquire() still works (its internal read()
        # re-enters the guard fresh on this thread).
        fresh = vol_b.reacquire("data/shared.txt")
        assert fresh == b"v2-from-A\nrebased-by-B"
        assert vol_b._guard_owner_ident is None  # released after reacquire too
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# T2 — write_cas recovery from a STICKY strict-deny (KTD-T), plus the
# bounded fail-closed terminal. Distinct from the conflict-classify convergence
# above: here B is INVALID at CAS time, so the stale-deny branch (NOT a
# version_mismatch conflict) drives the re-mint + retry.
# ---------------------------------------------------------------------------


def test_write_cas_recovers_from_sticky_strict_deny_and_converges(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """T2: a peer commit leaves THIS volume INVALID (sticky strict-deny). The
    very first OCC read inside write_cas is a strict-deny (stale_denied), so
    write_cas must re-mint identity (clears INVALID + the invalidation transient)
    and commit the rebased bytes — converge, NOT raise. No lost update: B's
    commit is rebased on A's v2."""
    target = _seed(tmp_path, content=b"v1")
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        vol_a.read("data/shared.txt")
        vol_b.read("data/shared.txt")  # B SHARED@v1
        vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID (sticky deny)
        # A's turn ends — release its grant so B's OCC commit is not blocked by
        # other_holder against A's lingering MODIFIED.
        _end_turn(vol_a)

        # Confirm B really is in the sticky-deny state BEFORE write_cas: a bare
        # version-aware read reports stale_denied=True (INVALID, not re-granted).
        _bytes, _ver, stale_denied, _gen, _stale, _differs = vol_b._read_with_version(
            "data/shared.txt"
        )
        assert stale_denied is True, "precondition: B must be a sticky strict-deny"

        seen: list[bytes] = []

        def make(current: bytes) -> bytes:
            seen.append(current)
            return current + b"\nrebased-by-B"

        # write_cas drives the stale-deny branch → re-mint → fresh hash-checked
        # read → CAS.
        vol_b.write_cas("data/shared.txt", make)
        assert target.read_bytes() == b"v2-from-A\nrebased-by-B"
        # The winning attempt derived from A's v2 (re-read after re-mint), never
        # the stale v1 buffer.
        assert b"v2-from-A" in seen[-1]
        assert not vol_b.is_degraded  # a deny/recovery is enforcement, not infra
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_fails_closed_with_typed_terminal_when_reads_stay_denied(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """T2: a comparand read that NEVER clears (every read is a strict-deny) must
    fail closed with a TYPED terminal — never a silent drop and never an
    infinite spin. The loop re-mints identity (NOT reacquire(), whose read would
    route the next comparand read through the coordinator's unchecked
    fresh-SHARED branch — the on-disk lost-update hole) and re-reads, bounded by
    the CONSECUTIVE-denied-streak limit (a denied read never POSTs a commit, so
    it must not consume the CAS budget — that one counts commit attempts).
    Crucially make_content() runs ONLY after a read that is NOT denied, so a
    perpetually-denied artifact derives and commits NOTHING — there is no stale
    buffer to land. Simulated by stubbing _read_with_version to always report
    stale_denied."""
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol.read("data/shared.txt") == b"v1"
        calls = {"n": 0}

        def always_denied(rel: str, **_kw: object):
            # Every comparand read is a deny (a deny carries no confirmed
            # generation and is a stale-status read).
            calls["n"] += 1
            return coherent_volume_module._ReadResult(
                data=b"v1",
                version=1,
                stale_denied=True,
                owner_generation=None,
                stale_status=True,
                content_differs=False,
            )

        vol._read_with_version = always_denied  # type: ignore[assignment]

        made = {"called": False}

        def make(_cur: bytes) -> bytes:
            made["called"] = True
            return b"should-not-commit"

        # Typed terminal (the denied-streak bound), never a silent loss.
        with pytest.raises(CoherenceError, match="stayed strict-denied"):
            vol.write_cas("data/shared.txt", make)
        # Bounded, not an infinite spin: exactly MAX_CAS_REACQUIRES + 1
        # consecutive denied reads, then the typed raise.
        assert calls["n"] == MAX_CAS_REACQUIRES + 1
        # make_content NEVER ran: a denied read derives/commits no bytes, so a
        # stale buffer can never land (the on-disk lost update is impossible).
        assert made["called"] is False
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# Unit 3 — install() builtins.open / io.open shim (opt-in, demo-grade).
#
# Routes managed-path opens through a process-singleton volume. The coverage
# matrix is the contract: builtins.open + pathlib are coordinated; os.open and
# subprocess are NOT. The shim preserves the sequential guard (the lost update
# is denied through open() too, via fail-closed close()).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _uninstall_shim_safety():
    """Safety net: never let an installed open()-shim leak across tests (a leaked
    builtins.open patch would corrupt every later test). No-op when not installed."""
    yield
    uninstall()


def test_shim_coordinates_open_round_trip(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A plain open() read+write of a managed path is coordinated, and the patch
    is reversed on context exit."""
    target = _seed(tmp_path, content=b"v1")
    original_open = builtins.open
    try:
        with coherent_workspace(tmp_path, managed=("data/**",), config=fast_cfg) as vol:
            assert builtins.open is not original_open  # patched while installed
            with open(target) as f:  # read via the shim
                assert f.read() == "v1"
            with open(target, "w") as f:  # write via the shim
                f.write("v2")
            assert target.read_bytes() == b"v2"
            # Proof the write was coordinated (routed through volume.write):
            assert "data/shared.txt" in vol._last_committed_hash
        assert builtins.open is original_open  # restored on exit
    finally:
        stop_coordinator(tmp_path)


def test_shim_install_is_idempotent(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A second install() returns the already-installed singleton volume (one
    workspace per process in v1)."""
    _seed(tmp_path)
    try:
        vol1 = install(tmp_path, managed=("data/**",), config=fast_cfg)
        vol2 = install(tmp_path, managed=("data/**",), config=fast_cfg)
        assert vol1 is vol2
    finally:
        uninstall()
        stop_coordinator(tmp_path)


def test_shim_covers_pathlib_write_text(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """pathlib Path.write_text IS coordinated — it calls io.open, which the shim
    patches alongside builtins.open (patching builtins.open alone would miss it)."""
    (tmp_path / "data").mkdir()
    try:
        with coherent_workspace(tmp_path, managed=("data/**",), config=fast_cfg) as vol:
            (tmp_path / "data/note.txt").write_text("hello")  # pathlib -> io.open -> shim
            assert (tmp_path / "data/note.txt").read_bytes() == b"hello"
            assert "data/note.txt" in vol._last_committed_hash  # coordinated, not bypassed
    finally:
        stop_coordinator(tmp_path)


def test_shim_does_not_cover_os_open_or_subprocess(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The documented boundary: os.open/os.write (raw fds) and subprocess/shell
    redirection bypass the shim — the bytes land but the volume never sees them."""
    target = _seed(tmp_path, rel="data/raw.txt", content=b"orig")
    try:
        with coherent_workspace(tmp_path, managed=("data/**",), config=fast_cfg) as vol:
            fd = os.open(str(target), os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(fd, b"via-os")
            finally:
                os.close(fd)
            subprocess.run(
                ["sh", "-c", f"printf '+sub' >> {target}"], check=True
            )
            assert target.read_bytes() == b"via-os+sub"  # both writes landed on disk
            # ... but neither was coordinated (the documented NOT-COVERED boundary).
            assert "data/raw.txt" not in vol._last_committed_hash
    finally:
        stop_coordinator(tmp_path)


def test_shim_inert_without_install() -> None:
    """Without install(), builtins.open / io.open are the real builtins — importing
    the module has no side effect on open()."""
    assert builtins.open.__name__ == "open"  # not our 'coherent_open' wrapper
    assert builtins.open is io.open


def test_shim_lost_update_is_denied_through_open(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The teeth, through the shim: B reads v1 via open(); a peer (explicit API,
    sibling-attached) commits v2; B writes a v1-derived buffer via open() →
    close() raises fail-closed and the stale bytes never land."""
    target = _seed(tmp_path, content=b"v1")
    try:
        with coherent_workspace(tmp_path, managed=("data/**",), config=fast_cfg):
            # Peer A: explicit API on the same workspace (sibling-attaches to the
            # coordinator the shim singleton spawned).
            vol_a = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
            with open(target) as f:  # B reads v1 via the shim -> SHARED@v1
                b_view = f.read()
            assert b_view == "v1"
            vol_a.write("data/shared.txt", b"v2-from-A")  # B -> INVALID
            # B writes a v1-derived buffer via open() (never re-read) -> deny on close.
            with pytest.raises(CoherenceError):
                with open(target, "w") as f:
                    f.write(b_view + "-edited-by-B")
            assert target.read_bytes() == b"v2-from-A"  # stale write did not land
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("managed", [("data/**",), ("**",)], ids=["data-glob", "all-glob"])
def test_shim_reattaches_after_fork(
    tmp_path: Path, fast_cfg: LifecycleConfig, managed: tuple[str, ...]
) -> None:
    """After a fork drops the endpoint, the next shim'd open lazily re-attaches
    under the child's fresh identity (simulated via a direct _after_fork call to
    avoid forking the coordinator's threads). Under ``**`` the re-attach's own
    reads of ``.coherence/`` files are shim'd opens of managed paths too — they
    must not re-enter the re-attach and recurse."""
    _seed(tmp_path, content=b"v1")
    try:
        with coherent_workspace(tmp_path, managed=managed, config=fast_cfg) as vol:
            assert vol.is_attached
            old_sid = vol.session_id
            vol._after_fork()  # simulate the child-side fork handler
            assert vol._endpoint is None and vol._needs_reattach
            with open(tmp_path / "data/shared.txt") as f:  # lazily re-attaches
                assert f.read() == "v1"
            assert vol.is_attached  # re-attached
            assert vol.session_id != old_sid  # fresh identity
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# Code-review regression tests (PR #91): fail-closed completeness, grant-leak
# safety, the no-op-skip grant finalization, fork/degrade edges, and the shim's
# exceptional-close discard.
# ---------------------------------------------------------------------------


def _raise_oserror(*_args: object, **_kwargs: object) -> None:
    raise OSError("simulated filesystem failure")


def test_fs_write_failure_releases_grant(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the atomic FS write fails AFTER pre-edit granted EXCLUSIVE, the grant is
    released via a post-edit success:false (not orphaned until the sweep), and the
    original OSError propagates.

    Asserted on the COORDINATOR, not only on the request being sent: the release
    must name the incarnation that holds the grant, and one that does not
    addresses the session's parent row and releases nothing — an orphaned
    EXCLUSIVE that still looks like a release on the wire."""
    import ccs.adapters.coherent_volume as cv_mod

    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        posts: list[tuple[str, dict]] = []
        real_post = cv_mod._coordinator_post

        def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            posts.append((path, dict(payload)))
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(cv_mod, "_coordinator_post", spy)
        monkeypatch.setattr(vol, "_atomic_write", _raise_oserror)

        with pytest.raises(OSError):
            vol.write("data/shared.txt", b"never-lands")

        assert any(
            p == "/hooks/post-edit" and pay.get("success") is False for p, pay in posts
        ), "FS-write failure must release the grant via post-edit success=false"
        assert _held(vol, _agent_id(vol)) == {}, (
            "the failure release reached the coordinator but left the grant held"
        )
    finally:
        stop_coordinator(tmp_path)


def _raise_runtime(*_args: object, **_kwargs: object) -> str:
    raise RuntimeError("simulated non-OSError in the pre-write window")


def test_non_oserror_in_write_window_releases_grant(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A NON-OSError raised after pre-edit granted EXCLUSIVE but before the
    post-edit commit (here from _disk_hash, in the no-op-skip check) must still
    release the grant via post-edit success:false — not orphan it until the
    crash-recovery sweep. The original handler caught only OSError around
    _atomic_write, leaving the hashing / disk-read window unprotected."""
    import ccs.adapters.coherent_volume as cv_mod

    _seed(tmp_path, rel="data/x.txt", content=b"orig")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write("data/x.txt", b"same")  # cache := H("same") so _disk_hash is reached next

        posts: list[tuple[str, dict]] = []
        real_post = cv_mod._coordinator_post

        def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            posts.append((path, dict(payload)))
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(cv_mod, "_coordinator_post", spy)
        monkeypatch.setattr(vol, "_disk_hash", _raise_runtime)  # non-OSError in the window

        with pytest.raises(RuntimeError):
            vol.write("data/x.txt", b"same")  # cache hit -> _disk_hash -> RuntimeError

        assert any(
            p == "/hooks/post-edit" and pay.get("success") is False for p, pay in posts
        ), "a non-OSError in the post-grant window must release the grant"
    finally:
        stop_coordinator(tmp_path)


def test_no_op_skip_still_finalizes_grant(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The no-op skip (identical bytes) skips the os.replace but MUST still call
    post-edit to finalize the EXCLUSIVE grant — otherwise the grant leaks. The
    earlier os.replace-spy test cannot see this failure mode."""
    import ccs.adapters.coherent_volume as cv_mod

    _seed(tmp_path, rel="data/x.txt", content=b"orig")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write("data/x.txt", b"same")  # establishes last_committed_hash

        posts: list[str] = []
        real_post = cv_mod._coordinator_post

        def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            posts.append(path)
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(cv_mod, "_coordinator_post", spy)
        vol.write("data/x.txt", b"same")  # identical -> no-op skip
        assert "/hooks/post-edit" in posts, "no-op skip must still finalize the grant"
    finally:
        stop_coordinator(tmp_path)


def test_write_fails_closed_on_watchdog_degrade(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watchdog-timeout degrade ({ok:true, degraded:true}) at pre-edit is an
    infra failure → in strict mode write() fails closed (raises), it does NOT
    proceed. Covers the degrade branch of _check_grant during an active write."""
    import ccs.adapters.coherent_volume as cv_mod

    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)  # strict
    try:
        def fake_post(endpoint: object, path: str, payload: dict, **kwargs: object) -> dict:
            if path == "/hooks/pre-edit":
                return {"ok": True, "degraded": True}  # watchdog-timeout envelope
            return {"ok": True}

        monkeypatch.setattr(cv_mod, "_coordinator_post", fake_post)
        with pytest.raises(CoherenceError):
            vol.write("data/shared.txt", b"x")
    finally:
        stop_coordinator(tmp_path)


def test_shim_exceptional_close_discards_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A `with open(p,'w') as f: ...` block whose body raises DISCARDS the
    buffered write rather than committing a partial/abandoned buffer (and leaks no
    grant — the acquire only happens on a clean commit)."""
    target = _seed(tmp_path, content=b"v1")
    try:
        with coherent_workspace(tmp_path, managed=("data/**",), config=fast_cfg):
            with pytest.raises(RuntimeError):
                with open(target, "w") as f:
                    f.write("garbage-must-not-commit")
                    raise RuntimeError("boom")
            assert target.read_bytes() == b"v1"  # buffer discarded, file unchanged
    finally:
        stop_coordinator(tmp_path)


def test_degrade_foreign_coordinator_drops_endpoint(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Under on_error='degrade', attaching to a foreign non-strict coordinator
    must drop the endpoint (not stay attached to a coordinator that can't enforce
    the managed paths while is_attached reports True)."""
    port = ensure_coordinator(tmp_path, config=fast_cfg)  # foreign: no strict yaml
    assert port > 0
    try:
        with pytest.warns(CoherenceDegradedWarning):
            vol = CoherentVolume(
                tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg
            )
        assert vol.is_degraded
        assert not vol.is_attached  # endpoint dropped — no false sense of coordination
    finally:
        stop_coordinator(tmp_path)


def test_nested_coherent_workspace_keeps_outer_shim(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A nested coherent_workspace must NOT uninstall the outer shim on inner
    exit — the outer context owns the patch."""
    _seed(tmp_path)
    try:
        with coherent_workspace(tmp_path, managed=("data/**",), config=fast_cfg):
            outer_open = builtins.open
            assert outer_open.__name__ == "coherent_open"  # outer installed
            with coherent_workspace(tmp_path, managed=("data/**",), config=fast_cfg):
                pass  # inner exit must NOT uninstall
            assert builtins.open is outer_open  # outer shim still active
        assert builtins.open.__name__ == "open"  # outer exit restores
    finally:
        stop_coordinator(tmp_path)


def test_write_rejects_non_bytes(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """write() rejects non-bytes input with TypeError (the bytes|bytearray contract)."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        with pytest.raises(TypeError):
            vol.write("data/shared.txt", "a string, not bytes")  # type: ignore[arg-type]
    finally:
        stop_coordinator(tmp_path)


def test_write_accepts_bytearray(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """write() accepts bytearray (matches the bytes|bytearray annotation)."""
    target = _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write("data/shared.txt", bytearray(b"from-bytearray"))
        assert target.read_bytes() == b"from-bytearray"
    finally:
        stop_coordinator(tmp_path)


def test_read_outside_root_raises(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A path that escapes the workspace root raises CoherenceError (not a silent
    coordinate-the-wrong-file)."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        with pytest.raises(CoherenceError):
            vol.read("/etc/hostname")  # absolute, outside the workspace root
    finally:
        stop_coordinator(tmp_path)


def test_stale_read_generation_is_cas_retry_eligible() -> None:
    """The read-generation fence reject reason is retry-eligible on the OCC
    cross-process path (reacquire + fresh read) -- classified 'conflict', not a
    terminal 'raise', and matched EXACTLY against the shared constant. Tested
    spawn-free: _classify_cas_response reads only the class attr."""
    from unittest.mock import MagicMock

    from ccs.core.exceptions import STALE_READ_GENERATION_REASON

    assert STALE_READ_GENERATION_REASON in CoherentVolume._CAS_RETRY_REASONS
    classify = CoherentVolume._classify_cas_response
    # spec'd mock: any future self-attribute the classifier grows raises
    # AttributeError here instead of silently returning a MagicMock value.
    stub = MagicMock(spec=CoherentVolume)
    stub._CAS_RETRY_REASONS = CoherentVolume._CAS_RETRY_REASONS
    assert classify(stub, {"ok": False, "reason": STALE_READ_GENERATION_REASON}) == "conflict"
    assert classify(stub, {"ok": True}) == "win"
    assert classify(stub, {"ok": False, "reason": "commit_cas_corruption"}) == "raise"


_CAS_ONCE = {
    "write_cas": lambda vol, rel, version: vol.write_cas(rel, lambda cur: cur + b"+mine"),
    "write_cas_at": lambda vol, rel, version: vol.write_cas_at(rel, version, b"mine"),
}


@pytest.mark.parametrize(
    "body",
    [
        {"ok": False, "reason": ["version_mismatch"]},
        {"ok": False, "reason": {"reason": "version_mismatch"}},
        {"ok": False, "reason": 7},
        {"ok": False},
    ],
    ids=["list", "object", "number", "absent"],
)
@pytest.mark.parametrize("surface", list(_CAS_ONCE))
def test_a_cas_answer_whose_reason_is_not_a_string_is_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    surface: str, body: dict,
) -> None:
    """A 200 answer to the commit that is not a win and whose reason is not a
    string — what a proxy in front of the coordinator could send; the
    coordinator itself always sends one — cannot be classified: not a
    retryable conflict (so no retry), not a deny or a rejection this client
    can name. Whether the commit landed at the coordinator is unknown, so it
    is ``CommitUnconfirmed`` (re-read; retry only if absent), and CAS-first
    means the bytes never touched disk. Before, a list or an object reason
    raised ``TypeError`` from the membership test."""
    from ccs.core.exceptions import CommitUnconfirmed

    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        _data, version = vol.read_with_version(rel)
        real_post = coherent_volume_module._coordinator_post
        commits: list[str] = []

        def unrecognisable(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/post-edit-cas":
                commits.append(path)
                return dict(body)
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", unrecognisable)
        with pytest.raises(CommitUnconfirmed):
            _CAS_ONCE[surface](vol, rel, version)

        assert commits == ["/hooks/post-edit-cas"], "sent once, never retried"
        assert target.read_bytes() == b"v1", "unconfirmed bytes never touch disk"
    finally:
        stop_coordinator(tmp_path)


# ----------------------------------------------------------------------
# on_stale_read knob (PH-A read-surface instance): opt-in enforce on a
# foreign-edit / stale-view strict deny at read time.
# ----------------------------------------------------------------------


def _seed_file(tmp_path: Path, rel: str = "data/x.txt", content: bytes = b"v1") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def test_on_stale_read_invalid_value_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CoherentVolume(tmp_path, managed=("data/**",), on_stale_read="bogus")


def test_on_stale_read_allow_is_default_and_swallows(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Default on_stale_read='allow' is back-compat: a foreign edit is detected
    coordinator-side but read() returns the current bytes, no raise."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol.read("data/x.txt") == b"v1"  # SHARED@v1
        target.write_bytes(b"v2")               # FOREIGN edit (not via the volume)
        assert vol.read("data/x.txt") == b"v2"  # swallowed -> fresh bytes
    finally:
        stop_coordinator(tmp_path)


def test_on_stale_read_raise_surfaces_foreign_edit(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """on_stale_read='raise': a SHARED holder whose tracked file was edited
    out-of-band gets StaleView on read, instead of silently receiving the
    foreign bytes."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_read="raise", config=fast_cfg
    )
    try:
        assert vol.read("data/x.txt") == b"v1"
        target.write_bytes(b"v2")
        with pytest.raises(StaleView):
            vol.read("data/x.txt")
    finally:
        stop_coordinator(tmp_path)


def test_reacquire_recovers_under_on_stale_read_raise(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """reacquire()'s recovery read bypasses on_stale_read='raise' so recovery is
    never blocked: it returns the current bytes without raising."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_read="raise", config=fast_cfg
    )
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"v2")
        with pytest.raises(StaleView):
            vol.read("data/x.txt")
        assert vol.reacquire("data/x.txt") == b"v2"  # recovery does not raise
    finally:
        stop_coordinator(tmp_path)


def test_refused_read_does_not_absolve_foreign_edit_for_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A read refused with StaleView hands the caller no bytes, so it must not
    advance the foreign-edit baseline: a write built from the pre-edit buffer is
    still denied and the foreign bytes survive."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_read="raise", config=fast_cfg
    )
    try:
        buf = vol.read("data/x.txt")
        target.write_bytes(b"HUMAN")              # FOREIGN edit (not via the volume)
        with pytest.raises(StaleView):
            vol.read("data/x.txt")                # refused: caller never sees HUMAN
        with pytest.raises(StaleView):
            vol.write("data/x.txt", buf + b"+agent")
        assert target.read_bytes() == b"HUMAN"    # foreign edit NOT clobbered
    finally:
        stop_coordinator(tmp_path)


def test_fail_closed_read_does_not_absolve_foreign_edit_for_write(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read that fails closed on a watchdog degrade (on_error='strict') also
    hands the caller no bytes, so it must not advance the baseline either."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)  # strict
    real_post = coherent_volume_module._coordinator_post

    def degraded_pre_read(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path == "/hooks/pre-read":
            return {"ok": True, "degraded": True}  # watchdog-timeout envelope
        return real_post(endpoint, path, payload, **kwargs)

    try:
        buf = vol.read("data/x.txt")
        target.write_bytes(b"HUMAN")
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", degraded_pre_read)
        with pytest.raises(CoherenceError):
            vol.read("data/x.txt")                # fails closed: caller never sees HUMAN
        with pytest.raises(StaleView):
            vol.write("data/x.txt", buf + b"+agent")
        assert target.read_bytes() == b"HUMAN"
    finally:
        stop_coordinator(tmp_path)


def test_refused_first_read_still_guards_a_peer_commit_landing_after_it(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused FIRST read returns no bytes, but it must still leave a baseline:
    without one, a write built from no read overwrites a peer's commit that reached
    disk after the refusal."""
    target = _seed_file(tmp_path, content=b"0")
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    reader = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_read="raise", config=fast_cfg
    )
    committed, release = threading.Event(), threading.Event()
    errors: list[BaseException] = []
    real_write = peer._atomic_write

    def lagging_write(abs_path: Path, data: bytes) -> None:
        committed.set()                           # CAS confirmed at the coordinator,
        if not release.wait(10):                  # held off disk until released
            raise AssertionError("peer disk write was never released")
        real_write(abs_path, data)

    monkeypatch.setattr(peer, "_atomic_write", lagging_write)

    def peer_commit(version: int) -> None:
        try:
            peer.write_cas_at("data/x.txt", version, b"1")
        except BaseException as exc:  # surfaced below
            errors.append(exc)
            committed.set()

    thread: threading.Thread | None = None
    try:
        _data, version = peer.read_with_version("data/x.txt")
        thread = threading.Thread(target=peer_commit, args=(version,))
        thread.start()
        assert committed.wait(10) and not errors, errors
        with pytest.raises(StaleView):
            reader.read("data/x.txt")             # first read, inside the window
        release.set()
        thread.join(10)
        assert not errors, errors
        assert target.read_bytes() == b"1"        # the peer's commit is on disk
        with pytest.raises(StaleView):
            reader.write("data/x.txt", b"blind")
        assert target.read_bytes() == b"1"        # ... and NOT overwritten
    finally:
        release.set()
        if thread is not None:
            thread.join(10)
        stop_coordinator(tmp_path)


def test_fail_closed_first_read_still_guards_a_later_write(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FIRST read that fails closed on a watchdog degrade returns no bytes but
    still leaves a baseline, so a write built from no read cannot overwrite an
    out-of-band edit made after it."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)  # strict
    real_post = coherent_volume_module._coordinator_post

    def degraded_pre_read(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path == "/hooks/pre-read":
            return {"ok": True, "degraded": True}  # watchdog-timeout envelope
        return real_post(endpoint, path, payload, **kwargs)

    try:
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", degraded_pre_read)
        with pytest.raises(CoherenceError):
            vol.read("data/x.txt")                # first read fails closed
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)
        target.write_bytes(b"HUMAN")              # FOREIGN edit after the refusal
        with pytest.raises(StaleView):
            vol.write("data/x.txt", b"blind")
        assert target.read_bytes() == b"HUMAN"    # foreign edit NOT clobbered
    finally:
        stop_coordinator(tmp_path)


def test_reacquire_after_refused_read_reseeds_then_write_succeeds(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """reacquire() returns the bytes it reads, so it DOES advance the baseline: a
    write rebuilt from those bytes after a refused read is not false-denied."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_read="raise", config=fast_cfg
    )
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"HUMAN")
        with pytest.raises(StaleView):
            vol.read("data/x.txt")
        fresh = vol.reacquire("data/x.txt")
        assert fresh == b"HUMAN"
        vol.write("data/x.txt", fresh + b"+agent")
        assert target.read_bytes() == b"HUMAN+agent"
    finally:
        stop_coordinator(tmp_path)


def test_fail_closed_reacquire_does_not_absolve_foreign_edit_for_write(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """reacquire() takes a fresh identity before it reads, so the coordinator
    grants the next write. When that read fails closed (on_error='strict') the
    caller gets no bytes, and the baseline is the only guard left: it must not
    advance, or a write from the pre-edit buffer clobbers the foreign edit."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)  # strict
    real_post = coherent_volume_module._coordinator_post

    def degraded_pre_read(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path == "/hooks/pre-read":
            return {"ok": True, "degraded": True}  # watchdog-timeout envelope
        return real_post(endpoint, path, payload, **kwargs)

    try:
        buf = vol.read("data/x.txt")
        target.write_bytes(b"HUMAN")
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", degraded_pre_read)
        with pytest.raises(CoherenceError):
            vol.reacquire("data/x.txt")           # fails closed: caller never sees HUMAN
        with pytest.raises(StaleView):
            vol.write("data/x.txt", buf + b"+agent")
        assert target.read_bytes() == b"HUMAN"
    finally:
        stop_coordinator(tmp_path)


def test_on_stale_read_raise_does_not_fire_on_unmanaged_path(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """on_stale_read='raise' only surfaces a STRICT-mode deny. A foreign edit to a
    path OUTSIDE the managed (strict) globs must NOT raise — the coordinator fires
    the deny only when is_strict_mode(path) is True."""
    target = _seed_file(tmp_path, rel="notes/x.txt", content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_read="raise", config=fast_cfg
    )
    try:
        assert vol.read("notes/x.txt") == b"v1"  # not under managed -> not strict
        target.write_bytes(b"v2")                # foreign edit on a non-strict path
        assert vol.read("notes/x.txt") == b"v2"  # no raise: returns fresh bytes
    finally:
        stop_coordinator(tmp_path)


def test_on_stale_read_raise_independent_of_on_error_degrade(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """on_stale_read and on_error govern independent branches: a foreign-edit deny
    still raises StaleView under on_error='degrade' (degrade governs infra
    failures, not the semantic stale-view deny)."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",),
        on_error="degrade", on_stale_read="raise", config=fast_cfg,
    )
    try:
        assert vol.read("data/x.txt") == b"v1"
        target.write_bytes(b"v2")
        with pytest.raises(StaleView):
            vol.read("data/x.txt")
    finally:
        stop_coordinator(tmp_path)


# ----------------------------------------------------------------------
# SB-23 pre-write content-CAS (on_stale_write): deny a write that would
# clobber a foreign / out-of-band edit since the last read/write.
# ----------------------------------------------------------------------


def test_on_stale_write_invalid_value_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CoherentVolume(tmp_path, managed=("data/**",), on_stale_write="bogus")


def test_write_denies_foreign_edit_by_default(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Default on_stale_write='raise': a write whose target was edited out-of-band
    since the last read is denied (StaleView) and does NOT clobber the foreign bytes."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol.read("data/x.txt") == b"v1"   # seeds baseline = hash(v1)
        target.write_bytes(b"v2")                # FOREIGN edit (not via the volume)
        with pytest.raises(StaleView):
            vol.write("data/x.txt", b"v3")
        assert target.read_bytes() == b"v2"      # foreign edit NOT clobbered
    finally:
        stop_coordinator(tmp_path)


def test_write_guard_seeded_by_read_only(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The read seeds the baseline even with no prior write — a first-read-then-write
    still guards (confirms read-seeding, not write, drives the guard)."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")                   # first access, never written
        target.write_bytes(b"foreign")
        with pytest.raises(StaleView):
            vol.write("data/x.txt", b"mine")
    finally:
        stop_coordinator(tmp_path)


def test_write_succeeds_when_disk_unchanged_and_advances_baseline(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """No foreign edit: the write proceeds; the own write advances the baseline so a
    second consecutive write does not false-deny."""
    _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")
        vol.write("data/x.txt", b"v2")           # disk matches baseline -> proceeds
        assert (tmp_path / "data/x.txt").read_bytes() == b"v2"
        vol.write("data/x.txt", b"v3")           # own write advanced baseline -> no false deny
        assert (tmp_path / "data/x.txt").read_bytes() == b"v3"
    finally:
        stop_coordinator(tmp_path)


def test_reacquire_after_sb23_deny_then_write_succeeds(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """After a foreign-edit deny, reacquire() re-seeds the baseline so the rebuilt
    write succeeds (no false deny after recovery)."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"v2")
        with pytest.raises(StaleView):
            vol.write("data/x.txt", b"clobber")
        assert vol.reacquire("data/x.txt") == b"v2"   # re-seeds baseline = hash(v2)
        vol.write("data/x.txt", b"v3-from-v2")        # rebuilt from fresh -> succeeds
        assert target.read_bytes() == b"v3-from-v2"
    finally:
        stop_coordinator(tmp_path)


def test_on_stale_write_allow_proceeds_and_clobbers(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """on_stale_write='allow' restores pre-SB-23 behavior: the foreign edit is not
    raised; the write proceeds and clobbers."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_write="allow", config=fast_cfg
    )
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"v2")
        vol.write("data/x.txt", b"v3")           # no raise; clobbers v2
        assert target.read_bytes() == b"v3"
    finally:
        stop_coordinator(tmp_path)


def test_write_after_foreign_delete_proceeds(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Foreign delete: the re-hash returns None (absent file) -> no-baseline skip; the
    write proceeds as a normal create (R6)."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")
        target.unlink()                          # FOREIGN delete
        vol.write("data/x.txt", b"recreated")    # None disk hash -> no deny
        assert target.read_bytes() == b"recreated"
    finally:
        stop_coordinator(tmp_path)


def test_post_edit_preempt_still_advances_observed_baseline(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch
) -> None:
    """R5: if the atomic write lands but the post-edit POST then preempts, the
    observed baseline must STILL be advanced (the bytes are on disk) so the instance
    does not self-false-deny on a later write."""
    from ccs.core.exceptions import CommitPreempted

    _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")
        orig_check = vol._check_grant

        def fake_check_grant(resp, rel, *, phase):
            if phase == "post-edit":
                raise CommitPreempted("simulated preempt")
            return orig_check(resp, rel, phase=phase)

        monkeypatch.setattr(vol, "_check_grant", fake_check_grant)
        with pytest.raises(CoherenceError):
            vol.write("data/x.txt", b"v2")
        # Bytes landed AND the observed baseline advanced to hash(v2).
        assert (tmp_path / "data/x.txt").read_bytes() == b"v2"
        assert vol._last_observed_hash["data/x.txt"] == vol._sha256_bytes(b"v2")
    finally:
        stop_coordinator(tmp_path)


def test_install_forwards_on_stale_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """install()/coherent_workspace() forward on_stale_write to the volume."""
    _seed_file(tmp_path, content=b"v1")
    vol = install(tmp_path, managed=("data/**",), on_stale_write="allow", config=fast_cfg)
    try:
        assert vol._on_stale_write == "allow"
    finally:
        uninstall()
        stop_coordinator(tmp_path)


def test_write_unmanaged_path_in_managed_volume_proceeds(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """SB-23 fires only for a path THIS volume manages. A volume managing other/**
    that reads+writes data/... (unmanaged here) does NOT deny a divergent disk —
    that path is not strict, so a coordinated change there is not necessarily a
    foreign edit (the bool(managed) gate would wrongly fire; the per-path match
    must not)."""
    target = _seed_file(tmp_path, rel="data/x.txt", content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("other/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")                   # seeds a baseline, but data/** is unmanaged
        target.write_bytes(b"v2")                # divergent disk on an unmanaged path
        vol.write("data/x.txt", b"v3")           # SB-23 does NOT fire -> proceeds
        assert target.read_bytes() == b"v3"
    finally:
        stop_coordinator(tmp_path)


def test_remint_preserves_other_paths_baseline(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A re-mint (here via reacquire of a DIFFERENT path) must NOT clear the
    foreign-edit baseline of OTHER managed paths — observed-disk hashes are
    disk-scoped, not identity-scoped. Otherwise every write_cas conflict (which
    re-mints) would blind SB-23 on all other previously-read paths."""
    a = _seed_file(tmp_path, rel="data/a.txt", content=b"a1")
    _seed_file(tmp_path, rel="data/b.txt", content=b"b1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/a.txt")               # seed A's baseline
        vol.read("data/b.txt")               # seed B's baseline
        vol.reacquire("data/b.txt")          # re-mints; A's baseline must SURVIVE
        a.write_bytes(b"foreign-a2")         # foreign edit on A
        with pytest.raises(StaleView):
            vol.write("data/a.txt", b"a3")   # A's baseline survived -> deny
    finally:
        stop_coordinator(tmp_path)


def test_write_denies_foreign_edit_independent_of_on_error_degrade(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """SB-23 is adapter-local: the foreign-edit deny fires regardless of on_error
    (the guard does not need the coordinator). Write-side mirror of
    test_on_stale_read_raise_independent_of_on_error_degrade."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg
    )
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"v2")
        with pytest.raises(StaleView):
            vol.write("data/x.txt", b"v3")
    finally:
        stop_coordinator(tmp_path)


def test_write_after_write_cas_win_no_false_deny(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A write_cas win advances the observed baseline, so a following plain write on
    the same path (no foreign edit) does NOT false-deny."""
    _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write_cas("data/x.txt", lambda cur: cur + b"-cas")  # win -> advances baseline
        vol.write("data/x.txt", b"plain-after-cas")             # no foreign edit -> succeeds
        assert (tmp_path / "data/x.txt").read_bytes() == b"plain-after-cas"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_seeds_baseline_then_plain_write_denies_foreign_edit(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The OCC read path seeds the baseline: after a write_cas win, a foreign edit
    then a plain write is denied (confirms the OCC-read seeding feeds the guard)."""
    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write_cas("data/x.txt", lambda cur: b"v2")   # win; baseline=hash(v2), disk=v2
        target.write_bytes(b"foreign-v3")                # foreign edit
        with pytest.raises(StaleView):
            vol.write("data/x.txt", b"v4")
    finally:
        stop_coordinator(tmp_path)


def test_write_managed_path_without_prior_read_proceeds(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """No baseline (a managed path never read through this volume) -> SB-23 skips;
    the write proceeds (the guard needs a prior observation to compare against)."""
    _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write("data/x.txt", b"v2")   # never read -> no baseline -> proceeds
        assert (tmp_path / "data/x.txt").read_bytes() == b"v2"
    finally:
        stop_coordinator(tmp_path)


def test_stale_write_deny_reason_is_byte_stable(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The foreign-edit deny raises with the exact static _STALE_WRITE_DENY_REASON
    constant — no path/hash/timestamp interpolation, so a model's retry loop sees
    identical text every time (KTD-P). Regression pin against future interpolation."""
    from ccs.adapters.coherent_volume import _STALE_WRITE_DENY_REASON

    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"v2")
        with pytest.raises(StaleView) as exc:
            vol.write("data/x.txt", b"v3")
        assert str(exc.value) == _STALE_WRITE_DENY_REASON  # static, no interpolation
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_on_foreign_edit_wedges_not_stale_view(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """SB-23 scope boundary: write_cas is version-CAS, not content-CAS — it does NOT
    raise StaleView. But on a strict path it is NOT unguarded: the read-side hash
    deny makes write_cas's comparand read (_read_with_version) fail closed, so a
    pre-existing foreign edit WEDGES the CAS (ViewWedged after the reacquire budget)
    rather than silently clobbering it. Pins both: SB-23's content-CAS guards only
    the plain write() path, and write_cas still fails closed (no silent loss)."""
    from ccs.core.exceptions import ViewWedged

    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"foreign-v2")    # foreign edit (coordinator version unchanged)
        with pytest.raises(ViewWedged):      # fail-closed — NOT StaleView, NOT a clobber
            vol.write_cas("data/x.txt", lambda cur: cur + b"-cas")
        assert target.read_bytes() == b"foreign-v2"  # foreign edit intact (not clobbered)
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_waits_between_denied_comparand_reads(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The denied comparand read polls a TRANSIENT that clears on ANOTHER writer's
    progress (the window between a peer's confirmed CAS and its disk write), so the
    poll must WAIT — yielding the CPU to that peer — not spin.

    Regression for the intermittent concurrent-writers demo red (2026-09-09): with
    no wait, the streak bound was denominated in local HTTP round trips, so it
    bought only ~40ms of wall clock and a peer descheduled inside that window
    (routine on a loaded CI runner) wedged the loser even though the view would
    have cleared moments later.

    Pins the SCHEDULE, not a scalar floor. An ``elapsed >= sum(schedule)`` assertion
    derives its own threshold from these same constants, so it moves with them:
    zeroing the base makes the bound ``>= 0`` and the fix can be reverted with the
    test still green (measured — the zeroed mutant passed). The recorded sequence
    plus the non-degeneracy assertions below fail on every reachable mutant: a
    removed wait, a zeroed or inverted constant, a schedule flattened to the cap,
    and an off-by-one in the exponent.
    """
    # Without this the rest is vacuous: a zeroed base makes every entry 0.0, so the
    # recorded sequence still matches and the elapsed floor still holds.
    assert DENIED_READ_BACKOFF_BASE_SEC > 0
    assert DENIED_READ_BACKOFF_CAP_SEC >= DENIED_READ_BACKOFF_BASE_SEC
    schedule = [
        min(DENIED_READ_BACKOFF_BASE_SEC * 2**i, DENIED_READ_BACKOFF_CAP_SEC)
        for i in range(MAX_CAS_REACQUIRES)  # one wait per denied read before the bound trips
    ]
    # The restore leg of WorkspaceVersioner waits on the same helper, counted from 1.
    assert [denied_read_backoff_sec(n) for n in range(1, MAX_CAS_REACQUIRES + 1)] == schedule

    # Record what the loop asks to wait, and still wait it. ``time`` is used nowhere
    # else in the adapter, so shimming that module-level name leaves the real
    # time.sleep untouched for every other caller, the coordinator client included.
    waits: list[float] = []

    def recording_sleep(seconds: float) -> None:
        waits.append(seconds)
        time.sleep(seconds)

    monkeypatch.setattr(
        coherent_volume_module, "time", SimpleNamespace(sleep=recording_sleep)
    )

    target = _seed_file(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/x.txt")
        target.write_bytes(b"foreign-v2")  # a denied view that never clears
        waits.clear()  # only the write_cas call's own waits are the subject
        started = time.perf_counter()
        with pytest.raises(ViewWedged):
            vol.write_cas("data/x.txt", lambda cur: cur + b"-cas")
        elapsed = time.perf_counter() - started
        observed = list(waits)
    finally:
        stop_coordinator(tmp_path)

    assert observed == schedule, (
        f"denied-read backoff schedule changed: waited {observed}, expected {schedule}"
    )
    # And the waits were real wall clock, not merely requested.
    assert elapsed >= sum(schedule), (
        f"write_cas wedged after {elapsed:.3f}s but its own backoff schedule is "
        f"{sum(schedule):.3f}s — the recorded waits did not actually elapse"
    )


# ---------------------------------------------------------------------------
# A stable session id across re-mints, and the release of an abandoned write
# grant.
#
# A re-mint sheds the sticky INVALID, the invalidation transient and the read
# generation by landing the next request on a FRESH coordinator row. The row key
# is the session id folded with a per-attempt incarnation that every request
# carries in the subagent field, so the session id stays put while the row moves.
# The abandoned incarnation is a different agent to the coordinator: a grant it
# still holds is foreign to this volume, and write() is the one path that leaves
# one (EXCLUSIVE, then MODIFIED) — so the re-mint that abandons it releases it.
# ---------------------------------------------------------------------------


def _count_stops(
    monkeypatch: pytest.MonkeyPatch, sent: list[tuple[str, dict, str]], vol: CoherentVolume
) -> None:
    """Record every coordinator request as ``(route, body, current incarnation at
    send time)`` and forward it unchanged."""
    real_post = coherent_volume_module._coordinator_post

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        sent.append((path, dict(payload), vol._incarnation))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)


def test_every_attempt_lands_on_its_own_coordinator_row(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The incarnation rides in the coordinator's subagent field, and the
    coordinator reads a value outside that field's shape as "no subagent" —
    silently, with no 400. Every attempt would then share the session's ONE parent
    row, the INVALID a peer's commit leaves there would outlive every re-mint, and
    recovery would wedge. Pinned two ways: the coordinator's own reader returns
    each incarnation verbatim, and its grant rows show each attempt on a row of
    its own while the parent row is never touched."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        parent_row = str(session_to_agent_id(vol.session_id))
        session = vol.session_id
        rows: list[str] = []
        for attempt in range(3):
            if attempt == 0:
                vol.read(rel)
            else:
                vol.reacquire(rel)
            assert read_subagent_id({"agent_id": vol._incarnation}) == vol._incarnation
            rows.append(_agent_id(vol))
            assert _held(vol, rows[-1]) == {rel: "SHARED"}
        assert len(set(rows)) == 3, "a re-mint must move the next request to a new row"
        assert parent_row not in rows
        assert _held(vol, parent_row) == {}
        assert vol.session_id == session
    finally:
        stop_coordinator(tmp_path)


def test_public_identity_accessors_name_what_every_request_sends(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``incarnation`` is the value the next request carries in its subagent
    field and ``agent_id`` is the coordinator row that request lands on, in the
    form ``/status`` reports it — so a caller can name the attempt to a third
    party without reading a private attribute (#262). Pinned against the bodies
    actually sent and against the coordinator's own ``/status`` rows; both change
    on a re-mint, and only on one."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        sent: list[tuple[str, dict, str, str]] = []
        real_post = coherent_volume_module._coordinator_post

        def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            sent.append((path, dict(payload), vol.incarnation, vol.agent_id))
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)

        session = vol.session_id
        # Stable between re-mints: reads do not move the identity.
        before = (vol.incarnation, vol.agent_id)
        vol.read(rel)
        vol.read(rel)
        assert (vol.incarnation, vol.agent_id) == before
        # /status keys this volume's grant on exactly the public agent_id.
        assert _held(vol, vol.agent_id) == {rel: "SHARED"}

        seen_agent_ids = [vol.agent_id]
        vol.reacquire(rel)  # a re-mint: new incarnation, same session
        assert vol.incarnation != before[0]
        assert vol.agent_id not in seen_agent_ids
        assert _held(vol, vol.agent_id) == {rel: "SHARED"}
        seen_agent_ids.append(vol.agent_id)

        _data, version = vol.read_with_version(rel)
        vol.write_cas_at(rel, version, b"v2")  # re-mints before its read
        assert vol.agent_id not in seen_agent_ids
        vol.write_cas(rel, lambda cur: cur + b"+cas")

        assert sent, "the spy saw no coordinator request"
        for route, body, incarnation, agent_id in sent:
            if route == "/hooks/session-stop":
                continue  # the release names the incarnation it abandons
            assert body["agent_id"] == incarnation, route
            assert str(session_to_agent_id(body["session_id"], body["agent_id"])) == agent_id
        assert vol.session_id == session
        # The public accessors read the private state they front; the private
        # names stay, because existing clients read them.
        assert vol.incarnation == vol._incarnation
        assert vol.agent_id == str(session_to_agent_id(vol._session_id, vol._incarnation))
        assert vol.agent_id == _agent_id(vol)
    finally:
        stop_coordinator(tmp_path)


def test_root_is_the_resolved_workspace_path(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``root`` is the path the volume resolved at construction — absolute, with
    symlinks followed — not the spelling the caller passed (#262)."""
    real = tmp_path / "real"
    _seed(real, content=b"v1")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    vol = CoherentVolume("link", managed=("data/**",), config=fast_cfg)
    try:
        assert vol.root == real.resolve()
        assert vol.root.is_absolute()
        assert vol.root == vol._root
        assert vol.read("data/shared.txt") == b"v1"
        assert vol.read(vol.root / "data" / "shared.txt") == b"v1"
    finally:
        stop_coordinator(real)


@pytest.mark.parametrize("name", ["incarnation", "agent_id", "root"])
def test_identity_accessors_are_read_only(
    tmp_path: Path, fast_cfg: LifecycleConfig, name: str
) -> None:
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        with pytest.raises(AttributeError):
            setattr(vol, name, "x")
    finally:
        stop_coordinator(tmp_path)


def test_every_request_names_the_current_incarnation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request the volume sends names its CURRENT incarnation; the one
    exception is the release of an abandoned write grant, which names the
    abandoned one. A request without it lands on the session's parent row: a read
    there registers a view the next attempt does not own, and a release there
    releases nothing."""
    rel, other = "data/shared.txt", "data/other.txt"
    _seed(tmp_path, content=b"v1")
    _seed(tmp_path, rel=other, content=b"o1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        sent: list[tuple[str, dict, str]] = []
        _count_stops(monkeypatch, sent, vol)

        vol.read(rel)
        vol.write_cas(rel, lambda cur: cur + b"+cas")
        vol.write(rel, b"v3")
        _data, version = vol.read_with_version(rel)
        vol.write_cas_at(rel, version, b"v4")  # re-mints: releases write()'s grant
        vol.reacquire(rel)
        _data, version = vol.read_with_version(rel)
        _data, other_version = vol.read_with_version(other)
        vol.atomic_publish([(rel, version, "v5"), (other, other_version, "o2")])

        routes = {route for route, _body, _current in sent}
        assert routes >= {
            "/hooks/pre-read",
            "/hooks/pre-edit",
            "/hooks/post-edit",
            "/hooks/post-edit-cas",
            "/hooks/session-stop",
            "/session/begin",
            "/session/commit_all",
        }, routes
        # write() ran once, so exactly one incarnation took a grant, and the one
        # release must name THAT incarnation -- not merely some other one.
        (write_incarnation,) = {
            body["agent_id"] for route, body, _c in sent if route == "/hooks/pre-edit"
        }
        stops = [body for route, body, _c in sent if route == "/hooks/session-stop"]
        assert [body["agent_id"] for body in stops] == [write_incarnation], (
            "the release did not name the incarnation write() left holding the grant"
        )
        for route, body, current in sent:
            assert body["session_id"] == vol.session_id, route
            if route != "/hooks/session-stop":
                assert body.get("agent_id") == current, route
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("lost", ["pre-edit answer", "post-edit"])
def test_write_cas_at_commits_over_a_grant_its_own_write_left_standing(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, lost: str
) -> None:
    """A write() that took EXCLUSIVE and failed before committing leaves the grant
    standing at the UNCHANGED version. A CAS at that version from the same volume
    re-mints first, and the old incarnation is a different agent to the
    coordinator, so the CAS met its own grant as ``other_holder`` with expected ==
    current — and every retry re-minted into the same refusal (#196). Two ways to
    strand the grant: the acquire landed but its answer was lost, or the commit
    never reached the coordinator. The grant has to be recorded at the acquire,
    not at the commit, or both are missed."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        _data, version = vol.read_with_version(rel)
        real_post = coherent_volume_module._coordinator_post

        def strand_the_grant(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if lost == "pre-edit answer" and path == "/hooks/pre-edit":
                real_post(endpoint, path, payload, **kwargs)  # the acquire lands ...
                raise CoordinatorUnavailable("simulated: the acquire's answer was lost")
            if lost == "post-edit" and path == "/hooks/post-edit" and payload.get("success"):
                raise CoordinatorUnavailable("simulated: the commit never arrived")
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", strand_the_grant)
        # Same bytes as on disk, so the only thing left over is the grant.
        with pytest.raises(CoherenceError):
            vol.write(rel, b"v1")
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)
        stranded = _agent_id(vol)
        assert _held(vol, stranded) == {rel: "EXCLUSIVE"}, "precondition: the grant stands"

        vol.write_cas_at(rel, version, b"v2-cas")

        assert target.read_bytes() == b"v2-cas"
        assert _held(vol, stranded) == {}, "the abandoned incarnation still holds a grant"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_at_after_a_committed_write_on_the_same_volume_commits(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A successful write() leaves this volume's incarnation MODIFIED — the grant
    outlives the call — and a CAS at the NEW version from the same volume met it
    as ``other_holder`` with expected == current (2 == 2): the volume refused by
    its own finished write. The re-mint that abandons the incarnation releases
    the grant, and afterwards the abandoned row holds nothing."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(rel, b"v2-pessimistic")
        writer_row = _agent_id(vol)
        assert _held(vol, writer_row) == {rel: "MODIFIED"}, "precondition: the grant stands"
        _data, version = vol.read_with_version(rel)

        vol.write_cas_at(rel, version, b"v3-cas")

        assert target.read_bytes() == b"v3-cas"
        assert _held(vol, writer_row) == {}, "the abandoned incarnation still holds a grant"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_directly_after_a_committed_write_on_the_same_volume_commits(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """write() then write_cas on the same file, with nothing in between. The
    loop's first attempt used to run under the incarnation write() left
    MODIFIED, so its comparand read was not the None-state, hash-checked read
    the loop relies on, and the commit was refused outright
    (``commit_cas_not_allowed ... occ_is_shared_or_invalid_only``) — write_cas_at
    worked after a write() and write_cas did not. The loop now rotates first
    when the volume's write() may still hold that file, and the grant is released."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(rel, b"v2-pessimistic")
        writer_row = _agent_id(vol)
        assert _held(vol, writer_row) == {rel: "MODIFIED"}, "precondition: the grant stands"

        vol.write_cas(rel, lambda cur: cur + b"+cas")

        assert target.read_bytes() == b"v2-pessimistic+cas"
        assert _held(vol, writer_row) == {}, "the abandoned incarnation still holds a grant"
    finally:
        stop_coordinator(tmp_path)


def test_uncontended_write_cas_without_a_prior_write_does_not_rotate(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rotation write_cas makes after a write() must not become a rotation on
    every write_cas: an optimistic-only commit with no contention is one
    comparand read and one CAS under the volume's current incarnation, with no
    release. Counts the requests, so an unconditional rotate-first cannot pass."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(rel)
        incarnation_before = vol._incarnation
        sent: list[tuple[str, dict, str]] = []
        _count_stops(monkeypatch, sent, vol)

        vol.write_cas(rel, lambda cur: cur + b"+cas")

        assert [route for route, _b, _c in sent] == ["/hooks/pre-read", "/hooks/post-edit-cas"]
        # A rotation that sends no request is invisible to the route list, so
        # compare against the incarnation the volume held before the call.
        assert {body.get("agent_id") for _r, body, _c in sent} == {incarnation_before}, (
            "an uncontended write_cas moved to a new incarnation")
        assert target.read_bytes() == b"v1+cas"
    finally:
        stop_coordinator(tmp_path)


def test_a_denied_write_on_another_path_keeps_the_grant_record(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """One incarnation can hold a write grant on one path and then be denied a
    write on another. The deny proves no grant was taken on the SECOND path only;
    if it dropped the incarnation's record, the grant on the first path would be
    forgotten, and the next optimistic commit there would run under the
    incarnation still holding it and be refused with no peer holding anything."""
    p, q = "data/p.txt", "data/q.txt"
    target = _seed(tmp_path, rel=p, content=b"p1")
    _seed(tmp_path, rel=q, content=b"q1")
    vol, peer = _pair(tmp_path, fast_cfg)
    try:
        vol.read(q)
        vol.write(p, b"p2")
        writer_row = _agent_id(vol)
        assert _held(vol, writer_row) == {p: "MODIFIED", q: "SHARED"}, "precondition"
        peer.read(q)
        peer.write(q, b"q2-peer")  # the volume's row goes INVALID on q
        _end_turn(peer)
        with pytest.raises(StaleView):
            vol.write(q, b"stale")  # denied, on the same incarnation that holds p

        vol.write_cas(p, lambda cur: cur + b"+cas")

        assert target.read_bytes() == b"p2+cas"
        assert _held(vol, writer_row) == {}, "the grant on the first path was never released"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_on_another_path_keeps_the_deny_a_peer_commit_left(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A write_cas right after a write() rotated to a new incarnation even when it
    committed a DIFFERENT file than the one write() held. The rotation is the only
    thing that shed the refusal a peer's commit had left on a third file, so a
    later plain write() of that file from bytes read before the peer's commit was
    admitted as a first write and replaced the peer's bytes. A commit is refused
    for a held grant only on the file it commits, so only that file warrants the
    rotation. ``on_stale_write="allow"`` switches off the local disk-hash check,
    so the coordinator's refusal is the only guard left, as it is when the peer's
    bytes have not reached disk yet or the volume is remote."""
    p, q, r = "data/p.txt", "data/q.txt", "data/r.txt"
    target = _seed(tmp_path, rel=p, content=b"p1")
    _seed(tmp_path, rel=q, content=b"q1")
    _seed(tmp_path, rel=r, content=b"r1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_stale_write="allow", config=fast_cfg
    )
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(p)
        vol.write(q, b"q2")  # the volume's incarnation now holds q
        peer.read(p)
        peer.write(p, b"p2-peer")  # the volume's row goes INVALID on p
        _end_turn(peer)

        vol.write_cas(r, lambda cur: cur + b"+cas")

        with pytest.raises(StaleView):
            vol.write(p, b"p1-stale")  # computed from the bytes read before the peer
        assert target.read_bytes() == b"p2-peer", "a stale write replaced the peer's commit"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("recovery", ["write_cas", "reacquire then write_cas"])
@pytest.mark.parametrize("earlier_write", [False, True])
def test_an_error_answer_after_the_acquire_keeps_the_grant_record(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    recovery: str,
    earlier_write: bool,
) -> None:
    """A pre-edit can take the grant and then fail inside the coordinator, which
    answers ``ok: false`` with an ``internal:`` reason. Only a stale deny proves
    no grant was taken. Read as one, that answer dropped the file from the
    record, so the next re-mint released nothing and the volume was refused by
    its own grant: a direct write_cas was refused outright, and after reacquire()
    every retry met ``other_holder``. An earlier write() of another file on the
    same attempt must not change that."""
    p, q = "data/p.txt", "data/q.txt"
    target = _seed(tmp_path, rel=p, content=b"p1")
    _seed(tmp_path, rel=q, content=b"q1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(p)
        if earlier_write:
            vol.write(q, b"q2")
        real_reground = coordinator_server_module._deliver_pending_reground
        armed = [True]

        def fail_after_the_acquire(coordinator, session_id, body, result, *, abort=None):
            if armed[0] and body.get("path") == p and "content_hash" not in body:
                armed[0] = False
                raise RuntimeError("simulated: failure after the acquire")
            return real_reground(coordinator, session_id, body, result, abort=abort)

        monkeypatch.setattr(
            coordinator_server_module, "_deliver_pending_reground", fail_after_the_acquire
        )
        with pytest.raises(StaleView, match="internal"):
            vol.write(p, b"p2")
        stranded = _agent_id(vol)
        assert _held(vol, stranded).get(p) == "EXCLUSIVE", "precondition: the grant stands"

        if recovery == "reacquire then write_cas":
            vol.reacquire(p)
        vol.write_cas(p, lambda cur: cur + b"+cas")

        assert target.read_bytes() == b"p1+cas"
        assert p not in _held(vol, stranded), "the grant the failed write() took was never released"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_rotates_for_a_file_an_earlier_attempt_still_holds(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A release that was not confirmed leaves an abandoned attempt holding the
    file. A write_cas of that file must rotate first, as it does for the current
    attempt's own grant, so the re-mint retries the release before the commit;
    otherwise the first commit is refused ``other_holder`` by the volume's own
    grant and spends one attempt of the budget. Counts the commits."""
    p, q, x = "data/p.txt", "data/q.txt", "data/x.txt"
    target = _seed(tmp_path, rel=p, content=b"p1")
    _seed(tmp_path, rel=q, content=b"q1")
    _seed(tmp_path, rel=x, content=b"x1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(p, b"p2")
        first_row = _agent_id(vol)
        real_post = coherent_volume_module._coordinator_post
        unconfirmed = [1]
        commits: list[str] = []

        def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/session-stop" and unconfirmed[0]:
                unconfirmed[0] -= 1
                return {"ok": False, "reason": "internal: RuntimeError"}  # not released
            answer = real_post(endpoint, path, payload, **kwargs)
            if path == "/hooks/post-edit-cas":
                commits.append(str(answer.get("reason") if isinstance(answer, dict) else answer))
            return answer

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)
        vol.reacquire(x)  # re-mint: the release of the first attempt is not confirmed
        vol.write(q, b"q2")  # the current attempt holds q, not p
        assert _held(vol, first_row).get(p) == "MODIFIED", "precondition: p is still held"

        vol.write_cas(p, lambda cur: cur + b"+cas")

        assert target.read_bytes() == b"p2+cas"
        assert len(commits) == 1, f"the volume's own grant refused a commit: {commits}"
        assert p not in _held(vol, first_row)
    finally:
        stop_coordinator(tmp_path)


def test_a_degraded_release_answer_is_not_a_confirmed_release(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watchdog-degraded session-stop answers ``ok: true`` whether or not the
    release ran. Treated as confirmed, it would drop the record while the grant
    still stands, and every later commit would be refused by the volume's own
    grant with nothing left to retry the release. It must stay recorded, and the
    next attempt must release it."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        vol.write(rel, b"v2")
        writer_row = _agent_id(vol)
        _data, version = vol.read_with_version(rel)
        real_post = coherent_volume_module._coordinator_post
        degraded_left = [1]

        def degrade_one_release(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/session-stop" and degraded_left[0]:
                degraded_left[0] -= 1
                return {"ok": True, "degraded": True}  # the release did not run
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", degrade_one_release)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CasVersionConflict):
                vol.write_cas_at(rel, version, b"v3")
        assert _held(vol, writer_row) == {rel: "MODIFIED"}, "precondition: the release never ran"

        vol.write_cas_at(rel, version, b"v3")

        assert target.read_bytes() == b"v3"
        assert _held(vol, writer_row) == {}
    finally:
        stop_coordinator(tmp_path)


def test_a_failed_release_stops_the_pass_and_keeps_every_record(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When one release fails, the rest would meet the same coordinator, so the
    pass stops: one re-mint spends at most one failed request, and every
    incarnation it did not confirm stays recorded for the next re-mint."""
    p, q, r = "data/p.txt", "data/q.txt", "data/r.txt"
    for rel in (p, q, r):
        _seed(tmp_path, rel=rel, content=b"1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        real_post = coherent_volume_module._coordinator_post
        stops: list[str] = []
        failing = [True]

        def fail_releases(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/session-stop":
                stops.append(payload["agent_id"])
                if failing[0]:
                    return {"ok": True, "degraded": True}
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", fail_releases)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vol.write(p, b"2")
            first_row = _agent_id(vol)
            vol.reacquire(r)  # re-mint: its release of the first incarnation fails
            vol.write(q, b"2")
            second_row = _agent_id(vol)
            stops.clear()
            vol.reacquire(r)  # re-mint with two incarnations recorded

        def write_grants(row: str) -> dict[str, str]:
            # SHARED rows from the reacquire reads block nothing and are not released.
            return {k: v for k, v in _held(vol, row).items() if v in ("MODIFIED", "EXCLUSIVE")}

        assert len(stops) == 1, f"one re-mint sent {len(stops)} failing releases"
        assert write_grants(first_row) == {p: "MODIFIED"}
        assert write_grants(second_row) == {q: "MODIFIED"}

        failing[0] = False
        vol.reacquire(r)
        assert write_grants(first_row) == {} and write_grants(second_row) == {}, (
            "an incarnation the failed pass skipped was dropped from the record")
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_after_write_and_reacquire_on_the_same_volume_commits(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The retry-loop form of the same self-refusal: after a write(), reacquire()
    moved the volume to a fresh identity but left the write's MODIFIED grant with
    the old one, so every write_cas attempt was refused as ``other_holder`` by the
    volume's own grant, each retry re-minted into the same refusal, and the loop
    exhausted its budget (``CasRetriesExhausted``) with no peer anywhere."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(rel, b"v2-pessimistic")
        assert vol.reacquire(rel) == b"v2-pessimistic"

        vol.write_cas(rel, lambda cur: cur + b"+cas")

        assert target.read_bytes() == b"v2-pessimistic+cas"
    finally:
        stop_coordinator(tmp_path)


def test_re_mint_spends_no_request_on_an_incarnation_without_a_write_grant(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An incarnation that only read, committed optimistically, or had its write
    DENIED holds at most SHARED/INVALID and blocks nothing; releasing it anyway
    would add a round trip to every re-mint — every optimistic retry — for
    nothing. The requests are COUNTED, so an unconditional release cannot pass.
    The second half keeps the zero honest: a write() that takes a grant costs
    exactly ONE release at the transition, however many re-mints follow it."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol, peer = _pair(tmp_path, fast_cfg)
    try:
        vol.read(rel)
        peer.read(rel)
        peer.write(rel, b"v2-peer")  # vol -> INVALID
        _end_turn(peer)
        sent: list[tuple[str, dict, str]] = []
        _count_stops(monkeypatch, sent, vol)

        def stops() -> int:
            return sum(1 for route, _b, _c in sent if route == "/hooks/session-stop")

        with pytest.raises(StaleView):
            vol.write(rel, b"stale")  # denied: no grant was taken
        assert _held(vol, _agent_id(vol)) == {}  # the denied incarnation is INVALID
        assert vol.reacquire(rel) == b"v2-peer"  # re-mint 1 abandons an INVALID row
        assert _held(vol, _agent_id(vol)) == {rel: "SHARED"}
        _data, version = vol.read_with_version(rel)
        vol.write_cas_at(rel, version, b"v3-cas")  # re-mint 2 abandons a SHARED row
        vol.reacquire(rel)  # re-mint 3
        rows = {body["agent_id"] for _r, body, _c in sent if "agent_id" in body}
        assert len(rows) == 4, f"expected four incarnations across three re-mints, saw {len(rows)}"
        assert stops() == 0, "a re-mint released an incarnation that held no write grant"

        vol.write(rel, b"v4-pessimistic")
        for _ in range(2):
            _data, version = vol.read_with_version(rel)
            vol.write_cas_at(rel, version, b"v5-cas")
        assert stops() == 1, "a write() grant costs exactly one release at the transition"
        assert target.read_bytes() == b"v5-cas"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_failed_release_is_kept_and_retried_at_the_next_re_mint(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, on_error: str
) -> None:
    """A release that does not reach the coordinator must not be forgotten: the
    grant it was for still stands, and dropping the record would bring the
    self-refusal back with nothing left to retry it. It fails like any other
    coordinator request (strict raises, degrade warns and the CAS is refused as
    ``other_holder`` — a typed signal, not a silent drop), and the next re-mint
    releases it and commits.

    The failing call runs on a worker with a bounded join, so a release made while
    holding the volume's identity lock — which deadlocks as soon as degrade mode
    records the failure, because that takes the same lock — fails here by name
    instead of hanging the run."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg)
    try:
        vol.write(rel, b"v2-pessimistic")
        writer_row = _agent_id(vol)
        _data, version = vol.read_with_version(rel)
        real_post = coherent_volume_module._coordinator_post
        failures = {"left": 1}

        def release_fails_once(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/session-stop" and failures["left"]:
                failures["left"] -= 1
                raise CoordinatorUnavailable("simulated: the release did not arrive")
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", release_fails_once)
        outcome: dict[str, BaseException | None] = {}

        def attempt() -> None:
            try:
                vol.write_cas_at(rel, version, b"v3-cas")
                outcome["raised"] = None
            except BaseException as exc:  # handed to the test thread below
                outcome["raised"] = exc

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            worker = threading.Thread(target=attempt, daemon=True)
            worker.start()
            worker.join(timeout=30)
        if worker.is_alive():
            pytest.fail(
                "write_cas_at never returned after its release failed: the release "
                "is waiting on the volume's identity lock, which it already holds"
            )
        raised = outcome["raised"]
        assert failures["left"] == 0, "the simulated failure never fired"
        if on_error == "strict":
            assert isinstance(raised, CoherenceError), raised
            assert not isinstance(raised, CasVersionConflict), raised
            assert "session-stop" in str(raised)
        else:
            assert isinstance(raised, CasVersionConflict), raised
            assert raised.reason == "other_holder"
            assert any(issubclass(w.category, CoherenceDegradedWarning) for w in caught)
        assert target.read_bytes() == b"v2-pessimistic"
        assert _held(vol, writer_row) == {rel: "MODIFIED"}

        vol.write_cas_at(rel, version, b"v3-cas")

        assert target.read_bytes() == b"v3-cas"
        assert _held(vol, writer_row) == {}
    finally:
        stop_coordinator(tmp_path)


def test_after_fork_forgets_the_parents_write_grants(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forked child starts with a copy of the grants the parent's write()
    recorded, but they belong to the parent's identity, which is still live in the
    parent. The child must release nothing — not in the fork handler, not at its
    first re-mint — and the parent's row must keep its grant. (Simulated in
    process, as the other fork tests are: the handler runs on the same object.)"""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(rel, b"v2-parent")
        parent_row = _agent_id(vol)
        assert _held(vol, parent_row) == {rel: "MODIFIED"}, "precondition: the grant stands"
        sent: list[tuple[str, dict, str]] = []
        _count_stops(monkeypatch, sent, vol)

        vol._after_fork()  # the child's fork handler
        vol._ensure_attached()  # the child's first operation re-attaches ...
        vol._remint()  # ... and re-mints

        assert [r for r, _b, _c in sent if r == "/hooks/session-stop"] == []
        assert _held(vol, parent_row) == {rel: "MODIFIED"}
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_real_fork_child_releases_nothing_and_the_parent_keeps_its_grant(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same property across a real ``os.fork()``: the registered fork handler
    runs in the child with the parent's endpoint still in hand, so a release there
    would reach the coordinator and revoke the grant the parent's in-flight write
    holds. The child reports how many releases it sent; the parent then checks its
    own grant on the coordinator."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(rel, b"v2-parent")
        parent_row = _agent_id(vol)
        assert _held(vol, parent_row) == {rel: "MODIFIED"}, "precondition: the grant stands"
        sent: list[tuple[str, dict, str]] = []
        _count_stops(monkeypatch, sent, vol)
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child: the fork handler has already run
            os.close(read_fd)
            try:
                vol._remint()  # the child's first re-mint
                stops = sum(1 for r, _b, _c in sent if r == "/hooks/session-stop")
                os.write(write_fd, f"{stops}|{len(vol._grant_incarnations)}".encode())
            except BaseException as exc:  # report, never hang the parent
                os.write(write_fd, f"child raised {exc!r}".encode())
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        ready, _w, _x = select.select([read_fd], [], [], 30)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("timed out waiting for the forked child's release count")
        report = os.read(read_fd, 256).decode("utf-8")
        os.close(read_fd)
        os.waitpid(pid, 0)

        assert report == "0|0", f"child: releases sent | grants still recorded = {report}"
        assert _held(vol, parent_row) == {rel: "MODIFIED"}
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# Caller principal (caller-principal plan, U5 / KTD14)
#
# A volume is a LONG-LIVED caller: it claims its session's principal once, at
# attach, holds it (and the mint nonce) in memory, and presents it on every
# request — it never looks a principal up by the identity it names (KTD5). That
# is accident-resistance, not unreadability: the coordinator stores every
# principal it issued in ``.coherence/state.db``, which any process of the same
# OS user can read. The session is stable across re-mints (U9), so a re-mint
# claims nothing; a forked child is a new session and claims its own. A claim
# whose answer was lost, and a principal the coordinator refuses, are recovered
# by claiming again with the SAME nonce (R20) — never by minting a new one.
# ---------------------------------------------------------------------------

import json  # noqa: E402

from ccs.adapters.claude_code.coordinator_server import (  # noqa: E402
    caller_principal_identity,
)

_PRINCIPAL_HEADER = "Coherence-Caller-Principal"  # frozen duplicate of the wire name


def _count_claims(monkeypatch: pytest.MonkeyPatch, claims: list[str]) -> None:
    """Record the session of every claim a volume sends, forwarding it."""
    real = getattr(coherent_volume_module, "claim_caller_principal", None)

    def spy(endpoint: object, session_id: str, nonce: str) -> object:
        claims.append(session_id)
        return real(endpoint, session_id, nonce)

    monkeypatch.setattr(coherent_volume_module, "claim_caller_principal", spy, raising=False)


def _record_principals(
    monkeypatch: pytest.MonkeyPatch, sent: list[tuple[str, str | None]]
) -> None:
    """Record ``(route, presented principal)`` for every request, forwarding it."""
    real_post = coherent_volume_module._coordinator_post

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        headers = kwargs.get("extra_headers") or {}
        sent.append((path, headers.get(_PRINCIPAL_HEADER)))  # type: ignore[union-attr]
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)


def test_a_volume_claims_once_and_presents_one_principal_across_every_re_mint(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many optimistic writes — uncontended, contended (a peer commits inside
    the retry window, forcing re-mints), a reacquire, and a pessimistic write
    followed by an optimistic commit (whose re-mint releases the stranded grant
    through the require-class stop) — and each volume holds exactly ONE
    binding: claimed once at construction, never at a re-mint, the same
    principal on every request. Prevents claiming at ``_remint`` (a principal
    row and two round trips per attempt, KTD14) and a re-mint that drops the
    principal and gets its writes refused."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"0")
    claims: list[str] = []
    _count_claims(monkeypatch, claims)
    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    try:
        assert sorted(claims) == sorted([vol_a.session_id, vol_b.session_id])
        sent: list[tuple[str, str | None]] = []
        _record_principals(monkeypatch, sent)
        incarnations = {vol_b._incarnation}

        for _ in range(3):
            vol_a.write_cas(rel, lambda cur: str(int(cur) + 1).encode())
        interfered = {"done": False}

        def bump_racing_peer(cur: bytes) -> bytes:
            if not interfered["done"]:
                interfered["done"] = True
                vol_a.write_cas(rel, lambda c: str(int(c) + 10).encode())
            incarnations.add(vol_b._incarnation)
            return str(int(cur) + 1).encode()

        vol_b.write_cas(rel, bump_racing_peer)
        incarnations.add(vol_b._incarnation)
        vol_b.reacquire(rel)
        incarnations.add(vol_b._incarnation)
        vol_b.write(rel, b"100")
        _data, version = vol_b.read_with_version(rel)
        vol_b.write_cas_at(rel, version, b"101")
        incarnations.add(vol_b._incarnation)

        assert len(incarnations) >= 3, "control: the scenario really re-minted"
        assert (tmp_path / rel).read_bytes() == b"101"
        assert sorted(claims) == sorted([vol_a.session_id, vol_b.session_id]), (
            "a re-mint claimed a principal"
        )
        assert {p for _r, p in sent} == {vol_a._principal, vol_b._principal}
        assert None not in {p for _r, p in sent}
        assert "/hooks/session-stop" in {r for r, _p in sent}, "control: a release was sent"
        assert vol_a._principal is not None and vol_b._principal is not None
    finally:
        stop_coordinator(tmp_path)


def test_the_volume_binding_is_the_coordinators_and_distinct_per_volume(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The principal a volume holds is the one the coordinator bound to the
    volume's session (read back from the durable store after the volumes
    stop), and two volumes hold different ones."""
    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry

    vol_a, vol_b = _pair(tmp_path, fast_cfg)
    held = {vol.session_id: vol._principal for vol in (vol_a, vol_b)}
    stop_coordinator(tmp_path)
    registry = SqliteArtifactRegistry(tmp_path / ".coherence" / "state.db")
    try:
        for sid, principal in held.items():
            assert principal is not None
            assert registry.get_caller_principal(caller_principal_identity(sid)) == principal
    finally:
        registry.close()
    assert len(set(held.values())) == 2


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires os.fork (POSIX)")
def test_a_forked_child_claims_its_own_principal_and_writes_under_it(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A forked child is a new session: it discards the principal it inherited
    in memory, claims its own on re-attach, and its require-class commit is
    admitted under it. The parent keeps its own and still writes."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        parent_principal = vol._principal
        assert parent_principal is not None
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:  # child
            os.close(read_fd)
            try:
                inherited = vol._principal
                vol.write(rel, b"v2-child")
                report = {
                    "inherited_dropped": inherited is None,
                    "session": vol.session_id,
                    "principal": vol._principal,
                }
                os.write(write_fd, json.dumps(report).encode())
            except BaseException as exc:  # report, never hang the parent
                os.write(write_fd, json.dumps({"error": repr(exc)}).encode())
            finally:
                os.close(write_fd)
                os._exit(0)
        os.close(write_fd)
        ready, _w, _x = select.select([read_fd], [], [], 30)
        if not ready:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
            pytest.fail("timed out waiting for the forked child's report")
        report = json.loads(os.read(read_fd, 4096).decode("utf-8"))
        os.close(read_fd)
        os.waitpid(pid, 0)

        assert "error" not in report, report
        assert report["inherited_dropped"] is True
        assert report["session"] != vol.session_id
        assert report["principal"] not in (None, parent_principal)
        assert (tmp_path / rel).read_bytes() == b"v2-child"
        vol.write(rel, b"v3-parent")
        assert vol._principal == parent_principal
    finally:
        stop_coordinator(tmp_path)


def test_a_coordinator_that_issues_no_principals_leaves_the_volume_headerless(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 404 on the claim (the sibling Node coordinator, an older Python one)
    is not a failure: the volume attaches, is not degraded, and sends no
    principal header — exactly what it sent before principals existed."""
    from ccs.cli._coherence_client import PrincipalClaim

    monkeypatch.setattr(
        coherent_volume_module, "claim_caller_principal",
        lambda *_a: PrincipalClaim("unsupported"), raising=False,
    )
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        sent: list[tuple[str, str | None]] = []
        _record_principals(monkeypatch, sent)
        assert vol.read("data/shared.txt") == b"v1"
        assert vol.is_attached and not vol.is_degraded
        assert sent and all(p is None for _r, p in sent)
    finally:
        stop_coordinator(tmp_path)


def test_a_claim_refused_at_attach_fails_closed_and_is_never_claimed_again(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim refused because the session is already bound under ANOTHER
    nonce routes through ``on_error``: strict raises (typed, reason
    ``caller_principal_claimed``); degrade warns once and runs without a
    principal. Nothing claims again for that session — not the re-mint, not a
    reacquire, not a write — because every retry would present the same nonce
    and meet the same first-claim refusal (KTD11); a new nonce would be a
    second claimant."""
    from ccs.cli._coherence_client import PrincipalClaim
    from ccs.core.exceptions import CallerPrincipalRefused

    calls: list[str] = []

    def refused(_endpoint: object, session_id: str, _nonce: str) -> PrincipalClaim:
        calls.append(session_id)
        return PrincipalClaim("refused", detail="caller_principal_claimed")

    monkeypatch.setattr(
        coherent_volume_module, "claim_caller_principal", refused, raising=False
    )
    _seed(tmp_path, content=b"v1")
    try:
        with pytest.raises(CallerPrincipalRefused, match="caller principal") as raised:
            CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
        assert raised.value.reason == "caller_principal_claimed"
        calls.clear()
        with pytest.warns(CoherenceDegradedWarning):
            vol = CoherentVolume(
                tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg
            )
        assert vol._principal is None
        vol._remint()
        vol.reacquire("data/shared.txt")
        vol.write("data/shared.txt", b"v2")
        assert len(calls) == 1
    finally:
        stop_coordinator(tmp_path)


def test_an_unconfirmed_claim_is_re_presented_with_the_same_nonce_until_it_settles(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An ``unconfirmed`` claim may have bound (R20): strict still raises at
    construction; degrade warns once — and before each later request the
    volume claims AGAIN with the SAME nonce, never a new one, until an answer
    settles it. A retry with the held nonce is the recovery R20 describes, not
    a re-mint: it adds no binding. Here the answer never settles, so every
    request is preceded by a claim, all presenting one nonce, and the volume
    stays without a principal (the session is unbound, so KTD15 admits it)."""
    from ccs.cli._coherence_client import PrincipalClaim

    nonces: list[str] = []

    def unconfirmed(_endpoint: object, _session_id: str, nonce: str) -> PrincipalClaim:
        nonces.append(nonce)
        return PrincipalClaim("unconfirmed", detail="simulated")

    monkeypatch.setattr(
        coherent_volume_module, "claim_caller_principal", unconfirmed, raising=False
    )
    _seed(tmp_path, content=b"v1")
    try:
        with pytest.raises(CoherenceError, match="caller principal"):
            CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
        nonces.clear()
        with pytest.warns(CoherenceDegradedWarning):
            vol = CoherentVolume(
                tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg
            )
        vol.reacquire("data/shared.txt")
        vol.write("data/shared.txt", b"v2")
        assert len(nonces) >= 3, nonces
        assert set(nonces) == {vol._mint_nonce}
        assert vol._principal is None
        assert (tmp_path / "data/shared.txt").read_bytes() == b"v2"
    finally:
        stop_coordinator(tmp_path)


# --- R20 on the long-lived surface: a lost answer, a refused principal -------


def _lose_the_first_claims_answer(
    monkeypatch: pytest.MonkeyPatch, nonces: list[str]
) -> list[object]:
    """The first claim REACHES the coordinator and binds; its answer is lost
    (reported ``unconfirmed``, as a transport failure after the commit or a
    late bind after the watchdog would be). Later claims pass through. Every
    nonce presented is recorded; the real answers are returned for asserting."""
    from ccs.cli._coherence_client import PrincipalClaim

    real = coherent_volume_module.claim_caller_principal
    answers: list[object] = []

    def lossy(endpoint: object, session_id: str, nonce: str) -> object:
        nonces.append(nonce)
        claim = real(endpoint, session_id, nonce)
        answers.append(claim)
        if len(nonces) == 1:
            assert claim.outcome == "bound", "control: the lost claim really bound"
            return PrincipalClaim("unconfirmed", detail="answer lost")
        return claim

    monkeypatch.setattr(coherent_volume_module, "claim_caller_principal", lossy)
    return answers


def _assert_no_secret_in(text: str, *secrets: object) -> None:
    """R5 on the client: no principal and no mint nonce in anything the volume
    logs, warns or raises."""
    for secret in secrets:
        assert isinstance(secret, str) and secret, "control: a real value to look for"
        assert secret not in text, "a principal or nonce reached the volume's output"


def test_a_claim_whose_answer_was_lost_is_recovered_before_the_next_request(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Degrade mode, the bind committed but its answer was lost: the volume
    re-presents the SAME nonce before its next request, gets back the very
    principal the lost claim bound, and its write is RECORDED (the version
    advances). Before, it ran for its whole lifetime without the principal
    its bound session requires: every commit refused, bytes on disk the
    coordinator never recorded, one warning and then silence."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    nonces: list[str] = []
    answers = _lose_the_first_claims_answer(monkeypatch, nonces)
    caplog.set_level(logging.DEBUG)
    try:
        with pytest.warns(CoherenceDegradedWarning) as warned:
            vol = CoherentVolume(
                tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg
            )
        assert vol._principal is None, "control: the answer was lost"
        vol.read(rel)
        vol.write(rel, b"v2")
        _data, version = vol.read_with_version(rel)

        assert version == 2, "the write was recorded"
        assert len(nonces) == 2 and len(set(nonces)) == 1
        assert vol._principal == answers[0].principal  # type: ignore[attr-defined]
        assert vol.degradation_count == 1
        _assert_no_secret_in(
            caplog.text + " ".join(str(w.message) for w in warned),
            vol._principal, vol._mint_nonce,
        )
    finally:
        stop_coordinator(tmp_path)


def test_a_forked_childs_lost_claim_answer_is_recovered_on_its_next_operation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict mode, the post-fork child (the fork handler run directly): its
    re-attach claim binds but the answer is lost, so that operation raises.
    The next one re-presents the child's SAME nonce, obtains the principal
    its session is bound to, and its write is recorded — it does not run on
    for its lifetime with every require-class request refused."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        nonces: list[str] = []
        answers = _lose_the_first_claims_answer(monkeypatch, nonces)
        vol._after_fork()
        with pytest.raises(CoherenceError, match="caller principal"):
            vol.read(rel)
        vol.read(rel)
        vol.write(rel, b"v2-child")
        _data, version = vol.read_with_version(rel)

        assert version == 2
        assert len(nonces) == 2 and len(set(nonces)) == 1
        assert vol._principal == answers[0].principal  # type: ignore[attr-defined]
    finally:
        stop_coordinator(tmp_path)


def test_a_refused_principal_is_re_claimed_with_the_same_nonce_and_retried_once(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The volume presents a principal the coordinator does not hold for its
    session (as after its binding store was reset): the request is refused
    as foreign, the volume claims with its SAME nonce, adopts the principal
    that claim returns, and retries the refused request ONCE — the write
    lands and is recorded, with one claim and no new nonce."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    caplog.set_level(logging.DEBUG)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(rel)
        bound, nonce = vol._principal, vol._mint_nonce
        stale = "X" * 43
        nonces: list[str] = []
        real_claim = coherent_volume_module.claim_caller_principal
        monkeypatch.setattr(
            coherent_volume_module, "claim_caller_principal",
            lambda ep, sid, n: (nonces.append(n), real_claim(ep, sid, n))[1],
        )
        sent: list[tuple[str, str | None]] = []
        _record_principals(monkeypatch, sent)
        vol._principal = stale

        vol.write(rel, b"v2")
        _data, version = vol.read_with_version(rel)

        assert version == 2
        assert nonces == [nonce]
        assert vol._principal == bound
        refused = [route for route, p in sent if p == stale]
        assert len(refused) == 1, sent
        assert sent[1] == (refused[0], bound), "the refused request is retried once"
        _assert_no_secret_in(caplog.text, bound, stale, nonce)
    finally:
        stop_coordinator(tmp_path)


def _http_error(path: str, status: int, body: object, phrase: str = "Bad Request") -> Exception:
    """An ``HTTPError`` for ``path`` carrying ``body`` as its JSON answer and
    ``phrase`` as the status line's reason phrase."""
    import urllib.error

    raw = io.BytesIO(json.dumps(body).encode())
    return urllib.error.HTTPError(path, status, phrase, {}, raw)  # type: ignore[arg-type]


def _answer_route(
    monkeypatch: pytest.MonkeyPatch, route: str, sent: list[str], answer: object
) -> None:
    """The n-th request to ``route`` (counting from 0) raises ``answer(n)``, or
    is forwarded when that is ``None``; every other request is forwarded.
    Records each route sent."""
    real_post = coherent_volume_module._coordinator_post
    count = {"n": 0}

    def answering(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        sent.append(path)
        if path == route:
            error = answer(count["n"])  # type: ignore[operator]
            count["n"] += 1
            if error is not None:
                raise error
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", answering)


def _refuse_route(
    monkeypatch: pytest.MonkeyPatch, route: str, reason: str, sent: list[str]
) -> None:
    """Every request to ``route`` is refused with the typed principal refusal;
    every other request is forwarded. Records each route sent."""
    body = {"error": f"refused ({reason})", "reason": reason}
    _answer_route(monkeypatch, route, sent, lambda _n: _http_error(route, 400, body))


def test_a_refusal_the_same_nonce_cannot_cure_raises_the_typed_refusal(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the claim with the held nonce hands back the very principal the
    coordinator refused, the refusal is not about staleness: strict raises
    :class:`CallerPrincipalRefused` carrying the wire ``reason`` (never an
    untyped 'HTTP Error 400'), after one claim and WITHOUT resending the
    request. The message carries neither the principal nor the nonce."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    caplog.set_level(logging.DEBUG)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        claims: list[str] = []
        _count_claims(monkeypatch, claims)
        sent: list[str] = []
        _refuse_route(monkeypatch, "/hooks/pre-edit", "caller_principal_foreign", sent)

        with pytest.raises(CallerPrincipalRefused) as raised:
            vol.write(rel, b"v2")

        assert raised.value.reason == "caller_principal_foreign"
        assert raised.value.settled is True, "a claim that answered is a settled refusal"
        assert sent.count("/hooks/pre-edit") == 1
        assert claims == [vol.session_id]
        assert (tmp_path / rel).read_bytes() == b"v1"
        _assert_no_secret_in(
            str(raised.value) + caplog.text, vol._principal, vol._mint_nonce
        )
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_session_bound_under_another_nonce_is_reported_and_never_re_claimed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, on_error: str,
) -> None:
    """The volume no longer holds the nonce its session was bound with, so its
    claim is refused as ``caller_principal_claimed``: the refused request
    raises the typed refusal in BOTH ``on_error`` modes — a principal refusal
    is the coordinator's definite answer, not an infrastructure failure, so
    degrade mode does not write the bytes to disk unrecorded as it would
    around an unreachable coordinator — and the volume stops: the next
    refusal does NOT claim again, so a permanently refused session costs no
    round trip per request."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    caplog.set_level(logging.DEBUG)
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    try:
        bound = vol._principal
        vol._mint_nonce, vol._principal = "Z" * 43, None
        claims: list[str] = []
        _count_claims(monkeypatch, claims)
        raised_text = ""
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            for _ in range(2):
                with pytest.raises(CallerPrincipalRefused) as raised:
                    vol.write(rel, b"v2")
                assert raised.value.reason == "caller_principal_absent"
                assert raised.value.settled is True, "bound under another nonce never changes"
                raised_text += str(raised.value)
        assert claims == [vol.session_id]
        assert (tmp_path / rel).read_bytes() == b"v1", "nothing was written unrecorded"
        assert not [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
        assert vol.degradation_count == 0, "a refusal is an answer, not a degradation"
        _assert_no_secret_in(
            raised_text + caplog.text + " ".join(str(w.message) for w in warned),
            bound, "Z" * 43,
        )
    finally:
        stop_coordinator(tmp_path)


# --- a principal refusal is an answer, in both on_error modes -----------------

_REFUSED_OPERATIONS = {
    "read": lambda vol, rel, version: vol.read(rel),
    "write": lambda vol, rel, version: vol.write(rel, b"v2"),
    "write_cas": lambda vol, rel, version: vol.write_cas(rel, lambda _cur: b"v2"),
    "write_cas_at": lambda vol, rel, version: vol.write_cas_at(rel, version, b"v2"),
    "atomic_publish": lambda vol, rel, version: vol.atomic_publish([(rel, version, b"v2")]),
}


@pytest.mark.parametrize(
    ("presented", "operation"),
    [
        ("foreign", "read"),
        ("foreign", "write"),
        ("foreign", "write_cas"),
        ("foreign", "write_cas_at"),
        ("foreign", "atomic_publish"),
        # An absent principal is refused only on the require class; the read
        # routes admit it, so the refusal lands on the commit.
        ("absent", "write"),
        ("absent", "write_cas"),
        ("absent", "write_cas_at"),
        ("absent", "atomic_publish"),
    ],
)
def test_a_refusal_recovery_cannot_cure_raises_the_typed_refusal_in_degrade_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig, presented: str, operation: str
) -> None:
    """Degrade mode, a session bound under a nonce this volume no longer
    holds: every operation that meets the refusal raises
    :class:`CallerPrincipalRefused` with the wire reason. Degrade governs an
    unreachable or timed-out coordinator; a refusal is the coordinator's
    definite answer, so it is never softened into one — not a
    ``CommitUnconfirmed`` (which sends the caller into unknown-outcome
    reconciliation for a request that changed nothing), not a
    ``CasVersionConflict`` against a degraded version 0, not a degraded read,
    and never bytes written to disk unrecorded."""
    from ccs.core.exceptions import CallerPrincipalRefused, CommitUnconfirmed

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    try:
        _data, version = vol.read_with_version(rel)
        vol._mint_nonce = "Z" * 43  # not the nonce the session was bound with
        vol._principal = "X" * 43 if presented == "foreign" else None

        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            with pytest.raises(CallerPrincipalRefused) as raised:
                _REFUSED_OPERATIONS[operation](vol, rel, version)

        assert raised.value.reason == f"caller_principal_{presented}"
        assert not isinstance(raised.value, (CommitUnconfirmed, CasVersionConflict))
        assert (tmp_path / rel).read_bytes() == b"v1"
        assert vol.degradation_count == 0
        assert not [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
    finally:
        stop_coordinator(tmp_path)


def _serve_in_process(tmp_path: Path, instance_id: str):
    """A strict coordinator for ``data/**`` running in THIS process, reachable
    through the usual pid file, so a test can reach into its registry. Policy
    is loaded once at construction, so the YAML is written first."""
    from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer

    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    for name in ("tracked.yaml", "strict_mode.yaml"):
        (coherence / name).write_text("- data/**\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id=instance_id)
    server.serve_in_thread()
    time.sleep(0.05)
    (coherence / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
    return server


def test_a_degrade_mode_write_lands_no_bytes_when_the_gates_store_read_fails(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On a session's first require-class request the coordinator's
    caller-principal gate reads the durable store — for a client that never
    claimed (KTD15), a cold-cache read on pre-edit — and that read can fail
    (``sqlite3.OperationalError`` once ``busy_timeout`` elapses). The
    coordinator answers it as its work body answers a raise: HTTP 200
    ``{ok: false, reason: "internal: ..."}``, which ``write()`` raises on in
    BOTH modes, landing nothing and recording no grant.

    Prevents the answer being the dispatcher's 500: a degrade-mode volume
    reads a non-200 as an unanswered request, skips its grant check, and
    writes with no grant and no peer invalidation — the lost update the
    volume exists to refuse. The coordinator runs in this process so its
    registry can be made to fail; the volume attaches to it as to any
    sibling-spawned strict coordinator."""
    import sqlite3

    from ccs.cli._coherence_client import PrincipalClaim
    from ccs.core.states import MESIState

    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    server = _serve_in_process(tmp_path, "gate-store-fault")
    # A client that never claims: a coordinator that issues no principal
    # leaves the session unbound, so its pre-edit is the gate's store read.
    monkeypatch.setattr(
        coherent_volume_module, "claim_caller_principal",
        lambda *_a: PrincipalClaim("unsupported"),
    )
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
        assert vol.is_attached and vol.strict_mode_active(), "control: attached, strict"
        assert vol.read(rel) == b"v1"  # accept-class, no principal: no store read

        def failing(identity):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(server.registry, "get_caller_principal", failing)
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            with pytest.raises(StaleView) as raised:
                vol.write(rel, b"v2")

        assert str(raised.value) == "internal: OperationalError", "the typed answer, verbatim"
        assert target.read_bytes() == b"v1", "bytes landed with no grant"
        # The answer has the shape the work body's exception arm gives after an
        # acquire that landed, so it cannot prove no grant was taken: the file
        # stays recorded, and the next re-mint's release of it is a no-op.
        assert rel in vol._grant_incarnations.get(vol._incarnation, set())
        artifact_id = server.registry.lookup_artifact_id_by_name(rel)
        assert artifact_id is not None, "control: the read registered the artifact"
        assert not {
            state for state in server.registry.get_state_map(artifact_id).values()
            if state in (MESIState.EXCLUSIVE, MESIState.MODIFIED)
        }, "the coordinator granted a write the caller was never told about"
        assert vol.degradation_count == 0, "the answer was read as an unanswered request"
        assert not [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
    finally:
        server.shutdown()


# --- the coordinator answers a pre-read with a failure envelope ----------------
#
# Two coordinator-side failures answer a pre-read as HTTP 200 ``{ok: false,
# reason: "internal: <Type>"}``: the caller-principal gate's store read raising
# (a BOUND session whose binding is not cached — what a coordinator restart
# leaves — under a locked registry) and the pre-read work body raising. Neither
# confirms the read was registered: the gate's raise registered nothing, and a
# body raise can fail after one of its registry calls already recorded a grant.
# The volume treats that answer as it treats an unanswered request: strict
# fails closed, degrade warns and counts.

#: FROZEN duplicate of the reason the coordinator's failure envelope carries
#: for the registry's own transient error (only the exception's type).
_INTERNAL_OPERATIONAL_ERROR = "internal: OperationalError"


def _cold_cache(server) -> None:
    """What a coordinator restart leaves behind: every binding durable, the
    in-process principal cache empty, so a bound session's next request is
    the gate's store read."""
    with server.service._caller_principal_lock:
        server.service._caller_principals.clear()


def _fail_the_pre_read(server, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """Make the coordinator answer the next pre-read with its failure envelope
    from one of its two arms: ``gate`` — the caller-principal gate's store
    read raises (cold cache, locked registry); ``body`` — the pre-read work
    body raises (the watchdog call itself)."""
    import sqlite3

    def raising(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    if fault == "gate":
        _cold_cache(server)
        monkeypatch.setattr(server.registry, "get_caller_principal", raising)
    else:
        monkeypatch.setattr(server, "run_with_watchdog", raising)


_READERS = {
    "read": lambda vol, rel: vol.read(rel),
    "read_with_version": lambda vol, rel: vol.read_with_version(rel),
}


@pytest.mark.parametrize("reader", sorted(_READERS))
@pytest.mark.parametrize("fault", ["gate", "body"])
def test_a_strict_read_the_coordinator_answers_with_a_failure_fails_closed_unregistered(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, fault: str, reader: str,
) -> None:
    """A strict volume whose pre-read the coordinator answers with its failure
    envelope raises the typed infrastructure failure, naming the envelope's
    reason (the exception's type, nothing else), and hands back nothing: no
    bytes, no artifact row or SHARED view on the coordinator. The path's first
    baseline still records what the disk held, so the next write is checked.
    Once the coordinator recovers the same read registers as usual.

    Prevents the read going through as REGISTERED: the volume returned the
    bytes with no view recorded, so a peer's later commit invalidated nothing
    and this instance's next write landed over it — the lost update the
    strict mode exists to refuse — while ``read()``'s contract that a read
    whose coherence cannot be registered fails closed read as true."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    caplog.set_level(logging.DEBUG)
    server = _serve_in_process(tmp_path, f"pre-read-{fault}-strict")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert vol.is_attached and vol.principal_claim_outcome == "bound", "control: bound"
        with monkeypatch.context() as faulted:
            _fail_the_pre_read(server, faulted, fault)
            with pytest.raises(CoherenceError) as raised:
                _READERS[reader](vol, rel)

        assert _INTERNAL_OPERATIONAL_ERROR in str(raised.value), "the envelope's reason is named"
        assert rel in str(raised.value)
        # A body raise can fail after part of the read was registered, so the
        # message may only say the registration is unconfirmed, never that it
        # did not happen.
        assert str(raised.value).endswith("the read is not confirmed as registered"), str(raised.value)
        assert vol.degradation_count == 0, "a strict volume degraded instead of raising"
        # This is the path's first read, so the first-read rule records what
        # the disk held even though the read failed: a later write() is still
        # checked against it. That seed is the disk's bytes and nothing else,
        # and it never replaces a baseline, so it absolves nothing (a failed
        # read after an earlier one leaves that baseline where it was; see
        # test_a_read_that_raises_leaves_the_foreign_edit_baseline_where_it_was).
        assert vol._last_observed_hash == {rel: _sha(b"v1")}, "more than the first-read seed"
        assert server.registry.lookup_artifact_id_by_name(rel) is None, "the read registered"
        _assert_no_secret_in(str(raised.value) + caplog.text, vol._principal, vol._mint_nonce)

        assert vol.read(rel) == b"v1", "control: the coordinator recovered, the read registers"
        assert server.registry.lookup_artifact_id_by_name(rel) is not None
    finally:
        server.shutdown()


@pytest.mark.parametrize("reader", sorted(_READERS))
@pytest.mark.parametrize("fault", ["gate", "body"])
def test_a_first_read_that_fails_still_guards_the_next_write_against_an_out_of_band_edit(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    fault: str, reader: str,
) -> None:
    """A path's first read records what the disk held before the coordinator's
    answer can fail it. A strict read of a path this volume never read is
    answered with the failure envelope and raises; the file is then edited out
    of band; a later ``write()`` of it must be refused as a foreign edit and
    leave the edit on disk.

    Prevents a failed first read leaving no baseline at all: ``write()`` then
    had nothing to compare the disk against and overwrote the edit, where the
    same sequence after a successful first read is refused. The first
    baseline never replaces an earlier one, so it cannot absolve an edit made
    after an earlier read."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    server = _serve_in_process(tmp_path, f"first-read-{fault}-{reader}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        with monkeypatch.context() as faulted:
            _fail_the_pre_read(server, faulted, fault)
            with pytest.raises(CoherenceError):
                _READERS[reader](vol, rel)
        target.write_bytes(b"out-of-band")

        with pytest.raises(StaleView):
            vol.write(rel, b"from-the-volume")

        assert target.read_bytes() == b"out-of-band", "write() overwrote the out-of-band edit"
    finally:
        server.shutdown()


@pytest.mark.parametrize("fails", [False, True])
def test_a_verification_read_of_a_path_never_read_records_no_baseline(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    """A verification read (``observe=False``, what the effect fence re-reads
    with) of a path this volume never read records no first baseline, whether
    its pre-read succeeds or is answered with the failure envelope. Only an
    observing read records one.

    Prevents a check that heals what it checks: the first-baseline seed is
    guarded by ``observe`` alone, and a seed from a verification read would
    make the fence's comparison an observation."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    server = _serve_in_process(tmp_path, f"verify-read-{fails}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        with monkeypatch.context() as faulted:
            if fails:
                _fail_the_pre_read(server, faulted, "gate")
                with pytest.raises(CoherenceError):
                    vol.read_with_version_generation(rel, observe=False)
            else:
                vol.read_with_version_generation(rel, observe=False)

        assert rel not in vol._last_observed_hash, "a verification read recorded a baseline"
    finally:
        server.shutdown()


@pytest.mark.parametrize("fault", ["gate", "body"])
def test_a_degrade_read_the_coordinator_answers_with_a_failure_degrades_like_an_unanswered_one(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """A degrade-mode volume returns the bytes, warns once and counts the
    degradation — exactly what it does for a request the coordinator never
    answered — and its OCC read reports the unconfirmed version ``0`` so a
    CAS from it loses cleanly. Before, the read went through with no signal
    at all: ``is_degraded`` stayed False while no view was registered."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    server = _serve_in_process(tmp_path, f"pre-read-{fault}-degrade")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
        assert vol.principal_claim_outcome == "bound" and not vol.is_degraded, "control"
        _fail_the_pre_read(server, monkeypatch, fault)
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            data = vol.read(rel)
            _data, version = vol.read_with_version(rel)

        assert data == b"v1"
        assert version == 0, "an unregistered OCC read reported a confirmed version"
        assert vol.is_degraded and vol.degradation_count == 2
        degraded = [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
        assert len(degraded) == 1, "warned once per instance"
        assert _INTERNAL_OPERATIONAL_ERROR in str(degraded[0].message)
    finally:
        server.shutdown()


def test_the_principal_claim_outcome_is_readable_on_every_arm(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``principal_claim_outcome`` is the read-only view of the session's last
    claim — what the MCP ``swg_status`` reports so an agent can tell "this
    session lost coordination" from a transient before its next write is
    refused. ``bound`` after attach; ``unsupported`` against a coordinator
    that issues none; ``refused`` once the session is known to be bound under
    another nonce (durable: nothing is claimed again); ``not_attempted``
    while no claim has been made for the session — a forked child before it
    re-attaches. Never a principal, never a nonce."""
    from ccs.cli._coherence_client import PrincipalClaim
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol.principal_claim_outcome == "bound"
        bound, nonce = vol._principal, vol._mint_nonce
        assert bound and nonce and bound not in ("bound", "refused")  # control

        vol._mint_nonce, vol._principal = "Z" * 43, None
        with pytest.raises(CallerPrincipalRefused):
            vol.write(rel, b"v2")
        assert vol.principal_claim_outcome == "refused"

        vol._after_fork()  # the child's session: nothing claimed yet
        assert vol.principal_claim_outcome == "not_attempted"

        monkeypatch.setattr(
            coherent_volume_module, "claim_caller_principal",
            lambda *_a: PrincipalClaim("unsupported"),
        )
        vol._ensure_attached()
        assert vol.principal_claim_outcome == "unsupported"
        with pytest.raises(AttributeError):
            vol.principal_claim_outcome = "bound"  # type: ignore[misc]
    finally:
        stop_coordinator(tmp_path)


def test_a_refused_acquire_leaves_no_grant_for_a_re_mint_to_release(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A principal refusal of ``pre-edit`` proves no grant was taken — the
    coordinator refuses before any mutation — exactly as an explicit deny
    does, so the incarnation is not recorded as holding one. Otherwise the
    next re-mint spends a ``session-stop`` releasing a grant that never
    existed. Control: an acquire the coordinator ADMITS is released by it."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        bound, nonce = vol._principal, vol._mint_nonce
        vol._mint_nonce, vol._principal = "Z" * 43, None
        with pytest.raises(CallerPrincipalRefused):
            vol.write(rel, b"v2")
        vol._mint_nonce, vol._principal = nonce, bound
        sent: list[tuple[str, str | None]] = []
        _record_principals(monkeypatch, sent)

        vol.reacquire(rel)
        assert "/hooks/session-stop" not in {r for r, _p in sent}

        vol.write(rel, b"v2")
        sent.clear()
        vol.reacquire(rel)
        assert "/hooks/session-stop" in {r for r, _p in sent}, "control: a real grant is released"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_request_refused_again_after_recovery_is_retried_exactly_once(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, on_error: str
) -> None:
    """Refused, recovered (the claim with the held nonce returns a principal
    that differs from the one presented), and refused AGAIN: the volume
    retries the request exactly ONCE and makes exactly ONE claim, then raises
    the typed refusal saying it was refused again — in both ``on_error``
    modes. A client that kept re-claiming would turn one refused request
    into an unbounded claim/request loop."""
    from ccs.cli._coherence_client import PRINCIPAL_REFUSED_AGAIN, PrincipalClaim
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    try:
        claims: list[str] = []

        def always_a_new_principal(_endpoint: object, _sid: str, nonce: str) -> PrincipalClaim:
            claims.append(nonce)
            return PrincipalClaim("bound", principal=f"{len(claims):043d}")

        monkeypatch.setattr(coherent_volume_module, "claim_caller_principal", always_a_new_principal)
        sent: list[str] = []
        _refuse_route(monkeypatch, "/hooks/pre-edit", "caller_principal_foreign", sent)

        with pytest.raises(CallerPrincipalRefused) as raised:
            vol.write(rel, b"v2")

        assert sent.count("/hooks/pre-edit") == 2, sent
        assert claims == [vol._mint_nonce]
        assert raised.value.reason == "caller_principal_foreign"
        assert raised.value.settled is True, "refused again after a confirmed claim is settled"
        assert PRINCIPAL_REFUSED_AGAIN in str(raised.value)
        assert (tmp_path / rel).read_bytes() == b"v1"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("route", ["/hooks/pre-edit", "/hooks/post-edit"])
def test_a_refusal_whose_recovery_claim_is_unconfirmed_is_not_settled_and_the_next_request_claims_again(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, route: str,
) -> None:
    """A request is refused for its principal and the recovery claim's answer
    is lost (a transport blip): the refused request still raises the typed
    refusal, but one that says it is NOT settled — the volume adopted
    ``unconfirmed``, and its very next request claims again with the SAME
    nonce by itself, obtains the bound principal, and lands. The durable arms
    (bound under another nonce; the claim handed back the refused principal)
    raise ``settled`` refusals. On the ``post-edit`` arm the not-settled value
    survives the re-raise that prefixes what the write left behind.

    Prevents the refusal over-claiming durability: a consumer that reads every
    principal refusal as "this session can never regain coordination" tells
    the agent to start a new session for a state the next call cures."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    caplog.set_level(logging.DEBUG)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        bound, nonce = vol._principal, vol._mint_nonce
        assert bound and nonce, "control: the session claimed at attach"
        nonces: list[str] = []
        answers = _lose_the_first_claims_answer(monkeypatch, nonces)
        sent: list[str] = []
        refusal = {"error": "refused (caller_principal_foreign)", "reason": "caller_principal_foreign"}
        # Only the FIRST request to ``route`` is refused; the coordinator
        # itself admits every one, so the next write lands once the claim is
        # settled — a refusal of the request, not of the session.
        _answer_route(
            monkeypatch, route, sent,
            lambda n: _http_error(route, 400, refusal) if n == 0 else None,
        )

        with pytest.raises(CallerPrincipalRefused) as raised:
            vol.write(rel, b"v2")

        assert raised.value.reason == "caller_principal_foreign"
        assert raised.value.settled is False, "an unconfirmed recovery claim is not a settled refusal"
        assert vol.principal_claim_outcome == "unconfirmed"
        assert nonces == [nonce], "the recovery claim presented the binding's own nonce"
        assert sent.count(route) == 1, "an unconfirmed claim retried nothing"

        vol.write(rel, b"v2")

        assert nonces == [nonce, nonce], "the next request claimed again by itself, same nonce"
        assert vol._principal == bound == answers[0].principal  # type: ignore[attr-defined]
        assert vol.principal_claim_outcome == "bound"
        assert target.read_bytes() == b"v2"
        _assert_no_secret_in(str(raised.value) + caplog.text, bound, nonce)
    finally:
        stop_coordinator(tmp_path)


# --- a reason that is not a string is not a principal refusal -----------------


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize(
    "reason",
    [["caller_principal_foreign"], {"reason": "caller_principal_absent"}],
    ids=["list", "object"],
)
def test_a_400_whose_reason_is_not_a_string_is_an_ordinary_failed_request(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    on_error: str, reason: object,
) -> None:
    """A 400 whose JSON ``reason`` is not a string — a list or an object, as a
    proxy or gateway in front of the coordinator might send — is not a
    caller-principal refusal. Classifying it used to raise ``TypeError`` out
    of the volume (an unhashable value reached a set-membership test), past
    degrade mode entirely. It is an ordinary rejected request, routed through
    ``on_error``: strict raises ``CoherenceError``; degrade warns and carries
    on as it does for any other rejected acquire. No recovery claim is made."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    try:
        claims: list[str] = []
        _count_claims(monkeypatch, claims)
        sent: list[str] = []
        body = {"error": "bad request", "reason": reason}
        _answer_route(
            monkeypatch, "/hooks/pre-edit", sent,
            lambda _n: _http_error("/hooks/pre-edit", 400, body),
        )
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            try:
                vol.write(rel, b"v2")
            except CoherenceError as exc:
                raised: CoherenceError | None = exc
            else:
                raised = None

        assert sent.count("/hooks/pre-edit") == 1
        assert claims == []
        if on_error == "strict":
            assert raised is not None and not isinstance(raised, CallerPrincipalRefused)
            assert (tmp_path / rel).read_bytes() == b"v1"
        else:
            assert [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
            assert not isinstance(raised, CallerPrincipalRefused)
    finally:
        stop_coordinator(tmp_path)


# --- no coordinator-supplied text reaches what the volume reports -------------


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize("scenario", ["the_claim_echoes", "the_retry_echoes"])
def test_no_coordinator_supplied_text_reaches_what_a_volume_raises_warns_or_logs(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, on_error: str, scenario: str,
) -> None:
    """A coordinator that echoes the nonce and principal it was sent in every
    free-text field — a refusal's ``error`` and ``detail``, a claim's
    ``reason``, ``detail`` and ``error``, and the status line's reason phrase
    — cannot get either into what the volume raises, warns or logs on the
    claim and recovery paths. Only a reason from the frozen vocabulary is
    repeated (anything else is reported as ``unrecognised``), and a failed
    HTTP request is reported by its status code.

    - ``the_claim_echoes``: the read is refused, and the recovery claim's
      answer is not a confirmation: its reason is the echo.
    - ``the_retry_echoes``: the recovery claim binds a new principal, and the
      retried read is answered 500 with the echo."""
    from ccs.cli import _coherence_client
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    caplog.set_level(logging.DEBUG)
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    try:
        adopted = "Q" * 43
        held = (vol._principal, vol._mint_nonce, adopted)
        echo = " ".join(held)  # type: ignore[arg-type]
        real_post = _coherence_client.post

        def claim_echoes(endpoint: object, path: str, body: dict, **kwargs: object) -> object:
            if path != "/principal/claim":
                return real_post(endpoint, path, body, **kwargs)  # type: ignore[arg-type]
            if scenario == "the_retry_echoes":
                return {"ok": True, "principal": adopted}
            return {"ok": False, "reason": echo, "detail": echo, "error": echo}

        monkeypatch.setattr(_coherence_client, "post", claim_echoes)
        refusal = {"error": echo, "detail": echo, "reason": "caller_principal_foreign"}
        answers = [
            _http_error("/hooks/pre-read", 400, refusal, phrase=echo),
            _http_error("/hooks/pre-read", 500, {"error": echo, "detail": echo}, phrase=echo),
        ]
        sent: list[str] = []
        _answer_route(monkeypatch, "/hooks/pre-read", sent, lambda n: answers[n])

        reported: list[str] = []
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            try:
                vol.read(rel)
            except CoherenceError as exc:
                reported.append(f"{type(exc).__name__}: {exc}")
        reported += [str(w.message) for w in warned]
        text = " ".join(reported) + caplog.text

        assert reported, "control: the scenario reported something"
        if scenario == "the_claim_echoes":
            assert "unrecognised" in text
            assert sent.count("/hooks/pre-read") == 1
        else:
            assert "HTTP 500" in text
            assert sent.count("/hooks/pre-read") == 2
        assert not (scenario == "the_retry_echoes" and "CallerPrincipalRefused" in text)
        if scenario == "the_claim_echoes":
            assert any(r.startswith(CallerPrincipalRefused.__name__) for r in reported)
        _assert_no_secret_in(text, *held)
    finally:
        stop_coordinator(tmp_path)


# --- a request that raises leaves the client's per-path beliefs untouched -----


def _sha(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _lose_the_next_claims_answer(monkeypatch: pytest.MonkeyPatch, nonces: list[str]) -> None:
    """The NEXT claim's answer is lost (reported ``unconfirmed``); later claims
    go to the coordinator. Records the nonce every claim presents."""
    from ccs.cli._coherence_client import PrincipalClaim

    real = coherent_volume_module.claim_caller_principal

    def lossy(endpoint: object, session_id: str, nonce: str) -> object:
        nonces.append(nonce)
        if len(nonces) == 1:
            return PrincipalClaim("unconfirmed", detail="answer lost")
        return real(endpoint, session_id, nonce)

    monkeypatch.setattr(coherent_volume_module, "claim_caller_principal", lossy)


_REFUSED_READS = {
    "read": lambda vol, rel: vol.read(rel),
    "read_with_version": lambda vol, rel: vol.read_with_version(rel),
    # write_cas's comparand read is the refused request here.
    "write_cas": lambda vol, rel: vol.write_cas(rel, lambda cur: cur + b"+mine"),
}


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize("refused", ["read", "read_with_version", "write_cas", "none"])
def test_a_read_refused_for_its_principal_does_not_absolve_a_foreign_edit(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    on_error: str, refused: str,
) -> None:
    """The volume reads v1; the file is then rewritten out of band (the
    coordinator never hears of it). A read is refused for its principal — the
    presented one is no longer the bound one, and the recovery claim's answer
    is lost once — so it raises and the caller never receives the foreign
    bytes. The next write() recovers the principal with the SAME nonce, so it
    reaches the coordinator admitted; it must still be DENIED as a foreign
    edit, and the disk must keep the foreign bytes.

    Before, the refused read had already moved the SB-23 baseline to the
    foreign bytes it never returned, so the write clobbered the edit — in
    strict and degrade mode, through read, read_with_version and write_cas's
    comparand read. ``none`` is the control: without the refused read the same
    write is denied."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    try:
        assert vol.read(rel) == b"v1"
        bound = vol._principal
        target.write_bytes(b"FOREIGN")
        nonces: list[str] = []
        if refused != "none":
            _lose_the_next_claims_answer(monkeypatch, nonces)
            vol._principal = "X" * 43  # no longer the bound one, as after a store reset
            with pytest.raises(CallerPrincipalRefused) as raised:
                _REFUSED_READS[refused](vol, rel)
            assert raised.value.reason == "caller_principal_foreign"
            assert vol._last_observed_hash[rel] == _sha(b"v1"), "the refused read moved the baseline"
        sent: list[tuple[str, str | None]] = []
        _record_principals(monkeypatch, sent)

        with pytest.raises(StaleView) as denied:
            vol.write(rel, b"mine-derived-from-v1")

        assert str(denied.value) == coherent_volume_module._STALE_WRITE_DENY_REASON
        assert target.read_bytes() == b"FOREIGN", "the out-of-band edit was clobbered"
        assert ("/hooks/pre-edit", bound) in sent, "control: the write reached the coordinator admitted"
        if refused != "none":
            assert nonces == [vol._mint_nonce] * 2, "recovered with the SAME nonce, once lost"
    finally:
        stop_coordinator(tmp_path)


_READS_THAT_RAISE = {
    "read": lambda vol, rel: vol.read(rel),
    "read_with_version": lambda vol, rel: vol.read_with_version(rel),
    # The comparand read raises; make_content never runs.
    "write_cas": lambda vol, rel: vol.write_cas(rel, lambda cur: cur + b"+mine"),
}


@pytest.mark.parametrize(
    ("cause", "how"),
    [("stale_view", "read"), ("watchdog", "read"), ("watchdog", "read_with_version"),
     ("watchdog", "write_cas")],
)
def test_a_read_that_raises_leaves_the_foreign_edit_baseline_where_it_was(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    cause: str, how: str,
) -> None:
    """The same rule for the other reads that raise without returning their
    bytes: under ``on_stale_read="raise"`` the strict deny surfaces as
    ``StaleView``; in strict mode a watchdog-degraded pre-read answer raises
    ``CoherenceError`` — from read, from read_with_version, and from
    write_cas's comparand read. None hands the caller the foreign bytes, so
    none may advance the baseline to them: a write() that ignores the raise is
    still denied as a foreign edit rather than clobbering it."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",),
        on_stale_read="raise" if cause == "stale_view" else "allow", config=fast_cfg,
    )
    try:
        assert vol.read(rel) == b"v1"
        target.write_bytes(b"FOREIGN")
        if cause == "watchdog":
            real_post = coherent_volume_module._coordinator_post

            def degraded_once(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
                if path == "/hooks/pre-read":
                    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)
                    return {"ok": True, "degraded": True}
                return real_post(endpoint, path, payload, **kwargs)

            monkeypatch.setattr(coherent_volume_module, "_coordinator_post", degraded_once)
        with pytest.raises(StaleView if cause == "stale_view" else CoherenceError):
            _READS_THAT_RAISE[how](vol, rel)
        assert vol._last_observed_hash[rel] == _sha(b"v1")

        with pytest.raises(StaleView) as denied:
            vol.write(rel, b"mine-derived-from-v1")

        assert str(denied.value) == coherent_volume_module._STALE_WRITE_DENY_REASON
        assert target.read_bytes() == b"FOREIGN"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("how", ["read", "read_with_version"])
def test_a_read_that_returns_still_seeds_the_baseline(
    tmp_path: Path, fast_cfg: LifecycleConfig, how: str
) -> None:
    """Control for the two tests above, in both directions: a read that
    RETURNS the bytes seeds the baseline with them. Advanced to the bytes a
    read returned after an out-of-band edit, the next write() replaces them;
    seeded by a read BEFORE an out-of-band edit, a write over it is denied."""
    def read(rel: str) -> bytes:
        return vol.read(rel) if how == "read" else vol.read_with_version(rel)[0]

    seen, unseen = _seed(tmp_path, "data/seen.txt"), _seed(tmp_path, "data/unseen.txt")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read("data/seen.txt")
        seen.write_bytes(b"v2-seen")
        if how == "read_with_version":
            # A strict read_with_version refuses bytes the coordinator never
            # recorded (a split pair), so it hands the caller nothing and the
            # baseline stays; the documented recovery is reacquire().
            with pytest.raises(StaleView):
                read("data/seen.txt")
            assert vol._last_observed_hash["data/seen.txt"] == _sha(b"v1")
            assert vol.reacquire("data/seen.txt") == b"v2-seen"
        else:
            assert read("data/seen.txt") == b"v2-seen"
        assert vol._last_observed_hash["data/seen.txt"] == _sha(b"v2-seen")
        vol.write("data/seen.txt", b"v3")
        assert seen.read_bytes() == b"v3"

        assert read("data/unseen.txt") == b"v1"
        assert vol._last_observed_hash["data/unseen.txt"] == _sha(b"v1")
        unseen.write_bytes(b"v2-unseen")
        with pytest.raises(StaleView):
            vol.write("data/unseen.txt", b"mine")
        assert unseen.read_bytes() == b"v2-unseen"
    finally:
        stop_coordinator(tmp_path)


# A CAS's comparand read hands its bytes on only when they are clean: write_cas
# gives them to make_content; write_cas_at (and a single-member publish, which
# takes its path) never gives them to anyone — the caller supplies the content.
_CAS_AFTER_A_READ = {
    "write_cas": lambda vol, rel, version: vol.write_cas(rel, lambda cur: cur + b"+cas"),
    "write_cas_at": lambda vol, rel, version: vol.write_cas_at(rel, version, b"cas"),
    "atomic_publish": lambda vol, rel, version: vol.atomic_publish([(rel, version, b"cas")]),
}


@pytest.mark.parametrize("cas", [*_CAS_AFTER_A_READ, "none"])
def test_a_cas_whose_comparand_read_is_denied_leaves_the_foreign_edit_baseline_where_it_was(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, cas: str,
) -> None:
    """The volume writes v1 and reads it back; the file is then rewritten out
    of band. A CAS's comparand read hash-checks the disk against the
    coordinator's record, so it is strict-DENIED and the CAS fails closed
    (``ViewWedged``) without handing anyone the foreign bytes — make_content
    never runs. So the baseline stays at v1 and the next write() is denied as
    a foreign edit, not admitted over it. Before, the denied read seeded the
    baseline with the foreign bytes and that write() clobbered the edit.
    ``none`` is the control: with no CAS the same write is denied."""
    monkeypatch.setattr(coherent_volume_module, "DENIED_READ_BACKOFF_CAP_SEC", 0.002)
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v0")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(rel)
        vol.write(rel, b"v1")  # the coordinator now records v1's hash
        _data, version = vol.read_with_version(rel)
        target.write_bytes(b"FOREIGN")
        if cas != "none":
            with pytest.raises(ViewWedged):
                _CAS_AFTER_A_READ[cas](vol, rel, version)
        assert vol._last_observed_hash[rel] == _sha(b"v1"), "a denied read moved the baseline"

        with pytest.raises(StaleView) as denied:
            vol.write(rel, b"mine-derived-from-v1")

        assert str(denied.value) == coherent_volume_module._STALE_WRITE_DENY_REASON
        assert target.read_bytes() == b"FOREIGN", "the out-of-band edit was clobbered"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_seeds_the_baseline_with_the_bytes_it_hands_make_content(
    tmp_path: Path, fast_cfg: LifecycleConfig,
) -> None:
    """The other direction: a CLEAN comparand read's bytes reach make_content,
    so the caller has seen them and they become the baseline — even when
    make_content then gives up. Here a peer commits v2 after the volume read
    v1; write_cas's first read is denied (this instance is INVALID), the
    re-minted read is clean and hands v2 to make_content, which raises. A
    write() derived from v2 then replaces v2 rather than being denied as a
    foreign edit."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(rel)
        peer.read(rel)
        peer.write(rel, b"v2")
        handed: list[bytes] = []

        def give_up(current: bytes) -> bytes:
            handed.append(current)
            raise LookupError("the caller gave up")

        with pytest.raises(LookupError):
            vol.write_cas(rel, give_up)

        assert handed == [b"v2"], "control: make_content saw exactly the clean read"
        assert vol._last_observed_hash[rel] == _sha(b"v2")
        vol.write(rel, b"mine-derived-from-v2")
        assert target.read_bytes() == b"mine-derived-from-v2"
    finally:
        stop_coordinator(tmp_path)


#: A pre-read answer a strict volume cannot take as registered, so the read
#: raises without handing its bytes on: the coordinator's watchdog timed out,
#: the request never reached it (``None``: the transport raises), or the
#: coordinator answered with its failure envelope.
_UNCONFIRMED_PRE_READ_ANSWERS: dict[str, dict[str, object] | None] = {
    "watchdog": {"ok": True, "degraded": True},
    "transport": None,
    "envelope": {"ok": False, "reason": _INTERNAL_OPERATIONAL_ERROR},
}


def _answer_the_second_pre_read(
    monkeypatch: pytest.MonkeyPatch, answer: dict[str, object] | None, answers: list[object]
) -> None:
    """The next pre-read goes to the coordinator; the one after it gets
    ``answer`` instead (``None``: the transport raises); every other request
    goes to the coordinator. Records what each pre-read got back."""
    real_post = coherent_volume_module._coordinator_post

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path == "/hooks/pre-read" and len(answers) == 1:
            answers.append(answer)
            if answer is None:
                raise CoordinatorUnavailable("connection refused")
            return answer
        got = real_post(endpoint, path, payload, **kwargs)
        if path == "/hooks/pre-read":
            answers.append(got)
        return got

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", post)


@pytest.mark.parametrize("retry_answer", sorted(_UNCONFIRMED_PRE_READ_ANSWERS))
def test_a_cas_whose_retry_read_fails_after_a_denied_read_still_guards_the_next_write(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    retry_answer: str,
) -> None:
    """The volume reads v1 and a peer commits v2, so this instance is
    INVALID. write_cas's first comparand read is DENIED for that alone — the
    disk holds exactly the bytes the coordinator records — so the CAS
    re-mints and reads again, and that read fails closed: the coordinator's
    watchdog timed out, the request never reached it, or the coordinator
    answered with its failure envelope. write_cas raises and make_content
    never runs, so the caller was never handed v2. A write() of content
    derived from v1 then goes out under the re-minted identity, which is not
    INVALID, so the coordinator admits it; only the foreign-edit baseline
    stands between it and the peer's commit. It must be denied, and the disk
    must keep v2.

    Prevents a denied comparand read moving the baseline whenever the
    coordinator did not dispute its bytes: here that set the baseline to v2,
    and the write() landed over the peer's commit without an error — a lost
    update."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        assert vol.read(rel) == b"v1"
        peer.read(rel)
        peer.write(rel, b"v2")
        answers: list[object] = []
        _answer_the_second_pre_read(
            monkeypatch, _UNCONFIRMED_PRE_READ_ANSWERS[retry_answer], answers
        )
        handed: list[bytes] = []

        def derive(current: bytes) -> bytes:
            handed.append(current)
            return current + b"+cas"

        with pytest.raises(CoherenceError) as failed:
            vol.write_cas(rel, derive)

        assert type(failed.value) is CoherenceError, "not the fail-closed read: " + repr(failed.value)
        first = answers[0]
        assert isinstance(first, dict), first
        assert first["hookSpecificOutput"]["permissionDecision"] == "deny", (
            "control: the first comparand read was admitted"
        )
        assert first["summary"]["hash_differs"] is False, (
            "control: the deny disputed the bytes, not this instance's standing"
        )
        assert len(answers) == 2, "control: the CAS read again after its read failed"
        assert handed == [], "control: make_content was handed bytes"
        assert vol._last_observed_hash[rel] == _sha(b"v1"), "the denied read moved the baseline"

        with pytest.raises(StaleView) as denied:
            vol.write(rel, b"mine-derived-from-v1")

        assert str(denied.value) == coherent_volume_module._STALE_WRITE_DENY_REASON
        assert target.read_bytes() == b"v2", "the peer's commit was overwritten"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("cas", ["write_cas_at", "atomic_publish"])
def test_a_cas_that_never_hands_its_comparand_bytes_on_never_seeds_the_baseline(
    tmp_path: Path, fast_cfg: LifecycleConfig, cas: str,
) -> None:
    """write_cas_at's comparand read is clean but its bytes reach no one: the
    caller supplies the content, merged against the version IT read. When a
    peer's commit makes that version stale the CAS is refused
    (``CasVersionConflict``), and the baseline must stay at what the caller
    read — v1 — so a write() of content merged against v1 is denied rather
    than replacing the peer's v2. Before, the unseen read seeded the baseline
    with v2 and re-registered the instance, so that write() was admitted: a
    lost update of the peer's commit."""
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        _data, version = vol.read_with_version(rel)
        peer.read(rel)
        peer.write(rel, b"v2")
        with pytest.raises(CasVersionConflict):
            _CAS_AFTER_A_READ[cas](vol, rel, version)
        assert vol._last_observed_hash[rel] == _sha(b"v1"), "an unseen read moved the baseline"

        with pytest.raises(StaleView):
            vol.write(rel, b"mine-merged-against-v1")

        assert target.read_bytes() == b"v2", "the peer's commit was overwritten"
    finally:
        stop_coordinator(tmp_path)


def _cas_at_loses_on_version(
    vol: CoherentVolume, rel: str, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean comparand read, then a stale ``expected_version``."""
    with pytest.raises(CasVersionConflict):
        vol.write_cas_at(rel, 0, b"mine")


def _cas_at_denied(
    vol: CoherentVolume, rel: str, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The disk no longer matches the coordinator's record: the read is denied."""
    target.write_bytes(b"FIRST-FOREIGN")
    with pytest.raises(ViewWedged):
        vol.write_cas_at(rel, 1, b"mine")


def _cas_denied(
    vol: CoherentVolume, rel: str, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every comparand read denied, so make_content never runs."""
    target.write_bytes(b"FIRST-FOREIGN")
    with pytest.raises(ViewWedged):
        vol.write_cas(rel, lambda current: current + b"+mine")


def _cas_at_read_raises(
    vol: CoherentVolume, rel: str, target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The comparand read's request fails (a watchdog-degraded answer in
    strict mode), so the read raises before any answer is used: the first
    observation is recorded before the request, as for a refused read()."""
    real_post = coherent_volume_module._coordinator_post

    def degraded_once(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path == "/hooks/pre-read":
            monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)
            return {"ok": True, "degraded": True}
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", degraded_once)
    with pytest.raises(CoherenceError):
        vol.write_cas_at(rel, 1, b"mine")


_FAILED_CAS_ON_AN_UNREAD_PATH = {
    "write_cas_at-version": _cas_at_loses_on_version,
    "write_cas_at-denied": _cas_at_denied,
    "write_cas-denied": _cas_denied,
    "write_cas_at-read-raises": _cas_at_read_raises,
}


@pytest.mark.parametrize("first_step", ["none", *_FAILED_CAS_ON_AN_UNREAD_PATH])
def test_a_failed_cas_on_a_path_never_read_still_guards_the_next_write(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    first_step: str,
) -> None:
    """A CAS's comparand read never MOVES a baseline the caller already has
    (the three tests above). On a path this volume has never observed there
    is no baseline to move, so the read is the volume's first observation and
    records one, the way a refused first read() does. Without it, a CAS that
    fails and an out-of-band edit that lands after it leave write() nothing to
    compare against, and write() overwrites the edit. ``none`` is the control:
    a volume that never looked at the path has no baseline, so the same
    write() is admitted; the guard comes from the CAS's read and nothing
    else."""
    monkeypatch.setattr(coherent_volume_module, "DENIED_READ_BACKOFF_CAP_SEC", 0.002)
    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v0")
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        peer.read(rel)
        peer.write(rel, b"v1")  # the coordinator now records v1 at version >= 1
        if first_step != "none":
            _FAILED_CAS_ON_AN_UNREAD_PATH[first_step](vol, rel, target, monkeypatch)
            assert rel in vol._last_observed_hash, "the failed CAS left no baseline"
        target.write_bytes(b"LATER-FOREIGN")

        if first_step == "none":
            vol.write(rel, b"mine")
            assert target.read_bytes() == b"mine", "control: no baseline, no guard"
            return
        with pytest.raises(StaleView) as denied:
            vol.write(rel, b"mine")

        assert str(denied.value) == coherent_volume_module._STALE_WRITE_DENY_REASON
        assert target.read_bytes() == b"LATER-FOREIGN", "the out-of-band edit was clobbered"
    finally:
        stop_coordinator(tmp_path)


# --- a commit refused after the bytes landed: what the error says -----------------
#
# FROZEN duplicates of the text write() raises when its commit is refused for
# the caller principal after the bytes reached disk — never built from the
# constants under test. Each clause is a claim about state (the disk, the
# coordinator's record of this write's commit, the peers, the grant) and a
# test below observes that state — on a path the coordinator tracks and on
# one it does not, whose admitted grant requests get the same answer; for a
# write whose bytes were already on disk and for ones the volume rewrote; and
# where the version moved for a reason other than this write's commit.

_WROTE = "This write put its bytes on disk at {rel}. "
_ALREADY_HELD = (
    "{rel} already held this write's bytes on disk, so this write left the file "
    "as it was. "
)
_UNRECORDED_WRITE = "The coordinator did not record this write's commit. {grant} "
_GRANT_TAKEN = (
    "If the coordinator tracked the path when it admitted this write's grant "
    "request, that request invalidated every peer it had recorded as holding a "
    "copy and took a grant this volume has not released; if it did not track "
    "the path, the request took no grant and invalidated no peer."
)
_GRANT_UNCONFIRMED = (
    "The answer to this write's grant request was not confirmed, so whether it "
    "invalidated any peer, and whether it took a grant this volume now holds, "
    "is not known."
)


def _coordinator_state(peer: CoherentVolume, rel: str) -> str | None:
    """The coordinator's own record of ``peer``'s current incarnation for
    ``rel``, read from ``/status`` — which changes nothing, and lists no
    INVALID row, so an invalidated holder reads ``None``. Read from the
    coordinator rather than through a read of the file, whose hash check
    would also deny a live copy once the disk moved."""
    status = coherent_volume_module._coordinator_get(peer._endpoint, "/status")
    agent = str(session_to_agent_id(peer.session_id, peer._incarnation))
    for session in status.get("sessions", []):
        if session.get("agent_id") == agent:
            return session.get("states", {}).get(rel)
    return None


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_commit_refused_for_its_principal_after_the_bytes_landed_says_they_did(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture, on_error: str,
) -> None:
    """write() holds the grant and has put the bytes on disk when its commit
    (post-edit) is refused for the caller principal, and recovery cannot cure
    it. The refusal itself changed nothing at the coordinator — the version
    did not advance — but the write had changed two things before it: the
    file, and (the path is tracked) every peer that held a copy, which the
    grant request invalidated (KTD-1). So the typed refusal says both, and
    that the grant was not released: it must not read like a request that
    had no effect, nor tell a peer it may keep its copy.

    The peer's state is observed, not assumed: a verification read finds it
    holding a live copy before the write and invalidated after; its own
    write is then refused as revoked. The grant stays recorded for the next
    re-mint to release; no principal or nonce reaches the message."""
    import traceback

    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    caplog.set_level(logging.DEBUG)
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(rel)
        peer.read(rel)
        assert _coordinator_state(peer, rel) == "SHARED", "control: the peer holds a live copy"
        bound, nonce = vol._principal, vol._mint_nonce
        real_post = coherent_volume_module._coordinator_post

        def lose_principal_after_the_grant(
            endpoint: object, path: str, payload: dict, **kwargs: object
        ) -> object:
            answer = real_post(endpoint, path, payload, **kwargs)
            if path == "/hooks/pre-edit" and payload.get("session_id") == vol.session_id:
                vol._mint_nonce, vol._principal = "Z" * 43, "X" * 43
            return answer

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", lose_principal_after_the_grant)
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            with pytest.raises(CallerPrincipalRefused) as raised:
                vol.write(rel, b"v2")

        message = str(raised.value)
        assert raised.value.reason == "caller_principal_foreign"
        assert target.read_bytes() == b"v2"
        assert message.startswith((_WROTE + _UNRECORDED_WRITE).format(rel=rel, grant=_GRANT_TAKEN))
        assert "(caller_principal_foreign)" in message
        assert vol._incarnation in vol._grant_incarnations, "the grant stays recorded"
        assert _coordinator_state(vol, rel) == "EXCLUSIVE", "and was not released"
        _data, version, _generation = peer.read_with_version_generation(rel, observe=False)
        assert version == 1, "the coordinator did not record the commit"
        assert _coordinator_state(peer, rel) is None, "the peer was invalidated, as the message says"
        # The pre-edit deny fires only for an INVALID editor, so this is the
        # invalidation itself, not the moved disk.
        with pytest.raises(StaleView, match="revoked"):
            peer.write(rel, b"peer")
        rendered = "".join(traceback.format_exception(raised.value))
        _assert_no_secret_in(
            rendered + caplog.text + " ".join(str(w.message) for w in warned),
            bound, nonce, "Z" * 43, "X" * 43,
        )
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("answer", ["degraded", "lost"])
def test_a_commit_refused_after_an_unconfirmed_grant_request_does_not_say_peers_were_invalidated(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, answer: str,
) -> None:
    """Degrade mode: the answer to write()'s grant request is watchdog-degraded
    or lost, so the write goes ahead best-effort without knowing whether the
    coordinator took the grant — and so whether it invalidated anyone. Here
    the request never reached it: the peer still holds a live copy. When the
    commit is then refused for the principal, the message says the grant and
    the peers' state are not known; saying peers were invalidated would be
    false here, and saying they were not would be false when the request did
    land."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
    peer = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.read(rel)
        peer.read(rel)
        bound, nonce = vol._principal, vol._mint_nonce
        real_post = coherent_volume_module._coordinator_post

        def unanswered_grant_request(
            endpoint: object, path: str, payload: dict, **kwargs: object
        ) -> object:
            if path == "/hooks/pre-edit" and payload.get("session_id") == vol.session_id:
                vol._mint_nonce, vol._principal = "Z" * 43, "X" * 43
                if answer == "degraded":
                    return {"ok": True, "degraded": True}
                raise CoordinatorUnavailable("simulated: the answer was lost")
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", unanswered_grant_request)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(CallerPrincipalRefused) as raised:
                vol.write(rel, b"v2")

        message = str(raised.value)
        assert target.read_bytes() == b"v2"
        assert message.startswith(
            (_WROTE + _UNRECORDED_WRITE).format(rel=rel, grant=_GRANT_UNCONFIRMED)
        )
        assert _coordinator_state(peer, rel) == "SHARED", "the grant request never reached the coordinator"
        _assert_no_secret_in(message, bound, nonce, "Z" * 43, "X" * 43)
    finally:
        stop_coordinator(tmp_path)


def _tracked_version(vol: CoherentVolume, rel: str) -> int | None:
    """The version the coordinator keeps for ``rel``, read from ``/status``
    (which changes nothing) — ``None`` when it keeps no record of the path."""
    status = coherent_volume_module._coordinator_get(vol._endpoint, "/status")
    for artifact in status.get("tracked_artifacts", []):
        if artifact.get("path") == rel:
            return artifact.get("version")
    return None


# Where the refused write goes: a path the coordinator tracks, one it does not
# track, and one the VOLUME manages but the coordinator is told to ignore AFTER
# the attach (the attach check sees the policy the coordinator started with; the
# untrack command reloads it) — so nothing the volume holds says whether the
# coordinator tracks a path.
_REFUSED_WRITE_PATHS = {
    "tracked": ("data/shared.txt", None),
    "untracked": ("notes/free.txt", None),
    "managed-but-ignored": ("data/shared.txt", "data/**"),
}

# What the refused write puts, against what the file and the volume already
# hold: (the bytes it writes, the disk writes it makes). The volume skips the
# disk only when the file holds bytes it COMMITTED itself (``same-bytes``). It
# rewrites them after a READ of the same bytes, which commits nothing
# (``read-then-same-bytes``), and when its committed bytes go back over a
# foreign edit under ``on_stale_write='allow'`` (``committed-over-foreign``) —
# neither the disk nor the volume's record alone says which, so the disk
# clause is checked against the writes counted in every arm.
_REFUSED_WRITE_BYTES = {
    "new-bytes": (b"v2", 1),
    "same-bytes": (b"v2", 0),
    "read-then-same-bytes": (b"v1", 1),
    "committed-over-foreign": (b"v2", 1),
}


@pytest.mark.parametrize("rewrite", list(_REFUSED_WRITE_BYTES))
@pytest.mark.parametrize("where", list(_REFUSED_WRITE_PATHS))
def test_every_clause_of_an_unrecorded_write_holds_wherever_the_write_went(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    where: str, rewrite: str,
) -> None:
    """The grant request's admitted answer is the same bare ``{"ok": true}``
    whether the coordinator tracks the path — it took EXCLUSIVE and
    invalidated every peer holding a copy — or does not, when it took no
    grant and invalidated nobody. The volume cannot tell which (its managed
    globs do not decide it: an ignored path is managed here and untracked
    there), so the message states both cases, and each is observed where it
    applies. The disk clause says whether THIS call wrote the file, and each
    arm counts the disk writes it made (see ``_REFUSED_WRITE_BYTES``).

    Every clause is checked against state: the disk (and whether this call
    wrote it), the coordinator's record of the commit, the grant, and the
    peer — which, on a tracked path, is invalidated and refused as revoked,
    and on an untracked one, held no copy the coordinator knew of and writes
    freely."""
    from ccs.core.exceptions import CallerPrincipalRefused

    rel, ignored = _REFUSED_WRITE_PATHS[where]
    data, expected_disk_writes = _REFUSED_WRITE_BYTES[rewrite]
    target = _seed(tmp_path, rel=rel, content=b"v1")
    on_stale_write = "allow" if rewrite == "committed-over-foreign" else "raise"
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_stale_write=on_stale_write, config=fast_cfg)
    peer = CoherentVolume(tmp_path, managed=("data/**",), on_stale_write="allow", config=fast_cfg)
    try:
        if ignored is not None:
            # Ignored after the attach: the coordinator reloads its policy and
            # untracks the managed path, and neither volume sees that.
            untracked = coherent_volume_module._coordinator_post(
                vol._endpoint, "/policy/untrack", {"paths": [ignored]}
            )
            assert untracked.get("ok") is True and untracked.get("removed") == [ignored], untracked
        vol.read(rel)
        if rewrite in ("same-bytes", "committed-over-foreign"):
            vol.write(rel, b"v2")  # recorded: the refused write below writes these bytes again
        peer.read(rel)
        if rewrite == "committed-over-foreign":
            target.write_bytes(b"foreign")  # out of band: the file no longer holds them
        version_before = _tracked_version(vol, rel)
        peer_before = _coordinator_state(peer, rel)
        disk_writes: list[Path] = []
        real_atomic_write = vol._atomic_write

        def counted_atomic_write(abs_path: Path, data: bytes) -> None:
            disk_writes.append(abs_path)
            real_atomic_write(abs_path, data)

        monkeypatch.setattr(vol, "_atomic_write", counted_atomic_write)
        real_post = coherent_volume_module._coordinator_post

        def lose_principal_after_the_grant(
            endpoint: object, path: str, payload: dict, **kwargs: object
        ) -> object:
            answer = real_post(endpoint, path, payload, **kwargs)
            if path == "/hooks/pre-edit" and payload.get("session_id") == vol.session_id:
                assert answer == {"ok": True}, "control: the admitted answer names neither case"
                vol._mint_nonce, vol._principal = "Z" * 43, "X" * 43
            return answer

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", lose_principal_after_the_grant)
        with pytest.raises(CallerPrincipalRefused) as raised:
            vol.write(rel, data)
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)

        disk = _WROTE if expected_disk_writes else _ALREADY_HELD
        assert str(raised.value).startswith(
            (disk + _UNRECORDED_WRITE).format(rel=rel, grant=_GRANT_TAKEN)
        )
        assert target.read_bytes() == data
        assert len(disk_writes) == expected_disk_writes, "the disk clause"
        assert _tracked_version(vol, rel) == version_before, "the commit was not recorded"
        if where == "tracked":
            assert version_before is not None, "control: the coordinator tracks the path"
            assert peer_before == "SHARED", "control: the peer held a copy"
            assert _coordinator_state(vol, rel) == "EXCLUSIVE", "a grant, not released"
            assert _coordinator_state(peer, rel) is None, "the peer was invalidated"
            with pytest.raises(StaleView, match="revoked"):
                peer.write(rel, b"peer")
            assert target.read_bytes() == data
        else:
            assert version_before is None, "control: the coordinator keeps no record of it"
            assert peer_before is None, "so the peer held no copy it knew of"
            assert _coordinator_state(vol, rel) is None, "no grant was taken"
            peer.write(rel, b"peer")  # nobody was invalidated: the peer's write is admitted
            assert target.read_bytes() == b"peer"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("moved_by", ["the-grant-request", "a-peer-commit"])
def test_the_commit_clause_holds_when_the_version_moved_for_another_reason(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, moved_by: str,
) -> None:
    """The message says the coordinator did not record this write's commit —
    not that no version moved, which is false in two cases. The first write
    to a tracked path the coordinator has never seen registers it through
    the grant request (no version, then 1); and a peer can commit between
    this write's grant and its refused commit (1, then 2). The commit clause
    is observed in each: the version is what it would be WITHOUT this
    write's commit — 1, where a recorded first write leaves 2 (control, on a
    second path), and one past the read, the peer's commit alone."""
    from ccs.core.exceptions import CallerPrincipalRefused

    fresh = moved_by == "the-grant-request"
    rel = "data/fresh.txt" if fresh else "data/shared.txt"
    target = tmp_path / rel if fresh else _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    peer = CoherentVolume(tmp_path, managed=("data/**",), on_stale_write="allow", config=fast_cfg)
    try:
        if not fresh:
            vol.read(rel)
        version_before = _tracked_version(vol, rel)
        real_post = coherent_volume_module._coordinator_post
        peer_commits: list[int | None] = []

        def refuse_the_commit(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            mine = payload.get("session_id") == vol.session_id
            if mine and path == "/hooks/post-edit" and not fresh and not peer_commits:
                peer.reacquire(rel)  # inside the window: after this write's grant, before its commit
                peer.write(rel, b"peer")
                peer_commits.append(_tracked_version(peer, rel))
            answer = real_post(endpoint, path, payload, **kwargs)
            if mine and path == "/hooks/pre-edit":
                vol._mint_nonce, vol._principal = "Z" * 43, "X" * 43
            return answer

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", refuse_the_commit)
        with pytest.raises(CallerPrincipalRefused) as raised:
            vol.write(rel, b"v2")
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)

        assert str(raised.value).startswith(
            (_WROTE + _UNRECORDED_WRITE).format(rel=rel, grant=_GRANT_TAKEN)
        )
        if fresh:
            assert version_before is None, "control: the coordinator had never seen the path"
            assert _tracked_version(vol, rel) == 1, "only the grant request registered it"
            assert target.read_bytes() == b"v2"
            peer.write("data/control.txt", b"v2")
            assert _tracked_version(peer, "data/control.txt") == 2, "control: a recorded first write"
        else:
            assert version_before == 1 and peer_commits == [2], "control: the peer committed in the window"
            assert _tracked_version(vol, rel) == 2, "only the peer's commit advanced it"
            assert target.read_bytes() == b"peer"
    finally:
        stop_coordinator(tmp_path)


# --- a commit that fails after the bytes landed: every failure, one path -------


class _AnsweringCoordinator(http.server.BaseHTTPRequestHandler):
    """Answers every POST with ``status`` — a redirect naming a ``Location``,
    or a server error — and records the routes it was sent."""

    status = 302
    seen: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        type(self).seen.append(self.path)
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(type(self).status)
        self.send_header("Location", "http://127.0.0.1:1/elsewhere")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


# FROZEN duplicates of what a failed commit reports, by the status answering it.
_FAILED_COMMIT = "coordinator request to /hooks/post-edit failed: HTTP {status}"
_REDIRECTED = ", a redirect, which this client never follows"


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize("status", [503, 302, 307, 308])
def test_a_commit_that_fails_after_the_bytes_landed_takes_one_path_whatever_its_status(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    status: int, on_error: str,
) -> None:
    """write()'s commit (post-edit) fails after its bytes reached disk. A
    failed commit takes ``on_error``'s path — strict raises a plain
    ``CoherenceError`` naming the request and its status; degrade warns once
    and the write stands — and a redirect is one more failed commit:
    refused, never followed, and reported by its status alone. Before, a
    redirect escaped as ``RedirectRefused`` in both modes: the trust refusal
    of a request that changed nothing, after the write had changed the file.
    The grant stays recorded, as for any failed commit."""
    import traceback

    rel = "data/shared.txt"
    target = _seed(tmp_path, content=b"v1")
    _AnsweringCoordinator.status, _AnsweringCoordinator.seen = status, []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _AnsweringCoordinator)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    try:
        vol.read(rel)
        real_post = coherent_volume_module._coordinator_post
        answering = CoordinatorEndpoint(port=httpd.server_address[1], bearer=vol._endpoint.bearer)

        def fail_the_commit(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            commit = path == "/hooks/post-edit" and payload.get("success") is True
            return real_post(answering if commit else endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", fail_the_commit)
        expected = _FAILED_COMMIT.format(status=status) + (_REDIRECTED if status < 400 else "")
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            if on_error == "strict":
                with pytest.raises(CoherenceError) as raised:
                    vol.write(rel, b"v2")
                assert type(raised.value) is CoherenceError, type(raised.value)
                assert str(raised.value) == expected
                reported = "".join(traceback.format_exception(raised.value))
            else:
                vol.write(rel, b"v2")
                assert vol.degradation_count == 1
                degraded = [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
                reported = " ".join(str(w.message) for w in degraded)
                assert reported == f"CoherentVolume degraded: {expected}"

        assert _AnsweringCoordinator.seen == ["/hooks/post-edit"], "control: the commit was answered"
        assert target.read_bytes() == b"v2", "control: the bytes had landed"
        assert vol._incarnation in vol._grant_incarnations, "the grant stays recorded"
        # The one path, down to the commit record the 5xx arm (the control)
        # leaves: strict raised before it; degrade wrote best-effort past it.
        committed = _sha(b"v2") if on_error == "degrade" else None
        assert vol._last_committed_hash.get(rel) == committed, "the commit record differs from a 5xx's"
        assert "elsewhere" not in reported, "the Location reached the report"
    finally:
        httpd.shutdown()
        httpd.server_close()
        stop_coordinator(tmp_path)


# --- a redirect echoing a nonce this session sent earlier -----------------------


class _EchoingRedirector(http.server.BaseHTTPRequestHandler):
    """Redirects every POST to a ``Location`` naming every mint nonce any
    request has sent it so far — a redirector that echoes what it was sent,
    into answers to requests that carry nothing of the kind."""

    nonces: list[str] = []
    locations: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        cls = type(self)
        if isinstance(body.get("mint_nonce"), str):
            cls.nonces.append(body["mint_nonce"])
        cls.locations.append("http://127.0.0.1:1/" + "-".join(cls.nonces))
        self.send_response(302)
        self.send_header("Location", cls.locations[-1])
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


def test_a_redirect_echoing_the_nonce_a_claim_sent_names_no_location(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Degrade mode: the volume's claim is redirected (unconfirmed — it
    attaches without a principal), and so is its read, whose ``Location``
    echoes the mint nonce the claims sent. The read carries no principal and
    no nonce of its own, but what answers it may have been sent one before:
    the refusal names the status only, so neither the error, its chain nor a
    warning carries the nonce. Before, a request carrying no principal
    material quoted its ``Location``, and with it the echoed nonce."""
    import traceback

    from ccs.core.exceptions import RedirectRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    _EchoingRedirector.nonces, _EchoingRedirector.locations = [], []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _EchoingRedirector)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    real_claim = coherent_volume_module.claim_caller_principal
    real_post = coherent_volume_module._coordinator_post

    def redirector(endpoint: CoordinatorEndpoint) -> CoordinatorEndpoint:
        return CoordinatorEndpoint(port=httpd.server_address[1], bearer=endpoint.bearer)

    def claim_redirected(endpoint: CoordinatorEndpoint, session_id: str, nonce: str) -> object:
        return real_claim(redirector(endpoint), session_id, nonce)

    def read_redirected(endpoint: CoordinatorEndpoint, path: str, payload: dict, **kwargs: object) -> object:
        return real_post(redirector(endpoint) if path == "/hooks/pre-read" else endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "claim_caller_principal", claim_redirected)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", read_redirected)
    try:
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always")
            vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
            with pytest.raises(RedirectRefused) as raised:
                vol.read(rel)

        nonces = list(_EchoingRedirector.nonces)
        assert nonces and vol._principal is None, "control: the claims were redirected"
        assert all(n in _EchoingRedirector.locations[-1] for n in nonces), "control: the echo was sent"
        assert raised.value.status == 302
        reported = "".join(traceback.format_exception(raised.value))
        reported += " ".join(str(w.message) for w in warned) + str(raised.value.location)
        _assert_no_secret_in(reported, *nonces)
        assert "127.0.0.1:1" not in reported, "the Location reached the error"
    finally:
        httpd.shutdown()
        httpd.server_close()
        stop_coordinator(tmp_path)


# --- the reason a second refusal carries; what rides an exception's chain -----


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_request_refused_again_reports_the_second_refusals_reason(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, on_error: str
) -> None:
    """Refused as ``caller_principal_absent``, recovered, and refused AGAIN
    as ``caller_principal_foreign``: the typed refusal carries the reason of
    the refusal it reports — the retry's — not the first one's. (Two equal
    reasons cannot tell which is carried.)"""
    from ccs.cli._coherence_client import PRINCIPAL_REFUSED_AGAIN, PrincipalClaim
    from ccs.core.exceptions import CallerPrincipalRefused

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg  # type: ignore[arg-type]
    )
    try:
        monkeypatch.setattr(
            coherent_volume_module, "claim_caller_principal",
            lambda *_a: PrincipalClaim("bound", principal="Q" * 43),
        )
        reasons = ["caller_principal_absent", "caller_principal_foreign"]
        sent: list[str] = []
        _answer_route(
            monkeypatch, "/hooks/pre-edit", sent,
            lambda n: _http_error(
                "/hooks/pre-edit", 400, {"error": "refused", "reason": reasons[n % 2]}
            ),
        )

        with pytest.raises(CallerPrincipalRefused) as raised:
            vol.write(rel, b"v2")

        assert sent.count("/hooks/pre-edit") == 2
        assert raised.value.reason == "caller_principal_foreign"
        assert "(caller_principal_foreign)" in str(raised.value)
        assert PRINCIPAL_REFUSED_AGAIN in str(raised.value)
    finally:
        stop_coordinator(tmp_path)


def _rendered(exc: BaseException) -> str:
    """What a traceback or ``logger.exception`` prints for ``exc``: its whole
    chain, as Python renders it."""
    import traceback

    return "".join(traceback.format_exception(exc))


def test_no_coordinator_supplied_text_rides_the_chain_of_what_a_volume_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strict mode: the read is refused, recovery adopts a new principal, and
    the retry is answered 500 with the held principal, the nonce and the
    adopted principal as the status line's reason phrase. The raised error
    names the status only — and so does its CHAIN: the ``HTTPError`` carrying
    the phrase is not on it, so a traceback or ``logger.exception`` prints no
    principal or nonce either."""
    from ccs.cli import _coherence_client

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        adopted = "Q" * 43
        held = (vol._principal, vol._mint_nonce, adopted)
        echo = " ".join(held)  # type: ignore[arg-type]
        real_post = _coherence_client.post

        def claim_binds(endpoint: object, path: str, body: dict, **kwargs: object) -> object:
            if path == "/principal/claim":
                return {"ok": True, "principal": adopted}
            return real_post(endpoint, path, body, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(_coherence_client, "post", claim_binds)
        answers = [
            _http_error("/hooks/pre-read", 400, {"reason": "caller_principal_foreign"}, phrase=echo),
            _http_error("/hooks/pre-read", 500, {"error": echo}, phrase=echo),
        ]
        sent: list[str] = []
        _answer_route(monkeypatch, "/hooks/pre-read", sent, lambda n: answers[n])

        with pytest.raises(CoherenceError) as raised:
            vol.read(rel)

        rendered = _rendered(raised.value)
        assert "HTTP 500" in rendered, "control: the retried request was the one answered 500"
        _assert_no_secret_in(rendered, *held)
    finally:
        stop_coordinator(tmp_path)


def _malformed_answers(
    monkeypatch: pytest.MonkeyPatch, route: str, error: Callable[[], Exception]
) -> list[str]:
    """Every request to ``route`` meets ``error()`` where the transport would
    read the answer — a malformed HTTP answer the client's ``_execute`` must
    classify; every other request goes to the coordinator. Returns the list
    of routes sent."""
    from ccs.cli import _coherence_client

    real_build = _coherence_client._build_opener
    # Plain-http openers are cached process-wide. An empty cache for this test
    # makes the next request build its opener through the patched seam, and the
    # wrapped opener goes with the test instead of staying in the cache.
    monkeypatch.setattr(_coherence_client, "_shared_openers", {})
    sent: list[str] = []

    class _Opener:
        def __init__(self, real: object) -> None:
            self._real = real

        def open(self, req: object, timeout: float | None = None) -> object:
            sent.append(req.selector)  # type: ignore[attr-defined]
            if req.selector == route:  # type: ignore[attr-defined]
                raise error()
            return self._real.open(req, timeout=timeout)  # type: ignore[attr-defined]

    monkeypatch.setattr(_coherence_client, "_build_opener", lambda ctx: _Opener(real_build(ctx)))
    return sent


_ECHOED_NONCE = "N" * 43


def _malformed(kind: str) -> Exception:
    import http.client

    if kind == "bad_status_line":
        return http.client.BadStatusLine(f"XHTTP/1.1 200 {_ECHOED_NONCE}\r\n")
    return http.client.IncompleteRead(f'{{"principal": "{_ECHOED_NONCE}'.encode(), 40)


@pytest.mark.parametrize("kind", ["bad_status_line", "incomplete_read"])
def test_a_malformed_claim_answer_is_unconfirmed_never_an_untyped_escape(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, kind: str,
) -> None:
    """A claim answered with a status line that is not HTTP, or with a body
    cut short, is a transport failure: the claim is ``unconfirmed`` (claimed
    again with the SAME nonce before the next request), degrade mode warns
    and constructs, strict mode raises ``CoherenceError`` — never the raw
    ``http.client`` exception, whose text is the line the coordinator sent.
    Nothing the coordinator sent reaches the message or its chain."""
    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    sent = _malformed_answers(monkeypatch, "/principal/claim", lambda: _malformed(kind))
    try:
        with pytest.raises(CoherenceError) as strict:
            CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
        with pytest.warns(CoherenceDegradedWarning) as warned:
            vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)

        assert vol._claim_outcome == "unconfirmed" and vol._principal is None
        assert vol.read(rel) == b"v1"
        assert sent.count("/principal/claim") == 3, "claimed again before the read"
        text = _rendered(strict.value) + " ".join(str(w.message) for w in warned)
        assert "malformed" in text
        assert _ECHOED_NONCE not in text
    finally:
        stop_coordinator(tmp_path)


def test_no_malformed_answer_text_rides_the_chain_of_a_strict_request_failure(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-read answered with a non-HTTP status line that echoes the
    principal the request presented: strict raises ``CoherenceError`` naming
    the malformed answer by its TYPE, and neither the message nor the chain
    carries the line."""
    import http.client

    rel = "data/shared.txt"
    _seed(tmp_path, content=b"v1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        principal = vol._principal
        assert principal
        _malformed_answers(
            monkeypatch, "/hooks/pre-read",
            lambda: http.client.BadStatusLine(f"XHTTP/1.1 200 {principal}\r\n"),
        )
        with pytest.raises(CoherenceError) as raised:
            vol.read(rel)
        rendered = _rendered(raised.value)
        assert "BadStatusLine" in rendered
        _assert_no_secret_in(rendered, principal, vol._mint_nonce)
    finally:
        stop_coordinator(tmp_path)
