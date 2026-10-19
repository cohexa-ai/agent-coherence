# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""U1 — the cross-process contender harness (guarantee-ladder plan).

The harness is what makes a kit "race" mean something: contenders are separate
OS processes rendezvoused at a real barrier, so an in-process lock (the RLock
that satisfies any single-process interleaving) cannot fake the result. The
barrier doubles as the concurrency witness — with ``parties == n`` a harness
that ran contenders serially would deadlock at the barrier, so a green run IS
proof of overlap.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

from ccs.testing import process_harness
from ccs.testing.process_harness import (
    ContenderError,
    ContenderSpec,
    DeclaredExemption,
    HarnessFailure,
    ProcessRaceHarness,
)

# Spawns real processes that race each other: run with -n 0 (tests/conftest.py).
pytestmark = pytest.mark.serial

# ---------------------------------------------------------------------------
# Top-level contender callables (spawn-context children unpickle by qualname,
# so they must be module-level and import-clean).
# ---------------------------------------------------------------------------


def _append_and_return(ctx, path_str: str, token: str) -> str:
    """Rendezvous, then append our token to a shared file, then return it."""
    ctx.barrier_wait()
    ctx.delay()
    path = Path(path_str)
    with path.open("a") as fh:
        fh.write(token + "\n")
    return token


def _raise_after_barrier(ctx) -> None:
    ctx.barrier_wait()
    raise RuntimeError("contender deliberately failed")


def _hang_forever(ctx) -> None:
    ctx.barrier_wait()
    time.sleep(3600)


def _return_delay(ctx) -> float:
    ctx.barrier_wait()
    ctx.delay()
    return ctx.delay_seconds


def _raise_before_barrier(ctx) -> None:
    raise RuntimeError("crashed before the rendezvous")


def _barrier_then_return(ctx) -> str:
    ctx.barrier_wait()
    return "made it"


# ---------------------------------------------------------------------------
# Scenarios (plan U1)
# ---------------------------------------------------------------------------


def test_two_contenders_run_concurrently_and_both_outcomes_arrive(tmp_path: Path) -> None:
    shared = tmp_path / "shared.txt"
    shared.touch()
    harness = ProcessRaceHarness(timeout_sec=30.0)
    result = harness.race(
        [
            ContenderSpec(_append_and_return, args=(str(shared), "A")),
            ContenderSpec(_append_and_return, args=(str(shared), "B"), delay_seconds=0.05),
        ]
    )
    # Both outcomes arrived on the channel, in contender order.
    assert [o.value for o in result.outcomes] == ["A", "B"]
    assert not result.errors
    # The shared-state witness: both writes landed (the barrier already proved
    # overlap — a serial run would have deadlocked and hit the timeout).
    assert sorted(shared.read_text().split()) == ["A", "B"]


def test_raising_contender_is_captured_and_sibling_survives(tmp_path: Path) -> None:
    shared = tmp_path / "shared.txt"
    shared.touch()
    harness = ProcessRaceHarness(timeout_sec=30.0)
    result = harness.race(
        [
            ContenderSpec(_append_and_return, args=(str(shared), "OK")),
            ContenderSpec(_raise_after_barrier),
        ]
    )
    ok, err = result.outcomes
    assert ok.value == "OK"
    assert isinstance(err.error, ContenderError)
    assert "deliberately failed" in str(err.error)
    assert result.errors == [err]


def test_hanging_contender_is_killed_and_reported_as_harness_failure() -> None:
    harness = ProcessRaceHarness(timeout_sec=2.0)
    with pytest.raises(HarnessFailure, match="timed out"):
        harness.race(
            [
                ContenderSpec(_return_delay),
                ContenderSpec(_hang_forever),
            ]
        )


def test_pre_barrier_crash_aborts_the_barrier_so_the_sibling_is_not_stranded() -> None:
    """A contender that crashes BEFORE the rendezvous must not strand its
    sibling at the barrier until the harness timeout: the harness aborts the
    barrier on the first collected error, so the run finishes promptly with
    the crasher's root-cause error in the result."""
    harness = ProcessRaceHarness(timeout_sec=30.0)
    start = time.monotonic()
    result = harness.race(
        [
            ContenderSpec(_raise_before_barrier),
            ContenderSpec(_barrier_then_return),
        ]
    )
    elapsed = time.monotonic() - start
    # Generous bound (spawn startup only) — well under the 30s harness timeout,
    # which the run would have ridden out before the abort existed.
    assert elapsed < 15.0
    crasher, sibling = result.outcomes
    assert isinstance(crasher.error, ContenderError)
    assert "RuntimeError" in crasher.error.exc_repr
    assert "crashed before the rendezvous" in str(crasher.error)
    # The sibling either broke at the aborted barrier (its own error outcome
    # pointing at the root cause) or happened to finish before the abort.
    if sibling.error is not None:
        assert "BrokenBarrierError" in sibling.error.exc_repr
    else:
        assert sibling.value == "made it"


def test_delay_parameter_reaches_the_child() -> None:
    harness = ProcessRaceHarness(timeout_sec=30.0)
    result = harness.race(
        [
            ContenderSpec(_return_delay, delay_seconds=0.0),
            ContenderSpec(_return_delay, delay_seconds=0.02),
        ]
    )
    assert [o.value for o in result.outcomes] == [0.0, 0.02]


# ---------------------------------------------------------------------------
# Exemption seam (R-8; KTD-4 is its first consumer) — Covers AE5.
# ---------------------------------------------------------------------------


class _InProcessOnlyBinding:
    """A binding that declares it cannot be shared across OS processes."""

    in_process_only = True
    in_process_only_reason = "GIL-serialized process memory by construction"


def test_in_process_only_binding_is_refused_with_a_recorded_exemption() -> None:
    harness = ProcessRaceHarness(timeout_sec=5.0)
    exemption = harness.refuse_to_race(_InProcessOnlyBinding(), clause="R-1 cross-process race")
    assert isinstance(exemption, DeclaredExemption)
    assert exemption.subject == "_InProcessOnlyBinding"
    assert "GIL-serialized" in exemption.reason
    # The run report names it — nothing is silently skipped (R-8).
    report = harness.report()
    assert any("R-1 cross-process race" in line and "_InProcessOnlyBinding" in line for line in report)


def test_refusal_requires_a_declared_reason() -> None:
    class _Undeclared:
        in_process_only = True  # no reason given

    harness = ProcessRaceHarness(timeout_sec=5.0)
    with pytest.raises(HarnessFailure, match="reason"):
        harness.refuse_to_race(_Undeclared(), clause="R-1 cross-process race")


# ---------------------------------------------------------------------------
# Park/resume (run_park_after_ack): stop a contender after its ack, give the
# parent a window, resume it on demand, return what it produces afterwards.
# ---------------------------------------------------------------------------


def _ack_pid_then_report(ctx, marker_str: str) -> dict:
    """Ack our pid; once resumed, leave a marker and report the flag we saw."""
    ctx.ack(os.getpid())
    Path(marker_str).write_text("resumed")
    return {"resumed_flag": ctx.resumed}


def _ack_while_threads_poll_the_flag(ctx, n_threads: int) -> int:
    """Ack while sibling threads poll the resume flag flat out — the shape of a
    heartbeat thread gated by the flag, and the load under which a locked
    primitive would be held by the stopped child when the parent touched it."""

    def _poll() -> None:
        while not ctx.resumed:
            pass

    pollers = [threading.Thread(target=_poll) for _ in range(n_threads)]
    for poller in pollers:
        poller.start()
    ctx.ack(os.getpid())
    for poller in pollers:
        poller.join()
    return n_threads


def _raise_before_ack(ctx) -> None:
    raise RuntimeError("failed before acking")


def _raise_after_resume(ctx) -> None:
    ctx.ack(os.getpid())
    raise RuntimeError("failed after resuming")


def _return_without_ack(ctx) -> str:
    return "never acked"


def _ack_twice(ctx) -> str:
    ctx.ack("first")
    ctx.ack("second")
    return "after both"


def _hang_before_ack(ctx) -> None:
    time.sleep(3600)


def _hang_after_resume(ctx) -> None:
    ctx.ack(os.getpid())
    time.sleep(3600)


def _ps_state(pid: int) -> str:
    """The process state letters from ``ps`` — an observer independent of the
    harness's own stop confirmation (``T`` = stopped, on macOS and Linux)."""
    return subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()


def _assert_reaped(pid: int) -> None:
    """The child is dead AND reaped: signal 0 finds no process (a zombie would
    still answer), and multiprocessing tracks no live child of ours."""
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert not multiprocessing.active_children()


def test_park_after_ack_stops_the_child_then_returns_its_post_resume_value(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "resumed.txt"
    seen: dict = {}

    def while_parked(pid: int) -> str:
        seen["state"] = _ps_state(pid)
        seen["marker_during_window"] = marker.exists()
        return "observed"

    result = process_harness.run_park_after_ack(
        ContenderSpec(_ack_pid_then_report, args=(str(marker),)),
        while_parked,
        timeout_sec=30.0,
    )
    # During the window the child was stopped and had not run past its ack.
    assert "T" in seen["state"]
    assert seen["marker_during_window"] is False
    # After the resume it ran on, saw the flag set, and its value came back.
    assert marker.read_text() == "resumed"
    assert result.value == {"resumed_flag": True}
    assert result.window == "observed"
    assert isinstance(result.acked, int)
    _assert_reaped(result.acked)


def test_parent_never_blocks_on_a_shared_primitive_while_the_child_is_stopped() -> None:
    """Every cycle stops the child while two of its threads poll the resume
    flag flat out. A locked gate (Event, Condition, locked Value) would be held
    by a stopped poller often enough that the parent's set() blocks against it
    and the cycle never finishes; the lock-free flag cannot be held."""
    cycles = 5
    pids: list[int] = []
    states: list[str] = []
    values: list[int] = []

    def while_parked(pid: int) -> None:
        pids.append(pid)
        states.append(_ps_state(pid))

    def run_cycles() -> None:
        for _ in range(cycles):
            result = process_harness.run_park_after_ack(
                ContenderSpec(_ack_while_threads_poll_the_flag, args=(2,)),
                while_parked,
                timeout_sec=20.0,
            )
            values.append(result.value)

    worker = threading.Thread(target=run_cycles, daemon=True)
    worker.start()
    worker.join(timeout=120.0)
    blocked = worker.is_alive()
    if blocked:  # leave no stopped child behind, even when the parent hung
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert not blocked, f"parent blocked against a stopped child after {len(values)} cycle(s)"
    assert values == [2] * cycles
    assert all("T" in state for state in states)
    for pid in pids:
        _assert_reaped(pid)


@pytest.mark.parametrize(
    ("contender", "text"),
    [(_raise_before_ack, "failed before acking"), (_raise_after_resume, "failed after resuming")],
)
def test_raising_contender_surfaces_contender_error(contender, text: str) -> None:
    with pytest.raises(ContenderError, match=text) as excinfo:
        process_harness.run_park_after_ack(ContenderSpec(contender), lambda acked: None, timeout_sec=30.0)
    assert "RuntimeError" in excinfo.value.exc_repr
    assert not multiprocessing.active_children()


def test_contender_returning_without_ack_is_a_harness_failure() -> None:
    windows: list = []
    with pytest.raises(HarnessFailure, match="without acking"):
        process_harness.run_park_after_ack(ContenderSpec(_return_without_ack), windows.append, timeout_sec=30.0)
    assert windows == []
    assert not multiprocessing.active_children()


def test_second_ack_is_a_harness_failure_not_a_post_resume_value() -> None:
    """A second ack would otherwise arrive where the post-resume value is read
    and be returned as if the contender had produced it."""
    with pytest.raises(HarnessFailure, match="exactly once"):
        process_harness.run_park_after_ack(ContenderSpec(_ack_twice), lambda acked: None, timeout_sec=30.0)
    assert not multiprocessing.active_children()


@pytest.mark.parametrize(
    ("contender", "phase"),
    [(_hang_before_ack, "the ack"), (_hang_after_resume, "the post-resume value")],
)
def test_hanging_contender_times_out_as_harness_failure(contender, phase: str) -> None:
    with pytest.raises(HarnessFailure, match=f"timed out .* awaiting {phase}"):
        process_harness.run_park_after_ack(ContenderSpec(contender), lambda acked: None, timeout_sec=2.0)
    assert not multiprocessing.active_children()


class _AbortWindow(BaseException):
    """Not an Exception: the kill must run on BaseException exits too."""


@pytest.mark.parametrize("failure", [RuntimeError, _AbortWindow])
def test_failure_while_the_child_is_stopped_kills_it_and_leaves_no_process(
    tmp_path: Path, failure: type[BaseException]
) -> None:
    pids: list[int] = []

    def while_parked(pid: int) -> None:
        pids.append(pid)
        assert "T" in _ps_state(pid)
        raise failure("the parent's window failed")

    with pytest.raises(failure, match="window failed"):
        process_harness.run_park_after_ack(
            ContenderSpec(_ack_pid_then_report, args=(str(tmp_path / "m.txt"),)),
            while_parked,
            timeout_sec=30.0,
        )
    assert len(pids) == 1
    _assert_reaped(pids[0])
    # Killed while stopped: it never ran past its ack.
    assert not (tmp_path / "m.txt").exists()


def test_unconfirmed_stop_is_a_harness_failure_and_the_stopped_child_is_killed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The SIGSTOP really lands, but the confirmation never reports it: the run
    fails rather than open the window unconfirmed, and the child still dies."""
    sent: list[tuple[int, int]] = []
    real_kill = os.kill

    def recording_kill(pid: int, sig: int) -> None:
        sent.append((pid, sig))
        real_kill(pid, sig)

    monkeypatch.setattr(os, "waitid", lambda *args: None)
    monkeypatch.setattr(os, "kill", recording_kill)
    windows: list = []
    with pytest.raises(HarnessFailure, match="stop"):
        process_harness.run_park_after_ack(
            ContenderSpec(_ack_pid_then_report, args=(str(tmp_path / "m.txt"),)),
            windows.append,
            timeout_sec=2.0,
        )
    assert windows == []
    stopped = [pid for pid, sig in sent if sig == signal.SIGSTOP]
    assert len(stopped) == 1
    monkeypatch.undo()
    _assert_reaped(stopped[0])


def test_platform_without_sigstop_is_a_harness_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(signal, "SIGSTOP")
    windows: list = []
    with pytest.raises(HarnessFailure, match="SIGSTOP"):
        process_harness.run_park_after_ack(ContenderSpec(_return_without_ack), windows.append, timeout_sec=5.0)
    assert windows == []
    assert not multiprocessing.active_children()


def test_stop_switch_off_sends_no_sigstop_but_keeps_the_window_and_the_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "resumed.txt"
    sent: list[int] = []
    real_kill = os.kill
    seen: dict = {}

    def recording_kill(pid: int, sig: int) -> None:
        sent.append(sig)
        real_kill(pid, sig)

    def while_parked(pid: int) -> str:
        # Give a premature resume time to show itself before looking.
        time.sleep(0.5)
        seen["state"] = _ps_state(pid)
        seen["marker_during_window"] = marker.exists()
        return "observed"

    monkeypatch.setattr(os, "kill", recording_kill)
    result = process_harness.run_park_after_ack(
        ContenderSpec(_ack_pid_then_report, args=(str(marker),)),
        while_parked,
        stop=False,
        timeout_sec=30.0,
    )
    assert signal.SIGSTOP not in sent and signal.SIGCONT not in sent
    # The window still ran between the ack and the resume: the child was alive,
    # running (not stopped), and still held at its ack.
    assert seen["state"] and "T" not in seen["state"]
    assert seen["marker_during_window"] is False
    assert result.window == "observed"
    assert result.value == {"resumed_flag": True}
    assert marker.read_text() == "resumed"
    monkeypatch.undo()
    _assert_reaped(result.acked)
