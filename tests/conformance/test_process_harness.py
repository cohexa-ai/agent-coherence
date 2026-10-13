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

import time
from pathlib import Path

import pytest

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
