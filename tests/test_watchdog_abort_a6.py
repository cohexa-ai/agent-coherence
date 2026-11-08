"""A6 — watchdog late-completion abort guard.

Before this fix, a handler that exceeded the 4s watchdog returned
``degraded: true`` but its work kept running in the pool and its mutation
(``service.write`` phantom grant, late ``service.invalidate`` grant-revocation)
landed afterward — detection-only via a counter. These tests pin the contained
mitigation: an abort Event set by the watchdog on timeout makes the late work
fail closed at the registry write lock (``abort_guard``) before it mutates.
"""

import threading
import time
from concurrent.futures import TimeoutError as FuturesTimeout
from uuid import uuid4

import pytest

import ccs.adapters.claude_code.coordinator_server as cs
from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.core.exceptions import WatchdogAbandoned
from ccs.core.states import MESIState

# --- guard correctness (deterministic, in-memory; no timing) --------------


def _service() -> CoordinatorService:
    return CoordinatorService(ArtifactRegistry())


def test_a6_write_aborts_when_abort_preset() -> None:
    """A preset abort makes write() fail closed before granting EXCLUSIVE."""
    svc = _service()
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    abort = threading.Event()
    abort.set()

    with pytest.raises(WatchdogAbandoned):
        svc.write(agent_id=agent, artifact_id=art.id, abort=abort)

    assert svc.registry.get_agent_state(art.id, agent) != MESIState.EXCLUSIVE, (
        "no phantom EXCLUSIVE grant must land when the watchdog already aborted"
    )


def test_a6_write_applies_when_abort_clear() -> None:
    """An unset abort (or None) is a no-op — the grant applies normally."""
    svc = _service()
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()

    svc.write(agent_id=agent, artifact_id=art.id, abort=threading.Event())
    assert svc.registry.get_agent_state(art.id, agent) == MESIState.EXCLUSIVE


def test_a6_invalidate_aborts_when_abort_preset_does_not_revoke() -> None:
    """The worst case: a late operator-drain release must NOT revoke a grant.
    The Stop path's twin is
    tests/test_self_release_reader.py::test_a_release_whose_request_already_timed_out_changes_nothing."""
    svc = _service()
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    svc.write(agent_id=agent, artifact_id=art.id)  # agent -> EXCLUSIVE
    assert svc.registry.get_agent_state(art.id, agent) == MESIState.EXCLUSIVE

    abort = threading.Event()
    abort.set()
    with pytest.raises(WatchdogAbandoned):
        svc.invalidate(
            agent_id=agent,
            artifact_id=art.id,
            new_version=art.version,
            issuer_agent_id=agent,
            issued_at_tick=1,
            abort=abort,
        )

    assert svc.registry.get_agent_state(art.id, agent) == MESIState.EXCLUSIVE, (
        "a late, aborted invalidate must not revoke the still-valid grant"
    )


# --- watchdog wiring (timing-bounded, uses the real sqlite coordinator) ----


def test_a6_watchdog_timeout_sets_abort_event(tmp_path) -> None:
    """run_with_watchdog SETS the caller's abort Event on timeout."""
    with cs.CoordinatorHTTPServer(tmp_path, port=0, instance_id="a6-set") as coordinator:
        release = threading.Event()
        abort = threading.Event()

        def slow_work() -> dict:
            release.wait(timeout=5.0)
            return {"ok": True}

        original = cs.HANDLER_TIMEOUT_SEC
        try:
            cs.HANDLER_TIMEOUT_SEC = 0.05
            with pytest.raises(FuturesTimeout):
                coordinator.run_with_watchdog(slow_work, abort=abort)
            assert abort.is_set(), "watchdog timeout must signal the runaway work to abort"
        finally:
            cs.HANDLER_TIMEOUT_SEC = original
            release.set()


def test_a6_late_write_aborts_with_no_phantom_grant(tmp_path) -> None:
    """End to end: a write whose handler times out aborts at the registry lock
    when it finally runs — no phantom EXCLUSIVE, and the abort is counted."""
    with cs.CoordinatorHTTPServer(tmp_path, port=0, instance_id="a6-e2e") as coordinator:
        art = coordinator.service.register_artifact(name="plan.md", content="v1")
        agent = uuid4()
        release = threading.Event()
        abort = threading.Event()

        def work() -> list:
            release.wait(timeout=5.0)  # block past the watchdog deadline
            return coordinator.service.write(
                agent_id=agent, artifact_id=art.id, abort=abort
            )

        original = cs.HANDLER_TIMEOUT_SEC
        try:
            cs.HANDLER_TIMEOUT_SEC = 0.05
            before = coordinator._watchdog_late_aborts_total
            with pytest.raises(FuturesTimeout):
                coordinator.run_with_watchdog(work, abort=abort)
            # The watchdog has set `abort`; release the work so it reaches the
            # guarded write, which must now abort rather than grant.
            release.set()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if coordinator._watchdog_late_aborts_total > before:
                    break
                time.sleep(0.02)
            assert coordinator._watchdog_late_aborts_total == before + 1, (
                "the aborted late completion must bump watchdog_late_aborts_total"
            )
            assert coordinator.registry.get_agent_state(art.id, agent) != MESIState.EXCLUSIVE, (
                "the late write must not have granted a phantom EXCLUSIVE"
            )
        finally:
            cs.HANDLER_TIMEOUT_SEC = original
            release.set()
