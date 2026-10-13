# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Session-handoff demo: the ``Session`` process primitive.

Each ``Session`` drives a real spawned OS process holding one ``CoherentVolume``
over a shared workspace; the first session created spawns the coordinator and
later ones attach. These tests pin the channel contract the demo's acts build
on: a deny crosses the process boundary as a typed ``SessionDenied`` — and each
act's deny is pinned to the mechanism that produced it (the coordinator, or the
file's own version check after an out-of-band rewind); any other child failure,
including a non-deny ``CoherenceError``, crosses as a typed ``SessionError``
carrying the child's traceback; a dead or hung child is bounded rather than a
stall, and teardown swallows whatever reply it finds; raw file I/O bypasses the
guard entirely (the negative control); and the two sessions really are distinct
processes.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from ccs.core.exceptions import RESTORE_OUTCOME_CONFLICT, RESTORE_OUTCOME_RESTORED
from examples.session_handoff import A_STATUS_1, A_STATUS_2, B_PICKUP, GUARDED, NOTES
from examples.session_handoff.broken import new_workspace
from examples.session_handoff.fixed import denied_by_coordinator
from examples.session_handoff.sessions import (
    _JOIN_TIMEOUT_SEC,
    _KIND_DENIED,
    Session,
    SessionDenied,
    SessionError,
)


@pytest.fixture
def workspace() -> Path:
    root = new_workspace("test")
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_session_write_denial_surfaces_as_session_denied(workspace: Path) -> None:
    a = Session(workspace, "A", GUARDED)
    try:
        b = Session(workspace, "B", GUARDED)
        try:
            a.write(A_STATUS_1)
            assert b.read() == A_STATUS_1  # B now holds a SHARED view
            a.write(A_STATUS_2)  # A's second commit invalidates B

            with pytest.raises(SessionDenied) as denied:
                b.write(B_PICKUP)  # B writes from its stale view -> denied
            assert denied.value.exc_name == "StaleView"
            assert denied.value.message  # the coordinator's reason travels verbatim
            # The coordinator denied, not the file's own version check.
            assert denied_by_coordinator(denied.value.message)
            assert (workspace / NOTES).read_bytes() == A_STATUS_2  # nothing clobbered

            assert b.reacquire() == A_STATUS_2  # fresh mandatory read
            b.write(B_PICKUP)  # recovered write lands
            assert (workspace / NOTES).read_bytes() == B_PICKUP
        finally:
            b.close()
    finally:
        a.close()


def test_session_raw_io_bypasses_the_guard(workspace: Path) -> None:
    a = Session(workspace, "A", GUARDED)
    try:
        b = Session(workspace, "B", GUARDED)
        try:
            a.raw_write(A_STATUS_1)
            assert b.raw_read() == A_STATUS_1
            a.raw_write(A_STATUS_2)

            b.raw_write(B_PICKUP)  # no exception: the stale write silently wins

            assert (workspace / NOTES).read_bytes() == B_PICKUP
        finally:
            b.close()
    finally:
        a.close()


def test_two_sessions_are_distinct_processes(workspace: Path) -> None:
    with Session(workspace, "A", GUARDED) as a, Session(workspace, "B", GUARDED) as b:
        assert a.pid != b.pid
        assert os.getpid() not in {a.pid, b.pid}


def test_session_unknown_command_surfaces_as_session_error(workspace: Path) -> None:
    a = Session(workspace, "A", GUARDED)
    try:
        with pytest.raises(SessionError) as err:
            a._call("bogus_op")  # the cheapest deterministic route into the child's error branch
        assert err.value.exc_name == "ValueError"
        assert "unknown session command" in err.value.message
        assert err.value.traceback and "ValueError" in err.value.traceback  # the child's traceback crosses

        a.raw_write(A_STATUS_1)  # the child loop survives a failed command
        assert a.raw_read() == A_STATUS_1
    finally:
        a.close()


def test_session_call_after_close_raises_session_error(workspace: Path) -> None:
    a = Session(workspace, "A", GUARDED)
    a.close()

    with pytest.raises(SessionError) as err:
        a.read()
    assert err.value.exc_name == "RuntimeError"
    assert "closed" in err.value.message

    a.close()  # closing twice is a no-op


def test_session_non_deny_coherence_error_surfaces_as_session_error(workspace: Path) -> None:
    a = Session(workspace, "A", GUARDED)
    try:
        a.write(A_STATUS_1)  # the ledger exists before the lookup
        with pytest.raises(SessionError) as err:
            a.restore("00000000-0000-0000-0000-000000000000")
        assert err.value.exc_name == "CheckpointUnknown"  # a CoherenceError, but not a typed deny
        assert err.value.traceback and "CheckpointUnknown" in err.value.traceback
    finally:
        a.close()


def test_close_tolerates_an_abandoned_reply(workspace: Path) -> None:
    """Teardown must not re-raise a reply left over from an abandoned command.

    An interrupt inside ``_await_reply`` abandons that command's reply in the
    queue; ``close()``'s own wait then collects it instead of the ``stop``
    reply. Nothing may escape teardown: every act calls ``close()`` and
    ``shutil.rmtree`` from the SAME ``finally``, so a raise here would skip the
    workspace cleanup. A deny is the reply that used to escape -- it is not a
    ``SessionError``, so the guard did not cover it.
    """
    a = Session(workspace, "A", GUARDED)
    # The exact tuple the child puts for a typed deny (sessions.py `_session_main`).
    a._reply_q.put((_KIND_DENIED, None, "StaleView", "stale read denied: handoff/notes.md", None))

    a.close()  # must not raise

    assert not a._proc.is_alive()  # the child is still reaped


def test_dead_child_surfaces_as_child_exited(workspace: Path) -> None:
    """A child that died surfaces as a typed ``SessionError``, never a stall."""
    a = Session(workspace, "A", GUARDED)
    try:
        a._proc.kill()
        a._proc.join(timeout=_JOIN_TIMEOUT_SEC)

        with pytest.raises(SessionError) as err:
            a.read()
        assert err.value.exc_name == "ChildExited"
        assert "before answering 'read'" in err.value.message
    finally:
        a.close()


def test_unanswered_command_times_out_and_terminates_the_child(workspace: Path) -> None:
    """A hung child hits the bounded deadline, is terminated, and raises."""
    a = Session(workspace, "A", GUARDED)
    try:
        # No command was sent, so no reply is coming -- the parent's view of a
        # child wedged mid-command. The wait must be bounded, not a stall.
        with pytest.raises(SessionError) as err:
            a._await_reply("probe", 0.5)
        assert err.value.exc_name == "TimeoutError"
        assert not a._proc.is_alive()  # the deadline branch terminates the child
    finally:
        a.close()


def test_act_removes_its_workspace_when_close_escapes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An act's temp workspace is removed even when teardown itself raises.

    ``close()`` swallows both channel types, but an interrupt landing in
    teardown is a ``BaseException`` and must keep propagating. Cleanup may not
    depend on that: it runs from its own ``finally``, so the caller loses the
    interrupt's meaning but never its temp directory.
    """
    from examples.session_handoff import broken, sessions

    created: list[Path] = []
    real_new_workspace = broken.new_workspace
    real_close = sessions.Session.close

    def recording_new_workspace(act: str) -> Path:
        made = real_new_workspace(act)
        created.append(made)
        return made

    def close_then_interrupt(self: sessions.Session) -> None:
        real_close(self)  # the child is still reaped
        raise KeyboardInterrupt

    monkeypatch.setattr(broken, "new_workspace", recording_new_workspace)
    monkeypatch.setattr(sessions.Session, "close", close_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        broken.run_broken()

    assert created, "the act never created a workspace"
    assert not created[0].exists()  # removed despite teardown raising


# --- the three acts: RED, GREEN, CONTROL -------------------------------------------------


def test_red_loses_the_pickup_line() -> None:
    from examples.session_handoff.broken import run_broken

    result = run_broken()

    assert result["act"] == "red"
    assert result["raised"] is None  # plain file I/O: nothing complained
    assert result["final"] == A_STATUS_2  # B's pickup line is gone
    assert result["expected"] == A_STATUS_2 + B_PICKUP
    assert result["b_line_present"] is False
    assert result["lost"] is True
    assert result["trace"] and all(isinstance(line, str) for line in result["trace"])


def test_green_denies_then_both_lines_survive() -> None:
    from examples.session_handoff.fixed import run_guarded

    result = run_guarded()

    assert result["act"] == "green"
    assert result["denied"] is True
    assert result["denial_exc"] == "StaleView"
    assert result["denial_message"]  # the coordinator's reason travels verbatim
    assert denied_by_coordinator(result["denial_message"])
    assert result["recovered"] is True
    assert result["final"] == A_STATUS_2 + B_PICKUP  # EXACT bytes: both lines survive
    assert result["expected"] == A_STATUS_2 + B_PICKUP
    assert result["b_line_present"] is True
    assert result["lost"] is False
    assert result["trace"] and all(isinstance(line, str) for line in result["trace"])


def test_control_with_guard_off_loses_again() -> None:
    from examples.session_handoff.fixed import run_control

    result = run_control()

    assert result["act"] == "control"
    assert result["denied"] is False  # NOTES sits outside the strict glob: no deny
    assert result["denial_exc"] is None
    assert result["denial_message"] is None
    assert result["recovered"] is False
    assert result["final"] == A_STATUS_2  # the loss returns
    assert result["b_line_present"] is False
    assert result["lost"] is True
    assert result["trace"] and all(isinstance(line, str) for line in result["trace"])


# --- the fourth act: HANDOFF (A stopped / A still working) -------------------------------


def test_handoff_stopped_rewinds_and_a_learns_on_next_write() -> None:
    from examples.session_handoff.handoff import run_handoff_stopped

    result = run_handoff_stopped()

    assert result["act"] == "handoff_stopped"
    assert isinstance(result["checkpoint_id"], str) and result["checkpoint_id"]
    assert len(result["members"]) == 1
    assert result["members"][0]["member_path"] == NOTES
    assert result["members"][0]["restore_tier"] == "restorable-unpinned"
    assert result["members"][0]["pin_state"] == "held"
    assert result["outcome"] == RESTORE_OUTCOME_RESTORED
    assert result["attempts"] == 1
    assert result["detail"]
    assert result["disk_after_restore"] == A_STATUS_1  # rewound to the handoff point
    assert result["a_denied"] is True  # A learns on its NEXT write, not at restore time
    assert result["a_denial_exc"] == "StaleView"
    assert result["a_denial_message"]
    # The restore bypassed A's live view; A's own version check catches it, not the coordinator.
    assert not denied_by_coordinator(result["a_denial_message"])
    assert result["a_reacquired"] == A_STATUS_1
    assert result["trace"] and all(isinstance(line, str) for line in result["trace"])
    assert result["checkpoint_id"] not in "\n".join(result["trace"])  # the uuid never enters the trace


def test_handoff_racing_concludes_conflict_not_clobber() -> None:
    from ccs.adapters.workspace import MAX_RESTORE_LEG_REDRIVES
    from examples.session_handoff.handoff import RACING_ATTEMPTS, run_handoff_racing
    from examples.session_handoff.sessions import still_working_bytes

    result = run_handoff_racing()

    # The leg budget admits the initial attempt plus every re-drive, then concludes.
    assert RACING_ATTEMPTS == MAX_RESTORE_LEG_REDRIVES + 1
    assert result["act"] == "handoff_racing"
    assert isinstance(result["checkpoint_id"], str) and result["checkpoint_id"]
    assert len(result["members"]) == 1 and result["members"][0]["member_path"] == NOTES
    assert result["outcome"] == RESTORE_OUTCOME_CONFLICT
    assert result["attempts"] == RACING_ATTEMPTS
    assert "re-drive budget exhausted" in result["detail"]
    # One racing edit per admitted attempt, so the last edit index equals the attempt count.
    assert still_working_bytes(RACING_ATTEMPTS) == f"status: still working, edit {RACING_ATTEMPTS}\n".encode()
    assert result["disk_after_restore"] == still_working_bytes(RACING_ATTEMPTS)
    assert result["restore_landed"] is False  # A's work survives; nothing was clobbered
    assert result["trace"] and all(isinstance(line, str) for line in result["trace"])
    assert result["checkpoint_id"] not in "\n".join(result["trace"])


# --- the runner: all five acts under one exit code -----------------------------------------


def test_session_handoff_demo_exits_zero() -> None:
    from examples.session_handoff.main import main

    assert main([]) == 0
