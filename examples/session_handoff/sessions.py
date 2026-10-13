# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The ``Session`` primitive: one agent session = one real OS process.

A :class:`Session` lives in the demo's parent process and drives a long-lived
spawned child that holds ONE ``CoherentVolume`` over the shared workspace. Two
sessions are therefore two processes, exactly as two agent sessions on one
host would be; the first child to construct its volume spawns the coordinator
(an in-process serving thread) and every later child attaches to it.

Commands cross a spawn-context queue pair and come back typed: one of the
volume's typed denies — ``StaleView`` or ``CommitPreempted``, from the
coordinator or from the file's own version check — surfaces in the parent as
:class:`SessionDenied`; anything else the child raises, including any other
``CoherenceError`` (an unreachable coordinator, an unknown checkpoint), surfaces
as :class:`SessionError` carrying the child's traceback. Checkpoint and restore
run in the child too, over a per-command ``WorkspaceVersioner`` — the ledger
is opened, used, and closed inside each command, never held across commands.

Teardown order matters: close sessions in reverse creation order. Closing the
FIRST session stops the coordinator its child spawned, which attached sessions
still need.
"""

from __future__ import annotations

import multiprocessing
import queue
import time
import traceback
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid5

from ccs.adapters.workspace import WorkspaceVersioner
from ccs.cli.workspace import RetainedContentResolver, WorkingTreeSource
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from examples.session_handoff import DEMO_CFG, NOTES, OWNER

_REPLY_TIMEOUT_SEC = 60.0
_STOP_TIMEOUT_SEC = 30.0
_JOIN_TIMEOUT_SEC = 10.0
_POLL_SEC = 0.05
_PARENT_WATCH_SEC = 1.0

_KIND_OK = "ok"
_KIND_DENIED = "denied"
_KIND_ERROR = "error"


class SessionDenied(Exception):
    """One of the volume's typed denies — ``StaleView`` or ``CommitPreempted`` —
    raised by the coordinator or by the file's own version check. Every other
    child-side failure is a :class:`SessionError`."""

    def __init__(self, exc_name: str, message: str) -> None:
        super().__init__(f"{exc_name}: {message}")
        self.exc_name = exc_name
        self.message = message


class SessionError(Exception):
    """Any other child-side failure, with the child's traceback attached."""

    def __init__(self, exc_name: str, message: str, traceback_text: str) -> None:
        super().__init__(f"{exc_name}: {message}")
        self.exc_name = exc_name
        self.message = message
        self.traceback = traceback_text


class Session:
    """Parent-side handle on one spawned session process.

    Constructing a ``Session`` spawns its child and eagerly builds the child's
    ``CoherentVolume`` (the ``init`` command), so the first ``Session`` created
    is the one whose child spawns the coordinator.
    """

    def __init__(self, workspace: Path, role: str, managed: tuple[str, ...]) -> None:
        self.role = role
        self._workspace = Path(workspace).resolve()
        self._closed = False
        ctx = multiprocessing.get_context("spawn")
        self._cmd_q = ctx.Queue()
        self._reply_q = ctx.Queue()
        self._proc = ctx.Process(
            target=_session_main,
            args=(self._cmd_q, self._reply_q, str(self._workspace), role, tuple(managed)),
            daemon=True,
        )
        self._proc.start()
        try:
            self._call("init")
        except BaseException:
            self._terminate()
            self._release_queues()
            self._closed = True
            raise

    # -- identity -----------------------------------------------------------

    @property
    def pid(self) -> int:
        """OS pid of the child process."""
        return int(self._proc.pid or 0)

    # -- plain file I/O (no CoherentVolume; the unguarded baseline) -----------

    def raw_read(self) -> bytes:
        return self._call("raw_read")

    def raw_write(self, data: bytes) -> None:
        self._call("raw_write", bytes(data))

    # -- CoherentVolume ------------------------------------------------------

    def read(self) -> bytes:
        return self._call("read")

    def write(self, data: bytes) -> None:
        self._call("write", bytes(data))

    def reacquire(self) -> bytes:
        return self._call("reacquire")

    # -- checkpoint / restore -------------------------------------------------

    def checkpoint(self, name: str) -> dict:
        return self._call("checkpoint", name)

    def members(self, checkpoint_id: str) -> list[dict]:
        return self._call("members", checkpoint_id)

    def restore(self, checkpoint_id: str, *, racing: bool = False) -> dict:
        return self._call("restore", checkpoint_id, bool(racing))

    # -- lifecycle -------------------------------------------------------------

    def close(self) -> None:
        """Stop the child (and, in the spawner, the coordinator); never hangs."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.is_alive():
                self._cmd_q.put(("stop",))
                # Both channel types, because the wait is not correlated to the
                # command: an interrupt that abandoned an earlier reply leaves it
                # in the queue, and this wait collects THAT instead of the stop
                # reply. Teardown must swallow it either way -- callers run
                # close() and their workspace cleanup from one ``finally``.
                with suppress(SessionError, SessionDenied):
                    self._await_reply("stop", _STOP_TIMEOUT_SEC)
            self._proc.join(timeout=_JOIN_TIMEOUT_SEC)
        finally:
            if self._proc.is_alive():
                self._terminate()
            self._release_queues()

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- the channel -----------------------------------------------------------

    def _call(self, op: str, *args: Any) -> Any:
        if self._closed:
            raise SessionError("RuntimeError", f"session {self.role!r} is closed", "")
        self._cmd_q.put((op, *args))
        return self._await_reply(op, _REPLY_TIMEOUT_SEC)

    def _await_reply(self, op: str, timeout_sec: float) -> Any:
        """Bounded wait; a dead or hung child is a typed ``SessionError``, never a stall."""
        deadline = time.monotonic() + timeout_sec
        while True:
            with suppress(queue.Empty):
                return self._unpack(self._reply_q.get(timeout=_POLL_SEC))
            if not self._proc.is_alive():
                raise SessionError(
                    "ChildExited",
                    f"session {self.role!r} exited (code {self._proc.exitcode}) before answering {op!r}",
                    "",
                )
            if time.monotonic() >= deadline:
                self._terminate()
                raise SessionError(
                    "TimeoutError",
                    f"session {self.role!r} did not answer {op!r} within {timeout_sec}s; terminated",
                    "",
                )

    @staticmethod
    def _unpack(reply: tuple[Any, ...]) -> Any:
        kind, value, exc_name, exc_str, tb_text = reply
        if kind == _KIND_OK:
            return value
        if kind == _KIND_DENIED:
            raise SessionDenied(exc_name, exc_str)
        raise SessionError(exc_name, exc_str, tb_text or "")

    def _terminate(self) -> None:
        if not self._proc.is_alive():
            return
        self._proc.terminate()
        self._proc.join(timeout=5.0)
        if self._proc.is_alive():  # pragma: no cover - stubborn child
            self._proc.kill()
            self._proc.join(timeout=5.0)

    def _release_queues(self) -> None:
        # cancel_join_thread first: a queue whose feeder still holds an item the
        # (dead) child never consumed would otherwise block close() forever.
        for q in (self._cmd_q, self._reply_q):
            with suppress(Exception):
                q.cancel_join_thread()
                q.close()


# --- child side ---------------------------------------------------------------


def still_working_bytes(edit: int) -> bytes:
    """The notes bytes after the racing source's ``edit``-th edit — the one place this format lives."""
    return f"status: still working, edit {edit}\n".encode()


class _StillWorkingSource:
    """Wrap a ``WorkingTreeSource``: every read of ``NOTES`` is followed by a
    fresh edit to the disk file — the handing-off session is still typing. The
    restorer's comparand is therefore stale by the time it CASes; the leg
    re-drives, reads again, and is beaten again, until the bounded budget
    exhausts into the honest absorbing ``conflict`` (nothing clobbered)."""

    def __init__(self, inner: WorkingTreeSource, workspace: Path) -> None:
        self._inner = inner
        self._workspace = workspace
        self._edits = 0

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        observed = self._inner.read_with_version(path)
        if path == NOTES:
            self._edits += 1
            (self._workspace / NOTES).write_bytes(still_working_bytes(self._edits))
        return observed

    def write_cas_at(self, path: str, expected_version: int, new_content: bytes) -> None:
        # Spelled out rather than left to __getattr__: the restore pre-flight
        # checks the FileRestoreTarget protocol statically, which never sees
        # attributes that only resolve through __getattr__.
        self._inner.write_cas_at(path, expected_version, new_content)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _open_registry(workspace: Path) -> SqliteArtifactRegistry:
    return SqliteArtifactRegistry(workspace / ".coherence" / "workspace.db", retain_versions=True)


@contextmanager
def _versioner(workspace: Path, role: str, *, racing: bool = False) -> Iterator[WorkspaceVersioner]:
    """Per-command versioner over the workspace ledger: open → use → close."""
    registry = _open_registry(workspace)
    try:
        service = CoordinatorService(registry)
        source: Any = WorkingTreeSource(workspace, registry, uuid5(OWNER, role))
        if racing:
            source = _StillWorkingSource(source, workspace)
        resolver = RetainedContentResolver(registry)
        versioner = WorkspaceVersioner(service=service, owner=OWNER, file_resolver=resolver)
        versioner.add_file_member(source, NOTES)
        yield versioner
    finally:
        registry.close()


def _member_row(member: Any) -> dict:
    return {
        "member_path": member.member_path,
        "restore_tier": member.restore_tier,
        "pin_state": member.pin_state,
    }


def _checkpoint(workspace: Path, role: str, name: str) -> dict:
    with _versioner(workspace, role) as versioner:
        result = versioner.checkpoint(name)
    return {
        "checkpoint_id": result.record.checkpoint_id,
        "members": [_member_row(m) for m in result.members],
    }


def _members(workspace: Path, checkpoint_id: str) -> list[dict]:
    registry = _open_registry(workspace)
    try:
        return [_member_row(m) for m in registry.get_checkpoint_members(checkpoint_id)]
    finally:
        registry.close()


def _restore(workspace: Path, role: str, checkpoint_id: str, *, racing: bool) -> dict:
    with _versioner(workspace, role, racing=racing) as versioner:
        report = versioner.restore(checkpoint_id)
    return {
        "members": [
            {
                "member_path": m.member_path,
                "outcome": m.outcome,
                "attempts": m.attempts,
                "detail": m.detail,
            }
            for m in report.members
        ]
    }


def _dispatch(op: str, args: tuple[Any, ...], state: dict[str, Any]) -> Any:
    workspace: Path = state["workspace"]
    role: str = state["role"]
    if op == "raw_read":
        return (workspace / NOTES).read_bytes()
    if op == "raw_write":
        (workspace / NOTES).write_bytes(args[0])
        return None
    if op == "checkpoint":
        return _checkpoint(workspace, role, args[0])
    if op == "members":
        return _members(workspace, args[0])
    if op == "restore":
        return _restore(workspace, role, args[0], racing=args[1])
    volume = state["volume"]
    if volume is None:
        raise RuntimeError("session volume is not initialised")
    if op == "read":
        return volume.read(NOTES)
    if op == "write":
        volume.write(NOTES, args[0])
        return None
    if op == "reacquire":
        return volume.reacquire(NOTES)
    raise ValueError(f"unknown session command {op!r}")


def _next_command(cmd_q: Any) -> tuple[Any, ...] | None:  # pragma: no cover - runs in the spawned child
    """Wait for the next command; ``None`` once the parent process is gone.

    A waiting ``get`` still returns the moment a command lands; the timeout only
    bounds how long the child goes without checking that its parent is alive.
    """
    while True:
        with suppress(queue.Empty):
            return tuple(cmd_q.get(timeout=_PARENT_WATCH_SEC))
        parent = multiprocessing.parent_process()
        if parent is None or not parent.is_alive():
            return None


def _session_main(
    cmd_q: Any, reply_q: Any, workspace_str: str, role: str, managed: tuple[str, ...]
) -> None:  # pragma: no cover - runs in the spawned child
    # Imported here so the child pays for the volume runtime only once alive.
    from ccs.adapters.claude_code.lifecycle import stop_coordinator
    from ccs.adapters.coherent_volume import CoherentVolume
    from ccs.core.exceptions import CommitPreempted, StaleView

    state: dict[str, Any] = {"workspace": Path(workspace_str), "role": role, "volume": None}
    while True:
        command = _next_command(cmd_q)
        if command is None:
            # ``daemon=True`` reaps this child only through the parent's atexit
            # hook, which a hard kill (SIGKILL, OOM, CI timeout) never runs; an
            # unbounded ``get`` would then block forever holding the coordinator.
            stop_coordinator(state["workspace"])
            return
        op, args = command[0], tuple(command[1:])
        try:
            if op == "stop":
                # ``stop_coordinator`` only knows coordinators spawned in THIS
                # process: True in the spawner's child, a no-op False elsewhere.
                reply_q.put((_KIND_OK, stop_coordinator(state["workspace"]), None, None, None))
                return
            if op == "init":
                state["volume"] = CoherentVolume(state["workspace"], managed=managed, config=DEMO_CFG)
                value = None
            else:
                value = _dispatch(op, args, state)
            reply_q.put((_KIND_OK, value, None, None, None))
        except (StaleView, CommitPreempted) as exc:
            # Only the volume's typed denies are denies; any other CoherenceError
            # (unreachable coordinator, CheckpointUnknown, ...) is a failure and
            # keeps its traceback via the branch below.
            # A deny carries no traceback: the parent raises SessionDenied from the
            # name and message alone, so formatting one here would only be discarded.
            reply_q.put((_KIND_DENIED, None, type(exc).__name__, str(exc), None))
        except BaseException as exc:  # noqa: BLE001 - the channel carries everything
            reply_q.put((_KIND_ERROR, None, type(exc).__name__, str(exc), traceback.format_exc()))
            if op == "stop":
                return
