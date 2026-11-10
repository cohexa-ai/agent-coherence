"""Whether the coordinator a ``server.pid`` names still runs (#503).

A client finds the coordinator through the port in ``.coherence/server.pid``
and sends the workspace secret with every request. Once the coordinator exits,
its port is free and any local process can bind it. Every Python coordinator
holds an exclusive ``flock`` on ``server.pid`` for as long as it runs, so a free
lock means the coordinator the file names is gone. A Node coordinator on a
host with no flock helper runs unlocked and records ``backend=node``; there its
pid decides.

Every answer leans toward "still running": only evidence that the coordinator
exited reads as gone, and anything this cannot observe (no ``fcntl``, a pid
file it cannot open) does not. Reads go through
:mod:`ccs.adapters.claude_code.regular_file`, so a link at ``server.pid`` or at
``.coherence`` is never followed.
"""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ccs.adapters.claude_code.regular_file import (
    ReadStatus,
    RegularFileRead,
    open_regular,
    read_regular_file,
    read_state_file,
)
from ccs.adapters.claude_code.state_dir import state_file_link_refusal
from ccs.coordinator.sqlite_registry import LOCK_CONTENDED_ERRNOS

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows has no flock
    fcntl = None  # type: ignore[assignment]

PID_FILE_NAME = "server.pid"
#: A pid file holds three short lines; anything longer is not one.
PID_FILE_MAX_BYTES = 4096
#: The ``backend=`` value a Node coordinator writes on the pid file's third line.
NODE_BACKEND = "node"


@dataclass(frozen=True)
class PidFileLines:
    """What a pid file says: ``<pid>``, ``<port>`` and an optional
    ``backend=<name>``. Each is ``None`` when its line is absent or malformed."""

    pid: int | None = None
    port: int | None = None
    backend: str | None = None


def parse_pid_file(text: str) -> PidFileLines:
    """The lines of a pid file, validated: a pid must be positive and a port
    between 1 and 65535."""
    lines = text.splitlines()
    pid = _int_line(lines, 0)
    port = _int_line(lines, 1)
    backend = None
    if len(lines) > 2 and lines[2].startswith("backend="):
        backend = lines[2][len("backend="):].strip() or None
    return PidFileLines(
        pid if pid is not None and pid > 0 else None,
        port if port is not None and 1 <= port <= 65535 else None,
        backend,
    )


def _int_line(lines: list[str], index: int) -> int | None:
    if index >= len(lines):
        return None
    try:
        return int(lines[index].strip())
    except ValueError:
        return None


def read_pid_file(coherence_dir: Path) -> RegularFileRead:
    """``server.pid`` read as a regular file only, never through a link at it
    or at ``coherence_dir``."""
    return read_state_file(coherence_dir, PID_FILE_NAME, max_bytes=PID_FILE_MAX_BYTES)


def read_pid_lines(pid_file: Path) -> PidFileLines:
    """The lines of ``pid_file``, read as a regular file only and never through
    a link; no lines when it is absent or cannot be read."""
    return lines_of(read_regular_file(pid_file, max_bytes=PID_FILE_MAX_BYTES))


def lines_of(read: RegularFileRead) -> PidFileLines:
    """The lines of a read that succeeded; empty lines for any other read."""
    if read.status is not ReadStatus.READ or read.data is None:
        return PidFileLines()
    return parse_pid_file(read.data.decode("utf-8", errors="replace"))


def coordinator_lock_held(pid_file: Path) -> bool | None:
    """True when another open file holds the coordinator's ``flock`` on
    ``pid_file``, False when the lock is free, None when this cannot tell (no
    ``fcntl``, or the file cannot be opened as a regular file).

    Probes with a SHARED non-blocking lock dropped at once, so two clients
    never read each other as a coordinator. A ``flock`` belongs to the open
    file, so the probe sees a coordinator running in this very process too."""
    if fcntl is None:
        return None
    try:
        fd = open_regular(pid_file)
    except OSError:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError as exc:
            return True if exc.errno in LOCK_CONTENDED_ERRNOS else None
        # Unlock before the close: a fork between the open and the close leaves
        # the child a copy of fd, whose shared lock would outlive this probe
        # and read as a running coordinator to the next start.
        with suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def pid_is_alive(pid: int) -> bool:
    """Whether a process may run with ``pid``. Signal 0 checks without
    signalling. EPERM means a process exists that this one may not signal,
    another user's or one a sandbox that denies signals hides, so it counts
    as alive; a pid no process can have does not."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, OverflowError):
        return False
    except PermissionError:
        return True
    return True


def exit_evidence(pid_file: Path, lines: PidFileLines) -> str | None:
    """What shows that the coordinator ``lines`` name has exited, or None while
    nothing does: its lock is free, or, for a Node coordinator running
    unlocked, its pid no longer exists. A held lock, or one this cannot probe,
    is no evidence."""
    if coordinator_lock_held(pid_file) is not False:
        return None
    if lines.backend != NODE_BACKEND:
        return "no longer holds the workspace lock"
    if lines.pid is not None and not pid_is_alive(lines.pid):
        return "is no longer running"
    return None


@dataclass(frozen=True)
class CoordinatorPin:
    """The coordinator a local client resolved: the lines its pid file held at
    that read."""

    coherence_dir: Path
    resolved: PidFileLines

    def departure(self) -> str | None:
        """Why the coordinator this pin names may no longer be the one on its
        port, or None when it still is or this cannot tell."""
        read = read_pid_file(self.coherence_dir)
        if read.status is not ReadStatus.READ:
            return self._unread_departure(read)
        lines, pid, port = lines_of(read), self.resolved.pid, self.resolved.port
        if (lines.pid, lines.port) != (pid, port):
            return (
                f"{read.path} now names pid {lines.pid} on port {lines.port}, "
                f"not the coordinator this client resolved (pid {pid}, port {port})"
            )
        evidence = exit_evidence(read.path, lines)
        if evidence is not None:
            return f"the coordinator {read.path} names (pid {pid}) {evidence}"
        return None

    def _unread_departure(self, read: RegularFileRead) -> str | None:
        """:meth:`departure` for a pid file this could not read. Gone or
        unreadable is cannot tell, so an attached client keeps working, except
        that a Node coordinator, which removes the file when it stops unlocked,
        is still judged by its pid where locks can be probed at all. A link or
        a file that is not a pid file is never read, and refuses."""
        if read.status in (ReadStatus.ABSENT, ReadStatus.UNREADABLE):
            if self._node_pid_has_exited():
                return f"the Node coordinator {read.path} named (pid {self.resolved.pid}) is no longer running"
            return None
        if read.status is ReadStatus.LINK:
            return state_file_link_refusal(read.path, self.coherence_dir)
        if read.status is ReadStatus.TOO_LARGE:
            return f"{read.path} is larger than a pid file ({PID_FILE_MAX_BYTES} bytes)"
        return f"{read.path} is no longer a regular file"

    def _node_pid_has_exited(self) -> bool:
        """Whether this pin names a Node coordinator whose pid no longer runs.
        Without ``fcntl`` nothing reads as gone, and no pid is
        signalled: there ``os.kill`` may not be a liveness check."""
        pid = self.resolved.pid
        if fcntl is None or self.resolved.backend != NODE_BACKEND or pid is None:
            return False
        return not pid_is_alive(pid)
