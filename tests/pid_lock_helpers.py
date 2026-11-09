# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The workspace lock a running coordinator holds, for tests that stand in a
coordinator of their own.

A client reaches a coordinator only while ``server.pid`` names it and it still
holds the ``flock`` a start takes on that file (#503). A test that writes the pid
file for a server it runs itself holds that lock too, through
:func:`hold_coordinator_lock`. The lock belongs to the open file, so a client in
the same process reads it as held. ``tests/conftest.py`` releases every lock
after each test. Standard library only, so the conftest can import it.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import suppress
from pathlib import Path

_HELD: dict[Path, int] = {}


def hold_coordinator_lock(pid_file: Path) -> None:
    """Hold the exclusive lock a running coordinator holds on ``pid_file``
    until the test ends. A pid file this test already holds stays held."""
    key = pid_file.resolve()
    if key in _HELD:
        return
    fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    _HELD[key] = fd


def write_coordinator_pid_file(coherence_dir: Path, port: int, pid: int = 12345) -> Path:
    """Write ``server.pid`` for a server the test runs on ``port``, and hold
    its lock as that server's coordinator would."""
    pid_file = coherence_dir / "server.pid"
    pid_file.write_text(f"{pid}\n{port}\n")
    hold_coordinator_lock(pid_file)
    return pid_file


def release_coordinator_locks() -> None:
    """Release every lock :func:`hold_coordinator_lock` took: the stand-in
    coordinators exit."""
    while _HELD:
        _, fd = _HELD.popitem()
        with suppress(OSError):
            os.close(fd)
