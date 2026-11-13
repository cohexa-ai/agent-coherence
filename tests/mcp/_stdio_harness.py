# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Real stdio sessions with the ``stale-write-guard-fs`` server, for journeys
that need more than one server process on one workspace.

The coordinator belongs to the test process: a volume built here binds it, with
the grant timers the test chooses, and stops it on exit. Each server is a real
subprocess, ``python -m ccs.mcp.server``, started with an explicit environment
whose ``PYTHONPATH`` is this checkout's ``src``, and driven by
``mcp.client.stdio`` as an agent host drives it. A server that does not bind the
coordinator does not stop it when it exits, so servers come and go while the
coordinator and every grant it records stay put.

Every server process started here is stopped when its block ends, and the block
then checks that none of them is still a child of this process.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import CoherenceError
from ccs.mcp.server import INSTRUCTIONS

#: This checkout's sources: the code every server process here runs.
SRC = Path(__file__).resolve().parents[2] / "src"
SERVER_MODULE = "ccs.mcp.server"
#: The managed globs the host volume and every server use; they must agree,
#: or a server's attach refuses the coordinator's policy.
MANAGED_GLOB = "data/**"
#: How long a start (imports, attach, initialize) or one tool call may take.
START_TIMEOUT_SEC = 60.0
CALL_TIMEOUT_SEC = 30.0
#: The states a ``/status`` session row holds a path in for writing.
_WRITE_STATES = ("EXCLUSIVE", "MODIFIED")


@dataclass(frozen=True)
class ToolAnswer:
    """One tool call's result as a stdio client receives it."""

    is_error: bool
    structured: dict
    texts: tuple[str, ...]

    @classmethod
    def of(cls, result: CallToolResult) -> ToolAnswer:
        texts = tuple(item.text for item in result.content if item.type == "text")
        return cls(bool(result.isError), dict(result.structuredContent or {}), texts)


class StdioServer:
    """One server process and the client session that drives it."""

    def __init__(self, name: str, session: ClientSession, pid: int) -> None:
        self.name = name
        self.pid = pid
        self._session = session

    async def call(self, tool: str, **arguments: object) -> ToolAnswer:
        result = await asyncio.wait_for(self._session.call_tool(tool, arguments), timeout=CALL_TIMEOUT_SEC)
        return ToolAnswer.of(result)

    async def status(self) -> dict:
        """The structured ``swg_status`` answer, which must not be an error."""
        answer = await self.call("swg_status")
        assert not answer.is_error, (self.name, answer)
        return answer.structured


ServerStarter = Callable[..., Awaitable[StdioServer]]


def host_config(*, heartbeat_sec: int, max_hold_sec: int) -> LifecycleConfig:
    """The host coordinator's lifecycle: no idle shutdown, a sweep every 0.2 s,
    and grant timers the test can wait out."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.2,
        grant_heartbeat_timeout_sec=heartbeat_sec,
        grant_max_hold_sec=max_hold_sec,
    )


@contextmanager
def host_coordinator(root: Path, config: LifecycleConfig, on_held: str = "displace") -> Iterator[CoherentVolume]:
    """A coordinator for ``root`` bound in this process by a volume built on
    it, which the servers attach to through ``SWG_ROOT``. The volume is closed
    and the coordinator stopped on exit, whatever the block raised."""
    host: CoherentVolume | None = None
    try:
        host = CoherentVolume(root, managed=(MANAGED_GLOB,), config=config, on_held=on_held)
        yield host
    finally:
        try:
            if host is not None:
                with contextlib.suppress(CoherenceError):
                    host.close()
        finally:
            stop_coordinator(root)


def server_parameters(root: Path, on_held: str | None) -> StdioServerParameters:
    """How an agent host starts the server, with nothing inherited beyond the
    SDK's default variables. ``on_held=None`` leaves ``SWG_ON_HELD`` unset."""
    env = {"PYTHONPATH": str(SRC), "SWG_ROOT": str(root), "SWG_MANAGED": MANAGED_GLOB}
    if on_held is not None:
        env["SWG_ON_HELD"] = on_held
    return StdioServerParameters(command=sys.executable, args=["-m", SERVER_MODULE], env=env, cwd=str(root))


def _server_children() -> set[int]:
    """This process's children that run the server module."""
    found = subprocess.run(
        ["pgrep", "-P", str(os.getpid()), "-f", SERVER_MODULE], capture_output=True, text=True, check=False
    )
    return {int(pid) for pid in found.stdout.split()}


async def _start(stack: AsyncExitStack, params: StdioServerParameters, errlog: Path) -> StdioServer:
    """Start one server whose stderr goes to ``errlog``, named by its stem."""
    name = errlog.stem
    before = _server_children()
    stream = stack.enter_context(errlog.open("w"))
    read, write = await stack.enter_async_context(stdio_client(params, errlog=stream))
    session = await stack.enter_async_context(ClientSession(read, write))
    init = await asyncio.wait_for(session.initialize(), timeout=START_TIMEOUT_SEC)
    # This checkout's server text: the process runs this checkout's code, not
    # an installed copy that PYTHONPATH failed to shadow.
    assert init.instructions == INSTRUCTIONS, f"{name} runs code from outside {SRC}"
    started = _server_children() - before
    assert len(started) == 1, f"{name}: expected one new server process, found {started}"
    return StdioServer(name, session, started.pop())


@asynccontextmanager
async def stdio_servers(root: Path, logs: Path) -> AsyncIterator[ServerStarter]:
    """Start servers on ``root`` with ``await start(name, on_held=...)``; each
    one's stderr goes to ``logs / f"{name}.stderr"``. On exit every server is
    stopped (its stdin closed, then terminated if it lingers), and none may be
    left running."""
    started: list[StdioServer] = []
    async with AsyncExitStack() as stack:

        async def start(name: str, *, on_held: str | None = None) -> StdioServer:
            server = await _start(stack, server_parameters(root, on_held), logs / f"{name}.stderr")
            started.append(server)
            return server

        yield start
    left = _server_children() & {server.pid for server in started}
    assert not left, f"server processes still running after their block: {sorted(left)}"


def write_holder(host: CoherentVolume, rel: str) -> str | None:
    """Who the coordinator itself lists as holding ``rel`` for writing, in
    the 32-hex spelling a refusal uses, read from its raw ``/status``; ``None``
    when no session row holds it."""
    status_doc = host.coordinator_status()
    assert isinstance(status_doc, dict) and isinstance(status_doc.get("sessions"), list), status_doc
    holders = [
        UUID(row["agent_id"]).hex for row in status_doc["sessions"]
        if row.get("states", {}).get(rel) in _WRITE_STATES
    ]
    assert len(holders) <= 1, holders
    return holders[0] if holders else None


def reclaims_by_trigger(host: CoherentVolume) -> dict[str, int]:
    """How many write grants the coordinator's sweep has taken back, by trigger."""
    status_doc = host.coordinator_status()
    assert isinstance(status_doc, dict), status_doc
    return dict(status_doc["sweep_reclaims_by_trigger"])


#: How long a poll waits for the state it awaits before the test fails: far
#: inside any grant timer a journey relies on not running out.
POLL_DEADLINE_SEC = 20.0


async def poll_status(
    server: StdioServer, done: Callable[[dict], bool], interval_sec: float
) -> tuple[dict, list[float]]:
    """Poll ``swg_status``, which changes nothing, every ``interval_sec`` until
    ``done`` holds for its answer. Returns that answer and the monotonic time
    each poll started. Fails after :data:`POLL_DEADLINE_SEC` without one."""
    started_at: list[float] = []
    end = time.monotonic() + POLL_DEADLINE_SEC
    while True:
        started_at.append(time.monotonic())
        status = await server.status()
        if done(status):
            return status, started_at
        assert time.monotonic() < end, f"{server.name}: swg_status never reached the awaited state: {status}"
        await asyncio.sleep(interval_sec)
