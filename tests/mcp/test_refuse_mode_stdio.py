# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Refuse mode as agents meet it: real server processes over stdio (#504).

Each journey runs two ``stale-write-guard-fs`` servers, or one beside a
volume, against one coordinator that this test process owns with grant timers
it chooses (``tests/mcp/_stdio_harness.py``). The servers are subprocesses
started as an agent host starts them; every answer is read as a stdio client
reads it, and every file is read back from disk.

Not repeated here, because ``tests/mcp/test_lifecycle.py`` already proves them
by subprocess: both refused starts (an ``SWG_ON_HELD`` value other than
``refuse`` or ``displace``, and ``SWG_ON_HELD=refuse`` against a coordinator
that does not advertise ``pre_edit_if_unheld``). It also checks in process
that a displace-mode write sends only the requests it always sent.

What these do not prove: one machine and loopback only, and the Python
coordinator only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

from ccs.adapters.coherent_volume import CoherentVolume
from tests.mcp._stdio_harness import (
    StdioServer,
    ToolAnswer,
    host_config,
    host_coordinator,
    poll_status,
    reclaims_by_trigger,
    stdio_servers,
    write_holder,
)

X = "data/x.md"
Y = "data/y.md"
#: Grant timers no journey outlasts, so nothing is reclaimed behind its back.
_JOURNEY_TIMERS = host_config(heartbeat_sec=60, max_hold_sec=120)
#: FROZEN copies of the flags each answer carries, never read from the code.
_OTHER_HOLDER = ("other_holder", "wait_then_reacquire", False)
_STALE_VIEW = ("stale_view", "reacquire", True)


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "data").mkdir(parents=True)
    for rel in (X, Y):
        (root / rel).write_text("seed\n")
    return root


def _flags(answer: ToolAnswer) -> tuple[object, object, object]:
    return answer.structured.get("reason"), answer.structured.get("recover"), answer.structured.get("retryable")


def _assert_other_holder(answer: ToolAnswer, holder: str | None) -> None:
    """A refused write: another agent, named as the coordinator names it,
    holds the path; the identical call is not safe to send again."""
    assert holder is not None, "control: the coordinator lists a holder"
    assert answer.is_error, answer
    assert _flags(answer) == _OTHER_HOLDER, answer.structured
    assert answer.structured["holder_agent_id"] == holder
    assert "Do not send this write again as it is" in answer.structured["next_step"]


async def _write_ok(server: StdioServer, rel: str, content: str) -> None:
    """A refuse-mode server's write of a managed path, which asked first."""
    answer = await server.call("swg_write", path=rel, content=content)
    assert answer.structured == {"ok": True, "path": rel, "opted_in": True}, (server.name, answer)


async def _release(server: StdioServer, rel: str) -> None:
    answer = await server.call("swg_release", path=rel)
    assert answer.structured == {"ok": True, "path": rel, "released": True}, (server.name, answer)


async def _rebuild_and_write(server: StdioServer, rel: str, addition: str) -> None:
    """The end of the wait: reacquire, rebuild from its bytes, write once."""
    fresh = await server.call("swg_reacquire", path=rel)
    assert not fresh.is_error, fresh
    await _write_ok(server, rel, fresh.structured["content"] + addition)


def _not_held_by_another(rel: str) -> Callable[[dict], bool]:
    """The wait's end as the next step words it: the path is listed and no
    longer held by another agent."""
    return lambda status: status["per_path"][rel].get("held_by") != "another_agent"


def _held_by(status: dict, rel: str) -> tuple[object, object]:
    entry = status["per_path"][rel]
    return entry.get("held_by"), entry.get("holder_agent_id")


# --- a refused writer waits, then writes on top of the holder's commit ------------


async def _wait_out_the_holder(root: Path, logs: Path, host: CoherentVolume) -> None:
    async with stdio_servers(root, logs) as start:
        a = await start("a", on_held="refuse")
        b = await start("b", on_held="refuse")
        await _write_ok(a, X, "A-v1\n")
        holder = write_holder(host, X)
        assert (await b.call("swg_read", path=X)).structured["content"] == "A-v1\n"

        _assert_other_holder(await b.call("swg_write", path=X, content="A-v1\nB on top\n"), holder)
        assert (root / X).read_text() == "A-v1\n"
        waiting = await b.status()
        assert waiting["held"] == [], "B holds nothing it must release before it waits"
        assert _held_by(waiting, X) == ("another_agent", holder)

        await _write_ok(a, X, "A-v2\n")
        await _release(a, X)
        await poll_status(b, _not_held_by_another(X), 0.2)
        await _rebuild_and_write(b, X, "B on top\n")


def test_a_refused_writer_that_follows_the_wait_writes_its_rebuild_on_top_of_the_holders_commit(
    tmp_path: Path,
) -> None:
    """Two refuse-mode servers. B is refused while A holds the file, waits
    by polling ``swg_status`` while A commits again and releases, then
    reacquires, rebuilds from those bytes and writes once. Prevents a wait
    that ends in B's old content over A's newer commit."""
    root = _workspace(tmp_path)
    with host_coordinator(root, _JOURNEY_TIMERS) as host:
        asyncio.run(_wait_out_the_holder(root, tmp_path, host))

    assert (root / X).read_text() == "A-v2\nB on top\n"


# --- two servers waiting on each other ----------------------------------------------


async def _each_refused_on_the_others_path(a: StdioServer, b: StdioServer, host: CoherentVolume) -> None:
    await _write_ok(a, X, "A wrote x\n")
    await _write_ok(b, Y, "B wrote y\n")
    _assert_other_holder(await a.call("swg_write", path=Y, content="A's y\n"), write_holder(host, Y))
    _assert_other_holder(await b.call("swg_write", path=X, content="B's x\n"), write_holder(host, X))


async def _release_and_finish(root: Path, logs: Path, host: CoherentVolume) -> None:
    async with stdio_servers(root, logs) as start:
        a = await start("a", on_held="refuse")
        b = await start("b", on_held="refuse")
        await _each_refused_on_the_others_path(a, b, host)
        assert (await a.status())["held"] == [X]
        assert (await b.status())["held"] == [Y]

        await _release(a, X)
        b_view, a_view = await b.status(), await a.status()
        assert _held_by(b_view, X) == (None, None), "A's release frees the path B waits on"
        assert _held_by(a_view, Y)[0] == "another_agent", "and only that one: B still holds A's"
        assert a_view["held"] == [] and b_view["held"] == [Y]

        await _release(b, Y)
        await poll_status(a, _not_held_by_another(Y), 0.2)
        await _rebuild_and_write(a, Y, "A on top\n")
        await _rebuild_and_write(b, X, "B on top\n")


def test_two_servers_waiting_on_each_other_both_finish_once_each_has_released(tmp_path: Path) -> None:
    """Two refuse-mode servers each hold one path and are refused on the
    other's. One release frees one direction only; once both have released,
    each rebuilds the other's commit and writes. Prevents two waiters that
    keep their holds and wait out the grant ceiling."""
    root = _workspace(tmp_path)
    with host_coordinator(root, _JOURNEY_TIMERS) as host:
        asyncio.run(_release_and_finish(root, tmp_path, host))

    assert (root / X).read_text() == "A wrote x\nB on top\n"
    assert (root / Y).read_text() == "B wrote y\nA on top\n"


#: The heartbeat timer of the reclaim journey, in the coordinator's whole
#: seconds. A grant is taken back once its holder's last heartbeat is this
#: many ticks old, so a poll at least every ``_HEARTBEAT_SEC - 1`` seconds
#: would keep it if a poll refreshed the heartbeat. Six seconds leave the
#: setup and every poll room on a loaded machine.
_HEARTBEAT_SEC = 6


async def _hold_while_polling(root: Path, logs: Path, host: CoherentVolume) -> list[float]:
    async with stdio_servers(root, logs) as start:
        a = await start("a", on_held="refuse")
        b = await start("b", on_held="refuse")
        await _each_refused_on_the_others_path(a, b, host)
        assert (await a.status())["held"] == [X], "control: A still holds its path as it starts to wait"

        # A waits on Y without releasing X, polling every 0.25 s; B makes no call.
        _, polls = await poll_status(a, _both_taken_back, 0.25)
    return polls


def _both_taken_back(status: dict) -> bool:
    """Neither A's own hold on X nor B's on Y stands any more."""
    return status["held"] == [] and _held_by(status, Y) == (None, None)


def test_a_waiters_status_polls_do_not_keep_the_grant_it_still_holds(tmp_path: Path) -> None:
    """Two refuse-mode servers each hold one path and are refused on the
    other's; neither releases. A polls ``swg_status`` far more often than the
    heartbeat timer and B makes no call at all, and the sweep takes both
    grants back on the heartbeat timer alike: a poll is not a heartbeat.
    Prevents a wait guidance that counts on polling to keep a waiter's holds,
    or to give them up."""
    root = _workspace(tmp_path)
    timers = host_config(heartbeat_sec=_HEARTBEAT_SEC, max_hold_sec=120)
    with host_coordinator(root, timers) as host:
        polls = asyncio.run(_hold_while_polling(root, tmp_path, host))
        reclaims = reclaims_by_trigger(host)

    gaps = [later - earlier for earlier, later in zip(polls, polls[1:])]
    assert gaps and max(gaps) < _HEARTBEAT_SEC - 1, f"control: polls close enough to refresh it: {gaps}"
    assert reclaims == {"reclaim_heartbeat": 2, "reclaim_max_hold": 0}


# --- a writer that never read retries ---------------------------------------------


async def _blind_retry(root: Path, logs: Path, host: CoherentVolume) -> None:
    async with stdio_servers(root, logs) as start:
        a = await start("a", on_held="refuse")
        b = await start("b", on_held="refuse")
        await _write_ok(a, X, "A-v1\n")
        holder = write_holder(host, X)

        for _attempt in range(2):
            _assert_other_holder(await b.call("swg_write", path=X, content="B never read\n"), holder)


def test_a_blind_writer_retrying_after_other_holder_is_told_not_to_retry(tmp_path: Path) -> None:
    """B never read the file and sends its write again after the refusal:
    both answers say ``retryable: false`` and ``wait_then_reacquire``.
    Prevents the row inviting the bare retry measured landing over the
    holder's commit once the path comes free."""
    root = _workspace(tmp_path)
    with host_coordinator(root, _JOURNEY_TIMERS) as host:
        asyncio.run(_blind_retry(root, tmp_path, host))

    assert (root / X).read_text() == "A-v1\n"


# --- refuse mode beside displace mode ---------------------------------------------


async def _displaced(root: Path, logs: Path, host: CoherentVolume) -> None:
    async with stdio_servers(root, logs) as start:
        holder = await start("refusing", on_held="refuse")
        taker = await start("displacing")
        await _write_ok(holder, X, "R-v1\n")
        await _displace_write(taker, X, "R-v1\nD on top\n")
        taker_id = write_holder(host, X)

        stale = await holder.call("swg_write", path=X, content="R-v2\n")
        assert stale.is_error and _flags(stale) == _STALE_VIEW, stale.structured
        assert "rebuild your content from the bytes it returns" in stale.structured["next_step"]
        assert "Never send the content you just sent again" in stale.structured["next_step"]
        assert (root / X).read_text() == "R-v1\nD on top\n"

        after = await holder.status()
        assert after["held"] == []
        assert _held_by(after, X) == ("another_agent", taker_id)


async def _displace_write(server: StdioServer, rel: str, content: str) -> None:
    """A displace-mode server reads and writes ``rel``: today's success answer,
    no ``opted_in``, whoever held the file."""
    assert not (await server.call("swg_read", path=rel)).is_error
    answer = await server.call("swg_write", path=rel, content=content)
    assert (answer.is_error, answer.structured, answer.texts) == (False, {"ok": True, "path": rel}, (f"wrote {rel}",))


def test_a_displace_mode_write_takes_a_refuse_mode_servers_file_and_its_status_names_the_taker(
    tmp_path: Path,
) -> None:
    """A displace-mode server takes a file a refuse-mode server holds: only
    opted-in writers honour a hold. The refuse-mode holder's next write is
    ``stale_view``, whose next step rebuilds from the taker's bytes, and its
    ``swg_status`` names the taker as another agent holding the path."""
    root = _workspace(tmp_path)
    with host_coordinator(root, _JOURNEY_TIMERS) as host:
        asyncio.run(_displaced(root, tmp_path, host))


async def _take_from_a_volume(root: Path, logs: Path, host: CoherentVolume) -> str | None:
    async with stdio_servers(root, logs) as start:
        taker = await start("displacing")
        await _displace_write(taker, X, "taken\n")
        return write_holder(host, X)


def test_a_displace_mode_write_of_a_path_a_refuse_mode_volume_holds_takes_it_as_today(tmp_path: Path) -> None:
    """A refuse-mode volume holds the file; a server started without
    ``SWG_ON_HELD`` writes it, and the write takes the file with exactly
    today's answer. Prevents refuse mode on one writer changing what a
    displace-mode server does or says."""
    root = _workspace(tmp_path)
    with host_coordinator(root, _JOURNEY_TIMERS, on_held="refuse") as host:
        host.write(X, b"held by the volume\n")
        assert write_holder(host, X) == UUID(host.agent_id).hex, "control: the volume holds the file"
        taker = asyncio.run(_take_from_a_volume(root, tmp_path, host))

        assert taker not in (None, UUID(host.agent_id).hex)
    assert (root / X).read_text() == "taken\n"
