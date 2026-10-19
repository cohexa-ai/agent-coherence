# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""U3 — cross-process coordinator-layer assertions (guarantee-ladder plan).

The coordinator is the serialization point under test, so the only honest
cross-process shape is: one coordinator process (for most cases here: the
parent, serving on a daemon thread with test-scale sweep timings — the shipped
CLI entrypoint hard-codes a 600 s heartbeat window, so a test-owned config is
the sanctioned route per the plan's stop condition (c)), and contender CLIENTS
as separate spawn-context OS processes driving the shipped client path
(`resolve_endpoint` + `post` from `ccs.cli._coherence_client`).

The SIGKILL case is the exception: it needs no sweep timings and must kill the
coordinator without killing pytest, so it launches the coordinator as its own
process through the shipped `agent-coherence-coordinator` entry point and
SIGKILLs the detached worker that entry point leaves running.

Precedence note (pins shipped semantics, sharpening the plan's literal
scenario text): when a peer HAS committed, the version leg fires first and a
zombie sees `version_mismatch`; `stale_read_generation` is observable exactly
when the version is UNMOVED — the case version-CAS structurally cannot see,
which is the fence's whole job. Both variants are asserted.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import pytest

import ccs
from ccs.adapters import coherent_volume
from ccs.adapters.claude_code import lifecycle
from ccs.adapters.claude_code.coordinator_server import session_to_agent_id
from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.adapters.substrate import (
    CoordinatedSubstrate,
    SubstrateCoordinatorSession,
    _write_strict_policy,
)
from ccs.cli._coherence_client import (
    CoordinatorEndpoint,
    CoordinatorUnavailable,
    get,
    post,
    resolve_endpoint,
)
from ccs.core.substrate import sha256_hex
from ccs.testing.process_harness import (
    ContenderSpec,
    ParkContext,
    ProcessRaceHarness,
    run_in_subprocess,
    run_park_after_ack,
)
from ccs.testing.substrate_conformance import ConformanceSubstrate, InMemoryStore

# Spawns real processes that race each other: run with -n 0 (tests/conftest.py).
pytestmark = pytest.mark.serial

_PATH = "CLAUDE.md"  # in DEFAULT_TRACKED_PATTERNS — coordinated without extra policy files
_SIBLING_PATH = "AGENTS.md"  # also in DEFAULT_TRACKED_PATTERNS


# ---------------------------------------------------------------------------
# Child entry points (top-level; spawn children unpickle by qualname).
# Each child resolves the endpoint from the workspace and drives the shipped
# client path. Return values are plain dicts (picklable).
# ---------------------------------------------------------------------------


def _version_of(pre_read_body: dict) -> int:
    """Extract the artifact version from either pre-read shape: the fresh
    branch carries top-level ``version``; a first observation of an existing
    artifact returns the stale-warning shape with ``summary.current_version``."""
    if "version" in pre_read_body:
        return pre_read_body["version"]
    return pre_read_body.get("summary", {}).get("current_version", 0)


def _child_pre_read(ctx, workspace_str: str, session_id: str) -> dict:
    from ccs.cli._coherence_client import post, resolve_endpoint

    ctx.barrier_wait()
    endpoint = resolve_endpoint(Path(workspace_str))
    return post(endpoint, "/hooks/pre-read", {"session_id": session_id, "path": _PATH})


def _child_pre_edit(ctx, workspace_str: str, session_id: str) -> dict:
    from ccs.cli._coherence_client import post, resolve_endpoint

    ctx.barrier_wait()
    endpoint = resolve_endpoint(Path(workspace_str))
    return post(endpoint, "/hooks/pre-edit", {"session_id": session_id, "path": _PATH})


def _child_commit_cas(
    ctx, workspace_str: str, session_id: str, expected_version: int, payload: str
) -> dict:
    from ccs.cli._coherence_client import post, resolve_endpoint

    ctx.barrier_wait()
    ctx.delay()
    endpoint = resolve_endpoint(Path(workspace_str))
    return post(
        endpoint,
        "/hooks/post-edit-cas",
        {
            "session_id": session_id,
            "path": _PATH,
            "content_hash": sha256_hex(payload.encode()),
            "expected_version": expected_version,
        },
    )


def _child_read_then_race_commit(ctx, workspace_str: str, session_id: str, payload: str) -> dict:
    """Pre-read (own session), rendezvous so every contender holds the same
    version, then race the OCC commit."""
    from ccs.cli._coherence_client import post, resolve_endpoint

    endpoint = resolve_endpoint(Path(workspace_str))
    pre = post(endpoint, "/hooks/pre-read", {"session_id": session_id, "path": _PATH})
    version = _version_of(pre)
    ctx.barrier_wait()  # everyone has read; race the commits
    ctx.delay()
    result = post(
        endpoint,
        "/hooks/post-edit-cas",
        {
            "session_id": session_id,
            "path": _PATH,
            "content_hash": sha256_hex(payload.encode()),
            "expected_version": version,
        },
    )
    result["_pre_read_version"] = version
    return result


def _child_commit_expect_unavailable(ctx, workspace_str: str, session_id: str) -> str:
    from ccs.cli._coherence_client import CoordinatorUnavailable, post, resolve_endpoint

    ctx.barrier_wait()
    try:
        endpoint = resolve_endpoint(Path(workspace_str))
        post(
            endpoint,
            "/hooks/post-edit-cas",
            {
                "session_id": session_id,
                "path": _PATH,
                "content_hash": sha256_hex(b"x"),
                "expected_version": 1,
            },
        )
    except CoordinatorUnavailable:
        return "typed-unavailable"
    except FileNotFoundError:
        return "typed-unavailable"  # endpoint files already torn down — still typed, still closed
    return "fabricated-success"


# ---------------------------------------------------------------------------
# Parked-live-holder children (run under run_park_after_ack). The holder is
# LIVE: a background thread heartbeats its grant until the parent resumes it,
# so only the stop, or the holder ending its own beats, can let the heartbeat
# the sweep reads go stale.
# ---------------------------------------------------------------------------

_BEAT_INTERVAL_SEC = 0.4
_LIVE_BEATS_WAIT_SEC = 15.0


class _GrantHeartbeat:
    """The holder's own grant heartbeat: a ``verify_only`` pre-read of the held
    path every ``_BEAT_INTERVAL_SEC``, until the parent resumes the holder or
    the holder ends it.

    ``verify_only`` refreshes the agent's heartbeat and changes nothing else:
    the holder stays EXCLUSIVE, and after a reclaim a beat neither re-grants
    the path nor pops the sweep's notice, so the lease refusal is always the
    reclaimed arm.
    """

    def __init__(self, ctx: ParkContext, endpoint: CoordinatorEndpoint, session_id: str) -> None:
        self._ctx = ctx
        self._endpoint = endpoint
        self._body = {"session_id": session_id, "path": _PATH, "verify_only": True}
        self._ended = threading.Event()
        self.beat_seconds: list[int] = []
        self.live_errors: list[str] = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not (self._ctx.resumed or self._ended.is_set()):
            sent_second = int(time.time())
            try:
                post(self._endpoint, "/hooks/pre-read", self._body)
            except (CoordinatorUnavailable, urllib.error.HTTPError) as exc:
                # A beat in flight across the stop may fail once resumed: a
                # straggler, ignored. A failure while live is the holder's own.
                if not self._ctx.resumed:
                    self.live_errors.append(repr(exc))
            else:
                self.beat_seconds.append(sent_second)
            self._ended.wait(_BEAT_INTERVAL_SEC)

    def await_live_beats(self, granted_by_second: int) -> int:
        """Return the beat count once two beats have landed, the latest sent in
        a later whole second than the grant: /status keeps whole unix seconds,
        so only then can it show a heartbeat newer than the grant."""
        deadline = time.monotonic() + _LIVE_BEATS_WAIT_SEC
        while time.monotonic() < deadline:
            self._raise_on_live_error()
            beats = list(self.beat_seconds)
            if len(beats) >= 2 and beats[-1] > granted_by_second:
                return len(beats)
            time.sleep(0.05)
        raise RuntimeError(f"no two beats landed past second {granted_by_second}: {self.beat_seconds}")

    def end(self) -> None:
        self._ended.set()
        self.join()

    def join(self) -> None:
        self._thread.join(timeout=10.0)
        if self._thread.is_alive():
            raise RuntimeError("the holder's heartbeat thread did not stop")
        self._raise_on_live_error()

    def _raise_on_live_error(self) -> None:
        if self.live_errors:
            raise RuntimeError(f"the holder's heartbeat failed while live: {self.live_errors}")


def _acquire_and_beat(
    ctx: ParkContext, workspace_str: str, session_id: str
) -> tuple[CoordinatorEndpoint, int, _GrantHeartbeat, dict[str, Any]]:
    """Read the held path, take it EXCLUSIVE through pre-edit, and heartbeat
    until two beats past the grant have landed. Also returns the ack facts."""
    endpoint = resolve_endpoint(Path(workspace_str))
    version = _version_of(post(endpoint, "/hooks/pre-read", {"session_id": session_id, "path": _PATH}))
    post(endpoint, "/hooks/pre-edit", {"session_id": session_id, "path": _PATH})
    # The coordinator stamps the grant no later than this second (one clock).
    granted_by_second = int(time.time())
    beat = _GrantHeartbeat(ctx, endpoint, session_id)
    beats = beat.await_live_beats(granted_by_second)
    return endpoint, version, beat, {"version": version, "pid": os.getpid(), "beats_at_ack": beats}


def _child_parked_holder(ctx: ParkContext, workspace_str: str, session_id: str) -> dict[str, Any]:
    """Heartbeat until the parent stops this process at the ack; once resumed,
    commit once through each route as the same process and session."""
    endpoint, version, beat, facts = _acquire_and_beat(ctx, workspace_str, session_id)
    ctx.ack(facts)
    beat.join()  # the resume flag gates the beats off before the commits
    content_hash = sha256_hex(b"parked holder edit")
    lease = post(
        endpoint,
        "/hooks/post-edit",
        {"session_id": session_id, "path": _PATH, "content_hash": content_hash, "success": True},
    )
    cas = post(
        endpoint,
        "/hooks/post-edit-cas",
        {"session_id": session_id, "path": _PATH, "content_hash": content_hash, "expected_version": version},
    )
    return {"lease": lease, "cas": cas}


def _child_beating_holder(ctx: ParkContext, workspace_str: str, session_id: str) -> dict[str, Any]:
    """Control: keep heartbeating through the parent's window until resumed,
    then return without committing."""
    _endpoint, _version, beat, facts = _acquire_and_beat(ctx, workspace_str, session_id)
    ctx.ack(facts)
    beat.join()
    return {"beats": len(beat.beat_seconds)}


def _child_holder_ending_beats_at_ack(ctx: ParkContext, workspace_str: str, session_id: str) -> dict[str, Any]:
    """Control: end the heartbeat at the ack and wait there, running, until
    resumed; then return without committing."""
    _endpoint, _version, beat, facts = _acquire_and_beat(ctx, workspace_str, session_id)
    beat.end()
    ctx.ack(facts)
    return {"beats": len(beat.beat_seconds)}


# ---------------------------------------------------------------------------
# Coordinator fixture: parent-hosted, test-scale timings (plan U3 approach).
# ---------------------------------------------------------------------------


def _spawn(
    tmp_path: Path,
    *,
    sweep_interval_sec: float,
    heartbeat_timeout_sec: int,
    max_hold_sec: int | None = None,
) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / _PATH).write_text("seed\n")
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=sweep_interval_sec,
        grant_heartbeat_timeout_sec=heartbeat_timeout_sec,
        grant_max_hold_sec=heartbeat_timeout_sec * 5 if max_hold_sec is None else max_hold_sec,
        port_file_retry_attempts=10,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
        spawn_self_probe_attempts=30,
    )
    port = ensure_coordinator(tmp_path, config=cfg)
    assert port > 0
    return tmp_path


@pytest.fixture
def coordinator_workspace(tmp_path: Path):
    """Fast-sweep coordinator: for the zombie/reclaim scenarios."""
    yield _spawn(tmp_path, sweep_interval_sec=0.2, heartbeat_timeout_sec=1)
    stop_coordinator(tmp_path)


@pytest.fixture
def coordinator_workspace_no_sweep(tmp_path: Path):
    """Sweep disabled: for scenarios where a live grant must NOT be reclaimed
    mid-test. A spawn-context child pays ~2 s importing ccs, so any scenario
    with more than one sequential child outlives a 1 s heartbeat window — the
    other_holder case needs the grant to survive, so the sweep stays off."""
    yield _spawn(tmp_path, sweep_interval_sec=0, heartbeat_timeout_sec=600)
    stop_coordinator(tmp_path)


_PARK_HEARTBEAT_TIMEOUT_SEC = 3
_PARK_SWEEP_INTERVAL_SEC = 0.2
# Far past any run, so the heartbeat is the only trigger the sweep can fire.
_PARK_MAX_HOLD_SEC = 60


@pytest.fixture
def parked_holder_workspace(tmp_path: Path):
    """A coordinator per run for the parked-holder tests. The 1 s window of
    ``coordinator_workspace`` reclaims a live holder: the sweep compares whole
    seconds, so the first second boundary after a beat reads as a full window.
    At 3 s a holder beating every 0.4 s keeps its grant, and the 0.2 s sweep
    still reclaims a stopped one within seconds."""
    (tmp_path / _SIBLING_PATH).write_text("seed\n")
    yield _spawn(
        tmp_path,
        sweep_interval_sec=_PARK_SWEEP_INTERVAL_SEC,
        heartbeat_timeout_sec=_PARK_HEARTBEAT_TIMEOUT_SEC,
        max_hold_sec=_PARK_MAX_HOLD_SEC,
    )
    stop_coordinator(tmp_path)


def _sid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def test_two_client_processes_racing_commits_exactly_one_wins(
    coordinator_workspace_no_sweep: Path,
) -> None:
    coordinator_workspace = coordinator_workspace_no_sweep
    harness = ProcessRaceHarness(timeout_sec=60.0)
    result = harness.race(
        [
            ContenderSpec(
                _child_read_then_race_commit,
                args=(str(coordinator_workspace), _sid(), "A"),
            ),
            ContenderSpec(
                _child_read_then_race_commit,
                args=(str(coordinator_workspace), _sid(), "B"),
                delay_seconds=0.05,
            ),
        ]
    )
    for o in result.outcomes:
        if o.error is not None:
            raise o.error
    bodies = [o.value for o in result.outcomes]
    winners = [b for b in bodies if b.get("ok")]
    losers = [b for b in bodies if not b.get("ok")]
    assert len(winners) == 1 and len(losers) == 1, bodies
    assert losers[0]["reason"] == "version_mismatch"
    assert losers[0]["current_version"] == winners[0]["version"]


def test_pessimistic_holder_blocks_occ_committer_with_other_holder(
    coordinator_workspace_no_sweep: Path,
) -> None:
    ws = str(coordinator_workspace_no_sweep)
    holder_sid, occ_sid = _sid(), _sid()
    # Holder acquires EXCLUSIVE in its own OS process (then exits WITHOUT
    # session-stop — the grant persists; heartbeats stop).
    pre = run_in_subprocess(ContenderSpec(_child_pre_read, args=(ws, holder_sid)))
    run_in_subprocess(ContenderSpec(_child_pre_edit, args=(ws, holder_sid)))
    # OCC committer in a second process, immediately (holder heartbeat fresh).
    occ_pre = run_in_subprocess(ContenderSpec(_child_pre_read, args=(ws, occ_sid)))
    body = run_in_subprocess(
        ContenderSpec(_child_commit_cas, args=(ws, occ_sid, _version_of(occ_pre), "occ"))
    )
    assert body.get("ok") is False, body
    assert body["reason"] == "other_holder", body
    assert _version_of(pre) == _version_of(occ_pre)


def test_zombie_fence_version_unmoved_stale_read_generation(coordinator_workspace: Path) -> None:
    """The crown assertion: reclaim bumps the generation while the version
    stands still; the zombie's commit is rejected by the fence — the case
    version-CAS alone would have ADMITTED."""
    ws = str(coordinator_workspace)
    zombie_sid = _sid()
    pre = run_in_subprocess(ContenderSpec(_child_pre_read, args=(ws, zombie_sid)))
    version = _version_of(pre)
    run_in_subprocess(ContenderSpec(_child_pre_edit, args=(ws, zombie_sid)))
    # The holder process is gone (no heartbeats). Poll the zombie's late
    # commit until the sweep has reclaimed: while still M/E the commit is the
    # typed caller-in-M/E precondition; after reclaim it must be the FENCE.
    deadline = time.monotonic() + 15.0
    body: dict = {}
    while time.monotonic() < deadline:
        body = run_in_subprocess(
            ContenderSpec(_child_commit_cas, args=(ws, zombie_sid, version, "zombie"))
        )
        if body.get("ok") is False and body.get("reason") == "stale_read_generation":
            break
        time.sleep(0.3)
    assert body.get("reason") == "stale_read_generation", (
        f"expected the read-generation fence to reject the reclaimed zombie; got {body}"
    )
    # Version never moved — version-CAS alone could not have seen this.
    assert body.get("current_version") == version, body


def test_zombie_after_peer_commit_sees_version_mismatch(coordinator_workspace: Path) -> None:
    """Precedence pin (RD-5 overlap): when a peer HAS committed after the
    reclaim, the version leg fires first — the zombie sees version_mismatch,
    not the fence reason."""
    ws = str(coordinator_workspace)
    zombie_sid, peer_sid = _sid(), _sid()
    pre = run_in_subprocess(ContenderSpec(_child_pre_read, args=(ws, zombie_sid)))
    version = _version_of(pre)
    run_in_subprocess(ContenderSpec(_child_pre_edit, args=(ws, zombie_sid)))
    # Wait out the reclaim (poll via the zombie's typed outcomes).
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        probe = run_in_subprocess(
            ContenderSpec(_child_commit_cas, args=(ws, zombie_sid, version, "probe"))
        )
        if probe.get("reason") == "stale_read_generation":
            break
        time.sleep(0.3)
    # Peer commits (version moves).
    peer_pre = run_in_subprocess(ContenderSpec(_child_pre_read, args=(ws, peer_sid)))
    peer = run_in_subprocess(
        ContenderSpec(_child_commit_cas, args=(ws, peer_sid, _version_of(peer_pre), "peer"))
    )
    assert peer.get("ok") is True, peer
    # The zombie's stale expected_version now loses on the VERSION leg.
    body = run_in_subprocess(
        ContenderSpec(_child_commit_cas, args=(ws, zombie_sid, version, "zombie"))
    )
    assert body.get("ok") is False and body["reason"] == "version_mismatch", body


def test_coordinator_killed_mid_run_yields_typed_unavailable(
    coordinator_workspace_no_sweep: Path,
) -> None:
    coordinator_workspace = coordinator_workspace_no_sweep
    ws = str(coordinator_workspace)
    sid = _sid()
    run_in_subprocess(ContenderSpec(_child_pre_read, args=(ws, sid)))
    stop_coordinator(coordinator_workspace)
    verdict = run_in_subprocess(ContenderSpec(_child_commit_expect_unavailable, args=(ws, sid)))
    assert verdict == "typed-unavailable"


# ---------------------------------------------------------------------------
# A coordinator process SIGKILLed under a child holding both strict surfaces.
# The coordinator is its own process here, launched through the shipped entry
# point; the parent signals only the worker that launch left running.
# ---------------------------------------------------------------------------

# One glob for both surfaces: the policy is written before the launch (it loads
# once), and a strict volume refuses a coordinator that does not enforce it.
_KILL_MANAGED = ("data/**",)
_KILL_FILE = "data/notes.md"
_KILL_REF = "data/row.bin"
_ENTRY_POINT = "agent-coherence-coordinator"
_LAUNCH_TIMEOUT_SEC = 30.0
_PORT_REFUSED_DEADLINE_SEC = 15.0
_PORT_REFUSED_POLL_SEC = 0.05
_KILL_CHILD_TIMEOUT_SEC = 60.0
_UNREACHABLE = "ccs.core.exceptions.CoordinatorUnreachable"
_SPAWN_CALLS = ("volume_connect_or_spawn", "lifecycle_connect_or_spawn", "ensure_coordinator")


class _CallCounter:
    """Counts the calls made through the functions it wraps (child only)."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def wrap(self, owner: Any, name: str, key: str) -> None:
        original = getattr(owner, name)
        self.counts[key] = 0

        def counted(*args: Any, **kwargs: Any) -> Any:
            self.counts[key] += 1
            return original(*args, **kwargs)

        setattr(owner, name, counted)

    def reset(self) -> None:
        self.counts = dict.fromkeys(self.counts, 0)

    def snapshot(self) -> dict[str, int]:
        return dict(self.counts)


def _arm_call_counters() -> _CallCounter:
    """Wrap, before anything is constructed, the call every coordinator request
    opens its connection through (http.client uses socket.create_connection;
    the attach probe's raw connect is not counted) and every spawn route."""
    counter = _CallCounter()
    counter.wrap(socket, "create_connection", "connections")
    counter.wrap(coherent_volume, "connect_or_spawn", "volume_connect_or_spawn")
    counter.wrap(lifecycle, "connect_or_spawn", "lifecycle_connect_or_spawn")
    counter.wrap(lifecycle, "ensure_coordinator", "ensure_coordinator")
    return counter


def _attach_strict_surfaces(root: Path) -> tuple[CoherentVolume, CoordinatedSubstrate, InMemoryStore]:
    volume = CoherentVolume(root, managed=_KILL_MANAGED, on_error="strict")
    store = InMemoryStore()
    store.set(_KILL_REF, b"row seed")
    session = SubstrateCoordinatorSession(root, managed=_KILL_MANAGED)
    return volume, CoordinatedSubstrate(ConformanceSubstrate(store), session), store


def _observe_op(
    op: Callable[[], Any], counter: _CallCounter, state: Callable[[], tuple[str, str]]
) -> dict[str, Any]:
    """Run one operation on fresh counters. The raise is reported by its
    qualified class name (an exception object need not pickle), beside the
    calls the operation made and the file hash and store token around it."""
    counter.reset()
    before = state()
    raised = message = None
    try:
        op()
    except Exception as exc:  # noqa: BLE001 - every outcome is reported, by name
        raised = f"{type(exc).__module__}.{type(exc).__qualname__}"
        message = str(exc)
    calls = counter.snapshot()
    return {"raised": raised, "message": message, "calls": calls, "before": before, "after": state()}


def _child_strict_surfaces_across_kill(ctx: ParkContext, workspace_str: str) -> dict[str, Any]:
    """Attach a strict volume and a substrate session, read once through each
    and ack. Once resumed (the parent has killed the coordinator), read and
    commit once through each surface and report each outcome."""
    counter = _arm_call_counters()
    root = Path(workspace_str)
    volume, substrate, store = _attach_strict_surfaces(root)
    volume.read(_KILL_FILE)
    _seen, token = substrate.read(_KILL_REF)
    ctx.ack({"attach_calls": counter.snapshot()})

    def state() -> tuple[str, str]:
        return sha256_hex((root / _KILL_FILE).read_bytes()), store.get(_KILL_REF)[1]

    ops: dict[str, Callable[[], Any]] = {
        "volume read": lambda: volume.read(_KILL_FILE),
        "volume write": lambda: volume.write(_KILL_FILE, b"written after the kill\n"),
        "substrate read": lambda: substrate.read(_KILL_REF),
        "substrate commit": lambda: substrate.commit(
            _KILL_REF, expected_token=token, new_bytes=b"row written after the kill"
        ),
    }
    return {name: _observe_op(op, counter, state) for name, op in ops.items()}


@dataclass(frozen=True)
class _LaunchedCoordinator:
    root: Path
    pid: int
    port: int
    pid_file_bytes: bytes


def _entry_point() -> str:
    """The shipped console script beside this interpreter, else on PATH. A
    missing script fails the test: skipping would report a kill never run."""
    beside = Path(sys.executable).parent / _ENTRY_POINT
    if beside.is_file():
        return str(beside)
    found = shutil.which(_ENTRY_POINT)
    if found is None:
        pytest.fail(f"{_ENTRY_POINT} is neither beside {sys.executable} nor on PATH")
    return found


def _tree_env() -> dict[str, str]:
    """The launch env, with this tree's ``src`` first on PYTHONPATH so the
    detached worker runs this tree's code, not another checkout's."""
    tree_src = str(Path(ccs.__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [tree_src, existing]))}


def _pid_file(root: Path) -> Path:
    return root / ".coherence" / "server.pid"


def _launch_coordinator(root: Path) -> _LaunchedCoordinator:
    launch = subprocess.run(
        [_entry_point(), "--root", str(root), "--quiet"],
        env=_tree_env(), capture_output=True, text=True, timeout=_LAUNCH_TIMEOUT_SEC, check=False,
    )
    assert launch.returncode == 0, f"{_ENTRY_POINT} failed ({launch.returncode}): {launch.stderr}"
    pid_file_bytes = _pid_file(root).read_bytes()
    pid_line, port_line = pid_file_bytes.decode().splitlines()[:2]
    return _LaunchedCoordinator(root, int(pid_line), int(port_line), pid_file_bytes)


def _command_line(pid: int) -> str:
    return subprocess.run(
        ["ps", "-ww", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()


def _is_own_worker(pid: int, root: Path) -> bool:
    """Whether ``pid`` is, right now, the detached worker launched for ``root``:
    never pytest itself (in-process fixtures write ``os.getpid()`` to
    server.pid) and never a process that reused the pid."""
    tokens = _command_line(pid).split()
    return pid != os.getpid() and "--_daemonized" in tokens and str(root) in tokens


def _await_port_refused(port: int) -> float:
    """Seconds until a connect to ``port`` is refused, polled against a
    wall-clock deadline."""
    started = time.monotonic()
    while time.monotonic() - started < _PORT_REFUSED_DEADLINE_SEC:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.25).close()
        except ConnectionRefusedError:
            return time.monotonic() - started
        except OSError:
            pass  # not a refusal (a timeout, say): keep polling
        time.sleep(_PORT_REFUSED_POLL_SEC)
    raise AssertionError(f"port {port} still accepted connections {_PORT_REFUSED_DEADLINE_SEC}s after SIGKILL")


def _kill_worker(launched: _LaunchedCoordinator, _acked: dict[str, Any]) -> dict[str, Any]:
    """The parent's window: snapshot server.pid, SIGKILL the worker it names
    once its command line proves it is ours, and wait until its port refuses."""
    pid_file_bytes = _pid_file(launched.root).read_bytes()
    assert pid_file_bytes == launched.pid_file_bytes, (pid_file_bytes, launched.pid_file_bytes)
    command = _command_line(launched.pid)
    assert _is_own_worker(launched.pid, launched.root), f"pid {launched.pid} is not our worker: {command!r}"
    os.kill(launched.pid, signal.SIGKILL)
    return {"pid_file_bytes": pid_file_bytes, "refused_after_sec": _await_port_refused(launched.port)}


def _signal_own_worker(root: Path, launched_pid: int | None) -> None:
    """Teardown: SIGKILL the launched worker only if it is still alive and its
    command line still names it, by the pid recorded at launch or, when the
    launch did not get that far, the pid server.pid names now."""
    try:
        recorded = int(_pid_file(root).read_text().splitlines()[0])
    except (OSError, ValueError, IndexError):
        recorded = None
    for pid in {launched_pid, recorded} - {None}:
        if _is_own_worker(pid, root):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.fixture
def launched_coordinator(tmp_path: Path) -> Iterator[_LaunchedCoordinator]:
    """A strict workspace whose coordinator runs as its own process, launched
    through the shipped entry point."""
    root = tmp_path.resolve()
    (root / ".git").mkdir()
    (root / _KILL_FILE).parent.mkdir(parents=True)
    (root / _KILL_FILE).write_text("seed\n")
    _write_strict_policy(root / ".coherence", _KILL_MANAGED)
    launched: _LaunchedCoordinator | None = None
    try:
        launched = _launch_coordinator(root)
        yield launched
    finally:
        _signal_own_worker(root, launched.pid if launched is not None else None)


def test_sigkilled_coordinator_refuses_both_strict_surfaces_typed(
    launched_coordinator: _LaunchedCoordinator,
) -> None:
    """A coordinator launched through the shipped entry point is SIGKILLed while
    a child holds a strict volume and a substrate session, each read once. Once
    the port refuses, the child's volume read, volume write, substrate read and
    substrate commit each raise ``CoordinatorUnreachable`` after exactly one
    connection attempt and no spawn or ensure call; the file, the substrate
    token and server.pid are as they were before the kill.

    What this does not prove: one machine and loopback only; no network
    partition; the hook client still fails open; attach-leg respawn is
    unchanged, so a fresh attach to a dead recorded port still respawns; Node
    is untouched.
    """
    launched = launched_coordinator
    result = run_park_after_ack(
        ContenderSpec(_child_strict_surfaces_across_kill, args=(str(launched.root),)),
        partial(_kill_worker, launched),
        stop=False,
        timeout_sec=_KILL_CHILD_TIMEOUT_SEC,
    )

    # The counters were live: attach connected and went through both spawn routes.
    attach = result.acked["attach_calls"]
    assert attach["connections"] >= 1, attach
    assert attach["volume_connect_or_spawn"] == attach["lifecycle_connect_or_spawn"] == 1, attach
    assert result.window["refused_after_sec"] < _PORT_REFUSED_DEADLINE_SEC, result.window

    assert list(result.value) == ["volume read", "volume write", "substrate read", "substrate commit"]
    for name, op in result.value.items():
        assert op["raised"] == _UNREACHABLE, (name, op)
        assert op["calls"]["connections"] == 1, (name, op)
        assert all(op["calls"][key] == 0 for key in _SPAWN_CALLS), (name, op)
        assert op["after"] == op["before"], (name, op)
    assert _pid_file(launched.root).read_bytes() == result.window["pid_file_bytes"]


# ---------------------------------------------------------------------------
# Parked live holder: out-of-band witnesses. The parent reads
# /status?detail=full while the holder is stopped (or, in the controls,
# running) and never acts for the holder.
# ---------------------------------------------------------------------------

_OPERATOR_HEADERS = {"Coherence-Local-Operator": "true"}
_STATUS_POLL_SEC = 0.1
_RECLAIM_WAIT_SEC = 15.0
# Two heartbeat windows plus sweeps: longer than the parked test's stop lasts.
_CONTROL_WINDOW_SEC = 2 * _PARK_HEARTBEAT_TIMEOUT_SEC + 3 * _PARK_SWEEP_INTERVAL_SEC


def _read_operator_status(endpoint: CoordinatorEndpoint) -> dict[str, Any]:
    return get(endpoint, "/status?detail=full", extra_headers=_OPERATOR_HEADERS)


def _readable(body: dict[str, Any]) -> bool:
    """A degraded answer carries null registry lists. It cannot tell held from
    not held, so a witness reads again rather than taking it for either."""
    return (
        not body.get("degraded")
        and body.get("sessions") is not None
        and body.get("tracked_artifacts") is not None
    )


def _poll_status(
    read: Callable[[], dict[str, Any]],
    decide: Callable[[dict[str, Any]], Any],
    *,
    timeout_sec: float,
    what: str,
) -> Any:
    """Read /status until ``decide`` answers (not None) on a readable body."""
    deadline = time.monotonic() + timeout_sec
    while True:
        body = read()
        if _readable(body):
            verdict = decide(body)
            if verdict is not None:
                return verdict
        if time.monotonic() >= deadline:
            raise AssertionError(f"could not witness {what} within {timeout_sec}s; last /status body: {body}")
        time.sleep(_STATUS_POLL_SEC)


def _session_row(body: dict[str, Any], session_id: str) -> dict[str, Any] | None:
    agent_id = str(session_to_agent_id(session_id))
    return next((row for row in body["sessions"] if row["agent_id"] == agent_id), None)


def _artifact_entry(body: dict[str, Any], path: str) -> dict[str, Any]:
    return next(entry for entry in body["tracked_artifacts"] if entry["path"] == path)


def _held_witness(body: dict[str, Any], *, session_id: str) -> dict[str, Any]:
    """The holder's grant, still EXCLUSIVE, with the times the sweep reads. A
    reclaim that landed first means the grant could not be witnessed held."""
    row = _session_row(body, session_id)
    if row is not None and _PATH in row["reclaimed"]:
        raise AssertionError(f"could not witness the held grant: the sweep reclaimed it first: {row}")
    assert row is not None and row["states"].get(_PATH) == "EXCLUSIVE", f"holder does not hold {_PATH}: {row}"
    artifact = _artifact_entry(body, _PATH)
    return {
        "granted_at": row["grants"][_PATH]["granted_at_unix_ts"],
        "last_heartbeat": row["last_heartbeat_unix_ts"],
        "version": artifact["version"],
        "owner_generation": artifact["owner_generation"],
    }


def _reclaim_witness(body: dict[str, Any], *, session_id: str) -> dict[str, Any] | None:
    """The sweep's reclaim of the holder's grant, once /status lists it."""
    row = _session_row(body, session_id)
    slot = row["reclaimed"].get(_PATH) if row is not None else None
    if slot is None:
        return None
    artifact = _artifact_entry(body, _PATH)
    return {
        "trigger": slot["trigger"],
        "max_hold_reclaims": body["sweep_reclaims_by_trigger"]["reclaim_max_hold"],
        "version": artifact["version"],
        "owner_generation": artifact["owner_generation"],
    }


def _assert_stopped(pid: int) -> None:
    # WNOWAIT leaves the stop report in place for the harness's own wait.
    report = os.waitid(os.P_PID, pid, os.WSTOPPED | os.WNOHANG | os.WNOWAIT)
    assert report is not None and report.si_code == os.CLD_STOPPED, f"holder {pid} is not stopped: {report}"


def _assert_lost_token(refusal: dict[str, Any]) -> None:
    """The refusal names the generation the claim was taken under, below the
    one that owns the path now."""
    assert {"captured_generation", "owner_generation"} <= refusal.keys(), refusal
    assert refusal["captured_generation"] < refusal["owner_generation"], refusal


def _assert_still_held(body: dict[str, Any], session_id: str) -> None:
    row = _session_row(body, session_id)
    assert row is not None and row["states"].get(_PATH) == "EXCLUSIVE", f"beating holder lost {_PATH}: {row}"
    assert _PATH not in row["reclaimed"], f"the sweep reclaimed a beating holder: {row}"


def _sibling_ordering(body: dict[str, Any], holder_sid: str, sibling_sid: str) -> dict[str, Any] | None:
    """The sibling's reclaim beside the holder's grant time, both from ``body``."""
    sibling = _session_row(body, sibling_sid)
    slot = sibling["reclaimed"].get(_SIBLING_PATH) if sibling is not None else None
    if slot is None:
        return None
    holder = _session_row(body, holder_sid)
    assert holder is not None, body
    return {
        "sibling_trigger": slot["trigger"],
        "sibling_reclaimed_at": slot["reclaimed_at_unix_ts"],
        "holder_granted_at": holder["grants"][_PATH]["granted_at_unix_ts"],
        "heartbeat_timeout": body["grant_heartbeat_timeout_sec"],
    }


def _watch_beating_holder(endpoint: CoordinatorEndpoint, holder_sid: str) -> dict[str, Any] | None:
    """Over the control window, the beating holder keeps its grant on every
    readable read, while a second session takes a grant after the holder's and
    never beats again. Returns the ordering from the latest body that lists the
    sibling's reclaim, or None when none did."""
    sibling_sid = _sid()
    post(endpoint, "/hooks/pre-read", {"session_id": sibling_sid, "path": _SIBLING_PATH})
    post(endpoint, "/hooks/pre-edit", {"session_id": sibling_sid, "path": _SIBLING_PATH})
    window_end = time.monotonic() + _CONTROL_WINDOW_SEC
    ordering = None
    while True:
        read_at = time.monotonic()
        body = _read_operator_status(endpoint)
        if _readable(body):
            _assert_still_held(body, holder_sid)
            ordering = _sibling_ordering(body, holder_sid, sibling_sid) or ordering
            if read_at >= window_end:
                return ordering
        elif read_at >= window_end + _PARK_HEARTBEAT_TIMEOUT_SEC:
            raise AssertionError(f"cannot tell whether the holder kept its grant to the window's end: {body}")
        time.sleep(_STATUS_POLL_SEC)


# ---------------------------------------------------------------------------
# Parked live holder: scenarios
# ---------------------------------------------------------------------------


def test_parked_live_holder_is_reclaimed_and_refused_naming_its_lost_token(
    parked_holder_workspace: Path,
) -> None:
    """A live lease holder, heartbeating as itself on the held path, is stopped
    right after its ack. While it is stopped the sweep reclaims its grant on
    the stale heartbeat, moving the owner generation and not the version. The
    same process and session, resumed, commits once per route, and each
    refusal names the generation its claim was taken under and the one that
    owns the path now.

    What this does not prove: one machine only; the stall it models is not a
    network partition; and there is no cross-host fencing token.
    """
    endpoint = resolve_endpoint(parked_holder_workspace)
    sid = _sid()
    read = partial(_read_operator_status, endpoint)

    def while_parked(acked: dict[str, Any]) -> dict[str, Any]:
        _assert_stopped(acked["pid"])
        held = _poll_status(
            read, partial(_held_witness, session_id=sid),
            timeout_sec=_PARK_HEARTBEAT_TIMEOUT_SEC, what="the held grant",
        )
        reclaim = _poll_status(
            read, partial(_reclaim_witness, session_id=sid),
            timeout_sec=_RECLAIM_WAIT_SEC, what="the heartbeat reclaim",
        )
        return {"held": held, "reclaim": reclaim}

    result = run_park_after_ack(
        ContenderSpec(_child_parked_holder, args=(str(parked_holder_workspace), sid)), while_parked
    )

    version, held, reclaim = result.acked["version"], result.window["held"], result.window["reclaim"]
    assert held["last_heartbeat"] > held["granted_at"], held
    assert reclaim["trigger"] == "reclaim_heartbeat", reclaim
    assert reclaim["max_hold_reclaims"] == 0, reclaim
    assert reclaim["version"] == version == held["version"], (held, reclaim)
    assert reclaim["owner_generation"] > held["owner_generation"], (held, reclaim)

    lease, cas = result.value["lease"], result.value["cas"]
    assert lease["ok"] is False and lease.get("reclaimed") is True, lease
    assert "reclaimed_by=reclaim_heartbeat" in lease["reason"], lease
    _assert_lost_token(lease)
    assert cas["ok"] is False and cas["reason"] == "stale_read_generation", cas
    assert cas["current_version"] == version, cas
    _assert_lost_token(cas)


def test_beating_holder_is_not_reclaimed_without_the_stop(parked_holder_workspace: Path) -> None:
    """Control, stop switch off: the same holder beats on through two heartbeat
    windows and keeps its grant. A sibling grant, taken after the holder's by a
    session that never beats, is reclaimed no earlier than one window past the
    holder's grant, so the sweep was running all the while the holder held.

    What this does not prove: one machine only; the stall it models is not a
    network partition; and there is no cross-host fencing token.
    """
    endpoint = resolve_endpoint(parked_holder_workspace)
    sid = _sid()
    result = run_park_after_ack(
        ContenderSpec(_child_beating_holder, args=(str(parked_holder_workspace), sid)),
        lambda _acked: _watch_beating_holder(endpoint, sid),
        stop=False,
    )
    ordering = result.window
    assert ordering is not None, "no readable /status body listed the sibling's reclaim within the window"
    assert ordering["sibling_trigger"] == "reclaim_heartbeat", ordering
    assert ordering["sibling_reclaimed_at"] >= ordering["holder_granted_at"] + ordering["heartbeat_timeout"], ordering


def test_holder_ending_its_beats_is_reclaimed_without_the_stop(parked_holder_workspace: Path) -> None:
    """Control, stop switch off: the holder ends its own beats at the ack and
    waits there, running. The sweep reclaims it on the stale heartbeat before
    the parent resumes it.

    What this does not prove: one machine only; the stall it models is not a
    network partition; and there is no cross-host fencing token.
    """
    endpoint = resolve_endpoint(parked_holder_workspace)
    sid = _sid()
    result = run_park_after_ack(
        ContenderSpec(_child_holder_ending_beats_at_ack, args=(str(parked_holder_workspace), sid)),
        lambda _acked: _poll_status(
            partial(_read_operator_status, endpoint), partial(_reclaim_witness, session_id=sid),
            timeout_sec=_RECLAIM_WAIT_SEC, what="the heartbeat reclaim",
        ),
        stop=False,
    )
    assert result.window["trigger"] == "reclaim_heartbeat", result.window
    assert result.window["max_hold_reclaims"] == 0, result.window


def test_degraded_status_read_is_cannot_tell_and_read_again() -> None:
    """A degraded /status answer (null registry lists) is neither held nor not
    held: the witness reads again and answers from the next readable body, and
    nothing but degraded answers fails as could-not-witness."""
    sid = _sid()
    degraded = {"detail": "full", "degraded": True, "tracked_artifacts": None, "sessions": None}
    held = {
        "detail": "full",
        "tracked_artifacts": [{"path": _PATH, "version": 1, "owner_generation": 0}],
        "sessions": [
            {
                "agent_id": str(session_to_agent_id(sid)),
                "states": {_PATH: "EXCLUSIVE"},
                "reclaimed": {},
                "grants": {_PATH: {"granted_at_unix_ts": 100}},
                "last_heartbeat_unix_ts": 101,
            }
        ],
    }
    reads = iter([degraded, degraded, held])
    witness = partial(_held_witness, session_id=sid)

    verdict = _poll_status(lambda: next(reads), witness, timeout_sec=5.0, what="the held grant")

    assert verdict == {"granted_at": 100, "last_heartbeat": 101, "version": 1, "owner_generation": 0}
    with pytest.raises(AssertionError, match="could not witness the held grant within"):
        _poll_status(lambda: degraded, witness, timeout_sec=0.3, what="the held grant")
