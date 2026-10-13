# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""U3 — cross-process coordinator-layer assertions (guarantee-ladder plan).

The coordinator is the serialization point under test, so the only honest
cross-process shape is: one coordinator process (here: the parent, serving on
a daemon thread with test-scale sweep timings — the shipped CLI entrypoint
hard-codes a 600 s heartbeat window, so a test-owned config is the sanctioned
route per the plan's stop condition (c)), and contender CLIENTS as separate
spawn-context OS processes driving the shipped client path
(`resolve_endpoint` + `post` from `ccs.cli._coherence_client`).

Precedence note (pins shipped semantics, sharpening the plan's literal
scenario text): when a peer HAS committed, the version leg fires first and a
zombie sees `version_mismatch`; `stale_read_generation` is observable exactly
when the version is UNMOVED — the case version-CAS structurally cannot see,
which is the fence's whole job. Both variants are asserted.
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.core.substrate import sha256_hex
from ccs.testing.process_harness import ContenderSpec, ProcessRaceHarness, run_in_subprocess

# Spawns real processes that race each other: run with -n 0 (tests/conftest.py).
pytestmark = pytest.mark.serial

_PATH = "CLAUDE.md"  # in DEFAULT_TRACKED_PATTERNS — coordinated without extra policy files


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
# Coordinator fixture: parent-hosted, test-scale timings (plan U3 approach).
# ---------------------------------------------------------------------------


def _spawn(tmp_path: Path, *, sweep_interval_sec: float, heartbeat_timeout_sec: int) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / _PATH).write_text("seed\n")
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=sweep_interval_sec,
        grant_heartbeat_timeout_sec=heartbeat_timeout_sec,
        grant_max_hold_sec=heartbeat_timeout_sec * 5,
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
