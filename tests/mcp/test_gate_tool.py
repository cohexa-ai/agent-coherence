"""``swg_gate`` — the pull-based effect fence on the MCP tool surface.

``gate()`` cannot be a tool: its ``decide``/``effect`` are Python callables, and
an MCP agent's decision and effect live between tool calls, outside this
process. So the agent carries the ``(version, owner_generation)`` pair from its
``swg_read`` and pulls a verdict here immediately before dispatching something
irreversible. These tests drive the sync ``_do_*`` helpers against a real
loopback coordinator, matching the Unit-4 harness.

The headline is the reclaim leg: the grant the agent read under is reclaimed
while the bytes never move, so the VERSION still matches and only the
generation reveals it. That is the exact case the version-only surface could
not see.
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.adapters.claude_code import lifecycle
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import post as _cc_post
from ccs.cli._coherence_client import resolve_endpoint, resolve_remote_endpoint
from ccs.mcp.server import _do_gate, _do_read
from ccs.mcp.session import SessionConfig

REL = "data/shared.txt"


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def _seed(tmp_path: Path, content: bytes = b"v1") -> Path:
    target = tmp_path / REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _vol(tmp_path: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=cfg)


def _config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(root=tmp_path.resolve(), managed=("data/**",))


def _comparand(result) -> tuple[int, int | None]:
    body = result.structuredContent
    return body["version"], body["owner_generation"]


def test_read_surfaces_the_generation_comparand(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """swg_read carries BOTH comparands, so an agent can hold the pair its
    later swg_gate call re-checks."""
    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        res = _do_read(vol, _config(tmp_path), REL)
        assert res.isError is False
        version, generation = _comparand(res)
        assert version >= 1
        assert generation == 0, "a freshly observed artifact is at epoch 0"
    finally:
        stop_coordinator(tmp_path)


def test_gate_proceeds_when_unchanged(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Negative control: nothing moved, so the agent is cleared to dispatch."""
    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        cfg = _config(tmp_path)
        version, generation = _comparand(_do_read(vol, cfg, REL))
        res = _do_gate(vol, cfg, REL, version, generation)
        assert res.isError is False
        assert res.structuredContent["decision"] == "proceed"
    finally:
        stop_coordinator(tmp_path)


def test_gate_denies_when_a_peer_moved_the_version(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The value axis: a peer commit between read and dispatch denies."""
    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        cfg = _config(tmp_path)
        version, generation = _comparand(_do_read(vol, cfg, REL))
        peer = _vol(tmp_path, fast_cfg)
        peer.write(REL, b"peer-moved-it")
        res = _do_gate(vol, cfg, REL, version, generation)
        assert res.isError is True
        assert res.structuredContent["reason"] == "stale_view"
    finally:
        stop_coordinator(tmp_path)


def test_gate_denies_when_the_grant_was_reclaimed_version_unchanged(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """THE headline: the sweep reclaims the agent's grant while the bytes never
    move. The version comparand still matches — only the generation reveals
    that the authority behind the decision is gone. A version-only surface
    would clear the agent to dispatch here."""
    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        cfg = _config(tmp_path)
        vol.write(REL, b"v1")  # take an M grant
        version, generation = _comparand(_do_read(vol, cfg, REL))
        coordinator = lifecycle._SPAWNED_REGISTRY[str(tmp_path.resolve())].coordinator
        reclaimed = coordinator.service.enforce_stable_grant_timeouts(
            current_tick=int(time.time()) + 999_999,
            heartbeat_timeout_ticks=1,
            max_hold_ticks=999_999_999,
        )
        assert reclaimed >= 1, "the sweep must reclaim the grant under test"

        res = _do_gate(vol, cfg, REL, version, generation)
        assert res.isError is True, "a reclaimed grant must DENY, not proceed"
        assert res.structuredContent["reason"] == "stale_view"
        # The typed cause must name the deny, not the residual "no generation
        # reported" bucket — an agent that cannot tell them apart would follow
        # the wrong recovery (restart a healthy daemon instead of reacquiring).
        assert res.structuredContent["hold_cause"] == "read_denied"
        # The bytes never moved: the version the agent held is still current.
        assert vol.read_with_version(REL)[1] == version
    finally:
        stop_coordinator(tmp_path)


def test_gate_names_a_file_changed_outside_the_coordinator_a_split_pair(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """After swg_read, the file is edited outside the coordinator. swg_gate
    holds with ``hold_cause=split_pair`` (#405): the bytes on disk are not the
    content the coordinator records at the version the agent holds. Not
    ``read_denied``, which says the view is INVALID and sends the agent to a
    reacquire that never clears this. The deny answer stays ``stale_view``."""
    target = _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        cfg = _config(tmp_path)
        version, generation = _comparand(_do_read(vol, cfg, REL))
        target.write_bytes(b"edited-outside-the-coordinator")

        res = _do_gate(vol, cfg, REL, version, generation)

        assert res.isError is True
        assert res.structuredContent["reason"] == "stale_view"
        assert res.structuredContent["hold_cause"] == "split_pair"
    finally:
        stop_coordinator(tmp_path)


def test_gate_refuses_a_missing_authority_comparand_as_an_agent_error(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """An agent that supplies no owner_generation gets a TYPED agent error, not
    the retryable coherence deny. Both refuse the effect — but a stale_view
    says "reacquire and retry", and retrying can never supply a comparand the
    agent failed to carry, so that answer would be an unbounded loop. Name the
    real problem instead."""
    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        cfg = _config(tmp_path)
        version, _generation = _comparand(_do_read(vol, cfg, REL))
        res = _do_gate(vol, cfg, REL, version, None)
        assert res.isError is True, "no comparand must never mean 'proceed'"
        assert res.structuredContent["reason"] == "missing_comparand"
        assert res.structuredContent["recover"] == "reread"
    finally:
        stop_coordinator(tmp_path)


def test_gate_rejects_a_path_outside_the_root(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The gate re-validates its path like every other tool on this surface."""
    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        res = _do_gate(vol, _config(tmp_path), "../escape.txt", 1, 0)
        assert res.isError is True
        assert res.structuredContent["reason"] == "invalid_path"
    finally:
        stop_coordinator(tmp_path)


def test_gate_does_not_absolve_a_foreign_edit_for_the_next_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A verification must not count as an observation. The gate re-reads the
    file to compare comparands and DISCARDS the bytes, so it must leave the
    foreign-edit baseline alone. Otherwise calling the safety tool would make
    the NEXT write less safe than not calling it: the write-side guard would
    see the out-of-band bytes as already-observed and clobber them instead of
    denying."""
    from ccs.mcp.server import _do_write

    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        cfg = _config(tmp_path)
        version, generation = _comparand(_do_read(vol, cfg, REL))
        # An out-of-band editor changes the file behind the coordinator.
        (tmp_path / REL).write_bytes(b"foreign-v2")

        gated = _do_gate(vol, cfg, REL, version, generation)
        assert gated.isError is True, "the gate itself must HOLD on a foreign edit"

        # The write-side foreign-edit guard must STILL fire — the gate call
        # cannot have laundered the edit the agent never saw.
        denied = _do_write(vol, cfg, REL, "AGENT-WRITE")
        assert denied.isError is True, (
            "calling swg_gate must not absolve a foreign edit for the next write"
        )
        assert (tmp_path / REL).read_bytes() == b"foreign-v2", (
            "the foreign bytes must survive; the agent's write was denied"
        )
    finally:
        stop_coordinator(tmp_path)


def test_gate_on_a_vanished_input_answers_check_path_not_reacquire(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A vanished input must not be advertised as retryable-by-reacquire: the
    agent would be told to reacquire a file that no longer exists, and
    swg_reacquire would then refuse it. The gate answers with the same
    file_not_found shape swg_read gives, so the recovery advice matches
    reality."""
    _seed(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        cfg = _config(tmp_path)
        version, generation = _comparand(_do_read(vol, cfg, REL))
        (tmp_path / REL).unlink()
        res = _do_gate(vol, cfg, REL, version, generation)
        assert res.isError is True
        assert res.structuredContent["reason"] == "file_not_found"
        assert res.structuredContent["recover"] == "check_path"
        assert res.structuredContent["retryable"] is False
    finally:
        stop_coordinator(tmp_path)


def test_gate_surfaces_grant_preempted_and_re_denies_on_bare_retry(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire contract for the write-preemption leg. A peer's pessimistic
    write-acquire preempts the holder while the ``(version, owner_generation)``
    pair stays put, so ``swg_gate`` must DENY with ``reason=stale_view`` and the
    typed ``hold_cause=grant_preempted`` -- distinct from the strict leg's
    ``read_denied``. And the deny is LEVEL-triggered at the wire: a bare second
    call with the same comparands (the H4 retry shape) re-denies rather than
    admitting, because the fence's verification read no longer re-grants the
    preempted holder. Uses the tracked-but-NOT-strict topology, where the
    preempted re-read is a warn re-grant rather than a strict deny."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_w = tmp_path / "w"
    root_w.mkdir()
    lifecycle.ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        _cc_post(ep, "/policy/track", {"paths": ["task.txt"]})
        vol = CoherentVolume(
            root_w,
            on_error="strict",
            on_stale_write="allow",
            remote_endpoint=resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer),
        )
        cfg = SessionConfig(root=root_w.resolve(), managed=("*",))
        (root_w / "task.txt").write_bytes(b"task-v1")
        vol.write("task.txt", b"task-v1")  # M grant at v1
        version, generation = _comparand(_do_read(vol, cfg, "task.txt"))
        # A peer write-acquires, silently revoking this holder's grant.
        resp = _cc_post(
            ep, "/hooks/pre-edit", {"session_id": str(uuid4()), "path": "task.txt"}
        )
        assert resp.get("ok") is True, f"peer write-acquire failed: {resp}"

        res = _do_gate(vol, cfg, "task.txt", version, generation)
        assert res.isError is True, "a preempted grant must DENY, not proceed"
        assert res.structuredContent["reason"] == "stale_view"
        # grant_preempted (not version_moved) already proves the version never
        # moved -- the peer acquired but did not commit. Do NOT read through the
        # volume here to re-check it: an ordinary observe=True read re-grants
        # SHARED and would launder the preempted state (finding #1's own
        # mechanism), defeating the level-triggered check below.
        assert res.structuredContent["hold_cause"] == "grant_preempted"

        # Level-triggered: a bare retry (no reacquire) must re-DENY, not admit.
        res2 = _do_gate(vol, cfg, "task.txt", version, generation)
        assert res2.isError is True, "a bare retry must re-deny, not admit"
        assert res2.structuredContent["hold_cause"] == "grant_preempted"
    finally:
        stop_coordinator(coord_root)
