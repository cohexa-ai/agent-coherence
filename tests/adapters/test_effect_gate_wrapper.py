# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the effect-ordering gate wrapper (``gate()``).

EO-8 regression: an escaping effect is HELD (never fires) when a gated input's
version advanced between capture and fire, and fires when the input is
unchanged. Generation-fence regression: the effect is also HELD when a sweep
reclaimed the grant the decision was read under — owner_generation advanced
while the version stayed put, the drift a version-only comparand cannot see.
Uses a REAL coordinator subprocess (the ``test_coherent_volume.py`` pattern)
with a FIXED-STALE BUFFER: the peer commit (or the sweep reclaim) is driven
explicitly inside ``decide()`` (between capture and re-validate), never a
sleep.
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import uuid4

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code import lifecycle
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.adapters.effect_gate import check_fence, gate
from ccs.cli._coherence_client import CoordinatorUnavailable, resolve_endpoint, resolve_remote_endpoint
from ccs.cli._coherence_client import post as _cc_post
from ccs.core.exceptions import (
    HOLD_GRANT_PREEMPTED,
    HOLD_GRANT_RECLAIMED,
    HOLD_READ_DENIED,
    CoherenceDegradedWarning,
    CoherenceError,
    StaleView,
)

REL = "data/config.txt"


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def _seed(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)


def test_gate_exported_from_adapters_package() -> None:
    """``gate`` is reachable from the package (the lazy ``__getattr__`` export)."""
    from ccs.adapters import gate as pkg_gate

    assert pkg_gate is gate


def test_escaping_effect_fires_when_unchanged(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Negative control: an unchanged input fires the escaping effect once and
    returns its result, so the HELD test proves the deny, not a vacuous pass."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(REL, b"config-v1")
        fired: list[str] = []

        def decide(data: bytes) -> str:
            assert data == b"config-v1"
            return "deploy"

        def effect(decision: str) -> str:
            fired.append(decision)
            return f"{decision}-fired"

        result = gate(vol, REL, decide=decide, effect=effect)
        assert result == "deploy-fired"
        assert fired == ["deploy"]
    finally:
        stop_coordinator(tmp_path)


def test_escaping_effect_held_when_input_advanced(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """EO-8: ``gate()`` HOLDs (raises ``StaleView``, the effect NEVER fires) when
    a peer advances the gated input's version between capture and fire. The peer
    commit is a fixed-stale buffer driven inside ``decide()``, not a sleep."""
    _seed(tmp_path)
    vol_a = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol_a.write(REL, b"config-v1")
        vol_b = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
        fired: list[str] = []

        def decide(data: bytes) -> str:
            # FIXED-STALE BUFFER: a PEER advances the input between capture and
            # the gate's re-validate -- deterministic, not a timing race.
            vol_b.write(REL, b"config-v2-peer")
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(StaleView) as exc_info:
            gate(vol_a, REL, decide=decide, effect=effect)

        assert fired == []  # the effect never fired on stale input
        # the peer's single write advanced the version by exactly one (monotonic),
        # so pin the specific drift, not merely that the two versions differ.
        assert exc_info.value.current_version == exc_info.value.expected_version + 1
    finally:
        stop_coordinator(tmp_path)


def test_escaping_effect_held_when_grant_reclaimed_strict_deny_leg(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Generation-fence regression, strict leg (the reclaim-zombie
    interleaving): a sweep reclaims THIS holder's stalled M grant between
    capture and fire. Nothing else writes the artifact, so the version
    comparand is UNCHANGED. A CoherentVolume installs its managed globs
    strict, so the zombie's re-validation read is the byte-stable strict deny
    — which deliberately carries NO generation. PRE-FIX, gate() extracted the
    unchanged version from the deny's summary, discarded ``stale_denied``,
    and FIRED the effect straight through the deny; POST-FIX the unconfirmed
    generation HOLDs it. The reclaim is a fixed-stale buffer driven
    explicitly inside ``decide()`` with synthetic far-future ticks, never a
    sleep."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(REL, b"config-v1")  # M grant, heartbeat seeded at now
        coordinator = lifecycle._SPAWNED_REGISTRY[str(tmp_path.resolve())].coordinator
        fired: list[str] = []

        def decide(data: bytes) -> str:
            # FIXED-STALE BUFFER: the sweep reclaims this holder's grant between
            # capture and re-validate (heartbeat_timeout_ticks=1 against a
            # far-future tick forces reclaim_heartbeat; max_hold stays off).
            reclaimed = coordinator.service.enforce_stable_grant_timeouts(
                current_tick=int(time.time()) + 999_999,
                heartbeat_timeout_ticks=1,
                max_hold_ticks=999_999_999,
            )
            assert reclaimed == 1, "sweep should have reclaimed exactly one grant"
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(StaleView) as exc_info:
            gate(vol, REL, decide=decide, effect=effect)

        assert fired == []  # the zombie's effect never fired through the deny
        exc = exc_info.value
        # THE pre-fix hole: the version never moved (the deny's summary still
        # matched), so a version-only comparand admitted the fire…
        assert exc.current_version == exc.expected_version
        # …and the HOLD came from the generation leg: confirmed at capture,
        # deliberately unconfirmed on the strict deny. The cause names the DENY
        # specifically — on the strict path that is how a reclaim reaches the
        # client, and it is recoverable, so it must not hide in the residual
        # "no generation reported" bucket (which includes the one case a
        # reacquire cannot fix).
        assert exc.expected_generation is not None
        assert exc.current_generation is None
        assert exc.hold_cause == HOLD_READ_DENIED
        assert "denied at re-read" in str(exc)
    finally:
        stop_coordinator(tmp_path)


def test_escaping_effect_held_when_grant_reclaimed_version_unchanged(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generation-fence regression, warn leg: against a coordinator that
    TRACKS the artifact without strict mode (the fleet / foreign-coordinator
    shape), the zombie's re-validation is a warn-stale re-grant carrying the
    SAME version and an ADVANCED owner_generation — the exact drift a
    version-only comparand cannot see. PRE-FIX gate() compared versions,
    found them equal, and FIRED; POST-FIX it HOLDs on the moved epoch, with
    the g→g+1 drift on the exception."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_w = tmp_path / "w"
    root_w.mkdir()
    lifecycle.ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        # Tracked (versioned + stale detection) but NOT strict: the reclaimed
        # holder's re-read is a warn-stale re-grant, not a deny.
        _cc_post(ep, "/policy/track", {"paths": ["task.txt"]})
        vol = CoherentVolume(
            root_w,
            on_error="strict",
            on_stale_write="allow",
            remote_endpoint=resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer),
        )
        (root_w / "task.txt").write_bytes(b"task-v1")
        vol.write("task.txt", b"task-v1")  # M grant, heartbeat seeded at now
        coordinator = lifecycle._SPAWNED_REGISTRY[
            str(coord_root.resolve())
        ].coordinator
        fired: list[str] = []

        def decide(data: bytes) -> str:
            reclaimed = coordinator.service.enforce_stable_grant_timeouts(
                current_tick=int(time.time()) + 999_999,
                heartbeat_timeout_ticks=1,
                max_hold_ticks=999_999_999,
            )
            assert reclaimed == 1, "sweep should have reclaimed exactly one grant"
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(StaleView) as exc_info:
            gate(vol, "task.txt", decide=decide, effect=effect)

        assert fired == []  # the zombie's effect never fired
        exc = exc_info.value
        # THE distinguishing shape: the version never moved…
        assert exc.current_version == exc.expected_version
        # …but the ownership epoch advanced by exactly the one reclaim.
        assert exc.expected_generation is not None
        assert exc.current_generation == exc.expected_generation + 1
        assert "grant reclaimed" in str(exc)
    finally:
        stop_coordinator(coord_root)


def test_escaping_effect_held_when_grant_preempted_at_unchanged_pair(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Write-preemption sibling of the sweep-reclaim regression above: a PEER's
    pessimistic write-acquire (the ``/hooks/pre-edit`` leg of an edit) silently
    takes this holder's grant between capture and fire. Unlike a sweep reclaim
    or a voluntary release, the acquire moves NOTHING the pair can see — no
    commit yet, so the version stays put, and ``trigger="write"`` is outside
    EPOCH_BUMP_TRIGGERS, so the ownership epoch stays put too. The
    ``(version, owner_generation)`` comparands are structurally blind here; the
    gate must instead notice that the caller's grant did not STAND at the
    re-validate read (the warn-leg re-read is a stale re-grant, not a fresh
    confirm). PRE-FIX the pair compared equal and the preempted holder's
    effect FIRED; POST-FIX it HOLDs with the typed ``grant_preempted`` cause."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_w = tmp_path / "w"
    root_w.mkdir()
    lifecycle.ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        # Tracked but NOT strict: the preempted holder's re-read is a warn-mode
        # stale re-grant carrying the SAME confirmed (version, generation).
        _cc_post(ep, "/policy/track", {"paths": ["task.txt"]})
        vol = CoherentVolume(
            root_w,
            on_error="strict",
            on_stale_write="allow",
            remote_endpoint=resolve_remote_endpoint("127.0.0.1", ep.port, ep.bearer),
        )
        (root_w / "task.txt").write_bytes(b"task-v1")
        vol.write("task.txt", b"task-v1")  # M grant at v1
        fired: list[str] = []

        def decide(data: bytes) -> str:
            # FIXED-STALE BUFFER: the peer write-ACQUIRES between capture and
            # re-validate — a real wire-level pre-edit with no commit behind
            # it, driven deterministically inside decide(), never a sleep.
            resp = _cc_post(
                ep,
                "/hooks/pre-edit",
                {"session_id": str(uuid4()), "path": "task.txt"},
            )
            assert resp.get("ok") is True, f"peer write-acquire failed: {resp}"
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(StaleView) as exc_info:
            gate(vol, "task.txt", decide=decide, effect=effect)

        assert fired == []  # the preempted holder's effect never fired
        exc = exc_info.value
        # THE distinguishing shape vs the sweep sibling: the version never
        # moved AND the ownership epoch never moved — both comparands equal
        # and confirmed —
        assert exc.current_version == exc.expected_version
        assert exc.expected_generation is not None
        assert exc.current_generation == exc.expected_generation
        # — so the HOLD keys on the grant not standing at the re-validate.
        assert exc.hold_cause == HOLD_GRANT_PREEMPTED
        assert "was not under a standing grant" in str(exc)
    finally:
        stop_coordinator(coord_root)


def test_grant_preempted_hold_is_level_triggered_across_bare_regates(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write-preemption HOLD must be LEVEL-triggered, not edge-triggered:
    a preempted holder that pulls the fence verdict twice with the SAME
    comparands — no ``reacquire()`` between — must HOLD BOTH times. This is the
    ``swg_gate`` retry shape (an agent that reads once, holds the pair, and
    asks again before dispatch), and the H4 reality that models retry a deny
    2-5 times.

    The hazard the fence's own re-read used to create: a warn-leg re-validate
    re-granted the caller SHARED (``post_stale_read``), healing the very grant
    loss it was checking, so the SECOND bare check took the fresh branch and
    ADMITTED — the peer still holds EXCLUSIVE. The verification read now travels
    ``verify_only`` so the coordinator does NOT re-grant on it; the caller stays
    INVALID and every bare re-check re-HOLDs until a real ``reacquire()``. This
    mirrors the strict leg (KTD-T: a strict deny never re-grants), so warn and
    strict converge. PRE-FIX the second check returned ``None`` (proceed)."""
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
        (root_w / "task.txt").write_bytes(b"task-v1")
        vol.write("task.txt", b"task-v1")  # M grant at v1
        # Capture the pair the agent would carry from its swg_read.
        _data, ev, eg = vol.read_with_version_generation("task.txt")
        # A peer write-acquires, silently revoking this holder's grant.
        resp = _cc_post(
            ep, "/hooks/pre-edit", {"session_id": str(uuid4()), "path": "task.txt"}
        )
        assert resp.get("ok") is True, f"peer write-acquire failed: {resp}"

        # Pull the fence verdict repeatedly with the SAME comparands, no
        # reacquire between — every pull must HOLD grant_preempted.
        for attempt in range(3):
            with pytest.raises(StaleView) as exc_info:
                check_fence(
                    vol,
                    "task.txt",
                    expected_version=ev,
                    expected_generation=eg,
                )
            assert exc_info.value.hold_cause == HOLD_GRANT_PREEMPTED, (
                f"attempt {attempt}: a bare re-check must re-HOLD, not admit — "
                "the fence read must not re-grant the preempted holder"
            )
            # The comparands never move: only the standing-grant leg sees it.
            assert exc_info.value.current_version == ev
            assert exc_info.value.current_generation == eg

        # Recovery is via reacquire(), which DOES re-mint a live grant — after
        # it the same path re-gates clean (the deny is recoverable, not sticky).
        vol.reacquire("task.txt")
        _d2, v2, g2 = vol.read_with_version_generation("task.txt")
        check_fence(vol, "task.txt", expected_version=v2, expected_generation=g2)
    finally:
        stop_coordinator(coord_root)


def test_escaping_effect_held_when_warn_mode_bytes_are_stale(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warn-mode value/comparand binding: the bytes handed to decide() and the
    comparand pair come from two instants (a disk read, then the coordinator
    round-trip). On a tracked-but-NOT-strict artifact a superseded reader is
    warn-allowed — re-granted with the CURRENT pair — so the pair would
    re-validate clean at the boundary while decide() consumed bytes the
    coordinator knows are superseded. The coordinator flags exactly that
    (``hash_differs``), and an unconfirmable view yields NO confirmed
    generation, so the gate HOLDs rather than firing a stale-derived effect."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    coord_root = tmp_path / "coord"
    coord_root.mkdir()
    root_w = tmp_path / "w"
    root_w.mkdir()
    root_p = tmp_path / "p"
    root_p.mkdir()
    lifecycle.ensure_coordinator(coord_root, config=fast_cfg)
    try:
        ep = resolve_endpoint(coord_root)
        _cc_post(ep, "/policy/track", {"paths": ["task.txt"]})

        def vol(root: Path) -> CoherentVolume:
            return CoherentVolume(
                root,
                on_error="strict",
                on_stale_write="allow",
                remote_endpoint=resolve_remote_endpoint(
                    "127.0.0.1", ep.port, ep.bearer
                ),
            )

        worker, peer = vol(root_w), vol(root_p)
        (root_w / "task.txt").write_bytes(b"plan-A")
        (root_p / "task.txt").write_bytes(b"plan-A")
        worker.write("task.txt", b"plan-A")
        # The peer supersedes the content; the worker's own disk still holds
        # plan-A, so its next read hands decide() superseded bytes.
        peer.write("task.txt", b"plan-B")
        fired: list[str] = []

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(StaleView) as exc_info:
            gate(worker, "task.txt", decide=lambda d: d.decode(), effect=effect)

        assert fired == []  # never fired on a decision derived from plan-A
        assert exc_info.value.current_generation is None
        assert "no confirmed ownership generation" in str(exc_info.value)
    finally:
        stop_coordinator(coord_root)


def test_gate_fires_on_the_callers_own_recent_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The fail-closed ``hash_differs`` rule must not over-trigger on the most
    common path there is: a caller gating an artifact it just wrote itself. Its
    own committed bytes ARE the canonical content, so the generation stays
    confirmed and the effect fires. Repeated write -> gate cycles too, so a
    commit-to-disk lag cannot accumulate into a permanent HOLD."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        fired: list[str] = []
        for content in (b"config-v1", b"config-v2"):
            vol.write(REL, content)
            fired.clear()
            gate(vol, REL, decide=lambda d: d.decode(), effect=fired.append)
            assert fired == [content.decode()], (
                "gating a path the caller just wrote must FIRE, never HOLD"
            )
    finally:
        stop_coordinator(tmp_path)


def test_gate_fires_on_own_write_in_warn_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same guarantee on the tracked-but-NOT-strict topology, where a superseded
    read is a warn-mode re-grant rather than a deny — the arm where an
    over-eager unconfirmed-generation rule would silently wedge every gate."""
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
        (root_w / "task.txt").write_bytes(b"plan-A")
        vol.write("task.txt", b"plan-A")
        fired: list[str] = []
        gate(vol, "task.txt", decide=lambda d: d.decode(), effect=fired.append)
        assert fired == ["plan-A"]
    finally:
        stop_coordinator(coord_root)


def test_effect_required(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """Omitting the ``effect`` callable is a caller error (required kw-only)."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(REL, b"config-v1")
        with pytest.raises(TypeError):
            gate(vol, REL, decide=lambda d: "x")  # type: ignore[call-arg]
    finally:
        stop_coordinator(tmp_path)


def test_vanished_input_holds(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """If the gated input vanishes between capture and re-validate, the gate
    HOLDs fail-closed (cannot prove unchanged) and never fires."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(REL, b"config-v1")
        fired: list[str] = []

        def decide(data: bytes) -> str:
            (tmp_path / REL).unlink()  # vanish between capture and re-validate
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "x"

        with pytest.raises(StaleView):
            gate(vol, REL, decide=decide, effect=effect)
        assert fired == []
    finally:
        stop_coordinator(tmp_path)


def test_gate_holds_end_to_end_when_coordinator_degrades(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """End-to-end degrade: a real coordinator taken down between capture and
    re-validate surfaces version 0 (degrade mode, fail-closed), so the gate
    HOLDs and the escaping effect never fires on an unconfirmed input."""
    _seed(tmp_path)
    vol = CoherentVolume(
        tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg
    )
    try:
        vol.write(REL, b"config-v1")
        fired: list[str] = []

        def decide(data: bytes) -> str:
            stop_coordinator(tmp_path)  # coordinator down before the re-validate read
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.warns(CoherenceDegradedWarning), pytest.raises(StaleView):
            gate(vol, REL, decide=decide, effect=effect)
        assert fired == []
    finally:
        stop_coordinator(tmp_path)


def test_gate_propagates_infra_error_in_strict_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Strict mode: an infra failure during the re-validate read raises a raw
    CoherenceError (NOT a StaleView HOLD), which propagates through gate()
    unmodified -- the effect still never fires."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)  # strict default
    try:
        vol.write(REL, b"config-v1")
        fired: list[str] = []

        def decide(data: bytes) -> str:
            stop_coordinator(tmp_path)
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(CoherenceError) as exc_info:
            gate(vol, REL, decide=decide, effect=effect)
        assert not isinstance(exc_info.value, StaleView)  # infra failure, not a HOLD
        assert fired == []
    finally:
        stop_coordinator(tmp_path)


def test_forked_child_check_fence_holds_on_the_parents_comparands(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A forked child pulling the verdict with comparands its parent captured
    must HOLD: the child re-mints its identity, so the grant the decision was
    read under is the parent's, never the child's. The fence's verification
    read re-attaches the child and is answered stale under the fresh identity
    -- on every retry, not just the first: a verify-only read must not grant
    that identity anything, or the child's next pull would be admitted. If the
    coordinator ever answered it fresh, the child would fire an effect decided
    under a grant it does not hold."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(REL, b"config-v1")
        _data, version, generation = vol.read_with_version_generation(REL)
        # Confirmed comparands, so the HOLD below cannot be an unconfirmed one.
        assert version > 0 and generation is not None

        vol._after_fork()  # simulate the child-side fork handler

        # Pull the verdict repeatedly with the same comparands and no
        # reacquire() between: every pull must HOLD, not only the first.
        for attempt in range(3):
            with pytest.raises(StaleView) as exc_info:
                check_fence(vol, REL, expected_version=version, expected_generation=generation)
            exc = exc_info.value
            assert exc.hold_cause == HOLD_GRANT_PREEMPTED, f"pull {attempt + 1}"
            assert (exc.current_version, exc.current_generation) == (version, generation)
        assert vol.is_attached  # the verification read re-attached the child
    finally:
        stop_coordinator(tmp_path)


def test_forked_child_gate_raises_without_firing_while_reattach_fails(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strict child forked between capture and re-validate, whose re-attach
    fails, raises the infra error from the re-validate read -- not a HOLD built
    on a version it never asked the coordinator for -- and the effect never
    fires."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)  # strict default
    try:
        vol.write(REL, b"config-v1")
        fired: list[str] = []

        def unavailable(*_args: object) -> None:
            raise CoordinatorUnavailable("server.pid missing (coordinator restarting)")

        def decide(data: bytes) -> str:
            vol._after_fork()  # the child continues the gate from here
            monkeypatch.setattr(coherent_volume_module, "resolve_endpoint", unavailable)
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(CoherenceError) as exc_info:
            gate(vol, REL, decide=decide, effect=effect)
        assert not isinstance(exc_info.value, StaleView)  # infra failure, not a HOLD
        assert fired == []
        assert not vol.is_attached
    finally:
        stop_coordinator(tmp_path)


def test_gate_accepts_pathlib_path(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """gate() accepts an os.PathLike input (pathlib.Path), not only str, and
    fires when unchanged -- exercising the PathLike type through the real
    read_with_version round-trip."""
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(REL, b"config-v1")
        result = gate(vol, Path(REL), decide=lambda d: "go", effect=lambda x: f"{x}-fired")
        assert result == "go-fired"
    finally:
        stop_coordinator(tmp_path)


def test_gate_does_not_absolve_a_foreign_edit_it_never_showed_the_caller(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Python-API twin of the MCP surface's paired-write regression.

    ``gate()`` makes TWO reads, and only the first is an observation: its bytes
    go to ``decide()``, so seeding the foreign-edit baseline there is correct.
    The boundary re-read is a VERIFICATION — it compares comparands and discards
    the bytes — so it must leave the baseline alone. An out-of-band edit landing
    between the two is therefore one the caller demonstrably never saw, and the
    caller's next write must still be denied.

    Without ``observe=False`` on the boundary read the baseline advances to the
    foreign bytes, the write-side guard sees them as already-observed, and the
    agent silently clobbers an edit it was never shown — so calling the safety
    check would make the NEXT write less safe than skipping it.

    Read the two HOLDs differently. The FIRST is only a precondition: the gate
    refuses, but through the coordinator's own hash-mismatch handling, which is
    independent of the flag under test — and *which* mechanism answers is
    timing-dependent (inside the self-commit-lag window the client's
    ``hash_differs`` rule reports an unconfirmed generation; outside it the
    coordinator strict-denies), so this deliberately does not pin a
    ``hold_cause`` there. The SECOND raise is the one that pins ``observe``:
    it is the SB-23 write-side guard, asserted on its own reason text so the
    test cannot pass on an unrelated deny.
    """
    _seed(tmp_path)
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=fast_cfg)
    try:
        vol.write(REL, b"config-v1")
        fired: list[str] = []

        def decide(data: bytes) -> str:
            assert data == b"config-v1", "the caller observes v1, and only v1"
            # FIXED-STALE BUFFER: an out-of-band editor lands AFTER the caller's
            # observation and BEFORE the boundary re-read — deterministic, no sleep.
            (tmp_path / REL).write_bytes(b"foreign-v2")
            return "deploy"

        def effect(decision: str) -> str:  # pragma: no cover - must never run
            fired.append(decision)
            return "should-not-fire"

        with pytest.raises(StaleView):  # precondition, not the assertion under test
            gate(vol, REL, decide=decide, effect=effect)
        assert fired == []

        # THE POINT: the caller was shown config-v1 and never foreign-v2, so the
        # write-side foreign-edit guard must still fire. Pinned on the guard's own
        # reason text — a bare `raises(StaleView)` would also accept an unrelated
        # deny (a sticky-INVALID view, a strict pre-edit refusal) and pass while
        # proving nothing.
        with pytest.raises(StaleView) as denied:
            vol.write(REL, b"agent-write")
        assert "out-of-band edit" in str(denied.value), (
            "the denial must come from the SB-23 foreign-edit guard, not from "
            f"some other refusal: {denied.value}"
        )
        assert (tmp_path / REL).read_bytes() == b"foreign-v2", (
            "the foreign bytes must survive — the gate must not have laundered "
            "an edit it read only to compare and then discarded"
        )
    finally:
        stop_coordinator(tmp_path)

# --- Fast unit tests of gate()'s pure logic (a stub volume, no coordinator) ----


class _StubVolume:
    """A minimal CoherentVolume stand-in for gate()'s pure logic: successive
    read_with_version_generation calls yield scripted
    (bytes, version, owner_generation) triples, or raise. A confirmed but
    arbitrary generation (7) is the default in version-focused tests so the
    version semantics stay isolated from the generation leg.

    Both grant-state flags are declared in the CLASS BODY, the way
    ``CoherentVolume`` declares them — the fence now refuses a volume that
    reports neither, so a double that omits them is testing the refusal, not
    the comparand legs. Subclasses below override them per read; the two
    refusal tests deliberately do NOT derive from this class."""

    #: Whether the last read was refused by the coordinator (strict deny), and
    #: whether it was served without a standing grant. False on every read here
    #: unless a subclass says otherwise.
    _last_read_denied: bool = False
    _last_read_stale: bool = False

    def __init__(self, reads: list) -> None:
        self._reads = list(reads)
        self._i = 0

    def read_with_version_generation(
        self, path: str, *, observe: bool = True
    ) -> tuple[bytes, int, int | None]:
        item = self._reads[self._i]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return item


def test_gate_fires_on_stub_unchanged() -> None:
    fired: list[str] = []

    def effect(x: str) -> str:
        fired.append(x)
        return "ok"

    result = gate(
        _StubVolume([(b"cfg", 5, 7), (b"cfg", 5, 7)]),
        "p",
        decide=lambda d: "go",
        effect=effect,
    )
    assert result == "ok"
    assert fired == ["go"]


def test_gate_holds_on_stub_moved() -> None:
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, 7), (b"cfg", 6, 7)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_version == 5
    assert exc.value.current_version == 6


def test_gate_holds_on_unconfirmed_version() -> None:
    """Fail-closed: an unresolved version (0 -- a degraded / degrade-mode read)
    HOLDs rather than firing on input the coordinator never confirmed."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 0, 7), (b"cfg", 0, 7)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_version == 0


def test_gate_holds_on_vanish_carries_none_current() -> None:
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, 7), FileNotFoundError()]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.current_version is None
    # _held() threads the capture generation through the vanish branch too, so
    # a handler reading the drift sees the same shape on every HOLD class.
    assert exc.value.expected_generation == 7
    assert exc.value.current_generation is None


def test_gate_holds_when_capture_read_degraded() -> None:
    """Degrade-mode capture: a capture read that could not be confirmed
    (version 0) HOLDs even though the re-read resolved to a real version."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 0, 7), (b"cfg", 5, 7)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_version == 0


def test_gate_holds_when_revalidate_read_degraded() -> None:
    """Degrade-mode re-read: a real capture (v5) whose re-read degrades to
    version 0 HOLDs -- the degrade-mode race the fail-closed guard covers.
    The ``unconfirmed`` term itself is pinned by the (0,0) case in
    test_gate_holds_on_unconfirmed_version (dropping the term fails that test);
    ``or`` vs ``and`` on the term is behaviorally equivalent across gate()'s
    reachable version domain, so no black-box test distinguishes them."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, 7), (b"cfg", 0, 7)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_version == 5
    assert exc.value.current_version == 0


def test_gate_holds_on_generation_moved_version_unchanged() -> None:
    """THE generation-fence case: the version comparand is unchanged (nothing
    re-wrote the artifact) but the ownership generation advanced (a sweep
    reclaimed the grant the decision was read under). A version-only check
    fires here; the gate must HOLD, with the drift on the exception."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, 7), (b"cfg", 5, 8)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_version == 5
    assert exc.value.current_version == 5  # the version NEVER moved
    assert exc.value.expected_generation == 7
    assert exc.value.current_generation == 8
    assert "grant reclaimed" in str(exc.value)


def test_gate_holds_on_lapsed_grant_with_pair_unchanged() -> None:
    """Pure-logic pin of the standing-grant leg: the re-validate read reports a
    LAPSED grant (a warn-mode stale re-grant) while BOTH comparands are
    unchanged and confirmed — the write-preemption shape. The pair alone would
    admit; the standing-grant check HOLDs with the typed cause."""

    class _LapsedStub(_StubVolume):
        def read_with_version_generation(
            self, path: str, *, observe: bool = True
        ) -> tuple[bytes, int, int | None]:
            out = super().read_with_version_generation(path, observe=observe)
            # The observe=False read is the fence's re-validate: the
            # coordinator answered it with a stale re-grant, not a confirm.
            self._last_read_stale = not observe
            return out

    with pytest.raises(StaleView) as exc:
        gate(
            _LapsedStub([(b"c", 5, 7), (b"c", 5, 7)]),
            "p",
            decide=lambda d: "go",
            effect=lambda x: x,
        )
    assert exc.value.hold_cause == HOLD_GRANT_PREEMPTED
    assert exc.value.expected_version == 5
    assert exc.value.current_version == 5
    assert exc.value.expected_generation == 7
    assert exc.value.current_generation == 7


def test_check_fence_refuses_a_volume_that_cannot_report_grant_state() -> None:
    """A volume that cannot report the grant state of its last read is REFUSED,
    not silently treated as standing under a live grant.

    The two flags used to be read through ``getattr(..., False)``, justified as
    compatibility for duck-typed volumes predating them. That allowance
    silently disabled the third fence leg — the one that catches a peer's
    pessimistic write-acquire, which moves NEITHER comparand — so the clean,
    unchanged pair below FIRED the escaping effect for any such object. Its
    twin above (``test_gate_holds_on_lapsed_grant_with_pair_unchanged``) is the
    same pair with the flags supplied, and still HOLDs: this closes the hole
    without disabling the leg.

    The refusal is caught on the DOCUMENTED BASE CLASS, never on the concrete
    type. A missing ``FenceComparands`` keyword raises ``TypeError`` and a bare
    attribute read raises ``AttributeError``; neither is catchable as a
    coherence failure, so the caller that holds today (``ccs.mcp.server``
    catches exactly this base around ``check_fence``) would CRASH instead of
    holding. A ``pytest.raises`` on the concrete type cannot see that
    regression — which is the whole reason this test exists.
    """

    class _GrantStateBlindVolume:
        """Duck-typed volume that answers with both comparands but never says
        whether its read was refused or served without a standing grant."""

        def read_with_version_generation(
            self, path: str, *, observe: bool = True
        ) -> tuple[bytes, int, int | None]:
            return (b"cfg", 5, 7)

    fired: list[str] = []
    with pytest.raises(CoherenceError) as exc:
        gate(
            _GrantStateBlindVolume(),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    # PRE-FIX this pair was confirmed, unchanged and (by default) ungrant-checked:
    # the effect fired. The refusal is worth nothing if the effect still ran.
    assert fired == []
    # Typed, not a HOLD: re-reading cannot supply a flag the volume never
    # declares, so answering with the retryable ``stale_view`` recovery would
    # send a cooperating agent into a reacquire loop that can never clear.
    assert not isinstance(exc.value, StaleView)


def test_check_fence_refuses_a_volume_reporting_only_the_deny_flag() -> None:
    """The half-equipped shape, which is exactly the fail-open the pair-only
    default produced: a volume that reports the strict-deny answer but not the
    standing-grant one keeps the deny leg and silently loses the preemption
    leg. Refused as well — the fence needs BOTH answers, and a partially
    equipped volume is the one an incremental implementer actually writes.

    Caught on the base class for the same reason as its sibling above.
    """

    class _HalfBlindVolume:
        #: Declared. Its sibling ``_last_read_stale`` deliberately is NOT — that
        #: asymmetry is the point of this double.
        _last_read_denied = False

        def read_with_version_generation(
            self, path: str, *, observe: bool = True
        ) -> tuple[bytes, int, int | None]:
            return (b"cfg", 5, 7)

    fired: list[str] = []
    with pytest.raises(CoherenceError) as exc:
        gate(
            _HalfBlindVolume(),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert not isinstance(exc.value, StaleView)


def test_grant_reclaimed_wins_when_lapsed_and_generation_both_moved() -> None:
    """Branch-priority pin for _held(): when the re-read is BOTH lapsed AND the
    ownership generation moved (a sweep reclaim that also left the caller
    without a standing grant), the more-specific ``grant_reclaimed`` cause must
    win over ``grant_preempted`` — the generation check sits before the lapsed
    check, so the g->g+1 drift is named rather than hidden behind the residual
    preemption label."""

    class _LapsedStub(_StubVolume):
        def read_with_version_generation(
            self, path: str, *, observe: bool = True
        ) -> tuple[bytes, int, int | None]:
            out = super().read_with_version_generation(path, observe=observe)
            self._last_read_stale = not observe
            return out

    with pytest.raises(StaleView) as exc:
        gate(
            # capture (v5,g7); re-validate (v5,g8) AND lapsed.
            _LapsedStub([(b"c", 5, 7), (b"c", 5, 8)]),
            "p",
            decide=lambda d: "go",
            effect=lambda x: x,
        )
    assert exc.value.hold_cause == HOLD_GRANT_RECLAIMED
    assert exc.value.current_generation == exc.value.expected_generation + 1


def test_gate_holds_on_unconfirmed_capture_generation() -> None:
    """Fail-closed: a capture read with no confirmed generation (an older
    coordinator that predates the fence, a strict deny, or a degraded read)
    HOLDs even though both versions match and the re-read carries one."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, None), (b"cfg", 5, 7)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_generation is None
    assert exc.value.current_generation == 7


def test_gate_holds_on_unconfirmed_revalidate_generation() -> None:
    """Fail-closed, the other leg: a confirmed capture whose re-read loses the
    generation confirmation (a deny or degrade mid-flight) HOLDs."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, 7), (b"cfg", 5, None)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_generation == 7
    assert exc.value.current_generation is None


def test_gate_holds_when_both_comparands_moved_version_detail_wins() -> None:
    """Both axes moved at once (a peer commit AND a reclaim). It HOLDs, carries
    both drifts, and the message reports the version move — the deliberate
    branch priority in _held(), which is otherwise unpinned."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, 7), (b"cfg", 6, 8)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert (exc.value.expected_version, exc.value.current_version) == (5, 6)
    assert (exc.value.expected_generation, exc.value.current_generation) == (7, 8)
    assert "moved to v6" in str(exc.value)


def test_gate_holds_when_generation_unconfirmed_on_both_reads() -> None:
    """The shape an older coordinator produces on EVERY read: matching versions
    and None generations on both sides. Two Nones are not "no movement" — the
    gate must HOLD loudly rather than silently reverting to the
    generation-blind check."""
    fired: list[str] = []
    with pytest.raises(StaleView) as exc:
        gate(
            _StubVolume([(b"cfg", 5, None), (b"cfg", 5, None)]),
            "p",
            decide=lambda d: "go",
            effect=fired.append,
        )
    assert fired == []
    assert exc.value.expected_generation is None
    assert exc.value.current_generation is None
    assert "no confirmed ownership generation" in str(exc.value)


def test_gate_requires_callable_effect() -> None:
    """The explicit non-callable ``effect`` guard (distinct from omitting it)."""
    with pytest.raises(TypeError):
        gate(_StubVolume([(b"x", 1, 7)]), "p", decide=lambda d: "go", effect=5)  # type: ignore[arg-type]


def test_gate_requires_callable_decide() -> None:
    with pytest.raises(TypeError):
        gate(_StubVolume([(b"x", 1, 7)]), "p", decide=None, effect=lambda x: x)  # type: ignore[arg-type]


def test_gate_hold_tolerates_non_pathlike_path() -> None:
    """The HOLD message builds via os.fspath() but falls back to str(), so a
    duck-typed volume with a non-PathLike path still raises StaleView rather than
    a masking TypeError (a real CoherentVolume rejects such a path earlier)."""
    with pytest.raises(StaleView):
        gate(
            _StubVolume([(b"cfg", 5, 7), (b"cfg", 6, 7)]),
            12345,  # type: ignore[arg-type]
            decide=lambda d: "go",
            effect=lambda x: x,
        )


def test_bare_stale_view_exposes_none_version_attrs() -> None:
    """A StaleView raised without drift (the coordinator deny sites) still
    exposes expected_version/current_version AND the generation pair as None,
    so a generic ``except StaleView`` handler reads the same shape gate()
    raises."""
    exc = StaleView("peer committed a newer version")
    assert exc.expected_version is None
    assert exc.current_version is None
    assert exc.expected_generation is None
    assert exc.current_generation is None


def test_hold_cause_is_typed_per_class() -> None:
    """Every HOLD class carries a distinct typed cause, so an agent branches on
    a value rather than substring-matching the human message — and can tell a
    HOLD reacquire() clears from one that it never will.

    All SEVEN shipped causes, asserted through the wrapper's public surface.
    This is the characterisation that pins the in-process verdict across the
    move of the branch table into ``ccs.core.fence``: the classification now has
    a second caller, and a reason that quietly changed identity during the move
    would keep every message-text assertion in this file green while sending an
    agent down the wrong recovery.
    """
    from ccs.core.exceptions import (
        HOLD_GENERATION_UNCONFIRMED,
        HOLD_GRANT_RECLAIMED,
        HOLD_INPUT_VANISHED,
        HOLD_VERSION_MOVED,
        HOLD_VERSION_UNCONFIRMED,
    )

    class _FlaggedStub(_StubVolume):
        """A stub whose re-validate read (``observe=False``) reports the
        coordinator's grant-state answer — a strict deny, a lapsed grant, or
        neither — the way a real volume records it on ``_last_read_*``."""

        def __init__(self, reads: list, *, denied: bool, lapsed: bool) -> None:
            super().__init__(reads)
            self._denied = denied
            self._lapsed = lapsed
            self._last_read_denied = False
            self._last_read_stale = False

        def read_with_version_generation(
            self, path: str, *, observe: bool = True
        ) -> tuple[bytes, int, int | None]:
            out = super().read_with_version_generation(path, observe=observe)
            self._last_read_denied = self._denied and not observe
            self._last_read_stale = self._lapsed and not observe
            return out

    cases = [
        ([(b"c", 5, 7), (b"c", 6, 7)], False, False, HOLD_VERSION_MOVED),
        ([(b"c", 5, 7), (b"c", 5, 8)], False, False, HOLD_GRANT_RECLAIMED),
        ([(b"c", 5, 7), FileNotFoundError()], False, False, HOLD_INPUT_VANISHED),
        ([(b"c", 5, 7), (b"c", 0, 7)], False, False, HOLD_VERSION_UNCONFIRMED),
        ([(b"c", 5, 7), (b"c", 5, None)], False, False, HOLD_GENERATION_UNCONFIRMED),
        # A strict deny: the coordinator refused the re-read, which on the
        # in-process wire also means it reported no generation.
        ([(b"c", 5, 7), (b"c", 5, None)], True, False, HOLD_READ_DENIED),
        # A peer's write-claim preemption: both comparands confirmed AND
        # unchanged, but the re-read was served without a standing grant.
        ([(b"c", 5, 7), (b"c", 5, 7)], False, True, HOLD_GRANT_PREEMPTED),
    ]
    for reads, denied, lapsed, expected_cause in cases:
        with pytest.raises(StaleView) as exc:
            gate(
                _FlaggedStub(reads, denied=denied, lapsed=lapsed),
                "p",
                decide=lambda d: "go",
                effect=lambda x: x,
            )
        assert exc.value.hold_cause == expected_cause, (reads, denied, lapsed)
    # A bare coordinator-raised StaleView carries no cause (uniform shape).
    assert StaleView("peer committed").hold_cause is None


def test_in_process_fence_still_admits_a_clean_pair_after_the_core_move() -> None:
    """The other half of the characterisation: a confirmed, unchanged pair under
    a standing grant still PROCEEDS.

    A move that only ever adds HOLDs is not behaviour-preserving either — an
    over-eager new leg (the content-claim leg is one) would hold every effect in
    the product, and every HOLD assertion above would still pass.
    """
    fired: list[str] = []
    result = gate(
        _StubVolume([(b"cfg", 5, 7), (b"cfg", 5, 7)]),
        "p",
        decide=lambda d: "go",
        effect=lambda x: (fired.append(x), "ok")[1],
    )
    assert result == "ok"
    assert fired == ["go"]
