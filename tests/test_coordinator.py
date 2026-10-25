# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Contract tests for coordinator registry/service operations."""

from __future__ import annotations

from uuid import uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.core.exceptions import CoherenceError
from ccs.core.hashing import compute_content_hash
from ccs.core.states import MESIState, TransientState
from ccs.core.types import ConflictDetail, FetchRequest


def _service() -> CoordinatorService:
    return CoordinatorService(ArtifactRegistry())


def test_fetch_first_holder_gets_exclusive() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent_a = uuid4()

    resp = svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_a, requested_at_tick=1))

    assert resp.state_grant == MESIState.EXCLUSIVE
    assert resp.version == 1
    assert svc.registry.get_agent_state(artifact.id, agent_a) == MESIState.EXCLUSIVE


def test_second_fetch_downgrades_existing_owner_to_shared() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent_a = uuid4()
    agent_b = uuid4()

    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_a, requested_at_tick=1))
    resp_b = svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_b, requested_at_tick=2))

    assert resp_b.state_grant == MESIState.SHARED
    assert svc.registry.get_agent_state(artifact.id, agent_a) == MESIState.SHARED
    assert svc.registry.get_agent_state(artifact.id, agent_b) == MESIState.SHARED


def test_write_invalidates_peers_and_grants_exclusive() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent_a = uuid4()
    agent_b = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_a, requested_at_tick=1))
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_b, requested_at_tick=2))

    signals = svc.write(agent_id=agent_a, artifact_id=artifact.id)

    assert len(signals) == 1
    assert signals[0].issuer_agent_id == agent_a
    assert signals[0].victim_agent_id == agent_b
    assert signals[0].victim_state == MESIState.SHARED
    assert svc.registry.get_agent_state(artifact.id, agent_a) == MESIState.EXCLUSIVE
    assert svc.registry.get_agent_state(artifact.id, agent_b) == MESIState.INVALID


def test_commit_requires_owner_state() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    non_owner = uuid4()
    with pytest.raises(CoherenceError):
        svc.commit(agent_id=non_owner, artifact_id=artifact.id, content="v2")


def test_commit_increments_version_monotonically() -> None:
    svc = _service()
    owner = uuid4()
    peer = uuid4()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=owner, requested_at_tick=1))
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=peer, requested_at_tick=2))
    svc.write(agent_id=owner, artifact_id=artifact.id)

    v2, _ = svc.commit(agent_id=owner, artifact_id=artifact.id, content="v2")
    svc.write(agent_id=owner, artifact_id=artifact.id)
    v3, _ = svc.commit(agent_id=owner, artifact_id=artifact.id, content="v3")

    assert v2.version == 2
    assert v3.version == 3
    assert svc.registry.get_agent_state(artifact.id, owner) == MESIState.MODIFIED
    assert svc.registry.get_agent_state(artifact.id, peer) == MESIState.INVALID


def test_invalidate_marks_agent_invalid() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent, requested_at_tick=1))
    signal = svc.invalidate(
        agent_id=agent,
        artifact_id=artifact.id,
        new_version=2,
        issuer_agent_id=uuid4(),
        issued_at_tick=10,
    )

    assert signal.new_version == 2
    assert svc.registry.get_agent_state(artifact.id, agent) == MESIState.INVALID


def test_upgrade_flow_grants_exclusive_and_invalidates_peer() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    owner = uuid4()
    peer = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=owner, requested_at_tick=1))
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=peer, requested_at_tick=2))

    signals = svc.upgrade(agent_id=owner, artifact_id=artifact.id, issued_at_tick=7)

    assert len(signals) == 1
    assert signals[0].issued_at_tick == 7
    assert svc.registry.get_agent_state(artifact.id, owner) == MESIState.EXCLUSIVE
    assert svc.registry.get_agent_state(artifact.id, peer) == MESIState.INVALID


def test_fetch_raises_coherence_error_when_content_missing() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    svc.registry.set_artifact_and_content(artifact.id, artifact, None)  # type: ignore[arg-type]

    with pytest.raises(CoherenceError):
        svc.fetch(
            FetchRequest(
                artifact_id=artifact.id,
                requesting_agent_id=uuid4(),
                requested_at_tick=1,
            )
        )


def test_write_and_commit_propagate_issued_tick_in_signals() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    owner = uuid4()
    peer = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=owner, requested_at_tick=1))
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=peer, requested_at_tick=2))

    write_signals = svc.write(agent_id=owner, artifact_id=artifact.id, issued_at_tick=11)
    assert write_signals
    assert all(signal.issued_at_tick == 11 for signal in write_signals)

    updated, commit_signals = svc.commit(
        agent_id=owner,
        artifact_id=artifact.id,
        content="v2",
        issued_at_tick=13,
    )
    assert updated.version == 2
    assert all(signal.issued_at_tick == 13 for signal in commit_signals)


def test_delete_returns_signals_for_non_invalid_holders_and_removes_artifact() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent_a = uuid4()
    agent_b = uuid4()
    agent_c = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_a, requested_at_tick=1))
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_b, requested_at_tick=2))
    # agent_c has INVALID state (never fetched)

    signals = svc.delete(agent_id=agent_a, artifact_id=artifact.id, issued_at_tick=5)

    assert len(signals) == 2
    assert all(s.artifact_id == artifact.id for s in signals)
    assert all(s.issued_at_tick == 5 for s in signals)
    assert not svc.registry.has_artifact(artifact.id)


def test_delete_absent_artifact_returns_empty_list() -> None:
    svc = _service()
    absent_id = uuid4()

    signals = svc.delete(agent_id=uuid4(), artifact_id=absent_id)

    assert signals == []


def test_delete_all_invalid_holders_removes_artifact_and_returns_empty() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent_a = uuid4()
    # Register agent state as INVALID explicitly
    svc.registry.set_agent_state(artifact.id, agent_a, MESIState.INVALID)

    signals = svc.delete(agent_id=uuid4(), artifact_id=artifact.id)

    assert signals == []
    assert not svc.registry.has_artifact(artifact.id)


def test_invalidate_returns_none_after_artifact_deleted() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent_a = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_a, requested_at_tick=1))
    svc.delete(agent_id=agent_a, artifact_id=artifact.id)

    result = svc.invalidate(
        agent_id=agent_a,
        artifact_id=artifact.id,
        new_version=1,
        issuer_agent_id=uuid4(),
        issued_at_tick=2,
    )

    assert result is None


def test_remove_artifact_unknown_id_is_silent() -> None:
    registry = ArtifactRegistry()
    registry.remove_artifact(uuid4())  # must not raise


def test_peer_transient_lifecycle_set_on_write_and_cleared_on_invalidate_ack() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    owner = uuid4()
    peer = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=owner, requested_at_tick=1))
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=peer, requested_at_tick=2))

    svc.write(agent_id=owner, artifact_id=artifact.id, issued_at_tick=5)
    assert svc.registry.get_agent_transient(artifact.id, peer) == TransientState.SIA
    assert svc.registry.get_transient_tick(artifact.id, peer) == 5

    svc.invalidate(
        agent_id=peer,
        artifact_id=artifact.id,
        new_version=2,
        issuer_agent_id=owner,
        issued_at_tick=6,
    )
    assert svc.registry.get_agent_transient(artifact.id, peer) is None


# --- Crash-recovery Unit 1: heartbeat + grant-slot bookkeeping ---


def test_record_heartbeat_returns_last_seen_tick() -> None:
    svc = _service()
    agent = uuid4()

    svc.record_heartbeat(agent_id=agent, now_tick=5)

    assert svc.registry.last_heartbeat_tick(agent) == 5


def test_record_heartbeat_returns_max_of_seen_ticks() -> None:
    """R12 monotonicity: out-of-order delivery must not regress the heartbeat."""
    svc = _service()
    agent = uuid4()

    svc.record_heartbeat(agent_id=agent, now_tick=10)
    svc.record_heartbeat(agent_id=agent, now_tick=8)

    assert svc.registry.last_heartbeat_tick(agent) == 10


def test_record_heartbeat_rejects_negative_tick() -> None:
    svc = _service()
    with pytest.raises(ValueError):
        svc.record_heartbeat(agent_id=uuid4(), now_tick=-1)


def test_last_heartbeat_tick_returns_none_when_unknown() -> None:
    svc = _service()
    assert svc.registry.last_heartbeat_tick(uuid4()) is None


def test_record_heartbeat_does_not_emit_state_log_entry() -> None:
    """R5 byte-identity: heartbeat must NOT emit log entries or bump _seq."""
    entries: list[dict] = []
    registry = ArtifactRegistry(state_log=entries.append, instance_id="inst-1")
    svc = CoordinatorService(registry)
    seq_before = registry._seq

    svc.record_heartbeat(agent_id=uuid4(), now_tick=42)

    assert entries == []
    assert registry._seq == seq_before


def test_invalid_to_exclusive_populates_granted_at_tick() -> None:
    svc = _service()
    agent = uuid4()
    artifact = svc.register_artifact(name="plan.md", content="v1")

    svc.registry.set_agent_state(
        artifact.id, agent, MESIState.EXCLUSIVE, trigger="test", tick=7
    )

    assert svc.registry.granted_at_tick(agent, artifact.id) == 7


def test_exclusive_to_modified_preserves_granted_at_tick() -> None:
    svc = _service()
    agent = uuid4()
    artifact = svc.register_artifact(name="plan.md", content="v1")

    svc.registry.set_agent_state(
        artifact.id, agent, MESIState.EXCLUSIVE, trigger="test", tick=3
    )
    svc.registry.set_agent_state(
        artifact.id, agent, MESIState.MODIFIED, trigger="test", tick=9
    )

    # E→M stays within M∪E — the original grant tick must be preserved (R8 diagnostic context).
    assert svc.registry.granted_at_tick(agent, artifact.id) == 3


def test_modified_to_invalid_clears_granted_at_tick() -> None:
    svc = _service()
    agent = uuid4()
    artifact = svc.register_artifact(name="plan.md", content="v1")

    svc.registry.set_agent_state(
        artifact.id, agent, MESIState.MODIFIED, trigger="test", tick=3
    )
    svc.registry.set_agent_state(
        artifact.id, agent, MESIState.INVALID, trigger="test", tick=9
    )

    assert svc.registry.granted_at_tick(agent, artifact.id) is None


def test_reclamation_slot_survives_shared_acquire_then_clears_on_exclusive() -> None:
    """jessieibarra path: SHARED re-fetch by a reclaimed agent must NOT clear the slot.
    Slot clears ONLY on M∪E re-acquire."""
    svc = _service()
    agent = uuid4()
    artifact = svc.register_artifact(name="plan.md", content="v1")

    # Seed a reclamation slot via the helper (simulates Unit 2's sweep result).
    svc.registry.record_last_reclamation(agent, artifact.id, "reclaim_max_hold", 200)
    assert svc.registry.get_last_reclamation(agent, artifact.id) == ("reclaim_max_hold", 200)

    # INVALID → SHARED: slot must SURVIVE.
    svc.registry.set_agent_state(
        artifact.id, agent, MESIState.SHARED, trigger="fetch", tick=210
    )
    assert svc.registry.get_last_reclamation(agent, artifact.id) == ("reclaim_max_hold", 200)

    # SHARED → EXCLUSIVE (M∪E acquire, prev not in M∪E): slot now clears.
    svc.registry.set_agent_state(
        artifact.id, agent, MESIState.EXCLUSIVE, trigger="write", tick=220
    )
    assert svc.registry.get_last_reclamation(agent, artifact.id) is None


def test_reclamation_slot_untouched_by_set_agent_transient() -> None:
    """Slot-clear bookkeeping fires only on stable transitions through set_agent_state."""
    svc = _service()
    agent = uuid4()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    svc.registry.record_last_reclamation(agent, artifact.id, "reclaim_heartbeat", 100)

    svc.registry.set_agent_transient(
        artifact.id, agent, TransientState.IED, entered_tick=105
    )
    assert svc.registry.get_last_reclamation(agent, artifact.id) == ("reclaim_heartbeat", 100)

    svc.registry.clear_agent_transient(artifact.id, agent)
    assert svc.registry.get_last_reclamation(agent, artifact.id) == ("reclaim_heartbeat", 100)


def test_set_agent_state_bookkeeping_emits_no_extra_log_entries() -> None:
    """R5 byte-identity: dict mutations on set_agent_state must NOT add state-log entries."""
    entries: list[dict] = []
    registry = ArtifactRegistry(state_log=entries.append, instance_id="inst-1")
    svc = CoordinatorService(registry)
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    entries_before = len(entries)
    seq_before = registry._seq

    registry.set_agent_state(artifact.id, agent, MESIState.EXCLUSIVE, trigger="t", tick=1)
    registry.set_agent_state(artifact.id, agent, MESIState.MODIFIED, trigger="t", tick=2)
    registry.set_agent_state(artifact.id, agent, MESIState.INVALID, trigger="t", tick=3)

    # Exactly 3 log entries total; bookkeeping itself produced none.
    assert len(entries) - entries_before == 3
    assert registry._seq - seq_before == 3


# ---- Unit 3: CrashRecoveryConfig + composition fail-fast --------------------

from ccs.coordinator.service import (  # noqa: E402
    CrashRecoveryConfig,
    validate_crash_recovery_config,
)
from ccs.strategies.lazy import LazyStrategy  # noqa: E402
from ccs.strategies.lease import LeaseStrategy  # noqa: E402


def test_crash_recovery_config_defaults_are_safe() -> None:
    """v0.9.0: bare CrashRecoveryConfig() is enabled-by-default with the
    retuned production thresholds (R6), and the first construction per process
    emits the one-shot transitional RuntimeWarning naming the v0.9.0 change.
    """
    # Reset the module-level emit-once flag so this test sees the warning
    # regardless of test ordering.
    from ccs.coordinator import service as _service_mod
    _service_mod._V090_FIRST_USE_WARNED = False
    with pytest.warns(RuntimeWarning, match="default changed in v0.9.0"):
        cfg = CrashRecoveryConfig()
    assert cfg.enabled is True
    assert cfg.heartbeat_timeout_ticks == 120
    assert cfg.max_hold_ticks == 900


def testvalidate_crash_recovery_config_disabled_always_accepts() -> None:
    # Even an obviously bad max_hold_ticks vs ttl is fine when disabled.
    cfg = CrashRecoveryConfig(enabled=False, max_hold_ticks=10)
    validate_crash_recovery_config(cfg, LeaseStrategy(ttl_ticks=300))


def testvalidate_crash_recovery_config_rejects_equal_ttl() -> None:
    cfg = CrashRecoveryConfig(enabled=True, max_hold_ticks=300)
    with pytest.raises(ValueError, match="max_hold_ticks=300"):
        validate_crash_recovery_config(cfg, LeaseStrategy(ttl_ticks=300))


def testvalidate_crash_recovery_config_rejects_below_ttl() -> None:
    cfg = CrashRecoveryConfig(enabled=True, max_hold_ticks=100)
    with pytest.raises(ValueError):
        validate_crash_recovery_config(cfg, LeaseStrategy(ttl_ticks=300))


def testvalidate_crash_recovery_config_accepts_above_ttl() -> None:
    cfg = CrashRecoveryConfig(enabled=True, max_hold_ticks=300)
    validate_crash_recovery_config(cfg, LeaseStrategy(ttl_ticks=200))


def testvalidate_crash_recovery_config_skips_non_lease_strategy() -> None:
    """Strategies without ttl_ticks (lazy/eager/etc.) silent-accept (R11 skip rule)."""
    cfg = CrashRecoveryConfig(enabled=True, max_hold_ticks=300)
    # LazyStrategy exposes no ttl_ticks attribute.
    validate_crash_recovery_config(cfg, LazyStrategy())


# Review fix ADV-03: ttl_ticks=0 was silently routed to the warning path
# (with misleading "non-integer" text) and skipped R11 entirely. Now it's
# treated like any other int ttl and validated against max_hold_ticks.


class _ZeroTTLStrategy:
    """Test fixture: strategy that exposes ttl_ticks=0 (degenerate case)."""

    ttl_ticks = 0


class _StringTTLStrategy:
    """Test fixture: strategy with a non-integer ttl_ticks attribute."""

    ttl_ticks = "300"


def testvalidate_crash_recovery_config_zero_ttl_passes_when_max_hold_positive() -> None:
    """ADV-03: ttl_ticks=0 with max_hold_ticks=1 passes R11 (1 > 0). No warning."""
    import warnings as _w  # local import keeps top-of-file unchanged
    cfg = CrashRecoveryConfig(enabled=True, max_hold_ticks=1)
    with _w.catch_warnings():
        _w.simplefilter("error")  # Any warning fails the test.
        validate_crash_recovery_config(cfg, _ZeroTTLStrategy())


def testvalidate_crash_recovery_config_zero_ttl_rejects_zero_max_hold() -> None:
    """ADV-03: ttl_ticks=0 with max_hold_ticks=0 IS a R11 violation (0 not > 0)."""
    # max_hold_ticks=0 is normally rejected by the int-validator at sweep time,
    # but the composition rule must still flag this combination at construction.
    cfg = CrashRecoveryConfig(enabled=True, max_hold_ticks=0)
    with pytest.raises(ValueError, match="max_hold_ticks=0"):
        validate_crash_recovery_config(cfg, _ZeroTTLStrategy())


def testvalidate_crash_recovery_config_non_integer_ttl_warns() -> None:
    """ADV-03: only genuinely non-integer ttl_ticks (e.g., string) triggers the warn path."""
    import warnings as _w
    cfg = CrashRecoveryConfig(enabled=True, max_hold_ticks=300)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        validate_crash_recovery_config(cfg, _StringTTLStrategy())
    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1
    assert "non-integer" in str(runtime_warnings[0].message)
    assert "'300'" in str(runtime_warnings[0].message) or '"300"' in str(runtime_warnings[0].message)


# Review fix COR-01 / REL-01: bookkeeping must run before the log emit so a
# state_log raise leaves state_by_agent and granted_at_tick_by_agent
# consistent. Without this fix, max-hold reclamation silently misses live
# agents whose granted_at_tick slot was never written.

from ccs.core.types import Artifact  # noqa: E402


def test_set_agent_state_failed_log_emit_keeps_bookkeeping_consistent() -> None:
    """COR-01: log emit raise must NOT leave state_by_agent inconsistent with bookkeeping.

    With the fix, bookkeeping runs BEFORE the log emit. So if the log raises:
    - state_by_agent shows the new state (pre-existing semantics — not changed)
    - granted_at_tick_by_agent has the M∪E entry (NEW — was missing before fix)
    - The next sweep can correctly evaluate max-hold for the agent.
    """
    seen = [0]

    def flaky_log(entry):
        seen[0] += 1
        # Fail on every call — first M∪E acquire's log emit raises.
        raise RuntimeError("simulated log emit failure")

    registry = ArtifactRegistry(
        state_log=flaky_log, agent_names=None, instance_id="test-instance"
    )
    artifact = Artifact(name="x", version=1)
    registry.register_artifact(artifact, content="v1")
    agent_id = uuid4()

    with pytest.raises(RuntimeError, match="simulated log emit failure"):
        registry.set_agent_state(
            artifact.id, agent_id, MESIState.EXCLUSIVE, trigger="write", tick=42
        )

    # State mutation survived (pre-existing behavior, unchanged):
    assert registry.get_agent_state(artifact.id, agent_id) == MESIState.EXCLUSIVE
    # Bookkeeping is now CONSISTENT with that state — granted_at_tick is recorded:
    assert registry.granted_at_tick(agent_id, artifact.id) == 42


def test_set_agent_state_failed_log_emit_consistent_for_me_exit_too() -> None:
    """COR-01 mirror case: log raise on M∪E exit still clears the slot."""
    fail_on_call = 3
    seen = [0]

    def flaky_log(entry):
        seen[0] += 1
        if seen[0] >= fail_on_call:
            raise RuntimeError("simulated log emit failure")

    registry = ArtifactRegistry(
        state_log=flaky_log, agent_names=None, instance_id="test-instance"
    )
    artifact = Artifact(name="x", version=1)
    registry.register_artifact(artifact, content="v1")
    agent_id = uuid4()

    # Calls 1 & 2 succeed (acquire EXCLUSIVE, internal step). Call 3 fails.
    registry.set_agent_state(
        artifact.id, agent_id, MESIState.EXCLUSIVE, trigger="write", tick=10
    )
    registry.set_agent_state(
        artifact.id, agent_id, MESIState.MODIFIED, trigger="commit", tick=11
    )
    # granted_at_tick survives the M↔E transition (preserved at acquire).
    assert registry.granted_at_tick(agent_id, artifact.id) == 10

    with pytest.raises(RuntimeError, match="simulated log emit failure"):
        registry.set_agent_state(
            artifact.id, agent_id, MESIState.INVALID, trigger="reclaim", tick=20
        )
    # Bookkeeping cleanup ran even though log emit raised.
    assert registry.get_agent_state(artifact.id, agent_id) == MESIState.INVALID
    assert registry.granted_at_tick(agent_id, artifact.id) is None


# ---- v0.9.0 C-flip: default flip + transitional warning ---------------------
#
# See docs/plans/2026-05-28-001-feat-c-flip-crash-recovery-default-on-plan.md
# Units 5 + 6. The v0.8.3 sentinel + DeprecationWarning are gone; the default
# is now enabled=True with retuned 120/900 thresholds. The first
# CrashRecoveryConfig construction per process emits a one-shot transitional
# RuntimeWarning. Tests below assert the flip, the retune, and the warning.


import warnings as _warnings  # noqa: E402

from ccs.coordinator import service as _service_module  # noqa: E402


@pytest.fixture
def reset_v090_first_use_flag():
    """Reset the module-level emit-once flag between tests.

    Without this, the first test to construct CrashRecoveryConfig() would set
    _V090_FIRST_USE_WARNED=True for the entire pytest session and later tests
    asserting the transitional warning would see zero warnings (false negatives).
    """
    _service_module._V090_FIRST_USE_WARNED = False
    yield
    _service_module._V090_FIRST_USE_WARNED = False


class TestCrashRecoveryV090TransitionalWarning:
    """Unit 5 — first CrashRecoveryConfig() per process emits one RuntimeWarning."""

    def test_first_construction_emits_runtime_warning(
        self, reset_v090_first_use_flag
    ) -> None:
        """Happy path: first construction in a fresh process emits one warning."""
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            CrashRecoveryConfig()
        runtime_warnings = [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 1
        assert "default changed in v0.9.0" in str(runtime_warnings[0].message)

    def test_warning_message_names_both_silence_paths(
        self, reset_v090_first_use_flag
    ) -> None:
        """Warning prose must give users BOTH the opt-in and opt-out recipes."""
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            CrashRecoveryConfig()
        # Filter to RuntimeWarning before indexing: an unrelated warning firing
        # first must not let the message assertions pass on the wrong text. The
        # length guard also turns a dropped emission into an informative
        # AssertionError rather than an IndexError.
        runtime_warnings = [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) >= 1, (
            "first CrashRecoveryConfig() must emit a RuntimeWarning; got 0"
        )
        msg = str(runtime_warnings[0].message)
        assert "enabled=True" in msg, "warning must name the recommended opt-in"
        assert "enabled=False" in msg, "warning must name the explicit opt-out"
        assert "v0.9.0" in msg, "warning must name the release that changed the default"

    def test_default_is_enabled_true(self, reset_v090_first_use_flag) -> None:
        """The flip: bare construction now yields enabled=True (was False)."""
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            cfg = CrashRecoveryConfig()
        assert cfg.enabled is True

    def test_emit_once_dedupes_consecutive_constructions(
        self, reset_v090_first_use_flag
    ) -> None:
        """Module-level flag means three constructions emit exactly ONE warning."""
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            CrashRecoveryConfig()
            CrashRecoveryConfig()
            CrashRecoveryConfig()
        runtime_warnings = [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 1, (
            f"expected 1 warning across three constructions, "
            f"got {len(runtime_warnings)}"
        )

    def test_explicit_false_still_warns_on_first_and_opts_out(
        self, reset_v090_first_use_flag
    ) -> None:
        """Explicit enabled=False opts out of the sweep, but the transitional
        warning still fires on the first construction — it announces the default
        change, not the user's choice (per Unit 5 edge case)."""
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            cfg = CrashRecoveryConfig(enabled=False)
        runtime_warnings = [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 1
        assert cfg.enabled is False

    def test_explicit_true_still_warns_on_first(
        self, reset_v090_first_use_flag
    ) -> None:
        """Symmetric to the explicit-False case: explicit enabled=True as the
        first construction ALSO fires the transitional warning (it announces the
        default change regardless of the value passed). Regression-gates the
        'fires on first construction regardless of how enabled was supplied'
        contract so a future `if enabled is not True` short-circuit is caught."""
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            cfg = CrashRecoveryConfig(enabled=True)
        runtime_warnings = [
            w for w in caught if issubclass(w.category, RuntimeWarning)
        ]
        assert len(runtime_warnings) == 1
        assert cfg.enabled is True

    def test_no_deprecation_warning_remains(
        self, reset_v090_first_use_flag
    ) -> None:
        """The v0.8.3 DeprecationWarning is fully removed in v0.9.0."""
        with _warnings.catch_warnings(record=True) as caught:
            _warnings.simplefilter("always")
            CrashRecoveryConfig()
            CrashRecoveryConfig(enabled=True)
            CrashRecoveryConfig(enabled=False)
        assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []

    def test_dataclass_remains_frozen(self, reset_v090_first_use_flag) -> None:
        """frozen=True invariant preserved through the warning mechanism."""
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            cfg = CrashRecoveryConfig()
        # Direct assignment must still raise FrozenInstanceError.
        with pytest.raises(Exception) as exc_info:
            cfg.enabled = False  # type: ignore[misc]
        assert "frozen" in str(exc_info.value).lower() or "FrozenInstanceError" in type(
            exc_info.value
        ).__name__


class TestCrashRecoveryDefaultsComposition:
    """Unit 6 — retuned defaults (120/900), safety floor, and same-PR tripwire."""

    def _bare(self) -> CrashRecoveryConfig:
        """Construct a default config with the transitional warning suppressed."""
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            return CrashRecoveryConfig()

    def test_defaults_are_retuned(self, reset_v090_first_use_flag) -> None:
        """R6: production-realistic defaults replace the v0.8.x sim-anchors."""
        cfg = self._bare()
        assert cfg.heartbeat_timeout_ticks == 120
        assert cfg.max_hold_ticks == 900

    def test_default_composition_does_not_raise(
        self, reset_v090_first_use_flag
    ) -> None:
        """Enabled-by-default config composes with a 300-tick lease (900 > 300)."""
        validate_crash_recovery_config(self._bare(), LeaseStrategy(ttl_ticks=300))

    def test_safety_floor_regression_gate(self, reset_v090_first_use_flag) -> None:
        """Hard floor governing all future tuning patches — distinct from the
        chosen 120/900 calibration: heartbeat_timeout_ticks >= 60 and
        max_hold_ticks > 300."""
        cfg = self._bare()
        assert cfg.heartbeat_timeout_ticks >= 60
        assert cfg.max_hold_ticks > 300

    def test_enabled_implies_heartbeat_floor_tripwire(
        self, reset_v090_first_use_flag
    ) -> None:
        """ADV-11 tripwire: a default config that is enabled MUST carry a
        heartbeat floor >= 60. Surfaces immediately if a future patch flips the
        default on without the Unit 6 retune (the aggressive-false-reclaim trap
        the Units 5+6 same-PR constraint guards against)."""
        cfg = self._bare()
        # Unconditional: the v0.9.0 default IS enabled, and an enabled default
        # MUST carry the heartbeat floor. Asserting both (not guarding the floor
        # behind `if cfg.enabled`) means a future patch that flips the default
        # back to disabled trips this test instead of passing vacuously.
        assert cfg.enabled is True
        assert cfg.heartbeat_timeout_ticks >= 60

    def test_old_v08x_values_still_settable(
        self, reset_v090_first_use_flag
    ) -> None:
        """Users who want the old sim-anchor thresholds can still pass them."""
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", RuntimeWarning)
            cfg = CrashRecoveryConfig(heartbeat_timeout_ticks=10, max_hold_ticks=1000)
        assert cfg.heartbeat_timeout_ticks == 10
        assert cfg.max_hold_ticks == 1000


# ---- v0.8.3 sentinel robustness (ADV-01 / ADV-03) — removed in v0.9.0 -------
#
# The falsy-sentinel mechanism (and its importlib.reload / subclass-without-
# super() robustness tests) is gone now that ``enabled`` is a plain bool field
# defaulting to True. Nothing replaces it: there is no sentinel to misread.


# ---- v0.9.0 transitional-warning emit-once under contention (ADV-02 lineage) -


class TestConcurrentFirstUseEmitOnce:
    """The _V090_FIRST_USE_LOCK keeps emit-once intact under thread contention
    (free-threaded Python 3.13+ removes the GIL that made the naive check safe).
    """

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_concurrent_construction_emits_one_warning(
        self, reset_v090_first_use_flag
    ) -> None:
        """Eight threads constructing configs simultaneously emit exactly one
        transitional RuntimeWarning. ``warnings.catch_warnings`` mutates global
        filter state and is not thread-safe, so we count via a lock-protected
        shim installed over ``warnings.warn`` (which the service module calls).
        The filterwarnings mark — applied before threads start — suppresses the
        single real emission so it does not escape to stderr.
        """
        import threading

        count_lock = threading.Lock()
        emissions: list[object] = []
        original_warn = _warnings.warn

        def _counting_warn(message, *args, **kwargs):  # type: ignore[no-untyped-def]
            if isinstance(message, str) and "default changed in v0.9.0" in message:
                with count_lock:
                    emissions.append(message)
            return original_warn(message, *args, **kwargs)

        thread_count = 8
        barrier = threading.Barrier(thread_count)

        def worker() -> None:
            barrier.wait()  # release all threads together to maximize contention
            CrashRecoveryConfig()

        _warnings.warn = _counting_warn  # type: ignore[assignment]
        try:
            threads = [threading.Thread(target=worker) for _ in range(thread_count)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            _warnings.warn = original_warn  # type: ignore[assignment]

        assert len(emissions) == 1, (
            f"expected exactly one transitional warning across {thread_count} "
            f"concurrent constructions, got {len(emissions)}"
        )


# ---------------------------------------------------------------------------
# Unit 3 — CoordinatorService.commit_cas (OCC service-level orchestration).
#
# The OCC writer reads (→ SHARED) → computes → commit_cas from S/I; it never
# takes EXCLUSIVE via write(). Two SHARED peers are the canonical setup: a
# matching version + no *pessimistic* M/E holder → WIN; a stale version →
# version_mismatch; a concurrent pessimistic E/M holder → other_holder; an
# over-shot version → CoherenceError (corruption).
# ---------------------------------------------------------------------------


def _two_shared_holders(svc: CoordinatorService):
    """Register an artifact and fetch it from two agents so both end SHARED.

    Returns ``(artifact, agent_a, agent_b)``. Both agents are SHARED at the
    artifact's current version — the eligible (S/I) starting point for an OCC
    ``commit_cas`` where the *other* holder is not a pessimistic M/E writer.
    """
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent_a = uuid4()
    agent_b = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_a, requested_at_tick=1))
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent_b, requested_at_tick=2))
    assert svc.registry.get_agent_state(artifact.id, agent_a) == MESIState.SHARED
    assert svc.registry.get_agent_state(artifact.id, agent_b) == MESIState.SHARED
    return artifact, agent_a, agent_b


def test_commit_cas_happy_path_commits_and_invalidates_peers() -> None:
    svc = _service()
    artifact, writer, peer = _two_shared_holders(svc)

    updated, signals = svc.commit_cas(
        agent_id=writer,
        artifact_id=artifact.id,
        expected_version=artifact.version,
        content_hash=compute_content_hash("v2"),
        issued_at_tick=7,
    )

    # Version advanced N -> N+1.
    assert updated.version == artifact.version + 1
    # Committer ends SHARED (an OCC writer holds no grant); the SHARED peer was
    # invalidated.
    assert svc.registry.get_agent_state(artifact.id, writer) == MESIState.SHARED
    assert svc.registry.get_agent_state(artifact.id, peer) == MESIState.INVALID
    # Signal shape mirrors commit(): one signal per invalidated peer, carrying
    # the new version, issued tick, and the committer as issuer.
    assert len(signals) == 1
    assert signals[0].artifact_id == artifact.id
    assert signals[0].new_version == updated.version
    assert signals[0].issued_at_tick == 7
    assert signals[0].issuer_agent_id == writer


def test_commit_cas_is_repeatable_for_same_agent_no_d4_rejection() -> None:
    """FIX 3: a winning commit_cas ends the committer SHARED (not MODIFIED), so
    the SAME agent can commit_cas AGAIN on the same artifact without tripping the
    D4 precondition (which rejects M/E callers). Two back-to-back wins by one
    agent: version 1→2→3, no CoherenceError. Before the fix the sticky MODIFIED
    grant made the second call hard-fail with 'use commit()'.
    """
    svc = _service()
    artifact, writer, _peer = _two_shared_holders(svc)
    assert artifact.version == 1

    # First OCC win → version 2, committer ends SHARED (the fix).
    first, _ = svc.commit_cas(
        agent_id=writer,
        artifact_id=artifact.id,
        expected_version=1,
        content_hash=compute_content_hash("v2"),
        issued_at_tick=5,
    )
    assert first.version == 2
    assert svc.registry.get_agent_state(artifact.id, writer) == MESIState.SHARED

    # Second OCC win by the SAME agent at the new version — must NOT be rejected
    # by D4 (SHARED is OCC-eligible) and must win again.
    second, _ = svc.commit_cas(
        agent_id=writer,
        artifact_id=artifact.id,
        expected_version=2,
        content_hash=compute_content_hash("v3"),
        issued_at_tick=6,
    )
    assert second.version == 3
    assert svc.registry.get_agent_state(artifact.id, writer) == MESIState.SHARED


def test_commit_cas_version_mismatch_returns_conflict_no_mutation() -> None:
    svc = _service()
    artifact, first, second = _two_shared_holders(svc)

    # First OCC writer wins → version bumps to 2, `second` invalidated.
    updated, _ = svc.commit_cas(
        agent_id=first,
        artifact_id=artifact.id,
        expected_version=artifact.version,
        content_hash=compute_content_hash("v2"),
    )
    assert updated.version == 2

    # `second` still holds the stale expected_version (1) → version_mismatch.
    result = svc.commit_cas(
        agent_id=second,
        artifact_id=artifact.id,
        expected_version=artifact.version,  # stale (==1, current is 2)
        content_hash=compute_content_hash("v2-stale"),
    )
    assert isinstance(result, ConflictDetail)
    assert result.reason == "version_mismatch"
    assert result.current_version == 2
    # No mutation: version unchanged, winner still SHARED (OCC writer holds no
    # grant), loser still INVALID.
    assert svc.registry.get_artifact(artifact.id).version == 2
    assert svc.registry.get_agent_state(artifact.id, first) == MESIState.SHARED
    assert svc.registry.get_agent_state(artifact.id, second) == MESIState.INVALID


def test_commit_cas_other_holder_returns_conflict_no_mutation() -> None:
    svc = _service()
    artifact, occ_writer, pessimist = _two_shared_holders(svc)

    # A pessimistic peer acquires EXCLUSIVE via write() (version unchanged at 1);
    # this also invalidates the OCC writer. write() leaves the invalidated peer
    # mid-transient until it ACKs, so apply the invalidation ACK to land the OCC
    # writer cleanly INVALID (S/I-eligible, no transient) — the realistic state
    # of a writer that observed the invalidation.
    svc.write(agent_id=pessimist, artifact_id=artifact.id)
    svc.invalidate(
        agent_id=occ_writer,
        artifact_id=artifact.id,
        new_version=artifact.version,
        issuer_agent_id=pessimist,
        issued_at_tick=4,
    )
    assert svc.registry.get_agent_state(artifact.id, pessimist) == MESIState.EXCLUSIVE
    assert svc.registry.get_agent_state(artifact.id, occ_writer) == MESIState.INVALID
    assert svc.registry.get_agent_transient(artifact.id, occ_writer) is None

    # Version still matches (write does not bump), but a pessimistic E holder is
    # present → other_holder (distinct from version_mismatch).
    result = svc.commit_cas(
        agent_id=occ_writer,
        artifact_id=artifact.id,
        expected_version=artifact.version,  # ==1, current is still 1
        content_hash=compute_content_hash("v2"),
    )
    assert isinstance(result, ConflictDetail)
    assert result.reason == "other_holder"
    assert result.current_version == 1
    # No mutation: version unchanged, pessimist keeps EXCLUSIVE.
    assert svc.registry.get_artifact(artifact.id).version == 1
    assert svc.registry.get_agent_state(artifact.id, pessimist) == MESIState.EXCLUSIVE


def test_commit_cas_expected_greater_than_current_raises_coherence_error() -> None:
    svc = _service()
    artifact, writer, _peer = _two_shared_holders(svc)

    # expected_version > current → corruption / multi-coordinator violation.
    with pytest.raises(CoherenceError, match="corruption"):
        svc.commit_cas(
            agent_id=writer,
            artifact_id=artifact.id,
            expected_version=artifact.version + 5,
            content_hash=compute_content_hash("v2"),
        )
    # No mutation.
    assert svc.registry.get_artifact(artifact.id).version == 1


def test_commit_cas_rejects_modified_or_exclusive_caller_pointing_to_commit() -> None:
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    owner = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=owner, requested_at_tick=1))
    # Pessimistic acquire → EXCLUSIVE.
    svc.write(agent_id=owner, artifact_id=artifact.id)
    assert svc.registry.get_agent_state(artifact.id, owner) == MESIState.EXCLUSIVE

    # EXCLUSIVE holder must use plain commit() (D4) — error points there.
    with pytest.raises(CoherenceError, match="commit"):
        svc.commit_cas(
            agent_id=owner,
            artifact_id=artifact.id,
            expected_version=artifact.version,
            content_hash=compute_content_hash("v2"),
        )
    # And the MODIFIED case rejects identically.
    svc.commit(agent_id=owner, artifact_id=artifact.id, content="v2")
    assert svc.registry.get_agent_state(artifact.id, owner) == MESIState.MODIFIED
    with pytest.raises(CoherenceError, match="commit"):
        svc.commit_cas(
            agent_id=owner,
            artifact_id=artifact.id,
            expected_version=2,
            content_hash=compute_content_hash("v3"),
        )


def test_commit_cas_rejects_caller_in_transient_state() -> None:
    svc = _service()
    artifact, writer, _peer = _two_shared_holders(svc)
    # Force the OCC writer mid-transient (e.g. ISG) — not eligible to commit.
    svc.registry.set_agent_transient(
        artifact.id, writer, TransientState.ISG, entered_tick=3
    )

    with pytest.raises(CoherenceError, match="transient"):
        svc.commit_cas(
            agent_id=writer,
            artifact_id=artifact.id,
            expected_version=artifact.version,
            content_hash=compute_content_hash("v2"),
        )
    # No mutation.
    assert svc.registry.get_artifact(artifact.id).version == 1


def test_commit_cas_artifact_not_found_raises_coherence_error() -> None:
    svc = _service()
    missing = uuid4()

    with pytest.raises(CoherenceError, match="artifact_not_found"):
        svc.commit_cas(
            agent_id=uuid4(),
            artifact_id=missing,
            expected_version=1,
            content_hash=compute_content_hash("v2"),
        )


def test_commit_cas_win_actually_invalidates_peers_in_registry() -> None:
    """Integration: on a winning CAS, peers are INVALID in the registry itself,
    not merely named in the returned signal list."""
    svc = _service()
    artifact = svc.register_artifact(name="plan.md", content="v1")
    writer = uuid4()
    peer_b = uuid4()
    peer_c = uuid4()
    for tick, agent in enumerate((writer, peer_b, peer_c), start=1):
        svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent, requested_at_tick=tick))
    # Three SHARED holders.
    assert svc.registry.get_agent_state(artifact.id, writer) == MESIState.SHARED

    updated, signals = svc.commit_cas(
        agent_id=writer,
        artifact_id=artifact.id,
        expected_version=artifact.version,
        content_hash=compute_content_hash("v2"),
    )

    # Both peers are actually INVALID in the registry's state map.
    state_map = svc.registry.get_state_map(artifact.id)
    assert state_map[peer_b] == MESIState.INVALID
    assert state_map[peer_c] == MESIState.INVALID
    # The OCC committer ends SHARED (it holds no grant).
    assert state_map[writer] == MESIState.SHARED
    # And the signal list covers exactly the invalidated peers.
    assert len(signals) == 2
    assert {s.issuer_agent_id for s in signals} == {writer}
    assert all(s.new_version == updated.version for s in signals)
    # Single-writer invariant holds afterward (no exception from the service).
    svc._validate_single_writer(artifact.id)
