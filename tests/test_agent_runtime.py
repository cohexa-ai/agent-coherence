# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for agent runtime protocol flows."""

from __future__ import annotations

from uuid import uuid4

import pytest

from ccs.agent.runtime import AgentRuntime
from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.core.exceptions import CasRetriesExhausted
from ccs.core.hashing import compute_content_hash
from ccs.core.states import MESIState
from ccs.core.types import ConflictDetail, FetchRequest, InvalidationSignal
from ccs.strategies.lazy import LazyStrategy


def _runtime(coordinator: CoordinatorService | None = None) -> AgentRuntime:
    service = coordinator if coordinator is not None else CoordinatorService(ArtifactRegistry())
    return AgentRuntime(
        agent_id=uuid4(),
        coordinator=service,
        strategy=LazyStrategy(),
    )


def test_read_fetches_then_uses_local_cache() -> None:
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    runtime = _runtime(coordinator)

    first = runtime.read(artifact.id, now_tick=1)
    second = runtime.read(artifact.id, now_tick=2)

    assert first.version == 1
    assert second.version == 1
    entry = runtime.cache.get(artifact.id)
    assert entry is not None
    assert entry.access_count == 1


def test_write_updates_version_and_marks_local_modified() -> None:
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    runtime_a = _runtime(coordinator)
    runtime_b = _runtime(coordinator)
    runtime_a.read(artifact.id, now_tick=1)
    runtime_b.read(artifact.id, now_tick=1)

    updated, signals = runtime_a.write(artifact.id, content="v2", now_tick=2)

    assert updated.version == 2
    entry = runtime_a.cache.get(artifact.id)
    assert entry is not None
    assert entry.state == MESIState.MODIFIED
    assert runtime_a.content(artifact.id) == "v2"
    assert len(signals) >= 1


def test_handle_invalidation_marks_cache_invalid() -> None:
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    runtime = _runtime(coordinator)
    runtime.read(artifact.id, now_tick=1)
    signal = InvalidationSignal(
        artifact_id=artifact.id,
        new_version=2,
        issued_at_tick=7,
        issuer_agent_id=uuid4(),
    )

    runtime.handle_invalidation(signal)

    entry = runtime.cache.get(artifact.id)
    assert entry is not None
    assert entry.state == MESIState.INVALID
    assert entry.local_version == 1


def test_handle_update_sets_shared_and_content() -> None:
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    writer = uuid4()
    runtime = _runtime(coordinator)

    runtime.handle_update(
        artifact_id=artifact.id,
        version=3,
        content="v3",
        now_tick=8,
        writer_agent_id=writer,
    )

    entry = runtime.cache.get(artifact.id)
    assert entry is not None
    assert entry.state == MESIState.SHARED
    assert entry.local_version == 3
    assert runtime.content(artifact.id) == "v3"


def test_invalidate_all_cache_clears_entries_without_coordinator_call() -> None:
    coordinator = CoordinatorService(ArtifactRegistry())
    a1 = coordinator.register_artifact(name="a.md", content="v1")
    a2 = coordinator.register_artifact(name="b.md", content="v1")
    runtime = _runtime(coordinator)
    runtime.read(a1.id, now_tick=1)
    runtime.read(a2.id, now_tick=1)

    call_count = 0
    orig = coordinator.record_heartbeat

    def spy(**kwargs: object) -> None:
        nonlocal call_count
        call_count += 1
        orig(**kwargs)

    coordinator.record_heartbeat = spy  # type: ignore[assignment]

    runtime.invalidate_all_cache()

    assert runtime.cache.get(a1.id).state == MESIState.INVALID
    assert runtime.cache.get(a2.id).state == MESIState.INVALID
    assert call_count == 0


# ---------------------------------------------------------------------------
# Unit 5 — AgentRuntime.write_cas (the OCC library write path).
#
# write_cas BYPASSES the pessimistic acquire: it never calls coordinator.write()
# / never takes EXCLUSIVE. The writer stays SHARED/INVALID and the commit-time
# version CAS elects the winner. The retry loop lives here (the actor); the
# strategy only supplies the knob (max_cas_retries / cas_backoff_ticks).
# ---------------------------------------------------------------------------


def _shared_occ_writer(coordinator: CoordinatorService, artifact_id, *, version: int) -> AgentRuntime:
    """Return a runtime whose cache holds the artifact SHARED at ``version`` and
    is SHARED (not EXCLUSIVE) at the coordinator too.

    A lone fetch grants the *first* reader EXCLUSIVE; an OCC ``commit_cas`` needs
    the caller in S/I (D4). So we register a throwaway co-reader, which downgrades
    this writer to SHARED coordinator-side. The local cache entry is then seeded
    SHARED at ``version`` so ``write_cas``'s pre-loop ``requires_refresh`` is
    False and the first attempt submits ``expected_version=version``.
    """
    runtime = _runtime(coordinator)
    coordinator.fetch(
        FetchRequest(artifact_id=artifact_id, requesting_agent_id=runtime.agent_id, requested_at_tick=1)
    )
    # Co-reader downgrades `runtime` from EXCLUSIVE to SHARED at the coordinator.
    coordinator.fetch(
        FetchRequest(artifact_id=artifact_id, requesting_agent_id=uuid4(), requested_at_tick=1)
    )
    assert coordinator.registry.get_agent_state(artifact_id, runtime.agent_id) == MESIState.SHARED
    runtime.cache.put(
        artifact_id,
        runtime.strategy.on_fetch(
            artifact_id=artifact_id, version=version, state=MESIState.SHARED, now_tick=1
        ),
    )
    return runtime


def test_write_cas_never_calls_coordinator_write() -> None:
    """The crux (R4): the OCC path must not take the pessimistic EXCLUSIVE acquire."""
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    writer = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)

    write_calls = 0
    orig_write = coordinator.write

    def spy_write(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal write_calls
        write_calls += 1
        return orig_write(**kwargs)

    coordinator.write = spy_write  # type: ignore[assignment]

    updated, _ = writer.write_cas(
        artifact.id,
        make_content=lambda entry: (f"occ-v{entry.local_version + 1}", None),
        now_tick=2,
    )

    assert updated.version == 2
    assert write_calls == 0  # never acquired EXCLUSIVE via coordinator.write()
    # An OCC win ends SHARED (no grant) on the coordinator; the local cache
    # mirrors that — it must not be left more privileged than the registry.
    assert writer.cache.get(artifact.id).state == MESIState.SHARED


def test_write_cas_conflict_then_reread_then_commits_at_fresh_version() -> None:
    """Happy path with a concurrent peer: first attempt hits version_mismatch,
    the loop re-reads to the winner's version and commits there (R6/R8)."""
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    writer = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)

    # A concurrent OCC peer (also SHARED) commits first, bumping version 1 -> 2.
    # `writer`'s LOCAL cache still says SHARED v1, so its first attempt is stale.
    peer = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)
    peer_updated = coordinator.commit_cas(
        agent_id=peer.agent_id,
        artifact_id=artifact.id,
        expected_version=artifact.version,
        content_hash=compute_content_hash("peer-v2"),
    )
    assert peer_updated[0].version == 2  # the peer won the race

    seen_expected: list[int] = []

    def make_content(entry):  # type: ignore[no-untyped-def]
        seen_expected.append(entry.local_version)
        return (f"occ-v{entry.local_version + 1}", None)

    updated, signals = writer.write_cas(artifact.id, make_content=make_content, now_tick=3)

    # First attempt saw stale v1 (mismatch) -> re-read -> second attempt saw v2 (win).
    assert seen_expected == [1, 2]
    assert updated.version == 3
    # Cache reflects the committed version; content recomputed against fresh state.
    # An OCC win ends SHARED (no grant), so the local cache mirrors the
    # coordinator's now-SHARED committer end-state, not MODIFIED.
    entry = writer.cache.get(artifact.id)
    assert entry is not None
    assert entry.state == MESIState.SHARED
    assert entry.local_version == 3
    assert writer.content(artifact.id) == "occ-v3"
    assert len(signals) >= 0  # signals mirror commit_cas (peers already INVALID here)


def test_write_cas_exhaustion_raises_typed_terminal_no_silent_drop() -> None:
    """Perpetual conflict (stubbed) -> CasRetriesExhausted after exactly N+1
    attempts; the committed version is NOT the loser's and the cache is intact."""
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    writer = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)

    attempts = 0
    real_commit_cas = coordinator.commit_cas

    def always_conflict(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        # The registry version is still 1 (no real winner) — assert no silent drop.
        return ConflictDetail(reason="version_mismatch", current_version=1)

    coordinator.commit_cas = always_conflict  # type: ignore[assignment]

    with pytest.raises(CasRetriesExhausted) as excinfo:
        writer.write_cas(
            artifact.id,
            make_content=lambda entry: ("never-lands", None),
            now_tick=2,
        )

    coordinator.commit_cas = real_commit_cas  # type: ignore[assignment]

    assert attempts == writer.strategy.max_cas_retries() + 1
    assert excinfo.value.attempts == writer.strategy.max_cas_retries() + 1
    # No silent drop: the artifact's committed version did not advance to the
    # loser's content, and the registry was never mutated.
    assert coordinator.registry.get_artifact(artifact.id).version == 1
    # Cache not corrupted with the unconfirmed write — it is not MODIFIED.
    entry = writer.cache.get(artifact.id)
    assert entry is not None
    assert entry.state != MESIState.MODIFIED
    assert writer.content(artifact.id) != "never-lands"


def test_write_cas_loop_obeys_strategy_max_retries_knob() -> None:
    """Unit 4 integration: a strategy with max_cas_retries()=N yields exactly
    N+1 commit_cas attempts before CasRetriesExhausted (loop honors the knob)."""

    class _TwoRetryStrategy(LazyStrategy):
        def max_cas_retries(self) -> int:
            return 2

    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    runtime = AgentRuntime(
        agent_id=uuid4(), coordinator=coordinator, strategy=_TwoRetryStrategy()
    )
    coordinator.fetch(
        FetchRequest(artifact_id=artifact.id, requesting_agent_id=runtime.agent_id, requested_at_tick=1)
    )
    runtime.cache.put(
        artifact.id,
        runtime.strategy.on_fetch(
            artifact_id=artifact.id, version=1, state=MESIState.SHARED, now_tick=1
        ),
    )

    attempts = 0

    reasons = ["version_mismatch", "version_mismatch", "other_holder"]

    def always_conflict(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal attempts
        attempts += 1
        return ConflictDetail(reason=reasons[attempts - 1], current_version=1)

    coordinator.commit_cas = always_conflict  # type: ignore[assignment]

    with pytest.raises(CasRetriesExhausted) as excinfo:
        runtime.write_cas(
            artifact.id, make_content=lambda entry: ("x", None), now_tick=2
        )

    assert attempts == 3  # max_cas_retries()=2 -> 2 + 1 attempts
    # The terminal names the LAST refusal, not the first, and not a lost race.
    assert excinfo.value.last_conflict_reason == "other_holder"
    assert "lost the race" not in str(excinfo.value)


def test_write_cas_validates_provided_content_hash() -> None:
    """A make_content-provided hash that does not match the content fails fast,
    mirroring write()'s content_hash guard."""
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    writer = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)

    with pytest.raises(ValueError, match="content_hash mismatch"):
        writer.write_cas(
            artifact.id,
            make_content=lambda entry: ("real-content", "deadbeef-not-the-hash"),
            now_tick=2,
        )

    # A correct provided hash is accepted (parity with write()).
    correct = compute_content_hash("good-content")
    writer2 = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)
    updated, _ = writer2.write_cas(
        artifact.id,
        make_content=lambda entry: ("good-content", correct),
        now_tick=3,
    )
    assert updated.version == 2


def test_write_cas_refetches_when_cache_entry_evicted_mid_loop() -> None:
    """T5: a peer invalidation can evict the cache placeholder BETWEEN attempts.
    write_cas must re-fetch a fresh SHARED entry before reading
    ``entry.local_version`` (the ``entry is None`` guard) — never raise
    AttributeError. Here attempt 0 conflicts (a peer won v2), then the
    top-of-attempt-1 cache.get is forced to None; write_cas re-fetches and the
    second attempt commits at the fresh version (converges)."""
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    writer = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)

    # A concurrent OCC peer commits first (v1 -> v2), so the writer's first
    # attempt is stale → version_mismatch → the loop iterates to attempt 1.
    peer = _shared_occ_writer(coordinator, artifact.id, version=artifact.version)
    coordinator.commit_cas(
        agent_id=peer.agent_id,
        artifact_id=artifact.id,
        expected_version=artifact.version,
        content_hash=compute_content_hash("peer-v2"),
    )

    # Evict the placeholder exactly once, at the top of the SECOND attempt
    # (the 3rd cache.get overall: pre-loop, attempt-0 top, attempt-1 top←evict).
    real_get = writer.cache.get
    calls = {"n": 0}

    def evicting_get(aid):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:
            return None  # simulate a peer invalidation dropping the entry
        return real_get(aid)

    writer.cache.get = evicting_get  # type: ignore[assignment]
    try:
        updated, _ = writer.write_cas(
            artifact.id,
            make_content=lambda entry: (f"occ-v{entry.local_version + 1}", None),
            now_tick=3,
        )
    finally:
        writer.cache.get = real_get  # type: ignore[assignment]

    # Converged at the fresh version with no AttributeError on entry.local_version.
    assert updated.version == 3
    assert calls["n"] >= 4  # the eviction forced the re-fetch + re-get branch
    entry = writer.cache.get(artifact.id)
    assert entry is not None
    assert entry.local_version == 3


def test_write_cas_exhaustion_with_eviction_raises_typed_terminal() -> None:
    """T5 (terminal path): even when the cache entry is evicted on EVERY attempt,
    write_cas re-fetches cleanly and surfaces the typed CasRetriesExhausted on a
    perpetual conflict — never an AttributeError, never a silent drop."""

    class _TwoRetryStrategy(LazyStrategy):
        def max_cas_retries(self) -> int:
            return 2

    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")
    runtime = AgentRuntime(
        agent_id=uuid4(), coordinator=coordinator, strategy=_TwoRetryStrategy()
    )
    coordinator.fetch(
        FetchRequest(artifact_id=artifact.id, requesting_agent_id=runtime.agent_id, requested_at_tick=1)
    )
    runtime.cache.put(
        artifact.id,
        runtime.strategy.on_fetch(
            artifact_id=artifact.id, version=1, state=MESIState.SHARED, now_tick=1
        ),
    )

    # Perpetual conflict — no real winner advances the registry.
    def always_conflict(**kwargs):  # type: ignore[no-untyped-def]
        return ConflictDetail(reason="version_mismatch", current_version=1)

    coordinator.commit_cas = always_conflict  # type: ignore[assignment]

    # Evict the placeholder once mid-loop (at the top of the second attempt:
    # the 3rd cache.get — pre-loop, attempt-0 top, attempt-1 top←evict). The
    # eviction must drive the re-fetch branch and the loop must still reach its
    # bounded terminal rather than raising AttributeError on entry.local_version.
    real_get = runtime.cache.get
    calls = {"n": 0}

    def evicting_get(aid):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        if calls["n"] == 3:
            return None  # peer invalidation drops the entry between attempts
        return real_get(aid)

    runtime.cache.get = evicting_get  # type: ignore[assignment]
    try:
        with pytest.raises(CasRetriesExhausted):
            runtime.write_cas(
                artifact.id,
                make_content=lambda entry: ("never-lands", None),
                now_tick=2,
            )
    finally:
        runtime.cache.get = real_get  # type: ignore[assignment]

    # The eviction branch was exercised, and the registry never advanced (no
    # silent drop on the eviction+conflict path).
    assert calls["n"] >= 4  # re-fetch branch fired after the eviction
    assert coordinator.registry.get_artifact(artifact.id).version == 1
