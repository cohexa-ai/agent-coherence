# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``invalid_reclamations``: the batched reclaim-cause read behind /status (#195).

Run against both registries through the shared ``registry`` fixture, so the
in-memory and sqlite arms answer the same read identically. The sweep and the
releases are driven through ``CoordinatorService`` rather than by writing the
slot directly, so what is asserted is what the shipped write path leaves.
"""

from __future__ import annotations

from uuid import uuid4

from ccs.coordinator.service import CoordinatorService
from ccs.core.states import MESIState
from ccs.core.types import FetchRequest


def _held(registry):
    svc = CoordinatorService(registry)
    artifact = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent, requested_at_tick=0))
    assert registry.get_agent_state(artifact.id, agent) == MESIState.EXCLUSIVE
    return svc, artifact, agent


def test_empty_registry_reports_nothing(registry) -> None:
    assert registry.invalid_reclamations() == {}


def test_heartbeat_reclaim_is_reported(registry) -> None:
    svc, artifact, agent = _held(registry)
    svc.record_heartbeat(agent_id=agent, now_tick=0)
    assert svc.enforce_stable_grant_timeouts(
        current_tick=100, heartbeat_timeout_ticks=10, max_hold_ticks=10_000
    ) == 1
    assert registry.invalid_reclamations() == {
        artifact.id: {agent: ("reclaim_heartbeat", 100)}
    }


def test_max_hold_reclaim_is_reported(registry) -> None:
    svc, artifact, agent = _held(registry)
    svc.record_heartbeat(agent_id=agent, now_tick=500)
    svc.enforce_stable_grant_timeouts(
        current_tick=500, heartbeat_timeout_ticks=10_000, max_hold_ticks=10
    )
    assert registry.invalid_reclamations() == {
        artifact.id: {agent: ("reclaim_max_hold", 500)}
    }


def test_voluntary_release_is_not_reported(registry) -> None:
    svc, artifact, agent = _held(registry)
    svc.invalidate(
        agent_id=agent,
        artifact_id=artifact.id,
        new_version=artifact.version,
        issuer_agent_id=agent,
        issued_at_tick=5,
    )
    assert registry.get_agent_state(artifact.id, agent) == MESIState.INVALID
    assert registry.invalid_reclamations() == {}


def test_shared_reread_is_excluded_and_reacquire_clears(registry) -> None:
    svc, artifact, agent = _held(registry)
    peer = uuid4()
    svc.record_heartbeat(agent_id=agent, now_tick=0)
    svc.enforce_stable_grant_timeouts(
        current_tick=100, heartbeat_timeout_ticks=10, max_hold_ticks=10_000
    )
    # A peer takes the write grant, so the ex-holder's re-read is SHARED.
    svc.write(agent_id=peer, artifact_id=artifact.id, issued_at_tick=101)
    svc.fetch(FetchRequest(artifact_id=artifact.id, requesting_agent_id=agent, requested_at_tick=102))
    assert registry.get_agent_state(artifact.id, agent) == MESIState.SHARED
    # The slot survives (the commit diagnostic reads it) but the pair is not
    # INVALID, so the read does not report it.
    assert registry.get_last_reclamation(agent, artifact.id) == ("reclaim_heartbeat", 100)
    assert registry.invalid_reclamations() == {}

    # A fresh write grant clears the slot; its voluntary end is a release.
    svc.write(agent_id=agent, artifact_id=artifact.id, issued_at_tick=200)
    svc.invalidate(
        agent_id=agent,
        artifact_id=artifact.id,
        new_version=registry.get_artifact(artifact.id).version,
        issuer_agent_id=agent,
        issued_at_tick=201,
    )
    assert registry.get_agent_state(artifact.id, agent) == MESIState.INVALID
    assert registry.invalid_reclamations() == {}


def test_only_the_reclaimed_pair_is_reported(registry) -> None:
    """Two artifacts, two agents: the read is keyed by artifact then agent and
    names only the pair the sweep pulled."""
    svc = CoordinatorService(registry)
    plan = svc.register_artifact(name="plan.md", content="v1")
    spec = svc.register_artifact(name="spec.md", content="v1")
    stale, live = uuid4(), uuid4()
    svc.fetch(FetchRequest(artifact_id=plan.id, requesting_agent_id=stale, requested_at_tick=0))
    svc.fetch(FetchRequest(artifact_id=spec.id, requesting_agent_id=live, requested_at_tick=0))
    svc.record_heartbeat(agent_id=stale, now_tick=0)
    svc.record_heartbeat(agent_id=live, now_tick=100)
    svc.enforce_stable_grant_timeouts(
        current_tick=100, heartbeat_timeout_ticks=10, max_hold_ticks=10_000
    )
    assert registry.get_agent_state(spec.id, live) == MESIState.EXCLUSIVE
    assert registry.invalid_reclamations() == {plan.id: {stale: ("reclaim_heartbeat", 100)}}
