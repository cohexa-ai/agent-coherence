"""Read-generation fence (Piece #2) -- dual-registry parity harness (R4b).

A NET-NEW parametrized harness (none existed -- the closest precedent,
test_occ_commit_cas.py, tests each registry in separate functions) that runs the
SAME fence assertions against BOTH ArtifactRegistry (in-memory) and
SqliteArtifactRegistry. Asserts on version / owner_generation, never on content
bytes (sqlite keeps no content per KTD-13). The two registries share no base
class, so this is the guard against silent divergence -- including the
duplicated RECLAIM_TRIGGERS constant. Uses a fixed-stale-buffer arm (a reclaimed
holder's captured read_generation), not a refetch-safe read->+1->write arm.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import STALE_READ_GENERATION_REASON, StaleReadGeneration
from ccs.core.states import MESIState
from ccs.core.types import Artifact, ConflictDetail


@pytest.fixture(params=["in_memory", "sqlite"])
def registry(request, tmp_path: Path):
    """Yield each registry implementation in turn so every test runs against
    both, identically."""
    if request.param == "in_memory":
        yield ArtifactRegistry()
    else:
        with SqliteArtifactRegistry(tmp_path / "state.db") as reg:
            yield reg


def _register(reg) -> Artifact:
    art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
    reg.register_artifact(art, content="ignored")
    return art


def _pair(reg, artifact_id) -> tuple[int, int]:
    """(version, owner_generation) from the one pair-atomic accessor."""
    got = reg.get_artifact_and_generation(artifact_id)
    assert got is not None
    artifact, generation = got
    return artifact.version, generation


def _poke_owner_generation(reg, artifact_id) -> None:
    """Advance owner_generation behind set_agent_state's back so a live grant
    carries a superseded claim -- a shape the service cannot reach, built the
    way the MWB-transient regression below builds it (raw SQL on sqlite, the
    record map in memory)."""
    if isinstance(reg, SqliteArtifactRegistry):
        reg._conn.execute(
            "UPDATE artifacts SET owner_generation = owner_generation + 1 WHERE id = ?",
            (artifact_id.hex,),
        )
    else:
        reg._records[artifact_id].owner_generation += 1


def test_parity_owner_generation_bumps_on_reclaim_only(registry) -> None:
    reg = registry
    art = _register(reg)
    a, b = uuid4(), uuid4()
    assert reg.get_owner_generation(art.id) == 0
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
    assert reg.get_owner_generation(art.id) == 1
    # A peer-invalidation (non-reclaim trigger) does NOT bump.
    reg.set_agent_state(art.id, b, MESIState.EXCLUSIVE, trigger="write", tick=11)
    reg.set_agent_state(art.id, b, MESIState.INVALID, trigger="peer_invalidation", tick=12)
    assert reg.get_owner_generation(art.id) == 1
    # The OTHER reclaim trigger (max-hold) bumps too — both sweep triggers
    # behave identically on both registries.
    reg.set_agent_state(art.id, b, MESIState.EXCLUSIVE, trigger="write", tick=13)
    reg.set_agent_state(art.id, b, MESIState.INVALID, trigger="reclaim_max_hold", tick=20)
    assert reg.get_owner_generation(art.id) == 2


def test_parity_read_generation_captured_at_claim(registry) -> None:
    reg = registry
    art = _register(reg)
    a, b, c = uuid4(), uuid4(), uuid4()
    assert reg.get_read_generation(art.id, c) is None  # never claimed
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)  # acquire
    assert reg.get_read_generation(art.id, a) == 0
    reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=2)  # fetch read
    assert reg.get_read_generation(art.id, b) == 0
    # A reclaim preserves a's captured value (it is NOT refreshed).
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
    assert reg.get_read_generation(art.id, a) == 0


def test_parity_commit_cas_fence_rejects_superseded_reader(registry) -> None:
    reg = registry
    art = _register(reg)
    a = uuid4()
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)  # read_gen 0
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)  # owner_gen 1
    # Fixed-stale-buffer arm: a's captured read_generation (0) < owner (1).
    res = reg.commit_cas(art.id, a, expected_version=1, content_hash="new")
    assert isinstance(res, ConflictDetail)
    assert res.reason == "stale_read_generation"
    assert reg.get_artifact(art.id).version == 1  # no phantom bump


def test_parity_absent_read_generation_admits_version_cas_arbitrates(registry) -> None:
    """admit-on-absent is INTENTIONAL (resolves the roadmap residual + the
    line-80 / Fencing.tla / Retention.tla disagreement, and the over-strict
    absent=reject rule the 7e77e6a OCC suite already rejected). A committer that
    never established a fence claim has no captured read_generation (None). The
    fence does NOT reject the absent operand -- a plain OCC writer's lost-update
    protection is version-CAS, checked BEFORE the fence -- so an absent-operand
    commit_cas is ADMITTED on a matching version and arbitrated by version-CAS
    (version_mismatch) on a stale one, never by a stale_read_generation reject.
    """
    reg = registry
    art = _register(reg)
    a, b = uuid4(), uuid4()
    assert reg.get_read_generation(art.id, a) is None  # never claimed -> absent
    # Matching version on an absent operand: the fence admits, version-CAS wins.
    res = reg.commit_cas(art.id, a, expected_version=1, content_hash="fresh")
    assert not isinstance(res, ConflictDetail)  # WIN -> (artifact, invalidated)
    assert reg.get_artifact(art.id).version == 2
    # Stale version on a still-absent operand: arbitrated by version-CAS
    # (version_mismatch), NOT rejected by the fence as stale_read_generation.
    assert reg.get_read_generation(art.id, b) is None
    res2 = reg.commit_cas(art.id, b, expected_version=1, content_hash="stale")
    assert isinstance(res2, ConflictDetail)
    assert res2.reason == "version_mismatch"
    assert reg.get_artifact(art.id).version == 2  # no phantom bump


def test_parity_pessimistic_fence_rejects_superseded_committer(registry) -> None:
    reg = registry
    art = _register(reg)
    a = uuid4()
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
    stale = Artifact(id=art.id, name="plan.md", version=2, content_hash="stale")
    with pytest.raises(StaleReadGeneration):
        reg.set_artifact_and_content(art.id, stale, "x", last_writer=a, fence_agent_id=a)
    assert reg.get_artifact(art.id).version == 1


def test_reclaim_trigger_constants_are_equal_across_registries() -> None:
    """The RECLAIM_TRIGGERS / EPOCH_BUMP_TRIGGERS / CLAIM_CAPTURE_TRIGGERS /
    SWEEP_RECLAIM_TRIGGERS constants are duplicated (the registries share no
    base class); this pins the copies equal so the bump, the capture and the
    reclaim slot can never silently diverge. The capture pin also
    guards the service.fetch() trigger string: a rename there without updating
    the constants would silently disable read-generation capture on fetches."""
    from ccs.coordinator.registry import CLAIM_CAPTURE_TRIGGERS as MEM_CAPTURE
    from ccs.coordinator.registry import EPOCH_BUMP_TRIGGERS as MEM_BUMP
    from ccs.coordinator.registry import RECLAIM_TRIGGERS as IN_MEMORY
    from ccs.coordinator.registry import SWEEP_RECLAIM_TRIGGERS as MEM_SWEEP
    from ccs.coordinator.registry_protocol import HANDOFF_TRIGGER
    from ccs.coordinator.sqlite_registry import CLAIM_CAPTURE_TRIGGERS as SQL_CAPTURE
    from ccs.coordinator.sqlite_registry import EPOCH_BUMP_TRIGGERS as SQL_BUMP
    from ccs.coordinator.sqlite_registry import RECLAIM_TRIGGERS as SQLITE
    from ccs.coordinator.sqlite_registry import SWEEP_RECLAIM_TRIGGERS as SQL_SWEEP

    assert IN_MEMORY == SQLITE == frozenset(
        {"reclaim_heartbeat", "reclaim_max_hold", "timeout"}
    )
    # EPOCH_BUMP_TRIGGERS is what the bump sites actually key on: every reclaim
    # trigger PLUS the voluntary "invalidate" release and the "handoff" a
    # transfer moves its giver INVALID under -- each ends a write claim without
    # moving the version. The version-moving peer invalidations
    # ("write"/"commit") stay out. The handoff is a trigger of its own, not a
    # reclaim and not a release: the reclaim set above is unchanged.
    assert MEM_BUMP == SQL_BUMP == IN_MEMORY | frozenset({"invalidate", "handoff"})
    assert len(MEM_BUMP) == 5
    # The constant the transfer path emits, pinned to the literal: renaming it
    # without the set would leave the bump keyed on a string nothing emits.
    assert HANDOFF_TRIGGER == "handoff"
    assert MEM_CAPTURE == SQL_CAPTURE == frozenset({"fetch"})
    # The triggers whose M/E -> INVALID transition records the pair's reclaim
    # slot in the same write: the stable-grant sweep's two, never the transient
    # "timeout", which /status does not report as a reclaim. Both registries
    # must key the slot on one set, or a backend would store a reclaim without
    # its cause.
    assert MEM_SWEEP is SQL_SWEEP
    assert MEM_SWEEP == frozenset({"reclaim_heartbeat", "reclaim_max_hold"})


def test_transient_timeout_eviction_bumps_and_fences(registry) -> None:
    """Review gated-fix regression: the transient-timeout fail-safe
    (enforce_transient_timeouts -> trigger="timeout") is a coordinator-side
    eviction too — it must bump owner_generation so a post-eviction zombie
    commit_cas is rejected by the fence (the version never moved, so
    version-CAS alone would have admitted it)."""
    reg = registry
    art = _register(reg)
    a = uuid4()
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)  # claim g0
    # Transient-timeout fail-safe evicts the M/E holder.
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="timeout", tick=10)
    assert reg.get_owner_generation(art.id) == 1
    # The zombie's commit_cas (version unchanged, no other holder, stale claim)
    # is rejected — previously this was admitted via equality on gen 0.
    res = reg.commit_cas(art.id, a, expected_version=1, content_hash="zombie")
    assert isinstance(res, ConflictDetail)
    assert res.reason == "stale_read_generation"
    assert reg.get_artifact(art.id).version == 1


def test_commit_fence_raise_clears_mwb_transient(registry) -> None:
    """Regression (review P1): a fence reject inside service.commit() must not
    leak the MWB transient it set moments earlier — a stuck MWB blocks the
    agent's next commit_cas (transient precondition) and makes the stable-grant
    sweep skip the pair until the transient timeout. Manufactures the exact
    race window: the grant is still EXCLUSIVE but a sweep superseded the claim
    (owner_generation advanced between commit()'s state check and the
    version persist)."""
    from ccs.coordinator.service import CoordinatorService

    reg = registry
    art = _register(reg)
    service = CoordinatorService(reg)
    a = uuid4()
    # Pessimistic acquire WITHOUT a prior fetch (P0-fix path): captures gen 0.
    service.write(agent_id=a, artifact_id=art.id, issued_at_tick=1)
    # Manufacture the race: generation advances while the state stays E.
    _poke_owner_generation(reg, art.id)

    with pytest.raises(StaleReadGeneration):
        service.commit(agent_id=a, artifact_id=art.id, content="late", issued_at_tick=2)

    # The MWB transient is cleared (not leaked) and no phantom bump landed.
    assert reg.get_agent_transient(art.id, a) is None
    assert reg.get_artifact(art.id).version == 1


def test_parity_version_and_generation_is_one_pair(registry) -> None:
    """``get_artifact_and_generation`` returns the pair the effect gate compares:
    a sweep reclamation moves ONLY the generation leg (the shape a version-only
    comparand cannot see), a commit moves ONLY the version leg, and an unknown
    artifact answers None rather than raising.

    Answering None is the rule every SCALAR read follows for an artifact the
    registry has never seen, and ``get_owner_generation`` is the sole
    exception -- it raises KeyError instead, on both backends. The rule is
    asserted over the whole accessor list rather than a sample, so a later
    read that starts raising fails here instead of quietly falsifying this
    sentence. (Container reads are outside the rule: ``get_state_map`` and
    ``get_transient_map`` answer an empty dict, ``has_artifact`` False.)"""
    reg = registry
    art = _register(reg)
    a = uuid4()
    assert _pair(reg, art.id) == (1, 0)
    # Sweep reclamation: generation bumps, version stays put.
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
    reg.set_agent_state(
        art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10
    )
    assert _pair(reg, art.id) == (1, 1)
    # A committed write: version bumps, generation stays put. The committer
    # re-acquires first so its captured claim matches the current epoch.
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=11)
    res = reg.commit_cas(art.id, a, expected_version=1, content_hash="new")
    assert not isinstance(res, ConflictDetail)  # WIN -> (artifact, invalidated)
    assert _pair(reg, art.id) == (2, 1)
    # Absent-artifact contract, pinned over the whole scalar-read surface so
    # the "sole exception" claim above cannot quietly gain a second member.
    absent = uuid4()
    for accessor in ("get_artifact", "get_artifact_and_generation", "get_content"):
        assert getattr(reg, accessor)(absent) is None, accessor
    for accessor in ("get_agent_state", "get_agent_transient",
                     "get_read_generation", "get_transient_tick",
                     "last_observed_version_for"):
        assert getattr(reg, accessor)(absent, a) is None, accessor
    if isinstance(reg, SqliteArtifactRegistry):  # no in-memory counterpart
        assert reg.get_artifact_updated_at(absent) is None
        assert reg.last_writer_for(absent) is None
    # get_owner_generation is the one scalar read that raises instead.
    with pytest.raises(KeyError):
        reg.get_owner_generation(absent)
    # Container reads sit outside the None rule -- pinned so the docstring's
    # carve-out is checked too, not just asserted in prose.
    assert reg.get_state_map(absent) == {}
    assert reg.get_transient_map(absent) == {}
    assert reg.has_artifact(absent) is False


def test_parity_fetch_downgrade_preserves_superseded_read_generation(registry) -> None:
    """Leaving M/E is a loss of authority, never a claim. service.fetch tags
    its peer downgrade "fetch" too, so the capture predicate must not treat
    that leg as the agent's own read and refresh a superseded value. The
    stale-under-live-grant shape is poked directly; after the downgrade the
    value is still PRESENT (not cleared to absent, which admit-on-absent
    would wave through) and the ex-holder's commit is still refused."""
    reg = registry
    art = _register(reg)
    a = uuid4()
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)  # claim g0
    _poke_owner_generation(reg, art.id)
    assert _pair(reg, art.id) == (1, 1)

    # A peer's fetch downgrades the holder: E -> S under trigger "fetch".
    reg.set_agent_state(art.id, a, MESIState.SHARED, trigger="fetch", tick=2)

    assert reg.get_read_generation(art.id, a) == 0  # preserved, and present
    res = reg.commit_cas(art.id, a, expected_version=1, content_hash="late")
    assert isinstance(res, ConflictDetail)
    assert res.reason == STALE_READ_GENERATION_REASON
    assert _pair(reg, art.id) == (1, 1)  # version unmoved


def test_parity_own_fetch_read_captures_current_epoch(registry) -> None:
    """The requester leg of fetch IS the agent's own claim: an I -> S read
    captures the current epoch (not merely epoch 0), and a S -> S re-read after
    a later bump re-captures -- the recovery path for a reclaimed SHARED
    agent. Guards against tightening the trigger arm to from_state == INVALID,
    which would strand that re-read."""
    reg = registry
    art = _register(reg)
    a, b = uuid4(), uuid4()
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
    assert reg.get_owner_generation(art.id) == 1
    # I -> S: b's first read captures the CURRENT epoch.
    reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=11)
    assert reg.get_read_generation(art.id, b) == 1
    # A further reclaim supersedes b's claim ...
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=12)
    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=20)
    assert reg.get_owner_generation(art.id) == 2
    # ... and b's own S -> S re-read re-captures, so its commit wins.
    reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=21)
    assert reg.get_read_generation(art.id, b) == 2
    res = reg.commit_cas(art.id, b, expected_version=1, content_hash="fresh")
    assert not isinstance(res, ConflictDetail)  # WIN -> (artifact, invalidated)
    assert reg.get_artifact(art.id).version == 2


def test_parity_fetch_regrant_within_me_leaves_read_generation_unchanged(registry) -> None:
    """An E -> E re-grant tagged "fetch" is neither an acquire nor a read, so
    it does not capture. Under the service an M/E holder's captured value
    already equals the epoch; the poke makes the two differ so the no-capture
    is observable rather than vacuous."""
    reg = registry
    art = _register(reg)
    a = uuid4()
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)  # claim g0
    _poke_owner_generation(reg, art.id)

    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="fetch", tick=2)

    assert reg.get_read_generation(art.id, a) == 0
    assert reg.get_owner_generation(art.id) == 1
