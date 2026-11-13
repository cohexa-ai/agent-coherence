# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""Registry-level tests for the SB-18 atomic multi-artifact publish (commit_all).

All-or-nothing: either every member of the write-set advances or none do, and a
partial batch is never observable. Parametrized over BOTH registry backends
(in-memory + sqlite) so the two produce identical outcomes (backend parity).
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.states import MESIState
from ccs.core.types import (
    Artifact,
    CasCorruption,
    CommitAllEntry,
    MultiCommitConflict,
    MultiCommitResult,
)


@pytest.fixture(params=["memory", "sqlite"])
def registry(request, tmp_path):
    if request.param == "memory":
        yield ArtifactRegistry()
    else:
        with SqliteArtifactRegistry(tmp_path / "state.db") as reg:
            yield reg


@pytest.fixture(params=["memory", "sqlite"])
def retaining_registry(request, tmp_path):
    """A retain_versions=True registry over BOTH backends, paired with the name of
    the per-backend version-capture method inside the apply loop (so a test can
    inject a mid-apply raise there)."""
    if request.param == "memory":
        yield ArtifactRegistry(retain_versions=True), "_capture_version"
    else:
        with SqliteArtifactRegistry(tmp_path / "state.db", retain_versions=True) as reg:
            yield reg, "_capture_version_sql"


def _register(reg, name, version):
    a = Artifact(id=uuid4(), name=name, version=version, content_hash="h")
    reg.register_artifact(a, content=f"{name}-v{version}")
    return a


def test_commit_all_win_bumps_every_member(registry):
    a = _register(registry, "a.md", 1)
    b = _register(registry, "b.md", 1)
    writer = uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    registry.set_agent_state(b.id, writer, MESIState.SHARED, tick=1)
    result = registry.commit_all(
        writer,
        {
            a.id: CommitAllEntry(expected_version=1, content_hash="ha2", content="a-v2"),
            b.id: CommitAllEntry(expected_version=1, content_hash="hb2", content="b-v2"),
        },
        tick=2,
    )
    assert isinstance(result, MultiCommitResult)
    assert result.versions == {a.id: 2, b.id: 2}
    assert registry.get_artifact(a.id).version == 2
    assert registry.get_artifact(b.id).version == 2


def test_commit_all_one_drifted_holds_whole_batch_zero_mutation(registry):
    a = _register(registry, "a.md", 1)
    b = _register(registry, "b.md", 1)
    writer = uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    registry.set_agent_state(b.id, writer, MESIState.SHARED, tick=1)
    # a peer advances b under the writer's feet -> the writer's expected_version=1 is stale.
    peer = uuid4()
    registry.set_agent_state(b.id, peer, MESIState.SHARED, tick=1)
    registry.commit_cas(b.id, peer, expected_version=1, content_hash="hb2", content="b-v2", tick=1)
    assert registry.get_artifact(b.id).version == 2

    result = registry.commit_all(
        writer,
        {
            a.id: CommitAllEntry(expected_version=1, content_hash="ha2", content="a-v2"),
            b.id: CommitAllEntry(expected_version=1, content_hash="hb-x", content="b-x"),
        },
        tick=3,
    )
    assert isinstance(result, MultiCommitConflict)
    assert set(result.per_artifact) == {b.id}
    assert result.per_artifact[b.id].reason == "version_mismatch"
    assert result.per_artifact[b.id].current_version == 2
    # ALL-OR-NOTHING: the passing member a was NOT mutated.
    assert registry.get_artifact(a.id).version == 1


def test_commit_all_other_holder_holds_whole_batch(registry):
    a = _register(registry, "a.md", 1)
    b = _register(registry, "b.md", 1)
    writer, peer = uuid4(), uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    registry.set_agent_state(b.id, writer, MESIState.SHARED, tick=1)
    registry.set_agent_state(b.id, peer, MESIState.EXCLUSIVE, tick=1)  # peer holds M/E on b
    result = registry.commit_all(
        writer,
        {
            a.id: CommitAllEntry(expected_version=1, content_hash="ha2"),
            b.id: CommitAllEntry(expected_version=1, content_hash="hb2"),
        },
        tick=2,
    )
    assert isinstance(result, MultiCommitConflict)
    assert result.per_artifact[b.id].reason == "other_holder"
    assert registry.get_artifact(a.id).version == 1  # all-or-nothing


def test_commit_all_corruption_aborts_batch(registry):
    a = _register(registry, "a.md", 1)
    writer = uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    result = registry.commit_all(
        writer,
        {a.id: CommitAllEntry(expected_version=5, content_hash="h")},  # expected > current
        tick=2,
    )
    assert isinstance(result, CasCorruption)
    assert registry.get_artifact(a.id).version == 1


def test_commit_all_empty_write_set_raises(registry):
    with pytest.raises(ValueError):
        registry.commit_all(uuid4(), {}, tick=1)


def test_commit_all_singleton_matches_commit_cas(registry):
    a = _register(registry, "a.md", 1)
    writer = uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    result = registry.commit_all(
        writer,
        {a.id: CommitAllEntry(expected_version=1, content_hash="h2", content="v2")},
        tick=2,
    )
    assert isinstance(result, MultiCommitResult)
    assert result.versions == {a.id: 2}
    assert registry.get_artifact(a.id).version == 2


def test_commit_all_win_invalidates_peers_of_every_member(registry):
    a = _register(registry, "a.md", 1)
    b = _register(registry, "b.md", 1)
    writer, peer = uuid4(), uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    registry.set_agent_state(b.id, writer, MESIState.SHARED, tick=1)
    registry.set_agent_state(a.id, peer, MESIState.SHARED, tick=1)  # peer caches a
    result = registry.commit_all(
        writer,
        {
            a.id: CommitAllEntry(expected_version=1, content_hash="ha2", content="a-v2"),
            b.id: CommitAllEntry(expected_version=1, content_hash="hb2", content="b-v2"),
        },
        tick=2,
    )
    assert isinstance(result, MultiCommitResult)
    # per-artifact invalidation: the peer cached `a`, so it is invalidated for a.id
    assert peer in result.invalidated[a.id]
    assert registry.get_agent_state(a.id, peer) == MESIState.INVALID


def test_commit_all_stale_read_generation_holds_whole_batch(registry):
    """A member whose committer's captured read_generation was superseded by a
    reclaim (version unchanged, no other holder — the reclaim-zombie the version
    CAS can't see) returns a stale_read_generation conflict that holds the WHOLE
    batch; the clean member is untouched (all-or-nothing)."""
    a = _register(registry, "a.md", 1)
    b = _register(registry, "b.md", 1)
    writer = uuid4()
    # writer captures read_generation=0 on `a` via E, then is reclaimed ->
    # owner_generation(a)=1 while its read_generation stays 0 (superseded).
    registry.set_agent_state(a.id, writer, MESIState.EXCLUSIVE, trigger="write", tick=1)
    registry.set_agent_state(a.id, writer, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10)
    assert registry.get_owner_generation(a.id) == 1
    # writer holds a clean SHARED view on `b`.
    registry.set_agent_state(b.id, writer, MESIState.SHARED, tick=1)

    result = registry.commit_all(
        writer,
        {
            a.id: CommitAllEntry(expected_version=1, content_hash="ha2", content="a-v2"),
            b.id: CommitAllEntry(expected_version=1, content_hash="hb2", content="b-v2"),
        },
        tick=11,
    )
    assert isinstance(result, MultiCommitConflict)
    assert result.per_artifact[a.id].reason == "stale_read_generation"
    # all-or-nothing: neither member advanced (b is clean but held with the batch).
    assert registry.get_artifact(a.id).version == 1
    assert registry.get_artifact(b.id).version == 1


def _reclaim(reg, art_id, holder, tick):
    """A sweep-style reclaim of ``holder``'s fresh grant: owner_generation +1,
    version unmoved."""
    reg.set_agent_state(art_id, holder, MESIState.EXCLUSIVE, trigger="write", tick=tick)
    reg.set_agent_state(art_id, holder, MESIState.INVALID, trigger="reclaim_heartbeat", tick=tick + 1)


def _peer_commit(reg, art_id):
    won = reg.commit_cas(art_id, uuid4(), expected_version=1, content_hash="peer")
    assert isinstance(won, tuple), won


def _refused_pairs(result):
    return {
        art_id: (detail.reason, detail.captured_generation, detail.owner_generation)
        for art_id, detail in result.per_artifact.items()
    }


def test_commit_all_each_refused_member_carries_its_own_pair(registry):
    """Every refused member reports the pair its own check read: the fenced
    member its superseded claim (2, 3); the member behind on version the claim
    it holds (1, 1); the claimless member nothing. Distinct pairs and owner
    generations per member, so a value leaked from one member to another, a
    swapped field or a hard-coded one fails."""
    fenced = _register(registry, "fenced.md", 1)
    behind = _register(registry, "behind.md", 1)
    claimless = _register(registry, "claimless.md", 1)
    writer = uuid4()
    _reclaim(registry, fenced.id, uuid4(), tick=1)
    _reclaim(registry, fenced.id, uuid4(), tick=3)
    _reclaim(registry, fenced.id, writer, tick=5)  # writer's claim 2, owner 3
    _reclaim(registry, behind.id, uuid4(), tick=1)
    registry.set_agent_state(behind.id, writer, MESIState.SHARED, trigger="fetch", tick=7)  # claim 1
    _peer_commit(registry, behind.id)
    _peer_commit(registry, claimless.id)

    result = registry.commit_all(
        writer,
        {
            fenced.id: CommitAllEntry(expected_version=1, content_hash="f2"),
            behind.id: CommitAllEntry(expected_version=1, content_hash="b2"),
            claimless.id: CommitAllEntry(expected_version=1, content_hash="c2"),
        },
        tick=9,
    )

    assert isinstance(result, MultiCommitConflict)
    assert _refused_pairs(result) == {
        fenced.id: ("stale_read_generation", 2, 3),
        behind.id: ("version_mismatch", 1, 1),
        claimless.id: ("version_mismatch", None, None),
    }
    assert registry.get_owner_generation(fenced.id) == 3
    assert registry.get_owner_generation(behind.id) == 1


def test_commit_all_other_holder_member_carries_an_equal_pair(registry):
    """A member refused because a pessimistic peer holds it reports the
    writer's current claim equal to that member's owner generation."""
    a = _register(registry, "a.md", 1)
    writer, holder = uuid4(), uuid4()
    _reclaim(registry, a.id, uuid4(), tick=1)
    _reclaim(registry, a.id, uuid4(), tick=3)
    registry.set_agent_state(a.id, writer, MESIState.SHARED, trigger="fetch", tick=5)  # claim 2
    registry.set_agent_state(a.id, holder, MESIState.EXCLUSIVE, trigger="write", tick=6)

    result = registry.commit_all(
        writer, {a.id: CommitAllEntry(expected_version=1, content_hash="a2")}, tick=7
    )

    assert isinstance(result, MultiCommitConflict)
    assert _refused_pairs(result) == {a.id: ("other_holder", 2, 2)}


def test_commit_all_win_retains_prior_versions(retaining_registry):
    """A WIN under retain_versions=True captures each member's committed version
    into retention history (retrievable byte-for-byte)."""
    registry, _capture_attr = retaining_registry
    a = _register(registry, "a.md", 1)
    writer = uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    result = registry.commit_all(
        writer,
        {a.id: CommitAllEntry(expected_version=1, content_hash="h2", content="a-v2")},
        tick=2,
    )
    assert isinstance(result, MultiCommitResult)
    assert registry.get_artifact(a.id).version == 2
    assert registry.get_content_at_version(a.id, 2) == "a-v2"
    assert registry.get_content_at_version(a.id, 1) == "a.md-v1"


def test_commit_all_apply_raise_rolls_back_zero_mutation(retaining_registry):
    """The load-bearing all-or-nothing net: a raise DURING the apply loop (after an
    earlier member already mutated + retained) leaves ZERO mutation on either
    backend — versions restored AND the earlier member's retained version undone
    (in-memory snapshot-restore / sqlite ROLLBACK), so a rolled-back batch never
    leaks a half-applied version into retention history."""
    registry, capture_attr = retaining_registry
    a = _register(registry, "a.md", 1)
    b = _register(registry, "b.md", 1)
    writer = uuid4()
    registry.set_agent_state(a.id, writer, MESIState.SHARED, tick=1)
    registry.set_agent_state(b.id, writer, MESIState.SHARED, tick=1)

    real_capture = getattr(registry, capture_attr)
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # member 1 captured, member 2 raises mid-apply
            raise RuntimeError("injected mid-apply fault")
        return real_capture(*args, **kwargs)

    setattr(registry, capture_attr, boom)
    with pytest.raises(RuntimeError):
        registry.commit_all(
            writer,
            {
                a.id: CommitAllEntry(expected_version=1, content_hash="ha2", content="a-v2"),
                b.id: CommitAllEntry(expected_version=1, content_hash="hb2", content="b-v2"),
            },
            tick=2,
        )
    setattr(registry, capture_attr, real_capture)

    # Zero mutation on BOTH members.
    assert registry.get_artifact(a.id).version == 1
    assert registry.get_artifact(b.id).version == 1
    # No half-applied version leaked into retention (member 1's capture was undone).
    assert registry.get_content_at_version(a.id, 2) is None
    assert registry.get_content_at_version(b.id, 2) is None
