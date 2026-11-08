# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""NoZombieRevoke — the read-generation fence as an OPERATION-CLASS property.

Regression suite for two defects in the grant-revoking path, both reproduced
single-host, in one process, under the registry ``RLock``. The shape is the one
Temporal closed in their semaphore pattern with signal pinning: "a late release
Signal from a holder whose lease already expired could revoke a subsequent
acquirer's permit".

The fence (``owner_generation`` vs a committer's captured ``read_generation``)
was checked on the three COMMIT paths only — ``commit_cas``, ``commit_all`` and
``set_artifact_and_content(fence_agent_id=…)`` — and neither checked nor
maintained by the other grant-revoking operation, ``invalidate``:

F1 — ZOMBIE REVOKE. ``CoordinatorService.invalidate`` revoked whatever grant it
     named with no validation at all, so a signal minted at one epoch destroyed
     a grant established at a later one. Fixed by pinning a PEER-issued
     invalidation to the target's ``last_observed_version``
     (``service._revoke_is_superseded``).

F2 — FENCE SUPPRESSION. ``invalidate`` moved an M/E holder to INVALID with
     ``trigger="invalidate"``, which was not in ``RECLAIM_TRIGGERS``, so
     ``owner_generation`` did not bump — and the sweep could never arm the fence
     afterwards, there being no M/E grant left to reclaim. Identical end-state,
     opposite verdict. Fixed by ``EPOCH_BUMP_TRIGGERS``.

The last two tests pin the guards that keep F1's fix from over-reaching:
dropping an invalidation is far more dangerous than applying one.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.adapters.base import CoherenceAdapterCore
from ccs.bus.event_bus import InMemoryEventBus
from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.states import MESIState
from ccs.core.types import Artifact, ConflictDetail

_M_OR_E = (MESIState.MODIFIED, MESIState.EXCLUSIVE)


def _clear_observation(reg, artifact_id, agent_id) -> None:
    """Put a live grant into the never-observed state, on either backend.

    Reproduces what the v5->v6 schema migration leaves behind: an
    ``agent_states`` row that predates ``last_observed_version`` and so carries
    no recorded observation (absent key in memory, NULL column in sqlite) while
    the grant itself is live. ``set_agent_state`` records an observation on
    every non-INVALID transition, so this state cannot be reached forward from
    a fresh store -- only inherited from one written before SB-10.
    """
    records = getattr(reg, "_records", None)
    if records is not None:
        records[artifact_id].last_observed_version_by_agent.pop(agent_id, None)
    else:
        reg._conn.execute(
            "UPDATE agent_states SET last_observed_version = NULL "
            "WHERE artifact_id = ? AND agent_id = ?",
            (artifact_id.hex, agent_id.hex),
        )


@pytest.fixture(params=["in_memory", "sqlite"])
def registry(request, tmp_path: Path):
    """Both registries, identically — the bump and the capture are duplicated
    across them (see test_fencing.py), so a fix must land in both."""
    if request.param == "in_memory":
        yield ArtifactRegistry()
    else:
        with SqliteArtifactRegistry(tmp_path / "state.db") as reg:
            yield reg


# ---------------------------------------------------------------------------
# F1 — a revoke minted at a superseded epoch must not land
# ---------------------------------------------------------------------------


def test_invalidate_rejects_issuer_from_a_superseded_generation(registry) -> None:
    """A revoke whose issuer captured epoch g must not revoke a grant the
    registry established at epoch g+1.

    The interleaving (single host, one process, one coordinator):

      t0  A acquires EXCLUSIVE on X          -> owner_generation 0, A.rg 0
      t1  A commits                          -> version 2; a signal is minted
      t2  the sweep reclaims A's grant       -> owner_generation 1  (epoch moved)
      t3  B acquires a FRESH EXCLUSIVE       -> B.rg 1              (new epoch)
      t4  the t1 signal is finally delivered -> it must NOT touch B's grant

    Before the fix nothing between t1 and t4 was compared, and t4 destroyed a
    grant issued two steps after the signal was minted.

    Runs on BOTH registries: the pin compares a value each backend writes
    independently, so a one-sided regression has to be visible here.
    """
    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    a, b = uuid4(), uuid4()

    svc.write(agent_id=a, artifact_id=art.id, issued_at_tick=1)
    svc.registry.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=1)
    updated, signals = svc.commit(agent_id=a, artifact_id=art.id, content="v2", issued_at_tick=2)
    # The genuine signal the coordinator minted for b at t1 — not a hand-written
    # one; it is replayed verbatim at t4 below.
    (signal,) = [s for s in signals if s.artifact_id == art.id]
    assert (signal.new_version, signal.issued_at_tick) == (updated.version, 2)

    svc.registry.set_agent_state(
        art.id, a, MESIState.INVALID, trigger="reclaim_heartbeat", tick=50
    )
    assert svc.registry.get_owner_generation(art.id) == 1

    svc.write(agent_id=b, artifact_id=art.id, issued_at_tick=60)
    assert svc.registry.get_read_generation(art.id, b) == 1
    assert svc.registry.get_agent_state(art.id, b) == MESIState.EXCLUSIVE

    svc.invalidate(
        agent_id=b,
        artifact_id=art.id,
        new_version=signal.new_version,
        issuer_agent_id=signal.issuer_agent_id,
        issued_at_tick=signal.issued_at_tick,
    )

    assert svc.registry.get_agent_state(art.id, b) in _M_OR_E, (
        "NoZombieRevoke: a revoke minted at epoch 0 destroyed a grant "
        f"established at epoch 1 (artifact is at version {updated.version})"
    )


def test_invalidate_rejects_signal_older_than_the_current_version(registry) -> None:
    """The weakest form of the same property, using only the operand
    ``invalidate`` already takes: a signal announcing ``new_version=N`` cannot
    be authority to revoke a claim the registry established at a version
    strictly greater than N."""
    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    a, b = uuid4(), uuid4()

    svc.write(agent_id=a, artifact_id=art.id, issued_at_tick=1)
    svc.commit(agent_id=a, artifact_id=art.id, content="v2", issued_at_tick=2)
    svc.write(agent_id=b, artifact_id=art.id, issued_at_tick=3)
    svc.commit(agent_id=b, artifact_id=art.id, content="v3", issued_at_tick=4)

    current = svc.registry.get_artifact(art.id).version
    assert current == 3
    assert svc.registry.get_agent_state(art.id, b) == MESIState.MODIFIED

    svc.invalidate(
        agent_id=b,
        artifact_id=art.id,
        new_version=2,  # stale: minted before b's own commit
        issuer_agent_id=a,
        issued_at_tick=2,
    )

    assert svc.registry.get_agent_state(art.id, b) in _M_OR_E, (
        "a v2 invalidation revoked a claim held at v3"
    )


def test_adapter_concurrent_publish_does_not_revoke_a_fresher_grant() -> None:
    """Reachability on a shipped surface, single host, one process.

    ``CoherenceAdapterCore.write`` mints its invalidation signals INSIDE the
    registry lock (``service.write`` / ``service.commit`` under
    ``abort_guard``) and publishes them AFTER that lock is released. The class
    documents concurrent callers ("LangGraph parallel branches" —
    ``adapters/base.py`` ``_sweep_lock``), so a peer can complete an entire
    read → acquire → commit cycle inside that window. Delivery then calls
    ``AgentRuntime.handle_invalidation`` -> ``coordinator.invalidate(agent_id=
    recipient, …)``, which threads no abort Event — so ``abort_guard(None)`` is
    a plain lock acquire and closes nothing here. The pin is what closes it.

    The gate below stops thread A exactly at the publish boundary; it forces no
    state the process cannot reach on its own.
    """

    class _GatedBus(InMemoryEventBus):
        def __init__(self) -> None:
            super().__init__()
            self.reached = threading.Event()
            self.release = threading.Event()
            self._armed = True

        def publish_invalidation(self, signal, *, recipients):  # type: ignore[override]
            if self._armed:
                self._armed = False
                self.reached.set()
                self.release.wait(timeout=5)
            return super().publish_invalidation(signal, recipients=recipients)

    bus = _GatedBus()
    core = CoherenceAdapterCore(event_bus=bus)
    core.register_agent("A", now_tick=0)
    agent_b = core.register_agent("B", now_tick=0)
    art = core.register_artifact(name="plan.md", content="v1")

    core.read(agent_name="A", artifact_id=art.id, now_tick=1)
    core.read(agent_name="B", artifact_id=art.id, now_tick=1)

    failures: list[BaseException] = []

    def _a_writes() -> None:
        try:
            core.write(agent_name="A", artifact_id=art.id, content="a-v2", now_tick=2)
        except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
            failures.append(exc)

    thread = threading.Thread(target=_a_writes, daemon=True)
    thread.start()
    try:
        assert bus.reached.wait(timeout=5), "A never reached the publish boundary"
        assert core.registry.get_artifact(art.id).version == 2

        # B's own full cycle, entirely inside A's publish window.
        core.write(agent_name="B", artifact_id=art.id, content="b-v3", now_tick=3)
        assert core.registry.get_artifact(art.id).version == 3
        assert core.registry.get_agent_state(art.id, agent_b) in _M_OR_E
    finally:
        bus.release.set()
        thread.join(timeout=5)

    assert not failures, f"A's write raised: {failures[0]!r}"
    assert core.registry.get_agent_state(art.id, agent_b) in _M_OR_E, (
        "the held v2 signal revoked B's claim, established at v3"
    )
    # Both halves, or the guarantee is half true. The coordinator keeping the
    # grant is worth nothing if the agent's own view was clamped on the way in:
    # handle_invalidation asks the coordinator BEFORE it touches the cache, so a
    # declined signal leaves B's cache alone too.
    entry = core._agents_by_name["B"].runtime.cache.get(art.id)
    assert entry is not None and entry.state in _M_OR_E, (
        "the coordinator kept B's grant but B's own cached view was invalidated "
        f"by the same stale signal (cache state {entry and entry.state})"
    )


# ---------------------------------------------------------------------------
# F2 — a revoke must leave the fence in the state a reclaim would have
# ---------------------------------------------------------------------------


def test_release_by_invalidate_arms_the_fence_like_a_sweep_reclaim(registry) -> None:
    """Same end-state, one differing step, opposite verdict.

    Both arms end identically: A's M/E grant is gone, a peer took and released
    the artifact, the version never moved, no M/E holder remains. They differ
    only in HOW A's grant went away.

      sweep reclaim  -> owner_generation bumps -> A's commit_cas is REJECTED
                        ``stale_read_generation`` (test_fencing.py pins this)
      invalidate     -> no bump -> the sweep has no M/E grant left to reclaim,
                        so the fence can never arm -> A's commit_cas is ADMITTED
                        and bumps the version

    The rejected commit is the one the fence exists for: the version never
    moved, so version-CAS structurally cannot see it (``backend_contract.py``
    R9_ATOMIC_BOUNDARY). Whichever verdict is right, the two paths cannot
    disagree — a revoked grant is a revoked grant.
    """
    reg = registry
    art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
    reg.register_artifact(art, content="ignored")
    svc = CoordinatorService(reg)
    a, b = uuid4(), uuid4()

    # A takes a write claim at epoch 0 and holds a buffer of version 1.
    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
    assert reg.get_read_generation(art.id, a) == 0

    # A's grant is revoked — by the OTHER grant-revoking operation.
    svc.invalidate(
        agent_id=a, artifact_id=art.id, new_version=1, issuer_agent_id=a, issued_at_tick=2
    )
    assert reg.get_agent_state(art.id, a) == MESIState.INVALID

    # The sweep runs and finds nothing to reclaim: A is already INVALID.
    reg.record_heartbeat(a, 1)
    assert (
        svc.enforce_stable_grant_timeouts(
            current_tick=100, heartbeat_timeout_ticks=10, max_hold_ticks=1000
        )
        == 0
    )

    # A peer takes the artifact and releases it without moving the version —
    # the shape a post-edit failure / session-stop release produces, and the
    # window in which the peer may have mutated shared bytes.
    reg.set_agent_state(art.id, b, MESIState.EXCLUSIVE, trigger="write", tick=101)
    svc.invalidate(
        agent_id=b, artifact_id=art.id, new_version=1, issuer_agent_id=b, issued_at_tick=102
    )
    assert reg.get_artifact(art.id).version == 1

    result = reg.commit_cas(art.id, a, expected_version=1, content_hash="zombie")

    assert isinstance(result, ConflictDetail), (
        "the zombie's commit was ADMITTED; the identical sweep-reclaim arm "
        "rejects it (test_parity_commit_cas_fence_rejects_superseded_reader)"
    )
    assert result.reason == "stale_read_generation"
    assert reg.get_artifact(art.id).version == 1, "phantom version bump"


def test_invalidate_of_an_me_holder_bumps_owner_generation(registry) -> None:
    """The mechanism behind F2, isolated: taking an M/E holder to INVALID is an
    epoch change however it is spelled. ``RECLAIM_TRIGGERS`` gates the bump on
    the sweep's three triggers, so the ``"invalidate"`` trigger — the only other
    way an M/E grant is revoked — silently leaves the epoch behind."""
    reg = registry
    art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
    reg.register_artifact(art, content="ignored")
    a = uuid4()

    reg.set_agent_state(art.id, a, MESIState.EXCLUSIVE, trigger="write", tick=1)
    assert reg.get_owner_generation(art.id) == 0

    reg.set_agent_state(art.id, a, MESIState.INVALID, trigger="invalidate", tick=2)

    assert reg.get_owner_generation(art.id) == 1, (
        "an M/E grant was revoked without moving the ownership epoch"
    )


# ---------------------------------------------------------------------------
# Guards on the F1 pin — dropping an invalidation is worse than applying one
# ---------------------------------------------------------------------------


def test_a_genuinely_behind_peer_is_still_invalidated(registry) -> None:
    """The pin must never swallow a real invalidation. A holder whose recorded
    observation is BEHIND the announced version is exactly the stale-read ->
    write case this layer exists to close: it is invalidated, pin or no pin.

    The version is moved through ``set_artifact_and_content`` -- the one write
    path that advances a version WITHOUT touching holder states, so b is left
    genuinely non-INVALID and behind. Every service-level commit path
    invalidates its peers, which is precisely why the pin is sound; this test
    reaches the branch that soundness argument leaves open.
    """
    reg = registry
    svc = CoordinatorService(reg)
    art = svc.register_artifact(name="plan.md", content="v1")
    a, b = uuid4(), uuid4()

    reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=1)
    assert reg.last_observed_version_for(art.id, b) == 1

    bumped = Artifact(id=art.id, name="plan.md", version=2, content_hash="h2")
    reg.set_artifact_and_content(art.id, bumped, "v2", last_writer=a)
    assert reg.get_artifact(art.id).version == 2
    assert reg.get_agent_state(art.id, b) == MESIState.SHARED
    assert reg.last_observed_version_for(art.id, b) == 1  # behind, and still holding

    svc.invalidate(
        agent_id=b,
        artifact_id=art.id,
        new_version=2,
        issuer_agent_id=a,
        issued_at_tick=4,
    )

    assert reg.get_agent_state(art.id, b) == MESIState.INVALID, (
        "a peer holding v1 was not invalidated by a v2 announcement"
    )


def test_absent_observation_is_admitted_not_dropped(registry) -> None:
    """Admit-on-absent: a live grant with NO recorded observation is invalidated,
    never pinned.

    Absence is not evidence of freshness. This is the conservative leg of the
    predicate -- the one that keeps an unknown observation on the APPLY side --
    and it is reachable in the field: the v5->v6 migration leaves pre-SB-10
    grants with no recorded observation while the grant stays live. Without this
    test, flipping ``observed is not None`` to ``observed is None`` passes the
    whole suite while silently dropping real invalidations.
    """
    reg = registry
    svc = CoordinatorService(reg)
    art = svc.register_artifact(name="plan.md", content="v1")
    a, b = uuid4(), uuid4()

    reg.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=1)
    _clear_observation(reg, art.id, b)
    assert reg.last_observed_version_for(art.id, b) is None
    assert reg.get_agent_state(art.id, b) == MESIState.SHARED

    svc.invalidate(
        agent_id=b,
        artifact_id=art.id,
        new_version=1,
        issuer_agent_id=a,
        issued_at_tick=2,
    )

    assert reg.get_agent_state(art.id, b) == MESIState.INVALID, (
        "an absent observation was treated as proof of freshness and the "
        "invalidation was dropped"
    )


def test_eager_update_then_declined_invalidation_leaves_no_transient() -> None:
    """A declined invalidation must not strand the transient it would have cleared.

    ``_write_impl`` sets a peer's SIA/EIA transient together with the INVALID
    transition and leaves the bus-delivered invalidation to clear it. An eager
    content push can move that peer back to SHARED first -- and then the pin
    legitimately declines the late signal, so the clear never happens unless
    ``handle_update`` does it, the way ``fetch`` already does. A stranded
    transient makes the stable-grant sweep skip the pair and fails the peer's
    next ``commit_cas`` on its transient precondition, until the timeout
    fail-safe force-invalidates an agent that is holding correct data.

    Delivery is driven by hand rather than through the bus, so the invalidation
    lands after the update -- the ordering the failure needs.
    """
    core = CoherenceAdapterCore(event_bus=InMemoryEventBus())
    agent_a = core.register_agent("A", now_tick=0)
    agent_b = core.register_agent("B", now_tick=0)
    art = core.register_artifact(name="plan.md", content="v1")
    runtime_b = core._agents_by_name["B"].runtime

    core.read(agent_name="B", artifact_id=art.id, now_tick=1)
    assert core.registry.get_agent_state(art.id, agent_b) is not MESIState.INVALID

    # A takes exclusivity: B goes INVALID and its transient is set, but nothing
    # has delivered the signal yet.
    signals = core.coordinator.write(agent_id=agent_a, artifact_id=art.id, issued_at_tick=2)
    assert core.registry.get_agent_state(art.id, agent_b) == MESIState.INVALID
    assert core.registry.get_agent_transient(art.id, agent_b) is not None
    (signal,) = [s for s in signals if s.artifact_id == art.id]

    # An eager push moves B back to SHARED at the current version.
    runtime_b.handle_update(
        artifact_id=art.id, version=1, content="v1", now_tick=3, writer_agent_id=None
    )
    assert core.registry.get_agent_state(art.id, agent_b) == MESIState.SHARED

    # The late signal now arrives and is correctly declined -- B is not behind it.
    runtime_b.handle_invalidation(signal)
    assert core.registry.get_agent_state(art.id, agent_b) == MESIState.SHARED

    assert core.registry.get_agent_transient(art.id, agent_b) is None, (
        "the declined invalidation left B's arbitration transient set; the sweep "
        "will skip this pair and B's next commit_cas will fail its precondition"
    )


def test_self_issued_release_is_never_pinned(registry) -> None:
    """An agent handing back its OWN claim is not a cross-agent revoke and is
    never dropped — this is the shape every Claude Code adapter call site uses
    (post-edit failure, session-stop release, operator drain). Its
    ``last_observed_version`` always equals the current version, so a pin that
    did not exempt it would make session-stop unable to release anything."""
    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    a = uuid4()

    svc.write(agent_id=a, artifact_id=art.id, issued_at_tick=1)
    current = svc.registry.get_artifact(art.id).version
    assert svc.registry.last_observed_version_for(art.id, a) == current

    svc.invalidate(
        agent_id=a,
        artifact_id=art.id,
        new_version=current,
        issuer_agent_id=a,  # self-release
        issued_at_tick=2,
    )

    assert svc.registry.get_agent_state(art.id, a) == MESIState.INVALID
    assert svc.registry.get_owner_generation(art.id) == 1, (
        "the release revoked a write claim without moving the version, so it "
        "must move the epoch (F2)"
    )


def test_peer_invalidation_still_clears_a_stranded_transient(registry) -> None:
    """``_write_impl`` invalidates peers directly and leaves their SIA/EIA
    transient for the bus-delivered invalidation to clear. The pin must not
    skip that: an already-INVALID target is pinned by nothing."""
    svc = CoordinatorService(ArtifactRegistry())
    art = svc.register_artifact(name="plan.md", content="v1")
    a, b = uuid4(), uuid4()

    svc.registry.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=1)
    svc.write(agent_id=a, artifact_id=art.id, issued_at_tick=2)

    assert svc.registry.get_agent_state(art.id, b) == MESIState.INVALID
    assert svc.registry.get_agent_transient(art.id, b) is not None

    svc.invalidate(
        agent_id=b,
        artifact_id=art.id,
        new_version=svc.registry.get_artifact(art.id).version,
        issuer_agent_id=a,
        issued_at_tick=3,
    )

    assert svc.registry.get_agent_transient(art.id, b) is None, (
        "the pin stranded a peer's invalidation transient"
    )


# ---------------------------------------------------------------------------
# R9 / AE1 — the sweep's own read-to-write window
# ---------------------------------------------------------------------------
#
# The two defects above are about a signal that outlived its authority. This
# one is about a DECISION that outlived the state it was made from, inside a
# single caller: ``enforce_stable_grant_timeouts`` reads the state map, the
# transient slot, the heartbeat and ``granted_at_tick`` in four separate
# registry calls, then writes INVALID in a fifth. Nothing holds those five
# together on EITHER backend: sqlite serializes each call on its RLock, but
# the sequence lives in the service layer, above any per-call lock. Both arms
# fail here today, and both must pass once the sweep decides and writes in
# one hold (the plan's U5).
#
# The GIL switch interval is forced down for the racing test: at the 5 ms
# default a ~100-bytecode window between two registry calls is practically
# never split, and the failure this test exists to show goes quiet. At 1 us
# it reproduces in well over half the rounds on both arms (measured: 92-137
# warnings per run in-memory, 9-19 on sqlite), which over 40 rounds puts a
# spurious full pass below 1e-4.

_RACE_SWITCH_INTERVAL = 1e-6

_SWEEP_ROUNDS = 40
_RENEWALS_PER_ROUND = 60
_SWEEPS_PER_ROUND = 60

_STALE_GRANT_TICK = 0          # over-held by construction at _SWEEP_TICK
_SWEEP_TICK = 1_000
_MAX_HOLD_TICKS = 100          # a grant renewed at _SWEEP_TICK is 0 ticks old
_HEARTBEAT_TIMEOUT_TICKS = 10_000  # never the deciding leg in these rounds

_NO_GRANTED_AT = "has no granted_at slot"


def _renew_grant(reg, artifact_id: object, agent_id: object, tick: int) -> None:
    """One renewal: hand the write claim back, take it again at a fresh tick.

    ``set_agent_state`` writes ``granted_at_tick`` on an M∪E ACQUIRE only --
    M<->E transitions deliberately preserve the original grant tick, because the
    agent has continuously held *some* write claim. So a holder refreshes its
    max-hold budget the only way the API allows, and the only way a real one
    does: it releases and re-acquires.
    """
    reg.set_agent_state(
        artifact_id, agent_id, MESIState.SHARED, trigger="invalidate", tick=tick, observed=False
    )
    reg.set_agent_state(
        artifact_id, agent_id, MESIState.EXCLUSIVE, trigger="write", tick=tick
    )


def test_sweep_does_not_reclaim_a_renewed_grant(registry, caplog) -> None:
    """A grant renewed at a fresh tick is not reclaimed by a concurrent sweep.

    R9 / AE1. Two windows open inside ``enforce_stable_grant_timeouts``, both
    between a read and the write that acts on it, with no lock holding the
    pair together:

      W1  the sweep reads ``granted_at_tick`` (or the heartbeat) while the
          grant is still the original over-held one, decides reclaim, and by
          the time it writes INVALID the holder has renewed. A grant that was
          fresh at write time is destroyed on a budget it no longer had.
          Proven reachable by a forced interleave (park the sweep inside
          ``last_heartbeat_tick``, heartbeat, release: the reclaim lands on a
          holder that just proved it was alive); a few-bytecode window, so
          racing alone never hit it in 160 measured rounds. The final-state
          assertion below is its regression net.

      W2  the sweep's snapshot sees M/E, then the renewal's release pops the
          ``granted_at_tick`` slot before the sweep reads it. The sweep finds
          an M/E holder with no grant tick -- a state the registry's own
          comment says "should not exist" -- logs it, and SKIPS the max-hold
          check for that holder: enforcement quietly not happening. This is
          the wide window; it is what reliably fails this test today, on both
          arms. The warning assertion below is its discriminator.

    The discriminator for W1 needs no timing heuristic. The renewer's LAST act
    in every round is an acquire, so if the holder is INVALID once both threads
    have joined, the sweep's write necessarily landed after that acquire. A
    sweep that reclaims the original stale grant BEFORE the first renewal is
    correct, and leaves the holder M/E via the renewal that follows -- so it is
    not counted, and the assertion cannot fire on a legitimate reclaim.

    Runs many rounds because both windows are genuine races, not gated ones: a
    gate inside the sweep's read-to-write window would deadlock against the
    very lock this test exists to justify -- and after the fix the gated
    interleave is exactly the schedule the lock makes unreachable.
    """
    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    registry.record_heartbeat(agent, _SWEEP_TICK)

    zombie_rounds: list[int] = []
    failures: list[BaseException] = []

    default_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(_RACE_SWITCH_INTERVAL)
    try:
        with caplog.at_level(logging.WARNING, logger="ccs.coordinator.service"):
            for round_no in range(_SWEEP_ROUNDS):
                # Reset to the over-held grant W1 needs: a live claim whose budget
                # ran out long before _SWEEP_TICK.
                registry.set_agent_state(
                    art.id, agent, MESIState.INVALID, trigger="invalidate", tick=_STALE_GRANT_TICK
                )
                registry.set_agent_state(
                    art.id, agent, MESIState.EXCLUSIVE, trigger="write", tick=_STALE_GRANT_TICK
                )

                start = threading.Barrier(2, timeout=5)

                def _renews() -> None:
                    try:
                        start.wait()
                        for _ in range(_RENEWALS_PER_ROUND):
                            _renew_grant(registry, art.id, agent, _SWEEP_TICK)
                    except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                        failures.append(exc)

                def _sweeps() -> None:
                    try:
                        start.wait()
                        for _ in range(_SWEEPS_PER_ROUND):
                            svc.enforce_stable_grant_timeouts(
                                current_tick=_SWEEP_TICK,
                                heartbeat_timeout_ticks=_HEARTBEAT_TIMEOUT_TICKS,
                                max_hold_ticks=_MAX_HOLD_TICKS,
                            )
                    except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                        failures.append(exc)

                renewer = threading.Thread(target=_renews, daemon=True)
                sweeper = threading.Thread(target=_sweeps, daemon=True)
                renewer.start()
                sweeper.start()
                renewer.join(timeout=5)
                sweeper.join(timeout=5)
                assert not renewer.is_alive() and not sweeper.is_alive(), (
                    f"round {round_no}: a worker never finished — the sweep and the "
                    "renewer are deadlocked against each other"
                )
                if failures:
                    break

                if registry.get_agent_state(art.id, agent) not in _M_OR_E:
                    zombie_rounds.append(round_no)

    finally:
        sys.setswitchinterval(default_switch_interval)

    assert not failures, f"a worker raised: {failures[0]!r}"

    assert not zombie_rounds, (
        f"the sweep reclaimed a renewed grant in {len(zombie_rounds)} of "
        f"{_SWEEP_ROUNDS} rounds (first: {zombie_rounds[0]}). The holder's last "
        "act was an acquire at a fresh tick, so the reclamation was decided on a "
        "granted_at_tick that had already been replaced when the write landed."
    )

    stranded = [r for r in caplog.records if _NO_GRANTED_AT in r.getMessage()]
    assert not stranded, (
        f"the sweep saw an M/E holder with no granted_at slot {len(stranded)} "
        "times — it read the state map and the grant tick either side of a "
        f"concurrent renewal (first: {stranded[0].getMessage()})"
    )


def test_sweep_still_reclaims_a_genuinely_over_held_grant(registry) -> None:
    """Teeth for the test above: the sweep must still do its job.

    Without this, ``test_sweep_does_not_reclaim_a_renewed_grant`` would pass
    against a sweep that reclaims nothing at all — including one broken by the
    serialization it exists to motivate.
    """
    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    registry.record_heartbeat(agent, _SWEEP_TICK)
    registry.set_agent_state(
        art.id, agent, MESIState.EXCLUSIVE, trigger="write", tick=_STALE_GRANT_TICK
    )

    reclaimed = svc.enforce_stable_grant_timeouts(
        current_tick=_SWEEP_TICK,
        heartbeat_timeout_ticks=_HEARTBEAT_TIMEOUT_TICKS,
        max_hold_ticks=_MAX_HOLD_TICKS,
    )

    assert reclaimed == 1
    assert registry.get_agent_state(art.id, agent) == MESIState.INVALID
    assert registry.get_last_reclamation(agent, art.id) is not None


# ---------------------------------------------------------------------------
# F3 — the update broadcast is the pin's un-fenced sibling
# ---------------------------------------------------------------------------
#
# ``handle_update`` used to apply a broadcast unconditionally: cache put,
# SHARED transitions, observation stamp, transient clears. The same
# publish-after-lock window the F1 test proves for invalidations exists for
# updates, and a stale update did three wrong things at once — regressed the
# recipient's view to older bytes marked valid (under the zero-staleness
# eager strategy), dropped a fresher M/E claim to SHARED with no epoch bump,
# and let ``set_agent_state`` stamp ``last_observed_version`` with the
# registry's CURRENT version while the cache held the OLDER body, poisoning
# the exact comparand ``_revoke_is_superseded`` trusts. The guard compares
# the broadcast's version against the registry's current version under one
# hold and skips the whole apply when it is behind — keeping only the
# transient clear, because a stranded transient stalls the sweep while an
# early clear is harmless. The fresh path is pinned by
# ``test_eager_update_then_declined_invalidation_leaves_no_transient``.


def test_stale_update_broadcast_does_not_regress_the_recipients_view() -> None:
    """End-to-end reachability: a held UPDATE must not overwrite newer bytes.

    Under the eager strategy a commit publishes full content AFTER the
    registry lock is released (``adapters/base.py``), so a peer can complete
    its own write inside that window. Before the fix the late v2 broadcast
    then landed on B — who wrote v3 itself — and B's cache and content view
    regressed to v2 marked SHARED-valid, while ``last_observed_version``
    stayed at 3: stale bytes served as fresh under ``staleness_bound() == 0``,
    and any v3-announcing heal signal would be declined by the pin.
    """

    class _GatedUpdateBus(InMemoryEventBus):
        def __init__(self) -> None:
            super().__init__()
            self.reached = threading.Event()
            self.release = threading.Event()
            self._armed = True

        def publish_update(self, event, *, recipients):  # type: ignore[override]
            if self._armed:
                self._armed = False
                self.reached.set()
                self.release.wait(timeout=5)
            return super().publish_update(event, recipients=recipients)

    bus = _GatedUpdateBus()
    core = CoherenceAdapterCore(strategy_name="eager", event_bus=bus)
    core.register_agent("A", now_tick=0)
    agent_b = core.register_agent("B", now_tick=0)
    art = core.register_artifact(name="plan.md", content="v1")

    core.read(agent_name="A", artifact_id=art.id, now_tick=1)
    core.read(agent_name="B", artifact_id=art.id, now_tick=1)

    failures: list[BaseException] = []

    def _a_writes() -> None:
        try:
            core.write(agent_name="A", artifact_id=art.id, content="a-v2", now_tick=2)
        except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
            failures.append(exc)

    thread = threading.Thread(target=_a_writes, daemon=True)
    thread.start()
    try:
        assert bus.reached.wait(timeout=5), "A never reached the update-publish boundary"
        assert core.registry.get_artifact(art.id).version == 2

        # B's own full cycle, entirely inside A's publish window.
        core.write(agent_name="B", artifact_id=art.id, content="b-v3", now_tick=3)
        assert core.registry.get_artifact(art.id).version == 3
    finally:
        bus.release.set()
        thread.join(timeout=5)

    assert not failures, f"A's write raised: {failures[0]!r}"
    entry = core._agents_by_name["B"].runtime.cache.get(art.id)
    assert entry is not None and entry.local_version == 3, (
        f"B's cached view regressed to the stale broadcast (version "
        f"{entry and entry.local_version}): older bytes marked valid while "
        "last_observed_version still claims version 3 was observed — the "
        "split that makes the pin decline the healing invalidation"
    )
    assert core.content(agent_name="B", artifact_id=art.id) == "b-v3", (
        "B's content view regressed to the stale broadcast under a "
        "zero-staleness strategy"
    )
    assert core.registry.last_observed_version_for(art.id, agent_b) == 3


def test_stale_update_does_not_downgrade_a_fresher_write_claim(registry) -> None:
    """Deterministic replay: a stale update on an M/E holder is dropped whole.

    The update-side twin of ``test_invalidate_rejects_signal_older_than_the_
    current_version``: B re-acquired EXCLUSIVE after the broadcast was minted,
    so the late event has no authority over B's claim — applying it would
    revoke the grant M -> SHARED with no epoch bump and regress B's cache.
    Only the transient clear survives the drop (a stranded slot stalls the
    stable-grant sweep; clearing early is harmless).
    """
    from ccs.agent.runtime import AgentRuntime
    from ccs.core.states import TransientState
    from ccs.strategies import build_strategy

    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    a, b = uuid4(), uuid4()

    # A commits v2; imagine its update broadcast held at the publish boundary.
    svc.registry.set_agent_state(art.id, b, MESIState.SHARED, trigger="fetch", tick=1)
    svc.write(agent_id=a, artifact_id=art.id, issued_at_tick=1)
    updated, _signals = svc.commit(agent_id=a, artifact_id=art.id, content="v2", issued_at_tick=2)
    assert updated.version == 2

    # B re-acquires a FRESH write claim after the broadcast was minted, with
    # the SIA/EIA-style transient a real invalidation would have left behind.
    svc.write(agent_id=b, artifact_id=art.id, issued_at_tick=3)
    assert svc.registry.get_agent_state(art.id, b) in _M_OR_E
    svc.registry.set_agent_transient(
        art.id, b, TransientState.SIA, entered_tick=3
    )

    runtime_b = AgentRuntime(
        agent_id=b, coordinator=svc, strategy=build_strategy("eager")
    )
    # The held v2 broadcast is finally delivered.
    runtime_b.handle_update(
        artifact_id=art.id, version=2, content="v2", now_tick=4, writer_agent_id=a
    )

    assert svc.registry.get_agent_state(art.id, b) in _M_OR_E, (
        "a stale update revoked B's write claim, established after the "
        "broadcast was minted — M -> SHARED with no epoch bump"
    )
    assert runtime_b.cache.get(art.id) is None, (
        "a stale update installed the broadcast's older bytes as a valid "
        "cache entry"
    )
    assert runtime_b.content(art.id) is None, (
        "a stale update recorded the broadcast's older content as B's view"
    )
    assert svc.registry.get_agent_transient(art.id, b) is None, (
        "the drop must still clear the transient — a stranded slot stalls "
        "the stable-grant sweep"
    )


def test_invalidation_for_a_deleted_artifact_still_clears_the_cache() -> None:
    """The decline fast-path must not shield a deleted artifact's stale cache.

    ``handle_invalidation`` returns early only when the coordinator DECLINED
    the signal for an artifact that still exists; a signal naming a deleted
    artifact falls through to the local clamp, because the deletion is itself
    the strongest reason the cached view is dead. Pins the
    ``has_artifact=False`` leg of the ``declined and has_artifact`` gate.
    """
    core = CoherenceAdapterCore(event_bus=InMemoryEventBus())
    core.register_agent("A", now_tick=0)
    agent_b = core.register_agent("B", now_tick=0)
    art = core.register_artifact(name="plan.md", content="v1")
    runtime_b = core._agents_by_name["B"].runtime

    core.read(agent_name="B", artifact_id=art.id, now_tick=1)
    assert runtime_b.cache.get(art.id) is not None

    (signal,) = core.coordinator.delete(
        agent_id=core._agents_by_name["A"].agent_id, artifact_id=art.id, issued_at_tick=2
    )
    runtime_b.handle_invalidation(signal)

    entry = runtime_b.cache.get(art.id)
    assert entry is None or entry.state == MESIState.INVALID, (
        "a signal naming a deleted artifact was declined-and-dropped, leaving "
        f"B's cached view alive (state {entry and entry.state}) for an "
        "artifact that no longer exists"
    )
    assert core.registry.get_agent_state(art.id, agent_b) is None
