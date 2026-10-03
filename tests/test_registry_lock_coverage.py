# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""The in-memory registry's concurrency contract: thread-safe, same as sqlite.

Companion to the plan's U2/U6: behavioral evidence here (concurrent writers
observe serialized effects), structural coverage (every public member holds
the lock) added by U6. The sqlite registry has carried this contract since it
existed; these tests are what retires the in-memory "single-threaded by
contract" era.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import FOREIGN_WRITE_OUTCOMES
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.states import MESIState
from ccs.core.types import Artifact, ConflictDetail, FetchRequest

_WRITERS = 4
_CALLS_PER_WRITER = 400


def test_state_log_sequence_is_gapless_under_concurrent_writers() -> None:
    """N threads writing states produce sequence numbers 1..N*calls exactly.

    ``set_agent_state`` does ``self._seq += 1`` then emits; without the
    registry lock that read-modify-write interleaves and two writers reserve
    the same number (a duplicate) or roll back over each other's reservation
    (a gap). U2's mutant: revert the lock widening and this fails within a
    few hundred calls at a 1 us switch interval.
    """
    emitted: list[int] = []
    reg = ArtifactRegistry(
        state_log=lambda entry: emitted.append(entry["sequence_number"]),
        instance_id="lock-coverage-test",
    )
    artifact = Artifact(id=uuid4(), name="plan.md", version=1)
    reg.register_artifact(artifact, "v1")
    agents = [uuid4() for _ in range(_WRITERS)]

    start = threading.Barrier(_WRITERS, timeout=5)
    failures: list[BaseException] = []

    def _writes(agent_id) -> None:
        try:
            start.wait()
            for tick in range(_CALLS_PER_WRITER):
                reg.set_agent_state(
                    artifact.id, agent_id, MESIState.SHARED, trigger="fetch", tick=tick
                )
        except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
            failures.append(exc)

    default_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=_writes, args=(a,), daemon=True) for a in agents]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            assert not t.is_alive(), "a writer never finished"
    finally:
        sys.setswitchinterval(default_switch_interval)

    assert not failures, f"a writer raised: {failures[0]!r}"
    total = _WRITERS * _CALLS_PER_WRITER
    assert sorted(emitted) == list(range(1, total + 1)), (
        f"sequence numbers are not gapless 1..{total}: "
        f"{len(emitted)} emitted, {len(set(emitted))} distinct, "
        f"min={min(emitted)}, max={max(emitted)}"
    )


@pytest.fixture(params=["in_memory", "sqlite"])
def registry(request, tmp_path: Path):
    """Both registries: the arbitration contract is now identical (KD1)."""
    if request.param == "in_memory":
        yield ArtifactRegistry()
    else:
        with SqliteArtifactRegistry(tmp_path / "state.db") as reg:
            yield reg


_CAS_ROUNDS = 20
_CAS_WRITERS = 4


def test_concurrent_cas_writers_produce_one_winner_and_typed_conflicts(registry) -> None:
    """N optimistic writers racing one version: exactly one WIN per round.

    The U4 property: ``commit_cas``'s version check, holder check, fence check
    and apply are ONE serialized step. Split them (the pre-U4 shape: legs
    outside the hold, apply inside) and two writers can both pass the version
    check before either applies -- two WINs from one expected_version, a
    version that advances twice, and a lost update. Reverting U4 fails this
    within a few rounds at a 1 us switch interval.

    Losers must get a typed ``ConflictDetail`` back, never an exception, and
    a conflict must leave the version exactly where the winner put it.
    """
    artifact = Artifact(id=uuid4(), name="plan.md", version=1)
    registry.register_artifact(artifact, "v1")
    writers = [uuid4() for _ in range(_CAS_WRITERS)]

    default_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for round_no in range(_CAS_ROUNDS):
            expected = registry.get_artifact(artifact.id).version
            results: dict[int, object] = {}
            failures: list[BaseException] = []
            start = threading.Barrier(_CAS_WRITERS, timeout=5)

            def _commits(slot: int, agent_id) -> None:
                try:
                    start.wait()
                    results[slot] = registry.commit_cas(
                        artifact.id,
                        agent_id,
                        expected_version=expected,
                        content_hash=f"h{slot}",
                        content=f"round{round_no}-writer{slot}",
                        tick=round_no,
                    )
                except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                    failures.append(exc)

            threads = [
                threading.Thread(target=_commits, args=(slot, agent), daemon=True)
                for slot, agent in enumerate(writers)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
                assert not t.is_alive(), f"round {round_no}: a writer never finished"

            assert not failures, f"round {round_no}: a writer raised: {failures[0]!r}"
            wins = [r for r in results.values() if isinstance(r, tuple)]
            conflicts = [r for r in results.values() if isinstance(r, ConflictDetail)]
            assert len(wins) == 1, (
                f"round {round_no}: {len(wins)} writers won the same "
                f"expected_version={expected} — the arbitration legs and the "
                f"apply are not one serialized step"
            )
            assert len(conflicts) == _CAS_WRITERS - 1
            assert all(c.reason == "version_mismatch" for c in conflicts)
            # The version moved exactly once: to the winner's next_version.
            (updated, _invalidated) = wins[0]
            assert updated.version == expected + 1
            assert registry.get_artifact(artifact.id).version == expected + 1
    finally:
        sys.setswitchinterval(default_switch_interval)


_FETCH_ROUNDS = 20
_FETCH_ITERS = 30
_M_OR_E = (MESIState.MODIFIED, MESIState.EXCLUSIVE)


def test_concurrent_fetch_and_write_never_leave_two_write_holders(registry) -> None:
    """U5's fetch guard: the grant decision and the grant land in one hold.

    ``fetch`` decides EXCLUSIVE-vs-SHARED from a state-map snapshot; without
    the guard a peer ``write`` landing between that read and the grant write
    leaves TWO M/E holders — the single-writer invariant the whole protocol
    exists to keep. Reverting the guard fails this 38-in-40 rounds at a 1 us
    switch interval, surfacing as ``single_writer_violated`` raises from
    whichever path validates first.
    """
    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    writer, reader = uuid4(), uuid4()

    default_switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for round_no in range(_FETCH_ROUNDS):
            barrier = threading.Barrier(2, timeout=5)
            failures: list[BaseException] = []

            def _writes() -> None:
                try:
                    barrier.wait()
                    for t in range(_FETCH_ITERS):
                        svc.write(agent_id=writer, artifact_id=art.id, issued_at_tick=t)
                        svc.invalidate(
                            agent_id=writer, artifact_id=art.id, new_version=1,
                            issuer_agent_id=writer, issued_at_tick=t,
                        )
                except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                    failures.append(exc)

            def _fetches() -> None:
                try:
                    barrier.wait()
                    for t in range(_FETCH_ITERS):
                        svc.fetch(FetchRequest(
                            artifact_id=art.id, requesting_agent_id=reader,
                            requested_at_tick=t,
                        ))
                except BaseException as exc:  # noqa: BLE001 — reported, not swallowed
                    failures.append(exc)

            threads = [
                threading.Thread(target=_writes, daemon=True),
                threading.Thread(target=_fetches, daemon=True),
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)
                assert not t.is_alive(), f"round {round_no}: a worker never finished"

            assert not failures, f"round {round_no}: {failures[0]!r}"
            write_holders = [
                ag for ag, st in registry.get_state_map(art.id).items() if st in _M_OR_E
            ]
            assert len(write_holders) <= 1, (
                f"round {round_no}: {len(write_holders)} write-state holders — "
                "fetch granted from a snapshot a concurrent write had already "
                "invalidated"
            )
    finally:
        sys.setswitchinterval(default_switch_interval)


# ---------------------------------------------------------------------------
# U6 — structural lock coverage: adding an unlocked public member fails CI
# ---------------------------------------------------------------------------
#
# The behavioral tests above prove specific sequences serialize; this section
# proves the CONTRACT structurally, member by member: every public method and
# property of both registries acquires the registry lock, except the named
# construction-time-immutable accessors. A new public member that forgets the
# lock fails here by name, instead of shipping a race.

import inspect
from types import SimpleNamespace

from ccs.coordinator.registry_protocol import CheckpointRecord
from ccs.core.states import TransientState

# The ONLY members allowed to skip the lock: accessors of fields assigned once
# in __init__ (or cached at construction, on sqlite) and never mutated.
# Anything else added here must argue immutability at its declaration site.
EXEMPT_MEMBERS = frozenset({
    "coordinator_epoch",  # _coordinator_epoch: minted/loaded at construction
    "instance_id",        # _instance_id: minted/loaded at construction
    "retention_meta",     # _retain_versions/_retention_policy: constructor-final
})

# The full public surface, frozen (KTD4): discovery below compares against
# these, so a NEW public member cannot appear without being enrolled — and
# enrolling it runs it through the lock check.
INMEM_SURFACE = frozenset({
    "abort_guard", "adjust_checkpoint_pin_refcount", "all_session_meta",
    "artifact_ids", "bind_caller_principal", "capture_version_vector",
    "clear_agent_transient",
    "commit_all", "commit_cas", "conflict_outcome_totals", "coordinator_epoch",
    "artifacts_with_detection_edge", "clear_detection_edges",
    "close_detection_run", "create_checkpoint", "detection_runs",
    "detection_uncoverable", "foreign_write_totals",
    "get_agent_state", "get_agent_transient", "get_artifact",
    "get_artifact_and_generation", "get_caller_principal", "get_checkpoint",
    "get_checkpoint_members",
    "get_content", "get_content_at_version", "get_last_reclamation",
    "get_owner_generation", "get_read_generation", "get_session_cut",
    "get_session_meta", "get_state_map", "get_transient_map",
    "get_transient_tick", "get_version_record", "granted_at_tick",
    "has_artifact", "instance_id", "invalid_reclamations", "last_heartbeat_tick",
    "last_observed_version_for", "list_checkpoints",
    "record_detection_tick", "record_detection_uncoverable",
    "record_foreign_write", "record_heartbeat",
    "record_last_reclamation", "register_artifact", "release_session",
    "remove_artifact", "retention_meta", "session_count", "set_agent_state",
    "set_agent_transient", "set_artifact_and_content",
    "set_checkpoint_member_pin", "set_checkpoint_member_restore",
    "set_checkpoint_restore_status", "valid_holders",
})
SQLITE_SURFACE = INMEM_SURFACE | frozenset({
    "artifact_names_under_prefix", "artifacts_held_by_agent", "close",
    "evict_stale_notices", "get_artifact_updated_at", "last_writer_for",
    "lookup_artifact_id_by_name", "peek_preemption_notice",
    "pop_pending_notices", "pop_preemption_notice", "record_preemption_notice",
    "resolve_or_register", "status_snapshot",
})


class _TrackingRLock:
    """Proxy over ``threading.RLock`` recording acquire/release events.

    Pattern from ``tests/test_diagnose_callback.py``: installed as the
    registry's ``_lock`` so a member body that holds the lock is observed
    doing so.
    """

    def __init__(self) -> None:
        self._inner = threading.RLock()
        self.events: list[str] = []

    def acquire(self, *args, **kwargs):
        self.events.append("acquire")
        return self._inner.acquire(*args, **kwargs)

    def release(self) -> None:
        self.events.append("release")
        self._inner.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


def _public_members(cls) -> dict[str, str]:
    """Discover the public surface: {name: "method" | "property"}."""
    surface: dict[str, str] = {}
    for name in dir(cls):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(cls, name)
        surface[name] = "property" if isinstance(attr, property) else "method"
    return surface


def _synth_value(param: inspect.Parameter):
    """One plausible argument per required parameter, by name then annotation.

    The value only needs to get the call INTO the method body — a KeyError on
    an unknown artifact is fine, because the lock acquire is the body's first
    act and the check swallows the exception.
    """
    name, ann = param.name, str(param.annotation)
    if name == "artifact":
        return Artifact(id=uuid4(), name="synth.md", version=1)
    if name == "state":
        return MESIState.SHARED
    if name == "states":
        return [MESIState.SHARED]
    if name == "transient_state":
        return TransientState.ISG
    if name == "checkpoint":
        # Real header: create_checkpoint fail-closed-validates owner and
        # duplicates BEFORE its lock, so a bogus value never reaches the hold.
        return CheckpointRecord(
            checkpoint_id="synth", name="synth", owner=uuid4(),
            created_at=1.0, created_at_tick=1, window_min=1.0, window_max=1.0,
        )
    if name == "artifact_ids":
        return [uuid4()]
    if name == "outcome":
        # record_foreign_write fail-closed-validates the outcome against the
        # fixed three-value vocabulary BEFORE its lock, so a bogus string never
        # reaches the hold — same shape as `checkpoint` above.
        return FOREIGN_WRITE_OUTCOMES[0]
    if name == "members":
        return []
    if name == "writes":
        return {uuid4(): SimpleNamespace(
            expected_version=1, content_hash="h", size_tokens=None, content=None
        )}
    if name == "read_set":
        return [uuid4()]
    if name == "abort":
        return None
    if "UUID" in ann or name.endswith("agent_id") or name.endswith("artifact_id") or name == "owner":
        return uuid4()
    if "float" in ann:
        return 1.0
    if "int" in ann:
        return 1
    if "bool" in ann:
        return False
    return "x"


def _assert_member_holds_lock(reg, name: str, kind: str) -> None:
    """Install the tracker, drive the member, and require an observed hold."""
    tracker = _TrackingRLock()
    reg._lock = tracker  # noqa: SLF001 — the seam under test

    if kind == "property":
        getattr(type(reg), name).fget(reg)
    elif name == "abort_guard":
        # A context manager: a plain call returns the manager without running
        # the body, observing no acquire — enter it properly.
        with reg.abort_guard():
            pass
    else:
        method = getattr(reg, name)
        kwargs = {
            p.name: _synth_value(p)
            for p in inspect.signature(method).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        }
        try:
            method(**kwargs)
        except Exception:  # noqa: BLE001 — only the acquire is under test
            pass

    assert tracker.events.count("acquire") >= 1, (
        f"{type(reg).__name__}.{name} ran without acquiring the registry lock "
        "— either serialize it or argue construction-time immutability in "
        "EXEMPT_MEMBERS"
    )
    assert tracker.events.count("acquire") == tracker.events.count("release"), (
        f"{type(reg).__name__}.{name} left the lock held: "
        f"{tracker.events.count('acquire')} acquires vs "
        f"{tracker.events.count('release')} releases"
    )


def test_inmem_surface_is_enrolled() -> None:
    """A new public member must be added to the frozen list — and thereby
    run through the lock check."""
    assert set(_public_members(ArtifactRegistry)) == set(INMEM_SURFACE)


def test_sqlite_surface_is_enrolled() -> None:
    assert set(_public_members(SqliteArtifactRegistry)) == set(SQLITE_SURFACE)


def test_exemptions_are_only_the_immutable_accessors() -> None:
    assert EXEMPT_MEMBERS == {"coordinator_epoch", "instance_id", "retention_meta"}


@pytest.mark.parametrize("member", sorted(INMEM_SURFACE - EXEMPT_MEMBERS))
def test_inmem_member_holds_the_lock(member: str) -> None:
    reg = ArtifactRegistry()
    _assert_member_holds_lock(reg, member, _public_members(ArtifactRegistry)[member])


@pytest.mark.parametrize("member", sorted(SQLITE_SURFACE - EXEMPT_MEMBERS))
def test_sqlite_member_holds_the_lock(member: str, tmp_path: Path) -> None:
    with SqliteArtifactRegistry(tmp_path / "cov.db") as reg:
        _assert_member_holds_lock(reg, member, _public_members(SqliteArtifactRegistry)[member])


def test_teeth_an_unlocked_member_fails_by_name() -> None:
    """The check must actually bite: a stub with one unlocked public method
    fails, and the failure names that method."""

    class _Unlocked:
        def __init__(self) -> None:
            self._lock = threading.RLock()

        def rogue_method(self) -> int:
            return 1  # touches no lock

    with pytest.raises(AssertionError, match="rogue_method"):
        _assert_member_holds_lock(_Unlocked(), "rogue_method", "method")


# ---------------------------------------------------------------------------
# Absent-artifact tolerance: the read accessors must match sqlite's
# SELECT-no-row behavior, because the thread-safety contract makes
# delete-vs-anything interleavings supported states, not caller errors.
# ---------------------------------------------------------------------------

_ABSENT_ACCESSOR_CASES = [
    ("get_agent_state", lambda reg, art, ag: reg.get_agent_state(art, ag), None),
    ("get_state_map", lambda reg, art, ag: reg.get_state_map(art), {}),
    ("get_agent_transient", lambda reg, art, ag: reg.get_agent_transient(art, ag), None),
    ("get_transient_map", lambda reg, art, ag: reg.get_transient_map(art), {}),
    ("get_transient_tick", lambda reg, art, ag: reg.get_transient_tick(art, ag), None),
    ("valid_holders", lambda reg, art, ag: reg.valid_holders(art), []),
    ("get_read_generation", lambda reg, art, ag: reg.get_read_generation(art, ag), None),
    (
        "last_observed_version_for",
        lambda reg, art, ag: reg.last_observed_version_for(art, ag),
        None,
    ),
]


@pytest.mark.parametrize(
    "accessor_name,call,expected",
    _ABSENT_ACCESSOR_CASES,
    ids=[c[0] for c in _ABSENT_ACCESSOR_CASES],
)
def test_read_accessors_tolerate_an_absent_artifact(registry, accessor_name, call, expected) -> None:
    """An absent artifact answers like sqlite's empty SELECT, never KeyError.

    The sweeps' per-pair holds re-read state for pairs snapshotted BEFORE a
    concurrent delete landed; on sqlite those reads return None/{}/[] and the
    pair is skipped, while the in-memory registry used to raise KeyError and
    crash the whole sweep. get_owner_generation stays KeyError-raising by
    documented contract on BOTH backends and is deliberately absent here.
    """
    assert call(registry, uuid4(), uuid4()) == expected


class _StaleSnapshotRegistry(ArtifactRegistry):
    """Serve the sweeps a pre-delete view of which pairs exist.

    Reproduces deterministically the interleaving the thread-safety contract
    permits: a sweep builds its walk list, a peer deletes the artifact, and
    only then does the sweep's per-pair hold re-read live state. Overrides the
    SNAPSHOT sources only -- every in-hold read stays live.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stale_ids: list = []
        self.stale_state_maps: dict = {}
        self.stale_transient_maps: dict = {}

    def capture_stale_snapshot(self) -> None:
        self.stale_ids = super().artifact_ids()
        self.stale_state_maps = {a: super(type(self), self).get_state_map(a) for a in self.stale_ids}
        self.stale_transient_maps = {
            a: super(type(self), self).get_transient_map(a) for a in self.stale_ids
        }

    def artifact_ids(self):
        return list(self.stale_ids) if self.stale_ids else super().artifact_ids()

    def get_state_map(self, artifact_id):
        if artifact_id in self.stale_state_maps:
            return dict(self.stale_state_maps[artifact_id])
        return super().get_state_map(artifact_id)

    def get_transient_map(self, artifact_id):
        if artifact_id in self.stale_transient_maps:
            return dict(self.stale_transient_maps[artifact_id])
        return super().get_transient_map(artifact_id)


def test_stable_sweep_survives_a_delete_between_snapshot_and_hold() -> None:
    """The grant sweep skips a vanished pair instead of crashing the walk."""
    reg = _StaleSnapshotRegistry()
    svc = CoordinatorService(reg)
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    reg.set_agent_state(art.id, agent, MESIState.EXCLUSIVE, trigger="write", tick=0)

    reg.capture_stale_snapshot()
    svc.delete(agent_id=agent, artifact_id=art.id)

    reclaimed = svc.enforce_stable_grant_timeouts(
        current_tick=1_000, heartbeat_timeout_ticks=10, max_hold_ticks=10
    )
    assert reclaimed == 0


def test_transient_sweep_survives_a_delete_between_snapshot_and_hold() -> None:
    """The transient sweep skips a vanished pair instead of crashing the walk."""
    from ccs.core.states import TransientState

    reg = _StaleSnapshotRegistry()
    svc = CoordinatorService(reg)
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    reg.set_agent_transient(art.id, agent, TransientState.ISG, entered_tick=0)

    reg.capture_stale_snapshot()
    svc.delete(agent_id=agent, artifact_id=art.id)

    expired = svc.enforce_transient_timeouts(current_tick=1_000, timeout_ticks=10)
    assert expired == 0


def test_transient_sweep_skips_a_pair_cleared_during_the_walk(registry) -> None:
    """The in-hold re-read owns the eviction decision, not the stale snapshot.

    A pair whose transient CLEARED after the walk list was built (the grant
    completed) must not be force-invalidated on the stale snapshot's say-so.
    Exercises the `get_agent_transient(...) is None: continue` guard the
    per-pair hold added; `get_transient_tick is None` is the second net
    behind it.
    """
    from ccs.core.states import TransientState

    svc = CoordinatorService(registry)
    art = svc.register_artifact(name="plan.md", content="v1")
    agent = uuid4()
    registry.set_agent_transient(art.id, agent, TransientState.ISG, entered_tick=0)
    registry.set_agent_state(art.id, agent, MESIState.SHARED, trigger="fetch", tick=1_000)

    if isinstance(registry, ArtifactRegistry):
        stale = _StaleSnapshotRegistry()
        # Rebuild the settled state on the wrapper so the stale snapshot can
        # carry the pre-clear transient while live state has none.
        svc = CoordinatorService(stale)
        art = svc.register_artifact(name="plan.md", content="v1")
        stale.set_agent_transient(art.id, agent, TransientState.ISG, entered_tick=0)
        stale.capture_stale_snapshot()
        stale.clear_agent_transient(art.id, agent)
        stale.set_agent_state(art.id, agent, MESIState.SHARED, trigger="fetch", tick=1_000)
        expired = svc.enforce_transient_timeouts(current_tick=1_000, timeout_ticks=10)
        assert expired == 0
        assert stale.get_agent_state(art.id, agent) == MESIState.SHARED, (
            "the sweep evicted a pair whose transient had already cleared -- "
            "the settled grant was destroyed on stale snapshot data"
        )
    else:
        # sqlite arm: the live-read path (no stale-snapshot seam on this
        # backend); the cleared pair is simply absent from the walk.
        registry.clear_agent_transient(art.id, agent)
        expired = svc.enforce_transient_timeouts(current_tick=1_000, timeout_ticks=10)
        assert expired == 0
        assert registry.get_agent_state(art.id, agent) == MESIState.SHARED
