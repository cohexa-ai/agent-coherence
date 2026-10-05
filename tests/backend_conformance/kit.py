# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The conformance scenarios — parametrizable over a registry factory.

Every scenario is a plain function taking a :class:`RegistryFactory` (a callable
that mints a fresh registry of one arm, tracking handles to close at teardown).
The scenarios split into two structurally-distinct families — the split the plan
demands be MECHANICAL, not a naming convention:

**MUST-MATCH** (``assert_*`` functions taking only a :class:`RegistryFactory`):
contract behaviors that MUST be identical on every conforming arm — CAS
arbitration, the read-generation fence (including admit-on-absent and the
verdict staying sticky across a peer's fetch), single-writer under
barrier-forced contention, and session fail-closed refusals. Each
hard-asserts the SAME correctness property regardless of arm. There is no arm
parameter and no expected-outcome parameter — the property is universal, so a
must-match function CANNOT express "arm X may differ".

**BACKEND-DEFINED** (``assert_declared_*`` functions taking a
:class:`RegistryFactory` AND a :class:`RestartDeclaration`): behaviors the
contract explicitly lets backends DIVERGE on — restart survival, sweep timing.
The function asserts the arm behaves *as ITS OWN declaration says* — the caller
passes ``RestartDeclaration.SURVIVES`` for sqlite and ``RestartDeclaration.LOST``
for in-memory, and the SAME function verifies each against its own declared
disposition. Because the signature REQUIRES a per-arm declaration, a
backend-defined scenario is structurally incapable of asserting cross-arm
identity — that is the mechanism that keeps the split honest (a reviewer cannot
accidentally turn restart-survival into a must-match assertion; the type won't
let them).

The :class:`RegistryFactory` protocol is deliberately narrow (mint one arm, close
handles) so a degraded-stub factory (``tests/test_kit_teeth.py``) plugs in
identically — the teeth test runs the SAME single-writer scenario against a
wrapper whose ``commit_cas`` skips the grant arbitration, and the scenario FAILS
it (proving the kit discriminates).
"""

from __future__ import annotations

import hashlib
import threading
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol
from uuid import UUID, uuid4

from ccs.coordinator.backend_contract import R18_LIVENESS_SOURCE, LivenessSourceObligation
from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    SESSION_INVALIDATED_REASON,
    STALE_READ_GENERATION_REASON,
    VERSION_MISMATCH_REASON,
    SessionInvalidated,
)
from ccs.core.states import MESIState
from ccs.core.types import (
    Artifact,
    ConflictDetail,
    FetchRequest,
    SessionReadRejection,
    SnapshotSession,
    VersionedContent,
)

# The reclaim trigger the fence keys on (a RECLAIM_TRIGGER bumps owner_generation
# WITHOUT a version move — exactly what version-CAS cannot see). Named here as the
# wire-stable constant, matched by identity, never by substring.
_RECLAIM_TRIGGER = "reclaim_heartbeat"

# The voluntary-release trigger. It is NOT a reclaim, but it ends a write claim
# at an UNCHANGED version exactly as a reclaim does, so R9 requires it to move
# the ownership epoch too (EPOCH_BUMP_TRIGGERS = RECLAIM_TRIGGERS + this). Named
# here as the wire-stable constant, matched by identity, never by substring.
_RELEASE_TRIGGER = "invalidate"

# The handoff trigger: a transfer moves its giver INVALID under this trigger,
# distinct from the release so the state log can tell the two apart. Given up
# from EXCLUSIVE or MODIFIED it ends a write claim at an UNCHANGED version
# exactly as a release does, so it moves the ownership epoch too; given up from
# SHARED it revokes no write claim and must not. Named here as the wire-stable
# literal, matched by identity, never by substring.
_HANDOFF_TRIGGER = "handoff"

# The single-writer conflict reason the arbitration leg emits when a version-
# matching OCC writer meets a pessimistic M/E peer. The registries return it as a
# bare ``ConflictDetail`` literal (there is no exported OTHER_HOLDER_REASON
# constant — the Literal in ``core.types`` IS the wire contract), so the kit pins
# the literal here as its single source of truth.
OTHER_HOLDER_REASON = "other_holder"


# ---------------------------------------------------------------------------
# The factory protocol — the one seam every arm (and the degraded stub) plugs in.
# ---------------------------------------------------------------------------


class RegistryFactory(Protocol):
    """Mint a fresh registry of one arm. Every scenario takes one of these.

    ``__call__`` returns a registry satisfying (at least) ``RegistryBase``. The
    factory OWNS handle lifecycle: it tracks what it mints and closes durable
    handles when :meth:`close_all` runs (the pytest fixture calls it at teardown).
    A ``db_path`` is exposed so the restart scenario can re-open the SAME durable
    store on a fresh handle (the sqlite "restart"); an in-memory factory leaves it
    ``None`` (a fresh in-memory instance is a fresh, empty store — the declared
    restart-loss).
    """

    def __call__(self) -> object:
        ...

    def close_all(self) -> None:
        ...

    @property
    def db_path(self) -> Path | None:
        ...


@dataclass
class InMemoryFactory:
    """Mints :class:`ArtifactRegistry`. Restart is LOSS: a fresh instance is a
    fresh empty store, so :attr:`db_path` is ``None`` — there is no shared store a
    "restart" could re-open."""

    retain_versions: bool = True

    def __call__(self) -> ArtifactRegistry:
        return ArtifactRegistry(retain_versions=self.retain_versions)

    def close_all(self) -> None:
        # In-memory registries hold no OS resource to release.
        return None

    @property
    def db_path(self) -> Path | None:
        return None


@dataclass
class SqliteFactory:
    """Mints :class:`SqliteArtifactRegistry` handles over ONE db file under
    ``tmp_path`` (isolated per test; never the shared ``state.db``, never a
    ``user_version`` bump, never a foreign-lineage marker). Tracks every handle so
    :meth:`close_all` closes them at teardown. Two calls return two INDEPENDENT
    handles on the SAME file — the real two-connection concurrency arm."""

    tmp_path: Path
    retain_versions: bool = True
    _handles: list[SqliteArtifactRegistry] = field(default_factory=list)

    def __call__(self) -> SqliteArtifactRegistry:
        reg = SqliteArtifactRegistry(
            self.tmp_path / "conformance.db", retain_versions=self.retain_versions
        )
        self._handles.append(reg)
        return reg

    def close_all(self) -> None:
        for reg in self._handles:
            reg.close()
        self._handles.clear()

    @property
    def db_path(self) -> Path | None:
        return self.tmp_path / "conformance.db"


# ---------------------------------------------------------------------------
# Small helpers shared across scenarios.
# ---------------------------------------------------------------------------


def _register(
    reg: object, artifact_id: UUID, *, version: int = 1, name: str = "plan.md"
) -> Artifact:
    art = Artifact(id=artifact_id, name=name, version=version, content_hash="h-init")
    reg.register_artifact(art, content="")  # type: ignore[attr-defined]
    return art


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _seed_shared_writer(reg: object, artifact_id: UUID, writer: UUID, *, tick: int = 1) -> None:
    """Put ``writer`` in SHARED on ``artifact_id`` (OCC-eligible; the grant is
    elected by the version check, not ``other_holder``)."""
    reg.set_agent_state(artifact_id, writer, MESIState.SHARED, tick=tick)  # type: ignore[attr-defined]


# ===========================================================================
# MUST-MATCH scenarios — identical correctness property on every arm.
# Signature: (factory) -> None. No arm param, no expected-outcome param.
# ===========================================================================


def assert_cas_arbitration_one_winner(factory: RegistryFactory) -> None:
    """CAS arbitration (MUST-MATCH). Two writers hold the SAME stale version N and
    each computes DISTINCT content from that one N; both call ``commit_cas`` under
    a barrier. EXACTLY one wins (returns ``(Artifact, list)``); the loser gets a
    typed ``ConflictDetail(version_mismatch)``; NEVER two winners, and the final
    version is N+1 (the loser's stale buffer did not clobber).

    Uses TWO independent handles on the durable arm (the real two-connection
    race); the in-memory arm degrades to two writers over the single store object.
    Only the correctness property is a hard assert; "did the threads overlap" is a
    ``RuntimeWarning`` (a constrained runner may serialize)."""
    writer_reg = factory()
    peer_reg = factory() if factory.db_path is not None else writer_reg
    artifact_id = uuid4()
    _register(writer_reg, artifact_id, version=1)

    writers = [uuid4(), uuid4()]
    # Each writer is SHARED on the store it commits through (sqlite: seed on both
    # handles' shared file; in-memory: one object).
    _seed_shared_writer(writer_reg, artifact_id, writers[0])
    _seed_shared_writer(peer_reg, artifact_id, writers[1])
    regs = {writers[0]: writer_reg, writers[1]: peer_reg}

    results: dict[UUID, object] = {}
    barrier = threading.Barrier(len(writers))
    overlap = threading.Event()
    in_flight = {"n": 0}
    lock = threading.Lock()

    def attempt(writer_id: UUID) -> None:
        content_hash = _hash(str(writer_id))  # FIXED stale buffer, keyed on identity
        barrier.wait()
        with lock:
            in_flight["n"] += 1
            if in_flight["n"] > 1:
                overlap.set()
        try:
            results[writer_id] = regs[writer_id].commit_cas(  # type: ignore[attr-defined]
                artifact_id, writer_id, expected_version=1, content_hash=content_hash, tick=5
            )
        finally:
            with lock:
                in_flight["n"] -= 1

    threads = [threading.Thread(target=attempt, args=(w,)) for w in writers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wins = [r for r in results.values() if isinstance(r, tuple)]
    conflicts = [r for r in results.values() if isinstance(r, ConflictDetail)]
    assert len(wins) == 1, f"single-writer CAS must elect exactly one winner, got {results}"
    assert len(conflicts) == 1, f"the loser must get a typed ConflictDetail, got {results}"
    # Two OCC (SHARED) writers → the loss is version-elected, so the reason is
    # ALWAYS version_mismatch (matched against the wire-stable constant).
    assert conflicts[0].reason == VERSION_MISMATCH_REASON, conflicts[0].reason
    assert conflicts[0].current_version == 2

    # The final version advanced by EXACTLY one — no double bump from the loser.
    final = writer_reg.get_artifact(artifact_id)  # type: ignore[attr-defined]
    assert final.version == 2
    winner_id = next(w for w, r in results.items() if isinstance(r, tuple))
    assert final.content_hash == _hash(str(winner_id)), "the loser silently clobbered the winner"

    if not overlap.is_set():
        warnings.warn(
            "CAS arbitration round did not observe overlapping in-flight writers; "
            "the runner may have serialized threads. The correctness property is "
            "still asserted; the race window was not stressed this run.",
            RuntimeWarning,
            stacklevel=2,
        )


def assert_single_writer_under_contention(factory: RegistryFactory) -> None:
    """Single-writer under contention (MUST-MATCH — the TEETH scenario). A
    pessimistic peer holds MODIFIED; an OCC writer at the CORRECT current version
    calls ``commit_cas``. The grant-arbitration leg of the R9 boundary MUST reject
    it with ``ConflictDetail(other_holder)`` — the version compare ALONE would let
    it through (the version matches), so this scenario isolates the grant leg.

    A barrier forces the OCC writer to fire while the M/E grant is held. The
    degraded stub (``test_kit_teeth.py``) skips exactly this grant check, so its
    OCC writer WINS while the M/E holder still holds — two writers — and this
    assertion fails on it. The failure message NAMES the missing tuple element
    (grant arbitration / other_holder)."""
    reg = factory()
    peer_reg = factory() if factory.db_path is not None else reg
    artifact_id = uuid4()
    _register(reg, artifact_id, version=1)

    holder = uuid4()
    occ_writer = uuid4()
    # The pessimistic peer takes MODIFIED (a genuine single-writer grant).
    reg.set_agent_state(artifact_id, holder, MESIState.MODIFIED, tick=1)  # type: ignore[attr-defined]
    # The OCC writer is SHARED and commits at the CORRECT current version (1) — so
    # only the grant leg, not the version leg, can reject it.
    _seed_shared_writer(peer_reg, artifact_id, occ_writer)

    barrier = threading.Barrier(2)
    result: dict[str, object] = {}

    def hold() -> None:
        barrier.wait()
        # The holder does nothing but keep MODIFIED live across the window.

    def commit() -> None:
        barrier.wait()
        result["occ"] = peer_reg.commit_cas(  # type: ignore[attr-defined]
            artifact_id, occ_writer, expected_version=1, content_hash=_hash("occ"), tick=5
        )

    th = threading.Thread(target=hold)
    tc = threading.Thread(target=commit)
    th.start()
    tc.start()
    th.join()
    tc.join()

    outcome = result["occ"]
    assert isinstance(outcome, ConflictDetail), (
        "single-writer VIOLATED: an OCC writer at the correct version WON while a "
        "peer held MODIFIED — the R9 grant-arbitration leg (other_holder over "
        "state_by_agent) was NOT enforced in the atomic step. The missing tuple "
        f"element is grant arbitration ({OTHER_HOLDER_REASON}); commit_cas returned "
        f"{outcome!r}"
    )
    assert outcome.reason == OTHER_HOLDER_REASON, (
        f"single-writer rejection carried the wrong reason: expected "
        f"{OTHER_HOLDER_REASON!r} (grant-arbitration leg), got {outcome.reason!r}"
    )
    # The version did NOT move — the arbitration rejected before any mutation.
    assert reg.get_artifact(artifact_id).version == 1  # type: ignore[attr-defined]


def assert_fence_rejects_superseded_read_generation(factory: RegistryFactory) -> None:
    """Read-generation fence — REJECT leg (MUST-MATCH). An agent acquires
    EXCLUSIVE (capturing its ``read_generation`` at the current ``owner_generation``);
    a sweep-class reclaim (``reclaim_heartbeat``) drives it to INVALID, which bumps
    ``owner_generation`` while the agent keeps its now-STALE ``read_generation``;
    the agent re-grants SHARED and commits at the correct version. The fence MUST
    reject it with ``ConflictDetail(stale_read_generation)`` — the version never
    moved, so version-CAS alone cannot catch this reclaim-zombie."""
    reg = factory()
    artifact_id = uuid4()
    _register(reg, artifact_id, version=1)
    agent = uuid4()

    # Acquire EXCLUSIVE → captures read_generation = owner_generation (0).
    reg.set_agent_state(artifact_id, agent, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    captured = reg.get_read_generation(artifact_id, agent)  # type: ignore[attr-defined]
    assert captured is not None, "an M/E acquire must capture a read_generation"

    # Sweep-class reclaim → INVALID with a RECLAIM_TRIGGER bumps owner_generation.
    reg.set_agent_state(  # type: ignore[attr-defined]
        artifact_id, agent, MESIState.INVALID, trigger=_RECLAIM_TRIGGER, tick=2
    )
    assert reg.get_owner_generation(artifact_id) > captured, (  # type: ignore[attr-defined]
        "a sweep-class reclaim must bump owner_generation past the captured "
        "read_generation, else the fence has nothing to reject"
    )

    # Re-grant SHARED (a re-read would land it back in S) WITHOUT a fresh capture:
    # a SHARED re-grant under any trigger but the agent's OWN "fetch" read does
    # not re-capture, so the stale read_generation survives.
    reg.set_agent_state(artifact_id, agent, MESIState.SHARED, trigger="peer_regrant", tick=3)  # type: ignore[attr-defined]

    result = reg.commit_cas(  # type: ignore[attr-defined]
        artifact_id, agent, expected_version=1, content_hash=_hash("zombie"), tick=4
    )
    assert isinstance(result, ConflictDetail), (
        "the fence FAILED to reject a reclaim-zombie: a superseded read_generation "
        f"committed while the version was unchanged, got {result!r}"
    )
    assert result.reason == STALE_READ_GENERATION_REASON, result.reason
    # No mutation — the version stayed put.
    assert reg.get_artifact(artifact_id).version == 1  # type: ignore[attr-defined]


def assert_peer_fetch_does_not_rearm_superseded_read_generation(factory: RegistryFactory) -> None:
    """Read-generation fence — PEER-FETCH leg (MUST-MATCH). The fence's verdict
    is STICKY: once a reclaimed holder's commit is refused
    ``stale_read_generation``, nothing ANOTHER agent does can make that same
    commit land — only the holder's OWN re-read or re-acquire re-captures. The
    reclaim-zombie is seeded exactly as in
    :func:`assert_fence_rejects_superseded_read_generation` (acquire, sweep-class
    reclaim, non-capturing SHARED re-grant); then a DIFFERENT agent reads the
    artifact through ``CoordinatorService.fetch`` — the producer of every
    ``"fetch"`` set. Afterwards the zombie's ``read_generation`` MUST be
    unchanged, its ``commit_cas`` at the unchanged version MUST still be refused,
    and the version MUST NOT move.

    The leg this pins: a peer's read is not the zombie's read. A backend whose
    ``"fetch"`` handling re-lists every non-INVALID holder under the capture
    trigger refreshes the zombie's operand on a read it never made, and the
    commit the fence just refused then LANDS at the very version it was refused
    at — a revoked write claim honored because someone ELSE read. Both shipped
    arms pass because ``fetch`` downgrades only an M/E holder and leaves a peer
    that is already SHARED alone, so no ``"fetch"`` set ever reaches the zombie;
    the re-arming wrapper in ``test_kit_teeth.py`` reproduces the re-listing
    inside the backend and FAILS here."""
    reg = factory()
    svc = CoordinatorService(reg)
    artifact_id = uuid4()
    _register(reg, artifact_id, version=1)
    zombie, peer = uuid4(), uuid4()

    # Acquire EXCLUSIVE → captures read_generation = owner_generation (0).
    reg.set_agent_state(artifact_id, zombie, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    captured = reg.get_read_generation(artifact_id, zombie)  # type: ignore[attr-defined]
    assert captured is not None, "an M/E acquire must capture a read_generation"

    # Sweep-class reclaim → INVALID with a RECLAIM_TRIGGER bumps owner_generation.
    reg.set_agent_state(  # type: ignore[attr-defined]
        artifact_id, zombie, MESIState.INVALID, trigger=_RECLAIM_TRIGGER, tick=2
    )
    epoch = reg.get_owner_generation(artifact_id)  # type: ignore[attr-defined]
    assert epoch > captured, (
        "a sweep-class reclaim must bump owner_generation past the captured "
        "read_generation, else the fence has nothing to reject"
    )

    # Re-grant SHARED WITHOUT a fresh capture (not the zombie's own "fetch" read),
    # so the superseded read_generation survives into the peer's read below.
    reg.set_agent_state(artifact_id, zombie, MESIState.SHARED, trigger="peer_regrant", tick=3)  # type: ignore[attr-defined]
    assert reg.get_read_generation(artifact_id, zombie) == captured, (  # type: ignore[attr-defined]
        "scenario precondition: a SHARED re-grant under a non-capture trigger must "
        "preserve the superseded read_generation"
    )

    # A DIFFERENT agent's genuine read, through the service. The zombie is a
    # non-INVALID peer, so the requester is granted SHARED and captures for
    # ITSELF — the one capture a fetch is allowed to make.
    response = svc.fetch(
        FetchRequest(artifact_id=artifact_id, requesting_agent_id=peer, requested_at_tick=4)
    )
    assert response.state_grant == MESIState.SHARED
    assert reg.get_read_generation(artifact_id, peer) == epoch, (  # type: ignore[attr-defined]
        "the requester's OWN read must capture the current owner_generation"
    )

    # The zombie's operand is untouched: nobody read on its behalf.
    rearmed = reg.get_read_generation(artifact_id, zombie)  # type: ignore[attr-defined]
    assert rearmed == captured, (
        "a PEER's fetch RE-ARMED the fence: the reclaimed holder's read_generation "
        f"moved from {captured} to {rearmed} on a read it never made. A backend must "
        "write an agent's read_generation only on that agent's OWN acquire or OWN "
        "read — never because another agent read the artifact"
    )
    result = reg.commit_cas(  # type: ignore[attr-defined]
        artifact_id, zombie, expected_version=1, content_hash=_hash("zombie"), tick=5
    )
    assert isinstance(result, ConflictDetail), (
        "a PEER's fetch RE-ARMED the fence: the reclaim-zombie's commit at the "
        f"unchanged version was ADMITTED after another agent read, got {result!r}"
    )
    assert result.reason == STALE_READ_GENERATION_REASON, result.reason
    # No mutation — the version stayed put.
    assert reg.get_artifact(artifact_id).version == 1  # type: ignore[attr-defined]


def assert_epoch_moves_on_voluntary_release(factory: RegistryFactory) -> None:
    """Read-generation fence — RELEASE-BUMP leg (MUST-MATCH). The epoch rule is
    not "the sweep did it" but "a write claim was revoked WITHOUT the version
    moving", which is the one condition version-CAS is structurally blind to. A
    voluntary release ends a claim at an unchanged version exactly as a reclaim
    does, so it MUST move ``owner_generation`` too.

    Without this leg a backend can revoke a grant through the release path,
    leave the epoch behind, and then ADMIT the very commit the reject leg above
    exists to catch — while still passing conformance. The failure is silent on
    both sides: the version never moves, so version-CAS sees nothing, and the
    epoch never moves, so the fence has nothing to compare.

    Contrast with :func:`assert_fence_rejects_superseded_read_generation`: that
    leg pins the fence's VERDICT, this one pins the operand the verdict needs."""
    reg = factory()
    artifact_id = uuid4()
    _register(reg, artifact_id, version=1)
    agent, peer = uuid4(), uuid4()

    # Acquire EXCLUSIVE → captures read_generation at the current epoch.
    reg.set_agent_state(artifact_id, agent, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    before = reg.get_owner_generation(artifact_id)  # type: ignore[attr-defined]
    captured = reg.get_read_generation(artifact_id, agent)  # type: ignore[attr-defined]
    assert captured is not None, "an M/E acquire must capture a read_generation"

    # Voluntary release — NOT a reclaim, and the version does not move.
    reg.set_agent_state(  # type: ignore[attr-defined]
        artifact_id, agent, MESIState.INVALID, trigger=_RELEASE_TRIGGER, tick=2
    )
    assert reg.get_artifact(artifact_id).version == 1, (  # type: ignore[attr-defined]
        "the release leg must not move the version — that is the whole point"
    )
    assert reg.get_owner_generation(artifact_id) > before, (  # type: ignore[attr-defined]
        "a voluntary release revoked a write claim without moving "
        "owner_generation: the fence can never arm for this holder afterwards, "
        "because no sweep has an M/E grant left to reclaim"
    )

    # And the epoch move is load-bearing: the released holder's commit at the
    # unchanged version is now rejected, exactly as a reclaimed holder's is.
    reg.set_agent_state(artifact_id, agent, MESIState.SHARED, trigger="peer_regrant", tick=3)  # type: ignore[attr-defined]
    result = reg.commit_cas(  # type: ignore[attr-defined]
        artifact_id, agent, expected_version=1, content_hash=_hash("zombie"), tick=4
    )
    assert isinstance(result, ConflictDetail), (
        "the released holder's commit was ADMITTED at an unchanged version; the "
        f"release-path epoch move is not load-bearing, got {result!r}"
    )
    assert result.reason == STALE_READ_GENERATION_REASON, result.reason
    assert reg.get_artifact(artifact_id).version == 1  # type: ignore[attr-defined]

    # A peer-invalidation (the version-MOVING revoke class) must NOT bump: that
    # path is arbitrated by version-CAS, and bumping there would over-fence.
    reg.set_agent_state(artifact_id, peer, MESIState.EXCLUSIVE, tick=5)  # type: ignore[attr-defined]
    steady = reg.get_owner_generation(artifact_id)  # type: ignore[attr-defined]
    reg.set_agent_state(artifact_id, peer, MESIState.INVALID, trigger="commit", tick=6)  # type: ignore[attr-defined]
    assert reg.get_owner_generation(artifact_id) == steady, (  # type: ignore[attr-defined]
        "a version-moving peer invalidation must NOT move the epoch"
    )


def assert_epoch_moves_on_handoff(factory: RegistryFactory) -> None:
    """Read-generation fence — HANDOFF-BUMP leg (MUST-MATCH). The twin of
    :func:`assert_epoch_moves_on_voluntary_release` for the handoff trigger.
    A giver that hands a path off from EXCLUSIVE or MODIFIED ends its write
    claim at an unchanged version, exactly as a release does, so the move to
    INVALID MUST move ``owner_generation``; a giver that hands off a standing
    SHARED read revoked no write claim, so its move MUST NOT.

    Each leg fails a different backend. One that leaves the epoch behind on a
    write-shape handoff ADMITS the giver's late commit at the unchanged version
    — the lost update the handoff exists to prevent, invisible to version-CAS.
    One that bumps on a SHARED-shape handoff REJECTS every bystander that read
    the path, as ``stale_read_generation``, for a revocation that never
    happened. Both legs are made load-bearing below, not just observed."""
    reg = factory()
    exclusive_art, modified_art, shared_art = uuid4(), uuid4(), uuid4()
    # One artifact per hold shape, so each leg starts from its own clean epoch.
    for name, artifact_id in (
        ("exclusive.md", exclusive_art), ("modified.md", modified_art), ("shared.md", shared_art),
    ):
        _register(reg, artifact_id, version=1, name=name)
    giver, bystander = uuid4(), uuid4()

    # EXCLUSIVE giver: the acquire captures read_generation at the epoch.
    reg.set_agent_state(exclusive_art, giver, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    before = reg.get_owner_generation(exclusive_art)  # type: ignore[attr-defined]
    assert reg.get_read_generation(exclusive_art, giver) is not None, (  # type: ignore[attr-defined]
        "an M/E acquire must capture a read_generation"
    )
    reg.set_agent_state(  # type: ignore[attr-defined]
        exclusive_art, giver, MESIState.INVALID, trigger=_HANDOFF_TRIGGER, tick=2
    )
    assert reg.get_artifact(exclusive_art).version == 1  # type: ignore[attr-defined]
    assert reg.get_owner_generation(exclusive_art) > before, (  # type: ignore[attr-defined]
        "an EXCLUSIVE giver's handoff revoked its write claim without moving "
        "owner_generation: its late commit at the unchanged version can no "
        "longer be fenced"
    )
    # Load-bearing: the giver's commit at the unchanged version is rejected,
    # exactly as a released holder's is.
    reg.set_agent_state(exclusive_art, giver, MESIState.SHARED, trigger="peer_regrant", tick=3)  # type: ignore[attr-defined]
    result = reg.commit_cas(  # type: ignore[attr-defined]
        exclusive_art, giver, expected_version=1, content_hash=_hash("late"), tick=4
    )
    assert isinstance(result, ConflictDetail), (
        "the EXCLUSIVE giver's commit was ADMITTED at the unchanged version after "
        f"its handoff; the handoff-path epoch move is not load-bearing, got {result!r}"
    )
    assert result.reason == STALE_READ_GENERATION_REASON, result.reason
    assert reg.get_artifact(exclusive_art).version == 1  # type: ignore[attr-defined]

    # MODIFIED giver: the same move from M bumps the same way.
    reg.set_agent_state(modified_art, giver, MESIState.EXCLUSIVE, tick=5)  # type: ignore[attr-defined]
    reg.set_agent_state(modified_art, giver, MESIState.MODIFIED, tick=6)  # type: ignore[attr-defined]
    before = reg.get_owner_generation(modified_art)  # type: ignore[attr-defined]
    reg.set_agent_state(  # type: ignore[attr-defined]
        modified_art, giver, MESIState.INVALID, trigger=_HANDOFF_TRIGGER, tick=7
    )
    assert reg.get_artifact(modified_art).version == 1  # type: ignore[attr-defined]
    assert reg.get_owner_generation(modified_art) > before, (  # type: ignore[attr-defined]
        "a MODIFIED giver's handoff revoked its write claim without moving "
        "owner_generation"
    )

    # SHARED giver: a bystander read the path first, capturing the epoch; the
    # giver's own standing read is handed off.
    reg.set_agent_state(shared_art, bystander, MESIState.SHARED, trigger="fetch", tick=8)  # type: ignore[attr-defined]
    reg.set_agent_state(shared_art, giver, MESIState.SHARED, trigger="fetch", tick=9)  # type: ignore[attr-defined]
    before = reg.get_owner_generation(shared_art)  # type: ignore[attr-defined]
    reg.set_agent_state(  # type: ignore[attr-defined]
        shared_art, giver, MESIState.INVALID, trigger=_HANDOFF_TRIGGER, tick=10
    )
    assert reg.get_owner_generation(shared_art) == before, (  # type: ignore[attr-defined]
        "a SHARED giver's handoff moved owner_generation: no write claim was "
        "revoked, so the bump fences bystanders for nothing"
    )
    # Load-bearing: the bystander's commit at the unchanged version still wins.
    result = reg.commit_cas(  # type: ignore[attr-defined]
        shared_art, bystander, expected_version=1, content_hash=_hash("bystander"), tick=11
    )
    assert isinstance(result, tuple), (
        "a bystander's commit was REJECTED after a SHARED giver's handoff; the "
        f"handoff must not fence a reader whose claim nobody revoked, got {result!r}"
    )
    assert reg.get_artifact(shared_art).version == 2  # type: ignore[attr-defined]


def assert_fence_admits_absent_read_generation(factory: RegistryFactory) -> None:
    """Read-generation fence — ADMIT-ON-ABSENT leg (MUST-MATCH). Reproduced EXACTLY
    per the fence-parity lesson (it has drifted once before). A plain OCC writer
    that NEVER established a fence claim (SHARED, no captured ``read_generation``)
    is ADMITTED even after ``owner_generation`` has been bumped by an UNRELATED
    agent's reclaim — the ``is not None`` predicate is load-bearing, and version-
    CAS (not the fence) is this writer's lost-update protection.

    Contrast with :func:`assert_fence_rejects_superseded_read_generation`: a
    PRESENT-and-superseded read_generation is rejected; an ABSENT one is admitted.
    Both legs run so the asymmetry is pinned, not just one side."""
    reg = factory()
    artifact_id = uuid4()
    _register(reg, artifact_id, version=1)

    # An UNRELATED agent acquires + is reclaimed, bumping owner_generation to > 0.
    other = uuid4()
    reg.set_agent_state(artifact_id, other, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]
    reg.set_agent_state(  # type: ignore[attr-defined]
        artifact_id, other, MESIState.INVALID, trigger=_RECLAIM_TRIGGER, tick=2
    )
    assert reg.get_owner_generation(artifact_id) > 0  # type: ignore[attr-defined]

    # A FRESH plain OCC writer: SHARED, never captured a read_generation (absent).
    plain = uuid4()
    _seed_shared_writer(reg, artifact_id, plain, tick=3)
    assert reg.get_read_generation(artifact_id, plain) is None, (  # type: ignore[attr-defined]
        "the admit-on-absent scenario requires the writer to have NO captured "
        "read_generation — a SHARED grant must not capture one"
    )

    result = reg.commit_cas(  # type: ignore[attr-defined]
        artifact_id, plain, expected_version=1, content_hash=_hash("plain-occ"), tick=4
    )
    assert isinstance(result, tuple), (
        "admit-on-absent VIOLATED: a plain OCC writer with NO fence claim was "
        f"rejected — version-CAS, not the fence, is its protection. Got {result!r}"
    )
    updated, _invalidated = result
    assert updated.version == 2, "the admitted OCC writer must WIN and bump the version"


def _pair_of(reg: object, artifact_id: UUID) -> tuple[int, int]:
    """``(version, owner_generation)`` via the one pair-atomic accessor."""
    got = reg.get_artifact_and_generation(artifact_id)  # type: ignore[attr-defined]
    assert got is not None, "artifact vanished mid-scenario"
    artifact, generation = got
    return artifact.version, generation


def assert_version_and_generation_pair_is_untearable(factory: RegistryFactory) -> None:
    """Pair-atomicity of ``get_artifact_and_generation`` (MUST-MATCH). The
    member's contract obliges a SINGLE-INSTANT snapshot: the two values must
    have coexisted, so a concurrent sweep reclaim (which moves the generation
    WITHOUT moving the version) can never tear the pair. A backend that serves
    it as two independent reads passes every other scenario in this kit while
    silently reopening the reclaim-zombie EFFECT hole — the effect gate captures
    a torn pair, the boundary read returns the same torn pair, and the escaping
    effect fires through a reclaim.

    Teeth without ground truth: a mutator strictly alternates commit
    (version + 1) and sweep-class reclaim (generation + 1) from (1, 0), so the
    true pair sequence is (1,0) → (2,0) → (2,1) → (3,1) → … and therefore
    ``version - generation`` is ALWAYS 1 or 2. A read that pairs an old version
    with a newer generation (or the reverse) lands outside that band and is
    reported as a tear. The alternation is pinned SEQUENTIALLY first, so a
    backend whose transitions do not produce it fails as a wrong precondition
    rather than as a phantom tear."""
    reg = factory()
    artifact_id = uuid4()
    _register(reg, artifact_id, version=1)

    def advance(round_index: int) -> None:
        """One commit (version + 1) then one reclaim (generation + 1). The
        re-acquire is load-bearing: ``commit_cas`` leaves the committer in
        SHARED, and only an M/E holder's reclaim bumps the generation."""
        agent = uuid4()
        reg.set_agent_state(  # type: ignore[attr-defined]
            artifact_id, agent, MESIState.EXCLUSIVE, trigger="write",
            tick=round_index,
        )
        result = reg.commit_cas(  # type: ignore[attr-defined]
            artifact_id, agent, expected_version=round_index + 1,
            content_hash=_hash(f"c{round_index}"), tick=round_index,
        )
        assert isinstance(result, tuple), (
            "scenario invariant: the single mutator's commit must WIN "
            f"(round {round_index}); a loss would desynchronize the band check"
        )
        reg.set_agent_state(  # type: ignore[attr-defined]
            artifact_id, agent, MESIState.EXCLUSIVE, trigger="write",
            tick=round_index,
        )
        reg.set_agent_state(  # type: ignore[attr-defined]
            artifact_id, agent, MESIState.INVALID, trigger=_RECLAIM_TRIGGER,
            tick=round_index,
        )
        # Re-check the alternation EVERY round, via the single-value accessors
        # rather than the pair read under test -- those cannot tear, so a
        # backend whose transition semantics differ fails here as an explicit
        # PRECONDITION error instead of being misreported as a torn pair.
        seen_version = reg.get_artifact(artifact_id).version  # type: ignore[union-attr]
        seen_generation = reg.get_owner_generation(artifact_id)  # type: ignore[attr-defined]
        assert (seen_version, seen_generation) == (round_index + 2, round_index + 1), (
            "scenario precondition: each round must move BOTH legs exactly once "
            "(a commit bumps the version, a sweep-class reclaim bumps the "
            f"generation); round {round_index} left the artifact at "
            f"({seen_version}, {seen_generation})"
        )

    assert _pair_of(reg, artifact_id) == (1, 0)
    advance(0)

    # Feedback coupling, not a fixed round count: the mutator keeps advancing
    # until the observer has finished ALL its reads, so every observed read
    # overlaps live mutation — a two-read backend cannot dodge the tear by the
    # mutator finishing early, and an atomic backend ends the scenario as soon
    # as the observer is done.
    observer_reads = 300
    observer_done = threading.Event()
    torn: list[tuple[int, int]] = []
    errors: list[BaseException] = []

    def mutate() -> None:
        try:
            i = 1
            while not observer_done.is_set() and i < 100_000:
                advance(i)
                i += 1
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
            observer_done.set()

    def observe() -> None:
        try:
            for _ in range(observer_reads):
                version, generation = _pair_of(reg, artifact_id)
                if version - generation not in (1, 2):
                    torn.append((version, generation))
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            observer_done.set()

    writer = threading.Thread(target=mutate, name="pair-mutator")
    reader = threading.Thread(target=observe, name="pair-observer")
    writer.start()
    reader.start()
    writer.join()
    reader.join()
    if errors:
        raise errors[0]
    assert not torn, (
        "get_artifact_and_generation returned a TORN pair — the two values never "
        "coexisted. The mutator alternates commit/reclaim from (1, 0), so "
        f"version - generation must stay in {{1, 2}}; observed {torn[:5]}. A "
        "backend must serve this member as ONE atomic snapshot (a single row "
        "read under the write lock, or an equivalent seqlock), never two reads."
    )


def assert_session_fail_closed_on_foreign_and_reaped(factory: RegistryFactory) -> None:
    """Session fail-closed refusals (MUST-MATCH). Two typed refusals a conforming
    backend must reproduce identically, NEVER serving live HEAD:

    1. A FOREIGN caller (not the session owner) reading a live session RAISES
       :class:`SessionInvalidated` (owner-isolation, R13) — matched by its typed
       ``reason``, not a message substring.
    2. A REAPED / released token classifies as ``session_invalidated`` on read —
       a typed :class:`SessionReadRejection`, never a live-HEAD fall-through.
    """
    reg = factory()
    svc = CoordinatorService(reg)
    artifact_id = uuid4()
    art = Artifact(id=artifact_id, name="plan.md", version=1, content_hash="h")
    reg.register_artifact(art, content="PINNED-V1")  # type: ignore[attr-defined]

    owner = uuid4()
    session = svc.begin_session(read_set=[artifact_id], owner=owner)
    assert isinstance(session, SnapshotSession)
    token = session.session_token

    # (1) A foreign caller is rejected fail-closed (owner-isolation).
    try:
        svc.session_read(token, artifact_id, caller=uuid4())
    except SessionInvalidated as exc:
        assert exc.reason == SESSION_INVALIDATED_REASON, exc.reason
    else:
        raise AssertionError(
            "a FOREIGN caller was NOT rejected — session owner-isolation (R13) "
            "must fail closed, never serve another owner's cut"
        )

    # (2) A reaped/released token classifies session_invalidated on read.
    reg.release_session(token)  # type: ignore[attr-defined]
    result = svc.session_read(token, artifact_id, caller=owner)
    assert isinstance(result, SessionReadRejection), (
        "a released session token served (or crashed) instead of failing closed"
    )
    assert result.reason == SESSION_INVALIDATED_REASON, result.reason


# ===========================================================================
# BACKEND-DEFINED scenarios — assert the arm behaves as ITS OWN declaration.
# Signature: (factory, declaration) -> None. The declaration param is what makes
# the split STRUCTURAL: a backend-defined function cannot assert cross-arm
# identity — it verifies each arm against the disposition it declared.
# ===========================================================================


class RestartDeclaration(Enum):
    """A backend's DECLARED restart-survival disposition (R11 / R6). The kit does
    NOT assert the two arms agree — it asserts EACH arm matches the disposition IT
    declared. Passing this per-arm is the mechanism that keeps restart-survival a
    backend-defined property and NOT a must-match one."""

    SURVIVES = "survives"
    """The backend re-homes ``session_meta`` / pins durably: a re-open of the SAME
    store on a fresh handle still finds the session cut and serves the owner's
    pinned bytes. The Tier-1 durability disposition (sqlite)."""

    LOST = "lost"
    """The backend is process-scoped: a fresh instance is a fresh, empty store —
    the session cut is GONE and a read fails closed. This is the HONEST in-memory
    declaration; it is NOT a Tier-1 / HA-statelessness claim, and the kit records
    it as such (asserted-as-declared, never flagged as a bug)."""


def assert_declared_restart_survival(
    factory: RegistryFactory, declaration: RestartDeclaration
) -> None:
    """Restart survival (BACKEND-DEFINED). Begin a session, then simulate a
    "restart" and assert the arm behaves as ``declaration`` says:

    - ``SURVIVES`` (durable arm): re-open the SAME db file on a FRESH handle; the
      session cut is still readable AND the owner can still serve the pinned bytes
      (identity survived). Requires ``factory.db_path`` (a shared store to re-open).
    - ``LOST`` (process-scoped arm): a FRESH instance has NO pins — the cut is
      ``None`` and a read FAILS CLOSED (``session_invalidated``), never live HEAD.
      The declared loss is asserted AS DECLARED — not treated as a regression.

    This function NEVER compares the two arms to each other. Each arm is checked
    only against the disposition it declared — that is the structural split."""
    if declaration is RestartDeclaration.SURVIVES:
        _assert_restart_survives(factory)
    else:
        _assert_restart_lost(factory)


def _assert_restart_survives(factory: RegistryFactory) -> None:
    db_path = factory.db_path
    assert db_path is not None, (
        "a SURVIVES declaration requires a durable store to re-open; this arm "
        "exposes no db_path (it is process-scoped and cannot declare SURVIVES)"
    )
    reg1 = factory()
    artifact_id = UUID(int=0xF00D5E55)
    owner = UUID(int=0x0114E5)
    art = Artifact(id=artifact_id, name="plan.md", version=1, content_hash="h")
    reg1.register_artifact(art, content="SURVIVOR-V1")  # type: ignore[attr-defined]
    session = CoordinatorService(reg1).begin_session(read_set=[artifact_id], owner=owner)
    assert isinstance(session, SnapshotSession)
    token = session.session_token

    # The "restart": a SECOND handle on the SAME store (the durable pin persisted).
    reg2 = factory()
    svc2 = CoordinatorService(reg2)
    assert svc2.registry.get_session_cut(token) == {artifact_id: 1}, (
        "declared SURVIVES but the session cut did NOT survive a re-open — the "
        "durable session_meta / pins did not re-home"
    )
    served = svc2.session_read(token, artifact_id, caller=owner)
    assert isinstance(served, VersionedContent), (
        "declared SURVIVES but the owner could not serve the pinned bytes after "
        "restart — the durable pin was not intact"
    )
    assert served.content == "SURVIVOR-V1"


def _assert_restart_lost(factory: RegistryFactory) -> None:
    assert factory.db_path is None, (
        "a LOST declaration is for process-scoped arms; this arm exposes a "
        "db_path, so a 'fresh instance' would re-open the SAME store and survive"
    )
    reg1 = factory()
    artifact_id = UUID(int=0xF00D5E55)
    owner = UUID(int=0x0114E5)
    art = Artifact(id=artifact_id, name="plan.md", version=1, content_hash="h")
    reg1.register_artifact(art, content="EPHEMERAL-V1")  # type: ignore[attr-defined]
    session = CoordinatorService(reg1).begin_session(read_set=[artifact_id], owner=owner)
    assert isinstance(session, SnapshotSession)
    token = session.session_token

    # The "restart": a BRAND-NEW instance — no shared store, no pins. This is the
    # DECLARED loss, asserted as declared (never as a bug).
    reg2 = factory()
    svc2 = CoordinatorService(reg2)
    assert svc2.registry.get_session_cut(token) is None, (
        "declared LOST but the session cut survived a fresh instance — the "
        "process-scoped contract was violated (this would be the surprising case)"
    )
    result = svc2.session_read(token, artifact_id, caller=owner)
    assert isinstance(result, SessionReadRejection), (
        "declared LOST but session_read did not fail closed on a fresh instance — "
        "it must reject, NEVER serve live HEAD"
    )
    assert result.reason == SESSION_INVALIDATED_REASON, result.reason


# ===========================================================================
# R18 — liveness-source declaration check (a DECLARED-source assertion).
# ===========================================================================


def assert_declared_liveness_source_matches_contract(
    factory: RegistryFactory,
    *,
    contract_record: LivenessSourceObligation = R18_LIVENESS_SOURCE,
) -> None:
    """R18 liveness-source check. Assert the backend's OBSERVED liveness behavior
    matches the source DECLARED in the Unit-4 contract record. For BOTH shipped
    registries the declared source is CALLER-SUPPLIED LOGICAL TICKS under a SINGLE
    coordinator (the single-host declaration, NOT a cross-host / HA claim).

    Observed behavior: ``record_heartbeat`` accepts a caller-supplied tick and
    ``last_heartbeat_tick`` reflects it verbatim — i.e. the registry stores the
    tick VALUE the caller supplies and does NOT synthesize an authoritative clock
    of its own (which is precisely why the contract record says this does NOT
    conform for the HA claim until a re-home supplies an authoritative source).
    The assertion pins that the OBSERVED store-the-caller's-tick behavior matches
    the DECLARED caller-supplied-tick source."""
    # The contract's shipped-source declaration must name the caller-supplied,
    # single-coordinator source (matched as declared text, the Unit-4 record).
    declared = contract_record.shipped_source
    assert "CALLER-SUPPLIED" in declared and "SINGLE coordinator" in declared, (
        "the R18 contract record's shipped_source no longer declares the "
        "caller-supplied single-coordinator source the kit checks against; the "
        "contract and the kit have drifted"
    )

    reg = factory()
    artifact_id = uuid4()
    _register(reg, artifact_id, version=1)
    agent = uuid4()
    reg.set_agent_state(artifact_id, agent, MESIState.EXCLUSIVE, tick=1)  # type: ignore[attr-defined]

    # Observed: the registry stores the CALLER's tick verbatim (no authoritative
    # clock of its own) — the single-host declaration made concrete.
    reg.record_heartbeat(agent, 100)  # type: ignore[attr-defined]
    assert reg.last_heartbeat_tick(agent) == 100, (  # type: ignore[attr-defined]
        "observed liveness source diverged from the declared caller-supplied tick: "
        "the registry did not store the caller's tick verbatim"
    )
    # A later caller tick is honored verbatim (monotonic max, still caller-driven).
    reg.record_heartbeat(agent, 250)  # type: ignore[attr-defined]
    assert reg.last_heartbeat_tick(agent) == 250  # type: ignore[attr-defined]
