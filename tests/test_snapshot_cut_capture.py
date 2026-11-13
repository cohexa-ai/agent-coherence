# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Unit 2 — atomic consistent-cut capture + pin-set store (SB-17 / TX-1).

Plan: ``docs/plans/2026-06-26-002-feat-read-side-transaction-snapshot-plan.md``
Unit 2. Requirement trace: **R1** (atomic cut), **R4** (pins held against GC),
**R6** (restart-survival sqlite-only — asserted in the parity suite), **R11**
(the cut is an INSPECTABLE ``{artifact_id: version}`` map, not an opaque handle).

A protocol proof bounds the REGISTRY, not the integration: ``Snapshot.tla`` proves
``NoReadSkewWithinCut`` / ``PinAlwaysRetained`` over the model; this file is the
integration-layer regression for the multi-read capture WINDOW. Per the
split-comparand learning the staleness probes use **fixed-stale buffers, not
counters** — a captured cut compared against a buffer literal, never a re-derived
expectation that could drift with the bug it is meant to catch.

Reason matching is ALWAYS ``reason == CONSTANT`` against the imported wire-stable
constant — never a substring of a human message (the typed-signal-not-substring
house rule).
"""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.retention import RetentionPolicy
from ccs.coordinator.service import CoordinatorService, SessionCapsConfig
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import UNKNOWN_ARTIFACT_REASON
from ccs.core.states import MESIState
from ccs.core.types import (
    Artifact,
    SnapshotSession,
    VersionedReadRejection,
)

# ---------------------------------------------------------------------------
# Builders — mirror tests/test_retention_parity.py's helpers.
# ---------------------------------------------------------------------------


def _register(reg, artifact_id: UUID, name: str, body: str) -> None:
    art = Artifact(id=artifact_id, name=name, version=1, content_hash="h")
    reg.register_artifact(art, content=body)


def _commit(reg, artifact_id: UUID, version: int, body: str | bytes) -> None:
    """Advance an artifact to ``version`` carrying ``body`` (a peer commit)."""
    nxt = Artifact(id=artifact_id, name="x", version=version, content_hash="h")
    reg.set_artifact_and_content(artifact_id, nxt, body)


def _commit_cas(reg, artifact_id: UUID, writer: UUID, expected: int, body: str) -> None:
    """A peer OCC commit via the registry ``commit_cas`` WIN (captures history)."""
    reg.set_agent_state(artifact_id, writer, MESIState.SHARED, tick=1)
    reg.commit_cas(
        artifact_id,
        writer,
        expected_version=expected,
        content_hash="h",
        content=body,
        tick=2,
    )


def _make_registries(tmp_path: Path, policy: RetentionPolicy | None = None):
    """An (in_memory, sqlite) pair WITHOUT retention (the version-map-only branch,
    the ``content=None`` ICP) unless a policy is given."""
    retain = policy is not None
    mem = ArtifactRegistry(retain_versions=retain, retention_policy=policy)
    sql = SqliteArtifactRegistry(
        tmp_path / "snap.db", retain_versions=retain, retention_policy=policy
    )
    return mem, sql


# ===========================================================================
# A — Integration: the consistent cut across a peer commit (fixed-stale buffer)
# ===========================================================================


class TestConsistentCut:
    """capture {A, B} → a peer commits B between logical reads → both are served
    at the CAPTURED versions, not the moved one (no cross-artifact read skew)."""

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_peer_commit_between_reads_does_not_taint_cut(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _make_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a, b = uuid4(), uuid4()
            _register(reg, a, "plan.md", "a1")
            _register(reg, b, "budget.md", "b1")
            owner = uuid4()

            # Capture the cut {A:1, B:1} at one linearization point.
            session = svc.begin_session(read_set=[a, b], owner=owner)
            assert isinstance(session, SnapshotSession)
            # FIXED-STALE BUFFER: the cut is pinned to this literal, not a
            # re-read of the registry (which the peer commit below would move).
            captured_cut = {a: 1, b: 1}
            assert dict(session.cut) == captured_cut

            # A peer advances B to v2 AFTER the cut was captured.
            _commit(reg, b, 2, "b2")

            # The session's cut is immutable — still the captured versions, never
            # the moved B@2. (Read skew would show B at 2 here.)
            assert dict(session.cut) == captured_cut
            assert session.cut[b] == 1, "cut tainted by a post-capture peer commit"
            # The registry's live version DID move (the cut is a snapshot, not a
            # lock): the peer commit landed.
            assert reg.get_artifact(b).version == 2
        finally:
            sql.close()

    def test_cut_is_inspectable_map_not_opaque_handle_R11(
        self, tmp_path: Path
    ) -> None:
        # R11 binding constraint: begin_session returns the cut as an INSPECTABLE
        # {artifact_id: version} map so the paper SB-18 commit_all(session_token,
        # writes) can read each expected_version out of it. An opaque handle would
        # foreclose SB-18. This pins the surface shape Unit 9's harness re-checks.
        mem, sql = _make_registries(tmp_path)
        try:
            svc = CoordinatorService(mem)
            a, b = uuid4(), uuid4()
            _register(mem, a, "a", "a1")
            _commit(mem, a, 2, "a2")  # A is at v2
            _register(mem, b, "b", "b1")  # B at v1
            session = svc.begin_session(read_set=[a, b], owner=uuid4())
            assert isinstance(session, SnapshotSession)
            # Inspectable: every per-artifact expected_version is readable.
            assert session.cut[a] == 2
            assert session.cut[b] == 1
            # The map is keyed by the artifact UUIDs (not an opaque token blob).
            assert set(session.cut.keys()) == {a, b}
            # The session_token is a SEPARATE field (the identity), NOT the cut.
            assert isinstance(session.session_token, str)
            assert session.session_token  # non-empty, server-minted
        finally:
            sql.close()

    def test_retain_versions_branch_indicator(self, tmp_path: Path) -> None:
        # retain_versions mirrors the store's retention posture (the deployment
        # branch indicator Unit 3 selects the serve strategy from).
        # content=None / retention-off store → False (eager branch later).
        mem_off, sql_off = _make_registries(tmp_path)
        try:
            s_off = CoordinatorService(mem_off).begin_session(read_set=[], owner=uuid4())
            assert isinstance(s_off, SnapshotSession)
            assert s_off.retain_versions is False
        finally:
            sql_off.close()
        # retain_versions=True store → True (lazy branch later).
        mem_on = ArtifactRegistry(
            retain_versions=True, retention_policy=RetentionPolicy(max_versions=8)
        )
        s_on = CoordinatorService(mem_on).begin_session(read_set=[], owner=uuid4())
        assert isinstance(s_on, SnapshotSession)
        assert s_on.retain_versions is True

    def test_coordinator_epoch_stamped(self, tmp_path: Path) -> None:
        mem, sql = _make_registries(tmp_path)
        try:
            for reg in (mem, sql):
                svc = CoordinatorService(reg)
                s = svc.begin_session(read_set=[], owner=uuid4())
                assert isinstance(s, SnapshotSession)
                assert s.coordinator_epoch == reg.coordinator_epoch
        finally:
            sql.close()


# ===========================================================================
# B — Edge: a peer commit interleaving the capture is serialized whole
# ===========================================================================


class TestSerializedInterleave:
    """A peer commit landing around the capture is serialized ENTIRELY before or
    after — never partially visible within the cut (the atomicity property)."""

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_interleave_is_all_or_nothing_across_two_artifacts(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _make_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a, b = uuid4(), uuid4()
            _register(reg, a, "a", "a1")
            _register(reg, b, "b", "b1")

            # Pre-capture: bump A to v2 so a stale read of A would be visible.
            _commit(reg, a, 2, "a2")

            session = svc.begin_session(read_set=[a, b], owner=uuid4())
            assert isinstance(session, SnapshotSession)
            # A coherent cut: A@2 (its current at capture) AND B@1 — both from the
            # SAME linearization point. A torn capture would mix a pre- and a
            # post-bump version across the two artifacts.
            assert dict(session.cut) == {a: 2, b: 1}
        finally:
            sql.close()


# ===========================================================================
# C — Edge (GC-hold): a pinned version is exempt from retention GC while live
# ===========================================================================


class TestGcHold:
    """A pinned version survives the inline retention GC while the session is
    live (the exemptions seam), and becomes collectible again after release."""

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_pinned_version_exempt_until_release(
        self, tmp_path: Path, arm: str
    ) -> None:
        # K=2 keeps only {current, current-1}. Pin v1 WHILE it is current, then
        # drive enough commits that v1 would normally be K-evicted — the pin must
        # hold it; after release the next commit collects it.
        policy = RetentionPolicy(max_versions=2)
        mem, sql = _make_registries(tmp_path, policy)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a = uuid4()
            _register(reg, a, "plan.md", "v1")  # v1 captured (retain on), current=1
            writer = uuid4()

            # Pin v1 via a session whose cut is {A:1} — captured while v1 is
            # still current.
            session = svc.begin_session(read_set=[a], owner=uuid4())
            assert isinstance(session, SnapshotSession)
            assert session.cut[a] == 1

            # Peers advance to v2 then v3 → without the pin, K=2 ({2,3}) would
            # evict v1. The pin exemption holds it.
            _commit_cas(reg, a, writer, expected=1, body="v2")  # K=2 → {1(pin),2}
            _commit_cas(reg, a, writer, expected=2, body="v3")  # K=2 {2,3} + pin 1
            assert reg.get_content_at_version(a, 1) == "v1", (
                "pinned v1 was GC-collected out from under a live session"
            )

            # Release → v1 is collectible again; the next capture GC drops it.
            reg.release_session(session.session_token)
            _commit_cas(reg, a, writer, expected=3, body="v4")  # K=2 evicts v1 now
            assert reg.get_content_at_version(a, 1) is None, (
                "released pin did not become collectible — exemption leaked"
            )
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_two_overlapping_sessions_hold_independent_pins(
        self, tmp_path: Path, arm: str
    ) -> None:
        # Two sessions pin different versions of the same artifact; releasing one
        # must NOT drop the other's pin (the union-of-live-pins exemption).
        policy = RetentionPolicy(max_versions=1)  # only current survives w/o pins
        mem, sql = _make_registries(tmp_path, policy)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a = uuid4()
            _register(reg, a, "plan.md", "v1")
            writer = uuid4()
            s1 = svc.begin_session(read_set=[a], owner=uuid4())  # pins v1
            _commit_cas(reg, a, writer, expected=1, body="v2")
            s2 = svc.begin_session(read_set=[a], owner=uuid4())  # pins v2
            _commit_cas(reg, a, writer, expected=2, body="v3")  # current=3

            assert isinstance(s1, SnapshotSession) and isinstance(s2, SnapshotSession)
            # Both pins held despite K=1.
            assert reg.get_content_at_version(a, 1) == "v1"
            assert reg.get_content_at_version(a, 2) == "v2"

            # Release s1 → v1 collectible, but s2's v2 still pinned.
            reg.release_session(s1.session_token)
            _commit_cas(reg, a, writer, expected=3, body="v4")
            assert reg.get_content_at_version(a, 1) is None, "s1's pin leaked"
            assert reg.get_content_at_version(a, 2) == "v2", (
                "s2's independent pin was dropped when s1 released"
            )
        finally:
            sql.close()


# ===========================================================================
# D — Edge (in-memory atomicity): concurrent capture + peer commit under the lock
# ===========================================================================


class TestInMemoryConcurrentAtomicity:
    """The registry lock makes the multi-artifact read + pin insert atomic
    across N artifacts AND serializes with the version-moving writes
    (register/commit), so a captured cut is always a
    SINGLE-linearization-point snapshot — never a read-skewed pair stitched
    across two writer commits. (Once the capture-only lock; now the one
    registry-wide lock every public method holds.)

    The probe enforces a real cross-artifact invariant: a writer always advances
    A *before* B to the same version, so at EVERY real instant
    ``version(A) >= version(B)``. A cut with ``B > A`` never existed at any
    instant — it would be the read skew the lock prevents (the ``{A:5, B:6}``
    shape, where B only reached 6 once A did)."""

    def test_concurrent_capture_never_observes_impossible_cut(self) -> None:
        reg = ArtifactRegistry()  # no retention; version-map capture only
        # This atomicity probe opens 4000 sessions in a tight loop and frees only
        # the registry pins (release_session) per iteration, so the service-level
        # owner-binding COUNT accumulates. Raise the Unit-7 max_sessions cap above
        # the loop count so the R14 bound (default 256) does not interfere with the
        # pre-existing read-skew probe (the caps surface is tested in
        # tests/test_session_identity_and_caps.py).
        svc = CoordinatorService(reg, session_caps=SessionCapsConfig(max_sessions=8192))
        a, b = uuid4(), uuid4()
        _register(reg, a, "a", "a1")
        _register(reg, b, "b", "b1")

        stop = threading.Event()
        skewed: list[dict] = []

        def writer() -> None:
            v = 1
            while not stop.is_set():
                v += 1
                # A first, then B → at every instant version(A) >= version(B).
                _commit(reg, a, v, f"a{v}")
                _commit(reg, b, v, f"b{v}")

        def capturer() -> None:
            for _ in range(4000):
                s = svc.begin_session(read_set=[a, b], owner=uuid4())
                assert isinstance(s, SnapshotSession)
                if s.cut[b] > s.cut[a]:
                    skewed.append(dict(s.cut))
                reg.release_session(s.session_token)

        tw = threading.Thread(target=writer)
        tw.start()
        capturer()
        stop.set()
        tw.join(timeout=10.0)
        assert not tw.is_alive(), "writer hung"

        # HARD property: no cut ever observed B ahead of A — every cut is a
        # genuine single-linearization-point snapshot (no read skew). Without the
        # version-moving writes taking the capture lock, the capture could read
        # A@v then a writer advances A,B to v+1, then read B@v+1 → impossible
        # {A:v, B:v+1}.
        assert not skewed, (
            f"capture observed read-skewed (impossible) cuts: {skewed[:5]} "
            f"(total {len(skewed)}) — the version-move/capture serialization broke"
        )


# ===========================================================================
# E — Security: an unknown id rejects the WHOLE capture, inserts NO pins
# ===========================================================================


class TestUnknownIdRejection:
    """A read_set with an unknown id → typed UNKNOWN_ARTIFACT rejection, NO
    partial cut registered (no pins inserted, no existence-probe oracle)."""

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_unknown_id_rejects_and_inserts_no_pins(
        self, tmp_path: Path, arm: str
    ) -> None:
        policy = RetentionPolicy(max_versions=1)  # so an erroneously-held pin shows
        mem, sql = _make_registries(tmp_path, policy)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            known = uuid4()
            unknown = uuid4()
            _register(reg, known, "plan.md", "v1")
            writer = uuid4()

            result = svc.begin_session(read_set=[known, unknown], owner=uuid4())
            # Typed rejection on the wire-stable constant — not a SnapshotSession.
            assert isinstance(result, VersionedReadRejection)
            assert result.reason == UNKNOWN_ARTIFACT_REASON
            assert result.artifact_id == unknown
            # No partial cut: the KNOWN id's version was NOT pinned. Prove it by
            # K-evicting v1 — if a pin had leaked, the exemption would hold it.
            _commit_cas(reg, known, writer, expected=1, body="v2")  # K=1 → drop v1
            assert reg.get_content_at_version(known, 1) is None, (
                "a partial cut was registered: the known id's v1 was pinned "
                "despite the unknown-id rejection"
            )
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_unknown_id_drops_owner_binding(self, tmp_path: Path, arm: str) -> None:
        # The rejected token must not linger as a half-open owned session.
        mem, sql = _make_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            result = svc.begin_session(read_set=[uuid4()], owner=uuid4())
            assert isinstance(result, VersionedReadRejection)
            # No owner binding survives a rejected capture (Unit 2 cleanup).
            assert svc._session_owners == {}
        finally:
            sql.close()


class TestWatchdogAbortOpensNoSession:
    """#340: ``begin_session`` threads the handler watchdog's ``abort`` into
    ``registry.abort_guard`` (A6), so a begin whose request already timed out
    fails closed at the registry lock: no token minted, no owner bound, no cut
    pinned, no session row for the cap to count. An unset abort opens the
    session as before."""

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_preset_abort_opens_no_session(self, tmp_path: Path, arm: str) -> None:
        from ccs.core.exceptions import WatchdogAbandoned

        policy = RetentionPolicy(max_versions=1)  # so a leaked pin would hold v1
        mem, sql = _make_registries(tmp_path, policy)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a, writer = uuid4(), uuid4()
            _register(reg, a, "plan.md", "v1")
            abort = threading.Event()
            abort.set()  # the handler watchdog already answered degraded

            with pytest.raises(WatchdogAbandoned):
                svc.begin_session(read_set=[a], owner=uuid4(), abort=abort)

            assert svc._session_owners == {}
            assert svc._session_heartbeats == {}
            assert svc._session_created == {}
            assert reg.session_count() == 0
            assert reg.all_session_meta() == {}
            _commit_cas(reg, a, writer, expected=1, body="v2")  # K=1 -> drop v1
            assert reg.get_content_at_version(a, 1) is None, "the aborted begin pinned v1"
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_unset_abort_opens_the_session(self, tmp_path: Path, arm: str) -> None:
        mem, sql = _make_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a, owner = uuid4(), uuid4()
            _register(reg, a, "plan.md", "v1")
            session = svc.begin_session(read_set=[a], owner=owner, abort=threading.Event())
            assert isinstance(session, SnapshotSession)
            assert session.cut == {a: 1}
            assert svc._session_owners[session.session_token] == owner
            assert reg.session_count() == 1
        finally:
            sql.close()


# ===========================================================================
# F — Owner-binding + non-mutating capture
# ===========================================================================


class TestOwnerBindingAndNonMutating:
    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_token_is_owner_bound_at_mint(self, tmp_path: Path, arm: str) -> None:
        mem, sql = _make_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a = uuid4()
            _register(reg, a, "a", "a1")
            owner = uuid4()
            s = svc.begin_session(read_set=[a], owner=owner)
            assert isinstance(s, SnapshotSession)
            assert svc._session_owners[s.session_token] == owner
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_capture_mints_no_mesi_grant_no_read_generation(
        self, tmp_path: Path, arm: str
    ) -> None:
        # The capture is non-mutating on the coherence plane: no MESI state for
        # the owner, and no read_generation captured (it never calls
        # set_agent_state / the fence path).
        mem, sql = _make_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a = uuid4()
            _register(reg, a, "a", "a1")
            owner = uuid4()
            svc.begin_session(read_set=[a], owner=owner)
            # The owner holds no MESI grant on the pinned artifact.
            assert reg.get_agent_state(a, owner) is None
            # No fence claim was captured for the owner.
            assert reg.get_read_generation(a, owner) is None
            # owner_generation untouched (no reclamation).
            assert reg.get_owner_generation(a) == 0
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_tokens_are_unique_per_session(self, tmp_path: Path, arm: str) -> None:
        mem, sql = _make_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a = uuid4()
            _register(reg, a, "a", "a1")
            tokens = set()
            for _ in range(8):
                s = svc.begin_session(read_set=[a], owner=uuid4())
                assert isinstance(s, SnapshotSession)
                tokens.add(s.session_token)
            assert len(tokens) == 8, "server-minted tokens collided"
        finally:
            sql.close()
