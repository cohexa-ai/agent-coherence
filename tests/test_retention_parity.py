# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Cross-registry retention parity — the genuinely net-new Unit 5 scenarios.

Plan item N v1, Unit 5 (``docs/plans/2026-06-10-001-feat-version-retention-
read-at-version-plan.md``). Requirement trace: **R3** (parity), suite-wide
success criteria.

This file is DELIBERATELY SMALL. Units 2-4 already established extensive
both-registry coverage:

- ``tests/test_retention.py`` runs a parametrized ``retention_registry`` fixture
  (in_memory + sqlite) over per-capture-point capture, K-eviction, T-expiry,
  ``content=None`` skip, no-capture-on-reject, and remove-drops-history.
- ``tests/test_read_at_version.py`` runs a parametrized ``make_registry`` factory
  (in_memory + sqlite) over the full per-reason matrix, fence non-capture (R6),
  MESI non-interaction (R7), and the rejection-payload pin.

Unit 5 adds ONLY the two cross-registry scenarios those per-unit suites do not
cover, because each runs one closure against one arm at a time rather than
asserting the two arms agree as a whole:

1. **Whole-sequence equivalence** (``TestWholeSequenceEquivalence``) — one
   identical operation sequence (register → K-evicting commits → delete-cascade
   → ``commit_cas(content=None)`` skip → T-expiry via a monkeypatched clock) run
   against BOTH registries built as a pair (the ``tests/test_registry.py:195``
   ``_build_pair`` / ``_normalize`` pattern), asserting the observable retention
   state — retained ``(version → content)`` maps (str AND bytes) and
   ``read_at_version`` ``(version → outcome)`` maps — is IDENTICAL across
   registries. This is the test that catches sqlite and in-memory DRIFTING
   apart; no per-unit test compares the two arms' full state.

2. **Peer-interleaving parity** (``TestPeerInterleavingParity``) — a second
   writer commits a new version BETWEEN this reader's operations; the reader's
   next ``read_at_version`` must reflect the AUTHORITATIVE persisted history, not
   a stale per-instance cached belief. This guards the exact bug class in
   ``docs/solutions/logic-errors/coherent-volume-write-noop-skip-stale-cache-
   disk-divergence-2026-06-08.md`` (an optimization that consulted a per-instance
   cache instead of the authoritative store → silent divergence). The sqlite arm
   is the REAL two-handle case (two ``SqliteArtifactRegistry`` handles on one db
   file); the in-memory arm is single-object (two handles is N/A) and instead
   proves the read path re-derives from the one authoritative store after a peer
   mutation. Per the ``tests/test_occ_commit_cas.py`` discipline the only HARD
   assertion is the correctness property; concurrency-realism is a
   ``RuntimeWarning`` and a ``threading.Barrier`` forces the window.

Reason matching is ALWAYS ``reason == CONSTANT`` against the imported wire-stable
constants — never a substring of a human message (the
typed-signal-not-substring house rule).
"""

from __future__ import annotations

import threading
import time
import warnings
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.retention import RetentionPolicy
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    CURRENT_VERSION_REASON,
    NOT_RETAINED_REASON,
    SESSION_INVALIDATED_REASON,
    UNKNOWN_ARTIFACT_REASON,
    SessionInvalidated,
)
from ccs.core.states import MESIState
from ccs.core.types import (
    Artifact,
    SessionReadRejection,
    SnapshotSession,
    VersionedContent,
    VersionedReadRejection,
)

# ---------------------------------------------------------------------------
# Pair builders — the tests/test_registry.py:195 `_build_pair` pattern, run the
# SAME closure against both registries and compare normalized outcomes.
# ---------------------------------------------------------------------------


def _build_pair(
    tmp_path: Path, policy: RetentionPolicy | None
) -> tuple[ArtifactRegistry, SqliteArtifactRegistry]:
    """An in-memory + a sqlite registry, both with retention on under the SAME
    policy, so one closure's observable retention state can be compared across
    them. The sqlite handle is closed by the caller (try/finally)."""
    return (
        ArtifactRegistry(retain_versions=True, retention_policy=policy),
        SqliteArtifactRegistry(
            tmp_path / "parity.db", retain_versions=True, retention_policy=policy
        ),
    )


def _commit(reg, artifact_id: UUID, version: int, body: str | bytes) -> None:
    nxt = Artifact(id=artifact_id, name="plan.md", version=version, content_hash="h")
    reg.set_artifact_and_content(artifact_id, nxt, body)


def _content_map(reg, artifact_id: UUID, versions: range) -> dict[int, object]:
    """The raw retained ``(version → content|None)`` map — the registry-agnostic
    normal form for "same set of retained versions, same content per version"."""
    return {v: reg.get_content_at_version(artifact_id, v) for v in versions}


def _outcome_map(svc: CoordinatorService, artifact_id: UUID, versions: range) -> dict:
    """The ``read_at_version`` ``(version → outcome)`` map normalized to a
    registry-agnostic tuple: a hit → ``("content", body)`` (carries the body so
    str/bytes type is compared too); a rejection → ``("reject", reason)``."""
    out: dict[int, tuple] = {}
    for v in versions:
        res = svc.read_at_version(artifact_id, v)
        if isinstance(res, VersionedContent):
            out[v] = ("content", res.content)
        else:
            assert isinstance(res, VersionedReadRejection)
            out[v] = ("reject", res.reason)
    return out


# ===========================================================================
# A — Whole-sequence cross-registry equivalence
# ===========================================================================


class TestWholeSequenceEquivalence:
    """Run ONE identical operation sequence against both registries and assert
    the observable retention state is byte-for-byte identical. This is the only
    test that compares the two arms' FULL state — the per-unit suites each run
    one arm at a time, so a silent sqlite-vs-in-memory drift would pass them all
    yet fail here."""

    # A FIXED artifact id so the (version → ...) maps are directly comparable
    # across the two registries (no per-arm uuid divergence in the keys).
    _FIXED_ID = UUID(int=0xA17FAC7)

    def _run_sequence(self, reg) -> None:
        """register v1 → K-evicting commits (v2,v3,v4) → commit_cas(content=None)
        WIN (v5, no capture) — the shared mutation history both arms replay.
        K=3 so v1 is evicted; v4 carries a bytes body to exercise the
        str-vs-bytes round trip inside the equivalence comparison."""
        art = Artifact(
            id=self._FIXED_ID, name="plan.md", version=1, content_hash="h"
        )
        reg.register_artifact(art, content="c1")
        for v, body in ((2, "c2"), (3, "c3"), (4, b"\x00\x04bytes")):
            _commit(reg, self._FIXED_ID, v, body)
        # commit_cas(content=None) WIN: current advances to v5 but NO row is
        # captured for v5 (the history-poisoning fix) — a versioned read of v5 is
        # therefore current_version, and there is no stale v5 body to diverge on.
        writer = uuid4()
        reg.set_agent_state(self._FIXED_ID, writer, MESIState.SHARED, tick=1)
        reg.commit_cas(
            self._FIXED_ID,
            writer,
            expected_version=4,
            content_hash="h5",
            content=None,
            tick=2,
        )

    def test_retained_content_maps_identical(self, tmp_path: Path) -> None:
        # Same retained-version set AND same content per version (str + bytes).
        mem, sql = _build_pair(tmp_path, RetentionPolicy(max_versions=3))
        try:
            self._run_sequence(mem)
            self._run_sequence(sql)
            probe = range(1, 7)
            mem_map = _content_map(mem, self._FIXED_ID, probe)
            sql_map = _content_map(sql, self._FIXED_ID, probe)
            assert mem_map == sql_map, (
                f"retained content drifted between registries:\n"
                f"  in-memory: {mem_map}\n  sqlite   : {sql_map}"
            )
            # Pin the expected shape too (so a both-arms-equally-wrong regression
            # is still caught): K=3 evicted v1; v2,v3 are str; v4 is the exact
            # bytes; v5 never captured (content=None WIN); v6 never existed.
            assert mem_map == {
                1: None,
                2: "c2",
                3: "c3",
                4: b"\x00\x04bytes",
                5: None,
                6: None,
            }
            # Type fidelity is part of "same content": str stays str, bytes stays
            # bytes, identically on both arms.
            assert isinstance(mem_map[2], str) and isinstance(sql_map[2], str)
            assert isinstance(mem_map[4], bytes) and isinstance(sql_map[4], bytes)
        finally:
            sql.close()

    def test_read_at_version_outcome_maps_identical(self, tmp_path: Path) -> None:
        # Same read_at_version outcome (VersionedContent body OR rejection reason)
        # for a probe set spanning every branch — across both registries.
        mem, sql = _build_pair(tmp_path, RetentionPolicy(max_versions=3))
        try:
            self._run_sequence(mem)
            self._run_sequence(sql)
            svc_mem, svc_sql = CoordinatorService(mem), CoordinatorService(sql)
            probe = range(1, 7)
            mem_out = _outcome_map(svc_mem, self._FIXED_ID, probe)
            sql_out = _outcome_map(svc_sql, self._FIXED_ID, probe)
            assert mem_out == sql_out, (
                f"read_at_version outcomes drifted between registries:\n"
                f"  in-memory: {mem_out}\n  sqlite   : {sql_out}"
            )
            # Expected shape: v1 K-evicted → not_retained; v2,v3 served; v4 served
            # (exact bytes); v5 is current (content=None advanced it) →
            # current_version; v6 > current → future_version.
            assert mem_out == {
                1: ("reject", NOT_RETAINED_REASON),
                2: ("content", "c2"),
                3: ("content", "c3"),
                4: ("content", b"\x00\x04bytes"),
                5: ("reject", CURRENT_VERSION_REASON),
                6: ("reject", "future_version"),
            }
        finally:
            sql.close()

    def test_delete_cascade_equivalence(self, tmp_path: Path) -> None:
        # remove_artifact drops history on BOTH registries identically: every
        # post-delete read is unknown_artifact (deleted ≡ never-existed) and the
        # raw getter returns None for every prior version on both arms.
        mem, sql = _build_pair(tmp_path, RetentionPolicy(max_versions=8))
        try:
            self._run_sequence(mem)
            self._run_sequence(sql)
            for reg in (mem, sql):
                reg.remove_artifact(self._FIXED_ID)
            probe = range(1, 6)
            assert _content_map(mem, self._FIXED_ID, probe) == _content_map(
                sql, self._FIXED_ID, probe
            ) == {v: None for v in probe}
            svc_mem, svc_sql = CoordinatorService(mem), CoordinatorService(sql)
            mem_out = _outcome_map(svc_mem, self._FIXED_ID, probe)
            sql_out = _outcome_map(svc_sql, self._FIXED_ID, probe)
            assert mem_out == sql_out
            assert all(
                v == ("reject", UNKNOWN_ARTIFACT_REASON) for v in mem_out.values()
            )
        finally:
            sql.close()

    def test_t_expiry_equivalence_under_monkeypatched_clock(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # T-axis physical-drop parity: capture v1@t=0, v2@t=5, then v3@t=100 (past
        # the 10s horizon) — the t=100 capture physically drops v1 AND v2 on both
        # registries; only the current row (v3) survives, identically.
        clock = {"now": 0.0}
        monkeypatch.setattr(time, "time", lambda: clock["now"])
        policy = RetentionPolicy(max_versions=100, max_age_seconds=10.0)
        mem, sql = _build_pair(tmp_path, policy)
        try:
            for reg in (mem, sql):
                clock["now"] = 0.0
                art = Artifact(
                    id=self._FIXED_ID, name="plan.md", version=1, content_hash="h"
                )
                reg.register_artifact(art, content="c1")  # captured at t=0
                clock["now"] = 5.0
                _commit(reg, self._FIXED_ID, 2, "c2")  # t=5
                clock["now"] = 100.0  # past horizon → v1,v2 dropped at this capture
                _commit(reg, self._FIXED_ID, 3, "c3")
            probe = range(1, 4)
            mem_map = _content_map(mem, self._FIXED_ID, probe)
            sql_map = _content_map(sql, self._FIXED_ID, probe)
            assert mem_map == sql_map == {1: None, 2: None, 3: "c3"}
        finally:
            sql.close()


# ===========================================================================
# B — Peer-interleaving parity (the CoherentVolume stale-cache bug class)
# ===========================================================================


class TestPeerInterleavingParity:
    """A peer mutates the store BETWEEN this reader's operations; the reader's
    next read must reflect the AUTHORITATIVE persisted history, never a stale
    per-instance cached belief (the
    ``coherent-volume-write-noop-skip-stale-cache-disk-divergence`` bug class:
    an optimization consulting a cached belief instead of the authoritative
    store → silent divergence).

    Discipline per ``tests/test_occ_commit_cas.py``: the ONLY hard assertion is
    the correctness property (reads reflect the authoritative store); a
    ``threading.Barrier`` forces the interleaving window, and any "did the
    interleave actually overlap" check is a ``RuntimeWarning``, never an assert.
    """

    @pytest.mark.usefixtures("two_writer_store")
    def test_sqlite_two_handles_reads_reflect_peer_commit(
        self, tmp_path: Path
    ) -> None:
        # The REAL two-handle case: writer A and reader B are two SEPARATE
        # SqliteArtifactRegistry handles on the SAME db file (WAL allows it; the
        # second open finds user_version=2 and runs no migration). B holds NO
        # per-instance cache of A's writes — its read path issues a fresh SELECT
        # under its own lock, so it MUST observe A's committed bytes.
        policy = RetentionPolicy(max_versions=8)
        db = tmp_path / "shared.db"
        writer_a = SqliteArtifactRegistry(
            db, retain_versions=True, retention_policy=policy
        )
        reader_b = SqliteArtifactRegistry(
            db, retain_versions=True, retention_policy=policy
        )
        svc_b = CoordinatorService(reader_b)
        try:
            artifact_id = uuid4()
            art = Artifact(
                id=artifact_id, name="plan.md", version=1, content_hash="h"
            )
            writer_a.register_artifact(art, content="c1")
            _commit(writer_a, artifact_id, 2, "c2")  # A: current=2, v1 history

            # B reads v1 BEFORE A's interleaved commit — sees the authoritative
            # store as it stands now (v1 retained, current=2).
            assert reader_b.get_artifact(artifact_id).version == 2
            pre = svc_b.read_at_version(artifact_id, 1)
            assert isinstance(pre, VersionedContent)
            assert pre.content == "c1"
            # v2 is current at this instant → current_version (not history yet).
            pre_v2 = svc_b.read_at_version(artifact_id, 2)
            assert isinstance(pre_v2, VersionedReadRejection)
            assert pre_v2.reason == CURRENT_VERSION_REASON

            # PEER A commits v3 BETWEEN B's operations — v2 becomes history.
            _commit(writer_a, artifact_id, 3, "c3")

            # B's NEXT reads must reflect A's interleaved commit from the
            # AUTHORITATIVE store, not a cached belief: v2 is now servable history
            # with EXACT bytes, and v3 is the new current. A phantom retained
            # version or a missing one (the bug class) would fail here.
            post_v2 = svc_b.read_at_version(artifact_id, 2)
            assert isinstance(post_v2, VersionedContent), (
                "reader B did not observe peer A's interleaved commit: v2 should "
                "be servable history after A advanced current to v3 — a stale "
                "per-instance belief would still call v2 the current version"
            )
            assert post_v2.content == "c2"  # exact persisted bytes, never v3's
            post_v3 = svc_b.read_at_version(artifact_id, 3)
            assert isinstance(post_v3, VersionedReadRejection)
            assert post_v3.reason == CURRENT_VERSION_REASON
            assert post_v3.current_version == 3
            # v1, K-bounded but K=8 here, is still retained authoritative history.
            assert reader_b.get_content_at_version(artifact_id, 1) == "c1"
        finally:
            reader_b.close()
            writer_a.close()

    @pytest.mark.usefixtures("two_writer_store")
    def test_sqlite_two_handles_peer_gc_evicts_from_readers_view(
        self, tmp_path: Path
    ) -> None:
        # The eviction direction of the same property: under a bounded K, peer
        # A's interleaved commits push an OLD version past the K window. Reader
        # B — deriving from the authoritative store — must see that version as
        # not_retained (it was authoritatively GC'd), NOT serve a phantom copy
        # from any cached belief.
        policy = RetentionPolicy(max_versions=2)  # keep current + 1
        db = tmp_path / "shared_gc.db"
        writer_a = SqliteArtifactRegistry(
            db, retain_versions=True, retention_policy=policy
        )
        reader_b = SqliteArtifactRegistry(
            db, retain_versions=True, retention_policy=policy
        )
        svc_b = CoordinatorService(reader_b)
        try:
            artifact_id = uuid4()
            art = Artifact(
                id=artifact_id, name="plan.md", version=1, content_hash="h"
            )
            writer_a.register_artifact(art, content="c1")
            _commit(writer_a, artifact_id, 2, "c2")  # K=2 → {1,2}, current=2

            # B sees v1 retained right now.
            assert isinstance(svc_b.read_at_version(artifact_id, 1), VersionedContent)

            # PEER A commits v3 → K=2 evicts v1 authoritatively ({2,3}).
            _commit(writer_a, artifact_id, 3, "c3")

            # B's next read of v1 MUST be not_retained — the authoritative GC is
            # what B derives from; a cached belief would still serve v1.
            out = svc_b.read_at_version(artifact_id, 1)
            assert isinstance(out, VersionedReadRejection), (
                "reader B served a phantom v1 after peer A's commit GC'd it — the "
                "read consulted a stale belief instead of the authoritative store"
            )
            assert out.reason == NOT_RETAINED_REASON
            # And v2 (now within the K window as history) is still authoritative.
            assert reader_b.get_content_at_version(artifact_id, 2) == "c2"
        finally:
            reader_b.close()
            writer_a.close()

    @pytest.mark.usefixtures("two_writer_store")
    def test_sqlite_two_handles_concurrent_interleave_barrier(
        self, tmp_path: Path
    ) -> None:
        # The same property under a genuine threading.Barrier-forced window
        # (test_occ_commit_cas.py discipline). Writer A commits v3 while reader B
        # reads around the barrier; the HARD assertion is the correctness
        # property — B's read of v2 is EITHER current_version (B won the race,
        # v2 still current) OR exact "c2" history (A won, v2 demoted) — NEVER
        # wrong bytes and never another reason. "Did the threads overlap" is a
        # RuntimeWarning only (a constrained runner may serialize them).
        policy = RetentionPolicy(max_versions=8)
        db = tmp_path / "shared_barrier.db"
        writer_a = SqliteArtifactRegistry(
            db, retain_versions=True, retention_policy=policy
        )
        reader_b = SqliteArtifactRegistry(
            db, retain_versions=True, retention_policy=policy
        )
        svc_b = CoordinatorService(reader_b)
        try:
            artifact_id = uuid4()
            art = Artifact(
                id=artifact_id, name="plan.md", version=1, content_hash="h"
            )
            writer_a.register_artifact(art, content="c1")
            _commit(writer_a, artifact_id, 2, "c2")  # current=2

            barrier = threading.Barrier(2)
            results: dict[str, object] = {}

            def commit_v3() -> None:
                barrier.wait()
                _commit(writer_a, artifact_id, 3, "c3")
                results["a_done"] = time.perf_counter()

            def read_v2() -> None:
                barrier.wait()
                results["b_read"] = svc_b.read_at_version(artifact_id, 2)
                results["b_done"] = time.perf_counter()

            ta = threading.Thread(target=commit_v3)
            tb = threading.Thread(target=read_v2)
            ta.start()
            tb.start()
            ta.join(timeout=10.0)
            tb.join(timeout=10.0)
            assert not ta.is_alive() and not tb.is_alive(), "threads hung"

            # HARD correctness property — the read is one of exactly two allowed
            # authoritative outcomes; never wrong bytes, never a mislabeled reason.
            out = results["b_read"]
            if isinstance(out, VersionedContent):
                assert out.content == "c2", (
                    "v2 served as history with WRONG bytes — a racing peer commit "
                    "corrupted the authoritative read"
                )
            else:
                assert isinstance(out, VersionedReadRejection)
                assert out.reason == CURRENT_VERSION_REASON, (
                    f"v2 read mislabeled under a racing commit: {out.reason}"
                )

            # After both threads settle, the authoritative store is unambiguous:
            # current=3, v2 is servable history with exact bytes (B re-derives it,
            # no phantom / no miss).
            final = svc_b.read_at_version(artifact_id, 2)
            assert isinstance(final, VersionedContent)
            assert final.content == "c2"
            assert reader_b.get_artifact(artifact_id).version == 3

            # Realism check — warning only, never an assertion (constrained
            # runners may serialize the two threads).
            if "a_done" in results and "b_done" in results:
                if abs(results["a_done"] - results["b_done"]) > 0.5:
                    warnings.warn(
                        "peer commit and reader did not overlap within 500ms; the "
                        "interleave window may not have been exercised on this "
                        "runner (correctness still asserted)",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            reader_b.close()
            writer_a.close()

    def test_in_memory_single_store_reads_reflect_peer_mutation(self) -> None:
        # In-memory has ONE registry object, so "two handles on one store" is N/A
        # (a second handle would be a DIFFERENT, empty store). The analogous
        # property: writer A and reader B share the ONE authoritative object;
        # after A mutates it between B's operations, B's read path — which always
        # re-derives from that object, holding no separate cached belief — sees
        # the mutation. This is the in-memory expression of "reads reflect the
        # authoritative store after a peer mutation".
        reg = ArtifactRegistry(
            retain_versions=True, retention_policy=RetentionPolicy(max_versions=8)
        )
        svc = CoordinatorService(reg)
        artifact_id = uuid4()
        art = Artifact(id=artifact_id, name="plan.md", version=1, content_hash="h")
        reg.register_artifact(art, content="c1")
        _commit(reg, artifact_id, 2, "c2")  # current=2, v1 history

        # B reads v2 → current at this instant.
        pre = svc.read_at_version(artifact_id, 2)
        assert isinstance(pre, VersionedReadRejection)
        assert pre.reason == CURRENT_VERSION_REASON

        # PEER mutation interleaved: A commits v3 → v2 demoted to history.
        _commit(reg, artifact_id, 3, "c3")

        # B's next read re-derives current from the authoritative store: v2 is now
        # servable history (exact bytes), v3 is current.
        post = svc.read_at_version(artifact_id, 2)
        assert isinstance(post, VersionedContent)
        assert post.content == "c2"
        assert svc.read_at_version(artifact_id, 3).reason == CURRENT_VERSION_REASON


# ===========================================================================
# C — str/bytes round-trip pin across BOTH registries (one focused assertion)
# ===========================================================================


class TestStrBytesRoundTripAcrossRegistries:
    """One focused pin: a str body and a bytes body each round-trip with TYPE
    preserved through BOTH registries' retention AND read_at_version. (Sections
    A/B exercise type fidelity inside larger sequences; this isolates it so a
    type-coercion regression has a single named failure.)"""

    @pytest.mark.parametrize(
        ("body", "py_type"),
        [("a-str-body", str), (b"\x00\xff-bytes-body", bytes)],
        ids=["str", "bytes"],
    )
    def test_body_round_trips_with_type_on_both_registries(
        self, tmp_path: Path, body: str | bytes, py_type: type
    ) -> None:
        mem = ArtifactRegistry(
            retain_versions=True, retention_policy=RetentionPolicy(max_versions=8)
        )
        sql = SqliteArtifactRegistry(
            tmp_path / "roundtrip.db",
            retain_versions=True,
            retention_policy=RetentionPolicy(max_versions=8),
        )
        try:
            for reg in (mem, sql):
                artifact_id = uuid4()
                art = Artifact(
                    id=artifact_id, name="plan.md", version=1, content_hash="h"
                )
                reg.register_artifact(art, content="seed")
                _commit(reg, artifact_id, 2, body)  # capture the typed body at v2
                _commit(reg, artifact_id, 3, "make-v2-history")  # v2 → history

                # Raw getter round-trips the exact value AND type.
                got = reg.get_content_at_version(artifact_id, 2)
                assert got == body
                assert isinstance(got, py_type), (
                    f"{type(reg).__name__} coerced the retained body type: "
                    f"expected {py_type.__name__}, got {type(got).__name__}"
                )

                # read_at_version serves the same value AND type.
                out = CoordinatorService(reg).read_at_version(artifact_id, 2)
                assert isinstance(out, VersionedContent)
                assert out.content == body
                assert isinstance(out.content, py_type)
        finally:
            sql.close()


# ===========================================================================
# D — Snapshot consistent-cut capture parity (SB-17 / TX-1, Unit 2)
# ===========================================================================


def _snap_pair(
    tmp_path: Path, name: str, policy: RetentionPolicy | None = None
) -> tuple[ArtifactRegistry, SqliteArtifactRegistry]:
    """An in-memory + sqlite pair for snapshot-capture parity. Retention is OFF
    by default (the version-map-only / content=None branch); a policy switches
    both arms to the lazy retain_versions=True branch."""
    retain = policy is not None
    return (
        ArtifactRegistry(retain_versions=retain, retention_policy=policy),
        SqliteArtifactRegistry(
            tmp_path / name, retain_versions=retain, retention_policy=policy
        ),
    )


class TestSnapshotCaptureParity:
    """``capture_version_vector`` behaves IDENTICALLY on both registries except
    restart-survival (sqlite persists pins; in-memory does not). The divergence
    is ASSERTED, never masked (success criterion + R6)."""

    def test_capture_and_unknown_id_reject_identical(self, tmp_path: Path) -> None:
        # One identical sequence on both arms: register {A,B}, capture {A,B} →
        # same cut; then capture {A, unknown} → same typed rejection. Compared as
        # registry-agnostic normal forms.
        mem, sql = _snap_pair(tmp_path, "snap_parity.db")
        try:
            # Use FIXED ids so the cut maps are directly comparable across arms.
            a = UUID(int=0xA)
            b = UUID(int=0xB)
            unknown = UUID(int=0xDEAD)

            def run(reg) -> tuple:
                art_a = Artifact(id=a, name="a.md", version=1, content_hash="h")
                art_b = Artifact(id=b, name="b.md", version=1, content_hash="h")
                reg.register_artifact(art_a, content="a1")
                reg.register_artifact(art_b, content="b1")
                # peer bumps A to v2 before the capture → coherent cut {A:2,B:1}.
                _commit(reg, a, 2, "a2")
                ok = reg.capture_version_vector([a, b], session_token="tok-ok")
                bad = reg.capture_version_vector([a, unknown], session_token="tok-bad")
                return ok, bad

            mem_ok, mem_bad = run(mem)
            sql_ok, sql_bad = run(sql)

            # The WIN cut is identical across registries.
            assert mem_ok == sql_ok == {a: 2, b: 1}
            # The rejection is the same wire-stable reason + offending id.
            for bad in (mem_bad, sql_bad):
                assert isinstance(bad, VersionedReadRejection)
                assert bad.reason == UNKNOWN_ARTIFACT_REASON
                assert bad.artifact_id == unknown
                assert bad.current_version is None
        finally:
            sql.close()

    def test_gc_hold_then_release_parity(self, tmp_path: Path) -> None:
        # Identical pin-hold-then-collectible behavior on both arms under K=1.
        policy = RetentionPolicy(max_versions=1)
        mem, sql = _snap_pair(tmp_path, "snap_gc_parity.db", policy)
        try:
            a = UUID(int=0xCAFE)

            def run(reg) -> tuple:
                art = Artifact(id=a, name="plan.md", version=1, content_hash="h")
                reg.register_artifact(art, content="v1")  # current=1, retain on
                # Pin v1 while current.
                cut = reg.capture_version_vector([a], session_token="held")
                assert cut == {a: 1}
                writer = uuid4()
                _commit_cas(reg, a, writer, 1, "v2")  # K=1 {2} + pin{1}
                held = reg.get_content_at_version(a, 1)  # must survive (pinned)
                reg.release_session("held")
                _commit_cas(reg, a, writer, 2, "v3")  # K=1 evicts v1 now
                after = reg.get_content_at_version(a, 1)  # collectible → None
                return held, after

            assert run(mem) == run(sql) == ("v1", None)
        finally:
            sql.close()

    def test_restart_survival_divergence_asserted_not_masked(
        self, tmp_path: Path
    ) -> None:
        # THE DELIBERATE DIVERGENCE (R6): sqlite pins survive a registry reopen on
        # the same db file; an in-memory registry is process-scoped, so a fresh
        # instance starts with NO pins. Asserted explicitly so a regression that
        # accidentally made them agree (either direction) is caught.
        policy = RetentionPolicy(max_versions=1)
        a = UUID(int=0xF00D)
        db = tmp_path / "snap_restart.db"

        # --- sqlite arm: pin survives reopen ---
        sql1 = SqliteArtifactRegistry(db, retain_versions=True, retention_policy=policy)
        try:
            art = Artifact(id=a, name="plan.md", version=1, content_hash="h")
            sql1.register_artifact(art, content="v1")
            assert sql1.capture_version_vector([a], session_token="survivor") == {a: 1}
        finally:
            sql1.close()
        # Reopen a SECOND handle (the "restart"): the pin row is still there, so a
        # K=1-evicting commit must STILL hold v1 (the durable exemption survived).
        sql2 = SqliteArtifactRegistry(db, retain_versions=True, retention_policy=policy)
        try:
            writer = uuid4()
            _commit_cas(sql2, a, writer, 1, "v2")  # K=1 {2}; pin{1} from the reopen
            assert sql2.get_content_at_version(a, 1) == "v1", (
                "sqlite pin did NOT survive restart — R6 durability regressed"
            )
        finally:
            sql2.close()

        # --- in-memory arm: a fresh instance has NO pins (process-scoped) ---
        mem1 = ArtifactRegistry(
            retain_versions=True, retention_policy=policy
        )
        art = Artifact(id=a, name="plan.md", version=1, content_hash="h")
        mem1.register_artifact(art, content="v1")
        assert mem1.capture_version_vector([a], session_token="ephemeral") == {a: 1}
        # A FRESH in-memory registry is the "restart": no shared store, no pins.
        mem2 = ArtifactRegistry(
            retain_versions=True, retention_policy=policy
        )
        art2 = Artifact(id=a, name="plan.md", version=1, content_hash="h")
        mem2.register_artifact(art2, content="v1")
        writer = uuid4()
        _commit_cas(mem2, a, writer, 1, "v2")  # K=1 {2}; NO pin survived to mem2
        assert mem2.get_content_at_version(a, 1) is None, (
            "in-memory pins unexpectedly survived a fresh instance — the "
            "process-scoped contract (R6) was violated"
        )


def _commit_cas(reg, artifact_id: UUID, writer: UUID, expected: int, body: str) -> None:
    """A peer OCC commit via the registry ``commit_cas`` WIN (captures history)
    — shared by the snapshot-capture parity tests."""
    reg.set_agent_state(artifact_id, writer, MESIState.SHARED, tick=1)
    reg.commit_cas(
        artifact_id,
        writer,
        expected_version=expected,
        content_hash="h",
        content=body,
        tick=2,
    )


# ===========================================================================
# E — Snapshot SESSION parity through the SERVICE surface (Unit 9 / R5, R6)
# ===========================================================================
#
# Section D (above, Unit 2) exercised ``capture_version_vector`` at the REGISTRY
# level. Unit 9 adds the parity that rides the SERVICE surface a real session
# uses — ``begin_session`` / ``session_read`` / ``get_session_cut`` — and pins
# the in-memory↔sqlite divergence (restart-survival, R6) on BOTH arms so it is
# ASSERTED, never masked. R5 (concurrent overlapping sessions hold independent
# immutable cuts) is exercised identically across the two registries.


def _register(reg, artifact_id: UUID, name: str, body: str) -> None:
    art = Artifact(id=artifact_id, name=name, version=1, content_hash="h")
    reg.register_artifact(art, content=body)


class TestSnapshotSessionParityThroughService:
    """``begin_session`` + ``session_read`` behave IDENTICALLY on both registries
    on the NON-restart paths (capture, serve, overlap, release); the ONE
    divergence — restart-survival — is asserted explicitly on BOTH arms (sqlite
    survives, in-memory wiped). The success criterion + R6: divergence ASSERTED,
    not hidden."""

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_begin_and_serve_identical_on_both_arms(
        self, tmp_path: Path, arm: str
    ) -> None:
        # The non-restart happy path is registry-agnostic: begin a session over a
        # retained store, serve the pinned bytes, and read back the cut via the
        # service — same observable result on both arms.
        mem, sql = _snap_pair(
            tmp_path, "sess_serve.db", RetentionPolicy(max_versions=8)
        )
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a = uuid4()
            _register(reg, a, "plan.md", "PINNED-V1")
            owner = uuid4()
            session = svc.begin_session(read_set=[a], owner=owner)
            assert isinstance(session, SnapshotSession)
            assert dict(session.cut) == {a: 1}
            # The pinned bytes serve identically (lazy / retain on).
            served = svc.session_read(session.session_token, a, caller=owner)
            assert isinstance(served, VersionedContent)
            assert served.version == 1
            assert served.content == "PINNED-V1"
            # The durable/ephemeral pin is readable back through the registry on
            # both arms while live.
            assert svc.registry.get_session_cut(session.session_token) == {a: 1}
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_overlapping_sessions_hold_independent_immutable_cuts_R5(
        self, tmp_path: Path, arm: str
    ) -> None:
        # R5: two concurrent overlapping sessions over the SAME artifact hold
        # INDEPENDENT immutable cuts — the second session pins the moved version,
        # the first stays at its captured version. Identical on both arms.
        mem, sql = _snap_pair(
            tmp_path, "sess_overlap.db", RetentionPolicy(max_versions=8)
        )
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            a = uuid4()
            _register(reg, a, "plan.md", "V1")
            owner = uuid4()
            s1 = svc.begin_session(read_set=[a], owner=owner)  # pins {a:1}
            writer = uuid4()
            _commit_cas(reg, a, writer, 1, "V2")  # current → 2
            s2 = svc.begin_session(read_set=[a], owner=owner)  # pins {a:2}

            assert isinstance(s1, SnapshotSession) and isinstance(s2, SnapshotSession)
            # Independent immutable cuts: s1 still at v1, s2 at v2.
            assert dict(s1.cut) == {a: 1}
            assert dict(s2.cut) == {a: 2}
            # Each serves ITS OWN pinned version (no read skew, no cross-leak).
            r1 = svc.session_read(s1.session_token, a, caller=owner)
            r2 = svc.session_read(s2.session_token, a, caller=owner)
            assert isinstance(r1, VersionedContent) and r1.version == 1
            assert r1.content == "V1"
            assert isinstance(r2, VersionedContent) and r2.version == 2
            assert r2.content == "V2"

            # Releasing s1 must NOT disturb s2's independent cut.
            reg.release_session(s1.session_token)
            assert svc.registry.get_session_cut(s2.session_token) == {a: 2}
        finally:
            sql.close()

    def test_restart_survival_divergence_both_arms_asserted(
        self, tmp_path: Path
    ) -> None:
        # THE Unit-9 divergence (R6), through the SERVICE surface and asserted on
        # BOTH arms so a regression that accidentally made them agree — in EITHER
        # direction — is caught:
        #   * sqlite arm: a FRESH SqliteArtifactRegistry on the same db file (the
        #     "restart") still finds the pin → get_session_cut returns the cut AND
        #     a session_read by the owner still SERVES the pinned bytes.
        #   * in-memory arm: a FRESH ArtifactRegistry has NO pins → get_session_cut
        #     is None AND session_read fails closed (session_invalidated), never a
        #     live-HEAD serve.
        policy = RetentionPolicy(max_versions=8)
        a = UUID(int=0xF00D5E55)
        owner = UUID(int=0x0114E5)
        db = tmp_path / "sess_restart.db"

        # --- sqlite arm: the pin SURVIVES a reopen (durable session_pins) ---
        sql1 = SqliteArtifactRegistry(db, retain_versions=True, retention_policy=policy)
        try:
            _register(sql1, a, "plan.md", "SURVIVOR-V1")
            session = CoordinatorService(sql1).begin_session(read_set=[a], owner=owner)
            assert isinstance(session, SnapshotSession)
            token = session.session_token
        finally:
            sql1.close()
        # The "restart": a SECOND handle on the same file. The pin row persisted.
        sql2 = SqliteArtifactRegistry(db, retain_versions=True, retention_policy=policy)
        try:
            svc2 = CoordinatorService(sql2)
            # The cut is still readable post-restart (durable session_pins).
            assert svc2.registry.get_session_cut(token) == {a: 1}, (
                "sqlite session cut did NOT survive restart — R6 durability "
                "regressed"
            )
            # And the owner can still SERVE the pinned bytes from the survived cut.
            served = svc2.session_read(token, a, caller=owner)
            assert isinstance(served, VersionedContent), (
                "sqlite session_read did not serve after restart — the durable "
                "pin should still be intact"
            )
            assert served.version == 1
            assert served.content == "SURVIVOR-V1"
            # R13 across the restart (F3): the durable owner-binding survives too,
            # so a leaked-token FOREIGN caller is STILL rejected — survival (R6)
            # does not open an owner-isolation hole.
            with pytest.raises(SessionInvalidated):
                svc2.session_read(token, a, caller=uuid4())
        finally:
            sql2.close()

        # --- in-memory arm: a FRESH registry has NO pins (process-scoped) ---
        mem1 = ArtifactRegistry(retain_versions=True, retention_policy=policy)
        _register(mem1, a, "plan.md", "EPHEMERAL-V1")
        ephemeral = CoordinatorService(mem1).begin_session(read_set=[a], owner=owner)
        assert isinstance(ephemeral, SnapshotSession)
        eph_token = ephemeral.session_token
        # The "restart": a brand-new in-memory registry — no shared store, no pins.
        mem2 = ArtifactRegistry(retain_versions=True, retention_policy=policy)
        _register(mem2, a, "plan.md", "EPHEMERAL-V1")
        svc_mem2 = CoordinatorService(mem2)
        # The cut is GONE (process-scoped) — get_session_cut is None.
        assert svc_mem2.registry.get_session_cut(eph_token) is None, (
            "in-memory pins unexpectedly survived a fresh instance — the "
            "process-scoped contract (R6) was violated"
        )
        # And the session correctly FAILS CLOSED — never a live-HEAD serve. The
        # in-shape (server-minted) token classifies session_invalidated under the
        # Unit-5 liveness taxonomy (the post-restart-unknown safety case).
        result = svc_mem2.session_read(eph_token, a, caller=owner)
        assert isinstance(result, SessionReadRejection), (
            "in-memory session_read served (or crashed) after a fresh-instance "
            "restart — it must fail closed, never serve live HEAD"
        )
        assert result.reason == SESSION_INVALIDATED_REASON
