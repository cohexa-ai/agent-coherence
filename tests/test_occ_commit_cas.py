# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Concurrent-writer + bounded-progress proof suite for the OCC commit-CAS
(plan Unit 7, R11/R12/R6).

This is the cross-cutting *empirical* proof that the atomic version-checked
compare-and-swap closes the concurrent lost-update across the real stack — the
runtime counterpart to ``formal/tla/OCC.tla``'s ``NoLostUpdate`` safety proof.
Default-suite: fast, offline, no pytest marker.

The load-bearing test-design rule (from
``docs/solutions/best-practices/coordinator-invalidation-not-mutex-honest-coherence-claims-2026-06-04.md``
Fact 4): every concurrent arm holds a **fixed stale buffer** — both writers read
version ``N`` and each computes *distinct* content from that *same* ``N``. A
"read counter → +1 → write" arm is refetch-safe by construction and **passes
even if enforcement is broken**, so it would not stress the CAS at all. The
``content_hash`` here is therefore derived from the *writer identity*, never from
a re-fetched version.

Concurrency-realism caveat (from
``docs/solutions/test-failures/mp-pool-pid-distinctness-flaky-on-constrained-ci-2026-05-18.md``):
a constrained runner may serialize threads. So the *only* hard assertion is the
correctness property (exactly one win, every loser typed-conflicted, version
advanced by exactly one, no silent clobber). Any "did the interleave actually
overlap" check is a :class:`RuntimeWarning`, never an assertion. A
``threading.Barrier`` forces the race window so the CAS is genuinely exercised
when the runner does run the threads in parallel.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from uuid import UUID, uuid4

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.agent.runtime import AgentRuntime
from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import CasRetriesExhausted, GiverFenced
from ccs.core.hashing import compute_content_hash
from ccs.core.states import MESIState
from ccs.core.types import Artifact, ConflictDetail, FetchRequest
from ccs.strategies.lazy import LazyStrategy

# ----------------------------------------------------------------------
# Fixtures + helpers
# ----------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "state.db"


def _make_artifact(name: str = "plan.md", version: int = 1, content_hash: str = "h1") -> Artifact:
    return Artifact(id=uuid4(), name=name, version=version, content_hash=content_hash)


def _seed_shared_writers(
    reg: SqliteArtifactRegistry, n: int, *, version: int = 1
) -> tuple[Artifact, list[UUID]]:
    """Register one artifact at ``version`` and seed ``n`` SHARED writers on it.

    OCC writers are S/I by construction (they bypass the pessimistic acquire),
    so the winner is elected by the version check, never ``other_holder``.
    """
    art = _make_artifact(version=version, content_hash="h-init")
    reg.register_artifact(art, content="")
    writers = [uuid4() for _ in range(n)]
    for w in writers:
        reg.set_agent_state(art.id, w, MESIState.SHARED, tick=1)
    return art, writers


def _run_barrier_cas_round(
    reg: SqliteArtifactRegistry,
    artifact_id: UUID,
    writers: list[UUID],
    expected_version: int,
    *,
    tick: int = 5,
) -> dict[UUID, object]:
    """Fire every writer at ``commit_cas`` from a shared barrier, each carrying a
    FIXED stale buffer (content derived from the writer id, NOT a re-fetch).

    Returns ``{writer_id: commit_cas_result}``. ``results`` is written from
    distinct keys per thread, so the plain dict is safe without a lock.
    """
    results: dict[UUID, object] = {}
    barrier = threading.Barrier(len(writers))
    overlap_seen = threading.Event()
    in_flight = {"n": 0}
    in_flight_lock = threading.Lock()

    def attempt(writer_id: UUID) -> None:
        # FIXED stale buffer: content is a function of the writer identity, not
        # of any re-read version. Both writers commit DISTINCT content computed
        # from the SAME expected_version — the shape that actually stresses CAS.
        content_hash = hashlib.sha256(writer_id.bytes).hexdigest()
        barrier.wait()
        with in_flight_lock:
            in_flight["n"] += 1
            if in_flight["n"] > 1:
                overlap_seen.set()
        try:
            results[writer_id] = reg.commit_cas(
                artifact_id,
                writer_id,
                expected_version=expected_version,
                content_hash=content_hash,
                tick=tick,
            )
        finally:
            with in_flight_lock:
                in_flight["n"] -= 1

    threads = [threading.Thread(target=attempt, args=(w,)) for w in writers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Concurrency-realism probe: NOT an assertion (a constrained runner may
    # serialize the threads). Only the correctness property below is asserted.
    if not overlap_seen.is_set():
        warnings.warn(
            "barrier CAS round did not observe overlapping in-flight writers; "
            "the runner may have serialized threads. The correctness property is "
            "still asserted, but the race window was not stressed this run.",
            RuntimeWarning,
            stacklevel=2,
        )
    return results


def _assert_exactly_one_win_rest_conflict(
    results: dict[UUID, object],
    *,
    n_writers: int,
    expected_current: int,
) -> tuple[Artifact, list[UUID]]:
    """Assert the no-lost-update correctness property over one CAS round and
    return the winner's ``(Artifact, invalidated_ids)``."""
    wins = [r for r in results.values() if isinstance(r, tuple)]
    conflicts = [r for r in results.values() if isinstance(r, ConflictDetail)]
    assert len(wins) == 1, f"expected exactly one win, got {len(wins)}: {results}"
    assert len(conflicts) == n_writers - 1, (
        f"every non-winner must get a typed ConflictDetail; "
        f"got {len(conflicts)} of {n_writers - 1}"
    )
    # Two OCC writers are both SHARED → the loser is elected by the version
    # check, so the reason is ALWAYS version_mismatch (never other_holder).
    assert all(c.reason == "version_mismatch" for c in conflicts), [
        c.reason for c in conflicts
    ]
    assert all(c.current_version == expected_current for c in conflicts)
    return wins[0]


# ----------------------------------------------------------------------
# R11 — the headline test: two concurrent writers, fixed stale buffers
# ----------------------------------------------------------------------


def test_r11_two_concurrent_writers_exactly_one_wins_no_clobber(db_path: Path) -> None:
    """R11 (headline). Two ``threading.Barrier``-synced writers, each holding a
    FIXED stale buffer at the SAME version N, both call ``commit_cas`` against
    ONE registry. Exactly one wins, the other gets a ``ConflictDetail``, the
    final version is N+1 (NOT N+2), and the WINNER's content is what persisted
    (no silent clobber)."""
    with SqliteArtifactRegistry(db_path) as reg:
        art, writers = _seed_shared_writers(reg, 2, version=1)

        results = _run_barrier_cas_round(reg, art.id, writers, expected_version=1)

        winner_artifact, invalidated = _assert_exactly_one_win_rest_conflict(
            results, n_writers=2, expected_current=2
        )
        # Final version advanced by EXACTLY one — the loser's stale buffer did
        # not produce a second bump (N+1, not N+2).
        persisted = reg.get_artifact(art.id)
        assert persisted.version == 2
        # The persisted content_hash is the WINNER's — no silent clobber by the
        # loser. The winner is the only writer whose result is a (Artifact, ...).
        winner_id = next(w for w, r in results.items() if isinstance(r, tuple))
        expected_winner_hash = hashlib.sha256(winner_id.bytes).hexdigest()
        assert persisted.content_hash == expected_winner_hash
        assert winner_artifact.content_hash == expected_winner_hash
        assert winner_artifact.version == 2
        # The winner ends SHARED (an OCC writer holds no grant); the loser was
        # invalidated.
        assert reg.get_agent_state(art.id, winner_id) == MESIState.SHARED
        loser_id = next(w for w, r in results.items() if isinstance(r, ConflictDetail))
        assert loser_id in invalidated
        assert reg.get_agent_state(art.id, loser_id) == MESIState.INVALID


# ----------------------------------------------------------------------
# N-writer (N >= 3) variant — one winner per round, no dropped write
# ----------------------------------------------------------------------


@pytest.mark.parametrize("n_writers", [3, 5, 8])
def test_n_writers_fixed_stale_buffers_exactly_one_wins(
    db_path: Path, n_writers: int
) -> None:
    """N (>=3) barrier-synced writers, all with fixed stale buffers at the same
    N → exactly one wins that round; every loser gets a ``ConflictDetail``; the
    final version is N+1; no acknowledged write was dropped (the persisted hash
    is the single winner's)."""
    with SqliteArtifactRegistry(db_path) as reg:
        art, writers = _seed_shared_writers(reg, n_writers, version=1)

        results = _run_barrier_cas_round(reg, art.id, writers, expected_version=1)

        _assert_exactly_one_win_rest_conflict(
            results, n_writers=n_writers, expected_current=2
        )
        persisted = reg.get_artifact(art.id)
        assert persisted.version == 2  # exactly one bump across N racers
        winner_id = next(w for w, r in results.items() if isinstance(r, tuple))
        assert persisted.content_hash == hashlib.sha256(winner_id.bytes).hexdigest()


# ----------------------------------------------------------------------
# R12 — bounded progress under sustained contention
# ----------------------------------------------------------------------


def test_r12_sustained_contention_version_strictly_advances(db_path: Path) -> None:
    """R12 bounded progress. Under sustained contention a writer retries
    (re-read → recompute → ``commit_cas``) and commits keep LANDING: the version
    strictly advances every round, no permanent livelock. Models the caller-side
    retry loop directly against the registry (the cross-process path runs the
    same loop via ``reacquire()``).

    The protagonist carries its OWN last-known expected_version (a fixed stale
    buffer between re-reads). A contending peer bumps the version at the top of
    each round, so the protagonist's first attempt is guaranteed stale → it must
    re-read (mirroring ``reacquire()``) and retry, and the retry lands."""
    with SqliteArtifactRegistry(db_path) as reg:
        art = _make_artifact(version=1, content_hash="h-init")
        reg.register_artifact(art, content="")
        protagonist = uuid4()
        peer = uuid4()
        reg.set_agent_state(art.id, protagonist, MESIState.SHARED, tick=1)
        reg.set_agent_state(art.id, peer, MESIState.SHARED, tick=1)

        observed_versions: list[int] = []
        rounds = 6
        # The protagonist's last-read version (its expected_version comparand).
        # Seeded stale on purpose so round 0's first attempt also has to retry.
        protagonist_expected = reg.get_artifact(art.id).version
        for round_idx in range(rounds):
            # A contending peer commits FIRST, bumping the version so the
            # protagonist's held expected_version goes stale.
            peer_now = reg.get_artifact(art.id).version
            peer_result = reg.commit_cas(
                art.id, peer, expected_version=peer_now,
                content_hash=hashlib.sha256(f"peer-{round_idx}".encode()).hexdigest(),
                tick=round_idx,
            )
            assert isinstance(peer_result, tuple)
            # Peer winning re-invalidated the peer's peers, including the
            # protagonist; re-grant SHARED so it is OCC-eligible again (a re-read
            # in production lands it back in S). The peer itself already ends
            # SHARED (an OCC writer holds no grant), so the round stays OCC-vs-OCC
            # (version-elected, not other_holder); the explicit re-grant below is
            # defensive and keeps the protagonist SHARED.
            reg.set_agent_state(art.id, protagonist, MESIState.SHARED, tick=round_idx)
            reg.set_agent_state(art.id, peer, MESIState.SHARED, tick=round_idx)

            # The protagonist runs a bounded retry loop with its (now stale)
            # held expected_version. On version_mismatch it re-reads (refreshing
            # the comparand) and retries; the retry lands.
            landed = False
            for attempt in range(LazyStrategy().max_cas_retries() + 1):
                result = reg.commit_cas(
                    art.id, protagonist, expected_version=protagonist_expected,
                    content_hash=hashlib.sha256(
                        f"prot-{round_idx}-{attempt}".encode()
                    ).hexdigest(),
                    tick=round_idx,
                )
                if isinstance(result, ConflictDetail):
                    # Re-read: refresh the comparand to the observed current
                    # version (mirrors reacquire()'s mandatory fresh read).
                    assert result.reason == "version_mismatch"
                    protagonist_expected = result.current_version
                    continue
                landed = True
                observed_versions.append(result[0].version)
                # The protagonist now ends SHARED at this version (OCC writer
                # holds no grant); its next-round comparand is this version. The
                # explicit set_agent_state is defensive (it is already SHARED).
                protagonist_expected = result[0].version
                reg.set_agent_state(art.id, protagonist, MESIState.SHARED, tick=round_idx)
                break
            assert landed, f"protagonist starved (no commit landed) in round {round_idx}"

        # Bounded progress: every protagonist commit advanced the version, and
        # the version is strictly monotonic across the whole run (no livelock,
        # no regression / lost update).
        assert observed_versions == sorted(observed_versions)
        assert len(set(observed_versions)) == len(observed_versions)
        # Final version reflects 2 commits/round (peer + protagonist) from N=1.
        assert reg.get_artifact(art.id).version == 1 + 2 * rounds


def test_r12_retry_exhaustion_surfaces_typed_terminal_no_silent_drop() -> None:
    """R12 terminal. A writer whose retries are exhausted surfaces the typed
    terminal ``CasRetriesExhausted`` — NEVER a silent drop. Driven through the
    library ``AgentRuntime.write_cas`` retry loop with a peer that keeps the
    version moving so every attempt loses."""
    coordinator = CoordinatorService(ArtifactRegistry())
    artifact = coordinator.register_artifact(name="plan.md", content="v1")

    # Put `runtime` in SHARED (a lone fetch grants EXCLUSIVE; a co-reader
    # downgrades it to SHARED so commit_cas's D4 precondition holds).
    runtime = AgentRuntime(
        agent_id=uuid4(), coordinator=coordinator, strategy=LazyStrategy()
    )
    coordinator.fetch(
        FetchRequest(artifact_id=artifact.id, requesting_agent_id=runtime.agent_id, requested_at_tick=1)
    )
    coordinator.fetch(
        FetchRequest(artifact_id=artifact.id, requesting_agent_id=uuid4(), requested_at_tick=1)
    )
    runtime.cache.put(
        artifact.id,
        runtime.strategy.on_fetch(
            artifact_id=artifact.id, version=artifact.version,
            state=MESIState.SHARED, now_tick=1,
        ),
    )

    # A peer that ALWAYS commits first, on every attempt, so the protagonist's
    # re-read is perpetually stale and it exhausts its retry budget. The peer is
    # a fresh SHARED writer minted per call (the prior peer is invalidated by its
    # own win, so a new identity keeps winning).
    real_commit_cas = coordinator.commit_cas
    attempt_count = {"n": 0}

    def _winning_peer_then_real(**kwargs):  # type: ignore[no-untyped-def]
        attempt_count["n"] += 1
        peer = uuid4()
        coordinator.fetch(
            FetchRequest(artifact_id=artifact.id, requesting_agent_id=peer, requested_at_tick=1)
        )
        coordinator.fetch(
            FetchRequest(artifact_id=artifact.id, requesting_agent_id=uuid4(), requested_at_tick=1)
        )
        peer_v = coordinator.registry.get_artifact(artifact.id).version
        peer_res = real_commit_cas(
            agent_id=peer, artifact_id=artifact.id,
            expected_version=peer_v, content_hash=compute_content_hash(f"peer-{peer_v}"),
        )
        assert isinstance(peer_res, tuple)  # peer won, version advanced
        # The protagonist's call now runs against the advanced version with its
        # stale expected_version → guaranteed version_mismatch.
        return real_commit_cas(**kwargs)

    coordinator.commit_cas = _winning_peer_then_real  # type: ignore[assignment]
    try:
        with pytest.raises(CasRetriesExhausted) as excinfo:
            runtime.write_cas(
                artifact.id,
                make_content=lambda entry: (f"prot-v{entry.local_version + 1}", None),
                now_tick=2,
            )
    finally:
        coordinator.commit_cas = real_commit_cas  # type: ignore[assignment]

    # The terminal is typed and reports the full attempt count (no silent drop).
    assert excinfo.value.attempts == LazyStrategy().max_cas_retries() + 1
    assert attempt_count["n"] == LazyStrategy().max_cas_retries() + 1
    # No silent drop: the protagonist's unconfirmed content never landed, and its
    # cache is not left MODIFIED with a write that did not commit.
    final = coordinator.registry.get_artifact(artifact.id)
    entry = runtime.cache.get(artifact.id)
    assert entry is not None
    assert entry.state != MESIState.MODIFIED
    # The protagonist's last attempted content is not the persisted writer.
    last_writer_content = f"prot-v{final.version}"
    assert runtime.content(artifact.id) != last_writer_content


# ----------------------------------------------------------------------
# A handed-off giver racing its successor, through the service (#185, U4)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_a_handed_off_giver_racing_its_successor_never_wins(db_path: Path, backend: str) -> None:
    """KTD1 under contention, through the service's ``commit_cas`` -- the race
    arms above call the registry directly, which has no giver fence. A giver
    that handed its standing read to a successor, and the successor, fire at
    the transfer version from a barrier, each with a FIXED stale buffer and a
    fresh incarnation carrying its session-level identity. Exactly one write
    lands and it is the successor's: the giver is refused with the giver reason
    when it runs first, or meets the version mismatch once the successor's win
    ended the record. Fails if the giver can ever win, which is the lost
    update the handoff exists to close."""
    reg = ArtifactRegistry() if backend == "memory" else SqliteArtifactRegistry(db_path)
    try:
        svc = CoordinatorService(reg)
        art = svc.register_artifact(name="plan.md", content="v1")
        giver, giver_read, successor = uuid4(), uuid4(), uuid4()
        reg.set_agent_state(art.id, giver_read, MESIState.SHARED, trigger="fetch", tick=1)
        (handed,) = svc.transfer(
            giver=giver, successor=successor, holders={art.id: giver_read}, successor_known=True
        )
        assert handed.transferred

        results: dict[UUID, object] = {}
        barrier = threading.Barrier(2)
        overlap_seen = threading.Event()
        in_flight = {"n": 0}
        in_flight_lock = threading.Lock()

        def attempt(session: UUID) -> None:
            content_hash = hashlib.sha256(session.bytes).hexdigest()
            barrier.wait()
            with in_flight_lock:
                in_flight["n"] += 1
                if in_flight["n"] > 1:
                    overlap_seen.set()
            try:
                results[session] = svc.commit_cas(
                    agent_id=uuid4(),
                    artifact_id=art.id,
                    expected_version=1,
                    content_hash=content_hash,
                    caller=session,
                )
            except GiverFenced as exc:
                results[session] = exc
            finally:
                with in_flight_lock:
                    in_flight["n"] -= 1

        previous = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        try:
            threads = [threading.Thread(target=attempt, args=(s,)) for s in (giver, successor)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        finally:
            sys.setswitchinterval(previous)

        if not overlap_seen.is_set():
            warnings.warn(
                "giver-vs-successor round did not observe overlapping in-flight "
                "writers; the runner may have serialized threads. The correctness "
                "property is still asserted.",
                RuntimeWarning,
                stacklevel=2,
            )
        won, lost = results[successor], results[giver]
        assert isinstance(won, tuple) and won[0].version == 2, won
        assert isinstance(lost, GiverFenced) or lost == ConflictDetail("version_mismatch", 2), lost
        persisted = reg.get_artifact(art.id)
        assert persisted.version == 2
        assert persisted.content_hash == hashlib.sha256(successor.bytes).hexdigest()
        record, live = reg.get_transfer_record(art.id)
        assert (record.status, live) == ("completed", False)
    finally:
        if isinstance(reg, SqliteArtifactRegistry):
            reg.close()


# ----------------------------------------------------------------------
# Cross-process funnel — same property through the coordinator HTTP path
# ----------------------------------------------------------------------


class _Client:
    """Tiny urllib client for the coordinator HTTP server. Returns (status, body)."""

    def __init__(self, host: str, port: int, secret: str) -> None:
        self.base = f"http://{host}:{port}"
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        }

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        data = json.dumps(body).encode("utf-8")
        req = urlrequest.Request(
            self.base + path, data=data, method="POST", headers=dict(self.headers)
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urlerror.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8") or "{}")


_SID_NS = uuid.UUID("22222222-2222-4222-8222-222222222222")


def _sid(label: str) -> str:
    return str(uuid.uuid5(_SID_NS, f"occ-unit7:{label}"))


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@pytest.fixture
def http_coordinator(tmp_path: Path):
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="unit7-instance")
    server.serve_in_thread()
    time.sleep(0.05)
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def http_client(http_coordinator) -> _Client:
    secret = load_secret(http_coordinator.coordinator_root)
    assert secret is not None
    return _Client("127.0.0.1", http_coordinator.port, secret)


def _occ_pre_read_version(client: _Client, sid: str, path: str, h: str) -> int:
    """pre-read a tracked path → register SHARED + seed, returning the version
    an OCC writer uses as ``expected_version`` (two response shapes, mirroring
    ``CoherentVolume._pre_read_version``)."""
    s, b = client.post("/hooks/pre-read", {"session_id": sid, "path": path, "content_hash": h})
    assert s == 200, b
    if isinstance(b.get("version"), int):
        return b["version"]
    summary = b.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("current_version"), int):
        return summary["current_version"]
    raise AssertionError(f"pre-read surfaced no OCC version: {b}")


def _artifact_version(coordinator, path: str) -> int:
    aid = coordinator.registry.lookup_artifact_id_by_name(path)
    assert aid is not None
    art = coordinator.registry.get_artifact(aid)
    assert art is not None
    return art.version


def test_cross_process_funnel_two_writers_no_lost_update(
    http_coordinator, http_client: _Client
) -> None:
    """The R11 property end-to-end through the coordinator HTTP path Unit 6
    wired: two OCC clients each do pre-read → ``post-edit-cas`` with the SAME
    stale ``expected_version`` (fixed stale buffers — distinct content from one
    N), barrier-synced. Exactly one wins (v2), the loser gets a typed
    ``version_mismatch``, and the final version is v1+1 (no lost update)."""
    a_sid, b_sid = _sid("funnelA"), _sid("funnelB")
    v1 = _occ_pre_read_version(http_client, a_sid, "plan.md", _hash("v1"))
    assert _occ_pre_read_version(http_client, b_sid, "plan.md", _hash("v1")) == v1

    barrier = threading.Barrier(2)
    results: dict[str, tuple[int, dict]] = {}

    def commit(sid: str, label: str) -> None:
        # Fixed stale buffer: each client writes DISTINCT content derived from
        # its label, both against the SAME expected_version v1 (NOT a re-fetch).
        barrier.wait()
        results[label] = http_client.post(
            "/hooks/post-edit-cas",
            {
                "session_id": sid,
                "path": "plan.md",
                "success": True,
                "content_hash": _hash(f"v2-{label}"),
                "expected_version": v1,
            },
        )

    threads = [
        threading.Thread(target=commit, args=(a_sid, "A")),
        threading.Thread(target=commit, args=(b_sid, "B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wins = [lbl for lbl, (_, body) in results.items() if body.get("ok") is True]
    losers = [lbl for lbl, (_, body) in results.items() if body.get("ok") is False]
    assert len(wins) == 1, f"exactly one OCC client must win end-to-end: {results}"
    assert len(losers) == 1
    loser_body = results[losers[0]][1]
    assert loser_body["reason"] == "version_mismatch"
    assert "degraded" not in loser_body  # a clean typed conflict, not a degrade
    # Final version is exactly v1+1 — the loser's stale buffer did NOT clobber.
    assert _artifact_version(http_coordinator, "plan.md") == v1 + 1


def test_cross_process_funnel_winner_body_is_n_plus_one(
    http_coordinator, http_client: _Client
) -> None:
    """End-to-end sanity for the winning body shape: a single OCC client's
    pre-read → post-edit-cas returns ``{ok: true, version: N+1}`` and the
    coordinator's authoritative version matches (the funnel actually commits)."""
    sid = _sid("funnelSolo")
    v = _occ_pre_read_version(http_client, sid, "plan.md", _hash("solo-v1"))
    s, b = http_client.post(
        "/hooks/post-edit-cas",
        {"session_id": sid, "path": "plan.md", "success": True,
         "content_hash": _hash("solo-v2"), "expected_version": v},
    )
    assert s == 200
    assert b == {"ok": True, "version": v + 1}
    assert _artifact_version(http_coordinator, "plan.md") == v + 1


# ----------------------------------------------------------------------
# Watchdog late-completion residual (deferred; documented behavior)
# ----------------------------------------------------------------------


def test_watchdog_late_completion_is_false_negative_ack_never_lost_update(
    db_path: Path,
) -> None:
    """The watchdog late-completion residual, pinned (T4 — deferred to the
    cross-host follow-on, but its *safety boundary* is asserted here).

    When a client's ``commit_cas`` is watchdog-degraded it raises fail-closed,
    yet the coordinator's commit can still land LATE (after the client gave up).
    This test pins what that late landing can and cannot do:

    - UNCONTENDED (no peer bumped the version in the window): the late completion
      WINS — version N→N+1 carrying the late writer's ``content_hash``. The client
      already raised, so it never learned the write landed: a **false-negative
      acknowledgment**, NOT a lost update. On its next read the client sees its
      own bytes at N+1 and reconverges idempotently.
    - CONTENDED (a peer committed N→N+1 inside the window): the SAME late
      completion — still carrying the stale ``expected_version=N`` — gets a typed
      :class:`ConflictDetail` (``current_version=N+1``), NOT a clobber. The
      version CAS makes a late landing safe against an *acknowledged* peer write.

    ⇒ the residual is a false-negative ack, never silent data loss. The only
    deferred concern is the registry↔disk divergence it can leave (version N+1
    recorded while the raising client wrote no on-disk bytes) — a cross-host
    follow-on item, not a single-host correctness hole.
    """
    # --- Uncontended: the late completion lands as a clean monotonic bump. ---
    with SqliteArtifactRegistry(db_path) as reg:
        art, (writer,) = _seed_shared_writers(reg, 1, version=1)
        late_hash = hashlib.sha256(b"late-uncontended").hexdigest()
        result = reg.commit_cas(
            art.id, writer, expected_version=1, content_hash=late_hash, tick=9
        )
        assert isinstance(result, tuple), "uncontended late completion must WIN"
        updated, _ = result
        assert updated.version == 2, "a benign phantom bump is N+1 (monotonic)"
        assert updated.content_hash == late_hash, "the late writer's bytes are recorded"

    # --- Contended: a peer's acked write is NOT clobbered by a late straggler. ---
    db2 = db_path.parent / "state-contended.db"
    with SqliteArtifactRegistry(db2) as reg:
        art, (peer, straggler) = _seed_shared_writers(reg, 2, version=1)
        peer_hash = hashlib.sha256(b"peer-acked").hexdigest()
        peer_result = reg.commit_cas(
            art.id, peer, expected_version=1, content_hash=peer_hash, tick=9
        )
        assert isinstance(peer_result, tuple), "the peer commit must land first"

        # The straggler's watchdog-degraded commit lands LATE, still carrying the
        # now-stale expected_version=1 → the version CAS rejects it as a conflict.
        straggler_hash = hashlib.sha256(b"straggler-late").hexdigest()
        late = reg.commit_cas(
            art.id, straggler, expected_version=1, content_hash=straggler_hash, tick=10
        )
        assert isinstance(late, ConflictDetail), (
            "a late completion must NOT clobber an acknowledged peer write"
        )
        assert late.current_version == 2

        # The acked peer's bytes survive intact — the straggler's conflict caused
        # no mutation, so there is no silent loss.
        survived = reg.get_artifact(art.id)
        assert survived is not None
        assert survived.version == 2
        assert survived.content_hash == peer_hash, "the acked peer's write must survive"
