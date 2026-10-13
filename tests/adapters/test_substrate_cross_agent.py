# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Cross-agent coherence over a substrate binding (Unit 5).

These drive the coordinator-mediated layer with a REAL coordinator subprocess
(spawned via the shipped lifecycle, torn down in ``finally``) and a FAKE
in-memory substrate, so the coordinator-mediated behaviour — pull invalidation,
the divergence taxonomy, the never-ship-a-store commit path — is exercised
without a real Postgres or S3.

The fake models the shared substrate state (one row / one object) in a store
shared by every agent's binding view, mirroring reality: distinct agents, one
underlying artifact. It can script an ``UNKNOWN`` write (landed or not) so the
reconciliation dispatch is reachable, and it reconciles by the same
token-identity logic both real bindings use.
"""

from __future__ import annotations

import dataclasses
import hashlib
import http.server
import logging
import os
import threading
import traceback
from pathlib import Path

import pytest

import ccs.adapters.substrate as substrate_module
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.substrate import (
    CasConflict,
    CasUnknown,
    CasWriteResult,
    CasWritten,
    CoordinatedSubstrate,
    CoordinatorConflict,
    CoordinatorWin,
    ReconcileDecision,
    ReconcileVerdict,
    SubstrateCoordinatorSession,
)
from ccs.core.exceptions import (
    COMMIT_UNCONFIRMED_REASON,
    VERSION_MISMATCH_REASON,
    CasVersionConflict,
    CoherenceError,
    CommitUnconfirmed,
    StaleView,
    ViewWedged,
)
from ccs.core.substrate import CapabilityDescriptor, Tier
from tests.test_coherence_client_tls import (
    _make_handler_class,
    _mint_bundle,
    _start_tls_server,
    requires_openssl,
)

REF = "workspace/shared.bin"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


# --- fake substrate ---------------------------------------------------------


class _FakeStore:
    """The shared substrate state (one row / one object), shared by all views."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, str]] = {}
        self._counter = 0

    def _mint(self) -> str:
        self._counter += 1
        return f"tok-{self._counter}"

    def seed(self, ref: str, data: bytes) -> str:
        token = self._mint()
        self._data[ref] = (data, token)
        return token

    def get(self, ref: str) -> tuple[bytes, str] | None:
        return self._data.get(ref)

    def set(self, ref: str, data: bytes) -> str:
        return self.seed(ref, data)

    def delete(self, ref: str) -> None:
        self._data.pop(ref, None)


def _descriptor(arm: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        tier=Tier.NATIVE_CAS,
        version_source="fake row-version" if arm == "row" else "fake object ETag",
        least_privilege="fake",
        consistency_note="fake single-primary",
    )


class _FakeSubstrate:
    """A per-agent binding view over the shared store, implementing the
    reconciling-substrate surface with realistic token-identity logic."""

    def __init__(self, store: _FakeStore, arm: str = "object") -> None:
        self._store = store
        self._arm = arm
        self._descriptor = _descriptor(arm)
        self.cas_calls: list[tuple[str, str, bytes]] = []
        # A QUEUE of scripted outcomes (not a single slot): a re-drive path issues
        # a second cas_write, so tests must script both legs independently.
        self._scripts: list[tuple[str, bool]] = []
        # Runs inside reconcile_after_unknown (before the verdict) — a seam to
        # inject a byte-identical peer's coordinator bump mid-commit.
        self.reconcile_hook = None

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def script_unknown(self, *, landed: bool) -> None:
        """Enqueue: the next scripted ``cas_write`` returns ``CasUnknown``;
        ``landed`` applies the write to the store."""
        self._scripts.append(("unknown", landed))

    def script_conflict(self) -> None:
        """Enqueue: the next scripted ``cas_write`` returns ``CasConflict`` (no
        write landed)."""
        self._scripts.append(("conflict", False))

    def script_ghost_conflict(self) -> None:
        """Enqueue: the next scripted ``cas_write`` applies the write (my bytes
        land under a fresh token — an in-flight ghost) THEN returns ``CasConflict``,
        so the re-drive sees a moved token carrying its own intended bytes."""
        self._scripts.append(("ghost", True))

    def read(self, artifact_ref: str) -> tuple[bytes, str]:
        entry = self._store.get(artifact_ref)
        if entry is None:
            raise KeyError(artifact_ref)
        return entry

    def cas_write(
        self, artifact_ref: str, *, expected_token: str, new_bytes: bytes
    ) -> CasWriteResult:
        self.cas_calls.append((artifact_ref, expected_token, bytes(new_bytes)))
        if self._scripts:
            kind, landed = self._scripts.pop(0)
            if kind == "unknown":
                if landed:
                    self._store.set(artifact_ref, bytes(new_bytes))
                return CasUnknown()
            if kind == "ghost":
                # The in-flight ghost landed my bytes under a fresh token, then the
                # (late) response is a conflict — the re-drive must detect the ghost.
                self._store.set(artifact_ref, bytes(new_bytes))
                return CasConflict()
            return CasConflict()
        entry = self._store.get(artifact_ref)
        if entry is None or entry[1] != expected_token:
            return CasConflict()
        return CasWritten(token=self._store.set(artifact_ref, bytes(new_bytes)))

    def reconcile_after_unknown(
        self, artifact_ref: str, *, expected_token: str, intended_hash: str
    ) -> ReconcileDecision:
        if self.reconcile_hook is not None:
            self.reconcile_hook()
        entry = self._store.get(artifact_ref)
        if entry is None:
            return ReconcileDecision(ReconcileVerdict.HOLD, None, None)
        observed_bytes, observed_token = entry
        if observed_token == expected_token:
            return ReconcileDecision(ReconcileVerdict.RE_DRIVE, observed_bytes, observed_token)
        if _sha256(observed_bytes) == intended_hash:
            return ReconcileDecision(ReconcileVerdict.CONVERGE, observed_bytes, observed_token)
        # The token moved and the bytes differ: object → CONFLICT, row → RE_DERIVE.
        verdict = ReconcileVerdict.CONFLICT if self._arm == "object" else ReconcileVerdict.RE_DERIVE
        return ReconcileDecision(verdict, observed_bytes, observed_token)


def _agent(
    store: _FakeStore, session: SubstrateCoordinatorSession, *, arm: str = "object"
) -> tuple[CoordinatedSubstrate, _FakeSubstrate]:
    fake = _FakeSubstrate(store, arm)
    return CoordinatedSubstrate(fake, session), fake


def _session(tmp_path: Path, fast_cfg: LifecycleConfig) -> SubstrateCoordinatorSession:
    return SubstrateCoordinatorSession(tmp_path, managed=("**",), config=fast_cfg)


# --- spawn: the pre-spawn policy write creates .coherence/ at 0700 ----------


def test_spawn_creates_coherence_dir_at_0700_without_tighten_warning(
    tmp_path: Path, fast_cfg: LifecycleConfig, caplog: pytest.LogCaptureFixture
) -> None:
    """The strict-policy write creates ``.coherence/`` ahead of the lifecycle
    and must do so at the 0700 the lifecycle requires; otherwise every fresh
    workspace spawns with a "tightened existing .coherence directory" warning
    about a directory the session itself created a moment earlier. umask is
    pinned to 022 so a permissive default ``mkdir`` is observable."""
    prior_umask = os.umask(0o022)
    try:
        caplog.set_level(logging.WARNING, logger="ccs.adapters.claude_code.lifecycle")
        _session(tmp_path, fast_cfg)
        assert ((tmp_path / ".coherence").stat().st_mode & 0o777) == 0o700
        tightened = [
            r.getMessage()
            for r in caplog.records
            if "tightened existing .coherence directory" in r.getMessage()
        ]
        assert tightened == []
    finally:
        os.umask(prior_umask)
        stop_coordinator(tmp_path)


# --- happy: pull invalidation before act (LOAD-BEARING) ---------------------


def test_peer_commit_denies_next_act_before_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        b, _fake_b = _agent(store, sb)
        _a_bytes, a_tok = a.read(REF)
        _b_bytes, b_tok = b.read(REF)

        b.commit(REF, expected_token=b_tok, new_bytes=b"v2")  # B wins; A invalidated

        # A's NEXT binding-mediated act is DENIED as the uniform typed conflict,
        # BEFORE the substrate is touched — the case a bare CAS never surfaces.
        with pytest.raises(StaleView):
            a.commit(REF, expected_token=a_tok, new_bytes=b"v2-from-A")
        assert fake_a.cas_calls == []  # deny-before-act: no substrate write attempted
    finally:
        stop_coordinator(tmp_path)


def test_read_rejects_unknown_on_stale_value(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    # A typo'd on_stale must fail closed (ValueError) rather than silently falling
    # through to the permissive 'allow' path — validated before the substrate is
    # even touched.
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fake_a = _agent(store, sa)
        with pytest.raises(ValueError):
            a.read(REF, on_stale="raize")
    finally:
        stop_coordinator(tmp_path)


def test_read_time_deny_surfaces_stale_view(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        b, _fb = _agent(store, sb)
        a.read(REF)
        _b_bytes, b_tok = b.read(REF)
        b.commit(REF, expected_token=b_tok, new_bytes=b"v2")

        # on_stale='allow' (default) returns bytes; 'raise' surfaces StaleView.
        assert a.read(REF) == store.get(REF)
        with pytest.raises(StaleView):
            a.read(REF, on_stale="raise")
    finally:
        stop_coordinator(tmp_path)


def test_identity_stable_across_read_and_commit_fresh_after_reacquire(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FakeStore()
    store.seed(REF, b"v1")
    seen: list[str] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if path in ("/hooks/pre-read", "/hooks/post-edit-cas"):
            seen.append(payload["session_id"])
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        before = a.session_id
        _bytes, tok = a.read(REF)
        a.commit(REF, expected_token=tok, new_bytes=b"v2")
        # The pre-read AND the post-edit-cas resolve to the SAME identity.
        assert set(seen) == {before}

        a.reacquire(REF)
        assert a.session_id != before  # a fresh id is minted ONLY on reacquire
    finally:
        stop_coordinator(tmp_path)


# --- divergence 1: coordinator-leg UNKNOWN (late-land / degrade) -------------


def test_divergence1_coordinator_leg_unknown_no_re_drive(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FakeStore()
    store.seed(REF, b"v1")
    real_post = substrate_module._coordinator_post

    def failing_commit(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if path == "/hooks/post-edit-cas":
            raise substrate_module.CoordinatorUnavailable("simulated commit timeout")
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", failing_commit)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        _bytes, tok = a.read(REF)
        # Substrate write LANDS (WIN), but the coordinator bump times out.
        with pytest.raises(CommitUnconfirmed):
            a.commit(REF, expected_token=tok, new_bytes=b"v2")
        assert len(fake_a.cas_calls) == 1  # NEVER blind re-drive after a landed write
        assert store.get(REF)[0] == b"v2"  # the substrate write is durable
    finally:
        stop_coordinator(tmp_path)


# --- divergence 2: substrate UNKNOWN, per arm -------------------------------


def test_divergence2_converge_drives_bump_and_invalidates_peer(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """PG-arm / never-converge-wedge negative: a landed-unknown write CONVERGES
    and its coordinator bump STILL fires — the peer is invalidated (the bump is
    not stranded)."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="row")
        b, _fb = _agent(store, sb, arm="row")
        _ab, _atok = a.read(REF)
        _bb, b_tok = b.read(REF)

        fake_a.script_unknown(landed=True)  # A's write lands but the ack is lost
        result = a.commit(REF, expected_token=_atok, new_bytes=b"v2")

        assert result.converged is True
        assert result.summary == "converged"  # never "landed" — attribution disclaimed
        assert len(fake_a.cas_calls) == 1  # no re-drive of a landed write
        # The converge bump fired → the peer is invalidated (not stranded).
        with pytest.raises(StaleView):
            b.commit(REF, expected_token=b_tok, new_bytes=b"vB")
    finally:
        stop_coordinator(tmp_path)


def test_divergence2_re_drive_under_held_token(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """S3-arm: an UNKNOWN write that did NOT land (token unmoved) is re-driven
    ONCE under the held token, then lands and bumps."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        _ab, a_tok = a.read(REF)

        fake_a.script_unknown(landed=False)  # not landed → token unmoved → RE_DRIVE
        result = a.commit(REF, expected_token=a_tok, new_bytes=b"v2")

        assert result.converged is False  # a clean re-drive win, not a converge
        assert len(fake_a.cas_calls) == 2  # initial (unknown) + one re-drive
        assert store.get(REF)[0] == b"v2"
    finally:
        stop_coordinator(tmp_path)


def test_divergence2_converge_complete_on_byte_identical_peer(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """S3-arm: a landed-unknown write whose bump loses to a byte-IDENTICAL peer
    completes as converged (coordinator already holds the intended hash) — NO
    re-drive, NO second bump."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        b, _fb = _agent(store, sb, arm="object")
        _ab, a_tok = a.read(REF)
        _bb, _b_tok = b.read(REF)  # B is SHARED@v1 so it can bump the coordinator

        intended = _sha256(b"v2")

        def peer_bumps_first() -> None:
            # A byte-identical peer carries b"v2" to the coordinator first.
            sb.commit_cas(REF, expected_version=1, content_hash=intended)

        fake_a.script_unknown(landed=True)  # A's substrate write landed (b"v2")
        fake_a.reconcile_hook = peer_bumps_first
        result = a.commit(REF, expected_token=a_tok, new_bytes=b"v2")

        assert result.converged is True
        assert len(fake_a.cas_calls) == 1  # no re-drive, no second substrate write
    finally:
        stop_coordinator(tmp_path)


def test_divergence2_conflict_on_different_bytes(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """S3-arm: an UNKNOWN write where the token moved to DIFFERENT bytes is a
    real peer conflict — typed, never re-driven."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        _ab, a_tok = a.read(REF)

        fake_a.script_unknown(landed=False)
        store.set(REF, b"foreign")  # a foreign writer moved the substrate

        with pytest.raises(CasVersionConflict):
            a.commit(REF, expected_token=a_tok, new_bytes=b"v2")
        assert len(fake_a.cas_calls) == 1  # never re-driven
    finally:
        stop_coordinator(tmp_path)


# --- never-ship-a-store on the wire -----------------------------------------


def test_commit_wire_payload_carries_hash_not_content(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _FakeStore()
    store.seed(REF, b"v1")
    captured: list[dict] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if path == "/hooks/post-edit-cas":
            captured.append(dict(payload))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        _bytes, tok = a.read(REF)
        a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert captured, "the commit must POST /hooks/post-edit-cas"
        for payload in captured:
            assert payload["content_hash"] == _sha256(b"v2")
            assert "content" not in payload  # bytes are NEVER sent
    finally:
        stop_coordinator(tmp_path)


# --- forbidden: coordinator-bump-first --------------------------------------


def test_coordinator_bump_first_is_forbidden(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write that FAILS the substrate CAS must never reach the coordinator —
    so a peer is never invalidated for a write that did not land."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    posts: list[str] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        posts.append(path)
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        b, fake_b = _agent(store, sb)
        _ab, a_tok = a.read(REF)
        _bb, b_tok = b.read(REF)

        fake_b.script_conflict()  # B's substrate CAS fails
        with pytest.raises(CasVersionConflict):
            b.commit(REF, expected_token=b_tok, new_bytes=b"vB")
        # The failed substrate CAS never drove a coordinator bump.
        assert "/hooks/post-edit-cas" not in posts

        # A was NOT invalidated → A commits cleanly (proof the bump never fired).
        result = a.commit(REF, expected_token=a_tok, new_bytes=b"vA")
        assert result.version >= 2
    finally:
        stop_coordinator(tmp_path)


# --- crash-between-legs → coordinator-behind → ViewWedged -------------------


def test_crash_between_legs_surfaces_view_wedged(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A substrate write that lands while the coordinator bump is skipped (a
    crash between the legs) leaves the coordinator behind; a peer's next binding
    read is a wedged view. Recovery is reacquire (the carve-out: a non-re-reading
    peer stays unprotected)."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        b, _fb = _agent(store, sb)
        a.read(REF)  # seeds the coordinator artifact @ v1 / hash(v1)

        # A's substrate write lands but the coordinator bump never fires (crash).
        store.set(REF, b"v2-crashed")

        with pytest.raises(ViewWedged):
            b.read(REF)
        # reacquire recovers the fresh bytes without raising.
        rec_bytes, _rec_tok = b.reacquire(REF)
        assert rec_bytes == b"v2-crashed"
    finally:
        stop_coordinator(tmp_path)


# --- cross-substrate uniformity ---------------------------------------------


@pytest.mark.parametrize("arm", ["row", "object"])
def test_uniform_typed_conflict_across_substrates(
    tmp_path: Path, fast_cfg: LifecycleConfig, arm: str
) -> None:
    """The SAME typed deny (StaleView) fires for a row-shaped fake AND an
    object-shaped fake — the cross-substrate uniformity co-headline."""
    ref = f"workspace/{arm}.bin"
    store = _FakeStore()
    store.seed(ref, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa, arm=arm)
        b, _fb = _agent(store, sb, arm=arm)
        _ab, a_tok = a.read(ref)
        _bb, b_tok = b.read(ref)
        b.commit(ref, expected_token=b_tok, new_bytes=b"v2")
        with pytest.raises(StaleView):
            a.commit(ref, expected_token=a_tok, new_bytes=b"vA")
    finally:
        stop_coordinator(tmp_path)


# --- admit-on-absent (the fence is inert by design) -------------------------


def test_admit_on_absent_occ_writer_commits(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """v1 captures no read_generation, so the OCC writer sits on the fence's
    admit-on-absent path: a clean commit LANDS (the substrate CAS arbitrates) —
    no fence rejection is claimed."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        _bytes, tok = a.read(REF)
        result = a.commit(REF, expected_token=tok, new_bytes=b"v2")
        assert result.version >= 2  # admitted; no stale_read_generation rejection
        assert result.converged is False
    finally:
        stop_coordinator(tmp_path)


# --- no-op: byte-identical commit mints no phantom advance ------------------


def test_noop_commit_touches_neither_leg(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Committing the exact bytes last observed advances NOTHING — no substrate
    write, no coordinator bump — so a byte-identical rewrite never invalidates a
    peer (Open Q C)."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    posts: list[str] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        posts.append(path)
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        _bytes, tok = a.read(REF)
        result = a.commit(REF, expected_token=tok, new_bytes=b"v1")  # identical

        assert result.noop is True
        assert result.summary == "unchanged"
        assert fake_a.cas_calls == []  # no substrate write
        assert "/hooks/post-edit-cas" not in posts  # no coordinator bump
    finally:
        stop_coordinator(tmp_path)


# --- re-drive: retry outcomes (the honesty boundary) ------------------------


def test_re_drive_retry_conflict_is_typed_conflict(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """RE_DRIVE whose retry loses to a peer (token still unmoved for me) surfaces
    the typed conflict after EXACTLY two substrate writes — never a third, never a
    blind win."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        _ab, a_tok = a.read(REF)

        fake_a.script_unknown(landed=False)  # 1st: unknown, not landed → RE_DRIVE
        fake_a.script_conflict()  # 2nd (the re-drive): conflict → typed conflict
        with pytest.raises(CasVersionConflict):
            a.commit(REF, expected_token=a_tok, new_bytes=b"v2")
        assert len(fake_a.cas_calls) == 2
    finally:
        stop_coordinator(tmp_path)


def test_re_drive_retry_second_unknown_is_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A SECOND unknown on the re-drive fails closed (CommitUnconfirmed) — it does
    NOT loop unbounded and NEVER bumps the coordinator."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    posts: list[str] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        posts.append(path)
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        _ab, a_tok = a.read(REF)

        fake_a.script_unknown(landed=False)  # 1st: unknown → RE_DRIVE
        fake_a.script_unknown(landed=False)  # 2nd: unknown again → fail-closed
        with pytest.raises(CommitUnconfirmed):
            a.commit(REF, expected_token=a_tok, new_bytes=b"v2")
        assert len(fake_a.cas_calls) == 2  # no third attempt
        assert "/hooks/post-edit-cas" not in posts  # never bumped
    finally:
        stop_coordinator(tmp_path)


def test_re_drive_detects_own_ghost_and_converges(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """RE_DRIVE whose retry conflicts because MY OWN in-flight ghost put landed
    converges (my bytes are present) and drives the bump — not a misleading peer
    conflict."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        _ab, a_tok = a.read(REF)

        fake_a.script_unknown(landed=False)  # 1st: unknown, not landed → RE_DRIVE
        fake_a.script_ghost_conflict()  # 2nd: ghost lands my bytes, THEN conflicts
        result = a.commit(REF, expected_token=a_tok, new_bytes=b"v2")

        assert result.converged is True  # my ghost carried my bytes
        assert len(fake_a.cas_calls) == 2
        assert store.get(REF)[0] == b"v2"
    finally:
        stop_coordinator(tmp_path)


# --- HOLD: absent operand wedges the view, never bumps ----------------------


def test_hold_verdict_wedges_without_bump(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown write whose operand is ABSENT at reconcile (a raced delete)
    HOLDs → ViewWedged, and NEVER fires the coordinator bump (no phantom advance
    on a deleted operand)."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    posts: list[str] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        posts.append(path)
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        _ab, a_tok = a.read(REF)

        fake_a.script_unknown(landed=False)
        fake_a.reconcile_hook = lambda: store.delete(REF)  # operand vanishes
        with pytest.raises(ViewWedged):
            a.commit(REF, expected_token=a_tok, new_bytes=b"v2")
        assert len(fake_a.cas_calls) == 1
        assert "/hooks/post-edit-cas" not in posts  # HOLD never bumps
    finally:
        stop_coordinator(tmp_path)


# --- converged/clean bump losing to a peer → typed conflict -----------------


def test_converged_bump_conflict_on_different_peer_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A converged write whose bump loses to a DIFFERENT-bytes peer is a real
    conflict (the coordinator hash does not match my intended) — NOT a false
    converge that would mask a lost update."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        b, _fb = _agent(store, sb, arm="object")
        _ab, a_tok = a.read(REF)
        _bb, _b_tok = b.read(REF)  # B is SHARED@v1 so it can bump the coordinator

        def peer_bumps_different() -> None:
            sb.commit_cas(REF, expected_version=1, content_hash=_sha256(b"peer-different"))

        fake_a.script_unknown(landed=True)  # A's write landed (b"v2") → CONVERGE
        fake_a.reconcile_hook = peer_bumps_different
        with pytest.raises(CasVersionConflict):
            a.commit(REF, expected_token=a_tok, new_bytes=b"v2")
        assert len(fake_a.cas_calls) == 1  # no re-drive of a landed write
    finally:
        stop_coordinator(tmp_path)


def test_clean_win_bump_conflict_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clean substrate win whose coordinator bump loses to a peer (a peer bumped
    between this agent's pre-read and its bump) surfaces the typed conflict."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    injected = {"done": False}
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        # Just before A's bump, let a peer bump the coordinator once (v1 → v2), so
        # A's clean-win bump at expected_version=1 conflicts. The guard keeps the
        # peer's own post-edit-cas from re-triggering the injection.
        if (
            path == "/hooks/post-edit-cas"
            and payload.get("session_id") == sa.session_id
            and not injected["done"]
        ):
            injected["done"] = True
            sb.commit_cas(REF, expected_version=1, content_hash=_sha256(b"peer"))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    try:
        a, fake_a = _agent(store, sa, arm="object")
        b, _fb = _agent(store, sb, arm="object")
        _ab, a_tok = a.read(REF)
        _bb, _b_tok = b.read(REF)  # B SHARED@v1 so its injected bump lands

        with pytest.raises(CasVersionConflict):
            a.commit(REF, expected_token=a_tok, new_bytes=b"vA")
        assert len(fake_a.cas_calls) == 1  # a clean win, no reconcile/re-drive
    finally:
        stop_coordinator(tmp_path)


# --- never-ship-a-store, made load-bearing at composition -------------------


def test_binding_declaring_it_sends_content_is_refused() -> None:
    """A binding that declares SENDS_CONTENT_TO_COORDINATOR=True is refused at
    composition — the never-ship-a-store floor is enforced, not merely declared."""

    class _ContentLeakingBinding:
        SENDS_CONTENT_TO_COORDINATOR = True

    with pytest.raises(CoherenceError):
        CoordinatedSubstrate(_ContentLeakingBinding(), object())  # type: ignore[arg-type]


# --- coordinator commit classification (fail-closed) ------------------------


def test_classify_commit_ok_is_win() -> None:
    result = substrate_module._classify_commit({"ok": True, "version": 5}, expected_version=4)
    assert isinstance(result, CoordinatorWin) and result.version == 5


def test_classify_commit_degraded_body_is_unconfirmed() -> None:
    with pytest.raises(CommitUnconfirmed):
        substrate_module._classify_commit({"ok": False, "degraded": True}, expected_version=4)


def test_classify_commit_unconfirmed_reason_is_unconfirmed() -> None:
    with pytest.raises(CommitUnconfirmed):
        substrate_module._classify_commit(
            {"ok": False, "reason": COMMIT_UNCONFIRMED_REASON}, expected_version=4
        )


def test_classify_commit_retryable_reason_is_conflict() -> None:
    result = substrate_module._classify_commit(
        {"ok": False, "reason": VERSION_MISMATCH_REASON, "current_version": 7},
        expected_version=4,
    )
    assert isinstance(result, CoordinatorConflict) and result.current_version == 7


def test_classify_commit_unknown_reason_fails_closed() -> None:
    with pytest.raises(CoherenceError):
        substrate_module._classify_commit({"ok": False, "reason": "mystery"}, expected_version=4)


# A non-win answer whose reason is not a string says nothing this client can
# act on — it is not a retryable conflict and not a rejection it can name — so
# its outcome is unknown. The real coordinator always sends a string reason;
# these are what a proxy or a gateway in front of it could send.
_UNRECOGNISABLE_REASONS = pytest.mark.parametrize(
    "body",
    [
        {"ok": False, "reason": [VERSION_MISMATCH_REASON]},
        {"ok": False, "reason": {"reason": VERSION_MISMATCH_REASON}},
        {"ok": False, "reason": 7},
        {"ok": False},
    ],
    ids=["list", "object", "number", "absent"],
)


@_UNRECOGNISABLE_REASONS
def test_classify_commit_an_unrecognisable_reason_is_unconfirmed(body: dict) -> None:
    """Classified defensively: an unhashable reason is not a ``TypeError``
    from the membership test, and no reason that is not a string reads as a
    conflict or as a named rejection."""
    with pytest.raises(CommitUnconfirmed) as raised:
        substrate_module._classify_commit(body, expected_version=4)

    assert VERSION_MISMATCH_REASON not in str(raised.value)


@_UNRECOGNISABLE_REASONS
def test_an_unrecognisable_bump_answer_after_the_substrate_write_is_the_bump_legs_unknown(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, body: dict,
) -> None:
    """The substrate CAS lands; the coordinator's answer to the bump is not a
    win and carries no reason this client can classify. The bump's outcome
    is unknown, so it takes the existing unknown path — ``CommitUnconfirmed``
    (re-read; retry only if absent) — and is never re-driven. Before, a list
    or object reason escaped as a ``TypeError`` after the write had landed."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    real_post = substrate_module._coordinator_post

    def unrecognisable_bump(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if path == "/hooks/post-edit-cas":
            return dict(body)
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", unrecognisable_bump)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        _bytes, tok = a.read(REF)

        with pytest.raises(CommitUnconfirmed):
            a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert store.get(REF)[0] == b"v2", "control: the substrate write landed"
        assert len(fake_a.cas_calls) == 1, "a landed write is never re-driven"
    finally:
        stop_coordinator(tmp_path)


class _Redirector(http.server.BaseHTTPRequestHandler):
    """Answers every POST with a redirect to ``location`` — which the test
    fills with the session's nonce and principal, as a coordinator (or a
    proxy in front of it) echoing what it was sent would. Never followed:
    the client refuses every 3xx before a second request is made."""

    code = 302
    location = ""
    seen: list[str] = []

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        type(self).seen.append(self.path)
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(type(self).code)
        self.send_header("Location", type(self).location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        return


@pytest.mark.parametrize(
    ("leg", "code", "principal"),
    [
        *[("bump", code, "presented") for code in (301, 302, 303, 307, 308)],
        ("bump", 302, "none"),
        ("pre-read", 302, "none"),
    ],
    ids=lambda value: str(value),
)
def test_a_redirected_bump_after_the_substrate_write_is_the_bump_legs_unknown(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    leg: str, code: int, principal: str,
) -> None:
    """The substrate CAS lands; the coordinator bump is answered with a
    redirect, refused and never followed. Whether the bump landed is not
    known — what answered may have passed it on — and the substrate already
    holds the new bytes, so it is the bump leg's unknown like every other
    failure there: ``CommitUnconfirmed`` (re-read; retry only if absent),
    never re-driven. Before, the typed ``RedirectRefused`` escaped after the
    write had landed, reading as a refusal that changed nothing.

    What is reported is the status: the ``Location`` — the coordinator's
    text, echoing the nonce and principal here — is on neither the message
    nor the chain, including when no principal was presented (``none``),
    where the refusal once quoted it. ``pre-read`` is the control leg: redirected
    before the substrate is touched, the commit is refused with nothing
    written — not the unknown."""
    from ccs.cli._coherence_client import CoordinatorEndpoint

    store = _FakeStore()
    store.seed(REF, b"v1")
    _Redirector.code, _Redirector.seen = code, []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Redirector)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    real_post = substrate_module._coordinator_post
    redirected = "/hooks/post-edit-cas" if leg == "bump" else "/hooks/pre-read"
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        _bytes, tok = a.read(REF)
        material = [sa._mint_nonce, sa._principal]
        if principal == "none":
            sa._principal = None  # the request presents no header (the refusal once named the Location then)
        _Redirector.location = f"http://127.0.0.1:1/{''.join(material)}"
        redirector = CoordinatorEndpoint(port=httpd.server_address[1], bearer=sa._endpoint.bearer)

        def redirect_one_leg(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
            return real_post(redirector if path == redirected else endpoint, path, payload, **kwargs)

        monkeypatch.setattr(substrate_module, "_coordinator_post", redirect_one_leg)
        with pytest.raises(CoherenceError) as raised:
            a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert _Redirector.seen == [redirected], "control: that leg was answered with the redirect"
        if leg == "bump":
            assert isinstance(raised.value, CommitUnconfirmed), type(raised.value)
            assert store.get(REF)[0] == b"v2", "control: the substrate write landed"
            assert len(fake_a.cas_calls) == 1, "a landed write is never re-driven"
        else:
            assert not isinstance(raised.value, CommitUnconfirmed), "nothing landed to be unknown"
            assert store.get(REF)[0] == b"v1" and fake_a.cas_calls == []
        assert f"HTTP {code}" in str(raised.value)
        rendered = "".join(traceback.format_exception(raised.value))
        assert all(isinstance(m, str) and m for m in material), "control: real values to look for"
        assert not [m for m in material if m in rendered], "a nonce or principal reached the error"
        assert "127.0.0.1:1" not in rendered, "the Location reached the error"
    finally:
        httpd.shutdown()
        httpd.server_close()
        stop_coordinator(tmp_path)


# --- a converged write whose completeness check fails, after it landed --------

# FROZEN duplicates of what a commit reports when the coordinator leg fails
# definitely after the substrate write is on the substrate — never built from
# the code. A confirmed write "landed"; a converged one claims no attribution.
_LANDED = "the substrate write to {ref!r} landed"
_CONVERGED = (
    "the substrate holds the bytes this write intended for {ref!r} (converged: "
    "which write put them there is not claimed)"
)
_PRINCIPAL_REFUSED_AFTER = (
    "{landed}, but the coordinator refused the caller principal of a request "
    "this commit sent ({reason}), so it recorded no bump from this write, and "
    "this write invalidated no peer. Not re-driven; re-read before retrying."
)
_TLS_FAILED_AFTER = (
    "{landed}, but a request this commit made to the coordinator failed TLS "
    "certificate verification, so it was never sent, and the coordinator "
    "recorded no bump from this write. Not re-driven; re-read before retrying."
)


def _refusing_port() -> int:
    """A loopback port that refuses connections and that no other test can take.

    A port freed after ``bind(("127.0.0.1", 0))`` can be handed to another
    test worker's port-0 bind before the connect, and the connect then reaches
    a stranger. Ports below 1024 are never handed out for port 0, and port 4
    is unassigned, so no standard service listens on it; the assert says so
    if something does. A socket bound without listening is no substitute: on
    macOS a connect to it hangs until its timeout instead of being refused."""
    import errno
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", 4)) == errno.ECONNREFUSED, "port 4 accepts connections here"
    return 4


@pytest.mark.parametrize(
    "failure",
    ["redirect-presented", "redirect-none", "unreachable", "http-503", "degraded", "principal-refused"],
)
def test_a_failed_completeness_check_of_a_converged_write_is_the_bump_legs_unknown(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """The substrate write lands under an unknown outcome and reconciles to
    CONVERGE. A byte-identical peer has already bumped the coordinator, so
    this writer's bump conflicts, and the pre-read that decides whether the
    converged write is complete — does the coordinator hold its hash? —
    fails. The substrate write has landed, so any failure there is the bump
    leg's unknown: ``CommitUnconfirmed`` (re-read; retry only if absent),
    never re-driven — as a principal refusal of the same request already
    was. Before, a redirect, an unreachable coordinator, a 5xx and a
    degraded answer each raised a plain ``CoherenceError`` after the write
    had landed.

    What a refusal reports is checked against state: the version DID
    advance here — the peer's identical bump — so the message names what
    this write did not do, not the version; and the write converged, so it
    claims no attribution for the bytes on the substrate. No nonce,
    principal or ``Location`` reaches the error."""
    from ccs.cli._coherence_client import CoordinatorEndpoint

    store = _FakeStore()
    store.seed(REF, b"v1")
    _Redirector.code, _Redirector.seen = (503 if failure == "http-503" else 302), []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Redirector)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    real_post = substrate_module._coordinator_post
    sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        b, _fb = _agent(store, sb)
        _ab, tok = a.read(REF)
        b.read(REF)  # B is SHARED@v1 so it can bump the coordinator
        material = [sa._mint_nonce, sa._principal]
        _Redirector.location = f"http://127.0.0.1:1/{''.join(material)}"
        answering = CoordinatorEndpoint(
            port=_refusing_port() if failure == "unreachable" else httpd.server_address[1],
            bearer=sa._endpoint.bearer,
        )
        fake_a.script_unknown(landed=True)  # A's substrate write landed (b"v2")
        fake_a.reconcile_hook = lambda: sb.commit_cas(REF, expected_version=1, content_hash=_sha256(b"v2"))
        sent: list[str] = []

        def fail_the_check(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
            if payload.get("session_id") != sa.session_id:
                return real_post(endpoint, path, payload, **kwargs)
            if path == "/hooks/post-edit-cas" or not sent:
                sent.extend([path] if path == "/hooks/post-edit-cas" else [])
                return real_post(endpoint, path, payload, **kwargs)
            sent.append(path)
            if failure == "degraded":
                return {"status": "fresh", "version": 2, "degraded": True}
            if failure == "principal-refused":
                return real_post(endpoint, path, payload, extra_headers={_PRINCIPAL_HEADER: "X" * 43})
            if failure == "redirect-none":
                kwargs["extra_headers"] = None
            return real_post(answering, path, payload, **kwargs)

        monkeypatch.setattr(substrate_module, "_coordinator_post", fail_the_check)
        with pytest.raises(CommitUnconfirmed) as raised:
            a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert sent[:2] == ["/hooks/post-edit-cas", "/hooks/pre-read"], "control: the bump, then the check"
        assert store.get(REF)[0] == b"v2", "control: the substrate write landed"
        assert len(fake_a.cas_calls) == 1, "a landed write is never re-driven"
        if failure.startswith("redirect") or failure == "http-503":
            assert _Redirector.seen == ["/hooks/pre-read"], "control: the check was answered there"
        if failure == "principal-refused":
            assert str(raised.value) == _PRINCIPAL_REFUSED_AFTER.format(
                landed=_CONVERGED.format(ref=REF), reason="caller_principal_foreign"
            )
            observer = _session(tmp_path, fast_cfg)
            assert observer.pre_read(REF, None).version == 2, "the peer's identical bump advanced it"
        rendered = "".join(traceback.format_exception(raised.value))
        assert all(isinstance(m, str) and m for m in material), "control: real values to look for"
        assert not [m for m in material if m in rendered], "a nonce or principal reached the error"
        assert "127.0.0.1:1/" not in rendered, "the Location reached the error"
    finally:
        httpd.shutdown()
        httpd.server_close()
        stop_coordinator(tmp_path)


@requires_openssl
@pytest.mark.parametrize("leg", ["bump", "pre-read"])
def test_a_tls_failure_on_the_bump_after_the_substrate_write_is_the_bump_legs_unknown(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, leg: str,
) -> None:
    """One leg's request goes to an ``https://`` endpoint whose certificate
    does not verify against the system trust store, which fails that request
    as ``TlsVerificationFailed`` before it is sent. On the bump, after the
    substrate write landed, that is the bump leg's unknown —
    ``CommitUnconfirmed``, never re-driven — whose message says what
    happened; the verification failure is its cause. Before, the
    ``TlsVerificationFailed`` escaped after the write had landed.
    ``pre-read`` is the control leg: failed before the substrate is touched,
    the trust refusal stands, with nothing written. (This used to reach TLS
    through an ``https://`` proxy in the environment; coordinator requests
    now skip any configured proxy, so the endpoint itself is https here.)"""
    from ccs.core.exceptions import TlsVerificationFailed

    (tmp_path / "certs").mkdir()
    tls = _start_tls_server(_mint_bundle(tmp_path / "certs"), _make_handler_class())
    store = _FakeStore()
    store.seed(REF, b"v1")
    real_post = substrate_module._coordinator_post
    over_tls = "/hooks/post-edit-cas" if leg == "bump" else "/hooks/pre-read"
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        _bytes, tok = a.read(REF)

        def to_an_unverifiable_endpoint(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
            if path != over_tls:
                return real_post(endpoint, path, payload, **kwargs)
            # The test CA is in no system trust store, and no CA file is named.
            unverifiable = dataclasses.replace(
                endpoint, scheme="https", port=tls.port, ca_file=None
            )
            return real_post(unverifiable, path, payload, **kwargs)

        monkeypatch.setattr(substrate_module, "_coordinator_post", to_an_unverifiable_endpoint)
        with pytest.raises(CoherenceError) as raised:
            a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert tls.handler_cls.seen_authorizations == [], "nothing was sent past the handshake"
        if leg == "bump":
            assert isinstance(raised.value, CommitUnconfirmed), type(raised.value)
            assert str(raised.value) == _TLS_FAILED_AFTER.format(landed=_LANDED.format(ref=REF))
            assert isinstance(raised.value.__cause__, TlsVerificationFailed)
            assert store.get(REF)[0] == b"v2", "control: the substrate write landed"
            assert len(fake_a.cas_calls) == 1, "a landed write is never re-driven"
        else:
            assert isinstance(raised.value, TlsVerificationFailed), type(raised.value)
            assert store.get(REF)[0] == b"v1" and fake_a.cas_calls == []
    finally:
        tls.shutdown()
        stop_coordinator(tmp_path)


# --- caller principal (caller-principal plan, U5) ----------------------------
#
# The substrate session is a long-lived caller holding its principal in memory.
# Unlike CoherentVolume (whose re-mint keeps the session, KTD14), its
# reacquire() mints a NEW session id — so a reacquire is a new identity and
# claims its own principal; the commit route requires it.

_PRINCIPAL_HEADER = "Coherence-Caller-Principal"  # frozen duplicate of the wire name


def test_the_session_presents_its_principal_and_a_reacquire_claims_a_new_one(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every request presents the principal bound to the CURRENT session id —
    the commit route refuses one that does not — and a reacquire, which mints
    a new session, claims a new principal rather than reusing the old one
    (which would be foreign to the new session and refused). Commits land
    before and after the reacquire."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    seen: list[tuple[str, str, object]] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        headers = kwargs.get("extra_headers") or {}
        seen.append((path, payload["session_id"], headers.get(_PRINCIPAL_HEADER)))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(substrate_module, "_coordinator_post", spy)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        _bytes, tok = a.read(REF)
        a.commit(REF, expected_token=tok, new_bytes=b"v2")
        first = sa._principal
        a.reacquire(REF)
        _bytes, tok = a.read(REF)
        a.commit(REF, expected_token=tok, new_bytes=b"v3")
        second = sa._principal

        assert first and second and first != second
        by_session: dict[str, set] = {}
        for _path, sid, principal in seen:
            by_session.setdefault(sid, set()).add(principal)
        assert len(by_session) == 2
        assert sorted(map(frozenset, by_session.values()), key=str) == sorted(
            [frozenset({first}), frozenset({second})], key=str
        )
        assert store.get(REF)[0] == b"v3"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("outcome", ["refused", "unconfirmed"])
def test_a_claim_that_does_not_bind_fails_closed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """Fail-closed like every other coordinator failure on this client: a claim
    that did not bind raises at construction instead of running a session
    whose commits the coordinator would refuse."""
    from ccs.cli._coherence_client import PrincipalClaim

    monkeypatch.setattr(
        substrate_module, "claim_caller_principal",
        lambda *_a: PrincipalClaim(outcome, detail="simulated"),  # type: ignore[arg-type]
        raising=False,
    )
    try:
        with pytest.raises(CoherenceError, match="caller principal"):
            _session(tmp_path, fast_cfg)
    finally:
        stop_coordinator(tmp_path)


def _record_claims(monkeypatch: pytest.MonkeyPatch, nonces: list[str]) -> None:
    """Record the nonce every claim presents, forwarding it to the real claim."""
    real = substrate_module.claim_caller_principal

    def spy(endpoint, session_id, nonce):  # noqa: ANN001, ANN202
        nonces.append(nonce)
        return real(endpoint, session_id, nonce)

    monkeypatch.setattr(substrate_module, "claim_caller_principal", spy)


def test_a_reacquire_whose_claim_fails_keeps_the_previous_session_and_principal(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``reacquire`` claims for the NEW session id before adopting it: a claim
    that fails (unconfirmed — a transport blip, a watchdog-degraded claim)
    raises and leaves the previous session id, principal and nonce paired as
    they were, so the session keeps working. Before, the id was swapped first
    and the object was left naming the new session with the old session's
    principal — every later call refused as foreign until another reacquire."""
    from ccs.cli._coherence_client import PrincipalClaim

    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        _bytes, tok = a.read(REF)
        before = (sa.session_id, sa._principal, getattr(sa, "_mint_nonce", None))
        real = substrate_module.claim_caller_principal
        monkeypatch.setattr(
            substrate_module, "claim_caller_principal",
            lambda *_a: PrincipalClaim("unconfirmed", detail="simulated"),
        )
        with pytest.raises(CoherenceError, match="caller principal"):
            sa.reacquire()
        monkeypatch.setattr(substrate_module, "claim_caller_principal", real)

        assert (sa.session_id, sa._principal, getattr(sa, "_mint_nonce", None)) == before
        result = a.commit(REF, expected_token=tok, new_bytes=b"v2")
        assert result.version == 2 and store.get(REF)[0] == b"v2"
    finally:
        stop_coordinator(tmp_path)


def test_a_refused_principal_is_re_claimed_with_the_retained_nonce_and_retried_once(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A principal the coordinator does not hold for this session (as after
    its binding store was reset) is refused as foreign; the session claims
    again with the nonce it RETAINED from its own claim — never a new one —
    adopts the principal returned, and retries the refused request once. The
    read and the commit both land; nothing logs the principal or the nonce."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    nonces: list[str] = []
    _record_claims(monkeypatch, nonces)
    caplog.set_level(logging.DEBUG)
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        bound = sa._principal
        sa._principal = "X" * 43

        _bytes, tok = a.read(REF)
        result = a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert result.version == 2
        assert len(nonces) == 2 and len(set(nonces)) == 1
        assert sa._principal == bound
        assert bound and bound not in caplog.text and nonces[0] not in caplog.text
    finally:
        stop_coordinator(tmp_path)


def test_a_definite_principal_refusal_is_typed_on_both_legs_never_commit_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the claim with the held nonce cannot cure the refusal (the session
    is bound under another nonce), the refusal surfaces as
    :class:`CallerPrincipalRefused` with the wire reason — on the read leg
    AND the commit leg. A principal refusal is definite (the coordinator
    committed nothing), so the commit leg must not call it
    ``CommitUnconfirmed``, which sends the caller into unknown-outcome
    reconciliation. Neither message carries a principal or a nonce."""
    from ccs.core.exceptions import CallerPrincipalRefused

    sa = _session(tmp_path, fast_cfg)
    try:
        bound = sa._principal
        sa._mint_nonce, sa._principal = "Z" * 43, "X" * 43
        with pytest.raises(CallerPrincipalRefused) as read_leg:
            sa.pre_read(REF, None)
        with pytest.raises(CallerPrincipalRefused) as commit_leg:
            sa.commit_cas(REF, expected_version=1, content_hash="a" * 64)

        for refusal in (read_leg.value, commit_leg.value):
            assert refusal.reason == "caller_principal_foreign"
            assert refusal.settled is True, "bound under another nonce never changes"
            assert not isinstance(refusal, CommitUnconfirmed)
            for secret in (bound, "Z" * 43, "X" * 43):
                assert secret and secret not in str(refusal)
    finally:
        stop_coordinator(tmp_path)


def test_a_refusal_whose_recovery_claim_is_unconfirmed_is_not_settled_and_the_next_request_claims_again(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The session presents a principal that is not the bound one; the request
    is refused and the recovery claim's answer is lost (a transport blip).
    The refusal raises typed but NOT settled: nothing was adopted from an
    unconfirmed claim, and the next request — refused again — claims again
    with the RETAINED nonce by itself, adopts the bound principal and lands.
    A refusal the claim answered (the bound-under-another-nonce arm, pinned
    above) is settled. Nothing rendered carries a principal or the nonce."""
    from ccs.cli._coherence_client import PrincipalClaim
    from ccs.core.exceptions import CallerPrincipalRefused

    caplog.set_level(logging.DEBUG)
    sa = _session(tmp_path, fast_cfg)
    try:
        bound, nonce = sa._principal, sa._mint_nonce
        assert bound and nonce, "control: the session claimed at construction"
        sa._principal = "X" * 43
        real = substrate_module.claim_caller_principal
        nonces: list[str] = []

        def lossy(endpoint, session_id, presented_nonce):  # noqa: ANN001, ANN202
            nonces.append(presented_nonce)
            claim = real(endpoint, session_id, presented_nonce)
            assert claim.outcome == "bound", "control: the lost claim really answered"
            if len(nonces) == 1:
                return PrincipalClaim("unconfirmed", detail="answer lost")
            return claim

        monkeypatch.setattr(substrate_module, "claim_caller_principal", lossy)

        with pytest.raises(CallerPrincipalRefused) as raised:
            sa.pre_read(REF, None)

        assert raised.value.reason == "caller_principal_foreign"
        assert raised.value.settled is False, "an unconfirmed recovery claim is not a settled refusal"
        assert sa._principal == "X" * 43, "a principal was adopted from an unconfirmed claim"
        assert nonces == [nonce], "the recovery claim presented the retained nonce"

        result = sa.pre_read(REF, None)

        assert result.version >= 0 and result.stale_denied is False
        assert nonces == [nonce, nonce], "the next request claimed again by itself, same nonce"
        assert sa._principal == bound
        rendered = str(raised.value) + caplog.text
        for secret in (bound, nonce, "X" * 43):
            assert secret not in rendered, "a principal or nonce was rendered"
    finally:
        stop_coordinator(tmp_path)


# --- a principal refusal of the bump, after the substrate write landed ---------


def test_a_bump_refused_for_its_principal_after_the_substrate_write_is_the_bump_legs_unknown(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The substrate CAS lands; then the coordinator refuses the bump for its
    caller principal and recovery cannot cure it (the session is bound under
    a nonce this client no longer holds). The refusal itself is definite —
    the coordinator recorded NOTHING of this write — but the commit is not:
    the substrate holds the new bytes while the coordinator is behind them,
    the very state an unconfirmed bump leaves. So it goes through the bump
    leg's existing handling and surfaces as ``CommitUnconfirmed`` (re-read;
    retry only if absent; never re-drive), whose message says the substrate
    write landed and the coordinator recorded no bump from it, with the typed
    refusal as its cause.

    Raised bare, the refusal would invite its own recovery — a refused request
    changed nothing, so resend it once the principal is right — and that
    re-drives a commit whose substrate write already landed: the last arm
    shows the resend reporting this writer's own write as a conflict."""
    from ccs.core.exceptions import CallerPrincipalRefused

    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, fake_a = _agent(store, sa)
        _bytes, tok = a.read(REF)
        bound, nonce = sa._principal, sa._mint_nonce
        # No principal, and a nonce the session was not bound with: the read
        # leg (accept class) is admitted, the bump (require class) is refused
        # as absent, and claiming again with this nonce is refused as claimed.
        sa._principal, sa._mint_nonce = None, "Z" * 43

        with pytest.raises(CommitUnconfirmed) as raised:
            a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert not isinstance(raised.value, CallerPrincipalRefused)
        cause = raised.value.__cause__
        assert isinstance(cause, CallerPrincipalRefused)
        assert cause.reason == "caller_principal_absent"
        message = str(raised.value)
        assert message == _PRINCIPAL_REFUSED_AFTER.format(
            landed=_LANDED.format(ref=REF), reason="caller_principal_absent"
        )
        for secret in (bound, nonce, "Z" * 43):
            assert secret and secret not in message
        assert store.get(REF)[0] == b"v2", "the substrate write is durable"
        assert len(fake_a.cas_calls) == 1, "a landed write is never re-driven"
        observer = _session(tmp_path, fast_cfg)
        assert observer.pre_read(REF, None).version == 1, "no bump was recorded from this write"

        sa._principal, sa._mint_nonce = bound, nonce
        with pytest.raises(CasVersionConflict):
            a.commit(REF, expected_token=tok, new_bytes=b"v2")
    finally:
        stop_coordinator(tmp_path)


def test_a_reacquire_adopts_the_nonce_its_claim_presented_with_the_session_id(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A successful reacquire adopts the new session id, its principal AND the
    nonce its claim presented — together. The nonce is what recovery claims
    again with: one kept from the previous session would be presented for a
    session it never bound, refused as ``caller_principal_claimed``, and the
    session would stay refused for good once its store was reset. Here the
    principal is made stale after the reacquire (as a reset leaves it) and the
    next request recovers with the reacquire's nonce and lands."""
    claims: list[tuple[str, str]] = []
    real = substrate_module.claim_caller_principal

    def spy(endpoint, session_id, nonce):  # noqa: ANN001, ANN202
        claims.append((session_id, nonce))
        return real(endpoint, session_id, nonce)

    monkeypatch.setattr(substrate_module, "claim_caller_principal", spy)
    store = _FakeStore()
    store.seed(REF, b"v1")
    sa = _session(tmp_path, fast_cfg)
    try:
        a, _fa = _agent(store, sa)
        _first_session, first_nonce = claims[0]
        a.reacquire(REF)
        new_session, new_nonce = claims[1]
        assert new_nonce != first_nonce
        assert (sa.session_id, sa._mint_nonce) == (new_session, new_nonce)

        sa._principal = "X" * 43
        _bytes, tok = a.read(REF)
        result = a.commit(REF, expected_token=tok, new_bytes=b"v2")

        assert result.version == 2 and store.get(REF)[0] == b"v2"
        assert claims[2:] == [(new_session, new_nonce)]
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("scenario", ["the_claim_echoes", "the_retry_echoes"])
def test_no_coordinator_supplied_text_reaches_what_the_session_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, scenario: str
) -> None:
    """A coordinator that echoes the nonce and principal it was sent in every
    free-text field — a claim's ``reason``/``detail``/``error``, a refusal's
    ``error``/``detail``, the status line's reason phrase — cannot get either
    into what the session raises on its claim and recovery paths: only a
    reason from the frozen vocabulary is repeated (anything else reads
    ``unrecognised``), and a failed request is reported by its status code."""
    import io
    import json
    import urllib.error

    from ccs.cli import _coherence_client

    sa = _session(tmp_path, fast_cfg)
    try:
        adopted = "Q" * 43
        held = (sa._principal, sa._mint_nonce, adopted)
        echo = " ".join(held)  # type: ignore[arg-type]
        real_claim_post = _coherence_client.post

        def claim_echoes(endpoint, path, body, **kwargs):  # noqa: ANN001, ANN003, ANN202
            if path != "/principal/claim":
                return real_claim_post(endpoint, path, body, **kwargs)
            if scenario == "the_retry_echoes":
                return {"ok": True, "principal": adopted}
            return {"ok": False, "reason": echo, "detail": echo, "error": echo}

        monkeypatch.setattr(_coherence_client, "post", claim_echoes)

        def error(status: int, body: dict) -> urllib.error.HTTPError:
            raw = io.BytesIO(json.dumps(body).encode())
            return urllib.error.HTTPError("/hooks/pre-read", status, echo, {}, raw)  # type: ignore[arg-type]

        answers = [
            error(400, {"error": echo, "detail": echo, "reason": "caller_principal_foreign"}),
            error(500, {"error": echo, "detail": echo}),
        ]
        real_post = substrate_module._coordinator_post

        def answering(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
            if path == "/hooks/pre-read":
                raise answers.pop(0)
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(substrate_module, "_coordinator_post", answering)

        reported: list[str] = []
        with pytest.raises(CoherenceError) as raised:
            sa.pre_read(REF, None)
        reported.append(f"{type(raised.value).__name__}: {raised.value}")
        if scenario == "the_claim_echoes":
            with pytest.raises(CoherenceError) as raised_again:
                sa.reacquire()
            reported.append(f"{type(raised_again.value).__name__}: {raised_again.value}")
            assert all("unrecognised" in r for r in reported), reported
        else:
            assert "HTTP 500" in reported[0], reported
        for secret in held:
            assert secret and secret not in " ".join(reported)
    finally:
        stop_coordinator(tmp_path)


# --- the retry bound and the reason a second refusal carries -------------------


def _refuse_pre_read(
    monkeypatch: pytest.MonkeyPatch, reasons: list[str], sent: list[str | None]
) -> None:
    """Every ``/hooks/pre-read`` is refused for its principal, the n-th with
    ``reasons[n % len(reasons)]``; records the principal each one presented.
    Every other request goes to the coordinator."""
    import io
    import json
    import urllib.error

    real_post = substrate_module._coordinator_post

    def refusing(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if path != "/hooks/pre-read":
            return real_post(endpoint, path, payload, **kwargs)
        sent.append((kwargs.get("extra_headers") or {}).get(_PRINCIPAL_HEADER))
        reason = reasons[(len(sent) - 1) % len(reasons)]
        body = io.BytesIO(json.dumps({"error": "refused", "reason": reason}).encode())
        raise urllib.error.HTTPError(path, 400, "Bad Request", {}, body)  # type: ignore[arg-type]

    monkeypatch.setattr(substrate_module, "_coordinator_post", refusing)


def test_a_request_refused_again_after_recovery_is_retried_exactly_once(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The substrate session's twin of the volume's bound: refused as
    ``caller_principal_absent``, recovered (the claim with the held nonce
    returns a principal that differs from the one presented), and refused
    AGAIN as ``caller_principal_foreign`` — the request is retried exactly
    ONCE after exactly ONE claim, and the typed refusal carries the SECOND
    refusal's reason and says it was refused again. A session that kept
    re-claiming would turn one refused request into an unbounded loop."""
    from ccs.cli._coherence_client import PRINCIPAL_REFUSED_AGAIN, PrincipalClaim
    from ccs.core.exceptions import CallerPrincipalRefused

    sa = _session(tmp_path, fast_cfg)
    try:
        claims: list[str] = []

        def always_a_new_principal(_endpoint, _sid, nonce):  # noqa: ANN001, ANN202
            claims.append(nonce)
            return PrincipalClaim("bound", principal=f"{len(claims):043d}")

        monkeypatch.setattr(substrate_module, "claim_caller_principal", always_a_new_principal)
        sent: list[str | None] = []
        _refuse_pre_read(monkeypatch, ["caller_principal_absent", "caller_principal_foreign"], sent)
        held = sa._principal

        with pytest.raises(CallerPrincipalRefused) as raised:
            sa.pre_read(REF, None)

        assert sent == [held, "1".zfill(43)], "the refused request, then ONE retry under the new principal"
        assert claims == [sa._mint_nonce]
        assert raised.value.reason == "caller_principal_foreign"
        assert raised.value.settled is True, "refused again after a claim that answered is settled"
        assert "(caller_principal_foreign)" in str(raised.value)
        assert PRINCIPAL_REFUSED_AGAIN in str(raised.value)
    finally:
        stop_coordinator(tmp_path)


def test_no_coordinator_supplied_text_rides_the_chain_of_what_the_session_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read is refused, recovery adopts a new principal, and the retry is
    answered 500 with the held principal, the nonce and the adopted principal
    as the status line's reason phrase. The raised error names the status
    only — and nothing on its CHAIN carries the phrase, so a traceback prints
    no principal or nonce either."""
    import io
    import json
    import traceback
    import urllib.error

    from ccs.cli import _coherence_client

    sa = _session(tmp_path, fast_cfg)
    try:
        adopted = "Q" * 43
        held = (sa._principal, sa._mint_nonce, adopted)
        echo = " ".join(held)  # type: ignore[arg-type]
        real_claim_post = _coherence_client.post

        def claim_binds(endpoint, path, body, **kwargs):  # noqa: ANN001, ANN003, ANN202
            if path == "/principal/claim":
                return {"ok": True, "principal": adopted}
            return real_claim_post(endpoint, path, body, **kwargs)

        monkeypatch.setattr(_coherence_client, "post", claim_binds)
        statuses = [400, 500]
        real_post = substrate_module._coordinator_post

        def answering(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
            if path != "/hooks/pre-read":
                return real_post(endpoint, path, payload, **kwargs)
            status = statuses.pop(0)
            body = {"reason": "caller_principal_foreign"} if status == 400 else {"error": echo}
            raw = io.BytesIO(json.dumps(body).encode())
            raise urllib.error.HTTPError(path, status, echo, {}, raw)  # type: ignore[arg-type]

        monkeypatch.setattr(substrate_module, "_coordinator_post", answering)

        with pytest.raises(CoherenceError) as raised:
            sa.pre_read(REF, None)

        rendered = "".join(traceback.format_exception(raised.value))
        assert "HTTP 500" in rendered, "control: the retried request was the one answered 500"
        for secret in held:
            assert secret and secret not in rendered
    finally:
        stop_coordinator(tmp_path)


# --- the coordinator answers a pre-read with its failure envelope --------------
#
# The coordinator answers a pre-read it could not decide as HTTP 200 ``{ok:
# false, reason: "internal: <Type>"}``: the caller-principal gate's store read
# raising (a bound session whose binding is not cached, which is what a
# coordinator restart leaves, under a locked registry) or the pre-read work body
# raising. That answer confirms nothing: no version, no deny, no recorded hash.
# The substrate session presents its principal on every request, so a cold
# cache sends its every pre-read through the gate's store read.

#: FROZEN duplicate of the reason the coordinator's failure envelope carries
#: for the registry's own transient error (only the exception's type).
_INTERNAL_OPERATIONAL_ERROR = "internal: OperationalError"


def _serve_in_process(tmp_path: Path, instance_id: str):
    """A strict coordinator for ``workspace/**`` running in THIS process,
    reachable through the usual pid file, so a test can reach into its
    registry. Policy is loaded once at construction, so the YAML is written
    first; a substrate session then attaches to it as to any spawned one."""

    from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer

    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    for name in ("tracked.yaml", "strict_mode.yaml"):
        (coherence / name).write_text("- workspace/**\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id=instance_id)
    server.serve_in_thread()
    (coherence / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
    return server


def _fail_the_next(
    server, monkeypatch: pytest.MonkeyPatch, route: str = "/hooks/pre-read"
) -> None:
    """Make the coordinator answer the next request on ``route`` any session
    sends with its failure envelope, from the gate: the principal cache is
    emptied (a restart) and the one store read that follows raises (a locked
    registry). Every other request, before and after, is answered as usual."""
    import sqlite3

    real_post = substrate_module._coordinator_post
    real_lookup = server.registry.get_caller_principal
    armed = [True]
    lookup_fails = [False]

    def lookup(identity):  # noqa: ANN001, ANN202
        if lookup_fails[0]:
            lookup_fails[0] = False
            raise sqlite3.OperationalError("database is locked")
        return real_lookup(identity)

    def post(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if path == route and armed[0]:
            armed[0] = False
            with server.service._caller_principal_lock:
                server.service._caller_principals.clear()
            lookup_fails[0] = True
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(server.registry, "get_caller_principal", lookup)
    monkeypatch.setattr(substrate_module, "_coordinator_post", post)


def test_a_read_answered_with_the_failure_envelope_fails_closed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read whose pre-read the coordinator answers with its failure envelope
    raises the read leg's infrastructure failure, naming the envelope's
    reason, and records no observed hash. It is not a deny (``StaleView``):
    nothing was decided about this reader's view.

    Prevents the answer passing as a clean pre-read: version 0, no deny, so
    the read returned bytes whose registration nobody confirmed."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    server = _serve_in_process(tmp_path, "substrate-read-envelope")
    try:
        sa = _session(tmp_path, fast_cfg)
        a, _fake = _agent(store, sa)
        _fail_the_next(server, monkeypatch)

        with pytest.raises(CoherenceError) as raised:
            a.read(REF)

        assert type(raised.value) is CoherenceError, type(raised.value)
        assert _INTERNAL_OPERATIONAL_ERROR in str(raised.value)
        assert REF in str(raised.value)
        assert REF not in a._observed_hash, "an observed hash was recorded for an unconfirmed read"
    finally:
        server.shutdown()


def test_a_commit_whose_pre_read_is_answered_with_the_failure_envelope_writes_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A commit whose pre-read the coordinator answers with its failure
    envelope raises before the substrate is touched.

    Prevents a double-applied write: the envelope read as a clean pre-read
    skipped the peer-invalidation deny and seeded the bump with version 0, so
    the substrate write landed, the bump then lost with ``version_mismatch``,
    and the caller got ``CasVersionConflict``, whose text says no write
    landed. A caller that re-merges and retries applies the write twice."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    server = _serve_in_process(tmp_path, "substrate-commit-envelope")
    try:
        sa = _session(tmp_path, fast_cfg)
        a, fake = _agent(store, sa)
        _bytes, token = a.read(REF)
        _fail_the_next(server, monkeypatch)

        with pytest.raises(CoherenceError) as raised:
            a.commit(REF, expected_token=token, new_bytes=b"v2")

        assert not isinstance(raised.value, CasVersionConflict), raised.value
        assert _INTERNAL_OPERATIONAL_ERROR in str(raised.value)
        assert fake.cas_calls == [], "the substrate was written on an unconfirmed pre-read"
        assert store.get(REF)[0] == b"v1"
    finally:
        server.shutdown()


def test_a_converged_commit_that_cannot_check_the_coordinators_hash_is_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A converged write whose bump lost to a peer checks whether the
    coordinator already holds its hash. When that check's pre-read is answered
    with the failure envelope the outcome is unknown, so the commit raises
    ``CommitUnconfirmed``: the bytes have landed and the bump was not proven.

    Prevents a COMPLETE nobody verified: the envelope carries no
    ``hash_differs``, which read as "the hashes match", so the commit reported
    itself converged and complete."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    server = _serve_in_process(tmp_path, "substrate-converge-envelope")
    try:
        sa, sb = _session(tmp_path, fast_cfg), _session(tmp_path, fast_cfg)
        a, fake_a = _agent(store, sa)
        b, _fb = _agent(store, sb)
        _ab, a_tok = a.read(REF)
        _bb, _b_tok = b.read(REF)
        intended = _sha256(b"v2")

        def peer_bumps_first_then_the_check_fails() -> None:
            sb.commit_cas(REF, expected_version=1, content_hash=intended)
            _fail_the_next(server, monkeypatch)

        fake_a.script_unknown(landed=True)
        fake_a.reconcile_hook = peer_bumps_first_then_the_check_fails

        with pytest.raises(CommitUnconfirmed) as raised:
            a.commit(REF, expected_token=a_tok, new_bytes=b"v2")

        assert _INTERNAL_OPERATIONAL_ERROR in str(raised.value)
        assert len(fake_a.cas_calls) == 1, "a landed write was re-driven"
    finally:
        server.shutdown()


def test_a_bump_answered_with_the_failure_envelope_after_the_write_landed_is_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bump is sent only after the substrate write landed. When the
    coordinator answers it with its failure envelope, whether the bump landed
    is unknown, so the commit raises ``CommitUnconfirmed``: re-read before
    retrying, never re-drive. The write is not re-driven.

    Prevents a definite "rejected" for an outcome nobody knows: a caller told
    the commit was refused retries it and applies the write twice, or gives up
    while the substrate holds bytes the coordinator may never have recorded."""
    store = _FakeStore()
    store.seed(REF, b"v1")
    server = _serve_in_process(tmp_path, "substrate-bump-envelope")
    try:
        sa = _session(tmp_path, fast_cfg)
        a, fake = _agent(store, sa)
        _bytes, token = a.read(REF)
        _fail_the_next(server, monkeypatch, "/hooks/post-edit-cas")

        with pytest.raises(CommitUnconfirmed) as raised:
            a.commit(REF, expected_token=token, new_bytes=b"v2")

        assert _INTERNAL_OPERATIONAL_ERROR in str(raised.value)
        assert len(fake.cas_calls) == 1, "a landed write was re-driven"
        assert store.get(REF)[0] == b"v2", "control: the substrate write landed"
    finally:
        server.shutdown()
