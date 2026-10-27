# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Unit 6 — the effect-gate wrapper (EO-5): "fire effect E iff read-set R
unchanged" (SB-17 / TX-1).

Plan: ``docs/plans/2026-06-26-002-feat-read-side-transaction-snapshot-plan.md``
Unit 6. Requirement trace: **EO-5** (the ergonomic EO-4 = SB-17 user surface).
Builds on Units 2-5: ``begin_session`` / cut-capture (U2), ``session_read`` (U3),
``session_commit`` (U4), the heartbeat lease + session-liveness sweep + fail-closed
``SessionInvalidated`` (U5). The gate COMPOSES those primitives; it re-implements
none of them.

``CoordinatorService.effect_gate`` is a PURE IN-PROCESS coordinator method (not the
CoherentVolume HTTP path), so fail-closed is enforced by the typed session results
directly — a dead session at any step RAISES ``SessionInvalidated``, never fires.

The two modes under test:

- **ATOMIC** (``commit=(artifact_id, content)``) — the whole cut is re-validated and
  the write routed through ``session_commit`` (``commit_cas`` at the pinned base)
  under one hold of the service and registry locks: any moved member HOLDS, and no
  writer on the same registry instance can land between the check and the commit. A
  raced target write loses the CAS cleanly (HELD with the shipped ``ConflictDetail``
  carried through verbatim). A writer on a second handle to the same store is
  outside the hold; that residual is OBSERVED here, not asserted away.
- **ESCAPING** (``effect=callable``) — re-validate, then fire the callable. The
  guarantee is "unchanged AS OF the re-validate point", NOT "as of the fire point":
  the residual re-validate->fire window is unclosable (EO-7) and is OBSERVED here,
  not asserted away.

Timing is exercised with FIXED-STALE BUFFERS (a peer commit driven explicitly at a
chosen point), never counters or wall-clock sleeps — the institutional steer for
the multi-read window (a protocol proof bounds the registry, not the integration).
The thread tests start their peer at a chosen point too; their ``_WINDOW_WAIT``
join only gives a peer the chance to land, and they pass deterministically when
the gate's lock hold works (the peer cannot land under it).
Reason matching is ALWAYS ``reason == CONSTANT`` against the imported wire-stable
constant, never a substring of a human message.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.retention import RetentionPolicy
from ccs.coordinator.service import CoordinatorService, SessionView
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    SESSION_INVALIDATED_REASON,
    VERSION_MISMATCH_REASON,
    CoherenceError,
    SessionInvalidated,
)
from ccs.core.states import MESIState
from ccs.core.types import (
    Artifact,
    ConflictDetail,
    DataPlaneDeferredRead,
    EffectFired,
    EffectHeld,
    VersionedContent,
)

_HB_TIMEOUT = 120
# How long a peer writer started inside the atomic gate's check-to-commit span
# gets to land before the gate's CAS runs. Without the gate's lock hold it lands
# in milliseconds; with the hold it is still blocked when this runs out.
_WINDOW_WAIT = 0.5


# ---------------------------------------------------------------------------
# Builders — mirror tests/test_session_commit.py / test_session_lifetime.py.
# ---------------------------------------------------------------------------


def _register(reg, artifact_id: UUID, name: str, body: str, version: int = 1) -> None:
    art = Artifact(id=artifact_id, name=name, version=version, content_hash="h1")
    reg.register_artifact(art, content=body)


def _peer_commit_cas(
    reg, artifact_id: UUID, writer: UUID, expected: int, body: str | None
) -> None:
    """A PEER OCC commit via the registry ``commit_cas`` WIN — advances current
    past the pin so a subsequent gate at the captured version is HELD."""
    reg.set_agent_state(artifact_id, writer, MESIState.SHARED, tick=1)
    reg.commit_cas(
        artifact_id,
        writer,
        expected_version=expected,
        content_hash="h2",
        content=body,
        tick=2,
    )


def _lazy_registries(tmp_path: Path):
    """An (in_memory, sqlite) LAZY pair — ``retain_versions=True`` (bodies in
    history) so ``session_read`` serves concrete pinned bytes on both arms."""
    pol = RetentionPolicy(max_versions=8)
    mem = ArtifactRegistry(retain_versions=True, retention_policy=pol)
    sql = SqliteArtifactRegistry(
        tmp_path / "gate_lazy.db", retain_versions=True, retention_policy=pol
    )
    return mem, sql


def _eager_registries(tmp_path: Path):
    """An (in_memory, sqlite) EAGER pair — ``retain_versions=False`` (the
    ``content=None`` ICP); ``session_read`` returns ``DataPlaneDeferredRead``."""
    mem = ArtifactRegistry(retain_versions=False)
    sql = SqliteArtifactRegistry(tmp_path / "gate_eager.db", retain_versions=False)
    return mem, sql


# ===========================================================================
# A — Happy path: read-set unchanged through re-validate -> effect fires.
# ===========================================================================


class TestHappyPathFires:
    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_escaping_effect_fires_when_unchanged(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a, b = uuid4(), uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")
            _register(reg, b, "budget.md", "BUDGET-V1")

            fired: list[object] = []

            def decide(view: SessionView) -> str:
                plan = view.read(a)
                budget = view.read(b)
                assert isinstance(plan, VersionedContent)
                assert isinstance(budget, VersionedContent)
                return f"{plan.content}+{budget.content}"

            def effect(decision: object) -> str:
                fired.append(decision)
                return "DEPLOYED"

            outcome = svc.effect_gate(
                read_set=[a, b], owner=owner, decide=decide, effect=effect
            )

            assert isinstance(outcome, EffectFired)
            # The callable fired EXACTLY once with the decision computed off the cut.
            assert fired == ["PLAN-V1+BUDGET-V1"]
            assert outcome.result == "DEPLOYED"
            assert outcome.revalidated_cut == {a: 1, b: 1}
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_commit_wins_when_unchanged(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            def decide(view: SessionView) -> object:
                return view.read(a)

            outcome = svc.effect_gate(
                read_set=[a],
                owner=owner,
                decide=decide,
                commit=(a, "PLAN-V2"),
            )

            assert isinstance(outcome, EffectFired)
            assert outcome.commit is not None
            updated_artifact, signals = outcome.commit
            assert updated_artifact.version == 2  # pinned 1 -> 2
            assert isinstance(signals, list)
            # The registry advanced.
            assert reg.get_artifact(a).version == 2
        finally:
            sql.close()


# ===========================================================================
# B — HELD: a peer advances a read-set member after capture -> NEVER fires.
# ===========================================================================


class TestHeldOnDrift:
    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_escaping_effect_held_when_member_moved(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a, b = uuid4(), uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")
            _register(reg, b, "budget.md", "BUDGET-V1")

            fired: list[object] = []

            def decide(view: SessionView) -> str:
                # A peer commits a READ-SET MEMBER between the pin and the effect
                # boundary — driven explicitly (a fixed-stale buffer, not a sleep)
                # from inside decide so the drift is deterministic.
                _peer_commit_cas(reg, b, uuid4(), expected=1, body="BUDGET-V2")
                return "decision-on-stale-budget"

            def effect(decision: object) -> str:
                fired.append(decision)
                return "DEPLOYED"

            outcome = svc.effect_gate(
                read_set=[a, b], owner=owner, decide=decide, effect=effect
            )

            assert isinstance(outcome, EffectHeld)
            # The effect was NEVER invoked — no fire on stale input.
            assert fired == []
            # The drift map names the moved member (pinned 1 -> current 2).
            assert outcome.moved == {b: (1, 2)}
            # No commit conflict for an escaping HELD (no CAS was attempted).
            assert outcome.conflict is None
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_commit_held_returns_conflict_when_target_moved(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            def decide(view: SessionView) -> object:
                # A peer advances the COMMIT TARGET after the pin.
                _peer_commit_cas(reg, a, uuid4(), expected=1, body="PLAN-V2-PEER")
                return view  # decision value unused for atomic mode

            outcome = svc.effect_gate(
                read_set=[a],
                owner=owner,
                decide=decide,
                commit=(a, "PLAN-V2-MINE"),
            )

            assert isinstance(outcome, EffectHeld)
            # The fast pre-CAS re-validate short-circuits here, so no commit was
            # attempted -> the drift map names the moved target, conflict is None.
            assert outcome.moved == {a: (1, 2)}
            # The registry still carries the PEER's bytes (mine never landed).
            assert reg.get_artifact(a).version == 2
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_commit_conflict_surfaced_when_cas_loses(
        self, tmp_path: Path, arm: str
    ) -> None:
        """The TRUE CAS-loss branch (finding F8): the pre-CAS re-validate sees the
        commit target as UNCHANGED (so it does NOT short-circuit), the CAS then
        RUNS and LOSES at the pinned base, and the shipped ``ConflictDetail`` is
        surfaced verbatim inside ``EffectHeld.conflict``. ``conflict is not None``
        is exactly what distinguishes this from the pre-CAS short-circuit path
        (which the prior version of this test actually exercised, leaving the CAS
        loss branch uncovered).

        Determinism: the TOCTOU window between the pre-CAS re-validate and the CAS
        is simulated by stubbing the FIRST ``_revalidate_cut`` to report no drift,
        then letting the REAL CAS lose against the peer-moved target."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            real_revalidate = svc._revalidate_cut
            calls = {"n": 0}

            def fake_revalidate(cut):
                calls["n"] += 1
                # 1st call = the pre-CAS short-circuit check: hide the drift so the
                # CAS actually runs. Later calls (post-conflict) are the REAL drift.
                if calls["n"] == 1:
                    return {}
                return real_revalidate(cut)

            svc._revalidate_cut = fake_revalidate

            def decide(view: SessionView) -> object:
                # The peer moves the commit target after the pin; the stubbed
                # pre-CAS check hides it, so the CAS at pinned base v1 loses to v2.
                _peer_commit_cas(reg, a, uuid4(), expected=1, body="PLAN-PEER-V2")
                return view

            outcome = svc.effect_gate(
                read_set=[a], owner=owner, decide=decide, commit=(a, "PLAN-MINE"),
            )

            assert calls["n"] == 2  # pre-CAS (hidden) THEN post-conflict (real)
            assert isinstance(outcome, EffectHeld)
            # DISTINGUISHING assertion: the CAS ran and lost, so the shipped
            # ConflictDetail is carried through verbatim (short-circuit → None).
            assert outcome.conflict is not None
            assert outcome.conflict.reason == "version_mismatch"
            # The post-conflict re-validate names the moved target.
            assert outcome.moved.get(a) == (1, 2)
            # The peer's bytes stand; mine never landed (no phantom write).
            assert reg.get_artifact(a).version == 2
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_commit_held_when_a_non_target_member_moved(
        self, tmp_path: Path, arm: str
    ) -> None:
        """#456: the commit is derived from EVERY read-set member, so a peer
        moving one the gate does not write must hold it, exactly as escaping
        mode holds. Before the fix the gate checked only the commit target,
        committed the plan from a superseded brief, and reported the brief
        unchanged in ``revalidated_cut``."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a, b = uuid4(), uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")
            _register(reg, b, "brief.md", "BRIEF-V1")

            def decide(view: SessionView) -> object:
                view.read(a)
                view.read(b)
                # A peer moves the brief, which the gate reads but does not write.
                _peer_commit_cas(reg, b, uuid4(), expected=1, body="BRIEF-V2")
                return view

            outcome = svc.effect_gate(
                read_set=[a, b],
                owner=owner,
                decide=decide,
                commit=(a, "PLAN-FROM-BRIEF-V1"),
            )

            assert isinstance(outcome, EffectHeld)
            assert outcome.moved == {b: (1, 2)}
            # Caught at the check, so no CAS was attempted.
            assert outcome.conflict is None
            # The plan derived from the superseded brief never landed.
            assert reg.get_artifact(a).version == 1
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_commit_held_when_a_non_target_member_vanished(
        self, tmp_path: Path, arm: str
    ) -> None:
        """A read-set member removed under the pin can no longer be proven
        unchanged, so the atomic gate holds (``current`` is ``None``) even when
        it is not the commit target."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a, b = uuid4(), uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")
            _register(reg, b, "brief.md", "BRIEF-V1")

            def decide(view: SessionView) -> object:
                view.read(b)
                reg.remove_artifact(b)
                return view

            outcome = svc.effect_gate(
                read_set=[a, b],
                owner=owner,
                decide=decide,
                commit=(a, "PLAN-FROM-A-REMOVED-BRIEF"),
            )

            assert isinstance(outcome, EffectHeld)
            assert outcome.moved == {b: (1, None)}
            assert outcome.conflict is None
            assert reg.get_artifact(a).version == 1
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_commit_held_with_nothing_moved_when_a_peer_holds_the_target(
        self, tmp_path: Path, arm: str
    ) -> None:
        """A pessimistic peer holding the target's write grant moves no version,
        so the re-validate passes and the CAS refuses the commit as
        ``other_holder``. The gate holds with that conflict and an EMPTY
        ``moved``: nothing moved, but the commit could not land."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            def decide(view: SessionView) -> object:
                view.read(a)
                reg.set_agent_state(a, uuid4(), MESIState.EXCLUSIVE, tick=1)
                return view

            outcome = svc.effect_gate(
                read_set=[a], owner=owner, decide=decide, commit=(a, "PLAN-V2"),
            )

            assert isinstance(outcome, EffectHeld)
            assert outcome.conflict is not None
            assert outcome.conflict.reason == "other_holder"
            assert outcome.moved == {}
            assert reg.get_artifact(a).version == 1
        finally:
            sql.close()


# ===========================================================================
# C — Fail-closed: the session is reaped mid-gate -> SessionInvalidated, no fire.
# ===========================================================================


class TestFailClosed:
    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_reaped_session_in_decide_fails_closed(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            fired: list[object] = []

            def decide(view: SessionView) -> object:
                # Reap THIS session mid-gate: drive the liveness sweep past the
                # heartbeat window (fixed ticks, no sleep). The session's pins are
                # released, so the subsequent read fails closed.
                reaped = svc.enforce_session_liveness(
                    current_tick=_HB_TIMEOUT + 1,
                    heartbeat_timeout_ticks=_HB_TIMEOUT,
                )
                assert reaped == 1
                return view.read(a)  # fails closed here

            def effect(decision: object) -> str:
                fired.append(decision)
                return "DEPLOYED"

            with pytest.raises(SessionInvalidated) as exc:
                svc.effect_gate(
                    read_set=[a],
                    owner=owner,
                    decide=decide,
                    effect=effect,
                    created_at_tick=0,
                )
            assert exc.value.reason == SESSION_INVALIDATED_REASON
            assert fired == []  # the effect NEVER ran
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_reaped_session_at_atomic_commit_fails_closed(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            def decide(view: SessionView) -> object:
                # Reap after the decision read but before the commit fires.
                svc.session_read(view._token, a, caller=owner)  # a valid pinned read first
                svc.enforce_session_liveness(
                    current_tick=_HB_TIMEOUT + 1,
                    heartbeat_timeout_ticks=_HB_TIMEOUT,
                )
                return view

            with pytest.raises(SessionInvalidated) as exc:
                svc.effect_gate(
                    read_set=[a],
                    owner=owner,
                    decide=decide,
                    commit=(a, "PLAN-V2"),
                    created_at_tick=0,
                )
            assert exc.value.reason == SESSION_INVALIDATED_REASON
            # The commit NEVER landed — the artifact stays at v1.
            assert reg.get_artifact(a).version == 1
        finally:
            sql.close()

    def test_unknown_artifact_in_read_set_fails_closed(self, tmp_path: Path) -> None:
        mem, _sql = _lazy_registries(tmp_path)
        svc = CoordinatorService(mem)
        a = uuid4()
        _register(mem, a, "plan.md", "PLAN-V1")
        ghost = uuid4()  # never registered

        with pytest.raises(SessionInvalidated):
            svc.effect_gate(
                read_set=[a, ghost],
                owner=uuid4(),
                decide=lambda view: None,
                effect=lambda d: "DEPLOYED",
            )


# ===========================================================================
# D — Atomic vs escaping: the residual re-validate->fire window (EO-7).
# ===========================================================================


class TestRevalidateFireWindow:
    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_escaping_window_is_observed_not_asserted_away(
        self, tmp_path: Path, arm: str
    ) -> None:
        """A peer commit landing in the RE-VALIDATE->FIRE window for an ESCAPING
        effect: the effect FIRES on a vector that was valid at re-validate but
        moved before the fire completed. The window is OBSERVED (the gate fired
        AND the member moved), NOT asserted away. This is the honest EO-5 / EO-7
        bound — only ATOMIC mode (``session.commit``) closes the window."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            # The side effect itself lands a peer commit on a read-set member as
            # its FIRST action — i.e. exactly inside the unclosable window, after
            # the gate's re-validate already passed. This is the deterministic
            # stand-in for a concurrent peer commit racing the fire (a fixed-stale
            # buffer, not a sleep): the gate cannot have seen it, because it fires
            # the callable only after re-validate.
            window_fired: list[bool] = []

            def decide(view: SessionView) -> object:
                return view.read(a)

            def escaping_effect(decision: object) -> str:
                # A peer moves the pinned member DURING the fire.
                _peer_commit_cas(reg, a, uuid4(), expected=1, body="PLAN-V2-RACED")
                window_fired.append(True)
                return "DEPLOYED-ON-NOW-STALE-VECTOR"

            outcome = svc.effect_gate(
                read_set=[a], owner=owner, decide=decide, effect=escaping_effect
            )

            # The gate FIRED (re-validate passed before the callable ran) ...
            assert isinstance(outcome, EffectFired)
            assert window_fired == [True]
            assert outcome.result == "DEPLOYED-ON-NOW-STALE-VECTOR"
            # ... yet the member MOVED inside the window: the effect fired on a
            # vector that is no longer current. The gate does NOT roll back (EO-7);
            # the residual window is real and is documented, not claimed away.
            assert reg.get_artifact(a).version == 2  # moved during fire
            assert outcome.revalidated_cut == {a: 1}  # what was proven at re-validate
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_mode_closes_the_window(
        self, tmp_path: Path, arm: str
    ) -> None:
        """The ATOMIC counterpart for the commit TARGET: when a peer moves it after
        the pin, the atomic commit HOLDS (no fire) and only the peer's write lands.
        The peer commits inside ``decide``, so the pre-commit check catches it;
        the CAS-loss branch is ``test_atomic_commit_conflict_surfaced_when_cas_loses``
        and a peer writing between the check and the CAS is
        ``test_atomic_peer_writer_cannot_land_between_check_and_commit``."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            def decide(view: SessionView) -> object:
                _peer_commit_cas(reg, a, uuid4(), expected=1, body="PLAN-V2-PEER")
                return view

            outcome = svc.effect_gate(
                read_set=[a], owner=owner, decide=decide, commit=(a, "PLAN-V2-MINE")
            )

            # Atomic mode HELD — the peer's commit is the only one that landed.
            assert isinstance(outcome, EffectHeld)
            assert reg.get_artifact(a).version == 2
            # The pinned-base CAS would surface version_mismatch had the pre-CAS
            # short-circuit not caught it first; either way nothing of mine landed.
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_peer_writer_cannot_land_between_check_and_commit(
        self, tmp_path: Path, arm: str
    ) -> None:
        """#456: a peer writing a NON-target member after the atomic gate's
        whole-cut check passed, but before its CAS, waits until the gate's commit
        has landed. The check and the commit run under one hold of the service
        and registry locks, so the plan commits from a brief that is still
        current and the peer's write lands after it. Without the hold the peer
        lands in between and the plan commits from a superseded brief.

        The peer is started the moment the whole-cut check returns (wrapping
        ``_revalidate_cut``) and given ``_WINDOW_WAIT`` seconds to land before
        the gate goes on to its CAS."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a, b = uuid4(), uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")
            _register(reg, b, "brief.md", "BRIEF-V1")

            peer = threading.Thread(
                target=_peer_commit_cas, args=(reg, b, uuid4(), 1, "BRIEF-V2")
            )
            peer_done_before_cas: list[bool] = []
            real_revalidate = svc._revalidate_cut

            def revalidate_then_start_peer(cut):
                moved = real_revalidate(cut)
                if not peer_done_before_cas:
                    peer.start()
                    peer.join(timeout=_WINDOW_WAIT)
                    peer_done_before_cas.append(not peer.is_alive())
                return moved

            svc._revalidate_cut = revalidate_then_start_peer

            def decide(view: SessionView) -> object:
                view.read(a)
                view.read(b)
                return view

            outcome = svc.effect_gate(
                read_set=[a, b],
                owner=owner,
                decide=decide,
                commit=(a, "PLAN-FROM-BRIEF-V1"),
            )
            peer.join(timeout=10)

            # The peer was still waiting a full _WINDOW_WAIT after the re-validate
            # returned (that it stays held through the CAS is
            # test_atomic_check_and_cas_share_one_registry_lock_hold) ...
            assert peer_done_before_cas == [False]
            assert isinstance(outcome, EffectFired)
            assert outcome.revalidated_cut == {a: 1, b: 1}
            assert reg.get_artifact(a).version == 2
            # ... and landed afterwards.
            assert not peer.is_alive()
            assert reg.get_artifact(b).version == 2
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_check_and_cas_share_one_registry_lock_hold(
        self, tmp_path: Path, arm: str
    ) -> None:
        """The re-validate and the registry CAS run inside ONE hold of the
        registry lock, with no release between them. The peer-writer test above
        starts its peer when the re-validate returns, so it cannot see a hold
        that ends before the CAS, or a second, separate hold around the CAS;
        either reopens #456's window for a writer on the same registry. This
        pins the structure without timing: each point records how deep in
        ``abort_guard`` it runs and how many outermost holds had ended."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a, b = uuid4(), uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")
            _register(reg, b, "brief.md", "BRIEF-V1")

            depth = [0]
            holds_ended = [0]
            seen: dict[str, tuple[int, int]] = {}
            real_abort_guard = reg.abort_guard
            real_commit_cas = reg.commit_cas
            real_revalidate = svc._revalidate_cut

            @contextmanager
            def counting_abort_guard(*args, **kwargs):
                with real_abort_guard(*args, **kwargs):
                    depth[0] += 1
                    try:
                        yield
                    finally:
                        depth[0] -= 1
                        if depth[0] == 0:
                            holds_ended[0] += 1

            def recording_revalidate(cut):
                seen.setdefault("re-validate", (depth[0], holds_ended[0]))
                return real_revalidate(cut)

            def recording_commit_cas(*args, **kwargs):
                seen.setdefault("cas", (depth[0], holds_ended[0]))
                return real_commit_cas(*args, **kwargs)

            reg.abort_guard = counting_abort_guard
            reg.commit_cas = recording_commit_cas
            svc._revalidate_cut = recording_revalidate

            def decide(view: SessionView) -> object:
                view.read(a)
                view.read(b)
                return view

            outcome = svc.effect_gate(
                read_set=[a, b], owner=owner, decide=decide, commit=(a, "PLAN-V2"),
            )

            assert isinstance(outcome, EffectFired)
            check_depth, ended_at_check = seen["re-validate"]
            cas_depth, ended_at_cas = seen["cas"]
            assert check_depth > 0
            assert cas_depth > 0
            # No outermost hold ended between the re-validate and the CAS.
            assert ended_at_cas == ended_at_check
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_atomic_gate_takes_its_locks_in_begin_session_order(
        self, tmp_path: Path, arm: str
    ) -> None:
        """The atomic gate holds the service lock and the registry lock across
        its re-validate and commit, and must take them in the order
        ``begin_session`` does: service lock first. A ``begin_session`` started
        once the gate holds the registry lock then waits for the service lock
        the gate already holds. Taken the other way round, the begin takes the
        service lock and waits for the registry lock, the gate waits for the
        service lock, and neither finishes.

        The begin is started the moment the gate enters the registry lock after
        ``decide`` (wrapping that registry's ``abort_guard``) and given
        ``_WINDOW_WAIT`` seconds to take what it can before the gate goes on."""
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        deadlocked = False
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            begun: list[object] = []
            begin = threading.Thread(
                target=lambda: begun.append(
                    svc.begin_session(read_set=[a], owner=uuid4())
                ),
                daemon=True,
            )
            # Armed by ``decide``, so the gate's own ``begin_session`` (which
            # also enters the registry lock) does not start the begin early.
            armed: list[bool] = []
            real_abort_guard = reg.abort_guard

            @contextmanager
            def abort_guard_then_begin(*args, **kwargs):
                with real_abort_guard(*args, **kwargs):
                    if armed and begin.ident is None:
                        begin.start()
                        begin.join(timeout=_WINDOW_WAIT)
                    yield

            reg.abort_guard = abort_guard_then_begin

            def decide(view: SessionView) -> object:
                view.read(a)
                armed.append(True)
                return view

            outcomes: list[object] = []
            gate = threading.Thread(
                target=lambda: outcomes.append(
                    svc.effect_gate(
                        read_set=[a],
                        owner=owner,
                        decide=decide,
                        commit=(a, "PLAN-V2"),
                    )
                ),
                daemon=True,
            )
            gate.start()
            gate.join(timeout=10)
            begin.join(timeout=10)
            deadlocked = gate.is_alive() or begin.is_alive()

            assert not deadlocked, "the atomic gate deadlocked against begin_session"
            assert isinstance(outcomes[0], EffectFired)
            assert len(begun) == 1
        finally:
            # A deadlocked thread still holds the registry lock, which close()
            # takes: leave the handle open so the test fails instead of hanging.
            if not deadlocked:
                sql.close()

    def test_atomic_second_store_handle_window_is_observed(
        self, tmp_path: Path
    ) -> None:
        """The residual the lock hold does not close: a writer on a SECOND
        registry handle to the same SQLite file is outside it, so it can move a
        non-target member between the gate's check and its CAS, and the gate
        still fires. A store has one writer by contract; checking the read-set
        inside the registry's own commit transaction would close this (#375).
        OBSERVED here, not asserted away."""
        _, sql = _lazy_registries(tmp_path)
        other = SqliteArtifactRegistry(
            tmp_path / "gate_lazy.db",
            retain_versions=True,
            retention_policy=RetentionPolicy(max_versions=8),
        )
        try:
            svc = CoordinatorService(sql)
            owner = uuid4()
            a, b = uuid4(), uuid4()
            _register(sql, a, "plan.md", "PLAN-V1")
            _register(sql, b, "brief.md", "BRIEF-V1")
            real_session_commit = svc.session_commit

            def session_commit_after_other_writer(*args, **kwargs):
                _peer_commit_cas(other, b, uuid4(), expected=1, body="BRIEF-V2")
                return real_session_commit(*args, **kwargs)

            svc.session_commit = session_commit_after_other_writer

            def decide(view: SessionView) -> object:
                view.read(b)
                return view

            outcome = svc.effect_gate(
                read_set=[a, b],
                owner=owner,
                decide=decide,
                commit=(a, "PLAN-FROM-BRIEF-V1"),
            )

            # The gate fired although the brief moved inside the window.
            assert isinstance(outcome, EffectFired)
            assert sql.get_artifact(a).version == 2
            assert sql.get_artifact(b).version == 2
        finally:
            other.close()
            sql.close()


# ===========================================================================
# E — Classification on exact reason constant, not substring.
# ===========================================================================


class TestReasonClassification:
    def test_held_carries_shipped_conflict_reason_exactly(
        self, tmp_path: Path
    ) -> None:
        """When the atomic CAS loses (rather than the pre-CAS short-circuit), the
        carried ``ConflictDetail.reason`` is the shipped ``version_mismatch``
        constant verbatim — matched with ``==``, never a substring. Driven by
        moving the target so the CAS arbitrates at the pinned base."""
        mem, _sql = _lazy_registries(tmp_path)
        svc = CoordinatorService(mem)
        owner = uuid4()
        a = uuid4()
        _register(mem, a, "plan.md", "PLAN-V1")

        # Bypass the gate's pre-CAS short-circuit to exercise the CAS-loss path
        # directly: call _effect_gate_atomic with a cut whose target is already
        # advanced. We pin at v1, advance to v2, then commit -> CAS loses.
        session = svc.begin_session(read_set=[a], owner=owner)
        _peer_commit_cas(mem, a, uuid4(), expected=1, body="PLAN-PEER")

        # The session's cut still pins v1; session_commit at pin 1 vs current 2
        # returns the shipped ConflictDetail(version_mismatch).
        result = svc.session_commit(session.session_token, a, "PLAN-MINE", caller=owner)
        assert isinstance(result, ConflictDetail)
        assert result.reason == VERSION_MISMATCH_REASON  # exact constant, not substring

    def test_fail_closed_reason_is_exact_constant(self, tmp_path: Path) -> None:
        mem, _sql = _lazy_registries(tmp_path)
        svc = CoordinatorService(mem)
        a = uuid4()
        _register(mem, a, "plan.md", "PLAN-V1")

        def decide(view: SessionView) -> object:
            svc.enforce_session_liveness(
                current_tick=_HB_TIMEOUT + 1, heartbeat_timeout_ticks=_HB_TIMEOUT
            )
            return view.read(a)

        with pytest.raises(SessionInvalidated) as exc:
            svc.effect_gate(
                read_set=[a], owner=uuid4(), decide=decide, effect=lambda d: None
            )
        assert exc.value.reason == SESSION_INVALIDATED_REASON


# ===========================================================================
# F — Caller-misuse + eager-branch + cleanup edges.
# ===========================================================================


class TestEdges:
    def test_requires_exactly_one_effect_mode(self, tmp_path: Path) -> None:
        mem, _sql = _lazy_registries(tmp_path)
        svc = CoordinatorService(mem)
        a = uuid4()
        _register(mem, a, "plan.md", "PLAN-V1")

        # Neither effect nor commit.
        with pytest.raises(ValueError):
            svc.effect_gate(read_set=[a], owner=uuid4(), decide=lambda v: None)

        # Both effect and commit.
        with pytest.raises(ValueError):
            svc.effect_gate(
                read_set=[a],
                owner=uuid4(),
                decide=lambda v: None,
                effect=lambda d: None,
                commit=(a, "X"),
            )

    def test_decide_reading_unpinned_artifact_is_hard_error(
        self, tmp_path: Path
    ) -> None:
        mem, _sql = _lazy_registries(tmp_path)
        svc = CoordinatorService(mem)
        a, b = uuid4(), uuid4()
        _register(mem, a, "plan.md", "PLAN-V1")
        _register(mem, b, "budget.md", "BUDGET-V1")

        def decide(view: SessionView) -> object:
            return view.read(b)  # b is NOT in the read-set

        with pytest.raises(CoherenceError):
            svc.effect_gate(
                read_set=[a], owner=uuid4(), decide=decide, effect=lambda d: None
            )

    def test_eager_branch_decide_gets_data_plane_deferred(
        self, tmp_path: Path
    ) -> None:
        """On the EAGER (``content=None`` / retain-off) branch the coordinator
        holds no body, so the decide view gets a ``DataPlaneDeferredRead``. The
        gate still re-validates by VERSION (byte-source-independent) and fires."""
        mem, sql = _eager_registries(tmp_path)
        try:
            svc = CoordinatorService(mem)
            owner = uuid4()
            a = uuid4()
            _register(mem, a, "plan.md", "PLAN-V1")

            seen: list[object] = []

            def decide(view: SessionView) -> object:
                r = view.read(a)
                seen.append(r)
                return r

            def effect(decision: object) -> str:
                return "FIRED"

            outcome = svc.effect_gate(
                read_set=[a], owner=owner, decide=decide, effect=effect
            )
            assert isinstance(outcome, EffectFired)
            assert len(seen) == 1
            assert isinstance(seen[0], DataPlaneDeferredRead)
            assert seen[0].version == 1
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_release_on_exit_drops_the_pin(self, tmp_path: Path, arm: str) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            captured_token: list[str] = []

            def decide(view: SessionView) -> object:
                captured_token.append(view._token)
                return view.read(a)

            svc.effect_gate(
                read_set=[a], owner=owner, decide=decide, effect=lambda d: "X"
            )
            # After the gate, the pin is released by default — the cut is gone.
            tok = captured_token[0]
            assert reg.get_session_cut(tok) is None
        finally:
            sql.close()

    @pytest.mark.parametrize("arm", ["in_memory", "sqlite"])
    def test_keep_session_when_release_disabled(
        self, tmp_path: Path, arm: str
    ) -> None:
        mem, sql = _lazy_registries(tmp_path)
        reg = mem if arm == "in_memory" else sql
        try:
            svc = CoordinatorService(reg)
            owner = uuid4()
            a = uuid4()
            _register(reg, a, "plan.md", "PLAN-V1")

            captured_token: list[str] = []

            def decide(view: SessionView) -> object:
                captured_token.append(view._token)
                return view.read(a)

            svc.effect_gate(
                read_set=[a],
                owner=owner,
                decide=decide,
                effect=lambda d: "X",
                release_on_exit=False,
            )
            tok = captured_token[0]
            assert reg.get_session_cut(tok) is not None  # still live
        finally:
            sql.close()
