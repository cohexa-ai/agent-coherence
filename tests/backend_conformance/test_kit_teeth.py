# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Teeth: a deliberately-degraded backend MUST FAIL the kit (plan Unit 5).

A conformance kit that every plausible implementation passes proves nothing. This
test builds a backend that is correct on the version-CAS leg but DROPS one leg of
the R9 atomic boundary — the grant arbitration — and asserts the kit's
single-writer-under-contention scenario CATCHES it. If the degraded stub ever
passed that scenario, the kit would be decorative; the assertion here is that it
does NOT (``pytest.raises(AssertionError)``), and that the failure message NAMES
the missing tuple element so a real backend author knows what they skipped.

The degraded stub (:class:`_GrantBlindRegistry`) wraps a real in-memory registry
and delegates EVERYTHING to it EXCEPT ``commit_cas``, which it reimplements as a
VERSION-ONLY compare-and-swap: it checks the artifact version but SKIPS the
``other_holder`` grant check that the real ``commit_cas`` performs in the same
atomic step. That is exactly the bug a naive "just do a version CAS in the
backend" implementation would ship — the version matches, so a version-only CAS
lets an OCC writer win while a pessimistic peer still holds MODIFIED, producing
TWO writers. The kit's single-writer scenario is written to isolate precisely
this leg (the OCC writer commits at the CORRECT version, so only the grant check
can stop it).

The MUST-MATCH scenarios that do NOT depend on the grant leg (pure version-CAS
arbitration, admit-on-absent, the fence staying sticky across a peer's fetch)
still PASS the stub — proving the teeth test fails the stub for the RIGHT reason
(the dropped grant leg), not because the stub is broken everywhere.

A third stub (:class:`_RearmingRegistry`) gives the peer-fetch scenario its
teeth: it delegates everything, but a requester's ``"fetch"`` set re-lists every
OTHER non-INVALID holder under the capture trigger — the effect the service's
fetch loop used to have when it rewrote peers that were already SHARED, moved
inside the backend. The kit's peer-fetch scenario MUST FAIL it.
"""

from __future__ import annotations

import time
from pathlib import Path
from uuid import UUID

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import CLAIM_CAPTURE_TRIGGERS, CasResult
from ccs.core.states import MESIState
from ccs.core.types import CasCorruption, ConflictDetail
from tests.backend_conformance import kit
from tests.backend_conformance.kit import RegistryFactory

_M_OR_E = frozenset({MESIState.MODIFIED, MESIState.EXCLUSIVE})


class _GrantBlindRegistry:
    """A degraded backend: correct version-CAS, but the grant-arbitration leg of
    the R9 boundary is DROPPED. Delegates everything to a wrapped real in-memory
    registry via ``__getattr__`` EXCEPT ``commit_cas``, which it reimplements to
    skip the ``other_holder`` check — so a version-matching OCC writer wins even
    while a pessimistic peer holds MODIFIED (single-writer violated)."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        # Every member except commit_cas comes straight from the real registry.
        return getattr(self._inner, name)

    def commit_cas(  # noqa: D401 - degraded on purpose
        self,
        artifact_id: UUID,
        agent_id: UUID,
        *,
        expected_version: int,
        content_hash: str,
        size_tokens: int | None = None,
        content: bytes | str | None = None,
        tick: int = 0,
        trigger: str = "commit_cas",
    ) -> CasResult:
        """Version-only CAS — the grant-arbitration leg is intentionally MISSING.

        It performs the version compare correctly (so the pure-CAS and
        admit-on-absent scenarios still pass), then WINS without checking whether a
        peer holds M/E. To win it delegates to the inner registry's real
        ``commit_cas`` AFTER neutralizing any peer M/E grant — mechanically the
        same effect as a backend that simply never consulted ``state_by_agent``:
        the peer grant does not block the write."""
        record = self._inner._records.get(artifact_id)  # noqa: SLF001 - test stub reaches in
        if record is None:
            raise KeyError(f"artifact {artifact_id} not in registry")
        current = record.artifact.version
        if expected_version > current:
            return CasCorruption(current_version=current)
        if expected_version < current:
            return ConflictDetail("version_mismatch", current)
        # THE DROPPED LEG: a correct commit_cas rejects here when a peer holds
        # M/E (ConflictDetail("other_holder")). This degraded stub does NOT —
        # it demotes every peer M/E grant so the inner real commit_cas cannot
        # see a competing holder, then wins on the version leg alone.
        for peer_id, state in list(record.state_by_agent.items()):
            if peer_id != agent_id and state in _M_OR_E:
                record.state_by_agent[peer_id] = MESIState.SHARED
        return self._inner.commit_cas(
            artifact_id,
            agent_id,
            expected_version=expected_version,
            content_hash=content_hash,
            size_tokens=size_tokens,
            content=content,
            tick=tick,
            trigger=trigger,
        )


class _DegradedFactory:
    """A :class:`RegistryFactory` minting :class:`_GrantBlindRegistry`. Single
    object per factory (the degraded stub is process-scoped, like in-memory), so
    ``db_path`` is ``None`` — the concurrency scenarios run against the one store
    object, which is all the single-writer teeth scenario needs."""

    def __init__(self) -> None:
        self._reg: _GrantBlindRegistry | None = None

    def __call__(self) -> _GrantBlindRegistry:
        if self._reg is None:
            self._reg = _GrantBlindRegistry()
        return self._reg

    def close_all(self) -> None:
        return None

    @property
    def db_path(self) -> Path | None:
        return None


class _TornPairRegistry:
    """A degraded backend whose ``get_artifact_and_generation`` is TWO reads
    instead of one atomic snapshot — the exact shortcut a BYO backend author
    would reach for, and the one the member's contract forbids. Everything else
    delegates to a real in-memory registry. The `sleep(0)` between the two reads
    just widens the interleaving window a real two-read backend has anyway."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def get_artifact_and_generation(self, artifact_id: UUID):
        artifact = self._inner.get_artifact(artifact_id)
        # A real (GIL-releasing) sleep, so the mutator thread actually runs
        # between the two reads — the interleaving window every two-read
        # backend has, made wide enough to hit deterministically.
        time.sleep(0.0005)
        return artifact, self._inner.get_owner_generation(artifact_id)


class _TornPairFactory:
    """A :class:`RegistryFactory` minting one process-scoped
    :class:`_TornPairRegistry` (same shape as :class:`_DegradedFactory`)."""

    def __init__(self) -> None:
        self._reg: _TornPairRegistry | None = None

    def __call__(self) -> _TornPairRegistry:
        if self._reg is None:
            self._reg = _TornPairRegistry()
        return self._reg

    def close_all(self) -> None:
        return None

    @property
    def db_path(self) -> Path | None:
        return None


class _RearmingRegistry:
    """A degraded backend that RE-ARMS the read-generation fence on a peer's
    read. Delegates everything to a wrapped real in-memory registry via
    ``__getattr__`` EXCEPT ``set_agent_state``: after delegating the call, any
    set carrying the capture trigger toward a non-INVALID state re-lists EVERY
    other non-INVALID holder of that artifact under the same trigger. That is
    the effect the service's fetch loop used to have when it rewrote peers that
    were already SHARED, reproduced inside the backend. The registry exposes no
    setter for ``read_generation``, so the refresh IS a re-issued
    ``set_agent_state`` at the holder's current state — which the real
    predicate cannot tell from that holder's own read, so it re-captures the
    current ``owner_generation`` on a read the holder never made.

    A per-target re-capture alone would have no teeth: the fixed service never
    routes a ``"fetch"`` set to a peer that is already SHARED, so only a wrapper
    that refreshes the OTHER holders can reach the zombie."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        # Every member except set_agent_state comes straight from the real registry.
        return getattr(self._inner, name)

    def set_agent_state(  # noqa: D401 - degraded on purpose
        self,
        artifact_id: UUID,
        agent_id: UUID,
        state: MESIState,
        *,
        trigger: str = "unknown",
        tick: int = 0,
        content_hash: str | None = None,
    ) -> None:
        self._inner.set_agent_state(
            artifact_id, agent_id, state, trigger=trigger, tick=tick, content_hash=content_hash
        )
        if trigger not in CLAIM_CAPTURE_TRIGGERS or state == MESIState.INVALID:
            return
        # THE INJECTED BUG: one agent's read re-lists every other non-INVALID
        # holder under the capture trigger. To the inner registry each re-set is
        # that holder's own read, so its read_generation is refreshed.
        for holder, holder_state in self._inner.get_state_map(artifact_id).items():
            if holder == agent_id or holder_state == MESIState.INVALID:
                continue
            self._inner.set_agent_state(
                artifact_id, holder, holder_state, trigger=trigger, tick=tick
            )


class _RearmingFactory:
    """A :class:`RegistryFactory` minting one process-scoped
    :class:`_RearmingRegistry` (same shape as :class:`_DegradedFactory`)."""

    def __init__(self) -> None:
        self._reg: _RearmingRegistry | None = None

    def __call__(self) -> _RearmingRegistry:
        if self._reg is None:
            self._reg = _RearmingRegistry()
        return self._reg

    def close_all(self) -> None:
        return None

    @property
    def db_path(self) -> Path | None:
        return None


def test_rearming_stub_fails_peer_fetch_scenario() -> None:
    """THE TEETH for the peer-fetch scenario. A backend that refreshes every
    non-INVALID holder's ``read_generation`` whenever a requester's ``"fetch"``
    set lands MUST FAIL the kit — otherwise the scenario would bless the very
    re-arm that let a refused commit land because someone else read. The failure
    message names the peer-fetch leg and the obligation a backend author skipped."""
    factory: RegistryFactory = _RearmingFactory()
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_peer_fetch_does_not_rearm_superseded_read_generation(factory)
    message = str(excinfo.value)
    assert "PEER's fetch RE-ARMED" in message
    # The wrapper refreshed the operand to the CURRENT epoch (0 → 1), which is
    # exactly the re-arm — not some unrelated corruption of the value.
    assert "moved from 0 to 1" in message
    assert "OWN acquire or OWN read" in message, (
        "the teeth failure must name the obligation a backend author skipped; "
        f"got: {message}"
    )


class _CauselessReclaimRegistry:
    """A degraded backend whose ``set_agent_state`` performs a sweep reclaim but
    records no reclamation slot, as a backend would that relied on a separate
    slot write the sweep no longer makes. Everything else is the real in-memory
    registry."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry()

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def set_agent_state(self, artifact_id: UUID, agent_id: UUID, state: MESIState, **kwargs: object) -> None:
        self._inner.set_agent_state(artifact_id, agent_id, state, **kwargs)  # type: ignore[arg-type]
        # THE DROPPED OBLIGATION: the real registry recorded the slot in the
        # transition; this stub forgets it.
        self._inner._records[artifact_id].last_reclamation_by_agent.pop(agent_id, None)  # noqa: SLF001


class _CauselessReclaimFactory:
    """A :class:`RegistryFactory` minting one process-scoped
    :class:`_CauselessReclaimRegistry` (same shape as :class:`_DegradedFactory`)."""

    def __init__(self) -> None:
        self._reg: _CauselessReclaimRegistry | None = None

    def __call__(self) -> _CauselessReclaimRegistry:
        if self._reg is None:
            self._reg = _CauselessReclaimRegistry()
        return self._reg

    def close_all(self) -> None:
        return None

    @property
    def db_path(self) -> Path | None:
        return None


def test_causeless_reclaim_stub_fails_reclaim_cause_scenario() -> None:
    """THE TEETH for the reclaim-cause scenario. A backend that reclaims a grant
    without recording why MUST FAIL the kit: the sweep writes no slot of its
    own, so on that backend every reclaim would read as a release."""
    factory: RegistryFactory = _CauselessReclaimFactory()
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_sweep_reclaim_records_its_cause(factory)
    message = str(excinfo.value)
    assert "reclamation slot None" in message
    assert "same write as the reclaim" in message, (
        "the teeth failure must name the obligation a backend author skipped; "
        f"got: {message}"
    )


def test_torn_pair_stub_fails_pair_atomicity_scenario() -> None:
    """THE TEETH for the pair-atomicity scenario. A backend serving
    ``get_artifact_and_generation`` as two independent reads MUST FAIL the kit —
    otherwise the scenario would bless the shortcut that reopens the
    reclaim-zombie effect hole. The failure message names the obligation."""
    factory: RegistryFactory = _TornPairFactory()
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_version_and_generation_pair_is_untearable(factory)
    message = str(excinfo.value)
    assert "TORN pair" in message
    assert "ONE atomic snapshot" in message, (
        "the teeth failure must name the obligation a backend author skipped; "
        f"got: {message}"
    )


def test_degraded_stub_fails_single_writer_scenario() -> None:
    """THE TEETH. The kit's single-writer-under-contention scenario MUST FAIL the
    grant-blind stub — proving the kit actually discriminates a backend that drops
    the grant-arbitration leg. The failure is a raised ``AssertionError`` whose
    message NAMES the missing tuple element (grant arbitration / other_holder)."""
    factory: RegistryFactory = _DegradedFactory()
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_single_writer_under_contention(factory)
    message = str(excinfo.value)
    assert "single-writer VIOLATED" in message
    assert kit.OTHER_HOLDER_REASON in message, (
        "the teeth failure must name the missing tuple element (the grant-"
        "arbitration / other_holder leg) so a backend author knows what they "
        f"skipped; got: {message}"
    )


def test_degraded_stub_still_passes_grant_independent_scenarios() -> None:
    """The stub only drops the GRANT leg — so the scenarios that do not depend on
    it (pure version-CAS arbitration, fence admit-on-absent) still PASS. This
    proves the teeth test fails the stub for the RIGHT reason (the dropped grant
    leg), not because the stub is broken across the board (which would make the
    single-writer failure uninformative)."""
    factory: RegistryFactory = _DegradedFactory()
    # Pure version-CAS: two SHARED writers, one winner, loser version_mismatch —
    # no grant leg involved, so the degraded stub is still correct here.
    kit.assert_cas_arbitration_one_winner(_DegradedFactory())
    # Admit-on-absent: a plain OCC writer with no fence claim wins — again grant-
    # independent (a fresh factory so no cross-scenario state bleed).
    kit.assert_fence_admits_absent_read_generation(factory)
    # Peer-fetch stickiness: the stub delegates set_agent_state, so a peer's
    # fetch reaches the real capture predicate and the zombie stays refused —
    # grant-independent, still correct (a fresh factory again).
    kit.assert_peer_fetch_does_not_rearm_superseded_read_generation(_DegradedFactory())


def test_degraded_stub_fence_reject_leg_still_holds() -> None:
    """The stub delegates the fence to the real inner registry, so the fence
    REJECT leg still works — the degradation is scoped to grant arbitration ONLY.
    Documents the exact blast radius of the injected bug (one leg, not the fence).
    """
    kit.assert_fence_rejects_superseded_read_generation(_DegradedFactory())
