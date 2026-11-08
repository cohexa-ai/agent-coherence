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

The refusal-pair stubs give each of the kit's three refusal-pair scenarios its
teeth. Each delegates everything to a real in-memory registry and drops the
generation pair at ONE entry point (``commit_cas``, ``commit_all``, or the
fenced persist's raise), keeping the reason and version, as a binding would
that kept its pre-pair refusal constructor. A fourth fills the pair only on
``stale_read_generation``, and the other reasons' checks MUST catch it.

The INVALID-only bump stub (:class:`_InvalidOnlyBumpRegistry`) gives the
voluntary-release scenario's release-to-reader leg its teeth. It moves the
ownership epoch when a holder leaves EXCLUSIVE or MODIFIED only if the holder
lands INVALID, as a backend would that keyed the bump on the target state. It
passes the release to INVALID and MUST FAIL the release to SHARED.

The follow-up-read stub (:class:`_FollowUpReadRegistry`) gives the two
refusal-pair races their teeth. It decides every refusal on the right snapshot,
then re-reads the owner leg alone, or both legs, after a 0.5 ms window at one
entry point. The sequential scenarios cannot see that; the race for its entry
point MUST FAIL it as a snapshot violation. A starving stub
(:class:`_StarvedMutatorRegistry`) keeps a race's mutator from running beside
its actor, and the race MUST FAIL it as starved, not as a snapshot violation.

The forget stub (:class:`_ForgetDroppingRegistry`) gives the forget scenario
its teeth, once per obligation: it forgets past a held write grant, deletes
the row that records a reclaim, leaves that row as a reader, deletes before
it refuses, keeps the heartbeat, miscounts the rows it removed, deletes a
peer's row, or refuses on a peer's write grant. The scenario MUST FAIL each,
naming the obligation dropped.
"""

from __future__ import annotations

import threading
import time
from functools import partial
from pathlib import Path
from uuid import UUID

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import CLAIM_CAPTURE_TRIGGERS, CasResult, forget_refusal
from ccs.core.exceptions import STALE_READ_GENERATION_REASON, StaleReadGeneration
from ccs.core.states import MESIState
from ccs.core.types import CasCorruption, ConflictDetail, MultiCommitConflict
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


def _without_pair(detail: ConflictDetail) -> ConflictDetail:
    return ConflictDetail(detail.reason, detail.current_version)


def _strip_batch(result: object, *, keep: frozenset[str] = frozenset()) -> object:
    """``result`` with the pair dropped from every refused member whose reason
    is not in ``keep``."""
    if not isinstance(result, MultiCommitConflict):
        return result
    return MultiCommitConflict(
        per_artifact={
            art: detail if detail.reason in keep else _without_pair(detail)
            for art, detail in result.per_artifact.items()
        }
    )


class _PairlessCasRegistry:
    """A degraded backend whose ``commit_cas`` refusals carry the reason and
    version but no generation pair. Everything else is the real in-memory
    registry."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def commit_cas(self, *args: object, **kwargs: object) -> CasResult:
        result = self._inner.commit_cas(*args, **kwargs)  # type: ignore[arg-type]
        return _without_pair(result) if isinstance(result, ConflictDetail) else result


class _PairlessBatchRegistry:
    """A degraded backend whose ``commit_all`` names every refused member with
    its reason and version but no generation pair."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def commit_all(self, *args: object, **kwargs: object) -> object:
        return _strip_batch(self._inner.commit_all(*args, **kwargs))  # type: ignore[arg-type]


class _PairlessFenceRaiseRegistry:
    """A degraded backend whose fenced persist still raises
    :class:`StaleReadGeneration`, with the same message, but without the pair."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def set_artifact_and_content(self, *args: object, **kwargs: object) -> None:
        try:
            self._inner.set_artifact_and_content(*args, **kwargs)  # type: ignore[arg-type]
        except StaleReadGeneration as exc:
            raise StaleReadGeneration(*exc.args) from None


class _FenceOnlyPairRegistry:
    """A degraded backend that reports the pair on ``stale_read_generation``
    only, at both commit entry points: the reading under which the pair is a
    fence detail rather than part of every refusal."""

    _KEEP = frozenset({STALE_READ_GENERATION_REASON})

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def commit_cas(self, *args: object, **kwargs: object) -> CasResult:
        result = self._inner.commit_cas(*args, **kwargs)  # type: ignore[arg-type]
        if isinstance(result, ConflictDetail) and result.reason not in self._KEEP:
            return _without_pair(result)
        return result

    def commit_all(self, *args: object, **kwargs: object) -> object:
        return _strip_batch(self._inner.commit_all(*args, **kwargs), keep=self._KEEP)  # type: ignore[arg-type]


class _OneStoreFactory:
    """A :class:`RegistryFactory` minting one process-scoped instance of
    ``stub`` (the shape of :class:`_DegradedFactory`, for any stub class)."""

    def __init__(self, stub: type) -> None:
        self._stub = stub
        self._reg: object | None = None

    def __call__(self) -> object:
        if self._reg is None:
            self._reg = self._stub()
        return self._reg

    def close_all(self) -> None:
        return None

    @property
    def db_path(self) -> Path | None:
        return None


def test_pairless_cas_stub_fails_commit_cas_pair_scenario() -> None:
    """THE TEETH for the ``commit_cas`` refusal-pair scenario. A binding that
    refuses the right commit with the right reason but drops the pair MUST FAIL
    the kit; the failure names the entry point, the reason and both fields."""
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_commit_cas_refusals_carry_the_generation_pair(_OneStoreFactory(_PairlessCasRegistry))
    message = str(excinfo.value)
    assert "commit_cas's stale_read_generation refusal DROPPED the generation pair" in message
    assert "captured_generation=2, owner_generation=3" in message
    assert "captured_generation=None, owner_generation=None" in message


def test_pairless_batch_stub_fails_commit_all_pair_scenario() -> None:
    """THE TEETH for the ``commit_all`` refusal-pair scenario: a batch that names
    its refused members without their pairs MUST FAIL the kit."""
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_commit_all_refusals_carry_each_members_generation_pair(
            _OneStoreFactory(_PairlessBatchRegistry)
        )
    message = str(excinfo.value)
    assert (
        "commit_all's stale_read_generation refusal for the fenced member DROPPED the "
        "generation pair" in message
    )
    assert "captured_generation=2, owner_generation=3" in message


def test_pairless_fence_raise_stub_fails_fenced_persist_pair_scenario() -> None:
    """THE TEETH for the fenced-persist refusal-pair scenario: a raise that keeps
    its type and message but drops the pair MUST FAIL the kit."""
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_fenced_persist_raise_carries_the_generation_pair(
            _OneStoreFactory(_PairlessFenceRaiseRegistry)
        )
    message = str(excinfo.value)
    assert (
        "set_artifact_and_content's StaleReadGeneration raise DROPPED the generation pair"
        in message
    )
    assert "captured_generation=2, owner_generation=3" in message


def test_fence_only_pair_stub_fails_the_other_reasons() -> None:
    """A binding that fills the pair only on the fence reason MUST FAIL both the
    ``other_holder`` check (``commit_cas``) and the claim-holding
    ``version_mismatch`` check (``commit_all``). Each scenario checks its fence
    refusal first, so these failures also show the stub's fence leg passed:
    the kit fails it for the reasons it drops, not for the one it keeps."""
    factory: RegistryFactory = _OneStoreFactory(_FenceOnlyPairRegistry)
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_commit_cas_refusals_carry_the_generation_pair(factory)
    assert "commit_cas's other_holder refusal DROPPED the generation pair" in str(excinfo.value)

    with pytest.raises(AssertionError) as excinfo:
        kit.assert_commit_all_refusals_carry_each_members_generation_pair(
            _OneStoreFactory(_FenceOnlyPairRegistry)
        )
    assert (
        "commit_all's version_mismatch refusal for the overtaken member DROPPED the "
        "generation pair" in str(excinfo.value)
    )


class _InvalidOnlyBumpRegistry:
    """A degraded backend that moves ``owner_generation`` on leaving EXCLUSIVE
    or MODIFIED under a bump trigger only when the holder lands INVALID. A
    holder that gives back its own write grant and keeps a read lands SHARED,
    so its claim is revoked with the epoch left behind. Everything else is the
    real in-memory registry."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def set_agent_state(self, artifact_id: UUID, agent_id: UUID, state: MESIState, **kwargs: object) -> None:
        record = self._inner._records[artifact_id]  # noqa: SLF001 - test stub reaches in
        epoch = record.owner_generation
        self._inner.set_agent_state(artifact_id, agent_id, state, **kwargs)  # type: ignore[arg-type]
        # THE INJECTED BUG: the real registry bumped on leaving M/E whatever the
        # target; this stub takes the bump back unless the holder landed INVALID.
        if state != MESIState.INVALID:
            record.owner_generation = epoch


def test_invalid_only_bump_stub_fails_the_release_to_reader_leg() -> None:
    """THE TEETH for the release-to-reader leg. A backend that moves the epoch
    only when a released holder lands INVALID MUST FAIL the voluntary-release
    scenario: a holder that gives back its write grant and keeps a read would
    have its commit on that claim admitted at a version that never moved. The
    scenario checks the release to INVALID first, so a failure naming a
    release-to-reader route also shows the stub's INVALID leg passed: the kit
    fails it for the SHARED landing it skips, not for the one it bumps on."""
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_epoch_moves_on_voluntary_release(_OneStoreFactory(_InvalidOnlyBumpRegistry))
    message = str(excinfo.value)
    assert "set_agent_state from EXCLUSIVE" in message, message
    assert "landing SHARED instead of INVALID does not exempt it" in message, message


# The gap a binding leaves when it decides a refusal, then reads the pair in a
# second step. A real (GIL-releasing) sleep, so the race's mutator runs inside
# it, as in _TornPairRegistry.
_FOLLOW_UP_WINDOW_SECONDS = 0.0005

_OWNER_LEG, _BOTH_LEGS = "owner", "both"
_CAS, _BATCH, _FENCE = "commit_cas", "commit_all", "fence"


class _FollowUpReadRegistry:
    """A degraded backend that decides every refusal on the right snapshot but
    fills the pair from a second read after it, at ONE entry point:
    ``commit_cas``, ``commit_all`` per member, or the fenced persist's raise.
    With ``legs="owner"`` only ``owner_generation`` is re-read, and the decided
    ``captured_generation`` is kept; with ``legs="both"`` both are re-read.
    Reason, version and message stay the decision's. Everything else is the
    real in-memory registry, so the sequential refusal-pair scenarios pass it:
    only a race can see the re-read."""

    def __init__(self, entry: str, legs: str) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)
        self._entry, self._legs = entry, legs

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def _reread(self, artifact_id: UUID, agent_id: UUID, decided: int | None) -> tuple[int | None, int | None]:
        owner = self._inner.get_owner_generation(artifact_id)
        captured = self._inner.get_read_generation(artifact_id, agent_id) if self._legs == _BOTH_LEGS else decided
        # The no-claim rule still holds on the re-read: no claim, no pair.
        return (None, None) if captured is None else (captured, owner)

    def _refill(self, detail: ConflictDetail, artifact_id: UUID, agent_id: UUID) -> ConflictDetail:
        captured, owner = self._reread(artifact_id, agent_id, detail.captured_generation)
        return ConflictDetail(
            detail.reason, detail.current_version, captured_generation=captured, owner_generation=owner
        )

    def commit_cas(self, artifact_id: UUID, agent_id: UUID, **kwargs: object) -> CasResult:
        result = self._inner.commit_cas(artifact_id, agent_id, **kwargs)  # type: ignore[arg-type]
        if self._entry != _CAS or not isinstance(result, ConflictDetail):
            return result
        time.sleep(_FOLLOW_UP_WINDOW_SECONDS)
        return self._refill(result, artifact_id, agent_id)

    def commit_all(self, agent_id: UUID, writes: object, **kwargs: object) -> object:
        result = self._inner.commit_all(agent_id, writes, **kwargs)  # type: ignore[arg-type]
        if self._entry != _BATCH or not isinstance(result, MultiCommitConflict):
            return result
        time.sleep(_FOLLOW_UP_WINDOW_SECONDS)
        return MultiCommitConflict(
            per_artifact={
                art: self._refill(detail, art, agent_id) for art, detail in result.per_artifact.items()
            }
        )

    def set_artifact_and_content(self, artifact_id: UUID, *args: object, **kwargs: object) -> None:
        try:
            self._inner.set_artifact_and_content(artifact_id, *args, **kwargs)  # type: ignore[arg-type]
        except StaleReadGeneration as exc:
            if self._entry != _FENCE:
                raise
            time.sleep(_FOLLOW_UP_WINDOW_SECONDS)
            fenced = kwargs["fence_agent_id"]
            captured, owner = self._reread(artifact_id, fenced, exc.captured_generation)  # type: ignore[arg-type]
            raise StaleReadGeneration(
                *exc.args, captured_generation=captured, owner_generation=owner
            ) from None


_RACES = {
    _CAS: kit.assert_commit_refusal_pairs_come_from_the_decision_snapshot,
    _BATCH: kit.assert_commit_refusal_pairs_come_from_the_decision_snapshot,
    _FENCE: kit.assert_fenced_persist_raise_pair_comes_from_the_decision_snapshot,
}
_ENTRY_IN_MESSAGE = {
    _CAS: "commit_cas's",
    _BATCH: "commit_all's",
    _FENCE: "set_artifact_and_content's StaleReadGeneration raise",
}


@pytest.mark.parametrize("legs", [_OWNER_LEG, _BOTH_LEGS])
@pytest.mark.parametrize("entry", [_CAS, _BATCH, _FENCE])
def test_follow_up_read_stub_fails_its_race(entry: str, legs: str) -> None:
    """THE TEETH for the snapshot rule. A binding that refuses correctly but reads the pair
    after the decision MUST FAIL the race for its entry point (the commit race for the
    two commit routes, the fence race for the fenced persist), as a snapshot violation naming
    that entry point, never as a starved run."""
    factory: RegistryFactory = _OneStoreFactory(partial(_FollowUpReadRegistry, entry, legs))  # type: ignore[arg-type]
    with pytest.raises(AssertionError) as excinfo:
        _RACES[entry](factory)
    message = str(excinfo.value)
    assert kit.SNAPSHOT_VIOLATION in message, message
    assert _ENTRY_IN_MESSAGE[entry] in message, message
    assert kit.STARVED_RACE not in message


def test_follow_up_read_stub_passes_the_sequential_pair_scenarios() -> None:
    """The re-read is invisible without concurrency: each follow-up-read stub
    passes its entry point's sequential refusal-pair scenario, so the races
    above are what catch it."""
    sequential = {
        _CAS: kit.assert_commit_cas_refusals_carry_the_generation_pair,
        _BATCH: kit.assert_commit_all_refusals_carry_each_members_generation_pair,
        _FENCE: kit.assert_fenced_persist_raise_carries_the_generation_pair,
    }
    for entry, scenario in sequential.items():
        for legs in (_OWNER_LEG, _BOTH_LEGS):
            scenario(_OneStoreFactory(partial(_FollowUpReadRegistry, entry, legs)))  # type: ignore[arg-type]


# The thread each race's actor runs in, by the name of its mutator's thread.
_ACTOR_OF_MUTATOR = {
    kit.COMMIT_RACE_MUTATOR_THREAD: kit.COMMIT_RACE_COMMITTER_THREAD,
    kit.FENCE_RACE_MUTATOR_THREAD: kit.FENCE_RACE_ZOMBIE_THREAD,
}


class _StarvedMutatorRegistry:
    """A correct backend on a runner that never schedules a race's mutator
    while the actor runs: the mutator's first move waits for the actor's
    thread to finish, so every refusal lands on the race's starting state."""

    def __init__(self) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def set_agent_state(self, *args: object, **kwargs: object) -> None:
        actor = _ACTOR_OF_MUTATOR.get(threading.current_thread().name)
        for thread in threading.enumerate():
            if thread.name == actor:
                thread.join()
        self._inner.set_agent_state(*args, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("race", [_CAS, _FENCE], ids=["commit_race", "fence_race"])
def test_starved_race_fails_as_starved_not_as_a_snapshot_violation(race: str) -> None:
    """A run whose mutator never ran beside the actor proves nothing, so it
    must FAIL rather than pass, and with its own message: a correct backend
    reported as a snapshot violation would send its author after a bug it does not
    have."""
    with pytest.raises(AssertionError) as excinfo:
        _RACES[race](_OneStoreFactory(_StarvedMutatorRegistry))
    message = str(excinfo.value)
    assert kit.STARVED_RACE in message, message
    assert kit.SNAPSHOT_VIOLATION not in message


# Every per-identity map an in-memory record keeps: one row's worth of state.
_PER_AGENT_MAPS = (
    "state_by_agent", "transient_by_agent", "transient_tick_by_agent", "granted_at_tick_by_agent",
    "last_reclamation_by_agent", "read_generation_by_agent", "last_observed_version_by_agent",
)


def _drop_row(record: object, agent_id: UUID) -> None:
    for name in _PER_AGENT_MAPS:
        getattr(record, name).pop(agent_id, None)


def _reclaim_rows(inner: ArtifactRegistry, agent_id: UUID) -> list:
    records = inner._records.values()  # noqa: SLF001 - test stub reaches in
    return [record for record in records if agent_id in record.last_reclamation_by_agent]


def _forget_unchecked(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """The delete with no checks: every row but a writer's, and the heartbeat."""
    doomed = [
        record for record in inner._records.values()  # noqa: SLF001 - test stub reaches in
        if agent_id in record.state_by_agent and record.state_by_agent[agent_id] not in _M_OR_E
    ]
    for record in doomed:
        _drop_row(record, agent_id)
    inner._heartbeat_by_agent.pop(agent_id, None)  # noqa: SLF001 - test stub reaches in
    return len(doomed)


def _forget_dropping_reclaim_rows(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """Forgets correctly, then deletes the rows that record a reclaim as well."""
    reclaim_rows = _reclaim_rows(inner, agent_id)
    removed = inner.forget_agent(agent_id)
    for record in reclaim_rows:
        _drop_row(record, agent_id)
    return removed


def _forget_keeping_a_reader(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """Forgets correctly, then puts back the state each kept reclaim row had,
    so an identity that read the file again stays a reader."""
    prior = [(record, record.state_by_agent.get(agent_id)) for record in _reclaim_rows(inner, agent_id)]
    removed = inner.forget_agent(agent_id)
    for record, state in prior:
        if state is not None:
            record.state_by_agent[agent_id] = state
    return removed


def _forget_after_a_partial_delete(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """Refuses a writer, but only after deleting its other rows and its heartbeat."""
    try:
        return inner.forget_agent(agent_id)
    except ValueError:
        _forget_unchecked(inner, agent_id)
        raise


def _forget_keeping_the_heartbeat(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """Forgets correctly, then puts the heartbeat back."""
    heartbeats = inner._heartbeat_by_agent  # noqa: SLF001 - test stub reaches in
    tick = heartbeats.get(agent_id)
    removed = inner.forget_agent(agent_id)
    if tick is not None:
        heartbeats[agent_id] = tick
    return removed


def _forget_counting_kept_rows(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """Forgets correctly, but counts the reclaim rows it keeps as removed too."""
    kept = len(_reclaim_rows(inner, agent_id))
    return inner.forget_agent(agent_id) + kept


def _forget_by_file(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """Forgets correctly, then deletes every other row on the files whose row
    it deleted: a delete keyed by file instead of by identity."""
    files = [
        record for record in inner._records.values()  # noqa: SLF001 - test stub reaches in
        if agent_id in record.state_by_agent and agent_id not in record.last_reclamation_by_agent
    ]
    removed = inner.forget_agent(agent_id)
    for record in files:
        for other in list(record.state_by_agent):
            _drop_row(record, other)
    return removed


def _forget_refusing_beside_a_writer(inner: ArtifactRegistry, agent_id: UUID) -> int:
    """Refuses while ANY identity holds a write grant on a file this one has a
    row on, so a peer's grant blocks the forget."""
    for artifact_id, record in inner._records.items():  # noqa: SLF001 - test stub reaches in
        if agent_id in record.state_by_agent and _M_OR_E.intersection(record.state_by_agent.values()):
            raise forget_refusal(agent_id, artifact_id)
    return inner.forget_agent(agent_id)


# Each forget variant by the obligation it drops.
_FORGET_VARIANTS = {
    "no_refusal": _forget_unchecked,
    "no_reclaim_row": _forget_dropping_reclaim_rows,
    "reader_kept": _forget_keeping_a_reader,
    "partial_refusal": _forget_after_a_partial_delete,
    "heartbeat_kept": _forget_keeping_the_heartbeat,
    "miscounted": _forget_counting_kept_rows,
    "peer_touched": _forget_by_file,
    "peer_refused": _forget_refusing_beside_a_writer,
}


class _ForgetDroppingRegistry:
    """A degraded backend whose ``forget_agent`` drops ONE of its obligations:
    it runs the :data:`_FORGET_VARIANTS` entry named ``dropped``, whose
    docstring says which. Everything else is the real in-memory registry."""

    def __init__(self, dropped: str) -> None:
        self._inner = ArtifactRegistry(retain_versions=True)
        self._forget = _FORGET_VARIANTS[dropped]

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def forget_agent(self, agent_id: UUID) -> int:
        return self._forget(self._inner, agent_id)


@pytest.mark.parametrize(
    ("dropped", "named"),
    [
        ("no_refusal", kit.FORGET_REFUSED_WHILE_WRITING),
        ("no_reclaim_row", kit.FORGET_DROPPED_RECLAIM_ROW),
        ("reader_kept", kit.FORGET_KEPT_A_READER),
        ("partial_refusal", kit.FORGET_CHANGED_WHILE_REFUSING),
        ("heartbeat_kept", kit.FORGET_KEPT_THE_HEARTBEAT),
        ("miscounted", kit.FORGET_MISCOUNTED),
        ("peer_touched", kit.FORGET_TOUCHED_A_PEER),
        ("peer_refused", kit.FORGET_REFUSED_FOR_A_PEER),
    ],
)
def test_forget_dropping_stub_fails_the_forget_scenario(dropped: str, named: str) -> None:
    """THE TEETH for the forget scenario. A backend that drops any one of the
    forget's obligations MUST FAIL the kit, and the failure names the one it
    dropped: forgetting past a held write grant, deleting the row that records
    a reclaim, leaving that row as a reader, deleting before a refusal,
    keeping the heartbeat, miscounting, touching a peer's row, or refusing on
    a peer's write grant."""
    factory: RegistryFactory = _OneStoreFactory(partial(_ForgetDroppingRegistry, dropped))  # type: ignore[arg-type]
    with pytest.raises(AssertionError) as excinfo:
        kit.assert_forget_agent_forgets_only_an_ended_identity(factory)
    assert named in str(excinfo.value), str(excinfo.value)


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
