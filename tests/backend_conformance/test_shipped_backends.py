# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Run the Tier-1 conformance kit over the two SHIPPED registries (plan Unit 5).

SQLite is verified at **Tier-1** (the full R9 tuple under one ``BEGIN IMMEDIATE``
transaction, plus durable restart-survival). The in-memory registry is verified
at its OWN honest declared tier: it passes every MUST-MATCH scenario (it hosts
the whole atomic boundary GIL-atomically under one process) but declares restart
**LOSS** — a fresh instance is a fresh, empty store, which is NOT a Tier-1 / HA
claim. The kit asserts the in-memory arm's declared restart-loss *as declared*,
never as a bug.

The MUST-MATCH vs BACKEND-DEFINED split is expressed in the test parametrization,
mirroring the kit API: the must-match tests take only the ``factory`` fixture and
run the SAME assertion on both arms; the backend-defined test takes the arm's
declared :class:`~tests.backend_conformance.kit.RestartDeclaration` too, and the
SAME kit function verifies each arm against ITS OWN declaration (sqlite SURVIVES,
in-memory LOST) — never against each other.

Fixtures use isolated ``tmp_path`` databases and close every handle at teardown;
they never bump ``PRAGMA user_version`` (Node-shared, currently v4) and never
write foreign lineage markers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from tests.backend_conformance import kit
from tests.backend_conformance.kit import (
    InMemoryFactory,
    RegistryFactory,
    RestartDeclaration,
    SqliteFactory,
)

# Arm id → (factory builder, declared restart disposition). The declaration is
# the arm's HONEST tier statement: sqlite re-homes session_meta/pins durably
# (SURVIVES); in-memory is process-scoped (LOST — NOT a Tier-1/HA claim).
_ARMS = {
    "in_memory": RestartDeclaration.LOST,
    "sqlite": RestartDeclaration.SURVIVES,
}


@pytest.fixture(params=sorted(_ARMS))
def factory(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[RegistryFactory]:
    """A registry factory for one arm. The sqlite arm mints handles over an
    isolated ``tmp_path`` db and closes them all at teardown; the in-memory arm
    holds no OS resource. Every scenario runs against BOTH arms via this fixture's
    parametrization — that is where the MUST-MATCH universality lives."""
    arm: str = request.param
    fac: RegistryFactory = InMemoryFactory() if arm == "in_memory" else SqliteFactory(tmp_path)
    try:
        yield fac
    finally:
        fac.close_all()


@pytest.fixture
def declared_restart(request: pytest.FixtureRequest) -> RestartDeclaration:
    """The DECLARED restart disposition for the current ``factory`` arm — resolved
    from the same parametrization so the backend-defined test verifies each arm
    against its own declaration."""
    # The active ``factory`` param is the arm id carried on the request node.
    arm = request.node.callspec.params["factory"]
    return _ARMS[arm]


# ---------------------------------------------------------------------------
# MUST-MATCH scenarios — the SAME assertion, both arms (identical property).
# ---------------------------------------------------------------------------


def test_cas_arbitration_one_winner(factory: RegistryFactory) -> None:
    kit.assert_cas_arbitration_one_winner(factory)


def test_single_writer_under_contention(factory: RegistryFactory) -> None:
    kit.assert_single_writer_under_contention(factory)


def test_fence_rejects_superseded_read_generation(factory: RegistryFactory) -> None:
    kit.assert_fence_rejects_superseded_read_generation(factory)


def test_peer_fetch_does_not_rearm_superseded_read_generation(factory: RegistryFactory) -> None:
    kit.assert_peer_fetch_does_not_rearm_superseded_read_generation(factory)


def test_epoch_moves_on_voluntary_release(factory: RegistryFactory) -> None:
    kit.assert_epoch_moves_on_voluntary_release(factory)


def test_epoch_moves_on_handoff(factory: RegistryFactory) -> None:
    kit.assert_epoch_moves_on_handoff(factory)


def test_version_and_generation_pair_is_untearable(factory: RegistryFactory) -> None:
    kit.assert_version_and_generation_pair_is_untearable(factory)


def test_fence_admits_absent_read_generation(factory: RegistryFactory) -> None:
    kit.assert_fence_admits_absent_read_generation(factory)


def test_session_fail_closed_on_foreign_and_reaped(factory: RegistryFactory) -> None:
    kit.assert_session_fail_closed_on_foreign_and_reaped(factory)


# ---------------------------------------------------------------------------
# R18 — declared liveness source matches the Unit-4 contract + observed behavior.
# ---------------------------------------------------------------------------


def test_declared_liveness_source_matches_contract(factory: RegistryFactory) -> None:
    kit.assert_declared_liveness_source_matches_contract(factory)


# ---------------------------------------------------------------------------
# BACKEND-DEFINED — each arm verified against ITS OWN declared disposition.
# (No cross-arm identity assertion: sqlite SURVIVES, in-memory LOST.)
# ---------------------------------------------------------------------------


def test_declared_restart_survival(
    factory: RegistryFactory, declared_restart: RestartDeclaration
) -> None:
    kit.assert_declared_restart_survival(factory, declared_restart)
