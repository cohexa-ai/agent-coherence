# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The caller principal: store, mint gate and validator (caller-principal plan, U4).

The coordinator authenticates the WORKSPACE (one bearer secret), not the caller:
the acting identity is a ``session_id`` the request body names, validated for
UUID shape only. A caller principal is a coordinator-issued value bound to ONE
acting identity on the first claim of that identity, so a request naming the
identity can be checked against it. This is accident-resistance and
attributability under the same-OS-user cooperative-trust model: bearer
possession stays full authority by design (``adapters/claude_code/auth.py``),
and nothing here is, or is described as, a boundary against a process that can
read ``.coherence/``.

Every behavioural scenario runs against BOTH registries through the shared
``registry`` fixture's two arms — the parametrization is the parity run. The
facts that belong to one registry sit in their own sections: the versioned
migration step and restart durability (sqlite), and the declared restart loss
(in-memory).

Scenarios:

- first claim mints; a later claim naming the same identity without the
  binding's mint nonce (a different nonce, or none) does not become it (R2);
- a lost mint response is recovered by the same caller presenting the same
  nonce, and nothing else re-obtains the principal (R20, KTD11);
- the validator refuses an absent principal and a foreign one under
  distinguishable reasons, with a control showing the right one accepted (R1);
- attribution's SESSION component is verified by the principal, the composite
  agent id and its caller-asserted subagent component are unchanged (R3);
- the principal outlives a grant reclamation AND a session-liveness sweep, and
  never occupies a snapshot-session slot (R4, KTD1, KTD13);
- the binding cache keeps the store's UNBOUND answer until this service binds
  the identity, and a ``cached_only`` lookup never reads the store (U6);
- the store is a versioned schema step landing as the final chain step, with
  the previous step stamping its own literal (KTD1).
"""

from __future__ import annotations

import secrets
import sqlite3
import threading
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.adapters.claude_code.coordinator_server import (
    PresentedCaller,
    caller_principal_identity,
    session_to_agent_id,
)
from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import (
    CallerPrincipalUncached,
    CoordinatorService,
    SessionCapsConfig,
    mint_nonce_problem,
)
from ccs.coordinator.sqlite_registry import (
    SCHEMA_USER_VERSION,
    CrossRuntimeSchemaError,
    SqliteArtifactRegistry,
)
from ccs.core.exceptions import (
    CALLER_PRINCIPAL_ABSENT_REASON,
    CALLER_PRINCIPAL_CLAIMED_REASON,
    CALLER_PRINCIPAL_FOREIGN_REASON,
    CALLER_PRINCIPAL_REASONS,
    CALLER_PRINCIPAL_REFUSAL_REASONS,
    HOLD_REASONS,
    SESSION_CAP_EXCEEDED_REASON,
    CallerPrincipalRefused,
)
from ccs.core.states import MESIState
from ccs.core.types import SnapshotSession, VersionedReadRejection

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nonce() -> str:
    """A client-side mint nonce, generated the way a claiming client does."""
    return secrets.token_urlsafe(32)


def _sid() -> str:
    """A fresh UUID-shaped session id (the wire's acting identity)."""
    return str(uuid4())


@pytest.fixture
def svc(registry) -> CoordinatorService:
    return CoordinatorService(registry)


def _claim(svc: CoordinatorService, identity: UUID, nonce: str) -> str:
    return svc.claim_caller_principal(identity=identity, mint_nonce=nonce)


def _refusal(svc: CoordinatorService, identity: UUID, principal: object) -> str:
    """The typed reason a validation refuses with; fails if it admits."""
    with pytest.raises(CallerPrincipalRefused) as refused:
        svc.validate_caller_principal(identity=identity, principal=principal)
    return refused.value.reason


def _last_writer(registry, artifact_id: UUID) -> UUID | None:
    """The recorded ``last_writer_id``. The in-memory registry has no public
    accessor for it (``last_writer_for`` is on the sqlite-extended surface), so
    that arm reads the record the accessor would read."""
    if isinstance(registry, SqliteArtifactRegistry):
        return registry.last_writer_for(artifact_id)
    return registry._records[artifact_id].last_writer  # noqa: SLF001


def _write_attributed(svc: CoordinatorService, caller: PresentedCaller, name: str) -> UUID:
    """Acquire + commit ``name`` as ``caller``; return the artifact id. The
    agent id the write rides is the one ``caller`` verifies — nothing else."""
    agent_id = caller.attributed_agent_id(svc)
    artifact = svc.register_artifact(name=name, content="v1")
    svc.write(agent_id=agent_id, artifact_id=artifact.id, issued_at_tick=1)
    svc.commit(agent_id=agent_id, artifact_id=artifact.id, content="v2", issued_at_tick=2)
    return artifact.id


# ---------------------------------------------------------------------------
# R2 — first claim wins; a later claim without the nonce is not that identity
# ---------------------------------------------------------------------------


def test_first_claim_mints_and_a_second_claim_without_the_nonce_is_refused(
    svc: CoordinatorService, registry
) -> None:
    identity = uuid4()
    principal = _claim(svc, identity, _nonce())
    assert isinstance(principal, str) and len(principal) >= 43

    with pytest.raises(CallerPrincipalRefused) as refused:
        _claim(svc, identity, _nonce())
    assert refused.value.reason == CALLER_PRINCIPAL_CLAIMED_REASON
    # The later claimant did not become the identity: the binding is still the
    # first claimant's, and the first claimant's principal still validates.
    assert registry.get_caller_principal(identity) == principal
    svc.validate_caller_principal(identity=identity, principal=principal)


def test_a_claim_presenting_no_nonce_is_refused_and_binds_nothing(
    svc: CoordinatorService, registry
) -> None:
    """A nonce-less claim is refused on an UNCLAIMED identity too: a binding
    with no nonce could never be recovered after a lost response (R20), so it
    is not created at all."""
    identity = uuid4()
    for missing in (None, ""):
        with pytest.raises(ValueError, match="mint_nonce"):
            svc.claim_caller_principal(identity=identity, mint_nonce=missing)
    assert registry.get_caller_principal(identity) is None


def test_a_claim_with_no_nonce_on_a_bound_identity_does_not_become_it(
    svc: CoordinatorService, registry
) -> None:
    identity = uuid4()
    principal = _claim(svc, identity, _nonce())
    with pytest.raises(ValueError, match="mint_nonce"):
        svc.claim_caller_principal(identity=identity, mint_nonce=None)
    assert registry.get_caller_principal(identity) == principal


@pytest.mark.parametrize(
    ("nonce", "usable"),
    [
        ("a" * 15, False),  # one below the floor
        ("a" * 16, True),  # the floor
        ("a" * 128, True),  # the ceiling
        ("a" * 129, False),  # one above the ceiling
        ("a" * 15 + "!", False),  # right length, outside the url-safe alphabet
        ("a" * 16 + "\n", False),  # a trailing newline must not slip past `$`
        (12345678901234567890, False),  # not a string at all
    ],
)
def test_mint_nonce_shape_is_pinned_on_both_sides_of_each_bound(
    nonce: object, usable: bool
) -> None:
    assert (mint_nonce_problem(nonce) is None) is usable


# ---------------------------------------------------------------------------
# R20 / KTD11 — the same nonce recovers a lost mint; nothing else does
# ---------------------------------------------------------------------------


def test_a_lost_mint_response_is_recovered_by_the_same_nonce(
    svc: CoordinatorService, registry
) -> None:
    identity = uuid4()
    nonce = _nonce()
    discarded = _claim(svc, identity, nonce)  # the response the caller never saw

    recovered = _claim(svc, identity, nonce)

    assert recovered == discarded
    assert registry.get_caller_principal(identity) == recovered
    svc.validate_caller_principal(identity=identity, principal=recovered)


def test_after_a_lost_response_a_different_nonce_still_does_not_become_it(
    svc: CoordinatorService,
) -> None:
    """The retry path must not widen the gate: the recovery is keyed on the
    nonce the first claimant persisted, so a second claimant cannot use a lost
    response as a window to take the identity."""
    identity = uuid4()
    nonce = _nonce()
    principal = _claim(svc, identity, nonce)
    with pytest.raises(CallerPrincipalRefused):
        _claim(svc, identity, _nonce())
    assert _claim(svc, identity, nonce) == principal


def test_concurrent_first_claims_bind_one_principal(
    svc: CoordinatorService, registry
) -> None:
    """Twelve claimants race for one unclaimed identity, each with its own
    nonce, released together by a barrier. Exactly one becomes the identity;
    every other claim is refused, and the store holds the winner's principal."""
    identity = uuid4()
    claimants = 12
    barrier = threading.Barrier(claimants)
    won: list[str] = []
    refused: list[str] = []
    lock = threading.Lock()

    def claim() -> None:
        nonce = _nonce()
        barrier.wait()
        try:
            principal = _claim(svc, identity, nonce)
        except CallerPrincipalRefused as exc:
            with lock:
                refused.append(exc.reason)
        else:
            with lock:
                won.append(principal)

    threads = [threading.Thread(target=claim) for _ in range(claimants)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(won) == 1, won
    assert refused == [CALLER_PRINCIPAL_CLAIMED_REASON] * (claimants - 1)
    assert registry.get_caller_principal(identity) == won[0]


def test_one_nonce_does_not_carry_across_identities(svc: CoordinatorService) -> None:
    """A nonce proves a retry of ONE claim. Reusing it for another identity
    mints that identity its own, different principal."""
    nonce = _nonce()
    first = _claim(svc, uuid4(), nonce)
    second = _claim(svc, uuid4(), nonce)
    assert first != second


# ---------------------------------------------------------------------------
# R1 — the validator: absent and foreign refused, the matching one admitted
# ---------------------------------------------------------------------------


def test_a_peer_naming_another_identity_without_its_principal_is_refused(
    svc: CoordinatorService,
) -> None:
    """The #188 reproduction at the validator — the service level, which is the
    surface this runs on. Session B names session A's identity. B holds a
    principal of its own and deliberately does NOT read A's stored principal:
    on the hook surface, where a principal is read back from the same ``0700``
    directory the identity names, that abstention is the entire content of the
    guarantee (KTD5). What the validator buys is that naming A is no longer
    enough to act as A."""
    sid_a, sid_b = _sid(), _sid()
    principal_a = _claim(svc, caller_principal_identity(sid_a), _nonce())
    principal_b = _claim(svc, caller_principal_identity(sid_b), _nonce())
    identity_a = caller_principal_identity(sid_a)

    assert _refusal(svc, identity_a, None) == CALLER_PRINCIPAL_ABSENT_REASON
    assert _refusal(svc, identity_a, "") == CALLER_PRINCIPAL_ABSENT_REASON
    assert _refusal(svc, identity_a, principal_b) == CALLER_PRINCIPAL_FOREIGN_REASON

    # Control: the matching principal is admitted, for both identities — so the
    # refusals above are about the MISMATCH, not a validator that refuses all.
    svc.validate_caller_principal(identity=identity_a, principal=principal_a)
    svc.validate_caller_principal(
        identity=caller_principal_identity(sid_b), principal=principal_b
    )


def test_a_well_formed_principal_is_foreign_to_every_identity_but_its_own(
    svc: CoordinatorService,
) -> None:
    identity = uuid4()
    principal = _claim(svc, identity, _nonce())
    lookalike = secrets.token_urlsafe(32)  # same shape, never minted
    assert len(lookalike) == len(principal)

    assert _refusal(svc, identity, lookalike) == CALLER_PRINCIPAL_FOREIGN_REASON
    assert _refusal(svc, identity, principal[:-1]) == CALLER_PRINCIPAL_FOREIGN_REASON
    assert _refusal(svc, identity, principal + "A") == CALLER_PRINCIPAL_FOREIGN_REASON
    assert _refusal(svc, identity, 42) == CALLER_PRINCIPAL_FOREIGN_REASON
    svc.validate_caller_principal(identity=identity, principal=principal)


def test_an_unclaimed_identity_admits_no_principal(svc: CoordinatorService) -> None:
    """No binding means no principal matches: a principal minted for some other
    identity is foreign here, and presenting none is absent."""
    other = _claim(svc, uuid4(), _nonce())
    unclaimed = uuid4()
    assert _refusal(svc, unclaimed, other) == CALLER_PRINCIPAL_FOREIGN_REASON
    assert _refusal(svc, unclaimed, None) == CALLER_PRINCIPAL_ABSENT_REASON


def test_refusal_reasons_are_distinct_and_never_a_hold() -> None:
    """KTD9: a principal refusal is a client error, never a hold — a hold
    invites a retry, and no retry supplies a principal the caller never had."""
    assert CALLER_PRINCIPAL_REASONS == {
        CALLER_PRINCIPAL_ABSENT_REASON,
        CALLER_PRINCIPAL_FOREIGN_REASON,
        CALLER_PRINCIPAL_CLAIMED_REASON,
    }
    assert len(CALLER_PRINCIPAL_REASONS) == 3
    assert CALLER_PRINCIPAL_REASONS.isdisjoint(HOLD_REASONS)


def test_a_route_refusal_carries_exactly_the_two_route_reasons() -> None:
    """The typed ``reason`` a route's 400 carries — what a client classifies a
    refusal by, and what its recovery (re-claim with its own nonce, retry once)
    keys on — is absent or foreign, never the mint's ``claimed``. Pinned against
    a frozen literal AND against the coordinator's refusal table, so a refusal
    the gate can send cannot fall outside the set a client matches."""
    from ccs.adapters.claude_code.coordinator_server import _CALLER_PRINCIPAL_ERRORS

    assert CALLER_PRINCIPAL_REFUSAL_REASONS == {
        "caller_principal_absent", "caller_principal_foreign",
    }
    assert set(_CALLER_PRINCIPAL_ERRORS) == CALLER_PRINCIPAL_REFUSAL_REASONS
    assert CALLER_PRINCIPAL_REFUSAL_REASONS < CALLER_PRINCIPAL_REASONS
    assert CALLER_PRINCIPAL_CLAIMED_REASON not in CALLER_PRINCIPAL_REFUSAL_REASONS
    assert CALLER_PRINCIPAL_REFUSAL_REASONS.isdisjoint(HOLD_REASONS)


# ---------------------------------------------------------------------------
# R3 — attribution: the session component is what the principal verifies
# ---------------------------------------------------------------------------


def test_a_write_with_a_valid_principal_records_the_composite_writer(
    svc: CoordinatorService, registry
) -> None:
    sid = _sid()
    principal = _claim(svc, caller_principal_identity(sid), _nonce())
    caller = PresentedCaller(session_id=sid, subagent_id=None, principal=principal)

    artifact_id = _write_attributed(svc, caller, "plan.md")

    assert _last_writer(registry, artifact_id) == session_to_agent_id(sid)


def test_a_different_principal_records_a_different_writer(
    svc: CoordinatorService, registry
) -> None:
    """The companion to the test above: a second identity's principal records
    that identity's composite id — so the assertion above cannot pass on a
    recorded constant."""
    sid_a, sid_b = _sid(), _sid()
    principal_a = _claim(svc, caller_principal_identity(sid_a), _nonce())
    principal_b = _claim(svc, caller_principal_identity(sid_b), _nonce())

    written_a = _write_attributed(
        svc, PresentedCaller(session_id=sid_a, subagent_id=None, principal=principal_a), "a.md"
    )
    written_b = _write_attributed(
        svc, PresentedCaller(session_id=sid_b, subagent_id=None, principal=principal_b), "b.md"
    )

    assert _last_writer(registry, written_a) == session_to_agent_id(sid_a)
    assert _last_writer(registry, written_b) == session_to_agent_id(sid_b)
    assert _last_writer(registry, written_a) != _last_writer(registry, written_b)


def test_a_caller_naming_a_peer_session_gets_no_attributed_identity(
    svc: CoordinatorService,
) -> None:
    """Presenting its own principal while naming a peer's session does not
    yield the peer's composite id — attribution never reaches the write."""
    sid_a, sid_b = _sid(), _sid()
    _claim(svc, caller_principal_identity(sid_a), _nonce())
    principal_b = _claim(svc, caller_principal_identity(sid_b), _nonce())
    impostor = PresentedCaller(session_id=sid_a, subagent_id=None, principal=principal_b)
    with pytest.raises(CallerPrincipalRefused) as refused:
        impostor.attributed_agent_id(svc)
    assert refused.value.reason == CALLER_PRINCIPAL_FOREIGN_REASON


def test_the_binding_is_readable_through_a_public_predicate(
    svc: CoordinatorService,
) -> None:
    """``is_caller_principal_bound`` answers whether an identity has EVER been
    claimed — the fact the require-class routes branch on (plan U6). A miss is
    cached, and this service's own claim replaces it: an identity bound after a
    False answer reads True at once."""
    identity = caller_principal_identity(_sid())
    assert svc.is_caller_principal_bound(identity) is False
    _claim(svc, identity, _nonce())
    assert svc.is_caller_principal_bound(identity) is True
    assert svc.is_caller_principal_bound(caller_principal_identity(_sid())) is False


def test_the_bound_predicate_reads_the_durable_store_after_a_restart(tmp_path: Path) -> None:
    """A binding made before a coordinator restart is still bound after it, so
    an identity a client claimed cannot become open to an unbound caller by the
    coordinator process going away (sqlite)."""
    identity = caller_principal_identity(_sid())
    with SqliteArtifactRegistry(tmp_path / "state.db") as first:
        _claim(CoordinatorService(first), identity, _nonce())
    with SqliteArtifactRegistry(tmp_path / "state.db") as second:
        assert CoordinatorService(second).is_caller_principal_bound(identity) is True


def test_attribution_admits_no_principal_only_while_the_identity_is_unbound(
    svc: CoordinatorService, registry
) -> None:
    """The version-skew rule (plan U6 / R16). A caller that has never claimed
    — an older client that predates the principal — presents none and names an
    identity nobody bound: it is attributed under the plain composite id, as
    before the principal existed. Once the identity is bound, the same
    request without a principal is refused as ABSENT (the #188 case: naming a
    newer client's session without its principal), and a principal presented
    for an unbound identity is FOREIGN, never ignored."""
    sid = _sid()
    unbound = PresentedCaller(session_id=sid, subagent_id="sub", principal=None)
    assert unbound.attributed_agent_id(svc) == session_to_agent_id(sid, "sub")
    written = _write_attributed(svc, unbound, "old-client.md")
    assert _last_writer(registry, written) == session_to_agent_id(sid, "sub")

    stray = _claim(svc, caller_principal_identity(_sid()), _nonce())
    with pytest.raises(CallerPrincipalRefused) as foreign:
        PresentedCaller(session_id=_sid(), subagent_id=None, principal=stray).attributed_agent_id(svc)
    assert foreign.value.reason == CALLER_PRINCIPAL_FOREIGN_REASON

    _claim(svc, caller_principal_identity(sid), _nonce())
    with pytest.raises(CallerPrincipalRefused) as absent:
        unbound.attributed_agent_id(svc)
    assert absent.value.reason == CALLER_PRINCIPAL_ABSENT_REASON


def test_two_subagents_of_one_session_keep_distinct_writers(
    svc: CoordinatorService, registry
) -> None:
    """The principal's unit of identity is the SESSION: both subagents present
    the parent's principal, and the caller-asserted subagent component still
    separates their composite ids, exactly as before the principal existed."""
    sid = _sid()
    principal = _claim(svc, caller_principal_identity(sid), _nonce())
    alpha = PresentedCaller(session_id=sid, subagent_id="alpha", principal=principal)
    beta = PresentedCaller(session_id=sid, subagent_id="beta", principal=principal)

    written_alpha = _write_attributed(svc, alpha, "alpha.md")
    written_beta = _write_attributed(svc, beta, "beta.md")

    assert _last_writer(registry, written_alpha) == session_to_agent_id(sid, "alpha")
    assert _last_writer(registry, written_beta) == session_to_agent_id(sid, "beta")
    assert len(
        {
            _last_writer(registry, written_alpha),
            _last_writer(registry, written_beta),
            session_to_agent_id(sid),
        }
    ) == 3


# ---------------------------------------------------------------------------
# R4 — three independent lifetimes: grant, snapshot session, principal
# ---------------------------------------------------------------------------


def test_a_grant_reclaimed_by_the_sweep_leaves_the_principal_valid(
    svc: CoordinatorService, registry
) -> None:
    sid = _sid()
    principal = _claim(svc, caller_principal_identity(sid), _nonce())
    caller = PresentedCaller(session_id=sid, subagent_id=None, principal=principal)
    holder = caller.attributed_agent_id(svc)
    artifact = svc.register_artifact(name="held.md", content="v1")
    svc.write(agent_id=holder, artifact_id=artifact.id, issued_at_tick=1)

    reclaimed = svc.enforce_stable_grant_timeouts(
        current_tick=100, heartbeat_timeout_ticks=10, max_hold_ticks=10_000
    )

    # The reclamation really happened (a vacuous sweep would prove nothing).
    assert reclaimed == 1
    assert registry.get_agent_state(artifact.id, holder) is MESIState.INVALID
    # A later request with the same principal succeeds: re-acquire and commit,
    # attributed to the same composite id.
    later = caller.attributed_agent_id(svc)
    svc.write(agent_id=later, artifact_id=artifact.id, issued_at_tick=101)
    svc.commit(agent_id=later, artifact_id=artifact.id, content="v2", issued_at_tick=102)
    assert _last_writer(registry, artifact.id) == session_to_agent_id(sid)


@pytest.mark.parametrize(("sweep_tick", "reaped"), [(9, 0), (10, 1)])
def test_a_session_sweep_past_the_age_ceiling_leaves_the_principal_row(
    registry, sweep_tick: int, reaped: int
) -> None:
    """Both sides of the absolute-age ceiling (10 ticks): one tick short the
    session survives, at the ceiling it is reaped — and on BOTH sides the
    principal ROW is present, asserted on the store, not inferred from a
    validation that could be served from the in-process cache."""
    svc = CoordinatorService(registry, session_caps=SessionCapsConfig(absolute_age_ticks=10))
    identity = uuid4()
    principal = _claim(svc, identity, _nonce())
    artifact = svc.register_artifact(name="pinned.md", content="v1")
    session = svc.begin_session(read_set=[artifact.id], owner=identity, created_at_tick=0)
    assert isinstance(session, SnapshotSession)

    count = svc.enforce_session_liveness(
        current_tick=sweep_tick, heartbeat_timeout_ticks=1_000_000
    )

    assert count == reaped
    assert registry.session_count() == 1 - reaped
    assert registry.get_caller_principal(identity) == principal


def test_releasing_a_session_leaves_the_principal_row(registry) -> None:
    """Release is the third session-table writer; with a control that the probe
    has teeth — the same release DOES drop the session's own row."""
    svc = CoordinatorService(registry)
    identity = uuid4()
    principal = _claim(svc, identity, _nonce())
    artifact = svc.register_artifact(name="released.md", content="v1")
    session = svc.begin_session(read_set=[artifact.id], owner=identity)
    assert isinstance(session, SnapshotSession)
    assert registry.get_session_meta(session.session_token) is not None

    registry.release_session(session.session_token)

    assert registry.get_session_meta(session.session_token) is None  # control
    assert registry.get_caller_principal(identity) == principal


def test_principals_are_invisible_to_the_session_enumeration_and_count(registry) -> None:
    svc = CoordinatorService(registry)
    for _ in range(5):
        _claim(svc, uuid4(), _nonce())
    assert registry.session_count() == 0
    assert registry.all_session_meta() == {}


def test_minting_principals_does_not_consume_snapshot_session_slots(registry) -> None:
    """``max_sessions=2``, many principals minted: the second session still
    opens (one below the cap) and the third is refused AT the cap — the
    threshold is pinned on both sides, and principals move it not at all."""
    svc = CoordinatorService(registry, session_caps=SessionCapsConfig(max_sessions=2))
    for _ in range(10):
        _claim(svc, uuid4(), _nonce())
    artifact = svc.register_artifact(name="capped.md", content="v1")
    owner = uuid4()

    first = svc.begin_session(read_set=[artifact.id], owner=owner)
    second = svc.begin_session(read_set=[artifact.id], owner=owner)
    third = svc.begin_session(read_set=[artifact.id], owner=owner)

    assert isinstance(first, SnapshotSession)
    assert isinstance(second, SnapshotSession)
    assert isinstance(third, VersionedReadRejection)
    assert third.reason == SESSION_CAP_EXCEEDED_REASON
    assert registry.session_count() == 2


# ---------------------------------------------------------------------------
# The registry member itself — first-claim-wins at the store
# ---------------------------------------------------------------------------


def test_bind_returns_the_bound_pair_and_never_rebinds(registry) -> None:
    identity = uuid4()
    assert registry.get_caller_principal(identity) is None

    first = registry.bind_caller_principal(identity, "principal-one", "nonce-one")
    second = registry.bind_caller_principal(identity, "principal-two", "nonce-two")

    assert first == ("principal-one", "nonce-one")
    assert second == ("principal-one", "nonce-one")
    assert registry.get_caller_principal(identity) == "principal-one"


def test_bind_keys_by_identity(registry) -> None:
    one, two = uuid4(), uuid4()
    registry.bind_caller_principal(one, "principal-one", "nonce")
    registry.bind_caller_principal(two, "principal-two", "nonce")
    assert registry.get_caller_principal(one) == "principal-one"
    assert registry.get_caller_principal(two) == "principal-two"


# ---------------------------------------------------------------------------
# Restart — the durable tier (sqlite) and the declared in-memory loss
# ---------------------------------------------------------------------------


def test_sqlite_binding_survives_a_restart_through_the_durable_tier(tmp_path: Path) -> None:
    """A fresh service over the same file has an empty in-process cache, so
    every answer below comes from the durable tier."""
    db = tmp_path / "restart.db"
    identity = uuid4()
    nonce = _nonce()
    with SqliteArtifactRegistry(db) as reg:
        principal = CoordinatorService(reg).claim_caller_principal(
            identity=identity, mint_nonce=nonce
        )

    with SqliteArtifactRegistry(db) as reg:
        fresh = CoordinatorService(reg)
        fresh.validate_caller_principal(identity=identity, principal=principal)
        assert _refusal(fresh, identity, secrets.token_urlsafe(32)) == (
            CALLER_PRINCIPAL_FOREIGN_REASON
        )
        with pytest.raises(CallerPrincipalRefused):
            fresh.claim_caller_principal(identity=identity, mint_nonce=_nonce())
        assert fresh.claim_caller_principal(identity=identity, mint_nonce=nonce) == principal


def _record_store_reads(registry, monkeypatch: pytest.MonkeyPatch) -> list[UUID]:
    """Record every durable-store read of a binding (``get_caller_principal``)."""
    reads: list[UUID] = []
    real = registry.get_caller_principal

    def recording(identity: UUID) -> str | None:
        reads.append(identity)
        return real(identity)

    monkeypatch.setattr(registry, "get_caller_principal", recording)
    return reads


def test_a_miss_is_cached_and_this_services_own_claim_replaces_it(
    svc: CoordinatorService, registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store's UNBOUND answer is cached (the negative tier): a second
    lookup reads nothing, so a client that never claims (KTD15) stops costing a
    store read per request. It holds only until THIS service binds the
    identity — the one writer of bindings in a coordinator process — and the
    claim replaces it: from the next lookup on, the identity is bound, an
    absent principal is refused as absent, and the bound one is admitted.

    Two services binding into one store is outside that model and is not what
    this cache is for: the coordinator holds the workspace lock and is the only
    process that claims."""
    identity = caller_principal_identity(_sid())
    reads = _record_store_reads(registry, monkeypatch)
    assert svc.is_caller_principal_bound(identity) is False
    assert svc.is_caller_principal_bound(identity) is False
    assert reads == [identity], "the UNBOUND answer was not cached"
    assert svc.is_caller_principal_bound(identity, cached_only=True) is False
    assert reads == [identity]

    principal = _claim(svc, identity, _nonce())
    assert svc.is_caller_principal_bound(identity) is True
    assert _refusal(svc, identity, None) == CALLER_PRINCIPAL_ABSENT_REASON
    svc.validate_caller_principal(identity=identity, principal=principal)
    assert reads == [identity], "the claim, not a re-read, replaced the cached answer"


def test_cached_only_never_reads_the_store(
    svc: CoordinatorService, registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``cached_only`` is how the coordinator's admission gate keeps the store
    read off the request thread: where the cache cannot answer it raises
    :class:`CallerPrincipalUncached` — for the bound predicate and for a
    presented principal alike — and reads nothing; an absent principal on a
    known-bound identity is still refused, since that needs no read."""
    identity = caller_principal_identity(_sid())
    reads = _record_store_reads(registry, monkeypatch)
    with pytest.raises(CallerPrincipalUncached):
        svc.is_caller_principal_bound(identity, cached_only=True)
    with pytest.raises(CallerPrincipalUncached):
        svc.validate_caller_principal(identity=identity, principal="p" * 43, cached_only=True)
    assert reads == []

    _claim(svc, identity, _nonce())
    assert svc.is_caller_principal_bound(identity, cached_only=True) is True
    with pytest.raises(CallerPrincipalRefused) as refused:
        svc.validate_caller_principal(identity=identity, principal=None, cached_only=True)
    assert refused.value.reason == CALLER_PRINCIPAL_ABSENT_REASON
    assert reads == []


def _bind_that_lands_then_raises(registry, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the store's bind commit and THEN raise — the one bind outcome the
    service cannot read back, so the claim cannot write the principal it bound
    into its cache."""
    real = registry.bind_caller_principal

    def lands_then_raises(identity: UUID, principal: str, mint_nonce: str):
        real(identity, principal, mint_nonce)
        raise RuntimeError("injected: the bind landed, then the call failed")

    monkeypatch.setattr(registry, "bind_caller_principal", lands_then_raises)


def test_a_bind_attempt_that_raises_drops_a_cached_unbound_answer(
    svc: CoordinatorService, registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raised bind does not prove nothing landed, so the cached UNBOUND
    answer is dropped and the next lookup reads the store. Kept, it would admit
    the claimed identity's absent-principal requests as an older client's."""
    identity = caller_principal_identity(_sid())
    assert svc.is_caller_principal_bound(identity) is False
    _bind_that_lands_then_raises(registry, monkeypatch)
    with pytest.raises(RuntimeError, match="injected"):
        _claim(svc, identity, _nonce())

    assert svc.is_caller_principal_bound(identity) is True
    assert _refusal(svc, identity, None) == CALLER_PRINCIPAL_ABSENT_REASON


def test_an_unbound_answer_read_before_a_bind_is_not_cached_after_it(
    svc: CoordinatorService, registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store read happens outside the service's lock, so a bind can finish
    between a lookup's read and its caching. A lookup does not cache the
    UNBOUND answer it read once a bind attempt has finished since it began. The
    interleaving is forced deterministically: the lookup's read returns
    "unbound", then a bind lands (and raises, so the claim cannot correct the
    cache itself), then the lookup resumes. The next lookup must see the
    binding; a cached stale answer would admit the claimed identity's absent-
    principal requests from then on."""
    identity = caller_principal_identity(_sid())
    real_read = registry.get_caller_principal
    raced: list[bool] = []

    def read_then_race_a_bind(read_identity: UUID) -> str | None:
        answer = real_read(read_identity)
        if not raced:
            raced.append(True)
            assert answer is None
            _bind_that_lands_then_raises(registry, monkeypatch)
            with pytest.raises(RuntimeError, match="injected"):
                _claim(svc, identity, _nonce())
        return answer

    monkeypatch.setattr(registry, "get_caller_principal", read_then_race_a_bind)
    svc.is_caller_principal_bound(identity)  # the raced lookup; its own answer is concurrent
    assert raced == [True], "the race was never forced"

    assert svc.is_caller_principal_bound(identity) is True, (
        "the UNBOUND answer read before the bind was cached after it")


def test_in_memory_binding_is_process_scoped_as_declared() -> None:
    """The in-memory registry declares restart LOSS (its tier statement in the
    backend conformance kit): a fresh instance is a fresh store, so the
    identity is unclaimed again. Asserted as the declared divergence."""
    identity = uuid4()
    principal = _claim(CoordinatorService(ArtifactRegistry()), identity, _nonce())
    fresh = CoordinatorService(ArtifactRegistry())
    assert _refusal(fresh, identity, principal) == CALLER_PRINCIPAL_FOREIGN_REASON


# ---------------------------------------------------------------------------
# The versioned schema step (sqlite only) — KTD1
# ---------------------------------------------------------------------------


def _user_version(db_path: Path) -> int:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA user_version").fetchone()[0]
    finally:
        conn.close()


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(str(db_path))
    try:
        return {
            r[0]
            for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def _principal_table_shape(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute("PRAGMA table_info(caller_principals)").fetchall()
    finally:
        conn.close()


def _revert_to_v7_shape(db_path: Path) -> None:
    """A current db rewound to what a v7 build produced: no principal table."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP TABLE IF EXISTS caller_principals")
        conn.execute("PRAGMA user_version = 7")
        conn.commit()
    finally:
        conn.close()


def _revert_to_v6_shape(db_path: Path) -> None:
    """Rewound one step further: the v7 agent_id index absent too."""
    _revert_to_v7_shape(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DROP INDEX IF EXISTS idx_agent_states_agent")
        conn.execute("PRAGMA user_version = 6")
        conn.commit()
    finally:
        conn.close()


def test_fresh_db_is_created_with_the_principal_table(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    with SqliteArtifactRegistry(db):
        pass
    assert _user_version(db) == SCHEMA_USER_VERSION
    assert "caller_principals" in _tables(db)


def test_v7_db_gains_the_principal_table_and_keeps_its_data(tmp_path: Path) -> None:
    db = tmp_path / "v7.db"
    with SqliteArtifactRegistry(db) as reg:
        artifact = CoordinatorService(reg).register_artifact(name="kept.md", content="v1")
    _revert_to_v7_shape(db)
    assert _user_version(db) == 7
    assert "caller_principals" not in _tables(db)

    with SqliteArtifactRegistry(db) as reg:
        assert reg.get_artifact(artifact.id) is not None
        identity = uuid4()
        principal = CoordinatorService(reg).claim_caller_principal(
            identity=identity, mint_nonce=_nonce()
        )
        assert reg.get_caller_principal(identity) == principal
    assert _user_version(db) == SCHEMA_USER_VERSION


def test_v6_origin_walk_lands_the_table_at_the_current_stamp(tmp_path: Path) -> None:
    """THE RE-STAMP TRAP, fourth arming. ``_migrate_v6_to_v7`` was the final
    step and stamped the constant; with the constant past 7 it must stamp its
    own literal 7, or a v6-origin db is stamped current WITHOUT the principal
    table and the chained v7->v8 loser-guard no-ops. Fails if that literal is
    reverted."""
    db = tmp_path / "v6.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v6_shape(db)
    assert _user_version(db) == 6

    with SqliteArtifactRegistry(db):
        pass

    assert _user_version(db) == SCHEMA_USER_VERSION
    assert "caller_principals" in _tables(db)


def test_upgraded_and_fresh_principal_tables_are_identical(tmp_path: Path) -> None:
    fresh, upgraded = tmp_path / "fresh.db", tmp_path / "upgraded.db"
    with SqliteArtifactRegistry(fresh):
        pass
    with SqliteArtifactRegistry(upgraded):
        pass
    _revert_to_v7_shape(upgraded)
    with SqliteArtifactRegistry(upgraded):
        pass
    assert _principal_table_shape(upgraded) == _principal_table_shape(fresh)
    assert _principal_table_shape(fresh)  # non-empty: the probe sees a table


def test_a_crash_before_the_v8_stamp_leaves_a_bootable_v7(tmp_path: Path) -> None:
    """The DDL and the stamp share one transaction: a failure between them
    rolls the table back with the stamp, and a clean reopen re-migrates."""
    db = tmp_path / "crash.db"
    with SqliteArtifactRegistry(db):
        pass
    _revert_to_v7_shape(db)

    class _Crash(RuntimeError):
        pass

    class _CrashingConn:
        def __init__(self, inner: sqlite3.Connection) -> None:
            self._inner = inner

        def execute(self, sql: str, *args):
            # The v7->v8 step stamps its own literal 8 since v9 (#191).
            if sql.strip() == "PRAGMA user_version = 8":
                raise _Crash("simulated kill before the stamp")
            return self._inner.execute(sql, *args)

        def __getattr__(self, name: str):
            return getattr(self._inner, name)

    reg = SqliteArtifactRegistry.__new__(SqliteArtifactRegistry)
    real = sqlite3.connect(str(db), isolation_level=None, check_same_thread=False)
    try:
        reg._conn = _CrashingConn(real)  # noqa: SLF001 — the seam under test
        reg._db_path = db  # noqa: SLF001
        with pytest.raises(_Crash):
            reg._migrate_v7_to_v8(None)  # noqa: SLF001
    finally:
        real.close()

    assert _user_version(db) == 7
    assert "caller_principals" not in _tables(db)
    with SqliteArtifactRegistry(db):
        pass
    assert _user_version(db) == SCHEMA_USER_VERSION
    assert "caller_principals" in _tables(db)


def test_a_v8_stamp_without_the_principal_table_is_refused(tmp_path: Path) -> None:
    """No ledger this build recognizes stamps 8 (or later) without the table,
    so the open fails closed rather than serving a store whose schema it
    cannot trust — with a control that a genuine store opens."""
    genuine, forged = tmp_path / "genuine.db", tmp_path / "forged.db"
    for db in (genuine, forged):
        with SqliteArtifactRegistry(db):
            pass
    conn = sqlite3.connect(str(forged))
    try:
        conn.execute("DROP TABLE caller_principals")
        conn.commit()
    finally:
        conn.close()
    assert _user_version(forged) == SCHEMA_USER_VERSION

    with SqliteArtifactRegistry(genuine):
        pass
    with pytest.raises(CrossRuntimeSchemaError, match="caller_principals"):
        SqliteArtifactRegistry(forged)


# ---------------------------------------------------------------------------
# Placement — pinned by test, since the architecture gate does not constrain it
# ---------------------------------------------------------------------------

_SRC = Path(__file__).resolve().parents[2] / "src" / "ccs"


def test_the_principal_store_is_a_coordinator_registry_member() -> None:
    """The store's home is the registry layer (KTD1): both registries implement
    the member in their own modules, and the table is created only in the
    sqlite registry. The architecture gate checks import direction, not
    placement, so this pin is what would notice a move."""
    assert ArtifactRegistry.bind_caller_principal.__module__ == "ccs.coordinator.registry"
    assert (
        SqliteArtifactRegistry.bind_caller_principal.__module__
        == "ccs.coordinator.sqlite_registry"
    )
    creators = sorted(
        str(path.relative_to(_SRC))
        for path in _SRC.rglob("*.py")
        if "CREATE TABLE caller_principals" in path.read_text(encoding="utf-8")
    )
    assert creators == ["coordinator/sqlite_registry.py"]


@pytest.mark.parametrize("interface_namespace", ["adapters", "cli", "mcp"])
def test_no_interface_module_owns_a_table(interface_namespace: str) -> None:
    """An interface-layer store would create its file outside the registry's
    ``0600`` creation path, and no test would notice the mode. None exists."""
    modules = sorted((_SRC / interface_namespace).rglob("*.py"))
    assert modules, f"no modules under {interface_namespace}: the sweep would see nothing"
    offenders = []
    for path in modules:
        text = path.read_text(encoding="utf-8")
        if "import sqlite3" in text or "CREATE TABLE" in text:
            offenders.append(str(path.relative_to(_SRC)))
    assert offenders == []
