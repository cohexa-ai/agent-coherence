# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoordinatorService.register_workspace_restore`` — WV plan Unit 5 (R4/R5).

Service-level coverage of the restore-registration primitive, run against BOTH
registries (the fencing-suite house pattern) so the resolution split is
parity-proven:

- the durable registry resolves member paths through ``resolve_or_register``
  (SqliteExtended); the in-memory registry through the service's name-scan +
  hash-only first-observation seed — BOTH mint at version 1 carrying the
  fingerprint, and neither ever stores member bytes;
- an empty write-set answers the typed ``empty_write_set`` and NEVER calls
  ``commit_all`` (which raises on empty, by contract);
- hash-identical members are ``skipped`` (the exactly-once filter);
- a differing member commits through ONE all-or-nothing ``commit_all``:
  monotonic forward version, peers invalidated atomically, signals returned;
- HELD batches surface as the typed ``refused`` (``other_holder`` /
  ``stale_read_generation`` — the fence rejecting a superseded controller);
- the abort Event threads into the commit path (A6) and fails it closed;
- the new ``registered`` restore status round-trips both registries;
- #191: a write-set is checked against the manifest (membership AND the
  captured fingerprint) before anything resolves, the optional receiver binds
  who may register, and the first registering controller claims the
  checkpoint — a different controller is refused ``already_registered``, the
  same one retries idempotently and is told so.
"""

from __future__ import annotations

import threading
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import CheckpointMember
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    CHECKPOINT_ALREADY_REGISTERED_REASON,
    CHECKPOINT_FINGERPRINT_MISMATCH_REASON,
    CHECKPOINT_NOT_A_MEMBER_REASON,
    CHECKPOINT_NOT_THE_RECEIVER_REASON,
    CHECKPOINT_UNKNOWN_REASON,
    RESTORE_STATUS_REGISTERED,
    STALE_READ_GENERATION_REASON,
    WORKSPACE_REGISTRATION_COMMITTED,
    WORKSPACE_REGISTRATION_EMPTY,
    WORKSPACE_REGISTRATION_REFUSED,
    CheckpointRegistrationRefused,
    CheckpointUnknown,
    WatchdogAbandoned,
)
from ccs.core.states import MESIState
from ccs.core.substrate import sha256_hex
from ccs.core.types import WorkspaceRestoreWrite

OWNER = uuid4()

FP_OLD = sha256_hex(b"old-body")
FP_NEW = sha256_hex(b"restored-body")


class _CountingService(CoordinatorService):
    """Counts commit_all invocations (the never-called-on-empty proof)."""

    def __init__(self, registry) -> None:
        super().__init__(registry)
        self.commit_all_calls = 0

    def commit_all(self, **kwargs):
        self.commit_all_calls += 1
        return super().commit_all(**kwargs)


@pytest.fixture(params=["in_memory", "sqlite"])
def registry(request, tmp_path: Path):
    """Both registry implementations, identically (the fencing house shape)."""
    if request.param == "in_memory":
        yield ArtifactRegistry()
    else:
        with SqliteArtifactRegistry(tmp_path / "state.db") as reg:
            yield reg


@pytest.fixture
def service(registry) -> _CountingService:
    return _CountingService(registry)


def _member(path: str, fingerprint: str | None) -> CheckpointMember:
    return CheckpointMember(
        member_path=path,
        artifact_id=None,
        native_token="7",
        fingerprint=fingerprint,
        captured_at=100.0,
    )


def _mint_checkpoint(
    service: CoordinatorService,
    members: "dict[str, str | None] | None" = None,
    *,
    receiver=None,
) -> str:
    """A checkpoint over ``members`` ({path: captured fingerprint}); by
    default the one member ``notes/plan.md`` captured at ``FP_NEW``."""
    rows = [
        _member(path, fingerprint)
        for path, fingerprint in (members or {"notes/plan.md": FP_NEW}).items()
    ]
    record = service.create_workspace_checkpoint(
        name="reg-cp",
        owner=OWNER,
        members=rows,
        window_min=100.0,
        window_max=100.0,
        issued_at_tick=101,
        receiver=receiver,
    )
    return record.checkpoint_id


def _artifact_named(registry, name: str):
    for artifact_id in registry.artifact_ids():
        artifact = registry.get_artifact(artifact_id)
        if artifact is not None and artifact.name == name:
            return artifact
    return None


# ---------------------------------------------------------------------------
# Typed empties and pre-flight refusals
# ---------------------------------------------------------------------------


def test_empty_writes_typed_empty_never_calls_commit_all(service) -> None:
    checkpoint_id = _mint_checkpoint(service)
    result = service.register_workspace_restore(
        checkpoint_id=checkpoint_id, controller=OWNER, writes=()
    )
    assert result.status is WORKSPACE_REGISTRATION_EMPTY
    assert result.versions == {} and result.signals == ()
    assert service.commit_all_calls == 0


def test_unknown_checkpoint_raises_typed_checkpoint_unknown(service) -> None:
    with pytest.raises(CheckpointUnknown) as excinfo:
        service.register_workspace_restore(
            checkpoint_id="no-such-checkpoint",
            controller=OWNER,
            writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
        )
    assert excinfo.value.reason is CHECKPOINT_UNKNOWN_REASON


def test_duplicate_paths_and_blank_fingerprint_rejected(service, registry) -> None:
    checkpoint_id = _mint_checkpoint(service)
    ids_before = set(registry.artifact_ids())
    with pytest.raises(ValueError, match="duplicate member_path"):
        service.register_workspace_restore(
            checkpoint_id=checkpoint_id,
            controller=OWNER,
            writes=(
                WorkspaceRestoreWrite("notes/plan.md", FP_NEW),
                WorkspaceRestoreWrite("notes/plan.md", FP_OLD),
            ),
        )
    with pytest.raises(ValueError, match="fingerprint"):
        service.register_workspace_restore(
            checkpoint_id=checkpoint_id,
            controller=OWNER,
            writes=(WorkspaceRestoreWrite("notes/plan.md", ""),),
        )
    # Nothing resolved or minted by either reject (fail-closed before work).
    assert set(registry.artifact_ids()) == ids_before
    assert service.commit_all_calls == 0


# ---------------------------------------------------------------------------
# Resolution — hash-only, first-observation parity across both registries
# ---------------------------------------------------------------------------


def test_first_observation_mints_hash_only_at_v1_and_skips(service, registry) -> None:
    """A path the coordinator never saw: the mint IS the registration —
    version 1 carrying the fingerprint, no commit, no member bytes stored."""
    checkpoint_id = _mint_checkpoint(service)
    result = service.register_workspace_restore(
        checkpoint_id=checkpoint_id,
        controller=OWNER,
        writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
    )
    assert result.status is WORKSPACE_REGISTRATION_EMPTY
    assert result.skipped == ("notes/plan.md",)
    assert service.commit_all_calls == 0

    artifact = _artifact_named(registry, "notes/plan.md")
    assert artifact is not None
    assert artifact.version == 1
    assert artifact.content_hash == FP_NEW
    # Hash-only: the coordinator holds NO member bytes for the mint.
    content = registry.get_content(artifact.id)
    assert content in (None, "", b"")

    # Idempotent: a second registration resolves the SAME artifact and skips.
    ids_before = set(registry.artifact_ids())
    again = service.register_workspace_restore(
        checkpoint_id=checkpoint_id,
        controller=OWNER,
        writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
    )
    assert again.status is WORKSPACE_REGISTRATION_EMPTY
    assert set(registry.artifact_ids()) == ids_before
    assert _artifact_named(registry, "notes/plan.md").version == 1


# ---------------------------------------------------------------------------
# The commit path — monotonic forward version, invalidation, all-or-nothing
# ---------------------------------------------------------------------------


def test_known_artifact_commits_forward_and_invalidates_peer(service, registry) -> None:
    checkpoint_id = _mint_checkpoint(service)
    artifact = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    peer = uuid4()
    registry.set_agent_state(artifact.id, peer, MESIState.SHARED, trigger="fetch", tick=1)

    result = service.register_workspace_restore(
        checkpoint_id=checkpoint_id,
        controller=OWNER,
        writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
        issued_at_tick=7,
    )

    assert result.status is WORKSPACE_REGISTRATION_COMMITTED
    assert result.versions == {"notes/plan.md": 2}
    assert service.commit_all_calls == 1
    updated = registry.get_artifact(artifact.id)
    assert updated.version == 2  # forward, never a re-stamp or decrement
    assert updated.content_hash == FP_NEW
    # The registered peer was invalidated atomically; the signal names it.
    assert registry.get_agent_state(artifact.id, peer) == MESIState.INVALID
    (signal,) = result.signals
    assert signal.artifact_id == artifact.id
    assert signal.new_version == 2
    assert signal.issuer_agent_id == OWNER


def test_hash_identical_member_skipped_no_bump(service, registry) -> None:
    checkpoint_id = _mint_checkpoint(service)
    artifact = service.register_artifact(
        name="notes/plan.md", content="restored-body", content_hash=FP_NEW
    )
    result = service.register_workspace_restore(
        checkpoint_id=checkpoint_id,
        controller=OWNER,
        writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
    )
    assert result.status is WORKSPACE_REGISTRATION_EMPTY
    assert result.skipped == ("notes/plan.md",)
    assert registry.get_artifact(artifact.id).version == 1
    assert service.commit_all_calls == 0


def test_pessimistic_holder_refuses_batch_all_or_nothing(service, registry) -> None:
    """One member blocked by an M/E holder HOLDS the whole batch: nothing
    registered, per-member typed reasons returned. Both writes are members of
    the checkpoint at their captured fingerprints (#191: a write outside the
    manifest is refused before the batch is ever built)."""
    checkpoint_id = _mint_checkpoint(
        service,
        {"notes/plan.md": FP_NEW, "docs/readme.md": sha256_hex(b"new-doc")},
    )
    blocked = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    healthy = service.register_artifact(
        name="docs/readme.md", content="old-doc", content_hash=sha256_hex(b"old-doc")
    )
    holder = uuid4()
    registry.set_agent_state(
        blocked.id, holder, MESIState.EXCLUSIVE, trigger="write", tick=1
    )

    result = service.register_workspace_restore(
        checkpoint_id=checkpoint_id,
        controller=OWNER,
        writes=(
            WorkspaceRestoreWrite("notes/plan.md", FP_NEW),
            WorkspaceRestoreWrite("docs/readme.md", sha256_hex(b"new-doc")),
        ),
    )

    assert result.status is WORKSPACE_REGISTRATION_REFUSED
    assert result.refused["notes/plan.md"].reason == "other_holder"
    # All-or-nothing: the healthy member did NOT advance either.
    assert registry.get_artifact(blocked.id).version == 1
    assert registry.get_artifact(healthy.id).version == 1
    assert result.versions == {} and result.signals == ()


def test_superseded_controller_refused_by_fence(service, registry) -> None:
    """The read-generation fence at the service seam: a controller whose grant
    a sweep reclaimed is rejected stale_read_generation — the late apply never
    lands (no phantom bump)."""
    checkpoint_id = _mint_checkpoint(service)
    artifact = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    registry.set_agent_state(
        artifact.id, OWNER, MESIState.EXCLUSIVE, trigger="write", tick=1
    )
    registry.set_agent_state(
        artifact.id, OWNER, MESIState.INVALID, trigger="reclaim_heartbeat", tick=10
    )

    result = service.register_workspace_restore(
        checkpoint_id=checkpoint_id,
        controller=OWNER,
        writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
    )

    assert result.status is WORKSPACE_REGISTRATION_REFUSED
    assert result.refused["notes/plan.md"].reason == STALE_READ_GENERATION_REASON
    updated = registry.get_artifact(artifact.id)
    assert updated.version == 1
    assert updated.content_hash == FP_OLD


def test_abort_event_fails_commit_path_closed(service, registry) -> None:
    """A6: a pre-set abort Event (the watchdog already timed out) fails the
    registration closed AT the registry write lock — no version bump lands
    after the client saw the degraded response."""
    checkpoint_id = _mint_checkpoint(service)
    artifact = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    abort = threading.Event()
    abort.set()
    with pytest.raises(WatchdogAbandoned):
        service.register_workspace_restore(
            checkpoint_id=checkpoint_id,
            controller=OWNER,
            writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
            abort=abort,
        )
    assert registry.get_artifact(artifact.id).version == 1
    assert registry.get_artifact(artifact.id).content_hash == FP_OLD


# ---------------------------------------------------------------------------
# The 'registered' restore status (the Unit-5 durable idempotency marker)
# ---------------------------------------------------------------------------


def test_registered_status_accepted_and_round_trips(service, registry) -> None:
    checkpoint_id = _mint_checkpoint(service)
    service.set_workspace_checkpoint_restore_status(
        checkpoint_id, RESTORE_STATUS_REGISTERED, updated_at=123.0
    )
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None
    assert record.restore_status == RESTORE_STATUS_REGISTERED
    assert record.restore_updated_at == 123.0


# ---------------------------------------------------------------------------
# The read-only registration query (the FIX-1 rebuild seam)
# ---------------------------------------------------------------------------


def test_workspace_member_registered_pure_read_never_mints(service, registry) -> None:
    """workspace_member_registered answers the idempotency-filter predicate as
    a PURE read: unknown path → False with nothing minted; known path → the
    content-hash comparison; never a resolve, never a mutation."""
    ids_before = set(registry.artifact_ids())
    assert service.workspace_member_registered("notes/plan.md", FP_NEW) is False
    assert set(registry.artifact_ids()) == ids_before  # no first-observation mint

    artifact = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    assert service.workspace_member_registered("notes/plan.md", FP_NEW) is False
    assert service.workspace_member_registered("notes/plan.md", FP_OLD) is True
    assert registry.get_artifact(artifact.id).version == 1  # untouched either way


# ---------------------------------------------------------------------------
# The in-memory first-observation mint race (the FIX-4 serialization)
# ---------------------------------------------------------------------------


class _ScanGateRegistry(ArtifactRegistry):
    """``artifact_ids()`` blocks ONCE mid-scan — widens the scan→mint window
    so the two-thread first-observation race is deterministic, not timing-
    dependent."""

    def __init__(self) -> None:
        super().__init__()
        self.scan_entered = threading.Event()
        self.release_scan = threading.Event()
        self._armed = True

    def artifact_ids(self):
        ids = super().artifact_ids()
        if self._armed:
            self._armed = False
            self.scan_entered.set()
            assert self.release_scan.wait(timeout=5.0)
        return ids


def test_in_memory_first_observation_race_mints_one_artifact() -> None:
    """FIX-4 regression (in-memory mint race): two concurrent first
    observations of one never-seen member_path on the IN-MEMORY registry must
    resolve to ONE artifact — the scan-then-mint is serialized service-side,
    honoring the parity claim with sqlite's atomic resolve_or_register."""
    registry = _ScanGateRegistry()
    service = CoordinatorService(registry)
    results: list = []

    def resolve() -> None:
        results.append(
            service._resolve_workspace_member_artifact("notes/plan.md", FP_NEW)
        )

    first = threading.Thread(target=resolve)
    first.start()
    assert registry.scan_entered.wait(timeout=5.0)  # first observer is mid-scan
    second = threading.Thread(target=resolve)
    second.start()
    # Serialized: the second observer must NOT complete while the first still
    # holds the mint lock mid-scan (pre-fix it minted a duplicate right here).
    second.join(timeout=0.5)
    assert second.is_alive()
    registry.release_scan.set()
    first.join(timeout=5.0)
    second.join(timeout=5.0)
    assert not first.is_alive() and not second.is_alive()

    assert len(results) == 2
    assert len(set(results)) == 1  # ONE artifact id, both observers agree
    named = [
        artifact
        for artifact in (registry.get_artifact(a) for a in registry.artifact_ids())
        if artifact is not None and artifact.name == "notes/plan.md"
    ]
    assert len(named) == 1  # exactly one mint
    assert named[0].version == 1
    assert named[0].content_hash == FP_NEW


# ---------------------------------------------------------------------------
# #191 — the write-set must match the manifest (membership half)
# ---------------------------------------------------------------------------


def _register(service, checkpoint_id: str, controller, *writes):
    return service.register_workspace_restore(
        checkpoint_id=checkpoint_id,
        controller=controller,
        writes=tuple(WorkspaceRestoreWrite(path, fp) for path, fp in writes),
    )


def _refusal(service, checkpoint_id: str, controller, *writes):
    with pytest.raises(CheckpointRegistrationRefused) as excinfo:
        _register(service, checkpoint_id, controller, *writes)
    return excinfo.value


def test_non_member_path_refused_before_any_mint(service, registry) -> None:
    """The issue's first case: ``secrets/other.md`` is not in the manifest.
    Refused typed, and the refusal lands BEFORE resolution — the coordinator
    never saw the path, and still has no artifact for it afterwards."""
    checkpoint_id = _mint_checkpoint(service)
    ids_before = set(registry.artifact_ids())

    exc = _refusal(
        service,
        checkpoint_id,
        OWNER,
        ("notes/plan.md", FP_NEW),
        ("secrets/other.md", FP_NEW),
    )

    assert exc.reason is CHECKPOINT_NOT_A_MEMBER_REASON
    assert exc.member_paths == ("secrets/other.md",)
    assert set(registry.artifact_ids()) == ids_before  # nothing minted
    assert service.commit_all_calls == 0
    # A refused registration claims nothing: the checkpoint is still open.
    assert registry.get_checkpoint(checkpoint_id).registered_by is None


def test_non_member_path_never_bumps_a_known_artifact(service, registry) -> None:
    """A known artifact outside the manifest is not version-bumped and its
    peers are not invalidated."""
    checkpoint_id = _mint_checkpoint(service)
    other = service.register_artifact(
        name="secrets/other.md", content="x", content_hash=FP_OLD
    )
    peer = uuid4()
    registry.set_agent_state(other.id, peer, MESIState.SHARED, trigger="fetch", tick=1)

    exc = _refusal(service, checkpoint_id, OWNER, ("secrets/other.md", FP_NEW))

    assert exc.reason is CHECKPOINT_NOT_A_MEMBER_REASON
    assert registry.get_artifact(other.id).version == 1
    assert registry.get_artifact(other.id).content_hash == FP_OLD
    assert registry.get_agent_state(other.id, peer) == MESIState.SHARED


def test_member_at_foreign_fingerprint_refused(service, registry) -> None:
    """The issue's third case: a member at a fingerprint of the caller's own
    choosing used to commit forward (v3). The captured fingerprint is the
    only one a restore registers."""
    checkpoint_id = _mint_checkpoint(service)
    artifact = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )

    exc = _refusal(
        service, checkpoint_id, OWNER, ("notes/plan.md", sha256_hex(b"mine"))
    )

    assert exc.reason is CHECKPOINT_FINGERPRINT_MISMATCH_REASON
    assert exc.member_paths == ("notes/plan.md",)
    assert registry.get_artifact(artifact.id).version == 1
    assert registry.get_artifact(artifact.id).content_hash == FP_OLD
    assert service.commit_all_calls == 0


def test_member_captured_without_fingerprint_matches_nothing(service, registry) -> None:
    """A member captured absent (or forward-only) has no fingerprint, so no
    write can register it — and nothing is minted for it either."""
    checkpoint_id = _mint_checkpoint(service, {"gone.md": None})
    ids_before = set(registry.artifact_ids())

    exc = _refusal(service, checkpoint_id, OWNER, ("gone.md", FP_NEW))

    assert exc.reason is CHECKPOINT_FINGERPRINT_MISMATCH_REASON
    assert set(registry.artifact_ids()) == ids_before


def test_membership_refusal_is_all_or_nothing(service, registry) -> None:
    """One bad write refuses the whole write-set: the good member is neither
    minted nor committed."""
    checkpoint_id = _mint_checkpoint(
        service, {"notes/plan.md": FP_NEW, "docs/readme.md": FP_NEW}
    )
    ids_before = set(registry.artifact_ids())

    exc = _refusal(
        service,
        checkpoint_id,
        OWNER,
        ("notes/plan.md", FP_NEW),
        ("docs/readme.md", FP_OLD),
    )

    assert exc.reason is CHECKPOINT_FINGERPRINT_MISMATCH_REASON
    assert exc.member_paths == ("docs/readme.md",)
    assert set(registry.artifact_ids()) == ids_before


# ---------------------------------------------------------------------------
# #191 — who may register (receiver / owner half)
# ---------------------------------------------------------------------------


def test_owner_authorizes_nothing_any_controller_may_register(service, registry) -> None:
    """Q1: with no receiver named, a controller other than the one that took
    the checkpoint registers it — a handoff is the point of a checkpoint.
    ``owner`` is provenance and is never compared."""
    checkpoint_id = _mint_checkpoint(service)
    receiver = uuid4()

    result = _register(service, checkpoint_id, receiver, ("notes/plan.md", FP_NEW))

    assert result.status is WORKSPACE_REGISTRATION_EMPTY
    assert result.retry_of_own_registration is False
    assert registry.get_checkpoint(checkpoint_id).registered_by == receiver
    assert registry.get_checkpoint(checkpoint_id).owner == OWNER


def test_second_controller_refused_already_registered(service, registry) -> None:
    """The issue's second case, and plan B7: a second register of the same
    checkpoint by an unrelated controller used to get a success-shaped
    ``empty_write_set``. It is now refused, typed, naming nobody."""
    checkpoint_id = _mint_checkpoint(service)
    artifact = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    first = _register(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))
    assert first.status is WORKSPACE_REGISTRATION_COMMITTED
    assert first.retry_of_own_registration is False

    intruder = uuid4()
    exc = _refusal(service, checkpoint_id, intruder, ("notes/plan.md", FP_NEW))

    assert exc.reason is CHECKPOINT_ALREADY_REGISTERED_REASON
    assert exc.member_paths == ()
    # The refusal names nobody: neither the registrant nor the intruder.
    assert str(OWNER) not in str(exc) and OWNER.hex not in str(exc)
    assert str(intruder) not in str(exc)
    assert registry.get_artifact(artifact.id).version == 2  # untouched by B
    assert registry.get_checkpoint(checkpoint_id).registered_by == OWNER


def test_same_controller_retry_is_idempotent_and_says_so(service, registry) -> None:
    """Q3: a retry by the controller that registered stays idempotent (no
    second bump) and its result says it was a retry of its own registration
    — "I registered this" is distinguishable from "someone else did"."""
    checkpoint_id = _mint_checkpoint(service)
    service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    first = _register(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))
    again = _register(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))

    assert first.status is WORKSPACE_REGISTRATION_COMMITTED
    assert first.retry_of_own_registration is False
    assert again.status is WORKSPACE_REGISTRATION_EMPTY
    assert again.skipped == ("notes/plan.md",)
    assert again.retry_of_own_registration is True
    assert _artifact_named(registry, "notes/plan.md").version == 2
    assert service.commit_all_calls == 1


def test_empty_write_set_claims_the_checkpoint(service, registry) -> None:
    """An empty write-set is still a registration of the checkpoint: it
    claims it, so a second controller is refused the same way."""
    checkpoint_id = _mint_checkpoint(service)
    first = _register(service, checkpoint_id, OWNER)
    assert first.status is WORKSPACE_REGISTRATION_EMPTY
    assert registry.get_checkpoint(checkpoint_id).registered_by == OWNER

    exc = _refusal(service, checkpoint_id, uuid4())
    assert exc.reason is CHECKPOINT_ALREADY_REGISTERED_REASON
    assert _register(service, checkpoint_id, OWNER).retry_of_own_registration is True


def test_named_receiver_binds_the_registration(service, registry) -> None:
    """A receiver named at creation is the only controller that may register;
    the owner itself is refused, and a refused attempt claims nothing."""
    receiver = uuid4()
    checkpoint_id = _mint_checkpoint(service, receiver=receiver)
    assert registry.get_checkpoint(checkpoint_id).receiver == receiver
    ids_before = set(registry.artifact_ids())

    for stranger in (OWNER, uuid4()):
        exc = _refusal(service, checkpoint_id, stranger, ("notes/plan.md", FP_NEW))
        assert exc.reason is CHECKPOINT_NOT_THE_RECEIVER_REASON
        assert exc.member_paths == ()
    assert set(registry.artifact_ids()) == ids_before
    assert registry.get_checkpoint(checkpoint_id).registered_by is None

    result = _register(service, checkpoint_id, receiver, ("notes/plan.md", FP_NEW))
    assert result.status is WORKSPACE_REGISTRATION_EMPTY
    assert registry.get_checkpoint(checkpoint_id).registered_by == receiver


def test_receiver_check_precedes_membership(service, registry) -> None:
    """A non-receiver learns only that it is not the receiver — the write-set
    is not evaluated for it."""
    checkpoint_id = _mint_checkpoint(service, receiver=uuid4())
    exc = _refusal(service, checkpoint_id, OWNER, ("secrets/other.md", FP_NEW))
    assert exc.reason is CHECKPOINT_NOT_THE_RECEIVER_REASON


def test_claim_survives_a_held_registration(service, registry) -> None:
    """The claim is sticky: a registration whose batch HELDs keeps it, so the
    same controller re-drives and nobody else takes the checkpoint mid-retry."""
    checkpoint_id = _mint_checkpoint(service)
    artifact = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=FP_OLD
    )
    holder = uuid4()
    registry.set_agent_state(
        artifact.id, holder, MESIState.EXCLUSIVE, trigger="write", tick=1
    )
    held = _register(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))
    assert held.status is WORKSPACE_REGISTRATION_REFUSED

    exc = _refusal(service, checkpoint_id, uuid4(), ("notes/plan.md", FP_NEW))
    assert exc.reason is CHECKPOINT_ALREADY_REGISTERED_REASON

    registry.set_agent_state(
        artifact.id, holder, MESIState.INVALID, trigger="release", tick=2
    )
    retried = _register(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))
    assert retried.status is WORKSPACE_REGISTRATION_COMMITTED
    assert retried.retry_of_own_registration is True


def test_concurrent_claim_loser_refused_before_resolution(
    service, registry, monkeypatch
) -> None:
    """The race: both controllers pass the header pre-check, one claims first.
    The loser is refused by the atomic claim — still before resolution, so it
    mints and commits nothing."""
    checkpoint_id = _mint_checkpoint(service)
    stale_header = registry.get_checkpoint(checkpoint_id)
    winner = uuid4()
    assert registry.claim_checkpoint_registration(checkpoint_id, winner) == (
        winner,
        True,
    )
    # The loser read the header before the winner's claim landed.
    monkeypatch.setattr(registry, "get_checkpoint", lambda _cid: stale_header)
    ids_before = set(registry.artifact_ids())

    exc = _refusal(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))

    assert exc.reason is CHECKPOINT_ALREADY_REGISTERED_REASON
    assert set(registry.artifact_ids()) == ids_before
    assert service.commit_all_calls == 0


def test_abort_event_fails_the_claim_closed(service, registry) -> None:
    """A6: the claim is a durable write, so a pre-set abort Event fails it
    closed — a watchdog-abandoned request claims nothing."""
    checkpoint_id = _mint_checkpoint(service)
    abort = threading.Event()
    abort.set()
    with pytest.raises(WatchdogAbandoned):
        service.register_workspace_restore(
            checkpoint_id=checkpoint_id,
            controller=OWNER,
            writes=(WorkspaceRestoreWrite("notes/plan.md", FP_NEW),),
            abort=abort,
        )
    assert registry.get_checkpoint(checkpoint_id).registered_by is None


@pytest.mark.parametrize(
    "write",
    [("secrets/other.md", FP_NEW), ("notes/plan.md", FP_OLD)],
    ids=["non_member", "fingerprint_mismatch"],
)
def test_already_registered_precedes_membership(service, registry, write) -> None:
    """Step 2 before step 3: once another controller holds the claim, an
    intruder learns only ``already_registered`` — its write-set (a stranger
    path, or a member at a foreign fingerprint) is not evaluated for it."""
    checkpoint_id = _mint_checkpoint(service)
    _register(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))

    exc = _refusal(service, checkpoint_id, uuid4(), write)

    assert exc.reason is CHECKPOINT_ALREADY_REGISTERED_REASON
    assert exc.member_paths == ()


def test_concurrent_own_retry_reports_retry_from_the_claim(
    service, registry, monkeypatch
) -> None:
    """Two concurrent registers by ONE controller: both read the header
    unclaimed, the other call claims first. The second is a retry of its own
    registration and must say so — the flag comes from the atomic claim, not
    from the header it read before claiming."""
    checkpoint_id = _mint_checkpoint(service)
    real_get = registry.get_checkpoint

    def _read_then_overlap(cid):
        record = real_get(cid)
        registry.claim_checkpoint_registration(cid, OWNER)
        return record

    monkeypatch.setattr(registry, "get_checkpoint", _read_then_overlap)
    result = _register(service, checkpoint_id, OWNER, ("notes/plan.md", FP_NEW))

    assert result.retry_of_own_registration is True


# ---------------------------------------------------------------------------
# #191 — the restore progress writes are gated like the registration
# ---------------------------------------------------------------------------


def _progress_writes(service, checkpoint_id: str, controller):
    """The two progress writes a restore makes, each by ``controller``."""
    return (
        lambda: service.set_workspace_checkpoint_restore_status(
            checkpoint_id, "concluded", updated_at=5.0, controller=controller
        ),
        lambda: service.set_workspace_checkpoint_member_restore(
            checkpoint_id,
            "notes/plan.md",
            restore_outcome="restored",
            deleted_at_restore=5.0,
            controller=controller,
        ),
    )


def test_non_receiver_cannot_drive_restore_progress(service, registry) -> None:
    """A controller the receiver binding excludes cannot conclude the
    checkpoint or record member outcomes (deletes register here) — which
    would turn the receiver's own restore into a no-op."""
    receiver = uuid4()
    checkpoint_id = _mint_checkpoint(service, receiver=receiver)
    for write in _progress_writes(service, checkpoint_id, uuid4()):
        with pytest.raises(CheckpointRegistrationRefused) as excinfo:
            write()
        assert excinfo.value.reason is CHECKPOINT_NOT_THE_RECEIVER_REASON
    record = registry.get_checkpoint(checkpoint_id)
    assert record.restore_status == "none"
    assert record.registered_by is None
    (member,) = registry.get_checkpoint_members(checkpoint_id)
    assert member.restore_outcome is None and member.deleted_at_restore is None

    for write in _progress_writes(service, checkpoint_id, receiver):
        write()
    assert registry.get_checkpoint(checkpoint_id).restore_status == "concluded"
    # A progress write makes no claim.
    assert registry.get_checkpoint(checkpoint_id).registered_by is None


def test_rival_of_the_registrant_cannot_drive_restore_progress(
    service, registry
) -> None:
    checkpoint_id = _mint_checkpoint(service)
    _register(service, checkpoint_id, OWNER)
    for write in _progress_writes(service, checkpoint_id, uuid4()):
        with pytest.raises(CheckpointRegistrationRefused) as excinfo:
            write()
        assert excinfo.value.reason is CHECKPOINT_ALREADY_REGISTERED_REASON
    for write in _progress_writes(service, checkpoint_id, OWNER):
        write()


def test_progress_controller_gate_unknown_checkpoint_is_keyerror(service) -> None:
    for write in _progress_writes(service, "no-such-checkpoint", OWNER):
        with pytest.raises(KeyError):
            write()


# ---------------------------------------------------------------------------
# #191 — the registry claim primitive (parity across both backends)
# ---------------------------------------------------------------------------


def test_registry_claim_first_wins_and_never_rebinds(service, registry) -> None:
    checkpoint_id = _mint_checkpoint(service)
    first, second = uuid4(), uuid4()
    assert registry.claim_checkpoint_registration(checkpoint_id, first) == (
        first,
        True,
    )
    assert registry.claim_checkpoint_registration(checkpoint_id, second) == (
        first,
        False,
    )
    assert registry.claim_checkpoint_registration(checkpoint_id, first) == (
        first,
        False,
    )
    assert registry.get_checkpoint(checkpoint_id).registered_by == first
    (listed,) = registry.list_checkpoints()
    assert listed.registered_by == first
    with pytest.raises(KeyError):
        registry.claim_checkpoint_registration("no-such-checkpoint", first)


def test_registry_refuses_a_pre_registered_header(registry) -> None:
    from ccs.coordinator.registry_protocol import CheckpointRecord

    header = CheckpointRecord(
        checkpoint_id="pre-claimed",
        name="cp",
        owner=OWNER,
        created_at=1.0,
        created_at_tick=1,
        window_min=1.0,
        window_max=1.0,
        registered_by=uuid4(),
    )
    with pytest.raises(ValueError, match="already registered"):
        registry.create_checkpoint(header, [_member("a.md", FP_NEW)])
    assert registry.get_checkpoint("pre-claimed") is None


def test_receiver_and_claim_survive_a_restart(tmp_path: Path) -> None:
    """sqlite: both #191 fields are durable."""
    receiver = uuid4()
    db = tmp_path / "state.db"
    with SqliteArtifactRegistry(db) as reg:
        checkpoint_id = _mint_checkpoint(CoordinatorService(reg), receiver=receiver)
        reg.claim_checkpoint_registration(checkpoint_id, receiver)
    with SqliteArtifactRegistry(db) as reg:
        record = reg.get_checkpoint(checkpoint_id)
        assert record.receiver == receiver
        assert record.registered_by == receiver
        exc = _refusal(
            CoordinatorService(reg), checkpoint_id, OWNER, ("notes/plan.md", FP_NEW)
        )
        assert exc.reason is CHECKPOINT_NOT_THE_RECEIVER_REASON
