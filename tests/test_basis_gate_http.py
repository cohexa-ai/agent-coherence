# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The registration a pre-read reports beside its version, on request.

A caller recording a basis needs, beside each version it read, the
registration of the row that version came from, so a later check can tell the
row it read from one registered in its place. ``/hooks/pre-read`` reports it
as ``registration`` only to a request that sets ``want_registration: true``:
on every admitted reply that reports a version (a first observation, a read on
a held grant, a warn-mode stale read), the row's registration, or ``null``
when no row snapshot backs the version or the row records no content claim.
Replies that report no version (an untracked path, a row gone before its
snapshot, a deny, a degraded answer) carry no key, and a request without the
flag gets today's reply unchanged.

The registration a reply is held to is what the registry reports for the row
(``basis_rows``), never what the route computes.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from tests.test_claude_code_coordinator_server import (
    _DEGRADE_DEADLINE_SEC,
    _await_abandoned_body_settled,
    _bytes_hash,
    _Client,
    _denied,
    _HeldRegistryLock,
    _PlanDesk,
)

_PATH = "plan.md"  # a default tracked pattern
_NO_CLAIM_HASH = "f" * 64  # the all-f placeholder: a recorded hash that claims no bytes


@contextmanager
def _desk_at(tmp_path: Path, *, strict: bool) -> Iterator[_PlanDesk]:
    """A coordinator serving in this process with ``plan.md`` (``v1``) on
    disk, in strict mode or in warn mode."""
    coherence = tmp_path / ".coherence"
    coherence.mkdir()
    if strict:
        (coherence / "strict_mode.yaml").write_text(f"- {_PATH}\n")
    (tmp_path / _PATH).write_bytes(b"v1\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="basis-registration")
    server.serve_in_thread()
    try:
        secret = load_secret(server.coordinator_root)
        assert secret is not None
        yield _PlanDesk(server, _Client("127.0.0.1", server.port, secret))
    finally:
        server.shutdown()


@pytest.fixture
def desk(tmp_path: Path) -> Iterator[_PlanDesk]:
    with _desk_at(tmp_path, strict=False) as warn_desk:
        yield warn_desk


@pytest.fixture
def strict_desk(tmp_path: Path) -> Iterator[_PlanDesk]:
    with _desk_at(tmp_path, strict=True) as strict:
        yield strict


def _registration_of(desk: _PlanDesk, path: str = _PATH) -> str:
    """The registration the registry reports for the one row named ``path``."""
    (row,) = desk.coordinator.registry.basis_rows([], [path])
    return row.registration


def _sid() -> str:
    return str(uuid.uuid4())


def _record_no_content_claim(desk: _PlanDesk) -> None:
    """Make the row of ``plan.md`` record the all-f placeholder, as a legacy
    row or the strict-mode demo's seeding leaves it. Written to the registry
    directly: no route records the placeholder."""
    registry = desk.coordinator.registry
    artifact_id = registry.lookup_artifact_id_by_name(_PATH)
    registry.set_artifact_and_content(
        artifact_id,
        dataclasses.replace(registry.get_artifact(artifact_id), content_hash=_NO_CLAIM_HASH),
        "",
    )


def _delete_before_the_next_snapshot(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> list[uuid.UUID]:
    """Delete the row the handler's next snapshot reads just before it reads
    it, as a peer's delete landing between two registry calls of one pre-read
    would. Returns the ids deleted, so a test can tell the snapshot was taken."""
    coordinator = desk.coordinator
    real_snapshot = coordinator.registry.get_artifact_and_generation
    deleted: list[uuid.UUID] = []

    def deleted_before_the_snapshot(artifact_id: uuid.UUID):
        if not deleted:
            deleted.append(artifact_id)
            coordinator.service.delete(agent_id=uuid.uuid4(), artifact_id=artifact_id)
        return real_snapshot(artifact_id)

    monkeypatch.setattr(
        coordinator.registry, "get_artifact_and_generation", deleted_before_the_snapshot)
    return deleted


# -- the admitted arms that report a version --------------------------------


def _first_observation(desk: _PlanDesk, **flag: object) -> dict:
    """A first read of ``plan.md``: the read registers the row."""
    return desk.read(_sid(), **flag)


def _held_grant(desk: _PlanDesk, **flag: object) -> dict:
    """A second read by a session still holding the grant its first read took."""
    reader = _sid()
    assert desk.read(reader)["status"] == "fresh"
    return desk.read(reader, **flag)


def _warn_stale(desk: _PlanDesk, **flag: object) -> dict:
    """A reader's read after a peer committed v2 over the v1 it read."""
    reader = _sid()
    assert desk.read(reader)["status"] == "fresh"
    desk.commit(_sid(), b"v2\n")
    return desk.read(reader, **flag)


_ARMS: dict[str, Callable[..., dict]] = {
    "first-observation": _first_observation,
    "held-grant": _held_grant,
    "warn-stale": _warn_stale,
}
_ARM_VERSION = {"first-observation": 1, "held-grant": 1, "warn-stale": 2}


def _reported_version(arm: str, answer: dict) -> int:
    return answer["summary"]["current_version"] if arm == "warn-stale" else answer["version"]


@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_an_opted_in_admitted_read_carries_the_rows_registration(
    desk: _PlanDesk, arm: str,
) -> None:
    """Each admitted arm that reports a version reports, beside it, the
    registration the registry holds for the row, in canonical dashed form."""
    answer = _ARMS[arm](desk, want_registration=True)

    assert _reported_version(arm, answer) == _ARM_VERSION[arm], answer
    assert answer["registration"] == _registration_of(desk), answer
    assert str(uuid.UUID(answer["registration"])) == answer["registration"]


def test_an_opted_in_fresh_read_adds_only_the_registration(desk: _PlanDesk) -> None:
    """The opt-in adds one key and changes nothing else in the fresh reply."""
    assert _first_observation(desk, want_registration=True) == {
        "status": "fresh", "version": 1, "registration": _registration_of(desk),
    }


# Today's reply shapes as literals, key order included: the JSON body is
# written in this order, so an equal key list is an unchanged body.
_TODAYS_KEYS = {
    "first-observation": ["status", "version"],
    "held-grant": ["status", "version"],
    "warn-stale": ["hookSpecificOutput", "status", "summary"],
}
_TODAYS_SUMMARY_KEYS = [
    "path", "current_version", "prior_version_seen_by_session", "last_writer_session_id",
    "last_writer_at_unix_ts", "warning_generated_at_unix_ts", "hash_differs",
]


@pytest.mark.parametrize("flag", [{}, {"want_registration": False}], ids=["absent", "false"])
@pytest.mark.parametrize("arm", sorted(_ARMS))
def test_without_the_flag_an_admitted_read_answers_todays_reply(
    desk: _PlanDesk, arm: str, flag: dict,
) -> None:
    """A request that does not ask for the registration, by leaving the flag
    out or sending it false, gets today's reply on each arm."""
    answer = _ARMS[arm](desk, **flag)

    assert list(answer) == _TODAYS_KEYS[arm], answer
    if arm == "warn-stale":
        assert answer["status"] == "stale", answer
        assert list(answer["summary"]) == _TODAYS_SUMMARY_KEYS, answer
        assert answer["summary"]["current_version"] == 2, answer
    else:
        assert answer == {"status": "fresh", "version": 1}


# -- null: a version no row snapshot backs, or a row with no content claim ----


def test_a_first_observation_whose_row_is_gone_before_its_snapshot_reports_null(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The read registers the row and grants it, and a peer's delete lands
    before the snapshot the reply is built from. The reply reports the seeded
    version with no row behind it, so the registration is ``null``."""
    deleted = _delete_before_the_next_snapshot(desk, monkeypatch)

    answer = desk.read(_sid(), want_registration=True)

    assert deleted, "the read never took its snapshot"
    assert answer == {"status": "fresh", "version": 1, "registration": None}


def test_a_first_observation_seeded_with_no_content_claim_reports_null(
    desk: _PlanDesk,
) -> None:
    """A first read whose hash is the all-f placeholder seeds a row that
    claims no bytes, so the registration is ``null``."""
    answer = desk.read(_sid(), content_hash=_NO_CLAIM_HASH, want_registration=True)

    assert answer == {"status": "fresh", "version": 1, "registration": None}


def test_a_first_read_with_no_hash_reports_null(desk: _PlanDesk) -> None:
    """A first read that sends no hash seeds a row that records none; the
    reply comes from the read's own grant and reports ``null``."""
    status, answer = desk.client.post(
        "/hooks/pre-read", {"session_id": _sid(), "path": _PATH, "want_registration": True})

    assert status == 200
    assert answer == {"status": "fresh", "version": 1, "registration": None}


def test_a_held_grant_on_a_row_with_no_content_claim_reports_null(desk: _PlanDesk) -> None:
    reader = _sid()
    assert desk.read(reader)["status"] == "fresh"
    _record_no_content_claim(desk)

    answer = desk.read(reader, want_registration=True)

    assert answer == {"status": "fresh", "version": 1, "registration": None}


def test_a_warn_stale_read_of_a_row_with_no_content_claim_reports_null(desk: _PlanDesk) -> None:
    reader = _sid()
    assert desk.read(reader)["status"] == "fresh"
    desk.commit(_sid(), b"v2\n")
    _record_no_content_claim(desk)

    answer = desk.read(reader, want_registration=True)

    assert answer["status"] == "stale" and not _denied(answer), answer
    assert answer["summary"]["current_version"] == 2, answer
    assert answer["registration"] is None, answer


# -- absent: replies that report no version ----------------------------------


def test_an_opted_in_untracked_read_carries_no_registration(desk: _PlanDesk) -> None:
    status, answer = desk.client.post(
        "/hooks/pre-read",
        {"session_id": _sid(), "path": "untracked.bin", "want_registration": True})

    assert status == 200
    assert answer == {"status": "fresh"}


def test_an_opted_in_read_of_a_row_gone_before_its_snapshot_carries_no_registration(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row existed when the read looked it up, and a peer's delete lands
    before the snapshot: the reply is the untracked admit, with no version and
    no registration."""
    assert desk.read(_sid())["status"] == "fresh"
    deleted = _delete_before_the_next_snapshot(desk, monkeypatch)

    answer = desk.read(_sid(), want_registration=True)

    assert deleted, "the read never took its snapshot"
    assert answer == {"status": "fresh"}


def test_an_opted_in_strict_stale_read_is_denied_with_no_registration(
    strict_desk: _PlanDesk,
) -> None:
    reader = _sid()
    assert strict_desk.read(reader)["status"] == "fresh"
    strict_desk.commit(_sid(), b"v2\n")

    answer = strict_desk.read(reader, want_registration=True)

    assert _denied(answer), answer
    assert "registration" not in answer, answer


def test_an_opted_in_holder_whose_disk_changed_is_denied_with_no_registration(
    strict_desk: _PlanDesk,
) -> None:
    """A strict reader still holding its grant hashes bytes nobody committed:
    the held-grant arm denies it, and the deny reports no registration."""
    reader = _sid()
    assert strict_desk.read(reader)["status"] == "fresh"
    strict_desk.file.write_bytes(b"v1\nchanged outside\n")

    answer = strict_desk.read(reader, want_registration=True)

    assert _denied(answer), answer
    assert "registration" not in answer, answer


def test_an_opted_in_degraded_read_carries_no_registration(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read that times out on registry contention answers the degraded
    advisory, which reports no version and no registration."""
    import ccs.adapters.claude_code.coordinator_server as server_mod

    coordinator = desk.coordinator
    reader = _sid()
    # Registers the row while the lock is free, so the held lock below blocks
    # only the timed-out read's own work.
    assert desk.read(reader)["status"] == "fresh"
    monkeypatch.setattr(server_mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    timeouts_before = coordinator._watchdog_timeouts_total
    aborts_before = coordinator._watchdog_late_aborts_total
    completions_before = coordinator._watchdog_late_completion_total

    with _HeldRegistryLock(coordinator) as held:
        status, answer = desk.client.post("/hooks/pre-read", {
            "session_id": reader, "path": _PATH, "want_registration": True,
            "content_hash": _bytes_hash(desk.file.read_bytes()),
        })
        held.release()
    _await_abandoned_body_settled(
        coordinator, aborts_before=aborts_before, completions_before=completions_before)

    assert coordinator._watchdog_timeouts_total == timeouts_before + 1, (
        "the read did not time out, so this measured the undegraded route")
    assert status == 200 and answer.get("degraded") is True, answer
    assert "registration" not in answer, answer


# -- a row registered again ---------------------------------------------------


def test_a_row_deleted_and_registered_again_reports_a_new_registration(
    desk: _PlanDesk,
) -> None:
    """The coordinator deletes ``plan.md``'s row and a later read registers
    it again: the registration the read reports changes, so a basis recorded
    before the delete cannot pass for one taken after it."""
    reader = _sid()
    before = desk.read(reader, want_registration=True)
    coordinator = desk.coordinator
    artifact_id = coordinator.registry.lookup_artifact_id_by_name(_PATH)
    coordinator.service.delete(agent_id=uuid.uuid4(), artifact_id=artifact_id)

    after = desk.read(reader, want_registration=True)

    assert before["registration"] is not None and after["registration"] is not None
    assert after["registration"] != before["registration"]
    assert after["registration"] == _registration_of(desk)
