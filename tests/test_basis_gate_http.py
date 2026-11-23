# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The basis routes, and the registration a pre-read reports for them.

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

``/basis/check`` answers whether every input a caller recorded is still the
version (and the registration) it read, at one instant; ``/basis/commit``
compare-and-swaps one target only while they all are. Both refuse a malformed
request with 400 and never answer it with a hold, keep the principal and
counter rules of the other routes, and hold rather than proceed whenever the
coordinator cannot confirm the answer in time.
"""

from __future__ import annotations

import dataclasses
import logging
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.basis import BasisMember
from ccs.core.exceptions import CoherenceError
from ccs.core.states import MESIState, TransientState
from tests.coordinator_helpers import serve_in_process
from tests.test_claude_code_coordinator_server import (
    _DEGRADE_DEADLINE_SEC,
    _await_abandoned_body_settled,
    _Background,
    _bytes_hash,
    _claimed,
    _Client,
    _DelayedCall,
    _denied,
    _hand_off,
    _HeldRegistryLock,
    _late_completion_criticals,
    _late_completion_warnings,
    _PlanDesk,
    _projection,
    _Session,
    _u3a_observable,
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


# ===========================================================================
# POST /basis/check and POST /basis/commit
# ===========================================================================

_CHECK = "/basis/check"
_COMMIT = "/basis/commit"
_UNTRACKED = "notes.txt"  # matches no default tracked pattern
_NEVER_READ = "task.md"  # a default tracked pattern nothing in these tests reads

# The answers a route gives when it cannot confirm one in time, as frozen
# literals in wire order: a check or commit that ran nothing on a stalled
# registry, and a commit whose outcome is unknown.
_CHECK_DEGRADED = {
    "answer": "hold", "moved": [], "reason": "version_unconfirmed", "recovery": "retry",
    "degraded": True, "held_by": "watchdog_timeout",
}
_COMMIT_NOTHING_RAN = {
    "answer": "hold", "moved": [], "reason": "version_unconfirmed", "recovery": "retry",
}
_COMMIT_OUTCOME_UNKNOWN = {
    "answer": "hold", "moved": [], "reason": "commit_unconfirmed", "degraded": True,
    "held_by": "watchdog_timeout",
}


def _member(path: object, version: object, registration: str | None = None, **extra: Any) -> dict:
    member: dict = {"path": path, "version": version}
    if registration is not None:
        member["registration"] = registration
    return {**member, **extra}


def _read_file(desk: _PlanDesk, who: _Session, path: str, data: bytes | None = None) -> dict:
    """``who`` reads ``path`` (writing ``data`` to disk first when given),
    sending the bytes' hash and asking for the registration, as a caller
    recording a basis does."""
    file = desk.coordinator.coordinator_root / path
    if data is not None:
        file.write_bytes(data)
    status, answer = desk.client.post("/hooks/pre-read", {
        "session_id": who.sid, "path": path, "want_registration": True,
        "content_hash": _bytes_hash(file.read_bytes()),
    }, principal=who.principal)
    assert status == 200 and answer["status"] == "fresh", answer
    return answer


def _recorded(desk: _PlanDesk, who: _Session, path: str, data: bytes | None = None) -> dict:
    """The member a caller records for ``path``: the version and the
    registration its read reported."""
    answer = _read_file(desk, who, path, data)
    assert answer["registration"] is not None, answer
    return _member(path, answer["version"], answer["registration"])


def _write(desk: _PlanDesk, path: str, data: bytes) -> None:
    """A peer's pessimistic write of ``path``: acquire, write the bytes,
    commit them. Moves the version by one."""
    peer = str(uuid.uuid4())
    status, edit = desk.client.post("/hooks/pre-edit", {"session_id": peer, "path": path})
    assert status == 200 and edit.get("ok") is True, edit
    (desk.coordinator.coordinator_root / path).write_bytes(data)
    status, done = desk.client.post("/hooks/post-edit", {
        "session_id": peer, "path": path, "success": True, "content_hash": _bytes_hash(data),
    })
    assert status == 200 and done.get("ok") is True, done


def _version(desk: _PlanDesk, path: str) -> int | None:
    registry = desk.coordinator.registry
    artifact_id = registry.lookup_artifact_id_by_name(path)
    return None if artifact_id is None else registry.get_artifact(artifact_id).version


def _post(desk: _PlanDesk, route: str, who: _Session, **fields: Any) -> tuple[int, dict]:
    return desk.client.post(route, {"session_id": who.sid, **fields}, principal=who.principal)


def _check(desk: _PlanDesk, who: _Session, basis: list[dict]) -> dict:
    status, answer = _post(desk, _CHECK, who, basis=basis)
    assert status == 200, answer
    return answer


def _commit(
    desk: _PlanDesk, who: _Session, basis: list[dict], target: dict, data: bytes = b"v2\n",
) -> dict:
    status, answer = _post(
        desk, _COMMIT, who, basis=basis, target=target, content_hash=_bytes_hash(data))
    assert status == 200, answer
    return answer


def _moved(path: str, expected: int, current: int | None, reason: str) -> dict:
    return {"path": path, "expected": expected, "current": current, "reason": reason}


def _rederive(*moved: dict) -> dict:
    return {"answer": "hold", "moved": list(moved), "recovery": "rederive"}


# -- 400: a malformed request never reaches an answer ----------------------


def _valid_body(route: str) -> dict:
    """A request that passes every shape check on ``route``: one member on
    ``plan.md``, and for a commit the same member as the target."""
    body: dict = {"session_id": str(uuid.uuid4()), "basis": [_member(_PATH, 1)]}
    if route == _COMMIT:
        body["target"] = _member(_PATH, 1)
        body["content_hash"] = _bytes_hash(b"v2\n")
    return body


def _with_member(member: dict) -> Callable[[dict], None]:
    def edit(body: dict) -> None:
        body["basis"] = [member]
    return edit


def _with_target(target: object) -> Callable[[dict], None]:
    def edit(body: dict) -> None:
        body["target"] = target
    return edit


def _set(key: str, value: object) -> Callable[[dict], None]:
    def edit(body: dict) -> None:
        body[key] = value
    return edit


def _drop(key: str) -> Callable[[dict], None]:
    def edit(body: dict) -> None:
        del body[key]
    return edit


def _target_shares_a_registration(body: dict) -> None:
    body["basis"] = [_member(_PATH, 1, _REGISTRATION)]
    body["target"] = _member("spec.md", 1, _REGISTRATION)


_REGISTRATION = str(uuid.UUID(int=7))

# Each malformed shape, on both routes, with a word the 400 must name, so the
# refusal is shown to come from the check this case is about.
_MALFORMED_ON_BOTH: dict[str, tuple[Callable[[dict], None], str]] = {
    "unknown top-level key": (_set("expected_version", 1), "expected_version"),
    "agentId": (_set("agentId", "sub-1"), "agentId"),
    "subagent_id": (_set("subagent_id", "sub-1"), "subagent_id"),
    "missing session_id": (_drop("session_id"), "session_id"),
    "unknown member key": (_with_member({**_member(_PATH, 1), "hash": "x"}), "hash"),
    "member not an object": (_set("basis", ["plan.md"]), "basis[0]"),
    "missing basis": (_drop("basis"), "basis"),
    "basis not a list": (_set("basis", {"path": _PATH, "version": 1}), "basis"),
    "empty basis": (_set("basis", []), "basis"),
    "65 members": (
        _set("basis", [_member(f"d{i}/plan.md", 1) for i in range(65)]), "64"),
    "repeated path": (_set("basis", [_member(_PATH, 1), _member(_PATH, 2)]), "twice"),
    "repeated spelling of one file": (
        _set("basis", [_member(_PATH, 1), _member(f"./{_PATH}", 1)]), "twice"),
    "repeated registration": (_set("basis", [
        _member(_PATH, 1, _REGISTRATION), _member("spec.md", 1, _REGISTRATION)]), "twice"),
    "version 0": (_with_member(_member(_PATH, 0)), "version"),
    "version true": (_with_member(_member(_PATH, True)), "version"),
    "version 2**63": (_with_member(_member(_PATH, 2**63)), "version"),
    "version a string": (_with_member(_member(_PATH, "1")), "version"),
    "missing version": (_with_member({"path": _PATH}), "version"),
    "registration not a UUID": (
        _with_member(_member(_PATH, 1, "not-a-uuid")), "registration"),
    "absolute path": (_with_member(_member("/etc/plan.md", 1)), "relative"),
    "dot-dot segment": (_with_member(_member("../plan.md", 1)), ".."),
    "control character": (_with_member(_member("pl\nan.md", 1)), "control"),
    "path over 1024": (_with_member(_member("a" * 1025, 1)), "1024"),
    "path not a string": (_with_member(_member(5, 1)), "path"),
    "missing path": (_with_member({"version": 1}), "path"),
    "unknown name_kind": (_with_member(_member(_PATH, 1, name_kind="dir")), "path"),
    "malformed agent_id": (_set("agent_id", "not ok!"), "agent_id"),
}

_MALFORMED_COMMIT_ONLY: dict[str, tuple[Callable[[dict], None], str]] = {
    "target disagrees with its member's version": (
        _with_target(_member(_PATH, 2)), "disagrees"),
    "target disagrees with its member's registration": (
        _with_target(_member(_PATH, 1, _REGISTRATION)), "disagrees"),
    "target spells its member's file another way at another version": (
        _with_target(_member(f"./{_PATH}", 2)), "disagrees"),
    "target names a member's registration under another path": (
        _target_shares_a_registration, "disagrees"),
    "missing target": (_drop("target"), "target"),
    "target not an object": (_with_target("plan.md"), "target"),
    "unknown target key": (_with_target({**_member(_PATH, 1), "content": "x"}), "content"),
    "target version 0": (_with_target(_member("spec.md", 0)), "version"),
    "target path absolute": (_with_target(_member("/spec.md", 1)), "relative"),
    "target unknown name_kind": (_with_target(_member("spec.md", 1, name_kind="dir")), "path"),
    "missing content_hash": (_drop("content_hash"), "content_hash"),
    "malformed content_hash": (_set("content_hash", "abc"), "content_hash"),
}

_MALFORMED_CASES = [
    (route, case) for case in _MALFORMED_ON_BOTH for route in (_CHECK, _COMMIT)
] + [(_COMMIT, case) for case in _MALFORMED_COMMIT_ONLY]


@pytest.mark.parametrize(
    ("route", "case"), _MALFORMED_CASES, ids=[f"{r[1:]}-{c}" for r, c in _MALFORMED_CASES])
def test_a_malformed_basis_request_answers_400_and_never_a_hold(
    desk: _PlanDesk, route: str, case: str,
) -> None:
    """A request no answer could describe is a client error, never a hold: a
    hold invites a retry, and no number of retries mends the request. The
    refusal names the field at fault, and the request reads and mints nothing
    (``plan.md`` was never read, so a lookup that registered would leave a
    row)."""
    edit, named = (_MALFORMED_ON_BOTH | _MALFORMED_COMMIT_ONLY)[case]
    body = _valid_body(route)
    edit(body)

    status, answer = desk.client.post(route, body)

    assert status == 400, answer
    assert set(answer) == {"error"} and named in answer["error"], answer
    assert desk.coordinator.registry.lookup_artifact_id_by_name(_PATH) is None


@pytest.mark.parametrize("route", [_CHECK, _COMMIT], ids=["check", "commit"])
def test_the_valid_body_the_malformed_cases_start_from_is_answered(
    desk: _PlanDesk, route: str,
) -> None:
    """Control for the 400 cases: unedited, the same body is answered (a hold:
    ``plan.md`` has no row yet)."""
    status, answer = desk.client.post(route, _valid_body(route))

    assert status == 200 and answer["answer"] == "hold", answer


# -- a body a CoherentVolume sends -------------------------------------------

_VOLUME_PATH = "data/shared.txt"
_FAST = LifecycleConfig(
    idle_shutdown_sec=0,
    sweep_interval_sec=0.1,
    notice_evict_max_age_sec=1.0,
    port_file_retry_attempts=20,
    port_file_retry_interval_sec=0.05,
    connect_retry_attempts=10,
    connect_retry_interval_sec=0.05,
)


def test_the_body_a_coherent_volume_sends_passes_both_schemas(tmp_path: Path) -> None:
    """A real volume's request carries what the volume adds to every body (its
    incarnation as ``agent_id``) and presents its session's principal; both
    routes accept it and answer, the commit as the volume's own agent."""
    file = tmp_path / _VOLUME_PATH
    file.parent.mkdir(parents=True)
    file.write_bytes(b"v1")
    server = serve_in_process(tmp_path, "basis-volume")
    volume = CoherentVolume(tmp_path, managed=("data/**",), config=_FAST)
    try:
        _, version = volume.read_with_version(_VOLUME_PATH)
        (row,) = server.registry.basis_rows([], [_VOLUME_PATH])
        member = BasisMember(_VOLUME_PATH, version, row.registration).to_wire()
        session = {"session_id": volume._session_id}

        checked = volume._post(_CHECK, {**session, "basis": [member]})
        committed = volume._post(_COMMIT, {
            **session, "basis": [member], "target": member,
            "content_hash": _bytes_hash(b"v2"),
        })

        assert checked == {
            "answer": "proceed", "checked": {_VOLUME_PATH: {"expected": 1, "current": 1}},
        }
        assert committed == {
            "answer": "proceed", "committed": {"path": _VOLUME_PATH, "version": 2},
        }
        artifact_id = server.registry.lookup_artifact_id_by_name(_VOLUME_PATH)
        assert server.registry.last_writer_for(artifact_id) == session_to_agent_id(
            volume._session_id, volume._incarnation)
    finally:
        volume.close()
        server.shutdown()


# -- /basis/check: what each member's state answers --------------------------


def test_an_unmoved_basis_proceeds_with_every_members_versions(desk: _PlanDesk) -> None:
    who = _claimed(desk.client)
    basis = [_recorded(desk, who, _PATH), _recorded(desk, who, "spec.md", b"s1\n")]

    assert _check(desk, who, basis) == {"answer": "proceed", "checked": {
        _PATH: {"expected": 1, "current": 1}, "spec.md": {"expected": 1, "current": 1},
    }}


def test_every_member_a_peer_moved_holds_with_its_own_reason(desk: _PlanDesk) -> None:
    """Both inputs moved before the check: the hold names each, never only the
    first, with the versions and ``rederive``."""
    who = _claimed(desk.client)
    basis = [_recorded(desk, who, _PATH), _recorded(desk, who, "spec.md", b"s1\n")]
    _write(desk, _PATH, b"v2\n")
    _write(desk, "spec.md", b"s2\n")

    assert _check(desk, who, basis) == _rederive(
        _moved(_PATH, 1, 2, "version_moved"), _moved("spec.md", 1, 2, "version_moved"))


def test_a_byte_identical_re_save_moves_the_version_and_holds(desk: _PlanDesk) -> None:
    """Only versions are compared: the same bytes committed again are a new
    version, so the basis holds."""
    who = _claimed(desk.client)
    basis = [_recorded(desk, who, _PATH)]
    _write(desk, _PATH, desk.file.read_bytes())

    assert _check(desk, who, basis) == _rederive(_moved(_PATH, 1, 2, "version_moved"))


def test_an_untracked_member_holds_unmanaged(desk: _PlanDesk) -> None:
    who = _claimed(desk.client)

    assert _check(desk, who, [_member(_UNTRACKED, 1)]) == _rederive(
        _moved(_UNTRACKED, 1, None, "basis_member_unmanaged"))


def test_a_tracked_member_never_read_holds_vanished_and_mints_no_row(desk: _PlanDesk) -> None:
    who = _claimed(desk.client)

    assert _check(desk, who, [_member(_NEVER_READ, 1)]) == _rederive(
        _moved(_NEVER_READ, 1, None, "input_vanished"))
    assert desk.coordinator.registry.lookup_artifact_id_by_name(_NEVER_READ) is None


def test_a_row_deleted_and_registered_again_holds_replaced(desk: _PlanDesk) -> None:
    """The coordinator deletes ``plan.md``'s row and a later read registers it
    again at version 1: the version matches, only the registration tells the
    two rows apart, so the old member holds ``basis_member_replaced``."""
    who = _claimed(desk.client)
    basis = [_recorded(desk, who, _PATH)]
    coordinator = desk.coordinator
    artifact_id = coordinator.registry.lookup_artifact_id_by_name(_PATH)
    coordinator.service.delete(agent_id=uuid.uuid4(), artifact_id=artifact_id)
    assert _read_file(desk, _claimed(desk.client), _PATH)["version"] == 1

    assert _check(desk, who, basis) == _rederive(_moved(_PATH, 1, 1, "basis_member_replaced"))


_KEY = "keys/plan.md"  # a name, not a file: nothing on disk carries it


def test_a_key_member_resolves_to_the_key_row(desk: _PlanDesk) -> None:
    """A key names a row that is no file. A member of ``name_kind`` key is
    found as that row: it proceeds while the key is unchanged, under the
    registration the key's read reported, and holds once a peer commits the
    key."""
    who, peer = _claimed(desk.client), _claimed(desk.client)
    status, key_read = desk.client.post("/hooks/pre-read", {
        "session_id": who.sid, "path": _KEY, "name_kind": "key", "want_registration": True,
        "content_hash": _bytes_hash(b"key v1"),
    }, principal=who.principal)
    assert status == 200 and key_read["version"] == 1 and key_read["registration"], key_read
    member = _member(_KEY, 1, key_read["registration"], name_kind="key")

    assert _check(desk, who, [member]) == {
        "answer": "proceed", "checked": {_KEY: {"expected": 1, "current": 1}},
    }
    assert _post(
        desk, "/hooks/post-edit-cas", peer, path=_KEY, name_kind="key", expected_version=1,
        content_hash=_bytes_hash(b"key v2"),
    ) == (200, {"ok": True, "version": 2})
    assert _check(desk, who, [member]) == _rederive(_moved(_KEY, 1, 2, "version_moved"))


def _check_observable(desk: _PlanDesk, who: _Session, artifact_id: uuid.UUID) -> dict:
    """Everything a pre-read moves around the same rows, read without
    consuming anything, minus the check's own attempt counter."""
    coordinator = desk.coordinator
    observable = _u3a_observable(coordinator, session_to_agent_id(who.sid), artifact_id)
    counters = dict(observable["counters"])
    endpoint = dict(counters["endpoint_counters"])
    endpoint.pop("basis_check_total", None)
    observable["counters"] = {**counters, "endpoint_counters": endpoint}
    observable["agent_names"] = sorted(str(a) for a, _ in coordinator.agent_names_snapshot())
    observable["compact_pending"] = coordinator.has_compact_pending(who.sid)
    observable["rows"] = sorted(map(str, coordinator.registry.status_snapshot()[0]))
    return observable


def test_a_check_changes_nothing_observable(desk: _PlanDesk) -> None:
    """A check is read-only and lookup-only on each answer it gives: version,
    content hash, owner generation, the caller's grant, heartbeat and notice,
    the stale and deny markers, the re-grounding flag, the display names, the
    rows, the files and every counter but its own attempt counter stay as they
    were. Each dimension holds a non-default value first (the read leaves
    the caller's heartbeat and grant)."""
    coordinator = desk.coordinator
    who = _claimed(desk.client)
    member = _recorded(desk, who, _PATH)
    artifact_id = coordinator.registry.lookup_artifact_id_by_name(_PATH)
    agent_id = session_to_agent_id(who.sid)
    coordinator.registry.record_preemption_notice(
        victim_agent_id=agent_id, artifact_id=artifact_id,
        preempter_agent_id=uuid.uuid4(), preempted_at_unix_ts=1234.0)
    coordinator.mark_compact_pending(who.sid)
    coordinator.mark_stale_warned(agent_id, artifact_id)
    coordinator.record_strict_deny(who.sid, _PATH)
    before = _check_observable(desk, who, artifact_id)
    assert before["heartbeat"] is not None and before["notice"] is not None
    assert before["grant"] == MESIState.SHARED and before["compact_pending"] is True

    answers = [
        _check(desk, who, [member]),
        _check(desk, who, [_member(_PATH, 7), _member(_NEVER_READ, 1), _member(_UNTRACKED, 1)]),
    ]

    assert [a["answer"] for a in answers] == ["proceed", "hold"], answers
    assert _check_observable(desk, who, artifact_id) == before


# -- /basis/commit: wins, holds and target refusals --------------------------


def test_a_commit_on_a_target_inside_the_basis_moves_only_the_target(desk: _PlanDesk) -> None:
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)
    spec = _recorded(desk, who, "spec.md", b"s1\n")

    answer = _commit(desk, who, [plan, spec], plan)

    assert answer == {"answer": "proceed", "committed": {"path": _PATH, "version": 2}}
    assert (_version(desk, _PATH), _version(desk, "spec.md")) == (2, 1)


def test_a_commit_on_a_target_outside_the_basis_moves_only_the_target(desk: _PlanDesk) -> None:
    who = _claimed(desk.client)
    spec = _recorded(desk, who, "spec.md", b"s1\n")
    _read_file(desk, who, _PATH)

    answer = _commit(desk, who, [spec], _member(_PATH, 1))

    assert answer == {"answer": "proceed", "committed": {"path": _PATH, "version": 2}}
    assert (_version(desk, _PATH), _version(desk, "spec.md")) == (2, 1)


def test_a_target_spelling_a_member_another_way_commits_as_that_member(desk: _PlanDesk) -> None:
    """A target that re-spells a basis member (``./plan.md`` for ``plan.md``)
    at the member's version and registration is that member: the commit wins
    under the member's own path, never an outcome-unknown hold."""
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)

    answer = _commit(desk, who, [plan], {**plan, "path": f"./{_PATH}"})

    assert answer == {"answer": "proceed", "committed": {"path": _PATH, "version": 2}}
    assert _version(desk, _PATH) == 2


def test_a_commit_win_carries_the_handoff_key(desk: _PlanDesk) -> None:
    """The successor's win on a path handed to it completes the record, and
    the answer carries the ``handoff`` key a compare-and-swap win carries."""
    giver, successor = _claimed(desk.client), _claimed(desk.client)
    _hand_off(desk.client, giver, successor, _PATH)

    answer = _commit(desk, successor, [_member(_PATH, 1)], _member(_PATH, 1))

    assert answer == {
        "answer": "proceed", "committed": {"path": _PATH, "version": 2},
        "handoff": _projection(
            giver, successor, "successor", status="completed", live=False, outcome="completed"),
    }


def test_a_moved_member_holds_the_commit_and_writes_nothing(desk: _PlanDesk) -> None:
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)
    spec = _recorded(desk, who, "spec.md", b"s1\n")
    _write(desk, "spec.md", b"s2\n")

    answer = _commit(desk, who, [spec, plan], plan)

    assert answer == _rederive(_moved("spec.md", 1, 2, "version_moved"))
    assert _version(desk, _PATH) == 1


def test_a_target_outside_the_basis_that_moved_holds_in_moved(desk: _PlanDesk) -> None:
    who = _claimed(desk.client)
    spec = _recorded(desk, who, "spec.md", b"s1\n")
    _read_file(desk, who, _PATH)
    _write(desk, _PATH, b"v2\n")

    answer = _commit(desk, who, [spec], _member(_PATH, 1), b"v3\n")

    assert answer == _rederive(_moved(_PATH, 1, 2, "version_moved"))
    assert _version(desk, _PATH) == 2


@pytest.mark.parametrize(
    ("target", "reason"),
    [(_UNTRACKED, "basis_member_unmanaged"), (_NEVER_READ, "input_vanished")],
    ids=["untracked", "never-read"])
def test_a_target_the_coordinator_does_not_manage_holds_and_never_wins(
    desk: _PlanDesk, target: str, reason: str,
) -> None:
    who = _claimed(desk.client)
    spec = _recorded(desk, who, "spec.md", b"s1\n")

    answer = _commit(desk, who, [spec], _member(target, 1))

    assert answer == _rederive(_moved(target, 1, None, reason))
    assert desk.coordinator.registry.lookup_artifact_id_by_name(target) is None


def test_a_peers_grant_on_the_target_holds_with_the_target_block(desk: _PlanDesk) -> None:
    """The basis held, the target's own compare-and-swap was refused: the
    hold carries the refusal apart from the members, with no recovery class,
    and nothing moves."""
    who = _claimed(desk.client)
    _read_file(desk, _claimed(desk.client), _PATH)
    assert desk.edit(str(uuid.uuid4())).get("ok") is True

    answer = _commit(desk, who, [_member(_PATH, 1)], _member(_PATH, 1))

    assert answer == {"answer": "hold", "moved": [], "target": {
        "path": _PATH, "reason": "other_holder", "current_version": 1,
    }}
    assert _version(desk, _PATH) == 1


def _in_transient(desk: _PlanDesk, who: _Session) -> dict:
    """``who`` holds ``plan.md`` SHARED and is left mid-invalidation, the
    registry shape a peer's invalidating write leaves."""
    _read_file(desk, who, _PATH)
    registry = desk.coordinator.registry
    registry.set_agent_transient(
        registry.lookup_artifact_id_by_name(_PATH), session_to_agent_id(who.sid),
        TransientState.SIA, entered_tick=1)
    return {}


def _holding_exclusive(desk: _PlanDesk, who: _Session) -> dict:
    assert desk.edit(who).get("ok") is True
    return {}


def _a_fenced_giver(desk: _PlanDesk, who: _Session) -> dict:
    successor = _claimed(desk.client)
    _hand_off(desk.client, who, successor, _PATH)
    return {"successor": successor.agent, "version_at_transfer": 1}


@pytest.mark.parametrize(
    ("caller_state", "reason"),
    [
        (_in_transient, "caller_in_transient_state"),
        (_holding_exclusive, "occ_is_shared_or_invalid_only"),
        (_a_fenced_giver, "handed_off"),
    ],
    ids=["transient", "exclusive", "fenced-giver"])
def test_a_refused_target_precondition_holds_with_its_typed_reason(
    desk: _PlanDesk, caller_state: Callable[[_PlanDesk, _Session], dict], reason: str,
) -> None:
    """The compare-and-swap preconditions the target keeps answer as a target
    block with their stable token, never as prose and never as an error, and
    nothing moves. A fenced giver's block names the successor and the version
    at transfer, as its compare-and-swap refusal does."""
    who = _claimed(desk.client)
    extra = caller_state(desk, who)

    answer = _commit(desk, who, [_member(_PATH, 1)], _member(_PATH, 1))

    assert answer == {"answer": "hold", "moved": [], "target": {
        "path": _PATH, "reason": reason, "current_version": 1, **extra,
    }}
    assert _version(desk, _PATH) == 1


def test_a_commit_error_that_is_no_target_refusal_answers_outcome_unknown(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coherence error from the commit other than the target's
    preconditions is never reported as a target refusal: such a raise may
    follow a win, so the answer is the commit's outcome-unknown hold."""
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)

    def corrupted(*_args: Any, **_kwargs: Any) -> None:
        raise CoherenceError("commit_cas_corruption (expected > current)")

    monkeypatch.setattr(desk.coordinator.service, "commit_if_basis", corrupted)

    assert _commit(desk, who, [plan], plan) == {
        "answer": "hold", "moved": [], "reason": "commit_unconfirmed", "degraded": True,
        "held_by": "handler_error",
    }


# -- the guarded run prevents what an unguarded one loses --------------------


def test_an_input_moved_before_the_commit_holds_where_a_plain_commit_lands(
    desk: _PlanDesk,
) -> None:
    """``plan.md`` was derived from ``spec.md`` at version 1, and a peer then
    moved ``spec.md``. A compare-and-swap of ``plan.md`` alone lands the stale
    derivation; the commit under the basis holds and writes nothing."""
    who = _claimed(desk.client)
    spec = _recorded(desk, who, "spec.md", b"s1\n")
    plan = _recorded(desk, who, _PATH)
    _write(desk, "spec.md", b"s2\n")

    guarded = _commit(desk, who, [spec, plan], plan)
    assert guarded == _rederive(_moved("spec.md", 1, 2, "version_moved"))
    assert _version(desk, _PATH) == 1

    status, plain = _post(
        desk, "/hooks/post-edit-cas", who, path=_PATH, expected_version=1,
        content_hash=_bytes_hash(b"v2\n"))
    assert (status, plain) == (200, {"ok": True, "version": 2}), "control: the plain commit lands"


def test_a_peer_forced_between_the_compare_and_the_target_write_lands_after_the_commit(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The basis compare and the target's compare-and-swap are one hold: a
    peer's write of a member, started while the commit sits between the two,
    waits for the commit to return and lands after it. The commit proceeds on
    the basis it compared and writes the target once."""
    who = _claimed(desk.client)
    spec = _recorded(desk, who, "spec.md", b"s1\n")
    plan = _recorded(desk, who, _PATH)
    peer = _claimed(desk.client)
    delayed = _DelayedCall(
        monkeypatch, desk.coordinator.registry, "commit_cas_if_basis", lambda *a, **kw: True)

    commit = _Background(desk.client, _COMMIT, {
        "session_id": who.sid, "basis": [spec, plan], "target": plan,
        "content_hash": _bytes_hash(b"v2\n"),
    }, who.principal)
    try:
        assert delayed.entered.wait(5.0), "the commit never reached its compare-and-swap"
        peer_write = _Background(desk.client, "/hooks/post-edit-cas", {
            "session_id": peer.sid, "path": "spec.md", "expected_version": 1,
            "content_hash": _bytes_hash(b"s2\n"),
        }, peer.principal)
        peer_waited = not peer_write.answered_within(0.3)
    finally:
        delayed.release()

    assert commit.result() == (
        200, {"answer": "proceed", "committed": {"path": _PATH, "version": 2}})
    assert peer_waited, "the peer's write landed inside the commit's hold"
    assert peer_write.result() == (200, {"ok": True, "version": 2})
    assert (_version(desk, _PATH), _version(desk, "spec.md")) == (2, 2)


# -- a coordinator that cannot answer in time holds --------------------------


@pytest.mark.parametrize(
    ("route", "expected"),
    [(_CHECK, _CHECK_DEGRADED), (_COMMIT, _COMMIT_NOTHING_RAN)],
    ids=["check", "commit"])
def test_a_stalled_registry_holds_within_the_handler_deadline(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch, route: str, expected: dict,
) -> None:
    """With the registry lock held, the route stops waiting for it a reserve
    before the handler watchdog and answers, from inside its own work, a hold
    that says nothing ran and the same request may succeed later: the check's
    is byte for byte its degraded hold, the commit's the nothing-ran hold. It
    counts one watchdog timeout and moves nothing."""
    import ccs.adapters.claude_code.coordinator_server as mod

    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)
    body: dict = {"session_id": who.sid, "basis": [plan]}
    if route == _COMMIT:
        body |= {"target": plan, "content_hash": _bytes_hash(b"v2\n")}
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", 2.0)
    monkeypatch.setattr(mod, "_BASIS_LOCK_RESERVE_SEC", 1.5)
    timeouts_before = desk.coordinator._watchdog_timeouts_total

    with _HeldRegistryLock(desk.coordinator) as held:
        started = time.monotonic()
        answer = _Background(desk.client, route, body, who.principal)
        answered = answer.answered_within(1.8)
        waited = time.monotonic() - started
        held.release()

    status, reply = answer.result()
    assert answered, f"no answer after {waited:.2f}s: the route waited out the registry lock"
    assert status == 200 and list(reply.items()) == list(expected.items()), reply
    assert desk.coordinator._watchdog_timeouts_total == timeouts_before + 1
    assert _version(desk, _PATH) == 1


def _late(
    desk: _PlanDesk,
    monkeypatch: pytest.MonkeyPatch,
    send: Callable[[], tuple[int, dict]],
    delayed: _DelayedCall,
) -> tuple[tuple[int, dict], dict]:
    """Send a request whose work ``delayed`` holds past a shortened watchdog
    deadline, release it, and wait for the abandoned body to settle. Returns
    the answer and how each watchdog counter moved."""
    import ccs.adapters.claude_code.coordinator_server as mod

    coordinator = desk.coordinator

    def counters() -> tuple[int, int, int]:
        return (coordinator._watchdog_timeouts_total, coordinator._watchdog_late_aborts_total,
                coordinator._watchdog_late_completion_total)

    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    before = counters()
    try:
        response = send()
        assert delayed.entered.is_set(), "the work never reached the delayed call"
    finally:
        delayed.release()
    _await_abandoned_body_settled(
        coordinator, aborts_before=before[1], completions_before=before[2])
    moved = [after - prior for after, prior in zip(counters(), before)]
    return response, dict(zip(("timeouts", "late_aborts", "late_completions"), moved))


def test_a_commit_abandoned_before_its_hold_answers_outcome_unknown_and_lands_nothing(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The commit's work reaches the locks only after the watchdog answered.
    The caller is told the outcome is unknown, never that nothing landed; the
    late work aborts at the registry guard and the target does not move."""
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)
    delayed = _DelayedCall(
        monkeypatch, desk.coordinator.service, "commit_if_basis", lambda *a, **kw: True)

    response, moved = _late(desk, monkeypatch, lambda: _post(
        desk, _COMMIT, who, basis=[plan], target=plan, content_hash=_bytes_hash(b"v2\n")),
        delayed)

    assert response[0] == 200, response
    assert list(response[1].items()) == list(_COMMIT_OUTCOME_UNKNOWN.items()), response
    assert moved == {"timeouts": 1, "late_aborts": 1, "late_completions": 0}
    assert _version(desk, _PATH) == 1


def test_a_commit_abandoned_inside_its_hold_answers_outcome_unknown(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """The commit won the guard and then outlived the watchdog: it lands
    afterwards, which is exactly why its answer may not say nothing landed.
    One late completion, reported as one that may have left a write."""
    import ccs.adapters.claude_code.coordinator_server as mod

    caplog.set_level(logging.WARNING, logger=mod.__name__)
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)
    delayed = _DelayedCall(
        monkeypatch, desk.coordinator.registry, "commit_cas_if_basis", lambda *a, **kw: True)

    response, moved = _late(desk, monkeypatch, lambda: _post(
        desk, _COMMIT, who, basis=[plan], target=plan, content_hash=_bytes_hash(b"v2\n")),
        delayed)

    assert response == (200, _COMMIT_OUTCOME_UNKNOWN)
    assert moved == {"timeouts": 1, "late_aborts": 0, "late_completions": 1}
    assert _version(desk, _PATH) == 2, "the late commit landed after the outcome-unknown answer"
    assert _late_completion_criticals(caplog) == 1


def test_a_late_check_counts_a_late_completion_and_raises_no_orphaned_grant_alarm(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A check that won its locks and finished after the watchdog answered took
    no write: one late completion, logged at WARNING, never the CRITICAL that
    sends an operator looking for orphaned grants."""
    import ccs.adapters.claude_code.coordinator_server as mod

    caplog.set_level(logging.WARNING, logger=mod.__name__)
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)
    delayed = _DelayedCall(
        monkeypatch, desk.coordinator.registry, "basis_rows", lambda *a, **kw: True)

    response, moved = _late(
        desk, monkeypatch, lambda: _post(desk, _CHECK, who, basis=[plan]), delayed)

    assert response[0] == 200, response
    assert list(response[1].items()) == list(_CHECK_DEGRADED.items()), response
    assert moved == {"timeouts": 1, "late_aborts": 0, "late_completions": 1}
    assert _late_completion_criticals(caplog) == 0
    assert _late_completion_warnings(caplog) == 1


def test_a_check_that_finds_the_lock_held_after_the_watchdog_answered_is_counted_once(
    desk: _PlanDesk, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check's work reaches the locks only after the watchdog answered,
    and then finds the registry lock held. The watchdog's degraded hold was
    the answer sent, and it was counted; the late work's own lock timeout
    records nothing more and ends as a late abort, not a late completion."""
    who = _claimed(desk.client)
    plan = _recorded(desk, who, _PATH)
    delayed = _DelayedCall(
        monkeypatch, desk.coordinator.service, "check_basis", lambda *a, **kw: True)

    with _HeldRegistryLock(desk.coordinator) as held:
        response, moved = _late(
            desk, monkeypatch, lambda: _post(desk, _CHECK, who, basis=[plan]), delayed)
        held.release()

    assert response[0] == 200, response
    assert list(response[1].items()) == list(_CHECK_DEGRADED.items()), response
    assert moved == {"timeouts": 1, "late_aborts": 1, "late_completions": 0}
