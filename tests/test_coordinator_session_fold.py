# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The session, handoff and workspace routes, which only the Python
coordinator serves, reach one row per file under any spelling the workspace
disk treats as that file, and treat two spellings of one file in one request
as each route treats a name sent twice: refused where a repeat would merge two
writes into one, collapsed where it only reads.

Checkpoint members that are not files (objects, forward-only members) are
keys: stored and matched exactly as sent, never tidied or folded.

Most tests force the profile, so they state the outcome on a disk that folds
case and on one that folds nothing. A test about the disk's own spelling runs
on the real disk's profile.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ccs.adapters.claude_code.fs_profile import FoldProfileResult
from ccs.coordinator.registry_protocol import CheckpointMember
from ccs.core.types import WorkspaceRestoreWrite
from tests import test_coordinator_door_fold as door_fold
from tests.test_claude_code_coordinator_server import _claimed, _hash, _read, _Session
from tests.test_coordinator_door_fold import (
    CASELESS,
    SID_A,
    TRACK_ALL,
    Door,
    make_workspace,
    rows_of,
    write_file,
)

# The hook routes' fixtures: a served coordinator on a forced or probed profile.
forced_profile = door_fold.forced_profile
open_door = door_fold.open_door

NONE = FoldProfileResult(profile="none", status="classified")
PROFILES = pytest.mark.parametrize("profile", [CASELESS, NONE], ids=["caseless", "none"])
FILE_TIERS = {"arbitration_tier": "no-arbiter", "restore_tier": "restorable-unpinned"}
OBJECT_TIERS = {"arbitration_tier": "native-cas", "restore_tier": "restorable"}
WRITES_REPEAT = {"error": "writes contains duplicate paths"}
GRANTS_REPEAT = {"error": "grants contains duplicate paths"}
MEMBERS_REPEAT = {"error": "members contains duplicate member_path values"}
REGISTER_REPEAT = {"error": "writes contains duplicate member_path values"}
FINGERPRINT = _hash("session-fold-captured")


def folds(profile: FoldProfileResult) -> bool:
    return profile.profile == "caseless"


@pytest.fixture
def served(tmp_path: Path, open_door):
    """A door over a workspace that tracks everything, on a forced profile."""

    def serve(profile: FoldProfileResult | None) -> Door:
        return open_door(make_workspace(tmp_path / "ws", TRACK_ALL), profile)

    return serve


def begin(door: Door, read_set: list[str]) -> dict:
    status, body = door.post("/session/begin", {"session_id": SID_A, "read_set": read_set})
    assert status == 200 and body.get("ok") is True, body
    return body


def commit_all(door: Door, token: str, writes: list[tuple[str, str]]) -> tuple[int, dict]:
    return door.post("/session/commit_all", {
        "session_id": SID_A, "session_token": token,
        "writes": [{"path": path, "content": content} for path, content in writes],
    })


def member(member_path: str, tiers: dict | None = None) -> dict:
    return {"member_path": member_path, "fingerprint": FINGERPRINT, "captured_at": 1.0, **(tiers or {})}


def checkpoint(door: Door, members: list[dict]) -> tuple[int, dict]:
    return door.post("/workspace/checkpoint", {
        "session_id": SID_A, "name": "fold", "window_min": 1.0, "window_max": 1.0,
        "members": members,
    })


def stored_members(door: Door, checkpoint_id: str) -> list[str]:
    return sorted(row.member_path for row in door.registry.get_checkpoint_members(checkpoint_id))


def recorded_checkpoint(door: Door, member_paths: list[str]) -> str:
    """A checkpoint an older build recorded: its route stored whatever it was sent."""
    record = door.server.service.create_workspace_checkpoint(
        name="older", owner=uuid4(), window_min=1.0, window_max=1.0,
        members=[
            CheckpointMember(
                member_path=path, artifact_id=None, native_token="v1",
                fingerprint=FINGERPRINT, captured_at=1.0, **FILE_TIERS,
            )
            for path in member_paths
        ],
    )
    return record.checkpoint_id


def transfer(door: Door, giver: _Session, successor: _Session, paths: list[str]) -> tuple[int, dict]:
    return door.post("/handoff/transfer", {
        "session_id": giver.sid, "successor": successor.agent,
        "grants": [{"path": path} for path in paths],
    }, principal=giver.principal)


# ------------------------------------------------------------------ sessions


@PROFILES
def test_begin_pins_two_spellings_of_one_file_once_and_answers_each_spelling(
    profile: FoldProfileResult, served
) -> None:
    door = served(profile)

    session = begin(door, ["a.txt", "A.txt"])
    repeated = begin(door, ["b.txt", "b.txt"])

    # Each spelling the request named finds its pin under its own bytes, as a
    # name sent twice does.
    assert session["cut"] == {"a.txt": 1, "A.txt": 1}
    assert repeated["cut"] == {"b.txt": 1}
    assert len(door.registry.get_session_cut(repeated["session_token"])) == 1
    pins = door.registry.get_session_cut(session["session_token"])
    if folds(profile):
        assert len(pins) == 1
        assert rows_of(door.server.coordinator_root) == [("a.txt", 1), ("b.txt", 1)]
    else:
        assert len(pins) == 2
        assert rows_of(door.server.coordinator_root) == [("A.txt", 1), ("a.txt", 1), ("b.txt", 1)]


def test_begin_names_a_new_row_by_the_disks_spelling(tmp_path: Path, open_door) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_ALL)
    write_file(root, "data/x.txt")
    door = open_door(root)

    session = begin(door, ["DATA/X.TXT"])

    assert session["cut"] == {"DATA/X.TXT": 1}
    expected = "data/x.txt" if door.folds_case else "DATA/X.TXT"
    assert rows_of(root) == [(expected, 1)]


@PROFILES
def test_commit_all_naming_two_spellings_of_one_file_is_refused_as_a_duplicate(
    profile: FoldProfileResult, served
) -> None:
    door = served(profile)
    token = begin(door, ["a.txt", "A.txt"])["session_token"]

    answer = commit_all(door, token, [("a.txt", "one"), ("A.txt", "two")])

    if folds(profile):
        # One of the two writes would vanish into the other.
        assert answer == (400, WRITES_REPEAT)
        assert rows_of(door.server.coordinator_root) == [("a.txt", 1)]
        return
    assert answer[0] == 200 and answer[1].get("ok") is True, answer
    assert answer[1]["versions"] == {"a.txt": 2, "A.txt": 2}
    assert rows_of(door.server.coordinator_root) == [("A.txt", 2), ("a.txt", 2)]


def test_a_tidy_and_an_untidy_spelling_are_one_write_on_every_profile(served) -> None:
    door = served(NONE)
    token = begin(door, ["a.txt"])["session_token"]

    answer = commit_all(door, token, [("a.txt", "one"), (".//a.txt", "two")])

    assert answer == (400, WRITES_REPEAT)
    assert rows_of(door.server.coordinator_root) == [("a.txt", 1)]


@pytest.mark.parametrize("mode", ["caseless", "a start without the writer lock"])
def test_session_read_and_commit_reach_the_files_row_under_another_spelling(
    mode: str, served, request: pytest.FixtureRequest
) -> None:
    profile = CASELESS
    alias = "A.TXT"
    if mode == "a start without the writer lock":
        request.getfixturevalue("two_writer_store")
        profile, alias = None, ".//a.txt"
    door = served(profile)
    token = begin(door, ["a.txt"])["session_token"]
    session = {"session_id": SID_A, "session_token": token, "path": alias}

    read = door.post("/session/read", session)
    commit = door.post("/session/commit", {**session, "content": "new"})
    batch = commit_all(door, begin(door, ["a.txt"])["session_token"], [(alias, "newer")])

    assert read[1].get("ok") is True, read
    assert commit[1].get("ok") is True and commit[1].get("version") == 2, commit
    assert batch[1].get("versions") == {alias: 3}, batch
    assert rows_of(door.server.coordinator_root) == [("a.txt", 3)]


# ------------------------------------------------------------------ handoff


@PROFILES
def test_a_transfer_naming_two_spellings_of_one_file_is_refused_as_a_duplicate(
    profile: FoldProfileResult, served
) -> None:
    door = served(profile)
    giver, successor = _claimed(door.client), _claimed(door.client)
    _read(door.client, giver, "a.txt")

    status, answer = transfer(door, giver, successor, ["a.txt", "A.txt"])

    if folds(profile):
        assert (status, answer) == (400, GRANTS_REPEAT)
        row = door.registry.lookup_artifact_id_by_name("a.txt")
        assert door.registry.get_transfer_record(row) is None
        return
    assert status == 200, answer
    assert [(g["path"], g["transferred"]) for g in answer["grants"]] == [("a.txt", True), ("A.txt", False)]


@pytest.mark.parametrize("mode", ["caseless", "a start without the writer lock"])
def test_a_handoff_reaches_the_files_record_under_another_spelling_and_answers_in_it(
    mode: str, served, request: pytest.FixtureRequest
) -> None:
    profile, given, accepted = CASELESS, "A.TXT", "a.TXT"
    if mode == "a start without the writer lock":
        request.getfixturevalue("two_writer_store")
        profile, given, accepted = None, "./a.txt", ".//a.txt"
    door = served(profile)
    giver, successor = _claimed(door.client), _claimed(door.client)
    _read(door.client, giver, "a.txt")

    status, answer = transfer(door, giver, successor, [given])
    accept = door.post(
        "/handoff/accept", {"session_id": successor.sid, "path": accepted}, principal=successor.principal
    )

    assert status == 200, answer
    assert [(g["path"], g["transferred"]) for g in answer["grants"]] == [(given, True)]
    assert accept[1].get("ok") is True, accept
    assert rows_of(door.server.coordinator_root) == [("a.txt", 1)]


# ------------------------------------------------------------------ checkpoints


@PROFILES
@pytest.mark.parametrize("second", ["A.txt", "./a.txt"])
def test_a_checkpoint_naming_one_file_twice_is_refused_as_a_duplicate(
    second: str, profile: FoldProfileResult, served
) -> None:
    door = served(profile)

    answer = checkpoint(door, [member("a.txt", FILE_TIERS), member(second, FILE_TIERS)])

    # Tidying names one file on every disk; case only where the disk folds it.
    if second == "./a.txt" or folds(profile):
        assert answer == (400, MEMBERS_REPEAT)
        assert door.registry.list_checkpoints() == []
        return
    assert answer[0] == 200 and answer[1].get("ok") is True, answer
    assert stored_members(door, answer[1]["checkpoint_id"]) == ["A.txt", "a.txt"]


def test_a_checkpoint_stores_a_file_members_path_as_sent(served) -> None:
    door = served(CASELESS)

    status, answer = checkpoint(door, [member(".//Data/X.txt", FILE_TIERS)])

    assert status == 200 and answer.get("ok") is True, answer
    assert stored_members(door, answer["checkpoint_id"]) == [".//Data/X.txt"]


def test_a_checkpoint_keeps_object_members_that_differ_only_by_case_byte_for_byte(served) -> None:
    door = served(CASELESS)

    status, answer = checkpoint(
        door, [member("s3://B/Key", OBJECT_TIERS), member("s3://B/key", OBJECT_TIERS)]
    )

    assert status == 200 and answer.get("ok") is True, answer
    assert stored_members(door, answer["checkpoint_id"]) == ["s3://B/Key", "s3://B/key"]


def test_an_older_clients_checkpoint_keeps_its_members_exact(served) -> None:
    door = served(CASELESS)
    # No name kind and no tiers: the wire's defaults, a forward-only member.
    bare = [{"member_path": path, "captured_at": 1.0} for path in ("s3://B/Key", "s3://B/key")]

    status, answer = checkpoint(door, bare)
    native = checkpoint(door, [member("s3://B//Key", OBJECT_TIERS), member("s3://B/./Key", OBJECT_TIERS)])

    assert status == 200 and answer.get("ok") is True, answer
    assert stored_members(door, answer["checkpoint_id"]) == ["s3://B/Key", "s3://B/key"]
    assert native[0] == 200 and native[1].get("ok") is True, native
    assert stored_members(door, native[1]["checkpoint_id"]) == ["s3://B/./Key", "s3://B//Key"]


# ------------------------------------------------------------------ restore


@PROFILES
def test_restoring_a_checkpoint_recorded_with_two_spellings_of_one_file_is_refused(
    profile: FoldProfileResult, served
) -> None:
    door = served(profile)
    checkpoint_id = recorded_checkpoint(door, ["a.txt", "A.txt"])
    writes = [{"member_path": path, "fingerprint": FINGERPRINT} for path in ("a.txt", "A.txt")]

    status, answer = door.post("/workspace/restore/register", {
        "session_id": SID_A, "checkpoint_id": checkpoint_id, "writes": writes,
    })

    if folds(profile):
        assert (status, answer) == (400, REGISTER_REPEAT)
        assert rows_of(door.server.coordinator_root) == []
        assert door.registry.get_checkpoint(checkpoint_id).registered_by is None
        return
    # Two files, each first seen at its captured fingerprint.
    assert status == 200 and answer.get("ok") is True, answer
    assert answer["skipped"] == ["a.txt", "A.txt"]
    assert rows_of(door.server.coordinator_root) == [("A.txt", 1), ("a.txt", 1)]


def test_the_service_refuses_a_restore_naming_one_file_twice_before_any_claim(served) -> None:
    door = served(CASELESS)
    checkpoint_id = recorded_checkpoint(door, ["a.txt", "A.txt"])
    writes = [WorkspaceRestoreWrite(member_path=path, fingerprint=FINGERPRINT) for path in ("a.txt", "A.txt")]

    with pytest.raises(ValueError, match="duplicate member_path 'A.txt'"):
        door.server.service.register_workspace_restore(
            checkpoint_id=checkpoint_id, controller=uuid4(), writes=writes
        )

    assert rows_of(door.server.coordinator_root) == []
    assert door.registry.get_checkpoint(checkpoint_id).registered_by is None
