# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The route door: every hook route that takes a file name reaches the one row
for that file under any spelling the workspace disk treats as the same,
answers in the request's own spelling, and keeps a key (a name that is not a
file) exactly as sent.

Most tests run on the real disk's profile, probed as the coordinator probes
it: case-insensitive APFS folds case and accent forms, a Linux disk folds
nothing. Where an outcome depends on the disk, each test states both: one
file on a disk that folds case, two files on one that does not. A test about
how names are keyed rather than about the disk forces the profile instead.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import pytest

import ccs.adapters.claude_code.coordinator_server as server_module
from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.bash_path_detector import detect_tracked_paths
from ccs.adapters.claude_code.bash_write_detector import detect_tracked_writes
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.fs_profile import FoldProfileResult
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.disk_spelling import native_spelling
from ccs.core.path_fold import FOLD_TABLE_UNICODE_VERSION
from ccs.core.states import MESIState
from tests.test_claude_code_coordinator_server import (
    _DEGRADE_DEADLINE_SEC,
    _GATE_ANSWER_BOUND_SEC,
    _OPERATOR_HEADER,
    _Background,
    _claimed,
    _Client,
    _hand_off,
    _hash,
    _HeldRegistryLock,
    _sid,
)

SID_A, SID_B, SID_C = _sid("door-a"), _sid("door-b"), _sid("door-c")
HASH_1, HASH_2 = _hash("door-1"), _hash("door-2")
PATH_ERROR = "missing or empty path"
LIMIT_ERROR = "path exceeds 1024 chars"
TRACK_DATA = {"tracked.yaml": '- "data/**"\n'}
TRACK_ALL = {"tracked.yaml": '- "**"\n'}
STRICT_PLAN = {"strict_mode.yaml": '- "plan.md"\n'}
CANNOT_TELL = FoldProfileResult(profile="none", status="cannot_tell", reason="test")
CASELESS = FoldProfileResult(profile="caseless", status="classified")
ROOT_PREFIX = "door-fold-root-"

Answer = tuple[int, dict]


@dataclass
class Door:
    server: CoordinatorHTTPServer
    client: _Client
    closed: bool = False

    @property
    def registry(self) -> SqliteArtifactRegistry:
        return self.server.registry

    @property
    def folds_case(self) -> bool:
        return self.registry.fold_index_profile == "caseless"

    def post(self, route: str, body: dict, **kwargs) -> Answer:
        return self.client.post(route, body, **kwargs)

    def as_session(self, session_id: str) -> Callable[..., Answer]:
        return lambda route, **fields: self.post(route, {"session_id": session_id, **fields})

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.server.shutdown()


@pytest.fixture
def forced_profile(monkeypatch: pytest.MonkeyPatch) -> dict[str, FoldProfileResult | None]:
    """The probe result the next start answers; ``None`` probes the real disk."""
    forced: dict[str, FoldProfileResult | None] = {"result": None}
    real_probe = CoordinatorHTTPServer._probe_fold_profile

    def probe(self: CoordinatorHTTPServer) -> FoldProfileResult:
        result = forced["result"]
        return real_probe(self) if result is None else result

    monkeypatch.setattr(CoordinatorHTTPServer, "_probe_fold_profile", probe)
    return forced


@pytest.fixture
def open_door(forced_profile) -> Iterator[Callable[..., Door]]:
    """Start a served coordinator on a workspace, the way lifecycle starts one."""
    doors: list[Door] = []

    def open_(root: Path, profile: FoldProfileResult | None = None) -> Door:
        forced_profile["result"] = profile
        server = CoordinatorHTTPServer(root, port=0)
        server.serve_in_thread()
        door = Door(server, _Client("127.0.0.1", server.port, load_secret(server.coordinator_root)))
        doors.append(door)
        return door

    yield open_
    for door in doors:
        door.close()


def file_request(server: CoordinatorHTTPServer, requested: str):
    """The door's reading of a file name, as the routes take it."""
    from ccs.adapters.claude_code import door

    return door.file_request(server, requested)


def make_workspace(root: Path, policy: dict[str, str] | None = None) -> Path:
    (root / ".coherence").mkdir(parents=True, mode=0o700)
    for name, text in (policy or {}).items():
        (root / ".coherence" / name).write_text(text, encoding="utf-8")
    return root


def write_file(root: Path, name: str, text: str = "x\n") -> None:
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text, encoding="utf-8")


def _store(root: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{root / '.coherence' / 'state.db'}?mode=ro", uri=True)


def rows_of(root: Path) -> list[tuple[str, int]]:
    """Every row's name and version, by name."""
    conn = _store(root)
    try:
        return sorted(conn.execute("SELECT name, version FROM artifacts").fetchall())
    finally:
        conn.close()


def row_names(door: Door) -> list[str]:
    return [name for name, _ in rows_of(door.server.coordinator_root)]


def state_of(door: Door, name: str, session_id: str) -> MESIState | None:
    conn = _store(door.server.coordinator_root)
    try:
        row = conn.execute("SELECT id FROM artifacts WHERE name = ?", (name,)).fetchone()
    finally:
        conn.close()
    assert row is not None, f"no row named {name}: {row_names(door)}"
    return door.registry.get_agent_state(UUID(hex=row[0]), session_to_agent_id(session_id))


def context_of(body: dict) -> str:
    return str(body.get("hookSpecificOutput", {}).get("additionalContext", ""))


def commit_as_b(door: Door, name: str) -> None:
    """Session B commits ``name``, so a session that read it before holds a stale copy."""
    b = door.as_session(SID_B)
    status, edit = b("/hooks/pre-edit", path=name)
    assert status == 200 and edit.get("ok") is True, edit
    assert b("/hooks/post-edit", path=name, success=True, content_hash=HASH_2) == (200, {"ok": True})


@pytest.fixture
def data_door(tmp_path: Path, open_door) -> Door:
    """A door over a workspace that tracks data/** and holds data/x.txt."""
    root = make_workspace(tmp_path / "ws", TRACK_DATA)
    write_file(root, "data/x.txt")
    return open_door(root)


# ------------------------------------------------------------------ aliases


def test_an_alias_pre_edit_meets_the_holder_of_the_file_as_its_own_spelling_would(
    data_door: Door,
) -> None:
    door = data_door
    held = door.post("/hooks/pre-edit", {"session_id": SID_A, "path": "data/x.txt"})
    opted = _claimed(door.client)
    opted_in = door.post(
        "/hooks/pre-edit", {"session_id": opted.sid, "path": "DATA/X.TXT", "if_unheld": True},
        principal=opted.principal,
    )
    ordinary = door.post("/hooks/pre-edit", {"session_id": SID_C, "path": "DATA/X.TXT"})

    assert held == (200, {"ok": True})
    assert row_names(door) == ["data/x.txt"]
    if not door.folds_case:
        # Another file on this disk, and one data/** does not name.
        assert [opted_in, ordinary] == [(200, {"ok": True}), (200, {"ok": True})]
        assert state_of(door, "data/x.txt", SID_A) == MESIState.EXCLUSIVE
        return
    holder = session_to_agent_id(SID_A).hex
    assert opted_in == (200, {"ok": False, "reason": "other_holder", "holder_agent_id": holder})
    assert (ordinary[1].get("ok"), ordinary[1].get("collision")) == (True, True)
    assert "DATA/X.TXT" in context_of(ordinary[1])
    assert state_of(door, "data/x.txt", SID_A) == MESIState.INVALID
    assert state_of(door, "data/x.txt", SID_C) == MESIState.EXCLUSIVE


def test_an_ordinary_pre_edit_meets_a_write_holder_on_its_fold_buckets_other_row(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_ALL)
    # Two rows for one file, as a start that folds nothing keeps them.
    split = open_door(root, CANNOT_TELL)
    split.post("/hooks/pre-read", {"session_id": SID_A, "path": "data/x.txt", "content_hash": HASH_1})
    split.post("/hooks/pre-edit", {"session_id": SID_C, "path": "DATA/X.TXT"})
    split.close()
    # The file is missing at this start, so the rows stay apart as one bucket.
    door = open_door(root, CASELESS)
    write_file(root, "data/x.txt")

    status, edit = door.post("/hooks/pre-edit", {"session_id": SID_B, "path": "data/x.txt"})

    assert row_names(door) == ["DATA/X.TXT", "data/x.txt"]
    assert (status, edit.get("ok"), edit.get("collision")) == (200, True, True)
    # C's write may still be landing, so a disk unlike the record proves nothing.
    assert "disk_differs" not in edit
    assert state_of(door, "DATA/X.TXT", SID_C) == MESIState.INVALID
    assert state_of(door, "data/x.txt", SID_B) == MESIState.EXCLUSIVE


def test_post_edit_cas_and_post_edit_through_an_alias_commit_the_file_the_alias_names(
    data_door: Door,
) -> None:
    door = data_door
    a, b = door.as_session(SID_A), door.as_session(SID_B)
    read = b("/hooks/pre-read", path="data/x.txt", content_hash=HASH_1)
    cas = b("/hooks/post-edit-cas", path="DATA/x.txt", content_hash=HASH_2, expected_version=1)
    edit = a("/hooks/pre-edit", path="DATA/X.TXT")
    commit = a("/hooks/post-edit", path="Data/x.TXT", success=True, content_hash=HASH_1)

    assert read[1].get("status") == "fresh"
    assert commit == (200, {"ok": True})
    versions = rows_of(door.server.coordinator_root)
    if not door.folds_case:
        # The aliases name other files, which data/** does not track.
        assert [cas, edit] == [(200, {"ok": True}), (200, {"ok": True})]
        assert versions == [("data/x.txt", 1)]
        return
    assert cas == (200, {"ok": True, "version": 2})
    assert edit[1].get("ok") is True
    assert versions == [("data/x.txt", 3)]
    assert state_of(door, "data/x.txt", SID_B) == MESIState.INVALID


def test_an_answer_repeats_the_path_in_the_bytes_the_request_sent(data_door: Door) -> None:
    door = data_door
    a = door.as_session(SID_A)
    a("/hooks/pre-read", path="data/x.txt", content_hash=HASH_1)
    commit_as_b(door, "data/x.txt")

    untidy = a("/hooks/pre-read", path=".//data/x.txt")
    bash = door.as_session(SID_C)("/hooks/pre-bash", command="cat ./data//x.txt")

    assert untidy[1].get("status") == "stale", untidy
    assert untidy[1]["summary"]["path"] == ".//data/x.txt"
    assert bash[1].get("stale_paths") == ["./data//x.txt"], bash
    if door.folds_case:
        assert a("/hooks/pre-read", path="DATA/X.TXT")[1].get("status") == "fresh"
        commit_as_b(door, "data/x.txt")
        stale = a("/hooks/pre-read", path="DATA/X.TXT")
        assert stale[1]["summary"]["path"] == "DATA/X.TXT"
    assert row_names(door) == ["data/x.txt"]


@pytest.mark.parametrize(
    "mode", ["the disk's own profile", "a start that cannot tell", "a start without the writer lock"]
)
def test_dot_slash_spellings_reach_the_tidy_row_on_every_start(
    mode: str, tmp_path: Path, open_door, request: pytest.FixtureRequest
) -> None:
    profile = CANNOT_TELL if mode == "a start that cannot tell" else None
    if mode == "a start without the writer lock":
        request.getfixturevalue("two_writer_store")
    root = make_workspace(tmp_path / "ws", TRACK_DATA)
    write_file(root, "data/x.txt")
    door = open_door(root, profile)
    door.post("/hooks/pre-read", {"session_id": SID_A, "path": "data/x.txt", "content_hash": HASH_1})

    read = door.post("/hooks/pre-read", {"session_id": SID_B, "path": ".//data/x.txt"})
    edit = door.post("/hooks/pre-edit", {"session_id": SID_C, "path": "././data/x.txt"})

    assert read[1].get("status") == "stale", read
    assert edit[1].get("ok") is True, edit
    assert row_names(door) == ["data/x.txt"]
    assert state_of(door, "data/x.txt", SID_A) == MESIState.INVALID
    assert state_of(door, "data/x.txt", SID_C) == MESIState.EXCLUSIVE


def test_a_strict_plan_md_denies_a_stale_edit_spelled_PLAN_MD_where_the_disk_folds_case(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", STRICT_PLAN)
    write_file(root, "plan.md")
    door = open_door(root)
    door.post("/hooks/pre-read", {"session_id": SID_A, "path": "plan.md", "content_hash": HASH_1})
    commit_as_b(door, "plan.md")

    status, edit = door.post("/hooks/pre-edit", {"session_id": SID_A, "path": "PLAN.MD"})

    if not door.folds_case:
        assert (status, edit) == (200, {"ok": True})
        return
    assert (edit.get("ok"), edit.get("status")) == (False, "stale")
    assert edit["summary"]["path"] == "PLAN.MD"
    assert edit["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert state_of(door, "plan.md", SID_A) == MESIState.INVALID


def test_a_strict_deny_under_one_spelling_counts_a_shell_read_of_another_as_a_route_around(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", STRICT_PLAN)
    write_file(root, "plan.md")
    door = open_door(root, CASELESS)
    a = door.as_session(SID_A)
    a("/hooks/pre-read", path="plan.md", content_hash=HASH_1)
    commit_as_b(door, "plan.md")
    before = door.server.counters_snapshot()["strict_mode_routed_around_via_bash_total"]

    # Neither spelling is the file's key, which is plan.md.
    denied = a("/hooks/pre-read", path="PLAN.MD", content_hash=HASH_1)
    shell = a("/hooks/pre-bash", command="cat Plan.md")

    assert denied[1]["hookSpecificOutput"]["permissionDecision"] == "deny", denied
    assert shell[1]["hookSpecificOutput"]["permissionDecision"] == "deny", shell
    after = door.server.counters_snapshot()["strict_mode_routed_around_via_bash_total"]
    assert after == before + 1


@pytest.mark.skipif(sys.platform != "darwin", reason="needs a disk that folds case")
def test_a_strict_glob_only_the_disks_spelling_matches_denies_a_stale_read_and_edit(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", {**TRACK_ALL, "strict_mode.yaml": '- "stra?e.md"\n'})
    write_file(root, "stra\u00dfe.md")
    door = open_door(root)
    if not door.folds_case:
        pytest.skip("the workspace disk does not fold case")
    a = door.as_session(SID_A)
    a("/hooks/pre-read", path="stra\u00dfe.md", content_hash=HASH_1)
    commit_as_b(door, "stra\u00dfe.md")

    # Folded, the glob's one character never meets "ss"; the disk's spelling
    # of the request is what it matches.
    read = a("/hooks/pre-read", path="STRASSE.MD", content_hash=HASH_1)
    edit = a("/hooks/pre-edit", path="STRASSE.MD")

    assert row_names(door) == ["stra\u00dfe.md"]
    assert read[1]["hookSpecificOutput"]["permissionDecision"] == "deny", read
    assert edit[1]["hookSpecificOutput"]["permissionDecision"] == "deny", edit


# ------------------------------------------------------------------ effect fence


def test_an_effect_fence_under_another_spelling_fences_the_files_row(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_DATA)
    write_file(root, "data/x.txt")
    door = open_door(root, CASELESS)
    a = door.as_session(SID_A)
    read = a("/hooks/pre-read", path="data/x.txt", content_hash=HASH_1, want_owner_generation=True)
    comparands = {
        "path": "DATA/X.TXT", "content_hash": HASH_1,
        "expected_version": read[1]["version"], "expected_generation": read[1]["owner_generation"],
    }

    proceed = a("/hooks/effect-fence", **comparands)
    commit_as_b(door, "data/x.txt")
    held = a("/hooks/effect-fence", **comparands)

    assert proceed == (200, {"verdict": "proceed"})
    assert held[1].get("verdict") == "hold", held
    assert row_names(door) == ["data/x.txt"]


# ------------------------------------------------------------------ pre-bash


def test_pre_bash_collapses_two_spellings_of_one_file_into_one_token_and_one_grant(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws")
    write_file(root, "plan.md")
    write_file(root, "spec.md")
    door = open_door(root)

    # A second token for a file the first token just registered would read as
    # stale: its grant is applied only after the deny decision.
    tidy = door.post("/hooks/pre-bash", {"session_id": SID_A, "command": "cat .//plan.md plan.md"})
    cased = door.post("/hooks/pre-bash", {"session_id": SID_B, "command": "cat spec.md SPEC.MD"})

    assert tidy == (200, {"status": "fresh"})
    assert cased == (200, {"status": "fresh"})
    assert row_names(door) == ["plan.md", "spec.md"]
    assert state_of(door, "spec.md", SID_B) == MESIState.SHARED
    held = door.registry.artifacts_held_by_agent(session_to_agent_id(SID_B), [MESIState.SHARED])
    assert len(held) == 1


def test_a_givers_shell_write_under_another_spelling_is_denied(tmp_path: Path, open_door) -> None:
    root = make_workspace(tmp_path / "ws")
    write_file(root, "plan.md")
    door = open_door(root, CASELESS)
    giver, successor = _claimed(door.client), _claimed(door.client)
    _hand_off(door.client, giver, successor, "plan.md")

    status, answer = door.post(
        "/hooks/pre-bash", {"session_id": giver.sid, "command": "echo x >> PLAN.MD"},
        principal=giver.principal,
    )

    assert (status, answer.get("reason")) == (200, "handed_off"), answer


def test_the_bash_read_detector_keeps_the_first_tracked_spelling_of_each_key() -> None:
    assert detect_tracked_paths("cat plan.md PLAN.MD", lambda path: True, str.lower) == ["plan.md"]
    # An untracked first spelling does not hide a tracked later one.
    only_upper = lambda path: path == "A.MD"  # noqa: E731
    assert detect_tracked_paths("cat a.md A.MD", only_upper, str.lower) == ["A.MD"]


def test_the_bash_write_detector_keeps_the_first_tracked_spelling_of_each_key() -> None:
    command = "echo x >> plan.md; echo y >> PLAN.MD"
    assert detect_tracked_writes(command, lambda path: True, key_of=str.lower) == ["plan.md"]
    only_upper = lambda path: path == "PLAN.MD"  # noqa: E731
    assert detect_tracked_writes(command, only_upper, key_of=str.lower) == ["PLAN.MD"]


def test_the_bash_write_detector_takes_every_spelling_of_the_root_it_is_given() -> None:
    command = "echo x >> /Users/x/Repo/plan.md; echo y >> /Users/x/REPO/spec.md"
    roots = ("/Users/x/REPO", "/Users/x/Repo")
    assert detect_tracked_writes(command, lambda path: True, root=roots) == ["plan.md", "spec.md"]
    assert detect_tracked_writes(command, lambda path: True, root=roots[0]) == ["spec.md"]


def test_a_shell_write_under_the_disks_spelling_of_the_root_is_seen(tmp_path: Path, open_door) -> None:
    disk_root = native_spelling(make_workspace(tmp_path / "Cased"))
    given_root = Path(disk_root).parent / "cased"
    if not given_root.exists():
        pytest.skip("the workspace disk does not fold case")
    door = open_door(given_root)

    written = server_module._tracked_writes(door.server, f"echo x >> {disk_root}/plan.md")

    assert str(door.server.coordinator_root) != disk_root
    assert written == ["plan.md"]


def test_a_bash_token_over_the_fold_limit_is_never_folded(tmp_path: Path, open_door) -> None:
    root = make_workspace(tmp_path / "ws")
    door = open_door(root)
    long = f"{'d/' * 520}plan.md"

    first = door.post("/hooks/pre-bash", {"session_id": SID_A, "command": f"cat {long}"})
    dotted = door.post("/hooks/pre-bash", {"session_id": SID_A, "command": f"cat ./{long}"})

    assert first == (200, {"status": "fresh"})
    assert dotted == (200, {"status": "fresh"})
    assert row_names(door) == [long]
    assert file_request(door.server, long.upper()).key == long.upper()


def test_a_bash_token_over_the_fold_limit_only_by_dot_slash_padding_reaches_its_file(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", STRICT_PLAN)
    write_file(root, "plan.md")
    door = open_door(root)
    door.post("/hooks/pre-read", {"session_id": SID_A, "path": "plan.md", "content_hash": HASH_1})
    commit_as_b(door, "plan.md")
    padded = f"{'./' * 520}plan.md"

    _, bash = door.post("/hooks/pre-bash", {"session_id": SID_A, "command": f"cat {padded}"})

    assert row_names(door) == ["plan.md"]
    assert bash.get("status") == "stale", bash
    assert bash.get("stale_paths") == [padded]
    assert bash["hookSpecificOutput"]["permissionDecision"] == "deny"


# ------------------------------------------------------------------ disk spelling


def test_a_new_row_for_an_existing_file_is_named_as_the_disk_spells_it(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_DATA)
    write_file(root, "Data/X.txt")
    door = open_door(root)

    door.post("/hooks/pre-read", {"session_id": SID_A, "path": "data/x.txt", "content_hash": HASH_1})
    door.post("/hooks/pre-read", {"session_id": SID_B, "path": "DATA/x.txt", "content_hash": HASH_1})

    assert row_names(door) == (["Data/X.txt"] if door.folds_case else ["data/x.txt"])


def test_a_new_file_takes_the_disk_spelling_of_the_directories_it_is_created_in(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_ALL)
    (root / "Data").mkdir()
    door = open_door(root)

    door.post("/hooks/pre-edit", {"session_id": SID_A, "path": "DATA/New.txt"})

    assert row_names(door) == (["Data/New.txt"] if door.folds_case else ["DATA/New.txt"])


@pytest.mark.parametrize("mode", ["a start that cannot tell", "a start without the writer lock"])
def test_an_unfolded_start_advertises_cannot_tell_and_names_rows_as_requested(
    mode: str, tmp_path: Path, open_door, request: pytest.FixtureRequest
) -> None:
    profile = CANNOT_TELL
    if mode == "a start without the writer lock":
        request.getfixturevalue("two_writer_store")
        profile = None  # the real probe, which cannot tell without the lock
    root = make_workspace(tmp_path / "ws", TRACK_ALL)
    write_file(root, "Data/X.txt")
    door = open_door(root, profile)
    a = door.as_session(SID_A)

    a("/hooks/pre-edit", path="data/x.txt")
    commit = a("/hooks/post-edit", path="data/x.txt", success=True, content_hash=HASH_1)
    door.post("/hooks/pre-read", {"session_id": SID_B, "path": "DATA/X.TXT", "content_hash": HASH_1})

    assert door.client.get("/status")[1]["fold_profile"] == "cannot_tell"
    assert commit == (200, {"ok": True})
    assert row_names(door) == ["DATA/X.TXT", "data/x.txt"]


def test_a_request_through_a_link_keeps_its_own_name(tmp_path: Path, open_door) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_ALL)
    write_file(root, "plan.md")
    (root / "data").mkdir()
    (root / "data" / "link.md").symlink_to("../plan.md")
    door = open_door(root)

    door.post("/hooks/pre-read", {"session_id": SID_A, "path": "data/link.md", "content_hash": HASH_1})

    # The link reaches another file, so the disk has no spelling of this name.
    assert row_names(door) == ["data/link.md"]


def test_a_disk_spelling_outside_the_workspace_root_is_never_kept(tmp_path: Path, open_door) -> None:
    root = make_workspace(tmp_path / "ws")
    write_file(root, "Inside.md")
    door = open_door(root)

    # A Bash token is not refused for "..": it resolves beside the root.
    assert file_request(door.server, "../outside.md").disk_spelling is None
    inside = file_request(door.server, "inside.md").disk_spelling
    assert inside == ("Inside.md" if door.folds_case else None)


def _seeded_workspace(root: Path) -> Path:
    make_workspace(root, TRACK_DATA)
    write_file(root, "Data/X.txt")
    return root


def _row_names_through(root: str, open_door) -> tuple[bool, list[str]]:
    door = open_door(Path(root))
    door.post("/hooks/pre-read", {"session_id": SID_A, "path": "data/x.txt", "content_hash": HASH_1})
    folds_case = door.folds_case
    door.close()
    return folds_case, [name for name, _ in rows_of(door.server.coordinator_root)]


def _remove_tree(path: str) -> None:
    if os.path.isabs(path) and os.path.basename(path).startswith(ROOT_PREFIX):
        shutil.rmtree(path, ignore_errors=True)


def test_the_coordinator_names_a_new_row_as_the_disk_spells_it_however_its_root_is_spelled(
    open_door,
) -> None:
    # tempfile's directory is /var/... on macOS, a link to /private/var.
    base = tempfile.mkdtemp(prefix=ROOT_PREFIX)
    try:
        folds_case, names = _row_names_through(str(_seeded_workspace(Path(base) / "ws")), open_door)
        assert names == (["Data/X.txt"] if folds_case else ["data/x.txt"])
        if not folds_case:
            return
        _seeded_workspace(Path(base) / "cased")
        _, cased = _row_names_through(os.path.join(native_spelling(base), "CASED"), open_door)
        assert cased == ["Data/X.txt"]
    finally:
        _remove_tree(base)


@pytest.mark.skipif(sys.platform != "darwin", reason="needs a disk that folds case")
def test_a_file_request_reaches_a_key_row_named_as_the_disk_spells_the_file(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_ALL)
    write_file(root, "Data/X.txt")
    door = open_door(root)
    if not door.folds_case:
        pytest.skip("the workspace disk does not fold case")
    key = {"path": "Data/X.txt", "name_kind": "key"}
    door.post("/hooks/pre-read", {"session_id": SID_A, **key, "content_hash": HASH_1})
    door.post("/hooks/pre-edit", {"session_id": SID_B, **key})
    door.post("/hooks/post-edit", {"session_id": SID_B, **key, "success": True, "content_hash": HASH_2})

    # The file's disk spelling is the key row's exact name: registering the
    # file would return that row, so looking the file up must find it too.
    status, read = door.post(
        "/hooks/pre-read", {"session_id": SID_A, "path": "data/x.txt", "content_hash": HASH_1}
    )

    assert (status, read.get("status")) == (200, "stale"), read
    assert rows_of(root) == [("Data/X.txt", 2)]


def test_a_file_lookup_finds_the_key_row_its_register_returns(tmp_path: Path) -> None:
    registry = SqliteArtifactRegistry(tmp_path / "state.db")
    try:
        registry.build_fold_index("caseless")
        key_row = registry.resolve_or_register("Data/X.txt", "", name_kind="key")
        file_name = {"key": "data/x.txt", "disk_spelling": "Data/X.txt"}

        found = registry.lookup_artifact_id_by_name("data/x.txt", **file_name)
        registered = registry.resolve_or_register("data/x.txt", "", **file_name)

        assert found == registered == key_row
    finally:
        registry.close()


# ------------------------------------------------------------------ keys

KEY_ROUTES: list[tuple[str, dict]] = [
    ("/hooks/pre-read", {}),
    ("/hooks/pre-edit", {}),
    ("/hooks/post-edit", {"success": True, "content_hash": HASH_1}),
    ("/hooks/post-edit-cas", {"content_hash": HASH_1, "expected_version": 1}),
    ("/hooks/effect-fence", {"content_hash": HASH_1, "expected_version": 1, "expected_generation": None}),
]


def test_keys_that_differ_only_by_case_are_two_rows_before_and_after_a_restart(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", TRACK_ALL)
    before = open_door(root)
    a = before.as_session(SID_A)
    a("/hooks/pre-read", path="ws-42", name_kind="key", content_hash=HASH_1)
    a("/hooks/pre-read", path="WS-42", name_kind="key", content_hash=HASH_2)
    assert row_names(before) == ["WS-42", "ws-42"]
    before.close()

    after = open_door(root)
    cas = after.post("/hooks/post-edit-cas", {
        "session_id": SID_A, "path": "WS-42", "name_kind": "key",
        "content_hash": HASH_1, "expected_version": 1,
    })

    assert cas == (200, {"ok": True, "version": 2})
    assert rows_of(root) == [("WS-42", 2), ("ws-42", 1)]


def test_a_key_keeps_every_byte_it_was_sent_with(tmp_path: Path, open_door) -> None:
    door = open_door(make_workspace(tmp_path / "ws", TRACK_ALL))
    key = "s3://bucket//Key"
    a = door.as_session(SID_A)

    read = a("/hooks/pre-read", path=key, name_kind="key", content_hash=HASH_1)
    cas = a("/hooks/post-edit-cas", path=key, name_kind="key", content_hash=HASH_2, expected_version=1)
    as_file = door.post("/hooks/pre-read", {"session_id": SID_B, "path": key, "content_hash": HASH_1})

    assert read[1].get("status") == "fresh"
    assert cas == (200, {"ok": True, "version": 2})
    assert as_file[1].get("status") == "fresh"
    assert row_names(door) == [key, "s3:/bucket/Key"]


def test_a_file_edit_of_a_key_rows_exact_name_commits_to_the_row_the_key_shares(
    tmp_path: Path, open_door
) -> None:
    door = open_door(make_workspace(tmp_path / "ws", TRACK_ALL), CASELESS)
    key = {"path": "docs/plan.md", "name_kind": "key"}
    door.post("/hooks/pre-read", {"session_id": SID_A, **key, "content_hash": HASH_1})
    b = door.as_session(SID_B)

    edit = b("/hooks/pre-edit", path="docs/plan.md")
    commit = b("/hooks/post-edit", path="docs/plan.md", success=True, content_hash=HASH_2)
    key_read = door.post("/hooks/pre-read", {"session_id": SID_A, **key})

    assert [edit, commit] == [(200, {"ok": True}), (200, {"ok": True})]
    assert key_read[1].get("status") == "stale"
    assert rows_of(door.server.coordinator_root) == [("docs/plan.md", 2)]


def test_a_name_kind_other_than_file_or_key_is_refused_with_the_route_path_error(
    tmp_path: Path, open_door
) -> None:
    door = open_door(make_workspace(tmp_path / "ws", TRACK_ALL))
    for route, extra in KEY_ROUTES:
        for name_kind in ["blob", "KEY", 7, None, ["key"]]:
            body = {"session_id": SID_A, "path": "data/x.txt", "name_kind": name_kind, **extra}
            assert door.post(route, body) == (400, {"error": PATH_ERROR}), (route, name_kind)
        file_body = {"session_id": SID_A, "path": "data/x.txt", "name_kind": "file", **extra}
        assert door.post(route, file_body)[0] == 200, route
    assert row_names(door) == ["data/x.txt"]


# ------------------------------------------------------------------ the limit


def test_a_path_over_1024_code_points_is_refused_with_the_route_path_error_as_a_file_or_a_key(
    tmp_path: Path, open_door
) -> None:
    door = open_door(make_workspace(tmp_path / "ws", TRACK_ALL))
    long = "a" * 1025
    for route, extra in KEY_ROUTES:
        for kind in [{}, {"name_kind": "key"}]:
            body = {"session_id": SID_A, "path": long, **kind, **extra}
            assert door.post(route, body) == (400, {"error": LIMIT_ERROR}), (route, kind)
    grep = door.post("/hooks/pre-grep", {"session_id": SID_A, "search_root": long})
    assert grep == (400, {"error": LIMIT_ERROR})
    # Counted in code points: 1024 of them take 4096 UTF-8 bytes.
    astral = "\U0001f600" * 1024
    assert door.post("/hooks/pre-read", {"session_id": SID_A, "path": astral})[0] == 200
    assert row_names(door) == [astral]


# ------------------------------------------------------------------ pre-grep

GREP_ROWS = ["data/b.md", "été/d.md", "database/c.md", "data/a.md"]


def _seed_grep_rows(door: Door) -> None:
    # Registered out of name order: an answer lists them in name order.
    for path in GREP_ROWS:
        door.post("/hooks/pre-read", {"session_id": SID_A, "path": path, "content_hash": HASH_1})


def _stale_paths(answer: Answer) -> list[str]:
    return answer[1].get("stale_paths", [])


def test_pre_grep_under_DATA_reaches_data_rows_and_under_an_accented_root_on_a_folding_disk(
    tmp_path: Path, open_door
) -> None:
    door = open_door(make_workspace(tmp_path / "ws", TRACK_ALL))
    _seed_grep_rows(door)

    upper = door.post("/hooks/pre-grep", {"session_id": SID_B, "search_root": "DATA"})
    accented = door.post("/hooks/pre-grep", {"session_id": SID_C, "search_root": "ÉTÉ"})
    file_root = door.post("/hooks/pre-grep", {"session_id": SID_A, "search_root": "DATA/A.MD"})

    assert _stale_paths(upper) == ["data/a.md", "data/b.md"]
    assert _stale_paths(accented) == (["été/d.md"] if door.folds_case else [])
    # A root that is a file reaches that file. Session A read it, so it is fresh.
    assert file_root == (200, {"status": "fresh"})
    commit_as_b(door, "data/a.md")
    stale = door.post("/hooks/pre-grep", {"session_id": SID_A, "search_root": "DATA/A.MD"})
    assert _stale_paths(stale) == (["data/a.md"] if door.folds_case else [])


def test_pre_grep_on_a_disk_that_folds_nothing_finds_what_the_sql_prefix_match_found(
    tmp_path: Path, open_door
) -> None:
    door = open_door(make_workspace(tmp_path / "ws", TRACK_ALL), CANNOT_TELL)
    _seed_grep_rows(door)

    upper = door.post("/hooks/pre-grep", {"session_id": SID_B, "search_root": "DATA"})
    accented = door.post("/hooks/pre-grep", {"session_id": SID_C, "search_root": "ÉTÉ"})
    everything = door.post("/hooks/pre-grep", {"session_id": SID_C, "search_root": ""})

    # SQLite's LIKE folds ASCII case only.
    assert _stale_paths(upper) == ["data/a.md", "data/b.md"]
    assert _stale_paths(accented) == []
    assert sorted(_stale_paths(everything)) == sorted(GREP_ROWS)


def test_a_strict_row_pre_grep_finds_is_matched_under_the_disks_fold(
    tmp_path: Path, open_door
) -> None:
    root = make_workspace(tmp_path / "ws", {**TRACK_ALL, **STRICT_PLAN})
    door = open_door(root, CASELESS)
    # A row named in a case the strict pattern does not spell.
    door.post("/hooks/pre-read", {"session_id": SID_A, "path": "PLAN.MD", "content_hash": HASH_1})
    commit_as_b(door, "PLAN.MD")

    grep = door.post("/hooks/pre-grep", {"session_id": SID_A, "search_root": ""})

    assert row_names(door) == ["PLAN.MD"]
    assert grep[1]["hookSpecificOutput"]["permissionDecision"] == "deny", grep


# ------------------------------------------------------------------ policy routes


def test_untrack_refuses_a_spelling_of_a_strict_path_where_the_disk_folds_case(
    tmp_path: Path, open_door
) -> None:
    door = open_door(make_workspace(tmp_path / "ws", STRICT_PLAN))

    status, answer = door.post("/policy/untrack", {"paths": ["PLAN.MD"]})

    if not door.folds_case:
        assert (status, answer["ok"], answer["removed"]) == (200, True, ["PLAN.MD"])
        return
    assert (status, answer["reason"]) == (409, "untrack_strict_path")
    assert answer["refused"] == [{"path": "PLAN.MD", "strict_patterns": ["plan.md"]}]


def test_policy_entries_are_patterns_kept_as_written(tmp_path: Path, open_door) -> None:
    root = make_workspace(tmp_path / "ws")
    door = open_door(root)

    status, answer = door.post("/policy/track", {"paths": ["./Data//*.MD", "a" * 1025]})

    assert status == 200
    assert answer["added"] == ["./Data//*.MD"]
    assert answer["rejected"] == [{"path": "a" * 1025, "reason": LIMIT_ERROR}]


# ------------------------------------------------------------------ /status

TIERS = [("/status", None), ("/status?detail=metrics", None), ("/status?detail=full", _OPERATOR_HEADER)]


def _advertised(door: Door, query: str, headers: dict | None) -> tuple[object, object]:
    status, body = door.client.get(query, headers_override=headers)
    assert status == 200, body
    return body.get("fold_profile"), body.get("fold_table_unicode")


def test_every_status_tier_names_the_fold_profile_and_the_fold_table_unicode_version(
    tmp_path: Path, open_door
) -> None:
    probed = open_door(make_workspace(tmp_path / "probed"))
    result = probed.server.fold_profile
    expected = result.profile if result.status == "classified" else "cannot_tell"
    cannot_tell = open_door(make_workspace(tmp_path / "cannot-tell"), CANNOT_TELL)
    caseless = open_door(make_workspace(tmp_path / "caseless"), CASELESS)

    for query, headers in TIERS:
        assert _advertised(probed, query, headers) == (expected, FOLD_TABLE_UNICODE_VERSION), query
        assert _advertised(cannot_tell, query, headers) == ("cannot_tell", FOLD_TABLE_UNICODE_VERSION)
        assert _advertised(caseless, query, headers) == ("caseless", FOLD_TABLE_UNICODE_VERSION)


def test_a_start_without_the_writer_lock_advertises_cannot_tell(
    tmp_path: Path, open_door, two_writer_store
) -> None:
    door = open_door(make_workspace(tmp_path / "ws"))
    for query, headers in TIERS:
        assert _advertised(door, query, headers) == ("cannot_tell", FOLD_TABLE_UNICODE_VERSION)


@pytest.mark.parametrize("query, headers", [TIERS[0], TIERS[2]], ids=["minimal", "full"])
def test_a_degraded_status_still_names_the_fold_profile(
    query: str, headers: dict | None, tmp_path: Path, open_door, monkeypatch: pytest.MonkeyPatch
) -> None:
    door = open_door(make_workspace(tmp_path / "ws"), CASELESS)
    monkeypatch.setattr(server_module, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    with _HeldRegistryLock(door.server) as held:
        answer = _Background(door.client, query, headers=headers)
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        held.release()
    status, body = answer.result()
    assert answered, "/status did not answer while the registry lock was held"
    assert (status, body["degraded"]) == (200, True)
    assert (body["fold_profile"], body["fold_table_unicode"]) == ("caseless", FOLD_TABLE_UNICODE_VERSION)
