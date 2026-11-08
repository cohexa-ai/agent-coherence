# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A holder's own release of its write grant keeps a read (#513).

At the end of a turn (Stop), when a subagent stops (SubagentStop), or when a
session reports a failed edit, the coordinator gives back the caller's own
EXCLUSIVE or MODIFIED grant. That release used to leave the caller INVALID, and
strict mode never grants an INVALID identity again on a denied Read, so a
session was locked out of the file it had just edited. It now leaves the caller
SHARED at the version it held, so its next Read is fresh and its next Edit is
admitted, while a copy a peer's acquire or commit made stale is still refused.

The release reads the caller's state, decides and applies inside one registry
hold. The rows at the two seams below pin that: a peer's acquire and commit run
on the releasing thread either as the release's first hold is entered or as
that first hold exits, and the caller must end in a state one of the two serial
orders produces.
"""

from __future__ import annotations

import hashlib
import itertools
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable
from ccs.coordinator.service import ReleaseRequest
from ccs.core.clock import monotonic_seconds
from ccs.core.exceptions import (
    SPLIT_PAIR_REASON,
    CommitUnconfirmed,
    CoordinatorUnreachable,
    StaleView,
    ViewWedged,
    WatchdogAbandoned,
)
from ccs.core.states import MESIState, TransientState
from ccs.core.types import ConflictDetail
from tests.coordinator_helpers import recorded_version, serve_in_process
from tests.test_claude_code_coordinator_server import (
    _claimed,
    _Client,
    _handed,
    _hash,
    _operator_status,
    _read,
    _row_for,
    _sid,
    _sweep,
    _transfer,
)

_STRICT = "data/plan.json"
_WARN = "data/notes.md"
_SUBAGENT = "worker-1"
_SEED_HASH = _hash("seed")
_OWN_HASH = _hash("own-edit")
_PEER_HASH = _hash("peer-edit")
_PEER = session_to_agent_id(_sid("seam-peer"))
_OPERATOR = {"Coherence-Local-Operator": "true"}
_ROUTES = ("stop", "subagent-stop", "failed-edit")
_POINTS = ("exit", "entry")
_REVOKED = "was revoked and no new version was committed"
_UPDATED = "was updated by agent"
_OUT_OF_BAND = "on disk no longer matches"


@dataclass(frozen=True)
class _Live:
    """A coordinator serving in this process, a client for it, and every
    state-log entry its registry has written."""

    server: CoordinatorHTTPServer
    client: _Client
    logged: list[dict]


def _write_policy(root: Path) -> None:
    """``data/**`` tracked, ``data/*.json`` strict: ``_STRICT`` is a strict
    file and ``_WARN`` a warn-mode one."""
    coherence = root / ".coherence"
    coherence.mkdir(exist_ok=True)
    (coherence / "tracked.yaml").write_text("- data/**\n")
    (coherence / "strict_mode.yaml").write_text("- data/*.json\n")


def _serve(root: Path, logged: list[dict]) -> _Live:
    server = CoordinatorHTTPServer(root, port=0, instance_id="self-release", state_log=logged.append)
    server.serve_in_thread()
    return _Live(server, _Client("127.0.0.1", server.port, load_secret(root)), logged)


@pytest.fixture
def live(tmp_path: Path) -> Iterator[_Live]:
    _write_policy(tmp_path)
    served = _serve(tmp_path, [])
    try:
        yield served
    finally:
        served.server.shutdown()


@dataclass(frozen=True)
class _Holder:
    """A session, or its subagent, and the route by which it gives its write
    grant back: a Stop, a SubagentStop, or a failed edit's report."""

    sid: str
    route: str

    @property
    def fields(self) -> dict:
        """The identity every request of this holder carries."""
        if self.route == "subagent-stop":
            return {"session_id": self.sid, "agent_id": _SUBAGENT}
        return {"session_id": self.sid}

    @property
    def agent_id(self) -> uuid.UUID:
        return session_to_agent_id(self.sid, _SUBAGENT if self.route == "subagent-stop" else None)

    @property
    def bytes_hash(self) -> str:
        """The hash of the bytes the holder has: its own commit, or the file it
        first read when its release reports that the edit failed."""
        return _SEED_HASH if self.route == "failed-edit" else _OWN_HASH


def _take(live: _Live, holder: _Holder, path: str) -> None:
    """The holder reads ``path`` and takes its write grant: MODIFIED after a
    commit, or EXCLUSIVE on the failed-edit route, whose release reports the
    edit that grant was for."""
    client = live.client
    assert client.post("/hooks/pre-read", {**holder.fields, "path": path, "content_hash": _SEED_HASH})[0] == 200
    status, answer = client.post("/hooks/pre-edit", {**holder.fields, "path": path})
    assert status == 200 and answer.get("ok") is True, answer
    if holder.route != "failed-edit":
        status, answer = client.post("/hooks/post-edit", {
            **holder.fields, "path": path, "success": True, "content_hash": _OWN_HASH,
        })
        assert status == 200 and answer.get("ok") is True, answer


def _release(live: _Live, holder: _Holder, path: str) -> tuple[int, dict]:
    if holder.route == "failed-edit":
        return live.client.post("/hooks/post-edit", {**holder.fields, "path": path, "success": False})
    return live.client.post("/hooks/session-stop", holder.fields)


def _released_answer(holder: _Holder, path: str) -> dict:
    """The answer a clean release sends, byte for byte as before the change."""
    if holder.route == "failed-edit":
        return {"ok": True, "released": True}
    return {"ok": True, "released_artifacts": [path]}


def _pre_read(live: _Live, holder: _Holder, path: str) -> dict:
    status, answer = live.client.post("/hooks/pre-read", {
        **holder.fields, "path": path, "content_hash": holder.bytes_hash,
    })
    assert status == 200, answer
    return answer


def _pre_edit(live: _Live, holder: _Holder, path: str) -> dict:
    status, answer = live.client.post("/hooks/pre-edit", {**holder.fields, "path": path})
    assert status == 200, answer
    return answer


def _artifact_id(live: _Live, path: str) -> uuid.UUID:
    artifact_id = live.server.registry.lookup_artifact_id_by_name(path)
    assert artifact_id is not None, f"{path} was never registered"
    return artifact_id


def _state(live: _Live, path: str, agent_id: uuid.UUID) -> MESIState | None:
    return live.server.registry.get_agent_state(_artifact_id(live, path), agent_id)


def _entries_for(logged: list[dict], agent_id: uuid.UUID) -> list[tuple[str, str, str]]:
    return [
        (entry["from_state"], entry["to_state"], entry["trigger"])
        for entry in logged if entry["agent_id"] == str(agent_id)
    ]


def _denied_reason(answer: dict) -> str:
    output = answer.get("hookSpecificOutput", {})
    assert output.get("permissionDecision") == "deny", answer
    return output["permissionDecisionReason"]


def _strict_denials(live: _Live) -> int:
    return live.client.get("/status?detail=metrics")[1]["strict_mode_denials_total"]


def _audit_log(live: _Live) -> Path:
    return live.server.coordinator_root / ".coherence" / "audit.log"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- the holder's next turn ---------------------------------------------------


@pytest.mark.parametrize("route", _ROUTES)
def test_a_holder_reads_and_edits_its_own_file_again_after_its_release(live: _Live, route: str) -> None:
    """After a Stop, a SubagentStop or a failed edit's report, with no peer
    and the file unchanged, the holder's strict Read is fresh and its Edit is
    admitted, and nothing counts or logs a strict deny. Before, the release
    left it INVALID and both were denied on every retry."""
    holder = _Holder(_sid(f"next-turn-{route}"), route)
    _take(live, holder, _STRICT)
    version = live.server.registry.get_artifact(_artifact_id(live, _STRICT)).version

    assert _release(live, holder, _STRICT) == (200, _released_answer(holder, _STRICT))

    assert _pre_read(live, holder, _STRICT) == {"status": "fresh", "version": version}
    assert _pre_edit(live, holder, _STRICT).get("ok") is True
    assert _strict_denials(live) == 0
    assert not _audit_log(live).exists(), "a strict deny was written to the audit log"


def test_a_warn_mode_read_after_a_stop_carries_no_stale_warning(live: _Live) -> None:
    """The warn-mode twin: the session's own release no longer reads back to it
    as "your grant was revoked"."""
    holder = _Holder(_sid("warn-next-turn"), "stop")
    _take(live, holder, _WARN)

    assert _release(live, holder, _WARN) == (200, _released_answer(holder, _WARN))

    assert _pre_read(live, holder, _WARN) == {"status": "fresh", "version": 2}


def test_a_failed_edit_report_from_a_write_holder_leaves_it_a_reader(live: _Live) -> None:
    holder = _Holder(_sid("failed-edit"), "failed-edit")
    _take(live, holder, _STRICT)
    assert _state(live, _STRICT, holder.agent_id) == MESIState.EXCLUSIVE, "precondition"

    assert _release(live, holder, _STRICT) == (200, {"ok": True, "released": True})

    assert _state(live, _STRICT, holder.agent_id) == MESIState.SHARED


def test_a_subagent_stop_leaves_the_subagent_a_reader_and_the_parent_untouched(live: _Live) -> None:
    holder = _Holder(_sid("subagent-stop"), "subagent-stop")
    parent = session_to_agent_id(holder.sid)
    other = "data/spec.json"
    assert live.client.post("/hooks/pre-edit", {"session_id": holder.sid, "path": other})[1].get("ok") is True
    _take(live, holder, _STRICT)

    assert _release(live, holder, _STRICT) == (200, {"ok": True, "released_artifacts": [_STRICT]})

    assert _state(live, _STRICT, holder.agent_id) == MESIState.SHARED
    assert _state(live, _STRICT, parent) is None
    assert _state(live, other, parent) == MESIState.EXCLUSIVE


def test_a_stop_releases_each_write_grant_to_a_read_and_leaves_a_read_alone(live: _Live) -> None:
    """One stop over EXCLUSIVE on one file, MODIFIED on another and SHARED on a
    third: the first two become SHARED, each moving its file's ownership epoch
    once and logging one step, the third is not touched, and the answer is
    today's."""
    sid = _sid("three-files")
    agent = session_to_agent_id(sid)
    held, edited, read = "data/held.json", "data/edited.json", "data/read.json"
    client = live.client
    for path in (held, edited):
        assert client.post("/hooks/pre-edit", {"session_id": sid, "path": path})[1].get("ok") is True
    client.post("/hooks/post-edit", {"session_id": sid, "path": edited, "success": True, "content_hash": _OWN_HASH})
    client.post("/hooks/pre-read", {"session_id": sid, "path": read, "content_hash": _SEED_HASH})
    registry = live.server.registry
    paths = (held, edited, read)
    epochs = {path: registry.get_owner_generation(_artifact_id(live, path)) for path in paths}
    mark = len(live.logged)

    status, answer = client.post("/hooks/session-stop", {"session_id": sid})

    assert status == 200 and set(answer) == {"ok", "released_artifacts"}, answer
    assert answer["ok"] is True and sorted(answer["released_artifacts"]) == sorted([held, edited])
    assert {path: _state(live, path, agent) for path in paths} == dict.fromkeys(paths, MESIState.SHARED)
    moved = {path: registry.get_owner_generation(_artifact_id(live, path)) - epochs[path] for path in paths}
    assert moved == {held: 1, edited: 1, read: 0}
    assert sorted(_entries_for(live.logged[mark:], agent)) == [
        ("EXCLUSIVE", "SHARED", "invalidate"), ("MODIFIED", "SHARED", "invalidate"),
    ]


# --- a peer's acquire or commit still leaves the holder stale ------------------


@pytest.mark.parametrize("commits", [True, False], ids=["acquired-and-committed", "acquired-only"])
def test_a_peer_write_after_the_release_leaves_the_holder_stale(live: _Live, commits: bool) -> None:
    """The read the release left is an ordinary read: a peer's acquire, with or
    without its commit, makes it INVALID, and the holder's Read is denied with
    the text that says which happened, and its Edit is denied."""
    holder = _Holder(_sid(f"peer-after-{commits}"), "stop")
    peer = _sid(f"peer-after-peer-{commits}")
    _take(live, holder, _STRICT)
    _release(live, holder, _STRICT)

    assert live.client.post("/hooks/pre-edit", {"session_id": peer, "path": _STRICT})[1].get("ok") is True
    if commits:
        live.client.post("/hooks/post-edit", {
            "session_id": peer, "path": _STRICT, "success": True, "content_hash": _PEER_HASH,
        })

    assert _state(live, _STRICT, holder.agent_id) == MESIState.INVALID
    assert (_UPDATED if commits else _REVOKED) in _denied_reason(_pre_read(live, holder, _STRICT))
    assert _pre_edit(live, holder, _STRICT).get("ok") is False


_EDIT_ROUTES = ("failed-edit", "stop")


def _edit_after_a_peer_commit(live: _Live, holder: _Holder, path: str) -> dict:
    """The holder reads ``path`` at v1, a peer's pre-edit and commit make v2
    and leave the holder INVALID, and the holder then sends its own pre-edit.
    Returns the answer to that pre-edit."""
    client = live.client
    peer = _sid(f"peer-of-{holder.sid}")
    client.post("/hooks/pre-read", {**holder.fields, "path": path, "content_hash": _SEED_HASH})
    assert client.post("/hooks/pre-edit", {"session_id": peer, "path": path})[1].get("ok") is True
    status, answer = client.post("/hooks/post-edit", {
        "session_id": peer, "path": path, "success": True, "content_hash": _PEER_HASH,
    })
    assert status == 200 and answer.get("ok") is True, answer
    assert _state(live, path, holder.agent_id) == MESIState.INVALID, "precondition: the peer's commit"
    return _pre_edit(live, holder, path)


def _read_of_the_peer_version(live: _Live, holder: _Holder, path: str) -> dict:
    status, answer = live.client.post("/hooks/pre-read", {
        **holder.fields, "path": path, "content_hash": _PEER_HASH,
    })
    assert status == 200, answer
    return answer


@pytest.mark.parametrize("route", _EDIT_ROUTES)
def test_a_released_warn_mode_edit_made_after_a_peer_commit_leaves_a_fresh_read(live: _Live, route: str) -> None:
    """A known limitation of warn mode. A peer's commit made the holder's copy
    stale, and warn mode still admits the holder's Edit, with the
    concurrent-edit note. That Edit takes the write grant, which records the
    peer's version as the one the holder has, though the holder never read it.
    If the edit then fails, or the turn ends before its commit, the release
    leaves the holder a reader of the peer's version, so its next Read is fresh
    and carries no stale warning. Strict mode denies that Edit before any grant
    is taken, so this cannot happen there (the next test)."""
    holder = _Holder(_sid(f"warn-after-peer-{route}"), route)
    edit = _edit_after_a_peer_commit(live, holder, _WARN)
    registry = live.server.registry
    artifact_id = _artifact_id(live, _WARN)

    assert edit.get("ok") is True and edit.get("collision") is True, edit
    assert registry.get_agent_state(artifact_id, holder.agent_id) == MESIState.EXCLUSIVE
    assert registry.last_observed_version_for(artifact_id, holder.agent_id) == 2
    assert _release(live, holder, _WARN) == (200, _released_answer(holder, _WARN))

    assert _state(live, _WARN, holder.agent_id) == MESIState.SHARED
    assert _read_of_the_peer_version(live, holder, _WARN) == {"status": "fresh", "version": 2}


@pytest.mark.parametrize("route", _EDIT_ROUTES)
def test_a_strict_edit_after_a_peer_commit_is_denied_so_its_release_keeps_the_copy_stale(
    live: _Live, route: str,
) -> None:
    """The strict twin of the row above: the Edit is denied before any grant is
    taken, so the release finds the holder INVALID and leaves it so, and its
    next Read is denied with the write wording."""
    holder = _Holder(_sid(f"strict-after-peer-{route}"), route)
    edit = _edit_after_a_peer_commit(live, holder, _STRICT)
    registry = live.server.registry
    artifact_id = _artifact_id(live, _STRICT)

    assert edit.get("ok") is False, edit
    assert registry.get_agent_state(artifact_id, holder.agent_id) == MESIState.INVALID
    assert registry.last_observed_version_for(artifact_id, holder.agent_id) == 1
    _release(live, holder, _STRICT)

    assert _state(live, _STRICT, holder.agent_id) == MESIState.INVALID
    assert _UPDATED in _denied_reason(_read_of_the_peer_version(live, holder, _STRICT))


class _Seam:
    """Runs a peer's acts once, on the releasing thread, at one point of the
    release's registry holds.

    It wraps ``registry.abort_guard``, the hold every service operation takes,
    and ``service.release_to_reader``, so it knows when a release is running.
    At ``entry`` it runs the acts as the release's first hold is entered,
    before that hold's body, so a release that decided from a state read taken
    before its hold acts on a read the peer has since made stale. At ``exit``
    it runs them as the release's first outermost hold exits, so a release that
    reads and decides in one hold and applies in a second lets the acts in
    between the two. A release that reads, decides and applies inside one hold
    sees the acts wholly before it or wholly after it at either point."""

    def __init__(self, live: _Live, point: str) -> None:
        self._real_guard = live.server.registry.abort_guard
        self._real_release = live.server.service.release_to_reader
        self._point = point
        self._local = threading.local()
        self._acts: Callable[[], None] | None = None
        self.fired = False

    def arm(self, acts: Callable[[], None]) -> None:
        """Run ``acts`` at the seam, once."""
        self._acts = acts

    @contextmanager
    def guard(self, abort: threading.Event | None = None, **kwargs: object) -> Iterator[None]:
        depth = getattr(self._local, "depth", 0)
        self._local.depth = depth + 1
        try:
            with self._real_guard(abort, **kwargs):
                if self._point == "entry" and self._releasing():
                    self._fire()
                yield
        finally:
            self._local.depth = depth
        if depth == 0 and self._point == "exit" and self._releasing():
            self._fire()

    def release(self, *args: object, **kwargs: object) -> object:
        self._local.releasing = True
        try:
            return self._real_release(*args, **kwargs)
        finally:
            self._local.releasing = False

    def _releasing(self) -> bool:
        return getattr(self._local, "releasing", False)

    def _fire(self) -> None:
        """Run the acts the first time only: the peer's own holds, taken while
        the release is still running, do not fire them again."""
        self._local.releasing = False
        if self._acts is None or self.fired:
            return
        self.fired = True
        self._acts()


def _seam_at(live: _Live, point: str, monkeypatch: pytest.MonkeyPatch) -> _Seam:
    seam = _Seam(live, point)
    monkeypatch.setattr(live.server.registry, "abort_guard", seam.guard)
    monkeypatch.setattr(live.server.service, "release_to_reader", seam.release)
    return seam


def _peer_acts(live: _Live, artifact_id: uuid.UUID, commits: bool) -> Callable[[], None]:
    """The peer's pre-edit and, when ``commits``, its commit, as the service
    runs them for the hooks."""
    service = live.server.service

    def act() -> None:
        tick = monotonic_seconds()
        service.write(agent_id=_PEER, artifact_id=artifact_id, issued_at_tick=tick)
        if commits:
            service.commit(
                agent_id=_PEER, artifact_id=artifact_id, content="",
                content_hash=_PEER_HASH, issued_at_tick=tick,
            )

    return act


_SEAM_CASES = list(itertools.product(_ROUTES, _POINTS))
_SEAM_IDS = [f"{route}-at-{point}" for route, point in _SEAM_CASES]


@pytest.mark.parametrize("case", _SEAM_CASES, ids=_SEAM_IDS)
def test_a_peer_write_and_commit_at_the_release_leave_the_holder_stale(
    live: _Live, monkeypatch: pytest.MonkeyPatch, case: tuple[str, str],
) -> None:
    """A peer's pre-edit and commit land as the release runs. Either serial
    order leaves the holder INVALID at the peer's version: released and then
    invalidated by the peer, or invalidated by the peer and then left as it
    was by the release. Its Read and Edit are strict-denied. A release that
    let them in between two holds, or acted on a state read before its hold,
    left it SHARED at a version it never held, and its stale copy was
    admitted."""
    route, point = case
    holder = _Holder(_sid(f"race-{route}-{point}"), route)
    _take(live, holder, _STRICT)
    artifact_id = _artifact_id(live, _STRICT)
    version = live.server.registry.get_artifact(artifact_id).version
    seam = _seam_at(live, point, monkeypatch)
    seam.arm(_peer_acts(live, artifact_id, True))

    assert _release(live, holder, _STRICT) == (200, _released_answer(holder, _STRICT))

    assert seam.fired, "the peer's acts never ran"
    assert live.server.registry.get_artifact(artifact_id).version == version + 1, "control: the peer committed"
    assert _state(live, _STRICT, holder.agent_id) == MESIState.INVALID
    _denied_reason(_pre_read(live, holder, _STRICT))
    assert _pre_edit(live, holder, _STRICT).get("ok") is False


@pytest.mark.parametrize("case", _SEAM_CASES, ids=_SEAM_IDS)
def test_a_peer_acquire_at_the_release_leaves_the_holder_stale_beside_the_peer(
    live: _Live, monkeypatch: pytest.MonkeyPatch, case: tuple[str, str],
) -> None:
    """A peer's pre-edit alone at the seam: the holder ends INVALID next to the
    peer's EXCLUSIVE and its Edit is denied, never SHARED beside a writer."""
    route, point = case
    holder = _Holder(_sid(f"acquire-{route}-{point}"), route)
    _take(live, holder, _STRICT)
    artifact_id = _artifact_id(live, _STRICT)
    seam = _seam_at(live, point, monkeypatch)
    seam.arm(_peer_acts(live, artifact_id, False))

    _release(live, holder, _STRICT)

    assert seam.fired, "the peer's acts never ran"
    assert _state(live, _STRICT, holder.agent_id) == MESIState.INVALID
    assert _state(live, _STRICT, _PEER) == MESIState.EXCLUSIVE
    assert _pre_edit(live, holder, _STRICT).get("ok") is False


@pytest.mark.parametrize("case", _SEAM_CASES, ids=_SEAM_IDS)
def test_a_release_with_nothing_at_the_seam_is_one_step_to_a_read(
    live: _Live, monkeypatch: pytest.MonkeyPatch, case: tuple[str, str],
) -> None:
    """The control for the two rows above, with the seam in place and no peer:
    one state-log step from the write claim to SHARED, the version the holder
    held recorded as observed, the ownership epoch moved once, and no hold tick
    left behind."""
    route, point = case
    holder = _Holder(_sid(f"control-{route}-{point}"), route)
    _take(live, holder, _STRICT)
    registry = live.server.registry
    artifact_id = _artifact_id(live, _STRICT)
    claim = registry.get_agent_state(artifact_id, holder.agent_id)
    epoch = registry.get_owner_generation(artifact_id)
    seam = _seam_at(live, point, monkeypatch)
    seam.arm(lambda: None)
    mark = len(live.logged)

    _release(live, holder, _STRICT)

    assert seam.fired, "the seam never ran"
    assert registry.get_agent_state(artifact_id, holder.agent_id) == MESIState.SHARED
    version = registry.get_artifact(artifact_id).version
    assert registry.last_observed_version_for(artifact_id, holder.agent_id) == version
    assert registry.get_owner_generation(artifact_id) == epoch + 1
    assert registry.granted_at_tick(holder.agent_id, artifact_id) is None
    assert _entries_for(live.logged[mark:], holder.agent_id) == [(claim.name, "SHARED", "invalidate")]


# --- a release from an identity with no write grant changes nothing ------------


@pytest.mark.parametrize(
    ("route", "transient"),
    [("failed-edit", TransientState.IED), ("stop", TransientState.MWB)],
    ids=["exclusive-with-a-pending-write", "modified-with-a-pending-commit"],
)
def test_a_release_clears_the_marker_a_write_holder_was_left_with(
    live: _Live, route: str, transient: TransientState,
) -> None:
    """A write or a commit that raised after setting its in-progress marker
    leaves the holder EXCLUSIVE or MODIFIED with that marker still set. The
    release clears it as it leaves the holder a reader, so no SHARED row keeps
    a marker for a write that is over."""
    holder = _Holder(_sid(f"marker-{route}"), route)
    _take(live, holder, _STRICT)
    registry = live.server.registry
    artifact_id = _artifact_id(live, _STRICT)
    registry.set_agent_transient(artifact_id, holder.agent_id, transient, entered_tick=monotonic_seconds())
    assert registry.get_agent_transient(artifact_id, holder.agent_id) == transient, "precondition"

    assert _release(live, holder, _STRICT) == (200, _released_answer(holder, _STRICT))

    assert registry.get_agent_state(artifact_id, holder.agent_id) == MESIState.SHARED
    assert registry.get_agent_transient(artifact_id, holder.agent_id) is None


def test_a_failed_edit_report_from_a_reader_keeps_its_read(live: _Live) -> None:
    sid = _sid("reader-reports")
    agent = session_to_agent_id(sid)
    live.client.post("/hooks/pre-read", {"session_id": sid, "path": _STRICT, "content_hash": _SEED_HASH})

    answer = live.client.post("/hooks/post-edit", {"session_id": sid, "path": _STRICT, "success": False})

    assert answer == (200, {"ok": True, "released": True})
    assert _state(live, _STRICT, agent) == MESIState.SHARED


def test_a_failed_edit_report_from_an_identity_with_no_row_creates_none(live: _Live) -> None:
    live.client.post("/hooks/pre-read", {
        "session_id": _sid("registers"), "path": _STRICT, "content_hash": _SEED_HASH,
    })
    stranger = _sid("never-touched")

    answer = live.client.post("/hooks/post-edit", {"session_id": stranger, "path": _STRICT, "success": False})

    assert answer == (200, {"ok": True, "released": True})
    assert _state(live, _STRICT, session_to_agent_id(stranger)) is None


def test_a_failed_edit_report_from_a_stale_reader_keeps_it_stale_and_clears_its_transient(live: _Live) -> None:
    """A peer's pre-edit left the reader INVALID with its acknowledgement
    pending. The reader's failed-edit report changes no state and clears that
    pending acknowledgement, as the release did before."""
    sid = _sid("stale-reader-reports")
    agent = session_to_agent_id(sid)
    live.client.post("/hooks/pre-read", {"session_id": sid, "path": _STRICT, "content_hash": _SEED_HASH})
    live.client.post("/hooks/pre-edit", {"session_id": _sid("stale-reader-peer"), "path": _STRICT})
    artifact_id = _artifact_id(live, _STRICT)
    registry = live.server.registry
    assert registry.get_agent_transient(artifact_id, agent) == TransientState.SIA, "precondition"

    live.client.post("/hooks/post-edit", {"session_id": sid, "path": _STRICT, "success": False})

    assert registry.get_agent_state(artifact_id, agent) == MESIState.INVALID
    assert registry.get_agent_transient(artifact_id, agent) is None


# --- what the release still guards ---------------------------------------------


def test_a_release_whose_request_already_timed_out_changes_nothing(live: _Live) -> None:
    """With the request's abort set before the hold is won, the release raises
    and changes nothing: the holder keeps its write claim, the epoch stays, and
    nothing is logged. A later stop releases the file to a read."""
    holder = _Holder(_sid("timed-out"), "stop")
    _take(live, holder, _STRICT)
    registry = live.server.registry
    artifact_id = _artifact_id(live, _STRICT)
    epoch = registry.get_owner_generation(artifact_id)
    mark = len(live.logged)
    abort = threading.Event()
    abort.set()

    with pytest.raises(WatchdogAbandoned):
        live.server.service.release_to_reader(
            ReleaseRequest(agent_id=holder.agent_id, artifact_id=artifact_id, issued_at_tick=monotonic_seconds()),
            abort=abort,
        )

    assert registry.get_agent_state(artifact_id, holder.agent_id) == MESIState.MODIFIED
    assert registry.get_owner_generation(artifact_id) == epoch
    assert live.logged[mark:] == []
    _release(live, holder, _STRICT)
    assert registry.get_agent_state(artifact_id, holder.agent_id) == MESIState.SHARED


def test_a_store_error_partway_through_a_stop_answers_internal_and_leaves_the_rest_held(
    live: _Live, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stop's loop catches only coherence errors, so a store error from the
    second of two releases ends its work: the answer is the generic internal
    envelope, which names neither file, the file released before the raise
    stays SHARED, because each release commits on its own, and the file the
    raise hit keeps its write grant. A later stop, with the store answering
    again, releases that one too."""
    sid = _sid("store-error")
    agent = session_to_agent_id(sid)
    paths = ("data/first.json", "data/second.json")
    for path in paths:
        assert live.client.post("/hooks/pre-edit", {"session_id": sid, "path": path})[1].get("ok") is True
    names = {_artifact_id(live, path): path for path in paths}
    real_release = live.server.service.release_to_reader
    reached: list[str] = []

    # The held rows come back in no fixed order, so the raise follows the call
    # count, not a path.
    def fail_after_the_first(request: ReleaseRequest, **kwargs: object) -> bool:
        reached.append(names[request.artifact_id])
        if len(reached) > 1:
            raise sqlite3.OperationalError("simulated: database is locked")
        return real_release(request, **kwargs)

    monkeypatch.setattr(live.server.service, "release_to_reader", fail_after_the_first)
    mark = len(live.logged)

    stop = live.client.post("/hooks/session-stop", {"session_id": sid})

    assert stop == (200, {"ok": False, "reason": "internal: OperationalError"})
    released, held = reached
    assert _state(live, released, agent) == MESIState.SHARED
    assert _state(live, held, agent) == MESIState.EXCLUSIVE
    assert _entries_for(live.logged[mark:], agent) == [("EXCLUSIVE", "SHARED", "invalidate")]
    monkeypatch.undo()
    assert live.client.post("/hooks/session-stop", {"session_id": sid}) == (
        200, {"ok": True, "released_artifacts": [held]},
    )
    assert _state(live, held, agent) == MESIState.SHARED


def test_a_write_claim_captured_before_the_release_cannot_commit_after_it(live: _Live) -> None:
    """The release still moves the ownership epoch: a commit on the claim the
    holder captured at its acquire is refused ``stale_read_generation``, and a
    fresh Read does not lift that."""
    holder = _Holder(_sid("fenced"), "stop")
    _take(live, holder, _STRICT)
    registry = live.server.registry
    artifact_id = _artifact_id(live, _STRICT)
    version = registry.get_artifact(artifact_id).version
    _release(live, holder, _STRICT)

    def late_commit() -> object:
        return registry.commit_cas(artifact_id, holder.agent_id, expected_version=version, content_hash=_hash("late"))

    refused = late_commit()
    assert isinstance(refused, ConflictDetail) and refused.reason == "stale_read_generation", refused
    assert _pre_read(live, holder, _STRICT)["status"] == "fresh"
    refused = late_commit()
    assert isinstance(refused, ConflictDetail) and refused.reason == "stale_read_generation", refused
    assert registry.get_artifact(artifact_id).version == version


def test_a_commit_sent_after_the_callers_own_failed_edit_is_refused_and_names_shared(live: _Live) -> None:
    """Two edits of one file in a turn, and the failing one reports first: its
    release leaves the caller a reader, so the other edit's commit is refused
    as before, and the refusal's reason, which names the caller's state, now
    says SHARED where it said INVALID."""
    sid = _sid("late-commit")
    client = live.client
    for _ in range(2):
        assert client.post("/hooks/pre-edit", {"session_id": sid, "path": _WARN})[1].get("ok") is True
    assert client.post("/hooks/post-edit", {"session_id": sid, "path": _WARN, "success": False}) == (
        200, {"ok": True, "released": True},
    )
    version = live.server.registry.get_artifact(_artifact_id(live, _WARN)).version

    status, answer = client.post("/hooks/post-edit", {
        "session_id": sid, "path": _WARN, "success": True, "content_hash": _OWN_HASH,
    })

    assert status == 200 and answer.get("ok") is False, answer
    assert answer["reason"].startswith("commit_not_allowed "), answer
    assert answer["reason"].endswith(" state=MESIState.SHARED"), answer
    assert live.server.registry.get_artifact(_artifact_id(live, _WARN)).version == version


def test_a_grant_the_sweep_reclaimed_before_the_stop_stays_reclaimed(live: _Live) -> None:
    holder = _Holder(_sid("reclaimed"), "stop")
    _take(live, holder, _STRICT)
    reclaim_tick = int(time.time()) + 999_999
    assert _sweep(live.server, reclaim_tick) == 1

    status, answer = _release(live, holder, _STRICT)

    assert status == 200 and answer["ok"] is True and answer["released_artifacts"] == [], answer
    assert _state(live, _STRICT, holder.agent_id) == MESIState.INVALID
    row = _row_for(_operator_status(live.client), holder.sid)
    assert row["states"] == {}
    assert row["reclaimed"] == {_STRICT: {"trigger": "reclaim_heartbeat", "reclaimed_at_unix_ts": reclaim_tick}}


@pytest.mark.parametrize(
    ("route", "transient"),
    [("stop", None), ("failed-edit", None), ("stop", TransientState.MWB), ("failed-edit", TransientState.IED)],
    ids=["stop", "failed-edit", "stop-over-a-pending-commit", "failed-edit-over-a-pending-write"],
)
def test_the_sweeps_leave_the_read_a_holders_own_release_left(
    live: _Live, route: str, transient: TransientState | None,
) -> None:
    """Far past every timeout, neither sweep touches the read the release left:
    it is no write grant for the stable-grant sweep to reclaim, and the release
    cleared any in-progress marker the write left, which the transient sweep
    would otherwise expire to INVALID. The row stays SHARED, the epoch stays
    where the release moved it, and nothing is logged for the holder. A peer's
    write grant on another file, reclaimed by the same pass, shows the pass
    ran past the timeout."""
    holder = _Holder(_sid(f"swept-{route}-{transient}"), route)
    _take(live, holder, _STRICT)
    registry = live.server.registry
    artifact_id = _artifact_id(live, _STRICT)
    if transient is not None:
        registry.set_agent_transient(artifact_id, holder.agent_id, transient, entered_tick=monotonic_seconds())
    _release(live, holder, _STRICT)
    peer = _sid(f"swept-peer-{route}-{transient}")
    assert live.client.post("/hooks/pre-edit", {"session_id": peer, "path": "data/held.json"})[1].get("ok") is True
    epoch = registry.get_owner_generation(artifact_id)
    mark = len(live.logged)
    far_past = monotonic_seconds() + 999_999

    assert live.server.service.enforce_transient_timeouts(current_tick=far_past, timeout_ticks=60) == 0
    assert _sweep(live.server, far_past) == 1, "control: the pass reclaims the peer's write grant"

    assert registry.get_agent_state(artifact_id, holder.agent_id) == MESIState.SHARED
    assert registry.get_owner_generation(artifact_id) == epoch
    assert _entries_for(live.logged[mark:], holder.agent_id) == []


def test_the_operator_drain_still_leaves_an_orphaned_write_grant_invalid(live: _Live) -> None:
    """``--prepare-for-migration`` releases write grants whose bytes on disk
    may never have been committed, so it keeps ending them INVALID."""
    holder = _Holder(_sid("drained"), "failed-edit")
    _take(live, holder, _STRICT)
    mark = len(live.logged)

    status, answer = live.client.post("/admin/prepare-for-migration", {}, headers_override=_OPERATOR)
    assert status == 200 and answer["draining"] is True, answer

    deadline = time.monotonic() + 5.0
    while not _entries_for(live.logged[mark:], holder.agent_id) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert _entries_for(live.logged[mark:], holder.agent_id) == [("EXCLUSIVE", "INVALID", "invalidate")]


# --- a later turn ----------------------------------------------------------------


def test_a_session_hands_a_file_on_in_a_later_turn(live: _Live) -> None:
    """A stop leaves the giver a reader, so a transfer in its next turn is
    admitted from SHARED instead of refused as not held."""
    giver, successor = _claimed(live.client), _claimed(live.client)
    _read(live.client, successor, "data/other.json")  # registers the successor
    client = live.client
    for route, body in (
        ("/hooks/pre-edit", {"path": _STRICT}),
        ("/hooks/post-edit", {"path": _STRICT, "success": True, "content_hash": _OWN_HASH}),
        ("/hooks/session-stop", {}),
    ):
        status, answer = client.post(route, {"session_id": giver.sid, **body}, principal=giver.principal)
        assert status == 200 and answer.get("ok") is True, answer

    answer = _transfer(client, giver, successor.agent, [{"path": _STRICT}])

    assert answer == (200, {"ok": True, "grants": [_handed(giver, successor, _STRICT, version=2)]})


def test_a_released_read_survives_a_coordinator_restart(tmp_path: Path) -> None:
    _write_policy(tmp_path)
    holder = _Holder(_sid("restart"), "stop")
    first = _serve(tmp_path, [])
    try:
        _take(first, holder, _STRICT)
        _release(first, holder, _STRICT)
    finally:
        first.server.shutdown()

    second = _serve(tmp_path, [])
    try:
        assert _pre_read(second, holder, _STRICT) == {"status": "fresh", "version": 2}
    finally:
        second.server.shutdown()


def test_a_change_made_outside_the_coordinator_after_a_stop_keeps_the_out_of_band_deny(live: _Live) -> None:
    """The file changes on disk between turns with nothing committed. The
    strict Read is denied with the out-of-band text, not the revoked one, and
    the Edit, the way out, is admitted with its warning."""
    target = live.server.coordinator_root / _STRICT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"first")
    sid = _sid("out-of-band")
    client = live.client
    client.post("/hooks/pre-read", {"session_id": sid, "path": _STRICT, "content_hash": _sha(b"first")})
    assert client.post("/hooks/pre-edit", {"session_id": sid, "path": _STRICT})[1].get("ok") is True
    client.post("/hooks/session-stop", {"session_id": sid})

    target.write_bytes(b"changed by the user")
    status, read = client.post("/hooks/pre-read", {
        "session_id": sid, "path": _STRICT, "content_hash": _sha(b"changed by the user"),
    })

    assert status == 200
    reason = _denied_reason(read)
    assert _OUT_OF_BAND in reason and _REVOKED not in reason, reason
    status, edit = client.post("/hooks/pre-edit", {"session_id": sid, "path": _STRICT})
    assert status == 200 and edit.get("ok") is True and edit.get("disk_differs") is True, edit


# --- CoherentVolume, whose failure releases take the same route ------------------


_VOLUME_REL = "data/shared.txt"
_FAST = LifecycleConfig(
    idle_shutdown_sec=0,
    sweep_interval_sec=0.1,
    notice_evict_max_age_sec=1.0,
    port_file_retry_attempts=20,
    port_file_retry_interval_sec=0.05,
    connect_retry_attempts=10,
    connect_retry_interval_sec=0.05,
)


def _volume_on(
    tmp_path: Path, *, on_stale_read: Literal["allow", "raise"] = "allow",
) -> tuple[CoordinatorHTTPServer, CoherentVolume, Path]:
    target = tmp_path / _VOLUME_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"v1")
    server = serve_in_process(tmp_path, "self-release-volume")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=_FAST, on_stale_read=on_stale_read)
    vol.read(_VOLUME_REL)
    return server, vol, target


class _Wire:
    """Every request the volume sends from here on, with its answer, except the
    next commit, whose answer is lost: after the coordinator ran it when
    ``delivered``, or before the request left, so the coordinator never saw it,
    when not. The release the volume sends next goes through."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, delivered: bool) -> None:
        self._real_post = coherent_volume_module._coordinator_post
        self._delivered = delivered
        self._commit_lost = False
        self._sent: list[tuple[str, dict, dict]] = []
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", self._post)

    def sent_to(self, route: str) -> list[tuple[dict, dict]]:
        """Each request sent to ``route``, with its answer, in order."""
        return [(payload, answer) for sent, payload, answer in self._sent if sent == route]

    def answers(self, route: str) -> list[dict]:
        return [answer for _, answer in self.sent_to(route)]

    def _post(self, endpoint: object, route: str, payload: dict, **kwargs: object) -> dict:
        if not self._commit_lost and route == "/hooks/post-edit" and payload.get("success") is True:
            self._commit_lost = True
            if not self._delivered:
                raise CoordinatorUnavailable("simulated: the commit never left", never_sent=True)
            self._real_post(endpoint, route, payload, **kwargs)
            raise CoordinatorUnavailable("simulated: the commit's answer was lost")
        answer = self._real_post(endpoint, route, payload, **kwargs)
        self._sent.append((route, payload, answer))
        return answer


def _appending(seen: list[bytes]) -> Callable[[bytes], bytes]:
    """A ``make_content`` that records the bytes each attempt derives from."""

    def make_content(current: bytes) -> bytes:
        seen.append(current)
        return current + b"+cas"

    return make_content


def test_a_volume_writes_again_after_a_failed_write_without_reacquiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, vol, target = _volume_on(tmp_path)
    try:
        incarnation = vol.incarnation
        real_write = vol._atomic_write
        failures = [OSError("simulated: the disk write failed")]

        def fail_once(abs_path: Path, data: bytes) -> None:
            if failures:
                raise failures.pop()
            real_write(abs_path, data)

        monkeypatch.setattr(vol, "_atomic_write", fail_once)
        with pytest.raises(OSError):
            vol.write(_VOLUME_REL, b"never lands")

        vol.write(_VOLUME_REL, b"v2")

        assert target.read_bytes() == b"v2"
        assert recorded_version(server, _VOLUME_REL) == 2
        assert vol.incarnation == incarnation, "the volume re-minted"
    finally:
        server.shutdown()


def test_a_volume_writes_again_after_an_unconfirmed_commit_whose_release_was_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, vol, target = _volume_on(tmp_path)
    try:
        incarnation = vol.incarnation
        real_post = coherent_volume_module._coordinator_post
        unanswered = [True]

        def lose_one_commit_answer(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            answer = real_post(endpoint, path, payload, **kwargs)
            if unanswered and path == "/hooks/post-edit" and payload.get("success") is True:
                unanswered.clear()
                raise CoordinatorUnavailable("simulated: the commit's answer was lost")
            return answer

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", lose_one_commit_answer)
        with pytest.raises(CommitUnconfirmed):
            vol.write(_VOLUME_REL, b"v2")

        vol.write(_VOLUME_REL, b"v3")

        assert target.read_bytes() == b"v3"
        assert recorded_version(server, _VOLUME_REL) == 3
        assert vol.incarnation == incarnation, "the volume re-minted"
    finally:
        server.shutdown()


def test_a_volume_cas_after_an_unconfirmed_commit_whose_release_was_answered_is_refused_once_then_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write()'s commit lands but its answer is lost, and the release it sends
    next is answered, so the volume records no grant on the file and write_cas
    starts without a re-mint, from the read the release left. That read's
    claim predates the release, which moved the ownership epoch, so the first
    commit is refused ``stale_read_generation``. The retry re-mints, and the
    coordinator checks the new identity's read against the content it records,
    so the commit that lands derives from the bytes recorded at its version."""
    server, vol, target = _volume_on(tmp_path)
    try:
        wire = _Wire(monkeypatch, delivered=True)
        with pytest.raises(CommitUnconfirmed):
            vol.write(_VOLUME_REL, b"v2")
        assert recorded_version(server, _VOLUME_REL) == 2, "control: the commit landed"
        assert wire.answers("/hooks/post-edit") == [{"ok": True, "released": True}], "control: the release"
        released = vol.incarnation
        seen: list[bytes] = []

        result = vol.write_cas(_VOLUME_REL, _appending(seen))

        [(first, refused), (retry, won)] = wire.sent_to("/hooks/post-edit-cas")
        assert first["agent_id"] == released and refused["reason"] == "stale_read_generation", refused
        assert retry["agent_id"] != released and won["ok"] is True, won
        [released_read, checked_read] = wire.answers("/hooks/pre-read")
        assert released_read["status"] == "fresh", released_read
        assert checked_read["status"] == "stale" and checked_read["summary"]["hash_differs"] is False, checked_read
        assert seen == [b"v2", b"v2"]
        assert (result.version, target.read_bytes()) == (3, b"v2+cas")
        assert recorded_version(server, _VOLUME_REL) == 3
    finally:
        server.shutdown()


def test_a_volume_cas_over_bytes_changed_after_an_unconfirmed_commit_never_lands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The row above with the file changed on disk after the commit whose answer
    was lost. Inside the coordinator's commit-lag window the read the release
    left is admitted fresh over bytes the coordinator does not record, and
    make_content derives from them; only the moved epoch refuses that commit.
    Every read after the re-mint is checked against the recorded content and
    denied, so the call ends ``ViewWedged`` and nothing lands."""
    server, vol, target = _volume_on(tmp_path)
    try:
        wire = _Wire(monkeypatch, delivered=True)
        with pytest.raises(CommitUnconfirmed):
            vol.write(_VOLUME_REL, b"v2")
        target.write_bytes(b"changed outside")
        seen: list[bytes] = []

        with pytest.raises(ViewWedged):
            vol.write_cas(_VOLUME_REL, _appending(seen))

        [released_read, *checked_reads] = wire.answers("/hooks/pre-read")
        assert released_read["status"] == "fresh" and released_read["hash_differs"] is True, released_read
        assert seen == [b"changed outside"]
        [refused] = wire.answers("/hooks/post-edit-cas")
        assert refused["reason"] == "stale_read_generation", refused
        assert checked_reads, "the retry never read"
        for answer in checked_reads:
            _denied_reason(answer)
        assert (target.read_bytes(), recorded_version(server, _VOLUME_REL)) == (b"changed outside", 2)
    finally:
        server.shutdown()


#: What a write()'s error says to do when its commit may not be recorded.
_WRITE_AGAIN = "write() the path again"
#: The cause a split-pair refusal names for this volume's own unrecorded write.
_UNRECORDED_CAUSE = "a write() whose commit did not reach the coordinator"


def test_a_volume_write_whose_commit_never_left_is_refused_on_read_until_its_next_write_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """write()'s bytes reach disk but its commit never leaves, and the release it
    sends next is answered: the coordinator still records v1, and the volume
    reads it with v2's bytes on disk. Under ``on_stale_read="raise"`` each read
    is denied by the coordinator as a change made outside it, and the volume
    raises its own split-pair refusal in place of that text; reading again
    does not clear it. The next write is admitted with the coordinator's note
    that the disk differs, which the volume does not pass on, and its commit
    records the bytes, after which a read is fresh.

    The error and the refusal say so: the bytes are unrecorded, reading again
    does not clear the refusal, and writing the path again records them."""
    server, vol, target = _volume_on(tmp_path, on_stale_read="raise")
    try:
        wire = _Wire(monkeypatch, delivered=False)
        with pytest.raises(CoordinatorUnreachable) as unrecorded:
            vol.write(_VOLUME_REL, b"v2")
        message = str(unrecorded.value)
        assert "did not record the write" in message and _WRITE_AGAIN in message, message
        assert "Re-read" not in message and "release() sends it again" not in message, message
        assert (target.read_bytes(), recorded_version(server, _VOLUME_REL)) == (b"v2", 1)
        assert wire.answers("/hooks/post-edit") == [{"ok": True, "released": True}], "control: the release"
        artifact_id = server.registry.lookup_artifact_id_by_name(_VOLUME_REL)
        assert server.registry.get_agent_state(artifact_id, uuid.UUID(vol.agent_id)) == MESIState.SHARED

        for _ in range(2):
            with pytest.raises(StaleView) as refused:
                vol.read(_VOLUME_REL)
            assert refused.value.reason == SPLIT_PAIR_REASON, refused.value
            assert _UNRECORDED_CAUSE in str(refused.value), refused.value
        denials = [_denied_reason(answer) for answer in wire.answers("/hooks/pre-read")]
        assert len(denials) == 2 and all(_OUT_OF_BAND in denial for denial in denials), denials

        vol.write(_VOLUME_REL, b"v3")

        assert wire.answers("/hooks/pre-edit")[-1].get("disk_differs") is True
        assert (target.read_bytes(), recorded_version(server, _VOLUME_REL)) == (b"v3", 2)
        assert vol.read(_VOLUME_REL) == b"v3"
    finally:
        server.shutdown()


def test_writing_the_same_bytes_again_records_a_write_whose_commit_never_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error's advice works with the bytes already on disk: writing them
    again commits them, and the refusal clears.

    Prevents advice that cannot recover the state it describes."""
    server, vol, target = _volume_on(tmp_path, on_stale_read="raise")
    try:
        _Wire(monkeypatch, delivered=False)
        with pytest.raises(CoordinatorUnreachable):
            vol.write(_VOLUME_REL, b"v2")
        with pytest.raises(StaleView):
            vol.reacquire(_VOLUME_REL)
            vol.read(_VOLUME_REL)

        vol.write(_VOLUME_REL, target.read_bytes())

        assert (target.read_bytes(), recorded_version(server, _VOLUME_REL)) == (b"v2", 2)
        assert vol.read(_VOLUME_REL) == b"v2"
    finally:
        server.shutdown()


def test_a_write_whose_commit_landed_but_was_not_answered_recovers_the_same_way(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the commit landed and only its answer was lost, the error says the
    coordinator may have recorded the write; a read is fresh, and following
    the same advice commits the same bytes at the next version.

    Prevents advice that holds only when the commit did not land."""
    server, vol, target = _volume_on(tmp_path, on_stale_read="raise")
    try:
        _Wire(monkeypatch, delivered=True)
        with pytest.raises(CommitUnconfirmed) as maybe:
            vol.write(_VOLUME_REL, b"v2")
        message = str(maybe.value)
        assert "may have recorded the write" in message and _WRITE_AGAIN in message, message
        assert vol.read(_VOLUME_REL) == b"v2"

        vol.write(_VOLUME_REL, b"v2")

        assert recorded_version(server, _VOLUME_REL) == 3
    finally:
        server.shutdown()


class _ReleaseLostToo(_Wire):
    """As :class:`_Wire`, and the release the volume sends after the lost commit
    gets no answer either."""

    _release_lost = False

    def _post(self, endpoint: object, route: str, payload: dict, **kwargs: object) -> dict:
        lost_release = self._commit_lost and not self._release_lost
        if lost_release and route == "/hooks/post-edit" and payload.get("success") is False:
            self._release_lost = True  # one lost release; later ones go through
            raise CoordinatorUnavailable("simulated: the release's answer was lost")
        return super()._post(endpoint, route, payload, **kwargs)


def test_a_write_whose_release_was_not_confirmed_names_release_to_send_it_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error says when the release of the write's grant was not confirmed,
    and names ``release(path)`` as the way to send it again, which frees the
    file for a refuse-mode peer.

    Prevents a grant left standing with no step named to release it."""
    server, vol, target = _volume_on(tmp_path)
    try:
        _ReleaseLostToo(monkeypatch, delivered=False)
        with pytest.raises(CoordinatorUnreachable) as unrecorded:
            vol.write(_VOLUME_REL, b"v2")
        assert "release() sends it again" in str(unrecorded.value), unrecorded.value

        vol.release(_VOLUME_REL)

        peer = CoherentVolume(tmp_path, managed=("data/**",), config=_FAST, on_held="refuse")
        peer.write(_VOLUME_REL, b"from the peer")
        assert target.read_bytes() == b"from the peer"
    finally:
        server.shutdown()
