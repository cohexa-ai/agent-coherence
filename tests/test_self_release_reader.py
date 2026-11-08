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

The release reads the caller's state and decides inside one registry hold. The
rows at the two seams below pin that: a peer's acquire and commit run on the
releasing thread either as the release's first hold is entered or once the
hold that ended the caller's write claim has exited, and the caller must end
in a state one of the two serial orders produces.
"""

from __future__ import annotations

import hashlib
import itertools
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable
from ccs.coordinator.service import ReleaseRequest
from ccs.core.clock import monotonic_seconds
from ccs.core.exceptions import CommitUnconfirmed, WatchdogAbandoned
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
_WRITE_CLAIM = (MESIState.EXCLUSIVE, MESIState.MODIFIED)
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


class _Seam:
    """Runs a peer's acts once, on the releasing thread, at one point of the
    release's registry holds.

    It wraps ``registry.abort_guard``, the hold every service operation takes.
    At ``exit`` it runs the acts once the outermost hold has exited with the
    watched identity no longer holding a write claim: before the release that
    identity is a writer, so the first such exit is the end of the release's
    hold, and a release split across two holds lets the acts in between them.
    At ``entry`` it runs them as the release operation's first hold is entered,
    before that hold's body, so a release that decided from a state read taken
    before its hold acts on a read the peer has since made stale. A release
    that reads and decides inside one hold sees the acts wholly before it or
    wholly after it at either point."""

    def __init__(self, live: _Live, point: str) -> None:
        self._registry = live.server.registry
        self._real_guard = live.server.registry.abort_guard
        self._real_release = live.server.service.release_to_reader
        self._point = point
        self._local = threading.local()
        self._watched: tuple[uuid.UUID, uuid.UUID] | None = None
        self._acts: Callable[[], None] | None = None
        self.fired = False

    def arm(self, watched: tuple[uuid.UUID, uuid.UUID], acts: Callable[[], None]) -> None:
        """Watch ``(artifact_id, agent_id)`` and run ``acts`` at the seam."""
        self._watched, self._acts = watched, acts

    @contextmanager
    def guard(self, abort: threading.Event | None = None, **kwargs: object) -> Iterator[None]:
        depth = getattr(self._local, "depth", 0)
        self._local.depth = depth + 1
        try:
            with self._real_guard(abort, **kwargs):
                if self._point == "entry" and getattr(self._local, "releasing", False):
                    self._local.releasing = False
                    self._fire()
                yield
        finally:
            self._local.depth = depth
        if depth == 0 and self._point == "exit" and not self._watched_holds_a_write_claim():
            self._fire()

    def release(self, *args: object, **kwargs: object) -> object:
        self._local.releasing = True
        try:
            return self._real_release(*args, **kwargs)
        finally:
            self._local.releasing = False

    def _fire(self) -> None:
        if self._acts is None or self.fired:
            return
        self.fired = True
        self._acts()

    def _watched_holds_a_write_claim(self) -> bool:
        return self._watched is None or self._registry.get_agent_state(*self._watched) in _WRITE_CLAIM


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
    seam.arm((artifact_id, holder.agent_id), _peer_acts(live, artifact_id, True))

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
    seam.arm((artifact_id, holder.agent_id), _peer_acts(live, artifact_id, False))

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
    seam.arm((artifact_id, holder.agent_id), lambda: None)
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


def _volume_on(tmp_path: Path) -> tuple[CoordinatorHTTPServer, CoherentVolume, Path]:
    target = tmp_path / _VOLUME_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"v1")
    server = serve_in_process(tmp_path, "self-release-volume")
    vol = CoherentVolume(tmp_path, managed=("data/**",), config=_FAST)
    vol.read(_VOLUME_REL)
    return server, vol, target


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
