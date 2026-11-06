# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume(on_held="refuse")``: a write that will not take a file
another agent holds.

The volume opts in on ``write()`` for a managed path and types each answer. It
attaches only to a coordinator that advertises the refusal. These tests run the
real volume against a real coordinator in this process and stub only true
boundaries: an operator view without the key, which every released build
answers, and answers no current coordinator sends but an older build does. They
cannot run an older build itself, so what one does is stated in the guide, not
proved here.
"""

from __future__ import annotations

import io
import json
import os
import urllib.error
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import UUID

import pytest

import ccs.adapters.claude_code.coordinator_server as server_module
import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.cli._coherence_client as client_module
from ccs.adapters.claude_code.coordinator_server import (
    CoordinatorHTTPServer,
    session_to_agent_id,
)
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorEndpoint, CoordinatorUnavailable
from ccs.core.exceptions import (
    CoherenceError,
    CommitPreempted,
    CoordinatorFailed,
    CoordinatorUnreachable,
    GiverFenced,
    OtherHolder,
    StaleView,
)
from ccs.testing.silent_listener import SilentListener
from tests.coordinator_helpers import serve_in_process, write_holders

_MANAGED = ("data/**",)
_PATH = "data/shared.txt"
_OTHER_PATH = "data/other.txt"
_THIRD_PATH = "data/third.txt"
_CLAIM_ROUTE = client_module.PRINCIPAL_CLAIM_ROUTE
_PRE_EDIT = "/hooks/pre-edit"
_POST_EDIT = "/hooks/post-edit"
_SESSION_STOP = "/hooks/session-stop"
_WIRE = Path(__file__).parents[1] / "fixtures" / "wire"
#: The answer a coordinator sends an opted-in pre-edit it cannot decide in time.
_UNAVAILABLE = json.loads((_WIRE / "pre_edit_unavailable.json").read_text())
#: A holder id in the refusal's spelling that names no agent in these tests.
_SOME_HOLDER = "5d0f3c9a8b7e4d21a6c3b2e1f0d9c8b7"


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``data/**`` running in this process."""
    (tmp_path / "data").mkdir()
    running = serve_in_process(tmp_path, "refuse-mode")
    yield running
    running.shutdown()


@dataclass
class _Wire:
    """Every POST the volume sends, and the answers a test puts in place of the
    coordinator's. ``answers`` maps a route to the answer's body, or to a
    callable ``(forward, payload)`` that decides for itself."""

    sent: list[tuple[str, dict]] = field(default_factory=list)
    answers: dict[str, object] = field(default_factory=dict)

    def routes(self) -> list[str]:
        return _routes(self.sent)

    def bodies(self, route: str) -> list[dict]:
        return [body for sent_route, body in self.sent if sent_route == route]


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    """Record every POST, the principal claim included. The claim goes out
    through the client module, every other request through the volume's alias."""
    recorded = _Wire()
    real_post = client_module.post

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        recorded.sent.append((path, dict(payload)))
        scripted = recorded.answers.get(path)
        if callable(scripted):
            return scripted(lambda body=payload: real_post(endpoint, path, body, **kwargs), payload)
        if scripted is not None:
            return scripted
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(client_module, "post", spy)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)
    return recorded


@pytest.fixture
def sent(wire: _Wire) -> list[tuple[str, dict]]:
    return wire.sent


def _routes(sent: list[tuple[str, dict]]) -> list[str]:
    return [route for route, _ in sent]


def _refuser(root: Path, cfg: LifecycleConfig, **kwargs: object) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg, on_held="refuse", **kwargs)


def _displacer(root: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg)


def _version_of(volume: CoherentVolume) -> int:
    return volume.read_with_version(_THIRD_PATH)[1]


def _session_level_id(volume: CoherentVolume) -> str:
    """The id a handoff's successor is named by."""
    return str(session_to_agent_id(volume.session_id))


def _status(code: int) -> Callable[[Callable[..., object], dict], object]:
    """A scripted answer: this HTTP status, the request never reaching the coordinator."""

    def answer(forward: Callable[..., object], payload: dict) -> object:
        raise urllib.error.HTTPError(
            _PRE_EDIT, code, "scripted", {}, io.BytesIO(b"{}"),  # type: ignore[arg-type]
        )

    return answer


def _timed_out_write(
    server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
    write: Callable[[], None],
) -> CoordinatorFailed:
    """Run ``write`` with the registry lock held past the handler budget, so the
    real coordinator answers ``unavailable``, and return once the work it
    abandoned has settled. The budget is put back before returning."""
    from tests.test_claude_code_coordinator_server import (
        _DEGRADE_DEADLINE_SEC,
        _await_abandoned_body_settled,
        _HeldRegistryLock,
    )

    aborts = server._watchdog_late_aborts_total
    completions = server._watchdog_late_completion_total
    budget = server_module.HANDLER_TIMEOUT_SEC
    monkeypatch.setattr(server_module, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    with _HeldRegistryLock(server) as held:
        with pytest.raises(CoordinatorFailed) as refused:
            write()
        held.release()
    _await_abandoned_body_settled(server, aborts_before=aborts, completions_before=completions)
    monkeypatch.setattr(server_module, "HANDLER_TIMEOUT_SEC", budget)
    return refused.value


def _operator_view_changed_to(
    monkeypatch: pytest.MonkeyPatch, change: Callable[[dict], dict]
) -> None:
    """Pass every operator view through ``change``; every other GET is untouched."""
    real_get = coherent_volume_module._coordinator_get

    def view(endpoint: object, path: str, **kwargs: object) -> object:
        doc = real_get(endpoint, path, **kwargs)
        return change(doc) if "detail=full" in path else doc

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", view)


def _without_the_key(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "pre_edit_if_unheld"}


# --- what a refuse volume needs -------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"on_held": "bogus"},
        {"on_held": None},
        {"on_held": "refuse", "on_error": "degrade"},
        {"on_held": "refuse", "managed": ()},
        {
            "on_held": "refuse",
            "remote_endpoint": CoordinatorEndpoint(port=1, bearer="not-a-real-token"),
        },
    ],
    ids=["unknown-value", "none", "degrade", "no-managed-globs", "remote"],
)
def test_a_volume_that_cannot_honour_refuse_mode_is_never_built(
    tmp_path: Path, fast_cfg: LifecycleConfig, kwargs: dict
) -> None:
    """Refuse mode opts in on managed paths and must fail closed, so a value it
    cannot honour raises before anything is spawned. Each message names
    ``on_held``: without its own check a remote endpoint would still raise, from
    the remote block, and tell the caller to set an environment flag.

    Prevents a volume built with ``on_held="refuse"`` that never opts in and
    silently displaces every holder, and one that writes best-effort detached."""
    built: dict = {"managed": _MANAGED, "config": fast_cfg, **kwargs}
    with pytest.raises(ValueError, match="on_held"):
        CoherentVolume(tmp_path, **built)
    assert not (tmp_path / ".coherence").exists()


# --- the capability check at attach ---------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        _without_the_key,
        lambda doc: {**doc, "pre_edit_if_unheld": "true"},
        lambda doc: {**doc, "pre_edit_if_unheld": 1},
        lambda doc: {**doc, "pre_edit_if_unheld": None},
        lambda doc: {**doc, "pre_edit_if_unheld": False},
    ],
    ids=["absent", "string", "one", "null", "false"],
)
def test_a_coordinator_that_does_not_advertise_the_refusal_is_refused_at_attach(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[dict], dict],
) -> None:
    """Anything but JSON ``true`` means the coordinator may answer an opted-in
    request by taking the file from its holder, so the volume fails closed
    before it claims a principal or sends a pre-edit. The error is the plain one
    a glob the coordinator does not enforce raises.

    Prevents a refuse-mode write landing over a holder on an older coordinator,
    which a version number cannot predict."""
    _operator_view_changed_to(monkeypatch, change)
    with pytest.raises(CoherenceError, match="pre_edit_if_unheld") as refused:
        _refuser(tmp_path, fast_cfg)
    assert type(refused.value) is CoherenceError
    assert sent == [], "the check runs before the claim and before any pre-edit"


def test_a_refuse_volume_adds_no_request_to_attach(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The key is read from the operator view the glob check already fetched.

    Prevents the check costing a second status read on every attach."""
    views: list[str] = []

    def counted(doc: dict) -> dict:
        views.append("full")
        return doc

    _operator_view_changed_to(monkeypatch, counted)
    volume = _refuser(tmp_path, fast_cfg)
    assert volume.is_attached
    assert len(views) == 1
    assert _routes(sent) == [_CLAIM_ROUTE]


def test_a_displace_volume_attaches_to_a_coordinator_without_the_key(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only refuse mode reads the key, so an ordinary volume attaches to an
    older coordinator as it always has."""
    _operator_view_changed_to(monkeypatch, _without_the_key)
    assert CoherentVolume(tmp_path, managed=_MANAGED, config=fast_cfg).is_attached


def test_a_forked_child_checks_the_coordinator_again(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forked child attaches afresh on its first write, so it reads the view
    again and refuses a coordinator that no longer advertises the refusal.

    Prevents a child trusting what its parent read about another coordinator."""
    volume = _refuser(tmp_path, fast_cfg)
    _operator_view_changed_to(monkeypatch, _without_the_key)
    sent.clear()

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: report through the pipe, never return into pytest
        os.close(read_fd)
        report: dict = {"raised": None}
        try:
            volume.write(_PATH, b"from-the-child")
        except BaseException as exc:  # noqa: BLE001 - reported to the parent
            report = {"raised": type(exc).__name__, "message": str(exc)}
        report["routes"] = _routes(sent)
        os.write(write_fd, json.dumps(report).encode("utf-8"))
        os.close(write_fd)
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as pipe:
        report = json.loads(pipe.read().decode("utf-8"))
    os.waitpid(pid, 0)

    assert report["raised"] == "CoherenceError", report
    assert "pre_edit_if_unheld" in report["message"], report
    assert "/hooks/pre-edit" not in report["routes"], report
    assert not (tmp_path / _PATH).exists()


def test_a_refused_operator_view_keeps_its_own_message(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    sent: list[tuple[str, dict]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Node coordinator answers 501 to the operator view, so a managed
    volume cannot attach to it in either mode. The glob check reports that
    first, naming the status, and the capability check does not cover it up.

    Prevents a caller on Node being told to upgrade a coordinator that honours
    the flag, when what is missing is the operator view."""
    real_get = coherent_volume_module._coordinator_get

    def not_served(endpoint: object, path: str, **kwargs: object) -> object:
        if "detail=full" in path:
            raise urllib.error.HTTPError(path, 501, "not served", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]
        return real_get(endpoint, path, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_get", not_served)
    with pytest.raises(CoherenceError, match="HTTP 501") as refused:
        _refuser(tmp_path, fast_cfg)
    assert "pre_edit_if_unheld" not in str(refused.value)
    assert sent == []


# --- the refusal ----------------------------------------------------------


def test_a_write_another_agent_holds_is_refused_naming_the_holder(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A volume that never read the file is refused while another agent holds
    it. It raises the typed refusal naming that agent, the holder keeps the file
    and its bytes, this write records no grant, and the refusal is not retried.

    Prevents a refuse-mode write landing over a holder, a holder named by a
    session id, a refusal read as a stale view, and a ``reacquire()`` that sends
    a release for a grant the refusal proved was never taken."""
    holder, refused = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    holder_id, incarnation = holder.agent_id, refused.incarnation
    wire.sent.clear()

    with pytest.raises(OtherHolder) as raised:
        refused.write(_PATH, b"from-the-refused")

    error = raised.value
    assert type(error) is OtherHolder
    assert error.artifact_id == _PATH
    assert UUID(error.holder_agent_id) == UUID(holder_id)
    assert holder.session_id not in str(error)
    assert UUID(holder.session_id).hex not in str(error)
    assert wire.routes() == [_PRE_EDIT], "one request: no retry, no release, no commit"
    assert wire.bodies(_PRE_EDIT)[0]["if_unheld"] is True
    assert refused.incarnation == incarnation, "a refusal does not re-mint"
    assert (tmp_path / _PATH).read_bytes() == b"from-the-holder"
    assert write_holders(server, _PATH) == {holder_id}

    wire.sent.clear()
    refused.reacquire(_PATH)
    assert _SESSION_STOP not in wire.routes(), "nothing to release: no grant was taken"
    assert write_holders(server, _PATH) == {holder_id}


def test_a_refusal_tells_the_caller_to_read_before_writing_again(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A program obeys the exception it caught, and a retry with the same bytes
    lands over whatever the holder committed once the file comes free.

    Prevents the error itself sending a caller back with a blind write."""
    holder, refused = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    with pytest.raises(OtherHolder, match="read the file again"):
        refused.write(_PATH, b"from-the-refused")


def test_a_refusal_withdraws_only_this_path_from_the_record(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The refused volume may hold another file. The refusal took no grant on
    this one, and leaves the other recorded, so ``reacquire()`` still releases it.

    Prevents a refusal stranding a real grant the volume holds elsewhere."""
    holder, refused = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    refused.write(_OTHER_PATH, b"its own")
    holder.write(_PATH, b"from-the-holder")
    with pytest.raises(OtherHolder):
        refused.write(_PATH, b"from-the-refused")
    assert write_holders(server, _OTHER_PATH) == {refused.agent_id}

    wire.sent.clear()
    refused.reacquire(_OTHER_PATH)
    assert _SESSION_STOP in wire.routes()
    assert write_holders(server, _OTHER_PATH) == set()
    assert write_holders(server, _PATH) == {holder.agent_id}


def test_a_holders_repeat_write_is_admitted(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The refusal is about another agent. A volume writing a file it holds is
    admitted, and the file stays with the same identity.

    Prevents a refuse volume being refused by itself on its second commit."""
    volume = _refuser(tmp_path, fast_cfg)
    volume.write(_PATH, b"first")
    holder_id = volume.agent_id
    wire.sent.clear()
    volume.write(_PATH, b"second")
    assert wire.routes() == [_PRE_EDIT, _POST_EDIT]
    assert (tmp_path / _PATH).read_bytes() == b"second"
    assert write_holders(server, _PATH) == {holder_id}


def test_a_refusal_after_a_principal_recovery_is_still_the_refusal(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A principal the coordinator does not recognise is recovered inside the
    write: the same nonce claims again and the pre-edit goes out again. What that
    second pre-edit answers is what the caller gets.

    Prevents the recovery turning a refusal into an admit or a different error."""
    holder, refused = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    refused._principal = "not-the-principal-bound-to-this-session"
    wire.sent.clear()
    with pytest.raises(OtherHolder):
        refused.write(_PATH, b"from-the-refused")
    assert wire.routes().count(_PRE_EDIT) == 2
    assert wire.routes().count(_CLAIM_ROUTE) == 1
    assert (tmp_path / _PATH).read_bytes() == b"from-the-holder"


def test_a_holder_that_lets_go_admits_the_refused_volume(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """``reacquire()`` frees the file and the refused volume's next write is
    admitted. It never read the file, so it overwrites what the holder committed
    with no signal: that is what a write built without reading does, and why the
    refusal says to read first.

    Prevents a released file staying refused, and pins the overwrite the docs
    warn about, so a change to it is seen."""
    holder, refused = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    with pytest.raises(OtherHolder):
        refused.write(_PATH, b"from-the-refused")
    holder.reacquire(_PATH)
    refused.write(_PATH, b"from-the-refused")
    assert (tmp_path / _PATH).read_bytes() == b"from-the-refused"
    assert write_holders(server, _PATH) == {refused.agent_id}


def test_a_volume_that_read_first_builds_on_what_the_holder_committed(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The safe way through: a volume that read the file is told its copy is
    stale, waits for the holder to let go, and builds on the holder's bytes."""
    (tmp_path / _PATH).write_bytes(b"seed")
    holder, reader = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    assert reader.read(_PATH) == b"seed"
    holder.write(_PATH, b"from-the-holder")
    with pytest.raises(StaleView):
        reader.write(_PATH, b"built on seed")
    holder.reacquire(_PATH)
    reader.write(_PATH, reader.reacquire(_PATH) + b" + reader")
    assert (tmp_path / _PATH).read_bytes() == b"from-the-holder + reader"


def test_a_default_volume_sends_no_flag_and_displaces(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Without ``on_held="refuse"`` nothing changes on the wire or in the outcome."""
    holder, writer = _displacer(tmp_path, fast_cfg), _displacer(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    wire.sent.clear()
    writer.write(_PATH, b"from-the-writer")
    assert "if_unheld" not in wire.bodies(_PRE_EDIT)[0]
    assert (tmp_path / _PATH).read_bytes() == b"from-the-writer"
    assert write_holders(server, _PATH) == {writer.agent_id}


def test_a_peer_commit_after_a_read_is_a_stale_view(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A stale copy answers the strict deny, ahead of any holder check, and the
    record is withdrawn as for any deny.

    Prevents refuse mode weakening the stale-write protection it sits beside."""
    (tmp_path / _PATH).write_bytes(b"seed")
    peer, reader = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    reader.read(_PATH)
    peer.write_cas(_PATH, lambda current: current + b" + peer")
    with pytest.raises(StaleView):
        reader.write(_PATH, b"built on seed")
    assert (tmp_path / _PATH).read_bytes() == b"seed + peer"
    wire.sent.clear()
    reader.reacquire(_PATH)
    assert _SESSION_STOP not in wire.routes()


def test_a_poller_meets_both_answers_while_a_holder_keeps_committing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """While another agent keeps committing, a reader goes stale, is refused
    after it re-reads, and goes stale again at the next commit.

    Prevents a poller written to catch only the refusal breaking on the deny."""
    (tmp_path / _PATH).write_bytes(b"seed")
    holder, reader = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    reader.read(_PATH)
    holder.write(_PATH, b"one")
    with pytest.raises(StaleView):
        reader.write(_PATH, b"x")
    assert reader.reacquire(_PATH) == b"one"
    with pytest.raises(OtherHolder) as raised:
        reader.write(_PATH, b"x")
    assert UUID(raised.value.holder_agent_id) == UUID(holder.agent_id)
    holder.write(_PATH, b"two")
    with pytest.raises(StaleView):
        reader.write(_PATH, b"x")
    assert (tmp_path / _PATH).read_bytes() == b"two"


@pytest.mark.parametrize(
    "answer",
    [
        {"ok": False, "reason": "other_holder", "holder_agent_id": _SOME_HOLDER},
        _UNAVAILABLE,
    ],
    ids=["other_holder", "unavailable"],
)
def test_the_same_reason_on_a_commit_is_a_preemption(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    answer: dict,
) -> None:
    """A commit's refusal means the bytes are already on disk unversioned, so
    the route decides what the reason means, and a post-edit stays a preemption
    whatever reason it carries.

    Prevents a caller told to back off when its write has already landed."""
    volume = _refuser(tmp_path, fast_cfg)

    def a_commit_only(forward: Callable[..., object], payload: dict) -> object:
        return answer if payload.get("success") is True else forward()

    wire.answers[_POST_EDIT] = a_commit_only
    with pytest.raises(CommitPreempted):
        volume.write(_PATH, b"from-the-volume")


def test_a_real_preemption_between_the_grant_and_the_commit(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """An ordinary pre-edit still displaces an opted-in holder, so a grant can
    be lost between acquire and commit. That reads as a preemption, as before."""
    volume, displacing = _refuser(tmp_path, fast_cfg), _displacer(tmp_path, fast_cfg)

    def displace_then_commit(forward: Callable[..., object], payload: dict) -> object:
        wire.answers.pop(_POST_EDIT)
        displacing._post(_PRE_EDIT, {"session_id": displacing.session_id, "path": _PATH})
        return forward()

    wire.answers[_POST_EDIT] = displace_then_commit
    with pytest.raises(CommitPreempted):
        volume.write(_PATH, b"from-the-volume")


# --- failure answers ------------------------------------------------------


def test_an_undecided_grant_request_raises_coordinator_failed(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real ``unavailable`` answer: nothing is written and no commit or
    release goes out, and the record stays, because abandoned work can still
    take the grant. The caller's way out is the retry the error names.

    Prevents ``unavailable`` reading as a stale view, a release that would leave
    this identity denied, and a withdrawn record that strands a late grant."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    volume.read(_PATH)
    wire.sent.clear()

    error = _timed_out_write(server, monkeypatch, lambda: volume.write(_PATH, b"from-the-volume"))

    assert type(error) is CoordinatorFailed
    assert wire.routes() == [_PRE_EDIT]
    assert (tmp_path / _PATH).read_bytes() == b"seed"
    volume.write(_PATH, b"from-the-volume")
    assert (tmp_path / _PATH).read_bytes() == b"from-the-volume"


def test_a_retry_after_unavailable_keeps_the_volumes_other_holds(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The same answer with the coordinator free: a release sent here would land
    and deny this identity its retry. The retry is admitted under the same
    identity and the volume's hold on another file stays.

    Prevents a bare release, and any re-mint, costing the caller its other holds."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    volume.write(_OTHER_PATH, b"its own")
    volume.read(_PATH)
    holder_of_other, incarnation = volume.agent_id, volume.incarnation
    wire.answers[_PRE_EDIT] = _UNAVAILABLE
    wire.sent.clear()

    with pytest.raises(CoordinatorFailed):
        volume.write(_PATH, b"from-the-volume")
    assert wire.routes() == [_PRE_EDIT], "no release after an undecided grant request"

    wire.answers.pop(_PRE_EDIT)
    volume.write(_PATH, b"from-the-volume")
    assert volume.incarnation == incarnation
    assert write_holders(server, _OTHER_PATH) == {holder_of_other}
    assert (tmp_path / _PATH).read_bytes() == b"from-the-volume"


@pytest.mark.parametrize(
    "answer",
    [
        {"ok": False, "degraded": True, "reason": "something else"},
        {"ok": False, "reason": "internal: OperationalError"},
        {"ok": False, "reason": "some coordinator prose"},
    ],
    ids=["degraded-without-the-reason", "internal", "prose"],
)
def test_only_the_reason_makes_an_answer_undecided(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    answer: dict,
) -> None:
    """Other ``ok: false`` answers keep the stale view they raise today, and the
    record, because an ``internal:`` answer can follow a grant that landed.

    Prevents the new arm keying on ``ok: false`` or ``degraded``, either of which
    would retype answers it does not know."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    wire.answers[_PRE_EDIT] = answer
    with pytest.raises(StaleView):
        volume.write(_PATH, b"from-the-volume")
    assert (tmp_path / _PATH).read_bytes() == b"seed"
    wire.answers.pop(_PRE_EDIT)
    wire.sent.clear()
    volume.reacquire(_PATH)
    assert _SESSION_STOP in wire.routes(), "the record stays: a grant may have landed"


def test_a_5xx_on_the_grant_request_raises_coordinator_failed(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A 5xx says nothing of whether the request ran, so nothing is written and
    the record stays.

    Prevents a non-answer being taken as permission to write."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    wire.answers[_PRE_EDIT] = _status(503)
    wire.sent.clear()
    with pytest.raises(CoordinatorFailed):
        volume.write(_PATH, b"from-the-volume")
    assert wire.routes() == [_PRE_EDIT]
    assert (tmp_path / _PATH).read_bytes() == b"seed"
    wire.answers.pop(_PRE_EDIT)
    wire.sent.clear()
    volume.reacquire(_PATH)
    assert _SESSION_STOP in wire.routes()


def test_a_silent_coordinator_on_the_grant_request_raises_unreachable(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No answer within the client's timeout is a refusal for an opted-in caller,
    and the record stays, because the request may have reached the coordinator."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    monkeypatch.setattr(client_module, "CLI_HTTP_TIMEOUT_SEC", 0.2)
    with SilentListener() as listener:
        volume._endpoint = replace(volume._endpoint, port=listener.port)
        with pytest.raises(CoordinatorUnreachable):
            volume.write(_PATH, b"from-the-volume")
    assert (tmp_path / _PATH).read_bytes() == b"seed"
    assert _PATH in volume._grant_incarnations.get(volume.incarnation, set())


def test_a_refused_connection_on_the_grant_request_raises_unreachable(
    tmp_path: Path, fast_cfg: LifecycleConfig, wire: _Wire
) -> None:
    """A coordinator that went away is the same refusal, with nothing written."""
    (tmp_path / "data").mkdir()
    (tmp_path / _PATH).write_bytes(b"seed")
    running = serve_in_process(tmp_path, "refuse-mode-gone")
    volume = _refuser(tmp_path, fast_cfg)
    running.shutdown()
    with pytest.raises(CoordinatorUnreachable):
        volume.write(_PATH, b"from-the-volume")
    assert (tmp_path / _PATH).read_bytes() == b"seed"


def test_a_grant_taken_by_abandoned_work_stands_until_reacquire(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Work the coordinator abandoned had already won the lock, so the grant landed
    though the answer said ``unavailable``. The volume keeps the record, a peer is
    refused naming this volume's incarnation, and ``reacquire()`` releases it.

    Prevents a withdrawn record stranding a grant the caller was never told about."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume, peer = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    stranded_id = volume.agent_id

    def land_then_answer_unavailable(forward: Callable[..., object], payload: dict) -> object:
        wire.answers.pop(_PRE_EDIT)
        forward()
        return _UNAVAILABLE

    wire.answers[_PRE_EDIT] = land_then_answer_unavailable
    with pytest.raises(CoordinatorFailed):
        volume.write(_PATH, b"from-the-volume")
    assert (tmp_path / _PATH).read_bytes() == b"seed"
    assert write_holders(server, _PATH) == {stranded_id}

    with pytest.raises(OtherHolder) as raised:
        peer.write(_PATH, b"from-the-peer")
    assert UUID(raised.value.holder_agent_id) == UUID(stranded_id)

    wire.sent.clear()
    volume.reacquire(_PATH)
    assert _SESSION_STOP in wire.routes()
    assert write_holders(server, _PATH) == set()
    peer.write(_PATH, b"from-the-peer")
    assert (tmp_path / _PATH).read_bytes() == b"from-the-peer"


def test_a_grant_whose_answer_was_lost_stands_until_reacquire(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The request landed and the answer did not arrive. The grant is real, so a
    peer is refused naming this volume's incarnation, and only keeping the record
    lets ``reacquire()`` release it.

    Prevents a peer blocked by a grant nothing will ever release short of the
    sweep."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume, peer = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    lost_id = volume.agent_id

    def land_then_lose_the_answer(forward: Callable[..., object], payload: dict) -> object:
        wire.answers.pop(_PRE_EDIT)
        forward()
        raise CoordinatorUnavailable("the answer did not arrive")

    wire.answers[_PRE_EDIT] = land_then_lose_the_answer
    with pytest.raises(CoordinatorUnreachable):
        volume.write(_PATH, b"from-the-volume")
    assert (tmp_path / _PATH).read_bytes() == b"seed"

    with pytest.raises(OtherHolder) as raised:
        peer.write(_PATH, b"from-the-peer")
    assert UUID(raised.value.holder_agent_id) == UUID(lost_id)

    wire.sent.clear()
    volume.reacquire(_PATH)
    assert _SESSION_STOP in wire.routes()
    peer.write(_PATH, b"from-the-peer")
    assert (tmp_path / _PATH).read_bytes() == b"from-the-peer"


@pytest.mark.parametrize(
    "answer",
    [
        {},
        {"ok": None},
        {"ok": 0},
        {"ok": "false"},
        {"status": "stale"},
        {"reason": "other_holder"},
    ],
    ids=["empty", "null", "zero", "string", "status-only", "reason-only"],
)
def test_an_opted_in_write_is_admitted_only_on_ok_true(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    answer: dict,
) -> None:
    """The guide's client rule: admit only on ``ok: true``. A coordinator never
    sends these bodies. A stranger on a reused port could, and each lands bytes
    if taken as an admit.

    Prevents a caller that asked not to displace anyone writing on an answer
    that says nothing."""
    (tmp_path / _PATH).write_bytes(b"seed")
    volume = _refuser(tmp_path, fast_cfg)
    wire.answers[_PRE_EDIT] = answer
    wire.sent.clear()
    with pytest.raises(CoordinatorUnreachable):
        volume.write(_PATH, b"from-the-volume")
    assert wire.routes() == [_PRE_EDIT]
    assert (tmp_path / _PATH).read_bytes() == b"seed"


def test_a_default_volume_still_writes_on_those_answers(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Control: the check is for opted-in requests only, so a default volume
    treats an answer that is not ``ok: false`` as it always has."""
    volume = _displacer(tmp_path, fast_cfg)
    wire.answers[_PRE_EDIT] = {}
    volume.write(_PATH, b"from-the-volume")
    assert (tmp_path / _PATH).read_bytes() == b"from-the-volume"


# --- what opting in needs, on the path ------------------------------------


def test_a_session_with_no_principal_sends_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The Python coordinator refuses an opted-in request without a principal,
    so a claim that settled as unsupported fails closed here. Reached by
    answering the claim 404: a coordinator that publishes its glob sets also
    issues principals.

    Prevents a request the coordinator is bound to refuse, and a claim sent to
    work around it."""
    (tmp_path / _PATH).write_bytes(b"seed")
    wire.answers[_CLAIM_ROUTE] = _status(404)
    volume = _refuser(tmp_path, fast_cfg)
    assert volume.principal_claim_outcome == "unsupported"
    wire.answers.pop(_CLAIM_ROUTE)
    wire.sent.clear()

    with pytest.raises(CoherenceError, match="principal") as raised:
        volume.write(_PATH, b"from-the-volume")

    assert type(raised.value) is CoherenceError
    assert wire.sent == []
    assert (tmp_path / _PATH).read_bytes() == b"seed"
    volume.reacquire(_PATH)
    assert _SESSION_STOP not in wire.routes(), "nothing was recorded"


def test_a_write_outside_managed_sends_no_flag_and_displaces(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Refuse mode covers managed paths only: on any other path an untrack could
    take the untracked exit around the check, so the volume does not opt in, and
    a holder of a default-tracked file is displaced.

    Pins the limit the docs state, so a change to it is seen."""
    holder, writer = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write("plan.md", b"from-the-holder")
    assert write_holders(server, "plan.md") == {holder.agent_id}
    wire.sent.clear()
    writer.write("plan.md", b"from-the-writer")
    assert "if_unheld" not in wire.bodies(_PRE_EDIT)[0]
    assert (tmp_path / "plan.md").read_bytes() == b"from-the-writer"
    assert write_holders(server, "plan.md") == {writer.agent_id}


# --- this volume's own abandoned grant ------------------------------------


def _with_an_abandoned_grant(volume: CoherentVolume, wire: _Wire) -> None:
    """Leave ``volume`` with a grant under an old incarnation: it writes, then a
    ``reacquire()`` rotates its identity and its release does not land."""
    volume.write(_PATH, b"first")
    wire.answers[_SESSION_STOP] = _status(503)
    with pytest.raises(CoordinatorFailed):
        volume.reacquire(_PATH)
    wire.answers.pop(_SESSION_STOP)


def test_a_refusal_by_this_volumes_own_old_grant_is_not_another_agent(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A release that did not land leaves a grant under an old incarnation, and
    the coordinator refuses the volume on it. No peer exists to wait for, so the
    volume says so and names the way out, which a second ``reacquire()`` proves.

    Prevents a caller backing off for half an hour from itself."""
    volume = _refuser(tmp_path, fast_cfg)
    _with_an_abandoned_grant(volume, wire)

    with pytest.raises(CoordinatorFailed, match="reacquire") as raised:
        volume.write(_PATH, b"second")
    assert type(raised.value) is CoordinatorFailed
    assert (tmp_path / _PATH).read_bytes() == b"first"

    volume.write(_PATH, volume.reacquire(_PATH) + b" + second")
    assert (tmp_path / _PATH).read_bytes() == b"first + second"


@pytest.mark.parametrize(
    "holder",
    ["not-a-uuid", None, 7],
    ids=["unparsable", "absent", "not-a-string"],
)
def test_a_holder_id_that_does_not_parse_is_never_this_volumes_own(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    holder: object,
) -> None:
    """The holder comes off the wire. An id that does not parse names no
    recorded incarnation, so the caller gets the refusal, with that string or
    ``None``, and never a ``ValueError`` out of ``write()``."""
    volume = _refuser(tmp_path, fast_cfg)
    _with_an_abandoned_grant(volume, wire)
    answer = {"ok": False, "reason": "other_holder"}
    if holder is not None:
        answer["holder_agent_id"] = holder
    wire.answers[_PRE_EDIT] = answer
    with pytest.raises(OtherHolder) as raised:
        volume.write(_PATH, b"second")
    assert raised.value.holder_agent_id == (holder if isinstance(holder, str) else None)


def test_a_peers_refusal_is_not_this_volumes_own(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A well-formed id that is none of this volume's incarnations is a peer, even
    while the volume records an abandoned grant.

    Prevents a real peer's refusal being reported as this volume's own, which
    would send the caller to ``reacquire()`` instead of waiting."""
    volume = _refuser(tmp_path, fast_cfg)
    _with_an_abandoned_grant(volume, wire)
    wire.answers[_PRE_EDIT] = {
        "ok": False, "reason": "other_holder", "holder_agent_id": _SOME_HOLDER,
    }
    with pytest.raises(OtherHolder) as raised:
        volume.write(_PATH, b"second")
    assert raised.value.holder_agent_id == _SOME_HOLDER


# --- an older coordinator at the write itself -----------------------------


def _strip_the_flag(forward: Callable[..., object], payload: dict) -> object:
    """Forward the opted-in pre-edit with the flag removed, which is what a build
    that ignores it decides: the current coordinator never answers an opted-in
    request with ``collision``, so this is the way to meet that answer."""
    return forward({k: v for k, v in payload.items() if k != "if_unheld"})


def test_an_admit_over_a_holder_releases_and_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A coordinator replaced after attach can answer ``collision``, which a
    build that honours the flag never sends. The holder may already be displaced,
    so the volume writes nothing, releases what it took, and says what happened.
    A confirmed release leaves nothing recorded.

    Prevents a refuse-mode write landing over a holder on the one answer that
    shows the coordinator ignored the flag."""
    holder, writer = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    wire.answers[_PRE_EDIT] = _strip_the_flag
    wire.sent.clear()

    with pytest.raises(CoherenceError, match="pre_edit_if_unheld") as raised:
        writer.write(_PATH, b"from-the-writer")

    assert type(raised.value) is CoherenceError
    assert "nothing changed hands" not in str(raised.value)
    assert wire.routes() == [_PRE_EDIT, _POST_EDIT]
    assert wire.bodies(_POST_EDIT)[0]["success"] is False
    assert (tmp_path / _PATH).read_bytes() == b"from-the-holder"
    assert write_holders(server, _PATH) == set()
    wire.answers.pop(_PRE_EDIT)
    wire.sent.clear()
    writer.reacquire(_PATH)
    assert _SESSION_STOP not in wire.routes()


def test_an_unanswered_release_after_a_displacing_admit_keeps_the_record(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The grant is real until a release is confirmed, so ``reacquire()`` must
    still send one.

    Prevents a release that did not land stranding the grant until the sweep."""
    holder, writer = _displacer(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    wire.answers[_PRE_EDIT] = _strip_the_flag
    wire.answers[_POST_EDIT] = _status(503)
    with pytest.raises(CoherenceError, match="pre_edit_if_unheld"):
        writer.write(_PATH, b"from-the-writer")
    wire.answers.clear()
    wire.sent.clear()
    writer.reacquire(_PATH)
    assert _SESSION_STOP in wire.routes()
    assert write_holders(server, _PATH) == set()


# --- letting go -----------------------------------------------------------


def test_a_holder_keeps_the_file_across_its_commits(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A successful ``write()`` sends no release, so a refuse volume protects its
    work between commits, not only inside one.

    Prevents a release after each commit, which would let any peer in between
    two steps of the holder's work."""
    holder, peer = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    wire.sent.clear()
    holder.write(_PATH, b"step one")
    assert wire.routes() == [_PRE_EDIT, _POST_EDIT]
    assert wire.bodies(_POST_EDIT)[0]["success"] is True
    with pytest.raises(OtherHolder):
        peer.write(_PATH, b"from-the-peer")
    holder.write(_PATH, b"step two")
    assert (tmp_path / _PATH).read_bytes() == b"step two"


def test_reacquire_lets_go_of_every_path(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """``reacquire()`` of one path releases every path this attempt holds, which
    is the cost of using it as the release.

    Pins what the docs state, so a change to it is seen."""
    holder, peer = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"x")
    holder.write(_OTHER_PATH, b"z")
    holder.reacquire(_PATH)
    peer.write(_PATH, b"from-the-peer")
    peer.write(_OTHER_PATH, b"from-the-peer")
    assert write_holders(server, _OTHER_PATH) == {peer.agent_id}


def test_the_next_write_after_reacquire_lands_under_the_new_identity(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """With no peer in the way, the holder carries on from the bytes it got."""
    holder = _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"one")
    old_id = holder.agent_id
    holder.write(_PATH, holder.reacquire(_PATH) + b" two")
    assert holder.agent_id != old_id
    assert write_holders(server, _PATH) == {holder.agent_id}
    assert (tmp_path / _PATH).read_bytes() == b"one two"


@pytest.mark.parametrize(
    "drop_holds",
    [
        lambda volume: volume.write_cas(_PATH, lambda current: current + b" + cas"),
        lambda volume: volume.write_cas_at(_THIRD_PATH, _version_of(volume), b"third"),
        lambda volume: volume.atomic_publish([(_THIRD_PATH, _version_of(volume), b"third")]),
    ],
    ids=["write_cas-of-a-held-path", "write_cas_at", "atomic_publish"],
)
def test_other_verbs_drop_every_hold_too(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    drop_holds: Callable[[CoherentVolume], object],
) -> None:
    """These verbs start a new attempt, which ends the holds of the last one, so
    a peer is admitted on a file the volume never named in the call.

    Pins what the docs list, so a change to it is seen."""
    (tmp_path / _THIRD_PATH).write_bytes(b"seed")
    holder, peer = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"x")
    holder.write(_OTHER_PATH, b"z")
    drop_holds(holder)
    peer.write(_OTHER_PATH, b"from-the-peer")
    assert write_holders(server, _OTHER_PATH) == {peer.agent_id}


def test_reacquire_of_a_deleted_file_has_released_when_it_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The release comes before the read, so a raise does not mean still held.

    Pins why the docs call it best-effort in both directions."""
    holder, peer = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_PATH, b"x")
    (tmp_path / _PATH).unlink()
    with pytest.raises(FileNotFoundError):
        holder.reacquire(_PATH)
    peer.write(_PATH, b"from-the-peer")
    assert (tmp_path / _PATH).read_bytes() == b"from-the-peer"


@pytest.mark.parametrize("mode", ["raise", "allow"])
def test_reacquire_drops_the_stale_deny_on_other_paths(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, mode: str
) -> None:
    """A new identity carries no stale view of the files the old one read, so a
    write from an old copy of another file meets only the content check: raised
    by default, landing under ``on_stale_write="allow"``. This is why the docs say
    to read every other file again after ``reacquire()``.

    Pins the cost of using it as the release, so a change to it is seen."""
    (tmp_path / _OTHER_PATH).write_bytes(b"seed")
    holder = _refuser(tmp_path, fast_cfg, on_stale_write=mode)
    peer = _displacer(tmp_path, fast_cfg)
    holder.read(_OTHER_PATH)
    holder.write(_PATH, b"x")
    peer.write_cas(_OTHER_PATH, lambda current: current + b" + peer")
    holder.reacquire(_PATH)
    if mode == "raise":
        with pytest.raises(StaleView):
            holder.write(_OTHER_PATH, b"from an old copy")
        assert (tmp_path / _OTHER_PATH).read_bytes() == b"seed + peer"
    else:
        holder.write(_OTHER_PATH, b"from an old copy")
        assert (tmp_path / _OTHER_PATH).read_bytes() == b"from an old copy"


# --- a live handoff -------------------------------------------------------


def test_a_refuse_giver_is_fenced_not_refused(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The giver of a live handoff is fenced by the handoff itself, ahead of any
    holder check, so waiting is no use to it."""
    (tmp_path / _PATH).write_bytes(b"seed")
    giver, successor = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    giver.read(_PATH)
    assert giver.transfer(_PATH, successor=_session_level_id(successor)).ok
    with pytest.raises(GiverFenced):
        giver.write(_PATH, b"from-the-giver")


def test_a_refuse_bystander_is_refused_while_the_successor_holds(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """Once the successor holds the file, a bystander gets the ordinary refusal."""
    (tmp_path / _PATH).write_bytes(b"seed")
    giver, successor = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    bystander = _refuser(tmp_path, fast_cfg)
    giver.read(_PATH)
    assert giver.transfer(_PATH, successor=_session_level_id(successor)).ok
    successor.write(_PATH, successor.read(_PATH) + b" + successor")
    with pytest.raises(OtherHolder) as raised:
        bystander.write(_PATH, b"from-the-bystander")
    assert UUID(raised.value.holder_agent_id) == UUID(successor.agent_id)


def test_a_refuse_bystander_overtakes_a_handoff_nobody_holds(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A transfer leaves nobody holding, so a bystander's opted-in write is
    admitted and overtakes the handoff with no signal to the writer. The successor
    is then refused by it. The record shows through a read, never through the
    refusal or the write.

    Pins the limit the docs state, so a change to it is seen."""
    (tmp_path / _PATH).write_bytes(b"seed")
    giver, successor = _refuser(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    bystander = _refuser(tmp_path, fast_cfg)
    giver.read(_PATH)
    assert giver.transfer(_PATH, successor=_session_level_id(successor)).ok

    bystander.write(_PATH, b"from-the-bystander")

    bystander.read(_PATH)
    handoff = bystander.read_handoff(_PATH)
    assert handoff is not None and handoff["status"] == "overtaken", handoff
    with pytest.raises(OtherHolder) as raised:
        successor.write(_PATH, b"from-the-successor")
    assert UUID(raised.value.holder_agent_id) == UUID(bystander.agent_id)
