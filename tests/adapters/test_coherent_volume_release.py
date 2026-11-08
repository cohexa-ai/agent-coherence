# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume.release(path)``: a volume gives back its write grant on one
file and keeps every other hold (#505).

The volume decides what to release from its own record of the attempts whose
``write()`` took a grant, because the coordinator answers a release the same way
whether it released a grant, held nothing, or was named the wrong attempt. These
tests run the real volume against a real coordinator in this process and script
only the answers a test needs the coordinator to give.
"""

from __future__ import annotations

import builtins
import contextlib
import socket
import threading
import urllib.error
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume, coherent_workspace, install, uninstall
from ccs.adapters.effect_gate import gate
from ccs.cli._coherence_client import CoordinatorUnavailable, resolve_endpoint, resolve_remote_endpoint
from ccs.core.exceptions import (
    CoherenceDegradedWarning,
    CoherenceError,
    CoordinatorFailed,
    CoordinatorUnreachable,
    InternalConcurrencyError,
    OtherHolder,
    StaleView,
)
from tests.adapters.test_coherent_volume_other_holder import (
    _MANAGED,
    _POST_EDIT,
    _SESSION_STOP,
    _status,
    _Wire,
)
from tests.coordinator_helpers import recorded_version, serve_in_process, write_holders
from tests.loopback_helpers import refused_endpoint

_X = "data/x.txt"
_ORIGINAL_OPEN = builtins.open
_Z = "data/z.txt"


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
    running = serve_in_process(tmp_path, "release")
    yield running
    running.shutdown()


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    """Every POST the volumes send, with answers a test can put in place."""
    import ccs.adapters.coherent_volume as coherent_volume_module
    import ccs.cli._coherence_client as client_module

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


def _holder(root: Path, cfg: LifecycleConfig, **kwargs: object) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg, **kwargs)


def _refuser(root: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg, on_held="refuse")


def _releases(wire: _Wire) -> list[dict]:
    """The releases sent: failed post-edits."""
    return [body for body in wire.bodies(_POST_EDIT) if body.get("success") is False]


def _no_answer(forward: Callable[..., object], payload: dict) -> object:
    raise CoordinatorUnavailable("scripted: the answer was lost")


def _not_an_object(forward: Callable[..., object], payload: dict) -> object:
    return ["not", "an", "object"]


def _held_answer(rel: str) -> dict:
    return {
        "ok": False,
        "reason": "release_failed",
        "grants": [{"path": rel, "held": True}],
    }


def test_releasing_one_path_admits_a_peer_there_and_keeps_the_others(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A volume that wrote two files gives one back: a refuse-mode peer is
    admitted on it and still refused on the other.

    Prevents a release that frees nothing, or one that frees every hold."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"x from the holder")
    holder.write(_Z, b"z from the holder")

    holder.release(_X)

    peer.write(_X, b"x from the peer")
    assert write_holders(server, _X) == {peer.agent_id}
    with pytest.raises(OtherHolder):
        peer.write(_Z, b"z from the peer")
    assert write_holders(server, _Z) == {holder.agent_id}


def test_a_released_path_stays_readable_and_writable_with_no_reacquire(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """After a release the volume keeps a read of the version it held, so it
    reads the file and its next write lands on the same attempt.

    Prevents a release that leaves the volume refused on its own file."""
    holder = _holder(tmp_path, fast_cfg, on_stale_read="raise")
    holder.write(_X, b"v1")
    attempt, version = holder.incarnation, recorded_version(server, _X)

    holder.release(_X)

    assert holder.read(_X) == b"v1"
    holder.write(_X, b"v2")
    assert holder.incarnation == attempt
    assert recorded_version(server, _X) == version + 1
    assert (tmp_path / _X).read_bytes() == b"v2"


def test_a_peer_write_after_the_release_makes_the_volumes_copy_stale(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A released path is a read like any other: a peer's write invalidates it.

    Prevents a release that lets the volume overwrite a peer's commit."""
    holder, peer = _holder(tmp_path, fast_cfg), _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    holder.release(_X)
    peer.write(_X, b"from the peer")

    with pytest.raises(StaleView):
        holder.write(_X, b"over the peer")
    assert (tmp_path / _X).read_bytes() == b"from the peer"


def _strand_a_grant(holder: CoherentVolume, wire: _Wire) -> str:
    """Leave ``holder`` with a grant on ``_X`` under an earlier attempt: it
    writes, then a ``reacquire()`` rotates the attempt and its release does not
    land. Returns that earlier attempt."""
    holder.write(_X, b"first")
    earlier = holder.incarnation
    wire.answers[_SESSION_STOP] = _status(503)
    with pytest.raises(CoordinatorFailed):
        holder.reacquire(_X)
    wire.answers.pop(_SESSION_STOP)
    return earlier


def test_release_names_every_attempt_that_recorded_the_path(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A path written under an earlier attempt and the current one: one release
    per attempt, each naming its own, the current one first.

    Prevents a release that names only the current attempt and leaves the
    earlier grant standing, which the coordinator would answer as released."""
    holder = _holder(tmp_path, fast_cfg)
    earlier = _strand_a_grant(holder, wire)
    holder.write(_X, b"second")
    wire.sent.clear()

    holder.release(_X)

    assert [body["agent_id"] for body in _releases(wire)] == [holder.incarnation, earlier]
    assert write_holders(server, _X) == set()


def test_after_releasing_two_attempts_a_transfer_presents_the_one_that_held_the_grant(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Both releases are answered as released, but only the current attempt
    held the grant; the earlier one was displaced by it. The volume's read now
    stands with the current attempt, so a transfer presents that one and hands
    the read over.

    Prevents a transfer refused as not held while the volume holds the read."""
    holder, successor = _holder(tmp_path, fast_cfg), _holder(tmp_path, fast_cfg)
    _strand_a_grant(holder, wire)
    holder.write(_X, b"second")
    holder.release(_X)

    result = holder.transfer(_X, successor=str(session_to_agent_id(successor.session_id)))

    assert result.ok is True
    assert [grant.transferred for grant in result.grants] == [True]


def test_a_transfer_after_a_release_presents_the_released_attempt(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A path read under an earlier attempt and written under the current one:
    after the release, the read the coordinator keeps is the current attempt's,
    so that is the claim a transfer presents.

    Prevents a transfer presenting the attempt that took the older read."""
    (tmp_path / _X).write_bytes(b"seed")
    (tmp_path / _Z).write_bytes(b"z1")
    holder, successor = _holder(tmp_path, fast_cfg), _holder(tmp_path, fast_cfg)
    holder.read(_X)
    holder.reacquire(_Z)
    holder.write(_X, b"written by the second attempt")
    holder.release(_X)
    wire.sent.clear()

    result = holder.transfer(_X, successor=str(session_to_agent_id(successor.session_id)))

    assert result.ok is True
    presented = [
        grant["agent_id"] for route, body in wire.sent if route == "/handoff/transfer" for grant in body["grants"]
    ]
    assert presented == [holder.incarnation]


def test_a_path_the_volume_did_not_write_sends_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Nothing is recorded for a path the volume only read, so nothing is sent.

    Prevents a request per release() of a path that holds no grant."""
    (tmp_path / _X).write_bytes(b"seed")
    holder = _holder(tmp_path, fast_cfg)
    holder.read(_X)
    wire.sent.clear()

    holder.release(_X)

    assert wire.sent == []


def test_another_spelling_of_a_held_path_releases_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The volume records a path by the spelling it was written under, so a
    release of another spelling (case here) finds nothing and sends nothing,
    and a refuse-mode peer is still refused on the file.

    Pins the limit the docs state until other spellings are handled (#506)."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.sent.clear()

    holder.release("data/X.TXT")

    assert _releases(wire) == []
    with pytest.raises(OtherHolder):
        peer.write(_X, b"from the peer")
    assert write_holders(server, _X) == {holder.agent_id}


@pytest.mark.parametrize("spelling", ["dot-slash", "absolute", "symlink"])
def test_a_spelling_that_resolves_to_the_held_path_releases_it(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, spelling: str
) -> None:
    """A spelling the volume resolves to the path it recorded is that path.

    Prevents a release that finds nothing for a path given another way."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    given: dict[str, object] = {"dot-slash": f"./{_X}", "absolute": tmp_path / _X}
    if spelling == "symlink":
        (tmp_path / "data" / "link.txt").symlink_to(tmp_path / _X)
        given["symlink"] = tmp_path / "data" / "link.txt"

    holder.release(given[spelling])  # type: ignore[arg-type]

    peer.write(_X, b"from the peer")
    assert write_holders(server, _X) == {peer.agent_id}


_NO_ACCOUNT = {"ok": False, "reason": "release_failed"}

_UNCONFIRMED = [
    pytest.param({"ok": True, "degraded": True}, CoherenceError, id="degraded"),
    pytest.param({"ok": False, "reason": "internal: RuntimeError"}, CoordinatorFailed, id="envelope"),
    pytest.param(_status(503), CoordinatorFailed, id="http-503"),
    pytest.param(_no_answer, CoordinatorUnreachable, id="no-answer"),
    pytest.param(_not_an_object, CoordinatorUnreachable, id="not-an-object"),
    pytest.param(_held_answer(_X), CoherenceError, id="grant-still-held"),
    pytest.param(_NO_ACCOUNT, CoherenceError, id="no-account-of-the-grant"),
]


@pytest.mark.parametrize(("answer", "raised"), _UNCONFIRMED)
def test_an_unconfirmed_release_raises_keeps_the_record_and_is_sent_again(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    answer: object,
    raised: type[CoherenceError],
) -> None:
    """Every answer that does not confirm the release raises in strict mode,
    typed by what went wrong, with a text that says the grant may still stand
    and to call release() again. The record keeps the path, so the next
    release() sends the request again and lands.

    Prevents a release reported done while the grant stands, and one the
    volume forgets it still owes."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.answers[_POST_EDIT] = answer

    with pytest.raises(raised, match=r"release\(\) again") as failure:
        holder.release(_X)
    assert type(failure.value) is raised
    assert _X in str(failure.value)

    wire.answers.pop(_POST_EDIT)
    wire.sent.clear()
    holder.release(_X)
    assert len(_releases(wire)) == 1
    peer.write(_X, b"from the peer")
    assert write_holders(server, _X) == {peer.agent_id}


def test_an_unconfirmed_release_warns_in_degrade_mode_and_keeps_the_record(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Degrade mode warns instead of raising and keeps the path recorded.

    Prevents a degrade volume dropping a grant it never released."""
    holder = _holder(tmp_path, fast_cfg, on_error="degrade")
    holder.write(_X, b"from the holder")
    wire.answers[_POST_EDIT] = {"ok": True, "degraded": True}

    with pytest.warns(CoherenceDegradedWarning):
        holder.release(_X)

    wire.answers.pop(_POST_EDIT)
    wire.sent.clear()
    holder.release(_X)
    assert len(_releases(wire)) == 1


@pytest.mark.parametrize("route", [_POST_EDIT, _SESSION_STOP])
def test_an_unanswered_release_in_degrade_mode_counts_one_degradation(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire, route: str
) -> None:
    """The request already counted the degradation when it got no answer, so
    the release or close that sent it does not count it again; the grant stays
    recorded.

    Prevents one lost answer counted as two degradations."""
    holder = _holder(tmp_path, fast_cfg, on_error="degrade")
    holder.write(_X, b"from the holder")
    wire.answers[route] = _no_answer
    let_go = (lambda: holder.release(_X)) if route == _POST_EDIT else holder.close

    with pytest.warns(CoherenceDegradedWarning):
        let_go()

    assert holder.degradation_count == 1
    assert holder._records_write_grant(_X)


def test_a_handed_off_grant_counts_as_released(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A per-grant answer that reports the grant handed off settles it: the
    volume holds nothing there to release again.

    Prevents a release that raises over a file the volume already handed on."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.answers[_POST_EDIT] = {
        "ok": False,
        "reason": "giver_fenced",
        "grants": [{"path": _X, "held": False, "cause": "handoff"}],
    }

    holder.release(_X)

    wire.answers.pop(_POST_EDIT)
    wire.sent.clear()
    holder.release(_X)
    assert _releases(wire) == []


def test_release_inside_a_write_cas_closure_works(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The guard is re-entrant for the thread that holds it, so a closure can
    let go of another file while its own commit is in flight.

    Prevents a release refused as concurrent use by the volume's own thread."""
    (tmp_path / _Z).write_bytes(b"z1")
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")

    def let_go_then_append(current: bytes) -> bytes:
        holder.release(_X)
        return current + b" + z2"

    holder.write_cas(_Z, let_go_then_append)

    peer.write(_X, b"from the peer")
    assert (tmp_path / _Z).read_bytes() == b"z1 + z2"


def test_release_from_another_thread_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A volume is one operation at a time: a release from a second thread
    while the first holds the volume raises and sends nothing.

    Prevents a release splitting another thread's operation across attempts."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    held = threading.Event()
    done = threading.Event()

    def hold() -> None:
        with holder._single_op_guard():
            held.set()
            done.wait(30)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(5)
    try:
        with pytest.raises(InternalConcurrencyError):
            holder.release(_X)
    finally:
        done.set()
        thread.join(5)


def test_a_release_keeps_the_foreign_edit_guard(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A release changes what the coordinator holds, not what the volume last
    saw on disk, so a change made outside the coordinator after the release is
    still caught at the next write.

    Prevents a release that lets a write replace an out-of-band edit unseen."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    holder.release(_X)
    (tmp_path / _X).write_bytes(b"edited outside the coordinator")

    with pytest.raises(StaleView):
        holder.write(_X, b"over the edit")
    assert (tmp_path / _X).read_bytes() == b"edited outside the coordinator"


# --- close() and with -------------------------------------------------------


def _two_attempts_with_grants(holder: CoherentVolume, wire: _Wire) -> str:
    """Leave ``holder`` with a grant on ``_X`` under an earlier attempt and one
    on ``_Z`` under the current attempt. Returns the earlier attempt."""
    earlier = _strand_a_grant(holder, wire)
    holder.write(_Z, b"z from the holder")
    return earlier


def test_close_gives_back_every_write_grant(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A volume with grants under two attempts closes: a refuse-mode peer is
    admitted on both files, and the volume detaches.

    Prevents a close that frees only the current attempt's grants."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    _two_attempts_with_grants(holder, wire)

    holder.close()

    assert holder.closed is True
    assert holder.is_attached is False
    peer.write(_X, b"x from the peer")
    peer.write(_Z, b"z from the peer")
    assert write_holders(server, _X) == write_holders(server, _Z) == {peer.agent_id}


def test_close_stops_at_the_first_request_with_no_answer(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """With two attempts recorded, a session-stop that gets no answer ends the
    walk, so a coordinator that stopped answering costs one client timeout.

    Prevents a silent coordinator costing a timeout per recorded attempt."""
    holder = _holder(tmp_path, fast_cfg)
    _two_attempts_with_grants(holder, wire)
    wire.sent.clear()
    wire.answers[_SESSION_STOP] = _no_answer

    with pytest.raises(CoordinatorUnreachable, match=r"close\(\) again"):
        holder.close()

    assert len(wire.bodies(_SESSION_STOP)) == 1


def test_release_stops_at_the_first_request_with_no_answer(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Prevents a release costing a client timeout per attempt that wrote the
    path when the coordinator has stopped answering."""
    holder = _holder(tmp_path, fast_cfg)
    _strand_a_grant(holder, wire)
    holder.write(_X, b"second")
    wire.sent.clear()
    wire.answers[_POST_EDIT] = _no_answer

    with pytest.raises(CoordinatorUnreachable, match=r"release\(\) again"):
        holder.release(_X)

    assert len(_releases(wire)) == 1


def test_a_close_answered_per_grant_as_released_or_handed_off_ends_cleanly(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """An account of each grant that reports none still held confirms the
    close: nothing stays recorded and close() returns.

    Prevents a close that raises over grants the coordinator let go."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.answers[_SESSION_STOP] = {
        "ok": False,
        "reason": "giver_fenced",
        "grants": [{"path": _X, "held": False, "cause": "handoff"}],
    }

    holder.close()

    assert holder.is_attached is False
    assert not holder._records_write_grant(_X)


def test_close_goes_on_past_a_grant_reported_held_and_then_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The coordinator answered, so a grant it reports still held does not
    stop the close: the other attempt's grants are released, then the close
    raises, and a second close() sends what is still recorded.

    Prevents one held grant leaving every later grant to the sweep."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    earlier = _two_attempts_with_grants(holder, wire)
    path_of = {earlier: _X, holder.incarnation: _Z}
    first: list[str] = []

    def held_for_the_first_attempt(forward: Callable[..., object], payload: dict) -> object:
        # Attempt ids are random, so hold whichever the walk visits first: the
        # walk must then go on to the other one.
        if first:
            return forward()
        first.append(payload["agent_id"])
        return _held_answer(path_of[payload["agent_id"]])

    wire.answers[_SESSION_STOP] = held_for_the_first_attempt

    with pytest.raises(CoherenceError, match=r"close\(\) again") as failure:
        holder.close()
    held_path = path_of[first[0]]
    released_path = _Z if held_path == _X else _X
    assert type(failure.value) is CoherenceError
    assert held_path in str(failure.value)
    peer.write(released_path, b"from the peer")

    wire.answers.pop(_SESSION_STOP)
    holder.close()
    peer.write(held_path, b"from the peer")
    assert write_holders(server, held_path) == {peer.agent_id}


_AFTER_CLOSE: dict[str, Callable[[CoherentVolume], object]] = {
    "read": lambda v: v.read(_X),
    "write": lambda v: v.write(_X, b"new"),
    "reacquire": lambda v: v.reacquire(_X),
    "release": lambda v: v.release(_X),
    "read_with_version": lambda v: v.read_with_version(_X),
    "read_with_version_generation": lambda v: v.read_with_version_generation(_X),
    "write_cas": lambda v: v.write_cas(_X, lambda current: current),
    "write_cas_at": lambda v: v.write_cas_at(_X, 1, b"new"),
    "atomic_publish": lambda v: v.atomic_publish([(_X, 1, b"new")]),
    "transfer": lambda v: v.transfer(_X, successor="0" * 32),
    "accept": lambda v: v.accept(_X),
    "decline": lambda v: v.decline(_X),
    "withdraw": lambda v: v.withdraw(_X),
    "enter": lambda v: v.__enter__(),
    "gate": lambda v: gate(v, _X, decide=lambda data: data, effect=lambda data: data),
    "strict_mode_active": lambda v: v.strict_mode_active(),
    "coordinator_status": lambda v: v.coordinator_status(),
    "managed_glob_enforcement": lambda v: v.managed_glob_enforcement(),
    "read_handoff": lambda v: v.read_handoff(_X),
}


@pytest.mark.parametrize("operation", list(_AFTER_CLOSE))
def test_every_operation_but_close_raises_on_a_closed_volume_and_sends_nothing(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    operation: str,
) -> None:
    """A closed volume refuses every operation without a request, and its
    properties stay readable.

    Prevents a closed volume taking a new grant nothing will ever release."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    holder.close()
    wire.sent.clear()

    with pytest.raises(CoherenceError, match="closed"):
        _AFTER_CLOSE[operation](holder)

    assert wire.sent == []
    assert (tmp_path / _X).read_bytes() == b"from the holder"
    assert holder.session_id and holder.incarnation and holder.agent_id


def test_closing_twice_sends_nothing_the_second_time(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """close() is the one operation a closed volume accepts: with nothing left
    recorded it sends nothing, so a with block whose body closed the volume
    exits cleanly.

    Prevents an exit path raising over a volume its caller already closed."""
    holder = _holder(tmp_path, fast_cfg)
    with holder as volume:
        volume.write(_X, b"from the holder")
        volume.close()
        wire.sent.clear()
        volume.close()
    assert wire.sent == []


def test_a_volume_that_wrote_nothing_closes_without_a_request(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Prevents a close spending a request on a volume that holds no grant."""
    (tmp_path / _X).write_bytes(b"seed")
    holder = _holder(tmp_path, fast_cfg)
    holder.read(_X)
    wire.sent.clear()

    holder.close()

    assert wire.sent == []
    assert holder.closed is True


_CLOSE_UNCONFIRMED = [
    pytest.param({"ok": True, "degraded": True}, CoherenceError, id="degraded"),
    pytest.param({"ok": False, "reason": "internal: RuntimeError"}, CoordinatorFailed, id="envelope"),
    pytest.param(_status(503), CoordinatorFailed, id="http-503"),
    pytest.param(_no_answer, CoordinatorUnreachable, id="no-answer"),
    pytest.param(_NO_ACCOUNT, CoherenceError, id="no-account-of-the-grant"),
]


@pytest.mark.parametrize(("answer", "raised"), _CLOSE_UNCONFIRMED)
def test_an_unconfirmed_close_raises_stays_closed_and_is_sent_again(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    answer: object,
    raised: type[CoherenceError],
) -> None:
    """A close the coordinator does not confirm raises in strict mode, typed by
    what went wrong, and leaves the volume closed with the grant recorded; a
    second close() sends it again and lands.

    Prevents a close reported done while a grant stands."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.answers[_SESSION_STOP] = answer

    with pytest.raises(raised, match=r"close\(\) again") as failure:
        holder.close()
    assert type(failure.value) is raised
    assert holder.closed is True
    with pytest.raises(CoherenceError, match="closed"):
        holder.read(_X)

    wire.answers.pop(_SESSION_STOP)
    holder.close()
    assert holder.is_attached is False
    peer.write(_X, b"from the peer")


def test_an_unconfirmed_close_warns_in_degrade_mode(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Prevents a degrade volume raising, or keeping quiet, over a close."""
    holder = _holder(tmp_path, fast_cfg, on_error="degrade")
    holder.write(_X, b"from the holder")
    wire.answers[_SESSION_STOP] = {"ok": True, "degraded": True}

    with pytest.warns(CoherenceDegradedWarning):
        holder.close()
    assert holder.closed is True


def test_a_close_refused_at_the_local_endpoint_logs_and_does_not_raise(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When no coordinator listens at the volume's local endpoint, as after the
    volume's own coordinator shut down when idle, the connect is refused, the
    caller cannot release the grants, and the next coordinator's sweep takes
    them back: a strict close logs the paths and returns, so a block whose
    writes all landed does not fail at its exit.

    Prevents a strict exit raising over grants nothing can release."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    holder._endpoint = refused_endpoint(holder._endpoint)

    with caplog.at_level("WARNING", logger="ccs.adapters.coherent_volume"):
        holder.close()

    assert holder.closed is True
    assert any(_X in record.getMessage() for record in caplog.records)


def _connect_failed(reason: BaseException) -> Callable[[Callable[..., object], dict], object]:
    """A request whose connect failed for ``reason``, chained as the client
    chains it: never sent, from urllib's ``URLError``."""

    def fail(forward: Callable[..., object], payload: dict) -> object:
        raise CoordinatorUnavailable("scripted: connect failed", never_sent=True) from urllib.error.URLError(reason)

    return fail


@pytest.mark.parametrize(
    "reason",
    [
        pytest.param(TimeoutError("timed out"), id="connect-timed-out"),
        pytest.param(socket.gaierror("name did not resolve"), id="name-did-not-resolve"),
    ],
)
def test_a_close_whose_connect_failed_otherwise_raises_and_keeps_the_record(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    reason: BaseException,
) -> None:
    """A connect that timed out or a name that did not resolve may have a live
    coordinator behind it, still holding the grants, so a strict close raises
    and a second close() sends them again.

    Prevents a close that returns quietly while a live coordinator still
    refuses peers on the volume's files."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.answers[_SESSION_STOP] = _connect_failed(reason)

    with pytest.raises(CoordinatorUnreachable, match=r"close\(\) again"):
        holder.close()
    assert holder._records_write_grant(_X)

    wire.answers.pop(_SESSION_STOP)
    holder.close()
    peer.write(_X, b"from the peer")


def test_a_refused_close_of_a_remote_volume_raises(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote coordinator that refuses a connect may be back a moment later
    with the grants still held, so the exception for a local endpoint does not
    apply.

    Prevents a remote volume's close returning quietly over standing grants."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    local = resolve_endpoint(tmp_path)
    remote = resolve_remote_endpoint("127.0.0.1", local.port, local.bearer)
    holder = CoherentVolume(tmp_path, config=fast_cfg, remote_endpoint=remote)
    holder.write(_X, b"from the remote holder")
    holder._endpoint = refused_endpoint(holder._endpoint)

    with pytest.raises(CoordinatorUnreachable, match=r"close\(\) again"):
        holder.close()
    assert holder._records_write_grant(_X)


def test_a_grant_reported_held_before_a_refused_connect_is_still_reported(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A coordinator answered during this close that a grant was still held, so
    the refusal that follows does not mean nothing was listening: the close
    raises and names the held grant.

    Prevents a held grant reported only to a log because the next request
    found the coordinator gone."""
    holder = _holder(tmp_path, fast_cfg)
    earlier = _two_attempts_with_grants(holder, wire)
    path_of = {earlier: _X, holder.incarnation: _Z}
    refused = _connect_failed(ConnectionRefusedError(61, "Connection refused"))
    answered: list[str] = []

    def held_then_refused(forward: Callable[..., object], payload: dict) -> object:
        if answered:
            return refused(forward, payload)
        answered.append(payload["agent_id"])
        return _held_answer(path_of[payload["agent_id"]])

    wire.answers[_SESSION_STOP] = held_then_refused

    with pytest.raises(CoordinatorUnreachable, match="still held") as failure:
        holder.close()
    assert path_of[answered[0]] in str(failure.value)


def test_close_inside_an_operation_is_refused(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """Prevents a close from inside a write_cas closure ending the volume
    under the commit that is still in flight."""
    (tmp_path / _Z).write_bytes(b"z1")
    holder = _holder(tmp_path, fast_cfg)

    def close_then_append(current: bytes) -> bytes:
        holder.close()
        return current + b" + z2"

    with pytest.raises(CoherenceError, match="inside"):
        holder.write_cas(_Z, close_then_append)
    assert holder.closed is False


def test_a_with_block_closes_the_volume_on_exit(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """Prevents a with block leaving its grants to the sweep."""
    peer = _refuser(tmp_path, fast_cfg)
    with _holder(tmp_path, fast_cfg) as holder:
        holder.write(_X, b"from the holder")

    assert holder.closed is True
    peer.write(_X, b"from the peer")


def test_a_with_block_whose_body_raised_keeps_that_error_and_logs_the_close(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The body's error is the one the caller needs: a close that fails on the
    way out is logged instead of replacing it.

    Prevents a coordinator outage hiding the error that ended the block."""
    with caplog.at_level("WARNING", logger="ccs.adapters.coherent_volume"):
        with pytest.raises(ValueError, match="the body failed"):
            with _holder(tmp_path, fast_cfg) as holder:
                holder.write(_X, b"from the holder")
                wire.answers[_SESSION_STOP] = _no_answer
                raise ValueError("the body failed")

    assert holder.closed is True
    assert any("close" in record.getMessage() for record in caplog.records)
    with pytest.raises(CoherenceError, match="closed"):
        holder.read(_X)


def test_a_with_block_whose_body_returned_raises_an_unconfirmed_close(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Prevents a clean-looking exit while a grant may still stand."""
    with pytest.raises(CoordinatorUnreachable, match=r"close\(\) again"):
        with _holder(tmp_path, fast_cfg) as holder:
            holder.write(_X, b"from the holder")
            wire.answers[_SESSION_STOP] = _no_answer


def test_a_closed_volume_stays_closed_in_a_forked_child(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The fork handler re-mints a child's identity but leaves it closed.

    Prevents a forked child of a closed volume taking grants again."""
    holder = _holder(tmp_path, fast_cfg)
    holder.close()

    holder._after_fork()

    assert holder.closed is True
    with pytest.raises(CoherenceError, match="closed"):
        holder.read(_X)


# --- exits that close their volume -------------------------------------------


@pytest.fixture
def shim_removed() -> Iterator[None]:
    """Never let an installed open() shim outlive the test."""
    yield
    with contextlib.suppress(CoherenceError):
        uninstall()


def test_leaving_coherent_workspace_closes_its_volume(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    shim_removed: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The block owns its volume, so leaving it gives back the volume's write
    grants and ends the volume. A writer closed inside the block committed, so
    the exit does not name it.

    Prevents every file the block wrote staying held until the sweep."""
    peer = _refuser(tmp_path, fast_cfg)
    with caplog.at_level("WARNING", logger="ccs.adapters.coherent_volume"):
        with coherent_workspace(tmp_path, managed=_MANAGED, config=fast_cfg) as volume:
            with open(tmp_path / _X, "wb") as handle:
                handle.write(b"through the shim")

    assert not any("still open" in record.getMessage() for record in caplog.records)
    assert volume.closed is True
    assert open is _ORIGINAL_OPEN
    peer.write(_X, b"from the peer")
    assert write_holders(server, _X) == {peer.agent_id}
    with pytest.raises(CoherenceError, match="closed"):
        volume.read(_X)


def test_a_nested_coherent_workspace_closes_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, shim_removed: None
) -> None:
    """The outer block owns the volume; the inner one only yields it.

    Prevents an inner block ending the outer block's volume."""
    with coherent_workspace(tmp_path, managed=_MANAGED, config=fast_cfg) as outer:
        with coherent_workspace(tmp_path, managed=_MANAGED, config=fast_cfg) as inner:
            assert inner is outer
        assert outer.closed is False
        outer.write(_X, b"after the inner block")
    assert outer.closed is True


def test_a_coherent_workspace_whose_body_raised_keeps_that_error(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    shim_removed: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """open() is restored and the body's error propagates; a close that fails
    on the way out is logged.

    Prevents a coordinator outage replacing the error that ended the block."""
    with caplog.at_level("WARNING", logger="ccs.adapters.coherent_volume"):
        with pytest.raises(ValueError, match="the body failed"):
            with coherent_workspace(tmp_path, managed=_MANAGED, config=fast_cfg) as volume:
                volume.write(_X, b"from the block")
                wire.answers[_SESSION_STOP] = _no_answer
                raise ValueError("the body failed")

    assert open is _ORIGINAL_OPEN
    assert volume.closed is True
    assert any("close" in record.getMessage() for record in caplog.records)


def test_a_coherent_workspace_whose_body_returned_raises_an_unconfirmed_close(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire, shim_removed: None
) -> None:
    """A block that ends normally raises a close the coordinator did not
    confirm, as close() does, after open() is restored.

    Prevents a clean-looking exit while the block's grants may still stand."""
    with pytest.raises(CoordinatorUnreachable, match=r"close\(\) again"):
        with coherent_workspace(tmp_path, managed=_MANAGED, config=fast_cfg) as volume:
            volume.write(_X, b"from the block")
            wire.answers[_SESSION_STOP] = _no_answer

    assert open is _ORIGINAL_OPEN


def test_install_replaces_a_shim_volume_that_was_closed(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, shim_removed: None
) -> None:
    """A with block or close() on the volume install() returned ends that
    volume, so the next install() builds a new one, and open() goes through it.

    Prevents every managed open() in the process raising over a closed volume
    nothing replaces."""
    closed = install(tmp_path, managed=_MANAGED, config=fast_cfg)
    with closed:
        closed.write(_X, b"from the first volume")

    replacement = install(tmp_path, managed=_MANAGED, config=fast_cfg)

    assert replacement is not closed and replacement.closed is False
    with open(tmp_path / _X, "rb") as handle:
        assert handle.read() == b"from the first volume"


def test_a_coherent_workspace_over_a_closed_shim_volume_owns_its_replacement(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, shim_removed: None
) -> None:
    """Prevents a block that yields a closed volume and leaves the shim
    installed past its exit."""
    install(tmp_path, managed=_MANAGED, config=fast_cfg).close()

    with coherent_workspace(tmp_path, managed=_MANAGED, config=fast_cfg) as volume:
        assert volume.closed is False
        volume.write(_X, b"from the block")

    assert volume.closed is True
    assert open is _ORIGINAL_OPEN


def test_uninstall_restores_open_and_closes_the_volume(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, shim_removed: None
) -> None:
    """install() builds the volume and owns it, so uninstall() closes it.

    Prevents an install() volume that nobody ever closes."""
    peer = _refuser(tmp_path, fast_cfg)
    volume = install(tmp_path, managed=_MANAGED, config=fast_cfg)
    volume.write(_X, b"from the shim's volume")

    uninstall()

    assert open is _ORIGINAL_OPEN
    assert volume.closed is True
    peer.write(_X, b"from the peer")
    uninstall()  # nothing installed: does nothing


def test_a_strict_close_failure_from_uninstall_raises_after_open_is_restored(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire, shim_removed: None
) -> None:
    """Prevents a failed close leaving open() patched, and a second
    uninstall() that cannot retry: the retry is the volume's close()."""
    volume = install(tmp_path, managed=_MANAGED, config=fast_cfg)
    volume.write(_X, b"from the shim's volume")
    wire.answers[_SESSION_STOP] = _no_answer

    with pytest.raises(CoordinatorUnreachable, match=r"close\(\) again"):
        uninstall()

    assert open is _ORIGINAL_OPEN
    wire.answers.pop(_SESSION_STOP)
    volume.close()
    assert volume.is_attached is False


def test_a_file_left_open_for_writing_past_the_block_is_named_and_writes_nothing(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    shim_removed: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A shim writer commits through the volume when it closes, so one still
    open when the block closes the volume has nowhere to commit: the exit names
    it, and closing it afterwards raises and leaves the file as it was.

    Prevents bytes lost with no sign when a writer outlives its block."""
    (tmp_path / _X).write_bytes(b"before the block")
    with caplog.at_level("WARNING", logger="ccs.adapters.coherent_volume"):
        with coherent_workspace(tmp_path, managed=_MANAGED, config=fast_cfg):
            handle = open(tmp_path / _X, "wb")
            handle.write(b"never committed")

    assert any(_X in record.getMessage() for record in caplog.records)
    with pytest.raises(CoherenceError, match="closed"):
        handle.close()
    assert (tmp_path / _X).read_bytes() == b"before the block"


def test_a_detached_volume_releases_and_closes_without_a_request(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A forked child drops its parent's grant record with the endpoint, so a
    detached volume has nothing of its own to release.

    Prevents a detached volume sending a release for grants it does not own."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the parent")
    holder._after_fork()
    wire.sent.clear()

    holder.release(_X)
    holder.close()

    assert wire.sent == []


@pytest.mark.parametrize(
    ("answer", "text"),
    [
        pytest.param(
            {"ok": True, "degraded": True},
            "the coordinator did not confirm the release of this volume's write grant on data/x.txt, "
            "so the grant may still stand. Call release() again to send it again.",
            id="not-confirmed",
        ),
        pytest.param(
            _held_answer(_X),
            "the coordinator answered the release of this volume's write grant on data/x.txt with the "
            "grant still held. Call release() again to send it again.",
            id="held",
        ),
        pytest.param(
            _NO_ACCOUNT,
            "the coordinator did not confirm the release of this volume's write grant on data/x.txt, "
            "so the grant may still stand. Call release() again to send it again.",
            id="no-account-of-the-grant",
        ),
    ],
)
def test_an_unconfirmed_release_says_exactly_what_happened(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire,
    answer: dict, text: str,
) -> None:
    """The text is fixed but for the path, so a caller that retries on it sees
    the same words each time. Prevents the release text drifting unseen."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.answers[_POST_EDIT] = answer

    with pytest.raises(CoherenceError) as failure:
        holder.release(_X)

    assert str(failure.value) == text


def test_an_unconfirmed_close_says_exactly_what_is_still_recorded(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """Prevents the close text drifting, or listing its paths in another order."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_Z, b"z from the holder")
    holder.write(_X, b"x from the holder")
    wire.answers[_SESSION_STOP] = {"ok": True, "degraded": True}

    with pytest.raises(CoherenceError) as failure:
        holder.close()

    assert str(failure.value) == (
        "the coordinator did not confirm the release of every write grant this volume holds, so these "
        "may still stand: data/x.txt, data/z.txt. The volume is closed; call close() again to send what "
        "is still recorded."
    )
