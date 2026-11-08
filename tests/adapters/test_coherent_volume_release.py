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

import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.adapters.effect_gate import gate
from ccs.cli._coherence_client import CoordinatorUnavailable
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

_X = "data/x.txt"
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


_UNCONFIRMED = [
    pytest.param({"ok": True, "degraded": True}, CoherenceError, id="degraded"),
    pytest.param({"ok": False, "reason": "internal: RuntimeError"}, CoordinatorFailed, id="envelope"),
    pytest.param(_status(503), CoordinatorFailed, id="http-503"),
    pytest.param(_no_answer, CoordinatorUnreachable, id="no-answer"),
    pytest.param(_not_an_object, CoordinatorUnreachable, id="not-an-object"),
    pytest.param(_held_answer(_X), CoherenceError, id="grant-still-held"),
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


def test_close_goes_on_past_a_grant_reported_held_and_then_raises(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The coordinator answered, so a grant it reports still held does not
    stop the close: the other attempt's grants are released, then the close
    raises, and a second close() sends what is still recorded.

    Prevents one held grant leaving every later grant to the sweep."""
    holder, peer = _holder(tmp_path, fast_cfg), _refuser(tmp_path, fast_cfg)
    earlier = _two_attempts_with_grants(holder, wire)

    def held_for_the_earlier_attempt(forward: Callable[..., object], payload: dict) -> object:
        return _held_answer(_X) if payload["agent_id"] == earlier else forward()

    wire.answers[_SESSION_STOP] = held_for_the_earlier_attempt

    with pytest.raises(CoherenceError, match=r"close\(\) again") as failure:
        holder.close()
    assert type(failure.value) is CoherenceError
    assert _X in str(failure.value)
    peer.write(_Z, b"z from the peer")

    wire.answers.pop(_SESSION_STOP)
    holder.close()
    peer.write(_X, b"x from the peer")
    assert write_holders(server, _X) == {peer.agent_id}


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


def _refused_connection(forward: Callable[..., object], payload: dict) -> object:
    raise CoordinatorUnavailable("scripted: connection refused", never_sent=True)


def test_a_close_that_reaches_no_coordinator_logs_and_does_not_raise(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When no coordinator is listening, as after the volume's own coordinator
    shut down when idle, the caller cannot release the grants and the next
    coordinator's sweep takes them back: a strict close logs the paths and
    returns, so a block whose writes all landed does not fail at its exit.

    Prevents a strict exit raising over grants nothing can release."""
    holder = _holder(tmp_path, fast_cfg)
    holder.write(_X, b"from the holder")
    wire.answers[_SESSION_STOP] = _refused_connection

    with caplog.at_level("WARNING", logger="ccs.adapters.coherent_volume"):
        holder.close()

    assert holder.closed is True
    assert any(_X in record.getMessage() for record in caplog.records)


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
