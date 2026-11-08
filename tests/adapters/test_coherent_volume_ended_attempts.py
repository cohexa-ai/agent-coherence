# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume`` ends the attempts it no longer uses (#525).

A volume starts a new attempt, a new coordinator identity, on every
compare-and-swap conflict, re-mint and ``reacquire()``. Each attempt that
reached the coordinator leaves records there: its per-file rows, its heartbeat,
its notices and its ``/status`` entry. The volume ends an attempt once it no
longer references it, with a session-stop carrying ``ended: true``, and the
coordinator then forgets it; at close it ends every attempt. These tests run
the real volume against a real coordinator in this process and script only the
answers a test needs the coordinator to give.
"""

from __future__ import annotations

import contextlib
import sqlite3
import urllib.error
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.cli._coherence_client as client_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import _GRANT_RELEASE_TIMEOUT_SEC, CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable, resolve_endpoint
from ccs.core.exceptions import CoherenceError, CoordinatorFailed
from tests.adapters.test_coherent_volume_other_holder import _status
from tests.coordinator_helpers import serve_in_process
from tests.loopback_helpers import refused_endpoint

_MANAGED = ("data/**",)
_X = "data/x.txt"
_Z = "data/z.txt"
_STOP = "/hooks/session-stop"
_CAS = "/hooks/post-edit-cas"
_LOGGER = "ccs.adapters.coherent_volume"
_CONFLICTS = 50
#: Rows and ``/status`` entries two volumes may keep however many conflicts
#: they had: each one's current attempt, the attempt holding its read, and an
#: attempt not yet ended because the latest re-mint came before its last read.
_A_HANDFUL = 6


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
    (tmp_path / _X).write_bytes(b"x1")
    (tmp_path / _Z).write_bytes(b"z1")
    running = serve_in_process(tmp_path, "ended-attempts")
    yield running
    running.shutdown()


@dataclass
class _Log:
    """Every POST the volumes send, with the answer each got (the exception,
    when it raised), and answers a test puts in place of the coordinator's.
    ``answers`` maps a route to a body, or to a callable ``(forward, payload)``
    that decides for itself."""

    sent: list[tuple[str, dict, object]] = field(default_factory=list)
    answers: dict[str, object] = field(default_factory=dict)

    def stops(self) -> list[dict]:
        return [body for route, body, _answer in self.sent if route == _STOP]

    def ends(self) -> list[str]:
        """The attempts each ended stop named, in the order sent."""
        return [body["agent_id"] for body in self.stops() if body.get("ended") is True]


@pytest.fixture
def log(monkeypatch: pytest.MonkeyPatch) -> _Log:
    """Every POST the volumes send, the principal claim included, with its
    answer, and answers a test can put in place."""
    recorded = _Log()
    real_post = client_module.post

    def answer(endpoint: object, path: str, payload: dict, kwargs: dict) -> object:
        scripted = recorded.answers.get(path)
        if callable(scripted):
            return scripted(lambda body=payload: real_post(endpoint, path, body, **kwargs), payload)
        if scripted is not None:
            return scripted
        return real_post(endpoint, path, payload, **kwargs)

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        try:
            got = answer(endpoint, path, payload, kwargs)
        except Exception as exc:
            recorded.sent.append((path, dict(payload), exc))
            raise
        recorded.sent.append((path, dict(payload), got))
        return got

    monkeypatch.setattr(client_module, "post", spy)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)
    return recorded


def _volume(root: Path, cfg: LifecycleConfig, **kwargs: object) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg, **kwargs)


def _named(body: dict) -> set[str]:
    """The attempts a request names: its own, and each handed-off grant's."""
    named = {body["agent_id"]} if isinstance(body.get("agent_id"), str) else set()
    for grant in body.get("grants", ()):
        if isinstance(grant, dict) and isinstance(grant.get("agent_id"), str):
            named.add(grant["agent_id"])
    return named


def _confirmed_end(route: str, body: dict, answer: object) -> bool:
    return route == _STOP and body.get("ended") is True and isinstance(answer, dict) and answer.get("ok") is True


def _sent_under_an_ended_attempt(log: _Log) -> list[tuple[str, dict]]:
    """Every request that names an attempt whose end the coordinator had
    already confirmed when it was sent."""
    ended: set[str] = set()
    late: list[tuple[str, dict]] = []
    for route, body, answer in log.sent:
        if _named(body) & ended:
            late.append((route, body))
        if _confirmed_end(route, body, answer):
            ended.add(body["agent_id"])
    assert ended, "control: the sequence ended no attempt, so the check proves nothing"
    return late


def _identity(vol: CoherentVolume, incarnation: str) -> str:
    return str(session_to_agent_id(vol.session_id, incarnation))


def _status_sessions(root: Path) -> list[str]:
    """The agent ids ``/status`` lists, read with the workspace's own endpoint,
    so a closed volume can be asked about."""
    status = client_module.get(resolve_endpoint(root), "/status")
    return [session["agent_id"] for session in status["sessions"]]


def _records(root: Path) -> dict[str, int]:
    """How many per-identity records the coordinator keeps, read from its own
    store, and how many sessions ``/status`` lists."""
    store = f"file:{root / '.coherence' / 'state.db'}?mode=ro"
    with contextlib.closing(sqlite3.connect(store, uri=True)) as conn:
        counts = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("agent_states", "heartbeats", "pending_notices")
        }
    counts["status_sessions"] = len(_status_sessions(root))
    return counts


def _numbered_incarnations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mint attempts in increasing order, so an earlier attempt sorts first and
    a walk that went by id alone would visit it first."""
    minted = iter(range(1, 1000))
    monkeypatch.setattr(CoherentVolume, "_new_incarnation", staticmethod(lambda: f"{next(minted):032x}"))


def _no_answer(forward: Callable[..., object], payload: dict) -> object:
    raise CoordinatorUnavailable("scripted: the answer was lost")


def _connect_failed(reason: BaseException) -> Callable[[Callable[..., object], dict], object]:
    """A request whose connect failed for ``reason``, chained as the client
    chains it: never sent, from urllib's ``URLError``."""

    def fail(forward: Callable[..., object], payload: dict) -> object:
        raise CoordinatorUnavailable("scripted: connect failed", never_sent=True) from urllib.error.URLError(reason)

    return fail


def _only_ends(answer: object) -> Callable[[Callable[..., object], dict], object]:
    """Answer an ended stop with ``answer``; send every other stop on."""

    def decide(forward: Callable[..., object], payload: dict) -> object:
        if payload.get("ended") is not True:
            return forward()
        return answer(forward, payload) if callable(answer) else answer

    return decide


def _answer_once(incarnation: str, answer: dict) -> Callable[[Callable[..., object], dict], object]:
    """Answer the first stop naming ``incarnation`` with ``answer``; send every
    other stop on, that attempt's later ones included."""
    answered: list[dict] = []

    def decide(forward: Callable[..., object], payload: dict) -> object:
        if answered or payload.get("agent_id") != incarnation:
            return forward()
        answered.append(payload)
        return answer

    return decide


# --- records stay bounded ------------------------------------------------------


def _lose_one_race(vol: CoherentVolume, peer: CoherentVolume) -> Callable[[bytes], bytes]:
    """``make_content`` for ``vol.write_cas``: on its first call the peer
    commits first, so the volume's compare-and-swap is refused once."""
    calls: list[bytes] = []

    def make(current: bytes) -> bytes:
        if not calls:
            peer.write_cas(_X, lambda theirs: theirs + b" +peer")
        calls.append(current)
        return current + b" +vol"

    return make


def _force_conflicts(vol: CoherentVolume, peer: CoherentVolume, count: int) -> None:
    for _ in range(count):
        vol.write_cas(_X, _lose_one_race(vol, peer))


def test_fifty_write_cas_conflicts_leave_a_handful_of_records(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """Two volumes force fifty compare-and-swap conflicts on one file. Every
    conflict and every stale read re-mints, yet the coordinator keeps rows,
    heartbeats, notices and ``/status`` entries for a handful of attempts, and
    no request names an attempt after its end.

    Prevents each conflict leaving its attempt in the coordinator for good,
    which grew ``/status`` and slowed every peer's pre-edit."""
    vol, peer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)

    _force_conflicts(vol, peer, _CONFLICTS)

    refused = [
        answer for route, body, answer in log.sent
        if route == _CAS and body["session_id"] == vol.session_id and answer.get("ok") is False
    ]
    assert len(refused) == _CONFLICTS, "control: each round's first compare-and-swap was refused"
    records = _records(tmp_path)
    assert all(count <= _A_HANDFUL for count in records.values()), records
    assert _sent_under_an_ended_attempt(log) == []


def test_an_attempt_a_transfer_still_presents_is_never_ended(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """The volume reads one file, then compare-and-swaps another three times,
    and each re-mint ends what the volume no longer uses. The attempt that took
    the first read is the one a transfer of that file presents, so no end names
    it and the transfer hands the read on.

    Prevents an end that forgets the claim a transfer must present, which
    would refuse the transfer while the volume holds the read."""
    giver, successor = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)
    giver.read(_Z)
    reader = giver.incarnation
    for round_ in range(3):
        _data, version = giver.read_with_version(_X)
        giver.write_cas_at(_X, version, b"x %d" % round_)
    assert log.ends(), "control: the re-mints ended the attempts nothing references"

    result = giver.transfer(_Z, successor=str(session_to_agent_id(successor.session_id)))

    assert result.ok is True
    assert reader not in log.ends()
    assert _sent_under_an_ended_attempt(log) == []


# --- what close() ends ---------------------------------------------------------


def test_closing_a_read_only_volume_ends_its_attempt_and_the_records_go(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """A volume that only read closes with one session-stop, which ends its
    attempt: the coordinator then keeps no row, heartbeat or ``/status`` entry
    for it, and the volume detaches.

    Prevents a closed volume's attempts staying in the coordinator for good."""
    vol = _volume(tmp_path, fast_cfg)
    vol.read(_X)
    attempt = vol.agent_id
    log.sent.clear()

    vol.close()

    assert [(route, body) for route, body, _answer in log.sent] == [
        (_STOP, {"session_id": vol.session_id, "agent_id": vol.incarnation, "ended": True}),
    ]
    assert vol.is_attached is False
    assert attempt not in _status_sessions(tmp_path)
    assert _records(tmp_path) == {"agent_states": 0, "heartbeats": 0, "pending_notices": 0, "status_sessions": 0}


def test_close_ends_every_attempt_that_reached_the_coordinator(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """A volume whose attempts read, wrote and compare-and-swapped closes: each
    attempt a request named is ended once, and the coordinator keeps nothing
    for any of them. The attempt minted by the last re-mint never sent a
    request, so nothing is spent on it.

    Prevents a close that ends only the attempts holding a write grant."""
    vol = _volume(tmp_path, fast_cfg)
    vol.read(_Z)
    vol.reacquire(_X)
    vol.write(_X, b"x2")
    with pytest.raises(FileNotFoundError):
        vol.write_cas_at("data/missing.txt", 0, b"never written")
    named = {body["agent_id"] for _route, body, _answer in log.sent if "agent_id" in body}
    unsent = vol.incarnation
    log.sent.clear()

    vol.close()

    assert sorted(log.ends()) == sorted(named)
    assert unsent not in named
    assert _records(tmp_path) == {"agent_states": 0, "heartbeats": 0, "pending_notices": 0, "status_sessions": 0}
    assert vol.is_attached is False


def test_a_grant_carrying_stop_goes_before_any_end_at_close(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    log: _Log,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The earlier attempt took only a read; the later one holds a write grant.
    Attempts are minted in sorting order here, so a walk by id would end the
    earlier one first; the close releases the grant first.

    Prevents a close spending its one timeout on an attempt that blocks no
    one while a grant still refuses every peer."""
    _numbered_incarnations(monkeypatch)
    vol = _volume(tmp_path, fast_cfg)
    vol.read(_Z)
    reader = vol.incarnation
    vol.reacquire(_X)
    vol.write(_X, b"x2")
    writer = vol.incarnation
    log.sent.clear()

    vol.close()

    assert [(body["agent_id"], body.get("ended")) for body in log.stops()] == [(writer, True), (reader, True)]


def test_a_close_told_a_grant_is_held_ends_only_the_attempts_that_hold_none(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    log: _Log,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The close's stop of the attempt that wrote is answered with its grant
    still held, so the close raises, and that attempt, which still records the
    grant, is sent no end of its own. The coordinator answered, so the attempt
    that only read is ended and leaves ``/status``. A second close() releases
    the grant and detaches, and no request names an attempt after its end.

    Prevents a raw end of an attempt whose release the close reports as not
    confirmed, and a close that leaves the attempts holding no grant in the
    coordinator because another attempt's grant was reported held."""
    _numbered_incarnations(monkeypatch)
    vol = _volume(tmp_path, fast_cfg)
    vol.read(_Z)
    reader = vol.incarnation
    vol.reacquire(_X)
    vol.write(_X, b"x2")
    writer = vol.incarnation
    log.answers[_STOP] = _answer_once(writer, {"ok": False, "grants": [{"path": _X, "held": True}]})
    log.sent.clear()

    with pytest.raises(CoherenceError, match="still held"):
        vol.close()

    assert [(body["agent_id"], body.get("ended")) for body in log.stops()] == [(writer, True), (reader, True)]
    assert _identity(vol, reader) not in _status_sessions(tmp_path)

    log.answers.pop(_STOP)
    vol.close()
    assert _sent_under_an_ended_attempt(log) == []
    assert vol.is_attached is False


def test_a_grant_carrying_stop_goes_before_any_end_at_a_re_mint(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    log: _Log,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At one re-mint the volume no longer uses two attempts: one that read,
    and one whose write grant an earlier re-mint failed to release. The grant
    is released, with the end, before the other attempt is ended, although
    attempts are minted in sorting order here.

    Prevents a re-mint ending idle attempts while its own grant still refuses
    the commit it is about to make."""
    _numbered_incarnations(monkeypatch)
    vol = _volume(tmp_path, fast_cfg)
    vol.read(_Z)
    reader = vol.incarnation
    with pytest.raises(FileNotFoundError):
        vol.write_cas_at("data/missing.txt", 0, b"never written")
    vol.write(_X, b"x2")
    writer = vol.incarnation
    log.answers[_STOP] = _status(503)
    with pytest.raises(CoordinatorFailed):
        vol.reacquire(_Z)
    log.answers.pop(_STOP)
    vol.read(_Z)
    log.sent.clear()

    vol.reacquire(_Z)

    assert [(body["agent_id"], body.get("ended")) for body in log.stops()] == [(writer, True), (reader, True)]
    assert {_identity(vol, writer), _identity(vol, reader)}.isdisjoint(_status_sessions(tmp_path))


# --- an end that does not land ---------------------------------------------------

_END_UNCONFIRMED = [
    pytest.param(_no_answer, id="no-answer"),
    pytest.param(_connect_failed(TimeoutError("timed out")), id="connect-timed-out"),
    pytest.param(_status(503), id="http-503"),
    pytest.param({"ok": True, "degraded": True}, id="degraded"),
    pytest.param({"ok": False, "reason": "internal: RuntimeError"}, id="envelope"),
    pytest.param({"ok": False, "grants": [{"path": _X, "held": True}]}, id="grant-held"),
]
#: A re-mint keeps an end refused at the local endpoint too; only close()
#: leaves it (see the test of a close refused at the local endpoint).
_END_UNCONFIRMED_AT_A_RE_MINT = [
    *_END_UNCONFIRMED,
    pytest.param(_connect_failed(ConnectionRefusedError(61, "Connection refused")), id="connect-refused"),
]


def _an_unreferenced_attempt(vol: CoherentVolume) -> str:
    """Leave ``vol`` with an attempt it no longer references: it read ``_X``,
    and a later attempt read it again. Returns that attempt."""
    vol.read(_X)
    earlier = vol.incarnation
    vol.reacquire(_X)
    return earlier


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize("answer", _END_UNCONFIRMED_AT_A_RE_MINT)
def test_an_unconfirmed_end_at_a_re_mint_is_logged_and_sent_again_at_the_next(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    log: _Log,
    caplog: pytest.LogCaptureFixture,
    on_error: str,
    answer: object,
) -> None:
    """An end the coordinator does not confirm is logged, never raised or
    counted as a degradation, and the attempt stays to be ended at the next
    re-mint, which ends it.

    Prevents an end that fails a re-mint over records that block no one, and
    one dropped after a failure, which would leave the records for good."""
    vol = _volume(tmp_path, fast_cfg, on_error=on_error)
    earlier = _an_unreferenced_attempt(vol)
    log.answers[_STOP] = _only_ends(answer)

    with caplog.at_level("WARNING", logger=_LOGGER):
        assert vol.reacquire(_X) == b"x1"

    assert log.ends() == [earlier]
    assert any("could not confirm the end" in record.getMessage() for record in caplog.records)
    assert (vol.degradation_count, vol.is_degraded) == (0, False)
    assert _identity(vol, earlier) in _status_sessions(tmp_path)

    log.answers.pop(_STOP)
    vol.reacquire(_X)
    assert log.ends().count(earlier) == 2
    assert _identity(vol, earlier) not in _status_sessions(tmp_path)


@pytest.mark.parametrize("answer", _END_UNCONFIRMED)
def test_an_unconfirmed_end_at_close_is_logged_and_sent_again_by_the_next_close(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    log: _Log,
    caplog: pytest.LogCaptureFixture,
    answer: object,
) -> None:
    """A strict close whose only unconfirmed request is the end of an attempt
    holding no grant returns, logs it, and stays closed with its endpoint, so a
    second close() sends the end again and then detaches. A connect that timed
    out may have a live coordinator behind it, so it is no exception (a refused
    connect at the local endpoint is: see the next test).

    Prevents a clean exit raising over records that block no one, and a close
    that forgets an attempt it could not end."""
    vol = _volume(tmp_path, fast_cfg)
    vol.read(_X)
    attempt = vol.incarnation
    log.answers[_STOP] = _only_ends(answer)

    with caplog.at_level("WARNING", logger=_LOGGER):
        vol.close()

    assert vol.closed is True
    assert vol.is_attached is True
    assert any("could not confirm the end" in record.getMessage() for record in caplog.records)
    assert (vol.degradation_count, vol.is_degraded) == (0, False)
    with pytest.raises(CoherenceError, match="closed"):
        vol.read(_X)

    log.answers.pop(_STOP)
    vol.close()
    assert log.ends() == [attempt, attempt]
    assert vol.is_attached is False
    assert _identity(vol, attempt) not in _status_sessions(tmp_path)


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_a_close_refused_at_the_local_endpoint_logs_the_attempts_once_and_detaches(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    log: _Log,
    caplog: pytest.LogCaptureFixture,
    on_error: str,
) -> None:
    """When no coordinator listens at the volume's local endpoint, as after its
    own coordinator shut down when idle, the connect of the first end is
    refused and nothing the caller does can end the attempts. The close logs
    them once, never raises or counts it, and detaches, so a second close()
    sends nothing.

    Prevents the close of a volume that only read staying attached and sending
    its ends again at every close() to a coordinator that is gone."""
    vol = _volume(tmp_path, fast_cfg, on_error=on_error)
    vol.read(_Z)
    reader = vol.incarnation
    vol.reacquire(_X)
    current = vol.incarnation
    vol._endpoint = refused_endpoint(vol._endpoint)
    log.sent.clear()

    with caplog.at_level("WARNING", logger=_LOGGER):
        vol.close()

    assert len(log.ends()) == 1, "the ends went on past a refused connect"
    messages = [record.getMessage() for record in caplog.records]
    left = [message for message in messages if "were not ended" in message]
    assert len(left) == 1 and reader in left[0] and current in left[0], messages
    assert not any("could not confirm the end" in message for message in messages)
    assert (vol.degradation_count, vol.is_degraded) == (0, False)
    assert vol.is_attached is False
    assert _identity(vol, reader) in _status_sessions(tmp_path), "control: no end reached the coordinator"

    log.sent.clear()
    vol.close()
    assert log.sent == []


def test_a_re_mint_whose_release_is_held_ends_nothing_else(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """A release the coordinator answers as still held stops the re-mint's
    pass, as before, and the attempts that hold no grant wait for the next
    re-mint rather than costing more requests now.

    Prevents a re-mint sending ends after the coordinator refused its
    release."""
    vol = _volume(tmp_path, fast_cfg)
    earlier = _an_unreferenced_attempt(vol)
    vol.write(_Z, b"z2")
    writer = vol.incarnation
    log.answers[_STOP] = {"ok": False, "reason": "release_failed", "grants": [{"path": _Z, "held": True}]}
    log.sent.clear()

    vol.reacquire(_X)

    assert [body["agent_id"] for body in log.stops()] == [writer]
    log.answers.pop(_STOP)
    log.sent.clear()
    vol.reacquire(_X)
    assert sorted(log.ends()) == sorted([writer, earlier])


def test_an_end_with_no_answer_stops_the_ends_after_it(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """Two attempts wait to be ended after earlier ends went unanswered. The
    first end that gets no answer stops the walk, so a coordinator that
    stopped answering costs one short timeout per re-mint, not one per
    attempt.

    Prevents a re-mint that waits out a timeout for every attempt it ends."""
    vol = _volume(tmp_path, fast_cfg)
    _an_unreferenced_attempt(vol)
    log.answers[_STOP] = _only_ends(_no_answer)
    vol.reacquire(_X)
    log.sent.clear()

    vol.reacquire(_X)

    assert len(log.ends()) == 1


def test_an_end_waits_no_longer_than_a_best_effort_grant_release(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    log: _Log,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each end of an attempt holding no grant, at a re-mint and at close, is
    bounded by the short timeout of write()'s best-effort grant release, so a
    coordinator that stopped answering costs a strict re-mint, or the close of
    a volume that only read, that much rather than a full client timeout.

    Prevents an end that holds up every re-mint for the client's whole timeout
    over records that block no one."""
    timeouts: list[object] = []
    spy = coherent_volume_module._coordinator_post

    def timed(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if payload.get("ended") is True:
            timeouts.append(kwargs.get("timeout"))
        return spy(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", timed)
    vol = _volume(tmp_path, fast_cfg)
    _an_unreferenced_attempt(vol)

    vol.reacquire(_X)
    vol.close()

    assert len(timeouts) == 3, "control: one end at the re-mint and two at close"
    assert set(timeouts) == {_GRANT_RELEASE_TIMEOUT_SEC}


def test_a_close_whose_release_got_no_answer_sends_no_end(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """The close's release of a write grant gets no answer, so it raises, and
    the attempt that only read is not sent its end: it would meet the same
    silent coordinator. A second close() sends both.

    Prevents a close against a silent coordinator costing a timeout more for
    an attempt that blocks no one."""
    vol = _volume(tmp_path, fast_cfg)
    vol.read(_Z)
    reader = vol.incarnation
    vol.reacquire(_X)
    vol.write(_X, b"x2")
    writer = vol.incarnation
    log.answers[_STOP] = _no_answer
    log.sent.clear()

    with pytest.raises(CoherenceError, match=r"close\(\) again"):
        vol.close()

    assert [body["agent_id"] for body in log.stops()] == [writer]
    log.answers.pop(_STOP)
    vol.close()
    assert log.ends() == [writer, writer, reader]
    assert vol.is_attached is False


def test_a_close_stop_reporting_another_grant_held_does_not_count_as_the_end(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """The close's stop of the attempt that wrote is answered grant by grant:
    the path the volume recorded for it was released, so its record empties
    and the close raises nothing, but another grant is reported held, and the
    coordinator keeps every record of an attempt holding a grant. The attempt
    stays to be ended, so the end after the releases is sent for it, and it
    leaves ``/status``.

    Prevents a close that takes an emptied grant record for a confirmed end,
    which would leave the attempt in the coordinator for good."""
    vol = _volume(tmp_path, fast_cfg)
    vol.write(_X, b"x2")
    writer = vol.incarnation
    released_but_another_held = {"ok": False, "grants": [{"path": _X, "held": False}, {"path": _Z, "held": True}]}
    log.answers[_STOP] = _answer_once(writer, released_but_another_held)
    log.sent.clear()

    vol.close()

    assert log.ends() == [writer, writer]
    assert _identity(vol, writer) not in _status_sessions(tmp_path)
    assert vol.is_attached is False


# --- a forked child ------------------------------------------------------------


def test_a_forked_child_ends_none_of_the_parents_attempts(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """The parent's attempts are the parent's to end: after the fork handler
    the child re-mints and closes, and ends only attempts of its own session.

    Prevents a child forgetting the identity its still-running parent sends
    under, whose stale copy a strict pre-edit would then admit."""
    vol = _volume(tmp_path, fast_cfg)
    parent_session = vol.session_id
    earlier = _an_unreferenced_attempt(vol)
    current = vol.incarnation
    log.sent.clear()

    vol._after_fork()
    vol.reacquire(_X)
    vol.close()

    assert log.ends(), "control: the child ended its own attempts"
    assert {body["session_id"] for body in log.stops()} == {vol.session_id}
    assert not {earlier, current} & {attempt for body in log.stops() for attempt in _named(body)}
    listed = _status_sessions(tmp_path)
    assert {str(session_to_agent_id(parent_session, inc)) for inc in (earlier, current)} <= set(listed)


# --- the invariant over a mixed run ----------------------------------------------


def test_no_request_is_sent_under_an_ended_attempt(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, log: _Log
) -> None:
    """Across reads, writes, a release, re-mints, a compare-and-swap that
    loses its race, a transfer and a close, no request names an attempt after
    the coordinator confirmed its end.

    Prevents a request under a forgotten identity, which a strict pre-edit
    would admit where its stale copy was denied."""
    vol, peer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)
    vol.read(_X)
    vol.write(_X, b"x2")
    vol.reacquire(_Z)
    vol.read(_X)
    vol.write(_X, b"x3")
    vol.release(_X)
    peer.write(_Z, b"z from the peer")
    peer.release(_Z)
    vol.write_cas(_Z, lambda current: current + b" +vol")
    vol.write_cas(_X, _lose_one_race(vol, peer))
    _data, version = vol.read_with_version(_Z)
    vol.write_cas_at(_Z, version, b"z3")
    vol.transfer(_Z, successor=str(session_to_agent_id(peer.session_id)))
    vol.reacquire(_X)
    vol.close()
    peer.close()

    assert _sent_under_an_ended_attempt(log) == []
    assert _records(tmp_path) == {"agent_states": 0, "heartbeats": 0, "pending_notices": 0, "status_sessions": 0}
