# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""What the raw coordinator client says about a request that got no answer.

``CoordinatorUnavailable`` carries ``never_sent``: ``True`` only when the
connection never carried the request (urllib's ``URLError`` arm: a refused or
failed connect, a failed send). Every other failure — a timeout or a reset
after the request went out, a status line that is not HTTP, an empty or
non-JSON body — leaves it ``False``, so a caller deciding whether a commit may
have landed treats it as delivered. ``claim_caller_principal`` marks the
``unconfirmed`` claim it built from a ``CoordinatorUnavailable`` with
``transport_failed``; an answered claim that did not confirm leaves it unset.
``post()`` takes a per-call ``timeout``.

What these tests do not prove: one machine and loopback only; they pin the
client's marks, not what any surface raises from them. The hook client's
output on a claim transport failure is pinned by existing tests
(``test_session_start_unreachable_coordinator_returns_empty`` for a refused
connect, ``test_a_hook_whose_claim_answer_is_malformed_still_sends_its_request_without_a_principal``
for a malformed answer), which this change leaves as they are.
"""

from __future__ import annotations

import errno
import http.client
import http.server
import json
import pickle
import socket
import threading
import time
import urllib.error
from collections.abc import Callable, Iterator
from typing import Any

import pytest

from ccs.cli import _coherence_client as cc
from ccs.cli._coherence_client import (
    CoordinatorEndpoint,
    CoordinatorMalformedAnswer,
    CoordinatorUnavailable,
    PrincipalClaim,
)

_CLAIM_ROUTE = "/principal/claim"  # frozen duplicate of the wire route


def _refusing_port() -> int:
    """A loopback port that refuses connections and that no other test can take.

    Port 4, not a freed ephemeral port: another test worker's port-0 bind can
    take a freed port before the connect, and the client would then talk to a
    stranger. Ports below 1024 are never handed out for port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", 4)) == errno.ECONNREFUSED, "port 4 accepts connections here"
    return 4


# --- the opener seam: a real _execute meeting a scripted transport ------------


class _Answer:
    """A 2xx answer whose body is ``raw``, read the way ``_execute`` reads one."""

    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def __enter__(self) -> _Answer:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._raw


def _scripted_transport(
    monkeypatch: pytest.MonkeyPatch, outcome: Callable[[], object]
) -> None:
    """Every request ``_execute`` sends meets ``outcome()``: an exception to
    raise where urllib's opener would raise it, or an answer to return."""

    class _Opener:
        def open(self, _req: object, timeout: float | None = None) -> object:
            result = outcome()
            if isinstance(result, BaseException):
                raise result
            return result

    # Plain-http openers are cached process-wide; an empty cache makes the next
    # request build its opener through the patched seam, and the scripted
    # opener goes with the test instead of staying in the cache.
    monkeypatch.setattr(cc, "_shared_openers", {})
    monkeypatch.setattr(cc, "_build_opener", lambda _ctx: _Opener())


def _endpoint(port: int = 1) -> CoordinatorEndpoint:
    return CoordinatorEndpoint(port=port, bearer="test-bearer")


def _raised_by_post(endpoint: CoordinatorEndpoint, **kwargs: Any) -> CoordinatorUnavailable:
    with pytest.raises(CoordinatorUnavailable) as raised:
        cc.post(endpoint, "/hooks/pre-read", {"session_id": "s"}, **kwargs)
    return raised.value


def test_a_refused_connect_through_the_opener_is_marked_never_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """urllib wraps an OSError from the connect in ``URLError``: the
    coordinator never received the request, and the client says so."""
    _scripted_transport(
        monkeypatch, lambda: urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
    )

    exc = _raised_by_post(_endpoint())

    assert type(exc) is CoordinatorUnavailable
    assert exc.never_sent is True


@pytest.mark.parametrize(
    "failure",
    [
        lambda: TimeoutError("timed out"),
        lambda: ConnectionResetError(errno.ECONNRESET, "reset"),
        lambda: http.client.RemoteDisconnected("closed without response"),
        lambda: http.client.BadStatusLine("XHTTP/1.1 200\r\n"),
    ],
    ids=["timeout", "reset", "remote-disconnected", "bad-status-line"],
)
def test_a_failure_after_the_request_went_out_is_not_marked_never_sent(
    monkeypatch: pytest.MonkeyPatch, failure: Callable[[], BaseException],
) -> None:
    """urllib raises these from reading the answer, after the request was
    sent: only positive evidence counts as "not delivered", so the mark stays
    unset and a commit leg treats the request as one that may have landed."""
    _scripted_transport(monkeypatch, failure)

    exc = _raised_by_post(_endpoint())

    assert type(exc) is CoordinatorUnavailable
    assert exc.never_sent is False


@pytest.mark.parametrize("raw", [b"", b"not json"], ids=["empty", "non-json"])
def test_a_malformed_answer_is_not_marked_never_sent(
    monkeypatch: pytest.MonkeyPatch, raw: bytes,
) -> None:
    """A 2xx whose body is empty or does not decode came back from whatever
    the request reached: delivered, so the mark stays unset."""
    _scripted_transport(monkeypatch, lambda: _Answer(raw))

    exc = _raised_by_post(_endpoint())

    assert type(exc) is CoordinatorMalformedAnswer
    assert exc.never_sent is False


def test_a_real_refused_port_is_marked_never_sent() -> None:
    """The real opener against a loopback port that refuses: the mark is set
    from what urllib actually raises, not from a scripted stand-in."""
    exc = _raised_by_post(_endpoint(_refusing_port()))

    assert type(exc) is CoordinatorUnavailable
    assert exc.never_sent is True


# --- the exception stays what it was for every existing caller -----------------


def test_the_mark_defaults_off_and_leaves_the_exception_as_it_was() -> None:
    """Every existing ``CoordinatorUnavailable("msg")`` (and subclass) call
    keeps working and reads the mark unset; the message and repr are those of
    the plain exception, and a pickled one keeps both message and mark."""
    plain = CoordinatorUnavailable("could not reach coordinator")
    marked = CoordinatorUnavailable("could not reach coordinator", never_sent=True)

    assert plain.never_sent is False
    assert CoordinatorMalformedAnswer("empty").never_sent is False
    assert CoordinatorUnavailable().never_sent is False
    assert str(marked) == str(plain) == "could not reach coordinator"
    assert repr(marked) == repr(plain) == "CoordinatorUnavailable('could not reach coordinator')"
    assert marked.args == plain.args == ("could not reach coordinator",)
    # Bytes this test pickled itself: the trip an exception takes across a
    # process boundary must not drop the mark or the message.
    restored = pickle.loads(pickle.dumps(marked))
    assert type(restored) is CoordinatorUnavailable
    assert str(restored) == "could not reach coordinator"
    assert restored.never_sent is True
    assert pickle.loads(pickle.dumps(plain)).never_sent is False


# --- claim_caller_principal: a transport failure vs an answer that did not confirm


class _ClaimAnswers(http.server.BaseHTTPRequestHandler):
    """Answers every request with ``answer``: ``(status, headers, body)``."""

    answer: tuple[int, dict[str, str], bytes] = (200, {}, b"{}")

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        status, headers, raw = type(self).answer
        self.send_response(status)
        for name, value in {"Content-Type": "application/json", **headers}.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args: Any) -> None:
        return


@pytest.fixture
def claim_server() -> Iterator[int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ClaimAnswers)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()
        _ClaimAnswers.answer = (200, {}, b"{}")


def test_a_claim_whose_connect_is_refused_is_unconfirmed_and_transport_failed() -> None:
    claim = cc.claim_caller_principal(_endpoint(_refusing_port()), "session-a", "N" * 43)

    assert claim.outcome == "unconfirmed"
    assert claim.principal is None
    assert claim.transport_failed is True


@pytest.mark.parametrize(
    "answer",
    [
        (302, {"Location": "http://127.0.0.1:1/elsewhere"}, b"{}"),
        (503, {}, json.dumps({"ok": False, "reason": "claim_unconfirmed"}).encode()),
        (200, {}, json.dumps({"ok": False, "degraded": True, "reason": "claim_unconfirmed"}).encode()),
    ],
    ids=["redirect", "non-404-status", "2xx-claim-unconfirmed"],
)
def test_a_claim_answered_without_confirming_is_not_transport_failed(
    claim_server: int, answer: tuple[int, dict[str, str], bytes],
) -> None:
    """A redirect, a non-404 status and a 2xx that binds nothing came from a
    coordinator that was reached: each is ``unconfirmed`` as today, and none
    is marked as a transport failure."""
    _ClaimAnswers.answer = answer

    claim = cc.claim_caller_principal(_endpoint(claim_server), "session-a", "N" * 43)

    assert claim.outcome == "unconfirmed"
    assert claim.transport_failed is False


def test_a_claim_built_without_the_flag_reads_it_unset() -> None:
    """Every existing positional or keyword constructor keeps working and
    reads ``transport_failed`` false; the flag is keyword-only."""
    assert PrincipalClaim("unconfirmed", detail="x").transport_failed is False
    assert PrincipalClaim("bound", "P" * 43).transport_failed is False
    with pytest.raises(TypeError):
        PrincipalClaim("unconfirmed", None, "x", True)  # type: ignore[misc]


# --- post(timeout=...) -----------------------------------------------------------


@pytest.fixture
def silent_listener() -> Iterator[int]:
    """A loopback listener that completes the handshake (the kernel's backlog)
    but never reads or answers: the request goes out, no answer comes back."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(4)
        yield listener.getsockname()[1]


def test_a_per_call_timeout_bounds_the_wait_on_a_silent_listener(silent_listener: int) -> None:
    """``post(..., timeout=0.2)`` gives up near 0.2 s rather than after the
    module's 6 s default; the request was sent, so the mark stays unset."""
    assert cc.CLI_HTTP_TIMEOUT_SEC >= 5, "control: the default must be far above the per-call bound"
    started = time.monotonic()
    exc = _raised_by_post(_endpoint(silent_listener), timeout=0.2)
    elapsed = time.monotonic() - started

    assert 0.15 <= elapsed < 3.0
    assert exc.never_sent is False


def test_without_a_timeout_post_uses_the_module_default_read_at_call_time(
    silent_listener: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``timeout`` means ``CLI_HTTP_TIMEOUT_SEC`` as it is when the request
    is made, so a test or caller that patches it is still obeyed."""
    monkeypatch.setattr(cc, "CLI_HTTP_TIMEOUT_SEC", 0.3)
    started = time.monotonic()
    _raised_by_post(_endpoint(silent_listener))
    elapsed = time.monotonic() - started

    assert 0.25 <= elapsed < 3.0
