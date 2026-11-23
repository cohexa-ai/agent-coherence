"""The urllib boundary behind ``CoordinatorUnavailable.never_sent``, through the
real opener and real loopback sockets (#382).

``never_sent`` is set only from urllib's ``URLError`` arm, which CPython's
``AbstractHTTPHandler.do_open`` raises only around ``h.request`` (connect and
send). These tests drive a real ``post()`` against listeners that fail at each
point after the request went out (before the status line, at it, and partway
through the body), and one that fails before, so a future urllib that wrapped
more of ``do_open``, or the reading of the answer, would turn a test red rather
than leave a delivered commit reported as never sent.

What these do not prove: one machine, loopback, CPython's urllib as installed.
"""

from __future__ import annotations

import select
import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from ccs.cli import _coherence_client as cc
from ccs.cli._coherence_client import CoordinatorEndpoint, CoordinatorUnavailable
from ccs.core.exceptions import CallerPrincipalRefused
from tests.loopback_helpers import refusing_port
from tests.pid_lock_helpers import release_coordinator_lock, write_coordinator_pid_file

_READ_TIMEOUT_SEC = 2.0
# An answer cut short: the status line and headers, then only the start of the
# body the headers promise.
_HEADERS_AND_PART_OF_THE_BODY = (
    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n"
    b'{"ok"'
)


def _read_request(conn: socket.socket) -> bytes:
    """Read one whole request: the headers, then Content-Length bytes."""
    conn.settimeout(_READ_TIMEOUT_SEC)
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = conn.recv(65536)
        if not chunk:
            return data
        data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    length = 0
    for line in head.split(b"\r\n")[1:]:
        name, _, value = line.partition(b":")
        if name.strip().lower() == b"content-length":
            length = int(value.strip())
    while len(body) < length:
        chunk = conn.recv(65536)
        if not chunk:
            break
        body += chunk
    return head + b"\r\n\r\n" + body


def _reset(conn: socket.socket) -> None:
    conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    conn.close()


@pytest.fixture
def one_shot() -> Iterator[Callable[[Callable[[socket.socket], None]], tuple[int, dict]]]:
    """A loopback listener that accepts one connection and hands it to ``act``;
    returns the port and a dict where the listener records what it read."""
    listeners: list[socket.socket] = []
    threads: list[threading.Thread] = []
    stop = threading.Event()

    def start(act: Callable[[socket.socket], None]) -> tuple[int, dict]:
        seen: dict = {}
        lsock = socket.socket()
        lsock.bind(("127.0.0.1", 0))
        lsock.listen(1)
        # close() from another thread does not wake a blocked accept() on Linux,
        # so a test that fails before connecting would strand this thread: poll
        # with a short timeout and stop on a flag, as proxy_recorder does.
        lsock.settimeout(0.05)
        listeners.append(lsock)

        def serve() -> None:
            while not stop.is_set():
                try:
                    conn, _ = lsock.accept()
                except TimeoutError:  # the poll interval; any other error surfaces
                    continue
                seen["request"] = _read_request(conn)
                act(conn)
                return

        t = threading.Thread(target=serve, daemon=True)
        t.start()
        threads.append(t)
        return lsock.getsockname()[1], seen

    yield start
    stop.set()
    for t in threads:
        t.join(timeout=5)
    for lsock in listeners:
        lsock.close()


def _post(port: int) -> CoordinatorUnavailable:
    endpoint = CoordinatorEndpoint(port=port, bearer="test-bearer")
    with pytest.raises(CoordinatorUnavailable) as raised:
        cc.post(endpoint, "/hooks/post-edit-cas", {"session_id": "s", "path": "p"}, timeout=3.0)
    return raised.value


def _close(conn: socket.socket) -> None:
    conn.close()


def _garbage(conn: socket.socket) -> None:
    conn.sendall(b"XHTTP/9 nope\r\n\r\n")
    conn.close()


def _headers_then_reset(conn: socket.socket) -> None:
    conn.sendall(_HEADERS_AND_PART_OF_THE_BODY)
    _reset(conn)


def _headers_then_close(conn: socket.socket) -> None:
    conn.sendall(_HEADERS_AND_PART_OF_THE_BODY)
    conn.close()


@pytest.mark.parametrize(
    ("act", "cause"),
    [
        (_reset, ConnectionResetError),
        (_close, ConnectionError),  # http.client.RemoteDisconnected
        # A malformed answer (BadStatusLine, IncompleteRead) is raised with no cause.
        (_garbage, None),
        (_headers_then_reset, ConnectionResetError),
        (_headers_then_close, None),
    ],
    ids=[
        "read-then-reset",
        "read-then-close",
        "read-then-garbage-status-line",
        "answer-headers-then-reset-mid-body",
        "answer-headers-then-close-mid-body",
    ],
)
def test_a_request_the_listener_read_whole_is_not_marked_never_sent(
    one_shot: Callable, act: Callable[[socket.socket], None], cause: type[BaseException] | None,
) -> None:
    """The listener read the whole request before failing it, so it could have
    acted on it: the client must not mark it never sent."""
    port, seen = one_shot(act)

    exc = _post(port)

    assert seen["request"].startswith(b"POST /hooks/post-edit-cas "), seen
    assert seen["request"].endswith(b'"path": "p"}'), "control: the listener read the whole body"
    assert exc.never_sent is False
    if cause is None:
        assert exc.__cause__ is None
    else:
        assert isinstance(exc.__cause__, cause)


def test_a_request_answered_by_no_one_after_it_went_out_is_not_marked_never_sent(
    one_shot: Callable,
) -> None:
    """Read whole, then held open with no answer until the client's timeout:
    the shape a stopped or wedged coordinator has, delivered."""
    hold = threading.Event()
    port, seen = one_shot(lambda conn: (hold.wait(5), conn.close()))
    started = time.monotonic()

    try:
        exc = _post(port)
    finally:
        hold.set()

    assert 2.5 <= time.monotonic() - started < 5.0
    assert seen["request"].endswith(b'"path": "p"}')
    assert exc.never_sent is False


def test_a_refused_connect_through_the_real_opener_is_marked_never_sent() -> None:
    """The control on the other side of the line: nothing listens, the connect
    is refused inside ``h.request``, and urllib wraps it in ``URLError``."""
    exc = _post(refusing_port())

    assert exc.never_sent is True
    assert isinstance(exc.__cause__, cc.urllib.error.URLError)
    assert isinstance(exc.__cause__.reason, ConnectionRefusedError)


# ----------------------------------------------------------------------
# Naming how a send failed: classify_send_failure, fed the exceptions real
# sockets produce. The names are a published contract, so they are written out
# here rather than imported.
# ----------------------------------------------------------------------


def _send_failure(port: int, timeout: float = 3.0) -> BaseException:
    """What a real ``post()`` to ``port`` raises."""
    endpoint = CoordinatorEndpoint(port=port, bearer="test-bearer")
    with pytest.raises(Exception) as raised:
        cc.post(endpoint, "/hooks/pre-edit", {"session_id": "s", "path": "p"}, timeout=timeout)
    return raised.value


def _answer(status_line: bytes, body: bytes = b"", headers: bytes = b"") -> Callable[[socket.socket], None]:
    """An act that answers the request with ``status_line`` and ``body``, framed by Content-Length."""

    def act(conn: socket.socket) -> None:
        head = b"%s\r\nContent-Type: application/json\r\nContent-Length: %d\r\n%sConnection: close\r\n\r\n"
        conn.sendall(head % (status_line, len(body), headers) + body)
        conn.close()

    return act


def test_a_refused_connect_is_named_no_listener() -> None:
    assert cc.classify_send_failure(_send_failure(refusing_port())) == "no_listener"


def test_a_full_accept_queue_is_named_transport_unanswered_though_nothing_was_sent() -> None:
    """A coordinator too busy to accept looks like this from outside: the
    connect times out. The exception says the request was never sent, and the
    name must still say nobody answered, not that nobody was listening."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)  # never accepted
        held = _fill_accept_queue(listener.getsockname())
        try:
            exc = _send_failure(listener.getsockname()[1], timeout=1.0)
        finally:
            for sock in held:
                sock.close()

    assert isinstance(exc, CoordinatorUnavailable) and exc.never_sent is True
    assert cc.classify_send_failure(exc) == "transport_unanswered"


def _fill_accept_queue(address: tuple[str, int]) -> list[socket.socket]:
    """Open connections until one cannot complete: the accept queue is full.
    Kernels queue a different number past the backlog, so count, don't assume."""
    held: list[socket.socket] = []
    for _ in range(64):
        probe = socket.socket()
        probe.setblocking(False)
        probe.connect_ex(address)
        _, writable, _ = select.select([], [probe], [], 0.3)
        if not writable or probe.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR) != 0:
            probe.close()
            return held
        held.append(probe)
    raise AssertionError("the accept queue never filled")


def _reset_on_accept_port() -> int:
    """A listener that resets its one connection as soon as it accepts it, unread."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve() -> None:
        with listener:
            listener.settimeout(5)
            conn, _ = listener.accept()
            _reset(conn)

    threading.Thread(target=serve, daemon=True).start()
    return listener.getsockname()[1]


def test_a_connection_reset_on_accept_is_named_transport_unanswered() -> None:
    assert cc.classify_send_failure(_send_failure(_reset_on_accept_port())) == "transport_unanswered"


@pytest.mark.parametrize(
    "act",
    [_reset, _close, _garbage, _headers_then_reset, _headers_then_close],
    ids=[
        "read-then-reset",
        "read-then-close",
        "read-then-garbage-status-line",
        "answer-headers-then-reset-mid-body",
        "answer-headers-then-close-mid-body",
    ],
)
def test_a_request_read_and_never_answered_is_named_transport_unanswered(
    one_shot: Callable, act: Callable[[socket.socket], None],
) -> None:
    port, _seen = one_shot(act)
    assert cc.classify_send_failure(_send_failure(port)) == "transport_unanswered"


def test_a_request_read_and_held_past_the_timeout_is_named_transport_unanswered(one_shot: Callable) -> None:
    hold = threading.Event()
    port, _seen = one_shot(lambda conn: (hold.wait(5), conn.close()))
    try:
        exc = _send_failure(port, timeout=1.0)
    finally:
        hold.set()
    assert cc.classify_send_failure(exc) == "transport_unanswered"


@pytest.mark.parametrize(
    ("status_line", "body"),
    [
        (b"HTTP/1.1 200 OK", b""),
        (b"HTTP/1.1 204 No Content", b""),
        (b"HTTP/1.1 200 OK", b"not json"),
        # Not UTF-8: the decode error's cause carries a string ``reason``.
        (b"HTTP/1.1 200 OK", b"\xff\xfe{}"),
    ],
    ids=["empty-200", "204", "non-json", "not-utf-8"],
)
def test_a_2xx_that_is_not_a_json_answer_is_named_malformed_answer(
    one_shot: Callable, status_line: bytes, body: bytes,
) -> None:
    port, _seen = one_shot(_answer(status_line, body))
    assert cc.classify_send_failure(_send_failure(port)) == "malformed_answer"


@pytest.mark.parametrize(
    "status_line",
    [b"HTTP/1.1 500 Internal Server Error", b"HTTP/1.1 503 Service Unavailable", b"HTTP/1.1 599 Unknown"],
    ids=["500", "503", "599"],
)
def test_a_5xx_is_named_http_unanswered(one_shot: Callable, status_line: bytes) -> None:
    port, _seen = one_shot(_answer(status_line, b'{"error": "x"}'))
    assert cc.classify_send_failure(_send_failure(port)) == "http_unanswered"


@pytest.mark.parametrize(
    ("status_line", "headers"),
    [
        (b"HTTP/1.1 400 Bad Request", b""),
        (b"HTTP/1.1 401 Unauthorized", b""),
        (b"HTTP/1.1 404 Not Found", b""),
        (b"HTTP/1.1 499 Unknown", b""),
        (b"HTTP/1.1 302 Found", b"Location: http://127.0.0.1:4/elsewhere\r\n"),
    ],
    ids=["400", "401", "404", "499", "302-redirect"],
)
def test_a_4xx_or_a_redirect_is_named_http_rejected(one_shot: Callable, status_line: bytes, headers: bytes) -> None:
    port, _seen = one_shot(_answer(status_line, b'{"error": "x"}', headers))
    assert cc.classify_send_failure(_send_failure(port)) == "http_rejected"


def test_a_coordinator_that_left_before_the_send_is_named_departed(tmp_path) -> None:
    coherence = tmp_path / ".coherence"
    coherence.mkdir()
    pid_file = write_coordinator_pid_file(coherence, refusing_port())
    (coherence / "hook.secret").write_text("test-secret")
    endpoint = cc.resolve_endpoint(tmp_path)
    release_coordinator_lock(pid_file)  # the coordinator exits after the endpoint resolved

    with pytest.raises(cc.CoordinatorDeparted) as raised:
        cc.post(endpoint, "/hooks/pre-edit", {"session_id": "s", "path": "p"})

    assert cc.classify_send_failure(raised.value) == "departed"


@pytest.mark.parametrize("settled", [True, False], ids=["settled", "unsettled"])
def test_a_refused_caller_principal_is_named_principal_refused(settled: bool) -> None:
    refusal = CallerPrincipalRefused("caller_principal_absent", "refused", settled=settled)
    assert cc.classify_send_failure(refusal) == "principal_refused"


@pytest.mark.parametrize(
    "exc",
    [ValueError("x"), AttributeError("x"), OSError("not wrapped by the client")],
    ids=["value-error", "attribute-error", "unwrapped-os-error"],
)
def test_anything_outside_the_send_failures_is_named_unexpected(exc: BaseException) -> None:
    assert cc.classify_send_failure(exc) == "unexpected"
