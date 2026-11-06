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

import socket
import struct
import threading
import time
from collections.abc import Callable, Iterator

import pytest

from ccs.cli import _coherence_client as cc
from ccs.cli._coherence_client import CoordinatorEndpoint, CoordinatorUnavailable
from tests.loopback_helpers import refusing_port

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
