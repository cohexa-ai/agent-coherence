# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The contract of :mod:`ccs.testing.silent_listener`.

``tests/test_silent_coordinator.py`` reads a coordinator that never answers
from these listeners, so what they claim is pinned here: a silent listener
never writes a byte and records each request it held; a forwarding listener
relays exactly its first N requests, answers included, and holds the rest.
"""

from __future__ import annotations

import http.server
import socket
import threading
import urllib.request
from collections.abc import Iterator

import pytest

from ccs.testing.silent_listener import ForwardingListener, ReceivedRequest, SilentListener

_REQUEST = b"POST /hooks/pre-read HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\n\r\n{}"


class _Answer(http.server.BaseHTTPRequestHandler):
    """Answers every POST 200 with ``ok``."""

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture
def upstream() -> Iterator[int]:
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Answer)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _post(port: int, route: str, timeout: float = 0.3) -> bytes:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{route}", data=b'{"n": 1}', method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as answer:
        return answer.read()


def test_a_silent_listener_holds_a_whole_request_and_never_writes() -> None:
    """The request is read in full and recorded; nothing comes back on the
    connection until the listener closes it, and then only the close."""
    with SilentListener() as silent, socket.create_connection(("127.0.0.1", silent.port)) as client:
        client.sendall(_REQUEST)
        client.settimeout(0.3)
        with pytest.raises(TimeoutError):
            client.recv(1)
        silent.close()
        client.settimeout(2)
        assert client.recv(1) == b"", "the listener wrote to the connection"

    assert silent.accepted == 1
    assert silent.silenced == [ReceivedRequest("/hooks/pre-read", b"{}")]
    assert silent.relayed == []


def test_a_connection_that_sends_no_whole_request_is_counted_and_not_recorded() -> None:
    with SilentListener() as silent:
        with socket.create_connection(("127.0.0.1", silent.port)) as client:
            client.sendall(_REQUEST[:20])
        with pytest.raises(TimeoutError):
            _post(silent.port, "/hooks/pre-edit")

    assert silent.accepted == 2
    assert silent.silenced == [ReceivedRequest("/hooks/pre-edit", b'{"n": 1}')]


def test_a_request_whose_body_is_cut_short_is_not_recorded() -> None:
    """A request counts only once its whole ``Content-Length`` body arrived."""
    with SilentListener() as silent:
        with socket.create_connection(("127.0.0.1", silent.port)) as client:
            client.sendall(_REQUEST.replace(b"Content-Length: 2", b"Content-Length: 10"))
        with pytest.raises(TimeoutError):
            _post(silent.port, "/whole")

    assert silent.accepted == 2
    assert [r.route for r in silent.silenced] == ["/whole"]


def test_close_shuts_down_a_connection_still_sending_its_request() -> None:
    """A connection closed mid-request sees the close, and its request is
    never recorded, even if the rest of it arrives afterwards."""
    with SilentListener() as silent, socket.create_connection(("127.0.0.1", silent.port)) as client:
        client.sendall(_REQUEST[:20])
        while silent.accepted == 0:
            threading.Event().wait(0.01)
        silent.close()
        client.settimeout(2)
        assert client.recv(1) == b""
        try:
            client.sendall(_REQUEST[20:])
        except OSError:
            pass  # the listener's side is already gone

    assert silent.silenced == []
    assert silent.relayed == []


def test_a_forwarding_listener_relays_its_first_requests_and_holds_the_rest(upstream: int) -> None:
    """Relaying two, the first two posts get the upstream's answer and the
    third times out: the listener counts all three and records each by what
    happened to it."""
    with ForwardingListener(upstream, relay=2) as listener:
        assert _post(listener.port, "/a") == b"ok"
        assert _post(listener.port, "/b") == b"ok"
        with pytest.raises(TimeoutError):
            _post(listener.port, "/c")

    assert listener.accepted == 3
    assert [r.route for r in listener.relayed] == ["/a", "/b"]
    assert [r.route for r in listener.silenced] == ["/c"]


def test_a_forwarding_listener_relaying_none_is_silent(upstream: int) -> None:
    with ForwardingListener(upstream, relay=0) as listener:
        with pytest.raises(TimeoutError):
            _post(listener.port, "/a")

    assert listener.relayed == []
    assert [r.route for r in listener.silenced] == ["/a"]


def test_a_negative_relay_count_is_refused() -> None:
    with pytest.raises(ValueError, match="relay must be >= 0"):
        ForwardingListener(1, relay=-1)
