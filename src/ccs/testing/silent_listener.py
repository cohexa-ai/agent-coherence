# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Loopback listeners that take a coordinator request and never answer it.

:class:`SilentListener` accepts each TCP connection, reads the whole HTTP
request, and holds the connection open without writing a byte, so the client
waits until its own read timeout, as it would on a coordinator that is wedged
after the connect. :class:`ForwardingListener` relays the first ``relay``
requests to a real coordinator, and the coordinator's answers back, then holds
every later request the same way, so the silence falls in the middle of an
operation whose earlier requests were answered::

    from dataclasses import replace
    from ccs.testing.silent_listener import ForwardingListener

    with ForwardingListener(volume._endpoint.port, relay=1) as listener:
        volume._endpoint = replace(volume._endpoint, port=listener.port)
        volume.write(path, data)  # pre-edit relayed, post-edit never answered

Both count at the far end, so a test needs no patch of the client to see what
an operation sent: ``accepted`` counts every connection, and each request
received in full (its route and body) is recorded in ``relayed`` if it was
picked for relaying, else in ``silenced``. A connection that sends no whole
request (it closes first, or :meth:`SilentListener.close` shuts it down) is
counted in ``accepted`` and recorded nowhere.

Test-only, one machine, loopback. A held connection is a live peer that does
not answer: not a half-open routed link, a dropped packet or a partition.
Requests are read as one HTTP/1.1 request per connection, the way the
coordinator client sends them (``Connection: close``, a ``Content-Length``
body); a relayed answer is copied until the coordinator closes the connection.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass

__all__ = ["ForwardingListener", "ReceivedRequest", "SilentListener"]

_HEADER_END = b"\r\n\r\n"
_CHUNK = 65536
# A relayed request waits at most this long on the coordinator: a wedged one
# must not leave the relay thread blocked for the rest of the test session.
_UPSTREAM_TIMEOUT_SEC = 30.0


@dataclass(frozen=True)
class ReceivedRequest:
    """One request a listener received in full."""

    route: str
    body: bytes


def _read_request(conn: socket.socket) -> tuple[bytes, ReceivedRequest] | None:
    """The raw bytes and the parsed request, or ``None`` if the peer closed
    before sending all of it."""
    raw = b""
    while _HEADER_END not in raw:
        chunk = conn.recv(_CHUNK)
        if not chunk:
            return None
        raw += chunk
    head, _sep, body = raw.partition(_HEADER_END)
    lines = head.decode("latin-1").split("\r\n")
    length = next(
        (int(value) for name, _c, value in (line.partition(":") for line in lines[1:])
         if name.strip().lower() == "content-length"),
        0,
    )
    while len(body) < length:
        chunk = conn.recv(_CHUNK)
        if not chunk:
            return None
        body += chunk
        raw += chunk
    request_line = lines[0].split(" ")
    route = request_line[1] if len(request_line) > 1 else ""
    return raw, ReceivedRequest(route, body)


class SilentListener:
    """Accepts TCP on ``127.0.0.1`` and never writes a byte to any connection.

    Each request is read in full and recorded in ``silenced``; its connection
    is then held open, and drained, until the client closes it or
    :meth:`close` shuts it down."""

    def __init__(self) -> None:
        self.accepted = 0
        self.relayed: list[ReceivedRequest] = []
        self.silenced: list[ReceivedRequest] = []
        self._lock = threading.Lock()
        self._closed = False
        self._conns: set[socket.socket] = set()
        self._sock = socket.create_server(("127.0.0.1", 0))
        self.port: int = self._sock.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def __enter__(self) -> SilentListener:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop accepting and shut down every open connection: held,
        relaying, or still sending its request. Idempotent."""
        try:
            # Linux does not wake an accept() blocked in another thread on
            # close(), only on shutdown(); macOS refuses this with ENOTCONN and
            # wakes it on close().
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        with self._lock:
            self._closed = True
            conns = list(self._conns)
        for conn in conns:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # the client already closed it

    def _take_relay(self) -> bool:
        """Whether the next request is relayed; called under ``_lock``."""
        return False

    def _accept_loop(self) -> None:
        while True:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return  # closed
            with self._lock:
                # Accepted while close() ran, after it took its list of
                # connections: shut it now, or nothing ever would.
                if self._closed:
                    conn.close()
                    return
                self.accepted += 1
                self._conns.add(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        with conn:
            try:
                self._serve(conn)
            finally:
                with self._lock:
                    self._conns.discard(conn)

    def _serve(self, conn: socket.socket) -> None:
        try:
            received = _read_request(conn)
        except OSError:
            return
        if received is None:
            return
        raw, request = received
        with self._lock:
            relay = self._take_relay()
            (self.relayed if relay else self.silenced).append(request)
        if relay:
            self._relay(conn, raw)
        else:
            self._hold(conn)

    def _hold(self, conn: socket.socket) -> None:
        while True:
            try:
                if not conn.recv(_CHUNK):
                    return
            except OSError:
                return

    def _relay(self, conn: socket.socket, raw: bytes) -> None:
        raise NotImplementedError  # only a ForwardingListener relays


class ForwardingListener(SilentListener):
    """Relays the first ``relay`` requests to the coordinator on
    ``127.0.0.1:upstream_port``, byte for byte, with its answers; holds every
    later one silent, as :class:`SilentListener` does. Requests are counted
    from this listener's creation, in the order they arrive in full, and each
    one picked for relaying is recorded in ``relayed`` before it is sent, so
    a relay that then fails to reach the coordinator is listed there too."""

    def __init__(self, upstream_port: int, *, relay: int) -> None:
        if relay < 0:
            raise ValueError(f"relay must be >= 0, got {relay}")
        self._upstream_port = upstream_port
        self._relay_left = relay
        super().__init__()

    def _take_relay(self) -> bool:
        if self._relay_left == 0:
            return False
        self._relay_left -= 1
        return True

    def _relay(self, conn: socket.socket, raw: bytes) -> None:
        try:
            with socket.create_connection(
                ("127.0.0.1", self._upstream_port), timeout=_UPSTREAM_TIMEOUT_SEC
            ) as upstream:
                upstream.sendall(raw)
                while chunk := upstream.recv(_CHUNK):
                    conn.sendall(chunk)
        except OSError:
            pass  # the client sees the connection close; the request stays recorded
