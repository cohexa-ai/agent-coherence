"""A coordinator port nothing answers on, shared by tests that need the real
client to fail exactly as it does against a coordinator that is gone."""

from __future__ import annotations

import errno
import socket
from dataclasses import replace

from ccs.cli._coherence_client import CoordinatorEndpoint


def refusing_port() -> int:
    """A loopback port that refuses connections and that no other test can take.

    A port freed after ``bind(("127.0.0.1", 0))`` can be handed to another
    test worker's port-0 bind before the connect, and the connect then reaches
    a stranger. Ports below 1024 are never handed out for port 0, and port 4
    is unassigned, so no standard service listens on it; the assert says so
    if something does. A socket bound without listening is no substitute: on
    macOS a connect to it hangs until its timeout instead of being refused."""
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", 4)) == errno.ECONNREFUSED, "port 4 accepts connections here"
    return 4


def refused_endpoint(endpoint: CoordinatorEndpoint) -> CoordinatorEndpoint:
    """``endpoint`` moved to :func:`refusing_port`, so a request to it is refused
    before it is sent."""
    return replace(endpoint, port=refusing_port())
