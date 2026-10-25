# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A coordinator that takes a request and never answers it.

Each strict surface attaches to a real coordinator, then is pointed at a
loopback listener from :mod:`ccs.testing.silent_listener` that holds the
connection open without writing a byte, so the client waits until its own read
timeout. With that timeout shrunk to :data:`TIMEOUT_SEC`, every case raises its
typed error within ``TIMEOUT_SEC + SLACK_SEC``; a ``write()``, which also sends
a release after its commit, adds the release's own timeout. The listener counts
connections at the far end, so each case also shows how many requests the
operation sent, and that none was retried.

- A request that commits nothing and gets no answer raises
  ``CoordinatorUnreachable``: a volume's ``read()`` and ``write()`` (its
  pre-edit), and a substrate binding's ``read()`` and ``commit()``. The
  commit raises from its first pre-read, before the substrate is touched.
- A commit request that reached the coordinator and got no answer may have
  landed, so it raises ``CommitUnconfirmed``, never ``CoordinatorUnreachable``:
  ``write()``'s post-edit (its bytes are already on disk), ``write_cas()``'s
  compare-and-swap (nothing touches disk), and a substrate commit's coordinator
  bump (the substrate write landed first, by design, and is not re-driven).

The coordinator runs in this process. That its registry's version did not move
is a control on the harness, not a property of the library: a silenced request
never reaches the coordinator, so the version stays put unless a request went
around the listener. Against a real coordinator that went silent, a commit
that raised ``CommitUnconfirmed`` may have moved it. Raising when the connect
is refused, or when the connection is dropped, is pinned in
``tests/adapters/test_coherent_volume.py`` and
``tests/adapters/test_substrate_cross_agent.py``; this file pins the hang.

What these do not prove: one machine and loopback only. A held connection is a
live peer that does not answer, not a half-open routed link, a dropped packet
or a network partition. The hook client is not exercised (it fails open by
design), and neither is the Node coordinator.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

import ccs.cli._coherence_client as coherence_client
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import _GRANT_RELEASE_TIMEOUT_SEC, CoherentVolume
from ccs.adapters.substrate import CoordinatedSubstrate, SubstrateCoordinatorSession
from ccs.cli._coherence_client import CoordinatorUnavailable
from ccs.core.exceptions import CoherenceError, CommitUnconfirmed, CoordinatorUnreachable
from ccs.testing.silent_listener import ForwardingListener, ReceivedRequest, SilentListener
from ccs.testing.substrate_conformance import ConformanceSubstrate, InMemoryStore
from tests.coordinator_helpers import recorded_version, serve_in_process, write_holders

#: The client's read timeout in every case here, shrunk from the shipped 6 s.
TIMEOUT_SEC = 0.5
#: What a case may take beyond the timeouts it waits out.
SLACK_SEC = 0.5

REL = "data/shared.txt"
#: Why the coordinator's version would move: a request went around the listener.
_BYPASSED = "control: a request reached the coordinator past the listener"


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``data/**`` in this process."""
    (tmp_path / "data").mkdir()
    (tmp_path / REL).write_bytes(b"v1")
    coordinator = serve_in_process(tmp_path, "silent-coordinator")
    yield coordinator
    coordinator.shutdown()


def _point_at(owner: CoherentVolume | SubstrateCoordinatorSession, port: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """Send ``owner``'s requests to the listener on ``port`` from now on, and
    shrink the client's timeout to :data:`TIMEOUT_SEC` (it is read per
    request). Attach and setup ran before, at the shipped timeout."""
    owner._endpoint = replace(owner._endpoint, port=port)
    monkeypatch.setattr(coherence_client, "CLI_HTTP_TIMEOUT_SEC", TIMEOUT_SEC)


def _raised_in(op: Callable[[], object]) -> tuple[CoherenceError, float]:
    """What ``op`` raised, and how long it took to raise it."""
    start = time.monotonic()
    with pytest.raises(CoherenceError) as raised:
        op()
    return raised.value, time.monotonic() - start


def _assert_ended_by_the_read_timeout(exc: CoherenceError, elapsed: float, *, waited: float) -> None:
    """``exc`` came from the client's read timeout on a delivered request, and
    the operation took the ``waited`` seconds of timeouts it sat through and at
    most :data:`SLACK_SEC` more. The floor and the cause are the controls: a
    connection that was reset or refused fails at once, with another cause."""
    cause = exc.__cause__
    assert isinstance(cause, CoordinatorUnavailable), repr(cause)
    assert isinstance(cause.__cause__, TimeoutError), repr(cause.__cause__)
    assert cause.never_sent is False, "control: the request was delivered"
    assert 0.9 * waited <= elapsed <= waited + SLACK_SEC, f"{elapsed:.3f}s for {waited}s of timeouts"


def _routes(requests: list[ReceivedRequest]) -> list[str]:
    return [request.route for request in requests]


def _volume(root: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(root, managed=("data/**",), config=cfg)


def _substrate(
    root: Path, cfg: LifecycleConfig,
) -> tuple[CoordinatedSubstrate, SubstrateCoordinatorSession, ConformanceSubstrate, InMemoryStore]:
    """A substrate binding over one in-memory object holding ``s1``, attached
    to the coordinator, with its session, binding view and store."""
    store = InMemoryStore()
    store.set(REL, b"s1")
    binding = ConformanceSubstrate(store)
    session = SubstrateCoordinatorSession(root, managed=("data/**",), config=cfg)
    return CoordinatedSubstrate(binding, session), session, binding, store


# --- a request that commits nothing: CoordinatorUnreachable ------------------


@pytest.mark.parametrize(("op", "route"), [("read", "/hooks/pre-read"), ("write", "/hooks/pre-edit")])
def test_a_volume_request_that_gets_no_answer_is_unreachable_within_the_timeout(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch,
    op: str, route: str,
) -> None:
    """A strict volume's ``read()``, or ``write()``'s pre-edit, sent to a
    coordinator that never answers raises ``CoordinatorUnreachable`` once the
    client's read timeout runs out, on one connection. The file keeps its
    bytes."""
    vol = _volume(tmp_path, fast_cfg)
    vol.read(REL)
    version = recorded_version(server, REL)
    with SilentListener() as silent:
        _point_at(vol, silent.port, monkeypatch)
        exc, elapsed = _raised_in(lambda: vol.read(REL) if op == "read" else vol.write(REL, b"v2"))

    assert type(exc) is CoordinatorUnreachable, repr(exc)
    _assert_ended_by_the_read_timeout(exc, elapsed, waited=TIMEOUT_SEC)
    assert silent.accepted == 1, "the request was retried"
    assert _routes(silent.silenced) == [route]
    assert (tmp_path / REL).read_bytes() == b"v1"
    assert recorded_version(server, REL) == version, _BYPASSED


@pytest.mark.parametrize("op", ["read", "commit"])
def test_a_substrate_request_that_gets_no_answer_is_unreachable_within_the_timeout(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch,
    op: str,
) -> None:
    """A substrate binding's ``read()``, or the pre-read a ``commit()`` sends
    first, sent to a coordinator that never answers raises
    ``CoordinatorUnreachable`` once the client's read timeout runs out, on one
    connection. The commit never reaches the substrate."""
    sub, session, binding, store = _substrate(tmp_path, fast_cfg)
    _bytes, token = sub.read(REL)
    version = recorded_version(server, REL)
    with SilentListener() as silent:
        _point_at(session, silent.port, monkeypatch)
        exc, elapsed = _raised_in(
            lambda: sub.read(REL) if op == "read" else sub.commit(REL, expected_token=token, new_bytes=b"s2")
        )

    assert type(exc) is CoordinatorUnreachable, repr(exc)
    _assert_ended_by_the_read_timeout(exc, elapsed, waited=TIMEOUT_SEC)
    assert silent.accepted == 1, "the request was retried"
    assert _routes(silent.silenced) == ["/hooks/pre-read"]
    assert binding.cas_calls == [], "the substrate was written with no coordinator answer"
    assert store.get(REL) == (b"s1", token)
    assert recorded_version(server, REL) == version, _BYPASSED


# --- a commit that reached the coordinator: CommitUnconfirmed ----------------


def test_a_write_whose_commit_gets_no_answer_is_unconfirmed_and_attempts_its_release(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``write()``'s pre-edit is answered, so it holds the grant and puts its
    bytes on disk; its commit (post-edit) then reaches a coordinator that never
    answers. It raises ``CommitUnconfirmed``, since the commit may have landed,
    and never reports success. Before raising it sends one release, which goes
    unanswered too and is bounded by its own 1 s timeout: two silent
    connections, no retry of either. The grant still stands at the
    coordinator and stays on the volume's record, so the next re-mint
    releases it."""
    vol = _volume(tmp_path, fast_cfg)
    vol.read(REL)
    holder = vol._incarnation
    version = recorded_version(server, REL)
    with ForwardingListener(vol._endpoint.port, relay=1) as listener:
        _point_at(vol, listener.port, monkeypatch)
        exc, elapsed = _raised_in(lambda: vol.write(REL, b"v2"))

    assert type(exc) is CommitUnconfirmed, repr(exc)
    _assert_ended_by_the_read_timeout(exc, elapsed, waited=TIMEOUT_SEC + _GRANT_RELEASE_TIMEOUT_SEC)
    assert listener.accepted == 3, "a request was retried"
    assert _routes(listener.relayed) == ["/hooks/pre-edit"]
    commit, release = listener.silenced
    assert (commit.route, json.loads(commit.body)["success"]) == ("/hooks/post-edit", True)
    assert release.route == "/hooks/post-edit"
    assert json.loads(release.body) == {
        "session_id": vol.session_id, "agent_id": holder, "path": REL, "success": False,
    }
    assert (tmp_path / REL).read_bytes() == b"v2", "control: the bytes went to disk before the commit"
    assert REL not in vol._last_committed_hash, "write() went on as though its commit landed"
    assert recorded_version(server, REL) == version, _BYPASSED
    assert write_holders(server, REL) == {str(session_to_agent_id(vol.session_id, holder))}
    assert vol._grant_incarnations.get(holder) == {REL}, "an unanswered release dropped the grant from the record"


def test_a_cas_whose_compare_and_swap_gets_no_answer_is_unconfirmed_and_touches_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``write_cas()``'s comparand read is answered; its compare-and-swap then
    reaches a coordinator that never answers. It raises ``CommitUnconfirmed``:
    the swap may have landed, so ``CoordinatorUnreachable``, which says nothing
    happened, would be wrong. Nothing touches disk, and the swap is sent
    once."""
    vol = _volume(tmp_path, fast_cfg)
    vol.read(REL)
    version = recorded_version(server, REL)
    with ForwardingListener(vol._endpoint.port, relay=1) as listener:
        _point_at(vol, listener.port, monkeypatch)
        exc, elapsed = _raised_in(lambda: vol.write_cas(REL, lambda current: current + b"+cas"))

    assert type(exc) is CommitUnconfirmed, repr(exc)
    _assert_ended_by_the_read_timeout(exc, elapsed, waited=TIMEOUT_SEC)
    assert listener.accepted == 2, "a request was retried"
    assert _routes(listener.relayed) == ["/hooks/pre-read"]
    assert _routes(listener.silenced) == ["/hooks/post-edit-cas"]
    assert (tmp_path / REL).read_bytes() == b"v1"
    assert recorded_version(server, REL) == version, _BYPASSED


def test_a_substrate_commit_whose_bump_gets_no_answer_is_unconfirmed_and_not_re_driven(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A substrate ``commit()``'s pre-read is answered and its substrate write
    lands, since the substrate goes first; the coordinator bump then reaches a
    coordinator that never answers. It raises ``CommitUnconfirmed``. The
    substrate holds the new bytes, written once and never re-driven, and the
    bump is sent once."""
    sub, session, binding, store = _substrate(tmp_path, fast_cfg)
    _bytes, token = sub.read(REL)
    version = recorded_version(server, REL)
    with ForwardingListener(session._endpoint.port, relay=1) as listener:
        _point_at(session, listener.port, monkeypatch)
        exc, elapsed = _raised_in(lambda: sub.commit(REL, expected_token=token, new_bytes=b"s2"))

    assert type(exc) is CommitUnconfirmed, repr(exc)
    _assert_ended_by_the_read_timeout(exc, elapsed, waited=TIMEOUT_SEC)
    assert listener.accepted == 2, "a request was retried"
    assert _routes(listener.relayed) == ["/hooks/pre-read"]
    assert _routes(listener.silenced) == ["/hooks/post-edit-cas"]
    assert len(binding.cas_calls) == 1, "the landed substrate write was re-driven"
    assert store.get(REL)[0] == b"s2"
    assert recorded_version(server, REL) == version, _BYPASSED
