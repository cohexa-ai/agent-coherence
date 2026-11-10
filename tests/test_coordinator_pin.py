"""A local client sends nothing to a coordinator that is gone (#503).

A client finds the coordinator through the port in ``.coherence/server.pid`` and
sends the workspace secret with every request. Once the coordinator exits, its
port is free and any local process can bind it. So before each request a local
client checks that the pid file still names the coordinator it resolved and
that the coordinator still holds the workspace lock (the ``flock`` a start
takes on ``server.pid``).
"""

from __future__ import annotations

import errno
import fcntl
import http.server
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccs.adapters.claude_code import coordinator_pin
from ccs.adapters.claude_code.state_dir import dir_link_refusal, entry_link_refusal
from ccs.cli._coherence_client import (
    CoordinatorUnavailable,
    coordinator_backend,
    get,
    post,
    resolve_endpoint,
    resolve_remote_endpoint,
)

_SECRET = "s3cret-for-the-test"


class _Recorder(http.server.BaseHTTPRequestHandler):
    """Answers every request with ``{"ok": true}`` and records what arrived."""

    def do_GET(self) -> None:  # noqa: N802 - http.server's handler name
        self._answer()

    def do_POST(self) -> None:  # noqa: N802 - http.server's handler name
        self._answer()

    def _answer(self) -> None:
        self.server.seen.append(self.headers.get("Authorization"))  # type: ignore[attr-defined]
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


@pytest.fixture
def listener() -> Iterator[http.server.ThreadingHTTPServer]:
    """Something listening on a port: a coordinator, or a stranger that bound
    the port a coordinator left."""
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    httpd.seen = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _write_pid_file(root: Path, pid: int, port: int, backend: str | None = None) -> Path:
    coherence = root / ".coherence"
    coherence.mkdir(mode=0o700, exist_ok=True)
    secret = coherence / "hook.secret"
    secret.write_text(f"{_SECRET}\n")
    secret.chmod(0o600)
    pid_file = coherence / "server.pid"
    pid_file.write_text(f"{pid}\n{port}\n" + (f"backend={backend}\n" if backend else ""))
    return pid_file


def _hold(pid_file: Path) -> int:
    """The lock a running coordinator holds on its pid file, as a descriptor
    the test closes when that coordinator 'exits'."""
    fd = os.open(pid_file, os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


def _exited_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(60)
    return child.pid


def _port(server: http.server.ThreadingHTTPServer) -> int:
    return server.server_address[1]


def test_a_running_coordinator_is_reached(tmp_path: Path, listener) -> None:
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    try:
        assert post(resolve_endpoint(tmp_path), "/hooks/x", {}) == {"ok": True}
        assert get(resolve_endpoint(tmp_path), "/status") == {"ok": True}
    finally:
        os.close(held)
    assert listener.seen == [f"Bearer {_SECRET}"] * 2


def test_a_coordinator_that_exited_is_refused_before_its_secret_is_sent(
    tmp_path: Path, listener
) -> None:
    """Its pid file still names its port, which a stranger now listens on."""
    _write_pid_file(tmp_path, os.getpid(), _port(listener))
    with pytest.raises(CoordinatorUnavailable) as caught:
        resolve_endpoint(tmp_path)
    assert caught.value.never_sent is True
    assert "no longer holds the workspace lock" in str(caught.value)
    assert listener.seen == []


def test_a_listener_on_the_port_after_the_coordinator_exits_receives_nothing(
    tmp_path: Path, listener
) -> None:
    """The client resolved a live coordinator; it then exited, and something
    else answers on its port. Nothing is sent, so the secret never leaves."""
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    endpoint = resolve_endpoint(tmp_path)
    os.close(held)
    for request in (lambda: post(endpoint, "/hooks/x", {}), lambda: get(endpoint, "/status")):
        with pytest.raises(CoordinatorUnavailable) as caught:
            request()
        assert caught.value.never_sent is True
    assert listener.seen == []


def test_a_new_coordinator_on_the_same_port_is_treated_as_a_new_one(
    tmp_path: Path, listener
) -> None:
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    try:
        endpoint = resolve_endpoint(tmp_path)
        pid_file.write_text(f"{os.getpid() + 1}\n{_port(listener)}\n")
        with pytest.raises(CoordinatorUnavailable) as caught:
            post(endpoint, "/hooks/x", {})
    finally:
        os.close(held)
    assert caught.value.never_sent is True
    assert listener.seen == []


def test_a_coordinator_that_is_stopping_is_refused(tmp_path: Path, listener) -> None:
    """A stop drops the port line first, so a reader sees no port."""
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    try:
        endpoint = resolve_endpoint(tmp_path)
        pid_file.write_text(f"{os.getpid()}\n")
        with pytest.raises(CoordinatorUnavailable):
            post(endpoint, "/hooks/x", {})
    finally:
        os.close(held)
    assert listener.seen == []


def test_a_pid_file_that_disappeared_cannot_tell_so_the_request_goes(
    tmp_path: Path, listener
) -> None:
    """An attached client keeps working when the pid file is gone."""
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    try:
        endpoint = resolve_endpoint(tmp_path)
        pid_file.unlink()
        assert post(endpoint, "/hooks/x", {}) == {"ok": True}
    finally:
        os.close(held)
    assert len(listener.seen) == 1


def test_a_pid_file_replaced_by_a_link_is_refused(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, listener
) -> None:
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    try:
        endpoint = resolve_endpoint(tmp_path)
        elsewhere = tmp_path_factory.mktemp("elsewhere") / "server.pid"
        elsewhere.write_text(f"{os.getpid()}\n{_port(listener)}\n")
        pid_file.unlink()
        pid_file.symlink_to(elsewhere)
        with pytest.raises(CoordinatorUnavailable):
            post(endpoint, "/hooks/x", {})
    finally:
        os.close(held)
    assert listener.seen == []


def test_without_flock_the_lock_cannot_tell_so_the_coordinator_is_reached(
    tmp_path: Path, listener, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ccs.adapters.claude_code import coordinator_pin

    monkeypatch.setattr(coordinator_pin, "fcntl", None)
    _write_pid_file(tmp_path, os.getpid(), _port(listener))
    assert post(resolve_endpoint(tmp_path), "/hooks/x", {}) == {"ok": True}
    assert len(listener.seen) == 1


def test_a_node_coordinator_that_runs_unlocked_is_judged_by_its_pid(
    tmp_path: Path, listener
) -> None:
    """A Node coordinator on a host with no flock helper holds no lock, so its
    pid decides: alive is reached, exited is refused."""
    _write_pid_file(tmp_path, os.getpid(), _port(listener), backend="node")
    assert post(resolve_endpoint(tmp_path), "/hooks/x", {}) == {"ok": True}
    _write_pid_file(tmp_path, _exited_pid(), _port(listener), backend="node")
    with pytest.raises(CoordinatorUnavailable) as caught:
        resolve_endpoint(tmp_path)
    assert caught.value.never_sent is True
    assert len(listener.seen) == 1


def test_a_remote_endpoint_is_not_pinned_to_a_pid_file(listener) -> None:
    """A remote coordinator has no pid file here; TLS, not this check, guards
    a routed link."""
    endpoint = resolve_remote_endpoint("127.0.0.1", _port(listener), _SECRET)
    assert post(endpoint, "/hooks/x", {}) == {"ok": True}


def test_the_backend_line_is_not_read_through_a_link(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "server.pid"
    elsewhere.write_text("1\n2\nbackend=node\n")
    (coherence / "server.pid").symlink_to(elsewhere)
    assert coordinator_backend(tmp_path) is None
    (coherence / "server.pid").unlink()
    (coherence / "server.pid").write_text("1\n2\nbackend=node\n")
    assert coordinator_backend(tmp_path) == "node"


def test_a_linked_pid_file_is_refused_at_resolve(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, listener
) -> None:
    """The file the link names describes a running coordinator, lock held, so
    only refusing the link keeps the secret from its port."""
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "server.pid"
    elsewhere.write_text(f"{os.getpid()}\n{_port(listener)}\n")
    held = _hold(elsewhere)
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    pid_file.unlink()
    pid_file.symlink_to(elsewhere)
    try:
        with pytest.raises(CoordinatorUnavailable) as caught:
            post(resolve_endpoint(tmp_path), "/hooks/x", {})
    finally:
        os.close(held)
    assert str(caught.value) == entry_link_refusal(pid_file)
    assert listener.seen == []


def test_a_pid_file_that_is_not_a_regular_file_is_refused_at_resolve(tmp_path: Path) -> None:
    pid_file = _write_pid_file(tmp_path, os.getpid(), 1)
    pid_file.unlink()
    pid_file.mkdir()
    with pytest.raises(CoordinatorUnavailable) as caught:
        resolve_endpoint(tmp_path)
    assert "not a regular file" in str(caught.value)


def test_a_node_coordinator_that_exits_after_resolve_is_named_as_no_longer_running(
    tmp_path: Path, listener
) -> None:
    """A Node coordinator holds no lock, so the refusal names the evidence it
    has: the pid is gone."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_pid_file(tmp_path, child.pid, _port(listener), backend="node")
        endpoint = resolve_endpoint(tmp_path)
    finally:
        child.kill()
        child.wait(60)
    with pytest.raises(CoordinatorUnavailable) as caught:
        post(endpoint, "/hooks/x", {})
    assert caught.value.never_sent is True
    assert f"(pid {child.pid}) is no longer running" in str(caught.value)
    assert listener.seen == []


def _flock_raising(code: int) -> SimpleNamespace:
    """``fcntl`` as the probe uses it, with every ``flock`` failing with ``code``."""

    def flock(_fd: int, _operation: int) -> None:
        raise OSError(code, os.strerror(code))

    return SimpleNamespace(flock=flock, LOCK_SH=fcntl.LOCK_SH, LOCK_NB=fcntl.LOCK_NB, LOCK_UN=fcntl.LOCK_UN)


def test_a_lock_error_that_is_not_contention_cannot_tell_so_the_coordinator_is_reached(
    tmp_path: Path, listener, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(coordinator_pin, "fcntl", _flock_raising(errno.EIO))
    _write_pid_file(tmp_path, os.getpid(), _port(listener))
    assert post(resolve_endpoint(tmp_path), "/hooks/x", {}) == {"ok": True}
    assert len(listener.seen) == 1


def test_a_pid_file_the_probe_cannot_open_cannot_tell_so_the_coordinator_is_reached(
    tmp_path: Path, listener, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_path: Path) -> int:
        raise PermissionError(errno.EACCES, "denied")

    monkeypatch.setattr(coordinator_pin, "open_regular", refuse)
    _write_pid_file(tmp_path, os.getpid(), _port(listener))
    assert post(resolve_endpoint(tmp_path), "/hooks/x", {}) == {"ok": True}
    assert len(listener.seen) == 1


def test_a_node_pid_file_with_no_pid_cannot_tell_so_the_coordinator_is_reached(
    tmp_path: Path, listener
) -> None:
    coherence = _write_pid_file(tmp_path, 1, _port(listener)).parent
    (coherence / "server.pid").write_text(f"\n{_port(listener)}\nbackend=node\n")
    assert post(resolve_endpoint(tmp_path), "/hooks/x", {}) == {"ok": True}
    assert len(listener.seen) == 1


def test_the_probe_unlocks_before_it_closes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fork between the probe's open and close leaves the child a copy of the
    descriptor; only an unlock keeps its shared lock from outliving the probe."""
    calls: list[int] = []

    def flock(fd: int, operation: int) -> None:
        calls.append(operation)
        fcntl.flock(fd, operation)

    monkeypatch.setattr(
        coordinator_pin,
        "fcntl",
        SimpleNamespace(flock=flock, LOCK_SH=fcntl.LOCK_SH, LOCK_NB=fcntl.LOCK_NB, LOCK_UN=fcntl.LOCK_UN),
    )
    pid_file = _write_pid_file(tmp_path, os.getpid(), 1)
    assert coordinator_pin.coordinator_lock_held(pid_file) is False
    assert calls == [fcntl.LOCK_SH | fcntl.LOCK_NB, fcntl.LOCK_UN]


def _make_unreadable(pid_file: Path) -> None:
    pid_file.chmod(0)


def _make_oversized(pid_file: Path) -> None:
    pid_file.write_text(pid_file.read_text() + "#" * coordinator_pin.PID_FILE_MAX_BYTES)


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        pytest.param(_make_unreadable, None, id="unreadable-cannot-tell"),
        pytest.param(_make_oversized, "is larger than a pid file", id="oversized-refuses"),
    ],
)
def test_a_pid_file_that_changes_after_resolve(
    tmp_path: Path, listener, change, refusal: str | None
) -> None:
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    try:
        endpoint = resolve_endpoint(tmp_path)
        change(pid_file)
        if refusal is None:
            assert post(endpoint, "/hooks/x", {}) == {"ok": True}
        else:
            with pytest.raises(CoordinatorUnavailable, match=refusal) as caught:
                post(endpoint, "/hooks/x", {})
            assert caught.value.never_sent is True
    finally:
        os.close(held)
        pid_file.chmod(0o600)
    assert len(listener.seen) == (1 if refusal is None else 0)


def test_a_coherence_dir_replaced_by_a_link_after_resolve_is_refused_by_name(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, listener
) -> None:
    pid_file = _write_pid_file(tmp_path, os.getpid(), _port(listener))
    held = _hold(pid_file)
    coherence = pid_file.parent
    try:
        endpoint = resolve_endpoint(tmp_path)
        moved = tmp_path_factory.mktemp("moved") / ".coherence"
        coherence.rename(moved)
        coherence.symlink_to(moved)
        with pytest.raises(CoordinatorUnavailable) as caught:
            post(endpoint, "/hooks/x", {})
    finally:
        os.close(held)
    assert str(caught.value).startswith(dir_link_refusal(coherence))
    assert listener.seen == []


def _kill_reports_eperm_for(pid: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """``os.kill(pid, 0)`` answers as for a process another user runs."""
    real_kill = os.kill

    def kill(target: int, signal: int) -> None:
        if target == pid:
            raise PermissionError(errno.EPERM, "Operation not permitted")
        real_kill(target, signal)

    monkeypatch.setattr(coordinator_pin.os, "kill", kill)


def test_a_node_pid_this_client_may_not_signal_cannot_tell_so_the_coordinator_is_reached(
    tmp_path: Path, listener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """EPERM shows a process exists: another user's, or this user's behind a
    sandbox that denies signals. That is no evidence the coordinator exited."""
    _write_pid_file(tmp_path, os.getpid(), _port(listener), backend="node")
    endpoint = resolve_endpoint(tmp_path)
    _kill_reports_eperm_for(os.getpid(), monkeypatch)
    assert post(endpoint, "/hooks/x", {}) == {"ok": True}
    assert len(listener.seen) == 1


def test_a_node_pid_no_process_can_have_is_not_the_coordinator(tmp_path: Path, listener) -> None:
    _write_pid_file(tmp_path, 2**70, _port(listener), backend="node")
    with pytest.raises(CoordinatorUnavailable) as caught:
        resolve_endpoint(tmp_path)
    assert caught.value.never_sent is True
    assert listener.seen == []


def test_a_node_coordinator_that_removes_its_pid_file_is_judged_by_its_pid(
    tmp_path: Path, listener
) -> None:
    """A Node coordinator running unlocked removes ``server.pid`` when it stops.
    While its pid runs, a missing file cannot tell and requests go; once the
    pid is gone, nothing more is sent."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        pid_file = _write_pid_file(tmp_path, child.pid, _port(listener), backend="node")
        endpoint = resolve_endpoint(tmp_path)
        pid_file.unlink()
        assert post(endpoint, "/hooks/x", {}) == {"ok": True}
    finally:
        child.kill()
        child.wait(60)
    with pytest.raises(CoordinatorUnavailable) as caught:
        post(endpoint, "/hooks/x", {})
    assert caught.value.never_sent is True
    assert f"(pid {child.pid}) is no longer running" in str(caught.value)
    assert len(listener.seen) == 1


def test_without_flock_a_node_pin_whose_pid_file_is_removed_still_reaches_the_coordinator(
    tmp_path: Path, listener, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Where the lock cannot be probed nothing reads as gone, and no pid is
    signalled to find out."""
    monkeypatch.setattr(coordinator_pin, "fcntl", None)
    pid_file = _write_pid_file(tmp_path, _exited_pid(), _port(listener), backend="node")
    endpoint = resolve_endpoint(tmp_path)
    pid_file.unlink()

    def no_signal(*_args: object) -> None:
        pytest.fail("signalled a pid where the lock cannot be probed")

    monkeypatch.setattr(coordinator_pin.os, "kill", no_signal)
    assert post(endpoint, "/hooks/x", {}) == {"ok": True}
    assert len(listener.seen) == 1
