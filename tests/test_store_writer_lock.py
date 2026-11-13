"""A store has one read-write registry: every other read-write open is refused (#453).

A second writer on one store breaks the registry's check-then-act sequences,
which only its own in-process lock serializes, and a second open in the same
process can drop the first handle's SQLite locks and lose its commit. So every
read-write ``SqliteArtifactRegistry`` takes an exclusive ``flock`` on a lock file
beside the store; ``read_only=True`` takes none.
"""

from __future__ import annotations

import errno
import fcntl
import gc
import os
import pickle
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

import ccs
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.coordinator import sqlite_registry
from ccs.coordinator.sqlite_registry import (
    SchemaVersionError,
    SqliteArtifactRegistry,
    StoreHeldError,
    trial_writer_open,
)
from ccs.core.exceptions import STORE_HELD_REASON

_TRY_OPEN = """
import sys
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry, StoreHeldError
try:
    SqliteArtifactRegistry(sys.argv[1]).close()
    print("granted")
except StoreHeldError:
    print("refused")
"""

_HOLD_THEN_FORK = """
import os, sys, time
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
registry = SqliteArtifactRegistry(sys.argv[1])
pid = os.fork()
if pid == 0:
    time.sleep(240); os._exit(0)
print(pid, flush=True)
time.sleep(240)
"""

_HOLD = """
import sys, time
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
registry = SqliteArtifactRegistry(sys.argv[1])
print("up", flush=True)
time.sleep(240)
"""

_TAKE_WRITE_LOCK = """
import sqlite3, sys
conn = sqlite3.connect(sys.argv[1], isolation_level=None, timeout=0.2)
try:
    conn.execute("BEGIN IMMEDIATE")
    print("took it")
except sqlite3.OperationalError:
    print("locked")
"""


def _child_env() -> dict[str, str]:
    """This environment with this tree's ``src`` first on PYTHONPATH, so a
    child interpreter runs the code under test, not another checkout's."""
    tree_src = str(Path(ccs.__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    return {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [tree_src, existing]))}


def _run_child(script: str, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True, text=True, timeout=240, check=True, env=_child_env(),
    )
    return done.stdout.strip()


def _start_child(script: str, *args: str) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdout=subprocess.PIPE, text=True, env=_child_env(),
    )


def test_a_second_open_in_this_process_is_refused(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    with SqliteArtifactRegistry(db):
        with pytest.raises(StoreHeldError) as caught:
            SqliteArtifactRegistry(db)
    assert caught.value.reason == STORE_HELD_REASON == "store_held"
    assert caught.value.lock_path == str(db.resolve()) + ".writer-lock"


def test_a_second_open_in_another_process_is_refused_until_the_holder_closes(
    tmp_path: Path,
) -> None:
    db = tmp_path / "state.db"
    with SqliteArtifactRegistry(db):
        assert _run_child(_TRY_OPEN, str(db)) == "refused"
    assert _run_child(_TRY_OPEN, str(db)) == "granted"


def test_a_symlink_to_a_held_store_meets_the_same_lock(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    link = tmp_path / "elsewhere.db"
    link.symlink_to(db)
    with SqliteArtifactRegistry(db):
        with pytest.raises(StoreHeldError) as caught:
            SqliteArtifactRegistry(link)
    assert caught.value.lock_path == str(db.resolve()) + ".writer-lock"


def test_a_read_only_open_stays_allowed_and_frees_nothing(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    with SqliteArtifactRegistry(db) as writer:
        artifact = writer.resolve_or_register("x", "a" * 64)
        with SqliteArtifactRegistry(db, read_only=True) as reader:
            assert reader.get_artifact(artifact) is not None
        assert _run_child(_TRY_OPEN, str(db)) == "refused"


def test_close_releases_the_lock(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    # Kept referenced, so close() frees the lock, not the garbage collector.
    registry = SqliteArtifactRegistry(db)
    registry.close()
    assert _run_child(_TRY_OPEN, str(db)) == "granted"
    assert registry.close() is None


def test_a_dropped_handle_releases_the_lock(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    dropped = SqliteArtifactRegistry(db)
    del dropped
    gc.collect()
    SqliteArtifactRegistry(db).close()


def test_a_second_close_does_not_free_a_later_holders_lock(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    first = SqliteArtifactRegistry(db)
    first.close()
    with SqliteArtifactRegistry(db):
        first.close()
        assert _run_child(_TRY_OPEN, str(db)) == "refused"


def test_a_failed_open_releases_the_lock(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = -1")
    conn.close()
    with pytest.raises(SchemaVersionError) as kept:
        SqliteArtifactRegistry(db)
    # ``kept`` holds the traceback, so the failed registry is still alive: only
    # the open's own release frees the lock here, not a finalizer.
    with pytest.raises(SchemaVersionError):
        SqliteArtifactRegistry(db)
    assert kept.value.reason is not None


def test_an_open_that_fails_before_its_connect_completes_releases_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "state.db"

    def interrupted(*_args: object, **_kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(sqlite_registry.sqlite3, "connect", interrupted)
    with pytest.raises(KeyboardInterrupt) as kept:
        SqliteArtifactRegistry(db)
    monkeypatch.undo()
    # ``kept`` holds the half-built registry alive: only the open's own release
    # frees the lock here.
    assert _run_child(_TRY_OPEN, str(db)) == "granted"
    assert kept.value is not None


def test_a_killed_holder_releases_the_lock(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    holder = _start_child(_HOLD, str(db))
    try:
        assert holder.stdout.readline().strip() == "up"
        assert _run_child(_TRY_OPEN, str(db)) == "refused"
    finally:
        holder.send_signal(signal.SIGKILL)
        holder.wait(60)
    assert _run_child(_TRY_OPEN, str(db)) == "granted"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")  # fork() beside xdist's threads
def test_a_forked_child_closing_its_copy_keeps_the_parents_lock(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    registry = SqliteArtifactRegistry(db)
    try:
        pid = os.fork()
        if pid == 0:
            try:
                registry.close()
            finally:
                os._exit(0)
        os.waitpid(pid, 0)
        assert _run_child(_TRY_OPEN, str(db)) == "refused"
    finally:
        registry.close()


def test_a_dead_parents_lock_does_not_survive_in_its_forked_child(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    parent = _start_child(_HOLD_THEN_FORK, str(db))
    child = None
    try:
        child = int(parent.stdout.readline())
        assert _run_child(_TRY_OPEN, str(db)) == "refused"
        parent.send_signal(signal.SIGKILL)
        parent.wait(60)
        assert _run_child(_TRY_OPEN, str(db)) == "granted"
    finally:
        parent.kill()
        parent.wait(60)
        if child is not None:
            os.kill(child, signal.SIGKILL)


def test_the_lock_is_not_inherited_across_exec(tmp_path: Path) -> None:
    with SqliteArtifactRegistry(tmp_path / "state.db") as registry:
        assert os.get_inheritable(registry._writer_lock.fd) is False


def test_a_refused_open_leaves_the_holders_write_transaction_locked(tmp_path: Path) -> None:
    """Closing any descriptor on a file drops this process's POSIX locks on it.
    An open that reached the store before refusing would free the holder's
    SQLite write lock, and a third process could then commit inside the
    holder's transaction and lose the holder's commit."""
    db = tmp_path / "state.db"
    with SqliteArtifactRegistry(db) as holder:
        holder._conn.execute("BEGIN IMMEDIATE")
        try:
            with pytest.raises(StoreHeldError):
                SqliteArtifactRegistry(db)
            third = _run_child(_TAKE_WRITE_LOCK, str(db))
        finally:
            holder._conn.execute("ROLLBACK")
    assert third == "locked"


def test_a_registry_says_whether_it_holds_the_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "state.db"
    writer = SqliteArtifactRegistry(db)
    with SqliteArtifactRegistry(db, read_only=True) as reader:
        assert reader.holds_writer_lock is False
    assert writer.holds_writer_lock is True
    writer.close()
    assert writer.holds_writer_lock is False
    monkeypatch.setattr(sqlite_registry, "fcntl", None)
    with SqliteArtifactRegistry(db) as unlocked:
        assert unlocked.holds_writer_lock is False


def test_without_fcntl_a_second_open_proceeds_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sqlite_registry, "fcntl", None)
    db = tmp_path / "state.db"
    with SqliteArtifactRegistry(db), SqliteArtifactRegistry(db):
        pass


@pytest.mark.parametrize("code", [errno.ENOLCK, errno.ENOTSUP])
def test_a_file_system_without_flock_warns_and_opens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    code: int,
) -> None:
    class NoFlock:
        LOCK_EX, LOCK_NB, LOCK_UN = fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN

        @staticmethod
        def flock(_fd: int, _op: int) -> None:
            raise OSError(code, os.strerror(code))

    monkeypatch.setattr(sqlite_registry, "fcntl", NoFlock)
    with SqliteArtifactRegistry(tmp_path / "state.db"):
        pass
    assert "without the single-writer lock" in caplog.text


def test_a_symlinked_lock_file_is_refused_and_its_target_untouched(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere"
    (tmp_path / "state.db.writer-lock").symlink_to(target)
    with pytest.raises(OSError) as caught:
        SqliteArtifactRegistry(tmp_path / "state.db")
    assert caught.value.errno == errno.ELOOP
    assert not target.exists()
    assert not (tmp_path / "state.db").exists(), "a refused open creates no store"


def test_the_lock_file_is_private(tmp_path: Path) -> None:
    old = os.umask(0)
    try:
        SqliteArtifactRegistry(tmp_path / "state.db").close()
    finally:
        os.umask(old)
    mode = stat.S_IMODE((tmp_path / "state.db.writer-lock").stat().st_mode)
    assert mode == 0o600


def test_a_bare_memory_path_takes_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with SqliteArtifactRegistry(":memory:"), SqliteArtifactRegistry(":memory:"):
        pass
    assert not (tmp_path / ":memory:.writer-lock").exists()


def test_the_store_checks_trial_open_works_on_a_copy_of_a_held_store(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    copy = tmp_path / "copy" / "state.db"
    copy.parent.mkdir()
    with SqliteArtifactRegistry(db):
        copy.write_bytes(db.read_bytes())
        trial_writer_open(copy)


def test_the_refusal_pickles_whole() -> None:
    # A round trip of an object this test built: it crosses a process boundary
    # (a pool worker's open) only as pickle.
    error = StoreHeldError("held", "/x/state.db.writer-lock")
    again = pickle.loads(pickle.dumps(error))
    assert (str(again), again.lock_path, again.reason) == (
        "held", "/x/state.db.writer-lock", "store_held",
    )


def _workspace(root: Path) -> Path:
    coherence = root / ".coherence"
    coherence.mkdir(mode=0o700)
    for name in ("tracked.yaml", "strict_mode.yaml"):
        (coherence / name).write_text("- data/**\n")
    return root


def test_a_live_coordinators_store_refuses_a_second_writer(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    db = root / ".coherence" / "state.db"
    server = CoordinatorHTTPServer(root, port=0, instance_id="holder")
    try:
        with pytest.raises(StoreHeldError):
            SqliteArtifactRegistry(db)
        with SqliteArtifactRegistry(db, read_only=True):
            pass
    finally:
        server.shutdown()
    SqliteArtifactRegistry(db).close()


# --- a lock path that is not a plain lock file -------------------------------------


def test_a_directory_at_the_lock_path_refuses_the_open_and_creates_no_store(
    tmp_path: Path,
) -> None:
    """A checkout can carry a directory where the lock file goes. Opening the
    store unlocked there would bring back the second writer this lock stops."""
    (tmp_path / "state.db.writer-lock").mkdir()
    with pytest.raises(OSError) as caught:
        SqliteArtifactRegistry(tmp_path / "state.db")
    assert caught.value.errno == errno.EISDIR
    assert not (tmp_path / "state.db").exists()


def test_a_fifo_at_the_lock_path_refuses_the_open_and_creates_no_store(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "state.db.writer-lock")
    with pytest.raises(OSError):
        SqliteArtifactRegistry(tmp_path / "state.db")
    assert not (tmp_path / "state.db").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root opens a mode-000 file")
def test_a_lock_file_this_user_cannot_open_refuses_the_open(tmp_path: Path) -> None:
    lock = tmp_path / "state.db.writer-lock"
    lock.touch()
    lock.chmod(0)
    try:
        with pytest.raises(PermissionError):
            SqliteArtifactRegistry(tmp_path / "state.db")
    finally:
        lock.chmod(0o600)
    assert not (tmp_path / "state.db").exists()


def test_an_existing_lock_file_is_made_private(tmp_path: Path) -> None:
    lock = tmp_path / "state.db.writer-lock"
    lock.touch()
    lock.chmod(0o644)
    SqliteArtifactRegistry(tmp_path / "state.db").close()
    assert stat.S_IMODE(lock.stat().st_mode) == 0o600


# --- a probe of the lock is not a holder -------------------------------------------


def test_an_open_that_meets_a_brief_shared_probe_is_granted(tmp_path: Path) -> None:
    """The store check probes the lock with a shared lock it drops at once. A
    writer open that lands in that instant waits it out instead of refusing."""
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    probe = os.open(tmp_path / "state.db.writer-lock", os.O_RDONLY)
    fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
    release = threading.Timer(0.03, os.close, args=(probe,))
    release.start()
    try:
        SqliteArtifactRegistry(db).close()
    finally:
        release.join()


# --- an interrupt as the lock is taken ----------------------------------------------


class _InterruptAfterLocking:
    """``fcntl`` whose exclusive flock locks, then raises KeyboardInterrupt."""

    LOCK_EX, LOCK_NB, LOCK_SH, LOCK_UN = fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_SH, fcntl.LOCK_UN

    @staticmethod
    def flock(fd: int, op: int) -> None:
        fcntl.flock(fd, op)
        if op & fcntl.LOCK_EX:
            raise KeyboardInterrupt


def test_an_interrupt_as_the_lock_is_taken_frees_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "state.db"
    monkeypatch.setattr(sqlite_registry, "fcntl", _InterruptAfterLocking)
    with pytest.raises(KeyboardInterrupt) as kept:
        SqliteArtifactRegistry(db)
    monkeypatch.undo()
    assert _run_child(_TRY_OPEN, str(db)) == "granted"
    assert kept.value is not None


def test_a_flock_that_fails_for_another_reason_refuses_the_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a file system whose flock does not work opens unlocked; any other
    flock failure leaves the lock untaken, so the store is not opened."""

    class BrokenFlock:
        LOCK_EX, LOCK_NB, LOCK_UN = fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN

        @staticmethod
        def flock(_fd: int, _op: int) -> None:
            raise OSError(errno.EBADF, os.strerror(errno.EBADF))

    monkeypatch.setattr(sqlite_registry, "fcntl", BrokenFlock)
    with pytest.raises(OSError) as caught:
        SqliteArtifactRegistry(tmp_path / "state.db")
    assert caught.value.errno == errno.EBADF
    assert not (tmp_path / "state.db").exists()
