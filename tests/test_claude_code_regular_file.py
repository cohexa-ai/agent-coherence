# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The shared regular-file reader (#498).

Every read under ``.coherence`` that can run without a start, and every
workspace-file hash, goes through it, so each way a path can stop being a
regular file must be answered at once, without following a link, blocking on
a FIFO or reading a device. A FIFO case runs its read in a thread with a
deadline and releases the FIFO afterwards, so a regression fails its test
instead of hanging the run.
"""

from __future__ import annotations

import builtins
import errno
import io
import os
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from ccs.adapters.claude_code.regular_file import (
    LinkRefused,
    NotRegularFile,
    ReadStatus,
    RegularFileRead,
    open_regular,
    read_regular_file,
    read_state_file,
)

_DEADLINE_SEC = 10

_DATA = b"line one\r\nline two\n\xff"


def _release_fifo(path: Path) -> None:
    """Give a reader that may be blocked in ``open`` a writer, then remove
    the FIFO, so a regression ends instead of parking a thread forever."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return
    try:
        path.unlink(missing_ok=True)
    finally:
        os.close(fd)


def _bounded(call: Callable[[], RegularFileRead], fifo: Path | None = None) -> RegularFileRead:
    """Run ``call`` in a thread with a deadline; fail if it blocks."""
    answers: list[RegularFileRead] = []
    reader = threading.Thread(target=lambda: answers.append(call()), daemon=True)
    try:
        reader.start()
        reader.join(timeout=_DEADLINE_SEC)
        assert not reader.is_alive(), f"the read blocked for {_DEADLINE_SEC}s"
    finally:
        if fifo is not None:
            _release_fifo(fifo)
    assert len(answers) == 1
    return answers[0]


def _make(shape: str, tmp_path: Path) -> tuple[Path, Path | None]:
    """Build ``shape`` and return the path to read plus any FIFO to release."""
    path = tmp_path / "file"
    target = tmp_path / "target"
    fifo: Path | None = None
    if shape in ("regular", "over_cap", "at_cap"):
        path.write_bytes({"regular": _DATA, "over_cap": b"12345", "at_cap": b"1234"}[shape])
    elif shape == "link_to_file":
        target.write_bytes(_DATA)
        path.symlink_to(target)
    elif shape == "dangling":
        path.symlink_to(tmp_path / "gone")
    elif shape in ("fifo", "link_to_fifo"):
        fifo = tmp_path / "pipe"
        os.mkfifo(fifo, 0o600)
        if shape == "fifo":
            path = fifo
        else:
            path.symlink_to(fifo)
    elif shape == "dev_null":
        path = Path(os.devnull)
    elif shape == "link_to_dev_null":
        path.symlink_to(os.devnull)
    elif shape == "directory":
        path.mkdir()
    elif shape == "leaf_loop":
        path.symlink_to(path)
    elif shape == "eacces":
        path.write_bytes(_DATA)
        path.chmod(0)
    return path, fifo


# (shape, status, errno, data). Data only for READ; a cap of 4 bytes everywhere.
_SHAPES = [
    ("absent", ReadStatus.ABSENT, errno.ENOENT, None),
    ("regular", ReadStatus.READ, None, _DATA),
    ("link_to_file", ReadStatus.LINK, errno.ELOOP, None),
    ("dangling", ReadStatus.LINK, errno.ELOOP, None),
    ("link_to_fifo", ReadStatus.LINK, errno.ELOOP, None),
    ("fifo", ReadStatus.NOT_REGULAR, errno.EINVAL, None),
    ("dev_null", ReadStatus.NOT_REGULAR, errno.EINVAL, None),
    ("link_to_dev_null", ReadStatus.LINK, errno.ELOOP, None),
    ("directory", ReadStatus.NOT_REGULAR, errno.EISDIR, None),
    ("leaf_loop", ReadStatus.LINK, errno.ELOOP, None),
    ("eacces", ReadStatus.UNREADABLE, errno.EACCES, None),
    ("over_cap", ReadStatus.TOO_LARGE, None, None),
    ("at_cap", ReadStatus.READ, None, b"1234"),
]


@pytest.mark.parametrize(("shape", "status", "code", "data"), _SHAPES, ids=[row[0] for row in _SHAPES])
def test_read_regular_file_classifies_each_path(
    shape: str, status: ReadStatus, code: int | None, data: bytes | None, tmp_path: Path
) -> None:
    """Each shape gets its own status, and the ``OSError`` behind a refusal
    is kept, so a caller can rebuild the text it prints today. A link is
    told by ``lstat``, never by the errno: macOS answers ENOTDIR for some
    links. A directory carries EISDIR, the errno a plain open reports, so
    the "(EISDIR)" texts survive. Prevents a read through a link, a block
    on a FIFO, an endless read of a device, or a read past the cap."""
    if shape == "eacces" and os.geteuid() == 0:
        pytest.skip("root reads a mode-000 file")
    path, fifo = _make(shape, tmp_path)
    max_bytes = 4 if shape in ("over_cap", "at_cap") else None
    cap = len(_DATA) if max_bytes is None else max_bytes

    result = _bounded(lambda: read_regular_file(path, max_bytes=max_bytes), fifo)

    assert result.status is status
    assert result.path == path
    assert (None if result.error is None else result.error.errno) == code
    assert result.data == data
    assert result.data is None or len(result.data) <= cap
    if status is ReadStatus.LINK:
        assert isinstance(result.error, LinkRefused)
    if status is ReadStatus.NOT_REGULAR:
        assert isinstance(result.error, NotRegularFile)


def test_read_regular_file_refuses_a_file_that_grew_past_the_cap_while_it_was_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The size on the descriptor is checked before the read, and the read
    stops one byte past the cap, so a file that grows in between is still
    refused. The growth is stood in for by an ``fstat`` that reports the
    size the file had before. Prevents an unbounded read of a file that
    grows after the check."""
    path = tmp_path / "file"
    path.write_bytes(b"12345")
    real_fstat = os.fstat

    def fstat_before_growth(fd: int) -> os.stat_result:
        fields = list(real_fstat(fd))
        fields[6] = 4  # st_size
        return os.stat_result(fields)

    monkeypatch.setattr(os, "fstat", fstat_before_growth)

    result = read_regular_file(path, max_bytes=4)

    assert (result.status, result.data) == (ReadStatus.TOO_LARGE, None)


@pytest.mark.parametrize("shape", ["fifo", "link_to_fifo"])
def test_read_regular_file_returns_at_once_on_a_fifo(shape: str, tmp_path: Path) -> None:
    """A FIFO with no writer blocks a plain open forever; the reader opens
    non-blocking and refuses it. Prevents a hook, a policy route or a
    coordinator start parked on a pipe a clone or an archive left behind."""
    path, fifo = _make(shape, tmp_path)

    result = _bounded(lambda: read_regular_file(path), fifo)

    assert result.status in (ReadStatus.NOT_REGULAR, ReadStatus.LINK)
    assert result.data is None


def test_read_regular_file_never_calls_io_open_or_os_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The volume's all-glob shim patches ``io.open`` and ``builtins.open`` and
    routes a managed path into the volume, which recurses; and lifecycle
    tests count ``os.close`` calls by descriptor number. So a read goes
    through ``os.open`` and ``io.FileIO`` only, and the descriptor is still
    closed. Prevents a recursion under the shim and a stray close that a
    lock descriptor's number later collides with."""
    path = tmp_path / "file"
    path.write_bytes(_DATA)
    coherence_dir = tmp_path / ".coherence"
    coherence_dir.mkdir()
    (coherence_dir / "hook.secret").write_bytes(b"secret")
    opened: list[int] = []
    closes: list[int] = []
    real_open, real_close = os.open, os.close

    def recording_open(*args: object, **kwargs: object) -> int:
        fd = real_open(*args, **kwargs)  # type: ignore[arg-type]
        opened.append(fd)
        return fd

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"io.open or builtins.open called with {args!r}")

    def recording_close(fd: int) -> None:
        closes.append(fd)
        real_close(fd)

    with monkeypatch.context() as patched:
        patched.setattr(io, "open", refuse)
        patched.setattr(builtins, "open", refuse)
        patched.setattr(os, "open", recording_open)
        patched.setattr(os, "close", recording_close)
        result = read_regular_file(path)
        secret = read_state_file(coherence_dir, "hook.secret")

    assert (result.status, result.data) == (ReadStatus.READ, _DATA)
    assert (secret.status, secret.data) == (ReadStatus.READ, b"secret")
    assert closes == []
    assert len(opened) == 2
    for fd in opened:
        with pytest.raises(OSError) as caught:
            os.fstat(fd)
        assert caught.value.errno == errno.EBADF


@pytest.mark.parametrize("shape", ["link_to_dir", "dangling"])
def test_read_state_file_refuses_a_linked_coherence_dir(
    shape: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``O_NOFOLLOW`` guards the last component only, so a leaf read through
    a linked ``.coherence`` would read the directory it names: another
    workspace's secret, sent as this one's bearer. The state-file read looks
    at ``.coherence`` itself first and opens nothing under a link. Prevents
    a client reading state from the directory a cloned link names."""
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "hook.secret").write_bytes(b"another workspace's secret")
    coherence_dir = tmp_path / "ws" / ".coherence"
    coherence_dir.parent.mkdir()
    coherence_dir.symlink_to(target if shape == "link_to_dir" else tmp_path / "gone")
    opened: list[object] = []
    real_open = os.open

    def recording_open(path: object, *args: object, **kwargs: object) -> int:
        opened.append(path)
        return real_open(path, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", recording_open)

    result = read_state_file(coherence_dir, "hook.secret")

    assert result.status is ReadStatus.LINK
    assert result.path == coherence_dir
    assert result.data is None
    assert isinstance(result.error, LinkRefused)
    assert result.error.errno == errno.ELOOP
    assert opened == []


def test_read_state_file_reads_a_regular_file_in_a_real_directory(tmp_path: Path) -> None:
    """The guard on ``.coherence`` leaves a real directory alone: its files
    read, and an absent one answers absent, not a link."""
    coherence_dir = tmp_path / ".coherence"
    coherence_dir.mkdir()
    (coherence_dir / "hook.secret").write_bytes(b"secret")

    assert read_state_file(coherence_dir, "hook.secret").data == b"secret"
    assert read_state_file(coherence_dir, "absent").status is ReadStatus.ABSENT
    assert read_state_file(tmp_path / "no-coherence", "hook.secret").status is ReadStatus.ABSENT


def test_open_regular_refuses_dev_null_given_directly() -> None:
    """A device is no regular file even when no link leads to it: copying
    ``/dev/zero`` would never end. ``/dev/null`` stands in for it, so a
    regression answers wrongly instead of filling a disk. Prevents a store
    check or a policy read that copies or reads a device given by name."""
    with pytest.raises(NotRegularFile) as caught:
        open_regular(Path(os.devnull))

    assert isinstance(caught.value, OSError)
    assert caught.value.errno == errno.EINVAL


def test_open_regular_returns_a_descriptor_for_a_regular_file(tmp_path: Path) -> None:
    """The open itself, for a caller that locks or writes: a regular file
    opens with the caller's flags and mode, and a dangling link is refused
    without creating the file it names."""
    path = tmp_path / "file.lock"
    fd = open_regular(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        assert os.fstat(fd).st_mode & 0o777 == 0o600
    finally:
        os.close(fd)
    link = tmp_path / "linked.lock"
    link.symlink_to(tmp_path / "outside.lock")

    with pytest.raises(LinkRefused):
        open_regular(link, os.O_CREAT | os.O_RDWR, 0o600)

    assert not (tmp_path / "outside.lock").exists()


def test_flags_are_read_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A platform without ``O_NONBLOCK`` or ``O_NOFOLLOW`` (tests delete
    them) still refuses a link: an ``lstat`` decides it, and the flags are
    looked up on each call, not when the module is imported. Prevents an
    ``AttributeError`` there, or a read through the link."""
    link = tmp_path / "file"
    link.symlink_to(os.devnull)
    regular = tmp_path / "regular"
    regular.write_bytes(_DATA)
    monkeypatch.delattr(os, "O_NONBLOCK")
    monkeypatch.delattr(os, "O_NOFOLLOW")

    assert read_regular_file(link).status is ReadStatus.LINK
    with pytest.raises(LinkRefused):
        open_regular(link)
    assert read_regular_file(regular).data == _DATA
