# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The start check on ``.coherence`` and its refusal texts (#498).

``.coherence`` must be absent or a real directory, with no symlink, FIFO,
socket or device directly inside it: a cloned repository can commit a link
there, and a start would then read or write whatever the link names. The
check must answer every shape from ``lstat`` and the directory listing
alone, so it opens no file, follows no link and runs no git command. A FIFO
case runs the check in a thread with a deadline and releases the FIFO
afterwards, so a regression fails its test instead of hanging the run.

The texts are wire text the Node coordinator and the plugin shims mirror, so
each builder is pinned to a full literal.
"""

from __future__ import annotations

import builtins
import io
import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

from ccs.adapters.claude_code.state_dir import (
    check_coherence_dir,
    dir_link_refusal,
    entry_link_refusal,
    lock_link_refusal,
    not_a_directory_refusal,
    policy_link_refusal,
    special_entry_refusal,
)
from ccs.core.exceptions import (
    COHERENCE_DIR_REFUSED_REASON,
    COHERENCE_LINK_KIND,
    COHERENCE_NOT_A_DIRECTORY_KIND,
    COHERENCE_SPECIAL_ENTRY_KIND,
    CoherenceDirRefused,
    CoordinatorUnreachable,
)

_DEADLINE_SEC = 5

#: Names a link may take: what a start writes, what it reads, a store
#: sidecar, a policy file, a name no coordinator uses, and one a ``.*`` glob
#: misses.
_ENTRY_NAMES = [
    "server.pid", "hook.secret", ".gitignore", "state.db-wal", "tracked.yaml", "notes.txt", "..x",
]


def _release_fifo(path: Path) -> None:
    """Give anything blocked opening ``path`` a writer, then remove it."""
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return
    try:
        path.unlink(missing_ok=True)
    finally:
        os.close(fd)


def _check_bounded(root: Path, fifo: Path | None = None) -> CoherenceDirRefused | None:
    """Run the check in a thread with a deadline: the refusal, or None for a pass."""
    answers: list[CoherenceDirRefused | None] = []

    def check() -> None:
        try:
            check_coherence_dir(root)
        except CoherenceDirRefused as exc:
            answers.append(exc)
        else:
            answers.append(None)

    worker = threading.Thread(target=check, daemon=True)
    try:
        worker.start()
        worker.join(timeout=_DEADLINE_SEC)
        assert not worker.is_alive(), f"the check blocked for {_DEADLINE_SEC}s"
    finally:
        if fifo is not None:
            _release_fifo(fifo)
    assert len(answers) == 1
    return answers[0]


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "ws"
    coherence_dir = root / ".coherence"
    coherence_dir.mkdir(parents=True)
    (coherence_dir / "state.db").write_bytes(b"")
    return root, coherence_dir


# Each builder: the refused path, and the expected kind, offending path and text.
_Shape = Callable[[Path], tuple[Path, Path | None, Path | None]]


def _absent(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    root = tmp_path / "ws"
    root.mkdir()
    return root, None, None


def _real_directory(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    root, coherence_dir = _workspace(tmp_path)
    (coherence_dir / "server.pid").write_text("123\n")
    return root, None, None


def _real_subdirectory(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    # A mounted disk's lost+found is one; neither coordinator creates one.
    root, coherence_dir = _workspace(tmp_path)
    (coherence_dir / "lost+found").mkdir()
    return root, None, None


def _hard_link(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    # Not a link by lstat, and git cannot clone one: outside the rule.
    root, coherence_dir = _workspace(tmp_path)
    outside = tmp_path / "outside.secret"
    outside.write_text("x")
    os.link(outside, coherence_dir / "hook.secret")
    return root, None, None


def _link_nested_in_a_subdirectory(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    # Only entries directly inside .coherence count: a start opens none deeper.
    root, coherence_dir = _workspace(tmp_path)
    (coherence_dir / "sub").mkdir()
    (coherence_dir / "sub" / "server.pid").symlink_to(tmp_path / "gone")
    return root, None, None


def _coherence_linked_to_a_directory(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    root = tmp_path / "ws"
    root.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (root / ".coherence").symlink_to(target, target_is_directory=True)
    return root, None, root / ".coherence"


def _coherence_dangling(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".coherence").symlink_to(tmp_path / "gone")
    return root, None, root / ".coherence"


def _coherence_regular_file(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    root = tmp_path / "ws"
    root.mkdir()
    (root / ".coherence").write_text("x")
    return root, None, root / ".coherence"


def _coherence_fifo(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    root = tmp_path / "ws"
    root.mkdir()
    os.mkfifo(root / ".coherence", 0o600)
    return root, root / ".coherence", root / ".coherence"


def _fifo_entry(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
    root, coherence_dir = _workspace(tmp_path)
    os.mkfifo(coherence_dir / "tracked.yaml", 0o600)
    return root, coherence_dir / "tracked.yaml", coherence_dir / "tracked.yaml"


def _entry_linked_to(how: str) -> _Shape:
    def build(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
        root, coherence_dir = _workspace(tmp_path)
        link = coherence_dir / "server.pid"
        fifo = None
        if how == "fifo":
            fifo = tmp_path / "pipe"
            os.mkfifo(fifo, 0o600)
            link.symlink_to(fifo)
        else:
            target = tmp_path / "elsewhere"
            target.mkdir()
            link.symlink_to(target, target_is_directory=True)
        return root, fifo, link
    return build


def _entry_link(name: str, dangling: bool) -> _Shape:
    def build(tmp_path: Path) -> tuple[Path, Path | None, Path | None]:
        root, coherence_dir = _workspace(tmp_path)
        target = tmp_path / "target"
        if not dangling:
            target.write_text("not this workspace's\n")
        (coherence_dir / name).symlink_to(target)
        return root, None, coherence_dir / name
    return build


_PASSES = [
    ("absent", _absent),
    ("real_directory", _real_directory),
    ("real_subdirectory", _real_subdirectory),
    ("hard_link", _hard_link),
    ("link_nested_in_a_subdirectory", _link_nested_in_a_subdirectory),
]
_REFUSALS = [
    ("coherence_link_to_a_directory", _coherence_linked_to_a_directory, COHERENCE_LINK_KIND),
    ("coherence_dangling_link", _coherence_dangling, COHERENCE_LINK_KIND),
    ("coherence_regular_file", _coherence_regular_file, COHERENCE_NOT_A_DIRECTORY_KIND),
    ("coherence_fifo", _coherence_fifo, COHERENCE_NOT_A_DIRECTORY_KIND),
    *[
        (f"{name}_{'dangling' if dangling else 'existing'}", _entry_link(name, dangling), COHERENCE_LINK_KIND)
        for name in _ENTRY_NAMES
        for dangling in (False, True)
    ],
    ("entry_link_to_a_fifo", _entry_linked_to("fifo"), COHERENCE_LINK_KIND),
    ("entry_link_to_a_directory", _entry_linked_to("directory"), COHERENCE_LINK_KIND),
    ("fifo_entry", _fifo_entry, COHERENCE_SPECIAL_ENTRY_KIND),
]


def _expected_text(kind: str, refused: Path) -> str:
    if kind == COHERENCE_NOT_A_DIRECTORY_KIND:
        return not_a_directory_refusal(refused)
    if kind == COHERENCE_SPECIAL_ENTRY_KIND:
        return special_entry_refusal(refused)
    if refused.name == ".coherence":
        return dir_link_refusal(refused)
    return entry_link_refusal(refused)


@pytest.mark.parametrize(("shape", "build"), _PASSES, ids=[row[0] for row in _PASSES])
def test_check_coherence_dir_passes_a_directory_a_start_may_use(
    shape: str, build: _Shape, tmp_path: Path
) -> None:
    """An absent ``.coherence`` (a start creates it), a real directory, a
    real subdirectory, a hard link and a link nested deeper all pass.
    Prevents a refusal of a mount's ``lost+found``, of the directory a
    lifecycle test plants at ``server.pid``, or of anything no start opens."""
    root, fifo, _ = build(tmp_path)

    assert _check_bounded(root, fifo) is None


@pytest.mark.parametrize(("shape", "build", "kind"), _REFUSALS, ids=[row[0] for row in _REFUSALS])
def test_check_coherence_dir_refuses_each_shape_a_start_must_not_use(
    shape: str, build: _Shape, kind: str, tmp_path: Path
) -> None:
    """A linked or dangling ``.coherence``, a non-directory one, and a link
    (dangling or not, to a file, a FIFO or a directory) or a FIFO directly
    inside it each raise the one typed refusal, whose ``kind`` names the rule
    and whose ``path`` names what was refused. Every entry counts, not a list
    of known names. Prevents a start that writes into, joins through, or
    blocks on what a cloned link names."""
    root, fifo, refused = build(tmp_path)
    assert refused is not None

    exc = _check_bounded(root, fifo)

    assert exc is not None, f"{shape} passed the check"
    assert exc.reason == COHERENCE_DIR_REFUSED_REASON
    assert exc.kind == kind
    assert exc.path == str(refused)
    assert str(exc) == _expected_text(kind, refused)
    assert not isinstance(exc, CoordinatorUnreachable)


def test_the_refusal_names_the_first_link_in_sorted_order_and_counts_the_rest(tmp_path: Path) -> None:
    """Several links are named by the first in sorted order plus a count,
    so the message stays bounded and the same on every run; links are
    answered before a FIFO. Prevents a refusal that names a different link
    each run, or prints a listing as long as the directory."""
    root, coherence_dir = _workspace(tmp_path)
    for name in ("tracked.yaml", "hook.secret", "server.pid"):
        (coherence_dir / name).symlink_to(tmp_path / "gone")
    os.mkfifo(coherence_dir / "0-fifo", 0o600)

    exc = _check_bounded(root, coherence_dir / "0-fifo")

    assert exc is not None
    assert exc.kind == COHERENCE_LINK_KIND
    assert exc.path == str(coherence_dir / "hook.secret")
    assert str(exc) == (
        f"refusing {coherence_dir}/hook.secret: it is a symlink, and .coherence must be a real "
        "directory with no link directly inside it. Replace the link with a copy of the file it "
        f"names, or mount a directory at {coherence_dir}. 2 more entries in .coherence are links too."
    )


def test_the_check_opens_no_file_and_runs_no_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The check is ``lstat`` and a listing: no open (a FIFO would block it,
    and a link would be followed) and no git command (a rule built on what
    git tracks was rejected: a clone's links are refused whatever git
    records), and no ``readlink`` (a refusal never prints a link's target).
    Prevents a check that reads what it is meant to refuse."""
    clean_root, coherence_dir = _workspace(tmp_path / "clean")
    (coherence_dir / "server.pid").write_text("123\n")
    linked_root, linked_dir = _workspace(tmp_path / "linked")
    (linked_dir / "hook.secret").symlink_to(tmp_path / "gone")
    fifo_root, fifo_dir = _workspace(tmp_path / "fifo")
    os.mkfifo(fifo_dir / "audit.log", 0o600)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"the check called an open, readlink or subprocess with {args!r}")

    try:
        with monkeypatch.context() as patched:
            for owner, name in [
                (os, "open"), (io, "open"), (builtins, "open"), (os, "readlink"),
                (subprocess, "run"), (subprocess, "Popen"),
            ]:
                patched.setattr(owner, name, refuse)
            check_coherence_dir(clean_root)
            with pytest.raises(CoherenceDirRefused) as linked:
                check_coherence_dir(linked_root)
            with pytest.raises(CoherenceDirRefused) as special:
                check_coherence_dir(fifo_root)
    finally:
        _release_fifo(fifo_dir / "audit.log")

    assert linked.value.kind == COHERENCE_LINK_KIND
    assert special.value.kind == COHERENCE_SPECIAL_ENTRY_KIND


@pytest.mark.skipif(os.geteuid() == 0, reason="root lists a mode-000 directory")
def test_an_unlistable_coherence_passes_the_check(tmp_path: Path) -> None:
    """A ``.coherence`` this user cannot list passes: lifecycle repairs its
    mode to 0700 and checks again, and git cannot record a directory mode, so
    a clone cannot produce one. Prevents a start refused for a mode the start
    itself repairs, which tests/test_claude_code_cli.py pins as spawning."""
    root, coherence_dir = _workspace(tmp_path)
    (coherence_dir / "server.pid").symlink_to(tmp_path / "gone")
    coherence_dir.chmod(0)
    try:
        assert _check_bounded(root) is None
    finally:
        coherence_dir.chmod(0o700)

    with pytest.raises(CoherenceDirRefused):
        check_coherence_dir(root)


def test_a_link_above_coherence_is_not_refused(tmp_path: Path) -> None:
    """Only ``.coherence`` and its entries count: a root reached through a
    link (macOS ``/tmp`` is one) passes. Prevents a refusal of every
    workspace under a linked home or temp directory."""
    root, coherence_dir = _workspace(tmp_path)
    (coherence_dir / "server.pid").write_text("123\n")
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)

    assert _check_bounded(alias) is None


def test_a_name_with_control_characters_is_kept_raw_in_the_exception(tmp_path: Path) -> None:
    """The exception carries the name as the directory holds it; escaping
    is the printer's job (``escape_nonprintable``), so a CLI and a host that
    match ``path`` both see the real name. Prevents a ``path`` that names no
    entry, or an escape applied twice."""
    root, coherence_dir = _workspace(tmp_path)
    name = "evil\x1b[2Jname"
    (coherence_dir / name).symlink_to(tmp_path / "gone")

    exc = _check_bounded(root)

    assert exc is not None
    assert exc.path == str(coherence_dir / name)
    assert f"refusing {coherence_dir}/{name}: it is a symlink" in str(exc)


def test_a_fifo_entry_is_refused_without_being_opened(tmp_path: Path) -> None:
    """A FIFO with no writer blocks a plain open forever, and a start opens
    ``state.db``, ``audit.log`` and the policy files that way. The check
    reads the entry's type from the listing, so it answers at once. Prevents
    a start, or the check itself, parked on a pipe an archive left."""
    root, coherence_dir = _workspace(tmp_path)
    (coherence_dir / "state.db").unlink()
    fifo = coherence_dir / "state.db"
    os.mkfifo(fifo, 0o600)

    exc = _check_bounded(root, fifo)

    assert exc is not None
    assert (exc.kind, exc.path) == (COHERENCE_SPECIAL_ENTRY_KIND, str(fifo))


def test_each_refusal_text_is_the_shared_literal() -> None:
    """Both coordinators and the plugin shims print these texts, so each is
    pinned whole: a reword here must be a reword there. Every link text names
    the way out (a copy, or a mount), and the store's names say to move its
    sidecars with it. Prevents a refusal that leaves a person without the
    fix, and drift between the runtimes."""
    coherence_dir = Path("/w/.coherence")

    assert dir_link_refusal(coherence_dir) == (
        "refusing /w/.coherence: it is a symlink, and .coherence must be a real directory. "
        "Replace the link with a real directory and copy in the files you want to keep "
        "(move state.db together with its -wal and -shm), or mount a directory at /w/.coherence."
    )
    assert entry_link_refusal(coherence_dir / "hook.secret") == (
        "refusing /w/.coherence/hook.secret: it is a symlink, and .coherence must be a real "
        "directory with no link directly inside it. Replace the link with a copy of the file it "
        "names, or mount a directory at /w/.coherence."
    )
    assert entry_link_refusal(coherence_dir / "state.db-wal", more=1) == (
        "refusing /w/.coherence/state.db-wal: it is a symlink, and .coherence must be a real "
        "directory with no link directly inside it. Replace the link with a copy of the file it "
        "names (move state.db together with its -wal and -shm), or mount a directory at "
        "/w/.coherence. 1 more entry in .coherence is a link too."
    )
    assert entry_link_refusal(coherence_dir / "workspace.db", more=3).endswith(
        "names (move state.db together with its -wal and -shm), or mount a directory at "
        "/w/.coherence. 3 more entries in .coherence are links too."
    )
    assert not_a_directory_refusal(coherence_dir) == (
        "refusing /w/.coherence: it is not a directory. Move it aside; a start then creates "
        ".coherence as a real directory."
    )
    assert special_entry_refusal(coherence_dir / "tracked.yaml") == (
        "refusing /w/.coherence/tracked.yaml: it is a FIFO, socket or device, which no coordinator "
        "creates, and opening it can block a start. Remove it, then start again."
    )
    assert special_entry_refusal(coherence_dir / "audit.log", more=2).endswith(
        "Remove it, then start again. 2 more entries in .coherence are FIFOs, sockets or devices too."
    )
    assert special_entry_refusal(coherence_dir / "audit.log", more=1).endswith(
        "Remove it, then start again. 1 more entry in .coherence is a FIFO, socket or device too."
    )
    assert policy_link_refusal(".coherence/tracked.yaml") == (
        ".coherence/tracked.yaml is a symlink; nothing was written. Replace it with a copy of the "
        "file it names, or mount a directory at .coherence."
    )
    assert lock_link_refusal(".coherence/tracked.yaml") == (
        ".coherence/tracked.yaml.lock is a symlink; nothing was written. Remove the link."
    )
