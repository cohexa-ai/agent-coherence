"""A client can tell "no coordinator port is known" from every other failure
to resolve a coordinator, by type.

A hook that denies when a coordinator it knows of does not answer still admits
when none is known: no ``.coherence``, no ``server.pid``, or a ``server.pid``
that a clean stop left with only its pid line. ``never_sent`` cannot separate
those from a coordinator that died (no port and a timeout both read ``False``),
so the no-port case raises its own subclass of :class:`CoordinatorUnavailable`.
Every caller that catches the base class keeps its exit code and output.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from ccs.adapters.claude_code.coordinator_pin import PID_FILE_MAX_BYTES, read_pid_file
from ccs.adapters.claude_code.regular_file import ReadStatus
from ccs.cli import (
    _coherence_client,
    coherence_coordinator,
    coherence_handoff,
    coherence_hook_client,
    coherence_status,
    coherence_track,
    coherence_untrack,
)
from ccs.cli._coherence_client import (
    CoordinatorPortUnknown,
    CoordinatorUnavailable,
    resolve_endpoint,
)

_SECRET = "s3cret-for-the-test"
_SESSION = "0b6f6c55-5b7e-4d8c-9a51-2f3c1d0e9a7b"


def _coherence(root: Path) -> Path:
    coherence = root / ".coherence"
    coherence.mkdir(mode=0o700, exist_ok=True)
    secret = coherence / "hook.secret"
    secret.write_text(f"{_SECRET}\n")
    secret.chmod(0o600)
    return coherence


def _no_coherence_dir(root: Path) -> None:
    pass


def _no_pid_file(root: Path) -> None:
    _coherence(root)


def _pid_line_only(root: Path) -> None:
    """What a clean stop leaves: the port line dropped, the pid kept."""
    (_coherence(root) / "server.pid").write_text(f"{os.getpid()}\n")


_NO_PORT_KNOWN: dict[str, Callable[[Path], None]] = {
    "no .coherence directory": _no_coherence_dir,
    "no server.pid": _no_pid_file,
    "a pid line only, as a clean stop leaves it": _pid_line_only,
}


def _no_port_message(root: Path) -> str:
    pid_file = root / ".coherence" / "server.pid"
    return (
        "no coordinator running for this workspace "
        f"(no port in {pid_file}); start one with `agent-coherence-coordinator`"
    )


@pytest.mark.parametrize("state", sorted(_NO_PORT_KNOWN))
def test_no_known_port_raises_the_no_port_subclass(tmp_path: Path, state: str) -> None:
    _NO_PORT_KNOWN[state](tmp_path)

    with pytest.raises(CoordinatorPortUnknown) as caught:
        resolve_endpoint(tmp_path)

    assert caught.value.never_sent is False
    assert str(caught.value) == _no_port_message(tmp_path)


def _exited_pid() -> int:
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait(60)
    return child.pid


def test_a_port_line_naming_a_dead_pid_stays_the_plain_class(tmp_path: Path) -> None:
    """A port is known, so a hook that cannot reach it must not read this as
    "no coordinator"."""
    pid_file = _coherence(tmp_path) / "server.pid"
    pid_file.write_text(f"{_exited_pid()}\n41234\nbackend=node\n")

    with pytest.raises(CoordinatorUnavailable) as caught:
        resolve_endpoint(tmp_path)

    assert type(caught.value) is CoordinatorUnavailable
    assert caught.value.never_sent is True
    assert "is no longer running" in str(caught.value)


def _oversized(root: Path) -> ReadStatus:
    (_coherence(root) / "server.pid").write_text("1\n" * PID_FILE_MAX_BYTES)
    return ReadStatus.TOO_LARGE


def _linked_pid_file(root: Path) -> ReadStatus:
    elsewhere = root / "elsewhere.pid"
    elsewhere.write_text(f"{os.getpid()}\n")
    (_coherence(root) / "server.pid").symlink_to(elsewhere)
    return ReadStatus.LINK


def _unreadable(root: Path) -> ReadStatus:
    pid_file = _coherence(root) / "server.pid"
    pid_file.write_text(f"{os.getpid()}\n41234\n")
    pid_file.chmod(0o000)
    return ReadStatus.UNREADABLE


def _linked_coherence_dir(root: Path) -> ReadStatus:
    """The directory the link names has no ``server.pid``, which says nothing
    about this workspace's coordinator."""
    elsewhere = root / "elsewhere"
    elsewhere.mkdir()
    (root / ".coherence").symlink_to(elsewhere)
    return ReadStatus.LINK


_MAY_HIDE_A_PORT: dict[str, Callable[[Path], ReadStatus]] = {
    "an oversized server.pid": _oversized,
    "a linked server.pid": _linked_pid_file,
    "an unreadable server.pid": _unreadable,
    "a linked .coherence": _linked_coherence_dir,
}


@pytest.mark.parametrize("state", sorted(_MAY_HIDE_A_PORT))
def test_a_pid_file_that_cannot_be_read_stays_the_plain_class(tmp_path: Path, state: str) -> None:
    expected_status = _MAY_HIDE_A_PORT[state](tmp_path)
    if expected_status is ReadStatus.UNREADABLE and os.access(tmp_path / ".coherence" / "server.pid", os.R_OK):
        pytest.skip("this process reads a mode-000 file, as root does")
    # The case under test must be the one the read reports, or a pass here
    # would prove nothing (a root user reads a mode-000 file, for one).
    assert read_pid_file(tmp_path / ".coherence").status is expected_status

    with pytest.raises(CoordinatorUnavailable) as caught:
        resolve_endpoint(tmp_path)

    assert type(caught.value) is CoordinatorUnavailable


def _status(root: Path) -> int:
    return coherence_status.main(["--root", str(root)])


def _status_self_test(root: Path) -> int:
    return coherence_status.main(["--root", str(root), "--self-test"])


def _track(root: Path) -> int:
    return coherence_track.main(["--root", str(root), "plan.md"])


def _untrack(root: Path) -> int:
    return coherence_untrack.main(["--root", str(root), "plan.md"])


def _accept(root: Path) -> int:
    return coherence_handoff.accept_main(["--root", str(root), "--session", _SESSION, "plan.md"])


def _hook_pre_edit(root: Path) -> int:
    return coherence_hook_client.main(["pre-edit", "--root", str(root)])


def _prepare_for_migration(root: Path) -> int:
    return coherence_coordinator._run_prepare_for_migration(root, quiet=False)


_CALLERS: dict[str, Callable[[Path], int]] = {
    "agent-coherence-status": _status,
    "agent-coherence-status --self-test": _status_self_test,
    "agent-coherence-track": _track,
    "agent-coherence-untrack": _untrack,
    "agent-coherence-accept": _accept,
    "agent-coherence-hook-client pre-edit": _hook_pre_edit,
    "agent-coherence-coordinator prepare-for-migration": _prepare_for_migration,
}

_HOOK_PAYLOAD = json.dumps(
    {"session_id": _SESSION, "tool_name": "Edit", "tool_input": {"file_path": "plan.md"}}
)


def _run(
    caller: Callable[[Path], int], root: Path, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = caller(root)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.mark.parametrize("state", sorted(_NO_PORT_KNOWN))
@pytest.mark.parametrize("caller", sorted(_CALLERS))
def test_every_caller_reports_no_known_port_as_it_did_before(
    tmp_path: Path,
    caller: str,
    state: str,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same workspace twice: once raising the subclass, once with the
    subclass swapped back to the plain class it replaced. Exit code, stdout
    and stderr must match byte for byte."""
    root = tmp_path / "ws"
    root.mkdir()
    _NO_PORT_KNOWN[state](root)
    with pytest.raises(CoordinatorPortUnknown):
        resolve_endpoint(root)

    monkeypatch.setattr(sys, "stdin", io.StringIO(_HOOK_PAYLOAD))
    with_subclass = _run(_CALLERS[caller], root, capsys)

    shutil.rmtree(root)
    root.mkdir()
    _NO_PORT_KNOWN[state](root)
    monkeypatch.setattr(sys, "stdin", io.StringIO(_HOOK_PAYLOAD))
    monkeypatch.setattr(_coherence_client, "CoordinatorPortUnknown", CoordinatorUnavailable)
    with_plain_class = _run(_CALLERS[caller], root, capsys)

    assert with_subclass == with_plain_class
