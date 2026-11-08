# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume.write()`` returns a ``WriteResult`` (#526).

``disk_differs`` relays one fact the coordinator reports when it admits the
write's grant request: the file held bytes it had not recorded, and this write
replaced them. It is positive evidence only; ``False`` says the coordinator did
not report it, which it also does not on paths it does not track. The volume
reads it without moving the baseline its foreign-edit guard compares against.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import ccs.adapters as adapters
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume, WriteResult
from ccs.core.exceptions import CoherenceDegradedWarning, StaleView
from tests.adapters.test_coherent_volume import _fail_first_resolve
from tests.coordinator_helpers import serve_in_process

_X = "data/x.txt"
_MANAGED = ("data/**",)


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``data/**`` running in this process."""
    (tmp_path / "data").mkdir()
    running = serve_in_process(tmp_path, "write-result")
    yield running
    running.shutdown()


def _volume(root: Path, cfg: LifecycleConfig, **kwargs: object) -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, config=cfg, **kwargs)


def _edited_outside(tmp_path: Path, cfg: LifecycleConfig) -> None:
    """A peer commits ``_X`` and lets go of it; then the file is changed on
    disk without the coordinator."""
    peer = _volume(tmp_path, cfg)
    peer.write(_X, b"committed by the peer")
    peer.release(_X)
    (tmp_path / _X).write_bytes(b"edited outside the coordinator")


def test_a_write_over_bytes_the_coordinator_did_not_record_reports_it(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The volume never read the file, so it has no baseline to deny the
    write on, and the write replaces the out-of-band edit. The coordinator saw
    the disk differ from what it records, and the result says so.

    Prevents the coordinator's report of a replaced edit being dropped."""
    _edited_outside(tmp_path, fast_cfg)
    writer = _volume(tmp_path, fast_cfg)

    result = writer.write(_X, b"from the writer")

    assert result == WriteResult(path=_X, disk_differs=True)
    assert (tmp_path / _X).read_bytes() == b"from the writer"


def test_a_plain_write_reports_no_difference(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """Prevents a result that reports a difference the coordinator did not see."""
    writer = _volume(tmp_path, fast_cfg)

    assert writer.write(_X, b"v1") == WriteResult(path=_X, disk_differs=False)
    assert writer.write(_X, b"v2") == WriteResult(path=_X, disk_differs=False)


def test_a_write_of_bytes_already_on_disk_returns_a_result(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The no-op write still returns, so a caller can read every result alike."""
    writer = _volume(tmp_path, fast_cfg)
    writer.write(_X, b"v1")

    assert writer.write(_X, b"v1") == WriteResult(path=_X, disk_differs=False)


def test_a_write_of_an_untracked_path_reports_no_difference(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The coordinator compares nothing on a path it does not track."""
    (tmp_path / "notes").mkdir()
    writer = _volume(tmp_path, fast_cfg)

    assert writer.write("notes/todo.txt", b"v1") == WriteResult(path="notes/todo.txt", disk_differs=False)


def test_a_detached_degrade_volume_returns_a_result_that_reports_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no coordinator nothing was compared, so the result says nothing."""
    writer = _volume(tmp_path, fast_cfg, on_error="degrade")
    writer._after_fork()
    _fail_first_resolve(monkeypatch)

    with pytest.warns(CoherenceDegradedWarning):
        result = writer.write(_X, b"best effort")

    assert writer.is_attached is False
    assert result == WriteResult(path=_X, disk_differs=False)


def test_reporting_a_difference_leaves_the_foreign_edit_guard_in_place(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The report moves nothing the next write's check compares against: an
    edit made outside the coordinator after this write is still denied.

    Prevents a reported difference absolving the next one."""
    _edited_outside(tmp_path, fast_cfg)
    writer = _volume(tmp_path, fast_cfg)
    assert writer.write(_X, b"from the writer").disk_differs is True
    (tmp_path / _X).write_bytes(b"edited outside again")

    with pytest.raises(StaleView):
        writer.write(_X, b"over the second edit")
    assert (tmp_path / _X).read_bytes() == b"edited outside again"


def test_write_result_is_exported_with_the_volume() -> None:
    assert adapters.WriteResult is WriteResult
