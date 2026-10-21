# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The coordinator's failure envelope on the volume's commit and read legs (#259).

When a route's work raises, the coordinator answers HTTP 200 ``{"ok": false,
"reason": "internal: <Type>"}``. The answer decides nothing, and a commit's
work can raise after the commit was recorded: the CAS and batch routes read
the registry after their commit, and the post-edit route records the version
before it updates the writer's state. So on a commit leg the outcome is
unknown (``CommitUnconfirmed``), never a corruption error, a preempt or a
"nothing was published" hold, each of which tells the caller its write did not
land. On a read the envelope is :class:`CoordinatorFailed`, a typed failure
the caller retries once, not the untyped error a coordinator bug gets.

Each commit case runs twice: ``before``, where the fault fires before the
registry records anything, and ``after``, where the commit is recorded first.
The client cannot tell the two apart, which is why both get the same type.
The coordinator runs in this process so its registry can be made to fail.
"""

from __future__ import annotations

import os
import sqlite3
import warnings
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import (
    CoherenceDegradedWarning,
    CoherenceError,
    CommitUnconfirmed,
    CoordinatorFailed,
    StaleView,
)
from ccs.core.states import MESIState

#: FROZEN duplicate of the envelope's reason for the fault these tests raise
#: (never imported from the code under test).
_ENVELOPE = "internal: OperationalError"
#: FROZEN duplicate of CoordinatorFailed's reason.
_COORDINATOR_FAILED = "coordinator_failed"

_REL = "data/shared.txt"
_OTHER = "data/other.txt"

_WHEN = pytest.mark.parametrize("when", ["before", "after"])
_MODE = pytest.mark.parametrize("mode", ["strict", "degrade"])


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def _seed(tmp_path: Path, rel: str = _REL, content: bytes = b"v1") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _serve_in_process(tmp_path: Path, instance_id: str):
    """A strict coordinator for ``data/**`` running in THIS process, reachable
    through the usual pid file, so a test can reach into its registry. Policy
    is loaded once at construction, so the YAML is written first."""
    from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer

    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    for name in ("tracked.yaml", "strict_mode.yaml"):
        (coherence / name).write_text("- data/**\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id=instance_id)
    server.serve_in_thread()
    (coherence / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
    return server


def _fail_registry(server, monkeypatch: pytest.MonkeyPatch, method: str, when: str) -> None:
    """Make the registry's ``method`` raise the error a locked store raises,
    ``before`` it records anything or ``after`` it has recorded its commit."""
    real = getattr(server.registry, method)

    def failing(*args, **kwargs):
        if when == "after":
            real(*args, **kwargs)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(server.registry, method, failing)


def _recorded_version(server, rel: str) -> int:
    artifact_id = server.registry.lookup_artifact_id_by_name(rel)
    assert artifact_id is not None, f"control: {rel} is registered"
    return server.registry.get_artifact(artifact_id).version


def _assert_landed_iff_after(server, rel: str, before_version: int, when: str) -> None:
    """Control: the ``after`` case really recorded the commit, and the
    ``before`` case really did not, so each test covers the case it names."""
    expected = before_version + 1 if when == "after" else before_version
    assert _recorded_version(server, rel) == expected, f"control: commit recorded ({when})"


@_WHEN
@_MODE
@pytest.mark.parametrize("operation", ["write_cas_at", "write_cas"])
def test_an_occ_commit_answered_with_the_failure_envelope_is_unconfirmed(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    mode: str,
    when: str,
) -> None:
    """An OCC commit answered with the envelope raises ``CommitUnconfirmed``
    in both modes, naming the envelope, and leaves the old bytes on disk (the
    write is CAS-first). The text says the version can move without the
    bytes, so the caller judges by content.

    Prevents the corruption arm's plain ``CoherenceError`` (MCP
    ``internal_error``, ``recover: none``): read as "the write failed", the
    caller abandons a commit the coordinator may have recorded, and a later
    blind retry re-drives it."""
    target = _seed(tmp_path)
    server = _serve_in_process(tmp_path, f"envelope-{operation}-{mode}-{when}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg)
        data, version = vol.read_with_version(_REL)
        assert data == b"v1"
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "commit_cas", when)
            with pytest.raises(CommitUnconfirmed) as raised:
                if operation == "write_cas_at":
                    vol.write_cas_at(_REL, version, b"v2")
                else:
                    vol.write_cas(_REL, lambda _current: b"v2")

        assert _ENVELOPE in str(raised.value)
        assert "content" in str(raised.value), "the text must say to judge by content"
        assert target.read_bytes() == b"v1", "a CAS-first write touched disk"
        _assert_landed_iff_after(server, _REL, version, when)
        assert vol.degradation_count == 0, "an answer was read as an unanswered request"
    finally:
        server.shutdown()


@_WHEN
@_MODE
def test_a_publish_answered_with_the_failure_envelope_is_unconfirmed_never_held(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    when: str,
) -> None:
    """A batch whose commit is answered with the envelope raises
    ``CommitUnconfirmed`` in both modes and writes no file.

    Prevents the HELD ``StaleView``, which says the batch was NOT published:
    in the ``after`` case every member is recorded at its new version, and a
    caller who believes nothing landed rebuilds and republishes over it."""
    target = _seed(tmp_path)
    other = _seed(tmp_path, _OTHER, b"o1")
    server = _serve_in_process(tmp_path, f"envelope-publish-{mode}-{when}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg)
        _data, version = vol.read_with_version(_REL)
        _other, other_version = vol.read_with_version(_OTHER)
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "commit_all", when)
            with pytest.raises(CommitUnconfirmed) as raised:
                vol.atomic_publish([(_REL, version, b"v2"), (_OTHER, other_version, b"o2")])

        assert not isinstance(raised.value, StaleView)
        assert _ENVELOPE in str(raised.value)
        assert "content" in str(raised.value), "the text must say to judge by content"
        assert (target.read_bytes(), other.read_bytes()) == (b"v1", b"o1"), "a file was written"
        _assert_landed_iff_after(server, _REL, version, when)
        _assert_landed_iff_after(server, _OTHER, other_version, when)
    finally:
        server.shutdown()


@_WHEN
@_MODE
def test_a_write_whose_commit_is_answered_with_the_failure_envelope_is_unconfirmed(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    when: str,
) -> None:
    """A ``write()`` whose commit (post-edit) is answered with the envelope
    raises ``CommitUnconfirmed`` in both modes, saying its bytes are on disk,
    and first releases the grant it took.

    Prevents ``CommitPreempted`` ("the commit did not land; reconcile"):
    in the ``after`` case the coordinator recorded the write. Also prevents
    the grant outliving the call: the writer stayed EXCLUSIVE at the
    coordinator, blocking every peer until this volume re-minted."""
    target = _seed(tmp_path)
    server = _serve_in_process(tmp_path, f"envelope-write-{mode}-{when}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg)
        _data, version = vol.read_with_version(_REL)
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "set_artifact_and_content", when)
            with warnings.catch_warnings(record=True) as warned:
                warnings.simplefilter("always")
                with pytest.raises(CommitUnconfirmed) as raised:
                    vol.write(_REL, b"v2")

        assert type(raised.value) is CommitUnconfirmed, type(raised.value)
        assert _ENVELOPE in str(raised.value)
        assert "on disk" in str(raised.value)
        assert target.read_bytes() == b"v2", "control: the write's bytes reached disk"
        _assert_landed_iff_after(server, _REL, version, when)
        artifact_id = server.registry.lookup_artifact_id_by_name(_REL)
        assert not {
            state for state in server.registry.get_state_map(artifact_id).values()
            if state in (MESIState.EXCLUSIVE, MESIState.MODIFIED)
        }, "the write's grant outlived the call"
        assert _REL not in vol._grant_incarnations.get(vol._incarnation, set())
        assert vol.degradation_count == 0
        assert not [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
    finally:
        server.shutdown()


@pytest.mark.parametrize(
    "read", ["read", "read_with_version", "read_with_version_generation"]
)
def test_a_strict_read_answered_with_the_failure_envelope_raises_coordinator_failed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, read: str
) -> None:
    """A strict read whose pre-read is answered with the envelope raises
    :class:`CoordinatorFailed`: a ``CoherenceError``, never a ``StaleView``
    (a caller reads a ``StaleView`` from a versioned read as the split pair),
    keeping the envelope and the "not confirmed as registered" text.

    Prevents the untyped ``CoherenceError`` the corruption arms raise, which
    the MCP mapper answers as ``internal_error``: the shape of a bug, for a
    condition one retry may clear."""
    _seed(tmp_path)
    server = _serve_in_process(tmp_path, f"envelope-{read}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)

        def raising(*_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as faulted:
            faulted.setattr(server, "run_with_watchdog", raising)
            with pytest.raises(CoherenceError) as raised:
                getattr(vol, read)(_REL)

        assert type(raised.value) is CoordinatorFailed, type(raised.value)
        assert raised.value.reason == _COORDINATOR_FAILED
        assert _ENVELOPE in str(raised.value)
        assert "the read is not confirmed as registered" in str(raised.value)
        assert server.registry.lookup_artifact_id_by_name(_REL) is None, "the read registered"
        assert vol.read(_REL) == b"v1", "control: the coordinator recovered"
    finally:
        server.shutdown()


def test_a_degrade_read_answered_with_the_failure_envelope_still_returns_the_bytes(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degrade mode keeps its contract for the envelope: it warns once,
    counts, and returns the bytes at version 0. Only the strict raise is
    typed."""
    _seed(tmp_path)
    server = _serve_in_process(tmp_path, "envelope-degrade-read")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)

        def raising(*_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as faulted:
            faulted.setattr(server, "run_with_watchdog", raising)
            with warnings.catch_warnings(record=True) as warned:
                warnings.simplefilter("always")
                pair = vol.read_with_version(_REL)

        assert pair == (b"v1", 0)
        assert vol.degradation_count == 1
        assert [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
    finally:
        server.shutdown()


def test_a_commit_answered_with_a_reason_outside_the_envelope_stays_untyped(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the envelope's exact prefix is the unknown outcome. Any other
    unclassified reason (what a coordinator that answers prose, like the Node
    one, sends for a definite reject) keeps the fail-closed plain
    ``CoherenceError``: it must never become a retry or an unconfirmed
    commit the caller reconciles."""
    from ccs.adapters import coherent_volume as coherent_volume_module

    _seed(tmp_path)
    server = _serve_in_process(tmp_path, "envelope-prose")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        _data, version = vol.read_with_version(_REL)
        real_post = coherent_volume_module._coordinator_post

        def prose(endpoint, path, payload, **kwargs):
            if path == "/hooks/post-edit-cas":
                return {"ok": False, "reason": "internal error: registry busy"}
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", prose)
        with pytest.raises(CoherenceError) as raised:
            vol.write_cas_at(_REL, version, b"v2")

        assert type(raised.value) is CoherenceError, type(raised.value)
    finally:
        server.shutdown()


def test_a_strict_read_answered_with_a_reason_outside_the_envelope_stays_untyped(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read side of the same rule: a pre-read answered ``ok: false`` with
    a reason that lacks the envelope's prefix keeps the untyped fail-closed
    error. ``CoordinatorFailed`` names one retry, so only the envelope may
    carry it."""
    from ccs.adapters import coherent_volume as coherent_volume_module

    _seed(tmp_path)
    server = _serve_in_process(tmp_path, "envelope-prose-read")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        real_post = coherent_volume_module._coordinator_post

        def prose(endpoint, path, payload, **kwargs):
            if path == "/hooks/pre-read":
                return {"ok": False, "reason": "internal error: registry busy"}
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", prose)
        with pytest.raises(CoherenceError) as raised:
            vol.read_with_version(_REL)

        assert type(raised.value) is CoherenceError, type(raised.value)
    finally:
        server.shutdown()
