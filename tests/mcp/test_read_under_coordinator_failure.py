# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""MCP tool calls when the coordinator fails: an answered failure, and death.

**An answered failure.** Two coordinator-side failures answer a pre-read as
HTTP 200 ``{ok: false, reason: "internal: <Type>"}``: the caller-principal
gate's store read raising (a bound session whose binding is not cached — what
a coordinator restart leaves — under a locked registry) and the pre-read work
body raising. Neither confirms the read was registered (a body raise can fail
after part of it was), so the server's strict volume must fail it closed and
the tool call must answer a non-ignorable deny, never content: a read the
agent takes for registered is one a peer's later commit invalidates nothing
for, and the agent's next write lands over it. The deny is typed
(``coordinator_failed``, #259) and names one retry. A write tool whose commit
gets the same answer may have landed, so it answers ``commit_unconfirmed``.
A CAS or a reacquire whose release of the volume's earlier write grant gets
it (#401) committed nothing, so it answers ``coordinator_failed`` too.

**Death.** Once the coordinator the server's strict volume attached to is
gone, no request reaches it, and the volume keeps that dead endpoint: it never
re-attaches, so it never reaches a coordinator started later. Every tool call
must answer ``coordinator_unavailable`` with the stop verb, not
``internal_error`` (the shape of a bug in the server) and not the unattached
refusal's ``retry_later`` (no retry in this server session can succeed).

What the death tests do not prove: they run on one machine over loopback, and
the coordinator is shut down in this process, so a refused connect is the only
failure driven (no network partition, no coordinator that hangs, no process
killed by a signal). The Claude Code hook client is not driven and still fails
open when its coordinator is gone. The server's attach leg, which still spawns
or attaches a coordinator at startup, is unchanged and not driven. The Node
client is untouched.

Driven through the sync ``_do_*`` helpers against a coordinator running in
THIS process, so its registry can be made to fail and its listening socket
closed, as the volume's own tests do.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from mcp.types import CallToolResult

from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import InvariantViolationError
from ccs.mcp.server import _do_gate, _do_reacquire, _do_read, _do_write, _do_write_cas
from ccs.mcp.session import SessionConfig
from tests.coordinator_helpers import answer_with_status, serve_in_process

#: FROZEN duplicate of the deny reason an answered failure envelope gets on a
#: read (#259): its own typed terminal, not the ``internal_error`` of a
#: failure the mapper recognises nothing in. A request that gets no answer at
#: all gets ``_UNREACHABLE_REASON`` instead (never imported from the code
#: under test).
_COORDINATOR_FAILED = "coordinator_failed"
#: FROZEN duplicates of the coordinator-unreachable terminal: its reason and
#: the house stop verb, never the unattached refusal's ``retry_later``.
_UNREACHABLE_REASON = "coordinator_unavailable"
_STOP_AND_REPORT = "stop_and_report"
#: FROZEN duplicates of the unconfirmed-commit terminal (#259).
_COMMIT_UNCONFIRMED = "commit_unconfirmed"
_READ_THEN_RETRY = "read_then_retry"


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


def _seed(tmp_path: Path, rel: str = "data/shared.txt", content: bytes = b"v1") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _fail_the_pre_read(server, monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    """Make the coordinator answer the next pre-read with its failure envelope
    from one of its two arms: ``gate`` — the caller-principal gate's store
    read raises (a cold cache, as after a restart, and a locked registry);
    ``body`` — the pre-read work body raises (the watchdog call itself)."""

    def raising(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    if fault == "gate":
        with server.service._caller_principal_lock:
            server.service._caller_principals.clear()
        monkeypatch.setattr(server.registry, "get_caller_principal", raising)
    else:
        monkeypatch.setattr(server, "run_with_watchdog", raising)


@pytest.mark.parametrize("fault", ["gate", "body"])
def test_a_read_the_coordinator_answers_with_a_failure_is_a_deny_never_content(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """``swg_read`` answers ``isError`` with the typed ``coordinator_failed``
    deny (#259): the stop verb, ``retryable: false``, and a next step that
    names one retry, then a report; no content, version or generation; the
    coordinator holds no view for the path; once the coordinator recovers the
    same call answers the content, registered. The envelope is an answer, so
    it is not ``coordinator_unavailable``, which is for a request that got
    none.

    Prevents the tool answering content with ``version: 0`` and no view
    registered: the agent reads that as a successful read and decides from
    it, and the server's own status reports nothing degraded. Also prevents
    ``internal_error`` / ``recover: none``, the shape of a bug in the server,
    for a condition the next call may clear."""
    rel = "data/shared.txt"
    _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, f"swg-read-{fault}")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert vol.is_attached and vol.principal_claim_outcome == "bound", "control: bound"
        with monkeypatch.context() as faulted:
            _fail_the_pre_read(server, faulted, fault)
            result = _do_read(vol, config, rel)

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert sc["reason"] == _COORDINATOR_FAILED
        assert sc["recover"] == _STOP_AND_REPORT
        assert sc["retryable"] is False
        assert "once more" in sc["next_step"], "the next step must bound the retry"
        assert "content" not in sc and "version" not in sc and "owner_generation" not in sc
        assert server.registry.lookup_artifact_id_by_name(rel) is None, "the read registered"
        assert vol.is_degraded is False, "the server's strict volume degraded"

        recovered = _do_read(vol, config, rel)
        assert recovered.isError is False, "control: the coordinator recovered"
        assert recovered.structuredContent["content"] == "v1"
        assert server.registry.lookup_artifact_id_by_name(rel) is not None
    finally:
        server.shutdown()


@pytest.mark.parametrize("tool", ["swg_gate", "swg_reacquire"])
def test_the_other_read_tools_answer_the_failure_envelope_with_coordinator_failed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """``swg_gate`` (the effect fence's verification read) and
    ``swg_reacquire`` reach the read failure by their own paths, and answer it
    as ``swg_read`` does (#259): ``coordinator_failed``, the stop verb, not
    retryable, and a next step that bounds the retry.

    Prevents the guide's promise for these two tools resting on ``swg_read``'s
    test alone."""
    rel = "data/shared.txt"
    _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, f"{tool}-envelope")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        live = _do_read(vol, config, rel)
        assert live.isError is False, "control: the live coordinator answers"
        with monkeypatch.context() as faulted:
            _fail_the_pre_read(server, faulted, "body")
            if tool == "swg_gate":
                result = _do_gate(
                    vol, config, rel,
                    live.structuredContent["version"],
                    live.structuredContent.get("owner_generation"),
                )
            else:
                result = _do_reacquire(vol, config, rel)

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert (sc["reason"], sc["recover"], sc["retryable"]) == (
            _COORDINATOR_FAILED, _STOP_AND_REPORT, False
        ), sc
        assert "once more" in sc["next_step"]
    finally:
        server.shutdown()


@pytest.mark.parametrize("tool", ["swg_write", "swg_write_cas"])
def test_a_write_whose_commit_the_coordinator_answers_with_a_failure_is_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, tool: str
) -> None:
    """``swg_write`` and ``swg_write_cas`` whose commit the coordinator answers
    with its failure envelope after recording it both answer
    ``commit_unconfirmed`` with ``read_then_retry`` (#259): the two write
    tools agree, and the coordinator holds the commit the agent is told may
    have landed.

    Prevents ``internal_error`` / ``recover: none`` on ``swg_write_cas`` and
    ``commit_preempted`` on ``swg_write``: each tells the agent its write did
    not land, so it abandons it, or later re-drives one that did."""
    rel = "data/shared.txt"
    _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, f"{tool}-envelope")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        live = _do_read(vol, config, rel)
        assert live.isError is False, "control: the live coordinator answers"
        version = live.structuredContent["version"]
        method = "set_artifact_and_content" if tool == "swg_write" else "commit_cas"
        real = getattr(server.registry, method)

        def recorded_then_raising(*args, **kwargs):
            real(*args, **kwargs)
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as faulted:
            faulted.setattr(server.registry, method, recorded_then_raising)
            if tool == "swg_write":
                result = _do_write(vol, config, rel, "v2")
            else:
                result = _do_write_cas(vol, config, {}, rel, version, "v2")

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert (sc["reason"], sc["recover"], sc["retryable"]) == (
            _COMMIT_UNCONFIRMED, _READ_THEN_RETRY, False
        ), sc
        artifact_id = server.registry.lookup_artifact_id_by_name(rel)
        assert server.registry.get_artifact(artifact_id).version == version + 1, (
            "control: the coordinator recorded the commit"
        )
    finally:
        server.shutdown()


def test_a_cas_whose_release_of_an_earlier_write_grant_fails_answers_coordinator_failed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``swg_write``, the volume's attempt still holds the file's grant,
    so the next ``swg_write_cas`` first releases it. When the coordinator
    answers that release with its failure envelope (the stop's store read
    raised), the tool answers the typed ``coordinator_failed`` deny (#401):
    the stop verb, ``retryable: false``, and a next step that bounds the
    retry. No conflict is counted and no commit lands.

    Prevents ``other_holder`` with ``wait_and_retry`` and ``retryable: true``:
    the grant refusing the commit is the volume's own, so the agent backs off
    for a peer that does not exist, and on a fault that repeats it never stops
    retrying."""
    rel = "data/shared.txt"
    target = _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, "swg-write-cas-release-envelope")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        wrote = _do_write(vol, config, rel, "v2")
        assert wrote.isError is False, "control: the write landed"
        live = _do_read(vol, config, rel)
        assert live.isError is False, "control: the live coordinator answers"
        version = live.structuredContent["version"]
        conflicts: dict[str, int] = {}

        def raising(*_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as faulted:
            # Only the session-stop handler reads this, so only the release fails.
            faulted.setattr(server.registry, "artifacts_held_by_agent", raising)
            result = _do_write_cas(vol, config, conflicts, rel, version, "v3")

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert (sc["reason"], sc["recover"], sc["retryable"]) == (
            _COORDINATOR_FAILED, _STOP_AND_REPORT, False
        ), sc
        assert "once more" in sc["next_step"]
        assert conflicts == {}, "the failed release counted as a CAS conflict"
        artifact_id = server.registry.lookup_artifact_id_by_name(rel)
        assert server.registry.get_artifact(artifact_id).version == version, "a commit landed"
        assert target.read_bytes() == b"v2"
    finally:
        server.shutdown()


def test_a_cas_refused_by_the_grant_its_release_left_held_answers_coordinator_failed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After ``swg_write``, the next ``swg_write_cas`` first releases that
    write's grant, and the coordinator answers the release grant by grant
    with the grant still held (#185). That answer is no coordinator failure,
    so the call goes on to its commit, and the volume's own grant refuses it.
    The tool answers ``coordinator_failed`` (#434): the stop verb,
    ``retryable: false``, and a next step that bounds the retry. No conflict
    is counted and the commit did not land.

    Prevents ``other_holder`` with ``wait_and_retry``: the agent would wait
    for a peer that does not exist."""
    rel = "data/shared.txt"
    target = _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, "swg-write-cas-release-held")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert _do_write(vol, config, rel, "v2").isError is False, "control: the write landed"
        version = _do_read(vol, config, rel).structuredContent["version"]
        conflicts: dict[str, int] = {}

        def refusing(*_args, **_kwargs):
            raise InvariantViolationError("release refused")

        with monkeypatch.context() as faulted:
            # The stop's per-grant release; its failure is answered per grant.
            faulted.setattr(server.service, "invalidate", refusing)
            result = _do_write_cas(vol, config, conflicts, rel, version, "v3")

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert (sc["reason"], sc["recover"], sc["retryable"]) == (
            _COORDINATOR_FAILED, _STOP_AND_REPORT, False
        ), sc
        assert "once more" in sc["next_step"]
        assert conflicts == {}, "the refusal counted as a CAS conflict"
        assert target.read_bytes() == b"v2"
    finally:
        server.shutdown()


def test_a_read_answered_with_an_http_5xx_answers_coordinator_failed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``swg_read`` whose pre-read is answered with an HTTP 503 answers the
    same typed deny as the failure envelope (#433): ``coordinator_failed``,
    the stop verb, ``retryable: false``, and no content.

    Prevents ``internal_error`` with ``recover: none`` for a coordinator
    failure that arrives as a status."""
    rel = "data/shared.txt"
    _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, "swg-read-5xx")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        answer_with_status(monkeypatch, lambda path, _payload: path == "/hooks/pre-read", 503)

        result = _do_read(vol, config, rel)

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert (sc["reason"], sc["recover"], sc["retryable"]) == (
            _COORDINATOR_FAILED, _STOP_AND_REPORT, False
        ), sc
        assert "content" not in sc
    finally:
        server.shutdown()


def test_a_reacquire_whose_release_of_an_earlier_write_grant_fails_answers_coordinator_failed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``swg_reacquire`` re-mints the volume's identity, so after ``swg_write``
    it first releases the grant that write's attempt still holds. When the
    coordinator answers that release with its failure envelope, the tool
    answers as ``swg_write_cas`` does above (#401): ``coordinator_failed``,
    the stop verb, ``retryable: false``, and a next step that names one retry
    and then a report. It answers no content, and the file is unchanged.

    Prevents the guide's promise for ``swg_reacquire`` resting on the CAS
    test alone: it reaches the release by its own path. Also prevents
    content answered as a recovery while the earlier grant may still stand,
    unreleased."""
    rel = "data/shared.txt"
    target = _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, "swg-reacquire-release-envelope")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert _do_read(vol, config, rel).isError is False, "control: the live coordinator answers"
        wrote = _do_write(vol, config, rel, "v2")
        assert wrote.isError is False, "control: the write landed"

        def raising(*_args, **_kwargs):
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as faulted:
            # Only the session-stop handler reads this, so only the release fails.
            faulted.setattr(server.registry, "artifacts_held_by_agent", raising)
            result = _do_reacquire(vol, config, rel)

        assert result.isError is True, result.structuredContent
        sc = result.structuredContent
        assert (sc["reason"], sc["recover"], sc["retryable"]) == (
            _COORDINATOR_FAILED, _STOP_AND_REPORT, False
        ), sc
        assert "once more" in sc["next_step"], "the next step must bound the retry"
        assert "report" in sc["next_step"], "the next step must end in a report"
        assert "content" not in sc
        assert target.read_bytes() == b"v2"
    finally:
        server.shutdown()


def _assert_stop_and_report(result: CallToolResult) -> None:
    """A non-ignorable ``coordinator_unavailable`` deny with the stop verb, not
    retryable, carrying its fixed next step, and no content or version."""
    assert result.isError is True, result.structuredContent
    sc = result.structuredContent
    assert sc["reason"] == _UNREACHABLE_REASON, sc
    assert sc["recover"] == _STOP_AND_REPORT, sc
    assert sc["retryable"] is False
    assert sc["next_step"], "the stop carries no next step"
    assert "content" not in sc and "version" not in sc and "owner_generation" not in sc


def _attach_then_shut_down(
    tmp_path: Path, fast_cfg: LifecycleConfig, instance_id: str
) -> tuple[CoherentVolume, SessionConfig, int]:
    """Attach the server's strict volume to a live coordinator and read the
    seeded path once (control: the coordinator answered), then shut the
    coordinator down. Its listening socket closes, so every later request is
    refused at connect. Returns the volume, the session config and the version
    the live read answered."""
    _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    server = serve_in_process(tmp_path, instance_id)
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        assert vol.is_attached and vol.principal_claim_outcome == "bound", "control: bound"
        live = _do_read(vol, config, "data/shared.txt")
        assert live.isError is False, "control: the live coordinator answers"
    finally:
        server.shutdown()
    return vol, config, live.structuredContent["version"]


def test_after_the_coordinator_dies_read_then_write_each_stop_and_report_and_the_volume_stays_attached(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """AE10. After the coordinator dies, ``swg_read`` then ``swg_write`` each
    answer ``coordinator_unavailable`` with ``stop_and_report``, not
    retryable, and the volume stays attached between and after the calls, so
    the second call is refused by the same typed stop as the first rather than
    by the unattached ``retry_later`` refusal. The write leaves the file's
    bytes as they were.

    Prevents ``internal_error``, which an agent reads as a bug in the server,
    and ``retry_later``, which sends it to retry a session that can never
    reach a coordinator again. Without the mapper's row both calls answer
    ``internal_error``.

    Not proven here: one machine and loopback only, a refused connect only
    (no partition), the hook client (still fails open), the server's attach
    leg (unchanged), the Node client (untouched)."""
    rel = "data/shared.txt"
    vol, config, _version = _attach_then_shut_down(tmp_path, fast_cfg, "swg-dead-read-write")

    read = _do_read(vol, config, rel)
    _assert_stop_and_report(read)
    assert vol.is_attached is True, "the read detached the volume"

    write = _do_write(vol, config, rel, "v2")
    _assert_stop_and_report(write)
    assert vol.is_attached is True, "the write detached the volume"
    assert vol.is_degraded is False, "the server's strict volume degraded"
    assert (tmp_path / rel).read_bytes() == b"v1", "the refused write reached disk"


def test_after_the_coordinator_dies_a_cas_stops_and_reports_from_its_comparand_read(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """``swg_write_cas`` at the version the live read answered, after the
    coordinator dies, answers ``coordinator_unavailable`` with
    ``stop_and_report``. The refusal comes from the comparand read, the first
    request the CAS sends: this volume never wrote, so its re-mint releases
    nothing, and the detail (the volume's verbatim message) names the pre-read
    route. No conflict is counted, the volume stays attached, and nothing
    reaches disk.

    Not proven here: as for the read-then-write test above."""
    rel = "data/shared.txt"
    vol, config, version = _attach_then_shut_down(tmp_path, fast_cfg, "swg-dead-cas")
    conflicts: dict[str, int] = {}

    result = _do_write_cas(vol, config, conflicts, rel, version, "v2")

    _assert_stop_and_report(result)
    assert "/hooks/pre-read" in result.structuredContent["detail"], "not the comparand read"
    assert conflicts == {}, "a transport failure counted as a CAS conflict"
    assert vol.is_attached is True, "the CAS detached the volume"
    assert (tmp_path / rel).read_bytes() == b"v1", "the refused CAS reached disk"
