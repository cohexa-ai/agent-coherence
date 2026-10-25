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

The release a re-mint sends for an earlier attempt's write grant (one
``session-stop`` naming the attempt a ``write()`` left holding the file) gets
the same envelope when the stop's store read raises (#401). A stop commits
nothing, so this is not an unknown outcome: strict raises
:class:`CoordinatorFailed` at that release, before the operation commits, and
degrade warns once and counts. Either way the grant stays recorded, so the
next re-mint sends the release again.
"""

from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import pytest

from ccs.adapters.claude_code.coordinator_server import session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import (
    CasRetriesExhausted,
    CasVersionConflict,
    CoherenceDegradedWarning,
    CoherenceError,
    CommitUnconfirmed,
    CoordinatorFailed,
    InvariantViolationError,
    StaleView,
)
from ccs.core.states import MESIState
from tests.coordinator_helpers import (
    answer_with_status,
    recorded_version,
    serve_in_process,
    write_holders,
)

#: FROZEN duplicate of the envelope's reason for the fault these tests raise
#: (never imported from the code under test).
_ENVELOPE = "internal: OperationalError"
#: FROZEN duplicate of CoordinatorFailed's reason.
_COORDINATOR_FAILED = "coordinator_failed"

_REL = "data/shared.txt"
_OTHER = "data/other.txt"
_THIRD = "data/third.txt"

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


def _fail_registry(server, monkeypatch: pytest.MonkeyPatch, method: str, when: str) -> None:
    """Make the registry's ``method`` raise the error a locked store raises,
    ``before`` it records anything or ``after`` it has recorded its commit."""
    real = getattr(server.registry, method)

    def failing(*args, **kwargs):
        if when == "after":
            real(*args, **kwargs)
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(server.registry, method, failing)


def _assert_landed_iff_after(server, rel: str, before_version: int, when: str) -> None:
    """Control: the ``after`` case really recorded the commit, and the
    ``before`` case really did not, so each test covers the case it names."""
    expected = before_version + 1 if when == "after" else before_version
    assert recorded_version(server, rel) == expected, f"control: commit recorded ({when})"


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
    server = serve_in_process(tmp_path, f"envelope-{operation}-{mode}-{when}")
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
    server = serve_in_process(tmp_path, f"envelope-publish-{mode}-{when}")
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
    server = serve_in_process(tmp_path, f"envelope-write-{mode}-{when}")
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
    server = serve_in_process(tmp_path, f"envelope-{read}")
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
    server = serve_in_process(tmp_path, "envelope-degrade-read")
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
    server = serve_in_process(tmp_path, "envelope-prose")
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
    server = serve_in_process(tmp_path, "envelope-prose-read")
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


#: The release a re-mint sends for an earlier attempt's write grants.
_STOP = "/hooks/session-stop"
#: Every route that carries a commit: write()'s, the OCC commit's, the batch's.
_COMMITS = frozenset({"/hooks/post-edit", "/hooks/post-edit-cas", "/session/commit_all"})

#: Each operation re-mints before it can commit, so after a write() it first
#: releases the grant that write() left (a batch of one member goes by the OCC
#: commit's path, a batch of two by its own).
_RE_MINTS = pytest.mark.parametrize(
    "operation", ["write_cas", "write_cas_at", "publish_one", "publish_two", "reacquire"]
)

#: How each operation ends under degrade once its release failed. The grant
#: the abandoned attempt still holds refuses the volume's own commit as another
#: agent's, and the volume names that grant with ``CoordinatorFailed`` (#434),
#: after that commit (``write_cas`` after one retry, see
#: :func:`_own_grant_commits`). A reacquire commits nothing and returns the
#: bytes.
_DEGRADE_ENDS = {
    "write_cas": ("CoordinatorFailed", None),
    "write_cas_at": ("CoordinatorFailed", None),
    "publish_one": ("CoordinatorFailed", None),
    "publish_two": ("CoordinatorFailed", None),
    "reacquire": ("returned", b"v2"),
}

#: FROZEN duplicates of what a commit raises when the volume's own earlier
#: write grant refused it (#434), by what that grant's release answered.
_OWN_GRANT_HELD = (
    "{rel} is held by this volume's own earlier write grant: the coordinator "
    "answered that grant's release with the grant still held, and refused this "
    "commit as another agent's. Nothing was committed. Retry once after a short "
    "wait, which sends the release again; if it fails again, report it to the "
    "operator."
)
def _own_grant_commits(operation: str) -> int:
    """How many commits ``operation`` sends before its own unreleased grant's
    refusal stops it: ``write_cas`` retries once, because its re-mint sends
    the release again (#434); the single-shot operations stop at the first."""
    return 2 if operation == "write_cas" else 1


_OWN_GRANT_UNRELEASED = (
    "{rel} may be held by this volume's own earlier write grant, whose release "
    "was not confirmed: the coordinator refused this commit because another "
    "write grant holds the file. Nothing was committed. Retry once after a "
    "short wait, which sends the release again; if it fails again, report it "
    "to the operator."
)


def _wire(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict, object]]:
    """Record the route and payload of every request the volume sends from
    here on, with the answer it got."""
    from ccs.adapters import coherent_volume as coherent_volume_module

    real_post = coherent_volume_module._coordinator_post
    sent: list[tuple[str, dict, object]] = []

    def recording(endpoint, path, payload, **kwargs):
        answer = real_post(endpoint, path, payload, **kwargs)
        sent.append((path, payload, answer))
        return answer

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", recording)
    return sent


def _stops(sent: list[tuple[str, dict, object]]) -> list[object]:
    return [answer for route, _payload, answer in sent if route == _STOP]


def _stopped_attempts(sent: list[tuple[str, dict, object]]) -> list[object]:
    """The attempt each release named, in the order they went out."""
    return [payload.get("agent_id") for route, payload, _answer in sent if route == _STOP]


def _commits(sent: list[tuple[str, dict, object]]) -> list[str]:
    return [route for route, _payload, _answer in sent if route in _COMMITS]


def _degraded_warning(warned: list[warnings.WarningMessage]) -> str:
    """The text of the one degradation warning ``warned`` holds."""
    degraded = [str(w.message) for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
    assert len(degraded) == 1, degraded
    return degraded[0]


#: FROZEN duplicate of degrade's warning for a release the envelope answered.
_DEGRADED_RELEASE_WARNING = (
    "CoherentVolume degraded: coordinator answered the release of this "
    f"volume's earlier write grant with a failure ({_ENVELOPE}); the grant may "
    "still stand, and the release is sent again at the next fresh attempt."
)


def _assert_claims_no_outcome(text: str) -> None:
    """Degrade's text for a release the envelope answered: what failed, that
    the grant may stand, that the release goes out again, and nothing about
    the operation's own outcome, which goes on and may commit."""
    assert text == _DEGRADED_RELEASE_WARNING, text
    assert "nothing was committed" not in text, f"the operation may commit: {text}"
    assert "Retry once" not in text, f"a retry may apply the change twice: {text}"
    assert _ENVELOPE in text, text
    assert "release" in text, "the text must name what failed"
    assert "the grant may still stand" in text, text
    assert "sent again" in text, "the text must say the release goes out again"


def _volume_after_a_write(
    tmp_path: Path, fast_cfg: LifecycleConfig, mode: str
) -> tuple[CoherentVolume, str, tuple[int, int]]:
    """A volume whose ``write()`` of ``_REL`` left its attempt holding the
    grant. Returns the volume, that attempt, and the versions of ``_REL`` and
    ``_OTHER`` an operation commits against."""
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg)
    vol.write(_REL, b"v2")
    writer = vol._incarnation
    _data, version = vol.read_with_version(_REL)
    _other, other_version = vol.read_with_version(_OTHER)
    return vol, writer, (version, other_version)


def _re_mint(vol: CoherentVolume, operation: str, versions: tuple[int, int]) -> object:
    """Run ``operation``, which re-mints before it commits (see
    ``_RE_MINTS``), and return what it returned."""
    version, other_version = versions
    if operation == "write_cas":
        return vol.write_cas(_REL, lambda current: current + b"+cas")
    if operation == "write_cas_at":
        return vol.write_cas_at(_REL, version, b"v3")
    if operation == "publish_one":
        return vol.atomic_publish([(_REL, version, b"v3")])
    if operation == "publish_two":
        return vol.atomic_publish([(_REL, version, b"v3"), (_OTHER, other_version, b"o2")])
    assert operation == "reacquire", operation
    return vol.reacquire(_REL)


def _ended_in(outcome: object) -> tuple[str, object]:
    """What an operation ended in: the error's type and the refusal it names
    (an exhausted ``write_cas`` names its last one), or that it returned."""
    if isinstance(outcome, CasRetriesExhausted):
        return type(outcome).__name__, outcome.last_conflict_reason
    if isinstance(outcome, CasVersionConflict):
        return type(outcome).__name__, outcome.reason
    if isinstance(outcome, BaseException):
        return type(outcome).__name__, None
    return "returned", outcome


@_RE_MINTS
def test_a_strict_release_answered_with_the_failure_envelope_raises_coordinator_failed_before_any_commit(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """After ``write()`` the volume's attempt still holds the file's grant,
    and each of these operations re-mints first, releasing that attempt with
    one ``session-stop``. When the envelope answers the release, strict raises
    :class:`CoordinatorFailed` at that first failed release, naming the
    envelope and saying nothing was committed. No commit is sent, and the
    abandoned attempt stays in the release record, so the next re-mint sends
    the release again.

    Prevents the operation carrying on under its new attempt with no error:
    its own grant, held by the old attempt, refuses the commit as another
    agent's (``other_holder``, or the batch's ``StaleView``), ``write_cas``
    re-sends the failing release on every attempt until its budget runs out,
    and ``reacquire()`` returns as if nothing failed. The caller backs off for
    a peer that does not exist and never learns the coordinator failed."""
    target = _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, f"envelope-release-{operation}-strict")
    try:
        vol, writer, versions = _volume_after_a_write(tmp_path, fast_cfg, "strict")
        writer_row = str(session_to_agent_id(vol.session_id, writer))
        assert write_holders(server, _REL) == {writer_row}, "control: write() left its grant"
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            sent = _wire(faulted)
            with pytest.raises(CoherenceError) as raised:
                _re_mint(vol, operation, versions)

        assert type(raised.value) is CoordinatorFailed, repr(raised.value)
        assert raised.value.reason == _COORDINATOR_FAILED
        assert _ENVELOPE in str(raised.value)
        assert "release" in str(raised.value), "the text must name what failed"
        assert "nothing was committed" in str(raised.value)
        stops = _stops(sent)
        assert len(stops) == 1, f"the operation went on past its failed release: {len(stops)} stops"
        assert stops[0].get("reason") == _ENVELOPE, f"control: the envelope answered {stops[0]!r}"
        assert not _commits(sent), "a commit went out after the release failed"
        assert _REL in vol._grant_incarnations.get(writer, set()), "the held grant left the record"
        assert write_holders(server, _REL) == {writer_row}, "control: the grant still stands"
        assert target.read_bytes() == b"v2"
        assert vol.degradation_count == 0
    finally:
        server.shutdown()


@_RE_MINTS
def test_a_degrade_release_answered_with_the_failure_envelope_warns_once_and_counts(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """Degrade mode warns once and counts every release the envelope
    answered, as for any other coordinator failure, and keeps the abandoned
    attempt in the release record. The warning names the envelope, says the
    grant may still stand and the release goes out again, and claims nothing
    about the operation, which goes on.

    Prevents the failure showing only as a log line: the volume reported
    nothing degraded (``degradation_count`` 0, no warning) while its own
    grant stood. Prevents too the strict text's "nothing was committed.
    Retry once", which is false here: the operation goes on after the
    warning.

    The operation's outcome after that is pinned in ``_DEGRADE_ENDS``: a
    commit is refused by the volume's own unreleased grant, which the volume
    names (#434)."""
    target = _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, f"envelope-release-{operation}-degrade")
    try:
        vol, writer, versions = _volume_after_a_write(tmp_path, fast_cfg, "degrade")
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            sent = _wire(faulted)
            with warnings.catch_warnings(record=True) as warned:
                warnings.simplefilter("always")
                try:
                    outcome: object = _re_mint(vol, operation, versions)
                except CoherenceError as exc:
                    outcome = exc

        stops = _stops(sent)
        assert stops, "control: no release was sent"
        assert all(answer.get("reason") == _ENVELOPE for answer in stops), (
            f"control: the envelope answered {stops!r}"
        )
        assert vol.degradation_count == len(stops), (
            f"{len(stops)} failed releases, {vol.degradation_count} counted"
        )
        _assert_claims_no_outcome(_degraded_warning(warned))
        assert _REL in vol._grant_incarnations.get(writer, set()), "the held grant left the record"
        assert _ended_in(outcome) == _DEGRADE_ENDS[operation], repr(outcome)
        if operation != "reacquire":
            assert str(outcome) == _OWN_GRANT_UNRELEASED.format(rel=_REL)
            assert len(_commits(sent)) == _own_grant_commits(operation), (
                "the call went on committing into its own grant"
            )
        assert target.read_bytes() == b"v2"
    finally:
        server.shutdown()


@pytest.mark.parametrize("operation", ["write_cas_at", "publish_one", "publish_two"])
def test_a_degrade_release_answered_with_the_failure_envelope_claims_no_outcome_for_a_commit_that_lands(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """The grant ``write()`` left stands on one file, and the operation
    commits other files (alone, as a batch of one, or as a batch of two).
    Under degrade the release the envelope answers warns once, and the
    operation goes on and commits: the bytes land and the coordinator's
    version moves. The warning says what failed and nothing of the commit.

    Prevents a warning that says nothing was committed and to retry once,
    over a commit that landed: a caller that follows it applies its change a
    second time, or is refused as a conflict with its own commit."""
    _seed(tmp_path)
    other = _seed(tmp_path, _OTHER, b"o1")
    third = _seed(tmp_path, _THIRD, b"t1")
    server = serve_in_process(tmp_path, f"envelope-release-elsewhere-{operation}")
    try:
        vol, writer, (_version, other_version) = _volume_after_a_write(
            tmp_path, fast_cfg, "degrade"
        )
        _third, third_version = vol.read_with_version(_THIRD)
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            sent = _wire(faulted)
            with warnings.catch_warnings(record=True) as warned:
                warnings.simplefilter("always")
                if operation == "write_cas_at":
                    vol.write_cas_at(_OTHER, other_version, b"o2")
                elif operation == "publish_one":
                    vol.atomic_publish([(_OTHER, other_version, b"o2")])
                else:
                    vol.atomic_publish(
                        [(_OTHER, other_version, b"o2"), (_THIRD, third_version, b"t2")]
                    )

        assert [answer.get("reason") for answer in _stops(sent)] == [_ENVELOPE], (
            "control: one release, answered with the envelope"
        )
        assert other.read_bytes() == b"o2", "control: the commit landed"
        assert recorded_version(server, _OTHER) == other_version + 1
        if operation == "publish_two":
            assert third.read_bytes() == b"t2", "control: the batch landed"
            assert recorded_version(server, _THIRD) == third_version + 1
        assert vol.degradation_count == 1
        _assert_claims_no_outcome(_degraded_warning(warned))
        assert vol._grant_incarnations.get(writer) == {_REL}, "the held grant left the record"
    finally:
        server.shutdown()


def test_a_degrade_re_mint_stops_at_the_first_release_the_failure_envelope_answers(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two earlier attempts each still hold a grant ``write()`` left, on
    different files. While the envelope answers every release, each re-mint
    sends one ``session-stop``, for the first attempt in the order the
    record is released in, counts one degradation, and leaves both records
    as they were: the other attempt gets no release and waits, recorded, for
    a later re-mint.

    Prevents a re-mint that goes on past a failed release: every abandoned
    attempt meets the same failing coordinator, so it would spend a failed
    request, and count a degradation, for each one instead of one."""
    _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    _seed(tmp_path, _THIRD, b"t1")
    server = serve_in_process(tmp_path, "envelope-release-stops-at-the-first")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="degrade", config=fast_cfg)
        vol.write(_REL, b"v2")
        first = vol._incarnation
        # Strand the first attempt: its release fails, so the re-mint leaves
        # it recorded, and the next attempt writes another file.
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                vol.reacquire(_OTHER)
        vol.write(_OTHER, b"o2")
        record = {first: {_REL}, vol._incarnation: {_OTHER}}
        assert vol._grant_incarnations == record, "control: two attempts hold a grant"
        assert vol.degradation_count == 1, "control: the first attempt's release failed"
        released_first = sorted(record)[0]

        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            sent = _wire(faulted)
            for re_mints in (1, 2):
                # A file neither attempt holds, so the read changes no grant.
                vol.reacquire(_THIRD)
                assert _stopped_attempts(sent) == [released_first] * re_mints, (
                    f"re-mint {re_mints} went on past its failed release"
                )
                assert [answer.get("reason") for answer in _stops(sent)] == (
                    [_ENVELOPE] * re_mints
                ), "control: the envelope answered every release"
                assert vol.degradation_count == 1 + re_mints
                assert vol._grant_incarnations == record, "a failed release changed the record"
    finally:
        server.shutdown()


_COMMITTING = pytest.mark.parametrize(
    "operation", ["write_cas", "write_cas_at", "publish_one", "publish_two"]
)


@_COMMITTING
def test_a_degrade_commit_refused_by_a_grant_whose_release_got_a_5xx_names_that_grant(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """The release is answered with an HTTP 503 before the coordinator ran
    it, so the grant stands; degrade warns and counts, and the call goes on.
    Its commit is refused ``other_holder`` by that grant, and the volume
    raises ``CoordinatorFailed`` saying its own earlier write grant may hold
    the file, after that commit (``write_cas`` after one retry) (#434). "May":
    a release whose answer
    was a status may or may not have run, so the holder could be a peer."""
    target = _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, f"5xx-release-{operation}-degrade")
    try:
        vol, writer, versions = _volume_after_a_write(tmp_path, fast_cfg, "degrade")
        with monkeypatch.context() as faulted:
            answer_with_status(faulted, lambda path, _p: path == _STOP, 503, forwarded=False)
            sent = _wire(faulted)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with pytest.raises(CoordinatorFailed) as raised:
                    _re_mint(vol, operation, versions)

        assert str(raised.value) == _OWN_GRANT_UNRELEASED.format(rel=_REL)
        assert len(_commits(sent)) == _own_grant_commits(operation), (
            "the call went on committing into its own grant"
        )
        assert vol.degradation_count == _own_grant_commits(operation), "one per failed release"
        assert write_holders(server, _REL) == {str(session_to_agent_id(vol.session_id, writer))}, (
            "control: the only write holder is the volume's own earlier attempt"
        )
        assert target.read_bytes() == b"v2"
    finally:
        server.shutdown()


@_MODE
@pytest.mark.parametrize("operation", ["write_cas_at", "publish_two"])
def test_a_commit_refused_by_a_peers_grant_stays_other_holder(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    operation: str,
    mode: str,
) -> None:
    """Control: a peer's write() holds the grant, and this volume holds no
    earlier grant of its own on the file. The refusal is the peer's and
    keeps its answer, ``CasVersionConflict(other_holder)`` or the batch's
    ``StaleView`` naming ``other_holder``, never the own-grant
    ``CoordinatorFailed``."""
    _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, f"peer-holds-{operation}-{mode}")
    try:
        peer = CoherentVolume(tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg)
        peer.write(_REL, b"peer")
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg)
        versions = (vol.read_with_version(_REL)[1], vol.read_with_version(_OTHER)[1])
        assert not vol._grant_incarnations, "control: this volume holds no grant of its own"

        with pytest.raises(CoherenceError) as raised:
            _re_mint(vol, operation, versions)

        if operation == "write_cas_at":
            assert _ended_in(raised.value) == ("CasVersionConflict", "other_holder")
        else:
            assert type(raised.value) is StaleView, repr(raised.value)
            assert raised.value.member_reason == "other_holder"
    finally:
        server.shutdown()


def test_a_commit_refused_for_another_reason_keeps_it_while_an_own_grant_is_recorded(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control: only an ``other_holder`` refusal can come from the volume's own
    grant. The release fails (degrade), so the record keeps the grant, and the
    commit is refused ``version_mismatch``, answered in place of the
    coordinator's own answer. The caller is told the version moved,
    ``CasVersionConflict(version_mismatch)``, never the own-grant
    ``CoordinatorFailed``."""
    from ccs.adapters import coherent_volume as coherent_volume_module

    _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, "own-grant-other-reason")
    try:
        vol, writer, versions = _volume_after_a_write(tmp_path, fast_cfg, "degrade")
        real_post = coherent_volume_module._coordinator_post

        def refuse_the_commit(endpoint, path, payload, **kwargs):
            if path == "/hooks/post-edit-cas":
                return {"ok": False, "reason": "version_mismatch", "current_version": versions[0] + 1}
            return real_post(endpoint, path, payload, **kwargs)

        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            faulted.setattr(coherent_volume_module, "_coordinator_post", refuse_the_commit)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(CoherenceError) as raised:
                    vol.write_cas_at(_REL, versions[0], b"v3")

        assert _REL in vol._grant_incarnations.get(writer, set()), "control: the grant is still recorded"
        assert _ended_in(raised.value) == ("CasVersionConflict", "version_mismatch")
    finally:
        server.shutdown()


def test_a_later_release_that_goes_unanswered_says_the_grant_may_be_held(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first call's release is answered with the grant still held, so its
    commit names the grant: the file "is held" by it. The next call's release
    gets the failure envelope instead (degrade), and its commit says the
    grant "may be held": what an earlier release reported is not what this
    one did."""
    _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, "own-grant-held-then-unanswered")
    try:
        vol, _writer, versions = _volume_after_a_write(tmp_path, fast_cfg, "degrade")

        def refusing(*_args, **_kwargs):
            raise InvariantViolationError("release refused")

        with monkeypatch.context() as faulted:
            faulted.setattr(server.service, "invalidate", refusing)
            with pytest.raises(CoordinatorFailed) as first:
                vol.write_cas_at(_REL, versions[0], b"v3")
        assert str(first.value) == _OWN_GRANT_HELD.format(rel=_REL), "control: the first release said held"

        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with pytest.raises(CoordinatorFailed) as second:
                    vol.write_cas_at(_REL, versions[0], b"v3")

        assert str(second.value) == _OWN_GRANT_UNRELEASED.format(rel=_REL)
    finally:
        server.shutdown()


def test_a_degrade_write_cas_whose_release_fails_once_recovers_in_the_same_call(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first release is answered with an HTTP 503 before the coordinator
    ran it, and the next is answered as usual. write_cas's first commit is
    refused by the volume's own grant; it re-mints, which sends the release
    again, the release is confirmed, and the second commit wins in the same
    call. One degradation is counted, the record is clear, and the new bytes
    are on disk.

    Prevents the own-grant stop (#434) costing a degrade volume the recovery
    its retry loop gave a one-off release failure."""
    target = _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, "release-fails-once-write-cas")
    try:
        vol, writer, versions = _volume_after_a_write(tmp_path, fast_cfg, "degrade")
        failed = [False]

        def the_first_release(path: str, _payload: dict) -> bool:
            if path != _STOP or failed[0]:
                return False
            failed[0] = True
            return True

        with monkeypatch.context() as faulted:
            answer_with_status(faulted, the_first_release, 503, forwarded=False)
            sent = _wire(faulted)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = vol.write_cas(_REL, lambda current: current + b"+cas")

        assert failed[0], "control: the first release was answered 503"
        assert result.version == versions[0] + 1
        assert target.read_bytes() == b"v2+cas"
        assert len(_commits(sent)) == 2, "the call did not retry once after its own grant refused it"
        assert vol.degradation_count == 1
        assert writer not in vol._grant_incarnations, "the confirmed release stayed in the record"
    finally:
        server.shutdown()


def test_a_commit_refused_by_a_peer_after_this_volumes_release_was_confirmed_stays_other_holder(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
) -> None:
    """Control: this volume wrote the file and its release was confirmed (a
    reacquire of another file re-minted), so its record is clear; a peer's
    write() then holds the file. The refusal is the peer's and keeps its
    answer, ``CasVersionConflict(other_holder)``. Tells "the record still
    holds a grant" apart from "this volume once wrote the path"."""
    _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, "released-then-peer-holds")
    try:
        vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        vol.write(_REL, b"v2")
        vol.reacquire(_OTHER)
        assert not vol._grant_incarnations, "control: the release was confirmed"
        peer = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
        peer.write(_REL, b"peer")
        version = vol.read_with_version(_REL)[1]

        with pytest.raises(CasVersionConflict) as raised:
            vol.write_cas_at(_REL, version, b"v3")

        assert raised.value.reason == "other_holder"
    finally:
        server.shutdown()


@pytest.mark.parametrize("operation", ["write_cas", "write_cas_at", "publish_one", "publish_two"])
def test_a_strict_operation_retried_after_a_failed_release_sends_it_again_and_commits(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    """The envelope answers the release once, and then the coordinator is
    healthy. The first call raises :class:`CoordinatorFailed`; the same call
    retried sends the release again, the coordinator confirms it, the
    attempt leaves the release record, its grant is gone, and the commit
    wins.

    Prevents a release that, once failed, is never sent again: the grant
    would stand for the life of the volume, and every later commit of the
    file would be refused as another agent's."""
    target = _seed(tmp_path)
    other = _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, f"envelope-release-recovers-{operation}")
    try:
        vol, writer, versions = _volume_after_a_write(tmp_path, fast_cfg, "strict")
        writer_row = str(session_to_agent_id(vol.session_id, writer))
        sent = _wire(monkeypatch)
        with monkeypatch.context() as faulted:
            _fail_registry(server, faulted, "artifacts_held_by_agent", "before")
            with pytest.raises(CoordinatorFailed):
                _re_mint(vol, operation, versions)
        assert _REL in vol._grant_incarnations.get(writer, set()), "control: the release failed"
        assert not _commits(sent), "control: the first call committed nothing"

        _re_mint(vol, operation, versions)

        stops = _stops(sent)
        assert _stopped_attempts(sent) == [writer, writer], "the retry did not send the release"
        assert stops[0].get("reason") == _ENVELOPE, f"control: the envelope answered {stops[0]!r}"
        assert stops[1].get("ok") is True and not stops[1].get("degraded"), (
            f"control: the retried release was not confirmed: {stops[1]!r}"
        )
        assert writer not in vol._grant_incarnations, "a confirmed release stayed in the record"
        assert writer_row not in write_holders(server, _REL), "the grant still stands"
        expected = b"v2+cas" if operation == "write_cas" else b"v3"
        assert target.read_bytes() == expected, "the retried commit did not land"
        assert recorded_version(server, _REL) == versions[0] + 1
        if operation == "publish_two":
            assert other.read_bytes() == b"o2", "the retried batch did not land"
        assert vol.degradation_count == 0
    finally:
        server.shutdown()


@_MODE
@_RE_MINTS
def test_a_commit_refused_by_a_grant_its_release_left_held_names_that_grant(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    mode: str,
) -> None:
    """A stop that ran but left the grant held answers per grant (#185):
    ``ok: false`` with a ``grants`` list, and no top-level reason. That is the
    coordinator's account of the grant, not a failure of the coordinator, so
    in both modes the release itself raises nothing, counts and warns
    nothing, the record keeps the grant, and the operation goes on: a
    reacquire returns the bytes. A commit goes out and is refused
    ``other_holder`` by that grant, and the volume raises
    ``CoordinatorFailed`` saying its own earlier write grant holds the file
    (#434), after that commit (``write_cas`` after one retry, whose re-mint
    sends the release again).

    Prevents the volume telling its caller to wait for a peer that does not
    exist: the CAS raised ``CasVersionConflict(other_holder)``, which MCP
    answers ``wait_and_retry``, ``write_cas`` spent its retry budget against
    its own grant, and the batch held as a peer's conflict. Prevents too the
    envelope's handling widening to every ``ok: false`` release."""
    target = _seed(tmp_path)
    _seed(tmp_path, _OTHER, b"o1")
    server = serve_in_process(tmp_path, f"held-release-{operation}-{mode}")
    try:
        vol, writer, versions = _volume_after_a_write(tmp_path, fast_cfg, mode)

        def refusing(*_args, **_kwargs):
            raise InvariantViolationError("release refused")

        with monkeypatch.context() as faulted:
            # The stop's per-grant release; its failure is answered per grant.
            faulted.setattr(server.service, "invalidate", refusing)
            sent = _wire(faulted)
            with warnings.catch_warnings(record=True) as warned:
                warnings.simplefilter("always")
                try:
                    outcome: object = _re_mint(vol, operation, versions)
                except CoherenceError as exc:
                    outcome = exc

        stops = _stops(sent)
        assert len(stops) == _own_grant_commits(operation), stops
        assert stops[0].get("ok") is False and "reason" not in stops[0], (
            f"control: not a per-grant answer: {stops[0]!r}"
        )
        assert [grant.get("held") for grant in stops[0]["grants"]] == [True], (
            "control: the grant is held"
        )
        if operation == "reacquire":
            assert _ended_in(outcome) == ("returned", b"v2"), repr(outcome)
        else:
            assert type(outcome) is CoordinatorFailed, repr(outcome)
            assert str(outcome) == _OWN_GRANT_HELD.format(rel=_REL)
            assert len(_commits(sent)) == _own_grant_commits(operation), (
                "the call went on committing into its own grant"
            )
            assert write_holders(server, _REL) == {str(session_to_agent_id(vol.session_id, writer))}, (
                "control: the only write holder is the volume's own earlier attempt"
            )
        assert vol.degradation_count == 0
        assert not [w for w in warned if issubclass(w.category, CoherenceDegradedWarning)]
        assert _REL in vol._grant_incarnations.get(writer, set()), "the held grant left the record"
        assert target.read_bytes() == b"v2"
    finally:
        server.shutdown()
