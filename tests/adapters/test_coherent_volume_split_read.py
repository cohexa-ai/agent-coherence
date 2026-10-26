# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A denied read inside a peer's commit→disk window must not hand out a comparand.

A peer's ``write_cas_at`` confirms its CAS at the coordinator and only THEN
writes its bytes to disk. A read that lands between the two sees the OLD bytes
on disk while the coordinator already reports the NEW version, and under strict
mode the coordinator denies it with ``hash_differs``. Handing that pair back as
``(bytes, version)`` is a split comparand: the caller derives from the old
bytes, passes the new version to ``write_cas_at``, and by the time the CAS runs
the peer's disk write has landed, so the version check passes and the peer's
update is overwritten.

The window is forced here rather than raced: the peer's ``_atomic_write`` waits
on an event the test releases, so every test below reads INSIDE the window.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.adapters.effect_gate import check_fence, gate
from ccs.core.exceptions import CasVersionConflict, CoherenceError, StaleView, ViewWedged

_PATH = "data/counter.txt"
_WAIT_SEC = 10.0
#: FROZEN duplicates of the refusal's typed reasons (#264): the split pair's
#: instance reason, and the class reason every other stale view keeps.
_SPLIT_PAIR = "split_pair"
_STALE_VIEW = "stale_view"


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


def _seed(tmp_path: Path, content: bytes = b"0") -> Path:
    target = tmp_path / "data" / "counter.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _vol(tmp_path: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=cfg)


class LaggingPeer:
    """A peer whose confirmed CAS is held OFF disk until the test releases it.

    ``write_cas_at`` calls ``_atomic_write`` only after the coordinator confirmed
    the CAS, so blocking there opens the commit→disk window on demand.
    """

    def __init__(
        self, vol: CoherentVolume, monkeypatch: pytest.MonkeyPatch, path: str = _PATH
    ) -> None:
        self._vol = vol
        self._path = path
        self._committed = threading.Event()
        self._release = threading.Event()
        self._errors: list[BaseException] = []
        self._thread: threading.Thread | None = None
        real_write = vol._atomic_write

        def lagging_write(abs_path: Path, data: bytes) -> None:
            self._committed.set()
            if not self._release.wait(_WAIT_SEC):
                raise AssertionError(
                    "timed out waiting for the test to release the peer's disk write"
                )
            real_write(abs_path, data)

        monkeypatch.setattr(vol, "_atomic_write", lagging_write)

    def commit(self, new_content: bytes) -> None:
        """Start the peer's CAS and return once it is confirmed but not on disk."""
        _data, version = self._vol.read_with_version(self._path)

        def run() -> None:
            try:
                self._vol.write_cas_at(self._path, version, new_content)
            except BaseException as exc:  # surfaced by finish()
                self._errors.append(exc)
                self._committed.set()

        self._thread = threading.Thread(target=run, name="lagging-peer")
        self._thread.start()
        assert self._committed.wait(_WAIT_SEC), (
            "timed out waiting for the peer's CAS to be confirmed at the coordinator"
        )
        assert not self._errors, f"peer CAS failed before its disk write: {self._errors!r}"

    def release_after(self, seconds: float) -> None:
        """Let the peer's disk write land after ``seconds``, from another thread."""
        timer = threading.Timer(seconds, self._release.set)
        timer.daemon = True
        timer.start()

    def finish(self) -> None:
        """Let the peer's disk write land and wait for its write_cas_at to return."""
        self._release.set()
        assert self._thread is not None
        self._thread.join(_WAIT_SEC)
        assert not self._thread.is_alive(), (
            "timed out waiting for the peer's write_cas_at to return after release"
        )
        assert not self._errors, f"peer CAS failed: {self._errors!r}"


@pytest.fixture
def volumes(tmp_path: Path, fast_cfg: LifecycleConfig):
    target = _seed(tmp_path)
    peer = _vol(tmp_path, fast_cfg)
    reader = _vol(tmp_path, fast_cfg)
    try:
        yield target, peer, reader
    finally:
        stop_coordinator(tmp_path)


def _increment(data: bytes) -> bytes:
    return str(int(data) + 1).encode()


# Two ways a reader reaches the window: never read the key (no grant, so the
# coordinator hash-checks it), or read it before the peer's commit (SHARED, then
# INVALID when the peer commits). Both are denied with hash_differs.
_READER_STATES = pytest.mark.parametrize(
    "read_before_peer", [False, True], ids=["no-prior-grant", "invalidated"]
)


@_READER_STATES
def test_read_with_version_refuses_a_split_pair_in_the_commit_window(
    volumes, monkeypatch: pytest.MonkeyPatch, read_before_peer: bool
) -> None:
    _target, peer_vol, reader = volumes
    if read_before_peer:
        reader.read_with_version(_PATH)
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    try:
        with pytest.raises(StaleView) as exc:
            reader.read_with_version(_PATH)
    finally:
        peer.finish()
    # The refusal tells the caller to wait out a peer's commit BEFORE concluding
    # the file changed outside the coordinator and writing: a write made inside
    # the window is overwritten when the peer's bytes land.
    guidance = str(exc.value)
    assert "reacquire()" in guidance
    assert guidance.index("few seconds") < guidance.index("write()")
    assert "overwritten" in guidance
    # Typed, so a caller tells the split from every other stale view by value
    # (#264); the class keeps the reason the MCP mapper answers.
    assert exc.value.reason == _SPLIT_PAIR
    assert StaleView.reason == _STALE_VIEW


@_READER_STATES
def test_read_modify_cas_through_the_window_loses_no_update(
    volumes, monkeypatch: pytest.MonkeyPatch, read_before_peer: bool
) -> None:
    """The whole read→derive→write_cas_at cycle, run the way a caller would:
    a refused read is retried once the peer has moved on. Before the refusal
    existed, the reader derived "1" from the old "0", CAS'd it at the peer's
    version, and won, so the file ended at "1" with the peer's increment gone."""
    target, peer_vol, reader = volumes
    if read_before_peer:
        reader.read_with_version(_PATH)
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    try:
        pair = reader.read_with_version(_PATH)
    except StaleView:
        pair = None
    peer.finish()
    if pair is None:
        pair = reader.read_with_version(_PATH)
    data, version = pair
    reader.write_cas_at(_PATH, version, _increment(data))
    assert target.read_bytes() == b"2"


@pytest.mark.parametrize("read", ["read_with_version", "read_with_version_generation"])
def test_refused_read_does_not_absolve_an_out_of_band_edit(volumes, read: str) -> None:
    """A refused read hands the caller no bytes, so it must not move the
    foreign-edit baseline either. When it did, a write of the caller's older
    buffer after an out-of-band edit passed the foreign-edit check and landed
    over the edit the caller never saw."""
    target, _peer, reader = volumes
    buffered, _version = reader.read_with_version(_PATH)
    target.write_bytes(b"HUMAN")
    with pytest.raises(StaleView) as refused:
        getattr(reader, read)(_PATH)
    with pytest.raises(StaleView) as denied:
        reader.write(_PATH, _increment(buffered))
    assert target.read_bytes() == b"HUMAN"
    # The split is told from the write's own deny by value, not prose (#264).
    assert (refused.value.reason, denied.value.reason) == (_SPLIT_PAIR, _STALE_VIEW)


@pytest.mark.parametrize("read", ["read_with_version", "read_with_version_generation"])
def test_fail_closed_read_does_not_absolve_an_out_of_band_edit(
    volumes, monkeypatch: pytest.MonkeyPatch, read: str
) -> None:
    """A read that fails closed (the coordinator answers degraded, so strict
    mode raises) returns no bytes either. It must not advance the foreign-edit
    baseline, or the caller's next write lands over an edit it never saw."""
    target, _peer, reader = volumes
    buffered, _version = reader.read_with_version(_PATH)
    target.write_bytes(b"HUMAN")
    real_post = reader._post

    def degraded_pre_read(endpoint_path: str, payload: dict) -> dict | None:
        if endpoint_path == "/hooks/pre-read":
            return {"ok": True, "degraded": True}
        return real_post(endpoint_path, payload)

    monkeypatch.setattr(reader, "_post", degraded_pre_read)
    with pytest.raises(CoherenceError):
        getattr(reader, read)(_PATH)
    monkeypatch.setattr(reader, "_post", real_post)
    with pytest.raises(StaleView):
        reader.write(_PATH, _increment(buffered))
    assert target.read_bytes() == b"HUMAN"


def test_refused_first_read_still_guards_a_later_write(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path's first read, refused inside a peer's commit window, still records
    what the disk held then. Otherwise the instance has no baseline at all, and
    a later write lands over the peer's commit once its bytes reach disk."""
    target, peer_vol, reader = volumes
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    try:
        with pytest.raises(StaleView):
            reader.read_with_version(_PATH)
    finally:
        peer.finish()
    with pytest.raises(StaleView):
        reader.write(_PATH, b"blind")
    assert target.read_bytes() == b"1"



def test_lost_write_cas_at_on_a_never_read_path_still_guards_a_later_write(
    volumes,
) -> None:
    """write_cas_at's comparand read discards its bytes, so it must not ADVANCE
    a baseline; but on a path this instance never read it still records the
    first one. Otherwise a lost CAS leaves no baseline, and a write() made after
    an out-of-band edit lands over it unchecked."""
    target, peer, agent = volumes
    peer.read(_PATH)
    peer.write(_PATH, b"1")
    with pytest.raises(CasVersionConflict):
        agent.write_cas_at(_PATH, 1, b"agent")
    target.write_bytes(b"HUMAN")
    with pytest.raises(StaleView):
        agent.write(_PATH, b"agent")
    assert target.read_bytes() == b"HUMAN"


def test_refused_write_cas_at_on_a_never_read_path_still_guards_a_later_write(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same first baseline is recorded when the comparand read itself is
    refused inside a peer's commit window (ViewWedged), not only when the CAS
    loses on version."""
    target, peer_vol, agent = volumes
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    try:
        with pytest.raises(ViewWedged):
            agent.write_cas_at(_PATH, 2, b"agent")
    finally:
        peer.finish()
    target.write_bytes(b"HUMAN")
    with pytest.raises(StaleView):
        agent.write(_PATH, b"agent")
    assert target.read_bytes() == b"HUMAN"


def test_failed_write_cas_at_read_on_a_never_read_path_still_guards_a_later_write(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first baseline is recorded before the comparand read's request, so a
    read that fails closed (the coordinator answers degraded) still leaves one."""
    target, _peer, agent = volumes
    real_post = agent._post

    def degraded_pre_read(endpoint_path: str, payload: dict) -> dict | None:
        if endpoint_path == "/hooks/pre-read":
            return {"ok": True, "degraded": True}
        return real_post(endpoint_path, payload)

    monkeypatch.setattr(agent, "_post", degraded_pre_read)
    with pytest.raises(CoherenceError):
        agent.write_cas_at(_PATH, 1, b"agent")
    monkeypatch.setattr(agent, "_post", real_post)
    target.write_bytes(b"HUMAN")
    with pytest.raises(StaleView):
        agent.write(_PATH, b"agent")
    assert target.read_bytes() == b"HUMAN"


def test_write_after_a_cas_lost_to_a_commit_still_reaching_disk_is_denied(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer's CAS is confirmed between write_cas_at's comparand read and its
    own CAS, and the peer's bytes have not reached disk. The disk still matches
    this volume's baseline, so only the coordinator can refuse the write() that
    follows the lost CAS: the comparand read registered this volume there, and
    the peer's commit invalidated it."""
    target, peer_vol, agent = volumes
    _data, version = agent.read_with_version(_PATH)
    peer = LaggingPeer(peer_vol, monkeypatch)
    real_post = agent._post
    raced = threading.Event()

    def post_with_a_peer_commit_first(endpoint_path: str, payload: dict) -> dict | None:
        if endpoint_path == "/hooks/post-edit-cas" and not raced.is_set():
            raced.set()
            peer.commit(b"1")
        return real_post(endpoint_path, payload)

    monkeypatch.setattr(agent, "_post", post_with_a_peer_commit_first)
    try:
        with pytest.raises(CasVersionConflict):
            agent.write_cas_at(_PATH, version, b"agent")
        with pytest.raises(StaleView):
            agent.write(_PATH, b"older")
    finally:
        peer.finish()
    assert raced.is_set()
    assert target.read_bytes() == b"1"
    assert agent.reacquire(_PATH) == b"1"

def _read_following_the_deny_guidance(
    vol: CoherentVolume, patience_sec: float = 3.0
) -> tuple[bytes, int] | bytes:
    """Recover from a refused read the way the deny text says to: reacquire and
    read again with backoff for a few seconds. Returns the pair once a read
    answers, or the reacquired bytes if the refusal outlasts the patience (the
    case the text says to write from)."""
    deadline = time.monotonic() + patience_sec
    wait = 0.01
    while True:
        try:
            return vol.read_with_version(_PATH)
        except StaleView:
            reacquired = vol.reacquire(_PATH)
            if time.monotonic() >= deadline:
                return reacquired
            time.sleep(wait)
            wait = min(wait * 2, 0.5)


def test_following_the_deny_guidance_through_a_slow_peer_commit_loses_nothing(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deny text tells a refused caller to retry for a few seconds before
    concluding the file changed outside the coordinator. That rests on the
    refusal clearing by itself once the peer's disk write lands, even after
    reacquire() calls inside the window. A caller that instead wrote after one
    refused retry landed its write inside the window, and the peer's disk write
    then overwrote it."""
    target, peer_vol, reader = volumes
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    peer.release_after(0.25)
    try:
        outcome = _read_following_the_deny_guidance(reader)
    finally:
        peer.finish()
    assert isinstance(outcome, tuple), "a 250ms commit window must clear within the retries"
    data, version = outcome
    assert data == b"1"
    reader.write_cas_at(_PATH, version, _increment(data))
    assert target.read_bytes() == b"2"


def test_read_with_version_still_answers_an_invalidated_reader_once_disk_landed(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of the refusal: a sticky-INVALID reader whose bytes DO
    match the coordinator's version is still denied, and that pair is sound, so
    it is returned as before (the documented sticky-INVALID read)."""
    _target, peer_vol, reader = volumes
    reader.read_with_version(_PATH)
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    peer.finish()
    # KTD-T keeps a strict deny sticky, so this probe leaves the reader INVALID.
    probe = reader._read_with_version(_PATH)
    assert probe.stale_denied is True, "precondition: the reader must still be INVALID"
    assert probe.content_differs is False, "precondition: its bytes must match the version"
    data, version = reader.read_with_version(_PATH)
    assert data == b"1"
    reader.write_cas_at(_PATH, version, _increment(data))
    assert (reader._root / _PATH).read_bytes() == b"2"


@_READER_STATES
def test_read_with_version_generation_refuses_a_split_pair_in_the_commit_window(
    volumes, monkeypatch: pytest.MonkeyPatch, read_before_peer: bool
) -> None:
    _target, peer_vol, reader = volumes
    if read_before_peer:
        reader.read_with_version_generation(_PATH)
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    try:
        with pytest.raises(StaleView) as exc:
            reader.read_with_version_generation(_PATH)
        assert exc.value.reason == _SPLIT_PAIR
        # The flags still describe the read that was refused.
        assert reader._last_read_denied is True
    finally:
        peer.finish()


def test_verification_read_reports_the_split_instead_of_raising(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """observe=False is the effect fence's re-validate read: it discards the
    bytes and classifies from the flags, so it must keep answering (the fence
    names the HOLD from ``_last_read_denied``) rather than raise untyped."""
    _target, peer_vol, reader = volumes
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    try:
        _data, _version, generation = reader.read_with_version_generation(
            _PATH, observe=False
        )
        assert reader._last_read_denied is True
        assert generation is None
    finally:
        peer.finish()


def test_gate_does_not_decide_from_bytes_read_in_the_commit_window(
    volumes, monkeypatch: pytest.MonkeyPatch
) -> None:
    _target, peer_vol, reader = volumes
    peer = LaggingPeer(peer_vol, monkeypatch)
    peer.commit(b"1")
    decided: list[bytes] = []
    try:
        with pytest.raises(StaleView):
            gate(
                reader,
                _PATH,
                decide=lambda data: decided.append(data),
                effect=lambda _decision: pytest.fail("effect fired on a split read"),
            )
    finally:
        peer.finish()
    assert decided == []


# --- one split pair, named the same on every surface (#405) ----------------
#
# The coordinator answers a split pair with one answer, a strict deny carrying
# ``hash_differs``, whichever surface asked. read() raising on a stale view
# raises the refusal the versioned reads raise, and the effect fence holds as
# ``split_pair``, never as ``read_denied``, which calls the view INVALID and
# names a reacquire that never clears it.

#: The start of the split-pair refusal's text, which an observing read raises.
_SPLIT_TEXT = "stale read refused: the bytes on disk are not the content the coordinator records"


def _raising_vol(tmp_path: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(
        tmp_path, managed=("data/**",), on_error="strict", on_stale_read="raise", config=cfg
    )


@_READER_STATES
def test_read_raising_on_a_stale_view_refuses_a_split_pair_in_the_commit_window(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    read_before_peer: bool,
) -> None:
    _seed(tmp_path)
    peer_vol = _vol(tmp_path, fast_cfg)
    reader = _raising_vol(tmp_path, fast_cfg)
    try:
        if read_before_peer:
            reader.read(_PATH)
        peer = LaggingPeer(peer_vol, monkeypatch)
        peer.commit(b"1")
        try:
            with pytest.raises(StaleView) as exc:
                reader.read(_PATH)
            assert exc.value.reason == _SPLIT_PAIR
            assert str(exc.value).startswith(_SPLIT_TEXT), str(exc.value)
        finally:
            peer.finish()
    finally:
        stop_coordinator(tmp_path)


def test_read_raising_on_a_stale_view_keeps_a_refusal_whose_bytes_match(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Control: a peer committed and its bytes have landed, so this reader is
    INVALID while the file holds exactly the content the coordinator
    records. The deny carries no ``hash_differs``: it is not a split pair,
    and ``read()`` raises the coordinator's own refusal, reason
    ``stale_view``, as before."""
    _seed(tmp_path)
    reader = _raising_vol(tmp_path, fast_cfg)
    peer = _vol(tmp_path, fast_cfg)
    try:
        reader.read(_PATH)
        peer.write(_PATH, b"1")

        with pytest.raises(StaleView) as refused:
            reader.read(_PATH)

        assert refused.value.reason == _STALE_VIEW
        assert not str(refused.value).startswith(_SPLIT_TEXT), str(refused.value)
        assert reader._last_read_denied is True, "control: the read was refused"
    finally:
        stop_coordinator(tmp_path)


def test_every_surface_names_a_file_changed_outside_the_coordinator_a_split_pair(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A strict reader reads v1, and the file is then changed outside the
    coordinator. ``read()`` (raising on a stale view) and both versioned
    reads raise the same refusal, reason ``split_pair``; the effect fence
    holds with ``hold_cause`` and ``reason`` ``split_pair``, and its text
    names the step that clears it, recording the bytes with ``write()``.

    Prevents one coordinator answer reading three ways: ``read()`` raised the
    class reason ``stale_view`` with the coordinator's text, and the fence
    held as ``read_denied``, "view is INVALID", whose reacquire never clears
    it, so an agent following it looped."""
    target = _seed(tmp_path)
    reader = _raising_vol(tmp_path, fast_cfg)
    try:
        _data, version, generation = reader.read_with_version_generation(_PATH)
        target.write_bytes(b"edited-outside-the-coordinator")

        for read in (reader.read, reader.read_with_version, reader.read_with_version_generation):
            with pytest.raises(StaleView) as refused:
                read(_PATH)
            assert refused.value.reason == _SPLIT_PAIR, read.__name__
            assert str(refused.value).startswith(_SPLIT_TEXT), (read.__name__, str(refused.value))
        with pytest.raises(StaleView) as held:
            check_fence(reader, _PATH, expected_version=version, expected_generation=generation)
        assert held.value.hold_cause == _SPLIT_PAIR
        assert held.value.reason == _SPLIT_PAIR
        assert "view is INVALID" not in str(held.value), str(held.value)
        assert "write()" in str(held.value), str(held.value)
    finally:
        stop_coordinator(tmp_path)


def test_a_split_pair_hold_clears_once_the_bytes_on_disk_are_written(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The recovery the hold names. A reacquire alone leaves the file's bytes
    unrecorded, so the fence holds as ``split_pair`` again; writing the bytes
    the reacquire returned records them, and the comparands of a fresh read
    then proceed."""
    target = _seed(tmp_path)
    reader = _raising_vol(tmp_path, fast_cfg)
    try:
        _data, version, generation = reader.read_with_version_generation(_PATH)
        target.write_bytes(b"edited-outside-the-coordinator")
        current = reader.reacquire(_PATH)
        assert current == b"edited-outside-the-coordinator"

        with pytest.raises(StaleView) as held:
            check_fence(reader, _PATH, expected_version=version, expected_generation=generation)
        assert held.value.hold_cause == _SPLIT_PAIR, "a reacquire alone cleared the split"

        reader.write(_PATH, current)
        _data, version, generation = reader.read_with_version_generation(_PATH)
        check_fence(reader, _PATH, expected_version=version, expected_generation=generation)
    finally:
        stop_coordinator(tmp_path)


def test_a_split_pair_hold_inside_a_peers_write_names_the_wait_first(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A peer's write() puts its bytes on disk before its commit moves the
    version, so a re-gate in that window is refused at an unchanged version:
    a ``split_pair`` hold. Its text gives the step that clears this case,
    re-gating after a few seconds, before the write() that records a change
    made outside the coordinator; once the peer's commit lands, the same gate
    holds as ``version_moved``.

    Prevents the hold sending a gated agent straight to write() a file a peer
    is still writing: that write preempts the peer's commit."""
    _seed(tmp_path)
    reader = _vol(tmp_path, fast_cfg)
    peer = _vol(tmp_path, fast_cfg)
    try:
        _data, version, generation = reader.read_with_version_generation(_PATH)
        at_commit, go = threading.Event(), threading.Event()
        real_post = peer._post

        def commit_waits(endpoint: str, payload: dict, **kwargs: object) -> object:
            if endpoint == "/hooks/post-edit" and payload.get("success") is True:
                at_commit.set()
                go.wait(_WAIT_SEC)
            return real_post(endpoint, payload, **kwargs)

        monkeypatch.setattr(peer, "_post", commit_waits)
        errors: list[BaseException] = []

        def run() -> None:
            try:
                peer.write(_PATH, b"peer-bytes")
            except BaseException as exc:  # surfaced below
                errors.append(exc)

        thread = threading.Thread(target=run, name="writing-peer")
        thread.start()
        try:
            assert at_commit.wait(_WAIT_SEC), "control: the peer's bytes are on disk, its commit unsent"
            with pytest.raises(StaleView) as held:
                check_fence(reader, _PATH, expected_version=version, expected_generation=generation)
            assert held.value.hold_cause == _SPLIT_PAIR
            text = str(held.value)
            assert "a peer's write still landing" in text, text
            assert text.index("after a few seconds") < text.index("write() the bytes"), text
        finally:
            go.set()
            thread.join(_WAIT_SEC)
        assert not thread.is_alive() and not errors, errors

        with pytest.raises(StaleView) as moved:
            check_fence(reader, _PATH, expected_version=version, expected_generation=generation)
        assert moved.value.hold_cause == "version_moved"
    finally:
        stop_coordinator(tmp_path)
