# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume.read_for_basis``, ``check_basis`` and ``commit_if_basis``.

A caller that read several files records, for each, the version and the
registration it read (``read_for_basis``), and later asks the coordinator
whether every one is still what it read (``check_basis``), or commits a result
only while they all are (``commit_if_basis``). These tests run real volumes
against a real coordinator serving in this process, so a test can read its
registry; a test scripts a coordinator answer only to reach an answer the
coordinator gives under a fault (a degraded or failed answer), and then says
so. What they pin is the client side: what each answer becomes, that silence
is never taken for unchanged, and that bytes reach disk only on a win.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import urllib.error
import uuid
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CasCommitResult, CoherentVolume
from ccs.cli._coherence_client import caller_principal_headers, resolve_endpoint, resolve_remote_endpoint
from ccs.core.basis import (
    BasisCheckAnswer,
    BasisCommitAnswer,
    BasisMember,
    CheckedMember,
    MovedMember,
    TargetRefusal,
)
from ccs.core.exceptions import (
    BasisCheckAbsent,
    CoherenceError,
    CommitUnconfirmed,
    CoordinatorFailed,
    CoordinatorUnreachable,
    GiverFenced,
    InvariantViolationError,
    StaleView,
)
from tests.coordinator_helpers import answer_with_status, recorded_version, serve_in_process, write_holders

_A = "data/a.txt"
_B = "data/b.txt"
_OUT = "data/out.txt"
_UNTRACKED = "notes/n.txt"
_CHECK = "/basis/check"
_COMMIT = "/basis/commit"
_PRE_READ = "/hooks/pre-read"
_BASIS_ROUTES = (_CHECK, _COMMIT)
#: Any well-formed registration; the requests that carry it never reach a row.
_SOME_REGISTRATION = "8f0c1d52-6a3e-4c1b-9b7e-2f4d5a6b7c8d"
_MODES = pytest.mark.parametrize("mode", ["strict", "degrade"])

# FROZEN duplicates of what the volume says, so a reworded text is a decision.
_NO_CONFIRMED_VERSION = (
    "cannot record a basis member for {rel}: the coordinator confirmed no version for "
    "it (it does not track the file, or its answer did not confirm the read), and a "
    "member records only a confirmed version."
)
_NO_CONTENT_RECORDED = (
    "cannot record a basis member for {rel}: the coordinator holds no content for the "
    "file yet, so its version vouches for no bytes and it reports no registration to "
    "record. Reading again does not change that: a commit of the file through the "
    "coordinator (write(), write_cas or write_cas_at) records its content, and a read "
    "after that commit records the member."
)
_REGISTRATION_NOT_REPORTED = (
    "cannot record a basis member for {rel}: the coordinator answered the read without "
    "a registration, so it does not serve the basis check. Only the Python coordinator "
    "serves it; the Node coordinator and older Python coordinators do not."
)
_ROUTE_NOT_SERVED = (
    "the coordinator does not serve {route} (HTTP 404), so it cannot answer for a basis: "
    "nothing ran, and a retry gets the same answer. Only the Python coordinator serves "
    "the basis check; the Node coordinator and older Python coordinators do not."
)
_DETACHED = (
    "this CoherentVolume has no coordinator attached, so it cannot {verb}: no "
    "coordinator can confirm it, and silence is never taken for unchanged. Nothing was "
    "sent."
)
_OWN_GRANT_HELD = (
    "{rel} is held by this volume's own earlier write grant: the coordinator "
    "answered that grant's release with the grant still held, and refused this "
    "commit as another agent's. Nothing was committed. Retry once after a short "
    "wait, which sends the release again; if it fails again, report it to the "
    "operator."
)
#: The answers a coordinator gives when it cannot confirm one, frozen in wire
#: order: a check that compared nothing, and a commit whose outcome is unknown.
_CHECK_DEGRADED = {
    "answer": "hold", "moved": [], "reason": "version_unconfirmed", "recovery": "retry",
    "degraded": True, "held_by": "watchdog_timeout",
}
_COMMIT_OUTCOME_UNKNOWN = {
    "answer": "hold", "moved": [], "reason": "commit_unconfirmed", "degraded": True,
    "held_by": "watchdog_timeout",
}
_ENVELOPE = {"ok": False, "reason": "internal: RuntimeError"}


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``data/**`` serving in this process, with
    ``data/a.txt``, ``data/b.txt`` and ``data/out.txt`` on disk."""
    for rel, content in ((_A, b"a1"), (_B, b"b1"), (_OUT, b"out1")):
        _seed(tmp_path, rel, content)
    coordinator = serve_in_process(tmp_path, "basis-volume")
    yield coordinator
    coordinator.shutdown()


VolumeFactory = Callable[..., CoherentVolume]


@pytest.fixture
def volumes(tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer) -> Iterator[VolumeFactory]:
    """Build volumes on the served workspace; each is closed before the
    coordinator stops."""
    built: list[CoherentVolume] = []

    def build(on_error: str = "strict") -> CoherentVolume:
        volume = CoherentVolume(tmp_path, managed=("data/**",), on_error=on_error, config=fast_cfg)
        built.append(volume)
        return volume

    yield build
    for volume in reversed(built):
        try:
            volume.close()
        except CoherenceError:
            pass  # a test that cut the volume off its coordinator leaves its close unconfirmed


def _seed(root: Path, rel: str, content: bytes) -> Path:
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _registration(server: CoordinatorHTTPServer, rel: str) -> str:
    """The registration the registry reports for the one row named ``rel``."""
    (row,) = server.registry.basis_rows([], [rel])
    return row.registration


def _baseline(vol: CoherentVolume, rel: str) -> str | None:
    """The foreign-edit baseline the volume keeps for ``rel``."""
    return vol._last_observed_hash.get(vol._key_of(rel))


def _wire(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict]]:
    """Record the route and payload of every request the volumes send from here on."""
    real_post = coherent_volume_module._coordinator_post
    sent: list[tuple[str, dict]] = []

    def recording(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        sent.append((path, dict(payload)))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", recording)
    return sent


def _answer(monkeypatch: pytest.MonkeyPatch, route: str, body: object) -> list[dict]:
    """Answer every request to ``route`` with ``body`` instead of sending it,
    as a coordinator under that fault would; every other request goes to the
    coordinator. Returns the payloads ``route`` was asked."""
    real_post = coherent_volume_module._coordinator_post
    asked: list[dict] = []

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path != route:
            return real_post(endpoint, path, payload, **kwargs)
        asked.append(dict(payload))
        return copy.deepcopy(body)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", post)
    return asked


def _basis_requests(sent: list[tuple[str, dict]]) -> list[str]:
    return [route for route, _payload in sent if route in _BASIS_ROUTES]


class _InThread:
    """Run ``call`` on its own thread; :meth:`result` joins it, bounded."""

    def __init__(self, call: Callable[[], object]) -> None:
        self.done = threading.Event()
        self._call = call
        self._outcome: tuple[str, object] = ("pending", None)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        try:
            self._outcome = ("returned", self._call())
        except BaseException as exc:  # handed to the caller of result()
            self._outcome = ("raised", exc)
        finally:
            self.done.set()

    def result(self, timeout: float = 10.0) -> object:
        assert self.done.wait(timeout), "the call did not finish in time"
        kind, value = self._outcome
        if kind == "raised":
            raise value  # type: ignore[misc]
        return value


# -- read_for_basis -----------------------------------------------------------


def test_read_for_basis_returns_the_bytes_and_the_member_the_coordinator_records(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The member carries the path as resolved, the version the coordinator
    records and the registration of the row it read, so a later check can
    tell that row from one registered in its place. The read asks for the
    registration, and its bytes become the foreign-edit baseline."""
    vol = volumes()
    sent = _wire(monkeypatch)

    data, member = vol.read_for_basis(_A)

    assert data == b"a1"
    assert member == BasisMember(_A, recorded_version(server, _A), _registration(server, _A), "file")
    assert [payload.get("want_registration") for route, payload in sent if route == _PRE_READ] == [True]
    assert _baseline(vol, _A) == _sha(b"a1")


def test_the_version_reads_send_and_return_what_they_did_before(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``read_with_version`` and ``read_with_version_generation`` send the same
    pre-read as before the basis reads existed, without the registration
    opt-in, and return the same shapes.

    Prevents a version read whose request changed under its existing callers."""
    vol = volumes()
    sent = _wire(monkeypatch)

    assert vol.read_with_version(_A) == (b"a1", 1)
    assert vol.read_with_version_generation(_B) == (b"b1", 1, 0)

    reads = [payload for route, payload in sent if route == _PRE_READ]
    assert [sorted(payload) for payload in reads] == [
        ["agent_id", "content_hash", "path", "session_id", "verify_only", "want_owner_generation"],
    ] * 2


def test_read_for_basis_of_an_untracked_file_raises_and_keeps_the_baseline(
    tmp_path: Path, volumes: VolumeFactory,
) -> None:
    """The coordinator reports no version for a file it does not track, so no
    member can be recorded for it."""
    _seed(tmp_path, _UNTRACKED, b"n1")
    vol = volumes()
    vol.read(_UNTRACKED)
    _seed(tmp_path, _UNTRACKED, b"n2")

    with pytest.raises(CoherenceError) as raised:
        vol.read_for_basis(_UNTRACKED)

    assert type(raised.value) is CoherenceError, repr(raised.value)
    assert str(raised.value) == _NO_CONFIRMED_VERSION.format(rel=_UNTRACKED)
    assert _baseline(vol, _UNTRACKED) == _sha(b"n1"), "a refused read moved the baseline"


@_MODES
def test_read_for_basis_answered_degraded_raises_in_both_modes_and_keeps_the_baseline(
    tmp_path: Path, volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """A degraded answer confirms nothing: strict raises the watchdog failure,
    and degrade warns and then refuses the read, rather than record a member
    nobody vouched for. The answer is scripted, as the watchdog sends it."""
    vol = volumes(on_error=mode)
    vol.read(_A)
    _seed(tmp_path, _A, b"a2")
    _answer(monkeypatch, _PRE_READ, {"status": "fresh", "degraded": True})

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(CoherenceError) as raised:
            vol.read_for_basis(_A)

    assert type(raised.value) is CoherenceError, repr(raised.value)
    if mode == "strict":
        assert str(raised.value) == f"coordinator watchdog timeout during read of {_A}"
    else:
        assert str(raised.value) == _NO_CONFIRMED_VERSION.format(rel=_A)
        assert vol.degradation_count == 1
    assert _baseline(vol, _A) == _sha(b"a1"), "a refused read moved the baseline"


def test_read_for_basis_answered_degraded_with_a_version_still_raises(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degrade volume whose read the coordinator answered degraded refuses
    to record a member even when the answer carries a version and a
    registration: the read was not confirmed, whatever the body says."""
    vol = volumes(on_error="degrade")
    _answer(monkeypatch, _PRE_READ, {
        "status": "fresh", "degraded": True, "version": 1, "registration": _SOME_REGISTRATION,
    })

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with pytest.raises(CoherenceError) as raised:
            vol.read_for_basis(_A)

    assert str(raised.value) == _NO_CONFIRMED_VERSION.format(rel=_A)


def test_read_for_basis_refused_after_a_peer_commit_raises_stale_view_and_keeps_the_baseline(
    volumes: VolumeFactory,
) -> None:
    """A peer committed the file after this volume read it: the strict read is
    refused, and no member is recorded from it."""
    vol, peer = volumes(), volumes()
    vol.read(_A)
    peer.write(_A, b"a2")

    with pytest.raises(StaleView) as raised:
        vol.read_for_basis(_A)

    assert raised.value.reason != "split_pair", "control: the bytes on disk are the peer's recorded ones"
    assert _baseline(vol, _A) == _sha(b"a1"), "a refused read moved the baseline"


def test_read_for_basis_of_bytes_the_coordinator_does_not_record_raises_split_pair(
    tmp_path: Path, volumes: VolumeFactory,
) -> None:
    """The file changed on disk outside the coordinator after this volume
    read it: the bytes are not the content at the version, so the read raises
    the split-pair refusal and records nothing."""
    vol = volumes()
    vol.read(_A)
    _seed(tmp_path, _A, b"foreign")

    with pytest.raises(StaleView) as raised:
        vol.read_for_basis(_A)

    assert raised.value.reason == "split_pair"
    assert _baseline(vol, _A) == _sha(b"a1"), "a refused read moved the baseline"


def test_read_for_basis_admitted_with_differing_content_raises_split_pair(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admitted read whose bytes the coordinator says are not its content
    at that version (a path it tracks without strict mode, or this volume's
    own commit still reaching disk) records no member either. The answer is
    the coordinator's own with ``hash_differs`` set, as those paths answer."""
    vol = volumes()
    real_post = coherent_volume_module._coordinator_post

    def differing(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        answer = real_post(endpoint, path, payload, **kwargs)
        if path == _PRE_READ:
            assert "registration" in answer and answer.get("status") == "fresh", f"control: admitted: {answer!r}"
            return {**answer, "hash_differs": True}
        return answer

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", differing)
    with pytest.raises(StaleView) as raised:
        vol.read_for_basis(_A)

    assert raised.value.reason == "split_pair"


def test_read_for_basis_of_a_file_with_no_recorded_content_names_the_cause_and_the_remedy(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, tmp_path: Path,
) -> None:
    """A file first seen through a shell read has a row that records no
    content, so the coordinator has no registration to report. The read
    raises, naming why and what records it, every time it is asked, and a
    commit of the file clears it."""
    seed = coherent_volume_module._coordinator_post(
        resolve_endpoint(tmp_path), "/hooks/pre-bash",
        {"session_id": str(uuid.uuid4()), "command": f"cat {_B}"},
    )
    assert seed.get("status") == "fresh", seed
    artifact_id = server.registry.lookup_artifact_id_by_name(_B)
    assert server.registry.get_artifact(artifact_id).content_hash is None, "control: no content recorded"
    vol = volumes()

    for _attempt in range(2):
        with pytest.raises(CoherenceError) as raised:
            vol.read_for_basis(_B)
        assert type(raised.value) is CoherenceError, repr(raised.value)
        assert str(raised.value) == _NO_CONTENT_RECORDED.format(rel=_B)

    vol.write(_B, b"b2")
    assert vol.read_for_basis(_B) == (b"b2", BasisMember(_B, 2, _registration(server, _B)))


@_MODES
def test_a_read_answered_without_a_registration_raises_basis_check_absent(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """An admitted read that reports a version but no registration comes from
    a coordinator that does not serve the basis check. The answer is the
    coordinator's own with the key removed, as an older one answers."""
    vol = volumes(on_error=mode)
    real_post = coherent_volume_module._coordinator_post

    def older(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        answer = real_post(endpoint, path, payload, **kwargs)
        if path == _PRE_READ:
            assert "registration" in answer, f"control: the coordinator reported one: {answer!r}"
            return {key: value for key, value in answer.items() if key != "registration"}
        return answer

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", older)
    with pytest.raises(BasisCheckAbsent) as raised:
        vol.read_for_basis(_A)

    assert str(raised.value) == _REGISTRATION_NOT_REPORTED.format(rel=_A)
    assert not vol.is_degraded


# -- check_basis --------------------------------------------------------------


def test_check_basis_proceeds_with_every_member_unmoved(volumes: VolumeFactory) -> None:
    vol = volumes()
    members = [vol.read_for_basis(_A)[1], vol.read_for_basis(_B)[1]]

    assert vol.check_basis(members) == BasisCheckAnswer(
        "proceed", checked=(CheckedMember(_A, 1, 1), CheckedMember(_B, 1, 1)),
    )


def test_check_basis_holds_with_every_member_a_peer_moved(volumes: VolumeFactory) -> None:
    vol, peer = volumes(), volumes()
    members = [vol.read_for_basis(_A)[1], vol.read_for_basis(_B)[1]]
    peer.write(_A, b"a2")
    peer.write(_B, b"b2")

    assert vol.check_basis(members) == BasisCheckAnswer("hold", moved=(
        MovedMember(_A, 1, 2, "version_moved"), MovedMember(_B, 1, 2, "version_moved"),
    ), recovery="rederive")


def test_check_basis_holds_after_a_peer_re_saves_identical_bytes(volumes: VolumeFactory) -> None:
    """Only versions are compared: identical bytes committed again move the
    version, and the check holds."""
    vol, peer = volumes(), volumes()
    _data, member = vol.read_for_basis(_A)
    peer.write(_A, b"a1")

    assert vol.check_basis([member]) == BasisCheckAnswer(
        "hold", moved=(MovedMember(_A, 1, 2, "version_moved"),), recovery="rederive",
    )


def test_a_member_of_the_other_kind_is_refused_before_anything_is_sent(
    tmp_path: Path, fast_cfg: LifecycleConfig, volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local volume's members name files, and a remote volume's name keys:
    a member of the other kind is refused before any request."""
    local = volumes()
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    endpoint = resolve_endpoint(tmp_path)
    (tmp_path / "remote-client").mkdir()
    remote = CoherentVolume(
        tmp_path / "remote-client", config=fast_cfg,
        remote_endpoint=resolve_remote_endpoint("127.0.0.1", endpoint.port, endpoint.bearer),
    )
    try:
        sent = _wire(monkeypatch)
        for vol, kind in ((local, "key"), (remote, "file")):
            member = BasisMember(_A, 1, name_kind=kind)
            with pytest.raises(ValueError, match="name_kind"):
                vol.check_basis([member])
            with pytest.raises(ValueError, match="name_kind"):
                vol.commit_if_basis([member], _OUT, 1, b"out2")
        assert _basis_requests(sent) == []
    finally:
        remote.close()


def test_a_remote_volume_records_and_checks_key_members(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A remote volume's disk is not the coordinator's, so its members name
    keys: the member it records is a key with the registration of the key's
    row, the check finds that row and holds once a peer commits the key, and a
    commit under it writes the volume's own file."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    endpoint = resolve_endpoint(tmp_path)
    remote = resolve_remote_endpoint("127.0.0.1", endpoint.port, endpoint.bearer)
    key = "data/k.txt"
    roots = [tmp_path / "remote-a", tmp_path / "remote-b"]
    for root in roots:
        _seed(root, key, b"k1")
    vol, peer = (CoherentVolume(root, config=fast_cfg, remote_endpoint=remote) for root in roots)
    try:
        _data, member = vol.read_for_basis(key)
        assert member == BasisMember(key, 1, _registration(server, key), "key")
        assert not (tmp_path / key).exists(), "control: the key is no file of the coordinator's"
        assert vol.check_basis([member]).answer == "proceed"

        assert vol.commit_if_basis([member], key, 1, b"k2") == CasCommitResult(path=key, version=2)
        assert (roots[0] / key).read_bytes() == b"k2"

        _seed(roots[1], key, b"k2")  # the peer's own disk, which the coordinator does not share
        peer.write_cas_at(key, 2, b"k3")
        assert vol.check_basis([BasisMember(key, 2, member.registration, "key")]) == BasisCheckAnswer(
            "hold", moved=(MovedMember(key, 2, 3, "version_moved"),), recovery="rederive",
        )
    finally:
        vol.close()
        peer.close()


_LONG_PATH = "data/" + "/".join(["d" * 50] * 21)  # over 1024 characters, no segment too long for a disk

#: Each basis the coordinator refuses with 400, run through both calls.
_REFUSED_BASES: dict[str, list[BasisMember]] = {
    "no member": [],
    "65 members": [BasisMember(f"data/m{index}.txt", 1) for index in range(65)],
    "one path twice": [BasisMember(_A, 1), BasisMember(_A, 1)],
    "one file under two spellings": [BasisMember(_A, 1), BasisMember("data/./a.txt", 1)],
    "one registration twice": [BasisMember(_A, 1, _SOME_REGISTRATION), BasisMember(_B, 1, _SOME_REGISTRATION)],
    "an absolute path": [BasisMember("/data/a.txt", 1)],
    "a traversal": [BasisMember("data/../a.txt", 1)],
    "a control character": [BasisMember("data/a\x07.txt", 1)],
    "a path over the length limit": [BasisMember(_LONG_PATH, 1)],
}
#: Each target the coordinator refuses with 400: the basis, the target and its version.
_REFUSED_TARGETS: dict[str, tuple[list[BasisMember], str, object]] = {
    "a target with a control character": ([BasisMember(_A, 1)], "data/out\x07.txt", 1),
    "a target over the length limit": ([BasisMember(_A, 1)], _LONG_PATH, 1),
    "a target that disagrees with its member": ([BasisMember(_A, 1)], _A, 2),
    "a target at version 0": ([BasisMember(_A, 1)], _OUT, 0),
    "a target version that is a bool": ([BasisMember(_A, 1)], _OUT, True),
}
_REFUSALS = [
    *((name, _CHECK, basis, None) for name, basis in _REFUSED_BASES.items()),
    *((name, _COMMIT, basis, (_OUT, 1)) for name, basis in _REFUSED_BASES.items()),
    *((name, _COMMIT, basis, (target, version)) for name, (basis, target, version) in _REFUSED_TARGETS.items()),
]


def _target_on_the_wire(target: str, version: object) -> dict:
    """The target as a caller would describe it on the wire: the path at the version."""
    return {"path": target, "version": version, "registration": None, "name_kind": "file"}


@pytest.mark.parametrize(
    ("route", "basis", "target"),
    [pytest.param(route, basis, target, id=f"{route}-{name}") for name, route, basis, target in _REFUSALS],
)
def test_a_request_the_coordinator_would_refuse_is_refused_locally_with_its_text(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch,
    route: str, basis: list[BasisMember], target: tuple[str, object] | None,
) -> None:
    """Each basis or target the coordinator answers 400 is refused before any
    request, with the coordinator's own text, so a caller's mistake never
    reaches the wire and never counts a degrade volume degraded. The same
    request sent as it stands is the coordinator's control."""
    vol = volumes(on_error="degrade")
    real_post = coherent_volume_module._coordinator_post
    sent = _wire(monkeypatch)

    with pytest.raises(ValueError) as raised:
        if target is None:
            vol.check_basis(basis)
        else:
            vol.commit_if_basis(basis, target[0], target[1], b"out2")

    assert _basis_requests(sent) == [], "a request the coordinator refuses was sent"
    assert vol.degradation_count == 0
    body: dict = {"session_id": vol.session_id, "agent_id": vol.incarnation, "basis": [m.to_wire() for m in basis]}
    if target is not None:
        body.update(target=_target_on_the_wire(*target), content_hash=_sha(b"out2"))
    with pytest.raises(urllib.error.HTTPError) as refused:
        real_post(vol._endpoint, route, body, extra_headers=caller_principal_headers(vol._principal))
    assert refused.value.code == 400
    assert str(raised.value) == json.loads(refused.value.read())["error"]


# -- no usable answer ---------------------------------------------------------

#: What each call raises, by mode and cause, when it gets no answer it can use.
_UNANSWERED: dict[tuple[str, str], dict[str, type[CoherenceError]]] = {
    ("strict", "stopped"): {"check": CoordinatorUnreachable, "commit": CoordinatorUnreachable},
    ("degrade", "stopped"): {"check": CoherenceError, "commit": CommitUnconfirmed},
    ("strict", "5xx"): {"check": CoordinatorFailed, "commit": CommitUnconfirmed},
    ("degrade", "5xx"): {"check": CoherenceError, "commit": CommitUnconfirmed},
    ("strict", "4xx"): {"check": CoherenceError, "commit": CoherenceError},
    ("degrade", "4xx"): {"check": CoherenceError, "commit": CommitUnconfirmed},
}


def _call(vol: CoherentVolume, op: str, member: BasisMember) -> object:
    if op == "check":
        return vol.check_basis([member])
    return vol.commit_if_basis([member], _OUT, 1, b"out2")


@pytest.mark.parametrize("op", ["check", "commit"])
@pytest.mark.parametrize(("mode", "cause"), list(_UNANSWERED))
def test_a_call_with_no_usable_answer_raises_in_both_modes(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    mode: str, cause: str, op: str,
) -> None:
    """A stopped coordinator, an HTTP 5xx or an HTTP 4xx raises in both
    ``on_error`` modes, never a proceed: strict raises by what it knows of the
    request, degrade warns and then raises. A commit's bytes never reach disk."""
    _seed(tmp_path, _A, b"a1")
    target = _seed(tmp_path, _OUT, b"out1")
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error=mode, config=fast_cfg)
    try:
        member = vol.read_for_basis(_A)[1]
        vol.read_for_basis(_OUT)
        if cause == "stopped":
            stop_coordinator(tmp_path)
        else:
            status = 500 if cause == "5xx" else 409
            answer_with_status(monkeypatch, lambda path, _p: path in _BASIS_ROUTES, status, forwarded=False)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            with pytest.raises(CoherenceError) as raised:
                _call(vol, op, member)

        assert type(raised.value) is _UNANSWERED[(mode, cause)][op], repr(raised.value)
        assert target.read_bytes() == b"out1", "an unconfirmed commit touched disk"
    finally:
        monkeypatch.undo()
        try:
            vol.close()
        except CoherenceError:
            pass  # a stopped coordinator cannot confirm the close
        stop_coordinator(tmp_path)


@_MODES
@pytest.mark.parametrize("op", ["read", "check", "commit"])
def test_a_detached_volume_raises_and_sends_nothing(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, op: str,
) -> None:
    """A volume with no coordinator attached raises in both modes: silence is
    never taken for unchanged, and a degrade volume's best-effort mode does
    not reach these calls."""
    vol = volumes(on_error=mode)
    endpoint, vol._endpoint = vol._endpoint, None  # detached, as a degrade volume whose attach failed runs
    sent = _wire(monkeypatch)
    verb = {"read": "record a basis member", "check": "check a basis", "commit": "commit under a basis"}[op]

    try:
        with pytest.raises(CoherenceError) as raised:
            if op == "read":
                vol.read_for_basis(_A)
            else:
                _call(vol, op, BasisMember(_A, 1))
    finally:
        vol._endpoint = endpoint

    assert type(raised.value) is CoherenceError, repr(raised.value)
    assert str(raised.value) == _DETACHED.format(verb=verb)
    assert sent == []
    assert (tmp_path / _OUT).read_bytes() == b"out1"


@_MODES
@pytest.mark.parametrize("op", ["check", "commit"])
def test_a_coordinator_without_the_basis_routes_raises_basis_check_absent(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, op: str,
) -> None:
    """A 404 on a basis route means the coordinator does not serve it, so
    nothing ran: ``BasisCheckAbsent`` in both modes, not counted degraded."""
    vol = volumes(on_error=mode)
    member = vol.read_for_basis(_A)[1]
    answer_with_status(monkeypatch, lambda path, _p: path in _BASIS_ROUTES, 404, forwarded=False)

    with pytest.raises(BasisCheckAbsent) as raised:
        _call(vol, op, member)

    route = _CHECK if op == "check" else _COMMIT
    assert str(raised.value) == _ROUTE_NOT_SERVED.format(route=route)
    assert not vol.is_degraded
    assert (tmp_path / _OUT).read_bytes() == b"out1"


@_MODES
def test_a_degraded_check_answer_is_a_hold_to_retry(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """The check the coordinator answers degraded compared nothing: a hold
    with ``version_unconfirmed`` and ``retry``, never a proceed."""
    vol = volumes(on_error=mode)
    member = vol.read_for_basis(_A)[1]
    _answer(monkeypatch, _CHECK, _CHECK_DEGRADED)

    assert vol.check_basis([member]) == BasisCheckAnswer(
        "hold", reason="version_unconfirmed", recovery="retry",
    )


_EITHER_MODE = {"strict": CoherenceError, "degrade": CoherenceError}
#: Answers to a check that confirm nothing, and what each raises by mode. A
#: strict volume's client already refuses JSON that is not an object as no
#: answer at all.
_UNUSABLE_CHECKS: dict[str, tuple[object, dict[str, type[CoherenceError]]]] = {
    "failure envelope": (_ENVELOPE, {"strict": CoordinatorFailed, "degrade": CoordinatorFailed}),
    "unknown answer": ({"answer": "maybe"}, _EITHER_MODE),
    "no answer key": ({"ok": True}, _EITHER_MODE),
    "proceed missing a member": (
        {"answer": "proceed", "checked": {_A: {"expected": 1, "current": 1}}}, _EITHER_MODE,
    ),
    "proceed at another version": (
        {"answer": "proceed", "checked": {_A: {"expected": 1, "current": 1}, _B: {"expected": 2, "current": 2}}},
        _EITHER_MODE,
    ),
    "not an object": ([], {"strict": CoordinatorUnreachable, "degrade": CoherenceError}),
}


@_MODES
@pytest.mark.parametrize("answer", list(_UNUSABLE_CHECKS))
def test_a_check_answer_that_confirms_nothing_raises(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, mode: str, answer: str,
) -> None:
    """A check answered with the failure envelope, an answer the client
    cannot classify, or a proceed that does not confirm every member at the
    version recorded for it raises in both modes, never a proceed."""
    vol = volumes(on_error=mode)
    members = [vol.read_for_basis(_A)[1], vol.read_for_basis(_B)[1]]
    body, raises = _UNUSABLE_CHECKS[answer]
    _answer(monkeypatch, _CHECK, body)

    with pytest.raises(CoherenceError) as raised:
        vol.check_basis(members)

    assert type(raised.value) is raises[mode], repr(raised.value)


# -- commit_if_basis ----------------------------------------------------------


@pytest.mark.parametrize("target_in_basis", [True, False], ids=["target-in-basis", "target-outside"])
def test_commit_if_basis_writes_the_bytes_on_a_win_and_returns_the_committed_version(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, tmp_path: Path, target_in_basis: bool,
) -> None:
    """Nothing moved: the target is committed at the next version, its bytes
    reach disk, they become the volume's baseline, and no other member's
    version moves."""
    vol = volumes()
    _data, member = vol.read_for_basis(_A)
    _out, out = vol.read_for_basis(_OUT)
    basis = [member, out] if target_in_basis else [member]

    result = vol.commit_if_basis(basis, _OUT, out.version, b"out2")

    assert result == CasCommitResult(path=_OUT, version=out.version + 1)
    assert (tmp_path / _OUT).read_bytes() == b"out2"
    assert recorded_version(server, _OUT) == out.version + 1
    assert recorded_version(server, _A) == member.version, "a member that is not the target moved"
    assert _baseline(vol, _OUT) == _sha(b"out2")


def test_commit_if_basis_writes_nothing_when_an_input_moved(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, tmp_path: Path,
) -> None:
    vol, peer = volumes(), volumes()
    _data, member = vol.read_for_basis(_A)
    _out, out = vol.read_for_basis(_OUT)
    peer.write(_A, b"a2")

    answer = vol.commit_if_basis([member], _OUT, out.version, b"out2")

    assert answer == BasisCommitAnswer(
        "hold", moved=(MovedMember(_A, 1, 2, "version_moved"),), recovery="rederive",
    )
    assert (tmp_path / _OUT).read_bytes() == b"out1"
    assert recorded_version(server, _OUT) == out.version


def test_commit_if_basis_writes_nothing_while_a_peer_holds_the_target(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, tmp_path: Path,
) -> None:
    """The basis held, and a peer's write grant on the target refused the
    target's compare-and-swap: the hold carries the refusal apart from the
    members, with no recovery class, and nothing is written."""
    vol, peer = volumes(), volumes()
    _data, member = vol.read_for_basis(_A)
    peer.write(_OUT, b"from the peer")
    _out, out = vol.read_for_basis(_OUT)
    assert write_holders(server, _OUT) == {peer.agent_id}, "control: the peer still holds the target"

    answer = vol.commit_if_basis([member], _OUT, out.version, b"out2")

    assert isinstance(answer, BasisCommitAnswer) and answer.answer == "hold", answer
    assert (answer.moved, answer.recovery) == ((), None)
    assert answer.target is not None and (answer.target.path, answer.target.reason) == (_OUT, "other_holder")
    assert (tmp_path / _OUT).read_bytes() == b"from the peer"
    assert recorded_version(server, _OUT) == out.version


def test_a_commit_refused_by_this_volumes_own_held_grant_names_that_grant(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """This volume's own ``write()`` left the target held, and the release its
    commit sends first was answered with the grant still held: the target's
    refusal is raised as ``write_cas_at`` raises it, naming that grant, never
    returned as a peer's hold to wait out."""
    vol = volumes()
    vol.write(_OUT, b"out2")
    _data, member = vol.read_for_basis(_A)
    _out, out = vol.read_for_basis(_OUT)

    def refusing(*_args: object, **_kwargs: object) -> object:
        raise InvariantViolationError("release refused")

    monkeypatch.setattr(server.service, "release_to_reader", refusing)
    with pytest.raises(CoordinatorFailed) as raised:
        vol.commit_if_basis([member], _OUT, out.version, b"out3")

    assert str(raised.value) == _OWN_GRANT_HELD.format(rel=_OUT)
    assert (tmp_path / _OUT).read_bytes() == b"out2"


#: Target refusals answered as the coordinator words them, and what each becomes.
_TARGET_REFUSALS: dict[str, tuple[dict, type[CoherenceError] | None]] = {
    "handed_off": ({
        "path": _OUT, "reason": "handed_off", "current_version": 1,
        "successor": "4f1b7c2e-0000-4000-8000-000000000001", "version_at_transfer": 1,
    }, GiverFenced),
    "occ_is_shared_or_invalid_only": (
        {"path": _OUT, "reason": "occ_is_shared_or_invalid_only", "current_version": 1}, CoherenceError,
    ),
    "stale_read_generation": ({
        "path": _OUT, "reason": "stale_read_generation", "current_version": 1,
        "captured_generation": 0, "owner_generation": 1,
    }, None),
    "caller_in_transient_state": (
        {"path": _OUT, "reason": "caller_in_transient_state", "current_version": 1}, None,
    ),
}


@pytest.mark.parametrize("reason", list(_TARGET_REFUSALS))
def test_a_target_refusal_is_raised_or_returned_by_its_reason(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reason: str,
) -> None:
    """A fenced giver raises ``GiverFenced`` as ``write_cas_at`` does, with
    the successor and the version at transfer; a refusal of a caller holding
    the file for writing raises; a retryable refusal is returned as the hold.
    Nothing is written either way. The answers are the coordinator's wire
    forms, scripted."""
    vol = volumes()
    member = vol.read_for_basis(_A)[1]
    block, raises = _TARGET_REFUSALS[reason]
    _answer(monkeypatch, _COMMIT, {"answer": "hold", "moved": [], "target": block})

    if raises is None:
        answer = vol.commit_if_basis([member], _OUT, 1, b"out2")
        assert answer == BasisCommitAnswer("hold", target=TargetRefusal.from_wire(block))
    else:
        with pytest.raises(CoherenceError) as raised:
            vol.commit_if_basis([member], _OUT, 1, b"out2")
        assert type(raised.value) is raises, repr(raised.value)
        if raises is GiverFenced:
            assert (raised.value.successor, raised.value.version_at_transfer) == (block["successor"], 1)
    assert (tmp_path / _OUT).read_bytes() == b"out1"


#: Answers to a commit that settle nothing: each may follow a commit that landed.
_UNSETTLED_COMMITS: dict[str, object] = {
    "outcome unknown": _COMMIT_OUTCOME_UNKNOWN,
    "unconfirmed without the degraded flag": {"answer": "hold", "moved": [], "reason": "commit_unconfirmed"},
    "failure envelope": _ENVELOPE,
    "no answer key": {"ok": True},
    "unknown answer": {"answer": "maybe"},
    "proceed naming another path": {"answer": "proceed", "committed": {"path": _A, "version": 2}},
    "proceed at another version": {"answer": "proceed", "committed": {"path": _OUT, "version": 7}},
    "not an object": [],
}


@_MODES
@pytest.mark.parametrize("answer", list(_UNSETTLED_COMMITS))
def test_a_commit_answer_that_settles_nothing_raises_commit_unconfirmed_and_writes_nothing(
    volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str, answer: str,
) -> None:
    """The coordinator's outcome-unknown hold, its failure envelope, or an
    answer the client cannot classify may follow a commit that landed:
    ``CommitUnconfirmed`` in both modes, and the bytes never reach disk."""
    vol = volumes(on_error=mode)
    member = vol.read_for_basis(_A)[1]
    _answer(monkeypatch, _COMMIT, _UNSETTLED_COMMITS[answer])

    with pytest.raises(CommitUnconfirmed):
        vol.commit_if_basis([member], _OUT, 1, b"out2")

    assert (tmp_path / _OUT).read_bytes() == b"out1"
    assert _baseline(vol, _OUT) != _sha(b"out2"), "an unconfirmed commit became the baseline"


# -- the guarded run prevents what an unguarded one loses ----------------------


def test_an_input_moved_before_the_commit_holds_where_an_unguarded_commit_loses_it(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, tmp_path: Path,
) -> None:
    """``out.txt`` was derived from ``a.txt`` at version 1, and a peer then
    moved ``a.txt``. The check and the commit under the basis hold and write
    nothing; a compare-and-swap of ``out.txt`` alone lands the stale
    derivation."""
    vol, peer = volumes(), volumes()
    _data, member = vol.read_for_basis(_A)
    _out, out = vol.read_for_basis(_OUT)
    peer.write(_A, b"a2")
    moved = (MovedMember(_A, 1, 2, "version_moved"),)

    assert vol.check_basis([member, out]) == BasisCheckAnswer("hold", moved=moved, recovery="rederive")
    assert vol.commit_if_basis([member, out], _OUT, out.version, b"derived from a1") == BasisCommitAnswer(
        "hold", moved=moved, recovery="rederive",
    )
    assert (tmp_path / _OUT).read_bytes() == b"out1"

    unguarded = vol.write_cas_at(_OUT, out.version, b"derived from a1")
    assert unguarded == CasCommitResult(path=_OUT, version=out.version + 1), "control: the plain commit lands"


def test_a_peer_forced_between_the_compare_and_the_target_write_lands_after_the_commit(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """The basis compare and the target's compare-and-swap are one hold: a
    peer's commit of a member, started while the commit sits between the two,
    lands only after the commit returns, and the commit proceeds on the basis
    it compared."""
    vol, peer = volumes(), volumes()
    _data, member = vol.read_for_basis(_A)
    _out, out = vol.read_for_basis(_OUT)
    entered, release = threading.Event(), threading.Event()
    real = server.registry.commit_cas_if_basis

    def delayed(*args: object, **kwargs: object) -> object:
        entered.set()
        release.wait(timeout=10.0)
        return real(*args, **kwargs)

    monkeypatch.setattr(server.registry, "commit_cas_if_basis", delayed)
    commit = _InThread(lambda: vol.commit_if_basis([member, out], _OUT, out.version, b"out2"))
    try:
        assert entered.wait(5.0), "the commit never reached its compare-and-swap"
        peer_write = _InThread(lambda: peer.write_cas_at(_A, member.version, b"a2"))
        peer_waited = not peer_write.done.wait(0.3)
    finally:
        release.set()

    assert commit.result() == CasCommitResult(path=_OUT, version=out.version + 1)
    assert peer_waited, "the peer's commit landed inside the commit's hold"
    assert peer_write.result() == CasCommitResult(path=_A, version=member.version + 1)
    assert (tmp_path / _OUT).read_bytes() == b"out2"


def test_nothing_moved_proceeds_and_a_second_readers_basis_still_proceeds_after_the_commit(
    server: CoordinatorHTTPServer, volumes: VolumeFactory, tmp_path: Path,
) -> None:
    """Two readers derive from ``a.txt``. The first commits its result under
    its basis; that commit moves only its own target, so the second reader's
    basis on ``a.txt`` still proceeds and its own commit lands."""
    first, second = volumes(), volumes()
    _seed(tmp_path, "data/second.txt", b"s1")
    first_basis = [first.read_for_basis(_A)[1]]
    second_basis = [second.read_for_basis(_A)[1]]
    first_out = first.read_for_basis(_OUT)[1]
    second_out = second.read_for_basis("data/second.txt")[1]

    assert first.check_basis(first_basis).answer == "proceed"
    assert first.commit_if_basis(first_basis, _OUT, first_out.version, b"first") == CasCommitResult(
        path=_OUT, version=first_out.version + 1,
    )
    assert second.check_basis(second_basis) == BasisCheckAnswer("proceed", checked=(CheckedMember(_A, 1, 1),))
    assert second.commit_if_basis(second_basis, "data/second.txt", second_out.version, b"second") == (
        CasCommitResult(path="data/second.txt", version=second_out.version + 1)
    )
    assert recorded_version(server, _A) == 1
