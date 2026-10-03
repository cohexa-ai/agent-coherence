# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""WorkspaceVersioner capture engine — WV plan Unit 3 (R1/R2/R8).

Covers, per the unit's test scenarios:

- clean capture over heterogeneous members (file + S3 + forward-only): tiers
  honest per member, the restore POINTER manifested (S3 versionId / file
  coordinator version — never the ETag CAS comparand), the skew-declared
  window recorded as [min, max] of the capture timestamps;
- ABSENT is a fact distinct from present-empty (no token/fingerprint vs
  ``sha256(b"")``);
- torn-cut detection: a write landing inside the window flags EXACTLY that
  member ``dirty_during_window`` (driven deterministically via the scripted
  file fake AND a second S3 writer racing between capture and verify);
- the unversioned-S3 honest refusal path (typed discovery, member described
  but ``forward_only`` — never ``restorable``);
- forward-only members enumerated, never token-captured;
- binary file member → typed capture-time refusal, nothing persisted;
- coordinator down → typed ``CheckpointPersistFailed``, NO partial manifest;
  and the abort Event threads end-to-end into the registry's ``abort_guard``;
- route level (the live-server house pattern): ``POST /workspace/checkpoint``
  round-trips through ``GET /workspace/checkpoints``; boundary validation
  fails closed on malformed member rows.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_object import CoherentObject
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.adapters.workspace import (
    BINARY_FILE_MEMBER_REASON,
    CHECKPOINT_NOT_PERSISTED_REASON,
    MAX_RESTORE_LEG_REDRIVES,
    STRUCTURAL_MEMBER_REFUSAL_REASON,
    BinaryFileMemberRefused,
    CheckpointPersistFailed,
    MemberRestoreOutcome,
    RestoreObservation,
    StructuralMemberRefused,
    WorkspaceVersioner,
)
from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.core.exceptions import (
    CHECKPOINT_ALREADY_REGISTERED_REASON,
    CHECKPOINT_NOT_THE_RECEIVER_REASON,
    CHECKPOINT_UNKNOWN_REASON,
    PIN_STATE_HELD,
    PIN_STATE_RELEASED,
    PIN_STATE_UNAVAILABLE,
    PIN_STATE_UNPINNED,
    RESTORE_MEMBER_OUTCOMES,
    RESTORE_OBSERVATION_DIFFERS,
    RESTORE_OBSERVATION_NO_LIVE_STATE,
    RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
    RESTORE_OBSERVATION_NOT_RECORDED,
    RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE,
    RESTORE_OBSERVATION_STATES,
    RESTORE_OUTCOME_CONFLICT,
    RESTORE_OUTCOME_CONVERGED,
    RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED,
    RESTORE_OUTCOME_HELD_UNCONFIRMED,
    RESTORE_OUTCOME_RESTORED,
    RESTORE_OUTCOME_TARGET_LOST,
    RESTORE_OUTCOMES_PROVING_NO_WRITE,
    RESTORE_STATUS_CONCLUDED,
    RESTORE_STATUS_IN_PROGRESS,
    RESTORE_STATUS_NONE,
    RESTORE_STATUS_REGISTERED,
    STALE_READ_GENERATION_REASON,
    WORKSPACE_REGISTRATION_COMMITTED,
    WORKSPACE_REGISTRATION_EMPTY,
    WORKSPACE_REGISTRATION_PRIOR_RUN,
    WORKSPACE_REGISTRATION_REFUSED,
    CasVersionConflict,
    CheckpointRegistrationRefused,
    CheckpointUnknown,
    CommitUnconfirmed,
    OccCallerTransientError,
    StaleView,
    ViewWedged,
    WatchdogAbandoned,
)
from ccs.core.invariants import check_monotonic_version
from ccs.core.states import MESIState
from ccs.core.substrate import sha256_hex
from ccs.core.types import ConflictDetail
from ccs.testing.s3_local import LocalS3Client

OWNER = uuid.uuid4()


# ---------------------------------------------------------------------------
# Deterministic fakes
# ---------------------------------------------------------------------------


class _TickClock:
    """Injectable monotonic_seconds stand-in: strictly increasing int ticks."""

    def __init__(self, start: int = 100) -> None:
        self._next = start

    def __call__(self) -> int:
        tick = self._next
        self._next += 1
        return tick


class _ScriptedFileSource:
    """A ``read_with_version`` fake with a per-path response script.

    Each programmed response is either ``(bytes, version)`` or an exception
    instance to raise. Responses are consumed in order; the LAST one repeats —
    so a one-entry script models a quiescent member (capture and verify see
    the same state) and a two-entry script models a write landing inside the
    window (capture sees the first, verify sees the second).
    """

    def __init__(self) -> None:
        self._script: dict[str, list[Any]] = {}

    def program(self, path: str, *responses: Any) -> None:
        self._script[path] = list(responses)

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        script = self._script.get(path)
        if not script:
            raise FileNotFoundError(f"no such file in workspace: {path}")
        item = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(item, BaseException):
            raise item
        return item


class _ForeignWriterOnSecondRead:
    """Wraps a LocalS3Client: the SECOND ``get_object`` of ``key`` first lands
    a foreign put — the deterministic "a peer wrote between the capture read
    and the verification read" race for the torn-cut test. Everything else
    delegates untouched."""

    def __init__(self, inner: LocalS3Client, bucket: str, key: str, body: bytes) -> None:
        self._inner = inner
        self._bucket = bucket
        self._key = key
        self._body = body
        self._reads = 0

    def get_object(self, **kwargs: Any) -> Any:
        if kwargs.get("Key") == self._key:
            self._reads += 1
            if self._reads == 2:
                self._inner.put_object(Bucket=self._bucket, Key=self._key, Body=self._body)
        return self._inner.get_object(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _DownService:
    """A persist seam whose coordinator is unreachable — every registration
    raises the transport-shaped error."""

    def create_workspace_checkpoint(self, **_kwargs: Any) -> Any:
        raise ConnectionError("coordinator unreachable")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ArtifactRegistry:
    return ArtifactRegistry()


@pytest.fixture
def service(registry: ArtifactRegistry) -> CoordinatorService:
    return CoordinatorService(registry)


def _versioner(
    service: Any, clock: Any | None = None, resolver: Any | None = None
) -> WorkspaceVersioner:
    return WorkspaceVersioner(
        service=service,
        owner=OWNER,
        clock=clock if clock is not None else _TickClock(),
        file_resolver=resolver,
    )


def _s3(versioned: bool = True) -> tuple[LocalS3Client, CoherentObject]:
    client = LocalS3Client()
    client.create_bucket("demo", versioned=versioned, object_lock=versioned)
    return client, CoherentObject("demo", client=client)


# ---------------------------------------------------------------------------
# Clean capture — tiers, pointers, window
# ---------------------------------------------------------------------------


def test_clean_capture_records_manifest_tiers_and_window(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"cfg-body")
    files = _ScriptedFileSource()
    files.program("notes/plan.md", (b"plan text", 7))

    versioner = _versioner(service, clock=_TickClock(100))
    versioner.add_file_member(files, "notes/plan.md")
    versioner.add_object_member(obj, "cfg.json")
    versioner.add_forward_only_member("effects/slack-notify")

    result = versioner.checkpoint("pre-refactor")

    # Persisted and readable back — the manifest is durable-facts only.
    record = registry.get_checkpoint(result.record.checkpoint_id)
    assert record is not None
    assert record.name == "pre-refactor"
    assert record.owner == OWNER
    # Window = [min, max] of the three capture ticks (100, 101, 102); the
    # persist tick (103) stamps created_at, never the window.
    assert (record.window_min, record.window_max) == (100.0, 102.0)
    assert record.created_at_tick == 103

    members = {m.member_path: m for m in registry.get_checkpoint_members(record.checkpoint_id)}
    assert set(members) == {"notes/plan.md", "s3://cfg.json", "effects/slack-notify"}

    file_row = members["notes/plan.md"]
    assert file_row.native_token == "7"  # the coordinator content-state pointer
    assert file_row.fingerprint == sha256_hex(b"plan text")
    assert file_row.arbitration_tier == "no-arbiter"
    assert file_row.restore_tier == "restorable-unpinned"  # retention pin is Unit 6
    assert file_row.absent is False and file_row.dirty_during_window is False

    s3_row = members["s3://cfg.json"]
    # The manifest pointer is the versionId; the ETag CAS comparand is NEVER
    # manifested (re-read live at restore time — the F4 split).
    assert s3_row.native_token == put["VersionId"]
    assert s3_row.native_token != put["ETag"]
    assert s3_row.fingerprint == sha256_hex(b"cfg-body")
    assert s3_row.arbitration_tier == "native-cas"
    assert s3_row.restore_tier == "restorable"
    assert s3_row.absent is False and s3_row.dirty_during_window is False

    fwd_row = members["effects/slack-notify"]
    assert fwd_row.native_token is None and fwd_row.fingerprint is None
    assert fwd_row.restore_tier == "forward_only"
    assert fwd_row.arbitration_tier == "no-arbiter"


def test_manifest_survives_via_registry_list(service: CoordinatorService, registry) -> None:
    files = _ScriptedFileSource()
    files.program("a.txt", (b"a", 1))
    versioner = _versioner(service)
    versioner.add_file_member(files, "a.txt")
    result = versioner.checkpoint("cp")
    assert [c.checkpoint_id for c in registry.list_checkpoints()] == [
        result.record.checkpoint_id
    ]


# ---------------------------------------------------------------------------
# ABSENT ≠ empty
# ---------------------------------------------------------------------------


def test_absent_member_distinct_from_present_empty(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    _client, obj = _s3()
    obj_client = obj  # readable alias
    # Present-EMPTY S3 member: a real zero-byte body.
    _client.put_object(Bucket="demo", Key="empty.bin", Body=b"")
    # ABSENT file member: never written.
    files = _ScriptedFileSource()  # nothing programmed -> FileNotFoundError

    versioner = _versioner(service)
    versioner.add_file_member(files, "gone.md")
    versioner.add_object_member(obj_client, "empty.bin")
    result = versioner.checkpoint("absent-vs-empty")

    members = {m.member_path: m for m in registry.get_checkpoint_members(result.record.checkpoint_id)}
    absent = members["gone.md"]
    empty = members["s3://empty.bin"]

    # ABSENT is a recorded FACT: no token, no fingerprint, absent=True.
    assert absent.absent is True
    assert absent.native_token is None and absent.fingerprint is None
    # Present-empty is a different fact: captured, fingerprinted as sha256(b"").
    assert empty.absent is False
    assert empty.fingerprint == sha256_hex(b"")
    assert empty.native_token is not None
    # The two records can never be conflated.
    assert (absent.absent, absent.fingerprint) != (empty.absent, empty.fingerprint)


# ---------------------------------------------------------------------------
# Torn-cut detection (dirty_during_window)
# ---------------------------------------------------------------------------


def test_intra_window_write_flags_exactly_the_dirty_member(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    _client, obj = _s3()
    _client.put_object(Bucket="demo", Key="steady.json", Body=b"steady")
    files = _ScriptedFileSource()
    # Capture sees (one, v3); the verification re-read sees (two, v4) — a
    # writer landed inside the window on THIS member only.
    files.program("torn.md", (b"one", 3), (b"two", 4))
    files.program("calm.md", (b"calm", 5))

    versioner = _versioner(service)
    versioner.add_file_member(files, "torn.md")
    versioner.add_file_member(files, "calm.md")
    versioner.add_object_member(obj, "steady.json")
    result = versioner.checkpoint("torn-cut")

    members = {m.member_path: m for m in registry.get_checkpoint_members(result.record.checkpoint_id)}
    assert members["torn.md"].dirty_during_window is True
    # The manifest still records the CAPTURED state, not the raced one.
    assert members["torn.md"].native_token == "3"
    assert members["torn.md"].fingerprint == sha256_hex(b"one")
    # Exactly that member — its peers verified quiescent.
    assert members["calm.md"].dirty_during_window is False
    assert members["s3://steady.json"].dirty_during_window is False


def test_s3_foreign_writer_between_capture_and_verify_flags_dirty(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    inner = LocalS3Client()
    inner.create_bucket("demo", versioned=True, object_lock=True)
    inner.put_object(Bucket="demo", Key="raced.json", Body=b"original")
    racing = _ForeignWriterOnSecondRead(inner, "demo", "raced.json", b"foreign-write")
    obj = CoherentObject("demo", client=racing)

    versioner = _versioner(service)
    versioner.add_object_member(obj, "raced.json")
    result = versioner.checkpoint("s3-race")

    (member,) = registry.get_checkpoint_members(result.record.checkpoint_id)
    assert member.dirty_during_window is True
    assert member.fingerprint == sha256_hex(b"original")  # the captured state


def test_member_vanishing_inside_window_flags_dirty(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    files = _ScriptedFileSource()
    files.program(
        "vanish.md", (b"here", 2), FileNotFoundError("no such file in workspace: vanish.md")
    )
    versioner = _versioner(service)
    versioner.add_file_member(files, "vanish.md")
    result = versioner.checkpoint("vanish")
    (member,) = registry.get_checkpoint_members(result.record.checkpoint_id)
    assert member.absent is False  # captured present…
    assert member.dirty_during_window is True  # …but not verified quiescent


# ---------------------------------------------------------------------------
# Honest refusals / honest tiers
# ---------------------------------------------------------------------------


def test_unversioned_s3_member_honest_refusal_path(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """No pre-probe: the typed VersionPointerUnconfirmed on the capture read IS
    the discovery. The member is still DESCRIBED — fingerprint, presence — but
    holds no pointer and is tiered forward_only, never restorable."""
    client, obj = _s3(versioned=False)
    client.put_object(Bucket="demo", Key="plain.txt", Body=b"unversioned")

    versioner = _versioner(service)
    versioner.add_object_member(obj, "plain.txt")
    result = versioner.checkpoint("honest")

    (member,) = registry.get_checkpoint_members(result.record.checkpoint_id)
    assert member.restore_tier == "forward_only"
    assert member.native_token is None  # the "null" sentinel NEVER lands in a manifest
    assert member.fingerprint == sha256_hex(b"unversioned")
    assert member.absent is False


def test_file_pointer_unconfirmed_version_is_forward_only(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """A file member whose coordinator version cannot be resolved (the volume's
    version-0 fallback) carries an UNCONFIRMED pointer: never manifested,
    never above forward_only (the Sentinel rule, file edition)."""
    files = _ScriptedFileSource()
    files.program("orphan.md", (b"body", 0))
    versioner = _versioner(service)
    versioner.add_file_member(files, "orphan.md")
    result = versioner.checkpoint("orphan")
    (member,) = registry.get_checkpoint_members(result.record.checkpoint_id)
    assert member.native_token is None
    assert member.restore_tier == "forward_only"
    assert member.fingerprint == sha256_hex(b"body")


def test_file_member_refused_at_capture_is_forward_only_not_a_raise(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """A CoherentVolume source refuses a versioned read whose disk bytes are not
    the content at the coordinator's version (a peer's commit still reaching
    disk, or an out-of-band edit). There is then no pointer to manifest, which
    is the unconfirmed-pointer case: the member is described, never above
    forward_only, and the checkpoint still completes."""
    files = _ScriptedFileSource()
    files.program("busy.md", StaleView("refused: bytes are not the content at the version"))
    versioner = _versioner(service)
    versioner.add_file_member(files, "busy.md")
    result = versioner.checkpoint("busy")
    (member,) = registry.get_checkpoint_members(result.record.checkpoint_id)
    assert member.absent is False
    assert member.native_token is None
    assert member.fingerprint is None
    assert member.restore_tier == "forward_only"
    # Neither pass could confirm the member, so it was not verified quiescent.
    assert member.dirty_during_window is True


def test_file_member_refused_at_verify_flags_dirty(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    files = _ScriptedFileSource()
    files.program(
        "doc.md", (b"body", 3), StaleView("refused: bytes are not the content at the version")
    )
    versioner = _versioner(service)
    versioner.add_file_member(files, "doc.md")
    result = versioner.checkpoint("verify-refused")
    (member,) = registry.get_checkpoint_members(result.record.checkpoint_id)
    assert member.native_token == "3"
    assert member.fingerprint == sha256_hex(b"body")
    assert member.dirty_during_window is True


def test_forward_only_members_enumerated_never_token_captured(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    files = _ScriptedFileSource()
    files.program("a.txt", (b"a", 1))
    versioner = _versioner(service)
    versioner.add_file_member(files, "a.txt")
    versioner.add_forward_only_member("effects/send-email")
    versioner.add_forward_only_member("effects/charge-card")
    result = versioner.checkpoint("effects")
    members = {m.member_path: m for m in result.members}
    for path in ("effects/send-email", "effects/charge-card"):
        row = members[path]
        assert row.native_token is None and row.fingerprint is None
        assert row.restore_tier == "forward_only"
        assert row.dirty_during_window is False
    # Enumerated in the DURABLE manifest too.
    stored = registry.get_checkpoint_members(result.record.checkpoint_id)
    assert {m.member_path for m in stored} >= {"effects/send-email", "effects/charge-card"}


def test_binary_file_member_typed_capture_refusal(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    files = _ScriptedFileSource()
    files.program("blob.bin", (b"\xff\xfe\x00\x01", 4))
    versioner = _versioner(service)
    versioner.add_file_member(files, "blob.bin")
    with pytest.raises(BinaryFileMemberRefused) as excinfo:
        versioner.checkpoint("binary")
    # Typed reason matched by IDENTITY (add-never-rename), naming the member.
    assert excinfo.value.reason is BINARY_FILE_MEMBER_REASON
    assert excinfo.value.member_path == "blob.bin"
    # Capture-time refusal: NOTHING persisted.
    assert registry.list_checkpoints() == []


# ---------------------------------------------------------------------------
# Persist failures — typed, no partial manifest
# ---------------------------------------------------------------------------


def test_coordinator_down_typed_failure_no_partial_manifest(
    registry: ArtifactRegistry,
) -> None:
    files = _ScriptedFileSource()
    files.program("a.txt", (b"a", 1))
    versioner = WorkspaceVersioner(service=_DownService(), owner=OWNER)
    versioner.add_file_member(files, "a.txt")
    with pytest.raises(CheckpointPersistFailed) as excinfo:
        versioner.checkpoint("down")
    assert excinfo.value.reason is CHECKPOINT_NOT_PERSISTED_REASON
    assert isinstance(excinfo.value.__cause__, ConnectionError)
    # The registry this test holds was never touched (the down service owns no
    # registry): the guarantee under test is the TYPED failure + the wording's
    # single-transaction claim, which the abort test below pins registry-side.
    assert registry.list_checkpoints() == []


def test_abort_event_threads_into_registry_guard(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """The A6 lesson, proven at the service seam: a pre-set abort Event (the
    watchdog already timed out) fails the registration closed AT the registry
    write lock — no manifest lands after the client saw the degraded
    checkpoint_unconfirmed response."""
    import threading

    from ccs.coordinator.registry_protocol import CheckpointMember

    abort = threading.Event()
    abort.set()
    member = CheckpointMember(
        member_path="a.txt",
        artifact_id=None,
        native_token="1",
        fingerprint=sha256_hex(b"a"),
        captured_at=100.0,
    )
    with pytest.raises(WatchdogAbandoned):
        service.create_workspace_checkpoint(
            name="aborted",
            owner=OWNER,
            members=[member],
            window_min=100.0,
            window_max=100.0,
            issued_at_tick=101,
            abort=abort,
        )
    assert registry.list_checkpoints() == []


def test_service_validation_fails_closed(service: CoordinatorService, registry) -> None:
    from ccs.coordinator.registry_protocol import CheckpointMember

    member = CheckpointMember(
        member_path="a.txt",
        artifact_id=None,
        native_token=None,
        fingerprint=None,
        captured_at=1.0,
    )
    with pytest.raises(ValueError):
        service.create_workspace_checkpoint(
            name="  ", owner=OWNER, members=[member], window_min=1.0, window_max=2.0
        )
    with pytest.raises(ValueError):
        service.create_workspace_checkpoint(
            name="cp", owner=OWNER, members=[], window_min=1.0, window_max=2.0
        )
    with pytest.raises(ValueError):
        service.create_workspace_checkpoint(
            name="cp", owner=OWNER, members=[member], window_min=2.0, window_max=1.0
        )
    assert registry.list_checkpoints() == []


# ---------------------------------------------------------------------------
# Registration-time guards
# ---------------------------------------------------------------------------


def test_duplicate_member_path_rejected_at_registration(
    service: CoordinatorService,
) -> None:
    files = _ScriptedFileSource()
    versioner = _versioner(service)
    versioner.add_file_member(files, "a.txt")
    with pytest.raises(ValueError):
        versioner.add_file_member(files, "a.txt")
    with pytest.raises(ValueError):
        versioner.add_forward_only_member("a.txt")


def test_checkpoint_requires_members_and_name(service: CoordinatorService) -> None:
    versioner = _versioner(service)
    with pytest.raises(ValueError):
        versioner.checkpoint("empty-workspace")
    versioner.add_forward_only_member("fx")
    with pytest.raises(ValueError):
        versioner.checkpoint("   ")


# ---------------------------------------------------------------------------
# Route level — the live-server house pattern (mirrors
# tests/test_claude_code_coordinator_server.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def coordinator(tmp_path: Path):
    from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer

    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="test-instance")
    server.serve_in_thread()
    time.sleep(0.05)
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def client(coordinator):
    import json as _json
    from urllib import error as urlerror
    from urllib import request as urlrequest

    from ccs.adapters.claude_code.auth import load_secret

    secret = load_secret(coordinator.coordinator_root)
    assert secret is not None
    base = f"http://127.0.0.1:{coordinator.port}"
    headers = {
        "Authorization": f"Bearer {secret}",
        "Host": "127.0.0.1",
        "Content-Type": "application/json",
    }

    def request(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = _json.dumps(body).encode("utf-8") if body is not None else None
        req = urlrequest.Request(base + path, data=data, method=method, headers=headers)
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                return resp.status, _json.loads(resp.read().decode("utf-8") or "{}")
        except urlerror.HTTPError as e:
            return e.code, _json.loads(e.read().decode("utf-8") or "{}")

    return request


def _member_wire(path: str, **overrides: Any) -> dict:
    wire = {
        "member_path": path,
        "native_token": "v000001",
        "fingerprint": sha256_hex(b"body"),
        "captured_at": 100.0,
        "absent": False,
        "dirty_during_window": False,
        "arbitration_tier": "native-cas",
        "restore_tier": "restorable",
    }
    wire.update(overrides)
    return wire


def test_route_checkpoint_roundtrip(client) -> None:
    sid = str(uuid.uuid4())
    status, body = client(
        "POST",
        "/workspace/checkpoint",
        {
            "session_id": sid,
            "name": "route-cp",
            "window_min": 100.0,
            "window_max": 102.0,
            "members": [
                _member_wire("s3://cfg.json"),
                _member_wire(
                    "notes/plan.md",
                    native_token="7",
                    arbitration_tier="no-arbiter",
                    restore_tier="restorable-unpinned",
                    dirty_during_window=True,
                ),
                _member_wire(
                    "effects/notify",
                    native_token=None,
                    fingerprint=None,
                    arbitration_tier="no-arbiter",
                    restore_tier="forward_only",
                ),
            ],
        },
    )
    assert status == 200 and body["ok"] is True
    checkpoint_id = body["checkpoint_id"]
    assert body["window_min"] == 100.0 and body["window_max"] == 102.0

    status, listing = client("GET", "/workspace/checkpoints")
    assert status == 200 and listing["ok"] is True
    (cp,) = listing["checkpoints"]
    assert cp["checkpoint_id"] == checkpoint_id
    assert cp["name"] == "route-cp"
    assert cp["restore_status"] == "none"
    members = {m["member_path"]: m for m in cp["members"]}
    assert members["notes/plan.md"]["dirty_during_window"] is True
    assert members["notes/plan.md"]["restore_tier"] == "restorable-unpinned"
    assert members["s3://cfg.json"]["arbitration_tier"] == "native-cas"
    assert members["effects/notify"]["restore_tier"] == "forward_only"
    assert members["effects/notify"]["native_token"] is None


def test_route_checkpoint_boundary_validation(client) -> None:
    sid = str(uuid.uuid4())

    def post(payload: dict) -> tuple[int, dict]:
        return client("POST", "/workspace/checkpoint", payload)

    base = {
        "session_id": sid,
        "name": "cp",
        "window_min": 1.0,
        "window_max": 2.0,
        "members": [_member_wire("a.txt")],
    }
    # Missing / blank name.
    status, _ = post({**base, "name": "   "})
    assert status == 400
    # Empty member list.
    status, _ = post({**base, "members": []})
    assert status == 400
    # Unknown tier vocabulary — closed set, fail-closed.
    status, _ = post({**base, "members": [_member_wire("a.txt", restore_tier="magic")]})
    assert status == 400
    # Malformed fingerprint.
    status, _ = post({**base, "members": [_member_wire("a.txt", fingerprint="beef")]})
    assert status == 400
    # Duplicate member paths.
    status, _ = post({**base, "members": [_member_wire("a.txt"), _member_wire("a.txt")]})
    assert status == 400
    # Inverted window.
    status, _ = post({**base, "window_min": 5.0, "window_max": 1.0})
    assert status == 400
    # Absolute member path refused (the server-side authoritative path gate).
    status, _ = post({**base, "members": [_member_wire("/etc/passwd")]})
    assert status == 400
    # Nothing persisted by any of the rejects.
    status, listing = client("GET", "/workspace/checkpoints")
    assert status == 200 and listing["checkpoints"] == []


# ===========================================================================
# Restore — WV plan Unit 4 (R3): per-member conditional legs under the
# TERMINATION CONTRACT (bounded re-drive, absorbing outcomes, complete
# per-member terminal report, crash-resume from durable state).
# ===========================================================================


class _FakeFileStore:
    """State-based file member store speaking the full FileRestoreTarget
    surface (``read_with_version`` + ``write_cas_at``), with schedulable
    foreign-edit interleaves.

    ``write_cas_at`` mirrors ``CoherentVolume``'s contract: commit iff the
    current version equals ``expected_version`` (else the typed
    ``CasVersionConflict``), new version = expected + 1 — DETECTION-guarded,
    no arbiter. A scheduled foreign edit lands immediately AFTER a read
    returns, so the returned (bytes, version) pair is already stale by CAS
    time — the file half of the delay-injection harness.
    """

    def __init__(self) -> None:
        self._state: dict[str, tuple[bytes, int]] = {}
        self._foreign_after_read: dict[str, list[bytes]] = {}
        self.cas_calls: dict[str, int] = {}
        #: Every ``expected_version`` the engine handed the CAS, in order —
        #: the comparand as the SUBSTRATE saw it, so a test can prove the
        #: reported pointer came from the same read rather than re-deriving it.
        self.cas_expected: dict[str, list[int]] = {}

    def put(self, path: str, data: bytes, version: int) -> None:
        self._state[path] = (bytes(data), version)

    def remove(self, path: str) -> None:
        self._state.pop(path, None)

    def state(self, path: str) -> tuple[bytes, int]:
        return self._state[path]

    def schedule_foreign_edit_after_read(self, path: str, *bodies: bytes) -> None:
        self._foreign_after_read.setdefault(path, []).extend(bodies)

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        if path not in self._state:
            raise FileNotFoundError(f"no such file in workspace: {path}")
        data, version = self._state[path]
        queue = self._foreign_after_read.get(path)
        if queue:
            foreign = queue.pop(0)
            self._state[path] = (bytes(foreign), version + 1)
        return data, version

    def write_cas_at(self, path: str, expected_version: int, new_content: bytes) -> None:
        self.cas_calls[path] = self.cas_calls.get(path, 0) + 1
        self.cas_expected.setdefault(path, []).append(expected_version)
        if path not in self._state:
            raise CasVersionConflict(path, expected_version, 0)
        _data, version = self._state[path]
        if version != expected_version:
            raise CasVersionConflict(path, expected_version, version)
        self._state[path] = (bytes(new_content), version + 1)


class _FakeResolver:
    """FileContentResolver fake: an explicit (path, version) -> bytes history."""

    def __init__(self) -> None:
        self._history: dict[tuple[str, int], bytes] = {}

    def keep(self, path: str, version: int, data: bytes) -> None:
        self._history[(path, int(version))] = bytes(data)

    def content_at(self, member_path: str, version: int) -> bytes:
        try:
            return self._history[(member_path, int(version))]
        except KeyError:
            raise KeyError(f"no retained version {version} for {member_path}") from None


class _ForeignWriterOnLiveReads:
    """The S3 half of the delay-injection harness (Unit-4 execution note).

    Interleaves a foreign put between the engine's live comparand read and its
    CAS put: the (bytes, ETag) the engine holds is already stale by CAS time.
    Pinned (VersionId) reads are exempt — the pinned target is immutable
    history. ``times=None`` = every live read (sustained contention);
    an integer bounds the interleaves (a single race → one retry).
    """

    def __init__(
        self, inner: LocalS3Client, bucket: str, key: str, *, times: int | None = None
    ) -> None:
        self._inner = inner
        self._bucket = bucket
        self._key = key
        self._times = times
        self.landed = 0

    def get_object(self, **kwargs: Any) -> Any:
        resp = self._inner.get_object(**kwargs)
        if kwargs.get("Key") == self._key and kwargs.get("VersionId") is None:
            if self._times is None or self.landed < self._times:
                self.landed += 1
                self._inner.put_object(
                    Bucket=self._bucket,
                    Key=self._key,
                    Body=f"foreign-{self.landed}".encode(),
                )
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _VanishOnCasPut:
    """UNKNOWN-outcome harness: an If-Match put of ``key`` deletes the object
    underneath, then dies transport-shaped (no S3 error code) — the put's
    outcome is UNKNOWN and the reconciliation read finds the object gone
    (HOLD), driving the ``held_unconfirmed`` terminal."""

    def __init__(self, inner: LocalS3Client, bucket: str, key: str) -> None:
        self._inner = inner
        self._bucket = bucket
        self._key = key

    def put_object(self, **kwargs: Any) -> Any:
        if kwargs.get("Key") == self._key and kwargs.get("IfMatch") is not None:
            self._inner.delete_object(Bucket=self._bucket, Key=self._key)
            raise ConnectionError("socket dropped mid-put")
        return self._inner.put_object(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CrashOnNthMemberRecord:
    """Kill-after-N-members crash: forwards to the real service but raises on
    the Nth durable member-progress write — a crash AFTER that member's leg
    landed on the substrate and BEFORE its outcome record.

    The CheckpointRestoreStore members are delegated EXPLICITLY (not via
    ``__getattr__``): runtime-checkable Protocol isinstance uses static
    attribute lookup on 3.12+, which a ``__getattr__`` fallthrough never
    satisfies.
    """

    def __init__(self, inner: Any, fail_on_call: int) -> None:
        self._inner = inner
        self._fail_on = fail_on_call
        self._calls = 0

    def get_workspace_checkpoint(self, checkpoint_id: str) -> Any:
        return self._inner.get_workspace_checkpoint(checkpoint_id)

    def get_workspace_checkpoint_members(self, checkpoint_id: str) -> Any:
        return self._inner.get_workspace_checkpoint_members(checkpoint_id)

    def set_workspace_checkpoint_restore_status(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.set_workspace_checkpoint_restore_status(*args, **kwargs)

    def set_workspace_checkpoint_member_restore(self, *args: Any, **kwargs: Any) -> Any:
        self._calls += 1
        if self._calls == self._fail_on:
            raise RuntimeError("simulated crash (kill -9)")
        return self._inner.set_workspace_checkpoint_member_restore(*args, **kwargs)

    def register_workspace_restore(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.register_workspace_restore(*args, **kwargs)

    def workspace_member_registered(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.workspace_member_registered(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Happy full restore — create + modify + delete legs, all terminal outcomes
# ---------------------------------------------------------------------------


def test_happy_full_restore_all_terminal_outcomes(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"cfg-v1")
    client.put_object(Bucket="demo", Key="gone.json", Body=b"keep-me")
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    files.put("notes/calm.md", b"calm", 5)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    # Unit 6: the default checkpoint() runs the pin legs — the file
    # verification pin consults the resolver, so the converged member's
    # version must be retained too (else it downgrades loudly to
    # forward_only, which the retention-gap pin tests cover).
    resolver.keep("notes/calm.md", 5, b"calm")

    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    versioner.add_file_member(files, "notes/calm.md")
    versioner.add_object_member(obj, "cfg.json")
    versioner.add_object_member(obj, "gone.json")
    versioner.add_object_member(obj, "ghost.json")  # ABSENT at capture
    versioner.add_forward_only_member("effects/notify")
    cp = versioner.checkpoint("full")

    # Post-capture divergence: a modify, a delete, a foreign create, a file edit.
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"cfg-v2")  # modify leg
    client.delete_object(Bucket="demo", Key="gone.json")  # create leg (marker)
    client.put_object(Bucket="demo", Key="ghost.json", Body=b"intruder")  # delete leg
    files.put("notes/plan.md", b"edited", 9)  # file modify leg

    report = versioner.restore(cp.record.checkpoint_id)

    assert report.status == RESTORE_STATUS_CONCLUDED
    by_path = report.members_by_path
    assert by_path["s3://cfg.json"].outcome == RESTORE_OUTCOME_RESTORED
    assert by_path["s3://cfg.json"].new_native_token is not None
    assert by_path["s3://gone.json"].outcome == RESTORE_OUTCOME_RESTORED
    assert by_path["s3://ghost.json"].outcome == RESTORE_OUTCOME_RESTORED
    assert by_path["s3://ghost.json"].deleted_at_restore is not None
    assert by_path["notes/plan.md"].outcome == RESTORE_OUTCOME_RESTORED
    assert by_path["notes/plan.md"].new_native_token == "10"  # 9 (live) + 1
    assert by_path["notes/calm.md"].outcome == RESTORE_OUTCOME_CONVERGED
    assert by_path["effects/notify"].outcome == RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED
    # Four members conclude `restored` and mean four different things to an
    # operator; the observation is the field that separates them in ONE report.
    assert by_path["s3://cfg.json"].observation.state == RESTORE_OBSERVATION_DIFFERS
    assert by_path["s3://gone.json"].observation.state == RESTORE_OBSERVATION_NO_LIVE_STATE
    assert (
        by_path["s3://ghost.json"].observation.state
        == RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE
    )
    assert by_path["notes/plan.md"].observation.state == RESTORE_OBSERVATION_DIFFERS
    assert by_path["notes/calm.md"].observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert (
        by_path["effects/notify"].observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    )

    # The substrates hold the manifest state again.
    data, _etag = obj.read("cfg.json")
    assert data == b"cfg-v1"
    data, _etag = obj.read("gone.json")
    assert data == b"keep-me"
    with pytest.raises(KeyError):
        obj.read("ghost.json")  # deleted (marker-current)
    assert files.state("notes/plan.md")[0] == b"plan text"

    # Durably mirrored: outcomes on the member rows, concluded on the header.
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED
    stored = {m.member_path: m for m in registry.get_checkpoint_members(cp.record.checkpoint_id)}
    for path, outcome in by_path.items():
        assert stored[path].restore_outcome == outcome.outcome


# ---------------------------------------------------------------------------
# Foreign-writer races (the delay-injection harness)
# ---------------------------------------------------------------------------


def _checkpoint_one_s3_member(
    service: CoordinatorService, client: LocalS3Client, key: str, body: bytes
) -> str:
    obj = CoherentObject("demo", client=client)
    versioner = _versioner(service)
    versioner.add_object_member(obj, key)
    return versioner.checkpoint(f"cp-{key}").record.checkpoint_id


def test_single_interleaved_foreign_write_retries_once_and_lands(
    service: CoordinatorService,
) -> None:
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    client.put_object(Bucket="demo", Key="raced.json", Body=b"original")
    checkpoint_id = _checkpoint_one_s3_member(service, client, "raced.json", b"original")
    # Diverge live so the leg must write, then interleave EXACTLY one foreign
    # put between the engine's live read and its CAS.
    client.put_object(Bucket="demo", Key="raced.json", Body=b"diverged")
    racing = _ForeignWriterOnLiveReads(client, "demo", "raced.json", times=1)

    restorer = _versioner(service)
    restorer.add_object_member(CoherentObject("demo", client=racing), "raced.json")
    report = restorer.restore(checkpoint_id)

    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.attempts == 2  # attempt 1 lost the If-Match race; attempt 2 landed
    data, _etag = CoherentObject("demo", client=client).read("raced.json")
    assert data == b"original"


def test_sustained_contention_exhausts_budget_into_conflict_no_livelock(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    client.put_object(Bucket="demo", Key="hot.json", Body=b"original")
    client.put_object(Bucket="demo", Key="calm.json", Body=b"steady")

    obj = CoherentObject("demo", client=client)
    versioner = _versioner(service)
    versioner.add_object_member(obj, "hot.json")
    versioner.add_object_member(obj, "calm.json")
    checkpoint_id = versioner.checkpoint("contended").record.checkpoint_id

    client.put_object(Bucket="demo", Key="hot.json", Body=b"diverged")
    racing = _ForeignWriterOnLiveReads(client, "demo", "hot.json", times=None)
    put_count_before = len(client.put_calls)

    restorer = _versioner(service)
    restorer.add_object_member(CoherentObject("demo", client=racing), "hot.json")
    restorer.add_object_member(CoherentObject("demo", client=racing), "calm.json")
    report = restorer.restore(checkpoint_id)

    by_path = report.members_by_path
    hot = by_path["s3://hot.json"]
    # Budget exhausted, absorbed as conflict — the restore still CONCLUDED and
    # the report names the losing member; the healthy peer converged untouched.
    assert hot.outcome == RESTORE_OUTCOME_CONFLICT
    assert hot.attempts == MAX_RESTORE_LEG_REDRIVES + 1
    # Every iteration READ a live state that differed from the capture, so the
    # tempting observation is the differs one — but the observation names what
    # the leg OVERWROTE and every If-Match attempt lost its race. Reporting a
    # discarded version here would assert a loss that provably did not happen.
    assert hot.observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert hot.observation.pointer is None
    assert by_path["s3://calm.json"].outcome == RESTORE_OUTCOME_CONVERGED
    assert report.status == RESTORE_STATUS_CONCLUDED
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED
    # NO livelock: the engine issued exactly budget-many If-Match puts (the
    # foreign writer's own unconditional puts ride the same log).
    engine_cas_puts = [
        c
        for c in client.put_calls[put_count_before:]
        if c["Key"] == "hot.json" and c["IfMatch"] is not None
    ]
    assert len(engine_cas_puts) == MAX_RESTORE_LEG_REDRIVES + 1


def test_file_member_foreign_edit_detection_labeled_no_arbiter(
    service: CoordinatorService,
) -> None:
    files = _FakeFileStore()
    files.put("doc.md", b"captured", 3)
    resolver = _FakeResolver()
    resolver.keep("doc.md", 3, b"captured")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "doc.md")
    checkpoint_id = versioner.checkpoint("file-race").record.checkpoint_id

    files.put("doc.md", b"edited", 5)
    # One foreign edit between the engine's read and its CAS: detection fires
    # (typed CasVersionConflict), ONE retry lands.
    files.schedule_foreign_edit_after_read("doc.md", b"foreign-edit")

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "doc.md")
    report = restorer.restore(checkpoint_id)

    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.attempts == 2
    # The no-arbiter honesty label: a file outcome is DETECTION, never
    # presented as substrate arbitration.
    assert "no-arbiter" in member.detail
    assert "detection" in member.detail.lower()
    assert "never substrate arbitration" in member.detail
    assert files.state("doc.md")[0] == b"captured"


def test_file_member_sustained_foreign_edits_conflict_no_arbiter(
    service: CoordinatorService,
) -> None:
    files = _FakeFileStore()
    files.put("doc.md", b"captured", 3)
    resolver = _FakeResolver()
    resolver.keep("doc.md", 3, b"captured")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "doc.md")
    checkpoint_id = versioner.checkpoint("file-storm").record.checkpoint_id

    files.put("doc.md", b"edited", 5)
    # More foreign edits than the budget: every CAS detects a moved version.
    files.schedule_foreign_edit_after_read(
        "doc.md", *[f"foreign-{i}".encode() for i in range(2 * MAX_RESTORE_LEG_REDRIVES)]
    )

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "doc.md")
    report = restorer.restore(checkpoint_id)

    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_CONFLICT
    assert member.attempts == MAX_RESTORE_LEG_REDRIVES + 1
    assert "no-arbiter" in member.detail
    assert files.cas_calls["doc.md"] == MAX_RESTORE_LEG_REDRIVES + 1  # bounded, no livelock


class _RefusingFileStore(_FakeFileStore):
    """A file store whose live reads a CoherentVolume source would refuse.

    ``refuse(times)`` makes the next ``times`` reads raise ``StaleView`` (every
    read when ``times`` is ``None``): the disk bytes are not the content at the
    coordinator's version, as during a peer's commit->disk window (transient)
    or after an out-of-band edit (lasting)."""

    def __init__(self) -> None:
        super().__init__()
        self._refusals: int | None = 0
        self._refuse_until = 0.0

    def refuse(self, times: int | None) -> None:
        self._refusals = times

    def refuse_for(self, seconds: float) -> None:
        """Refuse every read until ``seconds`` of wall-clock time have passed,
        as a peer's commit does until its disk write lands."""
        self._refuse_until = time.monotonic() + seconds

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        if time.monotonic() < self._refuse_until:
            raise StaleView("refused: bytes are not the content at the version")
        if self._refusals is None or self._refusals > 0:
            if self._refusals is not None:
                self._refusals -= 1
            raise StaleView("refused: bytes are not the content at the version")
        return super().read_with_version(path)


def _checkpoint_refusing_file(
    service: CoordinatorService, *, absent: bool = False
) -> tuple[_RefusingFileStore, _FakeResolver, str]:
    files = _RefusingFileStore()
    resolver = _FakeResolver()
    if not absent:
        files.put("doc.md", b"captured", 3)
        resolver.keep("doc.md", 3, b"captured")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "doc.md")
    return files, resolver, versioner.checkpoint("refusing").record.checkpoint_id


def _restore_refusing_file(
    service: CoordinatorService, files: _RefusingFileStore, resolver: _FakeResolver, checkpoint_id: str
) -> Any:
    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "doc.md")
    return restorer.restore(checkpoint_id)


def test_file_leg_lasting_refusal_concludes_conflict_without_a_write(
    service: CoordinatorService,
) -> None:
    """A restore leg whose live read keeps being refused cannot establish a
    comparand to CAS from. It must still conclude, as conflict with no write,
    rather than raise and leave the restore in_progress for every resume."""
    files, resolver, checkpoint_id = _checkpoint_refusing_file(service)
    files.put("doc.md", b"edited", 5)
    files.refuse(None)
    report = _restore_refusing_file(service, files, resolver, checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_CONFLICT
    assert member.observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert report.status == RESTORE_STATUS_CONCLUDED
    assert "doc.md" not in files.cas_calls
    assert files.state("doc.md")[0] == b"edited"


def test_file_leg_transient_refusal_redrives_and_lands(service: CoordinatorService) -> None:
    files, resolver, checkpoint_id = _checkpoint_refusing_file(service)
    files.put("doc.md", b"edited", 5)
    files.refuse(1)
    report = _restore_refusing_file(service, files, resolver, checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.attempts == 2
    assert files.state("doc.md")[0] == b"captured"


def test_file_leg_waits_out_a_refusal_that_clears_with_time(
    service: CoordinatorService,
) -> None:
    """A peer's commit reaches disk after some wall-clock time, not after some
    number of reads. Re-driving without waiting spends the whole budget on
    back-to-back reads inside the window and ends in a conflict the same
    restore would have landed a moment later."""
    files, resolver, checkpoint_id = _checkpoint_refusing_file(service)
    files.put("doc.md", b"edited", 5)
    files.refuse_for(0.04)
    report = _restore_refusing_file(service, files, resolver, checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert files.state("doc.md")[0] == b"captured"


def test_absent_leg_refused_read_means_the_file_exists(service: CoordinatorService) -> None:
    """A refusal is about a file that exists (its bytes cannot be paired with a
    version), so a member the manifest records ABSENT is a live divergence."""
    files, resolver, checkpoint_id = _checkpoint_refusing_file(service, absent=True)
    files.put("doc.md", b"appeared later", 2)
    files.refuse(None)
    report = _restore_refusing_file(service, files, resolver, checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_CONFLICT
    assert report.status == RESTORE_STATUS_CONCLUDED
    assert files.state("doc.md")[0] == b"appeared later"


def test_coherent_volume_member_edited_out_of_band_captures_and_restores_to_a_terminal(
    tmp_path: Path,
) -> None:
    """The same two paths with a real CoherentVolume as the file member: after
    an out-of-band edit, a checkpoint records the member as forward_only, and a
    restore of an earlier checkpoint concludes instead of raising StaleView."""
    target = tmp_path / "data" / "f.txt"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"H0")
    cfg = LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=cfg)
    try:
        registry = ArtifactRegistry()
        service = CoordinatorService(registry)
        resolver = _FakeResolver()
        versioner = _versioner(service, resolver=resolver)
        versioner.add_file_member(vol, "data/f.txt")
        before = versioner.checkpoint("before", pin=False).record.checkpoint_id
        (captured,) = registry.get_checkpoint_members(before)
        assert captured.native_token is not None
        resolver.keep("data/f.txt", int(captured.native_token), b"H0")

        target.write_bytes(b"HUMAN")
        after = versioner.checkpoint("after", pin=False).record.checkpoint_id
        (edited,) = registry.get_checkpoint_members(after)
        assert edited.restore_tier == "forward_only"
        assert edited.native_token is None

        report = versioner.restore(before)
        (member,) = report.members
        assert member.outcome == RESTORE_OUTCOME_CONFLICT
        assert report.status == RESTORE_STATUS_CONCLUDED
        assert target.read_bytes() == b"HUMAN"
    finally:
        stop_coordinator(tmp_path)


# ---------------------------------------------------------------------------
# Crash-resume (durable progress; no double-apply)
# ---------------------------------------------------------------------------


def test_crash_mid_restore_fresh_engine_resumes_without_double_apply(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    for key in ("a.json", "b.json", "c.json"):
        client.put_object(Bucket="demo", Key=key, Body=f"{key}-v1".encode())

    obj = CoherentObject("demo", client=client)
    versioner = _versioner(service)
    for key in ("a.json", "b.json", "c.json"):
        versioner.add_object_member(obj, key)
    checkpoint_id = versioner.checkpoint("crashy").record.checkpoint_id

    for key in ("a.json", "b.json", "c.json"):
        client.put_object(Bucket="demo", Key=key, Body=f"{key}-diverged".encode())

    # Crash on the SECOND member's durable outcome write: member a concluded
    # durably; member b's substrate write LANDED but its record did not.
    crashing = _CrashOnNthMemberRecord(service, fail_on_call=2)
    crashed = _versioner(crashing)
    for key in ("a.json", "b.json", "c.json"):
        crashed.add_object_member(obj, key)
    with pytest.raises(RuntimeError, match="simulated crash"):
        crashed.restore(checkpoint_id)

    # Mid-crash durable state: in_progress, member a terminal, b/c pending.
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_IN_PROGRESS
    stored = {m.member_path: m for m in registry.get_checkpoint_members(checkpoint_id)}
    assert stored["s3://a.json"].restore_outcome == RESTORE_OUTCOME_RESTORED
    assert stored["s3://b.json"].restore_outcome is None
    assert stored["s3://c.json"].restore_outcome is None

    puts_before_resume = [c["Key"] for c in client.put_calls]

    # A FRESH engine resumes from durable state alone.
    resumed = _versioner(service)
    for key in ("a.json", "b.json", "c.json"):
        resumed.add_object_member(obj, key)
    report = resumed.restore(checkpoint_id)

    by_path = report.members_by_path
    assert by_path["s3://a.json"].resumed_from_prior_run is True
    assert by_path["s3://a.json"].outcome == RESTORE_OUTCOME_RESTORED
    # Member b's pre-crash write already landed: token identity concludes it
    # CONVERGED with NO second write (no double-apply).
    assert by_path["s3://b.json"].outcome == RESTORE_OUTCOME_CONVERGED
    assert by_path["s3://c.json"].outcome == RESTORE_OUTCOME_RESTORED
    puts_after_resume = [c["Key"] for c in client.put_calls[len(puts_before_resume):]]
    assert "a.json" not in puts_after_resume  # skipped, not re-driven
    assert "b.json" not in puts_after_resume  # converged, not re-applied
    assert puts_after_resume.count("c.json") == 1
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED
    # Every substrate holds the manifest bytes exactly once.
    for key in ("a.json", "b.json", "c.json"):
        data, _etag = obj.read(key)
        assert data == f"{key}-v1".encode()


def test_concluded_checkpoint_restore_is_idempotent_report_only(
    service: CoordinatorService,
) -> None:
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    checkpoint_id = versioner.checkpoint("twice").record.checkpoint_id

    first = versioner.restore(checkpoint_id)
    puts_after_first = len(client.put_calls)
    second = versioner.restore(checkpoint_id)

    assert first.status == second.status == RESTORE_STATUS_CONCLUDED
    assert [m.outcome for m in second.members] == [m.outcome for m in first.members]
    assert all(m.resumed_from_prior_run for m in second.members)
    assert len(client.put_calls) == puts_after_first  # nothing re-driven


# ---------------------------------------------------------------------------
# Absorbing outcomes: expired pin, forward-only, degenerate shapes
# ---------------------------------------------------------------------------


def test_expired_pin_yields_target_lost_and_restore_concludes(
    service: CoordinatorService,
) -> None:
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    # pin=False: the deliberate capture-only shape — no legal hold lands, so
    # the manifested version has nothing protecting it (the Unit-6 pin-path
    # twin of this test proves the held version SURVIVES the same expiry).
    checkpoint_id = versioner.checkpoint("pinned", pin=False).record.checkpoint_id

    # A live writer makes the captured version noncurrent; a lifecycle rule
    # then expires it (no legal hold established): the pin target is gone.
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert expired and not held

    report = versioner.restore(checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_TARGET_LOST
    assert report.status == RESTORE_STATUS_CONCLUDED
    data, _etag = obj.read("cfg.json")
    assert data == b"v2"  # the live state was never touched


def test_forward_only_and_unversioned_members_skipped_and_enumerated(
    service: CoordinatorService,
) -> None:
    client, obj = _s3(versioned=False)
    client.put_object(Bucket="demo", Key="plain.txt", Body=b"unversioned")
    files = _FakeFileStore()
    files.put("a.txt", b"a", 1)
    resolver = _FakeResolver()
    resolver.keep("a.txt", 1, b"a")

    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "a.txt")
    versioner.add_object_member(obj, "plain.txt")  # tiered forward_only (no pointer)
    versioner.add_forward_only_member("effects/send-email")
    checkpoint_id = versioner.checkpoint("skips").record.checkpoint_id

    report = versioner.restore(checkpoint_id)
    by_path = report.members_by_path
    # BOTH forward-only shapes are enumerated in the report, never silent:
    # the declared action surface AND the unversioned-S3 capture refusal.
    assert by_path["effects/send-email"].outcome == RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED
    assert by_path["s3://plain.txt"].outcome == RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED
    assert by_path["a.txt"].outcome == RESTORE_OUTCOME_CONVERGED
    assert report.status == RESTORE_STATUS_CONCLUDED


def test_all_absent_degenerate_shape_concludes_cleanly(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """Every member ABSENT at capture and still absent live: nothing to write,
    nothing to delete — all converged, the restore concludes (no registration
    exists in this unit, so conclusion == status update + report)."""
    _client, obj = _s3()
    files = _FakeFileStore()  # nothing stored -> FileNotFoundError
    versioner = _versioner(service)
    versioner.add_file_member(files, "gone.md")
    versioner.add_object_member(obj, "gone.json")
    checkpoint_id = versioner.checkpoint("all-absent").record.checkpoint_id

    report = versioner.restore(checkpoint_id)
    assert {m.outcome for m in report.members} == {RESTORE_OUTCOME_CONVERGED}
    assert report.status == RESTORE_STATUS_CONCLUDED
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED


def test_absent_manifest_file_present_live_is_labeled_conflict(
    service: CoordinatorService,
) -> None:
    """The v1 residual: a live file the manifest records ABSENT cannot be
    deleted through the file seam — absorbed as conflict, labeled no-arbiter,
    and the restore still concludes."""
    files = _FakeFileStore()  # absent at capture
    versioner = _versioner(service)
    versioner.add_file_member(files, "late.md")
    checkpoint_id = versioner.checkpoint("late-file").record.checkpoint_id

    files.put("late.md", b"appeared later", 2)
    report = versioner.restore(checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_CONFLICT
    assert "no-arbiter" in member.detail
    assert report.status == RESTORE_STATUS_CONCLUDED
    assert files.state("late.md")[0] == b"appeared later"  # never touched


# ---------------------------------------------------------------------------
# HOLD terminals (UNCONFIRMED — never best-effort)
# ---------------------------------------------------------------------------


def test_s3_unknown_write_outcome_reconciles_to_held_unconfirmed(
    service: CoordinatorService,
) -> None:
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    client.put_object(Bucket="demo", Key="flaky.json", Body=b"original")
    checkpoint_id = _checkpoint_one_s3_member(service, client, "flaky.json", b"original")

    client.put_object(Bucket="demo", Key="flaky.json", Body=b"diverged")
    vanishing = _VanishOnCasPut(client, "demo", "flaky.json")
    restorer = _versioner(service)
    restorer.add_object_member(CoherentObject("demo", client=vanishing), "flaky.json")
    report = restorer.restore(checkpoint_id)

    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_HELD_UNCONFIRMED
    assert "HELD" in member.detail and "best-effort" in member.detail
    assert report.status == RESTORE_STATUS_CONCLUDED
    # A put WAS issued here and its outcome is unknowable, so this arm must not
    # borrow the state reserved for legs that never reached a write. Left
    # unpinned, this branch could be rewritten to claim a confirmed overwrite
    # with a fabricated pointer and no test in either suite would notice.
    assert member.observation.state == RESTORE_OBSERVATION_NOT_RECORDED
    assert member.observation.state != RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert member.observation.pointer is None


def test_an_absorbed_failure_does_not_vouch_for_the_bytes_it_may_have_destroyed(
    service: CoordinatorService,
) -> None:
    """A leg that raised mid-write reports no observation, never a quiet one.

    The absorbing boundary catches a deterministic failure raised from ANYWHERE
    inside a leg. The shipped file target truncates the live member before it
    writes, so an OSError from that write leaves the member truncated or half
    rewritten — and the boundary cannot tell that from a failure raised before
    the leg read anything. Reporting ``no_write_attempted`` would tell an
    operator nothing was overwritten over bytes this run destroyed, so the
    state must be the one that claims nothing either way.
    """

    class _RaisesMidWriteStore(_FakeFileStore):
        """Truncate-then-fail: the live content is gone, the write never finished."""

        def write_cas_at(self, path: str, expected_version: int, new_content: bytes) -> None:
            self.put(path, b"", expected_version)  # the truncate half landed
            raise OSError(28, "No space left on device")

    store = _RaisesMidWriteStore()
    member = _drive_one_file_arm(service, store)

    assert member.outcome == RESTORE_OUTCOME_TARGET_LOST
    assert member.observation.state == RESTORE_OBSERVATION_NOT_RECORDED
    assert member.observation.state != RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    # The premise: the peer's content really is gone from the store.
    assert store.read_with_version("doc.md")[0] == b""


def test_file_commit_unconfirmed_is_held(service: CoordinatorService) -> None:
    class _UnconfirmedStore(_FakeFileStore):
        def write_cas_at(self, path: str, expected_version: int, new_content: bytes) -> None:
            raise CommitUnconfirmed("transport failed mid-commit")

    files = _UnconfirmedStore()
    files.put("doc.md", b"captured", 3)
    resolver = _FakeResolver()
    resolver.keep("doc.md", 3, b"captured")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "doc.md")
    checkpoint_id = versioner.checkpoint("unconfirmed").record.checkpoint_id

    files.put("doc.md", b"edited", 5)
    report = versioner.restore(checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_HELD_UNCONFIRMED
    assert report.status == RESTORE_STATUS_CONCLUDED


class _EnvFailureFileStore(_FakeFileStore):
    """``read_with_version`` raises the armed exception — the member path
    replaced by a directory (deterministic pathology) or a transport blip
    (transient), armed AFTER capture so only the restore leg sees it."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_reads_with: BaseException | None = None

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        if self.fail_reads_with is not None:
            raise self.fail_reads_with
        return super().read_with_version(path)


def test_member_path_replaced_by_directory_absorbs_and_restore_concludes(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """FIX-2 regression (termination-contract hole): a deterministic
    member-environment pathology (the member path now a directory —
    IsADirectoryError, OSError family) escaping a restore leg is ABSORBED
    into the absorbing target_lost with the member recorded durably; the
    restore CONCLUDES instead of wedging in_progress with no report."""
    files = _EnvFailureFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    checkpoint_id = versioner.checkpoint("dirified").record.checkpoint_id
    files.fail_reads_with = IsADirectoryError(21, "Is a directory", "notes/plan.md")

    report = versioner.restore(checkpoint_id)

    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_TARGET_LOST
    assert "IsADirectoryError" in member.detail
    assert report.status == RESTORE_STATUS_CONCLUDED
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED
    (stored,) = registry.get_checkpoint_members(checkpoint_id)
    assert stored.restore_outcome == RESTORE_OUTCOME_TARGET_LOST  # durably terminal


def test_transient_transport_failure_still_raises_then_resumes(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """FIX-2 boundary: transient transport shapes (ConnectionError — an
    OSError subclass) are NOT absorbed — they propagate (the crash-resume
    path, BY DESIGN), leave restore_status=in_progress durably, and a later
    restore() resumes and concludes once the transient clears."""
    files = _EnvFailureFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    checkpoint_id = versioner.checkpoint("transient").record.checkpoint_id
    files.fail_reads_with = ConnectionError("volume transport down")

    with pytest.raises(ConnectionError):
        versioner.restore(checkpoint_id)
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_IN_PROGRESS
    (stored,) = registry.get_checkpoint_members(checkpoint_id)
    assert stored.restore_outcome is None  # still pending: nothing terminalized

    files.fail_reads_with = None  # the transient clears
    report = versioner.restore(checkpoint_id)
    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_CONVERGED  # live still matches
    assert report.status == RESTORE_STATUS_CONCLUDED


class _PathRefused(StructuralMemberRefused, ValueError):
    """The interface layer's refusal shape, modelled at the engine seam.

    Mirrors the CLI bridge's ``MemberPathRefused``: a structural refusal that is
    ALSO a ``ValueError``. Both bases are load-bearing here — the engine must
    absorb it through its OWN :class:`StructuralMemberRefused` base (it may
    never import the CLI module), and the ``ValueError`` half is exactly what
    made the old ``(OSError, UnicodeDecodeError)`` boundary miss it.
    """

    def __init__(self, member_path: str) -> None:
        super().__init__(
            f"member path {member_path!r} refused: hardlinked co-owner "
            "(external inode) — cannot be safely contained in v1"
        )
        self.member_path = member_path


class _StructuralRefusalFileStore(_FakeFileStore):
    """``read_with_version`` structurally REFUSES the armed member paths.

    Armed AFTER capture, so the manifest holds an ordinary drivable-looking row
    and only the restore leg meets the refusal — the real shape (a member
    hardlinked/re-pointed between checkpoint and restore).
    """

    def __init__(self) -> None:
        super().__init__()
        self.refuse: dict[str, BaseException] = {}

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        refusal = self.refuse.get(path)
        if refusal is not None:
            raise refusal
        return super().read_with_version(path)


def _two_member_versioner(
    service: CoordinatorService, files: _StructuralRefusalFileStore
) -> WorkspaceVersioner:
    files.put("notes/plan.md", b"plan text", 7)
    files.put("notes/refused.md", b"refused body", 3)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    resolver.keep("notes/refused.md", 3, b"refused body")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    versioner.add_file_member(files, "notes/refused.md")
    return versioner


def test_structural_member_refusal_absorbs_and_restore_concludes(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """A source's own structural refusal must NOT wedge the restore.

    Regression for the termination-contract hole: ``StructuralMemberRefused``
    is a ``ValueError``, so the old ``(OSError, UnicodeDecodeError)`` boundary
    let it propagate out of ``restore()`` — the sibling member landed durably
    but no report was ever returned, ``restore_status`` stayed ``in_progress``
    FOREVER, and every later restore of the id re-raised the identical refusal
    without terminalizing the poisoned row. It is now absorbed into the same
    ``target_lost`` the OSError family maps to (unreachable through its
    declared surface; a re-drive can never succeed).
    """
    files = _StructuralRefusalFileStore()
    versioner = _two_member_versioner(service, files)
    checkpoint_id = versioner.checkpoint("two-member").record.checkpoint_id
    files.put("notes/plan.md", b"drifted", 7)  # the sibling has real work to do
    files.refuse["notes/refused.md"] = _PathRefused("notes/refused.md")

    report = versioner.restore(checkpoint_id)

    assert report.status == RESTORE_STATUS_CONCLUDED
    by_path = report.members_by_path
    refused = by_path["notes/refused.md"]
    assert refused.outcome == RESTORE_OUTCOME_TARGET_LOST
    assert "structural member refusal absorbed" in refused.detail
    assert "_PathRefused" in refused.detail  # the refusal is NAMED, not swallowed
    assert "hardlinked co-owner" in refused.detail
    # The sibling still restores — one undrivable member must never poison the
    # whole checkpoint.
    assert by_path["notes/plan.md"].outcome == RESTORE_OUTCOME_RESTORED
    assert files.state("notes/plan.md")[0] == b"plan text"

    # Durably terminal on BOTH rows, and the header concluded.
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED
    stored = {m.member_path: m for m in registry.get_checkpoint_members(checkpoint_id)}
    assert stored["notes/refused.md"].restore_outcome == RESTORE_OUTCOME_TARGET_LOST
    assert stored["notes/plan.md"].restore_outcome == RESTORE_OUTCOME_RESTORED

    # A SECOND restore (refusal still armed) returns the durable report rather
    # than wedging: nothing is re-driven, so the refusal is never re-raised.
    again = versioner.restore(checkpoint_id)
    assert again.status == RESTORE_STATUS_CONCLUDED
    assert {m.member_path: m.outcome for m in again.members} == {
        "notes/plan.md": RESTORE_OUTCOME_RESTORED,
        "notes/refused.md": RESTORE_OUTCOME_TARGET_LOST,
    }
    assert all(m.resumed_from_prior_run for m in again.members)


def test_structural_refusal_at_capture_still_raises_nothing_persisted(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """The asymmetry is deliberate: only the RESTORE leg absorbs.

    A member the source refuses can never be driven, so it must never land in a
    manifest — capture propagates the typed refusal and persists nothing.
    """
    files = _StructuralRefusalFileStore()
    versioner = _two_member_versioner(service, files)
    files.refuse["notes/refused.md"] = _PathRefused("notes/refused.md")

    with pytest.raises(StructuralMemberRefused) as excinfo:
        versioner.checkpoint("refused-at-capture")

    assert excinfo.value.member_path == "notes/refused.md"
    assert excinfo.value.reason is STRUCTURAL_MEMBER_REFUSAL_REASON
    assert registry.list_checkpoints() == []  # no partial manifest


# ---------------------------------------------------------------------------
# Pre-flight refusals (typed; nothing started)
# ---------------------------------------------------------------------------


def test_unknown_checkpoint_typed_preflight_refusal(service: CoordinatorService) -> None:
    versioner = _versioner(service)
    with pytest.raises(CheckpointUnknown) as excinfo:
        versioner.restore("no-such-checkpoint")
    assert excinfo.value.reason is CHECKPOINT_UNKNOWN_REASON
    assert excinfo.value.checkpoint_id == "no-such-checkpoint"


def test_capture_only_service_refused_before_any_state(
    service: CoordinatorService,
) -> None:
    files = _FakeFileStore()
    files.put("a.txt", b"a", 1)
    versioner = _versioner(service)
    versioner.add_file_member(files, "a.txt")
    checkpoint_id = versioner.checkpoint("cap-only").record.checkpoint_id

    # A capture-only seam (create_workspace_checkpoint only) cannot record a
    # crash-resumable restore: refused with TypeError, before any progress.
    crippled = WorkspaceVersioner(service=_DownService(), owner=OWNER)
    crippled.add_file_member(files, "a.txt")
    with pytest.raises(TypeError, match="CheckpointRestoreStore"):
        crippled.restore(checkpoint_id)


def test_preflight_missing_binding_and_resolver_fail_before_status(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    files = _FakeFileStore()
    files.put("doc.md", b"body", 2)
    resolver = _FakeResolver()
    resolver.keep("doc.md", 2, b"body")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "doc.md")
    checkpoint_id = versioner.checkpoint("preflight").record.checkpoint_id

    # Fresh engine with NO member bindings declared.
    bare = _versioner(service, resolver=resolver)
    with pytest.raises(ValueError, match="no declared file/object member binding"):
        bare.restore(checkpoint_id)

    # Bindings declared but no resolver for an actionable file member.
    no_resolver = _versioner(service)
    no_resolver.add_file_member(files, "doc.md")
    with pytest.raises(ValueError, match="FileContentResolver"):
        no_resolver.restore(checkpoint_id)

    # Nothing was started: the checkpoint never left status "none".
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_NONE


# ---------------------------------------------------------------------------
# Service-level restore vocabulary (closed sets, fail-closed)
# ---------------------------------------------------------------------------


def test_service_restore_vocabulary_fails_closed(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    files = _FakeFileStore()
    files.put("a.txt", b"a", 1)
    versioner = _versioner(service)
    versioner.add_file_member(files, "a.txt")
    checkpoint_id = versioner.checkpoint("vocab").record.checkpoint_id

    with pytest.raises(ValueError, match="unknown restore status"):
        service.set_workspace_checkpoint_restore_status(
            checkpoint_id, "magic", updated_at=1.0
        )
    with pytest.raises(ValueError, match="unknown restore outcome"):
        service.set_workspace_checkpoint_member_restore(
            checkpoint_id, "a.txt", restore_outcome="magic"
        )
    # Nothing landed from either reject.
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_NONE
    (member,) = registry.get_checkpoint_members(checkpoint_id)
    assert member.restore_outcome is None


# ---------------------------------------------------------------------------
# Restore OBSERVATION vocabulary and type (restore-divergence-signal U1 /
# R1-R4): what a writing leg SAW, carried beside the unchanged outcome
# ---------------------------------------------------------------------------


def test_default_constructed_outcome_observes_no_write_attempted() -> None:
    """A member that never reached a write decision is not reported as lost.

    Prevents the conflation the five-state split exists to avoid. The converged,
    skipped and pre-write absorbing sites all construct the outcome without
    naming an observation; if the additive field defaulted to ``not_recorded``,
    every one of them would claim its observation went missing and an operator
    gate that fires on ``not_recorded`` would fire on a workspace nothing
    touched. ``no_write_attempted`` is the TRUTH at those sites: nothing was
    written, so nothing was overwritten.
    """
    outcome = MemberRestoreOutcome(
        member_path="notes/calm.md",
        outcome=RESTORE_OUTCOME_CONVERGED,
        attempts=0,
        detail="live state already matched the manifest",
    )
    assert outcome.observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert outcome.observation.state != RESTORE_OBSERVATION_NOT_RECORDED
    # Nothing was observed, so there is nothing to point at — never a sentinel.
    assert outcome.observation.pointer is None
    assert outcome.observation.fingerprint is None


@pytest.mark.parametrize(
    ("durable_outcome", "expected_state"),
    [
        # The row itself proves no write landed, by any run.
        (RESTORE_OUTCOME_CONVERGED, RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED),
        (RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED, RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED),
        (RESTORE_OUTCOME_CONFLICT, RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED),
        # A write landed, or may have, and what it overwrote is unrecoverable.
        (RESTORE_OUTCOME_RESTORED, RESTORE_OBSERVATION_NOT_RECORDED),
        (RESTORE_OUTCOME_HELD_UNCONFIRMED, RESTORE_OBSERVATION_NOT_RECORDED),
        # target_lost is absorbed from ANYWHERE in a leg, including after the
        # live file was truncated and partly rewritten, so it proves nothing.
        (RESTORE_OUTCOME_TARGET_LOST, RESTORE_OBSERVATION_NOT_RECORDED),
    ],
)
def test_a_rebuilt_row_reports_what_its_durable_outcome_proves(
    durable_outcome: str, expected_state: str
) -> None:
    """The rebuild reads the row's outcome instead of assuming it knows nothing.

    No leg ran in this run, so the tempting blanket answer is ``not_recorded``.
    But the durable outcome is not silent: a converged, skipped or conflicted
    member provably never wrote, by any run, so the quiet answer is established
    rather than assumed. Reporting ``not_recorded`` for those makes a re-restore
    of a workspace nothing ever touched fail the operator's gate on its second
    identical run — a false alarm is what gets a safety flag switched off.
    Every outcome in the closed vocabulary is covered here, so a new one cannot
    be added without deciding which side it falls on.
    """
    from ccs.coordinator.registry_protocol import CheckpointMember

    row = CheckpointMember(
        member_path="s3://cfg.json",
        artifact_id=None,
        native_token="v1",
        fingerprint=None,
        captured_at=1.0,
        restore_outcome=durable_outcome,
    )

    rebuilt = WorkspaceVersioner._outcome_from_durable_row(row)

    assert rebuilt.resumed_from_prior_run is True
    assert rebuilt.observation.state == expected_state
    assert rebuilt.observation.pointer is None
    assert rebuilt.observation.fingerprint is None


def test_every_member_outcome_is_decided_by_the_rebuild() -> None:
    """The parametrisation above covers the vocabulary, not a hand-picked subset.

    Derived from the closed set rather than a literal list: a seventh outcome
    added to the vocabulary fails here instead of silently inheriting whichever
    branch the rebuild's conditional happens to take.
    """
    from ccs.coordinator.registry_protocol import CheckpointMember

    assert RESTORE_OUTCOMES_PROVING_NO_WRITE < RESTORE_MEMBER_OUTCOMES
    decided = {
        outcome
        for outcome in RESTORE_MEMBER_OUTCOMES
        if WorkspaceVersioner._outcome_from_durable_row(
            CheckpointMember(
                member_path="m",
                artifact_id=None,
                native_token=None,
                fingerprint=None,
                captured_at=1.0,
                restore_outcome=outcome,
            )
        ).observation.state
        in (RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED, RESTORE_OBSERVATION_NOT_RECORDED)
    }
    assert decided == RESTORE_MEMBER_OUTCOMES


def test_re_restore_of_a_converged_checkpoint_stays_as_quiet_as_the_first_run(
    service: CoordinatorService,
) -> None:
    """Re-running a restore that overwrote nothing must not start claiming it did.

    End-to-end twin of the rebuild-site unit above, and the case a private
    helper cannot reach: an operator re-runs an already-concluded restore.
    Nothing was ever overwritten, so the second run must read exactly as quiet
    as the first — otherwise the identical command reports differently the
    second time, and under the operator's gate the second run fails.
    """
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    checkpoint_id = versioner.checkpoint("twice-observed").record.checkpoint_id

    first = versioner.restore(checkpoint_id)
    second = versioner.restore(checkpoint_id)

    # The first run converged: the live object already matched the manifest, so
    # no leg wrote. The rebuild reads that from the durable row rather than
    # claiming it knows nothing, so the second run is as quiet as the first.
    assert [m.outcome for m in first.members] == [RESTORE_OUTCOME_CONVERGED]
    assert [m.observation.state for m in second.members] == [
        RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    ]


def test_the_five_observation_states_are_distinct_identities() -> None:
    """Five states, none reachable from another by a substring match.

    Control flow keys off the typed value, never a fragment of prose, so the
    spellings must not nest: were one state's token a substring of another's, a
    consumer that reached for ``in`` would silently classify ``not_recorded`` as
    clean (or the reverse), which is exactly the collapse R3 forbids. Pinned as
    LITERALS — a set derived from the code under test moves its own goalposts.
    """
    states = [
        RESTORE_OBSERVATION_DIFFERS,
        RESTORE_OBSERVATION_NO_LIVE_STATE,
        RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE,
        RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
        RESTORE_OBSERVATION_NOT_RECORDED,
    ]
    assert len(set(states)) == 5  # pairwise distinct, not five aliases
    assert RESTORE_OBSERVATION_STATES == frozenset(
        {
            "observed_differs",
            "no_live_state",
            "present_not_comparable",
            "no_write_attempted",
            "not_recorded",
        }
    )
    assert len(RESTORE_OBSERVATION_STATES) == 5  # add+remove cannot slip past
    for one in states:
        others = [other for other in states if other != one]
        assert not any(one in other for other in others)


def test_restore_member_outcome_vocabulary_is_unchanged() -> None:
    """The observation is additive: no member outcome token was added.

    The outcome set is validated fail-closed in the coordinator and again at the
    HTTP boundary, and is pinned by the cross-implementation kit, so growing it
    is a wire change with three external consequences. Pinned as LITERALS here,
    never derived from the code under test, with cardinality asserted separately
    so a same-size add+remove cannot slip past set equality.
    """
    assert RESTORE_MEMBER_OUTCOMES == frozenset(
        {
            "restored",
            "converged",
            "conflict",
            "held_unconfirmed",
            "target_lost",
            "forward_only_skipped",
        }
    )
    assert len(RESTORE_MEMBER_OUTCOMES) == 6
    # The observation states are their OWN closed set — never merged into the
    # outcome vocabulary, which is what keeps `restored` meaning `restored`.
    assert RESTORE_OBSERVATION_STATES.isdisjoint(RESTORE_MEMBER_OUTCOMES)


def test_restore_observation_refuses_an_out_of_vocabulary_state() -> None:
    """An unknown state fails closed at construction, never reads as clean.

    A typo'd or invented state would be classified by no consumer: the gate
    fires on three named states, so an unrecognised one silently reports clean —
    the one outcome an honesty field must never produce. Refusing it where it is
    built is what keeps every consumer's branch total.
    """
    with pytest.raises(ValueError, match="unknown restore observation state"):
        RestoreObservation(state="magic")


def test_only_the_differs_state_may_carry_a_pointer_or_fingerprint() -> None:
    """A state that observed no comparand can never name one.

    The delete leg's probe verifies no content and the create-on-absent path
    discards nothing, so neither has a pointer or fingerprint to report; a
    rebuilt or never-attempted member has no read at all. Letting one of them
    carry values would over-claim — an operator reading a version number would
    believe the run compared content it never saw.
    """
    for state in (
        RESTORE_OBSERVATION_NO_LIVE_STATE,
        RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE,
        RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
        RESTORE_OBSERVATION_NOT_RECORDED,
    ):
        with pytest.raises(ValueError, match="observed no comparand"):
            RestoreObservation(state=state, pointer="7")
        with pytest.raises(ValueError, match="observed no comparand"):
            RestoreObservation(state=state, fingerprint="deadbeef")
    # The one state that DID read a comparand carries both halves (KTD5: a
    # pointer and a content fingerprint, from the read that fed the CAS).
    observed = RestoreObservation(
        state=RESTORE_OBSERVATION_DIFFERS, pointer="7", fingerprint="deadbeef"
    )
    assert (observed.pointer, observed.fingerprint) == ("7", "deadbeef")


# ---------------------------------------------------------------------------
# The FILE leg's observation (restore-divergence-signal U2 / R1-R2): what the
# version-CAS leg saw of the live state in the iteration that WON
# ---------------------------------------------------------------------------


def _file_checkpoint(
    service: CoordinatorService,
    files: _FakeFileStore,
    resolver: _FakeResolver,
    path: str,
    body: bytes,
    version: int,
    name: str,
) -> str:
    """Capture ONE file member at (body, version), with that version retained.

    The two-step shape every test below needs: a checkpoint taken while the
    member is quiescent, so whatever the restore leg later observes came from a
    write that landed AFTER the capture — never from the capture itself.
    """
    files.put(path, body, version)
    resolver.keep(path, version, body)
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, path)
    return versioner.checkpoint(name).record.checkpoint_id


def test_file_leg_restored_over_a_peer_write_names_the_discarded_version(
    service: CoordinatorService,
) -> None:
    """The issue's reproduction: the report names what the restore threw away.

    Two steps, because one cannot express the defect: a peer commits after the
    capture and STOPS (no contention, no re-drive), then the operator restores.
    The member concludes ``restored`` either way — the outcome vocabulary is
    unchanged by decision — so ``restored`` alone cannot distinguish "put back a
    state nothing had touched" from "discarded a colleague's committed work".
    Before the observation, an operator reading this report had no field that
    differed between the two, and the peer's version number was unrecoverable
    the instant the CAS advanced it.
    """
    files = _FakeFileStore()
    resolver = _FakeResolver()
    checkpoint_id = _file_checkpoint(
        service, files, resolver, "notes/plan.md", b"plan text", 7, "pre-peer"
    )

    # A peer commits and stops: content AND version move, then quiesce.
    files.put("notes/plan.md", b"peer's committed edit", 9)

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "notes/plan.md")
    (member,) = restorer.restore(checkpoint_id).members

    assert member.outcome == RESTORE_OUTCOME_RESTORED  # unchanged vocabulary
    assert member.observation.state == RESTORE_OBSERVATION_DIFFERS
    # Both halves name the state that was OVERWRITTEN, not the one that landed.
    assert member.observation.pointer == "9"
    assert member.observation.fingerprint == sha256_hex(b"peer's committed edit")
    # The minted pointer is a DIFFERENT value: new_native_token is what the
    # write produced (9 + 1), the observation is what it destroyed. Conflating
    # them would hand a Unit-5 registration the wrong artifact version.
    assert member.new_native_token == "10"
    assert member.new_native_token != member.observation.pointer
    assert files.state("notes/plan.md")[0] == b"plan text"


def test_file_leg_converged_member_observes_no_write_attempted(
    service: CoordinatorService,
) -> None:
    """A quiescent member reports the no-write state, never a divergence.

    The converged short-circuit returns before the leg resolves pinned bytes or
    touches the CAS, so there is no comparand to report — and reporting one
    would be a false alarm on the exact workspace an operator gate must stay
    quiet about. Pinned here rather than left to the field default: this is the
    control arm that makes the ``observed_differs`` assertions above mean
    something, because a leg that recorded ``observed_differs`` unconditionally
    would satisfy every other test in this section.
    """
    files = _FakeFileStore()
    resolver = _FakeResolver()
    checkpoint_id = _file_checkpoint(
        service, files, resolver, "notes/calm.md", b"steady", 4, "quiescent"
    )

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "notes/calm.md")
    (member,) = restorer.restore(checkpoint_id).members

    assert member.outcome == RESTORE_OUTCOME_CONVERGED
    assert member.observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert member.observation.pointer is None
    assert member.observation.fingerprint is None
    assert "notes/calm.md" not in files.cas_calls  # no write was even attempted


def test_file_leg_re_drive_observes_the_winning_iteration_not_the_first_read(
    service: CoordinatorService,
) -> None:
    """Under contention the report names the state the write actually replaced.

    KTD7: the live read sits INSIDE the budget loop, so an observation captured
    once and held across re-drives names a state the write did not overwrite —
    the engine would truthfully say ``restored`` while pointing at a version
    some other write had already superseded, which is worse than silence
    because it reads as evidence. One foreign edit is interleaved between the
    first read and its CAS; the first read saw (``edited``, 5) and the second,
    winning read saw (``foreign-edit``, 6). Only the latter may be reported.
    """
    files = _FakeFileStore()
    resolver = _FakeResolver()
    checkpoint_id = _file_checkpoint(
        service, files, resolver, "doc.md", b"captured", 3, "raced"
    )

    files.put("doc.md", b"edited", 5)
    files.schedule_foreign_edit_after_read("doc.md", b"foreign-edit")

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "doc.md")
    (member,) = restorer.restore(checkpoint_id).members

    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.attempts == 2  # attempt 1 lost the CAS, attempt 2 landed
    assert member.observation.state == RESTORE_OBSERVATION_DIFFERS
    # The WINNING iteration's read.
    assert member.observation.pointer == "6"
    assert member.observation.fingerprint == sha256_hex(b"foreign-edit")
    # Explicitly NOT the first read's — the mutation this test exists to kill.
    assert member.observation.pointer != "5"
    assert member.observation.fingerprint != sha256_hex(b"edited")


def test_file_leg_observation_pointer_is_the_version_the_cas_used(
    service: CoordinatorService,
) -> None:
    """The pointer is the CAS comparand itself, not a second read's answer.

    R1 requires the observation to come from the SAME read that produced the
    leg's comparand; a re-read would describe a state the write never compared
    against, and on a contended path the two answers differ. Asserted against
    the value the substrate received, recorded by the fake at the CAS boundary —
    the only vantage point from which "same read" is checkable at all.
    """
    files = _FakeFileStore()
    resolver = _FakeResolver()
    checkpoint_id = _file_checkpoint(
        service, files, resolver, "doc.md", b"captured", 3, "comparand"
    )

    files.put("doc.md", b"edited", 5)
    files.schedule_foreign_edit_after_read("doc.md", b"foreign-edit")

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "doc.md")
    (member,) = restorer.restore(checkpoint_id).members

    winning_comparand = files.cas_expected["doc.md"][-1]
    assert files.cas_expected["doc.md"] == [5, 6]  # both attempts, in order
    assert member.observation.pointer == str(winning_comparand)


def test_file_leg_records_differs_when_only_the_content_moved(
    service: CoordinatorService,
) -> None:
    """An unchanged pointer beside changed content is still a divergence.

    Proves the fingerprint does independent work. A source that rewrites bytes
    without advancing its version (a restored backup, a touch-preserving editor,
    a coarse mtime-derived version) hands the leg a pointer identical to the
    captured one — so a pointer-only observation would read as "nothing moved"
    and an operator gate keyed on it would pass over content it just destroyed.
    The CAS itself cannot catch this either: the comparand matches, so the write
    lands.
    """
    files = _FakeFileStore()
    resolver = _FakeResolver()
    checkpoint_id = _file_checkpoint(
        service, files, resolver, "doc.md", b"captured", 3, "silent-edit"
    )

    # Same version, different bytes — the CAS will pass, the content did not.
    files.put("doc.md", b"silently rewritten", 3)

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "doc.md")
    (member,) = restorer.restore(checkpoint_id).members

    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.observation.state == RESTORE_OBSERVATION_DIFFERS
    assert member.observation.pointer == "3"  # indistinguishable from capture
    assert member.observation.fingerprint == sha256_hex(b"silently rewritten")
    assert member.observation.fingerprint != sha256_hex(b"captured")


def _drive_one_file_arm(
    service: CoordinatorService, store: "_FakeFileStore"
) -> MemberRestoreOutcome:
    """Capture a member, let a peer edit it, then restore through ``store``."""
    resolver = _FakeResolver()
    checkpoint_id = _file_checkpoint(
        service, store, resolver, "doc.md", b"captured", 3, "no-landing"
    )
    store.put("doc.md", b"edited", 5)
    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(store, "doc.md")
    (member,) = restorer.restore(checkpoint_id).members
    return member


class _WedgedStore(_FakeFileStore):
    """The comparand view stayed strict-denied, so no write was ever issued."""

    def write_cas_at(self, path: str, expected_version: int, new_content: bytes) -> None:
        raise ViewWedged("the comparand view stayed strict-denied")


class _UnconfirmedStore(_FakeFileStore):
    """The bytes reached disk; only the version commit was refused.

    Mirrors the ordering the shipped command-line file target actually uses —
    write the member, THEN commit the ledger, and raise only when that commit
    is refused. The engine cannot see which order a target chose, which is the
    whole reason its arm must not claim no write was attempted.
    """

    def write_cas_at(self, path: str, expected_version: int, new_content: bytes) -> None:
        self.put(path, new_content, expected_version + 1)
        raise CommitUnconfirmed(
            f"file member {path!r}: the restored bytes landed on disk but "
            "the version-CAS ledger commit was refused"
        )


def test_file_leg_wedged_view_observes_no_write_attempted(
    service: CoordinatorService,
) -> None:
    """A wedged view issued no write at all, so the quiet answer is the true one.

    This is the control for the arm below: both read a live comparand and both
    conclude without a confirmed write, so a rule that keyed off either fact
    would give them the same answer. Only the wedged arm never reached the
    write, which is what ``no_write_attempted`` means.
    """
    member = _drive_one_file_arm(service, _WedgedStore())

    assert member.outcome == RESTORE_OUTCOME_CONFLICT
    assert member.observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert member.observation.pointer is None
    assert member.observation.fingerprint is None


def test_file_leg_unconfirmed_commit_does_not_claim_it_left_the_bytes_alone(
    service: CoordinatorService,
) -> None:
    """A write whose outcome was lost must not report that none was attempted.

    The shipped file target writes the member and then commits the ledger,
    raising here only when that commit is refused — so on the path this
    project actually ships, the peer's content is already gone when this arm
    runs. Reporting ``no_write_attempted`` reads as "nothing was overwritten"
    over bytes that were, which is the exact failure the observation exists to
    prevent. ``not_recorded`` is the answer that is true whichever order the
    target chose: a write was issued and this run cannot say what it destroyed.
    Asserted against the store above, which writes before it raises.
    """
    store = _UnconfirmedStore()
    member = _drive_one_file_arm(service, store)

    assert member.outcome == RESTORE_OUTCOME_HELD_UNCONFIRMED
    assert member.observation.state == RESTORE_OBSERVATION_NOT_RECORDED
    assert member.observation.state != RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    # It claims no loss either: nothing was confirmed, so nothing is named.
    assert member.observation.pointer is None
    assert member.observation.fingerprint is None
    # The premise this arm rests on: the bytes really did reach the store.
    assert store.read_with_version("doc.md")[0] == b"captured"


# ---------------------------------------------------------------------------
# The OBJECT legs' observation (restore-divergence-signal U3 / R1-R2): the
# native-CAS leg reports the versionId it discarded; the delete leg reports
# only that live state existed, because its probe compares no content
# ---------------------------------------------------------------------------


class _PointerlessOnCasPut:
    """Versioning SUSPENDED between the live read and the CAS put.

    The If-Match put still LANDS in the inner store — the response's ETag is
    real — but carries no ``VersionId``, which is exactly what an unversioned
    bucket returns. ``cas_write_versioned`` then raises
    ``VersionPointerUnconfirmed`` from a write that durably landed, the one arm
    where the leg overwrote live state yet can register nothing for it.
    """

    def __init__(self, inner: LocalS3Client, key: str) -> None:
        self._inner = inner
        self._key = key

    def put_object(self, **kwargs: Any) -> Any:
        resp = self._inner.put_object(**kwargs)
        if kwargs.get("Key") == self._key and kwargs.get("IfMatch") is not None:
            return {k: v for k, v in resp.items() if k != "VersionId"}
        return resp

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ConvergeOnCasPut:
    """UNKNOWN outcome whose reconciliation read finds the intended content.

    The If-Match put lands the SAME bytes unconditionally and then dies
    transport-shaped, so the engine sees ``CasUnknown`` and
    ``reconcile_after_unknown`` observes a MOVED token carrying the intended
    hash — verdict CONVERGE, terminal ``converged``, authorship not claimed.
    Whether this run's put or a peer produced that content is unknowable.
    """

    def __init__(self, inner: LocalS3Client, bucket: str, key: str) -> None:
        self._inner = inner
        self._bucket = bucket
        self._key = key

    def put_object(self, **kwargs: Any) -> Any:
        # Both conditional shapes: the modify leg sends If-Match, the
        # create-on-absent leg sends If-None-Match — matching only the former
        # would let the create arm land normally and never reach reconcile.
        conditional = kwargs.get("IfMatch") is not None or kwargs.get("IfNoneMatch") is not None
        if kwargs.get("Key") == self._key and conditional:
            self._inner.put_object(Bucket=self._bucket, Key=self._key, Body=kwargs["Body"])
            raise ConnectionError("socket dropped mid-put")
        return self._inner.put_object(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _object_checkpoint(
    service: CoordinatorService,
    client: LocalS3Client,
    key: str,
    body: bytes | None,
    name: str,
) -> str:
    """Capture ONE object member at ``body`` — or ABSENT when ``body`` is None.

    The two-step shape every test below needs (U2's file-leg helper, object
    side): the capture happens while the member is quiescent, so whatever the
    restore leg later observes came from a write that landed AFTER it.
    """
    if body is not None:
        client.put_object(Bucket="demo", Key=key, Body=body)
    versioner = _versioner(service)
    versioner.add_object_member(CoherentObject("demo", client=client), key)
    return versioner.checkpoint(name).record.checkpoint_id


def _restore_one_object(
    service: CoordinatorService, client: Any, key: str, checkpoint_id: str
) -> MemberRestoreOutcome:
    restorer = _versioner(service)
    restorer.add_object_member(CoherentObject("demo", client=client), key)
    (member,) = restorer.restore(checkpoint_id).members
    return member


def test_object_leg_restored_over_a_peer_write_names_the_discarded_version_id(
    service: CoordinatorService,
) -> None:
    """The report names the versionId the restore threw away — not the ETag.

    The object leg reads bytes, ETag and versionId from ONE response and today
    keeps only the first two: the ETag arbitrates the write and the versionId is
    dropped on the floor. That drop is what leaves ``restored`` unable to
    distinguish "put back a state nothing had touched" from "discarded a
    colleague's committed object", and the peer's versionId — the only handle
    that still resolves their bytes through S3 versioning — is unrecoverable
    from the report the moment the CAS mints a newer one.

    The ETag is asserted ABSENT from both halves on purpose (KTD5): it is the
    leg's comparand, it is not a pointer any S3 call accepts, and recording it
    where an operator expects a versionId would hand them a string that looks
    actionable and resolves nothing.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "cfg.json", b"captured", "pre-peer")

    # A peer commits and stops: content, ETag and versionId all move, then quiesce.
    peer = client.put_object(Bucket="demo", Key="cfg.json", Body=b"peer's committed object")

    member = _restore_one_object(service, client, "cfg.json", checkpoint_id)

    assert member.outcome == RESTORE_OUTCOME_RESTORED  # unchanged vocabulary
    assert member.observation.state == RESTORE_OBSERVATION_DIFFERS
    assert member.observation.pointer == peer["VersionId"]
    assert member.observation.fingerprint == sha256_hex(b"peer's committed object")
    # The comparand is NOT the observation: an ETag never resolves a version.
    assert member.observation.pointer != peer["ETag"]
    assert member.observation.fingerprint != peer["ETag"]
    # Nor is the pointer the one this write MINTED — that is new_native_token,
    # the state the operator still has; conflating them would name the survivor.
    assert member.new_native_token is not None
    assert member.observation.pointer != member.new_native_token
    # The discarded version is still resolvable BY that pointer (versioning
    # preserved it), which is what makes the recorded string worth reporting.
    resp = client.get_object(Bucket="demo", Key="cfg.json", VersionId=member.observation.pointer)
    assert resp["Body"].read() == b"peer's committed object"
    assert CoherentObject("demo", client=client).read("cfg.json")[0] == b"captured"


def test_object_leg_create_on_absent_records_no_live_state(
    service: CoordinatorService,
) -> None:
    """A create-on-absent leg discarded nothing, and must not read as divergence.

    The leg's live read raised ``KeyError`` and it wrote under the
    ``CREATE_IF_ABSENT`` comparand, so there was no live state to overwrite —
    the one landed arm where ``restored`` really does mean "put back a state
    nothing had touched". Collapsing it into the differs state would fire an
    operator's gate on every deleted-then-restored member, which is the most
    ordinary restore there is.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "gone.json", b"keep-me", "pre-delete")

    client.delete_object(Bucket="demo", Key="gone.json")  # marker-current: live absent

    member = _restore_one_object(service, client, "gone.json", checkpoint_id)

    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.observation.state == RESTORE_OBSERVATION_NO_LIVE_STATE
    assert member.observation.pointer is None
    assert member.observation.fingerprint is None
    assert CoherentObject("demo", client=client).read("gone.json")[0] == b"keep-me"


def test_object_delete_leg_records_presence_without_claiming_content(
    service: CoordinatorService,
) -> None:
    """The delete leg says live state EXISTED, and claims nothing about it.

    Its probe is the plain ``read`` — the versioned read refuses an unversioned
    bucket, and R1's constraint forbids a second call to fetch a pointer — so
    the leg never compares the live content with the capture and holds no
    versionId for it. It is nonetheless the one leg CERTAIN it destroyed live
    state, so silence would be the wrong answer too: the state alone is the
    honest middle, and a later gate can fire on it (a member captured absent
    that is present live was created after the capture) without the report ever
    asserting the deleted bytes were verified.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "ghost.json", None, "absent-fact")

    client.put_object(Bucket="demo", Key="ghost.json", Body=b"created after the capture")

    member = _restore_one_object(service, client, "ghost.json", checkpoint_id)

    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.deleted_at_restore is not None
    assert member.observation.state == RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE
    # Never the differs state: that one asserts a COMPARISON this leg never ran.
    assert member.observation.state != RESTORE_OBSERVATION_DIFFERS
    assert member.observation.pointer is None
    assert member.observation.fingerprint is None


def test_object_delete_leg_over_an_already_absent_member_observes_no_write(
    service: CoordinatorService,
) -> None:
    """Already-absent converges and observed no live state to destroy.

    The control arm for the presence state: without it, a delete leg recording
    ``present_not_comparable`` unconditionally would satisfy the test above
    while firing an operator's gate on a workspace where nothing was deleted at
    all — the probe's ``KeyError`` is precisely the evidence that distinguishes
    them.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "ghost.json", None, "absent-fact")

    member = _restore_one_object(service, client, "ghost.json", checkpoint_id)

    assert member.outcome == RESTORE_OUTCOME_CONVERGED
    assert member.observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert member.deleted_at_restore is None


def test_object_leg_converged_member_observes_no_write_attempted(
    service: CoordinatorService,
) -> None:
    """A quiescent object member reports the no-write state, never a divergence.

    The converged short-circuit returns before the leg resolves pinned bytes or
    touches the CAS, so there is no overwrite to describe. The control arm that
    makes the differs assertions mean something: a leg recording
    ``observed_differs`` unconditionally would pass every other test here.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "calm.json", b"steady", "quiescent")
    puts_before = len(client.put_calls)

    member = _restore_one_object(service, client, "calm.json", checkpoint_id)

    assert member.outcome == RESTORE_OUTCOME_CONVERGED
    assert member.observation.state == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert member.observation.pointer is None
    assert member.observation.fingerprint is None
    assert len(client.put_calls) == puts_before  # no write was even attempted


def test_object_leg_re_drive_observes_the_winning_iteration_not_the_first_read(
    service: CoordinatorService,
) -> None:
    """Under contention the report names the state the write actually replaced.

    KTD7: the live read sits INSIDE the budget loop, so a view captured once
    and held across re-drives names a version the write did not overwrite — the
    engine would truthfully say ``restored`` while pointing at a versionId some
    other write had already superseded, which is worse than silence because it
    reads as evidence. One foreign put is interleaved between the first live
    read and its CAS; only the second, winning read may be reported.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "raced.json", b"captured", "raced")
    diverged = client.put_object(Bucket="demo", Key="raced.json", Body=b"diverged")
    racing = _ForeignWriterOnLiveReads(client, "demo", "raced.json", times=1)

    member = _restore_one_object(service, racing, "raced.json", checkpoint_id)

    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.attempts == 2  # attempt 1 lost the If-Match race, attempt 2 landed
    assert member.observation.state == RESTORE_OBSERVATION_DIFFERS
    assert member.observation.fingerprint == sha256_hex(b"foreign-1")
    # Explicitly NOT the first read's — the hoist this test exists to kill.
    assert member.observation.pointer != diverged["VersionId"]
    assert member.observation.fingerprint != sha256_hex(b"diverged")
    # The winning read's versionId still resolves the bytes it named.
    resp = client.get_object(Bucket="demo", Key="raced.json", VersionId=member.observation.pointer)
    assert resp["Body"].read() == b"foreign-1"


def test_object_leg_pointerless_landing_still_names_what_it_overwrote(
    service: CoordinatorService,
) -> None:
    """A write that landed but minted no pointer still discarded live content.

    Versioning was suspended between the live read and the put, so
    ``cas_write_versioned`` raises after a durable landing whose ETag it
    captured. Nothing can be registered FOR the new state — ``new_native_token``
    stays None — but the state it replaced was read, compared and overwritten,
    and the live view already holds both halves. Staying silent here would drop
    the divergence signal on the one arm that cannot even be re-pinned, leaving
    an operator with the least recoverable member and the least information
    about it.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "cfg.json", b"captured", "pointerless")
    peer = client.put_object(Bucket="demo", Key="cfg.json", Body=b"peer's committed object")

    member = _restore_one_object(
        service, _PointerlessOnCasPut(client, "cfg.json"), "cfg.json", checkpoint_id
    )

    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.new_native_token is None  # nothing to pin or register
    assert member.observation.state == RESTORE_OBSERVATION_DIFFERS
    assert member.observation.pointer == peer["VersionId"]
    assert member.observation.fingerprint == sha256_hex(b"peer's committed object")
    assert CoherentObject("demo", client=client).read("cfg.json")[0] == b"captured"


def test_object_leg_unknown_write_reconciled_converge_records_not_recorded(
    service: CoordinatorService,
) -> None:
    """Authorship unknowable ⇒ the observation says "not recorded", never clean.

    A put was issued, its outcome was lost with the socket, and the
    reconciliation read found the live object byte-identical to the manifest.
    Either this run's put landed and discarded the peer's object, or it never
    landed and something else converged it — the terminal says ``converged``
    and its detail says authorship is not claimed, and that uncertainty is
    exactly what the observation must carry too.

    ``no_write_attempted`` is the wrong answer here for the same reason the file
    leg's unconfirmed arm could safely take it and this one cannot: there, the
    terminal was ``held_unconfirmed``, already the loudest in the vocabulary.
    Here the terminal is the QUIETEST one, so an under-claiming observation
    makes a leg that read divergent content and issued a write indistinguishable
    from a workspace nobody touched. ``observed_differs`` would over-claim in
    the other direction — it asserts the write discarded that content, which is
    the very thing no read can establish.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "flaky.json", b"captured", "unknown")
    client.put_object(Bucket="demo", Key="flaky.json", Body=b"peer's committed object")

    member = _restore_one_object(
        service, _ConvergeOnCasPut(client, "demo", "flaky.json"), "flaky.json", checkpoint_id
    )

    assert member.outcome == RESTORE_OUTCOME_CONVERGED
    assert "authorship not claimed" in member.detail
    assert member.observation.state == RESTORE_OBSERVATION_NOT_RECORDED
    assert member.observation.pointer is None
    assert member.observation.fingerprint is None


def test_object_leg_unknown_create_reconciled_converge_records_no_live_state(
    service: CoordinatorService,
) -> None:
    """On the create path, unknowable authorship still discards nothing.

    The companion to the test above, and the reason the unknown arm is not one
    blanket answer: what the leg overwrote is settled by its READ, not by whose
    put landed. The live read found the member absent, so no writer — this run
    or a peer — could have destroyed content that was not there. Answering
    ``not_recorded`` here would fire an operator's gate on a member that
    provably lost nothing.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    checkpoint_id = _object_checkpoint(service, client, "gone.json", b"keep-me", "unknown-create")
    client.delete_object(Bucket="demo", Key="gone.json")

    member = _restore_one_object(
        service, _ConvergeOnCasPut(client, "demo", "gone.json"), "gone.json", checkpoint_id
    )

    assert member.outcome == RESTORE_OUTCOME_CONVERGED
    assert member.observation.state == RESTORE_OBSERVATION_NO_LIVE_STATE
    assert member.observation.pointer is None


def test_object_legs_add_no_substrate_call_to_record_the_observation(
    service: CoordinatorService,
) -> None:
    """The observation rides the read the leg already issues (SPLIT-COMPARAND).

    The counts below are the ones the engine issued BEFORE the observation
    existed, pinned verbatim. A second read to fetch a pointer would describe
    bytes it never saw — and on the delete leg it is also impossible, because
    the versioned read refuses an unversioned bucket, so the honest state alone
    is what that leg can afford. Counted at the client, not on a mock's call
    list, so a helper quietly gaining a second GET is visible here.
    """
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=True)
    modify = _object_checkpoint(service, client, "cfg.json", b"captured", "cp-modify")
    create = _object_checkpoint(service, client, "gone.json", b"keep-me", "cp-create")
    delete = _object_checkpoint(service, client, "ghost.json", None, "cp-delete")
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"peer edit")
    client.delete_object(Bucket="demo", Key="gone.json")
    client.put_object(Bucket="demo", Key="ghost.json", Body=b"intruder")

    for checkpoint_id, key, expected in (
        # modify: one live comparand read + one pinned read, one If-Match put.
        (modify, "cfg.json", (2, 1, 0)),
        # create: the live read (absent) + one pinned read, one create put.
        (create, "gone.json", (2, 1, 0)),
        # delete: the presence probe alone, then the unconditional delete.
        (delete, "ghost.json", (1, 0, 1)),
    ):
        before = (len(client.get_calls), len(client.put_calls), len(client.delete_calls))
        member = _restore_one_object(service, client, key, checkpoint_id)
        after = (len(client.get_calls), len(client.put_calls), len(client.delete_calls))
        assert member.outcome == RESTORE_OUTCOME_RESTORED
        assert tuple(a - b for a, b in zip(after, before)) == expected, key
        # And the observation is still populated from those calls alone.
        assert member.observation.state != RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED


# ===========================================================================
# Registration — WV plan Unit 5 (R4/R5): coordinator registration of restore
# results (all-or-nothing for writes), registered-writer concurrency semantics
# (invalidation + fence), and restore monotonicity (forward commit carrying
# old bytes).
# ===========================================================================


class _RegistrationProbeService(CoordinatorService):
    """CoordinatorService instrumented for the Unit-5 tests: call counters,
    the durable status-write trace, and deterministic crash / race injection
    hooks (the delay-injection harness's registration half)."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        super().__init__(registry)
        self.commit_all_calls = 0
        self.register_calls = 0
        self.status_writes: list[str] = []
        #: Callable fired INSIDE the first commit_all, before it runs — lands
        #: a peer commit between the registration's comparand read and its CAS.
        self.before_first_commit_all: Any | None = None
        #: Callable fired INSIDE EVERY commit_all, before it runs — sustains
        #: registered-writer contention across the whole re-drive budget (the
        #: FIX-5 exhaustion driver).
        self.before_every_commit_all: Any | None = None
        #: 1-based register calls that raise OccCallerTransientError BEFORE
        #: delegating (a peer's commit left the controller mid-transient:
        #: retry-eligible, budget-bounded — the FIX-5 continue branch).
        self.transient_on_register_calls: set[int] = set()
        #: 1-based register call to crash BEFORE delegating (kill between the
        #: legs and the registration).
        self.crash_on_register_call: int | None = None
        #: Crash AFTER the registration landed (the marker-write window).
        self.crash_after_register_once: bool = False
        #: Status value whose FIRST durable write crashes (e.g. "concluded").
        self.crash_on_status_once: str | None = None

    def commit_all(self, **kwargs: Any) -> Any:
        self.commit_all_calls += 1
        if self.commit_all_calls == 1 and self.before_first_commit_all is not None:
            self.before_first_commit_all()
        if self.before_every_commit_all is not None:
            self.before_every_commit_all()
        return super().commit_all(**kwargs)

    def register_workspace_restore(self, **kwargs: Any) -> Any:
        self.register_calls += 1
        if self.register_calls in self.transient_on_register_calls:
            raise OccCallerTransientError(
                "controller invalidated mid-flight (scripted transient)"
            )
        if self.crash_on_register_call == self.register_calls:
            raise RuntimeError("simulated crash before registration")
        result = super().register_workspace_restore(**kwargs)
        if self.crash_after_register_once:
            self.crash_after_register_once = False
            raise RuntimeError("simulated crash after registration landed")
        return result

    def set_workspace_checkpoint_restore_status(
        self, checkpoint_id: str, status: str, **kwargs: Any
    ) -> None:
        if self.crash_on_status_once == status:
            self.crash_on_status_once = None
            raise RuntimeError(f"simulated crash on status write {status!r}")
        self.status_writes.append(status)
        super().set_workspace_checkpoint_restore_status(checkpoint_id, status, **kwargs)


def _probe_service() -> tuple[ArtifactRegistry, _RegistrationProbeService]:
    registry = ArtifactRegistry()
    return registry, _RegistrationProbeService(registry)


_FP_PLAN = sha256_hex(b"plan text")


def _checkpoint_diverged_file(
    service: Any,
) -> tuple[str, _FakeFileStore, _FakeResolver]:
    """One file member captured at (b'plan text', v7) then diverged live, so
    the restore leg must WRITE — the written-member registration shape."""
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    checkpoint_id = versioner.checkpoint("reg").record.checkpoint_id
    files.put("notes/plan.md", b"edited", 9)
    return checkpoint_id, files, resolver


def _restorer_for(service: Any, files: _FakeFileStore, resolver: _FakeResolver) -> Any:
    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "notes/plan.md")
    return restorer


def test_restored_file_member_registers_via_commit_all_and_invalidates_peer() -> None:
    """The written-file leg of the registration split: ONE all-or-nothing
    commit_all, hash-only, forward version bump, registered peer invalidated."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    peer = uuid.uuid4()
    registry.set_agent_state(art.id, peer, MESIState.SHARED, trigger="fetch", tick=1)
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    report = _restorer_for(service, files, resolver).restore(checkpoint_id)

    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_RESTORED
    reg = report.registration
    assert reg is not None
    # Typed status matched by IDENTITY (the constant object flows through).
    assert reg.status is WORKSPACE_REGISTRATION_COMMITTED
    assert reg.registered == {"notes/plan.md": 2}
    assert reg.attempts == 1
    assert service.commit_all_calls == 1

    # The coordinator side: forward commit carrying OLD bytes' hash — version
    # strictly increases, content hash is the manifest fingerprint, and the
    # body NEVER entered the coordinator (hash-only registration).
    updated = registry.get_artifact(art.id)
    assert updated is not None
    assert updated.version == 2
    check_monotonic_version(1, updated.version)  # never a decrement
    assert updated.content_hash == _FP_PLAN
    assert registry.get_content(art.id) == "old-body"  # unchanged: no bytes in

    # Registered-writer concurrency: the peer holding SHARED was invalidated
    # atomically by the commit and the signal count surfaces on the report.
    assert registry.get_agent_state(art.id, peer) == MESIState.INVALID
    assert reg.invalidated_peers == 1

    # The durable idempotency marker walked in_progress -> registered -> concluded.
    assert service.status_writes == [
        RESTORE_STATUS_IN_PROGRESS,
        RESTORE_STATUS_REGISTERED,
        RESTORE_STATUS_CONCLUDED,
    ]


def test_s3_written_member_registration_is_manifest_side_only() -> None:
    """The S3 leg of the split: a written BYO-substrate member registers
    MANIFEST-SIDE (its durable outcome row) — no coordinator artifact row is
    ever forced (the substrate owns identity), and the commit_all write-set
    holds ONLY the file member."""
    registry, service = _probe_service()
    service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"cfg-v1")
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    versioner.add_object_member(obj, "cfg.json")
    checkpoint_id = versioner.checkpoint("split").record.checkpoint_id
    files.put("notes/plan.md", b"edited", 9)
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"cfg-v2")

    restorer = _versioner(service, resolver=resolver)
    restorer.add_file_member(files, "notes/plan.md")
    restorer.add_object_member(obj, "cfg.json")
    report = restorer.restore(checkpoint_id)

    by_path = report.members_by_path
    assert by_path["notes/plan.md"].outcome == RESTORE_OUTCOME_RESTORED
    assert by_path["s3://cfg.json"].outcome == RESTORE_OUTCOME_RESTORED
    reg = report.registration
    assert reg is not None
    assert reg.status == WORKSPACE_REGISTRATION_COMMITTED
    assert reg.registered == {"notes/plan.md": 2}
    assert reg.substrate_registered == ("s3://cfg.json",)
    assert service.commit_all_calls == 1
    # No coordinator artifact identity was minted for the S3 member.
    names = {registry.get_artifact(aid).name for aid in registry.artifact_ids()}
    assert "s3://cfg.json" not in names
    # Its manifest-side registration IS the durable outcome row.
    stored = {m.member_path: m for m in registry.get_checkpoint_members(checkpoint_id)}
    assert stored["s3://cfg.json"].restore_outcome == RESTORE_OUTCOME_RESTORED


def test_delete_only_restore_records_only_never_calls_commit_all() -> None:
    """The delete leg of the split: manifest-side deleted_at_restore records
    ONLY — commit_all is NEVER called (asserted via the counting service), and
    the registration concludes typed-EMPTY."""
    registry, service = _probe_service()
    client, obj = _s3()
    versioner = _versioner(service)
    versioner.add_object_member(obj, "ghost.json")  # ABSENT at capture
    checkpoint_id = versioner.checkpoint("del-only").record.checkpoint_id
    client.put_object(Bucket="demo", Key="ghost.json", Body=b"intruder")

    restorer = _versioner(service)
    restorer.add_object_member(obj, "ghost.json")
    report = restorer.restore(checkpoint_id)

    (member,) = report.members
    assert member.outcome == RESTORE_OUTCOME_RESTORED
    assert member.deleted_at_restore is not None
    reg = report.registration
    assert reg is not None
    assert reg.status == WORKSPACE_REGISTRATION_EMPTY
    assert reg.deleted_recorded == ("s3://ghost.json",)
    assert service.commit_all_calls == 0
    # The empty write-set still reaches the service once (#191: it claims the
    # checkpoint), which answers empty without calling commit_all.
    assert service.register_calls == 1
    assert registry.get_checkpoint(checkpoint_id).registered_by == OWNER
    (stored,) = registry.get_checkpoint_members(checkpoint_id)
    assert stored.deleted_at_restore is not None
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED


def test_all_converged_empty_restore_concludes_with_empty_registration() -> None:
    """Every member converged/skipped: the commit write-set is empty — the
    status-update conclusion alone, commit_all never called."""
    registry, service = _probe_service()
    files = _FakeFileStore()
    files.put("calm.md", b"calm", 5)
    resolver = _FakeResolver()
    resolver.keep("calm.md", 5, b"calm")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "calm.md")
    versioner.add_forward_only_member("effects/notify")
    checkpoint_id = versioner.checkpoint("noop").record.checkpoint_id

    report = versioner.restore(checkpoint_id)

    by_path = report.members_by_path
    assert by_path["calm.md"].outcome == RESTORE_OUTCOME_CONVERGED
    assert by_path["effects/notify"].outcome == RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED
    reg = report.registration
    assert reg is not None
    assert reg.status == WORKSPACE_REGISTRATION_EMPTY
    assert reg.registered == {} and reg.deleted_recorded == ()
    assert service.commit_all_calls == 0
    assert service.register_calls == 1  # the claim (#191); commit_all never ran
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED


def test_crash_between_legs_and_registration_resume_registers_exactly_once() -> None:
    """Registration all-or-nothing under crash injection: a kill between the
    legs and the registration leaves NO partial coordinator state; the resumed
    run registers exactly once."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    service.crash_on_register_call = 1
    with pytest.raises(RuntimeError, match="before registration"):
        _restorer_for(service, files, resolver).restore(checkpoint_id)

    # No partial coordinator state: the artifact never moved, the marker never
    # landed, the member outcome is durable (the legs concluded).
    assert registry.get_artifact(art.id).version == 1
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_IN_PROGRESS
    (stored,) = registry.get_checkpoint_members(checkpoint_id)
    assert stored.restore_outcome == RESTORE_OUTCOME_RESTORED

    # Resume: the registration runs EXACTLY once (one commit_all, one bump).
    service.crash_on_register_call = None
    report = _restorer_for(service, files, resolver).restore(checkpoint_id)
    (member,) = report.members
    assert member.resumed_from_prior_run is True
    reg = report.registration
    assert reg is not None
    assert reg.status == WORKSPACE_REGISTRATION_COMMITTED
    assert reg.registered == {"notes/plan.md": 2}
    assert service.commit_all_calls == 1
    assert registry.get_artifact(art.id).version == 2
    assert service.status_writes == [
        RESTORE_STATUS_IN_PROGRESS,  # crashed run
        RESTORE_STATUS_IN_PROGRESS,  # resume
        RESTORE_STATUS_REGISTERED,
        RESTORE_STATUS_CONCLUDED,
    ]


def test_crash_after_registration_landed_before_marker_no_double_commit() -> None:
    """The marker-write crash window: the commit landed but 'registered' never
    did. The resumed registration re-runs and the hash filter answers it
    EMPTY — exactly-once, no second version bump."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    service.crash_after_register_once = True
    with pytest.raises(RuntimeError, match="after registration landed"):
        _restorer_for(service, files, resolver).restore(checkpoint_id)

    # The commit landed; the marker did not.
    assert registry.get_artifact(art.id).version == 2
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_IN_PROGRESS

    report = _restorer_for(service, files, resolver).restore(checkpoint_id)
    reg = report.registration
    assert reg is not None
    assert reg.status == WORKSPACE_REGISTRATION_EMPTY  # already at the fingerprint
    assert reg.skipped == ("notes/plan.md",)
    assert service.commit_all_calls == 1  # never a second commit
    assert registry.get_artifact(art.id).version == 2  # exactly one bump
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED


def test_crash_after_registered_marker_resume_skips_registration() -> None:
    """The durable marker honored: a crash AFTER 'registered' but before
    'concluded' resumes WITHOUT re-entering the registration step at all."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    service.crash_on_status_once = RESTORE_STATUS_CONCLUDED
    with pytest.raises(RuntimeError, match="concluded"):
        _restorer_for(service, files, resolver).restore(checkpoint_id)

    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_REGISTERED
    assert registry.get_artifact(art.id).version == 2
    register_calls_before = service.register_calls

    report = _restorer_for(service, files, resolver).restore(checkpoint_id)
    reg = report.registration
    assert reg is not None
    assert reg.status is WORKSPACE_REGISTRATION_PRIOR_RUN
    assert service.register_calls == register_calls_before  # step never re-entered
    assert service.commit_all_calls == 1
    assert registry.get_artifact(art.id).version == 2  # exactly once
    record = registry.get_checkpoint(checkpoint_id)
    assert record is not None and record.restore_status == RESTORE_STATUS_CONCLUDED


def test_registered_live_writer_racing_registration_retries_bounded_and_wins() -> None:
    """A registered live writer's commit lands between the registration's
    comparand read and its commit_all: the batch is HELD version_mismatch
    (all-or-nothing, nothing mutated), the seam re-drives from fresh
    comparands (bounded), and the landed registration invalidates the peer."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    peer = uuid.uuid4()
    registry.set_agent_state(art.id, peer, MESIState.SHARED, trigger="fetch", tick=1)
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    def _peer_commit_lands_first() -> None:
        result = registry.commit_cas(
            art.id,
            peer,
            expected_version=1,
            content_hash=sha256_hex(b"peer-body"),
            content="peer-body",
            tick=5,
        )
        assert not isinstance(result, ConflictDetail)  # the peer WON (v1 -> v2)

    service.before_first_commit_all = _peer_commit_lands_first
    report = _restorer_for(service, files, resolver).restore(checkpoint_id)

    reg = report.registration
    assert reg is not None
    assert reg.status == WORKSPACE_REGISTRATION_COMMITTED
    assert reg.attempts == 2  # attempt 1 HELD version_mismatch; attempt 2 landed
    assert service.commit_all_calls == 2
    # Forward past the peer's commit: v1 -> (peer) v2 -> (registration) v3.
    updated = registry.get_artifact(art.id)
    assert updated.version == 3
    check_monotonic_version(2, updated.version)
    assert updated.content_hash == _FP_PLAN
    assert reg.registered == {"notes/plan.md": 3}
    # The peer (SHARED after its own OCC win) was invalidated by the landing.
    assert registry.get_agent_state(art.id, peer) == MESIState.INVALID
    assert reg.invalidated_peers == 1


def test_superseded_controller_late_apply_rejected_by_fence() -> None:
    """The read-generation fence: a controller whose grant a sweep reclaimed
    is SUPERSEDED — its late registration apply is rejected
    (stale_read_generation), NEVER retried, and nothing registers
    (all-or-nothing). The restore still concludes, refusal reported."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    # The controller (the versioner's owner) held EXCLUSIVE (read_gen 0), then
    # the sweep reclaimed it (owner_gen 1): superseded.
    registry.set_agent_state(art.id, OWNER, MESIState.EXCLUSIVE, trigger="write", tick=1)
    registry.set_agent_state(
        art.id, OWNER, MESIState.INVALID, trigger="reclaim_heartbeat", tick=2
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    report = _restorer_for(service, files, resolver).restore(checkpoint_id)

    reg = report.registration
    assert reg is not None
    assert reg.status is WORKSPACE_REGISTRATION_REFUSED
    assert reg.refused == {"notes/plan.md": STALE_READ_GENERATION_REASON}
    assert service.commit_all_calls == 1  # the fence is terminal on sight
    # The late apply was fenced out: no phantom bump, hash untouched.
    updated = registry.get_artifact(art.id)
    assert updated.version == 1
    assert updated.content_hash == sha256_hex(b"old-body")
    # The restore still CONCLUDES; the refusal skipped the 'registered' marker.
    # REFUSED is TERMINAL: 'concluded' is absorbing, so a later restore()
    # REPORTS the refusal rebuilt from durable state and never re-attempts it
    # (only a crash BEFORE the concluded write resumes into the seam once
    # more — where the fence rejects a superseded controller again).
    assert report.status == RESTORE_STATUS_CONCLUDED
    assert service.status_writes == [
        RESTORE_STATUS_IN_PROGRESS,
        RESTORE_STATUS_CONCLUDED,
    ]


def test_restore_monotonicity_forward_commit_carrying_old_bytes() -> None:
    """R5: restore registers as a FORWARD commit — the coordinator version
    strictly increases past every live-writer commit while the content hash
    returns to the captured (old) fingerprint; never a decrement anywhere."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="v1-body", content_hash=sha256_hex(b"v1-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    # Live writers advance the coordinator past the capture: v1 -> v4.
    observed_versions = [registry.get_artifact(art.id).version]
    writer = uuid.uuid4()
    for i in range(3):
        current = registry.get_artifact(art.id).version
        result = registry.commit_cas(
            art.id,
            writer,
            expected_version=current,
            content_hash=sha256_hex(f"writer-{i}".encode()),
            content=f"writer-{i}",
            tick=10 + i,
        )
        assert not isinstance(result, ConflictDetail)
        observed_versions.append(registry.get_artifact(art.id).version)
    assert observed_versions == [1, 2, 3, 4]

    report = _restorer_for(service, files, resolver).restore(checkpoint_id)

    reg = report.registration
    assert reg is not None
    assert reg.status == WORKSPACE_REGISTRATION_COMMITTED
    updated = registry.get_artifact(art.id)
    # Strictly increasing across the whole history — old bytes, NEW version.
    observed_versions.append(updated.version)
    assert observed_versions == [1, 2, 3, 4, 5]
    for previous, current in zip(observed_versions, observed_versions[1:]):
        check_monotonic_version(previous, current)
        assert current > previous  # strict: a restore is never a re-stamp
    assert updated.content_hash == _FP_PLAN  # the captured state, re-imposed


def test_concluded_report_rebuilds_registration_from_durable_state() -> None:
    """A report rebuilt from durable rows (idempotent re-restore of a
    concluded checkpoint) REBUILDS its registration answer — the run-local
    detail is gone, but the durable truths (member rows + the coordinator's
    registered state) re-derive it: never a silent registration=None."""
    _registry, service = _probe_service()
    files = _FakeFileStore()
    files.put("calm.md", b"calm", 5)
    resolver = _FakeResolver()
    resolver.keep("calm.md", 5, b"calm")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "calm.md")
    checkpoint_id = versioner.checkpoint("re").record.checkpoint_id
    first = versioner.restore(checkpoint_id)
    assert first.registration is not None
    assert first.registration.status is WORKSPACE_REGISTRATION_EMPTY  # converged: no writes
    second = versioner.restore(checkpoint_id)
    assert second.status == RESTORE_STATUS_CONCLUDED
    assert second.registration is not None  # rebuilt, never None-means-fine
    assert second.registration.status is WORKSPACE_REGISTRATION_EMPTY


def test_concluded_report_rebuilds_committed_registration_as_prior_run() -> None:
    """The committed half of the rebuild: after a COMMITTED registration
    concludes, a re-restore re-reads every written member at its manifest
    fingerprint and answers registered_by_prior_run — commit_all and the
    registration seam are NOT re-driven (exactly-once across re-restores)."""
    _registry, service = _probe_service()
    service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)
    first = _restorer_for(service, files, resolver).restore(checkpoint_id)
    assert first.registration is not None
    assert first.registration.status is WORKSPACE_REGISTRATION_COMMITTED
    assert service.commit_all_calls == 1

    second = _restorer_for(service, files, resolver).restore(checkpoint_id)

    reg = second.registration
    assert reg is not None
    assert reg.status is WORKSPACE_REGISTRATION_PRIOR_RUN
    assert service.commit_all_calls == 1  # nothing re-committed
    assert service.register_calls == 1  # the seam itself was not re-driven


def test_fence_refused_re_restore_reports_refusal_not_success() -> None:
    """FIX-1 regression (REFUSED masked as success): after a fence-REFUSED
    registration concludes, a SECOND restore() must surface the refusal —
    registration.status REFUSED rebuilt from durable state, never None — and
    must not call commit_all again (REFUSED is terminal; concluded is
    absorbing)."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    registry.set_agent_state(art.id, OWNER, MESIState.EXCLUSIVE, trigger="write", tick=1)
    registry.set_agent_state(
        art.id, OWNER, MESIState.INVALID, trigger="reclaim_heartbeat", tick=2
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)

    first = _restorer_for(service, files, resolver).restore(checkpoint_id)
    assert first.registration is not None
    assert first.registration.status is WORKSPACE_REGISTRATION_REFUSED
    assert service.commit_all_calls == 1

    second = _restorer_for(service, files, resolver).restore(checkpoint_id)

    assert second.status == RESTORE_STATUS_CONCLUDED
    reg = second.registration
    assert reg is not None  # never silent None-means-fine
    assert reg.status is WORKSPACE_REGISTRATION_REFUSED
    assert "notes/plan.md" in reg.detail  # the un-registered member is named
    assert service.commit_all_calls == 1  # nothing re-attempted (terminal)
    assert service.register_calls == 1  # the seam itself was not re-driven
    # The fenced-out apply stayed out across BOTH runs: no phantom bump.
    updated = registry.get_artifact(art.id)
    assert updated.version == 1
    assert updated.content_hash == sha256_hex(b"old-body")


def test_registration_budget_exhaustion_under_sustained_contention() -> None:
    """FIX-5a: sustained version_mismatch contention (a registered peer lands
    a commit inside EVERY comparand gap) drives the registration re-drive
    loop to budget exhaustion — REFUSED (nothing registered, all-or-nothing),
    the restore still concludes WITHOUT the 'registered' marker, and a
    re-restore REPORTS the refusal rebuilt from durable state (post-FIX-1
    semantics), never re-attempting."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)
    writer = uuid.uuid4()

    def _peer_commit_every_gap() -> None:
        current = registry.get_artifact(art.id).version
        result = registry.commit_cas(
            art.id,
            writer,
            expected_version=current,
            content_hash=sha256_hex(f"peer-{current}".encode()),
            content=f"peer-{current}",
            tick=50 + current,
        )
        assert not isinstance(result, ConflictDetail)  # the peer always WINS

    service.before_every_commit_all = _peer_commit_every_gap
    report = _restorer_for(service, files, resolver).restore(checkpoint_id)

    reg = report.registration
    assert reg is not None
    assert reg.status is WORKSPACE_REGISTRATION_REFUSED
    assert reg.attempts == MAX_RESTORE_LEG_REDRIVES + 1  # budget fully consumed
    assert service.commit_all_calls == MAX_RESTORE_LEG_REDRIVES + 1
    assert reg.refused == {"notes/plan.md": "version_mismatch"}
    assert "budget exhausted" in reg.detail
    # Status honesty: concluded WITHOUT the 'registered' marker (refused path).
    assert report.status == RESTORE_STATUS_CONCLUDED
    assert service.status_writes == [
        RESTORE_STATUS_IN_PROGRESS,
        RESTORE_STATUS_CONCLUDED,
    ]
    # Nothing registered: the artifact carries the last PEER hash, not the
    # manifest fingerprint (all-or-nothing held every attempt).
    assert registry.get_artifact(art.id).content_hash != _FP_PLAN

    # Post-FIX-1: the re-restore rebuilds REFUSED (never None) and does not
    # re-attempt — commit_all and the seam are untouched by the second run.
    calls_before = (service.commit_all_calls, service.register_calls)
    second = _restorer_for(service, files, resolver).restore(checkpoint_id)
    assert second.registration is not None
    assert second.registration.status is WORKSPACE_REGISTRATION_REFUSED
    assert (service.commit_all_calls, service.register_calls) == calls_before


def _restorer_owned_by(
    service: Any, files: _FakeFileStore, resolver: _FakeResolver, owner: uuid.UUID
) -> WorkspaceVersioner:
    restorer = WorkspaceVersioner(
        service=service, owner=owner, clock=_TickClock(), file_resolver=resolver
    )
    restorer.add_file_member(files, "notes/plan.md")
    return restorer


def test_restore_by_non_receiver_refused_before_any_write() -> None:
    """#191: a checkpoint taken with a ``receiver`` restores only through a
    versioner owned by that receiver. Any other versioner — the one that took
    it included — is refused typed in pre-flight: no status write, no leg, no
    bytes, no claim."""
    registry, service = _probe_service()
    receiver = uuid.uuid4()
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    taker = _versioner(service, resolver=resolver)
    taker.add_file_member(files, "notes/plan.md")
    checkpoint = taker.checkpoint("handoff", receiver=receiver)
    assert checkpoint.record.receiver == receiver
    assert checkpoint.record.owner == OWNER
    checkpoint_id = checkpoint.record.checkpoint_id
    files.put("notes/plan.md", b"edited", 9)

    for stranger in (OWNER, uuid.uuid4()):
        with pytest.raises(CheckpointRegistrationRefused) as excinfo:
            _restorer_owned_by(service, files, resolver, stranger).restore(
                checkpoint_id
            )
        assert excinfo.value.reason is CHECKPOINT_NOT_THE_RECEIVER_REASON
    assert service.status_writes == []
    assert service.register_calls == 0
    assert files.read_with_version("notes/plan.md")[0] == b"edited"
    assert registry.get_checkpoint(checkpoint_id).registered_by is None

    report = _restorer_owned_by(service, files, resolver, receiver).restore(
        checkpoint_id
    )
    assert report.status == RESTORE_STATUS_CONCLUDED
    assert report.registration is not None
    assert report.registration.status in (
        WORKSPACE_REGISTRATION_COMMITTED,
        WORKSPACE_REGISTRATION_EMPTY,
    )
    assert registry.get_checkpoint(checkpoint_id).registered_by == receiver


def test_restore_of_a_checkpoint_another_owner_registered_is_refused() -> None:
    """#191, the versioner's half of ``already_registered``: once one owner's
    restore registered a checkpoint, another owner's restore is refused in
    pre-flight — it never re-drives the legs and never reads the first run's
    registration as its own ``registered_by_prior_run``. The first owner's
    re-restore still rebuilds its report."""
    registry, service = _probe_service()
    service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)
    first = _restorer_for(service, files, resolver).restore(checkpoint_id)
    assert first.registration.status is WORKSPACE_REGISTRATION_COMMITTED
    assert registry.get_checkpoint(checkpoint_id).registered_by == OWNER
    writes_before = list(service.status_writes)

    with pytest.raises(CheckpointRegistrationRefused) as excinfo:
        _restorer_owned_by(service, files, resolver, uuid.uuid4()).restore(
            checkpoint_id
        )
    assert excinfo.value.reason is CHECKPOINT_ALREADY_REGISTERED_REASON
    assert service.status_writes == writes_before

    again = _restorer_for(service, files, resolver).restore(checkpoint_id)
    assert again.registration.status is WORKSPACE_REGISTRATION_PRIOR_RUN


def test_restore_with_nothing_to_register_still_claims_the_checkpoint() -> None:
    """#191: a restore whose write-set is empty (every member converged)
    claims the checkpoint all the same — another owner's restore is refused
    ``already_registered`` instead of getting the first run's concluded
    report, whether or not the first run had bytes to write."""
    registry, service = _probe_service()
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    taker = _versioner(service, resolver=resolver)
    taker.add_file_member(files, "notes/plan.md")
    checkpoint_id = taker.checkpoint("calm").record.checkpoint_id

    first = _restorer_for(service, files, resolver).restore(checkpoint_id)
    assert first.registration.status == WORKSPACE_REGISTRATION_EMPTY
    assert registry.get_checkpoint(checkpoint_id).registered_by == OWNER

    with pytest.raises(CheckpointRegistrationRefused) as excinfo:
        _restorer_owned_by(service, files, resolver, uuid.uuid4()).restore(
            checkpoint_id
        )
    assert excinfo.value.reason is CHECKPOINT_ALREADY_REGISTERED_REASON


def test_registration_claimed_mid_restore_concludes_refused() -> None:
    """#191 race: a concurrent controller claims the checkpoint between this
    restore's pre-flight and its registration. The service's atomic claim
    refuses it ``already_registered``; the restore still CONCLUDES, with the
    registration ``refused`` (terminal, not retried) and nothing committed."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)
    rival = uuid.uuid4()
    real_register = service.register_workspace_restore

    def _rival_claims_first(**kwargs: Any) -> Any:
        registry.claim_checkpoint_registration(checkpoint_id, rival)
        return real_register(**kwargs)

    service.register_workspace_restore = _rival_claims_first  # type: ignore[method-assign]
    report = _restorer_for(service, files, resolver).restore(checkpoint_id)

    assert report.status == RESTORE_STATUS_CONCLUDED
    reg = report.registration
    assert reg is not None
    assert reg.status is WORKSPACE_REGISTRATION_REFUSED
    assert reg.refused == {"notes/plan.md": CHECKPOINT_ALREADY_REGISTERED_REASON}
    assert service.register_calls == 1  # terminal: never re-driven
    assert service.commit_all_calls == 0
    assert registry.get_artifact(art.id).version == 1
    assert service.status_writes == [
        RESTORE_STATUS_IN_PROGRESS,
        RESTORE_STATUS_CONCLUDED,
    ]


def test_registration_transient_invalidation_retries_and_lands() -> None:
    """FIX-5b: OccCallerTransientError raised by the registration call (a
    peer's commit left the controller mid-transient) is retry-eligible — the
    loop consumes one budget slot, continues, and the next attempt lands
    COMMITTED; the transient never escapes the restore."""
    registry, service = _probe_service()
    art = service.register_artifact(
        name="notes/plan.md", content="old-body", content_hash=sha256_hex(b"old-body")
    )
    checkpoint_id, files, resolver = _checkpoint_diverged_file(service)
    service.transient_on_register_calls = {1}

    report = _restorer_for(service, files, resolver).restore(checkpoint_id)

    reg = report.registration
    assert reg is not None
    assert reg.status is WORKSPACE_REGISTRATION_COMMITTED
    assert reg.attempts == 2  # attempt 1 raised the transient; attempt 2 landed
    assert service.register_calls == 2
    assert service.commit_all_calls == 1  # the transient attempt never committed
    updated = registry.get_artifact(art.id)
    assert updated.version == 2
    assert updated.content_hash == _FP_PLAN
    assert service.status_writes == [
        RESTORE_STATUS_IN_PROGRESS,
        RESTORE_STATUS_REGISTERED,
        RESTORE_STATUS_CONCLUDED,
    ]


# ---------------------------------------------------------------------------
# Route level — Unit-5 restore progress + registration endpoints (the remote
# half of the CheckpointRestoreStore seam; mirrors the checkpoint routes)
# ---------------------------------------------------------------------------


def _route_checkpoint(
    client,
    *,
    member_path: str = "notes/plan.md",
    fingerprint: str | None = None,
    receiver_session_id: str | None = None,
) -> str:
    sid = str(uuid.uuid4())
    payload: dict[str, Any] = {
        "session_id": sid,
        "name": "route-restore-cp",
        "window_min": 100.0,
        "window_max": 100.0,
        "members": [
            _member_wire(
                member_path,
                native_token="7",
                fingerprint=fingerprint or sha256_hex(b"plan text"),
                arbitration_tier="no-arbiter",
                restore_tier="restorable-unpinned",
            )
        ],
    }
    if receiver_session_id is not None:
        payload["receiver_session_id"] = receiver_session_id
    status, body = client("POST", "/workspace/checkpoint", payload)
    assert status == 200 and body["ok"] is True
    return body["checkpoint_id"]


def test_route_restore_progress_roundtrip(client) -> None:
    sid = str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client)

    status, body = client(
        "POST",
        "/workspace/restore/status",
        {"session_id": sid, "checkpoint_id": checkpoint_id, "status": "in_progress"},
    )
    assert status == 200 and body["ok"] is True

    status, body = client(
        "POST",
        "/workspace/restore/member",
        {
            "session_id": sid,
            "checkpoint_id": checkpoint_id,
            "member_path": "notes/plan.md",
            "restore_outcome": "restored",
        },
    )
    assert status == 200 and body["ok"] is True

    # The full status walk lands durably, including the Unit-5 marker.
    for next_status in ("registered", "concluded"):
        status, body = client(
            "POST",
            "/workspace/restore/status",
            {"session_id": sid, "checkpoint_id": checkpoint_id, "status": next_status},
        )
        assert status == 200 and body["ok"] is True

    status, listing = client("GET", "/workspace/checkpoints")
    assert status == 200
    (cp,) = listing["checkpoints"]
    assert cp["restore_status"] == "concluded"
    (member,) = cp["members"]
    assert member["restore_outcome"] == "restored"


def test_route_restore_progress_validation_fails_closed(client) -> None:
    sid = str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client)

    # Unknown status — closed vocabulary, boundary-gated.
    status, _ = client(
        "POST",
        "/workspace/restore/status",
        {"session_id": sid, "checkpoint_id": checkpoint_id, "status": "magic"},
    )
    assert status == 400
    # Unknown outcome — closed vocabulary.
    status, _ = client(
        "POST",
        "/workspace/restore/member",
        {
            "session_id": sid,
            "checkpoint_id": checkpoint_id,
            "member_path": "notes/plan.md",
            "restore_outcome": "magic",
        },
    )
    assert status == 400
    # Unknown checkpoint — typed reject body, nothing recorded.
    status, body = client(
        "POST",
        "/workspace/restore/status",
        {"session_id": sid, "checkpoint_id": "nope", "status": "in_progress"},
    )
    assert status == 200 and body["ok"] is False
    status, listing = client("GET", "/workspace/checkpoints")
    (cp,) = listing["checkpoints"]
    assert cp["restore_status"] == "none"


def test_route_restore_register_first_observation_then_commit(client) -> None:
    """Over HTTP against the sqlite-backed coordinator: the first registration
    of an unknown path mints hash-only via resolve_or_register (skipped —
    empty_write_set), a retry says it is one, a fingerprint the manifest did
    not capture is refused (#191 — it used to commit forward), and a SECOND
    checkpoint that captured the member at another fingerprint COMMITS it
    forward."""
    sid = str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client)
    fp_one = sha256_hex(b"plan text")
    fp_two = sha256_hex(b"restored text")

    # Empty writes: typed empty, never a commit.
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {"session_id": sid, "checkpoint_id": checkpoint_id, "writes": []},
    )
    assert status == 200 and body["ok"] is True
    assert body["status"] == "empty_write_set"

    # First observation: the mint IS the registration (skipped, no commit).
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {
            "session_id": sid,
            "checkpoint_id": checkpoint_id,
            "writes": [{"member_path": "notes/plan.md", "fingerprint": fp_one}],
        },
    )
    assert status == 200 and body["ok"] is True
    assert body["status"] == "empty_write_set"
    assert body["skipped"] == ["notes/plan.md"]
    # The empty call above already claimed the checkpoint for this session.
    assert body["retry_of_own_registration"] is True

    # A fingerprint this checkpoint did not capture: refused typed, nothing
    # bumped (pre-#191 this committed v1 -> v2 at the caller's hash).
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {
            "session_id": sid,
            "checkpoint_id": checkpoint_id,
            "writes": [{"member_path": "notes/plan.md", "fingerprint": fp_two}],
        },
    )
    assert status == 200 and body["ok"] is False
    assert body["reason"] == "fingerprint_mismatch"
    assert body["member_paths"] == ["notes/plan.md"]

    # A second checkpoint that captured the member at fp_two commits it
    # FORWARD (v1 -> v2), all-or-nothing.
    second = _route_checkpoint(client, fingerprint=fp_two)
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {
            "session_id": sid,
            "checkpoint_id": second,
            "writes": [{"member_path": "notes/plan.md", "fingerprint": fp_two}],
        },
    )
    assert status == 200 and body["ok"] is True
    assert body["status"] == "committed"
    assert body["versions"] == {"notes/plan.md": 2}
    assert body["refused"] == {}
    assert body["retry_of_own_registration"] is False


def test_route_restore_register_refuses_non_member_and_mints_nothing(
    client, coordinator
) -> None:
    """#191 membership half over HTTP: a path the checkpoint does not
    describe is refused with a typed reason, and the coordinator — which never
    saw the path — holds no artifact for it afterwards."""
    sid = str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client)
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {
            "session_id": sid,
            "checkpoint_id": checkpoint_id,
            "writes": [
                {"member_path": "secrets/other.md", "fingerprint": sha256_hex(b"x")}
            ],
        },
    )
    assert status == 200 and body["ok"] is False
    assert body["reason"] == "not_a_checkpoint_member"
    assert body["member_paths"] == ["secrets/other.md"]
    assert "detail" in body
    assert coordinator.registry.lookup_artifact_id_by_name("secrets/other.md") is None
    status, listing = client("GET", "/workspace/checkpoints")
    (cp,) = listing["checkpoints"]
    assert cp["registered_by"] is None


def test_route_restore_register_second_session_refused(client, coordinator) -> None:
    """#191 / plan B7 over HTTP: a second session registering a checkpoint
    another session already registered is refused ``already_registered``
    (it used to get a success-shaped ``empty_write_set``), and the refusal
    names neither session; the first session's retry is told it is a retry."""
    from ccs.adapters.claude_code.coordinator_server import session_to_agent_id

    first_sid, second_sid = str(uuid.uuid4()), str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client)
    write = {"member_path": "notes/plan.md", "fingerprint": sha256_hex(b"plan text")}

    def register(sid: str) -> dict:
        status, body = client(
            "POST",
            "/workspace/restore/register",
            {"session_id": sid, "checkpoint_id": checkpoint_id, "writes": [write]},
        )
        assert status == 200
        return body

    first = register(first_sid)
    assert first["ok"] is True and first["retry_of_own_registration"] is False

    second = register(second_sid)
    assert second["ok"] is False
    assert second["reason"] == "already_registered"
    assert second["member_paths"] == []
    first_agent = str(session_to_agent_id(first_sid))
    for leaked in (first_sid, first_agent, second_sid):
        assert leaked not in json.dumps(second)

    again = register(first_sid)
    assert again["ok"] is True
    assert again["status"] == "empty_write_set"
    assert again["retry_of_own_registration"] is True

    status, listing = client("GET", "/workspace/checkpoints")
    (cp,) = listing["checkpoints"]
    assert cp["registered_by"] == first_agent
    assert cp["receiver"] is None


def test_route_checkpoint_receiver_binds_the_registration(client) -> None:
    """#191 owner half over HTTP: ``receiver_session_id`` at creation names
    the one session allowed to register; the creating session and any other
    are refused ``not_the_receiver`` and claim nothing."""
    from ccs.adapters.claude_code.coordinator_server import session_to_agent_id

    receiver_sid = str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client, receiver_session_id=receiver_sid)
    write = {"member_path": "notes/plan.md", "fingerprint": sha256_hex(b"plan text")}

    status, listing = client("GET", "/workspace/checkpoints")
    (cp,) = listing["checkpoints"]
    assert cp["receiver"] == str(session_to_agent_id(receiver_sid))
    assert cp["owner"] != cp["receiver"]

    status, body = client(
        "POST",
        "/workspace/restore/register",
        {"session_id": str(uuid.uuid4()), "checkpoint_id": checkpoint_id,
         "writes": [write]},
    )
    assert status == 200 and body["ok"] is False
    assert body["reason"] == "not_the_receiver"
    status, listing = client("GET", "/workspace/checkpoints")
    assert listing["checkpoints"][0]["registered_by"] is None

    status, body = client(
        "POST",
        "/workspace/restore/register",
        {"session_id": receiver_sid, "checkpoint_id": checkpoint_id,
         "writes": [write]},
    )
    assert status == 200 and body["ok"] is True
    status, listing = client("GET", "/workspace/checkpoints")
    assert listing["checkpoints"][0]["registered_by"] == cp["receiver"]


def test_route_restore_progress_is_gated_like_the_registration(client) -> None:
    """#191 over HTTP: a session the receiver binding excludes cannot write
    the restore's progress — conclude the checkpoint, or record a member
    restored/deleted (delete legs register over ``/restore/member``) — which
    would otherwise turn the receiver's restore into a no-op. Once a session
    registered, a rival is refused ``already_registered`` there too."""
    receiver_sid, rival_sid = str(uuid.uuid4()), str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client, receiver_session_id=receiver_sid)
    member = {
        "checkpoint_id": checkpoint_id,
        "member_path": "notes/plan.md",
        "restore_outcome": "restored",
        "deleted_at_restore": 5.0,
    }
    conclude = {"checkpoint_id": checkpoint_id, "status": "concluded"}

    for route, payload in (
        ("/workspace/restore/member", member),
        ("/workspace/restore/status", conclude),
    ):
        status, body = client("POST", route, {"session_id": rival_sid, **payload})
        assert status == 200 and body["ok"] is False, (route, body)
        assert body["reason"] == "not_the_receiver"
        assert body["member_paths"] == []

    status, listing = client("GET", "/workspace/checkpoints")
    (cp,) = listing["checkpoints"]
    assert cp["restore_status"] == "none"
    assert cp["registered_by"] is None
    (row,) = cp["members"]
    assert row["restore_outcome"] is None

    # The receiver drives it, then a rival of the registrant is refused.
    write = {"member_path": "notes/plan.md", "fingerprint": sha256_hex(b"plan text")}
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {"session_id": receiver_sid, "checkpoint_id": checkpoint_id,
         "writes": [write]},
    )
    assert status == 200 and body["ok"] is True, body
    status, body = client(
        "POST", "/workspace/restore/status", {"session_id": receiver_sid, **conclude}
    )
    assert status == 200 and body["ok"] is True, body

    unbound = _route_checkpoint(client)
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {"session_id": receiver_sid, "checkpoint_id": unbound, "writes": [write]},
    )
    assert status == 200 and body["ok"] is True, body
    status, body = client(
        "POST",
        "/workspace/restore/member",
        {"session_id": rival_sid, **member, "checkpoint_id": unbound},
    )
    assert status == 200 and body["ok"] is False
    assert body["reason"] == "already_registered"


def test_route_checkpoint_receiver_session_id_is_shape_checked(client) -> None:
    status, body = client(
        "POST",
        "/workspace/checkpoint",
        {
            "session_id": str(uuid.uuid4()),
            "name": "cp",
            "window_min": 1.0,
            "window_max": 1.0,
            "members": [_member_wire("a.txt")],
            "receiver_session_id": "not a session id",
        },
    )
    assert status == 400
    assert "receiver_session_id" in body["error"]


def test_route_restore_register_boundary_validation(client) -> None:
    sid = str(uuid.uuid4())
    checkpoint_id = _route_checkpoint(client)
    fp = sha256_hex(b"plan text")

    # Unknown checkpoint — the typed identity-stable reason.
    status, body = client(
        "POST",
        "/workspace/restore/register",
        {
            "session_id": sid,
            "checkpoint_id": "nope",
            "writes": [{"member_path": "notes/plan.md", "fingerprint": fp}],
        },
    )
    assert status == 200 and body["ok"] is False
    assert body["reason"] == CHECKPOINT_UNKNOWN_REASON

    base = {"session_id": sid, "checkpoint_id": checkpoint_id}
    # Malformed fingerprint.
    status, _ = client(
        "POST",
        "/workspace/restore/register",
        {**base, "writes": [{"member_path": "a.md", "fingerprint": "beef"}]},
    )
    assert status == 400
    # Duplicate member paths.
    status, _ = client(
        "POST",
        "/workspace/restore/register",
        {
            **base,
            "writes": [
                {"member_path": "a.md", "fingerprint": fp},
                {"member_path": "a.md", "fingerprint": fp},
            ],
        },
    )
    assert status == 400
    # Non-list writes.
    status, _ = client(
        "POST", "/workspace/restore/register", {**base, "writes": "nope"}
    )
    assert status == 400
    # Absolute member path (the authoritative server-side path gate).
    status, _ = client(
        "POST",
        "/workspace/restore/register",
        {**base, "writes": [{"member_path": "/etc/passwd", "fingerprint": fp}]},
    )
    assert status == 400


# ---------------------------------------------------------------------------
# WV Unit 6 — pin legs (minimal, fail-closed): "restorable means restorable,
# or says restorable-unpinned loudly"
# ---------------------------------------------------------------------------


class _CaptureOnlyService:
    """A persist-only seam (no pin surface): delegates the single-transaction
    registration to a real service but speaks NOTHING else — the shape the
    fold-in's fail-fast gate must refuse when pins are needed."""

    def __init__(self, inner: CoordinatorService) -> None:
        self._inner = inner

    def create_workspace_checkpoint(self, **kwargs: Any) -> Any:
        return self._inner.create_workspace_checkpoint(**kwargs)


def _s3_no_lock() -> tuple[LocalS3Client, CoherentObject]:
    """A VERSIONED bucket WITHOUT Object Lock: history exists, no pin can."""
    client = LocalS3Client()
    client.create_bucket("demo", versioned=True, object_lock=False)
    return client, CoherentObject("demo", client=client)


def test_pin_established_survives_lifecycle_expiry_and_restore_succeeds(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """The pin's whole point: hold ON at checkpoint → the manifested version
    survives a lifecycle expiry of noncurrent versions → restore serves it."""
    client, obj = _s3()  # object_lock=True
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")

    cp = versioner.checkpoint("pinned")  # pin=True is the default

    (member,) = cp.members
    assert member.pin_state == PIN_STATE_HELD
    assert member.restore_tier == "restorable"  # the claim is now BACKED
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 1

    # A live writer makes the captured version noncurrent; the lifecycle rule
    # REFUSES the held version (the fake's teeth mirror real S3 semantics).
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert held == [put["VersionId"]] and expired == []

    report = versioner.restore(cp.record.checkpoint_id)
    assert report.members_by_path["s3://cfg.json"].outcome == RESTORE_OUTCOME_RESTORED
    data, _etag = obj.read("cfg.json")
    assert data == b"v1"  # the pinned bytes are back


def test_no_object_lock_pin_downgrades_loudly_and_durably(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """No Object Lock → the typed LegalHoldUnavailable → the LOUD durable
    downgrade: tier restorable-unpinned + pin_state pin_unavailable land in
    one registry write, surfaced in the checkpoint return AND the registry."""
    client, obj = _s3_no_lock()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")

    cp = versioner.checkpoint("unpinnable")

    (member,) = cp.members
    assert member.restore_tier == "restorable-unpinned"  # never restorable
    assert member.pin_state == PIN_STATE_UNAVAILABLE
    # Durable, not just the in-memory return: the registry rows agree.
    (stored,) = registry.get_checkpoint_members(cp.record.checkpoint_id)
    assert stored.restore_tier == "restorable-unpinned"
    assert stored.pin_state == PIN_STATE_UNAVAILABLE
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0


def test_downgraded_unpinned_member_expired_is_target_lost(
    service: CoordinatorService,
) -> None:
    """The pin-path extension of the Unit-4 expired-pin scenario: a member the
    pin legs DOWNGRADED (no Object Lock) is exactly the member a lifecycle
    expiry can take — restore then says target_lost, never best-effort."""
    client, obj = _s3_no_lock()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("unpinnable")
    (member,) = cp.members
    assert member.pin_state == PIN_STATE_UNAVAILABLE  # the loud label, upfront

    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert expired and not held  # nothing protected it

    report = versioner.restore(cp.record.checkpoint_id)
    assert (
        report.members_by_path["s3://cfg.json"].outcome == RESTORE_OUTCOME_TARGET_LOST
    )


def test_internal_release_decrements_and_releases_hold(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("held")
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True

    rows = versioner._release_checkpoint_pins(cp.record.checkpoint_id)

    (member,) = rows
    assert member.pin_state == PIN_STATE_RELEASED
    # Releasing un-backs the claim: the tier downgrade rides the same write.
    assert member.restore_tier == "restorable-unpinned"
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0

    # Idempotent: nothing held, nothing changes, no refcount underflow.
    rows = versioner._release_checkpoint_pins(cp.record.checkpoint_id)
    assert rows[0].pin_state == PIN_STATE_RELEASED
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0

    # A released pin is terminal for the re-drive too (deliberate drop —
    # pin_checkpoint must not resurrect it).
    rows_after = versioner.pin_checkpoint(cp.record.checkpoint_id)
    assert rows_after[0].pin_state == PIN_STATE_RELEASED
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False

    # And the lifecycle can now take the version (the release is real).
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert expired == [put["VersionId"]] and held == []


def test_shared_version_cross_checkpoint_release_keeps_hold(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """Two checkpoints pin the SAME (member_path, versionId). S3's hold is a
    flag, not a counter — the release's cross-checkpoint scan is the counter:
    the first release must leave the hold standing, the LAST drops it."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp1 = versioner.checkpoint("first")
    cp2 = versioner.checkpoint("second")  # no intervening write: same version
    assert cp1.members[0].native_token == cp2.members[0].native_token
    assert cp1.members[0].pin_state == PIN_STATE_HELD
    assert cp2.members[0].pin_state == PIN_STATE_HELD

    versioner._release_checkpoint_pins(cp1.record.checkpoint_id)

    # cp1's record moved; cp2 still holds, so the SUBSTRATE hold survives.
    (row1,) = registry.get_checkpoint_members(cp1.record.checkpoint_id)
    (row2,) = registry.get_checkpoint_members(cp2.record.checkpoint_id)
    assert row1.pin_state == PIN_STATE_RELEASED
    assert row2.pin_state == PIN_STATE_HELD
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert held == [put["VersionId"]] and expired == []

    # Last-out: cp2's release finds no other holder and drops the hold.
    versioner._release_checkpoint_pins(cp2.record.checkpoint_id)
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert expired == [put["VersionId"]] and held == []


class _PinBetweenScanAndDrop(WorkspaceVersioner):
    """Deterministic FIX-3 interleave: right AFTER the release's first
    shared-elsewhere scan answers, the injected callable lands a sibling
    checkpoint's pin on the same (member_path, native_token) — the
    'concurrent pin_checkpoint between the scan and the substrate drop'
    race, reproduced without threads."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.inject_after_first_scan: Any | None = None

    def _pin_shared_elsewhere(self, store: Any, checkpoint_id: str, row: Any) -> bool:
        result = WorkspaceVersioner._pin_shared_elsewhere(store, checkpoint_id, row)
        if self.inject_after_first_scan is not None:
            inject, self.inject_after_first_scan = self.inject_after_first_scan, None
            inject()
        return result


class _PinAfterLastInstantScan(WorkspaceVersioner):
    """The NARROWER interleave: the injected pin lands after BOTH scans — the
    admission check and the last-instant re-check — and therefore inside the
    window between that re-check and the substrate drop, which no re-check can
    close. Reproduced without threads by counting the scan calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.inject_after_scan_number: int | None = None
        self.inject: Any | None = None
        self._scans = 0

    def _pin_shared_elsewhere(self, store: Any, checkpoint_id: str, row: Any) -> bool:
        result = WorkspaceVersioner._pin_shared_elsewhere(store, checkpoint_id, row)
        self._scans += 1
        if self.inject is not None and self._scans == self.inject_after_scan_number:
            inject, self.inject = self.inject, None
            inject()
        return result


def test_release_checkpoint_replaces_a_hold_claimed_inside_the_drop_window(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """A pin landing AFTER the last-instant re-check still loses its hold —
    the one under-retention direction this engine refuses everywhere else.

    No re-check can close that window: the check is a registry read and the
    drop is a separate substrate call. Converging can. After the drop, the
    release re-reads and puts the hold back if a holder appeared, so the peer's
    ``held`` row is backed again rather than left claiming a version that is
    now expirable.
    """
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    releasing = _PinAfterLastInstantScan(
        service=service, owner=OWNER, clock=_TickClock()
    )
    releasing.add_object_member(obj, "cfg.json")
    cp_a = releasing.checkpoint("holder-a")

    sibling = _versioner(service)
    sibling.add_object_member(obj, "cfg.json")
    cp_b = sibling.checkpoint("holder-b", pin=False)  # invisible to both scans
    assert cp_b.members[0].native_token == cp_a.members[0].native_token

    # Scan 1 = admission check, scan 2 = last-instant re-check. Landing the
    # pin after scan 2 puts it exactly in the window the re-check cannot see.
    releasing.inject_after_scan_number = 2
    releasing.inject = lambda: sibling.pin_checkpoint(cp_b.record.checkpoint_id)

    releasing.release_checkpoint(cp_a.record.checkpoint_id)

    (row_a,) = registry.get_checkpoint_members(cp_a.record.checkpoint_id)
    (row_b,) = registry.get_checkpoint_members(cp_b.record.checkpoint_id)
    assert row_a.pin_state == PIN_STATE_RELEASED
    assert row_b.pin_state == PIN_STATE_HELD
    # B's claim is BACKED: the hold it took in the window was put back.
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert held == [put["VersionId"]] and expired == []
    # Last-out still drops.
    sibling.release_checkpoint(cp_b.record.checkpoint_id)
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False


def test_concurrent_pin_between_release_scan_and_drop_keeps_hold(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """FIX-3 regression (under-retention race): a concurrent pin_checkpoint
    recording `held` on the same (member_path, native_token) BETWEEN the
    release's shared-elsewhere scan and its substrate drop must not lose the
    hold it just claimed — the last-instant re-check sees the new holder and
    SKIPS the drop (over-retention direction, safe)."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    releasing = _PinBetweenScanAndDrop(service=service, owner=OWNER, clock=_TickClock())
    releasing.add_object_member(obj, "cfg.json")
    cp_a = releasing.checkpoint("holder-a")
    assert cp_a.members[0].pin_state == PIN_STATE_HELD

    sibling = _versioner(service)
    sibling.add_object_member(obj, "cfg.json")
    # Same version (no intervening write), deliberately unpinned so the
    # release's FIRST scan sees no other holder.
    cp_b = sibling.checkpoint("holder-b", pin=False)
    assert cp_b.members[0].native_token == cp_a.members[0].native_token
    assert cp_b.members[0].pin_state == PIN_STATE_UNPINNED

    releasing.inject_after_first_scan = lambda: sibling.pin_checkpoint(
        cp_b.record.checkpoint_id
    )

    releasing._release_checkpoint_pins(cp_a.record.checkpoint_id)

    # A's record is released; B's freshly-landed hold SURVIVES the release.
    (row_a,) = registry.get_checkpoint_members(cp_a.record.checkpoint_id)
    (row_b,) = registry.get_checkpoint_members(cp_b.record.checkpoint_id)
    assert row_a.pin_state == PIN_STATE_RELEASED
    assert row_b.pin_state == PIN_STATE_HELD
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True
    # The version is genuinely protected: lifecycle expiry cannot take it.
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert held == [put["VersionId"]] and expired == []
    # Last-out still drops: B's own release finds no other holder.
    sibling._release_checkpoint_pins(cp_b.record.checkpoint_id)
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False


# ---------------------------------------------------------------------------
# WV Unit 6 — the PUBLIC release verb: release_checkpoint (issue #192)
# ---------------------------------------------------------------------------


class _StoreAccessProbe:
    """Forwards the pin store surface to a real service, RECORDING each call.

    The ordering witness for R3: a blank/non-string checkpoint id must raise
    before ``release_checkpoint`` reaches the store at all, so ``touched``
    stays empty. A guard placed after the store read records
    ``get_workspace_checkpoint`` instead.

    The CheckpointPinStore members are delegated EXPLICITLY (not via
    ``__getattr__``): runtime-checkable Protocol isinstance uses static
    attribute lookup on 3.12+, which a ``__getattr__`` fallthrough never
    satisfies.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.touched: list[str] = []

    def create_workspace_checkpoint(self, **kwargs: Any) -> Any:
        self.touched.append("create_workspace_checkpoint")
        return self._inner.create_workspace_checkpoint(**kwargs)

    def get_workspace_checkpoint(self, checkpoint_id: str) -> Any:
        self.touched.append("get_workspace_checkpoint")
        return self._inner.get_workspace_checkpoint(checkpoint_id)

    def get_workspace_checkpoint_members(self, checkpoint_id: str) -> Any:
        self.touched.append("get_workspace_checkpoint_members")
        return self._inner.get_workspace_checkpoint_members(checkpoint_id)

    def list_workspace_checkpoints(self) -> Any:
        self.touched.append("list_workspace_checkpoints")
        return self._inner.list_workspace_checkpoints()

    def set_workspace_checkpoint_member_pin(self, *args: Any, **kwargs: Any) -> Any:
        self.touched.append("set_workspace_checkpoint_member_pin")
        return self._inner.set_workspace_checkpoint_member_pin(*args, **kwargs)

    def adjust_workspace_checkpoint_pin_refcount(self, *args: Any, **kwargs: Any) -> Any:
        self.touched.append("adjust_workspace_checkpoint_pin_refcount")
        return self._inner.adjust_workspace_checkpoint_pin_refcount(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ReleaseRaisesUntyped(CoherentObject):
    """A binding whose hold drop fails with an UNTYPED error — neither the
    ``KeyError`` (version gone) nor the :class:`LegalHoldUnavailable` (no
    Object Lock) the engine absorbs, so it propagates."""

    def release_legal_hold(self, artifact_ref: str, *, version_id: str) -> None:
        raise RuntimeError("substrate blew up mid-release")


def test_release_checkpoint_shared_version_drops_only_on_last_holder(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """AE1 — the public verb carries the cross-checkpoint sharing scan: two
    checkpoints pin the SAME (member_path, versionId); the FIRST release
    leaves the substrate hold standing (the peer still depends on it), the
    LAST one drops it. A delegation that lost the scan fails HERE."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp1 = versioner.checkpoint("first")
    cp2 = versioner.checkpoint("second")  # no intervening write: same version
    assert cp1.members[0].native_token == cp2.members[0].native_token

    rows = versioner.release_checkpoint(cp1.record.checkpoint_id)

    (released,) = rows
    assert released.pin_state == PIN_STATE_RELEASED
    (row2,) = registry.get_checkpoint_members(cp2.record.checkpoint_id)
    assert row2.pin_state == PIN_STATE_HELD
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True

    versioner.release_checkpoint(cp2.record.checkpoint_id)  # last out drops it

    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False


def test_release_checkpoint_different_versions_same_path_drop_independently(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """The scan's identity is ``(member_path, native_token)``, and the TOKEN
    half is what this pins. Two checkpoints over the same member path holding
    DIFFERENT versions are not sharing anything, so the earlier one's release
    must drop its own version's hold. A scan degraded to matching member_path
    alone would see the later checkpoint as a holder and skip the drop, and
    every other scan test pins both checkpoints at the SAME version, so none
    of them can tell the two comparisons apart."""
    client, obj = _s3()
    v1 = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    older = versioner.checkpoint("older")
    v2 = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    newer = versioner.checkpoint("newer")
    assert v1["VersionId"] != v2["VersionId"]
    assert older.members[0].member_path == newer.members[0].member_path
    assert older.members[0].native_token != newer.members[0].native_token
    assert obj.legal_hold_status("cfg.json", version_id=v1["VersionId"]) is True
    assert obj.legal_hold_status("cfg.json", version_id=v2["VersionId"]) is True

    versioner.release_checkpoint(older.record.checkpoint_id)

    # The older version is nobody else's: its hold DROPS.
    assert obj.legal_hold_status("cfg.json", version_id=v1["VersionId"]) is False
    # The newer checkpoint is untouched at the same member path.
    assert obj.legal_hold_status("cfg.json", version_id=v2["VersionId"]) is True
    (still_held,) = registry.get_checkpoint_members(newer.record.checkpoint_id)
    assert still_held.pin_state == PIN_STATE_HELD


def test_release_checkpoint_drops_hold_and_downgrades_the_tier(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """The happy path through the public verb: the substrate hold is dropped,
    the row reads ``released``, and the ``restorable`` claim is un-backed in
    the SAME write (no instant claims restorable without a pin behind it)."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("held")
    assert cp.members[0].restore_tier == "restorable"
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True

    rows = versioner.release_checkpoint(cp.record.checkpoint_id)

    (member,) = rows
    assert member.pin_state == PIN_STATE_RELEASED
    assert member.restore_tier == "restorable-unpinned"
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0
    # The release is real: the lifecycle can now take the version.
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert expired == [put["VersionId"]] and held == []


def test_release_checkpoint_twice_is_idempotent(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """AE2 — the second call returns the same durable rows, raises nothing,
    and moves no refcount (a released row is not walked a second time)."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("held")
    versioner.release_checkpoint(cp.record.checkpoint_id)
    after_first = tuple(registry.get_checkpoint_members(cp.record.checkpoint_id))

    rows = versioner.release_checkpoint(cp.record.checkpoint_id)

    assert rows == after_first  # byte-for-byte the same durable rows
    assert rows[0].pin_state == PIN_STATE_RELEASED
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0  # no underflow
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False


def test_release_checkpoint_bad_id_raises_before_any_store_access(
    service: CoordinatorService,
) -> None:
    """AE3 / R3 — a blank or non-string id raises ``ValueError`` BEFORE the
    store is reached. Asserting only that it raised would also pass for a
    guard placed after the store read; the probe pins the ORDERING."""
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    probe = _StoreAccessProbe(service)
    versioner = _versioner(probe)
    versioner.add_object_member(obj, "cfg.json")
    versioner.checkpoint("real")
    assert probe.touched  # the probe DOES see real store work ...
    probe.touched.clear()  # ... so an empty list below is a real signal

    bad_ids: list[Any] = ["", "   ", None, 7]
    for bad in bad_ids:
        with pytest.raises(ValueError):
            versioner.release_checkpoint(bad)

    assert probe.touched == []


def test_release_checkpoint_undeclared_member_preflight_changes_nothing(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """AE4 — a ``held`` S3 row whose object member is not declared on THIS
    versioner refuses pre-flight, before any write: the row, its tier, the
    refcount and the substrate hold are all exactly as they were."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("held")

    fresh = _versioner(service)  # no declared members: cannot reach the bucket
    with pytest.raises(ValueError):
        fresh.release_checkpoint(cp.record.checkpoint_id)

    (stored,) = registry.get_checkpoint_members(cp.record.checkpoint_id)
    assert stored.pin_state == PIN_STATE_HELD
    assert stored.restore_tier == "restorable"
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 1
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True


def test_release_checkpoint_mismatched_key_records_released_but_keeps_hold(
    registry: ArtifactRegistry, service: CoordinatorService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AE5 (CHARACTERIZATION of shipped engine behaviour) — the pre-flight
    only asks that SOME object member is declared at the member path, never
    that it is the one that placed the hold. Re-declaring through a binding
    whose key does not carry the pinned version therefore records ``released``
    while the hold SURVIVES, silently: the engine reads the resulting
    ``KeyError`` as "the hold is moot". This is the silent failure the public
    docstring's re-declaration precondition warns about."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    client.put_object(Bucket="demo", Key="other.json", Body=b"unrelated")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("held")

    # The same member PATH, re-declared over the WRONG key.
    mismatched = _versioner(service)
    mismatched.add_object_member(obj, "other.json", member_path="s3://cfg.json")

    with caplog.at_level(logging.WARNING, logger="ccs.adapters.workspace"):
        rows = mismatched.release_checkpoint(cp.record.checkpoint_id)

    # The silence is the bug: the caller gets no exception and no changed row,
    # so a log line is the ONLY signal this happened.
    (warned,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "s3://cfg.json" in warned.getMessage()
    assert put["VersionId"] in warned.getMessage()

    assert rows[0].pin_state == PIN_STATE_RELEASED  # the record moved ...
    assert rows[0].restore_tier == "restorable-unpinned"
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0
    # ... and the hold did NOT. The version stays un-expirable.
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert held == [put["VersionId"]] and expired == []


def test_release_checkpoint_without_pin_surface_raises_type_error(
    service: CoordinatorService,
) -> None:
    """R5 — the typed refusal survives the delegation: a capture-only seam
    cannot record a release outcome, so the PUBLIC verb fails fast."""
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("held")

    capture_only = _versioner(_CaptureOnlyService(service))
    capture_only.add_object_member(obj, "cfg.json")
    with pytest.raises(TypeError):
        capture_only.release_checkpoint(cp.record.checkpoint_id)


def test_release_checkpoint_untyped_substrate_error_propagates_and_strands(
    registry: ArtifactRegistry, service: CoordinatorService,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CHARACTERIZATION of the stranding R5 names: an untyped substrate error
    (neither ``KeyError`` nor ``LegalHoldUnavailable``) propagates, the rows
    already processed keep their dropped holds, and the FAILING member is
    left ``released`` with its hold still standing — record-before-drop, so
    no later release or re-pin will drop it. ``CoherentObject.
    release_legal_hold`` is the recovery."""
    client, healthy = _s3()
    broken = _ReleaseRaisesUntyped("demo", client=client)
    put_ok = client.put_object(Bucket="demo", Key="a-ok.json", Body=b"a")
    put_bad = client.put_object(Bucket="demo", Key="z-bad.json", Body=b"b")
    versioner = _versioner(service)
    versioner.add_object_member(broken, "z-bad.json")
    versioner.add_object_member(healthy, "a-ok.json")
    cp = versioner.checkpoint("two-members")
    # Members are walked in durable row order (member_path), so the healthy
    # member is processed BEFORE the one that blows up.
    assert [row.member_path for row in cp.members] == ["s3://a-ok.json", "s3://z-bad.json"]

    with caplog.at_level(logging.ERROR, logger="ccs.adapters.workspace"):
        with pytest.raises(RuntimeError):
            versioner.release_checkpoint(cp.record.checkpoint_id)

    # The row goes terminal, so this log line is the only record of WHICH
    # version was stranded -- and the version id is what the documented
    # recovery needs.
    (logged,) = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert "s3://z-bad.json" in logged.getMessage()
    assert put_bad["VersionId"] in logged.getMessage()

    rows = {row.member_path: row for row in registry.get_checkpoint_members(cp.record.checkpoint_id)}
    assert rows["s3://a-ok.json"].pin_state == PIN_STATE_RELEASED
    assert healthy.legal_hold_status("a-ok.json", version_id=put_ok["VersionId"]) is False
    # The failing member: recorded released, hold STRANDED on the substrate.
    assert rows["s3://z-bad.json"].pin_state == PIN_STATE_RELEASED
    assert healthy.legal_hold_status("z-bad.json", version_id=put_bad["VersionId"]) is True


def test_release_checkpoint_drops_the_hold_before_the_refcount_can_refuse(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """The refcount decrement is bookkeeping and it must not be able to skip
    the substrate drop.

    It fails CLOSED when the count would go below zero, which is what a second
    releaser running concurrently causes. Ordered before the drop, that raise
    aborts with the row already terminal and the hold still ON — and no later
    call walks a released row, so the hold is stranded for good. Ordered after,
    the same raise leaves only refcount drift, which this engine already
    documents as benign.

    Driving the refcount to zero out of band is the concurrent releaser's
    effect, made deterministic: the next decrement must refuse.
    """
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("held")
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True
    # Stand in for the peer releaser that already decremented this checkpoint.
    service.adjust_workspace_checkpoint_pin_refcount(cp.record.checkpoint_id, -1)

    with pytest.raises(ValueError):
        versioner.release_checkpoint(cp.record.checkpoint_id)

    # The bookkeeping refused, but the retention control was already cleared.
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False
    (row,) = registry.get_checkpoint_members(cp.record.checkpoint_id)
    assert row.pin_state == PIN_STATE_RELEASED
    # And the version is genuinely expirable now, not merely marked released.
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    expired, held = client.lifecycle_expire_noncurrent("demo", "cfg.json")
    assert expired == [put["VersionId"]] and held == []


def test_pin_checkpoint_after_release_checkpoint_is_one_way(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """``released`` is terminal: the re-drive must not resurrect a pin the
    caller deliberately dropped, so the hold stays off.

    The FILE member is the load-bearing half of this test. The released S3
    row also carries ``restorable-unpinned``, which ``_pin_eligible`` blocks
    on tier alone — so the S3 half stays released even with no ``pin_state``
    guard. A file row's ``restorable-unpinned`` tier is its ELIGIBLE tier, so
    for that member only the ``pin_state`` guard stands between ``released``
    and a resurrected pin.
    """
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_object_member(obj, "cfg.json")
    versioner.add_file_member(files, "notes/plan.md")
    cp = versioner.checkpoint("held")
    versioner.release_checkpoint(cp.record.checkpoint_id)

    rows = {row.member_path: row for row in versioner.pin_checkpoint(cp.record.checkpoint_id)}

    assert rows["s3://cfg.json"].pin_state == PIN_STATE_RELEASED
    assert rows["s3://cfg.json"].restore_tier == "restorable-unpinned"
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False
    # Tier does not gate the file leg — this row witnesses the pin_state guard.
    assert rows["notes/plan.md"].pin_state == PIN_STATE_RELEASED
    assert rows["notes/plan.md"].restore_tier == "restorable-unpinned"
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0


def test_release_checkpoint_unknown_id_raises_typed_refusal(
    service: CoordinatorService,
) -> None:
    """R5 — an unknown id is the typed ``CheckpointUnknown``, not a bare
    KeyError and not a silent no-op."""
    versioner = _versioner(service)
    with pytest.raises(CheckpointUnknown):
        versioner.release_checkpoint("nope")


def test_release_checkpoint_file_only_records_released_with_no_substrate_half(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """A file member's verification pin has no substrate half (``no-arbiter``:
    no hold was ever placed, so none is dropped) — but the release DOES record
    it ``released``, which ``pin_checkpoint`` will not re-drive."""
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")
    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    cp = versioner.checkpoint("file-only")
    (member,) = cp.members
    assert member.pin_state == PIN_STATE_HELD
    assert member.arbitration_tier == "no-arbiter"  # no substrate to hold

    rows = versioner.release_checkpoint(cp.record.checkpoint_id)

    assert rows[0].pin_state == PIN_STATE_RELEASED
    # The verification pin never upgraded the tier, so there is none to undo.
    assert rows[0].restore_tier == "restorable-unpinned"
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0
    assert versioner.pin_checkpoint(cp.record.checkpoint_id)[0].pin_state == (
        PIN_STATE_RELEASED
    )


def test_release_checkpoint_leaves_non_held_rows_alone(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """Only ``held`` rows are released. A ``pin=False`` capture's ``unpinned``
    row must survive untouched — walking it to the terminal ``released`` would
    strand a checkpoint that was never pinned (``pin_checkpoint`` skips
    ``released``) and would underflow the refcount."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("deferred", pin=False)

    rows = versioner.release_checkpoint(cp.record.checkpoint_id)

    assert rows[0].pin_state == PIN_STATE_UNPINNED
    assert rows[0].restore_tier == "restorable"
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0
    # Still pinnable: the deferred capture can be completed afterwards.
    assert versioner.pin_checkpoint(cp.record.checkpoint_id)[0].pin_state == (
        PIN_STATE_HELD
    )
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True


def test_pin_checkpoint_redrive_is_idempotent(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """pin=False leaves the documented (restorable, unpinned) claimed-but-not-
    backed pair; pin_checkpoint completes it; a second re-drive changes
    nothing (held members are skipped — the refcount cannot double-count)."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")

    cp = versioner.checkpoint("deferred", pin=False)
    (member,) = cp.members
    assert member.restore_tier == "restorable" and member.pin_state == PIN_STATE_UNPINNED
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is False

    rows = versioner.pin_checkpoint(cp.record.checkpoint_id)
    assert rows[0].pin_state == PIN_STATE_HELD
    assert obj.legal_hold_status("cfg.json", version_id=put["VersionId"]) is True
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 1

    rows = versioner.pin_checkpoint(cp.record.checkpoint_id)  # re-drive: no-op
    assert rows[0].pin_state == PIN_STATE_HELD
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 1

    with pytest.raises(CheckpointUnknown):
        versioner.pin_checkpoint("nope")


def test_capture_only_service_fails_fast_when_pins_needed(
    service: CoordinatorService,
) -> None:
    """The fold-in's fail-fast gate: a capture-only seam cannot record a pin
    outcome, so pin=True with pinnable members refuses BEFORE any capture
    read — never a silently unpinned 'restorable' manifest. pin=False is the
    explicit capture-only opt-out and still persists the honest pair."""
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    capture_only = _CaptureOnlyService(service)
    versioner = _versioner(capture_only)
    versioner.add_object_member(obj, "cfg.json")

    with pytest.raises(TypeError):
        versioner.checkpoint("needs-pins")

    cp = versioner.checkpoint("explicitly-unpinned", pin=False)
    (member,) = cp.members
    assert member.restore_tier == "restorable" and member.pin_state == PIN_STATE_UNPINNED

    # No pinnable work (a file member with NO resolver): the capture-only seam
    # is fine even at pin=True — nothing above restorable-unpinned is claimed.
    files = _ScriptedFileSource()
    files.program("a.txt", (b"a", 1))
    plain = _versioner(capture_only)
    plain.add_file_member(files, "a.txt")
    cp2 = plain.checkpoint("no-pin-work")
    assert cp2.members[0].restore_tier == "restorable-unpinned"
    assert cp2.members[0].pin_state == PIN_STATE_UNPINNED


def test_file_member_pin_verified_held_and_survives_registry_reopen(
    tmp_path: Path,
) -> None:
    """The file verification pin is durable state: held on the sqlite arm,
    read back HELD (refcount included) after a full registry reopen — the
    coordinator-restart survival the plan's Unit 6 scenario names."""
    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry

    db_path = tmp_path / "state.db"
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"plan text")

    reg = SqliteArtifactRegistry(db_path)
    try:
        service = CoordinatorService(reg)
        versioner = _versioner(service, resolver=resolver)
        versioner.add_file_member(files, "notes/plan.md")
        cp = versioner.checkpoint("durable-pin")
        (member,) = cp.members
        assert member.pin_state == PIN_STATE_HELD
        # v1 honesty: the verification pin never upgrades the tier — the
        # coordinator offers no per-version hold (the R13 disclosure).
        assert member.restore_tier == "restorable-unpinned"
        checkpoint_id = cp.record.checkpoint_id
    finally:
        reg.close()

    reopened = SqliteArtifactRegistry(db_path)
    try:
        (stored,) = reopened.get_checkpoint_members(checkpoint_id)
        assert stored.pin_state == PIN_STATE_HELD
        assert stored.restore_tier == "restorable-unpinned"
        record = reopened.get_checkpoint(checkpoint_id)
        assert record is not None and record.pin_refcount == 1
    finally:
        reopened.close()


def test_file_member_retention_gap_downgrades_loudly(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """Retention off / the version not retained → KeyError from the resolver
    seam → the captured state is ALREADY unreachable: the loud downgrade to
    forward_only; the restore then honestly skips and says so."""
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()  # nothing retained: the retention-off shape

    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    cp = versioner.checkpoint("gap")

    (member,) = cp.members
    assert member.pin_state == PIN_STATE_UNAVAILABLE
    assert member.restore_tier == "forward_only"
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0

    report = versioner.restore(cp.record.checkpoint_id)
    assert (
        report.members_by_path["notes/plan.md"].outcome
        == RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED
    )


def test_file_member_retained_bytes_mismatch_downgrades_loudly(
    service: CoordinatorService,
) -> None:
    """Retained bytes that no longer hash to the captured fingerprint are NOT
    the captured state — restoring them would be a silent wrong-content
    restore, so the pin refuses and downgrades loudly instead."""
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    resolver = _FakeResolver()
    resolver.keep("notes/plan.md", 7, b"NOT the captured bytes")

    versioner = _versioner(service, resolver=resolver)
    versioner.add_file_member(files, "notes/plan.md")
    cp = versioner.checkpoint("mismatch")

    (member,) = cp.members
    assert member.pin_state == PIN_STATE_UNAVAILABLE
    assert member.restore_tier == "forward_only"


def test_file_member_without_resolver_left_unpinned(
    service: CoordinatorService,
) -> None:
    """No resolver → no verification is possible and nothing above
    restorable-unpinned was ever claimed: the member is left untouched
    (unpinned), never downgraded and never falsely held."""
    files = _FakeFileStore()
    files.put("notes/plan.md", b"plan text", 7)
    versioner = _versioner(service)  # no resolver
    versioner.add_file_member(files, "notes/plan.md")
    cp = versioner.checkpoint("unverified")
    (member,) = cp.members
    assert member.pin_state == PIN_STATE_UNPINNED
    assert member.restore_tier == "restorable-unpinned"


def test_pin_vanished_version_downgrades_loudly(
    service: CoordinatorService,
) -> None:
    """The captured version is gone by pin time (raced version-targeted
    delete): nothing left to hold — the loud downgrade, and restore reports
    target_lost for it."""
    client, obj = _s3()
    put = client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("race", pin=False)  # capture first, no hold yet
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v2")
    client.delete_object(Bucket="demo", Key="cfg.json", VersionId=put["VersionId"])

    rows = versioner.pin_checkpoint(cp.record.checkpoint_id)

    assert rows[0].pin_state == PIN_STATE_UNAVAILABLE
    assert rows[0].restore_tier == "restorable-unpinned"
    report = versioner.restore(cp.record.checkpoint_id)
    assert (
        report.members_by_path["s3://cfg.json"].outcome == RESTORE_OUTCOME_TARGET_LOST
    )


def test_pin_checkpoint_missing_binding_preflight_fails_before_any_write(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    client, obj = _s3()
    client.put_object(Bucket="demo", Key="cfg.json", Body=b"v1")
    versioner = _versioner(service)
    versioner.add_object_member(obj, "cfg.json")
    cp = versioner.checkpoint("orphan", pin=False)

    fresh = _versioner(service)  # no declared members: cannot reach the bucket
    with pytest.raises(ValueError):
        fresh.pin_checkpoint(cp.record.checkpoint_id)
    # Nothing was written: the member is still the unpinned capture shape.
    (stored,) = registry.get_checkpoint_members(cp.record.checkpoint_id)
    assert stored.pin_state == PIN_STATE_UNPINNED
    record = registry.get_checkpoint(cp.record.checkpoint_id)
    assert record is not None and record.pin_refcount == 0


def test_service_pin_vocabulary_fails_closed(
    registry: ArtifactRegistry, service: CoordinatorService
) -> None:
    """The service refuses unvetted pin/tier strings BEFORE any write — the
    (restore_tier, pin_state) pair is the honesty surface and must stay
    readable by identity."""
    files = _ScriptedFileSource()
    files.program("a.txt", (b"a", 1))
    versioner = _versioner(service)
    versioner.add_file_member(files, "a.txt")
    cp = versioner.checkpoint("vocab")
    checkpoint_id = cp.record.checkpoint_id

    with pytest.raises(ValueError):
        service.set_workspace_checkpoint_member_pin(
            checkpoint_id, "a.txt", pin_state="super-pinned"
        )
    with pytest.raises(ValueError):
        service.set_workspace_checkpoint_member_pin(
            checkpoint_id, "a.txt", pin_state=PIN_STATE_HELD, restore_tier="better"
        )
    with pytest.raises(ValueError):  # refcount never below zero (fail-closed)
        service.adjust_workspace_checkpoint_pin_refcount(checkpoint_id, -1)
    (stored,) = registry.get_checkpoint_members(checkpoint_id)
    assert stored.pin_state == PIN_STATE_UNPINNED  # nothing landed
