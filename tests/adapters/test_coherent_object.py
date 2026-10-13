# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Driver-free unit tests for the CoherentObject S3 native-CAS binding.

The unit tests inject a FAKE S3 client (``FakeS3Client``) that scripts
``get_object`` / ``put_object`` and raises botocore-``ClientError``-shaped stubs
— no boto3, no real S3. The ``real_substrate`` integration tests below document
the same guarantees against a real S3 / S3-compatible endpoint; they are gated on
a bucket env var so they never run (or error) in a driver-free environment.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from ccs.adapters.coherent_object import (
    CREATE_IF_ABSENT,
    CoherentObject,
    ReconcileVerdict,
    S3PutOutcome,
    classify_put_exception,
    conditional_write_bucket_policy,
    least_privilege_iam_policy,
    s3_policy_docs,
)
from ccs.adapters.substrate import (
    CasConflict,
    CasUnknown,
    CasWritten,
    CoherenceSubstrate,
)
from ccs.core.exceptions import CoherenceError
from ccs.core.substrate import Tier


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _md5_etag(data: bytes) -> str:
    """A quoted MD5 ETag, mimicking a general-purpose bucket's plaintext ETag."""
    return '"' + hashlib.md5(data).hexdigest() + '"'


# --- botocore-shaped stubs (no boto3) ------------------------------------------


class _FakeClientError(Exception):
    """A botocore ``ClientError``-shaped stub: carries ``.response['Error']['Code']``."""

    def __init__(self, code: str, message: str = "") -> None:
        self.response = {"Error": {"Code": code, "Message": message}}
        super().__init__(f"{code}: {message}")


class _FakeConnectionError(Exception):
    """A botocore connection-error-shaped stub: NO ``.response`` (ambiguous transport)."""


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class FakeS3Client:
    """A minimal in-memory S3 client honoring If-Match / If-None-Match.

    ``objects`` maps key -> (bytes, etag). Set ``put_error`` / ``get_error`` to
    inject a scripted exception on the NEXT put / get (consumed once). Set
    ``next_etag`` to force the ETag the next put mints (used for opaque
    multipart/SSE-shaped ETags).
    """

    def __init__(self, objects: dict[str, tuple[bytes, str]] | None = None) -> None:
        self.objects: dict[str, tuple[bytes, str]] = dict(objects or {})
        self.put_calls: list[dict[str, object]] = []
        self.get_calls: list[dict[str, object]] = []
        self.put_error: Exception | None = None
        self.get_error: Exception | None = None
        self.next_etag: str | None = None

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, object]:  # noqa: N803 (boto3 kwargs)
        self.get_calls.append({"Bucket": Bucket, "Key": Key})
        if self.get_error is not None:
            err, self.get_error = self.get_error, None
            raise err
        if Key not in self.objects:
            raise _FakeClientError("NoSuchKey", "missing")
        data, etag = self.objects[Key]
        return {"Body": _FakeBody(data), "ETag": etag}

    def put_object(  # noqa: N803 (boto3 kwargs)
        self,
        *,
        Bucket: str,
        Key: str,
        Body: bytes,
        IfMatch: str | None = None,
        IfNoneMatch: str | None = None,
    ) -> dict[str, object]:
        self.put_calls.append(
            {"Bucket": Bucket, "Key": Key, "Body": Body, "IfMatch": IfMatch, "IfNoneMatch": IfNoneMatch}
        )
        if self.put_error is not None:
            err, self.put_error = self.put_error, None
            raise err
        exists = Key in self.objects
        if IfNoneMatch == "*" and exists:
            raise _FakeClientError("PreconditionFailed", "object exists")
        if IfMatch is not None:
            if not exists:
                raise _FakeClientError("NoSuchKey", "raced delete")
            if self.objects[Key][1] != IfMatch:
                raise _FakeClientError("PreconditionFailed", "etag moved")
        etag = self.next_etag or _md5_etag(Body)
        self.next_etag = None
        self.objects[Key] = (Body, etag)
        return {"ETag": etag}


def _make(objects: dict[str, tuple[bytes, str]] | None = None) -> tuple[CoherentObject, FakeS3Client]:
    client = FakeS3Client(objects)
    return CoherentObject("test-bucket", client=client), client


# --- module import / descriptor / conformance ----------------------------------


def test_module_imports_without_boto3() -> None:
    # boto3 is not installed in this environment; the module (and an injected-client
    # construction) must work regardless — the driver import is deferred.
    import importlib.util

    assert importlib.util.find_spec("boto3") is None
    obj, _ = _make()
    assert isinstance(obj, CoherentObject)


def test_descriptor_is_native_cas_over_object_etag() -> None:
    obj, _ = _make()
    desc = obj.descriptor
    assert desc.tier is Tier.NATIVE_CAS
    assert desc.version_source == "object ETag"
    # Guarantee wording is derived from the tier (never hand-written per binding).
    assert "version-CAS" in desc.guarantee_text


def test_satisfies_coherence_substrate_protocol_structurally() -> None:
    obj, _ = _make()
    # runtime_checkable structural conformance — presence of descriptor/read/cas_write.
    assert isinstance(obj, CoherenceSubstrate)


def test_coordinator_content_is_none_and_flag_false() -> None:
    obj, _ = _make()
    # content=None exposure: the mixin reads BOTH the method (what to pass) and the
    # class flag (the honest signal) — never thread bytes coordinator-side.
    assert obj.coordinator_commit_content() is None
    assert CoherentObject.SENDS_CONTENT_TO_COORDINATOR is False


# --- read ----------------------------------------------------------------------


def test_read_returns_bytes_and_etag_from_one_response() -> None:
    obj, client = _make({"k": (b"hello", '"etag-1"')})
    data, token = obj.read("k")
    assert data == b"hello"
    assert token == '"etag-1"'
    # ONE get_object — never Head-then-Get for the (bytes, token) pair.
    assert len(client.get_calls) == 1


def test_read_absent_object_raises_keyerror() -> None:
    obj, _ = _make()
    with pytest.raises(KeyError):
        obj.read("missing")


# --- cas_write: happy update ---------------------------------------------------


def test_cas_write_update_wins_with_response_etag() -> None:
    obj, client = _make({"k": (b"v1", '"old"')})
    result = obj.cas_write("k", expected_token='"old"', new_bytes=b"v2")
    assert isinstance(result, CasWritten)
    # The token is the ETag from the put RESPONSE, never computed client-side.
    assert result.token == _md5_etag(b"v2")
    assert client.put_calls[-1]["IfMatch"] == '"old"'


def test_cas_write_update_is_single_request_put_not_multipart() -> None:
    obj, client = _make({"k": (b"v1", '"old"')})
    obj.cas_write("k", expected_token='"old"', new_bytes=b"v2")
    # Structural pin: exactly one put_object, no transfer-manager / multipart calls.
    assert len(client.put_calls) == 1
    assert not hasattr(client, "create_multipart_upload")


# --- cas_write: create (If-None-Match) -----------------------------------------


def test_cas_write_create_on_absent_succeeds_then_second_conflicts() -> None:
    obj, client = _make()
    first = obj.cas_write("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"seed")
    assert isinstance(first, CasWritten)
    assert client.put_calls[-1]["IfNoneMatch"] == "*"
    # A second create → 412 → CasConflict (the object already exists).
    second = obj.cas_write("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"again")
    assert isinstance(second, CasConflict)


# --- cas_write: typed classification (412 / 409 / 404 / transport) -------------


def test_cas_write_412_precondition_failed_is_conflict() -> None:
    obj, client = _make({"k": (b"v1", '"current"')})
    # Stale comparand: the object is at "current", we present "stale".
    result = obj.cas_write("k", expected_token='"stale"', new_bytes=b"v2")
    assert isinstance(result, CasConflict)


def test_stale_token_write_returns_cas_conflict() -> None:
    # The substrate leg returns the typed CasConflict on a stale token; the
    # mapping to the coordinator-versioned CasVersionConflict is the cross-agent
    # layer's job (tested in tests/adapters/test_substrate_cross_agent.py), so the
    # binding no longer carries a private mapping seam.
    obj, _ = _make({"k": (b"v1", '"current"')})
    assert isinstance(obj.cas_write("k", expected_token='"stale"', new_bytes=b"v2"), CasConflict)


def test_cas_write_409_classified_retryable_but_no_write_landed() -> None:
    obj, client = _make({"k": (b"v1", '"cur"')})
    client.put_error = _FakeClientError("ConditionalRequestConflict", "retry")
    result = obj.cas_write("k", expected_token='"cur"', new_bytes=b"v2")
    # Contract 3-way: no write landed → CasConflict...
    assert isinstance(result, CasConflict)
    # ...but the fine-grained classifier marks it RETRYABLE (re-read etag, retry).
    assert classify_put_exception(_FakeClientError("ConditionalRequestConflict")) is S3PutOutcome.RETRYABLE


def test_cas_write_404_raced_delete_is_conflict_not_recreate() -> None:
    obj, client = _make()  # object absent → If-Match put 404s
    result = obj.cas_write("k", expected_token='"gone"', new_bytes=b"v2")
    assert isinstance(result, CasConflict)
    assert classify_put_exception(_FakeClientError("NoSuchKey")) is S3PutOutcome.RACED_DELETE


def test_cas_write_transport_failure_is_unknown() -> None:
    obj, client = _make({"k": (b"v1", '"cur"')})
    client.put_error = _FakeConnectionError("connection reset")
    result = obj.cas_write("k", expected_token='"cur"', new_bytes=b"v2")
    # "may or may not have landed" — never a confirmed loss.
    assert isinstance(result, CasUnknown)


def test_cas_write_non_cas_error_propagates() -> None:
    # A modeled but non-CAS error (AccessDenied) must NOT be swallowed as a CAS
    # outcome — it is a configuration fault, so it propagates.
    obj, client = _make({"k": (b"v1", '"cur"')})
    client.put_error = _FakeClientError("AccessDenied", "no perms")
    with pytest.raises(_FakeClientError):
        obj.cas_write("k", expected_token='"cur"', new_bytes=b"v2")
    assert classify_put_exception(_FakeClientError("AccessDenied")) is None


def test_cas_write_rejects_sentinel_token() -> None:
    # An absent/blank token may never seed an If-Match comparand (the Sentinel rule).
    obj, _ = _make({"k": (b"v1", '"cur"')})
    with pytest.raises(CoherenceError):
        obj.cas_write("k", expected_token="   ", new_bytes=b"v2")


def test_cas_write_never_computes_missing_etag() -> None:
    # A put response without an ETag is unverifiable → fail closed, never mint a token.
    class _NoEtagClient(FakeS3Client):
        def put_object(self, **kwargs: object) -> dict[str, object]:  # type: ignore[override]
            super().put_object(**kwargs)  # type: ignore[arg-type]
            return {}  # response missing ETag

    obj = CoherentObject("b", client=_NoEtagClient({"k": (b"v1", '"cur"')}))
    with pytest.raises(CoherenceError):
        obj.cas_write("k", expected_token='"cur"', new_bytes=b"v2")


# --- the four reconciliation arms ----------------------------------------------


def test_reconcile_404_holds() -> None:
    # (i) absent object / delete-marker → HOLD; never auto re-create, never a match
    # against sha256(b"").
    obj, _ = _make()
    decision = obj.reconcile_after_unknown("k", expected_token='"T_old"', intended_hash=_sha256_hex(b""))
    assert decision.verdict is ReconcileVerdict.HOLD
    assert decision.observed_bytes is None and decision.observed_token is None
    assert decision.bump_fires is False


def test_reconcile_etag_unmoved_re_drives_under_if_match_t_old() -> None:
    # (ii) token unmoved → not landed → RE_DRIVE, only under If-Match=T_old.
    obj, _ = _make({"k": (b"old", '"T_old"')})
    decision = obj.reconcile_after_unknown(
        "k", expected_token='"T_old"', intended_hash=_sha256_hex(b"intended")
    )
    assert decision.verdict is ReconcileVerdict.RE_DRIVE
    assert decision.re_drive_token == '"T_old"'
    assert decision.bump_fires is False


def test_reconcile_moved_and_bytes_match_converges_and_bump_still_fires() -> None:
    # (iii) token moved AND bytes byte-identical → CONVERGE; the coordinator bump
    # STILL fires and the surface says "converged", never "landed".
    intended = b"the-intended-bytes"
    obj, _ = _make({"k": (intended, '"T_new"')})
    decision = obj.reconcile_after_unknown(
        "k", expected_token='"T_old"', intended_hash=_sha256_hex(intended)
    )
    assert decision.verdict is ReconcileVerdict.CONVERGE
    assert decision.bump_fires is True  # the load-bearing invariant
    assert decision.observed_token == '"T_new"'  # adopt the observed ETag as comparand
    assert "converged" in decision.summary
    assert "landed" not in decision.summary  # attribution is NOT claimed


def test_reconcile_moved_and_bytes_differ_conflicts() -> None:
    # (iv) token moved AND bytes differ → typed conflict; never re-drive.
    obj, _ = _make({"k": (b"a-peer-write", '"T_new"')})
    decision = obj.reconcile_after_unknown(
        "k", expected_token='"T_old"', intended_hash=_sha256_hex(b"my-intended")
    )
    assert decision.verdict is ReconcileVerdict.CONFLICT
    assert decision.re_drive_token is None
    assert decision.bump_fires is False


def test_never_converge_wedge_negative_control() -> None:
    # NEGATIVE CONTROL: in the my-write-landed world (token moved + bytes match), a
    # decision that REFUSED to converge would strand the coordinator bump and wedge
    # every peer (ViewWedged, no v1 repair-forward). Assert this impl converges AND
    # signals the bump — never HOLD/CONFLICT here.
    intended = b"landed-write"
    obj, _ = _make({"k": (intended, '"T_new"')})
    decision = obj.reconcile_after_unknown(
        "k", expected_token='"T_old"', intended_hash=_sha256_hex(intended)
    )
    assert decision.verdict is ReconcileVerdict.CONVERGE
    assert decision.bump_fires is True
    assert decision.verdict not in {ReconcileVerdict.HOLD, ReconcileVerdict.CONFLICT}


def test_reconcile_converges_under_opaque_multipart_shaped_etag() -> None:
    # The converge test keys on sha256(bytes) and adopts the ETag as OPAQUE, so an
    # SSE-KMS/multipart-shaped ETag (not a content digest) still converges safely —
    # never fail closed on an opaque ETag format when the bytes match.
    intended = b"converge-me"
    opaque_etag = '"a1b2c3d4-9"'  # multipart-shaped: not an MD5 of the bytes
    obj, _ = _make({"k": (intended, opaque_etag)})
    decision = obj.reconcile_after_unknown(
        "k", expected_token='"T_old"', intended_hash=_sha256_hex(intended)
    )
    assert decision.verdict is ReconcileVerdict.CONVERGE
    assert decision.observed_token == opaque_etag


def test_reconcile_read_transport_failure_propagates() -> None:
    # A transport failure DURING the reconcile read is not a decision — it must
    # propagate (retry the reconcile), never silently become HOLD/CONVERGE.
    obj, client = _make({"k": (b"v", '"T_new"')})
    client.get_error = _FakeConnectionError("reset mid-reconcile")
    with pytest.raises(_FakeConnectionError):
        obj.reconcile_after_unknown("k", expected_token='"T_old"', intended_hash=_sha256_hex(b"v"))


# --- no-op short-circuit -------------------------------------------------------


def test_noop_write_short_circuits_before_the_put() -> None:
    # intended hash == current hash → no put, no coordinator bump (no phantom advance).
    current = b"unchanged"
    obj, client = _make({"k": (current, '"cur"')})
    result = obj.cas_write_if_changed(
        "k", expected_token='"cur"', new_bytes=current, current_hash=_sha256_hex(current)
    )
    assert result is None  # the None signals: skip the coordinator bump too
    assert client.put_calls == []  # the put was never issued


def test_changed_write_falls_through_to_cas_write() -> None:
    obj, client = _make({"k": (b"old", '"cur"')})
    result = obj.cas_write_if_changed(
        "k", expected_token='"cur"', new_bytes=b"new", current_hash=_sha256_hex(b"old")
    )
    assert isinstance(result, CasWritten)
    assert len(client.put_calls) == 1


# --- least-privilege / bucket-policy doc helpers -------------------------------


def test_policy_helpers_return_verified_shape() -> None:
    docs = s3_policy_docs("my-bucket", "prefix/key")
    blob = json.dumps(docs)
    # The verified 2026-07 shape: conditional-write enforcement + multipart
    # exemption + the writer's GetObject grant.
    assert "s3:if-match" in blob
    assert "s3:ObjectCreationOperation" in blob
    assert "s3:GetObject" in blob


def test_writer_iam_denies_delete_and_grants_get_put() -> None:
    policy = least_privilege_iam_policy("b", "k")
    statements = policy["Statement"]
    allow = next(s for s in statements if s["Effect"] == "Allow")
    deny = next(s for s in statements if s["Effect"] == "Deny")
    assert set(allow["Action"]) == {"s3:GetObject", "s3:PutObject"}
    assert "s3:DeleteObject" in deny["Action"]


def test_bucket_policy_denies_unconditional_put() -> None:
    policy = conditional_write_bucket_policy("b")
    stmt = policy["Statement"][0]
    assert stmt["Effect"] == "Deny"
    assert stmt["Condition"]["Null"] == {"s3:if-match": "true"}


# --- integration (real S3 / S3-compatible; deselected by default) --------------
#
# Marked ``real_substrate`` (deselected via ``-m 'not real_substrate'``, added by
# the Unit-5 orchestrator) AND gated on a real bucket env var, so they never run
# — or error — in this driver-free environment. They document the same guarantees
# against a real endpoint; Moto/LocalStack are excluded (they serialize → false
# green). Written to run correctly against a real bucket, not runnable here.

real_substrate = pytest.mark.real_substrate

_REAL_BUCKET = os.environ.get("CCS_REAL_S3_BUCKET")
_needs_real_s3 = pytest.mark.skipif(
    not _REAL_BUCKET, reason="set CCS_REAL_S3_BUCKET to a real (non-Moto) S3 bucket to run"
)


@real_substrate
@_needs_real_s3
def test_real_concurrent_lost_update_one_winner_loser_converges() -> None:
    # Two writers read the same ETag; one wins the If-Match put, the loser 412s →
    # CasConflict; the loser re-reads and converges/re-derives against the winner.
    obj = CoherentObject(_REAL_BUCKET, region=os.environ.get("CCS_REAL_S3_REGION"))
    key = "coherence-it/lost-update"
    obj.cas_write(key, expected_token=CREATE_IF_ABSENT, new_bytes=b"seed")
    _seed_bytes, token = obj.read(key)
    winner = obj.cas_write(key, expected_token=token, new_bytes=b"winner")
    assert isinstance(winner, CasWritten)
    loser = obj.cas_write(key, expected_token=token, new_bytes=b"loser-stale")
    assert isinstance(loser, CasConflict)


@real_substrate
@_needs_real_s3
def test_real_timed_out_put_that_landed_converges_on_moved_bytes_match() -> None:
    # A put whose ack was lost but which DID land: the reconcile read shows the ETag
    # moved and the bytes match intended → CONVERGE, adopting the observed ETag; the
    # coordinator bump still fires (the leg is never stranded). Surface says
    # "converged", never "landed".
    obj = CoherentObject(_REAL_BUCKET, region=os.environ.get("CCS_REAL_S3_REGION"))
    key = "coherence-it/timeout-landed"
    obj.cas_write(key, expected_token=CREATE_IF_ABSENT, new_bytes=b"seed")
    _bytes, t_old = obj.read(key)
    intended = b"landed-despite-timeout"
    obj.cas_write(key, expected_token=t_old, new_bytes=intended)  # simulate: ack lost
    decision = obj.reconcile_after_unknown(key, expected_token=t_old, intended_hash=_sha256_hex(intended))
    assert decision.verdict is ReconcileVerdict.CONVERGE
    assert decision.bump_fires is True
    assert "landed" not in decision.summary


@real_substrate
@_needs_real_s3
def test_real_raced_delete_404_holds_caller_recreates() -> None:
    # A foreign DELETE during UNKNOWN → the If-Match re-read 404s → HOLD; re-create
    # is the CALLER's decision via CREATE_IF_ABSENT after reacquire, never automatic.
    obj = CoherentObject(_REAL_BUCKET, region=os.environ.get("CCS_REAL_S3_REGION"))
    key = "coherence-it/raced-delete"
    decision = obj.reconcile_after_unknown(key, expected_token='"stale"', intended_hash=_sha256_hex(b"x"))
    assert decision.verdict is ReconcileVerdict.HOLD


# =============================================================================
# Workspace-Versioning axes (Unit 1, 2026-08-08) — STRICTLY ADDITIVE section.
# Everything above this line is the released v0.13.0 contract, untouched; the
# scenarios below cover only the NEW surface (versionId pointer, pinned reads,
# delete markers, legal hold, the 409 budget, client refresh) and run against
# the deterministic in-repo S3-semantics fake (ccs.testing.s3_local).
# =============================================================================

import ccs.adapters.coherent_object as coherent_object_module  # noqa: E402 (additive section)
from ccs.adapters.coherent_object import (  # noqa: E402 (additive section)
    LEGAL_HOLD_UNAVAILABLE_REASON,
    MAX_RETRYABLE_PUT_ATTEMPTS,
    NULL_VERSION_ID,
    VERSION_POINTER_UNCONFIRMED_REASON,
    ConditionalPutRetriesExhausted,
    LegalHoldUnavailable,
    ObjectDeletion,
    VersionedCasWritten,
    VersionPointerUnconfirmed,
)
from ccs.core.exceptions import CasRetriesExhausted  # noqa: E402 (additive section)
from ccs.core.substrate import (  # noqa: E402 (additive section)
    ArbitrationTier,
    CapabilityDescriptor,
    RestoreTier,
    derive_restore_tier,
)
from ccs.testing.s3_local import LocalS3Client  # noqa: E402 (additive section)


def _versioned(
    bucket: str = "wv", *, object_lock: bool = False
) -> tuple[CoherentObject, LocalS3Client]:
    client = LocalS3Client()
    client.create_bucket(bucket, versioned=True, object_lock=object_lock)
    return CoherentObject(bucket, client=client), client


def _unversioned(bucket: str = "flat") -> tuple[CoherentObject, LocalS3Client]:
    client = LocalS3Client()
    client.create_bucket(bucket, versioned=False)
    return CoherentObject(bucket, client=client), client


# --- the client-injection seam (the fake IS a boto3-shaped client) --------------


def test_shipped_read_and_cas_write_run_against_the_s3_fake() -> None:
    # The RELEASED surface runs against ccs.testing.s3_local unchanged — the
    # injection seam Unit 3's capture engine plugs the fake into.
    obj, _ = _versioned()
    first = obj.cas_write("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"seed")
    assert isinstance(first, CasWritten)
    data, token = obj.read("k")
    assert (data, token) == (b"seed", first.token)
    second = obj.cas_write("k", expected_token=token, new_bytes=b"v2")
    assert isinstance(second, CasWritten)
    assert isinstance(obj.cas_write("k", expected_token=token, new_bytes=b"stale"), CasConflict)


# --- the capture triple ---------------------------------------------------------


def test_read_versioned_returns_triple_from_one_get() -> None:
    obj, client = _versioned()
    win = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v1")
    assert isinstance(win, VersionedCasWritten)
    client.get_calls.clear()
    capture = obj.read_versioned("k")
    assert capture.data == b"v1"
    assert capture.etag == _md5_etag(b"v1")
    assert capture.version_id == win.version_id
    # ONE get_object for the WHOLE triple — the split-comparand rule, extended.
    assert len(client.get_calls) == 1


def test_pin_and_version_pinned_reread_roundtrips_old_bytes() -> None:
    obj, _ = _versioned()
    old = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"old")
    assert isinstance(old, VersionedCasWritten)
    new = obj.cas_write_versioned("k", expected_token=old.etag, new_bytes=b"new")
    assert isinstance(new, VersionedCasWritten)
    pinned = obj.read_pinned("k", version_id=old.version_id)
    assert pinned.data == b"old"
    assert pinned.etag == old.etag
    assert pinned.version_id == old.version_id
    head = obj.read_versioned("k")
    assert head.data == b"new"
    assert head.version_id == new.version_id


def test_restore_leg_if_match_win_mints_new_version_id() -> None:
    # Restore-as-forward-commit: the winning If-Match leg mints a NEW version
    # carrying the old bytes; both etag and version_id come from the response.
    obj, _ = _versioned()
    seed = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"seed")
    assert isinstance(seed, VersionedCasWritten)
    current = obj.read_versioned("k")
    win = obj.cas_write_versioned(
        "k", expected_token=current.etag, new_bytes=b"restored-old-bytes"
    )
    assert isinstance(win, VersionedCasWritten)
    assert win.version_id != seed.version_id
    assert win.etag == _md5_etag(b"restored-old-bytes")


# --- the "null"/absent sentinel (UNCONFIRMED → typed HOLD) ----------------------


def test_unversioned_bucket_capture_is_typed_refusal_never_restorable() -> None:
    obj, _ = _unversioned()
    assert isinstance(
        obj.cas_write("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v"), CasWritten
    )
    with pytest.raises(VersionPointerUnconfirmed) as exc_info:
        obj.read_versioned("k")
    # Typed reason matched by IDENTITY, never a message substring.
    assert exc_info.value.reason is VERSION_POINTER_UNCONFIRMED_REASON
    assert exc_info.value.version_id == NULL_VERSION_ID
    assert exc_info.value.etag is None  # the read path never sets the write-leg attr


def test_unversioned_bucket_write_leg_refusal_still_reports_landed_etag() -> None:
    # The write IS durable on an unversioned bucket; only the pointer is
    # unconfirmed — the typed HOLD carries the landed etag so the caller is
    # never blind about a durable write.
    obj, _ = _unversioned()
    with pytest.raises(VersionPointerUnconfirmed) as exc_info:
        obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v")
    assert exc_info.value.etag == _md5_etag(b"v")
    assert exc_info.value.version_id == NULL_VERSION_ID
    data, _token = obj.read("k")
    assert data == b"v"  # the write landed


def test_missing_version_id_key_is_the_same_typed_refusal() -> None:
    # A response with NO VersionId key at all classifies exactly like "null".
    obj = CoherentObject("b", client=FakeS3Client({"k": (b"v", '"e"')}))
    with pytest.raises(VersionPointerUnconfirmed) as exc_info:
        obj.read_versioned("k")
    assert exc_info.value.reason is VERSION_POINTER_UNCONFIRMED_REASON
    assert exc_info.value.version_id is None  # absent, not the "null" literal


def test_null_sentinel_never_seeds_a_pinned_read_or_hold_leg() -> None:
    obj, client = _versioned()
    for sentinel in (NULL_VERSION_ID, "", "   "):
        with pytest.raises(VersionPointerUnconfirmed):
            obj.read_pinned("k", version_id=sentinel)
        with pytest.raises(VersionPointerUnconfirmed):
            obj.set_legal_hold("k", version_id=sentinel)
        with pytest.raises(VersionPointerUnconfirmed):
            obj.legal_hold_status("k", version_id=sentinel)
    # Refused BEFORE any request left the client (the Sentinel rule).
    assert client.get_calls == []


def test_typed_hold_exceptions_carry_class_level_defaults() -> None:
    # A generic handler reads the attrs uniformly on ANY instance — the
    # class-level defaults hold when a raise site sets nothing.
    bare = VersionPointerUnconfirmed("x")
    assert bare.version_id is None
    assert bare.etag is None
    assert bare.reason is VERSION_POINTER_UNCONFIRMED_REASON
    held = LegalHoldUnavailable("x")
    assert held.version_id is None
    assert held.reason is LEGAL_HOLD_UNAVAILABLE_REASON


def test_expired_pin_read_pinned_raises_keyerror() -> None:
    # The version vanished (lifecycle/foreign permanent delete) → the absent
    # shape (KeyError), which the workspace layer maps to target-lost.
    obj, client = _versioned()
    win = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v1")
    assert isinstance(win, VersionedCasWritten)
    obj.cas_write_versioned("k", expected_token=win.etag, new_bytes=b"v2")
    client.delete_object(Bucket="wv", Key="k", VersionId=win.version_id)
    with pytest.raises(KeyError):
        obj.read_pinned("k", version_id=win.version_id)


# --- typed classification on the versioned surface (412 / 404 / transport) ------


def test_cas_write_versioned_412_is_typed_conflict() -> None:
    obj, _ = _versioned()
    obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v1")
    result = obj.cas_write_versioned("k", expected_token='"stale"', new_bytes=b"x")
    assert isinstance(result, CasConflict)


def test_cas_write_versioned_404_raced_delete_is_conflict_not_recreate() -> None:
    obj, _ = _versioned()  # the key is absent → the If-Match put 404s
    result = obj.cas_write_versioned("k", expected_token='"gone"', new_bytes=b"x")
    assert isinstance(result, CasConflict)


def test_cas_write_versioned_delete_marker_current_if_match_conflicts() -> None:
    # A delete IS an update: an If-Match leg racing a delete marker loses typed;
    # re-create stays a CALLER decision, never automatic.
    obj, _ = _versioned()
    win = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v1")
    assert isinstance(win, VersionedCasWritten)
    obj.delete("k")
    result = obj.cas_write_versioned("k", expected_token=win.etag, new_bytes=b"x")
    assert isinstance(result, CasConflict)


def test_cas_write_versioned_transport_failure_is_unknown() -> None:
    client = FakeS3Client({"k": (b"v1", '"cur"')})
    client.put_error = _FakeConnectionError("reset")
    obj = CoherentObject("b", client=client)
    result = obj.cas_write_versioned("k", expected_token='"cur"', new_bytes=b"x")
    assert isinstance(result, CasUnknown)


# --- the bounded 409 budget -----------------------------------------------------


class _Transient409Client(FakeS3Client):
    """409s the first ``transients`` puts, then delegates (minting a VersionId)."""

    def __init__(
        self, objects: dict[str, tuple[bytes, str]] | None = None, *, transients: int
    ) -> None:
        super().__init__(objects)
        self._transients = transients
        self.attempts = 0
        self.if_match_seen: list[str | None] = []

    def put_object(self, **kwargs: object) -> dict[str, object]:  # type: ignore[override]
        self.attempts += 1
        self.if_match_seen.append(kwargs.get("IfMatch"))  # type: ignore[arg-type]
        if self._transients > 0:
            self._transients -= 1
            raise _FakeClientError("ConditionalRequestConflict", "transient")
        resp = super().put_object(**kwargs)  # type: ignore[arg-type]
        resp["VersionId"] = f"t{self.attempts:03d}"
        return resp


def test_cas_write_versioned_409_retries_in_budget_under_same_comparand() -> None:
    client = _Transient409Client({"k": (b"v1", '"cur"')}, transients=2)
    obj = CoherentObject("b", client=client)
    win = obj.cas_write_versioned("k", expected_token='"cur"', new_bytes=b"v2")
    assert isinstance(win, VersionedCasWritten)
    assert client.attempts == 3  # 2 transients + the winning attempt
    # Every attempt carried the SAME If-Match comparand — a 409 retry never
    # re-reads a fresh token (a moved token must surface as a 412 conflict).
    assert client.if_match_seen == ['"cur"', '"cur"', '"cur"']


def test_cas_write_versioned_409_budget_exhausts_typed_terminal() -> None:
    client = _Transient409Client({"k": (b"v1", '"cur"')}, transients=10_000)
    obj = CoherentObject("b", client=client)
    with pytest.raises(ConditionalPutRetriesExhausted) as exc_info:
        obj.cas_write_versioned("k", expected_token='"cur"', new_bytes=b"v2")
    # The shipped budget family: existing CasRetriesExhausted handlers (and the
    # deny mapper, via the inherited reason) classify it without change.
    assert isinstance(exc_info.value, CasRetriesExhausted)
    assert exc_info.value.attempts == MAX_RETRYABLE_PUT_ATTEMPTS + 1
    # It bypasses the parent __init__, so the reason comes from the class default.
    assert exc_info.value.last_conflict_reason is None
    assert client.attempts == MAX_RETRYABLE_PUT_ATTEMPTS + 1  # initial + budget, never spins


# --- byte-identical concurrency converges ---------------------------------------


def test_byte_identical_concurrent_write_converges_not_re_driven() -> None:
    # A byte-identical peer won the token race while MY ack was lost: the
    # reconcile read sees the token moved AND the bytes match the intent →
    # CONVERGE (bump still fires), never a re-drive of the stale comparand.
    obj, client = _versioned()
    seed = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"seed")
    assert isinstance(seed, VersionedCasWritten)
    peer = CoherentObject("wv", client=client)
    peer_win = peer.cas_write_versioned("k", expected_token=seed.etag, new_bytes=b"same-bytes")
    assert isinstance(peer_win, VersionedCasWritten)
    decision = obj.reconcile_after_unknown(
        "k", expected_token=seed.etag, intended_hash=_sha256_hex(b"same-bytes")
    )
    assert decision.verdict is ReconcileVerdict.CONVERGE
    assert decision.bump_fires is True
    assert decision.observed_token == peer_win.etag


# --- delete-marker-aware delete -------------------------------------------------


def test_delete_versioned_mints_marker_and_pinned_history_survives() -> None:
    obj, _ = _versioned()
    win = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v1")
    assert isinstance(win, VersionedCasWritten)
    deletion = obj.delete("k")
    assert deletion.delete_marker is True
    assert isinstance(deletion.version_id, str)
    assert deletion.version_id != win.version_id  # the marker is its OWN version
    with pytest.raises(KeyError):
        obj.read("k")  # the latest is now the marker
    # The restore-read guarantee: pinned history serves THROUGH the marker.
    assert obj.read_pinned("k", version_id=win.version_id).data == b"v1"


def test_delete_unversioned_is_permanent_and_says_so() -> None:
    obj, _ = _unversioned()
    obj.cas_write("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v1")
    deletion = obj.delete("k")
    assert deletion == ObjectDeletion(delete_marker=False, version_id=None)
    with pytest.raises(KeyError):
        obj.read("k")


# --- legal hold (the pin leg) ---------------------------------------------------


def test_legal_hold_set_status_release_roundtrip() -> None:
    obj, _ = _versioned(object_lock=True)
    win = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"pin-me")
    assert isinstance(win, VersionedCasWritten)
    assert obj.legal_hold_status("k", version_id=win.version_id) is False
    obj.set_legal_hold("k", version_id=win.version_id)
    assert obj.legal_hold_status("k", version_id=win.version_id) is True
    obj.release_legal_hold("k", version_id=win.version_id)
    assert obj.legal_hold_status("k", version_id=win.version_id) is False


def test_legal_hold_blocks_simulated_lifecycle_delete() -> None:
    # The pin's teeth: a held noncurrent version survives the lifecycle pass and
    # still serves the pinned restore read; releasing it makes the SAME pass
    # reclaim it — "restorable" means restorable, or the tier says otherwise.
    obj, client = _versioned(object_lock=True)
    old = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"pinned")
    assert isinstance(old, VersionedCasWritten)
    obj.cas_write_versioned("k", expected_token=old.etag, new_bytes=b"newer")
    obj.set_legal_hold("k", version_id=old.version_id)
    expired, held = client.lifecycle_expire_noncurrent("wv", "k")
    assert held == [old.version_id]
    assert expired == []
    assert obj.read_pinned("k", version_id=old.version_id).data == b"pinned"
    obj.release_legal_hold("k", version_id=old.version_id)
    expired, held = client.lifecycle_expire_noncurrent("wv", "k")
    assert expired == [old.version_id]
    assert held == []


def test_legal_hold_unavailable_without_object_lock_is_typed() -> None:
    obj, _ = _versioned()  # versioned but NO Object Lock configuration
    win = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v")
    assert isinstance(win, VersionedCasWritten)
    with pytest.raises(LegalHoldUnavailable) as exc_info:
        obj.set_legal_hold("k", version_id=win.version_id)
    assert exc_info.value.reason is LEGAL_HOLD_UNAVAILABLE_REASON
    assert exc_info.value.version_id == win.version_id


def test_legal_hold_on_vanished_version_is_keyerror() -> None:
    obj, _ = _versioned(object_lock=True)
    win = obj.cas_write_versioned("k", expected_token=CREATE_IF_ABSENT, new_bytes=b"v")
    assert isinstance(win, VersionedCasWritten)
    with pytest.raises(KeyError):
        obj.set_legal_hold("k", version_id="v999999")


# --- operational-error client refresh -------------------------------------------


def test_operational_error_discards_self_owned_client_and_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[FakeS3Client] = []

    def _factory(region: str | None, endpoint_url: str | None) -> FakeS3Client:
        client = FakeS3Client({"k": (b"v1", '"cur"')})
        built.append(client)
        return client

    monkeypatch.setattr(coherent_object_module, "_require_boto3_client", _factory)
    obj = CoherentObject("b", region="us-east-1")
    assert len(built) == 1
    built[0].put_error = _FakeConnectionError("reset")
    result = obj.cas_write("k", expected_token='"cur"', new_bytes=b"v2")
    assert isinstance(result, CasUnknown)
    # The poisoned SELF-owned client was discarded; the next call rebuilds a
    # fresh one from the recorded region/endpoint — the documented "reconcile by
    # re-reading" recovery never re-hits a dead socket.
    data, _token = obj.read("k")
    assert data == b"v1"
    assert len(built) == 2
    assert built[1].get_calls and not built[0].get_calls


def test_injected_client_is_never_discarded() -> None:
    client = FakeS3Client({"k": (b"v1", '"cur"')})
    obj = CoherentObject("b", client=client)
    client.put_error = _FakeConnectionError("reset")
    assert isinstance(obj.cas_write("k", expected_token='"cur"', new_bytes=b"v2"), CasUnknown)
    # The caller-owned client is untouched: the SAME instance serves the re-read.
    data, _token = obj.read("k")
    assert data == b"v1"
    assert len(client.get_calls) == 1


# --- the capability axes --------------------------------------------------------


def test_descriptor_declares_native_cas_arbitration_and_no_static_restore_claim() -> None:
    obj, _ = _make()
    desc = obj.descriptor
    assert desc.arbitration_tier is ArbitrationTier.NATIVE_CAS
    # Restore tiering is BUCKET-dependent (versioning + Object Lock): derived at
    # capture via derive_restore_tier, never claimed statically.
    assert desc.restore_tier is None
    assert desc.versioned_pinnable is False


def test_derive_restore_tier_fail_closed_floor() -> None:
    assert derive_restore_tier(versioned=False, pinnable=False) is RestoreTier.FORWARD_ONLY
    assert derive_restore_tier(versioned=False, pinnable=True) is RestoreTier.FORWARD_ONLY
    assert (
        derive_restore_tier(versioned=True, pinnable=False) is RestoreTier.RESTORABLE_UNPINNED
    )
    assert derive_restore_tier(versioned=True, pinnable=True) is RestoreTier.RESTORABLE


def test_descriptor_refuses_restorable_overclaim_without_pinnable() -> None:
    with pytest.raises(ValueError):
        CapabilityDescriptor(
            tier=Tier.NATIVE_CAS,
            version_source="object ETag",
            restore_tier=RestoreTier.RESTORABLE,  # versioned_pinnable defaults False
        )


# --- integration (real S3; deselected by default — the fidelity oracle) ---------


def _real_create_or_update(obj: CoherentObject, key: str, payload: bytes) -> VersionedCasWritten:
    """Create-or-update against a real bucket a prior run may have seeded."""
    outcome = obj.cas_write_versioned(key, expected_token=CREATE_IF_ABSENT, new_bytes=payload)
    if isinstance(outcome, VersionedCasWritten):
        return outcome
    current = obj.read_versioned(key)
    retried = obj.cas_write_versioned(key, expected_token=current.etag, new_bytes=payload)
    assert isinstance(retried, VersionedCasWritten)
    return retried


@real_substrate
@_needs_real_s3
def test_real_versioned_capture_pin_restore_delete_marker_matrix() -> None:
    # Requires CCS_REAL_S3_BUCKET to have VERSIONING ENABLED.
    obj = CoherentObject(_REAL_BUCKET, region=os.environ.get("CCS_REAL_S3_REGION"))
    key = "coherence-it/wv-matrix"
    seed = _real_create_or_update(obj, key, b"old")
    head = obj.read_versioned(key)
    assert head.version_id == seed.version_id
    win = obj.cas_write_versioned(key, expected_token=head.etag, new_bytes=b"new")
    assert isinstance(win, VersionedCasWritten)
    assert win.version_id != seed.version_id
    assert obj.read_pinned(key, version_id=seed.version_id).data == b"old"
    deletion = obj.delete(key)
    assert deletion.delete_marker is True
    # The restore-read guarantee holds through the marker on the real substrate.
    assert obj.read_pinned(key, version_id=seed.version_id).data == b"old"


@real_substrate
@_needs_real_s3
def test_real_unversioned_bucket_typed_capture_refusal() -> None:
    unversioned = os.environ.get("CCS_REAL_S3_UNVERSIONED_BUCKET")
    if not unversioned:
        pytest.skip(
            "set CCS_REAL_S3_UNVERSIONED_BUCKET to a real bucket with versioning OFF"
        )
    obj = CoherentObject(unversioned, region=os.environ.get("CCS_REAL_S3_REGION"))
    key = "coherence-it/wv-unversioned"
    obj.cas_write(key, expected_token=CREATE_IF_ABSENT, new_bytes=b"seed")
    with pytest.raises(VersionPointerUnconfirmed):
        obj.read_versioned(key)


@real_substrate
@_needs_real_s3
def test_real_legal_hold_pin_set_status_release() -> None:
    lock_bucket = os.environ.get("CCS_REAL_S3_OBJECT_LOCK_BUCKET")
    if not lock_bucket:
        pytest.skip(
            "set CCS_REAL_S3_OBJECT_LOCK_BUCKET to a real Object-Lock-enabled bucket"
        )
    obj = CoherentObject(lock_bucket, region=os.environ.get("CCS_REAL_S3_REGION"))
    key = "coherence-it/wv-legal-hold"
    win = _real_create_or_update(obj, key, b"pin-me")
    obj.set_legal_hold(key, version_id=win.version_id)
    try:
        assert obj.legal_hold_status(key, version_id=win.version_id) is True
    finally:
        # Always release: a held version is undeletable and would strand cleanup.
        obj.release_legal_hold(key, version_id=win.version_id)
    assert obj.legal_hold_status(key, version_id=win.version_id) is False
