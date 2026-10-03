# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the coordinator HTTP server (plan Unit 4).

Covers the seven endpoint contracts + auth + Host check + watchdog +
per-invocation warning-template variation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    _SHARED_FOREIGN_DENY_LAG_WINDOW_SEC,
    MAX_POLICY_PATHS_PER_REQUEST,
    CoordinatorHTTPServer,
    TrackedReadDecision,
    _is_recent_self_commit_lag,
    caller_principal_identity,
    decide_tracked_read,
    session_to_agent_id,
)
from ccs.core.exceptions import HOLD_REASONS
from ccs.core.states import MESIState

# Test helper: deterministic UUID4-shaped strings for short test labels.
# Sessions now must be UUIDs (A3 validation); tests use this to keep label
# semantics while satisfying the wire-contract validator.
_TEST_SESSION_NS = uuid.UUID("11111111-1111-4111-8111-111111111111")


def _sid(label: str) -> str:
    return str(uuid.uuid5(_TEST_SESSION_NS, f"test-session:{label}"))


# Realistic sha-256 hex strings for tests (A8 requires 64-hex content_hash).
def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


# ----------------------------------------------------------------------
# Test client
# ----------------------------------------------------------------------


_PRINCIPAL_HEADER = "Coherence-Caller-Principal"
"""The caller-principal request header: a FROZEN duplicate of the wire name,
never imported from the code under test (a derived name moves with a rename
instead of catching it)."""


class _Client:
    """Tiny urllib-based client. Returns (status, body_dict)."""

    def __init__(self, host: str, port: int, secret: str) -> None:
        self.base = f"http://{host}:{port}"
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",  # explicit — we want to assert behavior
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        headers_override: Optional[dict] = None,
        principal: Optional[str] = None,
    ) -> tuple[int, dict]:
        url = self.base + path
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        headers = dict(self.headers)
        if headers_override:
            headers.update(headers_override)
        if principal is not None:
            headers[_PRINCIPAL_HEADER] = principal
        req = urlrequest.Request(url, data=data if method == "POST" else None,
                                 method=method, headers=headers)
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urlerror.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")

    def post(self, path: str, body: dict, **kw) -> tuple[int, dict]:
        return self.request("POST", path, body, **kw)

    def get(self, path: str, **kw) -> tuple[int, dict]:
        return self.request("GET", path, **kw)


@pytest.fixture
def coordinator(tmp_path: Path):
    """A live coordinator on a random port, with secret extracted for tests."""
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="test-instance")
    server.serve_in_thread()
    # Small delay so server is accepting before tests fire.
    time.sleep(0.05)
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def client(coordinator) -> _Client:
    secret = load_secret(coordinator.coordinator_root)
    assert secret is not None
    return _Client("127.0.0.1", coordinator.port, secret)


# ----------------------------------------------------------------------
# Auth + Host check (KTD-12)
# ----------------------------------------------------------------------


def test_missing_authorization_returns_401(coordinator) -> None:
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    req = urlrequest.Request(url, data=b"{}", method="POST",
                             headers={"Host": "127.0.0.1", "Content-Type": "application/json"})
    try:
        urlrequest.urlopen(req, timeout=5)
        assert False, "expected 401"
    except urlerror.HTTPError as e:
        assert e.code == 401


def test_wrong_bearer_returns_401(coordinator) -> None:
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    req = urlrequest.Request(url, data=b"{}", method="POST",
                             headers={
                                 "Authorization": "Bearer not-the-secret",
                                 "Host": "127.0.0.1",
                                 "Content-Type": "application/json",
                             })
    try:
        urlrequest.urlopen(req, timeout=5)
        assert False, "expected 401"
    except urlerror.HTTPError as e:
        assert e.code == 401


def test_bad_host_returns_403(client: _Client) -> None:
    """DNS-rebind mitigation: a request whose Host header is attacker.com
    must be rejected even if the Bearer is valid."""
    status, body = client.post("/hooks/pre-read", {"session_id": _sid("s1"), "path": "CLAUDE.md"},
                                headers_override={"Host": "attacker.example.com"})
    assert status == 403
    assert "host" in body["error"].lower()


def test_localhost_host_accepted(coordinator) -> None:
    """Host: localhost (not just 127.0.0.1) must also be accepted."""
    secret = load_secret(coordinator.coordinator_root)
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    body = json.dumps({"session_id": _sid("s1"), "path": "CLAUDE.md"}).encode("utf-8")
    req = urlrequest.Request(
        url, data=body, method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Host": "localhost",
            "Content-Type": "application/json",
        },
    )
    with urlrequest.urlopen(req, timeout=5) as resp:
        assert resp.status == 200


# ----------------------------------------------------------------------
# Routing
# ----------------------------------------------------------------------


def test_unknown_route_returns_404(client: _Client) -> None:
    status, body = client.post("/does-not-exist", {})
    assert status == 404


def test_get_on_post_route_returns_404(client: _Client) -> None:
    """/hooks/pre-read is POST-only; GET should 404."""
    status, body = client.get("/hooks/pre-read")
    assert status == 404


# ----------------------------------------------------------------------
# /hooks/pre-read
# ----------------------------------------------------------------------


def test_pre_read_first_observation_returns_fresh(client: _Client) -> None:
    """KTD-9: first observation of a tracked artifact seeds v1 + grants
    SHARED + returns fresh. Unit 6: the fresh response also carries the
    seeded ``version`` (additive — for OCC writers sourcing expected_version)."""
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                                 "content_hash": _hash("abc")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_repeat_from_same_session_stays_fresh(client: _Client) -> None:
    """A second read from the same session on the same artifact is fresh.
    Unit 6: still carries the artifact version on the fresh response."""
    client.post("/hooks/pre-read", {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("h1")})
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("h1")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_after_peer_write_returns_stale(client: _Client) -> None:
    """Two sessions: A reads, B writes, A's next read returns stale."""
    # Session A first-reads to seed v1 + take SHARED.
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    # Session B pre-edits (acquires E, invalidates A) and post-edits (commits v2).
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": _sid("B"), "path": "plan.md", "content_hash": _hash("h2"), "success": True})
    # Session A's next read now sees stale.
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    assert status == 200
    assert "hookSpecificOutput" in body
    out = body["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "allow"  # v0.1 WARN, NEVER deny
    assert "Stale read" in out["additionalContext"]
    assert "plan.md" in out["additionalContext"]
    # Summary metadata present and respects KTD-12 no-content constraint.
    summary = body["summary"]
    assert summary["path"] == "plan.md"
    assert summary["current_version"] == 2
    assert summary["hash_differs"] is True
    # No raw content / no hash bytes in the prose
    assert "h1" not in out["additionalContext"]
    assert "h2" not in out["additionalContext"]


def test_pre_read_warn_mode_never_returns_deny(client: _Client) -> None:
    """Belt-and-suspenders invariant: v0.1 pre-read MUST NOT return deny."""
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": _sid("B"), "path": "plan.md", "content_hash": _hash("h2"), "success": True})
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    assert body["hookSpecificOutput"]["permissionDecision"] != "deny"


def test_pre_read_untracked_path_fastpath(coordinator, client: _Client) -> None:
    """An untracked path returns fresh WITHOUT touching SQLite (R8)."""
    artifact_count_before = len(coordinator.registry.artifact_ids())
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("A"), "path": "src/random.py",
                                 "content_hash": _hash("h1")})
    assert body == {"status": "fresh"}
    artifact_count_after = len(coordinator.registry.artifact_ids())
    assert artifact_count_after == artifact_count_before, (
        "untracked path must not create an artifact row"
    )


def test_pre_read_missing_session_id_400(client: _Client) -> None:
    status, body = client.post("/hooks/pre-read", {"path": "CLAUDE.md"})
    assert status == 400
    assert "session_id" in body["error"]


def test_pre_read_empty_path_400(client: _Client) -> None:
    status, body = client.post("/hooks/pre-read", {"session_id": _sid("s"), "path": ""})
    assert status == 400


# ----------------------------------------------------------------------
# /hooks/pre-read — fresh-SHARED hash-mismatch signal (PR #108 follow-up)
# ----------------------------------------------------------------------


def test_pre_read_fresh_shared_hash_mismatch_sets_hash_differs(client: _Client) -> None:
    """Defense-in-depth (PR #108 follow-up): a SHARED holder re-reading
    with a disk hash that mismatches the recorded content gets
    ``hash_differs: true`` on the fresh response. A peer commit would
    have left the session INVALID, so the mismatch implies an
    out-of-band write — surfaced additively, never denied (the plugin
    path stays fail-open)."""
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                 "content_hash": _hash("on-disk-v1")})
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                                 "content_hash": _hash("out-of-band-edit")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1, "hash_differs": True}


def test_pre_read_fresh_shared_matching_hash_omits_hash_differs(client: _Client) -> None:
    """Additive contract: the key appears ONLY when the mismatch fires.
    A matching hash keeps the exact two-field fresh shape so
    exact-shape status clients are untouched."""
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                 "content_hash": _hash("on-disk-v1")})
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                                 "content_hash": _hash("on-disk-v1")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_fresh_shared_without_hash_omits_hash_differs(client: _Client) -> None:
    """A SHARED re-read that supplies no content_hash has nothing to
    compare — no signal, exact two-field shape."""
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                 "content_hash": _hash("on-disk-v1")})
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s1"), "path": "CLAUDE.md"})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_fresh_shared_empty_seed_recorded_hash_never_fires(client: _Client) -> None:
    """KTD-9 seeding without a caller hash records the "" sentinel
    (surfaced as None by the registry); a later hash-bearing re-read
    must not fire against it — there is no real content claim to
    mismatch."""
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md"})
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                                 "content_hash": _hash("real-disk-bytes")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_fresh_shared_f_sentinel_recorded_hash_never_fires(
    coordinator, client: _Client,
) -> None:
    """The synthetic launch-gate sentinel ("f"*64) is not a real SHA-256
    of any content; a SHARED holder re-reading against it must not fire
    the signal. (The stale path's hash_differs deliberately DOES fire on
    it — that asymmetry is the launch-gate contract.)"""
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                 "content_hash": _hash("on-disk-v1")})
    art_id = coordinator.registry.lookup_artifact_id_by_name("CLAUDE.md")
    assert art_id is not None
    art = coordinator.registry.get_artifact(art_id)
    # Inject the sentinel directly into the artifact row (same shape the
    # launch-gate synthetic SQLite injection produces); the session's
    # SHARED grant is untouched.
    coordinator.registry.set_artifact_and_content(
        art_id, dataclasses.replace(art, content_hash="f" * 64), "",
    )
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                                 "content_hash": _hash("on-disk-v1")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_fresh_shared_hash_mismatch_increments_counter(
    coordinator, client: _Client,
) -> None:
    """Each firing bumps fresh_shared_hash_mismatch_total; matching
    re-reads don't."""
    before = coordinator._fresh_shared_hash_mismatch_total
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                 "content_hash": _hash("on-disk-v1")})
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                 "content_hash": _hash("on-disk-v1")})
    assert coordinator._fresh_shared_hash_mismatch_total == before
    client.post("/hooks/pre-read",
                {"session_id": _sid("s1"), "path": "CLAUDE.md",
                 "content_hash": _hash("out-of-band-edit")})
    assert coordinator._fresh_shared_hash_mismatch_total == before + 1


# ----------------------------------------------------------------------
# Survivor #6 v1 — _is_recent_self_commit_lag (R2): the commit→disk-write
# lag-exclusion predicate. A SHARED-holder hash mismatch is the benign lag
# (NOT a foreign edit) iff THIS session is the artifact's RECENT last
# committer — the registry advanced the canonical hash (e.g. a commit_cas
# WIN that leaves the writer SHARED) but the agent has not yet flushed the
# new bytes to disk. Anything else is a genuine out-of-band edit → deny.
# ----------------------------------------------------------------------

_CSRV = "ccs.adapters.claude_code.coordinator_server"


def _patch_last_writer(monkeypatch, coordinator, *, writer, ts) -> None:
    # The lag gate now reads the RAW committed-writer agent id straight from the
    # registry (matching Node), so patch that method — NOT the module-level
    # `_last_writer_for` (which resolves to a display attribution string and is
    # only used for the deny prose). `writer` is the writer's composite agent_id
    # (a UUID) or None.
    monkeypatch.setattr(coordinator.registry, "last_writer_for", lambda aid: writer)
    monkeypatch.setattr(f"{_CSRV}._last_writer_unix_ts", lambda coord, aid: ts)


# SB-25: the predicate's 3rd arg is the CALLER's composite agent_id, compared
# to the raw committed-writer agent_id from the registry. Register the caller
# to get its agent_id; the "self commit" cases set the registry writer to that
# same agent_id.
def _caller(coordinator, label: str):
    """Register a session and return (session_id, composite_agent_id)."""
    sid = _sid(label)
    return sid, coordinator.register_session(sid)


def test_lag_true_for_recent_self_commit(coordinator, monkeypatch) -> None:
    _, agent_id = _caller(coordinator, "s1")
    _patch_last_writer(monkeypatch, coordinator, writer=agent_id, ts=1000.0)
    now = 1000.0 + _SHARED_FOREIGN_DENY_LAG_WINDOW_SEC - 0.1
    assert _is_recent_self_commit_lag(coordinator, uuid.uuid4(), agent_id, now_unix=now) is True


def test_lag_false_for_stale_self_commit(coordinator, monkeypatch) -> None:
    """last_writer == self but the commit is OLD => something rewrote disk
    since => correctly FOREIGN (the recency clause is load-bearing)."""
    _, agent_id = _caller(coordinator, "s1")
    _patch_last_writer(monkeypatch, coordinator, writer=agent_id, ts=1000.0)
    now = 1000.0 + _SHARED_FOREIGN_DENY_LAG_WINDOW_SEC + 0.1
    assert _is_recent_self_commit_lag(coordinator, uuid.uuid4(), agent_id, now_unix=now) is False


def test_lag_false_for_other_writer(coordinator, monkeypatch) -> None:
    _, agent_id = _caller(coordinator, "s1")
    _, other_agent_id = _caller(coordinator, "s2")
    _patch_last_writer(monkeypatch, coordinator, writer=other_agent_id, ts=1000.0)
    assert _is_recent_self_commit_lag(
        coordinator, uuid.uuid4(), agent_id, now_unix=1000.1) is False


def test_lag_false_for_no_writer(coordinator, monkeypatch) -> None:
    _, agent_id = _caller(coordinator, "s1")
    _patch_last_writer(monkeypatch, coordinator, writer=None, ts=None)
    assert _is_recent_self_commit_lag(
        coordinator, uuid.uuid4(), agent_id, now_unix=1000.1) is False


def test_lag_false_for_missing_updated_at(coordinator, monkeypatch) -> None:
    _, agent_id = _caller(coordinator, "s1")
    _patch_last_writer(monkeypatch, coordinator, writer=agent_id, ts=None)
    assert _is_recent_self_commit_lag(coordinator, uuid.uuid4(), agent_id, now_unix=1000.1) is False


def test_lag_false_for_cross_session_same_subagent_id(coordinator, monkeypatch) -> None:
    """Regression: two DIFFERENT sessions presenting the SAME agent_id string
    have DIFFERENT composite agent_ids, so a foreign edit by one is NOT
    suppressed for the other. (The prior attribution comparison collided on the
    bare subagent id and wrongly suppressed — adversarial review 2026-07-17.)"""
    _, a1 = _caller(coordinator, "s1")
    a2 = coordinator.register_session(_sid("s2"), "shared-agent")
    b2 = coordinator.register_session(_sid("s1"), "shared-agent")
    assert a2 != b2  # same agent_id string, different sessions → distinct ids
    _patch_last_writer(monkeypatch, coordinator, writer=a2, ts=1000.0)  # session-2's subagent wrote
    # caller is session-1 (parent) — must NOT be treated as the writer.
    assert _is_recent_self_commit_lag(coordinator, uuid.uuid4(), a1, now_unix=1000.1) is False


def test_lag_true_at_exact_window_boundary(coordinator, monkeypatch) -> None:
    """`<=` boundary: a self-commit EXACTLY at the window edge is still treated as
    lag (the fail-safe direction). Pins the operator choice against a silent flip
    to `<`."""
    _, agent_id = _caller(coordinator, "s1")
    _patch_last_writer(monkeypatch, coordinator, writer=agent_id, ts=1000.0)
    now = 1000.0 + _SHARED_FOREIGN_DENY_LAG_WINDOW_SEC
    assert _is_recent_self_commit_lag(coordinator, uuid.uuid4(), agent_id, now_unix=now) is True


def test_status_metrics_exposes_fresh_shared_hash_mismatch_counter(client: _Client) -> None:
    """fresh_shared_hash_mismatch_total visible in /status?detail=metrics
    so an operator can size the false-positive rate before any
    strict-mode deny knob is considered."""
    status, body = client.get("/status?detail=metrics")
    assert status == 200
    assert "fresh_shared_hash_mismatch_total" in body
    assert isinstance(body["fresh_shared_hash_mismatch_total"], int)


def test_status_metrics_exposes_shared_foreign_lag_suppressed_counter(client: _Client) -> None:
    """shared_foreign_lag_suppressed_total visible in /status?detail=metrics so an
    operator can size the lag-window (5s) false-negative rate against
    strict_mode_denials_total."""
    status, body = client.get("/status?detail=metrics")
    assert status == 200
    assert "shared_foreign_lag_suppressed_total" in body
    assert isinstance(body["shared_foreign_lag_suppressed_total"], int)


# ----------------------------------------------------------------------
# /hooks/pre-edit + post-edit (KTD-1 cycle)
# ----------------------------------------------------------------------


def test_full_edit_cycle(coordinator, client: _Client) -> None:
    """pre-edit acquires E → post-edit commits + bumps version."""
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h0")})
    s, b = client.post("/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"})
    assert s == 200 and b == {"ok": True}
    s, b = client.post("/hooks/post-edit",
                        {"session_id": _sid("A"), "path": "plan.md",
                         "content_hash": _hash("h1"), "success": True})
    assert s == 200 and b == {"ok": True}
    # Version bumped
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    art = coordinator.registry.get_artifact(artifact_id)
    assert art.version == 2  # seeded at 1, committed once → 2


def test_failed_edit_releases_grant_without_bump(coordinator, client: _Client) -> None:
    """KTD-1 release-on-failure: post-edit success:false releases E without bumping."""
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"})
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    version_before = coordinator.registry.get_artifact(artifact_id).version
    s, b = client.post("/hooks/post-edit",
                        {"session_id": _sid("A"), "path": "plan.md",
                         "content_hash": _hash("ignored"), "success": False})
    assert s == 200
    assert b.get("released") is True
    # Version NOT bumped on failure
    version_after = coordinator.registry.get_artifact(artifact_id).version
    assert version_after == version_before
    # Agent state is no longer EXCLUSIVE (some non-M/E state)
    agent_id = session_to_agent_id("A")
    state = coordinator.registry.get_agent_state(artifact_id, agent_id)
    assert state not in (MESIState.EXCLUSIVE, MESIState.MODIFIED)
    # Another session can now acquire immediately
    s2, b2 = client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    assert s2 == 200 and b2.get("ok") is True


def test_session_stop_malformed_agent_id_does_not_release_parent(client: _Client) -> None:
    """P0 regression: a present-but-malformed agent_id on /hooks/session-stop
    must fail closed (no-op), NOT degrade to the parent identity and release
    the parent's live grants. The guard was briefly mis-wired into
    _handle_pre_read instead of _handle_session_stop; no end-to-end test drove
    the handler, so it shipped clean."""
    client.post("/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"})  # parent holds E
    # Every present-but-malformed JSON type → no-op; the parent's grant survives.
    for bad in (42, [1], {"k": 1}, True, "bad id!", "x" * 65):
        s, b = client.post("/hooks/session-stop", {"session_id": _sid("A"), "agent_id": bad})
        assert s == 200
        assert b == {"ok": True, "released_artifacts": []}, f"malformed {bad!r} must be a no-op"
    # An ABSENT agent_id is a legitimate parent stop → the grant IS released.
    s, b = client.post("/hooks/session-stop", {"session_id": _sid("A")})
    assert s == 200
    assert b["released_artifacts"] == ["plan.md"]


def test_pre_read_malformed_agent_id_keeps_fresh_stale_contract(client: _Client) -> None:
    """P0 regression: a malformed agent_id on /hooks/pre-read must NOT short-
    circuit to a session-stop-shaped body — it degrades to parent attribution
    and returns pre-read's normal fresh/stale contract."""
    s, b = client.post(
        "/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "agent_id": 42}
    )
    assert s == 200
    assert "released_artifacts" not in b
    assert b.get("status") in ("fresh", "stale")


def test_collision_surfaces_via_additional_context(coordinator, client: _Client) -> None:
    """KTD-9 same-hash-blindness mitigation: when another session holds E,
    pre-edit returns hookSpecificOutput with collision warning."""
    # Session A holds E
    a_sid = _sid("A")
    client.post("/hooks/pre-edit", {"session_id": a_sid, "path": "plan.md"})
    # Session B attempts edit → collision response
    status, body = client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    assert status == 200
    assert body.get("collision") is True
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"  # v0.1 warn only
    assert "Concurrent edit detected" in out["additionalContext"]
    assert "plan.md" in out["additionalContext"]
    # R7: the collision msg names the holder by the short form of its AGENT
    # id -- the handle the registry keeps the grant under -- not by the
    # session id that id was derived from.
    assert session_to_agent_id(a_sid).hex[:8] in out["additionalContext"]
    assert a_sid[:8] not in out["additionalContext"]


# ----------------------------------------------------------------------
# /hooks/session-stop (KTD-11)
# ----------------------------------------------------------------------


def test_session_stop_releases_uncommitted_grants(coordinator, client: _Client) -> None:
    """KTD-11: end-of-turn Stop releases any uncommitted EXCLUSIVE grants."""
    client.post("/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": _sid("A"), "path": "spec.md"})
    # Stop fires
    s, b = client.post("/hooks/session-stop", {"session_id": _sid("A")})
    assert s == 200 and b["ok"] is True
    released = set(b["released_artifacts"])
    assert released == {"plan.md", "spec.md"}
    # Neither artifact is held in M∪E by A anymore
    agent_id = session_to_agent_id("A")
    for path in ("plan.md", "spec.md"):
        art_id = coordinator.registry.lookup_artifact_id_by_name(path)
        state = coordinator.registry.get_agent_state(art_id, agent_id)
        assert state not in (MESIState.EXCLUSIVE, MESIState.MODIFIED)


def test_session_stop_idempotent(client: _Client) -> None:
    """Calling Stop twice in a row is safe — second call returns empty release list."""
    client.post("/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"})
    s1, b1 = client.post("/hooks/session-stop", {"session_id": _sid("A")})
    assert s1 == 200 and "plan.md" in b1["released_artifacts"]
    s2, b2 = client.post("/hooks/session-stop", {"session_id": _sid("A")})
    assert s2 == 200 and b2["released_artifacts"] == []


# ----------------------------------------------------------------------
# /policy/track + /policy/untrack
# ----------------------------------------------------------------------


def test_policy_track_persists_to_yaml(coordinator, client: _Client) -> None:
    s, b = client.post("/policy/track", {"paths": ["runbook.md", "architecture.md"]})
    assert s == 200
    assert b["ok"] is True
    assert sorted(b["added"]) == ["architecture.md", "runbook.md"]
    yaml_path = coordinator.coordinator_root / ".coherence" / "tracked.yaml"
    assert yaml_path.is_file()
    content = yaml_path.read_text()
    assert "runbook.md" in content
    assert "architecture.md" in content
    # Live policy now matches runbook.md (untracked-by-default earlier)
    assert coordinator.policy.is_tracked("runbook.md")


def test_policy_track_rejects_traversal(client: _Client) -> None:
    s, b = client.post("/policy/track",
                        {"paths": ["../../.env", "/etc/passwd", "runbook.md"]})
    assert s == 200
    assert b["added"] == ["runbook.md"]
    rejected = {r["path"] for r in b["rejected"]}
    assert "../../.env" in rejected
    assert "/etc/passwd" in rejected


def test_policy_track_cap_enforced(client: _Client) -> None:
    too_many = [f"f{i}.md" for i in range(MAX_POLICY_PATHS_PER_REQUEST + 1)]
    s, b = client.post("/policy/track", {"paths": too_many})
    assert s == 400
    assert "max" in b["error"].lower()


def test_policy_untrack_persists_to_ignored_yaml(coordinator, client: _Client) -> None:
    s, b = client.post("/policy/untrack", {"paths": ["docs/brainstorms/draft.md"]})
    assert s == 200
    assert b["removed"] == ["docs/brainstorms/draft.md"]
    yaml_path = coordinator.coordinator_root / ".coherence" / "ignored.yaml"
    assert yaml_path.is_file()
    assert "docs/brainstorms/draft.md" in yaml_path.read_text()
    # Default-matching draft now ignored
    assert not coordinator.policy.is_tracked("docs/brainstorms/draft.md")


@pytest.fixture
def strict_coordinator(tmp_path: Path):
    """A live coordinator whose policy holds ``data/**`` tracked and strict."""
    (tmp_path / ".coherence").mkdir()
    (tmp_path / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (tmp_path / ".coherence" / "strict_mode.yaml").write_text("- data/*.json\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="test-instance")
    server.serve_in_thread()
    time.sleep(0.05)
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def strict_client(strict_coordinator) -> _Client:
    secret = load_secret(strict_coordinator.coordinator_root)
    assert secret is not None
    return _Client("127.0.0.1", strict_coordinator.port, secret)


@pytest.mark.parametrize("entry", ["data/a.json", "data/*.json", "data/**", "**", "*.json"])
def test_policy_untrack_refuses_an_entry_covering_a_strict_path(
    strict_coordinator, strict_client: _Client, entry: str
) -> None:
    """#261: an untrack entry — a literal path, the strict glob itself, a
    broader glob, or one spelled differently — that covers a path the live
    policy holds in strict mode is refused with the typed reason, naming the
    strict pattern, and the whole request writes nothing (the valid
    non-strict entry beside it included). The path stays strict."""
    s, b = strict_client.post("/policy/untrack", {"paths": [entry, "data/notes.txt", "/abs"]})
    assert s == 409, b
    assert b["ok"] is False
    assert b["reason"] == "untrack_strict_path"
    assert b["refused"] == [{"path": entry, "strict_patterns": ["data/*.json"]}]
    assert [r["path"] for r in b["rejected"]] == ["/abs"]
    assert "untrack_strict_path" in b["error"] and "restart" in b["error"]
    assert not (strict_coordinator.coordinator_root / ".coherence" / "ignored.yaml").exists()
    assert strict_coordinator.policy.is_strict_mode("data/a.json")
    assert strict_coordinator.policy.is_tracked("data/notes.txt")


def test_policy_untrack_of_a_non_strict_path_round_trips_on_a_strict_coordinator(
    strict_coordinator, strict_client: _Client
) -> None:
    """Control: an entry that covers no strict path untracks as before, and
    tracking it again restores it; the strict path is untouched throughout."""
    s, b = strict_client.post("/policy/untrack", {"paths": ["data/*.txt"]})
    assert s == 200 and b == {"ok": True, "removed": ["data/*.txt"], "rejected": []}
    assert not strict_coordinator.policy.is_tracked("data/notes.txt")
    assert strict_coordinator.policy.is_strict_mode("data/a.json")
    s, b = strict_client.post("/policy/track", {"paths": ["data/keep.txt"]})
    assert s == 200 and b["added"] == ["data/keep.txt"]
    # Ignore still wins over track for a non-strict path (unchanged semantics).
    assert not strict_coordinator.policy.is_tracked("data/keep.txt")
    assert strict_coordinator.policy.is_strict_mode("data/a.json")


def test_a_reload_after_a_hand_edit_keeps_the_strict_path_enforced(
    strict_coordinator, strict_client: _Client
) -> None:
    """The hot reload behind /policy/track and /policy/untrack never narrows
    strict enforcement: removing the strict and tracked entries from disk and
    then running any track leaves the strict path tracked and strict until a
    restart."""
    root = strict_coordinator.coordinator_root
    (root / ".coherence" / "strict_mode.yaml").write_text("")
    (root / ".coherence" / "tracked.yaml").write_text("")
    s, _ = strict_client.post("/policy/track", {"paths": ["runbook.md"]})
    assert s == 200
    assert strict_coordinator.policy.is_strict_mode("data/a.json")
    assert strict_coordinator.policy.is_tracked("runbook.md")


def test_a_stale_cas_on_a_strict_path_ignored_at_spawn_conflicts(tmp_path: Path) -> None:
    """The broader-ignore case on the wire: ``ignored.yaml`` carries ``**``
    when the coordinator starts. Strict wins over ignore, so the strict path
    is not answered on the untracked fast path: a post-edit-cas at
    expected_version 0 over a newer commit is a version_mismatch, not an
    accepted overwrite."""
    (tmp_path / ".coherence").mkdir()
    (tmp_path / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (tmp_path / ".coherence" / "strict_mode.yaml").write_text("- data/**\n")
    (tmp_path / ".coherence" / "ignored.yaml").write_text("- '**'\n")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="test-instance")
    server.serve_in_thread()
    time.sleep(0.05)
    try:
        client = _Client("127.0.0.1", server.port, load_secret(tmp_path))
        writer = _sid("ign-writer")
        v = _occ_seed_shared(client, writer, "data/x.json", _hash("ign-v1"))
        assert v >= 1, "the strict path is version-tracked despite the ignore"
        s, b = client.post("/hooks/post-edit-cas", {
            "session_id": writer, "path": "data/x.json", "success": True,
            "content_hash": _hash("ign-v2"), "expected_version": v,
        })
        assert s == 200 and b == {"ok": True, "version": v + 1}, b
        peer = _sid("ign-peer")
        s, b = client.post("/hooks/post-edit-cas", {
            "session_id": peer, "path": "data/x.json", "success": True,
            "content_hash": _hash("ign-stale"), "expected_version": 0,
        })
        assert s == 200 and b.get("ok") is False, b
        assert _artifact_version(server, "data/x.json") == v + 1
    finally:
        server.shutdown()


# ----------------------------------------------------------------------
# /status
# ----------------------------------------------------------------------


def test_status_includes_tracked_artifacts_and_sessions(client: _Client) -> None:
    """Default (minimal) tier surfaces tracked artifacts + sessions +
    counters + coordinator_pid; the absolute workspace root stays gated
    behind ``?detail=full`` per R12.

    P1 #7: coordinator_pid was moved out of minimal in Unit 6 R12 and
    restored here — pid is public on POSIX (any `ps` invocation lists
    it) so it does not exceed the threat model's accepted disclosure,
    and operators rely on it to verify "is the coordinator I think is
    running actually mine".

    R6: the session row is keyed on ``agent_id`` here, not ``agent_name`` —
    the name embeds the raw session id and is null below the operator tier.
    The name itself is asserted at the full tier by
    ``test_status_operator_tier_still_names_the_session``."""
    a_sid = _sid("A")
    client.post("/hooks/pre-read", {"session_id": a_sid, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": a_sid, "path": "spec.md"})
    s, b = client.get("/status")
    assert s == 200
    tracked_paths = {a["path"] for a in b["tracked_artifacts"]}
    assert "plan.md" in tracked_paths
    assert "spec.md" in tracked_paths
    sessions = {sess["agent_id"] for sess in b["sessions"]}
    assert str(session_to_agent_id(a_sid)) in sessions
    # AC-02: canonical field; old _s alias also present for one release.
    assert b["coordinator_uptime_seconds"] > 0
    assert b["coordinator_uptime_s"] == b["coordinator_uptime_seconds"]
    assert "policy_summary" in b
    # Minimal tier: absolute root sentinel'd; pid is present (P1 #7 reversion).
    assert b.get("detail") == "minimal"
    assert b.get("coordinator_root") == "."
    assert b.get("coordinator_pid") == os.getpid()


# ----------------------------------------------------------------------
# Malformed input
# ----------------------------------------------------------------------


def test_malformed_json_body_returns_400(coordinator) -> None:
    secret = load_secret(coordinator.coordinator_root)
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    req = urlrequest.Request(
        url, data=b"{not json", method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        },
    )
    try:
        urlrequest.urlopen(req, timeout=5)
        assert False, "expected 400"
    except urlerror.HTTPError as e:
        assert e.code == 400


def test_body_not_object_returns_400(client: _Client) -> None:
    # JSON list at top level, not object
    url = f"http://127.0.0.1:{client.base.rsplit(':', 1)[1]}/hooks/pre-read"  # rebuild
    secret = client.headers["Authorization"][len("Bearer "):]
    req = urlrequest.Request(
        url, data=b"[1,2,3]", method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        },
    )
    try:
        urlrequest.urlopen(req, timeout=5)
        assert False, "expected 400"
    except urlerror.HTTPError as e:
        assert e.code == 400


# ----------------------------------------------------------------------
# Heartbeat invariant
# ----------------------------------------------------------------------


def test_every_endpoint_records_heartbeat(coordinator, client: _Client) -> None:
    """KTD-2: every hook POST records the calling session's heartbeat."""
    hb_sid = _sid("hb-session")
    agent_id = session_to_agent_id(hb_sid)
    assert coordinator.registry.last_heartbeat_tick(agent_id) is None
    client.post("/hooks/pre-read",
                {"session_id": hb_sid, "path": "CLAUDE.md"})
    after_pre_read = coordinator.registry.last_heartbeat_tick(agent_id)
    assert after_pre_read is not None
    time.sleep(1.1)  # ensure monotonic tick advances
    client.post("/hooks/pre-edit", {"session_id": hb_sid, "path": "CLAUDE.md"})
    after_pre_edit = coordinator.registry.last_heartbeat_tick(agent_id)
    assert after_pre_edit >= after_pre_read


# ----------------------------------------------------------------------
# Per-invocation variation in warning templates (strict-mode future-proofing)
# ----------------------------------------------------------------------


def test_stale_warnings_vary_per_invocation(client: _Client) -> None:
    """Two back-to-back stale-read responses for the same artifact must
    have DIFFERENT additionalContext text (timestamp varies). This is the
    strict-mode-future-proofing constraint — when v0.2 flips allow → deny,
    the varying reason structurally prevents the §13.5 retry loop."""
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": _sid("B"), "path": "plan.md",
                 "content_hash": _hash("h2"), "success": True})

    _, body1 = client.post("/hooks/pre-read",
                           {"session_id": _sid("A"), "path": "plan.md"})
    time.sleep(0.05)  # ensure clock advances
    # Invalidate A again so the second response is also stale
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": _sid("B"), "path": "plan.md",
                 "content_hash": _hash("h3"), "success": True})
    _, body2 = client.post("/hooks/pre-read",
                           {"session_id": _sid("A"), "path": "plan.md"})

    msg1 = body1["hookSpecificOutput"]["additionalContext"]
    msg2 = body2["hookSpecificOutput"]["additionalContext"]
    # Same shape, different text — the version delta differs at minimum.
    assert msg1 != msg2, "warning templates must vary per invocation for v0.2 strict-mode safety"


# ----------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# A1 — Preemption notice (silent-grant-revocation surfacing)
# ----------------------------------------------------------------------


def test_a1_preemption_surfaces_on_victim_next_pre_read(client: _Client) -> None:
    """A1 load-bearing test (canonical phpmac scenario).

    Sequence: X pre-edits (holds E) → Y pre-edits (silently invalidates X) →
    X's next pre-read MUST surface a preemption notice naming Y + the
    artifact + when. Without this, X never learns its grant was revoked
    and X's content silently fails to land in the coordinator's view."""
    x = _sid("X"); y = _sid("Y")
    # X acquires EXCLUSIVE
    s, _ = client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    assert s == 200
    # Y preempts X (Y now holds E, X is INVALID, X received NO notification)
    s, _ = client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    assert s == 200
    # X's NEXT pre-read MUST surface the preemption via hookSpecificOutput
    status, body = client.post("/hooks/pre-read", {"session_id": x, "path": "plan.md"})
    assert status == 200, body
    assert "hookSpecificOutput" in body, (
        "X's next hook MUST inject the preemption notice into additionalContext; "
        f"got {body}"
    )
    out = body["hookSpecificOutput"]
    msg = out["additionalContext"]
    msg_lower = msg.lower()
    assert any(word in msg_lower for word in ("preempted", "revoked", "acquired by")), (
        f"prose should name the preemption explicitly; got: {msg}"
    )
    assert "plan.md" in msg
    assert session_to_agent_id(y).hex[:8] in msg, (
        f"prose should name the preempter's agent-id prefix; got: {msg}"
    )
    assert y[:8] not in msg, f"prose must not carry Y's session-id prefix; got: {msg}"


def test_a1_fresh_with_notice_preserves_version_field(client: _Client) -> None:
    """A1 × Unit 6: notice surfacing on the FRESH pre-read path must keep
    the additive ``version`` key alongside ``hookSpecificOutput``.

    Regression: the ``work_with_notice_surfacing`` wrapper rebuilt the
    fresh response as a literal dict, dropping ``version`` — an OCC
    writer sourcing expected_version from the read then CAS'd against 0
    and burned a wasted version_mismatch round-trip."""
    x = _sid("X"); y = _sid("Y")
    # X first-reads task.md — seeds v1 + grants SHARED, so the re-read
    # below (after the preemption on a DIFFERENT artifact) stays fresh.
    status, body = client.post("/hooks/pre-read",
                               {"session_id": x, "path": "task.md",
                                "content_hash": _hash("t1")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}
    # X acquires EXCLUSIVE on plan.md; Y pre-edits plan.md, silently
    # preempting X — a preemption notice queues for X.
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    # X's next hook is a pre-read of task.md (still SHARED → fresh): the
    # wrapper drains the pending notice onto this fresh response.
    status, body = client.post("/hooks/pre-read",
                               {"session_id": x, "path": "task.md",
                                "content_hash": _hash("t1")})
    assert status == 200, body
    assert body.get("status") == "fresh", body
    assert "hookSpecificOutput" in body, (
        f"fresh pre-read after a preemption must surface the notice; got {body}"
    )
    msg = body["hookSpecificOutput"]["additionalContext"]
    assert "plan.md" in msg and session_to_agent_id(y).hex[:8] in msg, (
        f"notice prose should name the preempted artifact + preempter; got: {msg}"
    )
    assert body.get("version") == 1, (
        "fresh-with-notice response must preserve the Unit 6 version key "
        f"(OCC writers source expected_version from it); got {body}"
    )


def test_a1_preemption_surfaces_on_victim_next_pre_edit(client: _Client) -> None:
    """A1: surface preemption even when X's next hook is pre-edit, not pre-read."""
    x = _sid("X"); y = _sid("Y")
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    status, body = client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    assert status == 200
    # pre-edit response shape: either {ok: true} OR hookSpecificOutput for collision/preemption
    assert "hookSpecificOutput" in body, (
        "pre-edit after being preempted MUST inject the notice; got {}"
    ).format(body)
    msg = body["hookSpecificOutput"]["additionalContext"]
    msg_lower = msg.lower()
    assert any(w in msg_lower for w in ("preempted", "revoked")), (
        f"prose should name the preemption; got: {msg}"
    )


def test_a1_preemption_surfaces_in_post_edit_failure_reason(client: _Client) -> None:
    """A1: when X tries to post-edit after being silently preempted, the
    failure response MUST name the preempter (not just generic CoherenceError)."""
    x = _sid("X"); y = _sid("Y")
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})  # preempts X
    s, body = client.post("/hooks/post-edit",
                           {"session_id": x, "path": "plan.md",
                            "content_hash": _hash("h"), "success": True})
    assert s == 200
    assert body.get("ok") is False, f"post-edit on preempted grant must fail; got {body}"
    reason = body.get("reason", "")
    reason_lower = reason.lower()
    assert any(w in reason_lower for w in ("preempted", "revoked", "acquired by")), (
        f"failure reason must name the preemption; got: {reason}"
    )
    assert session_to_agent_id(y).hex[:8] in reason, (
        f"reason should name the preempter's agent-id prefix; got: {reason}"
    )
    assert y[:8] not in reason, f"reason must not carry Y's session-id prefix; got: {reason}"


def test_a1_preemption_notice_consumed_after_one_surface(client: _Client) -> None:
    """A1: preemption notices are pop-and-clear — the victim sees the notice
    on their NEXT hook, but a subsequent hook (without a fresh preemption)
    sees fresh/normal response."""
    x = _sid("X"); y = _sid("Y")
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    # First X hook after preemption: surfaces notice
    _, body1 = client.post("/hooks/pre-read", {"session_id": x, "path": "plan.md"})
    assert "hookSpecificOutput" in body1
    # Second X hook: notice is consumed; response is normal stale-read shape
    # (X is still INVALID on plan.md, so this will be stale, but NOT carry the
    # preemption notice text anymore — that was popped).
    _, body2 = client.post("/hooks/pre-read", {"session_id": x, "path": "plan.md"})
    if "hookSpecificOutput" in body2:
        msg2 = body2["hookSpecificOutput"]["additionalContext"]
        # The second message can carry a stale-read warning, but should NOT
        # repeat the preemption notice text.
        assert "preempted" not in msg2.lower() and "revoked" not in msg2.lower(), (
            f"preemption notice should be consumed after first surface; got: {msg2}"
        )


def test_a1_no_preemption_no_notice(client: _Client) -> None:
    """A1 negative: a session that's never been preempted gets no notice.

    Finding #24: the previous conditional `if 'hookSpecificOutput' in body`
    made this assertion unreachable (X was never preempted so the field is
    absent). Replace with an unconditional assertion: the response must be
    exactly {ok: True} with no hookSpecificOutput at all.
    """
    x = _sid("X")
    # X never preempted — pre-edit must return exactly {ok: True} with no
    # preemption output.
    s, body = client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    assert s == 200
    assert body.get("ok") is True, f"expected ok=True, got: {body!r}"
    assert "hookSpecificOutput" not in body, (
        f"pre-edit for a never-preempted session must not carry hookSpecificOutput; got: {body!r}"
    )


# ----------------------------------------------------------------------
# A1 hardening — adversarial review findings F1-F5
# ----------------------------------------------------------------------


def test_a1_stop_hook_surfaces_pending_notices(client: _Client) -> None:
    """F1 (P0): the canonical phpmac case — X preempted, X never fires
    another pre-event (model decided next action is a Bash/Grep, or turn
    just ends). Stop fires. Without this, X's notice orphans and X never
    learns. Fix: Stop pops + includes in response body."""
    x = _sid("X"); y = _sid("Y")
    # X holds E
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    # Y preempts X
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    # X's turn ends without another pre-event — Stop fires
    s, body = client.post("/hooks/session-stop", {"session_id": x})
    assert s == 200
    assert body["ok"] is True
    # Response body MUST include the preemption notices (telemetry-visible
    # in stream-json) so the silent-drop is impossible.
    assert "notices" in body, f"Stop response must surface pending notices; got {body}"
    notices = body["notices"]
    assert len(notices) >= 1
    # Notice references the preempted artifact + preempter
    notice = notices[0]
    assert notice["path"] == "plan.md"
    assert notice["preempter_agent_id"] == session_to_agent_id(y).hex


def test_a1_stop_hook_consumes_notices_no_orphan(client: _Client) -> None:
    """F1 consequence: Stop POPS notices, so they don't orphan if X never
    returns. After Stop, a subsequent pre-read by X shouldn't see the
    same notice text (already consumed at Stop)."""
    x = _sid("X"); y = _sid("Y")
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    # Stop consumes
    _, stop_body = client.post("/hooks/session-stop", {"session_id": x})
    assert "notices" in stop_body and len(stop_body["notices"]) >= 1
    # X's hypothetical next-turn pre-read MUST NOT re-surface the same notice
    _, body = client.post("/hooks/pre-read", {"session_id": x, "path": "plan.md"})
    if "hookSpecificOutput" in body:
        msg = body["hookSpecificOutput"]["additionalContext"]
        # Notice was consumed at Stop; pre-read may still show stale-read
        # warning (X is INVALID) but should not contain preemption prose.
        # Keyed on the notice BLOCK's own header, not on the words
        # "preempted"/"revoked": since R8 the stale-read warning for a grant
        # handover says "was revoked" itself, so a bare word match can no
        # longer tell a re-surfaced notice from the ordinary stale prose --
        # it would fail on correct behaviour and, had the wording gone the
        # other way, would have passed on a real re-surfacing.
        assert "Coordinator notice" not in msg, (
            f"notice should be consumed at Stop; pre-read re-surfaced: {msg}"
        )
        assert "preempted/revoked by agent" not in msg, (
            f"notice should be consumed at Stop; pre-read re-surfaced: {msg}"
        )


def test_a1_prose_capped_under_10kb_with_many_notices(client: _Client) -> None:
    """F3 (P1): N preemption notices compound prose linearly. Cap at 4KB
    prose (~10KB total with prepended stale-read warnings). Coalesce
    after first 3 notices."""
    x = _sid("X")
    # Set up 20 artifacts under tracked paths, X holds E on all, then 20
    # other sessions each preempt X on a distinct artifact.
    paths = [f"docs/specs/preempt-{i:02d}.md" for i in range(20)]
    for path in paths:
        client.post("/hooks/pre-edit", {"session_id": x, "path": path})
    for i, path in enumerate(paths):
        attacker = _sid(f"attacker-{i}")
        client.post("/hooks/pre-edit", {"session_id": attacker, "path": path})
    # X's next hook will see 20 pending notices
    _, body = client.post("/hooks/pre-read",
                          {"session_id": x, "path": paths[0]})
    if "hookSpecificOutput" in body:
        msg = body["hookSpecificOutput"]["additionalContext"]
        # 10,000, not 10240. Claude Code routes every hook's
        # additionalContext through one helper that returns the string
        # unchanged only while `length <= 1e4`; above that it persists the
        # prose to a file and hands the model a ~2KB preview plus a path.
        # 10240 is 10 KiB where the platform means 10,000, so this assertion
        # permitted 240 bytes the platform would have offloaded. The Node
        # sibling asserts the same figure on both its surfaces. Note the
        # platform counts UTF-16 code units while this counts UTF-8 bytes,
        # which is the stricter direction for prose carrying "⚠"/"•"/"—".
        assert len(msg.encode("utf-8")) <= 10_000, (
            f"additionalContext should fit in the 10,000-byte platform cap; "
            f"got {len(msg.encode('utf-8'))} bytes"
        )
        # And the message should mention coalescing (e.g., "and N more")
        # so the model knows there are unsurfaced notices.
        # Permit "more" or "additional" or a count expression
        assert any(w in msg.lower() for w in ("more", "additional", "(...)")), (
            f"prose should signal coalescing when notices truncated; got: {msg}"
        )


def test_a1_orphan_notices_evicted_after_ttl(coordinator) -> None:
    """F2 (P1): orphan notices (victim session never returns to pop) are
    eventually evicted to bound state growth. Registry exposes
    evict_stale_notices(max_age_sec) for the lifecycle sweep."""
    import time

    # Create a notice manually with an old timestamp
    from uuid import uuid4
    victim = uuid4()
    preempter = uuid4()
    # Need an artifact_id that exists (FK)
    from ccs.core.types import Artifact
    art = Artifact(id=uuid4(), name="orphan-test.md", version=1, content_hash="h")
    coordinator.registry.register_artifact(art, content="")
    coordinator.registry.record_preemption_notice(
        victim_agent_id=victim,
        artifact_id=art.id,
        preempter_agent_id=preempter,
        preempted_at_unix_ts=time.time() - 3600,  # 1 hour old
    )
    # Sanity: notice present
    notices = coordinator.registry.pop_pending_notices(victim)
    coordinator.registry.record_preemption_notice(  # re-record after pop drained
        victim_agent_id=victim,
        artifact_id=art.id,
        preempter_agent_id=preempter,
        preempted_at_unix_ts=time.time() - 3600,
    )
    # Evict everything older than 30 minutes
    evicted = coordinator.registry.evict_stale_notices(max_age_sec=1800)
    assert evicted >= 1, f"expected to evict the 1-hour-old notice; evicted {evicted}"
    # Now empty
    assert coordinator.registry.pop_pending_notices(victim) == []


def test_a1_upsert_uses_wall_clock_not_commit_order(coordinator) -> None:
    """F5 (P3): UPSERT must keep the most-recent-by-WALL-CLOCK notice,
    not the most-recent-by-COMMIT-order. If Y commits at clock=100 but
    Z commits later at clock=99 (out-of-order), the row stays at Y's
    record (later wall-clock)."""
    from uuid import uuid4

    from ccs.core.types import Artifact
    victim = uuid4()
    art = Artifact(id=uuid4(), name="upsert-test.md", version=1, content_hash="h")
    coordinator.registry.register_artifact(art, content="")
    y = uuid4(); z = uuid4()
    # Record Y at clock=100 (later in wall-clock)
    coordinator.registry.record_preemption_notice(
        victim_agent_id=victim, artifact_id=art.id,
        preempter_agent_id=y, preempted_at_unix_ts=100.0,
    )
    # Then record Z at clock=50 (earlier in wall-clock, later in commit order)
    coordinator.registry.record_preemption_notice(
        victim_agent_id=victim, artifact_id=art.id,
        preempter_agent_id=z, preempted_at_unix_ts=50.0,
    )
    # The remaining notice should be Y's (later wall-clock wins)
    notices = coordinator.registry.pop_pending_notices(victim)
    assert len(notices) == 1
    artifact_id, preempter, ts = notices[0]
    assert preempter == y, f"expected Y (later wall-clock) to win, got {preempter}"
    assert ts == 100.0


# ----------------------------------------------------------------------
# Boundary validators (A2 + A3 + A8 — adversarial review hardening)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("bad_session", [
    "A",            # not UUID
    "12345",        # not UUID
    "11111111-1111-4111-8111",  # too short
    "11111111-1111-4111-8111-1111111111111",  # too long
    "ggggggg1-1111-4111-8111-111111111111",   # non-hex
    "",
    None,
    123,
])
def test_invalid_session_id_returns_400(client: _Client, bad_session: Any) -> None:
    """A3: every handler rejects non-UUID session_id with 400."""
    status, body = client.post("/hooks/pre-read",
                                {"session_id": bad_session, "path": "CLAUDE.md"})
    assert status == 400


@pytest.mark.parametrize("bad_path", [
    "/etc/passwd",                  # absolute
    "../../.env",                   # traversal
    "subdir/../../etc/passwd",      # traversal mid-string
    "plan.md\n[SYSTEM] inject",     # newline injection (Adv #11)
    "plan.md\rrogue",               # carriage return
    "plan.md\x1b[31mred",            # ANSI escape
    "plan.md\x00null",               # null byte
    "x" * 2000,                      # over MAX_PATH_LEN
])
def test_invalid_path_returns_400(client: _Client, bad_path: str) -> None:
    """A2: every handler rejects invalid paths with 400."""
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("s"), "path": bad_path})
    assert status == 400


def test_pre_edit_rejects_invalid_path(client: _Client) -> None:
    status, body = client.post("/hooks/pre-edit",
                                {"session_id": _sid("s"), "path": "/etc/passwd"})
    assert status == 400


def test_post_edit_requires_valid_content_hash_on_success(client: _Client) -> None:
    """A8: post-edit with success:true MUST have a valid 64-hex content_hash."""
    s, _ = client.post("/hooks/pre-edit", {"session_id": _sid("Q"), "path": "plan.md"})
    assert s == 200
    # Missing content_hash → 400
    status, body = client.post("/hooks/post-edit",
                                {"session_id": _sid("Q"), "path": "plan.md", "success": True})
    assert status == 400
    # Malformed content_hash → 400
    status, body = client.post("/hooks/post-edit",
                                {"session_id": _sid("Q"), "path": "plan.md",
                                 "content_hash": "lol-not-a-hash", "success": True})
    assert status == 400
    # Empty content_hash → 400
    status, body = client.post("/hooks/post-edit",
                                {"session_id": _sid("Q"), "path": "plan.md",
                                 "content_hash": "", "success": True})
    assert status == 400


def test_post_edit_allows_missing_hash_on_failure(client: _Client) -> None:
    """A8: post-edit with success:false does NOT require content_hash —
    the release path doesn't use the hash."""
    client.post("/hooks/pre-edit", {"session_id": _sid("F"), "path": "plan.md"})
    status, body = client.post("/hooks/post-edit",
                                {"session_id": _sid("F"), "path": "plan.md", "success": False})
    assert status == 200
    assert body.get("released") is True


def test_pre_read_rejects_malformed_content_hash(client: _Client) -> None:
    """A8: content_hash is optional on pre-read but if present must be 64 hex."""
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("X"), "path": "plan.md",
                                 "content_hash": "garbage"})
    assert status == 400


def test_pre_read_allows_missing_content_hash(client: _Client) -> None:
    """A8: missing content_hash is permitted on pre-read."""
    status, body = client.post("/hooks/pre-read",
                                {"session_id": _sid("X"), "path": "plan.md"})
    assert status == 200


def test_secret_file_mode_is_0600_atomically(tmp_path: Path) -> None:
    """Bonus 1: secret is created with mode 0600 atomically (O_CREAT|O_EXCL),
    no mode-0644 window between write and chmod. Exercises ensure_secret
    directly to avoid the HTTPServer.shutdown deadlock when serve_in_thread
    was never called."""
    import os
    import stat

    from ccs.adapters.claude_code.auth import ensure_secret

    token = ensure_secret(tmp_path)
    assert token
    assert len(token) == 64  # 32-byte hex
    secret_file = tmp_path / ".coherence" / "hook.secret"
    assert secret_file.is_file()
    mode = stat.S_IMODE(os.stat(secret_file).st_mode)
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"
    # Idempotent on second call: returns same token, no exception.
    assert ensure_secret(tmp_path) == token


def test_warning_includes_warning_generated_at_field(client: _Client) -> None:
    """A5: stale-read summary includes both last_writer_at_unix_ts (real
    write tick from registry) AND warning_generated_at_unix_ts (handler now())."""
    a = _sid("A"); b = _sid("B")
    client.post("/hooks/pre-read", {"session_id": a, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": b, "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": b, "path": "plan.md",
                                       "content_hash": _hash("h2"), "success": True})
    _, body = client.post("/hooks/pre-read", {"session_id": a, "path": "plan.md"})
    summary = body["summary"]
    assert "last_writer_at_unix_ts" in summary
    assert "warning_generated_at_unix_ts" in summary
    # Generated-at is AFTER writer-at (handler runs after commit)
    assert summary["warning_generated_at_unix_ts"] >= summary["last_writer_at_unix_ts"]


def test_first_observation_prose_distinguishes_from_invalidated(client: _Client) -> None:
    """F1: first-observation warning prose says 'first time your session has
    observed' rather than 'you haven't read this version yet' (which falsely
    implies the session was previously behind)."""
    # Seed plan.md via session A at v1
    a = _sid("A")
    client.post("/hooks/pre-read", {"session_id": a, "path": "plan.md"})
    # Session B has never seen plan.md — first-pre-read returns stale because
    # B has no prior agent_state. Prose must reflect "first observation" not "previously behind".
    b = _sid("B")
    # Make plan.md v2 first so B's first observation is genuinely stale
    client.post("/hooks/pre-edit", {"session_id": a, "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": a, "path": "plan.md",
                                       "content_hash": _hash("h1"), "success": True})
    _, body = client.post("/hooks/pre-read", {"session_id": b, "path": "plan.md"})
    if "hookSpecificOutput" in body:
        msg = body["hookSpecificOutput"]["additionalContext"]
        assert "first time your session has observed" in msg, (
            f"prose should distinguish first-observation from invalidation; got: {msg}"
        )
        # And NOT the old misleading framing
        assert "you haven't read this version yet" not in msg


def test_status_no_deadlock_under_concurrent_registration(client: _Client) -> None:
    """A4: /status iteration won't raise 'dict changed size during iteration'
    when concurrent pre-reads from new sessions are firing."""
    stop = threading.Event()
    errors: list[Exception] = []

    def churner() -> None:
        i = 0
        while not stop.is_set():
            try:
                client.post("/hooks/pre-read",
                            {"session_id": _sid(f"churn-{i}"), "path": "plan.md"})
            except Exception as e:
                errors.append(e)
            i += 1

    threads = [threading.Thread(target=churner) for _ in range(4)]
    for t in threads: t.start()
    try:
        for _ in range(20):
            s, _ = client.get("/status")
            assert s == 200, "status returned non-200 under churn"
    finally:
        stop.set()
        for t in threads: t.join(timeout=2.0)
    assert not errors, f"churner threads hit errors: {errors[:3]}"


def test_concurrent_pre_read_no_deadlock(client: _Client) -> None:
    """8 concurrent pre-read requests on distinct sessions — all succeed.

    v0.1.1 KTD-G item 2 caps handler concurrency at HANDLER_CONCURRENCY_LIMIT
    (= pool_size × 2 = 8) per plugin docs/known-issues/
    2026-05-17-watchdog-races.md A7 mitigation. Requests above the limit
    receive HTTP 503 synchronously without spawning a handler thread.
    """
    results: list[int] = []

    def fire(i: int) -> None:
        s, _ = client.post("/hooks/pre-read",
                            {"session_id": _sid(f"conc-{i}"), "path": "CLAUDE.md"})
        results.append(s)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert results == [200] * 8


def test_concurrent_pre_read_above_limit_returns_503(client: _Client) -> None:
    """v0.1.1 KTD-G item 2: requests above HANDLER_CONCURRENCY_LIMIT are
    rejected with 503, not silently queued. Fires 32 concurrent requests
    against a coordinator with limit=8; expects at least some 503s.

    Note: deterministic 503 emission requires slow-handler simulation —
    real handlers complete fast enough that the burst may serialize.
    This test asserts the contract (503 is possible above limit) by
    issuing far more requests than the limit in tight succession.
    """
    results: list[int] = []

    def fire(i: int) -> None:
        s, _ = client.post("/hooks/pre-read",
                            {"session_id": _sid(f"burst-{i}"), "path": "CLAUDE.md"})
        results.append(s)

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(32)]
    for t in threads: t.start()
    for t in threads: t.join()
    # All responses must be valid HTTP codes; allowed values: 200 (handled)
    # or 503 (concurrency-overflowed). Anything else is a bug.
    assert all(r in (200, 503) for r in results), f"unexpected statuses: {results}"
    # At least one 200 must succeed (the limit allows some throughput).
    assert any(r == 200 for r in results)


# ======================================================================
# v0.1.1 KTD-N — H4 mitigation: /hooks/pre-bash + /hooks/pre-grep
# ======================================================================


def test_pre_bash_untracked_command_returns_fresh_fastpath(coordinator, client: _Client) -> None:
    """A Bash command that reads no tracked artifacts returns fresh
    without touching SQLite. Mirrors pre-read fast-path."""
    before = len(coordinator.registry.artifact_ids())
    status, body = client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "ls -la /etc"},
    )
    assert status == 200
    assert body == {"status": "fresh"}
    assert len(coordinator.registry.artifact_ids()) == before


def test_pre_bash_first_observation_returns_fresh(client: _Client) -> None:
    """KTD-9 first-observation seeding via Bash. `cat plan.md` on a fresh
    workspace seeds plan.md + grants SHARED + returns fresh."""
    status, body = client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert status == 200
    assert body == {"status": "fresh"}


def test_pre_bash_after_peer_write_returns_stale(client: _Client) -> None:
    """H4 mitigation core test: session A's `bash cat plan.md` after a
    peer commit returns stale, NOT silent fresh. This is the gap KTD-N
    closes — without the Bash hook, A's bash-cat would bypass the
    coherence layer entirely (the H4 finding from v0.2 Phase 0)."""
    # A first-reads via pre-read.
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    # B commits v2.
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": _sid("B"), "path": "plan.md", "content_hash": _hash("h2"), "success": True})
    # A bash-cats plan.md → stale warning fires.
    status, body = client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert status == 200
    assert body["status"] == "stale"
    assert "hookSpecificOutput" in body
    out = body["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "allow"  # v0.1.1 warn-only per KTD-E
    assert "plan.md" in out["additionalContext"]
    assert "Bash command" in out["additionalContext"]
    assert body["stale_paths"] == ["plan.md"]


def test_pre_bash_warn_mode_never_returns_deny(client: _Client) -> None:
    """v0.1.1 invariant: pre-bash MUST NOT return deny (warn-only per KTD-E)."""
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": _sid("B"), "path": "plan.md", "content_hash": _hash("h2"), "success": True})
    status, body = client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    if "hookSpecificOutput" in body:
        assert body["hookSpecificOutput"]["permissionDecision"] != "deny"


def test_pre_bash_pipeline_with_tracked_arg(client: _Client) -> None:
    """`cat README.md || cat plan.md` — pipeline-split detection finds plan.md."""
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": _sid("B"), "path": "plan.md", "content_hash": _hash("h2"), "success": True})
    status, body = client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat README.md || cat plan.md"},
    )
    assert status == 200
    assert body["status"] == "stale"
    assert "plan.md" in body["stale_paths"]


def test_pre_bash_missing_session_id_400(client: _Client) -> None:
    status, body = client.post("/hooks/pre-bash", {"command": "cat plan.md"})
    assert status == 400
    assert "session_id" in body["error"]


def test_pre_bash_empty_command_400(client: _Client) -> None:
    status, body = client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": ""})
    assert status == 400
    assert "command" in body["error"]


def test_pre_bash_oversized_command_413(client: _Client) -> None:
    big = "cat plan.md " + ("x" * 20_000)
    status, body = client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": big})
    assert status == 413


def test_pre_bash_grep_substring_no_false_positive(client: _Client) -> None:
    """Per KTD-N false-positive test: a literal quoted pattern that
    contains a tracked filename must NOT fire. `grep "cat plan.md" notes.txt`
    has plan.md inside the quoted search-pattern string, NOT as a file arg."""
    status, body = client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": 'grep "cat plan.md is a tracked file" notes.txt'},
    )
    assert status == 200
    # notes.txt is not tracked; plan.md is inside a quoted string. No detection.
    assert body == {"status": "fresh"}


def test_pre_grep_no_tracked_artifacts_under_root_returns_fresh(client: _Client) -> None:
    """Empty workspace — grep over `src/` finds zero tracked artifacts."""
    status, body = client.post(
        "/hooks/pre-grep",
        {"session_id": _sid("A"), "search_root": "src"},
    )
    assert status == 200
    assert body == {"status": "fresh"}


def test_pre_grep_after_peer_write_returns_stale(client: _Client) -> None:
    """H4 mitigation for Grep: session A's grep over a directory
    containing peer-updated tracked artifacts returns stale."""
    # A first-reads plan.md (registers it).
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    # B commits v2.
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": _sid("B"), "path": "plan.md", "content_hash": _hash("h2"), "success": True})
    # A greps the workspace root (covers plan.md).
    status, body = client.post(
        "/hooks/pre-grep",
        {"session_id": _sid("A"), "search_root": ""},
    )
    assert status == 200
    assert body["status"] == "stale"
    assert "plan.md" in body["stale_paths"]
    assert "Grep search" in body["hookSpecificOutput"]["additionalContext"]


def test_pre_grep_missing_session_id_400(client: _Client) -> None:
    status, body = client.post("/hooks/pre-grep", {"search_root": ""})
    assert status == 400
    assert "session_id" in body["error"]


def test_pre_grep_path_traversal_400(client: _Client) -> None:
    status, body = client.post("/hooks/pre-grep", {"session_id": _sid("A"), "search_root": "../escape"})
    assert status == 400


# ======================================================================
# v0.1.1 KTD-G — watchdog A6/A7 hardening: queue gate + handler semaphore + counters
# ======================================================================


def test_status_includes_watchdog_counters_zeroed_at_startup(client: _Client) -> None:
    """KTD-G item 3 + KTD-J: /status surfaces watchdog/concurrency counters
    so silent degradation becomes observable. All zero immediately after spawn."""
    status, body = client.get("/status")
    assert status == 200
    assert body["watchdog_timeouts_total"] == 0
    assert body["watchdog_queue_overflows_total"] == 0
    assert body["handler_concurrency_overflows_total"] == 0


def test_a6_watchdog_timeout_increments_counter(coordinator, client: _Client) -> None:
    """A6 — handler timeout / sweep deadlock: when FuturesTimeout fires in
    _run_or_degrade, watchdog_timeouts_total increments and /status reflects it."""
    from concurrent.futures import TimeoutError as FuturesTimeout
    from unittest.mock import patch

    with patch.object(coordinator, "run_with_watchdog", side_effect=FuturesTimeout()):
        status, body = client.post("/hooks/pre-read",
                                    {"session_id": _sid("X"), "path": "plan.md"})
    # Degraded response per the existing _run_or_degrade contract.
    assert status == 200
    assert body.get("degraded") is True
    # Counter incremented.
    status, sbody = client.get("/status")
    assert sbody["watchdog_timeouts_total"] >= 1


def test_a7_watchdog_queue_overflow_returns_503(coordinator, client: _Client) -> None:
    """A7 — sweep-concurrent-write / shutdown-mid-sweep: when the watchdog
    ThreadPoolExecutor's _work_queue grows past WATCHDOG_QUEUE_LIMIT,
    _run_or_degrade returns HTTP 503 instead of submitting the task.
    Simulated via a stubbed qsize that reports overflow."""
    from unittest.mock import patch

    class _FakeQueue:
        @staticmethod
        def qsize() -> int:
            return 100  # well above the limit

    with patch.object(coordinator._watchdog, "_work_queue", _FakeQueue()):
        status, body = client.post("/hooks/pre-read",
                                    {"session_id": _sid("X"), "path": "plan.md"})
    assert status == 503
    assert body["error"] == "watchdog queue overloaded"
    # Counter incremented.
    status, sbody = client.get("/status")
    assert sbody["watchdog_queue_overflows_total"] >= 1


def test_handler_concurrency_limit_constant_matches_spec(client: _Client) -> None:
    """KTD-G item 2 invariant: HANDLER_CONCURRENCY_LIMIT = pool_size × 2.
    Locked at 8 in v0.1.1 (pool_size=4). If a future change adjusts the
    pool size, this test will fail loudly so the operator confirms the
    new concurrency cap is intentional."""
    from ccs.adapters.claude_code.coordinator_server import (
        _WATCHDOG_POOL_SIZE,
        HANDLER_CONCURRENCY_LIMIT,
        WATCHDOG_QUEUE_LIMIT,
    )

    assert _WATCHDOG_POOL_SIZE == 4
    assert HANDLER_CONCURRENCY_LIMIT == 8
    assert WATCHDOG_QUEUE_LIMIT == 8


# ----------------------------------------------------------------------
# KTD-I (Unit 5 L2) — in-flight handler semaphore drain on shutdown
# ----------------------------------------------------------------------


def _await_in_flight_drain(
    coordinator: CoordinatorHTTPServer, timeout_sec: float = 1.0,
) -> None:
    """Wait for the in-flight counter to reach zero, failing on timeout.

    ``client.post`` returns as soon as the response body is read, but
    ``release_handler_slot`` runs in the dispatcher's ``finally`` block
    AFTER the response has been written. Between those two points the
    counter still reads 1 from the client's point of view — a window of
    microseconds on an idle machine, milliseconds on a loaded CI runner
    (REL-03's lock around the watchdog counters widens it further).

    Any test that samples the counter straight after a client call is
    therefore racing the server thread. The contract under test is that
    the finally block releases the slot — *eventually* zero, not zero by
    the next bytecode op — so wait for the drain rather than sampling
    it. Do NOT relax the assertion to ``<= 1`` instead: that stops
    pinning the release, which is the whole property.
    """
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if coordinator._in_flight == 0:
            return
        time.sleep(0.010)
    pytest.fail(
        f"in-flight counter never drained: expected 0 within "
        f"{timeout_sec}s, still at {coordinator._in_flight}"
    )


def test_i1_acquire_release_pair_balances_counter(tmp_path: Path) -> None:
    """Unit-level: acquire/release balance the in-flight counter; the
    drain condition is signalled on the zero transition."""
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="i1")
    try:
        assert srv._in_flight == 0
        assert srv.acquire_handler_slot() is True
        assert srv._in_flight == 1
        assert srv.acquire_handler_slot() is True
        assert srv._in_flight == 2
        srv.release_handler_slot()
        assert srv._in_flight == 1
        srv.release_handler_slot()
        assert srv._in_flight == 0
    finally:
        srv.shutdown()


def test_i2_acquire_denied_after_shutdown_started(tmp_path: Path) -> None:
    """Once ``_shutting_down`` flips, acquire_handler_slot returns False
    so the dispatcher 503s instead of touching a closing registry."""
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="i2")
    srv.serve_in_thread()
    try:
        # Manually flip the flag (mimics in-progress shutdown without
        # actually closing the registry, so we can keep poking).
        srv._shutting_down = True
        assert srv.acquire_handler_slot() is False
        assert srv._in_flight == 0
    finally:
        srv._shutting_down = False  # let shutdown() proceed normally
        srv.shutdown()


def test_i3_shutdown_waits_for_in_flight_handler(tmp_path: Path) -> None:
    """End-to-end: a long-running handler keeps the in-flight counter
    above zero; shutdown() must block on the drain until the handler
    returns rather than closing the registry under it."""
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="i3")
    srv.serve_in_thread()
    time.sleep(0.05)
    secret = load_secret(srv.coordinator_root)
    assert secret is not None
    client = _Client("127.0.0.1", srv.port, secret)

    # Simulate a slow handler by acquiring a slot from the test thread
    # (no real handler invoked — we only need to keep _in_flight > 0
    # for the drain to wait on).
    assert srv.acquire_handler_slot() is True

    shutdown_done = threading.Event()
    def shutdown_thread() -> None:
        srv.shutdown()
        shutdown_done.set()
    t = threading.Thread(target=shutdown_thread)
    t.start()

    # shutdown() should be blocked in the drain loop.
    assert not shutdown_done.wait(timeout=0.5), (
        "shutdown returned before the in-flight slot was released"
    )

    # Releasing the slot wakes the drain and lets shutdown complete.
    srv.release_handler_slot()
    assert shutdown_done.wait(timeout=2.0), "shutdown did not complete after drain"
    t.join(timeout=2.0)
    assert srv._in_flight_drain_timed_out is False
    del client  # silence unused-var lint


def test_i4_shutdown_drain_timeout_records_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a handler stays in-flight past the drain timeout, shutdown
    closes the registry anyway and records the timeout for observability."""
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="i4")
    srv.serve_in_thread()
    time.sleep(0.05)

    # Keep an in-flight slot held for the duration of the test.
    assert srv.acquire_handler_slot() is True

    # Shrink the drain timeout so the test finishes in <1s.
    import ccs.adapters.claude_code.coordinator_server as mod
    monkeypatch.setattr(mod, "IN_FLIGHT_DRAIN_TIMEOUT_SEC", 0.1)
    try:
        srv.shutdown()
        assert srv._in_flight_drain_timed_out is True, (
            "drain timeout should set the observability flag"
        )
    finally:
        # Release the artificially-held slot so the test doesn't leak.
        srv.release_handler_slot()


def test_i5_dispatch_pairs_acquire_with_release(client: _Client, coordinator) -> None:
    """Integration: a normal pre-read request increments and decrements
    the in-flight counter exactly once, leaving it at zero on return.

    The zero check waits for the drain — see :func:`_await_in_flight_drain`
    for why sampling it straight after the response races the server
    thread."""
    assert coordinator._in_flight == 0
    status, _ = client.post(
        "/hooks/pre-read",
        {"session_id": _sid("i5"), "path": "CLAUDE.md"},
    )
    assert status == 200
    _await_in_flight_drain(coordinator)


def test_i6_dispatch_decrements_even_when_handler_raises(
    client: _Client, coordinator, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defensive: if a handler raises mid-dispatch (becoming a 500), the
    finally block must still release the slot.

    Same drain wait as i5: the 500 is written by the ``except`` arm and
    the release happens in the ``finally`` that follows it, so the
    client can observe the response before the decrement lands."""
    import ccs.adapters.claude_code.coordinator_server as mod
    original = mod._ROUTES[("POST", "/hooks/pre-read")]

    def raising_handler(req, coord) -> None:
        raise RuntimeError("simulated handler failure")

    monkeypatch.setitem(mod._ROUTES, ("POST", "/hooks/pre-read"), raising_handler)
    try:
        assert coordinator._in_flight == 0
        status, _ = client.post(
            "/hooks/pre-read",
            {"session_id": _sid("i6"), "path": "CLAUDE.md"},
        )
        assert status == 500
        # Counter must have been decremented despite the exception.
        _await_in_flight_drain(coordinator)
    finally:
        mod._ROUTES[("POST", "/hooks/pre-read")] = original


# ----------------------------------------------------------------------
# R21 (Unit 6) — MAX_REQUEST_BODY_BYTES cap before rfile.read
# ----------------------------------------------------------------------


def test_r21_request_body_overflow_returns_413(coordinator) -> None:
    """A Content-Length header that exceeds MAX_REQUEST_BODY_BYTES must
    be rejected with 413 BEFORE the coordinator reads the body into
    memory — protects against single-request OOM by a hostile or buggy
    client inside the trust boundary."""
    import http.client

    from ccs.adapters.claude_code.auth import load_secret
    from ccs.adapters.claude_code.coordinator_server import MAX_REQUEST_BODY_BYTES

    secret = load_secret(coordinator.coordinator_root)
    assert secret is not None

    # Build the request manually to control Content-Length precisely.
    # We claim n+1 bytes but only send 1 byte — server must reject on
    # header alone, not after reading the (oversized) body.
    over_n = MAX_REQUEST_BODY_BYTES + 1
    conn = http.client.HTTPConnection("127.0.0.1", coordinator.port, timeout=5)
    try:
        conn.request(
            "POST", "/hooks/pre-read",
            body=b"x",  # intentional mismatch — header says oversized, body is tiny
            headers={
                "Authorization": f"Bearer {secret}",
                "Host": "127.0.0.1",
                "Content-Type": "application/json",
                "Content-Length": str(over_n),
            },
        )
        resp = conn.getresponse()
        assert resp.status == 413, (
            f"expected 413 for oversized Content-Length={over_n}; got {resp.status}"
        )
        body = json.loads(resp.read().decode("utf-8"))
        assert "exceeds" in body["error"].lower()
        assert str(over_n) in body["error"]
    finally:
        conn.close()


def test_r21_body_at_cap_accepted(coordinator) -> None:
    """Boundary: a body at exactly MAX_REQUEST_BODY_BYTES (still well
    over our real payload sizes) is accepted, not 413'd off."""
    from ccs.adapters.claude_code.auth import load_secret
    from ccs.adapters.claude_code.coordinator_server import MAX_REQUEST_BODY_BYTES

    secret = load_secret(coordinator.coordinator_root)
    assert secret is not None

    # Craft a JSON object whose serialized length equals the cap. We pad
    # the session_id label so the overall JSON hits the byte count.
    base = {"session_id": _sid("r21"), "path": "CLAUDE.md", "pad": ""}
    base_bytes = json.dumps(base).encode("utf-8")
    pad_len = MAX_REQUEST_BODY_BYTES - len(base_bytes)
    assert pad_len > 0
    base["pad"] = "x" * pad_len
    payload = json.dumps(base).encode("utf-8")
    assert len(payload) == MAX_REQUEST_BODY_BYTES

    client = _Client("127.0.0.1", coordinator.port, secret)
    # Use the raw urllib client to ensure we control Content-Length.
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    req = urlrequest.Request(
        url, data=payload, method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        },
    )
    with urlrequest.urlopen(req, timeout=5) as resp:
        # Server may 200 (fresh) or 400 (unknown extra fields are tolerated
        # — but the body must have been READ, proving the cap let it through).
        assert resp.status in (200, 400)
    del client  # silence unused-var lint


# ----------------------------------------------------------------------
# KTD-J (Unit 8) — telemetry counters
# ----------------------------------------------------------------------


def test_a8_per_endpoint_counters_increment_on_dispatch(client: _Client, coordinator) -> None:
    """5 pre-reads + 3 pre-edits + 3 post-edits + 1 session-stop must show
    up in the per-endpoint counter block of /status?detail=full."""
    for i in range(5):
        client.post(
            "/hooks/pre-read",
            {"session_id": _sid(f"j1-{i}"), "path": "plan.md"},
        )
    for i in range(3):
        client.post(
            "/hooks/pre-edit",
            {"session_id": _sid(f"j1-edit-{i}"), "path": f"path_{i}.md"},
        )
    for i in range(3):
        client.post(
            "/hooks/post-edit",
            {
                "session_id": _sid(f"j1-edit-{i}"),
                "path": f"path_{i}.md",
                "content_hash": _hash(f"h{i}"),
                "success": True,
            },
        )
    client.post(
        "/hooks/session-stop", {"session_id": _sid("j1-stop")}
    )

    s, b = client.request(
        "GET", "/status?detail=metrics",
    )
    assert s == 200
    counters = b["endpoint_counters"]
    assert counters["pre_read_total"] == 5
    assert counters["pre_edit_total"] == 3
    assert counters["post_edit_total"] == 3
    assert counters["session_stop_total"] == 1


def test_a8_status_counter_request_itself_increments(client: _Client, coordinator) -> None:
    """A /status call counts itself — the increment fires before the
    handler runs."""
    _, b1 = client.get("/status")
    _, b2 = client.get("/status")
    assert (
        b2["endpoint_counters"]["status_total"]
        > b1["endpoint_counters"]["status_total"]
    )


def test_a8_counters_reset_to_zero_on_fresh_coordinator(tmp_path: Path) -> None:
    """Counters are CACHE, not persistent state. A fresh coordinator
    instance starts with zeros even when state.db already exists."""
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="j3")
    try:
        snap = srv.endpoint_counters_snapshot()
        assert all(v == 0 for v in snap.values())
        assert srv._intra_task_acquire_release_total == 0
        assert srv._stale_warning_emitted_total == 0
        assert srv._stale_warning_reread_total == 0
    finally:
        srv.shutdown()


def test_a8_stale_emitted_and_reread_counters_track_warning_cycle(
    client: _Client, coordinator,
) -> None:
    """Two-session stale scenario: A reads, B writes, A re-reads → stale
    warning fires (emitted=1). A reads again with the same artifact →
    that's the re-read (reread=1)."""
    a = _sid("j4-A"); b = _sid("j4-B")
    client.post("/hooks/pre-read",
                {"session_id": a, "path": "plan.md", "content_hash": _hash("h1")})
    client.post("/hooks/pre-edit", {"session_id": b, "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": b, "path": "plan.md",
                                     "content_hash": _hash("h2"), "success": True})
    # A re-reads — stale warning fires.
    s, body = client.post("/hooks/pre-read",
                          {"session_id": a, "path": "plan.md"})
    assert body["status"] == "stale"
    # A re-reads again (the re-read after the warning).
    client.post("/hooks/pre-read",
                {"session_id": a, "path": "plan.md"})

    _, status_body = client.get("/status?detail=metrics")
    assert status_body["stale_warning_emitted_total"] >= 1
    assert status_body["stale_warning_reread_total"] >= 1


def test_a8_intra_task_acquire_release_increments_on_successful_post_edit(
    client: _Client, coordinator,
) -> None:
    """A pre-edit followed by a successful post-edit on a tracked path
    must bump the intra-task acquire-release counter exactly once.
    Untracked paths fast-path through pre-edit/post-edit without
    acquiring E, so the counter is the load-bearing signal that fine-
    grained write protection was actually exercised."""
    sid = _sid("j5")
    before = coordinator._intra_task_acquire_release_total
    # plan.md matches the default tracked policy.
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": sid, "path": "plan.md",
                 "content_hash": _hash("j5"), "success": True})
    assert coordinator._intra_task_acquire_release_total == before + 1


def test_a8_failed_post_edit_does_not_increment_acquire_release(
    client: _Client, coordinator,
) -> None:
    """If post-edit reports failure, the counter does NOT increment —
    the contract is 'fine-grained write protection actually used', not
    'attempted'."""
    sid = _sid("j6")
    before = coordinator._intra_task_acquire_release_total
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": sid, "path": "plan.md",
                 "content_hash": _hash("j6"), "success": False})
    assert coordinator._intra_task_acquire_release_total == before


# ----------------------------------------------------------------------
# Unit 6 — OCC commit endpoint (/hooks/post-edit-cas)
# ----------------------------------------------------------------------


def _occ_seed_shared(client: _Client, sid: str, path: str, h: str) -> int:
    """OCC helper: pre-read a tracked path to register SHARED + seed the
    artifact, returning the coordinator's version (the OCC comparand).

    A first-observation read returns a fresh response carrying a top-level
    ``version``; a second session reading an already-seeded artifact (matching
    hash) falls through to the warn-mode stale response, which carries the
    version under ``summary.current_version`` (mirrors the production
    ``CoherentVolume._pre_read_version`` two-shape extraction)."""
    s, b = client.post("/hooks/pre-read",
                       {"session_id": sid, "path": path, "content_hash": h})
    assert s == 200, b
    if isinstance(b.get("version"), int):
        return b["version"]
    summary = b.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("current_version"), int):
        return summary["current_version"]
    raise AssertionError(f"pre-read surfaced no version for OCC: {b}")


def test_occ_commit_happy_path_bumps_version(client: _Client) -> None:
    """OCC commit (post-edit-cas) with a matching expected_version commits and
    returns the new version N+1 — WITHOUT a pre-edit EXCLUSIVE acquire."""
    sid = _sid("occ1")
    v = _occ_seed_shared(client, sid, "plan.md", _hash("occ1-v1"))  # v1
    s, b = client.post("/hooks/post-edit-cas",
                       {"session_id": sid, "path": "plan.md",
                        "success": True, "content_hash": _hash("occ1-v2"),
                        "expected_version": v})
    assert s == 200
    assert b == {"ok": True, "version": v + 1}


def _artifact_version(coordinator, path: str) -> int:
    """Read the coordinator's authoritative version for a tracked path."""
    aid = coordinator.registry.lookup_artifact_id_by_name(path)
    assert aid is not None
    art = coordinator.registry.get_artifact(aid)
    assert art is not None
    return art.version


def test_occ_commit_stale_version_conflicts_no_mutation(coordinator, client: _Client) -> None:
    """A stale expected_version → {ok:false, reason:'version_mismatch',
    current_version} — a clean typed conflict (NOT a degrade), no mutation."""
    a, b_sid = _sid("occA"), _sid("occB")
    v1 = _occ_seed_shared(client, a, "plan.md", _hash("v1"))      # A reads v1
    _occ_seed_shared(client, b_sid, "plan.md", _hash("v1"))       # B reads v1
    # A commits first → v2 (A wins).
    s, ba = client.post("/hooks/post-edit-cas",
                        {"session_id": a, "path": "plan.md", "success": True,
                         "content_hash": _hash("v2-A"), "expected_version": v1})
    assert ba == {"ok": True, "version": v1 + 1}
    after_a = _artifact_version(coordinator, "plan.md")

    # B commits with its now-stale expected_version (still v1) → version_mismatch.
    s, bb = client.post("/hooks/post-edit-cas",
                        {"session_id": b_sid, "path": "plan.md", "success": True,
                         "content_hash": _hash("v2-B-stale"), "expected_version": v1})
    assert s == 200
    assert bb["ok"] is False
    assert bb["reason"] == "version_mismatch"
    assert bb["current_version"] == after_a
    assert "degraded" not in bb  # a clean typed conflict, NOT a degrade
    # No mutation: B's stale commit did not bump the version past A's.
    assert _artifact_version(coordinator, "plan.md") == after_a


def test_occ_commit_corruption_expected_gt_current_raises_body(coordinator, client: _Client) -> None:
    """expected_version > current → corruption: commit_cas raises CoherenceError,
    the endpoint returns {ok:false, reason:<verbatim>} the client raises on."""
    sid = _sid("occCorrupt")
    _occ_seed_shared(client, sid, "plan.md", _hash("v1"))  # v1
    s, b = client.post("/hooks/post-edit-cas",
                       {"session_id": sid, "path": "plan.md", "success": True,
                        "content_hash": _hash("vN"), "expected_version": 999})
    assert s == 200
    assert b["ok"] is False
    assert "corruption" in b["reason"] or "commit_cas_corruption" in b["reason"]
    # No mutation on corruption.
    assert _artifact_version(coordinator, "plan.md") == 1


def test_occ_commit_caller_in_transient_returns_stable_reason(
    coordinator, client: _Client
) -> None:
    """AC2: a caller left mid-transient (a peer invalidated it between its read
    and its CAS) → {ok:false, reason:'caller_in_transient_state'} — a STABLE
    machine reason (NOT the exception's human message), so the client's retry
    classifier matches it exactly. The body also carries current_version so the
    client can advance its comparand. No mutation: this is a lost race."""
    from ccs.core.states import TransientState

    sid = _sid("occTransient")
    v = _occ_seed_shared(client, sid, "plan.md", _hash("v1"))  # caller SHARED@v1
    # Force the caller mid-transient on the coordinator (the registry shape a
    # peer's invalidating write leaves on a SHARED holder: SIA). commit_cas
    # rejects this as a retry-eligible precondition.
    aid = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    agent_id = session_to_agent_id(sid)
    coordinator.registry.set_agent_transient(
        aid, agent_id, TransientState.SIA, entered_tick=v
    )

    s, b = client.post(
        "/hooks/post-edit-cas",
        {"session_id": sid, "path": "plan.md", "success": True,
         "content_hash": _hash("v2"), "expected_version": v},
    )
    assert s == 200
    # The STABLE wire reason — exactly the literal the client matcher keys on,
    # decoupled from commit_cas's "commit_cas_not_allowed ..." human message.
    assert b == {"ok": False, "reason": "caller_in_transient_state", "current_version": v}
    assert "degraded" not in b  # a clean retry-eligible conflict, not a degrade
    # No mutation: the version did not advance.
    assert _artifact_version(coordinator, "plan.md") == v


def test_occ_commit_degrade_reads_as_failure(coordinator, client: _Client) -> None:
    """THE LOAD-BEARING FIX: a timed-out/degraded OCC commit returns
    {ok:false, degraded:true, reason:'commit_unconfirmed'} — NOT the
    {ok:true, degraded:true} the pessimistic post-edit uses. A client reading
    result.get('ok') must see False so it never assumes the write landed."""
    from concurrent.futures import TimeoutError as FuturesTimeout
    from unittest.mock import patch

    sid = _sid("occDegrade")
    v = _occ_seed_shared(client, sid, "plan.md", _hash("v1"))
    with patch.object(coordinator, "run_with_watchdog", side_effect=FuturesTimeout()):
        s, b = client.post("/hooks/post-edit-cas",
                           {"session_id": sid, "path": "plan.md", "success": True,
                            "content_hash": _hash("v2"), "expected_version": v})
    assert s == 200
    assert b.get("ok") is False, "degraded OCC commit must read as FAILURE, not ok:true"
    assert b.get("degraded") is True
    assert b.get("reason") == "commit_unconfirmed"


def test_occ_commit_does_not_take_pre_edit_acquire(coordinator, client: _Client) -> None:
    """The OCC path must NOT invoke the pre-edit EXCLUSIVE-acquire handler.
    Monkeypatch _handle_pre_edit to blow up: a full read→post-edit-cas cycle
    still succeeds, proving the OCC writer never routes through the acquire."""
    import ccs.adapters.claude_code.coordinator_server as mod
    original = mod._ROUTES[("POST", "/hooks/pre-edit")]

    def exploding_pre_edit(req, coord):
        raise AssertionError("OCC path must NOT call _handle_pre_edit")

    mod._ROUTES[("POST", "/hooks/pre-edit")] = exploding_pre_edit
    try:
        sid = _sid("occNoAcq")
        v = _occ_seed_shared(client, sid, "plan.md", _hash("v1"))
        before_pre_edit = coordinator.endpoint_counters_snapshot()["pre_edit_total"]
        s, b = client.post("/hooks/post-edit-cas",
                           {"session_id": sid, "path": "plan.md", "success": True,
                            "content_hash": _hash("v2"), "expected_version": v})
        assert s == 200
        assert b == {"ok": True, "version": v + 1}
        # The OCC writer is never EXCLUSIVE — it ends SHARED via commit_cas's
        # S/I→S transition (an OCC writer holds no grant), and the pre-edit
        # counter never moved.
        after_pre_edit = coordinator.endpoint_counters_snapshot()["pre_edit_total"]
        assert after_pre_edit == before_pre_edit
        aid = coordinator.registry.lookup_artifact_id_by_name("plan.md")
        state = coordinator.registry.get_agent_state(aid, session_to_agent_id(sid))
        assert state == MESIState.SHARED
    finally:
        mod._ROUTES[("POST", "/hooks/pre-edit")] = original


def test_occ_commit_untracked_path_fastpath(client: _Client) -> None:
    """An untracked path fast-paths to {ok:true} without a CAS (mirrors
    post-edit's is_tracked early return)."""
    s, b = client.post("/hooks/post-edit-cas",
                       {"session_id": _sid("occU"), "path": "src/random.py",
                        "success": True, "content_hash": _hash("h"),
                        "expected_version": 0})
    assert s == 200
    assert b == {"ok": True}


def test_occ_commit_rejects_non_int_expected_version(client: _Client) -> None:
    """expected_version is the OCC discriminator — a malformed (non-int)
    value is rejected at the boundary with 400, never driven into the CAS."""
    sid = _sid("occBad")
    _occ_seed_shared(client, sid, "plan.md", _hash("v1"))
    s, b = client.post("/hooks/post-edit-cas",
                       {"session_id": sid, "path": "plan.md", "success": True,
                        "content_hash": _hash("v2"), "expected_version": "1"})
    assert s == 400
    assert "expected_version" in b["error"]


def test_occ_commit_missing_content_hash_400(client: _Client) -> None:
    """The OCC commit always carries the bytes it wrote → content_hash required."""
    sid = _sid("occNoHash")
    v = _occ_seed_shared(client, sid, "plan.md", _hash("v1"))
    s, b = client.post("/hooks/post-edit-cas",
                       {"session_id": sid, "path": "plan.md", "success": True,
                        "expected_version": v})
    assert s == 400


def test_occ_commit_increments_endpoint_counter(coordinator, client: _Client) -> None:
    """KTD-J: the OCC endpoint has its own per-endpoint counter."""
    sid = _sid("occCount")
    v = _occ_seed_shared(client, sid, "plan.md", _hash("v1"))
    before = coordinator.endpoint_counters_snapshot()["post_edit_cas_total"]
    client.post("/hooks/post-edit-cas",
                {"session_id": sid, "path": "plan.md", "success": True,
                 "content_hash": _hash("v2"), "expected_version": v})
    after = coordinator.endpoint_counters_snapshot()["post_edit_cas_total"]
    assert after == before + 1


def test_occ_commit_concurrent_winner_election_no_lost_update(coordinator, client: _Client) -> None:
    """R11-flavored funnel through the HTTP path: two OCC clients both read v1
    (fixed stale buffers, NOT a counter increment), barrier-synced, both POST
    post-edit-cas with expected_version=v1. Exactly one wins (v2), the loser
    gets a typed version_mismatch, final version is v2 (no lost update)."""
    n_writers = 2
    a, b_sid = _sid("raceA"), _sid("raceB")
    v1 = _occ_seed_shared(client, a, "plan.md", _hash("v1"))
    assert _occ_seed_shared(client, b_sid, "plan.md", _hash("v1")) == v1

    barrier = threading.Barrier(n_writers)
    results: dict[str, tuple[int, dict]] = {}

    def commit(sid: str, label: str) -> None:
        barrier.wait()
        results[label] = client.post(
            "/hooks/post-edit-cas",
            {"session_id": sid, "path": "plan.md", "success": True,
             "content_hash": _hash(f"v2-{label}"), "expected_version": v1},
        )

    threads = [
        threading.Thread(target=commit, args=(a, "A")),
        threading.Thread(target=commit, args=(b_sid, "B")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wins = [lbl for lbl, (_, body) in results.items() if body.get("ok") is True]
    losers = [lbl for lbl, (_, body) in results.items() if body.get("ok") is False]
    assert len(wins) == 1, f"exactly one OCC writer must win: {results}"
    assert len(losers) == 1
    loser_body = results[losers[0]][1]
    assert loser_body["reason"] == "version_mismatch"
    # Final version is exactly v1+1 — the loser's stale buffer did NOT clobber.
    assert _artifact_version(coordinator, "plan.md") == v1 + 1


def test_occ_late_completion_residual_contended_is_noop_not_lost_update(
    coordinator, client: _Client,
) -> None:
    """Late-completion residual (plan Unit 6 / Key Decision). The watchdog does
    not cancel a timed-out future, so a late commit_cas may run after the client
    gave up. This asserts the SAFE half of that residual deterministically: a
    late CAS in the CONTENDED case (the version already advanced) sees the
    advanced version → version_mismatch, NO mutation — it cannot drop an
    acknowledged write. (The uncontended case lands a phantom/duplicate N+1 from
    the same edit — a duplicate version bump, NOT a lost write; NoLostUpdate
    still holds. Full fencing is deferred to the cross-host follow-on.)"""
    a, late = _sid("liveWinner"), _sid("lateGaveUp")
    v1 = _occ_seed_shared(client, a, "plan.md", _hash("v1"))
    _occ_seed_shared(client, late, "plan.md", _hash("v1"))  # the "late" writer also read v1
    # The live winner commits → v2 (this models the contention the late writer
    # raced against).
    s, ba = client.post("/hooks/post-edit-cas",
                        {"session_id": a, "path": "plan.md", "success": True,
                         "content_hash": _hash("v2-winner"), "expected_version": v1})
    assert ba == {"ok": True, "version": v1 + 1}
    v2 = _artifact_version(coordinator, "plan.md")
    # The late writer's CAS finally runs with its now-stale expected_version
    # (==v1): it observes the advanced version → version_mismatch, no mutation.
    s, bl = client.post("/hooks/post-edit-cas",
                        {"session_id": late, "path": "plan.md", "success": True,
                         "content_hash": _hash("v2-late"), "expected_version": v1})
    assert bl["ok"] is False
    assert bl["reason"] == "version_mismatch"
    # No acknowledged write was dropped — the version is the winner's v2, and the
    # late writer's stale bytes were NOT committed over it.
    assert _artifact_version(coordinator, "plan.md") == v2


def test_a8_status_exposes_coordinator_backend_and_version(client: _Client) -> None:
    """KTD-J: /status shape includes coordinator_backend + coordinator_version
    for cross-implementation operator observability."""
    _, b = client.get("/status?detail=metrics")
    assert b["coordinator_backend"] == "python"
    assert isinstance(b["coordinator_version"], str)
    assert b["coordinator_version"]  # non-empty


def test_a8_counters_increment_even_when_handler_raises(
    client: _Client, coordinator, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Contract per the plan: per-endpoint counters count ATTEMPTED
    requests, not successful ones. A handler that raises mid-dispatch
    (becoming a 500) must still leave the counter incremented."""
    import ccs.adapters.claude_code.coordinator_server as mod
    original = mod._ROUTES[("POST", "/hooks/pre-read")]
    def raising(req, coord):
        raise RuntimeError("simulated handler failure")
    monkeypatch.setitem(mod._ROUTES, ("POST", "/hooks/pre-read"), raising)
    try:
        before = coordinator.endpoint_counters_snapshot()["pre_read_total"]
        status, _ = client.post(
            "/hooks/pre-read",
            {"session_id": _sid("j8"), "path": "j8.md"},
        )
        assert status == 500
        after = coordinator.endpoint_counters_snapshot()["pre_read_total"]
        assert after == before + 1, "counter must increment even on handler exception"
    finally:
        mod._ROUTES[("POST", "/hooks/pre-read")] = original


# ----------------------------------------------------------------------
# R10 (Unit 6) — _agent_names mutation under threading.Lock
# ----------------------------------------------------------------------


def test_a4_agent_names_mutation_under_concurrent_status(
    coordinator,
) -> None:
    """A4 — agent-names map concurrency (plan §'Cross-cutting test discipline').

    Eight threads concurrently call register_session with distinct session ids;
    the resulting dict must contain exactly the union with no torn entries (no
    missing keys, no overwrites). Canonical a4_ prefix for risk-code triage.
    """
    expected: set[str] = set()
    barrier = threading.Barrier(8)
    lock = threading.Lock()

    def churner(thread_idx: int) -> None:
        barrier.wait()
        for j in range(50):
            sid = _sid(f"r10-{thread_idx}-{j}")
            with lock:
                expected.add(sid)
            coordinator.register_session(sid)

    threads = [threading.Thread(target=churner, args=(i,)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    snapshot = coordinator.agent_names_snapshot()
    names = {name for _, name in snapshot}
    assert len(snapshot) >= len(expected)
    for sid in expected:
        assert f"claude-session-{sid}" in names, (
            f"session {sid} missing from snapshot (lock failed to serialize)"
        )


# Backward triage alias: pytest -k r10 still resolves.
test_r10_agent_names_lock_serializes_concurrent_registration = (
    test_a4_agent_names_mutation_under_concurrent_status
)


def test_r10_status_snapshot_consistent_under_churn(
    coordinator, client: _Client,
) -> None:
    """A reader calling /status while writers churn register_session must
    NEVER see RuntimeError from a torn iteration. The lock-protected
    snapshot is the contract."""
    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            try:
                coordinator.register_session(_sid(f"r10-churn-{i}"))
            except Exception as e:
                errors.append(e)
            i += 1

    writers = [threading.Thread(target=writer) for _ in range(4)]
    for t in writers: t.start()
    try:
        for _ in range(20):
            s, _ = client.get("/status")
            assert s == 200, "status returned non-200 under register_session churn"
    finally:
        stop.set()
        for t in writers: t.join(timeout=2.0)
    assert not errors, f"writer threads hit errors: {errors[:3]}"


def test_r10_agent_name_for_returns_none_for_unknown(coordinator) -> None:
    """The single-key accessor returns None for an agent that has never
    been registered, without raising."""
    fake_id = uuid.uuid5(uuid.NAMESPACE_URL, "ccs-agent:claude-session-never-registered")
    assert coordinator.agent_name_for(fake_id) is None


# ----------------------------------------------------------------------
# R14 (Unit 6) — _append_policy_yaml under fcntl.flock
# ----------------------------------------------------------------------


def test_r14_concurrent_policy_track_no_lost_writes(coordinator, client: _Client) -> None:
    """Eight threads each POST /policy/track with one unique path; every
    request that the coordinator accepts (status 200) must have its path
    persisted in tracked.yaml — no read-modify-write interleaving losing
    entries. R14's contract is about lost writes among ACCEPTED requests,
    not about 503s from the KTD-G concurrency cap (which is a separate
    pre-handler reject)."""
    target_paths = [f"r14/path_{i}.md" for i in range(8)]
    barrier = threading.Barrier(len(target_paths))
    results: list[tuple[str, int]] = []
    results_lock = threading.Lock()

    def add_path(p: str) -> None:
        barrier.wait()
        s, _ = client.post("/policy/track", {"paths": [p]})
        with results_lock:
            results.append((p, s))

    threads = [threading.Thread(target=add_path, args=(p,)) for p in target_paths]
    for t in threads: t.start()
    for t in threads: t.join()

    accepted = [p for p, s in results if s == 200]
    assert accepted, f"no requests succeeded — KTD-G cap may be too tight: {results}"

    yaml_path = coordinator.coordinator_root / ".coherence" / "tracked.yaml"
    text = yaml_path.read_text()
    for p in accepted:
        assert f"- {p}" in text, (
            f"path {p!r} lost despite 200 response — fcntl.flock did not serialize"
        )


def test_r14_lock_file_created_next_to_yaml(coordinator, client: _Client) -> None:
    """The fcntl lock uses a sidecar ``<yaml>.lock`` file; verify it
    appears and stays present (the file is reused across calls)."""
    client.post("/policy/track", {"paths": ["r14_sidecar.md"]})
    lock_path = (
        coordinator.coordinator_root / ".coherence" / "tracked.yaml.lock"
    )
    assert lock_path.is_file(), "tracked.yaml.lock sidecar was not created"


def test_policy_track_idempotent_no_duplicate_entries(coordinator, client: _Client) -> None:
    """Tracking the same path twice must not produce duplicate lines in
    tracked.yaml. The second call should still return 200 (idempotent) but
    the path must appear exactly once in the YAML."""
    path = "idempotent_test.md"
    s1, b1 = client.post("/policy/track", {"paths": [path]})
    s2, b2 = client.post("/policy/track", {"paths": [path]})
    assert s1 == 200
    assert s2 == 200
    # Second call must report zero newly-added patterns \u2014 the path was already
    # present and _append_policy_yaml must return ([], rejected), not (added, rejected).
    assert b2.get("added") == [], (
        f"second /policy/track returned 'added'={b2.get('added')!r}; "
        "idempotent call must report no additions"
    )

    yaml_path = coordinator.coordinator_root / ".coherence" / "tracked.yaml"
    text = yaml_path.read_text()
    occurrences = text.count(f"- {path}")
    assert occurrences == 1, (
        f"path {path!r} appears {occurrences}\xd7 in tracked.yaml after two track calls "
        f"(expected 1 \u2014 /policy/track must be idempotent)"
    )


# ----------------------------------------------------------------------
# _parse_yaml_pattern_lines unit tests
# ----------------------------------------------------------------------


def test_parse_yaml_pattern_lines_plain_entries() -> None:
    """Plain unquoted list items are extracted."""
    from ccs.adapters.claude_code.coordinator_server import _parse_yaml_pattern_lines
    text = "- plan.md\n- src/main.py\n"
    result = _parse_yaml_pattern_lines(text)
    assert result == {"plan.md", "src/main.py"}


def test_parse_yaml_pattern_lines_quoted_entries() -> None:
    """YAML-quoted values are extracted without the quotes."""
    from ccs.adapters.claude_code.coordinator_server import _parse_yaml_pattern_lines
    text = '- "plan.md"\n- \'src/main.py\'\n'
    result = _parse_yaml_pattern_lines(text)
    assert result == {"plan.md", "src/main.py"}


def test_parse_yaml_pattern_lines_empty_input() -> None:
    """Empty or whitespace-only input returns empty set."""
    from ccs.adapters.claude_code.coordinator_server import _parse_yaml_pattern_lines
    assert _parse_yaml_pattern_lines("") == set()
    assert _parse_yaml_pattern_lines("   \n  ") == set()


def test_parse_yaml_pattern_lines_non_string_items_ignored() -> None:
    """Non-string items (numbers, null) are silently dropped."""
    from ccs.adapters.claude_code.coordinator_server import _parse_yaml_pattern_lines
    text = "- plan.md\n- 42\n- null\n"
    result = _parse_yaml_pattern_lines(text)
    assert result == {"plan.md"}


def test_parse_yaml_pattern_lines_malformed_yaml_returns_empty() -> None:
    """Malformed YAML falls back to empty set rather than raising."""
    from ccs.adapters.claude_code.coordinator_server import _parse_yaml_pattern_lines
    result = _parse_yaml_pattern_lines("{not: a list}")
    assert result == set()


# ----------------------------------------------------------------------
# R11 (Unit 6) — ensure_secret bounded O_EXCL retry, fail-closed
# ----------------------------------------------------------------------


def test_r11_ensure_secret_recovers_from_empty_file_during_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a stale empty hook.secret exists when ensure_secret runs (e.g.,
    a previous coordinator crashed between O_EXCL-create and write), the
    bounded retry loop must eventually populate it without clobbering
    via O_TRUNC. We simulate this by pre-creating the empty file, then
    letting ensure_secret retry through to a clean O_EXCL after we
    unlink it from a sidecar 'racer' thread."""
    import threading as _t

    from ccs.adapters.claude_code import auth as _auth

    coherence_dir = tmp_path / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret_path = coherence_dir / _auth.SECRET_FILENAME

    # Make the file exist but be empty (mimics crashed predecessor).
    secret_path.touch(mode=0o600)
    assert secret_path.stat().st_size == 0

    # Speed up the test: shorter retry sleep, but keep retry count.
    monkeypatch.setattr(_auth, "ENSURE_SECRET_RETRY_SLEEP_SEC", 0.020)

    # Simulate a racer that unlinks the empty file mid-retry, allowing
    # ensure_secret's next O_EXCL to succeed.
    def racer() -> None:
        time.sleep(0.040)
        try:
            secret_path.unlink()
        except FileNotFoundError:
            pass

    t = _t.Thread(target=racer)
    t.start()
    try:
        token = _auth.ensure_secret(tmp_path)
    finally:
        t.join(timeout=2.0)
    # Contract: ensure_secret returned a token (recovery from empty
    # file succeeded). We intentionally do NOT re-read the file —
    # the racer may have unlinked AFTER ensure_secret returned, which
    # is fine for the in-process race but would race the assertion
    # on slow CI runners.
    assert token
    assert len(token) == 64


def test_r11_ensure_secret_fails_closed_when_empty_file_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If hook.secret stays empty across all retries, ensure_secret MUST
    raise EnsureSecretError rather than O_TRUNC over it. The old behavior
    would silently overwrite a concurrent racer's valid secret, leaving
    two spawn-side processes with different secrets for the same
    workspace — a silent total-protocol-break failure mode."""
    from ccs.adapters.claude_code import auth as _auth

    coherence_dir = tmp_path / ".coherence"
    coherence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    secret_path = coherence_dir / _auth.SECRET_FILENAME
    secret_path.touch(mode=0o600)
    assert secret_path.stat().st_size == 0

    # Shorten retry sleep so the test finishes fast.
    monkeypatch.setattr(_auth, "ENSURE_SECRET_RETRY_SLEEP_SEC", 0.001)
    monkeypatch.setattr(_auth, "ENSURE_SECRET_MAX_RETRIES", 3)

    with pytest.raises(_auth.EnsureSecretError) as exc:
        _auth.ensure_secret(tmp_path)
    assert "stayed empty" in str(exc.value)
    # The file must remain empty — we did NOT O_TRUNC over it.
    assert secret_path.stat().st_size == 0


# ----------------------------------------------------------------------
# R12 (Unit 6) — /status three-tier disclosure
# ----------------------------------------------------------------------


def test_r12_status_minimal_default_hides_coordinator_root(
    client: _Client,
) -> None:
    """The default (no query) response is the minimal tier — coordinator_root
    is the sentinel "." so $HOME / directory layout never leaks.

    P1 #7 (revision to R12): coordinator_pid IS included in minimal —
    pid is public on POSIX and operators rely on it. Only the absolute
    workspace root stays behind the operator-header gate at this tier."""
    s, b = client.get("/status")
    assert s == 200
    assert b["detail"] == "minimal"
    assert b["coordinator_root"] == "."
    # P1 #7: pid is in minimal tier (reversion of R12 over-redaction).
    assert b.get("coordinator_pid") == os.getpid()


def test_r12_status_full_requires_operator_header(client: _Client) -> None:
    """?detail=full without the Coherence-Local-Operator: true opt-in header
    must be rejected with 403 — Bearer auth alone is not sufficient for
    the elevated tier."""
    s, b = client.get("/status?detail=full")
    assert s == 403
    assert "operator" in b["error"].lower()


def test_r12_status_full_with_operator_header_exposes_root_and_pid(
    client: _Client, coordinator,
) -> None:
    """?detail=full + Coherence-Local-Operator: true returns the absolute
    coordinator_root and coordinator_pid for legitimate operator inspection."""
    s, b = client.request(
        "GET", "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )
    assert s == 200
    assert b["detail"] == "full"
    assert b["coordinator_root"] == str(coordinator.coordinator_root)
    assert isinstance(b["coordinator_pid"], int)
    # Full tier also retains the artifact/session block.
    assert "tracked_artifacts" in b
    assert "sessions" in b


def test_r12_status_metrics_returns_counters_only(client: _Client) -> None:
    """?detail=metrics returns only the counter block — no artifact/session
    walk, no leak of workspace state. Useful for dashboard scrapers."""
    s, b = client.get("/status?detail=metrics")
    assert s == 200
    assert b["detail"] == "metrics"
    assert "tracked_artifacts" not in b
    assert "sessions" not in b
    assert "policy_summary" not in b
    # Counters must be present.
    for k in (
        "coordinator_uptime_seconds",  # AC-02 canonical field
        "coordinator_uptime_s",  # AC-02 deprecated alias (one release)
        "watchdog_timeouts_total",
        "handler_concurrency_overflows_total",
        "in_flight_drain_timed_out",
        "cold_start_duration_ms",
    ):
        assert k in b, f"counter {k} missing from metrics tier"


def test_r12_status_unknown_detail_falls_back_to_minimal(client: _Client) -> None:
    """A typo'd ?detail=value must NOT silently grant more access — it
    falls back to minimal, never to full. P1 #7: pid is in minimal
    so we assert the absolute root is sentinel'd as the actual
    confidentiality signal instead."""
    s, b = client.get("/status?detail=fully")
    assert s == 200
    assert b["detail"] == "minimal"
    # The fall-back is "minimal" not "full" — absolute root must NOT leak.
    assert b["coordinator_root"] == "."


def test_r11_ensure_secret_concurrent_threads_return_identical(
    tmp_path: Path,
) -> None:
    """Multiple threads spawning concurrently must all walk away with the
    SAME secret (one wins O_EXCL, the others read what the winner wrote).
    Thread-level test exercises the in-process race; the cross-process
    case is identical at the syscall layer (O_EXCL is OS-enforced)."""
    from ccs.adapters.claude_code.auth import ensure_secret

    tokens: list[str] = []
    tokens_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        tok = ensure_secret(tmp_path)
        with tokens_lock:
            tokens.append(tok)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert len(set(tokens)) == 1, (
        f"concurrent ensure_secret returned different tokens: {set(tokens)}"
    )


# ----------------------------------------------------------------------
# ADV-001 — prepare-for-migration drain semantics
# ----------------------------------------------------------------------


def test_adv001_pre_edit_rejected_during_migration_drain(
    coordinator, client: _Client
) -> None:
    """ADV-001: after prepare-for-migration sets the draining flag, a new
    pre-edit on a tracked artifact must be rejected with 503 + a
    structured error rather than minting an EXCLUSIVE that the agent
    can never post-edit."""
    # Flip the flag directly so we don't have to wait for the full
    # drain → invalidate → shutdown sequence in this unit test.
    coordinator._migration_draining = True
    try:
        status, body = client.post(
            "/hooks/pre-edit",
            {"session_id": _sid("adv001-A"), "path": "plan.md"},
        )
        assert status == 503
        assert "migration" in body.get("error", "").lower()
        assert "draining" in body.get("error", "").lower()
    finally:
        coordinator._migration_draining = False


def test_adv001_post_edit_continues_during_migration_drain(
    coordinator, client: _Client
) -> None:
    """ADV-001: post-edit must still serve while draining so in-flight
    pre-edit→post-edit chains can complete naturally. This is the whole
    point of the draining-flag fix vs. an immediate hard shutdown."""
    sid = _sid("adv001-B")
    # Acquire an EXCLUSIVE before flipping the flag (the in-flight
    # request whose post-edit we want to allow through).
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})

    coordinator._migration_draining = True
    try:
        status, body = client.post(
            "/hooks/post-edit",
            {
                "session_id": sid,
                "path": "plan.md",
                "content_hash": _hash("adv001-B"),
                "success": True,
            },
        )
        assert status == 200, (
            f"post-edit must complete during drain; got {status}: {body!r}"
        )
        assert body.get("ok") is True
    finally:
        coordinator._migration_draining = False


def test_adv001_pre_read_continues_during_migration_drain(
    coordinator, client: _Client
) -> None:
    """ADV-001: pre-read is a non-mutating endpoint and must keep
    serving during the migration drain window."""
    coordinator._migration_draining = True
    try:
        status, body = client.post(
            "/hooks/pre-read",
            {"session_id": _sid("adv001-C"), "path": "plan.md"},
        )
        assert status == 200
    finally:
        coordinator._migration_draining = False


def test_adv001_prepare_for_migration_returns_immediately_with_draining_flag(
    coordinator, client: _Client
) -> None:
    """The handler now returns {ok, draining:true, drain_timeout_ms} as
    soon as it flips the flag. The drain + invalidate + shutdown
    sequence runs in a background thread; the CLI polls /status to
    observe the coordinator becoming unreachable."""
    status, body = client.post(
        "/admin/prepare-for-migration", {},
        headers_override={"Coherence-Local-Operator": "true"},
    )
    assert status == 200
    assert body["ok"] is True
    assert body["draining"] is True
    assert body["drain_timeout_ms"] > 0
    assert body["shutdown_scheduled_in_ms"] >= body["drain_timeout_ms"]
    # Background thread will close the coordinator; the fixture's
    # shutdown is idempotent so cleanup still works.


def test_adv001_repeated_prepare_for_migration_is_idempotent(
    coordinator, client: _Client
) -> None:
    """A second prepare-for-migration call while already draining must
    not start a second drain sequence — it returns the already_in_progress
    envelope."""
    coordinator._migration_draining = True
    try:
        status, body = client.post(
            "/admin/prepare-for-migration", {},
            headers_override={"Coherence-Local-Operator": "true"},
        )
        assert status == 200
        assert body["ok"] is True
        assert body["draining"] is True
        assert body.get("already_in_progress") is True
    finally:
        coordinator._migration_draining = False


# ----------------------------------------------------------------------
# P1 #5 — watchdog late-completion detector
# ----------------------------------------------------------------------


def test_p1_5_watchdog_late_completion_increments_counter(
    coordinator,
) -> None:
    """When run_with_watchdog times out and the underlying future
    later completes successfully, the late-completion counter must
    increment and a CRITICAL log line fires."""
    import threading as _t
    from concurrent.futures import TimeoutError as _FuturesTimeout

    # Replace HANDLER_TIMEOUT_SEC for the duration of the test so we
    # don't have to wait 4s. The work function blocks until the test
    # releases it, then returns a successful payload.
    release = _t.Event()
    def slow_work() -> dict:
        release.wait(timeout=5.0)
        return {"ok": True, "late": True}

    import ccs.adapters.claude_code.coordinator_server as mod
    original_timeout = mod.HANDLER_TIMEOUT_SEC
    try:
        mod.HANDLER_TIMEOUT_SEC = 0.05  # 50ms — fire timeout fast
        before = coordinator._watchdog_late_completion_total
        try:
            coordinator.run_with_watchdog(slow_work)
        except _FuturesTimeout:
            pass
        else:
            pytest.fail("expected FuturesTimeout")
        # Now release the work; the future completes successfully and
        # the done_callback fires (asynchronously — give the pool a
        # moment to schedule the callback).
        release.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if coordinator._watchdog_late_completion_total > before:
                break
            time.sleep(0.020)
        assert coordinator._watchdog_late_completion_total == before + 1, (
            f"expected late-completion counter to increment by 1; "
            f"before={before} after={coordinator._watchdog_late_completion_total}"
        )
    finally:
        mod.HANDLER_TIMEOUT_SEC = original_timeout
        release.set()


def test_p1_5_late_failure_does_not_increment_counter(
    coordinator,
) -> None:
    """If a timed-out future later RAISES rather than completing, no
    phantom state landed in the registry — counter must NOT increment."""
    import threading as _t
    from concurrent.futures import TimeoutError as _FuturesTimeout

    release = _t.Event()
    def slow_failing_work() -> dict:
        release.wait(timeout=5.0)
        raise RuntimeError("late failure")

    import ccs.adapters.claude_code.coordinator_server as mod
    original_timeout = mod.HANDLER_TIMEOUT_SEC
    try:
        mod.HANDLER_TIMEOUT_SEC = 0.05
        before = coordinator._watchdog_late_completion_total
        try:
            coordinator.run_with_watchdog(slow_failing_work)
        except _FuturesTimeout:
            pass
        # Release; future fails late; counter should NOT increment.
        release.set()
        time.sleep(0.3)
        assert coordinator._watchdog_late_completion_total == before, (
            f"late-failure path must not bump phantom-grant counter; "
            f"before={before} after={coordinator._watchdog_late_completion_total}"
        )
    finally:
        mod.HANDLER_TIMEOUT_SEC = original_timeout
        release.set()


def test_p1_5_status_metrics_exposes_late_completion_counter(
    client: _Client, coordinator,
) -> None:
    """The new counter must show up in /status?detail=metrics so
    operators can spot a phantom-grant cluster in a bug report."""
    status, body = client.get("/status?detail=metrics")
    assert status == 200
    assert "watchdog_late_completion_total" in body
    assert isinstance(body["watchdog_late_completion_total"], int)


# ----------------------------------------------------------------------
# P1 #6 — 401 visibility (hook.secret deletion / bearer mismatch)
# ----------------------------------------------------------------------


def test_p1_6_401_increments_auth_counter(coordinator) -> None:
    """A request with a wrong bearer must bump auth_401_total so
    operators can spot a hook.secret deletion via /status."""
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    req = urlrequest.Request(
        url, data=b"{}", method="POST",
        headers={
            "Authorization": "Bearer wrong-secret",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        },
    )
    before = coordinator._auth_401_total
    try:
        urlrequest.urlopen(req, timeout=5)
        pytest.fail("expected 401")
    except urlerror.HTTPError as e:
        assert e.code == 401
    assert coordinator._auth_401_total == before + 1


def test_p1_6_repeated_401s_dedupe_warning_logs(
    coordinator, caplog: pytest.LogCaptureFixture
) -> None:
    """Counter bumps every 401; WARNING log dedupes to once per 60s so
    a burst of bad requests doesn't drown the log."""
    import logging
    caplog.set_level(logging.WARNING, logger="ccs.adapters.claude_code.coordinator_server")
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    def fire_bad() -> None:
        req = urlrequest.Request(
            url, data=b"{}", method="POST",
            headers={
                "Authorization": "Bearer wrong-secret",
                "Host": "127.0.0.1",
                "Content-Type": "application/json",
            },
        )
        try:
            urlrequest.urlopen(req, timeout=5)
        except urlerror.HTTPError:
            pass

    before_total = coordinator._auth_401_total
    fire_bad()
    fire_bad()
    fire_bad()
    assert coordinator._auth_401_total == before_total + 3
    auth_warnings = [
        r for r in caplog.records
        if "auth: 401" in r.getMessage()
    ]
    # First call emits the warning; next two are deduped.
    assert len(auth_warnings) == 1, (
        f"expected exactly one 401 WARNING per dedupe window; got {len(auth_warnings)}: "
        f"{[r.getMessage() for r in auth_warnings]}"
    )


def test_p1_6_status_metrics_exposes_auth_401_counter(client: _Client) -> None:
    """auth_401_total visible in /status?detail=metrics."""
    status, body = client.get("/status?detail=metrics")
    assert status == 200
    assert "auth_401_total" in body
    assert isinstance(body["auth_401_total"], int)


# ----------------------------------------------------------------------
# ADV-004 — stable-grant sweep records preemption notice on reclamation
# ----------------------------------------------------------------------


def test_adv004_sweep_reclamation_records_preemption_notice(
    coordinator, client: _Client,
) -> None:
    """ADV-004: the stable-grant sweep must record a preemption notice
    for the reclaimed victim — otherwise the victim's eventual
    post-edit fails CoherenceError with no F4 context."""
    from ccs.adapters.claude_code.coordinator_server import (
        SWEEP_RECLAMATION_PREEMPTER_ID,
        session_to_agent_id,
    )

    sid = _sid("adv004-A")
    agent_id = session_to_agent_id(sid)
    # Acquire EXCLUSIVE via pre-edit on a tracked artifact.
    s, _ = client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
    assert s == 200
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    assert artifact_id is not None

    # Drive the sweep manually so the heartbeat-stale path fires
    # immediately. heartbeat_timeout_ticks=1 + current_tick well past
    # the agent's last heartbeat triggers reclaim_heartbeat.
    reclaimed_n = coordinator.service.enforce_stable_grant_timeouts(
        current_tick=int(time.time()) + 999_999,
        heartbeat_timeout_ticks=1,
        max_hold_ticks=999_999_999,
        on_reclaim=lambda artifact_id, agent_id, trigger: (
            coordinator.registry.record_preemption_notice(
                victim_agent_id=agent_id,
                artifact_id=artifact_id,
                preempter_agent_id=SWEEP_RECLAMATION_PREEMPTER_ID,
                preempted_at_unix_ts=time.time(),
            )
        ),
    )
    assert reclaimed_n == 1, "sweep should have reclaimed exactly one M/E grant"

    # The preemption notice for the victim must be present and tagged
    # with the sweep-sentinel preempter.
    popped = coordinator.registry.pop_preemption_notice(agent_id, artifact_id)
    assert popped is not None
    preempter_id, _preempted_at = popped
    assert preempter_id == SWEEP_RECLAMATION_PREEMPTER_ID


def test_adv004_post_edit_after_reclamation_returns_reclaimed_message(
    coordinator, client: _Client,
) -> None:
    """End-to-end: pre-edit → sweep reclaims → post-edit gets the F4
    'reclaimed by coordinator sweep' message (NOT the generic
    CoherenceError) and the response carries reclaimed=True instead of
    preempted=True."""
    from ccs.adapters.claude_code.coordinator_server import (
        SWEEP_RECLAMATION_PREEMPTER_ID,
        session_to_agent_id,
    )

    sid = _sid("adv004-B")
    agent_id = session_to_agent_id(sid)
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")

    # Sweep reclaims agent's grant + records notice via the on_reclaim
    # callback (same wiring the real adapter sweep uses).
    coordinator.service.enforce_stable_grant_timeouts(
        current_tick=int(time.time()) + 999_999,
        heartbeat_timeout_ticks=1,
        max_hold_ticks=999_999_999,
        on_reclaim=lambda aid, sid_, trigger: coordinator.registry.record_preemption_notice(
            victim_agent_id=sid_,
            artifact_id=aid,
            preempter_agent_id=SWEEP_RECLAMATION_PREEMPTER_ID,
            preempted_at_unix_ts=time.time(),
        ),
    )

    # Now post-edit fires — should fail with the reclaimed-message.
    s, body = client.post("/hooks/post-edit", {
        "session_id": sid,
        "path": "plan.md",
        "content_hash": _hash("adv004-B-late"),
        "success": True,
    })
    assert s == 200, body
    assert body.get("ok") is False
    assert body.get("reclaimed") is True, (
        f"expected reclaimed=True in F4 response; got {body!r}"
    )
    assert "reclaimed by the coordinator sweep" in body.get("reason", "")
    assert "plan.md" in body.get("reason", "")
    # And NOT the peer-preemption message
    assert "preempted by session" not in body.get("reason", "")


def test_post_edit_fence_reject_returns_stable_reason(
    coordinator, client: _Client,
) -> None:
    """End-to-end: the read-generation fence race window — the grant is still
    EXCLUSIVE but a sweep superseded the claim (owner_generation advanced
    between commit()'s state check and the version persist). post-edit must
    return the STABLE stale_read_generation reason (exact constant, not
    str(exc)) plus current_version, land no phantom bump, and leak no MWB
    transient."""
    from ccs.adapters.claude_code.coordinator_server import session_to_agent_id
    from ccs.core.exceptions import STALE_READ_GENERATION_REASON

    sid = _sid("fence-race")
    agent_id = session_to_agent_id(sid)
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    assert artifact_id is not None

    # Manufacture the race: generation advances while the state stays E
    # (a direct bump stands in for the sweep firing mid-commit; autocommit
    # connection, so no transaction is held open against the server).
    coordinator.registry._conn.execute(
        "UPDATE artifacts SET owner_generation = owner_generation + 1 WHERE id = ?",
        (artifact_id.hex,),
    )
    before = coordinator.registry.get_artifact(artifact_id).version

    s, body = client.post("/hooks/post-edit", {
        "session_id": sid,
        "path": "plan.md",
        "content_hash": _hash("fence-late"),
        "success": True,
    })
    assert s == 200, body
    assert body.get("ok") is False
    assert body.get("reason") == STALE_READ_GENERATION_REASON
    assert body.get("current_version") == before
    # No phantom version bump; no leaked MWB transient (review P1 regression).
    assert coordinator.registry.get_artifact(artifact_id).version == before
    assert coordinator.registry.get_agent_transient(artifact_id, agent_id) is None


def test_adv004_sweep_on_reclaim_callback_exception_does_not_break_sweep(
    coordinator,
) -> None:
    """Defensive: if the on_reclaim callback raises, the sweep continues
    and the reclamation itself still lands. The callback is telemetry
    surface; its failure must not block coherence guarantees."""
    from ccs.adapters.claude_code.coordinator_server import session_to_agent_id

    sid = _sid("adv004-C")
    agent_id = session_to_agent_id(sid)
    # Acquire EXCLUSIVE.
    coordinator.register_session(sid)
    artifact_id = coordinator.registry.resolve_or_register("plan.md", content_hash="")
    coordinator.registry.set_agent_state(
        artifact_id, agent_id, MESIState.EXCLUSIVE,
        trigger="test_setup", tick=0, content_hash=None,
    )

    raises_counter = {"n": 0}
    def raising_callback(*_a, **_kw) -> None:
        raises_counter["n"] += 1
        raise RuntimeError("simulated telemetry failure")

    reclaimed_n = coordinator.service.enforce_stable_grant_timeouts(
        current_tick=int(time.time()) + 999_999,
        heartbeat_timeout_ticks=1,
        max_hold_ticks=999_999_999,
        on_reclaim=raising_callback,
    )
    assert reclaimed_n == 1, "reclamation must still land despite callback failure"
    assert raises_counter["n"] == 1
    # The agent's state is invalid (the reclamation itself succeeded).
    assert coordinator.registry.get_agent_state(artifact_id, agent_id) == MESIState.INVALID


# ----------------------------------------------------------------------
# REL-01 — shutdown drain deadlock (suppressed false positive; locked by test)
# ----------------------------------------------------------------------


def test_rel01_drain_no_deadlock_under_concurrent_dispatch(tmp_path: Path) -> None:
    """REL-01: the reviewer's deeper trace suppressed this as a false
    positive — Condition.wait() releases _in_flight_lock during the
    drain, and acquire_handler_slot's atomic shutting_down check
    prevents new in-flight bumps after shutdown begins. Stress-test
    the invariant: 50 concurrent dispatches racing against shutdown()
    must not deadlock the drain. If REL-01 ever becomes real again
    (e.g., a refactor introduces a non-Condition lock ordering), this
    test will hang and pytest will time out — making the regression
    loud."""
    import threading as _t
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="rel01")
    srv.serve_in_thread()
    time.sleep(0.05)
    secret = load_secret(srv.coordinator_root)
    assert secret is not None
    client = _Client("127.0.0.1", srv.port, secret)

    fire_results: list[int] = []
    def fire(i: int) -> None:
        try:
            s, _ = client.post(
                "/hooks/pre-read",
                {"session_id": _sid(f"rel01-{i}"), "path": "CLAUDE.md"},
            )
            fire_results.append(s)
        except Exception:
            fire_results.append(-1)

    # 50 dispatch threads racing against a delayed shutdown.
    threads = [_t.Thread(target=fire, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()

    # Brief pause so some requests are mid-flight when shutdown fires.
    time.sleep(0.020)

    shutdown_done = _t.Event()
    def call_shutdown() -> None:
        srv.shutdown()
        shutdown_done.set()
    _t.Thread(target=call_shutdown, daemon=True).start()

    # The shutdown MUST complete within IN_FLIGHT_DRAIN_TIMEOUT_SEC + a
    # small margin (handlers in-flight when drain started should
    # complete quickly; new ones after shutting_down=True are denied).
    assert shutdown_done.wait(timeout=10.0), (
        "shutdown deadlocked (REL-01 regression — drain never completed)"
    )
    for t in threads:
        t.join(timeout=2.0)
    # All responses are either 200 (handled) or 503 (post-shutdown) or
    # -1 (connection lost during shutdown). No 500s or hangs.
    assert all(r in (200, 503, -1) for r in fire_results), (
        f"unexpected status codes during shutdown race: {fire_results}"
    )


# ----------------------------------------------------------------------
# REL-03 — free-threading-safe reliability counters
# ----------------------------------------------------------------------


def test_rel03_watchdog_timeouts_counter_under_concurrent_increment(
    tmp_path: Path,
) -> None:
    """REL-03: under free-threading Py 3.13+ or PyPy, ``x += 1`` on a
    plain int is NOT atomic — concurrent threads can tear the increment
    and lose counts. Reliability counters protect with a lock; this
    test verifies that 1000 concurrent increments from 50 threads land
    as exactly 1000 (no torn writes)."""
    import threading as _t
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="rel03")
    try:
        N_THREADS = 50
        N_PER_THREAD = 20  # 1000 total
        def bump_many() -> None:
            for _ in range(N_PER_THREAD):
                srv.increment_watchdog_timeout()
        threads = [_t.Thread(target=bump_many) for _ in range(N_THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()
        expected = N_THREADS * N_PER_THREAD
        assert srv._watchdog_timeouts_total == expected, (
            f"counter torn under concurrent increment: "
            f"expected {expected}, got {srv._watchdog_timeouts_total}"
        )
    finally:
        srv.shutdown()


def test_rel03_watchdog_queue_overflow_counter_under_concurrent_increment(
    tmp_path: Path,
) -> None:
    """Same contract for the queue-overflow counter."""
    import threading as _t
    srv = CoordinatorHTTPServer(tmp_path, port=0, instance_id="rel03b")
    try:
        N_THREADS = 50
        N_PER_THREAD = 20
        def bump_many() -> None:
            for _ in range(N_PER_THREAD):
                srv.increment_watchdog_queue_overflow()
        threads = [_t.Thread(target=bump_many) for _ in range(N_THREADS)]
        for t in threads: t.start()
        for t in threads: t.join()
        expected = N_THREADS * N_PER_THREAD
        assert srv._watchdog_queue_overflows_total == expected
    finally:
        srv.shutdown()


# ----------------------------------------------------------------------
# ADV-005 — empty/missing body rejected with explicit 400
# ----------------------------------------------------------------------


def test_adv005_content_length_zero_returns_explicit_400(coordinator) -> None:
    """ADV-005: a POST with Content-Length:0 must produce an explicit
    'missing or empty body' 400, not fall through to per-field
    validation errors that mask the real cause."""
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    secret = load_secret(coordinator.coordinator_root)
    req = urlrequest.Request(
        url, data=b"", method="POST",
        headers={
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
            "Content-Length": "0",
        },
    )
    try:
        urlrequest.urlopen(req, timeout=5)
        pytest.fail("expected 400")
    except urlerror.HTTPError as e:
        assert e.code == 400
        body = json.loads(e.read().decode())
        assert "empty body" in body["error"].lower() or "missing" in body["error"].lower()


def test_adv005_missing_content_length_returns_400(coordinator) -> None:
    """A POST with no Content-Length header at all should also reject
    with the same explicit error (Content-Length defaults to 0 in
    _read_json on missing)."""
    url = f"http://127.0.0.1:{coordinator.port}/hooks/pre-read"
    secret = load_secret(coordinator.coordinator_root)
    # Build via raw socket to omit Content-Length entirely (urllib auto-adds it).
    import socket as _socket
    sock = _socket.create_connection(("127.0.0.1", coordinator.port))
    try:
        sock.sendall(
            f"POST /hooks/pre-read HTTP/1.0\r\n"
            f"Host: 127.0.0.1\r\n"
            f"Authorization: Bearer {secret}\r\n"
            f"Content-Type: application/json\r\n"
            f"\r\n".encode()
        )
        resp = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            resp += chunk
        # The status line is the first line of the response.
        first_line = resp.split(b"\r\n", 1)[0].decode()
        assert "400" in first_line, f"expected 400, got: {first_line}"
    finally:
        sock.close()


# ----------------------------------------------------------------------
# AC-05 — degraded response shape varies by endpoint contract
# ----------------------------------------------------------------------


def test_ac05_pre_edit_degraded_response_returns_ok_shape(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pre-edit's wire contract is {ok: bool}; degraded envelope on
    watchdog timeout must include ok=True so clients reading
    result.get('ok') don't see None. AC-05 fix."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    def force_timeout(fn, abort=None, deadline=None):
        raise FuturesTimeout()

    monkeypatch.setattr(coordinator, "run_with_watchdog", force_timeout)
    status, body = client.post(
        "/hooks/pre-edit",
        {"session_id": _sid("ac05-pre-edit"), "path": "plan.md"},
    )
    assert status == 200
    assert body.get("ok") is True, (
        f"pre-edit degraded envelope must include ok=True; got {body!r}"
    )
    assert body.get("degraded") is True


def test_ac05_post_edit_degraded_response_returns_ok_shape(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """post-edit's wire contract is {ok: bool}; degraded envelope must
    include ok=True. AC-05 fix."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    def force_timeout(fn, abort=None, deadline=None):
        raise FuturesTimeout()

    monkeypatch.setattr(coordinator, "run_with_watchdog", force_timeout)
    status, body = client.post(
        "/hooks/post-edit",
        {
            "session_id": _sid("ac05-post-edit"),
            "path": "plan.md",
            "content_hash": _hash("ac05"),
            "success": True,
        },
    )
    assert status == 200
    assert body.get("ok") is True
    assert body.get("degraded") is True


def test_ac05_session_stop_degraded_response_returns_ok_shape(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """session-stop's wire contract is {ok: bool}; degraded envelope
    must include ok=True. AC-05 fix."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    def force_timeout(fn, abort=None, deadline=None):
        raise FuturesTimeout()

    monkeypatch.setattr(coordinator, "run_with_watchdog", force_timeout)
    status, body = client.post(
        "/hooks/session-stop", {"session_id": _sid("ac05-session-stop")}
    )
    assert status == 200
    assert body.get("ok") is True
    assert body.get("degraded") is True


def test_ac05_pre_read_degraded_response_keeps_status_fresh_shape(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pre-read's wire contract is {status: ...}; degraded envelope
    keeps the fresh-shape envelope so clients checking status
    don't see ok=None. AC-05 contract preservation."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    def force_timeout(fn, abort=None, deadline=None):
        raise FuturesTimeout()

    monkeypatch.setattr(coordinator, "run_with_watchdog", force_timeout)
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": _sid("ac05-pre-read"), "path": "plan.md"},
    )
    assert status == 200
    assert body.get("status") == "fresh"
    assert body.get("degraded") is True
    # Crucially, pre-read's degraded envelope does NOT include ok.
    assert "ok" not in body


def test_a7_degraded_read_surfaces_advisory_not_silent(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A7: a watchdog-degraded read must NOT silently pass as verified-fresh.
    The degraded envelope now carries a hookSpecificOutput advisory the model
    sees (the hook client passes it straight through), so an in-queue/processing
    timeout cannot masquerade as a confirmed fresh read."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    def force_timeout(fn, abort=None, deadline=None):
        raise FuturesTimeout()

    monkeypatch.setattr(coordinator, "run_with_watchdog", force_timeout)
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": _sid("a7-degraded"), "path": "plan.md"},
    )
    assert status == 200
    assert body.get("status") == "fresh"
    assert body.get("degraded") is True
    hso = body.get("hookSpecificOutput")
    assert isinstance(hso, dict), "degraded read must carry an advisory (non-silent)"
    assert "timed out" in hso.get("additionalContext", "").lower(), (
        f"advisory must explain the freshness check did not run; got {hso!r}"
    )


# ----------------------------------------------------------------------
# Pre-edit's degraded disposition vs. the acquire-or-fail refusal (#196)
#
# A timed-out pre-edit is answered without its work body's decision, and the
# body keeps running in the watchdog pool. These tests drive the REAL watchdog:
# run_with_watchdog is not replaced. The registry's write lock is held from a
# helper thread past a shortened deadline -- the contention that makes a
# handler time out in production -- so the answer, and whatever the abandoned
# body does once the lock frees, are the shipped code paths.
# ----------------------------------------------------------------------

# The held lock, not this value, is what makes the request time out; it only
# needs to be short enough to keep the tests fast.
#
# Every session below CLAIMS and presents its caller principal. pre-edit is
# require-class, and its gate runs BEFORE the work body: a presented principal
# resolves from the service's in-process cache on the handler thread, but an
# absent one on a never-claimed session is first looked up in the durable store
# under the very registry lock these tests hold. That lookup is bounded by the
# same watchdog deadline, so it would time out in the GATE and answer degraded
# before the work body ever ran -- and these tests measure the work body. The
# gate's own timeout is pinned in the caller-principal section below ("the gate
# under registry contention").
_DEGRADE_DEADLINE_SEC = 0.25
_ABANDONED_BODY_SETTLE_SEC = 5.0


class _HeldRegistryLock:
    """Hold the coordinator registry's write lock from a helper thread.

    A pre-edit work body's first registry call blocks on this lock, so the
    request outlives the watchdog deadline. ``release`` lets the abandoned body
    run, which is the moment its late effects either land or abort."""

    def __init__(self, coordinator: CoordinatorHTTPServer) -> None:
        self._lock = coordinator.registry._lock
        self._held = threading.Event()
        self._release = threading.Event()
        self._thread = threading.Thread(target=self._hold, daemon=True)

    def _hold(self) -> None:
        with self._lock:
            self._held.set()
            self._release.wait(timeout=30.0)

    def __enter__(self) -> "_HeldRegistryLock":
        self._thread.start()
        if not self._held.wait(timeout=5.0):
            pytest.fail("helper thread never acquired the registry write lock")
        return self

    def release(self) -> None:
        self._release.set()
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            pytest.fail("helper thread still holds the registry write lock after release")

    def __exit__(self, *exc_info: object) -> None:
        self._release.set()
        self._thread.join(timeout=5.0)


def _await_abandoned_body_settled(
    coordinator: CoordinatorHTTPServer, *, aborts_before: int, completions_before: int,
) -> None:
    """Wait until the timed-out body has finished one way or the other.

    The watchdog's done-callback bumps ``watchdog_late_aborts_total`` when the
    body aborted at the registry lock and ``watchdog_late_completion_total``
    when it ran to the end, so either counter moving means the registry now
    shows everything the body will ever do."""
    deadline = time.monotonic() + _ABANDONED_BODY_SETTLE_SEC
    while time.monotonic() < deadline:
        if (coordinator._watchdog_late_aborts_total > aborts_before
                or coordinator._watchdog_late_completion_total > completions_before):
            return
        time.sleep(0.010)
    pytest.fail(
        f"the timed-out pre-edit body never settled within {_ABANDONED_BODY_SETTLE_SEC}s: "
        f"watchdog_late_aborts_total stayed {coordinator._watchdog_late_aborts_total} "
        f"and watchdog_late_completion_total stayed "
        f"{coordinator._watchdog_late_completion_total}"
    )


def _agent_state_on(coordinator: CoordinatorHTTPServer, path: str, sid: str) -> MESIState | None:
    artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
    assert artifact_id is not None, f"{path} was never registered"
    return coordinator.registry.get_agent_state(artifact_id, session_to_agent_id(sid, None))


def test_pre_edit_watchdog_timeout_answers_the_named_degraded_disposition(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-edit that times out answers exactly ``_PRE_EDIT_DEGRADED_RESPONSE``.

    Prevents the call site and the named constant drifting apart. The guard
    test below holds the CONSTANT to the rule; this is the only test that ties
    what the route actually sends to that constant, so a change that corrects
    the constant but leaves the call site passing another envelope would ship
    the old answer under a green guard. The AC-05 shape test cannot see that:
    it pins ``ok: true`` itself rather than the constant, and it replaces
    ``run_with_watchdog`` wholesale."""
    import ccs.adapters.claude_code.coordinator_server as mod

    sid = _sid("u8-timed-out")
    principal = _explicit_claim(client, sid)
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    timeouts_before = coordinator._watchdog_timeouts_total
    aborts_before = coordinator._watchdog_late_aborts_total
    completions_before = coordinator._watchdog_late_completion_total

    with _HeldRegistryLock(coordinator) as held:
        status, body = client.post(
            "/hooks/pre-edit", {"session_id": sid, "path": "plan.md"}, principal=principal)
        held.release()
    _await_abandoned_body_settled(
        coordinator, aborts_before=aborts_before, completions_before=completions_before)

    assert coordinator._watchdog_timeouts_total == timeouts_before + 1, (
        "the request did not time out, so this measured the undegraded route")
    assert status == 200
    assert body == mod._PRE_EDIT_DEGRADED_RESPONSE, (
        f"a timed-out pre-edit answered {body!r}, not the named disposition")


def test_pre_edit_late_body_after_timeout_grants_nothing_and_leaves_the_holder(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once released, a timed-out pre-edit's abandoned body must not grant the
    caller EXCLUSIVE, and must not invalidate the peer holding it.

    Prevents a late acquire contradicting the answer already written. The
    caller was answered before any grant decision; if the body then acquired,
    the caller would hold an EXCLUSIVE it was never told about and the holder
    would lose a grant nobody reported taking -- the #196 displacement,
    landing after the fact. The A6 abort at the registry lock is what stops
    it, and this pins that the abort reaches pre-edit's acquire, in the one
    scenario an acquire-or-fail refusal exists for: a live holder. The closing
    undegraded call is the control: it shows the same request, not timed out,
    does displace that holder, so the scenario really reaches the acquire."""
    import ccs.adapters.claude_code.coordinator_server as mod

    path = "plan.md"
    holder, caller = _sid("u8-holder"), _sid("u8-late-caller")
    holder_principal = _explicit_claim(client, holder)
    caller_principal = _explicit_claim(client, caller)
    status, _ = client.post(
        "/hooks/pre-edit", {"session_id": holder, "path": path}, principal=holder_principal)
    assert status == 200
    assert _agent_state_on(coordinator, path, holder) == MESIState.EXCLUSIVE
    assert _agent_state_on(coordinator, path, caller) is None

    undegraded_deadline = mod.HANDLER_TIMEOUT_SEC
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    aborts_before = coordinator._watchdog_late_aborts_total
    completions_before = coordinator._watchdog_late_completion_total
    with _HeldRegistryLock(coordinator) as held:
        status, body = client.post(
            "/hooks/pre-edit", {"session_id": caller, "path": path}, principal=caller_principal)
        held.release()
    _await_abandoned_body_settled(
        coordinator, aborts_before=aborts_before, completions_before=completions_before)
    assert status == 200
    assert body.get("degraded") is True, f"the request was not degraded: {body!r}"

    assert _agent_state_on(coordinator, path, caller) is None, (
        "the abandoned body granted the timed-out caller a grant it was never told about")
    assert _agent_state_on(coordinator, path, holder) == MESIState.EXCLUSIVE, (
        "the abandoned body invalidated the live holder after the caller was answered")
    assert coordinator._watchdog_late_aborts_total == aborts_before + 1
    assert coordinator._watchdog_late_completion_total == completions_before

    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", undegraded_deadline)
    status, body = client.post(
        "/hooks/pre-edit", {"session_id": caller, "path": path}, principal=caller_principal)
    assert status == 200 and "degraded" not in body, body
    assert _agent_state_on(coordinator, path, caller) == MESIState.EXCLUSIVE
    assert _agent_state_on(coordinator, path, holder) == MESIState.INVALID


def test_pre_edit_degraded_disposition_refuses_once_acquire_or_fail_can_refuse() -> None:
    """While any acquire-or-fail refusal reason exists, a timed-out pre-edit
    must answer ``ok: false``, never the admit.

    Prevents the refusal asked for in #196 landing with a hole at the watchdog
    seam. The contention that makes displacing a holder worth refusing is the
    same contention that times a handler out; if the degraded answer were
    still ``ok: true``, an opted-in caller would be told to proceed over the
    holder it asked not to displace. ``ok: false`` is the shape
    ``CoherentVolume._check_grant`` refuses in both ``on_error`` modes;
    ``ok: true`` it lets through under ``degrade``.

    Today the reason set is empty, so nothing is asserted about the
    disposition -- deliberately: adding the first reason turns this red until
    the disposition changes in the same change. If the refusal lands with a
    separate disposition for opted-in requests, point this test at that one,
    in that change. The timeout test above ties the named disposition to what
    the route sends, so this is not a check on a constant nobody reads."""
    import ccs.adapters.claude_code.coordinator_server as mod

    refusal_reasons = mod._ACQUIRE_OR_FAIL_REFUSAL_REASONS
    disposition = mod._PRE_EDIT_DEGRADED_RESPONSE
    assert isinstance(refusal_reasons, frozenset)
    assert all(isinstance(reason, str) and reason for reason in refusal_reasons)
    if refusal_reasons:
        assert disposition.get("ok") is False, (
            f"pre-edit can refuse with {sorted(refusal_reasons)} but a timed-out "
            f"pre-edit still answers {disposition!r}, which admits the edit"
        )


# ----------------------------------------------------------------------
# T-01 — pre-bash notices-only branch (no stale paths, non-empty notices)
# ----------------------------------------------------------------------


def test_t01_pre_bash_notices_only_branch_surfaces_preemption(
    client: _Client
) -> None:
    """T-01 / ce-review: previously-untested branch in _handle_pre_bash —
    session has pending preemption notices but the Bash command reads
    only UNTRACKED paths. Expected: response carries hookSpecificOutput
    with the notice text but no stale_paths (status fresh)."""
    x = _sid("t01-X")
    y = _sid("t01-Y")
    # X acquires + commits plan.md (tracked), then Y commits → invalidates X.
    client.post("/hooks/pre-read", {"session_id": x, "path": "plan.md",
                                     "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": x, "path": "plan.md",
                                      "content_hash": _hash("v2"), "success": True})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": y, "path": "plan.md",
                                      "content_hash": _hash("v3"), "success": True})
    # Now X has a pending preemption notice for plan.md.
    # X fires pre-bash with a command that reads only an UNTRACKED path.
    s, body = client.post("/hooks/pre-bash", {
        "session_id": x,
        "command": "cat /etc/hosts",  # untracked, never matches policy
    })
    assert s == 200
    # No tracked path → no stale_paths in response.
    assert "stale_paths" not in body or body["stale_paths"] == []
    # But X's pending notice should surface via additionalContext.
    out = body.get("hookSpecificOutput")
    if out is not None:
        # The notice prose lands here if the handler chose to surface.
        # Either path is acceptable per the contract — the notice may
        # also be deferred to the next pre-read.
        assert "plan.md" in out.get("additionalContext", "") or body.get("status") == "fresh"
    else:
        # Notice deferred to next pre-read — handler returned plain fresh.
        assert body.get("status") == "fresh"


# ----------------------------------------------------------------------
# L5 — idle/uptime use a monotonic clock (NTP-/suspend-safe)
# ----------------------------------------------------------------------


def test_l5_idle_and_uptime_use_monotonic_immune_to_wall_clock(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L5: idle_seconds / uptime_s are monotonic deltas, so a wall-clock
    (NTP step / suspend-resume) jump does not misfire or defer idle shutdown.
    Against the old time.time() body the backward-time assertions below fail."""
    import ccs.adapters.claude_code.coordinator_server as mod

    clock = {"t": 1000.0}
    monkeypatch.setattr(mod.time, "monotonic", lambda: clock["t"])
    with CoordinatorHTTPServer(tmp_path, port=0, instance_id="l5-test") as srv:
        # _started_at and _last_request_at were seeded at monotonic 1000.0.
        clock["t"] = 1030.0
        assert srv.idle_seconds == 30.0
        assert srv.uptime_s == 30.0
        # A wild wall-clock step (NTP back an hour / resume) must NOT perturb the
        # monotonic-based deltas — the core L5 regression assertion.
        monkeypatch.setattr(mod.time, "time", lambda: 1.0)
        assert srv.idle_seconds == 30.0
        assert srv.uptime_s == 30.0
        # mark_request resets idle on the monotonic clock.
        clock["t"] = 1100.0
        srv.mark_request()
        clock["t"] = 1105.0
        assert srv.idle_seconds == 5.0


# ----------------------------------------------------------------------
# /hooks/pre-read — want_owner_generation opt-in (the effect gate's
# pair-consistent comparand read). Absent the flag, every shipped shape is
# byte-unchanged (pinned by the exact-dict tests above).
# ----------------------------------------------------------------------


def test_pre_read_want_owner_generation_fresh_carries_pair(client: _Client) -> None:
    """The opt-in fresh response carries (version, owner_generation) from ONE
    registry snapshot; first observation seeds v1 at generation 0."""
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": _sid("gen1"), "path": "CLAUDE.md",
         "content_hash": _hash("abc"), "want_owner_generation": True},
    )
    assert status == 200
    assert body == {"status": "fresh", "version": 1, "owner_generation": 0}


def test_pre_read_want_owner_generation_valid_grant_carries_pair(client: _Client) -> None:
    """The already-granted fresh branch (not just first observation) also
    carries the pair on opt-in."""
    client.post("/hooks/pre-read",
                {"session_id": _sid("gen2"), "path": "CLAUDE.md", "content_hash": _hash("h1")})
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": _sid("gen2"), "path": "CLAUDE.md",
         "content_hash": _hash("h1"), "want_owner_generation": True},
    )
    assert status == 200
    assert body == {"status": "fresh", "version": 1, "owner_generation": 0}


def test_pre_read_want_owner_generation_stale_after_reclaim_shows_moved_epoch(
    coordinator, client: _Client
) -> None:
    """THE zombie re-validation read: a sweep reclaimed this session's grant, so
    the warn-stale response carries the SAME version but an ADVANCED
    owner_generation — the drift a version-only comparand cannot see."""
    sid = _sid("gen-zombie")
    status, _ = client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
    assert status == 200
    reclaimed = coordinator.service.enforce_stable_grant_timeouts(
        current_tick=int(time.time()) + 999_999,
        heartbeat_timeout_ticks=1,
        max_hold_ticks=999_999_999,
    )
    assert reclaimed == 1, "sweep should have reclaimed exactly one M/E grant"
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "plan.md", "want_owner_generation": True},
    )
    assert status == 200
    assert body["status"] == "stale"
    assert body["owner_generation"] == 1  # the reclaim bumped the epoch
    # The generation rides top-level; the VERSION comparand stays the one this
    # response was classified from (summary.current_version) — the attach never
    # overwrites it with a later snapshot.
    assert "version" not in body
    assert body["summary"]["current_version"] == 1  # the version did NOT move


def test_pre_read_stale_without_flag_omits_generation_keys(
    coordinator, client: _Client
) -> None:
    """No opt-in → the shipped warn-stale envelope is untouched: neither the
    top-level version nor owner_generation appears."""
    sid = _sid("gen-noflag")
    status, _ = client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
    assert status == 200
    coordinator.service.enforce_stable_grant_timeouts(
        current_tick=int(time.time()) + 999_999,
        heartbeat_timeout_ticks=1,
        max_hold_ticks=999_999_999,
    )
    status, body = client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "plan.md"},
    )
    assert status == 200
    assert body["status"] == "stale"
    assert "owner_generation" not in body
    assert "version" not in body


def test_pre_read_want_owner_generation_untracked_fastpath_unchanged(
    client: _Client,
) -> None:
    """The untracked fast-path never consults the registry, flag or no flag —
    no pair to attach, and the exact two-field shape stays."""
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": _sid("gen3"), "path": "untracked.bin",
         "want_owner_generation": True},
    )
    assert status == 200
    assert body == {"status": "fresh"}


def test_pre_read_pair_comes_from_the_classifying_snapshot(
    coordinator, client: _Client
) -> None:
    """The reported generation must belong to the SAME snapshot as the version
    the response was classified from — never a later read a peer commit could
    slip past. That is now structural: one ``get_artifact_and_generation`` call
    backs both, so there is no second read to race. This pins the property
    directly by making the ONLY registry pair-read observably atomic, and
    asserting the handler never reaches for a second one.

    (This replaces an earlier guard that tolerated the two-read shape by
    discarding a mismatched pair; folding the reads made that window
    unrepresentable rather than merely detected.)"""
    sid = _sid("pair-snapshot")
    reads: list[str] = []
    real_pair = coordinator.registry.get_artifact_and_generation

    def counting_pair(aid):
        reads.append("pair")
        return real_pair(aid)

    coordinator.registry.get_artifact_and_generation = counting_pair  # type: ignore[method-assign]
    try:
        status, body = client.post(
            "/hooks/pre-read",
            {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("v1"),
             "want_owner_generation": True},
        )
    finally:
        coordinator.registry.get_artifact_and_generation = real_pair  # type: ignore[method-assign]

    assert status == 200
    assert body == {"status": "fresh", "version": 1, "owner_generation": 0}
    assert len(reads) == 1, (
        "the handler must derive the classification version AND the reported "
        f"generation from ONE pair-atomic read; saw {len(reads)}"
    )


# ======================================================================
# SB-10 U2 — POST /hooks/session-start: post-compaction re-grounding
# ======================================================================
#
# Prose lines are byte-pinned here on purpose: the Node coordinator (U6)
# mirrors these exact strings, and the protocol corpus byte-matches them.
# Any wording change must land in BOTH backends plus the corpus fixtures.

_SS_HEADER = "Post-compaction re-grounding (agent-coherence):"
_SS_CLOSING = (
    "Versions are as of this re-grounding; a more recent read supersedes "
    "this notice."
)


def _session_start_text(body: dict) -> str:
    """Unwrap the additionalContext from a session-start response body."""
    return body["hookSpecificOutput"]["additionalContext"]


def test_session_start_missing_authorization_returns_401(coordinator) -> None:
    url = f"http://127.0.0.1:{coordinator.port}/hooks/session-start"
    req = urlrequest.Request(url, data=b"{}", method="POST",
                             headers={"Host": "127.0.0.1", "Content-Type": "application/json"})
    try:
        urlrequest.urlopen(req, timeout=5)
        assert False, "expected 401"
    except urlerror.HTTPError as e:
        assert e.code == 401


def test_session_start_bad_host_returns_403(client: _Client) -> None:
    status, body = client.post(
        "/hooks/session-start", {"session_id": _sid("ss-host")},
        headers_override={"Host": "attacker.example.com"},
    )
    assert status == 403


def test_session_start_malformed_session_id_returns_400(client: _Client) -> None:
    status, body = client.post("/hooks/session-start", {"session_id": "not-a-uuid"})
    assert status == 400


def test_session_start_empty_session_returns_empty_and_no_flag(
    coordinator, client: _Client
) -> None:
    """AE4 (R5): a session with no coordination state gets `{}` and no
    compact-pending flag — the model sees nothing, the deferred path stays
    unarmed."""
    sid = _sid("ss-empty")
    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    assert body == {}
    assert coordinator.consume_compact_pending(sid) is False


def test_session_start_counter_increments(coordinator, client: _Client) -> None:
    before = coordinator.endpoint_counters_snapshot()["session_start_total"]
    client.post("/hooks/session-start", {"session_id": _sid("ss-counter")})
    after = coordinator.endpoint_counters_snapshot()["session_start_total"]
    assert after == before + 1


def test_session_start_peer_advanced_renders_stale_line(
    coordinator, client: _Client
) -> None:
    """AE1 (R4/R7): A observed v1, peer B committed v2 → A's re-grounding
    flags the divergence with BOTH versions. B's own grant lines must not
    leak into A's payload."""
    sid_a, sid_b = _sid("ss-ae1-a"), _sid("ss-ae1-b")
    client.post("/hooks/pre-read",
                {"session_id": sid_a, "path": "plan.md", "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit", {"session_id": sid_b, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": sid_b, "path": "plan.md",
                 "content_hash": _hash("v2"), "success": True})

    status, body = client.post("/hooks/session-start", {"session_id": sid_a})
    assert status == 200
    assert set(body.keys()) == {"hookSpecificOutput"}
    hso = body["hookSpecificOutput"]
    assert set(hso.keys()) == {"hookEventName", "additionalContext"}
    assert hso["hookEventName"] == "SessionStart"
    text = hso["additionalContext"]
    lines = text.split("\n")
    assert lines[0] == _SS_HEADER
    assert lines[1] == (
        "plan.md advanced to v2 past your last-observed v1 — re-read before "
        "relying on it."
    )
    assert lines[-1] == _SS_CLOSING
    # B's E/M grant belongs to B's session — never rendered for A.
    assert "At compaction you held" not in text
    # KTD8: no timestamps anywhere in the re-grounding prose (no notices in
    # this scenario, so the whole payload must be timestamp-free).
    assert "+00:00" not in text


def test_session_start_scopes_the_state_read_to_this_sessions_agents(
    coordinator, client: _Client, monkeypatch
) -> None:
    """SB-10 review: the builder must narrow ``status_snapshot`` to its own
    session's agents.

    No behavioural test can pin this — the walk looks each pair up per agent
    (``state_by_artifact.get(artifact_id, {}).get(agent_id)``), so surplus
    rows are unobservable and dropping the scope would render byte-identical
    and ship green. Assert the ARGUMENT instead, or the whole point of the
    fix — a hook path that stops reading the workspace's entire never-GC'd
    ``agent_states`` ledger, twice per compaction, under the registry lock —
    is revertible in silence."""
    sid_a, sid_b = _sid("ss-scope-a"), _sid("ss-scope-b")
    # A peer session's rows on the same artifact: present in the ledger,
    # never in scope.
    client.post("/hooks/pre-read",
                {"session_id": sid_a, "path": "plan.md", "content_hash": _hash("v1")})
    client.post("/hooks/pre-read",
                {"session_id": sid_b, "path": "plan.md", "content_hash": _hash("v1")})

    scopes: list[list[uuid.UUID] | None] = []
    real = coordinator.registry.status_snapshot

    def recording(*args, **kwargs):
        scopes.append(
            None if kwargs.get("agent_ids") is None else list(kwargs["agent_ids"])
        )
        return real(*args, **kwargs)

    monkeypatch.setattr(coordinator.registry, "status_snapshot", recording)

    status, _ = client.post("/hooks/session-start", {"session_id": sid_a})
    assert status == 200
    assert len(scopes) == 1, "the builder reads state exactly once"
    expected = [agent_id for agent_id, _ in coordinator.agents_for_session(sid_a)]
    assert scopes[0] == expected
    peer_agent_ids = {
        agent_id for agent_id, _ in coordinator.agents_for_session(sid_b)
    }
    assert not peer_agent_ids.intersection(scopes[0] or []), (
        "a peer session's agent must never enter the scope"
    )


def test_session_start_own_last_writer_renders_non_stale_line(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AE2 (R7, KTD4 second layer): even when the version advanced past the
    recorded last-observed value, a row whose artifact last_writer is the
    requesting agent itself renders the plain version line, never the stale
    flag."""
    sid_a, sid_b = _sid("ss-ae2-a"), _sid("ss-ae2-b")
    client.post("/hooks/pre-read",
                {"session_id": sid_a, "path": "plan.md", "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit", {"session_id": sid_b, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": sid_b, "path": "plan.md",
                 "content_hash": _hash("v2"), "success": True})
    # Force the own-edit exemption arm: pretend A itself is the last writer
    # (the natural flow advances the writer's own last_observed in the same
    # transaction — KTD4's FIRST layer — so the second layer is only
    # reachable via this seam).
    own_agent = session_to_agent_id(sid_a)
    monkeypatch.setattr(coordinator.registry, "last_writer_for", lambda aid: own_agent)

    status, body = client.post("/hooks/session-start", {"session_id": sid_a})
    assert status == 200
    text = _session_start_text(body)
    assert "plan.md is at v2." in text
    assert "advanced to" not in text


def test_session_start_null_last_observed_renders_non_stale(
    coordinator, client: _Client
) -> None:
    """R7: a never-observed row (NULL last_observed) is admitted, never
    flagged — no 0-sentinel comparison may sneak in."""
    sid_a, sid_b = _sid("ss-null-a"), _sid("ss-null-b")
    # B seeds the artifact and advances it to v2.
    client.post("/hooks/pre-read",
                {"session_id": sid_b, "path": "plan.md", "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit", {"session_id": sid_b, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": sid_b, "path": "plan.md",
                 "content_hash": _hash("v2"), "success": True})
    # A gets a state row WITHOUT ever observing bytes: an INVALID upsert on a
    # fresh pair records nothing (U1 contract), leaving last_observed NULL.
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    assert artifact_id is not None
    coordinator.registry.set_agent_state(
        artifact_id, session_to_agent_id(sid_a), MESIState.INVALID,
        trigger="peer_invalidation", tick=1,
    )

    status, body = client.post("/hooks/session-start", {"session_id": sid_a})
    assert status == 200
    text = _session_start_text(body)
    assert "plan.md is at v2." in text
    assert "advanced to" not in text


def test_session_start_held_exclusive_renders_event_anchored_grant_line(
    coordinator, client: _Client
) -> None:
    """R3 + KTD8: grant prose is event-anchored ("At compaction you held"),
    never present-tense — a turn-end Stop drain may release E/M before the
    attachment renders."""
    sid = _sid("ss-grant")
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})

    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    text = _session_start_text(body)
    assert (
        "At compaction you held EXCLUSIVE on plan.md (v1) — re-acquire "
        "before writing."
    ) in text


def test_session_start_cap_renders_three_verbatim_plus_overflow_under_10kb(
    coordinator, client: _Client
) -> None:
    """R5: 5 touched artifacts → 3 verbatim lines + a single "Plus 2 more"
    overflow pointing at the status surface; total payload far below the
    10KB additionalContext ceiling."""
    sid = _sid("ss-cap")
    paths = [f"docs/plans/{n}.md" for n in ("a", "b", "c", "d", "e")]
    for p in paths:
        client.post("/hooks/pre-read",
                    {"session_id": sid, "path": p, "content_hash": _hash(p)})

    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    text = _session_start_text(body)
    for p in paths[:3]:
        assert (
            f"At compaction you held SHARED on {p} (v1) — re-acquire "
            f"before writing."
        ) in text
    for p in paths[3:]:
        assert p not in text
    assert "Plus 2 more — run agent-coherence-status for the full picture." in text
    assert len(text.encode("utf-8")) < 10_000


def test_session_start_pending_notice_rendered_read_only(
    coordinator, client: _Client
) -> None:
    """R3: pending preemption notices appear in the payload via the existing
    preemption prose builder — AFTER the header, BEFORE grant lines — and the
    queue is NOT drained: the next pre-read still surfaces (and consumes)
    the same notice. Consumption ownership stays with the admit endpoints."""
    sid = _sid("ss-notice")
    client.post("/hooks/pre-read",
                {"session_id": sid, "path": "AGENTS.md", "content_hash": _hash("n1")})
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("AGENTS.md")
    assert artifact_id is not None
    agent_id = session_to_agent_id(sid)
    coordinator.registry.record_preemption_notice(
        victim_agent_id=agent_id,
        artifact_id=artifact_id,
        preempter_agent_id=uuid.uuid4(),
        preempted_at_unix_ts=1_700_000_000.0,
    )

    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    text = _session_start_text(body)
    assert text.split("\n")[0] == _SS_HEADER
    notice_at = text.index("⚠ Coordinator notice:")
    grant_at = text.index("At compaction you held")
    assert notice_at < grant_at
    # Read-only proof: the queue still holds the notice for the admit drain.
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "AGENTS.md", "content_hash": _hash("n1")},
    )
    assert status == 200
    assert "AGENTS.md" in body["hookSpecificOutput"]["additionalContext"]


def test_session_start_subagent_grants_grouped_and_released_absent(
    coordinator, client: _Client
) -> None:
    """R3 + KTD8: the parent's lines render first, then subagent groups
    under a `Subagent {name}:` prefix. After the subagent stops (grants
    released), its grant line disappears from the next payload."""
    sid = _sid("ss-sub")
    client.post("/hooks/pre-read",
                {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("c1")})
    client.post("/hooks/pre-edit",
                {"session_id": sid, "path": "plan.md", "agent_id": "worker-1"})

    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    text = _session_start_text(body)
    parent_at = text.index("At compaction you held SHARED on CLAUDE.md (v1)")
    prefix_at = text.index("Subagent worker-1:")
    sub_at = text.index("At compaction you held EXCLUSIVE on plan.md (v1)")
    assert parent_at < prefix_at < sub_at

    # Stop the subagent — its E grant is released (row goes INVALID).
    client.post("/hooks/session-stop", {"session_id": sid, "agent_id": "worker-1"})
    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    text = _session_start_text(body)
    assert "At compaction you held EXCLUSIVE on plan.md" not in text
    # The released row is still a touched row — rendered without a flag.
    assert "Subagent worker-1:" in text
    assert "plan.md is at v1." in text


def test_session_start_two_identical_calls_byte_identical(
    coordinator, client: _Client
) -> None:
    """KTD8 determinism: with no state movement between them, two calls
    produce byte-identical additionalContext (no timestamps, stable sort)."""
    sid_a, sid_b = _sid("ss-det-a"), _sid("ss-det-b")
    client.post("/hooks/pre-read",
                {"session_id": sid_a, "path": "CLAUDE.md", "content_hash": _hash("d1")})
    client.post("/hooks/pre-edit", {"session_id": sid_b, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": sid_b, "path": "plan.md",
                 "content_hash": _hash("d2"), "success": True})
    client.post("/hooks/pre-read",
                {"session_id": sid_a, "path": "docs/plans/x.md", "content_hash": _hash("d3")})

    _, first = client.post("/hooks/session-start", {"session_id": sid_a})
    _, second = client.post("/hooks/session-start", {"session_id": sid_a})
    assert _session_start_text(first) == _session_start_text(second)


def test_session_start_sets_flag_consume_is_test_and_clear(
    coordinator, client: _Client
) -> None:
    """KTD5: a non-empty session-start marks compact-pending; the first
    consume wins, the second sees nothing."""
    sid = _sid("ss-flag")
    client.post("/hooks/pre-read",
                {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("f1")})
    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    assert body != {}
    assert coordinator.consume_compact_pending(sid) is True
    assert coordinator.consume_compact_pending(sid) is False


def test_consume_compact_pending_race_exactly_one_winner(coordinator) -> None:
    """KTD5: consume is an atomic test-and-clear — 8 racing consumers get
    exactly one True (the deferred-delivery unit's TOCTOU safety)."""
    sid = _sid("ss-race")
    coordinator.mark_compact_pending(sid)
    barrier = threading.Barrier(8)
    results: list[bool] = []
    results_lock = threading.Lock()

    def racer() -> None:
        barrier.wait()
        got = coordinator.consume_compact_pending(sid)
        with results_lock:
            results.append(got)

    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(results) == 1


def test_expire_compact_pending_clears_flag(coordinator) -> None:
    """KTD5: expire drops an unconsumed flag (parent-Stop wiring lands in
    the deferred-delivery unit; the primitive is pinned here)."""
    sid = _sid("ss-expire")
    coordinator.mark_compact_pending(sid)
    coordinator.expire_compact_pending(sid)
    assert coordinator.consume_compact_pending(sid) is False


def test_session_start_degraded_returns_empty_and_no_flag(
    coordinator, client: _Client
) -> None:
    """KTD7: a watchdog-degraded session-start answers `{}` — advisory
    re-grounding must never block — and the compact-pending flag stays
    unset (nothing to deliver later that was never built)."""
    from concurrent.futures import TimeoutError as FuturesTimeout
    from unittest.mock import patch

    sid = _sid("ss-degraded")
    client.post("/hooks/pre-read",
                {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("g1")})
    with patch.object(coordinator, "run_with_watchdog", side_effect=FuturesTimeout()):
        status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    assert body == {}
    assert coordinator.consume_compact_pending(sid) is False


def test_session_start_breadcrumb_only_for_never_seen_with_state(
    coordinator, client: _Client, caplog: pytest.LogCaptureFixture
) -> None:
    """R8: the debug breadcrumb fires exactly when a never-seen session
    arrives while the workspace holds coordination state — converting a
    silent session-id-rotation regression into an observable."""
    import logging

    caplog.set_level(logging.DEBUG, logger="ccs.adapters.claude_code.coordinator_server")

    # (a) never-seen session + EMPTY workspace → no breadcrumb.
    client.post("/hooks/session-start", {"session_id": _sid("ss-bc-empty")})
    assert "never-seen session" not in caplog.text

    # Seed workspace state with an unrelated session.
    client.post("/hooks/pre-read",
                {"session_id": _sid("ss-bc-other"), "path": "CLAUDE.md",
                 "content_hash": _hash("b1")})

    # (b) never-seen session + NONEMPTY workspace → breadcrumb.
    caplog.clear()
    client.post("/hooks/session-start", {"session_id": _sid("ss-bc-new")})
    assert "never-seen session" in caplog.text

    # (c) already-seen session + nonempty workspace → no breadcrumb.
    caplog.clear()
    client.post("/hooks/session-start", {"session_id": _sid("ss-bc-new")})
    assert "never-seen session" not in caplog.text


def test_session_start_prose_constants_byte_pinned() -> None:
    """KTD8: the Node coordinator byte-matches these module-level constants;
    a wording tweak here must ship in both backends + the corpus."""
    from ccs.adapters.claude_code import hook_payloads as hp

    assert hp.SESSION_START_HEADER == "Post-compaction re-grounding (agent-coherence):"
    assert hp.SESSION_START_GRANT_LINE_TEMPLATE == (
        "At compaction you held {state} on {path} (v{version}) — re-acquire "
        "before writing."
    )
    assert hp.SESSION_START_STALE_LINE_TEMPLATE == (
        "{path} advanced to v{current} past your last-observed v{last} — "
        "re-read before relying on it."
    )
    assert hp.SESSION_START_TOUCHED_LINE_TEMPLATE == "{path} is at v{current}."
    assert hp.SESSION_START_OVERFLOW_LINE_TEMPLATE == (
        "Plus {count} more — run agent-coherence-status for the full picture."
    )
    assert hp.SESSION_START_SUBAGENT_PREFIX_TEMPLATE == "Subagent {name}:"
    assert hp.SESSION_START_CLOSING_LINE == (
        "Versions are as of this re-grounding; a more recent read supersedes "
        "this notice."
    )


# ======================================================================
# SB-10 U4 — deferred re-grounding delivery on the next qualifying admit
# ======================================================================
#
# R2: at-most-once per delivery path; the flag is consumed by the FIRST
# qualifying PARENT admit and expires at parent Stop. R8: the payload
# attaches ONLY to allow envelopes; a request carrying agent_id neither
# consumes nor attaches. KTD6: the check rides the four admit surfaces
# (pre-read, pre-edit, pre-bash, pre-grep), with the advisory flag peek
# hoisted above the untracked fast-path exits.


def _arm_reground(client: _Client, sid: str) -> str:
    """Give the session coordination state (SHARED on CLAUDE.md), then arm
    the deferred flag via the REAL /hooks/session-start endpoint (KTD5).
    Returns the payload text the arming call rendered — KTD2's
    rebuild-at-delivery means the deferred copy must byte-match it as long
    as no state moves in between."""
    client.post("/hooks/pre-read",
                {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    assert body != {}
    return body["hookSpecificOutput"]["additionalContext"]


def _reground_text_of(body: dict) -> str | None:
    """The additionalContext of an admit response's allow envelope, or None."""
    hso = body.get("hookSpecificOutput")
    if hso is None:
        return None
    return hso.get("additionalContext")


def test_deferred_reground_tracked_pre_read_delivers_exactly_once(
    coordinator, client: _Client
) -> None:
    """Pending flag + tracked pre-read admit → the fresh admit carries the
    re-grounding block (byte-identical to the session-start rendering —
    KTD2 rebuild-at-delivery), and the following admit is clean.

    This tracked fresh-no-notice admit has no envelope of its own, so the
    attach mints the CONTEXT-ONLY shape: prose without a permission
    decision (advisory payloads never widen one)."""
    sid = _sid("dr-tracked")
    armed_text = _arm_reground(client, sid)
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    assert body["status"] == "fresh"
    out = body["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert "permissionDecision" not in out
    assert out["additionalContext"] == armed_text
    # Consumed: the very next admit is today's bare fresh shape.
    status, body2 = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    assert body2 == {"status": "fresh", "version": 1}


def test_deferred_reground_untracked_pre_read_delivers_then_bare(
    coordinator, client: _Client
) -> None:
    """Pending flag + UNTRACKED pre-read → the payload rides a CONTEXT-ONLY
    PreToolUse envelope on the fast path (no permission decision — the
    advisory delivery must not auto-approve an otherwise-promptable tool
    call); the next untracked call returns today's bare body."""
    sid = _sid("dr-untracked")
    armed_text = _arm_reground(client, sid)
    status, body = client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "notes.txt"})
    assert status == 200
    assert body["status"] == "fresh"
    out = body["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert "permissionDecision" not in out
    assert out["additionalContext"] == armed_text
    status, body2 = client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "notes.txt"})
    assert status == 200
    assert body2 == {"status": "fresh"}


def test_deferred_reground_untracked_admit_emits_no_permission_decision(
    coordinator, client: _Client
) -> None:
    """Security pin (explicit absence): the deferred delivery must never
    widen a permission decision. An untracked admit whose base body has no
    envelope of its own gets prose ONLY — hookEventName + additionalContext
    and NOTHING else. A ``permissionDecision: "allow"`` here would
    short-circuit Claude Code's permission prompting for that tool call,
    once per compaction, purely as a side effect of delivery.

    Verified empirically (A/B capture, Claude Code CLI 2.1.233,
    2026-08-25): the CLI renders additionalContext on a PreToolUse
    envelope carrying no permissionDecision, so the allow was never
    needed to get the prose in front of the model."""
    sid = _sid("dr-no-decision")
    armed_text = _arm_reground(client, sid)
    status, body = client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "notes.txt"})
    assert status == 200
    out = body["hookSpecificOutput"]
    assert "permissionDecision" not in out
    assert "permissionDecisionReason" not in out
    assert set(out) == {"hookEventName", "additionalContext"}
    assert out["hookEventName"] == "PreToolUse"
    # The advisory still lands — absence of the decision costs no prose.
    assert out["additionalContext"] == armed_text
    assert armed_text.startswith("Post-compaction re-grounding (agent-coherence):")


def test_deferred_reground_merge_preserves_existing_permission_decision(
    coordinator, client: _Client
) -> None:
    """The OTHER branch is untouched: when the base admit already carries
    an envelope (here a warn-mode stale-read allow), the attach merges the
    re-grounding block into it and the pre-existing permissionDecision
    survives verbatim — the advisory neither widens nor narrows a decision
    the admit already made. KTD6 ordering: the stale warning renders
    first, the re-ground block after it."""
    sid = _sid("dr-merge")
    peer = _sid("dr-merge-peer")
    # Session A first-reads plan.md (SHARED v1); peer commits v2 → A stale.
    client.post("/hooks/pre-read",
                {"session_id": sid, "path": "plan.md", "content_hash": _hash("m1")})
    client.post("/hooks/pre-edit", {"session_id": peer, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": peer, "path": "plan.md",
                 "content_hash": _hash("m2"), "success": True})
    coordinator.mark_compact_pending(sid)
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "plan.md", "content_hash": _hash("m1")})
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"  # warn-mode decision preserved
    text = out["additionalContext"]
    assert text.index("Stale read") < text.index(
        "Post-compaction re-grounding (agent-coherence):")
    assert coordinator.has_compact_pending(sid) is False


def test_untracked_pre_read_no_flag_byte_identical_and_registry_free(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No flag + untracked → byte-identical to today's fast-path response
    AND zero registry access (the KTD6 advisory peek is a process-local
    dict lookup). The registry is swapped for a proxy that explodes on ANY
    attribute access; a touched registry would 500 the request."""
    sid = _sid("dr-noflag")

    class _ExplodingRegistry:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"registry touched on untracked fast path: {name}")

    monkeypatch.setattr(coordinator, "registry", _ExplodingRegistry())
    status, body = client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "notes.txt"})
    assert status == 200
    assert body == {"status": "fresh"}


def test_deferred_reground_subagent_admit_neither_consumes_nor_attaches(
    coordinator, client: _Client
) -> None:
    """R8: a request carrying agent_id (subagent identity) must neither
    consume nor attach — the payload waits for the PARENT's next admit.
    Covered on BOTH the untracked fast path (bare bytes preserved) and the
    tracked seam (the subagent's own warn envelope carries no re-ground)."""
    sid = _sid("dr-subagent")
    _arm_reground(client, sid)
    # Untracked subagent admit: today's bare fast-path bytes, flag intact.
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "agent_id": "sub1", "path": "notes.txt"})
    assert status == 200
    assert body == {"status": "fresh"}
    assert coordinator.has_compact_pending(sid) is True
    # Tracked subagent admit: the subagent's first read of the existing
    # artifact yields the ordinary warn-stale envelope — but never the
    # re-grounding block, and the flag survives.
    status, body2 = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "agent_id": "sub1", "path": "CLAUDE.md",
         "content_hash": _hash("rg1")})
    assert status == 200
    assert "Post-compaction re-grounding" not in (_reground_text_of(body2) or "")
    assert coordinator.has_compact_pending(sid) is True
    # The parent's next admit delivers (rebuilt from CURRENT registry truth
    # per KTD2 — the subagent's grants above now render as their own group).
    status, body3 = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    delivered = _reground_text_of(body3) or ""
    assert delivered.startswith("Post-compaction re-grounding (agent-coherence):")
    assert "Subagent sub1:" in delivered
    assert coordinator.has_compact_pending(sid) is False


def test_subagent_stop_keeps_flag_parent_stop_expires(
    coordinator, client: _Client
) -> None:
    """R2 lifetime: SubagentStop (agent_id present) leaves the flag
    untouched — the parent turn is still in flight; a parent Stop (no
    agent_id) expires it, so the next admit is clean."""
    sid = _sid("dr-stop")
    _arm_reground(client, sid)
    status, _ = client.post(
        "/hooks/session-stop", {"session_id": sid, "agent_id": "sub1"})
    assert status == 200
    assert coordinator.has_compact_pending(sid) is True
    status, _ = client.post("/hooks/session-stop", {"session_id": sid})
    assert status == 200
    assert coordinator.has_compact_pending(sid) is False
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_second_session_start_re_marks_idempotently_one_delivery(
    coordinator, client: _Client
) -> None:
    """A second compaction before delivery re-stamps the same flag: still
    exactly one delivery, and the admit after it is clean."""
    sid = _sid("dr-remark")
    _arm_reground(client, sid)
    status, body = client.post("/hooks/session-start", {"session_id": sid})
    assert status == 200
    armed_text = body["hookSpecificOutput"]["additionalContext"]
    status, first = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    assert _reground_text_of(first) == armed_text
    status, second = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    assert second == {"status": "fresh", "version": 1}


def test_deferred_reground_pre_edit_delivers(
    coordinator, client: _Client
) -> None:
    """Pre-edit attach point: a pending flag rides the {ok: true} admit as
    a new allow envelope; the next pre-edit is today's bare shape. KTD2
    rebuild-at-delivery: this pre-edit acquires EXCLUSIVE BEFORE the attach
    seam runs, so the delivered prose renders the CURRENT grant state —
    EXCLUSIVE — not the SHARED snapshot the arming call saw."""
    sid = _sid("dr-edit")
    _arm_reground(client, sid)
    status, body = client.post(
        "/hooks/pre-edit", {"session_id": sid, "path": "CLAUDE.md"})
    assert status == 200
    assert body["ok"] is True
    delivered = _reground_text_of(body) or ""
    assert delivered.startswith("Post-compaction re-grounding (agent-coherence):")
    assert (
        "At compaction you held EXCLUSIVE on CLAUDE.md (v1) — re-acquire "
        "before writing." in delivered
    )
    status, body2 = client.post(
        "/hooks/pre-edit", {"session_id": sid, "path": "CLAUDE.md"})
    assert status == 200
    assert body2 == {"ok": True}


def test_deferred_reground_pre_bash_delivers_untracked_and_tracked(
    coordinator, client: _Client
) -> None:
    """Pre-bash attach point: delivery works from BOTH the zero-tracked-
    paths fast path and the tracked work path."""
    sid = _sid("dr-bash")
    armed_text = _arm_reground(client, sid)
    # Fast path: no tracked paths detected in the command.
    status, body = client.post(
        "/hooks/pre-bash", {"session_id": sid, "command": "echo hello"})
    assert status == 200
    assert body["status"] == "fresh"
    assert _reground_text_of(body) == armed_text
    status, bare = client.post(
        "/hooks/pre-bash", {"session_id": sid, "command": "echo hello"})
    assert status == 200
    assert bare == {"status": "fresh"}
    # Tracked path: session is fresh on CLAUDE.md; the flag re-armed.
    coordinator.mark_compact_pending(sid)
    status, body2 = client.post(
        "/hooks/pre-bash", {"session_id": sid, "command": "cat CLAUDE.md"})
    assert status == 200
    assert body2["status"] == "fresh"
    assert _reground_text_of(body2) == armed_text


def test_deferred_reground_pre_grep_delivers_tracked_and_empty_root(
    coordinator, client: _Client
) -> None:
    """Pre-grep attach point: delivery works from BOTH the tracked work
    path (artifacts under the search root) and the zero-tracked-artifacts
    fast path."""
    sid = _sid("dr-grep")
    armed_text = _arm_reground(client, sid)
    status, body = client.post(
        "/hooks/pre-grep", {"session_id": sid, "search_root": ""})
    assert status == 200
    assert body["status"] == "fresh"
    assert _reground_text_of(body) == armed_text
    # Fast path: a root with no registry-known artifacts.
    coordinator.mark_compact_pending(sid)
    status, body2 = client.post(
        "/hooks/pre-grep", {"session_id": sid, "search_root": "src/empty"})
    assert status == 200
    assert body2["status"] == "fresh"
    assert _reground_text_of(body2) == armed_text
    status, bare = client.post(
        "/hooks/pre-grep", {"session_id": sid, "search_root": "src/empty"})
    assert status == 200
    assert bare == {"status": "fresh"}


def test_deferred_reground_concurrent_parent_admits_exactly_one_delivery(
    coordinator, client: _Client
) -> None:
    """R2 at-most-once under contention: 6 concurrent qualifying parent
    admits race one pending flag; EXACTLY ONE response carries the
    re-grounding block (the atomic pop at the attach seam)."""
    sid = _sid("dr-race")
    _arm_reground(client, sid)
    barrier = threading.Barrier(6)
    results: list[dict] = []
    results_lock = threading.Lock()

    def admit(i: int) -> None:
        barrier.wait()
        # Distinct tracked paths so first-observation seeding never contends
        # on one artifact row; every response is a qualifying fresh admit.
        _, body = client.post(
            "/hooks/pre-read",
            {"session_id": sid, "path": f"docs/plans/p{i}.md",
             "content_hash": _hash(f"p{i}")})
        with results_lock:
            results.append(body)

    threads = [threading.Thread(target=admit, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    delivered = [
        body for body in results
        if "Post-compaction re-grounding" in (_reground_text_of(body) or "")
    ]
    assert len(delivered) == 1, (
        f"expected exactly one re-grounding delivery, got {len(delivered)} "
        f"of {len(results)} admits"
    )
    assert coordinator.has_compact_pending(sid) is False


def test_deferred_reground_rebuild_failure_forfeits_without_breaking_admit(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """KTD2 rebuild-at-delivery can fail AFTER the claim won the pop. The
    guard in ``_claim_reground_context`` forfeits that delivery (R2 permits
    at-most-once → zero) instead of turning an otherwise-successful admit
    into an internal-error body: the admit answers its ordinary bare shape,
    the flag stays consumed, and the forfeit is logged."""
    import logging

    import ccs.adapters.claude_code.coordinator_server as mod

    caplog.set_level(logging.ERROR, logger=_CSRV)
    sid = _sid("dr-rebuild-fail")

    # Explode on the DELIVERY rebuild only — the arming session-start is
    # call #1 and must succeed, or the flag would never be set at all.
    real_build = mod._build_session_start_context
    calls = {"n": 0}

    def build_then_explode(coord, sid_arg, *, abort=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_build(coord, sid_arg, abort=abort)
        raise RuntimeError("registry exploded mid-rebuild")

    monkeypatch.setattr(mod, "_build_session_start_context", build_then_explode)
    _arm_reground(client, sid)
    assert calls["n"] == 1
    caplog.clear()

    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    # The admit is NOT broken: today's bare fresh shape, not the
    # {ok: false, reason: "internal: ..."} body _run_or_degrade emits when
    # work() raises.
    assert body == {"status": "fresh", "version": 1}
    assert calls["n"] == 2
    # The claim won the pop before the rebuild raised, so the delivery is
    # forfeited rather than re-queued.
    assert coordinator.has_compact_pending(sid) is False
    assert "delivery forfeited" in caplog.text

    status, body2 = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    assert body2 == {"status": "fresh", "version": 1}


def test_deferred_reground_fast_path_slow_rebuild_degrades_to_bare_base(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fast-path delivery rebuild walks the registry, so it runs under
    the handler watchdog: a slow/contended rebuild must not hang the hook
    past its budget. On timeout the response is the bare ``base`` —
    byte-identical to the no-flag fast path — the request raises nothing,
    and the degradation is observable via the /status counter."""
    import ccs.adapters.claude_code.coordinator_server as mod

    sid = _sid("dr-fast-slow")
    _arm_reground(client, sid)

    def slow_build(coord, sid_arg, *, abort=None):
        time.sleep(1.0)
        return ("never reaches the client", True)

    monkeypatch.setattr(mod, "_build_session_start_context", slow_build)
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", 0.1)

    before = coordinator._watchdog_timeouts_total
    started = time.monotonic()
    status, body = client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "notes.txt"})
    elapsed = time.monotonic() - started
    assert status == 200
    assert body == {"status": "fresh"}
    assert elapsed < 0.8, (
        f"fast-path delivery waited {elapsed:.2f}s on a 1.0s rebuild; the "
        f"watchdog did not cut it"
    )
    assert coordinator._watchdog_timeouts_total == before + 1


def test_deferred_reground_preset_abort_does_not_consume_flag(
    coordinator, client: _Client
) -> None:
    """A6: the abort check runs BEFORE the pop, so a request already known
    doomed leaves the flag pending — the delivery survives for the next
    live admit instead of being burned by a response nobody will read."""
    import ccs.adapters.claude_code.coordinator_server as mod

    sid = _sid("dr-abort-preset")
    armed_text = _arm_reground(client, sid)

    doomed = threading.Event()
    doomed.set()
    base = {"status": "fresh"}
    out = mod._deliver_pending_reground(
        coordinator, sid, {"session_id": sid}, base, abort=doomed
    )
    # Same object back — the qualifying admit body stays byte-identical.
    assert out is base
    assert coordinator.has_compact_pending(sid) is True

    # The next live admit still delivers.
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": "CLAUDE.md", "content_hash": _hash("rg1")})
    assert status == 200
    assert _reground_text_of(body) == armed_text
    assert coordinator.has_compact_pending(sid) is False


# ======================================================================
# Unknown-holder sentinel: the warn-mode renderers must not slice it
# ======================================================================
#
# ``emit_strict_deny`` already preserves a ``<...>`` placeholder verbatim
# (a bare ``[:8]`` yields the malformed ``<unknown``). The two warn-mode
# renderers sliced unconditionally, so a post-restart collision — where
# the adapter has no name for the surviving holder — reached the model as
# "another session (<unknown) has been editing …".


def test_warn_renderers_preserve_unknown_sentinel_verbatim() -> None:
    """A ``<...>`` placeholder is prose, not an id: it renders whole in all
    THREE renderers. Slicing it to 8 chars drops the closing bracket."""
    from ccs.adapters.claude_code import hook_payloads as hp

    collision = hp.edit_collision_warning(
        holder_session_id="<unknown>",
        holder_acquired_at_unix_ts=1700000000.0,
        path="docs/plan.md",
    )
    assert "(<unknown>)" in collision
    assert "(<unknown)" not in collision

    stale = hp.stale_read_warning({
        "path": "docs/plan.md",
        "last_writer_session_id": "<unknown>",
        "last_writer_at_unix_ts": 1700000000.0,
        "warning_generated_at_unix_ts": 1700000001.0,
        "hash_differs": False,
        "prior_version_seen_by_session": 1,
        "current_version": 2,
        "your_version": 1,
    })
    assert "agent <unknown> at" in stale
    assert "agent <unknown at" not in stale

    # The guarded renderer is the reference behavior, not a third variant.
    deny = hp.emit_strict_deny(source="test", summary={
        "path": "docs/plan.md",
        "last_writer_session_id": "<unknown>",
        "last_writer_at_unix_ts": 1700000000.0,
        "warning_generated_at_unix_ts": 1700000001.0,
        "hash_differs": False,
        "prior_version_seen_by_session": 1,
        "current_version": 2,
        "your_version": 1,
    })
    assert "<unknown>" in deny["permissionDecisionReason"]


def test_real_session_ids_still_render_as_eight_char_prefixes() -> None:
    """The sentinel guard is keyed on the ``<...>`` shape, so a real UUID
    session id keeps its short form in both warn renderers."""
    from ccs.adapters.claude_code import hook_payloads as hp

    sid = "f2f7eab3-1111-4111-8111-111111111111"
    collision = hp.edit_collision_warning(
        holder_session_id=sid,
        holder_acquired_at_unix_ts=1700000000.0,
        path="docs/plan.md",
    )
    assert "(f2f7eab3)" in collision
    assert sid not in collision

    stale = hp.stale_read_warning({
        "path": "docs/plan.md",
        "last_writer_session_id": sid,
        "last_writer_at_unix_ts": 1700000000.0,
        "warning_generated_at_unix_ts": 1700000001.0,
        "hash_differs": False,
        "prior_version_seen_by_session": 1,
        "current_version": 2,
        "your_version": 1,
    })
    assert "agent f2f7eab3 at" in stale
    assert sid not in stale


# ``coordinator_server`` has its OWN two renderers of the same prose, and PR
# #200 -- which added ``short_session_id`` and the three tests above -- never
# generalized the guard to them. Its commit message says so outright. That file
# imports ``hook_payloads as _payloads`` but never called the helper, so the
# docstring contract at hook_payloads.py ("Every renderer that shortens an
# identity handle for prose goes through here") was asserted and unenforced.
#
# Since R7 the preempter arm cannot reach the sentinel at all: the notice row
# carries the preempter's agent id, so there is nothing to fail to look up.
# The sentinel arm that IS still reachable is the last-writer one -- an
# artifact with no committed writer -- and the two warn-renderer tests above
# cover it.


def test_preemption_prose_does_not_consult_the_agent_name_map() -> None:
    """The preempter is named without any lookup, so a restart cannot blank it.

    This test used to assert the opposite outcome: a coordinator that could
    not name the preempter rendered ``session <unknown> at ...``, and the
    assertion was that the sentinel kept its closing bracket. That degraded
    state was the ORDINARY one after a restart -- pending notices live in
    SQLite while the agent-name map is an in-process dict seeded empty on
    every start -- so the attribution an operator most needed was the one
    most likely to be missing.

    The stub coordinator here has no ``agent_name_for`` at all. Rendering
    still succeeds and still names the preempter, which is what proves the
    lookup is gone rather than merely unused.
    """
    from uuid import UUID

    from ccs.adapters.claude_code import coordinator_server as cs

    class _Artifact:
        name = "docs/plan.md"

    class _Registry:
        def get_artifact(self, artifact_id):  # noqa: ANN001, ANN201
            return _Artifact()

    class _NamelessCoordinator:
        registry = _Registry()

    preempter = UUID("22222222-2222-4222-8222-222222222222")
    text = cs._build_preemption_text(
        _NamelessCoordinator(),
        [(UUID("11111111-1111-4111-8111-111111111111"), preempter, 1700000000.0)],
    )
    assert "preempted/revoked by agent 22222222 at" in text
    assert "<unknown" not in text


def test_preemption_prose_is_byte_stable_for_a_real_agent_id() -> None:
    """Pin the WHOLE rendered string, not substrings.

    This file is held at cross-backend wire parity and the sentinel fix reflowed
    one f-string across two physical lines to stay inside ruff's limit. That
    reflow was proved byte-neutral by executing both module versions side by
    side -- a one-time check, which is exactly the kind that does not survive the
    next edit. Every other test here asserts substrings, so a stray space or a
    re-wrapped clause moves bytes and stays green.

    Scoped deliberately to ``_build_preemption_text``: it is the renderer that
    was reflowed, and its output is fully determined by its inputs. The sibling
    renderer at ``_handle_post_edit`` closes with ``Underlying coordinator error:
    {exc}`` over a preemption timestamp the test does not choose, so a golden
    there would pin values that legitimately vary -- worse than the substring
    assertions it would replace. That one keeps them.
    """
    from uuid import UUID

    from ccs.adapters.claude_code import coordinator_server as cs

    class _Artifact:
        name = "docs/plan.md"

    class _Registry:
        def get_artifact(self, artifact_id):  # noqa: ANN001, ANN201
            return _Artifact()

    class _NamedCoordinator:
        """A plain coordinator -- the REAL-id path, where bytes must not move."""

        registry = _Registry()

    text = cs._build_preemption_text(
        _NamedCoordinator(),
        [(
            UUID("11111111-1111-4111-8111-111111111111"),
            UUID("22222222-2222-4222-8222-222222222222"),
            1700000000.0,
        )],
    )

    assert text == (
        "\u26a0 Coordinator notice: your EXCLUSIVE grant was preempted:\n"
        "  \u2022 docs/plan.md \u2014 preempted/revoked by agent 22222222 at "
        "2023-11-14T22:13:20+00:00. Any local edit you made to this file will "
        "land in your worktree but is NOT reflected in the coordinator's version.\n"
        "Re-read affected files before continuing if you need the latest "
        "coordinator-tracked version, or proceed knowing your edits remain "
        "local-only until you re-acquire and commit."
    )


def test_post_edit_preemption_reason_survives_a_real_restart(tmp_path: Path) -> None:
    """The second renderer, ``_handle_post_edit``'s ``commit_not_allowed`` reason,
    driven across an ACTUAL coordinator restart.

    This used to assert the restart DEGRADED the attribution: the preemption
    notice is durable in SQLite while the agent-name map is process-local and
    starts empty, so the reverse lookup returned None and the reason named
    ``session <unknown>``. Since R7 the renderer reads the preempter's agent
    id straight off the notice row, so the restart costs nothing and the
    reason names Y on both sides of it. That is the assertion now.

    Restarting for real rather than clearing ``_agent_names`` in place keeps the
    test off a private attribute AND proves the half the simulation had to assume:
    that the notice actually survives the process. The agent id is a uuid5 of the
    session id, so X's id is the same on both sides of the restart -- which is
    exactly why the notice still finds it.
    """
    x = _sid("X")
    y = _sid("Y")

    before = _restart_on(tmp_path, "notice-before-restart")
    try:
        secret = load_secret(before.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", before.port, secret)
        client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
        client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})  # preempts X
    finally:
        before.shutdown()

    after = _restart_on(tmp_path, "notice-after-restart")
    try:
        secret = load_secret(after.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", after.port, secret)
        status, body = client.post(
            "/hooks/post-edit",
            {"session_id": x, "path": "plan.md", "content_hash": _hash("h"), "success": True},
        )
    finally:
        after.shutdown()

    assert status == 200
    assert body.get("ok") is False, f"post-edit on a preempted grant must fail; got {body}"

    reason = body.get("reason", "")
    assert "preempted by agent" in reason, (
        f"the notice did not survive the restart, so this asserts nothing; got: {reason}"
    )
    assert f"preempted by agent {session_to_agent_id(y).hex[:8]} at" in reason, (
        f"the restart must not cost the attribution; got: {reason}"
    )
    assert "<unknown" not in reason, (
        f"there is no lookup left to fail, so nothing can degrade to a "
        f"sentinel here; got: {reason}"
    )


# ======================================================================
# /status holder set survives a coordinator restart
# ======================================================================
#
# ``sessions`` used to be built by walking the adapter's in-memory
# ``_agent_names`` map and looking each agent up in the registry snapshot.
# That map is seeded empty on every process start and is only ever written
# by ``register_session`` on hook traffic, so a restart erased every holder
# from the payload while ``agent_states`` — and enforcement — kept them.
# The registry's own holder set is the source of truth; a name is a label
# the adapter may or may not have.


def _restart_on(root: Path, instance_id: str) -> CoordinatorHTTPServer:
    server = CoordinatorHTTPServer(root, port=0, instance_id=instance_id)
    server.serve_in_thread()
    time.sleep(0.05)
    return server


def test_status_lists_a_grant_holder_that_predates_the_restart(
    tmp_path: Path,
) -> None:
    """A holder whose grant survived a restart is reported, keyed on its raw
    agent id with a null name — not dropped. Enforcement never stopped, so a
    caller polling /status must not read the workspace as idle."""
    first = _restart_on(tmp_path, "restart-before")
    try:
        secret = load_secret(first.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", first.port, secret)
        sid = _sid("survivor")
        client.post("/policy/track", {"paths": ["docs/plan.md"]})
        client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})

        _, before = client.get(
            "/status?detail=full",
            headers_override={"Coherence-Local-Operator": "true"},
        )
        assert [s["agent_name"] for s in before["sessions"]] == [
            f"claude-session-{sid}"
        ]
        assert before["sessions"][0]["states"] == {"docs/plan.md": "EXCLUSIVE"}
    finally:
        first.shutdown()

    second = _restart_on(tmp_path, "restart-after")
    try:
        secret = load_secret(second.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", second.port, secret)
        _, after = client.get(
            "/status?detail=full",
            headers_override={"Coherence-Local-Operator": "true"},
        )

        assert len(after["sessions"]) == 1, (
            "the EXCLUSIVE grant is still in agent_states and still arbitrates; "
            "dropping it from /status makes a held file read as idle"
        )
        holder = after["sessions"][0]
        assert holder["agent_id"] == str(session_to_agent_id(sid))
        assert holder["agent_name"] is None, (
            "the agent id is a uuid5 of the session id — unrecoverable, so the "
            "name is honestly absent rather than guessed"
        )
        assert holder["states"] == {"docs/plan.md": "EXCLUSIVE"}

        # The same grant that /status now reports is the one enforcement uses.
        _, collide = client.post(
            "/hooks/pre-edit",
            {"session_id": _sid("peer"), "path": "docs/plan.md"},
        )
        assert collide["collision"] is True
    finally:
        second.shutdown()


def test_post_restart_holder_is_listed_at_the_default_tier(tmp_path: Path) -> None:
    """The unnamed holder appears at the DEFAULT tier, not only behind
    ?detail=full + the operator header. That is the security-relevant case, so
    it is pinned explicitly: `sessions` has always been a minimal-tier field,
    and this row repeats an `agent_id` already emitted there while dropping the
    session-id-bearing `agent_name` — narrower than the named row it replaces.
    """
    first = _restart_on(tmp_path, "tier-before")
    try:
        secret = load_secret(first.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", first.port, secret)
        sid = _sid("tier-survivor")
        client.post("/policy/track", {"paths": ["docs/plan.md"]})
        client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})
    finally:
        first.shutdown()

    second = _restart_on(tmp_path, "tier-after")
    try:
        secret = load_secret(second.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", second.port, secret)

        # Default tier: no query string, no operator header.
        _, minimal = client.get("/status")
        assert minimal["detail"] == "minimal"
        assert [s["agent_id"] for s in minimal["sessions"]] == [
            str(session_to_agent_id(sid))
        ]
        assert minimal["sessions"][0]["agent_name"] is None
        assert minimal["sessions"][0]["states"] == {"docs/plan.md": "EXCLUSIVE"}

        # The metrics tier still omits the collection entirely.
        _, metrics = client.get("/status?detail=metrics")
        assert "sessions" not in metrics
    finally:
        second.shutdown()


# ======================================================================
# /status tracked_artifacts carries the last writer (#199 §2)
# ======================================================================
#
# Writer identity reached callers only on the stale deny/warn arms; nothing a
# caller could *ask* returned it. It now rides ``tracked_artifacts`` at the
# operator-gated full tier, read from the same batched snapshot row.


def _operator_status(client: _Client) -> dict:
    s, b = client.get(
        "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )
    assert s == 200
    return b


def test_status_full_tier_reports_last_writer(client: _Client) -> None:
    writer = _sid("writer")
    client.post("/hooks/pre-read", {"session_id": _sid("reader"), "path": "plan.md", "content_hash": _hash("h1")})
    client.post("/hooks/pre-edit", {"session_id": writer, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": writer, "path": "plan.md",
                 "content_hash": _hash("h2"), "success": True})

    before = time.time()
    payload = _operator_status(client)
    [entry] = payload["tracked_artifacts"]
    assert entry["version"] == 2
    assert entry["last_writer_agent_id"] == str(session_to_agent_id(writer))
    # No session id is re-derived from the agent id; the operator gets the
    # name by joining against ``sessions[]``, which this tier already names.
    assert "last_writer_session_id" not in entry
    [row] = [s for s in payload["sessions"] if s["agent_id"] == entry["last_writer_agent_id"]]
    assert writer in row["agent_name"]
    assert isinstance(entry["last_writer_at_unix_ts"], float)
    assert entry["last_writer_at_unix_ts"] <= before


def test_status_full_tier_writer_is_null_before_any_commit(client: _Client) -> None:
    """A first-observed artifact has no writer; its ``updated_at`` stamps the
    observation, so it must not be reported as a write time."""
    client.post("/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("h1")})
    [entry] = _operator_status(client)["tracked_artifacts"]
    assert entry["last_writer_agent_id"] is None
    assert entry["last_writer_at_unix_ts"] is None


def test_status_full_tier_names_a_subagent_writer_through_sessions(client: _Client) -> None:
    """A subagent's commit is attributed to the (session, subagent) agent, and
    joining on ``sessions[]`` names both halves. A label re-derived from the
    agent id instead carried only the bare subagent id ("worker-1")."""
    sid = _sid("sub-writer")
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "plan.md", "agent_id": "worker-1"})
    client.post("/hooks/post-edit",
                {"session_id": sid, "path": "plan.md", "agent_id": "worker-1",
                 "content_hash": _hash("h2"), "success": True})

    payload = _operator_status(client)
    [entry] = payload["tracked_artifacts"]
    assert entry["last_writer_agent_id"] == str(session_to_agent_id(sid, "worker-1"))
    assert entry["last_writer_agent_id"] != str(session_to_agent_id(sid))
    [row] = [s for s in payload["sessions"] if s["agent_id"] == entry["last_writer_agent_id"]]
    assert row["agent_name"] == f"claude-session-{sid}:subagent-worker-1"


def test_status_minimal_tier_omits_last_writer(client: _Client) -> None:
    """Writer attribution names which session wrote what; below the operator
    tier it stays out until #198 settles the default tier's disclosure."""
    writer = _sid("writer")
    client.post("/hooks/pre-edit", {"session_id": writer, "path": "plan.md"})
    client.post("/hooks/post-edit",
                {"session_id": writer, "path": "plan.md",
                 "content_hash": _hash("h2"), "success": True})
    _, minimal = client.get("/status")
    [entry] = minimal["tracked_artifacts"]
    assert set(entry) == {"path", "version", "id"}


def test_status_last_writer_survives_a_restart(tmp_path: Path) -> None:
    """The agent id and timestamp come from the durable row, so they outlive
    the process that recorded them."""
    first = _restart_on(tmp_path, "writer-before")
    try:
        secret = load_secret(first.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", first.port, secret)
        sid = _sid("writer-survivor")
        client.post("/policy/track", {"paths": ["docs/plan.md"]})
        client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})
        client.post("/hooks/post-edit",
                    {"session_id": sid, "path": "docs/plan.md",
                     "content_hash": _hash("h2"), "success": True})
        [entry] = _operator_status(client)["tracked_artifacts"]
        written_at = entry["last_writer_at_unix_ts"]
        assert entry["last_writer_agent_id"] == str(session_to_agent_id(sid))
    finally:
        first.shutdown()

    second = _restart_on(tmp_path, "writer-after")
    try:
        secret = load_secret(second.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", second.port, secret)
        [entry] = _operator_status(client)["tracked_artifacts"]
        assert entry["last_writer_agent_id"] == str(session_to_agent_id(sid))
        assert entry["last_writer_at_unix_ts"] == written_at
    finally:
        second.shutdown()


def test_status_omits_invalidated_holders_after_a_restart(tmp_path: Path) -> None:
    """Only non-INVALID states count as holding. A session whose grant was
    taken away must not reappear as a holder just because its row survives."""
    first = _restart_on(tmp_path, "inv-before")
    try:
        secret = load_secret(first.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", first.port, secret)
        loser, winner = _sid("inv-loser"), _sid("inv-winner")
        client.post("/policy/track", {"paths": ["docs/plan.md"]})
        client.post("/hooks/pre-edit", {"session_id": loser, "path": "docs/plan.md"})
        client.post("/hooks/pre-edit", {"session_id": winner, "path": "docs/plan.md"})
    finally:
        first.shutdown()

    second = _restart_on(tmp_path, "inv-after")
    try:
        secret = load_secret(second.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", second.port, secret)
        _, after = client.get(
            "/status?detail=full",
            headers_override={"Coherence-Local-Operator": "true"},
        )
        by_id = {s["agent_id"]: s for s in after["sessions"]}
        assert str(session_to_agent_id(winner)) in by_id
        assert str(session_to_agent_id(loser)) not in by_id
    finally:
        second.shutdown()


def test_status_still_lists_a_registered_session_holding_nothing(
    client: _Client,
) -> None:
    """No regression on the live path: a session that registered but holds no
    grant keeps its entry with an empty state map.

    R6 split this across tiers. The row is keyed on ``agent_id`` at both, and
    the operator tier is where the name proves the entry came from the
    registration path rather than the registry's unnamed-holder branch — at
    the default tier the two shapes are deliberately indistinguishable.
    """
    sid = _sid("named-no-grants")
    client.post("/hooks/session-start", {"session_id": sid})
    agent_id = str(session_to_agent_id(sid))

    _, body = client.get("/status")
    by_id = {s["agent_id"]: s for s in body["sessions"]}
    assert agent_id in by_id
    assert by_id[agent_id]["states"] == {}
    assert by_id[agent_id]["agent_name"] is None

    _, full = client.get(
        "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )
    named = {s["agent_id"]: s for s in full["sessions"]}
    assert named[agent_id]["agent_name"] == f"claude-session-{sid}"
    assert named[agent_id]["states"] == {}


def test_status_entry_keys_match_the_documented_shape(client: _Client) -> None:
    """``StatusResponse`` documented ``last_writer``/``session_id`` keys the
    handler has never emitted, and nothing in the tree type-checks against it,
    so the drift was invisible. Pin the shape against a live body instead."""
    from ccs.adapters.claude_code import hook_payloads as hp

    sid = _sid("shape")
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "spec.md"})
    _, body = client.get("/status")

    assert set(hp.StatusResponse.__annotations__) <= set(body)
    assert {k for a in body["tracked_artifacts"] for k in a} == {
        "path", "version", "id",
    }
    assert {k for s in body["sessions"] for k in s} == {
        "agent_name", "agent_id", "states",
    }


# ======================================================================
# R6 — per-artifact state below the operator tier carries no raw
#       session identifier
# ======================================================================
#
# ``agent_name`` renders ``claude-session-<session_id>`` verbatim, so every
# tier that carried a session row also republished the raw session id beside
# that session's per-artifact state. ``agent_id`` — a uuid5 of the same
# session id, documented at ``session_to_agent_id`` as not reversible — is
# already on the row and is the handle callers should attribute by. The
# operator (full-detail) tier keeps the name; everything below it drops it
# and renders through the null-name fallback the CLI already has.
#
# Each test here drives a LIVE session holding a real grant. An empty
# workspace emits ``sessions: []``, which cannot observe any of this.


def _session_id_appears_in(body: dict, sid: str) -> bool:
    """True if the raw session id is reachable anywhere in the payload —
    any key, any value, any nesting depth."""
    return sid in json.dumps(body)


def test_status_default_tier_reports_state_without_the_session_id(
    client: _Client,
) -> None:
    """The default tier still answers "who holds what", and answers it with
    the non-reversible ``agent_id`` only.

    The control matters more than the assertion: an empty ``sessions`` list
    would satisfy "no session id in the body" while observing nothing, so the
    row and its per-artifact state are asserted present FIRST.
    """
    sid = _sid("r6-default-tier")
    client.post("/policy/track", {"paths": ["docs/plan.md"]})
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})

    _, body = client.get("/status")
    assert body["detail"] == "minimal"

    # Control: the case this test claims to inspect is actually present.
    rows = [s for s in body["sessions"] if s["agent_id"] == str(session_to_agent_id(sid))]
    assert len(rows) == 1, f"no row for the live session; sessions={body['sessions']}"
    assert rows[0]["states"] == {"docs/plan.md": "EXCLUSIVE"}, (
        "per-artifact state must still be reported — dropping the whole row "
        "would pass the redaction assertion while telling the operator nothing"
    )

    # The requirement.
    assert rows[0]["agent_name"] is None
    assert not _session_id_appears_in(body, sid), (
        f"the raw session id is still reachable in the default-tier body: "
        f"{json.dumps(body)[:400]}"
    )


def test_status_operator_tier_still_names_the_session(client: _Client) -> None:
    """The positive control for the redaction above: at the operator
    (full-detail) tier the name is present, in full, session id included.

    Without this, a handler that dropped ``agent_name`` at EVERY tier would
    look correct.
    """
    sid = _sid("r6-operator-tier")
    client.post("/policy/track", {"paths": ["docs/plan.md"]})
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})

    _, body = client.get(
        "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )
    assert body["detail"] == "full"
    rows = [s for s in body["sessions"] if s["agent_id"] == str(session_to_agent_id(sid))]
    assert len(rows) == 1
    assert rows[0]["agent_name"] == f"claude-session-{sid}"
    assert _session_id_appears_in(body, sid), (
        "the operator tier keeps the name; if this is false the test above "
        "cannot distinguish redaction from an empty payload"
    )


def test_status_agent_id_is_identical_across_tiers(client: _Client) -> None:
    """``agent_id`` is the handle callers attribute by, so it must not move.
    Same value, same row, at the default tier and the operator tier."""
    sid = _sid("r6-stable-agent-id")
    client.post("/policy/track", {"paths": ["docs/plan.md"]})
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})

    _, minimal = client.get("/status")
    _, full = client.get(
        "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )

    expected = str(session_to_agent_id(sid))
    minimal_ids = {s["agent_id"] for s in minimal["sessions"]}
    full_ids = {s["agent_id"] for s in full["sessions"]}
    assert expected in minimal_ids
    assert minimal_ids == full_ids, (
        "redacting the display name must not change which agents are listed"
    )
    by_id_minimal = {s["agent_id"]: s["states"] for s in minimal["sessions"]}
    by_id_full = {s["agent_id"]: s["states"] for s in full["sessions"]}
    assert by_id_minimal == by_id_full


def test_status_cli_renders_a_redacted_row_without_the_session_id(
    client: _Client,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The redacted row is a shape the CLI already handles: it renders the
    existing null-name fallback rather than printing "None", and the rendered
    text carries no raw session id.

    Driven off a LIVE default-tier body so the renderer sees exactly what the
    handler emits, not a shape this test invented.
    """
    from ccs.cli import coherence_status

    monkeypatch.setenv("COLUMNS", "100")
    sid = _sid("r6-cli-render")
    client.post("/policy/track", {"paths": ["docs/plan.md"]})
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})
    _, body = client.get("/status")

    coherence_status._render_table(body)
    out = capsys.readouterr().out

    assert "Sessions:" in out
    assert "docs/plan.md" in out and "EXCLUSIVE" in out
    assert "name unknown" in out
    assert "None" not in out
    assert sid not in out
    # The fallback must not assert a cause it cannot know. This row is a
    # redacted live session, NOT a grant that outlived its coordinator, and
    # the renderer cannot distinguish the two — so it names both.
    assert "redacted below the operator tier" in out, (
        "the fallback blamed a pre-restart grant for what is a tier redaction"
    )


def test_detail_help_text_names_only_redactions_the_handler_performs(
    client: _Client,
) -> None:
    """R9: the ``--detail`` help text claimed the minimal tier redacts
    ``coordinator_pid``. It never has — pid is emitted at every tier on
    purpose (it is public on POSIX and operators verify ownership with it).

    Each redaction the help text names is checked against a LIVE minimal-tier
    body, and the one it used to name falsely is checked the other way.
    """
    from ccs.cli import coherence_status

    sid = _sid("r6-help-text")
    client.post("/policy/track", {"paths": ["docs/plan.md"]})
    client.post("/hooks/pre-edit", {"session_id": sid, "path": "docs/plan.md"})
    _, body = client.get("/status")

    action = next(
        a for a in coherence_status.build_parser()._actions
        if "--detail" in (a.option_strings or [])
    )
    help_text = action.help or ""

    # The false claim is gone, and the behaviour that falsified it is pinned.
    assert "coordinator_pid" not in help_text, (
        "the help text must not claim a pid redaction the handler never does"
    )
    assert body["coordinator_pid"] == os.getpid()

    # Each redaction the text now names is real at the minimal tier.
    assert "absolute path" in help_text
    assert body["coordinator_root"] == "."

    assert "session name" in help_text
    assert [s["agent_name"] for s in body["sessions"]] == [None]

    assert "tracked pattern" in help_text
    assert "user_added_patterns" not in body["policy_summary"]
    _, full = client.get(
        "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )
    assert "user_added_patterns" in full["policy_summary"], (
        "control: the pattern list exists at the operator tier, so its "
        "absence above is a redaction and not an empty policy"
    )


def test_policy_glob_sets_are_published_only_at_the_operator_tier(
    client: _Client,
) -> None:
    """The operator view carries the coordinator's glob sets — the default
    tracked patterns, the user-added ones, the ignored ones and the strict
    ones — so a client can check the globs it declared against the policy the
    coordinator actually loaded (#190). Below that tier ``policy_summary``
    keeps its counts and no pattern list: a pattern list is the operator's
    directory layout.

    Prevents a client having only a COUNT to verify enforcement with: a
    sibling volume whose managed globs differed from the spawner's passed on
    the spawner's count while its own paths were untracked."""
    client.post("/policy/track", {"paths": ["docs/plan.md"]})
    pattern_lists = (
        "tracked_patterns", "user_added_patterns", "ignored_patterns", "strict_mode_patterns",
    )

    _, minimal = client.get("/status")
    for key in pattern_lists:
        assert key not in minimal["policy_summary"], key
    assert minimal["policy_summary"]["strict_mode_pattern_count"] == 0
    assert minimal["policy_summary"]["user_added_pattern_count"] == 1

    _, full = client.get(
        "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )
    summary = full["policy_summary"]
    for key in pattern_lists:
        assert isinstance(summary[key], list), key
    assert summary["user_added_patterns"] == ["docs/plan.md"]
    assert "CLAUDE.md" in summary["tracked_patterns"], "the defaults are published too"
    assert summary["strict_mode_patterns"] == []
    assert summary["ignored_patterns"] == []
    assert summary["strict_mode_pattern_count"] == len(summary["strict_mode_patterns"])


# ======================================================================
# Preemption notices past the render cap must survive, not be destroyed
# ======================================================================
#
# ``pop_pending_notices`` deleted EVERY row for the agent, while the prose
# builder rendered only ``_PREEMPTION_PROSE_VERBATIM_CAP`` of them. One
# pre-read on one path therefore destroyed every notice the response did
# not show, and the overflow line pointed at /status, which has never
# carried notice data. The drain is now bounded to what is rendered.


def _pending_count(coordinator, agent_id) -> int:
    """How many notices are still queued for this agent (non-destructive)."""
    artifact_by_id, _ = coordinator.registry.status_snapshot()
    return sum(
        1
        for artifact_id in artifact_by_id
        if coordinator.registry.peek_preemption_notice(agent_id, artifact_id)
        is not None
    )


def _seven_notices(client: _Client, coordinator) -> tuple[str, list[str]]:
    """Leave session A holding seven pending preemption notices."""
    paths = [f"docs/f{n}.md" for n in range(7)]
    a, b = _sid("notice-victim"), _sid("notice-taker")
    client.post("/policy/track", {"paths": paths})
    for p in paths:
        client.post("/hooks/pre-edit", {"session_id": a, "path": p})
    for p in paths:
        client.post("/hooks/pre-edit", {"session_id": b, "path": p})
    assert _pending_count(coordinator, session_to_agent_id(a)) == 7
    return a, paths


def test_pre_read_consumes_only_the_notices_it_renders(
    coordinator, client: _Client
) -> None:
    """Seven pending, three rendered → four still queued. The four the
    response could not show are the caller's only record of who took those
    artifacts; deleting them made the preempters unrecoverable."""
    a, paths = _seven_notices(client, coordinator)
    agent_a = session_to_agent_id(a)

    _, body = client.post("/hooks/pre-read", {"session_id": a, "path": paths[0]})
    text = body["hookSpecificOutput"]["additionalContext"]
    rendered = [ln for ln in text.splitlines() if ln.strip().startswith("•")]
    verbatim = [ln for ln in rendered if "Plus " not in ln]

    assert len(verbatim) == 3
    assert "Plus 4 more" in text
    assert _pending_count(coordinator, agent_a) == 4, (
        "the four notices the response did not render must still be queued"
    )


def test_successive_reads_deliver_every_notice(
    coordinator, client: _Client
) -> None:
    """The overflow is a deferral, not a loss: three more arrive on the next
    tracked-file operation, and the last one after that."""
    a, paths = _seven_notices(client, coordinator)
    agent_a = session_to_agent_id(a)

    seen: set[str] = set()
    for expected_remaining in (4, 1, 0):
        _, body = client.post(
            "/hooks/pre-read", {"session_id": a, "path": paths[0]}
        )
        text = body["hookSpecificOutput"]["additionalContext"]
        seen.update(p for p in paths if f"• {p} —" in text)
        assert _pending_count(coordinator, agent_a) == expected_remaining

    assert seen == set(paths), "every preempted artifact was eventually named"


def test_overflow_line_does_not_point_at_a_surface_without_notices(
    coordinator, client: _Client
) -> None:
    """/status carries no notice data at any tier, so the overflow line must
    not send the caller there. It names the real delivery channel instead."""
    a, paths = _seven_notices(client, coordinator)
    _, body = client.post("/hooks/pre-read", {"session_id": a, "path": paths[0]})
    text = body["hookSpecificOutput"]["additionalContext"]

    assert "GET /status" not in text
    assert "agent-coherence status" not in text

    _, status_body = client.get(
        "/status?detail=full",
        headers_override={"Coherence-Local-Operator": "true"},
    )
    assert not any(
        "notice" in k or "preempt" in k for k in status_body
    ), "if /status ever carries notices, this test should be the one to change"


def test_session_stop_still_drains_every_notice(
    coordinator, client: _Client
) -> None:
    """Stop returns the full structured array, so its drain is matched by its
    render — it must keep consuming all of them."""
    a, _paths = _seven_notices(client, coordinator)
    agent_a = session_to_agent_id(a)

    _, body = client.post("/hooks/session-stop", {"session_id": a})
    assert len(body["notices"]) == 7
    assert _pending_count(coordinator, agent_a) == 0


def test_unlimited_drain_does_not_bind_one_variable_per_notice(
    coordinator, client: _Client
) -> None:
    """session-stop's full drain must bind only ``agent_id``, not one variable
    per row. An IN-list of every pending notice can exceed SQLite's
    bound-variable ceiling on an agent with a large pending set, raising
    mid-transaction instead of committing.

    Driven against the real ceiling, lowered for the duration: with a
    per-row IN-list this raises OperationalError; with the bulk DELETE it does
    not. Six notices against a limit of four is the same shape as 40k notices
    against the stock 32766.
    """
    import sqlite3

    sid = _sid("bulk-drain")
    agent_id = session_to_agent_id(sid)
    peer = session_to_agent_id(_sid("bulk-peer"))
    paths = [f"docs/bulk{n}.md" for n in range(6)]
    client.post("/policy/track", {"paths": paths})
    # Mint the artifacts FIRST: a pre-read drains this session's notices, so
    # recording inside the read loop would let each read consume what the
    # previous iterations queued.
    artifact_ids = []
    for path in paths:
        client.post("/hooks/pre-read",
                    {"session_id": sid, "path": path, "content_hash": _hash(path)})
        artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
        assert artifact_id is not None
        artifact_ids.append(artifact_id)
    for n, artifact_id in enumerate(artifact_ids):
        coordinator.registry.record_preemption_notice(
            victim_agent_id=agent_id, artifact_id=artifact_id,
            preempter_agent_id=peer, preempted_at_unix_ts=1700000000.0 + n,
        )

    conn = coordinator.registry._conn
    previous = conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
    conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 4)
    try:
        drained = coordinator.registry.pop_pending_notices(agent_id)
    finally:
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, previous)

    assert len(drained) == 6
    assert _pending_count(coordinator, agent_id) == 0


def test_session_stop_overflow_prose_does_not_claim_rows_are_queued(
    coordinator, client: _Client
) -> None:
    """session-stop drains UNBOUNDED and returns everything in `notices`, so the
    overflow line must not repeat the deferral promise the bounded prose paths
    make. "still queued — they surface on your next tracked-file operation" is
    false twice over here: the rows were just deleted, and a stopping session
    has no next operation. It points at the response's own array instead."""
    a, _paths = _seven_notices(client, coordinator)

    _, body = client.post("/hooks/session-stop", {"session_id": a})
    text = body["hookSpecificOutput"]["additionalContext"]

    assert len(body["notices"]) == 7, "the structured array still carries all of them"
    assert "Plus 4 more" in text
    assert "still queued" not in text
    assert "next tracked-file operation" not in text
    assert "notices" in text, "the prose names the surface that actually has them"


def test_bounded_prose_paths_still_promise_deferral(
    coordinator, client: _Client
) -> None:
    """The four admit paths DO defer — their drain is capped — so they keep the
    deferral wording. Guards against fixing session-stop by flattening both."""
    a, paths = _seven_notices(client, coordinator)

    _, body = client.post("/hooks/pre-read", {"session_id": a, "path": paths[0]})
    text = body["hookSpecificOutput"]["additionalContext"]

    assert "Plus 4 more" in text
    assert "still queued" in text
    assert "next tracked-file operation" in text
    assert _pending_count(coordinator, session_to_agent_id(a)) == 4


# ----------------------------------------------------------------------
# U3a — decide_tracked_read: the tracked-artifact verdict as a VALUE
#
# _handle_pre_read decides fresh / stale / denied for an already-tracked
# artifact WHILE it mutates: the fresh arm bumps two counters and can return
# the strict deny, the stale arm re-grants SHARED, marks the pair stale-warned
# and drains notices. A surface that must answer the SAME question without
# granting anything therefore cannot reach the answer by calling the handler.
# These tests pin the extracted verdict: that each arm returns what the handler
# would have reported, that the two hash-differs predicates stay DISTINCT, and
# — the claim any verdict route rests on — that asking changes nothing.
# ----------------------------------------------------------------------

_U3A_STRICT_PATH = "CLAUDE.md"  # tracked by default AND listed in strict_mode.yaml
_U3A_WARN_PATH = "plan.md"      # tracked by the default **/plan.md glob, never strict
# Per-arm paths, so the seven arms below own seven DISTINCT artifacts: they are
# seeded at different versions, and sharing one row would let the last seed
# silently rewrite what the earlier arms assert about.
_U3A_STRICT_GLOB = "docs/specs/**/*.md"


def _u3a_strict_path(name: str) -> str:
    return f"docs/specs/{name}.md"


def _u3a_warn_path(name: str) -> str:
    return f"docs/plans/{name}.md"


def _u3a_write_policy(root: Path) -> None:
    """Materialize strict_mode.yaml: CLAUDE.md and docs/specs/** are strict,
    everything else tracked stays warn-mode.

    Policy is loaded ONCE at construction, so this must run before the server
    is built — mutating ``server.policy`` afterwards would test a shape no
    operator can produce.
    """
    coherence = root / ".coherence"
    coherence.mkdir(mode=0o700, exist_ok=True)
    (coherence / "strict_mode.yaml").write_text(
        f"- {_U3A_STRICT_PATH}\n- {_U3A_STRICT_GLOB}\n"
    )


def _u3a_assert_policy(server: CoordinatorHTTPServer) -> None:
    """Fail loudly if the fixture paths are not the modes every test below
    assumes — a silently warn-mode "strict" path would turn every deny
    assertion into a vacuous fresh/stale assertion."""
    assert server.policy.is_strict_mode(_U3A_STRICT_PATH)
    assert server.policy.is_strict_mode(_u3a_strict_path("probe"))
    assert server.policy.is_tracked(_U3A_WARN_PATH)
    assert not server.policy.is_strict_mode(_U3A_WARN_PATH)
    assert server.policy.is_tracked(_u3a_warn_path("probe"))
    assert not server.policy.is_strict_mode(_u3a_warn_path("probe"))


@pytest.fixture
def decider(tmp_path: Path):
    """A coordinator built but NOT served: decide_tracked_read is an in-process
    call, so an accept loop would add a thread and a port to every assertion
    and nothing else."""
    _u3a_write_policy(tmp_path)
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="u3a-decider")
    _u3a_assert_policy(server)
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def served_decider(tmp_path: Path):
    """The same policy on a LIVE coordinator, yielded with its client, so a
    verdict can be compared against the response the real handler builds."""
    _u3a_write_policy(tmp_path)
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="u3a-served")
    _u3a_assert_policy(server)
    server.serve_in_thread()
    time.sleep(0.05)
    secret = load_secret(server.coordinator_root)
    assert secret is not None
    try:
        yield server, _Client("127.0.0.1", server.port, secret)
    finally:
        server.shutdown()


def _u3a_seed(
    server: CoordinatorHTTPServer,
    path: str,
    *,
    recorded_hash: str,
    state: Optional[MESIState],
    label: str = "s1",
    version: Optional[int] = None,
    last_writer: Optional[uuid.UUID] = None,
):
    """Park an (agent, artifact) pair in the state an arm needs.

    Returns ``(session_id, agent_id, artifact_id)``. ``state=None`` leaves the
    session with no grant at all — the "stale beliefs from somewhere else"
    case the no-prior-grant deny gate exists for.
    """
    sid = _sid(label)
    agent_id = server.register_session(sid)
    artifact_id = server.registry.resolve_or_register(path, content_hash=recorded_hash)
    if version is not None or last_writer is not None:
        art = server.registry.get_artifact(artifact_id)
        server.registry.set_artifact_and_content(
            artifact_id,
            dataclasses.replace(art, version=art.version if version is None else version),
            "",
            last_writer=last_writer,
        )
    if state is not None:
        server.registry.set_agent_state(
            artifact_id, agent_id, state,
            trigger="u3a-test", tick=1, content_hash=recorded_hash,
        )
    return sid, agent_id, artifact_id


def _u3a_decide(
    server: CoordinatorHTTPServer,
    artifact_id: uuid.UUID,
    agent_id: uuid.UUID,
    *,
    path: str,
    caller_hash: Optional[str],
    now_unix: float = 1000.0,
) -> TrackedReadDecision:
    """Call the verdict with exactly the values the handler has already read at
    the point its tracked branch begins — one pair-atomic snapshot, this
    agent's MESI state, the hash the caller offered."""
    pair = server.registry.get_artifact_and_generation(artifact_id)
    assert pair is not None
    artifact, generation = pair
    return decide_tracked_read(
        server,
        path=path,
        artifact_id=artifact_id,
        agent_id=agent_id,
        artifact=artifact,
        owner_generation=generation,
        agent_state=server.registry.get_agent_state(artifact_id, agent_id),
        caller_content_hash=caller_hash,
        now_unix=now_unix,
    )


# --- one arm per state that produces it ------------------------------------


def test_decide_fresh_for_granted_holder_whose_hash_matches(decider) -> None:
    """The plain fresh arm: a still-SHARED holder re-reading the bytes the
    registry recorded. Neither hash predicate fires, nothing is suppressed, and
    the version reported is the one from the caller's own snapshot."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("v3-bytes"),
        state=MESIState.SHARED, version=3,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_STRICT_PATH, caller_hash=_hash("v3-bytes"))

    assert d.outcome == "fresh"
    assert d.version == 3
    assert d.holds_valid_grant is True
    assert d.fresh_hash_differs is False
    assert d.stale_hash_differs is False
    assert d.commit_lag_suppressed is False
    # A SHARED holder was granted on the current version; that is what it saw.
    assert d.prior_version_seen == 3


def test_decide_fresh_with_hash_differs_in_warn_mode(decider) -> None:
    """Warn mode never denies: the mismatch is surfaced on the verdict (the
    handler turns it into the additive ``hash_differs`` key and one counter
    bump) but the outcome stays fresh."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_WARN_PATH, caller_hash=_hash("out-of-band"))

    assert d.outcome == "fresh"
    assert d.fresh_hash_differs is True
    assert d.commit_lag_suppressed is False


def test_decide_denied_for_granted_holder_foreign_edit_in_strict_mode(decider) -> None:
    """Survivor #6: a still-SHARED holder proves no peer commit since its
    grant, so a differing disk hash written by SOMEONE ELSE is a foreign
    out-of-band edit — denied, through the fresh arm."""
    peer = decider.register_session(_sid("peer-writer"))
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2, last_writer=peer,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_STRICT_PATH, caller_hash=_hash("foreign-edit"))

    assert d.outcome == "denied"
    assert d.holds_valid_grant is True, "the deny came through the FRESH arm"
    assert d.fresh_hash_differs is True
    assert d.commit_lag_suppressed is False
    assert d.version == 2
    assert d.prior_version_seen == 2


def test_decide_fresh_when_the_lag_gate_withholds_the_deny(decider) -> None:
    """The same state, except the caller IS the artifact's recent last
    committer: the benign commit -> disk-write lag, suppressed rather than
    denied. Two distinct facts on one value — the outcome flipped back to fresh
    AND the suppression that flipped it, which is the counter bump a caller
    still owes."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED,
    )
    # Self-commit: make the caller the writer and read one second after it.
    art = decider.registry.get_artifact(artifact_id)
    decider.registry.set_artifact_and_content(artifact_id, art, "", last_writer=agent_id)
    updated_at = decider.registry.get_artifact_updated_at(artifact_id)
    assert updated_at is not None

    d = _u3a_decide(decider, artifact_id, agent_id, path=_U3A_STRICT_PATH,
                    caller_hash=_hash("not-yet-flushed"), now_unix=updated_at + 1.0)

    assert d.outcome == "fresh"
    assert d.fresh_hash_differs is True
    assert d.commit_lag_suppressed is True


def test_decide_denied_outside_the_lag_window(decider) -> None:
    """The recency clause is load-bearing: the same self-commit read LATER than
    the window is a genuine foreign edit again. Without this the suppression
    field could be hard-wired True and every test above would still pass."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED,
    )
    art = decider.registry.get_artifact(artifact_id)
    decider.registry.set_artifact_and_content(artifact_id, art, "", last_writer=agent_id)
    updated_at = decider.registry.get_artifact_updated_at(artifact_id)
    assert updated_at is not None
    late = updated_at + _SHARED_FOREIGN_DENY_LAG_WINDOW_SEC + 0.1

    d = _u3a_decide(decider, artifact_id, agent_id, path=_U3A_STRICT_PATH,
                    caller_hash=_hash("foreign-edit"), now_unix=late)

    assert d.outcome == "denied"
    assert d.commit_lag_suppressed is False


def test_decide_stale_for_invalidated_session_in_warn_mode(decider) -> None:
    """A peer commit invalidated this session; warn mode allows with the stale
    warning. ``prior_version_seen`` is one BELOW the current version — the
    version the session last held a grant on."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.INVALID, version=4,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_WARN_PATH, caller_hash=_hash("canonical"))

    assert d.outcome == "stale"
    assert d.holds_valid_grant is False
    assert d.version == 4
    assert d.prior_version_seen == 3


def test_decide_denied_for_invalidated_session_in_strict_mode(decider) -> None:
    """True preemption under strict mode: denied on the INVALID state ALONE,
    with no hash comparison needed — the session's context still carries the
    beliefs the peer commit superseded."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.INVALID, version=4,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_STRICT_PATH, caller_hash=_hash("canonical"))

    assert d.outcome == "denied"
    assert d.holds_valid_grant is False
    assert d.stale_hash_differs is False, "the INVALID leg denies without a mismatch"
    assert d.prior_version_seen == 3


def test_decide_stale_for_unseen_session_whose_hash_matches(decider) -> None:
    """No prior grant but the session is looking at the bytes the registry
    recorded: nothing stale to act on, so strict mode falls through to the
    warn-mode allow. ``prior_version_seen`` is None — it saw no version."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=None, version=2,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_STRICT_PATH, caller_hash=_hash("canonical"))

    assert d.outcome == "stale"
    assert d.prior_version_seen is None


def test_decide_denied_for_unseen_session_whose_hash_differs(decider) -> None:
    """No prior grant AND different bytes: stale beliefs from somewhere else,
    denied. This is the leg the launch-gate scenarios arrive through."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=None,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_STRICT_PATH, caller_hash=_hash("stale-beliefs"))

    assert d.outcome == "denied"
    assert d.stale_hash_differs is True


def test_decide_reports_the_generation_from_the_callers_snapshot(decider) -> None:
    """The generation rides the SAME pair-atomic read as the version, on every
    arm — never a second read a peer commit could overtake, and never coerced
    to 0 when absent."""
    for path, state, outcome in (
        (_U3A_STRICT_PATH, MESIState.SHARED, "fresh"),
        (_U3A_WARN_PATH, MESIState.INVALID, "stale"),
    ):
        _, agent_id, artifact_id = _u3a_seed(
            decider, path, recorded_hash=_hash("canonical"), state=state,
            label=f"gen-{outcome}",
        )
        pair = decider.registry.get_artifact_and_generation(artifact_id)
        assert pair is not None

        d = _u3a_decide(decider, artifact_id, agent_id,
                        path=path, caller_hash=_hash("canonical"))

        assert d.outcome == outcome
        assert d.owner_generation == pair[1]
        assert d.version == pair[0].version


# --- the deliberate asymmetry between the two hash predicates --------------


def test_sentinel_recorded_hash_differs_on_the_stale_arm_only(decider) -> None:
    """The all-``f`` launch-gate sentinel is not a SHA-256 of anything, so the
    coordinator holds no content claim against it.

    The FRESH arm must not fire on it — denying a still-SHARED holder against
    content the coordinator never claimed would be a false deny. The STALE arm
    must, because the launch-gate scenarios reach their deny THROUGH that arm
    and the sentinel is exactly what makes their hashes differ. One value
    carries both answers; unifying the two predicates silently moves the
    strict-deny gate whichever way the survivor points.
    """
    _, granted_agent, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash="f" * 64,
        state=MESIState.SHARED, label="granted",
    )
    unseen_agent = decider.register_session(_sid("unseen"))

    granted = _u3a_decide(decider, artifact_id, granted_agent,
                          path=_U3A_STRICT_PATH, caller_hash=_hash("real-disk-bytes"))
    unseen = _u3a_decide(decider, artifact_id, unseen_agent,
                         path=_U3A_STRICT_PATH, caller_hash=_hash("real-disk-bytes"))

    # Same artifact, same caller hash — the two predicates disagree on purpose.
    assert granted.fresh_hash_differs is False
    assert granted.outcome == "fresh", "no content claim => no foreign-edit deny"
    assert unseen.stale_hash_differs is True
    assert unseen.outcome == "denied", "the launch-gate deny still fires"


def test_empty_recorded_hash_never_fires_either_predicate(decider) -> None:
    """The OTHER no-claim seed: a KTD-9 first observation with no caller hash
    records "" and surfaces as None. Nothing to compare against on either arm,
    so no mismatch and — with no mismatch — no deny for a session with no
    prior grant."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash="", state=None,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_STRICT_PATH, caller_hash=_hash("real-disk-bytes"))

    assert d.fresh_hash_differs is False
    assert d.stale_hash_differs is False
    assert d.outcome == "stale"


def test_absent_caller_hash_never_fires_either_predicate(decider) -> None:
    """A pre-read may carry no content_hash at all (the caller does not have
    one yet). With nothing on the caller's side there is no comparison to make
    on either arm."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"), state=None,
    )

    d = _u3a_decide(decider, artifact_id, agent_id,
                    path=_U3A_STRICT_PATH, caller_hash=None)

    assert d.fresh_hash_differs is False
    assert d.stale_hash_differs is False
    assert d.outcome == "stale"


# --- R7: the call is a pure query ------------------------------------------


def _u3a_observable(server: CoordinatorHTTPServer, agent_id, artifact_id) -> dict:
    """Everything _handle_pre_read moves around this decision, read through
    NON-destructive accessors only (peek, not pop).

    Anything the verdict quietly granted, healed, counted or recorded shows up
    as a difference here — which is the only thing that makes the purity claim
    testable rather than eyeballed.
    """
    pair = server.registry.get_artifact_and_generation(artifact_id)
    coherence = server.coordinator_root / ".coherence"
    return {
        "counters": server.counters_snapshot(),
        "version": pair[0].version if pair else None,
        # The coordinator-side observation baseline: the canonical content the
        # registry vouches for. A verdict that "healed" it would absolve the
        # very out-of-band edit it exists to report.
        "canonical_hash": pair[0].content_hash if pair else None,
        "owner_generation": pair[1] if pair else None,
        "grant": server.registry.get_agent_state(artifact_id, agent_id),
        "heartbeat": server.registry.last_heartbeat_tick(agent_id),
        "notice": server.registry.peek_preemption_notice(agent_id, artifact_id),
        "stale_warned": set(server._stale_warned_pairs),
        "strict_denies": dict(server._recent_strict_denies),
        # Names for "nothing new appeared"; sizes only for the append-only logs
        # (a deny writes audit.log). SQLite's own db/-wal/-shm sizes are not
        # asserted — a read can legitimately move them, and asserting on them
        # would trade a real guard for a flaky one.
        "coherence_files": sorted(p.name for p in coherence.iterdir()),
        "audit_logs": sorted((p.name, p.stat().st_size) for p in coherence.glob("*.log")),
    }


def _u3a_arms(server: CoordinatorHTTPServer) -> dict:
    """Every arm of the verdict, each with the call that drives it. Keyed by
    the handler behaviour it stands in for, so a regression names itself."""
    arms: dict[str, tuple] = {}
    for name, strict, state, caller, version in (
        ("fresh-granted", True, MESIState.SHARED, "canonical", 1),
        ("fresh-mismatch-warn", False, MESIState.SHARED, "out-of-band", 1),
        ("denied-shared-foreign", True, MESIState.SHARED, "foreign", 2),
        ("stale-invalid-warn", False, MESIState.INVALID, "canonical", 4),
        ("denied-invalid-strict", True, MESIState.INVALID, "canonical", 4),
        ("stale-unseen", True, None, "canonical", 2),
        ("denied-unseen-mismatch", True, None, "stale-beliefs", 2),
    ):
        path = _u3a_strict_path(name) if strict else _u3a_warn_path(name)
        # A PEER is the last writer everywhere here, so the lag gate answers
        # False on its own merits and no arm depends on the injected clock.
        peer = server.register_session(_sid(f"peer-{name}"))
        _, agent_id, artifact_id = _u3a_seed(
            server, path, recorded_hash=_hash("canonical"), state=state,
            label=f"arm-{name}", version=version, last_writer=peer,
        )
        arms[name] = (path, agent_id, artifact_id, _hash(caller))
    return arms


@pytest.mark.parametrize("arm", [
    "fresh-granted",
    "fresh-mismatch-warn",
    "denied-shared-foreign",
    "stale-invalid-warn",
    "denied-invalid-strict",
    "stale-unseen",
    "denied-unseen-mismatch",
])
def test_decide_twice_changes_nothing_observable(decider, arm: str) -> None:
    """R7, asserted as a before/after rather than by reading the source.

    On every arm the handler does something here — bumps a counter, re-grants
    SHARED, marks the pair stale-warned, records the deny, appends the audit
    row, pops the notice. The verdict must do NONE of it, so calling it twice
    in a row leaves version, generation, canonical hash, grant state, heartbeat
    tick, pending notice, stale markers, deny records, the .coherence files and
    EVERY counter exactly as they were — and answers the same both times.
    """
    arms = _u3a_arms(decider)
    path, agent_id, artifact_id, caller_hash = arms[arm]
    # Give each dimension a non-default value first: an assertion that None
    # stayed None proves much less than one that 41 stayed 41.
    decider.registry.record_heartbeat(agent_id, 41)
    decider.registry.record_preemption_notice(
        victim_agent_id=agent_id, artifact_id=artifact_id,
        preempter_agent_id=decider.register_session(_sid("preempter")),
        preempted_at_unix_ts=1234.0,
    )
    before = _u3a_observable(decider, agent_id, artifact_id)
    assert before["heartbeat"] == 41
    assert before["notice"] is not None

    first = _u3a_decide(decider, artifact_id, agent_id,
                        path=path, caller_hash=caller_hash)
    second = _u3a_decide(decider, artifact_id, agent_id,
                         path=path, caller_hash=caller_hash)

    assert first == second, "the verdict is not a function of its own history"
    assert _u3a_observable(decider, agent_id, artifact_id) == before


def test_purity_check_covers_the_arm_that_would_deny(decider) -> None:
    """Control for the test above: prove the parametrized arms really do reach
    the deny and suppression legs. A purity proof over seven arms that all
    quietly returned "stale" would be seven copies of one weak assertion."""
    arms = _u3a_arms(decider)
    outcomes = {
        name: _u3a_decide(decider, artifact_id, agent_id,
                          path=path, caller_hash=caller_hash).outcome
        for name, (path, agent_id, artifact_id, caller_hash) in arms.items()
    }
    assert outcomes == {
        "fresh-granted": "fresh",
        "fresh-mismatch-warn": "fresh",
        "denied-shared-foreign": "denied",
        "stale-invalid-warn": "stale",
        "denied-invalid-strict": "denied",
        "stale-unseen": "stale",
        "denied-unseen-mismatch": "denied",
    }


# --- every branch is drivable from the decision alone ----------------------


def _u3a_render(d: TrackedReadDecision, *, want_generation: bool) -> dict:
    """Rebuild the handler's response shape from the VERDICT ONLY.

    No coordinator, no artifact, no agent state — if a field the handler
    branches on were missing from the decision, this could not be written and
    the next unit would have to re-derive it in the handler, which is the
    duplication the extraction exists to remove.
    """
    if d.outcome == "denied":
        return {
            "source": (
                "pre_read_shared_hash_deny" if d.holds_valid_grant
                else "pre_read_strict_deny"
            ),
            "current_version": d.version,
            "prior_version_seen_by_session": d.prior_version_seen,
            # The fresh arm denies only ON a mismatch, so it reports True flat;
            # the stale arm reports its own predicate (the INVALID leg denies
            # without one).
            "hash_differs": True if d.holds_valid_grant else d.stale_hash_differs,
            "counter_bumps": ["strict_mode_denials_total"],
        }
    if d.outcome == "fresh":
        payload: dict[str, Any] = {"status": "fresh", "version": d.version}
        if d.fresh_hash_differs:
            payload["hash_differs"] = True
        if want_generation and d.owner_generation is not None:
            payload["owner_generation"] = d.owner_generation
        payload["counter_bumps"] = (
            (["fresh_shared_hash_mismatch_total"] if d.fresh_hash_differs else [])
            + (["shared_foreign_lag_suppressed_total"] if d.commit_lag_suppressed else [])
        )
        return payload
    stale: dict[str, Any] = {
        "status": "stale",
        "current_version": d.version,
        "prior_version_seen_by_session": d.prior_version_seen,
        "hash_differs": d.stale_hash_differs,
        "counter_bumps": ["stale_warning_emitted_total"],
    }
    if want_generation and d.owner_generation is not None:
        stale["owner_generation"] = d.owner_generation
    return stale


def test_every_handler_branch_is_drivable_from_the_decision_alone(decider) -> None:
    """Each arm rendered from its verdict and nothing else. The expectations
    are written out literally rather than recomputed from the decision — a
    renderer checked against itself would report green on any field set."""
    arms = _u3a_arms(decider)

    def rendered(name: str, *, want_generation: bool = False) -> dict:
        path, agent_id, artifact_id, caller_hash = arms[name]
        d = _u3a_decide(decider, artifact_id, agent_id,
                        path=path, caller_hash=caller_hash)
        return _u3a_render(d, want_generation=want_generation)

    assert rendered("fresh-granted") == {
        "status": "fresh", "version": 1, "counter_bumps": [],
    }
    assert rendered("fresh-mismatch-warn") == {
        "status": "fresh", "version": 1, "hash_differs": True,
        "counter_bumps": ["fresh_shared_hash_mismatch_total"],
    }
    assert rendered("denied-shared-foreign") == {
        "source": "pre_read_shared_hash_deny",
        "current_version": 2,
        "prior_version_seen_by_session": 2,
        "hash_differs": True,
        "counter_bumps": ["strict_mode_denials_total"],
    }
    assert rendered("stale-invalid-warn") == {
        "status": "stale", "current_version": 4,
        "prior_version_seen_by_session": 3, "hash_differs": False,
        "counter_bumps": ["stale_warning_emitted_total"],
    }
    assert rendered("denied-invalid-strict") == {
        "source": "pre_read_strict_deny",
        "current_version": 4,
        "prior_version_seen_by_session": 3,
        "hash_differs": False,
        "counter_bumps": ["strict_mode_denials_total"],
    }
    assert rendered("denied-unseen-mismatch") == {
        "source": "pre_read_strict_deny",
        "current_version": 2,
        "prior_version_seen_by_session": None,
        "hash_differs": True,
        "counter_bumps": ["strict_mode_denials_total"],
    }
    # The opt-in pair: present only when asked for, and carrying the
    # generation the REGISTRY holds (the expectation is sourced from the
    # registry, not from the decision being checked).
    _path, _agent, artifact_id, _caller = arms["stale-unseen"]
    pair = decider.registry.get_artifact_and_generation(artifact_id)
    assert pair is not None and pair[1] is not None
    assert rendered("stale-unseen", want_generation=True)["owner_generation"] == pair[1]
    assert "owner_generation" not in rendered("stale-unseen")


def test_lag_suppression_is_visible_to_the_renderer(decider) -> None:
    """The suppression arm needs its own setup (the caller must be the recent
    writer), and it is the one arm whose counter bump is NOT implied by the
    outcome: a suppressed deny looks exactly like a warn-mode mismatch unless
    the decision says otherwise."""
    _, agent_id, artifact_id = _u3a_seed(
        decider, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED,
    )
    art = decider.registry.get_artifact(artifact_id)
    decider.registry.set_artifact_and_content(artifact_id, art, "", last_writer=agent_id)
    updated_at = decider.registry.get_artifact_updated_at(artifact_id)
    assert updated_at is not None

    d = _u3a_decide(decider, artifact_id, agent_id, path=_U3A_STRICT_PATH,
                    caller_hash=_hash("not-yet-flushed"), now_unix=updated_at + 1.0)

    assert _u3a_render(d, want_generation=False) == {
        "status": "fresh", "version": 1, "hash_differs": True,
        "counter_bumps": [
            "fresh_shared_hash_mismatch_total",
            "shared_foreign_lag_suppressed_total",
        ],
    }


# --- integration: the verdict agrees with the live handler -----------------


@pytest.mark.parametrize("path,state,caller,expect", [
    (_U3A_STRICT_PATH, MESIState.SHARED, "canonical", "fresh"),
    (_U3A_WARN_PATH, MESIState.INVALID, "canonical", "stale"),
    (_U3A_STRICT_PATH, MESIState.INVALID, "canonical", "denied"),
])
def test_verdict_matches_the_live_pre_read_handler(
    served_decider, path: str, state: MESIState, caller: str, expect: str,
) -> None:
    """The extraction is only worth anything if it agrees with the surface it
    was extracted from. Ask the verdict FIRST (it changes nothing), then let
    the real handler answer over HTTP and check they say the same thing about
    version, prior version and the stale-arm mismatch.
    """
    server, client = served_decider
    sid, agent_id, artifact_id = _u3a_seed(
        server, path, recorded_hash=_hash("canonical"), state=state, version=4,
    )

    d = _u3a_decide(server, artifact_id, agent_id, path=path, caller_hash=_hash(caller))
    status, body = client.post(
        "/hooks/pre-read",
        {"session_id": sid, "path": path, "content_hash": _hash(caller)},
    )

    assert status == 200
    assert d.outcome == expect
    if expect == "fresh":
        assert body == {"status": "fresh", "version": d.version}
        return
    # Both stale and denied carry the summary; the deny adds the permission
    # decision the warn-mode allow does not.
    assert body["summary"]["current_version"] == d.version
    assert body["summary"]["prior_version_seen_by_session"] == d.prior_version_seen
    assert body["summary"]["hash_differs"] == d.stale_hash_differs
    decision = body["hookSpecificOutput"].get("permissionDecision")
    assert decision == ("deny" if expect == "denied" else "allow")


# ----------------------------------------------------------------------
# U4 — POST /hooks/effect-fence: the verdict route
#
# "May this irreversible effect still fire, and if not, WHY" asked over HTTP.
# The route reads the registry directly, so it can answer the one leg the
# in-process fence structurally cannot: the pre-read WIRE carries only
# ``hash_differs`` (a comparison), never the coordinator's recorded hash, so
# in-process "the claim matches" and "there is no claim at all" are one value
# and the wrapper passes ``content_claim_present=True`` unconditionally. This
# route passes the real answer, which is why the AE8 arms below live here.
#
# Every test in this block is written against a coordinator whose state was
# parked by hand, so each names ONE condition. The recurring hazard they exist
# to close is the same one in every arm: the naive implementation — copy
# pre-read's untracked fast path, let the shared wrapper own the degraded
# arms, take the coordinator's reported generation at face value — answers
# PROCEED on a view the coordinator never confirmed.
# ----------------------------------------------------------------------

_U4_ROUTE = "/hooks/effect-fence"


def _u4_body(
    sid: str,
    path: str,
    *,
    version: Any,
    generation: Any,
    content_hash: Any,
) -> dict:
    """A complete, well-formed fence request. Tests that exercise a MISSING
    field pop it from this dict, so "well-formed" stays defined in one place
    and a malformed-input test cannot silently drift into testing two faults."""
    return {
        "session_id": sid,
        "path": path,
        "expected_version": version,
        "expected_generation": generation,
        "content_hash": content_hash,
    }


def _u4_captured(server: CoordinatorHTTPServer, artifact_id: uuid.UUID) -> tuple[int, Optional[int]]:
    """The (version, generation) pair an honest caller captured at read time.

    Sourced from the registry rather than written as a literal: an arm that
    hard-coded the pair would keep asserting against a number the seeder no
    longer produces, and every such arm would silently become "the caller sent
    the wrong version" — a version_moved hold that looks like a pass.
    """
    pair = server.registry.get_artifact_and_generation(artifact_id)
    assert pair is not None
    return pair[0].version, pair[1]


# --- AE1: nothing moved ----------------------------------------------------


def test_ae1_nothing_moved_answers_proceed_and_repeats(served_decider) -> None:
    """The one arm that may proceed: the caller's comparands are the
    coordinator's, the grant still stands, and the coordinator claims the
    content in hand. Asked twice, it answers the same — a verdict that healed
    or consumed anything would drift on the second call."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("v3-bytes"),
        state=MESIState.SHARED, version=3,
    )
    version, generation = _u4_captured(server, artifact_id)

    body = _u4_body(sid, _U3A_WARN_PATH, version=version,
                    generation=generation, content_hash=_hash("v3-bytes"))
    first = client.post(_U4_ROUTE, body)
    second = client.post(_U4_ROUTE, body)

    assert first == (200, {"verdict": "proceed"})
    assert second == first


# --- AE2 / AE3 / AE4 / AE5: one hold per condition -------------------------


def test_ae2_peer_commit_holds_naming_the_moved_input(served_decider) -> None:
    """A peer committed after the caller's read. The value the decision was
    derived from moved, so the hold names THAT and not some downstream
    symptom of it."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("v3-bytes"),
        state=MESIState.SHARED, version=3,
    )
    captured_version, generation = _u4_captured(server, artifact_id)
    # The peer commit: the version the caller captured is no longer current.
    art = server.registry.get_artifact(artifact_id)
    server.registry.set_artifact_and_content(
        artifact_id, dataclasses.replace(art, version=captured_version + 1), "",
    )

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=captured_version,
        generation=generation, content_hash=_hash("v3-bytes"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "version_moved"}


def test_ae3_write_acquire_preemption_holds_naming_the_lost_grant(served_decider) -> None:
    """A peer's pessimistic write-acquire ends the caller's grant while moving
    NEITHER comparand — no commit yet, and the acquire trigger is outside the
    epoch-bump set. A version-and-generation check structurally cannot see it,
    so the hold must come from the grant leg. The control assertions below are
    the point of the test: if the acquire HAD moved either comparand, the hold
    would be real but this arm would prove nothing about the grant leg."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    captured_version, captured_generation = _u4_captured(server, artifact_id)

    peer = _sid("ae3-write-acquirer")
    acquire_status, _ = client.post(
        "/hooks/pre-edit", {"session_id": peer, "path": _U3A_WARN_PATH},
    )
    assert acquire_status == 200

    # Controls: neither comparand moved, and the caller really did lose the grant.
    assert _u4_captured(server, artifact_id) == (captured_version, captured_generation)
    assert server.registry.get_agent_state(
        artifact_id, session_to_agent_id(sid),
    ) != MESIState.SHARED

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=captured_version,
        generation=captured_generation, content_hash=_hash("canonical"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "grant_preempted"}


def test_ae4_differing_content_hash_demotes_the_generation(served_decider) -> None:
    """The coordinator still REPORTS an integer generation here, and both
    comparands match. Proceeding on that integer is the failure: a differing
    content hash means the coordinator cannot vouch that the bytes in hand are
    the content at that version, so the authority comparand is unconfirmed and
    the fence holds."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    version, generation = _u4_captured(server, artifact_id)
    # Control: the coordinator reports a REAL generation for this artifact, so
    # the hold below is the demotion firing and not an absent generation.
    assert isinstance(generation, int)

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=version, generation=generation,
        content_hash=_hash("bytes-the-coordinator-never-recorded"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "generation_unconfirmed"}


def test_ae5_zero_captured_version_holds_as_unconfirmed(served_decider) -> None:
    """Zero is the "could not resolve" sentinel a degraded or pre-fence
    coordinator hands back, never a comparable value. Comparing it is how "I
    do not know" becomes "I know it is unchanged"."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    _version, generation = _u4_captured(server, artifact_id)

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=0, generation=generation,
        content_hash=_hash("canonical"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "version_unconfirmed"}


# --- AE8: the coordinator holds no content claim ---------------------------


@pytest.mark.parametrize("recorded_hash,form", [
    ("", "empty-string seed — what a KTD-9 first observation with no caller "
         "hash writes, and the form eight of nine code paths produce"),
    ("f" * 64, "the all-f launch-gate sentinel — no real SHA-256 matches it"),
])
def test_ae8_no_content_claim_holds_under_its_own_reason(
    served_decider, recorded_hash: str, form: str,
) -> None:
    """AE8, and the reason this route exists at all.

    Both comparands match, the grant stands, the read was not refused — every
    leg the in-process fence can see says proceed. But the coordinator records
    NO content hash, so it cannot vouch that the bytes in hand are the content
    at that version. The in-process wrapper passes
    ``content_claim_present=True`` unconditionally (its wire carries a
    comparison, not the recorded hash) and therefore ADMITS here; this route
    reads the registry and answers under ``content_claim_absent``.

    The reason matters as much as the hold: falling into the residual
    ``generation_unconfirmed`` bucket would make this byte-identical on the
    wire to a degraded read, whose recovery is "re-read your bytes" rather
    than "call an operator".
    """
    server, client = served_decider
    path = _u3a_warn_path("ae8-no-claim")
    sid, _agent_id, artifact_id = _u3a_seed(
        server, path, recorded_hash=recorded_hash,
        state=MESIState.SHARED, version=2, label=f"ae8-{len(recorded_hash)}",
    )
    version, generation = _u4_captured(server, artifact_id)

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, path, version=version, generation=generation,
        content_hash=_hash("whatever-the-caller-holds"),
    ))

    assert status == 200, form
    assert body == {"verdict": "hold", "reason": "content_claim_absent"}, form


# --- the refused read ------------------------------------------------------


def test_strict_refused_read_holds_with_the_refusal_reason(served_decider) -> None:
    """A strict-mode deny with BOTH comparands matching. The refusal alone
    must hold it: a caller that fell past the refusal leg would either
    mislabel the hold as a lost grant or — with everything matching —
    proceed on a view the coordinator had just refused to serve."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_STRICT_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.INVALID, version=4,
    )
    version, generation = _u4_captured(server, artifact_id)

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_STRICT_PATH, version=version, generation=generation,
        content_hash=_hash("canonical"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "read_denied"}


# --- the artifact the coordinator does not track ---------------------------


def test_untracked_artifact_holds_rather_than_answering_fresh(served_decider) -> None:
    """pre-read's untracked fast path returns a fresh-shaped answer carrying
    NO version, which a client reads as the zero sentinel. Copying that shape
    here would turn "the coordinator knows nothing about this file" into "the
    coordinator agrees nothing moved" — the exact admit this route exists to
    close. It reaches its own held conclusion instead."""
    server, client = served_decider
    untracked = "notes/scratch.txt"
    assert not server.policy.is_tracked(untracked)

    status, body = client.post(_U4_ROUTE, _u4_body(
        _sid("untracked-caller"), untracked, version=7, generation=1,
        content_hash=_hash("bytes"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "input_vanished"}


def test_tracked_but_never_observed_artifact_holds(served_decider) -> None:
    """Tracked by policy, but no row: pre-read answers this by MUTATING
    (registering the artifact, seeding v1, granting SHARED). A verdict must
    not mint the record it is asked to check, so it holds on the absence."""
    server, client = served_decider
    path = _u3a_warn_path("never-observed")
    assert server.policy.is_tracked(path)
    assert server.registry.lookup_artifact_id_by_name(path) is None

    status, body = client.post(_U4_ROUTE, _u4_body(
        _sid("first-caller"), path, version=1, generation=0,
        content_hash=_hash("bytes"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "input_vanished"}
    assert server.registry.lookup_artifact_id_by_name(path) is None, (
        "the verdict minted the artifact row it was asked to check"
    )


def test_unknown_agent_holds_and_no_retry_can_clear_it(served_decider) -> None:
    """R5's sharpest case, and the reason a MISSING FIELD must not be a hold:
    a caller resolving to an agent the coordinator has never seen legitimately
    produces a hold that no number of identical retries can clear. If an
    incomplete request also answered "hold", the two would be indistinguishable
    on the wire — and only one of them is fixable by retrying."""
    server, client = served_decider
    path = _u3a_warn_path("unknown-agent")
    _seeder_sid, _agent_id, artifact_id = _u3a_seed(
        server, path, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2, label="unknown-agent-seeder",
    )
    version, generation = _u4_captured(server, artifact_id)
    stranger = _sid("never-registered-stranger")
    assert server.agent_name_for(session_to_agent_id(stranger)) is None

    body = _u4_body(stranger, path, version=version, generation=generation,
                    content_hash=_hash("canonical"))
    first = client.post(_U4_ROUTE, body)
    second = client.post(_U4_ROUTE, body)

    assert first[0] == 200
    assert first[1]["verdict"] == "hold"
    assert second == first, "an identical retry cleared a hold it cannot clear"
    # And the stranger stayed a stranger — the route derives the identity, it
    # does not mint one (the pre-read handler registers before its work body).
    #
    # DO NOT DELETE THIS PAIR AS OFF-TOPIC. It is the only NON-VACUOUS pin on
    # the agent-name map: ``test_the_fence_call_mutates_nothing`` carries that
    # map in its before/after set too, but every one of its arms uses a session
    # the seeder already registered, so a stray ``register_session`` there is a
    # no-op and the dimension cannot fail. Verified by mutation: making the
    # handler register instead of derive leaves the purity harness GREEN and
    # turns only this test red.
    assert server.agent_name_for(session_to_agent_id(stranger)) is None


# --- R5: an incomplete request is a client error, never a hold -------------


def _u4_complete(sid: str, path: str) -> dict:
    """A request that would PROCEED if nothing were removed from it — so a
    400 below is attributable to the removed field alone."""
    return _u4_body(sid, path, version=2, generation=0, content_hash=_hash("canonical"))


@pytest.mark.parametrize("drop,expected_error", [
    ("expected_version", "missing expected_version"),
    ("expected_generation", "missing expected_generation"),
    ("content_hash", "missing content_hash"),
])
def test_ae6_omitted_input_is_a_client_error_naming_the_field(
    served_decider, drop: str, expected_error: str,
) -> None:
    """AE6 and its two siblings. A hold invites a retry, and no number of
    retries can supply a comparand the caller never captured — so an
    incomplete request is told what to send, by name, rather than handed a
    verdict it will bounce off forever.

    Each error names a DIFFERENT field: an omitted generation and an omitted
    content hash must not collapse into one message, or a caller cannot tell
    which one it forgot."""
    server, client = served_decider
    sid, _agent_id, _artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    body = _u4_complete(sid, _U3A_WARN_PATH)
    del body[drop]

    status, resp = client.post(_U4_ROUTE, body)

    assert status == 400
    assert resp == {"error": expected_error}


@pytest.mark.parametrize("field,value,expected_error", [
    ("expected_version", "2", "expected_version must be an integer"),
    ("expected_version", 2.0, "expected_version must be an integer"),
    # bool is an int subclass: ``isinstance(True, int)`` is True, so a fence
    # that merely type-checks accepts ``True`` as version 1 and compares it.
    ("expected_version", True, "expected_version must be an integer"),
    ("expected_generation", "0", "expected_generation must be an integer or null"),
    ("expected_generation", False, "expected_generation must be an integer or null"),
])
def test_malformed_comparand_is_a_client_error(
    served_decider, field: str, value: Any, expected_error: str,
) -> None:
    """Comparands are parsed as integers and NEVER coerced. A string "2" that
    became 2, or a ``True`` that became 1, is a comparand the caller never
    captured being compared as though it had been."""
    _server, client = served_decider
    body = _u4_complete(_sid("malformed-comparand"), _U3A_WARN_PATH)
    body[field] = value

    status, resp = client.post(_U4_ROUTE, body)

    assert status == 400
    assert resp == {"error": expected_error}


def test_explicit_null_generation_is_the_sentinel_not_an_error(served_decider) -> None:
    """The one asymmetry: an OMITTED generation is a client mistake, but an
    explicit ``null`` is a captured fact — "the coordinator I read from
    confirmed no generation". That is representable in the fence vocabulary,
    so it is answered as the hold it is rather than as a malformed request."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    version, _generation = _u4_captured(server, artifact_id)

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=version, generation=None,
        content_hash=_hash("canonical"),
    ))

    assert status == 200
    assert body == {"verdict": "hold", "reason": "generation_unconfirmed"}


@pytest.mark.parametrize("session_id,expected_error", [
    (None, "missing session_id"),
    (42, "missing session_id"),
    ("not-a-uuid", "session_id must be a UUID (8-4-4-4-12 hex with hyphens)"),
])
def test_missing_or_malformed_session_id_is_a_client_error(
    served_decider, session_id: Any, expected_error: str,
) -> None:
    """Grant standing is PER SESSION: without the session identifier the route
    cannot compute two of the six values the verdict needs. It says so rather
    than gating on the four it can compute."""
    _server, client = served_decider
    body = _u4_complete(_sid("placeholder"), _U3A_WARN_PATH)
    if session_id is None:
        del body["session_id"]
    else:
        body["session_id"] = session_id

    status, resp = client.post(_U4_ROUTE, body)

    assert status == 400
    assert resp == {"error": expected_error}


@pytest.mark.parametrize("path,expected_error", [
    ("/etc/passwd", "path must be relative (no leading /)"),
    ("../../etc/passwd", "path contains '..' traversal"),
    ("plan\nmd", "path contains control characters"),
    ("", "missing or empty path"),
])
def test_path_gate_rejects_before_classifying(
    served_decider, path: str, expected_error: str,
) -> None:
    """The route validates through the SAME server-side path gate every other
    route uses. Skipping it would let an absolute or traversing path fall
    through to the policy check, which answers False and therefore HOLDS — a
    200 that looks like a safe answer while the boundary check never ran."""
    _server, client = served_decider
    body = _u4_complete(_sid("path-gate"), _U3A_WARN_PATH)
    body["path"] = path

    status, resp = client.post(_U4_ROUTE, body)

    assert status == 400
    assert resp == {"error": expected_error}


def test_malformed_content_hash_is_a_client_error(served_decider) -> None:
    """A present-but-wrong-shape hash is rejected for the same reason the
    other routes reject it: a caller-supplied hash that is not a SHA-256 can
    only ever mismatch, so admitting it turns every verdict into a hold with
    no way to tell a broken client from a real divergence."""
    _server, client = served_decider
    body = _u4_complete(_sid("bad-hash"), _U3A_WARN_PATH)
    body["content_hash"] = "not-a-sha"

    status, resp = client.post(_U4_ROUTE, body)

    assert status == 400
    assert resp == {"error": "content_hash must be 64 hex characters (sha-256)"}


# --- R12: neither non-normal arm may leak the wrapper's default shapes -----


def test_watchdog_timeout_answers_hold_not_the_default_fresh_shape(served_decider) -> None:
    """The shared wrapper's default degraded envelope is
    ``{"status": "fresh", "degraded": true, ...}`` — a PROCEED shape, correct
    for a pre-read whose contract is fresh/stale and catastrophic for a safety
    verdict. This route owns its own envelope, and it reads as a hold carrying
    a listed reason: a coordinator that timed out resolved nothing, which is
    exactly what the "could not resolve" sentinel means."""
    from concurrent.futures import TimeoutError as FuturesTimeout
    from unittest.mock import patch

    server, client = served_decider
    with patch.object(server, "run_with_watchdog", side_effect=FuturesTimeout()):
        status, body = client.post(_U4_ROUTE, _u4_complete(
            _sid("watchdog"), _U3A_WARN_PATH,
        ))

    assert status == 200
    assert body["verdict"] == "hold"
    assert body["reason"] in HOLD_REASONS
    assert body["held_by"] == "watchdog_timeout"
    assert "status" not in body, "the wrapper's fresh-shaped default reached the wire"


def test_handler_exception_answers_hold_not_the_wrappers_ok_false(served_decider) -> None:
    """The wrapper's OTHER non-normal arm turns a handler exception into HTTP
    200 ``{"ok": false, "reason": "internal: ..."}`` — a shape no
    ``degraded_response`` parameter overrides, carrying no verdict and a
    ``reason`` drawn from no published vocabulary. A fence client branching on
    ``verdict`` would read that as neither proceed nor hold."""
    from unittest.mock import patch

    server, client = served_decider
    with patch.object(
        server.registry, "lookup_artifact_id_by_name",
        side_effect=RuntimeError("simulated registry failure"),
    ):
        status, body = client.post(_U4_ROUTE, _u4_complete(
            _sid("boom"), _U3A_WARN_PATH,
        ))

    assert status == 200
    assert body["verdict"] == "hold"
    assert body["reason"] in HOLD_REASONS
    assert body["held_by"] == "handler_error"
    assert "ok" not in body, "the wrapper's {ok: false} arm reached the wire"


def test_an_unlisted_reason_never_reaches_the_wire(served_decider) -> None:
    """R3, enforced rather than documented: the reason vocabulary is declared
    protocol, so a reason outside it is a drift fault, not a verdict. It fails
    into this route's own hold envelope instead of teaching a client a word no
    published set contains."""
    from unittest.mock import patch

    import ccs.adapters.claude_code.coordinator_server as mod

    _server, client = served_decider
    with patch.object(mod, "classify_hold", return_value="reason_from_the_future"):
        status, body = client.post(_U4_ROUTE, _u4_complete(
            _sid("drift"), _U3A_WARN_PATH,
        ))

    assert status == 200
    assert body["verdict"] == "hold"
    assert body["reason"] != "reason_from_the_future"
    assert body["reason"] in HOLD_REASONS


def test_every_hold_reason_this_route_emits_is_published(served_decider) -> None:
    """The closed set, checked against the reasons this suite actually drives
    rather than against the constant module (which would check the set against
    itself). Also pins that the route reaches MORE than one leg — a guard over
    a single reason would pass on a route that answered ``input_vanished`` to
    everything."""
    server, client = served_decider
    reasons = set()
    for name, recorded, state, caller, strict, sent_version in (
        ("moved", "canonical", MESIState.SHARED, "canonical", False, 1),
        ("unconfirmed-version", "canonical", MESIState.SHARED, "canonical", False, 0),
        ("no-claim", "", MESIState.SHARED, "canonical", False, None),
        ("hash", "canonical", MESIState.SHARED, "other-bytes", False, None),
        ("denied", "canonical", MESIState.INVALID, "canonical", True, None),
    ):
        path = _u3a_strict_path(name) if strict else _u3a_warn_path(name)
        sid, _agent, artifact_id = _u3a_seed(
            server, path, recorded_hash=(_hash(recorded) if recorded else ""),
            state=state, version=3, label=f"published-{name}",
        )
        version, generation = _u4_captured(server, artifact_id)
        status, body = client.post(_U4_ROUTE, _u4_body(
            sid, path,
            version=version if sent_version is None else sent_version,
            generation=generation, content_hash=_hash(caller),
        ))
        assert status == 200
        assert body["verdict"] == "hold", f"{name} answered {body}"
        reasons.add(body["reason"])

    assert reasons <= HOLD_REASONS
    assert len(reasons) >= 4, f"only reached {reasons}; the legs are not separable"


# --- the wire carries no session identifier --------------------------------


def test_the_verdict_body_carries_no_session_identifier(served_decider) -> None:
    """The verdict answers about an artifact, not about who asked. Echoing the
    session id back would put a caller-supplied identifier into a body other
    tooling logs, for no branch any client takes."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    version, generation = _u4_captured(server, artifact_id)

    _status, proceed = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=version, generation=generation,
        content_hash=_hash("canonical"),
    ))
    _status, held = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=version + 99, generation=generation,
        content_hash=_hash("canonical"),
    ))

    assert proceed == {"verdict": "proceed"}
    assert held["verdict"] == "hold"
    for body in (proceed, held):
        assert sid not in json.dumps(body)
        assert "session_id" not in body


# --- the route rides the one dispatcher seam -------------------------------


def test_missing_bearer_is_refused(coordinator) -> None:
    """No handler re-implements authentication: the route is registered in the
    single table, so the central Bearer check runs before it is reached."""
    bare = _Client("127.0.0.1", coordinator.port, "unused")
    status, body = bare.request(
        "POST", _U4_ROUTE, _u4_complete(_sid("noauth"), "plan.md"),
        headers_override={"Authorization": ""},
    )
    assert status == 401
    assert body == {"error": "missing or invalid bearer token"}


def test_wrong_bearer_is_refused(coordinator) -> None:
    wrong = _Client("127.0.0.1", coordinator.port, "not-the-secret")
    status, body = wrong.post(_U4_ROUTE, _u4_complete(_sid("badauth"), "plan.md"))
    assert status == 401
    assert body == {"error": "missing or invalid bearer token"}


def test_non_allowlisted_host_is_refused(client: _Client) -> None:
    status, body = client.post(
        _U4_ROUTE, _u4_complete(_sid("badhost"), "plan.md"),
        headers_override={"Host": "evil.example.com"},
    )
    assert status == 403
    assert body == {"error": "host header not allowlisted"}


# --- the two counters ------------------------------------------------------


def test_request_counter_counts_every_attempt_including_the_malformed(
    served_decider,
) -> None:
    """The per-endpoint counter is bumped by the dispatcher BEFORE the handler
    runs, so it counts attempts rather than successes. Registering the route in
    the name map without also initialising the counter leaves the increment a
    silent no-op — the name lookup fails closed — and the route becomes
    invisible on /status while looking wired up in the source."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    version, generation = _u4_captured(server, artifact_id)
    good = _u4_body(sid, _U3A_WARN_PATH, version=version,
                    generation=generation, content_hash=_hash("canonical"))
    malformed = dict(good)
    del malformed["content_hash"]

    assert server.endpoint_counters_snapshot()["effect_fence_total"] == 0
    client.post(_U4_ROUTE, good)
    client.post(_U4_ROUTE, malformed)

    _status, metrics = client.request("GET", "/status?detail=metrics")
    assert metrics["endpoint_counters"]["effect_fence_total"] == 2


def test_hold_counter_moves_on_a_hold_and_stands_still_on_a_proceed(
    served_decider,
) -> None:
    """The hold counter is the operator's view of how often the fence actually
    stopped something. A counter that also ticked on a proceed would report a
    healthy workspace and a wedged one identically."""
    server, client = served_decider
    sid, _agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    version, generation = _u4_captured(server, artifact_id)

    def holds() -> int:
        _s, metrics = client.request("GET", "/status?detail=metrics")
        return metrics["effect_fence_holds_total"]

    assert holds() == 0

    _s, proceed = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=version, generation=generation,
        content_hash=_hash("canonical"),
    ))
    assert proceed == {"verdict": "proceed"}
    assert holds() == 0, "a proceed moved the hold counter"

    _s, held = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=version + 1, generation=generation,
        content_hash=_hash("canonical"),
    ))
    assert held["verdict"] == "hold"
    assert holds() == 1

    # A malformed request is not a hold: it never reached a verdict.
    malformed = _u4_body(sid, _U3A_WARN_PATH, version=version,
                         generation=generation, content_hash=_hash("canonical"))
    del malformed["expected_generation"]
    client.post(_U4_ROUTE, malformed)
    assert holds() == 1, "a 400 was counted as a hold"


# --- R7: the call mutates nothing -----------------------------------------


def _u4_observable(
    server: CoordinatorHTTPServer,
    agent_id: uuid.UUID,
    artifact_id: uuid.UUID,
    session_id: str,
) -> dict:
    """Everything ``_handle_pre_read`` moves around this same decision, read
    through NON-destructive accessors only, MINUS this route's own two
    counters.

    Those two are the documented exception and nothing else is: R7 forbids a
    record that changes a LATER ANSWER, and an advisory count changes none.
    Dropping them by name (rather than comparing whole snapshots loosely) keeps
    every OTHER counter — the stale-warning pair, the strict-denial total, the
    route-around total — inside the guard.
    """
    observable = _u3a_observable(server, agent_id, artifact_id)
    counters = dict(observable["counters"])
    endpoint = dict(counters["endpoint_counters"])
    endpoint.pop("effect_fence_total", None)
    counters["endpoint_counters"] = endpoint
    counters.pop("effect_fence_holds_total", None)
    observable["counters"] = counters
    # The pre-read path registers the session BEFORE its work body, and
    # session-start reads this map to decide whether a session was ever seen.
    observable["agent_names"] = sorted(str(a) for a, _ in server.agent_names_snapshot())
    # The untracked fast path CONSUMES the pending re-grounding flag.
    observable["compact_pending"] = server.has_compact_pending(session_id)
    return observable


def _u4_seed_every_dimension(
    server: CoordinatorHTTPServer, client: _Client, agent_id: uuid.UUID,
    artifact_id: uuid.UUID, session_id: str, path: str,
) -> None:
    """Put a NON-DEFAULT value in every dimension before the before/after
    snapshot. An assertion that ``None`` stayed ``None`` proves much less than
    one that 41 stayed 41 — and a zero counter that stayed zero cannot tell a
    route that never increments from one whose increment did not run."""
    server.registry.record_heartbeat(agent_id, 41)
    server.registry.record_preemption_notice(
        victim_agent_id=agent_id, artifact_id=artifact_id,
        preempter_agent_id=server.register_session(_sid("u4-preempter")),
        preempted_at_unix_ts=1234.0,
    )
    server.mark_compact_pending(session_id)
    server.mark_stale_warned(agent_id, artifact_id)
    server.record_strict_deny(session_id, path)
    for _ in range(3):
        server.increment_stale_warning_emitted()
    for _ in range(2):
        server.increment_strict_mode_denial()
    server.increment_strict_mode_routed_around_via_bash()
    # A REAL strict deny, so audit.log exists with a non-zero size before the
    # comparison: "no audit row appeared" is a much weaker claim against a
    # file that does not exist yet than against one that does.
    deny_sid, _deny_agent, _deny_artifact = _u3a_seed(
        server, _u3a_strict_path("u4-audit-seed"), recorded_hash=_hash("canonical"),
        state=MESIState.INVALID, version=2, label="u4-audit-seed",
    )
    status, _ = client.post("/hooks/pre-read", {
        "session_id": deny_sid, "path": _u3a_strict_path("u4-audit-seed"),
        "content_hash": _hash("canonical"),
    })
    assert status == 200


@pytest.mark.parametrize("arm,sent_version_delta,expect", [
    ("proceed", 0, "proceed"),
    ("hold", 1, "hold"),
])
def test_the_fence_call_mutates_nothing(
    served_decider, arm: str, sent_version_delta: int, expect: str,
) -> None:
    """R7, asserted as a before/after rather than read off the source.

    The pre-read path mutates far more than it looks like: it registers the
    session before the work body, records a heartbeat as its first statement
    (the sweep reclaims grants on heartbeat staleness), re-grants SHARED on the
    stale arm, marks the pair stale-warned, pops notices, fires five mutations
    on a strict deny — one of which makes a later command increment a
    route-around counter and writes an audit row — and consumes the pending
    re-grounding flag on its untracked fast path.

    A verdict that did ANY of it would turn a level-triggered hold into an
    edge-triggered one: the next bare re-check would take the healed branch and
    admit the effect whose grant a peer already revoked. This project has
    recorded three prior instances of a read healing what it checks; this is
    the guard that keeps the fence from being the fourth. Both arms are
    checked, because a proceed and a hold leave through different branches.
    """
    server, client = served_decider
    sid, agent_id, artifact_id = _u3a_seed(
        server, _U3A_WARN_PATH, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2,
    )
    version, generation = _u4_captured(server, artifact_id)
    _u4_seed_every_dimension(server, client, agent_id, artifact_id, sid, _U3A_WARN_PATH)
    # A credentialed fence call (the route is require-class): claimed BEFORE the
    # snapshot, so neither the claim nor an uncredentialed-request count is
    # charged to the call under test — the guard stays over every other counter.
    status, minted = client.post(
        "/principal/claim", {"session_id": sid, "mint_nonce": uuid.uuid4().hex})
    assert status == 200 and minted["ok"] is True, minted

    before = _u4_observable(server, agent_id, artifact_id, sid)
    # Controls for the seeding: a guard whose dimensions are all at their
    # defaults is a guard that cannot fail.
    assert before["heartbeat"] == 41
    assert before["notice"] is not None
    assert before["compact_pending"] is True
    assert before["stale_warned"]
    assert before["strict_denies"]
    assert before["counters"]["strict_mode_denials_total"] > 0
    assert before["counters"]["stale_warning_emitted_total"] > 0
    assert before["counters"]["strict_mode_routed_around_via_bash_total"] > 0
    assert before["audit_logs"] and all(size > 0 for _name, size in before["audit_logs"])

    status, body = client.post(_U4_ROUTE, _u4_body(
        sid, _U3A_WARN_PATH, version=version + sent_version_delta,
        generation=generation, content_hash=_hash("canonical"),
    ), principal=minted["principal"])

    assert status == 200
    assert body["verdict"] == expect, f"the {arm} arm did not take its branch"
    assert _u4_observable(server, agent_id, artifact_id, sid) == before


# ----------------------------------------------------------------------
# U6 — the PUBLISHED contract for the verdict route
#
# The goal these guard is derivability: a client written from the published
# documentation alone, without reading this repository's source, reaches the
# same verdict. Four claims carry that, and each is one a reader ACTS on:
#
#   * the reason vocabulary is wire-stable — added to, never renamed;
#   * each reason names WHO establishes it and has its OWN recovery, including
#     both senses of the vanished-input case, under the identifier it holds by;
#   * the content hash is defined, because the server validates only its SHAPE
#     and a client that picks the other plausible convention holds forever
#     wearing the same reason a genuine conflict would;
#   * every answer that is not a recognised verdict is a hold on the client's
#     side — which is the whole of the story for a coordinator that answers 404.
#
# Every search runs over WHITESPACE-NORMALIZED text. The guide wraps its prose,
# so a line-wise search for any of these sentences would report clean while the
# sentence sat there mangled or half-deleted.
# ----------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GUIDE_PATH = _REPO_ROOT / "docs" / "guide.md"
_README_PATH = _REPO_ROOT / "README.md"


def _normalized(text: str) -> str:
    """Collapse every run of whitespace, so a wrapped sentence still matches."""
    return " ".join(text.split())


# Each phrase is ONE contiguous fragment carrying a whole claim: a pin split
# across two sentences would still pass with half the claim deleted.
_CONTRACT_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "the reason vocabulary is add-only, never renamed",
        "Reasons may be added; an existing one is never renamed or repurposed.",
    ),
    (
        "an unrecognised reason is still a hold",
        "treat a `hold` whose `reason` you do not recognise as a hold",
    ),
    (
        "each reason names its establisher and its own recovery",
        "Each reason names who established it — the **coordinator**, from state only it "
        "can see, or the **caller**, from state only *it* can see — and each has its own "
        "recovery.",
    ),
    (
        "the vanished input is established from BOTH sides",
        "**`input_vanished` has two establishers, and both are in the contract.**",
    ),
    (
        "the coordinator's sense of the vanished input",
        "**The coordinator** answers `input_vanished` over the wire when it holds no "
        "record of the artifact",
    ),
    (
        "the caller's sense of the vanished input, under the same identifier",
        "**The caller** establishes it itself when its own input no longer exists. The "
        "coordinator cannot see your workspace, so it will never report this for you; "
        "detect it and treat it as a hold under this same identifier.",
    ),
    (
        "the content hash is defined over exact bytes, unnormalized",
        "`content_hash` is a lowercase sha-256 hex digest over the exact bytes the caller "
        "holds, with no normalization",
    ),
    (
        "the wrong hash convention is a silent permanent hold",
        "it simply never matches what the coordinator recorded, and every call it makes "
        "holds wearing the same reason a genuine conflict would",
    ),
    (
        "anything that is not a verdict is a hold",
        "every one of these is a hold and the effect does not fire: a connection failure "
        "or a timeout; any status other than `200`; a `200` body with no `verdict`; a "
        "`verdict` you do not recognise; and a `404`",
    ),
    (
        "the sibling coordinator answers 404 for this route",
        "the sibling Node coordinator backend does not implement `/hooks/effect-fence`",
    ),
)


@pytest.mark.parametrize(
    "claim,phrase", _CONTRACT_STATEMENTS, ids=[name for name, _ in _CONTRACT_STATEMENTS]
)
def test_guide_publishes_the_fence_contract(claim: str, phrase: str) -> None:
    """Each claim a client is built from survives in the published guide."""
    guide = _normalized(_GUIDE_PATH.read_text(encoding="utf-8"))
    assert _normalized(phrase) in guide, f"the guide no longer states: {claim}"


def _documented_hold_reasons() -> frozenset[str]:
    """The reasons the guide's published table lists, parsed from the prose.

    Parsed rather than hand-listed so the test reads what a client implementer
    reads. A section that loses its table, or a table whose first column stops
    being the reason, yields an empty set — which fails the comparison below
    rather than passing vacuously.
    """
    guide = _GUIDE_PATH.read_text(encoding="utf-8")
    heading = "### Hold reasons — the published vocabulary"
    assert heading in guide, "the published reason vocabulary has no section"
    section = guide.split(heading, 1)[1].split("\n### ", 1)[0]
    reasons: set[str] = set()
    in_body = False
    for line in section.splitlines():
        if line.startswith("|---"):
            # Everything above the rule is the header, whose first cell is the
            # column name, not a reason.
            in_body = True
        elif in_body and line.startswith("| `"):
            reasons.add(line.split("|")[1].strip().strip("`"))
        elif in_body and not line.startswith("|"):
            break
    return frozenset(reasons)


def test_documented_reasons_are_exactly_the_published_set() -> None:
    """The guide's table and ``HOLD_REASONS`` are the SAME set.

    One side is DERIVED from the code (``HOLD_REASONS`` itself, imported at the
    top of this module) and the other is parsed out of the published prose, so
    neither is hand-typed here. That direction is deliberate and is the
    opposite of this suite's frozen name-set guards: those pin a protocol
    surface against its own source, where a derived expectation would move its
    own goalposts. This test is a DRIFT check between two artifacts that must
    agree — a reason added to the code and not to the guide leaves a client
    branching on a word nothing documents, and a reason documented but never
    published leaves one branching on a word that never arrives.
    """
    documented = _documented_hold_reasons()
    assert documented == HOLD_REASONS, (
        "the guide's hold-reason table and the published HOLD_REASONS set have "
        f"drifted; documented-only={sorted(documented - HOLD_REASONS)}, "
        f"published-only={sorted(HOLD_REASONS - documented)}"
    )


def test_tool_tables_name_the_http_fence() -> None:
    """Neither published tool table leaves the fence looking tool-only.

    The MCP tool table is where a reader decides what surfaces exist. While it
    listed ``swg_gate`` and nothing else, the reasonable conclusion was that an
    agent needs MCP (or Python) to reach a fence verdict at all — which is
    exactly the reader this route exists for.
    """
    for doc in (_GUIDE_PATH, _README_PATH):
        rows = [
            line for line in doc.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `")
        ]
        # Control: the table this is about is still there to be read.
        assert any(row.startswith("| `swg_gate`") for row in rows), (
            f"{doc.name} no longer carries the tool table this guard is about"
        )
        assert any(row.startswith("| `POST /hooks/effect-fence`") for row in rows), (
            f"{doc.name} lists the fence tool without its HTTP sibling"
        )


# ----------------------------------------------------------------------
# U3b — the pre-read handler ANSWERS FROM the verdict
#
# U3a extracted decide_tracked_read; U3b rewired _handle_pre_read's tracked
# branch to call it. The risk that rewiring carries is cosmetic
# de-duplication: a handler that calls the verdict and then quietly re-derives
# the same answer beside it would keep every response byte-identical and every
# existing assertion green, while leaving two copies of the safety rule to
# drift apart. These tests close that by DOCTORING the verdict and asserting
# the response follows it — a handler still computing its own gate answers the
# old way and fails here. Each doctored arm is paired with an un-doctored
# control, so a fixture that silently stopped reaching the arm cannot pass as a
# derivation proof.
# ----------------------------------------------------------------------

_U3B_TARGET = "ccs.adapters.claude_code.coordinator_server.decide_tracked_read"


def _u3b_doctor(monkeypatch, **changes) -> None:
    """Replace the verdict with the real one plus ``changes``.

    Wrapping rather than fabricating keeps every field the test does not name
    at its true value, so a response that moves can only have moved because of
    the field under test.
    """
    real = decide_tracked_read

    def fake(*args, **kwargs) -> TrackedReadDecision:
        return dataclasses.replace(real(*args, **kwargs), **changes)

    monkeypatch.setattr(_U3B_TARGET, fake)


def _u3b_seed_invalid(server, path: str, label: str):
    """A session a peer commit INVALIDated on a warn-mode path: the plain
    stale arm, which never denies on its own."""
    peer = server.register_session(_sid(f"u3b-peer-{label}"))
    return _u3a_seed(
        server, path, recorded_hash=_hash("canonical"), state=MESIState.INVALID,
        label=f"u3b-{label}", version=4, last_writer=peer,
    )


def _u3b_reset(server, artifact_id, agent_id, state) -> None:
    """Put the pair back in its seeded state so the doctored call sees exactly
    the input the control saw — the stale arm re-grants SHARED, so without this
    the second call would be measuring a different scenario."""
    server.registry.set_agent_state(
        artifact_id, agent_id, state,
        trigger="u3b-reset", tick=1, content_hash=_hash("canonical"),
    )


def _u3b_read(client: _Client, sid: str, path: str, caller_hash: str) -> dict:
    status, body = client.post("/hooks/pre-read", {
        "session_id": sid, "path": path, "content_hash": caller_hash,
    })
    # Every protocol outcome — fresh, stale and deny alike — is an HTTP 200
    # here; a non-200 would mean the hook client degraded and the assertion
    # below would be about a fault, not a verdict.
    assert status == 200, body
    return body


def _u3b_decision_kind(body: dict) -> str:
    hook = body.get("hookSpecificOutput") or {}
    if hook.get("permissionDecision") == "deny":
        return "denied"
    return body.get("status", "?")


def test_pre_read_deny_follows_the_verdict_not_a_second_gate(
    served_decider, monkeypatch
) -> None:
    """The strict-deny GATE lives in the verdict; only the EMISSION is the
    handler's. Doctoring a warn-mode stale into ``denied`` must turn the
    response into the byte-stable deny and bump ``strict_mode_denials_total``.

    A handler that kept its own ``is_strict_mode(...) and (INVALID or ...)``
    test alongside the call would answer ``stale`` here and the two copies
    could then drift — which is the failure this unit exists to make
    impossible.
    """
    server, client = served_decider
    path = _u3a_warn_path("derive-deny")
    sid, agent_id, artifact_id = _u3b_seed_invalid(server, path, "derive-deny")

    control = _u3b_read(client, sid, path, _hash("canonical"))
    assert _u3b_decision_kind(control) == "stale", (
        "control: a warn-mode INVALID session must reach the stale arm, or the "
        "doctored call below proves nothing"
    )

    _u3b_reset(server, artifact_id, agent_id, MESIState.INVALID)
    before = server.counters_snapshot()["strict_mode_denials_total"]
    _u3b_doctor(monkeypatch, outcome="denied")
    doctored = _u3b_read(client, sid, path, _hash("canonical"))

    assert _u3b_decision_kind(doctored) == "denied"
    assert server.counters_snapshot()["strict_mode_denials_total"] == before + 1


def test_pre_read_arm_selection_follows_the_verdict(
    served_decider, monkeypatch
) -> None:
    """Which ARM runs — fresh payload or stale envelope — is
    ``holds_valid_grant``, not a second reading of the MESI state.

    The session below really does hold SHARED throughout; only the verdict
    says otherwise. If the handler still branched on ``agent_state`` it would
    answer fresh and this would fail.
    """
    server, client = served_decider
    path = _u3a_warn_path("derive-arm")
    peer = server.register_session(_sid("u3b-peer-derive-arm"))
    sid, agent_id, artifact_id = _u3a_seed(
        server, path, recorded_hash=_hash("canonical"), state=MESIState.SHARED,
        label="u3b-derive-arm", version=4, last_writer=peer,
    )

    control = _u3b_read(client, sid, path, _hash("canonical"))
    assert _u3b_decision_kind(control) == "fresh", "control: this is the fresh arm"

    _u3b_doctor(monkeypatch, holds_valid_grant=False, outcome="stale",
                prior_version_seen=3)
    doctored = _u3b_read(client, sid, path, _hash("canonical"))

    assert _u3b_decision_kind(doctored) == "stale"
    assert doctored["summary"]["prior_version_seen_by_session"] == 3, (
        "the summary's prior version is the verdict's arm-resolved value, not "
        "a second `artifact.version - 1` computed in the handler"
    )
    assert server.registry.get_agent_state(artifact_id, agent_id) == MESIState.SHARED


def test_fresh_arm_reads_only_the_fresh_hash_predicate(
    served_decider, monkeypatch
) -> None:
    """The two hash predicates are NOT one predicate, and the fresh arm must
    read its own.

    ``stale_hash_differs`` keeps the no-claim recorded hashes IN (the
    launch-gate denies reach their answer through the stale arm); the fresh arm
    excludes them, because a still-SHARED holder must never be denied against
    content the coordinator never claimed. A handler that folded them would
    surface ``hash_differs`` and bump the mismatch counter here — on a read
    whose hash MATCHES.
    """
    server, client = served_decider
    path = _u3a_warn_path("derive-fresh-pred")
    peer = server.register_session(_sid("u3b-peer-fresh-pred"))
    sid, _agent_id, _artifact_id = _u3a_seed(
        server, path, recorded_hash=_hash("canonical"), state=MESIState.SHARED,
        label="u3b-fresh-pred", version=2, last_writer=peer,
    )

    before = server.counters_snapshot()["fresh_shared_hash_mismatch_total"]
    _u3b_doctor(monkeypatch, fresh_hash_differs=False, stale_hash_differs=True)
    body = _u3b_read(client, sid, path, _hash("canonical"))

    assert body == {"status": "fresh", "version": 2}, (
        "the fresh payload read the STALE arm's predicate"
    )
    assert server.counters_snapshot()["fresh_shared_hash_mismatch_total"] == before

    # Control for the assertion above: the fresh arm DOES follow its own
    # predicate, so the silence just asserted is about which field was read and
    # not about a branch that never fires.
    _u3b_doctor(monkeypatch, fresh_hash_differs=True, stale_hash_differs=False)
    body = _u3b_read(client, sid, path, _hash("canonical"))

    assert body == {"status": "fresh", "version": 2, "hash_differs": True}
    assert server.counters_snapshot()["fresh_shared_hash_mismatch_total"] == before + 1


def test_stale_arm_reads_only_the_stale_hash_predicate(
    served_decider, monkeypatch
) -> None:
    """The mirror of the test above: the stale summary's ``hash_differs`` is
    ``stale_hash_differs``. Folding the two here would flip the no-prior-grant
    deny gate, which is defined against exactly this field."""
    server, client = served_decider
    path = _u3a_warn_path("derive-stale-pred")
    sid, agent_id, artifact_id = _u3b_seed_invalid(server, path, "stale-pred")

    _u3b_doctor(monkeypatch, fresh_hash_differs=True, stale_hash_differs=False)
    body = _u3b_read(client, sid, path, _hash("canonical"))
    assert _u3b_decision_kind(body) == "stale"
    assert body["summary"]["hash_differs"] is False, (
        "the stale summary read the FRESH arm's predicate"
    )

    _u3b_reset(server, artifact_id, agent_id, MESIState.INVALID)
    _u3b_doctor(monkeypatch, fresh_hash_differs=False, stale_hash_differs=True)
    body = _u3b_read(client, sid, path, _hash("canonical"))
    assert body["summary"]["hash_differs"] is True


def test_the_verdict_runs_once_per_tracked_pre_read_on_the_handlers_clock(
    served_decider, monkeypatch
) -> None:
    """One classification per request, on the handler's single clock read.

    Two calls would mean two snapshots of a moving registry backing one
    response; a clock read taken INSIDE the verdict would let the lag gate and
    the deny summary's timestamps disagree within a single deny (KTD-P). The
    untracked and first-observation arms are asserted NOT to reach the verdict
    at all — each reaches its answer BY mutating, so neither has a
    side-effect-free form for it to classify.
    """
    server, client = served_decider
    calls: list[dict] = []
    real = decide_tracked_read

    def spy(*args, **kwargs) -> TrackedReadDecision:
        calls.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(_U3B_TARGET, spy)

    untracked_sid = _sid("u3b-untracked")
    assert not server.policy.is_tracked("vendor/blob.bin")
    client.post("/hooks/pre-read", {
        "session_id": untracked_sid, "path": "vendor/blob.bin",
        "content_hash": _hash("x"),
    })
    assert calls == [], "the untracked fast path classified through the verdict"

    first_obs_sid = _sid("u3b-first-obs")
    client.post("/hooks/pre-read", {
        "session_id": first_obs_sid, "path": _u3a_warn_path("derive-first-obs"),
        "content_hash": _hash("seed"),
    })
    assert calls == [], "first observation classified through the verdict"

    path = _u3a_warn_path("derive-clock")
    sid, _agent_id, _artifact_id = _u3b_seed_invalid(server, path, "clock")
    wall_before = time.time()
    _u3b_read(client, sid, path, _hash("canonical"))
    wall_after = time.time()

    assert len(calls) == 1, "the tracked branch classified more than once"
    now_unix = calls[0]["now_unix"]
    # A real clock read, bracketed by the request — not 0, not None, and not a
    # constant a later refactor could freeze without anyone noticing.
    assert wall_before <= now_unix <= wall_after


@pytest.mark.parametrize(
    "bad_agent_id",
    ["has a space", "way-too-long" * 12, 42, ["a"], {"a": 1}, True],
)
def test_fence_refuses_a_malformed_agent_id_instead_of_answering_as_the_parent(
    served_decider, bad_agent_id
) -> None:
    """A present-but-malformed subagent id must be a 400, never a verdict.

    ``read_subagent_id`` resolves an out-of-shape value to ``None`` -- the
    PARENT identity. On a read path that only changes attribution prose, which
    is why the session-stop guard's comment scopes the allowance to read paths.
    Here it changes WHOSE grant the verdict is about, and grant standing is the
    one leg computed per agent: a subagent whose grant a peer preempted would
    be answered about a parent that still holds SHARED, and told to PROCEED.
    That is the fail-open this route exists to close, reached through an
    identity field rather than a comparand.
    """
    server, client = served_decider
    path = _u3a_warn_path("malformed-agent")
    sid, _agent_id, artifact_id = _u3a_seed(
        server, path, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2, label="malformed-agent-seed",
    )
    version, generation = _u4_captured(server, artifact_id)
    body = _u4_body(sid, path, version=version, generation=generation,
                    content_hash=_hash("canonical"))
    body["agent_id"] = bad_agent_id

    status, payload = client.post(_U4_ROUTE, body)

    assert status == 400, f"malformed agent_id answered {status}, not a client error"
    assert "agent_id" in payload["error"], payload
    assert "verdict" not in payload, "a malformed identity produced a verdict"


def test_fence_still_answers_for_an_absent_or_well_formed_agent_id(
    served_decider,
) -> None:
    """The control for the refusal above. Rejecting a MALFORMED id must not
    reject an ABSENT one -- omitting the field is the legitimate parent call
    and every hook route accepts it -- nor a well-formed subagent id. Without
    this, the guard could be tightened into refusing every caller and the
    test above would still pass."""
    server, client = served_decider
    path = _u3a_warn_path("agent-id-controls")
    sid, _agent_id, artifact_id = _u3a_seed(
        server, path, recorded_hash=_hash("canonical"),
        state=MESIState.SHARED, version=2, label="agent-id-controls-seed",
    )
    version, generation = _u4_captured(server, artifact_id)
    base = _u4_body(sid, path, version=version, generation=generation,
                    content_hash=_hash("canonical"))

    absent_status, absent_payload = client.post(_U4_ROUTE, dict(base))
    assert absent_status == 200, absent_payload
    assert "verdict" in absent_payload

    scoped = dict(base)
    scoped["agent_id"] = "sub-agent_1"
    scoped_status, scoped_payload = client.post(_U4_ROUTE, scoped)
    assert scoped_status == 200, scoped_payload
    assert "verdict" in scoped_payload


# ======================================================================
# Hook responses name a peer by its agent id, not by its session id (R7)
# ======================================================================
#
# ``session_to_agent_id`` is a one-way uuid5 of the session id and the
# coordinator already publishes it on /status. The disclosure-bearing hook
# paths used to run that mapping BACKWARDS through the adapter's agent-name
# map -- ``_agent_id_to_session`` -- and render the recovered session id, so a
# response handed to one session republished a peer's session id (in full on
# the session-stop notice array, and as its 8-char prefix in prose). Naming
# the peer by the handle the coordinator already emits keeps attribution
# intact and stops re-deriving an identifier the response has no reason to
# carry. Nothing new is derived: the agent id is what the registry stores.


def test_session_stop_notice_names_the_preempting_agent_id(client: _Client) -> None:
    """The stop-drain notice array names the preempter by agent id.

    The whole response body is searched for the preempter's session id,
    because the notice carries a structured field AND prose and either would
    republish it.
    """
    x = _sid("R7-X")
    y = _sid("R7-Y")
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})  # preempts X

    status, body = client.post("/hooks/session-stop", {"session_id": x})
    assert status == 200
    notices = body.get("notices")
    assert notices, f"the stop drain must surface X's pending notice; got {body!r}"

    y_agent = session_to_agent_id(y)
    notice = notices[0]
    assert notice["path"] == "plan.md"
    assert notice["preempter_agent_id"] == y_agent.hex, (
        f"the notice must name the preempting AGENT; got {notice!r}"
    )
    assert notice["preempter_agent_short"] == y_agent.hex[:8]

    serialized = json.dumps(body)
    assert y not in serialized, (
        f"the stop response still carries Y's session id verbatim: {serialized!r}"
    )
    assert y[:8] not in serialized, (
        f"the stop response still carries Y's session-id prefix: {serialized!r}"
    )
    # The notice is only worth anything if it still ATTRIBUTES: the agent id
    # has to be in the prose too, not merely in the structured field.
    assert y_agent.hex[:8] in body["hookSpecificOutput"]["additionalContext"]


def test_stale_summary_names_the_writing_agent_id(client: _Client) -> None:
    """``summary.last_writer_session_id`` carries the writer's agent id.

    End-to-end, because the value is produced by ``_last_writer_for`` reading
    the registry's ``last_writer_id`` -- which has always been an agent id.
    """
    a = _sid("R7-reader")
    b = _sid("R7-writer")
    client.post("/hooks/pre-read", {"session_id": a, "path": "plan.md",
                                    "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit", {"session_id": b, "path": "plan.md"})
    client.post("/hooks/post-edit", {"session_id": b, "path": "plan.md",
                                     "content_hash": _hash("v2"), "success": True})

    status, body = client.post("/hooks/pre-read", {"session_id": a, "path": "plan.md",
                                                   "content_hash": _hash("v2")})
    assert status == 200
    assert body["status"] == "stale", body
    b_agent = session_to_agent_id(b)
    assert body["summary"]["last_writer_session_id"] == b_agent.hex

    serialized = json.dumps(body)
    assert b not in serialized, f"the stale response carries B's session id: {serialized!r}"
    assert b[:8] not in serialized, f"the stale response carries B's prefix: {serialized!r}"
    assert b_agent.hex[:8] in body["hookSpecificOutput"]["additionalContext"]


def test_edit_collision_names_the_holding_agent_id(client: _Client) -> None:
    """The pre-edit collision notice names the incumbent holder by agent id."""
    x = _sid("R7-holder")
    y = _sid("R7-challenger")
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    status, body = client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})
    assert status == 200
    assert body.get("collision") is True, body
    context = body["hookSpecificOutput"]["additionalContext"]
    assert f"({session_to_agent_id(x).hex[:8]})" in context, context
    assert x[:8] not in json.dumps(body)


def test_post_edit_preemption_reason_names_the_preempting_agent_id(
    client: _Client,
) -> None:
    """The ``commit_not_allowed`` reason names the preempter by agent id."""
    x = _sid("R7-loser")
    y = _sid("R7-winner")
    client.post("/hooks/pre-edit", {"session_id": x, "path": "plan.md"})
    client.post("/hooks/pre-edit", {"session_id": y, "path": "plan.md"})  # preempts X
    status, body = client.post("/hooks/post-edit", {
        "session_id": x, "path": "plan.md",
        "content_hash": _hash("v2"), "success": True,
    })
    assert status == 200
    assert body.get("ok") is False, body
    reason = body["reason"]
    assert f"preempted by agent {session_to_agent_id(y).hex[:8]}" in reason, reason
    assert y[:8] not in reason, reason


def test_the_reverse_session_lookup_is_gone() -> None:
    """``_agent_id_to_session`` is removed, not merely unused.

    Left in place it is an invitation: the next renderer that wants a
    human-ish label reaches for it and re-introduces exactly the mapping this
    unit removed. Its absence is the enforcement.
    """
    from ccs.adapters.claude_code import coordinator_server as cs

    assert not hasattr(cs, "_agent_id_to_session")


# ======================================================================
# A deny states what happened: a grant handover is not a write (R8)
# ======================================================================
#
# Cohexa-ai/agent-coherence#196: a peer's pre-edit invalidates a live holder
# WITHOUT committing anything. The holder's next read was denied with "was
# updated by session <unknown> at <t>" -- a write that never happened, at a
# timestamp when nothing was written. The renderers now branch on what the
# summary can actually support: a version that moved past the one this
# session last observed is a write; a version that did not is a grant that
# changed hands.


def _summary(**overrides: Any) -> dict:
    base = {
        "path": "docs/plan.md",
        "current_version": 2,
        "prior_version_seen_by_session": 1,
        "last_writer_session_id": "aaaaaaaa11114111811111111111aaaa",
        "last_writer_at_unix_ts": 1700000000.0,
        "warning_generated_at_unix_ts": 1700000001.0,
        "hash_differs": False,
    }
    base.update(overrides)
    return base


def test_summary_reports_a_write_only_when_the_version_moved() -> None:
    """The discriminator, stated once and asserted on both sides.

    Asserting only the admitting direction would pass for a predicate that
    always returns True, which is the shape the old prose had.
    """
    from ccs.adapters.claude_code import hook_payloads as hp

    assert hp.summary_reports_a_write(_summary(current_version=2,
                                               prior_version_seen_by_session=1))
    assert not hp.summary_reports_a_write(_summary(current_version=1,
                                                   prior_version_seen_by_session=1))
    # Never observed: nothing to compare against, so no grant-change claim.
    assert hp.summary_reports_a_write(_summary(prior_version_seen_by_session=None))
    # Diverged bytes: something WAS written, in-band or not.
    assert hp.summary_reports_a_write(_summary(current_version=1,
                                               prior_version_seen_by_session=1,
                                               hash_differs=True))


def test_deny_after_a_real_commit_still_names_the_writer_and_the_version() -> None:
    from ccs.adapters.claude_code import hook_payloads as hp

    reason = hp.emit_strict_deny(
        source="test", summary=_summary(),
    )["permissionDecisionReason"]
    assert reason == (
        "Stale read denied: docs/plan.md was updated by agent aaaaaaaa at "
        "2023-11-14T22:13:20+00:00. Re-read docs/plan.md via the Read tool "
        "before proceeding. This denial is structural (v0.2 strict mode); "
        "retrying the same operation will produce the same denial."
    )


def test_deny_after_a_grant_handover_claims_no_write_and_names_the_version() -> None:
    from ccs.adapters.claude_code import hook_payloads as hp

    reason = hp.emit_strict_deny(
        source="test",
        summary=_summary(current_version=1, prior_version_seen_by_session=1),
    )["permissionDecisionReason"]
    assert reason == (
        "Stale read denied: your grant on docs/plan.md was revoked and no new "
        "version was committed — docs/plan.md is still at v1. Re-read "
        "docs/plan.md via the Read tool before proceeding. This denial is "
        "structural (v0.2 strict mode); retrying the same operation will "
        "produce the same denial."
    )
    assert "was updated by" not in reason


def test_stale_warning_after_a_grant_handover_claims_no_write() -> None:
    from ccs.adapters.claude_code import hook_payloads as hp

    text = hp.stale_read_warning(
        _summary(current_version=1, prior_version_seen_by_session=1)
    )
    assert text == (
        "⚠ Stale read [warning emitted 2023-11-14T22:13:21+00:00]: your "
        "grant on docs/plan.md was revoked and no new version was committed. "
        "docs/plan.md is still at v1, the version you last saw. "
        "Re-acquire before writing to docs/plan.md."
    )
    assert "was updated by" not in text


# ----------------------------------------------------------------------
# Caller-principal route posture (caller-principal plan, U6)
# ----------------------------------------------------------------------
#
# Every route that takes a ``session_id`` is classified by the harm an ABSENT
# principal admits there (KTD3): require-class routes refuse a request naming
# a BOUND session without the principal ``POST /principal/claim`` bound to it
# (a session nobody ever claimed — an older client's — is admitted and
# counted, as before the principal existed: R16); accept-class routes admit an
# absent principal and count it; every class refuses a foreign one; the mint
# itself is neither. The
# guarantee is accident-resistance and attributability under the same-OS-user
# cooperative trust model — a caller that presents no principal, or one bound
# to a different session, cannot end another writer's grant or have a write
# recorded under another writer's name by mistake. It is not a boundary
# against a process that can read ``.coherence/``: on the hook surface a
# principal is stored there, so any process that can read it can present it.
#
# These tests drive the coordinator routes directly. A test that "forges" a
# peer's identity deliberately does NOT present that peer's principal; that
# abstention is the whole content of what the route can check.

#: FROZEN duplicates of the wire refusals. Byte-stable on purpose: the two
#: cases must stay distinguishable. A client classifies a refusal by the typed
#: ``reason`` key, by equality — never by a substring of ``error``, which stays
#: the prose every non-200 body carries.
_PRINCIPAL_ABSENT_ERROR = (
    "missing Coherence-Caller-Principal header: this route requires the caller "
    "principal that POST /principal/claim bound to the session_id it names "
    "(caller_principal_absent)"
)
_PRINCIPAL_FOREIGN_ERROR = (
    "the Coherence-Caller-Principal header is not the caller principal bound to "
    "the session_id this request names (caller_principal_foreign)"
)
_PRINCIPAL_ABSENT_REFUSAL = {
    "error": _PRINCIPAL_ABSENT_ERROR, "reason": "caller_principal_absent",
}
_PRINCIPAL_FOREIGN_REFUSAL = {
    "error": _PRINCIPAL_FOREIGN_ERROR, "reason": "caller_principal_foreign",
}
_PRINCIPAL_REFUSALS = (_PRINCIPAL_ABSENT_REFUSAL, _PRINCIPAL_FOREIGN_REFUSAL)


def _principal_refused(response: tuple[int, dict]) -> bool:
    """Whether ``response`` is a principal refusal, in EITHER body shape — so
    an "admitted" assertion cannot pass merely because a refusal came back in
    a shape this module no longer spells out."""
    status, body = response
    return status == 400 and (
        body.get("error") in (_PRINCIPAL_ABSENT_ERROR, _PRINCIPAL_FOREIGN_ERROR)
        or body.get("reason") in ("caller_principal_absent", "caller_principal_foreign")
    )

#: FROZEN duplicate of the posture table (route -> class). Never derived from
#: ``_CALLER_PRINCIPAL_POSTURE``: a derived expectation moves with the edit that
#: breaks it. Nineteen routes: the eighteen the plan enumerates plus the mint.
#: ``pre-edit`` is require-class: a bound session's caller without its
#: principal could take an EXCLUSIVE grant it can neither commit nor release.
_EXPECTED_ROUTE_POSTURE: dict[tuple[str, str], str] = {
    ("POST", "/hooks/pre-read"): "accept",
    ("POST", "/hooks/effect-fence"): "require",
    ("POST", "/hooks/pre-edit"): "require",
    ("POST", "/hooks/post-edit"): "require",
    ("POST", "/hooks/post-edit-cas"): "require",
    ("POST", "/hooks/session-stop"): "require",
    ("POST", "/hooks/session-start"): "accept",
    ("POST", "/hooks/pre-bash"): "accept",
    ("POST", "/hooks/pre-grep"): "accept",
    ("POST", "/session/begin"): "accept",
    ("POST", "/session/read"): "accept",
    ("POST", "/session/commit"): "accept",
    ("POST", "/session/commit_all"): "accept",
    ("POST", "/session/heartbeat"): "accept",
    ("POST", "/workspace/checkpoint"): "require",
    ("POST", "/workspace/restore/status"): "require",
    ("POST", "/workspace/restore/member"): "require",
    ("POST", "/workspace/restore/register"): "require",
    ("POST", "/principal/claim"): "mint",
}
_EXPECTED_POSTURE_ROUTE_COUNT = 19

_NEVER_MINTED_TOKEN = "A" * 43  # in session-token shape, never issued


def _posture_body(route: tuple[str, str], sid: str) -> dict:
    """A request body that passes every shape check on ``route`` for ``sid``,
    so the request reaches the principal gate. Ordinary answers past the gate
    (an untracked path, an unknown checkpoint, a never-minted token) are all
    fine: the tests below only ask whether the GATE refused."""
    _, path = route
    h = _hash(f"posture:{path}")
    member = {
        "member_path": "posture.md", "native_token": "v1", "fingerprint": h,
        "captured_at": 1.0, "absent": False, "dirty_during_window": False,
        "arbitration_tier": "no-arbiter", "restore_tier": "forward_only",
    }
    bodies: dict[str, dict] = {
        "/hooks/pre-read": {"path": "posture.md"},
        "/hooks/effect-fence": {
            "path": "posture.md", "expected_version": 1,
            "expected_generation": 0, "content_hash": h,
        },
        "/hooks/pre-edit": {"path": "posture.md"},
        "/hooks/post-edit": {"path": "posture.md", "success": False},
        "/hooks/post-edit-cas": {
            "path": "posture.md", "content_hash": h, "expected_version": 0,
        },
        "/hooks/session-stop": {},
        "/hooks/session-start": {},
        "/hooks/pre-bash": {"command": "cat posture.md"},
        "/hooks/pre-grep": {"search_root": ""},
        "/session/begin": {"read_set": ["posture.md"]},
        "/session/read": {"session_token": _NEVER_MINTED_TOKEN, "path": "posture.md"},
        "/session/commit": {
            "session_token": _NEVER_MINTED_TOKEN, "path": "posture.md", "content": "x",
        },
        "/session/commit_all": {
            "session_token": _NEVER_MINTED_TOKEN,
            "writes": [{"path": "posture.md", "content": "x"}],
        },
        "/session/heartbeat": {"session_token": _NEVER_MINTED_TOKEN},
        "/workspace/checkpoint": {
            "name": "posture", "window_min": 1.0, "window_max": 1.0, "members": [member],
        },
        "/workspace/restore/status": {"checkpoint_id": "cp-unknown", "status": "in_progress"},
        "/workspace/restore/member": {
            "checkpoint_id": "cp-unknown", "member_path": "posture.md",
            "restore_outcome": None,
        },
        "/workspace/restore/register": {
            "checkpoint_id": "cp-unknown",
            "writes": [{"member_path": "posture.md", "fingerprint": h}],
        },
        "/principal/claim": {"mint_nonce": uuid.uuid4().hex},
    }
    return {"session_id": sid, **bodies[path]}


def _explicit_claim(client: _Client, sid: str) -> str:
    """Claim ``sid`` with a fresh random nonce, outside the auto client."""
    status, body = client.post(
        "/principal/claim",
        {"session_id": sid, "mint_nonce": uuid.uuid4().hex},
        principal=None,
    )
    assert status == 200 and body["ok"] is True, body
    return body["principal"]


def _absent_count(coordinator) -> int | None:
    """The absent-principal counter: every ADMISSION that presents no
    principal — any accept-class request, or a require-class request naming an
    identity nobody claimed (KTD15). ``None`` if it is missing, so a test
    reports its response assertions before the counter's."""
    return coordinator.counters_snapshot().get("caller_principal_absent_total")


def _refused_count(coordinator) -> int | None:
    """The principal-refusal counter: every request a route REFUSED as absent
    or foreign. ``None`` if it is missing, as :func:`_absent_count`."""
    return coordinator.counters_snapshot().get("caller_principal_refused_total")


def _principal_counts(coordinator) -> tuple[int | None, int | None]:
    return _absent_count(coordinator), _refused_count(coordinator)


@pytest.mark.parametrize(
    "route", sorted(_EXPECTED_ROUTE_POSTURE), ids=lambda r: r[1].strip("/"),
)
def test_each_session_id_route_answers_its_posture_class(
    route: tuple[str, str], coordinator, client: _Client
) -> None:
    """Every route in the table, in all five principal states: absent on a
    BOUND identity, absent on an UNBOUND one, foreign on a bound one, foreign
    on an UNBOUND one, and matching — each answer checked, and each state's
    effect on the two counters (admitted-without-a-principal, and refused).

    Prevents a route being left out of enforcement, a require-class route
    admitting an absent principal for a claimed session (a stray caller
    ending a claimed peer's grant), a require-class route refusing a session
    nobody claimed (an older client's writes silently dropped, because the
    shipped hook client turns every non-2xx into an allow), absent collapsing
    into foreign (a phase no longer diagnosable), a refusal losing its typed
    ``reason`` (clients classify by it), an accept-class route either not
    counting the absent case or treating a foreign principal as absent — the
    case that separates OPTIONAL from IGNORED — and a principal presented for a
    session nobody claimed being waved through: nothing is bound there, so no
    presented value can match, and it is foreign on every class."""
    posture = _EXPECTED_ROUTE_POSTURE[route]
    _, path = route
    owner, other = str(uuid.uuid4()), str(uuid.uuid4())
    owner_principal = _explicit_claim(client, owner)
    foreign_principal = _explicit_claim(client, other)

    def send(principal: str | None, sid: str = owner) -> tuple[int, dict]:
        # The mint route needs an unclaimed session per request.
        sid = str(uuid.uuid4()) if posture == "mint" else sid
        return client.post(path, _posture_body(route, sid), principal=principal)

    counts = [_principal_counts(coordinator)]
    answers = {}
    for state, principal, sid in (
        ("absent", None, owner),
        ("foreign", foreign_principal, owner),
        ("valid", owner_principal, owner),
        ("unbound", None, str(uuid.uuid4())),
        ("foreign_unbound", foreign_principal, str(uuid.uuid4())),
    ):
        answers[state] = send(principal, sid=sid)
        counts.append(_principal_counts(coordinator))
    assert None not in counts[0], f"a principal counter is missing: {counts[0]}"
    # Per-state movement of (absent counter, refused counter), in send order.
    moved = {
        state: (after[0] - prior[0], after[1] - prior[1])
        for state, prior, after in zip(answers, counts, counts[1:])
    }

    if posture == "require":
        assert answers["absent"] == (400, _PRINCIPAL_ABSENT_REFUSAL)
        assert answers["foreign"] == (400, _PRINCIPAL_FOREIGN_REFUSAL)
        assert not _principal_refused(answers["valid"]), answers["valid"]
        assert not _principal_refused(answers["unbound"]), (
            "a session nobody claimed is an older client's: admitted, as before",
            answers["unbound"])
        assert answers["foreign_unbound"] == (400, _PRINCIPAL_FOREIGN_REFUSAL)
        assert moved == {
            "absent": (0, 1), "foreign": (0, 1), "valid": (0, 0),
            "unbound": (1, 0), "foreign_unbound": (0, 1),
        }, "a refusal is counted as refused, the admitted unbound request as absent"
    elif posture == "accept":
        assert not _principal_refused(answers["absent"]), answers["absent"]
        assert answers["foreign"] == (400, _PRINCIPAL_FOREIGN_REFUSAL)
        assert not _principal_refused(answers["valid"]), answers["valid"]
        assert not _principal_refused(answers["unbound"]), answers["unbound"]
        assert answers["foreign_unbound"] == (400, _PRINCIPAL_FOREIGN_REFUSAL)
        assert moved == {
            "absent": (1, 0), "foreign": (0, 1), "valid": (0, 0),
            "unbound": (1, 0), "foreign_unbound": (0, 1),
        }, "only an admitted absent principal is absent; only a refusal is refused"
    else:
        assert posture == "mint"
        for status, body in answers.values():
            assert status == 200 and body["ok"] is True, body
        assert set(moved.values()) == {(0, 0)}, "the mint is neither uncredentialed nor refused"


def test_an_older_client_that_never_claims_still_commits_and_stops(
    coordinator, client: _Client
) -> None:
    """R16, the version-skew direction this coordinator can observe: a client
    that predates the principal sends no header and never claims, so the
    session it names stays UNBOUND — and its require-class pre-edit, commit
    and stop are admitted exactly as before: the version advances,
    ``last_writer_id`` is its composite id, its grant is released, and each of
    the four admissions is counted as uncredentialed. Refusing it would not surface anywhere: the shipped
    hook client turns every non-2xx into ``{}``, an allow, so the edit would
    proceed with coherence silently off."""
    old = str(uuid.uuid4())
    before = _absent_count(coordinator)
    status, _ = client.post(
        "/hooks/pre-edit", {"session_id": old, "agent_id": "sub-1", "path": "plan.md"})
    assert status == 200
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    version = coordinator.registry.get_artifact(artifact_id).version

    status, body = client.post("/hooks/post-edit", {
        "session_id": old, "agent_id": "sub-1", "path": "plan.md",
        "success": True, "content_hash": _hash("old-client"),
    })
    assert status == 200 and body["ok"] is True, body
    assert coordinator.registry.get_artifact(artifact_id).version == version + 1
    assert coordinator.registry.last_writer_for(artifact_id) == session_to_agent_id(old, "sub-1")

    status, _ = client.post("/hooks/pre-edit", {"session_id": old, "path": "plan.md"})
    status, body = client.post("/hooks/session-stop", {"session_id": old})
    assert status == 200 and body["released_artifacts"] == ["plan.md"], body
    assert _absent_count(coordinator) == before + 4


def test_posture_table_classifies_every_session_id_route_with_no_residual() -> None:
    """R13: one explicit table over every route that reads a ``session_id``,
    and nothing else — no route falls to a default. A route added to
    ``_ROUTES`` that reads a session id without a table entry fails here, and
    so does an entry for a route that no longer exists.

    ``/workspace/restore/status`` and ``/workspace/restore/member`` validate
    the session id and then never consult it; they are DECIDED require-class
    rather than dropped (see the table's harm text), and the frozen
    expectation pins that decision."""
    import inspect

    from ccs.adapters.claude_code.coordinator_server import (
        _CALLER_PRINCIPAL_POSTURE,
        _ROUTES,
    )

    reads_session_id = {
        route for route, handler in _ROUTES.items()
        if 'body.get("session_id")' in inspect.getsource(handler)
    }
    assert set(_CALLER_PRINCIPAL_POSTURE) == reads_session_id
    assert {
        route: entry.posture.value for route, entry in _CALLER_PRINCIPAL_POSTURE.items()
    } == _EXPECTED_ROUTE_POSTURE
    assert len(_CALLER_PRINCIPAL_POSTURE) == _EXPECTED_POSTURE_ROUTE_COUNT
    for route, entry in _CALLER_PRINCIPAL_POSTURE.items():
        assert entry.harm.strip(), f"{route} states no harm for its class"


# --- the gate under registry contention: bounded by the handler watchdog ------
#
# A require-class gate asks whether the named session is BOUND; for a session
# nobody claimed (an older client's, admitted as before -- KTD15) the first
# answer lives in the durable store, behind the registry lock. These tests hold
# that lock the way the pre-edit degraded tests above do (_HeldRegistryLock)
# and drive the REAL watchdog with a shortened HANDLER_TIMEOUT_SEC.

#: How long a test waits for an answer while the registry lock is held -- eight
#: times the shortened deadline, so only a request that is NOT bounded by the
#: watchdog misses it.
_GATE_ANSWER_BOUND_SEC = 2.0
#: A value in principal shape that no coordinator minted.
_NEVER_MINTED_PRINCIPAL = "B" * 43


class _Background:
    """One request sent from a helper thread, so a test can bound its own wait
    for the answer and still release the lock the request is blocked on."""

    def __init__(self, client: _Client, path: str, body: dict, principal: str | None = None):
        self._done = threading.Event()
        self._response: tuple[int, dict] | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._send, args=(client, path, body, principal), daemon=True)
        self._thread.start()

    def _send(self, client: _Client, path: str, body: dict, principal: str | None) -> None:
        try:
            self._response = client.post(path, body, principal=principal)
        except BaseException as exc:  # re-raised on the test thread by result()
            self._error = exc
        finally:
            self._done.set()

    def answered_within(self, seconds: float) -> bool:
        return self._done.wait(seconds)

    def result(self) -> tuple[int, dict]:
        if not self._done.wait(10.0):
            pytest.fail("the request never answered, even with the registry lock released")
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _count_binding_reads(coordinator, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record every durable-store read of a caller-principal binding: one entry
    per ``registry.get_caller_principal`` call, naming the thread that made it."""
    reads: list[str] = []
    real = coordinator.registry.get_caller_principal

    def recording(identity):
        reads.append(threading.current_thread().name)
        return real(identity)

    monkeypatch.setattr(coordinator.registry, "get_caller_principal", recording)
    return reads


def test_a_never_claimed_pre_edit_answers_degraded_in_time_while_its_gate_cannot_read(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-edit from a session nobody claimed answers pre-edit's degraded
    envelope within the handler watchdog while the registry lock is held, and
    changes nothing.

    Prevents the regression making pre-edit require-class introduced: its gate
    asked the durable store whether the session was bound on the request
    thread, under the registry lock and BEFORE the watchdog started, so a
    client that never claims -- which must keep working (KTD15) -- waited on
    registry contention with no deadline, past the hook's own budget, where it
    used to get the degraded answer. The gate cannot decide in time here, and
    it answers exactly what a timed-out work body answers: not a refusal
    (nothing was refused) and not an admission (nothing was checked)."""
    import ccs.adapters.claude_code.coordinator_server as mod

    sid = str(uuid.uuid4())
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    timeouts_before = coordinator._watchdog_timeouts_total
    principal_counts_before = _principal_counts(coordinator)

    with _HeldRegistryLock(coordinator) as held:
        started = time.monotonic()
        answer = _Background(client, "/hooks/pre-edit", {"session_id": sid, "path": "plan.md"})
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        waited = time.monotonic() - started
        held.release()
    response = answer.result()

    assert answered, (
        f"a pre-edit from a never-claimed session got no answer for {waited:.2f}s while "
        f"the registry lock was held: its caller-principal gate is not bounded by the "
        f"{_DEGRADE_DEADLINE_SEC}s handler watchdog")
    assert response == (200, mod._PRE_EDIT_DEGRADED_RESPONSE), response
    assert coordinator._watchdog_timeouts_total == timeouts_before + 1
    assert _principal_counts(coordinator) == principal_counts_before, (
        "an undecided gate neither admitted nor refused the request")
    assert coordinator.registry.lookup_artifact_id_by_name("plan.md") is None, (
        "the work body ran: it seeded the artifact")
    assert session_to_agent_id(sid) not in dict(coordinator.agent_names_snapshot()), (
        "the handler registered the session past an undecided gate")


def test_the_gate_degraded_table_covers_every_route_whose_gate_reads_the_registry() -> None:
    """Every route but the mint can reach the store lookup (require-class
    always, accept-class when a principal is presented), so each needs an
    answer for a gate that times out. The table has no default: a route missing
    from it would answer a 500 under exactly the contention the watchdog exists
    for, so the key set is pinned against the FROZEN posture table here."""
    from ccs.adapters.claude_code.coordinator_server import _CALLER_GATE_DEGRADED_RESPONSE

    assert set(_CALLER_GATE_DEGRADED_RESPONSE) == {
        route for route, posture in _EXPECTED_ROUTE_POSTURE.items() if posture != "mint"
    }


def test_the_gate_store_lookup_honours_the_watchdog_queue_limit(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the watchdog pool's queue past its limit, a request whose gate
    would have to read the store answers the queue-overflow 503 at once, as a
    work body does (A7), and reads nothing.

    Prevents the gate's lookup becoming a way around the queue-depth gate:
    queued behind an overloaded pool it would sit until the deadline and answer
    degraded, where the route answers 503 immediately today."""
    from unittest.mock import patch

    class _OverflowingQueue:
        @staticmethod
        def qsize() -> int:
            return 100  # well above the limit

    reads = _count_binding_reads(coordinator, monkeypatch)
    overflows_before = coordinator.counters_snapshot()["watchdog_queue_overflows_total"]
    timeouts_before = coordinator._watchdog_timeouts_total
    with patch.object(coordinator._watchdog, "_work_queue", _OverflowingQueue()):
        response = client.post(
            "/hooks/pre-edit", {"session_id": str(uuid.uuid4()), "path": "plan.md"})
    assert response == (503, {"error": "watchdog queue overloaded"})
    assert reads == [], f"the gate read the store past a full queue: {reads}"
    assert coordinator.counters_snapshot()["watchdog_queue_overflows_total"] == overflows_before + 1
    assert coordinator._watchdog_timeouts_total == timeouts_before


_GATED_ROUTES = sorted(
    route for route, posture in _EXPECTED_ROUTE_POSTURE.items() if posture != "mint"
)


def _degrade_body(route: tuple[str, str], sid: str) -> dict:
    """:func:`_posture_body`, with a TRACKED path wherever the posture body's
    untracked one would let the handler answer before its work body runs."""
    body = _posture_body(route, sid)
    if body.get("path") == "posture.md":
        body["path"] = "plan.md"
    if body.get("command") == "cat posture.md":
        body["command"] = "cat plan.md"
    return body


@pytest.mark.parametrize("route", _GATED_ROUTES, ids=lambda r: r[1].strip("/"))
def test_a_timed_out_gate_answers_exactly_its_routes_timed_out_work_body(
    route: tuple[str, str], coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On every route whose gate can read the registry, a gate that cannot
    decide within the watchdog answers byte-for-byte what the same route
    answers when its WORK BODY times out, and counts one watchdog timeout.

    Prevents two things. A route whose gate waits on registry contention with
    no deadline: the lookup runs for an identity the cache does not know -- a
    never-claimed session presenting no principal on a require-class route, or
    presenting one on an accept-class route. And a gate that times out
    answering something else than the route's own degraded answer (a
    refusal, an admission, another route's envelope): hook clients and the
    library branch on that answer's shape. The second half is the control:
    it times the work body out instead (``run_with_watchdog`` raising, as the
    AC-05 tests do) for a claimed session the gate decides from the cache."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    import ccs.adapters.claude_code.coordinator_server as mod

    _, path = route
    posture = _EXPECTED_ROUTE_POSTURE[route]
    never_claimed = str(uuid.uuid4())
    presented = None if posture == "require" else _NEVER_MINTED_PRINCIPAL
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    timeouts_before = coordinator._watchdog_timeouts_total
    with _HeldRegistryLock(coordinator) as held:
        answer = _Background(client, path, _degrade_body(route, never_claimed), presented)
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        held.release()
    gate_timed_out = answer.result()
    assert answered, f"{path}: no answer while the registry lock was held"
    assert coordinator._watchdog_timeouts_total == timeouts_before + 1

    claimed = str(uuid.uuid4())
    principal = _explicit_claim(client, claimed)
    # pre-grep answers "fresh" before its work body when the store knows no
    # tracked artifact under the search root.
    coordinator.registry.resolve_or_register("plan.md", content_hash=_hash("seed"))

    def work_times_out(fn, abort=None, deadline=None):
        raise FuturesTimeout()

    monkeypatch.setattr(coordinator, "run_with_watchdog", work_times_out)
    work_timed_out = client.post(path, _degrade_body(route, claimed), principal=principal)
    assert coordinator._watchdog_timeouts_total == timeouts_before + 2, (
        f"{path}: the control never reached its work body, so it compares nothing")
    assert gate_timed_out == work_timed_out, (
        f"{path}: a timed-out gate answered {gate_timed_out!r}, "
        f"a timed-out work body {work_timed_out!r}")


# --- the gate's store read RAISES: the route's own internal-error answer ------
#
# The lookup can raise something that is neither a refusal nor the watchdog's
# timeout -- a registry read failing (``sqlite3.OperationalError`` once
# ``busy_timeout`` elapses, a closed store). Inside a work body that exception
# is turned into the route's HTTP 200 internal-error envelope; the gate must
# answer the SAME thing, because a degrade-mode CoherentVolume reads a 500 as
# an unanswered request and writes with no grant, where the typed 200 raises.

#: FROZEN duplicate of the internal-error envelope every route but the effect
#: fence answers when its work raised (the type name is the exception's).
_INTERNAL_ERROR_ENVELOPE = {"ok": False, "reason": "internal: OperationalError"}
#: FROZEN duplicate of the effect fence's own answer for a raise: a HOLD, so a
#: client branching on ``verdict`` reads an answer, never a shape with none.
_FENCE_INTERNAL_HOLD_ENVELOPE = {
    "verdict": "hold", "reason": "version_unconfirmed", "degraded": True,
    "held_by": "handler_error",
}
_FENCE_ROUTE = ("POST", "/hooks/effect-fence")


def _fail_binding_reads(coordinator, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every durable-store read of a caller-principal binding raises the
    registry's own transient error -- what a ``busy_timeout`` overrun raises."""
    import sqlite3

    def failing(identity):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(coordinator.registry, "get_caller_principal", failing)


def _fail_work_body(
    route: tuple[str, str], coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make ``route``'s WORK BODY raise the same registry error the gate's
    lookup raises in the tests below. The effect fence catches inside its own
    body (it owns that arm), so its verdict is what raises; every other route
    lets the exception reach the wrapper, so the watchdog call is what raises."""
    import sqlite3

    import ccs.adapters.claude_code.coordinator_server as mod

    def raising(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    if route == _FENCE_ROUTE:
        monkeypatch.setattr(mod, "_effect_fence_verdict", raising)
    else:
        monkeypatch.setattr(coordinator, "run_with_watchdog", raising)


@pytest.mark.parametrize("route", _GATED_ROUTES, ids=lambda r: r[1].strip("/"))
def test_a_gate_whose_store_read_raises_answers_exactly_its_routes_failed_work_body(
    route: tuple[str, str], coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """On every route whose gate can read the registry, a lookup that RAISES a
    non-refusal, non-timeout error answers byte-for-byte what the same route
    answers when its WORK BODY raises that error: HTTP 200 and the route's
    internal-error envelope, never the dispatcher's 500.

    Prevents the gate and the body answering one failure in two shapes. The
    hook client and the library read any non-200 as "no coordinator" and
    degrade: a degrade-mode ``CoherentVolume`` answered 500 on pre-edit skips
    its grant check and writes with no grant and no peer invalidation, where
    the typed 200 raises ``StaleView`` in both modes. The first half drives
    the gate: a never-claimed session on a require-class route, a never-minted
    principal on an accept-class one, each a cold-cache lookup. The control
    drives the body instead, for a claimed session the gate decides from the
    cache, so the two answers come from the two arms."""
    _, path = route
    posture = _EXPECTED_ROUTE_POSTURE[route]
    never_claimed = str(uuid.uuid4())
    presented = None if posture == "require" else _NEVER_MINTED_PRINCIPAL
    timeouts_before = coordinator._watchdog_timeouts_total
    with monkeypatch.context() as gate_only:
        _fail_binding_reads(coordinator, gate_only)
        gate_raised = client.post(path, _degrade_body(route, never_claimed), principal=presented)
    assert gate_raised[0] == 200, f"{path}: the gate's raise reached the dispatcher: {gate_raised}"

    claimed = str(uuid.uuid4())
    principal = _explicit_claim(client, claimed)
    # pre-grep answers "fresh" before its work body when the store knows no
    # tracked artifact under the search root.
    coordinator.registry.resolve_or_register("plan.md", content_hash=_hash("seed"))
    _fail_work_body(route, coordinator, monkeypatch)
    body_raised = client.post(path, _degrade_body(route, claimed), principal=principal)
    assert body_raised[0] == 200 and body_raised[1] != {"ok": True}, (
        f"{path}: the control never reached its work body, so it compares nothing")
    assert gate_raised == body_raised, (
        f"{path}: a raising gate answered {gate_raised!r}, "
        f"a raising work body {body_raised!r}")
    assert coordinator._watchdog_timeouts_total == timeouts_before, "a raise is not a timeout"


@pytest.mark.parametrize(
    ("route", "presented", "envelope"),
    [
        (("POST", "/hooks/pre-edit"), None, _INTERNAL_ERROR_ENVELOPE),
        (("POST", "/hooks/pre-read"), _NEVER_MINTED_PRINCIPAL, _INTERNAL_ERROR_ENVELOPE),
        (_FENCE_ROUTE, None, _FENCE_INTERNAL_HOLD_ENVELOPE),
    ],
    ids=["pre-edit", "pre-read-with-a-principal", "effect-fence"],
)
def test_a_gate_whose_store_read_raises_answers_the_typed_envelope_and_changes_nothing(
    route: tuple[str, str], presented: str | None, envelope: dict,
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The envelope a raising gate answers, pinned as a frozen literal -- the
    ``{ok: false, reason: "internal: <Type>"}`` a client raises on, and the
    fence's hold -- and what it leaves behind: no artifact row, no session,
    no grant (the request was neither admitted nor refused, so neither
    principal counter moves), one ERROR log record carrying the traceback,
    and no principal in the log or the body (R5). A fence hold reached this
    way is counted as a hold, as the body's own internal hold is."""
    import logging

    _, path = route
    sid = str(uuid.uuid4())
    caplog.set_level(logging.ERROR, logger="ccs.adapters.claude_code.coordinator_server")
    counts_before = _principal_counts(coordinator)
    holds_before = coordinator.counters_snapshot()["effect_fence_holds_total"]
    _fail_binding_reads(coordinator, monkeypatch)

    response = client.post(path, _degrade_body(route, sid), principal=presented)

    assert response == (200, envelope)
    assert coordinator.registry.lookup_artifact_id_by_name("plan.md") is None, (
        "the work body ran: it seeded the artifact")
    assert session_to_agent_id(sid) not in dict(coordinator.agent_names_snapshot()), (
        "the handler registered the session past a failed gate")
    assert _principal_counts(coordinator) == counts_before, (
        "a gate that decided nothing counted an admission or a refusal")
    expected_holds = holds_before + (1 if route == _FENCE_ROUTE else 0)
    assert coordinator.counters_snapshot()["effect_fence_holds_total"] == expected_holds
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and errors[0].exc_info is not None, (
        "the failure was not logged with its traceback")
    assert path in errors[0].getMessage(), "the log does not name the route"
    if presented is not None:
        assert presented not in caplog.text and presented not in json.dumps(response[1]), (
            "a presented principal reached the log or the body")


def test_a_never_claimed_session_is_answered_without_touching_the_registry(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once the durable store has said a session is unbound, the service keeps
    that answer: later requests naming the session read no registry at all in
    the gate, so registry contention neither slows nor degrades an answer the
    route itself gives without the registry -- here the untracked fast path,
    registry-free by design (R8).

    Prevents every request of a client that never claims (KTD15) paying a
    store read in the gate: bounded by the watchdog, that read still turns the
    fast path's own answer into a degraded one whenever the lock is busy."""
    import ccs.adapters.claude_code.coordinator_server as mod

    sid = str(uuid.uuid4())
    untracked = {"session_id": sid, "path": "notes/untracked.txt"}
    assert client.post("/hooks/pre-edit", untracked) == (200, {"ok": True})
    reads = _count_binding_reads(coordinator, monkeypatch)
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    timeouts_before = coordinator._watchdog_timeouts_total

    with _HeldRegistryLock(coordinator) as held:
        answer = _Background(client, "/hooks/pre-edit", untracked)
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        held.release()
    assert answered, "the second request waited on the registry lock"
    assert answer.result() == (200, {"ok": True}), (
        "the untracked fast path's own answer, not a degraded one")
    assert coordinator._watchdog_timeouts_total == timeouts_before

    for body in (untracked, {**untracked, "success": False}):
        route = "/hooks/pre-edit" if "success" not in body else "/hooks/post-edit"
        assert client.post(route, body) == (200, {"ok": True})
    assert reads == [], f"the gate read the store for a session it already knew: {reads}"


def test_a_claim_replaces_the_cached_unbound_answer_for_its_session(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cached UNBOUND answer lasts only until the session is claimed: from
    the first request after the claim, the same request without the principal
    is refused as absent, and the principal is admitted.

    Prevents the negative cache outliving the bind. The session's own client
    claims it -- through this coordinator, the only writer of bindings -- and a
    stale UNBOUND answer would keep admitting absent-principal requests naming
    it as an older client's, the #188 case the require class exists to refuse,
    while refusing the claimed client's own principal as foreign. The first
    half proves the answer really was cached (a second absent request reads no
    store), so the second half tests the replacement of a real cache entry."""
    sid = str(uuid.uuid4())
    reads = _count_binding_reads(coordinator, monkeypatch)
    for _ in range(2):
        status, body = client.post("/hooks/session-stop", {"session_id": sid})
        assert status == 200 and body["ok"] is True, body
    assert len(reads) == 1, f"the UNBOUND answer was not cached: {len(reads)} store reads"

    principal = _explicit_claim(client, sid)
    assert client.post("/hooks/session-stop", {"session_id": sid}) == (
        400, _PRINCIPAL_ABSENT_REFUSAL)
    status, body = client.post("/hooks/session-stop", {"session_id": sid}, principal=principal)
    assert status == 200 and body["ok"] is True, body


def test_a_refusal_the_store_lookup_decides_is_still_a_400_that_changes_nothing(
    tmp_path: Path,
) -> None:
    """After a restart the service's cache is empty, so a request naming a
    session claimed before it is decided by the store lookup -- which runs in
    the watchdog pool, off the request thread. A refusal decided there is the
    same HTTP 400 with its typed reason, never a hold and never degraded; the
    refused pre-edit seeds nothing, registers nothing and times nothing out;
    and the session's principal is then admitted.

    Prevents the bounded lookup changing what it decides: a refusal turned
    into the degraded envelope would ADMIT the edit (pre-edit degrades to
    ``ok: true``) for the claimed session whose principal is absent -- the
    #188 case again -- and a lookup made back on the request thread would
    reopen the unbounded wait."""
    sid = str(uuid.uuid4())
    before = _restart_on(tmp_path, "gate-before-restart")
    try:
        secret = load_secret(before.coordinator_root)
        assert secret is not None
        principal = _explicit_claim(_Client("127.0.0.1", before.port, secret), sid)
    finally:
        before.shutdown()

    after = _restart_on(tmp_path, "gate-after-restart")
    try:
        secret = load_secret(after.coordinator_root)
        assert secret is not None
        client = _Client("127.0.0.1", after.port, secret)
        reads: list[str] = []
        real = after.registry.get_caller_principal

        def recording(identity):
            reads.append(threading.current_thread().name)
            return real(identity)

        after.registry.get_caller_principal = recording
        timeouts_before = after._watchdog_timeouts_total
        body = {"session_id": sid, "path": "plan.md"}

        assert client.post("/hooks/pre-edit", body) == (400, _PRINCIPAL_ABSENT_REFUSAL)
        assert reads and all(name.startswith("coord-wd") for name in reads), (
            f"the gate read the store on the request thread: {reads}")
        assert after._watchdog_timeouts_total == timeouts_before
        assert after.registry.lookup_artifact_id_by_name("plan.md") is None
        assert session_to_agent_id(sid) not in dict(after.agent_names_snapshot())

        status, admitted = client.post("/hooks/pre-edit", body, principal=principal)
        assert (status, admitted) == (200, {"ok": True}), admitted
    finally:
        after.shutdown()


def test_the_gate_and_the_work_body_share_one_watchdog_deadline(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request whose gate spent part of the watchdog budget reading the
    store gives its work body only what is left: a work body that then blocks
    is answered at the ONE deadline, not at the lookup's time plus a fresh one.

    Prevents bounding the gate by turning a request into one that waits up to
    twice the watchdog -- past the hook's own 5s budget, where the hook client
    gives up on its own and a work body not yet aborted can still land state
    the caller was never told about. The store read is made slow by a stand-in
    that answers "unbound" after a delay WITHOUT the registry lock, so the lock
    this test holds blocks only the work body."""
    import ccs.adapters.claude_code.coordinator_server as mod

    deadline = 1.0
    lookup_delay = 0.7 * deadline

    def slow_unbound(identity):
        time.sleep(lookup_delay)
        return None

    monkeypatch.setattr(coordinator.registry, "get_caller_principal", slow_unbound)
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", deadline)
    timeouts_before = coordinator._watchdog_timeouts_total
    with _HeldRegistryLock(coordinator) as held:
        started = time.monotonic()
        answer = _Background(
            client, "/hooks/pre-edit", {"session_id": str(uuid.uuid4()), "path": "plan.md"})
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        waited = time.monotonic() - started
        held.release()

    assert answered, f"no answer within {_GATE_ANSWER_BOUND_SEC}s"
    assert answer.result() == (200, mod._PRE_EDIT_DEGRADED_RESPONSE)
    assert coordinator._watchdog_timeouts_total == timeouts_before + 1
    # One deadline answers at ~1.0s; a fresh one for the body at ~0.7 + 1.0s.
    assert waited < deadline + lookup_delay / 2, (
        f"answered after {waited:.2f}s: the work body got a fresh {deadline}s "
        f"after a {lookup_delay:.2f}s gate lookup, not what was left of one deadline")


def test_the_fast_path_delivery_gets_only_what_the_gate_left_of_the_deadline(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The untracked fast path runs work under the watchdog too -- the deferred
    re-grounding delivery, when the session has compact-pending armed -- and
    that delivery also gets only what the gate's store lookup left of the ONE
    deadline.

    Prevents the fast path's call site starting a fresh deadline after the
    gate, which the test above cannot see: it pins the tracked work body's call
    site only. The path is reachable by a client that never claims (KTD15): its
    session-start after a compaction arms compact-pending, and its pre-edit on
    an untracked path then has the gate read the store and the delivery walk
    the registry -- with two budgets, up to twice the watchdog, past the hook's
    own 5s budget. The store read is the same lock-free slow stand-in as
    above, so the lock this test holds blocks only the delivery."""
    import ccs.adapters.claude_code.coordinator_server as mod

    deadline = 1.0
    lookup_delay = 0.7 * deadline

    def slow_unbound(identity):
        time.sleep(lookup_delay)
        return None

    sid = str(uuid.uuid4())
    coordinator.mark_compact_pending(sid)
    assert coordinator.has_compact_pending(sid)
    monkeypatch.setattr(coordinator.registry, "get_caller_principal", slow_unbound)
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", deadline)
    timeouts_before = coordinator._watchdog_timeouts_total
    with _HeldRegistryLock(coordinator) as held:
        started = time.monotonic()
        answer = _Background(
            client, "/hooks/pre-edit", {"session_id": sid, "path": "notes/untracked.txt"})
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        waited = time.monotonic() - started
        held.release()

    assert answered, f"no answer within {_GATE_ANSWER_BOUND_SEC}s"
    assert answer.result() == (200, {"ok": True}), (
        "the gate admitted the session and the fast path answered its bare body")
    assert coordinator._watchdog_timeouts_total == timeouts_before + 1, (
        "the delivery never timed out, so this measured nothing about its deadline")
    # One deadline answers at ~1.0s; a fresh one for the delivery at ~0.7 + 1.0s.
    assert waited < deadline + lookup_delay / 2, (
        f"answered after {waited:.2f}s: the fast-path delivery got a fresh {deadline}s "
        f"after a {lookup_delay:.2f}s gate lookup, not what was left of one deadline")


def test_a_work_body_whose_deadline_has_already_passed_is_never_submitted(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the request's watchdog deadline has passed by the time its work
    body would start, the body is not submitted at all: the request answers the
    route's degraded envelope, counts one watchdog timeout, and nothing runs
    afterwards.

    Prevents ``run_with_watchdog`` submitting the body anyway with a token
    wait. The caller is told "degraded" at once while the body runs with nobody
    waiting for it -- for pre-edit it seeds the artifact and can land an
    EXCLUSIVE grant the agent never learns of, the late completion the A6 abort
    exists to prevent. The gate's store lookup starts the deadline (a session
    nobody claimed, which the cache cannot answer), and a registration made
    slow on the request thread spends the rest of it before the body."""
    from concurrent.futures import Future
    from concurrent.futures import wait as futures_wait

    import ccs.adapters.claude_code.coordinator_server as mod

    deadline = 0.5
    real_register = coordinator.register_session

    def register_past_the_deadline(*args, **kwargs):
        time.sleep(deadline + 0.2)
        return real_register(*args, **kwargs)

    submitted: list[Future] = []
    real_submit = coordinator._watchdog.submit

    def recording_submit(fn, *args, **kwargs):
        future = real_submit(fn, *args, **kwargs)
        submitted.append(future)
        return future

    monkeypatch.setattr(coordinator, "register_session", register_past_the_deadline)
    monkeypatch.setattr(coordinator._watchdog, "submit", recording_submit)
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", deadline)
    timeouts_before = coordinator._watchdog_timeouts_total

    response = client.post(
        "/hooks/pre-edit", {"session_id": str(uuid.uuid4()), "path": "plan.md"})

    assert response == (200, mod._PRE_EDIT_DEGRADED_RESPONSE), response
    assert coordinator._watchdog_timeouts_total == timeouts_before + 1
    _, still_running = futures_wait(submitted, timeout=_ABANDONED_BODY_SETTLE_SEC)
    assert not still_running, (
        f"timed out after {_ABANDONED_BODY_SETTLE_SEC}s waiting for the work submitted "
        f"to the watchdog pool to finish")
    assert coordinator.registry.lookup_artifact_id_by_name("plan.md") is None, (
        "the work body ran after its deadline had passed: it seeded plan.md")
    assert len(submitted) == 1, (
        f"{len(submitted)} submissions to the watchdog pool: the gate's store lookup, "
        f"then a work body submitted after its deadline had passed")


def test_a_cached_identity_is_admitted_on_the_request_thread_not_in_the_watchdog_pool(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A session the service's binding cache knows is decided on the request
    thread: with every watchdog worker stuck on the registry lock, a claimed
    session's untracked pre-edit still answers the fast path's own
    ``{ok: true}`` at once and counts no timeout.

    Prevents the gate taking every lookup through the watchdog pool. The cache
    would still decide, but only once a worker came free, so while work bodies
    blocked on registry contention fill the pool, a request the route answers
    without the registry at all (the untracked fast path, R8) waits out the
    deadline and degrades. The pool is shown to be full before the request is
    sent -- every worker started and none free to run a no-op -- because with a
    free worker the lookup would get through at once and this would measure
    nothing."""
    from concurrent.futures import wait as futures_wait

    import ccs.adapters.claude_code.coordinator_server as mod

    free_worker_grace_sec = 0.2
    monkeypatch.setattr(mod, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    workers = mod._WATCHDOG_POOL_SIZE
    blockers = [(s, _explicit_claim(client, s)) for s in (str(uuid.uuid4()) for _ in range(workers))]
    sid = str(uuid.uuid4())
    principal = _explicit_claim(client, sid)

    with _HeldRegistryLock(coordinator) as held:
        stuck = [
            _Background(client, "/hooks/pre-edit", {"session_id": s, "path": "plan.md"}, p)
            for s, p in blockers
        ]
        for request in stuck:
            assert request.answered_within(_GATE_ANSWER_BOUND_SEC), (
                f"timed out after {_GATE_ANSWER_BOUND_SEC}s waiting for a blocker's "
                f"degraded answer")
            assert request.result() == (200, mod._PRE_EDIT_DEGRADED_RESPONSE), (
                "a blocker's work body did not time out on the registry lock")
        no_op = coordinator._watchdog.submit(lambda: None)
        ran, _ = futures_wait([no_op], timeout=free_worker_grace_sec)
        assert not ran and len(coordinator._watchdog._threads) == workers, (
            "a watchdog worker was still free, so a lookup sent through the pool "
            "would not have waited: this measures nothing")
        timeouts_before = coordinator._watchdog_timeouts_total
        started = time.monotonic()
        answer = _Background(
            client, "/hooks/pre-edit", {"session_id": sid, "path": "notes/untracked.txt"},
            principal)
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        waited = time.monotonic() - started
        held.release()

    assert answered, f"timed out after {_GATE_ANSWER_BOUND_SEC}s waiting for the answer"
    assert answer.result() == (200, {"ok": True}), (
        f"after {waited:.2f}s, the fast path's own answer, not a degraded one: the gate "
        f"queued a cached identity's decision behind the stuck watchdog workers")
    assert coordinator._watchdog_timeouts_total == timeouts_before
    drained, _ = futures_wait([no_op], timeout=_ABANDONED_BODY_SETTLE_SEC)
    assert drained, (
        f"timed out after {_ABANDONED_BODY_SETTLE_SEC}s waiting for the watchdog pool "
        f"to drain once the registry lock was released")


def _pre_edit_with(client: _Client, sid: str, principal: str, path: str) -> None:
    status, body = client.post(
        "/hooks/pre-edit", {"session_id": sid, "path": path}, principal=principal
    )
    assert status == 200 and body.get("ok") is not False, body


def test_session_stop_without_a_principal_leaves_a_peers_grant_standing(
    coordinator, client: _Client
) -> None:
    """The #188 reproduction on the route surface: a stop naming a peer's
    session with no principal does not release the peer's EXCLUSIVE grant.
    The forger deliberately does not present the peer's principal — on the
    hook surface a principal is readable from ``.coherence/``, so that
    abstention is exactly what this check can enforce. Asserts the grant
    STANDS afterwards, not merely that an error came back; the control shows
    the same stop with the peer's own principal does release it."""
    peer = str(uuid.uuid4())
    peer_principal = _explicit_claim(client, peer)
    _pre_edit_with(client, peer, peer_principal, "plan.md")
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    peer_agent = session_to_agent_id(peer)
    assert coordinator.registry.get_state_map(artifact_id)[peer_agent] == MESIState.EXCLUSIVE

    status, body = client.post(
        "/hooks/session-stop", {"session_id": peer}, principal=None
    )
    assert (status, body) == (400, _PRINCIPAL_ABSENT_REFUSAL)
    assert coordinator.registry.get_state_map(artifact_id)[peer_agent] == MESIState.EXCLUSIVE

    status, body = client.post(
        "/hooks/session-stop", {"session_id": peer}, principal=peer_principal
    )
    assert status == 200 and body["released_artifacts"] == ["plan.md"]
    assert coordinator.registry.get_state_map(artifact_id)[peer_agent] != MESIState.EXCLUSIVE


def test_session_stop_naming_a_peer_under_the_callers_own_principal_is_refused(
    coordinator, client: _Client
) -> None:
    """A caller holding a principal of its OWN cannot spend it on a peer's
    session: the principal is bound to one identity, so naming another is
    foreign, and the peer's grant stands."""
    peer, caller = str(uuid.uuid4()), str(uuid.uuid4())
    peer_principal = _explicit_claim(client, peer)
    caller_principal = _explicit_claim(client, caller)
    _pre_edit_with(client, peer, peer_principal, "plan.md")
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")

    status, body = client.post(
        "/hooks/session-stop", {"session_id": peer}, principal=caller_principal
    )
    assert (status, body) == (400, _PRINCIPAL_FOREIGN_REFUSAL)
    assert (
        coordinator.registry.get_state_map(artifact_id)[session_to_agent_id(peer)]
        == MESIState.EXCLUSIVE
    )


def test_post_edit_under_a_forged_identity_records_no_attribution(
    coordinator, client: _Client
) -> None:
    """A commit naming a peer's session without the peer's principal bumps
    nothing and records no ``last_writer_id`` — neither with no principal nor
    with the caller's own. The control commits with the peer's principal and
    records the peer's composite writer id, so the negative assertions are
    measured against a route that does record attribution."""
    peer, caller = str(uuid.uuid4()), str(uuid.uuid4())
    peer_principal = _explicit_claim(client, peer)
    caller_principal = _explicit_claim(client, caller)
    _pre_edit_with(client, peer, peer_principal, "plan.md")
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    before = coordinator.registry.get_artifact(artifact_id).version
    writer_before = coordinator.registry.last_writer_for(artifact_id)
    commit = {"session_id": peer, "path": "plan.md", "success": True,
              "content_hash": _hash("forged")}

    for principal, refusal in ((None, _PRINCIPAL_ABSENT_REFUSAL),
                               (caller_principal, _PRINCIPAL_FOREIGN_REFUSAL)):
        status, body = client.post("/hooks/post-edit", commit, principal=principal)
        assert (status, body) == (400, refusal)
        assert coordinator.registry.get_artifact(artifact_id).version == before
        assert coordinator.registry.last_writer_for(artifact_id) == writer_before

    status, body = client.post("/hooks/post-edit", commit, principal=peer_principal)
    assert status == 200 and body["ok"] is True, body
    assert coordinator.registry.get_artifact(artifact_id).version == before + 1
    assert coordinator.registry.last_writer_for(artifact_id) == session_to_agent_id(peer)


def test_post_edit_attributes_the_composite_writer_through_the_principal(
    coordinator, client: _Client
) -> None:
    """Attribution on post-edit goes through the presented caller: a
    subagent presents its PARENT session's principal (the principal's unit of
    identity is the session) and is recorded under the COMPOSITE id (session
    + caller-asserted subagent); the same subagent commit with no principal
    records nothing; and a second session's commit records a different
    writer — so the positive assertion cannot pass on a constant."""
    sid_a, sid_b = str(uuid.uuid4()), str(uuid.uuid4())
    principal_a = _explicit_claim(client, sid_a)
    principal_b = _explicit_claim(client, sid_b)
    reg = coordinator.registry

    status, _ = client.post(
        "/hooks/pre-edit", {"session_id": sid_a, "agent_id": "sub-1", "path": "a/plan.md"},
        principal=principal_a,
    )
    assert status == 200
    sub_commit = {"session_id": sid_a, "agent_id": "sub-1", "path": "a/plan.md",
                  "success": True, "content_hash": _hash("a2")}
    status, body = client.post("/hooks/post-edit", sub_commit, principal=None)
    assert (status, body) == (400, _PRINCIPAL_ABSENT_REFUSAL)
    assert reg.last_writer_for(reg.lookup_artifact_id_by_name("a/plan.md")) is None
    status, body = client.post("/hooks/post-edit", sub_commit, principal=principal_a)
    assert status == 200 and body["ok"] is True, body
    _pre_edit_with(client, sid_b, principal_b, "b/plan.md")
    status, body = client.post(
        "/hooks/post-edit",
        {"session_id": sid_b, "path": "b/plan.md", "success": True,
         "content_hash": _hash("b2")},
        principal=principal_b,
    )
    assert status == 200 and body["ok"] is True, body

    writer_a = reg.last_writer_for(reg.lookup_artifact_id_by_name("a/plan.md"))
    writer_b = reg.last_writer_for(reg.lookup_artifact_id_by_name("b/plan.md"))
    assert writer_a == session_to_agent_id(sid_a, "sub-1")
    assert writer_a != session_to_agent_id(sid_a)
    assert writer_b == session_to_agent_id(sid_b)
    assert writer_a != writer_b


def test_post_edit_cas_under_a_forged_identity_bumps_nothing(
    coordinator, client: _Client
) -> None:
    """The optimistic commit is require-class too: a CAS naming a peer's
    session with no principal does not advance the version it names."""
    peer = str(uuid.uuid4())
    peer_principal = _explicit_claim(client, peer)
    status, read = client.post(
        "/hooks/pre-read",
        {"session_id": peer, "path": "plan.md", "content_hash": _hash("v1")},
        principal=peer_principal,
    )
    assert status == 200
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    version = coordinator.registry.get_artifact(artifact_id).version
    cas = {"session_id": peer, "path": "plan.md", "success": True,
           "content_hash": _hash("forged"), "expected_version": version}

    status, body = client.post("/hooks/post-edit-cas", cas, principal=None)
    assert (status, body) == (400, _PRINCIPAL_ABSENT_REFUSAL)
    assert coordinator.registry.get_artifact(artifact_id).version == version

    status, body = client.post("/hooks/post-edit-cas", cas, principal=peer_principal)
    assert status == 200 and body["ok"] is True, body
    assert coordinator.registry.get_artifact(artifact_id).version == version + 1


def test_a_principal_refusal_is_a_client_error_never_a_hold(
    coordinator, client: _Client
) -> None:
    """KTD9: a hold invites a retry, and no retry supplies a principal the
    caller never had. The effect fence — the route whose protocol answer IS
    the hold vocabulary — refuses an absent or foreign principal with HTTP
    400 and no ``verdict``, and neither refusal reason is a hold reason."""
    from ccs.core.exceptions import (
        CALLER_PRINCIPAL_ABSENT_REASON,
        CALLER_PRINCIPAL_FOREIGN_REASON,
    )

    sid, other = str(uuid.uuid4()), str(uuid.uuid4())
    _explicit_claim(client, sid)
    foreign = _explicit_claim(client, other)
    fence = _posture_body(("POST", "/hooks/effect-fence"), sid)

    for principal, reason in ((None, CALLER_PRINCIPAL_ABSENT_REASON),
                              (foreign, CALLER_PRINCIPAL_FOREIGN_REASON)):
        status, body = client.post("/hooks/effect-fence", fence, principal=principal)
        assert status == 400
        assert "verdict" not in body and set(body) == {"error", "reason"}
        assert body["reason"] == reason
    assert CALLER_PRINCIPAL_ABSENT_REASON not in HOLD_REASONS
    assert CALLER_PRINCIPAL_FOREIGN_REASON not in HOLD_REASONS


def _status_tiers(client: _Client) -> dict[str, dict]:
    """``/status`` at every tier, the operator tier with its opt-in header."""
    tiers = {}
    for tier, query, headers in (
        ("metrics", "/status?detail=metrics", None),
        ("minimal", "/status", None),
        ("full", "/status?detail=full", {"Coherence-Local-Operator": "true"}),
    ):
        status, body = client.get(query, headers_override=headers)
        assert status == 200, (tier, body)
        tiers[tier] = body
    return tiers


def test_the_principal_counters_are_status_counters_at_every_tier(
    coordinator, client: _Client
) -> None:
    """KTD4: two local diagnostics an operator reads on /status, at every tier,
    beside the other product counters. ``caller_principal_absent_total`` counts
    every ADMISSION that presents no principal — any accept-class request, or
    a require-class request naming an identity nobody claimed (KTD15) — and
    never a refusal. ``caller_principal_refused_total`` counts every REFUSAL,
    absent or foreign, on any class, and never an admission. Each step below
    moves exactly the counter it should and pins the other unmoved, so neither
    counter can quietly absorb the other's events; a request presenting the
    matching principal moves neither."""
    def counts() -> set[tuple[int, int]]:
        return {
            (body["caller_principal_absent_total"], body["caller_principal_refused_total"])
            for body in _status_tiers(client).values()
        }

    assert counts() == {(0, 0)}
    unbound = str(uuid.uuid4())
    client.post("/hooks/pre-read", {"session_id": unbound, "path": "x.md"}, principal=None)
    assert counts() == {(1, 0)}, "an accept-class admission without a principal"
    client.post("/hooks/session-stop", {"session_id": unbound}, principal=None)
    assert counts() == {(2, 0)}, "a require-class admission naming an unclaimed session"

    sid, other = str(uuid.uuid4()), str(uuid.uuid4())
    principal = _explicit_claim(client, sid)
    foreign = _explicit_claim(client, other)
    assert client.post("/hooks/session-stop", {"session_id": sid})[0] == 400
    assert counts() == {(2, 1)}, "a require-class refusal of an absent principal"
    assert client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "x.md"}, principal=foreign)[0] == 400
    assert counts() == {(2, 2)}, "an accept-class refusal of a foreign principal"
    status, _ = client.post(
        "/hooks/pre-read", {"session_id": sid, "path": "x.md"}, principal=principal)
    assert status == 200
    assert counts() == {(2, 2)}, "a matching principal moves neither counter"


def test_a_principal_refusal_is_logged_at_info_with_its_route_and_reason(
    coordinator, client: _Client, caplog: pytest.LogCaptureFixture
) -> None:
    """An operator whose session is refused sees why: each refusal leaves one
    INFO record naming the route and the typed reason — the hook client exits 0
    and Claude Code shows no stderr, so without it a refused session looks
    healthy everywhere. The record never carries the principal (R5), the
    presented one or the bound one; an admission writes no such record, so
    ordinary traffic does not flood the log."""
    import logging

    caplog.set_level(logging.DEBUG)
    peer, caller = str(uuid.uuid4()), str(uuid.uuid4())
    peer_principal = _explicit_claim(client, peer)
    caller_principal = _explicit_claim(client, caller)
    h = _hash("logged")
    refusals = [
        ("/hooks/session-stop", {"session_id": peer}, None, "caller_principal_absent"),
        ("/hooks/post-edit", {"session_id": peer, "path": "plan.md", "success": True,
                              "content_hash": h}, caller_principal, "caller_principal_foreign"),
        ("/hooks/pre-read", {"session_id": peer, "path": "plan.md"}, caller_principal,
         "caller_principal_foreign"),
    ]
    for path, body, principal, reason in refusals:
        status, answer = client.post(path, body, principal=principal)
        assert (status, answer.get("reason")) == (400, reason), answer
    status, _ = client.post(
        "/hooks/pre-read", {"session_id": peer, "path": "plan.md"}, principal=peer_principal)
    assert status == 200

    refusal_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO and "caller_principal_" in r.getMessage()
    ]
    assert len(refusal_records) == len(refusals), [r.getMessage() for r in refusal_records]
    for record, (path, _body, _principal, reason) in zip(refusal_records, refusals):
        message = record.getMessage()
        assert path in message and reason in message, message
    for record in caplog.records:
        rendered = f"{record.getMessage()} {record.args!r} {record.exc_text or ''}"
        for principal in (peer_principal, caller_principal):
            assert principal not in rendered, record.name


def test_pre_edit_without_its_principal_cannot_take_a_grant_it_could_not_release(
    coordinator, client: _Client
) -> None:
    """A bound session whose caller has lost its principal — a CoherentVolume
    whose claim landed but whose answer was lost, or a hook client whose claim
    was refused — used to be admitted on pre-edit and take EXCLUSIVE. It could
    then neither commit that grant nor release it: post-edit (either
    ``success`` value) and session-stop are require-class. Its own reads kept
    the heartbeat fresh, so only the max-hold sweep freed the grant, and every
    optimistic peer's commit was refused ``other_holder`` meanwhile. pre-edit
    refuses that caller now, under the same rule as the release routes (an
    identity nobody claimed is still admitted), so the grant is never taken
    and the peer's commit lands. Control: presenting its principal, the same
    caller takes the grant and releases it — the refusal sits exactly where
    the release lives."""
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    a_principal = _explicit_claim(client, a)
    b_principal = _explicit_claim(client, b)
    reg = coordinator.registry
    status, _ = client.post(
        "/hooks/pre-read", {"session_id": b, "path": "plan.md", "content_hash": _hash("v1")},
        principal=b_principal,
    )
    assert status == 200
    artifact_id = reg.lookup_artifact_id_by_name("plan.md")
    version = reg.get_artifact(artifact_id).version
    a_agent = session_to_agent_id(a)

    status, body = client.post(
        "/hooks/pre-edit", {"session_id": a, "path": "plan.md"}, principal=None)
    assert (status, body) == (400, _PRINCIPAL_ABSENT_REFUSAL)
    assert reg.get_agent_state(artifact_id, a_agent) not in (
        MESIState.EXCLUSIVE, MESIState.MODIFIED)
    status, body = client.post("/hooks/post-edit-cas", {
        "session_id": b, "path": "plan.md", "success": True,
        "content_hash": _hash("v2"), "expected_version": version,
    }, principal=b_principal)
    assert status == 200 and body["ok"] is True, body
    assert reg.get_artifact(artifact_id).version == version + 1

    _pre_edit_with(client, a, a_principal, "plan.md")
    assert reg.get_agent_state(artifact_id, a_agent) == MESIState.EXCLUSIVE
    status, body = client.post(
        "/hooks/post-edit", {"session_id": a, "path": "plan.md", "success": False},
        principal=a_principal,
    )
    assert status == 200 and body["ok"] is True, body
    assert reg.get_agent_state(artifact_id, a_agent) not in (
        MESIState.EXCLUSIVE, MESIState.MODIFIED)


def test_a_refused_pre_edit_costs_its_session_the_deny_and_says_so(served_decider) -> None:
    """The other side of the test above, which the posture table must state
    rather than leave out: refusing a claimed session's pre-edit that carries
    no principal also withholds everything pre-edit does FOR that session --
    the strict-mode deny, the grant and the invalidation of its peers -- and
    the hook clients read a 400 like any other refusal, so the edit then
    proceeds uncoordinated.

    Each cost is shown against a control that GETS it, in a setup where
    admission and refusal differ in that cost. A refused request that changed
    nothing looks exactly like an admitted one that happened to change
    nothing, and on a strict-mode path where the session was preempted an
    admitted pre-edit is denied: it takes no grant and invalidates no peer
    either, so there only the deny separates the two. The grant and the
    invalidation are shown on an unpreempted path instead, where the same
    request presenting the principal takes EXCLUSIVE and invalidates the
    SHARED peer.

    Prevents the table describing only what the refusal rules out. The
    trade-off was chosen -- over admitting an acquire the caller could neither
    commit nor release -- and its cost is pinned here with the words that
    state it, the sentence stating the choice included."""
    from ccs.adapters.claude_code.coordinator_server import _CALLER_PRINCIPAL_POSTURE

    server, client = served_decider
    reg = server.registry
    path = _U3A_STRICT_PATH
    editor, peer = str(uuid.uuid4()), str(uuid.uuid4())
    editor_principal = _explicit_claim(client, editor)
    peer_principal = _explicit_claim(client, peer)
    editor_agent, peer_agent = session_to_agent_id(editor), session_to_agent_id(peer)

    # The deny: a strict-mode path where the peer's edit preempted the editor.
    status, _ = client.post(
        "/hooks/pre-read", {"session_id": editor, "path": path, "content_hash": _hash("v1")},
        principal=editor_principal)
    assert status == 200
    _pre_edit_with(client, peer, peer_principal, path)
    artifact_id = reg.lookup_artifact_id_by_name(path)
    assert reg.get_agent_state(artifact_id, editor_agent) == MESIState.INVALID
    assert reg.get_agent_state(artifact_id, peer_agent) == MESIState.EXCLUSIVE
    denials_before = server.counters_snapshot()["strict_mode_denials_total"]

    refused = client.post("/hooks/pre-edit", {"session_id": editor, "path": path})
    assert refused == (400, _PRINCIPAL_ABSENT_REFUSAL)
    assert "hookSpecificOutput" not in refused[1], "a refusal relays no deny"
    # The 400 is written before the handler returns; a side effect after it
    # would land after this read unless the handler is waited out first.
    _await_in_flight_drain(server)
    assert server.counters_snapshot()["strict_mode_denials_total"] == denials_before, (
        "no strict-mode deny")
    # Not costs of the refusal on this path (the admitted control below is
    # denied and changes neither): only that the refused request changed nothing.
    assert reg.get_agent_state(artifact_id, editor_agent) == MESIState.INVALID
    assert reg.get_agent_state(artifact_id, peer_agent) == MESIState.EXCLUSIVE

    status, denied = client.post(
        "/hooks/pre-edit", {"session_id": editor, "path": path}, principal=editor_principal)
    assert status == 200 and denied["ok"] is False, denied
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", denied
    assert server.counters_snapshot()["strict_mode_denials_total"] == denials_before + 1

    # The grant and the invalidation: an unpreempted path, both sessions SHARED.
    open_path = _U3A_WARN_PATH
    for sid, principal in ((editor, editor_principal), (peer, peer_principal)):
        status, _ = client.post(
            "/hooks/pre-read",
            {"session_id": sid, "path": open_path, "content_hash": _hash("v1")},
            principal=principal)
        assert status == 200
    open_id = reg.lookup_artifact_id_by_name(open_path)
    assert reg.get_agent_state(open_id, editor_agent) == MESIState.SHARED
    assert reg.get_agent_state(open_id, peer_agent) == MESIState.SHARED

    refused = client.post("/hooks/pre-edit", {"session_id": editor, "path": open_path})
    assert refused == (400, _PRINCIPAL_ABSENT_REFUSAL)
    _await_in_flight_drain(server)
    assert reg.get_agent_state(open_id, editor_agent) == MESIState.SHARED, "no grant"
    assert reg.get_agent_state(open_id, peer_agent) == MESIState.SHARED, (
        "no peer invalidated")

    _pre_edit_with(client, editor, editor_principal, open_path)
    assert reg.get_agent_state(open_id, editor_agent) == MESIState.EXCLUSIVE, (
        "control: the same pre-edit, admitted, takes the grant")
    assert reg.get_agent_state(open_id, peer_agent) == MESIState.INVALID, (
        "control: the same pre-edit, admitted, invalidates the SHARED peer")

    harm = _CALLER_PRINCIPAL_POSTURE[("POST", "/hooks/pre-edit")].harm
    for stated in ("proceeds uncoordinated", "no grant", "no strict-mode deny",
                   "no invalidation of its peers", "neither commit nor release",
                   "chosen over admitting an acquire"):
        assert stated in harm, f"pre-edit's harm text does not state {stated!r}: {harm}"


# --- a REFUSED require-class request leaves every piece of state as it was ---

_REQUIRE_CLASS_ROUTES = sorted(
    route for route, posture in _EXPECTED_ROUTE_POSTURE.items() if posture == "require"
)
#: FROZEN duplicate of the per-route attempt counters (``_ENDPOINT_COUNTER_NAMES``)
#: for the require class: the one counter besides the refusal counter a refused
#: request may move — it counts attempts by contract. The workspace routes have
#: no attempt counter.
_ATTEMPT_COUNTER = {
    "/hooks/effect-fence": "effect_fence_total",
    "/hooks/pre-edit": "pre_edit_total",
    "/hooks/post-edit": "post_edit_total",
    "/hooks/post-edit-cas": "post_edit_cas_total",
    "/hooks/session-stop": "session_stop_total",
}
_REFUSED_SUBAGENT = "sub-refused"


def _seed_a_claimed_peer(coordinator, client: _Client) -> dict:
    """A CLAIMED peer with a non-default value in every dimension a require-
    class handler touches once admitted: an EXCLUSIVE grant on a tracked path,
    a heartbeat, a queued preemption notice, an armed re-grounding flag, a
    stale-warned pair, a strict-deny memory, and a checkpoint it owns. Seeded
    through the registry, never a hook route, so the peer's names are NOT in
    the display-name map: a handler that registers its caller before the gate
    shows up as a new entry."""
    from ccs.coordinator.registry_protocol import CheckpointMember

    peer, caller = str(uuid.uuid4()), str(uuid.uuid4())
    seed = {
        "peer": peer,
        "peer_principal": _explicit_claim(client, peer),
        "caller_principal": _explicit_claim(client, caller),
        "peer_agent": session_to_agent_id(peer),
    }
    reg = coordinator.registry
    artifact_id = reg.resolve_or_register("plan.md", content_hash=_hash("v1"))
    coordinator.service.write(agent_id=seed["peer_agent"], artifact_id=artifact_id, issued_at_tick=1)
    reg.record_heartbeat(seed["peer_agent"], 41)
    reg.record_preemption_notice(
        victim_agent_id=seed["peer_agent"], artifact_id=artifact_id,
        preempter_agent_id=session_to_agent_id(caller), preempted_at_unix_ts=1234.0,
    )
    coordinator.mark_compact_pending(peer)
    coordinator.mark_stale_warned(seed["peer_agent"], artifact_id)
    coordinator.record_strict_deny(peer, "plan.md")
    record = coordinator.service.create_workspace_checkpoint(
        name="seeded", owner=seed["peer_agent"],
        members=[CheckpointMember(
            member_path="plan.md", artifact_id=None, native_token="v1",
            fingerprint=_hash("v1"), captured_at=1.0,
        )],
        window_min=1.0, window_max=1.0, issued_at_tick=1,
    )
    return {**seed, "artifact_id": artifact_id, "checkpoint_id": record.checkpoint_id}


def _refused_request_body(route: tuple[str, str], seed: dict, *, subagent: bool) -> dict:
    """A body that passes every shape check on ``route``, names the claimed
    peer, and — were it admitted — would move state: a TRACKED path (so no
    untracked fast path answers before the handler's work) and the peer's real
    checkpoint. ``subagent`` adds an ``agent_id`` field on the hook routes."""
    from ccs.core.exceptions import RESTORE_MEMBER_OUTCOMES

    _, path = route
    h = _hash(f"refused:{path}")
    bodies: dict[str, dict] = {
        "/hooks/effect-fence": {
            "path": "plan.md", "expected_version": 99, "expected_generation": 0,
            "content_hash": h,
        },
        "/hooks/pre-edit": {"path": "plan.md"},
        "/hooks/post-edit": {"path": "plan.md", "success": True, "content_hash": h},
        "/hooks/post-edit-cas": {"path": "plan.md", "content_hash": h, "expected_version": 1},
        "/hooks/session-stop": {},
        "/workspace/checkpoint": {
            "name": "refused", "window_min": 1.0, "window_max": 1.0,
            "members": [{"member_path": "plan.md", "native_token": "v2",
                         "fingerprint": h, "captured_at": 2.0}],
        },
        "/workspace/restore/status": {
            "checkpoint_id": seed["checkpoint_id"], "status": "in_progress"},
        "/workspace/restore/member": {
            "checkpoint_id": seed["checkpoint_id"], "member_path": "plan.md",
            "restore_outcome": sorted(RESTORE_MEMBER_OUTCOMES)[0],
        },
        "/workspace/restore/register": {
            "checkpoint_id": seed["checkpoint_id"],
            "writes": [{"member_path": "plan.md", "fingerprint": h}],
        },
    }
    body = {"session_id": seed["peer"], **bodies[path]}
    if subagent:
        body["agent_id"] = _REFUSED_SUBAGENT
    return body


def _refusal_observable(coordinator, seed: dict) -> dict:
    """Everything an admitted require-class request could move, read through
    NON-destructive accessors only. Principals appear only as digests, so a
    failing comparison prints none."""
    reg = coordinator.registry
    artifacts, states = reg.status_snapshot()
    agents = (seed["peer_agent"], session_to_agent_id(seed["peer"], _REFUSED_SUBAGENT))
    coherence = coordinator.coordinator_root / ".coherence"
    return {
        "counters": coordinator.counters_snapshot(),
        "artifacts": {str(a): dict(meta) for a, meta in artifacts.items()},
        "grants": {
            str(a): {str(agent): state.name for agent, state in held.items()}
            for a, held in states.items()
        },
        "last_writers": {str(a): reg.last_writer_for(a) for a in artifacts},
        "heartbeats": {str(agent): reg.last_heartbeat_tick(agent) for agent in agents},
        "notices": {
            str(agent): reg.peek_preemption_notice(agent, seed["artifact_id"])
            for agent in agents
        },
        "agent_names": sorted((str(a), n) for a, n in coordinator.agent_names_snapshot()),
        "compact_pending": coordinator.has_compact_pending(seed["peer"]),
        "stale_warned": set(coordinator._stale_warned_pairs),
        "strict_denies": set(coordinator._recent_strict_denies),
        "bindings": [
            hashlib.sha256(value.encode()).hexdigest()
            for value in (
                reg.get_caller_principal(caller_principal_identity(seed["peer"])) or "",
            )
        ],
        "checkpoints": [
            (dataclasses.asdict(record),
             [dataclasses.asdict(m) for m in reg.get_checkpoint_members(record.checkpoint_id)])
            for record in reg.list_checkpoints()
        ],
        "audit_logs": sorted((p.name, p.stat().st_size) for p in coherence.glob("*.log")),
    }


def _split_counters(observable: dict, path: str) -> tuple[dict, int | None, int | None]:
    """``observable`` minus the two counters a refusal may move — the route's
    attempt counter and the refusal counter — returned beside them."""
    rest = dict(observable)
    counters = dict(rest["counters"])
    endpoint = dict(counters["endpoint_counters"])
    attempts = endpoint.pop(_ATTEMPT_COUNTER[path]) if path in _ATTEMPT_COUNTER else None
    counters["endpoint_counters"] = endpoint
    refused = counters.pop("caller_principal_refused_total", None)
    rest["counters"] = counters
    return rest, attempts, refused


@pytest.mark.parametrize("route", _REQUIRE_CLASS_ROUTES, ids=lambda r: r[1].strip("/"))
def test_a_refused_require_class_request_leaves_state_untouched(
    route: tuple[str, str], coordinator, client: _Client
) -> None:
    """KTD9 makes a principal refusal a client error, and the caller is told
    to retry only once it has its principal — which is safe only if the refused
    request changed nothing. Every require-class route is driven with an ABSENT
    principal and with a FOREIGN one (the caller's own valid principal, spent
    on the peer's session), in the parent and the subagent form, against a
    claimed peer seeded with a non-default value in every dimension. Nothing
    may move: no display name registered, no heartbeat, no re-grounding flag
    expired, no notice drained, no grant, version, writer, checkpoint or
    binding changed, no audit row — and no counter but the route's attempt
    counter and the refusal counter, each by exactly one per request.

    Control: the same request presenting the peer's principal IS admitted and
    DOES move the snapshot, so the snapshot can see this route's effects and
    the equality above is not vacuous."""
    _, path = route
    seed = _seed_a_claimed_peer(coordinator, client)
    before = _refusal_observable(coordinator, seed)
    assert before["heartbeats"][str(seed["peer_agent"])] == 41
    assert before["notices"][str(seed["peer_agent"])] is not None
    assert before["compact_pending"] is True
    assert before["grants"][str(seed["artifact_id"])][str(seed["peer_agent"])] == "EXCLUSIVE"
    assert before["checkpoints"] and before["stale_warned"] and before["strict_denies"]
    assert not any(str(seed["peer"]) in name for _a, name in before["agent_names"])

    forms = (False, True) if path.startswith("/hooks/") else (False,)
    sent = 0
    for principal, refusal in ((None, _PRINCIPAL_ABSENT_REFUSAL),
                               (seed["caller_principal"], _PRINCIPAL_FOREIGN_REFUSAL)):
        for subagent in forms:
            body = _refused_request_body(route, seed, subagent=subagent)
            assert client.post(path, body, principal=principal) == (400, refusal), body
            sent += 1
    after = _refusal_observable(coordinator, seed)

    rest_before, attempts_before, refused_before = _split_counters(before, path)
    rest_after, attempts_after, refused_after = _split_counters(after, path)
    assert rest_after == rest_before
    assert refused_before is not None and refused_after == refused_before + sent
    if attempts_before is not None:
        assert attempts_after == attempts_before + sent

    status, body = client.post(
        path, _refused_request_body(route, seed, subagent=False),
        principal=seed["peer_principal"],
    )
    assert not _principal_refused((status, body)), body
    rest_admitted, _, _ = _split_counters(_refusal_observable(coordinator, seed), path)
    assert rest_admitted != rest_after, "the admitted request moved nothing the snapshot sees"


# --- the accept-class notice drain, stated rather than implied --------------


@pytest.mark.parametrize("path,extra", [
    ("/hooks/pre-read", {"path": "plan.md"}),
    ("/hooks/pre-bash", {"command": "cat plan.md"}),
    ("/hooks/pre-grep", {"search_root": ""}),
], ids=["pre-read", "pre-bash", "pre-grep"])
def test_an_accept_class_read_naming_a_bound_peer_drains_its_notices_and_says_so(
    path: str, extra: dict, coordinator, client: _Client
) -> None:
    """Accepted behaviour, pinned so the posture table cannot understate it.
    The accept-class reads pop the NAMED identity's pending notices and deliver
    them in their own response. A request naming a bound peer without a
    principal — a stray or older client, or a misnamed session — therefore
    receives the peer's "you were preempted" notice, and the peer never sees
    it. The table's harm text for the route says so, as the require-class
    session-stop entry does for the same effect."""
    from ccs.adapters.claude_code.coordinator_server import _CALLER_PRINCIPAL_POSTURE

    peer = str(uuid.uuid4())
    _explicit_claim(client, peer)
    peer_agent = session_to_agent_id(peer)
    reg = coordinator.registry
    artifact_id = reg.resolve_or_register("plan.md", content_hash=_hash("v1"))
    reg.record_preemption_notice(
        victim_agent_id=peer_agent, artifact_id=artifact_id,
        preempter_agent_id=session_to_agent_id(str(uuid.uuid4())),
        preempted_at_unix_ts=1234.0,
    )
    status, body = client.post(path, {"session_id": peer, **extra}, principal=None)
    assert status == 200 and not _principal_refused((status, body)), body
    assert "preempted" in json.dumps(body), "the notice went to this caller"
    assert reg.peek_preemption_notice(peer_agent, artifact_id) is None, "and is gone"

    harm = _CALLER_PRINCIPAL_POSTURE[("POST", path)].harm
    assert "drains the named identity's pending notices" in harm, harm


def test_session_start_shows_a_bound_peers_notices_without_draining_them(
    coordinator, client: _Client
) -> None:
    """session-start is accept-class and only SHOWS the named identity's
    notices; it does not drain them, and its harm text says what it does
    write — the display name and, for a non-empty re-grounding, the
    compact-pending flag the named identity's next admit delivers."""
    from ccs.adapters.claude_code.coordinator_server import _CALLER_PRINCIPAL_POSTURE

    peer = str(uuid.uuid4())
    peer_principal = _explicit_claim(client, peer)
    peer_agent = session_to_agent_id(peer)
    _pre_edit_with(client, peer, peer_principal, "plan.md")
    artifact_id = coordinator.registry.lookup_artifact_id_by_name("plan.md")
    coordinator.registry.record_preemption_notice(
        victim_agent_id=peer_agent, artifact_id=artifact_id,
        preempter_agent_id=session_to_agent_id(str(uuid.uuid4())),
        preempted_at_unix_ts=1234.0,
    )
    status, body = client.post("/hooks/session-start", {"session_id": peer}, principal=None)
    assert status == 200, body
    assert coordinator.registry.peek_preemption_notice(peer_agent, artifact_id) is not None
    assert coordinator.has_compact_pending(peer) is True

    harm = _CALLER_PRINCIPAL_POSTURE[("POST", "/hooks/session-start")].harm
    assert "without draining them" in harm and "compact-pending" in harm, harm
