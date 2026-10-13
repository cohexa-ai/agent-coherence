# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""HTTP route tests for POST /session/commit_all (SB-18 atomic batch publish).

The wire generalization of ``/session/commit``: all-or-nothing over a write-set,
same R9 boundary lock (server sources every comparand from the pinned cut), same
content-free audit. Spins up a real coordinator + urllib client so the dispatch
seam (verify_bearer / verify_host / migration-drain) is exercised end-to-end.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    _MIGRATION_REJECTED_ROUTES,
    _ROUTES,
    CoordinatorHTTPServer,
)
from ccs.adapters.claude_code.session_audit_log import (
    _resolve_session_audit_log_path,
)

_TEST_SESSION_NS = uuid.UUID("33333333-3333-4333-8333-333333333333")


def _sid(label: str) -> str:
    return str(uuid.uuid5(_TEST_SESSION_NS, f"commit-all-route:{label}"))


class _Client:
    """Tiny urllib client returning (status, body_dict)."""

    def __init__(self, host: str, port: int, secret: str) -> None:
        self.base = f"http://{host}:{port}"
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        }

    def post(
        self, path: str, body: dict, *, headers_override: Optional[dict] = None
    ) -> tuple[int, dict]:
        url = self.base + path
        headers = dict(self.headers)
        if headers_override:
            headers.update(headers_override)
        req = urlrequest.Request(
            url, data=json.dumps(body).encode("utf-8"), method="POST", headers=headers
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urlerror.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")


@pytest.fixture
def coordinator(tmp_path: Path):
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="test-instance")
    server.serve_in_thread()
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def client(coordinator) -> _Client:
    secret = load_secret(coordinator.coordinator_root)
    assert secret is not None
    return _Client("127.0.0.1", coordinator.port, secret)


def _begin(client: _Client, sid: str, read_set: list[str]) -> dict:
    status, body = client.post(
        "/session/begin", {"session_id": sid, "read_set": read_set}
    )
    assert status == 200, body
    return body


# ----------------------------------------------------------------------
# Registration — rides the central _ROUTES seam + drains on migration
# ----------------------------------------------------------------------


def test_commit_all_route_registered_in_central_routes() -> None:
    assert ("POST", "/session/commit_all") in _ROUTES


def test_commit_all_route_rejected_while_draining() -> None:
    # A batch OCC write bumps versions the imminent shutdown would strand — it
    # must drain like /session/commit, not slip through mid-migration.
    assert ("POST", "/session/commit_all") in _MIGRATION_REJECTED_ROUTES


# ----------------------------------------------------------------------
# Happy path — atomic batch WIN, then all-or-nothing HELD at the same pin
# ----------------------------------------------------------------------


def test_happy_path_batch_commit_bumps_every_member(client: _Client) -> None:
    sid = _sid("happy")
    begin = _begin(client, sid, ["a.md", "b.md"])
    token = begin["session_token"]
    assert begin["cut"] == {"a.md": 1, "b.md": 1}

    status, out = client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": token,
            "writes": [
                {"path": "a.md", "content": "a-v2"},
                {"path": "b.md", "content": "b-v2"},
            ],
        },
    )
    assert status == 200, out
    assert out["ok"] is True
    assert out["versions"] == {"a.md": 2, "b.md": 2}

    # A SECOND batch at the same pin is HELD on BOTH members (R11), zero mutation.
    status, again = client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": token,
            "writes": [
                {"path": "a.md", "content": "a-v3"},
                {"path": "b.md", "content": "b-v3"},
            ],
        },
    )
    assert status == 200
    assert again["ok"] is False
    assert again["reason"] == "conflict"
    assert set(again["per_artifact"]) == {"a.md", "b.md"}
    assert again["per_artifact"]["a.md"]["reason"] == "version_mismatch"
    assert again["per_artifact"]["a.md"]["current_version"] == 2


def test_one_drifted_member_holds_whole_batch(client: _Client) -> None:
    """A peer session advances ONE member under the batch's feet -> the whole
    commit_all is HELD and the passing member is NOT mutated (all-or-nothing)."""
    sid = _sid("drift")
    begin = _begin(client, sid, ["x.md", "y.md"])
    token = begin["session_token"]

    # A different session advances y.md to v2, so the first session's pin is stale.
    other = _sid("drift-peer")
    peer = _begin(client, other, ["y.md"])
    status, peer_commit = client.post(
        "/session/commit",
        {"session_id": other, "session_token": peer["session_token"],
         "path": "y.md", "content": "y-peer"},
    )
    assert status == 200 and peer_commit["ok"] is True

    status, out = client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": token,
            "writes": [
                {"path": "x.md", "content": "x-v2"},
                {"path": "y.md", "content": "y-v2"},
            ],
        },
    )
    assert status == 200, out
    assert out["ok"] is False
    assert out["reason"] == "conflict"
    assert set(out["per_artifact"]) == {"y.md"}

    # ALL-OR-NOTHING: x.md was NOT bumped. A fresh session re-reads it at v1.
    check = _begin(client, _sid("drift-check"), ["x.md"])
    assert check["cut"]["x.md"] == 1


# ----------------------------------------------------------------------
# R9 boundary lock + fail-closed
# ----------------------------------------------------------------------


def test_member_not_in_cut_rejects_whole_batch(client: _Client) -> None:
    """A member absent from the pinned cut refuses the WHOLE batch; the in-cut
    member is left untouched."""
    # Seed b.md so it EXISTS but is not pinned by the committing session.
    _begin(client, _sid("seed-b"), ["b.md"])
    sid = _sid("not-in-cut")
    begin = _begin(client, sid, ["a.md"])  # only a.md pinned
    token = begin["session_token"]

    status, out = client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": token,
            "writes": [
                {"path": "a.md", "content": "a-v2"},
                {"path": "b.md", "content": "b-v2"},
            ],
        },
    )
    assert status == 200, out
    assert out["ok"] is False
    assert out["reason"] == "artifact_not_in_cut"
    # a.md untouched — a later fresh read still pins v1.
    check = _begin(client, _sid("not-in-cut-check"), ["a.md"])
    assert check["cut"]["a.md"] == 1


def test_forged_token_fails_closed(client: _Client) -> None:
    sid = _sid("forged")
    _begin(client, sid, ["z.md"])  # seed the path so failure is about the TOKEN
    status, out = client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": "totally-not-a-real-server-minted-token",
            "writes": [{"path": "z.md", "content": "x"}],
        },
    )
    assert status == 200
    assert out["ok"] is False
    # Never served as a live-HEAD write.
    assert out["reason"] != "" and out.get("versions") is None


def test_client_supplied_expected_version_ignored(client: _Client) -> None:
    """A smuggled per-member expected_version MUST NOT be honored — the server
    sources every comparand from the pinned cut."""
    sid = _sid("forge-ev")
    begin = _begin(client, sid, ["t.md"])
    token = begin["session_token"]
    status, out = client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": token,
            "writes": [{"path": "t.md", "content": "x", "expected_version": 999}],
        },
    )
    assert status == 200, out
    assert out["ok"] is True  # forged 999 ignored; real pinned v1 used as comparand
    assert out["versions"] == {"t.md": 2}


# ----------------------------------------------------------------------
# Wire validation (400 before the service)
# ----------------------------------------------------------------------


def test_empty_write_set_is_400(client: _Client) -> None:
    sid = _sid("empty")
    begin = _begin(client, sid, ["e.md"])
    status, body = client.post(
        "/session/commit_all",
        {"session_id": sid, "session_token": begin["session_token"], "writes": []},
    )
    assert status == 400
    assert "writes" in body["error"]


def test_duplicate_paths_are_400(client: _Client) -> None:
    sid = _sid("dup")
    begin = _begin(client, sid, ["d.md"])
    status, body = client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": begin["session_token"],
            "writes": [
                {"path": "d.md", "content": "one"},
                {"path": "d.md", "content": "two"},
            ],
        },
    )
    assert status == 400
    assert "duplicate" in body["error"]


def test_over_cap_write_set_is_400(client: _Client) -> None:
    sid = _sid("cap")
    begin = _begin(client, sid, ["c.md"])
    writes = [{"path": f"f{i}.md", "content": "x"} for i in range(65)]  # > 64
    status, body = client.post(
        "/session/commit_all",
        {"session_id": sid, "session_token": begin["session_token"], "writes": writes},
    )
    assert status == 400
    assert "writes" in body["error"]


# ----------------------------------------------------------------------
# Audit (R10a) — a batch WIN emits one content-free session_commit per member
# ----------------------------------------------------------------------


def test_batch_win_emits_per_member_content_free_audit(
    coordinator, client: _Client
) -> None:
    sid = _sid("audit")
    begin = _begin(client, sid, ["p.md", "q.md"])
    token = begin["session_token"]
    client.post(
        "/session/commit_all",
        {
            "session_id": sid,
            "session_token": token,
            "writes": [
                {"path": "p.md", "content": "p-v2"},
                {"path": "q.md", "content": "q-v2"},
            ],
        },
    )
    audit_path = _resolve_session_audit_log_path(coordinator.coordinator_root)
    records = [json.loads(line) for line in audit_path.read_text().strip().splitlines()]
    commits = [r for r in records if r["event"] == "session_commit"]
    # One audit record per batch member.
    assert len(commits) == 2
    for rec in commits:
        assert set(rec.keys()) == {
            "ts", "event", "session", "artifact", "pinned_version", "committed_version",
        }
        assert rec["pinned_version"] == 1 and rec["committed_version"] == 2
    # Content-free: neither the raw token nor the committed bytes ever logged.
    raw = audit_path.read_text()
    assert token not in raw
    assert "p-v2" not in raw and "q-v2" not in raw
