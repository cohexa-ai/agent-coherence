# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for v0.2 Unit 4 strict-mode telemetry: counters + minimal deny-site
audit-log append (KTD-V minimal + KTD-J extension).

Covers:
- Counter increments per handler deny path
- strict_mode_routed_around_via_bash detection (Read deny → Bash deny within
  STRICT_DENY_ROUTE_AROUND_WINDOW_SEC)
- /status?detail=metrics surfaces the new counters
- audit.log file mode 0o600, JSONL schema, concurrent-append atomicity,
  mode-drift handling, denials-only filter, bounded payload (no command
  bodies, no user content)
- Counters reset behavior across coordinator restarts (none — counters are
  per-process; restart is the operator-visible signal)"""

from __future__ import annotations

import hashlib
import json
import stat
import threading
import uuid
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from ccs.adapters.claude_code.audit_log import (
    _REQUIRED_MODE,
    _resolve_audit_log_path,
    append_strict_deny,
)
from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    STRICT_DENY_ROUTE_AROUND_WINDOW_SEC,
    CoordinatorHTTPServer,
)

# ----------------------------------------------------------------------
# Shared test plumbing — mirrors tests/integration/test_strict_mode.py
# ----------------------------------------------------------------------


_TEST_SESSION_NS = uuid.UUID("11111111-1111-4111-8111-111111111111")


def _sid(label: str) -> str:
    return str(uuid.uuid5(_TEST_SESSION_NS, f"strict-mode-telemetry:{label}"))


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _Client:
    def __init__(self, host: str, port: int, secret: str) -> None:
        self.base = f"http://{host}:{port}"
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        }

    def request(self, method: str, path: str, body: Optional[dict] = None) -> tuple[int, dict]:
        url = self.base + path
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        req = urlrequest.Request(
            url,
            data=data if method == "POST" else None,
            method=method,
            headers=self.headers,
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urlerror.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self.request("POST", path, body)

    def get(self, path: str) -> tuple[int, dict]:
        return self.request("GET", path)


def _write_policy(workspace: Path, *, tracked: list[str], strict: list[str]) -> None:
    coherence_dir = workspace / ".coherence"
    coherence_dir.mkdir(exist_ok=True, mode=0o700)
    if tracked:
        (coherence_dir / "tracked.yaml").write_text(
            "\n".join(f"- {p}" for p in tracked) + "\n"
        )
    if strict:
        (coherence_dir / "strict_mode.yaml").write_text(
            "\n".join(f"- {p}" for p in strict) + "\n"
        )


@pytest.fixture
def strict_workspace(tmp_path: Path) -> Path:
    _write_policy(tmp_path, tracked=["plan.md", "CLAUDE.md"], strict=["plan.md", "CLAUDE.md"])
    return tmp_path


@pytest.fixture
def coordinator(strict_workspace: Path):
    server = CoordinatorHTTPServer(strict_workspace, port=0, instance_id="unit4-test")
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


def _setup_stale_path(client: _Client, path: str) -> None:
    """Drive A reads → B reads + preempts → A is left INVALID."""
    client.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": path, "content_hash": _hash("v1")})
    client.post("/hooks/pre-read",
                {"session_id": _sid("B"), "path": path, "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit",
                {"session_id": _sid("B"), "path": path})
    client.post("/hooks/post-edit",
                {"session_id": _sid("B"), "path": path,
                 "content_hash": _hash("v2"), "success": True})


# ----------------------------------------------------------------------
# Counter increments per handler
# ----------------------------------------------------------------------


def test_pre_read_strict_deny_bumps_counter(coordinator, client: _Client) -> None:
    before = coordinator.counters_snapshot()["strict_mode_denials_total"]
    _setup_stale_path(client, "CLAUDE.md")
    # A's re-read on strict + stale → deny → +1.
    client.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")})
    after = coordinator.counters_snapshot()["strict_mode_denials_total"]
    assert after == before + 1


def test_pre_edit_strict_deny_bumps_counter(coordinator, client: _Client) -> None:
    before = coordinator.counters_snapshot()["strict_mode_denials_total"]
    _setup_stale_path(client, "plan.md")
    client.post("/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"})
    after = coordinator.counters_snapshot()["strict_mode_denials_total"]
    assert after == before + 1


def test_pre_bash_strict_deny_bumps_counter(coordinator, client: _Client) -> None:
    before = coordinator.counters_snapshot()["strict_mode_denials_total"]
    _setup_stale_path(client, "plan.md")
    client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    after = coordinator.counters_snapshot()["strict_mode_denials_total"]
    assert after == before + 1


def test_pre_grep_strict_deny_bumps_counter(coordinator, client: _Client) -> None:
    before = coordinator.counters_snapshot()["strict_mode_denials_total"]
    _setup_stale_path(client, "plan.md")
    client.post("/hooks/pre-grep", {"session_id": _sid("A"), "search_root": ""})
    after = coordinator.counters_snapshot()["strict_mode_denials_total"]
    assert after == before + 1


# ----------------------------------------------------------------------
# Route-around detection (KTD-J extension)
# ----------------------------------------------------------------------


def test_read_deny_then_bash_deny_bumps_route_around(coordinator, client: _Client) -> None:
    """KTD-J: model gets Read deny on strict-stale plan.md, routes around
    via `bash cat plan.md` within the window → route-around counter +1."""
    _setup_stale_path(client, "plan.md")
    route_around_before = coordinator.counters_snapshot()[
        "strict_mode_routed_around_via_bash_total"
    ]
    # Step 1: Read denies + records (session, path) in recent strict-denies.
    client.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("v1")})
    # Step 2: Bash cat plan.md from same session within 30s → route-around.
    client.post("/hooks/pre-bash",
                {"session_id": _sid("A"), "command": "cat plan.md"})
    route_around_after = coordinator.counters_snapshot()[
        "strict_mode_routed_around_via_bash_total"
    ]
    assert route_around_after == route_around_before + 1


def test_bash_deny_without_prior_read_does_not_bump_route_around(
    coordinator, client: _Client
) -> None:
    """Bash strict-deny WITHOUT a prior Read strict-deny on the same pair
    bumps strict_mode_denials_total but NOT route-around."""
    _setup_stale_path(client, "plan.md")
    route_around_before = coordinator.counters_snapshot()[
        "strict_mode_routed_around_via_bash_total"
    ]
    # Bash directly — no prior Read deny on (A, plan.md).
    client.post("/hooks/pre-bash",
                {"session_id": _sid("A"), "command": "cat plan.md"})
    route_around_after = coordinator.counters_snapshot()[
        "strict_mode_routed_around_via_bash_total"
    ]
    assert route_around_after == route_around_before  # No bump.


def test_grep_deny_does_not_bump_route_around(coordinator, client: _Client) -> None:
    """Grep strict-deny does NOT contribute to route_around_via_bash. The
    metric is bash-specific by the plan's KTD-J extension; Grep is a
    separate H4 surface."""
    _setup_stale_path(client, "plan.md")
    client.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("v1")})
    route_around_before = coordinator.counters_snapshot()[
        "strict_mode_routed_around_via_bash_total"
    ]
    client.post("/hooks/pre-grep", {"session_id": _sid("A"), "search_root": ""})
    route_around_after = coordinator.counters_snapshot()[
        "strict_mode_routed_around_via_bash_total"
    ]
    assert route_around_after == route_around_before


def test_route_around_window_constant_is_30_seconds() -> None:
    """Plan locks the route-around window at 30s (per Unit 4 Approach).
    Bound to the operator-facing docs (configuration.md) so tests guard
    against silent constant drift."""
    assert STRICT_DENY_ROUTE_AROUND_WINDOW_SEC == 30.0


# ----------------------------------------------------------------------
# /status surface
# ----------------------------------------------------------------------


def test_status_metrics_tier_exposes_strict_counters(coordinator, client: _Client) -> None:
    """The new counters surface in /status?detail=metrics so an operator
    scraping the metrics block sees them without enabling the full tier."""
    status, body = client.get("/status?detail=metrics")
    assert status == 200
    assert "strict_mode_denials_total" in body
    assert "strict_mode_routed_around_via_bash_total" in body
    assert "audit_log_mode_drift_total" in body


# ----------------------------------------------------------------------
# Warn-mode denials-only filter — no counter bump on non-strict stale
# ----------------------------------------------------------------------


def test_warn_mode_stale_does_not_bump_strict_counter(tmp_path: Path) -> None:
    """A workspace with NO strict_mode_paths emits warn-mode stale
    responses; strict_mode_denials_total stays 0."""
    _write_policy(tmp_path, tracked=["plan.md"], strict=[])
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="warn-only")
    server.serve_in_thread()
    try:
        secret = load_secret(server.coordinator_root)
        c = _Client("127.0.0.1", server.port, secret)
        _setup_stale_path(c, "plan.md")
        # A's re-read = warn-mode allow + stale-summary.
        c.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("v1")})
        assert server.counters_snapshot()["strict_mode_denials_total"] == 0
        # audit.log was never created (denials-only).
        assert not (tmp_path / ".coherence" / "audit.log").exists()
    finally:
        server.shutdown()


# ----------------------------------------------------------------------
# audit_log module — focused unit tests
# ----------------------------------------------------------------------


def test_audit_log_creates_file_with_required_mode(tmp_path: Path) -> None:
    """First append creates audit.log with 0o600 mode."""
    (tmp_path / ".coherence").mkdir(mode=0o700)
    audit_path = _resolve_audit_log_path(tmp_path)
    assert not audit_path.exists()
    ok = append_strict_deny(
        tmp_path,
        agent_id="abc-session", path="plan.md", tool="Read",
    )
    assert ok is True
    assert audit_path.exists()
    actual_mode = stat.S_IMODE(audit_path.stat().st_mode)
    assert actual_mode == _REQUIRED_MODE
    assert _REQUIRED_MODE == 0o600  # belt-and-suspenders


def test_audit_log_jsonl_schema(tmp_path: Path) -> None:
    """Schema: {ts, artifact, agent, tool, decision}. No schema_version.
    No extra fields."""
    (tmp_path / ".coherence").mkdir(mode=0o700)
    append_strict_deny(
        tmp_path,
        agent_id="11111111-1111-4111-8111-111111111111",
        path="docs/plans/x.md",
        tool="Read",
    )
    line = (tmp_path / ".coherence" / "audit.log").read_text().strip()
    record = json.loads(line)
    assert set(record.keys()) == {"ts", "artifact", "agent", "tool", "decision"}
    assert record["artifact"] == "docs/plans/x.md"
    assert record["agent"] == "11111111-1111-4111-8111-111111111111"
    assert record["tool"] == "Read"
    assert record["decision"] == "strict_deny"
    # ts is ISO-8601 — parse-back roundtrip catches malformed timestamps.
    from datetime import datetime
    datetime.fromisoformat(record["ts"])


def test_audit_log_payload_bounded_no_user_content(tmp_path: Path) -> None:
    """KTD-V bounded payload: even if a bash command body were leaked into
    the audit log somehow, the schema would not carry it. Snapshot test
    asserts no extra keys creep in beyond the locked set."""
    (tmp_path / ".coherence").mkdir(mode=0o700)
    append_strict_deny(
        tmp_path,
        agent_id="ses-1", path="CLAUDE.md", tool="Bash",
    )
    line = (tmp_path / ".coherence" / "audit.log").read_text().strip()
    record = json.loads(line)
    # The forbidden field set — adding any of these would expose more
    # surface than KTD-V scopes.
    forbidden = {
        "schema_version",
        "decision_context",
        "command",
        "tool_input",
        "additionalContext",
        "content",
        "content_hash",
        "user_message",
    }
    assert not (set(record.keys()) & forbidden), (
        f"audit.log record contains forbidden field(s): "
        f"{set(record.keys()) & forbidden}"
    )


def test_audit_log_concurrent_appends_atomic(tmp_path: Path) -> None:
    """Multiple threads writing simultaneously produce non-interleaved
    JSONL lines (O_APPEND atomicity for small payloads on POSIX)."""
    (tmp_path / ".coherence").mkdir(mode=0o700)
    N = 50
    threads: list[threading.Thread] = []
    for i in range(N):
        t = threading.Thread(
            target=append_strict_deny,
            args=(tmp_path,),
            kwargs=dict(agent_id=f"ses-{i:03d}", path=f"p{i}.md", tool="Read"),
        )
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    lines = (tmp_path / ".coherence" / "audit.log").read_text().strip().splitlines()
    assert len(lines) == N
    # Every line parses as a complete JSON object — no interleaving.
    parsed = [json.loads(line) for line in lines]
    agents = {p["agent"] for p in parsed}
    assert len(agents) == N  # All N sessions present.


def test_audit_log_mode_drift_returns_false_but_appends(tmp_path: Path) -> None:
    """Existing audit.log with mode 0o644 (operator chmod): the helper
    still appends (denials-only metadata is not credential material) but
    returns False so the caller can bump the drift counter."""
    (tmp_path / ".coherence").mkdir(mode=0o700)
    audit_path = _resolve_audit_log_path(tmp_path)
    # Seed with mode 0o644.
    audit_path.write_text("{}\n")
    audit_path.chmod(0o644)
    ok = append_strict_deny(
        tmp_path, agent_id="ses-1", path="plan.md", tool="Edit",
    )
    assert ok is False
    # Two lines now (seed + append).
    lines = audit_path.read_text().strip().splitlines()
    assert len(lines) == 2
    appended = json.loads(lines[1])
    assert appended["decision"] == "strict_deny"


def test_audit_log_append_failure_does_not_raise(tmp_path: Path) -> None:
    """OSError on append is LOGGED, not raised — the coordinator-state
    transaction has already committed by the time the audit append fires."""
    # No .coherence/ directory — open() will fail. Helper should swallow.
    # (Production never hits this — coordinator startup creates .coherence/.)
    ok = append_strict_deny(
        tmp_path, agent_id="ses-1", path="plan.md", tool="Read",
    )
    assert ok is False  # Failure surfaced via return, not raise.


# ----------------------------------------------------------------------
# End-to-end: handler deny populates audit.log
# ----------------------------------------------------------------------


def test_handler_deny_writes_audit_log_line(coordinator, client: _Client) -> None:
    """Integration: handler strict-deny calls audit_log.append_strict_deny;
    .coherence/audit.log gains exactly one line per deny."""
    audit_path = coordinator.coordinator_root / ".coherence" / "audit.log"
    assert not audit_path.exists()
    _setup_stale_path(client, "CLAUDE.md")
    client.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")})
    assert audit_path.exists()
    lines = audit_path.read_text().strip().splitlines()
    # Exactly one deny event from A's re-read.
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["tool"] == "Read"
    assert rec["artifact"] == "CLAUDE.md"
    assert rec["agent"] == _sid("A")
