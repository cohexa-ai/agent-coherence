# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Integration tests for v0.2 strict-mode handler decision-flip (plan Unit 2).

Covers (per plan Unit 2 Test scenarios):

- Happy path: strict + tracked + stale on each of the 4 PreToolUse handlers
  (Read, Edit/Write via pre-edit, Bash, Grep) returns ``permissionDecision:
  "deny"`` with the static reason text.
- Negative: strict + tracked + FRESH → allow (no stale-read).
- Negative: tracked + NOT strict + stale → warn (v0.1.1 behavior unchanged).
- Negative: NOT tracked + stale → allow passthrough (fast-path).
- Edge: Bash multi-path where ONLY one path is strict → deny (any strict
  match triggers).
- Edge: static deny reason byte-stable across N retries (KTD-T; H1
  falsification regression guard).
- KTD-U structural invariant: parameterized over allow-emitting call sites;
  each call site refuses to convert a TERMINAL_DENIAL_CLASSES-member input
  to allow.
- KTD-U coverage meta-test: the parameter list covers every call site of
  ``emit_allow`` in ``coordinator_server.py`` + ``hook_payloads.py``
  (static grep + count check).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    CoordinatorHTTPServer,
    session_to_agent_id,
)
from ccs.adapters.claude_code.hook_payloads import (
    STRICT_MODE_DENY_REASON_TEMPLATE,
    TERMINAL_DENIAL_CLASSES,
    emit_allow,
)
from ccs.core.states import MESIState

# ----------------------------------------------------------------------
# Test plumbing — mirrors tests/test_claude_code_coordinator_server.py
# ----------------------------------------------------------------------


_TEST_SESSION_NS = uuid.UUID("11111111-1111-4111-8111-111111111111")


def _sid(label: str) -> str:
    return str(uuid.uuid5(_TEST_SESSION_NS, f"strict-mode-test:{label}"))


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _Client:
    """Tiny urllib client mirroring test_claude_code_coordinator_server._Client."""

    def __init__(self, host: str, port: int, secret: str) -> None:
        self.base = f"http://{host}:{port}"
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
    ) -> tuple[int, dict]:
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
    """Materialize tracked.yaml + strict_mode.yaml under .coherence/."""
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
def workspace(tmp_path: Path) -> Path:
    """A fresh workspace per test."""
    return tmp_path


@pytest.fixture
def strict_coordinator(workspace: Path):
    """Coordinator with CLAUDE.md tracked + strict, plan.md tracked + strict,
    docs/plans/x.md tracked + warn-mode (NOT strict). Matches the most common
    test scenario shape across this file."""
    _write_policy(
        workspace,
        tracked=["CLAUDE.md", "plan.md", "docs/plans/x.md"],
        strict=["CLAUDE.md", "plan.md"],
    )
    server = CoordinatorHTTPServer(workspace, port=0, instance_id="strict-mode-test")
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


# ----------------------------------------------------------------------
# Happy path — strict deny on each handler (4 surfaces)
# ----------------------------------------------------------------------


def _setup_stale(client: _Client, path: str) -> None:
    """Set up the canonical stale scenario: sessions A and B both read path
    (each takes SHARED), B edits + commits (acquires EXCLUSIVE then MODIFIED,
    invalidates A), leaving A in INVALID state on path. The caller's NEXT
    operation on path from session A is the stale event under test.

    Both sessions must pre-read first because v0.2 strict-mode pre-edit
    requires the editor to have a fresh grant — an editor without prior
    read on a strict-mode artifact gets strict-deny (correct behavior;
    matches the operator's intent "agent MUST re-read before edit")."""
    client.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": path, "content_hash": _hash("v1")})
    client.post("/hooks/pre-read",
                {"session_id": _sid("B"), "path": path, "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit",
                {"session_id": _sid("B"), "path": path})
    client.post("/hooks/post-edit",
                {"session_id": _sid("B"), "path": path,
                 "content_hash": _hash("v2"), "success": True})


def test_pre_read_strict_tracked_stale_denies(strict_client: _Client) -> None:
    """Read on a strict + tracked + stale artifact returns deny + static reason."""
    _setup_stale(strict_client, "CLAUDE.md")
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert out["hookEventName"] == "PreToolUse"
    assert "permissionDecisionReason" in out
    reason = out["permissionDecisionReason"]
    assert "CLAUDE.md" in reason
    assert "Re-read" in reason


# ----------------------------------------------------------------------
# Survivor #6 v1 — SHARED-holder foreign-edit strict deny (read surface).
# Promotes the former fail-open allow for a SHARED holder whose supplied
# disk hash mismatches the canonical (an out-of-band / foreign edit), while
# preserving warn-mode, the sentinel guard, and the commit→disk-write-lag
# exclusion (R2).
# ----------------------------------------------------------------------


def test_pre_read_strict_shared_foreign_edit_denies(strict_client: _Client) -> None:
    """A still-SHARED reader (no peer commit since its grant) whose disk hash
    mismatches the canonical, with no commit through the coordinator (foreign
    edit), is DENIED in strict mode. last_writer is unset, so the lag-exclusion
    does not apply — this is the Dropbox out-of-band-edit case."""
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("foreign-v2")},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert out["hookEventName"] == "PreToolUse"
    assert "CLAUDE.md" in out["permissionDecisionReason"]
    assert "Re-read" in out["permissionDecisionReason"]


def test_pre_read_warn_shared_foreign_edit_still_allows(strict_client: _Client) -> None:
    """Warn-mode unchanged: on a tracked-but-NOT-strict path the SHARED-holder
    mismatch stays a fail-open allow + hash_differs (R1 warn preserved)."""
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "docs/plans/x.md", "content_hash": _hash("v1")},
    )
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "docs/plans/x.md", "content_hash": _hash("foreign-v2")},
    )
    assert status == 200
    assert body == {"status": "fresh", "version": 1, "hash_differs": True}


def test_pre_read_strict_shared_sentinel_never_denies(
    strict_coordinator, strict_client: _Client,
) -> None:
    """The launch-gate sentinel canonical ("f"*64) carries no content claim;
    a SHARED-holder re-read against it must not deny even in strict mode (the
    != _F_SENTINEL_CONTENT_HASH guard survives the promotion)."""
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    art_id = strict_coordinator.registry.lookup_artifact_id_by_name("CLAUDE.md")
    assert art_id is not None
    art = strict_coordinator.registry.get_artifact(art_id)
    strict_coordinator.registry.set_artifact_and_content(
        art_id, dataclasses.replace(art, content_hash="f" * 64), "",
    )
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_strict_shared_recent_self_commit_lag_allows(
    strict_coordinator, strict_client: _Client, monkeypatch,
) -> None:
    """FP-lag negative control (MANDATORY): a SHARED holder whose mismatch is
    its OWN recent commit (registry canonical advanced via a commit_cas WIN,
    disk not yet flushed) must NOT be denied — it falls through to the
    warn-mode allow. Distinguishes the benign commit→disk-write lag from a
    foreign edit. Without this exclusion a normal OCC-WIN-then-reread would
    false-deny."""
    import ccs.adapters.claude_code.coordinator_server as _csrv

    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    # Make CLAUDE.md's committed writer THIS caller (session s1) — the lag shape.
    caller_agent_id = strict_coordinator.register_session(_sid("s1"))
    monkeypatch.setattr(strict_coordinator.registry, "last_writer_for", lambda aid: caller_agent_id)
    monkeypatch.setattr(_csrv, "_last_writer_unix_ts", lambda coord, aid: time.time())
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("foreign-v2")},
    )
    assert status == 200
    # Lag-excluded: no deny, falls through to the warn-mode allow + hash_differs.
    assert body == {"status": "fresh", "version": 1, "hash_differs": True}


def test_pre_read_strict_shared_self_commit_lag_subagent_allows(
    strict_coordinator, strict_client: _Client, monkeypatch,
) -> None:
    """SB-25: the self-commit-lag suppression must recognize a SUBAGENT's OWN
    recent commit. The gate compares the raw committed-writer agent id against
    the CALLER's composite agent id; here the caller is a subagent
    (agent_id='suba') and the (monkeypatched) last writer is that same
    subagent's composite id → lag suppressed → warn-mode allow, not deny.
    Before SB-25 the comparison used the parent session_id and could never
    match for a subagent, wrongly denying its own re-read as a foreign edit."""
    import ccs.adapters.claude_code.coordinator_server as _csrv

    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "agent_id": "suba", "path": "CLAUDE.md",
         "content_hash": _hash("v1")},
    )
    # The committed writer is THIS subagent caller's composite agent id.
    caller_agent_id = strict_coordinator.register_session(_sid("s1"), "suba")
    monkeypatch.setattr(strict_coordinator.registry, "last_writer_for", lambda aid: caller_agent_id)
    monkeypatch.setattr(_csrv, "_last_writer_unix_ts", lambda coord, aid: time.time())
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "agent_id": "suba", "path": "CLAUDE.md",
         "content_hash": _hash("foreign-v2")},
    )
    assert status == 200
    assert body == {"status": "fresh", "version": 1, "hash_differs": True}


def test_pre_read_strict_shared_foreign_edit_records_telemetry(
    strict_coordinator, strict_client: _Client,
) -> None:
    """The SHARED-holder foreign-edit deny fires the SAME telemetry as the INVALID
    deny (shared _emit_pre_read_strict_deny helper): bump strict_mode_denials_total
    and record the (session, path) for route-around. Guards a refactor that returns
    the right HTTP body but drops the side-effects."""
    before = strict_coordinator._strict_mode_denials_total
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("foreign-v2")},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert strict_coordinator._strict_mode_denials_total == before + 1
    assert strict_coordinator.check_strict_deny_route_around(_sid("s1"), "CLAUDE.md") is True


def test_pre_read_strict_shared_lag_suppression_increments_counter(
    strict_coordinator, strict_client: _Client, monkeypatch,
) -> None:
    """The R2 lag-suppression path bumps shared_foreign_lag_suppressed_total so an
    operator can size the lag-window false-negative rate, and still falls through
    to the warn-mode allow (no deny)."""
    import ccs.adapters.claude_code.coordinator_server as _csrv

    before = strict_coordinator._shared_foreign_lag_suppressed_total
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    caller_agent_id = strict_coordinator.register_session(_sid("s1"))
    monkeypatch.setattr(strict_coordinator.registry, "last_writer_for", lambda aid: caller_agent_id)
    monkeypatch.setattr(_csrv, "_last_writer_unix_ts", lambda coord, aid: time.time())
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("foreign-v2")},
    )
    assert status == 200
    assert body == {"status": "fresh", "version": 1, "hash_differs": True}
    assert strict_coordinator._shared_foreign_lag_suppressed_total == before + 1


def test_pre_read_strict_remint_after_foreign_edit_still_denies(
    strict_client: _Client,
) -> None:
    """RR-2 refutation: after a foreign edit the canonical never advances (no
    commit), so a re-minted identity (the OCC write_cas remint, modeled as a fresh
    session) reading the foreign bytes is DENIED, not silently granted
    fresh-SHARED. There is no clean SHARED@foreign state for a later fresh-SHARED
    read to skip-check, and a repeat read keeps denying."""
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s1"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s2"), "path": "CLAUDE.md", "content_hash": _hash("foreign-v2")},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny"
    status2, body2 = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("s2"), "path": "CLAUDE.md", "content_hash": _hash("foreign-v2")},
    )
    assert status2 == 200
    assert body2["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_pre_edit_strict_tracked_stale_denies(strict_client: _Client) -> None:
    """Edit on a strict + tracked + stale artifact returns deny (the Edit/Write
    surface — pre-edit handles both per the hooks.json matcher Edit|Write)."""
    _setup_stale(strict_client, "plan.md")
    # Session A tries to edit plan.md without re-reading.
    status, body = strict_client.post(
        "/hooks/pre-edit",
        {"session_id": _sid("A"), "path": "plan.md"},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert out["hookEventName"] == "PreToolUse"
    assert "plan.md" in out["permissionDecisionReason"]


def test_pre_bash_strict_tracked_stale_denies(strict_client: _Client) -> None:
    """Bash command that reads a strict + tracked + stale artifact denies."""
    _setup_stale(strict_client, "plan.md")
    status, body = strict_client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "plan.md" in out["permissionDecisionReason"]


def test_pre_grep_strict_tracked_stale_denies(strict_client: _Client) -> None:
    """Grep over a directory containing a strict + tracked + stale artifact
    denies the whole grep command."""
    _setup_stale(strict_client, "plan.md")
    status, body = strict_client.post(
        "/hooks/pre-grep",
        {"session_id": _sid("A"), "search_root": ""},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "plan.md" in out["permissionDecisionReason"]


# ----------------------------------------------------------------------
# Negative paths — preserve v0.1.1 warn-mode behavior for non-strict
# ----------------------------------------------------------------------


def test_pre_read_strict_tracked_fresh_allows(strict_client: _Client) -> None:
    """Strict + tracked + FRESH (no stale event) returns the fresh allow shape
    — no deny when the artifact is up to date for this session."""
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    # First observation seeds SHARED → fresh response, no hookSpecificOutput.
    # Unit 6: the fresh response additively carries the seeded version.
    assert body == {"status": "fresh", "version": 1}


def test_pre_read_tracked_not_strict_stale_returns_warn(strict_client: _Client) -> None:
    """Tracked + NOT strict + stale returns warn-mode allow (v0.1.1 behavior
    preserved for warn-mode artifacts even when strict_mode is configured for
    other artifacts in the same workspace)."""
    # docs/plans/x.md is tracked but NOT in strict_mode_paths.
    _setup_stale(strict_client, "docs/plans/x.md")
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "docs/plans/x.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "Stale read" in out["additionalContext"]


def test_pre_read_untracked_path_strict_irrelevant(strict_client: _Client) -> None:
    """Untracked path takes the policy fast-path; strict mode never applies."""
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "untracked.txt"},
    )
    assert status == 200
    assert body == {"status": "fresh"}


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_pre_bash_multi_path_one_strict_one_warn_denies(strict_client: _Client) -> None:
    """`cat plan.md docs/plans/x.md` — plan.md is strict, docs/plans/x.md is
    warn-only. ANY strict-stale match triggers deny for the whole command."""
    _setup_stale(strict_client, "plan.md")           # strict
    _setup_stale(strict_client, "docs/plans/x.md")  # warn-only
    status, body = strict_client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat plan.md docs/plans/x.md"},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "plan.md" in out["permissionDecisionReason"]


def test_pre_bash_only_warn_paths_still_allows(strict_client: _Client) -> None:
    """If a Bash command touches only warn-mode tracked artifacts (none in
    strict), the v0.1.1 warn-mode allow shape is preserved."""
    _setup_stale(strict_client, "docs/plans/x.md")
    status, body = strict_client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat docs/plans/x.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_strict_deny_reason_byte_stable_across_retries(strict_client: _Client) -> None:
    """KTD-T (H1 falsification regression guard): the deny reason MUST be
    byte-identical across N retries of the same (session, path) staleness
    event. Per-invocation timestamp variation would re-introduce the opus
    prompt-injection retry hazard the Phase 0 falsifiability experiment
    surfaced."""
    _setup_stale(strict_client, "CLAUDE.md")
    reasons: list[str] = []
    for _ in range(5):
        _, body = strict_client.post(
            "/hooks/pre-read",
            {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
        )
        reasons.append(body["hookSpecificOutput"]["permissionDecisionReason"])
    assert len(set(reasons)) == 1, (
        f"Deny reason rotated across retries — KTD-P invariant violated. "
        f"Unique reasons: {set(reasons)}"
    )


# ----------------------------------------------------------------------
# SB-10 U4 (AE3): deferred re-grounding must never touch strict-deny bodies
# ----------------------------------------------------------------------


def test_strict_deny_with_pending_reground_flag_byte_identical_and_flag_survives(
    strict_coordinator, strict_client: _Client,
) -> None:
    """AE3 first half (KTD-P preservation): a strict-mode deny issued while
    the session's compact-pending flag is armed carries a deny envelope
    byte-identical to the no-flag baseline — the deferred re-grounding
    payload attaches ONLY to allow envelopes (R8) — and the deny neither
    consumes nor expires the flag (only a qualifying admit consumes, R2).

    Written FIRST per the unit's execution note: this test passes against
    the pre-U4 coordinator (no admit consumes the flag today) and must stay
    green throughout the deferred-injection implementation."""
    _setup_stale(strict_client, "CLAUDE.md")
    # Baseline deny envelope with NO flag armed.
    status, baseline = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    assert baseline["hookSpecificOutput"]["permissionDecision"] == "deny"
    # Arm the deferred-delivery flag, then retry the same denied read.
    strict_coordinator.mark_compact_pending(_sid("A"))
    status, with_flag = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    # The deny envelope is byte-stable (KTD-P) AND untouched by the pending
    # flag — no re-grounding prose may ride a deny.
    assert with_flag["hookSpecificOutput"] == baseline["hookSpecificOutput"]
    assert set(with_flag.keys()) == set(baseline.keys())
    # Flag still pending: the deny consumed nothing (asserted via the
    # test-and-clear primitive, which also cleans up the armed flag).
    assert strict_coordinator.consume_compact_pending(_sid("A")) is True


def test_strict_deny_then_allowed_warn_read_orders_notices_before_reground(
    strict_coordinator, strict_client: _Client,
) -> None:
    """AE3 second half: after a strict deny left the flag pending, the NEXT
    qualifying admit — a warn-mode stale re-read that also drains a pending
    preemption notice — delivers everything in one additionalContext, in
    the order notices → stale warning → re-grounding block."""
    # Stale strict artifact FIRST — its fresh pre-reads would otherwise
    # drain the preemption notice staged below (fresh-path notice surfacing).
    _setup_stale(strict_client, "CLAUDE.md")
    # Session A holds EXCLUSIVE on the warn-mode path; B preempts (recording
    # a notice for A) and commits, leaving A INVALID on docs/plans/x.md.
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "docs/plans/x.md", "content_hash": _hash("v1")},
    )
    strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("A"), "path": "docs/plans/x.md"},
    )
    strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("B"), "path": "docs/plans/x.md"},
    )
    strict_client.post(
        "/hooks/post-edit",
        {"session_id": _sid("B"), "path": "docs/plans/x.md",
         "content_hash": _hash("v2"), "success": True},
    )
    # Strict deny for A on CLAUDE.md with the flag armed: flag must survive
    # the deny (no consume) so the next qualifying admit can deliver.
    strict_coordinator.mark_compact_pending(_sid("A"))
    status, denied = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny"
    # Follow-up ALLOWED warn-mode re-read: notice + stale warning + re-ground.
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "docs/plans/x.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    text = out["additionalContext"]
    notice_at = text.index("Coordinator notice")
    stale_at = text.index("Stale read")
    reground_at = text.index("Post-compaction re-grounding (agent-coherence):")
    assert notice_at < stale_at < reground_at
    # Delivered exactly once: the flag is gone.
    assert strict_coordinator.consume_compact_pending(_sid("A")) is False


# ----------------------------------------------------------------------
# KTD-U structural invariant: emit_allow refuses terminal-denial conversion
# ----------------------------------------------------------------------


# Parameter list: ALL call sites of emit_allow in coordinator_server.py +
# hook_payloads.py. The meta-test below grep-counts emit_allow calls in those
# files and asserts this list has the same length. Adding a new emit_allow
# call site MUST extend this parameter list — that's the structural guarantee
# Unit 2's KTD-U design provides.
ALLOW_EMISSION_SOURCES: list[str] = [
    "stale_response_builder",        # hook_payloads.build_stale_response
    "collision_response_builder",    # hook_payloads.build_collision_response
    "pre_read_fresh_with_notice",    # coordinator_server._handle_pre_read
    "pre_edit_notice_only",          # coordinator_server._handle_pre_edit
    "pre_bash_stale_warn",           # coordinator_server._handle_pre_bash
    "pre_grep_stale_warn",           # coordinator_server._handle_pre_grep
    "watchdog_degraded_read",        # coordinator_server._DEFAULT_DEGRADED_RESPONSE (A7)
    # NOT listed: coordinator_server._attach_reground. The SB-10 deferred
    # re-grounding attach no longer emits an allow — a bare admit body now
    # gains a CONTEXT-ONLY PreToolUse envelope (hookEventName +
    # additionalContext, no permissionDecision) via
    # hook_payloads.emit_pretooluse_context, because an advisory payload
    # must never widen a permission decision. No emit_allow call site, so
    # no entry here.
]


@pytest.mark.parametrize("source", ALLOW_EMISSION_SOURCES)
def test_emit_allow_refuses_terminal_denial_class(source: str) -> None:
    """KTD-U invariant (structural, not behavioral): for every allow-emitting
    call site, ``emit_allow`` with a TERMINAL_DENIAL_CLASSES-member denial_class
    raises AssertionError. A future contributor cannot satisfy the test
    trivially by adding a new allow path — they must extend the parameter
    list and therefore think about the invariant."""
    terminal_class = next(iter(TERMINAL_DENIAL_CLASSES))
    with pytest.raises(AssertionError, match="TERMINAL_DENIAL_CLASSES"):
        emit_allow(source=source, denial_class=terminal_class)


def test_emit_allow_passes_for_non_terminal_class() -> None:
    """Sanity: emit_allow returns the allow envelope when denial_class is
    None or not in the terminal set."""
    out = emit_allow(source="sanity_test", additional_context="hello")
    assert out["permissionDecision"] == "allow"
    assert out["hookEventName"] == "PreToolUse"
    assert out["additionalContext"] == "hello"


def test_terminal_denial_classes_includes_strict_mode_deny() -> None:
    """The strict-mode deny class is in TERMINAL_DENIAL_CLASSES. This is the
    security marker the deny path uses."""
    assert "permissions_deny_strict_mode" in TERMINAL_DENIAL_CLASSES


# ----------------------------------------------------------------------
# KTD-U coverage meta-test: parameter list covers every emit_allow call
# ----------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCANNED_FILES = (
    _REPO_ROOT / "src" / "ccs" / "adapters" / "claude_code" / "coordinator_server.py",
    _REPO_ROOT / "src" / "ccs" / "adapters" / "claude_code" / "hook_payloads.py",
)


def _count_emit_allow_call_sites(path: Path) -> int:
    """AST-based call-site counter. Matches only real ``emit_allow(...)``
    Call nodes — not docstring mentions, error-message strings, or the
    function definition itself."""
    import ast

    tree = ast.parse(path.read_text())
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "emit_allow":
                count += 1
            elif isinstance(func, ast.Attribute) and func.attr == "emit_allow":
                count += 1
    return count


def test_ktd_u_emit_allow_call_sites_covered_by_parameter_list() -> None:
    """AST scan: total real ``emit_allow(...)`` call sites across
    coordinator_server.py + hook_payloads.py must equal
    ``len(ALLOW_EMISSION_SOURCES)``. Adding a new emit_allow call site
    requires extending the parameter list — that's the structural guarantee
    that future code reviews catch the invariant discussion before the new
    site can land."""
    call_count = sum(_count_emit_allow_call_sites(p) for p in _SCANNED_FILES)
    assert call_count == len(ALLOW_EMISSION_SOURCES), (
        f"emit_allow() call sites in scanned files: {call_count}; "
        f"ALLOW_EMISSION_SOURCES parameter list: {len(ALLOW_EMISSION_SOURCES)}. "
        f"Extend ALLOW_EMISSION_SOURCES or audit the new call site for KTD-U "
        f"invariant compliance."
    )


# ----------------------------------------------------------------------
# Static reason template format string sanity
# ----------------------------------------------------------------------


def test_strict_mode_deny_reason_template_is_static() -> None:
    """KTD-P (static deny text, NO template rotation). The template is a
    module-level constant; this test guards against accidental
    timestamp-of-rendering interpolation that would violate byte-stability."""
    expected_placeholders = {"path", "last_writer_short", "last_writer_ts_iso"}
    # Extract format-string placeholders.
    actual = set(
        m.group(1) for m in re.finditer(r"\{([a-z_]+)\}", STRICT_MODE_DENY_REASON_TEMPLATE)
    )
    assert actual == expected_placeholders, (
        f"STRICT_MODE_DENY_REASON_TEMPLATE placeholders changed: "
        f"expected {expected_placeholders}, got {actual}. Adding a "
        f"per-invocation field (e.g., warning_generated_at) violates "
        f"KTD-P byte-stability."
    )


def test_pre_read_strict_deny_ignores_want_owner_generation(
    strict_client: _Client,
) -> None:
    """A strict-mode deny never carries the pair, opt-in or not: the byte-stable
    deny payload is a compatibility surface (KTD-T), and the ABSENT generation
    already makes a generation-aware caller (the effect gate) HOLD fail-closed."""
    _setup_stale(strict_client, "CLAUDE.md")
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md",
         "content_hash": _hash("v1"), "want_owner_generation": True},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "owner_generation" not in body
    assert "version" not in body


# ----------------------------------------------------------------------
# A grant handover is not a write (R8) -- Cohexa-ai/agent-coherence#196
# ----------------------------------------------------------------------
#
# The reported sequence: A holds the artifact, B calls pre-edit and takes the
# grant, nothing is committed, and A's next read is denied with "was updated
# by session <unknown> at <t>". No commit happened, the version never moved,
# and the deny named a writer that does not exist. This is the end-to-end
# form of the two renderer tests in
# ``tests/test_claude_code_coordinator_server.py``.


def _setup_grant_handover(client: _Client, path: str) -> None:
    """A reads (SHARED on v1), B takes the grant via pre-edit, B never commits.

    A is INVALID afterwards with its last-observed version still v1, which is
    also the artifact's current version -- nothing was written.
    """
    client.post("/hooks/pre-read",
                {"session_id": _sid("A"), "path": path, "content_hash": _hash("v1")})
    client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": path})


def test_grant_handover_deny_reports_no_write_and_an_unchanged_version(
    strict_client: _Client,
) -> None:
    """Both halves, because either alone passes while the other is wrong.

    The prose alone would pass with a version the response got wrong, and the
    version alone would pass with prose still claiming a write.
    """
    _setup_grant_handover(strict_client, "CLAUDE.md")
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body

    reason = body["hookSpecificOutput"]["permissionDecisionReason"]
    assert "was updated by" not in reason, (
        f"no write happened, so the deny must not report one; got: {reason}"
    )
    assert "your grant on CLAUDE.md was revoked and no new version was committed" in reason, reason
    assert "CLAUDE.md is still at v1" in reason, reason

    summary = body["summary"]
    assert summary["current_version"] == 1, summary
    assert summary["prior_version_seen_by_session"] == 1, (
        f"A observed v1 and v1 is still current; reporting v0 invents a "
        f"version A never saw: {summary}"
    )


def test_a_real_commit_still_denies_with_the_write_wording(
    strict_client: _Client,
) -> None:
    """The control for the test above: when a peer really did commit, the
    deny keeps naming the write and the version it moved to."""
    _setup_stale(strict_client, "CLAUDE.md")
    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    assert status == 200
    reason = body["hookSpecificOutput"]["permissionDecisionReason"]
    assert "CLAUDE.md was updated by agent " in reason, reason
    assert "was revoked" not in reason, reason
    assert body["summary"]["current_version"] == 2, body["summary"]
    assert body["summary"]["prior_version_seen_by_session"] == 1, body["summary"]


def test_grant_handover_deny_is_byte_stable_across_retries(
    strict_client: _Client,
) -> None:
    """KTD-T still holds on the new arm: retrying reproduces the same bytes.

    The grant-change reason interpolates only the path and the current
    version, so unlike the write arm it carries no timestamp at all.
    """
    _setup_grant_handover(strict_client, "CLAUDE.md")
    reasons = set()
    for _ in range(4):
        _, body = strict_client.post(
            "/hooks/pre-read",
            {"session_id": _sid("A"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
        )
        reasons.add(body["hookSpecificOutput"]["permissionDecisionReason"])
    assert len(reasons) == 1, reasons


def test_pre_edit_deny_after_a_grant_handover_reports_no_write(
    strict_client: _Client,
) -> None:
    """The Edit surface takes the same arm: A's pre-edit after losing the
    grant must not be told the artifact was updated."""
    _setup_grant_handover(strict_client, "CLAUDE.md")
    status, body = strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("A"), "path": "CLAUDE.md"},
    )
    assert status == 200
    reason = body["hookSpecificOutput"]["permissionDecisionReason"]
    assert "was updated by" not in reason, reason
    assert "CLAUDE.md is still at v1" in reason, reason
    assert body["summary"]["prior_version_seen_by_session"] == 1, body["summary"]


# ----------------------------------------------------------------------
# A denied read is not an observation
# ----------------------------------------------------------------------
#
# pre-bash and pre-grep re-grant SHARED to every stale path they name, and
# they do it on the strict path too: the deny fires once and the retry goes
# through (Node's pre_bash.ts documents the same contract). A SHARED grant
# used to be an observation, full stop -- ``set_agent_state`` recorded the
# current version as the agent's ``last_observed_version`` for any
# non-INVALID target. So a DENIED ``cat plan.md``, which never ran, credited
# the session with having seen the version it was refused. When a peer then
# took the grant without committing, the stale summary compared that invented
# baseline against an unchanged version and told the session nothing had been
# written since "the version you last saw" -- a version it never saw.
#
# The allowed twin is NOT a bug and must stay: in warn mode the command runs
# and the session does read the current bytes, exactly like the pre-read
# warn path's ``post_stale_read`` re-grant. What decides the observation is
# whether the command was DENIED -- not the trigger's name, and not whether
# the individual path is strict (a warn-only path inside a denied command was
# not read either). The grant itself is unchanged in both directions.


def _agent(label: str) -> uuid.UUID:
    return session_to_agent_id(_sid(label))


def _observed(coordinator: CoordinatorHTTPServer, path: str, label: str) -> Optional[int]:
    artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
    assert artifact_id is not None, f"{path} is not registered"
    return coordinator.registry.last_observed_version_for(artifact_id, _agent(label))


def _mesi(coordinator: CoordinatorHTTPServer, path: str, label: str) -> Optional[MESIState]:
    artifact_id = coordinator.registry.lookup_artifact_id_by_name(path)
    assert artifact_id is not None, f"{path} is not registered"
    return coordinator.registry.get_agent_state(artifact_id, _agent(label))


def _take_grant_without_committing(
    coordinator: CoordinatorHTTPServer, client: _Client, path: str,
) -> None:
    """B takes the grant again and writes nothing, which leaves A INVALID at
    an UNCHANGED version -- the state in which the prose has to choose between
    "someone committed" and "only the grant moved". Asserted, not assumed: if
    A were not invalidated the read below would take the fresh arm and every
    wording assertion after it would be vacuous."""
    version_before = coordinator.registry.get_artifact(
        coordinator.registry.lookup_artifact_id_by_name(path)
    ).version
    status, body = client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": path})
    assert status == 200 and body.get("ok") is True, body
    assert _mesi(coordinator, path, "A") == MESIState.INVALID
    assert coordinator.registry.get_artifact(
        coordinator.registry.lookup_artifact_id_by_name(path)
    ).version == version_before, "a pre-edit must not move the version"


def test_denied_pre_bash_does_not_advance_the_observation_baseline(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    _setup_stale(strict_client, "plan.md")
    assert _observed(strict_coordinator, "plan.md", "A") == 1  # A read v1, then B committed v2

    status, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body

    assert _observed(strict_coordinator, "plan.md", "A") == 1, (
        "the command was denied, so A never read v2; crediting it as observed "
        "is what later makes an unchanged version look like the one A last saw"
    )
    # The grant is deliberately unchanged: the strict bash deny still fires
    # once and re-arms the session. Only the observation claim moved.
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.SHARED


def test_denied_pre_bash_then_a_grant_handover_still_reports_the_write(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The probe's sequence end to end: the prose and the version, both,
    because either alone passes while the other is wrong."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    _take_grant_without_committing(strict_coordinator, strict_client, "plan.md")

    status, body = strict_client.post(
        "/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    reason = body["hookSpecificOutput"]["permissionDecisionReason"]
    assert "plan.md was updated by agent " in reason, reason
    assert "no new version was committed" not in reason, reason
    assert body["summary"]["current_version"] == 2, body["summary"]
    assert body["summary"]["prior_version_seen_by_session"] == 1, body["summary"]


def test_allowed_pre_bash_advances_the_observation_baseline(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The warn twin: the command runs, A reads v2, and the baseline says so.
    A fix that stopped advancing on the bash trigger would re-break this in
    the old direction -- "you previously saw v1" about bytes A just read."""
    _setup_stale(strict_client, "docs/plans/x.md")
    assert _observed(strict_coordinator, "docs/plans/x.md", "A") == 1

    status, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat docs/plans/x.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "allow", body
    assert _observed(strict_coordinator, "docs/plans/x.md", "A") == 2
    assert _mesi(strict_coordinator, "docs/plans/x.md", "A") == MESIState.SHARED

    _take_grant_without_committing(strict_coordinator, strict_client, "docs/plans/x.md")
    status, body = strict_client.post(
        "/hooks/pre-read", {"session_id": _sid("A"), "path": "docs/plans/x.md"},
    )
    assert status == 200
    out = body["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow", body
    text = out["additionalContext"]
    assert (
        "your grant on docs/plans/x.md was revoked and no new version was committed"
        in text
    ), text
    assert "the version you last saw" in text, text
    assert "was updated by" not in text, text
    assert body["summary"]["prior_version_seen_by_session"] == 2, body["summary"]


def test_a_warn_path_inside_a_denied_bash_command_is_not_observed_either(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The deny is per COMMAND, so the key is the command's outcome, not the
    path's strictness: ``cat plan.md docs/plans/x.md`` never ran, and x.md --
    warn-only on its own -- was not read by it."""
    _setup_stale(strict_client, "plan.md")
    _setup_stale(strict_client, "docs/plans/x.md")

    status, body = strict_client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat plan.md docs/plans/x.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    assert _observed(strict_coordinator, "plan.md", "A") == 1
    assert _observed(strict_coordinator, "docs/plans/x.md", "A") == 1
    assert _mesi(strict_coordinator, "docs/plans/x.md", "A") == MESIState.SHARED


def test_first_observation_inside_a_denied_bash_command_records_no_observation(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """KTD-9 seeding registers a never-seen path and grants it SHARED. Inside
    a denied command that grant is not a read: the pair stays never-observed
    (None, never a 0-sentinel)."""
    _setup_stale(strict_client, "plan.md")

    status, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md CLAUDE.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    assert _mesi(strict_coordinator, "CLAUDE.md", "A") == MESIState.SHARED
    assert _observed(strict_coordinator, "CLAUDE.md", "A") is None


def test_first_observation_inside_an_allowed_bash_command_is_observed(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The allowed twin of the test above: the command runs, so the seed is a
    read of v1."""
    _setup_stale(strict_client, "docs/plans/x.md")

    status, body = strict_client.post(
        "/hooks/pre-bash",
        {"session_id": _sid("A"), "command": "cat docs/plans/x.md CLAUDE.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "allow", body
    assert _mesi(strict_coordinator, "CLAUDE.md", "A") == MESIState.SHARED
    assert _observed(strict_coordinator, "CLAUDE.md", "A") == 1


def test_denied_pre_grep_does_not_advance_the_observation_baseline(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    _setup_stale(strict_client, "plan.md")

    status, body = strict_client.post(
        "/hooks/pre-grep", {"session_id": _sid("A"), "search_root": ""},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    assert _observed(strict_coordinator, "plan.md", "A") == 1
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.SHARED


def test_allowed_pre_grep_advances_the_observation_baseline(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    _setup_stale(strict_client, "docs/plans/x.md")

    status, body = strict_client.post(
        "/hooks/pre-grep", {"session_id": _sid("A"), "search_root": "docs/plans"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "allow", body
    assert _observed(strict_coordinator, "docs/plans/x.md", "A") == 2


def test_an_allowed_stale_read_still_advances_the_observation_baseline(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """Control: the ordinary warn-mode read path is untouched by the fix."""
    _setup_stale(strict_client, "docs/plans/x.md")
    assert _observed(strict_coordinator, "docs/plans/x.md", "A") == 1

    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "docs/plans/x.md", "content_hash": _hash("v2")},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "allow", body
    assert _observed(strict_coordinator, "docs/plans/x.md", "A") == 2


def _reground_text(client: _Client) -> str:
    status, body = client.post("/hooks/session-start", {"session_id": _sid("A")})
    assert status == 200, body
    return body["hookSpecificOutput"]["additionalContext"]


def test_denied_pre_bash_keeps_the_post_compaction_stale_flag(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The baseline's other reader. After a compaction the re-grounding flags a
    file whose version moved past the one the session last observed. Crediting
    the denied ``cat`` as an observation of v2 silenced that flag for a file
    the session never read at v2 -- the payload said only "is at v2"."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    _take_grant_without_committing(strict_coordinator, strict_client, "plan.md")

    text = _reground_text(strict_client)
    assert "plan.md advanced to v2 past your last-observed v1" in text, text


def test_allowed_pre_bash_does_not_raise_a_false_post_compaction_stale_flag(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The allowed twin: A read v2 through the command, so there is nothing to
    flag -- a fix that stopped recording allowed reads would raise one."""
    _setup_stale(strict_client, "docs/plans/x.md")
    strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat docs/plans/x.md"},
    )
    _take_grant_without_committing(strict_coordinator, strict_client, "docs/plans/x.md")

    text = _reground_text(strict_client)
    assert "docs/plans/x.md is at v2." in text, text
    assert "advanced to" not in text, text


# ----------------------------------------------------------------------
# The retry a deny invites is a read
# ----------------------------------------------------------------------
#
# The flip side of the section above. A strict Bash or Grep deny re-grants
# SHARED without recording an observation, and it does so to let the retry go
# through: strict mode never re-grants on a denied Read (KTD-T), so running
# the command again is how a strict session recovers. That retry, and any Read
# the session takes instead, finds the grant already held and returns fresh.
# The command then runs and reads the current bytes, so the read has to be
# recorded there. Otherwise the session's baseline stays at the version before
# the deny (or at never-observed, on a first touch), and every later message
# computed from it is wrong in the unsafe direction: a peer's commit loses its
# "advanced past your last-observed" flag, and a later grant handover is
# reported as a write the session already read.
#
# Only a SHARED holder is credited, and only when the command runs. An E/M
# holder keeps its grant untouched, and a verify_only fence read records
# nothing, because it discards the bytes it read. A Grep credits no held file
# at all: its path set is every tracked file under its root, not what it
# showed, and crediting that would clear the flag on a file the deny just
# told the session to re-read. The safe side is the baseline staying behind
# until a Read or a retried Bash command records the read.


def test_the_bash_retry_a_deny_invites_advances_the_observation_baseline(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    _setup_stale(strict_client, "plan.md")
    _, denied = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", denied
    assert _observed(strict_coordinator, "plan.md", "A") == 1
    artifact_id = strict_coordinator.registry.lookup_artifact_id_by_name("plan.md")
    read_generation = strict_coordinator.registry.get_read_generation(artifact_id, _agent("A"))

    status, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert status == 200
    assert body == {"status": "fresh"}, body
    assert _observed(strict_coordinator, "plan.md", "A") == 2, (
        "the retry ran and read v2; leaving the baseline at v1 makes every "
        "later stale message report a version A has since read"
    )
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.SHARED
    # The credit moves the baseline and nothing else: a claim-capture trigger
    # here would also re-arm the read-generation fence for a read it did not
    # take through the effect gate.
    assert strict_coordinator.registry.get_read_generation(
        artifact_id, _agent("A")
    ) == read_generation


def test_a_bash_retry_then_a_grant_handover_reports_no_write(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The retry's end-to-end consequence, prose and summary both: A read v2
    through the retry, so when B takes the grant and writes nothing, A's next
    read must take the grant-change arm, not name a write A already read."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    _take_grant_without_committing(strict_coordinator, strict_client, "plan.md")

    status, body = strict_client.post(
        "/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md"},
    )
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    reason = body["hookSpecificOutput"]["permissionDecisionReason"]
    assert "was updated by" not in reason, reason
    assert "your grant on plan.md was revoked and no new version was committed" in reason, reason
    assert "plan.md is still at v2" in reason, reason
    assert body["summary"]["current_version"] == 2, body["summary"]
    assert body["summary"]["prior_version_seen_by_session"] == 2, body["summary"]


def test_a_first_touch_bash_retry_keeps_the_post_compaction_stale_flag(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """A's first contact with plan.md is a denied ``cat``: the deny records no
    observation, so the retry is A's only read of v1. If it goes unrecorded,
    A's baseline stays never-observed, and after B commits v2 the
    re-grounding says only "is at v2" -- the flag for a file A read at v1 and
    that moved since is gone."""
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("B"), "path": "plan.md", "content_hash": _hash("v1")},
    )
    _, denied = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", denied
    assert _observed(strict_coordinator, "plan.md", "A") is None

    _, retried = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert retried == {"status": "fresh"}, retried
    assert _observed(strict_coordinator, "plan.md", "A") == 1

    strict_client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "plan.md"})
    strict_client.post(
        "/hooks/post-edit",
        {"session_id": _sid("B"), "path": "plan.md",
         "content_hash": _hash("v2"), "success": True},
    )
    text = _reground_text(strict_client)
    assert "plan.md advanced to v2 past your last-observed v1" in text, text


def test_a_grep_after_a_denied_bash_does_not_count_as_reading_the_file(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """A's ``cat plan.md`` is denied; A's next call is a Grep with no path,
    which lists every tracked file under the root. The Grep never showed
    plan.md, so it must not stand in for the re-read the deny asked for: the
    baseline stays at v1, and after a grant handover A is still told about the
    write it has not read, prose and summary both."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})

    _, body = strict_client.post(
        "/hooks/pre-grep", {"session_id": _sid("A"), "search_root": ""},
    )
    assert body == {"status": "fresh"}, body
    assert _observed(strict_coordinator, "plan.md", "A") == 1

    _take_grant_without_committing(strict_coordinator, strict_client, "plan.md")
    _, read = strict_client.post(
        "/hooks/pre-read", {"session_id": _sid("A"), "path": "plan.md"},
    )
    assert read["hookSpecificOutput"]["permissionDecision"] == "deny", read
    reason = read["hookSpecificOutput"]["permissionDecisionReason"]
    assert "plan.md was updated by agent " in reason, reason
    assert read["summary"]["prior_version_seen_by_session"] == 1, read["summary"]


def test_a_grep_retry_leaves_the_baseline_for_the_read_to_record(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """A denied Grep's own retry goes through without crediting the held file;
    the Read that follows is what records v2."""
    _setup_stale(strict_client, "plan.md")
    _, denied = strict_client.post(
        "/hooks/pre-grep", {"session_id": _sid("A"), "search_root": ""},
    )
    assert denied["hookSpecificOutput"]["permissionDecision"] == "deny", denied

    _, retried = strict_client.post(
        "/hooks/pre-grep", {"session_id": _sid("A"), "search_root": ""},
    )
    assert retried == {"status": "fresh"}, retried
    assert _observed(strict_coordinator, "plan.md", "A") == 1

    _, read = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("v2")},
    )
    assert read["status"] == "fresh", read
    assert _observed(strict_coordinator, "plan.md", "A") == 2


def test_a_read_after_a_denied_bash_advances_the_observation_baseline(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The other way to recover: after the denied ``cat``, A uses Read. The
    grant is already held, so pre-read answers fresh, and that read ran too."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})

    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("v2")},
    )
    assert status == 200
    assert body["status"] == "fresh", body
    assert "hookSpecificOutput" not in body, body
    assert _observed(strict_coordinator, "plan.md", "A") == 2


def test_a_verify_only_read_after_a_denied_bash_records_no_observation(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The effect gate's verification read compares comparands and throws the
    bytes away, so it must not stand in for the read A still owes."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})

    status, body = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "plan.md", "verify_only": True},
    )
    assert status == 200
    assert body["status"] == "fresh", body
    assert _observed(strict_coordinator, "plan.md", "A") == 1


def test_a_bash_retry_that_is_denied_again_records_no_observation(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """A held path inside a command that is denied on ANOTHER path was not
    read either: the credit follows the command's outcome, as the grants do."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    _setup_stale(strict_client, "CLAUDE.md")

    _, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md CLAUDE.md"},
    )
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    assert "CLAUDE.md" in body["hookSpecificOutput"]["permissionDecisionReason"], body
    assert _observed(strict_coordinator, "plan.md", "A") == 1


def test_a_bash_read_leaves_a_held_exclusive_grant_alone(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """Only a SHARED holder is credited. Re-granting SHARED to every held path
    would downgrade a session's own write grant under it.

    Every hook-issued E/M grant records the current version, so no hook
    sequence leaves an E holder behind; the row is built through the registry
    instead, because a guard that only a behind baseline reaches is otherwise
    pinned by nothing."""
    _setup_stale(strict_client, "plan.md")
    artifact_id = strict_coordinator.registry.lookup_artifact_id_by_name("plan.md")
    strict_coordinator.registry.set_agent_state(
        artifact_id, _agent("A"), MESIState.EXCLUSIVE, trigger="test", tick=0, observed=False,
    )
    assert _observed(strict_coordinator, "plan.md", "A") == 1

    _, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert body == {"status": "fresh"}, body
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.EXCLUSIVE


def test_crediting_a_held_read_never_restores_a_revoked_grant(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The handler reads the grant before the deny decision and credits the
    read after it, so a peer can revoke the grant in between. The credit must
    re-check under the registry lock and leave the INVALID row alone: writing
    SHARED over it would clear a stale flag A has not seen yet."""
    from ccs.adapters.claude_code.coordinator_server import _record_held_read

    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.SHARED
    _take_grant_without_committing(strict_coordinator, strict_client, "plan.md")

    artifact_id = strict_coordinator.registry.lookup_artifact_id_by_name("plan.md")
    _record_held_read(
        strict_coordinator, artifact_id, _agent("A"), trigger="held_bash_read", tick=0,
    )
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.INVALID
    assert _observed(strict_coordinator, "plan.md", "A") == 1


# ----------------------------------------------------------------------
# A denied shell read re-arms the grant, not the right to write
# ----------------------------------------------------------------------
#
# A strict Bash / Grep deny re-grants SHARED so that the retry it invites goes
# through, and it records no observation, because the command never ran. The
# strict pre-edit gate used to look only for INVALID, so that re-armed grant
# also admitted the session's next Edit or Write: a whole-file write from the
# copy the session read before the peer's commit then overwrote that commit,
# with no deny anywhere after the shell read (Cohexa-ai/agent-coherence#275).
# The gate now also denies a SHARED holder whose last observed version is
# behind the current one. A retried Bash command or a Read records the
# observation and lifts it, so the way back the deny invites stays open (after a
# Grep deny only a Read does: a Grep never showed the file).

_DENIED_SHELL_READS = [
    ("/hooks/pre-bash", {"command": "cat plan.md"}),
    ("/hooks/pre-bash", {"command": "head -n 5 plan.md"}),
    ("/hooks/pre-bash", {"command": "grep v1 plan.md"}),
    ("/hooks/pre-bash", {"command": "sed -n 1p plan.md"}),
    ("/hooks/pre-grep", {"search_root": ""}),
]


def _admitted(body: dict) -> bool:
    """A pre-edit that let the edit through: no refusal, no deny envelope (an
    admit may still carry a context-only collision advisory)."""
    decision = (body.get("hookSpecificOutput") or {}).get("permissionDecision")
    return body.get("ok", True) is True and decision != "deny"


@pytest.mark.parametrize(
    ("route", "extra"), _DENIED_SHELL_READS,
    ids=["cat", "head", "grep", "sed-n", "grep-tool"],
)
def test_a_write_after_a_denied_shell_read_is_denied_until_the_session_reads(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
    route: str, extra: dict,
) -> None:
    """A read v1, B committed v2, and A's shell read of plan.md was denied, so
    A never saw v2. A's next pre-edit must be the strict deny naming the write
    A has not read (prior v1, current v2). Before the fix it was admitted, and
    A's whole-file write from its v1 copy replaced B's commit. A Read then
    records v2 and the same pre-edit is admitted."""
    _setup_stale(strict_client, "plan.md")
    status, body = strict_client.post(route, {"session_id": _sid("A"), **extra})
    assert status == 200
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    # Asserted, not assumed: the deny re-armed SHARED and recorded no read.
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.SHARED
    assert _observed(strict_coordinator, "plan.md", "A") == 1

    status, edit = strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"},
    )
    assert status == 200
    assert edit["ok"] is False, edit
    assert edit["hookSpecificOutput"]["permissionDecision"] == "deny", edit
    assert edit["summary"]["prior_version_seen_by_session"] == 1, edit
    assert edit["summary"]["current_version"] == 2, edit
    assert _mesi(strict_coordinator, "plan.md", "A") == MESIState.SHARED, (
        "the deny takes nothing: A keeps the re-armed grant its Read relies on"
    )

    _, read = strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("A"), "path": "plan.md", "content_hash": _hash("v2")},
    )
    assert read["status"] == "fresh", read
    _, edit = strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"},
    )
    assert _admitted(edit), edit


def test_a_write_after_the_retry_a_denied_bash_invites_is_admitted(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The other way back: A runs the denied ``cat`` again. The grant is held,
    so the command runs and A reads v2, and its next pre-edit is admitted. A
    gate that denied every re-armed holder would lock A out here."""
    _setup_stale(strict_client, "plan.md")
    strict_client.post("/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"})
    _, retry = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md"},
    )
    assert retry["status"] == "fresh", retry
    assert _observed(strict_coordinator, "plan.md", "A") == 2

    _, edit = strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("A"), "path": "plan.md"},
    )
    assert _admitted(edit), edit


def test_a_session_that_never_observed_the_path_edits_like_a_first_time_editor(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The deliberate edge. A's only grant on CLAUDE.md came from a denied
    command, so A holds SHARED with no recorded observation at all. Like a
    first-time editor (no state), A has not acted on any version, so the gate,
    which compares an observed version with the current one, admits it. Pinned
    so that a change to this is a decision, not a side effect."""
    strict_client.post(
        "/hooks/pre-read",
        {"session_id": _sid("B"), "path": "CLAUDE.md", "content_hash": _hash("v1")},
    )
    strict_client.post("/hooks/pre-edit", {"session_id": _sid("B"), "path": "CLAUDE.md"})
    strict_client.post(
        "/hooks/post-edit",
        {"session_id": _sid("B"), "path": "CLAUDE.md", "content_hash": _hash("v2"), "success": True},
    )
    _setup_stale(strict_client, "plan.md")
    _, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md CLAUDE.md"},
    )
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    assert _mesi(strict_coordinator, "CLAUDE.md", "A") == MESIState.SHARED
    assert _observed(strict_coordinator, "CLAUDE.md", "A") is None

    _, edit = strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("A"), "path": "CLAUDE.md"},
    )
    assert _admitted(edit), edit


def test_a_warn_path_re_armed_by_a_denied_command_still_edits(
    strict_coordinator: CoordinatorHTTPServer, strict_client: _Client,
) -> None:
    """The gate is strict-only. A denied command that names a strict path and a
    warn-only path re-arms SHARED on both without an observation; the warn-only
    path's next pre-edit is admitted as it always was. Fails if the new check is
    hoisted out of the strict branch and starts refusing warn-mode edits."""
    _setup_stale(strict_client, "plan.md")
    _setup_stale(strict_client, "docs/plans/x.md")
    _, body = strict_client.post(
        "/hooks/pre-bash", {"session_id": _sid("A"), "command": "cat plan.md docs/plans/x.md"},
    )
    assert body["hookSpecificOutput"]["permissionDecision"] == "deny", body
    assert _mesi(strict_coordinator, "docs/plans/x.md", "A") == MESIState.SHARED
    assert _observed(strict_coordinator, "docs/plans/x.md", "A") == 1

    _, edit = strict_client.post(
        "/hooks/pre-edit", {"session_id": _sid("A"), "path": "docs/plans/x.md"},
    )
    assert _admitted(edit), edit
