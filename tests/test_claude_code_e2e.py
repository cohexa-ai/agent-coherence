# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""End-to-end tests for the Claude Code adapter (Unit 8).

Codifies the AS-phpmac install walkthrough as automated tests. Covers
the scenarios from plan §Unit 8 that can be verified without a live
``claude`` CLI:

- Bootstrap permissions on first spawn (0700/.coherence + 0600/secret)
- Shared-secret Bearer auth (KTD-12) — 401 without, 200 with, 401 with wrong
- DNS-rebinding mitigation — non-127.0.0.1 Host header → 403
- KTD-13 — state.db schema has no `content` column on `artifacts` table
- Coordinator-down graceful degradation — kill the process, next hook no-ops
- Subprocess-spawn integration — full pipeline via real ``agent-coherence-*``
  binaries and a detached coordinator
- Refusing an edit of a held strict file (#519) -- the real hook-client
  process, one per event, against a coordinator of the test's own

Scenarios that require a live ``claude`` CLI session (multi-process
stale-read e2e, claude-agents probe) are exercised by the Phase E.0
probe procedure doc and re-verified manually on each Claude Code
minor-version bump.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import signal
import sqlite3
import stat
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from ccs.adapters.claude_code import hook_payloads
from ccs.adapters.claude_code.coordinator_server import (
    CoordinatorHTTPServer,
    session_to_agent_id,
)
from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import get, post, resolve_endpoint
from tests.pid_lock_helpers import write_coordinator_pid_file
from tests.protocol_corpus.harness import apply_setup, normalize_response

# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=10,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
        spawn_self_probe_attempts=30,
    )


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def live(workspace: Path, fast_cfg: LifecycleConfig):
    """In-process spawned coordinator. Yields (workspace, port, secret)."""
    port = ensure_coordinator(workspace, config=fast_cfg)
    assert port > 0
    secret = (workspace / ".coherence" / "hook.secret").read_text().strip()
    yield workspace, port, secret
    stop_coordinator(workspace)


# ----------------------------------------------------------------------
# Bootstrap permissions (plan §Unit 8 happy-path #7)
# ----------------------------------------------------------------------


def test_bootstrap_creates_coherence_dir_with_0700(
    live,
) -> None:
    """First-spawn creates .coherence/ with mode 0700 (owner rwx only)."""
    workspace, _, _ = live
    coh_dir = workspace / ".coherence"
    assert coh_dir.is_dir()
    mode = stat.S_IMODE(coh_dir.stat().st_mode)
    assert mode == 0o700, f"expected 0700, got {oct(mode)}"


def test_bootstrap_creates_secret_with_0600(live) -> None:
    """hook.secret is mode 0600 per KTD-12 — only owner can read."""
    workspace, _, _ = live
    secret_file = workspace / ".coherence" / "hook.secret"
    assert secret_file.is_file()
    mode = stat.S_IMODE(secret_file.stat().st_mode)
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_bootstrap_creates_state_db(live) -> None:
    """state.db must exist after first spawn."""
    workspace, _, _ = live
    assert (workspace / ".coherence" / "state.db").is_file()


def test_bootstrap_creates_gitignore_with_wildcard(live) -> None:
    """KTD-13: .coherence/.gitignore must contain ``*`` so a careless
    ``git add .`` doesn't accidentally commit state.db (containing
    MESI state + agent UUIDs) or hook.secret (a credential). The README
    claims these are 'auto-gitignored' — this verifies the implementation."""
    workspace, _, _ = live
    gitignore = workspace / ".coherence" / ".gitignore"
    assert gitignore.is_file(), (
        ".coherence/.gitignore missing — `git add .` would commit state.db + secret"
    )
    assert "*" in gitignore.read_text()


def test_bootstrap_gitignore_not_clobbered_on_respawn(workspace: Path, fast_cfg: LifecycleConfig) -> None:
    """If an operator customized .coherence/.gitignore, a respawn must NOT
    overwrite their changes (the implementation only writes when missing)."""
    coh_dir = workspace / ".coherence"
    coh_dir.mkdir(mode=0o700)
    custom_content = "# operator-customized\n*\n# end of custom\n"
    (coh_dir / ".gitignore").write_text(custom_content)
    port = ensure_coordinator(workspace, config=fast_cfg)
    try:
        assert port > 0
        assert (coh_dir / ".gitignore").read_text() == custom_content
    finally:
        stop_coordinator(workspace)


def test_bootstrap_creates_server_pid_with_pid_and_port(live) -> None:
    """server.pid carries <pid>\\n<port>\\n in that order."""
    workspace, port, _ = live
    pid_file = workspace / ".coherence" / "server.pid"
    lines = pid_file.read_text().splitlines()
    assert len(lines) >= 2
    assert int(lines[0]) == os.getpid()
    assert int(lines[1]) == port


# ----------------------------------------------------------------------
# Shared-secret Bearer auth (KTD-12, plan §Unit 8 happy-path #4)
# ----------------------------------------------------------------------


def _request(
    port: int,
    path: str,
    bearer: str | None,
    body: dict[str, Any] | None = None,
    host: str = "127.0.0.1",
) -> tuple[int, str]:
    """Make a raw authenticated request. Returns (status_code, body_text)."""
    headers: dict[str, str] = {"Host": host}
    if bearer is not None:
        headers["Authorization"] = f"Bearer {bearer}"
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        method="POST" if body is not None else "GET",
        headers=headers,
        data=data,
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def test_auth_rejects_request_without_bearer(live) -> None:
    """No Authorization header → 401."""
    _, port, _ = live
    status, body = _request(port, "/status", bearer=None)
    assert status == 401, f"expected 401, got {status}: {body}"


def test_auth_rejects_request_with_wrong_bearer(live) -> None:
    """Wrong token → 401."""
    _, port, _ = live
    status, body = _request(port, "/status", bearer="wrong-token-here")
    assert status == 401


def test_auth_accepts_request_with_correct_bearer(live) -> None:
    """Correct token → 200 + valid JSON status. R12 (Unit 6) + P1 #7
    revision: minimal tier surfaces ``coordinator_pid`` (pid is public
    on POSIX; operators rely on it) but the absolute workspace root
    stays sentinel'd ("."). The full tier (gated behind
    ``Coherence-Local-Operator: true``) is what exposes the absolute
    root."""
    _, port, secret = live
    status, body = _request(port, "/status", bearer=secret)
    assert status == 200
    payload = json.loads(body)
    assert payload["detail"] == "minimal"
    assert "tracked_artifacts" in payload
    # P1 #7: pid is in the minimal tier.
    assert payload["coordinator_pid"] == os.getpid()
    # Absolute root still sentinel'd at this tier.
    assert payload["coordinator_root"] == "."


# ----------------------------------------------------------------------
# DNS-rebinding mitigation (plan §Unit 8 integration security #1)
# ----------------------------------------------------------------------


def test_dns_rebind_request_with_attacker_host_rejected(live) -> None:
    """Host header check: anything other than 127.0.0.1 / localhost → 403.
    Defends against DNS-rebinding attacks where attacker.example.com
    resolves to 127.0.0.1 and a victim's browser makes credentialed
    requests against our local coordinator."""
    _, port, secret = live
    status, body = _request(
        port, "/status", bearer=secret, host="attacker.example.com",
    )
    assert status == 403, f"expected 403, got {status}: {body}"


def test_dns_rebind_localhost_alias_accepted(live) -> None:
    """`localhost` should still be accepted (legitimate alias for 127.0.0.1)."""
    _, port, secret = live
    status, _ = _request(
        port, "/status", bearer=secret, host="localhost",
    )
    assert status == 200


# ----------------------------------------------------------------------
# KTD-13: state.db must NOT have a content column on artifacts (plan
# §Unit 8 integration security #2 — disclosure surface defense)
# ----------------------------------------------------------------------


def test_state_db_artifacts_has_no_content_column(live) -> None:
    """KTD-13: storing raw file content in state.db expands the disclosure
    surface if .coherence/ leaks (e.g. accidentally committed). The
    schema must store only content_hash, never the bytes."""
    workspace, _, _ = live
    db_path = workspace / ".coherence" / "state.db"
    conn = sqlite3.connect(str(db_path))
    try:
        cols = conn.execute("PRAGMA table_info(artifacts)").fetchall()
        col_names = {c[1] for c in cols}
    finally:
        conn.close()
    assert "content" not in col_names, (
        f"KTD-13 violation: artifacts table has 'content' column. "
        f"Columns: {col_names}"
    )
    # And content_hash MUST be present (that's what we DO store)
    assert "content_hash" in col_names


# ----------------------------------------------------------------------
# Coordinator-down graceful degradation (plan §Unit 8 edge case #6)
# ----------------------------------------------------------------------


def test_coordinator_killed_mid_session_hook_client_no_ops(
    workspace: Path, fast_cfg: LifecycleConfig,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """If the coordinator is killed between two hook events, the next
    hook-client invocation must exit 0 with empty JSON (CC's hook
    contract permits silent success). The user's tool call proceeds."""
    import io
    port = ensure_coordinator(workspace, config=fast_cfg)
    assert port > 0
    pid = int((workspace / ".coherence" / "server.pid").read_text().splitlines()[0])

    # Kill the coordinator. In-process coordinator is this same pid — we can't
    # actually kill ourselves. Instead: shut down via stop_coordinator (which
    # is the in-process equivalent) and confirm hook-client degrades.
    stop_coordinator(workspace)
    # And remove the port from the pid file (simulating crash)
    (workspace / ".coherence" / "server.pid").write_text(f"{pid}\n")

    from ccs.cli import coherence_hook_client
    cc_payload = {
        "session_id": str(uuid.uuid4()),
        "tool_name": "Read",
        "tool_input": {"file_path": str(workspace / "any.md")},
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(cc_payload)))
    rc = coherence_hook_client.main(["pre-read", "--root", str(workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "{}"


# ----------------------------------------------------------------------
# Subprocess-spawn integration — what the AS-phpmac walkthrough does
# without a live claude CLI. Verifies the full agent-coherence-* binary
# pipeline as a stranger would invoke it.
# ----------------------------------------------------------------------


def _venv_bin(name: str) -> str:
    """Resolve a console script from this repo's venv."""
    repo_root = Path(__file__).parent.parent
    candidate = repo_root / ".venv" / "bin" / name
    if candidate.is_file():
        return str(candidate)
    # Fallback to PATH lookup (in CI the venv is set up differently)
    import shutil
    found = shutil.which(name)
    assert found is not None, f"binary not on PATH: {name}"
    return found


def test_e2e_subprocess_spawn_then_status(workspace: Path) -> None:
    """The full AS-phpmac chain via real subprocess invocations:
    1. `agent-coherence-coordinator` spawns a detached coordinator
    2. `agent-coherence-track` adds a path to the policy
    3. `agent-coherence-status` shows the policy state

    This is the regression test for the manual smoke that surfaced the
    daemon-thread bug (8015f80) — must run real subprocesses to exercise
    the parent-exits-but-coordinator-survives path."""
    coordinator_bin = _venv_bin("agent-coherence-coordinator")
    status_bin = _venv_bin("agent-coherence-status")
    track_bin = _venv_bin("agent-coherence-track")

    # Step 1: spawn
    spawn = subprocess.run(
        [coordinator_bin, "--root", str(workspace)],
        capture_output=True, text=True, timeout=30,
    )
    assert spawn.returncode == 0, f"spawn failed: {spawn.stderr}"
    assert spawn.stdout.startswith("port="), spawn.stdout
    port = int(spawn.stdout.strip().split("=")[1])

    try:
        # Step 2: track a path
        (workspace / "docs" / "specs").mkdir(parents=True, exist_ok=True)
        (workspace / "docs" / "specs" / "test.md").write_text("v1")
        track = subprocess.run(
            [track_bin, "--root", str(workspace), "docs/specs/test.md"],
            capture_output=True, text=True, timeout=10,
        )
        assert track.returncode == 0, f"track failed: {track.stderr}"

        # Step 3: status
        status = subprocess.run(
            [status_bin, "--root", str(workspace), "--json"],
            capture_output=True, text=True, timeout=10,
        )
        assert status.returncode == 0
        payload = json.loads(status.stdout)
        assert "policy_summary" in payload
        assert payload["policy_summary"]["user_added_pattern_count"] >= 1
    finally:
        # Cleanup — kill the detached coordinator process
        pid_file = workspace / ".coherence" / "server.pid"
        if pid_file.is_file():
            try:
                detached_pid = int(pid_file.read_text().splitlines()[0])
                os.kill(detached_pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, IndexError):
                pass


def test_e2e_subprocess_idempotent_second_spawn(workspace: Path) -> None:
    """Second `agent-coherence-coordinator` invocation must short-circuit
    to the existing port via the live-probe check, not re-fork."""
    coordinator_bin = _venv_bin("agent-coherence-coordinator")
    first = subprocess.run(
        [coordinator_bin, "--root", str(workspace)],
        capture_output=True, text=True, timeout=30,
    )
    assert first.returncode == 0
    port_1 = int(first.stdout.strip().split("=")[1])

    try:
        second = subprocess.run(
            [coordinator_bin, "--root", str(workspace)],
            capture_output=True, text=True, timeout=10,
        )
        assert second.returncode == 0
        port_2 = int(second.stdout.strip().split("=")[1])
        assert port_2 == port_1, "second spawn should reuse existing port"
    finally:
        pid_file = workspace / ".coherence" / "server.pid"
        if pid_file.is_file():
            try:
                detached_pid = int(pid_file.read_text().splitlines()[0])
                os.kill(detached_pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, IndexError):
                pass


# ----------------------------------------------------------------------
# Refusing an edit of a held strict file (#519), through the real
# hook-client process. Each test starts its own coordinator in a fresh
# workspace and runs ``python -m ccs.cli.coherence_hook_client`` as Claude
# Code runs a command hook: one process per event, the payload on stdin.
# The rows written once below as ``check_*`` functions run here against the
# Python coordinator, and in tests/protocol_corpus/test_hook_client_on_held.py
# against the Node coordinator, so both are held to the same deny bytes.
# ----------------------------------------------------------------------

PLAN = "docs/plan.md"
NOTES = "docs/notes.md"
DATA_GLOB = "data/**"
DATA_PLAN = "data/plan.md"
ADMIT_LINE = '{"ok": true}\n'
#: How a volume holder attaches; it never starts a coordinator in these rows.
VOLUME_CONFIG = LifecycleConfig(
    idle_shutdown_sec=0,
    port_file_retry_attempts=20,
    port_file_retry_interval_sec=0.05,
    connect_retry_attempts=10,
    connect_retry_interval_sec=0.05,
)
#: The sources every hook process runs: this checkout's, not an installed copy.
_SRC = Path(__file__).resolve().parents[1] / "src"
_HOOK_MODULE = "ccs.cli.coherence_hook_client"
#: Far above one hook process on a loaded machine; no row asserts a time bound.
_HOOK_TIMEOUT_SEC = 300.0
_WRITE_STATES = ("EXCLUSIVE", "MODIFIED")
#: FROZEN copies of today's texts, never built by the code under test, with
#: their timestamps as normalize_response scrubs them.
_COLLISION_CONTEXT = (
    "\u26a0 Concurrent edit detected at <TS> (UTC): another agent ({holder}) has been "
    "editing {path} since <TS>. Your edit will land in your own worktree, but only one "
    "session's commit will be accepted by the coordinator. Consider waiting for the other "
    "session to finish or coordinating which one should proceed."
)
_STALE_DENY_REASON = (
    "Stale read denied: {path} was updated by agent {writer} at <TS>. Re-read {path} via "
    "the Read tool before proceeding. This denial is structural (v0.2 strict mode); "
    "retrying the same operation will produce the same denial."
)


@dataclass(frozen=True)
class HookRun:
    """What one hook process printed."""

    stdout: str
    stderr: str


@dataclass(frozen=True)
class Caller:
    """A Claude Code session, or its subagent ``subagent``."""

    session_id: str
    subagent: str | None = None

    @classmethod
    def new(cls) -> Caller:
        return cls(str(uuid.uuid4()))

    def subagent_of(self, subagent: str) -> Caller:
        return Caller(self.session_id, subagent)

    @property
    def agent_id(self) -> UUID:
        """The id the coordinator holds files under; a deny names its first 8
        hex characters."""
        return session_to_agent_id(self.session_id, self.subagent)

    def payload(self, root: Path, rel: str = PLAN) -> dict[str, Any]:
        """The hook payload Claude Code sends for a tool call on ``rel``."""
        cc: dict[str, Any] = {
            "session_id": self.session_id,
            "tool_name": "Edit",
            "tool_input": {"file_path": str(root / rel)},
        }
        if self.subagent is not None:
            cc["agent_id"] = self.subagent
        return cc


def write_strict_workspace(root: Path) -> None:
    """``docs/plan.md`` strict, ``docs/notes.md`` tracked but not strict,
    ``data/**`` strict for a volume or an MCP server to manage, and the
    setting on. A coordinator reads its policy once, as it starts, so this
    runs before one starts."""
    apply_setup(root, {
        "tracked": [PLAN, NOTES, DATA_GLOB],
        "strict_mode": [PLAN, DATA_GLOB],
        "files": {rel: f"{rel} v1\n" for rel in (PLAN, NOTES, DATA_PLAN)},
    })
    set_hold_refusal(root, "refuse\n")


def set_hold_refusal(root: Path, content: str | None) -> None:
    """Write the setting file with ``content``, or remove it for ``None``.
    The hook client reads it on every event."""
    setting = root / ".coherence" / "hook_on_held"
    if content is None:
        setting.unlink(missing_ok=True)
    else:
        setting.write_text(content)


def run_hook(root: Path, subcommand: str, cc: dict[str, Any]) -> HookRun:
    """One hook process for ``subcommand`` with ``cc`` on its stdin."""
    done = subprocess.run(
        [sys.executable, "-m", _HOOK_MODULE, subcommand, "--root", str(root)],
        input=json.dumps(cc), capture_output=True, text=True, cwd=root,
        env={**os.environ, "PYTHONPATH": str(_SRC)}, timeout=_HOOK_TIMEOUT_SEC,
    )
    assert done.returncode == 0, done
    return HookRun(done.stdout, done.stderr)


def edit(root: Path, caller: Caller, rel: str = PLAN) -> HookRun:
    return run_hook(root, "pre-edit", caller.payload(root, rel))


def take(root: Path, caller: Caller, rel: str = PLAN) -> None:
    """``caller``'s edit of the free file ``rel`` is admitted with no other
    word, so it holds the file."""
    assert edit(root, caller, rel) == HookRun(ADMIT_LINE, "")


def write_holders(root: Path, rel: str) -> set[UUID]:
    """Who the coordinator's ``/status`` lists as holding ``rel`` for writing."""
    status = get(resolve_endpoint(root), "/status")
    return {
        UUID(row["agent_id"]) for row in status["sessions"]
        if row.get("states", {}).get(rel) in _WRITE_STATES
    }


def _deny_line(deny: dict[str, Any]) -> str:
    return json.dumps({"hookSpecificOutput": deny}) + "\n"


def held_deny(rel: str, holder: UUID) -> HookRun:
    """What the hook prints for an edit of ``rel`` refused because ``holder``
    holds it."""
    return HookRun(_deny_line(hook_payloads.emit_held_path_deny(path=rel, holder_agent_id=holder.hex)), "")


def parent_deny(rel: str) -> HookRun:
    return HookRun(_deny_line(hook_payloads.emit_held_by_parent_deny(path=rel)), "")


def unavailable_deny_line(rel: str) -> str:
    return _deny_line(hook_payloads.emit_admission_unavailable_deny(path=rel))


def todays_collision(rel: str, holder: UUID) -> dict[str, Any]:
    """Today's admit of an edit of ``rel`` while ``holder`` holds it, as
    normalize_response leaves it."""
    context = _COLLISION_CONTEXT.format(path=rel, holder=holder.hex[:8])
    return {
        "collision": True,
        "hookSpecificOutput": {
            "additionalContext": context, "hookEventName": "PreToolUse", "permissionDecision": "allow",
        },
        "ok": True,
    }


def _comparable(run: HookRun) -> dict[str, Any]:
    """A deny's stdout with its timestamps scrubbed, including the time the
    deny was written, which differs on every call."""
    return normalize_response(json.loads(run.stdout), ignore_keys=frozenset({"warning_generated_at_unix_ts"}))


def _decision(run: HookRun) -> object:
    return json.loads(run.stdout).get("hookSpecificOutput", {}).get("permissionDecision")


def _ends_ok(run: HookRun) -> None:
    assert json.loads(run.stdout).get("ok") is True and run.stderr == "", run


def check_a_held_strict_file_is_refused_naming_its_holder(root: Path, editor_read: bool) -> tuple[Caller, Caller]:
    """A takes the strict file. B, whether it read the file during the hold
    or never read it, is denied naming A, and A still holds the file.
    Returns A and B."""
    holder, editor = Caller.new(), Caller.new()
    take(root, holder)
    if editor_read:
        read = run_hook(root, "pre-read", editor.payload(root))
        assert _decision(read) != "deny", f"control: the read during the hold is admitted: {read}"
    assert edit(root, editor) == held_deny(PLAN, holder.agent_id)
    assert write_holders(root, PLAN) == {holder.agent_id}
    return holder, editor


def check_a_holder_that_stops_without_committing_frees_the_file(root: Path) -> None:
    """A lets go of the file by ending its session, with no commit; B's next
    edit is admitted and B holds the file."""
    holder, editor = check_a_held_strict_file_is_refused_naming_its_holder(root, editor_read=False)
    _ends_ok(run_hook(root, "session-stop", holder.payload(root)))
    take(root, editor)
    assert write_holders(root, PLAN) == {editor.agent_id}


def check_an_edit_of_a_held_file_takes_it_as_today(root: Path, rel: str) -> None:
    """B's edit of ``rel`` while A holds it is admitted with today's
    concurrent-edit note naming A, and B takes the file from A, as today."""
    holder, editor = Caller.new(), Caller.new()
    take(root, holder, rel)
    run = edit(root, editor, rel)
    assert run.stderr == ""
    assert normalize_response(json.loads(run.stdout)) == todays_collision(rel, holder.agent_id)
    assert write_holders(root, rel) == {editor.agent_id}


def check_a_subagent_refused_by_its_parents_hold_is_told_to_report_back(root: Path) -> None:
    parent = Caller.new()
    take(root, parent)
    assert edit(root, parent.subagent_of("sub1")) == parent_deny(PLAN)
    assert write_holders(root, PLAN) == {parent.agent_id}


def check_a_sibling_subagent_is_refused_naming_the_holding_subagent(root: Path) -> None:
    """``sub1`` holds the file and its sibling ``sub2`` edits it: the holder
    deny naming ``sub1``'s own id, not the deny meant for its parent's hold."""
    session = Caller.new()
    first, sibling = session.subagent_of("sub1"), session.subagent_of("sub2")
    take(root, first)
    assert edit(root, sibling) == held_deny(PLAN, first.agent_id)
    assert write_holders(root, PLAN) == {first.agent_id}


def check_a_refused_reader_stays_denied_after_the_holder_commits(root: Path) -> None:
    """#514 as it stands today: B read the file during A's hold and was
    refused; A commits and ends its session. B's next read and its edit get
    today's strict stale deny naming A, the edit's deny the one the setting-off
    path prints, so B cannot re-read its way back to an edit."""
    holder, editor = check_a_held_strict_file_is_refused_naming_its_holder(root, editor_read=True)
    (root / PLAN).write_text("docs/plan.md v2\n")
    _ends_ok(run_hook(root, "post-edit", holder.payload(root)))
    _ends_ok(run_hook(root, "session-stop", holder.payload(root)))

    read, opted_in = run_hook(root, "pre-read", editor.payload(root)), edit(root, editor)
    set_hold_refusal(root, None)
    today = edit(root, editor)

    reason = _STALE_DENY_REASON.format(path=PLAN, writer=holder.agent_id.hex[:8])
    for run in (read, opted_in, today):
        output = _comparable(run)["hookSpecificOutput"]
        assert (output["permissionDecision"], output["permissionDecisionReason"], run.stderr) == ("deny", reason, "")
    assert _comparable(opted_in) == _comparable(today)


def check_a_refuse_mode_volume_holding_a_strict_file_is_named(root: Path) -> None:
    """A ``CoherentVolume`` in refuse mode holds a strict file: the hook's
    edit of it is denied naming the volume, which keeps the file and its
    bytes."""
    volume = CoherentVolume(root, managed=(DATA_GLOB,), config=VOLUME_CONFIG, on_held="refuse")
    try:
        volume.write(DATA_PLAN, b"from the volume\n")
        holder = UUID(volume.agent_id)
        assert write_holders(root, DATA_PLAN) == {holder}, "control: the volume holds the file"
        assert edit(root, Caller.new(), DATA_PLAN) == held_deny(DATA_PLAN, holder)
        assert write_holders(root, DATA_PLAN) == {holder}
        assert (root / DATA_PLAN).read_bytes() == b"from the volume\n"
    finally:
        volume.close()


@pytest.fixture
def strict_server(workspace: Path) -> Iterator[CoordinatorHTTPServer]:
    """A Python coordinator in this process serving the strict workspace,
    reached by the hook processes through the usual pid file."""
    write_strict_workspace(workspace)
    server = CoordinatorHTTPServer(workspace, port=0, instance_id="hook-on-held-e2e")
    server.serve_in_thread()
    write_coordinator_pid_file(workspace / ".coherence", server.port, pid=os.getpid())
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def strict_workspace(workspace: Path, strict_server: CoordinatorHTTPServer) -> Path:
    """The workspace with the setting on, served by a Python coordinator."""
    return workspace


@pytest.mark.parametrize("editor_read", [False, True], ids=["editor-never-read", "editor-read-during-the-hold"])
def test_an_edit_of_a_strict_file_another_session_holds_is_denied_naming_the_holder(
    strict_workspace: Path, editor_read: bool,
) -> None:
    """Prevents a second session taking a strict file from its holder once
    the workspace asks the hook to refuse held files."""
    check_a_held_strict_file_is_refused_naming_its_holder(strict_workspace, editor_read)


def test_a_holder_that_stops_without_committing_frees_the_file_for_the_refused_editor(
    strict_workspace: Path,
) -> None:
    """Prevents a refusal that outlives the hold it refused for."""
    check_a_holder_that_stops_without_committing_frees_the_file(strict_workspace)


def test_an_edit_of_a_held_file_that_is_not_strict_takes_it_as_today(strict_workspace: Path) -> None:
    """Prevents the setting reaching a tracked file that is not strict."""
    check_an_edit_of_a_held_file_takes_it_as_today(strict_workspace, NOTES)


def test_a_subagent_refused_by_its_own_parents_hold_is_told_to_report_back(strict_workspace: Path) -> None:
    """Prevents a subagent told to wait on the session that started it."""
    check_a_subagent_refused_by_its_parents_hold_is_told_to_report_back(strict_workspace)


def test_a_sibling_subagent_is_refused_naming_the_subagent_that_holds_the_file(strict_workspace: Path) -> None:
    """Prevents a sibling told the file is held by the session that started it."""
    check_a_sibling_subagent_is_refused_naming_the_holding_subagent(strict_workspace)


@pytest.mark.parametrize("setting", [None, "displace\n"], ids=["setting-absent", "setting-displace"])
def test_with_the_setting_off_an_edit_of_a_held_strict_file_takes_it_as_today(
    strict_workspace: Path, setting: str | None,
) -> None:
    """Prevents the deny reaching a workspace that did not turn it on."""
    set_hold_refusal(strict_workspace, setting)
    check_an_edit_of_a_held_file_takes_it_as_today(strict_workspace, PLAN)


def test_a_refused_reader_stays_denied_after_the_holder_commits_as_today(strict_workspace: Path) -> None:
    """Pins the #514 lockout as today's behaviour: a fix to it changes this row."""
    check_a_refused_reader_stays_denied_after_the_holder_commits(strict_workspace)


def test_an_edit_of_a_strict_file_a_refuse_mode_volume_holds_is_denied_naming_it(strict_workspace: Path) -> None:
    """Prevents the hook taking a file a refuse-mode volume holds."""
    check_a_refuse_mode_volume_holding_a_strict_file_is_named(strict_workspace)


def test_an_edit_whose_caller_principal_is_refused_beyond_recovery_is_denied(strict_workspace: Path) -> None:
    """Another claimant bound B's session first, so B's opted-in edit is
    refused for its principal even after the recovery: the unavailable deny,
    where today's path prints ``{}``. Python coordinator only: the Node
    coordinator issues no caller principal."""
    editor = Caller.new()
    claim = {"session_id": editor.session_id, "mint_nonce": secrets.token_urlsafe(24)}
    assert post(resolve_endpoint(strict_workspace), "/principal/claim", claim).get("ok") is True

    refused = edit(strict_workspace, editor)
    set_hold_refusal(strict_workspace, None)
    today = edit(strict_workspace, editor)

    assert refused.stdout == unavailable_deny_line(PLAN)
    assert "caller principal refused" in refused.stderr
    assert today.stdout == "{}\n"


def test_an_edit_the_coordinator_cannot_decide_in_time_is_denied(
    strict_workspace: Path, strict_server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coordinator's watchdog answers before the edit's check runs: the
    unavailable deny, where today's path admits with a note that the edit
    proceeds unchecked. Python coordinator only: the Node coordinator's
    pre-edit has no watchdog."""
    import ccs.adapters.claude_code.coordinator_server as server_mod
    from tests.test_claude_code_coordinator_server import _HeldRegistryLock

    editor = Caller.new()
    # The read claims the session's principal while the registry is free, so
    # the held lock below blocks only the edits' own work.
    run_hook(strict_workspace, "pre-read", editor.payload(strict_workspace))
    monkeypatch.setattr(server_mod, "HANDLER_TIMEOUT_SEC", 0.25)
    with _HeldRegistryLock(strict_server):
        refused = edit(strict_workspace, editor)
        set_hold_refusal(strict_workspace, None)
        today = edit(strict_workspace, editor)

    assert refused == HookRun(unavailable_deny_line(PLAN), "")
    assert json.loads(today.stdout) == json.loads(json.dumps(server_mod._PRE_EDIT_DEGRADED_RESPONSE))


async def _edit_beside_an_mcp_holder(root: Path, logs: Path) -> tuple[HookRun, set[UUID]]:
    """A refuse-mode MCP server writes ``data/plan.md`` and so holds it; the
    hook edits it while the server still runs. Returns the hook's run and
    the holders listed before it."""
    from tests.mcp._stdio_harness import stdio_servers

    async with stdio_servers(root, logs) as start:
        server = await start("holder", on_held="refuse")
        answer = await server.call("swg_write", path=DATA_PLAN, content="from the server\n")
        assert answer.structured == {"ok": True, "path": DATA_PLAN, "opted_in": True}, answer
        holders = write_holders(root, DATA_PLAN)
        run = await asyncio.to_thread(edit, root, Caller.new(), DATA_PLAN)
        assert write_holders(root, DATA_PLAN) == holders
    return run, holders


def test_an_edit_of_a_strict_file_a_refuse_mode_mcp_server_holds_is_denied_naming_it(
    strict_workspace: Path, tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Prevents the hook taking a file a refuse-mode MCP server holds. The
    server needs the mcp extra, which the protocol corpus jobs do not install,
    so this row runs here only."""
    pytest.importorskip("mcp.server.fastmcp", reason="the MCP server needs the mcp extra")
    run, holders = asyncio.run(_edit_beside_an_mcp_holder(strict_workspace, tmp_path_factory.mktemp("mcp-logs")))

    (holder,) = holders
    assert run == held_deny(DATA_PLAN, holder)
    assert (strict_workspace / DATA_PLAN).read_text() == "from the server\n"
