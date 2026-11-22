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
- Refusing an edit of a held strict file (#519), and a stale strict reader
  reading the file again after one deny (#514) -- both real hook clients,
  this checkout's Python one and the plugin's Node one, one process per
  event, against a coordinator of the test's own

Rows that run the plugin's Node hook client are marked ``protocol_corpus``:
they need a built plugin, whose ``dist/coordinator.js`` the
``AGENT_COHERENCE_PLUGIN_DIST_PATH`` variable names (the client is
``hook_client.js`` beside it), so they run in the job that builds the paired
plugin branch, or locally with
``AGENT_COHERENCE_PLUGIN_DIST_PATH=<plugin>/dist/coordinator.js pytest -m
protocol_corpus tests/test_claude_code_e2e.py``. Without a resolvable dist
they xfail, as the Node rows of the parity corpora do.

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
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
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
    read_port_from_file,
    stop_coordinator,
)
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli import coherence_coordinator
from ccs.cli._coherence_client import get, post, resolve_endpoint
from ccs.core.exceptions import StaleView
from tests.pid_lock_helpers import write_coordinator_pid_file
from tests.protocol_corpus.harness import (
    _wait_for_coordinator,
    apply_setup,
    normalize_response,
    resolve_node_dist_path,
)

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
# Refusing an edit of a held strict file (#519), and a stale strict reader
# reading the file again after one deny (#514), through the real hook-client
# process. Each test starts its own coordinator in a fresh workspace and runs
# a hook client as Claude Code runs a command hook: one process per event,
# the payload on stdin. Each row runs ``python -m
# ccs.cli.coherence_hook_client`` and the plugin's Node client, and compares
# what they print parsed, because the two space their JSON differently. The
# rows written once below as ``check_*`` functions run here against the
# Python coordinator, and in tests/protocol_corpus/test_hook_client_on_held.py
# against the Node coordinator, so both are held to the same deny texts.
# ----------------------------------------------------------------------

PLAN = "docs/plan.md"
NOTES = "docs/notes.md"
DATA_GLOB = "data/**"
DATA_PLAN = "data/plan.md"
#: What an admitted edit prints, parsed.
ADMIT: dict[str, Any] = {"ok": True}
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
#: The stderr lines both hook clients print, as the shared fixture pins them.
_STDERR_LINES: dict[str, str] = json.loads(
    (Path(__file__).parent / "fixtures" / "hook_on_held" / "stderr_lines.json").read_text(encoding="utf-8")
)["lines"]
_HOOK_MODULE = "ccs.cli.coherence_hook_client"
#: Far above one hook process on a loaded machine; no row asserts a time bound.
_HOOK_TIMEOUT_SEC = 300.0
#: Far above one coordinator start on a loaded machine.
_COORDINATOR_START_TIMEOUT_SEC = 60.0
#: Far above one coordinator stop, its drain and store close included.
_WORKER_EXIT_TIMEOUT_SEC = 30.0
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
    "Stale read denied: {path} was updated by agent {writer} at <TS>. Read {path} again with "
    "the Read tool: the next Read returns the current version, and edits of {path} are denied "
    "until you have read it."
)
_GRANT_CHANGE_DENY_REASON = (
    "Stale read denied: your grant on {path} was revoked and no new version was committed "
    "— {path} is still at v{version}. Read {path} again with the Read tool: the next Read "
    "returns the current version, and edits of {path} are denied until you have read it."
)
_CLAIM_REFUSED_LINE = (
    "agent-coherence-hook-client: caller principal refused: this session is already bound under a "
    "different mint nonce; proceeding without a principal and NOT re-minting (routes that require one "
    "will refuse this session)\n"
)
#: What each client prints on stderr for an opted-in edit whose session
#: another claimant bound first: the refused claim, and, from the Node client
#: only, the recovery it gave up on.
_PRINCIPAL_REFUSED_STDERR = {
    "python": _CLAIM_REFUSED_LINE,
    "node": _CLAIM_REFUSED_LINE + (
        "agent-coherence-hook-client: coordinator refused this hook's caller principal "
        "(caller_principal_absent); this session is bound under a different mint nonce, so nothing "
        "was re-minted\n"
    ),
}
_OUT_OF_BAND_DENY_REASON = (
    "Stale read denied: {path} on disk does not match v{version}, the last version committed "
    "through the coordinator, and no session committed the change, so it was made outside the "
    "coordinator. Do not restore or overwrite {path}: the change may be your user's work. Tell "
    "your user that {path} changed outside the coordinator; the read stays denied until the "
    "change is committed through the coordinator. This denial is structural (v0.2 strict mode); "
    "retrying the same operation will produce the same denial."
)


@dataclass(frozen=True)
class HookRun:
    """What one hook process printed."""

    stdout: str
    stderr: str


@dataclass(frozen=True)
class HookClient:
    """A real hook client, started as Claude Code starts a command hook: one
    process per event, the payload on stdin."""

    name: str
    command: tuple[str, ...]

    def run(self, root: Path, subcommand: str, cc: dict[str, Any]) -> HookRun:
        """One hook process for ``subcommand`` with ``cc`` on its stdin."""
        done = subprocess.run(
            [*self.command, subcommand, "--root", str(root)],
            input=json.dumps(cc), capture_output=True, text=True, cwd=root,
            env={**os.environ, "PYTHONPATH": str(_SRC)}, timeout=_HOOK_TIMEOUT_SEC,
        )
        assert done.returncode == 0, done
        return HookRun(done.stdout, done.stderr)

    def read(self, root: Path, caller: Caller, rel: str = PLAN) -> HookRun:
        return self.run(root, "pre-read", caller.payload(root, rel))

    def edit(self, root: Path, caller: Caller, rel: str = PLAN) -> HookRun:
        return self.run(root, "pre-edit", caller.payload(root, rel))

    def take(self, root: Path, caller: Caller, rel: str = PLAN) -> None:
        """``caller``'s edit of the free file ``rel`` is admitted with no
        other word, so it holds the file."""
        assert_hook_output(self.edit(root, caller, rel), ADMIT)


#: This checkout's Python hook client.
PYTHON_CLIENT = HookClient("python", (sys.executable, "-m", _HOOK_MODULE))


def hook_client(name: str) -> HookClient:
    """The ``python`` client, or the plugin's ``node`` client, which is built
    beside the Node coordinator's dist."""
    if name == "python":
        return PYTHON_CLIENT
    dist = resolve_node_dist_path()
    if dist is None:
        pytest.xfail(
            "Node hook client not resolvable. Build the plugin checkout "
            "(npm ci && npm run build) or set AGENT_COHERENCE_PLUGIN_DIST_PATH."
        )
    client = dist.parent / "hook_client.js"
    assert client.is_file(), f"the Node hook client is not built at {client}"
    return HookClient("node", ("node", str(client)))


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
    """One Python hook-client process for ``subcommand`` with ``cc`` on its stdin."""
    return PYTHON_CLIENT.run(root, subcommand, cc)


@contextmanager
def coordinator_process(root: Path, command: Sequence[str]) -> Iterator[subprocess.Popen[bytes]]:
    """A coordinator for ``root`` that this test starts as its own child
    process, so that a row can SIGKILL it: a clean stop takes the port out of
    ``server.pid``, and a kill leaves the file naming a port nothing serves.
    Killed and reaped on the way out if it still runs."""
    env = {**os.environ, "PYTHONPATH": str(_SRC), "AGENT_COHERENCE_WORKSPACE": str(root)}
    with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            _wait_for_coordinator(
                root, timeout_sec=_COORDINATOR_START_TIMEOUT_SEC, proc=process, backend_label=Path(command[0]).name,
            )
            yield process
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def write_holders(root: Path, rel: str) -> set[UUID]:
    """Who the coordinator's ``/status`` lists as holding ``rel`` for writing."""
    status = get(resolve_endpoint(root), "/status")
    return {
        UUID(row["agent_id"]) for row in status["sessions"]
        if row.get("states", {}).get(rel) in _WRITE_STATES
    }


def grant_holders(root: Path, rel: str) -> set[UUID]:
    """Who the coordinator's ``/status`` lists with any grant on ``rel``. It
    lists no INVALID row, so a stale copy is absent."""
    status = get(resolve_endpoint(root), "/status")
    return {UUID(row["agent_id"]) for row in status["sessions"] if rel in row.get("states", {})}


def _deny(deny: dict[str, Any]) -> dict[str, Any]:
    return {"hookSpecificOutput": deny}


def held_deny(rel: str, holder: UUID) -> dict[str, Any]:
    """What the hook prints, parsed, for an edit of ``rel`` refused because
    ``holder`` holds it."""
    return _deny(hook_payloads.emit_held_path_deny(path=rel, holder_agent_id=holder.hex))


def parent_deny(rel: str) -> dict[str, Any]:
    return _deny(hook_payloads.emit_held_by_parent_deny(path=rel))


def unavailable_deny(rel: str) -> dict[str, Any]:
    return _deny(hook_payloads.emit_admission_unavailable_deny(path=rel))


def assert_hook_output(run: HookRun, expected: dict[str, Any], stderr: str = "") -> None:
    """``run`` printed one line that parses to ``expected``, so a deny's
    reason is compared as exact text, and ``stderr`` exactly. Parsed, not as
    bytes: the Python client prints ``{"ok": true}`` and the Node one
    ``{"ok":true}``."""
    assert run.stdout.endswith("\n") and (json.loads(run.stdout), run.stderr) == (expected, stderr), run


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


def check_an_edit_of_a_free_strict_file_is_admitted(root: Path, client: HookClient) -> None:
    """B's edit of the strict file nobody holds is admitted with no other
    word, and B holds the file."""
    editor = Caller.new()
    client.take(root, editor)
    assert write_holders(root, PLAN) == {editor.agent_id}


def check_a_held_strict_file_is_refused_naming_its_holder(
    root: Path, client: HookClient, editor_read: bool,
) -> tuple[Caller, Caller]:
    """A takes the strict file. B, whether it read the file during the hold
    or never read it, is denied naming A, and A still holds the file.
    Returns A and B."""
    holder, editor = Caller.new(), Caller.new()
    client.take(root, holder)
    if editor_read:
        read = client.read(root, editor)
        assert _decision(read) != "deny", f"control: the read during the hold is admitted: {read}"
    assert_hook_output(client.edit(root, editor), held_deny(PLAN, holder.agent_id))
    assert write_holders(root, PLAN) == {holder.agent_id}
    return holder, editor


def check_a_holder_that_stops_without_committing_frees_the_file(root: Path, client: HookClient) -> None:
    """A lets go of the file by ending its session, with no commit; B's next
    edit is admitted and B holds the file."""
    holder, editor = check_a_held_strict_file_is_refused_naming_its_holder(root, client, editor_read=False)
    _ends_ok(client.run(root, "session-stop", holder.payload(root)))
    client.take(root, editor)
    assert write_holders(root, PLAN) == {editor.agent_id}


def check_an_edit_of_a_held_file_takes_it_as_today(root: Path, client: HookClient, rel: str) -> None:
    """B's edit of ``rel`` while A holds it is admitted with today's
    concurrent-edit note naming A, and B takes the file from A, as today."""
    holder, editor = Caller.new(), Caller.new()
    client.take(root, holder, rel)
    run = client.edit(root, editor, rel)
    assert run.stderr == ""
    assert normalize_response(json.loads(run.stdout)) == todays_collision(rel, holder.agent_id)
    assert write_holders(root, rel) == {editor.agent_id}


def check_an_unknown_setting_takes_a_held_strict_file_as_today(root: Path, client: HookClient) -> None:
    """A takes the strict file with the setting on; then the setting holds a
    value that is neither refuse nor displace. B's edit is today's,
    admitted with the concurrent-edit note naming A, its one stderr line the
    one the shared fixture pins, and B takes the file from A."""
    holder, editor = Caller.new(), Caller.new()
    client.take(root, holder)
    set_hold_refusal(root, "yes\n")
    run = client.edit(root, editor)
    assert run.stderr == _STDERR_LINES["setting_unknown"], run
    assert normalize_response(json.loads(run.stdout)) == todays_collision(PLAN, holder.agent_id)
    assert write_holders(root, PLAN) == {editor.agent_id}


def check_a_subagent_refused_by_its_parents_hold_is_told_to_report_back(root: Path, client: HookClient) -> None:
    parent = Caller.new()
    client.take(root, parent)
    assert_hook_output(client.edit(root, parent.subagent_of("sub1")), parent_deny(PLAN))
    assert write_holders(root, PLAN) == {parent.agent_id}


def check_a_sibling_subagent_is_refused_naming_the_holding_subagent(root: Path, client: HookClient) -> None:
    """``sub1`` holds the file and its sibling ``sub2`` edits it: the holder
    deny naming ``sub1``'s own id, not the deny meant for its parent's hold."""
    session = Caller.new()
    first, sibling = session.subagent_of("sub1"), session.subagent_of("sub2")
    client.take(root, first)
    assert_hook_output(client.edit(root, sibling), held_deny(PLAN, first.agent_id))
    assert write_holders(root, PLAN) == {first.agent_id}


def check_an_edit_after_the_coordinator_is_killed_is_denied(
    root: Path, client: HookClient, coordinator: subprocess.Popen[bytes],
) -> None:
    """``coordinator`` admits an edit, then is SIGKILLed, so ``server.pid``
    still names its port. B's edit of the strict file is denied as one the
    coordinator could not confirm free; with the setting off, the same edit
    prints ``{}`` as today."""
    pid_file = root / ".coherence" / "server.pid"
    client.take(root, Caller.new())
    named = pid_file.read_text()
    coordinator.kill()
    coordinator.wait()
    assert pid_file.read_text() == named, "control: the kill left server.pid naming the coordinator's port"

    editor = Caller.new()
    assert_hook_output(client.edit(root, editor), unavailable_deny(PLAN))
    set_hold_refusal(root, None)
    assert_hook_output(client.edit(root, editor), {})


def committed_deny(rel: str, writer: UUID) -> str:
    """The stale deny naming ``writer``'s commit, as normalize_response leaves it."""
    return _STALE_DENY_REASON.format(path=rel, writer=writer.hex[:8])


def out_of_band_deny(rel: str, version: int) -> str:
    """The deny of a read whose bytes are not ``rel``'s recorded v``version``."""
    return _OUT_OF_BAND_DENY_REASON.format(path=rel, version=version)


def assert_denied(run: HookRun, reason: str) -> None:
    output = _comparable(run)["hookSpecificOutput"]
    assert (output["permissionDecision"], output["permissionDecisionReason"], run.stderr) == ("deny", reason, ""), run


def assert_read_admitted(run: HookRun, version: int) -> None:
    """A fresh Read at ``version``, with no stale warning and no note."""
    assert (json.loads(run.stdout), run.stderr) == ({"status": "fresh", "version": version}, ""), run


def assert_edit_admitted(run: HookRun) -> None:
    assert json.loads(run.stdout).get("ok") is True and _decision(run) != "deny" and run.stderr == "", run


def commit(root: Path, client: HookClient, writer: Caller, content: str) -> None:
    """``writer`` edits ``docs/plan.md`` and commits ``content``: the Edit's
    pre-edit, the bytes it writes, its post-edit."""
    assert_edit_admitted(client.edit(root, writer))
    (root / PLAN).write_text(content)
    _ends_ok(client.run(root, "post-edit", writer.payload(root)))


def read_then_peer_commit(root: Path, client: HookClient) -> tuple[Caller, Caller]:
    """B reads v1 and A commits v2, which makes B's copy stale. Returns B and A."""
    set_hold_refusal(root, None)
    reader, writer = Caller.new(), Caller.new()
    assert_read_admitted(client.read(root, reader), 1)
    commit(root, client, writer, "docs/plan.md v2\n")
    return reader, writer


def check_a_stale_reader_recovers_after_a_peer_commit(root: Path, client: HookClient) -> None:
    """B read v1 and A committed v2. B's Read is denied naming A's commit, and
    so is the Edit B sends before reading again; B's next Read is admitted at
    v2, and its Edit after that."""
    reader, writer = read_then_peer_commit(root, client)
    reason = committed_deny(PLAN, writer.agent_id)
    assert_denied(client.read(root, reader), reason)
    assert_denied(client.edit(root, reader), reason)
    assert_read_admitted(client.read(root, reader), 2)
    assert_edit_admitted(client.edit(root, reader))


def check_a_stale_editor_recovers_by_reading(root: Path, client: HookClient) -> None:
    """The Edit-first order: B's first call after A's commit is an Edit, denied
    naming A; B's next Read is admitted at v2, one deny and not two, and its
    Edit after that."""
    reader, writer = read_then_peer_commit(root, client)
    assert_denied(client.edit(root, reader), committed_deny(PLAN, writer.agent_id))
    assert_read_admitted(client.read(root, reader), 2)
    assert_edit_admitted(client.edit(root, reader))


def check_a_parent_recovers_after_its_subagent_commits(root: Path, client: HookClient) -> None:
    """A session read v1, and its own subagent committed v2 and stopped. The
    session's Read is denied naming the subagent; its next Read is admitted
    at v2, and its Edit after that."""
    set_hold_refusal(root, None)
    parent = Caller.new()
    subagent = parent.subagent_of("sub1")
    assert_read_admitted(client.read(root, parent), 1)
    commit(root, client, subagent, "docs/plan.md v2 by the subagent\n")
    _ends_ok(client.run(root, "subagent-stop", subagent.payload(root)))

    assert_denied(client.read(root, parent), committed_deny(PLAN, subagent.agent_id))
    assert_read_admitted(client.read(root, parent), 2)
    assert_edit_admitted(client.edit(root, parent))


def check_a_reader_recovers_after_a_grant_taken_and_given_back_uncommitted(root: Path, client: HookClient) -> None:
    """B read v1; A took the file and ended its session with no commit, which
    revoked B's grant at an unchanged v1. B's Read is denied with the
    grant-change text; its next Read is admitted at v1, and its Edit after
    that."""
    set_hold_refusal(root, None)
    reader, taker = Caller.new(), Caller.new()
    assert_read_admitted(client.read(root, reader), 1)
    assert_edit_admitted(client.edit(root, taker))
    _ends_ok(client.run(root, "session-stop", taker.payload(root)))

    assert_denied(client.read(root, reader), _GRANT_CHANGE_DENY_REASON.format(path=PLAN, version=1))
    assert_read_admitted(client.read(root, reader), 1)
    assert_edit_admitted(client.edit(root, reader))


def check_a_change_outside_the_coordinator_after_a_deny_stays_denied(root: Path, client: HookClient) -> None:
    """B was denied after A's commit, and the file then changed with no
    commit. B's Read of those bytes is denied with the out-of-band text, which
    names a commit through the coordinator as what clears it, and B's Edit
    stays denied."""
    reader, writer = read_then_peer_commit(root, client)
    assert_denied(client.read(root, reader), committed_deny(PLAN, writer.agent_id))
    (root / PLAN).write_text("docs/plan.md changed outside the coordinator\n")

    assert_denied(client.read(root, reader), out_of_band_deny(PLAN, 2))
    assert_denied(client.edit(root, reader), committed_deny(PLAN, writer.agent_id))


def check_a_missing_file_after_a_deny_stays_denied(root: Path, client: HookClient) -> None:
    """B was denied after A's commit, and the file was then deleted. B's Read,
    which carries no hash, is denied with the out-of-band text, and B's Edit
    stays denied."""
    reader, writer = read_then_peer_commit(root, client)
    assert_denied(client.read(root, reader), committed_deny(PLAN, writer.agent_id))
    (root / PLAN).unlink()

    assert_denied(client.read(root, reader), out_of_band_deny(PLAN, 2))
    assert_denied(client.edit(root, reader), committed_deny(PLAN, writer.agent_id))


def check_a_session_that_never_read_a_changed_file_stays_denied(root: Path, client: HookClient) -> None:
    """The control: A read v1 and the file then changed with no commit. C,
    which never read it, is denied with the out-of-band text, and its second
    Read is denied with it too: C has no stale copy to recover, so its deny
    clears only with a commit through the coordinator."""
    set_hold_refusal(root, None)
    first, stranger = Caller.new(), Caller.new()
    assert_read_admitted(client.read(root, first), 1)
    (root / PLAN).write_text("docs/plan.md changed outside the coordinator\n")

    for _ in range(2):
        assert_denied(client.read(root, stranger), out_of_band_deny(PLAN, 1))


def check_a_refused_reader_recovers_after_the_holder_commits(
    root: Path, first: HookClient, later: HookClient,
) -> None:
    """#514 with the held-file refusal on: B read the file during A's hold and
    ``first`` refused B's edit; A commits and ends its session. B's calls
    after that run through ``later``: one stale deny naming A, then an
    admitted Read at v2 and an admitted Edit, and B holds the file."""
    holder, editor = check_a_held_strict_file_is_refused_naming_its_holder(root, first, editor_read=True)
    (root / PLAN).write_text("docs/plan.md v2\n")
    _ends_ok(run_hook(root, "post-edit", holder.payload(root)))
    _ends_ok(run_hook(root, "session-stop", holder.payload(root)))

    assert_denied(later.read(root, editor), committed_deny(PLAN, holder.agent_id))
    assert_read_admitted(later.read(root, editor), 2)
    assert_edit_admitted(later.edit(root, editor))
    assert write_holders(root, PLAN) == {editor.agent_id}


def check_plain_volume_reads_after_a_peer_commit_stay_denied(root: Path) -> None:
    """A ``CoherentVolume`` is not a hook client: the reader's view went stale
    on the writer's commit, and two plain ``read()`` calls leave the reader
    with no grant on the coordinator. A ``write()`` of a buffer the reader
    built from v1 is then refused, and the writer's bytes stay. The volume's
    recovery stays ``reacquire()``."""
    writer = CoherentVolume(root, managed=(DATA_GLOB,), config=VOLUME_CONFIG)
    reader = CoherentVolume(root, managed=(DATA_GLOB,), config=VOLUME_CONFIG)
    try:
        stale_buffer = reader.read(DATA_PLAN) + b"appended from v1\n"
        writer.write(DATA_PLAN, b"v2 from the writer\n")

        for _ in range(2):
            assert reader.read(DATA_PLAN) == b"v2 from the writer\n"
            assert UUID(reader.agent_id) not in grant_holders(root, DATA_PLAN), "a plain read healed the stale view"
        with pytest.raises(StaleView):
            reader.write(DATA_PLAN, stale_buffer)
        assert (root / DATA_PLAN).read_bytes() == b"v2 from the writer\n"
    finally:
        reader.close()
        writer.close()


def check_a_refuse_mode_volume_holding_a_strict_file_is_named(root: Path, client: HookClient) -> None:
    """A ``CoherentVolume`` in refuse mode holds a strict file: the hook's
    edit of it is denied naming the volume, which keeps the file and its
    bytes."""
    volume = CoherentVolume(root, managed=(DATA_GLOB,), config=VOLUME_CONFIG, on_held="refuse")
    try:
        volume.write(DATA_PLAN, b"from the volume\n")
        holder = UUID(volume.agent_id)
        assert write_holders(root, DATA_PLAN) == {holder}, "control: the volume holds the file"
        assert_hook_output(client.edit(root, Caller.new(), DATA_PLAN), held_deny(DATA_PLAN, holder))
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


#: Both hook clients. The Node one needs the plugin's dist, so its rows run
#: only where the corpus does (see the module docstring).
HOOK_CLIENTS = [
    pytest.param("python", id="python-client"),
    pytest.param("node", id="node-client", marks=pytest.mark.protocol_corpus),
]


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_an_edit_of_a_strict_file_nobody_holds_is_admitted(strict_workspace: Path, client: str) -> None:
    """Prevents the refusal denying an edit no other agent stands in the way of."""
    check_an_edit_of_a_free_strict_file_is_admitted(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
@pytest.mark.parametrize("editor_read", [False, True], ids=["editor-never-read", "editor-read-during-the-hold"])
def test_an_edit_of_a_strict_file_another_session_holds_is_denied_naming_the_holder(
    strict_workspace: Path, editor_read: bool, client: str,
) -> None:
    """Prevents a second session taking a strict file from its holder once
    the workspace asks the hook to refuse held files."""
    check_a_held_strict_file_is_refused_naming_its_holder(strict_workspace, hook_client(client), editor_read)


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_holder_that_stops_without_committing_frees_the_file_for_the_refused_editor(
    strict_workspace: Path, client: str,
) -> None:
    """Prevents a refusal that outlives the hold it refused for."""
    check_a_holder_that_stops_without_committing_frees_the_file(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_an_edit_of_a_held_file_that_is_not_strict_takes_it_as_today(strict_workspace: Path, client: str) -> None:
    """Prevents the setting reaching a tracked file that is not strict."""
    check_an_edit_of_a_held_file_takes_it_as_today(strict_workspace, hook_client(client), NOTES)


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_subagent_refused_by_its_own_parents_hold_is_told_to_report_back(
    strict_workspace: Path, client: str,
) -> None:
    """Prevents a subagent told to wait on the session that started it."""
    check_a_subagent_refused_by_its_parents_hold_is_told_to_report_back(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_sibling_subagent_is_refused_naming_the_subagent_that_holds_the_file(
    strict_workspace: Path, client: str,
) -> None:
    """Prevents a sibling told the file is held by the session that started it."""
    check_a_sibling_subagent_is_refused_naming_the_holding_subagent(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
@pytest.mark.parametrize("setting", [None, "displace\n"], ids=["setting-absent", "setting-displace"])
def test_with_the_setting_off_an_edit_of_a_held_strict_file_takes_it_as_today(
    strict_workspace: Path, setting: str | None, client: str,
) -> None:
    """Prevents the deny reaching a workspace that did not turn it on."""
    set_hold_refusal(strict_workspace, setting)
    check_an_edit_of_a_held_file_takes_it_as_today(strict_workspace, hook_client(client), PLAN)


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_an_edit_after_the_coordinator_is_killed_is_denied(workspace: Path, client: str) -> None:
    """Prevents an edit admitted unchecked because the coordinator died with
    the file's holder unknown. The coordinator runs as this test's own
    process, so that it can be killed rather than stopped."""
    hook = hook_client(client)
    write_strict_workspace(workspace)
    command = (sys.executable, "-m", "ccs.cli.coherence_coordinator", "--_daemonized", "--root", str(workspace))
    with coordinator_process(workspace, command) as coordinator:
        check_an_edit_after_the_coordinator_is_killed_is_denied(workspace, hook, coordinator)


def test_prepare_for_migration_ends_the_detached_worker(workspace: Path) -> None:
    """Prevents --prepare-for-migration reporting a clean stop while the
    detached worker lives on, holding the workspace lock and its port line,
    so that every strict edit is denied with the refusal on and no session
    start can bring a coordinator up again (#510)."""
    write_strict_workspace(workspace)
    command = (sys.executable, "-m", "ccs.cli.coherence_coordinator", "--_daemonized", "--root", str(workspace))
    with coordinator_process(workspace, command) as worker:
        assert coherence_coordinator.main(["--root", str(workspace), "--prepare-for-migration", "--quiet"]) == 0
        try:
            worker.wait(timeout=_WORKER_EXIT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            pytest.fail(
                f"the detached worker is still alive {_WORKER_EXIT_TIMEOUT_SEC:.0f}s after "
                "prepare-for-migration reported a clean stop"
            )
    assert read_port_from_file(workspace / ".coherence" / "server.pid") is None
    # No coordinator to ask, so the strict edit goes ahead as today.
    assert_hook_output(PYTHON_CLIENT.edit(workspace, Caller.new()), {})
    with coordinator_process(workspace, command):
        PYTHON_CLIENT.take(workspace, Caller.new())


@pytest.mark.parametrize("later", HOOK_CLIENTS)
@pytest.mark.parametrize("first", HOOK_CLIENTS)
def test_a_refused_reader_reads_again_and_edits_after_the_holder_commits(
    strict_workspace: Path, first: str, later: str,
) -> None:
    """Prevents the #514 lockout after a held-file refusal: whichever client
    refused B, B's later calls through either client recover after one deny."""
    check_a_refused_reader_recovers_after_the_holder_commits(strict_workspace, hook_client(first), hook_client(later))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_with_an_unknown_setting_an_edit_of_a_held_strict_file_takes_it_as_today(
    strict_workspace: Path, client: str,
) -> None:
    """Prevents an unknown setting value either refusing or passing unreported:
    today's edit, and the one stderr line the shared fixture pins."""
    check_an_unknown_setting_takes_a_held_strict_file_as_today(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_stale_reader_reads_again_and_edits_after_a_peer_commit(strict_workspace: Path, client: str) -> None:
    """Prevents the #514 lockout: every retry of a stale Read denied, so the
    session could never edit the file again."""
    check_a_stale_reader_recovers_after_a_peer_commit(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_stale_editor_reads_again_and_edits(strict_workspace: Path, client: str) -> None:
    """Prevents a second deny for a session whose first call was the Edit."""
    check_a_stale_editor_recovers_by_reading(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_parent_reads_again_and_edits_after_its_subagent_commits(strict_workspace: Path, client: str) -> None:
    """Prevents the lockout of a session whose own subagent committed."""
    check_a_parent_recovers_after_its_subagent_commits(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_reader_reads_again_and_edits_after_a_grant_given_back_uncommitted(
    strict_workspace: Path, client: str,
) -> None:
    """Prevents the lockout of a reader whose grant was revoked with no commit."""
    check_a_reader_recovers_after_a_grant_taken_and_given_back_uncommitted(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_change_outside_the_coordinator_after_a_deny_stays_denied(strict_workspace: Path, client: str) -> None:
    """Prevents a Read admitting bytes no version records, which the next Edit
    would build on."""
    check_a_change_outside_the_coordinator_after_a_deny_stays_denied(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_missing_file_after_a_deny_stays_denied(strict_workspace: Path, client: str) -> None:
    """Prevents a Read with no hash counting as a read of the recorded version."""
    check_a_missing_file_after_a_deny_stays_denied(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_a_session_that_never_read_a_changed_file_stays_denied(strict_workspace: Path, client: str) -> None:
    """Prevents a deny with no stale copy behind it from clearing on a Read."""
    check_a_session_that_never_read_a_changed_file_stays_denied(strict_workspace, hook_client(client))


def test_plain_volume_reads_after_a_peer_commit_stay_denied(strict_workspace: Path) -> None:
    """Prevents a plain volume read healing a stale view, so that a write of an
    older buffer would be admitted."""
    check_plain_volume_reads_after_a_peer_commit_stay_denied(strict_workspace)


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_an_edit_of_a_strict_file_a_refuse_mode_volume_holds_is_denied_naming_it(
    strict_workspace: Path, client: str,
) -> None:
    """Prevents the hook taking a file a refuse-mode volume holds."""
    check_a_refuse_mode_volume_holding_a_strict_file_is_named(strict_workspace, hook_client(client))


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_an_edit_whose_caller_principal_is_refused_beyond_recovery_is_denied(
    strict_workspace: Path, client: str,
) -> None:
    """Another claimant bound B's session first, so B's opted-in edit is
    refused for its principal even after the recovery: the unavailable deny,
    where today's path prints ``{}``. Each client reports its own recovery on
    stderr. Python coordinator only: the Node coordinator issues no caller
    principal."""
    hook = hook_client(client)
    editor = Caller.new()
    claim = {"session_id": editor.session_id, "mint_nonce": secrets.token_urlsafe(24)}
    assert post(resolve_endpoint(strict_workspace), "/principal/claim", claim).get("ok") is True

    refused = hook.edit(strict_workspace, editor)
    set_hold_refusal(strict_workspace, None)
    today = hook.edit(strict_workspace, editor)

    assert_hook_output(refused, unavailable_deny(PLAN), _PRINCIPAL_REFUSED_STDERR[client])
    assert json.loads(today.stdout) == {}


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_an_edit_the_coordinator_cannot_decide_in_time_is_denied(
    strict_workspace: Path, strict_server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch, client: str,
) -> None:
    """The coordinator's watchdog answers before the edit's check runs: the
    unavailable deny, where today's path admits with a note that the edit
    proceeds unchecked. Python coordinator only: the Node coordinator's
    pre-edit has no watchdog."""
    import ccs.adapters.claude_code.coordinator_server as server_mod
    from tests.test_claude_code_coordinator_server import _HeldRegistryLock

    hook = hook_client(client)
    editor = Caller.new()
    # The read claims the session's principal while the registry is free, so
    # the held lock below blocks only the edits' own work.
    hook.read(strict_workspace, editor)
    monkeypatch.setattr(server_mod, "HANDLER_TIMEOUT_SEC", 0.25)
    with _HeldRegistryLock(strict_server):
        refused = hook.edit(strict_workspace, editor)
        set_hold_refusal(strict_workspace, None)
        today = hook.edit(strict_workspace, editor)

    assert_hook_output(refused, unavailable_deny(PLAN))
    assert json.loads(today.stdout) == json.loads(json.dumps(server_mod._PRE_EDIT_DEGRADED_RESPONSE))


async def _edit_beside_an_mcp_holder(root: Path, client: HookClient, logs: Path) -> tuple[HookRun, set[UUID]]:
    """A refuse-mode MCP server writes ``data/plan.md`` and so holds it; the
    hook edits it while the server still runs. Returns the hook's run and
    the holders listed before it."""
    from tests.mcp._stdio_harness import stdio_servers

    async with stdio_servers(root, logs) as start:
        server = await start("holder", on_held="refuse")
        answer = await server.call("swg_write", path=DATA_PLAN, content="from the server\n")
        assert answer.structured == {"ok": True, "path": DATA_PLAN, "opted_in": True}, answer
        holders = write_holders(root, DATA_PLAN)
        run = await asyncio.to_thread(client.edit, root, Caller.new(), DATA_PLAN)
        assert write_holders(root, DATA_PLAN) == holders
    return run, holders


@pytest.mark.parametrize("client", HOOK_CLIENTS)
def test_an_edit_of_a_strict_file_a_refuse_mode_mcp_server_holds_is_denied_naming_it(
    strict_workspace: Path, tmp_path_factory: pytest.TempPathFactory, client: str,
) -> None:
    """Prevents the hook taking a file a refuse-mode MCP server holds. The
    server needs the mcp extra, which the protocol corpus jobs do not install,
    so this row runs here only, and skips its Node-client case there."""
    pytest.importorskip("mcp.server.fastmcp", reason="the MCP server needs the mcp extra")
    hook = hook_client(client)
    logs = tmp_path_factory.mktemp("mcp-logs")
    run, holders = asyncio.run(_edit_beside_an_mcp_holder(strict_workspace, hook, logs))

    (holder,) = holders
    assert_hook_output(run, held_deny(DATA_PLAN, holder))
    assert (strict_workspace / DATA_PLAN).read_text() == "from the server\n"
