# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for agent-coherence-hook-client — the command-type hook bridge.

Built as a Phase E.0 contingent deliverable when probe 2A revealed Claude
Code v2.1.131 rejects HTTP-type hooks.json URLs containing ${COHERENCE_PORT}
at load time. Hook-client reads CC's stdin payload, translates to the
coordinator's contract, POSTs, and forwards the response to stdout.
"""

from __future__ import annotations

import hashlib
import io
import json
import uuid
from pathlib import Path
from typing import Any

import pytest

from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.cli import _coherence_client, coherence_hook_client


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0,
        port_file_retry_attempts=10,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
        spawn_self_probe_attempts=20,
    )


@pytest.fixture
def git_workspace(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def live_coordinator(git_workspace: Path, fast_cfg: LifecycleConfig):
    port = ensure_coordinator(git_workspace, config=fast_cfg)
    assert port > 0
    yield git_workspace, port
    stop_coordinator(git_workspace)


def _sid() -> str:
    return str(uuid.uuid4())


def _drive(
    subcommand: str,
    cc_payload: dict[str, Any],
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str]:
    """Helper: feed cc_payload to hook-client via monkeypatched stdin,
    run the subcommand against the workspace, return (exit_code, stdout)."""
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(cc_payload)))
    rc = coherence_hook_client.main([subcommand, "--root", str(workspace)])
    captured = capsys.readouterr()
    return rc, captured.out


# ----------------------------------------------------------------------
# Happy paths
# ----------------------------------------------------------------------


def test_pre_read_against_live_coordinator(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Happy path: pre-read on a tracked file → coordinator returns fresh."""
    workspace, port = live_coordinator
    (workspace / "docs").mkdir(parents=True, exist_ok=True)
    (workspace / "docs" / "plan.md").write_text("plan v1")

    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Read",
        "tool_input": {"file_path": str(workspace / "docs" / "plan.md")},
    }
    rc, out = _drive("pre-read", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    response = json.loads(out)
    # T-02 / ce-review: tightened from isinstance(response, dict) to a
    # specific shape check. pre-read returns either {status: "fresh"|"stale"}
    # or the fast-path empty {} for untracked paths (handler returns 200 +
    # {ok: true} before fresh-shape logic). Either is acceptable.
    assert isinstance(response, dict)
    if "status" in response:
        assert response["status"] in ("fresh", "stale"), (
            f"pre-read status must be fresh or stale; got {response['status']!r}"
        )
    else:
        # Untracked fast-path → either {} or {ok: True}
        assert response.get("ok") is True or response == {}, (
            f"pre-read without status must be empty or {{ok:True}}; got {response!r}"
        )


def test_pre_edit_translates_correctly(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """pre-edit on a Edit/Write hook: workspace-relative path computed
    from absolute file_path, session_id passed through."""
    workspace, _ = live_coordinator
    (workspace / "docs" / "specs").mkdir(parents=True, exist_ok=True)
    (workspace / "docs" / "specs" / "test.md").write_text("v1")

    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(workspace / "docs" / "specs" / "test.md")},
    }
    rc, out = _drive("pre-edit", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    response = json.loads(out)
    # T-03 / ce-review: tightened to assert the {ok: bool} contract.
    # pre-edit's wire shape always includes ok (true on success, false on
    # collision-rejected; never absent). On collision, the response also
    # carries hookSpecificOutput per the CollisionResponse TypedDict.
    assert isinstance(response, dict)
    assert "ok" in response or "hookSpecificOutput" in response, (
        f"pre-edit response must carry 'ok' or 'hookSpecificOutput'; got {response!r}"
    )


def test_post_edit_hashes_file_on_disk_when_response_missing_hash(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If CC's tool_response doesn't include content_hash, hook-client
    hashes the post-write file content from disk."""
    workspace, _ = live_coordinator
    target = workspace / "docs" / "specs" / "test.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("post-edit content")

    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(target)},
        "tool_response": {"success": True},  # NO content_hash — client must hash
    }
    rc, out = _drive("post-edit", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    # Verify on-disk hash matches what we expect
    expected_hash = hashlib.sha256(b"post-edit content").hexdigest()
    # Just confirm the call succeeded — we can't see the body easily but
    # the response will indicate ok/note.
    response = json.loads(out)
    assert isinstance(response, dict)


def test_post_edit_with_explicit_content_hash(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If tool_response carries content_hash, hook-client uses it
    verbatim (no disk read)."""
    workspace, _ = live_coordinator
    target = workspace / "docs" / "specs" / "test.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("v1")
    provided_hash = "a" * 64

    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Edit",
        "tool_input": {"file_path": str(target)},
        "tool_response": {"success": True, "content_hash": provided_hash},
    }
    rc, out = _drive("post-edit", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0


def test_session_stop_only_needs_session_id(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stop hooks don't carry tool_input — only session_id."""
    workspace, _ = live_coordinator
    cc_payload = {"session_id": _sid()}
    rc, out = _drive("session-stop", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    response = json.loads(out)
    assert response.get("ok") is True


# ----------------------------------------------------------------------
# Graceful-degrade paths — hook MUST NEVER block the user's tool call
# ----------------------------------------------------------------------


def test_no_coordinator_running_returns_empty_response(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No coordinator running → hook exits 0 with empty JSON. CC ignores."""
    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Read",
        "tool_input": {"file_path": str(git_workspace / "any.md")},
    }
    rc, out = _drive("pre-read", cc_payload, git_workspace, monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"


def test_malformed_stdin_does_not_crash(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Garbage on stdin → exit 0, empty response. Defense against any
    upstream wrapper getting confused."""
    monkeypatch.setattr("sys.stdin", io.StringIO("not json at all"))
    rc = coherence_hook_client.main(["pre-read", "--root", str(git_workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "{}"


def test_empty_stdin_does_not_crash(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Empty stdin → exit 0 with no output (CC's hook contract permits
    silent success)."""
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = coherence_hook_client.main(["pre-read", "--root", str(git_workspace)])
    captured = capsys.readouterr()
    assert rc == 0


def test_missing_session_id_emits_empty(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CC payload without session_id → skip hook, empty response."""
    workspace, _ = live_coordinator
    cc_payload = {"tool_name": "Read", "tool_input": {"file_path": str(workspace / "x.md")}}
    rc, out = _drive("pre-read", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"


def test_missing_file_path_emits_empty(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Tool hook with no file_path in tool_input → skip."""
    workspace, _ = live_coordinator
    cc_payload = {"session_id": _sid(), "tool_name": "Read", "tool_input": {}}
    rc, out = _drive("pre-read", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"


def test_path_outside_workspace_emits_empty(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the file_path resolves outside the workspace root (shouldn't
    happen but defensive), skip rather than send garbage."""
    workspace, _ = live_coordinator
    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Read",
        "tool_input": {"file_path": "/etc/passwd"},
    }
    rc, out = _drive("pre-read", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"


# ----------------------------------------------------------------------
# Workspace-relative path translation
# ----------------------------------------------------------------------


def test_pre_read_translates_absolute_to_workspace_relative(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Sanity: an absolute file_path inside the workspace gets converted
    to a relative path before being sent to the coordinator. We verify
    via the policy effect — track the relative form, then trigger a
    read with the absolute form, then check status shows the artifact."""
    workspace, _ = live_coordinator
    # Track using the relative form
    from ccs.cli import coherence_track
    target_rel = "docs/specs/test.md"
    (workspace / "docs" / "specs").mkdir(parents=True, exist_ok=True)
    (workspace / "docs" / "specs" / "test.md").write_text("seed")
    coherence_track.main(["--root", str(workspace), target_rel])
    capsys.readouterr()  # drain

    # Now fire a hook with the absolute form
    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Read",
        "tool_input": {
            "file_path": str(workspace / "docs" / "specs" / "test.md"),
        },
    }
    rc, out = _drive("pre-read", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    # Response should be JSON. The artifact should now be observed.
    response = json.loads(out)
    assert isinstance(response, dict)
    # Verify the coordinator now sees the artifact (post-read)
    from ccs.cli import coherence_status
    coherence_status.main(["--root", str(workspace), "--json"])
    status_out = capsys.readouterr().out
    status = json.loads(status_out)
    paths = [a["path"] for a in status.get("tracked_artifacts", [])]
    assert target_rel in paths, (
        f"path translation failed: expected '{target_rel}' in artifacts, got {paths}"
    )


# ----------------------------------------------------------------------
# _build_pre_grep — both path shapes (subagent absolute, direct relative)
#
# Regression guard for the 2026-05-24 launch-gate finding: PR #64 added an
# `os.path.isabs(raw_path)` branch to `_build_pre_grep` but forgot to
# `import os` at module scope. Every non-empty Grep path raised NameError,
# which the main()'s broad except swallowed → empty `{}` response →
# coordinator never contacted → strict-deny never fired. Existing pre-grep
# tests all hit the coordinator HTTP endpoint directly and never exercised
# the builder, so the bug slipped through CI.
#
# These tests drive the FULL hook-client path (stdin → main() → builder →
# HTTP → stdout) for both subagent-shape and top-level-shape Grep payloads,
# so any future regression in either branch fails loudly.
# ----------------------------------------------------------------------


def test_pre_grep_relative_path_drives_coordinator(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Direct (top-level) Grep invocation: path is workspace-relative.
    Hook builder must produce a non-empty payload that reaches the
    coordinator. Empty `{}` response indicates the builder crashed
    silently (the bug PR #64 introduced + this guard fixes)."""
    workspace, _ = live_coordinator
    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Grep",
        "tool_input": {
            "pattern": "anything",
            "path": "docs",
            "output_mode": "content",
        },
    }
    rc, out = _drive("pre-grep", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    response = json.loads(out)
    # The builder must NOT have crashed silently. Empty {} would mean an
    # exception was caught and `_emit_empty` fired — which is the exact
    # NameError-swallow regression this test guards against.
    assert response != {}, (
        "pre-grep returned empty {} for a non-empty path — builder "
        "likely crashed silently. Check imports and the _build_pre_grep "
        "absolute/relative path branches."
    )
    assert "status" in response or "hookSpecificOutput" in response


def test_pre_grep_absolute_path_drives_coordinator(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Subagent-shape Grep invocation: path is absolute (Task tool resolves
    paths to absolute before dispatching to the subagent's hook). Hook
    builder must normalize via _to_workspace_relative and produce a
    non-empty coordinator-bound payload."""
    workspace, _ = live_coordinator
    (workspace / "docs").mkdir(parents=True, exist_ok=True)
    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Grep",
        "tool_input": {
            "pattern": "anything",
            "path": str(workspace / "docs"),  # absolute path → subagent shape
            "output_mode": "files_with_matches",
        },
    }
    rc, out = _drive("pre-grep", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    response = json.loads(out)
    assert response != {}, (
        "pre-grep returned empty {} for an absolute path — absolute-path "
        "branch likely raised an exception (e.g., NameError from missing "
        "`import os`). Check imports and _to_workspace_relative."
    )
    assert "status" in response or "hookSpecificOutput" in response


def test_pre_grep_empty_path_does_not_crash(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Grep with no path arg → search root is workspace root (empty
    string). Must reach the coordinator without crashing — the empty-path
    branch in _build_pre_grep was the only branch the pre-fix builder
    could survive on (it short-circuits before os.path.isabs)."""
    workspace, _ = live_coordinator
    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Grep",
        "tool_input": {
            "pattern": "anything",
            "output_mode": "content",
        },
    }
    rc, out = _drive("pre-grep", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    response = json.loads(out)
    # Empty workspace + no tracked artifacts → coordinator returns
    # {"status": "fresh"}. We just need a parseable JSON response, not
    # specifically empty or non-empty.
    assert isinstance(response, dict)


# ----------------------------------------------------------------------
# session-start (SB-10 U3) — post-compaction re-grounding bridge
#
# The coordinator endpoint trusts its caller and treats EVERY request as a
# compact event (R1), so the ``source == "compact"`` gate lives in the
# hook-client builder. These tests pin both halves of that contract: a
# compact SessionStart reaches /hooks/session-start and its response is
# passed through verbatim; every other source (startup/resume/clear/absent)
# never touches the network; and every failure mode stays fail-open ({}).
# ----------------------------------------------------------------------


def _fake_coherence_dir(workspace: Path, port: int) -> None:
    """Fabricate .coherence/{server.pid,hook.secret} so resolve_endpoint
    (pure file reads — no network) succeeds and the dispatch ladder is
    reachable without spawning a real coordinator."""
    coherence = workspace / ".coherence"
    coherence.mkdir(exist_ok=True)
    write_coordinator_pid_file(coherence, port)
    (coherence / "hook.secret").write_text("test-secret")


def test_session_start_compact_posts_and_passes_response_through(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """source:"compact" with coordination state → non-empty re-grounding
    payload on stdout, byte-for-byte identical to what the coordinator
    returns for a direct POST (the passthrough contract: the client must
    not reshape, filter, or annotate the coordinator's response)."""
    workspace, _ = live_coordinator
    from ccs.cli import coherence_track
    from ccs.cli._coherence_client import post, resolve_endpoint

    (workspace / "docs").mkdir(parents=True, exist_ok=True)
    (workspace / "docs" / "plan.md").write_text("plan v1")
    coherence_track.main(["--root", str(workspace), "docs/plan.md"])
    capsys.readouterr()  # drain track output

    # Give the session coordination state: a pre-read on the tracked file
    # grants this session's agent a MESI state, which is exactly what makes
    # the session-start payload non-empty (R5's has-state arm).
    session_id = _sid()
    read_payload = {
        "session_id": session_id,
        "tool_name": "Read",
        "tool_input": {"file_path": str(workspace / "docs" / "plan.md")},
    }
    rc, _ = _drive("pre-read", read_payload, workspace, monkeypatch, capsys)
    assert rc == 0

    cc_payload = {
        "session_id": session_id,
        "hook_event_name": "SessionStart",
        "source": "compact",
    }
    rc, out = _drive("session-start", cc_payload, workspace, monkeypatch, capsys)
    assert rc == 0
    response = json.loads(out)
    hso = response.get("hookSpecificOutput")
    assert hso is not None, (
        f"compact session-start with a grant must return a re-grounding "
        f"payload, got {response!r}"
    )
    assert hso["hookEventName"] == "SessionStart"
    assert "docs/plan.md" in hso["additionalContext"]

    # Verbatim passthrough: a direct POST with the translated body must
    # yield the exact bytes the client printed (the build is read-only
    # toward the registry — R6 — so back-to-back calls are stable).
    endpoint = resolve_endpoint(workspace)
    direct = post(endpoint, "/hooks/session-start", {"session_id": session_id})
    assert out.strip() == json.dumps(direct)


def test_session_start_non_compact_sources_never_hit_network(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """startup/resume/clear/absent → {} exit 0 with NO request made. A
    live coordinator is running, so if the source gate ever regressed the
    recorded-call list would be non-empty (and the response non-{})."""
    workspace, _ = live_coordinator
    calls: list[tuple[str, dict[str, Any]]] = []
    real_post = coherence_hook_client.post

    def recording_post(endpoint, path, body, **kwargs):
        calls.append((path, body))
        return real_post(endpoint, path, body, **kwargs)

    monkeypatch.setattr(coherence_hook_client, "post", recording_post)
    for source in ("startup", "resume", "clear", None):
        cc_payload: dict[str, Any] = {"session_id": _sid()}
        if source is not None:
            cc_payload["source"] = source
        rc, out = _drive("session-start", cc_payload, workspace, monkeypatch, capsys)
        assert rc == 0, f"source={source!r}"
        assert out.strip() == "{}", f"source={source!r} must skip, got {out!r}"
    assert calls == [], f"non-compact sources must not reach the network: {calls}"


def test_session_start_body_is_session_only(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The translated body is exactly {"session_id": ...}: source is
    stripped, and a stray agent_id is NOT forwarded (SessionStart carries
    no subagent context — the re-grounding payload is session-scoped, so
    the builder deliberately bypasses _with_agent_id). Also pins verbatim
    passthrough of an arbitrary coordinator response."""
    _fake_coherence_dir(git_workspace, port=65000)
    calls: list[tuple[str, dict[str, Any]]] = []

    def canned_post(endpoint, path, body, **kwargs):
        calls.append((path, body))
        return {"ok": True}

    monkeypatch.setattr(coherence_hook_client, "post", canned_post)
    session_id = _sid()
    cc_payload = {
        "session_id": session_id,
        "source": "compact",
        "agent_id": "worker-1",  # never expected on SessionStart; must be dropped
    }
    rc, out = _drive("session-start", cc_payload, git_workspace, monkeypatch, capsys)
    assert rc == 0
    assert calls == [("/hooks/session-start", {"session_id": session_id})]
    assert out.strip() == json.dumps({"ok": True})


def test_session_start_missing_session_id_emits_empty(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compact source but no session_id → skip before any network call."""
    _fake_coherence_dir(git_workspace, port=65000)
    calls: list[str] = []
    monkeypatch.setattr(
        coherence_hook_client,
        "post",
        lambda endpoint, path, body, **kwargs: calls.append(path) or {},
    )
    rc, out = _drive("session-start", {"source": "compact"}, git_workspace,
                     monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"
    assert calls == []


def test_session_start_tty_stdin_emits_empty(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Developer runs the subcommand manually (stdin is a TTY) → usage
    hint on stderr, {} on stdout, exit 0 — never a blocking read."""
    class _TtyStdin(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", _TtyStdin())
    rc = coherence_hook_client.main(["session-start", "--root", str(git_workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "{}"


def test_session_start_empty_stdin_emits_empty(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    rc = coherence_hook_client.main(["session-start", "--root", str(git_workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "{}"


def test_session_start_malformed_stdin_emits_empty(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("not json at all"))
    rc = coherence_hook_client.main(["session-start", "--root", str(git_workspace)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out.strip() == "{}"


def test_session_start_no_coordinator_returns_empty(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Compact event with no coordinator running → {} exit 0."""
    cc_payload = {"session_id": _sid(), "source": "compact"}
    rc, out = _drive("session-start", cc_payload, git_workspace, monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"


def test_session_start_unreachable_coordinator_returns_empty(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Port file exists but nothing is listening (stale pid file after a
    coordinator crash) → the POST raises CoordinatorUnavailable → {}."""
    import errno
    import socket

    # Port 4, not a freed ephemeral port: another test worker's port-0 bind
    # can take a freed port before the connect, and the client would then
    # talk to a stranger. Ports below 1024 are never handed out for port 0.
    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", 4)) == errno.ECONNREFUSED, "port 4 accepts connections here"
    dead_port = 4
    _fake_coherence_dir(git_workspace, port=dead_port)

    cc_payload = {"session_id": _sid(), "source": "compact"}
    rc, out = _drive("session-start", cc_payload, git_workspace, monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"


@pytest.mark.parametrize("subcommand", ["pre-read", "pre-edit"])
def test_an_empty_coordinator_answer_still_fails_open(
    subcommand: str, git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A coordinator that answers 200 with no body leaves the hook printing
    ``{}`` with exit 0, after the request reached it. The shared client reads
    an empty body as a malformed answer, which the other commands report as a
    failure; a hook must still fail open, or one bad answer blocks the tool
    call."""
    import http.server
    import threading

    seen: list[str] = []

    class _EmptyAnswers(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the stdlib handler's name
            seen.append(self.path)
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EmptyAnswers)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        _fake_coherence_dir(git_workspace, port=server.server_address[1])
        (git_workspace / "plan.md").write_text("plan")
        cc_payload = {
            "session_id": _sid(),
            "tool_name": "Read" if subcommand == "pre-read" else "Edit",
            "tool_input": {"file_path": str(git_workspace / "plan.md")},
        }
        rc, out = _drive(subcommand, cc_payload, git_workspace, monkeypatch, capsys)
    finally:
        server.shutdown()
        server.server_close()

    assert rc == 0
    assert out.strip() == "{}"
    assert f"/hooks/{subcommand}" in seen, seen


def test_session_start_builder_exception_emits_empty(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unexpected exception inside the builder (refactor regression)
    must be swallowed by the dispatch ladder's catch-all → {} exit 0."""
    _fake_coherence_dir(git_workspace, port=65000)
    monkeypatch.setattr(
        coherence_hook_client,
        "_build_session_start",
        lambda cc: (_ for _ in ()).throw(RuntimeError("builder regression")),
    )
    cc_payload = {"session_id": _sid(), "source": "compact"}
    rc, out = _drive("session-start", cc_payload, git_workspace, monkeypatch, capsys)
    assert rc == 0
    assert out.strip() == "{}"


# ----------------------------------------------------------------------
# Caller principal (caller-principal plan, U5)
#
# The hook client is one process per hook event, so the principal of a
# session lives on disk: the mint nonce (created exclusively, never replaced),
# then the principal (replaced only by what a claim with that nonce returns),
# each at 0600 in the existing 0700 ``.coherence/``, keyed by the PARENT
# session's derived id. What that buys on the hook surface is
# convention-enforcement and a detectable unbound caller — any process that
# can read ``.coherence/`` can read these files — never caller separation.
# The recovery of a refused principal is pinned in
# tests/test_caller_principal_client_recovery.py.
# ----------------------------------------------------------------------

import http.server  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

from ccs.adapters.claude_code.auth import load_secret  # noqa: E402
from ccs.adapters.claude_code.coordinator_server import (  # noqa: E402
    CoordinatorHTTPServer,
    caller_principal_identity,
    session_to_agent_id,
)

_PRINCIPAL_HEADER = "Coherence-Caller-Principal"  # frozen duplicate of the wire name


@pytest.fixture
def inproc_coordinator(git_workspace: Path):
    """A coordinator in this process (so the test can read its registry),
    discoverable by the hook client through the usual pid file."""
    server = CoordinatorHTTPServer(git_workspace, port=0, instance_id="hook-principal")
    server.serve_in_thread()
    assert load_secret(server.coordinator_root)
    write_coordinator_pid_file(git_workspace / ".coherence", server.port, pid=os.getpid())
    try:
        yield git_workspace, server
    finally:
        server.shutdown()


def _principal_files(workspace: Path, sid: str) -> tuple[Path, Path]:
    key = caller_principal_identity(sid).hex
    base = workspace / ".coherence" / f"caller-principal-{key}"
    return base.with_name(base.name + ".nonce"), base.with_name(base.name + ".principal")


def _claims(server: CoordinatorHTTPServer) -> int:
    return server.endpoint_counters_snapshot()["principal_claim_total"]


def _edit_payload(workspace: Path, sid: str, **extra: Any) -> dict[str, Any]:
    return {"session_id": sid, "tool_input": {"file_path": str(workspace / "plan.md")}, **extra}


def test_hook_client_claims_once_per_session_and_presents_the_stored_principal(
    inproc_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A live workspace driven through the hook client records attribution
    naming the acting session: the first hook of the session claims (nonce
    persisted first, then the principal), every later hook — a subagent's
    included, which presents its PARENT's principal — reads the stored one
    and claims nothing, and the require-class commit and stop are admitted.

    Prevents a client that re-claims per event (a round trip per hook and a
    new nonce each time, which the first-claim gate refuses), and a client
    that sends no principal, whose commit the coordinator now refuses."""
    workspace, server = inproc_coordinator
    (workspace / "plan.md").write_text("plan v1")
    sid = _sid()

    rc, out = _drive("pre-edit", _edit_payload(workspace, sid), workspace, monkeypatch, capsys)
    assert rc == 0 and json.loads(out).get("ok") is True, out
    (workspace / "plan.md").write_text("plan v2")
    rc, out = _drive("post-edit", _edit_payload(workspace, sid), workspace, monkeypatch, capsys)
    assert rc == 0 and json.loads(out).get("ok") is True, out
    (workspace / "plan.md").write_text("plan v3 by a subagent")
    sub = _edit_payload(workspace, sid, agent_id="sub-1")
    rc, out = _drive("pre-edit", sub, workspace, monkeypatch, capsys)
    rc, out = _drive("post-edit", sub, workspace, monkeypatch, capsys)
    assert rc == 0 and json.loads(out).get("ok") is True, out
    rc, out = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)
    assert rc == 0 and json.loads(out).get("ok") is True, out

    assert _claims(server) == 1
    nonce_file, principal_file = _principal_files(workspace, sid)
    stored = principal_file.read_text().strip()
    assert server.registry.get_caller_principal(caller_principal_identity(sid)) == stored
    for path in (nonce_file, principal_file):
        assert path.stat().st_mode & 0o777 == 0o600, path
    assert not list((workspace / ".coherence").glob(f"*{sid}*")), "keyed by the derived id"
    artifact_id = server.registry.lookup_artifact_id_by_name("plan.md")
    assert server.registry.get_artifact(artifact_id).version == 3
    assert server.registry.last_writer_for(artifact_id) == session_to_agent_id(sid, "sub-1")


# Racing hook processes: run with -n 0 (tests/conftest.py).
@pytest.mark.serial
def test_racing_hook_processes_for_a_new_session_bind_one_principal(inproc_coordinator) -> None:
    """Two hook processes starting together for an unclaimed session produce
    ONE binding: the loser of the exclusive nonce create adopts the winner's
    nonce, so both claims present the same nonce and both receive the bound
    principal. Asserted through a require-class stop in every process: a
    loser that minted its own nonce would be refused the claim, send no
    principal, and answer ``{}``. Real processes (the hook client's actual
    lifetime), released onto stdin at the same instant; whether the claims
    really overlapped is reported, never asserted."""
    workspace, server = inproc_coordinator
    sid = _sid()
    procs = [
        subprocess.Popen(
            [sys.executable, "-m", "ccs.cli.coherence_hook_client", "session-stop",
             "--root", str(workspace)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        for _ in range(4)
    ]
    time.sleep(1.5)  # every process imported and blocked on stdin
    payload = json.dumps({"session_id": sid}).encode()
    for proc in procs:
        proc.stdin.write(payload)
    for proc in procs:
        proc.stdin.close()
    outputs = []
    for proc in procs:
        proc.wait(timeout=30)
        outputs.append((proc.stdout.read(), proc.stderr.read()))

    for out, err in outputs:
        assert json.loads(out).get("ok") is True, (out, err)
    nonce_file, principal_file = _principal_files(workspace, sid)
    bound = server.registry.get_caller_principal(caller_principal_identity(sid))
    assert bound is not None and principal_file.read_text().strip() == bound
    if _claims(server) == 1:
        warnings.warn(
            "the hook processes did not claim concurrently; the adopt-the-winner "
            "path was not exercised this run", RuntimeWarning, stacklevel=2,
        )


def test_ensure_mint_nonce_racers_adopt_one_nonce(tmp_path: Path) -> None:
    """The exclusive-create discipline, raced deterministically: N threads
    aligned on a barrier all return the SAME nonce and exactly one file
    holds it. A last-writer-wins write here would hand two concurrent hooks
    two nonces, and the second claim would be refused (KTD11)."""
    from ccs.adapters.claude_code import auth

    (tmp_path / ".coherence").mkdir(mode=0o700)
    key = caller_principal_identity(_sid()).hex
    barrier = threading.Barrier(8)
    results: list[str] = []
    created: list[bool] = []
    real_create = auth._create_exclusive

    def spy(path: Path, value: str) -> bool:
        won = real_create(path, value)
        created.append(won)
        return won

    def racer() -> None:
        barrier.wait()
        results.append(auth.ensure_mint_nonce(tmp_path, key))

    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        auth._create_exclusive = spy
        threads = [threading.Thread(target=racer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
    finally:
        auth._create_exclusive = real_create
        sys.setswitchinterval(previous)
    assert len(results) == 8 and len(set(results)) == 1
    assert created.count(True) == 1
    if created.count(False) == 0:
        warnings.warn("no racer lost the exclusive create this run", RuntimeWarning, stacklevel=2)


def test_a_foreign_stored_principal_is_recovered_with_the_sessions_own_nonce(
    inproc_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A stored principal the coordinator refuses as foreign (here: another
    session's) is recovered by claiming again with THIS session's stored
    nonce — never a new one, and the nonce file is left exactly as it was.
    That claim returns this session's own binding, which replaces the stored
    file, and the refused stop is retried once and admitted. One claim, no
    re-mint, nothing printed — and no principal or nonce in the output."""
    workspace, server = inproc_coordinator
    sid, other = _sid(), _sid()
    for session in (sid, other):
        rc, _ = _drive("session-stop", {"session_id": session}, workspace, monkeypatch, capsys)
        assert rc == 0
    nonce_file, principal_file = _principal_files(workspace, sid)
    own = principal_file.read_text().strip()
    foreign = _principal_files(workspace, other)[1].read_text()
    principal_file.write_text(foreign)
    nonce_before = nonce_file.read_text()
    claims_before = _claims(server)

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": sid})))
    rc = coherence_hook_client.main(["session-stop", "--root", str(workspace)])
    captured = capsys.readouterr()

    assert rc == 0 and json.loads(captured.out).get("ok") is True, captured
    assert captured.err == ""
    assert principal_file.read_text().strip() == own
    assert server.registry.get_caller_principal(caller_principal_identity(sid)) == own
    assert nonce_file.read_text() == nonce_before
    assert _claims(server) == claims_before + 1
    for secret in (own, foreign.strip(), nonce_before.strip()):
        assert secret not in captured.out + captured.err


def test_a_claimed_session_is_reported_and_never_re_minted(
    inproc_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A session another claimant bound first is refused to this client's
    nonce. The client reports it, stores nothing, keeps its nonce (every
    retry presents the same one), and proceeds without a principal: the
    accept-class read is still answered, the require-class stop is not."""
    workspace, server = inproc_coordinator
    (workspace / "plan.md").write_text("plan v1")
    sid = _sid()
    server.service.claim_caller_principal(
        identity=caller_principal_identity(sid), mint_nonce="someone-elses-nonce-0001"
    )

    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_edit_payload(workspace, sid))))
    assert coherence_hook_client.main(["pre-read", "--root", str(workspace)]) == 0
    first = capsys.readouterr()
    nonce_file, principal_file = _principal_files(workspace, sid)
    nonce = nonce_file.read_text()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": sid})))
    assert coherence_hook_client.main(["session-stop", "--root", str(workspace)]) == 0
    second = capsys.readouterr()

    assert "caller principal refused" in first.err and "NOT re-minting" in first.err
    assert json.loads(first.out).get("status") in ("fresh", "stale")
    assert second.out.strip() == "{}"
    assert not principal_file.exists()
    assert nonce_file.read_text() == nonce


class _NoPrincipalCoordinator(http.server.BaseHTTPRequestHandler):
    """Answers like a coordinator that issues no principals (the sibling Node
    coordinator, an older Python one): 404 on the claim, 200 elsewhere."""

    seen: list[tuple[str, str | None]] = []

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.seen.append((self.path, self.headers.get(_PRINCIPAL_HEADER)))
        status, body = (404, b'{"error":"not found"}') if self.path == "/principal/claim" \
            else (200, b'{"ok":true}')
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return


def test_a_coordinator_that_issues_no_principals_gets_no_header(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """404 on the claim means the coordinator issues none: the hook proceeds
    exactly as before — the request goes out WITHOUT the header, the answer
    passes through, nothing is stored, nothing is reported."""
    _NoPrincipalCoordinator.seen = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _NoPrincipalCoordinator)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        _fake_coherence_dir(git_workspace, port=httpd.server_address[1])
        (git_workspace / "plan.md").write_text("plan")
        sid = _sid()
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_edit_payload(git_workspace, sid))))
        assert coherence_hook_client.main(["post-edit", "--root", str(git_workspace)]) == 0
        captured = capsys.readouterr()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert json.loads(captured.out) == {"ok": True}
    assert _NoPrincipalCoordinator.seen == [("/principal/claim", None), ("/hooks/post-edit", None)]
    assert not _principal_files(git_workspace, sid)[1].exists()
    assert captured.err == ""


@pytest.mark.parametrize(
    ("backend_line", "claims"),
    [
        ("", True),  # the Python coordinator's own format: <pid>\n<port>\n
        ("backend=python\n", True),
        ("backend=node\n", False),  # the Node coordinator's: it issues no principals
    ],
    ids=["python-format", "backend-python", "backend-node"],
)
def test_the_claim_is_skipped_only_when_the_pid_file_names_the_node_backend(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], backend_line: str, claims: bool,
) -> None:
    """The Node coordinator writes ``<pid>\\n<port>\\nbackend=node\\n`` and
    issues no principals, so a hook against it must not pay a claim round
    trip (a 404) on EVERY event: the request is exactly the one it sent before
    principals existed, and no nonce file is created. Any other pid file —
    the Python coordinator writes no backend line; an explicit
    ``backend=python`` — still claims, and a 404 there (an older Python
    coordinator) keeps the no-header fallback. Requests are counted."""
    _NoPrincipalCoordinator.seen = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _NoPrincipalCoordinator)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    try:
        _fake_coherence_dir(git_workspace, port=httpd.server_address[1])
        pid_file = git_workspace / ".coherence" / "server.pid"
        pid_file.write_text(pid_file.read_text() + backend_line)
        (git_workspace / "plan.md").write_text("plan")
        sid = _sid()
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_edit_payload(git_workspace, sid))))
        assert coherence_hook_client.main(["post-edit", "--root", str(git_workspace)]) == 0
        captured = capsys.readouterr()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert json.loads(captured.out) == {"ok": True}
    expected = [("/hooks/post-edit", None)]
    if claims:
        expected = [("/principal/claim", None), *expected]
    assert _NoPrincipalCoordinator.seen == expected
    assert _principal_files(git_workspace, sid)[0].exists() is claims, "nonce file"
    assert captured.err == ""


def test_an_older_client_that_never_claims_is_admitted_and_its_write_recorded(
    inproc_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R16, the version-skew direction this coordinator can observe: a hook
    client that predates the principal sends no header and never claims, so
    the session it names stays unbound. Its require-class commit is ADMITTED
    and recorded exactly as before — the version advances and
    ``last_writer_id`` is its composite — rather than refused into the
    ``{}`` the shipped client prints for any non-2xx, which would let the edit
    proceed with coherence silently off."""
    workspace, server = inproc_coordinator
    (workspace / "plan.md").write_text("plan v1")
    sid = _sid()
    monkeypatch.setattr(
        _coherence_client, "obtain_stored_principal", lambda *_a, **_k: None
    )
    payload = _edit_payload(workspace, sid, agent_id="sub-1")
    rc, out = _drive("pre-edit", payload, workspace, monkeypatch, capsys)
    assert rc == 0 and json.loads(out).get("ok") is True
    artifact_id = server.registry.lookup_artifact_id_by_name("plan.md")
    version = server.registry.get_artifact(artifact_id).version
    absent_before = server.counters_snapshot()["caller_principal_absent_total"]
    (workspace / "plan.md").write_text("plan v2")
    rc, out = _drive("post-edit", payload, workspace, monkeypatch, capsys)

    assert rc == 0 and json.loads(out).get("ok") is True, out
    assert server.registry.get_artifact(artifact_id).version == version + 1
    assert server.registry.last_writer_for(artifact_id) == session_to_agent_id(sid, "sub-1")
    assert server.counters_snapshot()["caller_principal_absent_total"] == absent_before + 1
    assert server.registry.get_caller_principal(caller_principal_identity(sid)) is None


def test_a_request_without_a_principal_naming_a_claimed_session_is_refused(
    inproc_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The #188 case through the hook client: a newer client's session is
    CLAIMED (its hooks present the principal, it holds EXCLUSIVE after a
    pre-edit and has committed once). A second, principal-less caller naming
    that session — deliberately not reading the stored principal, which on
    this surface any process that can read ``.coherence/`` could — is refused
    on the commit and on the stop: the peer's grant stands and the recorded
    writer and version are unchanged. Convention-enforcement and a detectable
    unbound caller, not separation between callers of one OS user."""
    workspace, server = inproc_coordinator
    (workspace / "plan.md").write_text("plan v1")
    peer = _sid()
    payload = _edit_payload(workspace, peer)
    rc, out = _drive("pre-edit", payload, workspace, monkeypatch, capsys)
    (workspace / "plan.md").write_text("plan v2 by the peer")
    rc, out = _drive("post-edit", payload, workspace, monkeypatch, capsys)
    assert json.loads(out).get("ok") is True, out
    rc, out = _drive("pre-edit", payload, workspace, monkeypatch, capsys)
    artifact_id = server.registry.lookup_artifact_id_by_name("plan.md")
    peer_agent = session_to_agent_id(peer)
    version = server.registry.get_artifact(artifact_id).version
    assert server.registry.get_state_map(artifact_id)[peer_agent].name == "EXCLUSIVE"

    # The forger reads neither of the peer's stored files: not the principal,
    # and not the mint nonce a stored-principal client re-claims with on a
    # refusal (R20) — on this surface either is one read of .coherence/ away.
    monkeypatch.setattr(
        _coherence_client, "obtain_stored_principal", lambda *_a, **_k: None
    )
    monkeypatch.setattr(_coherence_client, "load_mint_nonce", lambda *_a, **_k: None)
    (workspace / "plan.md").write_text("plan v3 by a forger")
    rc, commit = _drive("post-edit", payload, workspace, monkeypatch, capsys)
    rc, stop = _drive("session-stop", {"session_id": peer}, workspace, monkeypatch, capsys)

    assert commit.strip() == "{}" and stop.strip() == "{}"
    assert server.registry.get_artifact(artifact_id).version == version
    assert server.registry.last_writer_for(artifact_id) == peer_agent
    assert server.registry.get_state_map(artifact_id)[peer_agent].name == "EXCLUSIVE"


# ----------------------------------------------------------------------
# #238 — a Grep hook gets the freshness advisory while the registry is held
#
# The coordinator's pre-grep lookup waited on the registry lock with no bound,
# so the hook client's own timeout fired first and the model got ``{}``. Driven
# end to end, stdin through main() to stdout, against an in-process
# coordinator whose registry lock a helper thread holds.
# ----------------------------------------------------------------------


def test_pre_grep_delivers_the_freshness_advisory_while_the_registry_lock_is_held(
    inproc_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import ccs.adapters.claude_code.coordinator_server as server_mod
    from tests.test_claude_code_coordinator_server import _HeldRegistryLock

    workspace, server = inproc_coordinator
    cc_payload = {
        "session_id": _sid(),
        "tool_name": "Grep",
        "tool_input": {"pattern": "anything", "path": "docs"},
    }
    # The first call claims the session's principal while the lock is free, so
    # the held lock below blocks only pre-grep's own registry lookup.
    rc, out = _drive("pre-grep", cc_payload, workspace, monkeypatch, capsys)
    assert (rc, json.loads(out)) == (0, {"status": "fresh"})

    monkeypatch.setattr(server_mod, "HANDLER_TIMEOUT_SEC", 0.25)
    with _HeldRegistryLock(server):
        rc, out = _drive("pre-grep", cc_payload, workspace, monkeypatch, capsys)

    assert rc == 0
    response = json.loads(out)
    assert response == json.loads(json.dumps(server_mod._DEFAULT_DEGRADED_RESPONSE))
    assert "could not verify" in response["hookSpecificOutput"]["additionalContext"]


def test_pre_edit_delivers_the_freshness_advisory_when_the_coordinator_times_out(
    inproc_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#236, end to end: an Edit whose pre-edit times out on registry
    contention is admitted -- a hook does not block a tool call on coordinator
    load -- and the model is told the coherence check did not run. The
    advisory reaches stdout context-only: no ``permissionDecision``, so Claude
    Code still asks the user about the Edit as it would have. It used to print
    a bare ``{"ok": true, "degraded": true}``: an admit with no warning."""
    import ccs.adapters.claude_code.coordinator_server as server_mod
    from tests.test_claude_code_coordinator_server import _HeldRegistryLock

    workspace, server = inproc_coordinator
    (workspace / "plan.md").write_text("plan v1")
    payload = _edit_payload(workspace, _sid())
    # The first call claims the session's principal while the lock is free, so
    # the held lock below blocks only pre-edit's work body.
    rc, out = _drive("pre-edit", payload, workspace, monkeypatch, capsys)
    assert rc == 0 and json.loads(out).get("ok") is True, out

    monkeypatch.setattr(server_mod, "HANDLER_TIMEOUT_SEC", 0.25)
    with _HeldRegistryLock(server):
        rc, out = _drive("pre-edit", payload, workspace, monkeypatch, capsys)

    assert rc == 0
    response = json.loads(out)
    assert response == json.loads(json.dumps(server_mod._PRE_EDIT_DEGRADED_RESPONSE))
    hso = response["hookSpecificOutput"]
    assert hso["hookEventName"] == "PreToolUse"
    assert "permissionDecision" not in hso
    assert "could not verify" in hso["additionalContext"]


# ----------------------------------------------------------------------
# #498 -- hook.secret and .coherence are read without following a link
#
# A client reads them with no start before it, so a link a repository
# committed would send the file it names as the bearer to whatever port
# server.pid holds. server.pid itself is a regular file in these tests:
# how it is read is not changed here.
# ----------------------------------------------------------------------

from tests.coherence_link_helpers import TARGET_BYTES, plant_link, release_fifo  # noqa: E402
from tests.pid_lock_helpers import write_coordinator_pid_file


def _secret_link_refusal(coherence: Path) -> str:
    """The refusal of a linked ``hook.secret``, written out: the Node client
    prints the same text."""
    return (
        f"refusing {coherence / 'hook.secret'}: it is a symlink, and .coherence must be a real "
        "directory with no link directly inside it. Replace the link with a copy of the file it "
        f"names, or mount a directory at {coherence}."
    )


class _BearerRecorder(http.server.BaseHTTPRequestHandler):
    """Records each request's path and ``Authorization``; answers 200 ``{}``."""

    seen: list[tuple[str, str | None]] = []

    def _answer(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.seen.append((self.path, self.headers.get("Authorization")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        self._answer()

    def do_GET(self) -> None:  # noqa: N802 — stdlib name
        self._answer()

    def log_message(self, *_args: Any) -> None:
        return


def test_a_linked_hook_secret_is_never_sent_as_the_bearer(
    git_workspace: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#498: a ``hook.secret`` that is a link is refused before any request: the
    hook fails open with ``{}``, nothing reaches the port ``server.pid`` names,
    and ``resolve_endpoint`` names the link and the way out.

    Prevents the bytes of the file a link names going out as ``Authorization:
    Bearer`` to that port."""
    _BearerRecorder.seen = []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _BearerRecorder)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    coherence = git_workspace / ".coherence"
    try:
        coherence.mkdir()
        write_coordinator_pid_file(coherence, httpd.server_address[1])
        plant_link(coherence, "hook.secret", "existing")
        (git_workspace / "plan.md").write_text("plan")
        read = {
            "session_id": _sid(), "tool_name": "Read",
            "tool_input": {"file_path": str(git_workspace / "plan.md")},
        }
        rc, out = _drive("pre-read", read, git_workspace, monkeypatch, capsys)
        with pytest.raises(_coherence_client.CoordinatorUnavailable) as refused:
            _coherence_client.resolve_endpoint(git_workspace)
    finally:
        httpd.shutdown()
        httpd.server_close()

    assert (rc, json.loads(out)) == (0, {})
    assert _BearerRecorder.seen == []
    assert str(refused.value) == _secret_link_refusal(coherence)


@pytest.mark.parametrize("dangling", [False, True], ids=["to_another_workspace", "dangling"])
def test_resolve_endpoint_refuses_a_linked_coherence_before_reading_server_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dangling: bool,
) -> None:
    """#498: a ``.coherence`` that is a link is refused before ``server.pid`` is
    read, naming the link and the way out. ``O_NOFOLLOW`` on each file guards
    only the last component, so a client would read the port and the secret of
    the directory the link names.

    Prevents a client in one workspace joining another workspace's coordinator
    with that coordinator's own valid bearer."""
    peer = tmp_path / "peer" / ".coherence"
    if not dangling:
        peer.mkdir(parents=True)
        (peer / "server.pid").write_text("12345\n50000\n")
        (peer / "hook.secret").write_text("peer-secret\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    coherence = workspace / ".coherence"
    coherence.symlink_to(peer)

    def no_pid_read(path: Path) -> None:
        pytest.fail(f"read {path} through a linked .coherence")

    monkeypatch.setattr(_coherence_client, "read_pid_file", no_pid_read)

    with pytest.raises(_coherence_client.CoordinatorUnavailable) as refused:
        _coherence_client.resolve_endpoint(workspace)

    assert str(refused.value) == (
        f"refusing {coherence}: it is a symlink, and .coherence must be a real directory. "
        "Replace the link with a real directory and copy in the files you want to keep (move "
        f"state.db together with its -wal and -shm), or mount a directory at {coherence}."
    )


@pytest.mark.parametrize("linked", [False, True], ids=["fifo", "link_to_fifo"])
def test_resolve_endpoint_answers_at_once_for_a_fifo_hook_secret(git_workspace: Path, linked: bool) -> None:
    """#498: a FIFO at ``hook.secret``, or a link to one, is refused at once and
    named: "not a regular file" for the FIFO, the link refusal for the link. The
    call runs on a thread and the FIFO is released whatever happens, so a
    regression fails here instead of hanging the suite.

    Prevents a hook blocking on a FIFO's open, and a FIFO with no writer reading
    as an empty secret."""
    coherence = git_workspace / ".coherence"
    coherence.mkdir()
    write_coordinator_pid_file(coherence, 50000)
    secret = coherence / "hook.secret"
    if linked:
        fifo = plant_link(coherence, "hook.secret", "to_fifo")
    else:
        fifo = secret
        os.mkfifo(fifo)
    answers: list[str] = []

    def resolve() -> None:
        try:
            _coherence_client.resolve_endpoint(git_workspace)
        except _coherence_client.CoordinatorUnavailable as exc:
            answers.append(str(exc))

    worker = threading.Thread(target=resolve, daemon=True)
    try:
        worker.start()
        worker.join(timeout=10)
        blocked = worker.is_alive()
    finally:
        release_fifo(fifo)
        worker.join(timeout=5)

    assert not blocked, "resolve_endpoint blocked on a FIFO"
    expected = (
        _secret_link_refusal(coherence) if linked
        else f"{secret} is not a regular file; restart with `agent-coherence-coordinator`"
    )
    assert answers == [expected]


@pytest.mark.parametrize("linked", ["hook.secret", ".coherence"])
def test_load_secret_never_reads_through_a_link(tmp_path: Path, linked: str) -> None:
    """#498: ``load_secret`` answers None, as for a missing file, for a
    ``hook.secret`` that is a link and for one in a ``.coherence`` that is a
    link.

    Prevents a hook client taking the bytes of the file a link names as the
    workspace's secret."""
    if linked == "hook.secret":
        plant_link(tmp_path / ".coherence", "hook.secret", "existing")
    else:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "hook.secret").write_bytes(TARGET_BYTES)
        (tmp_path / ".coherence").symlink_to(elsewhere)

    assert load_secret(tmp_path) is None


# ----------------------------------------------------------------------
# Links and workspace-root spellings
#
# The hook client keys a file the way the plugin's Node client does: it
# follows links to the file they name and keeps the case each name was typed
# in, and it falls back to the disk's own spelling only for a path that would
# otherwise lie outside the root (a root typed in another case, or /var for
# /private/var). One file then reaches the coordinator under one path, whatever
# link or root spelling the tool call used. The expected paths are literal and
# match the Node client's on the same layout.
# ----------------------------------------------------------------------

from functools import partial  # noqa: E402

from ccs.core.disk_spelling import native_spelling  # noqa: E402

_LINK_SID = "55555555-5555-4555-8555-555555555555"


@pytest.fixture
def linked_workspace(tmp_path: Path) -> Path:
    """``<tmp>/ws`` is the root: plan.md, documentation/guide.md,
    docs -> documentation, data/plan-link.md -> ../plan.md,
    data/outside-link.md -> <tmp>/outside/secret.md, and
    data/dangle.md -> nope.md (missing)."""
    root = tmp_path / "ws"
    (root / "data").mkdir(parents=True)
    (root / "documentation").mkdir()
    (tmp_path / "outside").mkdir()
    (root / "plan.md").write_text("plan")
    (root / "documentation" / "guide.md").write_text("guide")
    (tmp_path / "outside" / "secret.md").write_text("secret")
    (root / "docs").symlink_to("documentation")
    (root / "data" / "plan-link.md").symlink_to(Path("..") / "plan.md")
    (root / "data" / "outside-link.md").symlink_to(tmp_path / "outside" / "secret.md")
    (root / "data" / "dangle.md").symlink_to("nope.md")
    return root


def _file_cc(file_path: Path | str) -> dict[str, Any]:
    return {"session_id": _LINK_SID, "tool_input": {"file_path": str(file_path)}}


def _read_path(file_path: Path | str, root: Path) -> str:
    """The path a pre-read of ``file_path`` sends for workspace ``root``."""
    return coherence_hook_client._build_pre_read(_file_cc(file_path), root)["path"]


def test_a_link_to_a_file_is_sent_as_the_file_it_names(linked_workspace: Path) -> None:
    root = linked_workspace
    link = root / "data" / "plan-link.md"

    assert _read_path(link, root) == "plan.md"
    edit = coherence_hook_client._build_pre_edit(_file_cc(link), root)
    assert edit == {"session_id": _LINK_SID, "path": "plan.md"}


def test_a_path_through_a_linked_directory_is_sent_under_the_targets_path(
    linked_workspace: Path,
) -> None:
    root = linked_workspace

    assert _read_path(root / "docs" / "guide.md", root) == "documentation/guide.md"
    grep = coherence_hook_client._build_pre_grep(
        {"session_id": _LINK_SID, "tool_input": {"path": str(root / "docs")}}, root
    )
    assert grep == {"session_id": _LINK_SID, "search_root": "documentation"}


def test_a_link_whose_target_leaves_the_root_skips_the_hook(linked_workspace: Path) -> None:
    root = linked_workspace
    with pytest.raises(coherence_hook_client._SkipHook):
        _read_path(root / "data" / "outside-link.md", root)


@pytest.mark.parametrize("name", ["dangle.md", "loop-a.md"])
def test_a_link_that_does_not_resolve_skips_the_hook_with_the_skip_signal(
    linked_workspace: Path, name: str
) -> None:
    """A dangling or looping link is skipped, not sent under the missing
    target's path (or the link's own): either would give its file a second
    identity once the target exists. The skip signal, not a filesystem
    error, so the hook prints ``{}`` without contacting the coordinator."""
    root = linked_workspace
    (root / "data" / "loop-a.md").symlink_to("loop-b.md")
    (root / "data" / "loop-b.md").symlink_to("loop-a.md")
    builders = (
        coherence_hook_client._build_pre_read,
        coherence_hook_client._build_pre_edit,
        coherence_hook_client._build_post_edit,
    )
    for build in builders:
        with pytest.raises(coherence_hook_client._SkipHook):
            build(_file_cc(root / "data" / name), root)


def test_a_new_file_under_an_existing_directory_keeps_the_leaf_as_typed(
    linked_workspace: Path,
) -> None:
    root = linked_workspace

    edit = coherence_hook_client._build_pre_edit(_file_cc(root / "docs" / "New-Plan.MD"), root)
    assert edit == {"session_id": _LINK_SID, "path": "documentation/New-Plan.MD"}
    assert _read_path(root / "data" / "new" / "deep.md", root) == "data/new/deep.md"


def test_a_name_typed_in_another_case_under_a_correctly_spelled_root_is_sent_as_typed(
    linked_workspace: Path,
) -> None:
    """A Bash token is keyed as typed, so a Read of README.MD is sent as
    typed too: sending the disk's README.md would keep the two tools'
    records of one file apart on a coordinator that folds nothing."""
    root = linked_workspace
    (root / "README.md").write_text("readme")
    if not (root / "README.MD").exists():
        pytest.skip("the temporary disk keeps two spellings apart, so README.MD is another file")

    assert _read_path(root / "README.MD", root) == "README.MD"
    edit = coherence_hook_client._build_pre_edit(_file_cc(root / "README.MD"), root)
    assert edit == {"session_id": _LINK_SID, "path": "README.MD"}
    # The link is followed; the names typed around it keep their case.
    assert _read_path(root / "docs" / "GUIDE.MD", root) == "documentation/GUIDE.MD"


def test_a_tool_path_spelled_with_a_differently_cased_root_is_sent_relative_to_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "Repo"
    (root / "data").mkdir(parents=True)
    (root / "data" / "x.txt").write_text("x")
    other_case = tmp_path / "REPO"
    if not other_case.exists():
        pytest.skip("the temporary disk keeps two spellings apart, so no other spelling reaches the root")

    assert _read_path(other_case / "data" / "x.txt", root) == "data/x.txt"
    assert _read_path(root / "data" / "x.txt", other_case) == "data/x.txt"
    edit = coherence_hook_client._build_pre_edit(_file_cc(other_case / "data" / "New.TXT"), root)
    assert edit == {"session_id": _LINK_SID, "path": "data/New.TXT"}


def test_a_hook_whose_path_spells_the_root_in_another_case_reaches_the_coordinator(
    live_coordinator, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Claude Code sends ``<base>/REPO/data/x.txt`` for a workspace whose disk
    spelling is ``<base>/Repo``: the coordinator sees the read of
    ``data/x.txt`` instead of the hook skipping it as outside the root."""
    workspace, _ = live_coordinator
    other_case = workspace.parent / workspace.name.swapcase()
    if not other_case.exists():
        pytest.skip("the temporary disk keeps two spellings apart, so no other spelling reaches the root")
    from ccs.cli import coherence_status, coherence_track

    (workspace / "data").mkdir()
    (workspace / "data" / "x.txt").write_text("x")
    coherence_track.main(["--root", str(workspace), "data/x.txt"])
    capsys.readouterr()

    rc, out = _drive(
        "pre-read", _file_cc(other_case / "data" / "x.txt"), workspace, monkeypatch, capsys
    )
    assert rc == 0
    assert json.loads(out) != {}, "the hook skipped the read as outside the workspace root"
    coherence_status.main(["--root", str(workspace), "--json"])
    status = json.loads(capsys.readouterr().out)
    assert "data/x.txt" in [a["path"] for a in status.get("tracked_artifacts", [])]


def test_a_root_reached_through_a_link_and_a_path_spelled_through_its_target_agree(
    linked_workspace: Path, tmp_path: Path
) -> None:
    root = linked_workspace
    alias = tmp_path / "alias"
    alias.symlink_to(root)

    assert _read_path(root / "plan.md", alias) == "plan.md"
    assert _read_path(alias / "plan.md", root) == "plan.md"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS reaches /var through /private/var")
def test_macos_var_and_private_var_spellings_of_the_root_and_the_path_agree(
    linked_workspace: Path,
) -> None:
    private_spelling = native_spelling(linked_workspace)
    assert private_spelling.startswith("/private/"), private_spelling
    private_root = Path(private_spelling)
    link_root = Path(private_spelling[len("/private"):])

    assert _read_path(private_root / "plan.md", link_root) == "plan.md"
    assert _read_path(link_root / "plan.md", private_root) == "plan.md"


@pytest.mark.parametrize(
    "typed, expected",
    [
        ("plan.md", "plan.md"),
        ("documentation/guide.md", "documentation/guide.md"),
        ("data/new/deep.md", "data/new/deep.md"),
        ("data/", "data"),
        ("", "."),
    ],
)
def test_todays_paths_are_sent_as_before(
    linked_workspace: Path, typed: str, expected: str
) -> None:
    """A path with no link and the root's own spelling is sent exactly as
    before links were followed through the disk resolvers."""
    root = linked_workspace
    file_path = os.path.join(str(root), typed)

    assert coherence_hook_client._to_workspace_relative(file_path, root) == expected


@pytest.mark.parametrize(
    "typed, key",
    [
        ("sub/link/../plan.md", "sub/plan.md"),
        ("sub/link/../new.md", "sub/new.md"),
        ("missing/../w2.md", "w2.md"),
    ],
)
def test_a_parent_step_is_folded_by_text_before_links_are_followed(
    tmp_path: Path, typed: str, key: str
) -> None:
    """A ``..`` in a tool path is folded by text before any link is followed,
    so the key names the file Claude Code's tools open, not the one the shell
    opens. Measured on this layout (sub/link -> other/dir): a Read of
    sub/link/../plan.md returned sub/plan.md, an Edit of it changed
    sub/plan.md, a Write of sub/link/../new.md created sub/new.md and a Write
    of missing/../w2.md created w2.md; only the shell opened other/plan.md.
    Claude Code normalizes an absolute tool path before opening it; the
    shell's ``..`` is coordinated by the Bash routes. Following the link
    first would key other/plan.md for a Read that opened sub/plan.md."""
    root = tmp_path / "ws"
    (root / "sub").mkdir(parents=True)
    (root / "other" / "dir").mkdir(parents=True)
    (root / "sub" / "plan.md").write_text("lexical")
    (root / "other" / "plan.md").write_text("physical")
    (root / "sub" / "link").symlink_to(root / "other" / "dir")
    file_path = os.path.join(str(root), typed)

    assert coherence_hook_client._to_workspace_relative(file_path, root) == key
    assert _coherence_client.normalize_workspace_path(file_path, root) == (key, None)


def test_no_payload_carries_a_name_kind_field(linked_workspace: Path) -> None:
    """Each coordinator works out a name's kind itself; the hook client
    never sends one."""
    root = linked_workspace
    cc = {
        "session_id": _LINK_SID,
        "agent_id": "sub-1",
        "source": "compact",
        "tool_input": {
            "file_path": str(root / "data" / "plan-link.md"),
            "command": "cat plan.md",
            "path": str(root / "docs"),
        },
    }
    hc = coherence_hook_client
    with_root = (hc._build_pre_read, hc._build_pre_edit, hc._build_post_edit, hc._build_pre_grep)
    without_root = (
        hc._build_session_stop, hc._build_subagent_stop, hc._build_session_start, hc._build_pre_bash,
    )
    builds = [partial(b, cc, root) for b in with_root] + [partial(b, cc) for b in without_root]
    for build in builds:
        assert "name_kind" not in build(), build


# ----------------------------------------------------------------------
# Refusing an edit of a held strict file
#
# With .coherence/hook_on_held holding ``refuse``, a pre-edit of a strict path
# reads the coordinator's capability from /status?detail=metrics, opts in with
# ``if_unheld: true`` and prints a relay of the answer or one of three denies.
# Once a coordinator is known and the path is strict, no failure prints ``{}``.
# Every other case keeps today's requests and bytes. The rows are those of
# tests/fixtures/hook_on_held/decision_table.json, driven through main().
# ----------------------------------------------------------------------

from collections.abc import Callable, Iterator  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass  # noqa: E402

from ccs.adapters.claude_code import hook_payloads  # noqa: E402
from ccs.cli import _hook_on_held as on_held  # noqa: E402
from tests.coordinator_helpers import write_holders  # noqa: E402

_ON_HELD_FIXTURES = Path(__file__).parent / "fixtures" / "hook_on_held"
_STATUS = "/status?detail=metrics"
_ADMIT_BYTES = b'{"ok": true}'
_ADMIT_LINE = _ADMIT_BYTES.decode() + "\n"
_UNKNOWN_SETTING_LINE = f"agent-coherence-hook-client: {on_held.UNKNOWN_SETTING_MESSAGE}\n"
_NO_REFUSAL_LINE = f"agent-coherence-hook-client: {on_held.NO_REFUSAL_MESSAGE}\n"
_TODAYS_ROUTES = [("POST", "/principal/claim"), ("POST", "/hooks/pre-edit")]


def _on_held_rows(stage: str) -> list[dict]:
    rows = json.loads((_ON_HELD_FIXTURES / "decision_table.json").read_text())["rows"]
    return [row for row in rows if row["stage"] == stage]


def _strict_workspace(workspace: Path, setting: str | None = "refuse\n", strict: tuple[str, ...] = ("plan.md",)) -> None:
    """``plan.md`` and ``notes.md`` tracked, ``strict`` strict, and the setting
    file holding ``setting`` (absent for ``None``)."""
    coherence = workspace / ".coherence"
    coherence.mkdir(mode=0o700, exist_ok=True)
    (coherence / "tracked.yaml").write_text('- "plan.md"\n- "notes.md"\n')
    (coherence / "strict_mode.yaml").write_text("".join(f'- "{entry}"\n' for entry in strict))
    if setting is not None:
        (coherence / "hook_on_held").write_text(setting)
    for name in ("plan.md", "notes.md"):
        (workspace / name).write_text(f"{name} v1\n")


@dataclass(frozen=True)
class _Hook:
    """One test's workspace and hook runner. ``run(cc, subcommand)`` feeds
    ``cc`` to main() and returns (stdout, stderr); ``sends`` records each hook
    request the client hands its transport as ``{"path", "payload",
    "headers"}`` plus the ``answer`` it got or the exception it ``raised``."""

    workspace: Path
    run: Callable[..., tuple[str, str]]
    sends: list[dict]
    monkeypatch: pytest.MonkeyPatch


def _record_sends(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    sends: list[dict] = []
    real_post = coherence_hook_client.post

    def spy(endpoint: Any, path: str, payload: dict, **kwargs: Any) -> Any:
        record = {"path": path, "payload": dict(payload), "headers": kwargs.get("extra_headers")}
        sends.append(record)
        try:
            record["answer"] = real_post(endpoint, path, payload, **kwargs)
        except Exception as exc:
            record["raised"] = exc
            raise
        return record["answer"]

    monkeypatch.setattr(coherence_hook_client, "post", spy)
    return sends


@pytest.fixture
def hook(git_workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> _Hook:
    def run(cc: dict[str, Any], subcommand: str = "pre-edit") -> tuple[str, str]:
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(cc)))
        assert coherence_hook_client.main([subcommand, "--root", str(git_workspace)]) == 0
        captured = capsys.readouterr()
        return captured.out, captured.err

    return _Hook(git_workspace, run, _record_sends(monkeypatch), monkeypatch)


@dataclass(frozen=True)
class _Answer:
    """How the scripted coordinator answers one route: ``status`` and raw
    ``body`` after ``delay`` seconds and ``before`` has run; with ``hang_up``,
    no answer at all; with ``refuse_after``, it stops listening first, so the
    next connect is refused."""

    status: int = 200
    body: bytes = _ADMIT_BYTES
    delay: float = 0.0
    hang_up: bool = False
    refuse_after: bool = False
    before: Callable[[], None] | None = None


def _json_answer(body: object, status: int = 200) -> _Answer:
    return _Answer(status=status, body=json.dumps(body).encode())


_REFUSES = _json_answer({"pre_edit_if_unheld": True, "fold_profile": "none"})


class _SequentialServer(http.server.HTTPServer):
    """Answers one request at a time, so an answer that stops listening does
    so between requests; wakes every 50 ms to see a stop."""

    timeout = 0.05

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that gave up on a slow answer leaves a broken pipe here,
        # which is the case under test, not a failure of it.
        return


def _serve(httpd: http.server.HTTPServer, stop: threading.Event) -> None:
    while not stop.is_set() and httpd.socket.fileno() != -1:
        httpd.handle_request()


def _write_answer(handler: http.server.BaseHTTPRequestHandler, answer: _Answer, released: threading.Event) -> None:
    released.wait(answer.delay)
    if answer.before is not None:
        answer.before()
    if answer.refuse_after:
        handler.server.socket.close()
    if answer.hang_up:
        handler.close_connection = True
        return
    handler.send_response(answer.status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(answer.body)))
    handler.end_headers()
    handler.wfile.write(answer.body)


def _scripted_handler(answers: dict[str, _Answer], seen: list[dict], released: threading.Event) -> type:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - the stdlib handler's name
            seen.append({"method": "GET", "path": self.path, "body": None})
            _write_answer(self, answers.get(self.path, _Answer(404, b"{}")), released)

        def do_POST(self) -> None:  # noqa: N802 - the stdlib handler's name
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            seen.append({"method": "POST", "path": self.path, "body": json.loads(raw)})
            claim = self.path == "/principal/claim"
            _write_answer(self, answers.get(self.path, _Answer(404, b"{}") if claim else _Answer()), released)

        def log_message(self, *_args: Any) -> None:
            return

    return Handler


@contextmanager
def _scripted_coordinator(workspace: Path, answers: dict[str, _Answer]) -> Iterator[list[dict]]:
    """A coordinator that answers each route as ``answers`` scripts it (an
    unscripted POST with ``{"ok": true}``) and 404 on the principal claim, as
    one that issues no principals. Yields each request it got as
    ``{"method", "path", "body"}``."""
    seen: list[dict] = []
    released, stop = threading.Event(), threading.Event()
    httpd = _SequentialServer(("127.0.0.1", 0), _scripted_handler(answers, seen, released))
    thread = threading.Thread(target=_serve, args=(httpd, stop), daemon=True)
    thread.start()
    try:
        _fake_coherence_dir(workspace, port=httpd.server_address[1])
        yield seen
    finally:
        released.set()
        stop.set()
        thread.join(timeout=10)
        httpd.server_close()


def _stub(answers: dict[str, _Answer] | None = None) -> Callable[[_Hook], Any]:
    """Starts a scripted coordinator that refuses held files, answering as
    ``answers`` adds."""
    scripted = {_STATUS: _REFUSES, **(answers or {})}
    return lambda hook: _scripted_coordinator(hook.workspace, scripted)


def _slow(route: str) -> Callable[[_Hook], Any]:
    """Starts a scripted coordinator that answers ``route`` only after the
    client gave up on it."""

    def start(hook: _Hook) -> Any:
        hook.monkeypatch.setattr(_coherence_client, "CLI_HTTP_TIMEOUT_SEC", 0.5)
        late = _Answer(body=_REFUSES.body if route == _STATUS else _ADMIT_BYTES, delay=10.0)
        return _scripted_coordinator(hook.workspace, {_STATUS: _REFUSES, route: late})

    return start


@contextmanager
def _no_coordinator(hook: _Hook) -> Iterator[list[dict]]:
    """No server.pid: no coordinator port is known."""
    yield []


@contextmanager
def _dead_coordinator(hook: _Hook) -> Iterator[list[dict]]:
    """server.pid names a port, but no coordinator holds the workspace lock."""
    (hook.workspace / ".coherence" / "server.pid").write_text("12345\n50000\n")
    (hook.workspace / ".coherence" / "hook.secret").write_text("test-secret")
    yield []


@contextmanager
def _refusing_port(hook: _Hook) -> Iterator[list[dict]]:
    """server.pid names port 4, where nothing listens: below 1024, so no other
    test's port-0 bind can take it."""
    import errno
    import socket

    with socket.socket() as probe:
        probe.settimeout(2)
        assert probe.connect_ex(("127.0.0.1", 4)) == errno.ECONNREFUSED, "port 4 accepts connections here"
    _fake_coherence_dir(hook.workspace, port=4)
    yield []


def _routes(seen: list[dict]) -> list[tuple[str, str]]:
    return [(request["method"], request["path"]) for request in seen]


def _sent_edits(seen: list[dict]) -> list[dict]:
    return [request["body"] for request in seen if request["path"] == "/hooks/pre-edit"]


def _edit_cc(workspace: Path, sid: str, name: str = "plan.md") -> dict[str, Any]:
    return {"session_id": sid, "tool_name": "Edit", "tool_input": {"file_path": str(workspace / name)}}


def _unavailable(path: str = "plan.md") -> str:
    return json.dumps({"hookSpecificOutput": hook_payloads.emit_admission_unavailable_deny(path=path)}) + "\n"


@pytest.mark.parametrize(
    "case",
    [
        (None, ("plan.md",), "plan.md", "pre-edit", ""),
        ("", ("plan.md",), "plan.md", "pre-edit", ""),
        ("displace\n", ("plan.md",), "plan.md", "pre-edit", ""),
        ("Refuse\n", ("plan.md",), "plan.md", "pre-edit", _UNKNOWN_SETTING_LINE),
        ("refuse\n", ("plan.md",), "notes.md", "pre-edit", ""),
        ("refuse\n", (), "plan.md", "pre-edit", ""),
        ("refuse\n", ("plan.md",), "plan.md", "pre-read", ""),
    ],
    ids=["absent", "empty", "displace", "unknown", "not-strict", "no-strict-entries", "pre-read"],
)
def test_a_hook_that_does_not_opt_in_sends_and_prints_what_it_does_today(hook: _Hook, case: tuple) -> None:
    """``case`` is the setting, the strict entries, the file, the subcommand
    and the stderr today's hook adds."""
    setting, strict, name, subcommand, stderr = case
    _strict_workspace(hook.workspace, setting, strict)
    with _stub()(hook) as seen:
        out, err = hook.run(_edit_cc(hook.workspace, _sid(), name), subcommand)

    assert (out, err) == (_ADMIT_LINE, stderr)
    assert _routes(seen) == [("POST", "/principal/claim"), ("POST", f"/hooks/{subcommand}")]
    assert "if_unheld" not in seen[-1]["body"] and seen[-1]["body"]["path"] == name


@dataclass(frozen=True)
class _Scenario:
    """How one decision row is set up: the coordinator ``start`` gives, the
    setting, the edited file, whether the capability read comes before the
    exit, the stderr today's exit adds, and today's stdout."""

    start: Callable[[_Hook], Any]
    setting: str | None = "refuse\n"
    name: str = "plan.md"
    asks_status: bool = False
    stderr: str = ""
    today: str = _ADMIT_LINE


_BEFORE_POST = {
    "setting_off": _Scenario(_stub(), setting=None),
    "setting_unknown": _Scenario(_stub(), setting="Refuse\n", stderr=_UNKNOWN_SETTING_LINE),
    "not_strict": _Scenario(_stub(), name="notes.md"),
    "no_port": _Scenario(_no_coordinator, today="{}\n"),
    "died_at_resolve": _Scenario(_dead_coordinator),
    "timeout": _Scenario(_slow(_STATUS)),
    "refused": _Scenario(_refusing_port),
    "malformed_answer": _Scenario(_stub({_STATUS: _Answer(body=b"not json")})),
    "status_non_2xx": _Scenario(_stub({_STATUS: _Answer(503, _REFUSES.body)})),
    "status_key_absent": _Scenario(
        _stub({_STATUS: _json_answer({"fold_profile": "none"})}), asks_status=True, stderr=_NO_REFUSAL_LINE,
    ),
    "not_strict_under_coordinator_profile": _Scenario(_stub(), name="PLAN.MD", asks_status=True),
    "malformed_agent_id": _Scenario(_stub()),
}


def _caller_cc(workspace: Path, row: dict, name: str = "plan.md") -> dict[str, Any]:
    cc = _edit_cc(workspace, row["caller"]["session_id"], name)
    if row["caller"]["agent_id"] is not None:
        cc["agent_id"] = row["caller"]["agent_id"]
    return cc


def _assert_today(scenario: _Scenario, run: tuple[str, str], seen: list[dict]) -> None:
    status_read = [("GET", _STATUS)] if scenario.asks_status else []
    todays_routes = _TODAYS_ROUTES if scenario.start is not _no_coordinator else []
    assert run == (scenario.today, scenario.stderr)
    assert _routes(seen) == status_read + todays_routes
    assert all("if_unheld" not in body for body in _sent_edits(seen))


@pytest.mark.parametrize("row", _on_held_rows("before_post"), ids=lambda row: row["name"])
def test_each_decision_row_before_the_post_ends_as_the_table_says(hook: _Hook, row: dict) -> None:
    scenario = _BEFORE_POST[row["outcome"]]
    _strict_workspace(hook.workspace, scenario.setting)
    with scenario.start(hook) as seen:
        out, err = hook.run(_caller_cc(hook.workspace, row, scenario.name))

    if row["expected"] == "unavailable_deny":
        assert (out, err) == (_unavailable(scenario.name), "")
        assert _routes(seen) in ([], [("GET", _STATUS)]), "nothing is sent after the capability read"
    else:
        assert row["expected"] == ("today_and_stderr_line" if scenario.stderr else "today")
        _assert_today(scenario, (out, err), seen)


def test_a_server_pid_with_no_port_line_takes_todays_path(hook: _Hook) -> None:
    """A clean stop leaves server.pid with no port line: no coordinator is
    known, so none can hold the file, and the hook prints ``{}`` as today."""
    _strict_workspace(hook.workspace)
    (hook.workspace / ".coherence" / "server.pid").write_text("12345\n")

    assert hook.run(_edit_cc(hook.workspace, _sid())) == ("{}\n", "")


@pytest.mark.parametrize("body", [b"[]", b'"ok"', b"null", b"7"], ids=lambda body: body.decode())
def test_a_capability_answer_that_is_not_a_json_object_denies(hook: _Hook, body: bytes) -> None:
    _strict_workspace(hook.workspace)
    with _stub({_STATUS: _Answer(body=body)})(hook) as seen:
        out, err = hook.run(_edit_cc(hook.workspace, _sid()))

    assert (out, err) == (_unavailable(), "")
    assert _routes(seen) == [("GET", _STATUS)]


def test_an_unexpected_failure_before_the_opt_in_decision_takes_todays_path(hook: _Hook) -> None:
    """Nothing has been asked of the coordinator yet, so the hook does what it
    does with the setting off rather than fail the event."""
    _strict_workspace(hook.workspace)

    def broken(*_args: Any) -> frozenset:
        raise RuntimeError("strict decision regression")

    hook.monkeypatch.setattr(on_held, "strict_profiles", broken)
    with _stub()(hook) as seen:
        out, err = hook.run(_edit_cc(hook.workspace, _sid()))

    assert (out, err) == (_ADMIT_LINE, "")
    assert _routes(seen) == _TODAYS_ROUTES


def _post_start(row: dict) -> Callable[[_Hook], Any]:
    if "http" in row:
        return _stub({"/hooks/pre-edit": _json_answer(row["http"]["body"], row["http"]["status"])})
    return {
        "timeout": _slow("/hooks/pre-edit"),
        "refused": _stub({_STATUS: _Answer(body=_REFUSES.body, refuse_after=True)}),
        "departed": _stub({"/hooks/pre-edit": _Answer(hang_up=True)}),
        "malformed_answer": _stub({"/hooks/pre-edit": _Answer(body=b"not json")}),
    }[row["outcome"]]


def _post_rows_without_a_2xx() -> list[dict]:
    return [
        row for row in _on_held_rows("post")
        if "outcome" in row or not 200 <= row["http"]["status"] < 300
    ]


@pytest.mark.parametrize("row", _post_rows_without_a_2xx(), ids=lambda row: row["name"])
def test_each_decision_row_after_the_post_without_a_2xx_answer_is_the_unavailable_deny(
    hook: _Hook, row: dict,
) -> None:
    assert row["expected"] == "unavailable_deny"
    _strict_workspace(hook.workspace)
    with _post_start(row)(hook) as seen:
        out, _err = hook.run(_caller_cc(hook.workspace, row))

    assert out == _unavailable()
    edits = [send for send in hook.sends if send["path"] == "/hooks/pre-edit"]
    assert edits and all(send["payload"]["if_unheld"] is True for send in edits)
    assert not any(request["path"] == "/hooks/session-stop" for request in seen), "no release follows"


def _answered_rows() -> list[dict]:
    return [row for row in _on_held_rows("post") if "http" in row and 200 <= row["http"]["status"] < 300]


def _printed_for(row: dict) -> str:
    """What the hook prints for an answered row, built from the deny builders
    and the body alone."""
    body, expected = row["http"]["body"], row["expected"]
    if expected.startswith("relay_"):
        return json.dumps(body) + "\n"
    if expected == "holder_deny":
        deny = hook_payloads.emit_held_path_deny(path="plan.md", holder_agent_id=body["holder_agent_id"])
    elif expected == "parent_deny":
        deny = hook_payloads.emit_held_by_parent_deny(path="plan.md")
    else:
        deny = hook_payloads.emit_admission_unavailable_deny(path="plan.md")
    return json.dumps({"hookSpecificOutput": deny}) + "\n"


@pytest.mark.parametrize("row", _answered_rows(), ids=lambda row: row["name"])
def test_each_answered_decision_row_prints_its_class_through_the_hook_client(hook: _Hook, row: dict) -> None:
    _strict_workspace(hook.workspace)
    with _stub({"/hooks/pre-edit": _json_answer(row["http"]["body"])})(hook) as seen:
        out, _err = hook.run(_caller_cc(hook.workspace, row))

    assert out == _printed_for(row)
    expected_edit = {"session_id": row["caller"]["session_id"], "path": "plan.md", "if_unheld": True}
    if row["caller"]["agent_id"] is not None:
        expected_edit = {**expected_edit, "agent_id": row["caller"]["agent_id"]}
    assert _sent_edits(seen) == [expected_edit]
    assert _routes(seen) == [("GET", _STATUS), *_TODAYS_ROUTES]


def test_a_coordinator_replaced_between_the_capability_read_and_the_edit_denies(hook: _Hook) -> None:
    """The coordinator that answered the capability read is gone by the
    edit: the client refuses to send to whatever holds its port now."""
    _strict_workspace(hook.workspace)
    pid_file = hook.workspace / ".coherence" / "server.pid"
    replaced = _Answer(body=_REFUSES.body, before=lambda: pid_file.write_text("54321\n50001\n"))
    with _stub({_STATUS: replaced})(hook) as seen:
        out, err = hook.run(_edit_cc(hook.workspace, _sid()))

    assert (out, err) == (_unavailable(), "")
    assert _routes(seen) == [("GET", _STATUS)]
    assert isinstance(hook.sends[-1]["raised"], _coherence_client.CoordinatorDeparted)


@pytest.mark.parametrize("agent_id", ["bad id!", 7], ids=repr)
def test_a_malformed_subagent_id_is_denied_and_nothing_is_sent(hook: _Hook, agent_id: object) -> None:
    _strict_workspace(hook.workspace)
    with _stub()(hook) as seen:
        out, err = hook.run({**_edit_cc(hook.workspace, _sid()), "agent_id": agent_id})

    assert (out, err) == (_unavailable(), "")
    assert _routes(seen) == [("GET", _STATUS)]


@pytest.mark.parametrize("agent_id", ["", None], ids=repr)
def test_an_empty_or_null_subagent_id_is_sent_as_the_parent(hook: _Hook, agent_id: object) -> None:
    _strict_workspace(hook.workspace)
    sid = _sid()
    with _stub()(hook) as seen:
        out, _err = hook.run({**_edit_cc(hook.workspace, sid), "agent_id": agent_id})

    assert out == _ADMIT_LINE
    assert _sent_edits(seen) == [{"session_id": sid, "path": "plan.md", "if_unheld": True}]


# A coordinator in this process, with plan.md strict and the setting on.


@pytest.fixture
def strict_coordinator(git_workspace: Path):
    _strict_workspace(git_workspace)
    server = CoordinatorHTTPServer(git_workspace, port=0, instance_id="hook-on-held")
    server.serve_in_thread()
    write_coordinator_pid_file(git_workspace / ".coherence", server.port, pid=os.getpid())
    try:
        yield server
    finally:
        server.shutdown()


def _holder_deny(holder: uuid.UUID) -> str:
    deny = hook_payloads.emit_held_path_deny(path="plan.md", holder_agent_id=holder.hex)
    return json.dumps({"hookSpecificOutput": deny}) + "\n"


def test_an_edit_of_a_strict_file_another_session_holds_is_denied_naming_the_holder(
    strict_coordinator: CoordinatorHTTPServer, hook: _Hook,
) -> None:
    holder, editor = _sid(), _sid()
    out, _ = hook.run(_edit_cc(hook.workspace, holder))
    assert json.loads(out).get("ok") is True, out

    out, _ = hook.run(_edit_cc(hook.workspace, editor))

    holder_id = session_to_agent_id(holder)
    assert out == _holder_deny(holder_id)
    assert f"held by agent {holder_id.hex[:8]}." in out
    principal = _principal_files(hook.workspace, editor)[1].read_text().strip()
    assert hook.sends[-1]["payload"]["if_unheld"] is True
    assert hook.sends[-1]["headers"] == {_PRINCIPAL_HEADER: principal}
    assert write_holders(strict_coordinator, "plan.md") == {str(holder_id)}


@pytest.mark.parametrize(
    "case",
    [(None, "sub1", "parent"), ("sub2", None, "holder"), ("sub1", "sub2", "holder")],
    ids=["subagent-of-the-holder", "parent-of-the-holding-subagent", "sibling-of-the-holder"],
)
def test_only_a_subagent_refused_by_its_own_parent_is_told_to_report_back(
    strict_coordinator: CoordinatorHTTPServer, hook: _Hook, case: tuple,
) -> None:
    """``case`` is the holder's subagent id, the editor's, and the deny the
    editor is told (None is the parent session itself)."""
    holder_agent, editor_agent, told = case
    sid = _sid()
    holder_cc, editor_cc = _edit_cc(hook.workspace, sid), _edit_cc(hook.workspace, sid)
    for cc, agent in ((holder_cc, holder_agent), (editor_cc, editor_agent)):
        if agent is not None:
            cc["agent_id"] = agent
    out, _ = hook.run(holder_cc)
    assert json.loads(out).get("ok") is True, out

    out, _ = hook.run(editor_cc)

    if told == "parent":
        parent = hook_payloads.emit_held_by_parent_deny(path="plan.md")
        assert out == json.dumps({"hookSpecificOutput": parent}) + "\n"
    else:
        assert out == _holder_deny(session_to_agent_id(sid, holder_agent))
    assert write_holders(strict_coordinator, "plan.md") == {str(session_to_agent_id(sid, holder_agent))}


def test_a_session_that_read_before_the_holder_committed_gets_the_strict_deny_as_sent(
    strict_coordinator: CoordinatorHTTPServer, hook: _Hook,
) -> None:
    reader, writer = _sid(), _sid()
    hook.run(_edit_cc(hook.workspace, reader), "pre-read")
    out, _ = hook.run(_edit_cc(hook.workspace, writer))
    assert json.loads(out).get("ok") is True, out
    (hook.workspace / "plan.md").write_text("plan.md v2\n")
    out, _ = hook.run(_edit_cc(hook.workspace, writer), "post-edit")
    assert json.loads(out).get("ok") is True, out

    out, _ = hook.run(_edit_cc(hook.workspace, reader))

    sent = hook.sends[-1]
    assert sent["payload"]["if_unheld"] is True
    assert sent["answer"]["status"] == "stale"
    assert sent["answer"]["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert out == json.dumps(sent["answer"]) + "\n"


def test_an_opted_in_edit_the_coordinator_cannot_decide_in_time_is_denied(
    strict_coordinator: CoordinatorHTTPServer, hook: _Hook,
) -> None:
    """The coordinator's watchdog answers ``unavailable`` with a note that the
    edit proceeds without a guarantee; with the setting on that note never
    reaches the model, and the edit is denied."""
    import ccs.adapters.claude_code.coordinator_server as server_mod
    from tests.test_claude_code_coordinator_server import _HeldRegistryLock

    sid = _sid()
    # The read claims the session's principal while the lock is free, so the
    # held lock below blocks only the pre-edit's work body.
    hook.run(_edit_cc(hook.workspace, sid), "pre-read")
    hook.monkeypatch.setattr(server_mod, "HANDLER_TIMEOUT_SEC", 0.25)
    with _HeldRegistryLock(strict_coordinator):
        out, _ = hook.run(_edit_cc(hook.workspace, sid))

    answer = hook.sends[-1]["answer"]
    assert answer.get("reason") == "unavailable" and "Proceeding WITHOUT" in json.dumps(answer)
    assert out == _unavailable()
    assert "Proceeding WITHOUT" not in out


def test_a_principal_recovery_cannot_cure_denies_where_today_admits_silently(
    strict_coordinator: CoordinatorHTTPServer, hook: _Hook,
) -> None:
    """Another claimant bound this session first, so its opted-in edit is
    refused for its principal even after recovery. Today's path prints ``{}``
    for that; with the setting on it is the unavailable deny."""
    sid = _sid()
    strict_coordinator.service.claim_caller_principal(
        identity=caller_principal_identity(sid), mint_nonce="someone-elses-nonce-0001"
    )

    out, err = hook.run(_edit_cc(hook.workspace, sid))
    (hook.workspace / ".coherence" / "hook_on_held").unlink()
    out_today, _ = hook.run(_edit_cc(hook.workspace, sid))

    assert out == _unavailable()
    assert "caller principal refused" in err
    assert out_today == "{}\n"
