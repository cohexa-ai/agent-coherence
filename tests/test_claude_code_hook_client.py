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
    # Port-file format per lifecycle.read_port_from_file: line 1 pid, line 2 port.
    (coherence / "server.pid").write_text(f"12345\n{port}\n")
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
    import socket

    # Bind-then-close: the freed ephemeral port is near-certainly unbound,
    # so the client's connect gets refused instead of talking to a stranger.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        dead_port = probe.getsockname()[1]
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
    threading.Thread(target=server.serve_forever, daemon=True).start()
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
    time.sleep(0.05)
    assert load_secret(server.coordinator_root)
    (git_workspace / ".coherence" / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
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
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
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
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
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
