# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-hook-client`` — command-type hook handler bridge.

Phase E.0 probe 2A surfaced that Claude Code v2.1.131 rejects hooks.json
at LOAD TIME if any URL contains ``${COHERENCE_PORT}`` — strict-URL
schema validation runs before env-var expansion. Workaround: HTTP-type
hooks are NOT viable; the plugin uses command-type hooks that invoke
this client, which resolves the port from ``.coherence/server.pid``
directly (no env-var dependency).

## Subcommands

Each subcommand maps to one coordinator endpoint and translates the
Claude Code stdin contract → the coordinator's payload contract:

| Subcommand     | CC hook          | Coordinator endpoint      |
|----------------|------------------|---------------------------|
| pre-read       | PreToolUse:Read  | POST /hooks/pre-read      |
| pre-edit       | PreToolUse:Edit  | POST /hooks/pre-edit      |
| post-edit      | PostToolUse      | POST /hooks/post-edit     |
| session-stop   | Stop             | POST /hooks/session-stop  |
| subagent-stop  | SubagentStop     | POST /hooks/session-stop  |
| pre-bash       | PreToolUse:Bash  | POST /hooks/pre-bash      |
| pre-grep       | PreToolUse:Grep  | POST /hooks/pre-grep      |
| session-start  | SessionStart     | POST /hooks/session-start |

``subagent-stop`` shares the session-stop endpoint but scopes the release
to the one subagent's grants. ``session-start`` is the post-compaction
re-grounding hook: only ``source: "compact"`` payloads call out to the
coordinator — ordinary starts, resumes, and clears skip locally.

## stdin contract

Claude Code sends a JSON object on stdin with at least:
- ``session_id`` (UUID string)
- ``tool_name`` (for tool-related hooks)
- ``tool_input`` (dict; for Read/Edit/Write this has ``file_path``)
- ``tool_response`` (for PostToolUse only)

We translate:
- ``file_path`` (absolute) → workspace-relative path via the resolver.
- ``tool_response.success`` (for post-edit; defaults True if missing).
- ``tool_response.content_hash`` if precomputed by an upstream wrapper;
  otherwise the client hashes the post-edit file content on the fly.

## stdout contract

Whatever the coordinator returns. The coordinator's responses already
match Claude Code's ``hookSpecificOutput`` shape for the relevant cases
(stale-read warning, edit-collision warning). For uninteresting cases
(fresh read, ok commit) the response is a small JSON status object that
Claude Code ignores.

Exit code is always 0 on success — even if the coordinator returns
``ok: False``, that's a logical outcome surfaced to the model via
``additionalContext``, not a hook failure.

## Caller principal

Every request presents the session's caller principal in the
``Coherence-Caller-Principal`` header: once a session is claimed, the
coordinator refuses a request naming it without that principal on the routes
where an absent principal admits harm (pre-edit, post-edit, session-stop). One
process per hook event means the principal lives on disk — the mint nonce
first, created exclusively at 0600 and never replaced, then the principal, at
0600 in ``.coherence/``, both keyed by the parent session's derived id — and is
claimed once per session (see
:func:`~ccs.cli._coherence_client.obtain_stored_principal`). A coordinator
whose pid file says ``backend=node`` is not asked at all (it issues none), and
one that answers 404 on the claim (an older Python coordinator) gets no header
either — as before.

A request refused for its principal (a typed ``caller_principal_foreign`` /
``caller_principal_absent`` reason) is recovered by claiming again with the
SAME stored nonce: a principal that differs from the one presented replaces the
stored file and the request is retried once (R20 — this is what brings a
session back after ``state.db`` was reset under it). Anything recovery cannot
cure is reported on stderr and never repaired by deleting and re-minting (see
:func:`~ccs.cli._coherence_client.post_with_stored_principal`). Nothing here
prints a principal or a nonce. Any process that can read ``.coherence/`` can
read these files, so this is convention-enforcement and a detectable unbound
caller, not separation between callers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.error
from pathlib import Path
from typing import Any, Optional, Sequence

from ccs.adapters.claude_code.resolver import find_coordinator_root
from ccs.cli._coherence_client import (
    CoordinatorEndpoint,
    CoordinatorUnavailable,
    err,
    post,
    post_with_stored_principal,
    resolve_endpoint,
)
from ccs.core.exceptions import CallerPrincipalRefused


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-coherence-hook-client",
        description=(
            "Claude Code command-type hook handler that bridges stdin "
            "hook payloads to the local coherence coordinator."
        ),
    )
    parser.add_argument(
        "subcommand",
        choices=[
            "pre-read",
            "pre-edit",
            "post-edit",
            "session-stop",
            "subagent-stop",
            # v0.1.1 KTD-N — H4 mitigation: extended hook coverage.
            "pre-bash",
            "pre-grep",
            # SB-10 U3: post-compaction re-grounding (SessionStart hook).
            "session-start",
        ],
        help="Which coordinator endpoint to invoke. Maps 1:1 to the CC hook event.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override the coordinator root (default: walk up from cwd).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Wraps the real dispatch in a top-level except so that no
    unexpected exception (refactor regression, payload-builder bug, anything
    not anticipated) can violate the always-exit-0 hook contract.

    P2 ce-review fix #8 (reliability): top-level broad except guarantees the
    hook contract — emit ``{}`` and exit 0 no matter what fails.
    P3 ce-review fix #27 (agent-native): isatty guard prevents indefinite
    block when a developer runs the hook-client manually for testing.
    P3 ce-review fix #24 (cli-readiness): empty stdin path now emits ``{}``
    for consistency with the malformed-stdin path (both produce parseable
    JSON for any upstream wrapper doing json.loads on stdout).
    """
    try:
        return _main_inner(argv)
    except SystemExit:
        # argparse / explicit sys.exit — propagate normally
        raise
    except BaseException:
        # CC must never see the hook block its tool call. Even on
        # KeyboardInterrupt or an unexpected programming error, emit the
        # no-op response and exit 0.
        _emit_empty()
        return 0


def _main_inner(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Developer footgun: if stdin is a TTY, the read below would block
    # indefinitely. Emit a brief usage hint to stderr and exit clean so
    # someone testing manually understands what's expected.
    if sys.stdin.isatty():
        print(
            "agent-coherence-hook-client: stdin is a terminal — "
            "this command expects a Claude Code hook JSON payload on stdin.",
            file=sys.stderr,
        )
        _emit_empty()
        return 0

    # Read CC's hook stdin payload.
    try:
        raw = sys.stdin.read()
    except OSError:
        _emit_empty()
        return 0  # no stdin — emit {} for output consistency
    if not raw.strip():
        _emit_empty()  # consistent with malformed-stdin path
        return 0

    try:
        cc_payload = json.loads(raw)
    except json.JSONDecodeError:
        # Don't crash CC; degrade silently.
        _emit_empty()
        return 0

    # Resolve coordinator root + endpoint.
    root = args.root if args.root is not None else find_coordinator_root()
    if root is None:
        _emit_empty()
        return 0
    root_path = Path(root).resolve()

    try:
        endpoint = resolve_endpoint(root_path)
    except CoordinatorUnavailable:
        _emit_empty()
        return 0

    # Dispatch. Broader except below catches any unexpected exception from
    # payload builders or _call (e.g., AttributeError from a malformed
    # cc_payload, refactor-introduced exception type) so hook never blocks.
    try:
        if args.subcommand == "pre-read":
            payload = _build_pre_read(cc_payload, root_path)
            response = _call(endpoint, "/hooks/pre-read", payload, root_path)
        elif args.subcommand == "pre-edit":
            payload = _build_pre_edit(cc_payload, root_path)
            response = _call(endpoint, "/hooks/pre-edit", payload, root_path)
        elif args.subcommand == "post-edit":
            payload = _build_post_edit(cc_payload, root_path)
            response = _call(endpoint, "/hooks/post-edit", payload, root_path)
        elif args.subcommand == "session-stop":
            payload = _build_session_stop(cc_payload)
            response = _call(endpoint, "/hooks/session-stop", payload, root_path)
        elif args.subcommand == "subagent-stop":
            # SB-25 Unit 4: CC's SubagentStop event → release the SUBAGENT
            # identity's grants via the existing session-stop verb. agent_id
            # is REQUIRED — without it the release would strip the PARENT's
            # grants mid-session, so absence skips (fail-open {}).
            payload = _build_subagent_stop(cc_payload)
            response = _call(endpoint, "/hooks/session-stop", payload, root_path)
        elif args.subcommand == "pre-bash":
            payload = _build_pre_bash(cc_payload)
            response = _call(endpoint, "/hooks/pre-bash", payload, root_path)
        elif args.subcommand == "pre-grep":
            payload = _build_pre_grep(cc_payload, root_path)
            response = _call(endpoint, "/hooks/pre-grep", payload, root_path)
        elif args.subcommand == "session-start":
            # SB-10 U3: CC's SessionStart event → post-compaction
            # re-grounding. The builder gates on source == "compact", so
            # startup/resume/clear never reach the coordinator.
            payload = _build_session_start(cc_payload)
            response = _call(endpoint, "/hooks/session-start", payload, root_path)
        else:  # pragma: no cover — argparse already validates
            _emit_empty()
            return 0
    except CoordinatorUnavailable:
        _emit_empty()
        return 0
    except _SkipHook:
        _emit_empty()
        return 0
    except Exception:
        # P2 ce-review fix #8: any unexpected exception from a payload
        # builder, _call, or future refactor must NOT propagate — CC
        # requires the hook to exit clean. The top-level except in main()
        # also catches but that's BaseException-wide; this one preserves
        # KeyboardInterrupt propagation to the outer handler.
        _emit_empty()
        return 0

    if response is None:
        _emit_empty()
        return 0

    # Pass the coordinator's response straight through to CC. Coordinator
    # responses already match CC's hookSpecificOutput shape for the cases
    # that need to inject context (stale read, edit collision).
    print(json.dumps(response), flush=True)
    return 0


class _SkipHook(Exception):
    """Internal signal: this hook invocation has nothing meaningful to do
    (untracked file, missing required field, etc.). Emit empty response."""


def _emit_empty() -> None:
    """Standard 'no-op' hook response — empty JSON object, exit 0."""
    print("{}", flush=True)


def _call(
    endpoint: CoordinatorEndpoint, path: str, payload: dict[str, Any], root: Path
) -> Optional[dict[str, Any]]:
    """POST ``payload``, presenting the session's caller principal.

    The principal is obtained, presented and — when refused — recovered per
    :func:`~ccs.cli._coherence_client.post_with_stored_principal` (stored under
    ``.coherence/``; claimed once per session; re-claimed with the SAME stored
    nonce and retried once on a typed principal refusal). A coordinator that
    issues none (404 on the claim) gets no header and the hook proceeds exactly
    as before.
    """
    try:
        return post_with_stored_principal(
            endpoint, root, path, payload, report=_report, send=post
        )
    except CallerPrincipalRefused as exc:
        # The one rejection worth a stderr line: recovery could not cure it, so
        # coherence is off for this session on the routes that require one.
        # The message is built from the typed reason, never the coordinator's
        # prose, and carries no principal or nonce (R5). Nothing was deleted
        # or re-minted — that would reopen the first-claim gate.
        _report(str(exc))
        return None
    except urllib.error.HTTPError:
        # Coordinator rejected the request (validation error). Degrade
        # silently — the hook should NEVER block the user's tool call.
        return None


def _report(message: str) -> None:
    """One diagnostic line on stderr. Exit code and stdout are unaffected, so
    the always-exit-0 hook contract holds."""
    err(f"agent-coherence-hook-client: {message}")


# ----------------------------------------------------------------------
# Payload translators
# ----------------------------------------------------------------------


def _build_pre_read(cc: dict[str, Any], root: Path) -> dict[str, Any]:
    session_id = _require_session_id(cc)
    file_path = _require_file_path(cc)
    rel = _to_workspace_relative(file_path, root)
    body: dict[str, Any] = {"session_id": session_id, "path": rel}
    # v0.2 KTD-O: compute content_hash from disk so the coordinator's
    # strict-mode gate can disambiguate "session sees stale bytes"
    # (first-observer reading a peer-updated file → strict-deny) from
    # "session sees current bytes" (first-observer with content matching
    # what the registry recorded → warn-mode allow). Without this,
    # PreToolUse:Read carries no hash (Claude Code's Read tool reads
    # AFTER the hook fires, so tool_response.content_hash is undefined
    # at hook time), and the strict-deny gate's hash_differs branch is
    # unreachable. Best-effort: _hash_file returns None on OSError
    # (file missing, permission denied) — coordinator then falls back
    # to the INVALID-only branch of the strict-deny gate, preserving
    # v0.1.1 warn-mode behavior for non-strict workspaces.
    content_hash = _hash_file(Path(file_path))
    if content_hash is not None:
        body["content_hash"] = content_hash
    return _with_agent_id(cc, body)


def _build_pre_edit(cc: dict[str, Any], root: Path) -> dict[str, Any]:
    session_id = _require_session_id(cc)
    file_path = _require_file_path(cc)
    rel = _to_workspace_relative(file_path, root)
    return _with_agent_id(cc, {"session_id": session_id, "path": rel})


def _build_post_edit(cc: dict[str, Any], root: Path) -> dict[str, Any]:
    session_id = _require_session_id(cc)
    file_path = _require_file_path(cc)
    rel = _to_workspace_relative(file_path, root)
    tool_response = cc.get("tool_response") or {}
    # CC's tool_response shape varies; treat missing 'success' as True
    # (if CC fired PostToolUse at all, the tool didn't hard-fail).
    success = bool(tool_response.get("success", True))
    content_hash: Optional[str] = tool_response.get("content_hash")
    if success and content_hash is None:
        # Hash the post-write content from disk. The file path is
        # absolute per CC's stdin contract.
        content_hash = _hash_file(Path(file_path))
    body: dict[str, Any] = {
        "session_id": session_id,
        "path": rel,
        "success": success,
    }
    if content_hash is not None:
        body["content_hash"] = content_hash
    return _with_agent_id(cc, body)


def _build_session_stop(cc: dict[str, Any]) -> dict[str, Any]:
    session_id = _require_session_id(cc)
    return _with_agent_id(cc, {"session_id": session_id})


# Mirrors the coordinator's _SUBAGENT_ID_RE (charset + length). Kept in sync
# by hand across the client/server modules; a divergence is caught by the
# subagent-stop safety test. fullmatch semantics (see below) match the server.
_SUBAGENT_ID_RE = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _build_subagent_stop(cc: dict[str, Any]) -> dict[str, Any]:
    """SB-25 Unit 4: SubagentStop → scoped grant release. The subagent
    identity is mandatory AND shape-validated here (contrast _with_agent_id's
    optional thread): a SubagentStop payload with a missing OR malformed agent
    id must NOT fall back to the parent identity — the server would null a
    malformed id and degrade to releasing the PARENT's live grants (P1). Fail
    open (skip) on a doomed release rather than performing the wrong one."""
    session_id = _require_session_id(cc)
    aid = cc.get("agent_id", cc.get("agentId"))
    if not isinstance(aid, str) or not _SUBAGENT_ID_RE.fullmatch(aid):
        raise _SkipHook("agent_id missing or malformed for subagent-stop")
    return {"session_id": session_id, "agent_id": aid}


def _build_session_start(cc: dict[str, Any]) -> dict[str, Any]:
    """SB-10 U3: SessionStart → post-compaction re-grounding request.

    Only the ``compact`` source qualifies: the coordinator endpoint trusts
    its caller and treats EVERY request as a compact event (R1), so the
    source gate lives HERE — ``startup``/``resume``/``clear`` (and an
    absent source) must skip before any network call. SessionStart carries
    no subagent identity, so the body is session-only: deliberately NOT
    routed through ``_with_agent_id`` — the re-grounding payload is
    session-scoped (it always covers the parent plus every registered
    subagent), and a stray agent field must not narrow it.
    """
    if cc.get("source") != "compact":
        raise _SkipHook("SessionStart source is not 'compact'")
    session_id = _require_session_id(cc)
    return {"session_id": session_id}


def _build_pre_bash(cc: dict[str, Any]) -> dict[str, Any]:
    """v0.1.1 KTD-N — translate CC's Bash PreToolUse payload to the
    /hooks/pre-bash coordinator request shape.

    CC fires PreToolUse Bash with ``tool_input.command`` = the raw shell
    command string the model is about to execute. The coordinator's
    handler detects tracked-artifact reads in that command via the
    Bash path detector and surfaces stale-read warnings.
    """
    session_id = _require_session_id(cc)
    tool_input = cc.get("tool_input") or {}
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        raise _SkipHook("tool_input.command missing or empty")
    return _with_agent_id(cc, {"session_id": session_id, "command": command})


def _build_pre_grep(cc: dict[str, Any], root: Path) -> dict[str, Any]:
    """v0.1.1 KTD-N — translate CC's Grep PreToolUse payload to the
    /hooks/pre-grep coordinator request shape.

    CC's Grep tool takes a ``path`` arg (the search root). Empty/absent →
    workspace root.

    Path shapes observed in production (2026-05-24 launch-gate finding):
    - **Direct invocation:** workspace-relative ("src/lib", "./docs").
    - **Subagent (Task tool) invocation:** ABSOLUTE path
      ("/private/var/.../workspace/src/lib"). Initial v0.1.1 assumption
      that "Grep path is always workspace-relative" was incorrect —
      subagents inherit the parent's CWD shape differently, and CC
      resolves the path to absolute before passing to PreToolUse.

    Fix: normalize absolute paths via _to_workspace_relative. Relative
    paths pass through after stripping a leading "./" for parity with
    the validator's normalization expectations.
    """
    session_id = _require_session_id(cc)
    tool_input = cc.get("tool_input") or {}
    raw_path = tool_input.get("path", "")
    if raw_path is None:
        raw_path = ""
    if not isinstance(raw_path, str):
        raise _SkipHook("tool_input.path must be a string")
    if raw_path and os.path.isabs(raw_path):
        search_root = _to_workspace_relative(raw_path, root)
    else:
        search_root = raw_path[2:] if raw_path.startswith("./") else raw_path
    return _with_agent_id(cc, {"session_id": session_id, "search_root": search_root})


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _require_session_id(cc: dict[str, Any]) -> str:
    sid = cc.get("session_id")
    if not isinstance(sid, str) or not sid:
        raise _SkipHook("session_id missing")
    return sid


def _require_file_path(cc: dict[str, Any]) -> str:
    tool_input = cc.get("tool_input") or {}
    fp = tool_input.get("file_path")
    if not isinstance(fp, str) or not fp:
        raise _SkipHook("tool_input.file_path missing")
    return fp


def _with_agent_id(cc: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
    """SB-25: thread the optional subagent identity through to the
    coordinator. Accepts snake_case ``agent_id`` with a defensive camelCase
    ``agentId`` fallback (wire casing pinned by the R6 live capture).
    Absent/invalid → body unchanged (parent identity)."""
    aid = cc.get("agent_id", cc.get("agentId"))
    if isinstance(aid, str) and aid:
        body["agent_id"] = aid
    return body


def _to_workspace_relative(file_path: str, root: Path) -> str:
    """Convert CC's absolute file_path → workspace-relative path the
    coordinator expects. If the path is outside the workspace (shouldn't
    happen but defensive), skip the hook rather than send garbage."""
    try:
        rel = Path(file_path).resolve().relative_to(root)
    except ValueError:
        raise _SkipHook(f"path outside workspace root: {file_path}")
    return str(rel)


def _hash_file(path: Path) -> Optional[str]:
    """SHA-256 of the file at the given absolute path. Returns None on
    read failure (don't crash the hook)."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
