# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-track`` — add one or more paths to the tracked set.

Validates each path (relative, no traversal), then calls the coordinator's
POST /policy/track endpoint which both appends to ``tracked.yaml`` and
reloads the live policy. Idempotent.

Exit codes:
- 0: all paths accepted (or partially accepted with warnings)
- 1: not in a git repo / all paths rejected by validation
- 2: coordinator unreachable / HTTP error / a refused redirect / a TLS
  failure / an answer that is not a /policy/track answer
"""

from __future__ import annotations

import argparse
import urllib.error
from pathlib import Path
from typing import Any, Sequence

from ccs.adapters.claude_code.resolver import find_coordinator_root
from ccs.cli._coherence_client import (
    CoordinatorUnavailable,
    err,
    escape_nonprintable,
    http_error_line,
    http_status_from_error,
    normalize_workspace_path,
    post,
    resolve_endpoint,
)
from ccs.core.exceptions import RedirectRefused, TlsConfigError, TlsVerificationFailed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-coherence-track",
        description="Add one or more paths to the coordinator's tracked set.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more paths to track. Accepts workspace-relative paths "
            "(e.g. 'docs/plan.md') OR absolute paths inside the workspace "
            "root (auto-normalized to workspace-relative before send). "
            "Absolute paths outside the workspace are rejected. No '..' "
            "traversal."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override the coordinator root (default: walk up from cwd to git root).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = args.root if args.root is not None else find_coordinator_root()
    if root is None:
        err("agent-coherence-track: not in a git repository")
        return 1

    # Local pre-validation + normalization so we can fail fast without a
    # network round-trip. normalize_workspace_path accepts both relative
    # paths (e.g. "docs/plan.md") and absolute paths inside the workspace
    # root (e.g. "/Users/x/repo/docs/plan.md") — the latter auto-strips to
    # workspace-relative before send. Absolute paths outside root are
    # rejected. This matches the operator UX expectation when the path is
    # passed verbatim from the /agent-coherence:track skill template.
    invalid: list[tuple[str, str]] = []
    valid: list[str] = []
    for p in args.paths:
        normalized, reason = normalize_workspace_path(p, Path(root))
        if reason is not None:
            invalid.append((p, reason))  # original input in error for clarity
        else:
            valid.append(normalized)  # normalized form to coordinator

    if not valid:
        for p, reason in invalid:
            err(f"agent-coherence-track: rejected {p!r}: {reason}")
        return 1

    try:
        endpoint = resolve_endpoint(Path(root))
        payload = post(endpoint, "/policy/track", {"paths": valid})
    except (CoordinatorUnavailable, TlsVerificationFailed, TlsConfigError) as exc:
        err(f"agent-coherence-track: {escape_nonprintable(exc)}")
        return 2
    except urllib.error.HTTPError as exc:
        err(f"agent-coherence-track: {http_error_line(exc.code, http_status_from_error(exc))}")
        return 2
    except RedirectRefused as exc:
        err(f"agent-coherence-track: the coordinator redirected the request (HTTP {exc.status}); not followed")
        return 2
    if not isinstance(payload, dict):
        err("agent-coherence-track: the coordinator's answer is not a JSON object")
        return 2

    try:
        _report_answer(payload, Path(root))
    except (TypeError, AttributeError):
        err("agent-coherence-track: unexpected /policy/track answer shape")
        return 2
    for p, reason in invalid:
        err(f"agent-coherence-track: rejected {p!r}: {reason}")

    return 0


def _report_answer(payload: dict[str, Any], root: Path) -> None:
    """Print the coordinator's ``added`` and ``rejected`` paths, its text
    escaped (#245): a rejected path in the quoted ``repr`` form the command's
    own rejections use."""
    for p in payload.get("added", []):
        # Success → stdout (machine-parseable by callers). Warn-on-stderr
        # if the path doesn't exist on disk yet (operationally fine, but
        # worth surfacing as diagnostic info).
        on_disk = (root / p).exists()
        print(f"agent-coherence-track: tracked {escape_nonprintable(p)}", flush=True)
        if not on_disk:
            err(f"agent-coherence-track: warning: {escape_nonprintable(p)} does not exist on disk yet")
    for entry in payload.get("rejected", []):
        err(
            f"agent-coherence-track: rejected {entry.get('path', '')!r}: "
            f"{escape_nonprintable(entry.get('reason', ''))}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
