# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-untrack`` — append paths to the workspace ignored set.

Symmetric to :mod:`coherence_track`, but writes to ``ignored.yaml`` via
the coordinator's POST /policy/untrack endpoint. Does NOT delete existing
artifact rows from SQLite (preserves audit trail); future reads simply
suppress warnings because the path is excluded.

Exit codes:
- 0: all paths accepted
- 1: not in a git repo / all paths rejected by local validation
- 2: coordinator unreachable / HTTP error / a refused redirect / a TLS
  failure / an answer that is not a /policy/untrack answer
- 3: refused because a path is enforced in strict mode (#261) — the
  coordinator answered ``reason: untrack_strict_path`` and wrote nothing.
  Untracking a strict path takes a coordinator restart without its entry in
  ``.coherence/strict_mode.yaml``.
"""

from __future__ import annotations

import argparse
import urllib.error
from pathlib import Path
from typing import Any, Sequence

from ccs.adapters.claude_code.policy import UNTRACK_STRICT_PATH_REASON
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
        prog="agent-coherence-untrack",
        description="Append one or more paths to the coordinator's ignored set.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help=(
            "One or more paths to ignore. Accepts workspace-relative paths "
            "(e.g. 'docs/draft.md') OR absolute paths inside the workspace "
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
        err("agent-coherence-untrack: not in a git repository")
        return 1

    # See coherence_track.py for rationale: normalize_workspace_path lets
    # the operator pass absolute paths inside the workspace and the CLI
    # auto-strips to workspace-relative before send. Matches the
    # /agent-coherence:untrack skill template UX (which substitutes
    # $ARGUMENTS verbatim).
    invalid: list[tuple[str, str]] = []
    valid: list[str] = []
    for p in args.paths:
        normalized, reason = normalize_workspace_path(p, Path(root))
        if reason is not None:
            invalid.append((p, reason))
        else:
            valid.append(normalized)

    if not valid:
        for p, reason in invalid:
            err(f"agent-coherence-untrack: rejected {p!r}: {reason}")
        return 1

    try:
        endpoint = resolve_endpoint(Path(root))
        payload = post(endpoint, "/policy/untrack", {"paths": valid})
    except (CoordinatorUnavailable, TlsVerificationFailed, TlsConfigError) as exc:
        err(f"agent-coherence-untrack: {escape_nonprintable(exc)}")
        return 2
    except urllib.error.HTTPError as exc:
        body = http_status_from_error(exc)
        if body is not None and body.get("reason") == UNTRACK_STRICT_PATH_REASON:
            # Classified by the typed reason, never by the error text.
            _report_strict_refusal(body)
            return 3
        err(f"agent-coherence-untrack: {http_error_line(exc.code, body)}")
        return 2
    except RedirectRefused as exc:
        err(f"agent-coherence-untrack: the coordinator redirected the request (HTTP {exc.status}); not followed")
        return 2
    if not isinstance(payload, dict):
        err("agent-coherence-untrack: the coordinator's answer is not a JSON object")
        return 2

    try:
        removed = list(payload.get("removed", []))
    except TypeError:
        err("agent-coherence-untrack: unexpected /policy/untrack answer shape")
        return 2
    for p in removed:
        # Success → stdout (machine-parseable by callers).
        print(f"agent-coherence-untrack: untracked {escape_nonprintable(p)}", flush=True)
    for p, reason in invalid:
        err(f"agent-coherence-untrack: rejected {p!r}: {reason}")

    return 0


def _report_strict_refusal(body: dict[str, Any]) -> None:
    """One line per path the strict refusal names -- the path in the quoted
    ``repr`` form, its strict patterns escaped (#245) -- then what to do. An
    entry of the wrong shape is skipped and patterns that are not a list read
    ``?``: the refusal is the typed reason, whatever its entries hold."""
    refused = body.get("refused")
    for entry in refused if isinstance(refused, list) else []:
        if not isinstance(entry, dict):
            continue
        patterns = entry.get("strict_patterns", [])
        named = ", ".join(map(escape_nonprintable, patterns)) if isinstance(patterns, list) else "?"
        err(f"agent-coherence-untrack: refused {entry.get('path')!r}: enforced in strict mode by {named}")
    err(
        "agent-coherence-untrack: nothing was untracked. A strict path stays "
        "enforced while the coordinator runs; remove its entry from "
        ".coherence/strict_mode.yaml and restart the coordinator to untrack it."
    )


if __name__ == "__main__":
    raise SystemExit(main())
