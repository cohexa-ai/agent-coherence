# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Minimal deny-site audit log — appends strict-mode denial events as
JSONL to ``<coordinator_root>/.coherence/audit.log`` (v0.2 plan Unit 4,
KTD-V compressed).

Why a separate module:
- Single-responsibility helper that the 4 strict-deny call sites import.
- Easier to test in isolation without spinning a full coordinator.
- Future v0.2.x callback-surface plan (deferred per plan § Deferred to
  Separate Tasks) can layer atop this without changing call-site code.

KTD-V minimal-scope invariants (security review 2026-05-21):
- **Denial events only.** No warn-mode entries, no allow events.
  ``decision`` is always ``"strict_deny"`` in v0.2.
- **No schema_version field.** Adding it now would lock the schema; the
  v0.2.x callback surface design will introduce versioning when there's
  a real consumer.
- **Bounded payload.** No Bash command bodies, no tool input strings,
  no user content. Only path / session_id / tool name / ISO timestamp /
  decision marker. Test ``test_audit_log_payload_bounded_no_user_content``
  guards against accidental field expansion.
- **0o600 file mode.** Created with explicit mode flag. Mode drift on
  an existing file (operator chmod 0644) is logged as a warning + bumped
  on a counter; the append still proceeds because the data is denials-
  only metadata, not credential material.

Concurrent writes: ``os.O_APPEND`` is atomic for small writes on POSIX
(per the kernel; the write syscall is serialized between concurrent
appenders). One JSONL line per write keeps each append atomic — no
locking needed at the application layer.
"""

from __future__ import annotations

import json
import logging
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from ccs.adapters.claude_code.regular_file import open_regular

logger = logging.getLogger(__name__)


_AUDIT_LOG_FILENAME = "audit.log"
_REQUIRED_MODE = 0o600
"""KTD-V locked: audit.log must be mode 0o600 (owner read/write only).
Tests pin this constant via ``test_audit_log_required_mode_constant``."""


ToolSurface = Literal["Read", "Edit", "Write", "Bash", "Grep"]


def _resolve_audit_log_path(coordinator_root: Path) -> Path:
    """Compute ``<coordinator_root>/.coherence/audit.log``. The directory
    is expected to exist (coordinator startup creates it); this helper
    does NOT create it — failure to find it surfaces as a coordinator
    misconfiguration, not silent log loss."""
    return coordinator_root / ".coherence" / _AUDIT_LOG_FILENAME


def _check_mode_or_warn(path: Path) -> bool:
    """Return True iff existing file mode matches _REQUIRED_MODE. On
    drift, emit a logger.warning and return False — the caller decides
    whether to still append. KTD-V choice: still append (denials-only
    metadata is not credential material; refusing would silently drop
    audit data after an operator chmod)."""
    try:
        actual_mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return True  # File doesn't exist yet — will be created with 0o600.
    if actual_mode != _REQUIRED_MODE:
        logger.warning(
            "audit.log at %s has mode %o (expected %o). Continuing append "
            "— denials-only metadata is not credential material — but the "
            "drift is operator-visible via the audit_log_mode_drift_total "
            "counter. Run `chmod 0600 .coherence/audit.log` to restore.",
            path, actual_mode, _REQUIRED_MODE,
        )
        return False
    return True


def append_strict_deny(
    coordinator_root: Path,
    *,
    agent_id: str,
    path: str,
    tool: ToolSurface,
) -> bool:
    """Append a single JSONL line to ``.coherence/audit.log`` recording a
    v0.2 strict-mode denial event.

    Returns:
        True if the append succeeded; False on OSError (disk full, file
        unreadable, a symlink or anything not a regular file at the path).
        Errors are LOGGED, not raised — the caller's
        coordinator-state transaction has already committed when this is
        called, and we never want audit-log failures to corrupt that
        transaction by raising.

    Args:
        coordinator_root: Workspace root containing ``.coherence/``.
        agent_id: Session UUID of the denied agent (full UUID string, not
            short form; the operator's later forensic query needs the
            full identifier to join against ``state.db``).
        path: Repo-relative path of the artifact the deny fired on.
        tool: The Claude Code tool surface that originated the deny
            (one of the 5 hooked surfaces).

    KTD-V bounded payload — fields fixed at:
        ts        — ISO-8601 UTC timestamp
        artifact  — repo-relative path
        agent     — full session UUID
        tool      — tool surface name (Literal-bounded)
        decision  — always "strict_deny" in v0.2

    NO command bodies. NO content hashes. NO user prose. NO
    schema_version (deferred to v0.2.x callback surface plan)."""
    audit_path = _resolve_audit_log_path(coordinator_root)
    # Defensive mode check before append. Returns True on first-call
    # (file doesn't exist yet) or when mode is correct.
    mode_ok = _check_mode_or_warn(audit_path)
    record = {
        "ts": datetime.now(tz=timezone.utc).isoformat(),
        "artifact": path,
        "agent": agent_id,
        "tool": tool,
        "decision": "strict_deny",
    }
    payload = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        # O_APPEND atomic write for small (<PIPE_BUF=4096 bytes) payloads
        # on POSIX. Mode 0o600 only applied on file CREATION; existing
        # file keeps its mode (verified above). A regular file only, never
        # through a link (#498): deny rows would land in the file a link
        # names, and a FIFO with no reader would block the deny.
        fd = open_regular(
            audit_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            _REQUIRED_MODE,
        )
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
        # Return False if mode drift was detected (caller bumps the
        # drift counter); True otherwise. The append happened either
        # way — the boolean is purely an out-of-band signal for the
        # counter side-effect.
        return mode_ok
    except OSError as exc:
        logger.error(
            "audit.log append failed at %s: %s. Denial event NOT recorded "
            "in audit log (coordinator-state transaction already committed; "
            "the agent's deny went through normally).",
            audit_path, exc,
        )
        return False


__all__ = [
    "ToolSurface",
    "append_strict_deny",
]
