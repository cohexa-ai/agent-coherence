"""The hook client's refusal of held strict files.

A workspace turns it on with the one-line file ``.coherence/hook_on_held``
holding ``refuse``. With it on, the hook client's ``pre-edit`` of a strict
file asks the coordinator for the file only if no other agent holds it, and
turns any other answer into a deny. This module holds the decisions, with no
I/O of their own: the setting, whether a path is strict judged the way the
coordinator judges it, what the coordinator's ``/status`` says, whether a
subagent id may be sent, and what the hook prints for the coordinator's answer.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, get_args

from ccs.adapters.claude_code.coordinator_server import (
    caller_principal_identity,
    has_subagent_id_field,
    read_subagent_id,
)
from ccs.adapters.claude_code.hook_payloads import (
    emit_admission_unavailable_deny,
    emit_held_by_parent_deny,
    emit_held_path_deny,
)
from ccs.adapters.claude_code.policy import PolicyMatchContext, TrackedArtifactPolicy
from ccs.adapters.claude_code.regular_file import ReadStatus, read_state_file
from ccs.core.disk_spelling import native_spelling
from ccs.core.exceptions import OTHER_HOLDER_REASON
from ccs.core.path_fold import FoldProfile

SETTING_FILE = "hook_on_held"
SETTING_ON = "on"
SETTING_OFF = "off"
SETTING_UNKNOWN = "unknown"

UNKNOWN_SETTING_MESSAGE = (
    ".coherence/hook_on_held holds neither refuse nor displace, so edits of "
    "held strict files are not refused"
)
"""The one stderr line for an unknown value. Built from constants only."""

NO_REFUSAL_MESSAGE = (
    "the coordinator does not say it refuses edits of held files, so this edit "
    "of a strict file was not checked for another holder"
)
"""The one stderr line when ``/status`` lacks ``pre_edit_if_unheld: true``.
Built from constants only."""

STATUS_ROUTE = "/status?detail=metrics"
"""The tier that carries the capability without walking every session, which
on a busy coordinator costs seconds."""

RELAY_ADMIT = "relay_admit"
RELAY_DENY = "relay_deny"
HOLDER_DENY = "holder_deny"
PARENT_DENY = "parent_deny"
UNAVAILABLE_DENY = "unavailable_deny"

# The holder a refusal names is an agent id in its 32-hex form; anything else
# cannot be shown as a short id, so the answer is not read as a refusal.
_HOLDER_ID_RE = re.compile(r"[0-9a-f]{32}")

_ON_VALUE = b"refuse"
_OFF_VALUES = frozenset({b"", b"displace"})
# Far above any valid value; a longer file is an unknown value, not a read.
_SETTING_MAX_BYTES = 64

FOLD_PROFILES: tuple[FoldProfile, ...] = get_args(FoldProfile)
# The fold_profile a coordinator advertises when its start could not classify
# the disk; it keys names as typed, as under "none".
_UNCLASSIFIED_DISK = "cannot_tell"


def read_setting(coherence_dir: Path) -> str:
    """``on``, ``off`` or ``unknown`` for ``coherence_dir/hook_on_held``. A file
    that is absent, a link, not a regular file or unreadable (and a linked
    ``.coherence``) reads as off: it says nothing a user wrote. One trailing
    newline is the line ending a one-line file carries, so it is dropped."""
    read = read_state_file(coherence_dir, SETTING_FILE, max_bytes=_SETTING_MAX_BYTES)
    if read.status is ReadStatus.TOO_LARGE:
        return SETTING_UNKNOWN
    if read.status is not ReadStatus.READ:
        return SETTING_OFF
    return _setting_of(read.data or b"")


def _setting_of(data: bytes) -> str:
    value = data[:-1] if data.endswith(b"\n") else data
    if value == _ON_VALUE:
        return SETTING_ON
    if value in _OFF_VALUES:
        return SETTING_OFF
    return SETTING_UNKNOWN


def strict_profiles(policy: TrackedArtifactPolicy, root: Path, rel: str) -> frozenset[FoldProfile]:
    """The fold profiles under which ``rel`` is strict, each read with the
    disk spelling a coordinator on that profile would use. Before the client
    knows the coordinator's profile, a non-empty answer makes ``rel`` a
    candidate; once it knows, the profile's membership decides."""
    if not policy.strict_mode_paths:
        return frozenset()
    return frozenset(
        profile
        for profile in FOLD_PROFILES
        if policy.is_strict_mode(rel, _match_context(root, rel, profile))
    )


def _match_context(root: Path, rel: str, profile: FoldProfile) -> PolicyMatchContext:
    return PolicyMatchContext(profile, "file", _disk_spelling_of(root, rel, profile))


def _disk_spelling_of(root: Path, rel: str, profile: FoldProfile) -> str | None:
    """The disk's own spelling of ``rel``, which only a folding profile reads;
    one it cannot find leaves the other readings, as on the coordinator."""
    if profile == "none":
        return None
    try:
        return os.path.relpath(native_spelling(root / rel), native_spelling(root))
    except (OSError, ValueError):
        return None


def advertises_refusal(status: dict[str, Any]) -> bool:
    """Whether ``/status`` says this coordinator refuses a pre-edit that opts
    in while another agent holds the path. Only JSON ``true`` does: a
    coordinator without the key ignores the flag and displaces the holder."""
    return status.get("pre_edit_if_unheld") is True


def strict_for_coordinator(status: dict[str, Any], candidates: frozenset[FoldProfile]) -> bool:
    """Whether the path whose strict profiles are ``candidates`` is strict on
    the coordinator that answered ``status``. An absent ``fold_profile`` is a
    coordinator that folds nothing, and ``cannot_tell`` is a disk its start
    could not classify, which it keys as ``none``."""
    profile = status.get("fold_profile", "none")
    if profile == _UNCLASSIFIED_DISK:
        profile = "none"
    if profile in FOLD_PROFILES:
        return profile in candidates
    # A profile this client cannot read may fold the path onto a strict entry,
    # and it is strict under some profile: when unsure, refuse.
    return True


def malformed_subagent_id(cc: dict[str, Any]) -> bool:
    """Whether the Claude Code payload carries a subagent id the coordinator
    would refuse on an opted-in edit. Absent, null and empty are the parent,
    as on every route; sending a malformed one as the parent would let a
    subagent take its parent's place."""
    return has_subagent_id_field(cc) and read_subagent_id(cc) is None


def classify_answer(body: object, request: dict[str, Any]) -> str:
    """The outcome class of the coordinator's 2xx ``body`` to the opted-in
    ``request``. An allow-list: only a clean admit admits, a deny the
    coordinator wrote passes through, a holder refusal with a readable holder
    is named, and everything else is the unavailable deny. A ``collision`` or
    ``degraded`` key means the flag was ignored or the check did not run."""
    if not isinstance(body, dict):
        return UNAVAILABLE_DENY
    if body.get("ok") is True and "degraded" not in body and "collision" not in body:
        return RELAY_ADMIT
    output = body.get("hookSpecificOutput")
    if isinstance(output, dict) and output.get("permissionDecision") == "deny":
        return RELAY_DENY
    holder = body.get("holder_agent_id")
    if body.get("reason") != OTHER_HOLDER_REASON or not _is_holder_id(holder):
        return UNAVAILABLE_DENY
    return PARENT_DENY if _held_by_parent(holder, request) else HOLDER_DENY


def _is_holder_id(value: object) -> bool:
    return isinstance(value, str) and _HOLDER_ID_RE.fullmatch(value) is not None


def _held_by_parent(holder: str, request: dict[str, Any]) -> bool:
    """Only a subagent can recompute its parent's agent id from what it sent;
    a parent refused by its own subagent, or a sibling, cannot, so they are
    told of another holder."""
    if read_subagent_id(request) is None:
        return False
    return holder == caller_principal_identity(request["session_id"]).hex


def unavailable_deny(path: str) -> dict[str, Any]:
    """What the hook prints when the coordinator could not confirm that no
    other agent holds ``path``."""
    return {"hookSpecificOutput": emit_admission_unavailable_deny(path=path)}


def answer_for(body: Any, request: dict[str, Any]) -> Any:
    """What the hook prints for the coordinator's 2xx ``body`` to the opted-in
    ``request``: the body itself when it admits or already denies, else the
    deny its class names, with no other top-level key."""
    outcome = classify_answer(body, request)
    path = request["path"]
    if outcome in (RELAY_ADMIT, RELAY_DENY):
        return body
    if outcome == HOLDER_DENY:
        return {"hookSpecificOutput": emit_held_path_deny(path=path, holder_agent_id=body["holder_agent_id"])}
    if outcome == PARENT_DENY:
        return {"hookSpecificOutput": emit_held_by_parent_deny(path=path)}
    return unavailable_deny(path)
