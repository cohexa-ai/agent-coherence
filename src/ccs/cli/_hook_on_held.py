"""The hook client's refusal of held strict files.

A workspace turns it on with the one-line file ``.coherence/hook_on_held``
holding ``refuse``. With it on, the hook client's ``pre-edit`` of a strict
file asks the coordinator for the file only if no other agent holds it, and
turns any other answer into a deny. This module holds the decisions that come
before any request: the setting, and whether a path is strict, judged the way
the coordinator judges it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import get_args

from ccs.adapters.claude_code.policy import PolicyMatchContext, TrackedArtifactPolicy
from ccs.adapters.claude_code.regular_file import ReadStatus, read_state_file
from ccs.core.disk_spelling import native_spelling
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

_ON_VALUE = b"refuse"
_OFF_VALUES = frozenset({b"", b"displace"})
# Far above any valid value; a longer file is an unknown value, not a read.
_SETTING_MAX_BYTES = 64

FOLD_PROFILES: tuple[FoldProfile, ...] = get_args(FoldProfile)


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
