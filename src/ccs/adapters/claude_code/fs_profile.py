# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The workspace disk's fold profile, found once per coordinator start.

Eleven vectors ask the disk itself whether two spellings reach one file.
Vector 1 compares ``.COHERENCE`` with ``.coherence`` and creates nothing.
Vectors 2 to 11 each create one spelling with ``O_CREAT | O_EXCL`` in a fresh
directory under ``.coherence`` and look up the other. A vector answers yes when
both spellings reach the same ``(st_dev, st_ino)``. The answers either match
one of the known profiles exactly, or the disk is cannot-tell and folds
nothing: guessing a fold could merge two real files, or split one.

The probe runs only while this process holds the store's single-writer lock,
so it can sweep the probe directories a crashed start left without unlinking
the files of a start that is probing right now. Nothing about the profile is
written to the store: it is kept in memory and found again at every start.
The plugin's Node coordinator runs the same vectors, answer table, sweep and
reasons.
"""

from __future__ import annotations

import errno
import os
import secrets
import stat
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar

from ccs.core.path_fold import FoldProfile

LockState = Literal["held", "unlocked"]
ProbeStatus = Literal["classified", "cannot_tell"]

PROBE_DIR_PREFIX = "fold-probe-"

_T = TypeVar("_T")


@dataclass(frozen=True)
class FoldProfileResult:
    """The disk's profile; ``reason`` says why it could not be classified and
    is set on ``cannot_tell`` only."""

    profile: FoldProfile
    status: ProbeStatus
    reason: str | None = None


@dataclass(frozen=True)
class FoldProbeInput:
    """What the probe needs. ``log`` receives one line for each stale entry
    the sweep leaves in place; ``device_of`` is the device a path is on,
    replaced in tests to put ``.coherence`` on another one."""

    root: Path
    coherence_dir: Path
    lock_state: LockState
    log: Callable[[str], None]
    device_of: Callable[[Path], int] | None = None


#: Vectors 2 to 11: the spelling created, then the spelling looked up. Each
#: carries a ``p`` before it in the file name, as when the answers were measured.
_CREATED_VECTORS: tuple[tuple[str, str], ...] = (
    ("\u00e9", "e\u0301"),
    ("\u00df", "ss"),
    ("\u212a", "K"),
    ("\u212a", "k"),
    ("\uab70", "\u13a0"),
    ("\u03c2", "\u03c3"),
    ("a\u0323\u0302", "a\u0302\u0323"),
    ("\u0131", "i"),
    ("a\u200cb", "ab"),
    ("\uff41", "a"),
)

#: Measured: case-insensitive APFS, case-sensitive APFS, and a disk that folds nothing.
_KNOWN_ANSWERS: dict[str, FoldProfile] = {
    "11111111000": "caseless",
    "01010001000": "nfd",
    "00000000000": "none",
}

_CREATE_FLAGS = os.O_CREAT | os.O_EXCL | os.O_WRONLY


def _cannot_tell(reason: str) -> FoldProfileResult:
    return FoldProfileResult(profile="none", status="cannot_tell", reason=reason)


def classify_vector_answers(answers: Sequence[bool]) -> FoldProfileResult:
    """The profile whose measured answers these are, else cannot-tell."""
    pattern = "".join("1" if answer else "0" for answer in answers)
    profile = _KNOWN_ANSWERS.get(pattern)
    if profile is None:
        return _cannot_tell(f"answers {pattern} match no known profile")
    return FoldProfileResult(profile=profile, status="classified")


class _ProbeStepFailed(Exception):
    """One probe step failed; the message is the cannot-tell reason."""


def _error_code(exc: OSError) -> str:
    return errno.errorcode.get(exc.errno, str(exc)) if exc.errno is not None else str(exc)


def _attempt(failure: str, step: Callable[[], _T]) -> _T:
    try:
        return step()
    except OSError as exc:
        raise _ProbeStepFailed(f"{failure}: {_error_code(exc)}") from exc


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _same_file(first: os.stat_result, second: os.stat_result | None) -> bool:
    return second is not None and (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _answer_case_vector(root: Path) -> bool:
    # lstat, never stat: a committed .COHERENCE link to .coherence is a link,
    # not case folding.
    return _attempt(
        "vector 1 could not be read",
        lambda: _same_file(os.lstat(root / ".coherence"), _lstat_or_none(root / ".COHERENCE")),
    )


def _answer_created_vector(probe_dir: Path, vector: tuple[str, str], number: int) -> bool:
    created, probed = vector
    created_path = probe_dir / f"p{created}"
    _attempt(f"vector {number} could not be created", lambda: os.close(os.open(created_path, _CREATE_FLAGS, 0o600)))
    try:
        return _attempt(
            f"vector {number} could not be read",
            lambda: _same_file(os.lstat(created_path), _lstat_or_none(probe_dir / f"p{probed}")),
        )
    finally:
        _attempt(f"vector {number} could not be removed", lambda: os.unlink(created_path))


def _answer_created_vectors(coherence_dir: Path) -> list[bool]:
    probe_dir = coherence_dir / f"{PROBE_DIR_PREFIX}{os.getpid()}-{secrets.token_hex(4)}"
    _attempt("probe directory could not be created", lambda: os.mkdir(probe_dir, 0o700))
    try:
        return [
            _answer_created_vector(probe_dir, vector, number)
            for number, vector in enumerate(_CREATED_VECTORS, start=2)
        ]
    finally:
        _attempt("probe directory could not be removed", lambda: os.rmdir(probe_dir))


def _remove_regular_file(path: Path, log: Callable[[str], None]) -> bool:
    if not stat.S_ISREG(os.lstat(path).st_mode):
        log(f"left {path} in place: not a regular file")
        return False
    os.unlink(path)
    return True


def _sweep_entry(path: Path, log: Callable[[str], None]) -> bool:
    """Remove one stale probe directory: only the regular files directly in
    it, then the directory once it is empty. A link is never followed or
    entered."""
    try:
        if not stat.S_ISDIR(os.lstat(path).st_mode):
            log(f"left {path} in place: not a directory")
            return False
        # Every entry is tried, so each one left in place is logged.
        removed = [_remove_regular_file(path / name, log) for name in os.listdir(path)]
        if not all(removed):
            return False
        os.rmdir(path)
        return True
    except OSError as exc:
        log(f"left {path} in place: {_error_code(exc)}")
        return False


def _sweep_stale_probe_dirs(coherence_dir: Path, log: Callable[[str], None]) -> bool:
    names = _attempt(".coherence could not be listed", lambda: os.listdir(coherence_dir))
    swept = [_sweep_entry(coherence_dir / name, log) for name in names if name.startswith(PROBE_DIR_PREFIX)]
    return all(swept)


def _stat_device(path: Path) -> int:
    return _attempt(f"{path} could not be read", lambda: os.stat(path).st_dev)


def _classify_locked_disk(probe: FoldProbeInput) -> FoldProfileResult:
    # A planted entry can make the start cannot-tell; it never decides a profile.
    if not _sweep_stale_probe_dirs(probe.coherence_dir, probe.log):
        return _cannot_tell("stale probe entries left in place")
    device_of = probe.device_of or _stat_device
    # Probe files elsewhere would classify that other disk, not the workspace's.
    if device_of(probe.root) != device_of(probe.coherence_dir):
        return _cannot_tell("coherence dir on another device")
    case_answer = _answer_case_vector(probe.root)
    return classify_vector_answers([case_answer, *_answer_created_vectors(probe.coherence_dir)])


def probe_fold_profile(probe: FoldProbeInput) -> FoldProfileResult:
    """Classify the workspace disk. Never raises for a filesystem failure: a
    step that fails makes the disk cannot-tell, with the step as the reason."""
    # Without the lock another start may be probing: a sweep could unlink its files.
    if probe.lock_state != "held":
        return _cannot_tell("unlocked")
    try:
        return _classify_locked_disk(probe)
    except _ProbeStepFailed as exc:
        return _cannot_tell(str(exc))


def fold_profile_log_line(result: FoldProfileResult) -> str:
    """The one line a start logs about its disk."""
    line = f"workspace disk fold profile: {result.profile}"
    return line if result.status == "classified" else f"{line} (cannot tell: {result.reason or ''})"
