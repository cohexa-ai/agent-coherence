# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The basis check's vocabulary and its one per-member classifier.

A caller that read several files records, for each, the path, the version and
the registration it read. "Is every input still what I read, and if not, which
moved and why" is answered here, once, as pure functions and plain types, so the
in-process service, the coordinator route and the volume client cannot drift:
the same reason for the same state on every surface, and one wire form with one
parse for every answer. The module is core for the reason ``fence`` is: it takes
plain values only, and an adapters home would bar the application layer.

Nothing here reads a registry. The caller resolves a member by lookup only and
hands the classifier what it found (or that it found nothing).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from uuid import UUID

from .exceptions import (
    BASIS_HOLD_REASONS,
    BASIS_MEMBER_REPLACED,
    BASIS_MEMBER_UNMANAGED,
    COMMIT_UNCONFIRMED_REASON,
    GIVER_FENCED_REASON,
    HOLD_INPUT_VANISHED,
    HOLD_VERSION_MOVED,
    HOLD_VERSION_UNCONFIRMED,
    OCC_CALLER_TRANSIENT_REASON,
    OCC_IS_SHARED_OR_INVALID_ONLY_REASON,
    OTHER_HOLDER_REASON,
    STALE_READ_GENERATION_REASON,
)

# What the caller does next. ``rederive``: an input moved, so re-read and redo
# the work (re-running the check alone never helps). ``retry``: nothing moved,
# the same request may succeed later.
RECOVERY_REDERIVE = "rederive"
RECOVERY_RETRY = "retry"
RECOVERY_CLASSES: frozenset[str] = frozenset({RECOVERY_REDERIVE, RECOVERY_RETRY})

# One total map over the five basis reasons: a new reason cannot join the closed
# set without a recovery class, because the tests pin the map's keys to it.
BASIS_RECOVERY: Mapping[str, str] = MappingProxyType(
    {
        HOLD_VERSION_MOVED: RECOVERY_REDERIVE,
        HOLD_INPUT_VANISHED: RECOVERY_REDERIVE,
        BASIS_MEMBER_UNMANAGED: RECOVERY_REDERIVE,
        BASIS_MEMBER_REPLACED: RECOVERY_REDERIVE,
        HOLD_VERSION_UNCONFIRMED: RECOVERY_RETRY,
    }
)

# Why a commit's own compare-and-swap was refused while the basis was unchanged.
# These carry their token alone, with no recovery class, until the core recovery
# table assigns one; they stay out of BASIS_HOLD_REASONS because a basis move and
# a target refusal call for different recoveries.
BASIS_TARGET_REFUSAL_REASONS: frozenset[str] = frozenset(
    {
        OTHER_HOLDER_REASON,
        STALE_READ_GENERATION_REASON,
        OCC_CALLER_TRANSIENT_REASON,
        GIVER_FENCED_REASON,
        OCC_IS_SHARED_OR_INVALID_ONLY_REASON,
    }
)

NAME_KIND_FILE = "file"
NAME_KIND_KEY = "key"
NAME_KINDS: frozenset[str] = frozenset({NAME_KIND_FILE, NAME_KIND_KEY})

ANSWER_PROCEED = "proceed"
ANSWER_HOLD = "hold"
_ANSWERS: frozenset[str] = frozenset({ANSWER_PROCEED, ANSWER_HOLD})

_MAX_VERSION = 2**63 - 1  # the registry stores versions as signed 64-bit


def canonical_registration(value: object) -> str:
    """Return a registration in canonical dashed-UUID form, or raise ``ValueError``.

    Accepts a ``UUID`` or any string a ``UUID`` parses. Comparing registrations
    only in this form keeps the braced, undashed and dashed spellings of one id
    from reading as a replaced record.
    """
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        try:
            return str(UUID(value))
        except ValueError:
            pass
    raise ValueError(f"registration must be a UUID, got {value!r}")


def _check_version(value: object) -> int:
    # bool is an int subclass; True would otherwise pass as version 1.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"version must be an integer, got {value!r}")
    if not 1 <= value <= _MAX_VERSION:
        raise ValueError(f"version must be between 1 and {_MAX_VERSION}, got {value}")
    return value


@dataclass(frozen=True)
class BasisMember:
    """One input the caller read: its path, the confirmed version it read, the
    registration the read was made under, and what the path names.

    Validated on construction, so a member that could only hold (an unconfirmed
    version 0, a malformed registration) never exists. The registration is
    optional: a member built by hand may omit it and is then compared by version
    only. The fields are the serialized form; another process rebuilds the
    member with ``BasisMember(**fields)``.
    """

    path: str
    version: int
    registration: str | None = None
    name_kind: str = NAME_KIND_FILE

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError(f"path must be a non-empty string, got {self.path!r}")
        _check_version(self.version)
        if self.registration is not None:
            object.__setattr__(
                self, "registration", canonical_registration(self.registration)
            )
        if self.name_kind not in NAME_KINDS:
            raise ValueError(f"name_kind must be one of {sorted(NAME_KINDS)}")

    def to_wire(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "version": self.version,
            "registration": self.registration,
            "name_kind": self.name_kind,
        }

    @classmethod
    def from_wire(cls, fields: Mapping[str, Any]) -> BasisMember:
        return cls(**fields)


@dataclass(frozen=True)
class ResolvedRow:
    """What a lookup found for a member: the row's current version and the
    registration it carries (``None`` when the registry has none to give)."""

    version: int
    registration: str | None = None


def classify_member(
    *,
    recorded_version: int,
    recorded_registration: str | None,
    is_tracked: bool,
    row: ResolvedRow | None,
    name_held_by_other_record: bool = False,
) -> str | None:
    """Return the basis hold reason for one member, or ``None`` when unchanged.

    First matching row wins, and the order is load-bearing (a caller branches on
    the reason): unmanaged, then replaced, then vanished, then version moved.

    ``name_held_by_other_record`` is the in-process case where no record carries
    the member's registration but another live record carries its name: the
    thing read was deleted and the name now belongs to something else, which is
    ``replaced``, not ``vanished``. A lower current version also reads as
    replaced, because a row's version only rises during its life.
    """
    if not is_tracked:
        return BASIS_MEMBER_UNMANAGED
    if row is None:
        return BASIS_MEMBER_REPLACED if name_held_by_other_record else HOLD_INPUT_VANISHED
    if recorded_registration is not None and row.registration != recorded_registration:
        return BASIS_MEMBER_REPLACED
    if row.version < recorded_version:
        return BASIS_MEMBER_REPLACED
    if row.version != recorded_version:
        return HOLD_VERSION_MOVED
    return None


# ---------------------------------------------------------------------------
# answers: one wire form (a dict) and one parse each
# ---------------------------------------------------------------------------


def _int_or_none(value: object, *, what: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an integer or null, got {value!r}")
    return value


def _check_answer_name(answer: object) -> str:
    if answer not in _ANSWERS:
        raise ValueError(f"answer must be one of {sorted(_ANSWERS)}, got {answer!r}")
    return str(answer)


def _check_recovery(recovery: object) -> None:
    if recovery is not None and recovery not in RECOVERY_CLASSES:
        raise ValueError(f"unknown recovery class {recovery!r}")


@dataclass(frozen=True)
class CheckedMember:
    """A member that proceeded: the version it expected and the one it found."""

    path: str
    expected: int
    current: int


@dataclass(frozen=True)
class MovedMember:
    """A member that moved: expected and current version (``None`` when there is
    no row) and exactly one reason from the closed basis set."""

    path: str
    expected: int
    current: int | None
    reason: str

    def __post_init__(self) -> None:
        if self.reason not in BASIS_HOLD_REASONS:
            raise ValueError(f"unknown basis hold reason {self.reason!r}")

    def to_wire(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "expected": self.expected,
            "current": self.current,
            "reason": self.reason,
        }

    @classmethod
    def from_wire(cls, fields: Mapping[str, Any]) -> MovedMember:
        return cls(
            path=fields["path"],
            expected=fields["expected"],
            current=_int_or_none(fields["current"], what="current"),
            reason=fields["reason"],
        )


def _moved_to_wire(moved: tuple[MovedMember, ...]) -> list[dict[str, Any]]:
    return [member.to_wire() for member in moved]


@dataclass(frozen=True)
class BasisCheckAnswer:
    """The answer to a basis check.

    ``proceed`` carries every member's expected and current version. ``hold``
    carries every moved member with one reason each and the recovery class; a
    hold that names no member (a stalled registry, a degraded check) carries the
    ``reason`` alone, with its class.
    """

    answer: str
    checked: tuple[CheckedMember, ...] = ()
    moved: tuple[MovedMember, ...] = ()
    recovery: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _check_answer_name(self.answer)
        _check_recovery(self.recovery)
        if self.reason is not None and self.reason not in BASIS_HOLD_REASONS:
            raise ValueError(f"unknown basis hold reason {self.reason!r}")

    def to_wire(self) -> dict[str, Any]:
        if self.answer == ANSWER_PROCEED:
            return {
                "answer": ANSWER_PROCEED,
                "checked": {
                    m.path: {"expected": m.expected, "current": m.current}
                    for m in self.checked
                },
            }
        body: dict[str, Any] = {"answer": ANSWER_HOLD, "moved": _moved_to_wire(self.moved)}
        if self.reason is not None:
            body["reason"] = self.reason
        if self.recovery is not None:
            body["recovery"] = self.recovery
        return body

    @classmethod
    def from_wire(cls, body: Mapping[str, Any]) -> BasisCheckAnswer:
        answer = _check_answer_name(body.get("answer"))
        if answer == ANSWER_PROCEED:
            checked = tuple(
                CheckedMember(path, entry["expected"], entry["current"])
                for path, entry in body["checked"].items()
            )
            return cls(answer=answer, checked=checked)
        return cls(
            answer=answer,
            moved=tuple(MovedMember.from_wire(m) for m in body["moved"]),
            recovery=body.get("recovery"),
            reason=body.get("reason"),
        )


@dataclass(frozen=True)
class CommittedTarget:
    """The target a winning commit wrote, at the version it now has."""

    path: str
    version: int


@dataclass(frozen=True)
class TargetRefusal:
    """Why the target's own compare-and-swap was refused while the basis held.

    ``reason`` is one of :data:`BASIS_TARGET_REFUSAL_REASONS`. The generation
    pair is carried both or neither; ``successor`` and ``version_at_transfer``
    ride only a fenced giver's refusal.
    """

    path: str
    reason: str
    current_version: int | None
    captured_generation: int | None = None
    owner_generation: int | None = None
    successor: str | None = None
    version_at_transfer: int | None = None

    def __post_init__(self) -> None:
        if self.reason not in BASIS_TARGET_REFUSAL_REASONS:
            raise ValueError(f"unknown target refusal reason {self.reason!r}")
        if (self.captured_generation is None) != (self.owner_generation is None):
            raise ValueError("the generation pair is carried both or neither")

    def to_wire(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "path": self.path,
            "reason": self.reason,
            "current_version": self.current_version,
        }
        optional = {
            "captured_generation": self.captured_generation,
            "owner_generation": self.owner_generation,
            "successor": self.successor,
            "version_at_transfer": self.version_at_transfer,
        }
        body.update({key: value for key, value in optional.items() if value is not None})
        return body

    @classmethod
    def from_wire(cls, fields: Mapping[str, Any]) -> TargetRefusal:
        return cls(
            path=fields["path"],
            reason=fields["reason"],
            current_version=_int_or_none(fields["current_version"], what="current_version"),
            captured_generation=_int_or_none(
                fields.get("captured_generation"), what="captured_generation"
            ),
            owner_generation=_int_or_none(
                fields.get("owner_generation"), what="owner_generation"
            ),
            successor=fields.get("successor"),
            version_at_transfer=_int_or_none(
                fields.get("version_at_transfer"), what="version_at_transfer"
            ),
        )


@dataclass(frozen=True)
class BasisCommitAnswer:
    """The answer to a commit under a basis; basis moves and target refusals stay apart.

    - won: ``proceed`` with ``committed`` (and the top-level ``handoff`` key the
      commit routes attach, passed through untouched);
    - basis or target moved: ``hold`` with ``moved`` and ``rederive``;
    - basis unchanged, target refused: ``hold``, empty ``moved``, a ``target``
      block, no recovery class;
    - outcome unknown: ``hold`` with ``reason`` ``commit_unconfirmed``,
      ``degraded`` and ``held_by``, no recovery class;
    - nothing ran (registry deadline): ``hold`` with ``version_unconfirmed`` and
      ``retry``.
    """

    answer: str
    committed: CommittedTarget | None = None
    moved: tuple[MovedMember, ...] = ()
    recovery: str | None = None
    reason: str | None = None
    target: TargetRefusal | None = None
    degraded: bool = False
    held_by: str | None = None
    handoff: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _check_answer_name(self.answer)
        _check_recovery(self.recovery)
        allowed = BASIS_HOLD_REASONS | {COMMIT_UNCONFIRMED_REASON}
        if self.reason is not None and self.reason not in allowed:
            raise ValueError(f"unknown commit hold reason {self.reason!r}")

    def to_wire(self) -> dict[str, Any]:
        if self.answer == ANSWER_PROCEED:
            if self.committed is None:
                raise ValueError("a proceed answer carries the committed target")
            won: dict[str, Any] = {
                "answer": ANSWER_PROCEED,
                "committed": {"path": self.committed.path, "version": self.committed.version},
            }
            if self.handoff is not None:
                won["handoff"] = dict(self.handoff)
            return won
        body: dict[str, Any] = {"answer": ANSWER_HOLD, "moved": _moved_to_wire(self.moved)}
        if self.reason is not None:
            body["reason"] = self.reason
        if self.recovery is not None:
            body["recovery"] = self.recovery
        if self.target is not None:
            body["target"] = self.target.to_wire()
        if self.degraded:
            body["degraded"] = True
        if self.held_by is not None:
            body["held_by"] = self.held_by
        return body

    @classmethod
    def from_wire(cls, body: Mapping[str, Any]) -> BasisCommitAnswer:
        answer = _check_answer_name(body.get("answer"))
        if answer == ANSWER_PROCEED:
            won = body["committed"]
            return cls(
                answer=answer,
                committed=CommittedTarget(won["path"], won["version"]),
                handoff=body.get("handoff"),
            )
        target = body.get("target")
        return cls(
            answer=answer,
            moved=tuple(MovedMember.from_wire(m) for m in body["moved"]),
            recovery=body.get("recovery"),
            reason=body.get("reason"),
            target=None if target is None else TargetRefusal.from_wire(target),
            degraded=bool(body.get("degraded", False)),
            held_by=body.get("held_by"),
        )
