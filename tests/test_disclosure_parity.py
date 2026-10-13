# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Disclosure parity — a limitation admitted in a docstring is admitted to users.

Two shipped concurrency limitations were disclosed in source docstrings while the
user-facing docs described only the narrower case. A reader of the docs could not
learn what a reader of the source could.

This guards both directions, and it is deliberately two mechanisms rather than one:

* ``PINS`` ties each known disclosure to the user-doc sentence that carries it, so
  deleting either side fails here rather than silently widening the gap.
* ``test_every_disclosure_marker_is_pinned`` scans for the marker vocabulary the
  codebase uses for these admissions and fails on any occurrence that is not
  pinned. That is what catches the NEXT one — a new limitation cannot land
  disclosed only in source.

A generic detector is not possible: the two known disclosures share no phrasing.
The pin table is the honest form of the check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Vocabulary this codebase uses when a docstring admits a shipped limitation.
# An occurrence outside PINS is a disclosure the user docs may not carry.
_MARKERS = (
    "Disclosed single-controller assumption",
    "Fleet requirement (hard, v1)",
)


@dataclass(frozen=True)
class Pin:
    """One disclosure, on both sides of the boundary."""

    name: str
    source: str
    source_marker: str
    doc: str
    doc_phrase: str


PINS: tuple[Pin, ...] = (
    Pin(
        name="restore: concurrent DIFFERENT-checkpoint restores are not cross-serialized",
        source="src/ccs/adapters/workspace.py",
        source_marker="Disclosed single-controller assumption",
        doc="docs/guide.md",
        doc_phrase="two restores of **different** checkpoints that overlap on a member",
    ),
    Pin(
        name="volume: a mixed-globs fleet is unsupported and refused at attach",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="Fleet requirement (hard, v1)",
        doc="docs/guide.md",
        doc_phrase="must declare the same `managed` globs",
    ),
    Pin(
        name="release: idempotence covers repeat calls, not concurrent ones",
        source="src/ccs/adapters/workspace.py",
        source_marker="Idempotent means REPEAT calls, not concurrent ones",
        doc="docs/guide.md",
        doc_phrase="two processes releasing one checkpoint both decrement its pin refcount",
    ),
    Pin(
        name="volume: the same requirement on the project's front page",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="Fleet requirement (hard, v1)",
        doc="README.md",
        doc_phrase="must declare the **same** `managed` globs",
    ),
)


def _read(relative: str) -> str:
    path = _REPO_ROOT / relative
    assert path.is_file(), f"{relative} is missing"
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("pin", PINS, ids=lambda p: p.name)
def test_source_still_discloses_the_limitation(pin: Pin) -> None:
    """The docstring admission survives — deleting it must be a deliberate act."""
    assert pin.source_marker in _read(pin.source), (
        f"{pin.source} no longer contains {pin.source_marker!r}. If the limitation "
        f"is genuinely gone, remove its pin and the matching text in {pin.doc}."
    )


@pytest.mark.parametrize("pin", PINS, ids=lambda p: p.name)
def test_user_docs_carry_the_same_limitation(pin: Pin) -> None:
    """The user-facing side survives — a doc rewrite cannot quietly drop it."""
    assert pin.doc_phrase in _read(pin.doc), (
        f"{pin.doc} no longer discloses: {pin.name}. The source still admits it at "
        f"{pin.source} ({pin.source_marker!r}), so the docs would understate what ships."
    )


def test_every_disclosure_marker_is_pinned() -> None:
    """A NEW disclosed limitation cannot land in source alone.

    This is the half that catches the next one rather than the last two.
    """
    pinned = {(pin.source, pin.source_marker) for pin in PINS}
    unpinned: list[str] = []

    for path in sorted((_REPO_ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(_REPO_ROOT).as_posix()
        for marker in _MARKERS:
            if marker in text and (relative, marker) not in pinned:
                unpinned.append(f"{relative}: {marker!r}")

    assert not unpinned, (
        "Disclosed limitation(s) with no user-doc pin:\n  "
        + "\n  ".join(unpinned)
        + "\n\nAdd a Pin naming the user-doc sentence that carries it, or the docs "
        "will describe a narrower system than the one that ships."
    )
