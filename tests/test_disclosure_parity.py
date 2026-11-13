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
        doc_phrase="two versioners releasing one checkpoint both decrement its pin refcount",
    ),
    Pin(
        name="volume: the same requirement on the project's front page",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="Fleet requirement (hard, v1)",
        doc="README.md",
        doc_phrase="must declare the **same** `managed` globs",
    ),
    Pin(
        name="acquire-or-fail: an ordinary pre-edit still displaces an opted-in holder",
        source="src/ccs/coordinator/service.py",
        source_marker="an ordinary write still displaces an opted-in holder",
        doc="docs/guide.md",
        doc_phrase="an ordinary pre-edit still displaces an opted-in holder",
    ),
    Pin(
        name="acquire-or-fail: the same limit on the project's front page",
        source="src/ccs/coordinator/service.py",
        source_marker="an ordinary write still displaces an opted-in holder",
        doc="README.md",
        doc_phrase="an ordinary pre-edit still displaces an opted-in holder",
    ),
    Pin(
        name="refuse mode: a write outside managed still displaces",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="write outside ``managed`` sends no flag and still displaces",
        doc="docs/guide.md",
        doc_phrase="A write outside `managed` sends no flag and displaces",
    ),
    Pin(
        name="refuse mode: a write through a link under managed to a file outside it raises",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="through a link whose name is under ``managed`` to a file outside it",
        doc="docs/guide.md",
        doc_phrase="a link whose name is under `managed` to a file outside it",
    ),
    Pin(
        name="refuse mode: the same link check on the project's front page",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="through a link whose name is under ``managed`` to a file outside it",
        doc="README.md",
        doc_phrase="to a file outside it raises `LinkOutOfManaged`",
    ),
    Pin(
        name="refuse mode: the same link check in the security notes",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="through a link whose name is under ``managed`` to a file outside it",
        doc="docs/security.md",
        doc_phrase="raises `LinkOutOfManaged` and sends nothing",
    ),
    Pin(
        name="refuse mode: the link check sees the links as they stand at the call",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="decided against the links as they stand when",
        doc="docs/guide.md",
        doc_phrase="reads the links as they stand when `write()` is called",
    ),
    Pin(
        name="spellings: a disk the start probe cannot classify folds nothing",
        source="src/ccs/adapters/claude_code/fs_profile.py",
        source_marker="or the disk is cannot-tell and folds",
        doc="docs/guide.md",
        doc_phrase="A disk the start probe cannot classify folds nothing",
    ),
    Pin(
        name="spellings: the same in the security notes",
        source="src/ccs/adapters/claude_code/fs_profile.py",
        source_marker="or the disk is cannot-tell and folds",
        doc="docs/security.md",
        doc_phrase="a disk the start probe cannot classify folds nothing",
    ),
    Pin(
        name="spellings: a coordinator without its single-writer lock folds nothing",
        source="src/ccs/adapters/claude_code/coordinator_server.py",
        source_marker="without the single-writer lock names match exactly as given",
        doc="docs/guide.md",
        doc_phrase="without its single-writer lock folds nothing",
    ),
    Pin(
        name="spellings: one profile per start, so a mount that folds otherwise is not seen",
        source="src/ccs/adapters/claude_code/fs_profile.py",
        source_marker="The workspace disk's fold profile, found once per coordinator start.",
        doc="docs/guide.md",
        doc_phrase="classifies only the disk that holds the workspace",
    ),
    Pin(
        name="spellings: one profile per start, so per-directory folding is not seen",
        source="src/ccs/adapters/claude_code/fs_profile.py",
        source_marker="The workspace disk's fold profile, found once per coordinator start.",
        doc="docs/guide.md",
        doc_phrase="per-directory case folding",
    ),
    Pin(
        name="spellings: only measured disks are known; no CI job runs a folding Linux disk",
        source="src/ccs/adapters/claude_code/fs_profile.py",
        source_marker="Measured: case-insensitive APFS, case-sensitive APFS, and a disk that folds nothing.",
        doc="docs/guide.md",
        doc_phrase="no CI job runs on a Linux disk that folds names",
    ),
    Pin(
        name="spellings: a class too wide to enumerate, or with a reversed range, is not folded",
        source="src/ccs/adapters/claude_code/policy.py",
        source_marker="when the class is too wide to enumerate or holds a reversed range",
        doc="docs/guide.md",
        doc_phrase="written high to low** (`[z-a]`), meets the folded name as written",
    ),
    Pin(
        name="spellings: records whose file is missing at the start are not merged",
        source="src/ccs/coordinator/fold_merge.py",
        source_marker="a file that is absent right now confirms nothing",
        doc="docs/guide.md",
        doc_phrase="Records whose file is missing at the start",
    ),
    Pin(
        name="links: a tool call through a link out of the workspace is not coordinated",
        source="src/ccs/adapters/claude_code/policy.py",
        source_marker="whose target is outside the workspace. A Read, Edit or Write through it is not",
        doc="docs/guide.md",
        doc_phrase="a Read, Edit or Write through such a link is not coordinated",
    ),
    Pin(
        name="links: a Bash command's file arguments are matched as typed",
        source="src/ccs/cli/_coherence_client.py",
        source_marker="a Bash token is keyed as typed",
        doc="docs/guide.md",
        doc_phrase="file arguments are matched as typed",
    ),
    Pin(
        name="refuse mode: the release gives up every path",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="path this attempt holds, not only ``path``",
        doc="docs/guide.md",
        doc_phrase="`reacquire()` releases every path this attempt holds",
    ),
    Pin(
        name="refuse mode: other paths must be read again after the release",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="other tracked paths should be re-read before writing",
        doc="docs/guide.md",
        doc_phrase="So re-read every other path before you write it",
    ),
    Pin(
        name="refuse mode: a remote coordinator's same-port replacement goes unseen",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="a fixed-port replacement goes unseen",
        doc="docs/guide.md",
        doc_phrase="coordinator on a fixed port has no such check",
    ),
    Pin(
        name="refuse mode: a blind retry overwrites the holder's commit",
        source="src/ccs/core/exceptions.py",
        source_marker="overwrites what the holder committed",
        doc="docs/guide.md",
        doc_phrase="write, so a write built without reading lands over whatever the holder",
    ),
    Pin(
        name="node: a volume never starts the Node coordinator",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="The same for the Node coordinator, which no volume starts.",
        doc="docs/guide.md",
        doc_phrase="A volume never starts the Node coordinator",
    ),
    Pin(
        name="node: the same on the project's front page",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="The same for the Node coordinator, which no volume starts.",
        doc="README.md",
        doc_phrase="a volume never starts the Node coordinator",
    ),
    Pin(
        name="node: no snapshot sessions or handoffs, so their routes raise",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="The Node coordinator has no snapshot sessions or handoffs.",
        doc="docs/guide.md",
        doc_phrase="The Node coordinator has no snapshot sessions or handoffs",
    ),
    Pin(
        name="node: an opted-in write carries no principal where the claim was answered 404",
        source="src/ccs/adapters/coherent_volume.py",
        source_marker="404 (``unsupported``) shows that the coordinator issues none, as the Node",
        doc="docs/guide.md",
        doc_phrase="goes without one only when its claim was answered `404`",
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
