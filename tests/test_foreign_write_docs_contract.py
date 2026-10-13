# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The guide describes the coverage number the detector actually produces.

A foreign-write run's coverage count is the set git could report on at that
check, not every artifact the coordinator knows and the patterns track. Four
things follow from that, and all four are the kind of claim a reader acts on —
so each is pinned here rather than left to a manual read:

* the narrowed meaning of the number,
* the ranking qualifier, because the uncoverable note leads the report ONLY for
  a store that never recorded a check,
* the subset caveat, because a partial narrowing leaves no signal anywhere in
  the report,
* the remedy, because a file kept out of git on purpose needs no fixing and one
  you want watched needs `git add`, not a coordinator change.

Every search runs over WHITESPACE-NORMALIZED text. The guide wraps at about 79
columns, so every one of these sentences spans several lines, and a line-wise
grep for any of them would report clean while the sentence sat there mangled or
absent — a false green this repository has already been handed more than once.

``test_no_unqualified_coverage_claim`` is the other direction: the number must
never be described as a count of what the coordinator tracks irrespective of
git. It is a proximity check rather than a banned-phrase list, because the
phrasing that would reintroduce the old claim is not fixed — what matters is
that no description of the count survives without the git qualifier beside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest

_GUIDE = Path(__file__).resolve().parents[1] / "docs" / "guide.md"


def _normalized(text: str) -> str:
    """Collapse every run of whitespace, so a wrapped sentence still matches."""
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class Statement:
    """One claim the guide has to carry, and the text that carries it."""

    name: str
    phrase: str


# The four statements. Each phrase is one contiguous sentence fragment carrying
# the whole claim: a pin split across two sentences would still pass with half
# the claim deleted.
STATEMENTS: tuple[Statement, ...] = (
    Statement(
        name="the coverage number is narrowed to what git could report on",
        phrase="the widest scope that run ever watched, not the scope of its last check",
    ),
    Statement(
        name="the uncoverable note leads the report only for a store with no checks",
        phrase=(
            "only in a store that never recorded a check; where an earlier run did "
            "record checks, those runs lead instead and the note is read from "
            "`report.uncoverable`"
        ),
    ),
    Statement(
        name="the count is a subset, and a partial narrowing is not signalled",
        phrase=(
            "a subset of the files you asked to have tracked, and a check that could "
            "see only some of them leaves no separate mark anywhere in the report"
        ),
    ),
    Statement(
        name="the remedy: out-of-git files are by design, tracked ones need git",
        phrase=(
            "Files you deliberately keep out of git are outside the instrument by "
            "design and need no fixing; where you do want them watched, the tracked "
            "set has to name files git tracks"
        ),
    ),
)

# How the guide refers to the coverage number. Each occurrence has to sit beside
# the git qualifier; a cue standing alone is the old, unqualified claim.
_COVERAGE_CUES = ("covered_count", "files were in scope")

# The qualifier that makes such a description honest.
_GIT_QUALIFIER = re.compile(r"git (?:could|can)(?: actually)? report on")

# Normalized characters either side of a cue that count as "beside it". One
# guide bullet is longer than this, so the qualifier has to share the sentence
# or its neighbour rather than merely the section.
_PROXIMITY_CHARS = 400


@pytest.fixture(scope="module")
def guide() -> str:
    return _normalized(_GUIDE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("statement", STATEMENTS, ids=lambda s: s.name)
def test_guide_carries_statement(guide: str, statement: Statement) -> None:
    """Each of the four claims is present, wrapping notwithstanding."""
    assert statement.phrase in guide, (
        f"docs/guide.md no longer states: {statement.name}\n"
        f"missing text: {statement.phrase!r}"
    )


@pytest.mark.parametrize("cue", _COVERAGE_CUES)
def test_no_unqualified_coverage_claim(guide: str, cue: str) -> None:
    """The coverage number is never described without the git qualifier.

    Both halves matter. The cue must appear at all — a guide that stopped
    describing the number would otherwise pass this check by saying nothing —
    and every occurrence of it must have the qualifier within reach.
    """
    positions = [match.start() for match in re.finditer(re.escape(cue), guide)]
    assert positions, f"docs/guide.md no longer describes the coverage number ({cue!r})"
    for position in positions:
        window = guide[
            max(0, position - _PROXIMITY_CHARS) : position + len(cue) + _PROXIMITY_CHARS
        ]
        assert _GIT_QUALIFIER.search(window), (
            f"docs/guide.md describes {cue!r} with no note that the count is what "
            f"git could report on, near: ...{window[:200]}..."
        )
