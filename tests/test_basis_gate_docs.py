# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The guide's basis-check vocabulary is the one the code answers with.

A client branches on the reason a basis hold carries, on its recovery class,
and on the reason in a refused commit's target block. Each of those is a
closed set in the code, and each is published as a table in the guide's
"Basis check" section. These tests parse the tables out of the guide and
compare them with the code's sets, so a reason added on one side only fails
here: a client would otherwise branch on a word nothing documents, or on one
that never arrives.

The basis tables are their own. The effect fence's published vocabulary is a
different closed set, pinned by its own test in
``tests/test_claude_code_coordinator_server.py``; nothing here reads it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ccs.core.basis import BASIS_RECOVERY, BASIS_TARGET_REFUSAL_REASONS
from ccs.core.exceptions import BASIS_HOLD_REASONS

_GUIDE_PATH = Path(__file__).resolve().parents[1] / "docs" / "guide.md"

_SECTION_HEADING = "## Basis check"
_REASON_TABLE_HEADING = "### Hold reasons and recovery"
_TARGET_TABLE_HEADING = "### The target block"


def _normalized(text: str) -> str:
    """Collapse every run of whitespace, so a wrapped sentence still matches."""
    return " ".join(text.split())


def _basis_section(guide: str) -> str:
    """The guide's basis-check section, up to the next top-level heading."""
    assert f"\n{_SECTION_HEADING}\n" in guide, "the guide has no basis-check section"
    return guide.split(f"\n{_SECTION_HEADING}\n", 1)[1].split("\n## ", 1)[0]


def _first_table(section: str, heading: str) -> list[dict[str, str]]:
    """The first table under ``heading`` in ``section``, one dict per body
    row keyed by the header cells with their backticks stripped.

    A subsection that loses its heading or its table yields an empty list,
    which fails the comparisons below rather than passing vacuously.
    """
    assert f"\n{heading}\n" in section, f"the basis-check section has no {heading!r}"
    body = section.split(f"\n{heading}\n", 1)[1].split("\n### ", 1)[0]
    header: list[str] | None = None
    rows: list[dict[str, str]] = []
    for line in body.splitlines():
        if not line.startswith("|"):
            if rows:
                break
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if header is None:
            header = [cell.strip("`") for cell in cells]
        elif not line.startswith("|---"):
            rows.append(dict(zip(header, cells)))
    return rows


def _token(cell: str) -> str:
    """A cell that holds one wire token in backticks, as the token."""
    assert cell.startswith("`") and cell.endswith("`"), f"not a single token: {cell!r}"
    return cell.strip("`")


@pytest.fixture(scope="module")
def section() -> str:
    return _basis_section(_GUIDE_PATH.read_text(encoding="utf-8"))


def test_reason_table_is_exactly_the_closed_set(section: str) -> None:
    """The reasons the guide's table lists are ``BASIS_HOLD_REASONS``."""
    rows = _first_table(section, _REASON_TABLE_HEADING)
    documented = [_token(row["reason"]) for row in rows]
    assert len(documented) == len(set(documented)), f"a reason is listed twice: {documented}"
    assert set(documented) == BASIS_HOLD_REASONS, (
        "the guide's basis reason table and BASIS_HOLD_REASONS have drifted; "
        f"documented-only={sorted(set(documented) - BASIS_HOLD_REASONS)}, "
        f"published-only={sorted(BASIS_HOLD_REASONS - set(documented))}"
    )


def test_recovery_column_is_the_recovery_map(section: str) -> None:
    """Each reason's documented recovery class is the one ``BASIS_RECOVERY``
    assigns it: a client re-derives or retries on the guide's word."""
    rows = _first_table(section, _REASON_TABLE_HEADING)
    documented = {_token(row["reason"]): _token(row["recovery"]) for row in rows}
    assert documented == dict(BASIS_RECOVERY)


def test_target_table_is_exactly_the_refusal_set(section: str) -> None:
    """The target-block reasons the guide explains are
    ``BASIS_TARGET_REFUSAL_REASONS``, and none of them is a basis hold
    reason, which calls for a different recovery."""
    rows = _first_table(section, _TARGET_TABLE_HEADING)
    documented = [_token(row["target.reason"]) for row in rows]
    assert len(documented) == len(set(documented)), f"a reason is listed twice: {documented}"
    assert set(documented) == BASIS_TARGET_REFUSAL_REASONS, (
        "the guide's target-block table and BASIS_TARGET_REFUSAL_REASONS have drifted; "
        f"documented-only={sorted(set(documented) - BASIS_TARGET_REFUSAL_REASONS)}, "
        f"published-only={sorted(BASIS_TARGET_REFUSAL_REASONS - set(documented))}"
    )


# Each phrase is one contiguous fragment carrying a whole claim the section
# must keep: what the vocabulary excludes, and the limits a caller builds on.
_STATEMENTS: tuple[tuple[str, str], ...] = (
    (
        "input_vanished excludes untracked paths, unlike the fence",
        "**`input_vanished` here excludes untracked paths**: the fence folds an "
        "untracked path into it, while a basis check reports it as "
        "`basis_member_unmanaged`",
    ),
    (
        "the check is one instant, not an atomic act",
        "**One instant, not an atomic act.** A `proceed` says every member was current "
        "at the instant of the check.",
    ),
    (
        "versions only: a same-bytes re-save holds",
        "No content hash is compared, so a byte-identical re-save moves the version "
        "and holds.",
    ),
    (
        "single host, cooperative writers, the Python coordinator only",
        "**One host, cooperative writers, the Python coordinator only.**",
    ),
    (
        "an outcome-unknown commit is reconciled by reading the target",
        "re-read the target, and retry only if your commit is absent",
    ),
)


@pytest.mark.parametrize("claim,phrase", _STATEMENTS, ids=[name for name, _ in _STATEMENTS])
def test_section_states_the_contract(section: str, claim: str, phrase: str) -> None:
    assert _normalized(phrase) in _normalized(section), f"the guide no longer states: {claim}"
