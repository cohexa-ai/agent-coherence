# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""RED: the handoff as people actually run it — plain file I/O, no coordination.

Session A is mid-task and posts its status to the shared notes file. Teammate
B picks up: reads the notes and appends its pickup line. A — still working,
and never re-reading — writes its updated status from the notes it read
earlier. B's line is gone, and nothing complained. That is the lost update
over a shared scratch file, here across two REAL OS processes; ``fixed.py``
runs the same sequence through ``CoherentVolume`` and shows it denied and
recovered.

This module also hosts the small story vocabulary the other acts reuse
(``EXPECTED``, ``new_workspace``, ``render_notes``, ``verdict_line``) so every
act renders its trace the same way.

Deterministic and offline — sequenced, not raced.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from examples.session_handoff import A_STATUS_1, A_STATUS_2, B_PICKUP, NOTES, UNGUARDED
from examples.session_handoff.sessions import Session

#: What the notes file holds when BOTH sessions' lines survive.
EXPECTED = A_STATUS_2 + B_PICKUP


def new_workspace(act: str) -> Path:
    """Fresh temp workspace for one act, with the notes directory in place."""
    workspace = Path(tempfile.mkdtemp(prefix=f"session_handoff_{act}_"))
    (workspace / NOTES).parent.mkdir(parents=True, exist_ok=True)
    return workspace


def render_notes(data: bytes) -> str:
    """One-line rendering of notes-file bytes for the trace (lines joined by ' | ')."""
    return " | ".join(data.decode().splitlines()) or "<empty>"


def verdict_line(final: bytes) -> str:
    """Closing trace line: what the notes file holds, and whether B's line made it."""
    if final == EXPECTED:
        outcome = "both lines survive"
    elif B_PICKUP not in final:
        outcome = "B's pickup line is GONE"
    else:
        outcome = "unexpected content"
    return f"notes.md    {render_notes(final)}   ({outcome})"


def run_broken() -> dict[str, object]:
    """Sequenced raw read→write with no coordination; A's second write erases B's line.

    Returns a structured trace so the runner and tests can assert the lost update
    without parsing prose.
    """
    workspace = new_workspace("red")
    trace: list[str] = []
    a: Session | None = None
    try:
        # A Session always carries a volume; this act never touches it. Leaving NOTES
        # outside the strict glob keeps the arm honest: nothing is coordinating it.
        a = Session(workspace, "A", UNGUARDED)
        b = Session(workspace, "B", UNGUARDED)
        try:
            a.raw_write(A_STATUS_1)
            trace.append(f"A raw_write {render_notes(A_STATUS_1)}")
            seen = b.raw_read()
            trace.append(f"B raw_read  -> {render_notes(seen)}")
            b.raw_write(seen + B_PICKUP)
            trace.append(f"B raw_write {render_notes(seen + B_PICKUP)}")
            # A is still working and never re-read: its update is built from the
            # notes as they were BEFORE B's pickup, so B's line is not in it.
            a.raw_write(A_STATUS_2)
            trace.append(f"A raw_write {render_notes(A_STATUS_2)}   (from its earlier read; never re-read)")
            final = (workspace / NOTES).read_bytes()
            trace.append(verdict_line(final))
        finally:
            b.close()
    finally:
        # The workspace is removed even when spawning A itself failed, and even
        # when close() itself escapes -- an interrupt landing in teardown must
        # not cost the caller its temp directory.
        try:
            if a is not None:
                a.close()
        finally:
            shutil.rmtree(workspace, ignore_errors=True)
    return {
        "act": "red",
        "final": final,
        "expected": EXPECTED,
        "b_line_present": B_PICKUP in final,
        "lost": final != EXPECTED,
        # Plain file I/O has no deny surface, so nothing can complain here; the
        # key exists so all three acts share one result vocabulary.
        "raised": None,
        "trace": trace,
    }
