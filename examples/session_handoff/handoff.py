# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""HANDOFF: A hands work to B through a checkpoint, and the rewind has two outcomes.

Both acts open the same way. A posts its status to the shared notes file,
checkpoints it (``handoff-1`` / ``handoff-2``), and keeps working: one more
status write. B picks the checkpoint up and restores it, rewinding the notes to
the handoff point. What happens next depends on one thing — whether A has
stopped.

HANDOFF-a (``run_handoff_stopped``): A has stopped writing after that last
write (its process stays open — it hosts the coordinator B is attached to). B's
restore lands cleanly (``restored``, one attempt) and the disk holds the handoff
bytes again. A finds out only when it next writes: the restore went through the
versioner's own ledger, never through A's live coordinator view, so A's next
``write`` re-hashes the disk against the hash it last observed, sees a foreign
change, and is DENIED (``StaleView``). A ``reacquire``s and reads the handoff
bytes. Restoring file members while a live session runs bypasses its grants —
the session learns on its next access.

HANDOFF-b (``run_handoff_racing``): A is STILL WORKING while B restores. Every
read the restore takes is followed by one more edit from A, so the restore's
version-CAS is beaten each time (the disk no longer matches what the ledger just
observed), re-drives from a fresh read, and is beaten again — until the bounded
re-drive budget exhausts into ``conflict``. A's latest edit survives on disk,
nothing was clobbered, and B is told. The surprising half: a restore is not an
overwrite.

Sequenced, not raced: the racing source stands in for A's continued edits
deterministically, one edit per restore read. Each session is a real OS process
(see ``sessions.py``): the first spawns the coordinator, the second attaches.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from ccs.adapters.workspace import MAX_RESTORE_LEG_REDRIVES
from examples.session_handoff import A_STATUS_1, A_STATUS_2, GUARDED, NOTES
from examples.session_handoff.broken import new_workspace, render_notes
from examples.session_handoff.sessions import Session, SessionDenied

#: What A tries to write after the rewind: its own last status plus a fresh note.
A_STATUS_3 = A_STATUS_2 + b"note: added indexes\n"

#: Leg iterations a contended restore is allowed: the first attempt plus every
#: re-drive. Exhaustion is the ``conflict`` outcome — never a raise, never a
#: livelock — so this is also the racing act's exact ``attempts`` value.
RACING_ATTEMPTS = MAX_RESTORE_LEG_REDRIVES + 1

_Story = Callable[[Path, Session, Session], dict[str, object]]


def run_handoff_stopped() -> dict[str, object]:
    """HANDOFF-a: A stops writing after the handoff; B's restore rewinds, A learns on its next write."""
    return _with_two_sessions("handoff_stopped", _stopped_story)


def run_handoff_racing() -> dict[str, object]:
    """HANDOFF-b: A is still working during B's restore; the restore concludes ``conflict``."""
    return _with_two_sessions("handoff_racing", _racing_story)


def _with_two_sessions(act: str, story: _Story) -> dict[str, object]:
    """Fresh workspace, A then B, the story, then B then A closed (reverse creation order)."""
    workspace = new_workspace(act)
    a: Session | None = None
    try:
        a = Session(workspace, "A", GUARDED)  # first session: spawns the coordinator
        b = Session(workspace, "B", GUARDED)  # attaches to A's coordinator
        try:
            return story(workspace, a, b)
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


def _stopped_story(workspace: Path, a: Session, b: Session) -> dict[str, object]:
    trace: list[str] = []
    checkpoint = _post_and_checkpoint(a, "handoff-1", trace)
    checkpoint_id = str(checkpoint["checkpoint_id"])
    a.write(A_STATUS_2)
    trace.append(f"A write     {render_notes(A_STATUS_2)}   (one more status, then A stops writing)")
    members = _pick_up(b, checkpoint_id, trace)
    leg = _single_leg(b.restore(checkpoint_id), trace)
    disk_after_restore = _notes_on_disk(workspace, trace)
    denial = _write_after_rewind(a, trace)
    reacquired = _reacquire(a, trace)
    return {
        "act": "handoff_stopped",
        **_restore_fields(checkpoint_id, members, leg),
        "disk_after_restore": disk_after_restore,
        "a_denied": denial is not None,
        "a_denial_exc": denial.exc_name if denial else None,
        "a_denial_message": denial.message if denial else None,
        "a_reacquired": reacquired,
        "trace": trace,
    }


def _racing_story(workspace: Path, a: Session, b: Session) -> dict[str, object]:
    trace: list[str] = []
    checkpoint = _post_and_checkpoint(a, "handoff-2", trace)
    checkpoint_id = str(checkpoint["checkpoint_id"])
    a.write(A_STATUS_2)
    trace.append(f"A write     {render_notes(A_STATUS_2)}   (A is still working)")
    # ``racing=True`` stands in for A continuing to edit while B restores: every
    # read the restore takes is followed by one more edit on disk.
    leg = _single_leg(b.restore(checkpoint_id, racing=True), trace)
    disk_after_restore = _notes_on_disk(workspace, trace)
    return {
        "act": "handoff_racing",
        **_restore_fields(checkpoint_id, list(checkpoint["members"]), leg),
        "disk_after_restore": disk_after_restore,
        "restore_landed": disk_after_restore == A_STATUS_1,
        "trace": trace,
    }


def _post_and_checkpoint(a: Session, name: str, trace: list[str]) -> dict:
    """A posts its status and checkpoints the notes under ``name``."""
    a.write(A_STATUS_1)
    trace.append(f"A write     {render_notes(A_STATUS_1)}")
    checkpoint = a.checkpoint(name)
    # Only the NAME enters the trace: the checkpoint id is a uuid, fresh every run.
    trace.append(f"A checkpoint {name}")
    return checkpoint


def _pick_up(b: Session, checkpoint_id: str, trace: list[str]) -> list[dict]:
    """B lists what A left behind: one trace line per checkpoint member."""
    members = b.members(checkpoint_id)
    for row in members:
        trace.append(f"B members   {row['member_path']} ({row['restore_tier']}, {row['pin_state']})")
    return members


def _single_leg(report: dict, trace: list[str]) -> dict:
    """The restore report's one member leg, traced as B's restore line."""
    (leg,) = report["members"]
    # ``detail`` stays out of the trace: it is engine vocabulary, not the story.
    trace.append(f"B restore   -> {leg['outcome']} (attempts={leg['attempts']})")
    return leg


def _notes_on_disk(workspace: Path, trace: list[str]) -> bytes:
    """What the notes file holds after B's restore, with the verdict the bytes earn."""
    data = (workspace / NOTES).read_bytes()
    if data == A_STATUS_1:
        verdict = "rewound to the handoff point"
    else:
        verdict = "NOT rewound; A's latest edit survives"
    trace.append(f"notes.md    {render_notes(data)}   ({verdict})")
    return data


def _write_after_rewind(a: Session, trace: list[str]) -> SessionDenied | None:
    """A's next write, built from its own last write — stale against the restored disk."""
    try:
        a.write(A_STATUS_3)
    except SessionDenied as denied:
        # Name only in the trace, as GREEN does; the reason travels in ``a_denial_message``.
        trace.append(f"A write     {render_notes(A_STATUS_3)}   -> DENIED {denied.exc_name}")
        return denied
    trace.append(f"A write     {render_notes(A_STATUS_3)}   (landed over the restored bytes; nothing raised)")
    return None


def _reacquire(a: Session, trace: list[str]) -> bytes:
    """After the deny: a mandatory fresh read, which is how A learns about the rewind."""
    fresh = a.reacquire()
    trace.append(f"A reacquire -> {render_notes(fresh)}")
    return fresh


def _restore_fields(checkpoint_id: str, members: list[dict], leg: dict) -> dict[str, object]:
    """The result keys both acts share: what was checkpointed and how the one leg concluded."""
    return {
        "checkpoint_id": checkpoint_id,
        "members": members,
        "outcome": leg["outcome"],
        "attempts": leg["attempts"],
        "detail": leg["detail"],
    }
