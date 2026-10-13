# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Read-only open of a coordinator ``state.db``, shared by the offline readers.

:mod:`ccs.diagnose.conflict_counters` and :mod:`ccs.diagnose.foreign_writes`
are two reports over the same store, so the low-level open is the same fact
twice, not the sanctioned per-backend duplication the two registries carry.
Written out twice it drifts silently: a busy timeout re-derived from the
registry's budget, or a tightening of the missing-file translation, would land
in one report and leave the other answering on different terms.

What deliberately stays with each caller is the per-table
missing-versus-empty logic. The conflict reader maps an absent table and an
empty one to the same zero and documents that as zero recorded conflicts; the
foreign-write reader must keep the two apart, because a coverage claim depends
on knowing whether the detector ever ran. Those are different contracts over
the same connection, and merging them would cost one of them its honesty.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from urllib.parse import quote

__all__ = ["open_readonly_state_db"]

BUSY_TIMEOUT_MS = 1500
"""Matches the registry's own budget-derived value."""


def open_readonly_state_db(path: Path) -> sqlite3.Connection:
    """Open the coordinator store at ``path`` read-only, ready to query.

    Raises ``FileNotFoundError`` when nothing is there — a report against a
    store that does not exist is a caller error, never evidence of a quiet
    month — and ``ValueError`` for a path carrying a NUL, which no filesystem
    call would accept either. Every other failure to open propagates, so a
    broken read cannot be mistaken for an empty one. The caller owns the
    returned connection and must close it.
    """
    # A NUL cannot survive the round trip: quote() renders it %00 and SQLite
    # decodes it back into a C string that truncates there, which is the same
    # open-a-different-file bug the encoding below exists to prevent. Python
    # raises ValueError for a NUL in any real filesystem call, so raise it here
    # rather than let the URI layer turn it into a silent wrong-file read.
    if "\x00" in str(path):
        raise ValueError(f"NUL byte in coordinator database path: {path!r}")
    # mode=ro + uri=True: without the explicit uri flag sqlite3 treats the
    # string as a literal filename and can silently fall back to read-write.
    # quote() so the path is read as filename bytes and never as URI syntax:
    # a '#' would otherwise start a fragment that swallows the query, taking
    # mode=ro with it — SQLite then opens the truncated path read-WRITE,
    # creates it, and the reader reports a store it never read as empty.
    # safe="" because '/' left unescaped lets a leading '//' read as a URI
    # authority, which is the same wrong-file hazard; SQLite decodes %2F back
    # into a separator, so escaping it costs nothing. absolute() so a bare
    # ":memory:" cannot resolve to SQLite's in-memory store and answer with a
    # zero no database ever backed. surrogateescape so an undecodable
    # filesystem byte still reaches the open and fails as FileNotFoundError,
    # rather than escaping as UnicodeEncodeError past every documented type.
    # No pre-check stat: connect directly and translate the failure, so a
    # missing file cannot slip through a check-to-open race window.
    uri = quote(str(path.absolute()), safe="", errors="surrogateescape")
    try:
        conn = sqlite3.connect(f"file:{uri}?mode=ro", uri=True)
    except sqlite3.OperationalError as exc:
        if not path.exists():
            raise FileNotFoundError(f"no coordinator database at {path}") from exc
        raise
    try:
        # A transient writer lock must wait, never masquerade as an empty
        # result — whatever "empty" means to the report being built.
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    except BaseException:
        # The caller never received the handle, so it cannot close it.
        conn.close()
        raise
    return conn
