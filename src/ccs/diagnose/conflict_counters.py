# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Offline reader for the coordinator's conflict-outcome counters.

Guarantee-ladder U5 / KTD-6: the 30-day observation report aggregates typed
deny counts (``version_mismatch`` / ``other_holder`` / ``stale_read_generation``)
across coordinator restarts — so the reader must work against a CLOSED
``state.db``, without importing the coordinator (this is the interface layer;
raw sqlite only, read-only URI open, no wire or protocol_corpus surface).

Attribution is by agent identity only: no host identifier reaches the commit
path today, and a report built from these totals must say so rather than
invent one. A database that predates the instrumentation simply has no
``conflict_counters`` table; that reads as zero conflicts recorded — zero is a
reportable result, not an error. Any other read failure (a locked database,
a hot-WAL recovery failure, disk I/O) is never mapped to zero — it propagates
so the report cannot mistake a broken read for a quiet 30 days.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ._state_db import open_readonly_state_db

__all__ = ["read_conflict_totals"]


def read_conflict_totals(db_path: str | Path) -> dict[tuple[str, str, str], int]:
    """Return ``{(artifact_id_hex, agent_id_hex, reason): count}`` from a
    coordinator ``state.db``, opened read-only.

    Keys stay hex strings (the store's own representation) — this reader has
    no registry to resolve identities against. Missing file raises
    ``FileNotFoundError`` (a report against a nonexistent store is a caller
    error, not zero); a missing ``conflict_counters`` table (pre-instrumentation
    db) returns ``{}``. Every other ``sqlite3.OperationalError`` — a locked
    database, a hot-WAL recovery failure, disk I/O — is re-raised rather than
    swallowed, so it can never masquerade as zero conflicts.
    """
    conn = open_readonly_state_db(Path(db_path))
    try:
        rows = conn.execute(
            "SELECT artifact_id, agent_id, reason, count FROM conflict_counters"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return {}  # pre-instrumentation db: no table, zero recorded.
        raise
    finally:
        conn.close()
    return {
        (art_hex, agent_hex, reason): count
        for art_hex, agent_hex, reason, count in rows
    }
