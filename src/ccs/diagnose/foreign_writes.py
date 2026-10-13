# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Offline reader for the coordinator's foreign-write detection instrument.

Sibling of :mod:`ccs.diagnose.conflict_counters` and deliberately identical to
it on discipline — the report must work against a CLOSED ``state.db``, without
importing the coordinator (this is the interface layer; raw sqlite only,
read-only URI open, no wire surface) — and deliberately different on one thing:
the shape of "nothing to report".

The conflict reader maps an absent table and an empty one to the same value and
documents that as zero recorded conflicts. That is honest for a deny counter:
no table and no denials are the same news. It is NOT honest here, because a
consumer may only claim foreign writes are *detected* for a span the detector
actually observed, and a store where the detector never ran would otherwise read
as a clean month. So this reader answers with four states, and it keys them on
the observation ROW rather than on the table: all three detection tables are
created on every writer open, including a coordinator whose sweep thread was
never started, so their presence proves nothing.

The fourth state is the one an operator is most likely to hit and least likely
to expect. Detection can only watch a git work tree, and a coordinator rooted
outside one — a temp directory, an unpacked archive, a workspace nobody ran
``git init`` in — can never be polled. That is neither a zero nor an outage, and
collapsing it into either would be the same lie in a different direction:
``instrumented-zero`` would hand out a clean bill of health nothing earned, and
``not-instrumented`` would blame an instrument that worked correctly. It is its
own answer.

Attribution is by artifact only. Filesystem changes carry no writer identity,
so the report names what changed and never who changed it.

Every other read failure — a corrupt store, a locked database, disk I/O — is
raised rather than mapped to a state, so a broken read can never masquerade as a
quiet month.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ._state_db import open_readonly_state_db

__all__ = [
    "COUNTS",
    "INSTRUMENTED_ZERO",
    "NOT_COVERABLE",
    "NOT_INSTRUMENTED",
    "ForeignWriteReport",
    "ObservedRun",
    "UncoverableRun",
    "read_foreign_write_report",
]

NOT_INSTRUMENTED = "not-instrumented"
"""The detector never observed a tick against this store. Not a zero."""

NOT_COVERABLE = "not-coverable"
"""The detector ran and found nothing it could watch: the workspace is not
inside a git work tree, so the poll it depends on can never read anything.
Distinct from :data:`NOT_INSTRUMENTED`, which covers a detector that was never
started or that failed — this one worked, and correctly reported that there is
nothing here to observe."""

INSTRUMENTED_ZERO = "instrumented-zero"
"""The detector observed ticks and recorded no detections. A real zero."""

COUNTS = "counts"
"""The detector observed ticks and recorded detections."""


# Outcome name to counter column, in one place. The write side binds the same
# way; a positional SELECT here could drift from it and relabel counts silently,
# and every equal-count test would still pass.
_OUTCOME_COLUMNS: dict[str, str] = {
    "foreign": "foreign_count",
    "mediated": "mediated_count",
    "lag_suppressed": "lag_suppressed_count",
}


@dataclass(frozen=True)
class ObservedRun:
    """One coordinator run's observed detection interval."""

    run_id: str
    first_tick_unix: float
    last_tick_unix: float
    tick_count: int
    covered_count: int = 0
    """Artifacts in scope during this interval. A run that watched nothing is
    not a clean run, and a tick count alone cannot say which this was."""


@dataclass(frozen=True)
class UncoverableRun:
    """One coordinator run that had no watchable workspace.

    Not an :class:`ObservedRun` with a zero tick count: an observed run is an
    interval :meth:`ForeignWriteReport.covers` walks, and a run that polled
    nothing must never be able to answer a coverage question.
    """

    run_id: str
    reason: str
    """An opaque stable token written by the detector — ``no-git-work-tree``
    today. Reported rather than interpreted, so a token this release does not
    know still reaches the operator."""
    observed_at_unix: float


@dataclass(frozen=True)
class ForeignWriteReport:
    """What a closed store says about foreign writes, and about its own coverage.

    ``totals`` maps an artifact id (hex, the store's own representation — this
    reader has no registry to resolve identities against) to its per-outcome
    counts, omitting zero buckets. A count is one observation of a distinct
    on-disk content, not one per tick.
    """

    state: str
    runs: tuple[ObservedRun, ...]
    totals: dict[str, dict[str, int]]
    uncoverable: tuple[UncoverableRun, ...] = ()
    """Runs that found no git work tree to poll. Reported alongside ``runs``
    rather than instead of them: a store can hold both, when a workspace gained
    or lost its repository between coordinator runs."""

    @property
    def instrumented(self) -> bool:
        """Whether the detector observed any tick at all against this store.

        A not-coverable store is NOT instrumented by this measure, and that is
        the intended reading: the detector ran, but it observed no tick, so
        there is no span any claim can rest on.

        Not a liveness signal. True means a span exists somewhere in this store
        that a claim can rest on — not that the workspace is healthy NOW. Like
        ``state``, it cannot fall once it has risen, because nothing deletes a
        row. For "was THIS window watched", ask :meth:`covers`; for "did some
        run find nothing to watch", read ``uncoverable`` — whatever ``state``
        says."""
        return self.state in (INSTRUMENTED_ZERO, COUNTS)

    def covers(self, start_unix: float, end_unix: float) -> bool:
        """Whether observed ticks span ``[start_unix, end_unix]`` without a gap.

        This is the question a coverage claim actually asks, and the reason runs
        are recorded as intervals rather than as a total: a coordinator that was
        down for the middle of a session still presents a healthy tick count and
        a recent last tick, and only the intervals reveal the hole.

        A store the detector never ran against covers nothing.
        """
        if start_unix > end_unix:
            raise ValueError(
                f"span start {start_unix} is after its end {end_unix}"
            )
        cursor = start_unix
        for run in sorted(self.runs, key=lambda r: r.first_tick_unix):
            if run.first_tick_unix > cursor:
                return False  # a gap opens before this run picks up
            cursor = max(cursor, run.last_tick_unix)
            if cursor >= end_unix:
                return True
        return False


def _read_table(
    conn: sqlite3.Connection, sql: str
) -> list[tuple] | None:
    """Run ``sql``, returning ``None`` when the table simply does not exist.

    Only the missing-table case is absorbed, and only into ``None`` — never into
    an empty result, which is a different fact. Every other ``OperationalError``
    propagates."""
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return None
        raise


def read_foreign_write_report(db_path: str | Path) -> ForeignWriteReport:
    """Return the detection report held by a coordinator ``state.db``.

    Opened read-only, so the report can be run against a store copied off a
    machine after the fact without touching it. A missing file raises
    ``FileNotFoundError``: a report against a store that does not exist is a
    caller error, not evidence of zero foreign writes.
    """
    conn = open_readonly_state_db(Path(db_path))
    try:
        run_rows = _read_table(
            conn,
            "SELECT run_id, first_tick_unix, last_tick_unix, tick_count, "
            "covered_count FROM foreign_write_observations "
            "ORDER BY first_tick_unix",
        )
        outcomes = tuple(_OUTCOME_COLUMNS)
        selected = ", ".join(_OUTCOME_COLUMNS[name] for name in outcomes)
        counter_rows = _read_table(
            conn,
            f"SELECT artifact_id, {selected} FROM foreign_write_counters",
        )
        uncoverable_rows = _read_table(
            conn,
            "SELECT run_id, reason, observed_at_unix FROM "
            "foreign_write_uncoverable ORDER BY observed_at_unix",
        )
    finally:
        conn.close()

    runs = tuple(
        ObservedRun(
            run_id=run_id,
            first_tick_unix=float(first),
            last_tick_unix=float(last),
            tick_count=int(count),
            covered_count=int(covered),
        )
        for run_id, first, last, count, covered in (run_rows or ())
    )
    totals: dict[str, dict[str, int]] = {}
    for art_hex, *values in counter_rows or ():
        counts = {name: value for name, value in zip(outcomes, values) if value}
        if counts:
            totals[art_hex] = counts
    uncoverable = tuple(
        UncoverableRun(
            run_id=run_id, reason=reason, observed_at_unix=float(observed_at)
        )
        for run_id, reason, observed_at in (uncoverable_rows or ())
    )

    # Ordered by how much each fact PROVES, strongest first — and symmetric by
    # design, which is the half an earlier version of this comment left unsaid.
    #
    # Counts decide before anything else, and not merely because they are
    # "stronger": ``foreign_write_counters`` carries no run id and no timestamp,
    # so a detection is not a coverage claim and cannot be time-ordered at all.
    # An observed run then means a real zero. Only with NEITHER does the
    # uncoverable note decide — it explains an ABSENCE of ticks, so a store that
    # has ticks does not need it. And no row of any kind is the not-instrumented
    # signal, which is why the ROW and not the table carries this meaning.
    #
    # Deliberately NOT ranked by recency, in either direction. This store is an
    # append-only log across every coordinator lifetime that ever opened it
    # (``run_id`` is a fresh uuid per open, nothing is deleted), so an offline
    # reader has no "now" and "the newest fact" is a liveness notion ``state``
    # cannot carry. Ranking by recency would put ``state`` in direct
    # contradiction with ``covers()``: a store with real ticks and a newer note
    # would headline `not-coverable` while ``covers()`` still returned True over
    # those very ticks. ``covers(start, end)`` is the instrument for a window,
    # and it walks ``runs`` alone — untouched by any ranking here.
    if totals:
        state = COUNTS
    elif runs:
        state = INSTRUMENTED_ZERO
    elif uncoverable:
        state = NOT_COVERABLE
    else:
        state = NOT_INSTRUMENTED
    return ForeignWriteReport(
        state=state, runs=runs, totals=totals, uncoverable=uncoverable
    )
