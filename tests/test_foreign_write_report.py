# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""U3 — the offline foreign-write report and its four distinguishable states.

The sibling conflict-counter reader deliberately maps an absent table and an
empty one to the same value, and documents that as "zero recorded conflicts".
This reader cannot: a coverage claim is gated on the detector having actually
observed a span, so reading a store where it never ran as a clean zero is the
one failure the instrument exists to prevent. Hence four states, and a
discriminator that keys on the observation ROW rather than on the table — all
three tables are created on every writer open, including one whose sweep thread
was never started.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.diagnose.foreign_writes import (
    COUNTS,
    INSTRUMENTED_ZERO,
    NOT_COVERABLE,
    NOT_INSTRUMENTED,
    read_foreign_write_report,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _drop_detection_tables(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE foreign_write_counters")
    conn.execute("DROP TABLE foreign_write_observations")
    conn.execute("DROP TABLE foreign_write_uncoverable")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# The four states (R8, Success Criterion 2)
# ---------------------------------------------------------------------------


def test_a_store_predating_the_instrument_reads_as_not_instrumented(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    _drop_detection_tables(db)
    report = read_foreign_write_report(db)
    assert report.state == NOT_INSTRUMENTED
    assert report.instrumented is False
    assert report.totals == {}
    assert report.runs == ()


def test_tables_present_but_never_ticked_reads_as_not_instrumented(tmp_path: Path) -> None:
    """The load-bearing case, and the one the sibling pattern cannot express.
    A coordinator started with the sweep disabled still opens the store as a
    writer, so all three tables exist and are empty. That is not a zero."""
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    report = read_foreign_write_report(db)
    assert report.state == NOT_INSTRUMENTED
    assert report.instrumented is False


def test_ticked_with_nothing_found_reads_as_a_real_zero(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(100.0)
    reg.record_detection_tick(105.0)
    reg.close()

    report = read_foreign_write_report(db)
    assert report.state == INSTRUMENTED_ZERO
    assert report.instrumented is True
    assert report.totals == {}
    assert len(report.runs) == 1
    assert report.runs[0].tick_count == 2


def test_detections_read_back_per_artifact_and_outcome(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    art = uuid4()
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(100.0)
    reg.record_foreign_write(art, "foreign", HASH_A)
    reg.record_foreign_write(art, "lag_suppressed", HASH_B)
    reg.close()

    report = read_foreign_write_report(db)
    assert report.state == COUNTS
    assert report.totals == {art.hex: {"foreign": 1, "lag_suppressed": 1}}


# ---------------------------------------------------------------------------
# Coverage over a span (R12)
# ---------------------------------------------------------------------------


def test_a_span_inside_one_run_is_covered(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    for tick in (100.0, 110.0, 120.0):
        reg.record_detection_tick(tick)
    reg.close()
    assert read_foreign_write_report(db).covers(105.0, 115.0) is True


def test_a_span_reaching_past_the_last_tick_is_not_covered(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(100.0)
    reg.record_detection_tick(120.0)
    reg.close()
    assert read_foreign_write_report(db).covers(110.0, 130.0) is False


def test_a_gap_between_two_runs_is_not_covered(tmp_path: Path) -> None:
    """The case a cumulative tick count cannot see: the coordinator was down
    for the middle of the session, yet the total looks healthy."""
    db = tmp_path / "state.db"
    first = SqliteArtifactRegistry(db)
    first.record_detection_tick(100.0)
    first.record_detection_tick(150.0)
    first.close()
    second = SqliteArtifactRegistry(db)
    second.record_detection_tick(400.0)
    second.record_detection_tick(450.0)
    second.close()

    report = read_foreign_write_report(db)
    assert report.covers(110.0, 140.0) is True
    assert report.covers(410.0, 440.0) is True
    assert report.covers(140.0, 410.0) is False


def test_an_uninstrumented_store_covers_nothing(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    assert read_foreign_write_report(db).covers(0.0, 1.0) is False


def test_a_backwards_span_is_a_caller_error(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(100.0)
    reg.close()
    with pytest.raises(ValueError):
        read_foreign_write_report(db).covers(200.0, 100.0)


# ---------------------------------------------------------------------------
# Failure modes never read as zero
# ---------------------------------------------------------------------------


def test_a_missing_file_raises_rather_than_reading_as_zero(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_foreign_write_report(tmp_path / "nope.db")


def test_a_corrupt_store_raises_rather_than_reading_as_zero(tmp_path: Path) -> None:
    """A broken read must never masquerade as a quiet month."""
    db = tmp_path / "state.db"
    db.write_bytes(b"this is not a sqlite database" * 64)
    with pytest.raises(sqlite3.DatabaseError):
        read_foreign_write_report(db)


def test_the_reader_does_not_modify_the_store(tmp_path: Path) -> None:
    """Raw read-only URI open, like the conflict-counter sibling: pointing the
    report at a store copied off a machine must not touch it. Asserted on the
    bytes rather than on sidecar files, which belong to the writer that made
    them — the WAL sidecar outlives the writer's close and says nothing about
    this reader."""
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(100.0)
    reg.close()
    before_bytes = db.read_bytes()
    before_mtime = db.stat().st_mtime_ns

    report = read_foreign_write_report(db)

    assert report.state == INSTRUMENTED_ZERO
    assert db.read_bytes() == before_bytes
    assert db.stat().st_mtime_ns == before_mtime


def test_distinct_counts_are_not_transposed_in_the_report(tmp_path: Path) -> None:
    """The write side binds columns through one map; so must the reader. Every
    other count case here uses equal values, which a positional drift survives."""
    db = tmp_path / "state.db"
    art = uuid4()
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(100.0, covered_count=1)
    reg.record_foreign_write(art, "foreign", f"{0:064d}")
    for index in (1, 2):
        reg.record_foreign_write(art, "mediated", f"{index:064d}")
    for index in (3, 4, 5):
        reg.record_foreign_write(art, "lag_suppressed", f"{index:064d}")
    reg.close()

    report = read_foreign_write_report(db)
    assert report.totals == {art.hex: {"foreign": 1, "mediated": 2, "lag_suppressed": 3}}


def test_the_report_says_how_much_each_run_watched(tmp_path: Path) -> None:
    """A run that watched nothing is not a clean run."""
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(100.0, covered_count=7)
    reg.record_detection_tick(105.0, covered_count=9)
    reg.close()

    runs = read_foreign_write_report(db).runs
    assert len(runs) == 1 and runs[0].covered_count == 9


# ---------------------------------------------------------------------------
# The fourth state: a workspace nothing could ever watch
# ---------------------------------------------------------------------------


def _write_uncoverable(db: Path, run_id: str, reason: str, at: float) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO foreign_write_uncoverable VALUES (?, ?, ?)", (run_id, reason, at)
    )
    conn.commit()
    conn.close()


def test_an_uncoverable_workspace_is_not_the_never_instrumented_answer(
    tmp_path: Path,
) -> None:
    """Both stores hold no ticks and no counts, and they are different news.

    ``not-instrumented`` accuses the instrument; this store's detector ran and
    correctly reported there is nothing here to watch.
    """
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    _write_uncoverable(db, "r1", "no-git-work-tree", 100.0)

    report = read_foreign_write_report(db)
    assert report.state == NOT_COVERABLE
    assert report.instrumented is False
    assert report.runs == ()
    assert [(u.run_id, u.reason, u.observed_at_unix) for u in report.uncoverable] == [
        ("r1", "no-git-work-tree", 100.0)
    ]


def test_an_uncoverable_note_never_answers_a_coverage_question(tmp_path: Path) -> None:
    """The whole reason it is not an observed run with a zero tick count."""
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    _write_uncoverable(db, "r1", "no-git-work-tree", 100.0)

    assert read_foreign_write_report(db).covers(100.0, 100.0) is False


def test_a_later_watched_run_outranks_an_earlier_uncoverable_note(
    tmp_path: Path,
) -> None:
    """A workspace that gains a repository must not stay branded by the note
    its earlier run left behind — the note explains an ABSENCE of ticks, so a
    store that has ticks does not need it. It is still reported, because
    hiding it would lose the reason the earlier window is empty."""
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_uncoverable("no-git-work-tree", 100.0)
    reg.close_detection_run()
    reg.record_detection_tick(200.0, covered_count=3)
    reg.close()

    report = read_foreign_write_report(db)
    assert report.state == INSTRUMENTED_ZERO
    assert report.instrumented is True
    assert len(report.uncoverable) == 1
    assert report.covers(200.0, 200.0) is True


def test_an_unknown_reason_token_still_reaches_the_operator(tmp_path: Path) -> None:
    """The reason is reported, never interpreted: a token written by a newer
    detector must not be silently dropped by an older report."""
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    _write_uncoverable(db, "r1", "some-future-reason", 100.0)

    report = read_foreign_write_report(db)
    assert report.state == NOT_COVERABLE
    assert [u.reason for u in report.uncoverable] == ["some-future-reason"]


def test_a_store_predating_the_uncoverable_table_still_reads(tmp_path: Path) -> None:
    """A missing table is news about the schema, not about the workspace —
    the same tolerance the sibling readers give."""
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE foreign_write_uncoverable")
    conn.commit()
    conn.close()

    report = read_foreign_write_report(db)
    assert report.state == NOT_INSTRUMENTED
    assert report.uncoverable == ()


def test_an_uncoverable_note_newer_than_the_ticks_does_not_outrank_them(
    tmp_path: Path,
) -> None:
    """The MIRROR of the ordering test above, and the one that makes this a
    decision rather than an omission.

    State ranks by how much each fact PROVES, symmetrically — not by recency.
    Only one direction was pinned before, so a recency rewrite passed the whole
    suite. This store's newest fact is "nothing here can be watched", and it
    still reports a real zero, because the ticks it holds really happened.
    """
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_tick(200.0, covered_count=4)
    reg.close_detection_run()
    reg.record_detection_uncoverable("no-git-work-tree", 300.0)
    reg.close()

    report = read_foreign_write_report(db)
    assert report.state == INSTRUMENTED_ZERO
    assert report.instrumented is True
    assert len(report.uncoverable) == 1  # never hidden, whatever the state says
    assert report.covers(200.0, 200.0) is True
    assert report.covers(200.0, 300.0) is False  # the blind window still shows


def test_report_state_never_contradicts_covers(tmp_path: Path) -> None:
    """Why recency is not merely the other defensible option — it is wrong here.

    Ranking by recency would report this store as ``not-coverable`` with
    ``instrumented`` False, while ``covers()`` — which walks ``runs`` alone and
    is the instrument for a window — still answers True over the very ticks
    being denied. The invariant worth keeping is the one that holds in all four
    states: a store that is not instrumented covers nothing.
    """
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    for tick in (100.0, 500.0, 1000.0):
        reg.record_detection_tick(tick, covered_count=500)
    reg.close_detection_run()
    reg.record_detection_uncoverable("no-git-work-tree", 1100.0)
    reg.close()

    report = read_foreign_write_report(db)
    assert report.covers(100.0, 1000.0) is True
    assert report.instrumented is True
    assert report.state == INSTRUMENTED_ZERO


def test_two_reason_tokens_coexist_in_one_store_with_no_reader_change(
    tmp_path: Path,
) -> None:
    """R7, from the reader's side. The detector grew a SECOND uncoverable
    condition — a workspace git can poll but whose registered artifacts it can
    speak about none of — and that cost the report no state, no column, no
    migration and no frozen set of tokens.

    The proof is a store holding both: each note reaches the operator under its
    own token and in its own right, because ``UncoverableRun.reason`` is
    reported rather than interpreted. A reader that classified the token would
    have had to be taught the second one, and until it was it would have shown
    the operator a note with the reason removed.
    """
    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_detection_uncoverable("no-git-work-tree", 100.0)
    reg.close_detection_run()
    reg.record_detection_uncoverable("no-git-visible-artifact", 200.0)
    reg.close()

    report = read_foreign_write_report(db)
    assert report.state == NOT_COVERABLE
    assert report.instrumented is False
    assert [(u.reason, u.observed_at_unix) for u in report.uncoverable] == [
        ("no-git-work-tree", 100.0),
        ("no-git-visible-artifact", 200.0),
    ]
    assert len({u.run_id for u in report.uncoverable}) == 2
