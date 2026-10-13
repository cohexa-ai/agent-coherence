# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""U2 — foreign-write detection counters and the per-run liveness row.

The detector counts each newly observed on-disk content once per artifact as
``foreign`` / ``mediated`` / ``lag_suppressed`` (R5, R7, R13), edge-gated on the
disk hash it last counted so one unreconciled divergence cannot accumulate a
count every tick (R15, KTD14). A separate per-run observation row records that
the detector actually ran (R8, R12, KTD6): mirroring the conflict counters alone
cannot distinguish "ran and saw nothing" from "never ran", because all three tables
are created on every writer open whether or not the sweep thread exists.

Nothing here enforces: these are counts, and a count denies nothing (R9).
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import FOREIGN_WRITE_OUTCOMES
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry

HASH_A = "a" * 64
HASH_B = "b" * 64


@pytest.fixture(params=["memory", "sqlite"])
def registry(request, tmp_path: Path):
    if request.param == "memory":
        yield ArtifactRegistry()
    else:
        reg = SqliteArtifactRegistry(tmp_path / "state.db")
        yield reg
        reg.close()


# ---------------------------------------------------------------------------
# Outcome vocabulary (R5 — three classifications and nothing finer)
# ---------------------------------------------------------------------------


def test_outcome_vocabulary_is_exactly_three() -> None:
    assert FOREIGN_WRITE_OUTCOMES == ("foreign", "mediated", "lag_suppressed")


def test_an_unknown_outcome_is_refused(registry) -> None:
    """R5 fixes the classification at three values; a fourth is a caller bug,
    not a silently-created bucket."""
    with pytest.raises(ValueError):
        registry.record_foreign_write(uuid4(), "probably_foreign", HASH_A)


# ---------------------------------------------------------------------------
# Edge-gated counting (R15 / KTD14)
# ---------------------------------------------------------------------------


def test_first_observation_of_a_content_counts(registry) -> None:
    art = uuid4()
    assert registry.record_foreign_write(art, "foreign", HASH_A) is True
    assert registry.foreign_write_totals() == {art: {"foreign": 1}}


def test_repeating_the_same_disk_hash_never_recounts(registry) -> None:
    """The whole point of KTD14: git reports a dirty file on every tick until
    it is committed, so a level-triggered count would record one edit forever."""
    art = uuid4()
    registry.record_foreign_write(art, "foreign", HASH_A)
    for _ in range(10):
        assert registry.record_foreign_write(art, "foreign", HASH_A) is False
    assert registry.foreign_write_totals() == {art: {"foreign": 1}}


def test_a_second_distinct_content_counts_again(registry) -> None:
    art = uuid4()
    registry.record_foreign_write(art, "foreign", HASH_A)
    assert registry.record_foreign_write(art, "foreign", HASH_B) is True
    assert registry.foreign_write_totals() == {art: {"foreign": 2}}


def test_the_edge_gate_is_per_artifact_not_global(registry) -> None:
    first, second = uuid4(), uuid4()
    assert registry.record_foreign_write(first, "foreign", HASH_A) is True
    assert registry.record_foreign_write(second, "foreign", HASH_A) is True


def test_a_new_content_under_a_different_outcome_still_counts(registry) -> None:
    """The gate keys on content, not on outcome: an artifact that diverges,
    is re-mediated, then diverges again records all three observations."""
    art = uuid4()
    registry.record_foreign_write(art, "foreign", HASH_A)
    registry.record_foreign_write(art, "mediated", HASH_B)
    registry.record_foreign_write(art, "foreign", HASH_A)
    assert registry.foreign_write_totals() == {art: {"foreign": 2, "mediated": 1}}


# ---------------------------------------------------------------------------
# Three independent buckets (R7, R13 — a suppression is never folded in)
# ---------------------------------------------------------------------------


def test_three_outcome_buckets_are_counted_independently(registry) -> None:
    art = uuid4()
    registry.record_foreign_write(art, "foreign", HASH_A)
    registry.record_foreign_write(art, "mediated", HASH_B)
    registry.record_foreign_write(art, "lag_suppressed", "c" * 64)
    assert registry.foreign_write_totals() == {
        art: {"foreign": 1, "mediated": 1, "lag_suppressed": 1}
    }


def test_zero_detections_is_an_empty_mapping_not_an_error(registry) -> None:
    assert registry.foreign_write_totals() == {}


# ---------------------------------------------------------------------------
# Per-run observation rows (R8, R12 / KTD6)
# ---------------------------------------------------------------------------


def test_a_fresh_store_has_no_observation_row(registry) -> None:
    """Zero ticks observed is what "never ran" looks like from inside."""
    assert registry.detection_runs() == []


def test_a_tick_opens_this_run_and_later_ticks_extend_it(registry) -> None:
    registry.record_detection_tick(100.0)
    registry.record_detection_tick(105.0)
    registry.record_detection_tick(110.0)
    runs = registry.detection_runs()
    assert len(runs) == 1
    run = runs[0]
    assert (run.first_tick_unix, run.last_tick_unix, run.tick_count) == (100.0, 110.0, 3)


def test_two_runs_against_one_store_produce_two_rows(tmp_path: Path) -> None:
    """R12 asks whether ticks covered a span. A cumulative count cannot answer
    that across a restart, so each run gets its own interval."""
    db = tmp_path / "state.db"
    first = SqliteArtifactRegistry(db)
    first.record_detection_tick(100.0)
    first.record_detection_tick(105.0)
    first.close()

    second = SqliteArtifactRegistry(db)
    second.record_detection_tick(500.0)
    runs = sorted(second.detection_runs(), key=lambda r: r.first_tick_unix)
    second.close()

    assert len(runs) == 2
    assert (runs[0].first_tick_unix, runs[0].last_tick_unix, runs[0].tick_count) == (
        100.0,
        105.0,
        2,
    )
    assert (runs[1].first_tick_unix, runs[1].last_tick_unix, runs[1].tick_count) == (
        500.0,
        500.0,
        1,
    )
    assert runs[0].run_id != runs[1].run_id


# ---------------------------------------------------------------------------
# Durability and the read-only handle (sqlite only)
# ---------------------------------------------------------------------------


def test_counts_and_runs_survive_close_and_reopen(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    art = uuid4()
    reg = SqliteArtifactRegistry(db)
    reg.record_foreign_write(art, "foreign", HASH_A)
    reg.record_detection_tick(100.0)
    reg.close()

    reopened = SqliteArtifactRegistry(db)
    assert reopened.foreign_write_totals() == {art: {"foreign": 1}}
    assert len(reopened.detection_runs()) == 1
    reopened.close()


def test_reopening_does_not_recount_an_unchanged_divergence(tmp_path: Path) -> None:
    """The edge state is durable, so a restart does not re-observe every
    still-dirty artifact as new."""
    db = tmp_path / "state.db"
    art = uuid4()
    reg = SqliteArtifactRegistry(db)
    reg.record_foreign_write(art, "foreign", HASH_A)
    reg.close()

    reopened = SqliteArtifactRegistry(db)
    assert reopened.record_foreign_write(art, "foreign", HASH_A) is False
    assert reopened.foreign_write_totals() == {art: {"foreign": 1}}
    reopened.close()


def _table_names(db: Path) -> set[str]:
    import sqlite3

    conn = sqlite3.connect(db)
    try:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def test_a_writer_open_creates_every_table(tmp_path: Path) -> None:
    """Which is why their PRESENCE can never mean the detector ran."""
    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    assert {
        "foreign_write_counters",
        "foreign_write_observations",
        "foreign_write_uncoverable",
    } <= _table_names(db)


def test_a_read_only_open_creates_neither_table(tmp_path: Path) -> None:
    """Mirrors the conflict-counter contract: the write-free read-only open
    never runs the ensure, and its readers tolerate the absence."""
    import sqlite3

    db = tmp_path / "state.db"
    # A store at the current schema that predates the instrument: created by a
    # writer, then stripped, so the read-only open has a schema to accept but
    # no detection tables to find.
    SqliteArtifactRegistry(db).close()
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE foreign_write_counters")
    conn.execute("DROP TABLE foreign_write_observations")
    conn.execute("DROP TABLE foreign_write_uncoverable")
    conn.commit()
    conn.close()

    reg = SqliteArtifactRegistry(db, read_only=True)
    reg.close()

    tables = _table_names(db)
    assert "foreign_write_counters" not in tables
    assert "foreign_write_observations" not in tables
    assert "foreign_write_uncoverable" not in tables


def test_read_only_handle_tolerates_a_pre_instrumentation_store(tmp_path: Path) -> None:
    """A store written before this instrument shipped is at the current schema
    (the detection tables are outside the versioned migration chain) but has
    neither table. A read-only handle reports empty rather than raising —
    matching the conflict counters. Built by dropping the tables a writer open
    creates, since the read-only open refuses a schema-less file outright."""
    import sqlite3

    db = tmp_path / "state.db"
    SqliteArtifactRegistry(db).close()
    conn = sqlite3.connect(db)
    conn.execute("DROP TABLE foreign_write_counters")
    conn.execute("DROP TABLE foreign_write_observations")
    conn.execute("DROP TABLE foreign_write_uncoverable")
    conn.commit()
    conn.close()

    reg = SqliteArtifactRegistry(db, read_only=True)
    assert reg.foreign_write_totals() == {}
    assert reg.detection_runs() == []
    assert reg.detection_uncoverable() == []
    reg.close()


# ---------------------------------------------------------------------------
# Backend parity
# ---------------------------------------------------------------------------


def test_both_backends_expose_the_same_detection_surface() -> None:
    for name in (
        "record_foreign_write",
        "record_detection_tick",
        "foreign_write_totals",
        "detection_runs",
        "record_detection_uncoverable",
        "detection_uncoverable",
    ):
        assert callable(getattr(ArtifactRegistry(), name)), name
        assert hasattr(SqliteArtifactRegistry, name), name


def test_detection_writes_touch_no_artifact_row(tmp_path: Path) -> None:
    """R9 — the detector mutates nothing outside its own detection tables. A count
    against an unregistered artifact id must not seed one."""
    import sqlite3

    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    reg.record_foreign_write(UUID(int=1), "foreign", HASH_A)
    reg.record_detection_tick(100.0)
    reg.close()

    conn = sqlite3.connect(db)
    assert conn.execute("SELECT count(*) FROM artifacts").fetchone()[0] == 0
    conn.close()


def test_distinct_counts_are_not_transposed_between_outcomes(registry) -> None:
    """Three different values, so a mislabelling is observable at all.

    Every other count test here uses equal values across the buckets, which
    means a read side that drifted from the outcome-to-column map — hardcoding
    a column order, say — would still satisfy them. It does not satisfy this
    one. Swapping two entries in the map itself is deliberately NOT caught,
    and cannot be: both sides derive from that one map, so a swap is a
    consistent relabelling rather than a transposition."""
    art = uuid4()
    for index in range(1):
        registry.record_foreign_write(art, "foreign", f"{index:064d}")
    for index in range(1, 3):
        registry.record_foreign_write(art, "mediated", f"{index:064d}")
    for index in range(3, 6):
        registry.record_foreign_write(art, "lag_suppressed", f"{index:064d}")
    assert registry.foreign_write_totals() == {
        art: {"foreign": 1, "mediated": 2, "lag_suppressed": 3}
    }


def test_a_second_note_on_the_same_run_id_overwrites_rather_than_raising(registry) -> None:
    """Both backends must agree: a re-call with no ``close_detection_run``
    between lands on the same run id and OVERWRITES. In memory that is a dict
    keyed on run id; in sqlite it is ``ON CONFLICT (run_id) DO UPDATE``, and
    nothing else reaches that clause — a plain-INSERT mutant survived every
    other test and raises ``IntegrityError`` on exactly this call. Parametrized
    over both so the in-memory side is asserted too: a ``dict.setdefault``
    mutant there (first note wins) survived the whole suite until this ran on
    the memory backend as well. The first note is asserted BEFORE the second
    call: a final-state-only check is also satisfied by a mutant that drops the
    first write and stores the second — and the detector treats "did not raise"
    as durable, so that mutant would lose a workspace's only note."""

    def notes() -> list[tuple[str, float]]:
        return [(n.reason, n.observed_at_unix) for n in registry.detection_uncoverable()]

    registry.record_detection_uncoverable("r1", 100.0)
    assert notes() == [("r1", 100.0)]
    registry.record_detection_uncoverable("r2", 200.0)  # same run id: no close between
    assert notes() == [("r2", 200.0)]


def test_notes_from_separate_runs_come_back_in_observation_order(registry) -> None:
    """``detection_uncoverable`` sorts by ``observed_at_unix`` on both backends —
    not by insertion order and not by run id. Run ids are minted as random
    UUIDs, so left alone a sort-by-run-id mutant would pass whenever the two
    happened to sort the wrong way round — about half the runs. Pinning them so
    that lexical order and insertion order BOTH oppose timestamp order makes
    either mutant fail every time."""
    registry._detection_run_id = "0" * 32
    registry.record_detection_uncoverable("later", 300.0)
    registry.close_detection_run()
    registry._detection_run_id = "f" * 32
    registry.record_detection_uncoverable("earlier", 100.0)
    assert [n.reason for n in registry.detection_uncoverable()] == ["earlier", "later"]


def test_a_writer_handle_raises_on_a_missing_detection_table(tmp_path: Path) -> None:
    """Kills dropping the ``self._read_only and`` qualifier.

    The tolerance is scoped to a read-only handle on purpose: a writer open
    CREATES the detection tables, so their absence under one is a fault in the
    store, not news about its vintage. Returning ``[]`` there would answer a
    question about the workspace with a lie about the schema — and the mutation
    that removes the qualifier left all 96 tests in the three foreign-write
    suites green, because every existing test of this path opens read-only.
    """
    import sqlite3

    db = tmp_path / "state.db"
    reg = SqliteArtifactRegistry(db)
    try:
        conn = sqlite3.connect(db)
        conn.execute("DROP TABLE foreign_write_uncoverable")
        conn.commit()
        conn.close()

        with pytest.raises(sqlite3.OperationalError):
            reg.detection_uncoverable()
    finally:
        reg.close()
