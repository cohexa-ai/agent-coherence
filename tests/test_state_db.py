# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The shared read-only open, and the two promises it makes to both readers.

``ccs.diagnose._state_db`` is the one place the offline readers open a
coordinator ``state.db``, so its two guarantees are theirs: the handle is
read-only, and a failure is loud. Both are stated in prose the readers repeat
to their callers, and neither was pinned by a test — the readers' own suites
build every path from ``tmp_path``, whose components never carry a character
the URI layer treats as syntax.

That gap hid a real defect. The store path is interpolated into a SQLite URI
whose query string is the only thing that makes the handle read-only, so a path
containing ``#``, ``?`` or ``%`` was parsed as URI syntax rather than as
filename bytes: SQLite opened a different file, ``mode=ro`` was displaced into
the fragment and silently lost, and ``read_conflict_totals`` answered ``{}`` for
a store that held real conflicts. A fabricated zero is the exact failure this
subsystem exists to prevent, and the suite stayed green through all of it.

The older tests were not weak — their input domain could not express the bug.
``tmp_path`` yields only ``[A-Za-z0-9/_-]``, so no assertion added over that
fixture could have failed. The rule this file exists to carry forward: when a
fixture generates the input, the fixture silently decides which bugs are
reachable, so a path-taking API has to be tested on paths the platform allows
and the fixture never emits. ``URI_HOSTILE_NAMES`` below is that set for a
directory component; the tests after it cover the shapes that are not
directory names at all — a NUL, a ``//`` authority, and ``:memory:``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ccs.diagnose import _state_db
from ccs.diagnose._state_db import BUSY_TIMEOUT_MS, open_readonly_state_db
from ccs.diagnose.conflict_counters import read_conflict_totals
from ccs.diagnose.foreign_writes import INSTRUMENTED_ZERO, read_foreign_write_report

# Characters the URI layer treats as syntax: '#' starts a fragment, '?' starts
# the query, '%' introduces an escape. A space is the ordinary case that must
# keep working alongside them.
URI_HOSTILE_NAMES = ["proj#1", "proj?x", "proj%2e", "proj dir"]


def _seed_conflict_row(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE conflict_counters "
        "(artifact_id TEXT, agent_id TEXT, reason TEXT, count INT)"
    )
    conn.execute("INSERT INTO conflict_counters VALUES ('aa', 'bb', 'version_mismatch', 7)")
    conn.commit()
    conn.close()


def _seed_observation_row(db: Path) -> None:
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE foreign_write_observations (run_id TEXT, first_tick_unix REAL, "
        "last_tick_unix REAL, tick_count INT, covered_count INT)"
    )
    conn.execute(
        "CREATE TABLE foreign_write_counters (artifact_id TEXT, foreign_count INT, "
        "mediated_count INT, lag_suppressed_count INT)"
    )
    conn.execute("INSERT INTO foreign_write_observations VALUES ('r1', 10.0, 20.0, 5, 1)")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# A path is filename bytes, never URI syntax
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dirname", URI_HOSTILE_NAMES)
def test_a_uri_reserved_character_in_the_path_still_reads_the_real_store(
    tmp_path: Path, dirname: str
) -> None:
    """The worst failure this subsystem has: a reader that answers "zero" about
    a store it never opened. Unencoded, '#' truncated the path and SQLite
    created and read an empty database beside the real one, so the count-7 row
    below came back as ``{}`` — indistinguishable from an honestly quiet month.
    """
    db = tmp_path / dirname / "state.db"
    db.parent.mkdir()
    _seed_conflict_row(db)

    assert read_conflict_totals(db) == {("aa", "bb", "version_mismatch"): 7}


@pytest.mark.parametrize("dirname", URI_HOSTILE_NAMES)
def test_a_uri_reserved_character_in_the_path_keeps_the_detector_state(
    tmp_path: Path, dirname: str
) -> None:
    """The same displacement on the sibling reader: a store the detector really
    ran against must not read as one it never ran against."""
    db = tmp_path / dirname / "state.db"
    db.parent.mkdir()
    _seed_observation_row(db)

    assert read_foreign_write_report(db).state == INSTRUMENTED_ZERO


@pytest.mark.parametrize("dirname", URI_HOSTILE_NAMES)
def test_a_uri_reserved_character_in_the_path_stays_read_only(
    tmp_path: Path, dirname: str
) -> None:
    """``mode=ro`` lives in the URI query string, so anything that displaces the
    query silently hands back a writable handle to a store the caller was
    promised would not be touched."""
    db = tmp_path / dirname / "state.db"
    db.parent.mkdir()
    _seed_conflict_row(db)

    conn = open_readonly_state_db(db)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE proof_of_write (x)")
    finally:
        conn.close()


@pytest.mark.parametrize("dirname", URI_HOSTILE_NAMES)
def test_a_uri_reserved_character_in_the_path_creates_no_stray_file(
    tmp_path: Path, dirname: str
) -> None:
    """A truncated path does not fail loudly — SQLite creates the shorter name.
    Asserted on the parent directory, where the stray sibling appeared."""
    db = tmp_path / dirname / "state.db"
    db.parent.mkdir()
    _seed_conflict_row(db)
    before = sorted(p.name for p in tmp_path.iterdir())

    read_conflict_totals(db)

    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_missing_store_under_a_reserved_character_path_still_raises(
    tmp_path: Path,
) -> None:
    """Encoding must not cost the missing-file translation: an absent store is a
    caller error, never evidence of zero."""
    with pytest.raises(FileNotFoundError):
        read_conflict_totals(tmp_path / "proj#1" / "nope.db")


# ---------------------------------------------------------------------------
# Shapes that must not resolve to a store the caller did not name
# ---------------------------------------------------------------------------


def test_a_nul_byte_in_the_path_is_rejected_rather_than_truncated(
    tmp_path: Path,
) -> None:
    """Percent-encoding alone re-creates the very bug it fixes here: a NUL
    survives as %00, SQLite decodes it, and the C string truncates there — so
    a path ending '.../state.db\x00.evil' would read the real store and report
    it as the caller's. Python raises ValueError for a NUL in any real
    filesystem call; the URI layer must not quietly succeed instead."""
    db = tmp_path / "state.db"
    _seed_conflict_row(db)

    with pytest.raises(ValueError, match="NUL byte"):
        read_conflict_totals(Path(f"{db}\x00.evil"))


def test_an_in_memory_path_is_not_a_store_and_never_reads_as_zero() -> None:
    """':memory:' is SQLite's in-memory store, so an unqualified path reaching
    the URI intact answers {} for a database that never existed — a zero with
    nothing behind it, which is the one answer this reader may not invent."""
    with pytest.raises(FileNotFoundError):
        read_conflict_totals(Path(":memory:"))


def test_a_leading_double_slash_is_not_read_as_a_uri_authority(
    tmp_path: Path,
) -> None:
    """'//host/path' is authority syntax to the URI parser. Left unescaped it
    strips to '/path', so the reader opens a store one level up from the name
    it was given and reports it as that name's."""
    db = tmp_path / "state.db"
    _seed_conflict_row(db)

    with pytest.raises(FileNotFoundError):
        read_conflict_totals(Path(f"//localhost{db}"))


def test_an_undecodable_filesystem_byte_still_raises_file_not_found(
    tmp_path: Path,
) -> None:
    """POSIX filenames are bytes, and Python surfaces an undecodable one as a
    surrogate. Strict encoding would raise UnicodeEncodeError straight past
    both readers' documented failure types; the open must still be attempted so
    the answer stays FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_conflict_totals(tmp_path / "\udcff.db")


# ---------------------------------------------------------------------------
# The handle the caller never received must not leak
# ---------------------------------------------------------------------------


def test_the_busy_timeout_is_applied_to_the_returned_handle(tmp_path: Path) -> None:
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()

    conn = open_readonly_state_db(db)
    try:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == BUSY_TIMEOUT_MS
    finally:
        conn.close()


def test_the_connection_is_closed_when_the_busy_timeout_pragma_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PRAGMA used to run inside each caller's ``try``/``finally``, so a
    failure there already closed the handle. Moving it into the helper keeps
    that only because of the cleanup guard — without it the caller cannot close
    a handle it never received, and the failure leaks it."""
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()
    closed: list[str] = []
    real_connect = sqlite3.connect

    class _PragmaRejectingConnection(sqlite3.Connection):
        def execute(self, *args: object, **kwargs: object) -> sqlite3.Cursor:
            raise sqlite3.OperationalError("pragma rejected")

        def close(self) -> None:
            closed.append("closed")
            super().close()

    monkeypatch.setattr(
        _state_db.sqlite3,
        "connect",
        lambda *a, **k: real_connect(*a, factory=_PragmaRejectingConnection, **k),
    )

    with pytest.raises(sqlite3.OperationalError, match="pragma rejected"):
        open_readonly_state_db(db)

    assert closed == ["closed"], "the orphaned handle must be closed, not leaked"


def test_the_connection_is_closed_when_the_pragma_raises_a_base_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard catches BaseException, not Exception, and the breadth is the
    point: a Ctrl-C landing on the PRAGMA is exactly when a handle the caller
    never received would leak. Narrowing the guard passes every ordinary-error
    test, so this is the one that fails."""
    db = tmp_path / "state.db"
    sqlite3.connect(db).close()
    closed: list[str] = []
    real_connect = sqlite3.connect

    class _InterruptingConnection(sqlite3.Connection):
        def execute(self, *args: object, **kwargs: object) -> sqlite3.Cursor:
            raise KeyboardInterrupt

        def close(self) -> None:
            closed.append("closed")
            super().close()

    monkeypatch.setattr(
        _state_db.sqlite3,
        "connect",
        lambda *a, **k: real_connect(*a, factory=_InterruptingConnection, **k),
    )

    with pytest.raises(KeyboardInterrupt):
        open_readonly_state_db(db)

    assert closed == ["closed"], "an interrupt must not leak the handle either"
