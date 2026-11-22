# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for the CoherentRow Postgres native-CAS binding.

The unit tests are DRIVER-FREE: a fake connection stands in for psycopg at the
binding's connection seam, so no real driver and no database are needed. Where a
test needs psycopg's exception taxonomy (classifying an unconfirmed write), a
minimal stub ``psycopg`` module is installed for that test only.

The ``real_substrate``-marked tests exercise a live Postgres and skip unless
``CCS_TEST_PG_DSN`` (an owner DSN) — and, for the negative-grant test,
``CCS_TEST_PG_LIMITED_DSN`` (the non-owner coherence role) — are set. They are
deselected from the default suite by the orchestrator's marker configuration.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
from collections.abc import Callable

import pytest

from ccs.adapters.coherent_row import (
    CoherentRow,
    ReconcileVerdict,
    provisioning_sql,
)
from ccs.adapters.substrate import CasConflict, CasUnknown, CasWritten
from ccs.core.exceptions import CasVersionConflict, CoherenceError, CommitUnconfirmed, ViewWedged
from ccs.core.substrate import Tier


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --- driver-free fakes ------------------------------------------------------


class _FakeCursor:
    """A cursor context manager over one fake connection."""

    def __init__(self, conn: "_FakeConnection") -> None:
        self._conn = conn
        self.rowcount = 0

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self._conn._run(sql, params or ())
        self.rowcount = self._conn._last_rowcount

    def fetchone(self) -> tuple | None:
        return self._conn._last_row


class _FakeConnection:
    """A minimal psycopg-shaped connection that models the version trigger.

    ``rows`` maps an id to ``(value, version)``. An ``UPDATE ... WHERE version =
    ?`` only matches when the stored version equals the expected one, and lands
    ``OLD.version + 1`` (the owner-trigger semantics). Set ``raise_on_execute`` /
    ``raise_on_commit`` to simulate a driver failure, ``raise_on_select`` to fail
    only the reads, and ``after_update`` to run another writer's change between
    an UPDATE and the next statement.
    """

    def __init__(self, rows: dict[str, tuple[bytes, int]] | None = None) -> None:
        self.rows: dict[str, tuple[bytes, int]] = dict(rows or {})
        self.executed: list[tuple[str, tuple]] = []
        self.committed = 0
        self.rolled_back = 0
        self.raise_on_execute: BaseException | None = None
        self.raise_on_commit: BaseException | None = None
        self.raise_on_select: BaseException | None = None
        self.after_update: Callable[[dict[str, tuple[bytes, int]]], None] | None = None
        self._last_row: tuple | None = None
        self._last_rowcount = 0

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        if self.raise_on_commit is not None:
            raise self.raise_on_commit
        self.committed += 1

    def rollback(self) -> None:
        self.rolled_back += 1

    def _run(self, sql: str, params: tuple) -> None:
        self.executed.append((sql, params))
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        if sql.lstrip().upper().startswith("SELECT"):
            if self.raise_on_select is not None:
                raise self.raise_on_select
            (ref,) = params
            row = self.rows.get(ref)
            self._last_row = row
            self._last_rowcount = 1 if row is not None else 0
        else:  # UPDATE ... WHERE id = %s AND version = %s RETURNING version
            new_value, ref, expected = params
            current = self.rows.get(ref)
            if current is not None and current[1] == expected:
                new_version = current[1] + 1
                self.rows[ref] = (new_value, new_version)
                self._last_row = (new_version,)
                self._last_rowcount = 1
            else:
                self._last_row = None
                self._last_rowcount = 0
            if self.after_update is not None:
                self.after_update(self.rows)


@pytest.fixture
def fake_psycopg(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a stub ``psycopg`` module exposing the DB-API exception taxonomy."""
    mod = types.ModuleType("psycopg")

    class Error(Exception):
        pass

    class DatabaseError(Error):
        pass

    class OperationalError(DatabaseError):
        pass

    class InterfaceError(Error):
        pass

    class ProgrammingError(DatabaseError):
        pass

    mod.Error = Error  # type: ignore[attr-defined]
    mod.DatabaseError = DatabaseError  # type: ignore[attr-defined]
    mod.OperationalError = OperationalError  # type: ignore[attr-defined]
    mod.InterfaceError = InterfaceError  # type: ignore[attr-defined]
    mod.ProgrammingError = ProgrammingError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", mod)
    return mod


def _row(conn: _FakeConnection) -> CoherentRow:
    return CoherentRow(table="workspace_rows", connection=conn)


# --- read -------------------------------------------------------------------


def test_read_returns_bytes_and_token_from_one_query() -> None:
    conn = _FakeConnection({"k": (b"data", 3)})
    assert _row(conn).read("k") == (b"data", "3")


def test_read_absent_row_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        _row(_FakeConnection()).read("missing")


def test_read_sentinel_version_fails_closed() -> None:
    conn = _FakeConnection({"k": (b"x", 0)})
    with pytest.raises(CoherenceError):
        _row(conn).read("k")


# --- cas_write (typed Protocol outcomes) ------------------------------------


def test_cas_write_happy_bumps_version_and_returns_token() -> None:
    conn = _FakeConnection({"k": (b"old", 5)})
    outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"new")
    assert isinstance(outcome, CasWritten)
    assert outcome.token == "6"
    assert conn.rows["k"] == (b"new", 6)
    assert conn.committed == 1


def test_cas_write_conflict_on_zero_rowcount() -> None:
    conn = _FakeConnection({"k": (b"old", 6)})  # a peer already moved the version
    outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"new")
    assert isinstance(outcome, CasConflict)
    assert conn.rows["k"] == (b"old", 6)  # nothing landed
    assert outcome.current_token == _row(conn).read("k")[1] == "6"


def test_a_refusal_reports_the_token_read_back_not_one_computed() -> None:
    """Another writer moved the row twice: the refusal reports what a read
    returns afterwards, not the next version after the comparand."""
    conn = _FakeConnection({"k": (b"seed", 5)})
    conn.rows["k"] = (b"outside-1", 6)
    conn.rows["k"] = (b"outside-2", 7)
    outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"mine")
    assert isinstance(outcome, CasConflict)
    assert outcome.current_token == "7"


def test_a_refusal_reports_a_write_that_landed_after_it() -> None:
    """The token is read after the refusal, so a write landing between the
    refused UPDATE and that read is what gets reported."""
    conn = _FakeConnection({"k": (b"peer", 6)})
    conn.after_update = lambda rows: rows.__setitem__("k", (b"later", 7))
    outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"mine")
    assert isinstance(outcome, CasConflict)
    assert outcome.current_token == "7"


@pytest.mark.parametrize(
    "after_refusal",
    [
        pytest.param(lambda rows: rows.__setitem__("k", (b"re-inserted", 5)), id="expected-version-read-back"),
        pytest.param(lambda rows: rows.pop("k"), id="row-deleted"),
        pytest.param(lambda rows: rows.__setitem__("k", (b"legacy", 0)), id="sentinel-version"),
    ],
)
def test_a_refusal_reports_no_token_it_cannot_vouch_for(
    after_refusal: Callable[[dict[str, tuple[bytes, int]]], None],
) -> None:
    """A read-back equal to the comparand would invite re-driving it, and a
    vanished row or unusable version has no token: all report None, no raise."""
    conn = _FakeConnection({"k": (b"peer", 6)})
    conn.after_update = after_refusal
    outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"mine")
    assert isinstance(outcome, CasConflict)
    assert outcome.current_token is None


@pytest.mark.parametrize("error_name", ["OperationalError", "ProgrammingError"])
def test_a_failed_read_after_a_refusal_keeps_the_refusal(
    fake_psycopg: types.ModuleType, error_name: str, caplog: pytest.LogCaptureFixture
) -> None:
    """The refusal is confirmed before the read, so a failing read reports no
    token rather than an unknown outcome or an error, and leaks no DSN text."""
    conn = _FakeConnection({"k": (b"peer", 6)})
    conn.raise_on_select = getattr(fake_psycopg, error_name)("down; dsn=postgresql://u:hunter2@db/app")
    with caplog.at_level("DEBUG", logger="ccs.adapters.coherent_row"):
        outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"mine")
    assert isinstance(outcome, CasConflict)
    assert outcome.current_token is None
    assert conn.rolled_back >= 1
    assert "hunter2" not in caplog.text
    assert error_name in caplog.text  # the cause stays diagnosable without the driver text


def test_a_failed_read_after_a_refusal_drops_an_owned_connection(
    fake_psycopg: types.ModuleType,
) -> None:
    conns: list[_FakeConnection] = []

    def fake_connect(dsn, **kwargs):  # noqa: ANN001, ANN003, ANN202
        conn = _FakeConnection({"k": (b"peer", 6)})
        if not conns:
            conn.raise_on_select = fake_psycopg.OperationalError("server closed")  # type: ignore[attr-defined]
        conns.append(conn)
        return conn

    fake_psycopg.connect = fake_connect  # type: ignore[attr-defined]
    row = CoherentRow(table="workspace_rows", dsn="postgresql://u:sec@db/app")
    outcome = row.cas_write("k", expected_token="5", new_bytes=b"mine")
    assert isinstance(outcome, CasConflict) and outcome.current_token is None
    assert row.read("k") == (b"peer", "6")  # the next call dials a fresh connection
    assert len(conns) == 2


def test_a_failed_read_after_a_refusal_keeps_a_borrowed_connection(
    fake_psycopg: types.ModuleType,
) -> None:
    conn = _FakeConnection({"k": (b"peer", 6)})
    conn.raise_on_select = fake_psycopg.OperationalError("server closed")  # type: ignore[attr-defined]
    row = _row(conn)
    row.cas_write("k", expected_token="5", new_bytes=b"mine")
    assert row._connection is conn  # caller-owned: never dropped


def test_cas_write_unknown_on_operational_error_during_execute(
    fake_psycopg: types.ModuleType,
) -> None:
    conn = _FakeConnection({"k": (b"old", 5)})
    conn.raise_on_execute = fake_psycopg.OperationalError(  # type: ignore[attr-defined]
        "server closed the connection; dsn=postgresql://u:hunter2@db/app"
    )
    outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"new")
    assert isinstance(outcome, CasUnknown)
    assert conn.rolled_back >= 1


def test_cas_write_unknown_on_operational_error_during_commit(
    fake_psycopg: types.ModuleType,
) -> None:
    conn = _FakeConnection({"k": (b"old", 5)})
    conn.raise_on_commit = fake_psycopg.OperationalError("connection reset")  # type: ignore[attr-defined]
    outcome = _row(conn).cas_write("k", expected_token="5", new_bytes=b"new")
    assert isinstance(outcome, CasUnknown)


def test_cas_write_rejects_sentinel_token() -> None:
    with pytest.raises(CoherenceError):
        _row(_FakeConnection({"k": (b"x", 5)})).cas_write("k", expected_token="0", new_bytes=b"y")


def test_cas_write_rejects_non_integer_token() -> None:
    with pytest.raises(CoherenceError):
        _row(_FakeConnection({"k": (b"x", 5)})).cas_write("k", expected_token="abc", new_bytes=b"y")


# --- public write (typed exceptions at the caller surface) ------------------


def test_write_happy_returns_new_token() -> None:
    conn = _FakeConnection({"k": (b"old", 5)})
    assert _row(conn).write("k", expected_token="5", new_bytes=b"new") == "6"


def test_write_maps_conflict_to_cas_version_conflict() -> None:
    conn = _FakeConnection({"k": (b"old", 6)})
    with pytest.raises(CasVersionConflict) as excinfo:
        _row(conn).write("k", expected_token="5", new_bytes=b"new")
    assert excinfo.value.artifact_id == "k"
    assert excinfo.value.expected_version == 5
    assert excinfo.value.current_version == 6


def test_write_maps_unknown_to_commit_unconfirmed(fake_psycopg: types.ModuleType) -> None:
    conn = _FakeConnection({"k": (b"old", 5)})
    conn.raise_on_execute = fake_psycopg.OperationalError("dropped")  # type: ignore[attr-defined]
    with pytest.raises(CommitUnconfirmed):
        _row(conn).write("k", expected_token="5", new_bytes=b"new")


def test_write_view_wedged_when_row_absent_on_conflict_refetch() -> None:
    # A CAS conflict whose post-conflict re-fetch finds the row gone → ViewWedged
    # (a delete is itself an update). Exercises _current_version_or_wedged's
    # absent branch, which the row-present conflict path never reaches.
    conn = _FakeConnection({})  # absent: UPDATE finds nothing → conflict; refetch → absent
    with pytest.raises(ViewWedged):
        _row(conn).write("k", expected_token="5", new_bytes=b"new")


def test_dsn_owned_connection_reconnects_after_operational_error(
    fake_psycopg: types.ModuleType,
) -> None:
    # A DB blip must not permanently poison a self-owned connection: after an
    # operational error the dead connection is discarded, so the next call
    # reconnects instead of replaying the same failure forever.
    conns: list[_FakeConnection] = []

    def fake_connect(dsn, **kwargs):
        conn = _FakeConnection({"k": (b"v", 3)})
        if not conns:  # the first connection is born broken (server closed)
            conn.raise_on_execute = fake_psycopg.OperationalError("server closed")  # type: ignore[attr-defined]
        conns.append(conn)
        return conn

    fake_psycopg.connect = fake_connect  # type: ignore[attr-defined]
    row = CoherentRow(table="workspace_rows", dsn="postgresql://u:sec@db/app")

    assert isinstance(row.cas_write("k", expected_token="3", new_bytes=b"n"), CasUnknown)
    assert row.read("k") == (b"v", "3")  # reconnected to a fresh, healthy socket
    assert len(conns) == 2  # discarded the poisoned one and dialed again


def test_dsn_connection_applies_connect_and_statement_timeouts(
    fake_psycopg: types.ModuleType,
) -> None:
    # A hung DB must raise (→ reconcilable CasUnknown), not block forever: the
    # self-owned connection carries a connect_timeout and a statement_timeout.
    captured: dict[str, object] = {}

    def fake_connect(dsn, **kwargs):
        captured.update(kwargs)
        return _FakeConnection({"k": (b"v", 1)})

    fake_psycopg.connect = fake_connect  # type: ignore[attr-defined]
    row = CoherentRow(
        table="workspace_rows",
        dsn="postgresql://u@db/app",
        connect_timeout=7.5,
        statement_timeout_ms=12345,
    )
    row.read("k")
    assert captured["connect_timeout"] == 7.5
    assert "statement_timeout=12345" in str(captured["options"])


# --- credential scrubbing ---------------------------------------------------


def test_non_operational_error_reraised_scrubbed(fake_psycopg: types.ModuleType) -> None:
    conn = _FakeConnection({"k": (b"old", 5)})
    conn.raise_on_execute = fake_psycopg.ProgrammingError(  # type: ignore[attr-defined]
        "boom near dsn=postgresql://u:hunter2@db/app password=hunter2"
    )
    with pytest.raises(CoherenceError) as excinfo:
        _row(conn).cas_write("k", expected_token="5", new_bytes=b"new")
    message = str(excinfo.value)
    assert "hunter2" not in message
    assert "password" not in message
    assert "postgresql://" not in message
    # `raise ... from None` suppresses the chained driver error, so the traceback
    # cannot leak the DSN either.
    assert excinfo.value.__cause__ is None


# --- unknown reconciliation (token-identity authority) ----------------------


def test_reconcile_converges_only_on_expected_plus_one_and_bytes() -> None:
    intended = b"new"
    conn = _FakeConnection({"k": (intended, 6)})  # expected 5 -> landed at 6
    decision = _row(conn).reconcile_after_unknown(
        "k", expected_token="5", intended_hash=_sha(intended)
    )
    assert decision.verdict is ReconcileVerdict.CONVERGE
    assert decision.observed_bytes == intended
    assert decision.observed_token == "6"


def test_reconcile_bytes_match_but_wrong_version_does_not_converge() -> None:
    intended = b"new"
    conn = _FakeConnection({"k": (intended, 7)})  # version jumped past expected+1
    decision = _row(conn).reconcile_after_unknown(
        "k", expected_token="5", intended_hash=_sha(intended)
    )
    assert decision.verdict is ReconcileVerdict.RE_DERIVE


def test_reconcile_right_version_but_different_bytes_re_derives() -> None:
    conn = _FakeConnection({"k": (b"peer-content", 6)})  # a different writer won
    decision = _row(conn).reconcile_after_unknown(
        "k", expected_token="5", intended_hash=_sha(b"new")
    )
    assert decision.verdict is ReconcileVerdict.RE_DERIVE


def test_reconcile_version_unchanged_re_derives() -> None:
    conn = _FakeConnection({"k": (b"old", 5)})  # write never landed
    decision = _row(conn).reconcile_after_unknown(
        "k", expected_token="5", intended_hash=_sha(b"new")
    )
    assert decision.verdict is ReconcileVerdict.RE_DERIVE


def test_reconcile_absent_row_holds_never_matches_empty_hash() -> None:
    conn = _FakeConnection()  # row absent — intended_hash is sha256(b"") on purpose
    decision = _row(conn).reconcile_after_unknown(
        "k", expected_token="5", intended_hash=_sha(b"")
    )
    assert decision.verdict is ReconcileVerdict.HOLD
    assert decision.observed_bytes is None
    assert decision.observed_token is None


def test_reconcile_sentinel_version_holds() -> None:
    conn = _FakeConnection({"k": (b"", 0)})  # present-but-sentinel version
    decision = _row(conn).reconcile_after_unknown(
        "k", expected_token="5", intended_hash=_sha(b"")
    )
    assert decision.verdict is ReconcileVerdict.HOLD


# --- descriptor + honesty ---------------------------------------------------


def test_descriptor_is_native_cas_with_version_source() -> None:
    descriptor = _row(_FakeConnection()).descriptor
    assert descriptor.tier is Tier.NATIVE_CAS
    assert descriptor.version_source == "trigger-managed version column"
    assert "SELECT, UPDATE" in descriptor.least_privilege


def test_binding_never_threads_content_to_coordinator() -> None:
    assert CoherentRow.SENDS_CONTENT_TO_COORDINATOR is False
    assert _row(_FakeConnection()).coordinator_commit_content() is None


def test_satisfies_coherence_substrate_protocol() -> None:
    from ccs.adapters.substrate import CoherenceSubstrate

    assert isinstance(_row(_FakeConnection()), CoherenceSubstrate)


# --- deferred driver import -------------------------------------------------


def test_module_imports_without_psycopg_and_defers_error() -> None:
    if importlib.util.find_spec("psycopg") is not None:
        pytest.skip("psycopg installed; the deferred-import path only fires when it is absent")
    # Construction with a dsn does NOT import the driver...
    row = CoherentRow(table="workspace_rows", dsn="postgresql://u:secret@db/app")
    # ...the clear, extra-naming ImportError fires only on actual use.
    with pytest.raises(ImportError, match="coherent-row"):
        row.read("k")


# --- provisioning DDL / negative grants -------------------------------------


def test_provisioning_sql_emits_version_guard_and_negative_grants() -> None:
    script = provisioning_sql("workspace_rows", role="coherence_writer").as_script()
    # The version is minted from the STORED prior row, never a client value; the
    # COALESCE lets a pre-existing NULL/0 row advance to a usable version.
    assert 'COALESCE(OLD."version", 0) + 1' in script
    assert "never trust a client-supplied NEW.version" in script
    # A one-time backfill seeds pre-existing NULL/0 rows so they can be onboarded.
    assert 'UPDATE "workspace_rows" SET "version" = 1 WHERE "version" IS NULL OR "version" <= 0' in script
    # Least privilege: SELECT, UPDATE only; no escalation.
    assert "GRANT SELECT, UPDATE" in script
    assert "NOSUPERUSER" in script
    assert "NOCREATEDB" in script
    assert "NOCREATEROLE" in script
    # No GRANT confers ALTER / TRIGGER / DELETE, and no grant is re-grantable.
    grant_lines = [ln for ln in script.splitlines() if ln.strip().upper().startswith("GRANT")]
    assert grant_lines
    for line in grant_lines:
        upper = line.upper()
        assert "ALTER" not in upper
        assert "TRIGGER" not in upper
        assert "DELETE" not in upper
    assert "WITH GRANT OPTION" not in script


def test_provisioning_sql_trigger_names_are_single_quoted_identifiers() -> None:
    # Regression: each trigger name must be ONE quoted identifier, never the
    # quoted function name with a bareword suffix (`"..."_ins`), which Postgres
    # parses as two tokens — a syntax error at "_ins" that breaks the one-time
    # setup the whole NATIVE_CAS guarantee rides on.
    script = provisioning_sql("workspace_rows").as_script()
    assert '"_ins' not in script and '"_upd' not in script  # no quote-then-bareword
    assert 'CREATE TRIGGER "workspace_rows_coherence_version_ins"' in script
    assert 'CREATE TRIGGER "workspace_rows_coherence_version_upd"' in script
    assert 'DROP TRIGGER IF EXISTS "workspace_rows_coherence_version_ins"' in script


def test_provisioning_sql_rejects_unsafe_identifiers() -> None:
    with pytest.raises(ValueError):
        provisioning_sql('rows"; DROP TABLE users; --')


def test_binding_rejects_unsafe_table_identifier() -> None:
    with pytest.raises(ValueError):
        CoherentRow(table="rows; DROP TABLE users", connection=_FakeConnection())


# ---------------------------------------------------------------------------
# Integration tests against a REAL Postgres (deselected by default).
# ---------------------------------------------------------------------------

_IT_TABLE = "coherence_row_it"


@pytest.fixture
def real_pg():
    """Provision a real Postgres table + version trigger; yield ``(dsn, table)``."""
    dsn = os.environ.get("CCS_TEST_PG_DSN")
    if not dsn:
        pytest.skip("set CCS_TEST_PG_DSN (owner DSN) to run real Postgres tests")
    import psycopg  # noqa: PLC0415

    ddl = provisioning_sql(_IT_TABLE)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'DROP TABLE IF EXISTS "{_IT_TABLE}" CASCADE')
        cur.execute(
            f'CREATE TABLE "{_IT_TABLE}" '
            "(id text PRIMARY KEY, value bytea NOT NULL, version int NOT NULL DEFAULT 0)"
        )
        cur.execute(ddl.trigger_function)
        cur.execute(ddl.trigger_bindings)
        cur.execute(f'INSERT INTO "{_IT_TABLE}" (id, value) VALUES (%s, %s)', ("row1", b"seed"))
        conn.commit()
    try:
        yield dsn, _IT_TABLE
    finally:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{_IT_TABLE}" CASCADE')
            conn.commit()


@pytest.mark.real_substrate
def test_it_concurrent_lost_update_one_winner_then_converge(real_pg) -> None:
    dsn, table = real_pg
    agent_a = CoherentRow(table=table, dsn=dsn)
    agent_b = CoherentRow(table=table, dsn=dsn)
    _bytes_a, token_a = agent_a.read("row1")
    _bytes_b, token_b = agent_b.read("row1")  # both hold the same stale version
    assert token_a == token_b

    token_after = agent_a.write("row1", expected_token=token_a, new_bytes=b"from-a")
    with pytest.raises(CasVersionConflict):
        agent_b.write("row1", expected_token=token_b, new_bytes=b"from-b")

    _fresh, token_fresh = agent_b.read("row1")  # re-read the winner's version
    assert token_fresh == token_after
    agent_b.write("row1", expected_token=token_fresh, new_bytes=b"from-b")
    final_bytes, _ = agent_b.read("row1")
    assert final_bytes == b"from-b"


@pytest.mark.real_substrate
def test_it_trigger_authority_ignores_client_new_version(real_pg) -> None:
    dsn, table = real_pg
    import psycopg  # noqa: PLC0415

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT version FROM "{table}" WHERE id = %s', ("row1",))
        (before,) = cur.fetchone()
        # A raw client UPDATE that forges an arbitrary NEW.version...
        cur.execute(
            f'UPDATE "{table}" SET value = %s, version = %s WHERE id = %s',
            (b"x", 999_999, "row1"),
        )
        cur.execute(f'SELECT version FROM "{table}" WHERE id = %s', ("row1",))
        (after,) = cur.fetchone()
        conn.commit()
    assert after == before + 1  # the owner trigger overrode 999999 with OLD.version + 1


@pytest.mark.real_substrate
def test_it_foreign_write_moves_version_agent_cas_fails(real_pg) -> None:
    dsn, table = real_pg
    import psycopg  # noqa: PLC0415

    agent = CoherentRow(table=table, dsn=dsn)
    _bytes, token = agent.read("row1")
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:  # a non-agent writer
        cur.execute(f'UPDATE "{table}" SET value = %s WHERE id = %s', (b"foreign", "row1"))
        conn.commit()
    with pytest.raises(CasVersionConflict):
        agent.write("row1", expected_token=token, new_bytes=b"from-agent")


@pytest.mark.real_substrate
def test_it_unknown_write_reconciles_on_token_identity(real_pg) -> None:
    dsn, table = real_pg
    agent = CoherentRow(table=table, dsn=dsn)
    _bytes, token = agent.read("row1")
    intended = b"from-agent"
    # A committed write is the "the UPDATE actually landed" leg of an unknown; the
    # reconciliation converges only because the version is exactly expected + 1
    # under our identity AND the bytes hash to what we intended.
    new_token = agent.write("row1", expected_token=token, new_bytes=intended)
    decision = agent.reconcile_after_unknown(
        "row1", expected_token=token, intended_hash=hashlib.sha256(intended).hexdigest()
    )
    assert decision.verdict is ReconcileVerdict.CONVERGE
    assert decision.observed_token == new_token


@pytest.mark.real_substrate
def test_it_limited_role_cannot_disable_the_trigger() -> None:
    limited_dsn = os.environ.get("CCS_TEST_PG_LIMITED_DSN")
    if not limited_dsn:
        pytest.skip("set CCS_TEST_PG_LIMITED_DSN (non-owner coherence role) to run this test")
    import psycopg  # noqa: PLC0415

    with psycopg.connect(limited_dsn) as conn, conn.cursor() as cur:
        # The coherence role holds only SELECT, UPDATE — it is not the owner, so it
        # cannot ALTER the table or disable the version guard.
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            cur.execute(f'ALTER TABLE "{_IT_TABLE}" DISABLE TRIGGER ALL')


@pytest.mark.real_substrate
def test_it_xmin_wraparound_is_the_missed_conflict_hazard(real_pg) -> None:
    # Documents WHY xmin is not shipped: xmin is equality-only and 32-bit, so after
    # wraparound (or VACUUM FREEZE) a fresh xmin can EQUAL a stale captured one — a
    # missed conflict (false CAS-pass) the trigger version column cannot suffer.
    dsn, table = real_pg
    import psycopg  # noqa: PLC0415

    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(f'SELECT xmin::text::bigint FROM "{table}" WHERE id = %s', ("row1",))
        (xmin_before,) = cur.fetchone()
        cur.execute(f'UPDATE "{table}" SET value = %s WHERE id = %s', (b"y", "row1"))
        conn.commit()
        cur.execute(f'SELECT xmin::text::bigint FROM "{table}" WHERE id = %s', ("row1",))
        (xmin_after,) = cur.fetchone()
    # xmin moves on this update, but it is NOT monotonic and can repeat — unlike the
    # trigger version column, which only ever increases.
    assert xmin_after != xmin_before
