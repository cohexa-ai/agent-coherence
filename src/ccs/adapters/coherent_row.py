# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""CoherentRow — coherence over a Postgres row (native-CAS binding).

Bring coherence to a row you already run in Postgres. The row's ``value`` never
leaves Postgres — the coordinator holds only a monotonic version plus a
fixed-width fingerprint, never the bytes. No-lost-update rides the row's own
version column: a write is ``UPDATE ... SET value = ?, version = version + 1
WHERE id = ? AND version = ?``, so a peer who moved the version wins the race and
the stale writer's ``rowcount`` comes back ``0``.

**The version is minted by the database, not the client.** A confused-deputy-proof
``BEFORE INSERT`` / ``BEFORE UPDATE`` trigger owned by the table owner sets
``NEW.version := OLD.version + 1`` from the *stored* prior row — never a
client-supplied ``NEW.version``. The coherence login role is a dedicated,
non-owner role with only ``SELECT, UPDATE`` on the one table, so it cannot
``ALTER`` the table or disable the trigger to forge a version. :func:`provisioning_sql`
emits the exact DDL + least-privilege grants an operator applies once.

The binding reads ``(bytes, version)`` from a single query — the token and the
bytes it vouches for always come from the same read, so a concurrent update can
never be silently lost across a split read.

Install::

    pip install "agent-coherence[coherent-row]"

The ``psycopg`` (v3) driver is imported lazily, so importing this module without
the driver installed is fine — the clear install error only fires when a
connection is actually needed.

.. note::

   ``xmin`` (Postgres' system row-version) is a *weaker* fallback and is **not**
   shipped here. It is equality-only and 32-bit, so its value can repeat after
   wraparound (or a ``VACUUM FREEZE``) — a repeat is a *missed* conflict, a
   false CAS-pass that silently loses an update. The trigger-managed ``version``
   column is the only version source this binding uses.
"""

from __future__ import annotations

import logging
import re
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from ccs.adapters.substrate import (
    CasConflict,
    CasUnknown,
    CasWriteResult,
    CasWritten,
    ReconcileDecision,
    ReconcileVerdict,
    SubstrateToken,
)
from ccs.core.exceptions import (
    CasVersionConflict,
    CoherenceError,
    CommitUnconfirmed,
    ViewWedged,
)
from ccs.core.substrate import CapabilityDescriptor, Tier
from ccs.core.substrate import sha256_hex as _sha256_hex

if TYPE_CHECKING:
    from ccs.adapters.substrate import CoherenceSubstrate

logger = logging.getLogger(__name__)

__all__ = [
    "CoherentRow",
    "ProvisioningSql",
    "ReconcileDecision",
    "ReconcileVerdict",
    "provisioning_sql",
]

# The one capability this binding declares. NATIVE_CAS because the substrate's
# own atomic conditional write (UPDATE ... WHERE version = ?) rejects a lost
# update on the version axis; the guarantee wording is derived from the tier.
_DESCRIPTOR = CapabilityDescriptor(
    tier=Tier.NATIVE_CAS,
    version_source="trigger-managed version column",
    least_privilege=(
        "a dedicated non-owner login role with SELECT, UPDATE on the single "
        "coherence table only — no ALTER, no TRIGGER/DISABLE TRIGGER, no "
        "CREATEDB/CREATEROLE/SUPERUSER"
    ),
    consistency_note=(
        "single-primary Postgres: read-after-write is strongly consistent on the "
        "primary. Never point the CAS loop at a read replica — replica lag can "
        "serve a stale version and miss a conflict."
    ),
)

# A Postgres identifier we are willing to interpolate into SQL. Config-supplied
# table/column names are validated against this and then double-quoted, so a
# crafted name cannot break out of the identifier position.
_IDENTIFIER_RE = re.compile(r"\A[A-Za-z_][A-Za-z0-9_]*\Z")


def _quote_identifier(name: str) -> str:
    """Validate one identifier component and return it double-quoted."""
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"unsafe Postgres identifier: {name!r}")
    return f'"{name}"'


def _quote_table(table: str) -> str:
    """Validate and quote a possibly schema-qualified table name."""
    return ".".join(_quote_identifier(part) for part in table.split("."))


# --- the driver seam --------------------------------------------------------
#
# The binding depends on only this slice of the psycopg connection surface, so a
# fake connection (unit tests) and a pooled real connection (Unit 5) both satisfy
# it without pulling the driver into import time.


class PgCursor(Protocol):
    """The cursor surface the binding uses (a psycopg v3 cursor satisfies it)."""

    rowcount: int

    def execute(self, query: str, params: object = ...) -> object: ...

    def fetchone(self) -> tuple | None: ...


class PgConnection(Protocol):
    """The connection surface the binding uses."""

    def cursor(self) -> AbstractContextManager[PgCursor]: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...


def _require_psycopg():
    """Import psycopg or raise a clear install error naming the extra."""
    try:
        import psycopg  # noqa: PLC0415  (deferred by design — see the module docstring)
    except ImportError as exc:  # pragma: no cover - exercised only without the driver
        raise ImportError(
            "CoherentRow requires the psycopg v3 driver. Install it with: "
            'pip install "agent-coherence[coherent-row]"'
        ) from exc
    return psycopg


def _optional_psycopg():
    """Return the psycopg module if importable, else ``None`` (fail-soft).

    Used only to classify a driver error as an unconfirmed operational failure.
    When the driver is absent there is nothing to classify against, so callers
    treat an unrecognized error as a hard failure rather than guessing.
    """
    try:
        import psycopg  # noqa: PLC0415
    except ImportError:
        return None
    return psycopg


def _is_unconfirmed_error(exc: BaseException) -> bool:
    """True iff ``exc`` is a driver operational/interface error — i.e. the write
    may or may not have landed and must be reconciled, not assumed failed."""
    psycopg = _optional_psycopg()
    if psycopg is None:
        return False
    return isinstance(exc, (psycopg.OperationalError, psycopg.InterfaceError))


# The reconciliation seam (:class:`ReconcileVerdict` / :class:`ReconcileDecision`)
# is the UNIFIED type from ``ccs.adapters.substrate`` — one vocabulary shared with
# the S3 arm so the cross-agent commit dispatches uniformly. This binding uses
# only its Postgres-honest arms: CONVERGE (version ``expected + 1`` under MY
# identity), RE_DERIVE, and HOLD. Re-exported below for callers.


# --- provisioning DDL (emitted, never executed at runtime) ------------------


@dataclass(frozen=True)
class ProvisioningSql:
    """The one-time DDL + least-privilege grants that make the version guard
    agent-unremovable.

    :attr:`trigger_function` + :attr:`trigger_bindings` install the
    confused-deputy-proof version trigger (owned by the table owner);
    :attr:`role_grants` create the limited coherence login role. An operator
    applies :meth:`as_script` once, as the table owner — the binding never runs
    DDL at write time.
    """

    trigger_function: str
    trigger_bindings: str
    role_grants: str
    version_backfill: str = ""

    def as_script(self) -> str:
        """The full provisioning script, in apply order."""
        parts = [self.trigger_function, self.trigger_bindings, self.role_grants]
        if self.version_backfill:
            parts.append(self.version_backfill)
        return "\n\n".join(parts)


def provisioning_sql(
    table: str,
    *,
    role: str = "coherence_writer",
    version_column: str = "version",
    connection_limit: int = 16,
) -> ProvisioningSql:
    """Emit the version-guard trigger + least-privilege grants for one row table.

    The trigger derives the new version from the STORED prior row
    (``NEW.version := COALESCE(OLD.version, 0) + 1``) so a client that supplies
    its own ``NEW.version`` cannot forge one, and a pre-existing row with a
    ``NULL``/``0`` version still advances to a usable value. The role is a
    dedicated non-owner login with only ``SELECT, UPDATE`` on the one table — it
    holds no ``ALTER`` or ``TRIGGER`` privilege, so it cannot drop or disable the
    guard. ``connection_limit`` caps concurrent logins for the shared coherence
    role (raise it for a larger agent fleet).

    The emitted ``version_backfill`` seeds any pre-existing row sitting at
    ``NULL``/``0`` to version ``1`` so it can be onboarded — the binding refuses a
    version ``<= 0`` as a CAS comparand, so without this a legacy row would be
    unreadable and therefore un-writable through this binding.
    """
    if not isinstance(connection_limit, int) or isinstance(connection_limit, bool) or connection_limit < 1:
        raise ValueError(f"connection_limit must be a positive int, got {connection_limit!r}")
    tbl = _quote_table(table)
    ver = _quote_identifier(version_column)
    _quote_identifier(role)  # validate the role name before interpolating it
    base = table.split(".")[-1]
    fn = _quote_identifier(f"{base}_coherence_version")
    # Trigger names are WHOLE quoted identifiers, not a suffix concatenated onto
    # the already-quoted function name: `"..."_ins` is two tokens where Postgres
    # wants one (a syntax error at "_ins"), which would break the one-time setup.
    trig_ins = _quote_identifier(f"{base}_coherence_version_ins")
    trig_upd = _quote_identifier(f"{base}_coherence_version_upd")

    trigger_function = (
        f"-- Version guard: mint {ver} from the STORED prior row. Owned by the\n"
        f"-- table owner; never trust a client-supplied NEW.{version_column}.\n"
        f"CREATE OR REPLACE FUNCTION {fn}() RETURNS trigger AS $$\n"
        "BEGIN\n"
        "    IF TG_OP = 'UPDATE' THEN\n"
        f"        NEW.{ver} := COALESCE(OLD.{ver}, 0) + 1;\n"
        "    ELSE  -- INSERT: no stored prior, so the version starts at 1\n"
        f"        NEW.{ver} := 1;\n"
        "    END IF;\n"
        "    RETURN NEW;\n"
        "END;\n"
        "$$ LANGUAGE plpgsql;"
    )
    trigger_bindings = (
        f"DROP TRIGGER IF EXISTS {trig_ins} ON {tbl};\n"
        f"CREATE TRIGGER {trig_ins} BEFORE INSERT ON {tbl}\n"
        f"    FOR EACH ROW EXECUTE FUNCTION {fn}();\n"
        f"DROP TRIGGER IF EXISTS {trig_upd} ON {tbl};\n"
        f"CREATE TRIGGER {trig_upd} BEFORE UPDATE ON {tbl}\n"
        f"    FOR EACH ROW EXECUTE FUNCTION {fn}();"
    )
    role_grants = (
        f'-- Dedicated, login-limited, NON-owner coherence role. It can read and\n'
        f'-- update the row, and nothing else — it cannot ALTER the table, drop or\n'
        f'-- disable the version trigger, or create databases/roles.\n'
        f'CREATE ROLE "{role}" LOGIN CONNECTION LIMIT {connection_limit}\n'
        f"    NOSUPERUSER NOCREATEDB NOCREATEROLE NOINHERIT;\n"
        f'REVOKE ALL ON {tbl} FROM "{role}";\n'
        f'GRANT SELECT, UPDATE ON {tbl} TO "{role}";\n'
        f"-- Deliberately withheld: table ALTER, the TRIGGER privilege, row DELETE,\n"
        f"-- and any re-grant right."
    )
    version_backfill = (
        f"-- One-time backfill: a row that predates the version column sits at\n"
        f"-- NULL/0, which the binding refuses as a CAS comparand (must be > 0).\n"
        f"-- Seed such rows to 1 so they can be onboarded. Idempotent, and the\n"
        f"-- trigger's COALESCE keeps it safe whether the trigger is installed yet.\n"
        f"UPDATE {tbl} SET {ver} = 1 WHERE {ver} IS NULL OR {ver} <= 0;"
    )
    return ProvisioningSql(trigger_function, trigger_bindings, role_grants, version_backfill)


# --- the binding ------------------------------------------------------------


class CoherentRow:
    """Coherence over a single Postgres row, keyed by its ``id``.

    Construct with either a live/injected ``connection`` or a ``dsn`` (the driver
    is connected lazily on first use). The binding implements the
    :class:`~ccs.adapters.substrate.CoherenceSubstrate` surface: :meth:`read`
    returns ``(bytes, token)`` from one query, and :meth:`cas_write` maps to
    ``UPDATE ... WHERE version = ?`` and returns a typed win / conflict / unknown.

    The token is the row's version rendered as a string. It is NOT portable
    across artifact refs — the coordinator record is keyed by artifact id, so a
    token minted for one row can never arbitrate a write to another.
    """

    #: Bytes threaded to the coordinator on commit: never any. The coordinator
    #: holds only a version + a fixed-width fingerprint for this binding, so the
    #: row body is never shadowed coordinator-side (never-ship-a-store). This is
    #: the binding's self-declaration; the conformance kit asserts it, and
    #: :class:`~ccs.adapters.substrate.CoordinatedSubstrate` refuses at composition
    #: any binding that sets it True. The runtime enforcement lives in
    #: ``SubstrateCoordinatorSession.commit_cas`` (a content_hash-only payload).
    SENDS_CONTENT_TO_COORDINATOR: bool = False

    def __init__(
        self,
        table: str,
        *,
        id_column: str = "id",
        value_column: str = "value",
        version_column: str = "version",
        connection: PgConnection | None = None,
        dsn: str | None = None,
        connect_timeout: float | None = 10.0,
        statement_timeout_ms: int | None = 30_000,
    ) -> None:
        if connection is None and dsn is None:
            raise ValueError("CoherentRow needs a connection or a dsn (fail-closed)")
        self._table = table
        self._connection = connection
        self._dsn = dsn
        # Bounded waits for a self-owned (dsn) connection: a hung DB then raises
        # (→ CasUnknown, which is reconcilable) instead of blocking the CAS loop
        # forever. A caller-injected connection keeps its own settings.
        self._connect_timeout = connect_timeout
        self._statement_timeout_ms = statement_timeout_ms
        tbl = _quote_table(table)
        idc = _quote_identifier(id_column)
        val = _quote_identifier(value_column)
        ver = _quote_identifier(version_column)
        self._select_sql = f"SELECT {val}, {ver} FROM {tbl} WHERE {idc} = %s"
        # The client also increments version; the owner-trigger overrides it from
        # the stored prior, so the two agree and the guard holds even mid-rollout.
        self._update_sql = (
            f"UPDATE {tbl} SET {val} = %s, {ver} = {ver} + 1 "
            f"WHERE {idc} = %s AND {ver} = %s RETURNING {ver}"
        )

    # --- capability -----------------------------------------------------------

    @property
    def descriptor(self) -> CapabilityDescriptor:
        """The binding's honest capability declaration (native-CAS)."""
        return _DESCRIPTOR

    def coordinator_commit_content(self) -> None:
        """The content threaded to the coordinator on commit: always ``None``.

        The row body stays in Postgres; the coordinator holds only a version +
        fingerprint. A declarative companion to
        :attr:`SENDS_CONTENT_TO_COORDINATOR`, asserted by the conformance kit; the
        actual content-free payload is built in ``SubstrateCoordinatorSession.commit_cas``.
        """
        return None

    # --- read -----------------------------------------------------------------

    def read(self, artifact_ref: str) -> tuple[bytes, SubstrateToken]:
        """Return ``(bytes, token)`` for one row from a single ``SELECT``.

        Raises :class:`KeyError` if the row is absent and
        :class:`~ccs.core.exceptions.CoherenceError` if the stored version is an
        unusable sentinel (fail-closed — a bogus version must never seed a CAS
        comparand).
        """
        row = self._fetch_row(artifact_ref)
        if row is None:
            raise KeyError(f"no row {artifact_ref!r} in {self._table}")
        value, version = row
        return bytes(value), str(self._require_valid_version(version))

    # --- compare-and-set (the Protocol write leg) -----------------------------

    def cas_write(
        self,
        artifact_ref: str,
        *,
        expected_token: SubstrateToken,
        new_bytes: bytes,
    ) -> CasWriteResult:
        """Conditionally write ``new_bytes`` iff the row is still at ``expected_token``.

        Returns a typed outcome: :class:`~ccs.adapters.substrate.CasWritten` (the
        new token from ``RETURNING``), :class:`~ccs.adapters.substrate.CasConflict`
        (``rowcount == 0`` — nothing landed; ``current_token`` is the version a
        read after the refusal found, or ``None`` when that read finds no row, an
        unusable version or the expected one, or fails), or
        :class:`~ccs.adapters.substrate.CasUnknown` (a driver operational failure
        mid-write — the write may or may not be durable; reconcile via
        :meth:`reconcile_after_unknown`).
        """
        expected_version = self._parse_token(expected_token)
        conn = self._conn()
        params = (new_bytes, artifact_ref, expected_version)
        try:
            with conn.cursor() as cur:
                cur.execute(self._update_sql, params)
                row, rowcount = cur.fetchone(), cur.rowcount
            conn.commit()
        except Exception as exc:  # noqa: BLE001 - re-raised typed/scrubbed below
            self._rollback_quietly(conn)
            if _is_unconfirmed_error(exc):
                self._discard_owned_connection()
                return CasUnknown()
            raise self._scrubbed(exc, "cas_write") from None
        if rowcount == 0:
            return CasConflict(current_token=self._token_after_refusal(artifact_ref, expected_version))
        if row is None:
            return CasConflict()
        (new_version,) = row
        return CasWritten(token=str(self._require_valid_version(new_version)))

    def write(
        self,
        artifact_ref: str,
        *,
        expected_token: SubstrateToken,
        new_bytes: bytes,
    ) -> SubstrateToken:
        """The caller-facing write: like :meth:`cas_write`, but a conflict/unknown
        surfaces as the shipped typed exceptions instead of a return value.

        Returns the new token on a win. Raises
        :class:`~ccs.core.exceptions.CasVersionConflict` (carrying the artifact
        id, expected, and current version) on a conflict — the same typed
        retryable conflict the other adapters speak — and
        :class:`~ccs.core.exceptions.CommitUnconfirmed` on an unconfirmed write.
        """
        expected_version = self._parse_token(expected_token)
        outcome = self.cas_write(artifact_ref, expected_token=expected_token, new_bytes=new_bytes)
        if isinstance(outcome, CasWritten):
            return outcome.token
        if isinstance(outcome, CasConflict):
            raise CasVersionConflict(
                artifact_ref, expected_version, self._current_version_or_wedged(artifact_ref)
            )
        raise CommitUnconfirmed(
            f"Postgres CAS on {self._table} could not be confirmed; the write may or "
            "may not have landed — reconcile by re-reading before retrying."
        )

    # --- unknown reconciliation (token-identity authority) --------------------

    def reconcile_after_unknown(
        self,
        artifact_ref: str,
        *,
        expected_token: SubstrateToken,
        intended_hash: str,
    ) -> ReconcileDecision:
        """Decide what an unconfirmed write should do, by re-reading the row.

        Converges ONLY when the re-read shows the version advanced to exactly
        ``expected + 1`` AND the bytes hash to ``intended_hash`` — the version is
        the authority (a write-counting receipt), the hash is corroboration that
        never converges on its own (byte-identical concurrent writers and no-op
        rewrites would lie). An absent row or an unusable sentinel version holds
        (never a match against ``sha256(b"")``); anything else re-derives.
        """
        expected_version = self._parse_token(expected_token)
        row = self._fetch_row(artifact_ref)
        if row is None:
            return ReconcileDecision(ReconcileVerdict.HOLD, None, None)
        value, version = row
        if not _is_usable_version(version):
            return ReconcileDecision(ReconcileVerdict.HOLD, None, None)
        observed = bytes(value)
        token = str(version)
        landed_as_mine = version == expected_version + 1 and _sha256_hex(observed) == intended_hash
        verdict = ReconcileVerdict.CONVERGE if landed_as_mine else ReconcileVerdict.RE_DERIVE
        return ReconcileDecision(verdict, observed, token)

    # --- provisioning (convenience) -------------------------------------------

    def provisioning_sql(
        self, *, role: str = "coherence_writer", connection_limit: int = 16
    ) -> ProvisioningSql:
        """The version-guard DDL + least-privilege grants for this row's table."""
        return provisioning_sql(self._table, role=role, connection_limit=connection_limit)

    # --- internals ------------------------------------------------------------

    def _conn(self) -> PgConnection:
        if self._connection is None:
            psycopg = _require_psycopg()
            try:
                self._connection = psycopg.connect(self._dsn, **self._connect_kwargs())
            except Exception as exc:  # noqa: BLE001 - a DSN error carries the password
                raise self._scrubbed(exc, "connect") from None
        return self._connection

    def _connect_kwargs(self) -> dict[str, object]:
        """Bounded-wait connect args (connect + per-statement timeout). Passed as
        libpq conninfo kwargs, so they take precedence over the DSN string."""
        kwargs: dict[str, object] = {}
        if self._connect_timeout is not None:
            kwargs["connect_timeout"] = self._connect_timeout
        if self._statement_timeout_ms is not None:
            kwargs["options"] = f"-c statement_timeout={self._statement_timeout_ms}"
        return kwargs

    def _discard_owned_connection(self) -> None:
        """Drop a self-owned connection poisoned by an operational error so the
        next :meth:`_conn` reconnects. Without this, one DB blip (restart, network
        drop) leaves every later read/write hitting the same dead socket — the
        documented 're-read then retry' recovery could never succeed. A
        caller-injected connection (no dsn) is caller-owned and never dropped."""
        if self._dsn is not None:
            self._connection = None

    def _fetch_row(self, artifact_ref: str) -> tuple | None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(self._select_sql, (artifact_ref,))
                row = cur.fetchone()
            conn.rollback()  # end the read-only snapshot; nothing to commit
        except Exception as exc:  # noqa: BLE001 - re-raised scrubbed
            self._rollback_quietly(conn)
            if _is_unconfirmed_error(exc):
                self._discard_owned_connection()
            raise self._scrubbed(exc, "read") from None
        return row

    def _token_after_refusal(self, artifact_ref: str, expected_version: int) -> SubstrateToken | None:
        """The row's token read after a refused UPDATE, as a report only.

        The refusal already committed, so a failed read reports nothing rather
        than turning a confirmed conflict into an unknown outcome or an error. A
        read-back equal to the expected version (a re-inserted row, a disabled
        trigger) is not reported: it would invite re-driving the stale comparand.
        """
        try:
            row = self._fetch_row(artifact_ref)
        except CoherenceError as exc:  # already scrubbed; an owned conn is dropped
            # The scrubbed message names the driver error and the table, never the
            # driver text; no exc_info, since its context still holds that text.
            logger.debug("CoherentRow could not read the row after a refused write: %s", exc)
            return None
        if row is None:
            return None
        _value, version = row
        if not _is_usable_version(version) or version == expected_version:
            return None
        return str(int(version))  # rendered exactly as read() renders it

    def _current_version_or_wedged(self, artifact_ref: str) -> int:
        row = self._fetch_row(artifact_ref)
        if row is None:
            raise ViewWedged(
                f"row {artifact_ref!r} in {self._table} vanished during a CAS conflict; "
                "reacquire and re-decide (a delete is itself an update)."
            )
        return self._require_valid_version(row[1])

    def _scrubbed(self, exc: BaseException, op: str) -> CoherenceError:
        """A typed error that never echoes the driver message (it may carry the DSN
        password). Only the exception TYPE and the table name are surfaced."""
        return CoherenceError(
            f"Postgres substrate {op} on {self._table} failed with "
            f"{type(exc).__name__}; details suppressed to avoid leaking DSN credentials"
        )

    @staticmethod
    def _rollback_quietly(conn: PgConnection) -> None:
        try:
            conn.rollback()
        except Exception:  # noqa: BLE001 - best effort; a rollback failure must not mask the cause
            logger.debug("CoherentRow rollback after error failed", exc_info=True)

    @staticmethod
    def _parse_token(token: SubstrateToken) -> int:
        """Parse a token to its integer version, failing closed on a sentinel."""
        try:
            version = int(token)
        except (TypeError, ValueError):
            raise CoherenceError("unusable substrate token (not an integer version)") from None
        return CoherentRow._require_valid_version(version)

    @staticmethod
    def _require_valid_version(version: object) -> int:
        """Return a positive int version, or fail closed on a sentinel/absent one."""
        if not _is_usable_version(version):
            raise CoherenceError(
                "unconfirmed substrate version (absent/zero/sentinel); it may not "
                "seed a CAS comparand — hold and reacquire"
            )
        return int(version)  # type: ignore[arg-type]


def _is_usable_version(version: object) -> bool:
    """True iff ``version`` is a positive integer (not a bool, sentinel, or None)."""
    return isinstance(version, int) and not isinstance(version, bool) and version > 0


if TYPE_CHECKING:
    # Structural conformance: CoherentRow must satisfy the CoherenceSubstrate
    # Protocol (descriptor + read + cas_write). Checked statically, never by
    # inheritance — mirrors the registry-contract discipline.
    _protocol_check: type[CoherenceSubstrate] = CoherentRow
