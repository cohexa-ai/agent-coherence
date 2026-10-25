# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Replay CLI resolve mode — the wired ``read_at_version`` consumer (Unit 6 / R5b).

Requirement trace: **R5b** — one wired consumer (``agent-coherence-replay``
resolve mode), restart-survival proof. This is the test that makes the whole
feature's value story real: an embedder writes versions with retention on, the
process exits, and a SEPARATE read-only resolver recovers the exact bytes.

Coverage map (plan §Unit 6 Test scenarios):

- The consumer proof (happy path): cross-process restart → exact bytes, exit 0.
- Content-safe default: metadata only (no raw bytes) without ``--include-content``;
  ``--include-content`` yields base64 + ``content_encoding`` for bytes;
  ``--output-file`` writes raw bytes at ``0o600``.
- Per-rejection mapping: each of the 6 reasons → distinct exit code + JSON
  ``reason`` (``not_retained`` vs ``unknown_artifact`` distinguishable).
- Error paths: missing db (NO file created), corrupt/non-sqlite, v1-schema (no
  migration), needs-recovery (fault-injected), SQLITE_BUSY (fault-injected) —
  each a distinct typed error, non-zero exit, no partial output.
- name→id lookup: by workspace path works; a miss → typed error naming the lookup.
- Backward-compat: the existing ``agent-coherence-replay <session_dir>``
  invocation behaves byte-identically; ``--help`` documents the new mode.

Reason matching is ALWAYS ``== CONSTANT`` against the imported wire-stable
constants (the typed-signal-not-substring house rule). Runs against a bare
``[dev]`` import surface — no test here requires langgraph/etc.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sqlite3
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.cli.coherence_replay import build_parser, main
from ccs.coordinator.retention import RetentionPolicy
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    CURRENT_VERSION_REASON,
    EPOCH_MISMATCH_REASON,
    FUTURE_VERSION_REASON,
    NOT_RETAINED_REASON,
    RETENTION_OFF_REASON,
    UNKNOWN_ARTIFACT_REASON,
)
from ccs.core.types import Artifact, VersionedContent, VersionedReadRejection
from ccs.replay import (
    ResolverCorruptDatabaseError,
    ResolverInstanceMismatchError,
    ResolverMissingDatabaseError,
    ResolverNeedsRecoveryError,
    ResolverRequest,
    ResolverSchemaVersionError,
    ResolverUnknownArtifactPathError,
    resolve_version,
)

_K8 = RetentionPolicy(max_versions=8)


# ---------------------------------------------------------------------------
# Store builders — write with one registry, CLOSE (= process exit), then the
# resolver reopens the SAME path read-only. This is the restart boundary.
# ---------------------------------------------------------------------------


def _write_store(
    db_path: Path,
    bodies: tuple[str | bytes, ...],
    *,
    policy: RetentionPolicy | None = _K8,
    retain_versions: bool = True,
) -> Artifact:
    """Populate ``db_path`` then CLOSE the writer (simulating process exit).

    Registers v1=bodies[0] and pessimistic/CAS-commits bodies[1:] as v2.. The
    writer is fully closed before returning so the resolver opens a cold store
    over the same file — the cross-process restart the proof depends on.
    """
    writer = SqliteArtifactRegistry(
        db_path, retain_versions=retain_versions, retention_policy=policy
    )
    try:
        # register_artifact takes str content; a bytes v1 is seeded as a string
        # placeholder (tests that assert on bytes always do so for v2+ via CAS).
        first = bodies[0]
        art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
        writer.register_artifact(
            art, content=first if isinstance(first, str) else "seed"
        )
        agent = uuid4()
        version = 1
        for body in bodies[1:]:
            version += 1
            if isinstance(body, bytes):
                writer.set_agent_state(art.id, agent, _SHARED, tick=version)
                writer.commit_cas(
                    art.id, agent, expected_version=version - 1,
                    content_hash="h", content=body, tick=version,
                )
            else:
                nx = Artifact(id=art.id, name="plan.md", version=version, content_hash="h")
                writer.set_artifact_and_content(art.id, nx, body)
        return art
    finally:
        writer.close()


# Imported lazily-ish to keep the builder readable; MESIState is core, cheap.
from ccs.core.states import MESIState as _MESI  # noqa: E402

_SHARED = _MESI.SHARED


def _resolve_json(args: list[str]) -> tuple[int, dict | None]:
    """Run ``main(['resolve', ...])`` capturing stdout; parse the JSON object.

    Returns ``(exit_code, parsed_json_or_None)``. ``--json`` must be in ``args``
    for the second element to be populated.
    """
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = main(["resolve", *args])
    text = buf.getvalue().strip()
    obj = json.loads(text) if text else None
    return rc, obj


# ===========================================================================
# THE CONSUMER PROOF — restart-survival: write, exit, resolve exact bytes
# ===========================================================================


class TestConsumerProofRestartSurvival:
    """The R5b proof: a closed-then-reopened store yields the exact retained
    bytes at version k. This is the test that proves the feature's value."""

    def test_resolve_exact_bytes_after_writer_closed(self, tmp_path: Path) -> None:
        db = tmp_path / ".coherence" / "state.db"
        art = _write_store(db, ("body-v1", "body-v2", "body-v3"))  # current = 3
        # Writer is closed (process "exited"). Resolve v2 from the cold store.
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--include-content", "--json"]
        )
        assert rc == 0
        assert obj["kind"] == "resolved"
        assert obj["version"] == 2
        assert obj["content"] == "body-v2"  # EXACT bytes survived the restart
        assert obj["artifact_id"] == str(art.id)

    def test_resolve_by_raw_uuid_after_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        art = _write_store(db, ("body-v1", "body-v2", "body-v3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", str(art.id), "--version", "1",
             "--include-content", "--json"]
        )
        assert rc == 0
        assert obj["content"] == "body-v1"

    def test_resolve_bytes_body_round_trips_base64(self, tmp_path: Path) -> None:
        # A BLOB-stored bytes version survives and is base64 in --include-content.
        db = tmp_path / "state.db"
        art = _write_store(db, ("seed", b"\x00\x01bytes-v2", "body-v3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--include-content", "--json"]
        )
        import base64

        assert rc == 0
        assert obj["content_encoding"] == "base64"
        assert base64.b64decode(obj["content"]) == b"\x00\x01bytes-v2"


# ===========================================================================
# CONTENT-SAFE DEFAULT — metadata only unless explicitly asked
# ===========================================================================


class TestContentSafeDefault:
    def test_default_output_is_metadata_only_no_bytes(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("secret-v1", "secret-v2", "secret-v3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 0
        # The DEFAULT carries metadata only — no raw bytes anywhere.
        assert "content" not in obj
        assert "content_encoding" not in obj
        assert set(obj) >= {
            "version", "coordinator_epoch", "captured_at",
            "content_hash", "content_length",
        }
        assert obj["content_length"] == len("secret-v2")
        # And the secret literal must not appear in the serialized output.
        assert "secret-v2" not in json.dumps(obj)

    def test_default_human_output_omits_body(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("secret-v1", "secret-v2", "secret-v3"))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["resolve", "--db", str(db), "--artifact", "plan.md", "--version", "2"])
        out = buf.getvalue()
        assert rc == 0
        assert "content_hash" in out and "content_length" in out
        assert "secret-v2" not in out  # body never printed without the flag

    def test_content_hash_is_sha256_over_bytes(self, tmp_path: Path) -> None:
        import hashlib

        db = tmp_path / "state.db"
        _write_store(db, ("c1", "hash-me-v2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 0
        assert obj["content_hash"] == hashlib.sha256(b"hash-me-v2").hexdigest()

    def test_include_content_str_is_utf8_as_is(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "plain-v2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--include-content", "--json"]
        )
        assert rc == 0
        assert obj["content_encoding"] == "utf-8"
        assert obj["content"] == "plain-v2"

    def test_output_file_writes_raw_bytes_at_0600(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("seed", b"\x00\x01raw-v2", "c3"))
        out = tmp_path / "extracted.bin"
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--output-file", str(out), "--json"]
        )
        assert rc == 0
        # Raw bytes on disk, no base64; metadata (incl. output_file) on stdout.
        assert out.read_bytes() == b"\x00\x01raw-v2"
        assert (out.stat().st_mode & 0o777) == 0o600
        assert obj["output_file"] == str(out)
        # Default (no --include-content) still keeps bytes out of stdout.
        assert "content" not in obj

    def test_output_file_str_body_written_utf8(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "str-body-v2", "c3"))
        out = tmp_path / "out.txt"
        rc, _ = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--output-file", str(out), "--json"]
        )
        assert rc == 0
        assert out.read_bytes() == b"str-body-v2"


# ===========================================================================
# PER-REJECTION MAPPING — every reason a distinct exit code + JSON reason
# ===========================================================================


class TestPerRejectionMapping:
    """Each of the six read_at_version reasons surfaces distinguishably: a
    distinct exit code AND a wire-stable JSON ``reason`` (no prose parsing)."""

    def test_current_version_exit_5(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))  # current = 3
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "3", "--json"]
        )
        assert rc == 5
        assert obj["reason"] == CURRENT_VERSION_REASON

    def test_not_retained_exit_6(self, tmp_path: Path) -> None:
        # K=2 over three commits drops v1 → reading v1 is not_retained.
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"), policy=RetentionPolicy(max_versions=2))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "1", "--json"]
        )
        assert rc == 6
        assert obj["reason"] == NOT_RETAINED_REASON

    def test_unknown_artifact_exit_7(self, tmp_path: Path) -> None:
        # A raw UUID unknown to the registry flows to read_at_version →
        # unknown_artifact (NOT the by-path lookup error — that is exit 13).
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", str(uuid4()), "--version", "1", "--json"]
        )
        assert rc == 7
        assert obj["reason"] == UNKNOWN_ARTIFACT_REASON

    def test_retention_off_exit_8(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"), retain_versions=False, policy=None)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 8
        assert obj["reason"] == RETENTION_OFF_REASON

    def test_epoch_mismatch_exit_9(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--expected-epoch", "not-the-epoch", "--json"]
        )
        assert rc == 9
        assert obj["reason"] == EPOCH_MISMATCH_REASON

    def test_future_version_exit_10(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))  # current = 3
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "9", "--json"]
        )
        assert rc == 10
        assert obj["reason"] == FUTURE_VERSION_REASON

    def test_not_retained_and_unknown_artifact_are_distinct(self, tmp_path: Path) -> None:
        # The plan calls this out explicitly: a script must tell not_retained from
        # unknown_artifact WITHOUT parsing prose — distinct exit codes + reasons.
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"), policy=RetentionPolicy(max_versions=2))
        rc_nr, obj_nr = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "1", "--json"]
        )
        rc_uk, obj_uk = _resolve_json(
            ["--db", str(db), "--artifact", str(uuid4()), "--version", "1", "--json"]
        )
        assert (rc_nr, obj_nr["reason"]) == (6, NOT_RETAINED_REASON)
        assert (rc_uk, obj_uk["reason"]) == (7, UNKNOWN_ARTIFACT_REASON)
        assert rc_nr != rc_uk

    def test_rejection_carries_no_body_material(self, tmp_path: Path) -> None:
        # A rejection envelope must never leak content/hash/body fields.
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "3", "--json"]
        )
        assert rc == 5
        for forbidden in ("content", "content_hash", "content_length", "content_encoding"):
            assert forbidden not in obj

    def test_rejection_human_output_names_reason(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"), policy=RetentionPolicy(max_versions=2))
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["resolve", "--db", str(db), "--artifact", "plan.md", "--version", "1"])
        out = buf.getvalue()
        assert rc == 6
        assert NOT_RETAINED_REASON in out


# ===========================================================================
# ERROR PATHS — open/lookup failures: distinct typed errors, no partial output
# ===========================================================================


class TestErrorPaths:
    def test_missing_db_exit_11_no_file_created(self, tmp_path: Path) -> None:
        absent = tmp_path / "nope" / "state.db"
        rc, obj = _resolve_json(
            ["--db", str(absent), "--artifact", "plan.md", "--version", "1", "--json"]
        )
        assert rc == 11
        assert obj["reason"] == "db_missing"
        # The defining assertion: NO db materialized at the absent path.
        assert not absent.exists()
        assert not absent.parent.exists()

    def test_corrupt_non_sqlite_file_exit_12_db_corrupt(self, tmp_path: Path) -> None:
        bad = tmp_path / "garbage.db"
        bad.write_bytes(b"this is definitely not a sqlite database header\x00\x01")
        rc, obj = _resolve_json(
            ["--db", str(bad), "--artifact", "plan.md", "--version", "1", "--json"]
        )
        assert rc == 12
        assert obj["reason"] == "db_corrupt"

    def test_v1_schema_db_exit_12_no_migration(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _build_raw_v1_db(db)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "1", "--json"]
        )
        assert rc == 12
        assert obj["reason"] == "schema_version_mismatch"
        # No migration ran: user_version is STILL 1 after the read-only resolve.
        conn = sqlite3.connect(str(db))
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
        finally:
            conn.close()

    def test_needs_recovery_exit_12_fault_injected(self, tmp_path: Path, monkeypatch) -> None:
        # A real post-crash hot-WAL state cannot be created deterministically in
        # pytest; inject SQLITE_READONLY_RECOVERY on the ro connection's first
        # user_version read (the Unit 3 fault-injection approach).
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        from ccs.coordinator import sqlite_registry as sr

        class _RecoveryProxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if "user_version" in sql:
                    raise sqlite3.OperationalError(
                        "attempt to write a readonly database (SQLITE_READONLY_RECOVERY)"
                    )
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _RecoveryProxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 12
        assert obj["reason"] == "needs_recovery"

    def test_sqlite_busy_exit_12_db_busy_no_partial_output(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # Under WAL a read-only resolver cannot deterministically hit real
        # reader-side contention; inject SQLITE_BUSY ("database is locked") on the
        # ro connection's first read (per the feasibility review).
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        from ccs.coordinator import sqlite_registry as sr

        class _BusyProxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if "user_version" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _BusyProxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["resolve", "--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"])
        out = buf.getvalue().strip()
        obj = json.loads(out)
        assert rc == 12
        assert obj["reason"] == "db_busy"
        # No PARTIAL content output — the single object is an error envelope, not
        # a resolved/rejected payload, and carries no body material.
        assert obj["kind"] == "error"
        assert "content" not in obj

    def test_version_below_one_is_usage_error_exit_4(self, tmp_path: Path) -> None:
        # --version 0 is caller misuse (ValueError from the service), NOT a
        # resolver reason — surfaced as exit 4.
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "0", "--json"]
        )
        assert rc == 4
        assert obj["reason"] == "usage_error"

    def test_query_time_busy_maps_to_db_busy(self, tmp_path: Path, monkeypatch) -> None:
        # SQLITE_BUSY surfacing at QUERY time (after a clean read-only open)
        # must classify identically to the open-time path: db_busy, exit 12.
        # Inject on the artifact_versions SELECT — past every open-time read.
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        from ccs.coordinator import sqlite_registry as sr

        class _QueryBusyProxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if "artifact_versions" in sql:
                    raise sqlite3.OperationalError("database is locked")
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _QueryBusyProxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 12
        assert obj["reason"] == "db_busy"

    def test_query_time_recovery_maps_to_needs_recovery(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The wal_recovery signal at query time folds into needs_recovery —
        # same wire outcome as the open-time classification.
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        from ccs.coordinator import sqlite_registry as sr

        class _QueryRecoveryProxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if "artifact_versions" in sql:
                    raise sqlite3.OperationalError(
                        "attempt to write a readonly database (SQLITE_READONLY_RECOVERY)"
                    )
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _QueryRecoveryProxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 12
        assert obj["reason"] == "needs_recovery"

    def test_open_time_unreadable_signal_maps_to_needs_recovery(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The catch-all "unreadable" signal (an OperationalError that is
        # neither busy nor a recognized recovery state) keeps today's wire
        # outcome: needs_recovery, exit 12 (same operator remedy).
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        from ccs.coordinator import sqlite_registry as sr

        class _UnreadableProxy:
            def __init__(self, real):
                self._real = real

            def execute(self, sql, *a, **k):
                if "user_version" in sql:
                    raise sqlite3.OperationalError("no such table: sqlite_master_oops")
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                return _UnreadableProxy(conn)
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 12
        assert obj["reason"] == "needs_recovery"

    def test_no_output_file_written_on_rejection(self, tmp_path: Path) -> None:
        # A rejection must not produce a partial output file even if --output-file
        # was requested (no bytes to write).
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))  # current = 3
        out = tmp_path / "should-not-exist.bin"
        rc, _ = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "3",
             "--output-file", str(out), "--json"]
        )
        assert rc == 5  # current_version rejection
        assert not out.exists()


# ===========================================================================
# --db NAMES A FILE — never URI syntax (#418)
# ===========================================================================


class TestDbPathIsFilenameBytes:
    """tmp_path never emits '#', '?' or '%', so these names are the only way a
    test here can see the resolver answer from a file other than the one named."""

    @pytest.mark.parametrize("dirname", ["proj#2", "proj?x", "p%41q"])
    def test_a_reserved_character_in_the_db_path_resolves_the_named_store(
        self, tmp_path: Path, dirname: str
    ) -> None:
        # Unencoded, '#' and '?' reported a current store as schema v0 and left
        # a stray file at the cut-short name; '%41' reported it as db_corrupt.
        db = tmp_path / dirname / ".coherence" / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        before = sorted(p.name for p in tmp_path.iterdir())

        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--include-content", "--json"]
        )

        assert rc == 0
        assert obj["content"] == "c2"
        assert sorted(p.name for p in tmp_path.iterdir()) == before

    def test_a_copy_beside_the_live_store_answers_for_the_copy(
        self, tmp_path: Path
    ) -> None:
        # A forensic resolve against a saved copy returned the live store's
        # bytes: 'state.db#before-upgrade' was cut to the 'state.db' beside it.
        live = tmp_path / ".coherence" / "state.db"
        _write_store(live, ("live-v1", "live-v2", "live-v3"))
        other = tmp_path / "other" / "state.db"
        _write_store(other, ("copy-v1", "copy-v2", "copy-v3"))
        copy = live.with_name("state.db#before-upgrade")
        shutil.copyfile(other, copy)

        rc, obj = _resolve_json(
            ["--db", str(copy), "--artifact", "plan.md", "--version", "2",
             "--include-content", "--json"]
        )

        assert rc == 0
        assert obj["content"] == "copy-v2"

    def test_a_directory_at_db_is_needs_recovery_not_db_corrupt(
        self, tmp_path: Path
    ) -> None:
        # A failed connect now takes the registry's read-only classification,
        # like every later read: a store the open cannot read is needs_recovery.
        db = tmp_path / "state.db"
        db.mkdir()

        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "1", "--json"]
        )

        assert rc == 12
        assert obj["reason"] == "needs_recovery"

    def test_a_store_deleted_after_the_existence_check_is_db_missing(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The resolver and the registry each check that the store exists, then
        # connect; a store deleted in between failed the connect with a raw
        # error the resolver reported as db_corrupt (exit 12).
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        from ccs.coordinator import sqlite_registry as sr

        real_connect = sqlite3.connect

        def connect_after_delete(*a, **k):
            if a and "mode=ro" in str(a[0]):
                for path in tmp_path.glob("state.db*"):
                    path.unlink()
            return real_connect(*a, **k)

        monkeypatch.setattr(sr.sqlite3, "connect", connect_after_delete)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )

        assert rc == 11
        assert obj["reason"] == "db_missing"


# ===========================================================================
# NAME → ID LOOKUP — by workspace path, and a typed miss
# ===========================================================================


class TestNameToIdLookup:
    def test_resolve_by_workspace_path_works(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--include-content", "--json"]
        )
        assert rc == 0
        assert obj["content"] == "c2"

    def test_unknown_path_exit_13_names_the_lookup(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "docs/never-observed.md", "--version", "1", "--json"]
        )
        assert rc == 13
        assert obj["reason"] == "unknown_artifact_path"
        # The message names the failed path lookup (not a raw stack trace).
        assert "docs/never-observed.md" in obj["message"]

    def test_unknown_path_human_no_traceback(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        buf = io.StringIO()
        err = io.StringIO()
        import contextlib

        with redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = main(["resolve", "--db", str(db), "--artifact", "no/such/path.md", "--version", "1"])
        assert rc == 13
        # A clean message, never a Python traceback.
        assert "Traceback" not in err.getvalue()
        assert "no/such/path.md" in err.getvalue()


# ===========================================================================
# INSTANCE-ID CROSS-CHECK — identity guard before serving any bytes
# ===========================================================================


class TestInstanceIdCrossCheck:
    def test_matching_instance_id_serves(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        # Read the store's persisted instance_id via a read-only registry.
        with SqliteArtifactRegistry(db, read_only=True) as ro:
            inst = ro.instance_id
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--instance-id", inst, "--include-content", "--json"]
        )
        assert rc == 0
        assert obj["content"] == "c2"

    def test_mismatched_instance_id_exit_11(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--instance-id", "wrong-instance", "--json"]
        )
        assert rc == 11
        assert obj["reason"] == "instance_id_mismatch"
        # Identity guard fires BEFORE any bytes are served — no content leaked.
        assert "content" not in obj


# ===========================================================================
# RESOLVER LIBRARY SURFACE — resolve_version returns typed results / raises
# typed errors (the CLI is a thin shell over this; pin the library too)
# ===========================================================================


class TestResolverLibrarySurface:
    def test_resolve_version_returns_versioned_content(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        art = _write_store(db, ("c1", "c2", "c3"))
        out = resolve_version(
            ResolverRequest(db_path=db, selector="plan.md", version=2)
        )
        assert isinstance(out, VersionedContent)
        assert out.content == "c2"
        assert out.artifact_id == art.id

    def test_resolve_version_returns_rejection(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        out = resolve_version(
            ResolverRequest(db_path=db, selector="plan.md", version=3)
        )
        assert isinstance(out, VersionedReadRejection)
        assert out.reason == CURRENT_VERSION_REASON

    def test_resolve_version_missing_db_raises_and_creates_nothing(
        self, tmp_path: Path
    ) -> None:
        absent = tmp_path / "void" / "state.db"
        with pytest.raises(ResolverMissingDatabaseError):
            resolve_version(ResolverRequest(db_path=absent, selector="plan.md", version=1))
        assert not absent.exists()

    def test_resolve_version_v1_db_raises_schema_error(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _build_raw_v1_db(db)
        with pytest.raises(ResolverSchemaVersionError):
            resolve_version(ResolverRequest(db_path=db, selector="plan.md", version=1))

    def test_resolve_version_corrupt_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.db"
        bad.write_bytes(b"not-a-db" * 10)
        with pytest.raises(ResolverCorruptDatabaseError):
            resolve_version(ResolverRequest(db_path=bad, selector="plan.md", version=1))

    def test_resolve_version_unknown_path_raises_named(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        with pytest.raises(ResolverUnknownArtifactPathError, match="missing.md"):
            resolve_version(ResolverRequest(db_path=db, selector="missing.md", version=1))

    def test_resolve_version_instance_mismatch_raises(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        with pytest.raises(ResolverInstanceMismatchError):
            resolve_version(
                ResolverRequest(
                    db_path=db, selector="plan.md", version=2,
                    expected_instance_id="nope",
                )
            )

    def test_resolve_version_below_one_raises_value_error(self, tmp_path: Path) -> None:
        # The LIBRARY contract (not just the CLI's exit-4 translation): a sub-1
        # version propagates the service's ValueError out of resolve_version —
        # caller misuse, never a typed rejection and never a ResolverError.
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        for bad_version in (0, -1):
            with pytest.raises(ValueError, match="version must be >= 1"):
                resolve_version(
                    ResolverRequest(db_path=db, selector="plan.md", version=bad_version)
                )


# ===========================================================================
# NO LEAKED CONNECTION — every typed-raise path closes the read-only handle
# ===========================================================================


class _ConnectionCloseTracker:
    """Wrap ``sqlite3.connect`` for ``mode=ro`` URIs and track close() calls."""

    def __init__(self, monkeypatch, *, fault_sql: str | None = None,
                 fault_message: str | None = None) -> None:
        self.opened: list[object] = []
        self.closed: list[object] = []
        tracker = self

        class _Proxy:
            def __init__(self, real):
                self._real = real

            def close(self):
                tracker.closed.append(self)
                return self._real.close()

            def execute(self, sql, *a, **k):
                if fault_sql is not None and fault_sql in sql:
                    raise sqlite3.OperationalError(fault_message)
                return self._real.execute(sql, *a, **k)

            def __getattr__(self, n):
                return getattr(self._real, n)

        from ccs.coordinator import sqlite_registry as sr

        real_connect = sqlite3.connect

        def fake_connect(*a, **k):
            conn = real_connect(*a, **k)
            if a and "mode=ro" in str(a[0]):
                proxy = _Proxy(conn)
                tracker.opened.append(proxy)
                return proxy
            return conn

        monkeypatch.setattr(sr.sqlite3, "connect", fake_connect)

    def assert_all_closed(self) -> None:
        assert self.opened, "the fault never exercised a read-only open"
        assert len(self.closed) == len(self.opened), (
            f"{len(self.opened) - len(self.closed)} read-only sqlite "
            f"connection(s) leaked on the typed-raise path"
        )


class TestNoLeakedConnectionOnTypedRaise:
    """A typed open/lookup failure must never orphan the read-only handle —
    the post-connect validation closes the connection before raising."""

    def test_schema_version_mismatch_closes_connection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        db = tmp_path / "state.db"
        _build_raw_v1_db(db)
        tracker = _ConnectionCloseTracker(monkeypatch)
        with pytest.raises(ResolverSchemaVersionError):
            resolve_version(ResolverRequest(db_path=db, selector="plan.md", version=1))
        tracker.assert_all_closed()

    def test_needs_recovery_closes_connection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        tracker = _ConnectionCloseTracker(
            monkeypatch,
            fault_sql="user_version",
            fault_message=(
                "attempt to write a readonly database (SQLITE_READONLY_RECOVERY)"
            ),
        )
        with pytest.raises(ResolverNeedsRecoveryError):
            resolve_version(ResolverRequest(db_path=db, selector="plan.md", version=2))
        tracker.assert_all_closed()

    def test_instance_mismatch_closes_connection(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # The mismatch raises AFTER a successful open; the resolver's finally
        # must close the handle (pinned here so a refactor cannot drop it).
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        tracker = _ConnectionCloseTracker(monkeypatch)
        with pytest.raises(ResolverInstanceMismatchError):
            resolve_version(
                ResolverRequest(
                    db_path=db, selector="plan.md", version=2,
                    expected_instance_id="nope",
                )
            )
        tracker.assert_all_closed()


# ===========================================================================
# CLI CONTRACT HARDENING — argparse exit 2, envelopes, mapping pin, output file
# ===========================================================================


class TestArgparseExit2Contract:
    """argparse usage errors exit 2; with --json among argv a JSON error
    envelope reaches stdout first (stderr keeps the usage prose)."""

    def test_resolve_missing_required_arg_with_json_emits_envelope(self) -> None:
        buf, err = io.StringIO(), io.StringIO()
        with pytest.raises(SystemExit) as excinfo:
            with redirect_stdout(buf), redirect_stderr(err):
                main(["resolve", "--db", "/tmp/x", "--artifact", "a", "--json"])
        assert excinfo.value.code == 2
        obj = json.loads(buf.getvalue().strip())
        assert obj["kind"] == "error"
        assert obj["exit_code"] == 2
        assert obj["reason"] == "argument_error"
        assert "usage:" in err.getvalue()  # stderr prose preserved

    def test_resolve_missing_required_arg_without_json_keeps_stdout_empty(
        self,
    ) -> None:
        buf, err = io.StringIO(), io.StringIO()
        with pytest.raises(SystemExit) as excinfo:
            with redirect_stdout(buf), redirect_stderr(err):
                main(["resolve", "--db", "/tmp/x", "--artifact", "a"])
        assert excinfo.value.code == 2
        assert buf.getvalue() == ""  # no envelope without --json
        assert "usage:" in err.getvalue()

    def test_default_mode_bad_flag_with_json_emits_envelope(self) -> None:
        buf, err = io.StringIO(), io.StringIO()
        with pytest.raises(SystemExit) as excinfo:
            with redirect_stdout(buf), redirect_stderr(err):
                main(["--json", "--definitely-not-a-flag", "/tmp/session"])
        assert excinfo.value.code == 2
        obj = json.loads(buf.getvalue().strip())
        assert obj["exit_code"] == 2
        assert obj["reason"] == "argument_error"

    def test_default_mode_bad_flag_without_json_keeps_stdout_empty(self) -> None:
        buf, err = io.StringIO(), io.StringIO()
        with pytest.raises(SystemExit) as excinfo:
            with redirect_stdout(buf), redirect_stderr(err):
                main(["--definitely-not-a-flag", "/tmp/session"])
        assert excinfo.value.code == 2
        assert buf.getvalue() == ""


class TestResolveTokenDispatch:
    """The bare literal 'resolve' selects the subcommand; a path spelling of a
    session dir literally named resolve routes to the default replay mode."""

    def test_dot_slash_resolve_routes_to_replay_mode(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # A session dir literally named "resolve": invoking it as ./resolve
        # must hit the DEFAULT replay mode (here: exit 3, manifest missing —
        # a replay-mode outcome, not a resolve-mode argparse error).
        session = tmp_path / "resolve"
        session.mkdir()
        monkeypatch.chdir(tmp_path)
        rc = main(["./resolve"])
        assert rc == 3  # replay-mode trace error (manifest missing)

    def test_bare_resolve_token_still_selects_subcommand(self) -> None:
        # The bare word stays the subcommand selector (missing required args
        # → argparse exit 2), regardless of cwd contents.
        with pytest.raises(SystemExit) as excinfo:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                main(["resolve"])
        assert excinfo.value.code == 2


class TestResolveInternalErrorEnvelope:
    """An unexpected exception in resolve mode exits 4 AND emits the JSON error
    envelope under --json (stdout stays self-contained on every exit path)."""

    def test_unexpected_raise_with_json_emits_envelope(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        def boom(request):
            raise RuntimeError("simulated resolver bug")

        monkeypatch.setattr("ccs.replay.resolver.resolve_version", boom)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2", "--json"]
        )
        assert rc == 4
        assert obj["kind"] == "error"
        assert obj["exit_code"] == 4
        assert obj["reason"] == "internal_error"
        assert obj["exception"] == "RuntimeError"

    def test_unexpected_raise_without_json_keeps_stdout_empty(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))

        def boom(request):
            raise RuntimeError("simulated resolver bug")

        monkeypatch.setattr("ccs.replay.resolver.resolve_version", boom)
        rc = main(["resolve", "--db", str(db), "--artifact", "plan.md", "--version", "2"])
        captured = capsys.readouterr()
        assert rc == 4
        assert captured.out == ""
        assert "internal error" in captured.err
        assert "RuntimeError" in captured.err
        assert "Traceback" not in captured.err


class TestRejectionEnvelopeExitCode:
    """The rejection JSON envelope carries the REAL numeric exit code (the
    misleading exit_code: None placeholder is gone)."""

    def test_rejection_envelope_exit_code_matches_process_exit(
        self, tmp_path: Path
    ) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))  # current = 3
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "3", "--json"]
        )
        assert rc == 5
        assert obj["kind"] == "rejected"
        assert obj["exit_code"] == 5  # the real code, not None / absent

    def test_every_rejection_reason_envelope_carries_its_code(
        self, tmp_path: Path
    ) -> None:
        from ccs.cli.coherence_replay import _RESOLVE_REJECTION_EXIT_CODES

        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"), policy=RetentionPolicy(max_versions=2))
        scenarios = {
            CURRENT_VERSION_REASON: ["--artifact", "plan.md", "--version", "3"],
            NOT_RETAINED_REASON: ["--artifact", "plan.md", "--version", "1"],
            UNKNOWN_ARTIFACT_REASON: ["--artifact", str(uuid4()), "--version", "1"],
            EPOCH_MISMATCH_REASON: [
                "--artifact", "plan.md", "--version", "2",
                "--expected-epoch", "not-the-epoch",
            ],
            FUTURE_VERSION_REASON: ["--artifact", "plan.md", "--version", "9"],
        }
        for reason, argv in scenarios.items():
            rc, obj = _resolve_json(["--db", str(db), *argv, "--json"])
            assert obj["reason"] == reason
            assert rc == _RESOLVE_REJECTION_EXIT_CODES[reason]
            assert obj["exit_code"] == rc


class TestRejectionExitCodeMappingPin:
    """The mapping is exhaustively pinned against READ_AT_VERSION_REASONS and
    the rendered --help is generated from it (no hand-drift possible)."""

    def test_mapping_covers_reason_set_exactly(self) -> None:
        from ccs.cli.coherence_replay import _RESOLVE_REJECTION_EXIT_CODES
        from ccs.core.exceptions import READ_AT_VERSION_REASONS

        assert set(_RESOLVE_REJECTION_EXIT_CODES) == set(READ_AT_VERSION_REASONS)
        # Pin the exact reason -> code assignments (the wire contract).
        assert _RESOLVE_REJECTION_EXIT_CODES == {
            CURRENT_VERSION_REASON: 5,
            NOT_RETAINED_REASON: 6,
            UNKNOWN_ARTIFACT_REASON: 7,
            RETENTION_OFF_REASON: 8,
            EPOCH_MISMATCH_REASON: 9,
            FUTURE_VERSION_REASON: 10,
        }

    def test_resolve_epilog_mentions_every_reason_with_its_code(self) -> None:
        from ccs.cli.coherence_replay import (
            _RESOLVE_REJECTION_EXIT_CODES,
            build_resolve_parser,
        )

        help_text = build_resolve_parser().format_help()
        for reason, code in _RESOLVE_REJECTION_EXIT_CODES.items():
            assert f"{code:<3} {reason}" in help_text, (
                f"resolve --help is missing the '{code} {reason}' row"
            )
        # And the table is numerically ordered (0/2/4 precede the 5+ rows).
        positions = [help_text.index(f"{code:<3} {reason}")
                     for reason, code in _RESOLVE_REJECTION_EXIT_CODES.items()]
        assert positions == sorted(positions)
        for shared in ("  0 ", "  2 ", "  4 "):
            assert help_text.index(shared) < min(positions)

    def test_resolve_epilog_documents_exit_2(self) -> None:
        from ccs.cli.coherence_replay import build_resolve_parser

        help_text = build_resolve_parser().format_help()
        assert "argument/usage error" in help_text

    def test_default_epilog_documents_exit_2_collision_and_escape(self) -> None:
        help_text = build_parser().format_help()
        assert "argparse" in help_text  # exit-2 usage-error note
        assert "./resolve" in help_text  # the named-dir escape hatch


class TestOutputFileHardening:
    """_write_output_file: symlink refusal, atomic publish, 0600 preserved."""

    def test_symlink_target_refused_no_write_through(self, tmp_path: Path) -> None:
        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        real_target = tmp_path / "real-destination.bin"
        real_target.write_bytes(b"untouched")
        link = tmp_path / "planted-link.bin"
        link.symlink_to(real_target)
        buf, err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            rc = main([
                "resolve", "--db", str(db), "--artifact", "plan.md",
                "--version", "2", "--output-file", str(link), "--json",
            ])
        assert rc == 4  # usage error (typed config error), not a silent follow
        obj = json.loads(buf.getvalue().strip())
        assert obj["reason"] == "usage_error"
        assert "symlink" in obj["message"]
        # The bytes never reached the link's destination; the link survives.
        assert real_target.read_bytes() == b"untouched"
        assert link.is_symlink()

    def test_mid_write_failure_leaves_prior_content_intact(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import os as os_module

        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        out = tmp_path / "out.bin"
        out.write_bytes(b"prior-content")

        def exploding_replace(src, dst):
            raise OSError("simulated mid-write failure")

        monkeypatch.setattr(os_module, "replace", exploding_replace)
        buf, err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            rc = main([
                "resolve", "--db", str(db), "--artifact", "plan.md",
                "--version", "2", "--output-file", str(out),
            ])
        assert rc == 4
        # Prior content intact — never truncated/partial.
        assert out.read_bytes() == b"prior-content"
        # No leftover temp files in the directory.
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []

    def test_mid_write_failure_fresh_target_never_materializes(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import os as os_module

        db = tmp_path / "state.db"
        _write_store(db, ("c1", "c2", "c3"))
        out = tmp_path / "never-created.bin"

        def exploding_replace(src, dst):
            raise OSError("simulated mid-write failure")

        monkeypatch.setattr(os_module, "replace", exploding_replace)
        buf, err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf), redirect_stderr(err):
            rc = main([
                "resolve", "--db", str(db), "--artifact", "plan.md",
                "--version", "2", "--output-file", str(out),
            ])
        assert rc == 4
        assert not out.exists()  # no partial/empty target

    def test_success_path_still_0600_and_exact_bytes(self, tmp_path: Path) -> None:
        # Unchanged success contract after the atomic-write rework (the
        # original 0600 test lives in TestContentSafeDefault; this re-pins it
        # against the temp+replace implementation alongside an existing file).
        db = tmp_path / "state.db"
        _write_store(db, ("seed", b"\x00\x01raw-v2", "c3"))
        out = tmp_path / "extracted.bin"
        out.write_bytes(b"old")  # pre-existing target gets atomically replaced
        os.chmod(out, 0o644)
        rc, obj = _resolve_json(
            ["--db", str(db), "--artifact", "plan.md", "--version", "2",
             "--output-file", str(out), "--json"]
        )
        assert rc == 0
        assert out.read_bytes() == b"\x00\x01raw-v2"
        assert (out.stat().st_mode & 0o777) == 0o600  # replace swaps to 0600


# ===========================================================================
# BACKWARD-COMPAT — the existing positional invocation is byte-identical
# ===========================================================================


class TestBackwardCompat:
    """Adding the resolve mode must not change the default invariant-replay
    behavior at all. These mirror tests/test_cli_coherence_replay.py."""

    def _clean_session(self, tmp_path: Path) -> Path:
        session = tmp_path / "clean"
        session.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 0,
            "schema_note": "test fixture",
            "adapter_type": "test-fixture",
            "start_tick": 0,
            "end_tick": 10,
            "instance_id": "instance-A",
            "streams": ["state_log"],
            "agents": {},
            "artifacts": {},
        }
        (session / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        entries = [
            {
                "tick": 0, "artifact_id": "art-1", "agent_id": "agent-1",
                "agent_name": "agent-1", "from_state": "INVALID",
                "to_state": "EXCLUSIVE", "trigger": "write", "version": 1,
                "content_hash": "abc", "sequence_number": 1,
                "instance_id": "instance-A", "schema_version": "ccs.state_log.v2",
            },
        ]
        with (session / "state_log.jsonl").open("w", encoding="utf-8") as fh:
            for e in entries:
                fh.write(json.dumps(e) + "\n")
        return session

    def test_bare_positional_clean_trace_exits_zero(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = self._clean_session(tmp_path)
        rc = main([str(session)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "0 CONFIRMED" in out

    def test_bare_positional_missing_dir_exit_3(self, tmp_path: Path) -> None:
        rc = main([str(tmp_path / "does-not-exist")])
        assert rc == 3

    def test_bare_positional_parser_unchanged_shape(self) -> None:
        # The default parser still accepts a single positional + the original
        # flag set, and nothing about resolve leaked into it.
        ns = build_parser().parse_args(["/tmp/session", "--json", "--quiet"])
        assert ns.session_dir == Path("/tmp/session")
        assert ns.json is True and ns.quiet is True
        # No resolve-only attribute exists on the default namespace.
        for resolve_only in ("db", "artifact", "version", "include_content"):
            assert not hasattr(ns, resolve_only)

    def test_help_documents_the_resolve_mode(self) -> None:
        help_text = build_parser().format_help()
        assert "agent-coherence-replay" in help_text
        assert "replay_trace_format.md" in help_text  # original content intact
        assert "resolve" in help_text  # new mode advertised

    def test_resolve_help_documents_content_safety(self) -> None:
        from ccs.cli.coherence_replay import build_resolve_parser

        help_text = build_resolve_parser().format_help()
        assert "--include-content" in help_text
        assert "--output-file" in help_text
        assert "--db" in help_text and "--artifact" in help_text and "--version" in help_text


# ---------------------------------------------------------------------------
# Raw v1 db builder (mirrors tests/test_sqlite_registry.py::_build_raw_v1_db)
# ---------------------------------------------------------------------------


def _build_raw_v1_db(db_path: Path) -> str:
    """Construct a RAW v1 ``state.db`` BY HAND (full-shim variant) for the
    read-only-resolver schema-error path. Returns the artifact id hex.

    The resolver must reject a v1 db with a typed schema error and perform NO
    migration; this builds a complete v1 (fence columns + pending_notices +
    epoch seed) at ``user_version=1`` so the resolver's read-only open hits the
    version mismatch, not a missing-column error.
    """
    art_id, ag_id = uuid4().hex, uuid4().hex
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "CREATE TABLE artifacts (id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
            "version INTEGER NOT NULL, owner_generation INTEGER NOT NULL DEFAULT 0, "
            "content_hash TEXT NOT NULL, size_tokens INTEGER, last_writer_id TEXT, "
            "updated_at REAL NOT NULL)"
        )
        conn.execute("CREATE INDEX idx_artifacts_name ON artifacts(name)")
        conn.execute(
            "CREATE TABLE agent_states (artifact_id TEXT NOT NULL, agent_id TEXT NOT NULL, "
            "state TEXT NOT NULL, transient_state TEXT, transient_tick INTEGER, "
            "granted_at_tick INTEGER, last_reclaim_trigger TEXT, last_reclaim_tick INTEGER, "
            "read_generation INTEGER, PRIMARY KEY (artifact_id, agent_id), "
            "FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE)"
        )
        conn.execute(
            "CREATE TABLE heartbeats (agent_id TEXT PRIMARY KEY, last_tick INTEGER NOT NULL)"
        )
        conn.execute("CREATE TABLE registry_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO registry_meta (key, value) VALUES (?, ?)",
            [("instance_id", "wild-v1"), ("sequence_number", "0"),
             ("coordinator_epoch", "epoch-pre-migration")],
        )
        conn.execute(
            "CREATE TABLE pending_notices (agent_id TEXT NOT NULL, artifact_id TEXT NOT NULL, "
            "preempter_agent_id TEXT NOT NULL, preempted_at_unix_ts REAL NOT NULL, "
            "PRIMARY KEY (agent_id, artifact_id), "
            "FOREIGN KEY (artifact_id) REFERENCES artifacts(id) ON DELETE CASCADE)"
        )
        conn.execute(
            "INSERT INTO artifacts (id, name, version, owner_generation, content_hash, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (art_id, "plan.md", 3, 0, "deadbeef", 1.0),
        )
        conn.execute(
            "INSERT INTO agent_states (artifact_id, agent_id, state) VALUES (?, ?, ?)",
            (art_id, ag_id, "M"),
        )
        conn.execute("PRAGMA user_version = 1")
        conn.execute("COMMIT")
    finally:
        conn.close()
    return art_id
