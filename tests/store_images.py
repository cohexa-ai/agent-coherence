# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Coordinator stores as a crashed writer leaves them, for the read-only open tests.

A writer in this process holds a transaction open while the files are copied.
The copy is the image a SIGKILL leaves: the operating system keeps every write
the writer made, and no lock lives in the file bytes. Nothing holds the copy
open, so these run in the parallel lane; ``tests/test_sqlite_registry.py`` checks
the equivalence once against a real SIGKILLed writer (#419).
"""

from __future__ import annotations

import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from uuid import UUID, uuid4

from ccs.coordinator.retention import RetentionPolicy
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry, _read_only_uri
from ccs.core.types import Artifact

COMMITTED_V1 = "committed-v1"
COMMITTED_V2 = "committed-v2"
UNCOMMITTED_TABLE = "uncommitted_probe"


def sidecar_path(db: Path, suffix: str) -> Path:
    return db.with_name(db.name + suffix)


def kill_writer_mid_flight(script: str, *args: str) -> None:
    """Run ``script`` in a child Python, SIGKILL it once it prints READY, and
    reap it. Signals only the child this call started; one that never says
    READY is killed after 60 s rather than hanging the serial lane."""
    child = subprocess.Popen(
        [sys.executable, "-c", script, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    deadline = threading.Timer(60, child.kill)
    deadline.start()
    try:
        ready = child.stdout.readline()
        assert ready.strip() == b"READY", (ready, child.stderr.read() if child.poll() is not None else b"")
    finally:
        deadline.cancel()
        if child.poll() is None:
            os.kill(child.pid, signal.SIGKILL)
        child.wait(timeout=30)
        child.stdout.close()
        child.stderr.close()


def sqlite_error(message: str, name: str | None) -> sqlite3.OperationalError:
    """An OperationalError carrying the code SQLite would attach to it (none
    for ``None``: an error SQLite did not raise itself)."""
    error = sqlite3.OperationalError(message)
    if name is not None:
        error.sqlite_errorcode = getattr(sqlite3, name)
        error.sqlite_errorname = name
    return error


def _copy(live: Path, image: Path, suffixes: tuple[str, ...]) -> None:
    image.parent.mkdir(parents=True)
    for suffix in suffixes:
        shutil.copy2(sidecar_path(live, suffix), sidecar_path(image, suffix))


def open_crashing_writer(db: Path) -> tuple[SqliteArtifactRegistry, UUID]:
    """Open a writer on the store at ``db`` and leave it mid-transaction.

    It commits ``plan.md`` v1 and v2 with the WAL never checkpointed, then holds
    a transaction large enough to spill its uncommitted pages into the WAL.
    Shared with the real SIGKILLed writer in ``tests/test_sqlite_registry.py``.
    """
    writer = SqliteArtifactRegistry(
        db, retain_versions=True, retention_policy=RetentionPolicy(max_versions=8)
    )
    conn = writer._conn
    conn.execute("PRAGMA wal_autocheckpoint=0")
    art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
    writer.register_artifact(art, content=COMMITTED_V1)
    writer.set_artifact_and_content(
        art.id, Artifact(id=art.id, name="plan.md", version=2, content_hash="h"),
        COMMITTED_V2,
    )
    # A small cache spills the uncommitted pages into the WAL.
    conn.execute("PRAGMA cache_size=16")
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(f"CREATE TABLE {UNCOMMITTED_TABLE} (b BLOB)")
    for _ in range(512):
        conn.execute(f"INSERT INTO {UNCOMMITTED_TABLE} VALUES (randomblob(4096))")
    return writer, art.id


def crashed_wal_image(root: Path) -> tuple[Path, UUID]:
    """A WAL store whose writer stopped mid-transaction (``open_crashing_writer``).

    ``plan.md`` v1 and v2 are committed, and their only copy is in the WAL. An
    uncommitted table follows them in the WAL. Returns the image's
    ``state.db`` and the artifact id.
    """
    live = root / "live" / "state.db"
    SqliteArtifactRegistry(live).close()
    writer, art_id = open_crashing_writer(live)
    try:
        image = root / "store" / "state.db"
        _copy(live, image, ("", "-wal", "-shm"))
        writer._conn.execute("ROLLBACK")
    finally:
        writer.close()
    # Without these, the tests that read the image would pass vacuously.
    assert _artifacts_in_db_file_alone(image) == 0, "a commit reached state.db"
    assert _frames_after_last_commit(sidecar_path(image, "-wal")) > 0, "nothing spilled"
    return image, art_id


def _artifacts_in_db_file_alone(db: Path) -> int:
    conn = sqlite3.connect(_read_only_uri(db) + "&immutable=1", uri=True)
    try:
        return conn.execute("SELECT count(*) FROM artifacts").fetchone()[0]
    finally:
        conn.close()


def _frames_after_last_commit(wal: Path) -> int:
    """WAL frames after the last commit frame (one whose db-size field is set)."""
    data = wal.read_bytes()
    page_size, salt = int.from_bytes(data[8:12], "big"), data[16:24]
    frames = last_commit = 0
    offset = 32
    while offset + 24 + page_size <= len(data) and data[offset + 8:offset + 16] == salt:
        frames += 1
        if int.from_bytes(data[offset + 4:offset + 8], "big"):
            last_commit = frames
        offset += 24 + page_size
    return frames - last_commit


def hot_journal_image(root: Path) -> tuple[Path, UUID]:
    """A rollback-journal store whose writer stopped mid-transaction, leaving a
    hot ``state.db-journal`` (#420). ``plan.md`` v1 is committed."""
    live = root / "live" / "state.db"
    with SqliteArtifactRegistry(
        live, retain_versions=True, retention_policy=RetentionPolicy(max_versions=8)
    ) as writer:
        art = Artifact(id=uuid4(), name="plan.md", version=1, content_hash="h")
        writer.register_artifact(art, content=COMMITTED_V1)
    conn = sqlite3.connect(str(live), isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA cache_size=1")
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(f"CREATE TABLE {UNCOMMITTED_TABLE} (b BLOB)")
        for _ in range(300):
            conn.execute(f"INSERT INTO {UNCOMMITTED_TABLE} VALUES (randomblob(3000))")
        image = root / "store" / "state.db"
        _copy(live, image, ("", "-journal"))
        conn.execute("ROLLBACK")
    finally:
        conn.close()
    return image, art.id
