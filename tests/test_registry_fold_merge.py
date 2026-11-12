# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""One identity per file in the SQLite registry: the in-memory fold index,
names kept exact as keys, the bucket rule, and the start-time merge of file
rows an older build split for one file.

Older-shape stores are built the way an older build left them: the current
schema without the side table that lists key rows, then rows written
directly. Many groups use a tidy-only split (``./plan.md`` beside
``plan.md``), which shares one key under every profile, so those tests mean
the same thing on a disk that folds case and on one that does not.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import pytest

import ccs.adapters.claude_code.coordinator_server as server_module
from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    CoordinatorHTTPServer,
    prepare_fold_index,
    session_to_agent_id,
)
from ccs.adapters.claude_code.fs_profile import FoldProfileResult
from ccs.coordinator.fold_merge import (
    FOLD_MERGE_PREEMPTER_ID,
    FOLD_MERGE_TRIGGER,
    FoldGroup,
    FoldMergeOutcome,
    FoldRow,
    ResolvedFile,
    same_file_groups,
)
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SCHEMA_USER_VERSION, SqliteArtifactRegistry
from ccs.core.exceptions import (
    CommitNotAllowed,
    GiverFenced,
    InvariantViolationError,
    OtherHolder,
    StaleReadGeneration,
)
from ccs.core.path_fold import FoldProfile, fold_key
from ccs.core.states import MESIState
from ccs.core.types import Artifact, ConflictDetail, InvalidationSignal, SessionReadRejection
from ccs.replay import load, run_predicates
from ccs.replay.recorder import record_callbacks
from ccs.validation import validate_log
from tests.test_claude_code_coordinator_server import _Client, _sid

AGENT_A = UUID("a" * 32)
AGENT_B = UUID("b" * 32)
AGENT_C = UUID("c" * 32)
NFC = "data/café.txt"
NFD = "data/café.txt"
NOW = 1_000_000.0
TICK = int(NOW)
SERVER_LOGGER = server_module.logger.name


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Store:
    root: Path
    db: Path


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """A workspace whose store has the shape an older build left: the
    current schema without the side table that lists key rows."""
    root = tmp_path / "workspace"
    (root / ".coherence").mkdir(parents=True, mode=0o700)
    db = root / ".coherence" / "state.db"
    SqliteArtifactRegistry(db).close()
    conn = sqlite3.connect(db, isolation_level=None)
    try:
        conn.execute("DROP TABLE IF EXISTS artifact_name_kinds")
    finally:
        conn.close()
    return Store(root, db)


@pytest.fixture
def raw(store: Store) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(store.db, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=ON")
    yield conn
    conn.close()


@pytest.fixture
def opened(store: Store) -> Iterator[Callable[..., SqliteArtifactRegistry]]:
    """Open the store's writer; every registry opened is closed at the end."""
    registries: list[SqliteArtifactRegistry] = []

    def open_registry(**kwargs) -> SqliteArtifactRegistry:
        registry = SqliteArtifactRegistry(store.db, **kwargs)
        registries.append(registry)
        return registry

    yield open_registry
    for registry in registries:
        registry.close()


def seed_artifact(conn: sqlite3.Connection, name: str, **values) -> UUID:
    artifact_id = uuid4()
    writer = values.get("last_writer")
    conn.execute(
        "INSERT INTO artifacts (id, name, version, owner_generation, content_hash, "
        "size_tokens, last_writer_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            artifact_id.hex, name, values.get("version", 1), values.get("generation", 0),
            values.get("hash", ""), values.get("size_tokens"),
            writer.hex if writer is not None else None, values.get("updated_at", 1.0),
        ),
    )
    return artifact_id


def seed_state(
    conn: sqlite3.Connection, pair: tuple[UUID, UUID], state: MESIState, **values
) -> None:
    artifact_id, agent_id = pair
    conn.execute(
        "INSERT INTO agent_states (artifact_id, agent_id, state, granted_at_tick, "
        "last_observed_version, read_generation) VALUES (?, ?, ?, ?, ?, ?)",
        (
            artifact_id.hex, agent_id.hex, state.name, values.get("granted_at"),
            values.get("observed"), values.get("read_generation"),
        ),
    )


def seed_notice(
    conn: sqlite3.Connection, pair: tuple[UUID, UUID], preempter: UUID, at: float
) -> None:
    agent_id, artifact_id = pair
    conn.execute(
        "INSERT INTO pending_notices (agent_id, artifact_id, preempter_agent_id, "
        "preempted_at_unix_ts) VALUES (?, ?, ?, ?)",
        (agent_id.hex, artifact_id.hex, preempter.hex, at),
    )


def states_of(conn: sqlite3.Connection, agent_id: UUID) -> list[dict]:
    rows = conn.execute(
        "SELECT artifact_id, state, granted_at_tick, last_observed_version, read_generation, "
        "last_reclaim_trigger, last_reclaim_tick FROM agent_states WHERE agent_id = ? "
        "ORDER BY artifact_id",
        (agent_id.hex,),
    ).fetchall()
    keys = ("artifact_id", "state", "granted_at", "observed", "read_generation", "trigger", "tick")
    return [dict(zip(keys, (UUID(hex=row[0]), *row[1:]))) for row in rows]


def notices_of(conn: sqlite3.Connection, agent_id: UUID) -> list[tuple[UUID, UUID, float]]:
    rows = conn.execute(
        "SELECT artifact_id, preempter_agent_id, preempted_at_unix_ts FROM pending_notices "
        "WHERE agent_id = ? ORDER BY artifact_id",
        (agent_id.hex,),
    ).fetchall()
    return [(UUID(hex=row[0]), UUID(hex=row[1]), row[2]) for row in rows]


def artifact_row(conn: sqlite3.Connection, artifact_id: UUID) -> dict | None:
    row = conn.execute(
        "SELECT name, version, owner_generation, content_hash, size_tokens, last_writer_id, "
        "updated_at FROM artifacts WHERE id = ?",
        (artifact_id.hex,),
    ).fetchone()
    keys = ("name", "version", "generation", "hash", "size_tokens", "last_writer", "updated_at")
    return None if row is None else dict(zip(keys, row))


def names(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT name FROM artifacts ORDER BY rowid")]


_SNAPSHOT_TABLES = (
    "artifacts", "agent_states", "pending_notices", "transfer_records", "session_pins",
    "session_meta", "workspace_checkpoint_members", "conflict_counters",
    "foreign_write_counters", "artifact_versions",
)


def tables(conn: sqlite3.Connection) -> dict[str, list[tuple]]:
    """Every row a merge can touch, in a stable order."""
    return {
        name: sorted(conn.execute(f"SELECT * FROM {name}").fetchall(), key=repr)
        for name in _SNAPSHOT_TABLES
    }


def group_of(profile: FoldProfile, ids: list[UUID], **facts) -> FoldGroup:
    return FoldGroup(
        profile=profile,
        key=fold_key(facts.get("name", "plan.md"), profile),
        ids=tuple(ids),
        disk_spelling=facts.get("disk_spelling"),
        disk_hash=facts.get("disk_hash"),
    )


def classified(profile: FoldProfile) -> FoldProfileResult:
    return FoldProfileResult(profile=profile, status="classified")


def write_file(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def assert_at_most_one_writer(registry: SqliteArtifactRegistry, artifact_id: UUID) -> None:
    writers = [
        state for state in registry.get_state_map(artifact_id).values()
        if state in (MESIState.MODIFIED, MESIState.EXCLUSIVE)
    ]
    assert len(writers) <= 1, writers


def logged(caplog: pytest.LogCaptureFixture, needle: str) -> list[str]:
    return [
        record.getMessage() for record in caplog.records
        if record.name == SERVER_LOGGER and needle in record.getMessage()
    ]


# ---------------------------------------------------------------------------
# Identity: the fold index
# ---------------------------------------------------------------------------


def test_two_spellings_of_one_name_give_one_artifact_under_caseless_and_two_under_none(
    tmp_path: Path,
) -> None:
    caseless = SqliteArtifactRegistry(tmp_path / "caseless.db")
    none = SqliteArtifactRegistry(tmp_path / "none.db")
    try:
        caseless.build_fold_index("caseless")
        first = caseless.resolve_or_register("data/x.txt", "")
        assert caseless.resolve_or_register("DATA/X.TXT", "") == first
        assert caseless.lookup_artifact_id_by_name("Data/X.txt") == first
        assert len(caseless.artifact_ids()) == 1

        none.build_fold_index("none")
        lower = none.resolve_or_register("data/x.txt", "")
        assert none.resolve_or_register("DATA/X.TXT", "") != lower
        assert len(none.artifact_ids()) == 2
    finally:
        caseless.close()
        none.close()


def test_a_new_row_takes_the_disk_spelling_the_caller_found(opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    artifact_id = registry.resolve_or_register(
        "DATA/x.txt", "", key=fold_key("DATA/x.txt", "caseless"), disk_spelling="data/x.txt"
    )
    assert registry.get_artifact(artifact_id).name == "data/x.txt"
    assert registry.lookup_artifact_id_by_name("Data/X.TXT") == artifact_id
    assert registry.lookup_artifact_id_by_name(
        "DATA/X.txt", key=fold_key("DATA/X.txt", "caseless")
    ) == artifact_id


def test_a_name_with_no_disk_spelling_is_stored_as_tidied(opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    artifact_id = registry.resolve_or_register("./new//./f.txt", "")
    assert registry.get_artifact(artifact_id).name == "new/f.txt"


def test_a_bucket_resolves_every_spelling_to_its_lowest_rowid_row_and_never_mints_another(
    raw, opened
) -> None:
    first = seed_artifact(raw, "./plan.md")
    second = seed_artifact(raw, "plan.md")
    registry = opened()
    buckets = registry.build_fold_index("none")
    assert [(bucket.key, bucket.ids) for bucket in buckets] == [("plan.md", (first, second))]
    assert registry.lookup_artifact_id_by_name("plan.md") == first
    assert registry.resolve_or_register(".//plan.md", "") == first
    assert names(raw) == ["./plan.md", "plan.md"]
    assert registry.fold_bucket_peers(first) == [second]
    assert registry.fold_bucket_peers(second) == [first]


def test_an_insert_that_rolls_back_leaves_no_index_entry(raw, opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    raw.execute(
        "CREATE TRIGGER refuse_insert BEFORE INSERT ON artifacts "
        "BEGIN SELECT RAISE(ABORT, 'injected insert failure'); END"
    )
    # The existing lost-race answer: no row of that name exists to return.
    with pytest.raises(RuntimeError, match="injected"):
        registry.resolve_or_register("data/x.txt", "")
    assert registry._fold_index.by_key == {}  # noqa: SLF001 - the index itself is the subject
    raw.execute("DROP TRIGGER refuse_insert")
    fresh = registry.resolve_or_register("DATA/X.TXT", "")
    assert names(raw) == ["DATA/X.TXT"]
    assert registry.fold_bucket_peers(fresh) == []
    assert registry.lookup_artifact_id_by_name("data/x.txt") == fresh


def test_an_index_hit_is_rechecked_by_id_so_a_deleted_row_gives_way_to_one_fresh_row(
    raw, opened
) -> None:
    deleted = seed_artifact(raw, "data/x.txt")
    registry = opened()
    registry.build_fold_index("caseless")
    raw.execute("DELETE FROM artifacts WHERE id = ?", (deleted.hex,))

    assert registry.lookup_artifact_id_by_name("data/x.txt") is None
    fresh = registry.resolve_or_register("DATA/X.TXT", "")
    assert fresh != deleted
    assert registry.get_artifact(fresh).name == "DATA/X.TXT"
    assert registry.resolve_or_register("data/x.txt", "") == fresh
    assert names(raw) == ["DATA/X.TXT"]


def test_a_remove_followed_by_a_register_gives_one_fresh_row(raw, opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    removed = registry.resolve_or_register("data/x.txt", "")
    registry.remove_artifact(removed)
    fresh = registry.resolve_or_register("DATA/X.TXT", "")
    assert fresh != removed
    assert names(raw) == ["DATA/X.TXT"]
    assert registry.fold_bucket_peers(fresh) == []


def test_a_removed_row_leaves_its_bucket_at_once(raw, opened) -> None:
    first = seed_artifact(raw, "./plan.md")
    second = seed_artifact(raw, "plan.md")
    registry = opened()
    registry.build_fold_index("none")
    registry.remove_artifact(second)
    assert registry.fold_bucket_peers(first) == []


def test_without_the_index_a_name_is_stored_and_found_exactly_as_given(opened) -> None:
    registry = opened()
    artifact_id = registry.resolve_or_register("DATA/x.txt", "", disk_spelling="data/x.txt")
    assert registry.get_artifact(artifact_id).name == "DATA/x.txt"
    assert registry.lookup_artifact_id_by_name("DATA/x.txt") == artifact_id
    assert registry.lookup_artifact_id_by_name("data/x.txt") is None


def test_a_registered_artifact_joins_the_index(opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    artifact = Artifact(name="Docs/Guide.md", version=1)
    registry.register_artifact(artifact, "")
    assert registry.lookup_artifact_id_by_name("docs/guide.md") == artifact.id


def test_two_threads_registering_two_spellings_get_one_row(opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    rounds = 60
    results: list[tuple[int, UUID]] = []
    failures: list[BaseException] = []
    start = threading.Barrier(2, timeout=10)

    def register(spell: Callable[[int], str]) -> None:
        try:
            start.wait()
            for index in range(rounds):
                results.append((index, registry.resolve_or_register(spell(index), "")))
        except BaseException as exc:  # noqa: BLE001 - reported below
            failures.append(exc)

    switch = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [
            threading.Thread(target=register, args=(spell,), daemon=True)
            for spell in (lambda i: f"d{i}/plan.md", lambda i: f"D{i}/PLAN.MD")
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)
    finally:
        sys.setswitchinterval(switch)
    assert not failures, failures
    ids_by_round: dict[int, set[UUID]] = {}
    for index, artifact_id in results:
        ids_by_round.setdefault(index, set()).add(artifact_id)
    assert all(len(ids) == 1 for ids in ids_by_round.values())
    assert len(registry.artifact_ids()) == rounds


def test_a_prefix_listing_on_a_folding_profile_names_one_row_per_key(raw, opened) -> None:
    seed_artifact(raw, "docs/b.md")
    first = seed_artifact(raw, "Docs/A.md")
    seed_artifact(raw, "DOCS/a.md")
    seed_artifact(raw, "other/c.md")
    registry = opened()
    registry.build_fold_index("caseless")
    assert registry.artifact_names_under_prefix("DOCS/") == ["Docs/A.md", "docs/b.md"]
    assert registry.lookup_artifact_id_by_name("docs/a.md") == first
    assert len(registry.artifact_names_under_prefix(".")) == 3


def test_a_prefix_listing_names_a_buckets_next_row_when_its_head_is_gone(raw, opened) -> None:
    """A row deleted behind the index gives way to the next row of its
    bucket, as a lookup does; a bucket with no row left is not listed.

    Prevents a pre-grep that misses a file whose oldest spelling is gone."""
    head = seed_artifact(raw, "Docs/A.md")
    seed_artifact(raw, "DOCS/a.md")
    alone = seed_artifact(raw, "docs/b.md")
    registry = opened()
    registry.build_fold_index("caseless")
    raw.execute("DELETE FROM artifacts WHERE id IN (?, ?)", (head.hex, alone.hex))

    assert registry.artifact_names_under_prefix("docs") == ["DOCS/a.md"]
    assert registry.artifact_names_under_prefix("") == ["DOCS/a.md"]


def test_a_prefix_listing_is_sorted_by_the_stored_names_utf8_bytes(raw, opened) -> None:
    """Neither insertion order nor key order: ``Z`` sorts before ``a``, and a
    letter beyond the BMP after U+FF21, as UTF-8 orders them (UTF-16 would
    put the surrogate pair first).

    Prevents a listing whose order depends on the order rows were written."""
    for name in ("docs/b.md", "docs/\U0001d400.md", "docs/a.md", "docs/\uff21.md", "docs/Z.md"):
        seed_artifact(raw, name)
    registry = opened()
    registry.build_fold_index("caseless")

    assert registry.artifact_names_under_prefix("DOCS") == [
        "docs/Z.md", "docs/a.md", "docs/b.md", "docs/\uff21.md", "docs/\U0001d400.md",
    ]


def seed_names(conn: sqlite3.Connection, counts: dict[str, int]) -> None:
    """``counts[top]`` file rows under each ``top``, in one transaction."""
    conn.execute("BEGIN")
    for top, count in counts.items():
        for index in range(count):
            seed_artifact(conn, f"{top}/{index:04d}.md")
    conn.execute("COMMIT")


def listed_and_rows_read(registry: SqliteArtifactRegistry, prefix: str) -> tuple[list[str], int]:
    """The listing under ``prefix``, and how many rows it read from the store."""
    rows_read: list[tuple] = []

    def counting(_cursor: sqlite3.Cursor, row: tuple) -> tuple:
        rows_read.append(row)
        return row

    registry._conn.row_factory = counting  # noqa: SLF001 - the rows read are the subject
    try:
        return registry.artifact_names_under_prefix(prefix), len(rows_read)
    finally:
        registry._conn.row_factory = None  # noqa: SLF001


def test_a_subtree_listing_on_a_folding_profile_reads_only_the_subtrees_rows(
    raw, opened
) -> None:
    """A subtree that is a small part of the store reads only its own rows'
    names, by id, in more than one batch when there are many.

    Prevents a pre-grep that reads every name in the store under the
    registry lock, whose cost grows with the whole store."""
    seed_names(raw, {"other": 600, "Docs": 520})
    registry = opened()
    registry.build_fold_index("caseless")

    listed, rows_read = listed_and_rows_read(registry, "docs/")

    assert listed == [f"Docs/{index:04d}.md" for index in range(520)]
    assert rows_read == 520


def test_a_subtree_holding_most_of_the_store_is_read_by_one_scan(raw, opened) -> None:
    """Past half the store one scan of every row costs less than a lookup
    per row, so the listing reads the store once, as a listing of the root
    does, and names the same rows.

    Prevents a pre-grep over most of a large store taking longer than the
    scan it replaced."""
    seed_names(raw, {"other": 20, "Docs": 520})
    registry = opened()
    registry.build_fold_index("caseless")

    listed, rows_read = listed_and_rows_read(registry, "docs")

    assert listed == [f"Docs/{index:04d}.md" for index in range(520)]
    assert rows_read == 540
    assert listed_and_rows_read(registry, "")[1] == 540


# ---------------------------------------------------------------------------
# Names kept exact as keys
# ---------------------------------------------------------------------------


def test_a_key_keeps_its_exact_name_is_listed_and_stays_out_of_the_index(raw, opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    upper = registry.resolve_or_register("WS-42", "", name_kind="key")
    lower = registry.resolve_or_register("ws-42", "", name_kind="key")
    assert upper != lower
    assert registry.resolve_or_register("WS-42", "", name_kind="key") == upper
    assert registry.lookup_artifact_id_by_name("ws-42", name_kind="key") == lower
    assert registry.lookup_artifact_id_by_name("Ws-42") is None
    listed = raw.execute("SELECT artifact_id, kind FROM artifact_name_kinds").fetchall()
    assert sorted(listed) == sorted([(upper.hex, "key"), (lower.hex, "key")])

    registry.close()
    reopened = opened()
    assert reopened.build_fold_index("caseless") == []
    assert reopened.lookup_artifact_id_by_name("Ws-42") is None


def test_a_key_request_meeting_a_file_row_of_the_same_exact_name_shares_it(raw, opened) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    file_id = registry.resolve_or_register("data/x.txt", "")
    assert registry.resolve_or_register("data/x.txt", "", name_kind="key") == file_id
    assert registry.lookup_artifact_id_by_name("data/x.txt", name_kind="key") == file_id
    assert raw.execute("SELECT * FROM artifact_name_kinds").fetchall() == []


def test_a_file_request_of_a_key_rows_exact_name_reaches_it_by_lookup_and_by_register(
    raw, opened
) -> None:
    registry = opened()
    registry.build_fold_index("caseless")
    key_id = registry.resolve_or_register("docs/plan.md", "", name_kind="key")

    assert registry.lookup_artifact_id_by_name("docs/plan.md") == key_id
    registered = registry.resolve_or_register("docs/plan.md", "")
    assert registered == key_id
    registry.set_agent_state(registered, AGENT_A, MESIState.EXCLUSIVE)
    looked_up = registry.lookup_artifact_id_by_name("docs/plan.md")
    assert registry.get_agent_state(looked_up, AGENT_A) == MESIState.EXCLUSIVE
    assert registry.lookup_artifact_id_by_name("DOCS/PLAN.MD") is None
    assert registry.lookup_artifact_id_by_name("docs/plan.md", name_kind="key") == key_id
    assert raw.execute("SELECT artifact_id FROM artifact_name_kinds").fetchall() == [
        (key_id.hex,)
    ]


def test_the_side_table_is_created_at_open_with_no_user_version_change(store: Store) -> None:
    def has_side_table() -> bool:
        conn = sqlite3.connect(store.db)
        try:
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'artifact_name_kinds'"
            ).fetchone() is not None
        finally:
            conn.close()

    assert not has_side_table()
    SqliteArtifactRegistry(store.db).close()
    assert has_side_table()
    conn = sqlite3.connect(store.db)
    try:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_USER_VERSION
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The bucket rule: rows the merge left apart act as one file
# ---------------------------------------------------------------------------


def bucket_with_holder(raw, opened) -> tuple[SqliteArtifactRegistry, UUID, UUID]:
    first = seed_artifact(raw, "./plan.md")
    second = seed_artifact(raw, "plan.md")
    seed_state(raw, (second, AGENT_C), MESIState.EXCLUSIVE, granted_at=TICK - 10, observed=1,
               read_generation=0)
    registry = opened()
    registry.build_fold_index("none")
    return registry, first, second


def test_an_opted_in_alias_acquire_is_refused_while_another_agent_holds_the_other_row(
    raw, opened
) -> None:
    registry, first, _ = bucket_with_holder(raw, opened)
    assert registry.lookup_artifact_id_by_name("plan.md") == first
    before = tables(raw)
    with pytest.raises(OtherHolder) as refused:
        CoordinatorService(registry).write(
            agent_id=AGENT_A, artifact_id=first, issued_at_tick=TICK, if_unheld=True
        )
    assert refused.value.holder_agent_id == AGENT_C
    assert tables(raw) == before


def test_an_ordinary_acquire_in_a_bucket_demotes_write_holders_on_its_other_rows(
    raw, opened
) -> None:
    registry, first, second = bucket_with_holder(raw, opened)
    service = CoordinatorService(registry)
    signals = service.write(agent_id=AGENT_A, artifact_id=first, issued_at_tick=TICK)

    assert signals == [InvalidationSignal(
        artifact_id=second, new_version=1, issued_at_tick=TICK,
        issuer_agent_id=FOLD_MERGE_PREEMPTER_ID, victim_agent_id=AGENT_C,
        victim_state=MESIState.EXCLUSIVE,
    )]
    assert registry.get_agent_state(first, AGENT_A) == MESIState.EXCLUSIVE
    assert states_of(raw, AGENT_C) == [{
        "artifact_id": second, "state": "INVALID", "granted_at": None, "observed": 1,
        "read_generation": 0, "trigger": FOLD_MERGE_TRIGGER, "tick": TICK,
    }]
    with pytest.raises(CommitNotAllowed, match=f"reclaimed_by={FOLD_MERGE_TRIGGER}"):
        service.commit(agent_id=AGENT_C, artifact_id=second, content="", content_hash=sha256("C"))


def test_an_acquire_leaves_the_agents_own_grant_on_another_row_of_its_bucket(raw, opened) -> None:
    first = seed_artifact(raw, "./plan.md")
    second = seed_artifact(raw, "plan.md")
    seed_state(raw, (second, AGENT_A), MESIState.EXCLUSIVE, granted_at=TICK - 10, observed=1)
    registry = opened()
    registry.build_fold_index("none")
    signals = CoordinatorService(registry).write(
        agent_id=AGENT_A, artifact_id=first, issued_at_tick=TICK, if_unheld=True
    )
    assert signals == []
    assert registry.get_agent_state(second, AGENT_A) == MESIState.EXCLUSIVE


def test_the_giver_of_a_live_handoff_on_another_row_of_the_bucket_is_refused_its_acquire(
    raw, opened
) -> None:
    first = seed_artifact(raw, "./plan.md")
    second = seed_artifact(raw, "plan.md")
    seed_state(raw, (second, AGENT_C), MESIState.EXCLUSIVE, granted_at=TICK - 10, observed=1,
               read_generation=0)
    seed_transfer(raw, second, AGENT_A)
    registry = opened()
    registry.build_fold_index("none")
    before = tables(raw)
    with pytest.raises(GiverFenced) as refused:
        CoordinatorService(registry).write(agent_id=AGENT_A, artifact_id=first, issued_at_tick=TICK)
    assert (refused.value.successor, refused.value.version_at_transfer) == (AGENT_C, 1)
    assert registry.get_agent_state(second, AGENT_C) == MESIState.EXCLUSIVE
    assert tables(raw) == before


def test_the_pre_edit_holder_checks_count_a_holder_of_another_row_of_the_bucket(
    raw, opened
) -> None:
    registry, first, _ = bucket_with_holder(raw, opened)
    coordinator = SimpleNamespace(registry=registry)
    assert server_module._exclusive_holder(coordinator, first, exclude_agent=AGENT_A) == (
        AGENT_C, TICK - 10
    )
    assert server_module._a_write_is_in_flight(
        registry, first, now_unix=NOW, window_sec=5.0, exclude_agent=AGENT_A
    ) is True


def test_an_ordinary_pre_edit_in_a_bucket_warns_of_the_holder_of_the_other_row(
    store: Store, raw
) -> None:
    holder = session_to_agent_id(_sid("bucket-holder"))
    seed_artifact(raw, "./plan.md")
    second = seed_artifact(raw, "plan.md")
    seed_state(raw, (second, holder), MESIState.EXCLUSIVE, granted_at=int(time.time()),
               observed=1, read_generation=0)
    server = CoordinatorHTTPServer(store.root, port=0)
    server.serve_in_thread()
    try:
        client = _Client("127.0.0.1", server.port, load_secret(server.coordinator_root))
        status, body = client.post(
            "/hooks/pre-edit", {"session_id": _sid("bucket-editor"), "path": "plan.md"}
        )
    finally:
        server.shutdown()
    assert status == 200, body
    assert body.get("collision") is True, body
    assert holder.hex[:8] in body["hookSpecificOutput"]["additionalContext"]
    assert [(row, preempter) for row, preempter, _ in notices_of(raw, holder)] == [
        (second, FOLD_MERGE_PREEMPTER_ID)
    ]


# ---------------------------------------------------------------------------
# The merge: survivor, values, keeper, demotion
# ---------------------------------------------------------------------------


def test_the_merge_keeps_the_grant_of_the_holder_whose_commit_the_disk_holds(raw, opened) -> None:
    bytes_of_b = sha256("written by B at t=250\n")
    # B acquired the NFD spelling at t=100 and committed it at t=250; A
    # acquired the NFC spelling at t=200, so A never read what the disk holds.
    nfd = seed_artifact(raw, NFD, version=2, generation=1, hash=bytes_of_b, updated_at=250.0,
                        last_writer=AGENT_B)
    nfc = seed_artifact(raw, NFC, version=1, hash="", updated_at=200.0)
    seed_state(raw, (nfd, AGENT_B), MESIState.MODIFIED, granted_at=100, observed=2,
               read_generation=1)
    seed_state(raw, (nfc, AGENT_A), MESIState.EXCLUSIVE, granted_at=200, observed=1,
               read_generation=0)
    registry = opened()

    outcome = registry.merge_fold_group(
        group_of("caseless", [nfd, nfc], name=NFC, disk_spelling=NFC, disk_hash=bytes_of_b), NOW
    )
    assert outcome == FoldMergeOutcome(
        merged=True, survivor_id=nfc, removed_ids=(nfd,), keeper_agent_id=AGENT_B,
        demoted_writers=(AGENT_A,),
    )
    assert artifact_row(raw, nfd) is None
    assert artifact_row(raw, nfc) == {
        "name": NFC, "version": 3, "generation": 2, "hash": bytes_of_b, "size_tokens": None,
        "last_writer": AGENT_B.hex, "updated_at": 250.0,
    }
    assert states_of(raw, AGENT_B) == [{
        "artifact_id": nfc, "state": "MODIFIED", "granted_at": 100, "observed": 3,
        "read_generation": 2, "trigger": None, "tick": None,
    }]
    assert states_of(raw, AGENT_A) == [{
        "artifact_id": nfc, "state": "INVALID", "granted_at": None, "observed": 1,
        "read_generation": 0, "trigger": FOLD_MERGE_TRIGGER, "tick": TICK,
    }]
    assert notices_of(raw, AGENT_A) == [(nfc, FOLD_MERGE_PREEMPTER_ID, NOW)]
    assert notices_of(raw, AGENT_B) == []
    assert_at_most_one_writer(registry, nfc)

    service = CoordinatorService(registry)
    with pytest.raises(CommitNotAllowed, match=f"reclaimed_by={FOLD_MERGE_TRIGGER}"):
        service.commit(agent_id=AGENT_A, artifact_id=nfc, content="", content_hash=sha256("A"))
    refused = service.commit_cas(
        agent_id=AGENT_A, artifact_id=nfc, expected_version=1, content_hash=sha256("A")
    )
    assert isinstance(refused, ConflictDetail) and refused.reason == "version_mismatch"
    # The generation fence: the demoted holder's claim is below the new
    # generation, and the keeper's is the new generation itself.
    current = registry.get_artifact(nfc)
    with pytest.raises(StaleReadGeneration):
        registry.set_artifact_and_content(
            nfc, current, "", last_writer=AGENT_A, fence_agent_id=AGENT_A
        )
    updated, _ = service.commit(
        agent_id=AGENT_B, artifact_id=nfc, content="", content_hash=sha256("B again")
    )
    assert updated.version == 4
    assert_at_most_one_writer(registry, nfc)


def split_pair_with_writers(raw, hashes: tuple[str, str]) -> tuple[UUID, UUID]:
    first = seed_artifact(raw, "data/x.txt", version=1, hash=hashes[0], updated_at=100.0,
                          last_writer=AGENT_A)
    second = seed_artifact(raw, "DATA/X.TXT", version=3, hash=hashes[1], updated_at=150.0,
                           last_writer=AGENT_B, size_tokens=7)
    seed_state(raw, (first, AGENT_A), MESIState.EXCLUSIVE, granted_at=90, observed=1,
               read_generation=0)
    seed_state(raw, (second, AGENT_B), MESIState.MODIFIED, granted_at=120, observed=3,
               read_generation=0)
    return first, second


def demoted_at(artifact_id: UUID, observed: int) -> dict:
    return {
        "artifact_id": artifact_id, "state": "INVALID", "granted_at": None, "observed": observed,
        "read_generation": 0, "trigger": FOLD_MERGE_TRIGGER, "tick": TICK,
    }


def test_when_no_rows_hash_matches_the_disk_nobody_keeps_a_write_grant(raw, opened) -> None:
    first, second = split_pair_with_writers(raw, (sha256("one"), sha256("two")))
    registry = opened()
    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_hash=sha256("neither")), NOW
    )
    assert outcome.merged and outcome.keeper_agent_id is None
    # No disk spelling: the lowest rowid survives, carrying the latest write's values.
    assert artifact_row(raw, first) == {
        "name": "data/x.txt", "version": 4, "generation": 1, "hash": sha256("two"),
        "size_tokens": 7, "last_writer": AGENT_B.hex, "updated_at": 150.0,
    }
    assert artifact_row(raw, second) is None
    assert states_of(raw, AGENT_A) == [demoted_at(first, 1)]
    assert states_of(raw, AGENT_B) == [demoted_at(first, 3)]
    for agent in (AGENT_A, AGENT_B):
        assert notices_of(raw, agent) == [(first, FOLD_MERGE_PREEMPTER_ID, NOW)]
    assert_at_most_one_writer(registry, first)


def test_a_row_hash_in_another_format_matches_nothing(raw, opened) -> None:
    disk = sha256("on disk")
    first, second = split_pair_with_writers(raw, (disk.upper(), sha256("other")))
    registry = opened()
    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_hash=disk), NOW
    )
    assert outcome.merged and outcome.keeper_agent_id is None


def test_when_several_rows_match_the_disk_hash_nobody_keeps_a_write_grant(raw, opened) -> None:
    same = sha256("same bytes")
    first, second = split_pair_with_writers(raw, (same, same))
    registry = opened()
    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_spelling="DATA/X.TXT",
                 disk_hash=same),
        NOW,
    )
    assert outcome.survivor_id == second and outcome.keeper_agent_id is None
    assert states_of(raw, AGENT_A) == [demoted_at(second, 1)]
    assert states_of(raw, AGENT_B) == [demoted_at(second, 3)]


def test_a_lone_holder_among_several_rows_matching_the_disk_keeps_nothing(raw, opened) -> None:
    same = sha256("same bytes")
    # The grant goes only to the single holder of the single row whose hash is
    # the disk's; with two such rows, a lone holder on one of them does not fit.
    first = seed_artifact(raw, "data/x.txt", hash=same, updated_at=50.0)
    second = seed_artifact(raw, "DATA/X.TXT", hash=same, updated_at=100.0)
    seed_state(raw, (first, AGENT_A), MESIState.EXCLUSIVE, granted_at=40, observed=1,
               read_generation=0)
    seed_state(raw, (second, AGENT_B), MESIState.SHARED, observed=1)
    registry = opened()
    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_hash=same), NOW
    )
    assert outcome.keeper_agent_id is None and outcome.demoted_writers == (AGENT_A,)
    assert states_of(raw, AGENT_A) == [demoted_at(first, 1)]
    assert notices_of(raw, AGENT_A) == [(first, FOLD_MERGE_PREEMPTER_ID, NOW)]
    with pytest.raises(CommitNotAllowed):
        CoordinatorService(registry).commit(
            agent_id=AGENT_A, artifact_id=first, content="", content_hash=sha256("A")
        )


def test_when_the_row_matching_the_disk_has_no_write_holder_nobody_keeps_one(raw, opened) -> None:
    disk = sha256("on disk")
    # The row the disk matches is not the latest write: its values still win.
    first = seed_artifact(raw, "data/x.txt", hash=disk, updated_at=50.0)
    second = seed_artifact(raw, "DATA/X.TXT", hash=sha256("other"), updated_at=100.0)
    seed_state(raw, (first, AGENT_A), MESIState.SHARED, observed=1)
    seed_state(raw, (second, AGENT_B), MESIState.EXCLUSIVE, granted_at=40, observed=1,
               read_generation=0)
    registry = opened()
    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_hash=disk), NOW
    )
    assert outcome.keeper_agent_id is None
    assert states_of(raw, AGENT_B) == [demoted_at(first, 1)]
    assert states_of(raw, AGENT_A) == [{
        "artifact_id": first, "state": "INVALID", "granted_at": None, "observed": 1,
        "read_generation": None, "trigger": None, "tick": None,
    }]
    assert notices_of(raw, AGENT_A) == []
    assert artifact_row(raw, first)["hash"] == disk


def test_a_matching_row_with_two_write_holders_keeps_nobody(raw, opened) -> None:
    disk = sha256("on disk")
    # A store another writer touched: two holders on the row the disk matches.
    first = seed_artifact(raw, "data/x.txt", hash=disk)
    second = seed_artifact(raw, "DATA/X.TXT")
    for agent in (AGENT_A, AGENT_B):
        seed_state(raw, (first, agent), MESIState.EXCLUSIVE, granted_at=10, observed=1,
                   read_generation=0)
    registry = opened()
    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_hash=disk), NOW
    )
    assert outcome.keeper_agent_id is None
    assert outcome.demoted_writers == (AGENT_A, AGENT_B)


def test_one_agent_with_rows_on_both_spellings_ends_with_one_state_row_and_one_notice(
    raw, opened
) -> None:
    disk = sha256("on disk")
    first = seed_artifact(raw, "data/x.txt", version=4, hash=disk, updated_at=100.0)
    second = seed_artifact(raw, "DATA/X.TXT", version=2, updated_at=90.0)
    # C read both spellings and was preempted on each; B holds the matching
    # row and reads the other.
    seed_state(raw, (first, AGENT_C), MESIState.SHARED, observed=4)
    seed_state(raw, (second, AGENT_C), MESIState.INVALID, observed=1)
    seed_notice(raw, (AGENT_C, first), AGENT_A, 50.0)
    seed_notice(raw, (AGENT_C, second), AGENT_B, 80.0)
    seed_state(raw, (first, AGENT_B), MESIState.EXCLUSIVE, granted_at=60, observed=4,
               read_generation=0)
    seed_state(raw, (second, AGENT_B), MESIState.SHARED, observed=2)
    registry = opened()

    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_spelling="data/x.txt",
                 disk_hash=disk),
        NOW,
    )
    assert outcome.keeper_agent_id == AGENT_B
    assert states_of(raw, AGENT_C) == [{
        "artifact_id": first, "state": "INVALID", "granted_at": None, "observed": 1,
        "read_generation": None, "trigger": None, "tick": None,
    }]
    assert notices_of(raw, AGENT_C) == [(first, AGENT_B, 80.0)]
    assert states_of(raw, AGENT_B) == [{
        "artifact_id": first, "state": "EXCLUSIVE", "granted_at": 60, "observed": 5,
        "read_generation": 1, "trigger": None, "tick": None,
    }]
    assert_at_most_one_writer(registry, first)


def test_the_merge_preempter_id_is_the_uuid5_of_its_name() -> None:
    assert FOLD_MERGE_PREEMPTER_ID == uuid5(
        NAMESPACE_URL, "ccs-coordinator-fold-merge:merged-spellings"
    )
    assert FOLD_MERGE_PREEMPTER_ID.hex == "707e50c79be1538fa2ac5d849131619e"


def test_a_merge_whose_rows_changed_since_detection_is_skipped_and_changes_nothing(
    raw, opened
) -> None:
    first = seed_artifact(raw, "data/x.txt")
    second = seed_artifact(raw, "DATA/X.TXT")
    third = seed_artifact(raw, "Data/X.Txt")
    registry = opened()
    raw.execute("INSERT INTO artifact_name_kinds (artifact_id, kind) VALUES (?, 'key')",
                (second.hex,))
    before = tables(raw)
    keyed = registry.merge_fold_group(group_of("caseless", [first, second], name="data/x.txt"), NOW)
    assert not keyed.merged and keyed.reason
    assert tables(raw) == before

    raw.execute("DELETE FROM artifacts WHERE id = ?", (third.hex,))
    remaining = tables(raw)
    gone = registry.merge_fold_group(group_of("caseless", [first, third], name="data/x.txt"), NOW)
    assert not gone.merged
    assert tables(raw) == remaining

    raw.execute("DELETE FROM artifact_name_kinds")
    apart = registry.merge_fold_group(group_of("none", [first, second], name="data/x.txt"), NOW)
    assert not apart.merged
    assert tables(raw) == remaining


def test_the_merge_rolls_back_when_the_survivor_would_end_with_two_write_holders(
    raw, opened
) -> None:
    disk = sha256("on disk")
    first = seed_artifact(raw, "data/x.txt", hash=disk)
    second = seed_artifact(raw, "DATA/X.TXT")
    seed_state(raw, (first, AGENT_B), MESIState.EXCLUSIVE, granted_at=10, observed=1,
               read_generation=0)
    registry = opened()
    # Each time the keeper's row is written, a second writer appears beside it.
    raw.execute(
        f"CREATE TRIGGER second_writer AFTER INSERT ON agent_states WHEN NEW.agent_id = "
        f"'{AGENT_B.hex}' BEGIN INSERT INTO agent_states (artifact_id, agent_id, state) "
        f"VALUES (NEW.artifact_id, '{AGENT_C.hex}', 'EXCLUSIVE'); END"
    )
    before = tables(raw)
    with pytest.raises(InvariantViolationError, match="single_writer_violated"):
        registry.merge_fold_group(
            group_of("caseless", [first, second], name="data/x.txt", disk_hash=disk), NOW
        )
    assert tables(raw) == before


def test_the_merge_logs_one_state_entry_per_moved_or_demoted_agent_which_replay_accepts(
    tmp_path: Path, raw, opened
) -> None:
    disk = sha256("on disk")
    first = seed_artifact(raw, "data/x.txt", hash=disk)
    second = seed_artifact(raw, "DATA/X.TXT", version=2)
    seed_state(raw, (first, AGENT_B), MESIState.EXCLUSIVE, granted_at=10, observed=1,
               read_generation=0)
    seed_state(raw, (second, AGENT_A), MESIState.MODIFIED, granted_at=20, observed=2,
               read_generation=0)
    seed_state(raw, (second, AGENT_C), MESIState.SHARED, observed=2)
    session_dir = tmp_path / "trace"
    entries: list[dict] = []
    with record_callbacks(session_dir, streams={"state_log"}, accept_unverified=True,
                          state_log=entries.append) as (state_log, _):
        registry = opened(state_log=state_log, instance_id="fold-merge-test")
        registry.merge_fold_group(
            group_of("caseless", [first, second], name="data/x.txt", disk_hash=disk), NOW
        )
        registry.close()

    transitions = sorted(
        (entry["agent_id"], entry["from_state"], entry["to_state"]) for entry in entries
    )
    assert transitions == sorted([
        (str(AGENT_A), "MODIFIED", "INVALID"),
        (str(AGENT_B), "EXCLUSIVE", "EXCLUSIVE"),
        (str(AGENT_C), "SHARED", "INVALID"),
    ])
    assert {entry["trigger"] for entry in entries} == {FOLD_MERGE_TRIGGER}
    assert {entry["artifact_id"] for entry in entries} == {str(first)}
    assert {entry["version"] for entry in entries} == {3}
    assert validate_log(session_dir / "state_log.jsonl") == ([], [])
    findings, _ = run_predicates(load(session_dir))
    assert findings == []


# ---------------------------------------------------------------------------
# The merge: the tables only the Python coordinator keeps
# ---------------------------------------------------------------------------


def test_a_session_pinning_a_grouped_row_is_ended_and_its_next_read_finds_no_session(
    raw, opened
) -> None:
    first = seed_artifact(raw, "data/x.txt")
    second = seed_artifact(raw, "DATA/X.TXT")
    other = seed_artifact(raw, "other.md")
    for token, artifact_id in (("pins-first", first), ("pins-second", second), ("kept", other)):
        raw.execute("INSERT INTO session_pins VALUES (?, ?, 1)", (token, artifact_id.hex))
        raw.execute("INSERT INTO session_meta VALUES (?, ?, 1)", (token, AGENT_A.hex))
    registry = opened()
    registry.merge_fold_group(group_of("caseless", [first, second], name="data/x.txt"), NOW)

    assert raw.execute("SELECT session_token FROM session_pins").fetchall() == [("kept",)]
    assert raw.execute("SELECT session_token FROM session_meta").fetchall() == [("kept",)]
    answer = CoordinatorService(registry).session_read("pins-second", first, caller=AGENT_A)
    assert isinstance(answer, SessionReadRejection) and answer.reason == "session_not_found"


def seed_transfer(raw, artifact_id: UUID, giver: UUID) -> None:
    raw.execute(
        "INSERT INTO transfer_records (artifact_id, giver, holder, successor, "
        "version_at_transfer, hold_shape, cause, status, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, 1, 'EXCLUSIVE', 'handoff', 'pending', 1.0, 1.0)",
        (artifact_id.hex, giver.hex, giver.hex, AGENT_C.hex),
    )


def test_every_transfer_record_on_the_group_ends_withdrawn(raw, opened) -> None:
    first = seed_artifact(raw, "data/x.txt")
    second = seed_artifact(raw, "DATA/X.TXT")
    seed_transfer(raw, first, AGENT_A)
    seed_transfer(raw, second, AGENT_B)
    registry = opened()
    registry.merge_fold_group(group_of("caseless", [first, second], name="data/x.txt"), NOW)

    assert raw.execute(
        "SELECT artifact_id, status, updated_at FROM transfer_records"
    ).fetchall() == [(first.hex, "withdrawn", NOW)]
    record, live = registry.get_transfer_record(first)
    assert record.status == "withdrawn" and not live


def test_conflict_and_foreign_write_counters_are_summed(raw, opened) -> None:
    disk = sha256("on disk")
    first = seed_artifact(raw, "data/x.txt", hash=disk)
    second = seed_artifact(raw, "DATA/X.TXT")
    raw.executemany("INSERT INTO conflict_counters VALUES (?, ?, ?, ?)", [
        (first.hex, AGENT_A.hex, "version_mismatch", 2),
        (second.hex, AGENT_A.hex, "version_mismatch", 3),
        (second.hex, AGENT_B.hex, "other_holder", 1),
    ])
    raw.executemany("INSERT INTO foreign_write_counters VALUES (?, ?, ?, ?, ?, ?)", [
        (first.hex, 1, 0, 2, sha256("h1"), "foreign"),
        (second.hex, 4, 1, 0, sha256("h2"), "mediated"),
    ])
    registry = opened()
    registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_hash=disk), NOW
    )
    assert sorted(raw.execute("SELECT * FROM conflict_counters").fetchall()) == sorted([
        (first.hex, AGENT_A.hex, "version_mismatch", 5),
        (first.hex, AGENT_B.hex, "other_holder", 1),
    ])
    assert raw.execute("SELECT * FROM foreign_write_counters").fetchall() == [
        (first.hex, 5, 1, 2, disk, "")
    ]


def test_a_group_whose_merged_away_row_holds_retained_history_is_not_merged(raw, opened) -> None:
    first = seed_artifact(raw, "data/x.txt")
    second = seed_artifact(raw, "DATA/X.TXT")
    raw.execute("INSERT INTO artifact_versions VALUES (?, 1, 'body', 1.0)", (second.hex,))
    registry = opened()
    before = tables(raw)
    outcome = registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt"), NOW
    )
    assert not outcome.merged and "history" in outcome.reason
    assert tables(raw) == before


def test_a_checkpoint_member_is_repointed_and_its_registration_answers_the_same(
    raw, opened
) -> None:
    fingerprint = sha256("restored bytes")
    first = seed_artifact(raw, "data/x.txt", hash=sha256("older"), updated_at=10.0)
    second = seed_artifact(raw, "DATA/X.TXT", version=2, hash=fingerprint, updated_at=20.0)
    raw.execute(
        "INSERT INTO workspace_checkpoints (checkpoint_id, name, owner, created_at, "
        "created_at_tick, window_min, window_max, restore_status) "
        "VALUES ('cp', 'cp', ?, 1.0, 1, 1.0, 1.0, 'concluded')",
        (AGENT_A.hex,),
    )
    raw.execute(
        "INSERT INTO workspace_checkpoint_members (checkpoint_id, member_path, artifact_id, "
        "fingerprint, captured_at, arbitration_tier, restore_tier) "
        "VALUES ('cp', 'DATA/X.TXT', ?, ?, 1.0, 'no-arbiter', 'restorable')",
        (second.hex, fingerprint),
    )
    registry = opened()
    service = CoordinatorService(registry)
    assert service.workspace_member_registered("DATA/X.TXT", fingerprint) is True

    registry.merge_fold_group(
        group_of("caseless", [first, second], name="data/x.txt", disk_hash=fingerprint), NOW
    )
    registry.build_fold_index("caseless")
    assert service.workspace_member_registered("DATA/X.TXT", fingerprint) is True
    assert raw.execute(
        "SELECT member_path, artifact_id FROM workspace_checkpoint_members"
    ).fetchall() == [("DATA/X.TXT", first.hex)]


# ---------------------------------------------------------------------------
# Groups and the start
# ---------------------------------------------------------------------------


def test_rows_that_fold_together_but_reach_different_files_or_none_are_not_grouped() -> None:
    rows = [FoldRow(rowid=1, id=uuid4(), name="data/x.txt"),
            FoldRow(rowid=2, id=uuid4(), name="DATA/X.TXT")]

    def file(ino: int, dev: int = 1) -> ResolvedFile:
        return ResolvedFile(dev=dev, ino=ino, real_path=f"/w/{ino}", disk_spelling="data/x.txt")

    assert same_file_groups(rows, [file(10), file(11)]) == []
    assert same_file_groups(rows, [file(10), None]) == []
    assert same_file_groups(rows, [file(10, dev=2), file(10)]) == []
    assert same_file_groups(rows, [file(10), file(10)]) == [(rows, file(10))]


def tidy_split(store: Store, raw) -> tuple[UUID, UUID]:
    write_file(store.root, "data/x.txt", "bytes on disk\n")
    untidy = seed_artifact(raw, "./data/x.txt", hash=sha256("bytes on disk\n"))
    tidy = seed_artifact(raw, "data/x.txt", version=2)
    return untidy, tidy


def test_a_tidy_only_split_merges_at_start_and_the_row_with_the_disk_spelling_survives(
    store: Store, raw, opened, caplog
) -> None:
    untidy, tidy = tidy_split(store, raw)
    registry = opened()
    with caplog.at_level(logging.INFO, logger=SERVER_LOGGER):
        prepare_fold_index(registry, store.root, classified("none"))
    assert names(raw) == ["data/x.txt"]
    assert artifact_row(raw, untidy) is None
    assert artifact_row(raw, tidy)["version"] == 3
    assert artifact_row(raw, tidy)["hash"] == sha256("bytes on disk\n")
    assert registry.lookup_artifact_id_by_name("./data//x.txt") == tidy
    assert registry.fold_index_profile == "none"
    assert len(logged(caplog, "merged into")) == 1


def test_rows_listed_as_keys_are_not_merged(store: Store, raw, opened, caplog) -> None:
    write_file(store.root, "data/x.txt", "bytes\n")
    first = opened()
    key = first.resolve_or_register("./data/x.txt", "", name_kind="key")
    file_id = first.resolve_or_register("data/x.txt", "")
    first.close()
    before = tables(raw)
    registry = opened()
    with caplog.at_level(logging.INFO, logger=SERVER_LOGGER):
        prepare_fold_index(registry, store.root, classified("none"))
    assert tables(raw) == before
    assert logged(caplog, "fold merge") == [] and logged(caplog, "fold bucket") == []
    assert registry.lookup_artifact_id_by_name("./data/x.txt") == file_id
    assert registry.lookup_artifact_id_by_name("./data/x.txt", name_kind="key") == key


def test_rows_whose_file_resolves_outside_the_workspace_root_are_not_merged(
    tmp_path: Path, store: Store, raw, opened, caplog
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "x.txt").write_text("outside\n", encoding="utf-8")
    (store.root / "out.txt").symlink_to(outside / "x.txt")
    first = seed_artifact(raw, "./out.txt")
    second = seed_artifact(raw, "out.txt")
    before = tables(raw)
    registry = opened()
    with caplog.at_level(logging.INFO, logger=SERVER_LOGGER):
        prepare_fold_index(registry, store.root, classified("none"))
    assert tables(raw) == before
    assert registry.lookup_artifact_id_by_name("out.txt") == first
    buckets = logged(caplog, '"out.txt"')
    assert len(buckets) == 1 and f"{first.hex}, {second.hex}" in buckets[0], buckets


def test_a_start_with_no_groups_leaves_the_store_file_byte_identical(
    store: Store, raw, caplog
) -> None:
    seed_artifact(raw, "docs/a.md", hash=sha256("a"))
    # Two spellings of a file that is not on disk: a bucket, never a group.
    seed_artifact(raw, "gone.md")
    seed_artifact(raw, "./GONE.md")
    seed_state(raw, (seed_artifact(raw, "docs/b.md"), AGENT_A), MESIState.EXCLUSIVE,
               granted_at=5)
    raw.close()
    # The first open of this build adds the side table; the start measured is the next one.
    SqliteArtifactRegistry(store.db).close()
    before = store.db.read_bytes()

    registry = SqliteArtifactRegistry(store.db)
    with caplog.at_level(logging.INFO, logger=SERVER_LOGGER):
        prepare_fold_index(registry, store.root, classified("caseless"))
    registry.close()
    assert store.db.read_bytes() == before
    assert len(logged(caplog, '"gone.md"')) == 1


def test_a_merge_that_fails_mid_transaction_is_rolled_back_logged_and_keeps_the_profile(
    store: Store, raw, opened, caplog
) -> None:
    untidy, _ = tidy_split(store, raw)
    raw.execute(
        "CREATE TRIGGER refuse_delete BEFORE DELETE ON artifacts "
        "BEGIN SELECT RAISE(ABORT, 'injected merge failure'); END"
    )
    before = tables(raw)
    registry = opened()
    with caplog.at_level(logging.INFO, logger=SERVER_LOGGER):
        prepare_fold_index(registry, store.root, classified("caseless"))

    assert tables(raw) == before
    failures = logged(caplog, "injected merge failure")
    assert len(failures) == 1 and "rolled back" in failures[0], failures
    assert registry.fold_index_profile == "caseless"
    assert registry.lookup_artifact_id_by_name("DATA/X.TXT") == untidy
    assert registry.resolve_or_register("Data/x.txt", "") == untidy
    assert len(logged(caplog, '"data/x.txt"')) == 1


def test_an_unlocked_start_merges_nothing_and_builds_no_index(
    store: Store, raw, opened, caplog, two_writer_store
) -> None:
    _, tidy = tidy_split(store, raw)
    before = tables(raw)
    registry = opened()
    assert registry.holds_writer_lock is False
    with caplog.at_level(logging.INFO, logger=SERVER_LOGGER):
        prepare_fold_index(registry, store.root, classified("none"))
    assert tables(raw) == before
    assert registry.fold_index_profile is None
    assert registry.lookup_artifact_id_by_name("data/x.txt") == tidy
    assert registry.lookup_artifact_id_by_name("DATA/X.TXT") is None
    lines = logged(caplog, "fold")
    assert len(lines) == 1 and "single-writer lock" in lines[0], lines


def test_the_coordinator_merges_split_rows_before_its_service_exists(
    store: Store, raw, monkeypatch
) -> None:
    untidy, tidy = tidy_split(store, raw)
    seen: list[tuple[list[str], bool]] = []

    class RecordingService(CoordinatorService):
        def __init__(self, registry, **kwargs) -> None:
            seen.append((names(raw), registry.holds_writer_lock))
            super().__init__(registry, **kwargs)

    monkeypatch.setattr(server_module, "CoordinatorService", RecordingService)
    server = CoordinatorHTTPServer(store.root, port=0)
    try:
        assert seen == [(["data/x.txt"], True)]
        assert server.registry.lookup_artifact_id_by_name("./data/x.txt") == tidy
        assert server.registry.get_artifact(untidy) is None
    finally:
        server.shutdown()
