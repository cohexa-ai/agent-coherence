# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The start-time merge of file rows an older build split for one file.

An older build keyed rows by the request's spelling, so ``data/x.txt`` and
``DATA/X.TXT`` (or an NFC and an NFD spelling, or ``./data/x.txt``) could
each get a row for what the disk keeps as one file, each with its own write
grant. A group is file rows whose fold keys collide under the active profile
AND whose names reach the same existing file under the workspace root (same
``st_dev`` and ``st_ino``): a key alone could merge two real files the
profile does not model, and a file that is absent right now confirms nothing.

This module is pure. The coordinator gathers the file facts (the only
filesystem calls); the registry re-reads the rows inside its transaction,
plans with :func:`plan_fold_merge` and applies the plan. The plugin's Node
coordinator plans the same way, so the two keep the same grant, version and
notices for one store.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeVar
from uuid import UUID

from ccs.core.path_fold import FoldProfile, exceeds_fold_limit, fold_key
from ccs.core.states import MESIState

FOLD_MERGE_PREEMPTER_ID: UUID = UUID("707e50c79be1538fa2ac5d849131619e")
"""The preempter a merge-demoted holder's notice names, so the holder can tell
that no agent took its grant: ``uuid5(NAMESPACE_URL,
"ccs-coordinator-fold-merge:merged-spellings")``. A literal keeps the value
identical to the Node coordinator's; a test pins the derivation."""

FOLD_MERGE_TRIGGER = "fold_merge"
"""The trigger of every state change the merge and the bucket rule make, and
the reclaim trigger a demoted write holder's row records. Deliberately outside
the reclaim and generation-bump trigger sets: the merge sets the generation
itself, and the bucket rule's demoted row is unreachable by name."""

NameKind = Literal["file", "key"]
"""What a request names: a workspace file, keyed through the fold index, or a
key (a substrate row or object, a remote-endpoint volume's name), always
exact."""

_WRITER_STATES = frozenset({MESIState.MODIFIED, MESIState.EXCLUSIVE})
# Strongest first, for the state a demoted agent's log entry leaves.
_STATE_STRENGTH = (MESIState.MODIFIED, MESIState.EXCLUSIVE, MESIState.SHARED, MESIState.INVALID)



class _AgentRow(Protocol):
    @property
    def agent_id(self) -> str: ...


_Row = TypeVar("_Row", bound=_AgentRow)


def index_key_of(name: str, profile: FoldProfile) -> str:
    """The index key of a stored name. A name over the fold limit is never
    folded (the door refuses such a request), so it is listed under itself."""
    return name if exceeds_fold_limit(name) else fold_key(name, profile)


def is_inside_root(spelling: str) -> bool:
    """Whether a spelling relative to the workspace root names something
    strictly inside it."""
    return (
        spelling not in ("", ".", "..")
        and not spelling.startswith("../")
        and not os.path.isabs(spelling)
    )


@dataclass(frozen=True)
class FoldRow:
    """A file row as the start reads it: rowid order is insertion order."""

    rowid: int
    id: UUID
    name: str


@dataclass(frozen=True)
class FoldBucket:
    """A key several file rows share, lowest rowid first."""

    key: str
    ids: tuple[UUID, ...]


@dataclass(frozen=True)
class ResolvedFile:
    """The file a row's name reaches, as the disk spells it under the root."""

    dev: int
    ino: int
    real_path: str
    disk_spelling: str


@dataclass(frozen=True)
class FoldGroup:
    """Rows of one file, found read-only at start, and what the disk holds for
    it. ``ids`` is lowest rowid first; ``disk_hash`` is the lowercase sha-256
    hex of the file's raw bytes, the format the hook clients and the volume
    record, or None when the file could not be read."""

    profile: FoldProfile
    key: str
    ids: tuple[UUID, ...]
    disk_spelling: str | None
    disk_hash: str | None


@dataclass(frozen=True)
class FoldMergeOutcome:
    """A merge that ran (``merged``) or one the registry skipped, and why."""

    merged: bool
    survivor_id: UUID | None = None
    removed_ids: tuple[UUID, ...] = ()
    keeper_agent_id: UUID | None = None
    demoted_writers: tuple[UUID, ...] = ()
    reason: str | None = None


def colliding_rows(rows: Iterable[FoldRow], profile: FoldProfile) -> dict[str, list[FoldRow]]:
    """File rows whose keys collide, by key, each list in rowid order."""
    by_key: dict[str, list[FoldRow]] = {}
    for row in rows:
        by_key.setdefault(index_key_of(row.name, profile), []).append(row)
    return {key: keyed for key, keyed in by_key.items() if len(keyed) > 1}


def same_file_groups(
    rows: Sequence[FoldRow], resolved: Sequence[ResolvedFile | None]
) -> list[tuple[list[FoldRow], ResolvedFile]]:
    """Split rows that share a key by the file each name reaches; only a file
    two or more rows reach forms a group. ``resolved[i]`` belongs to
    ``rows[i]``, None when that name reaches no file."""
    by_file: dict[tuple[int, int], tuple[list[FoldRow], ResolvedFile]] = {}
    for row, file in zip(rows, resolved):
        if file is not None:
            by_file.setdefault((file.dev, file.ino), ([], file))[0].append(row)
    return [entry for entry in by_file.values() if len(entry[0]) > 1]


# ---------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeArtifactRow:
    """One grouped ``artifacts`` row; ids are the stored hex."""

    rowid: int
    id: str
    name: str
    version: int
    owner_generation: int
    content_hash: str
    size_tokens: int | None
    last_writer_id: str | None
    updated_at: float


@dataclass(frozen=True)
class MergeStateRow:
    """One ``agent_states`` row, in its column order."""

    artifact_id: str
    agent_id: str
    state: MESIState
    transient_state: str | None
    transient_tick: int | None
    granted_at_tick: int | None
    last_reclaim_trigger: str | None
    last_reclaim_tick: int | None
    read_generation: int | None
    last_observed_version: int | None


@dataclass(frozen=True)
class MergeNoticeRow:
    """One ``pending_notices`` row, in its column order."""

    agent_id: str
    artifact_id: str
    preempter_agent_id: str
    preempted_at_unix_ts: float


@dataclass(frozen=True)
class MergeSnapshot:
    """The group's rows as the registry read them inside its transaction;
    ``artifacts`` holds at least two rows, in rowid order."""

    artifacts: tuple[MergeArtifactRow, ...]
    states: tuple[MergeStateRow, ...]
    notices: tuple[MergeNoticeRow, ...]


@dataclass(frozen=True)
class MergeFacts:
    """What the disk holds for the group, and when the merge runs."""

    disk_spelling: str | None
    disk_hash: str | None
    now_unix: float


@dataclass(frozen=True)
class MergedValues:
    """The survivor's columns after the merge."""

    version: int
    owner_generation: int
    content_hash: str
    size_tokens: int | None
    last_writer_id: str | None
    updated_at: float


@dataclass(frozen=True)
class MergeTransition:
    """One agent's move onto the survivor, for its state-log entry."""

    agent_id: str
    from_state: MESIState
    to_state: MESIState


@dataclass(frozen=True)
class MergePlan:
    """Everything the registry writes: the survivor's values and one state row
    and at most one notice per agent, all on the survivor."""

    survivor_id: str
    removed_ids: tuple[str, ...]
    values: MergedValues
    states: tuple[MergeStateRow, ...]
    notices: tuple[MergeNoticeRow, ...]
    transitions: tuple[MergeTransition, ...]
    keeper_agent_id: str | None
    demoted_writers: tuple[str, ...]


def _choose_survivor(
    artifacts: Sequence[MergeArtifactRow], disk_spelling: str | None
) -> MergeArtifactRow:
    named = [row for row in artifacts if row.name == disk_spelling]
    return named[0] if named else artifacts[0]


def _sole_matching_row(
    artifacts: Sequence[MergeArtifactRow], disk_hash: str | None
) -> MergeArtifactRow | None:
    """The row whose recorded write the disk holds, when exactly one row's
    hash equals the disk's. A hash in another format equals nothing."""
    matching = [row for row in artifacts if disk_hash is not None and row.content_hash == disk_hash]
    return matching[0] if len(matching) == 1 else None


def _latest_write(artifacts: Sequence[MergeArtifactRow]) -> MergeArtifactRow:
    """The latest write; on a tie the oldest row, so the choice never depends
    on read order."""
    latest = artifacts[0]
    for row in artifacts[1:]:
        if row.updated_at > latest.updated_at:
            latest = row
    return latest


def _merged_values(
    artifacts: Sequence[MergeArtifactRow], matching: MergeArtifactRow | None
) -> MergedValues:
    source = matching if matching is not None else _latest_write(artifacts)
    # Above every grouped value, so no holder's recorded version names the
    # merged row's bytes and no claim captured before the merge passes the fence.
    return MergedValues(
        version=max(row.version for row in artifacts) + 1,
        owner_generation=max(row.owner_generation for row in artifacts) + 1,
        content_hash=source.content_hash,
        size_tokens=source.size_tokens,
        last_writer_id=source.last_writer_id,
        updated_at=source.updated_at,
    )


def _choose_keeper(
    states: Sequence[MergeStateRow], matching: MergeArtifactRow | None
) -> MergeStateRow | None:
    """The single write holder of the row the disk matches. Acquire times are
    never compared: whole-second ticks tie routinely, and a later acquire says
    nothing about which holder read the bytes on disk."""
    if matching is None:
        return None
    writers = [
        row for row in states if row.artifact_id == matching.id and row.state in _WRITER_STATES
    ]
    return writers[0] if len(writers) == 1 else None


def _rows_by_agent(rows: Iterable[_Row]) -> dict[str, list[_Row]]:
    by_agent: dict[str, list[_Row]] = {}
    for row in sorted(rows, key=lambda row: row.agent_id):
        by_agent.setdefault(row.agent_id, []).append(row)
    return by_agent


def _lowest(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present) if present else None


def _strongest(states: Iterable[MESIState]) -> MESIState:
    held = set(states)
    return next((state for state in _STATE_STRENGTH if state in held), MESIState.INVALID)


@dataclass(frozen=True)
class _Context:
    survivor_id: str
    now_unix: float
    rowid_of: dict[str, int]


def _demoted_state(rows: Sequence[MergeStateRow], context: _Context) -> MergeStateRow:
    """INVALID on the survivor, present and never absent, so its next commit
    is refused: a demoted writer records the merge as its reclaim, anyone else
    keeps the reclaim record of its oldest row that has one. Its read
    generation is its lowest, below the survivor's new generation."""
    ordered = sorted(rows, key=lambda row: context.rowid_of.get(row.artifact_id, 0))
    was_writer = any(row.state in _WRITER_STATES for row in ordered)
    reclaimed = next((row for row in ordered if row.last_reclaim_trigger is not None), None)
    return MergeStateRow(
        artifact_id=context.survivor_id,
        agent_id=ordered[0].agent_id,
        state=MESIState.INVALID,
        transient_state=None,
        transient_tick=None,
        granted_at_tick=None,
        last_reclaim_trigger=FOLD_MERGE_TRIGGER if was_writer else _reclaim_trigger(reclaimed),
        last_reclaim_tick=int(context.now_unix) if was_writer else _reclaim_tick(reclaimed),
        read_generation=_lowest(row.read_generation for row in ordered),
        last_observed_version=_lowest(row.last_observed_version for row in ordered),
    )


def _reclaim_trigger(row: MergeStateRow | None) -> str | None:
    return None if row is None else row.last_reclaim_trigger


def _reclaim_tick(row: MergeStateRow | None) -> int | None:
    return None if row is None else row.last_reclaim_tick


def _merged_states(
    states: Sequence[MergeStateRow], kept: MergeStateRow | None, context: _Context
) -> tuple[list[MergeStateRow], list[MergeTransition], list[str]]:
    """One state row per agent on the survivor: the keeper's as ``kept``,
    every other agent's demoted; with each agent's transition and the
    demoted write holders."""
    merged: list[MergeStateRow] = []
    transitions: list[MergeTransition] = []
    demoted_writers: list[str] = []
    for agent_id, rows in _rows_by_agent(states).items():
        if kept is not None and agent_id == kept.agent_id:
            merged.append(kept)
            transitions.append(MergeTransition(agent_id, kept.state, kept.state))
            continue
        if any(prior.state in _WRITER_STATES for prior in rows):
            demoted_writers.append(agent_id)
        merged.append(_demoted_state(rows, context))
        from_state = _strongest(prior.state for prior in rows)
        transitions.append(MergeTransition(agent_id, from_state, MESIState.INVALID))
    return merged, transitions, demoted_writers


def _merged_notices(
    notices: Sequence[MergeNoticeRow], demoted_writers: Sequence[str], context: _Context
) -> list[MergeNoticeRow]:
    """The latest notice per agent, the merge's own notices counted last so
    they win a tie."""
    from_merge = [
        MergeNoticeRow(agent_id, context.survivor_id, FOLD_MERGE_PREEMPTER_ID.hex, context.now_unix)
        for agent_id in demoted_writers
    ]
    merged: list[MergeNoticeRow] = []
    for rows in _rows_by_agent([*notices, *from_merge]).values():
        latest = rows[0]
        for row in rows[1:]:
            if row.preempted_at_unix_ts >= latest.preempted_at_unix_ts:
                latest = row
        merged.append(replace(latest, artifact_id=context.survivor_id))
    return merged


def plan_fold_merge(snapshot: MergeSnapshot, facts: MergeFacts) -> MergePlan:
    """Plan one group's merge. Pure: the registry applies the plan in one
    transaction."""
    survivor = _choose_survivor(snapshot.artifacts, facts.disk_spelling)
    matching = _sole_matching_row(snapshot.artifacts, facts.disk_hash)
    values = _merged_values(snapshot.artifacts, matching)
    keeper = _choose_keeper(snapshot.states, matching)
    context = _Context(
        survivor.id, facts.now_unix, {row.id: row.rowid for row in snapshot.artifacts}
    )
    # Moved to the new version and generation, so the keeper's next commit is admitted.
    kept = None if keeper is None else replace(
        keeper,
        artifact_id=survivor.id,
        last_observed_version=values.version,
        read_generation=values.owner_generation,
    )
    states, transitions, demoted_writers = _merged_states(snapshot.states, kept, context)
    return MergePlan(
        survivor_id=survivor.id,
        removed_ids=tuple(row.id for row in snapshot.artifacts if row.id != survivor.id),
        values=values,
        states=tuple(states),
        notices=tuple(_merged_notices(snapshot.notices, demoted_writers, context)),
        transitions=tuple(transitions),
        keeper_agent_id=keeper.agent_id if keeper is not None else None,
        demoted_writers=tuple(demoted_writers),
    )
