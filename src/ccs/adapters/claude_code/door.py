# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The door every hook route passes a request's name through before the
policy and the registry see it.

A request names a file (the default) or a key (``"name_kind": "key"``): a name
that is not a workspace file, such as a row or object key, kept exactly as
sent and looked up by that exact name. A file is tidied and keyed by the fold
the workspace disk applies, so every spelling the disk treats as one file
reaches one row. On a disk that folds, the door also finds the disk's own
spelling of the request: it names a new row, and a glob written against it
matches.

An answer repeats a request's name as :attr:`RequestName.requested`, the
bytes the client sent, because clients compare what comes back with what they
sent.

The route validates the path first (``validate_path``: the route's own checks
and the 1024 code-point limit, for files and keys alike); the door then reads
the name. It reads the fold profile from the coordinator, never from the
registry, whose lock a long registry call may hold: an untracked path must
still be answered without waiting on the registry.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from ccs.adapters.claude_code.policy import NameKind, PolicyMatchContext, TrackedArtifactPolicy
from ccs.coordinator.fold_merge import index_key_of, is_inside_root
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.disk_spelling import native_spelling
from ccs.core.path_fold import FoldProfile, fold_key, tidy_path


class DoorHost(Protocol):
    """What the door reads from the coordinator. ``fold_index_profile`` is the
    profile the registry's fold index keys by, or None while names are exact
    (a start without the single-writer lock); ``canonical_root`` is the
    workspace root as the disk spells it, resolved once at start."""

    registry: SqliteArtifactRegistry
    policy: TrackedArtifactPolicy
    fold_index_profile: FoldProfile | None
    canonical_root: str


@dataclass(frozen=True)
class RequestName:
    """A request's name as the routes use it.

    ``requested`` is the name as sent. ``name`` is what the registry is asked
    for: the tidied name of a file, the exact name of a key. ``key`` is the
    fold key a file is looked up by (a key's exact name for a key).
    ``disk_spelling`` is the file's spelling on a disk that folds, relative to
    the root, when the request reaches a place inside the root under the same
    key."""

    requested: str
    kind: NameKind
    name: str
    key: str
    disk_spelling: str | None
    policy: PolicyMatchContext


def policy_profile_of(host: DoorHost) -> FoldProfile:
    """The profile the policy folds names by: names stay exact without the
    fold index, as they do in the registry."""
    return host.fold_index_profile or "none"


def stored_file_policy(host: DoorHost) -> PolicyMatchContext:
    """The policy context of a stored row's name, which has no request
    behind it and so no disk spelling."""
    return PolicyMatchContext(profile=policy_profile_of(host))


def _name_kind_of(raw: object) -> NameKind | None:
    if raw == "file":
        return "file"
    return "key" if raw == "key" else None


def request_name(host: DoorHost, path: str, name_kind: object = "file") -> RequestName | None:
    """The name a validated request ``path`` carries, or None when its
    ``name_kind`` is neither ``"file"`` nor ``"key"``: the route then answers
    its path error, as for a path it cannot use."""
    kind = _name_kind_of(name_kind)
    if kind is None:
        return None
    if kind == "file":
        return file_request(host, path)
    policy = PolicyMatchContext(profile=policy_profile_of(host), kind="key")
    return RequestName(path, "key", path, path, None, policy)


def file_key(host: DoorHost, name: str) -> str:
    """The key every spelling of the file ``name`` is looked up by. A name
    over the fold limit is never folded: the index lists such a stored name
    under itself."""
    tidied = tidy_path(name)
    profile = host.fold_index_profile
    return tidied if profile is None else index_key_of(tidied, profile)


def file_request(host: DoorHost, requested: str) -> RequestName:
    """``requested`` as a file: a route's path, or a Bash token (which is not
    validated, so it may be absolute, climb out with ``..`` or run past the
    fold limit)."""
    tidied = tidy_path(requested)
    key = file_key(host, tidied)
    disk_spelling = _disk_spelling_of(host, tidied, key)
    policy = PolicyMatchContext(policy_profile_of(host), "file", disk_spelling)
    return RequestName(requested, "file", tidied, key, disk_spelling, policy)


def file_requests(host: DoorHost) -> Callable[[str], RequestName]:
    """:func:`file_request`, once per distinct name of one command: a disk
    spelling costs a resolver call, and the policy gate and the collapse of
    spellings by key both read each token's name."""
    known: dict[str, RequestName] = {}

    def name_of(requested: str) -> RequestName:
        name = known.get(requested)
        if name is None:
            name = known[requested] = file_request(host, requested)
        return name

    return name_of


def find_requested(host: DoorHost, name: RequestName) -> UUID | None:
    """The row the name reaches, or None. A read only."""
    if name.kind == "key":
        return host.registry.lookup_artifact_id_by_name(name.name, name_kind="key")
    return host.registry.lookup_artifact_id_by_name(
        name.name, key=name.key, disk_spelling=name.disk_spelling
    )


def resolve_requested(host: DoorHost, name: RequestName, content_hash: str) -> UUID:
    """:func:`find_requested`, registering the name at version 1 on a miss."""
    if name.kind == "key":
        return host.registry.resolve_or_register(name.name, content_hash, name_kind="key")
    return host.registry.resolve_or_register(
        name.name, content_hash, key=name.key, disk_spelling=name.disk_spelling
    )


def is_tracked_request(host: DoorHost, name: RequestName) -> bool:
    return host.policy.is_tracked(name.requested, name.policy)


def is_strict_request(host: DoorHost, name: RequestName) -> bool:
    return host.policy.is_strict_mode(name.requested, name.policy)


def _disk_spelling_of(host: DoorHost, tidied: str, key: str) -> str | None:
    """The disk's own spelling of a file request, relative to the root, on a
    disk that folds. A request through a link reaches another key, and so has
    none: the link's target is another file, whose name this request must not
    take. A spelling outside the root has none either."""
    profile = host.fold_index_profile
    if profile is None or profile == "none":
        return None
    spelling = _native_relative(host.canonical_root, tidied)
    if spelling is None or fold_key(spelling, profile) != key:
        return None
    return spelling


def _native_relative(root: str, tidied: str) -> str | None:
    """The platform resolver's spelling of ``tidied`` under ``root``,
    relative to it, or None when it lies outside the root or does not
    resolve: a dangling or looping link, a denied directory, a name too long
    or one a path cannot carry."""
    try:
        resolved = native_spelling(os.path.join(root, tidied))
    except (OSError, ValueError):
        return None
    spelling = os.path.relpath(resolved, root)
    return spelling if is_inside_root(spelling) else None
