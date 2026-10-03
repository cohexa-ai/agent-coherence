# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-workspace`` — checkpoint / list / restore / status (WV R6, R8).

Operator surface over the Workspace-Versioning engine
(:class:`ccs.adapters.workspace.WorkspaceVersioner`). The CLI drives an
IN-PROCESS :class:`~ccs.coordinator.service.CoordinatorService` over a durable
workspace-scoped SQLite registry (``<root>/.coherence/workspace.db``, retention
on) — pin legs run adapter-side and have no HTTP surface, so the whole verb set
runs in-process rather than through the hook coordinator's HTTP routes. The
db is OWNED by this CLI (the hook coordinator's ``state.db`` is untouched).

Verbs:

- ``checkpoint NAME --file P [--forward-only Q] [--no-pin]`` — capture a
  skew-declared cut over the named members and persist ONE manifest (pin legs
  on by default; ``--no-pin`` leaves the documented claimed-but-not-yet-backed
  ``(tier, unpinned)`` pair).
- ``list`` — every persisted checkpoint header.
- ``status ID`` — header + per-member honesty pairs ``(restore_tier,
  pin_state)``, torn-cut flags, restore outcomes.
- ``restore ID`` — drive the restore engine over the durable member rows;
  the per-member terminal report is the output, absorbing outcomes included.

File members ride a working-tree bridge (:class:`WorkingTreeSource`): one read
observes the disk bytes AND their version in the CLI's registry (first
observation mints v1; observed drift mints the next version with the bytes
retained), so the restore pointer and the fingerprint always describe one
observation and restore reads resolve through coordinator retention
(:class:`RetainedContentResolver`). The write leg is DETECTION-ONLY
(``no-arbiter``): a foreign edit is detected by hash/version comparison and
surfaced as a typed conflict, never presented as substrate arbitration.
S3 object members are captured/restored via the Python API (bindings carry
credentials the CLI cannot reconstruct) — ``status``/``list`` still render
them honestly.

Exit codes:
- 0: verb succeeded (restore: concluded with no absorbing/hold outcome)
- 1: not in a git repo / validation error (bad path, no members, pre-flight) /
     a typed coherence contention error (e.g. the observe-commit loop exhausted)
- 2: typed coherence refusal (binary member, unknown checkpoint, persist
     failure, member-path containment refusal — symlink component, hardlinked
     regular file (external co-owner), non-regular leaf (FIFO/socket/device),
     ``.coherence`` self-target, workspace escape, or an unreadable non-file
     member)
- 3: restore CONCLUDED but at least one member ended in an absorbing/hold
     outcome (``conflict`` / ``target_lost`` / ``held_unconfirmed``) or the
     registration was refused — the report on stdout carries the per-member truth
- 4: restore CONCLUDED clean by the codes above, but the operator asked
     (``restore --exit-nonzero-on-discarded-content``) for a run that put a
     member back over content the capture did not hold — or that holds no
     record of what it overwrote — to be a failure. Opt-in and never the
     default: the same run exits 0 without the flag, and both producers of 3
     take precedence over it. The flag is read after the engine returns, so it
     reports the write, it never prevents one.

A containment refusal is exit 2 on the CAPTURE leg (nothing persists). On the
RESTORE leg it is NOT: the engine's termination contract absorbs it into that
member's ``target_lost`` so the other members still conclude, and the refusal
surfaces as that member's outcome detail in the report — exit 3.

Under ``--json`` every error path ALSO emits a one-line JSON error envelope on
stdout (``{"kind": "error", "exit_code": N, "reason": ..., ...}`` — the
``coherence_replay`` envelope pattern) so machine consumers never have to parse
stderr prose; the human prose stays on stderr unchanged.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from ccs.adapters.claude_code.lifecycle import _ensure_coherence_dir
from ccs.adapters.claude_code.resolver import find_coordinator_root
from ccs.adapters.workspace import (
    BinaryFileMemberRefused,
    CheckpointPersistFailed,
    MemberRestoreOutcome,
    StructuralMemberRefused,
    WorkspaceCheckpoint,
    WorkspaceRestoreReport,
    WorkspaceVersioner,
)
from ccs.cli._coherence_client import err, normalize_workspace_path
from ccs.coordinator.registry_protocol import CheckpointMember, CheckpointRecord
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    PIN_STATE_UNPINNED,
    RESTORE_OBSERVATION_DIFFERS,
    RESTORE_OBSERVATION_NOT_RECORDED,
    RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE,
    RESTORE_OUTCOME_CONFLICT,
    RESTORE_OUTCOME_HELD_UNCONFIRMED,
    RESTORE_OUTCOME_TARGET_LOST,
    RESTORE_STATUS_CONCLUDED,
    WORKSPACE_REGISTRATION_REFUSED,
    CasVersionConflict,
    CheckpointRegistrationRefused,
    CheckpointUnknown,
    CoherenceError,
    CommitUnconfirmed,
)
from ccs.core.substrate import ArbitrationTier, RestoreTier, sha256_hex
from ccs.core.types import Artifact

_PROG = "agent-coherence-workspace"

#: The CLI's own durable registry, relative to the workspace root. Deliberately
#: NOT the hook coordinator's ``state.db``: pin legs and checkpoint manifests
#: are driven in-process by THIS CLI, and sharing a live server's db would
#: couple two writers the architecture keeps separate.
_WORKSPACE_DB_RELPATH = ".coherence/workspace.db"

#: Stable identity namespace: the checkpoint owner and the file-bridge writer
#: are minted per-workspace so re-invocations are the SAME principal.
_IDENTITY_NAMESPACE = uuid5(NAMESPACE_URL, "ccs-workspace-cli")

#: Honesty placement #2 (--help constraint note). Single tokens are asserted
#: by tests, so keep "single-host", "no-arbiter" and "detection-only" intact.
_SCOPE_NOTE = (
    "Scope: single-host coordinator only — checkpoints, pins and restore\n"
    "progress live in this workspace's local coordinator state and make NO\n"
    "cross-host claims. File members are detection-only ('no-arbiter'): a\n"
    "foreign edit racing a restore is detected adapter-locally and reported\n"
    "as a typed conflict, never arbitrated by the substrate. S3 object\n"
    "members are captured/restored via the Python API (see\n"
    "examples/workspace_versioning); this CLI still renders them honestly\n"
    "in 'status' and 'list'."
)

#: Honesty placement #4 (checkpoint output retention caveat).
FILE_RETENTION_CAVEAT = (
    "file members ride bounded coordinator retention (a K/T policy with no "
    "per-version hold in v1): a 'held' pin on a file member is a VERIFICATION "
    "of the retained bytes, not a retention guarantee — the tier stays "
    "restorable-unpinned, and an expired retention window surfaces as "
    "target_lost at restore time."
)

#: Honesty placement #3 (the claimed-but-not-yet-backed pair label).
CLAIMED_NOT_BACKED_LABEL = "claimed-but-not-yet-backed"

#: The exit-code-3 outcome set — CLI-PRIVATE and deliberately NOT
#: :data:`ccs.core.exceptions.RESTORE_ABSORBING_OUTCOMES` (DIFFERENT
#: membership: exit 3 includes ``held_unconfirmed`` — a HOLD, not an absorbing
#: outcome — and excludes ``forward_only_skipped``, a clean enumerated skip).
#: Named for what it decides so it cannot shadow the core vocabulary.
_EXIT3_OUTCOMES = frozenset(
    {
        RESTORE_OUTCOME_CONFLICT,
        RESTORE_OUTCOME_TARGET_LOST,
        RESTORE_OUTCOME_HELD_UNCONFIRMED,
    }
)

#: The observations that say this run put a checkpoint back over content the
#: capture did not hold, each mapped to the flag its member's line carries.
#: CLI-PRIVATE and named for what it decides, following :data:`_EXIT3_OUTCOMES`.
#:
#: ONE mapping, two consumers: the exit gate reads its keys and the human render
#: reads its values. They were two hand-maintained lists once, and a state that
#: fired the gate without earning a flag was exactly the defect that produced —
#: an operator handed a non-zero exit whose reason appeared nowhere in the
#: printed report. Adding a state here cannot now change one without the other.
#:
#: - ``observed_differs`` — a live state was read and it differed, so the write
#:   discarded content committed after the capture;
#: - ``present_not_comparable`` — the delete leg's probe established that live
#:   state EXISTED and destroyed it. A member captured ABSENT that is present
#:   live was created after the capture, so this discarded post-capture content
#:   even though the probe read no comparand to name it by;
#: - ``not_recorded`` — the run holds no observation, which is its own answer
#:   and never a clean one; a gate that passed it would report clean on exactly
#:   the runs that cannot say what they overwrote.
#:
#: ``no_live_state`` wrote onto nothing and ``no_write_attempted`` never
#: reached a write decision, so neither discarded anything, neither fires the
#: gate, and neither makes its line noisier.
_DISCARDED_CONTENT_FLAGS: Mapping[str, str] = {
    RESTORE_OBSERVATION_DIFFERS: "overwrote-differing-content",
    RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE: "destroyed-uncompared-content",
    RESTORE_OBSERVATION_NOT_RECORDED: "overwritten-content-not-recorded",
}

#: The exit-code-4 observation set, derived so it can never drift from the
#: flags above.
_DISCARDED_POST_CAPTURE_OBSERVATIONS = frozenset(_DISCARDED_CONTENT_FLAGS)

#: read_with_version's observe-commit loop bound: a racing second CLI process
#: can move the ledger between the lookup and the CAS; three attempts absorb
#: any realistic interleave without risking a livelock.
_OBSERVE_ATTEMPTS = 3

#: The coordinator's own state directory — never a workspace member (mirrors
#: ``ccs.mcp.uri._COHERENCE_DIR``; matched case-insensitively there and here).
_COHERENCE_DIR = ".coherence"

#: Envelope-local refusal slug for :class:`MemberPathRefused` — the
#: ``coherence_replay`` ``argument_error`` precedent, NOT a core wire constant
#: (no new vocabulary is minted in ``ccs.core.exceptions`` for a CLI-local
#: refusal).
MEMBER_PATH_REFUSED_REASON = "member_path_refused"


class MemberPathRefused(StructuralMemberRefused, ValueError):
    """Typed refusal: a member path failed containment validation at
    filesystem-access time (workspace escape, symlink component, hardlinked
    regular file, non-regular leaf, ``.coherence`` self-target, or an
    unreadable non-file member).

    Mirrors the MCP guard pattern (``ccs.mcp.uri._targets_coherence_state`` +
    ``_reject_path_escape``) at the :class:`WorkingTreeSource` seam. Keeps the
    stable message prefix ``member path ... refused:`` and the envelope-local
    slug :data:`MEMBER_PATH_REFUSED_REASON` as ``.reason`` — no core reason
    constant is minted for a CLI-local refusal.

    Two bases, each load-bearing:

    - :class:`~ccs.adapters.workspace.StructuralMemberRefused` — the ENGINE's
      vocabulary for "this source can never drive this member". The versioner's
      termination contract catches that base, so a refused member absorbs into
      ``target_lost`` and the restore CONCLUDES instead of wedging the
      checkpoint at ``in_progress`` forever (the engine never imports this
      module; the dependency points interface → engine, as it already did).
    - ``ValueError`` — the original base, kept so nothing that classified this
      as an argument-shaped refusal changes meaning.

    ``main`` maps it to exit 2 (a refusal, not caller misuse) and MUST keep
    that arm ahead of both the ``CoherenceError`` and the ``ValueError`` arms.
    A CAPTURE-time refusal still exits 2 unchanged; only the RESTORE leg
    absorbs, and the absorbed member surfaces in the report (exit 3) carrying
    this message as its outcome detail.
    """

    reason = MEMBER_PATH_REFUSED_REASON

    def __init__(self, member_path: str, detail: str) -> None:
        super().__init__(f"member path {member_path!r} refused: {detail}")
        self.member_path = member_path


def _is_within(candidate: str, root: str) -> bool:
    try:
        Path(candidate).relative_to(root)
        return True
    except ValueError:
        return False


def _validated_member_path(root: Path, path: str) -> Path:
    """Validate ``path`` against ``root`` at filesystem-access time; return the
    absolute target.

    Runs on EVERY ``read_with_version`` AND ``write_cas_at`` call — ``restore``
    replays PERSISTED member_paths that never re-pass the CLI-arg pre-check
    (``normalize_workspace_path``), so validating at parse time alone leaves
    the replay surface open. Mirrors ``ccs.mcp.uri``:

    - string-level: refuse empty/backslash, absolute, ``..`` traversal, and any
      path whose first normalized component is ``.coherence`` (the SQLite
      state, hook secret and pidfile are coordinator state, not members);
    - refuse EVERY symlink component INCLUDING the leaf — resolve-then-check
      alone leaves a swap-after-check TOCTOU window; refusing symlinks outright
      narrows the residual to the same stated same-uid window the MCP guard
      documents (the check-vs-open gap);
    - realpath containment + resolved-``.coherence`` re-check, belt-and-
      suspenders after the two checks above (a ``..``-free, symlink-free path
      cannot escape, but the mirror keeps the two guards from drifting);
    - refuse a NON-REGULAR leaf (FIFO, socket, block/char device) — opening one
      is not a bounded read (see :func:`_refuse_non_regular_stat`);
    - refuse a HARDLINKED regular-file leaf (``st_nlink > 1``) — a hard link is
      not a symlink and resolves in-tree, so it slips the two checks above, but
      it co-owns an external inode (see the inline note below).
    """
    if not path or "\\" in path:
        raise MemberPathRefused(path, "empty or backslash-carrying path")
    pure = Path(path)
    if pure.is_absolute():
        raise MemberPathRefused(path, "absolute paths are not workspace members")
    parts = pure.parts
    if any(part == ".." for part in parts):
        raise MemberPathRefused(path, "'..' traversal")
    if parts and parts[0].lower() == _COHERENCE_DIR:
        raise MemberPathRefused(
            path,
            "targets coordinator state (.coherence/**), which is never a "
            "workspace member",
        )
    candidate = root
    for part in parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise MemberPathRefused(
                path,
                f"component {part!r} is a symlink — a symlinked member can be "
                "re-pointed outside the workspace root between capture and "
                "restore",
            )
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(candidate)
    if not _is_within(target_real, root_real):
        raise MemberPathRefused(
            path, "escapes the workspace root (traversal or symlink)"
        )
    if _is_within(target_real, os.path.join(root_real, _COHERENCE_DIR)):
        raise MemberPathRefused(
            path, "resolves into coordinator state (.coherence/**)"
        )
    # HARDLINK defense: a regular-file member with a second directory entry
    # (``st_nlink > 1``) has an external co-owner. A hard link's
    # ``is_symlink()`` is False and ``realpath()`` resolves to this in-tree
    # path itself, so BOTH guards above pass — yet capturing it would read the
    # co-owned inode's bytes into ``workspace.db`` (an outside file, or the
    # ``.coherence/hook.secret`` HMAC secret hardlinked past the ``.coherence``
    # guard) and restoring would write through the shared inode, mutating the
    # outside file. ``follow_symlinks=False`` (an ``lstat``) because the leaf
    # is already proven non-symlink above; an absent/unstattable leaf is left
    # to the read leg (surfaces ABSENT) or the write leg (may create it fresh
    # at ``st_nlink == 1``). Directories carry ``st_nlink >= 2`` legitimately,
    # so :func:`_refuse_hardlinked_stat` gates on ``S_ISREG``.
    #
    # An ABSENT (or otherwise unstattable) leaf takes neither check: absence is
    # a FACT, not a refusal — the read leg surfaces it as the ABSENT member and
    # the write leg may create it fresh at ``st_nlink == 1``.
    try:
        leaf_stat = os.stat(candidate, follow_symlinks=False)
    except OSError:
        return candidate
    _refuse_non_regular_stat(leaf_stat, path)
    _refuse_hardlinked_stat(leaf_stat, path)
    return candidate


#: Chunk size for the O_NOFOLLOW-guarded member read.
_MEMBER_READ_CHUNK_BYTES = 1 << 20

#: Stable, distinct reason substring naming the v1 hardlink limitation. Asserted
#: by tests via the ``"hard link"`` token; kept separate from the symlink reason.
_HARDLINK_REFUSAL_DETAIL = (
    "is a hard link (st_nlink={nlinks} > 1) — a regular-file member with a "
    "second directory entry has an external co-owner; capturing it would read "
    "foreign bytes and restoring it would write through the shared inode, so it "
    "cannot be safely contained in v1 (hardlink limitation)"
)


def _refuse_hardlinked_stat(st: os.stat_result, member_path: str) -> None:
    """Raise :class:`MemberPathRefused` when ``st`` describes a regular file with
    more than one hard link (an external co-owner). No-op for directories,
    devices, or a single-link regular file (``st_nlink == 1``)."""
    if stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
        raise MemberPathRefused(
            member_path, _HARDLINK_REFUSAL_DETAIL.format(nlinks=st.st_nlink)
        )


#: Stable, distinct reason substring naming the v1 non-regular-file limitation.
#: Asserted by tests via the ``"not a regular file"`` token; kept separate from
#: the symlink and hard-link reasons.
_NON_REGULAR_REFUSAL_DETAIL = (
    "is not a regular file ({kind}) — a workspace member must be a bounded "
    "byte sequence: opening a FIFO or socket blocks until a peer appears "
    "(wedging the capture forever, with no timeout this process controls) and "
    "a device node is not workspace state at all, so only regular files can "
    "be members in v1 (non-regular-file limitation)"
)

#: The leaf shapes refused by :func:`_refuse_non_regular_stat`, most-specific
#: name first. Directories are deliberately ABSENT: they stat as non-regular
#: but already route to the established ``IsADirectoryError`` handling, which
#: names the problem better than a generic non-regular refusal would.
_NON_REGULAR_KINDS: tuple[tuple[Any, str], ...] = (
    (stat.S_ISFIFO, "FIFO / named pipe"),
    (stat.S_ISSOCK, "socket"),
    (stat.S_ISBLK, "block device"),
    (stat.S_ISCHR, "character device"),
)


def _refuse_non_regular_stat(st: os.stat_result, member_path: str) -> None:
    """Raise :class:`MemberPathRefused` when ``st`` describes a FIFO, socket, or
    block/character device.

    The gate that keeps the guarded opens BOUNDED. ``os.open`` on a FIFO with no
    peer blocks in the kernel until one appears — before ``O_NOFOLLOW``, before
    the fd-level hardlink re-check, before any timeout this process controls —
    so ``checkpoint --file pipe.txt`` hangs forever rather than refusing. Device
    nodes are the same shape of wrong: unbounded or side-effecting reads of
    something that is not workspace state.

    No-op for a regular file and for a DIRECTORY (already handled: the guarded
    open surfaces ``IsADirectoryError``, mapped to a typed refusal naming the
    directory). Symlinks never reach here — they are refused component-by-
    component above."""
    for predicate, kind in _NON_REGULAR_KINDS:
        if predicate(st.st_mode):
            raise MemberPathRefused(
                member_path, _NON_REGULAR_REFUSAL_DETAIL.format(kind=kind)
            )


def _read_member_bytes_nofollow(candidate: Path, member_path: str) -> bytes:
    """Read the leaf through ``O_RDONLY | O_NOFOLLOW`` and re-take the hardlink
    check on the open fd.

    Narrows the check-vs-open TOCTOU that ``_validated_member_path`` documents:
    a leaf swapped to a symlink after validation is rejected ATOMICALLY at open
    (``O_NOFOLLOW`` → ``OSError`` ELOOP), and a hard link swapped in after
    validation is caught by the ``os.fstat`` re-check on the very fd being read.
    ``O_NONBLOCK`` closes the same window for a leaf swapped to a FIFO (the open
    would otherwise block in the kernel until a peer writer appears, ahead of
    every check this function makes); it is a no-op on regular files, and the
    fd-level :func:`_refuse_non_regular_stat` then refuses the swapped-in leaf
    before any read.
    ``FileNotFoundError`` (the ABSENT fact) and every other ``OSError`` (ELOOP,
    EISDIR, EACCES, …) propagate for the caller to map; :class:`MemberPathRefused`
    is raised only for the fd-level hardlink / non-regular cases."""
    fd = os.open(candidate, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        leaf_stat = os.fstat(fd)
        _refuse_non_regular_stat(leaf_stat, member_path)
        _refuse_hardlinked_stat(leaf_stat, member_path)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, _MEMBER_READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _write_member_bytes_nofollow(candidate: Path, member_path: str, data: bytes) -> None:
    """Write the leaf through ``O_WRONLY | O_CREAT | O_NOFOLLOW`` (mode 0600).

    Deliberately NOT ``O_TRUNC``: the fd-level hardlink re-check MUST run before
    any truncation, so an outside inode co-owned via a hard link is never
    mutated — refuse first, then ``ftruncate`` + write. A leaf swapped to a
    symlink after validation is rejected atomically at open (``O_NOFOLLOW`` →
    ELOOP), so a restore write never follows a re-pointed leaf outside the root.
    ``O_NONBLOCK`` (a no-op on regular files) keeps the open itself bounded when
    the leaf was swapped to a FIFO after validation — it fails ENXIO instead of
    blocking on a missing reader — and the fd-level
    :func:`_refuse_non_regular_stat` refuses any other non-regular shape BEFORE
    the truncate, alongside the hardlink re-check.
    ``0o600`` applies only on CREATE; an existing member keeps its own mode."""
    fd = os.open(
        candidate, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600
    )
    try:
        leaf_stat = os.fstat(fd)
        _refuse_non_regular_stat(leaf_stat, member_path)
        _refuse_hardlinked_stat(leaf_stat, member_path)
        os.ftruncate(fd, 0)
        written = 0
        while written < len(data):
            n = os.write(fd, data[written:])
            if n == 0:  # defensive — a regular-file write always progresses
                break
            written += n
    finally:
        os.close(fd)


def _member_access_refusal_detail(exc: OSError) -> str:
    """Human detail for an ``OSError`` from the guarded member open (capture leg)."""
    if exc.errno == errno.ELOOP:
        return (
            "leaf became a symlink after validation (O_NOFOLLOW rejected it at "
            "open — the check-vs-open TOCTOU window); a symlinked leaf can be "
            "re-pointed outside the workspace root between capture and restore"
        )
    return f"not a readable regular file ({type(exc).__name__})"


# --- the working-tree bridge (FileMemberSource + FileRestoreTarget) -------------


class WorkingTreeSource:
    """Disk files bridged to the CLI's registry — the file-member binding.

    ``read_with_version`` returns ``(bytes, version)`` from ONE observation:
    the disk bytes are read, then the registry row for ``path`` is resolved —
    first observation mints v1 with the bytes retained; observed drift (disk
    hash != ledger hash) mints the next version via the registry's OCC
    ``commit_cas``, again with the bytes retained. The returned version's
    retained content therefore always hashes to the returned bytes, which is
    what makes :class:`RetainedContentResolver` a REAL resolver rather than a
    cache.

    ``write_cas_at`` is the restore leg: DETECTION-ONLY (``no-arbiter``). It
    refuses (typed :class:`~ccs.core.exceptions.CasVersionConflict`) when the
    ledger moved past ``expected_version`` OR when the disk bytes no longer
    hash to the ledger's content at ``expected_version`` (a foreign edit the
    ledger has not observed). A win writes the disk FIRST, then commits the
    ledger — a crash between the two leaves the disk restored and the ledger
    behind, which the next read re-observes (self-healing in the safe
    direction). Documented v1 residuals: a delete landing between the read and
    the CAS is recreated undetected, and two concurrent CLI processes could
    both pass detection (single-writer CLI assumption).

    Non-UTF-8 disk bytes are returned with version 0 (pointer UNCONFIRMED) and
    are never minted into the ledger — the versioner's typed
    :class:`~ccs.adapters.workspace.BinaryFileMemberRefused` fires at capture.

    Containment TOCTOU (honest residual). ``_validated_member_path`` refuses
    symlink components, non-regular leaves (FIFO, socket, block/char device) and
    hardlinked regular files at validation time, but the validation and the
    syscall are separate steps. Both legs close the LEAF window: the read/write
    go through ``os.open(..., O_NOFOLLOW | O_NONBLOCK)`` (a leaf swapped to a
    symlink after validation is rejected atomically at open with ELOOP; a leaf
    swapped to a FIFO cannot block the open waiting for a peer) with an
    ``os.fstat`` re-check on the very fd for BOTH the hardlink and the
    non-regular shapes (either one swapped in after validation is caught too,
    before any byte is read or truncated). The INTERMEDIATE-DIRECTORY-
    COMPONENT window is NOT closed: an attacker who swaps a mid-path directory
    for a symlink between the walk and the open can still redirect the target,
    because the open resolves the full path afresh. Full closure needs an
    ``openat``/``dirfd`` component-by-component walk (``O_NOFOLLOW`` at every
    hop), which is out of v1 scope. The safe direction is to migrate both legs
    to a dirfd walk rooted at the workspace root so no intermediate component is
    ever re-resolved by name. The trust model remains single-uid, single-host.
    """

    def __init__(self, root: Path, registry: SqliteArtifactRegistry, agent: UUID) -> None:
        self._root = root
        self._registry = registry
        self._agent = agent

    def _abs(self, path: str) -> Path:
        # Containment validation on EVERY filesystem access (both the read and
        # the write leg route through here) — restore replays persisted
        # member_paths that never re-pass the CLI-arg pre-check.
        return _validated_member_path(self._root, path)

    def read_with_version(self, path: str) -> tuple[bytes, int]:
        candidate = self._abs(path)
        try:
            data = _read_member_bytes_nofollow(candidate, path)
        except FileNotFoundError:
            raise  # the ABSENT fact — the engine's vocabulary, not an error
        except MemberPathRefused:
            raise  # hardlink swapped in post-validation (fd re-check)
        except OSError as exc:
            # ELOOP (leaf swapped to a symlink after validation — the guarded
            # open rejects it atomically), a directory member (IsADirectoryError),
            # a mid-path non-dir (NotADirectoryError), permissions, ... — a typed
            # refusal at the capture seam, never a raw traceback.
            raise MemberPathRefused(
                path, _member_access_refusal_detail(exc)
            ) from exc
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            # Never minted: the version stays UNCONFIRMED (0) and capture
            # refuses the member with the typed binary refusal.
            return (data, 0)
        digest = sha256_hex(data)
        for _ in range(_OBSERVE_ATTEMPTS):
            artifact_id = self._registry.lookup_artifact_id_by_name(path)
            if artifact_id is None:
                artifact = Artifact(name=path, version=1, content_hash=digest)
                self._registry.register_artifact(artifact, text)
                return (data, 1)
            current = self._registry.get_artifact(artifact_id)
            if current is None:  # pragma: no cover — removed between the two reads
                continue
            if current.content_hash == digest:
                return (data, current.version)
            result = self._registry.commit_cas(
                artifact_id,
                self._agent,
                expected_version=current.version,
                content_hash=digest,
                content=text,
            )
            if isinstance(result, tuple):
                return (data, result[0].version)
            # ConflictDetail: another observer moved the ledger — re-resolve.
        raise CoherenceError(
            f"file member {path!r}: could not record the observed content "
            f"state after {_OBSERVE_ATTEMPTS} attempts (another writer keeps "
            "moving the workspace ledger)"
        )

    def write_cas_at(self, path: str, expected_version: int, new_content: bytes) -> None:
        text = new_content.decode("utf-8")  # captured members are UTF-8 by the typed refusal
        artifact_id = self._registry.lookup_artifact_id_by_name(path)
        current = (
            self._registry.get_artifact(artifact_id) if artifact_id is not None else None
        )
        if artifact_id is None or current is None or current.version != expected_version:
            raise CasVersionConflict(
                path, expected_version, current.version if current is not None else 0
            )
        live = self._abs(path)  # validates + refuses a hardlinked member (exit 2)
        try:
            disk = _read_member_bytes_nofollow(live, path)
        except FileNotFoundError:
            disk = None
        # A hardlink swapped in at the fd raises MemberPathRefused (exit 2). An
        # ELOOP/EISDIR/... from the guarded open propagates as an OSError so the
        # restore termination boundary (_drive_member_absorbing) absorbs it into
        # an honest ``target_lost`` outcome — never a raw traceback, and the disk
        # is never followed through a swapped-in leaf symlink.
        if disk is not None and sha256_hex(disk) != current.content_hash:
            # DETECTION fired: the disk moved without the ledger observing it —
            # a foreign edit. no-arbiter: typed conflict, never arbitration.
            raise CasVersionConflict(path, expected_version, expected_version)
        live.parent.mkdir(parents=True, exist_ok=True)
        # disk first: the safe crash direction. The guarded open (O_NOFOLLOW +
        # fd hardlink re-check, truncate only AFTER the refusal gate) means a
        # leaf swapped to a symlink or a hardlinked co-owner is caught before any
        # bytes land on the outside inode.
        _write_member_bytes_nofollow(live, path, new_content)
        result = self._registry.commit_cas(
            artifact_id,
            self._agent,
            expected_version=expected_version,
            content_hash=sha256_hex(new_content),
            content=text,
        )
        if not isinstance(result, tuple):
            raise CommitUnconfirmed(
                f"file member {path!r}: the restored bytes landed on disk but "
                f"the version-CAS ledger commit was refused ({result!r}) — "
                "outcome unconfirmed, never best-effort"
            )


class RetainedContentResolver:
    """``FileContentResolver`` over the CLI registry's retention (real, not a cache)."""

    def __init__(self, registry: SqliteArtifactRegistry) -> None:
        self._registry = registry

    def content_at(self, member_path: str, version: int) -> bytes:
        artifact_id = self._registry.lookup_artifact_id_by_name(member_path)
        if artifact_id is None:
            raise KeyError(member_path)
        body = self._registry.get_content_at_version(artifact_id, version)
        if body is None:
            raise KeyError((member_path, version))
        return body.encode("utf-8") if isinstance(body, str) else bytes(body)


# --- parser ----------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Workspace checkpoint / list / restore / status over heterogeneous "
            "members (files + S3 objects + declared forward-only surfaces)."
        ),
        epilog=_SCOPE_NOTE,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override the workspace root (default: walk up from cwd to git root).",
    )
    common.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-parseable JSON instead of the human rendering.",
    )
    sub = parser.add_subparsers(dest="verb", required=True)

    p_ckpt = sub.add_parser(
        "checkpoint",
        parents=[common],
        help="Capture a named checkpoint over the declared members.",
        description=(
            "Capture a skew-declared cut over the named members and persist ONE "
            "manifest. Pin legs run by default; file-member pins are "
            "verification-only (see the retention caveat printed with the result)."
        ),
    )
    p_ckpt.add_argument("name", help="Checkpoint name (non-empty; not an id).")
    p_ckpt.add_argument(
        "--file",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Declare one file member (repeatable). Workspace-relative or an "
            "absolute path inside the root; no '..' traversal. Non-UTF-8 files "
            "are refused with a typed error (v1 limitation)."
        ),
    )
    p_ckpt.add_argument(
        "--forward-only",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            "Declare one forward-only member (repeatable): enumerated in the "
            "manifest, skipped at restore — an action/effect surface with no "
            "state to capture."
        ),
    )
    p_ckpt.add_argument(
        "--no-pin",
        action="store_true",
        help=(
            "Capture only — skip the pin legs. Members then persist the "
            "documented claimed-but-not-yet-backed (tier, unpinned) pair; "
            "'agent-coherence-workspace status' renders it explicitly."
        ),
    )

    sub.add_parser(
        "list",
        parents=[common],
        help="List every persisted checkpoint header.",
    )

    p_status = sub.add_parser(
        "status",
        parents=[common],
        help="Show one checkpoint's header + per-member honesty pairs.",
        description=(
            "Every member is labeled with its (restore_tier, pin_state) pair — "
            "(restorable, unpinned) is rendered as claimed-but-not-yet-backed — "
            "plus torn-cut flags and restore outcomes when present."
        ),
    )
    p_status.add_argument("checkpoint_id", help="The persisted checkpoint id.")

    p_restore = sub.add_parser(
        "restore",
        parents=[common],
        help="Restore a checkpoint (per-member terminal report).",
        description=(
            "Drive one conditional leg per durable member row under the "
            "termination contract. Absorbing outcomes (conflict / target_lost / "
            "held_unconfirmed) are REPORT content: the restore still concludes "
            "and the exit code distinguishes a restore with no such outcome "
            "(0) from a concluded-with-absorbed-outcomes one (3). Exit 0 is "
            "not a claim that nothing was overwritten: a restore that put a "
            "member back over content committed after the capture says so per "
            "member in the report and still exits 0 unless the flag below "
            "asks for its own code."
        ),
    )
    p_restore.add_argument("checkpoint_id", help="The persisted checkpoint id.")
    # Named for what it changes — the exit code — and nothing else. A fencing
    # name would promise a refusal this cannot deliver: the flag is read after
    # the engine returns, when every member has already been written and the
    # registration has already run.
    p_restore.add_argument(
        "--exit-nonzero-on-discarded-content",
        action="store_true",
        help=(
            "Exit 4 when the report says this run put a member back over "
            "content the checkpoint did not hold, or holds no record of what "
            "it overwrote. Read after every member has already been written, "
            "so it reports what happened and cannot prevent the write. A "
            "restore that also ends with an absorbing/hold outcome or a "
            "refused registration still exits 3."
        ),
    )
    return parser


# --- rendering helpers -----------------------------------------------------------


def _member_payload(row: CheckpointMember) -> dict[str, Any]:
    return {
        "member_path": row.member_path,
        "restore_tier": row.restore_tier,
        "pin_state": row.pin_state,
        "pair": f"({row.restore_tier}, {row.pin_state})",
        "claimed_not_backed": _claimed_not_backed(row),
        "absent": row.absent,
        "dirty_during_window": row.dirty_during_window,
        "arbitration_tier": row.arbitration_tier,
        "native_token": row.native_token,
        "fingerprint": row.fingerprint,
        "restore_outcome": row.restore_outcome,
        "deleted_at_restore": row.deleted_at_restore,
    }


def _claimed_not_backed(row: CheckpointMember) -> bool:
    return (
        row.restore_tier == RestoreTier.RESTORABLE.value
        and row.pin_state == PIN_STATE_UNPINNED
    )


def _member_line(row: CheckpointMember) -> str:
    """One member's honesty line: the (tier, pin) PAIR plus every flag present."""
    bits = [f"  {row.member_path}  ({row.restore_tier}, {row.pin_state})"]
    if _claimed_not_backed(row):
        # Honesty placement #3: (restorable, unpinned) is a CLAIM nothing backs
        # yet — never rendered as a plain restorable state.
        bits.append(CLAIMED_NOT_BACKED_LABEL)
    if row.absent:
        bits.append("ABSENT-at-capture")
    if row.dirty_during_window:
        bits.append("dirty-during-window")
    if row.restore_outcome is not None:
        bits.append(f"outcome={row.restore_outcome}")
    if row.deleted_at_restore is not None:
        bits.append("deleted-at-restore")
    return "  ".join(bits)


def _restore_outcome_line(outcome: MemberRestoreOutcome) -> str:
    """One restored member's summary line (the detail prose rides below it).

    Every observation the exit gate fires on earns a flag, in the shape
    ``resumed-from-prior-run`` already set, and it earns it by lookup rather
    than by a branch: both this render and the gate read
    :data:`_DISCARDED_CONTENT_FLAGS`, so a state cannot fire one and not the
    other. The two states absent from that mapping are the two the gate does
    not fire on — a write that landed on nothing discarded nothing, and a
    member whose leg never reached a write decision must not be made noisier.
    Both render exactly as they did before the observation existed.

    The differing case is annotated whatever the differing content WAS. The
    engine reads one live state and cannot separate a half-written file from a
    peer's committed work, so a rule that flagged only the second would be
    guessing; the flag reports what was seen, and the operator decides.
    """
    line = (
        f"  {outcome.member_path}  outcome={outcome.outcome}  "
        f"attempts={outcome.attempts}"
    )
    if outcome.resumed_from_prior_run:
        line += "  resumed-from-prior-run"
    observation = outcome.observation
    flag = _DISCARDED_CONTENT_FLAGS.get(observation.state)
    if flag is not None:
        line += f"  {flag}"
    if observation.state == RESTORE_OBSERVATION_DIFFERS and observation.pointer is not None:
        # Only this state MAY carry a pointer, and it is not guaranteed to (a
        # source can land a write without naming a version), so the version is
        # appended only when there is one rather than printed as an empty
        # claim. The digest stays off this summary line and in the JSON: 64
        # hex characters per member would bury the line it rides on.
        line += f"  overwritten-version={observation.pointer}"
    return line


def _record_payload(record: CheckpointRecord) -> dict[str, Any]:
    return {
        "checkpoint_id": record.checkpoint_id,
        "name": record.name,
        "restore_status": record.restore_status,
        "pin_refcount": record.pin_refcount,
        "window_min": record.window_min,
        "window_max": record.window_max,
        "created_at_tick": record.created_at_tick,
    }


def _outcome_payload(outcome: MemberRestoreOutcome) -> dict[str, Any]:
    return {
        "member_path": outcome.member_path,
        "outcome": outcome.outcome,
        "attempts": outcome.attempts,
        "detail": outcome.detail,
        "new_native_token": outcome.new_native_token,
        "deleted_at_restore": outcome.deleted_at_restore,
        "resumed_from_prior_run": outcome.resumed_from_prior_run,
        # Nested, mirroring how the ``registration`` block renders its own
        # companion dataclass as one keyed object: the three values are read
        # together and only mean anything together. Flat keys would also put
        # ``fingerprint`` and a pointer beside ``status``'s member payload
        # spellings, where they name the CAPTURED digest and token — the same
        # words for the opposite state. Under this key they unambiguously
        # describe what the restore overwrote. The block is always present,
        # so a consumer never has to read its absence as clean.
        "observation": {
            "state": outcome.observation.state,
            "pointer": outcome.observation.pointer,
            "fingerprint": outcome.observation.fingerprint,
        },
    }


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def _emit_error_envelope(
    args: argparse.Namespace,
    *,
    exit_code: int,
    reason: str,
    message: str,
    exc: BaseException | None = None,
) -> None:
    """Under ``--json``, emit the one-line JSON error envelope on stdout.

    Mirrors ``ccs.cli.coherence_replay``'s ``emit_json_error_envelope``
    pattern: stdout stays self-contained for machine consumers on EVERY exit
    path while the human prose stays on stderr. ``reason`` carries the typed
    ``.reason`` when the raising exception has one; otherwise an
    envelope-local slug (the ``argument_error`` precedent)."""
    if not getattr(args, "json", False):
        return
    print(
        json.dumps(
            {
                "kind": "error",
                "exit_code": exit_code,
                "reason": reason,
                "exception": type(exc).__name__ if exc is not None else None,
                "message": message,
            }
        ),
        flush=True,
    )


def _live_coordinator_pid(root: Path) -> int | None:
    """The pid of a live hook coordinator on this workspace, else ``None``.

    Mirrors the lifecycle module's pidfile discipline: line 1 of
    ``<root>/.coherence/server.pid`` is the holder's pid; line 2 (the port) may
    be legitimately absent while idle (``_rewrite_pidfile_drop_port``), so
    liveness is pid-parses AND process-alive (signal 0) — never port-present.
    """
    pid_file = root / _COHERENCE_DIR / "server.pid"
    try:
        first_line = pid_file.read_text(encoding="utf-8").splitlines()[0]
        pid = int(first_line.strip())
    except (OSError, IndexError, ValueError):
        return None
    if pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return pid  # alive under another uid — still a live coordinator
    except OSError:
        return None
    return pid


def _live_coordinator_warning(pid: int) -> str:
    return (
        f"{_PROG}: WARNING: a live hook coordinator (pid {pid}, "
        ".coherence/server.pid) is serving this workspace — restored file "
        "writes land on disk OUTSIDE the live coordinator's grant flow and "
        "bypass its grants, so peers may keep now-stale views. Stop the "
        "coordinator before restoring, or have agents re-read affected "
        "members after it."
    )


# --- the in-process stack --------------------------------------------------------


class _Stack:
    """One verb invocation's in-process service stack (context-managed)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        # KTD-13: create .coherence/ via the documented single implementation
        # (0700 perms + the '*' gitignore). A bare mkdir here would race ahead
        # of SqliteArtifactRegistry's protective mkdir(mode=0o700) (turning it
        # into a no-op): under umask 022 a fresh workspace would get a 0755
        # .coherence/ with NO .gitignore, and `git add .` would stage
        # workspace.db.
        coherence_dir = _ensure_coherence_dir(root)
        if coherence_dir is None:
            raise CoherenceError(
                f"cannot create {root / _COHERENCE_DIR} — the workspace "
                "registry needs a writable .coherence directory (read-only "
                "workspace?)"
            )
        db_path = root / _WORKSPACE_DB_RELPATH
        # Retention ON: the file-member resolver reads captured bytes back
        # through get_content_at_version — without it every file restore
        # would honestly (and uselessly) report target_lost.
        self.registry = SqliteArtifactRegistry(db_path, retain_versions=True)
        self.service = CoordinatorService(self.registry)
        self.owner = uuid5(_IDENTITY_NAMESPACE, f"owner:{root}")
        self.source = WorkingTreeSource(
            root, self.registry, uuid5(_IDENTITY_NAMESPACE, f"file-bridge:{root}")
        )
        self.resolver = RetainedContentResolver(self.registry)

    def versioner(self) -> WorkspaceVersioner:
        return WorkspaceVersioner(
            service=self.service, owner=self.owner, file_resolver=self.resolver
        )

    def close(self) -> None:
        self.registry.close()


# --- verbs -----------------------------------------------------------------------


def _cmd_checkpoint(stack: _Stack, args: argparse.Namespace) -> int:
    invalid: list[tuple[str, str]] = []
    files: list[str] = []
    for raw in args.file:
        normalized, reason = normalize_workspace_path(raw, stack.root)
        if reason is not None:
            invalid.append((raw, reason))
        else:
            files.append(normalized)
    if invalid:
        for raw, reason in invalid:
            err(f"{_PROG}: rejected {raw!r}: {reason}")
        _emit_error_envelope(
            args,
            exit_code=1,
            reason="invalid_member_path",
            message="; ".join(f"{raw!r}: {reason}" for raw, reason in invalid),
        )
        return 1
    if not files and not args.forward_only:
        message = (
            "checkpoint needs at least one member (--file and/or --forward-only)"
        )
        err(f"{_PROG}: {message}")
        _emit_error_envelope(args, exit_code=1, reason="no_members", message=message)
        return 1

    # Dup-name disclosure: names are labels, NOT unique keys — a second
    # checkpoint under an existing name is minted, never refused, but the
    # prior ids are disclosed so the operator is not silently ambiguous.
    prior_same_name = [
        record.checkpoint_id
        for record in stack.service.list_workspace_checkpoints()
        if record.name == args.name
    ]

    versioner = stack.versioner()
    for path in files:
        versioner.add_file_member(stack.source, path)
    for name in args.forward_only:
        versioner.add_forward_only_member(name)
    result: WorkspaceCheckpoint = versioner.checkpoint(args.name, pin=not args.no_pin)

    if args.json:
        payload: dict[str, Any] = {
            "checkpoint": _record_payload(result.record),
            "members": [_member_payload(row) for row in result.members],
            "retention_caveat": FILE_RETENTION_CAVEAT,
        }
        if prior_same_name:
            payload["existing_with_same_name"] = prior_same_name
        _print_json(payload)
        return 0
    record = result.record
    print(f"checkpoint {record.name!r} persisted: {record.checkpoint_id}")
    if prior_same_name:
        print(
            f"note: {len(prior_same_name)} prior checkpoint(s) already carry "
            f"the name {record.name!r}: {', '.join(prior_same_name)} — names "
            "are labels, not unique keys; status/restore target an id"
        )
    print(
        f"window: [{record.window_min:.0f}, {record.window_max:.0f}] "
        "(monotonic seconds; the skew is declared, not hidden)"
    )
    print("members:")
    for row in result.members:
        print(_member_line(row))
    # Honesty placement #4: the retention caveat rides EVERY checkpoint output.
    print(f"retention caveat: {FILE_RETENTION_CAVEAT}")
    return 0


def _cmd_list(stack: _Stack, args: argparse.Namespace) -> int:
    records = stack.service.list_workspace_checkpoints()
    if args.json:
        _print_json({"checkpoints": [_record_payload(r) for r in records]})
        return 0
    if not records:
        print("no checkpoints persisted for this workspace")
        return 0
    for record in records:
        print(
            f"{record.checkpoint_id}  {record.name!r}  "
            f"restore_status={record.restore_status}  "
            f"pin_refcount={record.pin_refcount}"
        )
    return 0


def _cmd_status(stack: _Stack, args: argparse.Namespace) -> int:
    record = stack.service.get_workspace_checkpoint(args.checkpoint_id)
    if record is None:
        raise CheckpointUnknown(args.checkpoint_id)
    members = stack.service.get_workspace_checkpoint_members(args.checkpoint_id)
    if args.json:
        _print_json(
            {
                "checkpoint": _record_payload(record),
                "members": [_member_payload(row) for row in members],
            }
        )
        return 0
    print(f"checkpoint {record.checkpoint_id}  {record.name!r}")
    # Honesty placement #3 (header half): restore_status + pin_refcount.
    print(
        f"restore_status={record.restore_status}  pin_refcount={record.pin_refcount}"
    )
    print("members:  (restore_tier, pin_state) per member")
    for row in members:
        print(_member_line(row))
    return 0


def _discarded_post_capture_content(outcome: MemberRestoreOutcome) -> bool:
    """Did this member's restore put the checkpoint back over content the
    capture did not hold?

    The opt-in exit gate's whole decision, over the member outcome alone — it
    reads no rendered payload, because the exit code is computed whether or not
    ``--json`` was asked for. Kept a named function rather than an inline
    comprehension so the two states the command line cannot reach are still
    decidable directly: ``restore`` refuses a checkpoint carrying a pending
    object member before the engine runs, so only the file leg's states arrive
    here end to end, while the delete leg's ``present_not_comparable`` and the
    create-on-absent ``no_live_state`` are covered at the engine level and
    through this predicate.
    """
    return outcome.observation.state in _DISCARDED_POST_CAPTURE_OBSERVATIONS


def _cmd_restore(stack: _Stack, args: argparse.Namespace) -> int:
    record = stack.service.get_workspace_checkpoint(args.checkpoint_id)
    if record is None:
        raise CheckpointUnknown(args.checkpoint_id)
    rows = stack.service.get_workspace_checkpoint_members(args.checkpoint_id)
    versioner = stack.versioner()
    undrivable = [
        row.member_path
        for row in rows
        if row.restore_outcome is None
        and row.arbitration_tier == ArbitrationTier.NATIVE_CAS.value
    ]
    if undrivable and record.restore_status != RESTORE_STATUS_CONCLUDED:
        message = (
            f"cannot restore {args.checkpoint_id}: S3 object members "
            f"({', '.join(sorted(undrivable))}) need their bindings/credentials, "
            "which live in the Python API (see examples/workspace_versioning) — "
            "nothing was started"
        )
        err(f"{_PROG}: {message}")
        _emit_error_envelope(
            args, exit_code=1, reason="undrivable_members", message=message
        )
        return 1
    bound_file_members = 0
    for row in rows:
        if row.arbitration_tier != ArbitrationTier.NO_ARBITER.value:
            continue
        needs_binding = row.absent or row.restore_tier != RestoreTier.FORWARD_ONLY.value
        if needs_binding:
            versioner.add_file_member(stack.source, row.member_path)
            bound_file_members += 1
    # Live-coordinator dual-write hazard (v1 remediation: WARN, never refuse).
    # Restored file writes land outside a live hook coordinator's grant flow;
    # skipped when this restore binds no file members (nothing will be written
    # through the working-tree bridge).
    live_pid = _live_coordinator_pid(stack.root) if bound_file_members else None
    if live_pid is not None:
        err(_live_coordinator_warning(live_pid))
    report: WorkspaceRestoreReport = versioner.restore(args.checkpoint_id)

    absorbed = [m for m in report.members if m.outcome in _EXIT3_OUTCOMES]
    registration_refused = (
        report.registration is not None
        and report.registration.status == WORKSPACE_REGISTRATION_REFUSED
    )
    rc = 3 if absorbed or registration_refused else 0
    # The operator's opt-in gate, evaluated ONLY on a run that would otherwise
    # exit 0: both of exit 3's producers — an absorbing/hold outcome and a
    # refused registration — take precedence, so a consumer reading 3 still
    # reads exactly what it always did.
    if rc == 0 and args.exit_nonzero_on_discarded_content:
        if any(_discarded_post_capture_content(m) for m in report.members):
            rc = 4

    if args.json:
        payload: dict[str, Any] = {
            "checkpoint_id": report.checkpoint_id,
            "status": report.status,
            "members": [_outcome_payload(m) for m in report.members],
        }
        if live_pid is not None:
            payload["live_coordinator_warning"] = True
        if report.registration is not None:
            payload["registration"] = {
                "status": report.registration.status,
                "detail": report.registration.detail,
                "registered": dict(report.registration.registered),
                "skipped": list(report.registration.skipped),
                "substrate_registered": list(report.registration.substrate_registered),
                "deleted_recorded": list(report.registration.deleted_recorded),
                "refused": dict(report.registration.refused),
                "invalidated_peers": report.registration.invalidated_peers,
                "attempts": report.registration.attempts,
            }
        _print_json(payload)
        return rc

    print(f"restore of checkpoint {report.checkpoint_id} {report.status.upper()}")
    print("members:")
    for outcome in report.members:
        print(_restore_outcome_line(outcome))
        print(f"    {outcome.detail}")
    if report.registration is not None:
        print(
            f"registration: {report.registration.status} — {report.registration.detail}"
        )
        print(
            f"  peers invalidated by registration: "
            f"{report.registration.invalidated_peers}  "
            f"(attempts={report.registration.attempts})"
        )
    if absorbed:
        print(
            "note: absorbed outcomes above are the honest per-member truth — "
            "the restore concluded, it did not silently succeed."
        )
    return rc


# --- entry point -----------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root_arg = args.root if args.root is not None else find_coordinator_root()
    if root_arg is None:
        message = "not in a git repository"
        err(f"{_PROG}: {message}")
        _emit_error_envelope(
            args, exit_code=1, reason="not_a_git_repository", message=message
        )
        return 1
    root = Path(root_arg).resolve()

    stack: _Stack | None = None
    try:
        stack = _Stack(root)
        if args.verb == "checkpoint":
            return _cmd_checkpoint(stack, args)
        if args.verb == "list":
            return _cmd_list(stack, args)
        if args.verb == "status":
            return _cmd_status(stack, args)
        return _cmd_restore(stack, args)
    except MemberPathRefused as exc:
        # Both a StructuralMemberRefused (→ CoherenceError) and a ValueError —
        # so this arm MUST precede BOTH of those arms below: a containment
        # refusal is exit 2 (a typed refusal), not caller misuse and not the
        # generic exit-1 coherence-error path. Reached on the CAPTURE leg (and
        # any refusal raised outside a restore leg); a refusal raised INSIDE a
        # restore leg never arrives here — the engine's termination contract
        # absorbs it into that member's target_lost and the run concludes.
        err(f"{_PROG}: refused ({exc.reason}): {exc}")
        _emit_error_envelope(
            args, exit_code=2, reason=exc.reason, message=str(exc), exc=exc
        )
        return 2
    except BinaryFileMemberRefused as exc:
        # Honesty placement #1: the typed refusal names the UTF-8 limitation.
        err(f"{_PROG}: refused ({exc.reason}): {exc}")
        _emit_error_envelope(
            args, exit_code=2, reason=exc.reason, message=str(exc), exc=exc
        )
        return 2
    except (CheckpointUnknown, CheckpointRegistrationRefused) as exc:
        # CheckpointRegistrationRefused (#191): the checkpoint names another
        # receiver, or another controller registered it — refused before the
        # restore wrote anything. The CLI uses one owner per workspace root,
        # so only a checkpoint created through another path can reach this.
        err(f"{_PROG}: refused ({exc.reason}): {exc}")
        _emit_error_envelope(
            args, exit_code=2, reason=exc.reason, message=str(exc), exc=exc
        )
        return 2
    except CheckpointPersistFailed as exc:
        err(f"{_PROG}: failed ({exc.reason}): {exc}")
        _emit_error_envelope(
            args, exit_code=2, reason=exc.reason, message=str(exc), exc=exc
        )
        return 2
    except CoherenceError as exc:
        # E.g. read_with_version's 3-attempt observe-commit contention path, or
        # a CasVersionConflict/CommitUnconfirmed escaping a leg: one line on
        # stderr, exit 1 — never a raw traceback.
        err(f"{_PROG}: {exc}")
        _emit_error_envelope(
            args,
            exit_code=1,
            reason=getattr(exc, "reason", "coherence_error"),
            message=str(exc),
            exc=exc,
        )
        return 1
    except ValueError as exc:
        err(f"{_PROG}: {exc}")
        _emit_error_envelope(
            args, exit_code=1, reason="validation_error", message=str(exc), exc=exc
        )
        return 1
    finally:
        if stack is not None:
            stack.close()


if __name__ == "__main__":
    raise SystemExit(main())
