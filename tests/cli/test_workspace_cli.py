# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-workspace`` CLI tests (WV plan Unit 8 / R6, R8).

Covers, per the plan's Unit-8 scenarios:

- per-verb happy paths (checkpoint / list / status / restore) in BOTH the human
  and the ``--json`` output modes;
- every enumerated honesty placement asserted present in the output:
  (1) the typed binary-member refusal naming the UTF-8 limitation,
  (2) the ``--help`` constraint note (single-host scope; no-arbiter file members),
  (3) ``status`` labeling EVERY member with the ``(restore_tier, pin_state)``
      pair — ``(restorable, unpinned)`` rendered explicitly as
      claimed-but-not-yet-backed — plus ``dirty_during_window`` and
      ``restore_outcome``, with ``pin_refcount`` + ``restore_status`` in the
      header,
  (4) the ``checkpoint`` retention caveat (file pins are verification, not a
      guarantee);
- error paths: binary member refusal (typed, non-zero), unknown checkpoint id
  (typed, non-zero), path traversal rejection, empty member set;
- the e2e example: exits 0 iff baseline-shows-loss AND guarded-prevents,
  offline, deterministic across two consecutive runs, with the origin
  Success-Criterion-1 trio (delete leg + forward-only skip + S3 foreign-writer
  conflict) visible in the run output.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from pathlib import Path
from uuid import UUID, uuid4

import pytest

import ccs
from ccs.adapters.workspace import MemberRestoreOutcome, RestoreObservation
from ccs.cli.workspace import (
    CLAIMED_NOT_BACKED_LABEL,
    FILE_RETENTION_CAVEAT,
    MEMBER_PATH_REFUSED_REASON,
    MemberPathRefused,
    WorkingTreeSource,
    _discarded_post_capture_content,
    _outcome_payload,
    _restore_outcome_line,
)
from ccs.cli.workspace import (
    main as workspace_main,
)
from ccs.coordinator.registry_protocol import CheckpointMember
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    RESTORE_OBSERVATION_DIFFERS,
    RESTORE_OBSERVATION_NO_LIVE_STATE,
    RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
    RESTORE_OBSERVATION_NOT_RECORDED,
    RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE,
    RESTORE_OBSERVATION_STATES,
    RESTORE_OUTCOME_RESTORED,
    STALE_READ_GENERATION_REASON,
    WORKSPACE_REGISTRATION_REFUSED,
)
from ccs.core.substrate import ArbitrationTier, RestoreTier, sha256_hex
from ccs.core.types import ConflictDetail, WorkspaceRegistrationResult

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "examples" / "workspace_versioning" / "main.py"


def _run(capsys, *args: str) -> tuple[int, str, str]:
    rc = workspace_main(list(args))
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def _seed_file(root: Path, rel: str = "docs/plan.md", body: bytes = b"plan v1\n") -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _checkpoint_id(capsys, root: Path) -> str:
    rc, out, _ = _run(capsys, "list", "--json", "--root", str(root))
    assert rc == 0
    return json.loads(out)["checkpoints"][0]["checkpoint_id"]


def _fabricate_checkpoint(
    root: Path, member: CheckpointMember, *, receiver: UUID | None = None
) -> str:
    """Persist a manifest row the CLI cannot mint itself (an S3-shaped member)
    directly through the CLI's own registry, so ``status``/``restore`` render
    real durable state rather than a mock."""
    db_path = root / ".coherence" / "workspace.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    registry = SqliteArtifactRegistry(db_path, retain_versions=True)
    try:
        service = CoordinatorService(registry)
        record = service.create_workspace_checkpoint(
            name="fabricated",
            owner=uuid4(),
            members=[member],
            window_min=1.0,
            window_max=2.0,
            receiver=receiver,
        )
        return record.checkpoint_id
    finally:
        registry.close()


# --- checkpoint: happy paths (human + JSON) + honesty placement #4 --------------


def test_checkpoint_happy_human_carries_pair_and_retention_caveat(
    tmp_path: Path, capsys
) -> None:
    _seed_file(tmp_path)
    rc, out, err = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path)
    )
    assert rc == 0, err
    assert "checkpoint 'cp1' persisted:" in out
    # The member line carries the honesty PAIR; the resolver-backed
    # verification pin lands "held" while the tier honestly stays unpinned-class.
    assert "docs/plan.md  (restorable-unpinned, held)" in out
    # Honesty placement #4: the retention caveat rides the checkpoint output.
    assert "retention caveat:" in out
    assert "VERIFICATION of the retained bytes, not a retention guarantee" in out
    assert "restorable-unpinned" in FILE_RETENTION_CAVEAT


def test_checkpoint_happy_json_shape(tmp_path: Path, capsys) -> None:
    _seed_file(tmp_path)
    rc, out, err = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--forward-only",
        "actions/deploy",
        "--json",
        "--root",
        str(tmp_path),
    )
    assert rc == 0, err
    payload = json.loads(out)
    assert payload["checkpoint"]["name"] == "cp1"
    assert payload["checkpoint"]["restore_status"] == "none"
    assert payload["retention_caveat"] == FILE_RETENTION_CAVEAT  # placement #4, JSON
    members = {m["member_path"]: m for m in payload["members"]}
    assert members["docs/plan.md"]["pair"] == "(restorable-unpinned, held)"
    assert members["docs/plan.md"]["claimed_not_backed"] is False
    assert members["actions/deploy"]["restore_tier"] == "forward_only"


def test_checkpoint_no_pin_persists_the_unpinned_pair(tmp_path: Path, capsys) -> None:
    _seed_file(tmp_path)
    rc, out, _ = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--no-pin",
        "--root",
        str(tmp_path),
    )
    assert rc == 0
    assert "docs/plan.md  (restorable-unpinned, unpinned)" in out


# --- list: happy paths ----------------------------------------------------------


def test_list_happy_human_and_empty(tmp_path: Path, capsys) -> None:
    rc, out, _ = _run(capsys, "list", "--root", str(tmp_path))
    assert rc == 0
    assert "no checkpoints persisted" in out

    _seed_file(tmp_path)
    _run(capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path))
    rc, out, _ = _run(capsys, "list", "--root", str(tmp_path))
    assert rc == 0
    assert "'cp1'" in out
    assert "restore_status=none" in out
    assert "pin_refcount=1" in out


def test_list_happy_json(tmp_path: Path, capsys) -> None:
    _seed_file(tmp_path)
    _run(capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path))
    rc, out, _ = _run(capsys, "list", "--json", "--root", str(tmp_path))
    assert rc == 0
    payload = json.loads(out)
    assert len(payload["checkpoints"]) == 1
    record = payload["checkpoints"][0]
    assert record["name"] == "cp1"
    assert record["pin_refcount"] == 1
    assert record["restore_status"] == "none"


# --- status: happy paths + honesty placement #3 ---------------------------------


def test_status_happy_human_header_and_pairs(tmp_path: Path, capsys) -> None:
    _seed_file(tmp_path)
    _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--forward-only",
        "actions/deploy",
        "--root",
        str(tmp_path),
    )
    ckpt = _checkpoint_id(capsys, tmp_path)
    rc, out, _ = _run(capsys, "status", ckpt, "--root", str(tmp_path))
    assert rc == 0
    # Placement #3 header half: restore_status + pin_refcount.
    assert "restore_status=none  pin_refcount=1" in out
    # Placement #3 member half: EVERY member labeled with the pair.
    assert "docs/plan.md  (restorable-unpinned, held)" in out
    assert "actions/deploy  (forward_only, unpinned)" in out


def test_status_happy_json(tmp_path: Path, capsys) -> None:
    _seed_file(tmp_path)
    _run(capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path))
    ckpt = _checkpoint_id(capsys, tmp_path)
    rc, out, _ = _run(capsys, "status", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    payload = json.loads(out)
    assert payload["checkpoint"]["pin_refcount"] == 1
    member = payload["members"][0]
    assert member["pair"] == "(restorable-unpinned, held)"
    assert member["claimed_not_backed"] is False
    assert member["dirty_during_window"] is False


def test_status_renders_claimed_but_not_yet_backed_pair(tmp_path: Path, capsys) -> None:
    """Placement #3's load-bearing case: (restorable, unpinned) is a CLAIM
    nothing backs yet and must be rendered as exactly that."""
    ckpt = _fabricate_checkpoint(
        tmp_path,
        CheckpointMember(
            member_path="s3://reports/summary.txt",
            artifact_id=None,
            native_token="v000001",
            fingerprint="a" * 64,
            captured_at=1.0,
            arbitration_tier=ArbitrationTier.NATIVE_CAS.value,
            restore_tier=RestoreTier.RESTORABLE.value,
        ),
    )
    rc, out, _ = _run(capsys, "status", ckpt, "--root", str(tmp_path))
    assert rc == 0
    assert "(restorable, unpinned)" in out
    assert CLAIMED_NOT_BACKED_LABEL in out

    rc, out, _ = _run(capsys, "status", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    member = json.loads(out)["members"][0]
    assert member["pair"] == "(restorable, unpinned)"
    assert member["claimed_not_backed"] is True


def test_status_renders_dirty_during_window(tmp_path: Path, capsys) -> None:
    ckpt = _fabricate_checkpoint(
        tmp_path,
        CheckpointMember(
            member_path="docs/plan.md",
            artifact_id=None,
            native_token="3",
            fingerprint="b" * 64,
            captured_at=1.0,
            dirty_during_window=True,
            arbitration_tier=ArbitrationTier.NO_ARBITER.value,
            restore_tier=RestoreTier.RESTORABLE_UNPINNED.value,
        ),
    )
    rc, out, _ = _run(capsys, "status", ckpt, "--root", str(tmp_path))
    assert rc == 0
    assert "dirty-during-window" in out
    rc, out, _ = _run(capsys, "status", ckpt, "--json", "--root", str(tmp_path))
    assert json.loads(out)["members"][0]["dirty_during_window"] is True


# --- restore: happy paths (human + JSON) ----------------------------------------


def test_restore_happy_human_brings_the_file_back(tmp_path: Path, capsys) -> None:
    plan = _seed_file(tmp_path)
    _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--forward-only",
        "actions/deploy",
        "--root",
        str(tmp_path),
    )
    ckpt = _checkpoint_id(capsys, tmp_path)
    plan.write_bytes(b"corrupted by a failed attempt\n")

    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    assert rc == 0
    assert "CONCLUDED" in out
    assert "docs/plan.md  outcome=restored" in out
    assert "actions/deploy  outcome=forward_only_skipped" in out
    # The file leg's honesty label: detection, never substrate arbitration.
    assert "no-arbiter" in out
    assert plan.read_bytes() == b"plan v1\n"

    # status now surfaces the durable outcomes (placement #3, outcome half).
    rc, out, _ = _run(capsys, "status", ckpt, "--root", str(tmp_path))
    assert rc == 0
    assert "restore_status=concluded" in out
    assert "outcome=restored" in out


def test_restore_happy_json(tmp_path: Path, capsys) -> None:
    plan = _seed_file(tmp_path)
    _run(capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path))
    ckpt = _checkpoint_id(capsys, tmp_path)
    plan.write_bytes(b"corrupted\n")

    rc, out, _ = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    payload = json.loads(out)
    assert payload["status"] == "concluded"
    member = payload["members"][0]
    assert member["member_path"] == "docs/plan.md"
    assert member["outcome"] == "restored"
    assert member["new_native_token"] is not None
    assert "registration" in payload
    assert plan.read_bytes() == b"plan v1\n"


def test_restore_absent_divergence_absorbed_as_no_arbiter_conflict(
    tmp_path: Path, capsys
) -> None:
    """A member captured ABSENT that exists live: the v1 file leg has no delete
    surface — the divergence is ABSORBED as a labeled conflict (exit 3), never
    a silent skip and never presented as arbitration."""
    _seed_file(tmp_path)  # a second, present member keeps the manifest non-empty
    rc, _, _ = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--file",
        "docs/ghost.md",  # does not exist -> the ABSENT fact
        "--root",
        str(tmp_path),
    )
    assert rc == 0
    ckpt = _checkpoint_id(capsys, tmp_path)
    (tmp_path / "docs" / "ghost.md").write_bytes(b"appeared after capture\n")

    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    assert rc == 3  # concluded, with an absorbed outcome
    assert "docs/ghost.md  outcome=conflict" in out
    assert "no-arbiter" in out
    assert "absorbed outcomes above are the honest per-member truth" in out


# --- error paths ----------------------------------------------------------------


def test_binary_member_typed_refusal_names_the_utf8_limitation(
    tmp_path: Path, capsys
) -> None:
    _seed_file(tmp_path, rel="blob.bin", body=b"\xff\xfe\x00\x01binary")
    rc, out, err = _run(
        capsys, "checkpoint", "cp1", "--file", "blob.bin", "--root", str(tmp_path)
    )
    assert rc == 2
    # Honesty placement #1: the typed reason + the UTF-8 limitation, verbatim.
    assert "binary_file_member_unsupported" in err
    assert "non-UTF-8" in err
    assert "v1 limitation" in err
    # Nothing persisted (the refusal fires BEFORE any manifest write).
    rc, out, _ = _run(capsys, "list", "--root", str(tmp_path))
    assert "no checkpoints persisted" in out


def test_unknown_checkpoint_id_is_a_clean_typed_error(tmp_path: Path, capsys) -> None:
    for verb in ("status", "restore"):
        rc, _, err = _run(capsys, verb, "no-such-checkpoint", "--root", str(tmp_path))
        assert rc == 2
        assert "checkpoint 'no-such-checkpoint' is unknown" in err


def test_checkpoint_rejects_traversal_and_outside_paths(tmp_path: Path, capsys) -> None:
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "../outside.md", "--root", str(tmp_path)
    )
    assert rc == 1
    assert "rejected" in err


def test_checkpoint_requires_at_least_one_member(tmp_path: Path, capsys) -> None:
    rc, _, err = _run(capsys, "checkpoint", "cp1", "--root", str(tmp_path))
    assert rc == 1
    assert "at least one member" in err


def test_restore_refuses_s3_members_with_a_clean_pointer_to_the_api(
    tmp_path: Path, capsys
) -> None:
    ckpt = _fabricate_checkpoint(
        tmp_path,
        CheckpointMember(
            member_path="s3://reports/summary.txt",
            artifact_id=None,
            native_token="v000001",
            fingerprint="a" * 64,
            captured_at=1.0,
            arbitration_tier=ArbitrationTier.NATIVE_CAS.value,
            restore_tier=RestoreTier.RESTORABLE.value,
        ),
    )
    rc, _, err = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    assert rc == 1
    assert "S3 object members" in err
    assert "nothing was started" in err
    # status still renders the member honestly.
    rc, out, _ = _run(capsys, "status", ckpt, "--root", str(tmp_path))
    assert rc == 0
    assert "s3://reports/summary.txt" in out


# --- member-path containment (symlink escape / .coherence / directory) ----------


def _escape_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A workspace root plus a sibling OUTSIDE directory holding a secret."""
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside secret\n")
    return root, outside


def test_checkpoint_refuses_symlinked_intermediate_dir_escape(
    tmp_path: Path, capsys
) -> None:
    """`--file evil_link/secret.txt` with evil_link -> an outside dir must be a
    typed exit-2 refusal — never a capture of the outside file."""
    root, outside = _escape_fixture(tmp_path)
    (root / "evil_link").symlink_to(outside)
    rc, _, err = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "evil_link/secret.txt",
        "--root",
        str(root),
    )
    assert rc == 2
    assert "refused" in err
    assert "symlink" in err
    # Nothing persisted: the refusal fires before any manifest write.
    rc, out, _ = _run(capsys, "list", "--root", str(root))
    assert "no checkpoints persisted" in out


def test_restore_refuses_persisted_member_crossing_symlink(
    tmp_path: Path, capsys
) -> None:
    """Restore replays PERSISTED member_paths that never re-pass the CLI arg
    pre-check: a fabricated manifest row crossing a symlink (the re-pointed-
    between-checkpoint-and-restore shape) must refuse at the filesystem-access
    seam and never write outside the root.

    The refusal is ENFORCED identically on the restore leg, but it is no longer
    RAISED there: the engine's termination contract absorbs it into the
    member's ``target_lost`` so the run concludes (exit 3) instead of wedging
    ``restore_status=in_progress`` forever. The refusal text rides the report."""
    root, outside = _escape_fixture(tmp_path)
    victim = outside / "secret.txt"
    ckpt = _fabricate_checkpoint(
        root,
        CheckpointMember(
            member_path="evil_link/secret.txt",
            artifact_id=None,
            native_token="1",
            fingerprint="c" * 64,
            captured_at=1.0,
            arbitration_tier=ArbitrationTier.NO_ARBITER.value,
            restore_tier=RestoreTier.RESTORABLE_UNPINNED.value,
        ),
    )
    (root / "evil_link").symlink_to(outside)
    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(root))
    assert rc == 3
    assert "outcome=target_lost" in out
    assert "symlink" in out
    assert victim.read_text() == "outside secret\n"  # never written through


def test_checkpoint_refuses_leaf_symlink(tmp_path: Path, capsys) -> None:
    root, outside = _escape_fixture(tmp_path)
    (root / "leaf.md").symlink_to(outside / "secret.txt")
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "leaf.md", "--root", str(root)
    )
    assert rc == 2
    assert "symlink" in err

    # An IN-root leaf symlink is refused too: ANY symlink component (leaf
    # included) can be re-pointed after the check (the TOCTOU window).
    (root / "real.md").write_text("real\n")
    (root / "alias.md").symlink_to(root / "real.md")
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "alias.md", "--root", str(root)
    )
    assert rc == 2
    assert "symlink" in err


def test_checkpoint_refuses_coherence_state_member(tmp_path: Path, capsys) -> None:
    """`.coherence/**` is coordinator state (the hook HMAC secret lives there)
    — capturing it into workspace.db is an info-disclosure, refused typed."""
    secret = tmp_path / ".coherence" / "hook.secret"
    secret.parent.mkdir(mode=0o700)
    secret.write_text("hmac-secret\n")
    rc, _, err = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        ".coherence/hook.secret",
        "--root",
        str(tmp_path),
    )
    assert rc == 2
    assert ".coherence" in err
    rc, out, _ = _run(capsys, "list", "--root", str(tmp_path))
    assert "no checkpoints persisted" in out


def test_restore_refuses_persisted_coherence_member(tmp_path: Path, capsys) -> None:
    ckpt = _fabricate_checkpoint(
        tmp_path,
        CheckpointMember(
            member_path=".coherence/hook.secret",
            artifact_id=None,
            native_token="1",
            fingerprint="d" * 64,
            captured_at=1.0,
            arbitration_tier=ArbitrationTier.NO_ARBITER.value,
            restore_tier=RestoreTier.RESTORABLE_UNPINNED.value,
        ),
    )
    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    # Absorbed into target_lost by the termination contract (the run concludes,
    # exit 3) — the refusal itself is unchanged: nothing under .coherence/ is
    # ever read or written by a restore leg.
    assert rc == 3
    assert "outcome=target_lost" in out
    assert ".coherence" in out
    assert (tmp_path / ".coherence" / "hook.secret").exists() is False


def test_directory_member_is_a_typed_refusal_not_a_traceback(
    tmp_path: Path, capsys
) -> None:
    (tmp_path / "somedir").mkdir()
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "somedir", "--root", str(tmp_path)
    )
    assert rc == 2
    assert "refused" in err
    assert "Traceback" not in err


# --- member-path containment (hardlink co-owner + leaf O_NOFOLLOW) --------------


def test_checkpoint_refuses_hardlink_to_outside_file(tmp_path: Path, capsys) -> None:
    """FIX 1 (P0): a hard link to an OUTSIDE file has ``is_symlink()==False`` and
    ``realpath()`` resolves in-tree, so the symlink + realpath guards pass — but
    capturing it would read the outside file's bytes into workspace.db. The
    ``st_nlink > 1`` check must refuse it (exit 2), and nothing is captured."""
    root, outside = _escape_fixture(tmp_path)
    victim = outside / "secret.txt"  # "outside secret\n"
    os.link(victim, root / "hardlinked.txt")  # a hard link, NOT a symlink
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "hardlinked.txt", "--root", str(root)
    )
    assert rc == 2
    assert "refused" in err
    assert "hard link" in err
    # The outside bytes were never captured — the refusal fires before any read.
    rc, out, _ = _run(capsys, "list", "--root", str(root))
    assert "no checkpoints persisted" in out
    assert victim.read_text() == "outside secret\n"


def test_restore_refuses_persisted_hardlinked_member(tmp_path: Path, capsys) -> None:
    """FIX 1 (P0): restore replays PERSISTED member_paths. A fabricated row whose
    live path is a hard link to an outside file must refuse at the filesystem-
    access seam — the outside file is neither read (into the ledger) nor written
    through the shared inode.

    The refusal is absorbed into ``target_lost`` (termination contract), so the
    restore CONCLUDES at exit 3 and the refusal rides the member's report line
    — the containment guarantee is identical, only the surfacing changed."""
    root, outside = _escape_fixture(tmp_path)
    victim = outside / "secret.txt"
    ckpt = _fabricate_checkpoint(
        root,
        CheckpointMember(
            member_path="hardlinked.txt",
            artifact_id=None,
            native_token="1",
            fingerprint="e" * 64,
            captured_at=1.0,
            arbitration_tier=ArbitrationTier.NO_ARBITER.value,
            restore_tier=RestoreTier.RESTORABLE_UNPINNED.value,
        ),
    )
    os.link(victim, root / "hardlinked.txt")
    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(root))
    assert rc == 3
    assert "outcome=target_lost" in out
    assert "hard link" in out
    # NOT written through: the outside file's bytes are untouched.
    assert victim.read_text() == "outside secret\n"


def test_checkpoint_refuses_hardlinked_coherence_secret(
    tmp_path: Path, capsys
) -> None:
    """FIX 1 (P0): hardlinking ``.coherence/hook.secret`` to a plain member name
    slips the ``.coherence/**`` string/realpath guard (the plain name resolves
    in-tree, outside ``.coherence``) — the ``st_nlink > 1`` check is what stops
    the HMAC secret from being captured."""
    secret = tmp_path / ".coherence" / "hook.secret"
    secret.parent.mkdir(mode=0o700, exist_ok=True)
    secret.write_text("hmac-secret\n")
    os.link(secret, tmp_path / "innocuous.txt")
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "innocuous.txt", "--root", str(tmp_path)
    )
    assert rc == 2
    assert "hard link" in err
    rc, out, _ = _run(capsys, "list", "--root", str(tmp_path))
    assert "no checkpoints persisted" in out


def test_checkpoint_single_link_file_is_not_a_false_positive(
    tmp_path: Path, capsys
) -> None:
    """Regression guard for FIX 1: an ordinary single-link file (``st_nlink==1``)
    must still checkpoint cleanly — the hardlink refusal fires ONLY on a real
    external co-owner."""
    path = _seed_file(tmp_path)
    assert path.stat().st_nlink == 1  # precondition: no external co-owner
    rc, out, err = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path)
    )
    assert rc == 0, err
    assert "checkpoint 'cp1' persisted:" in out


# --- member-path containment (non-regular leaves: the unbounded-open gate) ------


def _checkpoint_in_subprocess(
    root: Path, member: str, *, timeout: float = 20.0
) -> subprocess.CompletedProcess:
    """Run ONE ``checkpoint --file <member>`` in a child process under a hard
    timeout.

    The timeout IS the assertion: ``os.open`` on a FIFO with no peer writer
    blocks in the kernel, which no in-process assertion can catch — a
    regression would hang the test session forever instead of failing. A child
    process plus ``subprocess.run(timeout=...)`` turns that hang into a loud
    ``TimeoutExpired``.
    """
    src_dir = Path(ccs.__file__).resolve().parents[1]  # the dir holding ``ccs``
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(src_dir), *(
        [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []
    )])}
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from ccs.cli.workspace import main; sys.exit(main(sys.argv[1:]))",
            "checkpoint",
            "cp1",
            "--file",
            member,
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def test_fifo_member_is_refused_at_checkpoint_and_never_hangs(tmp_path: Path) -> None:
    """A FIFO member must be a typed exit-2 refusal, NOT an unbounded wait.

    The guarded open had no non-regular gate, so ``checkpoint --file pipe.txt``
    with no peer writer blocked inside ``os.open`` forever (``Path.read_bytes``
    blocked identically before it) — an operator-visible hang with no timeout
    the process controls. The stat gate now refuses the leaf before any open.
    """
    os.mkfifo(tmp_path / "pipe.txt")
    result = _checkpoint_in_subprocess(tmp_path, "pipe.txt")
    assert result.returncode == 2, result.stdout + result.stderr
    assert "refused" in result.stderr
    assert "not a regular file" in result.stderr
    assert "FIFO" in result.stderr
    assert "Traceback" not in result.stderr


def test_socket_member_is_refused_at_checkpoint(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """The sibling non-regular shape: a unix socket leaf is refused too (exit 2),
    and nothing is persisted."""
    monkeypatch.chdir(tmp_path)  # AF_UNIX paths are ~104 bytes; bind relatively
    sock = socket.socket(socket.AF_UNIX)
    try:
        sock.bind("sock.txt")
        rc, _, err = _run(
            capsys, "checkpoint", "cp1", "--file", "sock.txt", "--root", str(tmp_path)
        )
    finally:
        sock.close()
    assert rc == 2
    assert "not a regular file" in err
    assert "socket" in err
    rc, out, _ = _run(capsys, "list", "--root", str(tmp_path))
    assert "no checkpoints persisted" in out


def test_regular_file_beside_a_fifo_still_checkpoints(tmp_path: Path, capsys) -> None:
    """False-positive guard for the non-regular gate: the refusal is per-LEAF.
    An ordinary regular file in a workspace that also holds a FIFO must still
    capture cleanly."""
    os.mkfifo(tmp_path / "pipe.txt")
    _seed_file(tmp_path)
    rc, out, err = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path)
    )
    assert rc == 0, err
    assert "checkpoint 'cp1' persisted:" in out


def test_absent_member_is_a_fact_not_a_non_regular_refusal(
    tmp_path: Path, capsys
) -> None:
    """The stat gate must not turn ABSENCE into a refusal: an absent leaf takes
    no stat-based check at all and stays the ABSENT fact the manifest records."""
    rc, out, err = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/never.md", "--root", str(tmp_path)
    )
    assert rc == 0, err
    assert "docs/never.md" in out
    rc, out, _ = _run(
        capsys, "status", _checkpoint_id(capsys, tmp_path), "--json", "--root", str(tmp_path)
    )
    (member,) = json.loads(out)["members"]
    assert member["absent"] is True


def test_read_with_version_rejects_leaf_symlink_swapped_after_validation(
    tmp_path: Path, monkeypatch
) -> None:
    """FIX 2 (P1): the check-vs-open TOCTOU. Even when the validator is bypassed
    (simulating a leaf swapped to a symlink in the gap between validation and the
    read syscall), the read goes through ``O_NOFOLLOW`` — a leaf symlink is
    rejected atomically at open (ELOOP), so the outside target is never followed
    or read into the ledger."""
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("outside secret\n")
    evil_leaf = root / "leaf.md"
    evil_leaf.symlink_to(outside / "secret.txt")

    db_path = root / ".coherence" / "workspace.db"
    db_path.parent.mkdir(mode=0o700)
    registry = SqliteArtifactRegistry(db_path, retain_versions=True)
    try:
        source = WorkingTreeSource(root, registry, uuid4())
        # Stub the validator to wave the leaf through — the O_NOFOLLOW open is
        # the second, independent line of defense being exercised here.
        monkeypatch.setattr(source, "_abs", lambda p: evil_leaf)
        with pytest.raises(MemberPathRefused) as excinfo:
            source.read_with_version("leaf.md")
        assert "symlink" in str(excinfo.value)
    finally:
        registry.close()


# --- fresh-workspace .coherence hardening (0700 + gitignore) --------------------


def test_fresh_workspace_coherence_dir_is_0700_and_gitignored(
    tmp_path: Path, capsys
) -> None:
    """A fresh workspace's .coherence/ must come from the lifecycle's single
    implementation: 0700 perms + the '*' gitignore, so `git add .` can never
    stage workspace.db."""
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    _seed_file(tmp_path)
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path)
    )
    assert rc == 0, err
    coherence = tmp_path / ".coherence"
    assert (coherence.stat().st_mode & 0o777) == 0o700
    assert (coherence / ".gitignore").read_text() == "*\n"
    status = subprocess.run(
        ["git", "-C", str(tmp_path), "status", "--ignored", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "!! .coherence/" in status.stdout
    assert "workspace.db" not in status.stdout


def test_preexisting_0755_coherence_dir_is_retightened_to_0700(
    tmp_path: Path, capsys
) -> None:
    """FIX 3 (P2): ``mkdir(mode=0o700, exist_ok=True)`` does NOT chmod an
    already-existing directory, so a ``.coherence/`` that pre-exists at 0755
    would stay 0755 (exposing state.db, hook.secret, the pidfile). Running any
    verb must re-tighten it to 0700."""
    coherence = tmp_path / ".coherence"
    coherence.mkdir()
    os.chmod(coherence, 0o755)  # pre-existing loose mode (umask-independent)
    assert (coherence.stat().st_mode & 0o777) == 0o755

    _seed_file(tmp_path)
    rc, _, err = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path)
    )
    assert rc == 0, err
    assert (coherence.stat().st_mode & 0o777) == 0o700


# --- --json error envelope ------------------------------------------------------


def test_json_error_envelope_carries_typed_reason(tmp_path: Path, capsys) -> None:
    """Every error path under --json emits a parseable envelope on stdout
    carrying the typed reason; the stderr prose stays for humans."""
    # Typed refusal: binary member (exit 2).
    _seed_file(tmp_path, rel="blob.bin", body=b"\xff\xfe\x00\x01binary")
    rc, out, err = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "blob.bin",
        "--json",
        "--root",
        str(tmp_path),
    )
    assert rc == 2
    envelope = json.loads(out)
    assert envelope["kind"] == "error"
    assert envelope["exit_code"] == 2
    assert envelope["reason"] == "binary_file_member_unsupported"
    assert "non-UTF-8" in err  # stderr prose retained

    # Typed refusal: unknown checkpoint (exit 2).
    rc, out, _ = _run(capsys, "status", "nope", "--json", "--root", str(tmp_path))
    assert rc == 2
    envelope = json.loads(out)
    assert envelope["reason"] == "checkpoint_unknown"

    # Typed refusal: directory member (exit 2, member-path slug).
    (tmp_path / "somedir").mkdir()
    rc, out, _ = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "somedir",
        "--json",
        "--root",
        str(tmp_path),
    )
    assert rc == 2
    envelope = json.loads(out)
    assert envelope["reason"] == MEMBER_PATH_REFUSED_REASON

    # Validation error: traversal path (exit 1).
    rc, out, _ = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "../outside.md",
        "--json",
        "--root",
        str(tmp_path),
    )
    assert rc == 1
    envelope = json.loads(out)
    assert envelope["kind"] == "error"
    assert envelope["exit_code"] == 1
    assert envelope["reason"] == "invalid_member_path"


# --- restore registration honesty fields ----------------------------------------


def test_restore_surfaces_invalidated_peers_and_attempts(
    tmp_path: Path, capsys
) -> None:
    plan = _seed_file(tmp_path)
    _run(capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path))
    ckpt = _checkpoint_id(capsys, tmp_path)
    plan.write_bytes(b"corrupted\n")

    rc, out, _ = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    registration = json.loads(out)["registration"]
    assert isinstance(registration["invalidated_peers"], int)
    assert isinstance(registration["attempts"], int)

    # A fresh first restore in HUMAN mode (this run performs registration, so
    # the report always carries it) surfaces the same honesty fields.
    _run(capsys, "checkpoint", "cp2", "--file", "docs/plan.md", "--root", str(tmp_path))
    rc, out, _ = _run(capsys, "list", "--json", "--root", str(tmp_path))
    records = json.loads(out)["checkpoints"]
    ckpt2 = next(r["checkpoint_id"] for r in records if r["name"] == "cp2")
    plan.write_bytes(b"corrupted again\n")
    rc, out, _ = _run(capsys, "restore", ckpt2, "--root", str(tmp_path))
    assert rc == 0
    assert "peers invalidated by registration:" in out
    assert "attempts=" in out


# --- restore observation: what the run put the checkpoint back OVER -------------

# The seven keys a consumer written against the pre-observation payload reads.
# Frozen deliberately: derived from the builder at runtime this set would move
# its own goalposts, and the rename that breaks a consumer would report green.
PRE_OBSERVATION_MEMBER_KEYS = frozenset(
    {
        "member_path",
        "outcome",
        "attempts",
        "detail",
        "new_native_token",
        "deleted_at_restore",
        "resumed_from_prior_run",
    }
)

# Captured from the renderer BEFORE any annotation existed. A member whose leg
# never reached a write decision must still render exactly these bytes. The
# forward-only skip is the guaranteed-quiet case: a converged member can be
# rebuilt from durable state on a later run and then honestly reports an
# unrecorded observation, so "converged" is not a synonym for "quiet".
QUIET_MEMBER_LINE = "  actions/deploy  outcome=forward_only_skipped  attempts=0"

# What the restore lands on top of, so the file leg reads a live state that
# differs from the captured one.
DISCARDED_BYTES = b"committed by a peer after the capture\n"


def _checkpoint_then_diverge(capsys, root: Path) -> str:
    """Capture ``docs/plan.md`` plus a forward-only member, then overwrite the
    file — so one member is restored over differing content and the other never
    reaches a write decision at all."""
    plan = _seed_file(root)
    _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--forward-only",
        "actions/deploy",
        "--root",
        str(root),
    )
    ckpt = _checkpoint_id(capsys, root)
    plan.write_bytes(DISCARDED_BYTES)
    return ckpt


def _members_by_path(out: str) -> dict[str, dict]:
    return {m["member_path"]: m for m in json.loads(out)["members"]}


def test_restore_json_names_the_version_and_digest_the_write_discarded(
    tmp_path: Path, capsys
) -> None:
    """R8: the observation is machine-readable as keyed values — an operator's
    tooling reads WHICH version was discarded without parsing any prose."""
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)

    rc, out, _ = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    observation = _members_by_path(out)["docs/plan.md"]["observation"]
    assert observation["state"] == RESTORE_OBSERVATION_DIFFERS
    # The pointer is the version that was OVERWRITTEN, reachable as its own
    # value; the fingerprint is the digest of the bytes the restore replaced.
    assert isinstance(observation["pointer"], str) and observation["pointer"]
    assert observation["fingerprint"] == sha256_hex(DISCARDED_BYTES)


def test_restore_json_keeps_every_pre_observation_member_key(
    tmp_path: Path, capsys
) -> None:
    """A consumer reading only the keys that existed before the observation is
    unaffected: same names, same meanings."""
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)

    rc, out, _ = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    member = _members_by_path(out)["docs/plan.md"]
    assert PRE_OBSERVATION_MEMBER_KEYS <= set(member)
    assert member["outcome"] == "restored"
    assert member["attempts"] == 1
    assert member["new_native_token"] is not None
    assert member["deleted_at_restore"] is None
    assert member["resumed_from_prior_run"] is False
    assert "version-CAS" in member["detail"]


def test_restore_leaves_a_quiet_members_human_line_byte_identical(
    tmp_path: Path, capsys
) -> None:
    """A member that attempted no write is not made noisier by this change."""
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)

    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    assert rc == 0
    assert QUIET_MEMBER_LINE in out.splitlines()

    # ...and this guard can SEE the case it claims to cover: the pinned line
    # belongs to a member whose observation really is the no-write state, not
    # one that merely happens to render quietly.
    _run(capsys, "checkpoint", "cp2", "--file", "docs/plan.md", "--forward-only",
         "actions/deploy", "--root", str(tmp_path))
    records = json.loads(_run(capsys, "list", "--json", "--root", str(tmp_path))[1])
    ckpt2 = next(
        r["checkpoint_id"] for r in records["checkpoints"] if r["name"] == "cp2"
    )
    rc, out, _ = _run(capsys, "restore", ckpt2, "--json", "--root", str(tmp_path))
    assert rc == 0
    quiet = _members_by_path(out)["actions/deploy"]["observation"]
    assert quiet["state"] == RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED


def test_restore_human_line_says_it_overwrote_differing_content(
    tmp_path: Path, capsys
) -> None:
    """The operator reads from the run's own report that this restore did not
    put back a state nothing else had touched."""
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)

    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    assert rc == 0
    lines = out.splitlines()
    index = next(
        i
        for i, line in enumerate(lines)
        if line.startswith("  docs/plan.md  outcome=restored")
    )
    assert "overwrote-differing-content" in lines[index]
    assert "overwritten-version=" in lines[index]
    # The detail fragment the engine tests match on is still the NEXT line and
    # still reads exactly as it did — annotation rides the member line only.
    assert lines[index + 1] == (
        "    pinned bytes landed via the detection-guarded version-CAS "
        "(attempt 1; no-arbiter: adapter-local detection, never substrate "
        "arbitration)"
    )


def test_resumed_member_reports_an_unrecorded_observation_not_a_clean_one(
    tmp_path: Path, capsys
) -> None:
    """R3: a run that never made the observation says so on every surface —
    it never reads as the no-write-attempted (quiet) answer."""
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)
    _run(capsys, "restore", ckpt, "--root", str(tmp_path))  # the run that wrote

    rc, out, _ = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    observation = _members_by_path(out)["docs/plan.md"]["observation"]
    assert observation["state"] == RESTORE_OBSERVATION_NOT_RECORDED
    assert observation["state"] != RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED
    assert observation["pointer"] is None
    assert observation["fingerprint"] is None

    rc, out, _ = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    assert rc == 0
    line = next(
        line for line in out.splitlines() if line.startswith("  docs/plan.md")
    )
    assert "resumed-from-prior-run" in line
    assert "overwritten-content-not-recorded" in line


def _object_state_outcome(state: str) -> MemberRestoreOutcome:
    """One synthesized member carrying a state only an object leg can reach.

    ``restore`` refuses a checkpoint holding a pending object member before the
    engine runs, so these states never arrive through the command. Driving the
    payload builder and the line renderer directly is what keeps them covered.
    """
    return MemberRestoreOutcome(
        member_path="bucket/key",
        outcome=RESTORE_OUTCOME_RESTORED,
        attempts=1,
        detail="synthesized for the renderer",
        observation=RestoreObservation(state),
    )


@pytest.mark.parametrize(
    "state",
    [RESTORE_OBSERVATION_NO_LIVE_STATE, RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE],
)
def test_object_only_observation_states_carry_no_pointer(state: str) -> None:
    """Neither state read a comparand, so neither names a version or a digest.

    Both are constructed with the state alone, and the payload must render the
    two absent halves as explicit nulls rather than omitting them — a consumer
    that had to read a missing key would be reading absence as an answer.
    """
    payload = _outcome_payload(_object_state_outcome(state))
    assert payload["observation"] == {
        "state": state,
        "pointer": None,
        "fingerprint": None,
    }


def test_a_write_that_landed_on_nothing_leaves_the_line_quiet() -> None:
    """Create-on-absent discarded nothing, so its line must not be annotated.

    This is the control for the test below: it proves the renderer distinguishes
    the two object-only states rather than flagging whatever it does not
    recognise. The exit gate does not fire on this state either, so an annotated
    line here would report a loss the run did not cause.
    """
    line = _restore_outcome_line(_object_state_outcome(RESTORE_OBSERVATION_NO_LIVE_STATE))
    assert line == "  bucket/key  outcome=restored  attempts=1"


def test_a_destroyed_uncompared_member_says_so_on_its_own_line() -> None:
    """A state the exit gate fires on must never render as a quiet line.

    The delete leg's probe established that live state existed and destroyed it,
    which is why ``present_not_comparable`` sits in the exit-4 set beside
    ``observed_differs``. Were the line left unannotated, an operator running
    with the flag could be handed a non-zero exit whose reason appears nowhere
    in the human report — readable only by re-running under ``--json``. The
    wording claims no comparison, because the probe made none.
    """
    line = _restore_outcome_line(
        _object_state_outcome(RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE)
    )
    assert line == "  bucket/key  outcome=restored  attempts=1  destroyed-uncompared-content"
    # No version and no digest are appended: the probe read neither.
    assert "overwritten-version=" not in line


# --- restore exit code: the opt-in discarded-content gate -----------------------

# Spelled once. Inlined in six places this string renames in five of them and
# the sixth test keeps passing against a flag argparse no longer accepts.
DISCARD_FLAG = "--exit-nonzero-on-discarded-content"


def _synthetic_outcome(state: str) -> MemberRestoreOutcome:
    """A member outcome carrying exactly one observation state — the gate's
    input, without a command invocation around it."""
    return MemberRestoreOutcome(
        member_path="bucket/key",
        outcome=RESTORE_OUTCOME_RESTORED,
        attempts=1,
        detail="synthesized for the gate",
        observation=RestoreObservation(state),
    )


def test_divergent_restore_exits_four_only_under_the_flag(
    tmp_path: Path, capsys
) -> None:
    """R6 + R5: one run shape, one report — the operator's flag is the only
    thing that turns a restore over post-capture content into a failure."""
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)

    rc, out, _ = _run(capsys, "restore", ckpt, DISCARD_FLAG, "--root", str(tmp_path))
    assert rc == 4
    assert "overwrote-differing-content" in out

    # The same divergence in a workspace of its own, without the flag: still 0.
    # A second run against the FIRST checkpoint would observe not_recorded, a
    # different state — the default-path arm has to see observed_differs too.
    other = tmp_path / "second-workspace"
    other.mkdir()
    ckpt_other = _checkpoint_then_diverge(capsys, other)

    rc, out, _ = _run(capsys, "restore", ckpt_other, "--root", str(other))
    assert rc == 0
    assert "overwrote-differing-content" in out


def test_absorbed_outcome_takes_precedence_over_the_discard_code(
    tmp_path: Path, capsys
) -> None:
    """R7, exit 3's first producer: one member ends absorbed while another was
    restored over post-capture content. The run exits 3 and the report still
    carries BOTH facts — the new code never hides the older one."""
    plan = _seed_file(tmp_path)
    rc, _, _ = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--file",
        "docs/ghost.md",  # absent at capture -> present live -> absorbed
        "--root",
        str(tmp_path),
    )
    assert rc == 0
    ckpt = _checkpoint_id(capsys, tmp_path)
    plan.write_bytes(DISCARDED_BYTES)
    (tmp_path / "docs" / "ghost.md").write_bytes(b"appeared after capture\n")

    rc, out, _ = _run(capsys, "restore", ckpt, DISCARD_FLAG, "--root", str(tmp_path))
    assert rc == 3
    assert "docs/ghost.md  outcome=conflict" in out
    assert "overwrote-differing-content" in out


def test_refused_registration_takes_precedence_over_the_discard_code(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    """R7, exit 3's SECOND producer: a fence-refused registration holds the run
    at 3 even though a member was restored over post-capture content.

    The refusal is injected at the coordinator seam the engine calls rather
    than through the registry, because the CLI's file bridge commits every
    written member through that same registry: by the time registration runs,
    the artifact already carries the manifest fingerprint and the seam honestly
    answers ``empty_write_set``, so no fence is reachable end to end. The
    engine's refusal path and the CLI's exit computation are both real here.
    """
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)

    def _fenced(self, *, checkpoint_id, controller, writes, issued_at_tick=0, abort=None):
        return WorkspaceRegistrationResult(
            checkpoint_id=checkpoint_id,
            status=WORKSPACE_REGISTRATION_REFUSED,
            detail="the read-generation fence rejected a superseded controller",
            refused={
                write.member_path: ConflictDetail(
                    reason=STALE_READ_GENERATION_REASON, current_version=1
                )
                for write in writes
            },
        )

    monkeypatch.setattr(CoordinatorService, "register_workspace_restore", _fenced)

    rc, out, _ = _run(capsys, "restore", ckpt, DISCARD_FLAG, "--root", str(tmp_path))
    assert rc == 3
    # The guard can SEE its case: the refusal really reached the report, and
    # the discard the gate would otherwise have fired on is there beside it.
    assert f"registration: {WORKSPACE_REGISTRATION_REFUSED}" in out
    assert "overwrote-differing-content" in out


def test_quiescent_restore_exits_zero_under_the_flag(tmp_path: Path, capsys) -> None:
    """R5 under the flag: nothing touched the workspace since the capture, so
    one file member converges and one forward-only member is skipped. Neither
    wrote, neither discarded anything, and the run still exits 0."""
    _seed_file(tmp_path)
    _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--forward-only",
        "actions/deploy",
        "--root",
        str(tmp_path),
    )
    ckpt = _checkpoint_id(capsys, tmp_path)

    rc, out, _ = _run(
        capsys, "restore", ckpt, DISCARD_FLAG, "--json", "--root", str(tmp_path)
    )
    assert rc == 0
    # ...and this guard can SEE the case it claims to cover: both members
    # really hold the no-write state, rather than merely exiting quietly.
    assert {
        path: member["observation"]["state"]
        for path, member in _members_by_path(out).items()
    } == {
        "docs/plan.md": RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
        "actions/deploy": RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
    }


def test_re_running_a_restore_that_overwrote_nothing_still_exits_zero(
    tmp_path: Path, capsys
) -> None:
    """The same command twice over an untouched workspace must answer the same.

    This is the acceptance case for the whole opt-in flag: a run over a
    workspace nothing touched exits 0, which is what shows the gate can tell a
    member that never wrote from one whose observation was lost. The second run
    drives no leg — every member is already terminal — so its answer comes from
    the durable rows. Those rows record ``converged`` and
    ``forward_only_skipped``, which prove no write landed by any run, so
    claiming no observation there would fail an operator's pipeline on the
    identical second invocation and teach them to drop the flag.
    """
    _seed_file(tmp_path)
    _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--forward-only",
        "actions/deploy",
        "--root",
        str(tmp_path),
    )
    ckpt = _checkpoint_id(capsys, tmp_path)

    first, _, _ = _run(capsys, "restore", ckpt, DISCARD_FLAG, "--root", str(tmp_path))
    rc, out, _ = _run(
        capsys, "restore", ckpt, DISCARD_FLAG, "--json", "--root", str(tmp_path)
    )

    assert (first, rc) == (0, 0)
    members = _members_by_path(out)
    # The rebuild really is the path under test: nothing was re-driven.
    assert all(m["resumed_from_prior_run"] for m in members.values())
    assert {path: m["observation"]["state"] for path, m in members.items()} == {
        "docs/plan.md": RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
        "actions/deploy": RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED,
    }


def test_re_restore_of_a_concluded_checkpoint_exits_four_under_the_flag(
    tmp_path: Path, capsys
) -> None:
    """R3 on the exit surface: the second run holds no observation of its own,
    and an observation the run never made must not buy a clean exit."""
    ckpt = _checkpoint_then_diverge(capsys, tmp_path)
    rc, _, _ = _run(capsys, "restore", ckpt, "--root", str(tmp_path))
    assert rc == 0

    rc, out, _ = _run(
        capsys, "restore", ckpt, DISCARD_FLAG, "--json", "--root", str(tmp_path)
    )
    assert rc == 4
    state = _members_by_path(out)["docs/plan.md"]["observation"]["state"]
    assert state == RESTORE_OBSERVATION_NOT_RECORDED
    assert state != RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED  # never the quiet answer


def test_gate_fires_on_destroyed_presence_but_not_on_an_absent_target() -> None:
    """``restore`` refuses a checkpoint carrying a pending object member before
    the engine runs, so the two states only an object leg can reach are driven
    through the gate directly.

    Deleting an object a peer created after the capture DID destroy live state
    — a member captured absent that is present live was created after the
    capture — which is exactly what the flag exists for. Creating a member onto
    nothing discarded nothing at all.
    """
    assert (
        _discarded_post_capture_content(
            _synthetic_outcome(RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE)
        )
        is True
    )
    assert (
        _discarded_post_capture_content(
            _synthetic_outcome(RESTORE_OBSERVATION_NO_LIVE_STATE)
        )
        is False
    )
    # Driven from the ENGINE's closed set, not a literal list of five names.
    # The gate is allowlist membership, so an unlisted state is silently clean;
    # a test that enumerated the states itself would agree with the gate about
    # a sixth state neither of them had ever seen.
    assert {
        state: _discarded_post_capture_content(_synthetic_outcome(state))
        for state in RESTORE_OBSERVATION_STATES
    } == {
        RESTORE_OBSERVATION_DIFFERS: True,
        RESTORE_OBSERVATION_NO_LIVE_STATE: False,
        RESTORE_OBSERVATION_PRESENT_NOT_COMPARABLE: True,
        RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED: False,
        RESTORE_OBSERVATION_NOT_RECORDED: True,
    }


def test_every_state_the_gate_fails_a_run_for_says_so_on_its_line() -> None:
    """A non-zero exit whose reason appears nowhere in the printed report.

    That is what these two sets drifting apart produces, and they drifted once
    already: the delete leg's state fired the gate while rendering a line
    indistinguishable from a quiet member's. Derived from the engine's closed
    vocabulary on both sides, so the guard sees a sixth state instead of
    agreeing with the code about a state neither has met.
    """
    flagged = {
        state
        for state in RESTORE_OBSERVATION_STATES
        if _restore_outcome_line(_synthetic_outcome(state))
        != _restore_outcome_line(_synthetic_outcome(RESTORE_OBSERVATION_NO_WRITE_ATTEMPTED))
    }
    fires = {
        state
        for state in RESTORE_OBSERVATION_STATES
        if _discarded_post_capture_content(_synthetic_outcome(state))
    }
    assert flagged == fires
    assert fires  # a vacuous pass if both sets were somehow empty


def test_restore_help_says_the_flag_reports_and_cannot_prevent(capsys) -> None:
    """The flag is named for its exit behavior and ``--help`` says why: it is
    read after every member has already been written, so it can never fence
    one, and it does not displace the codes the restore already returns."""
    with pytest.raises(SystemExit) as excinfo:
        workspace_main(["restore", "--help"])
    assert excinfo.value.code == 0
    # argparse re-wraps help to the terminal width, so match on the prose with
    # its line breaks normalized away rather than on a formatted line.
    out = " ".join(capsys.readouterr().out.split())
    assert DISCARD_FLAG in out
    assert "cannot prevent the write" in out
    assert "still exits 3" in out


# --- duplicate-name disclosure --------------------------------------------------


def test_checkpoint_duplicate_name_is_disclosed_not_refused(
    tmp_path: Path, capsys
) -> None:
    _seed_file(tmp_path)
    rc, out, _ = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path)
    )
    assert rc == 0
    assert "prior checkpoint" not in out  # first mint: nothing to disclose
    first_id = _checkpoint_id(capsys, tmp_path)

    rc, out, _ = _run(
        capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path)
    )
    assert rc == 0  # disclosed, never refused — names are labels
    assert first_id in out
    assert "not unique keys" in out

    rc, out, _ = _run(
        capsys,
        "checkpoint",
        "cp1",
        "--file",
        "docs/plan.md",
        "--json",
        "--root",
        str(tmp_path),
    )
    assert rc == 0
    existing = json.loads(out)["existing_with_same_name"]
    assert first_id in existing
    assert len(existing) == 2


# --- live-coordinator dual-write warning ----------------------------------------


def test_restore_warns_when_live_coordinator_is_serving(tmp_path: Path, capsys) -> None:
    plan = _seed_file(tmp_path)
    _run(capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path))
    ckpt = _checkpoint_id(capsys, tmp_path)
    plan.write_bytes(b"corrupted\n")
    # Fake a LIVE pidfile: this test process's own pid is provably alive.
    (tmp_path / ".coherence" / "server.pid").write_text(f"{os.getpid()}\n12345\n")

    rc, out, err = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0  # WARN, never refuse (v1)
    assert "live hook coordinator" in err
    assert "bypass" in err
    payload = json.loads(out)
    assert payload["live_coordinator_warning"] is True
    assert plan.read_bytes() == b"plan v1\n"  # the restore still ran


def test_restore_does_not_warn_without_a_live_coordinator(
    tmp_path: Path, capsys
) -> None:
    plan = _seed_file(tmp_path)
    _run(capsys, "checkpoint", "cp1", "--file", "docs/plan.md", "--root", str(tmp_path))
    ckpt = _checkpoint_id(capsys, tmp_path)
    plan.write_bytes(b"corrupted\n")
    # A malformed pidfile (no parseable pid) is NOT a live coordinator.
    (tmp_path / ".coherence" / "server.pid").write_text("garbage\n")

    rc, out, err = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    assert "live hook coordinator" not in err
    assert "live_coordinator_warning" not in json.loads(out)


# --- honesty placement #2: the --help constraint note ---------------------------


def test_help_carries_the_constraint_note(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        workspace_main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "single-host" in out
    assert "no-arbiter" in out
    assert "detection-only" in out
    assert "cross-host" in out


# --- the e2e example: exit-code contract + determinism --------------------------


def _run_example(*flags: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(EXAMPLE), *flags],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO_ROOT),
    )


def test_example_baseline_arm_contract_and_determinism() -> None:
    """Exit 0 iff baseline-shows-loss AND guarded-prevents; two consecutive
    runs are byte-identical (offline + deterministic). The origin
    Success-Criterion-1 trio must be VISIBLE in the run output."""
    first = _run_example("--baseline")
    second = _run_example("--baseline")
    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0
    assert first.stdout == second.stdout  # deterministic across consecutive runs

    out = first.stdout
    assert "LOSS demonstrated" in out  # baseline-shows-loss
    # Success-Criterion 1: delete leg + forward-only skip + S3 conflict visible.
    assert "delete marker minted" in out
    assert "outcome=forward_only_skipped" in out
    assert "outcome=conflict" in out
    assert "outcome=restored" in out
    # Honesty labels ride the report.
    assert "no-arbiter" in out


def test_example_default_arm_guarded_only() -> None:
    result = _run_example()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Negative control" not in result.stdout
    assert "outcome=restored" in result.stdout


# --- the published claim: what a restore promises about post-capture content ----
#
# A restore puts captured bytes back OVER whatever is live. The engine reports
# that; it does not refuse it. Two guide tables carry that claim to anyone
# deciding whether to trust this verb — the restore outcome table and the CLI
# exit-code table — and both of them said something weaker before this run's
# signal existed ("the captured bytes landed via the member's conditional
# write"; "concluded with every member clean"). These pins are what keep the
# published text and the behaviour from drifting apart again.
#
# Every match runs over WHITESPACE-NORMALIZED guide text. The guide's tables
# are single very long lines today, but a re-wrap (or an editor's reflow) would
# defeat a line-wise search while leaving the sentence intact and correct — a
# false RED — and, worse, a half-deleted sentence could pass a shorter search.
# Each phrase below is therefore ONE contiguous fragment carrying a whole claim.

_GUIDE_PATH = REPO_ROOT / "docs" / "guide.md"

#: The corrected statements, keyed by what each one promises the reader.
_GUIDE_RESTORE_CLAIMS: dict[str, str] = {
    "the restored outcome says what the write landed over": (
        "over whatever was live at that moment, including content committed "
        "after the capture; the report below says, per member, what that write "
        "discarded"
    ),
    "restore is not a merge and nothing refuses the write": (
        "A restore is not a merge, and nothing on this path refuses a write: a "
        "member whose content moved after the capture is put back over, and "
        "that later content is gone"
    ),
    "exit 0 is not a claim that nothing was overwritten": (
        "never a claim that nothing was overwritten: a restore that put a "
        "member back over content committed after the capture also exits `0`, "
        "and says so per member in the report"
    ),
    "exit 4 has a row, and it is opt-in and after the fact": (
        "the restore concluded clean by the codes above, but at least one "
        "member's write discarded content the checkpoint did not hold, or the "
        "run holds no record of what that member's write overwrote"
    ),
}

#: Wordings the tables carried while the claim was wrong. A guide that reverts
#: to one of these has to fail even if the corrected sentence were also present
#: somewhere — the reader hits the table, not the search.
_RETIRED_GUIDE_WORDINGS: tuple[str, ...] = (
    "concluded with every member clean",
    "| `restored` | the captured bytes landed via the member's conditional write |",
)


def _normalized(text: str) -> str:
    """Collapse every run of whitespace, so a re-wrapped sentence still matches."""
    return re.sub(r"\s+", " ", text)


@pytest.fixture(scope="module")
def guide_text() -> str:
    return _normalized(_GUIDE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "phrase",
    list(_GUIDE_RESTORE_CLAIMS.values()),
    ids=list(_GUIDE_RESTORE_CLAIMS),
)
def test_guide_states_what_restore_promises(guide_text: str, phrase: str) -> None:
    """Each corrected table statement is present, wrapping notwithstanding."""
    assert _normalized(phrase) in guide_text, (
        "docs/guide.md no longer carries this restore claim.\n"
        f"missing text: {phrase!r}"
    )


@pytest.mark.parametrize("wording", _RETIRED_GUIDE_WORDINGS)
def test_guide_does_not_revert_to_the_over_claim(guide_text: str, wording: str) -> None:
    """The retired wordings never come back — a clean exit is not a clean workspace."""
    assert _normalized(wording) not in guide_text, (
        f"docs/guide.md reverted to the over-claiming wording: {wording!r}"
    )


# --- #191: a checkpoint bound to another receiver ------------------------------


def test_restore_of_a_checkpoint_bound_to_another_receiver_is_refused(
    tmp_path: Path, capsys
) -> None:
    """The CLI owns one controller per workspace root, so a checkpoint naming a
    different receiver (created through another path) is a typed exit-2
    refusal before the restore writes anything — never a stuck in_progress."""
    ckpt = _fabricate_checkpoint(
        tmp_path,
        CheckpointMember(
            member_path="effects/notify",
            artifact_id=None,
            native_token=None,
            fingerprint=None,
            captured_at=1.0,
            arbitration_tier=ArbitrationTier.NO_ARBITER.value,
            restore_tier=RestoreTier.FORWARD_ONLY.value,
        ),
        receiver=uuid4(),
    )
    rc, out, err = _run(capsys, "restore", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 2
    envelope = json.loads(out)
    assert envelope["kind"] == "error"
    assert envelope["reason"] == "not_the_receiver"
    assert "not_the_receiver" in err
    rc, out, _ = _run(capsys, "status", ckpt, "--json", "--root", str(tmp_path))
    assert rc == 0
    assert json.loads(out)["checkpoint"]["restore_status"] == "none"
