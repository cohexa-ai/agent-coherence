# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for TrackedArtifactPolicy (plan Unit 3).

Covers:
- Default-pattern matching across language ecosystems (cross-repo benchmark
  precursor — Unit 8 will do the 1000-path version against real samples)
- opt-in via tracked.yaml, opt-out via ignored.yaml
- Path-traversal guard rejecting absolute paths and `..` components
- Malformed YAML / unreadable files degrade gracefully
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ccs.adapters.claude_code.policy import (
    DEFAULT_TRACKED_PATTERNS,
    TrackedArtifactPolicy,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / ".coherence").mkdir()
    return tmp_path


# --------------------------------------------------------------------
# Default-pattern behavior
# --------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected_tracked",
    [
        # Defaults that SHOULD match
        ("CLAUDE.md", True),
        ("AGENTS.md", True),
        ("docs/specs/foo.md", True),
        ("docs/specs/deep/nested/spec.md", True),
        ("docs/plans/2026-05-13-001-feat.md", True),
        ("docs/brainstorms/idea.md", True),
        ("plan.md", True),
        ("spec.md", True),
        ("task.md", True),
        ("subdir/plan.md", True),
        ("a/b/c/spec.md", True),
        # Cross-language ecosystem files that MUST NOT match defaults
        ("README.md", False),
        ("src/main.py", False),
        ("package.json", False),
        ("node_modules/some-pkg/plan.md", True),  # plan.md at any depth — tracked by design
        ("Cargo.toml", False),
        ("src/lib.rs", False),
        ("models/user.py", False),
        ("requirements.txt", False),
        ("Dockerfile", False),
        ("CONTRIBUTING.md", False),
        (".github/workflows/ci.yml", False),
        ("docs/api/v1.md", False),  # docs/api/ is not in defaults
        ("docs/getting-started.md", False),  # docs/ root not in defaults
        ("scripts/build.sh", False),
        ("tests/test_foo.py", False),
    ],
)
def test_defaults(root: Path, path: str, expected_tracked: bool) -> None:
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_tracked(path) is expected_tracked, (
        f"path {path!r}: expected tracked={expected_tracked}, "
        f"got {policy.is_tracked(path)} (patterns: {DEFAULT_TRACKED_PATTERNS})"
    )


# --------------------------------------------------------------------
# tracked.yaml opt-in
# --------------------------------------------------------------------


def test_opt_in_via_tracked_yaml(root: Path) -> None:
    (root / ".coherence" / "tracked.yaml").write_text(
        "- src/important/*.py\n- runbook.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_tracked("src/important/foo.py")
    assert policy.is_tracked("runbook.md")
    # Defaults still apply
    assert policy.is_tracked("CLAUDE.md")
    # Unrelated paths still untracked
    assert not policy.is_tracked("src/main.py")


# --------------------------------------------------------------------
# ignored.yaml opt-out
# --------------------------------------------------------------------


def test_opt_out_via_ignored_yaml(root: Path) -> None:
    (root / ".coherence" / "ignored.yaml").write_text("- docs/brainstorms/**/*.md\n")
    policy = TrackedArtifactPolicy.load(root)
    # Default match suppressed by ignore
    assert not policy.is_tracked("docs/brainstorms/draft.md")
    # Other defaults unaffected
    assert policy.is_tracked("CLAUDE.md")
    assert policy.is_tracked("docs/specs/foo.md")


def test_ignore_wins_over_tracked(root: Path) -> None:
    """Ignore patterns override opt-in patterns (defensive default)."""
    (root / ".coherence" / "tracked.yaml").write_text("- src/special/*.py\n")
    (root / ".coherence" / "ignored.yaml").write_text("- src/special/*.py\n")
    policy = TrackedArtifactPolicy.load(root)
    assert not policy.is_tracked("src/special/foo.py")


# --------------------------------------------------------------------
# Dotfile paths — regression coverage for ce-review P1 finding #1
# --------------------------------------------------------------------
#
# Before the fix at policy.py:136, _normalize_relative used p.lstrip("./")
# which strips a SET of characters {".", "/"}, silently mangling dotfile
# paths (.env → env, .gitignore → gitignore). A user-added dotfile pattern
# in tracked.yaml stored as ".env" but normalized to "env" on is_tracked()
# check, making it unmatchable. The fix uses removeprefix("./") which only
# strips the literal "./" prefix.
#
# These parametrize entries would have caught the bug had they existed.


@pytest.mark.parametrize("dotfile_path", [
    ".env",
    ".gitignore",
    ".hidden/plan.md",
    ".coherence/state.db",
    "subdir/.env",
])
def test_dotfile_paths_track_correctly_when_added(root: Path, dotfile_path: str) -> None:
    """A user-added dotfile pattern in tracked.yaml must match the same
    dotfile path on is_tracked() — verifies the removeprefix fix on
    _normalize_relative."""
    (root / ".coherence" / "tracked.yaml").write_text(f"- {dotfile_path}\n")
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_tracked(dotfile_path), (
        f"dotfile {dotfile_path!r} added to tracked.yaml but is_tracked() returns False "
        f"— check _normalize_relative's prefix-stripping (must NOT use lstrip)"
    )


# --------------------------------------------------------------------
# Path-traversal guard (security-lens review)
# --------------------------------------------------------------------


def test_traversal_pattern_rejected(root: Path) -> None:
    """tracked.yaml with `../../.env` or `/etc/passwd` patterns must be
    rejected by the loader and not applied."""
    (root / ".coherence" / "tracked.yaml").write_text(
        "- '../../.env'\n- '/etc/passwd'\n- 'CLAUDE.md'\n"
    )
    policy = TrackedArtifactPolicy.load(root)
    # The good pattern in the same file still applies
    assert policy.is_tracked("CLAUDE.md")
    # The malicious patterns were rejected and recorded
    rejections = {p for p, _ in policy.rejected()}
    assert "../../.env" in rejections
    assert "/etc/passwd" in rejections
    # And they don't sneak through is_tracked() either
    assert not policy.is_tracked("../../.env")
    assert not policy.is_tracked("/etc/passwd")


def test_absolute_path_never_tracked(root: Path) -> None:
    """A path beginning with `/` is never tracked, even if some pattern
    would match its tail. is_tracked operates on parent-repo-relative paths."""
    policy = TrackedArtifactPolicy.load(root)
    assert not policy.is_tracked("/etc/CLAUDE.md")
    assert not policy.is_tracked("/Users/x/repo/plan.md")


def test_traversal_in_query_path_never_tracked(root: Path) -> None:
    """A query path containing `..` is never tracked, regardless of patterns."""
    policy = TrackedArtifactPolicy.load(root)
    assert not policy.is_tracked("../escapee/CLAUDE.md")
    assert not policy.is_tracked("docs/../../secret/CLAUDE.md")


# --------------------------------------------------------------------
# YAML degradation
# --------------------------------------------------------------------


def test_missing_yaml_files_use_defaults(root: Path) -> None:
    # No tracked.yaml, no ignored.yaml — just defaults.
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_tracked("CLAUDE.md")
    assert policy.user_added_patterns == ()
    assert policy.ignored_patterns == ()


def test_malformed_yaml_warns_and_uses_defaults(root: Path, caplog: pytest.LogCaptureFixture) -> None:
    (root / ".coherence" / "tracked.yaml").write_text("- valid\n- ['unterminated\n")
    policy = TrackedArtifactPolicy.load(root)
    assert "malformed YAML" in caplog.text or "could not read" in caplog.text
    # Defaults still apply
    assert policy.is_tracked("CLAUDE.md")
    # No user-added patterns from the malformed file
    assert policy.user_added_patterns == ()


def test_non_list_top_level_rejected(root: Path, caplog: pytest.LogCaptureFixture) -> None:
    (root / ".coherence" / "tracked.yaml").write_text("just-a-string\n")
    policy = TrackedArtifactPolicy.load(root)
    assert "must be a list of patterns" in caplog.text
    assert policy.user_added_patterns == ()


@pytest.mark.parametrize(
    ("file_name", "content", "says"),
    [
        ("tracked.yaml", "- runbook.md\n- *.md\n", "not valid YAML"),
        ("ignored.yaml", "- runbook.md\n- *.md\n", "not valid YAML"),
        ("strict_mode.yaml", "- data/**\n- *.md\n", "not valid YAML"),
        ("tracked.yaml", "just-a-string\n", "not a list"),
        ("tracked.yaml", b"- caf\xe9.md\n", "not valid YAML"),
        ("tracked.yaml", "- runbook.md\n- 2026-13-45\n", "not valid YAML"),
        ("tracked.yaml", "- runbook.md\n- !!bool x\n", "not valid YAML"),
        ("strict_mode.yaml", "- data/**\n- !!timestamp x\n", "not valid YAML"),
        ("ignored.yaml", "- runbook.md\n- !!int\n", "not valid YAML"),
        ("tracked.yaml", "[" * 2000 + "]" * 2000 + "\n", "not valid YAML"),
    ],
)
def test_a_policy_file_that_loads_as_no_entries_is_counted_as_rejected(
    root: Path, file_name: str, content: str | bytes, says: str
) -> None:
    """#284: a policy file that loads as no entries is reported as one rejected
    entry naming the file, so /status shows a non-zero rejected_pattern_count.
    Before, the only signal was a WARNING a detached coordinator sends to
    /dev/null, and /status read as a healthy workspace with no user patterns.
    Read as UTF-8 whatever the locale: bytes that are not UTF-8, an impossible
    date, a core tag on a value it cannot take and deep nesting are a file
    that does not load, not an exception out of the loader."""
    target = root / ".coherence" / file_name
    if isinstance(content, bytes):
        target.write_bytes(content)
    else:
        target.write_text(content)
    policy = TrackedArtifactPolicy.load(root)
    assert policy.summary()["rejected_pattern_count"] == 1
    [(pattern, reason)] = policy.rejected()
    assert pattern == f".coherence/{file_name}"
    assert says in reason and "loads as no entries" in reason


def test_a_policy_file_is_decoded_by_its_bytes_not_the_locale(root: Path) -> None:
    """The loader hands PyYAML the file's bytes, which it decodes as UTF-8, or
    as UTF-16 by a byte-order mark, whatever the locale, so the loader reads
    the encoding the writer writes. Reading text in the locale's encoding
    could load a different pattern from the one written, or none."""
    (root / ".coherence" / "tracked.yaml").write_bytes(
        "- runbook.md\n- café.md\n".encode("utf-16")
    )
    policy = TrackedArtifactPolicy.load(root)
    assert policy.user_added_patterns == ("runbook.md", "café.md")
    assert policy.rejected() == ()


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root reads a mode-000 file"
)
def test_an_unreadable_policy_file_is_counted_as_rejected(root: Path) -> None:
    """A policy file that exists but cannot be read loads as no entries too,
    and is reported the same way."""
    target = root / ".coherence" / "tracked.yaml"
    target.write_text("- runbook.md\n")
    target.chmod(0)
    try:
        policy = TrackedArtifactPolicy.load(root)
    finally:
        target.chmod(0o600)
    assert policy.rejected() == (
        (".coherence/tracked.yaml", "could not be read, so the file loads as no entries"),
    )


def test_non_string_pattern_in_list_rejected(root: Path) -> None:
    (root / ".coherence" / "tracked.yaml").write_text("- 42\n- runbook.md\n")
    policy = TrackedArtifactPolicy.load(root)
    # The bad entry is rejected; the good one survives.
    assert policy.is_tracked("runbook.md")
    rejections = {p for p, _ in policy.rejected()}
    assert "42" in rejections


# --------------------------------------------------------------------
# Path normalization
# --------------------------------------------------------------------


def test_leading_dot_slash_normalized(root: Path) -> None:
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_tracked("./CLAUDE.md")
    assert policy.is_tracked("./docs/specs/foo.md")


def test_empty_path_not_tracked(root: Path) -> None:
    policy = TrackedArtifactPolicy.load(root)
    assert not policy.is_tracked("")


# --------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------


def test_summary_includes_counts(root: Path) -> None:
    (root / ".coherence" / "tracked.yaml").write_text("- runbook.md\n- '/etc/passwd'\n")
    (root / ".coherence" / "ignored.yaml").write_text("- docs/brainstorms/**/*.md\n")
    policy = TrackedArtifactPolicy.load(root)
    summary = policy.summary(include_patterns=True)
    assert summary["default_pattern_count"] == len(DEFAULT_TRACKED_PATTERNS)
    assert summary["user_added_pattern_count"] == 1  # runbook.md
    assert summary["ignored_pattern_count"] == 1  # docs/brainstorms/**/*.md
    assert summary["rejected_pattern_count"] == 1  # /etc/passwd
    assert summary["user_added_patterns"] == ["runbook.md"]


def test_summary_user_added_patterns_empty_by_default(root: Path) -> None:
    policy = TrackedArtifactPolicy.load(root)
    summary = policy.summary(include_patterns=True)
    assert summary["user_added_patterns"] == []
