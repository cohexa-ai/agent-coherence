"""The hook client's refusal of held strict files: the workspace setting that
turns it on, and the strict decision it makes the way the coordinator does."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from ccs.adapters.claude_code.policy import TrackedArtifactPolicy
from ccs.cli import _hook_on_held as on_held

_FIXTURES = Path(__file__).parent / "fixtures" / "hook_on_held"


def _setting_rows() -> list[dict]:
    return json.loads((_FIXTURES / "setting_values.json").read_text())["rows"]


def _workspace(tmp_path: Path) -> Path:
    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    return tmp_path


@pytest.mark.parametrize("row", _setting_rows(), ids=lambda row: repr(row["content"]))
def test_each_setting_value_reads_as_its_documented_state(tmp_path: Path, row: dict) -> None:
    root = _workspace(tmp_path)
    if row["content"] is not None:
        (root / ".coherence" / "hook_on_held").write_bytes(row["content"].encode())

    assert on_held.read_setting(root / ".coherence") == row["expected"]


def test_a_missing_coherence_directory_reads_as_off(tmp_path: Path) -> None:
    assert on_held.read_setting(tmp_path / ".coherence") == "off"


def test_a_linked_coherence_directory_reads_as_off(tmp_path: Path) -> None:
    real = tmp_path / "elsewhere"
    real.mkdir()
    (real / "hook_on_held").write_text("refuse\n")
    (tmp_path / ".coherence").symlink_to(real)

    assert on_held.read_setting(tmp_path / ".coherence") == "off"


def test_a_linked_setting_file_reads_as_off(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    target = tmp_path / "target"
    target.write_text("refuse\n")
    (root / ".coherence" / "hook_on_held").symlink_to(target)

    assert on_held.read_setting(root / ".coherence") == "off"


def test_an_oversized_setting_file_is_an_unknown_value(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    (root / ".coherence" / "hook_on_held").write_text("refuse\n" * 64)

    assert on_held.read_setting(root / ".coherence") == "unknown"


def _strict_workspace(tmp_path: Path, strict: list[str]) -> Path:
    root = _workspace(tmp_path)
    (root / ".coherence" / "tracked.yaml").write_text('- "plan.md"\n')
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "".join(f'- "{entry}"\n' for entry in strict)
    )
    (root / "plan.md").write_text("plan\n")
    return root


def test_a_case_variant_is_strict_only_under_the_caseless_profile(tmp_path: Path) -> None:
    root = _strict_workspace(tmp_path, ["plan.md"])
    policy = TrackedArtifactPolicy.read_quietly(root)

    profiles = on_held.strict_profiles(policy, root, "PLAN.MD")

    assert "caseless" in profiles
    assert "none" not in profiles


def test_the_exact_spelling_is_strict_under_every_profile(tmp_path: Path) -> None:
    root = _strict_workspace(tmp_path, ["plan.md"])
    policy = TrackedArtifactPolicy.read_quietly(root)

    assert on_held.strict_profiles(policy, root, "plan.md") == frozenset({"none", "caseless", "nfd"})


def test_a_path_no_strict_entry_covers_is_strict_under_no_profile(tmp_path: Path) -> None:
    root = _strict_workspace(tmp_path, ["plan.md"])
    policy = TrackedArtifactPolicy.read_quietly(root)

    assert on_held.strict_profiles(policy, root, "notes.md") == frozenset()


def test_a_workspace_with_no_strict_entries_is_strict_under_no_profile(tmp_path: Path) -> None:
    root = _strict_workspace(tmp_path, [])
    policy = TrackedArtifactPolicy.read_quietly(root)

    assert on_held.strict_profiles(policy, root, "plan.md") == frozenset()


def test_reading_the_policy_quietly_logs_nothing(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    root = _strict_workspace(tmp_path, ["plan.md"])
    real = tmp_path / "real_ignored.yaml"
    real.write_text('- "plan.md"\n')
    os.symlink(real, root / ".coherence" / "ignored.yaml")

    with caplog.at_level(logging.WARNING, logger="ccs.adapters.claude_code.policy"):
        TrackedArtifactPolicy.load(root)
    assert caplog.records, "the loud load must warn here, or this test proves nothing"
    caplog.clear()

    with caplog.at_level(logging.WARNING, logger="ccs.adapters.claude_code.policy"):
        quiet = TrackedArtifactPolicy.read_quietly(root)

    assert caplog.records == []
    assert quiet.strict_mode_paths == ("plan.md",)
