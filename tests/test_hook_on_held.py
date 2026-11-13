"""The hook client's refusal of held strict files: the workspace setting that
turns it on, and the strict decision it makes the way the coordinator does."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from ccs.adapters.claude_code import hook_payloads
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


def _decision_rows() -> list[dict]:
    return json.loads((_FIXTURES / "decision_table.json").read_text())["rows"]


def _answered_rows() -> list[dict]:
    """The rows whose opted-in POST was answered 2xx: the answer alone decides
    them. Every other row ends before the POST or gets no 2xx, and is driven
    through the real hook client instead."""
    return [
        row for row in _decision_rows()
        if row["stage"] == "post" and "http" in row and 200 <= row["http"]["status"] < 300
    ]


def _request(row: dict) -> dict:
    """The opted-in request the row's caller sends, as the hook client builds it."""
    caller = row["caller"]
    request = {"session_id": caller["session_id"], "path": "docs/plan.md", "if_unheld": True}
    if caller["agent_id"] is not None:
        request["agent_id"] = caller["agent_id"]
    return request


def test_every_class_the_decision_table_names_after_the_post_has_an_answered_row() -> None:
    classes = {row["expected"] for row in _answered_rows()}

    assert classes == {"relay_admit", "relay_deny", "holder_deny", "parent_deny", "unavailable_deny"}


@pytest.mark.parametrize("row", _answered_rows(), ids=lambda row: row["name"])
def test_each_answered_row_is_classified_as_the_decision_table_says(row: dict) -> None:
    assert on_held.classify_answer(row["http"]["body"], _request(row)) == row["expected"]


def _deny(row: dict) -> dict:
    body, path = row["http"]["body"], "docs/plan.md"
    built = {
        "holder_deny": lambda: hook_payloads.emit_held_path_deny(
            path=path, holder_agent_id=body["holder_agent_id"]
        ),
        "parent_deny": lambda: hook_payloads.emit_held_by_parent_deny(path=path),
        "unavailable_deny": lambda: hook_payloads.emit_admission_unavailable_deny(path=path),
    }[row["expected"]]()
    return {"hookSpecificOutput": built}


@pytest.mark.parametrize("row", _answered_rows(), ids=lambda row: row["name"])
def test_each_answered_row_prints_the_answer_or_the_deny_its_class_names(row: dict) -> None:
    body = row["http"]["body"]

    printed = on_held.answer_for(body, _request(row))

    if row["expected"].startswith("relay_"):
        assert json.dumps(printed) == json.dumps(body)
    else:
        assert printed == _deny(row)


@pytest.mark.parametrize("body", [None, [], "ok", 7], ids=repr)
def test_an_answer_that_is_not_an_object_is_the_unavailable_class(body: object) -> None:
    request = {"session_id": "5f0c2a8e-3b1d-4c7a-9e2f-6a1b8d4c0e71", "path": "docs/plan.md"}

    assert on_held.classify_answer(body, request) == "unavailable_deny"


_SID = "5f0c2a8e-3b1d-4c7a-9e2f-6a1b8d4c0e71"
_PARENT = "bcac3323fcc3598c80dcdac488109608"  # the agent id of _SID's own session


def test_a_holder_named_under_another_reason_is_not_read_as_a_holder_refusal() -> None:
    body = {"ok": False, "reason": "handoff_other_holder", "holder_agent_id": "bc3013e123b458509bb8c8875e97ff53"}

    assert on_held.classify_answer(body, {"session_id": _SID, "path": "docs/plan.md"}) == "unavailable_deny"


def test_only_a_subagent_caller_is_told_its_parent_holds_the_file() -> None:
    body = {"ok": False, "reason": "other_holder", "holder_agent_id": _PARENT}
    parent_session = {"session_id": _SID, "path": "docs/plan.md"}
    subagent = {**parent_session, "agent_id": "a3f9c2e1b7d4"}

    assert on_held.classify_answer(body, subagent) == "parent_deny"
    assert on_held.classify_answer(body, parent_session) == "holder_deny"


@pytest.mark.parametrize(
    ("status", "strict"),
    [
        ({"fold_profile": "none"}, False),
        ({"fold_profile": "caseless"}, True),
        ({}, False),
        ({"fold_profile": "cannot_tell"}, False),
        ({"fold_profile": "folds_everything"}, True),
        ({"fold_profile": None}, True),
        ({"fold_profile": ["caseless"]}, True),
    ],
    ids=["none", "caseless", "absent-reads-as-none", "cannot-tell-keys-as-none", "unrecognised", "null", "not-a-string"],
)
def test_the_coordinators_fold_profile_decides_strict_and_an_unknown_one_refuses(
    status: dict, strict: bool
) -> None:
    candidates = frozenset({"caseless"})

    assert on_held.strict_for_coordinator(status, candidates) is strict


@pytest.mark.parametrize(
    ("value", "advertised"),
    [(True, True), ("true", False), (1, False), (None, False), (False, False)],
    ids=repr,
)
def test_only_json_true_advertises_refusing_held_files(value: object, advertised: bool) -> None:
    assert on_held.advertises_refusal({"pre_edit_if_unheld": value}) is advertised
    assert on_held.advertises_refusal({}) is False


@pytest.mark.parametrize(
    ("cc", "malformed"),
    [
        ({"agent_id": "a3f9c2e1b7d4"}, False),
        ({"agentId": "a3f9c2e1b7d4"}, False),
        ({}, False),
        ({"agent_id": None}, False),
        ({"agent_id": ""}, False),
        ({"agent_id": "bad id!"}, True),
        ({"agent_id": "a3f9/c2e1"}, True),
        ({"agent_id": "x" * 65}, True),
        ({"agent_id": "abc\n"}, True),
        ({"agent_id": 7}, True),
        ({"agentId": ["a"]}, True),
        ({"agent_id": None, "agentId": "bad id!"}, False),
    ],
    ids=repr,
)
def test_a_present_subagent_id_outside_the_coordinators_grammar_is_malformed(
    cc: dict, malformed: bool
) -> None:
    assert on_held.malformed_subagent_id(cc) is malformed


def test_the_stderr_line_for_a_coordinator_that_does_not_refuse_is_one_line() -> None:
    assert "\n" not in on_held.NO_REFUSAL_MESSAGE
    assert on_held.NO_REFUSAL_MESSAGE.isascii()
