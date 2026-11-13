# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The shared fixtures for an edit the hook client opts in to acquire-or-fail.

``tests/fixtures/hook_on_held/`` is read by the hook client's tests and copied
byte for byte by the Node hook client, so each rendered deny there must be what
the builders render, and the decision table and the setting's value vectors
must hold every row the two clients are checked against, with nothing in them
that contradicts the rules they document.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from ccs.adapters.claude_code import hook_payloads as hp
from ccs.adapters.claude_code.coordinator_server import caller_principal_identity

_FIXTURES = Path(__file__).parent / "fixtures" / "hook_on_held"

_RENDERED_DENIES: dict[str, Callable[..., dict[str, Any]]] = {
    "holder_deny.json": hp.emit_held_path_deny,
    "parent_deny.json": hp.emit_held_by_parent_deny,
    "unavailable_deny.json": hp.emit_admission_unavailable_deny,
}

#: The outcome classes a decision-table row may expect.
_OUTCOME_CLASSES = frozenset({
    "today",
    "today_and_stderr_line",
    "relay_admit",
    "relay_deny",
    "holder_deny",
    "parent_deny",
    "unavailable_deny",
})

#: Every exit of the opted-in decision before the POST is sent, by name.
_EXITS_BEFORE_THE_POST = frozenset({
    "setting_off",
    "setting_unknown",
    "not_strict",
    "no_port",
    "died_at_resolve",
    "timeout",
    "refused",
    "malformed_answer",
    "status_non_2xx",
    "status_key_absent",
    "not_strict_under_coordinator_profile",
    "malformed_agent_id",
})


def _load(name: str) -> dict[str, Any]:
    # ASCII, so a non-ASCII byte in a file the Node client copies fails here.
    return json.loads((_FIXTURES / name).read_text("ascii"))


def _rows(name: str) -> list[dict[str, Any]]:
    fixture = _load(name)
    assert isinstance(fixture["description"], str) and fixture["description"]
    return fixture["rows"]


@pytest.mark.parametrize("name", sorted(_RENDERED_DENIES))
def test_each_rendered_deny_is_what_its_builder_renders(name: str) -> None:
    """Compared parsed, because Python and Node serialise JSON differently,
    and the reason as exact text, because those are the bytes the model
    reads."""
    fixture = _load(name)
    built = {"hookSpecificOutput": _RENDERED_DENIES[name](**fixture["inputs"])}
    reason = fixture["stdout"]["hookSpecificOutput"]["permissionDecisionReason"]
    assert built["hookSpecificOutput"]["permissionDecisionReason"] == reason
    assert built == fixture["stdout"]


@pytest.mark.parametrize("name", sorted(_RENDERED_DENIES))
def test_each_rendered_deny_is_the_bare_deny_envelope(name: str) -> None:
    """Built like the strict deny: no top-level ``reason`` or ``decision``
    beside the envelope, and nothing in it but the deny."""
    stdout = _load(name)["stdout"]
    assert list(stdout) == ["hookSpecificOutput"]
    assert stdout["hookSpecificOutput"] == {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": stdout["hookSpecificOutput"]["permissionDecisionReason"],
    }


def test_every_decision_table_row_names_a_caller_one_cause_and_a_class() -> None:
    rows = _rows("decision_table.json")
    assert len({row["name"] for row in rows}) == len(rows)
    for row in rows:
        assert set(row["caller"]) == {"session_id", "agent_id"}, row["name"]
        assert row["stage"] in {"before_post", "post"}, row["name"]
        assert ("http" in row) != ("outcome" in row), row["name"]
        assert row["expected"] in _OUTCOME_CLASSES, row["name"]
    assert {row["expected"] for row in rows} == _OUTCOME_CLASSES


def test_the_decision_table_has_one_row_per_exit_before_the_post() -> None:
    exits = [row["outcome"] for row in _rows("decision_table.json") if row["stage"] == "before_post"]
    assert sorted(exits) == sorted(_EXITS_BEFORE_THE_POST)


def test_the_parent_deny_rows_are_held_by_the_callers_own_session() -> None:
    """A subagent is refused with the parent text only when the holder is its
    own session's agent id; a sibling, and a parent refused by its own
    subagent, get the holder text. The rows must say so, or the two clients
    would be checked against a table that contradicts the rule."""
    other_holder_rows = [
        row for row in _rows("decision_table.json")
        if row.get("http", {}).get("body", {}).get("reason") == "other_holder"
        and row["expected"] in {"parent_deny", "holder_deny"}
    ]
    assert {row["expected"] for row in other_holder_rows} == {"parent_deny", "holder_deny"}
    for row in other_holder_rows:
        caller = row["caller"]
        own_session = caller_principal_identity(caller["session_id"]).hex
        is_parent = bool(caller["agent_id"]) and row["http"]["body"]["holder_agent_id"] == own_session
        assert (row["expected"] == "parent_deny") == is_parent, row["name"]


def _setting_class(content: str | None) -> str:
    """The setting's documented grammar: one trailing newline is dropped, then
    the value is matched exactly."""
    if content is None:
        return "off"
    value = content[:-1] if content.endswith("\n") else content
    if value == "refuse":
        return "on"
    return "off" if value in {"", "displace"} else "unknown"


def test_every_setting_vector_follows_the_documented_grammar() -> None:
    rows = _rows("setting_values.json")
    assert [(row["content"], row["expected"]) for row in rows] == [
        (row["content"], _setting_class(row["content"])) for row in rows
    ]


def test_the_setting_vectors_cover_the_documented_values() -> None:
    contents = {row["content"] for row in _rows("setting_values.json")}
    assert {
        None, "", "displace", "refuse", "refuse\n",
        "Refuse", " refuse", "refuse\r\n", "refuse\n\n",
    } <= contents
