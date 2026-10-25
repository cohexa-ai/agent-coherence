"""The wire contract of an opted-in pre-edit refused for a timeout.

A ``POST /hooks/pre-edit`` that opts in with ``if_unheld: true`` and that the
coordinator cannot decide in time is refused, never admitted.
``tests/fixtures/wire/pre_edit_unavailable.json`` pins that answer's structure:
every key, and every value a client branches on. The advisory's wording is left
out (``<IGNORED>``), so another implementation need not copy this one's text,
but the advisory must be there. Hook clients and other coordinators compare
their answer against this one file.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Callable

import pytest

import ccs.adapters.claude_code.coordinator_server as coordinator_server
from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from tests.protocol_corpus.harness import IGNORED_SENTINEL, normalize_response
from tests.test_claude_code_coordinator_server import (
    _DEGRADE_DEADLINE_SEC,
    _claimed,
    _Client,
    _count_binding_reads,
    _pre_edit_request,
    _timed_out_at_its_gate,
    _timed_out_in_its_work,
)

_CONTRACT = Path(__file__).parent / "fixtures" / "wire" / "pre_edit_unavailable.json"
#: The one key whose value the contract leaves to each implementation.
_ADVISORY = frozenset({"additionalContext"})

#: FROZEN copy of what an ordinary timed-out pre-edit answers, in its wire key
#: order, so a change to the default answer fails here too.
_DEFAULT_TIMED_OUT_PRE_EDIT = {
    "ok": True,
    "degraded": True,
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "additionalContext": (
            "⚠ Coherence could not verify this file's freshness — the coordinator "
            "staleness check timed out under load. Proceeding WITHOUT a stale-read "
            "guarantee: if this file is shared with other agents or sessions, "
            "re-read it before relying on its contents."
        ),
    },
}


@pytest.fixture
def live_coordinator(tmp_path: Path):
    """A live coordinator on a random port."""
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="test-instance")
    server.serve_in_thread()
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def live_client(live_coordinator) -> _Client:
    secret = load_secret(live_coordinator.coordinator_root)
    assert secret is not None
    return _Client("127.0.0.1", live_coordinator.port, secret)


def _structure(answer: dict) -> dict:
    """``answer`` with the advisory's wording masked, as the contract compares it."""
    return normalize_response(answer, ignore_keys=_ADVISORY)


def _contract() -> dict:
    return _structure(json.loads(_CONTRACT.read_text(encoding="utf-8")))


def _keys_at_any_depth(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys_at_any_depth(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys_at_any_depth(v) for v in value))
    return set()


def _assert_honours_the_contract(answer: dict) -> None:
    assert _structure(answer) == _contract(), answer
    assert "permissionDecision" not in _keys_at_any_depth(answer), answer
    advisory = answer["hookSpecificOutput"]["additionalContext"]
    assert isinstance(advisory, str) and advisory.strip(), answer


def test_the_contract_masks_only_the_advisory_wording() -> None:
    """The committed contract leaves the advisory's text open and nothing else,
    so it cannot quietly stop pinning a key a client reads."""
    raw = json.loads(_CONTRACT.read_text(encoding="utf-8"))
    assert raw["hookSpecificOutput"]["additionalContext"] == IGNORED_SENTINEL
    masked = [k for k, v in raw.items() if v == IGNORED_SENTINEL]
    assert masked == [], f"the contract masks {masked}, which a client reads"


def test_the_opted_in_timeout_answer_honours_the_contract() -> None:
    """The coordinator's named answer for an opted-in timed-out pre-edit is the
    contract's shape: a refusal with the advisory and no permission decision."""
    _assert_honours_the_contract(coordinator_server._PRE_EDIT_UNAVAILABLE_RESPONSE)


def test_an_opted_in_pre_edit_timed_out_in_its_work_answers_the_contract(
    live_coordinator, live_client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the route sends when the work body cannot take the registry lock in
    time is the contract's shape, not only the constant it names."""
    who = _claimed(live_client)
    monkeypatch.setattr(coordinator_server, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)

    answer = _timed_out_in_its_work(live_coordinator, live_client, _pre_edit_request(who, True))

    _assert_honours_the_contract(answer)


def test_an_opted_in_pre_edit_timed_out_at_its_gate_answers_the_contract(
    live_coordinator, live_client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the route sends when the caller-principal gate cannot read the
    store in time is the contract's shape too."""
    who = _claimed(live_client)
    monkeypatch.setattr(coordinator_server, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    reads = _count_binding_reads(live_coordinator, monkeypatch)

    answer = _timed_out_at_its_gate(live_coordinator, live_client, _pre_edit_request(who, True))

    assert reads, "the request never reached the gate's store read"
    _assert_honours_the_contract(answer)


def test_the_default_timed_out_pre_edit_answer_is_unchanged() -> None:
    """An ordinary timed-out pre-edit still answers today's degraded admit, in
    the same bytes: the opted-in refusal must not change what every other
    caller gets."""
    assert json.dumps(coordinator_server._PRE_EDIT_DEGRADED_RESPONSE) == json.dumps(
        _DEFAULT_TIMED_OUT_PRE_EDIT)


def _add_top_level_key(answer: dict) -> None:
    answer["retry_after"] = 1


def _rename_reason(answer: dict) -> None:
    answer["reasons"] = answer.pop("reason")


def _set_value(path: tuple[str, ...], value: object) -> Callable[[dict], None]:
    def mutate(answer: dict) -> None:
        *parents, key = path
        target = answer
        for parent in parents:
            target = target[parent]
        target[key] = value
    return mutate


_BREAKING_CHANGES: list[tuple[str, Callable[[dict], None]]] = [
    ("added-key", _add_top_level_key),
    ("dropped-degraded", lambda answer: answer.pop("degraded")),
    ("renamed-reason", _rename_reason),
    ("ok-true", _set_value(("ok",), True)),
    ("other-reason", _set_value(("reason",), "other_holder")),
    ("not-degraded", _set_value(("degraded",), False)),
    ("other-event", _set_value(("hookSpecificOutput", "hookEventName"), "PostToolUse")),
    ("permission-decision", _set_value(("hookSpecificOutput", "permissionDecision"), "deny")),
    ("dropped-advisory", lambda answer: answer["hookSpecificOutput"].pop("additionalContext")),
]


@pytest.mark.parametrize(
    "change", [c for _, c in _BREAKING_CHANGES], ids=[name for name, _ in _BREAKING_CHANGES])
def test_the_contract_catches_a_changed_key_or_value(change: Callable[[dict], None]) -> None:
    """A key added, dropped or renamed, or a value a client branches on
    changed, no longer matches the contract."""
    answer = copy.deepcopy(coordinator_server._PRE_EDIT_UNAVAILABLE_RESPONSE)
    change(answer)
    assert _structure(answer) != _contract(), answer


def test_the_contract_leaves_the_advisory_wording_open() -> None:
    """Different advisory text still matches, so another implementation can
    word it its own way."""
    answer = copy.deepcopy(coordinator_server._PRE_EDIT_UNAVAILABLE_RESPONSE)
    answer["hookSpecificOutput"]["additionalContext"] = "The check timed out; refused."
    assert _structure(answer) == _contract()
