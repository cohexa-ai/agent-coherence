"""The wire contract of an opted-in pre-edit refused while another agent holds the path.

A ``POST /hooks/pre-edit`` that opts in with ``if_unheld: true``, decided in
time while another agent identity holds the path EXCLUSIVE or MODIFIED, is
refused with ``other_holder`` and names that holder.
``tests/fixtures/wire/pre_edit_other_holder.json`` pins that answer: its keys,
their order, and every value a client branches on. The holder's value differs
on every run, so it is masked there and its spelling is checked on its own: 32
lowercase hex characters, the agent id, never a session id. It is the shape any
other implementation of this answer has to match.

The corpus normaliser sorts keys and its identity scrubber matches only the
dashed spelling, so a structure comparison alone would pass a reordered body or
a holder spelled the wrong way. The contract check therefore also compares the
key order and the holder's spelling.
"""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Callable

import pytest

import ccs.adapters.claude_code.coordinator_server as server_module
from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    CoordinatorHTTPServer,
    session_to_agent_id,
)
from ccs.core.exceptions import HANDOFF_OTHER_HOLDER_REASON, OTHER_HOLDER_REASON
from tests.protocol_corpus.harness import IGNORED_SENTINEL, normalize_response
from tests.test_claude_code_coordinator_server import (
    _DEGRADE_DEADLINE_SEC,
    _GATE_ANSWER_BOUND_SEC,
    _OPERATOR_HEADER,
    _Background,
    _claimed,
    _Client,
    _hand_off,
    _HeldRegistryLock,
    _opted_in,
    _take,
)

_CONTRACT = Path(__file__).parent / "fixtures" / "wire" / "pre_edit_other_holder.json"
#: The one key whose value the contract leaves to each run.
_HOLDER = frozenset({"holder_agent_id"})
#: Carried only on a path with a transfer record, and then last.
_HANDOFF = "handoff"
_HOLDER_SPELLING = re.compile(r"[0-9a-f]{32}")
#: Keys that would make the refusal read as a deny envelope or a degraded admit.
_FORBIDDEN = frozenset({"hookSpecificOutput", "permissionDecision", "collision", "degraded"})

#: FROZEN copy of a refusal in its wire key order, which the breaking changes
#: below start from.
_REFUSAL = {
    "ok": False,
    "reason": "other_holder",
    "holder_agent_id": "5d0f3c9a8b7e4d21a6c3b2e1f0d9c8b7",
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


def _raw_contract() -> dict:
    return json.loads(_CONTRACT.read_text(encoding="utf-8"))


def _structure(answer: dict) -> str:
    """``answer`` with the holder's value masked and any ``handoff`` key left
    out, compared as JSON text: Python's ``0 == False`` would let a wrong wire
    type pass a dict comparison."""
    normalized = normalize_response(
        answer, ignore_keys=_HOLDER, optional_keys=frozenset({_HANDOFF}))
    return json.dumps(normalized, sort_keys=True)


def _keys_at_any_depth(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys_at_any_depth(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys_at_any_depth(v) for v in value))
    return set()


def _contract_breaches(answer: dict, *, with_handoff: bool = False) -> list[str]:
    """The contract checks ``answer`` fails, by name; empty when it honours
    the contract. ``with_handoff`` expects the ``handoff`` key a path with a
    transfer record adds, after the contract's own keys."""
    expected_order = [*_raw_contract(), *([_HANDOFF] if with_handoff else [])]
    holder = answer.get("holder_agent_id")
    checks = {
        "structure": _structure(answer) == _structure(_raw_contract()),
        "key order": list(answer) == expected_order,
        "holder spelling": isinstance(holder, str) and bool(_HOLDER_SPELLING.fullmatch(holder)),
        "no envelope or degraded marker": not _FORBIDDEN & _keys_at_any_depth(answer),
    }
    return [name for name, held in checks.items() if not held]


def test_the_contract_masks_only_the_holder() -> None:
    """The committed contract leaves the holder's value open and nothing else,
    so it cannot quietly stop pinning a key a client reads."""
    raw = _raw_contract()
    masked = [k for k, v in raw.items() if v == IGNORED_SENTINEL]
    assert masked == ["holder_agent_id"], f"the contract masks {masked}"


@pytest.mark.parametrize("subagent", [None, "sub1"], ids=["session", "subagent"])
def test_a_live_refusal_honours_the_contract_and_names_the_holders_agent_id(
    subagent: str | None, live_client: _Client
) -> None:
    """What the route sends an opted-in pre-edit while another agent holds the
    path is the contract's shape, and its holder is that agent's id as 32
    lowercase hex: never a session id, in either spelling.

    Prevents a client that matches on the fixture meeting a different body,
    and a holder the client cannot compare with the agent ids it knows."""
    holder, caller = _claimed(live_client), _claimed(live_client)
    _take(live_client, holder, **({"agent_id": subagent} if subagent else {}))

    status, answer = _opted_in(live_client, caller)

    assert status == 200
    assert _contract_breaches(answer) == [], answer
    assert answer["reason"] == OTHER_HOLDER_REASON
    named = answer["holder_agent_id"]
    assert named == session_to_agent_id(holder.sid, subagent).hex
    session_spellings = {s for who in (holder, caller) for s in (who.sid, uuid.UUID(who.sid).hex)}
    assert named not in session_spellings


def test_a_live_refusal_on_a_path_with_a_transfer_record_ends_with_the_handoff_key(
    live_client: _Client,
) -> None:
    """On a path whose handoff completed, the refusal a bystander gets has the
    contract's keys in order, then ``handoff`` as the one extra key, last.

    Prevents the transfer record's key landing among the contract's keys,
    which a client reading the body in order would trip over."""
    giver, successor, bystander = (_claimed(live_client) for _ in range(3))
    _hand_off(live_client, giver, successor, "plan.md")
    _take(live_client, successor)

    status, answer = _opted_in(live_client, bystander)

    assert status == 200
    assert _contract_breaches(answer, with_handoff=True) == [], answer
    assert answer["holder_agent_id"] == session_to_agent_id(successor.sid).hex


def test_the_frozen_refusal_honours_the_contract() -> None:
    """The breaking changes below start from a body that honours the
    contract, so each one's failure is the change's doing."""
    assert _contract_breaches(_REFUSAL) == []


def _reordered(answer: dict) -> dict:
    return dict(reversed(list(answer.items())))


def _dashed_holder(answer: dict) -> dict:
    return {**answer, "holder_agent_id": str(uuid.UUID(answer["holder_agent_id"]))}


def _uppercase_holder(answer: dict) -> dict:
    return {**answer, "holder_agent_id": answer["holder_agent_id"].upper()}


def _without_holder(answer: dict) -> dict:
    return {k: v for k, v in answer.items() if k != "holder_agent_id"}


def _with(key: str, value: object) -> Callable[[dict], dict]:
    def change(answer: dict) -> dict:
        return {**answer, key: value}
    return change


#: The first three pass the structure comparison alone: the normaliser sorts
#: keys and masks the holder whatever its spelling.
_BREAKING_CHANGES: list[tuple[str, Callable[[dict], dict]]] = [
    ("reordered-keys", _reordered),
    ("dashed-holder", _dashed_holder),
    ("uppercase-holder", _uppercase_holder),
    ("hook-specific-output", _with(
        "hookSpecificOutput", {"hookEventName": "PreToolUse", "additionalContext": "refused"})),
    ("degraded", _with("degraded", True)),
    ("ok-zero", _with("ok", 0)),
    ("other-reason", _with("reason", HANDOFF_OTHER_HOLDER_REASON)),
    ("missing-holder", _without_holder),
]


@pytest.mark.parametrize(
    "change", [c for _, c in _BREAKING_CHANGES], ids=[name for name, _ in _BREAKING_CHANGES])
def test_the_contract_catches_a_changed_key_order_or_value(
    change: Callable[[dict], dict],
) -> None:
    """A key reordered, added or dropped, a value a client branches on
    changed, or the holder spelled another way no longer honours the
    contract."""
    changed = change(_REFUSAL)
    assert _contract_breaches(changed) != [], changed


def test_the_contract_catches_a_deny_envelope_inside_the_handoff_key() -> None:
    """The ``handoff`` key is left out of the structure comparison, so the
    check for a deny envelope at any depth is what sees inside it."""
    changed = {**_REFUSAL, _HANDOFF: {"hookSpecificOutput": {"permissionDecision": "deny"}}}
    assert _contract_breaches(changed, with_handoff=True) == ["no envelope or degraded marker"]


@pytest.mark.parametrize(
    "query, headers",
    [
        ("/status", None),
        ("/status?detail=metrics", None),
        ("/status?detail=full", _OPERATOR_HEADER),
    ],
    ids=["minimal", "metrics", "full"],
)
def test_every_status_tier_advertises_the_refusal(
    query: str, headers: dict | None, live_client: _Client
) -> None:
    """A client reads ``pre_edit_if_unheld`` to tell this coordinator from one
    that ignores ``if_unheld`` and answers an opted-in request by displacing
    the holder. It is JSON ``true``, not a truthy value, at every tier.

    Prevents a client that reads one tier finding nothing there and refusing a
    coordinator that would have honoured the flag."""
    status, body = live_client.get(query, headers_override=headers)
    assert status == 200
    assert body["pre_edit_if_unheld"] is True


@pytest.mark.parametrize(
    "query, headers",
    [("/status", None), ("/status?detail=full", _OPERATOR_HEADER)],
    ids=["minimal", "full"],
)
def test_a_degraded_status_still_advertises_the_refusal(
    query: str,
    headers: dict | None,
    live_coordinator: CoordinatorHTTPServer,
    live_client: _Client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registry read that timed out still answers the key: the fact is about
    this build, so a busy registry cannot change it.

    Prevents a volume attaching during contention being told its coordinator
    does not honour the flag."""
    monkeypatch.setattr(server_module, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    with _HeldRegistryLock(live_coordinator) as held:
        answer = _Background(live_client, query, headers=headers)
        answered = answer.answered_within(_GATE_ANSWER_BOUND_SEC)
        held.release()
    status, body = answer.result()
    assert answered, "/status did not answer while the registry lock was held"
    assert (status, body["degraded"]) == (200, True)
    assert body["pre_edit_if_unheld"] is True
