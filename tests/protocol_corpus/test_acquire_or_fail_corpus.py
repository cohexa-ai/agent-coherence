# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Acquire-or-fail corpus: an opted-in pre-edit answers the same on both runtimes.

A ``POST /hooks/pre-edit`` that sends ``if_unheld: true`` asks to take the path
only if no other agent identity holds it EXCLUSIVE or MODIFIED. While one does,
the answer is ``{"ok": false, "reason": "other_holder", "holder_agent_id":
"<32 lowercase hex>"}`` and nothing is granted. Every row here runs on both
backends: the refusals (held EXCLUSIVE, held MODIFIED, sibling subagents, a
parent and its subagent either way) and the answers the opt-in must leave alone
(the holder's own repeat, an unheld path, the field absent or not JSON
``true``, an untracked path, SHARED readers only, an ordinary pre-edit
displacing an opted-in holder, and a stale strict-mode editor getting the
strict deny before any holder check).

The holder is pinned literally in each refusal row: the identity scrub matches
only the dashed spelling, so the 32-hex agent id is compared as written. The
held-EXCLUSIVE row's raw answer is also held to the wire contract the Python
coordinator ships (#466): its keys in order, and no envelope at any depth.

Principals. Python refuses an opted-in pre-edit without the caller principal,
whatever session it names, and refuses any request without it once that
session is claimed. Node issues no principal. So a session that opts in is
claimed in a Python-scoped preflight, and every later request naming it
presents the principal as a whole header value, which the harness omits on
Node. The harness ignores a preflight that answers 400, so a row whose setup
Python silently refused would test a different path and could still pass; the
load-time check in ``_fixtures`` refuses such a fixture before it runs.

The 30-minute reclaim has no row: the harness cannot move a Node coordinator's
clock, and Python never reclaims on acquire.

Marked ``protocol_corpus``: opt-in via ``pytest -m protocol_corpus``. A Node
row that cannot run FAILS rather than xfails: a row that never ran reports
green while checking nothing."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.protocol_corpus.harness import (
    ALL_BACKENDS,
    BACKEND_NODE,
    Fixture,
    FixtureContractError,
    _omit_headers_not_issued,
    _substitute,
    apply_preflight_requests,
    apply_setup,
    coordinator_running,
    execute_request,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

_FIXTURE_DIR = "acquire_or_fail"
_EXPECTED_FIXTURE_COUNT = 13
_HEADER = "Coherence-Caller-Principal"  # frozen duplicate of the wire name
_CLAIM_PATH = "/principal/claim"
_HELD_ROW = "acquire-or-fail-held-exclusive-refuses-an-opted-in-peer"
_REPEAT_ROW = "acquire-or-fail-holders-own-repeat-is-admitted"

_NODE_DIST_PATH = resolve_node_dist_path()
_NODE_DIST_UNRESOLVED = (
    "This row needs a real answer from the Node coordinator and got none: the "
    "plugin dist could not be resolved. Failing rather than xfailing is "
    "deliberate, since a row that never ran reports green while checking "
    "nothing. Build the plugin (npm ci && npm run build) or set "
    "AGENT_COHERENCE_PLUGIN_DIST_PATH to the absolute dist/coordinator.js path."
)


def _check_principals(fixture: Fixture) -> None:
    """Refuse a fixture whose Python setup would be silently refused.

    A session's Python-scoped claim must come before any request naming that
    session, and every later request naming it, preflights included, must
    present the claim's capture as the whole principal header value."""
    requests = [*(fixture.setup.get("preflight_requests") or []), fixture.request]
    named: set[str] = set()
    captures: dict[str, str] = {}
    for i, req in enumerate(requests):
        session = (req.get("body") or {}).get("session_id")
        where = f"{fixture.name}: request #{i} ({req.get('path')})"
        if req.get("path") == _CLAIM_PATH:
            if session in named:
                raise FixtureContractError(
                    f"{where} claims {session} after a request already named it; "
                    f"Python refuses or admits that request on other terms")
            capture = list((req.get("capture") or {}).items())
            if req.get("backends") != ["python"] or [v for _, v in capture] != ["principal"]:
                raise FixtureContractError(
                    f"{where}: a claim here is scoped to python and captures the principal")
            captures[session] = capture[0][0]
        elif session in captures:
            presented = (req.get("headers") or {}).get(_HEADER)
            if presented != f"${{{captures[session]}}}":
                raise FixtureContractError(
                    f"{where} names claimed session {session} without its principal "
                    f"header; Python answers it 400 and a preflight's 400 is ignored")
        named.add(session)


def _fixtures() -> list[Fixture]:
    fixtures = load_fixtures(_FIXTURE_DIR)
    for fixture in fixtures:
        _check_principals(fixture)
    return fixtures


def _by_name(name: str) -> Fixture:
    return next(f for f in _fixtures() if f.name == name)


_ROWS = [(f, b) for f in _fixtures() for b in f.backends]


@pytest.mark.parametrize("row", _ROWS, ids=[f"{f.name}[{b}]" for f, b in _ROWS])
def test_acquire_or_fail_fixture_response_matches_expected(
    row: tuple[Fixture, str], tmp_path: Path
) -> None:
    """The full response body is the assertion, the holder's agent id
    included."""
    fixture, backend = row
    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.fail(f"{fixture.name}: {_NODE_DIST_UNRESOLVED}")
    status, body = run_scenario(
        fixture=fixture, backend_id=backend, workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )
    expected = normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
    )
    assert status == fixture.expected["status"], f"{fixture.name}[{backend}]: {body!r}"
    assert body == expected, f"{fixture.name}[{backend}]\nexpected={expected!r}\nactual=  {body!r}"


def _run_raw(fixture: Fixture, backend: str, workspace: Path) -> tuple[int, dict[str, Any]]:
    """``run_scenario`` without the normaliser, which sorts keys."""
    apply_setup(workspace, fixture.setup)
    with coordinator_running(backend, workspace, _NODE_DIST_PATH) as live:
        captures = apply_preflight_requests(live, fixture.setup["preflight_requests"])
        request = _substitute(
            _omit_headers_not_issued(fixture.request, captures), captures,
            where=f"{fixture.name} request",
        )
        return execute_request(live, request)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_the_held_refusal_honours_the_wire_contract(backend: str, tmp_path: Path) -> None:
    """The raw refusal arrives as ``ok, reason, holder_agent_id`` in that
    order, with no envelope or degraded marker at any depth, and names the
    pinned holder."""
    from tests.test_pre_edit_other_holder_contract import _contract_breaches

    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.fail(_NODE_DIST_UNRESOLVED)
    fixture = _by_name(_HELD_ROW)
    status, raw = _run_raw(fixture, backend, tmp_path)
    assert status == 200, raw
    assert list(raw) == ["ok", "reason", "holder_agent_id"], raw
    assert _contract_breaches(raw) == [], raw
    assert raw["holder_agent_id"] == fixture.expected["body"]["holder_agent_id"], raw


def test_fixture_directory_is_actually_loaded() -> None:
    """A misspelled or emptied fixture directory loads as nothing and a
    parametrize over nothing reports green; the frozen count makes it red."""
    assert len(_fixtures()) == _EXPECTED_FIXTURE_COUNT


def test_every_row_runs_on_both_backends() -> None:
    assert all(f.backends == ALL_BACKENDS for f in _fixtures())


def test_the_node_rows_cannot_be_satisfied_by_a_skip() -> None:
    assert _NODE_DIST_PATH is not None, _NODE_DIST_UNRESOLVED
    assert _NODE_DIST_PATH.exists()


def _without_header(fixture: Fixture, index: int) -> Fixture:
    setup = copy.deepcopy(fixture.setup)
    del setup["preflight_requests"][index]["headers"]
    return replace(fixture, setup=setup)


def test_the_principal_check_refuses_a_setup_python_would_refuse() -> None:
    """Without its header, the holder's first pre-edit would be a 400 on
    Python that the harness ignores, and the row would test an unheld path.
    A claim moved after the first request naming its session is refused too."""
    repeat = _by_name(_REPEAT_ROW)
    _check_principals(repeat)  # the control passes

    holder_pre_edit = 1
    assert repeat.setup["preflight_requests"][holder_pre_edit]["path"] == "/hooks/pre-edit"
    with pytest.raises(FixtureContractError, match="without its principal header"):
        _check_principals(_without_header(repeat, holder_pre_edit))

    setup = copy.deepcopy(repeat.setup)
    setup["preflight_requests"].reverse()
    with pytest.raises(FixtureContractError, match="after a request already named it"):
        _check_principals(replace(repeat, setup=setup))
