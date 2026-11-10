# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Ended-attempt corpus: a session-stop that reports an attempt ended answers
the same on both runtimes (#525).

A ``CoherentVolume`` sends ``"ended": true`` with an attempt's ``agent_id`` on
``POST /hooks/session-stop`` once it no longer uses that attempt. The stop
answers exactly as one without the flag, and afterwards ``GET /status`` no
longer lists the attempt. These rows observe the listing only: that the
coordinator deletes the attempt's rows is pinned by the registry tests on each
backend. ``"ended": false`` answers as a stop without the field. ``ended:
true`` without an ``agent_id`` answers HTTP 400 "ended requires agent_id", and
an ``ended`` that is not a JSON boolean, null included, answers HTTP 400
"ended must be a boolean".

The ``/status`` rows compare ``sessions`` only. The two backends' ``/status``
envelopes differ in their other keys, so every other top-level key is dropped
on both sides (``optional_keys``). The ended row has a control that
sends the same requests without the flag and still lists the attempt, so an
empty or shortened listing cannot pass for a forget. The ``"ended": false``
row sends the control's requests with the flag set false and expects the
control's listing, so a backend that ends an identity on any boolean fails.

Marked ``protocol_corpus``: opt-in via ``pytest -m protocol_corpus``. A Node
row that cannot run FAILS rather than xfails: a row that never ran reports
green while checking nothing."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from tests.protocol_corpus.harness import (
    ALL_BACKENDS,
    BACKEND_NODE,
    Fixture,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

_FIXTURE_DIR = "ended_attempt"
_EXPECTED_FIXTURE_COUNT = 7
_STOP_PATH = "/hooks/session-stop"
_ANSWER_ROW = "ended-stop-answers-as-a-plain-stop"
_ENDED_STATUS_ROW = "ended-stop-removes-the-attempt-from-status"
_CONTROL_STATUS_ROW = "plain-subagent-stop-keeps-the-identity-in-status"
_FALSE_STATUS_ROW = "ended-false-stop-keeps-the-identity-in-status"

_NODE_DIST_PATH = resolve_node_dist_path()
_NODE_DIST_UNRESOLVED = (
    "This row needs a real answer from the Node coordinator and got none: the "
    "plugin dist could not be resolved. Failing rather than xfailing is "
    "deliberate, since a row that never ran reports green while checking "
    "nothing. Build the plugin (npm ci && npm run build) or set "
    "AGENT_COHERENCE_PLUGIN_DIST_PATH to the absolute dist/coordinator.js path."
)


def _fixtures() -> list[Fixture]:
    return load_fixtures(_FIXTURE_DIR)


def _by_name(name: str) -> Fixture:
    return next(f for f in _fixtures() if f.name == name)


_ROWS = [(f, b) for f in _fixtures() for b in f.backends]


@pytest.mark.parametrize("row", _ROWS, ids=[f"{f.name}[{b}]" for f, b in _ROWS])
def test_ended_attempt_fixture_response_matches_expected(
    row: tuple[Fixture, str], tmp_path: Path
) -> None:
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


def test_fixture_directory_is_actually_loaded() -> None:
    """A misspelled or emptied fixture directory loads as nothing and a
    parametrize over nothing reports green; the frozen count makes it red."""
    assert len(_fixtures()) == _EXPECTED_FIXTURE_COUNT


def test_every_row_runs_on_both_backends() -> None:
    assert all(f.backends == ALL_BACKENDS for f in _fixtures())


def test_the_node_rows_cannot_be_satisfied_by_a_skip() -> None:
    assert _NODE_DIST_PATH is not None, _NODE_DIST_UNRESOLVED
    assert _NODE_DIST_PATH.exists()


def test_the_status_rows_compare_the_session_listing_and_nothing_else() -> None:
    """Dropping ``sessions`` with the other keys would leave an empty body on
    both sides, equal whatever the coordinator lists."""
    status_rows = [f for f in _fixtures() if f.request["path"] == "/status"]
    assert {f.name for f in status_rows} == {
        _ENDED_STATUS_ROW, _CONTROL_STATUS_ROW, _FALSE_STATUS_ROW,
    }
    for fixture in status_rows:
        assert set(fixture.expected["body"]) == {"sessions"}, fixture.name
        assert "sessions" not in fixture.optional_keys, fixture.name
        assert not fixture.ignore_keys, fixture.name


def _without_ended(fixture: Fixture) -> dict:
    """The fixture's setup with ``ended`` removed from every stop it sends."""
    setup = copy.deepcopy(fixture.setup)
    for request in setup["preflight_requests"]:
        if request["path"] == _STOP_PATH:
            request["body"].pop("ended", None)
    return setup


def test_the_control_sends_the_ended_rows_requests_without_the_flag() -> None:
    """The control proves the listing would show the attempt; it proves that
    only while it differs from the ended row by the flag alone."""
    ended, control = _by_name(_ENDED_STATUS_ROW), _by_name(_CONTROL_STATUS_ROW)
    assert ended.setup != control.setup  # the flag is there to remove
    assert _without_ended(ended) == control.setup
    assert ended.optional_keys == control.optional_keys
    assert len(control.expected["body"]["sessions"]) == 1 + len(
        ended.expected["body"]["sessions"]
    )


def test_the_false_row_is_the_control_with_the_flag_set_false() -> None:
    """The false row pins that ``"ended": false`` ends nothing only while it
    sends the control's requests with the flag set false and expects the
    control's listing."""
    false_row, control = _by_name(_FALSE_STATUS_ROW), _by_name(_CONTROL_STATUS_ROW)
    assert false_row.setup["preflight_requests"][-1]["body"]["ended"] is False
    assert _without_ended(false_row) == control.setup
    assert false_row.request == control.request
    assert false_row.expected == control.expected
    assert false_row.optional_keys == control.optional_keys


def test_the_answer_row_sends_the_stop_the_status_row_runs() -> None:
    """The answer row's request is the ended row's last preflight, sent after
    the same setup, so the answer pinned is the one whose forget the status row
    observes."""
    answer, ended = _by_name(_ANSWER_ROW), _by_name(_ENDED_STATUS_ROW)
    *before, stop = ended.setup["preflight_requests"]
    assert answer.setup["preflight_requests"] == before
    assert {k: answer.setup[k] for k in ("tracked", "files")} == {
        k: ended.setup[k] for k in ("tracked", "files")
    }
    assert answer.request == stop
    assert answer.request["body"]["ended"] is True
