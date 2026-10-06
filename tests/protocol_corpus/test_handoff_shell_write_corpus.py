# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The handoff corpus row for the giver's shell write (#185), pinned.

``fixtures/handoff/39-giver-shell-write-is-denied-handed-off.json`` records
what ``/hooks/pre-bash`` answers the giver of a live handoff that appends to
the handed-off path through the shell: the giver's pre-edit deny, byte for
byte. Python-only -- the Node coordinator serves no handoff route, so it holds
no record that could fence a giver.

The row is replayed twice: as recorded, and with its request swapped for the
giver's pre-edit on the same path, which must answer the same expected body.
The second run is what ties the shell route's bytes to the edit route's, so a
deny the shell route built for itself fails here even if the row were
re-recorded from it.

The handoff corpus module that runs every row of ``fixtures/handoff/`` lives
with the handoff client work; when the two meet, this row joins its frozen
fixture count and its per-route giver rows, and this module folds into it.

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.protocol_corpus.harness import (
    BACKEND_PYTHON,
    Fixture,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

_FIXTURE_DIR = "handoff"
_SHELL_WRITE_ROW = "handoff-giver-shell-write-is-denied-handed-off"

# Resolved at import like every corpus module's, though this row never runs on
# Node: an explicit dist path that does not exist must stop the whole corpus at
# collection (tests/protocol_corpus/test_harness_node_dist_path.py).
_NODE_DIST_PATH = resolve_node_dist_path()


def _shell_write_row() -> Fixture:
    rows = [f for f in load_fixtures(_FIXTURE_DIR) if f.name == _SHELL_WRITE_ROW]
    assert len(rows) == 1, f"expected exactly one {_SHELL_WRITE_ROW!r} row, found {len(rows)}"
    return rows[0]


def _expected(fixture: Fixture) -> dict:
    return normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
        preserve_identity=fixture.preserve_identity,
    )


def test_the_givers_shell_write_row_is_python_only_and_names_a_shell_write() -> None:
    """The row runs where a handoff exists, and its request is a shell write
    that reads nothing -- so only the write check can answer it a deny."""
    fixture = _shell_write_row()
    assert fixture.backends == (BACKEND_PYTHON,)
    assert fixture.request["path"] == "/hooks/pre-bash"
    assert fixture.request["body"]["command"] == "echo '- gamma' >> plan.md"


def test_the_givers_shell_write_row_matches_expected(tmp_path: Path) -> None:
    """The FULL body is the assertion: a changed template byte, a dropped or
    added key, or the parties swapped all fail here."""
    fixture = _shell_write_row()
    status, body = run_scenario(
        fixture=fixture, backend_id=BACKEND_PYTHON, workspace=tmp_path, node_dist_path=_NODE_DIST_PATH,
    )
    assert status == fixture.expected["status"], body
    assert body == _expected(fixture)


def test_the_givers_pre_edit_answers_the_shell_write_rows_body(tmp_path: Path) -> None:
    """The same setup, the giver's pre-edit on the handed-off path instead of
    its shell write: the same expected body. Fails if the shell route's deny
    drifts from the edit route's."""
    fixture = _shell_write_row()
    pre_edit = dataclasses.replace(fixture, request={
        **fixture.request,
        "path": "/hooks/pre-edit",
        "body": {"session_id": fixture.request["body"]["session_id"], "path": "plan.md"},
    })
    status, body = run_scenario(
        fixture=pre_edit, backend_id=BACKEND_PYTHON, workspace=tmp_path, node_dist_path=_NODE_DIST_PATH,
    )
    assert status == fixture.expected["status"], body
    assert body == _expected(fixture)
