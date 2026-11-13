# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Cross-implementation strict-mode wire-shape corpus tests (plan Unit 7b).

Parametrized over (fixture, backend) like the warn-mode corpus. Each fixture
runs on the backends its ``"backends"`` list names: most strict rows run on
both the Python coordinator and the Claude Code plugin's Node coordinator,
and a row scoped to one backend says why in its description.

Marked ``protocol_corpus`` — opt-in via ``pytest -m protocol_corpus``."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    Fixture,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus


def _all_strict_mode_fixtures() -> list[Fixture]:
    return load_fixtures("strict_mode")


def _parametrize_rows() -> list[tuple[Fixture, str]]:
    rows: list[tuple[Fixture, str]] = []
    for fixture in _all_strict_mode_fixtures():
        for backend in fixture.backends:
            rows.append((fixture, backend))
    return rows


def _row_id(row: tuple[Fixture, str]) -> str:
    fixture, backend = row
    return f"{fixture.name}[{backend}]"


_ROWS = _parametrize_rows()
_NODE_DIST_PATH = resolve_node_dist_path()


@pytest.mark.parametrize("row", _ROWS, ids=[_row_id(r) for r in _ROWS] if _ROWS else None)
def test_strict_mode_fixture_response_matches_expected(
    row: tuple[Fixture, str],
    tmp_path: Path,
) -> None:
    fixture, backend = row
    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.xfail(
            "Node coordinator dist not resolvable. Build the plugin checkout "
            "(npm ci && npm run build) or set AGENT_COHERENCE_PLUGIN_DIST_PATH."
        )

    actual_status, actual_body = run_scenario(
        fixture=fixture,
        backend_id=backend,
        workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )

    expected_status = fixture.expected["status"]
    expected_body = normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
    )

    assert actual_status == expected_status, (
        f"{fixture.name}[{backend}]: status mismatch — "
        f"expected {expected_status}, got {actual_status}\n"
        f"body={actual_body!r}"
    )
    assert actual_body == expected_body, (
        f"{fixture.name}[{backend}]: body mismatch\n"
        f"expected={expected_body!r}\nactual=  {actual_body!r}"
    )


def test_collection_loaded_strict_mode_fixtures() -> None:
    """Self-test: at least 5 strict-mode fixtures are present so parametrize
    doesn't silently no-op. Catches the failure mode where the fixtures
    directory is empty."""
    fixtures = _all_strict_mode_fixtures()
    assert len(fixtures) >= 5, (
        f"Expected ≥5 strict-mode fixtures, found {len(fixtures)}. "
        f"Add coverage in tests/protocol_corpus/fixtures/strict_mode/."
    )
