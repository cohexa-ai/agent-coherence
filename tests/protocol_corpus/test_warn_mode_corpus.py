# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Cross-implementation warn-mode wire-shape corpus tests (plan Unit 7a).

Parametrized over (fixture, backend). Each row runs the fixture's request
against the named backend in an isolated tmp workspace and asserts the
normalized response matches the fixture's ``expected`` block.

Marked ``protocol_corpus`` — opt-in via ``pytest -m protocol_corpus``. Skipped
in default runs (see ``pyproject.toml`` ``addopts``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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


def _all_warn_mode_fixtures() -> list[Fixture]:
    """Loaded once at collection so parametrize ids stay stable."""
    return load_fixtures("warn_mode")


def _parametrize_rows() -> list[tuple[Fixture, str]]:
    """Cartesian product of (fixture, backend) restricted to each fixture's
    declared backends list."""
    rows: list[tuple[Fixture, str]] = []
    for fixture in _all_warn_mode_fixtures():
        for backend in fixture.backends:
            rows.append((fixture, backend))
    return rows


def _row_id(row: tuple[Fixture, str]) -> str:
    fixture, backend = row
    return f"{fixture.name}[{backend}]"


_ROWS = _parametrize_rows()
_NODE_DIST_PATH = resolve_node_dist_path()


@pytest.mark.parametrize("row", _ROWS, ids=[_row_id(r) for r in _ROWS] if _ROWS else None)
def test_warn_mode_fixture_response_matches_expected(
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

    # Pre-normalize the EXPECTED body too — fixtures can be authored either
    # with the post-normalization sentinels already in place ("<TS>", "<UUID>")
    # or with realistic-looking placeholders. Round-trip both through the same
    # normalizer so author choice doesn't affect equality.
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


#: The release baselines the #185 grant handoff is measured against, by
#: fixture name: two pin the release answers, three check that the release
#: frees the path, and one is the collision control those three rely on.
#: Named rather than numbered, because a count floor stays green when one of
#: them goes missing while other rows land.
_RELEASE_BASELINES = frozenset({
    "post-edit-failure-releases-exclusive-grant",
    "session-stop-releases-two-held-paths",
    "post-edit-failure-release-frees-the-path",
    "session-stop-release-frees-the-held-paths",
    "session-stop-release-frees-the-first-held-path",
    "pre-edit-collision-between-sessions",
})


def test_collection_loaded_fixtures() -> None:
    """Self-test: the fixtures are present so parametrize doesn't silently
    no-op. Catches the failure mode where the fixtures directory is empty or
    path-resolution is wrong, and a release baseline that goes missing or
    stops running on one of the backends."""
    fixtures = _all_warn_mode_fixtures()
    assert len(fixtures) >= 22, (
        f"Expected ≥22 warn-mode fixtures, found {len(fixtures)}. "
        f"Add coverage in tests/protocol_corpus/fixtures/warn_mode/."
    )
    by_name = {f.name: f for f in fixtures}
    missing = _RELEASE_BASELINES - by_name.keys()
    assert not missing, f"Release baseline rows missing: {sorted(missing)}"
    narrowed = sorted(
        name for name in _RELEASE_BASELINES
        if set(by_name[name].backends) != set(ALL_BACKENDS)
    )
    assert not narrowed, f"Release baseline rows not run on both backends: {narrowed}"


def test_normalizer_self_test_detects_real_divergence() -> None:
    """Harness self-test (per Unit 7a verification): deliberately-different
    bodies must NOT compare equal after normalization. Catches the failure
    mode where over-aggressive normalization rules false-pass real drift."""
    body_a: dict[str, Any] = {
        "schema_version": 1,
        "coordinator_uptime_seconds": 123.4,
        "instance_id": "11111111-1111-4111-8111-111111111111",
        "tracked_count": 5,
    }
    body_b: dict[str, Any] = {
        "schema_version": 1,
        "coordinator_uptime_seconds": 999.9,         # would be ignored
        "instance_id": "22222222-2222-4222-8222-222222222222",  # would be ignored
        "tracked_count": 7,                            # real divergence
    }
    norm_a = normalize_response(body_a)
    norm_b = normalize_response(body_b)
    assert norm_a != norm_b, (
        "Normalizer should NOT mask real differences in non-time/UUID fields. "
        f"norm_a={norm_a!r}, norm_b={norm_b!r}"
    )


def test_normalizer_handles_nested_uuid_and_timestamp() -> None:
    """Self-test: nested dicts and lists get walked, string-position UUID/ISO
    timestamps scrubbed."""
    body: dict[str, Any] = {
        "sessions": [
            {
                "agent_id": "11111111-1111-4111-8111-111111111111",
                "started_at": 1700000000.0,
                "last_request_at": "2026-05-23T12:34:56Z",
                "message": "agent 22222222-2222-4222-8222-222222222222 acquired at 2026-05-23T10:00:00Z",
            }
        ],
        "coordinator_uptime_seconds": 42,
    }
    out = normalize_response(body)
    session = out["sessions"][0]
    assert session["agent_id"] == "<UUID>"
    assert session["started_at"] == "<TS>"
    assert session["last_request_at"] == "<TS>"
    # In-string substitution: both the embedded UUID and timestamp scrubbed.
    assert "<UUID>" in session["message"]
    assert "<TS>" in session["message"]
    assert out["coordinator_uptime_seconds"] == "<UPTIME>"


def test_optional_keys_dropped_only_at_top_level() -> None:
    """AC3 harness self-test: ``optional_keys`` drops a key at the TOP-LEVEL
    response body only. A nested occurrence of the SAME key name is kept and
    compared, so a nested divergence still fails the diff (the bug was the drop
    firing at every dict depth, masking nested drift)."""
    optional = frozenset({"version"})

    # Top-level: dropped on both sides → the version difference is tolerated.
    top_a = {"status": "fresh", "version": 1}
    top_b = {"status": "fresh", "version": 99}
    assert normalize_response(top_a, optional_keys=optional) == normalize_response(
        top_b, optional_keys=optional
    )
    # And the top-level key is actually gone after normalization.
    assert "version" not in normalize_response(top_a, optional_keys=optional)

    # Nested: a same-named key one level deep is NOT dropped, so a real nested
    # divergence in that key still makes the two bodies compare UNEQUAL.
    nested_a = {"status": "fresh", "payload": {"version": 1}}
    nested_b = {"status": "fresh", "payload": {"version": 2}}
    norm_nested_a = normalize_response(nested_a, optional_keys=optional)
    norm_nested_b = normalize_response(nested_b, optional_keys=optional)
    assert "version" in norm_nested_a["payload"], (
        "nested optional-named key must be preserved, not dropped"
    )
    assert norm_nested_a != norm_nested_b, (
        "a nested divergence in an optional-named key must still fail the diff"
    )

    # A same-named key nested inside a LIST element is also preserved/compared.
    list_a = {"items": [{"version": 1}]}
    list_b = {"items": [{"version": 7}]}
    assert normalize_response(list_a, optional_keys=optional) != normalize_response(
        list_b, optional_keys=optional
    )
