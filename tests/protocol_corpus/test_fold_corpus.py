# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Cross-implementation corpus: one identity per file (#506).

Every spelling the workspace disk treats as one file reaches one record on
both coordinators, and a disk that keeps two spellings apart keeps two
records. Fixtures live in ``fixtures/fold/``. Where the answer depends on the
disk, a fixture carries ``expected_by_fold_profile`` and the row asserts the
entry for the profile the harness's probe finds under this run's own tmp
workspace (``expected_for_workspace``): ``caseless`` on a Mac's default APFS,
``nfd`` on a case-sensitive APFS volume, ``none`` on Linux CI. A profile the
fixture does not name fails the row.

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``. Skipped
in default runs (see ``pyproject.toml`` ``addopts``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.protocol_corpus.harness import (
    ALL_BACKENDS,
    BACKEND_NODE,
    FOLD_PROFILES,
    Fixture,
    expected_for_workspace,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

_FIXTURE_DIR = "fold"

#: The table version both coordinators advertise. FROZEN, never read from the
#: library: a table bump is a wire change and edits these rows on purpose.
_FOLD_TABLE_UNICODE = "17.0.0"

#: The rows #506 rests on, by name: an alias pre-edit (opted in or not), a
#: strict alias, a key pair and a ``./`` alias. Named rather than counted, so
#: one going missing or losing a backend cannot hide behind new rows.
_ALIAS_ROWS = frozenset({
    "fold-pre-edit-of-another-case-meets-the-holder-where-the-disk-folds-case",
    "fold-opted-in-pre-edit-of-another-case-is-refused-where-the-disk-folds-case",
    "fold-stale-edit-of-another-case-of-a-strict-file-is-denied-where-the-disk-folds-case",
    "fold-two-keys-that-differ-by-case-stay-two-records-on-every-disk",
    "fold-pre-edit-of-a-dot-slash-spelling-meets-the-holder-on-every-disk",
})


def _all_fold_fixtures() -> list[Fixture]:
    """Loaded once at collection so parametrize ids stay stable."""
    return load_fixtures(_FIXTURE_DIR)


def _row_id(row: tuple[Fixture, str]) -> str:
    fixture, backend = row
    return f"{fixture.name}[{backend}]"


_ROWS = [(fixture, backend) for fixture in _all_fold_fixtures() for backend in fixture.backends]
_NODE_DIST_PATH = resolve_node_dist_path()


@pytest.mark.parametrize("row", _ROWS, ids=[_row_id(r) for r in _ROWS] if _ROWS else None)
def test_fold_fixture_response_matches_expected(
    row: tuple[Fixture, str],
    tmp_path: Path,
) -> None:
    fixture, backend = row
    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.xfail(
            "Node coordinator dist not resolvable. Build the plugin checkout "
            "(npm ci && npm run build) or set AGENT_COHERENCE_PLUGIN_DIST_PATH."
        )
    # Before the coordinator starts: the probe runs in the workspace it will open.
    expected = expected_for_workspace(fixture, tmp_path)

    actual = run_scenario(
        fixture=fixture,
        backend_id=backend,
        workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )

    _assert_row_answers(row, expected, actual)


def _assert_row_answers(
    row: tuple[Fixture, str],
    expected: dict[str, Any],
    actual: tuple[int, Any],
) -> None:
    """The row's answer against the entry picked for its disk, normalised as
    every corpus row normalises its expectation."""
    fixture, backend = row
    actual_status, actual_body = actual
    expected_body = normalize_response(
        expected["body"],
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
        preserve_identity=fixture.preserve_identity,
    )
    assert actual_status == expected["status"], (
        f"{fixture.name}[{backend}]: status mismatch -- "
        f"expected {expected['status']}, got {actual_status}\nbody={actual_body!r}"
    )
    assert actual_body == expected_body, (
        f"{fixture.name}[{backend}]: body mismatch\n"
        f"expected={expected_body!r}\nactual=  {actual_body!r}"
    )


def test_collection_loaded_the_alias_rows_on_both_backends() -> None:
    """The fixtures are present, and each alias row runs on both coordinators:
    parity between them is the claim, so a row narrowed to one is a lost row."""
    by_name = {f.name: f for f in _all_fold_fixtures()}
    missing = _ALIAS_ROWS - by_name.keys()
    assert not missing, f"alias rows missing from fixtures/{_FIXTURE_DIR}/: {sorted(missing)}"
    narrowed = sorted(n for n in _ALIAS_ROWS if set(by_name[n].backends) != set(ALL_BACKENDS))
    assert not narrowed, f"alias rows not run on both backends: {narrowed}"
    assert any(f.expected_by_fold_profile for f in by_name.values()), (
        "no row asserts a per-profile answer, so no row depends on the disk"
    )


def test_each_fold_field_row_asserts_the_profile_it_is_filed_under() -> None:
    """A row that asserts the fold fields names, under each profile, that
    profile. A body pasted under the wrong key would pass only on the disk it
    was pasted from, and fail the others with a message blaming the
    coordinator."""
    field_rows = [
        f for f in _all_fold_fixtures()
        if f.expected_by_fold_profile
        and any("fold_profile" in e["body"] for e in f.expected_by_fold_profile.values())
    ]
    assert len(field_rows) >= 3, [f.name for f in field_rows]
    for fixture in field_rows:
        for profile, entry in fixture.expected_by_fold_profile.items():
            assert profile in FOLD_PROFILES
            assert entry["body"] == {
                "fold_profile": profile,
                "fold_table_unicode": _FOLD_TABLE_UNICODE,
            }, f"{fixture.name}: the {profile!r} entry asserts {entry['body']!r}"
