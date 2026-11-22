# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Basis check and commit corpus: the wire of ``/basis/check`` and
``/basis/commit``, pinned.

The two routes answer, from one caller-recorded basis, whether every input a
caller read is still the version (and registration) it read, and, on the commit
route, compare-and-swap the caller's result only if so. They are served by the
Python coordinator only. Python rows record what that coordinator answers from
fixtures under ``fixtures/basis_gate/``; three node-only rows record that the
sibling Node coordinator answers each route 404 and never reports a
``registration`` on a read.

What the rows cover, and how this module keeps them covering it:

- **Every hold reason a basis check can give** has a named row
  (:data:`_REASON_ROWS`) or a named exclusion with where it is pinned
  (:data:`_EXCLUDED_REASONS`), keyed on the code's own
  :data:`~ccs.core.exceptions.BASIS_HOLD_REASONS`.
- **Both routes** are the main request of at least one Python row.
- **The registration token** is used as captured from a read, never typed in,
  by the rows that need a matching one.

Not pinned here, each a written decision rather than a gap: the degraded
``version_unconfirmed`` hold and the outcome-unknown commit answer (the harness
has no seam to stall a registry past the handler deadline; the HTTP route tests
stall one), and that a hold registers no row (the harness only sees a reply; the
HTTP route tests read the registry afterwards).

A Node row that cannot run FAILS rather than xfails: an asymmetry row that
never ran reports green while checking nothing.

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
from uuid import UUID

import pytest

from ccs.core.basis import BASIS_TARGET_REFUSAL_REASONS
from ccs.core.exceptions import BASIS_HOLD_REASONS
from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    BACKEND_PYTHON,
    Fixture,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

_FIXTURE_DIR = "basis_gate"
_EXPECTED_FIXTURE_COUNT = 18

# Frozen duplicate of the code's vocabulary: a set derived from the code would
# move its own goalposts. Its cardinality is pinned separately so a same-size
# add and remove cannot slip past set equality.
_EXPECTED_HOLD_REASONS: frozenset[str] = frozenset({
    "version_moved",
    "input_vanished",
    "basis_member_unmanaged",
    "basis_member_replaced",
    "version_unconfirmed",
})
_EXPECTED_HOLD_REASON_COUNT = 5

# The row that pins each reason the harness can reach. A reason may appear in
# other rows too; this is the one whose removal must turn the coverage test red.
_REASON_ROWS: dict[str, str] = {
    "version_moved": "basis-check-hold-version-moved",
    "input_vanished": "basis-check-hold-input-vanished-tracked-path-never-read",
    "basis_member_unmanaged": "basis-check-hold-basis-member-unmanaged-untracked-path",
    "basis_member_replaced": "basis-check-hold-basis-member-replaced-different-registration",
}

# Reasons no row can drive, each with where it IS pinned. The coverage test
# requires this and _REASON_ROWS to partition BASIS_HOLD_REASONS exactly.
_EXCLUDED_REASONS: dict[str, str] = {
    # Only a stalled registry (or a lock not won within the handler deadline)
    # answers it, and the harness cannot hold the registry lock. Pinned by the
    # stalled-registry tests in tests/test_basis_gate_http.py.
    "version_unconfirmed": "degraded: the registry lock is not won inside the deadline",
}

_CHECK_ROUTE = "/basis/check"
_COMMIT_ROUTE = "/basis/commit"
_NODE_ROUTES = frozenset({_CHECK_ROUTE, _COMMIT_ROUTE})

# Rows by the behavior they pin that the reason map does not cover.
_ROWS_BY_ROLE: dict[str, str] = {
    "proceed with a captured registration": "basis-check-proceed-with-the-registration-a-read-reported",
    "record written only by compare-and-swap": "basis-check-proceed-record-written-only-by-compare-and-swap",
    "unknown top-level key": "basis-check-refuses-an-unknown-top-level-key-400",
    "unknown member key": "basis-check-refuses-an-unknown-member-key-400",
    "member at version zero": "basis-check-refuses-a-member-at-version-zero-400",
    "commit win": "basis-commit-win-target-in-basis",
    "commit hold on a moved basis": "basis-commit-hold-basis-moved",
    "commit hold with a target block": "basis-commit-hold-target-refused-other-holder",
    "byte-identical re-save": "basis-check-hold-byte-identical-re-save",
    "pre-read opted into registration": "basis-pre-read-opted-into-registration-reports-the-token",
}

_NODE_ROWS: dict[str, str] = {
    _CHECK_ROUTE: "basis-node-check-route-does-not-exist-404",
    _COMMIT_ROUTE: "basis-node-commit-route-does-not-exist-404",
    "/hooks/pre-read": "basis-node-pre-read-ignores-want-registration",
}

_NODE_DIST_PATH = resolve_node_dist_path()
_NODE_DIST_UNRESOLVED = (
    "The Node asymmetry rows need a REAL answer from the sibling coordinator, and this row got none: "
    "the plugin dist could not be resolved. Failing rather than xfailing is "
    "deliberate -- the claim under test is that the Node backend answers 404 for "
    "the basis routes and never reports a registration, and an xfail records that "
    "nobody asked. Build the plugin checkout (npm ci && npm run build) or set "
    "AGENT_COHERENCE_PLUGIN_DIST_PATH to the absolute dist/coordinator.js path."
)


def _fixtures() -> list[Fixture]:
    return load_fixtures(_FIXTURE_DIR)


def _by_name() -> dict[str, Fixture]:
    return {f.name: f for f in _fixtures()}


def _rows() -> list[tuple[Fixture, str]]:
    return [(f, b) for f in _fixtures() for b in f.backends]


_ROWS = _rows()


def _values(tree: Any, key: str) -> Iterator[Any]:
    """Every value stored under ``key`` anywhere in a JSON tree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == key:
                yield v
            yield from _values(v, key)
    elif isinstance(tree, list):
        for item in tree:
            yield from _values(item, key)


def _body(fixture: Fixture) -> Any:
    return fixture.expected["body"]


@pytest.mark.parametrize("row", _ROWS, ids=[f"{f.name}[{b}]" for f, b in _ROWS])
def test_basis_gate_fixture_response_matches_expected(
    row: tuple[Fixture, str], tmp_path: Path
) -> None:
    """The FULL response body is the assertion, after normalization on both
    sides. A renamed reason, a dropped or added key, a changed version or a
    changed recovery class all fail here; on Node, a route that stopped
    answering 404 or a read that began to report a registration fails here."""
    fixture, backend = row
    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.fail(f"{fixture.name}: {_NODE_DIST_UNRESOLVED}")
    status, body = run_scenario(
        fixture=fixture, backend_id=backend, workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )
    expected = normalize_response(
        _body(fixture),
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
        preserve_identity=fixture.preserve_identity,
    )
    assert status == fixture.expected["status"], (
        f"{fixture.name}[{backend}]: status mismatch -- expected "
        f"{fixture.expected['status']}, got {status}\nbody={body!r}"
    )
    assert body == expected, (
        f"{fixture.name}[{backend}]: body mismatch\nexpected={expected!r}\nactual=  {body!r}"
    )


def test_fixture_directory_is_actually_loaded() -> None:
    """A misspelled, moved or emptied fixture directory loads as ``[]`` and a
    parametrize over nothing reports green; the frozen count makes it red, and
    adding or removing a row is a deliberate edit of this literal."""
    fixtures = _fixtures()
    assert len(fixtures) == _EXPECTED_FIXTURE_COUNT, (
        f"Expected exactly {_EXPECTED_FIXTURE_COUNT} fixtures in "
        f"tests/protocol_corpus/fixtures/{_FIXTURE_DIR}/, found {len(fixtures)}. Adding or "
        f"removing one is a deliberate change to the pinned basis wire -- update this "
        f"literal in the same diff."
    )
    assert len({f.name for f in fixtures}) == len(fixtures), "fixture names must be unique"
    assert len(_ROWS) == len(fixtures), "every basis fixture runs on exactly one backend"


def test_every_hold_reason_has_its_row_or_a_named_exclusion() -> None:
    """Every basis hold reason is pinned by a named row or excluded by name
    with where it is pinned instead, never silently.

    Keyed on the code's own set: the frozen duplicate must equal it, the rows
    and exclusions must partition it exactly, and each row's expected body must
    carry its reason on a moved member, where the wire puts it and with the
    ``rederive`` recovery class. Deleting a reason's row is red here."""
    assert _EXPECTED_HOLD_REASONS == BASIS_HOLD_REASONS, (
        f"the hold vocabulary drifted: only in code "
        f"{sorted(BASIS_HOLD_REASONS - _EXPECTED_HOLD_REASONS)}, only here "
        f"{sorted(_EXPECTED_HOLD_REASONS - BASIS_HOLD_REASONS)}"
    )
    assert len(BASIS_HOLD_REASONS) == _EXPECTED_HOLD_REASON_COUNT

    rows, excluded = set(_REASON_ROWS), set(_EXCLUDED_REASONS)
    assert not rows & excluded, f"a reason is both pinned and excluded: {sorted(rows & excluded)}"
    assert rows | excluded == BASIS_HOLD_REASONS, (
        f"reasons with neither a row nor a named exclusion: "
        f"{sorted(BASIS_HOLD_REASONS - rows - excluded)}; unknown names: "
        f"{sorted((rows | excluded) - BASIS_HOLD_REASONS)}"
    )
    assert excluded == {"version_unconfirmed"}, (
        "the only exclusion is the degraded hold; any other is a new decision "
        "and needs its reason written into _EXCLUDED_REASONS"
    )

    by_name = _by_name()
    for reason, name in _REASON_ROWS.items():
        assert name in by_name, f"no row pins {reason!r}: {name!r} is missing"
        fixture = by_name[name]
        assert fixture.request["path"] == _CHECK_ROUTE, name
        body = _body(fixture)
        assert [m["reason"] for m in body["moved"]] == [reason], (
            f"{name} is the {reason!r} row but its expected body does not carry it "
            f"on a moved member: {body!r}"
        )
        assert body["answer"] == "hold" and body["recovery"] == "rederive", name


def test_no_row_expects_a_body_the_harness_cannot_drive() -> None:
    """The documented-unreachable twin: honest limits, stated as assertions.

    The harness cannot hold the registry lock past the handler deadline, so the
    degraded hold (``version_unconfirmed``, ``degraded``, ``held_by``) and the
    outcome-unknown commit answer have no row; the HTTP route tests pin them.
    If a row ever expects one, this goes red: rewrite it to assert that body's
    shape instead of its absence."""
    for fixture in _fixtures():
        body = _body(fixture)
        reasons = set(_values(body, "reason"))
        assert not reasons & set(_EXCLUDED_REASONS), fixture.name
        assert "commit_unconfirmed" not in reasons, fixture.name
        assert not list(_values(body, "degraded")), fixture.name
        assert not list(_values(body, "held_by")), fixture.name


def test_each_route_is_the_main_request_of_a_python_row() -> None:
    """A route no Python row sends to is a route whose answers nobody pins."""
    python_paths = {
        f.request["path"] for f in _fixtures() if f.backends == (BACKEND_PYTHON,)
    }
    assert {_CHECK_ROUTE, _COMMIT_ROUTE} <= python_paths


def test_the_named_rows_exist_with_the_shape_each_pins() -> None:
    """Each behavior the reason map does not cover has its named row, and the
    row drives the request the behavior needs."""
    by_name = _by_name()
    for role, name in _ROWS_BY_ROLE.items():
        assert name in by_name, f"no row pins {role!r}: {name!r} is missing"
        assert by_name[name].backends == (BACKEND_PYTHON,), name

    for role in ("unknown top-level key", "unknown member key", "member at version zero"):
        fixture = by_name[_ROWS_BY_ROLE[role]]
        assert fixture.expected["status"] == 400, role
        assert set(_body(fixture)) == {"error"}, role

    win = _body(by_name[_ROWS_BY_ROLE["commit win"]])
    assert win["answer"] == "proceed" and set(win["committed"]) == {"path", "version"}

    moved = by_name[_ROWS_BY_ROLE["commit hold on a moved basis"]]
    assert moved.request["path"] == _COMMIT_ROUTE
    assert _body(moved)["moved"][0]["reason"] == "version_moved"

    refused = by_name[_ROWS_BY_ROLE["commit hold with a target block"]]
    assert refused.request["path"] == _COMMIT_ROUTE
    target = _body(refused)["target"]
    assert target["reason"] == "other_holder" and target["reason"] in BASIS_TARGET_REFUSAL_REASONS
    assert _body(refused)["moved"] == [] and "recovery" not in _body(refused)

    resave = by_name[_ROWS_BY_ROLE["byte-identical re-save"]]
    saves = [r for r in resave.setup["preflight_requests"] if r["path"] == "/hooks/post-edit-cas"]
    read_hash = resave.setup["preflight_requests"][0]["body"]["content_hash"]
    assert [s["body"]["content_hash"] for s in saves] == [read_hash], (
        "the re-save must commit the very hash the first read recorded, or it "
        "pins an ordinary move instead of a byte-identical one"
    )

    opted = by_name[_ROWS_BY_ROLE["pre-read opted into registration"]]
    assert opted.request["path"] == "/hooks/pre-read"
    assert opted.request["body"]["want_registration"] is True
    assert "registration" in _body(opted)


def test_the_matching_rows_send_the_registration_a_read_reported() -> None:
    """A registration typed into a fixture would pass against any coordinator
    that ignored it. The rows that need a matching one capture it from a read
    that asked for it and send that value; the replaced row sends a different,
    canonical dashed UUID, which only the registration can tell from the row's
    own."""
    by_name = _by_name()
    for role in ("proceed with a captured registration", "record written only by compare-and-swap"):
        fixture = by_name[_ROWS_BY_ROLE[role]]
        captures = [
            r for r in fixture.setup["preflight_requests"]
            if "registration" in (r.get("capture") or {})
        ]
        assert len(captures) == 1, role
        assert captures[0]["path"] == "/hooks/pre-read"
        assert captures[0]["body"]["want_registration"] is True, role
        (member,) = fixture.request["body"]["basis"]
        assert member["registration"] == "${registration}", role

    replaced = by_name[_REASON_ROWS["basis_member_replaced"]]
    (member,) = replaced.request["body"]["basis"]
    sent = member["registration"]
    assert str(UUID(sent)) == sent, "a replaced member sends a canonical dashed UUID"
    assert "${" not in sent
    assert _body(replaced)["moved"][0]["expected"] == _body(replaced)["moved"][0]["current"], (
        "the replaced row sits at the row's real version; a version difference "
        "would answer version_moved and say nothing about the registration"
    )


def test_the_node_rows_cannot_be_satisfied_by_a_skip() -> None:
    """The Node asymmetry, asserted directly: the plugin dist resolves, and
    exactly the named node-only rows exist, each sent with the harness's valid
    bearer (Node rejects auth before routing, so a bad one answers 401 and says
    nothing about which routes exist). Every other row is Python-only."""
    assert _NODE_DIST_PATH is not None, _NODE_DIST_UNRESOLVED
    assert _NODE_DIST_PATH.exists(), f"resolved dist path does not exist: {_NODE_DIST_PATH}"

    by_name = _by_name()
    node_rows = [f for f in _fixtures() if BACKEND_NODE in f.backends]
    assert {f.name for f in node_rows} == set(_NODE_ROWS.values())
    for route in _NODE_ROUTES:
        fixture = by_name[_NODE_ROWS[route]]
        assert fixture.request["path"] == route
        assert fixture.expected == {"status": 404, "body": {"error": "not found"}}, fixture.name
    for fixture in node_rows:
        assert fixture.backends == (BACKEND_NODE,), fixture.name
        assert "headers" not in fixture.request, fixture.name
    python_rows = [f for f in _fixtures() if f not in node_rows]
    assert all(f.backends == (BACKEND_PYTHON,) for f in python_rows)

    read = by_name[_NODE_ROWS["/hooks/pre-read"]]
    assert read.request["body"]["want_registration"] is True
    assert read.expected["status"] == 200 and "registration" not in _body(read)


def test_no_row_ignores_or_drops_a_key() -> None:
    """Each ``ignore_keys`` or ``optional_keys`` entry is a hole in the catch
    surface. No basis row needs one: the answers carry no timestamp, and the
    one minted value (the registration) is scrubbed by the harness's UUID
    scrub."""
    for fixture in _fixtures():
        assert not fixture.ignore_keys, fixture.name
        assert not fixture.optional_keys, fixture.name
