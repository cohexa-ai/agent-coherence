# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Session, workspace and admin corpus: the Python-only routes (#193).

The snapshot-session routes (``/session/*``), the workspace-checkpoint and
restore routes (``/workspace/*``) and ``/admin/prepare-for-migration`` are
served by the Python coordinator only, so their rows declare
``backends: ["python"]`` — the same single-backend scoping the strict-mode and
status rows use. The one Node row pins that the sibling answers 404 on the
admin route, so a Node implementation growing one is a deliberate parity
change. The routes BOTH coordinators serve that #193 also listed
(``/hooks/post-edit-cas``, ``/policy/untrack``) are parity rows and live in
``warn_mode/`` with the other both-backend rows.

Every row asserts a full response body. These shapes are not inferable from the
status code — a refused ``/session/read`` and a HELD restore registration are
both HTTP 200, the second with ``ok: true`` — which is what a client in another
language needs a corpus row for rather than a pytest assertion it cannot run.

The restore-registration rows freeze the #191 contract: a write-set must name
members of the checkpoint at their captured fingerprints, a checkpoint naming a
receiver admits only that controller, the first registering controller claims
the checkpoint, and that controller's retry is not refused.

Marked ``protocol_corpus`` — opt-in via ``pytest -m protocol_corpus``. The Node
row FAILS rather than xfails when the sibling cannot run: an asymmetry row that
never ran reports green while checking nothing."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    BACKEND_PYTHON,
    FIXTURES_ROOT,
    MINTED_SENTINEL,
    UUID_SENTINEL,
    Fixture,
    FixtureContractError,
    build_fixture,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

# Frozen counts, per directory: a misspelled or emptied directory loads as
# nothing and a parametrize over nothing reports green.
_EXPECTED_COUNTS = {"session": 13, "workspace": 20, "admin": 3}

# Frozen duplicates of the code under test (tests/CLAUDE.md house rule): the
# derived controller ids of the fixtures' session ids, uuid5(NAMESPACE_URL,
# "ccs-agent:claude-session-<session id>").
_AGENT_A = "c7f92943-10c4-51ba-ada9-df7448871e72"
_AGENT_B = "d7f57e87-6689-5239-ab8e-6446ae5be1f0"

_NODE_DIST_PATH = resolve_node_dist_path()


def _fixtures(directory: str) -> list[Fixture]:
    return load_fixtures(directory)


_ROWS = [
    (fixture, backend)
    for directory in _EXPECTED_COUNTS
    for fixture in _fixtures(directory)
    for backend in fixture.backends
]


def _by_name(name: str) -> Fixture:
    for directory in _EXPECTED_COUNTS:
        for fixture in _fixtures(directory):
            if fixture.name == name:
                return fixture
    raise AssertionError(f"no fixture named {name!r}")


def _expected(fixture: Fixture, body: object | None = None) -> object:
    return normalize_response(
        fixture.expected["body"] if body is None else body,
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
        preserve_identity=fixture.preserve_identity,
    )


@pytest.mark.parametrize("row", _ROWS, ids=[f"{f.name}[{b}]" for f, b in _ROWS])
def test_session_workspace_fixture_response_matches_expected(
    row: tuple[Fixture, str], tmp_path: Path
) -> None:
    """The full body is the assertion, normalized on both sides with the
    fixture's own identity opt-in (a module that dropped it would compare a
    preserved actual against a scrubbed expected and fail red)."""
    fixture, backend = row
    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.fail(
            f"{fixture.name}: the Node row got no answer — the plugin dist could "
            f"not be resolved. Build the plugin (npm ci && npm run build) or set "
            f"AGENT_COHERENCE_PLUGIN_DIST_PATH."
        )
    status, body = run_scenario(
        fixture=fixture, backend_id=backend, workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )
    expected = _expected(fixture)
    assert status == fixture.expected["status"], f"{fixture.name}[{backend}]: {body!r}"
    assert body == expected, (
        f"{fixture.name}[{backend}]\nexpected={expected!r}\nactual=  {body!r}"
    )


def test_fixture_directories_are_actually_loaded() -> None:
    for directory, count in _EXPECTED_COUNTS.items():
        fixtures = _fixtures(directory)
        assert len(fixtures) == count, (
            f"{directory}/: expected {count} fixtures, found "
            f"{[f.name for f in fixtures]}"
        )


def test_every_registered_route_is_the_request_under_test_somewhere() -> None:
    """#193's finding was a list of routes the corpus never asserted. This
    keeps it from regrowing: every ``(method, path)`` the Python dispatcher
    registers must be the MAIN request of at least one fixture in some
    directory — a preflight does not count, its body is never asserted."""
    from ccs.adapters.claude_code.coordinator_server import _ROUTES

    asserted: set[tuple[str, str]] = set()
    for path in FIXTURES_ROOT.glob("*/*.json"):
        request = json.loads(path.read_text())["request"]
        asserted.add((request.get("method", "POST").upper(), request["path"].split("?")[0]))
    missing = sorted(set(_ROUTES) - asserted)
    assert not missing, (
        f"routes no fixture makes the request under test: {missing}. Add a "
        f"row asserting each one's response body."
    )


def test_python_only_rows_are_scoped_and_the_node_row_is_the_admin_404() -> None:
    """The Node coordinator serves none of these routes. A row left on the
    default (both backends) would run a Python-only route against Node and
    fail for that reason, not for the one its description gives."""
    for directory in _EXPECTED_COUNTS:
        for fixture in _fixtures(directory):
            if fixture.backends == (BACKEND_NODE,):
                assert fixture.request["path"] == "/admin/prepare-for-migration"
                assert fixture.expected["status"] == 404
            else:
                assert fixture.backends == (BACKEND_PYTHON,), fixture.name


def test_registration_names_who_registered_and_not_the_owner(tmp_path: Path) -> None:
    """DISTINGUISH for #191 provenance: owner A, receiver B, registered_by B.

    With the opt-in the RIGHT attribution matches and the one a pre-#191
    reading would write (registered by the owner) does not; without it, the two
    are the same bytes — which is why the fixture declares it."""
    fixture = _by_name("workspace-checkpoint-list-names-who-registered")
    _, actual = run_scenario(fixture=fixture, backend_id="python", workspace=tmp_path)
    row = actual["checkpoints"][0]
    assert (row["owner"], row["receiver"], row["registered_by"]) == (
        _AGENT_A, _AGENT_B, _AGENT_B,
    ), f"the identities did not survive verbatim: {row!r}"
    assert actual == _expected(fixture)

    wrong = copy.deepcopy(fixture.expected["body"])
    wrong["checkpoints"][0]["registered_by"] = _AGENT_A
    assert actual != _expected(fixture, wrong), (
        "crediting the registration to the owner still matched — the opt-in is "
        "not preserving the identity"
    )
    default = replace(fixture, preserve_identity=frozenset())
    assert normalize_response(
        fixture.expected["body"], ignore_keys=default.ignore_keys
    ) == normalize_response(wrong, ignore_keys=default.ignore_keys), (
        "without the opt-in the two attributions must collapse to the same "
        "bytes; if they already differ, this fixture is not exercising it"
    )


def test_minted_fields_are_scrubbed_by_name_and_stay_asserted_present() -> None:
    """#193 ask 2. ``session_token`` and ``coordinator_epoch`` normalize to a
    sentinel of their own, so no fixture needs an ``ignore_keys`` opt-out for
    them — and, unlike ``ignore_keys``, the scrub keeps the KEY in the diff: a
    heartbeat that grew an epoch differs from one that did not."""
    begin = {"ok": True, "session_token": "aKPsDgZpM84jtheqXAAtfn_x5APKHGiPHdIspQavJTI",
             "coordinator_epoch": "ec5390065a2a4af8887cb2badfa790c5"}
    out = normalize_response(begin)
    assert out["session_token"] == MINTED_SENTINEL
    assert out["coordinator_epoch"] == MINTED_SENTINEL
    assert MINTED_SENTINEL != UUID_SENTINEL

    heartbeat = {"ok": True, "refreshed": True}
    assert normalize_response(heartbeat) != normalize_response(
        {**heartbeat, "coordinator_epoch": "ec5390065a2a4af8887cb2badfa790c5"}
    )

    # A preserve_identity declaration cannot un-scrub a minted value.
    assert normalize_response(
        begin, preserve_identity=frozenset({"session_token"})
    )["session_token"] == MINTED_SENTINEL

    # And no fixture in the corpus still opts out of either field by hand.
    for path in FIXTURES_ROOT.glob("*/*.json"):
        ignored = set(json.loads(path.read_text()).get("ignore_keys", []))
        assert not ignored & {"session_token", "coordinator_epoch"}, path.name


def _raw(fixture: Fixture, **over: object) -> dict:
    base = {
        "name": fixture.name,
        "setup": copy.deepcopy(fixture.setup),
        "request": copy.deepcopy(fixture.request),
        "expected": copy.deepcopy(fixture.expected),
        "ignore_keys": sorted(fixture.ignore_keys),
        "preserve_identity": sorted(fixture.preserve_identity),
        "backends": list(fixture.backends),
    }
    base.update(over)
    return base


def test_preserve_identity_checks_every_occurrence_of_a_repeated_key() -> None:
    """The second ``_validate_preserve_identity`` point #193's comment raised:
    the check read only the LAST occurrence of a repeated key. A listing with
    two checkpoints whose first row asserts ``<UUID>`` for a preserved key and
    whose last asserts an identity must be refused — the first row's
    declaration would preserve nothing."""
    fixture = _by_name("workspace-checkpoint-list-names-who-registered")
    assert build_fixture(_raw(fixture), fixture.path).preserve_identity  # control

    expected = copy.deepcopy(fixture.expected)
    first = copy.deepcopy(expected["body"]["checkpoints"][0])
    first["owner"] = UUID_SENTINEL
    expected["body"]["checkpoints"].insert(0, first)
    with pytest.raises(FixtureContractError, match="Preserving a sentinel"):
        build_fixture(_raw(fixture, expected=expected), fixture.path)

    # A null occurrence beside a pinned one is admitted: the opt-in does not
    # apply to it and the default compares null literally.
    expected = copy.deepcopy(fixture.expected)
    second = copy.deepcopy(expected["body"]["checkpoints"][0])
    second["receiver"] = None
    expected["body"]["checkpoints"].append(second)
    build_fixture(_raw(fixture, expected=expected), fixture.path)

    # Every occurrence null: nothing to preserve.
    expected = copy.deepcopy(fixture.expected)
    expected["body"]["checkpoints"][0]["receiver"] = None
    with pytest.raises(FixtureContractError, match="string values only"):
        build_fixture(_raw(fixture, expected=expected), fixture.path)


def test_preserve_identity_over_a_minted_key_is_refused() -> None:
    fixture = _by_name("session-begin-mints-a-token-and-a-path-keyed-cut")
    with pytest.raises(FixtureContractError, match="minted field"):
        build_fixture(
            _raw(fixture, preserve_identity=["coordinator_epoch"]), fixture.path
        )
