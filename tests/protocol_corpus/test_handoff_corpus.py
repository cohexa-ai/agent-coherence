# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Targeted grant handoff corpus: the handoff wire, pinned (#185).

The four handoff routes (``/handoff/transfer``, ``/handoff/accept``,
``/handoff/decline``, ``/handoff/withdraw``) are served by the Python
coordinator only, and the handoff changes four existing bodies (pre-read,
pre-edit, post-edit, post-edit-cas) and ``/status`` while a path has a transfer
record. This module pins those answers byte for byte from fixtures under
``fixtures/handoff/``, in the asymmetry style of the effect-fence corpus:
Python rows record what the Python coordinator answers, and four node-only rows
record that the sibling Node coordinator answers each route 404.

What the rows cover, and how this module keeps them covering it:

- **Every transfer status** has a named row (:data:`_STATUS_FIXTURES`), keyed
  on the code's own :data:`~ccs.core.types.TRANSFER_STATUSES`. ``superseded``
  is never stored, so its row is the re-send of a superseded tuple.
- **Every wire reason** the handoff adds has a named row
  (:data:`_REASON_FIXTURES`) or a named exclusion with its reason written out
  (:data:`_EXCLUDED_REASONS`), keyed on
  :data:`~ccs.core.exceptions.HANDOFF_REASONS`. Removing a named row fails
  :func:`test_every_transfer_status_has_its_row` or
  :func:`test_every_wire_reason_has_its_row_or_a_named_exclusion`.
- **Each role's view** (giver while live and after the record ended,
  successor, bystander), the giver's refusal on pre-edit (with its deny
  envelope) and on compare-and-swap, the post-edit handoff arm, the reachable
  per-grant release answers, and the ``/status`` projection on both tiers.

Every expected body is hand-written from ``docs/guide.md``'s "Targeted grant
handoff" section and the byte-stable templates, and was verified against the
Python coordinator. The handoff key names the parties by session-level agent
id, which the harness's portability scrub would collapse to ``<UUID>``; every
row naming one declares ``preserve_identity`` for it, so a body crediting the
giver as successor fails rather than passing (asserted by
:func:`test_every_identity_a_row_names_is_compared_verbatim`).

Three things are NOT pinned here, each a written decision rather than a gap:
the four degraded (unconfirmed) bodies and a session stop that leaves a grant
held (the harness has no seam to abandon a work body or fail a release), and
``handoff_version_unconfirmed`` (no HTTP request can hold a claim on a
version-0 path). See :func:`test_no_handoff_fixture_expects_an_unreachable_body`
and :data:`_EXCLUDED_REASONS`.

A Node row that cannot run FAILS rather than xfails: an asymmetry row that
never ran reports green while checking nothing.

Marked ``protocol_corpus`` -- opt-in via ``pytest -m protocol_corpus``."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Iterator
from uuid import NAMESPACE_URL, uuid5

import pytest

from ccs.core.exceptions import (
    GIVER_FENCED_REASON,
    HANDOFF_REASONS,
    HANDOFF_TRANSFER_REFUSAL_REASONS,
    HANDOFF_UNCONFIRMED_REASONS,
)
from ccs.core.types import TRANSFER_STATUSES
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

_FIXTURE_DIR = "handoff"
_EXPECTED_FIXTURE_COUNT = 39

# Frozen expectations: deliberate duplicates of the code's vocabularies, per the
# tests/CLAUDE.md house rule. A set derived from the code would move its own
# goalposts; each is compared to the code's set below, and its cardinality is
# pinned separately so a same-size add+remove cannot slip past set equality.
_EXPECTED_STATUSES: frozenset[str] = frozenset({
    "pending", "completed", "declined", "withdrawn", "overtaken", "superseded",
})
_EXPECTED_STATUS_COUNT = 6
_EXPECTED_TRANSFER_REFUSAL_REASONS: frozenset[str] = frozenset({
    "handoff_to_self",
    "handoff_successor_unknown",
    "handoff_successor_malformed",
    "handoff_not_held",
    "handoff_version_unconfirmed",
    "handoff_in_flight",
    "handoff_other_holder",
    "handoff_ended",
})
_EXPECTED_VERB_REASONS: frozenset[str] = frozenset({
    "handoff_not_successor", "handoff_not_giver", "handoff_not_live",
})
_EXPECTED_UNCONFIRMED_REASONS: frozenset[str] = frozenset({
    "handoff_transfer_unconfirmed",
    "handoff_accept_unconfirmed",
    "handoff_decline_unconfirmed",
    "handoff_withdraw_unconfirmed",
})
_EXPECTED_REASON_COUNT = 16

# The row that pins each status. A status may appear in other rows too; this is
# the one whose removal must turn the coverage test red.
_STATUS_FIXTURES: dict[str, str] = {
    "pending": "handoff-transfer-pending-hands-on-an-exclusive-claim",
    "completed": "handoff-accept-completes-a-pending-handoff",
    "declined": "handoff-decline-ends-a-handoff",
    "withdrawn": "handoff-withdraw-ends-a-handoff",
    "overtaken": "handoff-bystander-acquire-overtakes-a-live-handoff",
    "superseded": "handoff-resend-of-a-superseded-transfer-answers-ended-superseded",
}

# The row that pins each reachable wire reason.
_REASON_FIXTURES: dict[str, str] = {
    "handoff_to_self": "handoff-transfer-to-self-is-refused",
    "handoff_successor_unknown": "handoff-transfer-to-an-unknown-successor-is-refused",
    "handoff_successor_malformed": "handoff-transfer-to-a-malformed-successor-is-refused-per-grant",
    "handoff_not_held": "handoff-mixed-transfer-answers-per-grant-not-held",
    "handoff_in_flight": "handoff-transfer-past-a-live-handoff-is-refused-in-flight",
    "handoff_other_holder": "handoff-transfer-under-a-foreign-write-holder-is-refused",
    "handoff_ended": "handoff-resend-after-decline-answers-ended",
    "handoff_not_successor": "handoff-accept-by-a-non-successor-is-refused",
    "handoff_not_giver": "handoff-withdraw-by-a-non-giver-is-refused",
    "handoff_not_live": "handoff-accept-of-an-ended-handoff-is-refused-not-live",
    "handed_off": "handoff-giver-pre-edit-is-denied-handed-off",
}

# The reasons no fixture can drive, each with why and where it IS pinned. The
# coverage test requires this set and _REASON_FIXTURES to partition
# HANDOFF_REASONS exactly, so a new reason must be given a row or an entry here.
_EXCLUDED_REASONS: dict[str, str] = {
    # The four per-verb degraded answers: the watchdog must abandon a work body,
    # and the harness has no seam to hold the registry lock past the deadline.
    # Pinned by tests/test_claude_code_coordinator_server.py::
    # test_a_handoff_verb_cut_short_by_the_watchdog_answers_unconfirmed_and_lands_nothing.
    "handoff_transfer_unconfirmed": "degraded: watchdog abandons the transfer's work body",
    "handoff_accept_unconfirmed": "degraded: watchdog abandons the accept's work body",
    "handoff_decline_unconfirmed": "degraded: watchdog abandons the decline's work body",
    "handoff_withdraw_unconfirmed": "degraded: watchdog abandons the withdraw's work body",
    # No HTTP request can hold a claim on a version-0 path: every registration
    # path (pre-read, pre-edit, pre-bash, /session/begin) seeds the artifact at
    # version 1 through resolve_or_register, and versions only rise. The pending
    # row's path is first seen through pre-edit and is admitted at v1, which is
    # the on-wire evidence. Pinned by tests/test_service_handoff.py::
    # test_a_mixed_transfer_answers_per_grant and
    # tests/coordinator/test_transfer_record.py::
    # test_an_unconfirmed_version_is_refused_and_writes_no_record, which
    # register a version-0 artifact directly.
    "handoff_version_unconfirmed": "no HTTP registration path seeds a version below 1",
}

# The giver's refusal on each write route a fixture can reach.
_GIVER_ROUTE_FIXTURES: dict[str, str] = {
    "/hooks/pre-edit": "handoff-giver-pre-edit-is-denied-handed-off",
    "/hooks/post-edit-cas": "handoff-giver-compare-and-swap-is-refused-handed-off",
    "/hooks/post-edit": "handoff-giver-post-edit-commit-is-refused-with-the-handoff-arm",
    "/hooks/pre-bash": "handoff-giver-shell-write-is-denied-handed-off",
}

# Each role's view of a record: the row, the role it is projected for, and
# whether the record is live.
_ROLE_FIXTURES: dict[str, tuple[str, str, bool]] = {
    "giver while live, on a read": (
        "handoff-giver-read-while-live-carries-the-notice", "giver", True),
    "giver after the record ended": (
        "handoff-giver-read-after-completion-carries-the-outcome", "giver", False),
    "successor": (
        "handoff-successor-read-carries-provenance-for-an-exclusive-hold", "successor", True),
    "bystander": ("handoff-bystander-read-carries-the-advisory", "bystander", True),
}

# Sentences a named row pins, hand-copied from docs/guide.md: the one an
# EXCLUSIVE hold adds to the successor's provenance, and the post-edit arm's
# statement that the edit is on disk without a version.
_EXCLUSIVE_SHAPE_SENTENCE = (
    "Agent 09f031b6 held an uncommitted write claim when it handed plan.md on, so "
    "the file on disk may differ from v1; read plan.md before editing it."
)
_ON_DISK_SENTENCE = (
    "Your edit landed in your local worktree but was not given a version by the "
    "coordinator."
)

# The keys that name a party, and the sessions the fixtures use. The session-
# level agent id is uuid5 over the session id (docs/guide.md, "Naming the
# successor"), recomputed here so the fixtures' literals are checked against the
# derivation rather than pasted twice.
_IDENTITY_KEYS: frozenset[str] = frozenset({"giver", "successor", "counterparty", "agent_id"})
_SESSIONS = {
    "giver": "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    "successor": "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    "bystander": "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
    "unknown": "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
}

# Noise a row may ignore, and why. Anything else is a hole in the catch surface.
_ALLOWED_IGNORE_KEYS: frozenset[str] = frozenset({
    # Wall-clock floats in today's stale-read summary (session_start/04 ignores them too).
    "last_writer_at_unix_ts",
    "warning_generated_at_unix_ts",
    # /status: the absolute workspace root, and the package version (warn_mode/08).
    "policy_summary",
    "coordinator_version",
    "coordinator_root",
})

_NODE_ROUTES = frozenset({
    "/handoff/transfer", "/handoff/accept", "/handoff/decline", "/handoff/withdraw",
})

_NODE_DIST_PATH = resolve_node_dist_path()
_NODE_DIST_UNRESOLVED = (
    "The Node asymmetry rows need a REAL answer from the sibling coordinator, and this row got none: "
    "the plugin dist could not be resolved. Failing rather than xfailing is "
    "deliberate -- the claim under test is that the Node backend answers 404 for "
    "the four handoff routes, and an xfail records that nobody asked. Build the "
    "plugin checkout (npm ci && npm run build) or set "
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
def test_handoff_fixture_response_matches_expected(
    row: tuple[Fixture, str], tmp_path: Path
) -> None:
    """The FULL response body is the assertion, after normalization on both
    sides with the row's own identity opt-in. A renamed reason, a changed
    template byte, a dropped or added key, or the parties swapped all fail
    here; on Node, a route that stopped answering 404 fails here."""
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
        f"removing one is a deliberate change to the pinned handoff wire -- update this "
        f"literal in the same diff."
    )
    assert len({f.name for f in fixtures}) == len(fixtures), "fixture names must be unique"
    assert len(_ROWS) == len(fixtures), "every handoff fixture runs on exactly one backend"


def test_every_transfer_status_has_its_row() -> None:
    """Every status in the code's vocabulary has its named row, and that row's
    expected body carries the status. Deleting a status row is red here (not
    only in the count), and a status added to ``TRANSFER_STATUSES`` without a
    row is red too.

    ``superseded`` is never stored, so its row must be what answers it: the
    re-send of a transfer tuple that an earlier preflight sent and a later one
    replaced."""
    assert _EXPECTED_STATUSES == TRANSFER_STATUSES, (
        f"the status vocabulary drifted: only in code "
        f"{sorted(TRANSFER_STATUSES - _EXPECTED_STATUSES)}, only here "
        f"{sorted(_EXPECTED_STATUSES - TRANSFER_STATUSES)}"
    )
    assert len(TRANSFER_STATUSES) == _EXPECTED_STATUS_COUNT
    assert set(_STATUS_FIXTURES) == TRANSFER_STATUSES

    by_name = _by_name()
    for status, name in _STATUS_FIXTURES.items():
        assert name in by_name, f"no row pins the {status!r} status: {name!r} is missing"
        fixture = by_name[name]
        assert fixture.backends == (BACKEND_PYTHON,), name
        assert status in set(_values(_body(fixture), "status")), (
            f"{name} is the {status!r} row but its expected body does not carry it"
        )

    superseded = by_name[_STATUS_FIXTURES["superseded"]]
    transfers = [
        req for req in superseded.setup["preflight_requests"]
        if req["path"] == "/handoff/transfer"
    ]
    assert superseded.request["path"] == "/handoff/transfer"
    assert len(transfers) == 2 and transfers[0] == superseded.request, (
        "the superseded row must re-send the FIRST transfer after a second one "
        "replaced it; anything else answers some other status"
    )
    assert transfers[1]["body"]["successor"] != superseded.request["body"]["successor"]


def test_every_wire_reason_has_its_row_or_a_named_exclusion() -> None:
    """Every handoff wire reason is pinned by a named row or excluded by name
    with its reason, never silently.

    Keyed on the code's own sets: the transfer refusals, the verb refusals, the
    giver's ``handed_off`` and the per-verb unconfirmed reasons must partition
    ``HANDOFF_REASONS``, the rows and exclusions must cover it exactly, and each
    row's expected body must carry its reason where the wire puts it (a
    per-grant ``reason`` for a transfer refusal, a top-level ``reason``
    otherwise). Deleting any reason row is red here."""
    assert HANDOFF_TRANSFER_REFUSAL_REASONS == _EXPECTED_TRANSFER_REFUSAL_REASONS
    assert HANDOFF_UNCONFIRMED_REASONS == _EXPECTED_UNCONFIRMED_REASONS
    verb_reasons = (
        HANDOFF_REASONS
        - HANDOFF_TRANSFER_REFUSAL_REASONS
        - HANDOFF_UNCONFIRMED_REASONS
        - {GIVER_FENCED_REASON}
    )
    assert verb_reasons == _EXPECTED_VERB_REASONS, (
        f"a handoff reason belongs to no known family: {sorted(verb_reasons ^ _EXPECTED_VERB_REASONS)}"
    )
    assert GIVER_FENCED_REASON == "handed_off"
    assert len(HANDOFF_REASONS) == _EXPECTED_REASON_COUNT

    rows, excluded = set(_REASON_FIXTURES), set(_EXCLUDED_REASONS)
    assert not rows & excluded, f"a reason is both pinned and excluded: {sorted(rows & excluded)}"
    assert rows | excluded == HANDOFF_REASONS, (
        f"reasons with neither a row nor a named exclusion: "
        f"{sorted(HANDOFF_REASONS - rows - excluded)}; unknown names: "
        f"{sorted((rows | excluded) - HANDOFF_REASONS)}"
    )
    assert HANDOFF_UNCONFIRMED_REASONS <= excluded
    assert excluded - HANDOFF_UNCONFIRMED_REASONS == {"handoff_version_unconfirmed"}, (
        "the only reachable-family exclusion is the version-0 refusal; any other "
        "is a new decision and needs its reason written into _EXCLUDED_REASONS"
    )

    by_name = _by_name()
    for reason, name in _REASON_FIXTURES.items():
        assert name in by_name, f"no row pins {reason!r}: {name!r} is missing"
        body = _body(by_name[name])
        if reason in HANDOFF_TRANSFER_REFUSAL_REASONS:
            carried = [g.get("reason") for g in body.get("grants", [])]
        else:
            carried = [body.get("reason")]
        assert reason in carried, (
            f"{name} is the {reason!r} row but its expected body does not carry it "
            f"where the wire does: {body!r}"
        )


def test_the_givers_refusal_is_pinned_on_each_write_route() -> None:
    """``handed_off`` on pre-edit carries the deny envelope; on the
    compare-and-swap route it carries none; on the post-edit commit it carries
    the context-only post-tool envelope with the on-disk sentence. All three
    keep the typed reason, the successor and the version at transfer at the
    top level, so a client classifies by ``reason`` and never by prose."""
    by_name = _by_name()
    for route, name in _GIVER_ROUTE_FIXTURES.items():
        assert name in by_name, f"no row pins the giver's refusal on {route}"
        fixture = by_name[name]
        assert fixture.request["path"] == route
        assert fixture.request["body"]["session_id"] == _SESSIONS["giver"]
        body = _body(fixture)
        assert body["ok"] is False and body["reason"] == "handed_off"
        assert {"successor", "version_at_transfer", "handoff"} <= set(body)
        assert body["handoff"]["role"] == "giver" and body["handoff"]["live"] is True

    deny = _body(by_name[_GIVER_ROUTE_FIXTURES["/hooks/pre-edit"]])["hookSpecificOutput"]
    assert deny["hookEventName"] == "PreToolUse"
    assert deny["permissionDecision"] == "deny"
    assert "hookSpecificOutput" not in _body(
        by_name[_GIVER_ROUTE_FIXTURES["/hooks/post-edit-cas"]])
    arm = _body(by_name[_GIVER_ROUTE_FIXTURES["/hooks/post-edit"]])["hookSpecificOutput"]
    assert arm["hookEventName"] == "PostToolUse" and "permissionDecision" not in arm
    assert _ON_DISK_SENTENCE in arm["additionalContext"]


def test_the_givers_shell_write_row_is_python_only_and_names_a_shell_write() -> None:
    """The shell-write row runs where a handoff exists, and its request is a
    shell write that reads nothing -- so only the write check can answer it a
    deny."""
    fixture = _by_name()[_GIVER_ROUTE_FIXTURES["/hooks/pre-bash"]]
    assert fixture.backends == (BACKEND_PYTHON,)
    assert fixture.request["body"]["command"] == "echo '- gamma' >> plan.md"


def test_the_givers_pre_edit_answers_the_shell_write_rows_body(tmp_path: Path) -> None:
    """The shell-write row's setup with the giver's pre-edit on the handed-off
    path instead of its shell write: the same expected body. Fails if the shell
    route's deny drifts from the edit route's, even if the row were re-recorded
    from a drifted shell route."""
    fixture = _by_name()[_GIVER_ROUTE_FIXTURES["/hooks/pre-bash"]]
    pre_edit = dataclasses.replace(fixture, request={
        **fixture.request,
        "path": "/hooks/pre-edit",
        "body": {"session_id": fixture.request["body"]["session_id"], "path": "plan.md"},
    })
    status, body = run_scenario(
        fixture=pre_edit, backend_id=BACKEND_PYTHON, workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )
    assert status == fixture.expected["status"], body
    assert body == normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
        preserve_identity=fixture.preserve_identity,
    )


def test_each_role_has_its_view_pinned() -> None:
    """The giver (while live, on a read, with its notice; and after the record
    ended), the successor and the bystander each have a row whose handoff key
    is projected for that role, and the successor's row is the EXCLUSIVE-shape
    one, which adds the sentence about an uncommitted write claim."""
    by_name = _by_name()
    for label, (name, role, live) in _ROLE_FIXTURES.items():
        assert name in by_name, f"no row pins the {label} view"
        fixture = by_name[name]
        assert fixture.request["path"] == "/hooks/pre-read", f"{label}: a read's view"
        key = _body(fixture)["handoff"]
        assert (key["role"], key["live"]) == (role, live), f"{label}: {key!r}"
        assert _body(fixture)["hookSpecificOutput"]["additionalContext"], label

    successor = _body(by_name[_ROLE_FIXTURES["successor"][0]])
    assert successor["handoff"]["hold_shape"] == "EXCLUSIVE"
    assert _EXCLUSIVE_SHAPE_SENTENCE in successor["hookSpecificOutput"]["additionalContext"]


def test_every_identity_a_row_names_is_compared_verbatim() -> None:
    """The handoff key's whole content is WHO handed what to whom. Its ids are
    hyphenated UUIDs, which the harness's portability default scrubs to
    ``<UUID>`` -- under the default, a body crediting the giver as the successor
    would pass. So every party key a row asserts must be in that row's
    ``preserve_identity``, and every asserted id must be the session-level agent
    id of one of the fixture sessions, recomputed from its session id here."""
    known = {
        str(uuid5(NAMESPACE_URL, f"ccs-agent:claude-session-{sid}"))
        for sid in _SESSIONS.values()
    }
    for fixture in _fixtures():
        body = _body(fixture)
        asserted = {
            key for key in _IDENTITY_KEYS
            if any(isinstance(v, str) for v in _values(body, key))
        }
        assert asserted <= fixture.preserve_identity, (
            f"{fixture.name} asserts {sorted(asserted - fixture.preserve_identity)} "
            f"without preserving them, so any party would match"
        )
        for key in asserted:
            for value in _values(body, key):
                if isinstance(value, str):
                    assert value in known, f"{fixture.name}: {key}={value!r} is no fixture session"

    # The swap the opt-in exists to catch really is invisible under the default.
    giver, successor = sorted(known)[:2]
    right = {"handoff": {"giver": giver, "successor": successor}}
    swapped = {"handoff": {"giver": successor, "successor": giver}}
    assert normalize_response(right) == normalize_response(swapped)
    keep = frozenset({"giver", "successor"})
    assert normalize_response(right, preserve_identity=keep) != normalize_response(
        swapped, preserve_identity=keep)


def test_the_status_rows_project_the_record_on_both_tiers() -> None:
    """``/status`` carries the record, without a role, on the default tier and
    the operator tier, and only the operator tier adds ``created_at_unix_ts``.
    That key is a wall-clock value, so it must be in the harness's
    timestamp scrub set or the operator row is unpinnable: asserted directly,
    so removing it from the set is red here with a readable reason."""
    by_name = _by_name()
    default = _body(by_name["handoff-status-default-tier-carries-the-record"])
    operator = _body(by_name["handoff-status-operator-tier-adds-the-created-timestamp"])
    (default_key,) = [e["handoff"] for e in default["tracked_artifacts"] if "handoff" in e]
    (operator_key,) = [e["handoff"] for e in operator["tracked_artifacts"] if "handoff" in e]
    assert "role" not in default_key and "role" not in operator_key
    assert "created_at_unix_ts" not in default_key
    assert set(operator_key) - set(default_key) == {"created_at_unix_ts"}
    assert normalize_response({"created_at_unix_ts": 1791276728.647885}) == {
        "created_at_unix_ts": "<TS>"
    }, "created_at_unix_ts must be in the harness's timestamp scrub set"


def test_no_handoff_fixture_expects_an_unreachable_body() -> None:
    """The documented-unreachable twin: honest limits, stated as assertions.

    The harness cannot abandon a work body or make a release fail, so three
    kinds of answer have no row here and are pinned by the route tests instead:

    - the four per-verb degraded bodies (``ok: false, degraded: true`` with a
      ``handoff_*_unconfirmed`` reason) -- by
      ``tests/test_claude_code_coordinator_server.py::
      test_a_handoff_verb_cut_short_by_the_watchdog_answers_unconfirmed_and_lands_nothing``;
    - a session stop that leaves a grant held (``ok: false`` with a grant
      ``{"held": true, "reason"}``), reachable only through a failing
      ``invalidate``, which outside the watchdog's abort guard no request can
      cause -- by ``tests/test_claude_code_coordinator_server.py::
      test_a_session_stop_that_keeps_a_grant_answers_per_grant``, which
      monkeypatches it. The reachable per-grant release answers are pinned:
      the giver's failed-edit report and the giver's clean session stop.
    - ``handoff_version_unconfirmed``; see :data:`_EXCLUDED_REASONS`.

    If a fixture ever expects one of these, this test goes red: rewrite it to
    assert that body's shape instead of its absence."""
    for fixture in _fixtures():
        body = _body(fixture)
        assert not list(_values(body, "degraded")), (
            f"{fixture.name} expects a degraded body; the harness cannot drive one"
        )
        reasons = set(_values(body, "reason"))
        assert not reasons & set(_EXCLUDED_REASONS), (
            f"{fixture.name} expects an excluded reason {sorted(reasons & set(_EXCLUDED_REASONS))}"
        )
        assert True not in list(_values(body, "held")), (
            f"{fixture.name} expects a grant still held after a release"
        )

    by_name = _by_name()
    failed_edit = _body(by_name["handoff-giver-failed-edit-answers-per-grant"])
    assert failed_edit["ok"] is False
    assert failed_edit["grants"] == [{
        "path": "plan.md", "held": False, "cause": "handoff",
        "successor": failed_edit["successor"], "version_at_transfer": 1,
    }]
    stop = by_name["handoff-giver-session-stop-keeps-todays-bytes"]
    assert stop.request["path"] == "/hooks/session-stop"
    assert _body(stop) == {"ok": True, "released_artifacts": ["spec.md"]}


def test_ignore_keys_are_limited_to_documented_noise() -> None:
    """Each ``ignore_keys`` entry is a hole in the catch surface. Only today's
    wall-clock and workspace-path fields may be ignored, never a handoff
    field, and no row may drop a key with ``optional_keys``."""
    for fixture in _fixtures():
        assert fixture.ignore_keys <= _ALLOWED_IGNORE_KEYS, (
            f"{fixture.name} ignores {sorted(fixture.ignore_keys - _ALLOWED_IGNORE_KEYS)}"
        )
        assert not fixture.optional_keys, fixture.name


def test_the_node_rows_cannot_be_satisfied_by_a_skip() -> None:
    """The Node asymmetry, asserted directly: the plugin dist resolves, and
    exactly the four handoff routes have a node-only 404 row sent with the
    harness's valid bearer (Node rejects auth before routing, so a bad one
    answers 401 and says nothing about which routes exist). Every other row is
    Python-only.

    Leave ``AGENT_COHERENCE_PLUGIN_DIST_PATH`` unset, point ``HOME`` elsewhere
    and run from a checkout with no sibling ``agent-coherence-plugin`` to watch
    this and the four Node rows go red."""
    assert _NODE_DIST_PATH is not None, _NODE_DIST_UNRESOLVED
    assert _NODE_DIST_PATH.exists(), f"resolved dist path does not exist: {_NODE_DIST_PATH}"

    node_rows = [f for f in _fixtures() if BACKEND_NODE in f.backends]
    assert {f.request["path"] for f in node_rows} == _NODE_ROUTES
    assert len(node_rows) == len(_NODE_ROUTES)
    for fixture in node_rows:
        assert fixture.backends == (BACKEND_NODE,), fixture.name
        assert fixture.expected == {"status": 404, "body": {"error": "not found"}}, fixture.name
        assert "headers" not in fixture.request, fixture.name
    python_rows = [f for f in _fixtures() if f not in node_rows]
    assert all(f.backends == (BACKEND_PYTHON,) for f in python_rows)
