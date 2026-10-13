# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Cross-implementation effect-fence verdict corpus (plan Unit 5, R9 + R10).

``POST /hooks/effect-fence`` answers "may this irreversible effect still fire,
and if not, WHY". The verdict envelope and the reason vocabulary are declared
protocol, and this module is where that claim stops being prose: every member
of :data:`~ccs.core.exceptions.HOLD_REASONS` is driven from a fixture through
the real HTTP route, the affirmative ``proceed`` arm is driven beside them, and
each fixture's asserted body is the FULL response, so a reason rename or an
extra key fails rather than passing a subset match.

Two silent failures this module exists to defeat, both of which report green:

1. **A fixture directory nothing names loads as empty.** Each corpus module
   loads fixtures by literal directory name, so a new directory no module names
   is collected by nothing at all. ``test_fixture_directory_is_actually_loaded``
   pins the count against a frozen literal, so an unloaded or silently emptied
   directory is red rather than vacuously green.
2. **A Node-backend row skips itself when the sibling build is unresolvable.**
   The warn-mode module ``xfail``s such rows, which is right for a parity
   fixture and WRONG for R10: the whole claim is that the sibling answers 404,
   and an xfail records that nothing asked. Node rows here therefore
   ``pytest.fail`` instead, and ``test_node_asymmetry_row_cannot_be_satisfied_
   by_a_skip`` asserts the resolution directly.

This module also owns the harness's own IDENTITY CAPABILITY (R15), under
``fixtures/harness_identity/``. It lives beside the verdict corpus rather than
in a module of its own because the capability exists FOR this corpus: the
effect fence is where a caller principal will have to be pinned, and KTD7 says
a principal fixture written before the harness can carry a minted value and
tell one identity from another would be a decorative assertion. The three
capability fixtures therefore drive routes that already ship — ``/session/begin``
mints a token, a stale read names the session that moved the bytes, and
``/status`` names which session holds which artifact — so the capability is
proven against real answers rather than against itself.

The third fixture is the one that OBSERVES the identity opt-in. The stale-read
attribution turned out to be spelled as 32-char hex, which the portability
scrub (8-4-4-4-12 only) never matched, so that fixture passes identically with
``preserve_identity`` deleted; ``sessions[].agent_id`` is the hyphenated form
the scrub does collapse, and pinning which agent holds which artifact is
vacuous there unless the opt-in is honoured.

Marked ``protocol_corpus`` — opt-in via ``pytest -m protocol_corpus``."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import replace
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest

from ccs.core.exceptions import HOLD_REASONS
from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    PRINCIPAL_SENTINEL,
    UUID_SENTINEL,
    Fixture,
    FixtureContractError,
    FixtureSubstitutionError,
    build_fixture,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

_FIXTURE_DIR = "effect_fence"

# Frozen expectations. Deliberate duplicates of the code under test, per the
# tests/CLAUDE.md house rule: a set DERIVED from the route would move its own
# goalposts, so the edit that breaks the contract would also update the
# expectation and this module would report green. Cardinality is pinned
# separately so a same-size add+remove cannot slip past set equality.
_EXPECTED_HOLD_REASONS: frozenset[str] = frozenset({
    "version_moved",
    "grant_reclaimed",
    "grant_preempted",
    "input_vanished",
    "version_unconfirmed",
    "read_denied",
    "content_claim_absent",
    "generation_unconfirmed",
})
_EXPECTED_HOLD_REASON_COUNT = 8
_EXPECTED_FIXTURE_COUNT = 13

# The complete verdict/error envelope. Every key any fixture may assert.
# ``degraded``/``held_by`` belong to the two degraded arms, which a fixture
# cannot drive (see test_degraded_arms_are_documented_as_unreachable_here).
_ENVELOPE_KEYS: frozenset[str] = frozenset({
    "verdict", "reason", "degraded", "held_by", "error",
})

# The content-hash convention, pinned as bytes rather than described: lowercase
# sha-256 hex over the EXACT bytes the caller holds, no normalization. The
# anchor fixture's file content carries trailing spaces AND a trailing newline
# precisely so a normalizing implementation produces a visibly different digest.
_CONVENTION_FIXTURE = "effect-fence-proceed-all-legs-confirmed"
_CONVENTION_PATH = "docs/plan.md"


def _all_effect_fence_fixtures() -> list[Fixture]:
    """Loaded once at collection so parametrize ids stay stable."""
    return load_fixtures(_FIXTURE_DIR)


def _parametrize_rows() -> list[tuple[Fixture, str]]:
    rows: list[tuple[Fixture, str]] = []
    for fixture in _all_effect_fence_fixtures():
        for backend in fixture.backends:
            rows.append((fixture, backend))
    return rows


def _row_id(row: tuple[Fixture, str]) -> str:
    fixture, backend = row
    return f"{fixture.name}[{backend}]"


_ROWS = _parametrize_rows()
_NODE_DIST_PATH = resolve_node_dist_path()

_NODE_DIST_UNRESOLVED = (
    "R10 needs a REAL answer from the sibling coordinator, and this row got "
    "none: the plugin dist could not be resolved. Failing rather than xfailing "
    "is deliberate — the claim under test is that the Node backend answers 404 "
    "for /hooks/effect-fence, and an xfail records that nobody asked, which is "
    "how a cross-implementation claim silently stops being checked. Build the "
    "plugin checkout (npm ci && npm run build) or set "
    "AGENT_COHERENCE_PLUGIN_DIST_PATH to the absolute dist/coordinator.js path."
)


@pytest.mark.parametrize("row", _ROWS, ids=[_row_id(r) for r in _ROWS] if _ROWS else None)
def test_effect_fence_fixture_response_matches_expected(
    row: tuple[Fixture, str],
    tmp_path: Path,
) -> None:
    """The FULL response body is the assertion, not a subset of it.

    A changed reason identifier, an added key, or a dropped one all fail here.
    That is the point of pinning the contract in the corpus: the vocabulary is
    add-never-rename, and a rename un-matches every consumer's ``reason ==
    CONSTANT`` branch, downgrading a HOLD each of them no longer recognises."""
    fixture, backend = row
    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.fail(f"{fixture.name}: {_NODE_DIST_UNRESOLVED}")

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
        f"expected {expected_status}, got {actual_status}\nbody={actual_body!r}"
    )
    assert actual_body == expected_body, (
        f"{fixture.name}[{backend}]: body mismatch\n"
        f"expected={expected_body!r}\nactual=  {actual_body!r}"
    )


def test_fixture_directory_is_actually_loaded() -> None:
    """Silent failure #1: a fixture directory no module names loads as EMPTY.

    ``load_fixtures`` returns ``[]`` for a directory that does not exist, and
    an empty parametrize list is collected as nothing — so a misspelled
    directory name, a moved fixtures root, or a wholesale deletion all report
    green. Pinning the count against a frozen literal is what makes any of
    those red."""
    fixtures = _all_effect_fence_fixtures()
    assert len(fixtures) == _EXPECTED_FIXTURE_COUNT, (
        f"Expected exactly {_EXPECTED_FIXTURE_COUNT} fixtures in "
        f"tests/protocol_corpus/fixtures/{_FIXTURE_DIR}/, found {len(fixtures)}. "
        f"Adding or removing one is a deliberate change to the pinned verdict "
        f"contract — update this literal in the same diff."
    )
    assert len(_ROWS) >= len(fixtures), "every fixture must contribute at least one row"


def test_every_hold_reason_has_a_fixture() -> None:
    """R9 EXHAUSTIVENESS, with NO carve-out.

    Every member of ``HOLD_REASONS`` is returnable by this route and is driven
    by a fixture here — including ``input_vanished``, which an earlier reading
    of the plan excluded on the grounds that the coordinator never stats the
    caller's workspace. It does not need to: the no-record path (untracked,
    never observed, or a row that vanished between lookup and read) builds
    comparands with ``current_version=None``, the vanish sentinel, so the
    branch table answers ``input_vanished`` on its first leg.

    A reason added to the vocabulary without a fixture therefore turns the
    corpus red, which is the enforcement this unit exists to add: the eighth
    reason (``content_claim_absent``) was split out of the residual bucket
    precisely because two cases were byte-identical on the wire, and nothing
    but a fixture pair can keep them distinguishable."""
    assert _EXPECTED_HOLD_REASONS == HOLD_REASONS, (
        "The published reason vocabulary drifted from this module's frozen "
        f"expectation.\n  only in HOLD_REASONS: {sorted(HOLD_REASONS - _EXPECTED_HOLD_REASONS)}"
        f"\n  only in expectation:  {sorted(_EXPECTED_HOLD_REASONS - HOLD_REASONS)}"
    )
    assert len(HOLD_REASONS) == _EXPECTED_HOLD_REASON_COUNT, (
        f"HOLD_REASONS cardinality moved: expected {_EXPECTED_HOLD_REASON_COUNT}, "
        f"got {len(HOLD_REASONS)}. Pinned separately from set equality so a "
        f"same-size add+remove cannot slip past."
    )

    covered = {
        f.expected["body"]["reason"]
        for f in _all_effect_fence_fixtures()
        if f.expected["body"].get("verdict") == "hold"
    }
    missing = HOLD_REASONS - covered
    assert not missing, (
        f"No fixture drives these HOLD reasons: {sorted(missing)}. Every member "
        f"of HOLD_REASONS is reachable from this route — if you believe one is "
        f"not, say so at the site rather than dropping it from the set."
    )
    assert covered <= HOLD_REASONS, (
        f"A fixture asserts a reason outside the published vocabulary: "
        f"{sorted(covered - HOLD_REASONS)}"
    )


def test_content_claim_absent_and_generation_unconfirmed_stay_distinguishable() -> None:
    """The one forced fixture the eighth reason costs.

    Fixtures 06 and 07 send BYTE-IDENTICAL request bodies. The only difference
    is whether the preflight pre-read carried a ``content_hash``, i.e. whether
    the coordinator holds a content claim at all. Before ``content_claim_absent``
    existed these two answered the same thing, with the same hint and the same
    retryability, for cases whose recovery differs: re-read your bytes, versus
    check the daemon's version and call an operator.

    Asserting the request bodies are identical is what makes the pair a
    CONTROL. Two fixtures that merely happen to return different reasons prove
    nothing if their inputs differ in five ways."""
    by_name = {f.name: f for f in _all_effect_fence_fixtures()}
    absent = by_name["effect-fence-hold-content-claim-absent-no-recorded-hash"]
    unconfirmed = by_name["effect-fence-hold-generation-unconfirmed-null-comparand"]

    assert absent.request == unconfirmed.request, (
        "The pair is only a control while the two requests are identical; they "
        f"differ:\n  absent=     {absent.request!r}\n  unconfirmed={unconfirmed.request!r}"
    )
    assert absent.expected["body"]["reason"] == "content_claim_absent"
    assert unconfirmed.expected["body"]["reason"] == "generation_unconfirmed"
    assert absent.expected["body"] != unconfirmed.expected["body"], (
        "identical requests must produce DIFFERENT wire answers, or the split "
        "bought nothing"
    )

    # And the one input that differs is the coordinator's content claim.
    def _preflight_hashes(fixture: Fixture) -> list[str | None]:
        return [
            req["body"].get("content_hash")
            for req in fixture.setup["preflight_requests"]
        ]

    assert _preflight_hashes(absent) == [None], (
        "the content_claim_absent arm must observe the artifact WITHOUT a "
        "content hash — that absent claim is the whole input under test"
    )
    assert _preflight_hashes(unconfirmed) == [
        unconfirmed.request["body"]["content_hash"]
    ], "the generation_unconfirmed arm must observe WITH the caller's own hash"


def test_content_hash_convention_is_sha256_of_exact_bytes() -> None:
    """The convention, recomputed rather than trusted.

    ``content_hash`` is the lowercase sha-256 hex of the EXACT bytes the caller
    holds, with no normalization. A fixture literal alone proves nothing — it
    is just a hex string both sides agree on — so this recomputes the digest
    from the anchor fixture's own file content and asserts the match, then
    asserts that the two obvious normalizations (``strip``, trailing-newline
    trim) produce DIFFERENT digests. That second half is the load-bearing one:
    without it, an implementation that stripped before hashing would satisfy a
    test written only against its own output."""
    by_name = {f.name: f for f in _all_effect_fence_fixtures()}
    anchor = by_name[_CONVENTION_FIXTURE]

    content: str = anchor.setup["files"][_CONVENTION_PATH]
    pinned: str = anchor.request["body"]["content_hash"]

    assert content.encode() == b"# coherence corpus fixture  \n", (
        "the anchor's bytes are pinned literally here too — trailing spaces and "
        "the trailing newline are what make the normalization arms below bite"
    )
    assert pinned == hashlib.sha256(content.encode()).hexdigest(), (
        f"content_hash is not the sha-256 of the exact bytes of "
        f"{_CONVENTION_PATH}.\n  pinned=   {pinned}\n  recomputed="
        f"{hashlib.sha256(content.encode()).hexdigest()}"
    )
    assert pinned == pinned.lower() and len(pinned) == 64, (
        "the wire form is 64 lowercase hex characters"
    )

    for label, variant in (
        ("strip()", content.strip()),
        ("rstrip newline", content.rstrip("\n")),
        ("rstrip whitespace", content.rstrip()),
    ):
        assert hashlib.sha256(variant.encode()).hexdigest() != pinned, (
            f"a {label} normalization produced the SAME digest as the exact "
            f"bytes, so this fixture cannot tell a normalizing implementation "
            f"from a conforming one — change the anchor's content"
        )

    # And the convention has teeth on the wire: the mismatch twin sends a digest
    # of different bytes against the same workspace and HOLDS where 01 proceeds.
    mismatch = by_name["effect-fence-hold-generation-unconfirmed-content-hash-mismatch"]
    assert mismatch.setup["files"] == anchor.setup["files"]
    assert mismatch.request["body"]["expected_version"] == anchor.request["body"]["expected_version"]
    assert mismatch.request["body"]["expected_generation"] == anchor.request["body"]["expected_generation"]
    assert mismatch.request["body"]["content_hash"] != pinned, (
        "the mismatch twin must differ from the anchor in its DIGEST alone"
    )
    assert anchor.expected["body"] == {"verdict": "proceed"}
    assert mismatch.expected["body"]["verdict"] == "hold"


def test_node_asymmetry_row_cannot_be_satisfied_by_a_skip() -> None:
    """Silent failure #2, asserted directly.

    R10 claims the sibling Node coordinator does not implement this route and
    answers 404. That claim is checked only if the Node row actually RAN, and
    the harness's own default for an unresolvable dist is ``xfail`` — a green
    result that recorded no answer. This asserts the resolution itself, so a
    machine or CI job without the built plugin fails the gate rather than
    quietly satisfying it.

    Leave ``AGENT_COHERENCE_PLUGIN_DIST_PATH`` unset, redirect ``HOME`` (the
    resolver's third fallback is ``~/projects/...``) and run from a checkout
    with no sibling ``agent-coherence-plugin`` to watch this go red — that is
    the mutation this assertion exists to fail against. Pointing the variable
    at a nonexistent file no longer reaches this assertion: the resolver
    refuses an explicit path that does not exist, so the module fails at
    collection instead."""
    assert _NODE_DIST_PATH is not None, _NODE_DIST_UNRESOLVED
    assert _NODE_DIST_PATH.exists(), (
        f"resolved dist path does not exist: {_NODE_DIST_PATH}"
    )

    node_rows = [f for f in _all_effect_fence_fixtures() if BACKEND_NODE in f.backends]
    assert len(node_rows) == 1, (
        f"expected exactly one Node-backed fixture recording the asymmetry, "
        f"found {[f.name for f in node_rows]}"
    )
    (asymmetry,) = node_rows
    assert asymmetry.expected["status"] == 404, (
        "the asymmetry fixture's whole content is the 404; a different status "
        "means Node grew the route and the two branch tables now need comparing"
    )
    assert asymmetry.request["path"] == "/hooks/effect-fence"
    assert "headers" not in asymmetry.request, (
        "the bearer must stay the harness's VALID one: the Node coordinator "
        "rejects auth before routing, so a bad bearer answers 401 and the 404 "
        "would say nothing about which routes exist"
    )


def test_expected_bodies_carry_only_the_verdict_envelope() -> None:
    """No unnormalized non-deterministic value hides in an asserted body.

    The harness normalizes by KEY NAME, and its UUID/ISO-8601 string scrubbers
    do not touch a non-dashed hexadecimal value — an epoch, a generation token
    or a bare digest in a response would diff literally and flake. Constraining
    every asserted body to the fixed envelope means such a field cannot appear
    without someone adding it here and deciding whether it needs an
    ``ignore_keys`` entry. Today none does, which is why no fixture declares
    one."""
    for fixture in _all_effect_fence_fixtures():
        extra = set(fixture.expected["body"]) - _ENVELOPE_KEYS
        assert not extra, (
            f"{fixture.name}: asserted body carries {sorted(extra)}, outside the "
            f"verdict/error envelope {sorted(_ENVELOPE_KEYS)}. If the route grew "
            f"a field, add it here and say whether its value is deterministic."
        )
        assert not fixture.ignore_keys, (
            f"{fixture.name} declares ignore_keys={sorted(fixture.ignore_keys)}; "
            f"each entry is a hole in the catch surface and the verdict envelope "
            f"needs none. Document the non-deterministic value in the fixture's "
            f"description if you are adding the first one."
        )
        assert not fixture.optional_keys, (
            f"{fixture.name} declares optional_keys; the verdict envelope is "
            f"identical on every backend that implements it"
        )
        body = fixture.expected["body"]
        if body.get("verdict") == "hold":
            assert set(body) == {"verdict", "reason"}, (
                f"{fixture.name}: a non-degraded hold is exactly "
                f"{{verdict, reason}}, got {sorted(body)}"
            )
        elif body.get("verdict") == "proceed":
            assert set(body) == {"verdict"}, (
                f"{fixture.name}: proceed carries NO reason key — its absence is "
                f"how a client tells it from a hold whose reason it does not know"
            )


def test_degraded_arms_are_documented_as_unreachable_here() -> None:
    """Honest limit of this corpus, stated rather than left to inference.

    The two degraded envelopes (``watchdog_timeout``, ``handler_error``) reuse
    ``version_unconfirmed`` and add ``degraded``/``held_by``. Neither is
    drivable from a fixture: one needs the watchdog to abandon the work body,
    the other needs the verdict helper to raise. Both are exercised by the unit
    tests around the route. The reason itself is covered here by fixture 03, so
    the vocabulary stays exhaustively pinned — what is NOT pinned here is the
    two-key degraded suffix, and this test exists so that gap is a written
    decision rather than an oversight."""
    degraded = [
        f for f in _all_effect_fence_fixtures()
        if "degraded" in f.expected["body"] or "held_by" in f.expected["body"]
    ]
    assert degraded == [], (
        "A fixture now drives a degraded arm — good. Rewrite this test to "
        f"assert its shape instead of its absence: {[f.name for f in degraded]}"
    )
    covered = {
        f.expected["body"]["reason"]
        for f in _all_effect_fence_fixtures()
        if f.expected["body"].get("verdict") == "hold"
    }
    assert "version_unconfirmed" in covered, (
        "the reason the degraded arms reuse must stay covered by a normal "
        "fixture, or dropping it here would silently uncover them too"
    )


# ----------------------------------------------------------------------
# U1 / R15 — the harness's own identity capability
# ----------------------------------------------------------------------
#
# KTD7, stated as the two things the harness could not do before this unit:
#
#   CARRY      — every preflight response was discarded and there was no
#                substitution, so no fixture could reach a value the
#                coordinator MINTS at run time.
#   DISTINGUISH— every 8-4-4-4-12 hex value was scrubbed to <UUID>
#                unconditionally, so "attributed to A" and "attributed to B"
#                normalized to the same bytes and BOTH passed.
#
# Both were observed red against the pre-unit harness on the two fixtures
# below before a line of harness code changed. Each test here keeps its own
# control, because a capability test that only exercises the happy path is the
# same decorative assertion in a different costume.

_IDENTITY_FIXTURE_DIR = "harness_identity"
_EXPECTED_IDENTITY_FIXTURE_COUNT = 3

# The two sessions the capability fixtures use. A reads; B writes.
_SESSION_A = "11111111-1111-4111-8111-111111111111"
_SESSION_B = "22222222-2222-4222-8222-222222222222"

# The writer field names the AGENT, not the session: the coordinator stopped
# reversing an agent id back to a session id. Both runtimes derive this the same
# way (uuid5 over the session id), and the hex form is what the stale-read
# summary emits.
_AGENT_A = uuid5(NAMESPACE_URL, f"ccs-agent:claude-session-{_SESSION_A}").hex
_AGENT_B = uuid5(NAMESPACE_URL, f"ccs-agent:claude-session-{_SESSION_B}").hex

# The SAME two agents, in the hyphenated 8-4-4-4-12 form ``/status`` emits
# (``str(session_to_agent_id(...))``). The spelling is the whole reason the
# status fixture exists: the hex form above is invisible to ``_UUID_RE``, so it
# survives the portability default untouched and an attribution asserting it
# would read identically with the opt-in deleted. These are scrubbed to
# ``UUID_SENTINEL`` by the default, on BOTH the key rule and the string-position
# rule, so a fixture asserting them is vacuous unless the opt-in is honoured.
_AGENT_A_UUID = str(uuid5(NAMESPACE_URL, f"ccs-agent:claude-session-{_SESSION_A}"))
_AGENT_B_UUID = str(uuid5(NAMESPACE_URL, f"ccs-agent:claude-session-{_SESSION_B}"))

_CAPTURE_FIXTURE = "harness-captured-token-reaches-the-coordinator"
_IDENTITY_FIXTURE = "harness-stale-read-names-the-writing-session"
_STATUS_IDENTITY_FIXTURE = "harness-status-names-which-session-holds-what"

# Well-formed and never minted: the coordinator splits its refusal by SHAPE
# (43 URL-safe base64 characters is a minted token's shape), so a control using
# a malformed string would be refused for the wrong reason and would say
# nothing about whether the right BYTES arrived. The shape is asserted in the
# test rather than trusted, so an edit to this constant that breaks it is red
# with a readable reason instead of silently flipping the refusal.
_NEVER_MINTED_TOKEN = "wF7nQd2pL0aZxYcV9sKrTbNmJhGfEdCuI4oPl1SqRt8"
_MINTED_TOKEN_LEN = 43
_MINTED_TOKEN_ALPHABET = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)

# Every corpus directory that predates this unit. None of them may acquire an
# identity opt-in by accident: the default is what lets one fixture set run
# against two implementations that mint different ids.
_PRE_EXISTING_DIRS = ("warn_mode", "strict_mode", "session_start", "effect_fence")


def _identity_fixtures() -> list[Fixture]:
    return load_fixtures(_IDENTITY_FIXTURE_DIR)


def _identity_fixture(name: str) -> Fixture:
    by_name = {f.name: f for f in _identity_fixtures()}
    assert name in by_name, f"{name!r} not in {sorted(by_name)}"
    return by_name[name]


_IDENTITY_ROWS = [
    (fixture, backend)
    for fixture in _identity_fixtures()
    for backend in fixture.backends
]


@pytest.mark.parametrize(
    "row",
    _IDENTITY_ROWS,
    ids=[_row_id(r) for r in _IDENTITY_ROWS] if _IDENTITY_ROWS else None,
)
def test_harness_identity_fixture_response_matches_expected(
    row: tuple[Fixture, str],
    tmp_path: Path,
) -> None:
    """The capability fixtures run as ordinary corpus rows.

    Same assertion as the verdict rows above — full body, after normalization —
    so the capability is carried by the same machinery the rest of the corpus
    uses rather than by a bespoke path only these two tests take."""
    fixture, backend = row
    actual_status, actual_body = run_scenario(
        fixture=fixture,
        backend_id=backend,
        workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )
    expected_body = normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
        preserve_identity=fixture.preserve_identity,
    )
    assert actual_status == fixture.expected["status"], (
        f"{fixture.name}[{backend}]: status mismatch — "
        f"expected {fixture.expected['status']}, got {actual_status}\n"
        f"body={actual_body!r}"
    )
    assert actual_body == expected_body, (
        f"{fixture.name}[{backend}]: body mismatch\n"
        f"expected={expected_body!r}\nactual=  {actual_body!r}"
    )


def test_identity_fixture_directory_is_actually_loaded() -> None:
    """Silent failure #1 again, for the new directory.

    A directory no module names loads as ``[]`` and parametrizes to nothing, so
    every test below would report green while asking the coordinator nothing at
    all. The count is pinned against a frozen literal for the same reason the
    verdict directory's is."""
    fixtures = _identity_fixtures()
    assert len(fixtures) == _EXPECTED_IDENTITY_FIXTURE_COUNT, (
        f"Expected exactly {_EXPECTED_IDENTITY_FIXTURE_COUNT} fixtures in "
        f"tests/protocol_corpus/fixtures/{_IDENTITY_FIXTURE_DIR}/, found "
        f"{len(fixtures)}: {[f.name for f in fixtures]}"
    )
    assert {f.name for f in fixtures} == {
        _CAPTURE_FIXTURE,
        _IDENTITY_FIXTURE,
        _STATUS_IDENTITY_FIXTURE,
    }
    assert len(_IDENTITY_ROWS) == len(fixtures)


def test_the_affirmative_answer_is_caused_by_the_captured_bytes(
    tmp_path: Path,
) -> None:
    """CARRY, with the control that makes it mean something.

    The capture fixture answers ``ok: true`` because the bytes the coordinator
    minted at ``/session/begin`` reached ``/session/read``. On its own that
    proves nothing — a route that accepted any token would answer the same. So
    the SAME fixture is re-run with one edit, a well-formed token that was
    never minted, and must answer ``session_not_found``. The affirmative answer
    is therefore attributable to the captured value and to nothing else."""
    fixture = _identity_fixture(_CAPTURE_FIXTURE)

    _, minted_body = run_scenario(
        fixture=fixture, backend_id="python", workspace=tmp_path
    )
    assert minted_body.get("ok") is True, (
        f"the captured token did not reach the coordinator: {minted_body!r}"
    )
    assert minted_body.get("served") == "data_plane_deferred"

    request = copy.deepcopy(fixture.request)
    request["body"]["session_token"] = _NEVER_MINTED_TOKEN
    control = replace(fixture, request=request)

    control_workspace = tmp_path / "control"
    control_workspace.mkdir()
    _, control_body = run_scenario(
        fixture=control, backend_id="python", workspace=control_workspace
    )
    assert len(_NEVER_MINTED_TOKEN) == _MINTED_TOKEN_LEN and set(
        _NEVER_MINTED_TOKEN
    ) <= _MINTED_TOKEN_ALPHABET, (
        "the control token must have a MINTED token's shape, or the refusal "
        "below is about the shape and not about the value"
    )
    assert control_body.get("ok") is False, (
        "the control must be REFUSED, or the fixture's ok:true says nothing "
        f"about which bytes were sent: {control_body!r}"
    )
    assert "served" not in control_body, (
        f"the control must serve nothing: {control_body!r}"
    )
    assert control_body.get("reason") == "session_invalidated", (
        "an in-shape token with no live cut is refused fail-closed as "
        f"session_invalidated: {control_body!r}"
    )
    assert control_body != minted_body


def test_an_unresolved_capture_reference_raises_instead_of_sending_the_literal(
    tmp_path: Path,
) -> None:
    """The fail-closed half of CARRY.

    If a ``${...}`` reference whose capture is missing were left on the wire as
    a literal, the coordinator would answer a rejection — and a fixture whose
    expectation happened to BE that rejection would pass while proving nothing
    about substitution. So an unresolved reference is an error at both seams:
    ``build_fixture`` refuses the fixture at load time, and ``run_scenario``
    refuses it at run time even when the dataclass is built around the loader.
    Both are asserted, because only the second one is reachable from a fixture
    mutated in memory — which is exactly how the rest of this module works."""
    fixture = _identity_fixture(_CAPTURE_FIXTURE)
    setup = copy.deepcopy(fixture.setup)
    del setup["preflight_requests"][1]["capture"]
    orphaned = replace(fixture, setup=setup)

    with pytest.raises(FixtureSubstitutionError) as run_exc:
        run_scenario(fixture=orphaned, backend_id="python", workspace=tmp_path)
    assert "minted_session_token" in str(run_exc.value), (
        f"the error must name the unresolved reference: {run_exc.value}"
    )

    raw = {
        "name": orphaned.name,
        "setup": setup,
        "request": fixture.request,
        "expected": fixture.expected,
        "ignore_keys": sorted(fixture.ignore_keys),
        "backends": list(fixture.backends),
    }
    with pytest.raises(FixtureContractError) as load_exc:
        build_fixture(raw, fixture.path)
    assert "minted_session_token" in str(load_exc.value)


def test_a_capture_of_an_absent_field_raises_rather_than_capturing_none(
    tmp_path: Path,
) -> None:
    """A capture that silently yields ``None`` is the same bug one level up.

    ``None`` would substitute as a JSON null, the coordinator would refuse it,
    and the fixture would again be asserting a rejection it did not intend. The
    capture names a field the mint response does not carry, and the harness
    must say so."""
    fixture = _identity_fixture(_CAPTURE_FIXTURE)
    setup = copy.deepcopy(fixture.setup)
    setup["preflight_requests"][1]["capture"] = {
        "minted_session_token": "no_such_field"
    }
    broken = replace(fixture, setup=setup)

    with pytest.raises(FixtureContractError) as exc:
        run_scenario(fixture=broken, backend_id="python", workspace=tmp_path)
    assert "no_such_field" in str(exc.value)


def test_identity_preserving_fixture_fails_on_a_different_identity(
    tmp_path: Path,
) -> None:
    """DISTINGUISH, asserted in both directions.

    The stale-read response names session B as the writer. With the opt-in
    declared, the actual body carries B's bytes verbatim — not ``<UUID>`` — so
    pinning B matches and pinning A does not. The second assertion is the
    load-bearing one: before this unit BOTH pins passed, which is what made an
    attribution fixture decorative."""
    fixture = _identity_fixture(_IDENTITY_FIXTURE)
    _, actual = run_scenario(
        fixture=fixture, backend_id="python", workspace=tmp_path
    )

    assert actual["summary"]["last_writer_session_id"] == _AGENT_B, (
        "the declared key must survive normalization verbatim, got "
        f"{actual['summary']['last_writer_session_id']!r}"
    )

    right = normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        preserve_identity=fixture.preserve_identity,
    )
    assert actual == right, f"expected={right!r}\nactual=  {actual!r}"

    wrong_body = copy.deepcopy(fixture.expected["body"])
    wrong_body["summary"]["last_writer_session_id"] = _AGENT_A
    wrong = normalize_response(
        wrong_body,
        ignore_keys=fixture.ignore_keys,
        preserve_identity=fixture.preserve_identity,
    )
    assert actual != wrong, (
        "a fixture naming the WRONG session still matched — the opt-in is not "
        "preserving the identity and the assertion is decorative"
    )


def test_the_writer_field_is_portable_by_derivation_not_by_scrubbing(
    tmp_path: Path,
) -> None:
    """Why this field no longer needs the scrub, and what still does.

    The writer is named by agent id in hex, which ``_UUID_RE`` does not match,
    so the portability default leaves it alone. That is not a hole: both
    runtimes derive the value the same way from the session id the fixture
    itself supplies, so it is deterministic rather than backend noise, and the
    A-pin and the B-pin differ for a real reason. The opt-in still matters
    because a genuinely UUID-shaped identity IS collapsed — asserted below, so
    a change to the default cannot pass unnoticed."""
    fixture = _identity_fixture(_IDENTITY_FIXTURE)
    default = replace(fixture, preserve_identity=frozenset())
    _, actual = run_scenario(
        fixture=default, backend_id="python", workspace=tmp_path
    )
    assert actual["summary"]["last_writer_session_id"] == _AGENT_B, (
        "the writer is deterministic hex and must survive the default "
        f"untouched, got {actual['summary']['last_writer_session_id']!r}"
    )

    pins = []
    for agent in (_AGENT_A, _AGENT_B):
        body = copy.deepcopy(fixture.expected["body"])
        body["summary"]["last_writer_session_id"] = agent
        pins.append(normalize_response(body, ignore_keys=fixture.ignore_keys))
    assert pins[0] != pins[1], (
        "two agents must stay distinguishable under the default — if they "
        "collapse, this field went back to being scrubbed and every "
        "attribution fixture asserting it is decorative again"
    )
    assert pins[1] == actual, f"expected={pins[1]!r}\nactual=  {actual!r}"

    # The scrub itself is intact for a UUID-shaped identity, which is what the
    # opt-in exists to override.
    shaped = normalize_response(
        {"session_id": _SESSION_A}, ignore_keys=frozenset()
    )
    assert shaped["session_id"] == UUID_SENTINEL, (
        "the portability default must still collapse a UUID-shaped identity"
    )


def test_status_identity_fixture_distinguishes_which_session_holds_what(
    tmp_path: Path,
) -> None:
    """DISTINGUISH over a field the default actually collapses.

    This is the test the opt-in is FOR. The fixture above asserts an attribution
    whose value is 32-char hex, which ``_UUID_RE`` (8-4-4-4-12 only) never
    matched, so that fixture reads identically with ``preserve_identity``
    deleted and cannot observe the opt-in at all. ``sessions[].agent_id`` on
    ``/status`` is the hyphenated form, which the default DOES collapse — so a
    two-holder ``/status`` body is where "A holds docs/plan.md, B holds
    docs/spec.md" is a claim rather than decoration.

    Four steps, in the order that makes a failure readable: the rows are real,
    the identities survive verbatim, the RIGHT pairing matches, the SWAPPED
    pairing does not — and last, the reason all of that is load-bearing, namely
    that under the portability default the right and swapped pairings are the
    same bytes."""
    fixture = _identity_fixture(_STATUS_IDENTITY_FIXTURE)
    _, actual = run_scenario(
        fixture=fixture, backend_id="python", workspace=tmp_path
    )

    # Control: the case this test claims to inspect is actually present. An
    # empty or single-row sessions list would satisfy a swap assertion while
    # observing nothing.
    rows = actual["sessions"]
    assert [r["states"] for r in rows] == [
        {"docs/plan.md": "SHARED"},
        {"docs/spec.md": "EXCLUSIVE"},
    ], f"the two-holder workspace did not materialize: {rows!r}"

    assert [r["agent_id"] for r in rows] == [_AGENT_A_UUID, _AGENT_B_UUID], (
        "the declared key must survive normalization verbatim; "
        f"got {[r['agent_id'] for r in rows]!r}"
    )

    right = normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        preserve_identity=fixture.preserve_identity,
    )
    assert actual == right, f"expected={right!r}\nactual=  {actual!r}"

    swapped_body = copy.deepcopy(fixture.expected["body"])
    swapped_body["sessions"][0]["agent_id"] = _AGENT_B_UUID
    swapped_body["sessions"][1]["agent_id"] = _AGENT_A_UUID
    swapped = normalize_response(
        swapped_body,
        ignore_keys=fixture.ignore_keys,
        preserve_identity=fixture.preserve_identity,
    )
    assert actual != swapped, (
        "a fixture crediting each artifact to the OTHER session still matched "
        "— the opt-in is not preserving the identity and the assertion is "
        "decorative"
    )

    # Why the swap above is a real distinction and not a tautology: strip the
    # opt-in and the two pairings are byte-identical, because both rows'
    # agent_id collapses to the sentinel. Passed explicitly rather than read
    # from the fixture, so this arm keeps saying what the DEFAULT does even if
    # the opt-in stops working.
    assert normalize_response(
        fixture.expected["body"], ignore_keys=fixture.ignore_keys
    ) == normalize_response(swapped_body, ignore_keys=fixture.ignore_keys), (
        "without the opt-in the right and swapped pairings must be the same "
        "bytes — if they already differ, this fixture is not exercising the "
        "opt-in and the corpus is back to shipping it unobserved"
    )

    # Both default rules reach this value, so neither alone explains the pass.
    assert (
        normalize_response({"agent_id": _AGENT_A_UUID})["agent_id"]
        == UUID_SENTINEL
    ), "the _UUID_KEYS rule must still collapse agent_id"
    assert (
        normalize_response({"note": f"held by {_AGENT_A_UUID}"})["note"]
        == f"held by {UUID_SENTINEL}"
    ), "the string-position rule must still collapse a hyphenated identity"


def test_no_pre_existing_fixture_declares_an_identity_opt_in() -> None:
    """The default stays the default everywhere it already applied.

    Each pre-existing corpus module calls ``normalize_response`` WITHOUT
    ``preserve_identity`` — it cannot pass a field it does not know about — so a
    declaration appearing in one of those directories would preserve the actual
    side while the expected side stayed scrubbed. That mismatch fails red
    rather than green, but it fails for an unreadable reason, so the
    declaration is pinned out of those directories here where the reason can be
    written down."""
    for mode in _PRE_EXISTING_DIRS:
        fixtures = load_fixtures(mode)
        assert fixtures, f"{mode}/ loaded empty — the directory name moved"
        offenders = [f.name for f in fixtures if f.preserve_identity]
        assert not offenders, (
            f"{mode}/ fixtures declare preserve_identity: {offenders}. Only a "
            f"module that passes preserve_identity into normalize_response can "
            f"honour it; the pre-existing modules do not."
        )


def test_a_decorative_identity_declaration_is_refused_at_load() -> None:
    """A declaration naming a key the fixture never asserts proves nothing.

    Four ways to write one that reads as protection and is not: naming a key
    absent from the expected body, naming a key whose asserted value is already
    the ``<UUID>`` sentinel (preserving a sentinel preserves nothing), naming a
    key that ``ignore_keys`` throws away first, and naming a principal field
    (R5 outranks the opt-in). All four are refused at load, so a decorative
    opt-in cannot reach a run.

    Each arm matches the phrase unique to ITS rule rather than the key name.
    That is not fussiness: with a shared substring, disabling the principal
    rule left the absent-key rule to raise the same-looking error and the test
    stayed green — an exception type cannot say which mechanism produced it."""
    fixture = _identity_fixture(_IDENTITY_FIXTURE)

    def _raw(**over: object) -> dict:
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

    # Sanity control: the real fixture's own declaration is accepted.
    assert build_fixture(_raw(), fixture.path).preserve_identity == frozenset(
        {"last_writer_session_id"}
    )

    with pytest.raises(FixtureContractError, match="does not carry"):
        build_fixture(_raw(preserve_identity=["absent_key"]), fixture.path)

    # A principal key is refused BEFORE the body is even consulted: R5 outranks
    # the opt-in, so "the body does not carry it" would be the wrong reason.
    with pytest.raises(FixtureContractError, match="R5 outranks"):
        build_fixture(_raw(preserve_identity=["principal"]), fixture.path)

    sentinel_expected = copy.deepcopy(fixture.expected)
    sentinel_expected["body"]["summary"]["last_writer_session_id"] = UUID_SENTINEL
    with pytest.raises(FixtureContractError, match="Preserving a sentinel"):
        build_fixture(_raw(expected=sentinel_expected), fixture.path)

    with pytest.raises(FixtureContractError, match="ignore_keys wins"):
        build_fixture(
            _raw(ignore_keys=["last_writer_session_id"]), fixture.path
        )


def test_a_principal_in_an_expected_body_is_rejected_rather_than_compared() -> None:
    """R5, enforced instead of described.

    A principal appears on exactly one response — the mint response that issues
    it — and never in a fixture's expected body. The failure mode this refuses
    is not a leak of secret bytes so much as a VACUOUS comparison: a principal
    normalizes to its own sentinel, so a hard-coded principal in an expected
    body would be replaced by that sentinel and would compare equal to any
    other principal, which is the decorative assertion again. The harness
    therefore refuses the fixture rather than quietly normalizing it, and
    refuses a ``${...}`` reference in an expected body too, since substitution
    is the only other way a minted value could get there."""
    fixture = _identity_fixture(_CAPTURE_FIXTURE)

    def _raw(expected_body: dict) -> dict:
        return {
            "name": "principal-probe",
            "setup": copy.deepcopy(fixture.setup),
            "request": copy.deepcopy(fixture.request),
            "expected": {"status": 200, "body": expected_body},
            "backends": ["python"],
        }

    with pytest.raises(FixtureContractError, match="expected body carries a principal"):
        build_fixture(_raw({"ok": True, "principal": "agent-7"}), fixture.path)

    with pytest.raises(FixtureContractError, match="expected body carries a principal"):
        build_fixture(
            _raw({"ok": True, "grant": {"caller_principal": "agent-7"}}),
            fixture.path,
        )

    with pytest.raises(FixtureContractError, match="minted_session_token"):
        build_fixture(
            _raw({"ok": True, "echo": "${minted_session_token}"}), fixture.path
        )

    # The sentinel itself is the one admissible form: it asserts the FIELD is
    # present without asserting whose principal it is.
    accepted = build_fixture(
        _raw({"ok": True, "principal": PRINCIPAL_SENTINEL}), fixture.path
    )
    assert accepted.expected["body"]["principal"] == PRINCIPAL_SENTINEL


def test_a_principal_normalizes_to_its_own_sentinel_not_the_uuid_one() -> None:
    """The sentinel has to be DISTINCT, or R5 cannot be enforced by name.

    If a principal collapsed into ``<UUID>`` the load-time refusal above could
    not tell a principal field from any other identifier, and the opt-in could
    un-hide one. So: principals get their own sentinel, and the opt-in does not
    outrank R5 — a key named in both still normalizes to ``<PRINCIPAL>``."""
    minted = "8a1f0c7e-2b34-4d56-9e78-0f1a2b3c4d5e"
    assert PRINCIPAL_SENTINEL != UUID_SENTINEL

    assert normalize_response({"principal": minted}) == {
        "principal": PRINCIPAL_SENTINEL
    }
    assert normalize_response({"grant": {"caller_principal": minted}}) == {
        "grant": {"caller_principal": PRINCIPAL_SENTINEL}
    }
    assert normalize_response(
        {"principal": minted}, preserve_identity=frozenset({"principal"})
    ) == {"principal": PRINCIPAL_SENTINEL}, (
        "R5 outranks the identity opt-in — a fixture must not be able to "
        "preserve a principal's bytes by declaring the key"
    )


def test_a_capture_reaches_a_later_preflight_too(tmp_path: Path) -> None:
    """Substitution is not only a main-request feature.

    A setup chain is the reason captures exist at all — mint, then USE, then
    assert something about the state that use produced — so a capture that
    reached only the main request would carry half the capability. The third
    preflight here reads with the minted token and captures a field that exists
    ONLY on the affirmative answer, so an unsubstituted preflight draws the
    refusal, the field is absent, and the run raises instead of arriving at a
    comparison. The assertion is that the whole scenario completes and still
    matches: a green run here means the chained token really was substituted."""
    fixture = _identity_fixture(_CAPTURE_FIXTURE)
    setup = copy.deepcopy(fixture.setup)
    setup["preflight_requests"].append({
        "method": "POST",
        "path": "/session/read",
        "body": {
            "session_id": _SESSION_A,
            "session_token": "${minted_session_token}",
            "path": "docs/plan.md",
        },
        "capture": {"served_branch": "served"},
    })
    chained = replace(fixture, setup=setup)

    _, body = run_scenario(
        fixture=chained, backend_id="python", workspace=tmp_path
    )
    assert body == normalize_response(
        fixture.expected["body"], ignore_keys=fixture.ignore_keys
    )


def test_a_capture_that_reads_null_raises(tmp_path: Path) -> None:
    """A field that EXISTS and is null is the absent-field bug one step on.

    ``None`` would substitute as a JSON null, the coordinator would refuse it,
    and the fixture would assert a rejection nobody meant to drive. The
    deferred read below really does answer ``content_hash: null``, so this is
    the live shape rather than a constructed one."""
    fixture = _identity_fixture(_CAPTURE_FIXTURE)
    setup = copy.deepcopy(fixture.setup)
    setup["preflight_requests"].append({
        "method": "POST",
        "path": "/session/read",
        "body": {
            "session_id": _SESSION_A,
            "session_token": "${minted_session_token}",
            "path": "docs/plan.md",
        },
        "capture": {"nullish": "content_hash"},
    })
    nulled = replace(fixture, setup=setup)

    with pytest.raises(FixtureContractError, match="null"):
        run_scenario(fixture=nulled, backend_id="python", workspace=tmp_path)


def test_a_re_bound_capture_name_is_refused(tmp_path: Path) -> None:
    """One name, one mint.

    If a second capture could overwrite a name, a later request would silently
    read the wrong mint — and the request would still be well-formed, so the
    only symptom would be an answer someone then writes an expectation around.
    Refused at load, and again at run time for the in-memory case the corpus
    tests themselves construct."""
    fixture = _identity_fixture(_CAPTURE_FIXTURE)
    setup = copy.deepcopy(fixture.setup)
    setup["preflight_requests"].append({
        "method": "POST",
        "path": "/session/begin",
        "body": {"session_id": _SESSION_A, "read_set": ["docs/plan.md"]},
        "capture": {"minted_session_token": "session_token"},
    })
    rebound = replace(fixture, setup=setup)

    with pytest.raises(FixtureContractError, match="minted_session_token"):
        run_scenario(fixture=rebound, backend_id="python", workspace=tmp_path)

    raw = {
        "name": fixture.name,
        "setup": setup,
        "request": copy.deepcopy(fixture.request),
        "expected": copy.deepcopy(fixture.expected),
        "ignore_keys": sorted(fixture.ignore_keys),
        "backends": list(fixture.backends),
    }
    with pytest.raises(FixtureContractError, match="minted_session_token"):
        build_fixture(raw, fixture.path)
