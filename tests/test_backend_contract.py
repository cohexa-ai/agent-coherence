# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Drift guards for the generalized backend atomic-boundary contract.

:mod:`ccs.coordinator.backend_contract` is pure documented vocabulary (enums,
frozen dataclasses, string constants) formalizing what a networked registry
backend must provide to re-home the coordinator's atomic boundary. These tests
are the DRIFT GUARDS the plan calls for:

- the member-classification map covers EXACTLY the 71 ``RegistryBase`` +
  ``SqliteExtended`` members — no more, no fewer — so a Protocol member added or
  removed in ``registry_protocol.py`` without a matching contract update FAILS
  CI (bidirectional guard);
- the ``coordinator_epoch`` PROPERTY is present in the map (property-omission
  teeth — ``@runtime_checkable`` cannot see properties, per the #133 lesson);
- every statelessness item names a disposition and ``coordinator_epoch``'s is
  never ``SAFELY_LOST`` (R12);
- a classification value outside the enum is unrepresentable (typed);
- the tier enum has exactly ``TIER_1`` and ``TIER_2``.

The expected 71-member set is FROZEN here (imported from the parity test's own
frozen name-sets), NOT derived from the Protocol at runtime — so a silent
Protocol edit cannot move the goalposts this test guards (the same discipline
``tests/test_registry_protocol_parity.py`` uses).
"""

from __future__ import annotations

import re

from ccs.coordinator.backend_contract import (
    MEMBER_CLASSIFICATION,
    R9_ATOMIC_BOUNDARY,
    R12_EPOCH,
    R13_CONTENT_POSTURE,
    R14_CREDENTIAL,
    R15_BACKEND_IDENTITY,
    R18_LIVENESS_SOURCE,
    STATELESSNESS_INVENTORY,
    TIER_DECLARATIONS,
    Disposition,
    MemberClass,
    Tier,
)

# The 71-member expected surface, imported from the parity test's FROZEN
# name-sets (48 base + 13 extended + 9 detection methods + 1 base property). Reusing
# those frozensets means this contract and the Protocol parity share ONE
# source of truth for the surface: if either the parity test or the Protocol
# changes the surface, the two drift guards fire together.
from tests.test_registry_protocol_parity import (  # noqa: E402
    BASE_METHODS,
    BASE_PROPERTIES,
    DETECTION_METHODS,
    EXTENDED_ONLY_METHODS,
)

EXPECTED_MEMBERS = (
    BASE_METHODS | EXTENDED_ONLY_METHODS | DETECTION_METHODS | BASE_PROPERTIES
)


# ---------------------------------------------------------------------------
# Member classification — exactly the 71 Protocol members, no drift
# ---------------------------------------------------------------------------


def test_member_map_covers_exactly_the_protocol_surface() -> None:
    """The classification map keys equal the 71-member Protocol surface EXACTLY
    — no more, no fewer. A member added to (or removed from) ``registry_protocol.py``
    without a matching contract update fails HERE (bidirectional drift guard)."""
    assert set(MEMBER_CLASSIFICATION) == EXPECTED_MEMBERS


def test_member_map_has_exactly_71_members() -> None:
    """Pin the count explicitly: 48 base methods (SB-18 ``commit_all``, the
    WV Unit-2 checkpoint surface — 8 members — then the effect-gate pair read
    ``get_artifact_and_generation``, then the SB-10 comparand read
    ``last_observed_version_for``, then the caller-principal pair
    ``bind_caller_principal`` / ``get_caller_principal``, then #191's
    ``claim_checkpoint_registration`` added) + 13 extended
    methods + 9 foreign-write detection methods on their own Protocol + 1
    base property = 71. Guards against a same-size add+remove that would slip
    past the set-equality check on cardinality alone."""
    assert len(MEMBER_CLASSIFICATION) == 71
    assert len(EXPECTED_MEMBERS) == 71


def test_coordinator_epoch_property_is_in_the_map() -> None:
    """The ``coordinator_epoch`` PROPERTY is classified (property-omission teeth,
    the #133 lesson: ``@runtime_checkable`` cannot see properties, so a
    method-only map would silently drop it — the exact gap that lesson exists to
    prevent)."""
    assert "coordinator_epoch" in MEMBER_CLASSIFICATION
    assert "coordinator_epoch" in BASE_PROPERTIES


def test_every_member_has_a_typed_classification() -> None:
    """Every mapped member carries a :class:`MemberClass` enum value — a
    classification outside the enum is unrepresentable (typed, not a bare
    string)."""
    for name, contract in MEMBER_CLASSIFICATION.items():
        assert isinstance(contract.member_class, MemberClass), name
        assert contract.name == name
        assert contract.surface in {"base", "sqlite_extended", "detection"}
        assert contract.rationale  # non-empty rationale authored from service.py


def test_member_surface_matches_the_protocol_split() -> None:
    """Each member's declared ``surface`` matches which Protocol frozenset it
    belongs to: base methods plus the base property are "base", extended-only
    methods are "sqlite_extended", and the detection instrument is its own
    "detection" surface — a backend may implement that one independently, or
    not at all, without losing the coordination surface."""
    base_surface = BASE_METHODS | BASE_PROPERTIES
    for name, contract in MEMBER_CLASSIFICATION.items():
        if name in base_surface:
            assert contract.surface == "base", name
        elif name in DETECTION_METHODS:
            assert contract.surface == "detection", name
        else:
            assert contract.surface == "sqlite_extended", name


def test_the_atomic_class_boundary_members_are_classified_atomic() -> None:
    """The members the service touches INSIDE its atomic mutation paths (authored
    from the ``service.py`` call sites) are ATOMIC_CLASS. This pins the core
    classification decision so a later edit that silently downgrades one to
    READ_ONLY/INDEPENDENT fails."""
    expected_atomic = {
        "commit_cas",
        "commit_all",
        "set_artifact_and_content",
        "set_agent_state",
        "set_agent_transient",
        "clear_agent_transient",
        "capture_version_vector",
        "abort_guard",
        "get_state_map",
    }
    actual_atomic = {
        name
        for name, contract in MEMBER_CLASSIFICATION.items()
        if contract.member_class is MemberClass.ATOMIC_CLASS
    }
    assert actual_atomic == expected_atomic


def test_classification_enum_has_exactly_three_classes() -> None:
    """The member-class taxonomy is exactly {ATOMIC_CLASS, INDEPENDENT,
    READ_ONLY} — a value outside it cannot be represented."""
    assert {m.name for m in MemberClass} == {
        "ATOMIC_CLASS",
        "INDEPENDENT",
        "READ_ONLY",
    }


# ---------------------------------------------------------------------------
# R11 statelessness inventory + R12 epoch never-safely-lost
# ---------------------------------------------------------------------------


def test_every_state_item_names_a_typed_disposition() -> None:
    """Every statelessness item names a :class:`Disposition` — nothing implicit
    (R11: every listed state has a stated re-home or safe-loss disposition)."""
    assert STATELESSNESS_INVENTORY  # non-empty
    for item in STATELESSNESS_INVENTORY:
        assert isinstance(item.disposition, Disposition), item.name
        assert item.consequence  # the stated consequence if it does not re-home


def test_coordinator_epoch_is_must_rehome_and_never_safely_lost() -> None:
    """``coordinator_epoch``'s disposition is MUST_REHOME and — the R12 assertion
    — is NEVER ``SAFELY_LOST`` (losing it invalidates every client-carried token
    at once)."""
    epoch_items = [
        item for item in STATELESSNESS_INVENTORY if item.name == "coordinator_epoch"
    ]
    assert len(epoch_items) == 1, "coordinator_epoch must appear exactly once"
    (epoch_item,) = epoch_items
    assert epoch_item.disposition is Disposition.MUST_REHOME
    assert epoch_item.disposition is not Disposition.SAFELY_LOST


def test_disposition_enum_has_exactly_three_values() -> None:
    assert {d.name for d in Disposition} == {
        "MUST_REHOME",
        "DERIVABLE",
        "SAFELY_LOST",
    }


# ---------------------------------------------------------------------------
# R10 tiers
# ---------------------------------------------------------------------------


def test_tier_enum_has_exactly_tier_1_and_tier_2() -> None:
    """Exactly two tiers (R10) — TIER_1 full-tuple and TIER_2 lease-decomposed."""
    assert {t.name for t in Tier} == {"TIER_1", "TIER_2"}


def test_tier_declarations_cover_both_tiers() -> None:
    assert set(TIER_DECLARATIONS) == {Tier.TIER_1, Tier.TIER_2}
    assert TIER_DECLARATIONS[Tier.TIER_1].ha_qualifies is True
    assert TIER_DECLARATIONS[Tier.TIER_2].ha_qualifies is False


# ---------------------------------------------------------------------------
# R9 atomic boundary — fence admit-on-absent reproduced exactly
# ---------------------------------------------------------------------------


def test_r9_boundary_names_the_three_tuple_legs() -> None:
    """The boundary names all three legs — version-CAS + grant arbitration +
    fence — not the version compare alone."""
    assert len(R9_ATOMIC_BOUNDARY.tuple_elements) == 3
    joined = " ".join(R9_ATOMIC_BOUNDARY.tuple_elements).lower()
    assert "version-cas" in joined
    assert "grant arbitration" in joined
    assert "read-generation fence" in joined


def test_r9_fence_admit_on_absent_is_stated_exactly() -> None:
    """The admit-on-absent asymmetry is reproduced EXACTLY (the fence-parity
    lesson — it has drifted once before): an ABSENT read_generation is ADMITTED
    (version-CAS arbitrates); only a PRESENT-and-superseded one is REJECTED."""
    text = R9_ATOMIC_BOUNDARY.fence_admit_on_absent
    assert "ABSENT read_generation is ADMITTED" in text
    assert "version-CAS arbitrates" in text
    assert "superseded" in text and "REJECTED" in text
    # The load-bearing `is not None` predicate is named (not treated as defensive).
    assert "is not None" in text


def test_r9_reference_semantics_include_the_same_lock_sweep() -> None:
    """Reference semantics = the single-process serialization AS A WHOLE — atomic
    mutations PLUS the same-lock sweep — and liveness eviction is stated to be a
    separate same-lock sweep, NOT inside commit_cas's transaction."""
    assert "as a whole" in R9_ATOMIC_BOUNDARY.reference_semantics.lower()
    assert "sweep" in R9_ATOMIC_BOUNDARY.reference_semantics.lower()
    note = R9_ATOMIC_BOUNDARY.liveness_eviction_note.lower()
    assert "separate" in note and "sweep" in note
    assert "not read inside commit_cas" in note


# ---------------------------------------------------------------------------
# R18 / R12 / R13 / R14 / R15 obligation records exist and are honest
# ---------------------------------------------------------------------------


def test_r18_liveness_source_names_shipped_source_as_non_ha() -> None:
    """R18: the shipped source is caller-supplied logical ticks under a single
    coordinator; it does NOT survive N coordinators."""
    assert R18_LIVENESS_SOURCE.conforming_sources
    assert "caller-supplied" in R18_LIVENESS_SOURCE.shipped_source.lower()
    assert "single coordinator" in R18_LIVENESS_SOURCE.shipped_source.lower()


def test_r12_epoch_backend_is_monotonic_int_local_stays_uuid() -> None:
    """R12: backend epoch is a monotonic-increasing integer; the shipped local
    epoch stays an opaque uuid4 string; the migration is deferred."""
    assert "monotonic" in R12_EPOCH.backend_contract.lower()
    assert "uuid4" in R12_EPOCH.shipped_local.lower()
    assert "deferred" in R12_EPOCH.migration_scope.lower()


def test_r13_content_posture_is_hash_only_with_declared_retention() -> None:
    """R13: hash-only baseline; retention is a declared opt-in that widens the
    disclosure surface."""
    assert "hash-only" in R13_CONTENT_POSTURE.baseline.lower()
    assert "retain_versions=true" in R13_CONTENT_POSTURE.retention_capability.lower()
    assert "widen" in R13_CONTENT_POSTURE.disclosure_note.lower()


def test_r14_r15_are_documented_obligations_only() -> None:
    """R14/R15 are documented obligations with typed placeholders — the module
    carries their vocabulary without any networked/connect code."""
    assert R14_CREDENTIAL.requirement_id == "R14"
    assert "0600" in R14_CREDENTIAL.obligation
    assert "O_NOFOLLOW" in R14_CREDENTIAL.discipline
    assert R15_BACKEND_IDENTITY.requirement_id == "R15"
    assert "fingerprint" in R15_BACKEND_IDENTITY.obligation.lower()
    assert "fail-closed" in R15_BACKEND_IDENTITY.obligation.lower()


# ---------------------------------------------------------------------------
# Non-goal (R2 trip-wire) + no networked code — module hygiene
# ---------------------------------------------------------------------------


def test_module_states_the_never_own_the_store_non_goal() -> None:
    """R2 trip-wire: the module docstring states the never-own-the-durable-store
    non-goal ("operated infrastructure we depend on, never a store we ship")."""
    import ccs.coordinator.backend_contract as mod

    assert mod.__doc__ is not None
    doc = mod.__doc__.lower()
    assert "never" in doc and "store we ship" in doc


# A surface size written as prose. The hyphenated form is the idiom both files
# use for "a surface of N members"; the spaced form also appears for a group
# that is NOT the surface ("the WV Unit-2 checkpoint surface - 8 members"), so
# the two are matched separately rather than with one looser pattern.
_HYPHENATED_CLAIM = re.compile(r"(\d+)-member\b")
_SPACED_CLAIM = re.compile(r"(\d+) members\b")
# The breakdown that explains a total: "(45 methods + 1 property)", "(+13
# methods)". These are what a reader consults to learn which Protocol owns what.
_BREAKDOWN = re.compile(r"(\d+) (?:methods|method|properties|property)\b")


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def test_every_member_count_written_in_prose_matches_the_frozen_sets() -> None:
    """Prose counts drift because nothing reads them. This reads all of them.

    The surface size has one source of truth, the frozen name sets above, and
    it is restated in prose in both files. Nothing checked any of those
    restatements, so a wrong one was invisible to the suite and to the linter
    -- which is how the surface sentence came to claim the all-three-Protocol
    total for a two-Protocol surface until a human noticed (PR #204).

    Deliberately a SWEEP rather than one assertion per known sentence. The
    first draft of this guard was per-sentence, and it silently missed a third
    restatement and all four breakdown figures; a restatement added tomorrow would
    have been missed the same way. Matching every occurrence means a new one is
    covered by construction, and the ordered comparison below means an added or
    deleted restatement fails here rather than passing unnoticed.

    Two counts are in play and they are not interchangeable, which is the trap
    that turned a stale figure into a wrong one:

    * the SURFACE is ``RegistryBase`` + ``SqliteExtended``, because the
      detection members deliberately sit on their own ``ForeignWriteDetection``;
    * the CLASSIFICATION covers all three.

    Reword past these patterns and this fails rather than quietly ceasing to
    guard -- an unexpected match list is the failure being prevented, so make
    the update on purpose.
    """
    import ccs.coordinator.backend_contract as mod

    surface = len(BASE_METHODS | EXTENDED_ONLY_METHODS | BASE_PROPERTIES)
    everything = len(EXPECTED_MEMBERS)

    assert mod.__file__ is not None
    module_src = _read(mod.__file__)

    # Ordered, exhaustive: an added or removed restatement changes the list.
    assert [int(n) for n in _HYPHENATED_CLAIM.findall(module_src)] == [
        surface,  # the module docstring's SURFACE sentence
        everything,  # MEMBER_CLASSIFICATION's own docstring
    ], "a hyphenated member count in backend_contract.py is wrong or unaccounted for"
    assert [int(n) for n in _SPACED_CLAIM.findall(module_src)] == [
        everything  # the classification comment's total
    ], "a spaced member count in backend_contract.py is wrong or unaccounted for"

    # The breakdown that must sum to that total. Guarding only the total lets a
    # member move between Protocols without anything noticing: the sum is
    # unchanged and the explanation is now wrong.
    assert [int(n) for n in _BREAKDOWN.findall(module_src)] == [
        len(BASE_METHODS),
        len(BASE_PROPERTIES),
        len(EXTENDED_ONLY_METHODS),
        len(DETECTION_METHODS),
    ], "the per-Protocol breakdown in backend_contract.py no longer sums as written"


def test_this_suites_own_prose_states_the_real_member_count() -> None:
    """The guard above reads the module; this file restates the count too.

    Three times, in its header, above the frozen set, and in a test docstring --
    and the file that pins the size is the last place a reader expects to find
    a stale one. Unordered, because every hyphenated claim here means the same
    whole surface, so a new correct mention is not a failure.
    """
    claims = {int(n) for n in _HYPHENATED_CLAIM.findall(_read(__file__))}
    assert claims == {len(EXPECTED_MEMBERS)}, (
        f"this file's own prose claims {sorted(claims)} members"
    )


def test_module_imports_no_networked_code() -> None:
    """The module is pure vocabulary — it imports no I/O, networking, or higher
    layers. Guards against a networked dependency creeping in."""
    import ccs.coordinator.backend_contract as mod

    source = mod.__file__
    assert source is not None
    with open(source, encoding="utf-8") as fh:
        text = fh.read()
    for forbidden in ("import socket", "import ssl", "import urllib", "import http", "requests"):
        assert forbidden not in text, forbidden


def test_r9_fence_capture_survives_every_peer_fetch() -> None:
    """The captured generation is the zombie's side of the fence check, and it
    is written only by the zombie's OWN claim: a peer's fetch that downgrades
    it to SHARED captures nothing, and a peer's fetch never rewrites an
    already-SHARED holder — so a superseded value survives, and the refusal
    stays sticky, until the zombie's own re-read or re-acquire. Two pins keep
    the clause from drifting back to the trigger-only rule."""
    text = R9_ATOMIC_BOUNDARY.fence_admit_on_absent
    assert "captures nothing" in text
    assert "never rewrites an already-SHARED holder" in text


def test_set_agent_state_member_contract_states_the_capture_rule() -> None:
    """The ``set_agent_state`` member contract carries the capture rule the
    registries implement: capture rides the agent's OWN claim-establishing
    transition, and a transition OUT of M/E captures nothing."""
    rationale = MEMBER_CLASSIFICATION["set_agent_state"].rationale
    assert "OWN claim" in rationale
    assert "captures nothing" in rationale
