# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The shared fold table and the fold functions built on it.

The vectors in the table file are the ones the plugin's Node coordinator runs
too, so every one of them is driven here under every profile. The pairs
measured on macOS record what a case-insensitive APFS volume (caseless) and a
case-sensitive one (nfd) actually did with each spelling pair. The table is
generated in the plugin repository; this copy must stay byte-identical, which
the pinned hash below holds.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from importlib import resources
from typing import Any, get_args

import pytest

from ccs.core.path_fold import (
    FOLD_LENGTH_LIMIT,
    FOLD_TABLE_UNICODE_VERSION,
    FoldProfile,
    exceeds_fold_limit,
    fold_key,
    policy_fold,
    policy_fold_text,
    tidy_path,
)

#: The SHA-256 of the table section, the same constant the plugin's test pins.
#: Changing the table means regenerating it in the plugin and updating this
#: constant in both repositories.
PINNED_TABLE_SHA256 = "c259d5c36c7e6dc56347b24b7cc263a82acd4c460754e186a565a3da68d39fe8"

FOLD_DATA = resources.files("ccs.core").joinpath("fold_data")
TABLE_SOURCE = FOLD_DATA.joinpath("fold_table.json").read_text(encoding="utf-8")
FOLD_DOCUMENT: dict[str, Any] = json.loads(TABLE_SOURCE)
VECTORS: dict[str, list[dict[str, Any]]] = FOLD_DOCUMENT["vectors"]
PROFILES: tuple[FoldProfile, ...] = ("caseless", "nfd", "none")


def same_key(first: str, second: str, profile: FoldProfile) -> bool:
    return fold_key(first, profile) == fold_key(second, profile)


def test_the_profile_type_names_the_three_profiles() -> None:
    assert set(get_args(FoldProfile)) == set(PROFILES)


def test_every_shared_spelling_pair_has_one_key_exactly_when_its_vector_says_so() -> None:
    wrong = [
        f"{profile}: {pair['note']} (expected one key: {pair[profile]})"
        for pair in VECTORS["pairs"]
        for profile in PROFILES
        if same_key(pair["a"], pair["b"], profile) != pair[profile]
    ]
    assert wrong == []


def test_the_shared_pairs_cover_every_multi_character_case_folding() -> None:
    firsts = {pair["a"] for pair in VECTORS["pairs"]}
    multi_character = [
        chr(int(code_point, 16))
        for code_point, fold in FOLD_DOCUMENT["table"]["case_folding"].items()
        if " " in fold
    ]
    assert len(multi_character) >= 100, f"only {len(multi_character)} multi-character folds"
    assert [character for character in multi_character if character not in firsts] == []


def test_nfd_joins_nfc_and_nfd_cafe_keeps_case_and_matches_hangul_to_its_jamo() -> None:
    assert same_key("caf\u00e9", "cafe\u0301", "nfd")
    assert not same_key("Caf\u00e9", "caf\u00e9", "nfd")
    # Marks before a later starter are ordered too, not only marks that end a name.
    assert same_key("a\u0323\u0302.md", "a\u0302\u0323.md", "nfd")
    for profile in ("nfd", "caseless"):
        assert same_key("\ud55c", "\u1112\u1161\u11ab", profile), profile


def test_none_only_tidies_so_case_and_accents_stay_apart() -> None:
    assert fold_key("./a//b/./c", "none") == "a/b/c"
    assert not same_key("Plan.md", "plan.md", "none")
    assert not same_key("caf\u00e9", "cafe\u0301", "none")


@pytest.mark.parametrize("vector", VECTORS["tidy"], ids=lambda vector: repr(vector["input"]))
def test_tidy_every_shared_vector(vector: dict[str, str]) -> None:
    assert tidy_path(vector["input"]) == vector["output"]


def test_caseless_joins_sharp_s_capital_sharp_s_ss_and_the_fi_ligature() -> None:
    for spelling in ("\u1e9e", "SS", "ss"):
        assert same_key("\u00df", spelling, "caseless"), spelling
    assert same_key("\ufb01", "fi", "caseless")


def test_caseless_keeps_dotless_i_apart_from_capital_i() -> None:
    assert not same_key("\u0131", "I", "caseless")


def test_caseless_joins_odos_with_a_final_sigma() -> None:
    assert same_key("\u039f\u0394\u039f\u03a3", "\u03bf\u03b4\u03bf\u03c2", "caseless")


def test_the_fold_reads_the_table_not_the_runtime_unicode_database() -> None:
    # Latin double thorn is new in Unicode 17; a fold built on unicodedata in
    # an older Python would leave it unfolded.
    assert fold_key("\ua7d2x", "caseless") == "\ua7d3x"


@pytest.mark.parametrize("vector", VECTORS["policy_fold"], ids=lambda vector: vector["note"])
def test_policy_fold_every_shared_vector(vector: dict[str, str]) -> None:
    for profile in ("caseless", "nfd"):
        assert policy_fold(vector["input"], profile) == vector[profile], profile


def test_policy_fold_turns_nfd_capitals_with_an_acute_into_nfc_lower_case() -> None:
    assert policy_fold("CAFE\u0301", "caseless") == "caf\u00e9"


def test_policy_fold_text_folds_without_tidying() -> None:
    assert policy_fold_text("Docs/./*.*//X", "caseless") == "docs/./*.*//x"
    assert policy_fold("Docs/./*.*//X", "caseless") == "docs/*.*/x"


def test_no_table_output_introduces_a_glob_character_or_a_slash() -> None:
    forbidden = {ord(character) for character in "*?[]!/"}
    table = FOLD_DOCUMENT["table"]
    outputs = [*table["decomposition"].values(), *table["case_folding"].values(), *table["composition"].values()]
    offending = [
        sequence for sequence in outputs if any(int(code_point, 16) in forbidden for code_point in sequence.split(" "))
    ]
    assert offending == []


def test_the_length_limit_counts_code_points_so_1025_is_over() -> None:
    assert FOLD_LENGTH_LIMIT == 1024
    assert exceeds_fold_limit("a" * 1025)
    for vector in VECTORS["limit"]:
        assert exceeds_fold_limit(vector["piece"] * vector["count"]) == vector["over_limit"], (
            f"{vector['count']} x {vector['piece']!r}"
        )


#: Combining classes 240 down to 1. A run of marks in this order is the worst
#: case for canonical ordering: every mark has to move past every later one.
DESCENDING_MARKS = ("\u0345", "\u035d", "\u035c", "\u0315", "\u0301", "\u0323", "\u031b", "\u0327", "\u0334")


def worst_order_run(length: int) -> str:
    mark_count = length - 1
    marks = (DESCENDING_MARKS[index * len(DESCENDING_MARKS) // mark_count] for index in range(mark_count))
    return "a" + "".join(marks)


def fastest_milliseconds(fold: Callable[[], str]) -> float:
    """The fastest of a few runs, so a scheduler pause on a loaded machine is
    not read as cost."""
    timings = []
    for _ in range(5):
        started = time.perf_counter()
        fold()
        timings.append((time.perf_counter() - started) * 1000)
    return min(timings)


def test_a_worst_order_run_of_1024_combining_code_points_folds_in_under_50_ms() -> None:
    run = worst_order_run(FOLD_LENGTH_LIMIT)
    assert len(run) == 1024
    assert not exceeds_fold_limit(run)
    folds: dict[str, Callable[[], str]] = {
        "caseless": lambda: fold_key(run, "caseless"),
        "nfd": lambda: fold_key(run, "nfd"),
        "policy fold": lambda: policy_fold(run, "caseless"),
    }
    for name, fold in folds.items():
        milliseconds = fastest_milliseconds(fold)
        assert milliseconds < 50, f"{name} took {milliseconds:.1f} ms"


def table_digest(table: Any) -> str:
    canonical = json.dumps(table, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def assert_table_pinned(document: dict[str, Any]) -> None:
    digest = table_digest(document["table"])
    assert digest == document["table_sha256"], "table_sha256 is not the hash of the table"
    assert digest == PINNED_TABLE_SHA256, "the fold table is not the pinned one"


def test_the_table_carries_the_unicode_license_notice_shipped_beside_it() -> None:
    notice = FOLD_DATA.joinpath("LICENSE-UNICODE").read_text(encoding="utf-8")
    assert "UNICODE LICENSE V3" in notice
    assert FOLD_DOCUMENT["license"] == notice


def test_the_table_hash_is_embedded_and_pinned_to_the_plugin_constant() -> None:
    assert FOLD_DOCUMENT["unicode_version"] == "17.0.0"
    assert FOLD_TABLE_UNICODE_VERSION == "17.0.0"
    assert_table_pinned(FOLD_DOCUMENT)


def test_a_one_byte_edit_inside_the_table_fails_the_pin() -> None:
    entry = '"00C0": "0041 0300"'
    assert entry in TABLE_SOURCE, f"{entry} is no longer in the table"
    edited = TABLE_SOURCE.replace(entry, '"00C0": "0041 0301"')
    assert len(edited.encode("utf-8")) == len(TABLE_SOURCE.encode("utf-8"))
    document = json.loads(edited)
    with pytest.raises(AssertionError, match="not the hash of the table"):
        assert_table_pinned(document)
    # An edit that also rewrites the embedded hash still fails the pinned constant.
    document["table_sha256"] = table_digest(document["table"])
    with pytest.raises(AssertionError, match="not the pinned one"):
        assert_table_pinned(document)


def test_a_profile_that_folds_nothing_or_no_profile_is_refused_by_the_folds() -> None:
    with pytest.raises(ValueError, match="not a folding profile"):
        policy_fold("Plan.md", "none")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="not a folding profile"):
        fold_key("Plan.md", "Caseless")  # type: ignore[arg-type]
