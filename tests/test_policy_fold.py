# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Policy on folded text: on a disk that folds names, every spelling of a file
meets the same globs, and every name a glob matched before still matches.

A name matches a glob when any of its readings does: the pattern as written
against the name as requested (tidied, for a file) and as typed; the pattern as
written against the spelling the disk stores; and the folded pattern against
the folded name. The last two apply only on a folding profile. A key, a name
that is not a workspace file, is matched by its exact name alone. The plugin's
Node coordinator runs the same scenarios against its own policy.
"""

from __future__ import annotations

import fnmatch
import itertools
import json
import logging
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from ccs.adapters.claude_code import policy as policy_module
from ccs.adapters.claude_code.coordinator_server import _handle_policy_untrack
from ccs.adapters.claude_code.fs_profile import FoldProfileResult
from ccs.adapters.claude_code.policy import PolicyMatchContext, TrackedArtifactPolicy, matches_any
from ccs.core import path_fold
from ccs.core.path_fold import FoldingProfile, FoldProfile, policy_fold_text

PROFILES: tuple[FoldProfile, ...] = ("none", "nfd", "caseless")

_POLICY_FILE_NAMES = {"tracked": "tracked.yaml", "ignored": "ignored.yaml", "strict": "strict_mode.yaml"}

PolicyFiles = dict[str, list[str]]
LoadPolicy = Callable[[PolicyFiles], TrackedArtifactPolicy]


def _write_policy(root: Path, files: PolicyFiles) -> None:
    (root / ".coherence").mkdir(parents=True, exist_ok=True)
    for kind, entries in files.items():
        text = "".join(f"- {json.dumps(entry, ensure_ascii=False)}\n" for entry in entries)
        (root / ".coherence" / _POLICY_FILE_NAMES[kind]).write_text(text, encoding="utf-8")


@pytest.fixture
def policy_with(tmp_path: Path) -> LoadPolicy:
    """Load a policy from the given files, each in a workspace of its own."""
    workspaces = itertools.count()

    def load(files: PolicyFiles) -> TrackedArtifactPolicy:
        root = tmp_path / f"ws{next(workspaces)}"
        _write_policy(root, files)
        return TrackedArtifactPolicy.load(root)

    return load


def file(profile: FoldProfile, disk_spelling: str | None = None) -> PolicyMatchContext:
    return PolicyMatchContext(profile=profile, kind="file", disk_spelling=disk_spelling)


def key(profile: FoldProfile) -> PolicyMatchContext:
    return PolicyMatchContext(profile=profile, kind="key")


def _stored_spelling(directory: Path, name: str) -> str:
    """The spelling a real directory stores and lists ``name`` under."""
    directory.mkdir()
    (directory / name).write_text("")
    [stored] = os.listdir(directory)
    return stored


def _link_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if "through the link" in record.getMessage()]


@pytest.fixture
def link_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.WARNING, logger=policy_module.__name__)
    return caplog


# ------------------------------------------------------------- folded matches


def test_a_strict_plan_md_makes_plan_md_strict_on_a_caseless_disk_and_not_on_one_that_keeps_case(
    policy_with: LoadPolicy,
) -> None:
    policy = policy_with({"strict": ["plan.md"]})
    assert policy.is_strict_mode("PLAN.MD", file("caseless"))
    assert policy.is_tracked("PLAN.MD", file("caseless"))
    for profile in ("none", "nfd"):
        assert not policy.is_strict_mode("PLAN.MD", file(profile)), profile
        assert not policy.is_tracked("PLAN.MD", file(profile)), profile
    assert not policy.is_strict_mode("PLAN.MD")


def test_data_double_star_covers_upper_case_data_on_a_caseless_disk(policy_with: LoadPolicy) -> None:
    policy = policy_with({"tracked": ["data/**"], "strict": ["data/**"]})
    assert policy.is_tracked("DATA/x.txt", file("caseless"))
    assert policy.is_strict_mode("DATA/x.txt", file("caseless"))
    assert not policy.is_tracked("DATA/x.txt", file("none"))


def test_a_question_mark_meets_the_composed_and_the_decomposed_request_by_the_folded_reading(
    policy_with: LoadPolicy,
) -> None:
    composed = "data/café.txt"
    decomposed = "data/café.txt"
    policy = policy_with({"tracked": ["data/caf?.txt"]})
    for profile in ("nfd", "caseless"):
        assert policy.is_tracked(composed, file(profile)), profile
        assert policy.is_tracked(decomposed, file(profile)), profile
    # A disk that folds nothing keeps the two forms apart in policy too: '?' meets one code point.
    assert policy.is_tracked(composed, file("none"))
    assert not policy.is_tracked(decomposed, file("none"))


def test_a_pattern_is_folded_where_its_literal_characters_stand_never_tidied(policy_with: LoadPolicy) -> None:
    # Tidied, the lone '.' between the stars would vanish and leave notes/**.
    policy = policy_with({"tracked": ["notes/*.*"]})
    assert policy.is_tracked("NOTES/A.B", file("caseless"))
    assert not policy.is_tracked("NOTES/AB", file("caseless"))


def test_a_name_over_the_fold_limit_is_matched_as_requested_only(policy_with: LoadPolicy) -> None:
    policy = policy_with({})
    deep = "D/" * 520 + "PLAN.MD"
    assert policy.is_tracked("D/PLAN.MD", file("caseless"))
    assert not policy.is_tracked(deep, file("caseless"))
    assert policy.is_tracked(deep.lower(), file("caseless"))


# -------------------------------------------------------------- disk spelling


def test_with_strasse_on_disk_a_question_mark_glob_matches_an_upper_case_request_by_the_disk_spelling(
    policy_with: LoadPolicy, tmp_path: Path
) -> None:
    disk = _stored_spelling(tmp_path / "disk", "straße.md")
    policy = policy_with({"tracked": ["stra?e.md"]})
    assert policy.is_tracked("STRASSE.MD", file("caseless", disk))
    # Folded, the request is strasse.md, which '?' cannot meet: only the disk spelling matches.
    assert not policy.is_tracked("STRASSE.MD", file("caseless"))


def test_with_a_capital_file_on_disk_a_class_matches_a_lower_case_request_on_a_caseless_disk(
    policy_with: LoadPolicy, tmp_path: Path
) -> None:
    disk = _stored_spelling(tmp_path / "disk", "A.md")
    policy = policy_with({"tracked": ["[A-Z]*.md"]})
    assert policy.is_tracked("a.md", file("caseless", disk))
    # The folded class holds the fold of each member, so the folded reading matches without the disk spelling too.
    assert policy.is_tracked("a.md", file("caseless"))
    # A disk that folds nothing matches by the request spelling alone.
    assert not policy.is_tracked("a.md", file("none", disk))


# ------------------------------------------------------------------ class fold

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


@pytest.mark.parametrize(
    ("pattern", "folded"),
    [
        # A positive class gains the fold of each member...
        ("docs/[A-Z]*.md", f"docs/[{_LETTERS}]*.md"),
        ("[P]", "[Pp]"),
        ("DOCS/[P]LAN.MD", "docs/[Pp]lan.md"),
        ("[\u212b]", "[\u00e5\u212b]"),
        # ...and is copied as written when it holds them all already.
        ("[a-z]", "[a-z]"),
        ("[Pp]", "[Pp]"),
        # A member that folds to several code points adds nothing.
        ("[\u00df]", "[\u00df]"),
        # A spelling outside the class (A to Z, the Kelvin sign, the long s)
        # folds to each of a to z, so the folded name meets no member.
        ("[!a-z]", "?"),
        # No code point folds to P or to any capital: copied as written.
        ("[!P]", "[!P]"),
        ("[!A-Z]", "[!A-Z]"),
        # The Kelvin sign, outside the class, folds to k: only K stays refused.
        ("[!kK]", "[!K]"),
        ("[!]a]", "[!]]"),
        ("[!-a]", "[!-]"),
        # After the negating '!', a member '!' leads as it sorts.
        ("[!!Ka]", "[!!K]"),
        # One spelling per member set: ']' first, '-' last, a '!' that would
        # lead a positive class moved after every other member...
        ("[]A]", "[]Aa]"),
        ("[A!]", "[Aa!]"),
        ("[-A!]", "[Aa!-]"),
        ("[]!A]", "[]!Aa]"),
        # ...and the rest in code point order, so U+FF41 comes before U+10400.
        ("[\uff21\U00010400]", "[\uff21\uff41\U00010400\U00010428]"),
        # An unterminated '[' is a literal.
        ("DOCS/[PLAN", "docs/[plan"),
    ],
)
def test_a_class_folds_to_the_folds_its_members_spell_on_a_caseless_disk(pattern: str, folded: str) -> None:
    assert policy_module._fold_glob_literals(pattern, "caseless") == folded


@pytest.mark.parametrize(
    ("pattern", "folded"),
    [
        ("DOCS/[A-Z]*.md", "DOCS/[A-Z]*.md"),
        ("[a-z]", "[a-z]"),
        ("[!a-z]", "[!a-z]"),
        ("[!0-9_]", "[!0-9_]"),
        ("[!P]", "[!P]"),
        # Case stays apart, but the Kelvin sign decomposes to K, outside the class.
        ("[!A-Z]", "[!ABCDEFGHIJLMNOPQRSTUVWXYZ]"),
        # The angstrom sign decomposes to the letter A with ring above.
        ("[\u212b]", "[\u00c5\u212b]"),
        ("[!\u00c5]", "?"),
    ],
)
def test_on_a_disk_that_folds_accents_only_a_canonical_decomposition_changes_a_class(
    pattern: str, folded: str
) -> None:
    assert policy_module._fold_glob_literals(pattern, "nfd") == folded


@pytest.mark.parametrize(
    "pattern",
    [
        pytest.param("[A-Z\u0391-\u03a9\u0410-\u042f]", id="more-members-than-the-limit"),
        pytest.param("[\u0100-\u017f]", id="a-range-wider-than-the-limit"),
    ],
)
def test_a_class_too_wide_to_enumerate_is_copied_as_written(pattern: str) -> None:
    assert policy_module._fold_glob_literals(pattern, "caseless") == pattern


def test_a_class_of_as_many_members_as_the_limit_is_folded_and_one_of_more_is_copied_as_written() -> None:
    # 64 code points from U+0100; caselessly the last, U+013F, folds to U+0140.
    at_limit = f"[{chr(0x100)}-{chr(0x13F)}]"
    past_limit = f"[{chr(0x100)}-{chr(0x140)}]"
    assert policy_module._fold_glob_literals(at_limit, "caseless") == f"[{''.join(map(chr, range(0x100, 0x141)))}]"
    assert policy_module._fold_glob_literals(past_limit, "caseless") == past_limit


def test_in_a_double_star_pattern_a_bracket_is_literal_text_folded_where_it_stands() -> None:
    assert policy_module._fold_glob_literals("DOCS/**/[A-Z]*.md", "caseless") == "docs/**/[a-z]*.md"
    assert policy_module._fold_glob_literals("data/**/[!P]", "caseless") == "data/**/[!p]"
    assert policy_module._fold_glob_literals("DOCS/**/[A-Z]*.md", "nfd") == "DOCS/**/[A-Z]*.md"


def test_every_pattern_folds_to_the_text_the_vectors_shared_with_the_node_coordinator_name() -> None:
    """The plugin's Node coordinator runs a byte-identical copy of this file
    against its own fold, so a pattern folds to the same text in both, and the
    two coordinators give every spelling of a file the same answer."""
    vectors = json.loads((Path(__file__).parent / "fixtures" / "policy_class_fold_vectors.json").read_text("ascii"))
    # Both profiles present, so an emptied or one-sided file cannot pass.
    assert {vector["profile"] for vector in vectors} == {"caseless", "nfd"}
    mismatches = [
        (vector["profile"], vector["pattern"], folded, vector["folded"])
        for vector in vectors
        if (folded := policy_module._fold_glob_literals(vector["pattern"], vector["profile"])) != vector["folded"]
    ]
    assert mismatches == []


#: Code points the class tests name, with their folds and spellings: ASCII,
#: Latin-1 and a few signs and letters that fold across blocks or planes.
_CLASS_ALPHABET = "".join(
    [*map(chr, range(0x20, 0x7F)), *map(chr, range(0xA0, 0x100))]
) + "\u017f\u0391\u03b1\u03bc\u212a\u212b\uff21\uff41\U00010400\U00010428"

_FOLDED_CLASSES = [
    "[A-Z]", "[a-z]", "[!a-z]", "[!A-Z]", "[P]", "[!P]", "[!kK]", "[]a]", "[!]a]", "[!-a]", "[A!]",
    "[-A!]", "[0-9_]", "[\u212b]", "[!\u00c5]", "[\u00df]", "[!\u00e9]", "[\u00c0-\u00de]",
    "[!\u00e0-\u00fe]", "[\uff21\U00010400]", "[!\U00010428]",
    # 39 members as written and 65 folded, past the 64 a class as written may list.
    "[A-Z0-9_.-]",
]


@pytest.fixture(scope="module")
def spellings_by_fold() -> dict[str, dict[str, set[str]]]:
    """Every code point of the basic plane and the Deseret block, grouped by
    the one code point it folds to under each folding profile: found by
    folding each of them, not from the policy's own inverse."""
    scanned = itertools.chain(range(0xD800), range(0xE000, 0x10000), range(0x10400, 0x10450))
    chars = [chr(code_point) for code_point in scanned]
    by_profile: dict[str, dict[str, set[str]]] = {}
    for profile in ("caseless", "nfd"):
        groups: dict[str, set[str]] = {}
        for char in chars:
            folded = policy_fold_text(char, profile)
            if len(folded) == 1:
                groups.setdefault(folded, set()).add(char)
        by_profile[profile] = groups
    return by_profile


@pytest.mark.parametrize("profile", ["caseless", "nfd"])
@pytest.mark.parametrize("pattern", _FOLDED_CLASSES)
def test_a_folded_class_meets_a_folded_character_exactly_when_the_class_meets_one_of_its_spellings(
    spellings_by_fold: dict[str, dict[str, set[str]]], pattern: str, profile: FoldingProfile
) -> None:
    """So every spelling of one file gets one answer from a class. Both
    readers of the folded class agree: fnmatch, which matches it, and the
    tokens the coverage search reads a folded pattern by."""
    folded_class = policy_module._fold_glob_literals(pattern, profile)
    [token] = policy_module._glob_tokens(folded_class, policy_module._FOLDED_CLASS_MEMBER_LIMIT)
    for folded_char, expected in _answers_by_folded_char(pattern, profile, spellings_by_fold):
        where = f"{folded_char!r} against {folded_class!r}"
        assert fnmatch.fnmatchcase(folded_char, folded_class) is expected, where
        assert policy_module._token_accepts(token, folded_char) is expected, where


@pytest.mark.parametrize("profile", ["caseless", "nfd"])
def test_a_class_fnmatch_negates_after_a_reversed_range_meets_every_spelling_alike_folded(
    spellings_by_fold: dict[str, dict[str, set[str]]], profile: FoldingProfile
) -> None:
    """fnmatch drops the reversed range z-a together with both its ends, so
    the '!' after them leads and negates the class: it reads [^X]. Folded by
    its members, a '!' and X, the class would turn away p, which fnmatch
    accepts. The search's tokens read it by its members too, a stated limit,
    so only fnmatch is checked."""
    pattern = "[z-a!X]"
    folded_class = policy_module._fold_glob_literals(pattern, profile)
    for folded_char, expected in _answers_by_folded_char(pattern, profile, spellings_by_fold):
        assert fnmatch.fnmatchcase(folded_char, folded_class) is expected, f"{folded_char!r} against {folded_class!r}"


def _answers_by_folded_char(
    pattern: str, profile: FoldingProfile, spellings_by_fold: dict[str, dict[str, set[str]]]
) -> Iterator[tuple[str, bool]]:
    """Each folded character of :data:`_CLASS_ALPHABET`, with the answer the
    folded class owes it: whether ``pattern`` as written meets one of its
    spellings."""
    for char in _CLASS_ALPHABET:
        folded_char = policy_fold_text(char, profile)
        if len(folded_char) == 1:
            spellings = spellings_by_fold[profile][folded_char]
            yield folded_char, any(fnmatch.fnmatchcase(spelling, pattern) for spelling in spellings)


@pytest.mark.parametrize("profile", ["caseless", "nfd"])
def test_the_fold_of_one_code_point_is_the_fold_of_its_text_for_every_code_point_scanned(
    spellings_by_fold: dict[str, dict[str, set[str]]], profile: FoldingProfile
) -> None:
    """The fold of one code point is looked up in a map of the table's
    domain, and every code point outside it is taken to fold to itself:
    checked against folding the text of each code point the scan reads."""
    folds = {spelling: folded for folded, spellings in spellings_by_fold[profile].items() for spelling in spellings}
    scanned = itertools.chain(range(0xD800), range(0xE000, 0x10000), range(0x10400, 0x10450))
    mismatches = [
        char for char in map(chr, scanned) if path_fold.fold_code_point(char, profile) != folds.get(char)
    ]
    assert mismatches == []


def test_a_class_is_folded_by_looking_its_members_up_not_by_folding_the_text_of_each(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An untrack request may carry 20 entries of 1,024 characters, each a
    run of 64-member classes: folding the text of every member took seconds
    of the request, where a lookup takes milliseconds."""
    path_fold.code_points_folding_to("a", "caseless")
    folded_texts: list[str] = []
    fold_text = path_fold.policy_fold_text

    def counting_fold_text(text: str, profile: FoldingProfile) -> str:
        folded_texts.append(text)
        return fold_text(text, profile)

    monkeypatch.setattr(path_fold, "policy_fold_text", counting_fold_text)
    assert policy_module._fold_glob_literals("[\u0100-\u013f][!\u0100-\u013f]", "caseless") != ""
    assert folded_texts == []


@pytest.mark.parametrize("profile", ["caseless", "nfd"])
def test_the_code_points_folding_to_a_character_are_those_a_scan_of_every_code_point_finds(
    spellings_by_fold: dict[str, dict[str, set[str]]], profile: FoldingProfile
) -> None:
    groups = spellings_by_fold[profile]
    scanned = set().union(*groups.values())
    for folded, spellings in groups.items():
        found = path_fold.code_points_folding_to(folded, profile)
        assert found & scanned == spellings, folded
    for char in _CLASS_ALPHABET:
        if char not in groups:
            assert not path_fold.code_points_folding_to(char, profile) & scanned, char
    # A compatibility ideograph outside the scan decomposes to one in it.
    assert "\U0002f800" in path_fold.code_points_folding_to("\u4e3d", profile)


def test_a_strict_class_holds_every_spelling_of_a_file_the_disk_stores_in_lower_case(
    policy_with: LoadPolicy,
) -> None:
    policy = policy_with({"tracked": ["docs/**"], "strict": ["docs/[A-Z]*.md"]})
    on_disk = file("caseless", "docs/plan.md")
    for spelling in ("docs/plan.md", "docs/Plan.md", "docs/PLAN.md"):
        assert policy.is_strict_mode(spelling, on_disk), spelling
        assert policy.is_strict_mode(spelling, file("caseless")), spelling
    # A disk that folds nothing keeps the spellings apart.
    assert not policy.is_strict_mode("docs/plan.md", file("none", "docs/plan.md"))


def test_a_strict_negated_class_holds_every_spelling_of_a_file_one_spelling_of_which_it_accepts(
    policy_with: LoadPolicy,
) -> None:
    policy = policy_with({"tracked": ["docs/**"], "strict": ["docs/[!a-z]*.md"]})
    on_disk = file("caseless", "docs/plan.md")
    for spelling in ("docs/plan.md", "docs/Plan.md", "docs/PLAN.md"):
        assert policy.is_strict_mode(spelling, on_disk), spelling
    assert not policy.is_strict_mode("docs/plan.md", file("none", "docs/plan.md"))


def test_a_negated_ignored_class_holding_every_spelling_of_its_members_ignores_no_spelling_of_them(
    policy_with: LoadPolicy,
) -> None:
    """No code point folds to P, and only P and p fold to p, so the class
    as written refuses every spelling of plan.md: folded, it must refuse the
    folded name too, not widen to any one character."""
    policy = policy_with({"tracked": ["docs/**"], "ignored": ["docs/[!Pp]*.md"]})
    for spelling in ("docs/plan.md", "docs/Plan.md", "docs/PLAN.md"):
        assert policy.is_tracked(spelling, file("caseless", "docs/plan.md")), spelling
    for spelling in ("docs/notes.md", "docs/NOTES.md"):
        assert not policy.is_tracked(spelling, file("caseless", "docs/notes.md")), spelling


def test_untracking_a_lower_case_spelling_a_strict_class_holds_is_refused_on_a_caseless_disk(
    policy_with: LoadPolicy,
) -> None:
    policy = policy_with({"tracked": ["docs/**"], "strict": ["docs/[A-Z]*.md"]})
    assert policy.strict_patterns_covering("docs/plan.md", "caseless") == ("docs/[A-Z]*.md",)
    assert policy.strict_patterns_covering("docs/plan.md", "none") == ()


def test_a_strict_class_fnmatch_negates_after_a_reversed_range_holds_every_spelling_of_a_file(
    policy_with: LoadPolicy,
) -> None:
    """fnmatch reads [z-a!X] as [^X]: it drops z-a with both its ends, and
    the '!' left first negates. As written it holds PLAN.md, so folded it
    must hold plan.md, the disk's own spelling, too."""
    policy = policy_with({"tracked": ["**"], "strict": ["[z-a!X]LAN.md"]})
    for spelling in ("plan.md", "Plan.md", "PLAN.md"):
        assert policy.is_strict_mode(spelling, file("caseless", "plan.md")), spelling


def test_untracking_a_name_a_strict_class_folded_past_64_members_does_not_spell_is_allowed(
    policy_with: LoadPolicy,
) -> None:
    """[A-Z0-9_.-] lists 39 members, 65 folded: past the 64 a class as
    written may list before the search reads it as any one character. The
    folded search still lists it, so an entry naming # or ~ there covers
    nothing, as on a disk that folds nothing."""
    policy = policy_with({"tracked": ["src/**"], "strict": ["src/[A-Z0-9_.-]*.py"]})
    for entry in ("src/#foo.py", "src/~x.py"):
        assert not policy.is_strict_mode(entry, file("caseless")), entry
        assert policy.strict_patterns_covering(entry, "caseless") == (), entry
        assert policy.strict_patterns_covering(entry, "none") == (), entry
    assert policy.strict_patterns_covering("src/foo.py", "caseless") == ("src/[A-Z0-9_.-]*.py",)
    # So does an entry's class, folded as the strict and tracked patterns are.
    hash_names = policy_with({"tracked": ["src/**"], "strict": ["src/#*.py"]})
    assert hash_names.strict_patterns_covering("src/[A-Z0-9_.-]*.py", "caseless") == ()
    assert hash_names.strict_patterns_covering("src/[#A-Z]*.py", "caseless") == ("src/#*.py",)


def test_the_folded_search_lists_a_range_of_up_to_twice_64_characters(policy_with: LoadPolicy) -> None:
    """A range of 100 is too wide to fold and is copied as written, but the
    folded search lists it, as it lists a class the fold grew: SRC/ meets
    src/ only folded, and # is not in the range, so nothing is covered."""
    policy = policy_with({"tracked": ["**"], "strict": ["SRC/[\u0100-\u0163]*.py"]})
    assert policy.strict_patterns_covering("src/#x.py", "caseless") == ()
    assert policy.strict_patterns_covering("src/\u0101x.py", "caseless") == ("SRC/[\u0100-\u0163]*.py",)


# ------------------------------------------------------------------------ keys


def test_a_key_is_matched_by_its_exact_name_only(policy_with: LoadPolicy) -> None:
    by_glob = policy_with({"tracked": ["reports/*.csv"]})
    assert not by_glob.is_tracked("Reports/q1.csv", key("caseless"))
    assert by_glob.is_tracked("Reports/q1.csv", file("caseless"))

    by_path = policy_with({"tracked": ["reports/q1.csv"]})
    assert not by_path.is_tracked("reports//q1.csv", key("caseless"))
    assert by_path.is_tracked("reports//q1.csv", file("none"))

    by_exact_name = policy_with({"tracked": ["reports//q1.csv"]})
    assert by_exact_name.is_tracked("reports//q1.csv", key("caseless"))


# ------------------------------------------------------------- strict coverage


class _UntrackRequest:
    """The two calls the untrack handler makes on its request."""

    def __init__(self, paths: list[str]) -> None:
        self.body = {"paths": paths}
        self.answers: list[tuple[int, dict]] = []

    def _read_json(self) -> dict:
        return self.body

    def _json(self, status: int, body: dict) -> None:
        self.answers.append((status, body))


@pytest.mark.parametrize(("profile", "refused"), [("caseless", True), ("nfd", False), ("none", False)])
def test_untracking_plan_md_while_a_strict_plan_md_covers_it_is_refused_on_a_caseless_disk(
    tmp_path: Path, profile: FoldProfile, refused: bool
) -> None:
    """The route decides coverage under the coordinator's own disk profile."""
    _write_policy(tmp_path, {"strict": ["plan.md"]})
    coordinator = SimpleNamespace(
        coordinator_root=tmp_path,
        policy=TrackedArtifactPolicy.load(tmp_path),
        fold_profile=FoldProfileResult(profile=profile, status="classified"),
    )
    request = _UntrackRequest(["PLAN.MD"])
    _handle_policy_untrack(request, coordinator)
    [(status, body)] = request.answers
    if refused:
        assert (status, body["refused"]) == (409, [{"path": "PLAN.MD", "strict_patterns": ["plan.md"]}])
        assert not (tmp_path / ".coherence" / "ignored.yaml").exists()
    else:
        assert (status, body["removed"]) == (200, ["PLAN.MD"])


def test_the_coverage_search_folds_both_sides_of_a_strict_glob_on_a_caseless_disk(policy_with: LoadPolicy) -> None:
    policy = policy_with({"tracked": ["data/**"], "strict": ["data/*.json"]})
    assert policy.strict_patterns_covering("DATA/**", "caseless") == ("data/*.json",)
    assert policy.strict_patterns_covering_each(["DATA/**"], "caseless") == (("data/*.json",),)
    assert policy.strict_patterns_covering("DATA/**", "none") == ()
    assert policy.strict_patterns_covering("DATA/**") == ()
    assert policy.strict_patterns_covering("DATA/*.txt", "caseless") == ()


def test_a_literal_strict_path_only_a_folded_tracked_pattern_reaches_is_covered_on_a_caseless_disk(
    policy_with: LoadPolicy,
) -> None:
    # **/plan.md reaches PLAN.MD only folded, so only a caseless disk holds it strict.
    policy = policy_with({"strict": ["PLAN.MD"]})
    assert policy.strict_patterns_covering("plan.md", "caseless") == ("PLAN.MD",)
    assert policy.strict_patterns_covering("plan.md", "none") == ()


def test_on_a_folding_disk_the_coverage_search_still_makes_every_refusal_the_unfolded_search_makes(
    policy_with: LoadPolicy,
) -> None:
    # Folded, the strict path is strasse.md, which '?' cannot meet; as written, it can.
    policy = policy_with({"tracked": ["**"], "strict": ["straße.md"]})
    for profile in PROFILES:
        assert policy.strict_patterns_covering("stra?e.md", profile) == ("straße.md",), profile


def test_the_folded_pass_draws_on_the_request_budget_the_search_as_written_drew_on(
    policy_with: LoadPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running out of the request's budget errs toward refusing on a folding
    disk too, where an entry's search runs twice."""
    policy = policy_with({"tracked": ["data/**"], "strict": ["data/*.json"]})
    # data/[0-9] covers nothing: its search as written takes 460 steps, folded as many again.
    monkeypatch.setattr(policy_module, "GLOB_INTERSECT_REQUEST_BUDGET", 700)
    assert policy.strict_patterns_covering_each(["data/[0-9]"], "none") == ((),)
    assert policy.strict_patterns_covering_each(["data/[0-9]"], "caseless") == (("data/*.json",),)


def test_the_folded_pass_does_not_search_again_a_strict_pattern_the_search_as_written_covered(
    policy_with: LoadPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strict pattern the search as written covered stays reported; the
    folded pass spends the request's budget only on the ones left open."""
    policy = policy_with({"tracked": ["data/**"], "strict": ["data/*.json", "data/[a-z]"]})
    # As written the entry covers data/*.json in 2,356 steps and rules out
    # data/[a-z] in 702; folded, data/[a-z] takes 702 more, 3,760 in all.
    # Searching data/*.json again would run past 4,000, and an entry out of
    # budget covers every strict pattern its literal ends allow: data/[a-z] too.
    monkeypatch.setattr(policy_module, "GLOB_INTERSECT_REQUEST_BUDGET", 4_000)
    assert policy.strict_patterns_covering_each(["data/[0-9]*.json"], "caseless") == (("data/*.json",),)


class _StepCount:
    def __init__(self) -> None:
        self.taken = 0


@pytest.fixture
def charged_steps(monkeypatch: pytest.MonkeyPatch) -> _StepCount:
    """Every step a coverage decision charges, counted once at the budget
    every other budget draws on, as the Node coordinator counts them."""
    count = _StepCount()

    class CountingBudget(policy_module._SearchBudget):
        def spend(self, steps: int) -> None:
            if self.parent is None:
                count.taken += steps
            super().spend(steps)

    monkeypatch.setattr(policy_module, "_SearchBudget", CountingBudget)
    return count


_UNFOLDED_STEPS: list[tuple[PolicyFiles, list[tuple[str, bool, int]]]] = [
    (
        {"tracked": ["data/**"], "strict": ["data/*.json"]},
        [
            ("data/**", True, 974),
            ("data/a.json", True, 982),
            ("**", True, 1074),
            ("*.json", True, 1146),
            ("data/a.txt", False, 0),
            ("./data/**", True, 974),
            ("data/[0-9]", False, 460),
            ("data/[!a-z].json", True, 2831),
            ("data/[a.json", True, 1195),
        ],
    ),
    (
        {"strict": ["CLAUDE.md"]},
        [("*.md", True, 10), ("C?AUDE.md", True, 10), ("**", True, 27), ("*/CLAUDE.md", False, 10)],
    ),
    (
        {"tracked": ["notes/**"], "strict": ["notes/\U0001f600.md"]},
        [("*.md", True, 11), ("notes/?.md", True, 11)],
    ),
]


def test_on_a_disk_that_folds_nothing_the_coverage_search_takes_the_steps_it_took_before(
    policy_with: LoadPolicy, charged_steps: _StepCount
) -> None:
    """The counts the plugin's Node coordinator pins for the same searches."""
    for files, entries in _UNFOLDED_STEPS:
        policy = policy_with(files)
        for entry, covers, steps in entries:
            before = charged_steps.taken
            [covering] = policy.strict_patterns_covering_each([entry], "none")
            assert (bool(covering), charged_steps.taken - before) == (covers, steps), entry


# ---------------------------------------------------------------------- safety


def test_an_ignored_pattern_never_takes_a_strict_path_out_of_tracking_under_any_spelling(
    policy_with: LoadPolicy,
) -> None:
    stored = "notes/café.md"
    policy = policy_with({
        "tracked": ["notes/**"],
        "strict": [stored],
        "ignored": ["**", "NOTES/CAFÉ.MD", "notes/café.md"],
    })
    spellings = [stored, "NOTES/CAFÉ.MD", "notes/café.md", "Notes/Café.md", "./NOTES//café.md"]
    for profile in PROFILES:
        for spelling in spellings:
            for context in (file(profile), file(profile, stored)):
                where = f"{spelling} on {profile}, disk {context.disk_spelling}"
                # Ignored '**' untracks every path strict does not hold.
                assert policy.is_tracked(spelling, context) == policy.is_strict_mode(spelling, context), where
    # Not vacuous: on a caseless disk every spelling is strict, and so tracked.
    for spelling in spellings:
        assert policy.is_strict_mode(spelling, file("caseless")), spelling


def test_dot_slash_padded_names_are_tracked_and_strict_under_data_double_star_on_every_profile(
    policy_with: LoadPolicy,
) -> None:
    policy = policy_with({"tracked": ["data/**"], "strict": ["data/**"]})
    for profile in PROFILES:
        for spelling in (".//data/x.txt", "././data/x.txt"):
            assert policy.is_tracked(spelling, file(profile)), f"{spelling} on {profile}"
            assert policy.is_strict_mode(spelling, file(profile)), f"{spelling} on {profile}"
    assert policy.is_tracked(".//data/x.txt")
    assert policy.is_strict_mode("././data/x.txt")


def test_a_glob_written_with_a_double_slash_or_a_dot_segment_still_matches_a_request_typed_the_same_way(
    policy_with: LoadPolicy,
) -> None:
    # Tidying the request must not lose a match the as-typed spelling had.
    policy = policy_with({"tracked": ["docs//plan.md", "notes/./todo.md"], "strict": ["notes/./todo.md"]})
    for profile in PROFILES:
        assert policy.is_tracked("docs//plan.md", file(profile)), profile
        assert policy.is_strict_mode("notes/./todo.md", file(profile)), profile
    assert policy.is_tracked("docs//plan.md")


@pytest.mark.parametrize("spelling", ["data\\x.txt", "data\\.\\x.txt", "data\\\\x.txt", ".\\data\\x.txt"])
@pytest.mark.parametrize("pattern", ["data/**", "data/x.txt", "data/*.txt"])
def test_a_backslash_name_keeps_matching_exactly_as_today(
    policy_with: LoadPolicy, spelling: str, pattern: str
) -> None:
    """A backslash is never a separator to the tidy or the folds, so a name
    holding one meets each glob exactly as it did before, when the matcher
    alone read it as a slash."""
    today = matches_any(spelling, [pattern])
    policy = policy_with({"tracked": [pattern], "strict": [pattern]})
    for context in (None, file("none")):
        args = (spelling,) if context is None else (spelling, context)
        assert policy.is_tracked(*args) is today, context
        assert policy.is_strict_mode(*args) is today, context
    for profile in ("nfd", "caseless"):
        # Folding only ever adds a match.
        assert policy.is_tracked(spelling, file(profile)) or not today, profile


# ------------------------------------------------------------- linked prefixes


def test_a_tracked_or_strict_glob_whose_literal_prefix_crosses_a_link_inside_the_root_is_logged_once(
    tmp_path: Path, link_log: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "real").mkdir()
    (tmp_path / "plain").mkdir()
    for link in ("linked", "aliased", "other"):
        os.symlink("real", tmp_path / link)
    _write_policy(tmp_path, {
        "tracked": ["aliased/**", "plain/**"],
        "strict": ["linked/*.md", "plain/*.md"],
        "ignored": ["other/**"],
    })

    loaded = TrackedArtifactPolicy.load(tmp_path)
    at_load = _link_lines(link_log)
    assert len(at_load) == 2, at_load
    assert 'tracked pattern "aliased/**" goes through the link "aliased"' in at_load[0]
    assert 'strict pattern "linked/*.md" goes through the link "linked"' in at_load[1]

    link_log.clear()
    reloaded = loaded.reloaded()
    assert _link_lines(link_log) == []

    with (tmp_path / ".coherence" / "tracked.yaml").open("a", encoding="utf-8") as tracked:
        tracked.write('- "linked/deep/**"\n')
    reloaded.reloaded()
    at_reload = _link_lines(link_log)
    assert len(at_reload) == 1, at_reload
    assert 'tracked pattern "linked/deep/**" goes through the link "linked"' in at_reload[0]


def test_a_linked_glob_of_a_policy_without_strict_patterns_is_logged_once_across_reloads(
    tmp_path: Path, link_log: pytest.LogCaptureFixture
) -> None:
    """A reload without strict patterns reads the files afresh; it still
    knows which patterns were logged."""
    (tmp_path / "real").mkdir()
    os.symlink("real", tmp_path / "linked")
    _write_policy(tmp_path, {"tracked": ["linked/**"]})

    loaded = TrackedArtifactPolicy.load(tmp_path)
    assert len(_link_lines(link_log)) == 1
    link_log.clear()
    loaded.reloaded().reloaded()
    assert _link_lines(link_log) == []


def test_a_strict_or_tracked_entry_naming_a_linked_file_is_logged_once_naming_the_target(
    tmp_path: Path, link_log: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "docs" / "v2").mkdir(parents=True)
    (tmp_path / "docs" / "v2" / "spec.md").write_text("spec")
    os.symlink(os.path.join("v2", "spec.md"), tmp_path / "docs" / "current.md")
    os.symlink("missing.md", tmp_path / "docs" / "dangle.md")
    _write_policy(tmp_path, {
        "tracked": ["docs/current.md", "docs/v2/spec.md"],
        "strict": ["docs/current.md", "docs/v2/spec.md", "docs/dangle.md"],
    })

    loaded = TrackedArtifactPolicy.load(tmp_path)
    at_load = _link_lines(link_log)
    # One line per entry, though the same entry is both tracked and strict.
    assert len(at_load) == 2, at_load
    assert at_load[0] == (
        'tracked pattern "docs/current.md" goes through the link "docs/current.md" inside the '
        "workspace. The hook client follows the link, so a Read, Edit or Write through it reaches "
        'the coordinator under the path of its target "docs/v2/spec.md", which this entry does not '
        "match; name the target to cover them."
    )
    assert at_load[1] == (
        'strict pattern "docs/dangle.md" goes through the link "docs/dangle.md" inside the '
        "workspace, which does not resolve. The hook client sends nothing for a Read, Edit or Write "
        "through it until its target exists, then follows it to the target's path, which this entry "
        "does not match; name the target to cover them."
    )

    link_log.clear()
    loaded.reloaded()
    assert _link_lines(link_log) == []


def test_an_entry_through_a_link_whose_target_is_outside_the_workspace_is_logged_as_not_coordinated(
    tmp_path: Path, link_log: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "ws"
    (root / "data").mkdir(parents=True)
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.md").write_text("secret")
    os.symlink(tmp_path / "outside" / "secret.md", root / "data" / "out.md")
    os.symlink(tmp_path / "outside", root / "ext")
    _write_policy(root, {"tracked": ["ext/**"], "strict": ["data/out.md"]})

    TrackedArtifactPolicy.load(root)
    at_load = _link_lines(link_log)
    assert len(at_load) == 2, at_load
    assert at_load[0].startswith('tracked pattern "ext/**" goes through the link "ext"')
    assert at_load[1] == (
        'strict pattern "data/out.md" goes through the link "data/out.md", whose target is outside '
        "the workspace. A Read, Edit or Write through it is not coordinated at all: the hook client "
        "sends nothing for a path outside the workspace, so no rewrite of this entry covers them."
    )
    for line in at_load:
        assert "not coordinated at all" in line
        assert "name the target" not in line


def test_a_link_whose_target_spells_the_workspace_in_another_case_is_read_as_inside_it(
    tmp_path: Path, link_log: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "Repo"
    (root / "docs").mkdir(parents=True)
    if not (tmp_path / "REPO").exists():
        pytest.skip("the temporary disk keeps two spellings apart, so REPO is another directory")
    # The hook client meets the root under the disk's spelling, as it does here.
    os.symlink(tmp_path / "REPO" / "docs", root / "ext")
    _write_policy(root, {"strict": ["ext/*.md"]})

    TrackedArtifactPolicy.load(root)
    at_load = _link_lines(link_log)
    assert len(at_load) == 1, at_load
    assert 'its target "docs", which this entry does not match' in at_load[0]


@pytest.mark.parametrize(
    "segment",
    [
        pytest.param("n" * 300, id="longer-than-any-disk-takes"),
        pytest.param("a\x00b", id="holding-a-nul"),
    ],
)
def test_a_literal_prefix_the_disk_cannot_look_up_is_not_logged_and_does_not_stop_the_load(
    tmp_path: Path, link_log: pytest.LogCaptureFixture, segment: str
) -> None:
    (tmp_path / "notes").mkdir()
    unreachable = f"notes/{segment}/**"
    _write_policy(tmp_path, {"tracked": [unreachable]})
    loaded = TrackedArtifactPolicy.load(tmp_path)
    assert loaded.user_added_patterns == (unreachable,)
    assert _link_lines(link_log) == []
