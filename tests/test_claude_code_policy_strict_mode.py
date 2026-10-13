# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for TrackedArtifactPolicy strict-mode opt-in (v0.2 plan Unit 1, KTD-O).

Covers:
- Per-artifact strict-mode opt-in via ``.coherence/strict_mode.yaml``.
- Intersection semantics: strict mode requires (tracked AND matches strict glob).
- Empty strict_mode_paths preserves v0.1.1 warn-mode behavior for every artifact.
- One-shot threshold warning when strict_mode_paths matches > 50 tracked artifacts.
- Path-traversal guard rejects the same malformed patterns as tracked/ignored.
- Summary surface exposes strict_mode_pattern_count.
"""

from __future__ import annotations

import itertools
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ccs.adapters.claude_code import policy as policy_module
from ccs.adapters.claude_code.policy import (
    _ANY_ONE,
    _CLASS,
    _LIT,
    DEFAULT_TRACKED_PATTERNS,
    STRICT_MODE_PATH_WARN_THRESHOLD,
    TrackedArtifactPolicy,
    _BudgetExhausted,
    _glob_regex,
    _glob_tokens,
    _globs_intersect,
    _normalize_relative,
    _SearchBudget,
    _tokens_match,
    matches_any,
)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / ".coherence").mkdir()
    return tmp_path


def _quoted_yaml(patterns: list[str]) -> str:
    return "".join(f"- '{p}'\n" for p in patterns)


def _intersect(patterns: tuple[str, ...], budget: _SearchBudget) -> bool:
    return _globs_intersect([_glob_tokens(p) for p in patterns], budget)


# --------------------------------------------------------------------
# Happy path — strict_mode_paths intersected with tracked_paths
# --------------------------------------------------------------------


def test_strict_mode_single_path_with_default_tracked(root: Path) -> None:
    """Single strict glob targeting a default-tracked artifact."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- docs/plans/feature-x.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    # The specific path is in strict mode.
    assert policy.is_strict_mode("docs/plans/feature-x.md")
    # A sibling tracked path is NOT in strict mode (default warn-mode).
    assert policy.is_tracked("docs/plans/feature-y.md")
    assert not policy.is_strict_mode("docs/plans/feature-y.md")


def test_strict_mode_with_claude_md(root: Path) -> None:
    """CLAUDE.md is default-tracked; strict-mode entry promotes it."""
    (root / ".coherence" / "strict_mode.yaml").write_text("- CLAUDE.md\n")
    policy = TrackedArtifactPolicy.load(root)

    assert policy.is_tracked("CLAUDE.md")
    assert policy.is_strict_mode("CLAUDE.md")


def test_strict_mode_glob_pattern_with_default_tracked(root: Path) -> None:
    """A ``**`` glob in strict_mode_paths matches deeply-nested artifacts."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- docs/plans/**/*.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    assert policy.is_strict_mode("docs/plans/feature-x.md")
    assert policy.is_strict_mode("docs/plans/nested/feature-y.md")
    # docs/plans/ is default-tracked but the glob also matches.
    assert policy.is_tracked("docs/plans/feature-x.md")


# --------------------------------------------------------------------
# Intersection semantics
# --------------------------------------------------------------------


def test_empty_strict_mode_paths_preserves_v011_behavior(root: Path) -> None:
    """No strict_mode.yaml at all → is_strict_mode is False for everything,
    including default-tracked artifacts. This is the back-compat invariant
    that gates v0.1.1 warn-mode preservation."""
    policy = TrackedArtifactPolicy.load(root)
    assert policy.strict_mode_paths == ()
    for path in ("CLAUDE.md", "docs/plans/x.md", "src/main.py", "untracked.txt"):
        assert not policy.is_strict_mode(path), (
            f"empty strict_mode_paths should produce False for every path; "
            f"got True for {path!r}"
        )


def test_strict_mode_requires_tracked_path_first(root: Path) -> None:
    """Strict-mode never applies to an untracked artifact, even when the
    strict_mode glob would match. Intersection semantics — tracked-set
    membership is the precondition."""
    # Add a strict_mode entry for a path that is NOT in the tracked defaults.
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- src/important/payment.py\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    # The path matches strict_mode_paths but is NOT tracked → not strict.
    assert not policy.is_tracked("src/important/payment.py")
    assert not policy.is_strict_mode("src/important/payment.py")


def test_strict_mode_with_tracked_yaml_opt_in(root: Path) -> None:
    """When the operator opts a path into the tracked set via tracked.yaml
    AND adds it to strict_mode.yaml, it becomes strict-mode."""
    (root / ".coherence" / "tracked.yaml").write_text(
        "- src/important/payment.py\n"
    )
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- src/important/payment.py\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    assert policy.is_tracked("src/important/payment.py")
    assert policy.is_strict_mode("src/important/payment.py")


def test_strict_mode_wins_over_ignored_yaml(root: Path, caplog: pytest.LogCaptureFixture) -> None:
    """A strict path stays tracked and strict even when ignored.yaml names it
    (#261). Before, ignore won: the path dropped to the untracked fast path,
    where a read reports version 0 and a CAS commit is accepted at any expected
    version, so a leftover ignored.yaml silently disarmed strict mode. The
    load names the overridden ignore entry once."""
    (root / ".coherence" / "tracked.yaml").write_text(
        "- src/special/secret.py\n"
    )
    (root / ".coherence" / "ignored.yaml").write_text(
        "- src/special/secret.py\n"
    )
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- src/special/secret.py\n"
    )
    with caplog.at_level(logging.WARNING, logger="ccs.adapters.claude_code.policy"):
        policy = TrackedArtifactPolicy.load(root)

    assert policy.is_tracked("src/special/secret.py")
    assert policy.is_strict_mode("src/special/secret.py")
    assert policy.ignored_patterns_overridden_by_strict() == ("src/special/secret.py",)
    assert any("strict wins" in r.getMessage() for r in caplog.records)


def test_a_broader_ignore_pattern_does_not_untrack_a_strict_path(root: Path) -> None:
    """The issue's second case: an ignore pattern that covers the strict glob
    under another spelling (``**``) takes non-strict paths off the tracked set
    as before, but never a strict one."""
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- data/**\n")
    (root / ".coherence" / "ignored.yaml").write_text("- '**'\n")
    policy = TrackedArtifactPolicy.load(root)

    assert policy.is_tracked("data/a.txt") and policy.is_strict_mode("data/a.txt")
    # Non-strict tracked paths: ignore still wins, unchanged.
    assert not policy.is_tracked("CLAUDE.md")
    assert not policy.is_tracked("docs/plans/x.md")


def test_ignore_still_wins_for_tracked_non_strict_paths(root: Path) -> None:
    """Control: with strict patterns live, an ignore on a non-strict tracked
    path untracks it as it always did."""
    (root / ".coherence" / "strict_mode.yaml").write_text("- CLAUDE.md\n")
    (root / ".coherence" / "ignored.yaml").write_text("- docs/plans/**/*.md\n")
    policy = TrackedArtifactPolicy.load(root)

    assert not policy.is_tracked("docs/plans/x.md")
    assert policy.ignored_patterns_overridden_by_strict() == ()


@pytest.mark.parametrize(
    ("entry", "covering"),
    [
        ("data/**", ("data/*.json",)),            # the issue's glob untrack
        ("data/a.json", ("data/*.json",)),        # a literal strict path
        ("**", ("data/*.json",)),                 # a broader pattern
        ("*.json", ("data/*.json",)),             # fnmatch '*' crosses '/'
        ("data/a.txt", ()),                       # tracked, not strict
        ("data/*.txt", ()),                       # disjoint glob
        ("notes/**", ()),                         # strict-free and untracked
        ("data?a.json", ("data/*.json",)),        # fnmatch '?' crosses '/'
        ("./data/a.json", ("data/*.json",)),      # a leading ./ is the same path
        ("./data/**", ("data/*.json",)),          # ... and the same glob
    ],
)
def test_strict_patterns_covering_decides_on_the_glob_languages(
    root: Path, entry: str, covering: tuple[str, ...]
) -> None:
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- data/*.json\n")
    policy = TrackedArtifactPolicy.load(root)

    assert policy.strict_patterns_covering(entry) == covering


@pytest.mark.parametrize(
    ("tracked", "strict", "entry", "covering"),
    [
        # Only the second tracked pattern reaches data/a.json: every tracked
        # pattern is tried, not just the first, and one is enough.
        (["data/[x]*.json", "data/**"], "data/*.json", "data/a.json", ("data/*.json",)),
        # The entry and the strict glob meet (data/c.json), but no tracked
        # path is among them, so nothing they share is strict.
        (["data/[ab]*.json"], "data/*", "data/c*.json", ()),
    ],
)
def test_a_strict_glob_is_covered_only_through_a_tracked_pattern(
    root: Path, tracked: list[str], strict: str, entry: str, covering: tuple[str, ...]
) -> None:
    """Strict mode is an intersection, so an entry covers a strict glob only
    where the two meet on a tracked path. Deciding it on the first tracked
    pattern alone, or requiring every tracked pattern, accepts the first
    entry's untrack of a strict path; skipping the tracked check refuses the
    second's ordinary one."""
    (root / ".coherence" / "tracked.yaml").write_text(_quoted_yaml(tracked))
    (root / ".coherence" / "strict_mode.yaml").write_text(_quoted_yaml([strict]))
    policy = TrackedArtifactPolicy.load(root)

    assert policy.strict_patterns_covering(entry) == covering


def test_a_literal_strict_path_no_tracked_pattern_reaches_is_not_covered(
    root: Path,
) -> None:
    """A literal strict path is decided by matching it, but only a tracked
    one is strict: an untrack over an untracked literal is not refused."""
    (root / ".coherence" / "strict_mode.yaml").write_text("- scratch/x.tmp\n")
    policy = TrackedArtifactPolicy.load(root)

    assert not policy.is_strict_mode("scratch/x.tmp")
    assert policy.strict_patterns_covering("scratch/*") == ()
    assert policy.strict_patterns_covering("scratch/x.tmp") == ()


@pytest.mark.parametrize(
    ("entry", "covering"),
    [
        ("./.github/ci.yml", (".github/*.yml",)),
        ("./.github/**", (".github/*.yml",)),
        (".github/**", (".github/*.yml",)),
        ("./.env", ()),
    ],
)
def test_a_leading_dot_slash_on_a_dotfile_entry_names_the_same_path(
    root: Path, entry: str, covering: tuple[str, ...]
) -> None:
    """Only the literal ``./`` is stripped from an entry (removeprefix): an
    lstrip of the characters ``.`` and ``/`` would turn ``./.github/ci.yml``
    into ``github/ci.yml`` and accept its untrack."""
    (root / ".coherence" / "tracked.yaml").write_text("- '.github/**'\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- '.github/*.yml'\n")
    policy = TrackedArtifactPolicy.load(root)

    assert policy.strict_patterns_covering(entry) == covering


def test_the_overlap_search_keeps_single_stars_within_a_segment(root: Path) -> None:
    """The search is exact for ``*``, ``**`` and ``?``: in a ``**`` pattern
    ``*`` never crosses ``/``, so an entry only a crossing star could reach
    is not covered. Letting it cross over-reports, refusing the untrack."""
    assert not _intersect(("**/a*b", "a/b"), _SearchBudget(100_000))
    (root / ".coherence" / "tracked.yaml").write_text("- src/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- 'src/a/b/x/[0-9]'\n")
    policy = TrackedArtifactPolicy.load(root)

    assert policy.strict_patterns_covering("src/*/x/**") == ()
    assert policy.strict_patterns_covering("src/*/*/x/**") == ("src/a/b/x/[0-9]",)


def test_a_root_level_path_is_covered_by_a_leading_double_star(root: Path) -> None:
    """``**/`` matches zero directories: the matcher swallows the slash after
    ``**``, so a strict ``**/plan.md`` holds the root-level ``plan.md``."""
    (root / ".coherence" / "strict_mode.yaml").write_text("- '**/plan.md'\n")
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_strict_mode("plan.md")

    assert policy.strict_patterns_covering("plan.md") == ("**/plan.md",)


# Every glob shape the matcher handles: ``**`` patterns (a run, a segment run,
# a segment character, the slash swallowed after ``**``) and fnmatch patterns
# (``*`` and ``?`` crossing '/', classes, negation, an unterminated '[').
_GRAMMAR_CORPUS = (
    "a", "a.x", "x", "*", "?", "**", "**/a", "a/**", "a/**/b", "**/*", "**/a?",
    "**/*.x", "a?", "a*b", "*.x", "a/*", "*/b", "?/?", "[ab]", "[!a]", "a[", "[a-b]/x",
    # A '*' meeting a segment boundary inside a ** pattern, a leading '/',
    # and a negated class whose only witness is '/'.
    "**/a*b", "*x/**", "/x/a", "a*/**", "a[!x]b",
)
_GRAMMAR_PATHS = tuple(
    "".join(chars)
    for length in range(1, 5)
    for chars in itertools.product("ab/x.", repeat=length)
)


def test_the_overlap_automaton_never_under_reports_the_matcher() -> None:
    """``_glob_tokens`` re-reads the glob grammar ``matches_any`` uses, so
    nothing else ties the overlap search to the matcher it must agree with.
    Over every path of up to four characters, wherever some path matches
    each pattern of a pair or triple, the search must say they intersect: an
    under-report accepts an untrack that covers a strict path. (It may
    over-report: its approximations err toward "intersects".)"""
    matching = {
        g: {p for p in _GRAMMAR_PATHS if matches_any(p, [g])} for g in _GRAMMAR_CORPUS
    }
    missed = [
        combo
        for size in (2, 3)
        for combo in itertools.combinations(_GRAMMAR_CORPUS, size)
        if set.intersection(*(matching[g] for g in combo))
        and not _intersect(combo, _SearchBudget(10_000_000))
    ]
    assert missed == []


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("a/b", "**/a*b", False),
        ("x/axb", "**/a*b", True),
        ("a/b", "**/a?b", False),
        ("src/x/state/a", "src/*/state/**", True),
        ("src/x/y/state/a", "src/*/state/**", False),
        # Over the regex run limit, matched by stepping the tokens.
        ("x/a/b", "**/*a*/*b*", True),
        ("a/x/b", "**/*a*/*b*", False),
    ],
)
def test_a_single_star_in_a_double_star_pattern_stays_within_a_segment(
    path: str, pattern: str, expected: bool
) -> None:
    """Literal expectations for the ``**`` matcher: its regex and the overlap
    search read the same tokens, so comparing the two cannot catch a ``*``
    or ``?`` that starts crossing ``/``, which would widen every ``**``
    pattern a user tracks, ignores or holds strict."""
    assert matches_any(path, [pattern]) is expected


_RUN_HEAVY = ("**/*a*b", "**/**/**/a", "a/**/*/*b*", "**/*a*/*b*", "*/**/*x*")


def test_the_linear_matcher_agrees_with_the_double_star_regex() -> None:
    """A ``**`` pattern with more runs than its regex can match without
    backtracking is matched by stepping its tokens instead. Over every short
    path, newlines included (the regex's ``.*`` stops at one, and its ``$``
    also matches before a final one), the two must agree."""
    patterns = [g for g in _GRAMMAR_CORPUS + _RUN_HEAVY if "**" in g]
    paths = _GRAMMAR_PATHS + tuple(
        "".join(chars) for length in range(1, 5) for chars in itertools.product("a/\nx", repeat=length)
    )
    disagree = [
        (g, p)
        for g in patterns
        for p in paths
        if _tokens_match(_glob_tokens(g), p)
        != (_glob_regex(_glob_tokens(g)).match(p) is not None)
    ]
    assert disagree == []


def test_the_coverage_shortcuts_never_under_report() -> None:
    """``strict_patterns_covering`` skips the search where literal ends rule a
    strict pattern out, matches a literal strict path instead of searching,
    and keeps only tracked literal paths. Over every (entry, strict, tracked)
    triple of the grammar corpus, wherever a short path is strict and matches
    the entry, the strict pattern must be reported."""
    paths = {n for n in map(_normalize_relative, _GRAMMAR_PATHS) if n is not None}
    matching = {g: {p for p in paths if matches_any(p, [g])} for g in _GRAMMAR_CORPUS}
    missed = []
    for strict, tracked in itertools.product(_GRAMMAR_CORPUS, repeat=2):
        strict_paths = matching[strict] & matching[tracked]
        if not strict_paths:
            continue
        policy = TrackedArtifactPolicy(
            coordinator_root=Path("."), tracked_patterns=(tracked,), strict_mode_paths=(strict,)
        )
        missed += [
            (entry, strict, tracked)
            for entry in _GRAMMAR_CORPUS
            if matching[entry] & strict_paths
            and policy.strict_patterns_covering(entry) != (strict,)
        ]
    assert missed == []


def test_a_strict_pattern_with_no_tracked_cover_is_not_covering(root: Path) -> None:
    """Strict mode is an intersection: a strict glob that reaches no tracked
    path puts nothing in strict mode, so untracking under it is not refused."""
    (root / ".coherence" / "strict_mode.yaml").write_text("- '**'\n")
    policy = TrackedArtifactPolicy.load(root)

    # notes/*.txt reaches no default tracked pattern (notes/** would: the
    # default **/plan.md matches notes/plan.md).
    assert policy.strict_patterns_covering("notes/*.txt") == ()
    assert policy.strict_patterns_covering("notes/**") == ("**",)
    assert policy.strict_patterns_covering("docs/**") == ("**",)


@pytest.mark.parametrize("negated", [False, True])
def test_a_class_range_too_wide_to_expand_errs_toward_covering(
    root: Path, negated: bool
) -> None:
    """A class range wider than the expand limit is approximated so the class
    accepts more, never less, in both polarities: an untrack of a path the
    strict class matches is refused, never answered as a silent untrack."""
    bang = "!" if negated else ""
    # [0-\u0300] holds '5'; [!a-\u0300] does too (5 sorts below 'a').
    strict = f"data/[{bang}{'a' if negated else '0'}-\u0300].json"
    (root / ".coherence" / "tracked.yaml").write_text("- data/*\n")
    (root / ".coherence" / "strict_mode.yaml").write_text(f"- '{strict}'\n")
    (root / ".coherence" / "ignored.yaml").write_text("- data/5.json\n")
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_strict_mode("data/5.json")

    assert _intersect(("data/5.json", strict), _SearchBudget(100_000))
    assert policy.strict_patterns_covering("data/5.json") == (strict,)
    assert policy.ignored_patterns_overridden_by_strict() == ("data/5.json",)


_STAR_HEAVY = "**/" + "/".join(f"*{c}*" for c in "abcdefgh")


def test_a_search_over_its_budget_stops() -> None:
    """The product search is exponential in a star-heavy glob's segments; a
    bounded search stops by raising, which every caller counts as
    "intersects" (the refusal side)."""
    # Disjoint, but only a search shows it: no segment of src/<digit> holds an 'a'.
    assert not _intersect(("**/*a*", "src/[0-9]"), _SearchBudget(1_000_000))
    with pytest.raises(_BudgetExhausted):
        _intersect((_STAR_HEAVY, "src/**/*.py"), _SearchBudget(50))


def test_literal_ends_rule_a_search_out_at_no_cost() -> None:
    """Every path a glob matches starts with its literal prefix and ends with
    its literal suffix, so globs whose prefixes or suffixes do not nest
    cannot intersect: decided before any step is spent."""
    budget = _SearchBudget(0)
    assert not _intersect(("**/*.log", "svc3/state/*.json"), budget)
    assert not _intersect(("svc3/**", "svc7/**"), budget)
    assert budget.remaining == 0


def test_a_star_heavy_untrack_entry_is_decided_within_the_budget(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A star-heavy entry is decided well inside the /policy/untrack handler
    (unbounded, this case explored the whole state space against every
    default tracked pattern: tens of seconds). A strict glob whose search
    runs out of its budget counts as covered, unless no tracked pattern's
    literal ends can meet it: ``docs/**/*.txt`` reaches no tracked path (the
    default ``docs/`` patterns end in ``.md``), so nothing under it is
    strict, and it is exactly not covered."""
    (root / ".coherence" / "tracked.yaml").write_text("- src/**/*.py\n")
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- 'docs/**/*.txt'\n- 'src/**/*.py'\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    started = time.monotonic()
    covering = policy.strict_patterns_covering(_STAR_HEAVY)
    assert time.monotonic() - started < 2.0
    assert "src/**/*.py" in covering

    monkeypatch.setattr(policy_module, "GLOB_INTERSECT_PATTERN_BUDGET", 1)
    assert policy.strict_patterns_covering(_STAR_HEAVY) == ("src/**/*.py",)


_STAR_HEAVY_LOG = _STAR_HEAVY + ".log"


def test_each_strict_glob_is_decided_on_its_own_budget(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One strict glob running out of search budget counts as covered, and
    only that one: a later strict glob is still decided. With one budget
    shared across the call, the first exhaustion marked every pattern after
    it as covering."""
    (root / ".coherence" / "tracked.yaml").write_text("- '**/*.log'\n")
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- '**/*.log'\n- 'notes/[0-9].log'\n"
    )
    policy = TrackedArtifactPolicy.load(root)
    # The first search needs about 13,600 steps; the second settles "disjoint"
    # (eight segments never fit notes/<digit>.log) in about 1,600.
    monkeypatch.setattr(policy_module, "GLOB_INTERSECT_PATTERN_BUDGET", 5_000)

    assert policy.strict_patterns_covering(_STAR_HEAVY_LOG) == ("**/*.log",)


def test_once_the_call_budget_runs_out_undecided_patterns_count_as_covered(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the entry's budget runs out, every strict pattern not yet
    decided counts as covered, unless its literal ends rule it out. That
    includes a literal strict path: matching it is charged like a search, so
    an entry against many literal paths stays bounded too. Within budget,
    ``notes/a.log`` (it ends like the entry) is decided exactly: the entry's
    eight segments cannot reach it."""
    (root / ".coherence" / "tracked.yaml").write_text("- '**/*.log'\n- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- '**/*.log'\n- CLAUDE.md\n- notes/a.log\n- 'data/*.json'\n- 'src/**/*.log'\n"
    )
    policy = TrackedArtifactPolicy.load(root)
    assert "notes/a.log" not in policy.strict_patterns_covering(_STAR_HEAVY_LOG)
    monkeypatch.setattr(policy_module, "GLOB_INTERSECT_CALL_BUDGET", 100)

    assert policy.strict_patterns_covering(_STAR_HEAVY_LOG) == (
        "**/*.log",
        "notes/a.log",
        "src/**/*.log",
    )


def test_an_untrack_request_is_decided_under_one_budget(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each entry of a request has its own budget, drawn from one for the
    whole request. Once that runs out, a later entry counts as covering every
    strict pattern its literal ends do not rule out, so a request of many
    costly entries is refused within the client's timeout instead of held
    past it."""
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- 'data/*.json'\n")
    policy = TrackedArtifactPolicy.load(root)
    # data/[0-9] covers nothing; only a search of about 460 steps shows it.
    monkeypatch.setattr(policy_module, "GLOB_INTERSECT_REQUEST_BUDGET", 700)

    assert policy.strict_patterns_covering_each(
        ["data/[0-9]", "data/[0-9]", "data/*.txt"]
    ) == ((), ("data/*.json",), ())


def test_the_spawn_override_diagnostic_is_bounded_in_total(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spawn-time diagnostic decides every ignored entry under one total
    budget, so a large or pathological ignored.yaml cannot hold the spawn up;
    once the budget runs out it errs toward reporting the entry overridden."""
    # Each entry ends like the strict glob but needs a letter that src/<digit>.py
    # never holds: disjoint, which only a search shows.
    entries = [f"**/*{c}*.py" for c in "abdefghijklmnoqtuvwxz"]
    (root / ".coherence" / "tracked.yaml").write_text("- src/**/*.py\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- 'src/[0-9].py'\n")
    (root / ".coherence" / "ignored.yaml").write_text(_quoted_yaml(entries))

    started = time.monotonic()
    policy = TrackedArtifactPolicy.load(root)
    assert time.monotonic() - started < 5.0
    assert policy.ignored_patterns_overridden_by_strict() == ()

    # Each entry takes about 855 steps, all 21 about 18,000: a budget per
    # entry would decide every one of them.
    monkeypatch.setattr(policy_module, "GLOB_INTERSECT_DIAGNOSTIC_BUDGET", 5_000)
    overridden = TrackedArtifactPolicy.load(root).ignored_patterns_overridden_by_strict()
    assert 0 < len(overridden) < len(entries)
    assert overridden == tuple(entries[-len(overridden):])


def _wide_class_entry(ranges: int) -> str:
    """An untrack entry whose one class holds ``ranges`` 512-code-point ranges:
    short on the wire, but each range used to add 512 characters to the
    search alphabet, and every explored state walked the whole alphabet."""
    body = "".join(
        chr(0x100 + 600 * k) + "-" + chr(0x100 + 600 * k + 511) for k in range(ranges)
    )
    return "*[" + body + "]*"


def test_a_wide_class_range_untrack_entry_is_decided_quickly(root: Path) -> None:
    """A /policy/untrack entry of 60 wide class ranges used to hold the
    handler for over 20 s and was then accepted (and replayed at every spawn
    from ignored.yaml). A class too wide to enumerate cheaply now counts as
    any one character, so the search is bounded and the answer errs toward
    "covers": the entry is refused, never accepted unchecked."""
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- 'data/*.json'\n")
    policy = TrackedArtifactPolicy.load(root)

    started = time.monotonic()
    covering = policy.strict_patterns_covering(_wide_class_entry(60))

    assert time.monotonic() - started < 1.0
    assert covering == ("data/*.json",)


def test_a_class_too_wide_to_enumerate_is_any_one_character() -> None:
    """Every class member joins the search alphabet, so a class over the
    member limit is approximated as any one character, in either polarity:
    it accepts more than the class, never less."""
    assert _glob_tokens("[\u0100-\u01ff]") == [(_ANY_ONE,)]
    assert _glob_tokens("[!\u0100-\u01ff]") == [(_ANY_ONE,)]
    assert _glob_tokens("[a-c]") == [(_CLASS, False, frozenset("abc"))]


def test_an_entry_of_many_unterminated_brackets_is_tokenized_quickly() -> None:
    """An unterminated ``[`` is a literal ``[``. Finding that out used to scan
    the rest of the pattern one character at a time, once per ``[``: 5,000 of
    them took over a second, outside every search budget."""
    started = time.monotonic()
    tokens = _glob_tokens("[" * 5_000)

    assert time.monotonic() - started < 0.25
    assert tokens == [(_LIT, "[")] * 5_000


def test_an_untrack_entry_of_many_distinct_characters_is_decided_quickly(
    root: Path,
) -> None:
    """Literal characters join the search alphabet too, and each explored
    state is charged for every character it tries. Charged per new state
    only, an entry of 400 distinct characters took over 3 s."""
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- 'data/*.json'\n")
    policy = TrackedArtifactPolicy.load(root)
    entry = "*" + "".join(chr(0x4E00 + i) for i in range(400)) + "*"

    started = time.monotonic()
    policy.strict_patterns_covering(entry)

    assert time.monotonic() - started < 1.0


def test_the_search_is_charged_at_the_size_of_each_state(root: Path) -> None:
    """Each transition is charged at the size of the state it leaves: an entry
    whose states grow with every run (``*ab*ab*...``) runs out of its budget
    in about 0.1 s. Charged one step per transition, the same budget lets it
    run for seconds."""
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- 'data/*.json'\n")
    policy = TrackedArtifactPolicy.load(root)

    started = time.monotonic()
    policy.strict_patterns_covering("*" + "ab*" * 300)

    assert time.monotonic() - started < 1.0


def test_an_ordinary_multi_segment_untrack_is_not_refused(root: Path) -> None:
    """Entries that cover no strict path are accepted. Each needs a search:
    the three-segment ``**/archive/*/*/*.md`` about 65,000 steps against one
    strict glob, the two-segment ``**/old/*/*.json`` about 550,000 against 45.
    Both used to run out of their budgets and be refused, naming strict
    patterns they do not cover."""
    strict = ["**/plans/*/*/*.md"] + [f"**/svc{i}/*/*.json" for i in range(45)]
    (root / ".coherence" / "tracked.yaml").write_text(_quoted_yaml(["**/*.md", "**/*.json"]))
    (root / ".coherence" / "strict_mode.yaml").write_text(_quoted_yaml(strict))
    policy = TrackedArtifactPolicy.load(root)

    assert policy.strict_patterns_covering_each(
        ["**/archive/*/*/*.md", "**/old/*/*.json"]
    ) == ((), ())
    # Control: an entry that does cover one strict glob names exactly that one.
    assert policy.strict_patterns_covering("**/svc7/a/*.json") == ("**/svc7/*/*.json",)


_BACKTRACKING = "**" * 12 + "Z"


def test_a_double_star_heavy_ignore_entry_does_not_stall_the_matcher(root: Path) -> None:
    """The ``**`` matcher's regex backtracks into every run: with an accepted
    ``'**' * 12 + 'Z'`` in ignored.yaml, every ``is_tracked`` call, on every
    hook, took several seconds. A pattern of that many runs is matched in
    time linear in the path."""
    (root / ".coherence" / "ignored.yaml").write_text(_quoted_yaml([_BACKTRACKING]))

    started = time.monotonic()
    policy = TrackedArtifactPolicy.load(root)
    assert policy.is_tracked("docs/plans/feature-y.md")

    assert time.monotonic() - started < 1.0


def test_a_literal_strict_path_is_matched_in_linear_time(root: Path) -> None:
    """A literal strict path is decided by matching the entry against it. With
    the entry's backtracking regex, a 30-character untrack entry held the
    handler for several seconds, and the same entry in ignored.yaml held the
    spawn before the port was bound."""
    entry = "**" * 12 + "Z**.md"  # its literal ends nest with the strict path's
    (root / ".coherence" / "strict_mode.yaml").write_text("- docs/plans/feature-x.md\n")
    (root / ".coherence" / "ignored.yaml").write_text(_quoted_yaml([entry]))

    started = time.monotonic()
    policy = TrackedArtifactPolicy.load(root)
    assert policy.strict_patterns_covering(entry) == ()
    assert policy.ignored_patterns_overridden_by_strict() == ()

    assert time.monotonic() - started < 1.0


_SEARCH_STEPS_SCRIPT = """
from ccs.adapters.claude_code.policy import _SearchBudget, _glob_tokens, _globs_intersect
budget = _SearchBudget(10**9)
_globs_intersect([_glob_tokens("**/*a*b*c*"), _glob_tokens("src/**/*.py")], budget)
print(10**9 - budget.remaining)
"""


def test_the_search_explores_in_the_same_order_in_every_process() -> None:
    """A set's iteration order is hash-randomized per process, and the order
    decides how far a bounded search gets before it finds a shared path, so
    which strict patterns a refusal names could change from one coordinator
    run to the next. The search alphabet is sorted: the steps it takes are
    the same under every hash seed."""
    steps = {
        subprocess.run(
            [sys.executable, "-c", _SEARCH_STEPS_SCRIPT],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for seed in ("1", "2", "3")
    }
    assert len(steps) == 1, steps


@pytest.mark.parametrize("shape", ["globs", "literals"])
def test_an_ordinary_untrack_is_not_refused_on_a_large_strict_set(
    root: Path, shape: str
) -> None:
    """An entry that covers no strict path is accepted however many strict
    patterns are live. One search budget used to be shared across every
    strict pattern in the call, and running out marked every pattern not yet
    decided as covering: 40 strict globs refused ``**/*.log`` naming 30 of
    them, and 60 literal strict paths refused it naming 3."""
    if shape == "globs":
        strict = [f"svc{i}/**" for i in range(40)]
        tracked = [f"svc{i}/state/*.json" for i in range(40)]
    else:
        strict = tracked = [f"pkg{i}/deep/nested/dir/manifest.json" for i in range(60)]
    (root / ".coherence" / "tracked.yaml").write_text(_quoted_yaml(tracked))
    (root / ".coherence" / "strict_mode.yaml").write_text(_quoted_yaml(strict))
    policy = TrackedArtifactPolicy.load(root)

    assert policy.strict_patterns_covering("**/*.log") == ()
    assert policy.strict_patterns_covering("*.tmp") == ()
    # Control: an entry that does cover one strict path names exactly that one.
    hit = "svc7/state/a.json" if shape == "globs" else "pkg7/deep/nested/dir/manifest.json"
    assert policy.strict_patterns_covering(hit) == (strict[7],)


def test_a_hot_reload_skips_the_spawn_time_override_diagnostic(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ignored-over-strict warning is a spawn-time diagnostic; a
    /policy/track or /policy/untrack reload does not recompute it."""
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- data/**\n")
    (root / ".coherence" / "ignored.yaml").write_text("- '**'\n")
    live = TrackedArtifactPolicy.load(root)
    calls: list[int] = []
    monkeypatch.setattr(
        TrackedArtifactPolicy,
        "ignored_patterns_overridden_by_strict",
        lambda self: calls.append(1) or (),
    )

    reloaded = live.reloaded()

    assert calls == []
    assert reloaded.is_strict_mode("data/a.txt")


def test_reload_never_narrows_strict_enforcement(root: Path) -> None:
    """A hot reload (track / untrack) keeps every strict and user-added
    pattern the live policy carries, so hand-removing one from disk cannot end
    enforcement mid-run; additions on disk still take effect."""
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- data/**\n")
    live = TrackedArtifactPolicy.load(root)
    (root / ".coherence" / "tracked.yaml").write_text("- extra/**\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- extra/**\n")

    reloaded = live.reloaded()

    assert reloaded.is_strict_mode("data/a.txt"), "a strict path stayed enforced"
    assert reloaded.is_strict_mode("extra/b.txt"), "an addition on disk took effect"
    assert reloaded.user_added_patterns == ("data/**", "extra/**")
    assert reloaded.strict_mode_paths == ("data/**", "extra/**")


def test_reload_without_strict_patterns_reads_the_files_verbatim(root: Path) -> None:
    (root / ".coherence" / "tracked.yaml").write_text("- data/**\n")
    live = TrackedArtifactPolicy.load(root)
    (root / ".coherence" / "tracked.yaml").write_text("- extra/**\n")

    reloaded = live.reloaded()

    assert reloaded.user_added_patterns == ("extra/**",)
    assert not reloaded.is_tracked("data/a.txt")


def test_strict_mode_for_path_with_invalid_normalization(root: Path) -> None:
    """Paths the normalizer rejects (absolute, traversal) never match
    strict-mode even when they would otherwise be in the glob set."""
    (root / ".coherence" / "tracked.yaml").write_text("- a/b/c.md\n")
    (root / ".coherence" / "strict_mode.yaml").write_text("- a/b/c.md\n")
    policy = TrackedArtifactPolicy.load(root)

    assert policy.is_strict_mode("a/b/c.md")  # sanity
    assert not policy.is_strict_mode("/absolute/a/b/c.md")
    assert not policy.is_strict_mode("../escape/a/b/c.md")


# --------------------------------------------------------------------
# Counting + threshold warning (KTD-O footgun guard)
# --------------------------------------------------------------------


def test_count_strict_mode_matches(root: Path) -> None:
    """count_strict_mode_matches over a candidate iterable."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- docs/plans/**/*.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    candidates = [
        "docs/plans/a.md",                  # tracked + matches strict glob
        "docs/plans/sub/b.md",              # tracked + matches strict glob
        "CLAUDE.md",                        # tracked but NOT in strict glob
        "src/main.py",                      # untracked
        "docs/brainstorms/c.md",            # tracked but NOT in strict glob
    ]
    assert policy.count_strict_mode_matches(candidates) == 2


def test_warn_if_strict_threshold_exceeded_one_shot(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The threshold warning fires once per policy instance — subsequent
    calls return False without re-emitting the WARNING (avoids log spam on
    repeated /status calls)."""
    (root / ".coherence" / "strict_mode.yaml").write_text("- '**'\n")
    policy = TrackedArtifactPolicy.load(root)

    # Need a tracked-AND-strict-matching candidate set above threshold.
    candidates = [f"docs/plans/p-{i:03d}.md" for i in range(60)]
    for p in candidates[:3]:
        assert policy.is_strict_mode(p), f"setup precondition: {p} must be strict"

    with caplog.at_level(logging.WARNING):
        emitted_first = policy.warn_if_strict_threshold_exceeded(candidates)
        emitted_second = policy.warn_if_strict_threshold_exceeded(candidates)

    assert emitted_first is True, "first call above threshold should emit"
    assert emitted_second is False, "second call should NOT re-emit (one-shot)"
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "strict_mode_paths matches" in warning_records[0].message
    assert "(> threshold 50)" in warning_records[0].message


def test_warn_if_strict_threshold_below_threshold_no_warning(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Threshold guard stays silent for small strict-mode opt-ins (the
    common case: operator opts in 1-5 specific files)."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- docs/plans/feature-x.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    candidates = ["docs/plans/feature-x.md", "CLAUDE.md", "src/main.py"]
    with caplog.at_level(logging.WARNING):
        emitted = policy.warn_if_strict_threshold_exceeded(candidates)
    assert emitted is False
    assert not any(
        "strict_mode_paths matches" in r.message
        for r in caplog.records
    )


def test_warn_if_strict_threshold_custom_threshold(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Caller can override the default threshold (e.g., a stricter
    operator-installed coordinator could pass threshold=10)."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- docs/plans/**/*.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    candidates = [f"docs/plans/p-{i:03d}.md" for i in range(12)]
    with caplog.at_level(logging.WARNING):
        emitted = policy.warn_if_strict_threshold_exceeded(
            candidates, threshold=10
        )
    assert emitted is True


def test_warn_if_strict_threshold_constant_value() -> None:
    """Default threshold is 50 per the plan. Bound to the operator-facing
    docs in the README + configuration.md (Unit 6) so tests guard against
    silent constant drift."""
    assert STRICT_MODE_PATH_WARN_THRESHOLD == 50


# --------------------------------------------------------------------
# YAML loading + path-traversal guard parity with tracked/ignored
# --------------------------------------------------------------------


def test_missing_strict_mode_yaml_loads_empty(root: Path) -> None:
    """No strict_mode.yaml file → strict_mode_paths is empty tuple."""
    policy = TrackedArtifactPolicy.load(root)
    assert policy.strict_mode_paths == ()


def test_strict_mode_yaml_path_traversal_rejected(root: Path) -> None:
    """Same path-traversal guard as tracked/ignored: absolute paths and
    ``..`` components rejected at load time, surface in rejected_patterns."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- /absolute/path.md\n"
        "- ../escape/path.md\n"
        "- legit/path.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)
    assert policy.strict_mode_paths == ("legit/path.md",)
    rejected_patterns = [p for p, _ in policy.rejected_patterns]
    assert "/absolute/path.md" in rejected_patterns
    assert "../escape/path.md" in rejected_patterns


def test_strict_mode_yaml_malformed_degrades_gracefully(
    root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed YAML → empty strict_mode_paths + WARNING; never crashes
    coordinator startup (the v0.1.1 graceful-degradation contract)."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "this is: not [valid: yaml\n"
    )
    with caplog.at_level(logging.WARNING):
        policy = TrackedArtifactPolicy.load(root)
    assert policy.strict_mode_paths == ()
    # No assertion on caplog message — tracked-yaml warning shape varies.


# --------------------------------------------------------------------
# Summary surface
# --------------------------------------------------------------------


def test_summary_exposes_strict_mode_pattern_count(root: Path) -> None:
    """The /status policy_summary block surfaces strict-mode pattern count
    so operators can see at a glance how much of the workspace is opted in."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- CLAUDE.md\n- docs/plans/feature-x.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)

    summary = policy.summary(include_patterns=True)
    assert summary["strict_mode_pattern_count"] == 2
    # The patterns themselves, so a client can check the globs it declared
    # against the coordinator's policy instead of a count (#190).
    assert summary["strict_mode_patterns"] == ["CLAUDE.md", "docs/plans/feature-x.md"]
    assert summary["tracked_patterns"] == list(DEFAULT_TRACKED_PATTERNS)
    assert summary["ignored_patterns"] == []
    # Existing summary keys preserved.
    assert "default_pattern_count" in summary
    assert "user_added_pattern_count" in summary
    assert "ignored_pattern_count" in summary
    assert "rejected_pattern_count" in summary


def test_summary_strict_mode_pattern_count_zero_when_unset(root: Path) -> None:
    policy = TrackedArtifactPolicy.load(root)
    assert policy.summary()["strict_mode_pattern_count"] == 0


# --------------------------------------------------------------------
# Pre-compiled glob cache covers strict_mode_paths too
# --------------------------------------------------------------------


def test_strict_mode_glob_uses_compiled_cache(root: Path) -> None:
    """PERF-2: pre-compiled glob cache covers strict_mode_paths the same way
    it covers tracked + ignored + user_added. No string-build overhead per
    is_strict_mode call for ``**`` patterns."""
    (root / ".coherence" / "strict_mode.yaml").write_text(
        "- docs/plans/**/*.md\n"
    )
    policy = TrackedArtifactPolicy.load(root)
    # The ``**`` pattern lives in the cache.
    assert "docs/plans/**/*.md" in policy._compiled_patterns
    # Sanity: matcher works.
    assert policy.is_strict_mode("docs/plans/deep/feature.md")
