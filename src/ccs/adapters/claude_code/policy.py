# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tracked-artifact policy — decide whether a given path is coordinated.

Per plan KTD-8: the coordinator watches a narrow set of paths by default
(``CLAUDE.md``, ``AGENTS.md``, anything under ``docs/{specs,plans,brainstorms}/``,
``plan.md|task.md|spec.md`` at any depth). Users can add patterns via
``.coherence/tracked.yaml`` and remove patterns via ``.coherence/ignored.yaml``.

Design constraints:

- **Loaded once at coordinator startup** (per scope-guardian review). Hot-reload
  on mtime was removed as speculative optimization. Pattern changes take effect
  on coordinator restart, which is cheap given idle-shutdown + lazy re-spawn.
- **Path-traversal guard** (per security-lens review): patterns containing
  ``..`` components or starting with ``/`` are rejected with WARNING and skipped.
  A single bad entry shouldn't disable the whole policy.
- **Cross-language friendly defaults**: defaults shouldn't false-positive in
  Node, Rust, Django, or other-language repos. Unit 8 1000-path benchmark covers.
- All policy decisions key on **parent-repo-relative** paths (KTD-7 normalization
  happens upstream in the hook handler).
- **A strict path stays enforced for the coordinator's lifetime** (#261). A
  path that is tracked and matches a strict pattern is never untracked by an
  ignored pattern (strict wins over ignore, at load and after every reload),
  the ``/policy/track`` and ``/policy/untrack`` hot reloads
  (:meth:`TrackedArtifactPolicy.reloaded`) never drop a strict or user-added
  pattern while strict patterns are live, and ``/policy/untrack`` refuses an
  entry that would cover a strict path (:meth:`TrackedArtifactPolicy.strict_patterns_covering`).
  The one way to stop enforcing a strict path is to restart the coordinator
  without its strict entry.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import yaml

logger = logging.getLogger(__name__)


STRICT_MODE_PATH_WARN_THRESHOLD: int = 50
"""v0.2 KTD-O footgun guard: if ``strict_mode_paths`` matches more than this
many tracked artifacts, the policy emits a one-shot WARNING via
:meth:`TrackedArtifactPolicy.warn_if_strict_threshold_exceeded`. The caller
(coordinator service after artifact registration / status snapshot) invokes
the check at runtime — the policy module itself cannot enumerate tracked
artifacts at load time because the artifact set lives in SQLite, not on disk
as concrete paths. Guards against an operator typing ``**`` in
``strict_mode.yaml`` and silently locking down the whole workspace."""


DEFAULT_TRACKED_PATTERNS: tuple[str, ...] = (
    # Repo-root coordination files
    "CLAUDE.md",
    "AGENTS.md",
    # Spec/plan/brainstorm directories
    "docs/specs/**/*.md",
    "docs/plans/**/*.md",
    "docs/brainstorms/**/*.md",
    # Conventional coordination filenames at any depth
    "**/plan.md",
    "**/task.md",
    "**/spec.md",
)
"""Patterns the policy module ships with by default. Cross-language safe —
the 1000-path benchmark in Unit 8 verifies 0 false positives across Node,
Rust, Django, and other-ecosystem path samples."""


UNTRACK_STRICT_PATH_REASON: str = "untrack_strict_path"
"""Typed reason ``POST /policy/untrack`` answers (HTTP 409, nothing written)
when an entry would untrack a path the live policy puts in strict mode (#261).
The body names, per refused entry, the strict patterns it covers. A client
classifies the refusal by equality on this value, never by the ``error`` text.
Untracking a strict path takes a coordinator restart without the strict entry:
a hot reload never narrows strict enforcement."""


def _compile_glob_pattern(pattern: str) -> re.Pattern[str] | None:
    """PERF-2 / finding #16: pre-compile a ``**``-containing glob pattern into
    a re.Pattern at construction time. Returns None for patterns without ``**``
    (those use fnmatch at match-time with no string-build overhead)."""
    if "**" not in pattern:
        return None
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                parts.append(".*")
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            else:
                parts.append("[^/]*")
                i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


@dataclass
class TrackedArtifactPolicy:
    """Decide whether a parent-repo-relative path is coordinated.

    Construct via :meth:`load`, never directly — the loader applies the
    path-traversal guard and reads optional YAML overrides.
    """

    coordinator_root: Path
    tracked_patterns: tuple[str, ...] = DEFAULT_TRACKED_PATTERNS
    ignored_patterns: tuple[str, ...] = ()
    user_added_patterns: tuple[str, ...] = field(default_factory=tuple)
    strict_mode_paths: tuple[str, ...] = field(default_factory=tuple)
    """v0.2 KTD-O: glob patterns that opt selected tracked artifacts into
    strict mode. An artifact is in strict mode iff it is in the tracked set
    AND matches at least one strict_mode_paths glob. Empty tuple → no
    artifacts in strict mode (back-compatible with v0.1 / v0.1.1, where
    every tracked artifact is warn-mode by default). Loaded from
    ``.coherence/strict_mode.yaml`` (parallel to tracked.yaml /
    ignored.yaml); same path-traversal guard applies."""
    rejected_patterns: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Patterns rejected by the path-traversal guard, with reason. Surfaced
    by :meth:`rejected` for status/debug visibility."""

    # PERF-2 / finding #16: pre-compiled regex cache for ``**`` patterns.
    # Built in __post_init__ so the cost is paid exactly once at load time.
    _compiled_patterns: dict[str, re.Pattern[str]] = field(
        default_factory=dict, repr=False, compare=False
    )
    # KTD-O one-shot threshold-warning guard. Reset on every reload() so a
    # post-track policy swap can re-emit the warning if the operator widened
    # the strict-mode glob in a hot-reload path.
    _strict_threshold_warning_emitted: bool = field(
        default=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Pre-compile all ``**``-containing patterns across all pattern sets."""
        for patterns in (
            self.tracked_patterns,
            self.ignored_patterns,
            self.user_added_patterns,
            self.strict_mode_paths,
        ):
            for p in patterns:
                if p not in self._compiled_patterns:
                    compiled = _compile_glob_pattern(p)
                    if compiled is not None:
                        self._compiled_patterns[p] = compiled

    @classmethod
    def load(cls, coordinator_root: Path | str) -> "TrackedArtifactPolicy":
        """Load policy: defaults + ``.coherence/tracked.yaml`` opt-in +
        ``.coherence/ignored.yaml`` opt-out + ``.coherence/strict_mode.yaml``
        v0.2 opt-in (KTD-O). All YAML files are optional. Patterns failing
        the path-traversal guard are rejected with WARNING."""
        policy = cls._read(coordinator_root)
        overridden = policy.ignored_patterns_overridden_by_strict()
        if overridden:
            logger.warning(
                "ignored pattern(s) %s cover paths in strict mode; strict wins, so "
                "those paths stay tracked and enforced (#261). To stop enforcing a "
                "strict path, remove its entry from .coherence/strict_mode.yaml and "
                "restart the coordinator.",
                ", ".join(overridden),
            )
        return policy

    @classmethod
    def _read(cls, coordinator_root: Path | str) -> "TrackedArtifactPolicy":
        """Read the policy files without :meth:`load`'s strict-override
        diagnostic, which hot reloads (:meth:`reloaded`) skip."""
        root = Path(coordinator_root).resolve()
        rejected: list[tuple[str, str]] = []

        added = _load_yaml_patterns(root / ".coherence" / "tracked.yaml", rejected)
        ignored = _load_yaml_patterns(root / ".coherence" / "ignored.yaml", rejected)
        strict = _load_yaml_patterns(root / ".coherence" / "strict_mode.yaml", rejected)
        return cls(
            coordinator_root=root,
            tracked_patterns=DEFAULT_TRACKED_PATTERNS,
            ignored_patterns=tuple(ignored),
            user_added_patterns=tuple(added),
            strict_mode_paths=tuple(strict),
            rejected_patterns=tuple(rejected),
        )

    def _matches_tracked(self, normalized: str) -> bool:
        # PERF-2 / finding #16: pass pre-compiled cache so _glob_match skips
        # the string-build loop for ``**`` patterns.
        cache = self._compiled_patterns
        return (
            matches_any(normalized, self.tracked_patterns, cache)
            or matches_any(normalized, self.user_added_patterns, cache)
        )

    def _matches_strict(self, normalized: str) -> bool:
        return bool(self.strict_mode_paths) and matches_any(
            normalized, self.strict_mode_paths, self._compiled_patterns
        )

    def is_tracked(self, parent_relative_path: str) -> bool:
        """Return True if the given parent-repo-relative path is coordinated.

        Algorithm: path is tracked if it matches any default OR user-added
        pattern, AND does not match any ignored pattern — unless it also
        matches a strict pattern: strict wins over ignore (#261). An ignore
        pattern, however broad (``**``), and however it arrived (a leftover
        ``ignored.yaml`` at spawn or ``/policy/untrack`` at runtime), never
        silently drops a strict path onto the untracked fast path, where a read
        reports version 0 and a CAS commit is accepted at any expected version.
        For every path no strict pattern matches, ignore wins ties as before.
        """
        normalized = _normalize_relative(parent_relative_path)
        if normalized is None:
            # Absolute path or .. traversal — never tracked.
            return False
        if not self._matches_tracked(normalized):
            return False
        if not matches_any(normalized, self.ignored_patterns, self._compiled_patterns):
            return True
        return self._matches_strict(normalized)

    def is_strict_mode(self, parent_relative_path: str) -> bool:
        """v0.2 KTD-O: return True iff the path is in strict mode.

        Strict-mode contract: a path is in strict mode iff it is in the
        tracked set (a default or user-added pattern) AND matches at least one
        strict_mode_paths glob. Intersection semantics — never applies to a
        path no tracked pattern matches, even when a strict_mode pattern would
        match it. An ignored pattern does not take a path out of strict mode
        (see :meth:`is_tracked`). Empty strict_mode_paths short-circuits to
        False (back-compat with v0.1.1).

        Hooks gate strict-mode behavior (``permissionDecision: "deny"``) on
        this method only; the v0.1.1 warn-mode allow path is preserved for
        every artifact that returns False here."""
        if not self.strict_mode_paths:
            # Fast path — no strict-mode opt-in; v0.1.1 behavior preserved.
            return False
        normalized = _normalize_relative(parent_relative_path)
        if normalized is None:
            return False
        return self._matches_tracked(normalized) and self._matches_strict(normalized)

    def strict_patterns_covering(self, pattern: str) -> tuple[str, ...]:
        """The strict patterns under which ``pattern`` — a path or a glob, as
        ``/policy/untrack`` receives it — matches at least one path this policy
        puts in strict mode: some path matches ``pattern``, a strict pattern,
        and a tracked (default or user-added) pattern at once.

        Decided on the glob languages themselves (:func:`globs_intersect`),
        so an entry spelled differently from the strict pattern is still
        caught: ``data/**`` and ``**`` both cover a strict ``data/*.json``;
        ``notes/**`` covers nothing when no tracked pattern reaches
        ``notes/``. Character classes are matched approximately, erring
        toward "covers" (a refusal, never a silent untrack), and so is a
        search that exceeds :data:`GLOB_INTERSECT_STATE_BUDGET` states, shared
        across the whole call: the strict pattern being decided and every one
        not yet decided then count as covered. Empty when no strict path is
        covered, which is always the case with no strict patterns."""
        if not self.strict_mode_paths:
            return ()
        tracked = tuple(dict.fromkeys((*self.user_added_patterns, *self.tracked_patterns)))
        budget = _StateBudget(GLOB_INTERSECT_STATE_BUDGET)
        covering: list[str] = []
        for index, strict in enumerate(self.strict_mode_paths):
            try:
                if _globs_intersect((pattern, strict), budget) and any(
                    _globs_intersect((pattern, strict, t), budget) for t in tracked
                ):
                    covering.append(strict)
            except _BudgetExhausted:
                covering.extend(self.strict_mode_paths[index:])
                break
        return tuple(covering)

    def ignored_patterns_overridden_by_strict(self) -> tuple[str, ...]:
        """Ignored patterns that cover a strict path, which strict therefore
        overrides (:meth:`is_tracked`). Logged once at :meth:`load` (spawn),
        not on hot reloads, so an operator learns that the ignore entry does
        not untrack those paths. Bounded like :meth:`strict_patterns_covering`,
        so a pathological entry may be reported as overridden when it is not."""
        return tuple(p for p in self.ignored_patterns if self.strict_patterns_covering(p))

    def reloaded(self) -> "TrackedArtifactPolicy":
        """Re-read the policy files for a hot reload (``/policy/track``,
        ``/policy/untrack``) without narrowing strict enforcement (#261).

        Ignored patterns are taken from disk as they are; they cannot untrack
        a strict path anyway. While this policy has strict patterns, every
        strict and user-added pattern it carries is kept even if the file on
        disk no longer lists it — strict mode needs both — so a hand-edited
        ``strict_mode.yaml`` or ``tracked.yaml`` followed by any track or
        untrack cannot end enforcement mid-run. Additions on disk still take
        effect. Without strict patterns a reload reads the files verbatim, as
        before."""
        fresh = TrackedArtifactPolicy._read(self.coordinator_root)
        if not self.strict_mode_paths:
            return fresh
        return TrackedArtifactPolicy(
            coordinator_root=fresh.coordinator_root,
            tracked_patterns=fresh.tracked_patterns,
            ignored_patterns=fresh.ignored_patterns,
            user_added_patterns=_union(self.user_added_patterns, fresh.user_added_patterns),
            strict_mode_paths=_union(self.strict_mode_paths, fresh.strict_mode_paths),
            rejected_patterns=fresh.rejected_patterns,
        )

    def count_strict_mode_matches(self, candidate_paths: Iterable[str]) -> int:
        """Count how many of ``candidate_paths`` are in strict mode.

        Used by the coordinator service (after registry status snapshot) to
        feed :meth:`warn_if_strict_threshold_exceeded` without forcing the
        policy module to know about the registry."""
        return sum(1 for p in candidate_paths if self.is_strict_mode(p))

    def warn_if_strict_threshold_exceeded(
        self,
        candidate_paths: Iterable[str],
        threshold: int = STRICT_MODE_PATH_WARN_THRESHOLD,
    ) -> bool:
        """v0.2 KTD-O footgun guard. One-shot WARNING emitter.

        Returns True if a warning was emitted (count > threshold) on this
        invocation. Subsequent calls return False until policy reload —
        guards against log-spam under repeated /status calls. The caller is
        the coordinator service; it invokes this after enumerating tracked
        artifacts so the count is concrete rather than glob-derived."""
        if self._strict_threshold_warning_emitted:
            return False
        count = self.count_strict_mode_matches(candidate_paths)
        if count > threshold:
            logger.warning(
                "strict_mode_paths matches %d tracked artifacts (> threshold %d). "
                "Verify .coherence/strict_mode.yaml is not over-broad — e.g., "
                "a literal '**' pattern silently opts the entire workspace into "
                "strict mode (permissionDecision: 'deny' on stale reads).",
                count,
                threshold,
            )
            self._strict_threshold_warning_emitted = True
            return True
        return False

    def rejected(self) -> Sequence[tuple[str, str]]:
        """Return (pattern, reason) pairs rejected by the path-traversal guard."""
        return self.rejected_patterns

    def summary(self, *, include_patterns: bool = False) -> dict[str, object]:
        """Compact summary for the ``/status`` endpoint and CLI display.

        The counts are always present. With ``include_patterns`` the four
        pattern lists (``tracked_patterns``, ``user_added_patterns``,
        ``ignored_patterns``, ``strict_mode_patterns``) are added: they are what
        a client checks its own declared globs against, and ``/status`` asks for
        them only in the operator view, since a pattern list is the operator's
        directory layout. A count alone let a volume whose globs differed from
        the spawner's pass its attach check (#190)."""
        summary: dict[str, object] = {
            "coordinator_root": str(self.coordinator_root),
            "default_pattern_count": len(self.tracked_patterns),
            "user_added_pattern_count": len(self.user_added_patterns),
            "ignored_pattern_count": len(self.ignored_patterns),
            "strict_mode_pattern_count": len(self.strict_mode_paths),
            "rejected_pattern_count": len(self.rejected_patterns),
        }
        if include_patterns:
            summary.update({
                "tracked_patterns": list(self.tracked_patterns),
                "user_added_patterns": list(self.user_added_patterns),
                "ignored_patterns": list(self.ignored_patterns),
                "strict_mode_patterns": list(self.strict_mode_paths),
            })
        return summary


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _union(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
    """Order-preserving union: ``first``, then what ``second`` adds."""
    return tuple(dict.fromkeys((*first, *second)))


# Glob tokens, mirroring _glob_match exactly. A ``**`` pattern is compiled by
# _compile_glob_pattern (``**`` -> any run, a following ``/`` swallowed;
# ``*`` -> a run without ``/``; ``?`` -> one char but ``/``; every other char,
# ``[`` included, literal). Any other pattern goes to fnmatch.fnmatchcase
# (``*`` -> any run, ``/`` included; ``?`` -> any one char; ``[...]`` a class).
_ANY_RUN = "any_run"
_SEG_RUN = "seg_run"
_ANY_ONE = "any_one"
_SEG_ONE = "seg_one"
_LIT = "lit"
_CLASS = "class"
_RANGE_EXPAND_LIMIT = 512


def _glob_tokens(pattern: str) -> list[tuple]:
    tokens: list[tuple] = []
    i = 0
    if "**" in pattern:
        while i < len(pattern):
            c = pattern[i]
            if c == "*" and i + 1 < len(pattern) and pattern[i + 1] == "*":
                tokens.append((_ANY_RUN,))
                i += 2
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            elif c == "*":
                tokens.append((_SEG_RUN,))
                i += 1
            elif c == "?":
                tokens.append((_SEG_ONE,))
                i += 1
            else:
                tokens.append((_LIT, c))
                i += 1
        return tokens
    while i < len(pattern):
        c = pattern[i]
        i += 1
        if c == "*":
            tokens.append((_ANY_RUN,))
        elif c == "?":
            tokens.append((_ANY_ONE,))
        elif c == "[":
            # fnmatch.translate: '!' negates, a leading ']' is literal, and an
            # unterminated '[' is a literal '['.
            j = i
            if j < len(pattern) and pattern[j] == "!":
                j += 1
            if j < len(pattern) and pattern[j] == "]":
                j += 1
            while j < len(pattern) and pattern[j] != "]":
                j += 1
            if j >= len(pattern):
                tokens.append((_LIT, "["))
                continue
            body = pattern[i:j]
            i = j + 1
            negated = body.startswith("!")
            if negated:
                body = body[1:]
            members: set[str] = set()
            too_wide = False
            k = 0
            while k < len(body):
                if k + 2 < len(body) and body[k + 1] == "-":
                    lo, hi = body[k], body[k + 2]
                    if ord(hi) - ord(lo) <= _RANGE_EXPAND_LIMIT:
                        members.update(chr(o) for o in range(ord(lo), ord(hi) + 1))
                    else:
                        too_wide = True
                        # Negated: excluding only the ends excludes too little,
                        # so the class accepts more than it does (approximate
                        # toward "intersects").
                        members.update((lo, hi))
                    k += 3
                else:
                    members.add(body[k])
                    k += 1
            if too_wide and not negated:
                # A positive class with a range too wide to expand accepts any
                # one character here — again more than it does, never less.
                tokens.append((_ANY_ONE,))
            else:
                tokens.append((_CLASS, negated, frozenset(members)))
        else:
            tokens.append((_LIT, c))
    return tokens


def _token_accepts(token: tuple, ch: str | None) -> bool:
    """Whether ``token`` consumes ``ch``; ``None`` stands for every character
    no pattern names (not ``/``)."""
    kind = token[0]
    if kind in (_ANY_RUN, _ANY_ONE):
        return True
    if kind in (_SEG_RUN, _SEG_ONE):
        return ch != "/"
    if kind == _LIT:
        return ch == token[1]
    negated, members = token[1], token[2]
    return (ch not in members) if negated else (ch in members)


def _closure(states: frozenset[int], tokens: list[tuple]) -> frozenset[int]:
    out = set(states)
    stack = list(states)
    while stack:
        i = stack.pop()
        if i < len(tokens) and tokens[i][0] in (_ANY_RUN, _SEG_RUN) and i + 1 not in out:
            out.add(i + 1)
            stack.append(i + 1)
    return frozenset(out)


def _advance(states: frozenset[int], tokens: list[tuple], ch: str | None) -> frozenset[int]:
    nxt: set[int] = set()
    for i in states:
        if i >= len(tokens) or not _token_accepts(tokens[i], ch):
            continue
        nxt.add(i if tokens[i][0] in (_ANY_RUN, _SEG_RUN) else i + 1)
    return _closure(frozenset(nxt), tokens)


GLOB_INTERSECT_STATE_BUDGET: int = 2000
"""How many product states one :meth:`TrackedArtifactPolicy.strict_patterns_covering`
call may explore before it stops deciding and answers "covers". Ordinary
globs settle in a few dozen states; a star-heavy entry can need exponentially
many, and the check runs inside the ``/policy/untrack`` handler's deadline."""


class _BudgetExhausted(Exception):
    pass


class _StateBudget:
    __slots__ = ("remaining",)

    def __init__(self, states: int) -> None:
        self.remaining = states

    def spend(self) -> None:
        self.remaining -= 1
        if self.remaining < 0:
            raise _BudgetExhausted


def globs_intersect(*patterns: str, max_states: int | None = None) -> bool:
    """True when some non-empty path matches every one of ``patterns`` under
    this module's matching rules (:func:`matches_any`).

    A product of the patterns' automata searched over an alphabet of every
    character the patterns name, ``/``, and one stand-in for every other
    character. Exact for ``*``, ``**`` and ``?``; a class range wider than 512
    code points is approximated so the class accepts more, never less.
    Every approximation errs toward True, and so does ``max_states``: a search
    that visits more product states than that stops and answers True."""
    budget = _StateBudget(max_states) if max_states is not None else None
    try:
        return _globs_intersect(patterns, budget)
    except _BudgetExhausted:
        return True


def _globs_intersect(patterns: Sequence[str], budget: _StateBudget | None) -> bool:
    token_lists = [_glob_tokens(p) for p in patterns]
    alphabet: set[str | None] = {"/", None}
    for tokens in token_lists:
        for token in tokens:
            if token[0] == _LIT:
                alphabet.add(token[1])
            elif token[0] == _CLASS:
                alphabet.update(token[2])
    start = tuple(_closure(frozenset({0}), t) for t in token_lists)
    seen = {start}
    frontier = [start]
    while frontier:
        state = frontier.pop()
        for ch in alphabet:
            nxt = tuple(_advance(s, t, ch) for s, t in zip(state, token_lists))
            if not all(nxt):
                continue
            if all(len(t) in s for s, t in zip(nxt, token_lists)):
                return True
            if nxt not in seen:
                if budget is not None:
                    budget.spend()
                seen.add(nxt)
                frontier.append(nxt)
    return False


def _normalize_relative(p: str) -> str | None:
    """Return the path with leading ``./`` stripped, or None if the path
    is absolute or contains ``..`` components (defense-in-depth even
    though the hook handler also normalizes upstream)."""
    if not p:
        return None
    # P1 ce-review fix (kieran-python + correctness convergence): use
    # removeprefix, NOT lstrip. lstrip strips a SET of characters {".", "/"}
    # so ".env" → "env", ".gitignore" → "gitignore", silently making dotfile
    # patterns in tracked.yaml unmatchable. removeprefix strips the literal
    # "./" prefix only (Python 3.9+).
    cleaned = p.removeprefix("./")
    if not cleaned:
        # Pure "." or "./"; not a file path.
        return None
    if p.startswith("/"):
        return None
    if ".." in Path(cleaned).parts:
        return None
    return cleaned


def matches_any(
    path: str,
    patterns: Iterable[str],
    compiled: dict[str, re.Pattern[str]] | None = None,
) -> bool:
    """Glob-match path against a list of patterns. Uses ``fnmatch`` for
    ``*``/``?`` semantics; ``**`` is treated as zero-or-more path segments.

    PERF-2 / finding #16: ``compiled`` is an optional pre-compiled pattern
    cache (keyed by pattern string). When provided, ``**`` patterns skip the
    string-build loop and use the cached re.Pattern directly."""
    posix_path = path.replace("\\", "/")
    for pattern in patterns:
        if _glob_match(posix_path, pattern, compiled):
            return True
    return False


def _glob_match(
    path: str,
    pattern: str,
    compiled: dict[str, re.Pattern[str]] | None = None,
) -> bool:
    """Match a posix-style path against a glob pattern supporting ``**``.

    PERF-2 / finding #16: when ``compiled`` is provided, ``**`` patterns use
    the pre-compiled re.Pattern directly, skipping the string-build loop."""
    # fnmatch handles ``*`` (any chars in segment) and ``?`` (single char).
    # For ``**`` (any number of path segments), convert to a regex-equivalent.
    if "**" not in pattern:
        return fnmatch.fnmatchcase(path, pattern)
    # Fast path: use the pre-compiled pattern if available.
    if compiled is not None and pattern in compiled:
        return compiled[pattern].match(path) is not None
    # Slow path (called without a cache, e.g. from tests): build on the fly.
    parts: list[str] = []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "*":
            if i + 1 < len(pattern) and pattern[i + 1] == "*":
                parts.append(".*")
                i += 2
                # Skip trailing slash after `**/`
                if i < len(pattern) and pattern[i] == "/":
                    i += 1
            else:
                parts.append("[^/]*")
                i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    regex_str = "^" + "".join(parts) + "$"
    return re.match(regex_str, path) is not None


def _load_yaml_patterns(
    yaml_path: Path, rejected: list[tuple[str, str]]
) -> list[str]:
    """Read a YAML file containing a list of pattern strings. Apply the
    path-traversal guard. Returns the surviving patterns; mutates the
    ``rejected`` list with (pattern, reason) for each rejection.

    Missing file → empty list (not an error).
    Malformed YAML → empty list + WARNING (do not crash hooks).
    Non-list top-level → empty list + WARNING.
    """
    if not yaml_path.is_file():
        return []

    try:
        raw = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError as exc:
        logger.warning("malformed YAML at %s; falling back to defaults: %s", yaml_path, exc)
        return []
    except OSError as exc:
        logger.warning("could not read %s; falling back to defaults: %s", yaml_path, exc)
        return []

    if raw is None:
        return []
    if not isinstance(raw, list):
        logger.warning(
            "%s top-level must be a list of patterns; got %s. Ignoring.",
            yaml_path,
            type(raw).__name__,
        )
        return []

    surviving: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            rejected.append((str(item), f"non-string pattern ({type(item).__name__})"))
            logger.warning("rejecting non-string pattern in %s: %r", yaml_path, item)
            continue
        reason = _validate_pattern(item)
        if reason is not None:
            rejected.append((item, reason))
            logger.warning("rejecting pattern in %s (%s): %r", yaml_path, reason, item)
            continue
        surviving.append(item)
    return surviving


def _validate_pattern(pattern: str) -> str | None:
    """Path-traversal guard. Returns None if pattern is acceptable, else
    a short reason string."""
    if not pattern:
        return "empty pattern"
    if pattern.startswith("/"):
        return "absolute path"
    # Split on both unix and windows separators to be safe; we only support
    # unix-style patterns in v0.1 but reject windows-style traversal too.
    parts = pattern.replace("\\", "/").split("/")
    if ".." in parts:
        return "contains '..' (path traversal)"
    return None
