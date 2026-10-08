# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""File-path detection in Bash command strings.

Used by the ``/hooks/pre-bash`` handler (v0.1.1 KTD-N) to identify tracked
artifacts a Bash command would READ, so the coherence layer can surface
stale-read warnings for the multi-tool-routing case the v0.2 Phase 0
falsifiability experiment documented (see
``docs/probes/2026-05-19-ktd-e-falsifiability/REPORT.md`` H4).

False-negative bias is deliberate per v0.1.1 plan KTD-N:
*"better to miss a clever bypass than to warn on unrelated Bash invocations
and erode trust"*. The detector recognizes a curated set of common
file-reading commands + eval patterns. Adversarial bypass (obfuscation
via parameter expansion, command substitution, etc.) is OUT of scope.
"""
from __future__ import annotations

import re
import shlex
from typing import Callable, Iterable

# Read-only commands that take file-path positional args.
# Order doesn't matter; membership check only.
_TRACKED_READ_COMMANDS: frozenset[str] = frozenset({
    "cat",
    "less",
    "more",
    "head",
    "tail",
    "awk",
    "sed",
    "xargs",
    "grep",
    "rg",
    "ugrep",
    "wc",  # word/byte/line count — reads file contents
})

# Eval-pattern matchers — detect inline Python/Perl/Ruby code that may
# read files. Captures the quoted body for substring file-path scanning.
_TRACKED_EVAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'python3?\s+-c\s+(["\'])(.*?)\1', re.S),
    re.compile(r'perl\s+-e\s+(["\'])(.*?)\1', re.S),
    re.compile(r'ruby\s+-e\s+(["\'])(.*?)\1', re.S),
)

# Heuristic file-path token regex for eval-pattern bodies. Conservative —
# only matches "<name>.<ext>" shapes where extension is alphanumeric.
# False-negative bias: misses paths without extensions. Public: the write
# detector scans program bodies for the same token shape.
PATH_TOKEN_RE: re.Pattern[str] = re.compile(
    r'(?<![A-Za-z0-9_/.\-])([A-Za-z0-9_./\-]+\.[A-Za-z0-9]+)'
)


def detect_tracked_paths(
    command: str,
    is_tracked: Callable[[str], bool],
) -> list[str]:
    """Return tracked-artifact paths the command would read.

    Algorithm:
    1. Split on pipeline / sequence separators (``|``, ``;``, ``&``,
       ``&&``, ``||``).
    2. For each segment, tokenize via ``shlex`` (handles quoting).
    3. If the FIRST non-flag token is a tracked-read command name, treat
       all subsequent non-flag positional args as candidate file paths.
       Filter through ``is_tracked(path)``.
    4. Across the full command, scan eval patterns (``python -c "..."``)
       for substring file-path-shaped tokens; filter via ``is_tracked``.

    Returns a deduplicated list preserving first-occurrence order.

    Args:
        command: Raw Bash command string from CC's ``tool_input.command``.
        is_tracked: Callable that returns True if a parent-repo-relative
            path is tracked by the policy (e.g.,
            ``coordinator.policy.is_tracked``).

    Returns:
        List of tracked-artifact paths the command would read. Empty if
        none detected. False-negatives expected for adversarial bypass.
    """
    seen: set[str] = set()
    paths: list[str] = []

    def _add(path: str) -> None:
        if path in seen:
            return
        seen.add(path)
        paths.append(path)

    # Step 1+2+3: pipeline segments.
    for segment in _split_pipeline(command):
        for path in _detect_in_segment(segment, is_tracked):
            _add(path)

    # Step 4: eval patterns across the full command.
    for pattern in _TRACKED_EVAL_PATTERNS:
        for match in pattern.finditer(command):
            body = match.group(2)
            for token in PATH_TOKEN_RE.findall(body):
                if is_tracked(token):
                    _add(token)

    return paths


def _split_pipeline(command: str) -> Iterable[str]:
    """Yield pipeline / sequence segments from a shell command.

    Splits on ``|``, ``;``, ``&&``, ``||``, ``&`` (sequence/pipeline
    separators). Does NOT correctly handle separators inside quoted
    strings — but in practice that's a near-zero false-positive cost
    (the inner segment would just tokenize to nonsense and yield no
    matches). False-negative bias preserved.
    """
    # Replace multi-char separators first so the single-char split is clean.
    normalized = command.replace("&&", "|").replace("||", "|")
    for sep in (";", "&"):
        normalized = normalized.replace(sep, "|")
    for segment in normalized.split("|"):
        segment = segment.strip()
        if segment:
            yield segment


_BASH_WRAPPER_COMMANDS: frozenset[str] = frozenset({
    # ADV-006: wrappers that pass-through to a subcommand. We skip the
    # wrapper word and re-evaluate the next position. False-negative
    # bias preserved — the detector won't recurse into `bash -c '...'`
    # body strings (that's a separate eval surface) and won't try to
    # decode command substitutions like ``$(...)``. Just the common
    # `eval cat plan.md` and `command cat plan.md` cases.
    "eval",
    "command",
    "exec",
    "builtin",
})


def _detect_in_segment(
    segment: str,
    is_tracked: Callable[[str], bool],
) -> Iterable[str]:
    """Detect tracked paths within a single pipeline segment."""
    try:
        tokens = shlex.split(segment, comments=False, posix=True)
    except ValueError:
        return  # malformed quoting; skip silently

    if not tokens:
        return

    # The first non-flag token is the command word. Walk forward through
    # tokens that look like the command word; skip flags but treat any
    # bare value following a tracked command as a file-path candidate.
    i = 0
    n = len(tokens)
    # ADV-006: bound the wrapper-skip loop so a pathological
    # ``eval eval eval ... cat plan.md`` cannot consume unbounded
    # iterations. 4 wrappers is generous (typical depth is 1).
    wrappers_skipped = 0
    MAX_WRAPPERS = 4
    while i < n:
        tok = tokens[i]
        # Skip leading env-var assignments like FOO=bar
        if "=" in tok and i == 0 and re.match(r"^[A-Z_][A-Z0-9_]*=", tok):
            i += 1
            continue
        # ADV-006: skip pass-through wrappers (`eval`, `command`, etc.)
        # and re-evaluate the next position as the actual command.
        if tok in _BASH_WRAPPER_COMMANDS and wrappers_skipped < MAX_WRAPPERS:
            wrappers_skipped += 1
            i += 1
            continue
        # Command word reached.
        if tok in _TRACKED_READ_COMMANDS:
            for arg in tokens[i + 1:]:
                # Skip flags (`-foo`, `--bar`, `-`).
                if arg.startswith("-"):
                    continue
                # Pipeline-via-xargs: if the next bare token is another
                # tracked command, the FOLLOWING args belong to that
                # command, not the current one. Don't claim them here.
                if arg in _TRACKED_READ_COMMANDS:
                    break
                if is_tracked(arg):
                    yield arg
        # Always break after the first non-assignment, non-wrapper token —
        # we don't walk through nested commands ourselves; the pipeline
        # split already separated those.
        break


__all__ = ["PATH_TOKEN_RE", "detect_tracked_paths"]
