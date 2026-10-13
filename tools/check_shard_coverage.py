# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Gate: prove the CI shards together ran every collected test exactly once.

The suite is split across N shard jobs per Python version (pytest-split), and
each shard runs two lanes: a parallel lane (xdist) and a serial lane. A split
that drops a test fails nothing (the test just stops running), so a per-version
gate job reads the node-id lists every shard wrote and checks the partition.

Input layout (written by tests/conftest.py):
    ROOT/<shard-dir>/<lane>-full.txt    every test the lane collected before splitting
    ROOT/<shard-dir>/<lane>-split.txt   the tests this shard's split selected
    ROOT/<shard-dir>/<lane>-ran.txt     the tests this shard produced a result for
for each lane in ("parallel", "serial"). One node id per line; the line is the
whole id, because parametrized ids carry spaces, brackets and quotes. Blank
lines are ignored, as is any entry of ROOT that is not a directory.

Verdicts, per lane:
    exit 2  CANNOT TELL  the shard-directory count differs from --shards; a list
                         file is missing or unreadable; the lane's full list is
                         empty in every shard (an empty collection proves nothing)
    exit 1  FAIL         full lists differ between shards (non-deterministic
                         collection); an id in two split lists or twice in one
                         (duplicate); a full-list id in no split (missed); a
                         split id absent from the full lists (unexpected); a
                         shard that did not run exactly what it selected
                         (not run / ran unselected)
    exit 0  PASS         otherwise; an empty split list is valid as long as the
                         splits together equal the full list

Combining lanes: any FAIL exits 1, otherwise any CANNOT TELL exits 2, otherwise
0. Both non-zero codes block the gate; FAIL outranks CANNOT TELL because it
names node ids someone can act on, and the cannot-tell lane is still printed.
A shard-count mismatch is checked before any lane and exits 2 at once, since no
lane can be judged against the wrong set of shards. "Cannot tell" never becomes
a pass.

Usage:
    python tools/check_shard_coverage.py --root DIR --shards N
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

LANES = ("parallel", "serial")
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_CANNOT_TELL = 2
VERDICT_NAMES = {EXIT_PASS: "PASS", EXIT_FAIL: "FAIL", EXIT_CANNOT_TELL: "CANNOT TELL"}
LIST_KINDS = ("full", "split", "ran")
# Enough to start debugging; a broken split can name thousands of ids.
MAX_NAMED = 20


class ShardLane(NamedTuple):
    """One shard's three lists for one lane."""

    name: str
    full: list[str]
    split: list[str]
    ran: list[str]


class LaneVerdict(NamedTuple):
    exit_code: int
    lines: list[str]


def read_node_ids(path: Path) -> list[str]:
    """One node id per non-blank line, kept whole (ids may contain spaces)."""
    text = path.read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line.strip()]


def load_lane(shard_dirs: Sequence[Path], lane: str) -> tuple[list[ShardLane], list[str]]:
    """Every shard's lists for ``lane``, plus a description of each file that could not be read."""
    loaded: list[ShardLane] = []
    problems: list[str] = []
    for shard_dir in shard_dirs:
        lists: dict[str, list[str]] = {}
        for kind in LIST_KINDS:
            path = shard_dir / f"{lane}-{kind}.txt"
            try:
                lists[kind] = read_node_ids(path)
            except (OSError, UnicodeDecodeError) as exc:
                problems.append(f"{shard_dir.name}/{path.name} ({type(exc).__name__})")
        if len(lists) == len(LIST_KINDS):
            loaded.append(ShardLane(shard_dir.name, lists["full"], lists["split"], lists["ran"]))
    return loaded, problems


def find_collection_drift(shards: Sequence[ShardLane]) -> tuple[list[str], list[str]]:
    """Shards whose full list differs from the first shard's, and the ids that differ."""
    reference = set(shards[0].full)
    drifted = [shard for shard in shards[1:] if set(shard.full) != reference]
    differing = set().union(*(set(shard.full) ^ reference for shard in drifted))
    return [shard.name for shard in drifted], sorted(differing)


def find_duplicates(shards: Sequence[ShardLane]) -> list[str]:
    """Each id selected more than once, with every shard that selected it (repeats included)."""
    owners: dict[str, list[str]] = {}
    for shard in shards:
        for node_id in shard.split:
            owners.setdefault(node_id, []).append(shard.name)
    return [f"{node_id} ({', '.join(names)})" for node_id, names in sorted(owners.items()) if len(names) > 1]


def find_run_gaps(shards: Sequence[ShardLane]) -> tuple[list[str], list[str]]:
    """Selected ids a shard produced no result for, and ids it ran without selecting.

    Selection is not execution: xdist can exit 0 with "no tests ran" after a
    worker dies during collection, leaving a full split list behind."""
    not_run: list[str] = []
    unselected: list[str] = []
    for shard in shards:
        not_run += [f"{node_id} ({shard.name})" for node_id in sorted(set(shard.split) - set(shard.ran))]
        unselected += [f"{node_id} ({shard.name})" for node_id in sorted(set(shard.ran) - set(shard.split))]
    return not_run, unselected


def format_category(title: str, entries: Sequence[str]) -> list[str]:
    lines = [f"  {title}: {len(entries)} node id(s)"]
    lines += [f"    {entry}" for entry in entries[:MAX_NAMED]]
    if len(entries) > MAX_NAMED:
        lines.append(f"    ... and {len(entries) - MAX_NAMED} more")
    return lines


def coverage_findings(shards: Sequence[ShardLane]) -> list[str]:
    """Report lines for every way the splits fail to partition the full list; empty when they do."""
    drifted, differing = find_collection_drift(shards)
    collected = set().union(*(shard.full for shard in shards))
    selected = set().union(*(shard.split for shard in shards))
    not_run, unselected = find_run_gaps(shards)
    categories = [
        (f"non-deterministic collection (full list of {', '.join(drifted)} differs from {shards[0].name})", differing),
        ("duplicate", find_duplicates(shards)),
        ("missed", sorted(collected - selected)),
        ("unexpected", sorted(selected - collected)),
        ("not run", not_run),
        ("ran unselected", unselected),
    ]
    return [line for title, entries in categories if entries for line in format_category(title, entries)]


def check_lane(lane: str, shard_dirs: Sequence[Path]) -> LaneVerdict:
    shards, problems = load_lane(shard_dirs, lane)
    if problems:
        reasons = [f"  missing or unreadable: {problem}" for problem in problems]
        return LaneVerdict(EXIT_CANNOT_TELL, [f"{lane}: {VERDICT_NAMES[EXIT_CANNOT_TELL]}", *reasons])
    if not any(shard.full for shard in shards):
        reason = "  full list is empty in every shard; an empty collection proves nothing"
        return LaneVerdict(EXIT_CANNOT_TELL, [f"{lane}: {VERDICT_NAMES[EXIT_CANNOT_TELL]}", reason])
    findings = coverage_findings(shards)
    if findings:
        return LaneVerdict(EXIT_FAIL, [f"{lane}: {VERDICT_NAMES[EXIT_FAIL]}", *findings])
    total = len(set(shards[0].full))
    return LaneVerdict(EXIT_PASS, [f"{lane}: {total} tests, {len(shards)} shards, each run once"])


def combine_exit_codes(codes: Sequence[int]) -> int:
    """FAIL outranks CANNOT TELL; both block the gate (see module docstring)."""
    if EXIT_FAIL in codes:
        return EXIT_FAIL
    if EXIT_CANNOT_TELL in codes:
        return EXIT_CANNOT_TELL
    return EXIT_PASS


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prove the CI shards ran every collected test exactly once.")
    parser.add_argument("--root", type=Path, required=True, help="Directory holding one subdirectory per shard.")
    parser.add_argument("--shards", type=int, required=True, help="Number of shards the run was split into.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        shard_dirs = sorted(path for path in args.root.iterdir() if path.is_dir())
    except (FileNotFoundError, NotADirectoryError):
        # No shard uploaded anything, so the root was never created: zero
        # shards, which is cannot tell, not a crash.
        shard_dirs = []
    if len(shard_dirs) != args.shards:
        names = ", ".join(path.name for path in shard_dirs) or "none"
        print(
            f"{VERDICT_NAMES[EXIT_CANNOT_TELL]}: found {len(shard_dirs)} shard directories under {args.root} "
            f"({names}), expected {args.shards}"
        )
        return EXIT_CANNOT_TELL
    verdicts = [check_lane(lane, shard_dirs) for lane in LANES]
    for verdict in verdicts:
        print("\n".join(verdict.lines))
    exit_code = combine_exit_codes([verdict.exit_code for verdict in verdicts])
    print(f"shard coverage: {VERDICT_NAMES[exit_code]}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
