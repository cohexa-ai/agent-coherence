# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Per-lane totals from the JUnit reports the test shards uploaded.

Informational only: a test gate prints these in its job summary, and the
coverage check (tools/check_shard_coverage.py), not these totals, decides the
gate. A module that skips itself at collection is reported once per pytest
run, so the skipped total counts it once per shard and lane; the node-id
lists are the exact count.

Usage: python3 tools/summarize_junit.py ROOT   (ROOT/<shard>/junit-<lane>.xml)
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

LANES = ("parallel", "serial")
COUNTS = ("tests", "failures", "errors", "skipped")


def lane_totals(root: Path, lane: str) -> tuple[dict[str, int], int]:
    """Summed testsuite counts for ``lane``, and how many reports were read."""
    totals = dict.fromkeys(COUNTS, 0)
    reports = sorted(root.glob(f"*/junit-{lane}.xml"))
    for report in reports:
        for suite in ET.parse(report).getroot().iter("testsuite"):
            for count in COUNTS:
                totals[count] += int(suite.get(count, 0))
    return totals, len(reports)


def main(argv: list[str]) -> int:
    root = Path(argv[0])
    for lane in LANES:
        try:
            totals, reports = lane_totals(root, lane)
        except (OSError, ET.ParseError, ValueError) as exc:
            print(f"{lane}: JUnit reports unreadable ({type(exc).__name__})")
            continue
        counts = ", ".join(f"{totals[count]} {count}" for count in COUNTS)
        print(f"{lane}: {counts} (from {reports} JUnit report(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
