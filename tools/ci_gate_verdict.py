# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Verdicts for the gate jobs that carry CI's required check names.

``Tests (Python 3.x)`` and ``TLA+ Model Check`` are gates over the jobs that
do the work (test shards, TLA+ bins). A gate runs under ``always()`` so it
reports on every outcome, including a cancelled run; this script decides that
report. Only the listed pass rows pass, and everything else fails, so a gate
can never turn green on work that failed, was cancelled, or silently did not
run.

Tests gate:
    shards ``success`` and coverage exit ``0``     PASS
    anything else                                  FAIL
    (coverage exit 1 = fail, 2 = cannot tell; any other value, or none, is
    read as cannot tell)

TLA+ gate:
    bins ``success``                                                     PASS
    bins ``skipped`` on ``pull_request`` with the plan job ``success``
        and its TLA+-needed output exactly ``false``                     PASS
    anything else                                                        FAIL

Runs on the runner's system Python with no setup step. Usage:
    python3 tools/ci_gate_verdict.py tests --shards RESULT --coverage-exit CODE
    python3 tools/ci_gate_verdict.py tla --bins RESULT --event EVENT --plan RESULT --tla-needed VALUE
Exit 0 on PASS, 1 on FAIL; the first line printed is the verdict and its reason.
"""

from __future__ import annotations

import argparse

COVERAGE_PASS = "0"
COVERAGE_FAIL = "1"


def tests_verdict(shards_result: str, coverage_exit: str) -> tuple[bool, str]:
    """Pass only when every shard succeeded and the coverage check proved it."""
    if shards_result != "success":
        return False, f"test shards ended {shards_result or 'with no result'}"
    if coverage_exit == COVERAGE_PASS:
        return True, "every shard succeeded and ran each collected test exactly once"
    if coverage_exit == COVERAGE_FAIL:
        return False, "the shards did not run each collected test exactly once"
    return False, f"coverage cannot tell (exit {coverage_exit or 'missing'}): the shards' runs are unproven"


def tla_verdict(bins_result: str, event: str, plan_result: str, tla_needed: str) -> tuple[bool, str]:
    """Pass when the bins succeeded, or when a pull request provably left every TLA+ input alone."""
    if bins_result == "success":
        return True, "every TLA+ bin passed"
    if bins_result == "skipped" and event == "pull_request" and plan_result == "success" and tla_needed == "false":
        return True, "this pull request changes no TLA+ input, so the bins were skipped"
    return False, (
        f"TLA+ bins ended {bins_result or 'with no result'} on {event or 'an unknown event'} "
        f"(plan {plan_result or 'missing'}, TLA+ needed {tla_needed or 'missing'!s})"
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decide a CI gate's verdict.")
    gates = parser.add_subparsers(dest="gate", required=True)
    tests = gates.add_parser("tests", help="The Tests (Python 3.x) gate.")
    tests.add_argument("--shards", required=True, help="needs.<shards job>.result")
    tests.add_argument("--coverage-exit", required=True, help="Exit code of tools/check_shard_coverage.py.")
    tla = gates.add_parser("tla", help="The TLA+ Model Check gate.")
    tla.add_argument("--bins", required=True, help="needs.<bins job>.result")
    tla.add_argument("--event", required=True, help="github.event_name")
    tla.add_argument("--plan", required=True, help="needs.plan.result")
    tla.add_argument("--tla-needed", required=True, help="The plan job's TLA+-needed output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.gate == "tests":
        passed, reason = tests_verdict(args.shards, args.coverage_exit)
    else:
        passed, reason = tla_verdict(args.bins, args.event, args.plan, args.tla_needed)
    print(f"{'PASS' if passed else 'FAIL'}: {reason}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
