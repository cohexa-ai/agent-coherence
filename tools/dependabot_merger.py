# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Pick the next step for the open Dependabot pull requests into dev.

.github/workflows/dependabot-merge.yml runs this every few minutes and carries
out what it prints with the forward-merge GitHub App. At most one step per run,
because each merge leaves every other PR behind dev:

    merge   the oldest eligible PR that is up to date (CLEAN) with every check
            passing; squash-merged, pinned to the head the gate judged
    update  when nothing can merge, no eligible PR's checks are running and no
            Dependabot head still awaits its gate verdict: the oldest eligible
            PR that is BEHIND with every check passing, once both its verdict
            and dev's last move are at least REFRESH_GRACE old, so Dependabot
            had its chance to refresh the PR itself; dev is merged into it
    none    otherwise

An eligible PR with a failed check or a merge conflict is never acted on; the
run prints a warning naming it, because only a person can re-run its checks or
ask Dependabot to recreate it.

A PR is eligible only when this script, independently of the gate, finds a
Dependabot branch for the Python lock or the TypeScript demo fixture, changed
files within that ecosystem's lockfile and manifest (tools/dependabot_gate.py
holds both lists), no draft and no fork, and a ``dependabot-gate`` status on
its current head that is exactly ``success`` / ``qualifies``, written by a
workflow (creator ``github-actions[bot]``). A status forged on any other PR
therefore merges nothing. GitHub still refuses a merge unless dev's required
checks passed on an up-to-date head.

Inputs (JSON, as the workflow fetches them):
    --prs           gh pr list --json number,headRefName,headRefOid,
                    mergeStateStatus,isDraft,isCrossRepository,files,
                    statusCheckRollup
    --gates         [{"sha": ..., "gate": <the newest dependabot-gate entry of
                    GET /commits/{sha}/statuses, or null>}, ...]
    --dev-moved-at  the committer time of dev's tip, which GitHub sets to the
                    merge time for the squash and merge commits it creates
Output: ``action=``, ``number=`` and ``head=`` lines appended to
--github-output, and one line per PR on stdout saying why it was or was not
picked.

Usage:
    python tools/dependabot_merger.py --prs FILE --gates FILE --now ISO8601 \\
        --dev-moved-at ISO8601 --github-output FILE
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependabot_gate import ALLOWED_FILES, FIXTURE  # noqa: E402

GATE_CONTEXT = "dependabot-gate"
GATE_QUALIFIES = "qualifies"
GATE_WRITER = "github-actions[bot]"
AWAITING_VERDICT = "awaiting the gate's verdict"
REFRESH_GRACE = timedelta(minutes=10)
BRANCH_PREFIXES = {
    "uv": "dependabot/uv/",
    "npm_and_yarn": f"dependabot/npm_and_yarn/{FIXTURE}/",
}
PASSING_STATES = frozenset({"SUCCESS", "NEUTRAL", "SKIPPED"})
FAILING_STATES = frozenset(
    {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE", "STALE"}
)


@dataclass(frozen=True)
class Candidate:
    number: int
    head: str
    merge_state: str
    checks: str  # "passing", "pending" or "failing"
    quiet_for: timedelta  # since the later of its gate verdict and dev's last move


@dataclass(frozen=True)
class Clock:
    now: datetime
    dev_moved_at: datetime


@dataclass(frozen=True)
class Step:
    action: str  # "merge", "update" or "none"
    number: int | None = None
    head: str | None = None


def ecosystem_of(branch: str) -> str | None:
    return next((eco for eco, prefix in BRANCH_PREFIXES.items() if branch.startswith(prefix)), None)


def ineligibility(pr: dict[str, Any], gate: dict[str, Any] | None) -> str | None:
    """Why this PR may not be merged or updated here, or None."""
    if pr.get("isDraft") or pr.get("isCrossRepository"):
        return "draft or fork"
    ecosystem = ecosystem_of(pr.get("headRefName", ""))
    if ecosystem is None:
        return "not a Python lock or demo fixture update"
    files = {entry.get("path") for entry in pr.get("files") or []}
    if not files:
        return "no changed files"
    if not files <= ALLOWED_FILES[ecosystem]:
        return f"changes {', '.join(sorted(files - ALLOWED_FILES[ecosystem]))}"
    if not gate:
        return AWAITING_VERDICT
    qualified = gate.get("state") == "success" and gate.get("description") == GATE_QUALIFIES
    if not qualified or (gate.get("creator") or {}).get("login") != GATE_WRITER:
        return "the gate has not qualified this head"
    return None


def check_summary(rollup: list[dict[str, Any]]) -> str:
    """'failing' if any check failed, else 'passing' if every check passed,
    else 'pending': a state this script does not know never counts as passed."""
    states = set()
    for check in rollup:
        if check.get("__typename") == "StatusContext":
            states.add(str(check.get("state", "")).upper())
        elif str(check.get("status", "")).upper() != "COMPLETED":
            states.add("PENDING")
        else:
            states.add(str(check.get("conclusion", "")).upper())
    if states & FAILING_STATES:
        return "failing"
    return "passing" if states <= PASSING_STATES else "pending"


def decide(candidates: list[Candidate]) -> Step:
    ordered = sorted(candidates, key=lambda candidate: candidate.number)
    for candidate in ordered:
        if candidate.merge_state == "CLEAN" and candidate.checks == "passing":
            return Step("merge", candidate.number, candidate.head)
    if any(candidate.checks == "pending" for candidate in ordered):
        return Step("none")
    for candidate in ordered:
        quiet = candidate.quiet_for >= REFRESH_GRACE
        if candidate.merge_state == "BEHIND" and candidate.checks == "passing" and quiet:
            return Step("update", candidate.number, candidate.head)
    return Step("none")


def _timestamp(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def candidates_from(prs: list[dict[str, Any]], gates: dict[str, Any], clock: Clock) -> list[Candidate]:
    """Eligible PRs, plus each Dependabot head still awaiting its verdict as a
    PR whose checks are running: it may qualify any moment, and an update now
    would leave it behind."""
    candidates = []
    for pr in sorted(prs, key=lambda pr: pr["number"]):
        gate = gates.get(pr["headRefOid"])
        reason = ineligibility(pr, gate)
        print(f"#{pr['number']}: {reason or 'eligible'} ({pr.get('mergeStateStatus')})")
        if reason == AWAITING_VERDICT:
            candidates.append(Candidate(pr["number"], pr["headRefOid"], AWAITING_VERDICT, "pending", timedelta(0)))
        elif reason is None:
            since = max(_timestamp(gate["updated_at"]), clock.dev_moved_at)
            checks = check_summary(pr.get("statusCheckRollup") or [])
            candidate = Candidate(pr["number"], pr["headRefOid"], pr["mergeStateStatus"], checks, clock.now - since)
            candidates.append(candidate)
    return candidates


def stuck(candidates: list[Candidate]) -> list[str]:
    """A warning for each qualified PR only a person can move."""
    return [
        f"#{candidate.number} qualifies but has a failed check; re-run it, or comment @dependabot recreate"
        if candidate.checks == "failing"
        else f"#{candidate.number} qualifies but conflicts with dev; comment @dependabot recreate"
        for candidate in candidates
        if candidate.checks == "failing" or candidate.merge_state == "DIRTY"
    ]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prs", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--now", type=_timestamp, required=True)
    parser.add_argument("--dev-moved-at", type=_timestamp, required=True)
    parser.add_argument("--github-output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prs = json.loads(args.prs.read_text())
    gates = {entry["sha"]: entry.get("gate") for entry in json.loads(args.gates.read_text())}
    candidates = candidates_from(prs, gates, Clock(args.now, args.dev_moved_at))
    for warning in stuck(candidates):
        print(f"::warning::{warning}")
    step = decide(candidates)
    print(f"next step: {step.action} {f'#{step.number}' if step.number else ''}".rstrip())
    with args.github_output.open("a") as output:
        output.write(f"action={step.action}\nnumber={step.number or ''}\nhead={step.head or ''}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
