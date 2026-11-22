# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""Stale approval demo: record an approval only while the plan and budget it approved are unchanged.

An approver reads a plan and its budget, approves them, and records what it
read. The approval is written to an approvals file, which the sender reads
before it sends, from another process and later. In between, a peer edits
both the plan and the budget. The target of the act is outside the inputs:
nobody edits the approvals file, so a per-file check of it sees nothing wrong.

    python -m examples.basis_gate.stale_approval --runs 20

Every run, on every arm:
  * the unguarded act records the approval against the edited plan (the failure),
  * gate() fires anyway: it captures the version at its own call, after the edit,
  * a check_fence loop holds but names only the first moved input,
  * check_basis holds and names both moved inputs,
  * commit_if_basis holds and leaves the approvals file untouched, and
  * a second approver of the same plan and budget still proceeds after the
    first approver's approval is committed.

Exit 0 only when every arm behaved so in every run. Single-host and offline:
each arm starts a local coordinator in a temporary workspace; no network, no
model, no API keys.
"""

from __future__ import annotations

from collections.abc import Mapping

from examples.basis_gate._arms import Scenario, parse_runs, run_arms

PLAN = "plan.md"
BUDGET = "budget.md"
APPROVALS = "approvals.md"


def approve(read: Mapping[str, bytes]) -> bytes:
    """The approver's act: the approvals file it read, plus its approval of
    the plan and budget it read."""
    plan = read[PLAN].decode().strip()
    budget = read[BUDGET].decode().strip()
    return read[APPROVALS] + f"approved: {plan} ({budget})\n".encode()


SCENARIO = Scenario(
    area="tasks",
    inputs=(PLAN, BUDGET),
    target=APPROVALS,
    seed={
        PLAN: b"send the launch email to the beta list\n",
        BUDGET: b"cap: 500 emails\n",
        APPROVALS: b"# approvals\n",
    },
    peer_edit={
        PLAN: b"send the launch email to every customer\n",
        BUDGET: b"cap: 50000 emails\n",
    },
    act=approve,
)


def main(argv: list[str] | None = None) -> int:
    runs = parse_runs(argv, "Stale approval demo (single-host).")
    print("Stale approval: an approver approves a plan, the plan moves, then the approval is written.\n")
    ok = run_arms(SCENARIO, runs)
    print("\nTakeaway: the approvals file never moved, so nothing that checks the file being")
    print("written can see that the plan under the approval did. gate() compares from its")
    print("own call, and a check_fence loop stops at the first moved input. commit_if_basis")
    print("writes the approval only while every input it approved is still the version read,")
    print("and names each one that moved when it holds. It rewrites only the approvals file,")
    print("so a second approver's basis stays valid.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

# Comparing notes on multi-agent coherence?
# https://github.com/Cohexa-ai/agent-coherence/discussions
