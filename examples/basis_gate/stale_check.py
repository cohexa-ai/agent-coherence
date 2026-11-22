# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""Stale check demo: act on a check only while the spec and fixture it read are unchanged.

A check reads a spec and a fixture, finds the fixture matches the spec, and
records what it read. Its act, stamping the fixture as checked against that
spec, runs later from another process. In between, a peer edits both the spec
and the fixture. The target of the act is one of the inputs: the fixture.

    python -m examples.basis_gate.stale_check --runs 20

Every run, on every arm:
  * the unguarded act stamps the old fixture over the peer's edit (the failure),
  * gate() fires anyway: it captures the version at its own call, after the edit,
  * a check_fence loop holds but names only the first moved input,
  * check_basis holds and names both moved inputs,
  * commit_if_basis holds and leaves the fixture untouched, and
  * a second reader of the spec still proceeds after a commit of the fixture.

Exit 0 only when every arm behaved so in every run. Single-host and offline:
each arm starts a local coordinator in a temporary workspace; no network, no
model, no API keys.
"""

from __future__ import annotations

from collections.abc import Mapping

from examples.basis_gate._arms import Scenario, parse_runs, run_arms

SPEC = "spec.md"
FIXTURE = "fixture.yaml"


def stamp_fixture(read: Mapping[str, bytes]) -> bytes:
    """The check's act: the fixture it read, stamped with the spec it checked it against."""
    spec = read[SPEC].decode().strip()
    return read[FIXTURE] + f"# checked against: {spec}\n".encode()


SCENARIO = Scenario(
    area="checks",
    inputs=(SPEC, FIXTURE),
    target=FIXTURE,
    seed={
        SPEC: b"GET /orders returns id, total\n",
        FIXTURE: b"orders: [{id: 1, total: 40}]\n",
    },
    peer_edit={
        SPEC: b"GET /orders returns id, total, currency\n",
        FIXTURE: b"orders: [{id: 1, total: 40, currency: EUR}]\n",
    },
    act=stamp_fixture,
)


def main(argv: list[str] | None = None) -> int:
    runs = parse_runs(argv, "Stale check demo (single-host).")
    print("Stale check: a check reads a spec and a fixture, a peer edits both, then the act runs.\n")
    ok = run_arms(SCENARIO, runs)
    print("\nTakeaway: a check is acted on later, after its inputs may have moved. gate()")
    print("and a check_fence loop each see only part of that: gate() compares from its own")
    print("call, and the loop stops at the first moved input. check_basis answers for every")
    print("input at one instant, against the versions the check recorded, and names each")
    print("one that moved. commit_if_basis lands the result only while they all held, and")
    print("rewrites only its target, so other readers' bases stay valid.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

# Comparing notes on multi-agent coherence?
# https://github.com/Cohexa-ai/agent-coherence/discussions
