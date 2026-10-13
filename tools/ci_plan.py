# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The "Plan checks" job: decide what this CI run needs, and never fail.

Every other job in ci.yml lists this job in ``needs`` and reads its outputs.
The rule for those outputs is fail-open: a consumer skips work only when a
job condition sees both the right event and an explicit value, so an empty or
missing output means "run". This script therefore writes every key on every
run, writes a key's run-everything value when it cannot decide it, and always
exits 0: a plan job that failed would hand its consumers to GitHub's
cascade-skip rules.

Outputs, appended to ``$GITHUB_OUTPUT``, each with its run-everything value
when it cannot be decided:
    tested-tree   the tree id of the checked-out commit (what this run tests);
                  empty when unknown
    tla-needed    ``false`` only on a pull request whose changes touch no TLA+
                  input (anything under formal/tla/, any makefile GNU make
                  reads, anything under .github/, this script and
                  tools/ci_gate_verdict.py); ``true`` on every other event,
                  on an empty or failed diff, and when unknown

On ``pull_request`` the checkout is the merge commit at depth 2, so its first
parent is the base: the diff between them is exactly what the PR changes.

Runs on the runner's system Python with no setup step, from the repository
root: ``python3 tools/ci_plan.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys

OUTPUT_KEYS = ("tested-tree", "tla-needed")
# What each output says when it cannot be decided: "run everything".
FALLBACKS = {"tested-tree": "", "tla-needed": "true"}
# Specs extend each other, so the whole directory is one input. The two
# scripts decide and judge the TLA+ run, so a change to them must run it.
# GNU make reads GNUmakefile, then makefile, before Makefile.
TLA_INPUT_PREFIXES = ("formal/tla/", ".github/")
TLA_INPUT_FILES = ("GNUmakefile", "makefile", "Makefile", "tools/ci_plan.py", "tools/ci_gate_verdict.py")


def tested_tree() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], capture_output=True, text=True, timeout=30, check=True
    )
    return result.stdout.strip()


def changed_paths() -> list[str]:
    """Paths the checked-out merge commit changes against its first parent (the base).

    The flags pin the output shape instead of inheriting it: with rename
    detection on, a file moved out of formal/tla/ lists only its new path, and
    a quoted non-ASCII path no longer starts with its directory.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", "HEAD^1", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [path for path in result.stdout.split("\0") if path]


def is_tla_input(path: str) -> bool:
    return path in TLA_INPUT_FILES or path.startswith(TLA_INPUT_PREFIXES)


def tla_needed() -> str:
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return "true"
    changed = changed_paths()
    # A diff that misfires can exit zero with no output: an empty list proves nothing.
    if not changed:
        return "true"
    return "true" if any(is_tla_input(path) for path in changed) else "false"


DECISIONS = {"tested-tree": tested_tree, "tla-needed": tla_needed}


def write_outputs(outputs: dict[str, str]) -> None:
    lines = "".join(f"{key}={outputs.get(key, '')}\n" for key in OUTPUT_KEYS)
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as out:
            out.write(lines)
    print(lines, end="")


def decide(key: str) -> str:
    try:
        return DECISIONS[key]()
    # Fail open by design: an output that cannot be decided takes its
    # run-everything value, and the other outputs are still decided.
    except Exception as exc:  # noqa: BLE001
        print(f"plan: could not decide {key} ({type(exc).__name__}: {exc}); it runs", file=sys.stderr)
        return FALLBACKS[key]


def main() -> int:
    write_outputs({key: decide(key) for key in OUTPUT_KEYS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
