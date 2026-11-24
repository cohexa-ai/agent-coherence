# Basis demos: act only while every input is still the version you read

A check, a review or an approval reads several shared files, records what it
saw, and is acted on later, sometimes from another process. Before the act
lands, it needs to know whether every input is still the version it read. These
two scripts show what each available guard answers when a peer edits two inputs
in between.

```
python -m examples.basis_gate.stale_check --runs 20
python -m examples.basis_gate.stale_approval --runs 20
```

Offline, no API keys, no model: each arm starts a local coordinator in a
temporary workspace, reuses it across runs with fresh files per run, and stops
it at the end. Each script prints a count per arm and exits `0` only when every
arm behaved as stated below in every run, else `1`.

## The two scripts

| Script | Inputs | What the act writes |
|---|---|---|
| `stale_check.py` | a spec and a fixture | the fixture, stamped as checked against the spec: the target is one of the inputs |
| `stale_approval.py` | a plan and its budget | an approval appended to an approvals file: the target is outside the inputs, and nobody edits it |

In both, the act lands a result computed from the bytes the reader recorded,
not from a fresh read. It runs from a second volume, as a later step in another
process would: that process holds only the recorded result. (A volume's own
stale-read deny refuses a write after a peer moved the very file it read; it
cannot see inputs read elsewhere, nor inputs other than the file it writes.)

## The arms

Every arm but the last starts the same way: the files are created and
committed, the reader records both inputs, and a peer edits both.

| Arm | Every run |
|---|---|
| unguarded act | writes the result built from the old bytes: it used a moved input |
| `gate()`, whose decision was made before the edit | proceeds: it captures the version at its own call, after the edit, so it has nothing earlier to compare |
| a `check_fence` loop, one call per input, recorded through `read_with_version_generation` | holds, but names only the first moved input: the first hold ends the loop |
| `check_basis` | holds, and names both moved inputs with the version each was read at, its version now, and why |
| `commit_if_basis` | holds; the target is unchanged on disk and at the coordinator |
| a second reader, nothing edited | after the first reader's `commit_if_basis` lands, the second reader's basis, which leaves the first reader's target out, still answers proceed |

The last arm is why the commit form rewrites only its target. Publishing every
input together would also hold on a moved input, but it rewrites each of them,
so every other reader's recorded versions go stale with it.

## The calls

```python
from ccs.adapters import CoherentVolume

vol = CoherentVolume(root, managed=("tasks/**",))

# Record: the bytes, and a member naming the path, the version and the registration read.
plan, plan_member = vol.read_for_basis("tasks/plan.md")
budget, budget_member = vol.read_for_basis("tasks/budget.md")
approvals, approvals_version = vol.read_with_version("tasks/approvals.md")
basis = [plan_member, budget_member]  # plain data; elsewhere, rebuild with BasisMember(**fields)

# Later, maybe in another process: are all of them still what was read?
answer = vol.check_basis(basis)
if answer.answer == "hold":
    for moved in answer.moved:  # every moved member, not just the first
        print(moved.path, moved.expected, moved.current, moved.reason)
    # answer.recovery: "rederive" (read again and redo the work) or "retry"

# Or land the result only while they all are, built from the bytes recorded above.
outcome = vol.commit_if_basis(basis, "tasks/approvals.md", approvals_version, approvals + b"approved\n")
# A win returns the committed version; a hold returns the answer and writes nothing.
```

## Honest scope

- **One instant.** `check_basis` compares every member at one instant. It does
  not make the act that follows atomic: an input can move right after a
  proceed. When the act is a file write, use `commit_if_basis`, which compares
  the basis and commits the target in one step.
- **Use the recorded bytes.** A proceed says the recorded versions are still
  current. Build the act from the bytes `read_for_basis` returned, not from a
  later read of the disk.
- **Versions and registrations only.** A byte-identical re-save moves the
  version and holds. A file deleted and created again is told apart by its
  registration, even when its version starts over.
- **Single host, cooperative.** The readers, the peer and the actor share one
  local coordinator, and each calls the check; nothing intercepts a writer that
  does not.
- **Python coordinator only.** The Node coordinator does not serve the check:
  `read_for_basis`, `check_basis` and `commit_if_basis` raise
  `BasisCheckAbsent` against it rather than answer proceed.
