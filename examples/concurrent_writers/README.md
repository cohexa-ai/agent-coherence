# Concurrent lost-update demo (commit-CAS)

Two agents update the same shared value **at the same time**. Without
coordination, one update is silently lost (last writer wins). With
`CoherentVolume.write_cas` — the optimistic commit-CAS write path shipped in
**v0.9.1** — the coordinator elects one winner, the loser is *told* it lost
(a typed, retryable conflict, never a silent drop), and it re-applies its update
on the winner's value. Both updates survive.

```bash
python -m examples.concurrent_writers.main
```

No API keys, no network. The broken case is pure offline file I/O; the fixed
case spawns a local coordinator subprocess and tears it down afterward.

## What it shows

| | Mechanism | Result |
|---|---|---|
| `broken.py` | two threads read the same value, then both write blindly over a plain file | **one update lost** — `final ∈ {105, 110}`, not `115` |
| `fixed.py` | the same race through `CoherentVolume.write_cas` | **both updates survive** — `final == 115` |

A read-barrier forces both writers to read the start value before either writes,
so the broken case loses an update *every* run (which writer wins is timing; that
an update is lost is not). The fixed case's winner is likewise non-deterministic,
but its invariant — `final == start + every delta` — is not.

## How `write_cas` prevents the loss

```python
vol.write_cas("data/tally.txt", lambda current: str(int(current) + delta).encode())
```

`write_cas` reads the current value (→ SHARED), derives the new bytes from it via
your `make_content` closure, and commits through the coordinator's
version-checked CAS. Two concurrent writers cannot both land the same version:
the serialized commit picks a winner, and the loser receives `version_mismatch`,
re-mints a fresh identity, takes ONE fresh **hash-checked** read (the bytes and
the version comparand always come from the same validated read — never from two
reads a peer could commit between), and **re-invokes `make_content` on the new
value** before retrying (the commit budget is `MAX_CAS_REACQUIRES`; on
exhaustion it raises `CasRetriesExhausted`, never a silent drop). Re-deriving
intent against the latest state is what turns the retry into an *update* rather
than a stale overwrite — so a closure that ignores its `current` argument
defeats the guard (the one fundamental OCC-proof boundary).

## How this differs from `examples/coherent_volume`

| | `examples/coherent_volume` | `examples/concurrent_writers` (here) |
|---|---|---|
| Failure | **sequential** stale-overwrite | **concurrent** lost update |
| Capability rung | 1 (sequential, single-host) | 2 (concurrent, single-host) |
| Mechanism | MESI invalidation → sticky `INVALID` deny + `reacquire()` | optimistic commit-CAS (`write_cas`) |
| Why CAS is needed | a peer commit invalidates the stale reader before it writes | the racy acquire can grant *both* writers — only a version-checked commit elects the winner |

The invalidation-deny model catches a writer whose read was *superseded before it
wrote*. It cannot catch two writers that read the same version and race to commit
— the acquire is non-atomic, so both can be granted. Commit-CAS closes exactly
that gap.

## Scope (honest)

Single-host: a loopback coordinator over SQLite-WAL. **Cross-host** concurrent
writers (a network coordinator + a durable cross-host registry) are the
demand-gated follow-on — see the roadmap's *Epic — Cross-Host Concurrency*
(Pieces #3–#5). Until that ships, multi-host prospects are pattern-only.

**Contention bound.** The guarantee `write_cas` makes is *every update lands,
or a typed error is raised* — never a silent loss. Under sustained same-key
contention approaching the commit budget (`MAX_CAS_REACQUIRES` = 8 → 9 CAS
attempts; ~8+ writers racing the SAME key at once), a writer can exhaust the
budget and raise `CasRetriesExhausted` — the honest fail-closed terminal, not a
dropped update. The 2-writer demo (and fleets up to roughly the budget) converge
cleanly. `CasRetriesExhausted.last_conflict_reason` tells contention
(`version_mismatch`) from a file another agent holds after a plain `write()`
(`other_holder`, or `caller_in_transient_state` when that write landed
mid-attempt), which no retry inside the call can get past.

Verified by `formal/tla/OCC.tla` (`NoLostUpdate`) and `formal/tla/Fencing.tla`
(`NoStaleApply`), both model-checked — these cover the *protocol*. The
client-integration path (`CoherentVolume.write_cas` over HTTP + the real
filesystem) is covered by the regression tests here: the comparand bytes and
version are always taken from one hash-checked read, so a stale read can never
win the version CAS (see `test_fixed_under_higher_contention_holds_the_honest_invariant`).
Tests: `tests/test_concurrent_writers_demo.py`.
