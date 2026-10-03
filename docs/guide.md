# agent-coherence User Guide

When two agents share state, one of them is usually reading a stale copy.
This guide shows how to drop in `agent-coherence` — surfacing those reads
and serving fresh artifacts on demand, with a one-line import change.

`agent-coherence` is **vendor-neutral by design**: the same protocol and
the same library work across LangGraph, CrewAI, AutoGen, the OpenAI Agents
SDK, and any custom orchestrator, with any model provider (Anthropic, OpenAI,
Google, Mistral, open-source). Pick the integration extra that matches your
stack; the concepts below apply uniformly.

Below: installation, namespace convention, sync strategies, observability,
telemetry, graceful degradation, examples, the `ccs-diagnose` CLI, the
full command-line toolset, and the API reference.

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [Namespace convention](#namespace-convention)
4. [Strategies](#strategies)
5. [Observability](#observability)
6. [State transitions log](#state-transitions-log)
7. [Content audit log](#content-audit-log)
8. [Crash recovery](#crash-recovery)
9. [Version retention and read-at-version](#version-retention-and-read-at-version)
10. [Coherent workspace (`CoherentVolume`)](#coherent-workspace-coherentvolume)
11. [BYO substrate bindings (`CoherentRow`, `CoherentObject`)](#byo-substrate-bindings-coherentrow-coherentobject)
12. [Workspace versioning & restore (`WorkspaceVersioner`)](#workspace-versioning--restore-workspaceversioner)
13. [Multi-artifact snapshot sessions](#multi-artifact-snapshot-sessions)
14. [Effect fence over HTTP](#effect-fence-over-http)
15. [Caller principal](#caller-principal)
16. [Acquire-or-fail on `pre-edit` (specified, not yet built)](#acquire-or-fail-on-pre-edit-specified-not-yet-built)
17. [`stale-write-guard-fs` MCP server](#stale-write-guard-fs-mcp-server)
18. [Inline benchmark mode](#inline-benchmark-mode)
19. [Telemetry](#telemetry)
20. [Graceful degradation](#graceful-degradation)
21. [Examples](#examples)
22. [Real-workload benchmarks](#real-workload-benchmarks)
23. [Benchmarking your own workload](#benchmarking-your-own-workload)
24. [`ccs-diagnose` — detect stale reads](#ccs-diagnose--detect-stale-reads)
25. [Conflict-outcome counters — how often did it actually fire?](#conflict-outcome-counters--how-often-did-it-actually-fire)
26. [Replay (v0.8.2+)](#replay-v082)
27. [Command-line tools](#command-line-tools)
28. [API reference](#api-reference)
29. [Low-level adapter API](#low-level-adapter-api)
30. [CrewAI and AutoGen adapters](#crewai-and-autogen-adapters)
31. [OpenAI Agents SDK adapter (experimental)](#openai-agents-sdk-adapter-experimental)

---

## Installation

Requires Python 3.11+. Pick the integration extra that matches your stack. The library is the same across all of them — only the adapter surface changes.

```bash
# LangGraph (drop-in CCSStore)
pip install "agent-coherence[langgraph]"

# CrewAI adapter
pip install "agent-coherence[crewai]"

# ccs-diagnose CLI (stale-read detector for LangGraph graphs)
pip install "agent-coherence[diagnose]"

# With OpenTelemetry metrics
pip install "agent-coherence[langgraph,otel]"

# With LangSmith tracing
pip install "agent-coherence[langgraph,langsmith]"

# OpenAI Agents SDK adapter (experimental, 0.x)
pip install "agent-coherence[openai-agents]"

# stale-write-guard-fs MCP server (coordinated file access for any MCP client)
pip install "agent-coherence[mcp]"

# BYO-substrate bindings (CoherentRow for Postgres / CoherentObject for S3)
pip install "agent-coherence[coherent-row]"
pip install "agent-coherence[coherent-object]"

# Substrate conformance corpus (for foreign implementations; not part of [all])
pip install "agent-coherence[conformance]"

# Everything (langgraph + crewai + otel + langsmith + benchmark + diagnose + openai-agents + mistral + mcp + coherent-row + coherent-object)
pip install "agent-coherence[all]"
```

For security-sensitive installs with full transitive hash pinning, see [the security guide](security.md#hash-pinned-install-for-security-sensitive-users) and the bundled `requirements-diagnose.txt`.

---

## Quick start

```python
# Before
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()

# After — one import change, no node code changes
from ccs.adapters import CCSStore
store = CCSStore(strategy="lazy")

graph = builder.compile(store=store)
```

Node code stays identical — `store.get()`, `store.put()`, and `store.search()` all
work the same way.

**What CCSStore does at the write boundary.** CCSStore provides read-side
coherence: when a peer commits a new version, your cached view is invalidated so
your next read is a fresh miss. It does not deny a stale write-back — `put` is not
version-CAS. For write-side lost-update prevention (a stale writer overwriting a
peer), route writes through [`CoherentVolume`](#coherent-workspace-coherentvolume)
or `write_cas`.

**In-process scope.** CCSStore coherence is in-process: two separate OS processes
each constructing their own CCSStore share nothing. For cross-process coordination
over files, use [`CoherentVolume`](#coherent-workspace-coherentvolume).

**The one-import swap assumes agent-carrying namespaces.** The drop-in is one import
change *only if* your namespaces already carry the agent identity in `namespace[0]`
(see [Namespace convention](#namespace-convention)) — that is what lets two agents
share one artifact while keeping private scratch private. A store keyed the
LangGraph-memory way, `(user_id, "memories")`, would collapse every user onto one
shared artifact, so migrate those call sites to put the agent in `namespace[0]`
before the swap.

**CCSStore and your existing store.** CCSStore's cached contents live for the
process lifetime, not on disk — it is not a database. If you already run Mem0,
Letta, LlamaIndex, or a LangGraph store, keep it: it stays your durability layer,
and CCSStore adds coherence for the cached view above it (a peer write invalidates a
stale read). It does not wrap or replace your backend's storage.

---

## Namespace convention

CCSStore overloads the `namespace` tuple that LangGraph passes to `get` and `put`:

| Position | Meaning | Example |
|----------|---------|---------|
| `namespace[0]` | Agent identity | `"planner"`, `"reviewer"` |
| `namespace[1:]` | Artifact scope | `("shared",)`, `("project", "v2")` |

**Two agents share an artifact when their scopes match:**

```python
# Both address the same "codebase" artifact
store.put(("reviewer_a", "shared"), "codebase", {...})
store.get(("reviewer_b", "shared"), "codebase")  # reads what reviewer_a wrote
```

**Agent-private artifacts:** include the agent name in the scope.

```python
store.put(("planner", "planner", "scratch"), "draft", {...})
# scope is ("planner", "scratch") — other agents cannot see this key
```

This convention is required. Namespaces with fewer than two elements raise
`ValueError`.

---

## Strategies

Pass `strategy=` to `CCSStore(...)` to control when invalidated entries are
re-fetched.

| Strategy | Behaviour | Best for |
|----------|-----------|----------|
| `"lazy"` *(default)* | Fetch on next read after invalidation | Most workloads |
| `"eager"` | Pre-fetch as soon as an invalidation signal arrives | Low-latency reads |
| `"lease"` | Entries expire after a TTL regardless of writes | Time-sensitive data |
| `"access_count"` | Fetch on every N-th access | High-read, low-write |
| `"broadcast"` | Always fetch — no local caching | Debugging, correctness testing |

Strategy-specific kwargs are forwarded directly:

```python
store = CCSStore(strategy="lease", lease_ticks=10)
store = CCSStore(strategy="access_count", threshold=3)
```

---

## Observability

Pass `on_metric` to receive a `StoreMetricEvent` after every operation:

```python
from ccs.adapters import CCSStore, StoreMetricEvent

events: list[StoreMetricEvent] = []
store = CCSStore(strategy="lazy", on_metric=events.append)

# ... run your graph ...

hits   = [e for e in events if e.operation == "get" and e.cache_hit]
misses = [e for e in events if e.operation == "get" and not e.cache_hit]
saved  = sum(e.tokens_consumed for e in misses) - len(hits)  # rough savings
```

### `StoreMetricEvent` fields

| Field | Type | Description |
|-------|------|-------------|
| `operation` | `str` | `"get"`, `"put"`, `"search.hit"`, or `"degraded"` |
| `namespace` | `tuple[str, ...]` | Full namespace including agent name |
| `key` | `str` | Artifact key |
| `agent_name` | `str` | First element of `namespace` |
| `tokens_consumed` | `int` | `1` on cache hit; estimated content size on miss |
| `cache_hit` | `bool` | `True` when served from local cache |
| `tick` | `int` | Logical clock at the time of the operation |

Token estimation: `max(1, len(json.dumps(value)) // 4)`. Override by including
`"__ccs_size_tokens__": N` in your artifact value.

---

## State transitions log

Pass `state_log` to receive a structured dict for every stable MESI state transition.
Intended for external tools — debuggers, visualizers, audit pipelines — that need to
correlate agent behavior with coherence state changes without coupling to CCS internals.

```python
import json

log = []
store = CCSStore(strategy="lazy", state_log=log.append)

# ... run your graph ...

# Write JSONL
with open("transitions.jsonl", "w") as f:
    for entry in log:
        f.write(json.dumps(entry) + "\n")
```

### Log entry schema

Each entry is a flat `dict` with exactly these eight keys:

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `int` | Monotonic operation counter within this `CCSStore` session |
| `artifact_id` | `str` | UUID of the artifact whose per-agent state changed |
| `agent_id` | `str` | UUID of the agent whose state changed |
| `agent_name` | `str \| None` | Agent display name (resolved from `namespace[0]`); `None` for low-level registry callers |
| `from_state` | `str` | Previous state: `"MODIFIED"`, `"EXCLUSIVE"`, `"SHARED"`, or `"INVALID"` |
| `to_state` | `str` | New state after the transition |
| `trigger` | `str` | Coordinator operation that caused the transition (see table below) |
| `version` | `int` | Artifact version number at the moment of the transition |

### Trigger vocabulary

| `trigger` | Fires when |
|-----------|-----------|
| `"register"` | Initial artifact registration; registering agent receives EXCLUSIVE |
| `"fetch"` | Fetch grant; the requester transitions to SHARED or EXCLUSIVE, and a peer holding EXCLUSIVE/MODIFIED is downgraded to SHARED under the same trigger (a peer already in SHARED is not touched) |
| `"write"` | Write request; peers are invalidated (→ INVALID), requester receives EXCLUSIVE |
| `"commit"` | Write commit; peers are invalidated (→ INVALID), committer transitions to MODIFIED |
| `"invalidate"` | Explicit invalidation signal; agent transitions to INVALID |
| `"timeout"` | Transient state timeout; agent force-invalidated (→ INVALID) |
| `"reclaim_heartbeat"` | Crash recovery: agent's heartbeat older than `heartbeat_timeout_ticks` |
| `"reclaim_max_hold"` | Crash recovery: grant held for at least `max_hold_ticks` |

The last four triggers move the artifact's **ownership epoch** when they take a
holder out of EXCLUSIVE/MODIFIED, because each of them ends a write claim
*without the version moving* — the one case a version check cannot see. A later
commit from that ex-holder is then rejected with `stale_read_generation` rather
than silently applied. `"write"` and `"commit"` do not move the epoch: those
paths move the version, so the version check already arbitrates them. An
agent's side of the check — the generation it captured — is written only by
its own acquire or its own read; being downgraded by another agent's fetch
never refreshes it, so an ex-holder rejected with `stale_read_generation`
stays rejected until it re-reads or re-acquires itself.

An explicit invalidation is also **pinned**: one issued by a peer is dropped as
obsolete if the agent it names has since observed a version at least as new as
the one the signal announces. Without that, an invalidation minted before a peer
was reclaimed and re-acquired would revoke the fresh grant it knows nothing
about. An agent releasing its *own* claim is never pinned.

### Error handling

The callback is called synchronously on the critical path. An exception in `state_log`
propagates out of the coordinator operation and may leave the log incomplete for that
batch. Provide a callback that catches its own exceptions for production use:

```python
def safe_log(entry: dict) -> None:
    try:
        emit_to_pipeline(entry)
    except Exception:
        logger.exception("state_log callback failed")

store = CCSStore(strategy="lazy", state_log=safe_log)
```

`state_log=None` (default) adds no overhead — the guard is a single `is not None` check.

### Callbacks run under the registry lock

`state_log` (and the crash-recovery sweep's `on_reclaim` callback) execute while the
coordinator's registry lock is **held**, on both backends. A callback that blocks —
waiting on another thread, a network sink with no timeout, a queue that can fill —
stalls every registry operation in the process for as long as it blocks, not just the
one being logged. Keep callbacks fast and non-blocking: append to an in-memory buffer
and drain it elsewhere, rather than doing I/O inline. Calling back into the store from
inside a callback is safe only for registry methods (the lock is reentrant); never
wait on other threads from inside one.

### Log validation

Verify a materialized JSONL log for gaps and schema drift:

```python
from ccs.validation import validate_log, CCS_STATE_LOG_SCHEMA_VERSION

gaps, mismatches = validate_log(
    "transitions.jsonl",
    schema_version=CCS_STATE_LOG_SCHEMA_VERSION,
)
# gaps: list of dropped-event positions; mismatches: list of schema version changes
# returns ([], []) on a clean log
```

`validate_log` is stdlib-only and importable independently of the CCS runtime, so log
consumers (audit pipelines, replay tools) can verify materialized logs without taking
on the rest of the CCS dependency surface.

---

## Content audit log

Pass `content_audit_log` to record every content delivery — what each agent actually saw,
when, and from which source. While `state_log` tracks MESI state transitions, the audit
log tracks content flow: cache hits, fetches, broadcasts, writes, and searches.

```python
audit = []
store = CCSStore(strategy="lazy", content_audit_log=audit.append)

# ... run your graph ...

# Each entry records one content delivery
for entry in audit:
    print(f"{entry['agent_name']} saw artifact {entry['artifact_id']} "
          f"via {entry['source']} (v{entry['version']})")
```

Enabling `content_audit_log` also enables version retention — the registry keeps a copy
of each artifact version so historical content can be retrieved for replay or debugging.

### Audit entry schema

| Field | Type | Description |
|-------|------|-------------|
| `tick` | `int` | Monotonic operation counter |
| `agent_id` | `str \| None` | UUID of the receiving agent; `None` for search records |
| `agent_name` | `str \| None` | Agent display name; `None` for search records |
| `artifact_id` | `str` | UUID of the artifact |
| `version` | `int \| None` | Artifact version at delivery; `None` on error |
| `content_hash` | `str \| None` | SHA-256 of the delivered content; `None` on error |
| `source` | `str` | `"cache_hit"`, `"fetch"`, `"broadcast"`, `"write"`, or `"search"` |
| `outcome` | `str` | `"content"`, `"empty"`, or `"error"` |
| `sequence_number` | `int` | Gap-free counter shared across all agents and sources |
| `instance_id` | `str` | Session identifier; matches `state_log` entries |
| `schema_version` | `str` | `"ccs.content_audit.v1"` |

### Source types

| `source` | Fires when |
|----------|-----------|
| `"cache_hit"` | Agent reads from its local cache (no coordinator round-trip) |
| `"fetch"` | Agent fetches from the coordinator (cache miss or refresh) |
| `"broadcast"` | Agent receives content pushed by a peer write (broadcast strategy) |
| `"write"` | Agent commits new content |
| `"search"` | Content returned via `SearchOp`; agent identity unknown |

### Cross-validation with state log

When both `content_audit_log` and `state_log` are enabled, `instance_id` is shared and
`content_hash` on write audit entries matches the corresponding state log commit entry.

---

## Crash recovery

When an agent crashes (OOM-kill, segfault) or livelocks (holds a grant indefinitely),
its `MODIFIED` or `EXCLUSIVE` grant blocks all other agents from writing to that artifact.
The crash-recovery extension reclaims stale grants automatically.

> **Default flipped in v0.9.0.** As of **v0.9.0**, `CrashRecoveryConfig()`
> defaults to `enabled=True` (it was `enabled=False` through v0.8.x), so a bare
> `CCSStore()` / `CoherenceAdapterCore()` now runs crash recovery. The first
> `CrashRecoveryConfig` construction per process emits a one-shot transitional
> `RuntimeWarning` flagging the change for anyone upgrading straight from
> v0.8.2 (removed in v0.10.0). To pin behavior and silence it, pass `enabled=`
> explicitly — `CrashRecoveryConfig(enabled=True)` to keep the new default, or
> `CrashRecoveryConfig(enabled=False)` to opt out. The v0.9.0 defaults were
> also retuned (`heartbeat_timeout_ticks` 10 → 120, `max_hold_ticks` 1000 →
> 900); see [CHANGELOG.md](../CHANGELOG.md) for the full migration notes.

### Enabling

```python
from ccs.coordinator.service import CrashRecoveryConfig

store = CCSStore(
    strategy="lazy",
    crash_recovery=CrashRecoveryConfig(
        enabled=True,
        heartbeat_timeout_ticks=120,
        max_hold_ticks=900,
    ),
)
```

The same `crash_recovery=` kwarg works on `LangGraphAdapter`, `CrewAIAdapter`,
`AutoGenAdapter`, and `CoherenceAdapterCore`.

### `CrashRecoveryConfig` fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Master switch (**default-on as of v0.9.0**). When `False`, `heartbeat()` and `recover()` are silent no-ops and the sweep does not run. |
| `heartbeat_timeout_ticks` | `int` | `120` | Reclaim a holder's grant if the gap between `now_tick` and the holder's last heartbeat is `>= heartbeat_timeout_ticks`. |
| `max_hold_ticks` | `int` | `900` | Reclaim a holder's grant if it has been continuously held in `MODIFIED`/`EXCLUSIVE` for `>= max_hold_ticks`, regardless of how recently the holder heartbeated. Bound the worst-case lock duration. |

**Tick semantics.** Ticks are a logical clock — the unit is whatever `now_tick` your application advances. For LangGraph, one node invocation per tick is a sensible default. For long-running tool calls or LLM calls, advance ticks at the granularity at which you can call `heartbeat()` or expect grants to be released.

### How it works

1. **Piggyback heartbeats.** Every `read()` / `write()` / `batch()` call automatically
   records a heartbeat for the calling agent. No application code change needed.
2. **Explicit heartbeat.** For long compute windows (LLM calls, blocking I/O) where no
   adapter method is invoked, call `heartbeat()` to signal liveness:
   ```python
   store.heartbeat(agent_name="planner", now_tick=current_tick)
   ```
3. **Reclamation sweep.** The coordinator reclaims any M/E grant whose holder either:
   - has not heartbeated within `heartbeat_timeout_ticks`, or
   - has held the grant for at least `max_hold_ticks` (regardless of heartbeat).
4. **Recovery after restart.** After a process restart or checkpoint reload, call
   `recover()` to invalidate the agent's stale local cache and re-seed its heartbeat:
   ```python
   store.recover(agent_name="planner", now_tick=current_tick)
   ```

### Cadence guidance

Call `heartbeat()` at least every `heartbeat_timeout_ticks / 3` ticks during long
compute windows. Per-step adapter methods (e.g., `before_node`, `batch`) already
heartbeat automatically.

### Composition rule

When using the `lease` strategy with crash recovery enabled, `max_hold_ticks` must
exceed `lease_ttl_ticks`. Equal or smaller values raise `ValueError` at startup.

### Flag-off behavior

With `enabled=False` (the opt-out — the v0.9.0 default is `enabled=True`),
`heartbeat()` and `recover()` are silent no-ops and the sweep never runs.
State-transition log output is then byte-identical to a build without crash
recovery. Note the inversion: omitting the `crash_recovery=` argument no
longer reproduces that output — pass `CrashRecoveryConfig(enabled=False)`
explicitly to get it.

### Disabling / rollback

To turn crash recovery off, pass `CrashRecoveryConfig(enabled=False)` explicitly.
(As of v0.9.0 the default is enabled, so omitting `crash_recovery=` no longer
disables it.) This is the rollback path if the default-on behavior ever surfaces
an issue in your workload — set `enabled=False` and the sweep stops immediately.
No data migration, no protocol incompatibility: the protocol behavior with
`enabled=False` is byte-identical to a build without crash recovery.

If you want to verify the sweep isn't reclaiming benign holders, watch the
state-transition log for `reclaim_heartbeat` and `reclaim_max_hold` triggers. If
they fire on agents that were healthy, raise `heartbeat_timeout_ticks` or
`max_hold_ticks` rather than disabling the feature.

### Reclamation diagnostics

When the sweep reclaims a grant, `CoherenceAdapterCore` logs a one-shot `WARNING`
on the `ccs.adapters.base` logger the **first** time that adapter instance
reclaims, with structured `extra` fields `trigger`, `agent_id_short`,
`artifact_id_short`, and `reclaim_count`. Subsequent reclamations on the same
instance are silent; a companion `DEBUG` log carries the full UUIDs. Bind a
handler to `ccs.adapters.base` to surface or ingest these events.

### Reference

For the formal protocol model (TLA+/TLC) covering single-writer, monotonic
versioning, and crash-recovery sweep invariants, see
[`formal/tla/README.md`](../formal/tla/README.md).

---

## Version retention and read-at-version

By default the coordinator keeps only the **current** version of each artifact.
Opt in to retaining a bounded history, and you can read back the exact bytes of
an earlier version.

### Enabling

Pass a `RetentionPolicy` to the registry (retention is off unless
`retain_versions=True`):

```python
from ccs.coordinator.retention import RetentionPolicy

policy = RetentionPolicy(max_versions=16, max_age_seconds=None)
```

`max_versions` keeps the K most-recent versions (including the current one,
which is never collected); `max_age_seconds` expires versions older than T
wall-clock seconds. Either axis can be `None` to disable it. GC is amortized —
it runs inline as new versions are committed; there is no background sweep.

### Durability

The in-memory `ArtifactRegistry` retains versions for the life of the process.
`SqliteArtifactRegistry` retains them **durably**, surviving a coordinator
restart, for in-process embedders. Enabling durable retention adds an
`artifact_versions` table via the store's first real schema-version bump
(v1 → v2), applied automatically and atomically the first time a v1 database is
opened. Durable retention is opt-in — a deployment that doesn't enable it stores
no version content. (The Claude Code hook/HTTP coordinator carries only content
hashes on the wire, so durable retention there is inert: there are no bodies to
store. It is an in-process-embedder feature.)

### Reading a version

```python
result = service.read_at_version(artifact_id, version)
```

The result is either a `VersionedContent` (`content`, `version`, `captured_at`,
`coordinator_epoch`) or a typed `VersionedReadRejection` whose `reason` is one of
six wire-stable constants:

| Reason | Meaning |
|---|---|
| `retention_off` | The registry is not retaining versions. |
| `unknown_artifact` | No such artifact. |
| `not_retained` | The version is not in the retained history (never captured, or collected / expired). |
| `current_version` | The requested version is the current one — read it through the normal `read` / `fetch` path, not the history surface. |
| `future_version` | The version is greater than the current version. |
| `epoch_mismatch` | The optional `expected_epoch` did not match (the store was reset). |

`read_at_version` is an **off-protocol read**: it grants no MESI state, joins no
invalidation set, and captures no read-generation fence claim — versioned reads
never affect a concurrent writer. It serves history only; current content is
always read through the protocol path.

### Reading from a stored coordinator

`agent-coherence-replay resolve` answers "bytes at version k" against a
`.coherence/state.db` on disk — useful for audit and post-hoc inspection:

```bash
agent-coherence-replay resolve --db ./.coherence/state.db \
    --artifact plans/plan.md --version 2 --json
```

The store is opened **read-only** (never created, never migrated). Output is
content-safe by default — `version`, `coordinator_epoch`, `captured_at`, content
hash and length — and emits the retained bytes only with `--include-content`
(base64 for binary) or `--output-file` (raw, written `0600`), so secrets don't
leak into terminals or CI logs. Each rejection reason and store error maps to a
distinct exit code with the wire-stable `reason` in the JSON envelope.

### Honesty boundary

Retention records and serves the bytes the coordinator committed at each version;
it does not make an agent's *use* of an old version safe. An agent that reads a
fresh current version through the protocol but writes content derived from an
older retained version still commits a fence-legal write of stale meaning — the
coherence guarantee is "write from the bytes your latest read returned," and
read-at-version makes an older version easy to fetch, so keep writes anchored to
the current read.

## Coherent workspace (`CoherentVolume`)

`CoherentVolume` brings the coherence guarantee to **plain files on disk** — no
framework required. It is an out-of-process coordinator *client*: it spawns (or
attaches to) a local coordinator over SQLite-WAL and routes reads and writes
through it. Your content stays on the real filesystem; the coordinator holds only
per-file MESI state, a content hash, and a version. Point a second volume in
another process at the same workspace and it attaches to the same coordinator, so
every process on the host shares one coherent view.

**Every volume coordinating a workspace must declare the same `managed` globs.**
An attaching volume adds no globs to a running coordinator's policy; only the
track and untrack commands change it. At attach, each volume checks every glob
it declared against the glob sets the coordinator publishes in its operator
view, by the coordinator's own rule: a glob is enforced when it is in the
coordinator's strict set and its tracked set and not in its ignored set. A glob
that is not fails the volume closed, naming it: under `on_error="strict"`
construction raises, and under `"degrade"` the volume warns once and runs
detached. When the sets cannot be read at all, because the coordinator is older
and publishes only counts, or is the Node coordinator, the volume fails closed
the same way and says that enforcement could not be confirmed.
`vol.managed_glob_enforcement()` returns the same three-way answer. The
comparison is literal and taken once, at attach: a coordinator that ignores the
paths under a broader pattern, or a path untracked after attach with the untrack
command, is not detected, and such paths answer every operation as untracked. A
fleet with mixed globs is still unsupported; it now refuses instead of running
unguarded.

```python
from ccs.adapters.coherent_volume import CoherentVolume

vol = CoherentVolume(workspace_root, managed=("plans/**", "memory/**"))
data = vol.read("plans/plan.md")            # bytes — registers a SHARED view
vol.write("plans/plan.md", revise(data))    # stale view? denied fail-closed
data = vol.reacquire("plans/plan.md")       # recover: clear the stale view + fresh read
```

| Parameter | Default | Meaning |
|---|---|---|
| `workspace_root` | — | Directory the volume manages; the coordinator's state lives in `<root>/.coherence/` |
| `managed` | `()` | Glob patterns for the files under coordination; unmanaged paths bypass the volume |
| `on_error` | `"strict"` | `"degrade"` warns once and falls back to plain IO instead of raising on a coordination failure. A [caller principal](#caller-principal) refusal the volume cannot recover from is a definite answer, not a failure, and raises `CallerPrincipalRefused` in both modes |
| `on_stale_read` | `"allow"` | `"raise"` — deny a re-read of a managed file whose on-disk bytes changed out-of-band |
| `on_stale_write` | `"raise"` | `"allow"` — restore last-writer-wins over a foreign edit (not recommended) |
| `config` | `None` | Coordinator settings as a `LifecycleConfig` (`from ccs.adapters.claude_code.lifecycle import LifecycleConfig`); `None` uses the defaults. Only the volume that starts the coordinator applies it |

`vol.session_id` is the volume's session with the coordinator, and it stays the
same for the volume's lifetime: `reacquire()` and the retries inside `write_cas`
clear a stale view by starting a fresh attempt under that same session. A forked
child gets a session of its own. The fresh attempt travels in the request's
`agent_id` field, so the volume needs a coordinator that reads it: this package's
from 0.13.0, or the Claude Code plugin's from 0.3.0.

To name the attempt to something else — a registry that joins the coordinator's
`/status` `sessions[].agent_id` against its writers, for example — read
`vol.agent_id`: the identity the coordinator keys the volume's next request on,
in the same string form `/status` reports. `vol.incarnation` is the per-attempt
part of it, the value every request carries in its `agent_id` field. Both change
only when an attempt starts (`reacquire()`, `write_cas_at`, `atomic_publish`, a
`write_cas` that retries or first releases its own `write()` grant, and a forked
child) and are stable in between, so read them after the operation whose
attempt you are reporting. `vol.root` is the workspace root as the volume
resolved it at construction (absolute, symlinks followed); it never changes.

### Concurrent writers: `write_cas`

Plain `write()` denies the *sequential* stale view. For **concurrent** same-key
contention, `write_cas(path, make_content)` is the optimistic path: it reads the
file, runs your `make_content` closure on the current bytes, and commits only if
the version is unchanged. The loser of a race gets the winner's value re-fed to
its closure through a bounded retry — one writer wins each round and no update is
silently dropped. A single-shot variant, `write_cas_at(path, expected_version,
content)`, commits against an explicit version with no retry loop. See the race
live: `python -m examples.concurrent_writers.main` runs two threads through the
identical update — a plain file loses one write, `write_cas` preserves both.

A volume can `write()` a file and then commit it through the optimistic lane. A
successful `write()` leaves the volume holding that file's write grant after it
returns; a `write_cas` of that file, `write_cas_at`, `atomic_publish` and
`reacquire()` release it before they go further, so the volume is not refused by
its own grant. A `write_cas` of another file leaves it held unless it has to
retry, because that grant cannot refuse it. The release is one extra request, made only after a `write()`:
a volume that only reads and commits optimistically never pays for it. Each of
these starts a fresh attempt, as `reacquire()` and every `write_cas` retry do,
and a fresh attempt clears every refusal the volume had, not only the one on the
file it commits. Re-read any other file before a plain `write()` of it.

When the retry budget runs out, `write_cas` raises `CasRetriesExhausted`. Its
`last_conflict_reason` says why the last attempt was refused:

- `version_mismatch`: other `write_cas` callers kept committing first. A later
  retry can win.
- `other_holder`: another agent wrote the file with a plain `write()` and still
  holds it. The version has not moved, so re-reading returns the same bytes.
- `caller_in_transient_state`: another agent's plain `write()` landed between
  your read and your commit. That agent may still hold the file, as with
  `other_holder`.
- `stale_read_generation`: your read was taken under a grant the coordinator has
  since reclaimed. `reacquire()` and re-read.

A `CoherentVolume` that wrote a file keeps holding it until its own next
`write_cas` of that file or any `write_cas` retry, `write_cas_at`,
`atomic_publish` or `reacquire()` releases it. Closing the volume or exiting the process does not release it, and
neither does a `session-stop` that names only `vol.session_id`, because each
attempt holds its grants under its own `agent_id`. The coordinator
takes the file back once the holder has made no coordinator calls for
`grant_heartbeat_timeout_sec` (600 s by default), or has held it for
`grant_max_hold_sec` (1800 s by default).
Both are `LifecycleConfig` fields, passed as `config` to the volume that starts
the coordinator. A Claude Code session releases what it holds when its turn ends.
So after `other_holder`, retry once the holder has released, not in a tight loop,
or use a plain `write()`, which takes the file over (the previous holder's next
write is then refused until it re-reads).

When a plain `write()` loses its commit to a peer on the volume (or MCP) path,
the raised `CommitPreempted` is **terminal for that attempt, not a transient to
retry blindly**: the version the write assumed no longer holds. Recover by
`reacquire()`-ing, re-reading the fresh version, and reconciling your change onto
it before committing again — a plain retry of the same bytes just loses the same
race.

### Atomic multi-file publish: `atomic_publish` (v0.12.0+)

`write_cas_at` lands one file. When an agent edits a *set* of files that must stay
consistent — a plan and its manifest, a config split across files —
`atomic_publish` lands them **all-or-nothing**:

```python
versions = vol.atomic_publish([
    ("proj/plan.md",     plan_version,     new_plan_bytes),
    ("proj/manifest.md", manifest_version, new_manifest_bytes),
])   # -> {"proj/plan.md": 2, "proj/manifest.md": 3}
```

Each member commits only if it is still at the `expected_version` you pass; if
every member matches, the batch **commits at the coordinator as one unit** and
every file is then materialized, and if any member moved, the **whole** publish is
held (`StaleView` / `CasVersionConflict`) with **nothing committed and no file
written** — a torn *commit* is never a reachable state. A single-member call takes
the direct CAS path; a multi-member call opens a
[snapshot session](#multi-artifact-snapshot-sessions) so the versions it checks
are captured at one point (no member read across a peer commit), which adds a
small capture→commit window — a peer winning it holds the publish rather than
tearing it. Recover the same way as a denied write: `reacquire()`, re-read the
fresh versions, and re-publish from them. A single-member publish accepts
arbitrary bytes; a multi-member publish requires UTF-8 text content.

The all-or-nothing guarantee is at the **coordinator commit**. Disk materialization
runs after it and is best-effort: every file is staged to a temp then renamed, so a
disk fault fails before any rename (disk stays uniformly old) and a rename failing
partway raises a typed `PublishMaterializationError` naming exactly which files
landed — never a bare error implying nothing published. A crash between renames can
still tear the on-disk set (no POSIX multi-file atomic rename exists); on that error
the coordinator is ahead of disk, so write each file that didn't land again with
`write()`, using the bytes you published (until then, `read_with_version` refuses
that file, because the bytes on disk are not what the coordinator recorded). Don't
retry the publish — it would version-mismatch. Run it:
`python -m examples.atomic_publish.main` (offline, deterministic, no keys), or add
`--baseline` to see the file-by-file torn pair it prevents.

**Volume-mediated writers only.** The staleness `atomic_publish` detects is
*version* drift, and only writes routed through a volume advance versions. An edit
that bypasses the volume — a human in an editor, a formatter, a script writing a
member directly — advances nothing, so a multi-file publish cannot see it: the
batch still commits and the out-of-band edit is silently overwritten. (A
single-file publish takes the direct CAS path, whose content-checked comparand
read fails closed on such an edit, as does plain `write()` — see
[Foreign-edit guards](#foreign-edit-guards).) Use `atomic_publish` only for file
sets whose every contending writer goes through a volume; for files that humans
or out-of-band tools also edit, use `write()` — its foreign-edit guard covers
exactly that case.

### Foreign-edit guards

Files also change *outside* the fleet — a human edit, a formatter, a regenerating
script. The volume checks a content hash at both boundaries:

- **Write boundary (on by default).** If a managed file's on-disk bytes changed
  out-of-band since this volume last read or wrote them, `write()` raises
  `StaleView` instead of clobbering the foreign edit. Recover with `reacquire()`:
  fresh read → re-derive → re-write.
- **Read boundary (opt-in).** With `on_stale_read="raise"`, re-reading a managed
  file whose bytes changed out-of-band raises `StaleView` instead of returning
  bytes the rest of your state wasn't computed from. In strict mode the
  coordinator enforces the same check server-side.

A volume never denies its own just-written bytes: the benign window between a
commit and its disk write is recognized and suppressed.

These guards are **content-hash checks at the volume boundary** — best-effort
point-in-time detection, not filesystem interception. An edit that bypasses the
volume is caught at the *next* volume read or write of that file, not blocked as
it happens.

They cover `read()` and `write()`. The CAS paths check *versions* instead:
`write_cas` / `write_cas_at` still fail closed on a foreign edit (their
content-checked comparand read wedges rather than clobbering), but a multi-file
`atomic_publish` **does not check disk content at all** — see the scope note in
[Atomic multi-file publish](#atomic-multi-file-publish-atomic_publish-v0120).

### The `open()` shim (demo-grade)

For code you'd rather not rewrite, `coherent_workspace()` / `install()` patch
`open()` and `pathlib` so managed-path opens route through the volume unchanged.
It covers text and binary read/write via `open()`/`pathlib` — not raw `os.open`,
subprocess redirection, `mmap`, or append/update modes, which delegate to the
original `open()` unchanged. The explicit `read`/`write`/`reacquire`/`write_cas`
API is the supported contract; the shim is a convenience.

Run the demo: `python -m examples.coherent_volume.main` (offline, deterministic,
no keys) — it reproduces the silent lost update, then prevents it.

### Worktrees and the workspace boundary

`CoherentVolume` coordinates by *path under one `workspace_root`*. Two git
worktrees of the same repo are separate directory trees, so `plans/plan.md` in
worktree A and `plans/plan.md` in worktree B are **different physical artifacts** to
the volume — a write in one does not invalidate the other unless both processes route
through the *same* shared workspace root. To make per-worktree sessions coordinate,
point every volume at one common root (for example the primary checkout). The Claude
Code plugin does this for you: it resolves the parent repo via
`git rev-parse --git-common-dir` so sessions in sibling worktrees share one
coordinator (`src/ccs/adapters/claude_code/resolver.py`).

### When you don't need this

Coherence is worth adding only when agents actually share mutable state through a
back channel your framework doesn't already serialize. You can skip it when:

- **Every agent owns an isolated workspace.** If sessions never write the same
  artifact — separate worktrees, separate branches, separate keys with no shared
  root — there is no lost update to prevent. (The moment they *do* converge on one
  file or one shared root, the race is back.)
- **A single database already arbitrates the writes.** If your shared state is rows
  behind one transactional store, its own transactions and row locks already give
  you last-committer-wins with no torn state. `CoherentVolume` targets *plain files*
  and in-memory agent state, where nothing is arbitrating.

The liveness tradeoff runs the other way: a crashed holder does not deadlock the
fleet. A stalled `MODIFIED`/`EXCLUSIVE` grant is bounded and auto-reclaimed by the
best-effort crash-recovery sweep (per artifact, never a global lock), and the
`write_cas` / `atomic_publish` CAS paths hold no lock at all — a loser just re-reads
and retries.

### Cross-host mode (experimental, default off)

Everything above is single-host. An experimental, demo-grade remote mode — gated
entirely by `CCS_REMOTE_COORDINATOR=1` (default off, loopback path byte-unchanged)
— lets a volume connect to a coordinator on another host: `CCS_REMOTE_HOST` /
`CCS_REMOTE_PORT` name the endpoint, and the bearer secret arrives via
`CCS_REMOTE_SECRET_FILE` (a mounted file — never an inline env var).

The client can speak **verified https** to a TLS-terminating front: set
`CCS_REMOTE_TLS=1` to use https with enforced certificate verification (and
`CCS_REMOTE_CA_FILE` to trust a private certificate authority). Verification is
fail-closed — an unverifiable certificate means the bearer is never sent — and a
verified-https connection needs no `CCS_REMOTE_INSECURE=1` acknowledgement. Without
https the transport is plaintext HTTP, so encryption is your out-of-band
responsibility (a WireGuard tunnel or a TLS-terminating proxy); to stop a silent
leak, the client **refuses to send the bearer to a non-loopback host** over
plaintext unless `CCS_REMOTE_INSECURE=1` acknowledges you secured the link
yourself, raising a typed `InsecureTransportRefused` otherwise. Symmetrically, a
coordinator that binds **beyond loopback** refuses to start unless the operator
asserts `CCS_TLS_TERMINATED=1` (a TLS front is present) or `CCS_SERVE_INSECURE=1`
(an acknowledged insecure link) — these are operator assertions, not enforcement.
Setup, the Docker two-container runner, and the full security boundary and
certificate requirements live in
[`examples/cross_host/README.md`](../examples/cross_host/README.md) and
[the security guide](security.md).

> An internal, experimental seam formalizes what a networked registry backend
> would have to provide; it is not a public extension point, and there is nothing
> for end users to configure today.

## BYO substrate bindings (`CoherentRow`, `CoherentObject`)

`CoherentVolume` brings coherence to files on disk. **BYO-substrate bindings** bring the same coherence to shared state that lives in a store you already run — a Postgres row, an S3 object — while the coordinator holds only coherence metadata (a monotonic version, per-agent MESI, a fixed-width `content_hash`, and optionally an opaque substrate token) and **never the bytes**. The bytes stay in your substrate; the coherence layer drops *under* it.

Install the binding you need (the drivers are optional extras):

```bash
pip install "agent-coherence[coherent-row]"     # Postgres — psycopg v3
pip install "agent-coherence[coherent-object]"  # S3 — boto3
```

### What you get over the substrate's own CAS

A substrate's native conditional write (`UPDATE … WHERE version = ?`, S3 `If-Match`) already rejects a single lost update — *at write time*. The binding adds the **cross-agent** layer over it:

- **Invalidation-before-act.** A peer's commit marks your cached read stale, so your next binding-mediated read/act is denied *before* you act on the moved state — the bare CAS never surfaces that.
- **Cross-substrate uniformity.** The same typed conflict (`StaleView` / `CasVersionConflict`) and the same `deny → reacquire()` recovery over a row, an object, a file (`CoherentVolume`), or a store key (`CCSStore`) — one coherence surface, not per-substrate error handling.

```python
from ccs.adapters.coherent_row import CoherentRow

row = CoherentRow(dsn="postgresql://…", table="workspaces")
data, token = row.read("ws-42")               # (bytes, token) from ONE read
# ... a peer commits a new version through the binding ...
row.commit("ws-42", expected_token=token, new_bytes=revised)
#   -> StaleView: your cached view moved. reacquire() for fresh bytes, re-decide, retry.
```

`CoherentObject` (S3) has the identical surface; its token is the object ETag, captured from the `put_object` response (never computed).

### Guarantee tiers (honest by construction)

Every binding declares a `CapabilityDescriptor` with a **tier**; the guarantee wording a user sees is derived from the tier, so a weaker binding can never present as enforcement:

| Tier | Substrate shape | Honest guarantee |
|---|---|---|
| `native-CAS` | atomic conditional write (PG version column, S3 `If-Match`) | no-lost-update on the version-CAS axis, **single-host**, with the timeout asterisk below |
| `detect-only` | no atomic conditional write | catches a *sequential* stale-read→write; cannot prevent a concurrent race |
| `forward-only` | an action / RPC (a Slack post, a Gmail send) — no object, no token | **effect ordering only**: decision-input freshness via deny-before-act; no CAS, no rollback, no duplicate-effect prevention |

### Coherence Manifest

A declarative manifest binds each artifact to a substrate + connection + tier, and is a named **trust boundary**. Credentials are references, never literals (`secret-file:PATH`, `aws-default`, `secret:uri`, or a least-preferred `env:VAR`); connection targets are SSRF-constrained — the deny runs on the *resolved* address (metadata/link-local hard-denied, RFC-1918 allowed only under an explicit `CCS_SUBSTRATE_ALLOW_PRIVATE` opt-in), and a plaintext credential to a routable host is refused unless `CCS_SUBSTRATE_INSECURE` is acked. `dry_run()` prints each artifact's tier at config time. See `docs/examples/manifest.example.yaml`.

### Least-privilege (provision the substrate down to what the binding needs)

- **Postgres** — a dedicated, login-limited **non-owner** role granted only `SELECT, UPDATE` on the one table (no `ALTER` / `TRIGGER` / `DELETE` / re-grant, `NOSUPERUSER NOCREATEDB NOCREATEROLE`), plus an **owner-managed** `BEFORE INSERT/UPDATE` trigger that mints `version := OLD.version + 1` from the *stored* prior — so a client that supplies its own `NEW.version` cannot forge it. `CoherentRow.provisioning_sql(...)` emits (never executes) both.
- **S3** — an IAM policy scoped to the exact key/prefix ARN with `s3:GetObject` + `s3:PutObject` only and explicit denies (no `s3:*`, no `s3:DeleteObject`), plus an **owner-managed** bucket policy that *requires* conditional writes (`Deny s3:PutObject` when `Null s3:if-match true`, with the `s3:ObjectCreationOperation` multipart exemption). `CoherentObject.conditional_write_bucket_policy(...)` / `least_privilege_iam_policy(...)` emit the verified shape.

### Honest scope

- **The read-generation fence over a substrate is a roadmap item, not shipped.** v1 OCC writers ride the fence's admit-on-absent path + the version-CAS. Nothing in these bindings claims the fence.
- **`native-CAS`, with the timeout asterisk.** The substrate CAS prevents the concurrent single-host lost update on the token axis; a coordinator-timeout *after* a durable substrate write is reconverged by a token-identity re-read, and registry↔substrate agreement is a *detectable signal*, not a held guarantee.
- **Single-host, subtractive.** When the substrate is itself distributed (S3, managed Postgres), the no-lost-update guarantee is the *substrate's* and is identical with or without this layer; the adapter's contribution (invalidation, uniformity) is single-host. Never run the S3 CAS loop through a Multi-Region Access Point / cross-Region replica, and never place agents on two hosts against one distributed substrate.
- **Coordinator-behind-substrate is unbounded in v1.** If a writer's substrate write lands but its process dies before the coordinator bump, peers are not invalidated until the *next* binding-mediated read of that artifact — a peer acting on an already-cached read is unprotected until it re-reads. Repair-forward is a roadmap item.
- **More backends are demand-gated, not shipped.** A Letta vendor-memory sidecar is a post-v1 candidate: Letta exposes no atomic conditional-write token, so a binding could only *detect* a sequential stale write via a client-held content shadow, not prevent a concurrent one — it would be a `detect-only` tier and lands only when a concrete need pulls it in.

### Try it, then harden

```bash
python -m examples.coherent_row.main               # offline, no keys — an in-memory substrate stand-in
python -m examples.coherent_object.main --baseline # see the silent stale act the binding catches
```

The demos run offline against a local coordinator with an in-memory substrate stand-in so the coordinator-mediated value (invalidation-before-act) is visible with zero setup. For production, point the same binding at a real Postgres / S3 and provision the least-privilege role/policy above. The tier-honesty conformance suite exercises both bindings against **real** substrates behind the `real_substrate` pytest marker (credentialed; `CCS_TEST_PG_DSN`, `CCS_REAL_S3_BUCKET`) — Moto/LocalStack are excluded because they serialize and would false-green a concurrency test.

## Workspace versioning & restore (`WorkspaceVersioner`)

Everything above keeps shared state *coherent while agents write it*. Workspace versioning answers a different question: an agent mutated a workspace whose members live in different backends — files on disk, objects in S3 — the attempt failed, and you want the workspace **back the way it was**, with per-member honesty about what can and cannot come back.

A **workspace checkpoint** is a named manifest over heterogeneous members, captured as a **skew-declared cut**:

- **Per-member capture.** One read per member records a restore pointer (the S3 versionId; the coordinator's content-state version for a file), a content fingerprint, and a timestamp. The checkpoint stores pointers and fingerprints — never a second copy of your bytes (S3 history lives in your bucket; file history in the coordinator's bounded retention).
- **The window is declared, not hidden.** Members are captured one at a time, so the manifest carries the capture window `[window_min, window_max]` rather than pretending the cut was instantaneous. After the window closes, every member is re-read once; any observed movement marks that member `dirty_during_window` — "not verified quiescent", never silently torn.
- **ABSENT is a fact.** A member missing at capture is recorded as absent — distinct from present-and-empty — and restore includes **delete legs**: a member captured absent that exists live is deleted to match the manifest (on a versioned bucket this mints a delete marker, so the object's history survives).

```python
from ccs.adapters.workspace import WorkspaceVersioner

wv = WorkspaceVersioner(service=service, owner=owner, file_resolver=resolver)
wv.add_file_member(source, "ws/notes.md")
wv.add_object_member(binding, "reports/summary.txt")   # a CoherentObject (S3)
wv.add_forward_only_member("actions/deploy-step")      # an action surface — named, not captured

checkpoint = wv.checkpoint("before-migration")         # pins on by default
report = wv.restore(checkpoint.record.checkpoint_id)
for member in report.members:
    print(member.member_path, member.outcome)          # per-member terminal truth
```

### Restore: one conditional leg per member, and it always concludes

`restore()` drives one conditional write per member under a **termination contract**: every member reaches exactly one terminal outcome, and the restore concludes with a frozen per-member report — never a livelock, never silent partial success.

| Outcome | Meaning |
|---|---|
| `restored` | the captured bytes landed via the member's conditional write — over whatever was live at that moment, including content committed after the capture; the report below says, per member, what that write discarded |
| `converged` | the live state already matched the manifest — nothing written |
| `conflict` | a live foreign writer won; the re-drive budget is bounded, and **the foreign writer's state survives** |
| `target_lost` | the captured version is no longer reachable (expired retention, a vanished S3 version), or the member itself can no longer be driven safely (its path became a symlink, a hardlink with an outside co-owner, or a non-regular file) — reported, never substituted |
| `forward_only_skipped` | a declared action surface (or an uncapturable member) — enumerated, skipped |
| `held_unconfirmed` | a write whose outcome could not be confirmed — held, never guessed |

**What a restore does not promise.** A restore is not a merge, and nothing on this path refuses a write: a member whose content moved after the capture is put back over, and that later content is gone. What the run gives you is the record of it. Every leg already reads the live state to build its conditional write's comparand, and each member now carries that read as an **observation** in one of five states:

| Observation | Meaning |
|---|---|
| `observed_differs` | a live state was read, it differed from the capture, and the write discarded it — the only state that names the version overwritten and a digest of the content overwritten |
| `no_live_state` | the write landed on nothing (create-on-absent) — it discarded nothing |
| `present_not_comparable` | a delete leg's probe established that live state existed and destroyed it, but read no comparand — so it names neither version nor digest |
| `no_write_attempted` | the member reached its terminal without a write decision: converged, forward-only skipped, or absorbed before any write was issued (a wedged view, an exhausted re-drive budget). Also what a resumed member reports when its recorded outcome proves no write landed |
| `not_recorded` | this run holds no observation for the member. Either a leg ran and its outcome was lost — an unconfirmed commit, a reconciled unknown write, or a failure absorbed from inside the leg, any of which may have destroyed live state — or no leg ran and the recorded outcome does not itself prove none ever did. Never read it as clean |

Nothing was added to the substrate wire for this: the values ride the read each leg was already making, and on an S3 leg the ETag stays the comparand while the `versionId` is the pointer. The human report flags every state the exit code below can fire on, on that member's line: `overwrote-differing-content` — and, where the leg named one, `overwritten-version=<pointer>` — for `observed_differs`, `destroyed-uncompared-content` for `present_not_comparable`, and `overwritten-content-not-recorded` for `not_recorded`. The two states it does not flag are the two that exit code does not fire on, and their lines read exactly as they did before. Under `--json` each member carries a nested `observation` block (`state`, `pointer`, `fingerprint`) alongside the keys it already had. If you would rather a run like that be a failure, `restore --exit-nonzero-on-discarded-content` makes it exit `4` — see the exit codes below; it is read after the engine returns, so it reports the write and cannot prevent it. One caveat before you put it in a loop: the observation is not persisted, so re-running a restore that *did* write reports `not_recorded` for those members and exits `4` again, whatever the second run found. It answers "did this run discard anything", not "is this checkpoint settled", so it does not belong in a retry-until-zero script.

Each member's leg rides its own backend's arbitration: S3 legs are a native conditional write (`If-Match` — the substrate arbitrates a racing foreign writer); file legs are a version-checked write whose foreign-edit signal is **detection only** (`no-arbiter`) — a foreign edit racing a file restore is detected and reported as a typed conflict, never presented as substrate arbitration. Restore progress is durable, so a restore interrupted mid-way resumes idempotently: already-terminal members are skipped, and a member whose live state already matches concludes `converged` without a second write.

### The honesty model: restore tiers and pin states

Every member carries a **restore tier**, derived from what its backend can actually promise — never asserted:

| Tier | Backend shape | What it honestly means |
|---|---|---|
| `restorable` | versioned S3 bucket (history + a per-version legal hold) | the captured version exists and a pin can back it |
| `restorable-unpinned` | file member over coordinator retention | history exists **now** and may expire — the retention window is a bound, not a pin |
| `forward_only` | unversioned bucket, action surface, unconfirmable pointer | described in the manifest, not restorable — stated at capture, not discovered at restore |

The durable truth is the **pair** `(restore_tier, pin_state)`. `restorable` is backed only by `pin_state="held"`; the pair `(restorable, unpinned)` — a capture whose pin legs haven't run, or a run that died before them — is rendered as **claimed-but-not-yet-backed**, never as a plain restorable state. Pins are fail-closed and loud: an S3 member whose bucket has no Object Lock configuration is durably downgraded to `restorable-unpinned` with `pin_state="pin_unavailable"` — the tier and the pin state land in one write, never silently.

**The file-member retention caveat, precisely.** File members ride the coordinator's [declared version retention](#version-retention-and-read-at-version), which is a bounded window (a count/age policy), not a per-version hold. A checkpoint pin on a file member **verifies** — it checks the captured version is currently retained and still matches its fingerprint — it **cannot extend** the window or exempt the version from the policy. If the version has aged out by restore time, the restore reports that member `target_lost` rather than restoring different content; expiry always surfaces, never silently. S3 members are the stronger case: their pin is a legal hold in *your* bucket on the captured version — the substrate's own retention, at your bucket's configuration and cost.

**Dropping a pin.** `WorkspaceVersioner.release_checkpoint(checkpoint_id)` releases the pins a checkpoint holds — the S3 legal holds it placed, and file members' verification pins. The manifest survives: this releases pins, it does not delete the checkpoint. Four things to know before calling it. It is **one-way** — a released member is terminal, `pin_checkpoint` will not re-drive it, and the captured version becomes eligible for lifecycle expiry and version-targeted delete from that instant. S3 members must be **re-declared on the versioner you release from** (`add_object_member`), through the **same binding and key that placed the hold**: the pre-flight only checks that an object member is declared at that member path, so a mismatched binding or key records the member `released` while the hold quietly survives. And a hold **shared** with another checkpoint survives until the last holder releases — with two bounds worth knowing, since a legal hold is a retention control: the sharing scan walks only that versioner's registry, so a holder recorded elsewhere (another workspace, a person, another tool) is invisible to it, and the scan alone cannot cover the instant between its last check and the drop itself — that instant is handled by re-reading *after* the drop and putting the hold back if a peer claimed it, so what remains is the narrower case where putting it back also fails, which is logged. Finally, it is **idempotent but not self-healing**: the release records `released` before it drops the substrate hold, so a crash or an untyped substrate error between the two strands a live hold on an already-terminal row — retrying `release_checkpoint` will not detect or repair it, since the row already reads terminal. Calling it **concurrently** fails differently, and no longer strands: two processes releasing one checkpoint both decrement its pin refcount, and the registry fails the surplus decrement with a `ValueError` — but that raise lands after the hold has already been dropped, so it costs you bookkeeping accuracy rather than a live hold. It does abort the rest of that checkpoint's members, which stay `held` for a later call. Treat a `ValueError` as "some members may have been released", never as "nothing happened". The recovery in both cases is calling `CoherentObject.release_legal_hold` directly on the binding, by version — the released row keeps the `native_token` you need for it.

**Binary members (v1 limitation).** A file member whose bytes are not UTF-8 text is refused at capture with a typed error — before anything persists. The file restore path rides a text wire, so capturing a member that could never be restored would be a silent over-claim.

### The CLI: `agent-coherence-workspace`

```bash
agent-coherence-workspace checkpoint before-migration --file notes.md --file plan.md
agent-coherence-workspace list
agent-coherence-workspace status <checkpoint-id>
agent-coherence-workspace restore <checkpoint-id>
```

Four verbs — `checkpoint` (with repeatable `--file` / `--forward-only`, and `--no-pin` for capture-only), `list`, `status`, `restore`. Every verb takes `--root` (override the workspace root; default walks up to the git root) and `--json` (machine-parseable output). `status` renders every member's `(restore_tier, pin_state)` pair — including the claimed-but-not-yet-backed label — plus torn-cut flags and restore outcomes; `checkpoint` output always carries the file-retention caveat. The CLI keeps its own durable state in `<root>/.coherence/workspace.db`.

Under `--json`, every error path *also* emits a one-line JSON envelope on stdout — `{"kind": "error", "exit_code": …, "reason": …, "message": …}` — so a script never has to parse stderr prose; the human message stays on stderr unchanged. Capturing a second checkpoint under an existing name is never refused (names are labels, not unique keys), but the prior ids are printed so the ambiguity is never silent — `status` and `restore` always target an id, not a name.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | the verb succeeded (restore: concluded with no member in `conflict` / `target_lost` / `held_unconfirmed`) — never a claim that nothing was overwritten: a restore that put a member back over content committed after the capture also exits `0`, and says so per member in the report |
| `1` | not in a git repository, or a validation error (no members, a `..` traversal in a member argument), or a typed contention error |
| `2` | a typed refusal: a non-UTF-8 member, an unknown checkpoint id, a persist failure, or a member path that fails containment at access time — a workspace escape, a symlink component, a hardlinked regular file with a co-owner outside the root, a non-regular file (FIFO, socket, device), or a `.coherence/**` self-target |
| `3` | the restore **concluded**, but at least one member ended in `conflict` / `target_lost` / `held_unconfirmed`, or the restore registration was refused — the per-member report on stdout is the truth; the exit code just tells you to read it |
| `4` | opt-in, and only with `restore --exit-nonzero-on-discarded-content`: the restore concluded clean by the codes above, but at least one member's write discarded content the checkpoint did not hold, or the run holds no record of what that member's write overwrote. Without the flag the same run exits `0`, and both producers of `3` take precedence over it. It is evaluated after the engine returns, so it reports the writes and never prevents one |

**Where a refusal lands.** A containment refusal at **capture** is a hard exit-`2` refusal with nothing persisted. The same refusal raised inside a **restore** leg is deliberately not: the termination contract absorbs it into that member's `target_lost` so every other member still concludes, and the refusal text becomes that member's outcome detail — exit `3`, never a checkpoint left stuck mid-restore.

**S3 members and the CLI.** S3 object members are captured and restored via the Python API shown above — their bindings carry credentials the CLI cannot (and should not) reconstruct. `restore` on a checkpoint with pending S3 members refuses cleanly and points you at the Python API; `status` and `list` still render those members honestly.

**The HTTP surface.** The coordinator serves the workspace verbs over local HTTP — `POST /workspace/checkpoint`, `GET /workspace/checkpoints`, `POST /workspace/restore/status`, `POST /workspace/restore/member`, and `POST /workspace/restore/register` — the same routes the Python API and the CLI ride. Pin orchestration has no HTTP route by design: pins are placed and released through the Python API only — `pin_checkpoint` and `release_checkpoint`, described above — because the substrate bindings that hold them carry your credentials, and credentials never belong on the coordinator's wire.

### Try it

```bash
python -m examples.workspace_versioning.main             # the guarded demo
python -m examples.workspace_versioning.main --baseline  # loss-first, then guarded
```

Offline, deterministic, no API keys. `--baseline` first demonstrates the loss (no history, no pointer — the original bytes are unrecoverable), then the guarded arm prevents it: a file restore, an S3 If-Match restore, a delete leg restoring the ABSENT fact, a sustained foreign writer honestly reported as `conflict` (the foreign winner survives), and a forward-only member enumerated and skipped. Exit code `0` iff the whole contract holds — the exit code is the contract.

### For implementers: the conformance corpus

The workspace family ships in the packaged conformance corpus (`ccs.testing.substrate_conformance`), so a foreign implementation of workspace checkpoint/restore can be tested against the same scenarios ours is. Implement the `WorkspaceConformanceBinding` protocol and **declare your capabilities honestly** (`declares_versioned` / `declares_pinnable` / `declares_restart_survival`); the suite splits into **MUST-MATCH** scenarios every implementation must reproduce (one-winner restore arbitration, torn-cut detection, bounded termination under contention, restore-as-forward-commit, ABSENT-is-a-fact) and **DECLARED** scenarios pinned to your own declarations (versioned vs unversioned history, pinnable vs `restorable-unpinned`, restart survival) — a binding passes by satisfying observable outcomes, never by mimicking our internals. The corpus imports and runs without pytest; `pip install 'agent-coherence[conformance]'` adds it so skipped scenarios report as skips under your test runner. The restore-registration design is model-checked (`formal/tla/WorkspaceVersion.tla`, run in CI).

The corpus's race scenarios run across real OS **process** boundaries: contenders are spawned processes rendezvoused at a barrier before their racing section, so a binding whose "compare-and-swap" is a read-then-write under an in-process lock is caught rather than false-greened — in-process serialization satisfies no clause of the contract. A binding that genuinely cannot be raced across processes (an in-memory fake) is not silently skipped; it declares the reason and the run report prints the exemption. Two further axes ship with the corpus. **Durability**: a binding's `CapabilityDescriptor` can declare the regime an acknowledged write survives (`in-process`, `process-crash`, or `os-crash`) together with the configuration facts the claim rides on; the corpus verifies the process-crash grade locally by SIGKILLing a writer after its acknowledged commit and reading the store back, while fsync-grade discrimination and managed-cluster failover runs are recorded as declared exemptions with their reasons rather than skipped. **The claim ladder** (`ccs.testing.claim_ladder`): every guarantee this project claims is a rung pairing its guarantee text with the exact README claim-phrases it backs and the tests that would fail without it — the repo's own suite resolves every named test at collection time and drift-guards the README in both directions, so a claim cannot outlive its proof, and no cross-host rung exists to claim.

**Scope, honestly.**

- Single-host coordinator: checkpoints, pins, and restore progress live in local coordinator state and make no cross-host claims.
- Restore is over **artifacts, never effects** — files and objects come back; a sent message does not.
- File members are detection-only (`no-arbiter`); only backends with a native conditional write arbitrate. And restore is a **forward** commit carrying old bytes — versions strictly increase, history is never rewritten.
- Restoring file members while a live coordinator session is running bypasses that session's grants — the session learns of the change on its next read, not before. The CLI warns when it detects a live coordinator; it does not refuse.
- Torn-cut detection has a tail window: a write landing in the final instants between the quiescence check and the manifest persisting can go undetected. And concurrent restores assume a single controller — run one restore at a time. Two restores of the *same* checkpoint are the obvious case; the sharper one is two restores of **different** checkpoints that overlap on a member. Those are not cross-serialized: each leg compares against the state it read, so both can honestly report the member `restored` while only the last write survives on disk. The engine serializes its own runs; ordering across controllers is the operator's contract in v1.

## Multi-artifact snapshot sessions

An agent that reads *several* artifacts one at a time can see a torn combination:
`plan.md` from before a peer's commit and `config.json` from after it. Each
individual read was current; the set never coexisted — read-skew. A **snapshot
session** pins a consistent cut of the artifacts you name, captured at a single
point, and serves every session read from that cut while peers keep writing.

Over HTTP, against a running coordinator (the same one `CoherentVolume` spawns):

| Endpoint | Request | Result |
|---|---|---|
| `POST /session/begin` | `{session_id, read_set: ["plans/plan.md", …]}` | `{session_token, cut: {path: version}, coordinator_epoch, retain_versions}` |
| `POST /session/read` | `{session_id, session_token, path}` | the artifact at its **pinned** version — never a newer one |
| `POST /session/commit` | `{session_id, session_token, path, content}` | wins only if no peer moved the artifact since the cut was pinned |
| `POST /session/heartbeat` | `{session_id, session_token}` | keeps the session's lease alive |

Every session call carries both identifiers: `session_id` (your client identity,
from which the server derives the caller) and the server-minted `session_token`
(the handle to the pinned cut).

The `session_token` is server-minted and unguessable; the cut is an inspectable
`{path: version}` map, so you can see exactly which versions your session is
pinned to. In-process, the same surface is
`CoordinatorService.begin_session(read_set=…, owner=…)` →
`session_read(session_token, artifact_id, caller=…)` →
`session_commit(session_token, artifact_id, content, caller=…)`.

Semantics, precisely:

- **Reads serve only from the cut.** An artifact that was *not* in the pinned
  read-set is refused with a typed rejection (`artifact_not_in_cut`) — never
  silently served from live state.
- **Reads are non-mutating.** A session read grants no ownership and blocks no
  writer; peers keep committing while you read.
- **Commits validate against the pinned base.** `session_commit` is a
  single-artifact optimistic commit: it wins only if the artifact's version still
  equals the cut's pinned version, and returns a typed, retryable conflict
  otherwise.
- **Sessions fail closed.** Pins have a bounded lifetime backed by a heartbeat
  lease. A session whose heartbeat lapses — or that is lost to a coordinator
  restart — is invalidated: later reads return a typed `session_invalidated`
  rejection telling the agent to re-establish, never a quiet fall-through to
  whatever is current. A token that was never a session at all (malformed, or
  never opened) is distinguished as `session_not_found`.
- **Bytes vs versions.** When the coordinator retains version bodies
  ([version retention](#version-retention-and-read-at-version)), a session read
  serves the pinned bytes directly. Otherwise it returns the pinned version and
  content hash as a typed signal, and the caller fetches the exact pinned bytes
  from its own data plane.

The no-torn-read property is model-checked: `NoReadSkewWithinCut` and
`PinAlwaysRetained` in `formal/tla/Snapshot.tla`, run by `make tla-check` in CI.

**Honesty boundary.** Snapshot sessions prevent **read-skew** — torn reads across
artifacts. They do not add write-skew prevention: commits validate per-artifact
against the pinned base, so two sessions that read one cut and write *different*
artifacts can still interleave. Single coordinator, single host.

## Effect fence over HTTP

An agent reads a shared artifact, decides something from it, and then fires an
effect that escapes the process — a deploy, a webhook, an opened PR, a charge.
Between the read and the effect the input can move underneath it, or the
authority it was read under can be taken away. The coordinator answers that
question directly, for any client that can make an HTTP request:

**`POST /hooks/effect-fence` — may this effect still fire, and if not, why?**

One artifact per call, against a running coordinator (the same one
`CoherentVolume` spawns). The route is a pure query: it grants nothing,
registers nothing, and heals nothing it checks — so a hold is level-triggered.
Asking again changes no state and gets the same answer until you actually
recover.

**Authentication.** This route carries the same two checks every coordinator
route does, and both run before it: the request must come from a loopback or
allowlisted host, and it must present the coordinator's bearer token —
`Authorization: Bearer <secret>`, where the secret is the contents of
`.coherence/hook.secret` in the workspace the coordinator was started for.
That file is created `0600`, so "any client" means any client running as the
same OS user, not any client on the network. A wrong or missing token answers
`401` and a non-allowlisted host answers `403`; by the rule below, your caller
treats both as a hold, so a misconfigured client fails closed rather than
firing.

### Request

| Field | Type | What it is |
|---|---|---|
| `session_id` | UUID string | Your client identity. Grant standing is **per session**, so the fence cannot answer without it. |
| `path` | string | The workspace-relative artifact whose state gates the effect. |
| `expected_version` | integer | The version you captured when you read the input. Sent as a JSON number, never a string — and never `0`, which is the "could not resolve" sentinel. |
| `expected_generation` | integer **or** `null` | The ownership generation you captured beside the version. |
| `content_hash` | 64-character lowercase hex | The digest of the bytes you actually hold. |
| `agent_id` | string, optional | A subagent id, as on every hook route; it makes a subagent its own coherence peer rather than part of the parent session. |

All five non-optional fields are required on every call. **Name them all**: a
client that omits one gets a `400` telling it which, not a verdict — and a
client that omits `session_id` in particular could otherwise spend a long time
reading holds it can never clear, because the answer depends on which session
holds the grant.

If your client has claimed a [caller principal](#caller-principal) for
`session_id`, send it in the `Coherence-Caller-Principal` header: this route is
require-class, so a request naming a claimed session without it answers `400`.

**Omitted and `null` are different answers for `expected_generation`, and the
difference is the point of the field.** Leaving the key out is a client that
never captured the comparand; no number of identical retries supplies a value
it does not have, so the route answers `400`. The key present as `null` is a
captured *fact* — "the coordinator I read from confirmed no generation" —
which is representable, flows through, and holds under
`generation_unconfirmed`.

**The content hash, exactly.** `content_hash` is a lowercase sha-256 hex digest
over the exact bytes the caller holds, with no normalization — no re-encoding,
no line-ending translation, no trailing-newline or whitespace fixup — so hash
the byte sequence you read, exactly as you read it. The server checks only that
it is 64 hex characters, so a client that hashes a decoded, re-encoded or
tidied-up copy is not told it picked the wrong convention: it simply never
matches what the coordinator recorded, and every call it makes holds wearing
the same reason a genuine conflict would.

### Response

Always HTTP `200` for a protocol outcome, in one of exactly two shapes:

```json
{"verdict": "proceed"}
{"verdict": "hold", "reason": "version_moved"}
```

`proceed` means every leg of the fence affirmatively cleared. `hold` always
carries a `reason` drawn from the vocabulary below.

Two degraded holds add `"degraded": true` and a `held_by` field naming which
arm answered — `watchdog_timeout` (the handler was abandoned before it compared
anything) or `handler_error` (the handler raised). Both report
`version_unconfirmed`, because a handler that ran no comparison resolved no
version. A client needs only `verdict`; `held_by` is for whoever is looking at
the coordinator.

A request that never reached a verdict is `400 {"error": "<what to send>"}`,
naming each missing or malformed field in its own words (`missing
expected_version`, `missing content_hash`, `content_hash must be 64 hex
characters (sha-256)`, and so on). **A malformed request is never a hold.** A
hold invites a retry, and a retry cannot supply a comparand the caller never
captured — so the fixable and the unfixable stay different answers on the wire.

### Anything that is not a verdict is a hold

The fence is a safety property only because the *absence* of a `proceed` stops
the effect. On the caller's side, every one of these is a hold and the effect
does not fire: a connection failure or a timeout; any status other than `200`;
a `200` body with no `verdict`; a `verdict` you do not recognise; and a `404`.

A `404` is what a coordinator that does not implement this route answers, and
one such coordinator ships today: **the sibling Node coordinator backend does
not implement `/hooks/effect-fence`**. There is no handshake to ask first and
none is coming — the `404` is the whole answer. An unimplemented route is a
visible gap; two coordinators quietly disagreeing about whether an effect may
fire would not be.

### Hold reasons — the published vocabulary

These strings are the wire contract, shared by every surface that answers this
question. **Reasons may be added; an existing one is never renamed or
repurposed.** Match on the whole value (`reason == "version_moved"`), never on
a substring of a human-readable message, and treat a `hold` whose `reason` you
do not recognise as a hold — a later coordinator may split a case out of the
residual bucket, exactly as `read_denied` and `content_claim_absent` were split
out of it.

Each reason names who established it — the **coordinator**, from state only it
can see, or the **caller**, from state only *it* can see — and each has its own
recovery. They are not interchangeable: "re-read your bytes" and "call an
operator" are different answers, and telling them apart is why some of these
reasons exist.

| `reason` | Established by | What it means | How you clear it |
|---|---|---|---|
| `version_moved` | the coordinator | a peer committed a newer version than the one the decision was derived from | re-read the input, re-decide, re-gate |
| `read_denied` | the coordinator | it refused the re-validate read outright — this view is invalid (strict mode; on a strict-mode workspace this is how a reclaim usually surfaces) | reacquire, take a fresh read, re-decide, re-gate |
| `grant_reclaimed` | the coordinator | both generations are confirmed and they differ: a sweep reclaimed the grant the decision was read under, while the version never moved | reacquire, re-decide, re-gate |
| `grant_preempted` | the coordinator | neither comparand moved, yet the re-validate read was served without a standing grant — a peer's write-claim acquire took the grant with no commit behind it yet | reacquire. A bare re-ask holds again: the fence does not re-grant on the read it checks with |
| `content_claim_absent` | the coordinator | it records no content claim for this artifact at all, so it cannot vouch that the bytes in hand are the content at the version it reports | re-read the artifact through a coordinated read; the coordinator records a claim on that observation. **Not an operator's problem** — this reason exists so this case stops arriving as `generation_unconfirmed` |
| `input_vanished` | **the coordinator or the caller** — see below | there is no establishable current state for the input | re-establish the input, then re-gate |
| `version_unconfirmed` | the coordinator | it resolved no version at all: a degraded read, or one of the two degraded arms above | restore the coordinator's health, then re-gate. A re-read will not clear it |
| `generation_unconfirmed` | **the coordinator or the caller** — see below | the residual bucket: no confirmed ownership generation | reacquire and re-gate **first** — that clears the recoverable forms. A hold that survives a *successful* reacquire is the last one: check the coordinator's version and restart it |

**`input_vanished` has two establishers, and both are in the contract.** They
mean the same thing — there is no establishable current state for the input —
and they are established from opposite sides:

- **The coordinator** answers `input_vanished` over the wire when it holds no
  record of the artifact: the path is not tracked, it is tracked but was never
  observed, or its row vanished between the lookup and the read. Recovery is to
  get the path tracked and read once through the coordinator, so a record
  exists to compare against.
- **The caller** establishes it itself when its own input no longer exists. The
  coordinator cannot see your workspace, so it will never report this for you;
  detect it and treat it as a hold under this same identifier. Recovery is to
  recreate or re-point the input.

**`generation_unconfirmed` likewise.** The caller establishes it by sending
`expected_generation: null` — its own read confirmed no generation. The
coordinator establishes it when it cannot confirm one now: a degraded read, an
out-of-band edit it could not confirm, or a coordinator too old to report
generations at all. The bucket is deliberately *not* a clean
permanent-versus-transient signal, which is why the recovery is retry-first and
persistence across a *successful* reacquire is the signal that a person is
needed.

**Which reason you get when several apply.** The legs are evaluated in a fixed
order and the first match is the answer: `input_vanished`,
`version_unconfirmed`, `version_moved`, `read_denied`, `content_claim_absent`,
`generation_unconfirmed`, `grant_reclaimed`, `grant_preempted`. The order is
part of the contract — a state that matches two legs has one right answer, and
an implementation that reorders them answers a different question.

### The same fence, three surfaces

| Surface | Call | Reach |
|---|---|---|
| Python | `gate(volume, path, decide=…, effect=…)` | in-process; holds raise `StaleView` carrying `hold_cause` |
| MCP | `swg_gate` | any MCP client; holds come back as a typed deny |
| HTTP | `POST /hooks/effect-fence` | any client at all — no Python, no MCP |

All three ask one classification, so a reason means the same thing on each. One
leg is not the same on each, and it matters: **only the HTTP route can answer
`content_claim_absent`.** The Python and MCP surfaces read a *comparison* from
the coordinator rather than its recorded hash, so "the claim matches" and
"there is no claim at all" arrive there as one value, and they assume a claim
exists. If your effect matters enough to fence, that is the leg to ask for over
HTTP.

The coordinator keeps advisory counts of how many fence calls it answered and
how many of those held, as instrumentation for whoever runs it. They are
advisory rather than auditable, and nothing in a client's decision path should
read them.

### Scope, honestly

The fence *orders* an effect; it never rolls one back. It answers before the
effect fires and does not undo it afterwards, so for an escaping effect there
is a residual check→fire window it narrows but cannot close. Single host,
single coordinator, and cooperative — an effect that never asks is never held.

**A narrower gate this vocabulary does not govern.** The coordinator also
offers a session-based gate in Python — `CoordinatorService.effect_gate`, which
pins a read-set and re-validates every member's *version* at the effect
boundary. It checks versions only: it does not apply the grant-standing leg, so
a peer's write-acquire that preempts the session's grant without moving any
version lets that gate fire where this fence holds. It is in-process Python
only — no HTTP route and no tool reaches it — and it answers in its own
fired/held result types, not in the vocabulary above.

## Caller principal

The coordinator authenticates the workspace, not the caller: one bearer secret
covers every process, and the session a request acts as is the `session_id` in
its body. A **caller principal** is a value the coordinator issues and binds to
one session, so that a request naming that session can be checked against it.
For a long-lived client it turns "a writer sent the wrong session id" — a copied
request, a stale id after a fork, a retry that picked up a peer's id — from a
silent success into a refusal.

It is not a security boundary. The bearer secret still gives full authority over
the workspace, and every principal is stored in `.coherence/state.db`, where any
process running as your OS user can read it. What a principal catches depends on
the client:

- A long-lived client — `CoherentVolume`, the MCP server, the substrate session —
  is issued one principal for its own session and presents only that one, so a
  request it sends naming any other session is refused.
- The Claude Code hook client runs once per hook event and finds its principal
  in `.coherence/` by the session id in the event. A hook carrying another
  session's id can therefore find that session's principal too. On this surface
  the principal makes a client that never claimed, or one presenting the wrong
  principal, visible; it does not tell sessions apart.

### You usually do nothing

The Claude Code hook client, `CoherentVolume`, the MCP server and the substrate
session claim a principal for their session and send it on every request. If
the binding disappears because `state.db` was deleted, the next request the
coordinator refuses makes each of them claim again with the nonce it kept, and
the request is sent once more with the principal that comes back. A claim whose
*answer* is lost is handled the way each client handles any unanswered
coordinator request: the hook client, and a `CoherentVolume` built with
`on_error="degrade"`, carry on and claim again with the same nonce before their
next request; a strict `CoherentVolume` (the MCP server's is strict) and the
substrate session fail closed at construction, so the object is never built,
and a later attempt is a new session with a new nonce — the first binding is
left unused. (A strict volume's forked child attaches on its first request
instead: that request raises, and its next one claims again with the same
nonce.) A client written before principals existed keeps working
unchanged: its sessions never claim one, and a request naming a session that
never claimed one is admitted exactly as before (and counted, see below).

The Claude Code plugin's Node coordinator issues no principals. It answers
`/principal/claim` with `404`, and clients proceed without one; the hook
clients skip the request altogether when `.coherence/server.pid` names the Node
backend.

The hook client keeps a session's nonce and principal in two files under
`.coherence/`; they are listed, with what to do about them, among the
coordinator's other local files in
[security.md](security.md#local-config-and-data-files).

### What is checked, and where

Every route that takes a `session_id` is in one of three classes:

| Class | Routes | A request naming a session that has claimed a principal… |
|---|---|---|
| require | `/hooks/pre-edit`, `/hooks/session-stop`, `/hooks/post-edit`, `/hooks/post-edit-cas`, `/hooks/effect-fence`, `/workspace/checkpoint`, `/workspace/restore/register`, `/workspace/restore/status`, `/workspace/restore/member` | …must present it. Without it, or with a different one, the answer is `400`. |
| accept | `/hooks/pre-read`, `/hooks/pre-bash`, `/hooks/pre-grep`, `/hooks/session-start`, and the five `/session/*` routes | …is admitted without it, but refused with a different one. |
| mint | `/principal/claim` | …is where principals come from; its gate is the mint nonce. |

The require class is the routes that take, release or commit a session's
grants, record who wrote a version, answer the effect fence about a session's
grant, or record workspace ownership — the things a caller naming the wrong
session could do to someone else's work. `pre-edit` is among them because a
grant taken without the principal could then be neither committed nor released.
The accept-class reads change no grant or version, but they do deliver the
named session's pending notices, so a read naming the wrong session can consume
advisories meant for it. The snapshot-session routes are accept-class because
each is already gated by the server-issued session token.

A refusal is HTTP `400` with an `error` field naming the header and a `reason`
field: `caller_principal_absent` when a claimed session is named with no
principal, `caller_principal_foreign` when the principal presented is not the
one bound to the named session. Branch on `reason`, not on the text. It is never
a hold, and a refused request changes nothing, so it is safe to send again once
the client has the right principal.

`GET /status` reports two counters at every tier: `caller_principal_absent_total`,
the requests this coordinator process admitted without a principal (a nonzero
value means some client is not sending one yet), and
`caller_principal_refused_total`, the requests it refused. Both are local
diagnostics that reset when the coordinator restarts.

### Writing your own HTTP client

1. Generate a mint nonce — 16 to 128 characters from `[A-Za-z0-9_-]`, for
   example `secrets.token_urlsafe(32)` — and keep it **before** you call. It is
   what lets you recover your principal if the response is lost.
2. `POST /principal/claim` with `{"session_id": ..., "mint_nonce": ...}`. The
   answer is `{"ok": true, "principal": "..."}`. A retry with the same nonce gets
   the same principal back. If the session is already bound under a different
   nonce, the answer is `{"ok": false, "reason": "caller_principal_claimed", "detail": ...}`
   and you do not become that session; a malformed request is `400`.
3. Send the principal in the `Coherence-Caller-Principal` header on every
   request that names the session. The principal appears in the claim response
   and nowhere else: never log it.
4. If a request is refused with `caller_principal_absent` or
   `caller_principal_foreign` — for example because `state.db` was deleted and
   the binding with it — claim again with the **same** nonce, and send the
   request once more with the principal you get back. Never pick a new nonce:
   that is what stops a client from taking over a session someone else has
   claimed.

A subagent shares its parent session's principal.

### Upgrading

Principals live in their own table, added by registry schema version 8. Like
earlier schema steps it is forward-only: once a workspace's `state.db` has been
opened by this version, an older release refuses it. Upgrade forward rather than
rolling back.

## Acquire-or-fail on `pre-edit` (specified, not yet built)

**Nothing in this section is implemented.** It fixes the shape of an opt-in
refusal so the change that builds it does not have to re-decide it.

Today `POST /hooks/pre-edit` grants EXCLUSIVE to whoever asks. A session already
holding the grant is set to INVALID — nothing is committed — and finds out on its
next request. There is no request that declines instead, so a real mutex cannot be
built on this route: the second session to ask always wins.

**Request.** `pre-edit` with `"if_unheld": true` in the body, carrying the
[caller principal](#caller-principal). An opted-in request without a principal is refused as a
malformed request (HTTP 400 naming the principal), because a refusal that a
caller could step around by naming the holder's session is not a refusal.

**Refusal.** HTTP 200 with
`{"ok": false, "reason": "other_holder", "holder_agent_id": "<agent id>"}` — the
same reason string `post-edit-cas` already returns when a commit meets a
session holding the write grant. The holder is named by agent id, never by
session id. Nothing changes hands: the holder keeps its grant and the caller
gains none.

**What the refused caller does.** Back off and retry the acquire. The refusal
carries no retry hint because none exists. Committing does not end a grant: a
holder that commits keeps the file MODIFIED, and the optimistic lane is no way
around it — `post-edit-cas` answers `other_holder` against a MODIFIED holder
too. A grant ends when its holder releases it — `session-stop`, which a Claude
Code session sends at the end of its turn, or a failed `post-edit` — or when the
coordinator reclaims it from a silent holder (no heartbeat for
`grant_heartbeat_timeout_sec`, or held longer than `grant_max_hold_sec`). Poll
with backoff rather than wait for a signal.

**Under load.** A timed-out opted-in request must not answer in a shape that
admits the edit: the contention that makes a holder worth respecting is the
contention that times a handler out. It answers `ok: false` with
`"degraded": true`, and the caller treats that as a refusal. Today a timed-out
`pre-edit` answers `{"ok": true, "degraded": true}`; a test in the coordinator
suite fails as soon as a refusal reason is registered while that is still the
answer.

**What it does not close.**

- A refused agent can still write the file with a shell command. The Bash hook
  looks for commands that read tracked files, not ones that write them, so a
  write that goes around the Edit and Write tools is not refused;
  [foreign-write detection](#foreign-write-detection--who-wrote-this-behind-my-back)
  reports it afterwards.
- A new kind of deny on the edit path is shown to the model, and that changes
  how it retries. The change that builds the refusal needs its own measurement
  of that before it ships.
- On the hook surface, any process that can read `.coherence/` can read another
  session's principal. The refusal separates writers that follow the protocol;
  it does not stop one that deliberately uses another's principal.

## `stale-write-guard-fs` MCP server

The coherent-workspace guarantee for agents that speak
[Model Context Protocol](https://modelcontextprotocol.io) — Claude Code, Cursor,
or a custom runtime — with no Python integration. The server wraps
`CoherentVolume` and exposes coordinated file access over stdio:

```bash
pip install "agent-coherence[mcp]"
```

Register it with your MCP client (the exact file depends on the client):

```json
{
  "mcpServers": {
    "stale-write-guard-fs": {
      "command": "stale-write-guard-fs",
      "env": { "SWG_ROOT": "/path/to/shared/workspace" }
    }
  }
}
```

`SWG_ROOT` selects the workspace (defaulting to the server's working directory).
By default the whole workspace is guarded; narrow it with `SWG_MANAGED`, a
comma-separated glob list (for example `SWG_MANAGED=plans/**,memory/**`).

| Tool | What it does |
|---|---|
| `swg_read` | Tracked read — registers the agent's view of the file. If the bytes on disk are not what the coordinator recorded, the read returns a `stale_view` deny with no version. Retry `swg_reacquire` + `swg_read` for a few seconds first, since a peer's commit may still be reaching disk; if it stays denied, the file was changed outside the coordinator (an out-of-band edit, or a commit whose disk write failed), so `swg_write` the reacquired content to record it |
| `swg_write` | Guarded write — a stale view or foreign edit returns a typed `stale_view` deny with `recover: reacquire`, never a silent overwrite |
| `swg_reacquire` | Recovery after a deny — clears the stale view + mandatory fresh read |
| `swg_write_cas` | Single-shot version-checked write for concurrent same-key contention |
| `swg_gate` | Effect fence — re-checks the `(version, owner_generation)` pair from your `swg_read` right before an irreversible external action (a webhook, a deploy, an opened PR), and denies if the value moved OR the grant it was read under was reclaimed OR a peer's write-claim preempted it (which moves neither comparand — the fence also re-checks that the grant still stands) |
| `swg_status` | Three-state coordination health: `on` / `off` / `unknown`, plus this session's `principal_claim` and the coordinator's two caller-principal counters |
| `POST /hooks/effect-fence` | **Not a tool — the HTTP sibling of `swg_gate`.** The coordinator answers the same fence verdict to any client that can make an HTTP request, with no MCP and no Python in the loop, and it is the only surface that can answer the no-content-claim leg. See [Effect fence over HTTP](#effect-fence-over-http) |

Denials are machine-readable: an agent parses the typed payload (for example
`reason: stale_view`, `recover: reacquire`) and self-heals instead of retrying
blindly. The server validates every file URI — path traversal and any access to
the coordinator's own state directory are rejected — and fails closed on IO
errors. Strict-mode, managed-path scoped.

**A session the coordinator refuses.** The server claims a
[caller principal](#caller-principal) for its session at start. When a later
request is refused for it and claiming again with the same nonce cannot cure
that — the session is bound under another nonce, or the claim hands back the
principal that was just refused — the tool call answers `reason:
caller_principal_absent`, `caller_principal_foreign` or
`caller_principal_claimed` with `recover: restart_session` and
`retryable: false`, and so does every `swg_write`, `swg_read` and `swg_gate`
after it: no tool call in that server session can regain coordination, and a
new server session claims its own principal. `swg_status` says so ahead of time
as `principal_claim: refused`. When instead the *answer* to that claim is lost
(a transport blip), the refusal is not yet settled: the tool call answers the
same `reason` with `recover: wait_and_retry` and `retryable: true`, the next
tool call claims again with the same nonce by itself before it runs — and
usually just succeeds — and `swg_status` reads `principal_claim: unconfirmed`
until it does. The other values are `bound`, `unsupported` (the coordinator
issues no principals) and `not_attempted`. `swg_status` also forwards the
coordinator's `caller_principal_absent_total` and
`caller_principal_refused_total`, `null` rather than `0` when the coordinator
is unreachable or does not report them.

**Multiple sessions, one workspace.** Multiple `stale-write-guard-fs` instances
pointed at the same `SWG_ROOT` attach to one coordinator, so a stale write is denied
across sessions; if the coordinator's session exits, peers fail closed.

**Wiring a client to prefer `swg_*` over native file tools.** Registering the
server exposes the `swg_*` tools, but an agent will still reach for its native
Write/Edit on a managed path unless you steer it there. Deny the native file tools
on managed paths — through the client's permission rules or a pre-tool hook — so
writes have to go through `swg_write` / `swg_write_cas` and inherit the stale-view
deny. The Claude Code adapter that wires this seam ships in
`ccs.adapters.claude_code`.

**Subagent identity (v0.13.0).** When a Claude Code hook payload carries an
`agent_id` alongside the parent `session_id`, the adapter folds both into the
identity derivation, so each subagent becomes its own coherence peer rather
than blending into the parent session. That buys two things: `last_writer`
attribution names the subagent that actually wrote (visible in the operator
`/status` view, `?detail=full`, as `claude-session-<sid>:subagent-<aid>`; the
default view omits the name because it embeds the raw session id), and two
sibling subagents racing the
same artifact are detected as a collision instead of passing as one writer.
With no `agent_id` in the payload the derivation is unchanged, byte-for-byte —
main-thread sessions behave exactly as before. On Claude Code's
`SubagentStop` event, the hook client's `subagent-stop` subcommand releases
just that subagent's grants; the `agent_id` is required there, so a payload
without one skips the release rather than stripping the parent's grants
mid-session. The Python and Node coordinator backends derive the identity
byte-identically; the protocol corpus pins the parity (sibling collision,
attribution, scoped release fixtures).

**Compaction-aware re-grounding (v0.14.0).** When Claude Code compacts a
session (auto-compaction or a manual `/compact`), the model's summary can
silently drop what the session held and what peers changed around the
boundary. Wire Claude Code's `SessionStart` hook to
`agent-coherence-hook-client session-start` and the coordinator re-grounds
the compacted session with a bounded payload: the grants it held at
compaction, event-anchored ("At compaction you held EXCLUSIVE on `plan.md`
(v7) — re-acquire before writing."), and each touched artifact's current
coordinated version — with a stale flag when a peer advanced it past the
session's last-observed version ("`plan.md` advanced to v9 past your
last-observed v7 — re-read before relying on it."). The payload is
session-scoped: the parent's lines render first, then each registered
subagent's under a `Subagent <name>:` prefix. The subcommand gates on
`source: "compact"` client-side, so ordinary starts, resumes, and clears
never reach the coordinator. Delivery: the payload renders at the next
user message and on `--resume`; a live autonomous loop additionally
receives it on its next tool admit — attached only to allow responses,
never to a strict-mode deny (deny bodies stay byte-identical). Bounded and
honest: at most three artifact lines plus an overflow summary pointing at
`agent-coherence-status`; a session with no coordination state emits
nothing; a coordinator that is down at the compact boundary fails open
with an empty response — coordination never blocks the session. One benign
duplicate is possible (a mid-loop delivery followed by the next user
turn's render); the closing line — "Versions are as of this re-grounding;
a more recent read supersedes this notice." — makes a second sighting
harmless. The Python and Node coordinator backends emit byte-identical
prose (protocol-corpus pinned). Staleness flags need an observation
baseline: rows recorded before this release have no last-observed version
and are deliberately never flagged, so detection becomes accurate as
sessions read and commit after the upgrade.

Run the red→green demo: `python -m examples.mcp_stale_write_guard.main`
(offline, deterministic, no keys).

## Inline benchmark mode

Measure token savings on your own workload without any external tooling:

```python
store = CCSStore(strategy="lazy", benchmark=True)

# ... run your graph ...

store.print_benchmark_summary()
```

`benchmark=False` (default) adds zero overhead — no counters are allocated.

For programmatic access, use `benchmark_summary()`:

```python
summary = store.benchmark_summary()
# {
#   "baseline_tokens": 4160,
#   "ccs_tokens": 1301,
#   "tokens_saved": 2859,
#   "token_reduction_pct": 68.7,
#   "cache_hit_rate": 0.75,
#   "n_operations": 16,
# }
```

`benchmark_summary()` raises `RuntimeError` if the store was not created with
`benchmark=True`.

---

## Telemetry

Structured metrics without changing node code.

### OpenTelemetry

```bash
pip install "agent-coherence[otel]"
```

```python
store = CCSStore(strategy="lazy", telemetry="opentelemetry")
```

CCSStore creates two Counter instruments on the globally-configured
`MeterProvider`:

| Instrument | Unit | Attributes |
|------------|------|------------|
| `ccs.store.operations` | `{operation}` | `ccs.operation`, `ccs.agent_name`, `ccs.cache_hit` |
| `ccs.store.tokens_consumed` | `{token}` | `ccs.operation`, `ccs.agent_name`, `ccs.cache_hit` |

If no SDK is configured, the OTel no-op provider discards everything at zero cost.

To use a specific provider instead of the global one:

```python
from ccs.adapters.telemetry.otel import OtelExporter
store = CCSStore(strategy="lazy", telemetry=OtelExporter(meter_provider=my_provider))
```

### LangSmith

```bash
pip install "agent-coherence[langsmith]"
```

```python
store = CCSStore(strategy="lazy", telemetry="langsmith")
```

Per-operation metadata is attached to the active LangSmith run tree via
`run.add_metadata(...)`. Keys attached to each event:

```
ccs.operation, ccs.agent_name, ccs.tokens_consumed, ccs.cache_hit, ccs.tick
```

If no LangSmith run is active, events are silently discarded.

### Custom exporter

```python
from ccs.adapters import TelemetryExporter, StoreMetricEvent

class DatadogExporter(TelemetryExporter):
    def on_event(self, event: StoreMetricEvent) -> None:
        statsd.increment("ccs.operations", tags=[f"agent:{event.agent_name}"])
        statsd.histogram("ccs.tokens", event.tokens_consumed)

store = CCSStore(strategy="lazy", telemetry=DatadogExporter())
```

`on_metric` and `telemetry` are independent — both fire for every event if both
are set.

---

## Graceful degradation

By default (`on_error="strict"`), a `CoherenceError` propagates and the graph
fails. Use `on_error="degrade"` to keep the graph running when the coherence
layer encounters an unexpected state:

```python
store = CCSStore(strategy="lazy", on_error="degrade")
```

In degrade mode:

- **`put`**: if `core.write` raises, the value is stored in a plain dict fallback
  and a `"degraded"` operation event is emitted.
- **`get`**: if `core.read` raises, the value is retrieved from the fallback dict
  (empty dict if nothing was previously stored there) and a `"degraded"` event
  is emitted.

A warning is logged at `WARNING` level for each degraded operation. Monitor
degradations via `on_metric`:

```python
events = []
store = CCSStore(strategy="lazy", on_error="degrade", on_metric=events.append)

# ... run graph ...

degraded = [e for e in events if e.operation == "degraded"]
if degraded:
    alert(f"{len(degraded)} degraded operations detected")
```

Use `on_error="strict"` (the default) in development and CI. Consider
`on_error="degrade"` in production environments where a coherence bug should not
take down the whole graph.

Two attributes let you check degradation state after the fact:

```python
store.is_degraded       # True after the first degraded operation
store.degradation_count  # total number of degraded operations
```

Use these to gate alerts or health checks without keeping a separate event list.

---

## Examples

All examples are runnable with `python -m examples.<name>.main` (or `.demo` where noted) from the project root.

Correctness demos lead; the token-savings / hit-rate demos follow.

| Example | Command | What it shows |
|---------|---------|---------------|
| Shared knowledge base | `python -m examples.shared_knowledge_base.demo` | Lost update in a shared RAG / memory corpus; `CoherentVolume` denies B's stale overwrite so both findings survive (offline, no keys) |
| Divergent memory | `python -m examples.divergent_memory.demo` | Two sessions record contradictory beliefs from a stale read; the stale write is denied fail-closed so the divergence never forms (offline, no keys) |
| CCSStore read side | `python -m examples.ccsstore_read_side.demo` | Read-side invalidation on a LangGraph `BaseStore` (peer commit → next `get()` serves the new version), the `put()`-is-not-version-CAS boundary, and the `write_cas` fix (offline, no keys) |
| Coherent volume | `python -m examples.coherent_volume.main` | Sequential stale-write deny + recovery on plain files (offline, no keys) |
| Concurrent writers | `python -m examples.concurrent_writers.main` | True-race lost update; `write_cas` preserves both updates (offline, no keys) |
| Effect gate | `python -m examples.effect_gate.main` | `gate()` holds an effect on a stale input; `--baseline` shows the stale fire (offline, no keys) |
| MCP stale-write guard | `python -m examples.mcp_stale_write_guard.main` | Red→green stale-write deny through the MCP server tools (offline, no keys) |
| Workspace versioning & restore | `python -m examples.workspace_versioning.main` | Checkpoint a mixed file + S3 workspace, then restore it with per-member honesty (`restored` / `conflict` / delete leg / forward-only skip); `--baseline` shows the unrecoverable loss first (offline, no keys) |
| Session handoff | `python -m examples.session_handoff.main` | Two sessions, each a real OS process, share a scratch file and hand off through a checkpoint: plain file I/O loses the second session's line with nothing raised; `CoherentVolume` denies the stale write and both lines survive; a rewind lands cleanly when the handing-off session stopped and concludes `conflict` when it is still writing (offline, no keys) |
| Conversations stale-read | `python -m examples.conversations_stale_read.main` | Two agents share one conversation; client-cache invalidation (offline, no keys) |
| Cross-host (experimental) | `python examples/cross_host/main.py` | Stale-write deny + effect ordering across a host boundary (local smoke; Docker runner in `examples/cross_host/`) |
| LangGraph planner | `python -m examples.langgraph_planner.main` | 4-agent, 1 artifact, 75% hit rate |
| Code review pipeline | `python -m examples.code_review.main` | 3-agent, SHARED state transitions |
| Research pipeline | `python -m examples.research_pipeline.main` | 4-agent, 3 artifacts, 60% hit rate |
| Shared codebase | `python -m examples.shared_codebase.main` | 4-agent code review, 37.6% savings, benchmark output |

### Code review pipeline

Three agents share a codebase artifact. The key behavior: `reviewer_b` reads the
same codebase that `reviewer_a` cached without either agent invalidating it, because
neither wrote to it. Both hold it in SHARED state simultaneously.

### Research pipeline

Four agents operate on three artifacts (`brief`, `findings`, `analysis`). The key
behavior: `researcher`'s write to `findings` does **not** invalidate `brief` held
by `analyst` — each artifact key has its own independent MESI state per agent.

### Conversations stale-read

Two agents share one conversation: one caches it locally, the other revises it, and
the first acts on a stale copy. `CoherenceAdapterCore` invalidates the stale cache so
the reader re-fetches before acting. Runs offline with no API keys. The companion
`probe.py` measured the OpenAI and Mistral Conversations *servers* as read-after-write
consistent (zero stale reads over 100 + 20 live trials), so the demo isolates the real
failure — the **client cache**, not the server. See
[`examples/conversations_stale_read/README.md`](../examples/conversations_stale_read/README.md)
for the full framing and the optional live consistency probe. This is the same mechanism the
[OpenAI Agents SDK adapter](#openai-agents-sdk-adapter-experimental) applies to a live
`Session`.

---

## Real-workload benchmarks

Results from real LangGraph graph executions using `GenericFakeChatModel` (no live
LLM calls). Run them yourself:

```bash
pip install "agent-coherence[langgraph,benchmark]"
make benchmark    # all three workloads, prints consolidated table
```

Or run individually:

```bash
python benchmarks/langgraph_real/bench_planner.py
python benchmarks/langgraph_real/bench_code_review.py
python benchmarks/langgraph_real/bench_high_churn.py
```

| Workload | Agents | Hit rate | Baseline | CCSStore | Savings |
|----------|--------|----------|----------|----------|---------|
| Planning (read-heavy) | 4 | 75% | 4,160 | 1,301 | 69% |
| Code review (write-moderate) | 3 | 60% | 5,320 | 2,835 | 47% |
| High-churn (write-heavy) | 4 | 50% | 3,250 | 2,317 | 29% |

*Tokens are approximate; real LLM content will vary.*

Hit rate and savings are lower-bounded by write frequency: more writes mean more
invalidations, more misses. The planning workload has 1 write and 12 reads (75% hit
rate). The high-churn workload has 4 writes and 8 reads (50% hit rate).

For the simulation-based results from the paper (84–95% savings), see
[reproduce.md](reproduce.md).

### Temporal cost: source drift between turns (TC-1)

The table above is the **spatial** dimension — savings grow with more agents sharing one artifact. The **temporal** dimension is orthogonal: a *single* agent (a RAG/memory reader) whose source drifts between its turns, where coherence-gating avoids re-fetching a chunk that didn't change. TC-1 is the pre-registered benchmark for it — a savings-regime map across change-rate × answer-sensitivity:

```bash
python tools/run_cost_sweep.py --rates 0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.5,0.75,1.0 \
  --sensitivities 0,0.5,1.0 --runs 50 --output benchmarks/results/cost_sweep_published.json
python tools/plot_cost_sweep.py    # savings-vs-change-rate curve
```

PASS at n=50: savings stay ≥ 30% while the source changes fewer than ~3 turns in 10 (`r ≤ 0.30`), crossing below 30% at `r ≈ 0.31` and falling to 0 at constant churn. The metric is **re-fetches-avoided** — a proxy / regime map, not a token-dollar invoice (`tools/cost_to_tokens.py` gives an assumption-parameterized dollar translation). Verdict + distinguisher triage: [`../benchmarks/cost_preregistration.md`](../benchmarks/cost_preregistration.md). **Don't splice these numbers into the spatial table above.** Shipped in `v0.9.3` (#116).

---

## Benchmarking your own workload

```bash
pip install "agent-coherence[langgraph,benchmark]"
ccs-benchmark --graph path/to/my_graph.py:build_graph
```

The factory function must accept a single `store` argument and return a compiled
LangGraph graph:

```python
def build_graph(store):
    builder = StateGraph(...)
    # ... add nodes/edges ...
    return builder.compile(store=store)
```

Pass a custom input state with `--initial-state`:

```bash
ccs-benchmark --graph my_graph.py:build_graph --initial-state '{"query": "hello"}'
```

The CLI runs the graph once and prints `print_benchmark_summary()` output. For
inline benchmarking without the CLI, see [Inline benchmark mode](#inline-benchmark-mode).

---

## `ccs-diagnose` — detect stale reads

A standalone CLI for detecting divergent reads in an existing LangGraph graph without changing any code. Passive callback, zero outbound network in v0, HTML + JSON reports. Install with `pip install "agent-coherence[diagnose]"`.

See [docs/ccs-diagnose.md](ccs-diagnose.md) for the full reference: usage, flags, exit codes, trust posture, calibration corpus, and the `langgraph-v0-preview` → `v1` promotion gate.

---

## Conflict-outcome counters — how often did it actually fire?

*New in v0.14.1.*

Every guarantee in this library ends in a typed refusal: `version_mismatch`,
`other_holder`, `stale_read_generation`. The counters answer the question that
follows — **how often does that actually happen in my fleet?** — with a number
instead of an argument.

The coordinator keeps a durable tally per `(artifact, agent, reason)` that
survives restarts. Read it back offline, against a coordinator that is no
longer running:

```python
from ccs.diagnose.conflict_counters import read_conflict_totals

totals = read_conflict_totals(".coherence/state.db")
# {(artifact_id_hex, agent_id_hex, "stale_read_generation"): 3, ...}

for (artifact, agent, reason), count in sorted(totals.items()):
    print(f"{reason:24} {count:>5}   artifact={artifact[:8]} agent={agent[:8]}")
```

The reader opens the database **read-only and raw** — it imports no coordinator
and needs no daemon — so you can point it at a `state.db` copied off a machine
after the fact.

**The honesty rules matter more than the numbers, so they are worth stating
plainly:**

- A database written **before** this release has no `conflict_counters` table.
  That reads as **zero recorded conflicts** — a real, reportable result.
- Every **other** read failure — a locked database, a hot-WAL recovery failure,
  disk I/O — is raised, never mapped to zero. A broken read can never
  masquerade as a quiet month.
- A missing file raises `FileNotFoundError`: a report against a store that does
  not exist is a caller error, not evidence of zero conflicts.
- **Attribution is by agent identity only.** No host identifier reaches the
  commit path today, so a report built from these totals says "agent", not
  "host" — the reader will not invent a dimension the data does not carry.

Zero is a result. *No table at all* is a different result, and the distinction
is exactly what makes a thirty-day observation window worth running.

---

## Foreign-write detection — who wrote this behind my back?

The coordinator sees writes that go through it. It does not see an editor, a
script, or a second tool that writes a shared file directly — and that is the
window where a stale value quietly spreads into everything derived from it. The
existing guards catch such an edit at the next read and at the next write. They
say nothing about the time in between.

Detection watches that window. While a coordinator is running, once per sweep it
asks git which of the files it coordinates have changed on disk, re-hashes just
those, and records what it found. It never denies anything. A detection is a
number in a report, not a refusal.

Read it back offline, against a coordinator that is no longer running:

```python
from ccs.diagnose.foreign_writes import read_foreign_write_report

report = read_foreign_write_report(".coherence/state.db")

print(report.state)        # 'not-instrumented' | 'not-coverable' | 'instrumented-zero' | 'counts'
for artifact, counts in sorted(report.totals.items()):
    print(f"{artifact[:8]}  {counts}")   # {'foreign': 2, 'mediated': 5}

for run in report.runs:                  # what each coordinator run watched
    # covered_count is the files git could report on, not every file you track
    print(run.tick_count, "checks over", run.covered_count, "files")

report.covers(started_at, ended_at)      # was that period watched end to end?
report.uncoverable                       # runs that found nothing to watch
```

Each file lands in one of three buckets:

- **foreign** — the bytes on disk are not the bytes the coordinator has, and
  nothing it knows explains that.
- **mediated** — the file changed, and the coordinator holds exactly those
  bytes. It was a write that went through it.
- **lag_suppressed** — the mismatch looked like a write that was still landing
  when the check ran. Counted separately, never as a foreign write and never as
  a clean result, so you can see how often the benefit of the doubt was given.

**The honesty rules matter more than the numbers, so they are worth stating
plainly:**

- **Zero is only zero when the detector actually ran.** A store it never ran
  against reports `not-instrumented`, which is a different answer from
  `instrumented-zero`. Turning the sweep off, reading a store from a coordinator
  that never started one, or running where nothing is in scope all give you the
  first — never a clean bill of health you did not earn. Each run also records
  how many files were in scope, because watching five hundred and finding
  nothing is a different result from watching none. That number counts only the
  files git could actually report on: one your patterns name and the coordinator
  knows, but that nobody has added to git, is not among them, because a check
  that can never see a file must not be counted as having watched it. Read it as
  the widest scope that run ever watched, not the scope of its last check — a
  run whose visible files fall away keeps reporting the larger number, and the
  run is not split when that happens. The count is therefore a subset of the
  files you asked to have tracked, and a check that could see only some of them
  leaves no separate mark anywhere in the report, so the report alone will not
  tell you how far the two have drifted apart.
- **A workspace the detector cannot watch is its own answer.** Detection can
  only watch a git work tree, and a coordinator rooted outside one — a temp
  directory, an unpacked archive, a directory nobody ran `git init` in — can
  never be polled. There is a second way to arrive at the same nothing: a real
  repository in which none of the files the coordinator knows and tracks are
  files git tracks, so git answers and has nothing to say about any of them.
  Either way, once the detector has at least one artifact it already knows and
  tracks, it writes a note that it looked and found nothing it could watch, and
  records no check for that sweep. `report.uncoverable` lists each such run,
  why it could watch nothing, and when it looked. That note leads the report —
  the state you read is `not-coverable` — only in a store that never recorded a
  check; where an earlier run did record checks, those runs lead instead and
  the note is read from `report.uncoverable`. Read it there whatever the state
  says. `not-coverable` is neither a zero nor an outage — the detector ran and
  correctly found nothing it could watch — and it is kept apart from
  `not-instrumented` so a healthy instrument is not accused of never having
  run. `covers()` is untouched by either condition: it walks recorded checks
  alone, so a span in which nothing was watched still answers `False`. With
  nothing in scope at all the detector never reaches git and records nothing,
  so a store that has never been polled reads `not-instrumented`, exactly as
  the bullet above says — that precedence is deliberate, and the state reflects
  what the store retains rather than the latest sweep. A workspace that gains
  a repository is watched from the next sweep, with no restart, and so is one
  whose files reach git's index. Files you deliberately keep out of git are
  outside the instrument by design and need no fixing; where you do want them
  watched, the tracked set has to name files git tracks.
- **A suppression expires.** A mismatch excused as a write still landing is
  re-examined once the window passes. If no write ever landed, it becomes a
  foreign write and is counted as one. The benefit of the doubt is temporary.
- **A count is one per version of the content, not one per check.** An edit
  nobody has reconciled is still there on the next check, and the one after
  that. It is counted once. Change the file again and that is a second count.
- **Detection covers files the coordinator already knows and git tracks.** A
  file that matches your patterns but has never been touched through the
  coordinator, one you have since stopped tracking, and one git ignores are all
  outside it. They are not reported as clean; they are not reported at all.
- **A broken check is never a quiet month.** If git cannot run, the check is not
  recorded as having happened, so the gap is visible in the report rather than
  reading as no news. A workspace with no repository at all is the one such
  condition nothing can fix, so it is logged once at debug level instead of once
  per sweep; the report is unaffected and still shows the gap. A repository that
  exists but is broken is not that case — git describes the two identically, so
  this is decided by looking for the repository, not by reading the message —
  and it stays loud.
- **`covers(start, end)` answers coverage, not the totals.** A coordinator that
  was down for the middle of a period still shows a healthy count of checks. Ask
  `covers` whether the period you care about was actually watched end to end.
  A stretch in which nothing at all was in scope — because the tracked set was
  emptied and later restored, say — ends the watched period too, so `covers`
  reports it as a gap rather than reading across it. So does a stretch in which
  git could report on none of the files in scope, because they were never added
  or an exclude rule hides them: the tracked set was not empty, but nothing in
  it was being watched. Narrowing the set to a smaller one git can still speak
  about does not: those files were still being watched. And a stretch in which
  the check did not run at all, because the machine was asleep or the
  coordinator was stalled, ends it as well.
- **Attribution is by file only.** A change on disk carries no author, so the
  report names what changed and never who changed it.

Reading the report never writes to the store, so you can point it at a database
copied off a machine after the fact.

## Replay (v0.8.2+)

`agent-coherence-replay` is an invariant-replay tool that walks a captured
coordinator session and reports breaches of the four core MESI invariants —
single-writer, monotonic-version, stale-read, lost-write — without re-executing
any agents. Capture rides on the existing `state_log` and `content_audit_log`
callback seams via `CCSStore.record_to(path)`.

### Capture and replay (LangGraph quickstart)

```python
from langgraph.config import get_store as lg_get_store
from langgraph.graph import END, START, StateGraph
from typing import TypedDict

from ccs.adapters.ccsstore import CCSStore


class GraphState(TypedDict):
    log: list[str]


def planner_node(state: GraphState) -> dict:
    store: CCSStore = lg_get_store()  # type: ignore[assignment]
    store.put(("planner", "shared"), "plan", {"step": 1})
    return {"log": [*state["log"], "planner: wrote plan"]}


def reviewer_node(state: GraphState) -> dict:
    store: CCSStore = lg_get_store()  # type: ignore[assignment]
    item = store.get(("reviewer", "shared"), "plan")
    assert item is not None
    return {"log": [*state["log"], "reviewer: read plan"]}


def build_graph(store: CCSStore):
    builder = StateGraph(GraphState)
    builder.add_node("planner", planner_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "reviewer")
    builder.add_edge("reviewer", END)
    return builder.compile(store=store)


# Wrap the store with record_to(...) for the duration of the run.
with CCSStore.record_to("/tmp/coherence-session", strategy="lazy") as store:
    graph = build_graph(store)
    graph.invoke({"log": []})
```

The session directory now contains a `manifest.json` plus one JSONL file per
captured stream (`state_log.jsonl`, `content_audit_log.jsonl`). Inspect it with:

```bash
# Human-readable findings + summary
agent-coherence-replay /tmp/coherence-session

# Machine-readable, one JSON object per line (per-finding + final summary)
agent-coherence-replay /tmp/coherence-session --json | jq .
```

### Exit codes

| Exit code | Meaning |
|---|---|
| `0` | Clean trace (or all SKIPPED entries are explicit compliance opt-outs). Also: `BrokenPipeError` from a closing consumer (e.g. `\| head -5`) — pipe-close is not a failure. |
| `1` | At least one CONFIRMED invariant breach |
| `2` | Capture-side bug: a manifest-declared stream is missing from the directory |
| `3` | Trace error (`MultiInstanceTraceError`, `TraceCorruptionError`, `ManifestMissingOrUnreadableError`, `SessionDirectoryNotFoundError`). Under `--json`, a final NDJSON line lands on stdout: `{"kind":"error","exit_code":3,"exception":"<ClassName>","message":"..."}` (in addition to the human-prose stderr line). |
| `4` | Internal error (uncaught exception inside replay — CLI bug, please file an issue). Distinct from exit 1 so agents can triage "tool crashed" vs "real coordination defect found." |

### Useful flags

- `--invariant <name>` (repeatable) — restrict to a subset of `single-writer`, `monotonic-version`, `stale-read`, `lost-write`.
- `--include-ambiguous` — show same-tick read/commit collisions as per-finding entries (suppressed from default output; always counted in summary).
- `--ambiguous-threshold N` (default `10`) — when the AMBIGUOUS count exceeds the threshold, the summary block emits a prominent callout naming both remedies (`--include-ambiguous` now; D+1 global-sequence-number capture eventually). Strict `>`; does not affect exit code.
- `--quiet` — suppress non-breach output; cron-friendly. Honored under both human and `--json` mode.
- `--json` — newline-delimited JSON conforming to the trace-format schema. Per-finding lines + one summary object; under exit 3, an `{"kind":"error", ...}` line lands on stdout (see Exit codes above).

### Capturing PII-constrained traces

Pass `streams={"state_log"}` to opt out of the content-audit stream while
keeping the other three invariants live:

```python
with CCSStore.record_to(
    "/tmp/coherence-session",
    streams={"state_log"},
    strategy="lazy",
) as store:
    ...
```

Replay then reports stale-read as SKIPPED with `opted_out=True` and the run
still exits 0.

### Refuse-if-exists safety

`CCSStore.record_to(path)` refuses to start when `path/manifest.json`
already exists (raises `SessionDirectoryNotEmptyError`). Without this
guard, a second capture against the same path would silently interleave
JSONL entries from two coordinator instances; `TRACE_CORRUPTION_DUPLICATE_SEQ`
would eventually fire at replay, but only AFTER findings from the mixed
session had already been emitted. Delete the directory or choose a
different path:

```python
import shutil
shutil.rmtree("/tmp/coherence-session", ignore_errors=True)
with CCSStore.record_to("/tmp/coherence-session", strategy="lazy") as store:
    ...
```

### Exception hierarchy

All replay-side exceptions inherit from `ccs.replay.ReplayError` with a
two-tier semantic split:

- `ReplayConfigurationError` — API misuse / wrong entry point
  (`UnverifiedAdapterCaptureError`, `SessionDirectoryNotEmptyError`).
- `ReplayTraceError` — trace structural defects (`ManifestMissingOrUnreadableError`,
  `MultiInstanceTraceError`, `TraceCorruptionError`,
  `SessionDirectoryNotFoundError`). The CLI catches this base class and
  maps every subclass to exit code 3, so future trace-error subclasses
  auto-route without touching the handler.

Catch the base class in your own scripts:

```python
from ccs.replay import ReplayTraceError, load, run_predicates

try:
    loaded = load(session_dir)
    findings, summary = run_predicates(loaded)
except ReplayTraceError as exc:
    # Manifest missing, multi-instance, duplicate seq, etc. — all caught here.
    log.error("Trace defect: %s", exc)
    raise
```

### Non-LangGraph adapters (CrewAI / AutoGen)

CrewAI and AutoGen capture is wired through the same `CoherenceAdapterCore`
seam via `ccs.replay.record_callbacks(...)`, but v1 only verifies the
LangGraph path end-to-end. Direct callers must pass `accept_unverified=True`
to acknowledge the v1 scope boundary — file an issue if the unverified path
breaks for your stack. Ergonomic per-adapter wrappers (`record_to` mirrors)
ship in the next release.

---

## Command-line tools

All bundled CLIs are installed as console scripts when you
`pip install agent-coherence`.

| Command | Extra needed | What it does |
|---|---|---|
| `ccs-diagnose` | `[diagnose]` | Detect stale reads / divergent versions in a LangGraph graph |
| `ccs-benchmark` | `[langgraph,benchmark]` | Measure token savings of `CCSStore` on your own LangGraph graph |
| `ccs-simulate` | — | Run a protocol-only simulation scenario from a YAML file |
| `ccs-compare` | — | Compare two or more strategies on the same scenario |
| `ccs-check-architecture` | — | Verify the four-layer architecture boundary (also runs in CI) |
| `agent-coherence-replay` | `[langgraph]` | Replay a captured coordinator session and report invariant breaches |
| `agent-coherence-workspace` | — | Checkpoint / list / status / restore a workspace of file and forward-only members; see [Workspace versioning & restore](#workspace-versioning--restore-workspaceversioner) |

Run any command with `--help` for the full option list.

### `ccs-simulate` and `ccs-compare`

```bash
# Run a single strategy against a YAML scenario
ccs-simulate --scenario benchmarks/scenarios/planning_canonical.yaml --strategy lazy

# Compare two or more strategies on the same scenario
ccs-compare --scenario benchmarks/scenarios/planning_canonical.yaml --strategies eager lazy
```

YAML scenarios in `benchmarks/scenarios/` define a deterministic workload
(agent count, artifacts, write probability, network latency/loss, strategy
config). Output is a `StrategyComparisonReport` printed to stdout — useful
when you want protocol-only numbers without spinning up a real LangGraph
graph. See [reproduce.md](reproduce.md) for the simulation methodology behind
the paper's headline numbers.

### `ccs-check-architecture`

```bash
# Architecture boundary check — fails non-zero if any layer imports upward
ccs-check-architecture
```

Designed for CI gating; runs on every push. The companion script `tools/check_release_readiness.py` (also runs in CI as the release-workflow preflight) is maintainer-only and intentionally not exposed as a console script — it queries this repo's GitHub admin settings and has no end-user use case.

---

## API reference

### `CCSStore(strategy, benchmark, on_metric, telemetry, on_error, state_log, content_audit_log, crash_recovery, **strategy_kwargs)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `strategy` | `str` | `"lazy"` | Synchronization strategy: `"lazy"`, `"eager"`, `"lease"`, `"access_count"`, `"broadcast"` |
| `benchmark` | `bool` | `False` | Enable inline token-savings measurement; access results via `benchmark_summary()` / `print_benchmark_summary()` |
| `on_metric` | `Callable[[StoreMetricEvent], None] \| None` | `None` | Callback fired after every operation with per-op metrics |
| `telemetry` | `str \| TelemetryExporter \| None` | `None` | `"opentelemetry"`, `"langsmith"`, a `TelemetryExporter` instance, or `None` |
| `on_error` | `str` | `"strict"` | `"strict"` to propagate `CoherenceError`; `"degrade"` to fall back silently |
| `state_log` | `Callable[[dict], None] \| None` | `None` | Callback fired on every stable MESI state transition; see [State transitions log](#state-transitions-log) |
| `content_audit_log` | `Callable[[dict], None] \| None` | `None` | Callback fired on every content delivery; see [Content audit log](#content-audit-log). Enables version retention. |
| `crash_recovery` | `CrashRecoveryConfig \| None` | `None` | Crash-recovery configuration; see [Crash recovery](#crash-recovery). `None` uses `CrashRecoveryConfig()` — enabled by default as of v0.9.0. |
| `**strategy_kwargs` | `Any` | — | Forwarded to the strategy constructor (`lease_ticks`, `threshold`, etc.) |

### Public imports

```python
from ccs.adapters import (
    CCSStore,
    StoreMetricEvent,
    TelemetryExporter,
    NoOpTelemetryExporter,
    OtelExporter,
    LangSmithExporter,
    build_telemetry,
)
from ccs.coordinator.service import CrashRecoveryConfig

# OpenAI Agents SDK adapter (experimental). These imports and wrap_session work on a
# bare install; only run_hooks() requires the openai-agents extra (deferred import).
from ccs.adapters import OpenAIAgentsAdapter, CoherenceSession
```

### `gate(volume, path, *, decide, effect)`

Order an escaping side effect (a deploy, an opened PR, a notification) against a shared input so it fires only on the input state it was decided from. `gate()` captures the input's `(version, ownership generation)` pair (one atomic coordinator snapshot, via `volume.read_with_version_generation`), runs `decide`, re-reads the pair at the effect boundary, and fires `effect` only if **both** are unchanged and confirmed — otherwise it raises `StaleView` (a HOLD) before the effect runs. The two comparands answer different questions: the version answers "is the value still the one `decide` saw"; the generation answers "is the grant it was read under still standing". A coordinator sweep that reclaims a stalled holder's grant advances the generation **without** a version move, so a version-only check would fire a reclaimed (zombie) holder's effect — the generation leg holds it. One revocation moves **neither** comparand: a peer's pessimistic write-acquire takes the holder's grant with no commit behind it yet (version unmoved) and no epoch bump (that trigger is deliberately outside the bump set). The gate closes that leg too — the re-read at the effect boundary must itself be served under a *standing* grant; a preempted holder's re-read comes back stale, and the effect holds with the typed `grant_preempted` cause. That re-read is a verification read the coordinator does not re-grant, so the hold is level-triggered: a bare re-gate holds again, and only `reacquire()` clears it.

```python
from ccs.adapters import CoherentVolume, gate

vol = CoherentVolume(workspace_root, managed=("deploy/**",))

# fires run_deploy(plan) only if deploy/config.txt is unchanged since decide() read it;
# else raises StaleView before the deploy runs — reacquire() and re-decide.
gate(vol, "deploy/config.txt", decide=plan_deploy, effect=run_deploy)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `volume` | `CoherentVolume` | A volume attached to the coordinator that tracks `path`. |
| `path` | `str \| os.PathLike[str]` | The workspace-relative managed artifact whose `(version, ownership generation)` pair gates the effect. |
| `decide` | `Callable[[bytes], D]` | Keyword-only. Reads the captured bytes and returns a decision passed to `effect`. |
| `effect` | `Callable[[D], R]` | Keyword-only. The escaping side effect; fired only if the input is unchanged at the re-read. |

Returns whatever `effect` returns. Raises `StaleView` — carrying `expected_version` / `current_version`, `expected_generation` / `current_generation`, and a typed `hold_cause` — if the input moved, vanished, lost the grant it was read under, or could not be confirmed.

Branch on `hold_cause` rather than the message. It carries one of the
published hold reasons — the same vocabulary, with the same meanings and the
same recoveries, that the coordinator answers with over HTTP: see
[Hold reasons](#hold-reasons--the-published-vocabulary). Reasons may be added
and are never renamed, so match the whole value and treat one you do not
recognise as a hold.

One reason in that vocabulary this wrapper cannot raise: `content_claim_absent`.
`gate()` reads a hash *comparison* from the coordinator rather than the hash it
records, so "the claim matches" and "there is no claim at all" arrive here as
one value and the wrapper assumes a claim exists. The
[HTTP fence](#effect-fence-over-http) reads the coordinator's own record and is
the surface that can answer that leg.

**Fail-closed comparands.** An unconfirmed version (`0` — a degraded read) or an unconfirmed generation (`None` — a coordinator deny, a degraded read, an out-of-band edit the coordinator could not confirm, or an older coordinator daemon from before this release's generation reporting) always HOLDs. In particular, this gate against an older coordinator daemon HOLDs loudly rather than silently reverting to the generation-blind check — restart the coordinator on the current version to clear it.

**Scope.** Escaping effects only — a pure *write* effect uses `volume.write_cas_at(path, expected_version, content)` directly. The gate *orders* effects and never rolls one back, so for an escaping effect there is a residual re-read→fire window it narrows but cannot close. Single-host and cooperative (the caller opts in). Gating several mutually-consistent inputs at once is a coordinator-side operation, not this single-input wrapper.

---

## Low-level adapter API

For CrewAI, AutoGen, or custom integrations, use the `before_node` / `commit_outputs`
surface directly:

```python
from ccs.adapters.langgraph import LangGraphAdapter
from ccs.coordinator.service import CrashRecoveryConfig

adapter = LangGraphAdapter(
    strategy_name="lazy",
    crash_recovery=CrashRecoveryConfig(enabled=True, heartbeat_timeout_ticks=120, max_hold_ticks=900),
)
for name in ("planner", "researcher", "executor"):
    adapter.register_agent(name, now_tick=0)
plan = adapter.register_artifact(name="plan.md", content="v1")

context = adapter.before_node(agent_name="planner", artifact_ids=[plan.id], now_tick=1)
adapter.commit_outputs(
    agent_name="planner",
    writes={plan.id: context[plan.id]["content"] + "\nStep 1"},
    now_tick=2,
)

# During long compute — bridge the heartbeat gap
adapter.core.heartbeat(agent_name="planner", now_tick=5)

# After process restart — invalidate stale cache and re-seed heartbeat
adapter.core.recover(agent_name="planner", now_tick=100)
```

The same pattern applies to `CrewAIAdapter` and `AutoGenAdapter` — all accept
`crash_recovery=` and expose `core.heartbeat()` / `core.recover()`.

Full example: [`examples/multi_agent_planning.py`](../examples/multi_agent_planning.py).

---

## CrewAI and AutoGen adapters

The protocol is framework-agnostic; only the adapter surface changes. Both
adapters share the same `register_agent`, `register_artifact`, `before_node`,
`commit_outputs`, `heartbeat`, and `recover` API as `LangGraphAdapter`.

### CrewAI

```bash
pip install "agent-coherence[crewai]"
```

```python
from ccs.adapters.crewai import CrewAIAdapter
from ccs.coordinator.service import CrashRecoveryConfig

adapter = CrewAIAdapter(
    strategy_name="lazy",
    crash_recovery=CrashRecoveryConfig(enabled=False),
)
for name in ("researcher", "writer", "editor"):
    adapter.register_agent(name, now_tick=0)
brief = adapter.register_artifact(name="brief.md", content="initial brief")

# Read shared state at task start
ctx = adapter.before_node(agent_name="researcher", artifact_ids=[brief.id], now_tick=1)

# Write task output back
adapter.commit_outputs(
    agent_name="researcher",
    writes={brief.id: ctx[brief.id]["content"] + "\nfindings: ..."},
    now_tick=2,
)
```

### AutoGen

```bash
pip install "agent-coherence[autogen]"
```

```python
from ccs.adapters.autogen import AutoGenAdapter

adapter = AutoGenAdapter(strategy_name="lazy")
# Same register_agent / register_artifact / before_node / commit_outputs surface.
```

### Custom orchestrators

```python
from ccs.adapters.base import CoherenceAdapterCore

adapter = CoherenceAdapterCore(strategy_name="lazy")
# Same surface. Wrap whatever framework you're using and call before_node /
# commit_outputs at the natural boundaries (typically: before a tool call or
# LLM step, and after the step produces new state).
```

Crash recovery (`heartbeat` / `recover`) is identical across all four adapters.

---

## OpenAI Agents SDK adapter (experimental)

> **Status: experimental (0.x).** The [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
> is itself 0.x and its surface churns; this adapter is pinned to
> `openai-agents>=0.17,<0.18` and may change with it. Install with
> `pip install "agent-coherence[openai-agents]"`.

The OpenAI Agents SDK has no `BaseStore`-style seam, so this adapter does **not**
use the `before_node` / `commit_outputs` surface. The coherence target here is the
SDK's **`Session`** — the agent's local conversation memory
(`get_items` / `add_items` / `pop_item` / `clear_session`). A peer that mutates a
shared session leaves this agent's cached view stale, *regardless of how consistent
the durable store is*. (The consistency probe measured the OpenAI and Mistral Conversations
servers read-after-write consistent — so the coherence value lives on the readers'
caches, not on the server. See the [Conversations stale-read example](#conversations-stale-read).)

The SDK exposes no Session hook/middleware API, so interception is by **composition**:
`wrap_session` wraps a caller-provided Session and overrides the four async methods.
Because the underlying Session is supplied by the caller, the adapter module imports
no `agents` symbol — it works against anything implementing the four-method protocol.

**Scope (v1):** in-process multi-agent coherence — peers registered on one
`OpenAIAgentsAdapter` / `CoherenceAdapterCore` per process, the same boundary as the
LangGraph / CrewAI / AutoGen adapters. Cross-service coherence needs the
out-of-process coordinator.

### Wrapping a Session

```python
from agents import Runner, SQLiteSession
from ccs.adapters import OpenAIAgentsAdapter

adapter = OpenAIAgentsAdapter(strategy_name="lazy")  # on_error defaults to "degrade"

# Every agent wraps the *same* session_id through the adapter, so their reads and
# writes coordinate on one shared coherence artifact (single registration, shared id).
planner_session = adapter.wrap_session(
    SQLiteSession("chat-1"), agent_name="planner", session_id="chat-1"
)
reviewer_session = adapter.wrap_session(
    SQLiteSession("chat-1"), agent_name="reviewer", session_id="chat-1"
)

await Runner.run(planner_agent, "draft the plan", session=planner_session)

# Before the reviewer acts, check whether a peer moved the conversation underneath it.
if reviewer_session.peer_mutated_since_read():
    await reviewer_session.get_items()  # take the cache miss, re-read the fresh version
```

`CoherenceSession` is a drop-in `Session`: pass it anywhere the SDK expects a session.
A mutation (`add_items` / `pop_item` / `clear_session`) persists to the underlying
Session **first**, then invalidates peers; `get_items` refreshes this agent's coherence
state so a prior peer write surfaces as a cache miss. The underlying Session stays the
durable source of truth for the items — the coherence layer governs *awareness*, not
storage.

`peer_mutated_since_read()` is conservative by design: it returns `True` when a peer
has mutated the session since this agent's last read, **and** when this agent has never
read yet (no baseline → "you must read first"). Call `get_items()` once to establish the
baseline; after that it reports only genuine peer mutations.

### RunHooks lifecycle (optional)

Attach `run_hooks(...)` to `Runner.run(..., hooks=...)` to thread coherence accounting
through a run. It tracks the active agent across handoffs and refreshes that agent's
coherence view at agent-start and tool-start, so a peer's mutation surfaces *before* the
agent acts. Writes remain the `CoherenceSession`'s job — the hooks coordinate awareness
and identity, not arbitrary tool side effects.

```python
hooks = adapter.run_hooks(session_id="chat-1")
await Runner.run(agent, "...", session=planner_session, hooks=hooks)
```

Importing `agents.RunHooks` is deferred until this call, so the module (and
`wrap_session`) stays usable on a bare install; `run_hooks` raises `ImportError` with an
install hint if the `openai-agents` extra is absent.

**Server-side conversations + handoffs.** Pass `server_conversation=True` when the run
uses a server-side `conversation_id`. Combined with a multi-agent handoff, the SDK
disables `input_filter` / nested handoff history, so handoff-history coherence is
unavailable — the first handoff then warns once with `CoherenceTopologyWarning` rather
than silently assuming it works. The supported topology is concurrent independent agents
sharing one `conversation_id` within one process.

### Parity and error handling

`OpenAIAgentsAdapter` mirrors the other adapters' constructor and exposes
`register_agent`, `register_artifact`, `heartbeat(agent_name=, now_tick=)`, and
`recover(agent_name=, now_tick=)`, plus `is_degraded` / `degradation_count`. It accepts
`crash_recovery=CrashRecoveryConfig(...)` like the rest. The SDK has no step counter, so
the adapter mints its own monotonic tick internally — you do not pass `now_tick` to
`wrap_session` / `run_hooks`.

One difference from `CCSStore`: `on_error` defaults to **`"degrade"`** here (best-effort
coherence that never swallows the underlying Session op), not `"strict"`. Use
`on_error="strict"` to propagate `CoherenceError`.

> **Degrade-mode caveat for concurrent writers.** Under `on_error="degrade"`, a
> `CoherenceError` from a mutation is swallowed after the Session write already
> succeeded. Because `core.write` grants EXCLUSIVE and then commits as two steps, a
> failed commit (e.g. a concurrent writer reclaimed the grant) can strand the writer
> holding a stable EXCLUSIVE grant with peers already invalidated. That grant is only
> reclaimed by the crash-recovery sweep. For concurrent-writer workloads on the same
> session under degrade, enable `CrashRecoveryConfig` so stranded grants self-heal.
