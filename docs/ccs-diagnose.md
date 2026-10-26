# `ccs-diagnose` — detect stale reads in your LangGraph graph

`ccs-diagnose` attaches a passive callback to an existing LangGraph graph and classifies its write pattern (`single_writer` / `shared_artifact` / `parallel_branch` / `mixed_pattern` / `insufficient`). It reports artifacts whose reads were handed divergent versions across nodes — divergence the runtime is already producing but never surfaces.

It runs as a **witness-quality** surface: it observes what the runtime *handed* a node, not what the node read. Upgrading to `CCSStore` lifts these observations into provable per-key attribution — same diagnose surface, no callback rewiring.

**`ccs-diagnose` is detection only — it enforces nothing.** It surfaces divergence; it never denies a read or a write. What *closes* a divergence once you've seen it depends on which side the race is on:

- **Read-side drift** (a peer committed a new version, your cached view went stale) → **CCSStore** provides read-side coherence: when a peer commits a new version, your cached view is invalidated so your next read is a fresh miss. It does not deny a stale write-back — `put` is not version-CAS.
- **Write-side lost-update** (a stale writer overwriting a peer) → route writes through **CoherentVolume** or **`write_cas`**.

**Scope:** `ccs-diagnose` observes a single in-process run — the versions the runtime hands nodes inside one process. Divergence that happens across separate OS processes (two hosts, or two processes over shared files) is invisible to it; for cross-process coordination over files, reach for CoherentVolume.

The CLI makes **zero outbound network requests** in v0.

---

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Common invocations](#common-invocations)
4. [Flags](#flags)
5. [Exit codes](#exit-codes)
6. [Trust posture](#trust-posture)
7. [Calibration corpus](#calibration-corpus)
8. [Promotion to `langgraph-v1`](#promotion-to-langgraph-v1)
9. [Sharing calibration data](#sharing-calibration-data)

---

## Install

```bash
pip install "agent-coherence[diagnose]"
```

## Quick start

```bash
ccs-diagnose --graph path/to/your_graph.py:build_graph

# Or run it right now against the bundled example graph — no setup:
ccs-diagnose --graph examples/langgraph_planner/main.py:build_graph_no_store
```

The factory must accept zero arguments (or accept an initial state file via `--state-file`) and return a compiled LangGraph graph. The CLI runs the graph once with a passive observer and writes two reports:

| Output | Default path | Purpose |
|---|---|---|
| HTML report | `diagnose_report.html` | Human-readable: classification verdict, divergence witnesses, tracked-artifacts panel, cost extrapolation, CTA |
| JSON report | `diagnose_report.json` | Machine-readable: full schema for piping into `jq`, dashboards, or `--show-payload` re-rendering |

## Common invocations

```bash
# Basic — run the pipeline and write both reports
ccs-diagnose --graph my_graph.py:build_graph

# Pass an initial state
ccs-diagnose --graph my_graph.py:build_graph --state-file initial_state.json

# Cost extrapolation (interactions per hour)
ccs-diagnose --graph my_graph.py:build_graph --volume 50

# Pipe JSON straight into jq (summary goes to stderr)
ccs-diagnose --graph my_graph.py:build_graph --output-json - | jq .verdict

# Re-render a previously emitted report without re-running the graph
ccs-diagnose --show-payload diagnose_report.json

# Strict mode — promote sequential-staleness exclusions back into the headline
ccs-diagnose --graph my_graph.py:build_graph --strict

# Restrict tracked keys
ccs-diagnose --graph my_graph.py:build_graph --ignore tmp_buf,debug_trace --track shared_doc

# Calibration corpus — appends a single payload entry to a local JSONL file
ccs-diagnose --graph my_graph.py:build_graph --calibration-record

# Non-TTY agent invocations (consent prompt would otherwise block)
ccs-diagnose --graph my_graph.py:build_graph --yes
```

## Flags

| Flag | Purpose |
|---|---|
| `--graph PATH:FUNCTION` | Path + factory name; required for the main pipeline |
| `--state-file PATH` | JSON or YAML file passed to `graph.invoke()` |
| `--output-html PATH` | HTML output (default: `diagnose_report.html`) |
| `--output-json PATH` | JSON output (default: `diagnose_report.json`; `-` for stdout) |
| `--no-json` | Suppress JSON output |
| `--volume N` | Interactions per hour for annualized cost extrapolation |
| `--lead-pain-type {cost,auditability,auto}` | Which secondary KPI rides the headline |
| `--cost-per-1k-tokens FLOAT` | Token cost assumption for extrapolation |
| `--strict` | Promote sequential-staleness exclusions back into the headline count |
| `--ignore KEY1,KEY2` | State keys to skip (supports `__`-prefixed) |
| `--track KEY1,KEY2` | Force-track keys (wins over `--ignore`) |
| `--warm-lead` | Switch the CTA to a warm-conversation 2-question seed |
| `--book-a-call-url URL` | Override CTA calendar URL (also: `CCS_DIAGNOSE_BOOK_A_CALL_URL`) |
| `--contact-email EMAIL` | Override CTA reply-to (also: `CCS_DIAGNOSE_CONTACT_EMAIL`) |
| `--show-payload PATH` | Re-render a previously emitted `report.json`; bypasses graph load |
| `--dry-run` | Print the payload that would be submitted (v0 has no network code) |
| `--no-network` / `--no-telemetry` | Suppress consent prompt and calibration write |
| `--yes` / `--non-interactive` | Skip consent prompt (required for non-TTY agent invocations) |
| `--calibration-record [PATH]` | Append payload to local JSONL calibration corpus |
| `--reset-token` | Regenerate the local installation token |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Generic error (graph load failure, JSON parse, etc.) |
| `2` | Usage error (argparse, mutually exclusive flags, bad URL/email) |
| `3` | Dependency missing (import error from graph or extras) |
| `4` | Schema mismatch (`--show-payload` version not supported) |
| `5` | I/O error (write/read failed; oversized input file) |

## Trust posture

`ccs-diagnose` makes **zero outbound network requests** in v0. The HTML renderer uses Jinja2 with `select_autoescape`. The `--book-a-call-url` and `--contact-email` values are validated against an allowlist that rejects `javascript:`, `data:`, and `vbscript:` schemes (the same allowlist guards the `CCS_DIAGNOSE_BOOK_A_CALL_URL` / `CCS_DIAGNOSE_CONTACT_EMAIL` env-var overrides).

Three independent kill switches suppress all telemetry-shaped output (no consent prompt, no calibration write, no payload generation):

- `DO_NOT_TRACK=1` (cross-tool consensus per consoledonottrack.com)
- `DISABLE_TELEMETRY=1`
- `CCS_DIAGNOSE_NO_TELEMETRY=1`

`--no-telemetry` and `--no-network` are the per-invocation equivalents.

See [security.md](security.md) for the full trust contract, supply-chain threat model, and attestation-verification commands.

## Calibration corpus

The `--calibration-record` flag appends one entry per run to a local JSONL file at `$XDG_DATA_HOME/ccs-diagnose/calibration.jsonl` (or `~/.local/share/ccs-diagnose/calibration.jsonl` when `XDG_DATA_HOME` is unset). The file is mode `0600`. Override the path with `--calibration-record /tmp/my_calibration.jsonl`.

Each entry contains:

- Stack name and version (e.g., `LangGraph 1.1.10`)
- Classifier verdict + confidence
- Coverage shape (counts only — no key names, no content, no hashes)
- Timestamp
- Schema version + sequence number + instance ID

Nothing else. The file is `validate_log`-compatible:

```bash
python -c "from ccs.validation import validate_log; \
    print(validate_log('$XDG_DATA_HOME/ccs-diagnose/calibration.jsonl', \
                       schema_version='ccs.diagnose.v0-preview'))"
```

A clean file returns `([], [])` — empty gap and schema-mismatch lists.

The append is gated by consent: runs without granted consent (or with `DO_NOT_TRACK` / `DISABLE_TELEMETRY` / `CCS_DIAGNOSE_NO_TELEMETRY` set, or `--no-telemetry` / `--no-network` passed) print a skip message and exit `0` without touching the file.

## Promotion to `langgraph-v1`

`ccs-diagnose` ships under the `langgraph-v0-preview` classifier. Submissions tagged `v0-preview` are excluded from the public benchmark — they accumulate in the local JSONL file you can share back with us to contribute to v1 promotion.

The `v0-preview` → `v1` promotion gate requires:

1. **>= 5 real production graphs** validated across **>= 3 distinct supervisor topologies**
2. The Tracked-Artifacts panel produces **zero unknown `__`-prefix surprises** on a current LangGraph release
3. The append-only prefix-stability rule survives a workload that **compacts message history mid-run** (real-world `trim_messages` pattern)

Once promoted, `v1`-tagged submissions populate the public benchmark; `v0-preview` calibration data is not retroactively migrated.

## Sharing calibration data

DM `vlad@agent-coherence.dev` (or open an issue on the repo) with your `calibration.jsonl` contents. Nothing is collected that isn't documented above; you can read every field before sharing.

### Broken pipes in console and embedded runs

On POSIX, the console entry point retains the usual quiet SIGPIPE behavior
for its own output. While loading, building and invoking the user graph, it
instead ignores SIGPIPE so graph code can catch `BrokenPipeError` from its
own sockets. This is a **process-wide**, not thread-local, window; it closes
when graph invocation returns, restoring the previous signal handler.

Within that window, `sys.stdout.write()` and `sys.stdout.flush()` (including
`print()`) are wrapped: a
broken stdout pipe ends the process immediately with shell status **141**,
without turning the error into an `insufficient` report. This also works
when graph workers print. Python's `subprocess` sees **+141** for this
explicit exit, versus **-13** for a SIGPIPE signal death outside the window.
The immediate exit intentionally does not run `finally`/`atexit` cleanup;
normal returns and ordinary graph errors restore both stdout and the signal
handler. Other stream errors and graph-owned socket errors are not swallowed.

This narrow protection does not wrap stderr, raw file-descriptor writes,
`sys.stdout.buffer`, delegated methods such as `writelines()`, or previously
captured stdout aliases. A small buffered print may only flush after this
window, retaining the usual signal behavior at that later write; the
no-report guarantee applies to broken stdout writes detected inside the
window. Graph-owned
threads/exporters and `atexit` handlers running after invocation returns are
outside the window and remain subject to the console's default SIGPIPE
handler, even if that code would otherwise catch `BrokenPipeError`.
Embedded callers of `main()` do not enter this guard or replace host streams
or signal handlers, including when called from a worker thread.
