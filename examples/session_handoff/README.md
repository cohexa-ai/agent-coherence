# Session handoff demo

Two agent sessions on one machine share a scratch file, `handoff/notes.md`, so
that "a teammate picks up exactly where you left off" — the workaround people
describe in [anthropics/claude-code#60082](https://github.com/anthropics/claude-code/issues/60082).
This demo runs that workaround as two **real OS processes** and shows what the
file does under it: without coordination, the second session's write quietly
erases the first session's line; with `CoherentVolume` on the file, the stale
write is denied and both lines survive. It then hands work off through a
checkpoint and shows the two honest outcomes of a rewind — clean when the
handing-off session has stopped writing, a bounded `conflict` (never a clobber) when it
is still typing.

Offline, deterministic, no API keys. Every act runs the same read→write
sequence; only the coordination differs.

## Prerequisites

- Python 3.11+
- From the repo root: `pip install -e .` (pulls `pyyaml`)
- Run from the repo root

```bash
python -m examples.session_handoff.main
```

Exits `0` only if **all five** hold:

| Act | Sequence | What it pins |
|-----|----------|--------------|
| **RED** — plain file I/O | A posts status → B reads, appends its pickup line → A (never re-read) writes its updated status | B's line is **gone**; nothing raised |
| **GREEN** — `handoff/**` guarded | the same sequence through `CoherentVolume` | A's stale write is **denied** (`StaleView`); A reacquires and rebuilds; both lines survive *exactly* |
| **CONTROL** — guard pointed elsewhere | the GREEN code with `other/**` as the strict glob | the loss returns — green depends on the deny, not on the re-read |
| **HANDOFF** — A stopped writing | A checkpoints, writes once more, stops writing → B restores the checkpoint → A writes again | restore lands in **one attempt**, disk rewound; A's next write is **denied**, and `reacquire` shows the handoff bytes |
| **HANDOFF** — A still working | A checkpoints and keeps editing → B restores | restore concludes **`conflict`** after its bounded budget; A's latest edit survives, nothing clobbered |

## Scope, honestly

- **Single host.** Both sessions run on one machine — the shape people are
  actually using: a shared scratch file between two sessions on the same box.
  Nothing here makes a cross-host claim.
- **The workspace half only.** A handoff has two halves: the conversation and
  the workspace. Only Anthropic can share the conversation; this demo covers the
  workspace — the file the sessions coordinate through and the checkpoint one
  hands the other.
- **Sequenced, not raced.** Each act runs its steps in a fixed order so the
  result is byte-identical every run. That is also the right model for a
  handoff: the loss in RED needs no timing window at all — a session that never
  re-reads before writing loses the other's line every time. In the last act a
  deterministic stand-in plays "A keeps typing", one edit per restore read.
- **File members are detection-guarded, never substrate-arbitrated.** The
  file's version check is the adapter's own detection of a foreign edit; the
  filesystem does not arbitrate. It catches the stale write and reports the
  conflict — that is the whole claim.
- **`restore` is a forward write of old bytes.** It writes the checkpointed
  content back as a new version; history is never rewritten, and a restore that
  meets a live writer stops and says so rather than overwriting.
- **Both sessions stay open.** In this demo the coordinator runs inside the
  first session's process, so that session stays open — idle, not exited —
  while the other one works. A handing-off session that exits, and a teammate
  who attaches afterwards, is a different shape and is not shown here.

## See also

- [`examples/mcp_stale_write_guard`](../mcp_stale_write_guard) — the same deny
  through the MCP server's tool contract, which is what a Claude Code session
  actually calls.
- [`examples/workspace_versioning`](../workspace_versioning) — the full restore
  surface across files and S3 objects, with per-member outcomes.

Comparing notes on multi-agent coherence?
https://github.com/Cohexa-ai/agent-coherence/discussions
