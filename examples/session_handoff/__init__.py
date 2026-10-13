# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Session-handoff demo: two sessions on one host share a scratch file.

Two long-lived agent sessions — each a REAL OS process holding its own
``CoherentVolume`` — coordinate through one shared scratch file,
``handoff/notes.md``, and hand work off asynchronously via a checkpoint. The
first session to construct its volume spawns the local coordinator; later
sessions attach to the same one. Offline, deterministic, no API keys.

This module holds only the shared constants so a spawned session process can
import it cheaply; the process primitive lives in :mod:`.sessions`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
# Every spawned session process must import ``ccs`` too (the coordinator itself
# serves in-thread inside the first session's child, so it inherits this path);
# propagate the src path so the demo runs from a bare checkout (harmless when
# installed).
_pp = os.environ.get("PYTHONPATH", "")
if str(SRC_ROOT) not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{SRC_ROOT}{os.pathsep}{_pp}" if _pp else str(SRC_ROOT)

from ccs.adapters.claude_code.lifecycle import LifecycleConfig  # noqa: E402

#: The one shared scratch file both sessions read and write.
NOTES = "handoff/notes.md"

A_STATUS_1 = b"status: migrating tables 1-3\n"
B_PICKUP = b"pickup: B taking tables 4-6\n"
A_STATUS_2 = b"status: migrating tables 1-4\n"

#: Strict glob covering ``NOTES`` — the guarded arm.
GUARDED = ("handoff/**",)
#: CONTROL: the strict glob deliberately leaves ``NOTES`` outside it, so the
#: same sequence runs with the deny disabled.
UNGUARDED = ("other/**",)

#: Stable owner identity for the demo's checkpoint manifests.
OWNER = uuid5(NAMESPACE_URL, "agent-coherence.examples.session_handoff")

#: Snappy local spawn for a one-command demo. ``idle_shutdown_sec=0`` DISABLES
#: idle shutdown so the coordinator cannot self-stop mid-run.
DEMO_CFG = LifecycleConfig(
    idle_shutdown_sec=0,
    sweep_interval_sec=0.1,
    port_file_retry_attempts=40,
    port_file_retry_interval_sec=0.05,
    connect_retry_attempts=20,
    connect_retry_interval_sec=0.05,
)
