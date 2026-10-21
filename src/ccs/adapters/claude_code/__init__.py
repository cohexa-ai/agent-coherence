# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Claude Code adapter — cross-process coherence for parallel Claude Code sessions.

This package wires the agent-coherence MESI protocol (via the existing
:class:`ccs.coordinator.service.CoordinatorService` and the new
:class:`ccs.coordinator.sqlite_registry.SqliteArtifactRegistry`) to Claude
Code's hook surface. See the plan at
``docs/plans/2026-05-13-001-feat-claude-code-coherence-plugin-v0.1-plan.md``.

Modules:
- :mod:`resolver` — find_coordinator_root via ``git rev-parse --git-common-dir``
- :mod:`policy` — tracked-artifact patterns + .coherence/{tracked,ignored}.yaml
- :mod:`auth` — shared-secret token generation + verification (KTD-12)
- :mod:`coordinator_server` — stdlib HTTP server wrapping CoordinatorService
- :mod:`lifecycle` — fcntl spawn, port file, idle shutdown
- :mod:`store_check` — what a coordinator start would do to the store (#345)
- :mod:`hook_payloads` — request/response JSON shapes (typed dicts)
"""
