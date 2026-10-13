# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Framework-facing integration adapters for CCS runtime.

Optional integrations such as ``CCSStore`` should not make the whole
``ccs.adapters`` package unimportable when their extra dependencies are absent.
Exports are therefore loaded lazily via ``__getattr__``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CCSStore",
    "CoherenceDegradedWarning",
    "CoherenceTopologyWarning",
    "CoherenceAdapterCore",
    "CoherentVolume",
    "CasCommitResult",
    "HandoffGrantResult",
    "HandoffTransferResult",
    "HandoffVerbResult",
    "HandoffWinOutcome",
    "gate",
    "coherent_workspace",
    "install",
    "uninstall",
    "CrashRecoveryConfig",
    "LangGraphAdapter",
    "CrewAIAdapter",
    "AutoGenAdapter",
    "OpenAIAgentsAdapter",
    "CoherenceSession",
    "StoreMetricEvent",
    "TelemetryExporter",
    "NoOpTelemetryExporter",
    "build_telemetry",
    "OtelExporter",
    "LangSmithExporter",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "AutoGenAdapter": (".autogen", "AutoGenAdapter"),
    "CCSStore": (".ccsstore", "CCSStore"),
    "CoherenceDegradedWarning": (".base", "CoherenceDegradedWarning"),
    "CoherenceTopologyWarning": (".base", "CoherenceTopologyWarning"),
    "CoherenceAdapterCore": (".base", "CoherenceAdapterCore"),
    # Re-exported from the coordinator package so the v0.9.0 opt-out
    # (CrashRecoveryConfig(enabled=False)) is reachable from ccs.adapters,
    # alongside the adapters whose default it now governs.
    "CrashRecoveryConfig": ("..coordinator.service", "CrashRecoveryConfig"),
    "CoherenceSession": (".openai_agents", "CoherenceSession"),
    "CoherentVolume": (".coherent_volume", "CoherentVolume"),
    # The typed results of the volume's handoff verbs and of a
    # compare-and-swap win (#185).
    "CasCommitResult": (".coherent_volume", "CasCommitResult"),
    "HandoffGrantResult": (".coherent_volume", "HandoffGrantResult"),
    "HandoffTransferResult": (".coherent_volume", "HandoffTransferResult"),
    "HandoffVerbResult": (".coherent_volume", "HandoffVerbResult"),
    "HandoffWinOutcome": (".coherent_volume", "HandoffWinOutcome"),
    "gate": (".effect_gate", "gate"),
    "coherent_workspace": (".coherent_volume", "coherent_workspace"),
    "install": (".coherent_volume", "install"),
    "uninstall": (".coherent_volume", "uninstall"),
    "CrewAIAdapter": (".crewai", "CrewAIAdapter"),
    "LangGraphAdapter": (".langgraph", "LangGraphAdapter"),
    "OpenAIAgentsAdapter": (".openai_agents", "OpenAIAgentsAdapter"),
    "LangSmithExporter": (".telemetry.langsmith", "LangSmithExporter"),
    "NoOpTelemetryExporter": (".telemetry", "NoOpTelemetryExporter"),
    "OtelExporter": (".telemetry.otel", "OtelExporter"),
    "StoreMetricEvent": (".events", "StoreMetricEvent"),
    "TelemetryExporter": (".telemetry", "TelemetryExporter"),
    "build_telemetry": (".telemetry", "build_telemetry"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

