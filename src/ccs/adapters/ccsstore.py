# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""CCSStore: LangGraph BaseStore adapter with MESI-style cache coherence."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from ccs.adapters.base import CoherenceAdapterCore
from ccs.adapters.events import CCS_METRIC_SCHEMA_VERSION, StoreMetricEvent  # re-exported for public API compatibility
from ccs.adapters.telemetry import TelemetryExporter, build_telemetry
from ccs.agent.runtime import CCS_CONTENT_AUDIT_LOG_SCHEMA_VERSION
from ccs.coordinator.service import CrashRecoveryConfig
from ccs.core.exceptions import CoherenceDegradedWarning, CoherenceError
from ccs.core.hashing import compute_content_hash
from ccs.core.identity import artifact_uuid
from ccs.core.states import MESIState
from ccs.core.types import Artifact

logger = logging.getLogger(__name__)

# CoherenceDegradedWarning is re-exported from ccs.core.exceptions (canonical home)
# so CCSStore and every other adapter raise/catch the same class.
__all__ = ["CCSStore", "CoherenceDegradedWarning", "StoreMetricEvent"]


class CCSStore(BaseStore):
    """Drop-in replacement for LangGraph's InMemoryStore with MESI cache coherence.

    A peer write invalidates all other agents' cached copies; their next get() is a
    cache miss (coordinator fetch) while subsequent reads within the same MESI grant
    are cache hits.

    Namespace convention: namespace[0] is the agent identity, namespace[1:] is the
    artifact scope. Two agents writing to ("planner", "shared") and ("reviewer",
    "shared") address the same artifact. For agent-private artifacts include the agent
    name in the scope: ("planner", "planner", "private").
    """

    def __init__(
        self,
        strategy: str = "lazy",
        on_metric: Callable[[StoreMetricEvent], None] | None = None,
        telemetry: str | TelemetryExporter | None = None,
        on_error: str = "strict",
        benchmark: bool = False,
        state_log: Callable[[dict[str, Any]], None] | None = None,
        content_audit_log: Callable[[dict[str, Any]], None] | None = None,
        crash_recovery: CrashRecoveryConfig | None = None,
        **strategy_kwargs: Any,
    ) -> None:
        if on_error not in ("strict", "degrade"):
            raise ValueError(f"on_error must be 'strict' or 'degrade'; got {on_error!r}")
        if not isinstance(benchmark, bool):
            raise TypeError(f"benchmark must be a bool; got {type(benchmark).__name__!r}")
        self._instance_id = str(uuid.uuid4())
        self._metric_seq: int = 0
        self._content_audit_log = content_audit_log
        self._audit_seq: list[int] = [0]
        self.core = CoherenceAdapterCore(
            strategy_name=strategy,
            state_log=state_log,
            instance_id=self._instance_id,
            content_audit_log=content_audit_log,
            audit_seq=self._audit_seq,
            retain_versions=content_audit_log is not None,
            crash_recovery=crash_recovery,
            **strategy_kwargs,
        )
        self._on_metric = on_metric
        self._telemetry: TelemetryExporter = build_telemetry(telemetry)
        self._on_error = on_error
        self._lock = threading.Lock()
        self._tick: int = 0
        # (scope, key) → artifact_id; scope is namespace[1:]
        self._artifact_map: dict[tuple[tuple[str, ...], str], uuid.UUID] = {}
        self._known_agents: set[str] = set()
        # artifact IDs removed via delete; bounded by distinct (scope, key) pairs ever deleted
        self._deleted_ids: set[uuid.UUID] = set()
        # full namespace (including agent) → set of keys; maintained on put/delete
        self._namespace_index: dict[tuple[str, ...], set[str]] = {}
        # plain dict fallback when on_error="degrade"; keyed by (scope, key)
        self._fallback_store: dict[tuple[tuple[str, ...], str], Any] = {}
        # degradation tracking (R8 additions)
        self._degradation_count: int = 0
        # inline benchmark counters — None when benchmark=False (zero overhead)
        self._bm_baseline: int | None = 0 if benchmark else None
        self._bm_actual: int | None = 0 if benchmark else None
        self._bm_ops: int | None = 0 if benchmark else None
        self._bm_hits: int | None = 0 if benchmark else None

    # ------------------------------------------------------------------
    # Replay capture — thin wrapper over ccs.replay.record_callbacks
    # ------------------------------------------------------------------

    @classmethod
    @contextmanager
    def record_to(
        cls,
        path: str | Path,
        *,
        streams: set[str] | frozenset[str] | None = None,
        **ccsstore_kwargs: Any,
    ) -> Iterator["CCSStore"]:
        """LangGraph-shaped capture context manager.

        ``with CCSStore.record_to(path) as store: ...`` produces an
        invariant-replay trace on disk per
        ``docs/proposals/replay_trace_format.md``.

        Composes any caller-supplied ``state_log`` / ``content_audit_log``
        callbacks with the file-writing callbacks; never overrides them.

        ``content_audit_log`` is ALWAYS passed to ``CCSStore.__init__``
        (even when ``streams={'state_log'}``) because
        ``retain_versions=content_audit_log is not None`` (init line 88)
        is required by the other three invariants. The opt-out only
        suppresses the on-disk JSONL writer.

        CCSStore is the verified adapter in v1, so this wrapper sets
        ``accept_unverified=True`` automatically — no
        ``UnverifiedAdapterCaptureError`` and no stderr warning. The
        opt-in gate is on ``record_callbacks`` for direct (CrewAI /
        AutoGen) callers.
        """
        # Import inside the method body — keeps the optional replay
        # surface out of the CCSStore import path for installs that
        # don't use it. (Same-layer import; no architecture violation.)
        from ccs.replay.recorder import record_callbacks

        caller_state_log = ccsstore_kwargs.pop("state_log", None)
        caller_audit_log = ccsstore_kwargs.pop("content_audit_log", None)
        # accept_unverified is enforced by CCSStore (verified adapter).
        # If a caller passes it here, drop it — the wrapper owns the flag.
        ccsstore_kwargs.pop("accept_unverified", None)

        # Mutable cell so the drain closure can see the store that
        # only exists after record_callbacks has opened the session.
        store_cell: list["CCSStore"] = []

        def _drain() -> tuple[dict[str, str], dict[str, str]]:
            if not store_cell:
                return {}, {}
            return _drain_store_registries(store_cell[0])

        with record_callbacks(
            path,
            streams=streams,
            accept_unverified=True,
            state_log=caller_state_log,
            content_audit_log=caller_audit_log,
            adapter_type="langgraph-ccsstore",
            drain_registries=_drain,
            _verified_caller=True,
        ) as (state_cb, audit_cb):
            store = cls(
                state_log=state_cb,
                content_audit_log=audit_cb,
                **ccsstore_kwargs,
            )
            store_cell.append(store)
            yield store

    # ------------------------------------------------------------------
    # BaseStore interface — batch is the single implementation point
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Any]) -> list[Result]:
        with self._lock:
            self._tick += 1
            tick = self._tick
            results: list[Any] = []
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._apply_get(op, tick))
                elif isinstance(op, PutOp):
                    if op.value is None:
                        self._apply_delete(op.namespace, op.key, tick)
                    else:
                        self._apply_put(op, tick)
                    results.append(None)
                elif isinstance(op, SearchOp):
                    results.append(self._apply_search(op, tick))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._apply_list_namespaces(op))
                else:
                    results.append(None)
            return results

    async def abatch(self, ops: Iterable[Any]) -> list[Result]:
        return await asyncio.to_thread(self.batch, ops)

    # ------------------------------------------------------------------
    # Op handlers
    # ------------------------------------------------------------------

    def _apply_get(self, op: GetOp, tick: int) -> Item | None:
        namespace = op.namespace
        key = op.key
        if len(namespace) < 2:
            raise ValueError(
                f"CCSStore requires namespace with at least 2 elements; got {namespace!r}"
            )

        scope_key = (tuple(namespace[1:]), key)
        if scope_key not in self._artifact_map:
            return None

        artifact_id = self._artifact_map[scope_key]
        agent_name = namespace[0]
        self._ensure_agent_registered(agent_name)

        entry = self.core.runtime(agent_name).cache.get(artifact_id)
        was_cache_hit = entry is not None and entry.state in (
            MESIState.SHARED,
            MESIState.EXCLUSIVE,
            MESIState.MODIFIED,
        )

        degraded = False
        degraded_warning: str | None = None
        try:
            resp = self.core.read(agent_name=agent_name, artifact_id=artifact_id, now_tick=tick)
            value = json.loads(resp.content) if resp.content else {}
            cache_hit = was_cache_hit
        except CoherenceError as exc:
            if self._on_error == "strict":
                raise
            logger.warning(
                "CCSStore: coherence error on get %r %r — degrading to fallback: %s",
                namespace, key, exc,
            )
            scope_key = (tuple(namespace[1:]), key)
            value = self._fallback_store.get(scope_key, {})
            degraded = True
            cache_hit = False
            self._degradation_count += 1
            if self._degradation_count == 1:
                degraded_warning = f"CCSStore degraded to fallback on get {namespace!r} {key!r}: {exc}"

        tokens = 1 if cache_hit else self._estimate_tokens(value)
        # Cache hit: no content was fetched from the coordinator, so transmission cost is 0.
        # We emit max(1, 0) = 1 to acknowledge the access without counting redundant transfer.
        tokens_saved = max(0, self._estimate_tokens(value) - 1) if cache_hit else 0
        self._emit_metric(
            operation="degraded" if degraded else "get",
            namespace=namespace,
            key=key,
            agent_name=agent_name,
            tokens_consumed=tokens,
            cache_hit=cache_hit,
            tokens_saved_estimate=tokens_saved,
        )

        if self._bm_baseline is not None:
            self._bm_baseline += self._estimate_tokens(value)
            self._bm_actual += tokens  # type: ignore[operator]
            self._bm_ops += 1  # type: ignore[operator]
            if cache_hit:
                self._bm_hits += 1  # type: ignore[operator]

        # Warn last: a caller that escalates the warning to an error makes warn()
        # raise, and none of the degradation's bookkeeping may depend on it.
        if degraded_warning is not None:
            warnings.warn(degraded_warning, CoherenceDegradedWarning, stacklevel=4)

        now = datetime.now(tz=timezone.utc)
        return Item(value=value, key=key, namespace=namespace, created_at=now, updated_at=now)

    def _apply_put(self, op: PutOp, tick: int) -> None:
        namespace = op.namespace
        key = op.key
        value = op.value
        if len(namespace) < 2:
            raise ValueError(
                f"CCSStore requires namespace with at least 2 elements; got {namespace!r}"
            )

        if op.ttl is not None:
            warnings.warn(
                "CCSStore does not enforce ttl in v0; the value will be stored without expiry.",
                UserWarning,
                stacklevel=2,
            )

        content_str = json.dumps(value, sort_keys=True, separators=(",", ":"))
        agent_name = namespace[0]

        self._ensure_agent_registered(agent_name)
        artifact_id = self._ensure_artifact_registered(namespace, key)

        degraded = False
        degraded_warning: str | None = None
        try:
            self.core.write(
                agent_name=agent_name, artifact_id=artifact_id, content=content_str, now_tick=tick
            )
        except CoherenceError as exc:
            if self._on_error == "strict":
                raise
            logger.warning(
                "CCSStore: coherence error on put %r %r — degrading to fallback: %s",
                namespace, key, exc,
            )
            scope_key = (tuple(namespace[1:]), key)
            self._fallback_store[scope_key] = value
            degraded = True
            self._degradation_count += 1
            if self._degradation_count == 1:
                degraded_warning = f"CCSStore degraded to fallback on put {namespace!r} {key!r}: {exc}"

        full_ns = tuple(namespace)
        if full_ns not in self._namespace_index:
            self._namespace_index[full_ns] = set()
        self._namespace_index[full_ns].add(key)

        self._emit_metric(
            operation="degraded" if degraded else "put",
            namespace=namespace,
            key=key,
            agent_name=agent_name,
            tokens_consumed=self._estimate_tokens(value),
            cache_hit=False,
        )

        # Warn last, as in _apply_get.
        if degraded_warning is not None:
            warnings.warn(degraded_warning, CoherenceDegradedWarning, stacklevel=4)

    def _apply_delete(self, namespace: tuple[str, ...], key: str, tick: int) -> None:
        if len(namespace) < 2:
            raise ValueError(
                f"CCSStore requires namespace with at least 2 elements; got {namespace!r}"
            )

        scope_key = (tuple(namespace[1:]), key)
        if scope_key not in self._artifact_map:
            return

        agent_name = namespace[0]
        artifact_id = self._artifact_map[scope_key]

        self._ensure_agent_registered(agent_name)
        caller_agent_id = self.core.agent_id_for(agent_name)

        signals = self.core.coordinator.delete(
            agent_id=caller_agent_id,
            artifact_id=artifact_id,
            issued_at_tick=tick,
        )

        if signals:
            peers = [
                self.core.agent_id_for(n)
                for n in self.core.agent_names()
                if n != agent_name
            ]
            for signal in signals:
                self.core.event_bus.publish_invalidation(signal, recipients=peers)

        self._deleted_ids.add(artifact_id)
        self._artifact_map.pop(scope_key, None)

        full_ns = tuple(namespace)
        if full_ns in self._namespace_index:
            self._namespace_index[full_ns].discard(key)
            if not self._namespace_index[full_ns]:
                del self._namespace_index[full_ns]

    def _apply_search(self, op: SearchOp, tick: int) -> list[SearchItem]:
        # Caller holds self._lock — _apply_search is only reachable through batch/abatch.
        if op.query is not None:
            warnings.warn(
                "CCSStore does not support semantic search in v0; query= is ignored.",
                UserWarning,
                stacklevel=2,
            )

        prefix = op.namespace_prefix
        results: list[SearchItem] = []

        for full_ns, keys in list(self._namespace_index.items()):
            if not _ns_starts_with(full_ns, prefix):
                continue
            scope = tuple(full_ns[1:])
            for key in list(keys):
                artifact_id = self._artifact_map.get((scope, key))
                if artifact_id is None:
                    continue
                raw = self.core.registry.get_content(artifact_id)
                if raw is None:
                    continue
                value = json.loads(raw)

                if op.filter and not _matches_filter(value, op.filter):
                    continue

                now = datetime.now(tz=timezone.utc)
                results.append(
                    SearchItem(
                        namespace=full_ns,
                        key=key,
                        value=value,
                        created_at=now,
                        updated_at=now,
                    )
                )

                self._emit_metric(
                    operation="search.hit",
                    namespace=full_ns,
                    key=key,
                    agent_name=full_ns[0] if full_ns else "",
                    tokens_consumed=self._estimate_tokens(value),
                    cache_hit=False,
                )

                if self._content_audit_log is not None:
                    artifact_meta = self.core.registry.get_artifact(artifact_id)
                    record_version = artifact_meta.version if artifact_meta else None
                    content_hash = compute_content_hash(raw) if raw is not None else None
                    self._audit_seq[0] += 1
                    try:
                        self._content_audit_log({
                            "tick": tick,
                            "agent_id": None,
                            "agent_name": None,
                            "artifact_id": str(artifact_id),
                            "version": record_version,
                            "content_hash": content_hash,
                            "source": "search",
                            "outcome": "error",
                            "sequence_number": self._audit_seq[0],
                            "instance_id": self._instance_id,
                            "schema_version": CCS_CONTENT_AUDIT_LOG_SCHEMA_VERSION,
                        })
                    except Exception:
                        self._audit_seq[0] -= 1
                        raise

        return results[op.offset : op.offset + op.limit]

    def _apply_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        namespaces: list[tuple[str, ...]] = list(self._namespace_index.keys())

        conditions = op.match_conditions or ()
        if conditions:
            namespaces = [
                ns for ns in namespaces if all(_matches_condition(ns, c) for c in conditions)
            ]

        if op.max_depth is not None:
            namespaces = list({ns[: op.max_depth] for ns in namespaces})

        return namespaces[op.offset : op.offset + op.limit]

    # ------------------------------------------------------------------
    # Crash recovery — public heartbeat / recover surface
    # ------------------------------------------------------------------

    def heartbeat(self, *, agent_name: str, now_tick: int) -> None:
        """Record a heartbeat for the named agent. No-op when crash recovery is disabled."""
        self.core.heartbeat(agent_name=agent_name, now_tick=now_tick)

    def recover(self, *, agent_name: str, now_tick: int) -> None:
        """Invalidate agent's local cache and record a recovery heartbeat."""
        self.core.recover(agent_name=agent_name, now_tick=now_tick)

    # ------------------------------------------------------------------
    # Degradation visibility (R8 additions)
    # ------------------------------------------------------------------

    @property
    def is_degraded(self) -> bool:
        """True if at least one coherence error has occurred on this instance."""
        return self._degradation_count > 0

    @property
    def degradation_count(self) -> int:
        """Number of coherence errors that degraded to fallback on this instance."""
        return self._degradation_count

    # ------------------------------------------------------------------
    # Inline benchmark mode (R9)
    # ------------------------------------------------------------------

    def benchmark_summary(self) -> dict[str, float | int]:
        """Return token-savings summary for this store instance.

        Raises RuntimeError if the store was not created with benchmark=True.
        """
        if self._bm_baseline is None:
            raise RuntimeError("CCSStore was not created with benchmark=True")
        baseline = self._bm_baseline
        actual = self._bm_actual  # type: ignore[assignment]
        ops = self._bm_ops  # type: ignore[assignment]
        hits = self._bm_hits  # type: ignore[assignment]
        tokens_saved = baseline - actual
        return {
            "baseline_tokens": baseline,
            "ccs_tokens": actual,
            "tokens_saved": tokens_saved,
            "token_reduction_pct": round(tokens_saved / baseline * 100, 1) if baseline > 0 else 0.0,
            "cache_hit_rate": round(hits / ops, 3) if ops > 0 else 0.0,
            "n_operations": ops,
        }

    def print_benchmark_summary(self) -> None:
        """Print a formatted benchmark summary table to stdout."""
        s = self.benchmark_summary()
        baseline = s["baseline_tokens"]
        actual = s["ccs_tokens"]
        saved = s["tokens_saved"]
        ops = s["n_operations"]
        hit_rate = s["cache_hit_rate"]
        reduction = s["token_reduction_pct"]
        print()
        print("  CCSStore Benchmark Summary")
        print(f"  {'─' * 38}")
        print(f"  Baseline tokens (no cache):  {baseline:>8}")
        print(f"  CCSStore tokens:             {actual:>8}")
        print(f"  Tokens saved:                {saved:>8}")
        print(f"  Token reduction:             {reduction:>7.1f}%")
        print(f"  Cache hit rate:              {hit_rate:>7.1%}  ({ops} get ops)")

    # ------------------------------------------------------------------
    # Metric emission helper (must only be called inside self._lock)
    # ------------------------------------------------------------------

    def _emit_metric(
        self,
        *,
        operation: str,
        namespace: tuple[str, ...],
        key: str,
        agent_name: str,
        tokens_consumed: int,
        cache_hit: bool,
        tokens_saved_estimate: int = 0,
        tick: int | None = None,
    ) -> None:
        # All callers are inside batch() which holds self._lock. Do not acquire
        # the lock here — a nested acquire would deadlock. Any future emission
        # site added outside the lock would silently break gap detection.
        self._metric_seq += 1
        event = StoreMetricEvent(
            operation=operation,
            namespace=namespace,
            key=key,
            agent_name=agent_name,
            tokens_consumed=tokens_consumed,
            cache_hit=cache_hit,
            tick=tick if tick is not None else self._tick,
            tokens_saved_estimate=tokens_saved_estimate,
            sequence_number=self._metric_seq,
            instance_id=self._instance_id,
            schema_version=CCS_METRIC_SCHEMA_VERSION,
        )
        if event.sequence_number < 1:
            raise RuntimeError(
                f"_emit_metric produced invalid sequence_number={event.sequence_number!r}; "
                "this is a bug in CCSStore — was _emit_metric called outside self._lock?"
            )
        if self._on_metric is not None:
            self._on_metric(event)
        self._telemetry.on_event(event)

    # ------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------

    def _ensure_agent_registered(self, agent_name: str) -> None:
        if agent_name not in self._known_agents:
            self.core.register_agent(agent_name)
            self._known_agents.add(agent_name)

    def _ensure_artifact_registered(self, namespace: tuple[str, ...], key: str) -> uuid.UUID:
        scope = tuple(namespace[1:])
        scope_str = ":".join(scope)
        artifact_id = artifact_uuid(scope_str, key)
        scope_key = (scope, key)

        if scope_key in self._artifact_map:
            return artifact_id

        if artifact_id in self._deleted_ids:
            self._deleted_ids.discard(artifact_id)

        self.core.registry.register_artifact(
            Artifact(id=artifact_id, name=f"{scope_str}:{key}", version=1),
            "",
        )
        self._artifact_map[scope_key] = artifact_id
        return artifact_id

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(value: dict[str, Any]) -> int:
        if "__ccs_size_tokens__" in value:
            return int(value["__ccs_size_tokens__"])
        return max(1, len(json.dumps(value, sort_keys=True, separators=(",", ":"))) // 4)


def _drain_store_registries(store: CCSStore) -> tuple[dict[str, str], dict[str, str]]:
    """Snapshot agents + artifacts from a CCSStore for manifest finalization.

    Called by ``CCSStore.record_to``'s drain closure on session
    ``__exit__``. Returns ``(agents_by_uuid, artifacts_by_uuid)`` —
    UUID strings to display names. Empty dicts when no MESI activity
    has fired.

    Routes through the public ``CoherenceAdapterCore`` snapshot accessors
    (``agent_names_snapshot`` / ``artifact_names_snapshot``) so a rename
    inside the coordinator surfaces at import time, not at manifest-
    finalization time.
    """
    core = store.core
    agents = {str(uid): name for uid, name in core.agent_names_snapshot().items()}
    artifacts = {str(uid): name for uid, name in core.artifact_names_snapshot().items()}
    return agents, artifacts


def _ns_starts_with(ns: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return ns[: len(prefix)] == prefix


def _matches_filter(value: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
    for field, condition in filter_dict.items():
        if isinstance(condition, dict):
            for op_key, expected in condition.items():
                actual = value.get(field)
                if op_key == "$eq":
                    if actual != expected:
                        return False
                elif op_key == "$ne":
                    if actual == expected:
                        return False
                else:
                    raise NotImplementedError(
                        f"CCSStore search filter operator {op_key!r} is not supported in v0. "
                        "Only $eq and $ne are supported."
                    )
        else:
            if value.get(field) != condition:
                return False
    return True


def _matches_condition(ns: tuple[str, ...], cond: MatchCondition) -> bool:
    path = cond.path
    if "*" in path:
        raise NotImplementedError(
            "CCSStore does not support wildcard '*' in MatchCondition.path in v0."
        )
    if cond.match_type == "prefix":
        return ns[: len(path)] == path
    if cond.match_type == "suffix":
        return ns[len(ns) - len(path) :] == path
    return False
