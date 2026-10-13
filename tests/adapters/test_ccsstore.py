# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for CCSStore LangGraph BaseStore adapter."""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any
from unittest.mock import Mock, patch
from uuid import NAMESPACE_URL, uuid5

import pytest

pytest.importorskip("langgraph.store.base")

from langgraph.store.base import GetOp, PutOp, SearchOp

from ccs.adapters.ccsstore import CCSStore, StoreMetricEvent
from ccs.core.states import MESIState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store(strategy: str = "lazy", **kwargs: Any) -> CCSStore:
    return CCSStore(strategy=strategy, **kwargs)


def _put(store: CCSStore, namespace: tuple[str, ...], key: str, value: dict) -> None:
    store.batch([PutOp(namespace=namespace, key=key, value=value)])


def _get(store: CCSStore, namespace: tuple[str, ...], key: str):
    return store.batch([GetOp(namespace=namespace, key=key)])[0]


def _delete(store: CCSStore, namespace: tuple[str, ...], key: str) -> None:
    store.batch([PutOp(namespace=namespace, key=key, value=None)])


# ---------------------------------------------------------------------------
# Get
# ---------------------------------------------------------------------------

def test_get_after_put_same_agent_is_cache_hit() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    result = _get(store, ("planner", "shared"), "plan")

    assert result is not None
    assert result.value == {"v": 1}
    get_event = next(e for e in events if e.operation == "get")
    assert get_event.cache_hit is True


def test_get_after_peer_put_is_cache_miss() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    # First get by reviewer: MESI miss → fetch
    result = _get(store, ("reviewer", "shared"), "plan")

    assert result is not None
    assert result.value == {"v": 1}
    get_event = next(e for e in events if e.operation == "get")
    assert get_event.cache_hit is False


def test_get_unknown_key_returns_none() -> None:
    store = _store()
    result = _get(store, ("planner", "shared"), "nonexistent")
    assert result is None


def test_get_unknown_key_does_not_emit_metric() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _get(store, ("planner", "shared"), "nonexistent")
    assert not events


def test_get_short_namespace_raises_value_error() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.batch([GetOp(namespace=("planner",), key="plan")])


def test_get_lazy_registers_agent_on_first_access() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"x": 1})
    # reviewer has never been registered; get should trigger lazy registration
    result = _get(store, ("reviewer", "shared"), "plan")
    assert result is not None
    assert "reviewer" in store._known_agents


def test_get_two_agents_both_reach_shared_state() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"x": 1})
    _get(store, ("planner", "shared"), "plan")
    _get(store, ("reviewer", "shared"), "plan")

    artifact_id = uuid5(NAMESPACE_URL, "ccs-artifact:shared:plan")
    # Check the coordinator registry (ground truth), not the local cache.
    # The local cache may lag behind — the registry is always consistent.
    planner_id = store.core.agent_id_for("planner")
    reviewer_id = store.core.agent_id_for("reviewer")
    assert store.core.registry.get_agent_state(artifact_id, planner_id) == MESIState.SHARED
    assert store.core.registry.get_agent_state(artifact_id, reviewer_id) == MESIState.SHARED


def test_get_after_peer_write_is_invalidated_then_refreshed() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    # reviewer reads → SHARED
    _get(store, ("reviewer", "shared"), "plan")
    # planner writes new version → reviewer invalidated
    _put(store, ("planner", "shared"), "plan", {"v": 2})

    artifact_id = uuid5(NAMESPACE_URL, "ccs-artifact:shared:plan")
    reviewer_entry = store.core.runtime("reviewer").cache.get(artifact_id)
    assert reviewer_entry is not None
    assert reviewer_entry.state == MESIState.INVALID

    result = _get(store, ("reviewer", "shared"), "plan")
    assert result is not None
    assert result.value == {"v": 2}


# ---------------------------------------------------------------------------
# Put
# ---------------------------------------------------------------------------

def test_put_creates_artifact_on_first_write() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    result = _get(store, ("planner", "shared"), "plan")
    assert result is not None
    assert result.value == {"v": 1}


def test_put_second_write_increments_version() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _put(store, ("planner", "shared"), "plan", {"v": 2})
    result = _get(store, ("planner", "shared"), "plan")
    assert result is not None
    assert result.value == {"v": 2}


def test_put_short_namespace_raises_value_error() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.batch([PutOp(namespace=("planner",), key="plan", value={"x": 1})])


def test_put_ttl_non_none_emits_user_warning() -> None:
    store = _store()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        store.batch([PutOp(namespace=("planner", "shared"), key="plan", value={"v": 1}, ttl=60.0)])
    assert any(issubclass(warning.category, UserWarning) for warning in w)


def test_put_index_param_accepted_without_error() -> None:
    store = _store()
    store.batch([PutOp(namespace=("planner", "shared"), key="plan", value={"v": 1}, index=["v"])])


def test_put_triggers_invalidation_to_peers() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _get(store, ("reviewer", "shared"), "plan")

    artifact_id = uuid5(NAMESPACE_URL, "ccs-artifact:shared:plan")
    reviewer_before = store.core.runtime("reviewer").cache.get(artifact_id)
    assert reviewer_before is not None and reviewer_before.state == MESIState.SHARED

    _put(store, ("planner", "shared"), "plan", {"v": 2})

    reviewer_after = store.core.runtime("reviewer").cache.get(artifact_id)
    assert reviewer_after is not None and reviewer_after.state == MESIState.INVALID


def test_put_emits_metric_event() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    put_events = [e for e in events if e.operation == "put"]
    assert len(put_events) == 1
    assert put_events[0].agent_name == "planner"
    assert put_events[0].cache_hit is False


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------

def test_delete_makes_subsequent_get_return_none() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _delete(store, ("planner", "shared"), "plan")
    assert _get(store, ("planner", "shared"), "plan") is None


def test_delete_absent_key_is_silent_no_op() -> None:
    store = _store()
    _delete(store, ("planner", "shared"), "nonexistent")  # must not raise


def test_delete_then_put_same_key_re_registers_successfully() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _delete(store, ("planner", "shared"), "plan")
    _put(store, ("planner", "shared"), "plan", {"v": 2})
    result = _get(store, ("planner", "shared"), "plan")
    assert result is not None
    assert result.value == {"v": 2}


def test_delete_sends_invalidation_to_peers_no_coherence_error() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _get(store, ("reviewer", "shared"), "plan")  # reviewer caches it

    # delete must propagate to reviewer without raising CoherenceError
    _delete(store, ("planner", "shared"), "plan")

    # reviewer get returns None (artifact gone)
    assert _get(store, ("reviewer", "shared"), "plan") is None


def test_delete_does_not_emit_metric_event() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    events.clear()
    _delete(store, ("planner", "shared"), "plan")
    assert not events


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search_by_agent_prefix_scopes_correctly() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"status": "active"})
    _put(store, ("reviewer", "shared"), "notes", {"status": "draft"})

    results = store.search(("planner",))
    keys = [r.key for r in results]
    assert "plan" in keys
    assert "notes" not in keys


def test_search_filter_primitive_eq() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "a", {"status": "active"})
    _put(store, ("planner", "shared"), "b", {"status": "draft"})

    results = store.search(("planner",), filter={"status": "active"})
    assert len(results) == 1
    assert results[0].key == "a"


def test_search_filter_explicit_ne() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "a", {"status": "active"})
    _put(store, ("planner", "shared"), "b", {"status": "draft"})

    results = store.search(("planner",), filter={"status": {"$ne": "draft"}})
    assert len(results) == 1
    assert results[0].key == "a"


def test_search_filter_explicit_eq_no_match_excluded() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "a", {"status": "active"})
    _put(store, ("planner", "shared"), "b", {"status": "draft"})
    results = store.search(("planner",), filter={"status": {"$eq": "active"}})
    assert len(results) == 1
    assert results[0].key == "a"


def test_search_filter_unsupported_operator_raises() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "a", {"score": 5})
    with pytest.raises(NotImplementedError):
        store.search(("planner",), filter={"score": {"$gt": 3}})


def test_search_query_param_emits_warning_returns_results() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "a", {"x": 1})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        results = store.search(("planner",), query="something")
    assert any(issubclass(warning.category, UserWarning) for warning in w)
    assert len(results) == 1  # still returns results, just unranked


def test_search_does_not_change_mesi_state() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    artifact_id = uuid5(NAMESPACE_URL, "ccs-artifact:shared:plan")

    # planner holds MODIFIED after put
    planner_entry_before = store.core.runtime("planner").cache.get(artifact_id)
    state_before = planner_entry_before.state if planner_entry_before else None

    store.search(("planner",))

    planner_entry_after = store.core.runtime("planner").cache.get(artifact_id)
    state_after = planner_entry_after.state if planner_entry_after else None
    assert state_before == state_after


def test_search_before_any_writes_returns_empty() -> None:
    store = _store()
    assert store.search(("planner",)) == []


def test_search_emits_one_metric_per_result() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "a", {"x": 1})
    _put(store, ("planner", "shared"), "b", {"x": 2})
    events.clear()
    store.search(("planner",))
    search_events = [e for e in events if e.operation == "search.hit"]
    assert len(search_events) == 2


# ---------------------------------------------------------------------------
# List namespaces
# ---------------------------------------------------------------------------

def test_list_namespaces_no_writes_returns_empty() -> None:
    store = _store()
    assert store.list_namespaces(prefix=()) == []


def test_list_namespaces_after_two_puts_shows_both() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _put(store, ("reviewer", "notes"), "review", {"v": 1})
    namespaces = store.list_namespaces(prefix=())
    assert ("planner", "shared") in namespaces
    assert ("reviewer", "notes") in namespaces


def test_list_namespaces_prefix_condition_filters() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _put(store, ("reviewer", "notes"), "review", {"v": 1})
    results = store.list_namespaces(prefix=("planner",))
    assert ("planner", "shared") in results
    assert ("reviewer", "notes") not in results


def test_list_namespaces_suffix_condition_filters() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _put(store, ("reviewer", "shared"), "notes", {"v": 1})
    _put(store, ("executor", "private"), "task", {"v": 1})
    results = store.list_namespaces(suffix=("shared",))
    assert ("planner", "shared") in results
    assert ("reviewer", "shared") in results
    assert ("executor", "private") not in results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_on_metric_none_default_no_error() -> None:
    store = _store()  # on_metric=None
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _get(store, ("planner", "shared"), "plan")  # no error


def test_metric_get_cache_hit_true() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    events.clear()
    _get(store, ("planner", "shared"), "plan")
    assert events[0].cache_hit is True
    assert events[0].operation == "get"


def test_metric_get_cache_miss_false() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    events.clear()
    _get(store, ("reviewer", "shared"), "plan")
    assert events[0].cache_hit is False


def test_metric_tokens_consumed_estimation_put() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    value = {"content": "x" * 400}
    _put(store, ("planner", "shared"), "plan", value)
    put_event = next(e for e in events if e.operation == "put")
    expected = max(1, len(json.dumps(value, sort_keys=True, separators=(",", ":"))) // 4)
    assert put_event.tokens_consumed == expected


def test_metric_tokens_consumed_cache_miss_is_full_size() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    value = {"content": "x" * 400}
    _put(store, ("planner", "shared"), "plan", value)
    events.clear()
    _get(store, ("reviewer", "shared"), "plan")  # cache miss
    get_event = events[0]
    expected = max(1, len(json.dumps(value, sort_keys=True, separators=(",", ":"))) // 4)
    assert get_event.tokens_consumed == expected


def test_metric_tokens_consumed_cache_hit_is_one() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"content": "x" * 400})
    events.clear()
    _get(store, ("planner", "shared"), "plan")  # cache hit (same agent that wrote)
    get_event = events[0]
    assert get_event.tokens_consumed == 1


def test_metric_custom_size_override() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"data": "x", "__ccs_size_tokens__": 42})
    put_event = next(e for e in events if e.operation == "put")
    assert put_event.tokens_consumed == 42


# ---------------------------------------------------------------------------
# Tick and threading
# ---------------------------------------------------------------------------

def test_tick_starts_at_zero_increments_per_batch() -> None:
    store = _store()
    assert store._tick == 0
    store.batch([PutOp(namespace=("planner", "shared"), key="a", value={"x": 1})])
    assert store._tick == 1
    store.batch([PutOp(namespace=("planner", "shared"), key="b", value={"x": 2})])
    assert store._tick == 2


def test_tick_appears_in_metric_event() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    store.batch([PutOp(namespace=("planner", "shared"), key="a", value={"x": 1})])
    assert events[0].tick == 1


def test_abatch_returns_same_as_batch() -> None:
    import asyncio

    store = _store()
    ops = [PutOp(namespace=("planner", "shared"), key="a", value={"x": 1})]
    batch_result = store.batch(ops)

    store2 = _store()
    abatch_result = asyncio.run(store2.abatch(ops))
    # Both return [None] for a single PutOp
    assert batch_result == abatch_result


# ---------------------------------------------------------------------------
# Full batch dispatch
# ---------------------------------------------------------------------------

def test_batch_mixed_ops_returns_results_in_order() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "a", {"v": 1})
    ops = [
        GetOp(namespace=("planner", "shared"), key="a"),
        PutOp(namespace=("planner", "shared"), key="b", value={"v": 2}),
        SearchOp(namespace_prefix=("planner",)),
    ]
    results = store.batch(ops)
    assert len(results) == 3
    assert results[0] is not None and results[0].value == {"v": 1}  # Item from Get
    assert results[1] is None  # Put returns None
    assert isinstance(results[2], list)  # Search returns list


def test_batch_value_error_on_short_namespace_propagates() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.batch([GetOp(namespace=("only_one",), key="k")])


def test_list_namespaces_max_depth_truncates_and_deduplicates() -> None:
    store = _store()
    _put(store, ("planner", "shared", "plans"), "a", {"v": 1})
    _put(store, ("planner", "shared", "plans"), "b", {"v": 2})
    _put(store, ("planner", "private"), "c", {"v": 3})
    # max_depth=2 → unique tuples of length 2
    results = store.list_namespaces(prefix=(), max_depth=2)
    assert ("planner", "shared") in results
    assert ("planner", "private") in results
    # original 3-element namespace should NOT appear
    assert ("planner", "shared", "plans") not in results


def test_delete_short_namespace_raises_value_error() -> None:
    store = _store()
    with pytest.raises(ValueError):
        store.batch([PutOp(namespace=("planner",), key="plan", value=None)])


def test_re_registration_after_delete_clears_deleted_ids() -> None:
    store = _store()
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _delete(store, ("planner", "shared"), "plan")

    from uuid import NAMESPACE_URL, uuid5
    artifact_id = uuid5(NAMESPACE_URL, "ccs-artifact:shared:plan")
    assert artifact_id in store._deleted_ids

    _put(store, ("planner", "shared"), "plan", {"v": 2})
    assert artifact_id not in store._deleted_ids


def test_metric_events_ordered_by_operation_sequence() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    ops = [
        PutOp(namespace=("planner", "shared"), key="a", value={"x": 1}),
        GetOp(namespace=("planner", "shared"), key="a"),
    ]
    store.batch(ops)
    assert events[0].operation == "put"
    assert events[1].operation == "get"


# ---------------------------------------------------------------------------
# Telemetry parameter
# ---------------------------------------------------------------------------

def test_ccsstore_default_has_noop_telemetry() -> None:
    from ccs.adapters.telemetry import NoOpTelemetryExporter
    store = _store()
    assert isinstance(store._telemetry, NoOpTelemetryExporter)


def test_ccsstore_telemetry_none_has_noop() -> None:
    from ccs.adapters.telemetry import NoOpTelemetryExporter
    store = _store(telemetry=None)
    assert isinstance(store._telemetry, NoOpTelemetryExporter)


def test_ccsstore_telemetry_exporter_receives_put_event() -> None:
    from ccs.adapters.telemetry import TelemetryExporter

    class CapturingExporter(TelemetryExporter):
        def __init__(self):
            self.events: list[StoreMetricEvent] = []
        def on_event(self, event: StoreMetricEvent) -> None:
            self.events.append(event)

    exporter = CapturingExporter()
    store = _store(telemetry=exporter)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    assert len(exporter.events) == 1
    assert exporter.events[0].operation == "put"


def test_ccsstore_telemetry_exporter_receives_get_event() -> None:
    from ccs.adapters.telemetry import TelemetryExporter

    class CapturingExporter(TelemetryExporter):
        def __init__(self):
            self.events: list[StoreMetricEvent] = []
        def on_event(self, event: StoreMetricEvent) -> None:
            self.events.append(event)

    exporter = CapturingExporter()
    store = _store(telemetry=exporter)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    exporter.events.clear()
    _get(store, ("planner", "shared"), "plan")
    assert len(exporter.events) == 1
    assert exporter.events[0].operation == "get"


def test_ccsstore_on_metric_and_telemetry_both_called() -> None:
    from ccs.adapters.telemetry import TelemetryExporter

    on_metric_events: list[StoreMetricEvent] = []

    class CapturingExporter(TelemetryExporter):
        def __init__(self):
            self.events: list[StoreMetricEvent] = []
        def on_event(self, event: StoreMetricEvent) -> None:
            self.events.append(event)

    exporter = CapturingExporter()
    store = _store(on_metric=on_metric_events.append, telemetry=exporter)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    assert len(on_metric_events) == 1
    assert len(exporter.events) == 1
    # Both receive the same event object
    assert on_metric_events[0] is exporter.events[0]


def test_ccsstore_telemetry_and_on_metric_none_no_error() -> None:
    store = _store(telemetry=None, on_metric=None)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    result = _get(store, ("planner", "shared"), "plan")
    assert result is not None


# ---------------------------------------------------------------------------
# Graceful degradation (on_error parameter)
# ---------------------------------------------------------------------------

def test_on_error_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="on_error"):
        CCSStore(on_error="bad")


def test_on_error_strict_is_default() -> None:
    store = _store()
    assert store._on_error == "strict"


def test_on_error_strict_reraises_coherence_error_on_get() -> None:
    from unittest.mock import patch

    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="strict")
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
        with pytest.raises(CoherenceError):
            _get(store, ("reviewer", "shared"), "plan")  # different agent → cache miss → core.read


def test_on_error_strict_reraises_coherence_error_on_put() -> None:
    from unittest.mock import patch

    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="strict")
    with patch.object(store.core, "write", side_effect=CoherenceError("simulated")):
        with pytest.raises(CoherenceError):
            _put(store, ("planner", "shared"), "plan", {"v": 1})


def test_on_error_degrade_put_emits_degraded_event() -> None:
    from unittest.mock import patch

    from ccs.core.exceptions import CoherenceError

    events: list[StoreMetricEvent] = []
    store = _store(on_error="degrade", on_metric=events.append)
    with patch.object(store.core, "write", side_effect=CoherenceError("simulated")):
        _put(store, ("planner", "shared"), "plan", {"v": 1})
    assert events[0].operation == "degraded"


def test_on_error_degrade_get_emits_degraded_event() -> None:
    from unittest.mock import patch

    from ccs.core.exceptions import CoherenceError

    events: list[StoreMetricEvent] = []
    store = _store(on_error="degrade", on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    events.clear()
    with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
        _get(store, ("reviewer", "shared"), "plan")  # reviewer cache miss → core.read → degrade
    assert events[0].operation == "degraded"


def test_on_error_degrade_get_returns_fallback_value() -> None:
    from unittest.mock import patch

    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="degrade")
    # Degraded put — value lands in _fallback_store
    with patch.object(store.core, "write", side_effect=CoherenceError("simulated")):
        _put(store, ("planner", "shared"), "plan", {"v": 42})
    # Degraded get from same scope — retrieves from _fallback_store
    with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
        result = _get(store, ("reviewer", "shared"), "plan")
    assert result is not None
    assert result.value == {"v": 42}


def test_on_error_degrade_does_not_raise_on_coherence_error() -> None:
    from unittest.mock import patch

    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="degrade")
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    # Neither put nor get should raise when on_error="degrade"
    with patch.object(store.core, "write", side_effect=CoherenceError("simulated")):
        _put(store, ("planner", "shared"), "plan", {"v": 2})  # no raise
    with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
        _get(store, ("reviewer", "shared"), "plan")  # no raise


# ---------------------------------------------------------------------------
# CoherenceDegradedWarning + degradation visibility (R8 additions)
# ---------------------------------------------------------------------------

def test_is_degraded_false_before_any_error() -> None:
    store = _store(on_error="degrade")
    assert store.is_degraded is False


def test_is_degraded_true_after_degraded_get() -> None:
    import warnings

    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="degrade")
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
            _get(store, ("reviewer", "shared"), "plan")
    assert store.is_degraded is True


def test_is_degraded_true_after_degraded_put() -> None:
    import warnings

    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="degrade")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch.object(store.core, "write", side_effect=CoherenceError("simulated")):
            _put(store, ("planner", "shared"), "plan", {"v": 1})
    assert store.is_degraded is True


def test_degradation_count_increments_per_error() -> None:
    import warnings

    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="degrade")
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
            _get(store, ("reviewer", "shared"), "plan")
            _get(store, ("reviewer", "shared"), "plan")
    assert store.degradation_count == 2


def test_degraded_warning_emitted_on_first_degradation() -> None:
    from ccs.adapters.ccsstore import CoherenceDegradedWarning
    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="degrade")
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    with pytest.warns(CoherenceDegradedWarning):
        with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
            _get(store, ("reviewer", "shared"), "plan")


def test_degraded_warning_not_emitted_on_second_degradation() -> None:
    import warnings

    from ccs.adapters.ccsstore import CoherenceDegradedWarning
    from ccs.core.exceptions import CoherenceError

    store = _store(on_error="degrade")
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    # First degradation fires the warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
            _get(store, ("reviewer", "shared"), "plan")
    # Second degradation must NOT fire CoherenceDegradedWarning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with patch.object(store.core, "read", side_effect=CoherenceError("simulated")):
            _get(store, ("reviewer", "shared"), "plan")
    degraded_warnings = [x for x in w if issubclass(x.category, CoherenceDegradedWarning)]
    assert len(degraded_warnings) == 0


@pytest.mark.parametrize("operation", ["get", "put"])
def test_degradation_is_recorded_when_the_degraded_warning_is_escalated(operation: str) -> None:
    # A caller that escalates CoherenceDegradedWarning to an error makes warn()
    # raise. Everything the degraded operation records must already be done by
    # then: otherwise is_degraded reports False for a store that fell back, the
    # metric stream and the namespace index miss the operation, and because the
    # warn-once latch never closes, every later failure raises instead of
    # serving the fallback.
    from ccs.adapters.ccsstore import CoherenceDegradedWarning
    from ccs.core.exceptions import CoherenceError

    events: list[StoreMetricEvent] = []
    store = _store(on_error="degrade", on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    if operation == "get":
        core_call = "read"
        def degrade() -> None:
            _get(store, ("reviewer", "shared"), "plan")
    else:
        core_call = "write"
        def degrade() -> None:
            _put(store, ("planner", "drafts"), "plan", {"v": 2})

    with patch.object(store.core, core_call, side_effect=CoherenceError("simulated")):
        with warnings.catch_warnings():
            warnings.simplefilter("error", CoherenceDegradedWarning)
            with pytest.raises(CoherenceDegradedWarning):
                degrade()
            assert store.is_degraded is True
            assert [e.operation for e in events].count("degraded") == 1
            if operation == "put":
                assert ("planner", "drafts") in store.list_namespaces()
            degrade()  # latch closed: the next failure falls back without raising
    assert store.degradation_count == 2


def test_coherence_degraded_warning_importable_from_adapters() -> None:
    from ccs.adapters import CoherenceDegradedWarning
    assert issubclass(CoherenceDegradedWarning, UserWarning)


# ---------------------------------------------------------------------------
# Unit 1: schema version constants and StoreMetricEvent new fields
# ---------------------------------------------------------------------------

def test_schema_version_constants_importable() -> None:
    from ccs.adapters.events import CCS_METRIC_SCHEMA_VERSION
    from ccs.coordinator.registry import CCS_STATE_LOG_SCHEMA_VERSION
    assert CCS_STATE_LOG_SCHEMA_VERSION == "ccs.state_log.v2"
    assert CCS_METRIC_SCHEMA_VERSION == "ccs.metric.v1"


def test_store_metric_event_new_fields_default_to_none() -> None:
    event = StoreMetricEvent(
        operation="get",
        namespace=("a", "b"),
        key="k",
        agent_name="a",
        tokens_consumed=1,
        cache_hit=False,
        tick=1,
    )
    assert event.sequence_number is None
    assert event.instance_id is None
    assert event.schema_version is None


def test_store_metric_event_accepts_new_fields_as_kwargs() -> None:
    event = StoreMetricEvent(
        operation="get",
        namespace=("a", "b"),
        key="k",
        agent_name="a",
        tokens_consumed=1,
        cache_hit=False,
        tick=1,
        sequence_number=3,
        instance_id="abc-123",
        schema_version="ccs.metric.v1",
    )
    assert event.sequence_number == 3
    assert event.instance_id == "abc-123"
    assert event.schema_version == "ccs.metric.v1"


# ---------------------------------------------------------------------------
# Unit 3: CCSStore instance_id, metric sequence counter, _emit_metric helper
# ---------------------------------------------------------------------------

def test_get_emits_sequence_number_1_then_2() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    events.clear()
    _get(store, ("planner", "shared"), "plan")
    _get(store, ("planner", "shared"), "plan")
    get_events = [e for e in events if e.operation == "get"]
    assert get_events[0].sequence_number == 2  # put was seq=1
    assert get_events[1].sequence_number == 3


def test_put_increments_metric_seq() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "a", {"v": 1})
    _put(store, ("planner", "shared"), "b", {"v": 2})
    put_events = [e for e in events if e.operation == "put"]
    assert put_events[0].sequence_number == 1
    assert put_events[1].sequence_number == 2


def test_instance_id_on_metric_events_matches_store_instance() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    assert events[0].instance_id == store._instance_id


def test_schema_version_on_metric_events() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    assert events[0].schema_version == "ccs.metric.v1"


def test_two_ccsstore_instances_have_distinct_instance_ids() -> None:
    store_a = _store()
    store_b = _store()
    assert store_a._instance_id != store_b._instance_id


def test_search_hit_increments_metric_seq_per_result() -> None:
    events: list[StoreMetricEvent] = []
    store = _store(on_metric=events.append)
    _put(store, ("planner", "shared"), "a", {"v": 1})
    _put(store, ("planner", "shared"), "b", {"v": 2})
    events.clear()
    store.batch([SearchOp(namespace_prefix=("planner",), filter=None, limit=10, offset=0)])
    search_events = [e for e in events if e.operation == "search.hit"]
    assert len(search_events) == 2
    seqs = [e.sequence_number for e in search_events]
    assert seqs[1] == seqs[0] + 1


def test_out_of_sequence_event_produces_gap_in_validate_log(tmp_path) -> None:
    """An event with sequence_number=0 triggers Gap(expected=1, found=0)."""
    import json

    from ccs.validation import Gap, validate_log
    bad_line = {
        "sequence_number": 0,
        "instance_id": "some-id",
        "schema_version": "ccs.metric.v1",
        "operation": "get",
    }
    log_file = tmp_path / "test.jsonl"
    log_file.write_text(json.dumps(bad_line) + "\n")
    gaps, _ = validate_log(log_file, stream="metrics")
    assert gaps == [Gap(stream="metrics", expected=1, found=0, at_index=0)]


def test_none_sequence_number_raises_value_error(tmp_path) -> None:
    """StoreMetricEvent with default None sequence_number raises ValueError in validate_log."""
    import json

    from ccs.validation import validate_log
    bad_line = {
        "sequence_number": None,
        "instance_id": "some-id",
        "schema_version": "ccs.metric.v1",
    }
    log_file = tmp_path / "test.jsonl"
    log_file.write_text(json.dumps(bad_line) + "\n")
    with pytest.raises(ValueError, match="sequence_number"):
        validate_log(log_file)


# ---------------------------------------------------------------------------
# Crash recovery: CCSStore heartbeat / recover (Unit 3)
# ---------------------------------------------------------------------------


def test_ccsstore_batch_piggyback_heartbeat() -> None:
    from ccs.coordinator.service import CrashRecoveryConfig

    store = CCSStore(
        strategy="lazy",
        crash_recovery=CrashRecoveryConfig(enabled=True, heartbeat_timeout_ticks=10, max_hold_ticks=1000),
    )
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _get(store, ("planner", "shared"), "plan")

    agent_id = store.core.agent_id_for("planner")
    assert store.core.registry.last_heartbeat_tick(agent_id) is not None


def test_ccsstore_explicit_heartbeat() -> None:
    from ccs.coordinator.service import CrashRecoveryConfig

    store = CCSStore(
        strategy="lazy",
        crash_recovery=CrashRecoveryConfig(enabled=True, heartbeat_timeout_ticks=10, max_hold_ticks=1000),
    )
    _put(store, ("planner", "shared"), "plan", {"v": 1})

    store.heartbeat(agent_name="planner", now_tick=42)

    agent_id = store.core.agent_id_for("planner")
    assert store.core.registry.last_heartbeat_tick(agent_id) == 42


def test_ccsstore_heartbeat_requires_now_tick() -> None:
    from ccs.coordinator.service import CrashRecoveryConfig

    store = CCSStore(
        strategy="lazy",
        crash_recovery=CrashRecoveryConfig(enabled=True, heartbeat_timeout_ticks=10, max_hold_ticks=1000),
    )
    _put(store, ("planner", "shared"), "plan", {"v": 1})

    with pytest.raises(TypeError):
        store.heartbeat(agent_name="planner")  # type: ignore[call-arg]


def test_ccsstore_recover_invalidates_and_heartbeats() -> None:
    from ccs.coordinator.service import CrashRecoveryConfig

    store = CCSStore(
        strategy="lazy",
        crash_recovery=CrashRecoveryConfig(enabled=True, heartbeat_timeout_ticks=10, max_hold_ticks=1000),
    )
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    _get(store, ("planner", "shared"), "plan")

    store.recover(agent_name="planner", now_tick=200)

    agent_id = store.core.agent_id_for("planner")
    runtime = store.core.runtime("planner")
    for entry in runtime.cache.entries().values():
        assert entry.state == MESIState.INVALID
    assert store.core.registry.last_heartbeat_tick(agent_id) == 200


def test_ccsstore_constructor_passthrough_failfast() -> None:
    from ccs.coordinator.service import CrashRecoveryConfig

    with pytest.raises(ValueError, match="composition violation"):
        CCSStore(
            strategy="lease",
            lease_ttl_ticks=300,
            crash_recovery=CrashRecoveryConfig(enabled=True, max_hold_ticks=300),
        )


def test_ccsstore_recover_checkpoint_restore_flow() -> None:
    from ccs.coordinator.service import CrashRecoveryConfig

    store = CCSStore(
        strategy="lazy",
        crash_recovery=CrashRecoveryConfig(enabled=True, heartbeat_timeout_ticks=10, max_hold_ticks=1000),
    )
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    agent_id = store.core.agent_id_for("planner")

    # Simulate stale grant reclamation
    store.core.coordinator.enforce_stable_grant_timeouts(
        current_tick=50, heartbeat_timeout_ticks=10, max_hold_ticks=1000,
    )

    # Recover after restart
    store.recover(agent_name="planner", now_tick=51)

    # Cache is cleared — next put should succeed (re-acquires grant)
    entry = store.core.runtime("planner").cache.entries()
    for e in entry.values():
        assert e.state == MESIState.INVALID

    # Fresh heartbeat recorded
    assert store.core.registry.last_heartbeat_tick(agent_id) == 51


def test_ccsstore_flag_off_heartbeat_noop() -> None:
    # Explicit enabled=False: post-v0.9.0 the bare default is enabled, so the
    # "flag off" contract requires opting out explicitly.
    from ccs.coordinator.service import CrashRecoveryConfig

    store = _store(crash_recovery=CrashRecoveryConfig(enabled=False))
    _put(store, ("planner", "shared"), "plan", {"v": 1})

    store.heartbeat(agent_name="planner", now_tick=42)

    agent_id = store.core.agent_id_for("planner")
    assert store.core.registry.last_heartbeat_tick(agent_id) is None


# ---- v0.9.0 C-flip — bare CCSStore adopts enabled-by-default ----------------
#
# See docs/plans/2026-05-28-001-feat-c-flip-crash-recovery-default-on-plan.md
# Units 5 + 6. Post-flip, bare CCSStore() (no crash_recovery= argument) adopts
# the enabled-by-default crash recovery. The v0.8.3 DeprecationWarning is gone;
# the only warning that may surface is the one-shot v0.9.0 transitional
# RuntimeWarning (suppressed suite-wide by the conftest neutralizer except in
# the dedicated tests/test_coordinator.py assertions).


def test_bare_ccsstore_adopts_enabled_default_no_deprecation_warning() -> None:
    """v0.9.0: bare CCSStore() emits no DeprecationWarning (the v0.8.3 warning is
    removed) and adopts the enabled-by-default crash recovery; at most one
    transitional RuntimeWarning may surface."""
    from ccs.coordinator import service as _service_mod

    # Reset the once-per-process flag so we deterministically observe the
    # transitional warning here (the conftest neutralizer otherwise pre-sets it).
    _service_mod._V090_FIRST_USE_WARNED = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        store = CCSStore(strategy="lazy")
    deprecation_warnings = [
        w for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert deprecation_warnings == [], (
        f"bare CCSStore construction must not emit DeprecationWarning, got: "
        f"{[str(w.message) for w in deprecation_warnings]}"
    )
    runtime_warnings = [
        w for w in caught if issubclass(w.category, RuntimeWarning)
    ]
    assert len(runtime_warnings) <= 1
    # The flip: bare construction now enables crash recovery.
    assert store.core._crash_recovery.enabled is True


# ---------------------------------------------------------------------------
# v0.9.0 Unit 9: CCSStore once-per-batch sweep (KD-9) + Unit 6 default-threshold
# false-reclaim regression. No production code changes — once-per-batch emerges
# from batch()'s shared per-batch tick + CoherenceAdapterCore's rate-limited
# _maybe_sweep (the sweep seam lives on store.core, not on CCSStore).
# ---------------------------------------------------------------------------


def _enabled_store(**overrides):
    from ccs.coordinator.service import CrashRecoveryConfig

    cfg = CrashRecoveryConfig(enabled=True, heartbeat_timeout_ticks=10, max_hold_ticks=1000)
    return _store(crash_recovery=cfg, **overrides)


def test_ccsstore_batch_sweeps_once_per_batch() -> None:
    """KD-9: a multi-op batch shares one tick, so the rate-limited sweep fires on
    the first op and skips the rest — exactly ONE coordinator sweep per batch."""
    store = _enabled_store()  # heartbeat_timeout=10 -> gate=5
    _put(store, ("planner", "shared"), "plan", {"v": 1})  # creates artifact; sweeps at tick 1
    # Spy AFTER setup so we count only the test batch's sweeps.
    store.core.coordinator.enforce_stable_grant_timeouts = Mock(return_value=0)
    store._tick = 100  # next batch runs at tick 101, well past the gate from tick 1
    store.batch([
        GetOp(namespace=("planner", "shared"), key="plan"),
        GetOp(namespace=("planner", "shared"), key="plan"),
        GetOp(namespace=("planner", "shared"), key="plan"),
    ])
    assert store.core.coordinator.enforce_stable_grant_timeouts.call_count == 1


def test_ccsstore_sweep_rate_limited_across_batches() -> None:
    """Per-batch rate-limit: consecutive batches at ticks 1, 2, 6 sweep on the
    first and third only (the second is within the gate=5 window)."""
    store = _enabled_store()  # gate = 5
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    store.core.coordinator.enforce_stable_grant_timeouts = Mock(return_value=0)
    # Reset the sweep gate so the batch timing below is exactly what we control.
    store.core._last_sweep_tick = None
    store._tick = 0
    _get(store, ("planner", "shared"), "plan")  # batch -> tick 1; first ever -> fire
    _get(store, ("planner", "shared"), "plan")  # batch -> tick 2; 2-1=1 < gate -> skip
    store._tick = 5
    _get(store, ("planner", "shared"), "plan")  # batch -> tick 6; 6-1=5 >= gate -> fire
    assert store.core.coordinator.enforce_stable_grant_timeouts.call_count == 2


def test_ccsstore_disabled_never_sweeps() -> None:
    from ccs.coordinator.service import CrashRecoveryConfig

    store = _store(crash_recovery=CrashRecoveryConfig(enabled=False))
    store.core.coordinator.enforce_stable_grant_timeouts = Mock(return_value=0)
    _put(store, ("planner", "shared"), "plan", {"v": 1})
    for _ in range(10):
        _get(store, ("planner", "shared"), "plan")
    store.core.coordinator.enforce_stable_grant_timeouts.assert_not_called()


def test_ccsstore_first_reclamation_diagnostic(caplog) -> None:
    """First reclamation through a CCSStore batch emits the diagnostic exactly
    once on the ``ccs.adapters.base`` logger; subsequent batches do not re-emit."""
    store = _enabled_store()  # hb=10, gate=5
    _put(store, ("holder", "shared"), "doc", {"v": 1})  # holder MODIFIED on 'doc' at tick 1
    with caplog.at_level(logging.WARNING, logger="ccs.adapters.base"):
        # A different agent operates on a DIFFERENT artifact to advance the tick
        # without disturbing holder's grant via the coherence protocol; only the
        # sweep can reclaim it.
        store._tick = 50
        _put(store, ("other", "shared"), "scratch", {"v": 1})  # tick 51: sweep reclaims holder (gap 50 >= 10)
        store._tick = 100
        _put(store, ("other", "shared"), "scratch", {"v": 2})  # tick 101: holder already reclaimed -> no re-emit
    records = [
        r for r in caplog.records
        if r.name == "ccs.adapters.base" and r.levelno == logging.WARNING
    ]
    assert len(records) == 1
    assert records[0].reclaim_count == 1


def test_ccsstore_default_thresholds_no_false_reclaim_then_reclaims() -> None:
    """Unit 6 CI-runnable false-reclaim regression (R10 supplement), exercised via
    CCSStore batch semantics with the v0.9.0 DEFAULT thresholds (heartbeat_timeout
    =120, gate=60). A held grant is NOT reclaimed while the holder's heartbeat gap
    is within the timeout, and IS reclaimed once it crosses. Guards future tuning
    patches against silently regressing the safety floor."""
    store = _store()  # bare -> enabled-by-default: hb=120, max_hold=900, gate=60
    _put(store, ("holder", "shared"), "doc", {"v": 1})  # holder MODIFIED at tick 1, last hb=1
    artifact_id = store._artifact_map[(("shared",), "doc")]
    holder_id = store.core.agent_id_for("holder")

    # Within the heartbeat window: drive a sweep at tick 61 (gap 60 < 120). Kept.
    store._tick = 60
    _put(store, ("other", "shared"), "scratch", {"v": 1})  # tick 61: sweep fires, holder gap 60 < 120
    assert store.core.registry.get_agent_state(artifact_id, holder_id) == MESIState.MODIFIED

    # Past the window: drive a sweep at tick 121 (gap 120 >= 120). Reclaimed.
    store._tick = 120
    _put(store, ("other", "shared"), "scratch", {"v": 2})  # tick 121: sweep fires, holder gap 120 >= 120
    assert store.core.registry.get_agent_state(artifact_id, holder_id) == MESIState.INVALID


# ---------------------------------------------------------------------------
# Unit 5 — OCC write_cas reachability through CCSStore
#
# CCSStore inherits write_cas via CoherenceAdapterCore (store.core). This is a
# reachability smoke test confirming the OCC path is wired through the verified
# representative adapter; the other four framework adapters inherit the same
# write_cas via CoherenceAdapterCore with no per-adapter change (verified by the
# CoherenceAdapterCore tests in tests/test_adapters.py).
# ---------------------------------------------------------------------------


def test_ccsstore_occ_write_cas_path_is_reachable() -> None:
    store = _store()
    _put(store, ("writer", "shared"), "doc", {"v": 1})
    artifact_id = store._artifact_map[(("shared",), "doc")]

    # Land two SHARED holders at the coordinator (the S/I OCC starting point).
    store.core.register_agent("writer")
    store.core.register_agent("peer")
    store.core.read(agent_name="writer", artifact_id=artifact_id, now_tick=10)
    store.core.read(agent_name="peer", artifact_id=artifact_id, now_tick=10)
    assert (
        store.core.registry.get_agent_state(artifact_id, store.core.agent_id_for("writer"))
        == MESIState.SHARED
    )

    # The inherited OCC path commits and invalidates the peer.
    updated = store.core.write_cas(
        agent_name="writer",
        artifact_id=artifact_id,
        make_content=lambda entry: (json.dumps({"v": entry.local_version + 1}), None),
        now_tick=11,
    )

    assert updated.version > 1
    assert (
        store.core.runtime("peer").cache.get(artifact_id).state == MESIState.INVALID
    )
