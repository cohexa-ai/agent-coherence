# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Full-surface conformance contract for the registry Protocols.

Before the Phase-1 extraction the two registries
(:class:`~ccs.coordinator.registry.ArtifactRegistry` and
:class:`~ccs.coordinator.sqlite_registry.SqliteArtifactRegistry`) shared no base
class and their parity was asserted piecemeal (one behaviour at a time). This
module pins the WHOLE surface:

- both registries are ``isinstance`` of :class:`RegistryBase`, and the SQLite one
  is additionally ``isinstance`` of :class:`SqliteExtended`;
- the exact expected method names (47 base + 13 extended) are present + callable
  on each registry, and the base **property** members (e.g. ``coordinator_epoch``)
  are present as properties — the callable checks cannot see them, so they are
  pinned separately;
- each registry method's parameter STRUCTURE (names, kinds, defaults — annotation
  strings ignored, since ``from __future__ import annotations`` stringifies them
  and they legitimately differ, e.g. ``get_content``'s ``str`` vs ``bytes``)
  matches the Protocol method's;
- a deliberately-incomplete stub missing one base method is NOT
  ``isinstance(RegistryBase)`` — proving conformance has teeth;
- :class:`CoordinatorService` (typed against :class:`RegistryBase`) works with a
  real registry end-to-end (register + read smoke).
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.registry_protocol import (
    ForeignWriteDetection,
    RegistryBase,
    SqliteExtended,
)
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.types import Artifact, FetchRequest

# The full expected surface, frozen here as the contract (not derived from the
# Protocol at runtime so a silent Protocol edit cannot also silently move the
# goalposts this test guards).
BASE_METHODS = frozenset(
    {
        "abort_guard",
        "adjust_checkpoint_pin_refcount",
        "all_session_meta",
        "artifact_ids",
        "bind_caller_principal",
        "capture_version_vector",
        "clear_agent_transient",
        "commit_all",
        "commit_cas",
        "create_checkpoint",
        "get_agent_state",
        "get_agent_transient",
        "get_artifact",
        "get_artifact_and_generation",
        "get_caller_principal",
        "get_checkpoint",
        "get_checkpoint_members",
        "get_content",
        "get_content_at_version",
        "get_last_reclamation",
        "get_owner_generation",
        "get_read_generation",
        "get_session_cut",
        "get_session_meta",
        "get_state_map",
        "get_transient_map",
        "get_transient_tick",
        "get_version_record",
        "granted_at_tick",
        "has_artifact",
        "invalid_reclamations",
        "last_heartbeat_tick",
        "last_observed_version_for",
        "list_checkpoints",
        "record_heartbeat",
        "record_last_reclamation",
        "register_artifact",
        "release_session",
        "remove_artifact",
        "retention_meta",
        "session_count",
        "set_agent_state",
        "set_agent_transient",
        "set_artifact_and_content",
        "set_checkpoint_member_pin",
        "set_checkpoint_member_restore",
        "set_checkpoint_restore_status",
        "valid_holders",
    }
)

EXTENDED_ONLY_METHODS = frozenset(
    {
        "artifact_names_under_prefix",
        "artifacts_held_by_agent",
        "close",
        "evict_stale_notices",
        "get_artifact_updated_at",
        "last_writer_for",
        "lookup_artifact_id_by_name",
        "peek_preemption_notice",
        "pop_pending_notices",
        "pop_preemption_notice",
        "record_preemption_notice",
        "resolve_or_register",
        "status_snapshot",
    }
)

DETECTION_METHODS = frozenset(
    {
        "artifacts_with_detection_edge",
        "clear_detection_edges",
        "close_detection_run",
        "detection_runs",
        "detection_uncoverable",
        "foreign_write_totals",
        "record_detection_tick",
        "record_detection_uncoverable",
        "record_foreign_write",
    }
)
"""The foreign-write detection surface, deliberately its own Protocol.

``SqliteExtended`` is runtime_checkable and the service layer isinstance-checks
it to pick the durable resolve path, so a structural check passes only when
every member is present. Folding instrumentation in would have silently demoted
any backend that implements the coordination surface but not this one."""

# Property members of the base contract. The method checks below filter by
# callable(), so a ``@property`` is structurally invisible to them; pin
# properties explicitly so a backend typed against RegistryBase cannot omit one,
# pass ``isinstance``, and still fail at runtime (e.g. ``coordinator_epoch`` on
# the read-fence path).
BASE_PROPERTIES = frozenset({"coordinator_epoch"})


@pytest.fixture
def inmem_registry() -> ArtifactRegistry:
    return ArtifactRegistry()


@pytest.fixture
def sqlite_registry(tmp_path: Path) -> Iterator[SqliteArtifactRegistry]:
    reg = SqliteArtifactRegistry(tmp_path / "parity.db")
    yield reg
    reg.close()


def _param_structure(func: object) -> list[tuple[str, object, object]]:
    """Return ``(name, kind, default)`` per parameter, EXCLUDING ``self`` and
    annotations. Annotations are excluded on purpose: ``from __future__
    import annotations`` makes them strings that legitimately differ between a
    registry and the Protocol (and between the two registries), so structure —
    names + kinds + defaults — is the conformance invariant we pin."""
    sig = inspect.signature(func)  # type: ignore[arg-type]
    return [
        (name, p.kind, p.default)
        for name, p in sig.parameters.items()
        if name != "self"
    ]


# ---------------------------------------------------------------------------
# isinstance — structural conformance via the runtime_checkable Protocols
# ---------------------------------------------------------------------------


def test_inmem_registry_is_registry_base(inmem_registry: ArtifactRegistry) -> None:
    assert isinstance(inmem_registry, RegistryBase)


def test_sqlite_registry_is_registry_base(
    sqlite_registry: SqliteArtifactRegistry,
) -> None:
    assert isinstance(sqlite_registry, RegistryBase)


def test_sqlite_registry_is_sqlite_extended(
    sqlite_registry: SqliteArtifactRegistry,
) -> None:
    assert isinstance(sqlite_registry, SqliteExtended)


def test_both_registries_implement_detection(
    inmem_registry: ArtifactRegistry, sqlite_registry: SqliteArtifactRegistry
) -> None:
    assert isinstance(inmem_registry, ForeignWriteDetection)
    assert isinstance(sqlite_registry, ForeignWriteDetection)


def test_a_backend_without_detection_is_still_sqlite_extended() -> None:
    """The regression this split exists to prevent.

    The service layer isinstance-checks SqliteExtended to choose the durable
    resolve path over a slower mint. A backend that implements the whole
    coordination surface but no instrumentation must keep taking that path."""

    class _CoordinationOnly:
        pass

    for name in BASE_METHODS | EXTENDED_ONLY_METHODS:
        setattr(_CoordinationOnly, name, lambda self, *a, **k: None)
    _CoordinationOnly.coordinator_epoch = property(lambda self: 0)

    backend = _CoordinationOnly()
    assert isinstance(backend, SqliteExtended)
    assert not isinstance(backend, ForeignWriteDetection)


# ---------------------------------------------------------------------------
# Full expected surface — names present + callable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BASE_METHODS))
def test_base_method_present_on_inmem(
    inmem_registry: ArtifactRegistry, name: str
) -> None:
    assert callable(getattr(inmem_registry, name))


@pytest.mark.parametrize("name", sorted(BASE_METHODS))
def test_base_method_present_on_sqlite(
    sqlite_registry: SqliteArtifactRegistry, name: str
) -> None:
    assert callable(getattr(sqlite_registry, name))


@pytest.mark.parametrize("name", sorted(EXTENDED_ONLY_METHODS))
def test_extended_method_present_on_sqlite(
    sqlite_registry: SqliteArtifactRegistry, name: str
) -> None:
    assert callable(getattr(sqlite_registry, name))


def test_protocol_surface_matches_expected_names() -> None:
    """The Protocols expose EXACTLY the frozen expected names (no drift)."""
    base_names = {
        n
        for n in dir(RegistryBase)
        if not n.startswith("_") and callable(getattr(RegistryBase, n))
    }
    extended_names = {
        n
        for n in dir(SqliteExtended)
        if not n.startswith("_") and callable(getattr(SqliteExtended, n))
    }
    detection_names = {
        n
        for n in dir(ForeignWriteDetection)
        if not n.startswith("_") and callable(getattr(ForeignWriteDetection, n))
    }
    assert base_names == BASE_METHODS
    assert extended_names == BASE_METHODS | EXTENDED_ONLY_METHODS
    assert detection_names == DETECTION_METHODS
    assert not (extended_names & DETECTION_METHODS), (
        "detection members must stay off SqliteExtended: it is isinstance-checked "
        "to choose the durable resolve path, and adding members narrows what passes"
    )
    base_props = {
        n
        for n in dir(RegistryBase)
        if not n.startswith("_") and isinstance(getattr(RegistryBase, n, None), property)
    }
    assert base_props == BASE_PROPERTIES


# ---------------------------------------------------------------------------
# Property members — the non-callable contract surface (coordinator_epoch, …).
# The method checks filter by callable(), so a @property is invisible to them;
# pinned separately so a RegistryBase-typed backend cannot omit one silently.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BASE_PROPERTIES))
def test_base_property_declared_on_protocol(name: str) -> None:
    assert isinstance(getattr(RegistryBase, name, None), property)


@pytest.mark.parametrize("name", sorted(BASE_PROPERTIES))
def test_base_property_present_on_inmem(
    inmem_registry: ArtifactRegistry, name: str
) -> None:
    assert isinstance(getattr(type(inmem_registry), name, None), property)
    assert isinstance(getattr(inmem_registry, name), str)


@pytest.mark.parametrize("name", sorted(BASE_PROPERTIES))
def test_base_property_present_on_sqlite(
    sqlite_registry: SqliteArtifactRegistry, name: str
) -> None:
    assert isinstance(getattr(type(sqlite_registry), name, None), property)
    assert isinstance(getattr(sqlite_registry, name), str)


# ---------------------------------------------------------------------------
# Parameter STRUCTURE parity (names + kinds + defaults; annotations ignored)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(BASE_METHODS))
def test_inmem_base_method_structure_matches_protocol(
    inmem_registry: ArtifactRegistry, name: str
) -> None:
    proto = getattr(RegistryBase, name)
    impl = getattr(type(inmem_registry), name)
    assert _param_structure(impl) == _param_structure(proto)


@pytest.mark.parametrize("name", sorted(BASE_METHODS))
def test_sqlite_base_method_structure_matches_protocol(
    sqlite_registry: SqliteArtifactRegistry, name: str
) -> None:
    proto = getattr(RegistryBase, name)
    impl = getattr(type(sqlite_registry), name)
    assert _param_structure(impl) == _param_structure(proto)


@pytest.mark.parametrize("name", sorted(EXTENDED_ONLY_METHODS))
def test_sqlite_extended_method_structure_matches_protocol(
    sqlite_registry: SqliteArtifactRegistry, name: str
) -> None:
    proto = getattr(SqliteExtended, name)
    impl = getattr(type(sqlite_registry), name)
    assert _param_structure(impl) == _param_structure(proto)


# ---------------------------------------------------------------------------
# Teeth — an incomplete stub is rejected
# ---------------------------------------------------------------------------


def test_incomplete_stub_is_not_registry_base() -> None:
    """A class implementing every base method EXCEPT one is not an instance of
    RegistryBase — so the structural check actually discriminates."""

    class _AlmostRegistry:
        pass

    # Attach every base method except a deliberately-omitted one.
    omitted = "commit_cas"
    for name in BASE_METHODS:
        if name == omitted:
            continue
        setattr(_AlmostRegistry, name, lambda self, *a, **k: None)

    instance = _AlmostRegistry()
    assert not hasattr(instance, omitted)
    assert not isinstance(instance, RegistryBase)


def test_full_base_but_incomplete_extended_is_not_sqlite_extended() -> None:
    """A class with every RegistryBase method AND every SqliteExtended method
    EXCEPT one is RegistryBase but NOT SqliteExtended — the extended surface
    discriminates too, not just the base."""

    class _AlmostSqlite:
        pass

    omitted = "status_snapshot"
    for name in BASE_METHODS | (EXTENDED_ONLY_METHODS - {omitted}):
        setattr(_AlmostSqlite, name, lambda self, *a, **k: None)
    for prop in BASE_PROPERTIES:  # full base surface includes the property members
        setattr(_AlmostSqlite, prop, property(lambda self: "epoch-0"))

    instance = _AlmostSqlite()
    assert isinstance(instance, RegistryBase)  # full base surface present
    assert not isinstance(instance, SqliteExtended)  # missing one extended method


def test_stub_missing_base_property_is_not_registry_base() -> None:
    """A class with every base METHOD but missing a base @property
    (``coordinator_epoch``) is NOT RegistryBase — the property is part of the
    contract, so a future backend cannot omit it and still pass ``isinstance``.
    This is the regression guard for the gap where the Protocol declared only
    methods and a conforming-per-isinstance backend would AttributeError on the
    first fence read."""

    class _NoEpoch:
        pass

    for name in BASE_METHODS:
        setattr(_NoEpoch, name, lambda self, *a, **k: None)
    # deliberately omit BASE_PROPERTIES (coordinator_epoch)
    instance = _NoEpoch()
    assert not any(hasattr(instance, p) for p in BASE_PROPERTIES)
    assert not isinstance(instance, RegistryBase)


# ---------------------------------------------------------------------------
# Service smoke — CoordinatorService (typed RegistryBase) drives a real registry
# ---------------------------------------------------------------------------


def test_service_against_protocol_typed_registry_smoke() -> None:
    """CoordinatorService is constructed against the RegistryBase parameter type
    and a trivial register/read round-trips through it."""
    registry: RegistryBase = ArtifactRegistry()
    svc = CoordinatorService(registry)

    artifact = svc.register_artifact(name="plan.md", content="seed-v1")
    assert isinstance(artifact, Artifact)

    resp = svc.fetch(
        FetchRequest(
            artifact_id=artifact.id,
            requesting_agent_id=uuid4(),
            requested_at_tick=1,
        )
    )
    assert resp.content == "seed-v1"
    assert resp.version == artifact.version
