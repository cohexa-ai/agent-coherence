# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The `registry` fixture contract: who owns it, and under which param ids.

``tests/conftest.py`` defines a tree-wide ``registry`` fixture. Ten modules
define their own and shadow it. That shadowing is load-bearing and, until this
module existed, entirely unasserted: pytest resolves the nearest fixture
silently, so a module that loses its local definition does not fail — it binds
to the shared one and keeps passing.

Two things break quietly when that happens. The seven modules whose arms are
id'd ``in_memory`` would silently start reporting ``memory``, invalidating any
``-k`` filter and every stored test id. And a module would start exercising a
registry built by different code than its own fixture built, while still
reporting green.

These are structural assertions over the test tree's own source, so they need
no optional extra and can never skip.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import NamedTuple

TESTS_ROOT = Path(__file__).parent

# Modules that own a local ``registry`` fixture, mapped to its param ids
# (``None`` = not parametrized). A module dropping out of this map has lost its
# local fixture and now silently resolves the shared one.
EXPECTED_LOCAL_FIXTURES: dict[str, tuple[str, ...] | None] = {
    "adapters/test_workspace_versioner.py": None,
    "coordinator/test_last_observed_version.py": ("in_memory", "sqlite"),
    "coordinator/test_workspace_checkpoints.py": ("in_memory", "sqlite"),
    "coordinator/test_workspace_registration.py": ("in_memory", "sqlite"),
    "test_commit_all_registry.py": ("memory", "sqlite"),
    "test_fencing.py": ("in_memory", "sqlite"),
    "test_fetch_peer_leg_fence.py": ("in_memory", "sqlite"),
    "test_foreign_write_instrumentation.py": ("memory", "sqlite"),
    "test_registry_lock_coverage.py": ("in_memory", "sqlite"),
    "test_zombie_revoke.py": ("in_memory", "sqlite"),
}

# Modules that deliberately resolve the shared fixture from conftest.
EXPECTED_SHARED_CONSUMERS: frozenset[str] = frozenset({
    "coordinator/test_caller_principal.py",
    "test_conflict_instrumentation.py",
    "test_invalid_reclamations.py",
})

SHARED_PARAM_IDS: tuple[str, ...] = ("memory", "sqlite")

# The shared fixture's sqlite filename must belong to it alone. `state.db` is
# the tree's generic default (~77 uses) and two coordinator suites reserve it
# for raw-sqlite probes while routing their own registry arm to `parity-arm.db`
# — a shared fixture using either name would alias a consumer's own path inside
# one `tmp_path`.
SHARED_DB_FILENAME = "shared-registry-arm.db"


def _fixture_params(node: ast.FunctionDef) -> tuple[bool, tuple[str, ...] | None]:
    """Return (is_registry_fixture, param_ids) for one function definition."""
    is_fixture = False
    params: tuple[str, ...] | None = None
    for dec in node.decorator_list:
        call = dec if isinstance(dec, ast.Call) else None
        target = call.func if call else dec
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            is_fixture = True
        elif isinstance(target, ast.Name) and target.id == "fixture":
            is_fixture = True
        if call and is_fixture:
            for kw in call.keywords:
                if kw.arg == "params" and isinstance(kw.value, (ast.List, ast.Tuple)):
                    params = tuple(
                        e.value for e in kw.value.elts if isinstance(e, ast.Constant)
                    )
    return is_fixture, params


class RegistryFixtureSurface(NamedTuple):
    """Where the `registry` name resolves, across one test tree."""

    local: dict[str, tuple[str, ...] | None]
    """Modules defining their own fixture, mapped to its param ids."""

    consumers: frozenset[str]
    """Modules requesting `registry` that resolve the shared conftest one."""


def scan_registry_fixtures(tests_root: Path) -> RegistryFixtureSurface:
    """Map the `registry` fixture surface of a test tree from its source.

    Takes the root as an argument so the scan can be exercised against a
    mutated copy of the tree — a guard that has never been shown to fail is
    not a guard.
    """
    local: dict[str, tuple[str, ...] | None] = {}
    consumers: set[str] = set()
    for path in sorted(tests_root.rglob("*.py")):
        rel = path.relative_to(tests_root).as_posix()
        tree = ast.parse(path.read_text())
        defines = False
        uses = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name == "registry":
                is_fixture, params = _fixture_params(node)
                if is_fixture:
                    defines = True
                    local[rel] = params
            elif node.name.startswith("test") and "registry" in (
                a.arg for a in node.args.args
            ):
                uses = True
        if uses and not defines:
            consumers.add(rel)
    return RegistryFixtureSurface(local=local, consumers=frozenset(consumers))


def test_local_registry_fixtures_still_shadow_the_shared_one() -> None:
    """A module that loses its local fixture binds to the shared one silently.

    Pytest resolves the nearest `registry` and reports nothing, so only this
    pinned inventory turns that into a failure.
    """
    found = scan_registry_fixtures(TESTS_ROOT).local
    found.pop("conftest.py", None)
    assert found == EXPECTED_LOCAL_FIXTURES


def test_only_declared_modules_resolve_the_shared_registry_fixture() -> None:
    """Requesting `registry` without a local fixture is no longer an error.

    Before conftest owned the name, that was a fixture-not-found failure. Now
    it silently succeeds, so the consumer set is pinned instead.
    """
    assert scan_registry_fixtures(TESTS_ROOT).consumers == EXPECTED_SHARED_CONSUMERS


def test_shared_registry_fixture_keeps_its_param_ids() -> None:
    """The shared fixture's ids are part of every consumer's test id."""
    local = scan_registry_fixtures(TESTS_ROOT).local
    assert local["conftest.py"] == SHARED_PARAM_IDS


def test_shared_registry_fixture_owns_its_sqlite_filename() -> None:
    """The shared fixture's db name must not collide inside a shared tmp_path.

    Consumers put their own sqlite files in the same `tmp_path`; a name the
    rest of the tree also uses would let the fixture's arm alias a consumer's
    raw-probe database.
    """
    conftest = (TESTS_ROOT / "conftest.py").read_text()
    assert f'"{SHARED_DB_FILENAME}"' in conftest

    # conftest.py declares it; this module names it as the pinned constant.
    owners = {"conftest.py", Path(__file__).name}
    others = [
        path.relative_to(TESTS_ROOT).as_posix()
        for path in sorted(TESTS_ROOT.rglob("*.py"))
        if path.name not in owners and f'"{SHARED_DB_FILENAME}"' in path.read_text()
    ]
    assert others == []
