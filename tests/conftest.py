# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Pytest configuration: collection guards, the shared ``registry`` fixture,
and the lanes CI splits the suite into.

`tests/test_diagnose_*.py` import the public surface of ``ccs.diagnose``,
which transitively imports ``langchain_core`` and ``langgraph`` (declared
in the ``[diagnose]`` optional extra). Users running ``pytest`` after a
bare ``pip install -e ".[dev]"`` would otherwise hit cryptic ImportError
during collection.

CI installs ``.[dev,diagnose]`` so all diagnose tests run. This guard is
the second layer of defense for local installs and for any future CI
matrix that exercises bare-install paths.

Lanes: CI runs each Python version as several shards (pytest-split), and
each shard as two lanes. ``--test-lane parallel`` keeps every test not
marked ``serial`` and runs on several xdist workers; ``--test-lane serial``
keeps only ``serial`` tests and runs with ``-n 0``. The lane is a command-line
option, never ``-m``: a command-line ``-m`` replaces the ``addopts``
exclusions instead of adding to them. ``--node-id-dir=DIR`` makes the run write
what each lane collected, what this shard selected and what it ran: the
lists a CI gate checks to prove every test ran exactly once
(``tools/check_shard_coverage.py``); give it with ``=``, because pytest reads
an existing DIR passed as a separate argument as a test path before this
conftest is loaded. With neither option, nothing changes.
"""

from __future__ import annotations

import importlib.util
import signal
from pathlib import Path

import pytest

SERIAL_MARKER = "serial"
PARALLEL_LANE, SERIAL_LANE = LANES = ("parallel", "serial")

collect_ignore_glob: list[str] = []

if importlib.util.find_spec("langchain_core") is None:
    collect_ignore_glob.append("test_diagnose_*.py")

# Live-API tests (Q6 probe smoke, OpenAI-adapter smoke) import `openai` /
# `mistralai` at module level. On a bare `[dev]` install those SDKs are absent;
# ignore the live-test glob so collection does not ImportError. The live tests
# also carry an in-module `pytest.importorskip` as a second layer, and the
# `live_api` marker keeps them out of the default `pytest -q` run regardless.
if importlib.util.find_spec("openai") is None or importlib.util.find_spec("mistralai") is None:
    collect_ignore_glob.append("test_*_live.py")


def _mcp_server_available() -> bool:
    # find_spec on a submodule raises ModuleNotFoundError when an ancestor is
    # absent (mcp missing, OR a partial transitive mcp without mcp.server).
    try:
        return importlib.util.find_spec("mcp.server.fastmcp") is not None
    except ModuleNotFoundError:
        return False


# The front-door demo test imports `examples.mcp_stale_write_guard`, which imports
# `ccs.mcp.server` -> the `mcp` SDK (the `[mcp]` extra). Skip it on a bare install
# (the tests/mcp/ package carries its own conftest guard for the rest). CI's Tests
# job installs `.[...,mcp]` so it runs there.
if not _mcp_server_available():
    collect_ignore_glob.append("test_mcp_front_door_demo.py")


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("lanes", "CI lanes (see tests/conftest.py)")
    group.addoption(
        "--test-lane",
        choices=LANES,
        default=None,
        help="Run only the tests of one lane: 'parallel' (not marked serial) or 'serial'.",
    )
    group.addoption(
        "--node-id-dir",
        default=None,
        help=(
            "Write this run's collected, selected and run node ids to <lane>-full.txt, <lane>-split.txt and "
            "<lane>-ran.txt in DIR. "
            "Pass it as --node-id-dir=DIR: an existing DIR given as a separate argument is taken for a test path "
            "before this conftest loads, and the option is then rejected."
        ),
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep the chosen lane, then record it as the lane's full list.

    ``trylast`` places this after the ``-m`` deselection and before
    pytest-split, whose own ``trylast`` hook is registered later (in
    ``pytest_configure``) and so runs after this one: the full list is the
    lane after the ``addopts`` exclusions and before splitting."""
    lane = config.getoption("test_lane")
    if lane is not None:
        want_serial = lane == SERIAL_LANE
        kept: list[pytest.Item] = []
        dropped: list[pytest.Item] = []
        for item in items:
            in_lane = (item.get_closest_marker(SERIAL_MARKER) is not None) == want_serial
            (kept if in_lane else dropped).append(item)
        if dropped:
            config.hook.pytest_deselected(items=dropped)
            items[:] = kept
    _write_node_ids(config, "full", items)


def pytest_collection_finish(session: pytest.Session) -> None:
    """Record what this run selected, after pytest-split chose its group."""
    _write_node_ids(session.config, "split", session.items)


def _write_node_ids(config: pytest.Config, kind: str, items: list[pytest.Item]) -> None:
    directory = config.getoption("node_id_dir")
    if directory is None or _is_secondary_xdist_worker(config):
        return
    path = Path(directory) / f"{config.getoption('test_lane') or 'all'}-{kind}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: a second writer for one list is a bug in this hook,
    # and the run should fail rather than let it overwrite the evidence.
    with path.open("x", encoding="utf-8") as out:
        out.writelines(f"{item.nodeid}\n" for item in items)


def _reject_serial_lane_on_workers(config: pytest.Config) -> None:
    if config.getoption("test_lane") == SERIAL_LANE and getattr(config.option, "numprocesses", None) not in (None, 0):
        raise pytest.UsageError(
            "--test-lane serial runs with -n 0: its tests start real processes that race each other "
            "and are not safe beside other workers"
        )


class _RanRecorder:
    """Writes ``<lane>-ran.txt``: every test that produced a result.

    The collection lists say what a shard selected; this says what it ran.
    xdist 3.8 exits 0 with "no tests ran" when a worker dies during
    collection, so a shard can pass with a full split list and nothing run.
    It lives in the process that receives every report: the xdist
    controller, or the only process under ``-n 0``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._ran: list[str] = []

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        # One entry per test: its call, or its setup when the setup skipped
        # or failed and no call followed.
        if report.when == "call" or (report.when == "setup" and not report.passed):
            self._ran.append(report.nodeid)

    def pytest_sessionfinish(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("x", encoding="utf-8") as out:
            out.writelines(f"{node_id}\n" for node_id in self._ran)


def _register_ran_recorder(config: pytest.Config) -> None:
    directory = config.getoption("node_id_dir")
    if directory is None or config.getoption("collectonly") or getattr(config, "workerinput", None) is not None:
        return
    path = Path(directory) / f"{config.getoption('test_lane') or 'all'}-ran.txt"
    config.pluginmanager.register(_RanRecorder(path), "lane-ran-recorder")


def _is_secondary_xdist_worker(config: pytest.Config) -> bool:
    """Every xdist worker collects the same items; only ``gw0`` writes them.
    The controller collects nothing, so it never reaches the writer."""
    workerinput = getattr(config, "workerinput", None)
    return workerinput is not None and workerinput["workerid"] != "gw0"


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """A shard whose split selects no tests for a lane has run correctly; the
    gate's coverage check, not pytest's exit 5, decides whether the shards
    together covered the lane (it fails on an empty full list).

    Only on evidence, though: "no tests ran" is also what xdist reports when a
    worker dies during collection, after ``gw0`` has already written a split
    full of tests that never ran. So exit 5 becomes 0 only when this shard's
    split list was written and is empty."""
    if exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED and _split_was_empty(session.config):
        session.exitstatus = pytest.ExitCode.OK


def _split_was_empty(config: pytest.Config) -> bool:
    directory, lane = config.getoption("node_id_dir"), config.getoption("test_lane")
    if directory is None or lane is None or not config.getoption("splits", None):
        return False
    try:
        return (Path(directory) / f"{lane}-split.txt").stat().st_size == 0
    except FileNotFoundError:
        return False


def pytest_configure(config):
    """Neutralize the v0.9.0 transitional RuntimeWarning at COLLECTION time.

    Module-level ``CrashRecoveryConfig`` constructions in test files (e.g. the
    ``CR_CFG`` constant in test_adapter_crash_recovery.py) run at import /
    collection — before the autouse fixture below ever fires. Pre-setting the
    flag here keeps a ``-W error::RuntimeWarning`` run from failing collection
    on that first construction. Dedicated warning-assertion tests still reset
    the flag to ``False`` to observe the warning.
    """
    from ccs.coordinator import service as _service

    _service._V090_FIRST_USE_WARNED = True
    _reject_serial_lane_on_workers(config)
    _register_ran_recorder(config)


@pytest.fixture(autouse=True)
def _neutralize_v090_first_use_warning():
    """Suite-wide neutralizer for the v0.9.0 ``CrashRecoveryConfig`` transitional
    ``RuntimeWarning`` (C-flip plan, Unit 5).

    The warning fires once per process on the FIRST ``CrashRecoveryConfig``
    construction — including explicit ones. Without this, whichever test
    constructs a config first (especially under ``warnings.simplefilter
    ("error")``) would observe the warning order-dependently. We mark the
    module-level flag "already warned" before every test so the warning is a
    no-op by default.

    The dedicated tests in ``tests/test_coordinator.py`` that ASSERT the
    warning opt back in via their own ``reset_v090_first_use_flag`` fixture
    (or an inline reset), which pytest runs after this autouse fixture and
    flips the flag to ``False`` so the warning fires for those tests only.
    """
    from ccs.coordinator import service as _service

    _service._V090_FIRST_USE_WARNED = True
    yield


@pytest.fixture(autouse=True)
def _restore_sigpipe_disposition():
    """Put SIGPIPE's disposition back after every test.

    Tests that change process-wide signal state must not affect later tests.
    Leaving SIGPIPE at its default action can kill the next test's process
    when it writes to a closed pipe or socket, without a Python traceback.
    """
    if not hasattr(signal, "SIGPIPE"):
        yield
        return
    saved = signal.getsignal(signal.SIGPIPE)
    yield
    if signal.getsignal(signal.SIGPIPE) != saved:
        signal.signal(signal.SIGPIPE, saved)


@pytest.fixture(params=["memory", "sqlite"])
def registry(request, tmp_path: Path):
    """Both registry implementations, identically — the parametrization IS the
    parity harness for registry-level surfaces.

    Shared here rather than per-module because the conflict-counter suite and
    the foreign-write-counter suite that follows it assert the same contract
    across both arms, from what had become byte-identical local copies. Modules
    that define their own ``registry`` shadow this one, so the pre-existing
    per-module fixtures keep their own param ids and teardown idiom;
    ``tests/test_registry_fixture_contract.py`` pins both halves of that split.

    The sqlite arm's filename is deliberately its own: consumers put their own
    sqlite files in the same ``tmp_path``, and the tree's default ``state.db``
    is a name two coordinator suites already reserve for raw-sqlite probes.
    """
    # Deferred like the ``ccs`` imports in the hooks above, so importing this
    # conftest stays free of the package under test. Nothing here needs `ccs`
    # until a test actually requests the fixture.
    from ccs.coordinator.registry import ArtifactRegistry
    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry

    if request.param == "memory":
        yield ArtifactRegistry()
    else:
        reg = SqliteArtifactRegistry(tmp_path / "shared-registry-arm.db")
        yield reg
        reg.close()
