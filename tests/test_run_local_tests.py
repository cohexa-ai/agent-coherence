# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The local test runner, ``tools/run_local_tests.py``.

The runner wraps the same two pytest lanes CI runs. Each guard below protects
one way a local run could check less than CI, or less than the tree it was
started in: an option that rewrites the selection, a path that leaves the
suite, a venv missing what CI installs, a child that imports another
checkout's code, or caches that never warm up.

Every test injects roots, interpreters and lock paths; none of them touches
the shared venv or the machine-wide turn, because these tests themselves run
inside a run that may hold it.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_local_tests", REPO_ROOT / "tools" / "run_local_tests.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A real git repository with a ``tests`` directory and one commit."""
    root = tmp_path / "main"
    (root / "tests" / "adapters").mkdir(parents=True)
    (root / "tests" / "test_x.py").write_text("def test_y():\n    pass\n")
    _git(root.parent, "init", "-q", str(root))
    _git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    return root


# --- options ---------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["-m", "not slow"],
        ["--store-durations"],
        ["-p", "no:xdist"],
        ["--rootdir", "x"],
        ["--rootdir=x"],
        ["-o", "addopts="],
        ["--test-lane", "serial"],
        ["--node-id-dir=/tmp/x"],
        ["--splits", "2", "--group", "1"],
        ["-mnot slow"],
        ["-pno:xdist"],
        ["-oaddopts="],
        ["-cx.ini"],
    ],
)
def test_options_that_override_ci_selection_or_runner_settings_are_refused(argv: list[str]) -> None:
    with pytest.raises(runner.Refused, match="plain pytest"):
        runner.split_args(argv)


def test_a_refused_option_exits_with_pytests_usage_code_before_anything_runs(repo: Path, monkeypatch, capsys) -> None:
    # From a throwaway repo, a guard that stopped refusing fails fast on the
    # missing venv instead of starting a full run on this checkout's turn.
    monkeypatch.chdir(repo)
    assert runner.main(["-m", "not slow"]) == runner.USAGE_ERROR == 4
    assert "plain pytest" in capsys.readouterr().err


def test_a_missing_venv_stops_the_command_with_the_preflight_code(repo: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(repo)
    assert runner.main([]) == runner.PREFLIGHT_FAILED == 6
    assert str(repo.resolve() / ".venv" / "bin" / "python") in capsys.readouterr().err


def test_a_worktree_run_uses_the_main_venv_and_its_own_tree(repo: Path, tmp_path: Path) -> None:
    """The venv is shared, but pytest, its imports and its config are the worktree's."""
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "agent-coherence"\n[project.optional-dependencies]\ndev = ["pytest"]\ndiagnose = []\nmcp = []\n'
    )
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (venv_bin / "python").chmod(0o755)
    worktree = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", "--detach", str(worktree))
    (worktree / "pyproject.toml").write_text((repo / "pyproject.toml").read_text())
    spec = runner.prepare(["tests/test_x.py"], worktree)
    assert spec.interpreter == repo.resolve() / ".venv" / "bin" / "python"
    assert spec.roots.tree == worktree.resolve()
    assert spec.env["PYTHONPATH"].split(os.pathsep)[0] == str(worktree.resolve() / "src")


def test_a_worker_count_is_refused_on_a_full_run_and_forwarded_on_a_targeted_one() -> None:
    with pytest.raises(runner.Refused, match="worker count"):
        runner.check_mode(runner.split_args(["-n", "2"]))
    targeted = runner.split_args(["-n", "2", "tests/test_x.py"])
    runner.check_mode(targeted)
    assert targeted.option_tokens() == ["-n", "2"]


@pytest.mark.parametrize(
    ("argv", "is_full"),
    [
        (["-k", "foo", "-x"], True),
        (["-k", "foo", "tests/test_x.py"], False),
        (["--deselect", "tests/test_x.py::test_y"], True),
        (["--ignore", "tests/adapters"], True),
        (["-kfoo"], True),
        (["--tb=short", "tests/test_x.py::test_y"], False),
    ],
)
def test_option_values_are_never_read_as_test_paths(argv: list[str], is_full: bool) -> None:
    assert runner.split_args(argv).is_full is is_full


def test_an_unknown_option_keeps_its_value_unless_the_value_is_a_test_path() -> None:
    assert runner.split_args(["--junitxml", "out.xml"]).option_tokens() == ["--junitxml", "out.xml"]
    assert runner.split_args(["--junitxml", "out.xml"]).is_full
    logged = runner.split_args(["--log-level", "DEBUG", "tests/test_x.py"])
    assert logged.paths == ("tests/test_x.py",) and logged.option_tokens() == ["--log-level", "DEBUG"]
    flagged = runner.split_args(["--cache-clear", "tests/test_x.py::test_y"])
    assert flagged.paths == ("tests/test_x.py::test_y",) and flagged.option_tokens() == ["--cache-clear"]


def test_bundled_short_flags_are_flags_not_unknown_options() -> None:
    args = runner.split_args(["-xvs", "tests/test_x.py"])
    assert args.paths == ("tests/test_x.py",)
    assert not runner.should_run_serial(1, runner.split_args(["-xv"]))


def test_narrowing_options_are_named_so_the_summary_can_report_them() -> None:
    assert runner.split_args(["-k", "foo", "--lf", "-x"]).narrowing() == ["-k", "--lf"]


# --- paths -----------------------------------------------------------------


def test_a_path_outside_tests_is_refused(repo: Path) -> None:
    with pytest.raises(runner.Refused, match="plain pytest"):
        runner.resolve_test_paths(["src/ccs/x.py"], cwd=repo, tree=repo)


def test_test_paths_and_ids_resolve_relative_to_the_tree(repo: Path) -> None:
    resolved = runner.resolve_test_paths(["tests/adapters", "tests/test_x.py::test_y"], cwd=repo, tree=repo)
    assert resolved == ["tests/adapters", "tests/test_x.py::test_y"]


def test_a_path_given_from_a_subdirectory_is_rewritten_relative_to_the_tree(repo: Path) -> None:
    resolved = runner.resolve_test_paths(["test_x.py::test_y"], cwd=repo / "tests", tree=repo)
    assert resolved == ["tests/test_x.py::test_y"]


# --- roots and interpreter -------------------------------------------------


def test_the_main_checkout_resolves_to_an_absolute_path_from_itself(repo: Path) -> None:
    roots = runner.find_roots(repo / "tests")
    assert roots.tree == repo.resolve()
    assert roots.main == repo.resolve()
    assert roots.common_dir == (repo / ".git").resolve()


def test_a_worktree_resolves_to_its_own_tree_and_the_main_checkout(repo: Path, tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", "--detach", str(worktree))
    roots = runner.find_roots(worktree)
    assert roots.tree == worktree.resolve()
    assert roots.main == repo.resolve()
    assert roots.common_dir == (repo / ".git").resolve()


def test_the_interpreter_is_the_main_checkouts_venv_unless_the_tree_has_one(tmp_path: Path) -> None:
    roots = runner.Roots(tree=tmp_path / "wt", main=tmp_path / "main", common_dir=tmp_path / "main" / ".git")
    assert runner.find_interpreter(roots) == tmp_path / "main" / ".venv" / "bin" / "python"
    (tmp_path / "wt" / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / "wt" / ".venv" / "bin" / "python").write_text("")
    assert runner.find_interpreter(roots) == tmp_path / "wt" / ".venv" / "bin" / "python"


# --- preflight -------------------------------------------------------------


def _pyproject(tmp_path: Path, extras: str) -> Path:
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "pyproject.toml").write_text(
        textwrap.dedent(
            f"""
            [project]
            name = "agent-coherence"
            [project.optional-dependencies]
            {extras}
            """
        )
    )
    return tree


def test_preflight_names_a_missing_interpreter(tmp_path: Path) -> None:
    missing = tmp_path / "main" / ".venv" / "bin" / "python"
    problem = runner.preflight(missing, tmp_path)
    assert problem is not None and str(missing) in problem


def test_preflight_passes_when_every_ci_extra_is_installed(tmp_path: Path) -> None:
    tree = _pyproject(
        tmp_path,
        'dev = ["pytest>=7.0"]\ndiagnose = ["agent-coherence[langgraph]"]\nlanggraph = []\nmcp = []',
    )
    assert runner.preflight(Path(sys.executable), tree) is None


def test_preflight_follows_self_references_and_names_what_is_missing(tmp_path: Path) -> None:
    tree = _pyproject(
        tmp_path,
        'dev = ["pytest>=7.0"]\ndiagnose = ["agent-coherence[extra]"]\n'
        'extra = ["surely-not-installed-ccs-probe>=1"]\nmcp = []',
    )
    problem = runner.preflight(Path(sys.executable), tree)
    assert problem is not None and "surely-not-installed-ccs-probe" in problem


def test_the_reinstall_command_names_the_venvs_owning_checkout_by_absolute_path(tmp_path: Path) -> None:
    interpreter = tmp_path / "main" / ".venv" / "bin" / "python"
    command = runner.reinstall_command(interpreter)
    assert command == f"{interpreter} -m pip install -e '{tmp_path / 'main'}[dev,diagnose,mcp]'"


# --- child environment -----------------------------------------------------


def test_the_child_environment_turns_bytecode_caching_back_on() -> None:
    env = runner.child_env({"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/bin"}, Path("/v/bin"), Path("/t"))
    assert "PYTHONDONTWRITEBYTECODE" not in env


def test_the_child_environment_puts_the_venv_and_the_tree_first() -> None:
    env = runner.child_env({"PATH": "/bin", "PYTHONPATH": "/extra"}, Path("/v/bin"), Path("/t"))
    assert env["PATH"].split(os.pathsep) == ["/v/bin", "/bin"]
    assert env["PYTHONPATH"].split(os.pathsep) == ["/t/src", "/extra"]


def test_a_console_script_started_in_the_child_environment_imports_the_trees_code(tmp_path: Path) -> None:
    """The shared venv's editable install points at the main checkout; a
    console script run from a worktree must still test the worktree."""
    tree = tmp_path / "tree"
    (tree / "src" / "ccs").mkdir(parents=True)
    (tree / "src" / "ccs" / "__init__.py").write_text("")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "ccs-probe-console"
    script.write_text(f"#!{sys.executable}\nimport ccs\nprint(ccs.__file__)\n")
    script.chmod(0o755)
    env = runner.child_env({"PATH": os.environ["PATH"]}, bin_dir, tree)
    out = subprocess.run(["ccs-probe-console"], env=env, capture_output=True, text=True, check=True).stdout
    assert Path(out.strip()).resolve() == (tree / "src" / "ccs" / "__init__.py").resolve()


# --- the machine-wide turn -------------------------------------------------


def _probe_is_held(lock_path: Path) -> bool:
    """True while some open file description holds the lock."""
    import fcntl

    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    finally:
        os.close(fd)
    return False


def _holder(tree: str, pid: int = 1) -> dict:
    return {"tree": tree, "pid": pid, "mode": "full run", "started": "2026-10-18T12:00:00"}


def test_a_held_turn_blocks_a_second_holder_until_it_is_released(tmp_path: Path) -> None:
    lock = tmp_path / "turn.lock"
    fd = runner.acquire_turn(lock, _holder("/a"), print)
    assert _probe_is_held(lock)
    runner.release_turn(fd)
    assert not _probe_is_held(lock)


def test_a_waiter_says_once_who_holds_the_turn_then_takes_it(tmp_path: Path, monkeypatch) -> None:
    import threading

    monkeypatch.setattr(runner, "POLL_SECONDS", 0.01)
    lock = tmp_path / "turn.lock"
    held = runner.acquire_turn(lock, _holder("/other/tree", pid=4242), print)
    threading.Timer(0.3, runner.release_turn, args=(held,)).start()
    said: list[str] = []
    fd = runner.acquire_turn(lock, _holder("/this/tree"), said.append)
    runner.release_turn(fd)
    waiting = [line for line in said if "waiting" in line]
    assert len(waiting) == 1
    assert "/other/tree" in waiting[0] and "4242" in waiting[0] and "2026-10-18T12:00:00" in waiting[0]


def test_a_waiter_says_when_its_own_tree_holds_the_turn(tmp_path: Path, monkeypatch) -> None:
    import threading

    monkeypatch.setattr(runner, "POLL_SECONDS", 0.01)
    lock = tmp_path / "turn.lock"
    held = runner.acquire_turn(lock, _holder("/same/tree"), print)
    threading.Timer(0.2, runner.release_turn, args=(held,)).start()
    said: list[str] = []
    runner.release_turn(runner.acquire_turn(lock, _holder("/same/tree"), said.append))
    assert any("held by this tree" in line for line in said)


def test_a_leftover_lock_file_is_taken_at_once_and_its_record_replaced(tmp_path: Path) -> None:
    lock = tmp_path / "turn.lock"
    lock.write_text('{"tree": "/gone/with/a/much/longer/path", "pid": 1, "mode": "full run", "started": "x"}\n')
    said: list[str] = []
    fd = runner.acquire_turn(lock, _holder("/new", pid=7), said.append)
    try:
        assert said == []
        assert runner.read_holder(lock)["tree"] == "/new"
    finally:
        runner.release_turn(fd)


def test_a_lane_runs_with_the_environment_and_directory_it_is_given(tmp_path: Path) -> None:
    check = "import os, sys; sys.exit(0 if os.environ.get('CCS_LANE_PROBE') == '1' and os.getcwd() == sys.argv[1] else 1)"
    env = {"CCS_LANE_PROBE": "1", "PATH": os.environ["PATH"]}
    assert runner.run_lane([sys.executable, "-c", check, os.path.realpath(tmp_path)], env, tmp_path, None) == 0


_TURN_HOLDER = """
import importlib.util, os, subprocess, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("rlt", sys.argv[1])
rlt = importlib.util.module_from_spec(spec)
sys.modules["rlt"] = rlt
spec.loader.exec_module(rlt)
fd = rlt.acquire_turn(Path(sys.argv[2]), {"tree": "/t", "pid": os.getpid(), "mode": "full run", "started": "x"}, print)
rlt.run_lane([sys.executable, "-c", sys.argv[3]], dict(os.environ), Path.cwd(), fd)
rlt.release_turn(fd)
"""


_INTERRUPTED_RUNNER = """
import importlib.util, os, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("rlt", sys.argv[1])
rlt = importlib.util.module_from_spec(spec)
sys.modules["rlt"] = rlt
spec.loader.exec_module(rlt)
try:
    rlt.run_lane([sys.executable, "-c", sys.argv[2]], dict(os.environ), Path.cwd(), None)
except KeyboardInterrupt:
    sys.exit(130)
"""


def _wait_for(predicate, timeout: float = 10.0) -> bool:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


@pytest.mark.serial
def test_the_turn_outlives_a_killed_runner_until_its_tests_exit(tmp_path: Path) -> None:
    """The Bash tool may background or kill the runner; its pytest still loads
    the machine, so the turn stays held until that pytest exits."""
    import signal

    lock, pid_file = tmp_path / "turn.lock", tmp_path / "lane.pid"
    lane = f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    holder = subprocess.Popen(
        [sys.executable, "-c", _TURN_HOLDER, str(REPO_ROOT / "tools" / "run_local_tests.py"), str(lock), lane]
    )
    try:
        assert _wait_for(lambda: pid_file.exists() and pid_file.read_text())
        os.kill(holder.pid, signal.SIGKILL)
        holder.wait(timeout=10)
        assert _probe_is_held(lock)
    finally:
        if pid_file.exists() and pid_file.read_text():
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        holder.kill()
    assert _wait_for(lambda: not _probe_is_held(lock))


@pytest.mark.serial
def test_processes_a_lane_starts_by_default_do_not_keep_the_turn(tmp_path: Path) -> None:
    """xdist workers and nested pytest runs are started with closed descriptors,
    so the turn ends with the lane's own pytest."""
    import signal

    lock, pid_file = tmp_path / "turn.lock", tmp_path / "grandchild.pid"
    lane_script = tmp_path / "lane.py"
    lane_script.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys, time
            from pathlib import Path
            pid_file = Path({str(pid_file)!r})
            sleeper = "import os, time; open(%r, 'w').write(str(os.getpid())); time.sleep(60)" % str(pid_file)
            subprocess.Popen([sys.executable, "-c", sleeper])
            while not (pid_file.exists() and pid_file.read_text()):
                time.sleep(0.05)
            """
        )
    )
    lane = f"exec(open({str(lane_script)!r}).read())"
    holder = subprocess.run(
        [sys.executable, "-c", _TURN_HOLDER, str(REPO_ROOT / "tools" / "run_local_tests.py"), str(lock), lane],
        timeout=30, check=False,
    )
    try:
        assert holder.returncode == 0
        assert pid_file.read_text()
        assert not _probe_is_held(lock)
    finally:
        if pid_file.exists() and pid_file.read_text():
            os.kill(int(pid_file.read_text()), signal.SIGKILL)


# --- orchestration ---------------------------------------------------------

SERIAL_MODULE = "tests/test_claude_code_lifecycle.py"
PLAIN_MODULE = "tests/test_bash_path_detector.py"


def test_the_serial_scan_finds_exactly_the_pinned_serial_modules() -> None:
    from tests.test_shard_lanes import EXPECTED_SERIAL_MODULES

    assert set(runner.serial_modules(REPO_ROOT)) == EXPECTED_SERIAL_MODULES


def test_serial_targets_keep_named_serial_modules_and_expand_directories() -> None:
    modules = ["tests/conformance/test_process_harness.py", SERIAL_MODULE]
    assert runner.serial_targets([PLAIN_MODULE], modules) == []
    assert runner.serial_targets([f"{SERIAL_MODULE}::test_a"], modules) == [f"{SERIAL_MODULE}::test_a"]
    assert runner.serial_targets(["tests/conformance"], modules) == ["tests/conformance/test_process_harness.py"]
    assert runner.serial_targets(["tests"], modules) == modules


class _FakeLanes:
    """Stands in for ``run_lane``: records each command and whether the turn was
    held, and writes the ran list a real lane would leave in --node-id-dir."""

    def __init__(self, codes: list[int], lock: Path, ran: int = 3, serial_selected: int = 1) -> None:
        self.codes, self.lock, self.ran, self.calls = list(codes), lock, ran, []
        self.serial_selected = serial_selected
        self.lock_held: list[bool] = []
        self.envs: list = []
        self.cwds: list = []

    def __call__(self, command, env, cwd, turn_fd):
        self.calls.append((list(command), turn_fd is not None))
        self.lock_held.append(_probe_is_held(self.lock))
        self.envs.append(env)
        self.cwds.append(cwd)
        lane = command[command.index("--test-lane") + 1]
        node_dir = next(token.split("=", 1)[1] for token in command if token.startswith("--node-id-dir="))
        Path(node_dir).mkdir(parents=True, exist_ok=True)
        Path(node_dir, f"{lane}-ran.txt").write_text("".join(f"t::{i}\n" for i in range(self.ran)))
        if "--co" in command:  # the targeted run's probe of which serial tests it selected
            Path(node_dir, "serial-full.txt").write_text("".join(f"s::{i}\n" for i in range(self.serial_selected)))
            return 0 if self.serial_selected else 5
        return self.codes.pop(0)


def _spec(tmp_path: Path, argv: list[str]):
    args = runner.split_args(argv)
    roots = runner.Roots(tree=REPO_ROOT, main=REPO_ROOT, common_dir=tmp_path)
    return runner.RunSpec(args=args, roots=roots, interpreter=Path(sys.executable), env=dict(os.environ), workers=8)


def test_a_full_run_holds_the_turn_for_both_lanes_and_releases_it(tmp_path: Path) -> None:
    lanes = _FakeLanes([0, 0], tmp_path / runner.LOCK_NAME)
    spec = _spec(tmp_path, [])
    assert runner.execute(spec, lanes, lambda line: None) == 0
    (parallel, parallel_held), (serial, serial_held) = lanes.calls
    assert parallel_held and serial_held
    assert parallel[parallel.index("-n") + 1] == "8"
    assert parallel[parallel.index("--dist") + 1] == "worksteal"
    assert parallel[parallel.index("--test-lane") + 1] == "parallel"
    assert serial[serial.index("-n") + 1] == "0" and serial[serial.index("--test-lane") + 1] == "serial"
    assert [token for token in serial if token.startswith("tests/")] == runner.serial_modules(REPO_ROOT)
    assert lanes.lock_held == [True, True]  # held at <git common dir>/<LOCK_NAME>, shared by every worktree
    assert lanes.envs == [spec.env, spec.env] and lanes.cwds == [REPO_ROOT, REPO_ROOT]
    assert not _probe_is_held(tmp_path / runner.LOCK_NAME)


def test_a_full_run_fails_when_its_parallel_lane_exits_0_having_run_nothing(tmp_path: Path) -> None:
    """xdist 3.8 exits 0 with "no tests ran" when a worker dies in collection."""
    said: list[str] = []
    assert runner.execute(_spec(tmp_path, []), _FakeLanes([0, 0], tmp_path / runner.LOCK_NAME, ran=0), said.append) == 1
    assert any("parallel lane ran no tests" in line for line in said)


def test_a_collect_only_full_run_is_not_reported_as_running_nothing(tmp_path: Path) -> None:
    said: list[str] = []
    assert runner.execute(_spec(tmp_path, ["--co"]), _FakeLanes([0, 0], tmp_path / runner.LOCK_NAME, ran=0), said.append) == 0
    assert not any("ran no tests" in line for line in said)


def test_a_targeted_run_of_a_plain_module_never_takes_the_turn(tmp_path: Path) -> None:
    lanes = _FakeLanes([0], tmp_path / runner.LOCK_NAME)
    assert runner.execute(_spec(tmp_path, [PLAIN_MODULE]), lanes, lambda line: None) == 0
    [(command, held)] = lanes.calls
    assert lanes.lock_held == [False]
    assert not held and PLAIN_MODULE in command and command[command.index("-n") + 1] == "0"


def test_a_targeted_run_takes_the_turn_only_for_its_serial_tests(tmp_path: Path) -> None:
    lanes = _FakeLanes([0, 0], tmp_path / runner.LOCK_NAME)
    assert runner.execute(_spec(tmp_path, [SERIAL_MODULE]), lanes, lambda line: None) == 0
    (parallel, parallel_held), (serial, serial_held) = lanes.calls
    assert not parallel_held and serial_held
    assert lanes.lock_held == [False, True]
    assert serial[-1] == SERIAL_MODULE


def test_a_test_id_in_a_serial_module_takes_the_turn_only_if_it_selects_serial_tests(tmp_path: Path) -> None:
    """Six modules mix serial and parallel tests; naming one parallel test in
    them must not wait behind another worktree's full run."""
    not_serial = _FakeLanes([0], tmp_path / runner.LOCK_NAME, serial_selected=0)
    assert runner.execute(_spec(tmp_path, [f"{SERIAL_MODULE}::test_a"]), not_serial, lambda line: None) == 0
    assert not_serial.lock_held == [False, False] and len(not_serial.calls) == 2
    serial = _FakeLanes([0, 0], tmp_path / runner.LOCK_NAME, serial_selected=1)
    assert runner.execute(_spec(tmp_path, ["-k", "race", SERIAL_MODULE]), serial, lambda line: None) == 0
    assert serial.lock_held == [False, False, True]


def test_a_worker_count_on_a_targeted_run_reaches_only_its_parallel_part(tmp_path: Path) -> None:
    lanes = _FakeLanes([0, 0], tmp_path / runner.LOCK_NAME)
    runner.execute(_spec(tmp_path, ["-n", "2", SERIAL_MODULE]), lanes, lambda line: None)
    (parallel, parallel_held), (serial, _) = lanes.calls
    assert not parallel_held
    assert parallel[parallel.index("-n") + 1] == "2" and parallel.count("-n") == 1
    assert serial[serial.index("-n") + 1] == "0" and serial.count("-n") == 1


@pytest.mark.parametrize(
    ("code", "argv", "runs_serial"),
    [(0, [], True), (1, [], True), (5, [], True), (2, [], False), (3, [], False), (4, [], False), (1, ["-x"], False)],
)
def test_the_serial_lane_follows_a_failed_parallel_lane_unless_told_to_stop(code, argv, runs_serial) -> None:
    assert runner.should_run_serial(code, runner.split_args(argv)) is runs_serial


@pytest.mark.parametrize(
    ("parallel", "serial", "expected"),
    [(0, 5, 0), (1, 0, 1), (5, 5, 5), (0, 1, 1), (0, None, 0), (5, None, 5), (2, None, 2)],
)
def test_lane_exit_codes_combine_into_one(parallel, serial, expected) -> None:
    assert runner.combine_exit_codes(parallel, serial, parallel_ran_nothing=False) == expected


@pytest.mark.parametrize(
    ("argv", "codes", "lanes_run", "expected"),
    [
        ([], [1, 0], 2, 1),
        ([], [0, 1], 2, 1),
        ([], [2], 1, 2),
        ([], [4], 1, 4),
        (["-x"], [1], 1, 1),
        ([SERIAL_MODULE], [1, 0], 2, 1),
        ([SERIAL_MODULE], [2], 1, 2),
        ([SERIAL_MODULE], [0, 1], 2, 1),
    ],
)
def test_a_run_combines_its_lanes_into_one_exit_code(tmp_path: Path, argv, codes, lanes_run, expected) -> None:
    lanes = _FakeLanes(codes, tmp_path / runner.LOCK_NAME)
    assert runner.execute(_spec(tmp_path, argv), lanes, lambda line: None) == expected
    assert len(lanes.calls) == lanes_run


def test_the_summary_says_when_options_narrowed_the_selection(tmp_path: Path) -> None:
    said: list[str] = []
    runner.execute(_spec(tmp_path, ["-k", "foo", "--lf"]), _FakeLanes([0, 0], tmp_path / runner.LOCK_NAME), said.append)
    assert any("selection narrowed by -k --lf" in line for line in said)


def test_a_real_targeted_run_reports_each_lane_and_a_summary(tmp_path: Path) -> None:
    said: list[str] = []
    spec = runner.RunSpec(
        args=runner.split_args([PLAIN_MODULE, "tests/test_cost_drift_check.py"]),
        roots=runner.Roots(tree=REPO_ROOT, main=REPO_ROOT, common_dir=tmp_path),
        interpreter=Path(sys.executable),
        env=runner.child_env(os.environ, Path(sys.executable).parent, REPO_ROOT),
        workers=2,
    )
    assert runner.execute(spec, runner.run_lane, said.append) == 0
    assert any("parallel lane" in line and "targeted run" in line for line in said)
    assert any(re.search(r"done .* parallel lane exit 0, [1-9]\d* ran", line) for line in said)


@pytest.mark.serial
def test_ctrl_c_lets_the_lane_finish_its_own_interrupt_handling(tmp_path: Path) -> None:
    """A terminal Ctrl-C reaches the runner and its pytest together; the runner
    must wait for pytest's teardown instead of killing it mid-way."""
    import signal

    ready, finished = tmp_path / "ready", tmp_path / "finished"
    lane = (
        "import signal, sys, time\n"
        "def stop(*_):\n"
        f"    time.sleep(1.0); open({str(finished)!r}, 'w').write('done'); sys.exit(2)\n"
        "signal.signal(signal.SIGINT, stop)\n"
        f"open({str(ready)!r}, 'w').write('ok')\n"
        "time.sleep(30)\n"
    )
    # A session of its own, so the interrupt reaches only this test's processes.
    runner_proc = subprocess.Popen(
        [sys.executable, "-c", _INTERRUPTED_RUNNER, str(REPO_ROOT / "tools" / "run_local_tests.py"), lane],
        start_new_session=True,
    )
    try:
        assert _wait_for(ready.exists)
        os.killpg(runner_proc.pid, signal.SIGINT)
        assert runner_proc.wait(timeout=20) == 130
        assert finished.read_text() == "done"
    finally:
        if runner_proc.poll() is None:
            os.killpg(runner_proc.pid, signal.SIGKILL)
