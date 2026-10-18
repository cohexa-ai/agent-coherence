# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Run the test suite locally the way CI does, from any worktree.

Usage (from the main checkout or any worktree)::

    python3 tools/run_local_tests.py                     # full run, both lanes
    python3 tools/run_local_tests.py tests/test_x.py -x  # targeted run

A full run takes a machine-wide turn shared by every worktree of this clone,
so two full runs never fight for the CPU; a waiting run says who holds it.
A targeted run starts at once, and only its serial-marked tests wait for the
turn. Tests run under the main checkout's ``.venv`` with bytecode caching on,
whatever the launching shell inherited.

This script is started by whatever ``python3`` is first on ``PATH``, so it
uses only the standard library and must still run on Python 3.9.
"""

from __future__ import annotations

import datetime
import errno
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

USAGE_ERROR = 4  # pytest's own code for a usage error
PREFLIGHT_FAILED = 6  # pytest never exits with this
INTERRUPTED = 130
NO_TESTS = 5  # pytest: nothing was selected
PREFIX = "[run_local_tests]"
CI_EXTRAS = ("dev", "diagnose", "mcp")
LOCK_NAME = "ccs-local-tests.lock"
POLL_SECONDS = 0.5
SERIAL_MARK = re.compile(r"\bpytest\.mark\.serial\b")  # the pattern tests/test_shard_lanes.py pins

# Options whose value may follow as a separate token, so it is never read as a test path.
VALUE_OPTIONS = frozenset(
    "-k -m -n -p -c -o -r -W --tb --maxfail --deselect --ignore --ignore-glob --durations --durations-min "
    "--basetemp --dist --numprocesses --rootdir --confcutdir --import-mode --capture --color --override-ini "
    "--config-file --splits --group --durations-path --splitting-algorithm --test-lane --node-id-dir".split()
)
FLAG_OPTIONS = frozenset(
    "-x -v -vv -vvv -q -qq -s -l --exitfirst --verbose --quiet --showlocals --lf --last-failed --ff "
    "--failed-first --nf --new-first --sw --stepwise --sw-skip --pdb --trace --runxfail --setup-show "
    "--full-trace --no-header --no-summary --strict-markers --collect-only --co --store-durations "
    "--clean-durations".split()
)
REFUSED = frozenset(
    "-m -p -c -o --rootdir --override-ini --config-file --test-lane --node-id-dir --store-durations "
    "--clean-durations --durations-path --splits --group --splitting-algorithm".split()
)
WORKER_OPTIONS = frozenset({"-n", "--numprocesses", "--dist"})
NARROWING = ("-k", "--lf", "--last-failed", "--deselect", "--ignore", "--ignore-glob", "--sw", "--stepwise")
STOP_EARLY = frozenset({"-x", "--exitfirst", "--maxfail"})


class Refused(Exception):
    """An argument this command will not run; plain pytest is the way to run it."""


@dataclass(frozen=True)
class Option:
    name: str
    tokens: Tuple[str, ...]


@dataclass(frozen=True)
class Args:
    paths: Tuple[str, ...]
    options: Tuple[Option, ...]

    @property
    def is_full(self) -> bool:
        return not self.paths

    def names(self) -> List[str]:
        return [option.name for option in self.options]

    def option_tokens(self, drop: frozenset = frozenset()) -> List[str]:
        return [token for option in self.options if option.name not in drop for token in option.tokens]

    def narrowing(self) -> List[str]:
        return [name for name in self.names() if name in NARROWING]


@dataclass(frozen=True)
class Roots:
    tree: Path
    main: Path
    common_dir: Path


# --- arguments --------------------------------------------------------------


def _option_name(token: str) -> str:
    if token.startswith("--"):
        return token.split("=", 1)[0]
    if len(token) > 2 and token[:2] in VALUE_OPTIONS:
        return token[:2]  # -kfoo, -n4
    return token


def _looks_like_path(token: str) -> bool:
    return "/" in token or "::" in token or token.endswith(".py")


def _take_option(argv: Sequence[str], index: int) -> Tuple[Option, int]:
    token = argv[index]
    name = _option_name(token)
    if name in REFUSED:
        raise Refused(f"{name} changes which tests CI's lanes select or a setting this command owns; use plain pytest")
    attached = "=" in token or (name != token and not token.startswith("--"))
    if name in VALUE_OPTIONS and not attached and index + 1 < len(argv):
        return Option(name, (token, argv[index + 1])), index + 2
    unknown = name not in VALUE_OPTIONS and name not in FLAG_OPTIONS and "=" not in token
    if unknown and index + 1 < len(argv) and _looks_like_path(argv[index + 1]):
        raise Refused(f"write {name} as {name}=value so its value is not read as a test path, or use plain pytest")
    return Option(name, (token,)), index + 1


def split_args(argv: Sequence[str]) -> Args:
    """Split command-line arguments into test paths and forwarded pytest options."""
    paths: List[str] = []
    options: List[Option] = []
    index = 0
    while index < len(argv):
        if argv[index].startswith("-") and argv[index] != "-":
            option, index = _take_option(argv, index)
            options.append(option)
        else:
            paths.append(argv[index])
            index += 1
    return Args(tuple(paths), tuple(options))


def check_mode(args: Args) -> None:
    """A full run sets its own worker count; a targeted run may choose one."""
    if args.is_full and WORKER_OPTIONS.intersection(args.names()):
        raise Refused("a full run sets its own worker count; pass test paths to choose one, or use plain pytest")


def resolve_test_paths(paths: Sequence[str], cwd: Path, tree: Path) -> List[str]:
    """Rewrite paths and node ids relative to the tree root; each must sit under tests/."""
    tests_dir = (tree / "tests").resolve()
    resolved = []
    for path in paths:
        file_part, sep, rest = path.partition("::")
        absolute = (cwd / file_part).resolve()
        if absolute != tests_dir and tests_dir not in absolute.parents:
            raise Refused(f"{path} is outside tests/; use plain pytest to run it")
        relative = absolute.relative_to(tree.resolve()).as_posix()
        resolved.append(relative + sep + rest)
    return resolved


# --- roots, interpreter, preflight, environment -----------------------------


def find_roots(cwd: Path) -> Roots:
    """The tree this command runs in, the main checkout, and the shared git directory."""
    proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel", "--git-common-dir"],
        cwd=cwd, capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        raise Refused(f"{cwd} is not inside a git checkout of this repository")
    toplevel, common = proc.stdout.splitlines()[:2]
    common_dir = (cwd / common).resolve()  # the main checkout prints a relative ".git"
    return Roots(tree=Path(toplevel).resolve(), main=common_dir.parent, common_dir=common_dir)


def find_interpreter(roots: Roots) -> Path:
    for root in (roots.tree, roots.main):
        candidate = root / ".venv" / "bin" / "python"
        if candidate.exists():
            return candidate
    return roots.main / ".venv" / "bin" / "python"


_PREFLIGHT_PROBE = r"""
import json, re, sys, tomllib
from importlib import metadata

def canon(name):
    return re.sub(r"[-_.]+", "-", name).lower()

project = tomllib.load(open(sys.argv[1], "rb"))["project"]
extras, own, wanted = project.get("optional-dependencies", {}), canon(project["name"]), set()
todo, seen = sys.argv[2].split(","), set()
while todo:
    extra = todo.pop()
    if extra in seen:
        continue
    seen.add(extra)
    for requirement in extras.get(extra, []):
        if ";" in requirement:
            continue  # environment markers are not evaluated here
        match = re.match(r"\s*([A-Za-z0-9._-]+)\s*(?:\[([^\]]*)\])?", requirement)
        if canon(match.group(1)) == own:
            todo.extend(part.strip() for part in (match.group(2) or "").split(",") if part.strip())
        else:
            wanted.add(match.group(1))

def installed(name):
    try:
        metadata.distribution(name)
        return True
    except metadata.PackageNotFoundError:
        return False

print(json.dumps(sorted(name for name in wanted if not installed(name))))
"""


def preflight(interpreter: Path, tree: Path) -> Optional[str]:
    """None when the interpreter has everything CI installs; otherwise what is wrong."""
    if not interpreter.exists():
        return f"no project venv interpreter at {interpreter}; create the venv in the main checkout"
    proc = subprocess.run(
        [str(interpreter), "-c", _PREFLIGHT_PROBE, str(tree / "pyproject.toml"), ",".join(CI_EXTRAS)],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return f"{interpreter} could not check its packages: {proc.stderr.strip()}"
    missing = json.loads(proc.stdout)
    if missing:
        listed = ", ".join(missing)
        return f"{interpreter} lacks what CI installs ({listed}); install it with:\n  {reinstall_command(interpreter)}"
    return None


def reinstall_command(interpreter: Path) -> str:
    """The venv's own checkout by absolute path: a relative path run from a worktree
    would point the shared editable install at that worktree."""
    owner = interpreter.parent.parent.parent
    return f"{interpreter} -m pip install -e '{owner}[{','.join(CI_EXTRAS)}]'"


def child_env(base: Mapping[str, str], venv_bin: Path, tree: Path) -> Dict[str, str]:
    """The environment every pytest this command starts runs with."""
    env = dict(base)
    env.pop("PYTHONDONTWRITEBYTECODE", None)  # agent shells set it; it keeps every run cold
    env["PATH"] = os.pathsep.join(filter(None, [str(venv_bin), base.get("PATH", "")]))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tree / "src"), base.get("PYTHONPATH", "")]))
    return env


# --- the machine-wide turn --------------------------------------------------


def read_holder(lock_path: Path) -> Optional[Dict[str, object]]:
    """The record the current or last holder wrote, if it can be read."""
    try:
        record = json.loads(lock_path.read_text(encoding="utf-8") or "null")
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _describe_holder(record: Optional[Mapping[str, object]], own_tree: object) -> str:
    if record is None:
        return "another run"
    who = "this tree" if record.get("tree") == own_tree else str(record.get("tree"))
    return f"{who} (pid {record.get('pid')}, {record.get('mode')}) since {record.get('started')}"


def _try_lock(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EWOULDBLOCK, errno.EACCES, errno.EAGAIN):
            return False
        raise
    return True


def acquire_turn(lock_path: Path, holder: Mapping[str, object], say: Callable[[str], None]) -> int:
    """Wait for the turn and return the descriptor that holds it.

    Polls instead of blocking so Ctrl-C ends the wait at once. The file is
    never deleted: the lock lives on the open descriptor, so a leftover file
    from a finished or killed run is harmless."""
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    started = time.monotonic()
    announced = False
    while not _try_lock(fd):
        if not announced:
            say(f"waiting for the test turn: held by {_describe_holder(read_holder(lock_path), holder.get('tree'))}")
            announced = True
        time.sleep(POLL_SECONDS)
    if announced:
        say(f"got the test turn after {time.monotonic() - started:.0f}s")
    os.ftruncate(fd, 0)
    os.pwrite(fd, (json.dumps(dict(holder)) + "\n").encode("utf-8"), 0)
    return fd


def release_turn(fd: int) -> None:
    os.close(fd)


def run_lane(command: Sequence[str], env: Mapping[str, str], cwd: Path, turn_fd: Optional[int]) -> int:
    """Run one pytest lane. The child inherits the turn's descriptor, so the turn
    lasts exactly as long as the lane does, even if this process is killed."""
    pass_fds = (turn_fd,) if turn_fd is not None else ()
    return subprocess.run(list(command), env=dict(env), cwd=cwd, pass_fds=pass_fds, check=False).returncode


# --- lanes ------------------------------------------------------------------


def serial_modules(tree: Path) -> List[str]:
    """Modules that mark tests serial, found the way tests/test_shard_lanes.py pins them."""
    found = (path for path in (tree / "tests").rglob("*.py") if SERIAL_MARK.search(path.read_text(encoding="utf-8")))
    return sorted(path.relative_to(tree).as_posix() for path in found)


def serial_targets(paths: Sequence[str], modules: Sequence[str]) -> List[str]:
    """The named paths that are, or contain, a serial module."""
    targets: List[str] = []
    for path in paths:
        file_part = path.partition("::")[0].rstrip("/")
        if file_part in modules:
            targets.append(path)
        else:
            targets.extend(module for module in modules if module.startswith(file_part + "/"))
    return list(dict.fromkeys(targets))


def should_run_serial(parallel_code: int, args: Args) -> bool:
    """The serial lane still runs after failures, so one run shows all of them."""
    if parallel_code not in (0, 1, NO_TESTS):
        return False  # interrupted, internal error or usage error
    return not (parallel_code == 1 and STOP_EARLY.intersection(args.names()))


def combine_exit_codes(parallel: int, serial: Optional[int], parallel_ran_nothing: bool) -> int:
    codes = [1 if parallel_ran_nothing else parallel] + ([] if serial is None else [serial])
    failed = [code for code in codes if code not in (0, NO_TESTS)]
    if failed:
        return failed[0]
    return 0 if 0 in codes else NO_TESTS


@dataclass(frozen=True)
class RunSpec:
    args: Args
    roots: Roots
    interpreter: Path
    env: Mapping[str, str]
    workers: int = field(default_factory=lambda: os.cpu_count() or 4)

    @property
    def lock_path(self) -> Path:
        return self.roots.common_dir / LOCK_NAME

    def holder(self, mode: str) -> Dict[str, object]:
        started = datetime.datetime.now().isoformat(timespec="seconds")
        return {"tree": str(self.roots.tree), "pid": os.getpid(), "mode": mode, "started": started}


Executor = Callable[[Sequence[str], Mapping[str, str], Path, Optional[int]], int]


def _lane_command(spec: RunSpec, lane: str, workers: List[str], node_dir: Path) -> List[str]:
    base = [str(spec.interpreter), "-m", "pytest", "-q", *workers, "--test-lane", lane, f"--node-id-dir={node_dir}"]
    forwarded = spec.args.option_tokens(drop=WORKER_OPTIONS if lane == "serial" else frozenset())
    return base + forwarded


def _ran_count(node_dir: Path, lane: str) -> int:
    ran = node_dir / f"{lane}-ran.txt"
    return len(ran.read_text(encoding="utf-8").splitlines()) if ran.exists() else 0


@dataclass
class _Run:
    spec: RunSpec
    executor: Executor
    say: Callable[[str], None]
    node_dir: Path

    def lane(self, lane: str, workers: List[str], targets: Sequence[str], turn_fd: Optional[int]) -> int:
        mode = "full run" if self.spec.args.is_full else "targeted run"
        self.say(f"{PREFIX} {mode} · {lane} lane · {workers[1] if workers else 'chosen'} workers")
        command = _lane_command(self.spec, lane, workers, self.node_dir) + list(targets)
        return self.executor(command, self.spec.env, self.spec.roots.tree, turn_fd)

    def serial_under_turn(self, targets: Sequence[str]) -> int:
        fd = acquire_turn(self.spec.lock_path, self.spec.holder("serial tests"), self.say)
        try:
            return self.lane("serial", ["-n", "0"], targets, fd)
        finally:
            release_turn(fd)


def _run_full(run: _Run) -> Tuple[int, Optional[int], bool]:
    spec = run.spec
    fd = acquire_turn(spec.lock_path, spec.holder("full run"), run.say)
    try:
        parallel = run.lane("parallel", ["-n", str(spec.workers), "--dist", "worksteal"], (), fd)
        nothing = parallel == 0 and _ran_count(run.node_dir, "parallel") == 0
        modules = serial_modules(spec.roots.tree)
        serial = None
        if modules and should_run_serial(parallel, spec.args):
            serial = run.lane("serial", ["-n", "0"], modules, fd)
    finally:
        release_turn(fd)
    return parallel, serial, nothing


def _run_targeted(run: _Run) -> Tuple[int, Optional[int], bool]:
    args = run.spec.args
    workers = [] if WORKER_OPTIONS.intersection(args.names()) else ["-n", "0"]
    parallel = run.lane("parallel", workers, args.paths, None)
    targets = serial_targets(args.paths, serial_modules(run.spec.roots.tree))
    serial = run.serial_under_turn(targets) if targets and should_run_serial(parallel, args) else None
    return parallel, serial, False


def _summarize(run: _Run, codes: Tuple[int, Optional[int], bool], elapsed: float) -> None:
    parallel, serial, nothing = codes
    ran = _ran_count(run.node_dir, "parallel")
    parts = [f"parallel lane exit {parallel}, {ran} ran"]
    if nothing:
        parts.append("the parallel lane ran no tests although pytest exited 0")
    if serial is None:
        parts.append("serial lane not run")
    else:
        parts.append(f"serial lane exit {serial}, {_ran_count(run.node_dir, 'serial')} ran")
    if run.spec.args.narrowing():
        parts.append(f"selection narrowed by {' '.join(run.spec.args.narrowing())}")
    run.say(f"{PREFIX} done in {elapsed:.0f}s · " + " · ".join(parts))


def execute(spec: RunSpec, executor: Executor = run_lane, say: Callable[[str], None] = print) -> int:
    """Run the lanes a full or targeted run needs and return one exit code."""
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="ccs-local-tests-") as node_dir:
        run = _Run(spec, executor, say, Path(node_dir))
        codes = _run_full(run) if spec.args.is_full else _run_targeted(run)
        _summarize(run, codes, time.monotonic() - started)
    return combine_exit_codes(*codes)


# --- entry point ------------------------------------------------------------


class PreflightFailed(Exception):
    """The venv is missing or lacks what CI installs."""


def prepare(argv: Sequence[str], cwd: Path) -> RunSpec:
    """Check the arguments and the venv, and describe the run; nothing runs yet."""
    args = split_args(argv)
    check_mode(args)
    roots = find_roots(cwd)
    args = Args(tuple(resolve_test_paths(args.paths, cwd, roots.tree)), args.options)
    interpreter = find_interpreter(roots)
    problem = preflight(interpreter, roots.tree)
    if problem:
        raise PreflightFailed(problem)
    env = child_env(os.environ, interpreter.parent, roots.tree)
    return RunSpec(args=args, roots=roots, interpreter=interpreter, env=env)


def _say(line: str) -> None:
    print(line, flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        spec = prepare(sys.argv[1:] if argv is None else argv, Path.cwd())
    except Refused as exc:
        print(f"{PREFIX} refused: {exc}", file=sys.stderr)
        return USAGE_ERROR
    except PreflightFailed as exc:
        print(f"{PREFIX} {exc}", file=sys.stderr)
        return PREFLIGHT_FAILED
    try:
        return execute(spec, run_lane, _say)
    except KeyboardInterrupt:
        print(f"{PREFIX} interrupted", file=sys.stderr)
        return INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
