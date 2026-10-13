# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The "Plan checks" job's decisions, run the way CI runs them.

Every other job reads the plan job's outputs, and every skip they allow needs
an explicit value. So the script must write every output key on every run and
never exit non-zero: a missing or empty value makes consumers run everything,
which is the safe direction, while a plan job that failed would leave its
consumers to GitHub's cascade rules instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "ci_plan.py"
OUTPUT_KEYS = frozenset({"tested-tree", "tla-needed"})
BASE_FILES = (
    "formal/tla/OCC.tla",
    "formal/tla/OCC_CI.cfg",
    "Makefile",
    ".github/workflows/ci.yml",
    "src/ccs/x.py",
    "tests/test_x.py",
    "docs/guide.md",
)


def _run_plan(
    tmp_path: Path, env_overrides: dict[str, str] | None = None, cwd: Path = REPO_ROOT
) -> tuple[int, dict[str, str]]:
    output = tmp_path / "github_output"
    output.write_text("")
    env = {**os.environ, "GITHUB_OUTPUT": str(output), **(env_overrides or {})}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd=cwd, env=env, capture_output=True, text=True, timeout=60, check=False
    )
    values = dict(line.split("=", 1) for line in output.read_text().splitlines() if "=" in line)
    return proc.returncode, values


def test_the_tested_tree_is_the_checked_out_tree(tmp_path: Path) -> None:
    """The tree id names exactly what this run tests, whatever commit or merge
    carried it; a later push can recognise an identical tree by it."""
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    exit_code, values = _run_plan(tmp_path)
    assert exit_code == 0
    assert set(values) == OUTPUT_KEYS
    assert values["tested-tree"] == expected


def test_an_error_still_writes_every_key_and_exits_zero(tmp_path: Path) -> None:
    """With ``git`` unreachable the plan cannot decide anything, so it writes
    every key with its run-everything value instead of failing: no tested
    tree (nothing can match it) and TLA+ needed."""
    exit_code, values = _run_plan(
        tmp_path, {"PATH": str(tmp_path / "no-git-here"), "GITHUB_EVENT_NAME": "pull_request"}
    )
    assert exit_code == 0
    assert values == {"tested-tree": "", "tla-needed": "true"}


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True)


def _pull_request_checkout(
    tmp_path: Path, written: tuple[str, ...] = (), moved: tuple[tuple[str, str], ...] = (), *, depth: int = 2
) -> Path:
    """What the plan job checks out on a pull request: the merge commit of the
    PR branch into a base that moved on since it branched, cloned at the
    workflow's ``fetch-depth``. Its first parent is the base, so ``HEAD^1``
    to ``HEAD`` is exactly what the PR changes."""
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init", "-q", "-b", "dev")
    _git(source, "config", "user.email", "t@example.com")
    _git(source, "config", "user.name", "t")
    for name in BASE_FILES:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text("base\n")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "base")
    _git(source, "checkout", "-qb", "pr")
    for name in written:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_text("changed\n")
    for old, new in moved:
        (source / new).parent.mkdir(parents=True, exist_ok=True)
        _git(source, "mv", old, new)
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "change", "--allow-empty")
    _git(source, "checkout", "-q", "dev")
    (source / "src" / "ccs" / "landed_meanwhile.py").write_text("base moved on\n")
    _git(source, "add", "-A")
    _git(source, "commit", "-qm", "base moves on")
    _git(source, "merge", "-q", "--no-ff", "pr", "-m", "merge pr into dev")
    checkout = tmp_path / "checkout"
    _git(tmp_path, "clone", "-q", f"--depth={depth}", source.resolve().as_uri(), str(checkout))
    return checkout


@pytest.mark.parametrize(
    ("written", "needed"),
    [
        pytest.param(("src/ccs/x.py", "tests/test_x.py"), "false", id="code-and-tests-only"),
        pytest.param(("docs/guide.md",), "false", id="docs-only"),
        pytest.param(("formal/tla/OCC_CI.cfg",), "true", id="a-model-config"),
        pytest.param(("formal/tla/NewSpec.tla",), "true", id="a-new-spec"),
        pytest.param(("Makefile",), "true", id="the-makefile"),
        # GNU make reads GNUmakefile, then makefile, before Makefile, so adding
        # either changes what `make tla-check-N` runs. (On a case-insensitive
        # filesystem the lowercase case edits Makefile, which needs TLA+ too.)
        pytest.param(("GNUmakefile",), "true", id="a-gnumakefile"),
        pytest.param(("makefile",), "true", id="a-lowercase-makefile"),
        pytest.param((".github/workflows/ci.yml",), "true", id="the-workflow"),
        pytest.param((".github/actions/setup/action.yml",), "true", id="a-composite-action"),
        pytest.param(("tools/ci_plan.py",), "true", id="the-plan-script"),
        pytest.param(("tools/ci_gate_verdict.py",), "true", id="the-gate-script"),
        pytest.param((), "true", id="an-empty-diff"),
        # git lists --name-only paths sorted, so a TLA+ input sorts both
        # before and after the other paths here.
        pytest.param(("docs/guide.md", "formal/tla/OCC.tla"), "true", id="docs-and-a-spec"),
        pytest.param(("formal/tla/OCC.tla", "src/ccs/x.py"), "true", id="a-spec-and-code"),
        pytest.param(("src/ccs/x.py", "tests/test_x.py", "Makefile"), "true", id="code-tests-and-the-makefile"),
        pytest.param(("formal/tla/Café.tla",), "true", id="a-non-ascii-spec-path"),
    ],
)
def test_a_pull_request_needs_tla_only_when_a_tla_input_changed(tmp_path: Path, written, needed) -> None:
    """The bins may skip only on a pull request that provably left every TLA+
    input alone: the specs and configs (specs extend each other, so the whole
    directory), the Makefile that runs them, anything under .github, and the
    two scripts that decide and judge the run. A path git would quote is still
    an input; an empty diff proves nothing, so it runs them."""
    checkout = _pull_request_checkout(tmp_path, written)
    exit_code, values = _run_plan(tmp_path, {"GITHUB_EVENT_NAME": "pull_request"}, cwd=checkout)
    assert exit_code == 0
    assert values["tla-needed"] == needed


@pytest.mark.parametrize(
    "moved",
    [
        pytest.param(("formal/tla/OCC.tla", "formal/common/OCC.tla"), id="a-spec-moved-out"),
        pytest.param((".github/workflows/ci.yml", "ci/ci.yml"), id="the-workflow-moved-out"),
        pytest.param(("Makefile", "GNUmakefile"), id="the-makefile-renamed"),
    ],
)
def test_a_pull_request_that_moves_an_input_away_needs_tla(tmp_path: Path, moved) -> None:
    """Moving a file out of an input path changes that input: the file the
    bins depend on is gone from where they expect it, so they must run even
    though nothing under the input path was added or edited."""
    checkout = _pull_request_checkout(tmp_path, moved=(moved,))
    exit_code, values = _run_plan(tmp_path, {"GITHUB_EVENT_NAME": "pull_request"}, cwd=checkout)
    assert exit_code == 0
    assert values["tla-needed"] == "true"


@pytest.fixture(scope="module")
def _code_only_checkout(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One code-only pull request, read by every event below; as a pull request it would skip."""
    return _pull_request_checkout(tmp_path_factory.mktemp("code-only"), ("src/ccs/x.py",))


@pytest.mark.parametrize("event", ["push", "workflow_dispatch", "schedule", ""])
def test_every_other_event_needs_tla_whatever_the_diff(tmp_path: Path, _code_only_checkout: Path, event) -> None:
    """Only a pull request can skip the model check; a push to main or dev, a
    tag, a dispatch and the weekly run always run it."""
    exit_code, values = _run_plan(tmp_path, {"GITHUB_EVENT_NAME": event}, cwd=_code_only_checkout)
    assert exit_code == 0
    assert values["tla-needed"] == "true"


def test_a_checkout_without_the_base_still_decides_the_tree(tmp_path: Path) -> None:
    """At depth 1 the base commit is missing, so the diff fails and TLA+ is
    needed; the tested tree does not depend on the base and is still decided.
    One undecidable output must not take the others with it."""
    checkout = _pull_request_checkout(tmp_path, ("src/ccs/x.py",), depth=1)
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=checkout, capture_output=True, text=True, check=True
    ).stdout.strip()
    exit_code, values = _run_plan(tmp_path, {"GITHUB_EVENT_NAME": "pull_request"}, cwd=checkout)
    assert exit_code == 0
    assert values == {"tested-tree": expected, "tla-needed": "true"}
