# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for tools/dependabot_gate.py — which Dependabot updates may merge themselves.

The gate's verdict is the only thing standing between a Dependabot pull request
and an unattended merge into dev, and CI cannot stand in for it: CI installs
from pyproject.toml, never from uv.lock, so a lockfile-only update that drags
other packages down (as the mcp 1.30.0 update did to crewai and json-repair)
passes every check. Each case asserts the exit code AND the printed reason, and
"cannot tell" (exit 2) is pinned apart from "refuse" (exit 1) and never comes
back as "qualifies" (exit 0). ``tools/`` is not an importable package, so the
module is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
MINOR = "version-update:semver-minor"
FIXTURE = "examples/refactor_demo/fixture_repo_ts"


def _import_gate():
    spec = importlib.util.spec_from_file_location("dependabot_gate", _REPO_ROOT / "tools" / "dependabot_gate.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves a class's module through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gate():
    return _import_gate()


def _lock(*packages: tuple[str, str], extra: str = "") -> str:
    """A uv.lock holding the project (dynamically versioned) and these packages."""
    blocks = ['version = 1\n\n[[package]]\nname = "agent-coherence"\nsource = { editable = "." }\n']
    blocks += [f'[[package]]\nname = "{name}"\nversion = "{version}"\n' for name, version in packages]
    return "\n".join(blocks) + extra


OLD_PYTHON, NEW_PYTHON = "python_full_version < '3.12'", "python_full_version >= '3.12'"


def _fork(name: str, *forks: tuple[str, str]) -> str:
    """Lock entries for one package locked per Python-version fork: (version, marker)."""
    return "".join(
        f'\n[[package]]\nname = "{name}"\nversion = "{version}"\nresolution-markers = ["{marker}"]\n'
        for version, marker in forks
    )


# The shape of the real regression: mcp moved up, and to keep it the resolver
# moved crewai and json-repair down and dropped crewai's newer dependencies.
BASE_LOCK = _lock(("crewai", "1.15.27"), ("json-repair", "0.60.1"), ("lancedb", "0.30.0"), ("mcp", "1.28.1"))
DOWNGRADE_LOCK = _lock(("crewai", "1.6.1"), ("json-repair", "0.25.2"), ("mcp", "1.30.0"))
UPGRADE_LOCK = _lock(("crewai", "1.15.27"), ("json-repair", "0.60.1"), ("lancedb", "0.30.0"), ("mcp", "1.28.2"))


@pytest.mark.parametrize(
    ("ecosystem", "update_type", "group", "files", "reason"),
    [
        ("github_actions", MINOR, "", (".github/workflows/ci.yml",), "github_actions update"),
        ("", MINOR, "", ("uv.lock",), "unknown ecosystem update"),
        ("pip", MINOR, "", ("uv.lock",), "pip update"),
        ("uv", MINOR, "uv", ("uv.lock",), "grouped update (uv)"),
        ("uv", "version-update:semver-major", "", ("uv.lock",), "version-update:semver-major update"),
        ("uv", "", "", ("uv.lock",), "unclassified update"),
        ("uv", MINOR, "", (), "no changed files"),
        ("uv", MINOR, "", ("uv.lock", "pyproject.toml"), "changes pyproject.toml"),
        ("uv", MINOR, "", ("uv.lock", "tools/dependabot_gate.py"), "changes tools/dependabot_gate.py"),
        ("npm_and_yarn", MINOR, "", (f"{FIXTURE}/package.json", "uv.lock"), "changes uv.lock"),
    ],
)
def test_wait_reason_names_why_a_person_decides(gate, ecosystem, update_type, group, files, reason) -> None:
    """Anything outside the narrow allow-list waits: an unknown ecosystem or an
    empty update type is never read as harmless."""
    assert gate.wait_reason(gate.Update(ecosystem, update_type, group, files)) == reason


@pytest.mark.parametrize(
    ("ecosystem", "update_type", "files"),
    [
        ("uv", MINOR, ("uv.lock",)),
        ("uv", "version-update:semver-patch", ("uv.lock", "requirements-diagnose.txt")),
        ("npm_and_yarn", MINOR, (f"{FIXTURE}/package.json", f"{FIXTURE}/package-lock.json")),
    ],
)
def test_wait_reason_lets_a_lock_or_fixture_patch_or_minor_through(gate, ecosystem, update_type, files) -> None:
    assert gate.wait_reason(gate.Update(ecosystem, update_type, "", files)) is None


def test_lock_regressions_names_each_package_moved_down_and_each_dropped(gate) -> None:
    assert gate.lock_regressions(BASE_LOCK, DOWNGRADE_LOCK) == (
        ["crewai 1.15.27 -> 1.6.1", "json-repair 0.60.1 -> 0.25.2"],
        ["lancedb 0.30.0"],
    )


def test_lock_regressions_accepts_moves_up_and_additions(gate) -> None:
    added = UPGRADE_LOCK + '\n[[package]]\nname = "websockets"\nversion = "16.0"\n'
    assert gate.lock_regressions(BASE_LOCK, added) == ([], [])


def test_lock_regressions_compares_versions_not_strings(gate) -> None:
    """Version order, not text order: "1.10.0" sorts below "1.9.0" as text."""
    assert gate.lock_regressions(_lock(("x", "1.9.0")), _lock(("x", "1.10.0"))) == ([], [])
    assert gate.lock_regressions(_lock(("x", "1.10.0")), _lock(("x", "1.9.0"))) == (["x 1.10.0 -> 1.9.0"], [])


@pytest.mark.parametrize(
    "order",
    [(("1.26.4", OLD_PYTHON), ("2.1.0", NEW_PYTHON)), (("2.1.0", NEW_PYTHON), ("1.26.4", OLD_PYTHON))],
    ids=["ascending", "descending"],
)
def test_dropping_the_older_fork_of_a_package_is_not_a_downgrade(gate, order) -> None:
    """uv locks one package at several versions, keyed by resolution markers,
    when Python versions need different ones."""
    forked = _lock(extra=_fork("numpy", *order))
    assert gate.lock_regressions(forked, _lock(("numpy", "2.1.0"))) == ([], [])
    assert gate.lock_regressions(forked, _lock(("numpy", "1.26.4"))) == (["numpy 1.26.4, 2.1.0 -> 1.26.4"], [])


def test_a_downgrade_inside_one_fork_is_a_downgrade(gate) -> None:
    """A split that keeps the newest version for new Pythons can still drag
    old Pythons back to a vulnerable release; the highest version alone
    would hide it."""
    base = _lock(("json-repair", "0.60.1"))
    split = _lock(extra=_fork("json-repair", ("0.25.2", OLD_PYTHON), ("0.61.0", NEW_PYTHON)))
    assert gate.lock_regressions(base, split) == (["json-repair 0.60.1 -> 0.25.2, 0.61.0"], [])


def test_a_downgrade_in_a_shared_fork_is_a_downgrade_even_when_the_range_holds(gate) -> None:
    """Swapping two forks' versions leaves the lowest and highest unchanged;
    each fork is compared with its own counterpart."""
    base = _lock(extra=_fork("x", ("1.0", OLD_PYTHON), ("2.0", NEW_PYTHON)))
    swapped = _lock(extra=_fork("x", ("2.0", OLD_PYTHON), ("1.0", NEW_PYTHON)))
    assert gate.lock_regressions(base, swapped) == (["x 1.0, 2.0 -> 1.0, 2.0"], [])


def _depends(name: str, version: str, *needs: str) -> str:
    """A lock entry for ``name`` that depends on each of ``needs`` ("pkg" or "pkg==ver")."""
    deps = ", ".join(
        f'{{ name = "{need.split("==")[0]}", version = "{need.split("==")[1]}" }}'
        if "==" in need
        else f'{{ name = "{need}" }}'
        for need in needs
    )
    return f'\n[[package]]\nname = "{name}"\nversion = "{version}"\ndependencies = [{deps}]\n'


PYPI_SOURCE = 'source = { registry = "https://pypi.org/simple" }\n'


def _sourced(name: str, version: str, source: str = PYPI_SOURCE) -> str:
    return f'\n[[package]]\nname = "{name}"\nversion = "{version}"\n{source}'


def test_judge_lock_lets_a_package_that_changes_its_source_wait(gate) -> None:
    """A version bump keeps the registry; a package suddenly fetched from
    elsewhere is a person's call."""
    base = _lock(extra=_sourced("x", "1.0"))
    moved = _lock(extra=_sourced("x", "1.1", 'source = { registry = "https://mirror.example/simple" }\n'))
    assert gate.judge_lock(base, moved) == (10, "waits for a person: uv.lock takes 1 from a new source: x")


def test_judge_lock_lets_a_new_package_from_outside_pypi_wait(gate) -> None:
    base = _lock(extra=_sourced("x", "1.0"))
    added = base + _sourced("y", "0.1", 'source = { git = "https://example.invalid/y.git" }\n')
    assert gate.judge_lock(base, added) == (10, "waits for a person: uv.lock takes 1 from a new source: y")


def test_judge_lock_qualifies_a_new_package_from_pypi(gate) -> None:
    base = _lock(extra=_sourced("x", "1.0"))
    assert gate.judge_lock(base, base + _sourced("y", "0.1")) == (0, "qualifies")


def test_unlocked_dependencies_finds_a_dependency_the_lock_no_longer_holds(gate) -> None:
    """A git merge of two lock updates can keep q's dependency on s while the
    other side removed s; no resolver would write that lock."""
    merged = _lock(("r", "1.1"), extra=_depends("q", "1.1", "s"))
    assert gate.unlocked_dependencies(merged) == ["s"]


def test_unlocked_dependencies_checks_a_named_version_and_optional_groups(gate) -> None:
    optional = '\n[[package]]\nname = "p"\nversion = "1.0"\n[package.optional-dependencies]\nx = [{ name = "t" }]\n'
    lock = _lock(("s", "1.0"), extra=_depends("q", "1.1", "s==2.0") + optional)
    assert gate.unlocked_dependencies(lock) == ["s 2.0", "t"]


def test_unlocked_dependencies_accepts_a_complete_lock(gate) -> None:
    lock = _lock(("s", "1.0"), extra=_depends("q", "1.1", "s", "s==1.0"))
    assert gate.unlocked_dependencies(lock) == []


@pytest.mark.parametrize(
    "text",
    [
        "not = [valid toml",
        "version = 1\n",
        _lock(("x", "not a version")),
        _lock(("x\nstate=success", "1.0")).replace("x\nstate", "x\\nstate"),
        _lock(("x", "1.0"), extra='\n[[package]]\nname = "q"\nversion = "1.0"\ndependencies = [{ name = "s t" }]\n'),
    ],
    ids=["broken toml", "no packages", "unreadable version", "line break in a name", "space in a dependency name"],
)
def test_an_unreadable_lock_cannot_tell(gate, text) -> None:
    with pytest.raises(gate.CannotTell):
        gate.lock_regressions(text, BASE_LOCK)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _repo_with(tmp_path: Path, base: dict[str, str], head: dict[str, str | None]) -> tuple[Path, str, str]:
    """A git repo with a base commit and a head commit on top of it; a None
    value in ``head`` deletes that file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    for files in (base, head):
        for path, text in files.items():
            target = repo / path
            if text is None:
                target.unlink()
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "--allow-empty", "-m", "commit")
    return repo, _git(repo, "rev-parse", "HEAD~1"), _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def run_gate(gate, capsys, monkeypatch):
    """Run the gate's CLI in ``repo`` and return (exit code, printed reason)."""

    def run(repo: Path, revs: tuple[str, str], update: tuple[str, str, str]) -> tuple[int, str]:
        monkeypatch.chdir(repo)
        ecosystem, update_type, group = update
        argv = ["--base", revs[0], "--head", revs[1], "--ecosystem", ecosystem, "--update-type", update_type]
        code = gate.main([*argv, "--group", group])
        return code, capsys.readouterr().out.strip()

    return run


def test_main_qualifies_a_lock_update_that_moves_only_up(run_gate, tmp_path) -> None:
    repo, base, head = _repo_with(tmp_path, {"uv.lock": BASE_LOCK}, {"uv.lock": UPGRADE_LOCK})
    assert run_gate(repo, (base, head), ("uv", MINOR, "")) == (0, "qualifies")


def test_main_refuses_a_lock_update_that_moves_packages_down(run_gate, tmp_path) -> None:
    repo, base, head = _repo_with(tmp_path, {"uv.lock": BASE_LOCK}, {"uv.lock": DOWNGRADE_LOCK})
    code, reason = run_gate(repo, (base, head), ("uv", MINOR, ""))
    assert code == 1
    assert reason == "uv.lock lowers 2: crewai 1.15.27 -> 1.6.1, json-repair 0.60.1 -> 0.25.2"


def test_main_lets_a_lock_update_that_only_drops_a_package_wait(run_gate, tmp_path) -> None:
    """Upgrades drop dependencies they no longer need; that is a person's call, not a red gate."""
    dropped = _lock(("crewai", "1.15.27"), ("json-repair", "0.60.1"), ("mcp", "1.28.2"))
    repo, base, head = _repo_with(tmp_path, {"uv.lock": BASE_LOCK}, {"uv.lock": dropped})
    assert run_gate(repo, (base, head), ("uv", MINOR, "")) == (
        10,
        "waits for a person: uv.lock removes 1: lancedb 0.30.0",
    )


def test_main_refuses_a_lock_that_depends_on_a_package_it_does_not_lock(run_gate, tmp_path) -> None:
    unlocked = UPGRADE_LOCK + _depends("q", "1.1", "s")
    repo, base, head = _repo_with(tmp_path, {"uv.lock": BASE_LOCK}, {"uv.lock": unlocked})
    assert run_gate(repo, (base, head), ("uv", MINOR, "")) == (1, "uv.lock depends on 1 it does not lock: s")


def test_main_lets_a_major_wait_without_reading_the_lock(run_gate, tmp_path) -> None:
    """A major waits for a person whatever the lock says; that is not an error."""
    repo, base, head = _repo_with(tmp_path, {"uv.lock": BASE_LOCK}, {"uv.lock": DOWNGRADE_LOCK})
    major = ("uv", "version-update:semver-major", "")
    assert run_gate(repo, (base, head), major) == (
        10,
        "waits for a person: version-update:semver-major update",
    )


def test_main_qualifies_a_fixture_update_without_a_lock(run_gate, tmp_path) -> None:
    files = {f"{FIXTURE}/package-lock.json": '{"v": 1}'}
    repo, base, head = _repo_with(tmp_path, files, {f"{FIXTURE}/package-lock.json": '{"v": 2}'})
    assert run_gate(repo, (base, head), ("npm_and_yarn", MINOR, "")) == (0, "qualifies")


def test_main_cannot_tell_when_the_head_drops_the_lock(run_gate, tmp_path) -> None:
    repo, base, head = _repo_with(tmp_path, {"uv.lock": BASE_LOCK}, {"uv.lock": None})
    code, reason = run_gate(repo, (base, head), ("uv", MINOR, ""))
    assert code == 2
    assert reason.startswith("cannot tell: git show ")


def test_main_cannot_tell_when_a_revision_is_missing(run_gate, tmp_path) -> None:
    repo, _base, head = _repo_with(tmp_path, {"uv.lock": BASE_LOCK}, {"uv.lock": UPGRADE_LOCK})
    code, reason = run_gate(repo, ("0" * 40, head), ("uv", MINOR, ""))
    assert code == 2
    assert reason.startswith("cannot tell: git diff ")
