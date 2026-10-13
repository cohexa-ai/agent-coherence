# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Which Node coordinator the corpus runs: an explicit path is never replaced.

``resolve_node_dist_path`` used to ignore an ``AGENT_COHERENCE_PLUGIN_DIST_PATH``
that did not exist and fall back to ``../agent-coherence-plugin`` or
``~/projects/agent-coherence-plugin``. A mistyped path therefore ran every Node
row against whatever dist a fallback checkout happened to hold — on a
developer machine, the live sibling — and reported green for a dist nobody
asked to test. Set, the variable is now the only candidate; unset, the
fallbacks resolve exactly as before.

Not marked ``protocol_corpus``: none of this needs a Node build, so it runs
in the default suite."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.protocol_corpus import harness

_ENV = "AGENT_COHERENCE_PLUGIN_DIST_PATH"
"""FROZEN duplicate of the variable's name, never imported from the harness: a
derived name would follow a rename instead of catching it."""

#: The corpus modules that resolve the dist at import. A frozen SUBSET check:
#: the glob below may find more, never fewer.
_CORPUS_MODULES = frozenset({
    "test_caller_principal_corpus.py",
    "test_effect_fence_corpus.py",
    "test_session_start_corpus.py",
    "test_strict_mode_corpus.py",
    "test_warn_mode_corpus.py",
})


def _place_dist(root: Path) -> Path:
    dist = root / "agent-coherence-plugin" / "dist" / "coordinator.js"
    dist.parent.mkdir(parents=True)
    dist.write_text("// stand-in: resolution only checks that the path exists\n")
    return dist.resolve()


@pytest.fixture
def fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Both fallback checkouts PRESENT, so a resolver that falls back has
    somewhere to land: the sibling of a stand-in repo root, and
    ``~/projects`` under a stand-in HOME. Neither is the machine's own, so the
    live sibling checkout can never be what a test sees."""
    work = tmp_path / "work"
    repo_root = work / "agent-coherence"
    repo_root.mkdir(parents=True)
    monkeypatch.setattr(harness, "REPO_ROOT", repo_root)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(_ENV, raising=False)
    return _place_dist(work), _place_dist(home / "projects")


def test_an_explicit_dist_path_that_does_not_exist_raises_instead_of_falling_back(
    fallbacks: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sibling, _ = fallbacks
    assert harness.resolve_node_dist_path() == sibling, (
        "control: unset, the same setup resolves to the sibling fallback"
    )
    missing = tmp_path / "typo" / "dist" / "coordinator.js"
    monkeypatch.setenv(_ENV, str(missing))

    with pytest.raises(FileNotFoundError) as refused:
        harness.resolve_node_dist_path()

    assert _ENV in str(refused.value), str(refused.value)
    assert repr(str(missing)) in str(refused.value), str(refused.value)


def test_an_explicit_dist_path_set_empty_raises_instead_of_falling_back(
    fallbacks: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Set but empty is still set — the shape a CI expression that expanded to
    nothing leaves behind — and an empty path resolves to the working
    directory, which exists."""
    monkeypatch.setenv(_ENV, "")

    with pytest.raises(FileNotFoundError) as refused:
        harness.resolve_node_dist_path()

    assert f"{_ENV} is set to ''" in str(refused.value), str(refused.value)


def test_an_existing_explicit_dist_path_wins_over_both_fallbacks(
    fallbacks: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chosen = _place_dist(tmp_path / "chosen")
    assert chosen not in fallbacks
    monkeypatch.setenv(_ENV, str(chosen))

    assert harness.resolve_node_dist_path() == chosen


def test_unset_resolves_the_sibling_then_home_then_nothing(
    fallbacks: tuple[Path, Path],
) -> None:
    """The unset case keeps its resolution order, down to ``None`` — which the
    parity corpora turn into an xfail and the asymmetry corpora into a fail."""
    sibling, home_dist = fallbacks
    assert harness.resolve_node_dist_path() == sibling
    sibling.unlink()
    assert harness.resolve_node_dist_path() == home_dist
    home_dist.unlink()
    assert harness.resolve_node_dist_path() is None


def _collect_corpus(env_value: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    corpus_dir = Path(harness.__file__).resolve().parent
    modules = sorted(p.name for p in corpus_dir.glob("test_*_corpus.py"))
    assert _CORPUS_MODULES <= set(modules), sorted(_CORPUS_MODULES - set(modules))
    env = {**os.environ, _ENV: env_value}
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "-m", "protocol_corpus", *(str(corpus_dir / m) for m in modules)],
        capture_output=True, text=True, cwd=harness.REPO_ROOT, env=env, timeout=120,
    )
    return proc, modules


def test_a_missing_explicit_dist_path_stops_every_corpus_module_at_collection(
    tmp_path: Path,
) -> None:
    """Loud, not skipped: each corpus module resolves at import, so the refusal
    is a collection error in every one of them and no row is collected, let
    alone run against a fallback. Control: an existing path collects them all."""
    stand_in = _place_dist(tmp_path / "stand-in")
    proc, modules = _collect_corpus(str(stand_in))
    assert proc.returncode == 0, f"control did not collect:\n{proc.stdout}\n{proc.stderr}"
    collected = {line.split("::", 1)[0].rsplit("/", 1)[-1]
                 for line in proc.stdout.splitlines() if "::" in line}
    assert collected == set(modules), sorted(set(modules) - collected)

    missing = tmp_path / "typo" / "coordinator.js"
    proc, modules = _collect_corpus(str(missing))

    out = proc.stdout + proc.stderr
    assert proc.returncode == pytest.ExitCode.INTERRUPTED, out
    assert "::" not in proc.stdout, f"a row was collected:\n{out}"
    sections = dict(
        (chunk.split(" ", 1)[0].rsplit("/", 1)[-1], chunk)
        for chunk in out.split("ERROR collecting ")[1:]
    )
    for module in modules:
        section = sections.get(module, "")
        assert "FileNotFoundError" in section and f"{_ENV} is set to" in section, (
            f"{module} did not stop at collection on the missing dist:\n{out}"
        )
        assert str(missing) in section, section
