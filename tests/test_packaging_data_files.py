# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Guard: every non-Python asset under ``src/ccs/`` ships in the distribution.

``ccs/output/report.py`` and ``ccs/diagnose/render.py`` read their HTML
templates off the filesystem next to the module. Nothing in the import system
notices when such a file is missing from the wheel — the failure surfaces only
at render time, in the installed-from-PyPI environment, as a bare
``FileNotFoundError``. The repo's own test run never sees it, because a source
checkout always has the file.

So this module asserts the packaging *declaration* instead of the built
artifact: it replays setuptools' own ``build_py.find_data_files`` algorithm
against ``pyproject.toml`` and fails if any asset in the tree is left
undeclared. That runs in a plain ``pytest -q`` with no wheel build and no
setuptools import; ``test_replayed_data_files_match_setuptools`` pins the
replay to the real backend wherever setuptools happens to be importable.
"""

from __future__ import annotations

import fnmatch
import os
import subprocess
import sys
import tomllib
from glob import glob
from pathlib import Path

import pytest

#: This module's own object, for the tests that monkeypatch its helpers.
#: ``sys.modules[__name__]`` rather than a dotted re-import: tests/ is not a
#: package, and a re-import would hand the patches a different module object
#: than the one under test.
mod = sys.modules[__name__]

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
SRC_ROOT = REPO_ROOT / "src"

# setuptools always folds these two in on top of the declared patterns
# (``setuptools.command.build_py._IMPLICIT_DATA_FILES``), so a ``py.typed``
# added later is covered without anyone touching pyproject.toml.
_IMPLICIT_DATA_PATTERNS = ("*.pyi", "py.typed")


def _iter_packages(src_root: Path) -> list[str]:
    """Dotted names of every directory setuptools would treat as a package.

    ``[tool.setuptools.packages.find]`` runs with ``namespaces = true`` by
    default under pyproject.toml, so discovery is NOT gated on ``__init__.py``
    — ``ccs.output.templates`` is itself a (namespace) package. Enumerating
    every directory therefore matches the backend, and keeps this replay from
    rejecting a valid ``"ccs.output.templates" = ["*.html"]`` declaration.
    ``test_replayed_data_files_match_setuptools`` is what holds that claim
    honest: if setuptools disagreed about any package, the two resolved sets
    would part ways there.
    """
    return sorted(
        str(directory.relative_to(src_root)).replace(os.sep, ".")
        for directory in src_root.glob("**/")
        if directory != src_root
        and "__pycache__" not in directory.parts
        # `<name>.egg-info` appears under src/ the moment anyone runs
        # `pip install -e .`. setuptools does not discover it as a package, and
        # counting it is fail-open in direction: an invented package only ever
        # ADDS to the declared set, shrinking the undeclared difference this
        # module reports.
        and not any(part.endswith(".egg-info") for part in directory.parts)
    )


def _patterns_for(spec: dict[str, list[str]], package: str, implicit: tuple[str, ...] = ()) -> list[str]:
    """Patterns ``spec`` applies to ``package``, joined to its directory.

    ``implicit`` carries setuptools' built-in includes; the exclude side
    passes none, matching ``exclude_data_files``, which does not fold them in.
    """
    package_dir = SRC_ROOT / Path(*package.split("."))
    raw = [*implicit, *spec.get("", []), *spec.get(package, [])]
    return [os.path.join(str(package_dir), *pattern.split("/")) for pattern in raw]


def _included_files(patterns: list[str]) -> set[str]:
    """Files matched by include patterns — setuptools' ``find_data_files``.

    ``glob(..., recursive=True)`` is the call it makes, so ``*`` stops at a
    path separator and only an explicit ``**`` descends. That is precisely why
    a bare ``"*"`` never reaches ``templates/comparison_report.html``.
    """
    return {hit for pattern in patterns for hit in glob(pattern, recursive=True) if os.path.isfile(hit)}


def _excluded_files(patterns: list[str], candidates: set[str]) -> set[str]:
    """Files dropped by exclude patterns — setuptools' ``exclude_data_files``.

    Note the asymmetry with the include side: excludes are ``fnmatch``ed
    against the already-globbed paths rather than re-globbed, so here a ``*``
    *does* cross a path separator.
    """
    return {hit for pattern in patterns for hit in fnmatch.filter(sorted(candidates), pattern)}


def _declared_data_files() -> set[Path]:
    """Every file the packaging config declares as package data."""
    config = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    setuptools_table = config.get("tool", {}).get("setuptools", {})
    package_data = setuptools_table.get("package-data", {})
    exclude_data = setuptools_table.get("exclude-package-data", {})

    declared: set[Path] = set()
    for package in _iter_packages(SRC_ROOT):
        included = _included_files(_patterns_for(package_data, package, _IMPLICIT_DATA_PATTERNS))
        declared |= {Path(hit) for hit in included - _excluded_files(_patterns_for(exclude_data, package), included)}
    return declared


def _git_toplevel(repo_root: Path) -> Path | None:
    """The working-tree root git resolves for ``repo_root``, or None.

    ``git -C <dir>`` searches UPWARD for a ``.git``, so a checkout sitting
    inside another repository — an unpacked sdist under someone's ``build/``,
    a vendored copy — is answered by THAT repository. Its ignore rules then
    describe a tree they know nothing about. For a git worktree this correctly
    returns the worktree root rather than the main checkout.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    resolved = completed.stdout.strip()
    return Path(resolved).resolve() if resolved else None


def _tracked(repo_root: Path, source: str) -> bool:
    """Is `source` — the file an ignore rule came from — part of the repo?

    A rule in a tracked ``.gitignore`` is the repository's own decision and
    applies to everyone; a rule in ``.git/info/exclude`` or a ``core.excludesFile``
    exists only on this machine. That is the whole distinction, so ask git
    rather than pattern-matching the path.
    """
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "ls-files", "--error-unmatch", "--", source],
            capture_output=True, text=True, timeout=30,
        ).returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        # Unknown provenance: treat as the repo's own rule, so the candidate is
        # NOT subtracted and the guard stays strict.
        return True


def _locally_ignored_paths(
    candidates: "set[Path] | frozenset[Path]", *, repo_root: Path | None = None
) -> set[Path]:
    """The candidates hidden by a DEVELOPER-LOCAL ignore rule, and only those.

    ``--exclude-standard`` honours the committed ``.gitignore`` as well as the
    machine-local sources, and several committed patterns in this repo reach
    inside ``src/ccs/`` (``states/``, ``build/``, ``dist/``, ``*.pyd``). Hiding
    those from the guard would remove exactly the case it exists to catch: a
    git-ignored asset is GUARANTEED missing from the wheel, so a runtime read
    of it fails for every installer. Only a rule local to one machine describes
    a file nobody was ever going to ship.

    ``git check-ignore -v -z`` names the source of the matching rule, so the
    split is decided by provenance rather than by guessing from the path.

    FAILS CLOSED. Every failure returns an empty set: nothing is subtracted and
    the guard stays at its strictest. That asymmetry is why this subtracts a
    narrow ignored set instead of walking ``git ls-files`` (tracked-only) — a
    tracked-only walk fails OPEN, yielding no assets, no undeclared set, and a
    module reporting green with nothing left to check.
    """
    root = REPO_ROOT if repo_root is None else repo_root
    if not candidates:
        return set()
    # Refuse to answer unless git resolves to the tree we asked about. A
    # foreign context reports candidates as ignored under ITS rules, which
    # empties the walk and leaves the guard passing with nothing to check —
    # the fail-open this whole filter is built to avoid.
    if _git_toplevel(root) != root.resolve():
        return set()
    ordered = sorted(candidates)
    stdin = "\0".join(str(path) for path in ordered) + "\0"
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "-v", "-z", "--stdin"],
            input=stdin, capture_output=True, text=True, timeout=30,
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return set()
    # 0 = at least one path ignored, 1 = none ignored (a normal answer, NOT an
    # error). Anything else — 128 for "not a git repository" — is a failure.
    if completed.returncode not in (0, 1):
        return set()

    fields = completed.stdout.split("\0")
    hidden: set[Path] = set()
    provenance: dict[str, bool] = {}
    # Records are <source>\0<linenum>\0<pattern>\0<pathname>\0.
    for index in range(0, len(fields) - 3, 4):
        source, _linenum, _pattern, pathname = fields[index:index + 4]
        if not pathname:
            continue
        if source not in provenance:
            provenance[source] = _tracked(root, source)
        if not provenance[source]:
            hidden.add(Path(pathname))
    return hidden


def _shipped_assets() -> set[Path]:
    """Every non-Python, non-cache, non-git-ignored file under ``src/ccs/``.

    The git-ignored subtraction is what keeps this honest on a developer
    machine: local-only files (per-package ``CLAUDE.md`` instructions, a
    macOS ``.DS_Store``) sit under ``src/ccs/`` but are absent from the CI
    checkout, so without it this guard was red locally and green in CI — for
    files that were never going to ship either way.
    """
    walked = {
        path
        for path in (SRC_ROOT / "ccs").glob("**/*")
        if path.is_file() and path.suffix != ".py" and "__pycache__" not in path.parts
    }
    return walked - _locally_ignored_paths(walked)


def test_every_non_python_asset_is_declared_package_data() -> None:
    """A new asset that nobody declared fails here, not in a user's install."""
    undeclared = sorted(path.relative_to(REPO_ROOT) for path in _shipped_assets() - _declared_data_files())

    assert not undeclared, (
        "These files live under src/ccs/ but would NOT ship in the wheel or sdist:\n"
        + "\n".join(f"  - {path}" for path in undeclared)
        + "\n\nAnything read at runtime via Path(__file__) must be declared, or an\n"
        "installed-from-PyPI import raises FileNotFoundError. Add the owning\n"
        "package to [tool.setuptools.package-data] in pyproject.toml, naming the\n"
        'subdirectory explicitly (e.g. "ccs.output" = ["templates/*.html"]) — a\n'
        'bare "*" is globbed against the package directory and stops at the\n'
        "path separator, so it never reaches a file one level down."
    )


def test_the_html_templates_are_declared() -> None:
    """The two templates whose absence this guard was written for."""
    declared = _declared_data_files()

    for relative in (
        "ccs/diagnose/templates/diagnose_report.html",
        "ccs/output/templates/comparison_report.html",
    ):
        assert SRC_ROOT / relative in declared, f"{relative} is not declared package data"


def test_replayed_data_files_match_setuptools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the replay above to what the real build backend resolves.

    The ``dev`` extra installs setuptools precisely so this runs in CI. It
    still degrades to a skip rather than an error, because the guard above
    must keep working in a bare venv — neither a fresh ``python -m venv`` nor
    GitHub's setup-python leaves setuptools importable on 3.12, so a hard
    requirement here would take the whole module down with it.
    """
    setuptools = pytest.importorskip("setuptools", reason="build backend not installed in this env")
    from setuptools.config.pyprojecttoml import apply_configuration

    # setuptools resolves package_dir and the dynamic version relative to cwd.
    monkeypatch.chdir(REPO_ROOT)
    distribution = setuptools.Distribution()
    apply_configuration(distribution, str(PYPROJECT_PATH))

    # ``include_package_data`` defaults on under pyproject.toml, and its leg of
    # find_data_files shells out to egg_info to read SOURCES.txt — which would
    # write *.egg-info into the working tree mid-test. Both sides of this
    # comparison are the *declared* patterns, which is the thing under guard;
    # the red-state build confirmed the manifest leg contributes nothing here
    # (no MANIFEST.in, no revision-control plugin).
    distribution.include_package_data = False

    build_py = distribution.get_command_obj("build_py")
    build_py.ensure_finalized()
    # setuptools reports source dirs relative to cwd; this comparison is
    # against absolute paths, so anchor them back on the repo root.
    resolved = {
        (REPO_ROOT / source_dir / name).resolve()
        for _package, source_dir, _build_dir, names in build_py.data_files
        for name in names
    }

    assert resolved == _declared_data_files()


# ----------------------------------------------------------------------
# The ignore filter: what git hides can never ship, so it is not an asset
# ----------------------------------------------------------------------
#
# ``_shipped_assets`` walks the filesystem, so it used to count every
# non-.py file under src/ccs/ — including local-only files git ignores.
# Six ``CLAUDE.md`` files (and, on macOS, any ``.DS_Store``) made this guard
# permanently red on a developer machine while staying green in CI, where
# ``actions/checkout`` materializes tracked content only. A guard that is
# always red is a guard nobody reads.


def test_shipped_assets_excludes_paths_git_reports_ignored(monkeypatch) -> None:
    """The walk consults the ignore filter. Proven with a sentinel — a real
    on-disk asset forced into the ignored set must drop out of the walk."""
    sentinel = SRC_ROOT / "ccs" / "output" / "templates" / "comparison_report.html"
    assert sentinel.is_file(), "fixture drifted: the sentinel asset must exist"
    assert sentinel in mod._shipped_assets(), "sentinel must be present before filtering"

    monkeypatch.setattr(
        mod, "_locally_ignored_paths", lambda candidates, **_kw: {sentinel}
    )
    assert sentinel not in mod._shipped_assets()


def test_ignored_lookup_fails_closed_when_git_is_unavailable(monkeypatch) -> None:
    """No git → an EMPTY ignored set, so nothing is subtracted and the guard
    stays at its strictest.

    This is why the filter subtracts the IGNORED set rather than switching the
    walk to `git ls-files` (tracked-only): a tracked-only walk would fail OPEN
    — a missing git yields no assets at all, no undeclared set, and a silently
    green guard that has stopped checking anything.
    """
    def _boom(*_a, **_kw):
        raise FileNotFoundError("git not on PATH")

    probe = {SRC_ROOT / "ccs" / "adapters" / "CLAUDE.md"}

    monkeypatch.setattr(mod.subprocess, "run", _boom)
    assert mod._locally_ignored_paths(probe) == set()

    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            mod.subprocess.CalledProcessError(128, "git")
        ),
    )
    assert mod._locally_ignored_paths(probe) == set()

    # `text=True` decodes git's output, so a path this locale cannot decode
    # raises UnicodeDecodeError — a ValueError, which neither OSError nor
    # SubprocessError covers. Without it in the except clause the helper
    # CRASHES instead of failing closed, contradicting its own docstring.
    def _undecodable(*_a, **_kw):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(mod.subprocess, "run", _undecodable)
    assert mod._locally_ignored_paths(probe) == set()

    # check-ignore exits 128 outside a repository. Exit 1 ("nothing ignored")
    # is a NORMAL answer and must not be mistaken for one of these.
    # The stdout here MUST be parseable, or the assertion cannot see its own
    # failure: with empty output, accepting 128 yields an empty set anyway and
    # the test passes either way. This record parses into a subtraction, so
    # only a genuine fail-closed returns the empty set.
    record = "\0".join(
        ["/nowhere/.git/info/exclude", "1", "CLAUDE.md", str(next(iter(probe)))]
    ) + "\0"
    monkeypatch.setattr(
        mod.subprocess, "run",
        lambda *_a, **_kw: mod.subprocess.CompletedProcess([], 128, record, "fatal"),
    )
    assert mod._locally_ignored_paths(probe) == set(), (
        "exit 128 is a failure (not a git repository) and must subtract nothing"
    )


def test_declared_templates_are_never_filtered_out() -> None:
    """Anti-fail-open: the two tracked, declared assets must survive the
    filter on every machine. An over-broad ignore filter would empty the
    walk and turn this whole module green while checking nothing."""
    shipped = mod._shipped_assets()
    for rel in (
        "ccs/output/templates/comparison_report.html",
        "ccs/diagnose/templates/diagnose_report.html",
    ):
        assert SRC_ROOT / rel in shipped, f"{rel} vanished from the asset walk"

    # NOT `ignored & shipped`: `shipped` is the walk MINUS the filter, so that
    # intersection is empty by construction and passes for every possible
    # filter — including one returning every asset, which empties the walk
    # entirely. Intersect against the DECLARED set instead. Nothing the
    # packaging config says ships may be filtered out of the walk, and unlike
    # the two names above that assertion keeps its teeth when a third asset is
    # added.
    declared = _declared_data_files()
    swallowed = mod._locally_ignored_paths(declared | shipped) & declared
    assert not swallowed, (
        "the ignore filter hid an asset that pyproject declares as shipping: "
        f"{sorted(swallowed)}"
    )

    ignored = mod._locally_ignored_paths(shipped)
    # Under src/ccs specifically, not merely somewhere in the repo:
    # SRC_ROOT.parent is REPO_ROOT, which is a parent of EVERY repo path, so
    # asserting on it would pass for docs/ and tests/ alike — a check that
    # cannot see the case it names.
    package_root = SRC_ROOT / "ccs"
    for path in ignored:
        assert package_root in path.parents, (
            f"ignore filter reached outside the coordinated package: {path}"
        )


# ----------------------------------------------------------------------
# Only DEVELOPER-LOCAL ignore rules may hide an asset from the guard
# ----------------------------------------------------------------------
#
# `--exclude-standard` honours the repo's committed .gitignore as well as
# .git/info/exclude. Several committed patterns reach inside src/ccs/
# (`states/`, `build/`, `dist/`, `*.pyd`), and a file they hide is exactly
# the case this guard exists to catch: git-ignoring an asset GUARANTEES it is
# missing from the wheel, so the runtime read fails for every installer.
# Only a rule that is local to one machine describes a file nobody ships.


def test_committed_gitignore_rules_never_hide_an_asset() -> None:
    """A path the TRACKED .gitignore hides stays visible to the guard.

    Runs identically everywhere: .gitignore ships to every clone, so these
    paths are ignored in CI too. They need not exist — check-ignore answers
    about rules, not about the filesystem.
    """
    committed = {
        SRC_ROOT / "ccs" / "coordinator" / "states" / "snapshot.json",  # states/
        SRC_ROOT / "ccs" / "output" / "build" / "page.html",            # build/
        SRC_ROOT / "ccs" / "dist" / "bundle.json",                      # dist/
        SRC_ROOT / "ccs" / "core" / "fast.pyd",                         # *.pyd
    }
    hidden = mod._locally_ignored_paths(committed)
    assert hidden == set(), (
        "a committed .gitignore rule must not excuse an undeclared asset — "
        f"these were wrongly hidden from the guard: {sorted(hidden)}"
    )


def test_a_machine_local_exclude_does_hide_an_asset(tmp_path: Path) -> None:
    """The other half: a rule in .git/info/exclude IS local noise and is
    subtracted. Built in a throwaway repo so the assertion does not depend on
    what this developer happens to have in their own exclude file."""
    import subprocess as sp

    sp.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text("tracked-rule.json\n", encoding="utf-8")
    (tmp_path / ".git" / "info").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".git" / "info" / "exclude").write_text("LOCALNOTE.md\n", encoding="utf-8")
    sp.run(["git", "-C", str(tmp_path), "add", ".gitignore"], check=True)

    local = tmp_path / "pkg" / "LOCALNOTE.md"
    committed = tmp_path / "pkg" / "tracked-rule.json"
    plain = tmp_path / "pkg" / "asset.html"

    hidden = mod._locally_ignored_paths({local, committed, plain}, repo_root=tmp_path)
    assert hidden == {local}, (
        "only the .git/info/exclude rule is developer-local; the tracked "
        f".gitignore rule and the unignored path must survive. got {sorted(hidden)}"
    )


def test_a_foreign_git_context_subtracts_nothing(tmp_path: Path) -> None:
    """`git -C <dir>` searches UPWARD for a .git, so a checkout sitting inside
    another repository answers from THAT repository's ignore rules.

    An outer `.gitignore` covering the containing directory then reports every
    candidate as ignored, the walk empties, and the primary guard passes while
    checking nothing. Pin that the filter refuses to answer unless git resolves
    to the tree it was asked about.
    """
    import subprocess as sp

    outer = tmp_path / "outer"
    inner = outer / "build" / "checkout"
    inner.mkdir(parents=True)
    sp.run(["git", "init", "-q", str(outer)], check=True)
    (outer / ".gitignore").write_text("build/\n", encoding="utf-8")

    candidate = inner / "src" / "ccs" / "asset.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_text("{}", encoding="utf-8")

    # The outer repo really does call it ignored — the hazard is genuine.
    outer_view = sp.run(
        ["git", "-C", str(inner), "check-ignore", "-q", str(candidate)],
        capture_output=True,
    )
    assert outer_view.returncode == 0, "fixture drifted: outer repo must ignore it"

    hidden = mod._locally_ignored_paths({candidate}, repo_root=inner)
    assert hidden == set(), (
        "a foreign git context must subtract nothing — otherwise an outer "
        "repo's ignore rules can empty the walk and the guard checks nothing"
    )


def test_this_worktree_is_its_own_git_toplevel() -> None:
    """The containment check must not reject the normal case. A git worktree's
    toplevel is the worktree root, not the main checkout, so the real tree
    still gets a real answer."""
    probe = {SRC_ROOT / "ccs" / "adapters" / "CLAUDE.md"}
    # Not an assertion about THIS machine's exclude file: only that the filter
    # answers at all here rather than bailing out as foreign.
    assert mod._git_toplevel(REPO_ROOT) == REPO_ROOT
    assert isinstance(mod._locally_ignored_paths(probe), set)


def test_iter_packages_skips_build_artifact_directories(tmp_path: Path) -> None:
    """`_iter_packages` enumerates directories, and a source checkout that has
    been `pip install -e`'d carries `<name>.egg-info` under src/.

    setuptools does not treat that as a package; counting it does, and every
    extra "package" only ever ADDS to the declared set, shrinking the
    undeclared difference the guard reports. Fail-open in direction, so it is
    pinned here rather than left to the setuptools parity test below, which
    skips whenever the backend is not importable.
    """
    (tmp_path / "ccs" / "output").mkdir(parents=True)
    (tmp_path / "agent_coherence.egg-info").mkdir()
    (tmp_path / "agent_coherence.egg-info" / "nested").mkdir()

    found = set(mod._iter_packages(tmp_path))
    assert "ccs" in found and "ccs.output" in found
    assert not {name for name in found if "egg-info" in name}, (
        f"build-artifact directories counted as packages: {sorted(found)}"
    )


def test_replayed_package_list_matches_setuptools(monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing parity test compares resolved DATA FILES, so a package the
    replay invents but setuptools does not is invisible there whenever that
    package contributes no data — which is exactly the egg-info case. Compare
    the package sets directly."""
    setuptools = pytest.importorskip("setuptools", reason="build backend not installed in this env")
    from setuptools.config.pyprojecttoml import apply_configuration

    monkeypatch.chdir(REPO_ROOT)
    distribution = setuptools.Distribution()
    apply_configuration(distribution, str(PYPROJECT_PATH))

    assert set(distribution.packages or []) == set(mod._iter_packages(SRC_ROOT))
