# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""What a ``v*`` tag is allowed to publish, and how.

``tools/check_release_tag.py`` decides; the two workflows act on its answer:

* a tag must name the package version exactly, and a ``.devN`` version is
  refused outright -- it is a snapshot of an unreleased branch;
* ``aN`` / ``bN`` / ``rcN`` go to PyPI (pip skips them unless asked) and
  become a GitHub *pre-release*;
* ``publish-mcp.yml`` reads that flag back from the GitHub release and keeps
  pre-releases off the MCP Registry.

The tool is loaded by file path (``tools/`` is not a package). The workflow
wiring is read from the YAML itself, and the registry workflow's resolve step
is executed under bash with ``gh`` stubbed -- a copy of either would keep
passing after the real file changed.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOL = _REPO_ROOT / "tools" / "check_release_tag.py"
_RELEASE_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "release.yml"
_MCP_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "publish-mcp.yml"
_RESOLVE_STEP = "Resolve the released tag"


def _import_tool():
    spec = importlib.util.spec_from_file_location("check_release_tag", _TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Registered before exec: @dataclass looks its module up in sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


tool = _import_tool()


# ---------------------------------------------------------------------------
# classify_release
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["0.15.0", "0.8.4.1", "0.15.0.post1", "1.0.0"])
def test_final_version_passes_as_a_final_release(version: str) -> None:
    kind = tool.classify_release(f"v{version}", version)
    assert kind.version == version
    assert kind.is_prerelease is False


@pytest.mark.parametrize("version", ["0.15.0a1", "0.15.0b2", "0.15.0rc1", "0.8.0a1"])
def test_alpha_beta_rc_pass_as_a_prerelease(version: str) -> None:
    assert tool.classify_release(f"v{version}", version).is_prerelease is True


@pytest.mark.parametrize("version", ["0.15.0.dev0", "0.15.0rc1.dev0", "0.15.0.post1.dev3"])
def test_dev_version_is_refused_even_when_the_tag_matches(version: str) -> None:
    with pytest.raises(tool.ReleaseTagError, match="development version"):
        tool.classify_release(f"v{version}", version)


@pytest.mark.parametrize(
    ("tag", "version"),
    [
        # Tagging the release before the version bump is committed.
        ("v0.15.0", "0.15.0.dev0"),
        ("v0.15.0rc1", "0.15.0"),
        ("0.15.0", "0.15.0"),
        ("v0.15.0-rc1", "0.15.0rc1"),
    ],
)
def test_tag_that_does_not_name_the_version_is_refused(tag: str, version: str) -> None:
    with pytest.raises(tool.ReleaseTagError, match="does not match package version"):
        tool.classify_release(tag, version)


def test_unparseable_version_is_refused() -> None:
    with pytest.raises(tool.ReleaseTagError, match="not a valid PEP 440 version"):
        tool.classify_release("vnext", "next")


# ---------------------------------------------------------------------------
# main(): exit code and the GITHUB_OUTPUT line
# ---------------------------------------------------------------------------


def _package_init(tmp_path: Path, version: str) -> Path:
    path = tmp_path / "__init__.py"
    path.write_text(f'"""pkg."""\n\n__version__ = "{version}"\n')
    return path


def _run_main(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, tag: str, version: str
) -> tuple[int, str]:
    output = tmp_path / "github_output"
    output.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    code = tool.main(["--tag", tag, "--package-init", str(_package_init(tmp_path, version))])
    return code, output.read_text()


@pytest.mark.parametrize(
    ("tag", "version", "expected"),
    [("v0.15.0", "0.15.0", "false"), ("v0.15.0rc1", "0.15.0rc1", "true")],
)
def test_main_writes_the_prerelease_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tag: str, version: str, expected: str
) -> None:
    code, output = _run_main(tmp_path, monkeypatch, tag=tag, version=version)
    assert code == 0
    assert output == f"prerelease={expected}\n"


@pytest.mark.parametrize(
    ("tag", "version"),
    [("v0.15.0.dev0", "0.15.0.dev0"), ("v0.15.0", "0.15.0.dev0")],
)
def test_main_fails_and_writes_no_output_for_a_refused_tag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tag: str,
    version: str,
) -> None:
    code, output = _run_main(tmp_path, monkeypatch, tag=tag, version=version)
    assert code == 1
    assert output == ""
    assert tag in capsys.readouterr().err


def test_main_reads_the_tag_from_release_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The workflow passes the tag through RELEASE_TAG, not --tag."""
    output = tmp_path / "github_output"
    output.touch()
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    monkeypatch.setenv("RELEASE_TAG", "v0.15.0rc1")
    assert tool.main(["--package-init", str(_package_init(tmp_path, "0.15.0rc1"))]) == 0
    assert output.read_text() == "prerelease=true\n"


def test_main_refuses_a_dev_tag_given_through_release_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("RELEASE_TAG", "v0.15.0.dev0")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    assert tool.main(["--package-init", str(_package_init(tmp_path, "0.15.0.dev0"))]) == 1
    assert "development version" in capsys.readouterr().err


def test_main_accepts_a_tag_without_github_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A local dry run has no GITHUB_OUTPUT; the answer is printed instead."""
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    package_init = _package_init(tmp_path, "0.15.0")
    assert tool.main(["--tag", "v0.15.0", "--package-init", str(package_init)]) == 0
    assert "a final release" in capsys.readouterr().out


def test_main_without_a_tag_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RELEASE_TAG", raising=False)
    assert tool.main(["--package-init", str(_package_init(tmp_path, "0.15.0"))]) == 1


def test_main_fails_when_the_package_init_has_no_version(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    package_init = tmp_path / "__init__.py"
    package_init.write_text('"""pkg."""\n')
    assert tool.main(["--tag", "v0.15.0", "--package-init", str(package_init)]) == 1
    assert "Could not find __version__" in capsys.readouterr().err


def test_repo_package_version_is_readable() -> None:
    """The regex still finds __version__ in the real file the workflow reads."""
    version = tool.read_package_version(_REPO_ROOT / "src" / "ccs" / "__init__.py")
    from ccs import __version__

    assert version == __version__


# ---------------------------------------------------------------------------
# release.yml wiring
# ---------------------------------------------------------------------------


def _jobs(workflow: Path) -> dict:
    return yaml.safe_load(workflow.read_text())["jobs"]


def _needs(job: dict) -> set[str]:
    needs = job.get("needs") or []
    return {needs} if isinstance(needs, str) else set(needs)


def _step(job: dict, *, step_id: str | None = None, uses: str | None = None) -> dict:
    for step in job.get("steps") or []:
        if step_id is not None and step.get("id") == step_id:
            return step
        if uses is not None and str(step.get("uses", "")).startswith(uses):
            return step
    raise AssertionError(f"no step with id={step_id!r} uses={uses!r}")


def test_release_build_job_runs_the_tag_check_and_exports_its_answer() -> None:
    build = _jobs(_RELEASE_WORKFLOW)["build"]
    check = _step(build, step_id="release-tag")
    run = check["run"]
    assert "python tools/check_release_tag.py" in run
    # A fresh setup-python interpreter has no `packaging`; without the
    # install the check dies on its import instead of judging the tag.
    assert "python -m pip install packaging" in run
    assert run.index("pip install packaging") < run.index("python tools/check_release_tag.py")
    assert check["env"]["RELEASE_TAG"] == "${{ github.ref_name }}"
    assert build["outputs"]["prerelease"] == "${{ steps.release-tag.outputs.prerelease }}"


def test_release_tag_check_runs_before_anything_is_built() -> None:
    steps = _jobs(_RELEASE_WORKFLOW)["build"]["steps"]
    check_at = next(i for i, s in enumerate(steps) if s.get("id") == "release-tag")
    build_at = next(i for i, s in enumerate(steps) if s.get("run") == "python -m build")
    assert check_at < build_at


def test_github_release_is_marked_prerelease_from_the_tag_check() -> None:
    job = _jobs(_RELEASE_WORKFLOW)["github-release"]
    assert "build" in _needs(job), "needs.build.outputs is empty unless build is in needs"
    release_step = _step(job, uses="softprops/action-gh-release@")
    assert release_step["with"]["prerelease"] == "${{ needs.build.outputs.prerelease }}"


def test_nothing_publishes_before_the_tag_check_passes() -> None:
    """The tag check runs in build; both publishing jobs wait on it.

    The GitHub release also waits for PyPI, so a failed or rejected PyPI
    publish leaves no GitHub release for the registry workflow to pick up.
    """
    jobs = _jobs(_RELEASE_WORKFLOW)
    assert "build" in _needs(jobs["publish"])
    assert {"build", "publish"} <= _needs(jobs["github-release"])


# ---------------------------------------------------------------------------
# publish-mcp.yml: pre-releases stay off the MCP Registry
# ---------------------------------------------------------------------------


def _mcp_steps() -> list[dict]:
    return _jobs(_MCP_WORKFLOW)["publish"]["steps"]


def test_every_registry_step_is_skipped_for_a_prerelease() -> None:
    steps = _mcp_steps()
    resolve_at = next(i for i, s in enumerate(steps) if s.get("name") == _RESOLVE_STEP)
    assert steps[resolve_at]["id"] == "release"
    later = steps[resolve_at + 1 :]
    assert later, "the resolve step must come before the registry steps"
    for step in later:
        assert step.get("if") == "steps.release.outputs.prerelease == 'false'", step.get("name")


def test_resolve_step_reads_the_tag_from_the_triggering_event() -> None:
    """The resolve tests below set these variables directly; this pins where they come from."""
    steps = {step.get("name"): step for step in _mcp_steps()}
    env = steps[_RESOLVE_STEP]["env"]
    assert env["EVENT_RELEASE_TAG"] == "${{ github.event.release.tag_name }}"
    assert env["WORKFLOW_RUN_TAG"] == "${{ github.event.workflow_run.head_branch }}"
    sync = steps["Sync server.json version to the released tag"]
    assert sync["env"]["RELEASE_TAG"] == "${{ steps.release.outputs.tag }}"


_GH_STUB = textwrap.dedent(
    """\
    #!/usr/bin/env bash
    # gh release view [TAG] --repo R --json FIELD -q EXPR
    echo "$*" >> "$AC_STUB_GH_LOG"
    [ "$1" = release ] && [ "$2" = view ] || exit 2
    shift 2
    tag=""
    case "${1:-}" in --*) ;; *) tag="$1"; shift ;; esac
    field=""
    while [ $# -gt 0 ]; do
      case "$1" in --json) field="$2"; shift 2 ;; *) shift ;; esac
    done
    case "$field" in
      tagName) echo "${tag:-$AC_STUB_LATEST}" ;;
      isPrerelease)
        if [ -n "${AC_STUB_GH_FAIL:-}" ]; then
          echo "release not found" >&2
          exit 1
        elif [ -n "${AC_STUB_PRERELEASE_ANSWER:-}" ]; then
          echo "$AC_STUB_PRERELEASE_ANSWER"
        else
          case " $AC_STUB_PRERELEASES " in
            *" $tag "*) echo true ;;
            *) echo false ;;
          esac
        fi ;;
      *) exit 3 ;;
    esac
    """
)


def _resolve_step_script() -> str:
    for step in _mcp_steps():
        if step.get("name") == _RESOLVE_STEP:
            return step["run"]
    raise AssertionError(f"no {_RESOLVE_STEP!r} step in {_MCP_WORKFLOW}")


def _run_resolve(
    tmp_path: Path,
    *,
    event_release_tag: str = "",
    workflow_run_tag: str = "",
    latest: str = "v0.14.1",
    prereleases: tuple[str, ...] = (),
    prerelease_answer: str = "",
    gh_fails: bool = False,
) -> tuple[subprocess.CompletedProcess[str], dict[str, str], str]:
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    (stub_bin / "gh").write_text(_GH_STUB)
    (stub_bin / "gh").chmod(0o755)
    output = tmp_path / "github_output"
    output.touch()
    gh_log = tmp_path / "gh.log"
    gh_log.touch()
    env = {
        **os.environ,
        "PATH": f"{stub_bin}:{os.environ.get('PATH', '')}",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_REPOSITORY": "example/agent-coherence",
        "GH_TOKEN": "unused",
        # Set explicitly, never inherited from the caller's environment.
        "EVENT_RELEASE_TAG": event_release_tag,
        "WORKFLOW_RUN_TAG": workflow_run_tag,
        "AC_STUB_GH_LOG": str(gh_log),
        "AC_STUB_LATEST": latest,
        "AC_STUB_PRERELEASES": " ".join(prereleases),
        "AC_STUB_PRERELEASE_ANSWER": prerelease_answer,
        **({"AC_STUB_GH_FAIL": "1"} if gh_fails else {}),
    }
    # A `run:` with no `shell:` executes as `bash -e {0}` on the runner: no
    # pipefail, so a failure inside a pipe would not stop the step there
    # either. The fail-closed tests below must not pass on a stricter shell.
    proc = subprocess.run(
        ["bash", "-e", "-c", _resolve_step_script()],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    outputs = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )
    return proc, outputs, gh_log.read_text()


_needs_bash = pytest.mark.skipif(
    shutil.which("bash") is None, reason="the step's script runs under bash, as on the runner"
)


@_needs_bash
def test_workflow_run_after_an_rc_skips_the_registry(tmp_path: Path) -> None:
    proc, outputs, _ = _run_resolve(
        tmp_path, workflow_run_tag="v0.15.0rc1", prereleases=("v0.15.0rc1",)
    )
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"tag": "v0.15.0rc1", "prerelease": "true"}
    assert "not publishing it to the MCP Registry" in proc.stdout


@_needs_bash
def test_workflow_run_publishes_the_tag_that_ran_not_the_latest_release(tmp_path: Path) -> None:
    """The latest GitHub release is never a pre-release.

    Resolving workflow_run through it would hand the registry the previous
    final release after an rc run, so the run's own tag must win.
    """
    proc, outputs, gh_log = _run_resolve(
        tmp_path, workflow_run_tag="v0.15.0", latest="v0.14.1"
    )
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"tag": "v0.15.0", "prerelease": "false"}
    assert "--json tagName" not in gh_log, "must not fall back to the latest release"


@_needs_bash
def test_release_event_for_a_prerelease_skips_the_registry(tmp_path: Path) -> None:
    proc, outputs, _ = _run_resolve(
        tmp_path, event_release_tag="v0.15.0b1", prereleases=("v0.15.0b1",)
    )
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"tag": "v0.15.0b1", "prerelease": "true"}


@_needs_bash
def test_manual_dispatch_publishes_the_latest_release(tmp_path: Path) -> None:
    proc, outputs, _ = _run_resolve(tmp_path, latest="v0.14.1")
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"tag": "v0.14.1", "prerelease": "false"}


@_needs_bash
def test_unreadable_prerelease_flag_fails_the_job(tmp_path: Path) -> None:
    """Cannot tell is not 'false': an unknown answer must not publish."""
    proc, outputs, _ = _run_resolve(
        tmp_path, workflow_run_tag="v0.15.0", prerelease_answer="null"
    )
    assert proc.returncode != 0
    assert "prerelease" not in outputs


@_needs_bash
def test_failed_release_lookup_fails_the_job(tmp_path: Path) -> None:
    proc, outputs, _ = _run_resolve(tmp_path, workflow_run_tag="v0.15.0", gh_fails=True)
    assert proc.returncode != 0
    assert outputs == {}
