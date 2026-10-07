"""The protocol-corpus gate's plugin-ref resolution, executed rather than read.

The gate builds the plugin branch whose name matches the branch under test.
With no sibling it falls back to the plugin branch of the line the change is
bound for: plugin ``dev`` for a pull request into ``dev`` or a hand-dispatched
run on a feature branch, plugin ``main`` otherwise. Its job summary then says
which ref it proved against, and -- when it fell back to ``main`` -- how far
that ref trails the plugin's own ``dev``.

Nothing exercised that shell block, so a wrong answer could only be found by
reading it. These tests extract the step's ``run`` script straight from
``ci.yml`` and run it under bash with ``git`` and ``curl`` stubbed, asserting
the sentence it writes to ``$GITHUB_STEP_SUMMARY``.

The case that motivates the file: a push to this repo's ``main`` pairs with
the plugin's ``main`` DELIBERATELY (the step's own comment says so), and a
guard that re-derives "did we fall back?" from the resolved value cannot tell
that apart from a real fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
STEP_NAME = "Resolve plugin ref"

pytestmark = pytest.mark.skipif(
    shutil.which("jq") is None or shutil.which("bash") is None,
    reason="the step's script needs bash and jq, as the GitHub runner has",
)


def _resolve_step_script() -> str:
    """The `run:` body of the Resolve plugin ref step, from the workflow itself.

    Read from ci.yml rather than duplicated here: a copy would keep passing
    after the real step changed, which is the failure mode this file exists
    to prevent.
    """
    parsed = yaml.safe_load(CI_WORKFLOW.read_text())
    for job in parsed["jobs"].values():
        for step in job.get("steps") or []:
            if step.get("name") == STEP_NAME:
                return step["run"]
    raise AssertionError(f"no {STEP_NAME!r} step in {CI_WORKFLOW}")


def _run_step(
    tmp_path: Path,
    *,
    sibling_ref: str,
    base_ref: str = "",
    event_name: str = "push",
    ref_name: str = "",
    plugin_branches: tuple[str, ...] = ("main", "dev"),
    curl_body: str = '{"status":"diverged","ahead_by":33,"behind_by":2}',
    curl_fails: bool = False,
) -> str:
    """Execute the real step with git and curl stubbed; return the summary text."""
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()

    # `git ls-remote --exit-code --heads <remote> <ref>` probes existence;
    # `git ls-remote <remote> refs/heads/<ref>` reads the sha. Both are stubbed
    # off one branch list so a test names only what exists on the plugin side.
    (stub_bin / "git").write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            [ "$1" = ls-remote ] || exit 0
            shift
            want_exit_code=0
            while [ "${1:-}" = --exit-code ] || [ "${1:-}" = --heads ]; do
              [ "$1" = --exit-code ] && want_exit_code=1
              shift
            done
            ref="${2:-}"
            name="${ref#refs/heads/}"
            case " $AC_STUB_BRANCHES " in
              *" $name "*)
                echo "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef	refs/heads/$name"
                exit 0 ;;
              *)
                # git ls-remote --exit-code exits 2 when nothing matched.
                [ "$want_exit_code" = 1 ] && exit 2
                exit 0 ;;
            esac
            """
        )
    )
    (stub_bin / "curl").write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            [ -n "${AC_STUB_CURL_FAIL:-}" ] && exit 6
            printf '%s' "$AC_STUB_CURL_BODY"
            """
        )
    )
    for name in ("git", "curl"):
        (stub_bin / name).chmod(0o755)

    summary = tmp_path / "summary.md"
    summary.touch()
    output = tmp_path / "output.txt"
    output.touch()

    env = {
        **os.environ,
        "PATH": f"{stub_bin}:{os.environ.get('PATH', '')}",
        "SIBLING_REF": sibling_ref,
        # Set explicitly, never inherited: a stray BASE_REF in the caller's
        # environment would otherwise pick the fallback for the test.
        "BASE_REF": base_ref,
        "EVENT_NAME": event_name,
        "REF_NAME": ref_name or sibling_ref,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
        "AC_STUB_BRANCHES": " ".join(plugin_branches),
        "AC_STUB_CURL_BODY": curl_body,
        **({"AC_STUB_CURL_FAIL": "1"} if curl_fails else {}),
    }
    # -e -o pipefail matches the GitHub runner's default shell for `run:`.
    proc = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", _resolve_step_script()],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        "the resolve step must never fail the parity job:\n"
        f"exit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    return summary.read_text()


def _resolved_ref(tmp_path: Path) -> str:
    """The ref the step handed to the checkout, from its GITHUB_OUTPUT line."""
    for line in (tmp_path / "output.txt").read_text().splitlines():
        if line.startswith("ref="):
            return line[len("ref="):]
    raise AssertionError("the step wrote no ref= output")


def test_push_to_main_is_the_intended_pairing_not_a_fallback(tmp_path: Path) -> None:
    """A push to `main` pairs with the plugin's `main` on purpose.

    `github.ref_name` is `main`, the plugin has a `main`, so the sibling probe
    matches and the ref is chosen through the sibling path. Calling that "the
    fallback" is wrong, and it is wrong on this repo's own release-branch CI
    run. A guard keyed on `$ref = main` cannot tell the two apart.
    """
    summary = _run_step(tmp_path, sibling_ref="main")
    assert "agent-coherence-plugin@main" in summary
    assert "fallback" not in summary, (
        "main<->main is the documented deliberate pairing, not a fallback; got:\n" + summary
    )


def test_sibling_branch_is_named_without_a_staleness_note(tmp_path: Path) -> None:
    summary = _run_step(tmp_path, sibling_ref="dev")
    assert "agent-coherence-plugin@dev" in summary
    assert "fallback" not in summary
    assert "behind" not in summary


def test_missing_sibling_falls_back_and_says_how_far_behind(tmp_path: Path) -> None:
    summary = _run_step(tmp_path, sibling_ref="fix/no-such-plugin-branch")
    assert "agent-coherence-plugin@main" in summary
    assert "the fallback" in summary
    assert "33 commit(s) behind" in summary


def test_a_tag_falls_back_too(tmp_path: Path) -> None:
    """Release tags have no plugin counterpart; the fallback is expected there."""
    summary = _run_step(tmp_path, sibling_ref="v0.5.0")
    assert "the fallback" in summary


def test_fallback_that_is_level_with_dev_says_level_not_nothing(tmp_path: Path) -> None:
    """Zero distance is a real answer and must not look like a failed lookup.

    It is the state right after a release, when `main` and `dev` agree -- the
    one moment the fallback is harmless, which is worth saying out loud.
    """
    summary = _run_step(
        tmp_path,
        sibling_ref="fix/no-such-plugin-branch",
        curl_body='{"status":"identical","ahead_by":0,"behind_by":0}',
    )
    assert "the fallback" in summary
    assert "level with" in summary
    assert "unavailable" not in summary


def test_failed_staleness_lookup_says_so_rather_than_going_quiet(tmp_path: Path) -> None:
    """A rate-limited or unreachable API must not read as "level with dev".

    Both produce an empty result, and collapsing them prints the bare sentence
    this note exists to qualify -- reassurance generated by a failure.
    """
    summary = _run_step(
        tmp_path, sibling_ref="fix/no-such-plugin-branch", curl_fails=True
    )
    assert "the fallback" in summary
    assert "unavailable" in summary, (
        "a failed lookup must be distinguishable from a zero distance; got:\n" + summary
    )


def test_a_pull_request_into_dev_with_no_twin_pairs_with_plugin_dev(tmp_path: Path) -> None:
    """The case that left every library pull request into `dev` red.

    With no same-named plugin branch, the old fallback was plugin `main`,
    which trails plugin `dev` -- so rows `dev`'s own push run passes (it
    pairs with plugin `dev`) failed on every pull request into `dev`. The
    fallback now follows the line the change merges into.
    """
    summary = _run_step(
        tmp_path,
        sibling_ref="fix/no-such-plugin-branch",
        base_ref="dev",
        event_name="pull_request",
    )
    assert _resolved_ref(tmp_path) == "dev"
    assert "agent-coherence-plugin@dev" in summary
    assert "the line this change merges into" in summary
    assert "behind" not in summary, (
        "the staleness note measures main against dev; it means nothing for dev:\n"
        + summary
    )


def test_a_dispatched_run_on_a_stacked_branch_pairs_with_plugin_dev(tmp_path: Path) -> None:
    """Stacked pull requests get no PR checks, so their CI is dispatched by hand."""
    summary = _run_step(
        tmp_path,
        sibling_ref="feat/stacked-branch",
        event_name="workflow_dispatch",
    )
    assert _resolved_ref(tmp_path) == "dev"
    assert "agent-coherence-plugin@dev" in summary


def test_a_same_named_twin_still_wins_over_the_dev_fallback(tmp_path: Path) -> None:
    summary = _run_step(
        tmp_path,
        sibling_ref="fix/paired-change",
        base_ref="dev",
        event_name="pull_request",
        plugin_branches=("main", "dev", "fix/paired-change"),
    )
    assert _resolved_ref(tmp_path) == "fix/paired-change"
    assert "fallback" not in summary


@pytest.mark.parametrize(
    ("base_ref", "event_name", "sibling_ref"),
    [
        ("main", "pull_request", "fix/no-such-plugin-branch"),
        ("", "push", "v0.5.0"),
    ],
    ids=["pull-request-into-main", "tag"],
)
def test_changes_bound_for_main_still_fall_back_to_plugin_main(
    tmp_path: Path, base_ref: str, event_name: str, sibling_ref: str
) -> None:
    """A release pull request and a tag prove parity against what ships."""
    summary = _run_step(
        tmp_path, sibling_ref=sibling_ref, base_ref=base_ref, event_name=event_name
    )
    assert _resolved_ref(tmp_path) == "main"
    assert "the fallback" in summary
    assert "33 commit(s) behind" in summary


def test_a_plugin_without_dev_degrades_to_main_rather_than_failing(tmp_path: Path) -> None:
    summary = _run_step(
        tmp_path,
        sibling_ref="fix/no-such-plugin-branch",
        base_ref="dev",
        event_name="pull_request",
        plugin_branches=("main",),
    )
    assert _resolved_ref(tmp_path) == "main"
    assert "the fallback" in summary
