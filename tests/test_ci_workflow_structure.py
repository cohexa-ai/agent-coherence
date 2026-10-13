# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The shape of ``.github/workflows/ci.yml``, read rather than trusted.

A workflow mistake does not fail a test: it fails silently on GitHub. An
invalid workflow reports no checks at all, so every required check on ``dev``
waits forever; a job with no timeout holds one of the org's shared job slots
for six hours when a worker hangs; an unpinned runner image changes the
toolchain between runs of the same commit. These tests load the workflow with ``yaml.safe_load`` and assert the
properties the CI layout depends on.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

RUNNER_IMAGE = "ubuntu-24.04"
MAX_JOB_TIMEOUT_MINUTES = 60
RESOLVE_STEP = "Resolve plugin ref"
PLUGIN_REPOSITORY = "Cohexa-ai/agent-coherence-plugin"
PLUGIN_LOCKFILE = "_plugin_checkout/package-lock.json"
PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
NEEDS_REFERENCE = re.compile(r"\bneeds\.(?P<job>[A-Za-z0-9_-]+)\.(?:outputs\.(?P<output>[A-Za-z0-9_-]+)|result)\b")


def _workflow() -> dict[str, Any]:
    return yaml.safe_load(CI_WORKFLOW.read_text())


def _jobs() -> dict[str, dict[str, Any]]:
    return _workflow()["jobs"]


def _strings(node: Any) -> Iterator[str]:
    """Every string anywhere under ``node``: conditions, env values, run bodies."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _needs(job: dict[str, Any]) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if isinstance(needs, str) else list(needs)


def _all_steps() -> Iterator[tuple[str, dict[str, Any]]]:
    for job_name, job in _jobs().items():
        for step in job.get("steps") or []:
            yield job_name, step


def _corpus_steps() -> list[dict[str, Any]]:
    return _jobs()["protocol-corpus"]["steps"]


def _index_of(steps: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> int:
    matches = [i for i, step in enumerate(steps) if predicate(step)]
    assert len(matches) == 1, f"expected exactly one matching step, found {len(matches)}"
    return matches[0]


def _is_plugin_checkout(step: dict[str, Any]) -> bool:
    return (
        str(step.get("uses", "")).startswith("actions/checkout@")
        and (step.get("with") or {}).get("repository") == PLUGIN_REPOSITORY
    )


def _is_setup_node(step: dict[str, Any]) -> bool:
    return str(step.get("uses", "")).startswith("actions/setup-node@")


def test_every_job_pins_the_runner_image_and_bounds_its_time() -> None:
    """A moving ``ubuntu-latest`` or a job with the default 6-hour timeout fails here."""
    for name, job in _jobs().items():
        assert job.get("runs-on") == RUNNER_IMAGE, f"{name} runs on {job.get('runs-on')!r}"
        timeout = job.get("timeout-minutes")
        assert isinstance(timeout, int) and 0 < timeout <= MAX_JOB_TIMEOUT_MINUTES, (
            f"{name} timeout-minutes is {timeout!r}"
        )


def test_only_pull_request_runs_cancel_each_other() -> None:
    """Runs group per PR and cancel there; every other event keys on its own run id.

    Pinned as literals: a group keyed on the commit would let a ``main`` push,
    a tag push and a dispatch on one commit cancel each other's pending runs.
    """
    concurrency = _workflow()["concurrency"]
    assert concurrency == {
        "group": (
            "${{ github.workflow }}-${{ github.event_name == 'pull_request'"
            " && format('pr-{0}', github.event.pull_request.number)"
            " || format('run-{0}', github.run_id) }}"
        ),
        "cancel-in-progress": "${{ github.event_name == 'pull_request' }}",
    }


def test_corpus_sets_up_node_22_after_the_plugin_checkout_with_an_npm_cache() -> None:
    """The npm cache keys on the plugin's lockfile, which exists only after its checkout."""
    steps = _corpus_steps()
    checkout = _index_of(steps, _is_plugin_checkout)
    setup_node = _index_of(steps, _is_setup_node)
    build = _index_of(steps, lambda step: step.get("name") == "Build Node coordinator")
    assert checkout < setup_node < build
    assert steps[setup_node]["with"] == {
        "node-version": "22",
        "cache": "npm",
        "cache-dependency-path": PLUGIN_LOCKFILE,
    }


def test_resolve_plugin_ref_keeps_its_contract() -> None:
    """The step that ``tests/test_parity_gate_ref_resolution.py`` executes stays where it is.

    Its name, id and environment are how that test finds and runs it, and the
    plugin checkout reads its output.
    """
    jobs_with_step = [job_name for job_name, step in _all_steps() if step.get("name") == RESOLVE_STEP]
    assert jobs_with_step == ["protocol-corpus"]

    steps = _corpus_steps()
    resolve = _index_of(steps, lambda candidate: candidate.get("name") == RESOLVE_STEP)
    step = steps[resolve]
    assert step["id"] == "plugin_ref"
    assert step["env"] == {
        "SIBLING_REF": "${{ github.head_ref || github.ref_name }}",
        "BASE_REF": "${{ github.base_ref }}",
        "EVENT_NAME": "${{ github.event_name }}",
        "REF_NAME": "${{ github.ref_name }}",
    }

    checkout = _index_of(steps, _is_plugin_checkout)
    assert resolve < checkout
    assert steps[checkout]["with"]["ref"] == "${{ steps.plugin_ref.outputs.ref }}"
    corpus_run = _index_of(steps, lambda candidate: candidate.get("name") == "Run protocol corpus")
    assert steps[corpus_run]["env"] == {
        "AGENT_COHERENCE_PLUGIN_DIST_PATH": ("${{ github.workspace }}/_plugin_checkout/dist/coordinator.js"),
    }


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    """A tag can be moved to other code; a full commit SHA cannot."""
    for job_name, step in _all_steps():
        if "uses" in step:
            assert PINNED_ACTION.match(step["uses"]), f"{job_name}: {step['uses']!r}"


def test_every_needs_entry_names_an_existing_job() -> None:
    """A ``needs`` on a missing job makes the whole workflow invalid: no checks report."""
    jobs = _jobs()
    for name, job in jobs.items():
        for needed in _needs(job):
            assert needed in jobs, f"{name} needs unknown job {needed!r}"


def test_every_needs_reference_is_declared() -> None:
    """``needs.<job>.outputs.<key>`` resolves only for a listed job and a declared output.

    GitHub exposes a job's outputs only to jobs that list it in ``needs``; an
    unlisted job reads as empty, which silently flips any condition built on it.
    """
    jobs = _jobs()
    for name, job in jobs.items():
        for text in _strings(job):
            for match in NEEDS_REFERENCE.finditer(text):
                needed, output = match["job"], match["output"]
                assert needed in _needs(job), f"{name} reads needs.{needed} without needing it"
                if output is not None:
                    declared = jobs[needed].get("outputs") or {}
                    assert output in declared, f"{name} reads undeclared {needed}.outputs.{output}"


# The names `dev`'s branch protection requires. Frozen on purpose (tests/CLAUDE.md):
# a job renamed in ci.yml leaves its required check pending forever, so the
# rename has to be a deliberate edit here too.
REQUIRED_CHECKS = frozenset(
    {
        "Lint",
        "Architecture Checks",
        "Tests (Python 3.11)",
        "Tests (Python 3.12)",
        "Benchmark Smoke",
        "Benchmark Regression",
        "TLA+ Model Check",
        "Protocol Corpus (Python ↔ Node parity)",
        "Build Package",
    }
)
REQUIRED_CHECK_COUNT = 9
TEST_GATES = {"tests-py311": "test-shards-py311", "tests-py312": "test-shards-py312"}
TEST_GATE_PYTHON = {"tests-py311": "3.11", "tests-py312": "3.12"}
OTHER_WORKFLOWS = ("publish-mcp.yml", "release.yml", "weekly-full-ci.yml")
COVERAGE_STEP = "Check that the shards ran every test exactly once"
VERDICT_STEP = "Decide the verdict"
TESTS_VERDICT_RUN = 'python3 tools/ci_gate_verdict.py tests --shards "$SHARDS" --coverage-exit "$COVERAGE_EXIT"'
# `dev` brings the xdist and pytest-split the lanes need; `diagnose` and `mcp`
# bring modules whose tests tests/conftest.py drops from collection when they
# are missing, which the coverage check cannot see.
SHARD_INSTALL = 'python -m pip install -e ".[dev,diagnose,mcp]"'


def _job_names() -> dict[str, str]:
    return {job_id: job.get("name", job_id) for job_id, job in _jobs().items()}


def test_every_required_check_is_one_non_matrix_job() -> None:
    """A required name on a matrix job that GitHub skips as a whole renders
    unexpanded and never reports; a required name on two jobs is ambiguous."""
    assert len(REQUIRED_CHECKS) == REQUIRED_CHECK_COUNT
    names = list(_job_names().values())
    for check in REQUIRED_CHECKS:
        assert names.count(check) == 1, f"{check!r} names {names.count(check)} jobs"
    jobs = _jobs()
    for job_id, name in _job_names().items():
        if name in REQUIRED_CHECKS:
            assert "strategy" not in jobs[job_id], f"required check {name!r} is a matrix job"
    for workflow in OTHER_WORKFLOWS:
        other = yaml.safe_load((CI_WORKFLOW.parent / workflow).read_text())
        other_names = {job.get("name", job_id) for job_id, job in other["jobs"].items()}
        assert not other_names & REQUIRED_CHECKS, f"{workflow} reuses a required name"


def test_the_corpus_needs_nothing_and_build_and_smoke_need_only_the_plan() -> None:
    """A needed job that is skipped or failed cascades; the corpus must run
    whatever else happens, and Build Package no longer waits on the long jobs."""
    jobs = _jobs()
    assert "needs" not in jobs["protocol-corpus"]
    assert _needs(jobs["build"]) == ["plan"]
    assert _needs(jobs["benchmark-smoke"]) == ["plan"]


def test_plan_consumers_run_unless_cancelled() -> None:
    """A plan job that died must not cascade-skip its consumers: a skipped
    required check would read as passing. Gates run under always(); every
    other consumer's condition starts with !cancelled()."""
    for job_id, job in _jobs().items():
        if "plan" in _needs(job):
            condition = job.get("if", "")
            assert condition == "${{ always() }}" or condition.startswith("${{ !cancelled()"), (
                f"{job_id} would cascade-skip on a dead plan job"
            )


def test_the_test_gates_always_run_and_judge_their_shards() -> None:
    """A gate skipped or cancelled with its run would leave the required check
    in a state GitHub might count as passing; under always() it reports, and
    its verdict reads its own shards' result and the coverage check's exit
    code exactly. A hard-wired exit, an `|| true` or another leg's artifacts
    would each pass the gate with nothing proven."""
    jobs = _jobs()
    for gate, shards in TEST_GATES.items():
        job = jobs[gate]
        assert job["if"] == "${{ always() }}"
        assert _needs(job) == [shards]
        download = _step(gate, "Download every shard's lists")
        assert download["with"] == {"pattern": f"test-results-py{TEST_GATE_PYTHON[gate]}-shard-*", "path": "shards"}
        coverage = _step(gate, COVERAGE_STEP)
        assert coverage["id"] == "coverage"
        assert '--root shards --shards "$TEST_SHARD_COUNT"' in coverage["run"]
        assert _step(gate, VERDICT_STEP) == {
            "name": VERDICT_STEP,
            "env": {
                "SHARDS": f"${{{{ needs.{shards}.result }}}}",
                "COVERAGE_EXIT": "${{ steps.coverage.outputs.exit }}",
            },
            "run": TESTS_VERDICT_RUN,
        }
        _assert_gate_cannot_swallow_a_failure(gate, "Download every shard's lists")


def _assert_gate_cannot_swallow_a_failure(gate: str, tolerant_step: str | None) -> None:
    """A gate job reports its verdict step's exit, unless something lets that
    step fail quietly: `continue-on-error` on the job or the step turns a FAIL
    green, and an `if:` on a step can skip the verdict altogether. Only the
    named download step may tolerate a failure, because a missing artifact
    already reads as "cannot tell" to the coverage check."""
    job = _jobs()[gate]
    assert "continue-on-error" not in job, f"{gate} would pass with a failed verdict"
    for step in job["steps"]:
        if step.get("name") == tolerant_step:
            continue
        assert "continue-on-error" not in step and "if" not in step, (
            f"{gate}: {step.get('name')!r} can swallow a failure"
        )


def _write_stub(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)


@pytest.mark.parametrize("check_exit", [0, 1, 2])
def test_the_coverage_step_hands_the_checks_exit_code_to_the_verdict(tmp_path: Path, check_exit: int) -> None:
    """The coverage step runs under GitHub's `bash -e`, yet must finish and
    record the coverage check's own exit code, whatever it is: the verdict
    step reads only that value, so recording another command's status (or
    stopping before the write) would let a failed or unprovable run pass."""
    stub = f"import sys\nprint('coverage stub')\nsys.exit({check_exit})\n"
    _write_stub(tmp_path / "tools" / "check_shard_coverage.py", stub)
    _write_stub(tmp_path / "tools" / "summarize_junit.py", "print('junit stub')\n")
    output, summary = tmp_path / "github_output", tmp_path / "step_summary"
    env = {
        "PATH": os.environ["PATH"],
        "TEST_SHARD_COUNT": "2",
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
    }
    for gate in TEST_GATES:
        output.write_text("")
        proc = subprocess.run(
            ["bash", "-e", "-c", _step(gate, COVERAGE_STEP)["run"]],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert output.read_text().splitlines() == [f"exit={check_exit}"]


def _shard_steps(job_id: str) -> list[dict[str, Any]]:
    return _jobs()[job_id]["steps"]


def _normalized_leg(job_id: str, version: str) -> str:
    return yaml.safe_dump(_jobs()[job_id]).replace(version, "X").replace("py" + version.replace(".", ""), "pyX")


def test_both_python_legs_run_identical_shards() -> None:
    """The legs must differ only in the Python version; a step that exists in
    one leg would silently check less in the other."""
    assert _normalized_leg("test-shards-py311", "3.11") == _normalized_leg("test-shards-py312", "3.12")


def test_both_python_legs_judge_identically() -> None:
    """The gates must differ only in the Python version too; a gate reading
    the other leg's lists would prove that leg's run, not its own."""
    assert _normalized_leg("tests-py311", "3.11") == _normalized_leg("tests-py312", "3.12")


def test_both_legs_install_every_extra_the_suite_collects() -> None:
    """tests/conftest.py drops whole modules from collection when an optional
    extra is missing, and the coverage check only proves each *collected* test
    ran once: an extra dropped from the install line would shrink the suite
    with every check green."""
    for job_id in TEST_GATES.values():
        installs = [step["run"] for step in _shard_steps(job_id) if "pip install -e" in step.get("run", "")]
        assert len(installs) == 1, f"{job_id} has {len(installs)} editable installs"
        assert installs[0].splitlines()[-1] == SHARD_INSTALL


def test_the_shard_matrix_matches_the_shard_count() -> None:
    """pytest-split's --splits, the matrix and the gate's --shards must agree,
    or a group would never run and the coverage check would see too few shards."""
    count = int(_workflow()["env"]["TEST_SHARD_COUNT"])
    for job_id in TEST_GATES.values():
        assert _jobs()[job_id]["strategy"]["matrix"]["shard"] == list(range(1, count + 1))


def _step(job_id: str, name: str) -> dict[str, Any]:
    steps = [step for step in _shard_steps(job_id) if step.get("name") == name]
    assert len(steps) == 1, f"{job_id} has {len(steps)} {name!r} steps"
    return steps[0]


def test_the_lanes_run_as_tests_conftest_expects() -> None:
    """The lane is a command-line option, never `-m` (which would replace the
    addopts exclusions); the serial lane runs without workers and still runs
    after a parallel-lane failure; the node-id directory is one argument, or
    pytest takes the existing directory for a test path; the upload survives
    a failed or cancelled shard so the gate can see what happened."""
    for job_id in TEST_GATES.values():
        parallel = _step(job_id, "Run the parallel lane")
        serial = _step(job_id, "Run the serial lane")
        upload = _step(job_id, "Upload node-id lists and test reports")
        assert "-n 4 --dist worksteal --test-lane parallel" in parallel["run"]
        assert "-n 0 --test-lane serial" in serial["run"]
        assert serial["if"] == "${{ !cancelled() }}"
        assert upload["if"] == "${{ always() }}"
        for step in (parallel, serial):
            assert " -m " not in f" {step['run']} "
            assert '"--node-id-dir=lists/shard-$SHARD"' in step["run"]
            assert "--durations-path tests/.test_durations" in step["run"]


MAKEFILE = REPO_ROOT / "Makefile"
TLA_BINS = ("tla-check-1", "tla-check-2", "tla-check-3")
# The nine (spec, config) pairs `make tla-check` has always run. Frozen on
# purpose: a bin that dropped one would check less with every job green.
# Every config is the spec's _CI one except MESI_Standalone, which has none;
# AtomicPublish stays out (its state space does not converge in CI).
TLA_SPEC_CONFIGS = frozenset(
    {
        ("MESI_Standalone.tla", "MESI_Standalone.cfg"),
        ("CrashRecovery.tla", "CrashRecovery_CI.cfg"),
        ("OCC.tla", "OCC_CI.cfg"),
        ("Fencing.tla", "Fencing_CI.cfg"),
        ("ZombieRevoke.tla", "ZombieRevoke_CI.cfg"),
        ("EffectGate.tla", "EffectGate_CI.cfg"),
        ("Retention.tla", "Retention_CI.cfg"),
        ("Snapshot.tla", "Snapshot_CI.cfg"),
        ("WorkspaceVersion.tla", "WorkspaceVersion_CI.cfg"),
    }
)
TLA_SPEC_CONFIG_COUNT = 9
TLC_RUN = re.compile(r"\$\(TLC\) -config formal/tla/(\S+) formal/tla/(\S+)")


def _make_targets() -> dict[str, tuple[list[str], list[str]]]:
    """Each Makefile target's prerequisites and recipe lines."""
    targets: dict[str, tuple[list[str], list[str]]] = {}
    current = None
    for line in MAKEFILE.read_text().splitlines():
        rule = re.match(r"^([A-Za-z0-9_.-]+):(?!=)\s*([^#]*)", line)
        if rule and not line.startswith("\t"):
            current = rule.group(1)
            targets[current] = (rule.group(2).split(), [])
        elif line.startswith("\t") and current is not None:
            targets[current][1].append(line.strip())
        else:
            current = None
    return targets


def _tlc_pairs(recipe: list[str]) -> list[tuple[str, str]]:
    return [(spec, config) for config, spec in (TLC_RUN.search(line).groups() for line in recipe if "$(TLC)" in line)]


def test_the_tla_bins_cover_exactly_the_nine_specs_with_their_configs() -> None:
    """Splitting the sweep must not drop, repeat or re-configure a spec; a
    lower bound in a config silently checks less."""
    assert len(TLA_SPEC_CONFIGS) == TLA_SPEC_CONFIG_COUNT
    targets = _make_targets()
    pairs = [pair for bin_target in TLA_BINS for pair in _tlc_pairs(targets[bin_target][1])]
    assert len(pairs) == len(set(pairs)), "a spec runs in two bins"
    assert set(pairs) == TLA_SPEC_CONFIGS
    assert targets["tla-check"][0] == list(TLA_BINS)
    assert not _tlc_pairs(targets["tla-check"][1]), "tla-check must run only through its bins"


def test_every_tla_target_checks_java_first_and_never_cleans_up() -> None:
    """A bin CI calls directly still checks Java; `-cleanup` deletes the shared
    states/ directory under the other bins."""
    targets = _make_targets()
    for bin_target in TLA_BINS:
        assert "tla-java-check" in targets[bin_target][0]
    makefile = MAKEFILE.read_text()
    assert "-cleanup" not in makefile
    assert re.search(r"^\.NOTPARALLEL:", makefile, re.MULTILINE), "make -j would start three -workers auto JVMs"


def test_the_tla_bins_skip_only_on_a_pull_request_the_plan_cleared() -> None:
    """The skip needs both the event and the exact string `false`: an empty,
    missing or misspelled plan output runs the bins, and no other event can
    skip them."""
    job = _jobs()["tla-bins"]
    assert job["if"] == (
        "${{ !cancelled() && !(github.event_name == 'pull_request' && needs.plan.outputs.tla-needed == 'false') }}"
    )
    assert job["strategy"]["matrix"]["bin"] == [1, 2, 3]
    assert [step.get("run") for step in job["steps"] if "run" in step] == ['make "tla-check-$BIN"']


def test_the_tla_gate_always_runs_and_judges_the_bins() -> None:
    """The gate carries the required name; it must report on every outcome and
    read the bins' result, the event and the plan's decision."""
    job = _jobs()["tla-check"]
    assert job["name"] == "TLA+ Model Check"
    assert job["if"] == "${{ always() }}"
    assert _needs(job) == ["plan", "tla-bins"]
    assert job["steps"][-1] == {
        "name": VERDICT_STEP,
        "env": {
            "BINS": "${{ needs.tla-bins.result }}",
            "EVENT": "${{ github.event_name }}",
            "PLAN": "${{ needs.plan.result }}",
            "TLA_NEEDED": "${{ needs.plan.outputs.tla-needed }}",
        },
        "run": (
            'python3 tools/ci_gate_verdict.py tla --bins "$BINS" --event "$EVENT" --plan "$PLAN" --tla-needed "$TLA_NEEDED"'
        ),
    }
    _assert_gate_cannot_swallow_a_failure("tla-check", None)


def _is_checkout(step: dict[str, Any]) -> bool:
    return str(step.get("uses", "")).startswith("actions/checkout@")


def test_the_plan_job_checks_out_the_base_with_the_merge() -> None:
    """tools/ci_plan.py diffs the merge commit against its first parent; at
    the default depth of 1 that parent is missing, the diff fails, and every
    pull request runs the bins without anything turning red."""
    steps = _jobs()["plan"]["steps"]
    checkout = steps[_index_of(steps, _is_checkout)]
    assert checkout["with"] == {"fetch-depth": 2}


WEEKLY_WORKFLOW = CI_WORKFLOW.parent / "weekly-full-ci.yml"
WEEKLY_JOB = "full-run"
WEEKLY_STEP = "Dispatch ci.yml on dev and wait for that run"
# Minutes a dispatched run may sit queued behind other runs on the org's
# shared job cap before its own jobs start.
WEEKLY_QUEUE_MARGIN_MINUTES = 15
WEEKLY_CRON = re.compile(r"^\d{1,2} \d{1,2} \* \* [0-6]$")
RUN_TAG_RUN_NAME = "${{ inputs.run_tag && format('Weekly full CI ({0})', inputs.run_tag) || '' }}"


def _weekly() -> dict[str, Any]:
    return yaml.safe_load(WEEKLY_WORKFLOW.read_text())


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # YAML 1.1 reads a bare `on` key as the boolean True.
    return workflow.get("on", workflow.get(True)) or {}


def _weekly_step() -> dict[str, Any]:
    steps = [step for step in _weekly()["jobs"][WEEKLY_JOB]["steps"] if step.get("name") == WEEKLY_STEP]
    assert len(steps) == 1, f"the weekly job has {len(steps)} {WEEKLY_STEP!r} steps"
    return steps[0]


def test_the_weekly_run_fires_once_a_week_and_by_hand() -> None:
    """One cron entry that names a single weekday; a second entry or a range
    would run the full suite more often than the job cap was sized for."""
    triggers = _triggers(_weekly())
    assert set(triggers) == {"schedule", "workflow_dispatch"}
    crons = [entry["cron"] for entry in triggers["schedule"]]
    assert len(crons) == 1 and WEEKLY_CRON.match(crons[0]), crons


def test_the_weekly_run_can_only_dispatch_and_read_runs() -> None:
    """A scheduled job runs from `main` with the repository's token; anything
    wider than `actions: write` would let it write to the repository. It needs
    no checkout, so it uses no actions at all."""
    weekly = _weekly()
    assert weekly["permissions"] == {"actions": "write"}
    assert list(weekly["jobs"]) == [WEEKLY_JOB]
    job = weekly["jobs"][WEEKLY_JOB]
    assert job["runs-on"] == RUNNER_IMAGE
    assert "permissions" not in job
    assert all("uses" not in step for step in job["steps"])
    env = _weekly_step()["env"]
    assert env["CI_REF"] == "dev"
    # The workflow token is what `permissions` limits, and its failure is what
    # notifies the last editor of the schedule; a stored token would bypass both.
    assert env["GH_TOKEN"] == "${{ github.token }}"
    assert env["GH_REPO"] == "${{ github.repository }}"


def _longest_ci_chain_minutes() -> int:
    """The longest sum of `timeout-minutes` along any chain of `needs` in ci.yml."""
    jobs = _jobs()
    memo: dict[str, int] = {}

    def chain(job_id: str) -> int:
        if job_id not in memo:
            job = jobs[job_id]
            memo[job_id] = job["timeout-minutes"] + max((chain(need) for need in _needs(job)), default=0)
        return memo[job_id]

    return max(chain(job_id) for job_id in jobs)


def test_the_weekly_run_outlasts_the_longest_ci_run() -> None:
    """The scheduler waits for the run it dispatched. Its own wait must cover
    the longest CI run plus queue time, and finish before the job's timeout,
    so a slow run ends with the scheduler's own message instead of a kill."""
    env = _weekly_step()["env"]
    find_seconds, wait_seconds = int(env["FIND_TIMEOUT_SECONDS"]), int(env["WAIT_TIMEOUT_SECONDS"])
    assert wait_seconds >= (_longest_ci_chain_minutes() + WEEKLY_QUEUE_MARGIN_MINUTES) * 60
    timeout = _weekly()["jobs"][WEEKLY_JOB]["timeout-minutes"]
    assert (find_seconds + wait_seconds) / 60 + 1 <= timeout <= MAX_JOB_TIMEOUT_MINUTES


def test_ci_names_a_dispatched_run_after_its_tag() -> None:
    """The scheduler finds its own run by the tag in the run's name, so a
    concurrent dispatch or push is never mistaken for it; every other run
    keeps GitHub's default name, because an empty name falls back to it."""
    workflow = _workflow()
    assert workflow["run-name"] == RUN_TAG_RUN_NAME
    run_tag = _triggers(workflow)["workflow_dispatch"]["inputs"]["run_tag"]
    assert run_tag["type"] == "string" and run_tag["required"] is False and run_tag["default"] == ""


def _dispatched_title_format() -> str:
    """The title ci.yml gives a tagged run, read from ci.yml itself, so the
    fake below can only answer with names the real workflow produces."""
    match = re.search(r"format\('([^']*\{0\}[^']*)'", _workflow()["run-name"])
    assert match, _workflow()["run-name"]
    return match.group(1)


FAKE_GH = """#!/usr/bin/env python3
# Stands in for gh. Each call must have the exact shape the step uses, or
# the fake exits 2. The dispatched run is in progress until a `run watch`
# succeeds; a watch listed as failing in FAKE_WATCH_EXITS leaves it running.
import json, os, sys

args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
tag_file, state_file = os.environ["FAKE_GH_TAG"], os.environ["FAKE_GH_STATE"]
title = os.environ["FAKE_TITLE_FORMAT"].replace("{0}", "%s")


def bump(name):
    path = os.environ["FAKE_GH_DIR"] + "/" + name
    count = (int(open(path).read()) if os.path.exists(path) else 0) + 1
    with open(path, "w") as out:
        out.write(str(count))
    return count


if args[:2] == ["workflow", "run"]:
    with open(tag_file, "w") as out:
        out.write(next(a.split("=", 1)[1] for a in args if a.startswith("run_tag=")))
    sys.exit(0)
if args[:2] == ["run", "list"]:
    expected = ["--workflow", "ci.yml", "--branch", "dev", "--event", "workflow_dispatch"]
    if any(args[args.index(flag) + 1] != value for flag, value in zip(expected[::2], expected[1::2]) if flag in args) \\
            or not all(flag in args for flag in expected[::2]):
        sys.exit(2)
    fields = args[args.index("--json") + 1].split(",") if "--json" in args else []
    if not {"databaseId", "displayTitle"} <= set(fields):
        sys.exit(2)
    if bump("list-calls") <= int(os.environ.get("FAKE_LIST_FAILS", "0")):
        sys.exit(1)
    # A decoy whose tag starts with ours: a plain substring match would take it.
    runs = [{"databaseId": 11, "displayTitle": title % "weekly-9-10"}]
    if os.environ.get("FAKE_RUN_APPEARS") == "1" and os.path.exists(tag_file):
        runs.append({"databaseId": 4242, "displayTitle": title % open(tag_file).read()})
    print(json.dumps([{field: run[field] for field in fields} for run in runs]))
    sys.exit(0)
if args[:2] == ["run", "watch"]:
    exits = [int(code) for code in os.environ.get("FAKE_WATCH_EXITS", "").split(",") if code]
    call = bump("watch-calls")
    code = exits[call - 1] if call <= len(exits) else 0
    if code == 0 and args[2] == "4242":
        with open(state_file, "w") as out:
            out.write("completed")
    sys.exit(code)
if args[:2] == ["run", "view"]:
    if args[2:] != ["4242", "--json", "status,conclusion", "--jq", '.status + " " + .conclusion']:
        sys.exit(2)
    if os.path.exists(state_file):
        print("completed " + os.environ["FAKE_CONCLUSION"])
    else:
        print("in_progress ")
    sys.exit(0)
sys.exit(2)
"""


def _run_weekly_step(tmp_path: Path, **fake: str) -> tuple[subprocess.CompletedProcess[str], list[list[str]], str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)
    log, summary = tmp_path / "gh.log", tmp_path / "summary.md"
    step = _weekly_step()
    env = {
        **{key: value for key, value in step["env"].items() if not str(value).startswith("${{")},
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_RUN_ID": "9",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_STEP_SUMMARY": str(summary),
        # Generous deadlines: a passing case returns as soon as it is done, and
        # a loaded test runner must not turn a slow loop into a failure.
        "FIND_TIMEOUT_SECONDS": "30",
        "WAIT_TIMEOUT_SECONDS": "30",
        "POLL_SECONDS": "0",
        "WATCH_SECONDS": "1",
        "FAKE_GH_LOG": str(log),
        "FAKE_GH_DIR": str(tmp_path),
        "FAKE_GH_TAG": str(tmp_path / "tag"),
        "FAKE_GH_STATE": str(tmp_path / "state"),
        "FAKE_TITLE_FORMAT": _dispatched_title_format(),
        **fake,
    }
    proc = subprocess.run(
        ["bash", "-e", "-c", step["run"]], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc, calls, summary.read_text() if summary.exists() else ""


def _watched_ids(calls: list[list[str]]) -> set[str]:
    return {call[2] for call in calls if call[:2] in (["run", "watch"], ["run", "view"])}


def test_the_weekly_run_passes_only_when_the_run_it_dispatched_succeeded(tmp_path: Path) -> None:
    """It dispatches ci.yml on dev with its own tag, waits for that run (not
    the decoy whose tag merely starts with its own), and reports its result."""
    proc, calls, summary = _run_weekly_step(tmp_path, FAKE_RUN_APPEARS="1", FAKE_CONCLUSION="success")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert calls[0] == ["workflow", "run", "ci.yml", "--ref", "dev", "-f", "run_tag=weekly-9-1"]
    assert _watched_ids(calls) == {"4242"}
    assert "success" in summary


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "timed_out", ""])
def test_the_weekly_run_fails_unless_the_run_it_dispatched_succeeded(tmp_path: Path, conclusion: str) -> None:
    """A failed or cancelled run, or a completed run whose result cannot be
    read, turns the weekly run red; only `success` passes."""
    proc, _, _ = _run_weekly_step(tmp_path, FAKE_RUN_APPEARS="1", FAKE_CONCLUSION=conclusion)
    assert proc.returncode != 0
    assert f"ended {conclusion or 'with no readable result'}" in proc.stdout


def test_the_weekly_run_keeps_waiting_when_a_watch_ends_early(tmp_path: Path) -> None:
    """`gh run watch` exits on any failed API call while the run is still
    going; the scheduler checks the run's status and watches again instead
    of reporting a failure that has not happened."""
    proc, calls, _ = _run_weekly_step(tmp_path, FAKE_RUN_APPEARS="1", FAKE_CONCLUSION="success", FAKE_WATCH_EXITS="1,1")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sum(call[:2] == ["run", "watch"] for call in calls) == 3


def test_the_weekly_run_says_so_when_its_run_is_still_going_at_the_deadline(tmp_path: Path) -> None:
    """A run that never completes inside the wait is not reported as failed:
    the message says it was still running, and the weekly run still fails."""
    proc, _, _ = _run_weekly_step(
        tmp_path,
        FAKE_RUN_APPEARS="1",
        FAKE_CONCLUSION="success",
        FAKE_WATCH_EXITS=",".join(["1"] * 50),
        WAIT_TIMEOUT_SECONDS="2",
    )
    assert proc.returncode != 0
    assert "still running" in proc.stdout


def test_the_weekly_run_fails_when_its_run_never_appears(tmp_path: Path) -> None:
    """If the dispatched run cannot be found, nothing was proven: fail rather
    than watch the decoy or pass."""
    proc, calls, _ = _run_weekly_step(
        tmp_path, FAKE_RUN_APPEARS="0", FAKE_CONCLUSION="success", FIND_TIMEOUT_SECONDS="3"
    )
    assert proc.returncode != 0
    assert "never appeared" in proc.stdout
    assert not _watched_ids(calls)


def test_the_weekly_run_rides_out_a_failing_run_listing(tmp_path: Path) -> None:
    """A transient API error while looking for the run is retried until the deadline."""
    proc, _, _ = _run_weekly_step(tmp_path, FAKE_RUN_APPEARS="1", FAKE_CONCLUSION="success", FAKE_LIST_FAILS="2")
    assert proc.returncode == 0, proc.stdout + proc.stderr
