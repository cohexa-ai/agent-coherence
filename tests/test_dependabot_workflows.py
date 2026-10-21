# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The Dependabot gate and merger workflows, read and run rather than trusted.

dependabot-gate.yml judges each Dependabot PR and records the verdict as a
commit status; dependabot-merge.yml merges what the gate qualified, with the
forward-merge App's token. Two properties matter more than any other here:
the gate can never be talked into a verdict by the PR it judges (it runs the
rules as dev has them, and the job that writes statuses runs no third-party
code), and the App's private key can never be read by a run a pull request
controls (it lives in an environment only dev may use, in a job that checks
out nothing). The behaviour tests run each workflow's own shell, under the
shell the workflow declares, against a strict fake ``gh`` that rejects any
call shape the step does not use.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"
GATE_WORKFLOW = WORKFLOWS / "dependabot-gate.yml"
MERGE_WORKFLOW = WORKFLOWS / "dependabot-merge.yml"
RUNNER_IMAGE = "ubuntu-24.04"
PINNED_ACTION = re.compile(r"^[^@\s]+@[0-9a-f]{40}$")
FETCH_METADATA = "dependabot/fetch-metadata@25dd0e34f4fe68f24cc83900b1fe3fe149efef98"
APP_TOKEN_ACTION = "actions/create-github-app-token@64ab729a6294000f5aa4a4bc0b73d1f32bc45510"
JUDGE_STEP = "Judge the update"
RECORD_STEP = "Record the verdict on the PR's head"
PLAN_STEP = "Pick the next step"
ACT_STEP = "Merge or update the PR"
DEPENDABOT_ONLY = (
    "github.event.pull_request.user.login == 'dependabot[bot]' && "
    "!startsWith(github.head_ref, 'dependabot/github_actions/')"
)
READ_ONLY = {"contents": "read", "pull-requests": "read", "checks": "read", "statuses": "read", "actions": "read"}
# GitHub runs `shell: bash` as this; with no shell named it runs `bash -e {0}`,
# which has no pipefail.
BASH = ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c"]
MINOR = "version-update:semver-minor"
SHA_A, SHA_B, SHA_C = "a" * 40, "b" * 40, "c" * 40


def _load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # YAML 1.1 reads a bare `on` key as the boolean True.
    return workflow.get("on", workflow.get(True)) or {}


def _step(workflow: dict[str, Any], job: str, name: str) -> dict[str, Any]:
    steps = [step for step in workflow["jobs"][job]["steps"] if step.get("name") == name]
    assert len(steps) == 1, f"{job} has {len(steps)} {name!r} steps"
    return steps[0]


@pytest.mark.parametrize("path", [GATE_WORKFLOW, MERGE_WORKFLOW], ids=lambda path: path.name)
def test_every_job_pins_its_runner_bounds_its_time_and_pins_its_actions(path: Path) -> None:
    for name, job in _load(path)["jobs"].items():
        assert job["runs-on"] == RUNNER_IMAGE, f"{name} runs on {job['runs-on']!r}"
        assert isinstance(job["timeout-minutes"], int) and 0 < job["timeout-minutes"] <= 10
        for step in job["steps"]:
            if "uses" in step:
                assert PINNED_ACTION.match(step["uses"]), f"{name}: {step['uses']!r}"


@pytest.mark.parametrize("path", [GATE_WORKFLOW, MERGE_WORKFLOW], ids=lambda path: path.name)
def test_every_step_runs_under_bash_with_pipefail(path: Path) -> None:
    """With no shell named, GitHub runs `bash -e`, and a failure before a pipe passes."""
    workflow = _load(path)
    assert workflow["defaults"] == {"run": {"shell": "bash"}}
    for job in workflow["jobs"].values():
        assert all("shell" not in step for step in job["steps"])


def test_the_gate_judges_every_dependabot_pr_into_dev_except_actions_bumps() -> None:
    """pull_request, never pull_request_target, and never an Actions bump: a
    PR runs this workflow as the PR has it, so an Actions bump would run its
    new action code here."""
    gate = _load(GATE_WORKFLOW)
    assert _triggers(gate) == {"pull_request": {"branches": ["dev"], "types": ["opened", "synchronize", "reopened"]}}
    assert gate["concurrency"] == {
        "group": "dependabot-gate-${{ github.event.pull_request.number }}",
        "cancel-in-progress": True,
    }
    assert list(gate["jobs"]) == ["judge", "record"]
    assert gate["jobs"]["judge"]["if"] == DEPENDABOT_ONLY
    assert gate["jobs"]["record"]["if"] == f"always() && {DEPENDABOT_ONLY}"


def test_only_the_record_job_writes_and_it_runs_no_action() -> None:
    """The job that runs third-party actions can only read; the one that writes
    statuses runs none and writes only to this PR's own head."""
    gate = _load(GATE_WORKFLOW)
    judge, record = gate["jobs"]["judge"], gate["jobs"]["record"]
    assert gate["permissions"] == {}
    assert judge["permissions"] == {"contents": "read", "pull-requests": "read"}
    assert record["permissions"] == {"statuses": "write"}
    assert record["needs"] == "judge"
    assert all("uses" not in step for step in record["steps"])
    assert "env" not in judge and "env" not in record
    assert _step(gate, "record", RECORD_STEP)["env"] == {
        "JUDGE_RESULT": "${{ needs.judge.result }}",
        "GH_TOKEN": "${{ github.token }}",
        "GH_REPO": "${{ github.repository }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
        "STATE": "${{ needs.judge.outputs.state }}",
        "DESCRIPTION": "${{ needs.judge.outputs.description }}",
        "RUN_URL": "${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}",
    }
    assert judge["outputs"] == {
        "state": "${{ steps.judge.outputs.state }}",
        "description": "${{ steps.judge.outputs.description }}",
    }
    assert "secrets." not in GATE_WORKFLOW.read_text()


def test_the_judge_reads_the_metadata_and_checks_out_the_head_without_credentials() -> None:
    gate = _load(GATE_WORKFLOW)
    steps = gate["jobs"]["judge"]["steps"]
    assert steps[0]["uses"] == FETCH_METADATA
    assert "with" not in steps[0], "alert-lookup and friends call the API with wider needs"
    assert steps[1]["uses"].startswith("actions/checkout@")
    assert steps[1]["with"] == {
        "ref": "${{ github.event.pull_request.head.sha }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert _step(gate, "judge", JUDGE_STEP)["env"] == {
        "ECOSYSTEM": "${{ steps.metadata.outputs.package-ecosystem }}",
        "UPDATE_TYPE": "${{ steps.metadata.outputs.update-type }}",
        "GROUP": "${{ steps.metadata.outputs.dependency-group }}",
        "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    }


def test_the_merger_runs_on_a_schedule_or_by_hand_and_never_for_a_pull_request() -> None:
    """A pull request event would run the workflow as the PR has it."""
    assert _triggers(_load(MERGE_WORKFLOW)) == {"schedule": [{"cron": "*/10 * * * *"}], "workflow_dispatch": None}


def test_the_merger_writes_only_through_the_app_token() -> None:
    merge = _load(MERGE_WORKFLOW)
    assert merge["permissions"] == READ_ONLY
    assert merge["concurrency"] == {"group": "dependabot-merge", "cancel-in-progress": False}
    assert list(merge["jobs"]) == ["plan", "act"]
    for job in merge["jobs"].values():
        assert "permissions" not in job


def test_the_plan_hands_act_exactly_the_step_it_picked() -> None:
    """A miswired output leaves the head empty, and gh drops an empty
    --match-head-commit, which would merge whatever the head is by then."""
    merge = _load(MERGE_WORKFLOW)
    plan, act = merge["jobs"]["plan"], merge["jobs"]["act"]
    assert plan["outputs"] == {
        "action": "${{ steps.plan.outputs.action }}",
        "number": "${{ steps.plan.outputs.number }}",
        "head": "${{ steps.plan.outputs.head }}",
    }
    assert act["env"] == {
        "GH_REPO": "${{ github.repository }}",
        "ACTION": "${{ needs.plan.outputs.action }}",
        "PR": "${{ needs.plan.outputs.number }}",
        "HEAD_SHA": "${{ needs.plan.outputs.head }}",
    }
    assert act["needs"] == "plan"
    assert act["if"] == "needs.plan.outputs.action != 'none' || github.event_name == 'workflow_dispatch'"
    step = _step(merge, "act", ACT_STEP)
    assert step["if"] == "env.ACTION != 'none'"
    assert step["env"] == {"GH_TOKEN": "${{ steps.app-token.outputs.token || steps.update-token.outputs.token }}"}


def test_only_the_act_job_can_reach_the_key_and_it_runs_only_the_token_action() -> None:
    """The key sits in an environment only dev may use; the job that holds it
    checks out nothing. Only a branch update gets `workflows`, which GitHub
    requires to merge dev's workflow changes into a branch."""
    merge = _load(MERGE_WORKFLOW)
    plan, act = merge["jobs"]["plan"], merge["jobs"]["act"]
    assert "environment" not in plan
    assert act["environment"] == "dependabot-merge"
    assert {step["uses"] for step in act["steps"] if "uses" in step} == {APP_TOKEN_ACTION}
    key = {"client-id": "${{ vars.FORWARD_MERGE_APP_CLIENT_ID }}", "private-key": "${{ secrets.BOT_KEY }}"}
    writes = {"permission-contents": "write", "permission-pull-requests": "write"}
    mint = _step(merge, "act", "Mint the App token")
    assert (mint["id"], mint["if"], mint["with"]) == ("app-token", "env.ACTION != 'update'", {**key, **writes})
    update = _step(merge, "act", "Mint the App token for a branch update")
    assert (update["id"], update["if"]) == ("update-token", "env.ACTION == 'update'")
    assert update["with"] == {**key, **writes, "permission-workflows": "write"}
    refuse = _step(merge, "plan", "Refuse a repository- or organization-level copy of the key")
    assert refuse["env"] == {"KEY_OUTSIDE_ENVIRONMENT": "${{ secrets.BOT_KEY != '' }}"}
    assert MERGE_WORKFLOW.read_text().count("secrets.BOT_KEY") == 4


FAKE_GH = """#!/usr/bin/env python3
import json, os, re, subprocess, sys
args = sys.argv[1:]
with open(os.environ["FAKE_GH_LOG"], "a") as log:
    log.write(json.dumps(args) + "\\n")
repo = "repos/owner/repo/"
sha = re.compile(r"^[0-9a-f]{40}$")
fields = "number,headRefName,headRefOid,mergeStateStatus,isDraft,isCrossRepository,files,statusCheckRollup"
if args == ["pr", "list", "--base", "dev", "--state", "open", "--author", "app/dependabot",
            "--limit", "50", "--json", fields]:
    sys.stdout.write(open(os.environ["FAKE_PRS"]).read())
    sys.exit(0)
if args == ["api", repo + "commits/dev", "--jq", ".commit.committer.date"]:
    print(os.environ["FAKE_DEV_MOVED_AT"])
    sys.exit(0)
if len(args) == 4 and args[0] == "api" and args[2] == "--jq" and args[1].startswith(repo + "commits/") \\
        and args[1].endswith("/statuses?per_page=100"):
    head = args[1][len(repo + "commits/"):-len("/statuses?per_page=100")]
    statuses = json.load(open(os.environ["FAKE_STATUSES"])).get(head, [])
    out = subprocess.run(["jq", "-c", args[3]], input=json.dumps(statuses),
                         capture_output=True, text=True, check=True)
    sys.stdout.write(out.stdout)
    sys.exit(0)
if len(args) == 10 and args[:2] == ["api", repo + "statuses/" + os.environ.get("HEAD_SHA", "")]:
    sys.exit(0)
if len(args) == 6 and args[:2] == ["pr", "merge"] and args[2].isdigit() \\
        and args[3:5] == ["--squash", "--match-head-commit"] and sha.match(args[5]):
    sys.exit(0)
if len(args) == 6 and args[:3] == ["api", "-X", "PUT"] \\
        and re.fullmatch(r"repos/owner/repo/pulls/[0-9]+/update-branch", args[3]) \\
        and args[4] == "-f" and args[5].startswith("expected_head_sha=") and sha.match(args[5][18:]):
    sys.exit(0)
sys.exit(2)
"""


def _write_fake_gh(bin_dir: Path) -> None:
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(0o755)


def _run_step(step: dict[str, Any], cwd: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], list]:
    """Run a step's script as GitHub runs `shell: bash`, with the fake gh first on PATH."""
    log = cwd.parent / "gh.log"
    full_env = {
        "PATH": f"{cwd.parent / 'bin'}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "FAKE_GH_LOG": str(log),
        "GH_REPO": "owner/repo",
        "RUNNER_TEMP": str(cwd.parent),
        "GITHUB_OUTPUT": str(cwd.parent / "github_output"),
        "GITHUB_STEP_SUMMARY": str(cwd.parent / "summary.md"),
        **env,
    }
    proc = subprocess.run(
        [*BASH, step["run"]], cwd=cwd, env=full_env, capture_output=True, text=True, timeout=120, check=False
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return proc, calls


def _outputs(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "github_output"
    return dict(line.split("=", 1) for line in path.read_text().splitlines()) if path.exists() else {}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def _lock(*packages: tuple[str, str]) -> str:
    return "version = 1\n" + "".join(f'\n[[package]]\nname = "{n}"\nversion = "{v}"\n' for n, v in packages)


BASE_LOCK = _lock(("crewai", "1.15.27"), ("json-repair", "0.60.1"), ("mcp", "1.28.1"))
UP_LOCK = _lock(("crewai", "1.15.27"), ("json-repair", "0.60.1"), ("mcp", "1.28.2"))
DOWN_LOCK = _lock(("crewai", "1.6.1"), ("json-repair", "0.25.2"), ("mcp", "1.30.0"))
ALWAYS_QUALIFIES = "print('qualifies')\n"


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _gate_repo(tmp_path: Path, dev_files: dict[str, str] | None = None) -> tuple[Path, str]:
    """A repo whose dev carries the real gate and BASE_LOCK; returns it and the merge base."""
    _write_fake_gh(tmp_path / "bin")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "dev")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "test")
    gate = (REPO_ROOT / "tools" / "dependabot_gate.py").read_text()
    return repo, _commit(repo, {"tools/dependabot_gate.py": gate, "uv.lock": BASE_LOCK, **(dev_files or {})}, "dev")


def _judge(repo: Path, base_sha: str, head_sha: str, update_type: str = MINOR):
    step = _step(_load(GATE_WORKFLOW), "judge", JUDGE_STEP)
    env = {"BASE_SHA": base_sha, "HEAD_SHA": head_sha, "ECOSYSTEM": "uv", "UPDATE_TYPE": update_type, "GROUP": ""}
    proc, calls = _run_step(step, repo, env)
    assert calls == [], "the judge must not write anything"
    return proc, _outputs(repo.parent)


def _branch(repo: Path, merge_base: str, files: dict[str, str]) -> str:
    _git(repo, "checkout", "-q", "-b", "dependabot/uv/x", merge_base)
    return _commit(repo, files, "dependabot")


def test_the_judge_qualifies_a_lock_that_moves_only_up(tmp_path: Path) -> None:
    repo, merge_base = _gate_repo(tmp_path)
    head = _branch(repo, merge_base, {"uv.lock": UP_LOCK})
    proc, outputs = _judge(repo, merge_base, head)
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"state": "success", "description": "qualifies"}


def test_the_judge_lets_a_major_wait_and_stays_green(tmp_path: Path) -> None:
    repo, merge_base = _gate_repo(tmp_path)
    head = _branch(repo, merge_base, {"uv.lock": UP_LOCK})
    proc, outputs = _judge(repo, merge_base, head, update_type="version-update:semver-major")
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"state": "success", "description": "waits for a person: version-update:semver-major update"}


def test_the_judge_refuses_a_lock_downgrade_and_goes_red(tmp_path: Path) -> None:
    repo, merge_base = _gate_repo(tmp_path)
    head = _branch(repo, merge_base, {"uv.lock": DOWN_LOCK})
    proc, outputs = _judge(repo, merge_base, head)
    reason = "uv.lock lowers 2: crewai 1.15.27 -> 1.6.1, json-repair 0.60.1 -> 0.25.2"
    assert proc.returncode == 1
    assert outputs == {"state": "failure", "description": reason}
    assert f"::error::{reason}" in proc.stdout


def test_the_judge_cannot_tell_an_unreadable_lock_and_goes_red(tmp_path: Path) -> None:
    repo, merge_base = _gate_repo(tmp_path)
    head = _branch(repo, merge_base, {"uv.lock": "not = [valid toml"})
    proc, outputs = _judge(repo, merge_base, head)
    assert proc.returncode == 1
    assert outputs["state"] == "failure"
    assert outputs["description"].startswith("cannot tell: unreadable uv.lock")
    assert "::error::cannot tell" in proc.stdout


def test_the_judge_uses_devs_rules_not_the_prs(tmp_path: Path) -> None:
    """A PR that rewrites the gate to always qualify is still judged by dev's
    copy, which sends any change to the gate itself to a person."""
    repo, merge_base = _gate_repo(tmp_path)
    head = _branch(repo, merge_base, {"uv.lock": DOWN_LOCK, "tools/dependabot_gate.py": ALWAYS_QUALIFIES})
    proc, outputs = _judge(repo, merge_base, head)
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"state": "success", "description": "waits for a person: changes tools/dependabot_gate.py"}


def test_the_judge_compares_with_the_merge_base_not_devs_tip(tmp_path: Path) -> None:
    """dev moved on after the PR branched (a later crewai): judged against
    dev's tip, the PR's older crewai would look like a downgrade."""
    repo, merge_base = _gate_repo(tmp_path)
    head = _branch(repo, merge_base, {"uv.lock": UP_LOCK})
    _git(repo, "checkout", "-q", "dev")
    newer = _lock(("crewai", "1.16.0"), ("json-repair", "0.60.1"), ("mcp", "1.28.1"))
    dev_tip = _commit(repo, {"uv.lock": newer}, "dev")
    _git(repo, "checkout", "-q", head)
    proc, outputs = _judge(repo, dev_tip, head)
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"state": "success", "description": "qualifies"}


def test_the_judge_sees_a_rename_into_an_allowed_file(tmp_path: Path) -> None:
    """With rename detection, `git mv tools/x requirements-diagnose.txt` lists
    only the allowed name; the gate must also see the file that left."""
    payload = "".join(f"line {n}\n" for n in range(40))
    repo, merge_base = _gate_repo(tmp_path, {"tools/payload.txt": payload})
    _git(repo, "checkout", "-q", "-b", "dependabot/uv/x", merge_base)
    _git(repo, "mv", "tools/payload.txt", "requirements-diagnose.txt")
    head = _commit(repo, {}, "dependabot")
    proc, outputs = _judge(repo, merge_base, head)
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"state": "success", "description": "waits for a person: changes tools/payload.txt"}


def test_the_judge_writes_a_reason_with_a_line_break_as_one_line(tmp_path: Path) -> None:
    """A file name may carry a line break; written as is, the rest of the name
    would become an output line of its own, here a state that overrides the
    judge's."""
    repo, merge_base = _gate_repo(tmp_path)
    head = _branch(repo, merge_base, {"uv.lock": UP_LOCK, "x\nstate=failure": "payload\n"})
    proc, outputs = _judge(repo, merge_base, head)
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"state": "success", "description": "waits for a person: changes x state=failure"}


def _record(
    tmp_path: Path, state: str, description: str, judge_result: str = "success"
) -> tuple[subprocess.CompletedProcess[str], list]:
    _write_fake_gh(tmp_path / "bin")
    work = tmp_path / "work"
    work.mkdir()
    step = _step(_load(GATE_WORKFLOW), "record", RECORD_STEP)
    env = {"HEAD_SHA": SHA_C, "STATE": state, "DESCRIPTION": description, "RUN_URL": "https://example.invalid/run/1"}
    return _run_step(step, work, {**env, "JUDGE_RESULT": judge_result})


def _status(state: str, description: str) -> list[str]:
    return [
        "api",
        f"repos/owner/repo/statuses/{SHA_C}",
        "-f",
        f"state={state}",
        "-f",
        "context=dependabot-gate",
        "-f",
        f"description={description}",
        "-f",
        "target_url=https://example.invalid/run/1",
    ]


@pytest.mark.parametrize("state", ["success", "failure"])
def test_the_record_step_writes_the_judges_verdict_to_this_prs_head(tmp_path: Path, state: str) -> None:
    proc, calls = _record(tmp_path, state, "qualifies")
    assert proc.returncode == 0, proc.stderr
    assert calls == [_status(state, "qualifies")]


@pytest.mark.parametrize("state", ["", "qualifies", "pending"])
def test_the_record_step_records_a_failure_when_the_judge_did_not_decide(tmp_path: Path, state: str) -> None:
    """A judge that died, was cancelled or wrote garbage never becomes a pass."""
    proc, calls = _record(tmp_path, state, "qualifies")
    assert proc.returncode == 0, proc.stderr
    assert calls == [_status("failure", "the gate could not run; see the run")]


@pytest.mark.parametrize("judge_result", ["failure", "cancelled", "skipped", ""])
def test_the_record_step_never_passes_a_judge_job_that_did_not_succeed(tmp_path: Path, judge_result: str) -> None:
    """A "success" that reached the record job from a red judge is forged or
    garbled; only a judge that succeeded may pass a head."""
    proc, calls = _record(tmp_path, "success", "qualifies", judge_result)
    assert proc.returncode == 0, proc.stderr
    assert calls == [_status("failure", "the gate could not run; see the run")]


def test_the_record_step_fits_the_description_to_githubs_limit(tmp_path: Path) -> None:
    proc, calls = _record(tmp_path, "failure", "x" * 300)
    assert proc.returncode == 0, proc.stderr
    assert calls == [_status("failure", "x" * 140)]


def _qualified(updated_at: str) -> list[dict]:
    """The status list, newest first: the gate's verdict above an older one."""
    gate = {"context": "dependabot-gate", "state": "success", "description": "qualifies"}
    older = {**gate, "state": "failure", "description": "uv.lock lowers 1: x 2 -> 1"}
    bot = {"login": "github-actions[bot]"}
    return [{**gate, "updated_at": updated_at, "creator": bot}, {**older, "updated_at": "2019-01-01T00:00:00Z"}]


def _plan(tmp_path: Path, prs: list[dict], statuses: dict, dev_moved_at: str):
    assert shutil.which("jq"), "the plan step needs jq, as GitHub's runner image has"
    _write_fake_gh(tmp_path / "bin")
    work = tmp_path / "work"
    shutil.copytree(REPO_ROOT / "tools", work / "tools", ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "fake_prs.json").write_text(json.dumps(prs))
    (tmp_path / "fake_statuses.json").write_text(json.dumps(statuses))
    step = _step(_load(MERGE_WORKFLOW), "plan", PLAN_STEP)
    env = {
        "FAKE_PRS": str(tmp_path / "fake_prs.json"),
        "FAKE_STATUSES": str(tmp_path / "fake_statuses.json"),
        "FAKE_DEV_MOVED_AT": dev_moved_at,
    }
    proc, calls = _run_step(step, work, env)
    return proc, calls, _outputs(tmp_path)


def _uv_pr(number: int, head: str, state: str = "CLEAN") -> dict:
    passed = [{"__typename": "CheckRun", "status": "COMPLETED", "conclusion": "SUCCESS"}]
    return {
        "number": number,
        "headRefName": f"dependabot/uv/p{number}-1.1",
        "headRefOid": head,
        "mergeStateStatus": state,
        "isDraft": False,
        "isCrossRepository": False,
        "files": [{"path": "uv.lock"}],
        "statusCheckRollup": passed,
    }


def test_the_plan_step_reads_the_prs_and_gate_statuses_and_picks_a_merge(tmp_path: Path) -> None:
    prs = [_uv_pr(7, SHA_A), _uv_pr(8, SHA_B)]
    statuses = {SHA_A: [{"context": "ci", "state": "success"}], SHA_B: _qualified("2026-10-14T11:00:00Z")}
    proc, calls, outputs = _plan(tmp_path, prs, statuses, "2026-10-14T11:00:00Z")
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"action": "merge", "number": "8", "head": SHA_B}
    assert [call[:2] for call in calls] == [
        ["pr", "list"],
        ["api", f"repos/owner/repo/commits/{SHA_A}/statuses?per_page=100"],
        ["api", f"repos/owner/repo/commits/{SHA_B}/statuses?per_page=100"],
        ["api", "repos/owner/repo/commits/dev"],
    ]
    assert "#7: awaiting the gate's verdict (CLEAN)" in proc.stdout


def test_the_plan_step_picks_an_update_for_a_long_behind_pr(tmp_path: Path) -> None:
    prs = [_uv_pr(9, SHA_C, "BEHIND")]
    proc, _calls, outputs = _plan(tmp_path, prs, {SHA_C: _qualified("2020-01-01T00:00:00Z")}, "2020-01-01T00:00:00Z")
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"action": "update", "number": "9", "head": SHA_C}


def test_the_plan_step_waits_when_dev_just_moved(tmp_path: Path) -> None:
    """An old verdict on a PR that fell behind a moment ago still gives
    Dependabot its chance: the step must hand the planner dev's real tip time."""
    just_now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    prs = [_uv_pr(9, SHA_C, "BEHIND")]
    proc, _calls, outputs = _plan(tmp_path, prs, {SHA_C: _qualified("2020-01-01T00:00:00Z")}, just_now)
    assert proc.returncode == 0, proc.stderr
    assert outputs == {"action": "none", "number": "", "head": ""}


def test_the_plan_step_fails_when_the_planner_fails(tmp_path: Path) -> None:
    """Under `bash -e` alone the planner's crash would vanish into the pipe to tee."""
    no_time = [
        {
            "context": "dependabot-gate",
            "state": "success",
            "description": "qualifies",
            "creator": {"login": "github-actions[bot]"},
        }
    ]
    proc, _calls, outputs = _plan(tmp_path, [_uv_pr(9, SHA_C)], {SHA_C: no_time}, "2026-10-14T11:00:00Z")
    assert proc.returncode != 0
    assert "action" not in outputs


def _act(tmp_path: Path, env: dict[str, str]) -> tuple[subprocess.CompletedProcess[str], list]:
    _write_fake_gh(tmp_path / "bin")
    work = tmp_path / "work"
    work.mkdir()
    return _run_step(_step(_load(MERGE_WORKFLOW), "act", ACT_STEP), work, env)


@pytest.mark.parametrize(
    ("action", "call"),
    [
        ("merge", ["pr", "merge", "12", "--squash", "--match-head-commit", SHA_C]),
        ("update", ["api", "-X", "PUT", "repos/owner/repo/pulls/12/update-branch", "-f", f"expected_head_sha={SHA_C}"]),
    ],
)
def test_the_act_step_merges_or_updates_pinned_to_the_judged_head(tmp_path: Path, action: str, call: list[str]) -> None:
    proc, calls = _act(tmp_path, {"ACTION": action, "PR": "12", "HEAD_SHA": SHA_C})
    assert proc.returncode == 0, proc.stderr
    assert calls == [call]


@pytest.mark.parametrize(
    ("action", "pr", "head"),
    [
        ("close", "12", SHA_C),
        ("merge", "12", ""),
        ("merge", "", SHA_C),
        ("update", "12; true", SHA_C),
        ("merge", "12", "HEAD"),
    ],
)
def test_the_act_step_refuses_anything_but_a_numbered_pr_at_a_full_sha(tmp_path, action, pr, head) -> None:
    proc, calls = _act(tmp_path, {"ACTION": action, "PR": pr, "HEAD_SHA": head})
    assert proc.returncode == 1
    assert calls == []
