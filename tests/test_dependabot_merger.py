# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Tests for tools/dependabot_merger.py — which Dependabot PR the merger acts on next.

The merger merges with the forward-merge App's token, so what it picks lands
on dev with no person in the loop. It must pick only a PR the gate qualified
on its current head, and must re-check the branch and files itself, because a
commit status can be written by any workflow with ``statuses: write``. Each
merge leaves the other PRs behind dev, so it takes one step per run, gives
Dependabot time to refresh a PR before merging dev into it, and never updates
a branch while another PR's checks or verdict are still pending.
``tools/`` is not an importable package, so the module is loaded by file path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
FIXTURE = "examples/refactor_demo/fixture_repo_ts"
QUALIFIED = {
    "state": "success",
    "description": "qualifies",
    "updated_at": "2026-10-14T11:00:00Z",
    "creator": {"login": "github-actions[bot]"},
}
PASSED = [{"__typename": "CheckRun", "name": "Lint", "status": "COMPLETED", "conclusion": "SUCCESS"}]
# Literal on purpose (tests/CLAUDE.md): derived from the module, a state dropped
# from the module would drop out of the test with it.
FAILING = ["FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED", "STARTUP_FAILURE", "STALE"]


def _import_merger():
    tools = str(_REPO_ROOT / "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location("dependabot_merger", _REPO_ROOT / "tools" / "dependabot_merger.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses resolves a class's module through sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def merger():
    return _import_merger()


def _pr(number: int, **overrides) -> dict:
    """An open Dependabot uv PR, up to date with every check passing."""
    pr = {
        "number": number,
        "headRefName": f"dependabot/uv/pkg-{number}",
        "headRefOid": f"{number:040x}",
        "mergeStateStatus": "CLEAN",
        "isDraft": False,
        "isCrossRepository": False,
        "files": [{"path": "uv.lock"}],
        "statusCheckRollup": PASSED,
    }
    return {**pr, **overrides}


@pytest.mark.parametrize(
    ("overrides", "gate", "reason"),
    [
        ({"isDraft": True}, QUALIFIED, "draft or fork"),
        ({"isCrossRepository": True}, QUALIFIED, "draft or fork"),
        (
            {"headRefName": "dependabot/github_actions/actions/checkout-7.0.0"},
            QUALIFIED,
            "not a Python lock or demo fixture update",
        ),
        ({"headRefName": "feature/uv.lock"}, QUALIFIED, "not a Python lock or demo fixture update"),
        (
            {"headRefName": "dependabot/npm_and_yarn/other/typescript-7.1.0"},
            QUALIFIED,
            "not a Python lock or demo fixture update",
        ),
        ({"files": []}, QUALIFIED, "no changed files"),
        (
            {"files": [{"path": "uv.lock"}, {"path": ".github/workflows/ci.yml"}]},
            QUALIFIED,
            "changes .github/workflows/ci.yml",
        ),
        (
            {"files": [{"path": f"{FIXTURE}/package-lock.json"}]},
            QUALIFIED,
            f"changes {FIXTURE}/package-lock.json",
        ),
        (
            {"headRefName": f"dependabot/npm_and_yarn/{FIXTURE}/typescript-7.0.3", "files": [{"path": "uv.lock"}]},
            QUALIFIED,
            "changes uv.lock",
        ),
        ({}, None, "awaiting the gate's verdict"),
        ({}, {**QUALIFIED, "state": "failure"}, "the gate has not qualified this head"),
        ({}, {**QUALIFIED, "creator": {"login": "someone"}}, "the gate has not qualified this head"),
        ({}, {k: v for k, v in QUALIFIED.items() if k != "creator"}, "the gate has not qualified this head"),
        (
            {},
            {**QUALIFIED, "description": "waits for a person: version-update:semver-major update"},
            "the gate has not qualified this head",
        ),
    ],
)
def test_ineligibility_rechecks_what_the_gate_judged(merger, overrides, gate, reason) -> None:
    """A forged "qualifies" status on an Actions bump, or on files of the other
    ecosystem, still merges nothing."""
    assert merger.ineligibility(_pr(1, **overrides), gate) == reason


@pytest.mark.parametrize(
    ("branch", "files"),
    [
        ("dependabot/uv/mcp-1.28.2", ["uv.lock", "requirements-diagnose.txt"]),
        (f"dependabot/npm_and_yarn/{FIXTURE}/typescript-7.0.3", [f"{FIXTURE}/package-lock.json"]),
    ],
)
def test_ineligibility_passes_a_qualified_lock_or_fixture_update(merger, branch, files) -> None:
    pr = _pr(1, headRefName=branch, files=[{"path": path} for path in files])
    assert merger.ineligibility(pr, QUALIFIED) is None


def _run(status: str, conclusion: str = "") -> dict:
    return {"__typename": "CheckRun", "status": status, "conclusion": conclusion}


@pytest.mark.parametrize(
    ("rollup", "summary"),
    [
        ([], "passing"),
        (PASSED + [_run("COMPLETED", "SKIPPED"), _run("COMPLETED", "NEUTRAL")], "passing"),
        (PASSED + [{"__typename": "StatusContext", "state": "SUCCESS"}], "passing"),
        (PASSED + [_run("IN_PROGRESS")], "pending"),
        (PASSED + [{"__typename": "StatusContext", "state": "PENDING"}], "pending"),
        (PASSED + [_run("COMPLETED", "")], "pending"),
        (PASSED + [_run("COMPLETED", "SOMETHING_NEW")], "pending"),
        ([_run("QUEUED"), _run("COMPLETED", "TIMED_OUT")], "failing"),
    ],
)
def test_check_summary_passes_only_known_passing_states(merger, rollup, summary) -> None:
    """A state the merger does not know never counts as passed."""
    assert merger.check_summary(rollup) == summary


@pytest.mark.parametrize("state", FAILING)
def test_check_summary_fails_on_every_failing_check_state(merger, state) -> None:
    assert merger.check_summary(PASSED + [_run("COMPLETED", state)]) == "failing"
    assert merger.check_summary(PASSED + [{"__typename": "StatusContext", "state": state}]) == "failing"


def _candidate(merger, number: int, state: str = "CLEAN", checks: str = "passing", quiet: int = 30):
    return merger.Candidate(number, f"{number:040x}", state, checks, timedelta(minutes=quiet))


def test_decide_merges_the_oldest_up_to_date_pr(merger) -> None:
    step = merger.decide([_candidate(merger, 9), _candidate(merger, 4), _candidate(merger, 7, "BEHIND")])
    assert step == merger.Step("merge", 4, f"{4:040x}")


def test_decide_never_merges_a_pr_whose_checks_are_not_all_passing(merger) -> None:
    for checks in ("pending", "failing"):
        assert merger.decide([_candidate(merger, 4, checks=checks)]).action == "none"


@pytest.mark.parametrize("state", ["BLOCKED", "DIRTY", "UNKNOWN", "UNSTABLE", "HAS_HOOKS"])
def test_decide_merges_only_a_clean_pr(merger, state) -> None:
    assert merger.decide([_candidate(merger, 4, state)]).action == "none"


def test_decide_updates_the_oldest_behind_pr_once_dependabot_had_its_chance(merger) -> None:
    step = merger.decide([_candidate(merger, 9, "BEHIND"), _candidate(merger, 5, "BEHIND")])
    assert step == merger.Step("update", 5, f"{5:040x}")


def test_decide_leaves_a_recently_moved_behind_pr_to_dependabot(merger) -> None:
    """Dependabot refreshes a PR within minutes after a sibling merges; an
    update here would stop it ever rebasing that PR again."""
    assert merger.decide([_candidate(merger, 5, "BEHIND", quiet=9)]).action == "none"
    assert merger.decide([_candidate(merger, 5, "BEHIND", quiet=10)]).action == "update"


def test_decide_updates_nothing_while_another_prs_checks_are_running(merger) -> None:
    """Updating now would leave the running PR behind again the moment either merges."""
    step = merger.decide([_candidate(merger, 5, "BEHIND"), _candidate(merger, 6, "BLOCKED", checks="pending")])
    assert step.action == "none"


def test_decide_never_updates_a_behind_pr_with_a_failed_check(merger) -> None:
    assert merger.decide([_candidate(merger, 5, "BEHIND", checks="failing")]).action == "none"


def _main(merger, tmp_path: Path, prs: list[dict], gates: list[dict], dev_moved: datetime) -> list[str]:
    (tmp_path / "prs.json").write_text(json.dumps(prs))
    (tmp_path / "gates.json").write_text(json.dumps(gates))
    output = tmp_path / "github_output"
    output.write_text("")
    argv = ["--prs", str(tmp_path / "prs.json"), "--gates", str(tmp_path / "gates.json"), "--now", NOW.isoformat()]
    assert merger.main([*argv, "--dev-moved-at", dev_moved.isoformat(), "--github-output", str(output)]) == 0
    return output.read_text().splitlines()


def _gate_at(minutes_ago: int) -> dict:
    return {**QUALIFIED, "updated_at": (NOW - timedelta(minutes=minutes_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")}


def test_main_writes_the_step_and_says_why_each_pr_was_or_was_not_picked(merger, tmp_path, capsys) -> None:
    prs = [
        _pr(6),
        _pr(4, headRefName="dependabot/github_actions/actions/setup-java-6.0.1"),
        _pr(5, mergeStateStatus="BEHIND"),
    ]
    gates = [{"sha": pr["headRefOid"], "gate": QUALIFIED} for pr in prs]
    assert _main(merger, tmp_path, prs, gates, NOW - timedelta(hours=1)) == [
        "action=merge",
        "number=6",
        f"head={6:040x}",
    ]
    assert capsys.readouterr().out.splitlines() == [
        "#4: not a Python lock or demo fixture update (CLEAN)",
        "#5: eligible (BEHIND)",
        "#6: eligible (CLEAN)",
        "next step: merge #6",
    ]


def test_main_writes_none_when_nothing_is_eligible(merger, tmp_path) -> None:
    gates = [{"sha": f"{4:040x}", "gate": {**QUALIFIED, "state": "failure"}}]
    assert _main(merger, tmp_path, [_pr(4)], gates, NOW) == ["action=none", "number=", "head="]


@pytest.mark.parametrize(
    ("verdict_minutes_ago", "dev_moved_minutes_ago", "action"),
    [(11, 11, "update"), (9, 60, "none"), (60, 9, "none"), (60, 10, "update")],
)
def test_main_updates_a_behind_pr_only_once_verdict_and_dev_are_both_quiet(
    merger, tmp_path, verdict_minutes_ago, dev_moved_minutes_ago, action
) -> None:
    """The grace runs from the later of the verdict and dev's last move: an old
    verdict on a PR that fell behind a minute ago still gives Dependabot time."""
    prs = [_pr(5, mergeStateStatus="BEHIND")]
    gates = [{"sha": f"{5:040x}", "gate": _gate_at(verdict_minutes_ago)}]
    lines = _main(merger, tmp_path, prs, gates, NOW - timedelta(minutes=dev_moved_minutes_ago))
    assert lines[0] == f"action={action}"
    if action == "update":
        assert lines[1:] == ["number=5", f"head={5:040x}"]


def test_main_updates_nothing_while_a_dependabot_head_awaits_its_verdict(merger, tmp_path, capsys) -> None:
    """A refreshed head may qualify any moment; updating another PR now would
    leave it behind before its checks finish."""
    prs = [_pr(5, mergeStateStatus="BEHIND"), _pr(6, mergeStateStatus="BLOCKED")]
    gates = [{"sha": f"{5:040x}", "gate": _gate_at(60)}, {"sha": f"{6:040x}", "gate": None}]
    assert _main(merger, tmp_path, prs, gates, NOW - timedelta(hours=1))[0] == "action=none"
    assert "#6: awaiting the gate's verdict (BLOCKED)" in capsys.readouterr().out


def test_main_warns_about_a_qualified_pr_only_a_person_can_move(merger, tmp_path, capsys) -> None:
    failed = [_run("COMPLETED", "FAILURE")]
    prs = [_pr(5, statusCheckRollup=failed), _pr(6, mergeStateStatus="DIRTY")]
    gates = [{"sha": pr["headRefOid"], "gate": QUALIFIED} for pr in prs]
    assert _main(merger, tmp_path, prs, gates, NOW - timedelta(hours=1))[0] == "action=none"
    out = capsys.readouterr().out
    assert "::warning::#5 qualifies but has a failed check; re-run it, or comment @dependabot recreate" in out
    assert "::warning::#6 qualifies but conflicts with dev; comment @dependabot recreate" in out
