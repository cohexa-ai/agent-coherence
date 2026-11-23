# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""The six arms both basis demos run, each against its own coordinator.

A reader reads two inputs and records what it read. Before the act runs, a
peer edits both inputs (every arm but the second reader's). The act lands a
result computed from the recorded bytes, the fixed stale buffer, never from a
fresh read. It runs from another volume, as a later step in another process
would: that process holds only what the reader recorded, so the stale-read
deny a volume keeps for its own reads cannot help it.

Each arm makes one claim per run, measured on what reached disk and what the
coordinator answered, never assumed from the code path taken.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ccs.adapters.claude_code.lifecycle import (  # noqa: E402  (after sys.path setup)
    LifecycleConfig,
    stop_coordinator,
)
from ccs.adapters.coherent_volume import CasCommitResult, CoherentVolume  # noqa: E402
from ccs.adapters.effect_gate import check_fence, gate  # noqa: E402
from ccs.core.basis import ANSWER_HOLD, ANSWER_PROCEED, BasisMember  # noqa: E402
from ccs.core.exceptions import StaleView  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    """One demo: the files under ``area``, and the act that lands a result.

    ``target`` is the file the act writes; it may be one of the two inputs.
    ``act`` turns the bytes the reader recorded (each input, and the target)
    into the content the act lands.
    """

    area: str
    inputs: tuple[str, str]
    target: str
    seed: Mapping[str, bytes]
    peer_edit: Mapping[str, bytes]
    act: Callable[[Mapping[str, bytes]], bytes]


@dataclass(frozen=True)
class Workspace:
    """One coordinator's workspace and the sessions an arm uses."""

    root: Path
    owner: CoherentVolume  # creates and commits every file before anyone reads
    reader: CoherentVolume  # reads the inputs and records what it read
    second: CoherentVolume  # a second reader of the inputs
    peer: CoherentVolume  # edits both inputs after the reader recorded them
    actor: CoherentVolume  # runs the act later, holding only what was recorded
    auditor: CoherentVolume  # reads the target's version to report on it


@dataclass(frozen=True)
class Recorded:
    """What the reader recorded: the bytes, the basis and the target's version."""

    buffers: Mapping[str, bytes]
    basis: tuple[BasisMember, ...]
    target_version: int


@dataclass(frozen=True)
class Outcome:
    behaved: bool
    detail: str


@dataclass(frozen=True)
class Arm:
    label: str
    claim: str
    run: Callable[[Workspace, Scenario, int], Outcome]


def _fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def _open_workspace(root: Path, scenario: Scenario) -> Workspace:
    def volume() -> CoherentVolume:
        return CoherentVolume(root, managed=(f"{scenario.area}/**",), config=_fast_cfg())

    return Workspace(root, volume(), volume(), volume(), volume(), volume(), volume())


def _seed(ws: Workspace, scenario: Scenario, run: int) -> dict[str, str]:
    """Create and commit every file of this run, so each has a recorded
    content and a registration before anyone reads it."""
    files = {name: f"{scenario.area}/run-{run}/{name}" for name in scenario.seed}
    (ws.root / scenario.area / f"run-{run}").mkdir(parents=True)
    for name, path in files.items():
        ws.owner.write(path, scenario.seed[name])
        ws.owner.release(path)  # a write grant left held would refuse the peer's edit
    return files


def _target_read(ws: Workspace, scenario: Scenario, files: Mapping[str, str]) -> tuple[bytes, int]:
    """The reader's read of a target that is not one of the inputs."""
    return ws.reader.read_with_version(files[scenario.target])


def _record(ws: Workspace, scenario: Scenario, files: Mapping[str, str]) -> Recorded:
    buffers: dict[str, bytes] = {}
    basis: list[BasisMember] = []
    for name in scenario.inputs:
        buffers[name], member = ws.reader.read_for_basis(files[name])
        basis.append(member)
    if scenario.target in scenario.inputs:
        return Recorded(buffers, tuple(basis), basis[scenario.inputs.index(scenario.target)].version)
    buffers[scenario.target], target_version = _target_read(ws, scenario, files)
    return Recorded(buffers, tuple(basis), target_version)


def _peer_edits_inputs(ws: Workspace, scenario: Scenario, files: Mapping[str, str]) -> dict[str, int]:
    """The peer edits both inputs; returns the version each moved to."""
    moved_to: dict[str, int] = {}
    for name in scenario.inputs:
        _, version = ws.peer.read_with_version(files[name])
        committed = ws.peer.write_cas_at(files[name], version, scenario.peer_edit[name])
        moved_to[name] = committed.version
    return moved_to


def _on_disk(ws: Workspace, path: str) -> bytes:
    return (ws.root / path).read_bytes()


def _drift(names: Iterable[str], recorded: Mapping[str, int], now: Mapping[str, int]) -> str:
    return ", ".join(f"{name} v{recorded[name]}->v{now[name]}" for name in names)


def _recorded_versions(scenario: Scenario, basis: Iterable[BasisMember]) -> dict[str, int]:
    return {name: member.version for name, member in zip(scenario.inputs, basis)}


def unguarded(ws: Workspace, scenario: Scenario, run: int) -> Outcome:
    """The act runs on the recorded bytes with no check at all."""
    files = _seed(ws, scenario, run)
    rec = _record(ws, scenario, files)
    now = _peer_edits_inputs(ws, scenario, files)
    result = scenario.act(rec.buffers)
    ws.actor.write(files[scenario.target], result)
    recorded = _recorded_versions(scenario, rec.basis)
    moved = [name for name in scenario.inputs if now[name] != recorded[name]]
    behaved = _on_disk(ws, files[scenario.target]) == result and bool(moved)
    return Outcome(behaved, f"wrote {scenario.target} from {_drift(scenario.inputs, recorded, now)}")


def gated(ws: Workspace, scenario: Scenario, run: int) -> Outcome:
    """``gate()`` around the act, whose decision was made before the inputs
    moved: the gate captures the input's version at its own call, after the
    move, so it has nothing earlier to compare against."""
    files = _seed(ws, scenario, run)
    rec = _record(ws, scenario, files)
    now = _peer_edits_inputs(ws, scenario, files)
    decided = scenario.act(rec.buffers)
    name = scenario.inputs[0]
    try:
        gate(
            ws.actor,
            files[name],
            decide=lambda _current_bytes: decided,
            effect=lambda result: ws.actor.write(files[scenario.target], result),
        )
    except StaleView:
        return Outcome(False, f"gate() held on {name}")
    recorded = _recorded_versions(scenario, rec.basis)
    behaved = _on_disk(ws, files[scenario.target]) == decided and now[name] != recorded[name]
    return Outcome(behaved, f"fired a decision made at {name} v{recorded[name]} with {name} at v{now[name]}")


def fence_loop(ws: Workspace, scenario: Scenario, run: int) -> Outcome:
    """``check_fence`` once per input, recorded through
    ``read_with_version_generation``: the first hold ends the loop."""
    files = _seed(ws, scenario, run)
    buffers: dict[str, bytes] = {}
    comparands: dict[str, tuple[int, int | None]] = {}
    for name in scenario.inputs:
        buffers[name], version, generation = ws.reader.read_with_version_generation(files[name])
        comparands[name] = (version, generation)
    if scenario.target not in scenario.inputs:
        buffers[scenario.target], _ = _target_read(ws, scenario, files)
    now = _peer_edits_inputs(ws, scenario, files)
    try:
        for name, (version, generation) in comparands.items():
            check_fence(ws.reader, files[name], expected_version=version, expected_generation=generation)
    except StaleView as held:
        named = [name for name in scenario.inputs if files[name] in str(held)]
    else:
        ws.actor.write(files[scenario.target], scenario.act(buffers))
        return Outcome(False, "every check_fence passed; the act ran")
    moved = {name for name in scenario.inputs if now[name] != comparands[name][0]}
    behaved = len(named) == 1 and named[0] in moved and len(moved) == 2
    return Outcome(behaved, f"held naming {', '.join(named)}; {len(moved)} inputs had moved")


def basis_check(ws: Workspace, scenario: Scenario, run: int) -> Outcome:
    """``check_basis`` on the recorded basis; the act runs only on proceed."""
    files = _seed(ws, scenario, run)
    rec = _record(ws, scenario, files)
    _peer_edits_inputs(ws, scenario, files)
    answer = ws.actor.check_basis(rec.basis)
    if answer.answer == ANSWER_PROCEED:
        ws.actor.write(files[scenario.target], scenario.act(rec.buffers))
        return Outcome(False, "check_basis proceeded; the act ran")
    named = {member.path for member in answer.moved}
    behaved = answer.answer == ANSWER_HOLD and named == {files[name] for name in scenario.inputs}
    moved = ", ".join(f"{Path(m.path).name} v{m.expected}->v{m.current} ({m.reason})" for m in answer.moved)
    return Outcome(behaved, f"held naming {moved}; recovery {answer.recovery}")


def basis_commit(ws: Workspace, scenario: Scenario, run: int) -> Outcome:
    """``commit_if_basis`` lands the act's result only while the basis holds."""
    files = _seed(ws, scenario, run)
    rec = _record(ws, scenario, files)
    _peer_edits_inputs(ws, scenario, files)
    target = files[scenario.target]
    before = (_on_disk(ws, target), ws.auditor.read_with_version(target)[1])
    outcome = ws.actor.commit_if_basis(rec.basis, target, rec.target_version, scenario.act(rec.buffers))
    after = (_on_disk(ws, target), ws.auditor.read_with_version(target)[1])
    if isinstance(outcome, CasCommitResult):
        return Outcome(False, f"committed {scenario.target} at v{outcome.version}")
    behaved = outcome.answer == ANSWER_HOLD and after == before
    detail = f"held with {len(outcome.moved)} moved; {scenario.target} still v{after[1]}, bytes unchanged"
    return Outcome(behaved, detail)


def second_reader(ws: Workspace, scenario: Scenario, run: int) -> Outcome:
    """Nothing moves. The first reader's commit rewrites only its target, so
    a second reader whose basis leaves that target out still proceeds."""
    files = _seed(ws, scenario, run)
    rec = _record(ws, scenario, files)
    theirs = tuple(ws.second.read_for_basis(files[name])[1] for name in scenario.inputs if name != scenario.target)
    outcome = ws.actor.commit_if_basis(rec.basis, files[scenario.target], rec.target_version, scenario.act(rec.buffers))
    answer = ws.second.check_basis(theirs)
    if not isinstance(outcome, CasCommitResult):
        return Outcome(False, f"the first reader's commit held: {outcome}")
    behaved = answer.answer == ANSWER_PROCEED
    names = ", ".join(Path(member.path).name for member in theirs)
    detail = f"{scenario.target} committed at v{outcome.version}; basis {{{names}}} answered {answer.answer}"
    return Outcome(behaved, detail)


ARMS: tuple[Arm, ...] = (
    Arm("unguarded act", "used a moved input", unguarded),
    Arm("gate()", "proceeded on a moved input", gated),
    Arm("check_fence loop", "held, naming one of the two moved inputs", fence_loop),
    Arm("check_basis", "held, naming both moved inputs", basis_check),
    Arm("commit_if_basis", "held, the target unchanged on disk and in version", basis_commit),
    Arm("second reader", "proceeded after the first reader's commit", second_reader),
)


def _run_count(text: str) -> int:
    """A ``--runs`` value: a whole number of at least 1, refused before any
    coordinator starts, since a report with no runs has nothing to show."""
    try:
        runs = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
    if runs < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {runs}")
    return runs


def parse_runs(argv: list[str] | None, description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--runs", type=_run_count, default=5, help="runs per arm, at least 1 (default 5)")
    return parser.parse_args(argv).runs


def _run_arm(arm: Arm, scenario: Scenario, runs: int) -> list[Outcome]:
    """Every run of one arm, on one coordinator in a fresh workspace, with
    fresh paths per run; the coordinator stops whatever happens."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        try:
            ws = _open_workspace(root, scenario)
            return [arm.run(ws, scenario, run) for run in range(1, runs + 1)]
        finally:
            stop_coordinator(root)


def run_arms(scenario: Scenario, runs: int) -> bool:
    """Run every arm, print its count and the contract; True only when every
    arm behaved as claimed in every run."""
    print(f"Counts over {runs} run(s) per arm (each arm on its own coordinator):")
    held: list[tuple[Arm, bool]] = []
    for arm in ARMS:
        outcomes = _run_arm(arm, scenario, runs)
        count = sum(outcome.behaved for outcome in outcomes)
        print(f"  {arm.label:<17} {count}/{runs} {arm.claim}")
        print(f"  {'':<17} run 1: {outcomes[0].detail}")
        held.append((arm, count == runs))
    print("\nContract:")
    for arm, ok in held:
        print(f"  {arm.label + ' ' + arm.claim:<66}: {ok}")
    return all(ok for _, ok in held)
