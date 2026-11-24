# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents
"""Smoke tests for the basis demos (examples/basis_gate).

Each script runs every arm against real coordinators and exits 0 only when
every arm behaved as its contract states, so the demos cannot silently rot.
The forced failure swaps the volume's basis check for one that always
proceeds: the check arm then acts on moved inputs, and the script must say so
by exiting 1 rather than print a contract it no longer meets.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable

import pytest

from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.basis import BasisCheckAnswer, BasisMember, CheckedMember

_SCRIPTS = pytest.mark.parametrize(
    "script", ["examples.basis_gate.stale_check", "examples.basis_gate.stale_approval"]
)


def _always_proceeds(self: CoherentVolume, basis: Iterable[BasisMember]) -> BasisCheckAnswer:
    """A basis check that answers proceed without asking the coordinator."""
    return BasisCheckAnswer(
        answer="proceed",
        checked=tuple(CheckedMember(m.path, m.version, m.version) for m in basis),
    )


@_SCRIPTS
def test_basis_gate_demo_exits_zero(script: str) -> None:
    main = importlib.import_module(script).main

    assert main(["--runs", "1"]) == 0


@_SCRIPTS
def test_basis_gate_demo_exits_one_when_the_check_proceeds_on_moved_inputs(
    script: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    main = importlib.import_module(script).main
    monkeypatch.setattr(CoherentVolume, "check_basis", _always_proceeds)

    assert main(["--runs", "1"]) == 1


@_SCRIPTS
@pytest.mark.parametrize("runs", ["0", "-3", "two"])
def test_basis_gate_demo_refuses_a_run_count_below_one_before_any_run(
    script: str, runs: str, capsys: pytest.CaptureFixture[str]
) -> None:
    main = importlib.import_module(script).main

    with pytest.raises(SystemExit) as refused:
        main(["--runs", runs])

    assert refused.value.code == 2
    streams = capsys.readouterr()
    assert "--runs" in streams.err
    assert streams.out == ""
