# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Acquire-or-fail on the service: a write that asks to take an artifact only
if nobody holds it (``write(..., if_unheld=True)``).

The holder check runs inside the write's own hold, after the giver fence and
before the first state write, so the check and the grant are one decision: a
writer that finds another agent holding the artifact EXCLUSIVE or MODIFIED is
refused with :class:`OtherHolder` naming that agent, and nothing changes.

Every scenario runs against BOTH registries. Identities follow the HTTP shape:
the session-level identity is the parent's composite agent id, passed as
``caller``, and a subagent writes under its own composite. "Another agent" is
any composite other than the writer's own.

Scenarios:

- an EXCLUSIVE or MODIFIED holder refuses an opted-in writer and keeps its
  state, and the refusal leaves no transient, observed version or log entry;
- SHARED readers are not holders; a holder's own repeat is admitted;
- siblings refuse each other, and a parent and its subagent refuse each other;
- the giver of a live handoff still gets the giver fence while the successor
  holds the artifact, and a refused bystander leaves the record's label alone;
- an ordinary write displaces exactly as before;
- two opted-in writers held together at the write's entry end with one grant,
  and the same race with the check moved before the hold grants both.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from ccs.coordinator.registry import ArtifactRegistry
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import OTHER_HOLDER_REASON, GiverFenced, OtherHolder
from ccs.core.states import MESIState
from ccs.core.types import TRANSFER_STATUS_COMPLETED, TRANSFER_STATUS_OVERTAKEN
from tests.test_service_handoff import _hand_off, _label, _Log, _path, _read

_BARRIER_TIMEOUT_SEC = 5.0
_RACE_ROUNDS = 20


@pytest.fixture(params=["memory", "sqlite"])
def service(request: pytest.FixtureRequest, tmp_path: Path):
    """The service over each registry."""
    if request.param == "memory":
        yield CoordinatorService(ArtifactRegistry())
        return
    registry = SqliteArtifactRegistry(tmp_path / "acquire-or-fail.db")
    yield CoordinatorService(registry)
    registry.close()


@pytest.fixture(params=["memory", "sqlite"])
def logged_service(request: pytest.FixtureRequest, tmp_path: Path):
    """The service over each registry with a state log wired in."""
    log = _Log()
    if request.param == "memory":
        yield CoordinatorService(ArtifactRegistry(state_log=log, instance_id="acquire")), log
        return
    registry = SqliteArtifactRegistry(
        tmp_path / "acquire-or-fail-logged.db", state_log=log, instance_id="acquire"
    )
    yield CoordinatorService(registry), log
    registry.close()


def _acquire(service: CoordinatorService, path: UUID, agent: UUID, session: UUID | None = None) -> Any:
    """An opted-in acquire by ``agent``, an agent of ``session`` (its own
    session when ``None``, as a parent is)."""
    return service.write(
        agent_id=agent, artifact_id=path, caller=session or agent, if_unheld=True
    )


def _commit(service: CoordinatorService, path: UUID, agent: UUID) -> None:
    """``agent`` holds ``path`` and commits it, so it is MODIFIED."""
    service.write(agent_id=agent, artifact_id=path, caller=agent)
    service.commit(agent_id=agent, artifact_id=path, content="v2", caller=agent)


def _refused(service: CoordinatorService, path: UUID, agent: UUID, session: UUID | None = None) -> OtherHolder:
    with pytest.raises(OtherHolder) as caught:
        _acquire(service, path, agent, session)
    return caught.value


def _snapshot(service: CoordinatorService, path: UUID, agents: tuple[UUID, ...]) -> dict:
    """What a refusal must leave as it was, read without side effects. Notices
    are the route's, not the service's, so the route's tests cover them."""
    registry = service.registry
    return {
        "states": registry.get_state_map(path),
        "transients": registry.get_transient_map(path),
        "observed": {agent: registry.last_observed_version_for(path, agent) for agent in agents},
        "record": registry.get_transfer_record(path),
    }


# ---------------------------------------------------------------------------
# A holder refuses
# ---------------------------------------------------------------------------


def test_an_exclusive_holder_refuses_an_opted_in_writer_and_nothing_changes(service) -> None:
    path = _path(service)
    holder, caller = uuid4(), uuid4()
    service.write(agent_id=holder, artifact_id=path, caller=holder)
    before = _snapshot(service, path, (holder, caller))

    refused = _refused(service, path, caller)

    assert (refused.reason, refused.artifact_id, refused.holder_agent_id) == (
        OTHER_HOLDER_REASON,
        path,
        holder,
    )
    assert isinstance(refused.holder_agent_id, UUID)
    assert _snapshot(service, path, (holder, caller)) == before
    assert service.registry.get_agent_state(path, caller) is None


def test_a_modified_holder_refuses_and_stays_modified(service) -> None:
    path = _path(service)
    holder, caller = uuid4(), uuid4()
    _commit(service, path, holder)

    assert _refused(service, path, caller).holder_agent_id == holder
    assert service.registry.get_agent_state(path, holder) is MESIState.MODIFIED


def test_a_refusal_adds_no_state_log_entry_while_a_displacing_write_does(logged_service) -> None:
    service, log = logged_service
    path = _path(service)
    holder, caller = uuid4(), uuid4()
    service.write(agent_id=holder, artifact_id=path, caller=holder)
    logged = len(log.entries)

    _refused(service, path, caller)
    assert len(log.entries) == logged

    service.write(agent_id=caller, artifact_id=path, caller=caller)
    assert len(log.entries) > logged


# ---------------------------------------------------------------------------
# Who counts as a holder
# ---------------------------------------------------------------------------


def test_shared_readers_are_not_holders(service) -> None:
    path = _path(service)
    reader, caller = uuid4(), uuid4()
    _read(service, path, reader)

    _acquire(service, path, caller)

    assert service.registry.get_agent_state(path, caller) is MESIState.EXCLUSIVE
    assert service.registry.get_agent_state(path, reader) is MESIState.INVALID


def test_a_holders_own_repeat_is_admitted(service) -> None:
    path = _path(service)
    holder = uuid4()
    _commit(service, path, holder)

    _acquire(service, path, holder)

    assert service.registry.get_agent_state(path, holder) is MESIState.EXCLUSIVE


def test_sibling_subagents_refuse_each_other(service) -> None:
    path = _path(service)
    session, first, second = uuid4(), uuid4(), uuid4()
    _acquire(service, path, first, session)

    assert _refused(service, path, second, session).holder_agent_id == first


def test_a_subagent_is_refused_on_a_file_its_parent_holds(service) -> None:
    """The parent's composite is the session identity the route passes as
    ``caller``, so a check that skipped the session identity rather than the
    writer's own composite would admit the subagent over its parent."""
    path = _path(service)
    parent, subagent = uuid4(), uuid4()
    _commit(service, path, parent)

    assert _refused(service, path, subagent, parent).holder_agent_id == parent
    assert service.registry.get_agent_state(path, parent) is MESIState.MODIFIED


def test_a_parent_is_refused_on_a_file_its_subagent_holds(service) -> None:
    path = _path(service)
    parent, subagent = uuid4(), uuid4()
    _acquire(service, path, subagent, parent)

    assert _refused(service, path, parent).holder_agent_id == subagent


# ---------------------------------------------------------------------------
# Precedence against a handoff
# ---------------------------------------------------------------------------


def test_the_giver_fence_answers_before_the_holder_check(service) -> None:
    """The successor holds the handed path, so a holder check placed before
    the giver fence would answer the giver with a retryable refusal and hide
    its terminal fence."""
    handoff = _hand_off(service)
    service.write(agent_id=handoff.successor, artifact_id=handoff.path, caller=handoff.successor)
    assert _label(service, handoff.path) == (TRANSFER_STATUS_COMPLETED, None, True)

    with pytest.raises(GiverFenced):
        _acquire(service, handoff.path, uuid4(), handoff.giver)


def test_a_refused_bystander_leaves_the_records_label_alone(service) -> None:
    handoff = _hand_off(service)
    service.write(agent_id=handoff.successor, artifact_id=handoff.path, caller=handoff.successor)
    label = _label(service, handoff.path)

    assert _refused(service, handoff.path, uuid4()).holder_agent_id == handoff.successor
    assert _label(service, handoff.path) == label


def test_a_bystander_on_an_unheld_pending_record_is_admitted_and_overtakes(service) -> None:
    handoff = _hand_off(service)
    bystander = uuid4()

    _acquire(service, handoff.path, bystander)

    assert _label(service, handoff.path) == (TRANSFER_STATUS_OVERTAKEN, bystander, True)


# ---------------------------------------------------------------------------
# The default write
# ---------------------------------------------------------------------------


def test_an_ordinary_write_still_displaces_a_holder(service) -> None:
    path = _path(service)
    holder, caller = uuid4(), uuid4()
    service.write(agent_id=holder, artifact_id=path, caller=holder)

    signals = service.write(agent_id=caller, artifact_id=path, caller=caller)

    assert len(signals) == 1
    assert service.registry.get_agent_state(path, caller) is MESIState.EXCLUSIVE
    assert service.registry.get_agent_state(path, holder) is MESIState.INVALID


# ---------------------------------------------------------------------------
# The race
# ---------------------------------------------------------------------------


def _hold_both_at_write_entry(monkeypatch: pytest.MonkeyPatch, *, check_first: bool) -> list[bool]:
    """Hold both racers at the write's entry, outside its hold, until both
    arrive. Returns the record of racers that passed the barrier; a racer
    whose barrier broke raises and never writes.

    With ``check_first`` this is the moved-check mutant: each racer checks for
    a holder, then waits, then takes the guarded write without the flag."""
    real = CoordinatorService.write
    barrier = threading.Barrier(2, timeout=_BARRIER_TIMEOUT_SEC)
    passed: list[bool] = []

    def held(self: CoordinatorService, **kwargs: Any) -> Any:
        if check_first and kwargs.get("if_unheld"):
            self._refuse_other_holder(kwargs["artifact_id"], kwargs["agent_id"])
            kwargs["if_unheld"] = False
        barrier.wait()
        passed.append(True)
        return real(self, **kwargs)

    monkeypatch.setattr(CoordinatorService, "write", held)
    return passed


def _race(service: CoordinatorService, path: UUID) -> dict[UUID, object]:
    """Two opted-in writers on ``path`` at once; each one's return or error."""
    outcomes: dict[UUID, object] = {}

    def contend(agent: UUID) -> None:
        try:
            outcomes[agent] = _acquire(service, path, agent)
        except Exception as exc:
            outcomes[agent] = exc

    threads = [threading.Thread(target=contend, args=(uuid4(),)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(_BARRIER_TIMEOUT_SEC * 2)
    return outcomes


def test_two_opted_in_writers_held_together_end_with_one_grant(service, monkeypatch) -> None:
    passed = _hold_both_at_write_entry(monkeypatch, check_first=False)
    for round_no in range(_RACE_ROUNDS):
        path = _path(service, f"race-{round_no}.md")
        outcomes = _race(service, path)
        assert len(passed) == 2 * (round_no + 1), "INCONCLUSIVE: the racers were not held together"

        refusals = [o for o in outcomes.values() if isinstance(o, OtherHolder)]
        winners = [agent for agent, o in outcomes.items() if isinstance(o, list)]
        assert len(refusals) == 1 and len(winners) == 1, outcomes
        assert refusals[0].holder_agent_id == winners[0]
        assert service.registry.get_state_map(path) == {winners[0]: MESIState.EXCLUSIVE}


def test_the_same_race_with_the_check_moved_before_the_hold_grants_both(service, monkeypatch) -> None:
    """The mutation: each racer checks before the hold and finds nobody, so
    both are granted. Proves the race above can see a check that is not part
    of the grant's decision."""
    passed = _hold_both_at_write_entry(monkeypatch, check_first=True)
    for round_no in range(_RACE_ROUNDS):
        path = _path(service, f"mutant-{round_no}.md")
        outcomes = _race(service, path)
        assert len(passed) == 2 * (round_no + 1), "INCONCLUSIVE: the racers were not held together"

        assert all(isinstance(o, list) for o in outcomes.values()), outcomes
