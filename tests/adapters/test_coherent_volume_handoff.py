# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``CoherentVolume`` as a handoff client (#185).

A volume hands a path to a named successor, accepts, declines or withdraws a
handoff, and gets typed results back. Every request a volume sends names its
CURRENT incarnation, but the claim a transfer gives up may be held by an
earlier one: a read stays with the incarnation that took it when a later
re-mint rotates the volume, and a write grant stays with the incarnation that
wrote. So a transfer presents, per path, the incarnation that holds the claim;
presenting the current one hands over nothing and is refused as not held. A
giver's later write is refused with the typed ``handed_off`` reason on
both the pre-edit and the compare-and-swap route, and the volume raises one
terminal for it, which it never retries.

Every test runs against a live coordinator the first volume spawns; every
other volume sibling-attaches to it, so each volume is its own session.
"""

from __future__ import annotations

import io
import json
import urllib.error
import warnings
from pathlib import Path
from uuid import uuid4

import pytest

import ccs.adapters as adapters
import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.coordinator_server import session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import (
    CasCommitResult,
    CoherentVolume,
    HandoffGrantResult,
    HandoffTransferResult,
    HandoffVerbResult,
    HandoffWinOutcome,
)
from ccs.cli._coherence_client import CoordinatorUnavailable, resolve_endpoint
from ccs.cli._coherence_client import post as _cc_post
from ccs.core.exceptions import CoherenceDegradedWarning, CoherenceError, CommitUnconfirmed, GiverFenced

_MANAGED = ("data/**",)
_PLAN = "data/plan.md"
_OTHER = "data/other.md"
_IN_FLIGHT_DETAIL = (  # frozen duplicate of the coordinator's static text
    "another session's handoff of this path is live; it ends when its "
    "successor declines, its giver withdraws, or any session writes the path"
)


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Coordinator config tuned for fast tests (no idle shutdown)."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


def _seed(tmp_path: Path, rel: str, content: bytes = b"v1") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _volumes(tmp_path: Path, cfg: LifecycleConfig, count: int) -> list[CoherentVolume]:
    """``count`` volumes on one workspace and one coordinator: the first spawns
    it, the rest sibling-attach. Each is its own session with a bound caller
    principal, which is what makes its session-level id a known successor."""
    return [CoherentVolume(tmp_path, managed=_MANAGED, config=cfg) for _ in range(count)]


def _agent(vol: CoherentVolume) -> str:
    """The volume's session-level agent id, the id a successor is named by and
    every answer about a record uses: derived from the session id alone, the
    way the coordinator derives it."""
    return str(session_to_agent_id(vol.session_id))


def _record_posts(
    monkeypatch: pytest.MonkeyPatch, sent: list[tuple[str, dict]]
) -> None:
    """Record every coordinator request as ``(route, body)`` and forward it."""
    real_post = coherent_volume_module._coordinator_post

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        sent.append((path, dict(payload)))
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)


def _presented(sent: list[tuple[str, dict]]) -> list[dict]:
    """The grants of every transfer request sent."""
    return [grant for route, body in sent if route == "/handoff/transfer" for grant in body["grants"]]


def _held(vol: CoherentVolume, row: str) -> dict[str, str]:
    """What the COORDINATOR says grant row ``row`` holds, ``{path: state}``, from
    ``/status``, which lists every non-INVALID row: ``{}`` means nothing."""
    status = vol.coordinator_status()
    assert status is not None
    for session in status["sessions"]:
        if session["agent_id"] == row:
            return dict(session["states"])
    return {}


def _status_handoff(vol: CoherentVolume, rel: str) -> dict | None:
    status = vol.coordinator_status()
    assert status is not None
    [entry] = [e for e in status["tracked_artifacts"] if e["path"] == rel]
    return entry.get("handoff")


# --- the giver terminal, on both routes, never retried ----------------------


@pytest.mark.parametrize("surface", ["write_cas_at", "write_cas"])
def test_a_givers_cas_at_the_transfer_version_raises_the_giver_terminal(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, surface: str
) -> None:
    """A reads plan.md at v1 and hands it to B, writes another
    path, then compare-and-swaps plan.md at the transfer version from a fresh
    incarnation. Before the fence that sequence won and overwrote the file. It
    is refused with the typed giver terminal naming B and v1, the disk is
    unchanged, the version stays 1, and exactly one commit is posted: the
    terminal is never retried, by the single-shot CAS or by the loop."""
    target = _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        assert giver.read(_PLAN) == b"plan v1"
        result = giver.transfer(_PLAN, successor=_agent(successor))
        assert result.ok
        giver.write(_OTHER, b"other v2")
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        with pytest.raises(GiverFenced) as raised:
            if surface == "write_cas_at":
                giver.write_cas_at(_PLAN, 1, b"late giver bytes")
            else:
                giver.write_cas(_PLAN, lambda cur: cur + b" + late giver edit")

        assert type(raised.value) is GiverFenced
        assert raised.value.reason == "handed_off"
        assert raised.value.artifact_id == _PLAN
        assert raised.value.successor == _agent(successor)
        assert raised.value.version_at_transfer == 1
        assert target.read_bytes() == b"plan v1"
        assert [route for route, _b in sent].count("/hooks/post-edit-cas") == 1
        assert successor.read_with_version(_PLAN) == (b"plan v1", 1)
    finally:
        stop_coordinator(tmp_path)


def test_a_givers_multi_file_publish_including_a_handed_path_raises_the_giver_terminal(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A hands plan.md to B, then publishes plan.md and other.md together.
    The coordinator refuses the batch as handed off, and the volume raises the
    same typed giver terminal as on its single-path writes, naming B and v1,
    rather than the generic batch conflict, whose recovery (reacquire and
    re-decide) never clears the fence and would loop. Nothing is written."""
    target = _seed(tmp_path, _PLAN, b"plan v1")
    other = _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        assert giver.read_with_version(_PLAN) == (b"plan v1", 1)
        assert giver.read_with_version(_OTHER) == (b"other v1", 1)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok

        with pytest.raises(GiverFenced) as raised:
            giver.atomic_publish([(_PLAN, 1, b"late plan"), (_OTHER, 1, b"other v2")])

        assert type(raised.value) is GiverFenced
        assert raised.value.artifact_id == _PLAN
        assert raised.value.successor == _agent(successor)
        assert raised.value.version_at_transfer == 1
        assert target.read_bytes() == b"plan v1"
        assert other.read_bytes() == b"other v1"
    finally:
        stop_coordinator(tmp_path)

def test_the_givers_pre_edit_raises_the_same_terminal_without_a_reacquire(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coordinator answers a fenced giver's pre-edit with the
    typed ``handed_off`` reason AND a deny envelope whose prose is the human
    text, which the volume's deny-reason helper prefers. Classified from that
    prose the answer read as an ordinary stale view (retryable, recover by
    reacquire), and a reacquire cannot clear a fence keyed on the session: the
    giver loops. It raises the same terminal type as the compare-and-swap
    route, sends nothing after the refused pre-edit (no reacquire, no
    re-mint), writes nothing, and leaves no grant recorded, so the next
    re-mint spends no release on it."""
    target = _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        incarnation = giver._incarnation
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        with pytest.raises(GiverFenced) as raised:
            giver.write(_PLAN, b"late giver bytes")

        assert type(raised.value) is GiverFenced
        assert raised.value.successor == _agent(successor)
        assert raised.value.version_at_transfer == 1
        assert [route for route, _b in sent] == ["/hooks/pre-edit"]
        assert giver._incarnation == incarnation
        assert target.read_bytes() == b"plan v1"

        sent.clear()
        giver.reacquire(_PLAN)
        assert "/hooks/session-stop" not in [route for route, _b in sent], (
            "the refused pre-edit took no grant, yet the re-mint released one")
    finally:
        stop_coordinator(tmp_path)


# --- the transfer presents the incarnation holding the claim ----------------


def test_a_transfer_after_a_re_mint_presents_the_incarnation_holding_the_read(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read plan.md, then compare-and-swapped another path,
    which re-minted its incarnation. The read stays with the incarnation that
    took it, so a transfer presenting the CURRENT incarnation offers a claim
    that holds nothing on plan.md and is refused as not held. The transfer
    presents the read's incarnation and hands over the standing SHARED read."""
    _seed(tmp_path, _PLAN, b"plan v4")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        reader = giver._incarnation
        _data, other_version = giver.read_with_version(_OTHER)
        giver.write_cas_at(_OTHER, other_version, b"other v2")
        assert giver._incarnation != reader, "precondition: the CAS re-minted"
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert result == HandoffTransferResult(grants=(HandoffGrantResult(
            path=_PLAN, transferred=True, giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=1, hold_shape="SHARED", status="pending",
        ),))
        assert result.ok is True
        assert _presented(sent) == [{"path": _PLAN, "agent_id": reader}]
    finally:
        stop_coordinator(tmp_path)


def test_a_multi_file_publishs_members_are_read_before_they_can_be_handed_on(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A multi-file publish commits through its snapshot session, which leaves
    the members held by that session's commit and not by any incarnation the
    volume can present. A transfer right after it is refused ``handoff_not_held``,
    as ``transfer()`` and ``atomic_publish()`` say, and a read of the member
    registers a claim the transfer then hands on at the published version."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read_with_version(_PLAN)
        giver.read_with_version(_OTHER)
        assert giver.atomic_publish([(_PLAN, 1, b"plan v2"), (_OTHER, 1, b"other v2")]) == {
            _PLAN: 2, _OTHER: 2,
        }

        refused = giver.transfer(_PLAN, successor=_agent(successor))
        giver.read(_PLAN)
        handed = giver.transfer(_PLAN, successor=_agent(successor))

        assert refused.grants[0].reason == "handoff_not_held"
        [grant] = handed.grants
        assert (grant.transferred, grant.version_at_transfer, grant.hold_shape) == (True, 2, "SHARED")
    finally:
        stop_coordinator(tmp_path)


def test_a_transfer_after_a_cas_win_presents_the_winning_incarnation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The compare-and-swap re-mints, takes its comparand read under the new
    incarnation, and wins; the coordinator leaves that incarnation SHARED and
    the first reader INVALID as a peer of the win. A map written only by the
    plain read still named the first reader, and the transfer was refused as
    not held. It presents the winning incarnation and gives up SHARED."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        first_reader = giver._incarnation
        giver.write_cas_at(_PLAN, 1, b"plan v2")
        winner = giver._incarnation
        assert winner != first_reader, "precondition: the CAS re-minted"
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert result.grants == (HandoffGrantResult(
            path=_PLAN, transferred=True, giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=2, hold_shape="SHARED", status="pending",
        ),)
        assert _presented(sent) == [{"path": _PLAN, "agent_id": winner}]
    finally:
        stop_coordinator(tmp_path)


def test_a_transfer_after_a_pessimistic_write_presents_the_writing_incarnation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read plan.md, rotated its incarnation by reacquiring another path,
    then wrote plan.md with the pessimistic write. The writing incarnation's
    acquire invalidated the earlier reader, so presenting the read's
    incarnation is refused as not held; the write grant is the claim, and the
    transfer gives up MODIFIED."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        reader = giver._incarnation
        giver.reacquire(_OTHER)
        giver.write(_PLAN, b"plan v2")
        writer = giver._incarnation
        assert writer != reader, "precondition: the reacquire re-minted"
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert result.grants == (HandoffGrantResult(
            path=_PLAN, transferred=True, giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=2, hold_shape="MODIFIED", status="pending",
        ),)
        assert _presented(sent) == [{"path": _PLAN, "agent_id": writer}]
    finally:
        stop_coordinator(tmp_path)


def test_a_verify_only_read_does_not_move_the_presented_incarnation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A verification read (the effect fence's ``observe=False``) asks the
    coordinator not to register a view, so after a rotation it leaves the new
    incarnation holding nothing on the path. Recorded as the path's reader, it
    would make the transfer present that empty claim and be refused as not
    held. The transfer still presents the incarnation that holds the read."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        reader = giver._incarnation
        giver.reacquire(_OTHER)
        assert giver._incarnation != reader, "precondition: the reacquire re-minted"
        giver.read_with_version_generation(_PLAN, observe=False)
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert result.grants == (HandoffGrantResult(
            path=_PLAN, transferred=True, giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=1, hold_shape="SHARED", status="pending",
        ),)
        assert _presented(sent) == [{"path": _PLAN, "agent_id": reader}]
    finally:
        stop_coordinator(tmp_path)


def test_a_write_of_a_path_never_read_presents_the_writing_incarnation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pessimistic write of a path the volume never read: the write grant is
    the only claim, and the transfer gives it up as MODIFIED. The transferred
    path leaves the write record with it, so the next re-mint spends no
    release on an incarnation that recorded nothing else."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.write(_PLAN, b"plan v2")
        writer = giver._incarnation
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert result.grants == (HandoffGrantResult(
            path=_PLAN, transferred=True, giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=2, hold_shape="MODIFIED", status="pending",
        ),)
        assert _presented(sent) == [{"path": _PLAN, "agent_id": writer}]
        sent.clear()
        giver.reacquire(_OTHER)
        assert "/hooks/session-stop" not in [route for route, _b in sent]
    finally:
        stop_coordinator(tmp_path)


def test_a_transferred_path_leaves_the_write_record_and_the_rest_stays(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One incarnation wrote two paths and hands one of them on. The record
    keeps that incarnation for the other path, which it still holds: the next
    re-mint releases it, one stop naming the writing incarnation, and the
    grant is gone afterwards."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    spare = "data/spare.md"
    _seed(tmp_path, spare, b"spare v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.write(_PLAN, b"plan v2")
        giver.write(_OTHER, b"other v2")
        writer = giver._incarnation
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        giver.reacquire(spare)

        stops = [body for route, body in sent if route == "/hooks/session-stop"]
        assert [body["agent_id"] for body in stops] == [writer]
        assert _held(giver, str(session_to_agent_id(giver.session_id, writer))) == {}
    finally:
        stop_coordinator(tmp_path)


def test_a_stranded_write_grant_is_presented_ahead_of_the_current_incarnation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The write record names an incarnation that is no longer current when
    its release at a re-mint was not confirmed: the abandoned incarnation
    still holds the grant. The transfer presents it, not the current
    incarnation (which holds nothing on the path and would be refused as not
    held), and once it is handed on the record no longer names it, so the
    next re-mint sends it no release."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.write(_PLAN, b"plan v2")
        writer = giver._incarnation
        real_post = coherent_volume_module._coordinator_post
        unconfirmed = [1]

        def refuse_one_release(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/session-stop" and unconfirmed[0]:
                unconfirmed[0] -= 1
                return {"ok": False, "reason": "internal: RuntimeError"}  # not released
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", refuse_one_release)
        giver.reacquire(_OTHER)
        assert unconfirmed == [0] and giver._incarnation != writer, "precondition"
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert result.grants == (HandoffGrantResult(
            path=_PLAN, transferred=True, giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=2, hold_shape="MODIFIED", status="pending",
        ),)
        assert _presented(sent) == [{"path": _PLAN, "agent_id": writer}]
        sent.clear()
        giver.reacquire(_OTHER)
        assert "/hooks/session-stop" not in [route for route, _b in sent]
    finally:
        stop_coordinator(tmp_path)


def test_a_refused_grant_is_returned_typed_beside_a_transferred_one(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A transfer naming a path the volume holds and one it never touched
    answers per grant, in request order: the first transferred, the second
    refused as not held. A refusal is a typed value, never a raise, and the
    top-level result is success only when every grant transferred."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)

        result = giver.transfer([_PLAN, _OTHER], successor=_agent(successor))

        assert result == HandoffTransferResult(grants=(
            HandoffGrantResult(
                path=_PLAN, transferred=True, giver=_agent(giver), successor=_agent(successor),
                version_at_transfer=1, hold_shape="SHARED", status="pending",
            ),
            HandoffGrantResult(path=_OTHER, transferred=False, reason="handoff_not_held"),
        ))
        assert result.ok is False
    finally:
        stop_coordinator(tmp_path)


# --- the compare-and-swap win and its handoff outcome -----------------------


def test_a_cas_win_returns_its_version_and_the_handoff_outcome(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The successor's compare-and-swap win completes the handoff and says so;
    a bystander's win on a live handoff marks it overtaken and the result
    names the pair it wrote past and the bystander. A win on a path with no
    record carries no outcome. Each returns the version it committed."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor, bystander = _volumes(tmp_path, fast_cfg, 3)
    try:
        giver.read(_PLAN)
        giver.read(_OTHER)
        assert giver.transfer([_PLAN, _OTHER], successor=_agent(successor)).ok

        completed = successor.write_cas_at(_PLAN, 1, b"plan v2 by the successor")
        overtaken = bystander.write_cas(_OTHER, lambda cur: cur + b" + bystander")
        plain = bystander.write_cas_at(_PLAN, 2, b"plan v3 by the bystander")

        assert completed == CasCommitResult(path=_PLAN, version=2, handoff=HandoffWinOutcome(
            outcome="completed", giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=1,
        ))
        assert overtaken == CasCommitResult(path=_OTHER, version=2, handoff=HandoffWinOutcome(
            outcome="overtaken", giver=_agent(giver), successor=_agent(successor),
            version_at_transfer=1, counterparty=_agent(bystander),
        ))
        assert plain == CasCommitResult(path=_PLAN, version=3, handoff=None)
    finally:
        stop_coordinator(tmp_path)


# --- accept, decline, withdraw ---------------------------------------------


def test_accept_decline_and_withdraw_return_typed_results(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The three verbs as typed values: a party's verb is taken and reports
    the record's status after it; a verb from the wrong party, on an ended
    record, or on a path with no record is refused with the typed reason and
    the status it carries, without raising."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        giver.read(_OTHER)
        assert successor.accept(_OTHER) == HandoffVerbResult(
            path=_OTHER, ok=False, reason="handoff_not_live")
        assert giver.transfer([_PLAN, _OTHER], successor=_agent(successor)).ok

        assert giver.accept(_PLAN) == HandoffVerbResult(
            path=_PLAN, ok=False, reason="handoff_not_successor", status="pending")
        assert successor.withdraw(_PLAN) == HandoffVerbResult(
            path=_PLAN, ok=False, reason="handoff_not_giver", status="pending")
        assert successor.accept(_PLAN) == HandoffVerbResult(path=_PLAN, ok=True, status="completed")
        assert giver.withdraw(_PLAN) == HandoffVerbResult(path=_PLAN, ok=True, status="withdrawn")
        assert successor.decline(_PLAN) == HandoffVerbResult(
            path=_PLAN, ok=False, reason="handoff_not_live", status="withdrawn")
        assert successor.decline(_OTHER) == HandoffVerbResult(path=_OTHER, ok=True, status="declined")
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize(
    ("verb", "route", "reason"),
    [
        ("transfer", "/handoff/transfer", "handoff_transfer_unconfirmed"),
        ("accept", "/handoff/accept", "handoff_accept_unconfirmed"),
        ("decline", "/handoff/decline", "handoff_decline_unconfirmed"),
        ("withdraw", "/handoff/withdraw", "handoff_withdraw_unconfirmed"),
    ],
)
@pytest.mark.parametrize("on_error", ["strict", "degrade"])
def test_an_unconfirmed_handoff_answer_raises_in_both_modes(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    verb: str, route: str, reason: str, on_error: str,
) -> None:
    """A verb the watchdog cut short answers that its outcome is unknown: it
    may still land. Returned as a value it would read as a refusal that
    changed nothing, or as a success; it raises the unconfirmed terminal in
    both ``on_error`` modes, and a transfer drops nothing from what the volume
    records, so the same transfer sent again presents the same claim."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=fast_cfg)
    successor = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=fast_cfg)
    try:
        giver.read(_PLAN)
        reader = giver._incarnation
        real_post = coherent_volume_module._coordinator_post

        def unconfirmed(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == route:
                return {"ok": False, "degraded": True, "reason": reason}
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", unconfirmed)
        with pytest.raises(CommitUnconfirmed) as raised:
            if verb == "transfer":
                giver.transfer(_PLAN, successor=_agent(successor))
            else:
                getattr(giver, verb)(_PLAN)
        assert reason in str(raised.value)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", real_post)
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        assert _presented(sent) == [{"path": _PLAN, "agent_id": reader}]
    finally:
        stop_coordinator(tmp_path)


#: Answers to accept, decline or withdraw that settle nothing: the
#: coordinator's failure envelope, a refusal with no reason, no ``ok`` at all,
#: and an ``ok`` that is not a boolean. Each verb may have landed.
_UNCLASSIFIABLE_VERB_ANSWERS = [
    {"ok": False, "reason": "internal: RuntimeError"},
    {"ok": False},
    {},
    {"ok": "yes"},
]


@pytest.mark.parametrize("answer", _UNCLASSIFIABLE_VERB_ANSWERS)
@pytest.mark.parametrize("verb", ["accept", "decline", "withdraw"])
def test_a_verb_answer_that_is_not_a_typed_refusal_raises_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    verb: str, answer: dict,
) -> None:
    """Only the verbs' own typed refusals are a refusal that changed nothing.
    Any other answer that is not ``ok: true`` -- the coordinator's failure
    envelope, a refusal with no reason, an answer without ``ok`` -- does not
    settle the outcome, and the verb may have landed: an accept may have
    completed the handoff, a decline or withdraw ended it. Returned as a
    refusal it would tell the caller nothing changed, so it raises the
    unconfirmed terminal."""
    _seed(tmp_path, _PLAN, b"plan v1")
    volume = CoherentVolume(tmp_path, managed=_MANAGED, on_error="strict", config=fast_cfg)
    try:
        volume.read(_PLAN)
        real_post = coherent_volume_module._coordinator_post

        def unclassifiable(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == f"/handoff/{verb}":
                return dict(answer)
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", unclassifiable)
        with pytest.raises(CommitUnconfirmed, match="no outcome this client can classify"):
            getattr(volume, verb)(_PLAN)
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize("paths", [[], [_PLAN, "./" + _PLAN]], ids=["no-path", "one-path-twice"])
def test_a_transfer_of_no_path_or_one_path_twice_is_a_caller_error_never_sent(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    on_error: str, paths: list[str],
) -> None:
    """The coordinator answers HTTP 400 to an empty grant list or a path named
    twice, which strict mode raised as a generic ``CoherenceError`` and degrade
    mode as ``CommitUnconfirmed``, "may have landed", for a request that
    certainly did not. Like ``atomic_publish`` with an empty or repeated
    write-set, it is the caller's error: ``ValueError`` in both modes, and
    nothing is sent."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=fast_cfg)
    successor = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=fast_cfg)
    try:
        giver.read(_PLAN)
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        with pytest.raises(ValueError):
            giver.transfer(paths, successor=_agent(successor))

        assert _presented(sent) == []
    finally:
        stop_coordinator(tmp_path)


#: FROZEN: answers to a one-path transfer that do not settle the outcome. Each
#: reaches the grant-list check: none is degraded, and each is a JSON object.
_UNCLASSIFIABLE_TRANSFER_ANSWERS = [
    {"ok": False, "reason": "internal: RuntimeError"},  # the failure envelope
    {"grants": []},  # a grant short
    {"grants": [{"path": _PLAN, "transferred": True}, {"path": _OTHER, "transferred": True}]},
    {"grants": [{"path": _OTHER, "transferred": True}]},  # another path
    {"grants": [{"path": _PLAN, "transferred": "yes"}]},  # not a bool
    {"grants": ["data/plan.md"]},  # not an entry
]


@pytest.mark.parametrize("answer", _UNCLASSIFIABLE_TRANSFER_ANSWERS)
def test_a_transfer_answer_without_one_readable_grant_per_path_raises_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, answer: dict
) -> None:
    """A transfer is settled only by one readable entry per path, in order.
    Without the check, ``zip`` drops a path the coordinator never answered and
    ``HandoffTransferResult.ok`` can read ``True``, and the failure envelope
    escapes as a ``TypeError`` no caller maps to "look at the handoff first".
    The transfer may have landed, so each raises the unconfirmed terminal and
    the volume still presents its claim on the path."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        real_post = coherent_volume_module._coordinator_post

        def unclassifiable(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/handoff/transfer":
                return json.loads(json.dumps(answer))
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", unclassifiable)
        with pytest.raises(CommitUnconfirmed, match="no outcome this client can classify"):
            giver.transfer(_PLAN, successor=_agent(successor))
        assert giver._claim_incarnation(_PLAN) in giver._read_incarnations.values()
    finally:
        stop_coordinator(tmp_path)


#: The status a verb leaves on the path's record once it has landed.
_LANDED_STATUS = {"transfer": "pending", "accept": "completed", "decline": "declined", "withdraw": "withdrawn"}


def _http_error(code: int, body: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "http://coordinator", code, "x", {}, io.BytesIO(json.dumps(body).encode())  # type: ignore[arg-type]
    )


def _fail_on(route: str, kind: str) -> object:
    """``_coordinator_post`` that fails ``route`` the way ``kind`` names: the
    answer lost after the request was forwarded (a reset, an HTTP 500), or an
    HTTP 503 before it was."""
    real_post = coherent_volume_module._coordinator_post

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path != route:
            return real_post(endpoint, path, payload, **kwargs)
        if kind == "http_503_before_forwarding":
            raise _http_error(503, {"error": "handler concurrency exceeded"})
        real_post(endpoint, path, payload, **kwargs)
        if kind == "reset_after_forwarding":
            raise CoordinatorUnavailable("simulated: the answer was lost")
        raise _http_error(500, {"error": "internal"})

    return post


def _act(verb: str, giver: CoherentVolume, successor: CoherentVolume) -> object:
    """Run ``verb`` from the side that sends it: the giver transfers and
    withdraws, the successor accepts and declines."""
    if verb == "transfer":
        return giver.transfer(_PLAN, successor=_agent(successor))
    return getattr(successor if verb in ("accept", "decline") else giver, verb)(_PLAN)


def _volume_pair(tmp_path: Path, cfg: LifecycleConfig, on_error: str, verb: str) -> tuple[CoherentVolume, CoherentVolume]:
    """A giver that read the path and a successor; for every verb but transfer,
    the giver has already handed the path to the successor. Call it inside the
    test's ``try``: the first volume spawns the coordinator its ``finally``
    stops."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=cfg)
    successor = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=cfg)
    giver.read(_PLAN)
    if verb != "transfer":
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
    return giver, successor


@pytest.mark.parametrize("kind", ["reset_after_forwarding", "http_500_after_forwarding", "http_503_before_forwarding"])
@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_a_handoff_verb_whose_answer_is_lost_raises_unconfirmed_in_both_modes(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, verb: str, on_error: str, kind: str
) -> None:
    """A lost answer or a 5xx on a handoff verb is an unknown outcome in BOTH
    modes (#276). Under strict it used to raise a plain ``CoherenceError``,
    which reads as "nothing happened" although the verb may have landed; the
    caller would then repeat it. The record shows the verb did land whenever
    the request was forwarded."""
    try:
        giver, successor = _volume_pair(tmp_path, fast_cfg, on_error, verb)
        before = (_status_handoff(giver, _PLAN) or {}).get("status")
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", _fail_on(f"/handoff/{verb}", kind))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CoherenceDegradedWarning)
            with pytest.raises(CommitUnconfirmed) as raised:
                _act(verb, giver, successor)
        assert type(raised.value) is CommitUnconfirmed
        assert "whether it landed is unknown" in str(raised.value)
        monkeypatch.undo()
        after = (_status_handoff(giver, _PLAN) or {}).get("status")
        assert after == (before if kind == "http_503_before_forwarding" else _LANDED_STATUS[verb])
        sender = successor if verb in ("accept", "decline") else giver
        assert sender.degradation_count == (1 if on_error == "degrade" else 0)
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("answer", [[], "ok"], ids=["list", "string"])
@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_a_handoff_verb_answer_that_is_not_a_json_object_raises_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, verb: str, answer: object
) -> None:
    """A body that parses as JSON but is not an object used to escape as an
    ``AttributeError`` on every surface, the MCP tools included."""
    try:
        giver, successor = _volume_pair(tmp_path, fast_cfg, "strict", verb)
        real_post = coherent_volume_module._coordinator_post

        def not_an_object(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            body = real_post(endpoint, path, payload, **kwargs)
            return answer if path == f"/handoff/{verb}" else body

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", not_an_object)
        with pytest.raises(CommitUnconfirmed, match="no outcome this client can classify"):
            _act(verb, giver, successor)
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("verb", ["transfer", "accept"])
def test_a_handoff_request_the_coordinator_refused_outright_stays_a_coherence_error_under_strict(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    """A 4xx is the coordinator's definite refusal: nothing landed, so it is
    not an unknown outcome and keeps strict mode's ``CoherenceError``."""
    try:
        giver, successor = _volume_pair(tmp_path, fast_cfg, "strict", verb)
        before = _status_handoff(giver, _PLAN)
        real_post = coherent_volume_module._coordinator_post

        def refuse(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == f"/handoff/{verb}":
                raise _http_error(400, {"error": "grants contains duplicate paths"})
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", refuse)
        with pytest.raises(CoherenceError) as raised:
            _act(verb, giver, successor)
        assert type(raised.value) is CoherenceError
        monkeypatch.undo()
        assert _status_handoff(giver, _PLAN) == before
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_a_handoff_request_the_coordinator_refused_outright_is_unconfirmed_under_degrade(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    """Degrade mode does not tell a refusal from a failure, so a 4xx warns and
    raises ``CommitUnconfirmed``, as ``transfer()`` documents, never a
    ``CoherenceError`` that reads as "nothing happened" and never a value."""
    try:
        giver, successor = _volume_pair(tmp_path, fast_cfg, "degrade", verb)
        real_post = coherent_volume_module._coordinator_post

        def refuse(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == f"/handoff/{verb}":
                raise _http_error(400, {"error": "grants contains duplicate paths"})
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", refuse)
        with pytest.warns(CoherenceDegradedWarning), pytest.raises(CommitUnconfirmed):
            _act(verb, giver, successor)
    finally:
        stop_coordinator(tmp_path)


# --- fail-closed edges --------------------------------------------------------


@pytest.mark.parametrize("on_error", ["strict", "degrade"])
@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_a_handoff_verb_on_a_volume_with_no_coordinator_fails_closed_in_both_modes(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    on_error: str, verb: str,
) -> None:
    """A volume that lost its coordinator endpoint cannot hand anything on or
    settle anything. Each verb raises ``CoherenceError`` naming the missing
    endpoint, in both modes, and sends nothing: without the check a
    degrade-mode volume posted to no endpoint and reported the verb as
    unconfirmed, "may have landed", when it never left the process."""
    try:
        giver, successor = _volume_pair(tmp_path, fast_cfg, on_error, verb)
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)
        giver._endpoint = None
        successor._endpoint = None

        with pytest.raises(CoherenceError, match="endpoint unavailable") as raised:
            _act(verb, giver, successor)

        assert type(raised.value) is CoherenceError
        assert [route for route, _body in sent if route.startswith("/handoff/")] == []
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("read_call", ["read", "read_with_version"])
@pytest.mark.parametrize("answer", ["strict_deny", "degraded"])
def test_a_denied_or_degraded_re_read_does_not_move_the_presented_incarnation(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    read_call: str, answer: str,
) -> None:
    """A read registers a claim only when the coordinator admitted it. A strict
    deny (here, of a fresh incarnation reading bytes edited out of band) and a
    watchdog-degraded answer, which still says ``status: fresh``, register
    nothing, so the transfer still presents the incarnation whose read holds
    the path. Recording the denied or degraded read would present an
    incarnation that holds nothing, refused as not held."""
    target = _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    on_error = "strict" if answer == "strict_deny" else "degrade"
    giver = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=fast_cfg)
    successor = CoherentVolume(tmp_path, managed=_MANAGED, on_error=on_error, config=fast_cfg)
    try:
        giver.read(_PLAN)
        reader = giver._incarnation
        giver.reacquire(_OTHER)  # a re-mint: later requests name a new incarnation
        assert giver._incarnation != reader, "precondition"
        if answer == "strict_deny":
            target.write_bytes(b"plan edited out of band")
        else:
            real_post = coherent_volume_module._coordinator_post

            def degraded(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
                if path == "/hooks/pre-read" and payload.get("path") == _PLAN:
                    return {"status": "fresh", "degraded": True}  # the watchdog's answer
                return real_post(endpoint, path, payload, **kwargs)

            monkeypatch.setattr(coherent_volume_module, "_coordinator_post", degraded)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CoherenceDegradedWarning)
            try:
                getattr(giver, read_call)(_PLAN)
            except CoherenceError:
                pass  # a split pair is refused; the claim question is unchanged
        monkeypatch.undo()
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert _presented(sent) == [{"path": _PLAN, "agent_id": reader}]
        assert result.ok
    finally:
        stop_coordinator(tmp_path)


def test_an_accept_of_a_live_handoff_a_bystander_overtook_names_the_bystander(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A bystander's acquire of the path without a write overtakes the handoff
    and leaves it live. The successor's accept is taken but changes nothing:
    it answers ``overtaken``, naming the bystander as ``counterparty``."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    bystander = str(uuid4())
    try:
        giver.read(_PLAN)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        acquired = _cc_post(
            resolve_endpoint(tmp_path), "/hooks/pre-edit", {"session_id": bystander, "path": _PLAN}
        )
        assert acquired.get("ok") is True, "precondition: the bystander holds the path"

        assert successor.accept(_PLAN) == HandoffVerbResult(
            path=_PLAN, ok=True, status="overtaken",
            counterparty=str(session_to_agent_id(bystander)),
        )
    finally:
        stop_coordinator(tmp_path)


def test_the_latest_of_two_stranded_write_grants_is_the_one_presented(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two incarnations stranded by unconfirmed releases both record the path,
    but the later one's write took the grant from the earlier one. The
    transfer presents the later writer; the earlier holds nothing and would
    be refused as not held."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        real_post = coherent_volume_module._coordinator_post

        def refuse_releases(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/session-stop":
                return {"ok": False, "reason": "internal: RuntimeError"}  # not released
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", refuse_releases)
        giver.write(_PLAN, b"plan v2")
        first = giver._incarnation
        giver.reacquire(_OTHER)
        giver.write(_PLAN, b"plan v3")
        second = giver._incarnation
        giver.reacquire(_OTHER)
        assert len({first, second, giver._incarnation}) == 3, "precondition"
        monkeypatch.undo()
        sent: list[tuple[str, dict]] = []
        _record_posts(monkeypatch, sent)

        result = giver.transfer(_PLAN, successor=_agent(successor))

        assert _presented(sent) == [{"path": _PLAN, "agent_id": second}]
        assert result.ok
    finally:
        stop_coordinator(tmp_path)


def test_a_single_file_publish_of_a_handed_path_raises_the_giver_terminal(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A one-member publish takes the compare-and-swap route, not the batch
    session, and is fenced the same way: the giver terminal naming the
    successor and the version at transfer, and nothing written."""
    target = _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        assert giver.read_with_version(_PLAN) == (b"plan v1", 1)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok

        with pytest.raises(GiverFenced) as raised:
            giver.atomic_publish([(_PLAN, 1, b"late plan")])

        assert (raised.value.successor, raised.value.version_at_transfer) == (_agent(successor), 1)
        assert target.read_bytes() == b"plan v1"
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("verb", ["transfer", "withdraw"])
def test_a_handoff_verb_whose_principal_retry_loses_its_answer_raises_unconfirmed(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, verb: str
) -> None:
    """The one retry after a caller-principal refusal is the same verb sent
    again, so a lost answer to it is just as unknown: strict mode raises
    ``CommitUnconfirmed``, not the generic error, and the verb has landed."""
    try:
        giver, successor = _volume_pair(tmp_path, fast_cfg, "strict", verb)
        real_post = coherent_volume_module._coordinator_post
        sends: list[str] = []

        def refuse_then_lose(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path != f"/handoff/{verb}":
                return real_post(endpoint, path, payload, **kwargs)
            sends.append(path)
            real_post(endpoint, path, payload, **kwargs)  # the first send is refused here
            raise CoordinatorUnavailable("simulated: the retry's answer was lost")

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", refuse_then_lose)
        giver._principal = "X" * 43  # a principal the coordinator does not hold for this session

        with pytest.raises(CommitUnconfirmed) as raised:
            _act(verb, giver, successor)

        assert type(raised.value) is CommitUnconfirmed
        assert len(sends) == 2, "refused once for its principal, then retried once"
        monkeypatch.undo()
        assert (_status_handoff(giver, _PLAN) or {}).get("status") == _LANDED_STATUS[verb]
    finally:
        stop_coordinator(tmp_path)


# --- the giver exits --------------------------------------------------------


def test_a_restarted_writer_cannot_hand_on_a_live_handoff_until_its_write_overtakes_it(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Giver exit. Volume A hands plan.md to B and is discarded; its session
    id and principal lived only in its process, so nothing can act as A
    again. A second volume, a new session standing in for the restarted
    writer, reads the path: its transfer is refused as another session's
    handoff in flight, naming the pending pair and what ends it; its
    compare-and-swap at the transfer version wins and reports overtaking the
    pair; its transfer is then admitted and replaces the record."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor, restarted = _volumes(tmp_path, fast_cfg, 3)
    try:
        giver.read(_PLAN)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        giver_id = _agent(giver)
        del giver

        restarted.read(_PLAN)
        refused = restarted.transfer(_PLAN, successor=_agent(successor))
        assert refused == HandoffTransferResult(grants=(HandoffGrantResult(
            path=_PLAN, transferred=False, reason="handoff_in_flight",
            giver=giver_id, successor=_agent(successor), detail=_IN_FLIGHT_DETAIL,
        ),))

        won = restarted.write_cas_at(_PLAN, 1, b"plan v2 by the restarted writer")
        assert won == CasCommitResult(path=_PLAN, version=2, handoff=HandoffWinOutcome(
            outcome="overtaken", giver=giver_id, successor=_agent(successor),
            version_at_transfer=1, counterparty=_agent(restarted),
        ))

        admitted = restarted.transfer(_PLAN, successor=_agent(successor))
        assert admitted.grants == (HandoffGrantResult(
            path=_PLAN, transferred=True, giver=_agent(restarted), successor=_agent(successor),
            version_at_transfer=2, hold_shape="SHARED", status="pending",
        ),)
        assert _status_handoff(restarted, _PLAN) == {
            "giver": _agent(restarted), "successor": _agent(successor),
            "version_at_transfer": 2, "hold_shape": "SHARED", "status": "pending", "live": True,
        }
    finally:
        stop_coordinator(tmp_path)


# --- the handoff key a read received ----------------------------------------


def test_read_handoff_reports_the_key_the_latest_read_of_the_path_received(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The read calls return bytes and versions (public contracts), not the
    ``handoff`` key their answers carry, so a caller that must relay a read's
    provenance (the MCP read tool) asks ``read_handoff``: the key the answer to
    this volume's latest read of the path carried, from either read call, or
    ``None``. The successor's read reports the pending record projected for
    it. Once a bystander's win moved the version, the successor's next read is
    strict-denied, which carries no key, so it reports ``None`` rather than the
    earlier answer; the bystander's own read reports the record it overtook. A
    path never read reports ``None``, and so does a forked child, a new
    session that read nothing."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor, bystander = _volumes(tmp_path, fast_cfg, 3)
    try:
        giver.read(_PLAN)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        never_read = successor.read_handoff(_PLAN)

        successor.read_with_version(_PLAN)
        pending = successor.read_handoff(_PLAN)
        bystander.write_cas_at(_PLAN, 1, b"plan v2 by a bystander")
        successor.read(_PLAN)
        after_denied_read = successor.read_handoff(_PLAN)
        bystander.read(_PLAN)
        overtaken = bystander.read_handoff(_PLAN)
        bystander_id = _agent(bystander)  # before the fork re-mints the session
        bystander._after_fork()  # simulate the child-side fork handler

        assert never_read is None
        assert pending == {
            "role": "successor", "giver": _agent(giver), "successor": _agent(successor),
            "version_at_transfer": 1, "hold_shape": "SHARED", "status": "pending", "live": True,
        }
        assert after_denied_read is None
        assert overtaken == {
            "role": "bystander", "giver": _agent(giver), "successor": _agent(successor),
            "version_at_transfer": 1, "hold_shape": "SHARED", "status": "overtaken",
            "live": False, "counterparty": bystander_id,
        }
        assert bystander.read_handoff(_PLAN) is None
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("read_call", ["read", "read_with_version", "read_with_version_generation"])
def test_a_strict_denied_re_read_sets_last_read_denied_whichever_read_method_took_it(
    tmp_path: Path, fast_cfg: LifecycleConfig, read_call: str
) -> None:
    """The guide sends a library giver to ``read_handoff()`` and
    ``last_read_denied`` after its re-read of a path it handed off. The re-read
    is strict-denied and carries no ``handoff`` key, so ``read_handoff()`` is
    ``None``, and only the flag says that ``None`` means "not told" rather than
    "no record". A flag set only by ``read_with_version_generation`` told a
    giver whose ``read()`` or ``read_with_version()`` was denied that the read
    was admitted."""
    _seed(tmp_path, _PLAN, b"plan v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok

        getattr(giver, read_call)(_PLAN)

        assert giver.last_read_denied is True
        assert giver.read_handoff(_PLAN) is None
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("read_call", ["read", "read_with_version", "reacquire"])
def test_an_admitted_read_clears_last_read_denied_whichever_read_method_took_it(
    tmp_path: Path, fast_cfg: LifecycleConfig, read_call: str
) -> None:
    """The flag answers for the volume's LATEST read: after a denied re-read of
    the handed-off path, an admitted read of another path by any read method
    clears it. A flag left set would mark that admitted read's missing
    ``handoff`` key as "not told" when the path simply has no record."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver, successor = _volumes(tmp_path, fast_cfg, 2)
    try:
        giver.read(_PLAN)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        giver.read_with_version_generation(_PLAN)
        assert giver.last_read_denied is True, "precondition: the re-read was strict-denied"

        getattr(giver, read_call)(_OTHER)

        assert giver.last_read_denied is False
    finally:
        stop_coordinator(tmp_path)


def test_a_read_that_fails_after_a_denied_one_clears_last_read_denied(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flag is reset before each read's request, beside the handoff key, so
    a read whose answer is lost leaves it ``False``: the denied answer it would
    otherwise keep belongs to an earlier read, not to this one."""
    _seed(tmp_path, _PLAN, b"plan v1")
    _seed(tmp_path, _OTHER, b"other v1")
    giver = CoherentVolume(tmp_path, managed=_MANAGED, on_error="degrade", config=fast_cfg)
    successor = CoherentVolume(tmp_path, managed=_MANAGED, on_error="degrade", config=fast_cfg)
    try:
        giver.read(_PLAN)
        assert giver.transfer(_PLAN, successor=_agent(successor)).ok
        giver.read(_PLAN)
        assert giver.last_read_denied is True, "precondition: the re-read was strict-denied"
        real_post = coherent_volume_module._coordinator_post

        def lose_the_read(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == "/hooks/pre-read":
                raise CoordinatorUnavailable("simulated: the answer was lost")
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", lose_the_read)
        with pytest.warns(CoherenceDegradedWarning):
            giver.read(_OTHER)

        assert giver.last_read_denied is False
    finally:
        stop_coordinator(tmp_path)


# --- public names -----------------------------------------------------------


def test_the_handoff_results_are_exported_through_the_lazy_map() -> None:
    """The typed results are public API: importable from ``ccs.adapters`` by
    name, resolved through the package's lazy export map like every other
    adapter export, and the very classes the volume returns."""
    expected = {
        "CasCommitResult": CasCommitResult,
        "HandoffGrantResult": HandoffGrantResult,
        "HandoffTransferResult": HandoffTransferResult,
        "HandoffVerbResult": HandoffVerbResult,
        "HandoffWinOutcome": HandoffWinOutcome,
    }
    for name, cls in expected.items():
        assert name in adapters.__all__
        assert adapters._EXPORTS[name] == (".coherent_volume", name)
        assert getattr(adapters, name) is cls
        assert name in coherent_volume_module.__all__


def test_the_result_types_are_frozen() -> None:
    """A typed result is a value: a caller holding one cannot change what the
    coordinator answered."""
    grant = HandoffGrantResult(path=_PLAN, transferred=False, reason="handoff_not_held")
    with pytest.raises(AttributeError):
        grant.transferred = True  # type: ignore[misc]

