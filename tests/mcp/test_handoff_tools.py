"""The MCP handoff tools (#185) against a real loopback coordinator.

An MCP session hands its own claim on a path to another session with
``swg_transfer``; the successor takes it with ``swg_accept`` (or a write) or
refuses it with ``swg_decline``; the giver withdraws with ``swg_withdraw`` only
on its user's or host's instruction. After its own transfer the giver's writes
on the path are refused with the typed ``handed_off`` terminal, which the deny
mapper must answer as a stop: read as a stale view it would send the giver to
reacquire, and no reacquire clears a fence keyed on the session. The successor
names itself by the session-level agent id its own ``swg_status`` reports.

Tests drive the sync ``_do_*`` helpers (the tool logic) with real
``CoherentVolume`` instances, as the other tool modules do. The first volume
spawns the coordinator; every other volume sibling-attaches, so each volume is
its own session with a bound caller principal.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pytest

import ccs.adapters.coherent_volume as coherent_volume_module
from ccs.adapters.claude_code.coordinator_server import session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable, resolve_endpoint
from ccs.cli._coherence_client import post as _cc_post
from ccs.mcp.server import (
    _do_accept,
    _do_decline,
    _do_reacquire,
    _do_read,
    _do_status,
    _do_transfer,
    _do_withdraw,
    _do_write,
    _do_write_cas,
)
from ccs.mcp.session import SessionConfig
from ccs.mcp.status import build_status, handoff_from_status

PLAN = "data/plan.md"
OTHER = "data/other.md"

#: FROZEN duplicates (never imported from the code under test): the giver
#: terminal's wire reason and the recover verb the MCP surface gives it.
_GIVER_REASON = "handed_off"
_GIVER_RECOVER = "stop_and_report"
#: FROZEN duplicate: the wire reason of an answer that does not settle an outcome.
_UNCONFIRMED_REASON = "commit_unconfirmed"
#: FROZEN duplicate: the recover verb an unconfirmed handoff tool answers. Not
#: the generic ``read_then_retry``: reading the path and transferring again is
#: the recipe that hands it on a second time once the successor has written it.
_HANDOFF_UNCONFIRMED_RECOVER = "check_handoff"
#: FROZEN duplicates: what each verb's unconfirmed next step must say "landed"
#: looks like. A transfer's names the version, so an earlier round's ended
#: record between the same sessions is not read as this transfer.
_LANDED_MARKER = {
    "transfer": "version_at_transfer",
    "accept": "Status completed",
    "decline": "Status declined",
    "withdraw": "Status withdrawn",
}
#: FROZEN duplicates (never read off the code under test): the recover verb a
#: refused handoff tool answers, per reason. None is retryable: nothing changed,
#: and the same call gets the same answer.
_REFUSAL_RECOVER = {
    "handoff_to_self": "fix_successor",
    "handoff_successor_unknown": "fix_successor",
    "handoff_successor_malformed": "fix_successor",
    "handoff_not_held": "check_handoff",
    "handoff_version_unconfirmed": "stop_and_report",
    "handoff_in_flight": "stop_and_report",
    "handoff_other_holder": "stop_and_report",
    "handoff_ended": "stop_and_report",
    "handoff_not_successor": "stop_and_report",
    "handoff_not_giver": "stop_and_report",
    "handoff_not_live": "stop_and_report",
}


def _refused(structured: dict) -> dict:
    """A refused result or grant with its fixed ``next_step`` checked and
    dropped (what each row's text must say is pinned per reason in
    ``tests/mcp/test_deny_mapping.py``)."""
    shape = dict(structured)
    next_step = shape.pop("next_step")
    assert isinstance(next_step, str) and next_step.startswith("Nothing"), next_step
    return shape


#: The recover verbs that would send a fenced giver back to try again: none of
#: them can clear a fence keyed on the session.
_LOOPING_RECOVERS = ("reacquire", "read_then_merge", "reacquire_and_reread", "wait_and_retry")


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
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


def _vol(tmp_path: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=cfg)


def _config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(root=tmp_path.resolve(), managed=("data/**",))


def _agent(vol: CoherentVolume) -> str:
    """A volume's session-level agent id, derived here the way the coordinator
    derives it (from the session id alone), never read off the code under
    test."""
    return str(session_to_agent_id(vol.session_id))


# --- the giver's own write after its own transfer is a stop ----------------


@pytest.mark.parametrize("route", ["swg_write", "swg_write_cas"])
def test_a_givers_write_after_its_own_transfer_is_a_stop_a_reacquire_does_not_change(
    tmp_path: Path, fast_cfg: LifecycleConfig, route: str
) -> None:
    """The MCP session reads plan.md, hands it to another
    session, then writes it. Without a row of its own the giver's refusal read
    as an internal error (compare-and-swap) or, on an older client, as a stale
    view to reacquire, and no reacquire clears a fence keyed on the session,
    so a cooperating giver looped. The write answers the typed giver terminal:
    not retryable, recover ``stop_and_report``, naming the successor and the
    version at transfer. A reacquire followed by the same write answers the
    same bytes, and nothing reached disk."""
    target = _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    conflicts: dict[str, int] = {}

    def late_write():
        if route == "swg_write":
            return _do_write(giver, config, PLAN, "late giver bytes")
        return _do_write_cas(giver, config, conflicts, PLAN, 1, "late giver bytes")

    try:
        assert _do_read(giver, config, PLAN).isError is False
        assert _do_transfer(giver, config, [PLAN], _agent(successor)).isError is False

        first = late_write()

        body = first.structuredContent
        assert first.isError is True
        assert body["reason"] == _GIVER_REASON
        assert body["retryable"] is False
        assert body["recover"] == _GIVER_RECOVER
        assert body["recover"] not in _LOOPING_RECOVERS
        assert body["successor"] == _agent(successor)
        assert body["version_at_transfer"] == 1

        assert _do_reacquire(giver, config, PLAN).isError is False
        second = late_write()

        assert second.model_dump() == first.model_dump()
        assert target.read_bytes() == b"plan v1"
        assert conflicts == {}, "a giver refusal is not a CAS conflict to count"
    finally:
        stop_coordinator(tmp_path)


# --- a path the MCP session does not hold is refused, never moved ----------


def test_the_transfer_tool_on_a_path_a_hook_session_holds_is_refused_as_not_held(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A hook session holds data/y.md EXCLUSIVE and the MCP session
    holds nothing on it. The transfer tool answers that grant refused as
    ``handoff_not_held``, as a non-ignorable error, records no handoff, and
    leaves the hook session's grant where it was: it never silently
    succeeds."""
    rel = "data/y.md"
    target = _seed(tmp_path, rel, b"y v1")
    config = _config(tmp_path)
    mcp_session = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    hook_session = str(uuid4())
    try:
        acquired = _cc_post(
            resolve_endpoint(tmp_path), "/hooks/pre-edit", {"session_id": hook_session, "path": rel}
        )
        assert acquired.get("ok") is True, f"precondition: the hook session holds {rel}"

        result = _do_transfer(mcp_session, config, [rel], _agent(successor))

        assert result.isError is True
        assert result.structuredContent["ok"] is False
        [grant] = result.structuredContent["grants"]
        assert _refused(grant) == {
            "path": rel, "transferred": False, "reason": "handoff_not_held",
            "recover": _REFUSAL_RECOVER["handoff_not_held"], "retryable": False,
        }
        status = mcp_session.coordinator_status()
        assert status is not None
        [entry] = [e for e in status["tracked_artifacts"] if e["path"] == rel]
        assert "handoff" not in entry
        [hook_row] = [
            s for s in status["sessions"] if s["agent_id"] == str(session_to_agent_id(hook_session))
        ]
        assert hook_row["states"] == {rel: "EXCLUSIVE"}
        assert target.read_bytes() == b"y v1"
    finally:
        stop_coordinator(tmp_path)


# --- provenance on a read, the win outcome on a compare-and-swap -----------


def test_the_successors_read_carries_the_handoff_key_and_a_path_with_no_record_none(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The successor's read of a path handed to it carries the record
    projected for it -- who handed it on, at which version, from which hold
    shape -- in its structured content. A read of a path with no record
    carries no ``handoff`` key at all, so its result is what it was before."""
    _seed(tmp_path, PLAN, b"plan v1")
    _seed(tmp_path, OTHER, b"other v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        assert _do_transfer(giver, config, [PLAN], _agent(successor)).isError is False

        read = _do_read(successor, config, PLAN)
        plain = _do_read(successor, config, OTHER)

        assert read.isError is False
        assert read.structuredContent["content"] == "plan v1"
        assert read.structuredContent["handoff"] == {
            "role": "successor",
            "giver": _agent(giver),
            "successor": _agent(successor),
            "version_at_transfer": 1,
            "hold_shape": "SHARED",
            "status": "pending",
            "live": True,
        }
        assert plain.isError is False
        assert "handoff" not in plain.structuredContent
    finally:
        stop_coordinator(tmp_path)


def test_the_givers_strict_denied_re_read_still_carries_its_handoff_key(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """On an enforced path the giver's own re-read after its transfer is the
    coordinator's strict deny, whose bytes never carry the ``handoff`` key.
    The read tool still returns the bytes, and without the key a giver reads
    the path as having no record -- the silent all-clear the hook surface's
    giver read notice exists to prevent. The record comes from ``/status``
    instead, projected with the giver's role."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        assert _do_transfer(giver, config, [PLAN], _agent(successor)).isError is False

        reread = _do_read(giver, config, PLAN)

        assert giver.last_read_denied is True, "precondition: the re-read was strict-denied"
        assert reread.isError is False
        assert reread.structuredContent["content"] == "plan v1"
        assert reread.structuredContent["handoff"] == {
            "role": "giver",
            "giver": _agent(giver),
            "successor": _agent(successor),
            "version_at_transfer": 1,
            "hold_shape": "SHARED",
            "status": "pending",
            "live": True,
        }
        assert "handoff_unknown" not in reread.structuredContent
    finally:
        stop_coordinator(tmp_path)


def test_a_givers_denied_re_read_through_a_symlink_takes_the_record_from_status(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The coordinator knows the file by its resolved name, and ``/status``
    lists it under that name. A giver that read, handed off and re-read the
    file through an in-root symlink gets a denied re-read, so the read tool
    asks ``/status``. Looking the record up under the link's name found no
    entry and answered "no record" for a live handoff."""
    _seed(tmp_path, PLAN, b"plan v1")
    (tmp_path / "data" / "plan-link.md").symlink_to("plan.md")
    link = "data/plan-link.md"
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, link)
        assert _do_transfer(giver, config, [link], _agent(successor)).isError is False

        reread = _do_read(giver, config, link)

        assert giver.last_read_denied is True, "precondition: the re-read was strict-denied"
        assert reread.structuredContent["handoff"]["role"] == "giver"
        assert reread.structuredContent["handoff"]["status"] == "pending"
    finally:
        stop_coordinator(tmp_path)


def test_a_denied_read_whose_record_cannot_be_fetched_says_so(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When ``/status`` cannot be read after a denied read, the record is
    unknown, not absent: the result says ``handoff_unknown`` rather than
    omitting the key the way a path with no record does."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        assert _do_transfer(giver, config, [PLAN], _agent(successor)).isError is False
        monkeypatch.setattr(giver, "coordinator_status", lambda: None)

        reread = _do_read(giver, config, PLAN)

        assert giver.last_read_denied is True, "precondition: the re-read was strict-denied"
        assert reread.isError is False
        assert reread.structuredContent["handoff_unknown"] is True
        assert "handoff" not in reread.structuredContent
    finally:
        stop_coordinator(tmp_path)


def test_a_cas_win_carries_what_it_did_to_the_handoff(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The successor's compare-and-swap win completes the handoff and its
    result says so; a bystander's win overtakes a live handoff and its result
    names the pair it wrote past and itself as the counterparty. A win that
    labelled no live handoff carries no ``handoff`` key."""
    _seed(tmp_path, PLAN, b"plan v1")
    _seed(tmp_path, OTHER, b"other v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    bystander = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        _do_read(giver, config, OTHER)
        assert _do_transfer(giver, config, [PLAN, OTHER], _agent(successor)).isError is False

        completed = _do_write_cas(successor, config, {}, PLAN, 1, "plan v2 by the successor")
        overtaken = _do_write_cas(bystander, config, {}, OTHER, 1, "other v2 by a bystander")
        plain = _do_write_cas(bystander, config, {}, PLAN, 2, "plan v3 by a bystander")

        assert completed.isError is False
        assert completed.structuredContent["handoff"] == {
            "outcome": "completed",
            "giver": _agent(giver),
            "successor": _agent(successor),
            "version_at_transfer": 1,
        }
        assert overtaken.isError is False
        assert overtaken.structuredContent["handoff"] == {
            "outcome": "overtaken",
            "giver": _agent(giver),
            "successor": _agent(successor),
            "version_at_transfer": 1,
            "counterparty": _agent(bystander),
        }
        assert plain.isError is False
        assert "handoff" not in plain.structuredContent
    finally:
        stop_coordinator(tmp_path)


# --- the four verbs for the MCP session's own claims -----------------------


def test_the_transfer_tool_answers_per_grant_and_only_all_transferred_is_success(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A transfer of a path the session read and one it never touched
    answers per grant, in order: the first transferred with its record, the
    second refused as not held. The result is an error unless every grant
    transferred, so a partly refused transfer is never read as done."""
    _seed(tmp_path, PLAN, b"plan v1")
    _seed(tmp_path, OTHER, b"other v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)

        result = _do_transfer(giver, config, [PLAN, OTHER], _agent(successor))

        assert result.isError is True
        structured = dict(result.structuredContent)
        transferred, refused = structured.pop("grants")
        assert transferred == {
            "path": PLAN,
            "transferred": True,
            "giver": _agent(giver),
            "successor": _agent(successor),
            "version_at_transfer": 1,
            "hold_shape": "SHARED",
            "status": "pending",
        }
        assert _refused(refused) == {
            "path": OTHER, "transferred": False, "reason": "handoff_not_held",
            "recover": "check_handoff", "retryable": False,
        }
        # The top level speaks for the refused grant, and its next step first
        # says the transferred path must never be sent again.
        assert structured["next_step"].startswith("Not every path was handed off.")
        assert structured["next_step"].endswith(refused["next_step"])
        del structured["next_step"]
        assert structured == {
            "ok": False, "reason": "handoff_not_held", "recover": "check_handoff",
            "retryable": False,
            "detail": f"{PLAN}: transferred to {_agent(successor)} at v1 (pending)\n"
            f"{OTHER}: not transferred (handoff_not_held)",
        }
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize(
    ("paths", "detail"),
    [
        ([], "paths must name at least one path, each once"),
        ([PLAN, "./" + PLAN], "paths must name at least one path, each once"),
        ([PLAN, "data/plan-link.md"], "transfer names a path more than once: data/plan.md"),
    ],
    ids=["no-path", "one-path-twice", "a-path-and-its-symlink"],
)
def test_the_transfer_tool_answers_no_path_or_one_path_twice_as_a_path_error(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    paths: list[str], detail: str,
) -> None:
    """An empty path list, or one path named twice in two spellings, is the
    agent's own input mistake. It reached the coordinator, which answered
    HTTP 400, and the tool answered ``internal_error`` with ``recover: none``,
    an infrastructure fault the agent stops or escalates on. It is answered
    like every other bad input, ``invalid_path`` / ``fix_path``, and no
    transfer is sent. An in-root symlink to the path is a third spelling: the
    tool's own check sees two names, and the volume, which resolves the link,
    refuses the pair."""
    _seed(tmp_path, PLAN, b"plan v1")
    (tmp_path / "data" / "plan-link.md").symlink_to("plan.md")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        sent: list[str] = []
        real_post = coherent_volume_module._coordinator_post

        def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            sent.append(path)
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)

        result = _do_transfer(giver, config, paths, _agent(successor))

        assert result.isError is True
        structured = result.structuredContent
        assert (structured["reason"], structured["recover"], structured["retryable"]) == (
            "invalid_path", "fix_path", False,
        )
        assert structured["detail"] == detail  # the tool's own check, or the volume's
        assert "/handoff/transfer" not in sent
    finally:
        stop_coordinator(tmp_path)


def test_an_accept_after_a_bystanders_write_is_refused_not_live_as_the_description_says(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A bystander's write moves the version, which ends the handoff, so the
    successor's accept is refused ``handoff_not_live`` with status
    ``overtaken``, as swg_accept's description says. An acquire with no write
    since leaves it live, and that accept is taken; the description used to
    promise that answer after a write too."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    bystander = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        assert _do_transfer(giver, config, [PLAN], _agent(successor)).isError is False
        bystander.write_cas_at(PLAN, 1, b"plan v2 by a bystander")

        result = _do_accept(successor, config, PLAN)

        assert result.isError is True
        structured = result.structuredContent
        assert (structured["reason"], structured["status"]) == ("handoff_not_live", "overtaken")
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_a_handoff_tool_on_a_session_whose_coordinator_is_gone_answers_unavailable(
    tmp_path: Path, fast_cfg: LifecycleConfig, verb: str
) -> None:
    """With the coordinator endpoint lost, every handoff tool answers
    ``coordinator_unavailable``, the answer an agent retries later, rather than
    ``internal_error`` from a volume that cannot reach anyone."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    volume = _vol(tmp_path, fast_cfg)
    try:
        volume._endpoint = None
        if verb == "transfer":
            result = _do_transfer(volume, config, [PLAN], str(uuid4()))
        else:
            result = {"accept": _do_accept, "decline": _do_decline, "withdraw": _do_withdraw}[verb](
                volume, config, PLAN
            )

        assert result.isError is True
        assert result.structuredContent["reason"] == "coordinator_unavailable"
    finally:
        stop_coordinator(tmp_path)


def test_a_successors_read_after_a_bystander_overtook_reports_the_overtaken_record(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The successor's first read carries the pending record. A bystander's
    win then overtakes it and moves the version, so the successor's next read
    is strict-denied and carries no key: the read tool must take the record
    from ``/status`` and report it overtaken, not relay the earlier read's
    pending key as if nothing had happened."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    bystander = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        assert _do_transfer(giver, config, [PLAN], _agent(successor)).isError is False
        first = _do_read(successor, config, PLAN)
        assert first.structuredContent["handoff"]["status"] == "pending", "precondition"
        bystander.write_cas_at(PLAN, 1, b"plan v2 by a bystander")

        second = _do_read(successor, config, PLAN)

        assert successor.last_read_denied is True, "precondition: the re-read was strict-denied"
        assert second.structuredContent["handoff"]["status"] == "overtaken"
        assert second.structuredContent["handoff"]["counterparty"] == _agent(bystander)
    finally:
        stop_coordinator(tmp_path)


@pytest.mark.parametrize(
    ("verb", "tool"), [("accept", _do_accept), ("decline", _do_decline), ("withdraw", _do_withdraw)]
)
def test_a_verb_tool_whose_answer_settles_nothing_is_the_unconfirmed_deny(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch,
    verb: str, tool: object,
) -> None:
    """An answer the volume cannot classify (here the coordinator's failure
    envelope) reaches the agent as the unconfirmed deny, which sends it to the
    path's handoff before acting again -- never as "refused ...; nothing
    changed", which would be false if the verb landed."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    volume = _vol(tmp_path, fast_cfg)
    try:
        _do_read(volume, config, PLAN)
        real_post = coherent_volume_module._coordinator_post

        def failure_envelope(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
            if path == f"/handoff/{verb}":
                return {"ok": False, "reason": "internal: RuntimeError"}
            return real_post(endpoint, path, payload, **kwargs)

        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", failure_envelope)
        result = tool(volume, config, PLAN)  # type: ignore[operator]

        assert result.isError is True
        assert result.structuredContent["reason"] == _UNCONFIRMED_REASON
        assert result.structuredContent["recover"] == _HANDOFF_UNCONFIRMED_RECOVER
        assert result.structuredContent["retryable"] is False
    finally:
        stop_coordinator(tmp_path)


def test_the_accept_decline_and_withdraw_tools_answer_typed_results(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A party's verb is taken and answers the record's status after it; a
    verb from the wrong party or with no live record is refused with its typed
    reason, as a non-ignorable error, and changes nothing."""
    _seed(tmp_path, PLAN, b"plan v1")
    _seed(tmp_path, OTHER, b"other v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        _do_read(giver, config, OTHER)
        no_record = _do_accept(successor, config, OTHER)
        assert _do_transfer(giver, config, [PLAN, OTHER], _agent(successor)).isError is False

        not_successor = _do_accept(giver, config, PLAN)
        not_giver = _do_withdraw(successor, config, PLAN)
        accepted = _do_accept(successor, config, PLAN)
        withdrawn = _do_withdraw(giver, config, PLAN)
        not_live = _do_decline(successor, config, PLAN)
        declined = _do_decline(successor, config, OTHER)

        for result, verb, path, reason, status in (
            (no_record, "accept", OTHER, "handoff_not_live", None),
            (not_successor, "accept", PLAN, "handoff_not_successor", "pending"),
            (not_giver, "withdraw", PLAN, "handoff_not_giver", "pending"),
            (not_live, "decline", PLAN, "handoff_not_live", "withdrawn"),
        ):
            assert result.isError is True
            expected = {
                "path": path, "ok": False, "reason": reason,
                "recover": _REFUSAL_RECOVER[reason], "retryable": False,
                "detail": f"{verb} {path}: refused ({reason}); nothing changed",
            }
            if status is not None:
                expected["status"] = status
            assert _refused(result.structuredContent) == expected
        assert accepted.isError is False
        assert accepted.structuredContent == {"path": PLAN, "ok": True, "status": "completed"}
        assert withdrawn.isError is False
        assert withdrawn.structuredContent == {"path": PLAN, "ok": True, "status": "withdrawn"}
        assert declined.isError is False
        assert declined.structuredContent == {"path": OTHER, "ok": True, "status": "declined"}
    finally:
        stop_coordinator(tmp_path)


# --- the status tool -------------------------------------------------------


def test_the_status_tool_shows_a_paths_record_and_omits_the_key_where_there_is_none(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The status tool renders the record from the coordinator's default
    ``/status`` tier on the handed-off path's entry, and an entry with no
    record keeps exactly the shape it had."""
    _seed(tmp_path, PLAN, b"plan v1")
    _seed(tmp_path, OTHER, b"other v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        _do_read(giver, config, PLAN)
        _do_read(giver, config, OTHER)
        assert _do_transfer(giver, config, [PLAN], _agent(successor)).isError is False

        per_path = _do_status(giver, config).structuredContent["per_path"]

        assert per_path[PLAN] == {
            "version": 1,
            "status": "enforced",
            "handoff": {
                "giver": _agent(giver),
                "successor": _agent(successor),
                "version_at_transfer": 1,
                "hold_shape": "SHARED",
                "status": "pending",
                "live": True,
            },
        }
        assert per_path[OTHER] == {"version": 1, "status": "enforced"}
    finally:
        stop_coordinator(tmp_path)


def test_the_id_a_successors_status_tool_reports_is_the_successor_a_transfer_names(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """Session B reads its own id off its status tool and session A
    passes that value to its transfer tool: the grant transfers, and the
    successor the coordinator answers (normalised, hyphenated lower-case) is
    that value. The id is session-level, so B re-minting its incarnation
    does not change it -- a giver holding it still names B."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    giver = _vol(tmp_path, fast_cfg)
    successor = _vol(tmp_path, fast_cfg)
    try:
        successor_id = _do_status(successor, config).structuredContent["session_agent_id"]
        _do_read(giver, config, PLAN)

        moved = _do_transfer(giver, config, [PLAN], successor_id)

        assert moved.isError is False
        [grant] = moved.structuredContent["grants"]
        assert grant["transferred"] is True
        assert grant["successor"] == successor_id

        incarnation = successor._incarnation
        assert _do_reacquire(successor, config, PLAN).isError is False
        assert successor._incarnation != incarnation, "precondition: the reacquire re-minted"
        assert _do_status(successor, config).structuredContent["session_agent_id"] == successor_id
    finally:
        stop_coordinator(tmp_path)


class _StubVolume:
    """Just what ``build_status`` reads, for a fixed session id."""

    is_attached = True
    is_degraded = False
    principal_claim_outcome = "bound"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    def coordinator_status(self) -> dict | None:
        return None


#: A fixed session id and the agent id the coordinator derives from it,
#: computed by hand as uuid5(NAMESPACE_URL, "ccs-agent:claude-session-" + id).
_FIXED_SESSION = "0f8fad5b-d9cb-469f-a165-70867728950e"
_FIXED_AGENT = "e9ecb9fa-4642-5d3e-9961-4ede23e20191"


def test_the_status_tools_id_is_the_coordinators_derivation_for_a_fixed_session() -> None:
    """The status tool's id is the session-level agent id the
    coordinator derives from the session id alone, pinned as a literal, so a
    drift in either derivation is caught here rather than as a successor the
    coordinator refuses as unknown."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))

    status = build_status(_StubVolume(_FIXED_SESSION), config)

    assert status["session_agent_id"] == _FIXED_AGENT


def test_the_status_tool_derives_its_id_through_the_coordinators_function(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never a second copy of the derivation string. The name the status
    module derives with IS the coordinator module's function, and the status
    value comes from calling it with the session id alone: a local copy of the
    string would agree today and drift silently the day the coordinator's
    changes."""
    import ccs.mcp.status as status_module
    from ccs.adapters.claude_code import coordinator_server

    assert status_module.session_to_agent_id is coordinator_server.session_to_agent_id
    calls: list[tuple[str, str | None]] = []

    def spy(session_id: str, subagent_id: str | None = None) -> UUID:
        calls.append((session_id, subagent_id))
        return UUID(int=7)

    monkeypatch.setattr(status_module, "session_to_agent_id", spy)
    config = SessionConfig(root=Path("/x"), managed=("data/**",))

    status = build_status(_StubVolume(_FIXED_SESSION), config)

    assert calls == [(_FIXED_SESSION, None)]
    assert status["session_agent_id"] == str(UUID(int=7))


class _StatusDocVolume:
    """Just what ``handoff_from_status`` reads: a session id, a fixed
    ``/status`` document (``None``: unreachable) and a root the path is
    resolved against (one with no files, so every path keeps its name)."""

    def __init__(self, session_id: str, status_doc: dict | None) -> None:
        self.session_id = session_id
        self._status_doc = status_doc
        self.root = Path("/nonexistent-volume-root")

    def coordinator_status(self) -> dict | None:
        return self._status_doc


_OTHER_AGENT = "11111111-2222-5333-8444-555555555555"


def _status_with(record: dict | None) -> dict:
    entry: dict = {"path": PLAN, "version": 3}
    if record is not None:
        entry["handoff"] = record
    return {"tracked_artifacts": [{"path": OTHER, "version": 1}, entry]}


@pytest.mark.parametrize(
    ("giver", "successor", "role"),
    [
        (_FIXED_AGENT, _OTHER_AGENT, "giver"),
        (_OTHER_AGENT, _FIXED_AGENT, "successor"),
        (_OTHER_AGENT, str(UUID(int=9)), "bystander"),
    ],
)
def test_a_status_record_is_projected_with_this_sessions_role(
    giver: str, successor: str, role: str
) -> None:
    """``/status`` carries the record with no role; the read tool adds the
    one the coordinator would give this session in a hook body, from the
    session-level agent id alone."""
    record = {
        "giver": giver,
        "successor": successor,
        "version_at_transfer": 3,
        "hold_shape": "MODIFIED",
        "status": "pending",
        "live": True,
    }
    volume = _StatusDocVolume(_FIXED_SESSION, _status_with(record))

    assert handoff_from_status(volume, PLAN) == (True, {"role": role, **record})


def test_a_status_lookup_tells_no_record_from_cannot_tell() -> None:
    """A path ``/status`` lists without a record, or does not list, has no
    record; an unreachable ``/status``, or one with no artifact list, cannot
    tell -- and cannot-tell never collapses into "no record"."""
    assert handoff_from_status(_StatusDocVolume(_FIXED_SESSION, _status_with(None)), PLAN) == (
        True,
        None,
    )
    assert handoff_from_status(
        _StatusDocVolume(_FIXED_SESSION, {"tracked_artifacts": []}), PLAN
    ) == (True, None)
    assert handoff_from_status(_StatusDocVolume(_FIXED_SESSION, None), PLAN) == (False, None)
    assert handoff_from_status(_StatusDocVolume(_FIXED_SESSION, {"detail": "minimal"}), PLAN) == (
        False,
        None,
    )


def _answer_lost_after_forwarding(route: str, answer: object) -> object:
    """``_coordinator_post`` that forwards ``route`` and then loses its answer
    (``answer`` None) or hands back ``answer`` in its place."""
    real_post = coherent_volume_module._coordinator_post

    def post(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        if path != route:
            return real_post(endpoint, path, payload, **kwargs)
        real_post(endpoint, path, payload, **kwargs)
        if answer is None:
            raise CoordinatorUnavailable("simulated: the answer was lost")
        return answer

    return post


@pytest.mark.parametrize("answer", [None, []], ids=["lost", "not_an_object"])
@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_a_handoff_tool_whose_answer_is_lost_is_the_check_handoff_deny(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch, verb: str, answer: object
) -> None:
    """#276: a lost answer used to reach the agent as ``internal_error`` (or,
    for a non-object answer, escape the tool as an ``AttributeError``) although
    the verb had landed. It is the unconfirmed deny, and its recover verb sends
    the agent to the path's handoff -- not to read and transfer again, which
    after the successor's write is a second handoff. The next step is the
    verb's own: "do not withdraw" belongs to a transfer only, and would drop a
    withdraw that never landed."""
    _seed(tmp_path, PLAN, b"plan v1")
    config = _config(tmp_path)
    volume = _vol(tmp_path, fast_cfg)
    peer = _vol(tmp_path, fast_cfg)
    try:
        _do_read(volume, config, PLAN)
        if verb != "transfer":
            assert volume.transfer(PLAN, successor=_agent(peer)).ok
        lose = _answer_lost_after_forwarding(f"/handoff/{verb}", answer)
        monkeypatch.setattr(coherent_volume_module, "_coordinator_post", lose)
        if verb == "transfer":
            result = _do_transfer(volume, config, [PLAN], _agent(peer))
        else:
            actor = peer if verb in ("accept", "decline") else volume
            tool = {"accept": _do_accept, "decline": _do_decline, "withdraw": _do_withdraw}[verb]
            result = tool(actor, config, PLAN)

        assert result.isError is True
        body = result.structuredContent
        assert body["reason"] == _UNCONFIRMED_REASON
        assert body["recover"] == _HANDOFF_UNCONFIRMED_RECOVER
        assert body["retryable"] is False
        assert _LANDED_MARKER[verb] in body["next_step"]
        assert ("do not withdraw" in body["next_step"]) is (verb == "transfer")
    finally:
        stop_coordinator(tmp_path)
