"""``swg_write``'s answers, each reached through the volume's own raise site.

Every row here comes from a real ``CoherentVolume.write`` driven through
``_do_write`` against a coordinator running in this process. A holder, an
out-of-band edit, a link or a held registry lock makes the volume raise; where
only a failing coordinator would, its answer is scripted at the client
boundary, the one place a test can put it. So a row cannot drift from the
raise site it answers for: each test records what the volume raised and checks
the answer's detail against it byte for byte.

What these do not prove: one machine and loopback only, and the Python
coordinator only.
"""

from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import pytest
from mcp.types import CallToolResult

import ccs.adapters.claude_code.coordinator_server as server_module
import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.cli._coherence_client as client_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import CoordinatorUnavailable
from ccs.core.exceptions import (
    CoherenceError,
    CommitUnconfirmed,
    CoordinatorFailed,
    CoordinatorUnreachable,
    LinkOutOfManaged,
    OtherHolder,
    StaleView,
)
from ccs.mcp.server import _do_reacquire, _do_read, _do_write
from ccs.mcp.session import SessionConfig
from tests.coordinator_helpers import serve_in_process

_MANAGED = ("data/**",)
_PATH = "data/shared.txt"
_PRE_EDIT = "/hooks/pre-edit"
_POST_EDIT = "/hooks/post-edit"
_SESSION_STOP = "/hooks/session-stop"
#: The answer a coordinator sends an opted-in grant request it cannot decide in time.
_UNAVAILABLE = json.loads((Path(__file__).parents[1] / "fixtures" / "wire" / "pre_edit_unavailable.json").read_text())

#: FROZEN: each library method a detail can name, and the tool a next step
#: names for it, since an agent cannot call the method.
_TOOL_FOR = {"release()": "swg_release", "reacquire()": "swg_reacquire", "write()": "swg_write"}
#: FROZEN: what no swg_write next step says: a library call the agent cannot
#: make, or a route around the guard a detail may point to.
_FORBIDDEN = ("release()", "reacquire()", "write()", "close()", "on_held", "build a new volume")
#: FROZEN: the words that open the sentence every row after which a write grant
#: may still stand ends with.
_RELEASE_WORDS = "If you stop here instead, call swg_release on the path first"


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    return LifecycleConfig(
        idle_shutdown_sec=0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


@pytest.fixture
def server(tmp_path: Path) -> Iterator[CoordinatorHTTPServer]:
    """A strict coordinator for ``data/**`` running in this process."""
    (tmp_path / "data").mkdir()
    (tmp_path / _PATH).write_bytes(b"seed")
    running = serve_in_process(tmp_path, "write-rows")
    yield running
    running.shutdown()


@dataclass
class _Wire:
    """Every POST the volume sends, and the answers a test puts in place of the
    coordinator's: a route maps to a body, or to a callable ``(forward,
    payload)`` that decides for itself."""

    sent: list[tuple[str, dict]] = field(default_factory=list)
    answers: dict[str, object] = field(default_factory=dict)

    def bodies(self, route: str) -> list[dict]:
        return [body for sent_route, body in self.sent if sent_route == route]


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    recorded = _Wire()
    real_post = client_module.post

    def spy(endpoint: object, path: str, payload: dict, **kwargs: object) -> object:
        recorded.sent.append((path, dict(payload)))
        scripted = recorded.answers.get(path)
        if callable(scripted):
            return scripted(lambda body=payload: real_post(endpoint, path, body, **kwargs), payload)
        if scripted is not None:
            return scripted
        return real_post(endpoint, path, payload, **kwargs)

    monkeypatch.setattr(client_module, "post", spy)
    monkeypatch.setattr(coherent_volume_module, "_coordinator_post", spy)
    return recorded


def _volume(root: Path, cfg: LifecycleConfig, on_held: str = "refuse") -> CoherentVolume:
    return CoherentVolume(root, managed=_MANAGED, on_error="strict", on_held=on_held, config=cfg)


def _config(root: Path, on_held: str = "refuse") -> SessionConfig:
    return SessionConfig(root=root.resolve(), managed=_MANAGED, on_held=on_held)


def _raises_of(volume: CoherentVolume, monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Record every error ``volume.write`` raises on its way to the deny mapper."""
    raised: list[BaseException] = []
    real_write = volume.write

    def write(path: str, data: bytes) -> object:
        try:
            return real_write(path, data)
        except BaseException as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(volume, "write", write)
    return raised


def _row(result: CallToolResult, raised: BaseException) -> dict:
    """The deny's structured content, once it holds what every swg_write row
    keeps: the volume's text verbatim as its detail and first text item, a
    next step as the second, no library call or route around the guard in
    that step, and the tool for each library method the detail names."""
    assert result.isError is True, result.structuredContent
    row = result.structuredContent
    assert row["detail"] == str(raised)
    assert [item.text for item in result.content] == [row["detail"], row["next_step"]]
    for forbidden in _FORBIDDEN:
        assert forbidden not in row["next_step"].lower(), forbidden
    for method, tool in _TOOL_FOR.items():
        if method in row["detail"]:
            assert tool in row["next_step"], (method, tool)
    return row


def _flags(row: dict) -> tuple[str, str, bool]:
    return row["reason"], row["recover"], row["retryable"]


def _http_status(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(_POST_EDIT, code, "scripted", {}, io.BytesIO(b"{}"))  # type: ignore[arg-type]


# --- another agent holds the path ---------------------------------------------


def test_a_write_another_refuse_mode_agent_holds_is_told_to_release_wait_then_reacquire(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One agent holds the path; the other's write is refused naming it by the
    refusal's spelling, 32 lowercase hex. A bare retry of the same bytes lands
    over the holder's commit once the path comes free, so the flag is false and the verb
    puts the wait before a reacquire. The next step gives the wait in order:
    let go of your own holds, poll swg_status, then reacquire, rebuild and
    write once, with its numbers, the two holders that can be the agent itself,
    and the report when the agent's own bound runs out.

    Prevents an agent that obeys the flag resending its bytes over the
    holder's commit, and two agents that each wait on the other's hold."""
    holder, writer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    raised = _raises_of(writer, monkeypatch)

    row = _row(_do_write(writer, _config(tmp_path), _PATH, "from-the-writer"), raised[-1])

    assert type(raised[-1]) is OtherHolder
    assert _flags(row) == ("other_holder", "wait_then_reacquire", False)
    assert row["holder_agent_id"] == UUID(holder.agent_id).hex
    assert (tmp_path / _PATH).read_bytes() == b"from-the-holder"
    words = row["next_step"]
    steps = ["swg_release on every path swg_status lists in held", "Poll swg_status",
             "call swg_reacquire on the path", "call swg_write once"]
    assert all(step in words for step in steps), steps
    assert [words.find(step) for step in steps] == sorted(words.find(step) for step in steps)
    for phrase in (
        "Nothing was written", "holder_agent_id", "earlier attempt", "release_unconfirmed",
        "at least 4 seconds", "double the wait", "jitter", "up to 30 minutes",
        "about 10 minutes on this package's coordinator", "the Claude Code plugin's",
        "Claude Code hooks session", "previous run", "your own bound", "report",
    ):
        assert phrase in words, phrase


def test_following_the_other_holder_steps_lands_the_write_on_the_holders_commit(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The same refusal followed to its end on the writer's side: once the
    holder commits and lets go, the steps the row names (reacquire, rebuild,
    write once) land the writer's change on top of the holder's commit.

    Prevents a recovery that loses the holder's bytes."""
    config = _config(tmp_path)
    holder, writer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)
    holder.write(_PATH, b"holder-v1")
    assert _do_write(writer, config, _PATH, "writer").structuredContent["reason"] == "other_holder"
    holder.write(_PATH, b"holder-v2")
    assert holder.release(_PATH) is True

    rebuilt = _do_reacquire(writer, config, _PATH).structuredContent["content"] + " + writer"
    landed = _do_write(writer, config, _PATH, rebuilt)

    assert landed.isError is False, landed.structuredContent
    assert (tmp_path / _PATH).read_bytes() == b"holder-v2 + writer"


# --- this server's own earlier attempt holds the path ------------------------


def test_a_write_refused_by_this_servers_own_earlier_attempt_is_sent_to_swg_reacquire(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The release of an earlier attempt's grant got the
    coordinator's failure envelope, so that grant stands and refuses this
    server's write. No peer exists and a plain retry never sends the release
    again: the row is ``coordinator_failed`` / ``reacquire``, a repeat answers
    the same, and ``swg_reacquire`` then ``swg_write`` is admitted.

    Prevents an agent told to make the same call once more, which only ever
    gets the same answer."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "first").isError is False
    wire.answers[_SESSION_STOP] = {"ok": False, "reason": "internal: OperationalError"}
    assert _do_reacquire(volume, config, _PATH).structuredContent["reason"] == "coordinator_failed"
    wire.answers.pop(_SESSION_STOP)
    raised = _raises_of(volume, monkeypatch)

    first = _row(_do_write(volume, config, _PATH, "second"), raised[-1])
    again = _row(_do_write(volume, config, _PATH, "second"), raised[-1])

    assert type(raised[-1]) is CoordinatorFailed and raised[-1].needs_reacquire is True
    assert _flags(first) == _flags(again) == ("coordinator_failed", "reacquire", False)
    words = first["next_step"]
    for phrase in ("Nothing was written", "earlier attempt", "No other agent holds",
                   "do not send the same write again", "call swg_reacquire on the path",
                   "call swg_write once"):
        assert phrase in words, phrase
    assert "once more" not in words
    reacquired = _do_reacquire(volume, config, _PATH)
    assert reacquired.isError is False, reacquired.structuredContent
    landed = _do_write(volume, config, _PATH, reacquired.structuredContent["content"] + " + second")
    assert landed.isError is False, landed.structuredContent
    assert (tmp_path / _PATH).read_bytes() == b"first + second"


# --- the coordinator answered without deciding --------------------------------


def test_an_undecided_grant_request_names_one_retry_and_the_release(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coordinator could not decide the grant request in time (its registry
    lock held past the watchdog), and the work it abandoned can still take the
    grant. The row keeps the house stop with one retry, adds the release
    sentence, and leaves the path unwritten; the retry it names is admitted.

    Prevents an agent that gives up without the release that frees the path."""
    from tests.test_claude_code_coordinator_server import (
        _DEGRADE_DEADLINE_SEC,
        _await_abandoned_body_settled,
        _HeldRegistryLock,
    )

    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    volume.read(_PATH)
    raised = _raises_of(volume, monkeypatch)
    aborts, completions = server._watchdog_late_aborts_total, server._watchdog_late_completion_total
    budget = server_module.HANDLER_TIMEOUT_SEC
    monkeypatch.setattr(server_module, "HANDLER_TIMEOUT_SEC", _DEGRADE_DEADLINE_SEC)
    with _HeldRegistryLock(server) as held:
        result = _do_write(volume, config, _PATH, "from-the-volume")
        held.release()
    _await_abandoned_body_settled(server, aborts_before=aborts, completions_before=completions)
    monkeypatch.setattr(server_module, "HANDLER_TIMEOUT_SEC", budget)

    row = _row(result, raised[-1])
    assert type(raised[-1]) is CoordinatorFailed and raised[-1].needs_reacquire is False
    assert _flags(row) == ("coordinator_failed", "stop_and_report", False)
    assert "once more" in row["next_step"]
    assert _RELEASE_WORDS in row["next_step"]
    assert (tmp_path / _PATH).read_bytes() == b"seed"
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False


def test_the_release_an_undecided_grant_request_names_lets_a_peer_in(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
) -> None:
    """Work the coordinator abandoned had already taken the grant though the
    answer said ``unavailable``: a peer is refused naming this server's
    attempt until the agent, giving up, releases the path as the row says.

    Prevents a give-up that leaves a peer refused for 10 minutes or more."""
    config = _config(tmp_path)
    volume, peer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)

    def land_then_answer_unavailable(forward: Callable[..., object], payload: dict) -> object:
        wire.answers.pop(_PRE_EDIT)
        forward()
        return _UNAVAILABLE

    wire.answers[_PRE_EDIT] = land_then_answer_unavailable
    gave_up = _do_write(volume, config, _PATH, "from-the-volume").structuredContent
    assert (gave_up["reason"], gave_up["recover"]) == ("coordinator_failed", "stop_and_report")
    refused = _do_write(peer, config, _PATH, "from-the-peer").structuredContent
    assert (refused["reason"], refused["holder_agent_id"]) == ("other_holder", UUID(volume.agent_id).hex)

    assert volume.release(_PATH) is True

    assert _do_write(peer, config, _PATH, "from-the-peer").isError is False
    assert (tmp_path / _PATH).read_bytes() == b"from-the-peer"


def test_a_grant_request_answered_with_neither_admit_nor_refusal_names_no_restart(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coordinator answered the opted-in grant request with ``{}``: it is
    running, so the row says nothing of a restart; nothing was written, one
    retry is named, and the release sentence closes it. The retry is admitted.

    Prevents an agent sending its user to restart a server whose coordinator
    is up, and a give-up that strands a grant the request may have taken."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    raised = _raises_of(volume, monkeypatch)
    wire.answers[_PRE_EDIT] = {}

    row = _row(_do_write(volume, config, _PATH, "from-the-volume"), raised[-1])

    assert type(raised[-1]) is CoordinatorUnreachable and raised[-1].answered is True
    assert _flags(row) == ("coordinator_unavailable", "stop_and_report", False)
    words = row["next_step"]
    assert "restart" not in words.lower()
    assert "Nothing was written" in words and "once more" in words
    assert _RELEASE_WORDS in words
    wire.answers.pop(_PRE_EDIT)
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    assert (tmp_path / _PATH).read_bytes() == b"from-the-volume"


# --- the commit ---------------------------------------------------------------


def _commit_never_sent(forward: Callable[..., object], payload: dict) -> object:
    """The commit's connection is refused before it carried the request; the
    release that follows it reaches the coordinator."""
    if payload.get("success") is True:
        raise CoordinatorUnavailable("scripted: connection refused", never_sent=True)
    return forward()


@pytest.mark.parametrize("on_held", ["refuse", "displace"])
def test_a_commit_never_sent_answers_the_gone_row_with_the_rewrite_and_the_restart_window(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
    on_held: str,
) -> None:
    """The commit never left, so the coordinator did not record the bytes on
    disk. The row stays the stop that says to restart this server, sends the
    agent to swg_reacquire and compare before the swg_write that records the
    bytes once it is back (a restarted server has no record of what was read,
    so a write without that check could land over a later commit), and says swg_release
    cannot reach a coordinator that is gone. On a refuse-mode server it adds
    how long the restarted server can be refused by this run's grants; a
    displace-mode server displaces them, so its row has no window.

    Prevents a restarted refuse-mode server whose agent reads its own old
    grant as a peer it must wait on without being told for how long."""
    volume = _volume(tmp_path, fast_cfg, on_held)
    raised = _raises_of(volume, monkeypatch)
    wire.answers[_POST_EDIT] = _commit_never_sent

    row = _row(_do_write(volume, _config(tmp_path, on_held), _PATH, "unrecorded"), raised[-1])

    assert type(raised[-1]) is CoordinatorUnreachable and raised[-1].answered is False
    assert _flags(row) == ("coordinator_unavailable", "stop_and_report", False)
    words = row["next_step"]
    assert "restart this MCP server" in words
    assert words.index("call swg_reacquire on the path first") < words.index("swg_write them, which records them")
    assert "swg_release cannot reach" in words
    windowed = all(phrase in words for phrase in ("other_holder", "about 10 minutes", "up to 30 minutes"))
    assert windowed is (on_held == "refuse")


def _commit_answered_503(release_answer: object) -> Callable[[Callable[..., object], dict], object]:
    """The commit reaches the coordinator, which records it, and the answer is
    a 503; the release that follows gets ``release_answer``."""

    def answer(forward: Callable[..., object], payload: dict) -> object:
        if payload.get("success") is True:
            forward()
            raise _http_status(503)
        return release_answer

    return answer


def test_a_commit_answered_5xx_is_written_again_and_names_the_release(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytes are on disk and the commit may have landed. The detail says
    to write the path again, and, the release not being confirmed, that a
    release sends it again; the row is ``write_again``, still not retryable,
    and its next step names one more swg_write with the same content, a stop
    if it fails the same way, and swg_release. Following it records the bytes.

    Prevents an agent sent to re-read bytes a read refuses until they are
    recorded, and one told to retry on the flag alone."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    raised = _raises_of(volume, monkeypatch)
    wire.answers[_POST_EDIT] = _commit_answered_503({})

    row = _row(_do_write(volume, config, _PATH, "maybe-recorded"), raised[-1])

    assert type(raised[-1]) is CommitUnconfirmed
    assert "release() sends it again" in row["detail"]
    assert _flags(row) == ("commit_unconfirmed", "write_again", False)
    words = row["next_step"]
    for phrase in ("call swg_write on the path once more", "the same content or a change built from it",
                   "fails the same way", "stop and report"):
        assert phrase in words, phrase
    assert _RELEASE_WORDS in words
    wire.answers.pop(_POST_EDIT)
    assert _do_write(volume, config, _PATH, "maybe-recorded").isError is False
    settled = _do_read(volume, config, _PATH)
    assert settled.structuredContent["content"] == "maybe-recorded", settled.structuredContent


# --- a link out of the managed files ------------------------------------------


def test_a_write_through_a_link_out_of_the_managed_files_is_a_stop_that_forbids_other_routes(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``data/l`` links to ``other``, outside ``data/**``. Refuse mode cannot keep
    that file from its holder, so the volume refuses before it sends anything.
    The row is its own typed stop, whose next step forbids writing the target
    another way; it was ``internal_error`` / ``none``, the shape of a bug.

    Prevents an agent writing the target by its own path to get past it."""
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "t.md").write_bytes(b"t0")
    (tmp_path / "data" / "l").symlink_to(tmp_path / "other")
    volume = _volume(tmp_path, fast_cfg)
    raised = _raises_of(volume, monkeypatch)
    wire.sent.clear()

    row = _row(_do_write(volume, _config(tmp_path), "data/l/t.md", "through-the-link"), raised[-1])

    assert type(raised[-1]) is LinkOutOfManaged
    assert _flags(row) == ("link_out_of_managed", "stop_and_report", False)
    for phrase in ("Stop", "nothing was sent or written", "Do not write that file by any other route",
                   "its own path", "own file tools", "Report"):
        assert phrase in row["next_step"], phrase
    assert wire.sent == []
    assert (tmp_path / "other" / "t.md").read_bytes() == b"t0"


# --- a stale view -------------------------------------------------------------


def test_a_stale_write_is_told_to_rebuild_from_swg_reacquire_and_a_stale_read_is_not(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An edit outside this server since its last write: ``swg_write`` keeps
    ``stale_view`` / ``reacquire`` / retryable and gains a next step that says
    to rebuild from swg_reacquire's bytes, never to resend the old content or
    re-read with the agent's own file tools. Another session's ``swg_read``
    meets the same edit with the same type and answers exactly as before, with
    no next step.

    Prevents an agent resending the content the refusal just turned away."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "v1").isError is False
    (tmp_path / _PATH).write_bytes(b"edited elsewhere")
    raised = _raises_of(volume, monkeypatch)

    row = _row(_do_write(volume, config, _PATH, "v2"), raised[-1])
    read = _do_read(_volume(tmp_path, fast_cfg), config, _PATH)

    assert type(raised[-1]) is StaleView
    assert _flags(row) == ("stale_view", "reacquire", True)
    for phrase in ("Nothing was written", "call swg_reacquire on the path",
                   "rebuild your content from the bytes it returns",
                   "Never send the content you just sent again", "own file tools"):
        assert phrase in row["next_step"], phrase
    assert _RELEASE_WORDS in row["next_step"]
    assert (tmp_path / _PATH).read_bytes() == b"edited elsewhere"
    assert read.isError is True
    assert set(read.structuredContent) == {"reason", "recover", "retryable", "detail"}
    assert _flags(read.structuredContent) == ("stale_view", "reacquire", True)


# --- a failure with no recovery of its own -------------------------------------


def _strip_the_flag(forward: Callable[..., object], payload: dict) -> object:
    """Forward the opted-in grant request without its flag, which is what a
    coordinator that ignores the flag decides, so the admit carries the
    ``collision`` warning no current coordinator sends an opted-in request."""
    return forward({key: value for key, value in payload.items() if key != "if_unheld"})


def test_an_unrecognized_write_failure_says_stop_release_and_report(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admit over a holder, from a coordinator that ignores the flag, whose
    release went unconfirmed: the volume raises a plain ``CoherenceError``
    telling a library caller to reacquire and build a new volume. The row stays
    ``internal_error`` / ``none`` and gains a next step: stop, give the grant
    back with swg_release (swg_reacquire if that does not confirm), report,
    and write the path no other way.

    Prevents an agent left with a stranded grant and a library call it cannot
    make."""
    holder, writer = _volume(tmp_path, fast_cfg, "displace"), _volume(tmp_path, fast_cfg)
    holder.write(_PATH, b"from-the-holder")
    raised = _raises_of(writer, monkeypatch)
    wire.answers[_PRE_EDIT] = _strip_the_flag
    wire.answers[_POST_EDIT] = {}

    row = _row(_do_write(writer, _config(tmp_path), _PATH, "from-the-writer"), raised[-1])

    assert type(raised[-1]) is CoherenceError
    assert "reacquire()" in row["detail"] and "uild a new volume" in row["detail"]
    assert _flags(row) == ("internal_error", "none", False)
    for phrase in ("Stop", "Do not send it again", "call swg_release on the path",
                   "release_unconfirmed twice", "swg_reacquire", "report"):
        assert phrase in row["next_step"], phrase


# --- success ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "asked"),
    [(_PATH, True), ("DATA/shared.txt", None), ("plan.md", False)],
    ids=["managed", "another-spelling", "tracked-not-managed"],
)
def test_a_refuse_mode_write_says_whether_it_asked_first(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    path: str,
    asked: bool | None,
) -> None:
    """On a refuse-mode server a written path's answer carries ``opted_in``,
    judged before the write by the volume's own rule, so it is exactly whether
    the grant request asked only if nobody holds the file (``None`` here: the
    disk decides whether another case is the managed file). A write that did
    not ask says so in its text.

    Prevents an agent believing a write was guarded that could take a file
    from its holder."""
    (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
    volume = _volume(tmp_path, fast_cfg)
    wire.sent.clear()

    result = _do_write(volume, _config(tmp_path), path, "written")

    assert result.isError is False, result.structuredContent
    sent_flag = wire.bodies(_PRE_EDIT)[-1].get("if_unheld") is True
    assert result.structuredContent["opted_in"] is sent_flag
    if asked is not None:
        assert sent_flag is asked
    key = result.structuredContent["path"]
    assert result.structuredContent == {"ok": True, "path": key, "opted_in": sent_flag}
    text = result.content[0].text
    if sent_flag:
        assert text == f"wrote {key}"
    else:
        assert text.startswith(f"wrote {key} ") and "without asking first" in text


def test_a_displace_mode_write_answers_as_before(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A server started without the option keeps today's success answer."""
    volume = _volume(tmp_path, fast_cfg, "displace")

    result = _do_write(volume, _config(tmp_path, "displace"), _PATH, "written")

    assert result.isError is False
    assert result.structuredContent == {"ok": True, "path": _PATH}
    assert [item.text for item in result.content] == [f"wrote {_PATH}"]
