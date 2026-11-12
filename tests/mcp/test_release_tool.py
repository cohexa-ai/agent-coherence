# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``swg_release``: an agent gives back this session's write grant on one path.

Every answer of a release that ran comes from a real ``CoherentVolume.release``
driven through ``_do_release`` against a coordinator running in this process,
with the path written through ``_do_write`` first where it holds one. Where only a
failing coordinator would make the release go unconfirmed, its answer is
scripted at the client boundary, the one place a test can put it, and each
test records what the volume raised and checks the answer's detail against it
byte for byte.

What these do not prove: one machine and loopback only, and the Python
coordinator only.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from mcp.types import CallToolResult

import ccs.adapters.coherent_volume as coherent_volume_module
import ccs.cli._coherence_client as client_module
from ccs.adapters.claude_code.coordinator_server import CoordinatorHTTPServer, session_to_agent_id
from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import REDIRECT_LOCATION_WITHHELD, CoordinatorUnavailable
from ccs.core.exceptions import (
    CoherenceError,
    CoordinatorFailed,
    CoordinatorUnreachable,
    InternalConcurrencyError,
    RedirectRefused,
)
from ccs.mcp.server import (
    _do_reacquire,
    _do_read,
    _do_release,
    _do_status,
    _do_transfer,
    _do_write,
    build_server,
)
from ccs.mcp.session import SessionConfig
from tests.adapters.test_coherent_volume_other_holder import _status, _Wire
from tests.coordinator_helpers import serve_in_process, write_holders

_MANAGED = ("data/**",)
_PATH = "data/shared.txt"
_POST_EDIT = "/hooks/post-edit"
_SESSION_STOP = "/hooks/session-stop"
#: FROZEN: the flags of the row a release the coordinator did not confirm answers.
_UNCONFIRMED_FLAGS = ("release_unconfirmed", "release_again", True)
#: FROZEN: how the text of a release that gave a grant back starts, and the
#: words that keep it from calling the path free.
_RELEASED_TEXT = f"released this session's write grant on {_PATH}."
_NOT_FREE = "The path is not necessarily free: another agent may hold it."
_KEEPS_A_READ = "This session keeps a read of it at the version it held."
#: In principal / nonce shape, minted by nobody.
_OTHER_PRINCIPAL = "X" * 43
_OTHER_NONCE = "Z" * 43


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
    running = serve_in_process(tmp_path, "release-tool")
    yield running
    running.shutdown()


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _Wire:
    """Every POST the volumes send, with answers a test can put in place."""
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


def _answer(result: CallToolResult) -> tuple[dict, str]:
    """A success's structured content and its one text item."""
    assert result.isError is False, result.structuredContent
    [item] = result.content
    return result.structuredContent, item.text


def _raises_of(volume: CoherentVolume, monkeypatch: pytest.MonkeyPatch) -> list[BaseException]:
    """Record every error ``volume.release`` raises on its way to the mapper."""
    raised: list[BaseException] = []
    real_release = volume.release

    def release(path: str) -> bool:
        try:
            return real_release(path)
        except BaseException as exc:
            raised.append(exc)
            raise

    monkeypatch.setattr(volume, "release", release)
    return raised


def _flags(row: dict) -> tuple[str, str, bool]:
    return row["reason"], row["recover"], row["retryable"]


# --- a release that gives a grant back ----------------------------------------


@pytest.mark.parametrize("on_held", ["refuse", "displace"])
def test_releasing_a_held_path_answers_released_and_a_waiting_refuse_mode_peer_is_admitted_next(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, on_held: str
) -> None:
    """This session wrote the path, so it holds it, and a refuse-mode peer's
    write is refused. The release answers ``released: true``; its text says so,
    that the path is not necessarily free, and that this session keeps a read.
    The peer's next write is admitted. The tool exists in both modes.

    Prevents an agent that cannot let go of one path without dropping every
    hold, and a peer left waiting on a grant the agent is done with."""
    config = _config(tmp_path, on_held)
    volume, peer = _volume(tmp_path, fast_cfg, on_held), _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    assert _do_write(peer, _config(tmp_path), _PATH, "from-the-peer").structuredContent["reason"] == "other_holder"

    structured, text = _answer(_do_release(volume, config, _PATH))

    assert structured == {"ok": True, "path": _PATH, "released": True}
    assert text == f"{_RELEASED_TEXT} {_NOT_FREE} {_KEEPS_A_READ}"
    assert write_holders(server, _PATH) == set()
    assert _do_write(peer, _config(tmp_path), _PATH, "from-the-peer").isError is False
    assert (tmp_path / _PATH).read_bytes() == b"from-the-peer"


def test_a_release_after_a_peer_displaced_the_grant_does_not_call_the_path_free(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """A displace-mode peer took the path while this session's record stood.
    The release gives that record back and answers ``released: true``, and the
    peer still holds the path, which is why the text says it is not
    necessarily free.

    Prevents an agent told a path is free while another agent holds it."""
    config = _config(tmp_path)
    volume, displacer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg, "displace")
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    assert _do_write(displacer, _config(tmp_path, "displace"), _PATH, "taken").isError is False

    structured, text = _answer(_do_release(volume, config, _PATH))

    assert structured["released"] is True
    assert _NOT_FREE in text
    assert write_holders(server, _PATH) == {displacer.agent_id}


def test_releasing_a_path_an_earlier_attempt_holds_gives_that_grant_back(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A reacquire's release of the earlier attempt's grant got the failure
    envelope, so that attempt still holds the path and ``swg_status`` lists it
    as this session's. The release answers ``released: true`` and the grant
    is gone from ``/status``.

    Prevents an agent that cannot give back a grant its own earlier attempt
    stranded, short of dropping every hold."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "first").isError is False
    wire.answers[_SESSION_STOP] = {"ok": False, "reason": "internal: OperationalError"}
    assert _do_reacquire(volume, config, _PATH).structuredContent["reason"] == "coordinator_failed"
    wire.answers.pop(_SESSION_STOP)
    before = _do_status(volume, config).structuredContent
    assert (before["held"], before["per_path"][_PATH]["held_by"]) == ([_PATH], "earlier_attempt")

    structured, _text = _answer(_do_release(volume, config, _PATH))

    assert structured == {"ok": True, "path": _PATH, "released": True}
    after = _do_status(volume, config).structuredContent
    assert after["held"] == []
    assert "held_by" not in after["per_path"][_PATH]
    assert write_holders(server, _PATH) == set()


# --- a release with nothing to give back --------------------------------------


def test_a_repeat_release_answers_released_false_and_sends_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """The first release gave the grant back; the repeat has nothing recorded
    to give, answers ``released: false`` saying nothing was sent, and the wire
    sees no request.

    Prevents an agent told a grant went back when nothing was sent."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    assert _do_release(volume, config, _PATH).structuredContent["released"] is True
    wire.sent.clear()

    structured, text = _answer(_do_release(volume, config, _PATH))

    assert structured == {"ok": True, "path": _PATH, "released": False}
    assert text == f"released nothing on {_PATH}: this session has no write grant recorded there, so nothing was sent"
    assert wire.sent == []


@pytest.mark.parametrize(
    "path", [_PATH, "data/missing.md", "notes.md"], ids=["read-not-written", "missing-file", "untracked"]
)
def test_a_path_this_session_never_wrote_answers_released_false_and_sends_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire, path: str
) -> None:
    """A path this session only read, a missing file, and a file outside the
    tracked globs: no write of this session recorded a grant on any of them,
    so the release answers ``released: false`` and sends nothing.

    Prevents a request per release of a path this session never held."""
    (tmp_path / "notes.md").write_bytes(b"notes")
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_read(volume, config, _PATH).isError is False
    wire.sent.clear()

    structured, _text = _answer(_do_release(volume, config, path))

    assert structured == {"ok": True, "path": path, "released": False}
    assert wire.sent == []


def test_releasing_a_path_this_session_handed_off_leaves_the_handoff_record_unchanged(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """This session wrote the path and handed it to another session. The
    transfer moved the grant, so the release has nothing to give, sends
    nothing, and the handoff record ``swg_status`` shows is the same after it.

    Prevents a release that ends or changes a handoff this session gave."""
    config = _config(tmp_path)
    giver, successor = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)
    assert _do_write(giver, config, _PATH, "from-the-giver").isError is False
    successor_id = str(session_to_agent_id(successor.session_id))
    assert _do_transfer(giver, config, [_PATH], successor_id).isError is False
    before = _do_status(giver, config).structuredContent["per_path"][_PATH]["handoff"]
    wire.sent.clear()

    structured, _text = _answer(_do_release(giver, config, _PATH))

    assert structured == {"ok": True, "path": _PATH, "released": False}
    assert wire.sent == []
    assert _do_status(giver, config).structuredContent["per_path"][_PATH]["handoff"] == before
    giver_id = str(session_to_agent_id(giver.session_id))
    assert (before["giver"], before["status"], before["live"]) == (giver_id, "pending", True)


# --- after a release ------------------------------------------------------------


def test_after_a_release_a_write_needs_no_reacquire_until_a_peer_commits(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """This session keeps a read at the version it held: its next write is
    admitted with no reacquire while nobody else wrote, and once a peer
    commits, its write answers ``stale_view`` and the peer's bytes stay.

    Prevents a release that locks the agent out of its own file, and one that
    lets it overwrite a peer's commit."""
    config = _config(tmp_path)
    volume, peer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "v1").isError is False
    assert _do_release(volume, config, _PATH).structuredContent["released"] is True
    assert _do_write(volume, config, _PATH, "v2").isError is False
    assert _do_release(volume, config, _PATH).structuredContent["released"] is True
    assert _do_write(peer, config, _PATH, "from-the-peer").isError is False

    stale = _do_write(volume, config, _PATH, "v3")

    assert stale.structuredContent["reason"] == "stale_view"
    assert (tmp_path / _PATH).read_bytes() == b"from-the-peer"


# --- a release the coordinator did not confirm ----------------------------------


def _no_answer(forward: Callable[..., object], payload: dict) -> object:
    raise CoordinatorUnavailable("scripted: the answer was lost")


def _redirected(forward: Callable[..., object], payload: dict) -> object:
    """What the client raises for a 3xx, which it never follows."""
    raise RedirectRefused(REDIRECT_LOCATION_WITHHELD, status=307)


_STILL_HELD = {"ok": False, "reason": "release_failed", "grants": [{"path": _PATH, "held": True}]}

_UNCONFIRMED = [
    pytest.param(_no_answer, CoordinatorUnreachable, id="lost-answer"),
    pytest.param({"ok": False, "reason": "internal: RuntimeError"}, CoordinatorFailed, id="envelope"),
    pytest.param(_status(503), CoordinatorFailed, id="http-503"),
    pytest.param(_status(409), CoherenceError, id="http-409"),
    pytest.param({"ok": True, "degraded": True}, CoherenceError, id="degraded"),
    pytest.param(_STILL_HELD, CoherenceError, id="grant-still-held"),
    pytest.param(_redirected, RedirectRefused, id="redirect"),
]


@pytest.mark.parametrize(("answer", "raised_type"), _UNCONFIRMED)
def test_a_release_the_coordinator_did_not_confirm_answers_release_again_and_a_repeat_confirms(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
    answer: object,
    raised_type: type[CoherenceError],
) -> None:
    """The release got no answer, the failure envelope, a 5xx, a 4xx, a
    degraded answer, one reporting the grant still held, or a redirect. Each
    answers ``release_unconfirmed`` / ``release_again`` / retryable, the
    volume's text verbatim, and a next step: a repeat is safe, at most twice
    more, then report that the grant may still stand. The grant does stand (a
    peer is still refused), and the repeat it names answers ``released: true``
    and lets the peer in.

    Prevents an agent sent to restart a server whose next release succeeds,
    and one that retries a failing release without end."""
    config = _config(tmp_path)
    volume, peer = _volume(tmp_path, fast_cfg), _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    raised = _raises_of(volume, monkeypatch)
    wire.answers[_POST_EDIT] = answer

    result = _do_release(volume, config, _PATH)

    assert type(raised[-1]) is raised_type
    row = result.structuredContent
    assert result.isError is True and _flags(row) == _UNCONFIRMED_FLAGS
    assert row["detail"] == str(raised[-1])
    assert [item.text for item in result.content] == [row["detail"], row["next_step"]]
    _assert_release_again_words(row["next_step"])
    wire.answers.pop(_POST_EDIT)
    assert _do_write(peer, config, _PATH, "from-the-peer").structuredContent["reason"] == "other_holder"
    assert _do_release(volume, config, _PATH).structuredContent == {"ok": True, "path": _PATH, "released": True}
    assert _do_write(peer, config, _PATH, "from-the-peer").isError is False


def _assert_release_again_words(words: str) -> None:
    """The next step of a release the coordinator did not confirm: names the
    tool, never the library method or a restart, and bounds the repeats."""
    for phrase in ("may still stand", "swg_release on the path again is safe", "at most twice more",
                   "report to your user or host", "swg_reacquire"):
        assert phrase in words, phrase
    assert "release()" not in words
    assert "restart" not in words.lower()


# --- other refusals -------------------------------------------------------------


@contextlib.contextmanager
def _inside_another_thread(volume: CoherentVolume) -> Iterator[None]:
    """Hold ``volume`` from a second thread for the block, as an operation in
    flight there would."""
    held, done = threading.Event(), threading.Event()

    def hold() -> None:
        with volume._single_op_guard():
            held.set()
            done.wait(30)

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(5)
    try:
        yield
    finally:
        done.set()
        thread.join(5)


def test_a_release_refused_for_this_sessions_principal_keeps_the_principal_row(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer
) -> None:
    """The session presents a principal that is not the bound one and a nonce
    that is not the binding's, so the coordinator refuses the release and the
    recovery claim: the row is the settled principal refusal every tool gives.

    Prevents a principal refusal read as a release to repeat."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    volume._principal, volume._mint_nonce = _OTHER_PRINCIPAL, _OTHER_NONCE

    result = _do_release(volume, config, _PATH)

    assert result.isError is True
    assert _flags(result.structuredContent) == ("caller_principal_foreign", "restart_session", False)


def test_a_release_failure_of_any_other_type_is_an_internal_error(
    tmp_path: Path,
    fast_cfg: LifecycleConfig,
    server: CoordinatorHTTPServer,
    wire: _Wire,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Another thread is inside the volume, so the release raises the
    concurrency error, a subclass of the plain error the release row answers.
    The row is keyed on the exact type: this one is ``internal_error``, and
    nothing was sent.

    Prevents a failure that is not an unconfirmed release being answered as
    one the agent should repeat."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    raised = _raises_of(volume, monkeypatch)

    with _inside_another_thread(volume):
        wire.sent.clear()
        result = _do_release(volume, config, _PATH)

    assert type(raised[-1]) is InternalConcurrencyError
    assert result.isError is True
    assert _flags(result.structuredContent) == ("internal_error", "none", False)
    assert result.structuredContent["detail"] == str(raised[-1])
    assert wire.sent == []


def test_an_invalid_path_is_refused_before_anything_is_sent(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """A path that leaves the workspace is the agent's own input mistake.

    Prevents a release of a path outside the workspace reaching the volume."""
    volume = _volume(tmp_path, fast_cfg)
    wire.sent.clear()

    result = _do_release(volume, _config(tmp_path), "../outside.txt")

    assert result.isError is True
    assert _flags(result.structuredContent) == ("invalid_path", "fix_path", False)
    assert wire.sent == []


def test_an_unattached_volume_answers_coordinator_unavailable_and_sends_nothing(
    tmp_path: Path, fast_cfg: LifecycleConfig, server: CoordinatorHTTPServer, wire: _Wire
) -> None:
    """This session holds the path, then loses its endpoint. The release
    answers ``coordinator_unavailable`` and sends nothing: a volume with no
    endpoint would answer that it gave the grant back without asking.

    Prevents an agent told its grant went back when no request was sent."""
    config = _config(tmp_path)
    volume = _volume(tmp_path, fast_cfg)
    assert _do_write(volume, config, _PATH, "from-the-volume").isError is False
    volume._endpoint = None
    wire.sent.clear()

    result = _do_release(volume, config, _PATH)

    assert result.isError is True
    assert _flags(result.structuredContent) == ("coordinator_unavailable", "retry_later", False)
    assert wire.sent == []


# --- the registered tool --------------------------------------------------------


def test_the_release_tool_is_registered_as_a_write_whose_description_gives_its_answers() -> None:
    """A client lists ``swg_release`` taking one path, not read-only, and its
    description says what each answer means: released or not, never that the
    path is free, the read this session keeps, the unconfirmed row's values,
    the holder's duty to release a path it finished writing, and the scope.

    Prevents a model that never learns it should let go of a path, or reads
    ``released: true`` as a free path."""
    tools = {tool.name: tool for tool in asyncio.run(build_server().list_tools())}
    tool = tools["swg_release"]

    assert tool.annotations is not None and tool.annotations.readOnlyHint is False
    assert set(tool.inputSchema["properties"]) == {"path"}
    description = tool.description or ""
    for phrase in ("released=true", "released=false", "does not mean the path is free", "keeps a read",
                   "reason=release_unconfirmed", "recover=release_again", "retryable=true",
                   "When you finish writing a path, swg_release it", "SINGLE-HOST only"):
        assert phrase in description, phrase
