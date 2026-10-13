# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The caller-principal retry protocol, pinned ONCE for both long-lived clients.

``CoherentVolume`` and ``SubstrateCoordinatorSession`` each carry their own
copy of what a request refused for its caller principal does: send once; on a
typed ``caller_principal_*`` refusal claim again with the session's SAME mint
nonce; adopt a principal that differs from the one presented and retry the
request exactly ONCE; a 404 on the claim retries once without the header;
anything the claim cannot cure raises the typed ``CallerPrincipalRefused``,
and a refused retry is reported under the RETRY's reason. The two copies are
deliberate — #250 tracks extracting one helper — so this module drives both
classes through ONE scripted refuse/recover/retry matrix over the wire (a
loopback coordinator answering from a script; nothing is spawned or stubbed
below the public API) and asserts every row identically on:

1. every POST, in order — the request route, ``/principal/claim``, and nothing
   else — so a second claim, a third send, or a send that never went out fails;
2. the ``Coherence-Caller-Principal`` header each request POST presented — the
   held one first, the recovered one on the retry, none after a 404;
3. the outcome — the 2xx answer consumed, or ``CallerPrincipalRefused`` with
   the exact typed reason and the exact message.

Every expectation is a hand-written duplicate of the documented protocol and
its constants; nothing is derived from the code under test, so a reworded
constant or a re-ordered send turns a row red and becomes a decision. The one
intended difference between the copies is pinned at the bottom, on its own.

Principals and the mint nonce are asserted by token — the held one, the
recovered one, the attach nonce — so a row that fails prints which one a POST
carried and never a value; the leak check names what leaked, not what it was.
That holds on every failure path, under pytest's default flags and under
``-l``: whatever the client lets out, its typed error or a crash, is kept as a
value (:class:`_Ran`), leak-checked FIRST, and a crash is then reported by
type and location only, without a traceback — a traceback entry prints the
arguments of its frame, and on the claim path the nonce is one; and no
test-frame local ever holds a principal or the nonce (they live in the
:class:`_Script`, whose repr names counts), so a dump of locals prints tokens.
"""

from __future__ import annotations

import contextlib
import functools
import http.server
import json
import logging
import sys
import threading
import traceback
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.adapters.substrate import PreReadResult, SubstrateCoordinatorSession
from ccs.core.exceptions import (
    CallerPrincipalRefused,
    CoherenceDegradedWarning,
    CoherenceError,
)
from tests.test_caller_principal_client_recovery import (
    _ABSENT,
    _CLAIMED,
    _FOREIGN,
    _PRINCIPAL_HEADER,
    _refusal,
    _value,
)

_CLAIM_ROUTE = "/principal/claim"  # frozen duplicate of the wire route

_P = _value("P")  # the principal the attach claim binds (the HELD one)
_Q = _value("Q")  # the principal a recovery claim returns instead

# The case table and every wire assertion name a principal or a nonce by one
# of these tokens, never by value: what a failing row prints is "the held
# principal" and "the attach nonce", not the principal and not the nonce, so
# no such value reaches an assertion diff. The values themselves live only in
# the coordinator's script and in the leak check, which names what it found.
_HELD = "the held principal"
_RECOVERED = "another principal"  # the one a recovery claim binds instead
_UNEXPECTED_PRINCIPAL = "a principal outside the script"
_SAME_NONCE = "the attach nonce"
"""In a wire row: the mint nonce a recovery claim must present, which is the
one the coordinator saw on the ATTACH claim — read off the wire, never off
the client."""
_ANOTHER_NONCE = "a nonce other than the attach nonce"

_ANSWER = "answer the request"
"""Placeholder in a request-answer script for the admitted 2xx body, which
differs by route (the driver supplies it)."""

Answer = tuple[int, dict[str, Any]]
WireRow = tuple[str, str | None, str | None]  # (path, principal header, mint nonce) of one POST

# Claim answers that carry a principal are scripted by token too; the script
# resolves them (:func:`_claim_answer`) so the case table prints no value.
_BINDS_RECOVERED = "the claim binds another principal"
_BINDS_HELD = "the claim binds the held principal"
_CLAIM_REFUSED: Answer = (200, {"ok": False, "reason": _CLAIMED, "detail": "bound"})
_CLAIM_UNCONFIRMED: Answer = (200, {"ok": False, "reason": "claim_unconfirmed"})  # the watchdog envelope
_NO_PRINCIPALS: Answer = (404, {"error": "not found"})
_UNTYPED_REJECTION: Answer = (400, {"error": "rejected", "reason": "caller_principal_revoked"})
"""A 400 whose ``reason`` is a string outside the frozen refusal vocabulary: an
ordinary rejected request, never a principal refusal to recover from."""


def _refused_message(reason: str, detail: str) -> str:
    """The message the documented protocol builds for a refusal a client
    reports — hand-written here, not imported from the client."""
    return f"coordinator refused the caller principal ({reason}): {detail}"


# What recovery reports when the claim cannot cure the refusal, and what the
# one retry reports when it is refused too — frozen duplicates of the
# documented constants, byte for byte.
_SAME_PRINCIPAL = "claiming with the held nonce returned the principal that was refused"
_BOUND_ELSEWHERE = "the session is bound under a different mint nonce (caller_principal_claimed); not re-minting"
_UNCONFIRMED = "the claim was not confirmed (claim not confirmed (reason=claim_unconfirmed))"
_REFUSED_AGAIN = "refused again after claiming with the held nonce; not retrying"


def _leak_report(text: str, **secrets: str) -> str | None:
    """What leaked, by NAME, when a principal or nonce is in ``text``; ``None``
    when none is. Checks against the real values but reports the leaking
    lines with each value replaced by ``<name>``. It raises nothing itself —
    its arguments ARE the values, and a traceback entry lists a frame's
    arguments — so the caller fails on the report, without a traceback."""
    unusable = sorted(name for name, value in secrets.items() if not isinstance(value, str) or not value)
    if unusable:
        return f"control: a secret to check for is not a non-empty string: {unusable}"
    leaked = {name: value for name, value in secrets.items() if value in text}
    if not leaked:
        return None

    def redact(line: str) -> str:
        for name, value in leaked.items():
            line = line.replace(value, f"<{name}>")
        return line

    where = [redact(line) for line in text.splitlines() if any(value in line for value in leaked.values())]
    return f"a principal or nonce reached what the client reported ({', '.join(sorted(leaked))}): {where[:5]}"


class _ClientLog(logging.Handler):
    """Every line the library logs, at DEBUG, kept for the leak check — and
    kept OUT of pytest's own captured-log report, which prints on any
    failure and would show a leaked value. :func:`_client_log` attaches this
    to the library's logger tree with propagation off for the test's
    duration; the leak check reports a leaking line redacted."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # a format its arguments do not fit
            # ``Handler.handleError`` would print the message AND the
            # arguments to stderr; kept raw here instead, still checked.
            message = f"<unformattable {record.msg!r} with arguments {record.args!r}>"
        self.lines.append(f"{record.name}:{record.lineno} {message}")


@contextlib.contextmanager
def _client_log() -> Iterator[_ClientLog]:
    """The library's logging captured at DEBUG into a :class:`_ClientLog`,
    with propagation to the root logger (and so to pytest's report) off;
    the logger's level and propagation are restored afterwards."""
    library = logging.getLogger("ccs")
    handler = _ClientLog()
    level, propagate = library.level, library.propagate
    library.setLevel(logging.DEBUG)
    library.propagate = False
    library.addHandler(handler)
    try:
        yield handler
    finally:
        library.removeHandler(handler)
        library.setLevel(level)
        library.propagate = propagate


def _claim_answer(answer: Answer | str) -> Answer:
    """A scripted claim answer with its principal token resolved to the value
    the coordinator sends: the held one, or the one a recovery claim binds."""
    if answer == _BINDS_HELD:
        return 200, {"ok": True, "principal": _P}
    if answer == _BINDS_RECOVERED:
        return 200, {"ok": True, "principal": _Q}
    assert not isinstance(answer, str), "control: a claim answer is a token above or a status and body"
    return answer


# --- the scripted coordinator --------------------------------------------------


class _Script:
    """What the loopback coordinator answers each route from, in order, and
    every POST it saw. A route with no answer left is answered 500, so an
    extra send a mutant makes shows up twice: on the wire and as a failure
    the client reports.

    The script is the only place a test holds a principal or the nonce: the
    raw wire stays in ``seen`` and leaves only as tokens (:meth:`redacted`)
    or as the leak check's secrets, and the repr names counts, not rows, so
    a dump of test-frame locals (``-l``) prints no value."""

    def __init__(self, queued: dict[str, list[Answer]], admitted: dict[str, Answer]) -> None:
        self.queued = {route: list(answers) for route, answers in queued.items()}
        self.admitted = dict(admitted)
        self.seen: list[WireRow] = []

    def __repr__(self) -> str:
        return f"<_Script: {len(self.seen)} POSTs seen; scripted routes {sorted(self.queued)}>"

    def next_answer(self, path: str) -> Answer:
        if self.queued.get(path):
            return self.queued[path].pop(0)
        if path in self.admitted:
            return self.admitted[path]
        return 500, {"error": "unscripted request"}

    def _claimed_at_attach(self) -> bool:
        """Whether the client's first POST — the first of its life — was a
        claim carrying a mint nonce. Binds no row to a local."""
        return (
            bool(self.seen)
            and self.seen[0][0] == _CLAIM_ROUTE
            and isinstance(self.seen[0][2], str)
            and bool(self.seen[0][2])
        )

    def attach_nonce(self) -> str:
        """The mint nonce the client's attach claim presented, read off the
        wire. The control fails without a traceback: a traceback would list
        this frame, and the row it would have to name holds the nonce."""
        if not self._claimed_at_attach():
            pytest.fail("control: the client's first POST was not a claim carrying a mint nonce", pytrace=False)
        return self.seen[0][2]  # type: ignore[return-value]

    def secrets(self) -> dict[str, str]:
        """What the leak check looks for: both scripted principals and, once
        the client has claimed, the attach nonce."""
        found = {"held": _P, "recovered": _Q}
        if self._claimed_at_attach():
            found["nonce"] = self.seen[0][2]  # type: ignore[assignment]
        return found

    def redacted(self) -> list[WireRow]:
        """The wire as seen, with every principal and every mint nonce
        replaced by its token, so a wire assertion that fails prints WHICH
        principal and WHICH nonce a POST carried, never the values."""
        nonce = self.attach_nonce()
        principals = {None: None, _P: _HELD, _Q: _RECOVERED}
        # A path carrying a principal or the nonce (a client that put one in
        # the URL) prints the token, not the value.
        tokens = {_P: _HELD, _Q: _RECOVERED, nonce: _SAME_NONCE}
        return [
            (
                functools.reduce(lambda text, pair: text.replace(*pair), tokens.items(), path),
                principals.get(header, _UNEXPECTED_PRINCIPAL),
                None if presented is None else _SAME_NONCE if presented == nonce else _ANOTHER_NONCE,
            )
            for path, header, presented in self.seen
        ]


class _ScriptedServer(http.server.HTTPServer):
    def __init__(self, handler: type[http.server.BaseHTTPRequestHandler], script: _Script) -> None:
        super().__init__(("127.0.0.1", 0), handler)
        self.script = script

    def handle_error(self, request: Any, client_address: Any) -> None:
        """The stdlib prints the handler's traceback to stderr, which pytest
        shows on a failure; a crash in the handler is named by type only. The
        row still fails: the POST it did not answer is an error on the wire."""
        sys.stderr.write(f"scripted coordinator: the request handler crashed: {type(sys.exception()).__name__}\n")


class _Coordinator(http.server.BaseHTTPRequestHandler):
    """Answers every POST from the server's :class:`_Script` and records it as
    ``(path, principal header, mint nonce)``; answers the volume's attach-time
    ``GET /status`` with strict mode on. Nothing else a coordinator does."""

    server: _ScriptedServer

    def do_GET(self) -> None:  # noqa: N802 — stdlib name
        # The volume checks each glob it declared against the published strict,
        # tracked and ignored sets, not the count; publish the one glob these
        # volumes declare, and no ignored pattern, so the attach is confirmed
        # and the protocol under test runs.
        self._answer(200, {"policy_summary": {
            "strict_mode_pattern_count": 1,
            "tracked_patterns": ["data/**"],
            "user_added_patterns": [],
            "ignored_patterns": [],
            "strict_mode_patterns": ["data/**"],
        }})

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        self.server.script.seen.append((self.path, self.headers.get(_PRINCIPAL_HEADER), body.get("mint_nonce")))
        self._answer(*self.server.script.next_answer(self.path))

    def _answer(self, status: int, answer: dict[str, Any]) -> None:
        raw = json.dumps(answer).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args: Any) -> None:
        return


@contextlib.contextmanager
def _scripted_coordinator(root: Path, script: _Script) -> Iterator[None]:
    """A workspace whose ``.coherence/`` points a client at a loopback
    coordinator answering from ``script``. The port is live before the client
    attaches, so both classes ATTACH (no policy write, no spawn)."""
    coherence = root / ".coherence"
    coherence.mkdir(mode=0o700)
    httpd = _ScriptedServer(_Coordinator, script)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    (coherence / "server.pid").write_text(f"12345\n{httpd.server_address[1]}\n")
    (coherence / "hook.secret").write_text("test-secret")
    try:
        yield
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def fast_cfg() -> LifecycleConfig:
    """Lifecycle config with a short connect budget; no coordinator is spawned
    here, the budget only bounds the attach-time probe."""
    return LifecycleConfig(
        idle_shutdown_sec=0,
        sweep_interval_sec=0.1,
        notice_evict_max_age_sec=1.0,
        port_file_retry_attempts=20,
        port_file_retry_interval_sec=0.05,
        connect_retry_attempts=10,
        connect_retry_interval_sec=0.05,
    )


# --- the two clients, behind one shape ------------------------------------------


class _VolumeDriver:
    """``CoherentVolume``: the request under test is ``write()``'s pre-edit
    (the grant request, the first POST a write makes). An admitted pre-edit
    is followed by the write's post-edit, which the script admits every time
    and which presents whatever principal the pre-edit settled."""

    request_route = "/hooks/pre-edit"
    admitted: Answer = (200, {"ok": True})
    follow_up: dict[str, Answer] = {"/hooks/post-edit": (200, {"ok": True})}
    answered = b"v2"
    #: How the volume reports a 400 that is not a principal refusal (strict).
    untyped_report = "coordinator request to /hooks/pre-edit failed: HTTP 400"

    def __init__(self, on_error: str) -> None:
        self.on_error = on_error
        self.kind = f"volume-{on_error}"

    def construct(self, root: Path, cfg: LifecycleConfig) -> CoherentVolume:
        return CoherentVolume(root, managed=("data/**",), on_error=self.on_error, config=cfg)  # type: ignore[arg-type]

    def act(self, client: Any, root: Path) -> bytes:
        client.write("data/shared.txt", b"v2")
        return (root / "data/shared.txt").read_bytes()

    def tail(self, outcome: object, last_header: str | None) -> list[WireRow]:
        """The POSTs the write makes after the protocol's last request POST:
        the post-edit, under the same principal, once the pre-edit was
        answered — and in degrade mode also after an untyped rejection, which
        the volume degrades over and writes best-effort through."""
        if isinstance(outcome, _Answered) or (self.on_error == "degrade" and isinstance(outcome, _NotARefusal)):
            return [("/hooks/post-edit", last_header, None)]
        return []

    def assert_untyped(self, ran: _Ran) -> None:
        """Reported as the class documents: strict raises the plain
        ``CoherenceError`` naming the status; degrade warns once, counts, and
        the write lands best-effort. Never the typed refusal, in either."""
        if self.on_error == "strict":
            assert type(ran.raised) is CoherenceError and str(ran.raised) == self.untyped_report
            return
        assert ran.raised is None and ran.witness == self.answered
        assert [type(w.message) for w in ran.warned] == [CoherenceDegradedWarning]

    def assert_quiet(self, ran: _Ran) -> None:
        """A principal refusal is the coordinator's definite answer, not an
        infrastructure failure: in BOTH modes it is raised, never softened
        into a degrade — the volume is not degraded and warned nothing."""
        assert not ran.client.is_degraded and ran.warned == []


class _SessionDriver:
    """``SubstrateCoordinatorSession``: the request under test is the read
    leg, ``pre_read`` — one POST, no follow-up. The commit leg runs the same
    ``_post``."""

    kind = "substrate"
    request_route = "/hooks/pre-read"
    admitted: Answer = (200, {"status": "fresh", "version": 7})
    follow_up: dict[str, Answer] = {}
    answered = PreReadResult(version=7, stale_denied=False, hash_differs=False, prior_version_seen=None)
    #: How the session reports a 400 that is not a principal refusal.
    untyped_report = "coordinator /hooks/pre-read failed (fail-closed): HTTP 400"

    def construct(self, root: Path, cfg: LifecycleConfig) -> SubstrateCoordinatorSession:
        return SubstrateCoordinatorSession(root, managed=("**",), config=cfg)

    def act(self, client: Any, root: Path) -> PreReadResult:
        return client.pre_read("workspace/shared.bin", None)

    def tail(self, outcome: object, last_header: str | None) -> list[WireRow]:
        return []

    def assert_untyped(self, ran: _Ran) -> None:
        """Reported as the class documents: fail-closed, the read leg's plain
        ``CoherenceError`` naming the status — never the typed refusal."""
        assert type(ran.raised) is CoherenceError and str(ran.raised) == self.untyped_report

    def assert_quiet(self, ran: _Ran) -> None:
        assert ran.warned == []


_DRIVERS = (_VolumeDriver("strict"), _VolumeDriver("degrade"), _SessionDriver())


# --- one run of a client, kept as values ------------------------------------------


def _typename(obj: object) -> str:
    return "none" if obj is None else type(obj).__name__


class _Ran:
    """What one run of a client's public operation let out, kept as values —
    the answer, the client's own typed error, or a crash — so that the test,
    not an escaping exception, decides what a failure prints. A crash is any
    exception that is not the client's :class:`CoherenceError`; it is never
    re-raised: a traceback entry prints its frame's arguments, and on the
    claim path the mint nonce is one. The warnings and the library's log
    lines of the run are kept here too. The repr names types and counts
    only, because ``-l`` dumps every test-frame local."""

    def __init__(self, client: Any = None) -> None:
        self.client = client
        #: POSTs the attach cost; ``None`` until a client was constructed here.
        self.attached: int | None = None
        self.phase = "attaching" if client is None else "acting"
        self.witness: object = None
        self.raised: CoherenceError | None = None
        self.crashed: BaseException | None = None
        self.warned: list[Any] = []
        self.log: list[str] = []

    def __repr__(self) -> str:
        return (
            f"<_Ran {self.phase}: client={_typename(self.client)} witness={_typename(self.witness)}"
            f" raised={_typename(self.raised)} crashed={_typename(self.crashed)}"
            f" {len(self.warned)} warnings {len(self.log)} log lines>"
        )

    def step(self, phase: str, do: Callable[[], Any]) -> Any:
        """One step of the client's life; ``None`` when it raised, with what
        it raised kept as the client's report or as a crash."""
        self.phase = phase
        try:
            return do()
        except CoherenceError as exc:
            self.raised = exc
        except _PASS_THROUGH:
            raise
        except BaseException as exc:  # anything else the client let out — SystemExit included — is an outcome here
            self.crashed = exc
        return None


#: What a client step never swallows: an interrupt, and pytest's own outcomes
#: (a failed control inside a step must fail the row as itself). Anything else
#: — a SystemExit included, which pytest would otherwise render with every
#: frame's arguments — is kept as the crash it is and reported by type.
_PASS_THROUGH = (
    KeyboardInterrupt,
    pytest.fail.Exception,
    pytest.skip.Exception,
    pytest.xfail.Exception,
    pytest.exit.Exception,
)


def _run(driver: Any, script: _Script, root: Path, cfg: LifecycleConfig, client: Any = None) -> _Ran:
    """Attach a client — or take the one handed in — and run the driver's
    public operation once, with the library's DEBUG log and every warning
    captured for the leak check and kept out of pytest's own report."""
    ran = _Ran(client)
    with _client_log() as log, warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        if ran.client is None:
            ran.client = ran.step("attaching", lambda: driver.construct(root, cfg))
            ran.attached = None if ran.client is None else len(script.seen)
        if ran.client is not None:
            ran.witness = ran.step("acting", lambda: driver.act(ran.client, root))
    ran.log, ran.warned = log.lines, list(warned)
    return ran


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and every cause or context behind it, each once."""
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        yield link
        link = link.__cause__ if link.__cause__ is not None or link.__suppress_context__ else link.__context__


def _reported(ran: _Ran) -> str:
    """Everything the client reported, and everything a failure could print
    of it: the raised or crashing error's whole chain as a traceback renders
    it and as ``repr`` shows it (an assertion message prints the repr), the
    answer, every warning, every log line."""
    parts: list[str] = []
    for exc in (ran.raised, ran.crashed):
        if exc is not None:
            parts.append("".join(traceback.format_exception(exc)))
            parts.extend(repr(link) for link in _chain(exc))
    parts.append(repr(ran.witness))
    parts.extend(f"{w.message!r} {w.message}" for w in ran.warned)
    parts.extend(ran.log)
    return "\n".join(parts)


def _assert_nothing_leaks(script: _Script, ran: _Ran) -> None:
    """No principal and no nonce in anything the client reported (the house
    rule, on the client side). Runs FIRST, before anything else a failing row
    prints, and fails WITHOUT a traceback naming what leaked by NAME: the
    values live in the script and in the report, never in a local here."""
    report = _leak_report(_reported(ran), **script.secrets())
    if report is not None:
        pytest.fail(report, pytrace=False)


def _where(exc: BaseException) -> str:
    """The exception's type and where it was raised — the innermost frame,
    after the innermost one inside the library when that differs — which are
    names and line numbers, never values."""
    frames = traceback.extract_tb(exc.__traceback__)
    inside = [frame for frame in frames if "ccs" in Path(frame.filename).parts]
    marks = [f"{Path(frame.filename).name}:{frame.lineno} in {frame.name}" for frame in (*inside[-1:], *frames[-1:])]
    return f"{type(exc).__qualname__} at {' -> '.join(dict.fromkeys(marks)) or 'no traceback'}"


def _fail_if_escaped(ran: _Ran) -> None:
    """A row whose client crashed, or never attached, fails here — after the
    leak check, without a traceback — naming a crash by type and location
    only: its arguments are exactly what a traceback would print."""
    if ran.crashed is not None:
        pytest.fail(f"the client crashed while {ran.phase}: {_where(ran.crashed)}; its arguments are withheld", pytrace=False)
    if ran.client is None:
        # Its message is withheld too: a client that failed before its first
        # POST holds a nonce the wire never saw, so the leak check cannot know
        # it, and the message is the one place it could surface.
        pytest.fail(f"the client did not attach: {_where(ran.raised)}; its message is withheld", pytrace=False)


# --- the matrix ------------------------------------------------------------------


@dataclass(frozen=True)
class _Answered:
    """The request route's 2xx answer was consumed: the operation completed."""


@dataclass(frozen=True)
class _Refused:
    """``CallerPrincipalRefused`` with exactly this typed reason and message,
    and whether it is the session's settled state: ``False`` only when the
    recovery claim's answer was lost, so the next request claims again with
    the same nonce by itself."""

    reason: str
    message: str
    settled: bool


@dataclass(frozen=True)
class _NotARefusal:
    """Not recovered (no claim, no retry) and not typed as a principal refusal;
    reported the way the class reports any rejected request."""


@dataclass(frozen=True)
class _Case:
    id: str
    #: ``_HELD`` when the attach claim binds a principal; ``None`` when the
    #: coordinator answers that claim 404 (it issues none), so none is held.
    held: str | None
    #: The request route's answers, in order (``_ANSWER`` = the admitted body).
    requests: tuple[Answer | str, ...]
    #: ``/principal/claim``'s answers AFTER the attach claim, in order (a
    #: token for an answer that binds a principal).
    claims: tuple[Answer | str, ...]
    #: Every POST the protocol makes, in order: ``("request", <header
    #: token>)`` or ``("claim", _SAME_NONCE)``.
    wire: tuple[tuple[str, str | None], ...]
    outcome: _Answered | _Refused | _NotARefusal


_CASES = (
    _Case(
        id="refused-then-a-different-principal-is-adopted-and-the-retry-is-answered",
        held=_HELD,
        requests=(_refusal(_FOREIGN), _ANSWER),
        claims=(_BINDS_RECOVERED,),
        wire=(("request", _HELD), ("claim", _SAME_NONCE), ("request", _RECOVERED)),
        outcome=_Answered(),
    ),
    _Case(
        id="refused-and-the-claim-returns-the-refused-principal-stops",
        held=_HELD,
        requests=(_refusal(_FOREIGN),),
        claims=(_BINDS_HELD,),
        wire=(("request", _HELD), ("claim", _SAME_NONCE)),
        outcome=_Refused(_FOREIGN, _refused_message(_FOREIGN, _SAME_PRINCIPAL), settled=True),
    ),
    _Case(
        id="refused-and-the-claim-is-refused-as-claimed-stops-without-re-minting",
        held=_HELD,
        requests=(_refusal(_FOREIGN),),
        claims=(_CLAIM_REFUSED,),
        wire=(("request", _HELD), ("claim", _SAME_NONCE)),
        outcome=_Refused(_FOREIGN, _refused_message(_FOREIGN, _BOUND_ELSEWHERE), settled=True),
    ),
    _Case(
        id="refused-and-the-claim-is-unconfirmed-stops",
        held=_HELD,
        requests=(_refusal(_FOREIGN),),
        claims=(_CLAIM_UNCONFIRMED,),
        wire=(("request", _HELD), ("claim", _SAME_NONCE)),
        outcome=_Refused(_FOREIGN, _refused_message(_FOREIGN, _UNCONFIRMED), settled=False),
    ),
    _Case(
        id="refused-and-the-claim-answers-404-retries-once-without-the-header",
        held=_HELD,
        requests=(_refusal(_FOREIGN), _ANSWER),
        claims=(_NO_PRINCIPALS,),
        wire=(("request", _HELD), ("claim", _SAME_NONCE), ("request", None)),
        outcome=_Answered(),
    ),
    _Case(
        id="refused-404-and-the-headerless-retry-is-refused-again-under-its-own-reason",
        held=_HELD,
        requests=(_refusal(_FOREIGN), _refusal(_ABSENT)),
        claims=(_NO_PRINCIPALS,),
        wire=(("request", _HELD), ("claim", _SAME_NONCE), ("request", None)),
        outcome=_Refused(_ABSENT, _refused_message(_ABSENT, _REFUSED_AGAIN), settled=True),
    ),
    _Case(
        id="refused-recovered-and-refused-again-raises-the-retrys-reason-with-no-third-send",
        held=None,
        requests=(_refusal(_ABSENT), _refusal(_FOREIGN)),
        claims=(_BINDS_RECOVERED,),
        wire=(("request", None), ("claim", _SAME_NONCE), ("request", _RECOVERED)),
        outcome=_Refused(_FOREIGN, _refused_message(_FOREIGN, _REFUSED_AGAIN), settled=True),
    ),
    _Case(
        id="a-rejection-whose-reason-is-not-a-known-string-is-not-recovered",
        held=_HELD,
        requests=(_UNTYPED_REJECTION,),
        claims=(),
        wire=(("request", _HELD),),
        outcome=_NotARefusal(),
    ),
)


def _script_for(driver: Any, case: _Case) -> _Script:
    """The coordinator's script for ``case`` against ``driver``: the attach
    claim (binding the held principal, or 404 when nothing is held), then the
    case's claim answers; the request route's answers with the driver's
    admitted body filled in; and the driver's follow-up routes admitted every
    time."""
    attach = _NO_PRINCIPALS if case.held is None else _BINDS_HELD
    claims = [_claim_answer(answer) for answer in (attach, *case.claims)]
    requests = [driver.admitted if answer == _ANSWER else answer for answer in case.requests]
    return _Script({_CLAIM_ROUTE: claims, driver.request_route: requests}, driver.follow_up)


def _expected_wire(driver: Any, case: _Case) -> list[WireRow]:
    """Every POST the coordinator must have seen, in order and by token: the
    attach claim, the case's rows rendered onto the driver's request route,
    then the driver's own follow-up POSTs — nothing else. Compared with the
    wire as :meth:`_Script.redacted` renders it."""
    rows: list[WireRow] = [(_CLAIM_ROUTE, None, _SAME_NONCE)]
    for kind, value in case.wire:
        if kind == "claim":
            assert value == _SAME_NONCE
            rows.append((_CLAIM_ROUTE, None, _SAME_NONCE))
        else:
            rows.append((driver.request_route, value, None))
    last_kind, last_header = case.wire[-1]
    return rows + driver.tail(case.outcome, last_header if last_kind == "request" else None)


def _assert_refused(ran: _Ran, reason: str, message: str, *, settled: bool) -> None:
    """The typed refusal, with exactly this reason, exactly this message, and
    exactly this ``settled``: both copies of the protocol must type a lost
    recovery answer as not settled, and every other stop as settled."""
    assert type(ran.raised) is CallerPrincipalRefused, f"not the typed refusal: {ran.raised!r}"
    assert (ran.raised.reason, str(ran.raised), ran.raised.settled) == (reason, message, settled)


def _assert_outcome(driver: Any, case: _Case, ran: _Ran) -> None:
    outcome = case.outcome
    if isinstance(outcome, _NotARefusal):
        driver.assert_untyped(ran)
        return
    driver.assert_quiet(ran)
    if isinstance(outcome, _Answered):
        assert ran.raised is None and ran.witness == driver.answered
        return
    _assert_refused(ran, outcome.reason, outcome.message, settled=outcome.settled)


@pytest.mark.parametrize("case", _CASES, ids=[case.id for case in _CASES])
@pytest.mark.parametrize("driver", _DRIVERS, ids=[driver.kind for driver in _DRIVERS])
def test_both_long_lived_clients_run_one_refuse_recover_retry_protocol(
    driver: Any, case: _Case, tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """One scripted answer sequence, one hand-written expectation, asserted
    identically on the volume (strict AND degrade — the refusal raises in
    both) and the substrate session, over the real request routes each class
    posts to. Fails when either copy sends the request more than twice,
    claims more than once, claims with a nonce other than its attach claim's,
    presents the wrong principal on the retry (or any after a 404), reports a
    second refusal under the FIRST refusal's reason, recovers a rejection
    that carries no typed reason, or softens the refusal in degrade mode."""
    script = _script_for(driver, case)
    with _scripted_coordinator(tmp_path, script):
        ran = _run(driver, script, tmp_path, fast_cfg)

    # The leak check runs FIRST: once it has passed, nothing a later failure
    # prints — the redacted wire, the reason and message, a warning — can
    # carry a principal or the nonce either. A crash is reported next, by type.
    _assert_nothing_leaks(script, ran)
    _fail_if_escaped(ran)
    assert ran.attached == 1, "control: attaching cost exactly the attach claim"
    assert script.redacted() == _expected_wire(driver, case)
    _assert_outcome(driver, case, ran)


# --- the one intended difference between the two copies -------------------------

_NOT_CLAIMING_AGAIN = "the session is bound under a different mint nonce (caller_principal_claimed); not claiming again"

_SECOND_REFUSED_REQUEST: dict[str, tuple[tuple[tuple[str, str | None], ...], str]] = {
    # what the SECOND refused request costs on the wire, and what it reports
    "volume-strict": ((("request", _HELD),), _refused_message(_FOREIGN, _NOT_CLAIMING_AGAIN)),
    "substrate": ((("request", _HELD), ("claim", _SAME_NONCE)), _refused_message(_FOREIGN, _BOUND_ELSEWHERE)),
}


@pytest.mark.parametrize("driver", [_DRIVERS[0], _DRIVERS[2]], ids=list(_SECOND_REFUSED_REQUEST))
def test_a_session_known_bound_under_another_nonce_is_claimed_again_by_the_substrate_session_only(
    driver: Any, tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A DELIBERATE, documented difference between the two copies, pinned so
    that a change to either side is a decision rather than drift (#250 tracks
    extracting one shared helper, where it will have to be decided once).

    Both clients meet the same first refused request: the claim with the held
    nonce answers ``caller_principal_claimed`` — the session is bound under
    another nonce, which never changes — and both stop. On the NEXT refused
    request the volume, which records that outcome (``principal_claim_outcome
    == "refused"``), sends NO claim: a permanently refused session costs it no
    claim round trip per request, and it reports that it is not claiming
    again. The substrate session keeps no such record and claims again on
    every refused request, reporting the claim's own answer each time. Fails
    when the volume starts claiming again, or the substrate session starts
    short-circuiting, or either reports the other's message."""
    second_wire, second_message = _SECOND_REFUSED_REQUEST[driver.kind]
    script = _Script(
        {
            _CLAIM_ROUTE: [_claim_answer(_BINDS_HELD), _CLAIM_REFUSED, _CLAIM_REFUSED],
            driver.request_route: [_refusal(_FOREIGN)] * 2,
        },
        driver.follow_up,
    )
    with _scripted_coordinator(tmp_path, script):
        first = _run(driver, script, tmp_path, fast_cfg)
        _assert_nothing_leaks(script, first)
        _fail_if_escaped(first)
        first_wire: list[WireRow] = [
            (_CLAIM_ROUTE, None, _SAME_NONCE),
            (driver.request_route, _HELD, None),
            (_CLAIM_ROUTE, None, _SAME_NONCE),
        ]
        assert script.redacted() == first_wire
        _assert_refused(first, _FOREIGN, _refused_message(_FOREIGN, _BOUND_ELSEWHERE), settled=True)
        driver.assert_quiet(first)
        if isinstance(first.client, CoherentVolume):
            assert first.client.principal_claim_outcome == "refused", "the record the volume's short-circuit reads"

        second = _run(driver, script, tmp_path, fast_cfg, client=first.client)

    _assert_nothing_leaks(script, second)
    _fail_if_escaped(second)
    rendered: list[WireRow] = [
        (driver.request_route, value, None) if kind == "request" else (_CLAIM_ROUTE, None, _SAME_NONCE)
        for kind, value in second_wire
    ]
    assert script.redacted() == first_wire + rendered
    _assert_refused(second, _FOREIGN, second_message, settled=True)
    driver.assert_quiet(second)
