# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The one-shot clients' caller-principal recovery (caller-principal plan R20).

The hook client keeps a session's mint nonce and principal on disk under
``.coherence/`` — one process per invocation leaves it no other place.
(``agent-coherence-status --self-test`` does not: it runs as one process and
holds fresh sessions' principals in memory.) A request the coordinator refuses with
a typed ``caller_principal_foreign`` / ``caller_principal_absent`` reason is
recovered by re-claiming with the SAME stored nonce: a principal that differs
from the one presented replaces the stored file (write-then-rename, 0600) and
the refused request is retried ONCE; the same principal, a claim bound under
another nonce, or a claim that does not confirm is reported and stops; a 404
retries once without the header. Nothing re-mints, nothing deletes the nonce,
and nothing prints a principal or a nonce.

What this buys on the hook surface is accident-resistance and a detectable
unbound caller, not separation between callers: every one of these files is
readable by any process of the same OS user.
"""

from __future__ import annotations

import http.server
import io
import json
import os
import secrets
import threading
import time
import urllib.error
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ccs.adapters.claude_code import auth
from ccs.adapters.claude_code.coordinator_server import (
    CoordinatorHTTPServer,
    caller_principal_identity,
)
from ccs.cli import _coherence_client, coherence_hook_client, coherence_status

_PRINCIPAL_HEADER = "Coherence-Caller-Principal"  # frozen duplicate of the wire name
_FOREIGN = "caller_principal_foreign"  # frozen duplicates of the wire reasons
_ABSENT = "caller_principal_absent"
_CLAIMED = "caller_principal_claimed"


def _sid() -> str:
    return str(uuid.uuid4())


def _files(workspace: Path, sid: str) -> tuple[Path, Path]:
    base = workspace / ".coherence" / f"caller-principal-{caller_principal_identity(sid).hex}"
    return base.with_name(base.name + ".nonce"), base.with_name(base.name + ".principal")


def _value(tag: str) -> str:
    """A 43-character principal/nonce-shaped value, distinct per tag and not a
    palindrome (so a reversed or re-cut value can never pass for it)."""
    return (tag * 3 + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP")[:43]


def _drive(
    subcommand: str, payload: dict[str, Any], workspace: Path,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> tuple[str, str]:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert coherence_hook_client.main([subcommand, "--root", str(workspace)]) == 0
    captured = capsys.readouterr()
    return captured.out, captured.err


def _assert_nothing_leaks(text: str, *secrets: str) -> None:
    """R5 on the client side: no principal and no nonce in anything a client
    prints or logs. Asserts against the real values, so a leak cannot hide."""
    for secret in secrets:
        assert secret and secret not in text, "a principal or nonce reached client output"


# --- a scripted coordinator ---------------------------------------------------


class _Scripted(http.server.BaseHTTPRequestHandler):
    """Answers the claim from ``claims`` (one answer per claim, in order) and
    every other route through ``route(header)``; records every request as
    ``(path, header)`` for a route and ``(path, nonce)`` for a claim."""

    seen: list[tuple[str, str | None]] = []
    claims: list[tuple[int, dict[str, Any]]] = []
    route: Any = None
    phrase: str | None = None  # the status line's reason phrase; None = the standard one

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        if self.path == "/principal/claim":
            self.seen.append((self.path, body.get("mint_nonce")))
            status, answer = self.claims.pop(0)
        else:
            header = self.headers.get(_PRINCIPAL_HEADER)
            self.seen.append((self.path, header))
            status, answer = type(self).route(header)
        raw = json.dumps(answer).encode()
        self.send_response(status, type(self).phrase)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *_args: Any) -> None:
        return


def _refusal(reason: str) -> tuple[int, dict[str, Any]]:
    return 400, {"error": f"principal refused ({reason})", "reason": reason}


@pytest.fixture
def scripted(tmp_path: Path):
    """A git workspace wired to a scripted coordinator, with a stored nonce and
    principal for one session already on disk."""
    (tmp_path / ".git").mkdir()
    coherence = tmp_path / ".coherence"
    coherence.mkdir(mode=0o700)
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    (coherence / "server.pid").write_text(f"12345\n{httpd.server_address[1]}\n")
    (coherence / "hook.secret").write_text("test-secret")
    sid = _sid()
    nonce_file, principal_file = _files(tmp_path, sid)
    nonce_file.write_text(_value("N") + "\n")
    principal_file.write_text(_value("P") + "\n")
    _Scripted.seen, _Scripted.claims, _Scripted.route = [], [], None
    _Scripted.phrase = None
    try:
        yield tmp_path, sid
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_refused_stored_principal_is_replaced_by_the_same_nonces_claim_and_retried_once(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The recovery path end to end on the wire: the refused request, ONE
    claim presenting the stored nonce (never a new one), the stored file
    replaced by the principal that claim returned, the request retried ONCE
    under it — and when the coordinator refuses the retry too, reported and
    stopped: no second claim, no third send. The nonce file is untouched and
    the replaced principal file is 0600."""
    workspace, sid = scripted
    nonce_file, principal_file = _files(workspace, sid)
    _Scripted.claims = [(200, {"ok": True, "principal": _value("Q")})]
    _Scripted.route = staticmethod(lambda _h: _refusal(_FOREIGN))

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [
        ("/hooks/session-stop", _value("P")),
        ("/principal/claim", _value("N")),
        ("/hooks/session-stop", _value("Q")),
    ]
    assert out.strip() == "{}"
    assert _FOREIGN in err
    assert principal_file.read_text().strip() == _value("Q")
    assert principal_file.stat().st_mode & 0o777 == 0o600
    assert nonce_file.read_text().strip() == _value("N")
    _assert_nothing_leaks(out + err, _value("P"), _value("Q"), _value("N"))


def test_a_session_id_with_a_trailing_newline_is_never_claimed_for_nor_recovered(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The coordinator's shape check matches with ``$``, which in Python also
    admits one trailing newline; the Node client's does not. So neither client
    claims for such an id — not on the first hook, not in recovery, even with
    a nonce stored for it (as a client that did claim would have left) — and
    the two clients keep sharing one binding per session. The refusal is
    reported by its typed reason."""
    workspace, sid = scripted
    sid_nl = sid + "\n"
    nonce_file, _principal = _files(workspace, sid_nl)
    nonce_file.write_text(_value("N") + "\n")
    _Scripted.claims = [(200, {"ok": True, "principal": _value("Q")})]
    _Scripted.route = staticmethod(lambda _h: _refusal(_ABSENT))

    out, err = _drive("session-stop", {"session_id": sid_nl}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [("/hooks/session-stop", None)], "no claim, no retry"
    assert out.strip() == "{}"
    assert f"({_ABSENT})" in err and "malformed" in err
    assert nonce_file.read_text() == _value("N") + "\n"
    _assert_nothing_leaks(out + err, _value("N"), _value("Q"))


def test_a_recovered_request_is_answered_under_the_new_principal(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """The success arm: the retry under the re-claimed principal is admitted
    and its answer is what the hook prints; nothing is reported."""
    workspace, sid = scripted
    _Scripted.claims = [(200, {"ok": True, "principal": _value("Q")})]
    _Scripted.route = staticmethod(
        lambda h: (200, {"ok": True}) if h == _value("Q") else _refusal(_FOREIGN)
    )

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert json.loads(out) == {"ok": True}
    assert err == ""
    assert [path for path, _ in _Scripted.seen] == [
        "/hooks/session-stop", "/principal/claim", "/hooks/session-stop",
    ]


def test_a_reclaim_that_returns_the_presented_principal_reports_and_stops(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """If the claim hands back the very principal that was refused, the refusal
    is not about staleness: report, and do not retry the request at all."""
    workspace, sid = scripted
    _, principal_file = _files(workspace, sid)
    _Scripted.claims = [(200, {"ok": True, "principal": _value("P")})]
    _Scripted.route = staticmethod(lambda _h: _refusal(_FOREIGN))

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [
        ("/hooks/session-stop", _value("P")),
        ("/principal/claim", _value("N")),
    ]
    assert out.strip() == "{}" and _FOREIGN in err
    assert principal_file.read_text().strip() == _value("P")
    _assert_nothing_leaks(out + err, _value("P"), _value("N"))


def test_a_reclaim_refused_as_claimed_reports_and_never_re_mints(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Another claimant holds the session: report and stop. The stored nonce
    and principal stay exactly as they were — nothing is deleted, no new
    nonce is generated — and the request is not retried."""
    workspace, sid = scripted
    nonce_file, principal_file = _files(workspace, sid)
    _Scripted.claims = [(200, {"ok": False, "reason": _CLAIMED, "detail": "bound"})]
    _Scripted.route = staticmethod(lambda _h: _refusal(_FOREIGN))

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [
        ("/hooks/session-stop", _value("P")),
        ("/principal/claim", _value("N")),
    ]
    assert out.strip() == "{}" and _CLAIMED in err
    assert nonce_file.read_text().strip() == _value("N")
    assert principal_file.read_text().strip() == _value("P")
    _assert_nothing_leaks(out + err, _value("P"), _value("N"))


def test_a_reclaim_whose_answer_is_lost_raises_a_refusal_that_is_not_settled(scripted) -> None:
    """The one-shot client's refusal is typed on ``settled`` like the
    long-lived clients': a reclaim answered with the coordinator's degraded
    claim envelope (its answer lost) stops, but the refusal it raises is NOT
    settled — the next invocation claims again with the stored nonce by
    itself — while a reclaim that hands back the refused principal raises a
    settled one. Neither message carries the principal or the nonce."""
    from ccs.core.exceptions import CallerPrincipalRefused

    workspace, sid = scripted
    endpoint = _coherence_client.resolve_endpoint(workspace)
    _Scripted.route = staticmethod(lambda _h: _refusal(_FOREIGN))
    _Scripted.claims = [(200, {"ok": False, "degraded": True, "reason": "claim_unconfirmed"})]

    with pytest.raises(CallerPrincipalRefused) as lost:
        _coherence_client.post_with_stored_principal(
            endpoint, workspace, "/hooks/session-stop", {"session_id": sid}, report=lambda _m: None
        )

    assert lost.value.reason == _FOREIGN
    assert lost.value.settled is False, "a lost reclaim answer is not a settled refusal"
    assert _Scripted.seen == [
        ("/hooks/session-stop", _value("P")),
        ("/principal/claim", _value("N")),
    ], "the request was retried after an unconfirmed reclaim"

    _Scripted.seen.clear()
    _Scripted.claims = [(200, {"ok": True, "principal": _value("P")})]
    with pytest.raises(CallerPrincipalRefused) as answered:
        _coherence_client.post_with_stored_principal(
            endpoint, workspace, "/hooks/session-stop", {"session_id": sid}, report=lambda _m: None
        )

    assert answered.value.reason == _FOREIGN
    assert answered.value.settled is True, "a reclaim that answered is a settled refusal"
    _assert_nothing_leaks(str(lost.value) + str(answered.value), _value("P"), _value("N"))


def test_a_reclaim_answered_404_retries_once_without_the_header(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A 404 on the claim means the coordinator issues no principals now: the
    refused request is retried once WITHOUT the header."""
    workspace, sid = scripted
    _Scripted.claims = [(404, {"error": "not found"})]
    _Scripted.route = staticmethod(
        lambda h: (200, {"ok": True}) if h is None else _refusal(_FOREIGN)
    )

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [
        ("/hooks/session-stop", _value("P")),
        ("/principal/claim", _value("N")),
        ("/hooks/session-stop", None),
    ]
    assert json.loads(out) == {"ok": True}


def test_an_absent_refusal_after_an_unconfirmed_claim_recovers_with_the_stored_nonce(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """R20 inside one invocation: the claim's answer is lost (the watchdog's
    unconfirmed envelope), so the request goes out with no header and is
    refused as absent — the bind landed. The same stored nonce re-claims, the
    principal is stored, and the request is retried under it."""
    workspace, sid = scripted
    _, principal_file = _files(workspace, sid)
    principal_file.unlink()
    _Scripted.claims = [
        (200, {"ok": False, "degraded": True, "reason": "claim_unconfirmed"}),
        (200, {"ok": True, "principal": _value("Q")}),
    ]
    _Scripted.route = staticmethod(
        lambda h: (200, {"ok": True}) if h == _value("Q") else _refusal(_ABSENT)
    )

    out, _err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [
        ("/principal/claim", _value("N")),
        ("/hooks/session-stop", None),
        ("/principal/claim", _value("N")),
        ("/hooks/session-stop", _value("Q")),
    ]
    assert json.loads(out) == {"ok": True}
    assert principal_file.read_text().strip() == _value("Q")


def test_a_loser_that_gave_up_on_a_stalled_winners_nonce_recovers_once_it_lands(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Two hooks race on a new session. The winner created the nonce file and
    stalled before writing it, past the loser's bounded wait: the loser gives
    up for now (no header) — it does not treat the half-written file as
    final. By the time its request is refused as absent (the winner's claim
    bound the session), the winner's write has landed; the loser's recovery
    reads the nonce THEN, claims with it, and its request is answered."""
    workspace, sid = scripted
    nonce_file, principal_file = _files(workspace, sid)
    principal_file.unlink()
    nonce_file.write_text("")  # the winner's exclusive create; its write has not landed
    monkeypatch.setattr(auth.time, "sleep", lambda _sec: None)  # the wait elapses first
    _Scripted.claims = [(200, {"ok": True, "principal": _value("Q")})]

    def route(header: str | None) -> tuple[int, dict[str, Any]]:
        if header == _value("Q"):
            return 200, {"ok": True}
        nonce_file.write_text(_value("N") + "\n")  # the stalled winner's write lands
        return _refusal(_ABSENT)

    _Scripted.route = staticmethod(route)

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [
        ("/hooks/session-stop", None),
        ("/principal/claim", _value("N")),
        ("/hooks/session-stop", _value("Q")),
    ]
    assert json.loads(out) == {"ok": True}
    assert principal_file.read_text().strip() == _value("Q")
    _assert_nothing_leaks(out + err, _value("N"), _value("Q"))


def test_a_one_shot_request_refused_again_reports_the_second_refusals_reason(scripted) -> None:
    """Refused as ``caller_principal_absent``, recovered with the stored
    nonce, and refused AGAIN as ``caller_principal_foreign``: the typed
    refusal — whose text the hook client prints — carries the retry's reason,
    not the first one's. (Two equal reasons cannot tell which is carried.)"""
    from ccs.core.exceptions import CallerPrincipalRefused

    workspace, sid = scripted
    reasons = iter([_ABSENT, _FOREIGN])
    _Scripted.claims = [(200, {"ok": True, "principal": _value("Q")})]
    _Scripted.route = staticmethod(lambda _h: _refusal(next(reasons)))
    endpoint = _coherence_client.resolve_endpoint(workspace)

    with pytest.raises(CallerPrincipalRefused) as raised:
        _coherence_client.post_with_stored_principal(
            endpoint, workspace, "/hooks/session-stop", {"session_id": sid}, report=lambda _m: None
        )

    assert [path for path, _h in _Scripted.seen] == [
        "/hooks/session-stop", "/principal/claim", "/hooks/session-stop",
    ]
    assert raised.value.reason == _FOREIGN
    assert f"({_FOREIGN})" in str(raised.value) and f"({_ABSENT})" not in str(raised.value)
    assert _coherence_client.PRINCIPAL_REFUSED_AGAIN in str(raised.value)


def test_a_refusal_without_a_stored_nonce_is_reported_and_claims_nothing(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Recovery needs the nonce the binding was made with. Without one on
    disk the client has nothing to re-claim with: report and stop — never
    generate a nonce at recovery time, which would be a second claimant."""
    workspace, sid = scripted
    nonce_file, _ = _files(workspace, sid)
    nonce_file.unlink()
    _Scripted.route = staticmethod(lambda _h: _refusal(_FOREIGN))

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [("/hooks/session-stop", _value("P"))]
    assert out.strip() == "{}" and _FOREIGN in err
    assert not nonce_file.exists()


def test_a_refusal_without_a_typed_reason_is_not_recovered(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """Refusals are classified by the typed ``reason`` field, never by the
    prose: a 400 whose error text names the header and the reason but whose
    body carries no ``reason`` key is an ordinary validation failure — no
    recovery, no principal report, the silent ``{}`` every other rejected
    hook gets."""
    workspace, sid = scripted
    _Scripted.route = staticmethod(
        lambda _h: (400, {"error": f"the {_PRINCIPAL_HEADER} header ({_FOREIGN})"})
    )

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [("/hooks/session-stop", _value("P"))]
    assert out.strip() == "{}"
    assert err == ""


@pytest.mark.parametrize(
    "reason",
    [["caller_principal_foreign"], {"reason": "caller_principal_absent"}, 7, 1.5, True, None, ""],
    ids=["list", "object", "int", "float", "bool", "null", "empty"],
)
def test_a_refusal_reason_that_is_not_a_known_string_classifies_as_no_refusal(
    reason: object,
) -> None:
    """A 400 body's ``reason`` is classified by exact membership in the typed
    refusal vocabulary — and only a string can be a member. A list or an
    object (a proxy, a gateway, a coordinator bug) used to raise
    ``TypeError`` from the membership test itself, escaping every client's
    handling of an ordinary rejected request."""
    body = io.BytesIO(json.dumps({"error": "bad request", "reason": reason}).encode())
    exc = urllib.error.HTTPError("/hooks/pre-edit", 400, "Bad Request", {}, body)  # type: ignore[arg-type]

    assert _coherence_client.principal_refusal_reason(exc) is None


@pytest.mark.parametrize("reason", [_FOREIGN, _ABSENT])
def test_a_typed_refusal_reason_is_classified(reason: str) -> None:
    """Control for the test above: the two typed refusal reasons ARE
    classified, on a 400 only."""
    def error(status: int) -> urllib.error.HTTPError:
        body = io.BytesIO(json.dumps({"error": "refused", "reason": reason}).encode())
        return urllib.error.HTTPError("/hooks/pre-edit", status, "x", {}, body)  # type: ignore[arg-type]

    assert _coherence_client.principal_refusal_reason(error(400)) == reason
    assert _coherence_client.principal_refusal_reason(error(403)) is None


def test_no_coordinator_supplied_text_reaches_the_hook_clients_stderr(
    scripted, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A coordinator that echoes the nonce and the principals back in every
    free-text field — the claim's ``reason``, ``detail`` and ``error``, a
    refusal's ``error`` and ``detail``, the status line's reason phrase —
    cannot get any of them onto the hook client's stderr. What the client
    reports on the claim and recovery paths is built from the frozen
    vocabulary: a known reason by name, anything else as ``unrecognised``, an
    HTTP failure by its status code.

    The session has a stored nonce and no stored principal: the first claim is
    answered 500, the request is refused as absent, and the recovery claim
    (the stored nonce, as always) is answered with the echo as its reason."""
    workspace, sid = scripted
    _, principal_file = _files(workspace, sid)
    principal_file.unlink()
    echo = " ".join((_value("N"), _value("P"), _value("Q")))
    garbled = {"ok": False, "reason": echo, "detail": echo, "error": echo}
    _Scripted.claims = [(500, garbled), (200, garbled)]
    _Scripted.route = staticmethod(
        lambda _h: (400, {"error": echo, "detail": echo, "reason": _ABSENT})
    )
    _Scripted.phrase = echo

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    assert _Scripted.seen == [
        ("/principal/claim", _value("N")),
        ("/hooks/session-stop", None),
        ("/principal/claim", _value("N")),
    ]
    assert out.strip() == "{}"
    assert f"({_ABSENT})" in err and "unrecognised" in err
    _assert_nothing_leaks(out + err, _value("N"), _value("P"), _value("Q"))


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ({"ok": False, "degraded": True, "reason": "claim_unconfirmed"}, "reason=claim_unconfirmed"),
        ({"ok": False, "reason": _value("N")}, "reason=unrecognised"),
        ({"ok": False, "reason": [_value("N")]}, "reason=unrecognised"),
        ({"ok": True, "principal": ""}, "reason=unrecognised"),
    ],
    ids=["watchdog", "echoed-nonce", "non-string", "no-principal"],
)
def test_an_unconfirmed_claim_names_only_a_known_reason(
    scripted, answer: dict[str, Any], expected: str,
) -> None:
    """The detail of a claim that did not confirm repeats the coordinator's
    ``reason`` only when it is a known token — the watchdog's
    ``claim_unconfirmed`` — and reads ``unrecognised`` for anything else."""
    workspace, sid = scripted
    _Scripted.claims = [(200, answer)]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    claim = _coherence_client.claim_caller_principal(endpoint, sid, _value("N"))

    assert claim.outcome == "unconfirmed"
    assert expected in claim.detail
    assert _value("N") not in claim.detail


@pytest.mark.parametrize(
    ("status", "answer", "token"),
    [
        (503, {"error": "watchdog queue overloaded"}, "unrecognised"),
        (503, {"ok": False, "reason": "claim_unconfirmed"}, "claim_unconfirmed"),
        (500, {"reason": _value("N"), "error": _value("N")}, "unrecognised"),
        (500, {"reason": [_value("N")], "detail": _value("N")}, "unrecognised"),
        (500, _value("N"), "unrecognised"),
        (409, {"ok": False, "reason": _CLAIMED, "detail": _value("N")}, _CLAIMED),
    ],
    ids=["queue-overflow", "known-reason", "echoed-reason", "non-string", "not-an-object",
         "claimed-off-contract"],
)
def test_a_claim_answered_with_an_error_status_names_its_status_and_only_a_known_reason(
    scripted, status: int, answer: Any, token: str,
) -> None:
    """A claim answered with an error status (not 404) did not confirm: its
    detail names the status and the answer's ``reason`` as a known token, or
    ``unrecognised``, and nothing else of the answer. A
    ``caller_principal_claimed`` reason on such a status is reported, not taken
    for the claim contract's refusal, which is a 200 body.

    The Node client decides and reports these answers the same way (status and
    token), so the two clients cannot tell an operator different things about
    one coordinator answer."""
    workspace, sid = scripted
    _Scripted.claims = [(status, answer)]
    _Scripted.phrase = _value("N")
    endpoint = _coherence_client.resolve_endpoint(workspace)

    claim = _coherence_client.claim_caller_principal(endpoint, sid, _value("N"))

    assert claim.outcome == "unconfirmed"
    assert claim.detail == f"claim answered HTTP {status}, reason={token}"


@pytest.mark.parametrize(
    ("answer", "outcome"),
    [
        ({"ok": True, "principal": _value("P")}, "bound"),
        ({"ok": False, "reason": _CLAIMED, "detail": _value("N")}, "refused"),
    ],
    ids=["bound", "claimed"],
)
def test_any_2xx_claim_answer_is_the_claim_contract(
    scripted, answer: dict[str, Any], outcome: str,
) -> None:
    """A 201 carries the claim contract like a 200: a principal binds, a
    ``caller_principal_claimed`` refuses. The Node client applies the same
    2xx rule, so a coordinator answering 201 binds one session identically
    under both clients."""
    workspace, sid = scripted
    _Scripted.claims = [(201, answer)]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    claim = _coherence_client.claim_caller_principal(endpoint, sid, _value("N"))

    assert claim.outcome == outcome
    assert claim.principal == (_value("P") if outcome == "bound" else None)


def test_the_watchdog_claim_reason_is_the_coordinators() -> None:
    """The client's copy of the watchdog-degraded claim reason is a twin of
    the coordinator's envelope; the two must not drift apart."""
    from ccs.adapters.claude_code import coordinator_server

    assert coordinator_server._PRINCIPAL_CLAIM_DEGRADED_RESPONSE["reason"] == "claim_unconfirmed"
    assert "claim_unconfirmed" in _coherence_client.REPORTABLE_CLAIM_REASONS


# --- against a real coordinator: the binding store is reset under a session ---


def _start(workspace: Path) -> CoordinatorHTTPServer:
    server = CoordinatorHTTPServer(workspace, port=0, instance_id="principal-recovery")
    server.serve_in_thread()
    (workspace / ".coherence" / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
    return server


def _reset_store(server: CoordinatorHTTPServer, workspace: Path) -> CoordinatorHTTPServer:
    """Stop the coordinator, delete ``state.db`` (the purge docs/security.md
    describes, and the corrupted-store recovery), and start a fresh one. The
    ``caller-principal-*`` files stay behind."""
    server.shutdown()
    for suffix in ("", "-wal", "-shm"):
        (workspace / ".coherence" / f"state.db{suffix}").unlink(missing_ok=True)
    return _start(workspace)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "plan.md").write_text("plan v1")
    return tmp_path


def test_a_stored_principal_that_outlived_its_binding_store_is_recovered(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """After ``state.db`` is deleted and recreated, a session that lives on
    (a long session, ``--resume``) presents a principal the fresh store never
    bound — refused as foreign on EVERY route. Its first hook re-claims with
    the stored nonce, which binds the session afresh; the stored principal is
    replaced by the new binding and the hook is answered. Every later hook
    presents the new one: exactly one claim for the whole recovery. Without
    this, coherence stayed off for that session on every route, for good."""
    server = _start(workspace)
    sid = _sid()
    edit = {"session_id": sid, "tool_input": {"file_path": str(workspace / "plan.md")}}
    try:
        out, _ = _drive("pre-edit", edit, workspace, monkeypatch, capsys)
        assert json.loads(out).get("ok") is True
        nonce_file, principal_file = _files(workspace, sid)
        nonce, old = nonce_file.read_text(), principal_file.read_text().strip()
        server = _reset_store(server, workspace)
        claims_before = server.endpoint_counters_snapshot()["principal_claim_total"]

        answers, errs = [], []
        for sub in ("pre-read", "pre-edit", "post-edit", "session-stop"):
            if sub == "post-edit":
                (workspace / "plan.md").write_text("plan v2")
            out, err = _drive(sub, edit, workspace, monkeypatch, capsys)
            answers.append(json.loads(out))
            errs.append(err)

        assert answers[0].get("status") == "fresh", answers
        assert all(a.get("ok") is True for a in answers[1:]), answers
        bound = server.registry.get_caller_principal(caller_principal_identity(sid))
        assert bound is not None and bound != old
        assert principal_file.read_text().strip() == bound
        assert principal_file.stat().st_mode & 0o777 == 0o600
        assert nonce_file.read_text() == nonce, "the nonce is never replaced"
        claims = server.endpoint_counters_snapshot()["principal_claim_total"] - claims_before
        assert claims == 1
        _assert_nothing_leaks("".join(errs), old, bound, nonce.strip())
    finally:
        server.shutdown()


def test_the_self_test_claims_fresh_sessions_and_never_depends_on_stored_principals(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """``--self-test`` drives two synthetic sessions that are FRESH on every
    run and holds their principals in memory for the run, so it does not
    depend on anything a previous run left in ``.coherence/``: deleting the
    ``caller-principal-*`` files, or resetting ``state.db``, never breaks a
    later run (with fixed session ids, a run after the files were deleted
    claimed with a new nonce, was refused as ``caller_principal_claimed``, and
    failed at pre-edit). It stores no principal file and prints no principal
    or nonce."""
    claimed: list[tuple[str, str, str | None]] = []
    real_claim = _coherence_client.claim_caller_principal

    def spy(endpoint: Any, session_id: str, nonce: str) -> Any:
        claim = real_claim(endpoint, session_id, nonce)
        claimed.append((session_id, nonce, claim.principal))
        return claim

    monkeypatch.setattr(coherence_status, "claim_caller_principal", spy, raising=False)
    server = _start(workspace)
    try:
        assert coherence_status._run_self_test(workspace) == 0, capsys.readouterr().err
        # What a hook of some real session left behind, deleted with the rest.
        _drive("session-stop", {"session_id": _sid()}, workspace, monkeypatch, capsys)
        for stored in (workspace / ".coherence").glob("caller-principal-*"):
            stored.unlink()
        assert coherence_status._run_self_test(workspace) == 0, capsys.readouterr().err
        server = _reset_store(server, workspace)
        assert coherence_status._run_self_test(workspace) == 0, capsys.readouterr().err
        captured = capsys.readouterr()

        sessions = [sid for sid, _n, _p in claimed]
        assert len(sessions) == 6 and len(set(sessions)) == 6, "two fresh sessions per run"
        assert all(p for _s, _n, p in claimed), "every run's sessions were bound"
        assert list((workspace / ".coherence").glob("caller-principal-*")) == []
        values = [v for _s, n, p in claimed for v in (n, p) if v]
        _assert_nothing_leaks(captured.out + captured.err, *values)
    finally:
        server.shutdown()


def test_a_torn_stored_principal_is_repaired_by_one_claim(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A ``.principal`` left empty (a write that never landed, or a
    truncation) reads as absent; the next hook claims with the stored nonce
    and gets the bound principal back. That claim REPAIRS the file, so the
    torn file costs one round trip, not one per hook for the rest of the
    session. Control: three hooks with an intact file claim nothing."""
    server = _start(workspace)
    sid = _sid()
    try:
        _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)
        nonce_file, principal_file = _files(workspace, sid)
        bound = principal_file.read_text().strip()

        def claims_over_three_hooks() -> int:
            before = server.endpoint_counters_snapshot()["principal_claim_total"]
            for _ in range(3):
                out, _err = _drive(
                    "session-stop", {"session_id": sid}, workspace, monkeypatch, capsys
                )
                assert json.loads(out).get("ok") is True, out
            return server.endpoint_counters_snapshot()["principal_claim_total"] - before

        assert claims_over_three_hooks() == 0, "control: an intact file claims nothing"
        principal_file.write_text("")
        assert claims_over_three_hooks() == 1
        assert principal_file.read_text().strip() == bound
        assert principal_file.stat().st_mode & 0o777 == 0o600
    finally:
        server.shutdown()


# --- the mint-nonce file: an interrupted write is never overwritten ---------


# Ages are literals on BOTH sides of the 2 s grace (never derived from the
# constant): an age just past it pins that the grace is not ten times longer,
# an age just inside it that it is not ten times shorter.
@pytest.mark.parametrize(
    ("age_sec", "waits"),
    [(0.0, True), (1.0, True), (3.0, False), (3600.0, False)],
    ids=["young-0s", "young-1s", "old-3s", "old-3600s"],
)
def test_a_torn_nonce_file_is_waited_on_only_while_it_could_still_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, age_sec: float, waits: bool,
) -> None:
    """An empty nonce file is a racer's exclusive create whose write has not
    landed YET — or never will (killed, ENOSPC). A young one is waited on
    (the bounded retry: a loser must not treat a winner's in-progress write as
    final); an old one is treated as abandoned and reported at once instead of
    charging every later hook of the session the whole bounded wait — the
    outcome after the wait would be the same. Either way the file is never
    overwritten or removed.

    The two reports say different things. An abandoned write stays broken
    until someone acts, so its report names the file and the step. A young
    one may still be completed by its live writer, so its report must not
    tell the operator the session runs without a principal until the file is
    fixed: only this invocation proceeds without one, and the next either
    adopts the nonce or reports an interrupted write."""
    (tmp_path / ".coherence").mkdir(mode=0o700)
    key = caller_principal_identity(_sid()).hex
    nonce_file = tmp_path / ".coherence" / f"caller-principal-{key}.nonce"
    nonce_file.touch(mode=0o600)
    stamp = time.time() - age_sec
    os.utime(nonce_file, (stamp, stamp))
    sleeps: list[float] = []
    monkeypatch.setattr(auth.time, "sleep", sleeps.append)

    with pytest.raises(auth.MintNonceUnavailable) as caught:
        auth.ensure_mint_nonce(tmp_path, key)

    message = str(caught.value)
    assert bool(sleeps) is waits
    assert nonce_file.read_text() == ""
    # Named in full, as the hook.secret error is and as the Node client prints
    # it (byte for byte: the frozen reports below): nothing repairs the file
    # automatically, and only the abandoned report names the removal step.
    if waits:
        assert message == _YOUNG_NONCE_REPORT.format(path=nonce_file)
    else:
        assert message == _ABANDONED_NONCE_REPORT.format(path=nonce_file)


# --- the mint-nonce file: the grace window, pinned on both sides of its edge ---
#
# FROZEN duplicates of the reports the Node client's own tests pin byte for byte
# (``youngMessage`` / ``abandonedMessage`` in the sibling's
# ``src/test/caller_principal.test.ts``) — never built from the constants under
# test. The two clients print the same text for the same file.

_YOUNG_NONCE_REPORT = (
    "{path} exists but its write was still in progress across 5 attempts; not overwriting it. "
    "This invocation proceeds without a principal; a later one adopts the nonce if that write "
    "lands, or reports {path} as an interrupted write once it has stayed incomplete for 2 s."
)
_ABANDONED_NONCE_REPORT = (
    "{path} exists but holds no complete nonce (an interrupted write); not overwriting it. "
    "The session runs without a principal until it is fixed: remove {path} by hand if no "
    "hook of this session is running."
)
_BOUNDED_WAITS = 4  # 5 attempts, a wait between each (frozen, like the text above)


def _torn_nonce(tmp_path: Path) -> tuple[Path, str, float]:
    """An empty nonce file — an exclusive create whose write has not landed —
    and a whole second a minute ago to stamp it with, so a pinned clock gives
    an exact age."""
    (tmp_path / ".coherence").mkdir(mode=0o700)
    key = caller_principal_identity(_sid()).hex
    nonce_file = tmp_path / ".coherence" / f"caller-principal-{key}.nonce"
    nonce_file.touch(mode=0o600)
    return nonce_file, key, float(int(time.time()) - 60)


def _pin_clock(
    monkeypatch: pytest.MonkeyPatch, now: float, *, each_wait_advances: float = 0.0
) -> list[float]:
    """Pin the clock ``auth`` reads to ``now``; each bounded wait is recorded
    (and returned) and advances the pinned clock by ``each_wait_advances``."""
    clock = [now]
    waits: list[float] = []

    def wait(seconds: float) -> None:
        waits.append(seconds)
        clock[0] += each_wait_advances

    monkeypatch.setattr(auth, "time", SimpleNamespace(time=lambda: clock[0], sleep=wait))
    return waits


def _ensure_fails(tmp_path: Path, key: str) -> str:
    with pytest.raises(auth.MintNonceUnavailable) as caught:
        auth.ensure_mint_nonce(tmp_path, key)
    return str(caught.value)


@pytest.mark.parametrize(
    ("age_sec", "young"),
    [(1.999, True), (2.0, True), (2.001, False), (2.1, False)],
    ids=["1.999s-young", "2.000s-young", "2.001s-old", "2.100s-old"],
)
def test_the_torn_nonce_grace_edge_is_two_seconds_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, age_sec: float, young: bool,
) -> None:
    """The grace is 2 s and exclusive, to the millisecond, as the Node
    client's is: a torn file aged 1.999 s or exactly 2 s is young — the whole
    bounded wait, then the young report — and one aged 2.001 s or 2.1 s is
    abandoned — reported at once, with the removal step. The clock is pinned,
    so the age the rule computes is exactly the one named. An inclusive edge,
    or a grace moved anywhere off 2 s by more than a millisecond, turns an arm
    red; so does either report's text drifting from the Node client's."""
    nonce_file, key, stamp = _torn_nonce(tmp_path)
    os.utime(nonce_file, (stamp, stamp))
    assert nonce_file.stat().st_mtime == stamp, "precondition: the stamp is exact"
    waits = _pin_clock(monkeypatch, stamp + age_sec)

    message = _ensure_fails(tmp_path, key)

    if young:
        assert len(waits) == _BOUNDED_WAITS
        assert message == _YOUNG_NONCE_REPORT.format(path=nonce_file)
    else:
        assert waits == []
        assert message == _ABANDONED_NONCE_REPORT.format(path=nonce_file)
    assert nonce_file.read_text() == "", "left exactly as it was"


def test_the_torn_nonce_age_is_re_read_at_every_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file 1.99 s old at the first attempt is young and waited on; the wait
    takes 20 ms, so at the second attempt it is 2.01 s old and is reported as
    abandoned THEN — one wait, not the whole bounded wait. A rule that judged
    the age once, at the first attempt, would wait four times and call it
    young."""
    nonce_file, key, stamp = _torn_nonce(tmp_path)
    os.utime(nonce_file, (stamp, stamp))
    waits = _pin_clock(monkeypatch, stamp + 1.99, each_wait_advances=0.02)

    message = _ensure_fails(tmp_path, key)

    assert len(waits) == 1
    assert message == _ABANDONED_NONCE_REPORT.format(path=nonce_file)


@pytest.mark.parametrize(
    ("atime_offset", "mtime_offset", "young"),
    [(59.5, 0.0, False), (0.0, 59.5, True)],
    ids=["written-a-minute-ago-read-just-now", "written-just-now-read-a-minute-ago"],
)
def test_the_torn_nonce_age_is_the_time_since_it_was_last_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    atime_offset: float, mtime_offset: float, young: bool,
) -> None:
    """The age is measured from the file's last WRITE (mtime) — not its last
    access (atime) and not its last metadata change (ctime, which is "now"
    here). Every other test stamps atime equal to mtime, which cannot tell
    them apart; here they differ by almost a minute, with the clock pinned a
    minute after the whole-second stamp."""
    nonce_file, key, stamp = _torn_nonce(tmp_path)
    os.utime(nonce_file, (stamp + atime_offset, stamp + mtime_offset))
    assert nonce_file.stat().st_mtime == stamp + mtime_offset, "precondition"
    waits = _pin_clock(monkeypatch, stamp + 60.0)

    message = _ensure_fails(tmp_path, key)

    if young:
        assert len(waits) == _BOUNDED_WAITS
        assert message == _YOUNG_NONCE_REPORT.format(path=nonce_file)
    else:
        assert waits == []
        assert message == _ABANDONED_NONCE_REPORT.format(path=nonce_file)


def _reported_young(message: str, waits: list[float], nonce_file: Path) -> bool:
    """Whether ``ensure_mint_nonce`` treated the torn file as young: the whole
    bounded wait, then the young report."""
    return len(waits) == _BOUNDED_WAITS and message == _YOUNG_NONCE_REPORT.format(path=nonce_file)


# The Node client's rows for the next four tests say they hold "as in the Python
# rule"; these are that rule's own pins, so the two cannot drift apart unseen.


@pytest.mark.parametrize(
    ("mtime_fraction", "age_sec", "young"),
    [(0.75, 1.5, True), (0.6, 2.1, False)],
    ids=["fraction-1.5s-young", "fraction-2.1s-old"],
)
def test_the_torn_nonce_age_is_read_from_the_mtime_at_full_precision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mtime_fraction: float, age_sec: float, young: bool,
) -> None:
    """Every other test stamps a whole second; here the mtime has a fraction,
    and the age is measured from it exactly. Stamped S+0.75 and read 1.5 s
    later the file is young — truncated to S it would read 2.25 s and be
    reported abandoned, telling the operator to remove a file whose writer
    may still be alive. Stamped S+0.6 and read 2.1 s later it is abandoned —
    rounded up to S+1 it would read 1.7 s and be waited on."""
    nonce_file, key, stamp = _torn_nonce(tmp_path)
    mtime = stamp + mtime_fraction
    os.utime(nonce_file, (mtime, mtime))
    # Within a microsecond: 0.6 has no exact binary form, and a filesystem
    # keeping whole seconds fails here loudly rather than passing silently.
    assert abs(nonce_file.stat().st_mtime - mtime) < 1e-6, "precondition: the fraction is stored"
    waits = _pin_clock(monkeypatch, mtime + age_sec)

    message = _ensure_fails(tmp_path, key)

    if young:
        assert _reported_young(message, waits, nonce_file)
    else:
        assert waits == []
        assert message == _ABANDONED_NONCE_REPORT.format(path=nonce_file)


def test_a_symlinked_torn_nonce_is_aged_by_the_file_it_points_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The age is the linked file's — stat, not lstat: a young, still-empty
    target behind a link stamped a minute ago is waited on, and neither the
    link nor its target is touched."""
    nonce_file, key, stamp = _torn_nonce(tmp_path)
    target = nonce_file.with_name(nonce_file.name + ".target")
    nonce_file.rename(target)
    os.utime(target, (stamp + 59, stamp + 59))
    nonce_file.symlink_to(target)
    os.utime(nonce_file, (stamp, stamp), follow_symlinks=False)
    assert nonce_file.lstat().st_mtime == stamp, "precondition: the link itself is a minute old"
    waits = _pin_clock(monkeypatch, stamp + 60)

    message = _ensure_fails(tmp_path, key)

    assert _reported_young(message, waits, nonce_file)
    assert nonce_file.is_symlink() and target.read_text() == ""


def test_a_torn_nonce_whose_age_cannot_be_read_is_waited_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dangling link: the file exists as a name, but there is no age to
    read. Waiting is the safe default — the file is not reported abandoned
    on an age the rule never saw — and nothing is created at the target."""
    nonce_file, key, stamp = _torn_nonce(tmp_path)
    missing = nonce_file.with_name("missing")
    nonce_file.unlink()
    nonce_file.symlink_to(missing)
    waits = _pin_clock(monkeypatch, stamp + 3600)

    message = _ensure_fails(tmp_path, key)

    assert _reported_young(message, waits, nonce_file)
    assert not missing.exists()


def test_a_torn_nonce_stamped_in_the_future_is_waited_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An mtime ten minutes ahead of the clock (the clock stepped back): a
    negative age is not past the grace, so it is waited on — never read as
    ten minutes old."""
    nonce_file, key, stamp = _torn_nonce(tmp_path)
    os.utime(nonce_file, (stamp, stamp))
    waits = _pin_clock(monkeypatch, stamp - 600)

    message = _ensure_fails(tmp_path, key)

    assert _reported_young(message, waits, nonce_file)


def test_the_torn_nonce_age_is_not_the_files_creation_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mtime is stamped an hour AFTER the file was created and the clock
    pinned 1 s after that: the mtime reads young, the creation time an hour
    old. Every other test stamps the mtime before creation, which on macOS
    also lowers the birthtime to it, so a birthtime rule would pass them.
    Where the platform reports no birthtime there is nothing to confuse, and
    the young outcome is asserted all the same."""
    nonce_file, key, _stamp = _torn_nonce(tmp_path)
    ahead = float(int(time.time()) + 3600)
    os.utime(nonce_file, (ahead, ahead))
    birth = getattr(nonce_file.stat(), "st_birthtime", None)
    if birth is not None:
        assert ahead + 1 - birth > 2.0, "precondition: the birthtime reads past the grace"
    waits = _pin_clock(monkeypatch, ahead + 1)

    message = _ensure_fails(tmp_path, key)

    assert _reported_young(message, waits, nonce_file)


_UNAVAILABLE_REPORT = "caller principal unavailable: no usable mint nonce ({message})"
"""FROZEN duplicate of the line the Node client's tests pin around either
report (``unavailableReport`` in the sibling's
``src/test/caller_principal.test.ts``) — never built from the client's own
f-string."""


@pytest.mark.parametrize(
    ("age_sec", "report"),
    [(0.0, _YOUNG_NONCE_REPORT), (60.0, _ABANDONED_NONCE_REPORT)],
    ids=["young", "old"],
)
def test_a_torn_nonce_file_reaches_the_hook_report_as_the_node_client_words_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, age_sec: float, report: str,
) -> None:
    """The one line a hook prints when no mint nonce is usable, byte for byte:
    the Node client pins the same frozen line, so a rewording of the wrapper
    on either side alone turns that side red. Nothing is claimed (no principal
    is presented) and the file is left exactly as it was."""
    session_id = _sid()
    (tmp_path / ".coherence").mkdir(mode=0o700)
    key = caller_principal_identity(session_id).hex
    nonce_file = tmp_path / ".coherence" / f"caller-principal-{key}.nonce"
    nonce_file.touch(mode=0o600)
    stamp = float(int(time.time()) - 60)
    os.utime(nonce_file, (stamp, stamp))
    _pin_clock(monkeypatch, stamp + age_sec)
    reports: list[str] = []

    principal = _coherence_client.obtain_stored_principal(
        _coherence_client.CoordinatorEndpoint(port=9, bearer="s" * 32),
        tmp_path, session_id, reports.append,
    )

    assert principal is None
    assert reports == [_UNAVAILABLE_REPORT.format(message=report.format(path=nonce_file))]
    assert nonce_file.read_text() == ""
    assert sorted(p.name for p in (tmp_path / ".coherence").iterdir()) == [nonce_file.name]


# --- malformed answers, redirects: a transport failure names only its type -----


class _RawCoordinator:
    """A loopback listener that answers each request with the next canned RAW
    response — bytes sent verbatim, then the connection closed — so it can send
    what no HTTP server library would: a status line that is not HTTP, a body
    shorter than its Content-Length. Records ``(path, principal header, mint
    nonce)`` for every request."""

    def __init__(self, responses: list[bytes]) -> None:
        import socket

        self.responses = list(responses)
        self.seen: list[tuple[str, str | None, str | None]] = []
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _addr = self._sock.accept()
            except OSError:
                return
            with conn:
                self._answer(conn)

    def _answer(self, conn: Any) -> None:
        raw = b""
        while b"\r\n\r\n" not in raw:
            chunk = conn.recv(65536)
            if not chunk:
                return
            raw += chunk
        head, _sep, body = raw.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        headers = {k.strip().lower(): v.strip() for k, _s, v in (h.partition(":") for h in lines[1:])}
        while len(body) < int(headers.get("content-length", "0")):
            body += conn.recv(65536)
        nonce = json.loads(body).get("mint_nonce") if body else None
        self.seen.append((lines[0].split(" ")[1], headers.get(_PRINCIPAL_HEADER.lower()), nonce))
        conn.sendall(self.responses.pop(0))

    def close(self) -> None:
        self._sock.close()


def _http_answer(status: int, body: object, headers: dict[str, str] | None = None) -> bytes:
    raw = json.dumps(body).encode()
    lines = [f"HTTP/1.1 {status} X", "Content-Type: application/json",
             f"Content-Length: {len(raw)}", "Connection: close"]
    lines += [f"{k}: {v}" for k, v in (headers or {}).items()]
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + raw


def _not_http(echo: str) -> bytes:
    """A status line that is not HTTP, carrying ``echo`` (http.client's
    ``BadStatusLine`` text IS this line)."""
    return f"NOTHTTP 200 {echo}\r\n\r\n".encode()


def _cut_short(echo: str) -> bytes:
    """A 200 whose body stops long before its Content-Length (``IncompleteRead``
    holds the partial body)."""
    return (
        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 400\r\n\r\n"
        + f'{{"ok": true, "principal": "{echo}'.encode()
    )


def _redirect(echo: str, code: int = 302) -> bytes:
    """A redirect whose well-formed ``Location`` carries ``echo``."""
    return _http_answer(code, {}, {"Location": f"http://127.0.0.1:1/{echo}"})


def _bad_location(echo: str, code: int = 302) -> bytes:
    """A redirect whose ``Location`` does not parse and carries ``echo``: the
    bracketed host is not an IP address, and the stdlib's ``ValueError`` for
    it quotes the host — ``echo`` included."""
    return _http_answer(code, {}, {"Location": _bad_url(echo)})


def _bad_url(echo: str) -> str:
    return f"http://[x{echo}]/"


@pytest.fixture
def raw_workspace(tmp_path: Path):
    """A git workspace whose ``server.pid`` points at a :class:`_RawCoordinator`
    (its responses set by the test via ``raw.responses``)."""
    (tmp_path / ".git").mkdir()
    (tmp_path / ".coherence").mkdir(mode=0o700)
    raw = _RawCoordinator([])
    (tmp_path / ".coherence" / "server.pid").write_text(f"12345\n{raw.port}\n")
    (tmp_path / ".coherence" / "hook.secret").write_text("test-secret")
    try:
        yield tmp_path, raw
    finally:
        raw.close()


def _rendered(exc: BaseException) -> str:
    """What a traceback or ``logger.exception`` prints for ``exc``: its whole
    chain, as Python renders it."""
    import traceback

    return "".join(traceback.format_exception(exc))


@pytest.mark.parametrize(("answer", "kind"), [(_not_http, "BadStatusLine"), (_cut_short, "IncompleteRead")])
def test_a_malformed_claim_answer_is_unconfirmed_and_names_only_its_type(
    raw_workspace, answer: Any, kind: str,
) -> None:
    """``claim_caller_principal`` promises that transport-shaped failures come
    back as ``unconfirmed``. A status line that is not HTTP, or a body cut
    short, is one — urllib does not wrap it, and it used to escape as the raw
    ``http.client`` exception, whose text is the very line the coordinator
    sent (here, the nonce echoed back). The detail names the failure's type."""
    workspace, raw = raw_workspace
    raw.responses = [answer(_value("N"))]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    claim = _coherence_client.claim_caller_principal(endpoint, _sid(), _value("N"))

    assert claim.outcome == "unconfirmed"
    assert kind in claim.detail
    assert _value("N") not in claim.detail


def test_a_malformed_answer_to_a_request_names_only_its_type_down_its_whole_chain(
    raw_workspace,
) -> None:
    """A request presenting a principal is answered with a status line that
    echoes it: :func:`post` raises ``CoordinatorUnavailable`` — the transport
    failure type every caller already handles — naming the malformed answer by
    type, and neither its message nor its chain carries the line."""
    workspace, raw = raw_workspace
    raw.responses = [_not_http(_value("P"))]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    with pytest.raises(_coherence_client.CoordinatorUnavailable) as raised:
        _coherence_client.post(
            endpoint, "/hooks/pre-edit", {"session_id": _sid()},
            extra_headers={_PRINCIPAL_HEADER: _value("P")},
        )

    rendered = _rendered(raised.value)
    assert "BadStatusLine" in rendered
    _assert_nothing_leaks(rendered, _value("P"))


@pytest.mark.parametrize(
    "answer", [_not_http, _cut_short, _redirect, _bad_location],
    ids=["not-http", "cut-short", "redirect", "redirect-bad-location"],
)
def test_a_hook_whose_claim_answer_is_malformed_still_sends_its_request_without_a_principal(
    raw_workspace, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    answer: Any,
) -> None:
    """The first hook of a session claims; the claim's answer is malformed or
    a redirect (echoing the nonce). The claim is unconfirmed — a redirect is
    never followed, and only a 2xx carries the claim contract, as the Node
    client decides it — so the hook goes on WITHOUT a principal: the request
    is sent and its answer printed, as the Node client does, instead of the
    hook dropping its request. The nonce stays on disk for the next hook's
    claim, and nothing prints it."""
    workspace, raw = raw_workspace
    sid = _sid()
    nonce_file, _principal = _files(workspace, sid)
    raw.responses = [answer("NONCE-ECHO"), _http_answer(200, {"ok": True})]

    out, err = _drive("session-stop", {"session_id": sid}, workspace, monkeypatch, capsys)

    nonce = nonce_file.read_text().strip()
    assert [(path, header) for path, header, _n in raw.seen] == [
        ("/principal/claim", None), ("/hooks/session-stop", None),
    ]
    assert raw.seen[0][2] == nonce, "the claim presented the stored nonce"
    assert json.loads(out) == {"ok": True}
    _assert_nothing_leaks(out + err, nonce)


@pytest.mark.parametrize(
    ("answer", "kind"),
    [(_not_http, "BadStatusLine"), (_redirect, "HTTP 302"), (_bad_location, "HTTP 302")],
    ids=["not-http", "redirect", "redirect-bad-location"],
)
def test_the_self_test_reports_a_malformed_claim_answer_as_unconfirmed(
    raw_workspace, capsys: pytest.CaptureFixture[str], answer: Any, kind: str,
) -> None:
    """``--self-test`` against a coordinator whose claim answer is malformed,
    or a redirect: it reports the claim as unconfirmed and exits 3 — no
    traceback, and nothing the answer carried (the nonce the self-test sent,
    echoed into the status line or the ``Location``)."""
    workspace, raw = raw_workspace
    sent: list[str] = []
    real_token = coherence_status.secrets.token_urlsafe

    def recorded_token(n: int) -> str:
        sent.append(real_token(n))
        return sent[-1]

    raw.responses = [answer("ECHO")]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(coherence_status.secrets, "token_urlsafe", recorded_token)
        rc = coherence_status._run_self_test(workspace)
    captured = capsys.readouterr()

    assert rc == 3
    assert "unconfirmed" in captured.err and kind in captured.err
    assert "Traceback" not in captured.err
    assert raw.seen[0][2] == sent[0]
    _assert_nothing_leaks(captured.out + captured.err, *sent, "ECHO")


@pytest.mark.parametrize("code", [302, 308])
@pytest.mark.parametrize("answer", [_redirect, _bad_location], ids=["redirect", "bad-location"])
def test_a_redirected_claim_is_unconfirmed_and_names_no_location(
    raw_workspace, answer: Any, code: int,
) -> None:
    """A claim answered 3xx with a ``Location`` that echoes the nonce it
    carried is never followed, and — as the Node client decides any answer
    outside 2xx — it is an ``unconfirmed`` claim, which the next claim with
    the SAME nonce settles; it does not raise. Its detail names the status,
    never the ``Location``.

    A later request carrying no principal material, redirected with the
    same echo, is refused naming no ``Location`` either: it sent no nonce,
    but what answers it had been sent one. Before, such a request's refusal
    quoted its ``Location`` — and with it the nonce the claim sent."""
    from ccs.core.exceptions import RedirectRefused

    workspace, raw = raw_workspace
    later = _redirect(_value("N"), 307)
    assert _value("N").encode() in later, "control: the later redirect echoes the nonce"
    raw.responses = [answer(_value("N"), code), later]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    claim = _coherence_client.claim_caller_principal(endpoint, _sid(), _value("N"))
    with pytest.raises(RedirectRefused) as later_refusal:
        _coherence_client.get(endpoint, "/status")

    assert claim.outcome == "unconfirmed" and claim.principal is None
    assert f"HTTP {code}" in claim.detail
    _assert_nothing_leaks(claim.detail, _value("N"))
    assert [header for _path, header, _n in raw.seen] == [None, None], "control: no principal sent"
    assert later_refusal.value.status == 307, "control: the later request was redirected"
    assert later_refusal.value.location == _coherence_client.REDIRECT_LOCATION_WITHHELD
    _assert_nothing_leaks(_rendered(later_refusal.value), _value("N"))


def test_a_redirected_request_presenting_a_principal_names_no_location(raw_workspace) -> None:
    """A request that presented a caller principal is refused as a redirect —
    the typed trust refusal, never followed — but a ``Location`` echoing the
    header never reaches the refusal: not its message, its chain, its
    ``location``, nor the MCP tool result built from it."""
    from ccs.core.exceptions import RedirectRefused
    from ccs.mcp.deny import deny_result

    workspace, raw = raw_workspace
    raw.responses = [_http_answer(307, {}, {"Location": f"http://127.0.0.1:1/{_value('P')}"})]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    with pytest.raises(RedirectRefused) as raised:
        _coherence_client.post(
            endpoint, "/hooks/pre-edit", {"session_id": _sid()},
            extra_headers={_PRINCIPAL_HEADER: _value("P")},
        )

    refused = raised.value
    tool_result = deny_result(refused)
    assert raw.seen[0][1] == _value("P"), "control: the principal was presented"
    assert refused.status == 307
    _assert_nothing_leaks(
        _rendered(refused) + str(refused.location) + json.dumps(tool_result.structuredContent),
        _value("P"),
    )


_REDIRECT_CODES = [301, 302, 303, 307, 308]


@pytest.mark.parametrize("code", _REDIRECT_CODES)
@pytest.mark.parametrize("material", ["principal-header", "claim-route", "none"])
def test_a_location_that_does_not_parse_is_withheld_before_it_is_parsed(
    raw_workspace, monkeypatch: pytest.MonkeyPatch, code: int, material: str,
) -> None:
    """A redirect is refused naming no ``Location`` — BEFORE anything parses
    it — whether the request carried a principal (header), a mint nonce (the
    claim) or neither (what answers it may have been sent one earlier). A
    ``Location`` whose bracketed host is not an address makes the stdlib's
    redirect handling raise a ``ValueError`` quoting the host (precondition,
    below): parsed first, the echoed value escaped untyped, in the message
    and down the chain, from every principal path. Now it is the typed
    refusal, with the status only, and the stdlib's redirect handling never
    saw the ``Location`` at all."""
    import urllib.parse
    import urllib.request

    from ccs.core.exceptions import RedirectRefused

    workspace, raw = raw_workspace
    echo = _value("P")
    with pytest.raises(ValueError, match=echo):  # precondition: parsing it quotes the echo
        urllib.parse.urlsplit(_bad_url(echo))
    parsed: list[str] = []
    real_urlparse = urllib.request.urlparse

    def recorded_urlparse(url: str, *args: Any, **kwargs: Any) -> Any:
        parsed.append(url)
        return real_urlparse(url, *args, **kwargs)

    # The name the stdlib's http_error_30x resolves the Location with.
    monkeypatch.setattr(urllib.request, "urlparse", recorded_urlparse)
    raw.responses = [_bad_location(echo, code)]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    with pytest.raises(RedirectRefused) as raised:
        if material == "principal-header":
            _coherence_client.post(
                endpoint, "/hooks/pre-edit", {"session_id": _sid()},
                extra_headers={_PRINCIPAL_HEADER: echo},
            )
        elif material == "claim-route":
            _coherence_client.post(
                endpoint, _coherence_client.PRINCIPAL_CLAIM_ROUTE,
                {"session_id": _sid(), "mint_nonce": echo},
            )
        else:
            _coherence_client.get(endpoint, "/status")

    assert raised.value.status == code
    assert raised.value.location == _coherence_client.REDIRECT_LOCATION_WITHHELD
    assert len(raw.seen) == 1, "control: the request was answered"
    assert not [url for url in parsed if echo in url], "the Location was parsed"
    _assert_nothing_leaks(_rendered(raised.value), echo)


def test_a_location_that_does_not_parse_is_a_typed_refusal_on_any_request(raw_workspace) -> None:
    """Without principal material the same answer is still the typed refusal
    — never a ``ValueError`` out of the transport — and since the
    ``Location`` names no URL it could follow, none is named."""
    from ccs.core.exceptions import RedirectRefused

    workspace, raw = raw_workspace
    raw.responses = [_bad_location("ECHO")]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    with pytest.raises(RedirectRefused) as raised:
        _coherence_client.get(endpoint, "/status")

    assert raised.value.location == _coherence_client.REDIRECT_LOCATION_WITHHELD
    _assert_nothing_leaks(_rendered(raised.value), "ECHO")


def test_a_claim_with_a_proxy_configured_still_withholds_the_location(
    raw_workspace, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With an HTTP proxy configured, a claim whose redirect echoes the nonce
    is still refused naming no ``Location``. Sent through a proxy, the
    request line would carry the absolute URL (urllib rewrites the selector
    to it), so a check keyed on the selector no longer saw the claim route
    and quoted the nonce the ``Location`` echoed. Coordinator requests now
    skip any configured proxy, so the claim keeps its route on the request
    line — and no refusal reads the route any more (none names its
    ``Location``), so neither shape of the request line changes what this
    one names."""
    from ccs.core.exceptions import RedirectRefused

    workspace, raw = raw_workspace
    for name in ("no_proxy", "NO_PROXY", "https_proxy", "HTTPS_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{raw.port}")
    raw.responses = [_redirect(_value("N"))]
    endpoint = _coherence_client.resolve_endpoint(workspace)

    with pytest.raises(RedirectRefused) as raised:
        _coherence_client.post(
            endpoint, _coherence_client.PRINCIPAL_CLAIM_ROUTE,
            {"session_id": _sid(), "mint_nonce": _value("N")},
        )

    # The proxy is this same server: had the claim gone through it, the
    # request line would carry the absolute URL instead of the route.
    assert raw.seen[0][0] == _coherence_client.PRINCIPAL_CLAIM_ROUTE, (
        "control: the claim skipped the configured proxy"
    )
    assert raised.value.location == _coherence_client.REDIRECT_LOCATION_WITHHELD
    _assert_nothing_leaks(_rendered(raised.value), _value("N"))


# --- what a one-shot client's refusal carries on its chain ---------------------


@pytest.mark.parametrize("recovery", ["stops", "refused_again"])
def test_no_coordinator_supplied_text_rides_the_chain_of_a_one_shot_refusal(
    scripted, recovery: str,
) -> None:
    """The refused request's status line carries the stored principal, the
    nonce and the re-claimed principal as its reason phrase. The
    ``CallerPrincipalRefused`` a one-shot client raises — whether recovery
    stops (the session is bound under another nonce) or the retry is refused
    again — names the typed reason, and the ``HTTPError`` carrying the phrase
    is not on its chain, so a caller's traceback prints none of them."""
    from ccs.core.exceptions import CallerPrincipalRefused

    workspace, sid = scripted
    _Scripted.phrase = " ".join((_value("P"), _value("N"), _value("Q")))
    _Scripted.claims = (
        [(200, {"ok": False, "reason": _CLAIMED})]
        if recovery == "stops"
        else [(200, {"ok": True, "principal": _value("Q")})]
    )
    _Scripted.route = staticmethod(lambda _h: _refusal(_FOREIGN))
    endpoint = _coherence_client.resolve_endpoint(workspace)

    with pytest.raises(CallerPrincipalRefused) as raised:
        _coherence_client.post_with_stored_principal(
            endpoint, workspace, "/hooks/session-stop", {"session_id": sid}, report=lambda _m: None
        )

    rendered = _rendered(raised.value)
    assert raised.value.reason == _FOREIGN
    assert len(_Scripted.seen) == (2 if recovery == "stops" else 3), "control: the scenario ran"
    _assert_nothing_leaks(rendered, _value("P"), _value("N"), _value("Q"))


# --- the self-test reports status and known tokens, never an answer -----------


class _EchoCoordinator(http.server.BaseHTTPRequestHandler):
    """Answers the self-test's four steps and its ``/status`` read as a healthy
    coordinator does — binding a fresh principal per claim — except at
    ``echo_at``, where it answers so the step FAILS, with the principal the
    request presented and every nonce claimed so far in every field."""

    echo_at: str | None = None
    nonces: list[str] = []
    principals: list[str] = []
    pre_reads = 0

    def _send(self, answer: object) -> None:
        raw = json.dumps(answer).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _echo(self) -> str:
        return " ".join([self.headers.get(_PRINCIPAL_HEADER) or "", *type(self).nonces])

    def _redirect(self, at: str, echo: str) -> None:
        # The status is the step's suffix (``status-redirect-307``), else 302.
        self.send_response(int(at[-3:]) if at[-3:].isdigit() else 302)
        self.send_header(
            "Location", _bad_url(echo) if at.endswith("bad-location") else f"http://127.0.0.1:1/{echo}"
        )
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802 — stdlib name
        cls = type(self)
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        at, echo = cls.echo_at, self._echo()
        if self.path == "/principal/claim":
            cls.nonces.append(body["mint_nonce"])
            cls.principals.append(secrets.token_urlsafe(32))
            self._send({"ok": True, "principal": cls.principals[-1]})
        elif self.path == "/hooks/pre-read":
            cls.pre_reads += 1
            first = cls.pre_reads == 1
            if at == "pre-read-1":
                self._send({"status": echo, "error": echo, "detail": echo})
            elif first:
                self._send({"status": "fresh", "version": 1})
            elif at == "pre-read-2":
                self._send({"status": echo, "version": echo})
            elif at == "stale-prose":
                self._send({"status": "stale", "hookSpecificOutput": {"additionalContext": echo}})
            else:
                self._send({"status": "stale", "hookSpecificOutput": {"additionalContext": "plan.md changed"}})
        elif self.path == "/hooks/pre-edit":
            if at is not None and at.startswith(("pre-edit-redirect", "pre-edit-bad-location")):
                self._redirect(at, echo.replace(" ", ""))
            elif at == "pre-edit":
                self._send({"ok": False, "reason": echo, "error": echo})
            elif at == "pre-edit-list":
                self._send([echo])
            else:
                self._send({"ok": True})
        else:  # /hooks/post-edit
            self._send({"ok": False, "reason": echo} if at == "post-edit" else {"ok": True})

    def do_GET(self) -> None:  # noqa: N802 — stdlib name
        at = type(self).echo_at
        if at is not None and at.startswith(("status-redirect", "status-bad-location")):
            self._redirect(at, self._echo().replace(" ", ""))
            return
        counters: dict[str, object] = {"pre_read_total": 2, "post_edit_total": 1}
        if type(self).echo_at == "counters":
            # Too few, and a field carrying the echo beside them.
            counters = {"pre_read_total": 1, "post_edit_total": 0, "note": self._echo()}
        self._send({"stale_warning_emitted_total": 1, "endpoint_counters": counters})

    def log_message(self, *_args: Any) -> None:
        return


# Where each echo makes the self-test fail: the step its report names.
_SELF_TEST_FAILURES = {
    None: None,
    "pre-read-1": "expected fresh on first pre-read",
    "pre-edit": "pre-edit failed",
    "pre-edit-list": "/hooks/pre-edit answered with a non-object answer",
    "pre-edit-redirect": "/hooks/pre-edit was redirected (HTTP 302)",
    "pre-edit-redirect-307": "/hooks/pre-edit was redirected (HTTP 307)",
    "pre-edit-bad-location": "/hooks/pre-edit was redirected (HTTP 302)",
    "post-edit": "post-edit failed",
    "pre-read-2": "expected stale warning",
    "stale-prose": "stale-warning prose did not mention plan.md",
    "counters": "endpoint counters did not reflect",
    "status-redirect": "/status was redirected (HTTP 302); not followed",
    "status-redirect-307": "/status was redirected (HTTP 307); not followed",
    "status-redirect-308": "/status was redirected (HTTP 308); not followed",
    "status-bad-location": "/status was redirected (HTTP 302); not followed",
}


@pytest.mark.parametrize("echo_at", list(_SELF_TEST_FAILURES), ids=lambda at: at or "control")
def test_the_self_test_reports_a_failed_step_by_status_and_known_tokens_only(
    workspace: Path, capsys: pytest.CaptureFixture[str], echo_at: str | None,
) -> None:
    """A coordinator (or anything in front of it) that echoes the principal
    header and the claimed nonces into the answer that fails a step cannot get
    either onto the self-test's output: each failure is reported by the step,
    the answer's status, ok flag and known reason token, the prose's length,
    or the counters as numbers — never the answer itself. A redirect —
    including the ``/status`` read's, which carries no principal but is
    answered by what the claims sent their nonces to — is reported by the
    status it was answered with, and no traceback. Every failing step exits
    3; the control run, with no echo, passes."""
    _EchoCoordinator.echo_at, _EchoCoordinator.pre_reads = echo_at, 0
    _EchoCoordinator.nonces, _EchoCoordinator.principals = [], []
    httpd = http.server.HTTPServer(("127.0.0.1", 0), _EchoCoordinator)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
    (workspace / ".coherence").mkdir(mode=0o700)
    (workspace / ".coherence" / "server.pid").write_text(f"12345\n{httpd.server_address[1]}\n")
    (workspace / ".coherence" / "hook.secret").write_text("test-secret")
    try:
        rc = coherence_status._run_self_test(workspace)
    finally:
        httpd.shutdown()
        httpd.server_close()
    captured = capsys.readouterr()

    assert rc == (0 if echo_at is None else 3), captured.err
    assert "Traceback" not in captured.err
    assert len(_EchoCoordinator.principals) == 2, "control: both sessions were bound"
    failure = _SELF_TEST_FAILURES[echo_at]
    assert failure is None or failure in captured.err, "control: it failed at the echoing step"
    _assert_nothing_leaks(
        captured.out + captured.err, *_EchoCoordinator.principals, *_EchoCoordinator.nonces
    )
    for location in ("http://127.0.0.1:1/", "http://[x"):
        assert location not in captured.out + captured.err, "a redirect's Location was echoed"
