"""Unit 3 (R2): fail-closed remote semantics — typed 401 + the HTTP-200 trap.

Under strict mode (which remote pins) a degraded-200 body and an ok:false deny
already fail closed; this verifies that holds on the remote path, AND that a 401
(wrong/missing secret) raises the dedicated RemoteAuthFailed rather than
degrading silently.
"""

from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest

import ccs.adapters.coherent_volume as cv
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.cli._coherence_client import PrincipalClaim, resolve_remote_endpoint
from ccs.core.exceptions import CoherenceError, RemoteAuthFailed


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://10.0.0.5:8080/x", code, "err", {}, None)


# These tests exercise 401 / degraded-body fail-closed semantics for a REMOTE
# (non-loopback) host — NOT the plaintext-transport guard — so each resolve call
# acknowledges the test-only insecure link explicitly (deliberate per-call ack).
_INSECURE_ACK = {"CCS_REMOTE_INSECURE": "1"}


def _remote_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, probe=None
) -> CoherentVolume:
    """Construct a remote-mode volume with a stubbed-reachable attach probe.

    Attach also claims the session's caller principal (plan U5); the stub
    coordinator binds it, so construction reaches the op under test."""
    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    monkeypatch.setattr(
        cv, "_coordinator_get", probe or (lambda ep, path, **k: {"ok": True})
    )
    monkeypatch.setattr(
        cv, "claim_caller_principal",
        lambda ep, sid, nonce: PrincipalClaim("bound", principal="P" * 43),
    )
    remote = resolve_remote_endpoint("10.0.0.5", 8080, "secret", env=_INSECURE_ACK)
    return CoherentVolume(tmp_path, on_error="strict", remote_endpoint=remote)


def test_remote_auth_failed_is_coherence_error():
    assert issubclass(RemoteAuthFailed, CoherenceError)


def test_op_401_raises_remote_auth_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vol = _remote_volume(tmp_path, monkeypatch)

    def raise_401(ep, path, payload, **k):
        raise _http_error(401)

    monkeypatch.setattr(cv, "_coordinator_post", raise_401)
    with pytest.raises(RemoteAuthFailed):
        vol._post("/hooks/pre-edit", {"session_id": vol.session_id, "path": "x"})


@pytest.mark.parametrize("verb", ["transfer", "accept", "decline", "withdraw"])
def test_a_handoff_verb_answered_401_raises_remote_auth_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verb: str
):
    """The handoff verbs treat a lost answer as an unknown outcome
    (``CommitUnconfirmed``), so a 401 must still be caught first: read as a
    failed request, a wrong remote secret would surface as "the verb may have
    landed" instead of a loud auth failure."""
    vol = _remote_volume(tmp_path, monkeypatch)
    (tmp_path / "plan.md").write_bytes(b"v1")

    def raise_401(ep, path, payload, **k):
        raise _http_error(401)

    monkeypatch.setattr(cv, "_coordinator_post", raise_401)
    with pytest.raises(RemoteAuthFailed):
        if verb == "transfer":
            vol.transfer("plan.md", successor="0" * 32)
        else:
            getattr(vol, verb)("plan.md")


def test_attach_401_raises_remote_auth_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def probe_401(ep, path, **k):
        raise _http_error(401)

    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    monkeypatch.setattr(cv, "_coordinator_get", probe_401)
    remote = resolve_remote_endpoint("10.0.0.5", 8080, "secret", env=_INSECURE_ACK)
    with pytest.raises(RemoteAuthFailed):
        CoherentVolume(tmp_path, on_error="strict", remote_endpoint=remote)


def test_non_401_http_error_still_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vol = _remote_volume(tmp_path, monkeypatch)

    def raise_403(ep, path, payload, **k):
        raise _http_error(403)

    monkeypatch.setattr(cv, "_coordinator_post", raise_403)
    with pytest.raises(CoherenceError):
        vol._post("/hooks/pre-edit", {"session_id": vol.session_id, "path": "x"})


def test_degraded_200_body_fails_closed_on_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vol = _remote_volume(tmp_path, monkeypatch)
    (tmp_path / "f.txt").write_text("hi", encoding="utf-8")
    monkeypatch.setattr(
        cv, "_coordinator_post", lambda ep, path, payload, **k: {"degraded": True}
    )
    with pytest.raises(CoherenceError):
        vol.read_with_version("f.txt")


def test_attach_403_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Hardening: any non-2xx at the attach probe (e.g. 403 Host mismatch) fails
    CLOSED at construction rather than deferring the misconfig to the first op."""
    def probe_403(ep, path, **k):
        raise _http_error(403)

    monkeypatch.setenv("CCS_REMOTE_COORDINATOR", "1")
    monkeypatch.setattr(cv, "_coordinator_get", probe_403)
    remote = resolve_remote_endpoint("10.0.0.5", 8080, "secret", env=_INSECURE_ACK)
    with pytest.raises(CoherenceError):
        CoherentVolume(tmp_path, on_error="strict", remote_endpoint=remote)
