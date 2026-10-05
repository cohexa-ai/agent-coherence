# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Wire-stability, server-capture boundary lock, and endpoint-auth tests for
the snapshot-session HTTP endpoints (SB-17 / TX-1, Unit 8 — R7 / R9 / R10a).

Covers:
- Endpoint auth: every ``/session/*`` route rides the central ``_ROUTES``
  ``verify_bearer`` + ``verify_host`` seam (401 no/bad bearer, 403 bad Host) —
  NOT a parallel router.
- Boundary lock (R9): a client-supplied pinned-version / cut / forged-or-
  replayed token / client-asserted owner CANNOT forge or bypass the
  server-side capture. The "client carries the cut" path FAILS the guard.
- Wire (R7): new session reasons are ADDITIVE; existing reason sets unchanged.
- Audit (R10a): begin / commit / invalidate emit content-free JSONL records.
- Happy-path round trip: begin → read → commit over HTTP with a valid
  bearer/host and a UUID-shaped (caller-asserted) session_id.
- Caller-principal mint (plan U4): ``POST /principal/claim`` binds on first
  claim, recovers a lost response by nonce, and a principal never appears on
  any response but the mint's, in any log record, or in any file but the store.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional
from urllib import error as urlerror
from urllib import request as urlrequest

import pytest

from ccs.adapters.claude_code.auth import load_secret
from ccs.adapters.claude_code.coordinator_server import (
    _ROUTES,
    CoordinatorHTTPServer,
)
from ccs.adapters.claude_code.session_audit_log import (
    _resolve_session_audit_log_path,
)
from ccs.core.exceptions import (
    READ_AT_VERSION_REASONS,
    SESSION_BEGIN_CAP_REASONS,
    SESSION_COMMIT_REASONS,
    SESSION_READ_REASONS,
)

_TEST_SESSION_NS = uuid.UUID("22222222-2222-4222-8222-222222222222")


def _sid(label: str) -> str:
    return str(uuid.uuid5(_TEST_SESSION_NS, f"session-wire:{label}"))


class _Client:
    """Tiny urllib client returning (status, body_dict)."""

    def __init__(self, host: str, port: int, secret: str) -> None:
        self.base = f"http://{host}:{port}"
        self.headers = {
            "Authorization": f"Bearer {secret}",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        body: Optional[dict] = None,
        *,
        headers_override: Optional[dict] = None,
        principal: Optional[str] = None,
    ) -> tuple[int, dict]:
        url = self.base + path
        data = json.dumps(body).encode("utf-8") if body is not None else b""
        headers = dict(self.headers)
        if headers_override:
            headers.update(headers_override)
        if principal is not None:
            # The caller principal header (caller-principal plan U5); a frozen
            # literal rather than the code's constant.
            headers["Coherence-Caller-Principal"] = principal
        req = urlrequest.Request(
            url, data=data if method == "POST" else None, method=method, headers=headers
        )
        try:
            with urlrequest.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urlerror.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8") or "{}")

    def post(self, path: str, body: dict, **kw) -> tuple[int, dict]:
        return self.request("POST", path, body, **kw)


@pytest.fixture
def coordinator(tmp_path: Path):
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="test-instance")
    server.serve_in_thread()
    time.sleep(0.05)
    try:
        yield server
    finally:
        server.shutdown()


@pytest.fixture
def client(coordinator) -> _Client:
    secret = load_secret(coordinator.coordinator_root)
    assert secret is not None
    return _Client("127.0.0.1", coordinator.port, secret)


_SESSION_ROUTES = [
    "/session/begin",
    "/session/read",
    "/session/commit",
    "/session/heartbeat",
]


# ----------------------------------------------------------------------
# Endpoint auth — rides the central _ROUTES seam (NOT a parallel router)
# ----------------------------------------------------------------------


def test_session_routes_registered_in_central_routes() -> None:
    """All four session routes are in the central _ROUTES table, so the
    dispatcher's verify_bearer + verify_host run before any handler — there
    is no parallel router that could bypass auth."""
    for route in _SESSION_ROUTES:
        assert ("POST", route) in _ROUTES


@pytest.mark.parametrize("route", _SESSION_ROUTES)
def test_session_endpoint_no_bearer_returns_401(coordinator, route: str) -> None:
    url = f"http://127.0.0.1:{coordinator.port}{route}"
    req = urlrequest.Request(
        url, data=b"{}", method="POST",
        headers={"Host": "127.0.0.1", "Content-Type": "application/json"},
    )
    try:
        urlrequest.urlopen(req, timeout=5)
        raise AssertionError("expected 401")
    except urlerror.HTTPError as e:
        assert e.code == 401


@pytest.mark.parametrize("route", _SESSION_ROUTES)
def test_session_endpoint_bad_bearer_returns_401(coordinator, route: str) -> None:
    url = f"http://127.0.0.1:{coordinator.port}{route}"
    req = urlrequest.Request(
        url, data=b"{}", method="POST",
        headers={
            "Authorization": "Bearer not-the-secret",
            "Host": "127.0.0.1",
            "Content-Type": "application/json",
        },
    )
    try:
        urlrequest.urlopen(req, timeout=5)
        raise AssertionError("expected 401")
    except urlerror.HTTPError as e:
        assert e.code == 401


@pytest.mark.parametrize("route", _SESSION_ROUTES)
def test_session_endpoint_bad_host_returns_403(client: _Client, route: str) -> None:
    status, _body = client.post(
        route, {"session_id": _sid("h")},
        headers_override={"Host": "attacker.example.com"},
    )
    assert status == 403


# ----------------------------------------------------------------------
# Happy-path round trip (begin → read → commit)
# ----------------------------------------------------------------------


def _begin(client: _Client, sid: str, read_set: list[str]) -> dict:
    status, body = client.post(
        "/session/begin", {"session_id": sid, "read_set": read_set}
    )
    assert status == 200, body
    return body


def test_happy_path_begin_read_commit(client: _Client) -> None:
    sid = _sid("happy")
    # First-observation seeds v1 for the path.
    begin = _begin(client, sid, ["plan.md"])
    assert begin["ok"] is True
    assert "session_token" in begin
    token = begin["session_token"]
    assert begin["cut"] == {"plan.md": 1}
    assert "coordinator_epoch" in begin and "retain_versions" in begin

    # Read the pinned version.
    status, read = client.post(
        "/session/read", {"session_id": sid, "session_token": token, "path": "plan.md"}
    )
    assert status == 200, read
    assert read["ok"] is True
    assert read["version"] == 1
    # Default registry retains no bodies (retain_versions=False) → EAGER branch
    # → the coordinator defers byte-serving to the data plane (typed, not a
    # crash). The LAZY content-serve branch is exercised at the service layer
    # in tests/test_session_read.py (the HTTP server hardwires retain=False).
    assert read["served"] == "data_plane_deferred"
    assert begin["retain_versions"] is False

    # Commit against the pinned base → WIN, version bumps to 2.
    status, commit = client.post(
        "/session/commit",
        {"session_id": sid, "session_token": token, "path": "plan.md", "content": "new"},
    )
    assert status == 200, commit
    assert commit["ok"] is True
    assert commit["version"] == 2

    # A SECOND commit at the same pin is HELD (R11 exactly-one-commit).
    status, commit2 = client.post(
        "/session/commit",
        {"session_id": sid, "session_token": token, "path": "plan.md", "content": "again"},
    )
    assert status == 200
    assert commit2["ok"] is False
    assert commit2["reason"] == "version_mismatch"


def test_heartbeat_refreshes_owned_session(client: _Client) -> None:
    sid = _sid("hb")
    begin = _begin(client, sid, ["hb.md"])
    token = begin["session_token"]
    status, hb = client.post(
        "/session/heartbeat", {"session_id": sid, "session_token": token}
    )
    assert status == 200
    assert hb == {"ok": True, "refreshed": True}


# ----------------------------------------------------------------------
# Boundary lock (R9) — no client can forge / bypass the server-side capture
# ----------------------------------------------------------------------


def test_client_supplied_cut_is_ignored_not_trusted(client: _Client) -> None:
    """A client that smuggles a ``cut`` / ``pinned_version`` / ``expected_version``
    into /session/read or /session/commit MUST NOT have it honored — the server
    is authoritative. The forged fields are ignored; the server reads the pinned
    base from the registry by token. (The day a client legitimately carries the
    cut is cross-host — this guard FAILS if anyone wires that in here.)"""
    sid = _sid("forge-cut")
    begin = _begin(client, sid, ["target.md"])  # pins target.md@v1
    token = begin["session_token"]

    # Forge a pinned_version=999 + a fake cut in the read request. The server
    # ignores them and serves the REAL pinned v1.
    status, read = client.post(
        "/session/read",
        {
            "session_id": sid, "session_token": token, "path": "target.md",
            "pinned_version": 999, "cut": {"target.md": 999}, "version": 999,
        },
    )
    assert status == 200, read
    assert read["ok"] is True
    assert read["version"] == 1  # the server-captured pin, NOT the forged 999.

    # Forge expected_version=999 on commit. The server uses the real pinned v1
    # as the comparand → WIN bumps to v2 (a forged 999 would corrupt-error).
    status, commit = client.post(
        "/session/commit",
        {
            "session_id": sid, "session_token": token, "path": "target.md",
            "content": "x", "expected_version": 999, "pinned_version": 999,
        },
    )
    assert status == 200, commit
    assert commit["ok"] is True
    assert commit["version"] == 2  # pinned-base CAS, forged comparand ignored.


def test_forged_session_token_cannot_bypass_capture(client: _Client) -> None:
    """A made-up / never-minted session token has no server-side cut. A read or
    commit against it fails CLOSED (typed reason), NEVER served live HEAD."""
    sid = _sid("forged-token")
    # Seed the artifact so it exists (so the failure is about the TOKEN, not the
    # path).
    _begin(client, sid, ["existing.md"])

    forged = "totally-not-a-real-server-minted-token"
    status, read = client.post(
        "/session/read",
        {"session_id": sid, "session_token": forged, "path": "existing.md"},
    )
    assert status == 200, read
    assert read["ok"] is False
    assert read["reason"] in SESSION_READ_REASONS  # fail-closed, not live HEAD

    status, commit = client.post(
        "/session/commit",
        {"session_id": sid, "session_token": forged, "path": "existing.md", "content": "x"},
    )
    assert status == 200
    assert commit["ok"] is False
    assert commit["reason"] in SESSION_COMMIT_REASONS


def test_replayed_token_after_release_fails_closed(coordinator, client: _Client) -> None:
    """A token whose session has been reaped (its cut released) cannot be
    replayed to read/commit — it fails closed, never serves the (now stale)
    cut from live HEAD."""
    sid = _sid("replay2")
    begin = _begin(client, sid, ["replay2.md"])
    token = begin["session_token"]
    # Release the session's pins directly on the wrapped registry (what the
    # liveness sweep does when a heartbeat goes stale).
    coordinator.registry.release_session(token)
    status, read = client.post(
        "/session/read", {"session_id": sid, "session_token": token, "path": "replay2.md"}
    )
    assert status == 200, read
    assert read["ok"] is False
    assert read["reason"] in SESSION_READ_REASONS  # fail-closed


def test_foreign_caller_cannot_read_anothers_cut(coordinator, client: _Client) -> None:
    """R13 owner isolation surfaced at the wire: a request naming a DIFFERENT
    session_id (a sibling) cannot read another session's cut even with the
    leaked token — it fails closed with session_invalidated. The session_id is
    caller-asserted; what this pins is that the owner check keys on it."""
    owner_sid = _sid("owner")
    begin = _begin(client, owner_sid, ["owned.md"])
    token = begin["session_token"]

    foreign_sid = _sid("foreign")  # a request naming a different session
    status, read = client.post(
        "/session/read",
        {"session_id": foreign_sid, "session_token": token, "path": "owned.md"},
    )
    assert status == 200, read
    assert read["ok"] is False
    assert read["reason"] == "session_invalidated"


def test_client_cannot_assert_owner_field(coordinator, client: _Client) -> None:
    """A client-asserted ``owner`` / ``caller`` field MUST NOT bind or rebind the
    session owner — the owner is derived from the request's session_id only.
    A foreign caller supplying the real owner's id as ``owner``/``caller`` still
    fails closed (the server ignores those fields)."""
    owner_sid = _sid("owner-bind")
    begin = _begin(client, owner_sid, ["bound.md"])
    token = begin["session_token"]

    foreign_sid = _sid("foreign-bind")
    # Try to impersonate the owner via a client-supplied owner/caller field.
    status, read = client.post(
        "/session/read",
        {
            "session_id": foreign_sid, "session_token": token, "path": "bound.md",
            "owner": owner_sid, "caller": owner_sid,
        },
    )
    assert status == 200, read
    assert read["ok"] is False
    assert read["reason"] == "session_invalidated"  # owner came from session_id, not the field


# ----------------------------------------------------------------------
# Wire (R7) — new reasons are additive; existing sets unchanged
# ----------------------------------------------------------------------


def test_session_reason_sets_are_disjoint_and_additive() -> None:
    """The session reason sets are NET-NEW closed sets, disjoint from the
    bare read_at_version contract — additive, never folded in (R7)."""
    assert SESSION_READ_REASONS.isdisjoint(READ_AT_VERSION_REASONS)
    assert SESSION_COMMIT_REASONS.isdisjoint(READ_AT_VERSION_REASONS)
    assert SESSION_BEGIN_CAP_REASONS.isdisjoint(READ_AT_VERSION_REASONS)
    # The pre-Unit-2 frozen read_at_version reasons are unchanged (6 reasons).
    assert "current_version" in READ_AT_VERSION_REASONS
    assert "unknown_artifact" in READ_AT_VERSION_REASONS


def test_begin_unknown_artifact_reason_stays_in_read_at_version_set(client: _Client) -> None:
    """begin_session's unknown-id rejection reuses the existing unknown_artifact
    reason (not a parallel one) — but unknown PATHS get seeded as first
    observations, so this asserts the cap-reason additive surface instead via a
    too-large read_set."""
    # An over-cap read_set is rejected at the WIRE (400) before the service —
    # the service-level read_set_too_large stays the authoritative cap.
    sid = _sid("cap")
    big = [f"f{i}.md" for i in range(65)]  # > MAX_SESSION_READ_SET_PATHS (64)
    status, body = client.post("/session/begin", {"session_id": sid, "read_set": big})
    assert status == 400
    assert "read_set" in body["error"]


# ----------------------------------------------------------------------
# Audit (R10a) — begin / commit / invalidate emit content-free records
# ----------------------------------------------------------------------


def test_audit_emits_begin_commit_invalidate_content_free(coordinator, client: _Client) -> None:
    sid = _sid("audit")
    begin = _begin(client, sid, ["audit.md"])
    token = begin["session_token"]
    # Commit → WIN (emits a session_commit audit event).
    client.post(
        "/session/commit",
        {"session_id": sid, "session_token": token, "path": "audit.md", "content": "v2"},
    )
    # Reap then read with the dead token → fail-closed → session_invalidate event.
    coordinator.registry.release_session(token)
    client.post(
        "/session/read", {"session_id": sid, "session_token": token, "path": "audit.md"}
    )

    audit_path = _resolve_session_audit_log_path(coordinator.coordinator_root)
    records = [json.loads(line) for line in audit_path.read_text().strip().splitlines()]
    events = {r["event"] for r in records}
    assert {"session_begin", "session_commit", "session_invalidate"} <= events

    # Content-free: no record carries body / hash / prose / raw token.
    forbidden = {"content", "content_hash", "body", "command", "token", "session_token"}
    raw = audit_path.read_text()
    assert token not in raw  # raw token never logged (only its hash)
    assert "v2" not in raw  # the committed body bytes never logged
    for record in records:
        assert not (set(record.keys()) & forbidden)

    begin_rec = next(r for r in records if r["event"] == "session_begin")
    # ids + versions only.
    assert set(begin_rec.keys()) == {"ts", "event", "session", "cut"}
    commit_rec = next(r for r in records if r["event"] == "session_commit")
    assert set(commit_rec.keys()) == {
        "ts", "event", "session", "artifact", "pinned_version", "committed_version",
    }


# ----------------------------------------------------------------------
# F1 — clock-domain basis: the sweep tick MUST share the wall-clock basis the
#      heartbeat handlers seed from, else the sweep never reaps over HTTP.
# ----------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from ccs.adapters.claude_code import lifecycle as _lifecycle  # noqa: E402
from ccs.adapters.claude_code.coordinator_server import (  # noqa: E402
    _session_read_content_fields,
    monotonic_seconds,
)


def test_monotonic_seconds_is_wall_clock_not_monotonic() -> None:
    # The shared tick basis — handlers seed created_at + heartbeats from this, and
    # (after F1) the sweep reads it too — MUST be wall-clock int(time.time()), NOT
    # int(time.monotonic()) (boot-relative, ~1.7e9 below wall-clock). The F1 bug
    # was the sweep on the monotonic basis while seeds used wall-clock, so the
    # staleness diff was permanently negative and nothing reaped over HTTP.
    now = monotonic_seconds()
    assert abs(now - int(time.time())) <= 2
    assert abs(now - int(time.monotonic())) > 1_000_000


def test_sweep_loop_uses_wall_clock_basis() -> None:
    # Drive ONE iteration of the REAL _sweep_loop with stubs and capture the tick
    # it passes to the enforce_* sweeps. It must be wall-clock (== monotonic_seconds
    # basis) so the arithmetic against a wall-clock-seeded lease is positive. This
    # fails if anyone reverts the sweep tick to int(time.monotonic()).
    captured: dict = {}
    coord = SimpleNamespace()

    class _Svc:
        def enforce_transient_timeouts(self, *, current_tick, timeout_ticks):
            captured["transient"] = current_tick
            return 0

        def enforce_stable_grant_timeouts(
            self, *, current_tick, heartbeat_timeout_ticks, max_hold_ticks, on_reclaim
        ):
            captured["grant"] = current_tick
            return 0

        def enforce_session_liveness(self, *, current_tick, heartbeat_timeout_ticks):
            captured["session"] = current_tick
            coord.shutting_down = True  # exit the loop after one iteration
            return 0

    class _Reg:
        def evict_stale_notices(self, *, max_age_sec):
            return 0

        def evict_transfer_records(self, *, max_age_sec):
            return 0

        def record_preemption_notice(self, **kw):
            pass

    coord.shutting_down = False
    coord.service = _Svc()
    coord.registry = _Reg()
    entry = SimpleNamespace(coordinator=coord)
    cfg = SimpleNamespace(
        sweep_interval_sec=0.01,
        transient_timeout_sec=5,
        grant_heartbeat_timeout_sec=120,
        grant_max_hold_sec=300,
        notice_evict_max_age_sec=600,
        transfer_record_evict_max_age_sec=86_400,
    )

    _lifecycle._sweep_loop(entry, cfg)

    assert "session" in captured
    # Wall-clock basis (shared with the heartbeat seeds) — NOT monotonic.
    assert abs(captured["session"] - int(time.time())) <= 2
    assert abs(captured["session"] - int(time.monotonic())) > 1_000_000
    # All three sweeps ride ONE now_tick per iteration.
    assert captured["transient"] == captured["grant"] == captured["session"]


# ----------------------------------------------------------------------
# F5 — /session/read serves non-UTF-8 bytes losslessly (base64), not a lossy
#      replace-decode that would break the client-side hash round-trip.
# ----------------------------------------------------------------------


def test_read_content_fields_text_serves_plain_string() -> None:
    assert _session_read_content_fields("plain text") == {"content": "plain text"}
    assert _session_read_content_fields("héllo".encode("utf-8")) == {"content": "héllo"}


def test_read_content_fields_non_utf8_serves_base64() -> None:
    raw = b"\xff\xfe\x00\x01PNG\x89"  # not valid UTF-8
    import base64 as _b64

    fields = _session_read_content_fields(raw)
    assert fields["content_encoding"] == "base64"
    assert "content" not in fields
    # Lossless round-trip — the client can reconstruct EXACT bytes (so its hash
    # of the body matches the pinned content_hash; a lossy decode could not).
    assert _b64.b64decode(fields["content_b64"]) == raw


# ----------------------------------------------------------------------
# Input validation — malformed session request bodies return 400 (not 500/200).
# ----------------------------------------------------------------------


def test_begin_non_list_read_set_returns_400(client: _Client) -> None:
    status, _body = client.post(
        "/session/begin", {"session_id": _sid("badrs"), "read_set": "not-a-list"}
    )
    assert status == 400


def test_begin_non_string_read_set_member_returns_400(client: _Client) -> None:
    status, _body = client.post(
        "/session/begin", {"session_id": _sid("badrs2"), "read_set": [123]}
    )
    assert status == 400


def test_read_missing_session_token_returns_400(client: _Client) -> None:
    status, _body = client.post(
        "/session/read", {"session_id": _sid("notok"), "path": "x.md"}
    )
    assert status == 400


def test_read_empty_session_token_returns_400(client: _Client) -> None:
    status, _body = client.post(
        "/session/read",
        {"session_id": _sid("emptytok"), "session_token": "", "path": "x.md"},
    )
    assert status == 400


def test_commit_non_string_content_returns_400(client: _Client) -> None:
    sid = _sid("badcontent")
    begin = _begin(client, sid, ["bc.md"])
    token = begin["session_token"]
    status, _body = client.post(
        "/session/commit",
        {"session_id": sid, "session_token": token, "path": "bc.md", "content": 123},
    )
    assert status == 400


# ----------------------------------------------------------------------
# F7 — /session/commit is rejected (503) on a draining coordinator, like the
#      other version-bumping writes, so a commit can't land and be stranded.
# ----------------------------------------------------------------------


def test_session_commit_rejected_while_draining(client: _Client, coordinator) -> None:
    sid = _sid("drain")
    begin = _begin(client, sid, ["drain.md"])
    token = begin["session_token"]
    # Enter migration-draining (what /admin/prepare-for-migration flips).
    coordinator._migration_draining = True
    try:
        status, body = client.post(
            "/session/commit",
            {"session_id": sid, "session_token": token, "path": "drain.md", "content": "x"},
        )
        # A version-bumping write on a draining coordinator must be REJECTED (503),
        # not allowed to land and be stranded by the imminent shutdown.
        assert status == 503, body
    finally:
        coordinator._migration_draining = False


def test_session_read_still_served_while_draining(client: _Client, coordinator) -> None:
    # Non-mutating /session/read is NOT in the rejected set — it keeps serving so
    # in-flight readers complete during the drain.
    sid = _sid("drain-read")
    begin = _begin(client, sid, ["dr.md"])
    token = begin["session_token"]
    coordinator._migration_draining = True
    try:
        status, _body = client.post(
            "/session/read", {"session_id": sid, "session_token": token, "path": "dr.md"}
        )
        assert status == 200
    finally:
        coordinator._migration_draining = False


# ----------------------------------------------------------------------
# Caller-principal mint (coordinator caller principal plan, U4)
# ----------------------------------------------------------------------
#
# The service- and registry-level behaviour (both registries) is pinned in
# tests/coordinator/test_caller_principal.py. These cover the wire: the route,
# its counter, its refusals, and R5 — a principal crosses the wire on the mint
# response and nowhere else. Which routes REQUIRE a principal is U6.

import hashlib  # noqa: E402
import logging  # noqa: E402
import secrets  # noqa: E402

from ccs.adapters.claude_code.coordinator_server import (  # noqa: E402
    _ENDPOINT_COUNTER_NAMES,
    PresentedCaller,
    _is_recent_self_commit_lag,
    caller_principal_identity,
    session_to_agent_id,
)
from ccs.core.exceptions import (  # noqa: E402
    CALLER_PRINCIPAL_CLAIMED_REASON,
    HOLD_REASONS,
)

_CLAIM = "/principal/claim"


def _claim_wire(client: _Client, sid: str, nonce: object) -> tuple[int, dict]:
    return client.post(_CLAIM, {"session_id": sid, "mint_nonce": nonce})


def test_principal_claim_route_is_registered_and_counted(coordinator, client: _Client) -> None:
    """Registered in the central table (so the bearer + Host seam applies) and
    in BOTH counter registrations — the increment helper silently ignores a
    name missing from the counter dict, so the bump itself is asserted."""
    assert ("POST", _CLAIM) in _ROUTES
    assert _ENDPOINT_COUNTER_NAMES[("POST", _CLAIM)] == "principal_claim_total"
    before = coordinator.endpoint_counters_snapshot()["principal_claim_total"]
    status, _ = _claim_wire(client, _sid("counted"), secrets.token_urlsafe(32))
    assert status == 200
    assert coordinator.endpoint_counters_snapshot()["principal_claim_total"] == before + 1


def test_principal_claim_without_bearer_is_401(coordinator) -> None:
    url = f"http://127.0.0.1:{coordinator.port}{_CLAIM}"
    req = urlrequest.Request(
        url, data=b"{}", method="POST",
        headers={"Host": "127.0.0.1", "Content-Type": "application/json"},
    )
    with pytest.raises(urlerror.HTTPError) as err:
        urlrequest.urlopen(req, timeout=5)
    assert err.value.code == 401


def test_mint_binds_on_first_claim_and_a_second_claimant_is_refused(
    coordinator, client: _Client
) -> None:
    sid = _sid("mint-first")
    nonce = secrets.token_urlsafe(32)
    status, first = _claim_wire(client, sid, nonce)
    assert status == 200 and first["ok"] is True
    principal = first["principal"]

    status, other = _claim_wire(client, sid, secrets.token_urlsafe(32))
    assert status == 200
    assert other["ok"] is False
    assert other["reason"] == CALLER_PRINCIPAL_CLAIMED_REASON
    assert other["reason"] not in HOLD_REASONS
    assert "principal" not in other
    assert principal not in json.dumps(other)
    # The binding is still the first claimant's.
    assert coordinator.registry.get_caller_principal(caller_principal_identity(sid)) == principal


def test_mint_retry_with_the_same_nonce_returns_the_same_principal(client: _Client) -> None:
    """R20 over the wire: the first response is discarded, the retry presents
    the persisted nonce and receives the principal that was bound."""
    sid = _sid("mint-retry")
    nonce = secrets.token_urlsafe(32)
    _claim_wire(client, sid, nonce)  # response lost
    status, retry = _claim_wire(client, sid, nonce)
    assert status == 200 and retry["ok"] is True
    status, again = _claim_wire(client, sid, nonce)
    assert again["principal"] == retry["principal"]


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"session_id": "not-a-uuid", "mint_nonce": "a" * 43}, 400),
        ({"mint_nonce": "a" * 43}, 400),
        ({"session_id": "SID"}, 400),  # no nonce
        ({"session_id": "SID", "mint_nonce": "a" * 15}, 400),  # one below the floor
        ({"session_id": "SID", "mint_nonce": "a" * 16}, 200),  # the floor
        ({"session_id": "SID", "mint_nonce": "a" * 129}, 400),  # one above the ceiling
        ({"session_id": "SID", "mint_nonce": "a" * 15 + "?"}, 400),  # alphabet
    ],
)
def test_mint_validates_its_fields_at_the_boundary(
    client: _Client, body: dict, status: int
) -> None:
    body = {k: (_sid(f"boundary-{status}-{len(str(v))}") if v == "SID" else v) for k, v in body.items()}
    got, answer = client.post(_CLAIM, body)
    assert got == status, answer
    if status == 400:
        assert "error" in answer and "principal" not in answer


def test_self_commit_lag_still_compares_the_composite_writer(coordinator) -> None:
    """A subagent's foreign-edit case behaves as before: the lag suppression
    fires only for the composite id that committed, never for a sibling
    subagent of the same session, nor for the parent — even though all three
    present the SAME session principal."""
    sid = _sid("lag")
    principal = coordinator.service.claim_caller_principal(
        identity=caller_principal_identity(sid), mint_nonce=secrets.token_urlsafe(32)
    )
    alpha = PresentedCaller(session_id=sid, subagent_id="alpha", principal=principal)
    beta = PresentedCaller(session_id=sid, subagent_id="beta", principal=principal)
    writer = alpha.attributed_agent_id(coordinator.service)
    artifact_id = coordinator.registry.resolve_or_register(
        "lag.md", content_hash=hashlib.sha256(b"v1").hexdigest()
    )
    coordinator.service.write(agent_id=writer, artifact_id=artifact_id, issued_at_tick=1)
    coordinator.service.commit(
        agent_id=writer, artifact_id=artifact_id, content="v2", issued_at_tick=2
    )
    now = time.time()

    assert _is_recent_self_commit_lag(coordinator, artifact_id, writer, now_unix=now)
    assert not _is_recent_self_commit_lag(
        coordinator, artifact_id, beta.attributed_agent_id(coordinator.service), now_unix=now
    )
    assert not _is_recent_self_commit_lag(
        coordinator, artifact_id, session_to_agent_id(sid), now_unix=now
    )


def test_a_principal_appears_only_on_the_mint_response(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R5: a principal appears on exactly one response — the mint response —
    and never in any other response body, any status tier, any log record, the
    state log, or any file the coordinator writes other than its store. Every
    other route below is driven with the same session and must answer 200, so
    the absence is observed on requests that actually ran. Control: the store
    file DOES contain the principal, so the file scan can see one."""
    caplog.set_level(logging.DEBUG)
    state_log: list[dict] = []
    server = CoordinatorHTTPServer(
        tmp_path, port=0, instance_id="r5", state_log=state_log.append
    )
    server.serve_in_thread()
    time.sleep(0.05)
    try:
        client = _Client("127.0.0.1", server.port, load_secret(server.coordinator_root))
        sid = _sid("r5")
        nonce = secrets.token_urlsafe(32)
        _, minted = _claim_wire(client, sid, nonce)
        principal = minted["principal"]
        _, retried = _claim_wire(client, sid, nonce)
        assert retried["principal"] == principal  # also a mint response

        h1 = hashlib.sha256(b"one").hexdigest()
        others = [
            ("POST", _CLAIM, {"session_id": sid, "mint_nonce": secrets.token_urlsafe(32)}),
            ("POST", "/hooks/pre-read", {"session_id": sid, "path": "r5.md", "content_hash": h1}),
            ("POST", "/hooks/pre-edit", {"session_id": sid, "path": "r5.md"}),
            ("POST", "/hooks/post-edit", {
                "session_id": sid, "path": "r5.md",
                "content_hash": hashlib.sha256(b"two").hexdigest(), "success": True,
            }),
            ("POST", "/hooks/session-start", {"session_id": sid}),
            ("POST", "/session/begin", {"session_id": sid, "read_set": ["r5.md"]}),
            ("POST", "/hooks/session-stop", {"session_id": sid}),
            ("GET", "/status", None),
            ("GET", "/status?detail=metrics", None),
        ]
        bodies = []
        for method, path, body in others:
            # The request PRESENTS the principal (the commit and stop are
            # require-class): R5 is about what comes back, and the principal
            # must still appear in none of it.
            status, answer = client.request(method, path, body, principal=principal)
            assert status == 200, (path, answer)
            bodies.append(json.dumps(answer))
        status, full = client.request(
            "GET", "/status?detail=full",
            headers_override={"Coherence-Local-Operator": "true"},
        )
        assert status == 200
        bodies.append(json.dumps(full))
        assert '"ok": false' in bodies[0]  # the refused claim really was refused
    finally:
        server.shutdown()

    for body in bodies:
        assert principal not in body
    logged = "\n".join(
        f"{r.getMessage()} {r.exc_text or ''}" for r in caplog.records
    )
    assert logged  # the capture saw the requests
    assert principal not in logged
    assert principal not in json.dumps(state_log)

    store = {"state.db", "state.db-wal", "state.db-shm"}
    store_bytes = b"".join(
        p.read_bytes() for p in tmp_path.rglob("*") if p.is_file() and p.name in store
    )
    assert principal.encode() in store_bytes  # control: the scan can see one
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.name not in store]
    assert written  # audit logs, secret: the scan has something to read
    for path in written:
        assert principal.encode() not in path.read_bytes(), path


def test_a_refused_request_carries_no_principal_into_any_log_body_or_status_tier(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """R5 on the REFUSAL path — the branch most likely to grow a diagnostic,
    and one the admitted-traffic guard above never reaches. Every refusal shape
    is driven: an absent principal naming a bound session, a caller presenting
    its OWN valid principal against a peer's session (require and accept
    class), the peer's principal against the caller's session, and a valid
    principal presented for a session nobody claimed. Each answers 400 with its
    typed reason, and neither principal appears in any response body, any
    /status tier, any log record at any level — message, arguments or
    traceback — or the state log.

    Control: the capture sees the refusal path. Each refusal left a record
    naming its reason, so "not in the log" is measured against a log that
    recorded the refusals, not one that never saw them."""
    caplog.set_level(logging.DEBUG)
    state_log: list[dict] = []
    server = CoordinatorHTTPServer(
        tmp_path, port=0, instance_id="r5-refused", state_log=state_log.append
    )
    server.serve_in_thread()
    time.sleep(0.05)
    try:
        client = _Client("127.0.0.1", server.port, load_secret(server.coordinator_root))
        caller, peer, unclaimed = _sid("r5-refused-caller"), _sid("r5-refused-peer"), _sid("r5-none")
        _, minted = _claim_wire(client, caller, secrets.token_urlsafe(32))
        caller_principal = minted["principal"]
        _, minted = _claim_wire(client, peer, secrets.token_urlsafe(32))
        peer_principal = minted["principal"]
        h = hashlib.sha256(b"refused").hexdigest()
        absent, foreign = "caller_principal_absent", "caller_principal_foreign"
        refusals = [
            ("/hooks/session-stop", {"session_id": peer}, None, absent),
            ("/hooks/pre-edit", {"session_id": peer, "path": "r5.md"}, None, absent),
            ("/hooks/post-edit", {"session_id": peer, "path": "r5.md", "success": True,
                                  "content_hash": h}, caller_principal, foreign),
            ("/hooks/session-stop", {"session_id": peer}, caller_principal, foreign),
            ("/hooks/effect-fence", {"session_id": peer, "path": "r5.md",
                                     "expected_version": 1, "expected_generation": 0,
                                     "content_hash": h}, caller_principal, foreign),
            ("/hooks/pre-read", {"session_id": peer, "path": "r5.md"}, caller_principal, foreign),
            ("/hooks/session-stop", {"session_id": caller}, peer_principal, foreign),
            ("/hooks/post-edit-cas", {"session_id": unclaimed, "path": "r5.md",
                                      "content_hash": h, "expected_version": 0},
             caller_principal, foreign),
        ]
        bodies = []
        for path, body, principal, reason in refusals:
            status, answer = client.post(path, body, principal=principal)
            assert (status, answer.get("reason")) == (400, reason), (path, answer)
            bodies.append(json.dumps(answer))
        for query, headers in (
            ("/status", None),
            ("/status?detail=metrics", None),
            ("/status?detail=full", {"Coherence-Local-Operator": "true"}),
        ):
            status, answer = client.request("GET", query, headers_override=headers)
            assert status == 200, (query, answer)
            bodies.append(json.dumps(answer))
    finally:
        server.shutdown()

    principals = (caller_principal, peer_principal)
    assert len(set(principals)) == 2 and all(principals)
    for body in bodies:
        for principal in principals:
            assert principal not in body
    rendered = [
        f"{r.levelname} {r.name} {r.getMessage()} {r.args!r} {r.exc_text or ''}"
        for r in caplog.records
    ]
    for reason in (absent, foreign):
        expected = sum(1 for *_rest, r in refusals if r == reason)
        seen = sum(1 for line in rendered if reason in line)
        assert seen >= expected, (
            f"{seen} records name {reason} for {expected} refusals: the capture "
            f"cannot see the refusal path, so its silence proves nothing"
        )
    for line in rendered:
        for principal in principals:
            assert principal not in line
    for principal in principals:
        assert principal not in json.dumps(state_log)


def test_a_degraded_mint_reads_as_failure_and_carries_no_principal(
    coordinator, client: _Client, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A watchdog-timed-out claim must not read as success, and must not carry
    a principal: the claimant recovers a claim that landed late by retrying
    with its persisted nonce (R20), never from a degraded body."""
    from concurrent.futures import TimeoutError as FuturesTimeout

    def _timeout(fn, abort=None, deadline=None):
        raise FuturesTimeout()

    monkeypatch.setattr(coordinator, "run_with_watchdog", _timeout)
    status, body = _claim_wire(client, _sid("degraded"), secrets.token_urlsafe(32))
    assert status == 200
    assert body == {"ok": False, "degraded": True, "reason": "claim_unconfirmed"}
    assert body["reason"] not in HOLD_REASONS


# ----------------------------------------------------------------------
# Targeted grant handoff routes (#185, U5)
# ----------------------------------------------------------------------

from ccs.adapters.claude_code.coordinator_server import (  # noqa: E402
    _MIGRATION_REJECTED_ROUTES,
)

#: FROZEN duplicate of the four handoff routes and their attempt counters.
_HANDOFF_ROUTE_COUNTERS = {
    "/handoff/transfer": "handoff_transfer_total",
    "/handoff/accept": "handoff_accept_total",
    "/handoff/decline": "handoff_decline_total",
    "/handoff/withdraw": "handoff_withdraw_total",
}


@pytest.mark.parametrize("route", sorted(_HANDOFF_ROUTE_COUNTERS))
def test_handoff_route_is_registered_counted_and_served_during_a_drain(
    route: str, coordinator, client: _Client
) -> None:
    """Each handoff verb is registered in the central table (so the bearer and
    Host seam applies), is counted in BOTH counter registrations -- the
    increment helper silently ignores a name missing from the counter dict,
    so the bump itself is asserted -- and keeps serving during a migration
    drain: a transfer initiates no write, and its epoch move is the
    release-class bump the drain performs itself (R37)."""
    counter = _HANDOFF_ROUTE_COUNTERS[route]
    assert ("POST", route) in _ROUTES
    assert _ENDPOINT_COUNTER_NAMES[("POST", route)] == counter
    assert ("POST", route) not in _MIGRATION_REJECTED_ROUTES
    before = coordinator.endpoint_counters_snapshot()[counter]
    coordinator._migration_draining = True
    try:
        status, body = client.post(route, {"session_id": _sid("handoff-drain")})
    finally:
        coordinator._migration_draining = False
    assert status == 400, (status, body)  # the handler answered: not the drain's 503
    assert coordinator.endpoint_counters_snapshot()[counter] == before + 1


@pytest.mark.parametrize("route", sorted(_HANDOFF_ROUTE_COUNTERS))
def test_handoff_route_without_bearer_is_401(coordinator, route: str) -> None:
    url = f"http://127.0.0.1:{coordinator.port}{route}"
    req = urlrequest.Request(
        url, data=b"{}", method="POST",
        headers={"Host": "127.0.0.1", "Content-Type": "application/json"},
    )
    with pytest.raises(urlerror.HTTPError) as err:
        urlrequest.urlopen(req, timeout=5)
    assert err.value.code == 401


# ----------------------------------------------------------------------
# Transfer-record eviction on the sweep (#185, U6, KTD10)
# ----------------------------------------------------------------------

from ccs.adapters.claude_code import foreign_write_detector as _detector  # noqa: E402
from ccs.coordinator.registry_protocol import TransferRequest  # noqa: E402
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry  # noqa: E402
from ccs.core.states import MESIState  # noqa: E402
from ccs.core.types import Artifact  # noqa: E402


class _OneTickService:
    """The grant and session sweeps as no-ops; the last ends the loop after
    the tick it runs in, so the eviction passes behind it run exactly once."""

    def __init__(self, coordinator: SimpleNamespace) -> None:
        self._coordinator = coordinator

    def enforce_transient_timeouts(self, **_kwargs) -> int:
        return 0

    def enforce_stable_grant_timeouts(self, **_kwargs) -> int:
        return 0

    def enforce_session_liveness(self, **_kwargs) -> int:
        self._coordinator.shutting_down = True
        return 0


def _sweep_one_tick(registry: object, *, transfer_record_evict_max_age_sec: float) -> None:
    """Drive ONE tick of the real ``_sweep_loop`` over ``registry``."""
    coord = SimpleNamespace(shutting_down=False, registry=registry)
    coord.service = _OneTickService(coord)
    cfg = SimpleNamespace(
        sweep_interval_sec=0.01,
        transient_timeout_sec=5,
        grant_heartbeat_timeout_sec=120,
        grant_max_hold_sec=300,
        notice_evict_max_age_sec=600,
        transfer_record_evict_max_age_sec=transfer_record_evict_max_age_sec,
    )
    _lifecycle._sweep_loop(SimpleNamespace(coordinator=coord), cfg)


def _handed_off(registry: SqliteArtifactRegistry, name: str) -> uuid.UUID:
    """A path at v1 whose SHARED reader handed it to a successor: a live record."""
    artifact = Artifact(id=uuid.uuid4(), name=name, version=1, content_hash="h0")
    registry.register_artifact(artifact, "")
    holder = uuid.uuid4()
    registry.set_agent_state(artifact.id, holder, MESIState.SHARED, trigger="fetch", tick=1)
    request = TransferRequest(
        giver=uuid.uuid4(), successor=uuid.uuid4(), holders={artifact.id: holder},
        successor_known=True,
    )
    [outcome] = registry.transfer_grants(request, tick=2)
    assert outcome.transferred, outcome
    return artifact.id


def test_the_transfer_record_eviction_knob_defaults_to_one_day() -> None:
    """KTD10: an ended record outlives a writer left idle overnight, so its
    next touch still learns how its handoff ended."""
    assert _lifecycle.LifecycleConfig().transfer_record_evict_max_age_sec == 86_400.0


def test_the_sweep_evicts_ended_transfer_records_older_than_its_knob_and_never_a_live_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KTD10 on the sweep: a declined record, and a pending one whose version
    moved with no label written, are kept while younger than the knob and
    evicted once older; a live record survives any knob, because removing it
    would lift the giver's fence with nobody having ended the handoff."""
    monkeypatch.setattr(_detector, "run_detection_pass", lambda *_args, **_kwargs: 0)
    with SqliteArtifactRegistry(tmp_path / "sweep.db") as registry:
        live = _handed_off(registry, "live.md")
        declined = _handed_off(registry, "declined.md")
        registry.set_transfer_status(declined, "declined")
        moved = _handed_off(registry, "moved.md")
        won = registry.commit_cas(moved, uuid.uuid4(), expected_version=1, content_hash="h1")
        assert isinstance(won, tuple), won
        assert registry.get_transfer_record(moved)[0].status == "pending"

        _sweep_one_tick(registry, transfer_record_evict_max_age_sec=3600.0)
        assert [registry.get_transfer_record(a)[1] for a in (live, declined, moved)] == [
            True, False, False]

        # The tick sleeps one sweep interval (10ms) before it runs, so every
        # stamp above is older than a 1ms knob by then.
        _sweep_one_tick(registry, transfer_record_evict_max_age_sec=0.001)
        assert registry.get_transfer_record(live)[1] is True
        assert registry.get_transfer_record(declined) is None
        assert registry.get_transfer_record(moved) is None


def test_the_sweep_evicts_transfer_records_after_notices_under_the_same_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KTD10: the eviction runs right after notice eviction with its own knob,
    inside the best-effort guard the other passes share, so a failing eviction
    costs this tick nothing else and never takes the sweep down."""
    calls: list[tuple[str, float]] = []
    detected: list[int] = []

    class _Registry:
        def evict_stale_notices(self, *, max_age_sec: float) -> int:
            calls.append(("notices", max_age_sec))
            return 0

        def evict_transfer_records(self, *, max_age_sec: float) -> int:
            calls.append(("transfer_records", max_age_sec))
            raise RuntimeError("injected eviction failure")

    def _detection(*_args, **_kwargs) -> int:
        detected.append(1)
        return 0

    monkeypatch.setattr(_detector, "run_detection_pass", _detection)
    _sweep_one_tick(_Registry(), transfer_record_evict_max_age_sec=7200.0)

    assert calls == [("notices", 600), ("transfer_records", 7200.0)]
    assert detected == [1]
