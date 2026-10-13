# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Caller-principal corpus: the route posture across the two runtimes (plan U6).

Only the Python coordinator mints a caller principal and enforces the route
posture; the sibling Node coordinator answers 404 on ``/principal/claim`` and
ignores the header (KTD12). This module pins that asymmetry the way the
effect-fence corpus pins a route one implementation lacks: node-only rows
record what Node does, Python-only rows what Python does, and both-backend rows
assert only what the two agree on.

What a principal buys is accident-resistance and attributability under the
same-OS-user cooperative trust model. On the hook surface the principal is
stored under ``.coherence/``, so there it is convention-enforcement and a
detectable unbound caller, never separation between callers.

Also here, because they need the built sibling: the harness's backend-scoped
preflight (the capability those rows are written in), and the cross-runtime
client check — the Node hook client obtains its principal from the PYTHON
coordinator and its commit is attributed to its session, and the Python hook
client presents the principal the Node client stored.

Marked ``protocol_corpus`` — opt-in via ``pytest -m protocol_corpus``. A Node
row that cannot run FAILS rather than xfails: an unrun asymmetry row reports
green while checking nothing."""

from __future__ import annotations

import io
import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from tests.protocol_corpus.harness import (
    BACKEND_NODE,
    BACKEND_PYTHON,
    NOT_ISSUED,
    Fixture,
    FixtureContractError,
    FixtureSubstitutionError,
    _omit_headers_not_issued,
    _substitute,
    build_fixture,
    load_fixtures,
    normalize_response,
    resolve_node_dist_path,
    run_scenario,
)

pytestmark = pytest.mark.protocol_corpus

_FIXTURE_DIR = "caller_principal"
_EXPECTED_FIXTURE_COUNT = 18
_HEADER = "Coherence-Caller-Principal"  # frozen duplicate of the wire name

_NODE_DIST_PATH = resolve_node_dist_path()
_NODE_DIST_UNRESOLVED = (
    "KTD12 needs a REAL answer from the sibling coordinator and this row got "
    "none: the plugin dist could not be resolved. Failing rather than xfailing "
    "is deliberate — an asymmetry row that never ran reports green while "
    "checking nothing. Build the plugin (npm ci && npm run build) or set "
    "AGENT_COHERENCE_PLUGIN_DIST_PATH to the absolute dist/coordinator.js path."
)


def _fixtures() -> list[Fixture]:
    return load_fixtures(_FIXTURE_DIR)


def _rows() -> list[tuple[Fixture, str]]:
    return [(f, b) for f in _fixtures() for b in f.backends]


_ROWS = _rows()


@pytest.mark.parametrize("row", _ROWS, ids=[f"{f.name}[{b}]" for f, b in _ROWS])
def test_caller_principal_fixture_response_matches_expected(
    row: tuple[Fixture, str], tmp_path: Path
) -> None:
    """The full response body is the assertion. A Python refusal that changed
    its text, a Node route that started refusing, or a Node that grew the mint
    all fail here."""
    fixture, backend = row
    if backend == BACKEND_NODE and _NODE_DIST_PATH is None:
        pytest.fail(f"{fixture.name}: {_NODE_DIST_UNRESOLVED}")
    status, body = run_scenario(
        fixture=fixture, backend_id=backend, workspace=tmp_path,
        node_dist_path=_NODE_DIST_PATH,
    )
    expected = normalize_response(
        fixture.expected["body"],
        ignore_keys=fixture.ignore_keys,
        optional_keys=fixture.optional_keys,
    )
    assert status == fixture.expected["status"], f"{fixture.name}[{backend}]: {body!r}"
    assert body == expected, f"{fixture.name}[{backend}]\nexpected={expected!r}\nactual=  {body!r}"


def test_fixture_directory_is_actually_loaded() -> None:
    """A misspelled or emptied fixture directory loads as nothing and a
    parametrize over nothing reports green; the frozen count makes it red."""
    assert len(_fixtures()) == _EXPECTED_FIXTURE_COUNT


def test_the_asymmetry_is_recorded_on_both_sides() -> None:
    """KTD12 in the corpus, stated as pairs: for each require-class check the
    corpus pins, a Python row that REFUSES and a Node row that ADMITS the same
    request after the same setup — and a Node row pinning the 404 on the mint.
    A pair with one side missing would let either runtime drift silently.

    The pair is a control only while EVERYTHING before the answer matches:
    the setup as well as the request. And the setup must claim the session the
    request names, unscoped and capture-free, so it runs on Node too (where it
    answers 404): without the claim the session is unbound, Python admits it
    as well (KTD15), and the "Node" row records nothing Python does not also
    do — which is what fixtures 04 and 08 once did, green on both runtimes."""
    by_name = {f.name: f for f in _fixtures()}
    pairs = [
        ("caller-principal-python-session-stop-refuses-an-absent-principal",
         "caller-principal-node-session-stop-admits-an-absent-principal"),
        ("caller-principal-python-post-edit-refuses-an-absent-principal",
         "caller-principal-node-post-edit-admits-an-absent-principal"),
        ("caller-principal-python-pre-edit-refuses-an-absent-principal",
         "caller-principal-node-pre-edit-admits-an-absent-principal"),
    ]
    for python_name, node_name in pairs:
        refuse, admit = by_name[python_name], by_name[node_name]
        assert refuse.backends == (BACKEND_PYTHON,) and admit.backends == (BACKEND_NODE,)
        assert refuse.request == admit.request, "the pair is a control only while the requests match"
        assert refuse.setup == admit.setup, "…and only while the setups match"
        assert _HEADER not in (refuse.request.get("headers") or {}), python_name
        claims = [
            req for req in refuse.setup.get("preflight_requests") or []
            if req["path"] == "/principal/claim"
        ]
        assert [req["body"]["session_id"] for req in claims] == [
            refuse.request["body"]["session_id"]
        ], f"{python_name}: the setup must claim the session the request names"
        assert all("backends" not in req and "capture" not in req for req in claims), (
            f"{python_name}: a scoped or capturing claim never reaches Node"
        )
        assert refuse.expected["status"] == 400 and admit.expected["status"] == 200
        assert refuse.expected["body"].get("reason") == "caller_principal_absent"
    mint = by_name["caller-principal-node-sibling-does-not-implement-claim-404"]
    assert mint.backends == (BACKEND_NODE,) and mint.expected["status"] == 404
    assert "headers" not in mint.request, "the bearer must stay the harness's valid one"


def test_every_principal_refusal_row_carries_its_typed_reason() -> None:
    """The refusal's wire contract: HTTP 400 with ``error`` (the prose every
    non-200 carries) AND a typed ``reason`` a client classifies by equality.
    Every 400 row here pins both, and the two agree, so a Python refusal that
    dropped the key — the one field the clients branch on — goes red."""
    refusals = [f for f in _fixtures() if f.expected["status"] == 400]
    assert len(refusals) == 5
    for row in refusals:
        body = row.expected["body"]
        assert set(body) == {"error", "reason"}, row.name
        assert body["reason"] in {"caller_principal_absent", "caller_principal_foreign"}
        assert body["error"].endswith(f"({body['reason']})"), row.name


def test_the_node_rows_cannot_be_satisfied_by_a_skip() -> None:
    """The resolution itself is asserted: a machine without the built plugin
    fails this gate instead of quietly passing it."""
    assert _NODE_DIST_PATH is not None, _NODE_DIST_UNRESOLVED
    assert _NODE_DIST_PATH.exists()
    assert any(BACKEND_NODE in f.backends for f in _fixtures())


# ----------------------------------------------------------------------
# The harness capability these rows are written in: a backend-scoped preflight
# ----------------------------------------------------------------------


def _raw(preflight: list[dict], request: dict, backends: list[str] | None = None) -> dict:
    return {
        "name": "scoped-capture-probe",
        "setup": {"preflight_requests": preflight},
        "request": request,
        "expected": {"status": 200, "body": {}},
        **({"backends": backends} if backends else {}),
    }


_SCOPED_CLAIM = {
    "method": "POST", "path": "/principal/claim", "backends": ["python"],
    "body": {"session_id": "c0c0c0c0-0000-4000-8000-00000000000a", "mint_nonce": "corpus-mint-nonce-a"},
    "capture": {"principal_a": "principal"},
}


def test_a_backend_scoped_capture_outside_a_whole_header_value_is_refused_at_load() -> None:
    """A capture only one backend issues has no honest stand-in anywhere but
    an omittable header: in a body it would go out as a literal or a null that
    draws a rejection a fixture could mistake for the thing it asserts."""
    in_body = {"method": "POST", "path": "/hooks/session-stop",
               "body": {"session_id": "${principal_a}"}}
    with pytest.raises(FixtureContractError, match="outside a request header"):
        build_fixture(_raw([_SCOPED_CLAIM], in_body), Path("probe.json"))
    embedded = {"method": "POST", "path": "/hooks/session-stop", "body": {},
                "headers": {_HEADER: "Bearer ${principal_a}"}}
    with pytest.raises(FixtureContractError, match="whole value"):
        build_fixture(_raw([_SCOPED_CLAIM], embedded), Path("probe.json"))
    whole = {"method": "POST", "path": "/hooks/session-stop", "body": {},
             "headers": {_HEADER: "${principal_a}"}}
    build_fixture(_raw([_SCOPED_CLAIM], whole), Path("probe.json"))  # the control loads


def test_a_scope_must_name_known_backends_and_never_the_main_request() -> None:
    bad_scope = {**_SCOPED_CLAIM, "backends": ["ruby"]}
    with pytest.raises(FixtureContractError, match="subset"):
        build_fixture(_raw([bad_scope], {"method": "GET", "path": "/status"}), Path("probe.json"))
    empty_scope = {**_SCOPED_CLAIM, "backends": []}
    with pytest.raises(FixtureContractError, match="subset"):
        build_fixture(_raw([empty_scope], {"method": "GET", "path": "/status"}), Path("probe.json"))
    scoped_main = {"method": "GET", "path": "/status", "backends": ["python"]}
    with pytest.raises(FixtureContractError, match="main request"):
        build_fixture(_raw([], scoped_main), Path("probe.json"))


def test_a_header_naming_a_capture_not_issued_here_is_omitted_not_sent() -> None:
    """Where the scoped preflight did not run, the header is dropped — what a
    conforming client does after a 404 on the claim — and the literal
    ``${...}`` never reaches the wire. Any other use of a not-issued value is
    refused at run time too, for fixtures mutated in memory past the load
    check."""
    captures = {"principal_a": NOT_ISSUED}
    request = {"method": "POST", "path": "/hooks/session-stop", "body": {"session_id": "x"},
               "headers": {_HEADER: "${principal_a}", "X-Other": "kept"}}
    sent = _substitute(_omit_headers_not_issued(request, captures), captures, where="probe")
    assert sent["headers"] == {"X-Other": "kept"}
    issued = {"principal_a": "P" * 43}
    sent = _substitute(_omit_headers_not_issued(request, issued), issued, where="probe")
    assert sent["headers"] == {_HEADER: "P" * 43, "X-Other": "kept"}
    with pytest.raises(FixtureSubstitutionError, match="not issued"):
        _substitute({"body": {"p": "${principal_a}"}}, captures, where="probe")


def test_a_python_refusal_row_binds_the_identity_it_refuses() -> None:
    """An absent principal is refused only for a CLAIMED session (a session
    nobody claimed is an older client's and is admitted), so every Python row
    asserting ``caller_principal_absent`` must claim the session it names in a
    preflight first. A row that stopped claiming would assert a refusal the
    coordinator no longer gives — or, worse, pin a rule that refuses older
    clients."""
    absent_rows = [
        f for f in _fixtures()
        if f.expected["body"].get("reason") == "caller_principal_absent"
    ]
    assert len(absent_rows) == 3
    for row in absent_rows:
        claimed = {
            req["body"]["session_id"]
            for req in row.setup.get("preflight_requests") or []
            if req["path"] == "/principal/claim" and "backends" not in req
        }
        assert row.request["body"]["session_id"] in claimed, row.name


# ----------------------------------------------------------------------
# Cross-runtime client check: the Node hook client against the Python
# coordinator, and the two clients sharing one stored principal
# ----------------------------------------------------------------------


def _node_hook_client(sub: str, payload: dict, root: Path) -> tuple[int, str, str]:
    assert _NODE_DIST_PATH is not None, _NODE_DIST_UNRESOLVED
    client = _NODE_DIST_PATH.parent / "hook_client.js"
    assert client.exists(), f"sibling hook client not built at {client}"
    done = subprocess.run(
        ["node", str(client), sub, "--root", str(root)],
        input=json.dumps(payload), capture_output=True, text=True, timeout=30,
    )
    return done.returncode, done.stdout, done.stderr


def test_the_node_hook_client_is_attributed_through_the_python_coordinator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The sibling client claims from the Python coordinator, persists the
    principal where the Python client would look, and its commit is recorded
    under the acting session's composite writer id. Then the Python hook
    client, for the same session, presents the principal the Node client
    stored — no second claim — and its require-class stop is admitted. One
    binding per session, whichever runtime's client a hook happens to run.

    The files live in ``.coherence/``: sharing them across the two clients is
    convention, readable by any process that can read the directory."""
    from ccs.adapters.claude_code.auth import load_secret
    from ccs.adapters.claude_code.coordinator_server import (
        CoordinatorHTTPServer,
        caller_principal_identity,
        session_to_agent_id,
    )
    from ccs.cli import coherence_hook_client

    if _NODE_DIST_PATH is None:
        pytest.fail(_NODE_DIST_UNRESOLVED)
    (tmp_path / ".git").mkdir()
    (tmp_path / "plan.md").write_text("plan v1")
    server = CoordinatorHTTPServer(tmp_path, port=0, instance_id="cross-runtime-client")
    server.serve_in_thread()
    try:
        assert load_secret(server.coordinator_root)
        (tmp_path / ".coherence" / "server.pid").write_text(f"{os.getpid()}\n{server.port}\n")
        sid = str(uuid.uuid4())
        edit = {"session_id": sid, "tool_input": {"file_path": str(tmp_path / "plan.md")}}

        rc, out, err = _node_hook_client("pre-edit", edit, tmp_path)
        assert rc == 0 and json.loads(out).get("ok") is True, (out, err)
        (tmp_path / "plan.md").write_text("plan v2 by the node client")
        rc, out, err = _node_hook_client("post-edit", edit, tmp_path)
        assert rc == 0 and json.loads(out).get("ok") is True, (out, err)

        artifact_id = server.registry.lookup_artifact_id_by_name("plan.md")
        assert server.registry.last_writer_for(artifact_id) == session_to_agent_id(sid)
        key = caller_principal_identity(sid).hex
        stored = (tmp_path / ".coherence" / f"caller-principal-{key}.principal").read_text().strip()
        assert server.registry.get_caller_principal(caller_principal_identity(sid)) == stored
        claims = server.endpoint_counters_snapshot()["principal_claim_total"]
        assert claims == 1, "the node client claimed once and then read its stored principal"

        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps({"session_id": sid})))
        assert coherence_hook_client.main(["session-stop", "--root", str(tmp_path)]) == 0
        answer = json.loads(capsys.readouterr().out)
        assert answer.get("ok") is True and "released_artifacts" in answer, answer
        assert server.endpoint_counters_snapshot()["principal_claim_total"] == claims
    finally:
        server.shutdown()
