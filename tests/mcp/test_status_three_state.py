"""Unit 4 — swg_status is honest about the THREE states.

The load-bearing bug (SC5): never report ``unknown`` (coordinator unreachable)
as ``off`` (reachable, no strict patterns). A caller that reads ``off`` may write
unguarded; ``unknown`` must not collapse to that. The 3-state logic is tested as
a pure function over synthetic ``/status`` docs, plus one live ``on`` check.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.mcp.server import _STATUS_DESC, _do_status
from ccs.mcp.session import SessionConfig
from ccs.mcp.status import _coordinator_state, _per_path, build_status, handoff_from_status


class _StubVolume:
    def __init__(
        self,
        attached: bool,
        degraded: bool = False,
        session: str = "sess-1",
        claim: str = "bound",
        doc: dict | None = None,
    ) -> None:
        self.is_attached = attached
        self.is_degraded = degraded
        self.session_id = session
        self.principal_claim_outcome = claim
        self._doc = doc

    def coordinator_status(self) -> dict | None:
        return self._doc


#: A frozen copy of the text ``swg_status`` adds when ``per_path`` is null,
#: never imported from the code under test.
_UNAVAILABLE_MARKER = "per_path=unavailable"


def _doc(count: int | None) -> dict:
    summary = {} if count is None else {"strict_mode_pattern_count": count}
    return {"policy_summary": summary}


def test_build_status_reports_the_claim_outcome_and_forwards_the_principal_counters() -> None:
    """``principal_claim`` is the volume's own claim outcome, so an agent can
    tell "this session lost coordination" (``refused``) ahead of its next
    refused call; the two counters are forwarded from the coordinator's
    ``/status`` document verbatim. A reported zero is a zero."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))
    doc = {**_doc(1), "caller_principal_absent_total": 3, "caller_principal_refused_total": 0}

    status = build_status(_StubVolume(True, claim="refused", doc=doc), config)

    assert status["coordinator"] == "on"
    assert status["principal_claim"] == "refused"
    assert status["caller_principal_absent_total"] == 3
    assert status["caller_principal_refused_total"] == 0


@pytest.mark.parametrize(
    "doc",
    [
        None,
        _doc(1),
        {**_doc(1), "caller_principal_absent_total": "3", "caller_principal_refused_total": True},
    ],
    ids=["unreachable", "older-coordinator", "not-an-integer"],
)
def test_build_status_never_reports_an_unreported_principal_counter_as_zero(doc) -> None:
    """Cannot-tell never collapses into zero (the SC5 rule, applied to the
    counters): an unreachable coordinator, an older one that publishes no
    principal counters, and a value that is not an integer (a bool is an int
    subclass and is excluded) each read ``None`` — a reader that saw ``0``
    would take it for "no refusals". The claim outcome is the volume's own
    and is reported regardless."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))

    status = build_status(_StubVolume(True, claim="unconfirmed", doc=doc), config)

    assert status["caller_principal_absent_total"] is None
    assert status["caller_principal_refused_total"] is None
    assert status["principal_claim"] == "unconfirmed"


@pytest.mark.parametrize("count", [1, 0], ids=["on", "off"])
def test_build_status_forwards_the_registry_schema_version(count: int) -> None:
    """``registry_schema_version`` is forwarded from the coordinator's
    ``/status`` document verbatim, as the principal counters are, whether
    strict enforcement is on or off: it describes the store, not the policy.
    Prevents an agent having to reach the coordinator itself to learn whether
    an older release could still open the workspace's store."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))
    doc = {**_doc(count), "registry_schema_version": 10}

    status = build_status(_StubVolume(True, doc=doc), config)

    assert status["registry_schema_version"] == 10


@pytest.mark.parametrize(
    "doc",
    [
        None,
        _doc(1),
        {**_doc(1), "registry_schema_version": None},
        {**_doc(1), "registry_schema_version": "10"},
        {**_doc(1), "registry_schema_version": True},
    ],
    ids=["unreachable", "older-or-node-coordinator", "null", "not-an-integer", "bool"],
)
def test_build_status_never_invents_a_registry_schema_version(doc) -> None:
    """Cannot-tell stays ``None`` (the SC5 rule): an unreachable coordinator,
    one that sends no ``registry_schema_version`` (an older release, or the
    Node coordinator, whose ``schema_version`` counts a different ledger), and
    a value that is not an integer each read ``None``. Prevents an agent
    comparing a made-up or foreign number against a release's schema."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))
    if doc is not None:
        doc = {**doc, "schema_version": 5}

    status = build_status(_StubVolume(True, doc=doc), config)

    assert status["registry_schema_version"] is None


def test_status_text_does_not_carry_the_registry_schema_version() -> None:
    """Pin: the schema version is structured content only; the text result is
    the coordinator state, as before."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))
    doc = {**_doc(1), "tracked_artifacts": [], "registry_schema_version": 10}

    result = _do_status(_StubVolume(True, doc=doc), config)

    assert result.structuredContent["registry_schema_version"] == 10
    assert _texts(result) == ["coordinator=on"]


def test_state_on_when_reachable_with_strict_patterns() -> None:
    assert _coordinator_state(_StubVolume(True), _doc(2)) == "on"


def test_state_off_when_reachable_with_no_strict_patterns() -> None:
    assert _coordinator_state(_StubVolume(True), _doc(0)) == "off"


def test_state_unknown_when_unreachable_never_off() -> None:
    """SC5: a None status doc (unreachable) is unknown, NOT off."""
    assert _coordinator_state(_StubVolume(True), None) == "unknown"


def test_state_unknown_when_unattached() -> None:
    assert _coordinator_state(_StubVolume(False), _doc(2)) == "unknown"


def test_state_unknown_when_count_missing() -> None:
    assert _coordinator_state(_StubVolume(True), _doc(None)) == "unknown"


def test_per_path_enforced_vs_not_registered() -> None:
    doc = {
        "tracked_artifacts": [
            {"path": "data/a.txt", "version": 3},
            {"path": "other/b.txt", "version": 1},
        ]
    }
    config = SessionConfig(root=Path("/x"), managed=("data/**",))
    per_path = _per_path(config, doc)
    assert per_path["data/a.txt"] == {"version": 3, "status": "enforced"}
    assert per_path["other/b.txt"] == {"version": 1, "status": "not_registered"}


def test_per_path_empty_when_no_status() -> None:
    assert _per_path(SessionConfig(root=Path("/x"), managed=("data/**",)), None) == {}


def _degraded_doc() -> dict:
    """A default-tier ``/status`` answer whose registry read timed out: the
    summary and counters as normal, both lists null, and ``degraded: true``."""
    return {
        **_doc(1),
        "caller_principal_absent_total": 0,
        "caller_principal_refused_total": 0,
        "tracked_artifacts": None,
        "sessions": None,
        "degraded": True,
    }


def test_per_path_is_none_when_status_is_degraded() -> None:
    """A degraded ``/status`` cannot say which paths are tracked, so
    ``per_path`` is ``None``, never ``{}``; the coordinator state still comes
    from the summary, so it stays ``on``. Prevents a busy registry reading as
    "nothing tracked" -- with ``coordinator`` on, ``per_path`` is the only
    signal."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))

    status = build_status(_StubVolume(True, doc=_degraded_doc()), config)

    assert status["per_path"] is None
    assert status["coordinator"] == "on"


def _texts(result) -> list[str]:
    return [item.text for item in result.content]


def test_status_text_says_per_path_unavailable_when_status_is_degraded() -> None:
    """The text result of a degraded ``/status`` says the tracked paths could
    not be told. Prevents a client that shows only the text channel taking
    ``coordinator=on``, which reads exactly like a healthy answer, for one."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))

    result = _do_status(_StubVolume(True, doc=_degraded_doc()), config)

    assert result.structuredContent["per_path"] is None
    text = " ".join(_texts(result))
    assert text.startswith("coordinator=on ")
    assert _UNAVAILABLE_MARKER in text
    assert "do not treat this as nothing tracked" in text


@pytest.mark.parametrize(
    ("doc", "expected"),
    [
        ({**_doc(1), "tracked_artifacts": []}, ["coordinator=on"]),
        ({**_doc(1), "tracked_artifacts": [{"path": "data/a.txt", "version": 2}]}, ["coordinator=on"]),
        ({**_doc(0), "tracked_artifacts": []}, ["coordinator=off"]),
        (None, ["coordinator=unknown"]),
    ],
    ids=["on-empty", "on-tracked", "off", "unreachable"],
)
def test_status_text_is_unchanged_when_per_path_is_reported(doc, expected) -> None:
    """Pin: an answer whose per-path state was reported keeps its text byte
    for byte; only a null ``per_path`` adds the marker."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))

    assert _texts(_do_status(_StubVolume(True, doc=doc), config)) == expected


@pytest.mark.parametrize(
    "artifacts",
    ["missing", None, {"data/a.txt": 1}],
    ids=["key-missing", "null-without-degraded", "dict-not-list"],
)
def test_per_path_is_none_for_any_answer_without_an_artifact_list(artifacts) -> None:
    """Any ``/status`` answer without an artifact list cannot tell what is
    tracked, marked ``degraded`` or not, so ``per_path`` is ``None`` and the
    text says so. Prevents keying on ``degraded`` alone, which would read a
    malformed answer as "nothing tracked"."""
    config = SessionConfig(root=Path("/x"), managed=("data/**",))
    doc = _doc(1) if artifacts == "missing" else {**_doc(1), "tracked_artifacts": artifacts}

    result = _do_status(_StubVolume(True, doc=doc), config)

    assert result.structuredContent["per_path"] is None
    assert _UNAVAILABLE_MARKER in " ".join(_texts(result))


def test_the_status_description_says_when_per_path_is_null_and_when_it_is_empty() -> None:
    """The description gives ``per_path`` the meanings the code gives it: null
    when the ``/status`` answer carries no list of tracked paths, a busy
    registry being one cause, with the text result naming it; ``{}`` when the
    coordinator state is ``unknown``. Prevents an agent taking a busy
    registry's ``coordinator=on`` answer for an empty workspace, or an
    unreachable coordinator's ``{}`` for one."""
    assert "per_path is null, not {}" in _STATUS_DESC
    assert "carries no list of tracked paths" in _STATUS_DESC
    assert "With coordinator=unknown, per_path is {} and says nothing about what is tracked" in _STATUS_DESC
    assert "registry was busy" in _STATUS_DESC
    assert "retry shortly" in _STATUS_DESC
    assert "do not treat it as nothing tracked" in _STATUS_DESC
    assert _UNAVAILABLE_MARKER in _STATUS_DESC


def test_handoff_from_status_cannot_tell_on_a_degraded_status() -> None:
    """Pin: a degraded ``/status`` has no artifact list, so a path's transfer
    record cannot be told -- ``(False, None)``, never "no record"."""
    volume = _StubVolume(True, doc=_degraded_doc())

    assert handoff_from_status(volume, "data/a.txt") == (False, None)


# --- live integration --------------------------------------------------------


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


def test_build_status_live_reports_on(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "shared.txt").write_bytes(b"v1")
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    # build_volume ignores the config's LifecycleConfig; construct directly for speed.
    from ccs.adapters.coherent_volume import CoherentVolume

    volume = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        status = build_status(volume, config)
        assert status["coordinator"] == "on"
        assert status["is_attached"] is True
        assert status["is_degraded"] is False
        assert status["single_host_only"] is True
        assert status["heterogeneous_scope_detectable"] is False
        assert status["managed"] == ["data/**"]
        # A healthy session: its principal is bound, and the live coordinator
        # reports both counters as integers (zero here — nothing was refused).
        assert status["principal_claim"] == "bound"
        assert status["caller_principal_absent_total"] == 0
        assert status["caller_principal_refused_total"] == 0
    finally:
        stop_coordinator(tmp_path)


def test_build_status_live_forwards_the_stores_schema_version(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """Against a real coordinator, ``registry_schema_version`` is the schema
    version stamped in the workspace's own store, read from the file here
    rather than from the code under test. Prevents the forward reading a key
    the coordinator does not send, which the stub documents cannot see."""
    (tmp_path / "data").mkdir()
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    from ccs.adapters.coherent_volume import CoherentVolume

    volume = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        status = build_status(volume, config)
        store = sqlite3.connect(f"{(tmp_path / '.coherence' / 'state.db').as_uri()}?mode=ro", uri=True)
        try:
            on_disk = store.execute("PRAGMA user_version").fetchone()[0]
        finally:
            store.close()
        assert type(status["registry_schema_version"]) is int
        assert status["registry_schema_version"] == on_disk
    finally:
        stop_coordinator(tmp_path)
