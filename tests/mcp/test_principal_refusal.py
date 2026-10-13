# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""A session whose caller principal the coordinator refuses, on the MCP surface.

The server's volume is strict, and once its session is known to be bound
under another mint nonce the volume claims nothing again: every later
``swg_write`` / ``swg_read`` / ``swg_gate`` raises the typed
``CallerPrincipalRefused``. What the agent must see is that typed reason and a
recover verb saying the state is durable for this server session — never the
generic ``internal_error`` / ``none`` it cannot tell from a coordinator bug —
and ``swg_status`` must say so ahead of the next refused call. Driven through
the sync ``_do_*`` helpers against a real loopback coordinator, as the other
tool tests are.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.mcp.server import _do_gate, _do_read, _do_status, _do_write
from ccs.mcp.session import SessionConfig

#: FROZEN duplicates of the wire reason and the recover verbs (never imported
#: from the code under test): the verb for a refusal that is settled for the
#: session, and the one for a refusal whose recovery claim's answer was lost.
_FOREIGN = "caller_principal_foreign"
_RECOVER = "restart_session"
_UNSETTLED_RECOVER = "wait_and_retry"
#: In principal / nonce shape, minted by nobody: a principal that is not the
#: session's, and a nonce that is not the one its binding was made under.
_OTHER_PRINCIPAL = "X" * 43
_OTHER_NONCE = "Z" * 43


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


def _seed(tmp_path: Path, rel: str = "data/shared.txt", content: bytes = b"v1") -> Path:
    target = tmp_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _rendered(result) -> str:
    """Everything a client can see of one tool result."""
    return result.content[0].text + json.dumps(result.structuredContent)


def test_a_refused_session_gets_the_typed_deny_on_every_later_tool_call_and_status_says_so(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """The session is bound; the volume then presents a principal that is not
    the bound one and holds a nonce that is not the binding's, so the
    coordinator refuses every route and the recovery claim is refused too.
    From then on each tool call answers ``reason: caller_principal_foreign``,
    ``recover: restart_session``, ``retryable: false`` — the write lands
    nothing, the read and the gate meet the same answer without a further
    claim — and ``swg_status`` reports ``principal_claim: refused`` beside the
    coordinator's two principal counters while still saying the coordinator
    is ``on`` and attached (the coordinator is fine; this session is not).
    Nothing rendered carries a principal or a nonce."""
    target = _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        bound, nonce = vol._principal, vol._mint_nonce
        assert bound and nonce, "control: the session claimed a principal at attach"
        assert _do_status(vol, config).structuredContent["principal_claim"] == "bound"
        read = _do_read(vol, config, "data/shared.txt")
        assert read.isError is False
        version = read.structuredContent["version"]
        generation = read.structuredContent["owner_generation"]

        vol._principal, vol._mint_nonce = _OTHER_PRINCIPAL, _OTHER_NONCE
        results = {
            "write": _do_write(vol, config, "data/shared.txt", "v2"),
            "read": _do_read(vol, config, "data/shared.txt"),
            "gate": _do_gate(vol, config, "data/shared.txt", version, generation),
        }
        for name, result in results.items():
            sc = result.structuredContent
            assert result.isError is True, name
            assert sc["reason"] == _FOREIGN, (name, sc)
            assert sc["reason"] != "internal_error", name
            assert sc["recover"] == _RECOVER, (name, sc)
            assert sc["retryable"] is False, name
            for secret in (bound, nonce, _OTHER_PRINCIPAL, _OTHER_NONCE):
                assert secret not in _rendered(result), f"{name}: a principal or nonce was rendered"
        assert target.read_bytes() == b"v1", "the refused write landed"

        status = _do_status(vol, config).structuredContent
        assert status["coordinator"] == "on" and status["is_attached"] is True
        assert status["principal_claim"] == "refused"
        # Three refused requests (pre-edit, pre-read, effect-fence); the
        # recovery claim's own refusal is the mint's answer, not the gate's.
        assert status["caller_principal_refused_total"] == 3
        assert status["caller_principal_absent_total"] == 0
    finally:
        stop_coordinator(tmp_path)


def test_a_refusal_whose_recovery_claim_is_unconfirmed_is_retryable_and_the_next_call_recovers(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The session is bound and the volume presents a principal that is not
    the bound one while holding the binding's own nonce; the coordinator
    refuses the write, and the recovery claim's answer is lost (a transport
    blip). The tool call answers the typed reason with ``recover:
    wait_and_retry`` and ``retryable: true`` — not ``restart_session`` — and
    ``swg_status`` reads ``principal_claim: unconfirmed``; the next tool call
    claims again with the same nonce by itself, succeeds, and status reads
    ``bound``. Nothing rendered carries a principal or the nonce.

    Prevents the deny over-claiming durability: ``restart_session`` /
    ``retryable: false`` for a state the very next call cures would send the
    agent off to a new session, and every later call in the old one worked."""
    import ccs.adapters.coherent_volume as coherent_volume_module
    from ccs.cli._coherence_client import PrincipalClaim

    target = _seed(tmp_path)
    config = SessionConfig(root=tmp_path.resolve(), managed=("data/**",))
    vol = CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=fast_cfg)
    try:
        bound, nonce = vol._principal, vol._mint_nonce
        assert bound and nonce, "control: the session claimed a principal at attach"
        assert _do_read(vol, config, "data/shared.txt").isError is False
        real_claim = coherent_volume_module.claim_caller_principal
        nonces: list[str] = []

        def lossy(endpoint, session_id, presented_nonce):  # noqa: ANN001, ANN202
            nonces.append(presented_nonce)
            claim = real_claim(endpoint, session_id, presented_nonce)
            assert claim.outcome == "bound", "control: the lost claim really answered"
            return PrincipalClaim("unconfirmed", detail="answer lost") if len(nonces) == 1 else claim

        monkeypatch.setattr(coherent_volume_module, "claim_caller_principal", lossy)
        vol._principal = _OTHER_PRINCIPAL

        denied = _do_write(vol, config, "data/shared.txt", "v2")
        sc = denied.structuredContent
        assert denied.isError is True
        assert sc["reason"] == _FOREIGN and sc["reason"] != "internal_error"
        assert sc["recover"] == _UNSETTLED_RECOVER and sc["recover"] != _RECOVER
        assert sc["retryable"] is True
        assert target.read_bytes() == b"v1", "the refused write landed"
        assert nonces == [nonce], "the recovery claim presented the binding's own nonce"
        status = _do_status(vol, config).structuredContent
        assert status["principal_claim"] == "unconfirmed"
        assert status["coordinator"] == "on" and status["is_attached"] is True

        read = _do_read(vol, config, "data/shared.txt")
        assert read.isError is False, read.structuredContent
        assert read.structuredContent["content"] == "v1"
        assert nonces == [nonce, nonce], "the next call claimed again by itself, same nonce"
        assert vol._principal == bound
        assert _do_status(vol, config).structuredContent["principal_claim"] == "bound"
        for secret in (bound, nonce, _OTHER_PRINCIPAL):
            assert secret not in _rendered(denied), "a principal or nonce was rendered"
    finally:
        stop_coordinator(tmp_path)
