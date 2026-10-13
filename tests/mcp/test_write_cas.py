"""Unit 5 — swg_write_cas (Option A): the concurrent single-host regime, honestly.

Option A is a SINGLE-SHOT version-checked CAS: commit IFF current == the agent's
expected_version, else a TYPED conflict (current_version returned) — never an
auto-merge, never a silent overwrite (the split-comparand lost update). The
per-session counter bounds a cooperating agent's retry loop.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ccs.adapters.claude_code.lifecycle import LifecycleConfig, stop_coordinator
from ccs.adapters.coherent_volume import CoherentVolume
from ccs.core.exceptions import CasVersionConflict
from ccs.mcp.server import (
    _READ_DESC,
    _WRITE_CAS_DESC,
    MAX_CAS_CONFLICTS,
    _do_reacquire,
    _do_read,
    _do_write,
    _do_write_cas,
)
from ccs.mcp.session import SessionConfig
from tests.adapters.test_coherent_volume_split_read import LaggingPeer


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


def _seed(tmp_path: Path, content: bytes = b"v1") -> Path:
    target = tmp_path / "data" / "shared.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return target


def _vol(tmp_path: Path, cfg: LifecycleConfig) -> CoherentVolume:
    return CoherentVolume(tmp_path, managed=("data/**",), on_error="strict", config=cfg)


def _config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(root=tmp_path.resolve(), managed=("data/**",))


_PATH = "data/shared.txt"


# --- happy -------------------------------------------------------------------


def test_write_cas_commits_at_current_version(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    target = _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        version = _do_read(vol, config, _PATH).structuredContent["version"]
        result = _do_write_cas(vol, config, {}, _PATH, version, "v1-merged")
        assert result.isError is False
        assert target.read_bytes() == b"v1-merged"
    finally:
        stop_coordinator(tmp_path)


# --- stale expected_version → typed conflict, no write -----------------------


def test_write_cas_stale_version_is_typed_conflict(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    target = _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol_a = _vol(tmp_path, fast_cfg)
    vol_b = _vol(tmp_path, fast_cfg)
    try:
        version = _do_read(vol_a, config, _PATH).structuredContent["version"]
        # Peer B commits → current advances past A's read version.
        _do_read(vol_b, config, _PATH)
        assert _do_write(vol_b, config, _PATH, "v2-from-b").isError is False

        conflicts: dict[str, int] = {}
        result = _do_write_cas(vol_a, config, conflicts, _PATH, version, "stale-merge")
        assert result.isError is True
        sc = result.structuredContent
        assert sc["reason"] == "version_mismatch"
        assert sc["recover"] == "read_then_merge"
        assert sc["retryable"] is False
        assert sc["current_version"] > version  # the agent learns where to re-CAS
        assert target.read_bytes() == b"v2-from-b"  # A's stale CAS did NOT land
        assert conflicts[_PATH] == 1  # cooperating-agent counter ticked
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_expected_zero_loses_cleanly(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    target = _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        version = _do_read(vol, config, _PATH).structuredContent["version"]
        assert version > 0
        result = _do_write_cas(vol, config, {}, _PATH, 0, "overwrite")
        assert result.isError is True
        assert result.structuredContent["reason"] == "version_mismatch"
        assert target.read_bytes() == b"v1"  # no silent overwrite
    finally:
        stop_coordinator(tmp_path)


# --- cooperating-agent counter exhaustion ------------------------------------


def test_write_cas_counter_exhausts_for_cooperating_agent(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol_a = _vol(tmp_path, fast_cfg)
    vol_b = _vol(tmp_path, fast_cfg)
    try:
        stale = _do_read(vol_a, config, _PATH).structuredContent["version"]
        _do_read(vol_b, config, _PATH)
        _do_write(vol_b, config, _PATH, "advanced")  # current now != stale forever

        conflicts: dict[str, int] = {}
        # MAX_CAS_CONFLICTS conflicts return version_mismatch ...
        for _ in range(MAX_CAS_CONFLICTS):
            r = _do_write_cas(vol_a, config, conflicts, _PATH, stale, "x")
            assert r.structuredContent["reason"] == "version_mismatch"
        # ... the next one trips the cooperating-agent bound.
        exhausted = _do_write_cas(vol_a, config, conflicts, _PATH, stale, "x")
        assert exhausted.structuredContent["reason"] == "cas_exhausted"
        assert exhausted.structuredContent["retryable"] is False
    finally:
        stop_coordinator(tmp_path)


# --- no silent loss (the core property) --------------------------------------


def test_write_cas_two_writers_no_silent_loss(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """Two writers read the same version and both CAS at it: exactly one wins, the
    other is a typed version_mismatch, and the file holds the winner's content —
    never a silent lost update."""
    target = _seed(tmp_path, b"base")
    config = _config(tmp_path)
    vol_a = _vol(tmp_path, fast_cfg)
    vol_b = _vol(tmp_path, fast_cfg)
    try:
        va = _do_read(vol_a, config, _PATH).structuredContent["version"]
        vb = _do_read(vol_b, config, _PATH).structuredContent["version"]
        assert va == vb

        res_a = _do_write_cas(vol_a, config, {}, _PATH, va, "a-merged")
        res_b = _do_write_cas(vol_b, config, {}, _PATH, vb, "b-merged")

        wins = [r for r in (res_a, res_b) if not r.isError]
        conflicts = [r for r in (res_a, res_b) if r.isError]
        assert len(wins) == 1
        assert len(conflicts) == 1
        assert conflicts[0].structuredContent["reason"] == "version_mismatch"

        winner = b"a-merged" if not res_a.isError else b"b-merged"
        assert target.read_bytes() == winner
    finally:
        stop_coordinator(tmp_path)


# --- adapter primitive -------------------------------------------------------


def test_adapter_write_cas_at_raises_typed_conflict(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    _seed(tmp_path, b"v1")
    vol_a = _vol(tmp_path, fast_cfg)
    vol_b = _vol(tmp_path, fast_cfg)
    try:
        _bytes, va = vol_a.read_with_version(_PATH)
        vol_b.read_with_version(_PATH)
        vol_b.write(_PATH, b"v2")  # advance current
        with pytest.raises(CasVersionConflict) as exc:
            vol_a.write_cas_at(_PATH, va, b"stale")
        assert exc.value.expected_version == va
        assert exc.value.current_version > va
    finally:
        stop_coordinator(tmp_path)


# --- honesty surface ---------------------------------------------------------


def test_write_cas_description_states_cooperating_caveat() -> None:
    assert "COOPERATING" in _WRITE_CAS_DESC
    assert "livelock-proof" in _WRITE_CAS_DESC
    assert "version_mismatch" in _WRITE_CAS_DESC
    assert "auto-merge" in _WRITE_CAS_DESC


# --- fail-closed + counter semantics -----------------------------------------


def test_write_cas_missing_file_is_file_not_found(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A CAS on a non-existent file is a non-deny client error (file_not_found),
    not an escaped FileNotFoundError."""
    (tmp_path / "data").mkdir()
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        result = _do_write_cas(vol, config, {}, "data/nope.txt", 0, "x")
        assert result.isError is True
        assert result.structuredContent["reason"] == "file_not_found"
    finally:
        stop_coordinator(tmp_path)


def test_write_cas_counter_resets_after_a_win(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """A win mid-streak resets the per-path conflict counter — a subsequent
    conflict starts fresh at 1, not at the prior streak."""
    _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        conflicts: dict[str, int] = {}
        # expected_version=0 always conflicts against a v>0 file.
        for _ in range(MAX_CAS_CONFLICTS - 1):
            assert _do_write_cas(vol, config, conflicts, _PATH, 0, "x").structuredContent["reason"] == "version_mismatch"
        assert conflicts[_PATH] == MAX_CAS_CONFLICTS - 1

        # A win resets the streak ...
        version = _do_read(vol, config, _PATH).structuredContent["version"]
        assert _do_write_cas(vol, config, conflicts, _PATH, version, "won").isError is False
        assert _PATH not in conflicts

        # ... so the next conflict starts at 1, not near the exhaustion bound.
        assert _do_write_cas(vol, config, conflicts, _PATH, 0, "x").structuredContent["reason"] == "version_mismatch"
        assert conflicts[_PATH] == 1
    finally:
        stop_coordinator(tmp_path)


# --- SB-23: foreign-edit-at-write via swg_write -------------------------------


def test_swg_write_denies_foreign_edit(tmp_path: Path, fast_cfg: LifecycleConfig) -> None:
    """SB-23 via swg_write: a plain write that would clobber a foreign / out-of-band
    edit (no peer commit — the disk changed OUTSIDE the coordinator) is the
    recoverable ``stale_view`` deny, not a silent overwrite."""
    target = _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        _do_read(vol, config, _PATH)               # seeds the SB-23 baseline
        target.write_bytes(b"foreign-v2")          # out-of-band edit (not via the volume)
        denied = _do_write(vol, config, _PATH, "clobber")
        assert denied.isError is True
        sc = denied.structuredContent
        assert sc["reason"] == "stale_view"
        assert sc["recover"] == "reacquire"
        assert target.read_bytes() == b"foreign-v2"  # foreign edit NOT clobbered
    finally:
        stop_coordinator(tmp_path)


def test_swg_write_still_denies_a_foreign_edit_after_a_refused_swg_read(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A refused swg_read returns no content, so it must not count as having
    seen the foreign edit: the agent's next swg_write, built from its older
    read, is still denied rather than landing over the edit."""
    target = _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        _do_read(vol, config, _PATH)
        target.write_bytes(b"foreign-v2")
        refused = _do_read(vol, config, _PATH)
        assert refused.isError is True
        assert refused.structuredContent["reason"] == "stale_view"
        denied = _do_write(vol, config, _PATH, "v1-plus-agent-edit")
        assert denied.isError is True
        assert denied.structuredContent["reason"] == "stale_view"
        assert target.read_bytes() == b"foreign-v2"
    finally:
        stop_coordinator(tmp_path)


def test_swg_read_refused_after_out_of_band_edit_recovers_by_reacquire_then_write(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """After an out-of-band edit every swg_read is refused, and swg_reacquire
    alone does not change that: the coordinator has never recorded the bytes on
    disk. Writing the reacquired content records them, and reads answer again.
    This is the recovery the swg_read description names for a refusal that
    outlasts its retries (the retries are skipped here: the cause is known)."""
    target = _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol = _vol(tmp_path, fast_cfg)
    try:
        _do_read(vol, config, _PATH)
        target.write_bytes(b"foreign")
        assert _do_read(vol, config, _PATH).structuredContent["reason"] == "stale_view"
        reacquired = _do_reacquire(vol, config, _PATH)
        assert reacquired.structuredContent["content"] == "foreign"
        assert _do_read(vol, config, _PATH).structuredContent["reason"] == "stale_view"
        merged = reacquired.structuredContent["content"] + "+agent"
        assert _do_write(vol, config, _PATH, merged).isError is False
        read = _do_read(vol, config, _PATH)
        assert read.isError is False
        assert read.structuredContent["content"] == "foreign+agent"
        assert isinstance(read.structuredContent["version"], int)
    finally:
        stop_coordinator(tmp_path)


def test_swg_write_after_a_lost_swg_write_cas_is_still_denied(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A lost CAS discards the bytes of its own comparand read, so that read must
    not count as the agent having seen the peer's commit. When it did, an
    agent falling back from the conflict to swg_write with its older content
    overwrote the peer's committed update."""
    target = _seed(tmp_path, b"v1")
    config = _config(tmp_path)
    vol_a = _vol(tmp_path, fast_cfg)
    vol_b = _vol(tmp_path, fast_cfg)
    try:
        version = _do_read(vol_a, config, _PATH).structuredContent["version"]
        _do_read(vol_b, config, _PATH)
        assert _do_write(vol_b, config, _PATH, "v2-from-b").isError is False

        lost = _do_write_cas(vol_a, config, {}, _PATH, version, "v1+agent")
        assert lost.structuredContent["reason"] == "version_mismatch"
        fallback = _do_write(vol_a, config, _PATH, "v1+agent")
        assert fallback.isError is True
        assert fallback.structuredContent["reason"] == "stale_view"
        assert target.read_bytes() == b"v2-from-b"
    finally:
        stop_coordinator(tmp_path)


def test_read_description_names_the_write_recovery_for_a_lasting_refusal() -> None:
    # Whole word: "swg_write_cas" is already in the description and says nothing
    # about recovering from a refused read.
    assert re.search(r"\bswg_write\b", _READ_DESC)
    # ...and only after retrying, since a peer's commit still reaching disk
    # clears on its own and a write made inside that window is overwritten.
    assert "few seconds" in _READ_DESC


# --- the refusal reason reaches the caller -----------------------------------
#
# The coordinator distinguishes four CAS refusals; each needs different
# recovery. They all used to arrive as ``reason == "version_mismatch"``,
# because the class pinned it as a CLASS attribute and no raise site consulted
# the wire body. The worst case is ``other_holder``: the version has NOT moved,
# so the exception's own advice ("re-read at current and re-merge") produces a
# byte-identical CAS that fails identically until the holder releases.


def test_cas_conflict_defaults_to_version_mismatch() -> None:
    """Back-compat: consumers that never pass a reason keep the old value."""
    exc = CasVersionConflict("data/shared.txt", 3, 4)
    assert exc.reason == "version_mismatch"
    assert "version_mismatch" in str(exc)


def test_cas_conflict_carries_the_wire_reason() -> None:
    """A supplied reason lands on the instance AND in the message — the text is
    what ends up in logs and in anything that parses it."""
    exc = CasVersionConflict("data/shared.txt", 1, 1, reason="other_holder")
    assert exc.reason == "other_holder"
    assert str(exc).startswith("other_holder artifact=data/shared.txt")
    assert "version_mismatch" not in str(exc)
    assert CasVersionConflict.reason == "version_mismatch", (
        "the class default must survive so an `except` clause keyed on it works"
    )


def test_adapter_reports_other_holder_not_version_mismatch(
    tmp_path: Path, fast_cfg: LifecycleConfig
) -> None:
    """A pessimistic peer holding EXCLUSIVE refuses the CAS with the version
    unchanged. Relabelling that as a version mismatch tells the caller to merge
    and re-CAS, which cannot make progress until the holder lets go."""
    import uuid

    from ccs.cli._coherence_client import post, resolve_endpoint

    _seed(tmp_path, b"v1")
    vol = _vol(tmp_path, fast_cfg)
    try:
        _bytes, version = vol.read_with_version(_PATH)
        peer = post(
            resolve_endpoint(tmp_path),
            "/hooks/pre-edit",
            {"session_id": str(uuid.uuid4()), "path": _PATH},
        )
        assert peer["ok"] is True

        with pytest.raises(CasVersionConflict) as exc:
            vol.write_cas_at(_PATH, version, b"v2")

        assert exc.value.reason == "other_holder"
        assert exc.value.expected_version == exc.value.current_version == version
    finally:
        stop_coordinator(tmp_path)


# --- swg_read inside a peer's commit→disk window ------------------------------


def test_swg_read_in_a_peer_commit_window_returns_no_version(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read that lands after a peer's CAS is confirmed but before its bytes
    reach disk sees the old content under the new version. swg_read must refuse
    it as a recoverable stale_view and carry no version an agent could CAS at."""
    _seed(tmp_path, b"0")
    config = _config(tmp_path)
    peer = LaggingPeer(_vol(tmp_path, fast_cfg), monkeypatch, _PATH)
    agent = _vol(tmp_path, fast_cfg)
    try:
        peer.commit(b"1")
        try:
            read = _do_read(agent, config, _PATH)
        finally:
            peer.finish()
        assert read.isError is True
        sc = read.structuredContent
        assert sc["reason"] == "stale_view"
        assert sc["recover"] == "reacquire"
        assert sc["retryable"] is True
        assert "version" not in sc
        assert "content" not in sc
    finally:
        stop_coordinator(tmp_path)


def test_swg_read_then_write_cas_through_a_peer_commit_window_loses_no_update(
    tmp_path: Path, fast_cfg: LifecycleConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """swg_read → merge → swg_write_cas, run the way a cooperating agent runs it
    (a stale_view read is recovered by swg_reacquire, then read again). When
    swg_read answered inside the window, the agent merged "1" from the old "0"
    at the peer's version and the CAS won, so the file ended at "1"."""
    target = _seed(tmp_path, b"0")
    config = _config(tmp_path)
    peer = LaggingPeer(_vol(tmp_path, fast_cfg), monkeypatch, _PATH)
    agent = _vol(tmp_path, fast_cfg)
    try:
        peer.commit(b"1")
        read = _do_read(agent, config, _PATH)
        peer.finish()
        if read.isError:
            assert read.structuredContent["recover"] == "reacquire"
            assert _do_reacquire(agent, config, _PATH).isError is False
            read = _do_read(agent, config, _PATH)
        assert read.isError is False
        sc = read.structuredContent
        merged = str(int(sc["content"]) + 1)
        result = _do_write_cas(agent, config, {}, _PATH, sc["version"], merged)
        assert result.isError is False
        assert target.read_bytes() == b"2"
    finally:
        stop_coordinator(tmp_path)
