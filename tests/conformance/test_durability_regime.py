# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""U6 — durability-regime declaration + the kill-the-primary case (R-5, KTD-5).

The honest local claim is PROCESS-CRASH durability: SIGKILL destroys
user-space state only, so a green kill case proves an acknowledged commit's
bytes had at least reached the OS — and can NEVER discriminate fsync-grade
regimes (`synchronous` NORMAL vs FULL differ only under OS crash/power loss).
That grade, and the managed-cluster run, are DECLARED exemptions (R-8), never
silent skips. The negative control acks while its bytes are still buffered in
process memory — the loss SIGKILL genuinely can catch — proving the case has
teeth.
"""

from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

from ccs.core.substrate import CapabilityDescriptor, DurabilityRegime, Tier, sha256_hex
from ccs.testing.process_harness import (
    ContenderSpec,
    HarnessFailure,
    ProcessRaceHarness,
    run_kill_after_ack,
)
from ccs.testing.substrate_conformance import (
    AcknowledgedWriteLost,
    InMemoryBinding,
    InMemoryStore,
    LwwSubstrate,
    UndeclaredDurabilityRegime,
    assert_durability_regime_is_declared,
    assert_process_crash_durability,
    assert_sqlite_acknowledged_commit_survives_kill,
    declare_deferred_durability_exemptions,
)

pytestmark = [
    pytest.mark.skipif(
        not hasattr(signal, "SIGKILL"),
        reason=f"kill primitive unavailable on {sys.platform}; the case records a "
        "declared R-8 exemption on such platforms instead of running",
    ),
    # Kills and restarts real processes: run with -n 0 (tests/conftest.py).
    pytest.mark.serial,
]


# ---------------------------------------------------------------------------
# The real case: sqlite registry, SIGKILL between ack and next operation
# ---------------------------------------------------------------------------


def test_sqlite_acknowledged_commit_survives_sigkill(tmp_path: Path) -> None:
    """AE2's mechanism on the existing store: the child process commits (WIN),
    acknowledges, parks, and is SIGKILLed; the reopened db serves the
    acknowledged version with the acknowledged content hash."""
    assert_sqlite_acknowledged_commit_survives_kill(tmp_path / "state.db")


# ---------------------------------------------------------------------------
# Negative control: acks while the bytes are still in PROCESS memory
# ---------------------------------------------------------------------------

_PAYLOAD = b"acknowledged-but-buffered"


def _buffered_ack_child(ctx, path_str: str) -> None:
    """Acks a 'durable' write whose bytes never left the user-space buffer.

    ``ctx.ack`` parks and never returns, so the handle's frame stays alive,
    unflushed, until SIGKILL destroys it — a clean return (which would flush
    at interpreter exit and void the control) is impossible by construction."""
    handle = open(path_str, "wb")  # noqa: SIM115 — held open on purpose
    handle.write(_PAYLOAD)
    ctx.ack(
        {
            "path": path_str,
            "fingerprint": sha256_hex(_PAYLOAD),
        }
    )


def _verify_buffered_payload_on_disk(ack: dict) -> bool:
    target = Path(ack["path"])
    if not target.exists():
        return False
    return sha256_hex(target.read_bytes()) == ack["fingerprint"]


def test_buffered_in_process_ack_fails_the_case(tmp_path: Path) -> None:
    """Teeth: a binding that acks before handing bytes to the OS loses them
    under SIGKILL, and the case rejects it with the typed loss."""
    target = tmp_path / "buffered.bin"
    with pytest.raises(AcknowledgedWriteLost):
        assert_process_crash_durability(
            ContenderSpec(fn=_buffered_ack_child, args=(str(target),)),
            _verify_buffered_payload_on_disk,
        )


def _returns_without_acking(ctx, path_str: str) -> None:
    handle = open(path_str, "wb")  # noqa: SIM115
    handle.write(_PAYLOAD)
    # BUG under test: no ack — nothing acknowledged exists to kill against
    # (and ack() parking by construction means a return past an ack cannot
    # happen at all, so this is the only reachable violation).


def test_contender_that_returns_without_acking_is_a_harness_failure(
    tmp_path: Path,
) -> None:
    """The case needs an acknowledged write to kill against; a contender that
    completes without acking is refused loudly, never treated as a pass."""
    with pytest.raises(HarnessFailure, match="ack"):
        run_kill_after_ack(
            ContenderSpec(fn=_returns_without_acking, args=(str(tmp_path / "x.bin"),))
        )


# ---------------------------------------------------------------------------
# Descriptor axis: declared regime + typed refusal
# ---------------------------------------------------------------------------


def test_undeclared_regime_is_refused_with_typed_reason() -> None:
    descriptor = CapabilityDescriptor(
        tier=Tier.NATIVE_CAS,
        version_source="etag",
        least_privilege="test",
        consistency_note="test",
    )
    with pytest.raises(UndeclaredDurabilityRegime):
        assert_durability_regime_is_declared(descriptor)


def test_declared_regime_requires_stated_facts() -> None:
    """A regime name with no configuration facts is unverifiable — refused."""
    descriptor = CapabilityDescriptor(
        tier=Tier.NATIVE_CAS,
        version_source="etag",
        durability_regime=DurabilityRegime.PROCESS_CRASH,
    )
    with pytest.raises(UndeclaredDurabilityRegime, match="facts"):
        assert_durability_regime_is_declared(descriptor)


def test_facts_without_a_regime_are_rejected_at_construction() -> None:
    """Facts qualifying no regime are an over-claim vector: fail-closed."""
    with pytest.raises(ValueError, match="durability_facts"):
        CapabilityDescriptor(
            tier=Tier.NATIVE_CAS,
            version_source="etag",
            durability_facts="journal_mode=WAL",
        )


def test_kit_fake_descriptors_declare_in_process_regime() -> None:
    """The kit's own bindings speak the axis honestly: process memory only."""
    binding = InMemoryBinding()
    assert_durability_regime_is_declared(binding.descriptor)
    assert binding.descriptor.durability_regime is DurabilityRegime.IN_PROCESS
    lww = LwwSubstrate(InMemoryStore())
    assert_durability_regime_is_declared(lww.descriptor)
    assert lww.descriptor.durability_regime is DurabilityRegime.IN_PROCESS


# ---------------------------------------------------------------------------
# Deferred grades are DECLARED exemptions in the kit report (R-8 / KTD-5)
# ---------------------------------------------------------------------------


def test_deferred_grades_are_declared_exemptions_in_report() -> None:
    harness = ProcessRaceHarness()
    fsync, managed = declare_deferred_durability_exemptions(harness)
    report = harness.report()
    assert any(fsync.reason in line for line in report)
    assert any(managed.reason in line for line in report)
    assert "VFS" in fsync.reason  # the named future closer, not a vague deferral
    assert "release" in managed.reason  # KTD-5: a per-binding-release run
