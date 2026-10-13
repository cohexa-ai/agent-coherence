# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The BYO-substrate conformance corpus: tier honesty + the Workspace-Versioning
family.

PACKAGED (WV plan Unit 9 / R11): this corpus ships inside the package
(``ccs.testing`` — ``where = ["src"]``) so a foreign implementation can import
and run it without vendoring this repository's test tree. The in-repo runner
(``tests/conformance/``) imports FROM here through a thin shim; this module
never imports from the test tree. pytest is OPTIONAL: the corpus imports and
runs without it (raise-expectations use an internal shim); only scenario SKIP
reporting needs it — ``pip install 'agent-coherence[conformance]'``.

A descriptor-parametrized kit that certifies a substrate binding's declared tier
is HONEST and, crucially, that the **adapter's** invalidation value — not the
bare substrate compare-and-set — is what passes. The scenarios are plain
functions taking a :class:`ConformanceBinding` (a seam that mints binding views
over ONE shared artifact) and a session factory (an agent identity on ONE
coordinator). The default arm plugs an in-memory fake substrate; the
``real_substrate``-gated arm plugs the real ``CoherentRow`` / ``CoherentObject``
bindings against a real Postgres / S3 (see ``tests/conformance/test_tier_honesty.py``).

The two load-bearing native-CAS scenarios are deliberately distinct:

- :func:`assert_racing_writers_one_winner` — a non-coordinated (fixed-stale)
  writer moves the substrate; the loser gets the uniform typed retryable conflict
  from the **substrate CAS**. This is the guarantee a bare CAS already gives —
  ``(i)`` alone is the bare CAS.
- :func:`assert_invalidation_before_act` — a peer commit denies the other's
  binding-mediated act with a typed ``StaleView`` **before** its write reaches
  the substrate CAS. The deny being ``StaleView`` (the coordinator's pull
  invalidation), NOT the substrate's ``CasVersionConflict``, is the proof that the
  coordinator layer is load-bearing — ``(ii)`` is what a bare CAS never surfaces.

The kit REJECTS an overclaiming binding: the split-comparand scenario
(:func:`assert_rejects_split_comparand`) is the PR-#107 negative control that a
version-CAS / ``NoLostUpdate`` check does not catch, and it MUST fail a
deliberately-split binding.

Deferred with the fence work (do NOT write speculatively): (iii) fence-rejection
and (iv) clock-domain sweep-liveness. Those assertions join the kit only when the
Phase-0 spike un-defers the read-generation fence — v1 bindings ride
admit-on-absent + the version-CAS and claim no fence.

**The Workspace-Versioning family (WV plan Unit 9 / R11)** lives at the bottom
of this module: :class:`WorkspaceConformanceBinding` (the shipped
:class:`ConformanceBinding` seam extended with the WV capability members),
outcome-level scenario functions split MUST-MATCH vs DECLARED exactly like the
tier kit, the reference :class:`LocalWorkspaceBinding` arm over the
deterministic S3-semantics fake, and the :class:`TornCutBlindWorkspaceBinding`
teeth stub. The family's fixture-disclosure rule is binding: scenario prose and
assertions describe failure scenarios and required FINAL OBSERVABLE OUTCOMES
only — a foreign implementation passes by satisfying outcomes, never by
mimicking any particular internal mechanism.


**In-process serialization does not satisfy any clause of this contract.**
The hand-sequenced scenarios below are TYPED-OUTCOME contract tests — they pin
what a binding returns at each interleaving, and a single process can sequence
any interleaving it likes. Only the cross-process races (the
``assert_cross_process_*`` family, riding :mod:`ccs.testing.process_harness`)
prove a binding's conditional write is atomic rather than read-then-write; a
binding that cannot be raced cross-process carries a DECLARED exemption with a
stated reason (R-8), never a silent skip.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from multiprocessing.managers import BaseManager
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Iterator,
    NoReturn,
    Protocol,
    Sequence,
    runtime_checkable,
)
from uuid import UUID, uuid4

# pytest backs SKIP reporting ONLY — the corpus itself must import and run
# without the test framework installed (the packaged-importable claim above).
# Guarded like the optional substrate drivers (see coherent_object's boto3
# deferral): absence degrades to None here and fails actionably at USE time.
try:
    import pytest
except ImportError:  # pragma: no cover — exercised by the clean-import test
    pytest = None  # type: ignore[assignment]

import ccs.adapters.substrate as substrate_module
from ccs.adapters.coherent_object import CoherentObject
from ccs.adapters.substrate import (
    CasConflict,
    CasUnknown,
    CasWriteResult,
    CasWritten,
    CoordinatedSubstrate,
    ReconcileDecision,
    ReconcileVerdict,
    ReconcilingSubstrate,
    SubstrateCoordinatorSession,
)
from ccs.adapters.workspace import WorkspaceVersioner
from ccs.coordinator.service import CoordinatorService
from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
from ccs.core.exceptions import (
    RESTORE_OUTCOME_CONFLICT,
    RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED,
    RESTORE_OUTCOME_TARGET_LOST,
    RESTORE_STATUS_CONCLUDED,
    RESTORE_SUCCESS_OUTCOMES,
    CasVersionConflict,
    StaleView,
)
from ccs.core.substrate import (
    CapabilityDescriptor,
    DurabilityRegime,
    RestoreTier,
    Tier,
    retention_is_empty_for,
    sha256_hex,
)
from ccs.testing.process_harness import (
    ContenderSpec,
    ProcessRaceHarness,
    run_kill_after_ack,
)
from ccs.testing.s3_local import LocalS3Client

if TYPE_CHECKING:
    # Annotation-only: importing the corpus must never pull the server stack
    # (SubstrateCoordinatorSession defers the same import for the same reason).
    from ccs.adapters.claude_code.lifecycle import LifecycleConfig

__all__ = [
    "WORKSPACE_BINDING_MEMBERS",
    "ConformanceBinding",
    "ConformanceSubstrate",
    "CoordinatorHarness",
    "InMemoryBinding",
    "InMemoryStore",
    "LocalWorkspaceBinding",
    "LwwSubstrate",
    "RestoredMemberFacts",
    "SessionFactory",
    "SplitComparandSubstrate",
    "TornCutBlindWorkspaceBinding",
    "WorkspaceCheckpointFacts",
    "WorkspaceConformanceBinding",
    "WorkspaceMemberFacts",
    "WorkspaceRestoreFacts",
    "assert_abort_after_partial_visibility",
    "AcknowledgedWriteLost",
    "FSYNC_GRADE_EXEMPTION_REASON",
    "MANAGED_KILL_RUN_EXEMPTION_REASON",
    "UndeclaredDurabilityRegime",
    "assert_durability_regime_is_declared",
    "assert_process_crash_durability",
    "assert_sqlite_acknowledged_commit_survives_kill",
    "declare_deferred_durability_exemptions",
    "assert_coordinator_retention_empty",
    "assert_cross_process_one_winner_native_cas",
    "assert_cross_process_lost_update_is_detected",
    "assert_cross_process_rejects_read_then_write",
    "CrossProcessLostUpdate",
    "assert_detect_only_silent_lost_update",
    "assert_forward_only_honest",
    "assert_in_process_binding_exemption_is_declared",
    "assert_invalidation_before_act",
    "assert_native_cas_descriptor",
    "assert_never_ship_a_store_wire",
    "assert_racing_writers_one_winner",
    "assert_read_pair_is_atomic",
    "assert_read_pair_split_binding_is_rejected",
    "assert_rejects_split_comparand",
    "assert_split_view_is_rejected",
    "assert_torn_cut_blind_binding_is_rejected",
    "assert_workspace_binding_declares_all",
    "assert_wv_absent_is_a_fact",
    "assert_wv_declared_pin_honest",
    "assert_wv_declared_versioning_honest",
    "assert_wv_monotonic_restore",
    "assert_wv_restart_survival_per_declaration",
    "assert_wv_restore_one_winner",
    "assert_wv_restore_terminates_under_contention",
    "assert_wv_torn_cut_detection",
    "run_native_cas_conformance",
    "run_workspace_conformance",
    "workspace_protocol_members",
]

# The wording a non-native tier must never carry: enforcement / CAS / rollback /
# duplicate-effect claims. Matched case-insensitively, so the guarantee text must
# express its disclaimers without these trigger words.
_FORBIDDEN_ENFORCEMENT_WORDS = ("enforce", "cas", "rollback", "dedup")

_PYTEST_INSTALL_HINT = (
    "reporting a conformance-scenario SKIP requires pytest — install it with: "
    "pip install 'agent-coherence[conformance]'"
)


@contextmanager
def _expect_raises(exc_type: type[BaseException], match: str | None = None) -> Iterator[None]:
    """A dependency-free ``pytest.raises``: the packaged corpus must RUN, not
    just import, without the test framework. Mirrors pytest's semantics — the
    block must raise ``exc_type``, and ``match`` (when given) is
    ``re.search``-ed against the exception text."""
    try:
        yield
    except exc_type as caught:
        if match is not None and re.search(match, str(caught)) is None:
            raise AssertionError(
                f"{type(caught).__name__} was raised but its message did not "
                f"match {match!r}: {caught}"
            ) from caught
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised; nothing was")


def _skip_scenario(reason: str) -> NoReturn:
    """Report a scenario skip through pytest when it is installed. Without
    pytest there is no channel that distinguishes 'skipped' from 'passed', so
    fail closed AT USE with the install hint (never a silent green)."""
    if pytest is None:
        raise ModuleNotFoundError(f"{_PYTEST_INSTALL_HINT} (skip reason: {reason})")
    pytest.skip(reason)
    raise AssertionError("unreachable: pytest.skip always raises")  # pragma: no cover


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fresh_ref(prefix: str = "conformance") -> str:
    """A unique artifact ref, so sequential scenarios on ONE coordinator never
    collide on shared per-artifact state."""
    return f"{prefix}/{uuid4().hex}.bin"


# ===========================================================================
# The in-memory fake substrate (default arm) — defined HERE, never imported
# from the tests package, so the kit is self-contained.
# ===========================================================================


def _fake_descriptor(arm: str) -> CapabilityDescriptor:
    return CapabilityDescriptor(
        tier=Tier.NATIVE_CAS,
        version_source="fake row-version" if arm == "row" else "fake object ETag",
        least_privilege="in-memory conformance fake",
        consistency_note="fake single-primary",
        durability_regime=DurabilityRegime.IN_PROCESS,
        durability_facts="process memory only (dict store); dies with the process",
    )


class InMemoryStore:
    """The shared substrate state (one row / one object per ref), shared by every
    binding view — distinct agents, one underlying artifact."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[bytes, str]] = {}
        self._counter = 0
        #: Substrate-read count — a split (bytes, token) read shows up as >1 per
        #: view.read(), which the read-pair-atomicity control asserts against.
        self.read_count = 0

    def set(self, ref: str, data: bytes) -> str:
        self._counter += 1
        token = f"tok-{self._counter}"
        self._data[ref] = (bytes(data), token)
        return token

    def get(self, ref: str) -> tuple[bytes, str] | None:
        self.read_count += 1
        return self._data.get(ref)


class ConformanceSubstrate:
    """A per-agent binding view over the shared store, implementing the
    reconciling-substrate surface with realistic token-identity logic.

    Supports two test seams: :meth:`script_unknown` (force an UNKNOWN write, so
    the abort-after-partial-visibility reconciliation is reachable) and
    :meth:`schedule_peer_write` (a peer write that lands right after this view's
    atomic read — used by the split-comparand control, which OVERRIDES ``read``).
    """

    #: The body never reaches the coordinator (never-ship-a-store).
    SENDS_CONTENT_TO_COORDINATOR: bool = False

    def __init__(self, store: InMemoryStore, arm: str = "object") -> None:
        self._store = store
        self._arm = arm
        self._descriptor = _fake_descriptor(arm)
        self.cas_calls: list[tuple[str, str, bytes]] = []
        self._script: tuple[str, bool] | None = None
        self._scheduled_peer: bytes | None = None

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def coordinator_commit_content(self) -> None:
        return None

    def script_unknown(self, *, landed: bool) -> None:
        """Next ``cas_write`` returns ``CasUnknown``; ``landed`` applies the write."""
        self._script = ("unknown", landed)

    def schedule_peer_write(self, data: bytes) -> None:
        """A peer write that lands immediately AFTER this view's atomic read — so
        this (conforming) view's token is the PRE-peer token and a later CAS
        under it conflicts. The split control overrides ``read`` to capture the
        POST-peer token instead."""
        self._scheduled_peer = bytes(data)

    def read(self, artifact_ref: str) -> tuple[bytes, str]:
        entry = self._store.get(artifact_ref)
        if entry is None:
            raise KeyError(artifact_ref)
        # Atomic pair captured BEFORE any scheduled peer write lands.
        pair = entry
        if self._scheduled_peer is not None:
            self._store.set(artifact_ref, self._scheduled_peer)
            self._scheduled_peer = None
        return pair

    def cas_write(self, artifact_ref: str, *, expected_token: str, new_bytes: bytes) -> CasWriteResult:
        self.cas_calls.append((artifact_ref, expected_token, bytes(new_bytes)))
        script, self._script = self._script, None
        if script is not None:
            _kind, landed = script
            if landed:
                self._store.set(artifact_ref, bytes(new_bytes))
            return CasUnknown()
        entry = self._store.get(artifact_ref)
        if entry is None or entry[1] != expected_token:
            return CasConflict()
        return CasWritten(token=self._store.set(artifact_ref, bytes(new_bytes)))

    def reconcile_after_unknown(
        self, artifact_ref: str, *, expected_token: str, intended_hash: str
    ) -> ReconcileDecision:
        entry = self._store.get(artifact_ref)
        if entry is None:
            return ReconcileDecision(ReconcileVerdict.HOLD, None, None)
        observed_bytes, observed_token = entry
        if observed_token == expected_token:
            return ReconcileDecision(ReconcileVerdict.RE_DRIVE, observed_bytes, observed_token)
        if _sha256(observed_bytes) == intended_hash:
            return ReconcileDecision(ReconcileVerdict.CONVERGE, observed_bytes, observed_token)
        verdict = ReconcileVerdict.CONFLICT if self._arm == "object" else ReconcileVerdict.RE_DERIVE
        return ReconcileDecision(verdict, observed_bytes, observed_token)


class SplitComparandSubstrate(ConformanceSubstrate):
    """The PR-#107 lost-update bug: ``(bytes, token)`` from TWO reads.

    ``read`` returns the bytes from read A but the token from a LATER read B, with
    a peer write landing between the two sub-reads. The returned token vouches for
    bytes it never described, so a subsequent CAS under it PASSES and silently
    loses the peer's write — the exact bug a version-CAS / ``NoLostUpdate`` check
    does not catch. The kit MUST reject this binding.
    """

    def read(self, artifact_ref: str) -> tuple[bytes, str]:
        entry = self._store.get(artifact_ref)
        if entry is None:
            raise KeyError(artifact_ref)
        bytes_a = entry[0]  # read A: bytes
        if self._scheduled_peer is not None:  # a peer writes BETWEEN the two reads
            self._store.set(artifact_ref, self._scheduled_peer)
            self._scheduled_peer = None
        fresh = self._store.get(artifact_ref)
        token_b = fresh[1] if fresh is not None else entry[1]  # read B: token only
        return bytes_a, token_b  # SPLIT: stale bytes, fresh token


class LwwSubstrate:
    """A detect-only substrate model: NO atomic compare-and-set — every write is
    last-write-wins. Used by the forced detect-only arm to demonstrate a SILENT
    lost update (the tier catches a sequential stale-read→write but cannot prevent
    a concurrent race)."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._descriptor = CapabilityDescriptor(
            tier=Tier.DETECT_ONLY,
            version_source=None,
            least_privilege="in-memory detect-only fake",
            consistency_note="last-write-wins; no atomic CAS",
            durability_regime=DurabilityRegime.IN_PROCESS,
            durability_facts="process memory only (dict store); dies with the process",
        )

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._descriptor

    def read(self, artifact_ref: str) -> tuple[bytes, str]:
        entry = self._store.get(artifact_ref)
        if entry is None:
            raise KeyError(artifact_ref)
        return entry

    def cas_write(self, artifact_ref: str, *, expected_token: str, new_bytes: bytes) -> CasWriteResult:
        # No atomic CAS: expected_token is ignored, the write always lands.
        del expected_token
        return CasWritten(token=self._store.set(artifact_ref, bytes(new_bytes)))


# ===========================================================================
# The binding seam — the ONE abstraction every arm (fake + real) plugs into.
# ===========================================================================


@runtime_checkable
class ConformanceBinding(Protocol):
    """What a conformance run needs from a binding: a descriptor, a way to seed
    and non-coordinated-write the shared artifact, and a factory for fresh binding
    views over it. The in-memory fake and the real PG/S3 bindings both satisfy it.
    """

    @property
    def descriptor(self) -> CapabilityDescriptor: ...

    def seed(self, ref: str, data: bytes) -> None: ...

    def foreign_write(self, ref: str, data: bytes) -> None: ...

    def make_view(self) -> ReconcilingSubstrate: ...


class InMemoryBinding:
    """A :class:`ConformanceBinding` backed by the in-memory fake substrate.

    Declared ``in_process_only`` (KTD-4): the store is GIL-serialized process
    memory by construction and cannot be shared across OS processes — a
    spawn-context child would receive a pickled private copy and the race
    would be vacuous. The cross-process arm runs against the manager-served
    stand-in instead; this binding's exclusion is a DECLARED R-8 exemption,
    never a silent skip."""

    in_process_only = True
    in_process_only_reason = (
        "GIL-serialized process memory by construction; the cross-process race "
        "runs against the manager-served substrate instead (KTD-4)"
    )

    def __init__(self, arm: str = "object") -> None:
        self._store = InMemoryStore()
        self._arm = arm

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return _fake_descriptor(self._arm)

    def seed(self, ref: str, data: bytes) -> None:
        self._store.set(ref, data)

    def foreign_write(self, ref: str, data: bytes) -> None:
        # A non-coordinated (fixed-stale) writer moves the substrate WITHOUT
        # telling the coordinator — the racing-writer the substrate CAS arbitrates.
        self._store.set(ref, data)

    def make_view(self) -> ConformanceSubstrate:
        return ConformanceSubstrate(self._store, self._arm)

    def substrate_read_count(self) -> int:
        """Total substrate reads so far — lets the read-pair-atomicity control
        assert that one ``view.read()`` fetches ``(bytes, token)`` in ONE read."""
        return self._store.read_count


class _SplitReadBinding(InMemoryBinding):
    """A ConformanceBinding whose view fetches ``(bytes, token)`` in TWO substrate
    reads — the read-pair-atomicity control MUST reject it (teeth certification)."""

    def make_view(self) -> SplitComparandSubstrate:
        return SplitComparandSubstrate(self._store, self._arm)


# ===========================================================================
# Native-CAS scenarios — substrate-agnostic (fake AND real_substrate).
# ===========================================================================


def assert_native_cas_descriptor(descriptor: CapabilityDescriptor) -> None:
    """The declared tier is native-CAS with the honest timeout-asterisk wording."""
    assert descriptor.tier is Tier.NATIVE_CAS
    assert descriptor.version_source, "native-CAS must name its version source"
    text = descriptor.guarantee_text
    assert "enforces no-lost-update on the version-CAS axis only" in text
    assert "token-identity reconciliation" in text
    assert "single-host" in text


def assert_racing_writers_one_winner(
    binding: ConformanceBinding, make_session: "SessionFactory"
) -> None:
    """(i) TYPED-OUTCOME contract test (hand-sequenced, single-process): the
    SUBSTRATE CAS arbitrates — a non-coordinated writer moves the substrate;
    the loser gets the uniform typed retryable conflict. This is the guarantee
    a bare CAS already gives (it is NOT the coordinator's add). It pins the
    OUTCOME VOCABULARY only; atomicity under real contention is proved by
    ``assert_cross_process_one_winner_native_cas`` — in-process sequencing
    satisfies no atomicity clause."""
    ref = _fresh_ref()
    binding.seed(ref, b"v1")
    agent = CoordinatedSubstrate(binding.make_view(), make_session())
    _bytes, tok = agent.read(ref)

    # A concurrent (non-coordinated) writer moves the substrate — the coordinator
    # is NOT told, so the pre-read stays clean and the substrate CAS is the arbiter.
    binding.foreign_write(ref, b"v2-foreign")

    with _expect_raises(CasVersionConflict):
        agent.commit(ref, expected_token=tok, new_bytes=b"vA")
    # The foreign write is the one winner; the loser's bytes never landed.
    assert binding.make_view().read(ref)[0] == b"v2-foreign"


def assert_invalidation_before_act(
    binding: ConformanceBinding, make_session: "SessionFactory"
) -> None:
    """(ii) The COORDINATOR layer denies a peer-invalidated act with ``StaleView``
    BEFORE the substrate CAS. The deny being ``StaleView`` (not the substrate's
    ``CasVersionConflict``) is the proof the coordinator layer is load-bearing —
    the case a bare CAS never surfaces."""
    ref = _fresh_ref()
    binding.seed(ref, b"v1")
    view_a = binding.make_view()
    agent_a = CoordinatedSubstrate(view_a, make_session())
    agent_b = CoordinatedSubstrate(binding.make_view(), make_session())

    _ab, a_tok = agent_a.read(ref)
    _bb, b_tok = agent_b.read(ref)
    agent_b.commit(ref, expected_token=b_tok, new_bytes=b"v2")  # B wins; A invalidated

    with _expect_raises(StaleView):
        agent_a.commit(ref, expected_token=a_tok, new_bytes=b"vA")
    # Deny-before-act: the substrate write was never attempted (fake-observable).
    if hasattr(view_a, "cas_calls"):
        assert view_a.cas_calls == []


def assert_never_ship_a_store_wire(
    binding: ConformanceBinding, make_session: "SessionFactory"
) -> None:
    """(a) Binding behavior: the commit path threads ``content=None`` and the wire
    payload carries the fixed-width ``content_hash`` but NO content."""
    view = binding.make_view()
    assert getattr(view, "SENDS_CONTENT_TO_COORDINATOR", None) is False
    assert view.coordinator_commit_content() is None

    ref = _fresh_ref()
    binding.seed(ref, b"v1")
    captured: list[dict] = []
    real_post = substrate_module._coordinator_post

    def spy(endpoint, path, payload, **kwargs):  # noqa: ANN001, ANN003, ANN202
        if path == "/hooks/post-edit-cas":
            captured.append(dict(payload))
        return real_post(endpoint, path, payload, **kwargs)

    substrate_module._coordinator_post = spy
    try:
        agent = CoordinatedSubstrate(view, make_session())
        _bytes, tok = agent.read(ref)
        agent.commit(ref, expected_token=tok, new_bytes=b"v2")
    finally:
        substrate_module._coordinator_post = real_post

    assert captured, "a commit must POST /hooks/post-edit-cas"
    for payload in captured:
        assert "content" not in payload, "bytes are NEVER sent to the coordinator"
        assert isinstance(payload.get("content_hash"), str)


def assert_read_pair_is_atomic(binding: ConformanceBinding, make_session: "SessionFactory") -> None:
    """Read-pair atomicity: one ``view.read()`` fetches ``(bytes, token)`` in a
    SINGLE substrate read (the PR-#107 control, now certified for shipped bindings).

    A split read — bytes from read A, token from a LATER read B — is the PR-#107
    lost update: the token vouches for bytes it never described, so a CAS under it
    passes on a stale comparand and silently loses a concurrent peer. A binding
    that issues exactly one substrate read for the pair CANNOT split. The count is
    read from ``substrate_read_count`` when the binding instruments it (the fake,
    always; a real binding when its driver is wrapped); it skips otherwise rather
    than false-green."""
    del make_session  # a bare-substrate property; no coordinator needed
    read_count = getattr(binding, "substrate_read_count", None)
    if read_count is None:
        return  # binding does not instrument substrate reads — cannot certify here
    ref = _fresh_ref()
    binding.seed(ref, b"base")
    view = binding.make_view()
    before = read_count()
    view.read(ref)
    delta = read_count() - before
    assert delta == 1, (
        f"read-pair split: view.read() issued {delta} substrate reads, not 1 — a "
        "(bytes, token) pair read non-atomically can silently lose a peer (PR-#107)"
    )


def assert_read_pair_split_binding_is_rejected() -> None:
    """Confirm the read-pair-atomicity control has TEETH: a split-read binding
    (two substrate reads per pair) fails it (helper for the MUST-FAIL test)."""
    with _expect_raises(AssertionError):
        assert_read_pair_is_atomic(_SplitReadBinding("object"), None)  # type: ignore[arg-type]


def run_native_cas_conformance(
    binding: ConformanceBinding, make_session: "SessionFactory"
) -> None:
    """The substrate-agnostic native-CAS arm — run against the fake AND real
    substrates. (i) proves the bare CAS; (ii) proves the coordinator's
    invalidation-before-act; (a) pins never-ship-a-store on the wire; and the
    read-pair-atomicity control certifies the SHIPPED read is not split."""
    assert_native_cas_descriptor(binding.descriptor)
    assert_racing_writers_one_winner(binding, make_session)
    assert_invalidation_before_act(binding, make_session)
    assert_never_ship_a_store_wire(binding, make_session)
    assert_read_pair_is_atomic(binding, make_session)


# ===========================================================================
# Fake-driven scenarios — abort reconciliation + the negative controls.
# ===========================================================================


def assert_abort_after_partial_visibility(
    binding: ConformanceBinding, make_session: "SessionFactory"
) -> None:
    """A commit whose substrate write LANDED but whose ack was aborted mid-commit
    must reconcile by token-identity (CONVERGE), never assume nothing landed."""
    ref = _fresh_ref()
    binding.seed(ref, b"v1")
    view = binding.make_view()
    if not hasattr(view, "script_unknown"):
        _skip_scenario("abort-after-partial-visibility needs a scriptable substrate")

    agent = CoordinatedSubstrate(view, make_session())
    _bytes, tok = agent.read(ref)
    view.script_unknown(landed=True)  # the write lands; the ack aborts mid-commit

    result = agent.commit(ref, expected_token=tok, new_bytes=b"v2")
    assert result.converged is True, "an aborted-but-landed write must reconcile, not re-drive"
    assert result.summary == "converged"  # never "landed" — attribution disclaimed
    assert binding.make_view().read(ref)[0] == b"v2"  # the landed write is adopted, not lost


def assert_rejects_split_comparand(store_arm: str = "object") -> None:
    """The PR-#107 negative control (MANDATORY): a CONFORMING view passes the
    read-pair-consistency check; a SPLIT view silently loses the peer's write.

    A conforming view captures ``(bytes, token)`` atomically, so a peer write that
    lands after the read leaves its CAS-comparand STALE → the later CAS conflicts →
    the peer's write survives. A split view captures the token from a later read,
    so its CAS PASSES on a token that never described its bytes → the peer's write
    is silently lost. This function ASSERTS "the peer's write survives"; it PASSES
    for the conforming view and RAISES ``AssertionError`` for the split view (the
    MUST-FAIL helper :func:`assert_split_view_is_rejected` wraps the split call
    and asserts the raise).
    """
    _run_split_control(ConformanceSubstrate, store_arm)


def assert_split_view_is_rejected(store_arm: str = "object") -> None:
    """Confirm the split view fails the control (helper for the MUST-FAIL test)."""
    with _expect_raises(AssertionError):
        _run_split_control(SplitComparandSubstrate, store_arm)


def _run_split_control(view_cls: type[ConformanceSubstrate], arm: str) -> None:
    ref = _fresh_ref()
    store = InMemoryStore()
    store.set(ref, b"base")
    view = view_cls(store, arm)
    view.schedule_peer_write(b"peer-change")

    _bytes, token = view.read(ref)
    result = view.cas_write(ref, expected_token=token, new_bytes=b"derived-from-stale")

    final = store.get(ref)[0]
    assert final == b"peer-change" and isinstance(result, CasConflict), (
        "a split comparand silently lost the peer's write (PR-#107): the CAS "
        "passed on a token that vouched for bytes it never described"
    )


def assert_detect_only_silent_lost_update() -> None:
    """The forced detect-only arm: force the interleave and prove the lost update
    ACTUALLY occurred and was SILENT (no raise) — "not prevented" demonstrated,
    not merely unobserved. Also pins the descriptor's honest (non-enforcement)
    tier text."""
    ref = _fresh_ref()
    store = InMemoryStore()
    store.set(ref, b"v1")
    sub = LwwSubstrate(store)

    _bytes, tok = sub.read(ref)  # two writers both read v1
    # Forced interleave: a peer write lands, then the racing writer writes under
    # the SAME (now stale) token. A detect-only substrate has no atomic CAS, so
    # BOTH land — the peer's write is silently lost.
    peer_result = sub.cas_write(ref, expected_token=tok, new_bytes=b"peer")
    racer_result = sub.cas_write(ref, expected_token=tok, new_bytes=b"racer")

    assert isinstance(peer_result, CasWritten) and isinstance(racer_result, CasWritten)
    assert store.get(ref)[0] == b"racer", "the racer silently overwrote the peer (a race not prevented)"
    # The descriptor's skip-with-reason: detection wording, never enforcement.
    text = sub.descriptor.guarantee_text.lower()
    assert "catches a sequential stale-read" in text
    assert "cannot prevent a concurrent race" in text
    for word in _FORBIDDEN_ENFORCEMENT_WORDS:
        assert word not in text, f"detect-only text must not contain {word!r}"



# ===========================================================================
# Cross-process substrate-CAS races (guarantee-ladder plan U2; R-1/R-2).
#
# In-process serialization does not satisfy any clause of this contract: the
# original race assertions above are hand-sequenced interleavings and remain
# valuable as TYPED-OUTCOME contract tests, but only a race between separate
# OS processes proves a binding's conditional write is atomic rather than
# read-then-write. The packaged S3 stand-in is an in-process fake, so the
# harness owns the sharing vehicle: a substrate-owner process serves ONE
# locked stand-in instance and contenders reach it through a manager proxy.
# The manager Server dispatches each client CONNECTION on its own thread with
# no dispatch lock, and the packaged fake is single-threaded by design -- so
# call-atomicity is NOT free. The served stand-ins below run every public
# call under one per-call lock, making a NATIVE conditional put (check+write
# inside one call) atomic by construction rather than by GIL scheduling
# accident. The raced boundary stays BETWEEN proxied calls: a client-side
# check-then-write (two calls with a real gap) races and is caught. A real
# PG/S3 endpoint, when configured, replaces the proxy with the true network
# boundary (the `real_substrate` arm).
# ===========================================================================


class CrossProcessLostUpdate(AssertionError):
    """A cross-process race produced a lost update (more than one apparent
    winner, or a winner whose bytes did not survive). For a conforming binding
    this is a conformance FAILURE; for the read-then-write teeth control it is
    the expected, asserted detection."""


class _SubstrateManager(BaseManager):
    """The substrate-owner process: serves ONE locked stand-in instance to all
    contenders. The stand-in's per-call lock makes each proxied method call
    atomic inside the owner process (the manager Server would otherwise
    interleave them, one dispatch thread per client connection); the gap
    BETWEEN calls is the raced boundary."""


class _CallAtomicS3Client(LocalS3Client):
    """Manager-served stand-in whose every public call runs under one lock.

    The manager Server dispatches each client CONNECTION on its own thread,
    and the packaged fake is single-threaded by design — so without this
    lock the conditional put's check-then-append would be atomic only by
    GIL scheduling accident. The lock makes call-atomicity a property of
    the substrate-owner process by construction; the raced boundary stays
    BETWEEN proxied calls, exactly what the contenders race."""

    def __init__(self) -> None:
        super().__init__()
        self._call_lock = threading.Lock()

    def create_bucket(self, name: str, **kwargs: Any) -> None:
        with self._call_lock:
            return super().create_bucket(name, **kwargs)

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        with self._call_lock:
            return super().put_object(**kwargs)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        with self._call_lock:
            return super().get_object(**kwargs)


class _CallAtomicInMemoryStore(InMemoryStore):
    """Same construction for the read-then-write control's store."""

    def __init__(self) -> None:
        super().__init__()
        self._call_lock = threading.Lock()

    def set(self, ref: str, data: bytes) -> str:
        with self._call_lock:
            return super().set(ref, data)

    def get(self, ref: str) -> tuple[bytes, str] | None:
        with self._call_lock:
            return super().get(ref)


# The registered NAMES stay the packaged classes' names so every call site is
# unchanged; the SERVED classes are the locked stand-ins above.
_SubstrateManager.register("LocalS3Client", _CallAtomicS3Client)
_SubstrateManager.register("InMemoryStore", _CallAtomicInMemoryStore)

_XP_BUCKET = "xp-race"
_XP_KEY = "artifact"


def _xp_native_cas_contender(ctx, client_proxy, etag: str, payload: bytes) -> str:
    """Conforming contender: both phases barrier-aligned, then ONE proxied
    conditional put -- the check and the write execute atomically inside the
    substrate-owner process (native CAS, the S3 If-Match shape)."""
    ctx.barrier_wait()  # phase 1: everyone is running
    ctx.barrier_wait()  # phase 2: everyone holds the same pre-read token
    ctx.delay()
    try:
        client_proxy.put_object(Bucket=_XP_BUCKET, Key=_XP_KEY, Body=payload, IfMatch=etag)
    except Exception as exc:  # noqa: BLE001 - manager transports remote errors opaquely
        if "PreconditionFailed" in repr(exc):
            return "conflict"
        raise
    return "won"


def _xp_read_then_write_contender(ctx, store_proxy, expected_token: str, payload: bytes) -> str:
    """The qm defect class: the conditional is assembled CLIENT-side from two
    proxied calls (read+compare, then unconditional write) with a real gap."""
    ctx.barrier_wait()  # phase 1: everyone is running
    entry = store_proxy.get(_XP_KEY)
    ctx.barrier_wait()  # phase 2: everyone has read the SAME token
    ctx.delay()
    if entry is None or entry[1] != expected_token:
        return "conflict"
    store_proxy.set(_XP_KEY, payload)  # unconditional -- the gap is the bug
    return "won"


def assert_cross_process_one_winner_native_cas(
    *, contenders: int = 2, delays: tuple[float, ...] = (0.0, 0.05), timeout_sec: float = 60.0
) -> None:
    """R-1 + R-2, positive arm: N contender OS processes race one native
    conditional write; exactly one wins, every loser gets the typed conflict,
    and a final authoritative read shows the winner's bytes (the loser's write
    left no trace). Must hold at EVERY delay vector including all-zero."""
    assert len(delays) == contenders, "one delay per contender"
    manager = _SubstrateManager()
    manager.start()
    try:
        client = manager.LocalS3Client()
        client.create_bucket(_XP_BUCKET, versioned=True)
        seeded = client.put_object(Bucket=_XP_BUCKET, Key=_XP_KEY, Body=b"seed")
        etag = seeded["ETag"]
        payloads = [f"contender-{i}".encode() for i in range(contenders)]
        harness = ProcessRaceHarness(timeout_sec=timeout_sec)
        result = harness.race(
            [
                ContenderSpec(
                    _xp_native_cas_contender,
                    args=(client, etag, payloads[i]),
                    delay_seconds=delays[i],
                )
                for i in range(contenders)
            ]
        )
        for outcome in result.outcomes:
            if outcome.error is not None:
                raise outcome.error
        verdicts = [o.value for o in result.outcomes]
        winners = [i for i, v in enumerate(verdicts) if v == "won"]
        if len(winners) != 1:
            raise CrossProcessLostUpdate(
                f"expected exactly one winner, got {len(winners)}: {verdicts} (lost update)"
            )
        assert verdicts.count("conflict") == contenders - 1, verdicts
        final = client.get_object(Bucket=_XP_BUCKET, Key=_XP_KEY)["Body"].read()
        assert final == payloads[winners[0]], (
            "the winner's bytes did not survive -- a loser's write left a trace (R-2)"
        )
    finally:
        manager.shutdown()


def _run_broken_read_then_write_race(*, raise_on_loss: bool = False) -> list[str]:
    """Race the read-then-write control; return verdicts. With
    ``raise_on_loss`` the detected lost update raises CrossProcessLostUpdate,
    mirroring the conforming assertion's failure shape."""
    manager = _SubstrateManager()
    manager.start()
    try:
        store = manager.InMemoryStore()
        token = store.set(_XP_KEY, b"seed")
        harness = ProcessRaceHarness(timeout_sec=60.0)
        result = harness.race(
            [
                ContenderSpec(_xp_read_then_write_contender, args=(store, token, b"A"), delay_seconds=0.0),
                ContenderSpec(_xp_read_then_write_contender, args=(store, token, b"B"), delay_seconds=0.1),
            ]
        )
        for outcome in result.outcomes:
            if outcome.error is not None:
                raise outcome.error
        verdicts = [o.value for o in result.outcomes]
        if raise_on_loss and verdicts.count("won") != 1:
            raise CrossProcessLostUpdate(
                f"read-then-write binding produced a lost update: {verdicts}"
            )
        return verdicts
    finally:
        manager.shutdown()


def assert_in_process_binding_exemption_is_declared(binding: object | None = None) -> list[str]:
    """R-8 / KTD-4: an in-process-only binding is REFUSED a cross-process race
    with its declared reason recorded in the run report — nothing is silently
    skipped. Returns the report lines so a runner can print them. The default
    subject is the in-memory binding, the contract's first declared exemption."""
    subject = binding if binding is not None else InMemoryBinding()
    harness = ProcessRaceHarness()
    harness.refuse_to_race(subject, clause="R-1 cross-process race")
    report = harness.report()
    assert report and any("R-1 cross-process race" in line for line in report), report
    return report


def assert_cross_process_rejects_read_then_write() -> None:
    """R-1 teeth certification (the suite would be vacuous without it): a
    binding whose conditional is assembled client-side from separate calls
    MUST be caught -- both contenders report "won" and one update is silently
    lost. The two-phase barrier makes the catch deterministic: both contenders
    hold the same pre-read token before either writes."""
    verdicts = _run_broken_read_then_write_race()
    assert verdicts.count("won") == 2, (
        f"the read-then-write control was NOT caught (verdicts: {verdicts}); "
        "the cross-process race has lost its teeth"
    )


def assert_cross_process_lost_update_is_detected() -> None:
    """The same read-then-write control, surfaced as the TYPED failure: runs
    the broken binding and raises :class:`CrossProcessLostUpdate` on the
    detected loss (the loss is expected — callers assert the raise). Public
    wrapper so runners never import the module-private helper."""
    _run_broken_read_then_write_race(raise_on_loss=True)


def assert_forward_only_honest() -> None:
    """The forward-only arm (descriptor-level, pre-network): effect-ordering-only
    text with NO enforcement/CAS/rollback/dedup wording, and the no-version_source
    validation pinned. No binding ships for this tier in v1."""
    descriptor = CapabilityDescriptor(
        tier=Tier.FORWARD_ONLY,
        version_source=None,
        least_privilege="an action-backend credential",
        consistency_note="the effect is forward-only",
    )
    text = descriptor.guarantee_text.lower()
    assert "effect ordering only" in text
    assert "freshness" in text
    assert "deny-before-act" in text
    for word in _FORBIDDEN_ENFORCEMENT_WORDS:
        assert word not in text, f"forward-only text must not contain {word!r}"
    # A forward-only descriptor declaring a version_source is rejected: an action
    # mints no token to compare.
    with _expect_raises(ValueError, match="version_source"):
        CapabilityDescriptor(tier=Tier.FORWARD_ONLY, version_source="etag")


# ===========================================================================
# Durability regime — declaration + the kill-the-primary case (U6, R-5/KTD-5).
#
# The honest local claim is PROCESS-CRASH grade: SIGKILL destroys user-space
# state only (bytes handed to the OS survive in the page cache and are flushed
# by the kernel), so the local case proves an acknowledged commit's bytes had
# at least reached the OS — and can NEVER discriminate fsync-grade regimes.
# Both deeper grades are DECLARED exemptions (R-8), never silent skips.
#
# RUNBOOK — the per-binding-release managed kill-the-primary run (KTD-5).
# This is the release-procedure template the deferred exemption points at; it
# is prose here because the repo ships no managed binding yet:
#   1. Provision the managed substrate (e.g. RDS Postgres, DynamoDB) at the
#      binding's declared configuration — record the descriptor's
#      durability_facts verbatim in the run log.
#   2. Drive the binding through a committed, ACKNOWLEDGED write (the same
#      ack-then-park protocol as run_kill_after_ack; ack only after the
#      substrate confirmed the commit).
#   3. Kill the PRIMARY node through the provider's own failure lever
#      (reboot-with-failover, AZ kill), not a client disconnect.
#   4. After failover, read back through a fresh connection: the acknowledged
#      write is present → the declared regime holds; absent → record a
#      refutation against the descriptor's claim, and the binding's regime
#      must be downgraded before release.
#   5. File the run (date, substrate version, facts, outcome) with the
#      release notes; the exemption stays declared until a run replaces it.
# ===========================================================================


class UndeclaredDurabilityRegime(AssertionError):
    """A binding offered no durability regime (or an unverifiable one).

    Typed refusal: the kit never assumes a grade for a silent descriptor —
    an undeclared axis is NOT process-crash by default."""


class AcknowledgedWriteLost(AssertionError):
    """An ACKNOWLEDGED write was missing after the writer was SIGKILLed.

    The binding acked before its bytes reached the OS — the process-crash
    grade is violated (the loss a kill case genuinely can catch)."""


def assert_durability_regime_is_declared(descriptor: CapabilityDescriptor) -> None:
    """The descriptor speaks the durability axis: a regime AND its facts.

    A regime without configuration facts is refused too — a claim nothing
    qualifies cannot be checked against the substrate's actual settings."""
    if descriptor.durability_regime is None:
        raise UndeclaredDurabilityRegime(
            "binding declares no durability_regime; refusing to assume a "
            "grade (an undeclared axis is not process-crash by default)"
        )
    if not descriptor.durability_facts:
        raise UndeclaredDurabilityRegime(
            f"durability_regime {descriptor.durability_regime.value!r} is "
            "declared without configuration facts; an unverifiable claim is "
            "refused (state e.g. the journal/synchronous settings it rides on)"
        )


def assert_process_crash_durability(
    spec: ContenderSpec,
    verify_durable: "Callable[[Any], bool]",
    *,
    timeout_sec: float = 60.0,
) -> None:
    """The generic kill-the-primary case at PROCESS-CRASH grade.

    ``spec.fn`` performs the write, ``ctx.ack(facts)``s it, then parks (see
    :func:`ccs.testing.process_harness.run_kill_after_ack` for the protocol);
    the harness SIGKILLs it on the ack. ``verify_durable(facts)`` then reads
    the substrate with the writer destroyed and returns whether the
    acknowledged write is present — False raises the typed
    :class:`AcknowledgedWriteLost`. Deterministic: the ack is the only
    synchronization, and the assertion holds at any interleaving after it."""
    facts = run_kill_after_ack(spec, timeout_sec=timeout_sec)
    if not verify_durable(facts):
        raise AcknowledgedWriteLost(
            f"acknowledged write missing after SIGKILL: {facts!r} — the "
            "binding acked before its bytes reached the OS"
        )


def _durability_sqlite_child(ctx: Any, db_path_str: str, artifact_hex: str, agent_hex: str) -> None:
    """Child: open the sqlite registry, commit a WIN, ack it, park for the kill."""
    from ccs.core.states import MESIState
    from ccs.core.types import Artifact

    registry = SqliteArtifactRegistry(Path(db_path_str))
    artifact = Artifact(
        id=UUID(hex=artifact_hex), name="durability.md", version=1, content_hash="seed"
    )
    registry.register_artifact(artifact, "seed-content")
    agent = UUID(hex=agent_hex)
    registry.set_agent_state(artifact.id, agent, MESIState.SHARED, trigger="fetch", tick=0)
    committed_hash = sha256_hex(b"the acknowledged bytes")
    result = registry.commit_cas(
        artifact.id, agent, expected_version=1, content_hash=committed_hash
    )
    updated = result[0]  # WIN returns (updated_artifact, invalidated_agent_ids)
    # The ack goes out only AFTER the registry's COMMIT returned — exactly the
    # acknowledgement whose durability the case certifies. The registry handle
    # stays open (never .close()d) so SIGKILL destroys a LIVE writer; ack()
    # parks and never returns.
    ctx.ack(
        {
            "artifact": artifact_hex,
            "version": updated.version,
            "content_hash": committed_hash,
        }
    )


def assert_sqlite_acknowledged_commit_survives_kill(db_path: "Path | str") -> None:
    """The kill-the-primary case on the existing store (KTD-5's local leg).

    A spawned child opens :class:`SqliteArtifactRegistry` at ``db_path``,
    commits a WIN, acknowledges, parks, and is SIGKILLed; the reopened store
    (writer open — it runs WAL recovery over the dead writer's hot journal)
    must serve the acknowledged version with the acknowledged content hash."""
    db = Path(db_path)
    artifact_hex, agent_hex = uuid4().hex, uuid4().hex

    def _verify(facts: Any) -> bool:
        reopened = SqliteArtifactRegistry(db)
        try:
            artifact = reopened.get_artifact(UUID(hex=facts["artifact"]))
            return (
                artifact is not None
                and artifact.version == facts["version"]
                and artifact.content_hash == facts["content_hash"]
            )
        finally:
            reopened.close()

    assert_process_crash_durability(
        ContenderSpec(
            fn=_durability_sqlite_child, args=(str(db), artifact_hex, agent_hex)
        ),
        _verify,
    )


#: Why no local case can certify fsync-grade (OS-crash) durability (R-8).
FSYNC_GRADE_EXEMPTION_REASON = (
    "SIGKILL destroys user-space state only — bytes in the OS page cache "
    "survive and are flushed by the kernel — so no local kill can "
    "discriminate fsync-grade regimes (sqlite synchronous NORMAL vs FULL "
    "differ only under OS crash/power loss); a crash-simulating VFS is the "
    "named future closer"
)

#: Why the managed kill-the-primary run is deferred to release time (KTD-5).
MANAGED_KILL_RUN_EXEMPTION_REASON = (
    "kill-the-primary against a managed cluster is a per-binding-release "
    "run (see this section's runbook template), not a CI case; deferred "
    "with this declaration, never silently skipped"
)


def declare_deferred_durability_exemptions(
    harness: ProcessRaceHarness,
) -> "tuple[Any, Any]":
    """Record the two deferred durability grades as DECLARED exemptions (R-8)
    so the kit report names them; returns ``(fsync_grade, managed_run)``."""
    fsync = harness.declare_exemption(
        subject="durability",
        clause="R-5 fsync-grade discrimination",
        reason=FSYNC_GRADE_EXEMPTION_REASON,
    )
    managed = harness.declare_exemption(
        subject="durability",
        clause="R-5 managed kill-the-primary (KTD-5)",
        reason=MANAGED_KILL_RUN_EXEMPTION_REASON,
    )
    return fsync, managed


# ===========================================================================
# never-ship-a-store — the coordinator RETENTION backstop (b).
# ===========================================================================


def _dump_retention_rows(db_path) -> list[dict]:
    """Dump the coordinator's ``artifact_versions`` table (data-in for the
    ``retention_is_empty_for`` predicate). Any row is a content-proportional
    shadow the floor forbids for a binding's artifacts."""
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute("SELECT artifact_id, version FROM artifact_versions").fetchall()
    finally:
        con.close()
    return [{"artifact_id": UUID(hex=aid), "version": version} for aid, version in rows]


def assert_coordinator_retention_empty(tmp_path: Path) -> None:
    """(b) Coordinator backstop, WITH ``retain_versions=True`` (else it false-greens
    exactly when it matters): the binding's hash-only registration leg leaves the
    ``artifact_versions`` table EMPTY, while a content-bearing register does NOT —
    proving the zero-rows result is a real property, not a retention-off vacuum.

    Note: the row-count probe cannot see a hypothetical NEW content-proportional
    store OUTSIDE ``artifact_versions`` — that is the core ``never_ship_a_store``
    predicate's job (Unit 1) plus the per-binding review checklist.
    """
    from ccs.coordinator.sqlite_registry import SqliteArtifactRegistry
    from ccs.core.types import Artifact

    db_path = tmp_path / "state.db"
    with SqliteArtifactRegistry(db_path, retain_versions=True) as reg:
        # The binding registers via the hash-only resolve_or_register path — never
        # register_artifact(artifact, content), which lands a v1 body under retention.
        binding_id = reg.resolve_or_register("byo/row", "a" * 64)
        rows = _dump_retention_rows(db_path)
        assert retention_is_empty_for(rows, [binding_id]), (
            "the binding's hash-only registration must leave artifact_versions empty"
        )

        # Teeth: a content-bearing register DOES capture a body row under the same
        # retention — so the zero-rows result above cannot be a vacuous pass.
        bearing = Artifact(id=uuid4(), name="content-bearing", version=1, content_hash="b" * 64)
        reg.register_artifact(bearing, content="a body")
        rows_after = _dump_retention_rows(db_path)
        assert not retention_is_empty_for(rows_after, [bearing.id]), (
            "retention must be ACTIVE — a content-bearing register must leave a row"
        )


# ===========================================================================
# Session harness — a factory that mints agent identities over ONE coordinator.
# ===========================================================================


@runtime_checkable
class SessionFactory(Protocol):
    """Mint a fresh :class:`SubstrateCoordinatorSession` (a new agent identity) on
    ONE coordinator root. Two calls are two agents sharing one artifact."""

    def __call__(self) -> SubstrateCoordinatorSession: ...


class CoordinatorHarness:
    """Owns the coordinator lifecycle for a conformance run: each call mints a new
    agent identity on ONE root; :meth:`close` tears the coordinator down."""

    def __init__(self, root: Path | str, config: LifecycleConfig | None) -> None:
        self._root = Path(root)
        self._config = config

    def __call__(self) -> SubstrateCoordinatorSession:
        return SubstrateCoordinatorSession(self._root, managed=("**",), config=self._config)

    def close(self) -> None:
        from ccs.adapters.claude_code.lifecycle import stop_coordinator

        stop_coordinator(self._root)


# ---------------------------------------------------------------------------
# real_substrate note (MinIO): the racing-writers + invalidation assertions ALSO
# run against real Postgres AND real S3 when the `real_substrate` marker is
# selected and credentials are present (CCS_TEST_PG_DSN / CCS_REAL_S3_BUCKET) —
# wired in test_tier_honesty.py. Moto/LocalStack are excluded (they serialize →
# concurrency false-greens). For a local S3-compatible fixture, PIN MinIO to its
# FINAL release tag/digest (upstream archived read-only 2026-04-25 — API drift is
# impossible, but so are CVE fixes; never expose it) — its conditional-write
# semantics are adequate (real 412s, atomic, wildcard If-None-Match) EXCEPT it
# documents no 409 ConditionalRequestConflict path, so the 409-retry branch is
# stub-tested locally and integration-verified against real AWS only. Do NOT stand
# up MinIO here — this note is the record; the env-gated tests do the rest.
# ---------------------------------------------------------------------------


# ===========================================================================
# ===========================================================================
# The Workspace-Versioning conformance family (WV plan Unit 9 / R11).
#
# The corpus that tests OUR workspace-versioning implementation and any FOREIGN
# one, on the same seam discipline as the tier kit above:
#
# - MUST-MATCH scenarios pin behavior every implementation must reproduce
#   exactly (one-winner restore arbitration, torn-cut detection, restore
#   termination, monotonic forward restore, ABSENT-as-a-fact).
# - DECLARED scenarios pin behavior *per the binding's own declaration*
#   (pinnable vs restorable-unpinned, versioned vs unversioned history,
#   restart survival) — an implementation may declare less, never lie more.
#
# FIXTURE-DISCLOSURE RULE (binding, companion doc R1/R2): scenario prose and
# assertions describe failure scenarios and required FINAL OBSERVABLE OUTCOMES
# only — captured facts, restore outcomes from the closed public vocabulary,
# live bytes, version pointers. A foreign implementation must be able to pass
# by satisfying the outcomes, never by mimicking this repository's internals.
# ===========================================================================
# ===========================================================================


# --- the observable facts a WV conformance run asserts against ---------------


@dataclass(frozen=True)
class WorkspaceMemberFacts:
    """One member's captured, observable checkpoint facts.

    ``absent`` records absent-at-capture (a FACT, distinct from an empty
    body); ``fingerprint`` is the sha-256 hex of the captured bytes (``None``
    iff absent); ``dirty_during_window`` is the torn-cut flag (the member was
    NOT verified quiescent across the capture window); ``restore_tier`` is the
    member's honest restore claim (``restorable`` / ``restorable-unpinned`` /
    ``forward_only``); ``version_pointer`` is the substrate version the
    checkpoint points at (``None`` when the substrate keeps no history).
    """

    ref: str
    absent: bool
    fingerprint: str | None
    dirty_during_window: bool
    restore_tier: str
    version_pointer: str | None


@dataclass(frozen=True)
class WorkspaceCheckpointFacts:
    """One checkpoint's observable facts: an id + per-member capture facts."""

    checkpoint_id: str
    members: tuple[WorkspaceMemberFacts, ...]

    def member(self, ref: str) -> WorkspaceMemberFacts:
        for member in self.members:
            if member.ref == ref:
                return member
        raise KeyError(f"no member {ref!r} in checkpoint {self.checkpoint_id!r}")


@dataclass(frozen=True)
class RestoredMemberFacts:
    """One member's terminal restore outcome, from the closed public vocabulary
    (``restored`` / ``converged`` / ``conflict`` / ``held_unconfirmed`` /
    ``target_lost`` / ``forward_only_skipped``)."""

    ref: str
    outcome: str


@dataclass(frozen=True)
class WorkspaceRestoreFacts:
    """One restore run's observable conclusion.

    ``concluded`` is the termination contract's observable: a restore that
    starts must finish with EVERY member carrying exactly one terminal
    outcome — never a hang, never a raise that loses the per-member report.
    """

    checkpoint_id: str
    concluded: bool
    members: tuple[RestoredMemberFacts, ...]

    def member(self, ref: str) -> RestoredMemberFacts:
        for member in self.members:
            if member.ref == ref:
                return member
        raise KeyError(f"no member {ref!r} in restore of {self.checkpoint_id!r}")


# --- the WV binding seam — the shipped ConformanceBinding, extended -----------


@runtime_checkable
class WorkspaceConformanceBinding(ConformanceBinding, Protocol):
    """What a WV conformance run needs from an implementation under test.

    EXTENDS the shipped :class:`ConformanceBinding` (descriptor / seed /
    foreign_write / make_view) with the workspace-versioning capability
    members. Every member is outcome-level: capability declarations
    (properties), workspace verbs returning observable facts, observation
    channels, and scenario seams phrased as substrate/world EVENTS (a write
    landing inside a capture window, a foreign writer that keeps winning, a
    lifecycle expiring history, a process restart) — never as any particular
    implementation's mechanism.
    """

    # --- capability declarations (the DECLARED split rides on these) ---------

    @property
    def declares_versioned(self) -> bool:
        """True iff the substrate keeps per-member version history."""
        ...

    @property
    def declares_pinnable(self) -> bool:
        """True iff a checkpointed version can be pinned against expiry."""
        ...

    @property
    def declares_restart_survival(self) -> bool:
        """True iff checkpoints survive a process restart."""
        ...

    # --- workspace verbs ------------------------------------------------------

    def checkpoint_workspace(self, refs: Sequence[str]) -> WorkspaceCheckpointFacts:
        """Capture one checkpoint over ``refs`` (present or absent members)."""
        ...

    def restore_workspace(self, checkpoint_id: str) -> WorkspaceRestoreFacts:
        """Restore a checkpoint; MUST return a concluded per-member report."""
        ...

    # --- observation channels (never routed through the engine under test) ----

    def read_live(self, ref: str) -> bytes | None:
        """The member's live bytes, or ``None`` when absent."""
        ...

    def live_version_pointer(self, ref: str) -> str | None:
        """The member's current substrate version pointer (``None`` when the
        substrate keeps no history or the member is absent)."""
        ...

    def read_at_pointer(self, ref: str, pointer: str) -> bytes | None:
        """The bytes at a historical version pointer, or ``None`` when gone."""
        ...

    # --- scenario seams (world events, not mechanisms) ------------------------

    def write_during_capture(self, ref: str, data: bytes) -> None:
        """Arrange that ``data`` lands on ``ref`` INSIDE the next checkpoint's
        capture window (after ``ref`` is captured, before the cut concludes)."""
        ...

    def sustain_contention(self, ref: str) -> None:
        """Arrange a foreign writer that races EVERY subsequent restore
        attempt on ``ref`` and always wins (each attempt observes freshly
        moved state)."""
        ...

    def expire_noncurrent_history(self, ref: str) -> None:
        """Simulate the substrate's lifecycle expiring every noncurrent
        version of ``ref``; versions under an established pin survive, as the
        substrate guarantees. Raise ``NotImplementedError`` when this arm
        cannot simulate expiry on demand (the scenario then verifies the
        declaration only)."""
        ...

    def restart(self) -> None:
        """Simulate a process restart: durable state survives per the
        binding's declaration; in-memory engine state does not."""
        ...


#: The full WV Protocol surface (inherited + WV members) — the drift anchor.
#: A member added to :class:`WorkspaceConformanceBinding` without a matching
#: update here fails the bidirectional drift-guard test (house rule: Protocol
#: members incl. properties + frozenset drift guard + teeth).
WORKSPACE_BINDING_MEMBERS: Final[frozenset[str]] = frozenset(
    {
        # inherited from the shipped ConformanceBinding
        "descriptor",
        "seed",
        "foreign_write",
        "make_view",
        # declarations
        "declares_versioned",
        "declares_pinnable",
        "declares_restart_survival",
        # workspace verbs
        "checkpoint_workspace",
        "restore_workspace",
        # observation channels
        "read_live",
        "live_version_pointer",
        "read_at_pointer",
        # scenario seams
        "write_during_capture",
        "sustain_contention",
        "expire_noncurrent_history",
        "restart",
    }
)


def _declared_protocol_members(protocol: type) -> frozenset[str]:
    """The public members a Protocol declares, inherited Protocol bases included.

    CPython exposes this as ``__protocol_attrs__``, but only from 3.12 — and this
    package supports 3.11 (``requires-python = ">=3.11"``), where the attribute
    does not exist. So the members are computed here instead: walk the MRO,
    skipping ``object`` and the ``Protocol``/``Generic`` scaffolding, and collect
    annotations and class attributes. Underscored names are excluded rather than
    replicating CPython's exclusion list, which differs between versions: a
    conformance seam is public by construction, so anything underscored is
    machinery, not contract.

    ``test_portable_protocol_members_match_the_interpreter`` pins this against
    ``__protocol_attrs__`` on every interpreter that has it, so the portable path
    cannot drift from CPython's own notion of the surface unnoticed.
    """
    return frozenset(
        name
        for base in protocol.__mro__
        if base is not object and base.__name__ not in {"Protocol", "Generic"}
        for name in (*vars(base), *getattr(base, "__annotations__", {}))
        if not name.startswith("_")
    )


def workspace_protocol_members() -> frozenset[str]:
    """The members :class:`WorkspaceConformanceBinding` actually declares —
    compared against :data:`WORKSPACE_BINDING_MEMBERS` by the drift-guard test
    (bidirectional: neither the Protocol nor the anchor may move alone)."""
    return _declared_protocol_members(WorkspaceConformanceBinding)


def assert_workspace_binding_declares_all(binding: WorkspaceConformanceBinding) -> None:
    """The instance-level drift guard: every WV Protocol member is present on
    the binding under test (properties included) before any scenario runs."""
    missing = sorted(
        member for member in WORKSPACE_BINDING_MEMBERS if not hasattr(binding, member)
    )
    assert not missing, (
        f"WorkspaceConformanceBinding drift: {missing} not declared by "
        f"{type(binding).__name__} — the WV conformance family cannot run "
        "against a partial seam"
    )


# ===========================================================================
# MUST-MATCH scenarios — behavior EVERY implementation must reproduce.
# ===========================================================================


def assert_wv_absent_is_a_fact(binding: WorkspaceConformanceBinding) -> None:
    """MUST-MATCH — ABSENT is a fact, not an empty body.

    Failure scenario: a member is absent when the checkpoint is cut and a
    sibling member is present but EMPTY; later writes create/overwrite both.
    Required final outcomes: the checkpoint records the absent member as
    absent with NO fingerprint, records the empty member as present with the
    empty-content fingerprint, and a restore brings ABSENCE back for the
    first and EMPTINESS back for the second — never conflating the two.
    """
    absent_ref = _fresh_ref("wv")
    empty_ref = _fresh_ref("wv")
    binding.seed(empty_ref, b"")
    checkpoint = binding.checkpoint_workspace([absent_ref, empty_ref])

    absent_facts = checkpoint.member(absent_ref)
    empty_facts = checkpoint.member(empty_ref)
    assert absent_facts.absent is True and absent_facts.fingerprint is None, (
        "absent-fact leg: an absent-at-capture member must be recorded as the "
        "ABSENT fact (no fingerprint), never as an empty body"
    )
    assert empty_facts.absent is False and empty_facts.fingerprint == _sha256(b""), (
        "absent-fact leg: a present-but-empty member must be recorded PRESENT "
        "with the empty-content fingerprint — empty is a body, absent is not"
    )

    binding.foreign_write(absent_ref, b"created-after-the-cut")
    binding.foreign_write(empty_ref, b"filled-after-the-cut")
    report = binding.restore_workspace(checkpoint.checkpoint_id)
    assert report.concluded
    assert report.member(absent_ref).outcome in RESTORE_SUCCESS_OUTCOMES
    assert report.member(empty_ref).outcome in RESTORE_SUCCESS_OUTCOMES
    assert binding.read_live(absent_ref) is None, (
        "absent-fact leg: restore must bring back ABSENCE for a member the "
        "checkpoint recorded absent"
    )
    assert binding.read_live(empty_ref) == b"", (
        "absent-fact leg: restore must bring back the EMPTY body for a member "
        "the checkpoint recorded empty"
    )


def assert_wv_torn_cut_detection(binding: WorkspaceConformanceBinding) -> None:
    """MUST-MATCH — torn-cut detection fires on an intra-window write.

    Failure scenario: while the checkpoint's cut is being taken, a write lands
    on one member inside the capture window; a sibling member stays quiescent.
    Required final outcome: the checkpoint marks EXACTLY the moved member as
    not-verified-quiescent (``dirty_during_window``) and leaves the quiescent
    member clean — a torn cut is declared, never silently presented as clean.
    """
    moved_ref = _fresh_ref("wv")
    quiet_ref = _fresh_ref("wv")
    binding.seed(moved_ref, b"moved-base")
    binding.seed(quiet_ref, b"quiet-base")
    binding.write_during_capture(moved_ref, b"landed-inside-the-window")

    checkpoint = binding.checkpoint_workspace([moved_ref, quiet_ref])
    assert checkpoint.member(moved_ref).dirty_during_window is True, (
        "torn-cut detection leg: a write landing inside the capture window "
        "must mark that member dirty_during_window — a clean flag here is a "
        "silently torn checkpoint"
    )
    assert checkpoint.member(quiet_ref).dirty_during_window is False, (
        "torn-cut detection leg: a quiescent member must NOT be flagged — the "
        "torn-cut signal must name exactly the moved member"
    )


def assert_wv_restore_one_winner(binding: WorkspaceConformanceBinding) -> None:
    """MUST-MATCH — one-winner restore arbitration.

    Failure scenario: after the checkpoint is cut, a foreign (non-coordinated)
    writer moves the member; the restore then runs against the moved state.
    Required final outcomes: the restore's write and the foreign write resolve
    to EXACTLY one winner — here (no further contention) the restore lands the
    captured bytes as the live state and reports success — and on a versioned
    substrate the foreign write SURVIVES at its own version pointer:
    arbitration, never obliteration, and never a torn interleaving.
    """
    ref = _fresh_ref("wv")
    binding.seed(ref, b"captured-state")
    checkpoint = binding.checkpoint_workspace([ref])

    binding.foreign_write(ref, b"foreign-state")
    foreign_pointer = binding.live_version_pointer(ref)

    report = binding.restore_workspace(checkpoint.checkpoint_id)
    assert report.concluded
    live = binding.read_live(ref)
    assert live in (b"captured-state", b"foreign-state"), (
        "one-winner arbitration leg: the final live bytes must be exactly one "
        "writer's — a mixed/torn body means the restore write was not "
        "arbitrated atomically"
    )
    assert live == b"captured-state", (
        "one-winner arbitration leg: with no further contention the restore "
        "must win against an already-landed foreign write (its write runs "
        "against the current state, not a stale assumption)"
    )
    assert report.member(ref).outcome in RESTORE_SUCCESS_OUTCOMES, (
        "one-winner arbitration leg: a landed restore must report success — "
        "the reported outcome must match the observable winner"
    )
    if foreign_pointer is not None:
        assert binding.read_at_pointer(ref, foreign_pointer) == b"foreign-state", (
            "one-winner arbitration leg: the losing foreign write must survive "
            "as history on a versioned substrate — arbitration, never erasure"
        )


def assert_wv_restore_terminates_under_contention(
    binding: WorkspaceConformanceBinding,
) -> None:
    """MUST-MATCH — restore terminates under sustained contention.

    Failure scenario: a foreign writer races EVERY restore attempt on the
    member and always wins. Required final outcomes: the restore RETURNS (no
    livelock) with a concluded report; the contended member's terminal outcome
    is ``conflict`` (an honest loss); the sustained winner's state survives —
    the restore must not blind-overwrite a writer it could not beat.
    """
    ref = _fresh_ref("wv")
    binding.seed(ref, b"captured-state")
    checkpoint = binding.checkpoint_workspace([ref])

    binding.foreign_write(ref, b"foreign-opening-move")
    binding.sustain_contention(ref)
    report = binding.restore_workspace(checkpoint.checkpoint_id)

    assert report.concluded, (
        "termination contract: a restore under sustained contention must still "
        "conclude with a per-member report — never hang, never abandon"
    )
    assert report.member(ref).outcome == RESTORE_OUTCOME_CONFLICT, (
        "termination contract: sustained foreign wins must conclude as the "
        "absorbing 'conflict' outcome — an honest loss, not a success claim"
    )
    live = binding.read_live(ref)
    assert live is not None and live != b"captured-state", (
        "one-winner arbitration leg: under sustained contention the foreign "
        "writer's state must survive — a live state matching the checkpoint "
        "means the restore blind-overwrote a writer it reported losing to"
    )


def assert_wv_monotonic_restore(binding: WorkspaceConformanceBinding) -> None:
    """MUST-MATCH — restore is a forward commit, never a rewind.

    Failure scenario: the member moved on after the checkpoint; the restore
    brings the old bytes back. Required final outcomes: the live bytes equal
    the captured bytes UNDER A NEW version pointer (neither the checkpointed
    pointer nor the pre-restore live pointer), and the pre-restore state
    remains readable at its own pointer — restoring old content must never
    rewind or erase the version history.
    """
    ref = _fresh_ref("wv")
    binding.seed(ref, b"first-state")
    checkpoint = binding.checkpoint_workspace([ref])
    captured_pointer = checkpoint.member(ref).version_pointer
    assert captured_pointer is not None, (
        "monotonic-restore leg: a versioned member must checkpoint a version "
        "pointer to restore from"
    )

    binding.foreign_write(ref, b"second-state")
    pre_restore_pointer = binding.live_version_pointer(ref)

    report = binding.restore_workspace(checkpoint.checkpoint_id)
    assert report.concluded
    assert report.member(ref).outcome in RESTORE_SUCCESS_OUTCOMES
    assert binding.read_live(ref) == b"first-state"

    post_restore_pointer = binding.live_version_pointer(ref)
    assert post_restore_pointer not in (captured_pointer, pre_restore_pointer), (
        "monotonic-restore leg: restoring old bytes must mint a NEW version — "
        "reusing an old pointer is a version-axis rewind"
    )
    assert binding.read_at_pointer(ref, captured_pointer) == b"first-state"
    if pre_restore_pointer is not None:
        assert binding.read_at_pointer(ref, pre_restore_pointer) == b"second-state", (
            "monotonic-restore leg: the pre-restore state must remain readable "
            "at its own pointer — restore is forward, never erasure"
        )


# ===========================================================================
# DECLARED scenarios — behavior per the binding's own declaration.
# ===========================================================================


def assert_wv_declared_versioning_honest(binding: WorkspaceConformanceBinding) -> None:
    """DECLARED — versioned vs unversioned history, per declaration.

    A versioned binding must checkpoint a version pointer and claim at least
    ``restorable-unpinned``. An unversioned binding must still DESCRIBE the
    member (fingerprint) but claim only ``forward_only`` — and a restore must
    SKIP the member and say so, leaving the live state untouched. No history
    may never masquerade as restorability.
    """
    ref = _fresh_ref("wv")
    binding.seed(ref, b"declared-body")
    checkpoint = binding.checkpoint_workspace([ref])
    facts = checkpoint.member(ref)
    assert facts.fingerprint == _sha256(b"declared-body"), (
        "capture must DESCRIBE the member (fingerprint) on every arm — "
        "describing is not claiming restorability"
    )
    if binding.declares_versioned:
        assert facts.version_pointer is not None, (
            "a versioned member must manifest the version pointer its restore "
            "would read from"
        )
        assert facts.restore_tier in (
            RestoreTier.RESTORABLE.value,
            RestoreTier.RESTORABLE_UNPINNED.value,
        ), "a versioned member claims restorable or restorable-unpinned"
        return
    assert facts.version_pointer is None, (
        "an unversioned member has no history to point at — a pointer here "
        "is an over-claim"
    )
    assert facts.restore_tier == RestoreTier.FORWARD_ONLY.value, (
        "no history → the only honest tier is forward_only (described, "
        "never restorable)"
    )
    binding.foreign_write(ref, b"moved-on")
    report = binding.restore_workspace(checkpoint.checkpoint_id)
    assert report.concluded
    assert report.member(ref).outcome == RESTORE_OUTCOME_FORWARD_ONLY_SKIPPED, (
        "a forward_only member must be SKIPPED and say so — restoring it "
        "would be fabricating history"
    )
    assert binding.read_live(ref) == b"moved-on", (
        "a skipped member's live state must be untouched"
    )


def assert_wv_declared_pin_honest(binding: WorkspaceConformanceBinding) -> None:
    """DECLARED — pinnable vs ``restorable-unpinned``, per declaration.

    A pinnable binding checkpoints the member as ``restorable`` and the pinned
    version SURVIVES a history expiry: the restore still lands the captured
    bytes. An unpinnable (but versioned) binding must say
    ``restorable-unpinned`` loudly — and once expiry takes the captured
    version, the restore must report ``target_lost``: an honest loss, never a
    silent wrong-content restore.
    """
    if not binding.declares_versioned:
        return  # no history axis: the versioning scenario already pins this arm
    ref = _fresh_ref("wv")
    binding.seed(ref, b"precious-state")
    checkpoint = binding.checkpoint_workspace([ref])
    facts = checkpoint.member(ref)
    if binding.declares_pinnable:
        assert facts.restore_tier == RestoreTier.RESTORABLE.value, (
            "a pinnable member must checkpoint as 'restorable' — the claim its "
            "pin backs"
        )
    else:
        assert facts.restore_tier == RestoreTier.RESTORABLE_UNPINNED.value, (
            "an unpinnable member must say 'restorable-unpinned' LOUDLY — "
            "claiming 'restorable' without a pin is the silent-decay lie"
        )

    binding.foreign_write(ref, b"newer-state")
    try:
        binding.expire_noncurrent_history(ref)
    except NotImplementedError:
        return  # declaration verified; this arm cannot simulate expiry on demand

    report = binding.restore_workspace(checkpoint.checkpoint_id)
    assert report.concluded
    outcome = report.member(ref).outcome
    if binding.declares_pinnable:
        assert outcome in RESTORE_SUCCESS_OUTCOMES, (
            "the pin's teeth: a pinned version must survive history expiry and "
            "still restore"
        )
        assert binding.read_live(ref) == b"precious-state"
        return
    assert outcome == RESTORE_OUTCOME_TARGET_LOST, (
        "honest loss: an expired unpinned target must report 'target_lost' — "
        "never restore different content silently"
    )
    assert binding.read_live(ref) == b"newer-state", (
        "an honest loss leaves the live state untouched"
    )


def assert_wv_restart_survival_per_declaration(
    binding: WorkspaceConformanceBinding,
) -> None:
    """DECLARED — checkpoints survive a process restart, per declaration.

    A binding declaring restart survival must restore a pre-restart checkpoint
    after :meth:`WorkspaceConformanceBinding.restart` exactly as before it. A
    binding declaring no survival is exempt (nothing observable to verify).
    """
    if not (binding.declares_restart_survival and binding.declares_versioned):
        return  # nothing declared to verify on this arm
    ref = _fresh_ref("wv")
    binding.seed(ref, b"pre-restart-state")
    checkpoint = binding.checkpoint_workspace([ref])
    binding.foreign_write(ref, b"post-checkpoint-state")

    binding.restart()

    report = binding.restore_workspace(checkpoint.checkpoint_id)
    assert report.concluded, (
        "restart survival: a declared-durable checkpoint must still restore "
        "after a process restart"
    )
    assert report.member(ref).outcome in RESTORE_SUCCESS_OUTCOMES
    assert binding.read_live(ref) == b"pre-restart-state"


def run_workspace_conformance(binding: WorkspaceConformanceBinding) -> None:
    """The WV family arm — run against the deterministic S3-semantics fake
    (default) AND, ``real_substrate``-gated, against a real versioned bucket.

    Drift guard first, then the DECLARED scenarios (both arms), then the
    MUST-MATCH restore family (which needs the history axis a versioned
    declaration provides — the unversioned arm's honest behavior is pinned by
    the versioning declaration scenario instead).
    """
    assert_workspace_binding_declares_all(binding)
    assert_wv_declared_versioning_honest(binding)
    assert_wv_torn_cut_detection(binding)
    if binding.declares_versioned:
        assert_wv_absent_is_a_fact(binding)
        assert_wv_restore_one_winner(binding)
        assert_wv_restore_terminates_under_contention(binding)
        assert_wv_monotonic_restore(binding)
    assert_wv_declared_pin_honest(binding)
    assert_wv_restart_survival_per_declaration(binding)


# ===========================================================================
# The reference arm — the shipped WV engine over the deterministic S3 fake.
# ===========================================================================


_WV_BUCKET = "wv-conformance"


class _WorkspaceSeamClient:
    """Wraps a boto3-shaped S3 client with the two WV scenario seams, phrased
    as substrate events:

    - :meth:`arm_write_during_capture` — after ``key``'s next plain read is
      served, the scheduled body lands BEFORE the next read of a DIFFERENT
      key. With ``key`` captured first and a second member still to capture,
      the write lands strictly inside the capture window (the WV torn-cut
      scenario orders its members exactly so); re-reads of ``key`` itself
      within one observation never trigger the landing.
    - :meth:`arm_contention` — every plain read of ``key`` is followed by a
      fresh foreign put, so any state observed for a conditional write is
      already stale by write time (the sustained foreign winner).

    Version-pinned reads are exempt from both seams — pinned history is
    immutable and no live writer can race it.
    """

    def __init__(self, inner: Any, bucket: str) -> None:
        self._inner = inner
        self._bucket = bucket
        self._armed_capture: dict[str, bytes] = {}
        self._due_capture: dict[str, bytes] = {}
        self._contended: dict[str, int] = {}

    def arm_write_during_capture(self, key: str, body: bytes) -> None:
        self._armed_capture[key] = bytes(body)

    def arm_contention(self, key: str) -> None:
        self._contended.setdefault(key, 0)

    def get_object(self, **kwargs: Any) -> Any:
        key = kwargs.get("Key")
        for due_key in [due for due in self._due_capture if due != key]:
            body = self._due_capture.pop(due_key)
            self._inner.put_object(Bucket=self._bucket, Key=due_key, Body=body)
        response = self._inner.get_object(**kwargs)
        if kwargs.get("VersionId") is None and isinstance(key, str):
            if key in self._armed_capture:
                self._due_capture[key] = self._armed_capture.pop(key)
            if key in self._contended:
                self._contended[key] += 1
                body = f"foreign-sustained-{self._contended[key]}".encode()
                self._inner.put_object(Bucket=self._bucket, Key=key, Body=body)
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _client_error_code(error: BaseException) -> str | None:
    """The S3 error code off a botocore-``ClientError``-shaped exception (the
    fake's :class:`S3LocalClientError` and the real driver's error share the
    ``.response['Error']['Code']`` duck shape), or ``None``."""
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        code = response.get("Error", {}).get("Code")
        if isinstance(code, str):
            return code
    return None


class LocalWorkspaceBinding:
    """The reference :class:`WorkspaceConformanceBinding`: the shipped WV
    engine (``WorkspaceVersioner`` over ``CoherentObject``) on the
    deterministic S3-semantics fake, with a durable coordinator store under
    ``state_dir`` (restart survival is REAL, not simulated).

    Three default-arm shapes via the constructor flags: versioned+pinnable
    (the full claim), versioned-unpinnable (the loud ``restorable-unpinned``
    downgrade), and unversioned (the honest ``forward_only`` refusal).
    Observation channels read the substrate directly — never through the
    engine under test.

    The ``real_substrate`` arm rides the SAME adapter: pass ``client=`` (a
    boto3 S3 client) and ``bucket=`` (a real bucket with versioning enabled;
    ``pinnable=True`` additionally requires Object Lock) — the scenario seams
    wrap whichever client is supplied, and the on-demand history-expiry seam
    degrades to ``NotImplementedError`` where the substrate offers no such
    event (the pin scenario then verifies the declaration only).
    """

    def __init__(
        self,
        state_dir: "Path | str",
        *,
        versioned: bool = True,
        pinnable: bool = True,
        client: Any | None = None,
        bucket: str | None = None,
    ) -> None:
        if pinnable and not versioned:
            raise ValueError(
                "a per-version pin requires version history (the S3 rule): "
                "pinnable=True needs versioned=True"
            )
        if client is None:
            client = LocalS3Client()
            client.create_bucket(_WV_BUCKET, versioned=versioned, object_lock=pinnable)
            bucket = _WV_BUCKET
        elif bucket is None:
            raise ValueError("an injected client needs its bucket name (bucket=)")
        self._raw = client
        self._bucket = bucket
        self._seams = _WorkspaceSeamClient(self._raw, bucket)
        self._object = CoherentObject(bucket, client=self._seams)
        self._versioned = versioned
        self._pinnable = pinnable
        self._owner = uuid4()
        self._db_path = Path(state_dir) / f"wv-conformance-{uuid4().hex}.db"
        self._checkpoint_refs: dict[str, tuple[str, ...]] = {}
        self._registry: SqliteArtifactRegistry | None = None
        self._service = self._build_service()

    # --- ConformanceBinding (inherited seam) ----------------------------------

    @property
    def descriptor(self) -> CapabilityDescriptor:
        return self._object.descriptor

    def seed(self, ref: str, data: bytes) -> None:
        self._raw.put_object(Bucket=self._bucket, Key=ref, Body=bytes(data))

    def foreign_write(self, ref: str, data: bytes) -> None:
        # A non-coordinated writer: straight to the substrate, the engine and
        # its coordinator are never told.
        self._raw.put_object(Bucket=self._bucket, Key=ref, Body=bytes(data))

    def make_view(self) -> CoherentObject:
        return CoherentObject(self._bucket, client=self._raw)

    # --- declarations ---------------------------------------------------------

    @property
    def declares_versioned(self) -> bool:
        return self._versioned

    @property
    def declares_pinnable(self) -> bool:
        return self._pinnable

    @property
    def declares_restart_survival(self) -> bool:
        return True  # the coordinator store is durable under state_dir

    # --- workspace verbs ------------------------------------------------------

    def checkpoint_workspace(self, refs: Sequence[str]) -> WorkspaceCheckpointFacts:
        versioner = self._versioner(refs)
        checkpoint = versioner.checkpoint(f"wv-conformance-{uuid4().hex[:8]}")
        checkpoint_id = checkpoint.record.checkpoint_id
        self._checkpoint_refs[checkpoint_id] = tuple(refs)
        return WorkspaceCheckpointFacts(
            checkpoint_id=checkpoint_id,
            members=tuple(
                WorkspaceMemberFacts(
                    ref=member.member_path,
                    absent=member.absent,
                    fingerprint=member.fingerprint,
                    dirty_during_window=member.dirty_during_window,
                    restore_tier=member.restore_tier,
                    version_pointer=member.native_token,
                )
                for member in checkpoint.members
            ),
        )

    def restore_workspace(self, checkpoint_id: str) -> WorkspaceRestoreFacts:
        # A FRESH engine per restore: the run is driven from durable state,
        # exactly what a foreign consumer of the corpus would have to do.
        versioner = self._versioner(self._checkpoint_refs[checkpoint_id])
        report = versioner.restore(checkpoint_id)
        return WorkspaceRestoreFacts(
            checkpoint_id=checkpoint_id,
            concluded=report.status == RESTORE_STATUS_CONCLUDED,
            members=tuple(
                RestoredMemberFacts(ref=member.member_path, outcome=member.outcome)
                for member in report.members
            ),
        )

    # --- observation channels (raw substrate, engine bypassed) ----------------

    def read_live(self, ref: str) -> bytes | None:
        try:
            response = self._raw.get_object(Bucket=self._bucket, Key=ref)
        except Exception as error:
            if _client_error_code(error) == "NoSuchKey":
                return None
            raise
        return response["Body"].read()

    def live_version_pointer(self, ref: str) -> str | None:
        try:
            response = self._raw.get_object(Bucket=self._bucket, Key=ref)
        except Exception as error:
            if _client_error_code(error) == "NoSuchKey":
                return None
            raise
        pointer = response.get("VersionId")
        return None if pointer in (None, "null") else pointer

    def read_at_pointer(self, ref: str, pointer: str) -> bytes | None:
        try:
            response = self._raw.get_object(
                Bucket=self._bucket, Key=ref, VersionId=pointer
            )
        except Exception as error:
            if _client_error_code(error) in ("NoSuchVersion", "MethodNotAllowed"):
                return None
            raise
        return response["Body"].read()

    # --- scenario seams -------------------------------------------------------

    def write_during_capture(self, ref: str, data: bytes) -> None:
        self._seams.arm_write_during_capture(ref, data)

    def sustain_contention(self, ref: str) -> None:
        self._seams.arm_contention(ref)

    def expire_noncurrent_history(self, ref: str) -> None:
        expire = getattr(self._raw, "lifecycle_expire_noncurrent", None)
        if expire is None:
            raise NotImplementedError(
                "this substrate offers no on-demand history expiry (a real "
                "bucket's lifecycle cannot be triggered from here)"
            )
        expire(self._bucket, ref)

    def restart(self) -> None:
        # The substrate (the S3 fake) outlives the process by design; the
        # coordinator store is durable on disk — only the in-process engine
        # and service die and are rebuilt.
        self._service = self._build_service()

    def close(self) -> None:
        if self._registry is not None:
            self._registry.close()
            self._registry = None

    # --- internals ------------------------------------------------------------

    def _build_service(self) -> CoordinatorService:
        self.close()
        self._registry = SqliteArtifactRegistry(self._db_path)
        return CoordinatorService(self._registry)

    def _versioner(self, refs: Sequence[str]) -> WorkspaceVersioner:
        versioner = WorkspaceVersioner(service=self._service, owner=self._owner)
        for ref in refs:
            versioner.add_object_member(self._object, ref, member_path=ref)
        return versioner


# ===========================================================================
# Teeth — the degraded stub (drops EXACTLY ONE WV leg) + its MUST-FAIL helper.
# ===========================================================================


class TornCutBlindWorkspaceBinding(LocalWorkspaceBinding):
    """TEETH: the foreign stand-in that drops EXACTLY ONE WV leg — torn-cut
    detection. Every checkpoint reports every member clean, whatever landed
    inside the window. It MUST fail :func:`assert_wv_torn_cut_detection`
    (whose message names the dropped leg) and MUST pass every unrelated
    scenario — the family's failures localize to the leg that is broken.
    """

    def checkpoint_workspace(self, refs: Sequence[str]) -> WorkspaceCheckpointFacts:
        facts = super().checkpoint_workspace(refs)
        return WorkspaceCheckpointFacts(
            checkpoint_id=facts.checkpoint_id,
            members=tuple(
                replace(member, dirty_during_window=False) for member in facts.members
            ),
        )


def assert_torn_cut_blind_binding_is_rejected(state_dir: "Path | str") -> None:
    """Confirm the WV family has TEETH: the torn-cut-blind stub fails exactly
    the torn-cut scenario, and the failure message NAMES the dropped leg."""
    stub = TornCutBlindWorkspaceBinding(state_dir)
    try:
        with _expect_raises(AssertionError, match="torn-cut detection leg"):
            assert_wv_torn_cut_detection(stub)
    finally:
        stub.close()
