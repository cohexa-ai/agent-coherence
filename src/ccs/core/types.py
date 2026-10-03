# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Core domain dataclasses for artifact coherence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Optional
from uuid import UUID, uuid4

from .states import MESIState, TransientState


@dataclass(frozen=True)
class Artifact:
    """Named shared artifact tracked by the coherence coordinator."""

    id: UUID = field(default_factory=uuid4)
    name: str = ""
    version: int = 0
    content_hash: Optional[str] = None
    size_tokens: Optional[int] = None
    depends_on: tuple[UUID, ...] = ()


@dataclass(frozen=True)
class ArtifactCacheEntry:
    """Per-agent cached view of one artifact."""

    artifact_id: UUID
    state: MESIState
    local_version: int
    access_count: int = 0
    acquired_at_tick: int = 0
    expires_at_tick: Optional[int] = None
    transient_state: Optional[TransientState] = None
    transient_entered_tick: Optional[int] = None


@dataclass(frozen=True)
class ConflictDetail:
    """Typed result of an optimistic-concurrency commit that did NOT mutate.

    Returned (never raised) by the registry ``commit_cas`` primitive and the
    service ``commit_cas`` orchestration when a compare-and-swap loses the
    race (OCC write API, plan R2 / R-OCC-2). The three retry-eligible reasons:

    - ``"version_mismatch"`` — the caller's ``expected_version`` is *behind*
      the registry's current version (another writer committed first). Two
      concurrent OCC writers (both SHARED) are arbitrated here: the serialized
      transaction lets the first win and the second observes this.
    - ``"other_holder"`` — the version matched, but a *pessimistic* peer holds
      MODIFIED or EXCLUSIVE during the OCC compute window (OCC-vs-pessimistic
      coexistence guard). Not how two OCC writers are arbitrated.
    - ``"stale_read_generation"`` — the version matched and no peer holds M/E,
      but the committer's CAPTURED read_generation was superseded by a sweep
      reclamation — the read-generation fence (Piece #2). Only the generation
      catches this; version-CAS cannot (the version is unchanged on a
      no-successor reclaim). An absent read_generation is NOT this conflict
      (a plain OCC writer is arbitrated by version-CAS).

    Both are retry-eligible (re-read → recompute → retry). ``current_version``
    is the registry's authoritative version at the point the conflict was
    detected, so the caller can re-seed its retry. Corruption
    (``expected_version > current``) is signalled separately (it is never a
    ``ConflictDetail``) and the service layer raises ``CoherenceError`` for it.
    """

    reason: Literal["version_mismatch", "other_holder", "stale_read_generation"]
    current_version: int


@dataclass(frozen=True)
class CasCorruption:
    """Typed registry signal that an OCC compare-and-swap saw an impossible
    state: ``expected_version > current_version``.

    A correct single-coordinator system cannot produce this — an honest writer
    only ever observes a version ≤ the current one. It indicates corruption or
    a second coordinator writing the same store. Kept DISTINCT from
    ``ConflictDetail`` (a retry-eligible conflict) so the service layer can map
    it to a non-retryable ``CoherenceError`` (plan R2). The registry returns
    this sentinel rather than raising so all in-transaction outcomes are typed
    returns and the ``BEGIN IMMEDIATE`` region stays a single clean commit path.
    """

    current_version: int


@dataclass(frozen=True)
class MultiCommitResult:
    """The WIN aggregate of an atomic multi-artifact publish (SB-18 / commit_all).

    Returned (never raised) by the registry ``commit_all`` primitive when EVERY
    member of the write-set committed as one unit. ``versions`` maps each
    artifact id to its NEW (post-bump) version — the N-artifact analog of
    ``CasResult``'s single ``version``. ``invalidated`` maps each member artifact id
    to the peer agent ids whose cached views that member's commit invalidated — the
    caller (the service) builds one ``InvalidationSignal`` per (artifact, peer) and
    publishes them to the event bus AFTER the apply commits, never mid-batch
    (broadcast-after-commit). All-or-nothing: a ``MultiCommitResult`` means every
    member advanced; a partial batch is never a reachable outcome
    (``NoPartialPublish``, ``AtomicPublish.tla``). The registry return carries no
    ``coordinator_epoch`` — the service stamps that onto the wire response, exactly
    as the single-artifact CAS path does.
    """

    versions: Mapping[UUID, int]
    invalidated: Mapping[UUID, tuple[UUID, ...]]


@dataclass(frozen=True)
class MultiCommitConflict:
    """The all-or-nothing HELD aggregate of an atomic multi-artifact publish
    (SB-18 / commit_all).

    Returned (never raised) by the registry ``commit_all`` primitive when ANY
    member of the write-set was blocked, so ZERO members were mutated (the
    all-or-nothing bail). ``per_artifact`` names each FAILING member's own typed
    :class:`ConflictDetail` reason (``version_mismatch`` / ``other_holder`` /
    ``stale_read_generation``) independently — one held vector can fail different
    members for different reasons in a single call, so the aggregate carries a
    typed per-member reason map, never a flattened prose message (the
    typed-signal-not-substring house rule). The caller re-reads the named members
    at their ``current_version`` and re-publishes.

    Promoted from the frozen paper design's ``_PaperMultiCommitConflict``
    (``tests/test_sb18_non_foreclosure.py``) — same shape, composing the shipped
    ``ConflictDetail`` verbatim.
    """

    per_artifact: Mapping[UUID, ConflictDetail]


@dataclass(frozen=True)
class CommitAllEntry:
    """One member of an atomic multi-artifact publish write-set (SB-18 /
    commit_all). The per-artifact analog of ``commit_cas``'s keyword args —
    a single-shot version-checked comparand the caller supplies; the primitive
    NEVER re-reads or re-derives it (the split-comparand discipline).
    """

    expected_version: int
    content_hash: str
    size_tokens: int | None = None
    content: bytes | str | None = None


@dataclass(frozen=True)
class WorkspaceRestoreWrite:
    """One WRITTEN file member handed to the restore-registration step (WV
    plan Unit 5 / R4–R5).

    Hash-only by construction: the coordinator registration of a restore is
    METADATA — ``fingerprint`` (the captured sha-256 the leg re-imposed on the
    substrate) is the only content-shaped field, and no variant of this entry
    ever carries bytes. ``member_path`` doubles as the coordinator artifact
    NAME (``resolve_or_register``'s key), which is how the registration
    resolves — never mints — content-carrying identity.
    """

    member_path: str
    fingerprint: str


@dataclass(frozen=True)
class WorkspaceRegistrationResult:
    """The typed terminal answer of one restore-registration call (WV plan
    Unit 5 / R4–R5) — ``CoordinatorService.register_workspace_restore``.

    A typed RETURN, never an exception (the ``ConflictDetail`` discipline).
    ``status`` is one of the closed
    :data:`ccs.core.exceptions.WORKSPACE_REGISTRATION_STATUSES` (matched by
    identity):

    - ``committed`` — the ``commit_all`` batch landed all-or-nothing:
      ``versions`` maps each registered member path to its NEW coordinator
      version (strictly greater than the pre-registration version — restore is
      a forward commit carrying old bytes, never a decrement) and ``signals``
      carries one :class:`InvalidationSignal` per (member, invalidated peer)
      for broadcast-after-commit;
    - ``empty_write_set`` — nothing needed a commit (every write was already
      registered — ``skipped`` names the hash-identical members — or the
      caller handed no writes): **``commit_all`` was never called** (it raises
      on an empty write-set, by contract);
    - ``refused`` — the batch was HELD: NOTHING mutated (all-or-nothing) and
      ``refused`` names each failing member's own typed
      :class:`ConflictDetail` (``version_mismatch`` / ``other_holder`` /
      ``stale_read_generation`` — the last is the read-generation fence
      rejecting a superseded controller's late apply).

    ``skipped`` members were already registered at the manifest fingerprint
    (the artifact's current ``content_hash`` equals the write's fingerprint):
    the idempotency filter that makes crash-resumed registration exactly-once
    — a re-run of an already-landed registration re-answers without a second
    version bump.

    ``retry_of_own_registration`` (#191) is ``True`` when this controller had
    already claimed the checkpoint's registration before this call — the call
    is a retry of its OWN registration, so an ``empty_write_set`` with
    ``skipped`` members reads "I registered this", never "someone else did".
    A registration claimed by a DIFFERENT controller never reaches a result:
    it is refused ``already_registered`` (:class:`~ccs.core.exceptions.
    CheckpointRegistrationRefused`).
    """

    checkpoint_id: str
    status: str
    detail: str
    versions: Mapping[str, int] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()
    signals: tuple["InvalidationSignal", ...] = ()
    refused: Mapping[str, ConflictDetail] = field(default_factory=dict)
    retry_of_own_registration: bool = False


@dataclass(frozen=True)
class VersionedContent:
    """A successfully resolved retained version (plan item N v1, Unit 4 / R5).

    The WIN return of :meth:`CoordinatorService.read_at_version` — a typed
    return (the ``ConflictDetail`` discipline), never an exception. Carries the
    exact retained body and the forensic metadata a consumer needs:

    - ``content`` is the registry-committed body with its ORIGINAL Python type
      (``str`` for a TEXT row, ``bytes`` for a BLOB row — the sqlite
      affinity-NONE column round-trips by value; the in-memory record stores the
      body as-supplied). It is "the content the registry committed", not what
      reached any client's disk (the watchdog-ack honesty boundary).
    - ``captured_at`` is the wall-clock ``time.time()`` capture timestamp — kept
      on the success surface because forensic consumers need it (omitting it
      would bake a surface change into the first consumer's contract) and it is
      also the T-expiry reference.
    - ``coordinator_epoch`` is the store's epoch, stamped on every answer so a
      consumer can pin which store-incarnation served the bytes.

    The current version is never served here (``read_at_version`` rejects
    ``version == current`` with ``current_version``): this is a HISTORY surface.
    """

    artifact_id: UUID
    version: int
    content: str | bytes
    captured_at: float
    coordinator_epoch: str


@dataclass(frozen=True)
class VersionedReadRejection:
    """A typed read-at-version rejection (plan item N v1, Unit 4 / R5).

    The non-WIN return of :meth:`CoordinatorService.read_at_version` — a typed
    return (the ``ConflictDetail`` discipline), never an exception. ``reason`` is
    exactly one of the six wire-stable constants in
    :mod:`ccs.core.exceptions` (:data:`~ccs.core.exceptions.READ_AT_VERSION_REASONS`);
    consumers match with ``==`` against that set, never on a human message.

    **Carries NO content, NO content hash, NO body material** — only the reason,
    the requested/current versions, and the epoch. This is enforced by a test
    that pins the exact field set: a rejection is the one surface that must never
    leak bytes (e.g. an ``epoch_mismatch`` from a different store-incarnation, or
    a ``future_version`` that hints at a second coordinator), so the dataclass
    structurally cannot.

    Field semantics:

    - ``current_version`` is the registry's authoritative current version at the
      point the reason was decided, or ``None`` when it genuinely cannot be known
      (``unknown_artifact`` — there is no current version for a missing
      artifact). The single-scope read means a racing commit cannot mislabel it.
    - ``coordinator_epoch`` is the store's epoch — ALWAYS populated (``str``,
      never ``None``): the registry always has one, even for ``unknown_artifact``
      (the artifact may be missing; the store is not), so every rejection tells
      the consumer which store answered.
    """

    reason: str
    artifact_id: UUID
    requested_version: int
    current_version: int | None
    coordinator_epoch: str


@dataclass(frozen=True)
class SnapshotSession:
    """A consistent multi-artifact snapshot session (SB-17 / TX-1, Unit 2 / R1).

    The WIN return of :meth:`CoordinatorService.begin_session` — a frozen value
    object (the ``VersionedContent`` typed-return discipline) that pins a
    coherent CUT of the read-set: a ``{artifact_id: version}`` vector captured at
    ONE linearization point so no peer commit is *partially* visible within the
    set (no cross-artifact read skew). The capture is non-mutating (it mints no
    MESI grant and captures no ``read_generation`` — a reader is not an owner).

    **R11 BINDING CONSTRAINT — the cut is an INSPECTABLE per-artifact map, NOT
    an opaque handle.** ``cut`` exposes every pinned ``(artifact_id, version)``
    pair by design: the paper SB-18 atomic multi-publish signature
    (``commit_all(session_token, writes) -> MultiCommitResult``) is constructible
    from v1 *only* because each artifact's ``expected_version`` can be read out
    of this map. An opaque token that hid the per-artifact versions would leave
    SB-18 no public way to obtain the expected-versions vector → foreclosed.
    Unit 2 freezes this inspectable shape; Unit 9's harness re-checks it
    mechanically. Do NOT replace ``cut`` with an opaque handle.

    Field semantics:

    - ``session_token`` is the server-minted, owner-bound session identity
      (``secrets.token_urlsafe``; R13). It is NOT a snapshot handle — the cut is
      carried inspectably in ``cut`` — it identifies the SESSION (for later
      ``session.read`` / ``session.commit`` / heartbeat / release calls) and is
      bound to the creating caller's identity at mint time.
    - ``cut`` is the pinned version-vector ``{artifact_id: version}`` — the
      consistent cut, one entry per read-set member, captured atomically. This is
      the load-bearing inspectable surface (see the R11 note above).
    - ``coordinator_epoch`` is the store's epoch, stamped to mirror
      :class:`VersionedContent` / :class:`VersionedReadRejection` so a consumer
      can pin which store-incarnation captured the cut (a durable sqlite session
      survives restart only within the same epoch; an in-memory session does
      not survive at all — R6).
    - ``retain_versions`` is the deployment branch indicator: ``True`` when the
      store retains bodies durably (the LAZY serve branch — Unit 3 serves pinned
      bytes from ``artifact_versions``); ``False`` for the ``content=None`` /
      retention-off ICP (the EAGER serve branch is resolved at the serve layer
      in Unit 3, not here). Unit 2 records the branch; it captures the
      version-MAP only and does NOT capture bytes in the coordinator.
    """

    session_token: str
    cut: Mapping[UUID, int]
    coordinator_epoch: str
    retain_versions: bool


@dataclass(frozen=True)
class DataPlaneDeferredRead:
    """A pinned-version read the COORDINATOR cannot serve bytes for — the bytes
    live in the data plane (SB-17 / TX-1, Unit 3 / R2, the EAGER branch).

    The honest, typed signal of :meth:`CoordinatorService.session_read` when the
    session pins a real version but the coordinator holds NO body for it: the
    ``retain_versions=False`` / ``commit_cas(content=None)`` ICP, where bodies
    are never persisted in ``artifact_versions`` and the canonical bytes live in
    the CoherentVolume data plane. This is NOT an error and NOT a crash — it
    carries the pinned ``version`` + ``coordinator_epoch`` (+ ``content_hash`` if
    the registry knows it) so the caller can fetch the exact pinned bytes from
    the data plane. The actual data-plane byte serve is **Unit 6
    (CoherentVolume)** — this result is the boundary signal "coordinator pinned
    the version; ask the data plane for the bytes."

    **Carries NO body** by type (mirrors the :class:`VersionedReadRejection`
    no-leak discipline): only the pinned coordinates a data-plane fetch needs.
    The distinction from a rejection is deliberate — a rejection means "no valid
    pin"; this means "valid pin, bytes elsewhere".

    Field semantics:

    - ``artifact_id`` / ``version`` are the PINNED coordinates (the cut entry),
      not the current ones — the data plane must be asked for exactly this
      version's bytes.
    - ``content_hash`` is the pinned version's hash when the registry knows it
      (the artifact metadata carries a hash even when the body is not retained),
      else ``None`` — a hint the data-plane fetch can verify against; never a
      body.
    - ``coordinator_epoch`` is the store's epoch, stamped to mirror
      :class:`VersionedContent` / :class:`SnapshotSession` so the caller can pin
      which store-incarnation captured the cut.
    """

    artifact_id: UUID
    version: int
    content_hash: str | None
    coordinator_epoch: str


@dataclass(frozen=True)
class SessionReadRejection:
    """A typed :meth:`CoordinatorService.session_read` rejection (SB-17 / TX-1,
    Unit 3 / R2) — never a live-HEAD fall-through.

    The non-serve return when the call cannot be honored against a valid pin: a
    typed RETURN (the ``ConflictDetail`` / ``VersionedReadRejection`` discipline),
    never an exception and NEVER the current bytes served as if pinned. ``reason``
    is exactly one of the wire-stable constants in
    :data:`~ccs.core.exceptions.SESSION_READ_REASONS`; consumers match with
    ``==``, never on a human message.

    The reasons (Unit 5 ADDED the heartbeat-liveness ``session_invalidated`` axis):

    - ``session_not_found`` — the ``session_token`` has no pinned cut AND does not
      look like a server-minted token (a genuinely-never-opened / malformed
      token). Kept reachable additively as the clearly-bogus-token signal.
    - ``session_invalidated`` (Unit 5) — the pins are UNAVAILABLE for a token that
      was, or still structurally looks like, a real session: reaped by the
      session-liveness sweep (stale heartbeat), GC-raced, or wiped by an in-memory
      coordinator restart. Fails CLOSED here so a previously-valid token is NEVER
      served live HEAD as if still pinned (the post-restart-unknown safety case).
    - ``artifact_not_in_cut`` — the token IS a live session, but ``artifact_id``
      was not in its captured read-set. Reading an un-pinned artifact mid-session
      is out of scope (a session wanting fresh data starts a new session); it is
      rejected here, NOT served from live HEAD.

    **Carries NO body** by type (the no-leak discipline): only the reason, the
    artifact id, and the epoch. ``coordinator_epoch`` is ALWAYS populated.
    """

    reason: str
    artifact_id: UUID
    coordinator_epoch: str


@dataclass(frozen=True)
class SessionCommitRejection:
    """A typed :meth:`CoordinatorService.session_commit` VALIDATION rejection
    (SB-17 / TX-1, Unit 4 / R3) — never a silent fall-through.

    Returned (never raised) when the commit cannot even be ATTEMPTED against a
    valid pin: the token has no live cut, or the artifact was not in the captured
    read-set. It is the pre-commit gate ONLY — it is NOT how an OCC outcome is
    surfaced. The OCC taxonomy is preserved byte-for-byte from the shipped
    ``commit_cas``:

    - a lost race → the shipped :class:`ConflictDetail` (``version_mismatch`` /
      ``other_holder`` / ``stale_read_generation``), returned UNCHANGED;
    - corruption (``expected_version > current``) → a RAISED ``CoherenceError``
      (the service maps the registry's :class:`CasCorruption` sentinel), never a
      rejection.

    ``reason`` is exactly one of :data:`~ccs.core.exceptions.SESSION_COMMIT_REASONS`
    (the SAME token/pin vocabulary as :class:`SessionReadRejection`); consumers
    match ``reason == CONSTANT``, never on a human message.

    - ``session_not_found`` — the ``session_token`` has no pinned cut (unknown,
      never opened, or released; an in-memory restart also lands here until the
      Unit-5 liveness taxonomy splits it).
    - ``artifact_not_in_cut`` — the token IS a live session, but ``artifact_id``
      was not in its captured read-set. Committing an un-pinned artifact is out
      of scope (a session commits only against what it pinned).

    **Carries NO body** by type (the no-leak discipline shared with
    :class:`SessionReadRejection`): only the reason, the artifact id, and the
    epoch. ``coordinator_epoch`` is ALWAYS populated.
    """

    reason: str
    artifact_id: UUID
    coordinator_epoch: str


@dataclass(frozen=True)
class EffectHeld:
    """The effect-gate RE-VALIDATE step found the read-set moved — the effect was
    NOT fired (SB-17 / TX-1, Unit 6 / EO-5).

    The non-fire return of :meth:`CoordinatorService.effect_gate`: between
    ``begin_session`` (the pin) and the effect boundary, at least one read-set
    member's CURRENT version no longer equals its pinned version, so firing would
    act on stale input. The gate HOLDS — it never fires an escaping effect and
    never lets the atomic commit land — and returns this typed value. NEVER an
    exception (the ``ConflictDetail`` discipline), and NEVER a partial fire.

    Recovery is to open a NEW session, re-read the fresh cut, re-decide, and
    re-gate — exactly like a ``version_mismatch`` HELD; the dead decision is not
    retry-eligible in place (its inputs changed underneath it).

    Field semantics:

    - ``moved`` is the per-artifact drift map ``{artifact_id: (pinned, current)}``
      for EVERY read-set member whose current version diverged from its pin — the
      full set, not just the first, so the caller can see the whole blast radius
      of the change it raced. ``current`` is ``None`` when the artifact vanished
      under the live pin (a deleted/GC-raced member is also "moved" — it can no
      longer be proven unchanged, so the gate holds fail-closed).
    - ``conflict`` is populated ONLY for the ATOMIC (``session.commit``) mode when
      the commit lost its OCC race AT the pinned base — the shipped
      :class:`ConflictDetail`, carried through verbatim so the caller sees the
      same taxonomy a bare ``session_commit`` would surface. It is ``None`` for
      an escaping-effect HELD (no commit was attempted) and for an atomic HELD
      caught at the pre-commit re-validate (``moved`` populated, no CAS attempted).
    - ``coordinator_epoch`` mirrors the other session results so the caller can
      pin which store-incarnation evaluated the gate.
    """

    moved: Mapping[UUID, "tuple[int, int | None]"]
    coordinator_epoch: str
    conflict: Optional["ConflictDetail"] = None


@dataclass(frozen=True)
class EffectFired:
    """The effect-gate fired the effect — the read-set was unchanged AS OF the
    re-validate point (SB-17 / TX-1, Unit 6 / EO-5).

    The fire return of :meth:`CoordinatorService.effect_gate`. Its guarantee is
    mode-dependent and stated honestly (EO-5 / EO-7):

    - **ATOMIC mode** (the effect IS ``session.commit``): the commit rode the
      shipped ``commit_cas`` at the pinned base in the SAME arbitration step, so
      "unchanged" and "fire" are one atomic event — there is NO window. ``commit``
      carries the ``(updated_artifact, signals)`` WIN tuple.
    - **ESCAPING mode** (the effect is a non-commit side effect — deploy / charge
      / click — a caller-supplied callable): the gate re-validated the whole
      vector and THEN fired the callable. The guarantee is "the read-set was
      unchanged **as of the re-validate point**", NOT "as of the fire point": a
      peer can commit a read-set member in the residual RE-VALIDATE→FIRE window,
      after the check passed but before the callable ran. The gate gates pre-fire
      and NEVER rolls back (EO-7), so this window is unclosable for escaping
      effects and is NOT claimed away. ``result`` carries whatever the callable
      returned (or ``None``).

    Field semantics:

    - ``revalidated_cut`` is the per-artifact version map ``{artifact_id: version}``
      proven equal to the pin at the re-validate point — the exact vector the fire
      decision was justified against.
    - ``commit`` is the ``session.commit`` WIN tuple in ATOMIC mode, else ``None``.
    - ``result`` is the escaping callable's return value in ESCAPING mode, else
      ``None``.
    - ``coordinator_epoch`` mirrors the other session results.
    """

    revalidated_cut: Mapping[UUID, int]
    coordinator_epoch: str
    commit: Optional["tuple[Artifact, list[InvalidationSignal]]"] = None
    result: object = None


@dataclass(frozen=True)
class InvalidationSignal:
    """Lightweight invalidation signal sent to agents."""

    artifact_id: UUID
    new_version: int
    issued_at_tick: int
    issuer_agent_id: UUID


@dataclass(frozen=True)
class FetchRequest:
    """Request to fetch canonical artifact content/version."""

    artifact_id: UUID
    requesting_agent_id: UUID
    requested_at_tick: int


@dataclass(frozen=True)
class FetchResponse:
    """Fetch response containing granted state and content payload."""

    artifact_id: UUID
    version: int
    content: str
    state_grant: MESIState


@dataclass(frozen=True, kw_only=True)
class FenceComparands:
    """Everything the effect fence classifies, as ONE named, frozen value.

    The input to :func:`ccs.core.fence.classify_hold` — the branch table that
    decides whether an irreversible effect may still fire. Both surfaces that
    ask that question build one of these: the in-process wrapper
    (``adapters.effect_gate``) and the coordinator route that answers the same
    question over HTTP. One object, not six positionals: two ints, two optional
    ints and two bools in a row is a transposition hazard, and this is the one
    path in the product where a silently swapped pair IS the lost update the
    fence exists to prevent. ``kw_only`` makes the swap unexpressible rather
    than merely unlikely, and NO field carries a default — a caller that
    forgets the grant state must fail at construction, naming the field it
    omitted, instead of silently gating on a fabricated "nothing was wrong".

    Field names come from the PROTOCOL, not from either caller's internals, so
    core never acquires volume-shaped vocabulary and the adapter stays the only
    place a private ``_last_read_*`` attribute is read:

    - ``expected_version`` / ``current_version`` — the value comparand:
      "is this still the version the decision was derived from". ``0`` is the
      "could not resolve" SENTINEL on either side (a degraded or pre-fence
      coordinator), never a comparable value; ``current_version is None`` means
      the input VANISHED between capture and re-validate.
    - ``expected_generation`` / ``current_generation`` — the authority
      comparand: "is the grant it was read under still standing". ``None`` is
      the sentinel for "the coordinator confirmed no generation" and is NEVER
      coerced to ``0``, which is a REAL generation. A sweep reclamation moves
      this pair while the version stays put — the drift a version-only check
      structurally cannot see.
    - ``read_refused`` — the coordinator REFUSED the re-validate read
      (strict-mode deny). A distinct, recoverable answer, and on the strict
      path it is how a sweep reclaim actually reaches a client.
    - ``grant_did_not_stand`` — the re-validate read was served WITHOUT a
      standing grant. A peer's pessimistic write-acquire preempts a holder
      while moving NEITHER comparand (no commit yet, and ``trigger="write"``
      is outside ``EPOCH_BUMP_TRIGGERS``), so this is the only leg that sees it.
    - ``content_claim_present`` — the coordinator records SOME content claim
      for this artifact (see
      :func:`ccs.core.fence.coordinator_holds_content_claim`). ``False`` means
      it holds no hash at all and therefore cannot vouch that the bytes in hand
      are the content at the version it reports.
    """

    expected_version: int
    current_version: int | None
    expected_generation: int | None
    current_generation: int | None
    read_refused: bool
    grant_did_not_stand: bool
    content_claim_present: bool

