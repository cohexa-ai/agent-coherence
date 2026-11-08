---------------------------- MODULE Retention ----------------------------
(* Bounded version-retention amendment to the fenced MESI protocol.
   Proves that retaining a K-bounded history of committed versions, and
   serving read-at-version requests from it, changes NOTHING about the
   coordinator protocol: every inherited safety invariant (single-writer,
   monotonic versioning, the crash-recovery I3-I6 family, NoStaleApply)
   is re-checked with retention enabled, and a versioned read is proven
   to be a protocol no-op.

   Plan: docs/plans/2026-06-10-001-feat-version-retention-read-at-version-plan.md
   (Unit 1). Origin: the 2026-06-10 version-retention requirements brief
   (docs/brainstorms/).
   Models R4 (GC invisible, current never collectible), R6 (fence
   non-capture), R7 (off-protocol read), R8 (this spec), and the atomic
   half of R3 (capture happens atomically at commit).

   Adds:
     - history       : per-artifact map version -> retained content marker.
                       Content is abstracted as the version number itself:
                       within a behavior the content at (artifact, version)
                       is unique (monotonic versions, one committer per
                       version), so the marker is distinct per version yet
                       deterministic. A richer marker (e.g. the writer id)
                       would only add a decorative state-space dimension --
                       no checked property reads the marker value -- and
                       measured against the Fencing_CI budget that dimension
                       is what blows the CI gate. Real byte payloads are
                       out of the question for the same reason.
     - collectedRead : a sticky history flag -- TRUE iff a versioned read
                       ever observed a hole inside the K-window strictly
                       below the current version, i.e. the service would
                       have had to answer for a version the bounded-
                       retention contract promises is servable but which
                       was collected (or never retained). Mirrors the
                       staleApply / lostUpdate idiom: the flag flips only
                       when the modeled implementation would have done the
                       wrong thing; the correct spec never sets it.
     - MaxRetained   : the K bound. K counts retained rows INCLUDING the
                       current version's row; the current version is never
                       collectible (R4). Mirrors RetentionPolicy.max_versions.
     - RetentionCommitAction : FencingCommitAction + atomic retain + inline
                       K-GC (see crux below).
     - VersionedReadAction   : a read-at-version request -- a pure protocol
                       no-op (see below).

   KEY MODELING DECISION (the crux). Fencing's crux is that a SINGLE
   FencingCommitAction models BOTH the pessimistic commit and the OCC
   commit_cas, because the fence guard is uniform across both paths. The
   retention capture inherits that uniformity: the implementation retains
   inside the SAME sqlite transaction / same GIL-atomic apply step as the
   commit it captures, on every capture point. So Retention REPLACES the
   one FencingCommitAction with RetentionCommitAction: identical guard,
   identical protocol effect, PLUS extend history with the new version and
   apply inline K-GC -- commit + retain + GC are ONE action, atomic by
   construction. Per-capture-point coverage (pessimistic vs commit_cas vs
   register_artifact) is discharged by the Python parity suite, not TLC.
   Mutant recipe 5 (README) splits the retain from the bump into two
   separately-interleavable actions -- the exact crash window the
   same-transaction discipline excludes -- and TLC then finds the
   NoCollectedRead violation.

   THE READ IS A PROTOCOL NO-OP -- and how that is checked. The
   implementation's read_at_version never calls set_agent_state /
   set_agent_transient: no MESI transition, no invalidation membership,
   no read_generation capture (R6/R7 hold by construction there).
   VersionedReadAction models this: it leaves EVERY protocol variable
   unchanged (base MESI vars, crash-recovery vars, ownerGeneration,
   readGeneration, staleApply). Structural UNCHANGED alone is not
   checkable evidence, though: a mutated read that refreshes the reader's
   readGeneration produces transitions extensionally identical to a
   legitimate ObserveGenAction, so no state invariant (NoStaleApply
   included) can see it. The no-op claim is therefore checked as an
   explicit TLC ACTION PROPERTY:

       ReadAtVersionIsProtocolNoOp ==
           [][VersionedReadAction => UNCHANGED fenceVars]_retentionVars

   TLC evaluates [][A]_v action properties on every explored transition
   (cheap, no liveness graph), so mutant recipe 6 -- the fence-refreshing
   read -- has a defined failure signal: the property fires even though
   NoStaleApply stays green. The alternative encoding (a ghost variable
   capturing pre/post fence-state equality) was rejected because it adds a
   state dimension; the action property adds none.

   MODELING SCOPE OF THE READ. The reader carries NO agent binding: on
   this path the requester has no protocol identity (R7) -- the read's
   effect (none) is the same for every agent. Mutant recipe 6 reintroduces
   an agent binder to express the fence-refresh bug. The requested version
   ranges over ServableHistory(art): the K-window STRICTLY below the
   current version -- the only requests whose answer depends on retention
   state. Requests outside it -- at the current version (R5
   current_version rejection), below the window (honest not_retained for
   a legitimately collected version), or above the current version
   (future_version) -- are structurally identical typed rejections with
   the same empty effect and can never flip the flag even in mutants;
   enumerating them would only add stutter transitions and TLC time, so
   the Python request-surface tests (plan Unit 4) own them.

   K=1 CAVEAT. With MaxRetained = 1 the window strictly below the current
   version is empty, so the read action is never enabled, collectedRead
   can never flip, and the flag has no teeth (the GC also never has a
   non-current row to drop). Both cfgs therefore use MaxRetained = 2,
   which keeps K-eviction reachable inside the version bound AND gives
   mutants 5/7/8 a violation to find. The ASSUME below only enforces the
   implementation floor (max_versions >= 1, RetentionPolicy.__post_init__).

   DELIBERATELY NOT MODELED (one-line whys; README out-of-scope table):
     (a) artifact delete -- Artifacts is a CONSTANT throughout the module
         chain; delete-drops-history (cascade) is owned by Python tests.
     (b) register_artifact -- no register action exists in the chain; the
         initial state retains version 1, covering the trivial case.
     (c) retention-policy changes across runs -- MaxRetained is a per-run
         CONSTANT; a re-opened store is equivalent to a fresh bounded store.
     (d) coordinator restart / durability -- inherited exclusion from
         Fencing (in TLA the state IS the durable truth); restart-survival
         of retained rows is an implementation/test property (replay
         resolver, plan Unit 6).

   KNOWN SIBLING GAP (do not over-claim): the implementation's
   trigger="timeout" transient eviction does NOT bump owner_generation;
   only the crash-recovery sweep triggers (heartbeat / max_hold) do. This
   spec models exactly the sweep that Fencing models and claims no
   eviction coverage beyond it.

   WHAT IS PROVEN: safety PRESERVATION -- the inherited invariants still
   hold with the retention machinery composed in, plus NoCollectedRead and
   the read-no-op action property. NOT proven: behavioral equivalence to
   Fencing (that would need a refinement mapping; deliberately out of
   scope). LIVENESS is discharged as safety + prose per the repo's
   safety-only TLC convention (README). *)

EXTENDS Fencing

CONSTANT MaxRetained

(* Implementation floor: RetentionPolicy(max_versions < 1) is a ValueError. *)
ASSUME MaxRetained >= 1

VARIABLES history, collectedRead

retentionVars == <<fenceVars, history, collectedRead>>

--------------------------------------------------------------------
(* Retention helpers *)
--------------------------------------------------------------------

(* The versions the K-bound contract promises are servable: the newest
   MaxRetained versions, current included (K counts the current row).
   In the correct spec DOMAIN history[art] equals this window at every
   reachable state -- every commit retains its version and the GC keeps
   exactly the newest K rows. *)
MustRetain(art) ==
    LET lo == IF version[art] > MaxRetained
              THEN version[art] - MaxRetained + 1
              ELSE 1
    IN lo..version[art]

(* The load-bearing request range for the modeled read: the K-window
   STRICTLY below the current version. Requests at the current version
   (R5 current_version rejection) and below the window (honest
   not_retained for a legitimately collected version) are structurally
   identical empty-effect rejections that can never flip the flag -- even
   in mutants -- so enumerating them would only add stutter transitions
   and TLC time; the Python request-surface tests (plan Unit 4) own them. *)
ServableHistory(art) ==
    LET lo == IF version[art] > MaxRetained
              THEN version[art] - MaxRetained + 1
              ELSE 1
    IN lo..(version[art] - 1)

(* Retain newVer and apply inline K-GC in one step: extend the map, then
   drop the oldest row iff the row count exceeds MaxRetained (row-count
   semantics, matching collectible_versions in the implementation). A
   single drop suffices: every extend adds exactly one row to a map the
   GC already bounded at MaxRetained, so the domain never exceeds
   MaxRetained + 1 in any reachable state -- including under the README
   mutants, whose retain steps also GC in-step. The drop selects the
   OLDEST row, so with MaxRetained >= 1 the GC can NEVER drop the newest
   (current) version -- R4's "current never collectible" holds by
   construction. Mutant recipe 7 (README) flips the selection to the
   newest row, making the current version collectible. *)
RetainAndCollect(art, newVer) ==
    LET extended == [v \in (DOMAIN history[art]) \cup {newVer} |->
                        IF v = newVer THEN newVer ELSE history[art][v]]
        dom == DOMAIN extended
        oldest == CHOOSE m \in dom : \A w \in dom : m <= w
        keep == IF Cardinality(dom) <= MaxRetained THEN dom ELSE dom \ {oldest}
    IN [history EXCEPT ![art] = [v \in keep |-> extended[v]]]

(* The wrong thing the sticky flag records: a request for a version the
   contract promises (inside the K-window, strictly below current -- the
   current version is answered by the protocol path / current_version
   rejection, never by history) whose row is absent -- collected or never
   retained. The correct spec keeps DOMAIN history[art] = MustRetain(art),
   so this is unreachable-TRUE; mutants 5/7/8 make it reachable. *)
WouldServeCollected(art, v) ==
    /\ v < version[art]
    /\ v \in MustRetain(art)
    /\ v \notin DOMAIN history[art]

--------------------------------------------------------------------
(* Initialization *)
--------------------------------------------------------------------

(* Version 1 exists at init (MESI Init); its retained row models the
   register_artifact capture -- the trivial case of the unmodeled
   register action (see header note (b)). *)
RetentionInit ==
    /\ FencingInit
    /\ history = [art \in Artifacts |-> [v \in {1} |-> 1]]
    /\ collectedRead = FALSE

--------------------------------------------------------------------
(* RetentionCommitAction: FencingCommitAction + atomic retain + K-GC.
   Guard and protocol effect are IDENTICAL to Fencing's commit (equality
   admits, present-but-superseded is a clean no-op, an absent operand admits
   -- version-CAS arbitrates it, not the fence); the admitted branches (absent
   and WIN) additionally extend history with the new version and collect beyond
   K -- one atomic action, the same-transaction discipline. Replaces the
   inherited FencingCommitAction (the crux above). *)
--------------------------------------------------------------------

RetentionCommitAction ==
    \E ag \in Agents, art \in Artifacts :
        /\ version[art] < MaxVersion                (* finite bound *)
        /\ LET rg == readGeneration[ag][art]
               og == ownerGeneration[art]
               newVer == version[art] + 1
           IN \/ (* ADMIT (absent operand): a plain OCC writer with no fence
                    claim -- version-CAS (OCC.tla) arbitrates, not the fence.
                    Applies AND retains atomically (same-transaction
                    discipline), exactly like WIN; no captured generation, so
                    staleApply cannot be implicated. *)
                 /\ rg = None
                 /\ version' = [version EXCEPT ![art] = newVer]
                 /\ history' = RetainAndCollect(art, newVer)
                 /\ UNCHANGED staleApply
              \/ (* WIN: claim present and current -> apply the write AND retain
                    it, atomically. Version bump, history extend, and K-GC are
                    one step -- mutant recipe 5 splits them and TLC finds
                    the crash-window NoCollectedRead violation. *)
                 /\ rg /= None
                 /\ rg = og
                 /\ version' = [version EXCEPT ![art] = newVer]
                 /\ history' = RetainAndCollect(art, newVer)
                 (* Same teeth as Fencing: the WIN guard `rg = og` makes
                    `rg < og` always FALSE here; removing the guard (the
                    Fencing mutant) lets a superseded commit win and TLC
                    reports a NoStaleApply violation. *)
                 /\ staleApply' = (staleApply \/ (rg < og))
              \/ (* CONFLICT: present but superseded (rg < og, reclaimed since
                    the claim) -- clean no-op: no version bump, no mutation, and
                    NO capture (rejected writes leave history unchanged). *)
                 /\ rg /= None
                 /\ ~(rg = og)
                 /\ UNCHANGED <<version, staleApply, history>>
        /\ UNCHANGED <<clock, mesiState, lastHeartbeat, grantedAtTick,
                       lastReclamation, ownerGeneration, readGeneration,
                       silentRevoke, collectedRead>>

--------------------------------------------------------------------
(* VersionedReadAction: a read-at-version request. A pure protocol no-op:
   every protocol variable is listed UNCHANGED (the action property below
   re-checks this against the action, not just its syntax). The only
   writable slot is the sticky bug flag, and the correct spec never sets
   it (WouldServeCollected is unreachable-TRUE). No reader binding: the
   requester has no protocol identity on this path (R7). *)
--------------------------------------------------------------------

(* The UNCHANGED tuple is deliberately hoisted OUTSIDE the quantifiers
   and ordered by how often each variable changes (mesiState first, not
   declaration order): the action property below re-evaluates this
   formula on every explored transition, and protocol transitions must
   falsify it on the first compared variable without ever enumerating
   the window (TLC evaluates conjuncts in order). Semantically identical
   to the nested, declaration-ordered form. *)
VersionedReadAction ==
    /\ UNCHANGED <<mesiState, readGeneration, version, lastHeartbeat,
                   clock, grantedAtTick, lastReclamation, ownerGeneration,
                   staleApply, silentRevoke, history>>
    /\ \E art \in Artifacts :
         \E v \in ServableHistory(art) :
           collectedRead' = (collectedRead \/ WouldServeCollected(art, v))

--------------------------------------------------------------------
(* Specification *)
--------------------------------------------------------------------

(* Inherited actions keep the retention variables unchanged. The fence
   wrapping mirrors Fencing's own Next (FencingWriteAction captures the
   read-generation at the E/M acquire); FencingCommitAction is DELIBERATELY
   replaced by RetentionCommitAction (the crux modeling decision above) --
   in the retention world every version-bumping commit retains atomically. *)
RetentionNext ==
    \/ (CRFetchAction      /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply, silentRevoke, history, collectedRead>>)
    \/ (FencingWriteAction /\ UNCHANGED <<history, collectedRead>>)
    \/ (FencingInvalidateAction /\ UNCHANGED <<history, collectedRead>>)
    \/ (FencingReleaseToReaderAction /\ UNCHANGED <<history, collectedRead>>)
    \/ (CRTickAction       /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply, silentRevoke, history, collectedRead>>)
    \/ (HeartbeatAction    /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply, silentRevoke, history, collectedRead>>)
    \/ (FencingSweepAction /\ UNCHANGED <<history, collectedRead>>)
    \/ (ObserveGenAction   /\ UNCHANGED <<history, collectedRead>>)
    \/ RetentionCommitAction
    \/ VersionedReadAction

RetentionSpec == RetentionInit /\ [][RetentionNext]_retentionVars

--------------------------------------------------------------------
(* Invariants *)
--------------------------------------------------------------------

RetentionTypeOK ==
    /\ FencingTypeOK
    /\ collectedRead \in BOOLEAN
    /\ \A art \in Artifacts :
         /\ DOMAIN history[art] \subseteq (1..MaxVersion)
         /\ Cardinality(DOMAIN history[art]) <= MaxRetained
         (* Pins the content abstraction: the marker IS the version. *)
         /\ \A v \in DOMAIN history[art] : history[art][v] = v

(* The headline retention property: no versioned read ever observed a
   collected (or never-retained) version inside the promised K-window.
   NoStaleApply, SingleWriter, MonotonicVersion, and the CR invariants
   (I3-I6) are inherited and re-checked to validate composition --
   safety preservation, not behavioral equivalence (header note). *)
NoCollectedRead == collectedRead = FALSE

(* The read-no-op ACTION PROPERTY (cfg: PROPERTY, not INVARIANT): any
   transition satisfying VersionedReadAction changes no fence/MESI/CR
   variable. This is what catches a fence-refreshing read (mutant 6),
   which no state invariant can see -- a refreshed claim is
   indistinguishable from a legitimate ObserveGenAction. *)
ReadAtVersionIsProtocolNoOp ==
    [][VersionedReadAction => UNCHANGED fenceVars]_retentionVars

==========================================================================
