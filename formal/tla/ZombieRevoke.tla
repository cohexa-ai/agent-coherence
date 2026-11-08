-------------------------- MODULE ZombieRevoke --------------------------
(* NoZombieRevoke -- the pin on the REVOKE, the dual of the fence on the
   commit. Sibling amendment to Fencing.tla (which it EXTENDS).

   Fencing.tla proves that a superseded CLAIM cannot write (NoStaleApply). This
   module proves the other half: a superseded SIGNAL cannot revoke. Neither
   subsumes the other -- they guard opposite ends of the same asymmetry.

   THE HAZARD (a shipped-code defect this amendment was written against). An invalidation is minted inside the coordinator's registry lock
   and applied later, outside it. Between mint and apply the target may be
   reclaimed by the sweep, re-acquire, and re-read. Applying the stale signal
   then revokes a grant established AFTER the signal was issued -- and neither
   guard on the commit side can see it: the target's version did not move, so
   version-CAS is blind, and the fence only ever looks at committers. Temporal
   hit this shape in their semaphore pattern and closed it with signal pinning,
   "a late release Signal from a holder whose lease already expired could revoke
   a subsequent acquirer's permit".

   WHY A SEPARATE MODULE. The pin needs a per-(agent, artifact) observation
   variable, and Fencing is EXTENDed by EffectGate, Retention and Snapshot --
   all of which would pay that state-space cost (measured: EffectGate 1m40s ->
   5m24s) for a property none of them is about. Fencing keeps the parts that
   add no variable (the epoch bump on a release, NoSilentRevoke); the pin lives
   here.

   Adds:
     - observedCurrent : one bit per (agent, artifact) -- TRUE iff the agent has
                         observed the CURRENT version. The abstraction of the
                         registry's last_observed_version (SB-10 R6/R7); see
                         ObservedCurrentNext for why one bit suffices.
     - zombieRevoke    : a sticky history flag -- TRUE iff any peer-issued
                         invalidation ever revoked a write claim it was not
                         authority over.
     - ZRInvalidateAction : FencingInvalidateAction PLUS an issuer and the pin.
                         Replaces it in the next-state relation.

   LIVENESS is discharged as safety + prose, per the repo's safety-only TLC
   convention. TLC checks NoZombieRevoke (plus everything inherited). *)

EXTENDS Fencing

VARIABLES observedCurrent, zombieRevoke

zrVars == <<fenceVars, observedCurrent, zombieRevoke>>

--------------------------------------------------------------------
(* ObservedCurrentNext: the abstraction of the registry's
   last_observed_version. Tracking the exact observed version would multiply the
   state space by |versions|^|Agents|; the pin only ever asks "is this agent's
   copy behind the version the signal announces?", and a signal is minted at
   some version <= the current one. So one bit per (agent, artifact) suffices:
   TRUE means the agent observed the CURRENT version -- no signal can be ahead
   of it -- and FALSE means it may be behind, which is the direction the pin
   resolves toward APPLYING. That asymmetry is deliberate and matches the
   implementation: wrongly DROPPING an invalidation would leave a genuinely
   stale copy marked valid, which is the stale-read -> write hole the whole
   layer exists to close, so the pin drops only what it can prove obsolete.

   Derived from unprimed vs primed state, never from an action's existential
   binding (the UpdatedGrantedAtTick discipline): an agent observes when it
   ENTERS a non-INVALID state, and every observation except the committer's own
   goes stale when the version moves. An OCC committer that stays S/I is
   conservatively marked stale too.

   ORDERING: reads mesiState' and version', and TLC evaluates conjuncts in
   textual order -- so every use must sit AFTER the clauses that bind both,
   including the UNCHANGED tuples that bind them. Each use below is last in its
   disjunct for exactly that reason. *)
--------------------------------------------------------------------

ObservedCurrentNext ==
    [ag \in Agents |-> [art \in Artifacts |->
        IF mesiState[art][ag] = "I" /\ mesiState'[art][ag] /= "I"
        THEN TRUE
        ELSE IF version'[art] /= version[art]
        THEN mesiState'[art][ag] \in MorE
        ELSE observedCurrent[ag][art]]]

--------------------------------------------------------------------
(* Initialization *)
--------------------------------------------------------------------

ZRInit ==
    /\ FencingInit
    /\ observedCurrent = [ag \in Agents |-> [art \in Artifacts |-> FALSE]]
    /\ zombieRevoke = FALSE

--------------------------------------------------------------------
(* ZRInvalidateAction: FencingInvalidateAction (the release + the epoch bump)
   PLUS an issuer and the pin. Written out rather than conjoined because the
   pin has to inspect the TARGET, which is bound inside CRInvalidateAction's
   existential -- the same reason Fencing writes out FencingSweepAction rather
   than conjoining SweepAction. *)
--------------------------------------------------------------------

ZRInvalidateAction ==
    \E ag \in Agents, art \in Artifacts, iss \in Agents :
        /\ mesiState[art][ag] /= "I"
        (* THE PIN. `iss` is the issuer; a signal is minted at some version <=
           the current one and applied here, arbitrarily later. A PEER-issued
           signal may land only on a target that is BEHIND what it announces --
           observedCurrent = FALSE. A target that has observed the current
           version cannot be behind any signal, so no signal is authority over
           ITS claim. A self-issued release (iss = ag: an agent handing back its
           own claim to INVALID, such as an operator drain) is not a
           cross-agent revoke and is never pinned. The holder's own release at
           the end of a turn, when a subagent stops, or on a failed edit keeps
           a read instead; that is FencingReleaseToReaderAction, also
           self-issued and never pinned (ZRNext). Removing the
           ~observedCurrent disjunct is the mutant (recipe 16): it re-enables
           the zombie branch below and TLC reports NoZombieRevoke. *)
        /\ \/ iss = ag
           \/ ~observedCurrent[ag][art]
        /\ zombieRevoke' = (zombieRevoke
                            \/ (iss /= ag
                                /\ mesiState[art][ag] \in MorE
                                /\ observedCurrent[ag][art]))
        /\ mesiState' = [mesiState EXCEPT ![art][ag] = "I"]
        /\ version' = version
        /\ \A a2 \in RevokedMorE : ownerGeneration[a2] < MaxGen   (* finite bound *)
        /\ ownerGeneration' = [a2 \in Artifacts |->
             IF a2 \in RevokedMorE THEN ownerGeneration[a2] + 1
             ELSE ownerGeneration[a2]]
        /\ silentRevoke' = (silentRevoke
                           \/ (\E a2 \in RevokedMorE :
                                 ownerGeneration'[a2] = ownerGeneration[a2]))
        /\ grantedAtTick'   = UpdatedGrantedAtTick
        /\ lastReclamation' = UpdatedLastReclamation
        /\ UNCHANGED <<clock, lastHeartbeat, readGeneration, staleApply>>
        /\ observedCurrent' = ObservedCurrentNext

--------------------------------------------------------------------
(* Specification. Mirrors FencingNext with ZRInvalidateAction substituted for
   FencingInvalidateAction, and observedCurrent maintained on every step that
   moves a grant or the version. The release that keeps a read is the
   holder's own act on its own row, so it has no issuer to pin and can never
   set zombieRevoke; it leaves observedCurrent as it was, since the holder
   neither reads nor moves the version. *)
--------------------------------------------------------------------

ZRNext ==
    \/ (CRFetchAction      /\ UNCHANGED <<ownerGeneration, readGeneration,
                                          staleApply, silentRevoke, zombieRevoke>>
                           /\ observedCurrent' = ObservedCurrentNext)
    \/ (FencingWriteAction /\ UNCHANGED zombieRevoke
                           /\ observedCurrent' = ObservedCurrentNext)
    \/ ZRInvalidateAction
    \/ (FencingReleaseToReaderAction /\ UNCHANGED zombieRevoke
                                     /\ observedCurrent' = ObservedCurrentNext)
    \/ (CRTickAction       /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply,
                                          silentRevoke, observedCurrent, zombieRevoke>>)
    \/ (HeartbeatAction    /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply,
                                          silentRevoke, observedCurrent, zombieRevoke>>)
    \/ (FencingSweepAction /\ UNCHANGED zombieRevoke
                           /\ observedCurrent' = ObservedCurrentNext)
    \/ (ObserveGenAction   /\ UNCHANGED <<observedCurrent, zombieRevoke>>)
    \/ (FencingCommitAction /\ UNCHANGED zombieRevoke
                            /\ observedCurrent' = ObservedCurrentNext)

ZRSpec == ZRInit /\ [][ZRNext]_zrVars

--------------------------------------------------------------------
(* Invariants *)
--------------------------------------------------------------------

ZRTypeOK ==
    /\ FencingTypeOK
    /\ zombieRevoke \in BOOLEAN
    /\ \A ag \in Agents, art \in Artifacts :
         observedCurrent[ag][art] \in BOOLEAN

(* The headline property. No invalidation ever revoked a write claim it was not
   authority over -- a signal minted at one epoch and applied to a claim
   established later. Scoped to a PEER-issued revoke landing on an M/E holder
   whose observation is current: a self-release is the holder's own choice, and
   a target already behind the announcement is exactly who the invalidation is
   FOR. The whole inherited invariant set is re-checked alongside, to validate
   that pinning the revoke breaks nothing the fence proved. *)
NoZombieRevoke == zombieRevoke = FALSE

==========================================================================
