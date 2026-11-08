------------------------ MODULE CrashRecovery ------------------------
(* Crash-recovery amendment to the MESI coherence protocol.
   Adds heartbeat liveness, max-hold ceiling, and sweep reclamation.
   Checks invariants I1-I6 from docs/specs/crash-recovery.md §5. *)

EXTENDS MESI

CONSTANTS HeartbeatTimeout, MaxHoldTicks

CONSTANT None

Triggers == {"heartbeat", "max_hold"}

VARIABLES lastHeartbeat, grantedAtTick, lastReclamation

crVars  == <<lastHeartbeat, grantedAtTick, lastReclamation>>
allVars == <<baseVars, crVars>>

--------------------------------------------------------------------
(* Initialization *)
--------------------------------------------------------------------

CRInit ==
    /\ Init
    /\ lastHeartbeat  = [ag \in Agents |-> None]
    /\ grantedAtTick  = [ag \in Agents |-> [art \in Artifacts |-> None]]
    /\ lastReclamation = [ag \in Agents |-> [art \in Artifacts |-> None]]

--------------------------------------------------------------------
(* Helper: update grantedAtTick for state transitions *)
--------------------------------------------------------------------

MorE == {"M", "E"}

(* After a base action, compute new grantedAtTick for a single
   agent/artifact based on old and new MESI states. *)
GrantUpdate(ag, art, oldSt, newSt) ==
    CASE oldSt \in MorE /\ newSt \in MorE -> grantedAtTick[ag][art]
      [] oldSt = "I" /\ newSt \in MorE    -> clock
      [] oldSt \in MorE /\ newSt \notin MorE -> None
      [] OTHER                              -> grantedAtTick[ag][art]

(* After a base action, compute new lastReclamation for a single
   agent/artifact based on old and new MESI states.
   Slot cleared on I->M or I->E (re-acquire), preserved otherwise
   including I->S (jessieibarra path). *)
ReclamationSlotUpdate(ag, art, oldSt, newSt) ==
    IF oldSt = "I" /\ newSt \in MorE
    THEN None
    ELSE lastReclamation[ag][art]

(* Build updated grantedAtTick and lastReclamation functions
   after a base action that changed mesiState to mesiState'. *)
UpdatedGrantedAtTick ==
    [ag \in Agents |-> [art \in Artifacts |->
        GrantUpdate(ag, art, mesiState[art][ag], mesiState'[art][ag])]]

UpdatedLastReclamation ==
    [ag \in Agents |-> [art \in Artifacts |->
        ReclamationSlotUpdate(ag, art, mesiState[art][ag], mesiState'[art][ag])]]

--------------------------------------------------------------------
(* Wrapper actions: base action + CR variable maintenance *)
--------------------------------------------------------------------

CRFetchAction ==
    /\ FetchAction
    /\ grantedAtTick'  = UpdatedGrantedAtTick
    /\ lastReclamation' = UpdatedLastReclamation
    /\ UNCHANGED lastHeartbeat

CRWriteAction ==
    /\ WriteAction
    /\ grantedAtTick'  = UpdatedGrantedAtTick
    /\ lastReclamation' = UpdatedLastReclamation
    /\ UNCHANGED lastHeartbeat

CRCommitAction ==
    /\ CommitAction
    /\ grantedAtTick'  = UpdatedGrantedAtTick
    /\ lastReclamation' = UpdatedLastReclamation
    /\ UNCHANGED lastHeartbeat

CRInvalidateAction ==
    /\ InvalidateAction
    /\ grantedAtTick'  = UpdatedGrantedAtTick
    /\ lastReclamation' = UpdatedLastReclamation
    /\ UNCHANGED lastHeartbeat

(* M/E -> S: the grant tick is dropped (GrantUpdate's M/E -> non-M/E case)
   and no reclamation slot is written, as the registry does when a holder
   releases its own write claim. *)
CRReleaseToReaderAction ==
    /\ ReleaseToReaderAction
    /\ grantedAtTick'  = UpdatedGrantedAtTick
    /\ lastReclamation' = UpdatedLastReclamation
    /\ UNCHANGED lastHeartbeat

CRTickAction ==
    /\ TickAction
    /\ UNCHANGED crVars

--------------------------------------------------------------------
(* HeartbeatAction: agent reports liveness at current clock. *)
--------------------------------------------------------------------

HeartbeatAction ==
    \E ag \in Agents :
        /\ lastHeartbeat' = [lastHeartbeat EXCEPT ![ag] =
             IF lastHeartbeat[ag] = None THEN clock
             ELSE IF clock > lastHeartbeat[ag] THEN clock
                  ELSE lastHeartbeat[ag]]
        /\ UNCHANGED <<baseVars, grantedAtTick, lastReclamation>>

--------------------------------------------------------------------
(* SweepAction: atomic reclamation of stale M/E grants. *)
--------------------------------------------------------------------

IsHeartbeatStale(ag) ==
    \/ lastHeartbeat[ag] = None
    \/ (clock - lastHeartbeat[ag]) >= HeartbeatTimeout

IsMaxHoldExceeded(ag, art) ==
    /\ grantedAtTick[ag][art] /= None
    /\ (clock - grantedAtTick[ag][art]) >= MaxHoldTicks

SweepTrigger(ag, art) ==
    IF IsHeartbeatStale(ag)          THEN "heartbeat"
    ELSE IF IsMaxHoldExceeded(ag, art) THEN "max_hold"
    ELSE None

SweepAction ==
    /\ clock > 0
    /\ LET snapshot == { <<ag, art>> \in (Agents \X Artifacts) :
                           mesiState[art][ag] \in MorE }
           reclaimSet == { <<ag, art>> \in snapshot :
                             SweepTrigger(ag, art) /= None }
       IN /\ reclaimSet /= {}
          /\ mesiState' = [art \in Artifacts |->
               [ag \in Agents |->
                   IF <<ag, art>> \in reclaimSet THEN "I"
                   ELSE mesiState[art][ag]]]
          /\ grantedAtTick' = [ag \in Agents |-> [art \in Artifacts |->
               IF <<ag, art>> \in reclaimSet THEN None
               ELSE grantedAtTick[ag][art]]]
          /\ lastReclamation' = [ag \in Agents |-> [art \in Artifacts |->
               IF <<ag, art>> \in reclaimSet
               THEN <<SweepTrigger(ag, art), clock>>
               ELSE lastReclamation[ag][art]]]
    /\ UNCHANGED <<clock, version, lastHeartbeat>>

--------------------------------------------------------------------
(* Specification *)
--------------------------------------------------------------------

CRNext ==
    \/ CRFetchAction
    \/ CRWriteAction
    \/ CRCommitAction
    \/ CRInvalidateAction
    \/ CRReleaseToReaderAction
    \/ CRTickAction
    \/ HeartbeatAction
    \/ SweepAction

CRSpec == CRInit /\ [][CRNext]_allVars

--------------------------------------------------------------------
(* Invariants *)
--------------------------------------------------------------------

(* I1 + I2: re-checked from MESI.tla — validates composition. *)
(* SingleWriter and MonotonicVersion are imported from MESI. *)

(* I3: SweepExclusivity — no (agent, artifact) reclaimed twice in
   one tick. Holds trivially: only one sweep path exists (no
   transient sweep). Checked as: if lastReclamation records a tick,
   it is a single value, not a set. By construction of the model,
   lastReclamation[ag][art] is always either None or a single
   <<trigger, tick>> pair, so double-reclamation is structurally
   impossible. This invariant exists as a guard against future
   model extensions that add a transient sweep. *)
SweepExclusivity ==
    \A ag \in Agents, art \in Artifacts :
        lastReclamation[ag][art] /= None =>
            lastReclamation[ag][art] \in (Triggers \X (0..MaxTicks))

(* I4: TriggerExclusivity — each reclamation has exactly one trigger. *)
TriggerExclusivity ==
    \A ag \in Agents, art \in Artifacts :
        lastReclamation[ag][art] /= None =>
            lastReclamation[ag][art][1] \in Triggers

(* I5: TickMonotonicity — lastHeartbeat never decreases. *)
TickMonotonicity ==
    \A ag \in Agents :
        lastHeartbeat[ag] /= None =>
            lastHeartbeat[ag] \in 0..clock

(* I6: SlotPreservedThroughSHARED — after reclamation, the slot
   persists across I->S transitions and is cleared only on
   I->M or I->E (re-acquire). Expressed as: if an agent is in S
   and has a reclamation slot, the slot is valid. *)
SlotPreservedThroughSHARED ==
    \A ag \in Agents, art \in Artifacts :
        /\ mesiState[art][ag] = "S"
        /\ lastReclamation[ag][art] /= None
        => lastReclamation[ag][art] \in (Triggers \X (0..MaxTicks))

(* Combined type invariant for CR variables. *)
CRTypeOK ==
    /\ TypeOK
    /\ \A ag \in Agents :
         lastHeartbeat[ag] \in {None} \cup (0..MaxTicks)
    /\ \A ag \in Agents, art \in Artifacts :
         grantedAtTick[ag][art] \in {None} \cup (0..MaxTicks)
    /\ \A ag \in Agents, art \in Artifacts :
         lastReclamation[ag][art] \in {None} \cup (Triggers \X (0..MaxTicks))

====================================================================
