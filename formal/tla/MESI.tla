--------------------------- MODULE MESI ---------------------------
(* Base MESI coherence protocol — library module.
   Exports Init, action operators, and invariants.
   Does NOT define Next or Spec; downstream modules compose these. *)

EXTENDS Naturals, FiniteSets

CONSTANTS NumAgents, NumArtifacts, MaxTicks

Agents    == 1..NumAgents
Artifacts == 1..NumArtifacts

States == {"M", "E", "S", "I"}

VARIABLES clock, mesiState, version

baseVars == <<clock, mesiState, version>>

--------------------------------------------------------------------
(* Initialization *)
--------------------------------------------------------------------

Init ==
    /\ clock = 0
    /\ mesiState = [art \in Artifacts |-> [ag \in Agents |-> "I"]]
    /\ version = [art \in Artifacts |-> 1]

--------------------------------------------------------------------
(* Helper operators *)
--------------------------------------------------------------------

HoldersInStates(art, stateSet) ==
    { ag \in Agents : mesiState[art][ag] \in stateSet }

NonInvalidPeers(art, requester) ==
    { ag \in Agents : ag /= requester /\ mesiState[art][ag] /= "I" }

--------------------------------------------------------------------
(* Protocol actions — each constrains ONLY baseVars.
   Downstream modules conjoin CR-variable maintenance. *)
--------------------------------------------------------------------

(* fetch: requester I->E (no peers) or I->S (has peers).
   Peers in E or M are downgraded to S. No S->E branch. *)
FetchAction ==
    \E ag \in Agents, art \in Artifacts :
        /\ mesiState[art][ag] = "I"
        /\ LET peers == NonInvalidPeers(art, ag)
           IN IF peers = {}
              THEN
                /\ mesiState' = [mesiState EXCEPT ![art][ag] = "E"]
              ELSE
                /\ mesiState' = [mesiState EXCEPT
                     ![art] = [peer \in Agents |->
                         IF peer = ag THEN "S"
                         ELSE IF peer \in peers
                                  /\ mesiState[art][peer] \in {"E", "M"}
                              THEN "S"
                              ELSE mesiState[art][peer]]]
        /\ version' = version
        /\ UNCHANGED clock

(* write: requester any->E; invalidate all non-I peers to I. *)
WriteAction ==
    \E ag \in Agents, art \in Artifacts :
        /\ mesiState' = [mesiState EXCEPT
             ![art] = [peer \in Agents |->
                 IF peer = ag THEN "E"
                 ELSE IF peer \in NonInvalidPeers(art, ag)
                      THEN "I"
                      ELSE mesiState[art][peer]]]
        /\ version' = version
        /\ UNCHANGED clock

(* commit: requires requester in E or M; transition to M,
   bump version, invalidate all non-I peers. *)
(* MaxVersion bounds the version space for finite model checking.
   Without this, M->M commits create unbounded version growth. *)
MaxVersion == MaxTicks + NumAgents

CommitAction ==
    \E ag \in Agents, art \in Artifacts :
        /\ mesiState[art][ag] \in {"E", "M"}
        /\ version[art] < MaxVersion
        /\ mesiState' = [mesiState EXCEPT
             ![art] = [peer \in Agents |->
                 IF peer = ag THEN "M"
                 ELSE IF peer \in NonInvalidPeers(art, ag)
                      THEN "I"
                      ELSE mesiState[art][peer]]]
        /\ version' = [version EXCEPT ![art] = version[art] + 1]
        /\ UNCHANGED clock

(* invalidate: requester any->I, no peer side-effects. *)
InvalidateAction ==
    \E ag \in Agents, art \in Artifacts :
        /\ mesiState[art][ag] /= "I"
        /\ mesiState' = [mesiState EXCEPT ![art][ag] = "I"]
        /\ version' = version
        /\ UNCHANGED clock

(* release to reader: a holder gives back its own write claim and keeps a
   read, M or E -> S, at the version it held; no peer side-effects. This is
   the holder's own release at the end of a turn, when a subagent stops, or
   on a failed edit. A release report from an agent in S or I changes
   nothing, so it is a stutter and is not modeled. *)
ReleaseToReaderAction ==
    \E ag \in Agents, art \in Artifacts :
        /\ mesiState[art][ag] \in {"M", "E"}
        /\ mesiState' = [mesiState EXCEPT ![art][ag] = "S"]
        /\ version' = version
        /\ UNCHANGED clock

(* tick: advance the global clock by 1. *)
TickAction ==
    /\ clock < MaxTicks
    /\ clock' = clock + 1
    /\ UNCHANGED <<mesiState, version>>

--------------------------------------------------------------------
(* Invariants *)
--------------------------------------------------------------------

TypeOK ==
    /\ clock \in 0..MaxTicks
    /\ \A art \in Artifacts : version[art] \in 1..MaxVersion
    /\ \A art \in Artifacts, ag \in Agents :
         mesiState[art][ag] \in States

(* I1: at most one agent holds M or E per artifact. *)
SingleWriter ==
    \A art \in Artifacts :
        Cardinality(HoldersInStates(art, {"M", "E"})) <= 1

(* I2: artifact version lower bound. This state predicate checks
   version >= 1 at every reachable state. It does NOT check the
   stronger temporal property (version'[art] >= version[art]);
   that holds by construction since only CommitAction modifies
   version, and it increments by 1. *)
MonotonicVersion ==
    \A art \in Artifacts : version[art] >= 1

====================================================================
