---------------------------- MODULE Snapshot ----------------------------
(* Consistent multi-artifact snapshot-session amendment to the bounded-
   retention fenced MESI protocol. Proves that a session which captures a
   version-vector across SEVERAL artifacts at ONE linearization point
   (begin_session) reads a coherent cut -- no cross-artifact READ SKEW -- and
   that the versions it pins are held against the K-bounded GC for the
   session's lifetime (the exemptions seam), without disturbing any inherited
   safety invariant.

   Plan: docs/plans/2026-06-26-002-feat-read-side-transaction-snapshot-plan.md
   (Unit 1). Detailed spec: docs/brainstorms/2026-06-14-sb17-snapshot-sessions-v1-requirements.md.
   Origin (read-side, EO-4 = SB-17 = TX-1). Composes with Retention via EXTENDS.

   THE PROBLEM. With only the inherited single-artifact reads, an agent that
   reasons over several artifacts stitches N independent reads. A peer commit
   can land between read 1 and read N, so the agent observes artifact A at the
   version it had at t1 and artifact B at the version it had at t2 > t1, with a
   commit in between -- a snapshot that corresponds to NO single instant of the
   global version vector (cross-artifact read skew). The fix is a CONSISTENT
   CUT: capture every artifact's version at ONE atomic linearization point.

   Adds:
     - snapshot     : per-session captured version-vector, [Sessions ->
                      [Artifacts -> (1..MaxVersion) \cup {None}]]. A live
                      session's entry is a full capture (no None); a dead
                      session's entry is all-None. The pin map.
     - sessionLive  : [Sessions -> BOOLEAN]. Session liveness -- a STATE
                      COMPONENT THE GC READS (R4): a pinned version is exempt
                      from collection only while its session is live.
     - readSkew     : a sticky history flag -- TRUE iff any commit ever
                      interleaved a session that was only PARTIALLY captured
                      (some artifacts pinned, others not). Under the atomic
                      capture (correct spec) a session is never partial, so the
                      flag is unreachable-TRUE; the documented split mutant
                      makes it reachable. Mirrors the staleApply / collectedRead
                      idiom: the flag flips only when the modeled implementation
                      would have read a skewed cut; the correct spec never sets
                      it.
     - BeginSessionAction : the ATOMIC cut capture -- a not-live session pins
                      the CURRENT version of every artifact in ONE step.
                      Non-mutating: no MESI grant, no version bump, no fence
                      capture, no history change (UNCHANGED retentionVars).
     - EndSessionAction   : release the pins (sessionLive -> FALSE); the pinned
                      versions become collectible again (R4 liveness gate).
     - SnapshotCommitAction : RetentionCommitAction with the pin-aware GC
                      (SnapRetainAndCollect) PLUS the read-skew detector.
                      Replaces the inherited RetentionCommitAction.

   KEY MODELING DECISION (the crux). Retention's crux is that retain + bump +
   GC are ONE action (same-transaction), and mutant recipe 5 splits them to
   expose the crash window. Snapshot mirrors it for the READ side: the cut
   capture is ONE atomic action over ALL artifacts, so the correct spec can
   NEVER reach a state where a session is partially captured. The read-skew
   detector therefore lives in an action that is ALWAYS present -- the commit
   (SnapshotCommitAction sets readSkew TRUE when it interleaves a partially-
   captured session) -- and is vacuous in the correct spec because partial
   captures are unreachable. The MUTANT (README) splits BeginSessionAction into
   per-artifact CaptureArtifactAction steps; partial captures become reachable,
   a commit can interleave one, and TLC finds the NoReadSkewWithinCut violation.
   We model atomic capture STRUCTURALLY (one action), not by deleting a
   precondition -- exactly as the plan-review required.

   THE EXEMPTIONS SEAM (R14, bounded-exemptions GC-safety). A live session's
   pinned version must survive the K-bounded GC even after the retention window
   has slid past it -- the implementation's collectible_versions(exemptions=...)
   excludes pinned versions from collection. SnapRetainAndCollect keeps
   (newest-K window) UNION (versions pinned by live sessions). A pin is never
   collected while its session is live (PinAlwaysRetained), and the history can
   exceed K by at most the number of live sessions -- the BOUNDED-exemptions
   property the relaxed SnapshotTypeOK pins (Cardinality <= MaxRetained +
   |Sessions|). The window-union-exempt GC preserves NoCollectedRead by
   construction (it only KEEPS MORE than the window, never less). Mutant recipe
   10 (README) makes the GC ignore exemptions and TLC finds a pinned version
   collected out from under a live session (PinAlwaysRetained violation).

   DELIBERATELY OUT OF MODEL:
     (a) WRITE SKEW (R5) -- two sessions each reading a coherent cut and each
         committing one artifact can still violate a cross-artifact application
         invariant. That is SB-18 (atomic multi-artifact write), out of this
         plan. This spec proves READ-skew freedom only; the model has only
         single-artifact commits, so it asserts nothing about cross-artifact
         write atomicity.
     (b) the session-read SERVE path and session.commit -- the session reads
         from the pin (a pure lookup of snapshot[s]) and commits via the
         inherited version-CAS; neither adds protocol state. Their correctness
         is the runtime units' (plan Units 3/4) and is owned by the Python
         suite. This spec models the CUT and its GC-safety, which is what the
         protocol-level invariants can express.
     (c) coordinator restart / durability -- inherited exclusion (in TLA the
         state IS the durable truth). Restart-survival of pins is sqlite-only
         and is an implementation/test property (plan R6 / Unit 9).

   WHAT IS PROVEN: safety PRESERVATION -- every inherited invariant
   (SingleWriter, MonotonicVersion, the CR I3-I6 family, NoStaleApply,
   NoCollectedRead, ReadGenBounded) re-checked with the session machinery
   composed in, PLUS NoReadSkewWithinCut and PinAlwaysRetained. NOT proven:
   behavioral equivalence (no refinement mapping). LIVENESS is discharged as
   safety + prose per the repo's safety-only TLC convention (README). *)

EXTENDS Retention

(* The set of session identities (model values). Small for finite checking;
   one session already exposes read skew and the pin-GC exemption. *)
CONSTANT Sessions

VARIABLES snapshot, sessionLive, readSkew

snapVars == <<retentionVars, snapshot, sessionLive, readSkew>>

--------------------------------------------------------------------
(* Session helpers *)
--------------------------------------------------------------------

(* The versions pinned by currently-live sessions for an artifact -- the GC
   exemption set (R14). Read by SnapRetainAndCollect; a pinned version is
   never collected while its session is live. A dead session contributes
   nothing (its entry is all-None and sessionLive is FALSE). *)
PinnedVersions(art) ==
    { snapshot[s][art] : s \in {ss \in Sessions : sessionLive[ss]} }

(* A session is PARTIALLY captured iff some artifact is pinned and some other
   artifact is not. Under the atomic BeginSessionAction this is ALWAYS FALSE
   (capture is all-or-nothing in one step) -- the read-skew detector below is
   therefore vacuous in the correct spec. The split mutant (README recipe 9)
   makes it reachable. *)
PartiallyCaptured(s) ==
    /\ \E a \in Artifacts : snapshot[s][a] /= None
    /\ \E a \in Artifacts : snapshot[s][a] = None

(* Retain newVer and apply inline K-GC WITH the exemptions seam: keep the
   newest-K window (current included) UNION every version pinned by a live
   session. Versions outside the window that no live session pins are
   collected. A pinned version is held beyond K (the seam). With MaxRetained
   >= 1 the current version is always in the window, so it is never collectible
   (R4). Mirrors collectible_versions(exemptions=PinnedVersions). Mutant recipe
   10 drops the UNION PinnedVersions, collecting a live pin. *)
SnapRetainAndCollect(art, newVer) ==
    LET extended == [v \in (DOMAIN history[art]) \cup {newVer} |->
                        IF v = newVer THEN newVer ELSE history[art][v]]
        windowLo == IF newVer > MaxRetained THEN newVer - MaxRetained + 1 ELSE 1
        window   == windowLo..newVer
        keepDom  == (DOMAIN extended) \cap (window \cup PinnedVersions(art))
    IN [history EXCEPT ![art] = [v \in keepDom |-> extended[v]]]

--------------------------------------------------------------------
(* Initialization *)
--------------------------------------------------------------------

SnapshotInit ==
    /\ RetentionInit
    /\ snapshot = [s \in Sessions |-> [art \in Artifacts |-> None]]
    /\ sessionLive = [s \in Sessions |-> FALSE]
    /\ readSkew = FALSE

--------------------------------------------------------------------
(* BeginSessionAction: the ATOMIC consistent-cut capture. A not-live session
   pins the CURRENT version of EVERY artifact in ONE step -- one linearization
   point, so no peer commit can be partially visible within the cut. Wholly
   non-mutating: UNCHANGED retentionVars (no MESI grant, no version bump, no
   read_generation capture, no history change). This atomicity is the crux:
   it makes PartiallyCaptured unreachable, which is why the read-skew detector
   in the commit is vacuous in the correct spec. *)
--------------------------------------------------------------------

BeginSessionAction ==
    \E s \in Sessions :
        /\ ~sessionLive[s]
        /\ snapshot' = [snapshot EXCEPT ![s] = [art \in Artifacts |-> version[art]]]
        /\ sessionLive' = [sessionLive EXCEPT ![s] = TRUE]
        /\ UNCHANGED <<retentionVars, readSkew>>

--------------------------------------------------------------------
(* EndSessionAction: release the session's pins (R4 -- liveness gates the GC
   exemption). Once dead, the pinned versions are collectible again on the next
   commit. Non-mutating. *)
--------------------------------------------------------------------

EndSessionAction ==
    \E s \in Sessions :
        /\ sessionLive[s]
        /\ snapshot' = [snapshot EXCEPT ![s] = [art \in Artifacts |-> None]]
        /\ sessionLive' = [sessionLive EXCEPT ![s] = FALSE]
        /\ UNCHANGED <<retentionVars, readSkew>>

--------------------------------------------------------------------
(* SnapshotCommitAction: RetentionCommitAction (identical guard + protocol
   effect -- equality admits, present-but-superseded is a clean no-op, an
   absent operand admits via version-CAS) but the admitted branches use the
   PIN-AWARE GC (SnapRetainAndCollect), and the action carries the READ-SKEW
   DETECTOR: readSkew flips iff this commit interleaves a partially-captured
   session. PartiallyCaptured is unreachable under the atomic capture, so the
   detector is vacuous here; the split mutant makes it bite. Replaces the
   inherited RetentionCommitAction. *)
--------------------------------------------------------------------

SnapshotCommitAction ==
    \E ag \in Agents, art \in Artifacts :
        /\ version[art] < MaxVersion                (* finite bound *)
        /\ LET rg == readGeneration[ag][art]
               og == ownerGeneration[art]
               newVer == version[art] + 1
           IN \/ (* ADMIT (absent operand): plain OCC writer, version-CAS
                    arbitrates. Apply + retain (pin-aware) atomically. *)
                 /\ rg = None
                 /\ version' = [version EXCEPT ![art] = newVer]
                 /\ history' = SnapRetainAndCollect(art, newVer)
                 /\ UNCHANGED staleApply
              \/ (* WIN: claim present and current. *)
                 /\ rg /= None
                 /\ rg = og
                 /\ version' = [version EXCEPT ![art] = newVer]
                 /\ history' = SnapRetainAndCollect(art, newVer)
                 /\ staleApply' = (staleApply \/ (rg < og))
              \/ (* CONFLICT: present but superseded -- clean no-op. *)
                 /\ rg /= None
                 /\ ~(rg = og)
                 /\ UNCHANGED <<version, staleApply, history>>
        (* Read-skew detector: a commit landing while a session is mid-capture
           is the skew window. Vacuous under atomic capture (no partial
           sessions); the split mutant (README recipe 9) makes it reachable. *)
        /\ readSkew' = (readSkew \/ (\E s \in Sessions : PartiallyCaptured(s)))
        /\ UNCHANGED <<clock, mesiState, lastHeartbeat, grantedAtTick,
                       lastReclamation, ownerGeneration, readGeneration,
                       silentRevoke, collectedRead, snapshot, sessionLive>>

--------------------------------------------------------------------
(* Specification *)
--------------------------------------------------------------------

(* Inherited actions keep the session variables unchanged. The retention
   wrapping mirrors Retention's own Next; RetentionCommitAction is DELIBERATELY
   replaced by SnapshotCommitAction (the crux). *)
SnapshotNext ==
    \/ (CRFetchAction       /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply, silentRevoke, history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (FencingWriteAction  /\ UNCHANGED <<history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (FencingInvalidateAction /\ UNCHANGED <<history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (FencingReleaseToReaderAction /\ UNCHANGED <<history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (CRTickAction        /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply, silentRevoke, history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (HeartbeatAction     /\ UNCHANGED <<ownerGeneration, readGeneration, staleApply, silentRevoke, history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (FencingSweepAction  /\ UNCHANGED <<history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (ObserveGenAction    /\ UNCHANGED <<history, collectedRead, snapshot, sessionLive, readSkew>>)
    \/ (VersionedReadAction /\ UNCHANGED <<snapshot, sessionLive, readSkew>>)
    \/ SnapshotCommitAction
    \/ BeginSessionAction
    \/ EndSessionAction

SnapshotSpec == SnapshotInit /\ [][SnapshotNext]_snapVars

--------------------------------------------------------------------
(* Invariants *)
--------------------------------------------------------------------

(* Relaxes Retention's history-cardinality bound to admit the exemptions seam:
   the history can exceed K by at most the number of live sessions (each pins
   at most one version per artifact). Re-states the retention type with the
   relaxed bound and adds the session-variable types + the
   live <=> fully-captured well-formedness the atomic capture maintains. *)
SnapshotTypeOK ==
    /\ FencingTypeOK
    /\ collectedRead \in BOOLEAN
    /\ \A art \in Artifacts :
         /\ DOMAIN history[art] \subseteq (1..MaxVersion)
         /\ Cardinality(DOMAIN history[art]) <= MaxRetained + Cardinality(Sessions)
         /\ \A v \in DOMAIN history[art] : history[art][v] = v
    /\ readSkew \in BOOLEAN
    /\ \A s \in Sessions :
         /\ sessionLive[s] \in BOOLEAN
         /\ \A art \in Artifacts : snapshot[s][art] \in ({None} \cup (1..MaxVersion))
         (* a live session is fully captured; a dead one is all-None *)
         /\ sessionLive[s] <=> (\A art \in Artifacts : snapshot[s][art] /= None)

(* The headline read-side property: no commit ever interleaved a partially-
   captured session -- equivalently, every session reads a coherent cut. In the
   correct spec the atomic BeginSessionAction makes a partial capture
   unreachable, so this is unreachable-TRUE; README mutant recipe 9 (split the
   capture) makes it reachable and TLC finds a skewed read. NoStaleApply,
   NoCollectedRead, SingleWriter, MonotonicVersion and the CR invariants are
   inherited and re-checked to validate composition. *)
NoReadSkewWithinCut == readSkew = FALSE

(* The exemptions-seam safety property (R14 / R4): every version a LIVE session
   has pinned is still in the artifact's retained history -- the K-bounded GC
   never collects a pin out from under its session. Holds by construction
   (SnapRetainAndCollect keeps window UNION PinnedVersions); README mutant
   recipe 10 (drop the union) makes it fail once the window slides past a pin. *)
PinAlwaysRetained ==
    \A s \in Sessions :
        sessionLive[s] =>
            \A art \in Artifacts : snapshot[s][art] \in DOMAIN history[art]

==========================================================================
