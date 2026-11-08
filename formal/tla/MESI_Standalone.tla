----------------------- MODULE MESI_Standalone -----------------------
(* Standalone wrapper for the base MESI protocol model.
   Assembles Next from MESI.tla actions and defines Spec for TLC. *)

EXTENDS MESI

Next ==
    \/ FetchAction
    \/ WriteAction
    \/ CommitAction
    \/ InvalidateAction
    \/ ReleaseToReaderAction
    \/ TickAction

Spec == Init /\ [][Next]_baseVars

====================================================================
