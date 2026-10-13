# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Session handoff — loss first, then guarded: two sessions on one host, one shared notes file.

Runs five acts, each with two REAL OS processes, and exits 0 only if ALL hold:
  - RED       — plain file I/O → A's second write erases B's pickup line; nothing raised.
  - GREEN     — the same sequence through ``CoherentVolume`` → A's stale write is
                DENIED by the coordinator (``StaleView``); A reacquires and both
                lines survive EXACTLY.
  - CONTROL   — the green code with the guard pointed elsewhere → the loss returns,
                proving green depends on the deny, not on the re-read.
  - HANDOFF a — A checkpoints, writes once more, stops; B restores → the disk rewinds
                in one attempt and A learns on its NEXT write (denied by the file's
                own version check, then reacquire).
  - HANDOFF b — A is STILL WORKING while B restores → the restore concludes a bounded
                ``conflict``; A's latest edit survives; nothing was clobbered.

Constructed, deterministic, offline (a local coordinator; loopback only, no network).

    python -m examples.session_handoff.main

Every session is a spawned child that re-imports this module, so all work stays
under the ``__main__`` guard at the bottom.
"""

from __future__ import annotations

from collections.abc import Callable

from ccs.core.exceptions import RESTORE_OUTCOME_CONFLICT, RESTORE_OUTCOME_RESTORED
from examples.session_handoff import A_STATUS_1, A_STATUS_2, broken, fixed, handoff
from examples.session_handoff.broken import EXPECTED
from examples.session_handoff.handoff import RACING_ATTEMPTS
from examples.session_handoff.sessions import still_working_bytes

#: (result key, act runner, title line) — in the order the story is told: loss first.
_ACTS: tuple[tuple[str, Callable[[], dict], str], ...] = (
    ("red", broken.run_broken, "RED — plain file I/O, no coordination"),
    ("green", fixed.run_guarded, "GREEN — the same sequence through CoherentVolume (handoff/** guarded)"),
    ("control", fixed.run_control, "CONTROL — the same code with the guard pointed elsewhere (other/**)"),
    ("stopped", handoff.run_handoff_stopped, "HANDOFF — A hands off through a checkpoint, then stops writing"),
    ("racing", handoff.run_handoff_racing, "HANDOFF — A hands off through a checkpoint, and is still working"),
)


def _show_act(title: str, result: dict) -> None:
    """One act's title line and its trace, indented."""
    print(title)
    for line in result["trace"]:
        print(f"  {line}")
    if result.get("a_denial_message"):
        # Only the HANDOFF reason is printed: it is a fixed string. GREEN's reason
        # embeds a session hash and a timestamp, so it never reaches stdout.
        print(f"  deny reason A saw: {result['a_denial_message']}")
    print()


def _run_acts() -> dict[str, dict]:
    results: dict[str, dict] = {}
    for key, run, title in _ACTS:
        results[key] = run()
        _show_act(title, results[key])
    return results


def _red_checks(red: dict) -> dict[str, bool]:
    return {
        "RED      plain file I/O: B's pickup line is gone and nothing raised": (
            red["raised"] is None and red["final"] == A_STATUS_2 and not red["b_line_present"] and bool(red["lost"])
        ),
    }


def _green_checks(green: dict) -> dict[str, bool]:
    return {
        "GREEN    A's stale write is denied by the coordinator (StaleView)": (
            bool(green["denied"])
            and green["denial_exc"] == "StaleView"
            and fixed.denied_by_coordinator(green["denial_message"])
        ),
        "GREEN    A reacquires and rebuilds; both lines survive (exact)": (
            bool(green["recovered"]) and green["final"] == EXPECTED and not green["lost"]
        ),
    }


def _control_checks(control: dict) -> dict[str, bool]:
    return {
        "CONTROL  with the guard off, the loss returns": (
            not control["denied"]
            and bool(control["lost"])
            and not control["b_line_present"]
            and control["final"] == A_STATUS_2
        ),
    }


def _stopped_checks(stopped: dict) -> dict[str, bool]:
    return {
        "HANDOFF  A stopped writing: B's restore lands in one attempt; the disk is rewound": (
            stopped["outcome"] == RESTORE_OUTCOME_RESTORED
            and stopped["attempts"] == 1
            and stopped["disk_after_restore"] == A_STATUS_1
        ),
        "HANDOFF  A stopped writing: A's next write is denied by the file's own version check (StaleView); "
        "reacquire shows the handoff bytes": (
            bool(stopped["a_denied"])
            and stopped["a_denial_exc"] == "StaleView"
            and not fixed.denied_by_coordinator(stopped["a_denial_message"])
            and stopped["a_reacquired"] == A_STATUS_1
        ),
    }


def _racing_checks(racing: dict) -> dict[str, bool]:
    return {
        f"HANDOFF  A still working: restore concludes conflict after {RACING_ATTEMPTS} attempts; A's edit survives": (
            racing["outcome"] == RESTORE_OUTCOME_CONFLICT
            and racing["attempts"] == RACING_ATTEMPTS
            and racing["disk_after_restore"] == still_working_bytes(RACING_ATTEMPTS)
            and not racing["restore_landed"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    # ``argv`` is accepted for parity with the other demos' runners; there are no
    # flags — RED always runs first (loss first, then the guard).
    print("session handoff — two sessions on one host, one shared notes file (loss first, then guarded).")
    print("Same read→write sequence each time; only the coordination differs.\n")

    results = _run_acts()

    # Exact pins, not inequalities: a green that merely "differs from the broken
    # value" could still be wrong.
    checks = {
        **_red_checks(results["red"]),
        **_green_checks(results["green"]),
        **_control_checks(results["control"]),
        **_stopped_checks(results["stopped"]),
        **_racing_checks(results["racing"]),
    }
    for label, passed in checks.items():
        print(f"  [{'ok' if passed else 'FAIL'}] {label}")

    ok = all(checks.values())
    print("\nGREEN" if ok else "\nRED — a check failed; do not trust this build")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
