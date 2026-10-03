# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-status`` — print tracked artifacts × sessions × MESI states.

Reads the coordinator's GET /status endpoint and renders a terminal-friendly
table. Backs the ``/agent-coherence status`` slash command.

Exit codes:
- 0: status fetched and printed (including "no coordinator running")
- 1: not in a git repo
- 2: coordinator running but returned an error
- 3: --self-test exercised but the smoke scenario failed

KTD-J (Unit 8): ``--self-test`` runs an end-to-end smoke against a real
coordinator. Two synthetic sessions exercise the stale-read warning
path; the smoke fails if (a) the coordinator is unreachable, (b) the
stale-warning response shape is wrong, or (c) counters do not increment
in the expected pattern. README's post-install step points operators at
this command so silent install regressions are caught locally before
they reach a real agent session.
"""

from __future__ import annotations

import argparse
import secrets
import shutil
import urllib.error
import uuid
from pathlib import Path
from typing import Any, Sequence

from ccs.adapters.claude_code.resolver import find_coordinator_root
from ccs.cli._coherence_client import (
    NODE_BACKEND,
    CoordinatorEndpoint,
    CoordinatorUnavailable,
    caller_principal_headers,
    claim_caller_principal,
    coordinator_backend,
    err,
    get,
    http_status_from_error,
    post,
    principal_refusal_reason,
    reportable_reason,
    resolve_endpoint,
)
from ccs.core.exceptions import RedirectRefused


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-coherence-status",
        description="Show tracked artifacts and per-session MESI states for this workspace.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override the coordinator root (default: walk up from cwd to git root).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the raw JSON response instead of the rendered table.",
    )
    # KTD-J (Unit 8): --detail mirrors the /status three-tier disclosure
    # model. Default 'full' so the local-operator CLI keeps surfacing pid
    # + absolute root + session names + all counters; 'metrics' for scrapers
    # that want only the counter block; 'minimal' for a redacted view
    # safe to paste in bug reports.
    parser.add_argument(
        "--detail",
        choices=["minimal", "full", "metrics"],
        default="full",
        help=(
            "Disclosure tier (default: full). 'minimal' redacts absolute paths, "
            "session names and user-added tracked patterns, and still reports "
            "per-session artifact state (the process id is reported at every "
            "tier); 'metrics' returns counters only; 'full' is the operator "
            "view used by /agent-coherence status."
        ),
    )
    # KTD-J (Unit 8): post-install smoke. Drives a two-session stale-read
    # scenario against a real coordinator; exits non-zero if the smoke
    # detects a regression. README's "After install:" step calls this.
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run an end-to-end smoke against a live coordinator. "
            "Validates pre-read/pre-edit/post-edit chain, stale-warning "
            "emission, and counter increments. Exits 0 on success, 3 on "
            "failure with an actionable diagnostic on stderr."
        ),
    )
    parser.add_argument(
        "--show-policy",
        action="store_true",
        help=(
            "Show user-added tracked paths that have not yet been observed "
            "(i.e., no pre-read hook has fired for them yet). These paths "
            "are in the policy but absent from the artifact registry."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = args.root if args.root is not None else find_coordinator_root()
    if root is None:
        err("agent-coherence-status: not in a git repository")
        return 1

    if args.self_test:
        return _run_self_test(Path(root), json_mode=args.json)

    try:
        endpoint = resolve_endpoint(Path(root))
        # R12 (Unit 6) + KTD-J (Unit 8): the detail tier is selected via
        # --detail. Only 'full' needs the Coherence-Local-Operator opt-in
        # header; the lower tiers degrade by design if the header is
        # missing, but we always set it from this CLI since it's a
        # legitimate local operator.
        payload = get(
            endpoint,
            f"/status?detail={args.detail}",
            extra_headers={"Coherence-Local-Operator": "true"},
        )
    except CoordinatorUnavailable as exc:
        err(f"agent-coherence-status: {exc}")
        return 0  # graceful — no coordinator is a normal state
    except urllib.error.HTTPError as exc:
        body = http_status_from_error(exc)
        msg = (body or {}).get("error", str(exc))
        err(f"agent-coherence-status: HTTP {exc.code}: {msg}")
        return 2

    if args.json:
        import json as _json
        if args.show_policy:
            observed = {a.get("path", "") for a in payload.get("tracked_artifacts", [])}
            payload["policy_pending_first_read"] = [
                p for p in payload.get("policy_summary", {}).get("user_added_patterns", [])
                if p not in observed
            ]
        print(_json.dumps(payload, indent=2), flush=True)
        return 0

    if args.detail == "metrics":
        _render_metrics(payload)
    else:
        _render_table(payload, show_policy=args.show_policy)
    return 0


def _run_self_test(root: Path, *, json_mode: bool = False) -> int:
    """KTD-J (Unit 8): end-to-end smoke against a live coordinator.

    Two synthetic sessions A and B, fresh on every run, drive the stale-read
    warning path:

    1. A pre-reads ``plan.md`` (tracked by default policy) — fresh. In a new
       workspace this read seeds v1; on a later run A is a first observer of
       the artifact an earlier run registered, which is answered stale and
       grants A a view, so A reads once more and that read must be fresh.
    2. B pre-edits ``plan.md`` — acquires EXCLUSIVE.
    3. B post-edits — commits v2, releases EXCLUSIVE.
    4. A pre-reads ``plan.md`` again — stale warning fires.
    5. Verify ``status: stale`` + ``hookSpecificOutput.additionalContext``
       prose contains the path.
    6. Verify ``stale_warning_emitted_total`` incremented in ``/status``.

    Returns 0 on success, 3 on any verification failure. Diagnostics
    print to stderr so the README's post-install step ("run
    ``agent-coherence-status --self-test``") gives the operator an
    actionable error rather than a stack trace.
    """
    try:
        endpoint = resolve_endpoint(root)
    except CoordinatorUnavailable as exc:
        err(
            f"agent-coherence-status --self-test: coordinator unreachable "
            f"({exc}). Spawn one first by running any hook (or "
            f"``agent-coherence-coordinator``)."
        )
        return 3

    # Two FRESH synthetic sessions per run. One process drives both from start
    # to finish, so each claims its caller principal once, with a nonce
    # generated here, and holds both in memory for the run (a long-lived
    # caller, plan KTD5): nothing is stored under .coherence/, so a run never
    # depends on what an earlier run left there — deleting the
    # caller-principal-* files or resetting state.db cannot break a later run
    # — and neither value is ever printed. pre-edit and post-edit require the
    # principal once a session is bound.
    sid_a, sid_b = str(uuid.uuid4()), str(uuid.uuid4())
    path = "plan.md"  # part of DEFAULT_TRACKED_PATTERNS
    principals: dict[str, str | None] = {}
    for sid in (sid_a, sid_b):
        claimed, principals[sid] = _claim_self_test_principal(endpoint, root, sid)
        if not claimed:
            return 3

    # What a failure reports is built from the step, the HTTP status, the
    # answer's known tokens and numbers (_answer_summary) — never the answer
    # itself: a coordinator, or anything in front of it, that echoed the
    # principal header into a field would otherwise print it here.
    def _step(name: str, body: dict[str, Any]) -> dict[str, Any] | None:
        headers = caller_principal_headers(principals[body["session_id"]])
        try:
            answer = post(endpoint, name, body, extra_headers=headers)
        except urllib.error.HTTPError as exc:
            reason = principal_refusal_reason(exc)
            if reason is not None:
                err(f"--self-test: {name} refused the caller principal ({reason})")
            else:
                err(f"--self-test: {name} returned HTTP {exc.code}")
            return None
        except CoordinatorUnavailable as exc:
            err(f"--self-test: {name} failed: {exc}")
            return None
        except RedirectRefused as exc:
            # Refused, never followed; reported by its status alone, since
            # the Location is the coordinator's text.
            err(f"--self-test: {name} was redirected (HTTP {exc.status}); not followed")
            return None
        if not isinstance(answer, dict):
            err(f"--self-test: {name} answered with {_answer_summary(answer)}")
            return None
        return answer

    # Step 1 — A's first read seeds the artifact, or (a later run) observes
    # the one an earlier run seeded; either way A then holds a current view.
    r1 = _step("/hooks/pre-read", {
        "session_id": sid_a, "path": path,
        "content_hash": "a" * 64,
    })
    if r1 is not None and r1.get("status") != "fresh":
        r1 = _step("/hooks/pre-read", {"session_id": sid_a, "path": path})
    if r1 is None:
        return 3
    if r1.get("status") != "fresh":
        err(f"--self-test: expected fresh on first pre-read, got {_answer_summary(r1)}")
        return 3

    # Step 2 — B pre-edits.
    r2 = _step("/hooks/pre-edit", {"session_id": sid_b, "path": path})
    if r2 is None:
        return 3
    if not r2.get("ok", True):
        err(f"--self-test: pre-edit failed: {_answer_summary(r2)}")
        return 3

    # Step 3 — B commits.
    r3 = _step("/hooks/post-edit", {
        "session_id": sid_b, "path": path,
        "content_hash": "b" * 64, "success": True,
    })
    if r3 is None:
        return 3
    if not r3.get("ok"):
        err(f"--self-test: post-edit failed: {_answer_summary(r3)}")
        return 3

    # Step 4 — A re-reads → expect stale.
    r4 = _step("/hooks/pre-read", {"session_id": sid_a, "path": path})
    if r4 is None:
        return 3
    if r4.get("status") != "stale":
        err(
            f"--self-test: expected stale warning on A's re-read after B's "
            f"commit, got {_answer_summary(r4)}. "
            f"This usually means the hooks aren't wired or the coordinator "
            f"is running a stale build."
        )
        return 3
    out = r4.get("hookSpecificOutput")
    ctx = out.get("additionalContext") if isinstance(out, dict) else None
    if not isinstance(ctx, str) or path not in ctx:
        shape = f"{len(ctx)} characters" if isinstance(ctx, str) else "absent"
        err(
            f"--self-test: stale-warning prose did not mention {path} "
            f"(additionalContext: {shape})"
        )
        return 3

    # Step 5 — counters reflect the activity.
    try:
        status = get(
            endpoint, "/status?detail=metrics",
            extra_headers={"Coherence-Local-Operator": "true"},
        )
    except urllib.error.HTTPError as exc:
        err(f"--self-test: /status returned HTTP {exc.code}")
        return 3
    except CoordinatorUnavailable as exc:
        err(f"--self-test: /status failed: {exc}")
        return 3
    except RedirectRefused as exc:
        err(f"--self-test: /status was redirected (HTTP {exc.status}); not followed")
        return 3
    if not isinstance(status, dict):
        err(f"--self-test: /status answered with {_answer_summary(status)}")
        return 3
    if _count(status, "stale_warning_emitted_total") < 1:
        err(
            "--self-test: stale_warning_emitted_total did not increment "
            "(coordinator KTD-J counters appear to be inert)."
        )
        return 3
    eps = status.get("endpoint_counters")
    eps = eps if isinstance(eps, dict) else {}
    pre_reads, post_edits = _count(eps, "pre_read_total"), _count(eps, "post_edit_total")
    if pre_reads < 2 or post_edits < 1:
        err(
            f"--self-test: endpoint counters did not reflect the four-step "
            f"scenario: pre_read_total={pre_reads}, post_edit_total={post_edits}"
        )
        return 3

    if json_mode:
        import json as _json
        steps = [
            "pre-read fresh",
            "pre-edit",
            "post-edit commit",
            f"pre-read STALE ({path})",
        ]
        print(_json.dumps({"self_test": "pass", "steps_observed": steps, "error": None}), flush=True)
    else:
        print("agent-coherence-status --self-test: OK", flush=True)
        print(
            f"  pre-read fresh → pre-edit → post-edit commit → pre-read STALE "
            f"({path}) — all four steps observed.",
            flush=True,
        )
    return 0


def _claim_self_test_principal(
    endpoint: CoordinatorEndpoint, root: Path, session_id: str
) -> tuple[bool, str | None]:
    """Claim ``session_id``'s caller principal for the self-test, presenting a
    nonce generated for this claim alone: ``(True, principal)`` when bound,
    ``(True, None)`` when the coordinator issues none (a 404, or a Node
    coordinator, which is not asked — as the hook client does not ask it), and
    ``(False, None)`` after reporting a claim that did not bind. The session
    is fresh, so a refusal means something else claimed it first."""
    if coordinator_backend(root) == NODE_BACKEND:
        return True, None
    claim = claim_caller_principal(endpoint, session_id, secrets.token_urlsafe(32))
    if claim.outcome == "bound":
        return True, claim.principal
    if claim.outcome == "unsupported":
        return True, None
    err(f"--self-test: caller principal not obtained ({claim.outcome}: {claim.detail})")
    return False, None


_SELF_TEST_STATUSES: frozenset[str] = frozenset({"fresh", "stale"})
"""The pre-read ``status`` values the self-test names; any other reads
``unrecognised``."""


def _answer_summary(answer: object) -> str:
    """An answer described by what the self-test knows of it — its ``status``
    (a known value), ``ok`` and ``degraded`` flags, and ``reason`` as a known
    token (:func:`reportable_reason`) — never by the answer's own text."""
    if answer is None:
        return "no answer"
    if not isinstance(answer, dict):
        return "a non-object answer"
    status = answer.get("status")
    known_status = status if isinstance(status, str) and status in _SELF_TEST_STATUSES else None
    parts = [
        f"status={known_status or ('absent' if status is None else 'unrecognised')}",
        f"ok={_flag(answer.get('ok'))}",
    ]
    if answer.get("degraded") is True:
        parts.append("degraded=true")
    if "reason" in answer:
        parts.append(f"reason={reportable_reason(answer['reason'])}")
    return ", ".join(parts)


def _flag(value: object) -> str:
    if value is None:
        return "absent"
    if isinstance(value, bool):
        return "true" if value else "false"
    return "unrecognised"


def _count(counters: dict[str, Any], key: str) -> int:
    """A counter from a ``/status`` answer, or ``0`` when absent or not an
    integer (so it is reported as a number, never as the answer's text)."""
    value = counters.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _terminal_columns() -> int:
    """Best-effort terminal width. Honors ``COLUMNS``; falls back to 80 when
    stdout is not a tty (the common case when this runs under an agent's shell
    tool), so rows never exceed the narrowest mainstream terminal."""
    return shutil.get_terminal_size(fallback=(80, 24)).columns


def _elide_middle(text: str, max_len: int) -> str:
    """Shorten *text* to *max_len* columns with a middle ellipsis so a path's
    top-level dir (head) and filename (tail) both stay visible. Keeping the
    whole row within the terminal width is what stops the version column from
    soft-wrapping onto its own line."""
    if max_len <= 1 or len(text) <= max_len:
        return text
    keep = max_len - 1  # one column for the ellipsis
    head = keep // 2
    return f"{text[:head]}…{text[-(keep - head):]}"


def _render_table(payload: dict[str, Any], *, show_policy: bool = False) -> None:
    """Manual column alignment — stdlib only, no rich/tabulate."""
    tracked = payload.get("tracked_artifacts", [])
    sessions = payload.get("sessions", [])
    policy = payload.get("policy_summary", {})
    # AC-02 cross-backend parity: prefer canonical ``coordinator_uptime_seconds``
    # (KTD-J _seconds convention), fall back to deprecated ``coordinator_uptime_s``
    # so we keep working against pre-rename coordinators during the
    # deprecation window.
    uptime = payload.get("coordinator_uptime_seconds",
                         payload.get("coordinator_uptime_s", 0.0))
    pid = payload.get("coordinator_pid", 0)
    backend = payload.get("coordinator_backend", "python")
    version = payload.get("coordinator_version", "")

    header_bits: list[str] = []
    if pid:
        header_bits.append(f"pid={pid}")
    header_bits.append(f"uptime={uptime:.0f}s")
    header_bits.append(f"backend={backend}")
    if version:
        header_bits.append(f"version={version}")
    print("Coordinator: " + " ".join(header_bits))
    print()

    # Policy section first — distinguishes "what's eligible to be tracked"
    # (defaults + user-added patterns) from "what's been observed so far"
    # (tracked_artifacts, which requires at least one Read to seed).
    if policy:
        default_n = policy.get("default_pattern_count", 0)
        user_n = policy.get("user_added_pattern_count", 0)
        ignored_n = policy.get("ignored_pattern_count", 0)
        print(
            "Policy: "
            f"{default_n} default pattern(s), "
            f"{user_n} user-added, "
            f"{ignored_n} ignored"
        )
        print()

    if not tracked:
        # Disambiguate: empty registry vs empty policy. Policy may match
        # paths the registry hasn't observed yet (first Read seeds them).
        if policy and (policy.get("default_pattern_count", 0) + policy.get("user_added_pattern_count", 0)) > 0:
            print(
                "No artifacts observed yet (paths matching the policy will "
                "be registered on first Read)."
            )
        else:
            print("No tracked artifacts (policy is empty).")
    else:
        print("Observed artifacts:")
        # Legend: the version column is opaque without it (the original report
        # was "what does the number mean?"). It's the artifact's coherence
        # revision — that's what stale-read detection compares against.
        print("  version = artifact revision: starts at 1, +1 on every committed edit")
        print("  (a read is flagged stale when its version is behind the current one)")
        print()
        ver_w = len("version")
        # Cap the path column to the terminal so long paths get middle-elided
        # instead of padding every row past the screen edge — which is what
        # soft-wrapped the version onto the next line. ``+1`` is a safety column.
        chrome = 2 + 2 + ver_w + 1
        max_path_w = max(len("path"), _terminal_columns() - chrome)
        longest = max(len(a.get("path", "")) for a in tracked)
        path_w = min(longest, max_path_w)
        print(f"  {'path':<{path_w}}  {'version':>{ver_w}}")
        print(f"  {'-' * path_w}  {'-' * ver_w}")
        for a in tracked:
            label = _elide_middle(a.get("path", ""), path_w)
            print(f"  {label:<{path_w}}  {a.get('version', 0):>{ver_w}}")
    print()

    if show_policy:
        observed_paths = {a.get("path", "") for a in tracked}
        pending = [
            p for p in policy.get("user_added_patterns", [])
            if p not in observed_paths
        ]
        if pending:
            print("Tracked (pending first read):")
            for p in pending:
                print(f"  {p}")
        else:
            print("Tracked (pending first read): none")
        print()

    if not sessions:
        print("No active sessions.")
    else:
        print("Sessions:")
        for s in sessions:
            sid = s.get("agent_id", "?")
            # A null agent_name has two causes and the renderer cannot tell
            # them apart: the name was redacted because this response is below
            # the operator tier (R6 — it embeds the raw session id), or the
            # coordinator holds the grant but never knew a name for its holder
            # (a grant that outlived the process that issued it). Either way
            # the session id is a one-way uuid5 input and is not recoverable
            # from agent_id, so name both causes and assert neither — and do
            # not print "None".
            name = s.get("agent_name") or (
                "(name unknown — redacted below the operator tier, "
                "or a grant predating this coordinator)"
            )
            per_artifact = dict(s.get("states", {}))
            # #195: the operator tier names the paths this session lost to the
            # coordinator sweep. They are not held grants, so they render in
            # the same column under their own label rather than as a state.
            for path, cause in (s.get("reclaimed") or {}).items():
                per_artifact[path] = (
                    f"reclaimed ({cause.get('trigger', '?')} at tick {cause.get('tick', '?')})"
                )
            print(f"  {sid[:8]}  {name}")
            if not per_artifact:
                print("    (no held grants)")
                continue
            # Same elision as the artifacts table so a long held path can't
            # push the MESI state column off-screen onto a wrapped line.
            state_w = max(len(s) for s in per_artifact.values())
            chrome = 4 + 2 + state_w + 1
            max_path_w = max(1, _terminal_columns() - chrome)
            path_w = min(max(len(p) for p in per_artifact), max_path_w)
            for path, state in sorted(per_artifact.items()):
                print(f"    {_elide_middle(path, path_w):<{path_w}}  {state}")

    # KTD-J (Unit 8): counters section. Only printed when the payload
    # actually carries counter data — the minimal tier strips them.
    _render_counter_block(payload)


def _render_counter_block(payload: dict[str, Any]) -> None:
    """KTD-J counter block, printed after the artifacts/sessions section
    of the full-tier table. No-op if the payload doesn't carry counters
    (e.g., minimal tier responses)."""
    endpoint_counters = payload.get("endpoint_counters") or {}
    has_endpoint_counters = any(v for v in endpoint_counters.values())
    keys_present = [
        k for k in (
            "intra_task_acquire_release_total",
            "stale_warning_emitted_total",
            "stale_warning_reread_total",
            "watchdog_timeouts_total",
            "watchdog_queue_overflows_total",
            "handler_concurrency_overflows_total",
            "cold_start_duration_ms",
            "sweep_reclaims_total",
        ) if k in payload
    ]
    if not has_endpoint_counters and not keys_present:
        return

    print()
    print("Counters:")
    if endpoint_counters:
        # Stable order so operator-facing output is diff-friendly.
        for name in (
            "pre_read_total",
            "pre_edit_total",
            "post_edit_total",
            "session_stop_total",
            "pre_bash_total",
            "pre_grep_total",
            "policy_track_total",
            "policy_untrack_total",
            "status_total",
        ):
            value = endpoint_counters.get(name, 0)
            print(f"  {name:<40}  {value}")
    for name in keys_present:
        value = payload.get(name, 0)
        if isinstance(value, float):
            value_str = f"{value:.1f}"
        else:
            value_str = str(value)
        print(f"  {name:<40}  {value_str}")


def _render_metrics(payload: dict[str, Any]) -> None:
    """KTD-J `--detail metrics` rendering — counter block only, no
    artifact/session detail. Used by dashboard scrapers that want a
    consistent counter format without parsing JSON."""
    backend = payload.get("coordinator_backend", "python")
    version = payload.get("coordinator_version", "")
    print(f"Coordinator metrics: backend={backend} version={version}")
    _render_counter_block(payload)


if __name__ == "__main__":
    raise SystemExit(main())
