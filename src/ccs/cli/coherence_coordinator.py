# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-coordinator`` — lazy-spawn the workspace coordinator.

Invoked by the plugin's ``bin/ensure-coordinator`` shim from the
``SessionStart`` hook. Cheap to re-invoke: if a coordinator is already
running for the workspace, returns immediately.

## Detach model

The coordinator must outlive this CLI invocation — Claude Code's
SessionStart hook expects the shim to exit promptly so the session can
proceed. We can't simply call :func:`ensure_coordinator` and return,
because :func:`ensure_coordinator` starts the HTTP server in a *daemon*
thread that dies the moment the parent process exits.

Pattern:

1. Parent process probes ``.coherence/server.pid``. If a coordinator is
   already up and TCP-reachable, print its port and exit 0 — no fork.
2. Otherwise, parent spawns a detached child via :class:`subprocess.Popen`
   with ``start_new_session=True``. The child runs in ``--_daemonized``
   mode: calls :func:`ensure_coordinator`, then blocks indefinitely so
   its daemon threads stay alive.
3. Parent polls ``.coherence/server.pid`` for the port to appear (the
   child writes it inside :func:`ensure_coordinator`). Once the port is
   visible and TCP-reachable, parent prints it and exits 0.

The ``--_daemonized`` flag is intentionally underscore-prefixed: this is
the internal worker mode, not part of the user-facing CLI contract.

## Store check (#345)

``--check-store [--json]`` starts nothing. It says what a start would do to
``<root>/.coherence/state.db`` — open it unchanged, create it, migrate it
(after which a coordinator built for the old schema refuses it), or refuse it
— by running the start's writer open on a private copy
(:func:`ccs.adapters.claude_code.store_check.check_store`), and exits with
that outcome's code. It never creates ``.coherence/``. ``--json`` prints one
object (``kind: "store_check"``) carrying the outcome, the exit code, the
store path, ``coordinator_backend`` (always ``"python"``: the answer is for
this build's Python coordinator), ``registry_schema_version`` (this build's)
and ``store_registry_schema_version`` (the store's, or null), the reason and
a message; without it one plain line is printed, which ``--quiet``
suppresses.

Exit codes (spawn mode):
- 0: coordinator is up and reachable (port printed to stdout)
- 1: not in a git repository (no coordinator root resolved)
- 2: coordinator failed to spawn (platform unsupported, FS read-only,
     port never appeared in the file, or detached child crashed), or a
     usage error

Exit codes (``--check-store``; 1 and 2 as above, 3-9 left to spawn mode):
- 0: current — a start opens the store unchanged
- 10: fresh — no store (or an empty one); a start creates it
- 11: will_migrate — a start migrates the store in place
- 12: newer — the store is past this build's schema; a start refuses it
- 13: foreign — the Node coordinator's ledger wrote it; a start refuses it
- 14: unusable — a start fails on the store itself
- 15: live — a coordinator is running; a start joins it, migrating nothing
  while it runs (the store is not read, so its schema is unknown)
- 16: cannot_tell — the check could not observe the answer
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

from ccs.adapters.claude_code.auth import is_loopback_host, validate_bind_host
from ccs.adapters.claude_code.lifecycle import (
    LifecycleConfig,
    ensure_coordinator,
    stop_coordinator,
)
from ccs.adapters.claude_code.lifecycle import (
    read_port_from_file as _read_port_from_file,
)
from ccs.adapters.claude_code.lifecycle import (
    tcp_probe as _tcp_probe,
)
from ccs.adapters.claude_code.resolver import find_coordinator_root
from ccs.adapters.claude_code.store_check import StoreCheckOutcome, check_store
from ccs.cli._coherence_client import err, escape_nonprintable

logger = logging.getLogger(__name__)

# --check-store exit codes (#345): a CLI contract a host hard-codes, so fixed
# once released. 0 is "nothing to do"; the rest sit at 10+ so a check code can
# never collide with a present or future spawn-mode code (1 = not a git repo,
# 2 = spawn failure or usage; 3-9 stay free for spawn mode).
_CHECK_STORE_EXIT_CODES: dict[StoreCheckOutcome, int] = {
    StoreCheckOutcome.CURRENT: 0,
    StoreCheckOutcome.FRESH: 10,
    StoreCheckOutcome.WILL_MIGRATE: 11,
    StoreCheckOutcome.NEWER: 12,
    StoreCheckOutcome.FOREIGN: 13,
    StoreCheckOutcome.UNUSABLE: 14,
    StoreCheckOutcome.LIVE: 15,
    StoreCheckOutcome.CANNOT_TELL: 16,
}

# Import-time exhaustiveness pin (the coherence_replay pattern): an outcome
# added to StoreCheckOutcome without a code, a shared code, or a code that
# reads as "not a git repo" / "usage" must fail LOUDLY at import, not as a
# KeyError mid-check or a host acting on the wrong answer.
if (
    set(_CHECK_STORE_EXIT_CODES) != set(StoreCheckOutcome)
    or len(set(_CHECK_STORE_EXIT_CODES.values())) != len(_CHECK_STORE_EXIT_CODES)
    or set(_CHECK_STORE_EXIT_CODES.values()) & {1, 2}
):
    raise RuntimeError(
        "ccs.cli.coherence_coordinator: _CHECK_STORE_EXIT_CODES must map every "
        "StoreCheckOutcome to its own exit code, none of them 1 or 2 "
        f"(codes={sorted((o.value, c) for o, c in _CHECK_STORE_EXIT_CODES.items())}, "
        f"outcomes={sorted(o.value for o in StoreCheckOutcome)})."
    )

#: How long the parent polls for the detached child to write the port
#: file. Generous: cold Python interpreter + SQLite WAL setup + bind
#: can take several seconds on slow disks. A wall-clock deadline, not an
#: attempt count — each probe carries its own internal retry budget, and
#: multiplying the two silently inflated the window several-fold against
#: an unreachable routed host (past Claude Code's default hook timeout).
_DETACH_PORT_WAIT_TIMEOUT_SEC = 10.0
_DETACH_PORT_WAIT_INTERVAL_SEC = 0.1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-coherence-coordinator",
        description="Lazy-spawn the Claude Code coherence coordinator for this workspace.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override the coordinator root (default: walk up from cwd to git root).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the 'port=N' line on stdout (still exits 0 on success).",
    )
    parser.add_argument(
        "--bind-host",
        default="127.0.0.1",
        help=(
            "Address to bind the coordinator to (default: 127.0.0.1). A non-loopback "
            "private-range address (RFC-1918) enables the cross-host demo and requires "
            "CCS_REMOTE_COORDINATOR=1; wildcard / public / loopback-alias are rejected."
        ),
    )
    parser.add_argument(
        "--prepare-for-migration",
        action="store_true",
        help=(
            "Atomically release all EXCLUSIVE/MODIFIED grants and shut "
            "down the coordinator for this workspace. RUN THIS before "
            "switching the coherence backend between Python and Node — "
            "see the v0.1.1 migration runbook."
        ),
    )
    parser.add_argument(
        "--check-store",
        action="store_true",
        help=(
            "Start nothing: say what starting a coordinator would do to "
            ".coherence/state.db (open it as is, create it, migrate it, or "
            "refuse it) and exit with that outcome's code: 0 current, 10 fresh, "
            "11 will_migrate, 12 newer, 13 foreign, 14 unusable, 15 live, "
            "16 cannot_tell."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --check-store: print the answer as one JSON object.",
    )
    parser.add_argument(
        "--_daemonized",
        action="store_true",
        help=argparse.SUPPRESS,  # hidden — internal detached-worker flag
    )
    parser.add_argument(
        "--no-detach",
        action="store_true",
        help=argparse.SUPPRESS,  # hidden — test mode: in-process spawn
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _reject_check_store_misuse(parser, args)

    root_arg = args.root if args.root is not None else find_coordinator_root()
    if root_arg is None:
        err("agent-coherence-coordinator: not in a git repository")
        return 1
    root = Path(root_arg).resolve()

    # #345: before the bind-host validation, because the check binds and dials
    # nothing, and before any spawn path, because it must never run
    # _ensure_coherence_dir (a check leaves the workspace as it found it).
    if args.check_store:
        return _run_check_store(root, as_json=args.json, quiet=args.quiet)

    # Validate here, not only in the forked child's server construction:
    # the parent dials this host directly for the reuse/spawn probes, and
    # 0.0.0.0 aliases to localhost at connect time — an unvalidated
    # wildcard would "pass" against a loopback coordinator instead of
    # being rejected as documented. Loopback returns without reading env.
    try:
        validate_bind_host(args.bind_host)
    except ValueError as exc:
        err(f"agent-coherence-coordinator: {exc}")
        return 2

    # Daemonized worker mode: bind + serve, then block forever so the
    # daemon HTTP/sweep/idle threads stay alive.
    if args._daemonized:
        return _run_daemonized(root, bind_host=args.bind_host)

    # Unit 8 (Decision 1): release-all-grants + shutdown for backend
    # switch safety. Routed through the running coordinator so we
    # invalidate every M/E grant from the same process that owns the
    # SQLite registry (the operator's CLI invocation may not be that
    # process when the coordinator was detached on a prior session).
    if args.prepare_for_migration:
        return _run_prepare_for_migration(
            root, quiet=args.quiet, bind_host=args.bind_host
        )

    # Test mode: keep the legacy in-process behavior so test_claude_code_cli
    # can spawn + assert in the same Python process. Not for users.
    if args.no_detach:
        port = ensure_coordinator(root, bind_host=args.bind_host)
        if port == -1:
            err("agent-coherence-coordinator: coordinator could not be spawned")
            return 2
        if not args.quiet:
            print(f"port={port}", flush=True)
        return 0

    # Normal user path: probe for an existing coordinator first.
    pid_file = root / ".coherence" / "server.pid"
    existing_port = _read_port_from_file(pid_file)
    cfg = LifecycleConfig()
    if existing_port is not None and _tcp_probe(
        existing_port, cfg, bind_host=args.bind_host
    ):
        if not args.quiet:
            print(f"port={existing_port}", flush=True)
        return 0

    # No live coordinator — fork a detached child to run it.
    return _spawn_detached(root, quiet=args.quiet, bind_host=args.bind_host)


def _reject_check_store_misuse(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Usage errors (exit 2) for the #345 flags: ``--json`` only shapes the
    check's answer, and the check never combines with a mode that starts or
    stops a coordinator (a "check" that drained grants or spawned would be
    the very side effect a host runs it to avoid)."""
    if args.json and not args.check_store:
        parser.error("--json requires --check-store")
    if not args.check_store:
        return
    for flag, given in (
        ("--prepare-for-migration", args.prepare_for_migration),
        ("--no-detach", args.no_detach),
        ("--_daemonized", args._daemonized),
    ):
        if given:
            parser.error(f"--check-store cannot be combined with {flag}")


def _run_check_store(root: Path, *, as_json: bool, quiet: bool) -> int:
    """#345: answer what a start would do to the store, and exit with the
    outcome's code. ``--json`` always prints its one object (a host parses
    it); the plain line obeys ``--quiet``."""
    result = check_store(root)
    exit_code = _CHECK_STORE_EXIT_CODES[result.outcome]
    if as_json:
        # Two schema-version keys, never a bare ``schema_version``: the
        # build's and the store's must not be readable as each other.
        print(
            json.dumps(
                {
                    "kind": "store_check",
                    "outcome": result.outcome.value,
                    "exit_code": exit_code,
                    "db_path": str(result.db_path),
                    "coordinator_backend": "python",
                    "registry_schema_version": result.expected_version,
                    "store_registry_schema_version": result.found_version,
                    "reason": result.reason,
                    "message": result.message,
                }
            ),
            flush=True,
        )
    elif not quiet:
        # The message carries the store's own text (an object name inside a
        # SQLite error), so it is escaped like any foreign string printed to
        # a terminal; json.dumps above already escapes control characters.
        message = escape_nonprintable(result.message)
        print(f"check-store: {result.outcome.value}: {message}", flush=True)
    return exit_code


def _run_prepare_for_migration(
    root: Path, *, quiet: bool, bind_host: str = "127.0.0.1"
) -> int:
    """Unit 8: backend-switch safety entry point.

    Connects to the running coordinator, posts /admin/prepare-for-migration
    with the operator opt-in header, then polls the pid file + TCP probe
    until the coordinator is genuinely gone. Returns 0 on clean
    shutdown, 0 also when no coordinator was running (idempotent — safe
    to script as a pre-switch step that may or may not find a live
    coordinator), 2 on any error reaching the coordinator.
    """
    import urllib.error as _urlerr

    from ccs.cli._coherence_client import (
        CoordinatorUnavailable,
        resolve_endpoint,
    )
    from ccs.cli._coherence_client import (
        post as _post,
    )
    from ccs.core.exceptions import InsecureTransportRefused

    pid_file = root / ".coherence" / "server.pid"
    if not pid_file.exists():
        if not quiet:
            print("prepare-for-migration: no coordinator running (idempotent no-op)", flush=True)
        return 0

    try:
        endpoint = resolve_endpoint(root)
    except CoordinatorUnavailable as exc:
        # The pid file existed but the coordinator isn't reachable — it
        # may have died uncleanly. Nothing to release in the registry
        # via HTTP; idempotent no-op.
        if not quiet:
            print(
                f"prepare-for-migration: coordinator not reachable ({exc}); "
                "treating as clean — nothing to release.",
                flush=True,
            )
        return 0

    # The local resolver only reads the port + secret files, so it always
    # reports loopback. A routed coordinator listens ONLY on its bind
    # address — both the release POST and the shutdown poll below must
    # target that host or the whole path silently talks to nothing.
    if not is_loopback_host(bind_host):
        endpoint = dataclasses.replace(endpoint, host=bind_host)

    try:
        # M-04 / finding #28: use the shared _coherence_client.post() with
        # extra_headers instead of the now-deleted _post_with_operator_header().
        # Uses CLI_HTTP_TIMEOUT_SEC (6.0s) instead of the old hardcoded 5s.
        resp = _post(endpoint, "/admin/prepare-for-migration", {},
                     extra_headers={"Coherence-Local-Operator": "true"})
    except _urlerr.HTTPError as exc:
        err(f"agent-coherence-coordinator: prepare-for-migration HTTP {exc.code}")
        return 2
    except (CoordinatorUnavailable, InsecureTransportRefused) as exc:
        # Documented contract: 2 on any error reaching the coordinator.
        # Without this, a refused connection escapes as a traceback (exit 1,
        # colliding with the not-a-git-repo code) and grants stay live.
        err(f"agent-coherence-coordinator: prepare-for-migration: {exc}")
        return 2

    if not resp.get("ok"):
        err(f"agent-coherence-coordinator: prepare-for-migration returned {resp!r}")
        return 2

    released = int(resp.get("released", 0))
    errors = resp.get("errors") or []

    # Poll until the coordinator is no longer TCP-reachable. The server-
    # side schedules shutdown ~100ms after responding, then the in-flight
    # drain + serve_forever exit add a bit more — 5s budget is generous.
    cfg = LifecycleConfig()
    poll_deadline = time.monotonic() + 5.0
    while time.monotonic() < poll_deadline:
        port = _read_port_from_file(pid_file)
        if port is None or not _tcp_probe(port, cfg, bind_host=endpoint.host):
            break
        time.sleep(0.1)
    else:
        err(
            "agent-coherence-coordinator: prepare-for-migration: "
            "coordinator failed to shut down within 5s after release"
        )
        return 2

    if not quiet:
        print(
            f"prepare-for-migration: released {released} grant(s); "
            f"coordinator shutdown clean.",
            flush=True,
        )
        if errors:
            print(f"  errors: {errors}", flush=True)
    return 0


def _run_daemonized(root: Path, *, bind_host: str = "127.0.0.1") -> int:
    """Internal worker — called via ``--_daemonized``. Run the coordinator
    and block indefinitely. Exits on SIGTERM (sent by stop_coordinator)
    or natural idle-shutdown (lifecycle's idle loop also detaches via
    ``return`` in its loop, leaving the main thread to block here)."""
    port = ensure_coordinator(root, bind_host=bind_host)
    if port == -1:
        return 2

    # Block the main thread so the daemon HTTP/sweep/idle threads stay
    # alive. KP-7: use the public wait_for_shutdown API rather than
    # reaching into lifecycle._SPAWNED_REGISTRY + entry.shutdown_done.
    # The wait blocks until idle-shutdown completes, stop_coordinator
    # fires, or the shutdown sequence aborts.
    from ccs.adapters.claude_code.lifecycle import (
        wait_for_shutdown,
    )
    try:
        wait_for_shutdown(root)
    except KeyboardInterrupt:
        stop_coordinator(root)
    return 0


def _spawn_detached(root: Path, *, quiet: bool, bind_host: str = "127.0.0.1") -> int:
    """Spawn a detached child running the daemonized worker, then poll
    the port file until the child writes it (or timeout)."""
    pid_file = root / ".coherence" / "server.pid"

    # Use subprocess.Popen with start_new_session=True so the child runs
    # in its own process group and survives the parent's exit. stdio
    # redirected to /dev/null so the child doesn't keep the parent's
    # terminal busy.
    try:
        subprocess.Popen(
            [
                sys.executable, "-m", "ccs.cli.coherence_coordinator",
                "--_daemonized", "--root", str(root), "--bind-host", bind_host,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        err(f"agent-coherence-coordinator: could not spawn detached worker: {exc}")
        return 2

    cfg = LifecycleConfig()
    deadline = time.monotonic() + _DETACH_PORT_WAIT_TIMEOUT_SEC
    while time.monotonic() < deadline:
        port = _read_port_from_file(pid_file)
        if port is not None:
            if _tcp_probe(port, cfg, bind_host=bind_host):
                if not quiet:
                    print(f"port={port}", flush=True)
                return 0
            # A routed bind can never converge while a loopback-bound
            # coordinator holds the workspace flock: the child joins it via
            # the loser path and exits without binding the routed host.
            # Name the mismatch now instead of stalling out the window.
            if not is_loopback_host(bind_host) and _tcp_probe(
                port, cfg, bind_host="127.0.0.1"
            ):
                err(
                    "agent-coherence-coordinator: a coordinator for this "
                    f"workspace is already running on 127.0.0.1:{port}; stop "
                    f"it before re-binding to {bind_host}"
                )
                return 2
        time.sleep(_DETACH_PORT_WAIT_INTERVAL_SEC)

    err(
        "agent-coherence-coordinator: detached worker did not become reachable "
        f"within {_DETACH_PORT_WAIT_TIMEOUT_SEC:.0f}s"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
