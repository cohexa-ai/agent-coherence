# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-transfer``, ``-accept``, ``-decline`` and ``-withdraw``:
the four targeted-grant-handoff verbs (#185), each acting as one named Claude
Code session.

| Console script               | Coordinator route          | Run as        |
|------------------------------|----------------------------|---------------|
| ``agent-coherence-transfer`` | ``POST /handoff/transfer`` | the giver     |
| ``agent-coherence-accept``   | ``POST /handoff/accept``   | the successor |
| ``agent-coherence-decline``  | ``POST /handoff/decline``  | the successor |
| ``agent-coherence-withdraw`` | ``POST /handoff/withdraw`` | the giver     |

The session a verb acts as is ``--session``, else the harness variable
``CLAUDE_CODE_SESSION_ID``, which Claude Code sets in a session's Bash tool
shell. With neither, the verb refuses with a usage error and sends nothing. A
subagent's shell names its PARENT session, so a claim a subagent holds is
transferred with ``--subagent-id``. Every verb prints the session-level agent
id it acted as and which of the two named the session, never the raw session
id, so a shell whose variable names some other session shows it.

The verb presents the session's stored caller principal exactly as that
session's hook events do
(:func:`~ccs.cli._coherence_client.post_with_stored_principal`): for a session
with none stored it creates the session's mint nonce, then claims and stores
the principal, as the session's first hook event would. A session bound under
another mint nonce (a ``CoherentVolume`` or an MCP session, whose nonce lives
only in its process) is refused, and nothing is deleted or re-minted.

The four routes are served by the Python coordinator only. A verb learns that
this coordinator does not serve it from the pid file's backend line, with no
round trip, or else from the verb route's own 404. The claim route's answer is
never read for this: a coordinator that issues no principals may still serve
the verb.

Exit codes:

- 0: done. Every named grant transferred, or the verb was taken.
- 1: usage. A bad command line, a path that fails validation, not in a git
  repository, or no session to act as. Nothing is sent.
- 2: the coordinator is unreachable or answered any other HTTP error (a
  caller-principal refusal among them), refused the verb or a grant, or could
  not confirm the outcome.
- 4: this coordinator does not serve the verb.

3 is not used: it is the status self-test's.
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.error
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

from ccs.adapters.claude_code.coordinator_server import caller_principal_identity
from ccs.adapters.claude_code.resolver import find_coordinator_root
from ccs.cli._coherence_client import (
    NODE_BACKEND,
    CoordinatorUnavailable,
    coordinator_backend,
    err,
    http_status_from_error,
    normalize_workspace_path,
    post_with_stored_principal,
    resolve_endpoint,
)
from ccs.core.exceptions import HANDOFF_NOT_HELD_REASON, CallerPrincipalRefused, RedirectRefused

SESSION_ENV_VAR = "CLAUDE_CODE_SESSION_ID"
"""The harness variable naming the Claude Code session whose shell runs the verb."""

EXIT_DONE = 0
EXIT_USAGE = 1
EXIT_FAILED = 2
EXIT_NOT_SERVED = 4

_EPILOG = (
    "exit codes: 0 done; 1 usage (nothing sent); 2 coordinator unreachable, "
    "HTTP error, refused or unconfirmed; 4 this coordinator does not serve the verb"
)

_PYTHON_ONLY = "the handoff verbs are served by the Python coordinator only"

_NO_SESSION = (
    "no session to act as: pass --session <session id>, or run this from a "
    f"Claude Code session's shell, where {SESSION_ENV_VAR} names the session"
)

_NOT_HELD_HINT = (
    "hint: a Claude Code session's write grant ends when its turn ends. If an "
    "earlier transfer of {path} may have landed, check its handoff in "
    "agent-coherence-status first: a handoff from this session to that "
    "successor made at the version it held, live or ended, means it landed, so "
    "do not transfer again, and if it shows another session's handoff, ask "
    "before transferring; otherwise have the giver session read {path}, then "
    "transfer it again"
)
"""Printed after a ``handoff_not_held`` refusal, beside the wire reason, which
is printed unchanged. Client-side only: the coordinator's answer is the same
for a hook session and for a volume."""


@dataclass(frozen=True)
class _Session:
    """The session a verb acts as, and what named it (the flag or the variable)."""

    session_id: str
    source: str

    def acted_as(self) -> str:
        """The line naming the session-level agent id, never the session id."""
        agent_id = caller_principal_identity(self.session_id)
        return f"acting as session agent {agent_id} (session from {self.source})"


@dataclass(frozen=True)
class _Verb:
    """One verb: its name, what its request adds to ``session_id``, and how its
    answer is reported (returning the exit code)."""

    name: str
    description: str
    request: Callable[[argparse.Namespace, list[str]], dict[str, Any]]
    report: Callable[[_Verb, list[str], dict[str, Any]], int]
    taken: str = ""

    @property
    def prog(self) -> str:
        return f"agent-coherence-{self.name}"

    @property
    def route(self) -> str:
        return f"/handoff/{self.name}"

    def say(self, message: str) -> None:
        print(f"{self.prog}: {message}", flush=True)

    def complain(self, message: str) -> None:
        err(f"{self.prog}: {message}")


class _UsageParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose usage error exits 1: argparse's own exit 2 is
    what these verbs answer for a coordinator failure."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _base_parser(verb: _Verb) -> _UsageParser:
    parser = _UsageParser(prog=verb.prog, description=verb.description, epilog=_EPILOG)
    parser.add_argument(
        "--session",
        default=None,
        metavar="SESSION_ID",
        help=(
            f"The Claude Code session to act as (default: ${SESSION_ENV_VAR}, the "
            "session whose shell runs this). A subagent's shell names its parent session."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override the coordinator root (default: walk up from cwd to git root).",
    )
    return parser


def _path_parser(verb: _Verb) -> _UsageParser:
    parser = _base_parser(verb)
    parser.add_argument(
        "paths", nargs=1, metavar="path",
        help="The handed-off path, workspace-relative or absolute inside the workspace.",
    )
    return parser


def _resolve_session(flag: str | None) -> _Session | None:
    """``--session`` when given, else the harness variable, else ``None``."""
    if flag is not None:
        return _Session(flag, "--session")
    from_env = os.environ.get(SESSION_ENV_VAR)
    return _Session(from_env, SESSION_ENV_VAR) if from_env else None


def _normalized_paths(verb: _Verb, raw: list[str], root: Path) -> list[str] | None:
    """The paths in workspace-relative form, or ``None`` after reporting each
    one that fails validation."""
    normalized: list[str] = []
    rejected = False
    for path in raw:
        candidate, reason = normalize_workspace_path(path, root)
        if reason is not None:
            verb.complain(f"rejected {path!r}: {reason}")
            rejected = True
        normalized.append(candidate)
    return None if rejected else normalized


def _run(verb: _Verb, parser: _UsageParser, argv: Sequence[str] | None) -> int:
    args = parser.parse_args(argv)
    root = args.root if args.root is not None else find_coordinator_root()
    if root is None:
        verb.complain("not in a git repository")
        return EXIT_USAGE
    session = _resolve_session(args.session)
    if session is None:
        parser.print_usage(sys.stderr)
        verb.complain(_NO_SESSION)
        return EXIT_USAGE
    paths = _normalized_paths(verb, args.paths, Path(root))
    if paths is None:
        return EXIT_USAGE
    if coordinator_backend(Path(root)) == NODE_BACKEND:
        verb.complain(
            f"this coordinator does not serve {verb.name}: the workspace runs the "
            f"Node coordinator (backend={NODE_BACKEND} in its pid file); {_PYTHON_ONLY}"
        )
        return EXIT_NOT_SERVED
    verb.say(session.acted_as())
    payload = {"session_id": session.session_id, **verb.request(args, paths)}
    answer = _send(verb, Path(root), payload)
    return answer if isinstance(answer, int) else verb.report(verb, paths, answer)


def _send(verb: _Verb, root: Path, payload: dict[str, Any]) -> dict[str, Any] | int:
    """POST the verb as the session ``payload`` names, presenting its stored
    principal: the answer, or the exit code of a failure already reported.
    A principal refusal's text is built from constants and carries no
    principal or nonce."""
    try:
        endpoint = resolve_endpoint(root)
        answer = post_with_stored_principal(endpoint, root, verb.route, payload, report=verb.complain)
    except (CoordinatorUnavailable, CallerPrincipalRefused) as exc:
        verb.complain(str(exc))
        return EXIT_FAILED
    except RedirectRefused as exc:
        verb.complain(f"the coordinator redirected the request (HTTP {exc.status}); not followed")
        return EXIT_FAILED
    except urllib.error.HTTPError as exc:
        return _http_failure(verb, exc)
    if not isinstance(answer, dict):
        verb.complain("the coordinator's answer is not a JSON object")
        return EXIT_FAILED
    return answer


def _http_failure(verb: _Verb, exc: urllib.error.HTTPError) -> int:
    """A non-2xx answer to the verb route itself. Its 404 is the coordinator
    not serving the verb; the claim's 404 never reaches here (the helper reads
    it as "this coordinator issues no principals")."""
    if exc.code == 404:
        verb.complain(
            f"this coordinator does not serve {verb.name} "
            f"(POST {verb.route} answered HTTP 404); {_PYTHON_ONLY}"
        )
        return EXIT_NOT_SERVED
    if exc.code == 400:
        # The principal helper already read a 400's body to tell a principal
        # refusal from this, so its text is gone; a 400 that gets here is a
        # request the coordinator could not read.
        verb.complain(
            "HTTP 400: the coordinator could not read the request; "
            "check the session id, the subagent id and the paths"
        )
        return EXIT_FAILED
    body = http_status_from_error(exc)
    error = body.get("error") if isinstance(body, dict) else None
    verb.complain(f"HTTP {exc.code}: {error}" if isinstance(error, str) else f"HTTP {exc.code}")
    return EXIT_FAILED


def _field(answer: dict[str, Any], key: str) -> str:
    value = answer.get(key)
    return "?" if value is None else str(value)


def _report_unconfirmed(verb: _Verb, answer: dict[str, Any]) -> int:
    verb.complain(
        f"the coordinator could not confirm the {verb.name} ({_field(answer, 'reason')}); "
        "its outcome is unknown: check the path's handoff in agent-coherence-status "
        "before acting again"
    )
    return EXIT_FAILED


def _report_grant(verb: _Verb, path: str, entry: dict[str, Any] | None) -> bool:
    """Report one grant of a transfer answer; whether it transferred."""
    if entry is None:
        verb.complain(f"{path}: the answer reports no outcome for it")
        return False
    if entry.get("transferred") is True:
        verb.say(
            f"transferred {path} to {_field(entry, 'successor')} at version "
            f"{_field(entry, 'version_at_transfer')} (gave up {_field(entry, 'hold_shape')}; "
            f"status {_field(entry, 'status')})"
        )
        return True
    status = f", status {entry['status']}" if entry.get("status") is not None else ""
    verb.complain(f"{path} not transferred ({_field(entry, 'reason')}{status})")
    if entry.get("reason") == HANDOFF_NOT_HELD_REASON:
        verb.complain(_NOT_HELD_HINT.format(path=path))
    return False


def _report_transfer(verb: _Verb, paths: list[str], answer: dict[str, Any]) -> int:
    """One line per named grant, in the order named; done only when the answer
    is ``ok`` and every grant transferred."""
    if answer.get("degraded") is True:
        return _report_unconfirmed(verb, answer)
    grants = answer.get("grants")
    entries = {
        entry["path"]: entry
        for entry in (grants if isinstance(grants, list) else [])
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    every_grant = True
    for path in paths:
        every_grant = _report_grant(verb, path, entries.get(path)) and every_grant
    return EXIT_DONE if answer.get("ok") is True and every_grant else EXIT_FAILED


def _report_settled(verb: _Verb, paths: list[str], answer: dict[str, Any]) -> int:
    """An accept's, decline's or withdraw's answer: taken, or refused with its
    typed reason, each with the record's status and counterparty when given."""
    if answer.get("degraded") is True:
        return _report_unconfirmed(verb, answer)
    path = paths[0]
    details = [f"{key} {answer[key]}" for key in ("status", "counterparty") if answer.get(key) is not None]
    if answer.get("ok") is True:
        suffix = f" ({', '.join(details)})" if details else ""
        verb.say(f"{verb.taken} the handoff of {path}{suffix}")
        return EXIT_DONE
    verb.complain(f"{path}: refused ({', '.join([_field(answer, 'reason'), *details])})")
    return EXIT_FAILED


def _transfer_request(args: argparse.Namespace, paths: list[str]) -> dict[str, Any]:
    request: dict[str, Any] = {
        "successor": args.successor,
        "grants": [{"path": path} for path in paths],
    }
    if args.subagent_id is not None:
        request["agent_id"] = args.subagent_id
    return request


def _path_request(args: argparse.Namespace, paths: list[str]) -> dict[str, Any]:
    return {"path": paths[0]}


_TRANSFER = _Verb(
    "transfer",
    "Hand this session's claims on one or more paths to a named successor session.",
    _transfer_request,
    _report_transfer,
)
_ACCEPT = _Verb(
    "accept",
    "Accept, as the successor, a handoff of a path without writing it.",
    _path_request,
    _report_settled,
    taken="accepted",
)
_DECLINE = _Verb(
    "decline",
    "Decline, as the successor, a handoff of a path; the giver may write it again.",
    _path_request,
    _report_settled,
    taken="declined",
)
_WITHDRAW = _Verb(
    "withdraw",
    "Withdraw, as the giver, this session's handoff of a path.",
    _path_request,
    _report_settled,
    taken="withdrew",
)


def transfer_main(argv: Sequence[str] | None = None) -> int:
    """``agent-coherence-transfer``: hand the session's claims on the paths to
    ``--successor``."""
    parser = _base_parser(_TRANSFER)
    parser.add_argument(
        "paths", nargs="+", metavar="path",
        help="One or more paths, workspace-relative or absolute inside the workspace.",
    )
    parser.add_argument(
        "--successor", required=True, metavar="AGENT_ID",
        help="The successor's session-level agent id (hyphenated or 32 hex digits).",
    )
    parser.add_argument(
        "--subagent-id", default=None, metavar="SUBAGENT_ID",
        help=(
            "The subagent holding the claims. A subagent's shell names its parent "
            "session, so a claim a subagent holds is transferred only with this."
        ),
    )
    return _run(_TRANSFER, parser, argv)


def accept_main(argv: Sequence[str] | None = None) -> int:
    """``agent-coherence-accept``: accept a handoff as its successor."""
    return _run(_ACCEPT, _path_parser(_ACCEPT), argv)


def decline_main(argv: Sequence[str] | None = None) -> int:
    """``agent-coherence-decline``: decline a handoff as its successor."""
    return _run(_DECLINE, _path_parser(_DECLINE), argv)


def withdraw_main(argv: Sequence[str] | None = None) -> int:
    """``agent-coherence-withdraw``: withdraw a handoff as its giver."""
    return _run(_WITHDRAW, _path_parser(_WITHDRAW), argv)
