# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""``agent-coherence-migrate-rules`` — propose ``permissions.deny`` entries
from prose rules in CLAUDE.md.

Per the v0.1.1 plan (Unit 9, R19): many operators have tool-class rules
written as prose in CLAUDE.md ("use rg, not grep", "never use sudo",
"avoid python -c"). The model often violates these because prose rules
are advisory at best — the runtime cannot enforce them.

``permissions.deny`` in ``.claude/settings.local.json`` is the
configuration-level enforcement equivalent. The CLI scans the workspace
CLAUDE.md for prose patterns that map cleanly to ``Bash(...)`` deny
entries and proposes the JSON to copy-paste. With ``--apply``, it writes
the entries to ``.claude/settings.local.json`` after a confirmation
prompt.

This is BEST-EFFORT pattern matching. False positives (proposing a deny
that shouldn't fire) waste operator time but cause no harm — the operator
reviews before pasting. False negatives (missing a rule) are the same as
not running the helper. The exit-codes are tuned for shell-scriptability:

- 0: completed successfully (proposals printed, or --apply succeeded,
  or no rules detected)
- 1: workspace root could not be resolved
- 2: --apply confirmation declined, its lock could not be taken, or
  settings.json could not be written
"""

from __future__ import annotations

import argparse
import contextlib
import errno
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Sequence

from ccs.adapters.claude_code.resolver import find_coordinator_root

# A blocking LOCK_EX never yields EAGAIN, and Python already retries an
# interrupted flock (PEP 475), so these retries are defensive. Any other
# errno means the lock cannot be taken, and writing without it would let a
# concurrent --apply or hand edit be clobbered, so --apply refuses instead.
_LOCK_RETRY_ERRNOS = frozenset({errno.EINTR, errno.EAGAIN})
_LOCK_ATTEMPTS = 3
_LOCK_RETRY_PAUSE_SECONDS = 0.05


@dataclass(frozen=True)
class _RuleRule:
    """One detection rule. Matches one or more regex against CLAUDE.md
    and proposes ``Bash(...)`` deny entries on hit."""

    name: str
    triggers: tuple[re.Pattern[str], ...]
    deny_entries: tuple[str, ...]
    explanation: str


def _re(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


# Detection rules — heuristic, best-effort. Order matters: more specific
# triggers should appear first so they win when both match.
_RULES: tuple[_RuleRule, ...] = (
    _RuleRule(
        name="grep_to_rg",
        triggers=(
            _re(r"use\s+rg\b"),
            _re(r"\brg\b\s+instead\s+of\s+\bgrep\b"),
            _re(r"prefer\s+\brg\b"),
            _re(r"prefer\s+ripgrep"),
        ),
        deny_entries=("Bash(grep:*)", "Bash(grep *)"),
        explanation=(
            "CLAUDE.md prefers `rg` over `grep`. The runtime cannot stop "
            "the model from invoking `grep` unless `permissions.deny` "
            "enforces it. Allow `rg` separately if not already permitted."
        ),
    ),
    _RuleRule(
        name="no_python_dash_c",
        triggers=(
            _re(r"(don'?t|do not|avoid|never)\s+use\s+python3?\s*-c"),
            _re(r"no\s+python3?\s*-c"),
        ),
        deny_entries=("Bash(python -c *)", "Bash(python3 -c *)"),
        explanation=(
            "CLAUDE.md restricts `python -c` / `python3 -c` invocations "
            "(typically for security or auditability)."
        ),
    ),
    _RuleRule(
        name="no_perl_dash_e",
        triggers=(
            _re(r"(don'?t|do not|avoid|never)\s+(use\s+)?perl\s*-e"),
            _re(r"no\s+perl\s*-e"),
        ),
        deny_entries=("Bash(perl -e *)",),
        explanation="CLAUDE.md restricts `perl -e` eval invocations.",
    ),
    _RuleRule(
        name="no_ruby_dash_e",
        triggers=(
            _re(r"(don'?t|do not|avoid|never)\s+(use\s+)?ruby\s*-e"),
            _re(r"no\s+ruby\s*-e"),
        ),
        deny_entries=("Bash(ruby -e *)",),
        explanation="CLAUDE.md restricts `ruby -e` eval invocations.",
    ),
    _RuleRule(
        name="no_sudo",
        triggers=(
            _re(r"(don'?t|do not|avoid|never)\s+use\s+sudo"),
            _re(r"no\s+sudo\b"),
            _re(r"never\s+sudo"),
        ),
        deny_entries=("Bash(sudo *)",),
        explanation=(
            "CLAUDE.md prohibits sudo invocations. `permissions.deny` "
            "is the load-bearing enforcement — prose alone cannot stop "
            "the model from suggesting sudo commands."
        ),
    ),
    _RuleRule(
        name="cat_for_files",
        triggers=(
            _re(r"(don'?t|do not|avoid|never)\s+use\s+cat\s+(to\s+read|for\s+(reading\s+)?files?)"),
            _re(r"use\s+(the\s+)?Read\s+tool[\s,]+(not|instead\s+of)\s+cat"),
        ),
        deny_entries=("Bash(cat:*)",),
        explanation=(
            "CLAUDE.md routes file reads through the Read tool. Denying "
            "`cat` at the permissions layer keeps the coherence layer "
            "informed of every read (KTD-N's H4 routing mitigation)."
        ),
    ),
    _RuleRule(
        name="sed_for_files",
        triggers=(
            _re(r"(don'?t|do not|avoid|never)\s+use\s+sed\s+(to\s+edit|for\s+editing\s+files?)"),
            _re(r"use\s+(the\s+)?Edit\s+tool[\s,]+(not|instead\s+of)\s+sed"),
        ),
        deny_entries=("Bash(sed:*)",),
        explanation=(
            "CLAUDE.md routes edits through the Edit tool. Denying "
            "`sed` at the permissions layer keeps the single-writer "
            "invariant safe."
        ),
    ),
    _RuleRule(
        name="awk_for_files",
        triggers=(
            _re(r"(don'?t|do not|avoid|never)\s+use\s+awk\s+(to\s+(read|edit)|for\s+(reading|editing)\s+files?)"),
        ),
        deny_entries=("Bash(awk:*)",),
        explanation="CLAUDE.md restricts `awk` for file reads/edits.",
    ),
)


@dataclass
class _DetectedRule:
    name: str
    matched_text: str
    deny_entries: tuple[str, ...]
    explanation: str


@dataclass
class _Report:
    claude_md_path: Path
    detected: list[_DetectedRule] = field(default_factory=list)
    # Deny entries already present in settings.local.json — we don't
    # propose duplicates.
    already_present: set[str] = field(default_factory=set)

    @property
    def proposed_entries(self) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for rule in self.detected:
            for entry in rule.deny_entries:
                if entry in self.already_present or entry in seen:
                    continue
                seen.add(entry)
                out.append(entry)
        return out


def _read_claude_md(workspace: Path) -> str | None:
    """Read CLAUDE.md from the workspace root. Returns None if absent —
    detection has nothing to do but the CLI still exits 0 to keep the
    helper safe to script in setup wizards."""
    path = workspace / "CLAUDE.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _existing_deny_entries(workspace: Path) -> set[str]:
    """Read .claude/settings.local.json (and ./settings.json for completeness)
    and return the union of existing ``permissions.deny`` entries so we
    don't propose duplicates."""
    out: set[str] = set()
    for rel in (".claude/settings.local.json", ".claude/settings.json"):
        path = workspace / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        deny = ((data or {}).get("permissions") or {}).get("deny") or []
        if isinstance(deny, list):
            out.update(str(x) for x in deny if isinstance(x, str))
    return out


def detect_rules(workspace: Path) -> _Report:
    """Run the heuristic detector against ``<workspace>/CLAUDE.md``.

    Public entry point — kept stable so the function can be re-used by
    tests and by any future ``ce-work`` automation. Workspace is the
    coordinator root (typically the repo root)."""
    report = _Report(claude_md_path=workspace / "CLAUDE.md")
    text = _read_claude_md(workspace)
    if text is None:
        return report
    report.already_present = _existing_deny_entries(workspace)
    for rule in _RULES:
        match: re.Match[str] | None = None
        for trigger in rule.triggers:
            match = trigger.search(text)
            if match is not None:
                break
        if match is None:
            continue
        # Pull a short snippet around the match for the operator
        # so they can verify the detection isn't a false positive.
        start = max(0, match.start() - 40)
        end = min(len(text), match.end() + 40)
        snippet = text[start:end].strip().replace("\n", " ")
        report.detected.append(_DetectedRule(
            name=rule.name,
            matched_text=snippet,
            deny_entries=rule.deny_entries,
            explanation=rule.explanation,
        ))
    return report


def _format_proposed_settings_json(entries: list[str]) -> str:
    """Render a minimal copy-paste-ready settings.local.json fragment."""
    return json.dumps(
        {"permissions": {"deny": entries}},
        indent=2,
        ensure_ascii=False,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-coherence-migrate-rules",
        description=(
            "Propose permissions.deny entries from prose rules in CLAUDE.md. "
            "Default is flag-only — prints proposals; use --apply to write "
            "them into .claude/settings.local.json after a confirmation prompt."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Override the workspace root (default: walk up from cwd to git root).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write proposed entries into .claude/settings.local.json "
            "after a confirmation prompt. Without this flag, the CLI "
            "prints proposals only. Requires --yes in non-interactive / "
            "piped contexts (CLR-05)."
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help=(
            "Skip the --apply confirmation prompt. Required in non-interactive "
            "contexts (CI, agent pipelines, piped stdin). Without this flag, "
            "--apply will error if stdin is not a TTY."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    root_arg = args.root if args.root is not None else find_coordinator_root()
    if root_arg is None:
        print(
            "agent-coherence-migrate-rules: not in a git repository",
            file=sys.stderr,
        )
        return 1
    root = Path(root_arg).resolve()

    report = detect_rules(root)

    if not report.claude_md_path.is_file():
        print(
            f"agent-coherence-migrate-rules: no CLAUDE.md at {report.claude_md_path}; "
            "nothing to migrate.",
            flush=True,
        )
        return 0

    if not report.detected:
        print(
            "agent-coherence-migrate-rules: no tool-class rule patterns "
            f"detected in {report.claude_md_path}.",
            flush=True,
        )
        return 0

    print("Detected tool-class rules in CLAUDE.md:")
    print()
    for rule in report.detected:
        print(f"  [{rule.name}]")
        print(f"    matched: …{rule.matched_text}…")
        print(f"    why: {rule.explanation}")
        print(f"    proposed deny: {list(rule.deny_entries)}")
        print()

    proposed = report.proposed_entries
    if not proposed:
        print(
            "All matching deny entries are already present in "
            ".claude/settings*.json — nothing to add.",
            flush=True,
        )
        return 0

    print("Proposed .claude/settings.local.json fragment:")
    print()
    print(_format_proposed_settings_json(proposed))
    print()

    if not args.apply:
        print(
            "Copy the fragment above into .claude/settings.local.json "
            "(merging with any existing permissions.deny entries), or "
            "re-run with --apply to write it automatically.",
            flush=True,
        )
        return 0

    return _apply_entries(root, proposed, prompt_for_confirmation=not args.yes)


def _lock_sidecar(fcntl_module: ModuleType, lock_path: Path) -> int:
    """Open ``lock_path`` and take a blocking exclusive lock on it.

    Returns the locked descriptor. An EINTR or EAGAIN is retried, up to
    ``_LOCK_ATTEMPTS`` attempts in all; a failed lock is retried on the
    descriptor already open, and a failed open opened nothing to close.
    Any other ``OSError``, or the last retryable one, is raised with the
    descriptor closed."""
    lock_fd: int | None = None
    attempt = 1
    while True:
        try:
            if lock_fd is None:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
            fcntl_module.flock(lock_fd, fcntl_module.LOCK_EX)
            return lock_fd
        except OSError as exc:
            if exc.errno not in _LOCK_RETRY_ERRNOS or attempt >= _LOCK_ATTEMPTS:
                if lock_fd is not None:
                    with contextlib.suppress(OSError):
                        os.close(lock_fd)
                raise
        attempt += 1
        time.sleep(_LOCK_RETRY_PAUSE_SECONDS)


def _print_lock_refusal(lock_path: Path, settings_path: Path, exc: OSError) -> None:
    errno_name = errno.errorcode.get(exc.errno, f"errno {exc.errno}")
    print(
        f"agent-coherence-migrate-rules: could not lock {lock_path} "
        f"({errno_name}: {exc.strerror or exc}); not writing {settings_path}. "
        "Remove the lock file if it is stale (no other --apply running), "
        "or copy the fragment printed above into that file by hand.",
        file=sys.stderr,
    )


def _apply_entries(
    workspace: Path, proposed: list[str], *, prompt_for_confirmation: bool
) -> int:
    """Merge ``proposed`` into ``.claude/settings.local.json``'s
    ``permissions.deny`` list and write back atomically.

    Returns 0 on success, 2 if the operator declined the prompt, if the
    sidecar lock could not be taken (the file is then neither read nor
    written), or if the file could not be written."""
    settings_path = workspace / ".claude" / "settings.local.json"
    if prompt_for_confirmation:
        # CLR-05 / finding #7: block in non-interactive context so pipelines
        # fail fast instead of hanging on stdin. Require --yes in CI/agent use.
        if not sys.stdin.isatty():
            print(
                "agent-coherence-migrate-rules: --apply requires --yes in "
                "non-interactive / piped contexts",
                file=sys.stderr,
            )
            return 2
        try:
            print(
                f"Apply {len(proposed)} deny entr"
                f"{'y' if len(proposed) == 1 else 'ies'} to {settings_path}? [y/N] ",
                end="", flush=True,
            )
            answer = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            print("agent-coherence-migrate-rules: aborted by user", file=sys.stderr)
            return 2
        if answer not in ("y", "yes"):
            print("agent-coherence-migrate-rules: not applying", file=sys.stderr)
            return 2

    settings_path.parent.mkdir(parents=True, exist_ok=True)

    # ADV-007: two operators (or two CI runs) invoking --apply concurrently
    # both read settings.local.json, both build the same proposed list,
    # both write — last write wins. With identical proposed lists the
    # outcome is functionally correct, but ANY concurrent unrelated edit
    # to settings.local.json during the window gets clobbered. Wrap the
    # read-modify-write in fcntl.flock on a sidecar lock file so two
    # --apply runs serialize.
    #
    # POSIX-only (matches the rest of the codebase per lifecycle.py
    # _FCNTL_AVAILABLE guard). Where fcntl is missing (Windows) there is no
    # lock to take, so --apply proceeds silently — the race is wider but the
    # broader Windows story is deferred to v0.1.1+ per CLAUDE.md. Where
    # fcntl exists, a lock that cannot be taken refuses (fail closed).
    try:
        import fcntl  # type: ignore[import-not-found]
    except ImportError:
        fcntl = None  # type: ignore[assignment]

    lock_path = settings_path.with_suffix(settings_path.suffix + ".lock")
    lock_fd: int | None = None
    if fcntl is not None:
        try:
            lock_fd = _lock_sidecar(fcntl, lock_path)
        except OSError as exc:
            _print_lock_refusal(lock_path, settings_path, exc)
            return 2

    try:
        if settings_path.is_file():
            try:
                data = json.loads(settings_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                print(
                    f"agent-coherence-migrate-rules: existing {settings_path} is not valid JSON ({exc}); "
                    "refusing to overwrite.",
                    file=sys.stderr,
                )
                return 2
            if not isinstance(data, dict):
                print(
                    f"agent-coherence-migrate-rules: existing {settings_path} root is not an object; "
                    "refusing to overwrite.",
                    file=sys.stderr,
                )
                return 2
        else:
            data = {}

        permissions = data.setdefault("permissions", {})
        if not isinstance(permissions, dict):
            print(
                f"agent-coherence-migrate-rules: existing permissions key in {settings_path} "
                "is not an object; refusing to overwrite.",
                file=sys.stderr,
            )
            return 2
        deny = permissions.setdefault("deny", [])
        if not isinstance(deny, list):
            print(
                f"agent-coherence-migrate-rules: existing permissions.deny in {settings_path} "
                "is not a list; refusing to overwrite.",
                file=sys.stderr,
            )
            return 2
        seen = {x for x in deny if isinstance(x, str)}
        appended = 0
        for entry in proposed:
            if entry not in seen:
                deny.append(entry)
                seen.add(entry)
                appended += 1

        # SEC-03: write-to-tmpfile + atomic rename. Plain write_text()
        # truncates the destination first; a crash mid-write leaves the
        # file partially-truncated and the next claude session sees a
        # JSONDecodeError on parse + falls back to defaults — silently
        # losing previously-applied entries. Atomic rename keeps the old
        # file intact until the new file is fully written.
        try:
            payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
            tmp_path = settings_path.with_suffix(settings_path.suffix + ".tmp")
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(str(tmp_path), str(settings_path))
        except OSError as exc:
            # Best-effort cleanup of the tmp file if rename failed.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            print(
                f"agent-coherence-migrate-rules: could not write {settings_path}: {exc}",
                file=sys.stderr,
            )
            return 2
        print(
            f"agent-coherence-migrate-rules: appended {appended} entry/ies to {settings_path}",
            flush=True,
        )
        return 0
    finally:
        # Release the lock + drop the fd. Leave the sidecar .lock file on
        # disk (it's an empty marker; cheaper to keep than to race a
        # cleanup with the next operator).
        if lock_fd is not None:
            try:
                if fcntl is not None:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
            try:
                os.close(lock_fd)
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
