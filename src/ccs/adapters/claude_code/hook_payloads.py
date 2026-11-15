# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Typed JSON shapes for the coordinator's wire contract (KTD per Unit 4).

These TypedDicts define every request body and response shape the HTTP
coordinator accepts/emits. The cc_hook_stdin contract test in Unit 8
round-trips realistic Claude Code hook stdin payloads through this
contract; CI catches drift on every Claude Code minor version bump.

Per-invocation variation in WARN-mode templates (v0.1.1 design):
- Every additionalContext payload on the WARN-mode allow path includes a
  timestamp + last-writer session id, so the exact prose differs on every
  invocation. The original §13.5 rationale was that varied text sidesteps
  the model retry loop on identical denials. NOTE: the v0.2 Phase 0
  falsifiability experiment (see ``docs/probes/2026-05-19-ktd-e-falsifiability/REPORT.md``)
  inverted this finding for the DENY path — varied text actually WORSENS
  opus (5 retries vs 2 with static text; opus reads varied deny text as
  prompt-injection patterns and retries to disambiguate). v0.2 strict-mode
  deny therefore uses a STATIC reason template
  (``STRICT_MODE_DENY_REASON_TEMPLATE``) byte-stable across retries.

Constraint per origin §7.4 + KTD-12: the structured ``summary`` metadata
NEVER includes raw file content, content hashes themselves, diff text, or
content-derived data. Only path / version / session-id / timestamp.

v0.2 KTD-U structural invariant (security):
- ``TERMINAL_DENIAL_CLASSES`` enumerates denial classes that MUST NEVER be
  converted to ``permissionDecision: "allow"``. Every allow-emission path
  routes through ``emit_allow()`` which asserts membership; tests in
  ``tests/integration/test_strict_mode.py`` parametrize over the
  call-site list and a meta-test grep-counts call sites in this file +
  ``coordinator_server.py`` to force list extension on every new allow
  path. See plan Unit 2 (KTD-P, KTD-Q, KTD-U).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Literal, NotRequired, TypedDict

from ccs.core.types import (
    TRANSFER_STATUS_COMPLETED,
    TRANSFER_STATUS_DECLINED,
    TRANSFER_STATUS_OVERTAKEN,
    TRANSFER_STATUS_PENDING,
    TRANSFER_STATUS_WITHDRAWN,
)

# ----------------------------------------------------------------------
# v0.2 strict-mode helpers — KTD-P (static deny text), KTD-U (terminal
# denial invariant)
# ----------------------------------------------------------------------


HANDOFF_GIVER_DENIAL_CLASS: str = "permissions_deny_handoff_giver"
"""#185: the giver of a live handoff, denied on pre-edit in strict and warn
mode alike (:func:`emit_handoff_giver_deny`). Terminal: no allow path may
convert it."""

HELD_PATH_DENIAL_CLASS: str = "permissions_deny_held_path"
"""An opted-in edit refused because another agent holds the path, under the
holder text and the parent text alike (:func:`emit_held_path_deny`,
:func:`emit_held_by_parent_deny`). Terminal: no allow path may convert it."""

ADMISSION_UNAVAILABLE_DENIAL_CLASS: str = "permissions_deny_admission_unavailable"
"""An opted-in edit refused because the coordinator could not confirm that no
other agent holds the path (:func:`emit_admission_unavailable_deny`).
Terminal: admitting an edit nobody checked would let it displace the very
holder the opt-in was sent to protect."""

TERMINAL_DENIAL_CLASSES: frozenset[str] = frozenset({
    "permissions_deny_strict_mode",
    HANDOFF_GIVER_DENIAL_CLASS,
    HELD_PATH_DENIAL_CLASS,
    ADMISSION_UNAVAILABLE_DENIAL_CLASS,
})
"""KTD-U security invariant: denial classes that MUST NEVER be converted to
``permissionDecision: "allow"``. Any code path emitting allow checks
membership via :func:`emit_allow`; passing a terminal class raises
AssertionError. Tests at ``tests/integration/test_strict_mode.py``
parametrize the call-site list and meta-test the count so a future
contributor adding a new allow path is forced to extend the parameter list
(and therefore consider the invariant) rather than satisfying the test
trivially. Adding a new terminal class extends the boundary; never remove
an entry without a security review."""


_READ_AGAIN_SENTENCE: str = (
    "Read {path} again with the Read tool: the next Read returns the current "
    "version, and edits of {path} are denied until you have read it."
)
"""What follows the first sentence of both stale-copy denies (#514). The
session's next Read of the file is admitted when its bytes are the recorded
content, so the text says so; an Edit sent first is still denied. It also
holds after a strict Bash or Grep deny, whose re-grant (#242) admits the next
Read on the fresh arm. Both coordinators emit these bytes."""

STRICT_MODE_DENY_REASON_TEMPLATE: str = (
    "Stale read denied: {path} was updated by agent {last_writer_short} "
    "at {last_writer_ts_iso}. " + _READ_AGAIN_SENTENCE
)
"""KTD-P static deny text for the arm where a peer really did commit.
Byte-stable across retries of the same (session, artifact) staleness event
because every substitution is deterministic per-artifact (path),
per-preempter (last_writer_short), or per-commit-tick (last_writer_ts_iso).
The Phase 0 H1 falsification proved varied deny text WORSENS opus behavior
(5 retries vs 2 with static text); this template guards against accidental
re-introduction of per-invocation fields. The format-string placeholder set
is locked by ``test_strict_mode_deny_reason_template_is_static``.

It no longer says a retry gets the same denial (#514): the session's next
Read is admitted, which is what the text now tells it.

``last_writer_short`` shortens an AGENT id (R7): the registry stores
``artifacts.last_writer_id`` as an agent id and the response now renders it
as-is instead of mapping it back to the session id it was derived from."""


GRANT_CHANGE_DENY_REASON_TEMPLATE: str = (
    "Stale read denied: your grant on {path} was revoked and no new version "
    "was committed — {path} is still at v{current_version}. "
    + _READ_AGAIN_SENTENCE
)
"""The deny text for the arm where nothing was written (R8).

Cohexa-ai/agent-coherence#196: a peer's ``pre-edit`` invalidates a live
holder WITHOUT committing. The holder's next read was denied with "was
updated by session <unknown> at <t>" — a write that never happened, named
against a writer that does not exist, at a timestamp when nothing was
written. :func:`summary_reports_a_write` picks between the two templates.

Carries no timestamp at all: a revocation the summary can see has no
event tick of its own (``last_writer_at_unix_ts`` is the last real commit,
which is not what happened here), and the version is the honest thing to
report. That makes this arm byte-stable for the same reason the other one
is — every substitution is per-artifact."""


OUT_OF_BAND_DENY_REASON_TEMPLATE: str = (
    "Stale read denied: {path} on disk no longer matches v{current_version}, "
    "the version you last read, and no session committed the change, so it "
    "was made outside the coordinator. Do not restore or overwrite {path}: "
    "the change may be your user's work. Tell your user that {path} changed "
    "outside the coordinator; the read stays denied until the change is "
    "committed through the coordinator. This denial is structural (v0.2 "
    "strict mode); retrying the same operation will produce the same denial."
)
"""The deny text for a change made outside the coordinator (#278).

The reader still holds a valid grant on the current version, yet the bytes it
just hashed differ from the recorded content: no commit explains them. The
write template named the last committer at the time of its commit -- a write
that did not make this change, at a time it was not made, or
``agent <unknown>`` when nothing was ever committed. Only the SHARED-holder
arm picks this text (``out_of_band`` on :func:`emit_strict_deny`); a reader
with no valid grant whose bytes differ gets
:data:`STALE_OUT_OF_BAND_DENY_REASON_TEMPLATE` or
:data:`WRITE_IN_FLIGHT_DENY_REASON_TEMPLATE` instead. Re-reading does not
clear the deny, and the one move left to the model -- putting the old bytes
back -- would discard a change that may be the user's own edit, so the text
says to leave the file alone and report it. Per-artifact placeholders only, so
it is byte-stable."""

STALE_OUT_OF_BAND_DENY_REASON_TEMPLATE: str = (
    "Stale read denied: {path} on disk does not match v{current_version}, "
    "the last version committed through the coordinator, and no session "
    "committed the change, so it was made outside the coordinator. Do not "
    "restore or overwrite {path}: the change may be your user's work. Tell "
    "your user that {path} changed outside the coordinator; the read stays "
    "denied until the change is committed through the coordinator. This "
    "denial is structural (v0.2 strict mode); retrying the same operation "
    "will produce the same denial."
)
"""The out-of-band text for a reader with no valid grant (#514): its copy is
stale, or it never read the file, and the bytes it hashed are not the recorded
content, or it sent no hash because the file is gone, while no other session
holds an uncommitted write. A Read of those bytes is not admitted, so the text
promises none and names what clears the deny: a commit through the
coordinator. Unlike :data:`OUT_OF_BAND_DENY_REASON_TEMPLATE` it does not say
"the version you last read", which a never-read session never had, and it
names no writer, where the commit text named ``agent <unknown>`` for a write
nobody committed. Per-artifact placeholders only, so it is byte-stable."""

WRITE_IN_FLIGHT_DENY_REASON_TEMPLATE: str = (
    "Stale read denied: another agent is writing {path} and has not "
    "committed it yet, so {path} on disk does not match v{current_version}, "
    "the last version committed through the coordinator. Wait for that write "
    "to be committed, then Read {path} again with the Read tool."
)
"""The same reader's text when another session holds an uncommitted write on
the file (#514): its bytes can reach the disk before its commit does, which
is the coordinator's own write in flight, not a change made outside it. The
commit is what clears the deny, so the text says to wait for it and then read
again. Per-artifact placeholders only, so it is byte-stable."""

OUT_OF_BAND_CHANGE_NOTICE_TEMPLATE: str = (
    "⚠ {path} on disk differs from v{version}, the last content committed "
    "through the coordinator, and no session committed the change: it was "
    "made outside the coordinator. Read {path} before relying on it, and do "
    "not write it from an earlier read; that write would replace the change."
)
"""The warn-mode counterpart, attached to an admitted read as context only
(#278). The fresh answer's ``hash_differs`` flag is a top-level key the model
never sees, so without this sentence a reader holding a valid grant was told
nothing when its file changed underneath it."""


def emit_allow(
    *,
    source: str,
    additional_context: str | None = None,
    denial_class: str | None = None,
) -> dict[str, Any]:
    """Build the ``hookSpecificOutput`` envelope for a ``permissionDecision:
    "allow"`` response. ALL allow emissions route through this helper.

    KTD-U enforcement: if ``denial_class`` is in :data:`TERMINAL_DENIAL_CLASSES`,
    raises ``AssertionError`` with a diagnostic naming the source call site.
    A code path that knows it's converting a terminal-class denial back to
    allow cannot satisfy this check; that's the structural invariant.

    Args:
        source: short identifier of the call site (e.g. ``"pre_read_fresh_with_notice"``).
            Used by the KTD-U meta-test for parameter-list coverage and the
            AssertionError diagnostic.
        additional_context: optional ``additionalContext`` prose. Omitted when
            None — the model gets a quiet allow.
        denial_class: optional denial classification the caller is converting
            to allow. Almost always ``None`` for legitimate allow paths; the
            argument exists so tests can synthesize "this caller refuses to
            convert TERMINAL_DENIAL_CLASSES inputs to allow."
    """
    assert denial_class not in TERMINAL_DENIAL_CLASSES, (
        f"emit_allow(source={source!r}, denial_class={denial_class!r}): "
        f"refused to convert TERMINAL_DENIAL_CLASSES member to allow. "
        f"This is the KTD-U security invariant — strict-mode denials are "
        f"structurally terminal."
    )
    out: dict[str, Any] = {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",
    }
    if additional_context is not None:
        out["additionalContext"] = additional_context
    return out


def short_session_id(identity_id: str) -> str:
    """The 8-char short form of an identity handle, EXCEPT for a ``<...>``
    sentinel.

    A placeholder like ``"<unknown>"`` is prose, not an identifier: slicing it
    to 8 chars drops the closing angle bracket and ships malformed text
    ("<unknown"). Real handles are 36-char session UUIDs or 32-char agent-id
    hex, so an 8-char prefix is unambiguous whenever one is present. Every
    renderer that shortens an identity handle for prose goes through here —
    the guard used to live in :func:`emit_strict_deny` alone, and the two
    warn-mode renderers sliced the sentinel.

    The argument is named for what it is rather than for what it was: since
    R7 the hook responses feed this AGENT ids, not session ids. The function
    name is kept because it is the cross-backend pair of Node's
    ``shortSessionId`` and renaming it would move bytes in neither runtime's
    output while touching both.
    """
    if identity_id.startswith("<") and identity_id.endswith(">"):
        return identity_id
    return identity_id[:8]


def summary_reports_a_write(summary: "StaleSummary") -> bool:
    """Does this summary support the claim that the artifact was WRITTEN?

    R8. Three admitting cases, one refusing one:

    - ``prior_version_seen_by_session is None`` — the session never observed
      this artifact, so there is no grant of its own that could have changed
      hands and no baseline to call unchanged. Whatever is recorded is the
      only story available.
    - ``hash_differs`` — the bytes the caller just hashed differ from the
      coordinator's recorded content. Something was written, in-band or out.
    - ``current_version > prior_version_seen_by_session`` — a version landed
      past the one this session last saw. That is a commit.

    Otherwise the version this session observed is still the current one and
    its bytes still match: nothing was written, and the only thing that moved
    is the grant (Cohexa-ai/agent-coherence#196). Both branches are pinned by
    tests, because a predicate asserted only in the admitting direction is
    indistinguishable from one that always admits — which is what the single
    template this replaces effectively was.
    """
    prior = summary.get("prior_version_seen_by_session")
    if prior is None:
        return True
    if summary["hash_differs"]:
        return True
    return summary["current_version"] > prior


def summary_reports_an_out_of_band_change(summary: "StaleSummary") -> bool:
    """Were the bytes changed with no commit behind them? (#278)

    The session observed a version that is still the current one, and the
    bytes it just hashed differ from the recorded content. A commit would have
    moved the version; a grant changing hands would not have moved the bytes.
    What is left is a change made outside the coordinator. A session that never
    observed the artifact has no version of its own to compare, so this never
    holds for it."""
    prior = summary.get("prior_version_seen_by_session")
    return (
        prior is not None
        and summary["hash_differs"]
        and summary["current_version"] == prior
    )


def emit_strict_deny(
    *,
    source: str,
    summary: "StaleSummary",
    out_of_band: bool = False,
) -> dict[str, Any]:
    """Build the ``hookSpecificOutput`` envelope for a v0.2 strict-mode
    deny response.

    The ``permissionDecisionReason`` is rendered via
    :data:`OUT_OF_BAND_DENY_REASON_TEMPLATE` when the caller denies a reader
    that holds a valid grant (``out_of_band``) and the bytes changed with no
    commit behind them (#278), else :data:`GRANT_CHANGE_DENY_REASON_TEMPLATE`
    when the summary cannot support a write claim (R8), else
    :data:`STRICT_MODE_DENY_REASON_TEMPLATE`. The summary alone cannot pick the
    first: a reader a peer's acquire revoked can show the same version and a
    differing hash, and that change went through the coordinator. All three
    are static and byte-stable across retries per KTD-P. The ``source``
    argument is preserved for telemetry (Unit 4 audit-log append) and
    parameter-list parity with :func:`emit_allow`.
    """
    if out_of_band and summary_reports_an_out_of_band_change(summary):
        return {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": OUT_OF_BAND_DENY_REASON_TEMPLATE.format(
                path=summary["path"],
                current_version=summary["current_version"],
            ),
        }
    if not summary_reports_a_write(summary):
        reason = GRANT_CHANGE_DENY_REASON_TEMPLATE.format(
            path=summary["path"],
            current_version=summary["current_version"],
        )
        return {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    last_writer_full = summary.get("last_writer_session_id") or "<unknown>"
    last_writer_short = short_session_id(last_writer_full)
    last_writer_ts_iso = datetime.fromtimestamp(
        summary["last_writer_at_unix_ts"], tz=timezone.utc
    ).isoformat()
    reason = STRICT_MODE_DENY_REASON_TEMPLATE.format(
        path=summary["path"],
        last_writer_short=last_writer_short,
        last_writer_ts_iso=last_writer_ts_iso,
    )
    return {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }


def emit_disk_mismatch_deny(
    *, path: str, current_version: int, write_in_flight: bool
) -> dict[str, Any]:
    """The pre-read strict deny for a reader with no valid grant whose bytes
    are not the recorded content (#514): its copy is stale, or it never read
    the file. Reading again does not clear this deny, so the text names what
    does: :data:`WRITE_IN_FLIGHT_DENY_REASON_TEMPLATE` when another session
    holds an uncommitted write, else
    :data:`STALE_OUT_OF_BAND_DENY_REASON_TEMPLATE`. The caller decides both
    facts, since neither is in the summary."""
    template = (
        WRITE_IN_FLIGHT_DENY_REASON_TEMPLATE
        if write_in_flight
        else STALE_OUT_OF_BAND_DENY_REASON_TEMPLATE
    )
    return {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": template.format(
            path=path, current_version=current_version,
        ),
    }


# ----------------------------------------------------------------------
# Held-path denies: the hook client's answers to a strict edit it opted in
# to acquire-or-fail
# ----------------------------------------------------------------------
#
# Static like the strict templates, for the same measured reason, and pinned
# by the fixtures in tests/fixtures/hook_on_held/ so another client can
# reproduce them byte for byte, so ASCII only. Each reads as a sentence behind
# Claude Code's "PreToolUse:Edit hook error: " prefix. None names a setting,
# file, command or shell route: a refusal that hands the model its way around
# is a self-bypass the retry loop would take. None says retrying gives the
# same denial, because a held file does come free.

HELD_PATH_DENY_REASON_TEMPLATE: str = (
    "Edit denied: {path} is held by agent {holder}. It comes free when that "
    "agent lets go of it, or at the latest once that agent has held it for 30 "
    "minutes. Do not retry now: do other work that does not need {path}, or "
    "stop and tell your user. If you hold {path} yourself through another "
    "tool, such as an MCP server, release it there first."
)
"""Another agent holds the path. ``holder`` is the holder's short agent id,
never a session id. The text names both ways the file comes free and gives a
window rather than a moment, and never a "turn", which one session cannot see
in another. The window's end is the one both coordinators keep (#628): the
Python coordinator's sweep frees a grant its holder has held for
``grant_max_hold_sec`` (30 minutes by default), or sooner once the holder has
been quiet for ``grant_heartbeat_timeout_sec`` (10 minutes); the Node
coordinator runs no timer, and gives an opted-in edit a grant its holder has
held for 30 minutes, so the refused session's next edit after that gets the
file. The 10 minutes are left out because the Node coordinator does not keep
them. The last sentence is there because a hold this session took through
another client is a different agent to the coordinator, so a session can be
refused by its own hold."""

HELD_BY_PARENT_DENY_REASON_TEMPLATE: str = (
    "Edit denied: {path} is held by the session that started you. Do not wait "
    "for it: finish without this change and report the change you wanted to "
    "make back to that session."
)
"""A subagent refused by its own parent's hold. The parent is waiting on the
subagent, so it does not let go of the path while the subagent runs and
waiting would never end; the subagent hands the change back instead. It shares
:data:`HELD_PATH_DENIAL_CLASS` with the holder text."""

ADMISSION_UNAVAILABLE_DENY_REASON_TEMPLATE: str = (
    "Edit denied: the coordinator could not confirm that no other agent holds "
    "{path}. Try the edit once more; if it is denied again, stop and tell "
    "your user."
)
"""Every other answer to an opted-in edit, and no answer at all: one text that
stays true whatever the cause -- a timeout, a coordinator that went away, an
answer in a shape the client does not know, or a request that could not be
sent as it was. It never claims the edit left nothing behind, because a
timed-out request was measured landing late on both coordinators. A strict
edit refused because the coordinator cannot be reached should reuse this text
and its class, so one state has one deny."""


def _pretooluse_deny(reason: str) -> dict[str, Any]:
    """The deny envelope the strict deny uses: the decision and its reason,
    with no top-level ``reason`` or ``decision`` beside it."""
    return {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }


def emit_held_path_deny(*, path: str, holder_agent_id: str) -> dict[str, Any]:
    """The ``hookSpecificOutput`` deny for an edit of ``path`` refused because
    another agent holds it, naming ``holder_agent_id`` (the refusal's 32-hex
    holder) by its short form. Class :data:`HELD_PATH_DENIAL_CLASS`."""
    return _pretooluse_deny(HELD_PATH_DENY_REASON_TEMPLATE.format(
        path=path, holder=short_session_id(holder_agent_id),
    ))


def emit_held_by_parent_deny(*, path: str) -> dict[str, Any]:
    """The ``hookSpecificOutput`` deny for a subagent's edit of ``path``
    refused by its own parent's hold. Class :data:`HELD_PATH_DENIAL_CLASS`."""
    return _pretooluse_deny(HELD_BY_PARENT_DENY_REASON_TEMPLATE.format(path=path))


def emit_admission_unavailable_deny(*, path: str) -> dict[str, Any]:
    """The ``hookSpecificOutput`` deny for an opted-in edit of ``path`` whose
    answer was not an admit, a deny to relay or a holder refusal. Class
    :data:`ADMISSION_UNAVAILABLE_DENIAL_CLASS`."""
    return _pretooluse_deny(ADMISSION_UNAVAILABLE_DENY_REASON_TEMPLATE.format(path=path))


# ----------------------------------------------------------------------
# Targeted grant handoff (#185): the giver's deny and the handoff prose
# ----------------------------------------------------------------------
#
# Byte-stable like the strict templates: every placeholder is a short agent
# id, a path or a version, so the corpus can pin the rendered bytes.
# None of this prose names a console script or invokes the withdraw verb:
# withdraw is named only as the giver's exit taken on its user's or host's
# instruction, because a terminal deny that hands the model the command lifting
# it is a self-bypass the measured retry loop would take. Python-only:
# the Node backend serves no handoff, so there is no mirror to keep in step.

_HANDOFF_FENCED_CLAUSE = (
    "you handed {path} to agent {successor_short} at v{version_at_transfer}, "
    "so this session can no longer write it."
)
_HANDOFF_EXITS_SENTENCE = (
    "The handoff ends when agent {successor_short} writes {path}, when agent "
    "{successor_short} declines it, or when you withdraw it on your user's or "
    "host's instruction."
)
_HANDOFF_STOP_SENTENCE = "Stop and report to your user that {path} was handed off."

HANDOFF_GIVER_DENY_REASON_TEMPLATE: str = (
    "Edit denied: " + _HANDOFF_FENCED_CLAUSE + " "
    + _HANDOFF_EXITS_SENTENCE + " "
    + _HANDOFF_STOP_SENTENCE + " "
    + "This denial is structural; retrying the same operation will produce "
    + "the same denial."
)
"""The giver's pre-edit deny. Placeholders: ``path``,
``successor_short`` and ``version_at_transfer`` only, locked by
``test_handoff_giver_deny_reason_template_is_static``. It names the three exits
-- the successor writes, the successor declines, the giver withdraws on its
user's or host's instruction -- and tells the giver to stop and report, so the
model has a next action that is not a retry."""

HANDOFF_GIVER_READ_TEMPLATE: str = (
    "Handoff: " + _HANDOFF_FENCED_CLAUSE + " "
    + _HANDOFF_EXITS_SENTENCE + " "
    + "Until the handoff ends, do not change {path} by any route, a shell "
    + "command included. "
    + _HANDOFF_STOP_SENTENCE
)
"""The live giver's admitted pre-read: context-only prose, appended after any
stale warning or notice, so the generic "Re-acquire before writing" advice is
followed by the specific statement that this session handed the path off. A
read is not fenced and no hook stops a shell write: in a live measurement a
giver took a silent re-read as an all-clear and appended to the file through
the shell. The deny's fenced clause, exits and stop sentences, plus one
sentence ruling out every route; same placeholders as the deny."""

HANDOFF_EDIT_ON_DISK_SENTENCE: str = (
    "Your edit landed in your local worktree but was not given a version by "
    "the coordinator."
)
"""The post-edit handoff arm's statement of fact, as the preemption and
reclaim arms already make it -- no instruction to revert or re-edit. It is on
the post-edit arm only: the pre-edit deny lands nothing, and the
compare-and-swap refusal writes nothing before a win."""

HANDOFF_GIVER_COMMIT_REFUSED_TEMPLATE: str = (
    "Commit refused: " + _HANDOFF_FENCED_CLAUSE + " "
    + HANDOFF_EDIT_ON_DISK_SENTENCE + " "
    + _HANDOFF_EXITS_SENTENCE + " "
    + _HANDOFF_STOP_SENTENCE
)
"""The post-edit handoff arm: the giver's in-flight edit reported after
the transfer landed. Delivered through the post-tool context envelope
(:func:`emit_posttooluse_context`), beside the typed reason. Same placeholders
as the deny."""

HANDOFF_PROVENANCE_TEMPLATE: str = (
    "Handoff: agent {giver_short} handed {path} to this session at "
    "v{version_at_transfer}; the hold it gave up was {hold_shape}."
)
"""The successor's provenance -- by which agent, at which version, from
which hold shape (``EXCLUSIVE``, ``MODIFIED`` or ``SHARED``)."""

HANDOFF_UNCOMMITTED_CLAIM_TEMPLATE: str = (
    "Agent {giver_short} held an uncommitted write claim when it handed {path} "
    "on, so the file on disk may differ from v{version_at_transfer}; read "
    "{path} before editing it."
)
"""Added to the provenance for an ``EXCLUSIVE`` hold shape only. Worded
as a possibility: an EXCLUSIVE holder may or may not have written to disk, and
the coordinator cannot tell. ``MODIFIED`` committed what it wrote, and
``SHARED`` never held a write claim."""

HANDOFF_READ_FIRST_TEMPLATE: str = (
    "⚠ You have not read {path} at v{version_at_transfer} or later; read it "
    "before editing."
)
"""The successor's acquire, when its last observed version on the path is
absent or below the version at transfer. A warning, never a deny: the
strict-mode gate stays INVALID-only, so a fresh successor is warned."""

HANDOFF_OVERTAKEN_TEMPLATE: str = (
    "Handoff overtaken: the handoff of {path} to this session from agent "
    "{giver_short} at v{version_at_transfer} was overtaken by agent "
    "{counterparty_short}."
)
"""The successor's touch after a bystander acquired or wrote past
the handoff."""

HANDOFF_BYSTANDER_ADVISORY_TEMPLATE: str = (
    "Handoff in progress: agent {giver_short} handed {path} to agent "
    "{successor_short} at v{version_at_transfer}. This session is not a party "
    "to it; its edits are admitted and are recorded as overtaking the handoff."
)
"""A third session is never refused because of a live handoff; it is
told whose handoff it is writing past."""

HANDOFF_ENDED_UNRECORDED_TEMPLATE: str = (
    "Handoff ended: the handoff of {path} at v{version_at_transfer} was ended "
    "by a write at a later version whose writer was not recorded."
)
"""One sentence, for the giver and the successor alike, for a record that
is no longer live while its status is still ``pending`` -- the version moved
and nothing labelled the record (a crash, or a failed label write, between a
win and its completion)."""

_HANDOFF_ENDED_PREFIX = (
    "Handoff ended: your handoff of {path} to agent {successor_short} at "
    "v{version_at_transfer} "
)
HANDOFF_GIVER_OUTCOME_TEMPLATES: dict[str, str] = {
    TRANSFER_STATUS_COMPLETED: (
        _HANDOFF_ENDED_PREFIX + "was completed by agent {successor_short}."
    ),
    TRANSFER_STATUS_OVERTAKEN: (
        _HANDOFF_ENDED_PREFIX + "was overtaken by agent {counterparty_short}."
    ),
    TRANSFER_STATUS_DECLINED: (
        _HANDOFF_ENDED_PREFIX + "was declined by agent {successor_short}."
    ),
    TRANSFER_STATUS_WITHDRAWN: _HANDOFF_ENDED_PREFIX + "was withdrawn.",
}
"""The giver's next touch after its handoff ended reports how it ended,
one template per ended status. A record still pending when it ended renders
:data:`HANDOFF_ENDED_UNRECORDED_TEMPLATE` instead."""

#: The caller's role in a record, as the ``handoff`` key's ``role`` says it.
#: Wire values: add, never rename.
HANDOFF_ROLE_GIVER = "giver"
HANDOFF_ROLE_SUCCESSOR = "successor"
HANDOFF_ROLE_BYSTANDER = "bystander"


class HandoffKey(TypedDict):
    """The ``handoff`` key a pre-read, pre-edit, post-edit or compare-and-swap
    body carries while its path has a transfer record: the record projected
    for the caller's role, session-level agent ids only (hyphenated)."""

    role: Literal["giver", "successor", "bystander"]
    giver: str
    successor: str
    version_at_transfer: int
    hold_shape: Literal["EXCLUSIVE", "MODIFIED", "SHARED"]
    status: str
    live: bool
    counterparty: NotRequired[str]
    outcome: NotRequired[str]


def emit_handoff_giver_deny(
    *,
    source: str,
    path: str,
    successor_id: str,
    version_at_transfer: int,
) -> dict[str, Any]:
    """Build the ``hookSpecificOutput`` deny envelope for the giver of a live
    handoff, the strict-deny-style builder of the third template.

    The reason renders :data:`HANDOFF_GIVER_DENY_REASON_TEMPLATE`, static and
    byte-stable across retries and across strict and warn mode. Its class is
    :data:`HANDOFF_GIVER_DENIAL_CLASS`. ``source`` is kept for parity with
    :func:`emit_strict_deny` and :func:`emit_allow`."""
    reason = HANDOFF_GIVER_DENY_REASON_TEMPLATE.format(
        path=path,
        successor_short=short_session_id(successor_id),
        version_at_transfer=version_at_transfer,
    )
    return {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": reason,
    }


def handoff_commit_refused_text(
    *, path: str, successor_id: str, version_at_transfer: int
) -> str:
    """The post-edit handoff arm's prose (:data:`HANDOFF_GIVER_COMMIT_REFUSED_TEMPLATE`)."""
    return HANDOFF_GIVER_COMMIT_REFUSED_TEMPLATE.format(
        path=path,
        successor_short=short_session_id(successor_id),
        version_at_transfer=version_at_transfer,
    )


def handoff_context_text(
    handoff: HandoffKey, *, path: str, unread: bool = False
) -> str | None:
    """The context-only prose for ``handoff``, the key a body carries for its
    caller, or ``None`` when the caller's role has nothing to be told.

    - successor: the provenance while the record is live and not overtaken
      (plus the read-before-editing sentence for an EXCLUSIVE hold, and the
      read-first warning when ``unread``), the overtaken sentence once a
      bystander took the path;
    - giver: while the record is live, that it handed the path off and may
      not change it by any route -- reached on an admitted pre-read, since a
      live giver's pre-edit is answered by its deny, which the seam leaves
      untouched; the outcome once the record is no longer live;
    - bystander: the advisory naming giver and successor, while it is live.

    ``unread`` is the successor's acquire whose last observed version on the
    path was absent or below the version at transfer. A record no longer
    live while still ``pending`` renders the one unrecorded-writer sentence for
    giver and successor alike."""
    if handoff["role"] == HANDOFF_ROLE_BYSTANDER:
        return _bystander_advisory(handoff, path) if handoff["live"] else None
    if not handoff["live"] and handoff["status"] == TRANSFER_STATUS_PENDING:
        return HANDOFF_ENDED_UNRECORDED_TEMPLATE.format(
            path=path, version_at_transfer=handoff["version_at_transfer"]
        )
    if handoff["role"] == HANDOFF_ROLE_GIVER:
        if handoff["live"]:
            return HANDOFF_GIVER_READ_TEMPLATE.format(**_handoff_fields(handoff, path))
        return _giver_outcome(handoff, path)
    return _successor_context(handoff, path, unread=unread)


def _handoff_fields(handoff: HandoffKey, path: str) -> dict[str, Any]:
    """The placeholder values every handoff template draws from: short ids,
    the path and the version, nothing else."""
    counterparty = handoff.get("counterparty")
    return {
        "path": path,
        "giver_short": short_session_id(handoff["giver"]),
        "successor_short": short_session_id(handoff["successor"]),
        "counterparty_short": short_session_id(counterparty) if counterparty else "",
        "version_at_transfer": handoff["version_at_transfer"],
        "hold_shape": handoff["hold_shape"],
    }


def _bystander_advisory(handoff: HandoffKey, path: str) -> str:
    return HANDOFF_BYSTANDER_ADVISORY_TEMPLATE.format(**_handoff_fields(handoff, path))


def _giver_outcome(handoff: HandoffKey, path: str) -> str | None:
    template = HANDOFF_GIVER_OUTCOME_TEMPLATES.get(handoff["status"])
    return template.format(**_handoff_fields(handoff, path)) if template else None


def _successor_context(handoff: HandoffKey, path: str, *, unread: bool) -> str | None:
    fields = _handoff_fields(handoff, path)
    if handoff["status"] == TRANSFER_STATUS_OVERTAKEN:
        return HANDOFF_OVERTAKEN_TEMPLATE.format(**fields)
    if not handoff["live"]:
        return None
    sentences = [HANDOFF_PROVENANCE_TEMPLATE.format(**fields)]
    if handoff["hold_shape"] == "EXCLUSIVE":
        sentences.append(HANDOFF_UNCOMMITTED_CLAIM_TEMPLATE.format(**fields))
    if unread:
        sentences.append(HANDOFF_READ_FIRST_TEMPLATE.format(**fields))
    return " ".join(sentences)


# ----------------------------------------------------------------------
# Request bodies (from hook handlers → coordinator)
# ----------------------------------------------------------------------


class PreReadRequest(TypedDict):
    session_id: str
    path: str  # parent-repo-relative; KTD-7 normalization happens client-side
    content_hash: NotRequired[str]


class PreEditRequest(TypedDict):
    session_id: str
    path: str


class PostEditRequest(TypedDict):
    session_id: str
    path: str
    content_hash: str
    success: bool


class SessionStopRequest(TypedDict):
    session_id: str


class PolicyTrackRequest(TypedDict):
    paths: list[str]


class PolicyUntrackRequest(TypedDict):
    paths: list[str]


# ----------------------------------------------------------------------
# Response bodies (coordinator → hook handler)
# ----------------------------------------------------------------------


class StaleSummary(TypedDict):
    """Structured stale-read metadata. NEVER includes content/hash bytes.

    Two timestamps serve different purposes:
    - `last_writer_at_unix_ts` is from the registry's `artifacts.updated_at`
      (the REAL commit wall-clock); semantically honest.
    - `warning_generated_at_unix_ts` is `now()` at handler time; provides
      structural per-invocation variation that survives the case where two
      reads of the same stale state somehow occur (defense-in-depth against
      the §13.5 retry-loop hazard once v0.2 strict mode flips allow→deny).
    """
    path: str
    current_version: int
    #: The version this session last actually OBSERVED, read from the
    #: registry's per-agent ``last_observed_version`` rather than inferred as
    #: ``current_version - 1``. The inference assumed the invalidation came
    #: from a commit; when a peer merely took the grant, it reported a
    #: version the session never saw and made an unchanged version look
    #: changed (Cohexa-ai/agent-coherence#196). ``None`` = never observed.
    prior_version_seen_by_session: int | None
    #: The writer's AGENT id (``artifacts.last_writer_id``), or the literal
    #: ``<unknown>`` when nothing has been committed. Named for the session
    #: id it used to carry; the wire key is kept so hook scripts, the CLI and
    #: the recorded corpus keep parsing by exact shape.
    last_writer_session_id: str
    last_writer_at_unix_ts: float
    warning_generated_at_unix_ts: float
    hash_differs: bool


class FreshResponse(TypedDict):
    """Fresh-read response shape.

    AC-08: ``hookSpecificOutput`` is OPTIONAL and present when the
    session has pending preemption notices that the wrapper at
    ``_handle_pre_read.work_with_notice_surfacing`` attaches. Typed
    consumers should treat ``hookSpecificOutput`` as ``NotRequired``
    on every endpoint that returns a fresh envelope (pre-read,
    pre-bash, pre-grep).
    """
    status: Literal["fresh"]
    hookSpecificOutput: NotRequired["PreToolUseHookOutput"]
    # AC-05: also present on watchdog-timeout degraded responses.
    degraded: NotRequired[bool]


class StaleResponse(TypedDict):
    """The complete wire shape for a stale-read response (AC-04 / finding #25).

    ``build_stale_response`` emits all three fields; tests access
    ``body['status']`` and ``body['summary']['path']`` directly.
    Document the full shape so typed consumers have an accurate contract.
    """
    hookSpecificOutput: "PreToolUseHookOutput"
    status: Literal["stale"]
    summary: StaleSummary


class PreToolUseHookOutput(TypedDict):
    hookEventName: Literal["PreToolUse"]
    permissionDecision: Literal["allow", "deny"]
    additionalContext: str
    permissionDecisionReason: NotRequired[str]


class PreToolUseContextOutput(TypedDict):
    """SB-10: context-only ``hookSpecificOutput`` envelope for a PreToolUse
    response — advisory prose with NO permission decision.

    Distinct from :class:`PreToolUseHookOutput` by the ABSENCE of
    ``permissionDecision``: this envelope delivers text to the model
    without touching the tool call's permission outcome, so Claude Code's
    ordinary prompting still applies. Used by the SB-10 deferred
    re-grounding attach, whose payload is advisory (KD3) and must never
    widen a permission decision. Empirically the CLI renders
    ``additionalContext`` on a PreToolUse envelope carrying no
    ``permissionDecision`` (A/B capture against Claude Code CLI 2.1.233,
    2026-08-25)."""
    hookEventName: Literal["PreToolUse"]
    additionalContext: str


class PostToolUseContextOutput(TypedDict):
    """#185: context-only ``hookSpecificOutput`` envelope for a PostToolUse
    response -- prose for the model after the tool ran, with no decision.

    Carries the post-edit handoff arm to the giver whose in-flight edit was
    refused a version. That Claude Code relays ``additionalContext``
    from a PostToolUse hook to the model is an assumption, observed once in
    the giver-deny measurement; if it is not relayed the arm is a wire-level
    answer and the giver learns its outcome on its next pre-read or
    pre-edit."""
    hookEventName: Literal["PostToolUse"]
    additionalContext: str


class SessionStartHookOutput(TypedDict):
    """SB-10 U2: ``hookSpecificOutput`` envelope for the SessionStart hook.

    Unlike PreToolUse there is no permissionDecision — SessionStart cannot
    gate anything (KD3: re-grounding is advisory, never blocking); the
    envelope carries only the re-grounding prose."""
    hookEventName: Literal["SessionStart"]
    additionalContext: str


class OkResponse(TypedDict):
    ok: Literal[True]


class CollisionResponse(TypedDict):
    """Edit collision (another session holds EXCLUSIVE). v0.1 warn-only —
    permissionDecision stays "allow"; v0.2 strict mode flips this to "deny"."""
    hookSpecificOutput: PreToolUseHookOutput


class SessionStopResponse(TypedDict):
    ok: Literal[True]
    released_artifacts: list[str]  # parent-repo-relative paths


class PolicyTrackResponse(TypedDict):
    ok: Literal[True]
    added: list[str]
    rejected: list[dict]  # [{"path": "...", "reason": "..."}, ...]


class PolicyUntrackResponse(TypedDict):
    ok: Literal[True]
    removed: list[str]
    rejected: list[dict]  # [{"path": "...", "reason": "..."}, ...] — AC-06 / finding #27


class StatusResponse(TypedDict):
    """The ``GET /status`` body, as ``_handle_status`` actually emits it.

    ``tracked_artifacts`` entries are ``{"path", "version", "id"}``;
    ``sessions`` entries are ``{"agent_name", "agent_id", "states"}``, where
    ``agent_name`` is ``None`` for a holder the adapter has no name for (a
    grant that outlived the coordinator process that issued it). At the
    operator tier each entry also carries ``reclaimed``: ``{path: {"trigger",
    "reclaimed_at_unix_ts"}}`` for each path whose last write grant the
    coordinator sweep reclaimed, with no write grant there since. The earlier
    annotation documented ``last_writer`` and ``session_id`` keys the handler
    has never emitted; nothing in the tree type-checks against this TypedDict,
    so the drift went unnoticed.

    Keys only some answers carry are comments, not annotations: a test holds
    every annotation to a normal default-tier body.
    """

    # A degraded answer (#238: the registry lock was not won within the
    # handler budget) sends both lists as null and adds "degraded": true; the
    # rest of its tier's keys are present. A null list means "cannot tell",
    # never "nothing tracked". A normal answer never carries "degraded".
    tracked_artifacts: list[dict]  # [{"path": "...", "version": int, "id": "..."}, ...]
    # The operator tier adds per entry "owner_generation": int (#187).
    # The operator tier adds per session "reclaimed": {path: {"trigger": str, "reclaimed_at_unix_ts": int}},
    # "grants": {path: {"granted_at_unix_ts": int|None}} for its EXCLUSIVE and
    # MODIFIED paths, and "last_heartbeat_unix_ts": int|None (#187).
    sessions: list[dict]  # [{"agent_name": str|None, "agent_id": "...", "states": {path: state_name}}, ...]
    # The operator tier adds "grant_heartbeat_timeout_sec" and
    # "grant_max_hold_sec": int|None, the sweep's enforced thresholds (#187).
    # AC-02: canonical name follows KTD-J convention (full-word _seconds
    # suffix). ``coordinator_uptime_s`` is emitted alongside as a
    # deprecated alias for one release; consumers should migrate to the
    # canonical name. Removed in v0.2.
    coordinator_uptime_seconds: float
    coordinator_uptime_s: float
    coordinator_pid: int


class ErrorResponse(TypedDict):
    error: str


# ----------------------------------------------------------------------
# Warning templates — per-invocation variation for v0.2 strict-mode safety
# ----------------------------------------------------------------------


def stale_read_warning(summary: StaleSummary) -> str:
    """Build the stale-read additionalContext message.

    Per-invocation variation: both `last_writer_at_unix_ts` (real commit tick)
    and `warning_generated_at_unix_ts` (handler-time `now()`) appear in the
    prose. The latter guarantees byte-different text on every invocation,
    structurally precluding the §13.5 retry-loop hazard when v0.2 strict
    mode flips allow → deny.

    F1 fix: distinguishes "first observation of this artifact" from
    "previously-seen-but-now-invalidated" cases with accurate prose.

    R8: when :func:`summary_reports_a_write` refuses, the artifact was NOT
    written and the session's grant is simply gone — the warn arm says that
    instead of naming a writer and a commit tick that describe a different
    event. ``warning_generated_at_unix_ts`` still renders on both arms, so
    the per-invocation variation above is unaffected.

    Constraint: no content bytes, no content hashes, no diff text.
    """
    if not summary_reports_a_write(summary):
        return _grant_change_warning(summary)
    last_writer_short = short_session_id(summary["last_writer_session_id"])
    last_writer_ts = datetime.fromtimestamp(
        summary["last_writer_at_unix_ts"], tz=timezone.utc
    ).isoformat()
    generated_ts = datetime.fromtimestamp(
        summary["warning_generated_at_unix_ts"], tz=timezone.utc
    ).isoformat()
    prior = summary.get("prior_version_seen_by_session")
    if prior is not None:
        prior_clause = f"you previously saw v{prior}"
    else:
        prior_clause = (
            "this is the first time your session has observed this artifact "
            "(another session in this workspace registered it before you)"
        )
    if summary["hash_differs"]:
        divergence = (
            "Your worktree's current content also differs from the coordinator's "
            "last-recorded hash, which suggests in-flight local edits or a "
            "different branch checkout."
        )
    else:
        divergence = (
            "Your worktree's content matches the last-recorded hash; the divergence "
            "is purely about version-tracking metadata."
        )
    return (
        f"⚠ Stale read [warning emitted {generated_ts}]: {summary['path']} was "
        f"updated by agent {last_writer_short} at {last_writer_ts}. "
        f"Current version is v{summary['current_version']}; {prior_clause}. "
        f"{divergence} "
        f"Consider re-reading {summary['path']} before acting on stale assumptions."
    )


def _grant_change_warning(summary: StaleSummary) -> str:
    """The warn-mode counterpart of :data:`GRANT_CHANGE_DENY_REASON_TEMPLATE`.

    Reached only when :func:`summary_reports_a_write` refuses, which is what
    makes "the version you last saw" exact rather than an inference:
    ``prior_version_seen_by_session`` equals ``current_version`` there.

    This prose deliberately says NOTHING about content. ``hash_differs`` is
    False on three different states -- the caller sent no hash, the
    coordinator holds none, or the two were compared and agreed -- and only
    the third is a match, so a message asserting the worktree still matches
    would be stating something never measured.

    The advice differs from the write arm on purpose. Nothing moved under the
    reader, so re-reading buys it nothing; what it lost is the grant, and the
    next thing that will fail is a write.
    """
    generated_ts = datetime.fromtimestamp(
        summary["warning_generated_at_unix_ts"], tz=timezone.utc
    ).isoformat()
    path = summary["path"]
    return (
        f"⚠ Stale read [warning emitted {generated_ts}]: your grant on {path} "
        f"was revoked and no new version was committed. {path} is still at "
        f"v{summary['current_version']}, the version you last saw. "
        f"Re-acquire before writing to {path}."
    )


def edit_collision_warning(
    holder_session_id: str,
    holder_acquired_at_unix_ts: float,
    path: str,
) -> str:
    """Build the edit-collision additionalContext message (KTD-1 +
    KTD-9 same-hash-blindness mitigation).

    Per-invocation variation: holder session id + acquired-at timestamp
    + the unique current time at message-build time all change between
    invocations. Future v0.2 strict mode can flip allow → deny safely.
    """
    holder_short = short_session_id(holder_session_id)
    holder_ts = datetime.fromtimestamp(
        holder_acquired_at_unix_ts, tz=timezone.utc
    ).isoformat()
    detected_ts = datetime.now(tz=timezone.utc).isoformat()
    return (
        f"⚠ Concurrent edit detected at {detected_ts} (UTC): another agent "
        f"({holder_short}) has been editing {path} since {holder_ts}. "
        f"Your edit will land in your own worktree, but only one session's "
        f"commit will be accepted by the coordinator. Consider waiting for the "
        f"other session to finish or coordinating which one should proceed."
    )


def build_stale_response(summary: StaleSummary) -> dict:
    """Top-level hookSpecificOutput for a v0.1.1 warn-mode stale-read
    PreToolUse response. Routes through :func:`emit_allow` to satisfy the
    KTD-U structural invariant.

    Strict-mode callers must call :func:`emit_strict_deny` directly +
    wrap the result; this builder is reserved for the warn-mode (allow)
    path that v0.1.1 ships and v0.2 preserves for non-strict-mode
    artifacts."""
    return {
        "hookSpecificOutput": emit_allow(
            source="stale_response_builder",
            additional_context=stale_read_warning(summary),
        ),
        "status": "stale",
        "summary": summary,
    }


def build_collision_response(
    holder_session_id: str,
    holder_acquired_at_unix_ts: float,
    path: str,
) -> dict:
    """Top-level hookSpecificOutput for an edit-collision PreToolUse response.

    Edit collisions are distinct from stale-reads: the editor wants
    EXCLUSIVE but another session holds it. A pre-edit that does not opt in
    gets this shape, an allow with additionalContext, in warn and strict mode
    alike. Contention is not out of strict mode's reach, though: a pre-edit
    that opts in with ``if_unheld: true``, as the hook client does for a
    strict path when the workspace asks it to refuse held files, is refused
    with ``other_holder`` instead, and a client that opted in reads a
    ``collision`` key as a deny, since it means the opt-in was ignored."""
    return {
        "hookSpecificOutput": emit_allow(
            source="collision_response_builder",
            additional_context=edit_collision_warning(
                holder_session_id, holder_acquired_at_unix_ts, path
            ),
        ),
        "ok": True,
        "collision": True,
    }


# ----------------------------------------------------------------------
# SB-10 post-compaction re-grounding prose (KTD8)
# ----------------------------------------------------------------------
#
# Byte-parity contract: the Node coordinator (plan U6) mirrors these exact
# strings, and the protocol corpus byte-matches the rendered payload. NO
# timestamps may appear in any of them (corpus normalization keys stay
# untouched), and grant prose is EVENT-ANCHORED, not present-tense — a
# turn-end Stop drain can release E/M before the attachment ever renders,
# so "you hold" would emit a false claim. Any wording change must land in
# both backends plus the corpus fixtures in the same change.


SESSION_START_HEADER: str = "Post-compaction re-grounding (agent-coherence):"
"""First line of every non-empty re-grounding payload."""

SESSION_START_GRANT_LINE_TEMPLATE: str = (
    "At compaction you held {state} on {path} (v{version}) — re-acquire "
    "before writing."
)
"""R3 held-grant line. ``state`` is the full MESI state name
(EXCLUSIVE/MODIFIED/SHARED); ``version`` is the CURRENT coordinated
version from the snapshot, not the granted-at version."""

SESSION_START_STALE_LINE_TEMPLATE: str = (
    "{path} advanced to v{current} past your last-observed v{last} — "
    "re-read before relying on it."
)
"""R4 stale-divergence line (KD1 shape B): both versions render so the
model can judge how far behind its cached view is."""

SESSION_START_TOUCHED_LINE_TEMPLATE: str = "{path} is at v{current}."
"""R4 touched-but-current line — also the R7 admit rendering for
never-observed rows and own-edit-exempt rows."""

SESSION_START_OVERFLOW_LINE_TEMPLATE: str = (
    "Plus {count} more — run agent-coherence-status for the full picture."
)
"""R5 overflow line, mirroring the ``_build_preemption_text`` cap pattern:
at most 3 artifact lines render verbatim; the rest coalesce here."""

SESSION_START_SUBAGENT_PREFIX_TEMPLATE: str = "Subagent {name}:"
"""KTD8 grouping: the parent agent's lines render first (no prefix), then
each registered subagent's lines under this prefix, groups sorted by
agent name."""

SESSION_START_CLOSING_LINE: str = (
    "Versions are as of this re-grounding; a more recent read supersedes "
    "this notice."
)
"""Self-qualifier, always the last line when any lines rendered — R2
accepts one residual duplicate delivery, so the prose must read correctly
when seen twice (a later read wins over a stale re-emission)."""


def emit_pretooluse_context(*, additional_context: str) -> PreToolUseContextOutput:
    """Build a context-only ``hookSpecificOutput`` envelope for a PreToolUse
    response: ``additionalContext`` prose and nothing else.

    Deliberately NOT routed through :func:`emit_allow` — and deliberately
    emitting no ``permissionDecision``. An advisory payload must never
    widen a permission decision: promoting a bare admit body to
    ``permissionDecision: "allow"`` just to carry prose would
    short-circuit Claude Code's own permission prompting for that tool
    call. The KTD-U meta-test counts ``emit_allow`` call sites as
    allow-path surface, which this is not.

    Empirical basis: a PreToolUse ``hookSpecificOutput`` with
    ``hookEventName`` + ``additionalContext`` and no ``permissionDecision``
    IS rendered to the model — A/B capture against the installed Claude
    Code CLI 2.1.233 on 2026-08-25 (the marker-primed model quoted the
    injected line verbatim in both arms).
    """
    return {
        "hookEventName": "PreToolUse",
        "additionalContext": additional_context,
    }


def emit_posttooluse_context(*, additional_context: str) -> PostToolUseContextOutput:
    """Build a context-only ``hookSpecificOutput`` envelope for a PostToolUse
    response (#185). Not routed through :func:`emit_allow`: a post-tool hook
    decides nothing, and the KTD-U meta-test counts ``emit_allow`` call sites
    as allow-path surface, which this is not."""
    return {
        "hookEventName": "PostToolUse",
        "additionalContext": additional_context,
    }


def emit_session_start(*, additional_context: str) -> SessionStartHookOutput:
    """Build the ``hookSpecificOutput`` envelope for a SessionStart response.

    The first non-PreToolUse builder in this module — hookEventName is
    hardcoded per envelope kind, matching the existing builders' style.
    Deliberately NOT routed through :func:`emit_allow`: there is no
    permissionDecision on SessionStart, and the KTD-U meta-test counts
    ``emit_allow`` call sites as allow-path surface, which this is not.
    """
    return {
        "hookEventName": "SessionStart",
        "additionalContext": additional_context,
    }


def now_unix() -> float:
    """Single source of truth for the coordinator's notion of 'now', so
    tests can mock it cleanly later."""
    return time.time()
