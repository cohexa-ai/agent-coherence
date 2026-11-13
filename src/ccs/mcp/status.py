# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The ``swg_status`` payload: an honest, 3-state view of coherence enforcement.

The load-bearing honesty bug this avoids (SC5): conflating ``off`` (coordinator
reachable, no strict patterns) with ``unknown`` (coordinator unreachable). A
caller that reads ``off`` may assume it is safe to write unguarded; ``unknown``
must NOT collapse to that. The shipped ``strict_mode_active()`` returns ``False``
for both, so this composes the raw ``/status`` instead.

``per_path`` enforcement is the CLIENT's belief -- a tracked artifact this
server's volume judges managed, as its ``write()`` judges a path, folded
spellings and links included -- not a cross-checked coordinator fact. This
server's own globs are checked against the coordinator's published policy when
its volume attaches (a mismatch fails the volume closed), but a PEER's differing
scope is still not visible here (``heterogeneous_scope_detectable=false``).
Under the same SC5 rule ``per_path`` is ``None``, never ``{}``, when ``/status``
carries no artifact list, as a degraded answer does when its registry was busy
(#238).

Write holders come from the same default ``/status`` tier: a session row
holding a path ``EXCLUSIVE`` or ``MODIFIED``, matched to its ``per_path`` entry
by fold key. A held entry carries ``holder_agent_id``, the holder's attempt id
in the 32-hex spelling refusals name a holder by (the Python coordinator lists
it hyphenated, the Node coordinator already in hex), and ``held_by``:
``this_attempt``, ``earlier_attempt`` (an attempt this session started before
its current one, whose grant still stands) or ``another_agent``. An entry
nobody holds carries neither key. ``held`` lists, sorted, the paths whose
holder is this session, either attempt; it is ``None``, never ``[]``, when the
answer carries no sessions list or no artifact list, or the coordinator is
unreachable, because the holders cannot be told. ``on_held`` is this server's
mode, ``refuse`` or ``displace``. Holder ids are per attempt, so they never
equal ``session_agent_id``; ``held_by`` is what tells this session's holds
apart. Every caller of one server process is one agent to the coordinator, so
``this_attempt`` covers a sibling caller too.

``principal_claim`` is the session's own caller-principal state
(:attr:`~ccs.adapters.coherent_volume.CoherentVolume.principal_claim_outcome`):
``refused`` means every later ``swg_write`` / ``swg_read`` / ``swg_gate`` is
answered with the typed ``caller_principal_*`` deny while the coordinator is
still ``on`` — the session, not the coordinator, lost coordination — so an
agent can tell that from a transient before its next call; ``unconfirmed`` is
that transient — the last claim's answer was lost, and the next tool call
claims again with the same nonce by itself before it runs. The two
``caller_principal_*_total`` counters are forwarded from the coordinator's
``/status`` document and, under the same SC5 rule as the three states, are
``None`` (never ``0``) when the coordinator is unreachable or does not report
them. ``registry_schema_version``, the schema version of the coordinator's
store (#294), is forwarded the same way: ``None`` when the coordinator is
unreachable or its answer carries no such field. The strict volume this server
runs never attaches to a release up to v0.14.1, or to a Node coordinator of
plugin 0.6.0 or earlier, which does not serve the operator view. It attaches to
a later Node coordinator, whose answer carries no such field: that
coordinator's own ``schema_version`` counts a different ledger and is never
read in its place.

``session_agent_id`` is this session's session-level agent id: the value
another session names as the successor when it hands this session a path
(#185). It is derived from the volume's session id through the coordinator
module's own function, never a second copy of the derivation string, so it is
exactly the id the coordinator resolves; it does not change when the volume
re-mints its incarnation, and the coordinator knows it as a successor only
while this session's principal claim is bound. A tracked path with a transfer
record carries it as ``handoff`` on its ``per_path`` entry, as the
coordinator's default ``/status`` tier projects it; a path with none has no
such key.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ccs.adapters.claude_code.coordinator_server import parse_agent_id, session_to_agent_id
from ccs.adapters.claude_code.hook_payloads import (
    HANDOFF_ROLE_BYSTANDER,
    HANDOFF_ROLE_GIVER,
    HANDOFF_ROLE_SUCCESSOR,
)
from ccs.coordinator.fold_merge import index_key_of
from ccs.core.exceptions import CoherenceError
from ccs.core.path_fold import FOLD_TABLE_UNICODE_VERSION, FoldProfile

if TYPE_CHECKING:
    from ccs.adapters.coherent_volume import CoherentVolume
    from ccs.mcp.session import SessionConfig

# The text-channel line ``swg_status`` adds when ``per_path`` is ``None``, so a
# client that surfaces only text still tells "cannot tell" from a healthy answer.
PER_PATH_UNAVAILABLE_TEXT = (
    "per_path=unavailable: the coordinator could not report which paths are "
    "tracked; retry shortly and do not treat this as nothing tracked"
)

# The text-channel line ``swg_status`` gives when ``held`` is ``None``, so a
# client that surfaces only text never reads "cannot tell" as nobody holding.
HELD_UNAVAILABLE_TEXT = (
    "held=unavailable: the coordinator could not report which agent holds each "
    "path; retry shortly and do not treat this as no path being held"
)

_Holders = dict[str, dict[str, str]]
_WRITE_STATES = ("EXCLUSIVE", "MODIFIED")
_HELD_BY = {"current": "this_attempt", "earlier": "earlier_attempt", None: "another_agent"}
# The holds that are this session's, either attempt: the ``held`` list.
_OWN_HOLDS = ("this_attempt", "earlier_attempt")
# The holds the text channel names one per line: the ones a refused write
# waits out or this session must release by name.
_WAITED_ON_HOLDS = ("another_agent", "earlier_attempt")

# The advertised profiles that fold. A tuple, not a set: ``fold_profile``
# comes off the wire, and an unhashable value there must compare unequal,
# not raise.
_FOLDING_PROFILES: tuple[FoldProfile, ...] = ("caseless", "nfd")


def build_status(volume: CoherentVolume, config: SessionConfig) -> dict:
    """Compose the ``swg_status`` payload from the coordinator ``/status`` + the
    volume's local view + this server's managed scope."""
    status_doc = volume.coordinator_status()  # None if unattached / unreachable
    profile = fold_profile_of(status_doc)
    holders = _holders_by_key(volume, status_doc, profile)
    per_path = _with_holders(_per_path(volume, status_doc), holders, profile)
    return {
        "coordinator": _coordinator_state(volume, status_doc),
        "is_attached": volume.is_attached,
        "is_degraded": volume.is_degraded,
        "session_id": volume.session_id,
        "session_agent_id": str(session_to_agent_id(volume.session_id)),
        "principal_claim": volume.principal_claim_outcome,
        "caller_principal_absent_total": _counter(status_doc, "caller_principal_absent_total"),
        "caller_principal_refused_total": _counter(status_doc, "caller_principal_refused_total"),
        "registry_schema_version": _counter(status_doc, "registry_schema_version"),
        "managed": list(config.managed),
        "on_held": config.on_held,
        "held": _held(per_path, holders),
        "per_path": per_path,
        "single_host_only": True,
        # v1 cannot tell a guarded workspace from a heterogeneous multi-host or
        # differently-scoped one. The gap is loud here, not programmatically
        # detectable (per-glob detection → v1.1).
        "heterogeneous_scope_detectable": False,
    }


def status_text(status: Mapping[str, Any]) -> str:
    """The text channel of ``swg_status``, built from :func:`build_status`'s
    payload, for clients that surface only text.

    Line one is the coordinator state, followed by
    :data:`PER_PATH_UNAVAILABLE_TEXT` when ``per_path`` is ``None``. Line two
    is the mode and the ``held`` list, or :data:`HELD_UNAVAILABLE_TEXT`. Then
    one line per path another agent or an earlier attempt of this session
    holds, sorted, with its ``held_by`` and ``holder_agent_id``: what a refused
    write waits out, and what this session must release. Names are written as
    JSON strings, so every file name reads back exactly."""
    first = f"coordinator={status['coordinator']}"
    if status["per_path"] is None:
        # A client that shows only the text channel must still see that the
        # tracked paths could not be told: coordinator=on alone reads healthy.
        first = f"{first} {PER_PATH_UNAVAILABLE_TEXT}"
    held = status["held"]
    listed = HELD_UNAVAILABLE_TEXT if held is None else f"held={json.dumps(held)}"
    return "\n".join([first, f"on_held={status['on_held']} {listed}", *_holder_lines(status["per_path"])])


def _holder_lines(per_path: Mapping[str, Mapping[str, Any]] | None) -> list[str]:
    return [
        f"path={json.dumps(path)} held_by={entry['held_by']} holder_agent_id={entry['holder_agent_id']}"
        for path, entry in sorted((per_path or {}).items())
        if entry.get("held_by") in _WAITED_ON_HOLDS
    ]


def _coordinator_state(volume: CoherentVolume, status_doc: dict | None) -> str:
    """``on`` (reachable + strict patterns) / ``off`` (reachable + none) /
    ``unknown`` (unattached or unreachable) — ``unknown`` is NEVER reported as
    ``off``."""
    if not volume.is_attached or status_doc is None:
        return "unknown"
    summary = status_doc.get("policy_summary")
    count = summary.get("strict_mode_pattern_count") if isinstance(summary, dict) else None
    if not isinstance(count, int) or isinstance(count, bool):
        return "unknown"
    return "on" if count > 0 else "off"


def _counter(status_doc: dict | None, key: str) -> int | None:
    """A ``/status`` counter (or other integer field, such as
    ``registry_schema_version``) forwarded verbatim, or ``None`` when the
    coordinator is unreachable or reports none (an older coordinator) or
    reports something that is not an integer — never ``0``, which a reader
    would take for "nothing was counted". ``bool`` is an ``int`` subclass and
    is excluded like everywhere else on this surface."""
    if not isinstance(status_doc, dict):
        return None
    value = status_doc.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _per_path(volume: CoherentVolume, status_doc: dict | None) -> dict | None:
    """Per tracked artifact: its version and whether it is ``enforced`` (this
    server's volume judges it managed, as its ``write()`` does) or merely
    ``not_registered`` for strict enforcement by this server, plus its
    transfer record as ``handoff`` when the coordinator reports one.

    ``None`` when ``/status`` carries no artifact list: a degraded answer
    (#238), whose registry read timed out, carries it as null, and any other
    answer without a list cannot tell either -- never "nothing tracked". An
    unreachable coordinator still gives ``{}``, with ``coordinator`` reported
    ``unknown``."""
    per_path: dict[str, dict] = {}
    if not isinstance(status_doc, dict):
        return per_path
    artifacts = status_doc.get("tracked_artifacts")
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        path = artifact.get("path")
        if not isinstance(path, str):
            continue
        enforced = _is_managed(volume, path)
        entry = {
            "version": artifact.get("version"),
            "status": "enforced" if enforced else "not_registered",
        }
        handoff = artifact.get("handoff")
        if isinstance(handoff, dict):
            entry["handoff"] = handoff
        per_path[path] = entry
    return per_path


def _is_managed(volume: CoherentVolume, path: str) -> bool:
    """Whether this server's writes of the listed ``path`` are coordinated.
    A name that now resolves outside the root, as a tracked file replaced by a
    link out of the workspace does, is one this server cannot write at all,
    so it is not enforced; and one such name must not fail the whole answer."""
    try:
        return volume.is_managed(path)
    except CoherenceError:
        return False


def _holders_by_key(volume: CoherentVolume, status_doc: dict | None, profile: FoldProfile) -> _Holders | None:
    """Each file's write holder keyed by the file's fold key under the
    profile ``/status`` advertises, or ``None`` when the answer carries no
    sessions list (unreachable, degraded, or malformed) or a row it cannot
    read: cannot tell, which is never "nobody holds"."""
    sessions = status_doc.get("sessions") if isinstance(status_doc, dict) else None
    if not isinstance(sessions, list):
        return None
    holders: _Holders = {}
    for session in sessions:
        holds = _write_holds(session)
        if holds is None:  # a row that cannot be read could hold any path
            return None
        for name, agent in holds:
            key = index_key_of(name, profile)
            # Rows an older build split list one file under two spellings;
            # the first holder listed is the one reported, never two.
            if key not in holders:
                holders[key] = _holder_of(volume, agent)
    return holders


def _write_holds(session: object) -> list[tuple[str, UUID]] | None:
    """The names a ``/status`` session row holds for writing, each with the
    row's agent id; ``None`` for a row that does not say whose it is or what
    it holds."""
    row = session if isinstance(session, dict) else {}
    agent, states = parse_agent_id(row.get("agent_id")), row.get("states")
    if agent is None or not isinstance(states, dict):
        return None
    return [(name, agent) for name, state in states.items() if state in _WRITE_STATES]


def _holder_of(volume: CoherentVolume, agent: UUID) -> dict[str, str]:
    # 32 hex, the spelling a refusal names its holder by, so an agent matches
    # the two byte for byte; the Python coordinator's /status hyphenates it.
    holder_agent_id = agent.hex
    return {"holder_agent_id": holder_agent_id, "held_by": _HELD_BY[volume.attempt_of(holder_agent_id)]}


def _with_holders(per_path: dict | None, holders: _Holders | None, profile: FoldProfile) -> dict | None:
    """``per_path`` with each held entry's holder keys added; an entry nobody
    holds keeps exactly the dict it had."""
    if per_path is None or not holders:
        return per_path
    return {path: {**entry, **holders.get(index_key_of(path, profile), {})} for path, entry in per_path.items()}


def _held(per_path: dict | None, holders: _Holders | None) -> list[str] | None:
    """The sorted paths this session holds, either attempt, or ``None`` when
    the holders or the tracked paths cannot be told."""
    if per_path is None or holders is None:
        return None
    return sorted(path for path, entry in per_path.items() if entry.get("held_by") in _OWN_HOLDS)


def handoff_from_status(volume: CoherentVolume, path: str) -> tuple[bool, dict | None]:
    """``path``'s transfer record as the coordinator's default ``/status`` tier
    reports it, projected for this session the way a read's ``handoff`` key
    is: ``(True, projection)``; ``(True, None)`` when ``/status`` lists no
    record for the path; ``(False, None)`` when ``/status`` could not be read
    -- cannot tell, which is never "no record".

    The role is the one the coordinator gives this session in a hook body:
    ``giver`` or ``successor`` when the record names this session's
    session-level agent id, else ``bystander``. ``/status`` carries no role,
    because it has no caller to be a party to the record.

    ``/status`` lists the file under the name its coordinator stores, the
    spelling on disk, resolved against the root with symlinks followed. So
    ``path`` is resolved the same way and compared by its fold key under
    :func:`fold_profile_of` the same document: under an in-root link's own
    name, or a spelling the disk opens as the file, it matched nothing and a
    live record read as none. Rows a start left unmerged are listed under
    each spelling, so the record is taken from whichever of them carries
    one."""
    status_doc = volume.coordinator_status()
    artifacts = status_doc.get("tracked_artifacts") if isinstance(status_doc, dict) else None
    if not isinstance(artifacts, list):
        return False, None
    profile = fold_profile_of(status_doc)
    resolved = (volume.root / path).resolve()
    known_as = resolved.relative_to(volume.root).as_posix() if resolved.is_relative_to(volume.root) else path
    wanted = index_key_of(known_as, profile)
    for artifact in artifacts:
        listed = artifact.get("path") if isinstance(artifact, dict) else None
        if not isinstance(listed, str) or index_key_of(listed, profile) != wanted:
            continue
        record = artifact.get("handoff")
        if isinstance(record, dict):
            return True, {"role": _handoff_role(volume, record), **record}
    return True, None


def fold_profile_of(status_doc: object) -> FoldProfile:
    """The profile to compare file names by, as the coordinator's default
    ``/status`` tier advertises it in ``fold_profile``, so two spellings it
    keys as one file compare equal here too.

    ``none`` (names compared as typed, tidied) whenever this server cannot
    fold as the coordinator does: ``/status`` unread, no ``fold_profile`` (a
    coordinator that stores names as sent), ``cannot_tell`` (a disk its start
    could not classify, keyed as ``none``), a value this build does not know,
    or a ``fold_table_unicode`` other than this build's table, whose keys
    could differ from the coordinator's."""
    if not isinstance(status_doc, dict):
        return "none"
    if status_doc.get("fold_table_unicode") != FOLD_TABLE_UNICODE_VERSION:
        return "none"
    profile = status_doc.get("fold_profile")
    return profile if profile in _FOLDING_PROFILES else "none"


def _handoff_role(volume: CoherentVolume, record: dict) -> str:
    me = str(session_to_agent_id(volume.session_id))
    if record.get("giver") == me:
        return HANDOFF_ROLE_GIVER
    if record.get("successor") == me:
        return HANDOFF_ROLE_SUCCESSOR
    return HANDOFF_ROLE_BYSTANDER
