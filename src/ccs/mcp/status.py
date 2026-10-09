# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""The ``swg_status`` payload: an honest, 3-state view of coherence enforcement.

The load-bearing honesty bug this avoids (SC5): conflating ``off`` (coordinator
reachable, no strict patterns) with ``unknown`` (coordinator unreachable). A
caller that reads ``off`` may assume it is safe to write unguarded; ``unknown``
must NOT collapse to that. The shipped ``strict_mode_active()`` returns ``False``
for both, so this composes the raw ``/status`` instead.

``per_path`` enforcement is the CLIENT's belief — a tracked artifact that matches
THIS server's managed globs — not a cross-checked coordinator fact. This
server's own globs are checked against the coordinator's published policy when
its volume attaches (a mismatch fails the volume closed), but a PEER's differing
scope is still not visible here (``heterogeneous_scope_detectable=false``).
Under the same SC5 rule ``per_path`` is ``None``, never ``{}``, when ``/status``
carries no artifact list, as a degraded answer does when its registry was busy
(#238).

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
them.

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

from typing import TYPE_CHECKING

from ccs.adapters.claude_code.coordinator_server import session_to_agent_id
from ccs.adapters.claude_code.hook_payloads import (
    HANDOFF_ROLE_BYSTANDER,
    HANDOFF_ROLE_GIVER,
    HANDOFF_ROLE_SUCCESSOR,
)
from ccs.adapters.claude_code.policy import matches_any

if TYPE_CHECKING:
    from ccs.adapters.coherent_volume import CoherentVolume
    from ccs.mcp.session import SessionConfig

# The text-channel line ``swg_status`` adds when ``per_path`` is ``None``, so a
# client that surfaces only text still tells "cannot tell" from a healthy answer.
PER_PATH_UNAVAILABLE_TEXT = (
    "per_path=unavailable: the coordinator could not report which paths are "
    "tracked; retry shortly and do not treat this as nothing tracked"
)


def build_status(volume: CoherentVolume, config: SessionConfig) -> dict:
    """Compose the ``swg_status`` payload from the coordinator ``/status`` + the
    volume's local view + this server's managed scope."""
    status_doc = volume.coordinator_status()  # None if unattached / unreachable
    return {
        "coordinator": _coordinator_state(volume, status_doc),
        "is_attached": volume.is_attached,
        "is_degraded": volume.is_degraded,
        "session_id": volume.session_id,
        "session_agent_id": str(session_to_agent_id(volume.session_id)),
        "principal_claim": volume.principal_claim_outcome,
        "caller_principal_absent_total": _counter(status_doc, "caller_principal_absent_total"),
        "caller_principal_refused_total": _counter(status_doc, "caller_principal_refused_total"),
        "managed": list(config.managed),
        "per_path": _per_path(config, status_doc),
        "single_host_only": True,
        # v1 cannot tell a guarded workspace from a heterogeneous multi-host or
        # differently-scoped one. The gap is loud here, not programmatically
        # detectable (per-glob detection → v1.1).
        "heterogeneous_scope_detectable": False,
    }


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
    """A ``/status`` counter forwarded verbatim, or ``None`` when the
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


def _per_path(config: SessionConfig, status_doc: dict | None) -> dict | None:
    """Per tracked artifact: its version and whether it is ``enforced`` (matches
    this server's managed globs) or merely ``not_registered`` for strict
    enforcement by this server, plus its transfer record as ``handoff`` when
    the coordinator reports one.

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
        enforced = matches_any(path, config.managed)
        entry = {
            "version": artifact.get("version"),
            "status": "enforced" if enforced else "not_registered",
        }
        handoff = artifact.get("handoff")
        if isinstance(handoff, dict):
            entry["handoff"] = handoff
        per_path[path] = entry
    return per_path


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

    ``/status`` lists the file under the name the volume sends, resolved
    against its root with symlinks followed, so ``path`` is matched under
    that name: under an in-root link's own name it matches nothing and a
    live record would read as none."""
    status_doc = volume.coordinator_status()
    artifacts = status_doc.get("tracked_artifacts") if isinstance(status_doc, dict) else None
    if not isinstance(artifacts, list):
        return False, None
    resolved = (volume.root / path).resolve()
    known_as = resolved.relative_to(volume.root).as_posix() if resolved.is_relative_to(volume.root) else path
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("path") != known_as:
            continue
        record = artifact.get("handoff")
        if not isinstance(record, dict):
            return True, None
        return True, {"role": _handoff_role(volume, record), **record}
    return True, None


def _handoff_role(volume: CoherentVolume, record: dict) -> str:
    me = str(session_to_agent_id(volume.session_id))
    if record.get("giver") == me:
        return HANDOFF_ROLE_GIVER
    if record.get("successor") == me:
        return HANDOFF_ROLE_SUCCESSOR
    return HANDOFF_ROLE_BYSTANDER
