# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Per-session configuration + coordinator binding for the stale-write-guard-fs server.

The MCP client spawns this server as a stdio subprocess per session; the server
owns exactly ONE :class:`~ccs.adapters.coherent_volume.CoherentVolume` for that
session's workspace. Constructing the volume self-spawns/attaches the loopback
coordinator (strict-only, fail-closed); on a normal exit the server lifespan
closes the volume, giving back its write grants, then calls
``stop_coordinator``, which stops the coordinator only when this process
started it, and then for every peer attached to it. The server NEVER constructs
a degrade-mode volume — a degrade write skips the version check and silently
re-opens the lost update, so strict-only is the honesty floor.

``SWG_ON_HELD`` picks what a write does to a managed file another agent holds:
``displace`` (unset or empty too, the default) takes it, as the server always
did; ``refuse`` builds the volume with ``on_held="refuse"``, so the write is
refused while another agent holds the file. It is matched exactly,
because a near-miss read as the default would leave an operator believing holds
are honoured.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ccs.adapters.claude_code.lifecycle import LifecycleConfig
from ccs.adapters.coherent_volume import CoherentVolume, OnHeld

# Env keys the MCP client sets when spawning the server subprocess.
ROOT_ENV = "SWG_ROOT"
MANAGED_ENV = "SWG_MANAGED"
ON_HELD_ENV = "SWG_ON_HELD"
# Guard the whole workspace unless the client narrows it. A non-empty managed set
# is REQUIRED for the INVALID-deny to fire (is_strict_mode ⊂ is_tracked).
_DEFAULT_MANAGED: tuple[str, ...] = ("**",)


def _on_held_from(environ: Mapping[str, str]) -> OnHeld:
    """What a write does to a file another agent holds, from ``SWG_ON_HELD``.

    Matched exactly: a near-miss such as ``Refuse`` raises rather than starting
    a server whose writes take files from their holders while its operator
    believes they are refused. Unset or empty is the default, ``displace``."""
    raw = environ.get(ON_HELD_ENV, "")
    if raw in ("", "displace"):
        return "displace"
    if raw == "refuse":
        return "refuse"
    raise ValueError(
        f"{ON_HELD_ENV} must be 'refuse' or 'displace' (unset or empty means 'displace'), got {raw!r}"
    )


@dataclass(frozen=True)
class SessionConfig:
    """Resolved workspace root, the managed (strict-enforced) glob set, and what
    a write does to a file another agent holds."""

    root: Path
    managed: tuple[str, ...]
    on_held: OnHeld = "displace"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> SessionConfig:
        """Resolve config from the spawn environment. ``SWG_ROOT`` defaults to the
        process cwd; ``SWG_MANAGED`` is a comma-separated glob list (default ``**``);
        ``SWG_ON_HELD`` is ``refuse`` or ``displace`` (unset or empty: ``displace``).

        Raises ``ValueError`` for any other ``SWG_ON_HELD``, so no caller starts
        on a value it cannot honour."""
        environ = os.environ if env is None else env
        root = Path(environ.get(ROOT_ENV) or os.getcwd()).resolve()
        raw = environ.get(MANAGED_ENV, "")
        managed = tuple(g.strip() for g in raw.split(",") if g.strip()) or _DEFAULT_MANAGED
        return cls(root=root, managed=managed, on_held=_on_held_from(environ))


def build_volume(config: SessionConfig) -> CoherentVolume:
    """Construct the session's strict-only volume.

    ``on_error='strict'`` is non-negotiable: a degrade-mode volume writes with no
    version check (the fail-open hole), so the server never constructs one.
    Construction self-spawns/attaches the coordinator and RAISES (fail-closed) if
    it cannot attach or enable strict — the server then refuses to start.
    ``idle_shutdown_sec=0`` keeps the coordinator alive for the server's lifetime.
    ``on_stale_write='raise'`` (SB-23) is the write-surface complement to strict
    mode: a ``swg_write`` that would clobber a foreign / out-of-band edit is denied
    rather than silently overwriting it. It is the constructor default; set
    explicitly here to match the strict-only honesty floor.
    ``on_held='refuse'`` makes a write to a managed file another agent holds
    raise instead of taking it; the attach then raises
    :class:`~ccs.core.exceptions.RefuseModeUnsupported` when the coordinator
    cannot be relied on to refuse.
    """
    volume = CoherentVolume(
        config.root,
        managed=config.managed,
        on_error="strict",
        on_stale_write="raise",
        on_held=config.on_held,
        config=LifecycleConfig(idle_shutdown_sec=0),
    )
    # Strict construction guarantees attachment (else it raised); assert loudly so
    # a future regression that admits an unattached volume is caught at startup,
    # not at the first silent lost-write.
    if not volume.is_attached:
        raise RuntimeError("coherence coordinator did not attach; strict construction must fail closed")
    return volume
