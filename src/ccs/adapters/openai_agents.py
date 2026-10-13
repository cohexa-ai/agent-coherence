# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""OpenAI Agents SDK coherence adapter.

The adapter's primary durable surface (post-Q6, 2026-05-31) is the **Session
cache**: the OpenAI Agents SDK's ``Session`` is local conversation memory
(``get_items``/``add_items``/``pop_item``/``clear_session``), and a peer that
mutates a shared session leaves this agent's cached view stale — regardless of
how consistent the underlying durable store is. The Q6 probe found the
Conversations *server* consistent, so the coherence value lives here, on the
readers' caches.

The SDK exposes no Session hook/middleware API, so interception is by
*composition*: ``CoherenceSession`` wraps a caller-provided Session, overrides
the four async methods, and routes coherence accounting through
``CoherenceAdapterCore``. Because the underlying Session is supplied by the
caller, this module imports no ``agents`` symbol — it works against anything
implementing the four-method protocol.

Scope (v1): in-process multi-agent coherence — peers registered on one
``CoherenceAdapterCore``/process (the same boundary as the LangGraph/CrewAI/
AutoGen adapters). Cross-service coherence needs the out-of-process coordinator.

**Status: experimental (0.x).** The OpenAI Agents SDK is itself 0.x and churns;
this surface may change with it. Pinned to ``openai-agents>=0.17,<0.18``.

Install::

    pip install "agent-coherence[openai-agents]"

Usage::

    from agents import Runner, SQLiteSession
    from ccs.adapters.openai_agents import OpenAIAgentsAdapter

    adapter = OpenAIAgentsAdapter()
    session = adapter.wrap_session(SQLiteSession("chat-1"), agent_name="planner", session_id="chat-1")
    # Drive the run; the optional RunHooks track identity + refresh coherence:
    await Runner.run(agent, "...", session=session, hooks=adapter.run_hooks(session_id="chat-1"))
    if session.peer_mutated_since_read():  # a peer changed the conversation
        ...  # re-read before acting
"""

from __future__ import annotations

import logging
import threading
import warnings
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from ccs.adapters.base import CoherenceAdapterCore
from ccs.coordinator.service import CrashRecoveryConfig
from ccs.core.exceptions import CoherenceDegradedWarning, CoherenceError, CoherenceTopologyWarning
from ccs.core.states import MESIState
from ccs.core.types import Artifact

if TYPE_CHECKING:  # annotation-only; never imported at runtime (keeps the module SDK-free)
    from agents import RunHooks

logger = logging.getLogger(__name__)

# CoherenceDegradedWarning is the canonical class from ccs.core.exceptions so every
# adapter raises/catches the same type; re-exported here for convenience.
__all__ = ["OpenAIAgentsAdapter", "CoherenceSession", "CoherenceDegradedWarning"]


class OpenAIAgentsAdapter:
    """Coherence facade for the OpenAI Agents SDK, parity with the other adapters.

    Mirrors the LangGraph/CrewAI/AutoGen constructor shape and composes with
    ``CrashRecoveryConfig``; mints its own monotonic tick (the SDK has no step
    counter) under a lock, following the ``CCSStore`` pattern.
    """

    def __init__(
        self,
        *,
        strategy_name: str = "lazy",
        core: CoherenceAdapterCore | None = None,
        crash_recovery: CrashRecoveryConfig | None = None,
        on_error: Literal["strict", "degrade"] = "degrade",
        **kwargs: Any,
    ) -> None:
        # Runtime guard kept as defence-in-depth for untyped callers.
        if on_error not in ("strict", "degrade"):
            raise ValueError(f"on_error must be 'strict' or 'degrade', got {on_error!r}")
        self._on_error = on_error
        self.core = (
            core
            if core is not None
            else CoherenceAdapterCore(strategy_name=strategy_name, crash_recovery=crash_recovery, **kwargs)
        )
        self._lock = threading.Lock()
        self._tick = 0
        self._sessions: dict[str, UUID] = {}
        self._degradation_count = 0

    # --- parity passthroughs ------------------------------------------------

    def register_agent(self, name: str) -> UUID:
        """Register one agent identity (idempotent)."""
        return self.core.register_agent(name)

    def register_artifact(self, *, name: str, content: str, size_tokens: int | None = None) -> Artifact:
        """Register a shared artifact in the coordinator directory."""
        return self.core.register_artifact(name=name, content=content, size_tokens=size_tokens)

    def heartbeat(self, *, agent_name: str, now_tick: int) -> None:
        """Record a heartbeat for the named agent. No-op when crash recovery is disabled."""
        self.core.heartbeat(agent_name=agent_name, now_tick=now_tick)

    def recover(self, *, agent_name: str, now_tick: int) -> None:
        """Invalidate the agent's local cache and record a recovery heartbeat."""
        self.core.recover(agent_name=agent_name, now_tick=now_tick)

    # --- Session coherence (the primary surface) ----------------------------

    def _session_artifact_id(self, session_id: str) -> UUID:
        """Resolve (registering once) the shared coherence artifact for a session_id.

        All agents and the RunHooks that touch the same ``session_id`` through this
        adapter resolve the same artifact id, so their reads/writes coordinate.
        """
        with self._lock:
            artifact_id = self._sessions.get(session_id)
            if artifact_id is None:
                artifact = self.core.register_artifact(name=f"session:{session_id}", content="")
                artifact_id = artifact.id
                self._sessions[session_id] = artifact_id
        return artifact_id

    def wrap_session(self, underlying: Any, *, agent_name: str, session_id: str) -> CoherenceSession:
        """Wrap a caller-provided Session so peer mutations invalidate this agent.

        All agents wrapping the same ``session_id`` through this adapter share one
        coherence artifact (single registration, shared id) — no deterministic
        artifact-id trick required because the adapter coordinates registration.
        """
        self.core.register_agent(agent_name)
        return CoherenceSession(
            underlying=underlying,
            adapter=self,
            agent_name=agent_name,
            artifact_id=self._session_artifact_id(session_id),
            session_id=session_id,
        )

    def run_hooks(self, *, session_id: str, server_conversation: bool = False) -> RunHooks:
        """Build a ``RunHooks`` that threads coherence accounting through a run.

        Attach the returned object to ``Runner.run(..., hooks=...)``. It tracks the
        active agent across handoffs and refreshes that agent's coherence view at
        agent-start and tool-start, so a peer's mutation of the shared session
        surfaces before the agent acts. Writes remain the ``CoherenceSession``'s job
        (this coordinates *awareness and identity*, not arbitrary tool side effects).

        Set ``server_conversation=True`` when the run uses a server-side
        ``conversation_id``; combined with multi-agent handoffs the SDK disables
        ``input_filter`` / nested handoff history, and the hooks warn once that
        handoff-history coherence is unavailable in that mode.

        Importing ``agents.RunHooks`` is deferred to here so this module stays
        importable (and ``CoherenceSession`` usable) without the ``openai-agents``
        extra installed.
        """
        hooks_cls = _coherence_run_hooks_class()
        return hooks_cls(adapter=self, session_id=session_id, server_conversation=server_conversation)

    # --- degradation contract ----------------------------------------------

    @property
    def is_degraded(self) -> bool:
        return self._degradation_count > 0

    @property
    def degradation_count(self) -> int:
        return self._degradation_count

    def _next_tick(self) -> int:
        with self._lock:
            self._tick += 1
            return self._tick

    def _record_degraded(self, exc: CoherenceError) -> None:
        # Guard the warn-once + counter under the same lock as the tick: the Session
        # ops are async and may race, and without the lock two coroutines can both
        # observe count == 0 and double-emit, or lose an increment.
        with self._lock:
            first = self._degradation_count == 0
            self._degradation_count += 1
        if first:
            # Log before warning: a caller that escalates the warning to an error
            # makes warn() raise, and the log line must not depend on it.
            logger.warning("OpenAI Agents adapter degraded under on_error='degrade': %s", exc)
            warnings.warn(f"OpenAI Agents adapter degraded: {exc}", CoherenceDegradedWarning, stacklevel=3)

    def _handle_coherence_error(self, exc: CoherenceError) -> None:
        """Single on_error dispatch shared by the Session wrapper and RunHooks.

        ``strict`` re-raises (``raise exc`` — explicit, and preserves the original
        traceback); ``degrade`` records + warns once and lets the real work stand.
        Must be called from within the ``except`` block that caught ``exc``.
        """
        if self._on_error == "strict":
            raise exc
        self._record_degraded(exc)


class CoherenceSession:
    """Coherence-tracking wrapper over an OpenAI Agents SDK ``Session``.

    Implements the four-method async protocol by composition. A mutation
    (``add_items``/``pop_item``/``clear_session``) commits through the coordinator
    and invalidates peer agents' cached views; ``get_items`` refreshes this
    agent's coherence state so a prior peer mutation surfaces as a cache miss.
    The underlying Session remains the durable source of truth for the items
    themselves — the coherence layer governs *awareness*, not storage.

    Degrade-mode limitation: under ``on_error='degrade'`` a ``CoherenceError`` from a
    mutation is swallowed. ``CoherenceAdapterCore.write`` grants the writer EXCLUSIVE
    and *then* commits as two coordinator steps; if the commit step fails (e.g. a
    concurrent writer reclaimed the grant), the writer can be left holding a stable
    EXCLUSIVE grant with peers already invalidated. That stranded grant is only
    reclaimed by the crash-recovery sweep, which is disabled by default. For
    concurrent-writer workloads on the same session under degrade, enable
    ``CrashRecoveryConfig`` so stranded grants self-heal.
    """

    def __init__(
        self,
        *,
        underlying: Any,
        adapter: OpenAIAgentsAdapter,
        agent_name: str,
        artifact_id: UUID,
        session_id: str,
    ) -> None:
        self._underlying = underlying
        self._adapter = adapter
        self._agent_name = agent_name
        self._artifact_id = artifact_id
        self.session_id = session_id
        self._has_read = False  # has this agent established a coherence baseline yet?

    async def get_items(self, limit: int | None = None) -> list[Any]:
        """Read items; refresh coherence state first so peer writes show as misses."""
        self._coherence_read()
        self._has_read = True
        return await self._underlying.get_items(limit)

    async def add_items(self, items: list[Any]) -> None:
        """Persist items to the underlying Session, then invalidate peers."""
        await self._underlying.add_items(items)  # durable write first
        self._coherence_commit("add")

    async def pop_item(self) -> Any:
        """Pop from the underlying Session, then invalidate peers."""
        item = await self._underlying.pop_item()
        self._coherence_commit("pop")
        return item

    async def clear_session(self) -> None:
        """Clear the underlying Session, then invalidate peers."""
        await self._underlying.clear_session()
        self._coherence_commit("clear")

    def peer_mutated_since_read(self) -> bool:
        """True if this agent should re-read before trusting its session view.

        Conservative by design: returns True when a peer mutated the session since
        this agent's last read (cache entry is INVALID) **and also** when this agent
        has never read yet — without a baseline read there is no "since last read",
        so the only safe answer is "you must read first". Call ``get_items`` to
        establish the baseline; only then does this report genuine peer mutations.
        """
        if not self._has_read:
            return True
        entry = self._adapter.core.runtime(self._agent_name).cache.get(self._artifact_id)
        return entry is not None and entry.state == MESIState.INVALID

    # --- coherence plumbing with the degrade contract -----------------------

    def _coherence_read(self) -> None:
        try:
            self._adapter.core.read(
                agent_name=self._agent_name,
                artifact_id=self._artifact_id,
                now_tick=self._adapter._next_tick(),
            )
        except CoherenceError as exc:
            self._degrade_or_raise(exc)

    def _coherence_commit(self, op: str) -> None:
        try:
            self._adapter.core.write(
                agent_name=self._agent_name,
                artifact_id=self._artifact_id,
                content=f"mutated:{op}",
                now_tick=self._adapter._next_tick(),
            )
        except CoherenceError as exc:
            self._degrade_or_raise(exc)

    def _degrade_or_raise(self, exc: CoherenceError) -> None:
        # The underlying Session op already succeeded — under degrade, coherence is
        # best-effort and we never swallow the real work, only the accounting.
        self._adapter._handle_coherence_error(exc)


# --- RunHooks lifecycle integration (Unit 7) -------------------------------
#
# Defined lazily so importing this module never pulls `agents` — the SDK-free
# CoherenceSession surface stays importable on a bare `[dev]` install. The class
# subclasses agents.RunHooks (needed for the Runner's default-method contract)
# and is built once, on first `OpenAIAgentsAdapter.run_hooks(...)`.

_RUN_HOOKS_CLASS: type | None = None
_RUN_HOOKS_CLASS_LOCK = threading.Lock()


def _coherence_run_hooks_class() -> type:
    global _RUN_HOOKS_CLASS
    if _RUN_HOOKS_CLASS is not None:
        return _RUN_HOOKS_CLASS

    # Lock the build so two threads racing the first call can't define two distinct
    # class objects (matters under free-threaded Python; harmless under the GIL).
    with _RUN_HOOKS_CLASS_LOCK:
        if _RUN_HOOKS_CLASS is not None:
            return _RUN_HOOKS_CLASS
        return _build_run_hooks_class()


def _build_run_hooks_class() -> type:
    global _RUN_HOOKS_CLASS
    try:
        from agents import RunHooks  # deferred third-party import
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "OpenAIAgentsAdapter.run_hooks requires the OpenAI Agents SDK. "
            'Install it with: pip install "agent-coherence[openai-agents]"'
        ) from exc

    class CoherenceRunHooks(RunHooks):
        """Threads coherence accounting through an Agents SDK run (Unit 7).

        Tracks the active agent across handoffs and refreshes that agent's
        coherence view at agent-start and tool-start, so a peer's mutation of the
        shared session surfaces *before* the agent acts. Writes stay the
        ``CoherenceSession``'s job — these hooks coordinate awareness and identity,
        not arbitrary tool side effects (the coherence surface is session state,
        not every tool call).

        Supported topology: concurrent independent agents sharing one
        ``conversation_id`` within one process. With ``server_conversation=True``
        plus a handoff, the SDK disables ``input_filter`` / nested handoff history;
        the first handoff then warns once (``CoherenceTopologyWarning``) that
        handoff-history coherence is unavailable — never a silent assumption.

        The coordinator handle is carried by this hooks instance (runtime-only,
        never serialized to the model), which satisfies the "don't send the
        coordinator to the LLM" goal the same way ``RunContextWrapper.context``
        would.
        """

        active_agent: str | None
        read_count: int

        def __init__(self, *, adapter: OpenAIAgentsAdapter, session_id: str, server_conversation: bool = False) -> None:
            super().__init__()
            self._adapter = adapter
            self._session_id = session_id
            self._server_conversation = server_conversation
            self.active_agent = None
            self.read_count = 0
            self._handoff_topology_warned = False

        def _refresh(self, agent_name: str) -> None:
            """Refresh one agent's coherence view of the shared session (a peer write
            shows up as a cache miss). Honours the adapter's on_error contract.

            ``read_count`` counts *successful* coordinator reads; it stays flat while a
            refresh is degrading. The coordinator read is in-process (pure dict work),
            so calling it synchronously from these async hooks does not block the loop.
            """
            self._adapter.register_agent(agent_name)
            artifact_id = self._adapter._session_artifact_id(self._session_id)
            try:
                self._adapter.core.read(
                    agent_name=agent_name,
                    artifact_id=artifact_id,
                    now_tick=self._adapter._next_tick(),
                )
                self.read_count += 1
            except CoherenceError as exc:
                self._adapter._handle_coherence_error(exc)

        async def on_agent_start(self, context: Any, agent: Any) -> None:
            self.active_agent = agent.name
            self._refresh(agent.name)

        async def on_tool_start(self, context: Any, agent: Any, tool: Any) -> None:
            self._refresh(agent.name)

        async def on_handoff(self, context: Any, from_agent: Any, to_agent: Any) -> None:
            self.active_agent = to_agent.name
            if self._server_conversation and not self._handoff_topology_warned:
                self._handoff_topology_warned = True
                warnings.warn(
                    "OpenAI Agents handoff with a server-side conversation_id: the SDK "
                    "disables input_filter / nested handoff history, so handoff-history "
                    "coherence is unavailable. Use concurrent agents sharing one "
                    "conversation_id (no handoff history rewriting) instead.",
                    CoherenceTopologyWarning,
                    stacklevel=2,
                )
            self._refresh(to_agent.name)

    _RUN_HOOKS_CLASS = CoherenceRunHooks
    return _RUN_HOOKS_CLASS
