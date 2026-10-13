# Copyright (c) 2026 agent-coherence contributors.
# The Coherence Protocol for AI Agents

"""Agent runtime encapsulating read/write/invalidation protocol flows."""

from __future__ import annotations

from typing import Any, Callable, Optional
from uuid import UUID

from ccs.coordinator.service import CoordinatorService
from ccs.core.exceptions import CasRetriesExhausted
from ccs.core.hashing import compute_content_hash
from ccs.core.states import MESIState
from ccs.core.types import (
    Artifact,
    ArtifactCacheEntry,
    ConflictDetail,
    FetchRequest,
    FetchResponse,
    InvalidationSignal,
)
from ccs.strategies.base import SyncStrategy

from .cache import ArtifactCache

# A content producer for the OCC write path. Called once per attempt with the
# freshly-read cache entry (SHARED, carrying the winner's local_version) so the
# caller re-derives content against current state on every retry. Returns the
# content body and an OPTIONAL precomputed content_hash (None → the runtime
# computes it via compute_content_hash, matching how `write` derives the hash).
MakeContent = Callable[[ArtifactCacheEntry], "tuple[str, str | None]"]

CCS_CONTENT_AUDIT_LOG_SCHEMA_VERSION = "ccs.content_audit.v1"


class AgentRuntime:
    """Reusable protocol participant for one agent identity."""

    def __init__(
        self,
        *,
        agent_id: UUID,
        coordinator: CoordinatorService,
        strategy: SyncStrategy,
        cache: Optional[ArtifactCache] = None,
        content_audit_log: Callable[[dict[str, Any]], None] | None = None,
        audit_seq: list[int] | None = None,
        agent_name: str | None = None,
        instance_id: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.coordinator = coordinator
        self.strategy = strategy
        self.cache = cache if cache is not None else ArtifactCache()
        self._content_by_artifact: dict[UUID, str] = {}
        self._content_audit_log = content_audit_log
        self._audit_seq = audit_seq if audit_seq is not None else [0]
        self._agent_name = agent_name
        self._instance_id = instance_id

    def read(self, artifact_id: UUID, *, now_tick: int) -> FetchResponse:
        """Return artifact view from cache or coordinator."""
        entry = self.cache.get(artifact_id)
        if entry is None or self.strategy.requires_refresh(entry, now_tick=now_tick):
            return self._fetch(artifact_id, now_tick=now_tick)

        touched = self.strategy.on_read(entry, now_tick=now_tick)
        self.cache.put(artifact_id, touched)
        cached_content = self._content_by_artifact.get(artifact_id, "")
        self._record_content_view(
            artifact_id=artifact_id,
            version=touched.local_version,
            content=cached_content,
            source="cache_hit",
            now_tick=now_tick,
        )
        return FetchResponse(
            artifact_id=artifact_id,
            version=touched.local_version,
            content=cached_content,
            state_grant=touched.state,
        )

    def write(
        self,
        artifact_id: UUID,
        *,
        content: str,
        now_tick: int,
        content_hash: str | None = None,
        size_tokens: int | None = None,
    ) -> tuple[Artifact, list[InvalidationSignal]]:
        """Write new artifact content through coordinator protocol.

        Raises:
            StaleReadGeneration: the read-generation fence rejected the commit —
                a sweep reclaimed this agent's claim in the race window between
                acquire and commit (``ccs.core.exceptions``). Retry-eligible,
                but THIS path has no built-in retry loop (unlike ``write_cas``):
                the caller re-acquires / re-reads and calls ``write()`` again.
            CoherenceError: the grant was already reclaimed before commit (the
                error names the reclaim trigger and tick), or another protocol
                precondition failed.
        """
        if content_hash is not None:
            computed = compute_content_hash(content)
            if content_hash != computed:
                raise ValueError(
                    f"content_hash mismatch: caller provided {content_hash!r}, "
                    f"computed {computed!r}"
                )

        entry = self.cache.get(artifact_id)
        if entry is None or self.strategy.requires_refresh(entry, now_tick=now_tick):
            self._fetch(artifact_id, now_tick=now_tick)

        write_signals = self.coordinator.write(
            agent_id=self.agent_id,
            artifact_id=artifact_id,
            issued_at_tick=now_tick,
        )
        updated, commit_signals = self.coordinator.commit(
            agent_id=self.agent_id,
            artifact_id=artifact_id,
            content=content,
            issued_at_tick=now_tick,
            content_hash=content_hash,
            size_tokens=size_tokens,
        )
        self.cache.put(
            artifact_id,
            self.strategy.on_fetch(
                artifact_id=artifact_id,
                version=updated.version,
                state=MESIState.MODIFIED,
                now_tick=now_tick,
            ),
        )
        self._record_content_view(
            artifact_id=artifact_id,
            version=updated.version,
            content=content,
            source="write",
            now_tick=now_tick,
        )
        return updated, [*write_signals, *commit_signals]

    def write_cas(
        self,
        artifact_id: UUID,
        *,
        make_content: MakeContent,
        now_tick: int,
    ) -> tuple[Artifact, list[InvalidationSignal]]:
        """Optimistic-concurrency write that BYPASSES the pessimistic acquire.

        The OCC counterpart to :meth:`write` (plan Unit 5, R6/R8). Unlike
        ``write`` — which calls ``coordinator.write()`` to take EXCLUSIVE before
        ``coordinator.commit()`` — ``write_cas`` **never acquires EXCLUSIVE**. The
        writer reads (→ SHARED), computes locally (stays SHARED/INVALID), and
        commits via ``coordinator.commit_cas``; the commit-time version CAS (not a
        lock on the acquire) elects the winner. ``coordinator.write()`` is never
        called on this path.

        Retry: policy lives in the strategy (``max_cas_retries`` /
        ``cas_backoff_ticks``), the loop lives HERE (the actor) — strategies are
        policy, not actors. ``expected_version`` is sourced from the cache entry's
        ``local_version`` each attempt; on a :class:`ConflictDetail` the loop
        re-reads (re-fetch → refreshes ``local_version`` to the winner's new
        version) and recomputes content via ``make_content`` against that fresh
        state, then retries. There is NO auto-escalation to EXCLUSIVE (D5).

        ``make_content`` is invoked once per attempt with the freshly-read cache
        entry and returns ``(content, content_hash | None)``; a ``None`` hash is
        computed via :func:`compute_content_hash` (mirroring ``write``).

        Args:
            artifact_id: The shared artifact to commit to.
            make_content: Producer re-applying the caller's intent against the
                freshly-read state for THIS attempt (see :data:`MakeContent`).
            now_tick: Logical tick for the attempt(s).

        Returns:
            ``(updated_artifact, signals)`` on the winning commit. ``signals``
            mirrors ``write``'s shape so the adapter publishes them unchanged.

        Raises:
            CasRetriesExhausted: every allowed attempt was refused — a typed
                terminal, NEVER a silent drop, whose ``last_conflict_reason``
                names the last refusal. The cache is left at the latest
                refreshed version (no unconfirmed write cached).
            CoherenceError: the coordinator reported corruption
                (``expected_version > current``) or a precondition failure
                (caller mid-transient / in M/E) — non-retryable, propagated.
            ValueError: ``make_content`` returned a content_hash that does not
                match the content (same guard as ``write``).
        """
        # Ensure a fresh read first so the cache entry exists and SHARED with a
        # populated local_version (the OCC writer is now S, never E).
        entry = self.cache.get(artifact_id)
        if entry is None or self.strategy.requires_refresh(entry, now_tick=now_tick):
            self._fetch(artifact_id, now_tick=now_tick)

        last_current_version = -1
        last_conflict_reason: str | None = None
        max_attempts = self.strategy.max_cas_retries() + 1
        for attempt in range(max_attempts):
            entry = self.cache.get(artifact_id)
            if entry is None:
                # A peer invalidation can drop the placeholder; re-fetch to land
                # a fresh SHARED entry before reading expected_version.
                self._fetch(artifact_id, now_tick=now_tick)
                entry = self.cache.get(artifact_id)
                assert entry is not None  # _fetch always populates the cache

            expected_version = entry.local_version
            content, provided_hash = make_content(entry)
            content_hash = self._resolve_content_hash(content, provided_hash)

            result = self.coordinator.commit_cas(
                agent_id=self.agent_id,
                artifact_id=artifact_id,
                expected_version=expected_version,
                content_hash=content_hash,
                issued_at_tick=now_tick,
                content=content,
            )

            if isinstance(result, ConflictDetail):
                last_current_version = result.current_version
                last_conflict_reason = result.reason
                # Re-read: re-fetch refreshes the cache entry to the winner's new
                # version (SHARED, fresh local_version) so the next attempt's
                # expected_version + make_content() see current state.
                self._fetch(artifact_id, now_tick=now_tick)
                # Consult the strategy's backoff schedule (policy seam, D1). In
                # this synchronous logical-tick model there is no wall clock to
                # sleep against, so the deterministic value documents the bound a
                # tick-driven scheduler would honor; the loop itself does not block.
                if attempt < max_attempts - 1:
                    _ = self.strategy.cas_backoff_ticks(attempt)
                continue

            updated, signals = result
            # An OCC win ends the committer SHARED on the coordinator (it holds no
            # grant — see commit_cas). The local cache must mirror that: a
            # coherence layer cannot leave the agent's local view (MODIFIED) more
            # privileged than the coordinator's now-SHARED end-state. Contrast the
            # pessimistic write() above, which legitimately caches MODIFIED because
            # it acquired EXCLUSIVE.
            self.cache.put(
                artifact_id,
                self.strategy.on_fetch(
                    artifact_id=artifact_id,
                    version=updated.version,
                    state=MESIState.SHARED,
                    now_tick=now_tick,
                ),
            )
            self._record_content_view(
                artifact_id=artifact_id,
                version=updated.version,
                content=content,
                source="write_cas",
                now_tick=now_tick,
            )
            return updated, signals

        raise CasRetriesExhausted(
            artifact_id=artifact_id,
            attempts=max_attempts,
            last_current_version=last_current_version,
            last_conflict_reason=last_conflict_reason,
        )

    def _resolve_content_hash(self, content: str, provided_hash: str | None) -> str:
        """Return the content_hash for content, validating a caller-provided one.

        Mirrors ``write``'s hash handling: a provided hash must match the
        computed one (fail fast on mismatch), else compute it.
        """
        computed = compute_content_hash(content)
        if provided_hash is not None and provided_hash != computed:
            raise ValueError(
                f"content_hash mismatch: caller provided {provided_hash!r}, "
                f"computed {computed!r}"
            )
        return computed

    def invalidate_all_cache(
        self,
        *,
        invalidated_version: int | None = None,
        issued_at_tick: int = 0,
    ) -> None:
        """Invalidate all local cache entries (recovery use). Does not touch coordinator state."""
        self.cache.invalidate_all(
            invalidated_version=invalidated_version,
            issued_at_tick=issued_at_tick,
        )

    def handle_invalidation(self, signal: InvalidationSignal) -> None:
        """Apply invalidation event from coordinator/event bus.

        The coordinator decides FIRST, and the local cache follows its verdict.
        Order is load-bearing: ``invalidate`` can decline a signal the
        NoZombieRevoke pin finds obsolete (``service._revoke_is_superseded``),
        and clamping the cache before asking would drop this agent's view for a
        grant the coordinator then keeps -- a split the pre-pin code could not
        produce, because the coordinator always applied.

        A ``None`` return from an artifact the registry does not know is the one
        case that still clears the cache: the artifact is gone, so there is
        nothing for the local copy to be coherent WITH. Only a live artifact
        whose revoke was declined leaves the cache alone.
        """
        declined = (
            self.coordinator.invalidate(
                agent_id=self.agent_id,
                artifact_id=signal.artifact_id,
                new_version=signal.new_version,
                issuer_agent_id=signal.issuer_agent_id,
                issued_at_tick=signal.issued_at_tick,
            )
            is None
        )
        if declined and self.coordinator.registry.has_artifact(signal.artifact_id):
            return
        self.cache.invalidate(
            signal.artifact_id,
            invalidated_version=max(signal.new_version - 1, 0),
            issued_at_tick=signal.issued_at_tick,
        )

    def handle_update(
        self,
        *,
        artifact_id: UUID,
        version: int,
        content: str,
        now_tick: int,
        writer_agent_id: UUID | None = None,
    ) -> None:
        """Apply eager-broadcast content update from peer/coordinator.

        Pinned like the invalidation path: a broadcast is minted inside the
        registry lock and delivered after it is released, so by delivery time
        the artifact may have moved past it -- the recipient may even hold a
        FRESH write claim established after the mint. Applying a stale update
        would revoke that claim M -> SHARED with no epoch bump, regress the
        cache to older bytes marked valid, and let ``set_agent_state`` stamp
        ``last_observed_version`` with the registry's CURRENT version while
        the cache holds the OLDER body -- poisoning the comparand
        ``_revoke_is_superseded`` trusts. One version comparison under one
        guard hold drops the whole apply; only the transient clears survive
        the drop (a stranded slot stalls the stable-grant sweep, while an
        early clear is harmless).
        """
        with self.coordinator.registry.abort_guard():
            artifact = self.coordinator.registry.get_artifact(artifact_id)
            if artifact is None:
                # Deleted since the mint; the delete's own invalidation path
                # owns the cache teardown.
                return
            state = self.coordinator.registry.get_agent_state(artifact_id, self.agent_id)
            # Two stale shapes. The broadcast's bytes are behind the WORLD
            # (the version moved past the mint), or the recipient holds a
            # LIVE write claim: an update can complete an INVALID/SHARED
            # revalidation, never revoke M/E -- a same-version broadcast
            # against an M/E holder necessarily predates the acquire, because
            # the commit that minted it would have invalidated this agent
            # first. (The writer-relinquish leg below is different: it
            # downgrades the WRITER, and only ever on the fresh path.)
            if version < artifact.version or state in (MESIState.MODIFIED, MESIState.EXCLUSIVE):
                self.coordinator.registry.clear_agent_transient(artifact_id, self.agent_id)
                if writer_agent_id is not None:
                    self.coordinator.registry.clear_agent_transient(artifact_id, writer_agent_id)
                return
            self.cache.put(
                artifact_id,
                self.strategy.on_fetch(
                    artifact_id=artifact_id,
                    version=version,
                    state=MESIState.SHARED,
                    now_tick=now_tick,
                ),
            )
            self._record_content_view(
                artifact_id=artifact_id,
                version=version,
                content=content,
                source="broadcast",
                now_tick=now_tick,
            )
            # An eager push IS an observation, so it completes the grant transition
            # the same way ``service.fetch`` does -- state, then clear the transient.
            # ``_write_impl`` / ``_commit_impl`` set a peer's SIA/EIA transient
            # alongside the INVALID transition and leave the bus-delivered
            # invalidation to clear it; once an update has moved that peer back to
            # SHARED, the NoZombieRevoke pin can legitimately decline the late
            # signal, and without this the transient would strand -- stalling the
            # stable-grant sweep (which skips a pair mid-transient) and the peer's
            # next commit_cas until the transient-timeout fail-safe fires. The
            # in-hold version check above is what makes this stamp truthful: it
            # only ever lands when the broadcast's bytes ARE the current version.
            self.coordinator.registry.set_agent_state(
                artifact_id, self.agent_id, MESIState.SHARED, trigger="update", tick=now_tick
            )
            self.coordinator.registry.clear_agent_transient(artifact_id, self.agent_id)
            if writer_agent_id is not None:
                self.coordinator.registry.set_agent_state(
                    artifact_id, writer_agent_id, MESIState.SHARED, trigger="update", tick=now_tick
                )
                self.coordinator.registry.clear_agent_transient(artifact_id, writer_agent_id)

    def content(self, artifact_id: UUID) -> str | None:
        """Return locally cached content body for artifact if present."""
        return self._content_by_artifact.get(artifact_id)

    def _fetch(self, artifact_id: UUID, *, now_tick: int) -> FetchResponse:
        response = self.coordinator.fetch(
            FetchRequest(
                artifact_id=artifact_id,
                requesting_agent_id=self.agent_id,
                requested_at_tick=now_tick,
            )
        )
        self.cache.put(
            artifact_id,
            self.strategy.on_fetch(
                artifact_id=artifact_id,
                version=response.version,
                state=response.state_grant,
                now_tick=now_tick,
            ),
        )
        self._record_content_view(
            artifact_id=artifact_id,
            version=response.version,
            content=response.content,
            source="fetch",
            now_tick=now_tick,
        )
        return response

    def _record_content_view(
        self,
        *,
        artifact_id: UUID,
        version: int | None,
        content: str | None,
        source: str,
        now_tick: int,
    ) -> str | None:
        """Record a content delivery event and update local content dict.

        Returns the computed content_hash, or None on error/empty outcomes.
        """
        if content is not None and source != "cache_hit":
            self._content_by_artifact[artifact_id] = content

        if content is None:
            outcome = "error"
            record_version = None
            content_hash = None
        elif version is None or version == 0:
            outcome = "empty"
            record_version = None
            content_hash = None
        else:
            outcome = "content"
            record_version = version
            content_hash = compute_content_hash(content)

        if self._content_audit_log is not None:
            self._audit_seq[0] += 1
            entry: dict[str, Any] = {
                "tick": now_tick,
                "agent_id": str(self.agent_id),
                "agent_name": self._agent_name,
                "artifact_id": str(artifact_id),
                "version": record_version,
                "content_hash": content_hash,
                "source": source,
                "outcome": outcome,
                "sequence_number": self._audit_seq[0],
                "instance_id": self._instance_id,
                "schema_version": CCS_CONTENT_AUDIT_LOG_SCHEMA_VERSION,
            }
            try:
                self._content_audit_log(entry)
            except Exception:
                self._audit_seq[0] -= 1
                raise

        return content_hash

