from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import uuid

from .conversation_memory_models import ConversationContext, ConversationStatusSnapshot, ConversationTurn
from .conversation_memory_repository import (
    ConversationMemoryRepository,
    InMemoryConversationMemoryRepository,
    PostgresConversationMemoryRepository,
)


@dataclass(slots=True)
class ConversationMemoryRuntimeHandle:
    enabled: bool
    repository: ConversationMemoryRepository | None = None
    fallback_repository: InMemoryConversationMemoryRepository = field(default_factory=InMemoryConversationMemoryRepository)
    degraded_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.repository is not None and self.degraded_reason is None

    def load_context(self, conversation_id: str, max_turns: int = 8) -> ConversationContext:
        normalized_id = str(conversation_id).strip()
        if self.repository is None or self.degraded_reason is not None:
            context = self.fallback_repository.load_context(normalized_id, max_turns=max_turns)
            return _with_debug(context, enabled=self.enabled, available=False, degraded_reason=self.degraded_reason)
        try:
            context = self.repository.load_context(normalized_id, max_turns=max_turns)
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            context = self.fallback_repository.load_context(normalized_id, max_turns=max_turns)
            return _with_debug(context, enabled=self.enabled, available=False, degraded_reason=self.degraded_reason)
        return _with_debug(context, enabled=self.enabled, available=True, degraded_reason=None)

    def append_turn(
        self,
        conversation_id: str,
        role: str,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        turn = ConversationTurn(
            turn_id=str(uuid.uuid4()),
            conversation_id=str(conversation_id).strip(),
            role=str(role).strip(),
            text=str(text),
            metadata=_stringify_metadata(metadata),
        )
        self.fallback_repository.append_turn(turn)
        if self.repository is not None and self.degraded_reason is None:
            try:
                self.repository.append_turn(turn)
            except Exception as exc:  # noqa: BLE001
                self.degraded_reason = f"{type(exc).__name__}: {exc}"
        return turn

    def update_summary(
        self,
        conversation_id: str,
        summary: str,
        resolved_slots: dict[str, Any] | None = None,
    ) -> None:
        normalized_slots = _stringify_metadata(resolved_slots)
        normalized_id = str(conversation_id).strip()
        self.fallback_repository.update_summary(normalized_id, str(summary), normalized_slots)
        if self.repository is not None and self.degraded_reason is None:
            try:
                self.repository.update_summary(normalized_id, str(summary), normalized_slots)
            except Exception as exc:  # noqa: BLE001
                self.degraded_reason = f"{type(exc).__name__}: {exc}"

    def status_snapshot(self) -> ConversationStatusSnapshot:
        repository = self.repository if self.repository is not None and self.degraded_reason is None else self.fallback_repository
        return ConversationStatusSnapshot(
            enabled=bool(self.enabled),
            available=bool(self.repository is not None and self.degraded_reason is None),
            conversation_count=repository.conversation_count(),
            turn_count=repository.turn_count(),
            degraded_reason=self.degraded_reason,
        )


def create_conversation_memory_runtime(
    *,
    dsn: str | None,
) -> ConversationMemoryRuntimeHandle:
    normalized_dsn = str(dsn or "").strip()
    if not normalized_dsn:
        return ConversationMemoryRuntimeHandle(
            enabled=True,
            degraded_reason="conversation_memory_dsn_missing",
        )
    try:
        repository = PostgresConversationMemoryRepository(normalized_dsn)
        repository.apply_schema()
    except Exception as exc:  # noqa: BLE001
        return ConversationMemoryRuntimeHandle(
            enabled=True,
            degraded_reason=f"{type(exc).__name__}: {exc}",
        )
    return ConversationMemoryRuntimeHandle(
        enabled=True,
        repository=repository,
    )


def _stringify_metadata(metadata: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(metadata, dict):
        return {}
    return {str(key): str(value) for key, value in metadata.items()}


def _with_debug(
    context: ConversationContext,
    *,
    enabled: bool,
    available: bool,
    degraded_reason: str | None,
) -> ConversationContext:
    debug = dict(context.debug)
    debug.update(
        {
            "enabled": bool(enabled),
            "available": bool(available),
            "degraded_reason": degraded_reason,
        }
    )
    return ConversationContext(
        conversation_id=context.conversation_id,
        summary=context.summary,
        resolved_slots=dict(context.resolved_slots),
        recent_turns=list(context.recent_turns),
        debug=debug,
    )
