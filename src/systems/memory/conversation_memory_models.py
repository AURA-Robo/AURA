from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    turn_id: str
    conversation_id: str
    role: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class ConversationContext:
    conversation_id: str
    summary: str
    resolved_slots: dict[str, str]
    recent_turns: list[ConversationTurn]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConversationStatusSnapshot:
    enabled: bool
    available: bool
    conversation_count: int
    turn_count: int
    degraded_reason: str | None = None
