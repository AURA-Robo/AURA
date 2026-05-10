from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import re
from typing import Any


DEFAULT_AGENT_MEMORY_SCOPE = "global"
DEFAULT_AGENT_MEMORY_BLOCK_LIMIT = 4000
WORKING_MEMORY_BLOCK_LIMIT = 2000


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def normalize_agent_memory_label(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", str(value or "").strip().lower()).strip("_")
    if not normalized:
        raise ValueError("agent memory block label is required")
    return normalized


def normalize_agent_memory_scope(value: str | None) -> str:
    normalized = " ".join(str(value or DEFAULT_AGENT_MEMORY_SCOPE).strip().split())
    return normalized or DEFAULT_AGENT_MEMORY_SCOPE


def normalize_agent_memory_tags(values: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    normalized: list[str] = []
    for raw_value in values:
        tag = re.sub(r"\s+", "_", str(raw_value or "").strip().lower())
        if tag and tag not in normalized:
            normalized.append(tag)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class AgentMemoryBlockInput:
    value: str
    description: str | None = None
    limit: int | None = None
    read_only: bool | None = None
    scope: str | None = None


@dataclass(frozen=True, slots=True)
class AgentMemoryBlock:
    label: str
    description: str
    value: str
    limit: int
    read_only: bool
    scope: str
    version: int
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True, slots=True)
class AgentMemoryPassageInput:
    content: str
    tags: tuple[str, ...] | list[str] | None = None
    scene_scope: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class AgentMemoryPassage:
    passage_id: str
    content: str
    tags: tuple[str, ...]
    scene_scope: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    rank: float | None = None


@dataclass(frozen=True, slots=True)
class AgentMemoryMetadata:
    enabled: bool
    available: bool
    core_block_count: int = 0
    archival_passage_count: int = 0
    recall_turn_count: int = 0
    object_memory_count: int = 0
    knowledge_fact_count: int = 0
    archival_tags: tuple[str, ...] = ()
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class AgentMemoryContext:
    core_blocks: list[AgentMemoryBlock]
    archival_passages: list[AgentMemoryPassage]
    conversation_summary: str
    recent_turns: list[dict[str, str]]
    object_memory: list[dict[str, Any]]
    knowledge_facts: list[dict[str, Any]]
    metadata: AgentMemoryMetadata
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentMemoryStatusSnapshot:
    enabled: bool
    available: bool
    core_block_count: int
    archival_passage_count: int
    archival_tags: tuple[str, ...]
    degraded_reason: str | None = None


DEFAULT_AGENT_MEMORY_BLOCKS = (
    AgentMemoryBlock(
        label="persona",
        description="Humanoid identity and interaction style. System-owned and always visible.",
        value=(
            "AURA is a grounded humanoid robotics agent that plans conservatively, "
            "reports uncertainty clearly, and does not invent execution state."
        ),
        limit=DEFAULT_AGENT_MEMORY_BLOCK_LIMIT,
        read_only=True,
        scope=DEFAULT_AGENT_MEMORY_SCOPE,
        version=1,
    ),
    AgentMemoryBlock(
        label="operator_profile",
        description="Stable operator preferences and durable personalization facts.",
        value="",
        limit=DEFAULT_AGENT_MEMORY_BLOCK_LIMIT,
        read_only=False,
        scope=DEFAULT_AGENT_MEMORY_SCOPE,
        version=1,
    ),
    AgentMemoryBlock(
        label="mission_policy",
        description="Safety, permission, and deployment constraints. System-owned policy memory.",
        value="Follow configured knowledge rules, respect runtime limits, and ask for clarification when targets are ambiguous.",
        limit=DEFAULT_AGENT_MEMORY_BLOCK_LIMIT,
        read_only=True,
        scope=DEFAULT_AGENT_MEMORY_SCOPE,
        version=1,
    ),
    AgentMemoryBlock(
        label="environment_baseline",
        description="Long-lived scene, room, and alias facts that are not spatial pose memory.",
        value="",
        limit=DEFAULT_AGENT_MEMORY_BLOCK_LIMIT,
        read_only=False,
        scope=DEFAULT_AGENT_MEMORY_SCOPE,
        version=1,
    ),
    AgentMemoryBlock(
        label="working_memory",
        description="Current task scratchpad and handoff state. Keep this short and current.",
        value="",
        limit=WORKING_MEMORY_BLOCK_LIMIT,
        read_only=False,
        scope=DEFAULT_AGENT_MEMORY_SCOPE,
        version=1,
    ),
    AgentMemoryBlock(
        label="capabilities",
        description="Current high-level robot capability reminders. System-owned and read-only.",
        value="Can route dialogue, plan navigation/check-state tasks, use object memory for spatial targets, and report task state.",
        limit=DEFAULT_AGENT_MEMORY_BLOCK_LIMIT,
        read_only=True,
        scope=DEFAULT_AGENT_MEMORY_SCOPE,
        version=1,
    ),
)
