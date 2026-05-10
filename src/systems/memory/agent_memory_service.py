from __future__ import annotations

from collections.abc import Sequence
import uuid

from .agent_memory_models import (
    DEFAULT_AGENT_MEMORY_BLOCKS,
    AgentMemoryBlock,
    AgentMemoryBlockInput,
    AgentMemoryPassage,
    AgentMemoryPassageInput,
    AgentMemoryStatusSnapshot,
    normalize_agent_memory_label,
    normalize_agent_memory_scope,
    normalize_agent_memory_tags,
    utc_now,
)
from .agent_memory_repository import AgentMemoryRepository


class AgentMemoryService:
    def __init__(self, repository: AgentMemoryRepository) -> None:
        self.repository = repository

    def ensure_default_blocks(self) -> None:
        existing = {block.label for block in self.repository.list_blocks()}
        for block in DEFAULT_AGENT_MEMORY_BLOCKS:
            if block.label not in existing:
                self.repository.upsert_block(block)

    def list_blocks(self) -> list[AgentMemoryBlock]:
        return self.repository.list_blocks()

    def update_block(
        self,
        label: str,
        block_input: AgentMemoryBlockInput,
        *,
        allow_read_only: bool = False,
    ) -> AgentMemoryBlock:
        normalized_label = normalize_agent_memory_label(label)
        current = self.repository.get_block(normalized_label)
        if current is None:
            raise KeyError(f"agent memory block not found: {normalized_label}")
        if current.read_only and not allow_read_only:
            raise PermissionError(f"agent memory block is read-only: {normalized_label}")

        next_limit = int(block_input.limit if block_input.limit is not None else current.limit)
        if next_limit <= 0:
            raise ValueError("agent memory block limit must be greater than zero")
        next_value = str(block_input.value)
        if len(next_value) > next_limit:
            raise ValueError(
                f"agent memory block value exceeds limit for {normalized_label}: "
                f"{len(next_value)} > {next_limit}"
            )

        updated = AgentMemoryBlock(
            label=normalized_label,
            description=current.description if block_input.description is None else str(block_input.description),
            value=next_value,
            limit=next_limit,
            read_only=current.read_only if block_input.read_only is None else bool(block_input.read_only),
            scope=current.scope if block_input.scope is None else normalize_agent_memory_scope(block_input.scope),
            version=current.version + 1,
            updated_at=utc_now(),
        )
        self.repository.upsert_block(updated)
        return updated

    def insert_passage(self, passage_input: AgentMemoryPassageInput) -> AgentMemoryPassage:
        content = " ".join(str(passage_input.content or "").strip().split())
        if not content:
            raise ValueError("agent memory passage content is required")
        metadata = passage_input.metadata if isinstance(passage_input.metadata, dict) else {}
        timestamp = passage_input.created_at or utc_now()
        passage = AgentMemoryPassage(
            passage_id=str(uuid.uuid4()),
            content=content,
            tags=normalize_agent_memory_tags(passage_input.tags),
            scene_scope=_normalize_scene_scope(passage_input.scene_scope),
            metadata=dict(metadata),
            created_at=timestamp,
            updated_at=timestamp,
        )
        self.repository.insert_passage(passage)
        return passage

    def search_passages(
        self,
        query: str,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[AgentMemoryPassage]:
        return self.repository.search_passages(
            query,
            tags=tags,
            tag_match_mode=tag_match_mode,
            scene_scope=scene_scope,
            top_k=top_k,
        )

    def list_passages(
        self,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 50,
    ) -> list[AgentMemoryPassage]:
        return self.repository.list_passages(
            tags=tags,
            tag_match_mode=tag_match_mode,
            scene_scope=scene_scope,
            top_k=top_k,
        )

    def status_snapshot(self, *, enabled: bool) -> AgentMemoryStatusSnapshot:
        return AgentMemoryStatusSnapshot(
            enabled=bool(enabled),
            available=True,
            core_block_count=self.repository.block_count(),
            archival_passage_count=self.repository.passage_count(),
            archival_tags=self.repository.unique_tags(),
            degraded_reason=None,
        )


def _normalize_scene_scope(scene_scope: str | None) -> str | None:
    normalized = " ".join(str(scene_scope or "").strip().split())
    return normalized or None
