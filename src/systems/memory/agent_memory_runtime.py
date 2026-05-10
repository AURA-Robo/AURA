from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent_memory_models import (
    AgentMemoryBlockInput,
    AgentMemoryContext,
    AgentMemoryMetadata,
    AgentMemoryPassage,
    AgentMemoryPassageInput,
    AgentMemoryStatusSnapshot,
)
from .agent_memory_repository import PostgresAgentMemoryRepository
from .agent_memory_service import AgentMemoryService
from .conversation_memory_models import ConversationContext
from .knowledge_models import KnowledgeContext
from .object_memory_models import ObjectMemoryContext


DURABLE_WRITE_MARKERS = (
    "prefer",
    "preference",
    "remember",
    "correction",
    "correct",
    "means",
    "call this",
    "usually",
    "located",
    "near",
    "항상",
    "기억",
    "정정",
    "수정",
)


@dataclass(slots=True)
class HumanoidMemoryRuntimeHandle:
    enabled: bool
    service: AgentMemoryService | None = None
    degraded_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.service is not None and self.degraded_reason is None

    def compile_context(
        self,
        utterance: str,
        *,
        conversation_context: ConversationContext,
        object_memory_context: ObjectMemoryContext | None,
        knowledge_context: KnowledgeContext | None,
        scene_scope: str | None,
        top_k: int = 5,
    ) -> AgentMemoryContext:
        conversation_summary, recent_turns = _conversation_payload(conversation_context)
        object_memory = _object_memory_payload(object_memory_context)
        knowledge_facts = _knowledge_fact_payload(knowledge_context)

        if self.service is None:
            return AgentMemoryContext(
                core_blocks=[],
                archival_passages=[],
                conversation_summary=conversation_summary,
                recent_turns=recent_turns,
                object_memory=object_memory,
                knowledge_facts=knowledge_facts,
                metadata=AgentMemoryMetadata(
                    enabled=bool(self.enabled),
                    available=False,
                    recall_turn_count=len(recent_turns),
                    object_memory_count=len(object_memory),
                    knowledge_fact_count=len(knowledge_facts),
                    degraded_reason=self.degraded_reason,
                ),
            )

        try:
            blocks = self.service.list_blocks()
            passages = self.service.search_passages(
                _retrieval_query(utterance, conversation_context),
                scene_scope=scene_scope,
                top_k=top_k,
            )
            status = self.service.status_snapshot(enabled=self.enabled)
            return AgentMemoryContext(
                core_blocks=blocks,
                archival_passages=passages,
                conversation_summary=conversation_summary,
                recent_turns=recent_turns,
                object_memory=object_memory,
                knowledge_facts=knowledge_facts,
                metadata=AgentMemoryMetadata(
                    enabled=bool(self.enabled),
                    available=status.available,
                    core_block_count=status.core_block_count,
                    archival_passage_count=status.archival_passage_count,
                    recall_turn_count=len(recent_turns),
                    object_memory_count=len(object_memory),
                    knowledge_fact_count=len(knowledge_facts),
                    archival_tags=status.archival_tags,
                    degraded_reason=status.degraded_reason,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return AgentMemoryContext(
                core_blocks=[],
                archival_passages=[],
                conversation_summary=conversation_summary,
                recent_turns=recent_turns,
                object_memory=object_memory,
                knowledge_facts=knowledge_facts,
                metadata=AgentMemoryMetadata(
                    enabled=bool(self.enabled),
                    available=False,
                    recall_turn_count=len(recent_turns),
                    object_memory_count=len(object_memory),
                    knowledge_fact_count=len(knowledge_facts),
                    degraded_reason=self.degraded_reason,
                ),
            )

    def status_snapshot(self) -> AgentMemoryStatusSnapshot:
        if self.service is None:
            return AgentMemoryStatusSnapshot(
                enabled=bool(self.enabled),
                available=False,
                core_block_count=0,
                archival_passage_count=0,
                archival_tags=(),
                degraded_reason=self.degraded_reason,
            )
        try:
            snapshot = self.service.status_snapshot(enabled=self.enabled)
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return AgentMemoryStatusSnapshot(
                enabled=bool(self.enabled),
                available=False,
                core_block_count=0,
                archival_passage_count=0,
                archival_tags=(),
                degraded_reason=self.degraded_reason,
            )
        if self.degraded_reason and snapshot.degraded_reason is None:
            return AgentMemoryStatusSnapshot(
                enabled=snapshot.enabled,
                available=False,
                core_block_count=snapshot.core_block_count,
                archival_passage_count=snapshot.archival_passage_count,
                archival_tags=snapshot.archival_tags,
                degraded_reason=self.degraded_reason,
            )
        return snapshot

    def list_blocks(self):
        if self.service is None:
            raise RuntimeError("agent memory service unavailable")
        return self.service.list_blocks()

    def update_block(self, label: str, block_input: AgentMemoryBlockInput):
        if self.service is None:
            raise RuntimeError("agent memory service unavailable")
        return self.service.update_block(label, block_input)

    def insert_passage(self, passage_input: AgentMemoryPassageInput):
        if self.service is None:
            raise RuntimeError("agent memory service unavailable")
        return self.service.insert_passage(passage_input)

    def list_passages(
        self,
        *,
        tags: tuple[str, ...] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 50,
    ) -> list[AgentMemoryPassage]:
        if self.service is None:
            raise RuntimeError("agent memory service unavailable")
        return self.service.list_passages(
            tags=tags,
            tag_match_mode=tag_match_mode,
            scene_scope=scene_scope,
            top_k=top_k,
        )

    def search_passages(
        self,
        query: str,
        *,
        tags: tuple[str, ...] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[AgentMemoryPassage]:
        if self.service is None:
            raise RuntimeError("agent memory service unavailable")
        return self.service.search_passages(
            query,
            tags=tags,
            tag_match_mode=tag_match_mode,
            scene_scope=scene_scope,
            top_k=top_k,
        )

    def record_interaction(
        self,
        *,
        conversation_id: str,
        utterance: str,
        reply_text: str,
        route: str,
        resolved_slots: dict[str, str] | None = None,
        scene_scope: str | None = None,
        task_status: str | None = None,
    ) -> None:
        if self.service is None:
            return
        content, tags = _durable_passage_for_interaction(
            conversation_id=conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route=route,
            resolved_slots=resolved_slots or {},
            task_status=task_status,
        )
        if content is None:
            return
        try:
            self.service.insert_passage(
                AgentMemoryPassageInput(
                    content=content,
                    tags=tags,
                    scene_scope=scene_scope,
                    metadata={
                        "conversation_id": conversation_id,
                        "route": route,
                        "task_status": task_status,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"


def create_humanoid_memory_runtime(
    *,
    dsn: str | None,
    object_memory_dsn: str | None = None,
    auto_migrate: bool = True,
) -> HumanoidMemoryRuntimeHandle:
    normalized_dsn = str(dsn or "").strip() or str(object_memory_dsn or "").strip()
    if not normalized_dsn:
        return HumanoidMemoryRuntimeHandle(
            enabled=True,
            degraded_reason="agent_memory_dsn_missing",
        )

    try:
        repository = PostgresAgentMemoryRepository(normalized_dsn)
        if auto_migrate:
            repository.apply_schema()
        service = AgentMemoryService(repository)
        service.ensure_default_blocks()
    except Exception as exc:  # noqa: BLE001
        return HumanoidMemoryRuntimeHandle(
            enabled=True,
            degraded_reason=f"{type(exc).__name__}: {exc}",
        )

    return HumanoidMemoryRuntimeHandle(enabled=True, service=service)


def _conversation_payload(conversation_context: ConversationContext) -> tuple[str, list[dict[str, str]]]:
    summary = str(getattr(conversation_context, "summary", "") or "")
    recent_turns = [
        {
            "role": str(turn.role),
            "text": str(turn.text),
        }
        for turn in getattr(conversation_context, "recent_turns", [])
    ]
    return summary, recent_turns


def _object_memory_payload(object_memory_context: ObjectMemoryContext | None) -> list[dict[str, Any]]:
    if object_memory_context is None:
        return []
    recent_seen = getattr(object_memory_context, "recent_seen", None)
    if isinstance(recent_seen, list):
        return [dict(item) for item in recent_seen if isinstance(item, dict)]
    return []


def _knowledge_fact_payload(knowledge_context: KnowledgeContext | None) -> list[dict[str, Any]]:
    if knowledge_context is None:
        return []
    facts = getattr(knowledge_context, "facts", None)
    return [
        {
            "text": str(getattr(fact, "text", "")),
            "source_anchor": getattr(fact, "source_anchor", None),
            "rank": getattr(fact, "rank", None),
        }
        for fact in facts or []
    ]


def _retrieval_query(utterance: str, conversation_context: ConversationContext) -> str:
    slots = getattr(conversation_context, "resolved_slots", {}) or {}
    slot_text = " ".join(str(value) for value in slots.values() if value)
    return " ".join(part for part in (utterance, slot_text) if str(part or "").strip())


def _durable_passage_for_interaction(
    *,
    conversation_id: str,
    utterance: str,
    reply_text: str,
    route: str,
    resolved_slots: dict[str, str],
    task_status: str | None,
) -> tuple[str | None, tuple[str, ...]]:
    normalized_utterance = " ".join(str(utterance or "").strip().split())
    normalized_reply = " ".join(str(reply_text or "").strip().split())
    lower_utterance = normalized_utterance.lower()
    tags: list[str] = [str(route or "interaction").strip().lower() or "interaction"]

    durable_marker = next((marker for marker in DURABLE_WRITE_MARKERS if marker in lower_utterance), None)
    if durable_marker is not None:
        if "prefer" in lower_utterance or "preference" in lower_utterance:
            tags.append("preference")
        if durable_marker in {"correction", "correct", "means", "정정", "수정"}:
            tags.append("correction")
        return (
            f"Conversation {conversation_id}: operator said '{normalized_utterance}'. "
            f"Assistant replied '{normalized_reply}'.",
            tuple(tags),
        )

    terminal_statuses = {"completed", "complete", "success", "failed", "failure", "cancelled", "canceled", "error"}
    status_text = str(task_status or "").strip().lower()
    if str(route).strip().lower() == "task" and status_text in terminal_statuses:
        slots_text = ", ".join(f"{key}={value}" for key, value in sorted(resolved_slots.items()) if value)
        tags.append("task_outcome")
        return (
            f"Conversation {conversation_id}: route={route}, status={status_text}, "
            f"utterance='{normalized_utterance}', reply='{normalized_reply}', slots={slots_text or 'none'}.",
            tuple(tags),
        )

    return None, ()
