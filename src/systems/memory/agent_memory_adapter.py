from __future__ import annotations

from typing import Any

from .agent_memory_models import AgentMemoryContext


def agent_memory_context_payload(context: AgentMemoryContext) -> dict[str, Any]:
    return {
        "core_blocks": [
            {
                "label": block.label,
                "description": block.description,
                "value": block.value,
                "limit": block.limit,
                "read_only": block.read_only,
                "scope": block.scope,
                "version": block.version,
                "updated_at": block.updated_at.isoformat(),
            }
            for block in context.core_blocks
        ],
        "archival_passages": [
            {
                "passage_id": passage.passage_id,
                "content": passage.content,
                "tags": list(passage.tags),
                "scene_scope": passage.scene_scope,
                "metadata": dict(passage.metadata),
                "created_at": passage.created_at.isoformat(),
                "updated_at": passage.updated_at.isoformat(),
                "rank": passage.rank,
            }
            for passage in context.archival_passages
        ],
        "conversation_summary": context.conversation_summary,
        "recent_turns": [dict(turn) for turn in context.recent_turns],
        "object_memory": [dict(item) for item in context.object_memory],
        "knowledge_facts": [dict(item) for item in context.knowledge_facts],
        "metadata": {
            "enabled": context.metadata.enabled,
            "available": context.metadata.available,
            "core_block_count": context.metadata.core_block_count,
            "archival_passage_count": context.metadata.archival_passage_count,
            "recall_turn_count": context.metadata.recall_turn_count,
            "object_memory_count": context.metadata.object_memory_count,
            "knowledge_fact_count": context.metadata.knowledge_fact_count,
            "archival_tags": list(context.metadata.archival_tags),
            "degraded_reason": context.metadata.degraded_reason,
        },
    }


def render_agent_memory_context(context: AgentMemoryContext | None, *, max_chars: int = 6000) -> str:
    if context is None:
        return "Agent memory:\n(unavailable)"

    lines: list[str] = ["Agent memory:"]
    if context.metadata.degraded_reason:
        lines.append(f"- degraded: {context.metadata.degraded_reason}")

    if context.core_blocks:
        lines.append("Core blocks:")
        for block in context.core_blocks:
            value = _compact(block.value) or "(empty)"
            lines.append(f"- {block.label}: {value}")

    if context.conversation_summary:
        lines.append(f"Recall summary: {_compact(context.conversation_summary)}")

    if context.archival_passages:
        lines.append("Archival passages:")
        for passage in context.archival_passages:
            tag_text = ",".join(passage.tags) or "untagged"
            scope_text = passage.scene_scope or "global"
            lines.append(f"- [{scope_text}; {tag_text}] {_compact(passage.content)}")

    if context.object_memory:
        lines.append("Object memory summaries:")
        for item in context.object_memory[:10]:
            lines.append(f"- {_compact(_render_dict(item))}")

    if context.knowledge_facts:
        lines.append("Knowledge facts:")
        for item in context.knowledge_facts[:8]:
            text = item.get("text") or item.get("reason") or _render_dict(item)
            lines.append(f"- {_compact(str(text))}")

    rendered = "\n".join(lines)
    if len(rendered) <= max_chars:
        return rendered
    return rendered[: max(0, max_chars - 16)].rstrip() + "\n...(truncated)"


def _compact(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _render_dict(value: dict[str, Any]) -> str:
    parts = [f"{key}={item}" for key, item in sorted(value.items()) if item is not None]
    return ", ".join(parts)
