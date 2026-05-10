from __future__ import annotations

from systems.memory.agent_memory_models import AgentMemoryBlock, AgentMemoryContext, AgentMemoryMetadata
from systems.memory.conversation_memory_models import ConversationContext
from systems.memory.knowledge_models import utc_now
from systems.reasoning.dialogue import DialogueService


def test_dialogue_prompt_includes_compiled_agent_memory_context() -> None:
    now = utc_now()
    agent_memory = AgentMemoryContext(
        core_blocks=[
            AgentMemoryBlock(
                label="operator_profile",
                description="Operator preferences",
                value="Use short Korean confirmations.",
                limit=1024,
                read_only=False,
                scope="global",
                version=1,
                updated_at=now,
            )
        ],
        archival_passages=[],
        conversation_summary="The operator asked for navigation status.",
        recent_turns=[],
        object_memory=[{"class": "chair", "room": "lab", "age_sec": 4}],
        knowledge_facts=[],
        metadata=AgentMemoryMetadata(enabled=True, available=True),
    )
    service = DialogueService(completion=None, model="test", timeout=1.0)

    messages = service.build_messages(
        "hello",
        language="ko",
        conversation_context=ConversationContext(
            conversation_id="conv-1",
            summary="",
            resolved_slots={},
            recent_turns=[],
        ),
        agent_memory_context=agent_memory,
        scene_preset="warehouse",
    )

    system_prompt = messages[0]["content"]
    assert "Agent memory:" in system_prompt
    assert "operator_profile" in system_prompt
    assert "Use short Korean confirmations." in system_prompt
    assert "chair" in system_prompt
