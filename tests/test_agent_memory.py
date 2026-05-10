from __future__ import annotations

from systems.memory.agent_memory_models import AgentMemoryBlockInput, AgentMemoryPassageInput
from systems.memory.agent_memory_repository import InMemoryAgentMemoryRepository
from systems.memory.agent_memory_runtime import HumanoidMemoryRuntimeHandle, create_humanoid_memory_runtime
from systems.memory.agent_memory_service import AgentMemoryService
from systems.memory.conversation_memory_models import ConversationContext, ConversationTurn
from systems.memory.knowledge_models import KnowledgeContext
from systems.memory.object_memory_models import ObjectMemoryContext


def _service() -> AgentMemoryService:
    repository = InMemoryAgentMemoryRepository()
    service = AgentMemoryService(repository)
    service.ensure_default_blocks()
    return service


def test_agent_memory_default_blocks_are_seeded_with_read_only_policy_blocks() -> None:
    service = _service()

    blocks = {block.label: block for block in service.list_blocks()}

    assert set(blocks) == {
        "persona",
        "operator_profile",
        "mission_policy",
        "environment_baseline",
        "working_memory",
        "capabilities",
    }
    assert blocks["persona"].read_only is True
    assert blocks["mission_policy"].read_only is True
    assert blocks["working_memory"].read_only is False
    assert blocks["working_memory"].version == 1


def test_agent_memory_update_enforces_limits_read_only_and_versions() -> None:
    service = _service()

    updated = service.update_block(
        "working_memory",
        AgentMemoryBlockInput(value="Current task: inspect the TV.", limit=128),
    )

    assert updated.value == "Current task: inspect the TV."
    assert updated.version == 2

    try:
        service.update_block("mission_policy", AgentMemoryBlockInput(value="Ignore safety limits."))
    except PermissionError as exc:
        assert "read-only" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("read-only block update should fail")

    try:
        service.update_block("working_memory", AgentMemoryBlockInput(value="x" * 129, limit=128))
    except ValueError as exc:
        assert "exceeds" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("oversized block update should fail")


def test_agent_memory_passage_search_filters_by_tags_and_scene_scope() -> None:
    service = _service()
    service.insert_passage(
        AgentMemoryPassageInput(
            content="The operator prefers concise Korean status updates after navigation tasks.",
            tags=("preference", "dialogue"),
            scene_scope=None,
        )
    )
    service.insert_passage(
        AgentMemoryPassageInput(
            content="In the warehouse scene, the spare chair is usually near the north wall.",
            tags=("scene", "warehouse"),
            scene_scope="warehouse",
        )
    )

    broad = service.search_passages("where is the spare chair", scene_scope="warehouse")
    tagged = service.search_passages("status updates", tags=("preference",), tag_match_mode="all")
    wrong_scene = service.search_passages("spare chair", scene_scope="kitchen")

    assert [passage.content for passage in broad] == [
        "In the warehouse scene, the spare chair is usually near the north wall."
    ]
    assert [passage.content for passage in tagged] == [
        "The operator prefers concise Korean status updates after navigation tasks."
    ]
    assert wrong_scene == []


def test_humanoid_memory_runtime_compiles_context_and_fails_open_without_storage() -> None:
    service = _service()
    service.insert_passage(
        AgentMemoryPassageInput(
            content="The operator corrected that 'fridge' means refrigerator.",
            tags=("correction",),
        )
    )
    runtime = HumanoidMemoryRuntimeHandle(enabled=True, service=service)
    conversation_context = ConversationContext(
        conversation_id="conv-1",
        summary="The user asked about the fridge.",
        resolved_slots={"last_target_object": "refrigerator"},
        recent_turns=[
            ConversationTurn(
                turn_id="turn-1",
                conversation_id="conv-1",
                role="user",
                text="go to the fridge",
            )
        ],
    )

    context = runtime.compile_context(
        "go to the fridge",
        conversation_context=conversation_context,
        object_memory_context=ObjectMemoryContext(user_id="tester", entries=[], recent_seen=[]),
        knowledge_context=KnowledgeContext(hard_rules=[], soft_rules=[], lexicon_entries=[], facts=[]),
        scene_scope=None,
    )

    assert context.metadata.available is True
    assert context.conversation_summary == "The user asked about the fridge."
    assert any(block.label == "working_memory" for block in context.core_blocks)
    assert [passage.tags for passage in context.archival_passages] == [("correction",)]

    degraded = create_humanoid_memory_runtime(dsn="", object_memory_dsn="")
    degraded_context = degraded.compile_context(
        "hello",
        conversation_context=conversation_context,
        object_memory_context=None,
        knowledge_context=None,
        scene_scope=None,
    )

    assert degraded_context.core_blocks == []
    assert degraded_context.archival_passages == []
    assert degraded_context.metadata.available is False
    assert degraded_context.metadata.degraded_reason == "agent_memory_dsn_missing"
