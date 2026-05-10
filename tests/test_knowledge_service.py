from __future__ import annotations

from systems.memory.knowledge_models import KnowledgeDocumentInput
from systems.memory.knowledge_repository import InMemoryKnowledgeRepository
from systems.memory.knowledge_service import KnowledgeService


def test_knowledge_document_stays_inactive_until_published() -> None:
    repository = InMemoryKnowledgeRepository()
    service = KnowledgeService(repository)
    document = service.register_document(
        KnowledgeDocumentInput(
            title="Warehouse rules",
            body_markdown="""
# Warehouse
Handle warehouse inventory conservatively.

```knowledge-rule
[
  {
    "id": "deny-fridge-nav",
    "action": "deny_task",
    "enforcement": "hard",
    "conditions": {
      "intent": "navigate_to_object",
      "target_object": "refrigerator"
    },
    "reason": "Do not navigate directly to the refrigerator."
  }
]
```

```knowledge-lexicon
[
  {
    "object_alias": "fridge",
    "canonical_object": "refrigerator"
  },
  {
    "room_alias": "galley",
    "canonical_room": "kitchen"
  }
]
```

The refrigerator stores regulated samples and should be handled with extra caution.
""",
            publish=False,
        )
    )

    assert document.status == "draft"
    draft_context = service.retrieve_for_plan("go to the fridge")
    assert draft_context.hard_rules == []
    assert draft_context.lexicon_entries == []
    assert draft_context.facts == []

    published = service.publish_document(document.document_id)
    assert published.status == "published"

    context = service.retrieve_for_plan("go to the fridge")
    assert len(context.hard_rules) == 1
    assert context.hard_rules[0].action == "deny_task"
    assert len(context.lexicon_entries) == 2
    assert {entry.mapping_type for entry in context.lexicon_entries} == {"object", "room"}
    assert len(context.facts) == 2
    assert any("regulated samples" in chunk.text for chunk in context.facts)


def test_knowledge_service_uses_last_good_cache_when_fact_search_fails() -> None:
    class _FlakyRepository(InMemoryKnowledgeRepository):
        def __init__(self) -> None:
            super().__init__()
            self.fail_search = False

        def search_published_chunks(self, query: str, *, scene_scope: str | None = None, top_k: int = 5):  # type: ignore[override]
            if self.fail_search:
                raise RuntimeError("fts unavailable")
            return super().search_published_chunks(query, scene_scope=scene_scope, top_k=top_k)

    repository = _FlakyRepository()
    service = KnowledgeService(repository)
    service.register_document(
        KnowledgeDocumentInput(
            title="Kitchen rules",
            body_markdown="""
```knowledge-rule
{"action": "force_target_room", "conditions": {"target_object": "refrigerator"}, "room": "kitchen"}
```

The refrigerator is usually in the kitchen.
""",
            publish=True,
        )
    )

    first = service.retrieve_for_plan("go to the refrigerator")
    assert len(first.hard_rules) == 1
    assert len(first.facts) == 1

    repository.fail_search = True
    degraded = service.retrieve_for_plan("go to the refrigerator")
    status = service.status_snapshot()

    assert len(degraded.hard_rules) == 1
    assert degraded.facts == []
    assert status.available is True
    assert status.degraded_reason == "RuntimeError: fts unavailable"
