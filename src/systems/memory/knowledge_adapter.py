from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .knowledge_models import (
    KnowledgeContext,
    KnowledgeFactChunk,
    KnowledgeGuardResult,
    KnowledgeLexiconEntry,
    KnowledgeRule,
)
from .knowledge_service import KnowledgeService, _apply_hard_rules


def retrieve_knowledge_for_plan(
    service: KnowledgeService,
    instruction: str,
    *,
    scene_scope: str | None = None,
    top_k: int = 5,
) -> KnowledgeContext:
    return service.retrieve_for_plan(
        instruction,
        scene_scope=scene_scope,
        top_k=top_k,
    )


def inject_knowledge_context_into_plan_request(
    request: dict[str, Any],
    context: KnowledgeContext,
) -> dict[str, Any]:
    return {
        **request,
        "knowledge_context": knowledge_context_payload(context),
    }


def knowledge_context_payload(context: KnowledgeContext) -> dict[str, Any]:
    return {
        "hard_rules": [knowledge_rule_payload(row) for row in context.hard_rules],
        "soft_rules": [knowledge_rule_payload(row) for row in context.soft_rules],
        "lexicon_entries": [knowledge_lexicon_payload(row) for row in context.lexicon_entries],
        "facts": [knowledge_fact_payload(row) for row in context.facts],
    }


def knowledge_rule_payload(rule: KnowledgeRule) -> dict[str, Any]:
    payload = asdict(rule)
    for key in ("created_at", "updated_at", "published_at"):
        value = payload.get(key)
        payload[key] = None if value is None else value.isoformat()
    return payload


def knowledge_lexicon_payload(entry: KnowledgeLexiconEntry) -> dict[str, Any]:
    payload = asdict(entry)
    for key in ("created_at", "updated_at"):
        payload[key] = payload[key].isoformat()
    return payload


def knowledge_fact_payload(chunk: KnowledgeFactChunk) -> dict[str, Any]:
    payload = asdict(chunk)
    for key in ("created_at", "updated_at"):
        payload[key] = payload[key].isoformat()
    return payload


def apply_knowledge_guards(
    task_frame: dict[str, Any],
    *,
    context: KnowledgeContext,
    utterance: str | None = None,
) -> KnowledgeGuardResult:
    return _apply_hard_rules(task_frame, list(context.hard_rules), utterance=utterance)


def lexicon_alias_maps(entries: list[dict[str, Any]] | list[KnowledgeLexiconEntry]) -> dict[str, dict[str, tuple[str, ...]]]:
    object_aliases: dict[str, list[str]] = {}
    attribute_aliases: dict[str, list[str]] = {}
    room_aliases: dict[str, list[str]] = {}

    for raw in entries:
        row = raw if isinstance(raw, dict) else asdict(raw)
        mapping_type = str(row.get("mapping_type") or "").strip().lower()
        canonical = str(row.get("canonical") or "").strip().lower()
        alias = str(row.get("alias") or "").strip().lower()
        if not mapping_type or not canonical or not alias:
            continue
        if mapping_type == "object":
            object_aliases.setdefault(canonical, []).append(alias)
        elif mapping_type == "attribute":
            attribute_aliases.setdefault(canonical, []).append(alias)
        elif mapping_type == "room":
            room_aliases.setdefault(canonical, []).append(alias)

    return {
        "object": {key: tuple(dict.fromkeys(values)) for key, values in object_aliases.items()},
        "attribute": {key: tuple(dict.fromkeys(values)) for key, values in attribute_aliases.items()},
        "room": {key: tuple(dict.fromkeys(values)) for key, values in room_aliases.items()},
    }
