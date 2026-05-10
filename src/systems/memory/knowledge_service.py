from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import re
import uuid
from typing import Any

from .knowledge_models import (
    KnowledgeContext,
    KnowledgeDocumentInput,
    KnowledgeDocumentRecord,
    KnowledgeFactChunk,
    KnowledgeGuardResult,
    KnowledgeLexiconEntry,
    KnowledgeRule,
    KnowledgeRuleAuditRecord,
    KnowledgeStatusSnapshot,
    content_hash,
    normalize_alias,
    normalize_document_status,
    normalize_lexicon_type,
    normalize_rule_action,
    normalize_rule_enforcement,
    normalize_scope_kind,
    normalize_scope_value,
    utc_now,
)
from .knowledge_repository import KnowledgeRepository


_FENCE_START_RE = re.compile(r"^```(knowledge-rule|knowledge-lexicon)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


class KnowledgeService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        *,
        default_fact_top_k: int = 5,
    ) -> None:
        self.repository = repository
        self.default_fact_top_k = max(1, int(default_fact_top_k))
        self._published_rules_cache: list[KnowledgeRule] = []
        self._published_lexicon_cache: list[KnowledgeLexiconEntry] = []
        self._published_document_count = 0
        self._active_hard_rule_count = 0
        self._lexicon_entry_count = 0
        self._last_refresh_ok: bool | None = None
        self._degraded_reason: str | None = None
        self._last_applied_rule_ids: list[str] = []
        self._cache_ready = False

    def status_snapshot(self, *, enabled: bool = True) -> KnowledgeStatusSnapshot:
        available = self._cache_ready
        return KnowledgeStatusSnapshot(
            enabled=bool(enabled),
            available=bool(available),
            knowledge_enabled=bool(enabled),
            published_document_count=self._published_document_count,
            active_hard_rule_count=self._active_hard_rule_count,
            lexicon_entry_count=self._lexicon_entry_count,
            last_refresh_ok=self._last_refresh_ok,
            last_applied_rule_ids=list(self._last_applied_rule_ids),
            degraded_reason=self._degraded_reason,
        )

    def register_document(
        self,
        document_input: KnowledgeDocumentInput,
        *,
        document_id: str | None = None,
    ) -> KnowledgeDocumentRecord:
        normalized_input = self._normalize_document_input(document_input)
        normalized_document_id = str(document_id or "").strip()
        existing = self.repository.get_document(normalized_document_id) if normalized_document_id else None
        if normalized_document_id and existing is None:
            raise KeyError(f"knowledge document not found: {normalized_document_id}")
        now = utc_now()
        next_status = "published" if normalized_input.publish else "draft"
        record = KnowledgeDocumentRecord(
            document_id=(existing.document_id if existing is not None else str(uuid.uuid4())),
            title=normalized_input.title,
            body_markdown=normalized_input.body_markdown,
            scope_kind=normalize_scope_kind(normalized_input.scope_kind),
            scope_value=normalize_scope_value(normalized_input.scope_value)
            if normalize_scope_kind(normalized_input.scope_kind) == "scene"
            else None,
            status=normalize_document_status(next_status),
            content_hash=content_hash(normalized_input.body_markdown),
            version=1 if existing is None else existing.version + 1,
            metadata=dict(normalized_input.metadata),
            created_at=now if existing is None else existing.created_at,
            updated_at=now,
            published_at=now if normalized_input.publish else None,
        )
        rules, lexicon_entries, chunks = self._compile_document(record)
        self.repository.upsert_document_bundle(
            record,
            rules=rules,
            lexicon_entries=lexicon_entries,
            chunks=chunks,
        )
        self._refresh_cache_fail_open()
        return record

    def publish_document(self, document_id: str) -> KnowledgeDocumentRecord:
        now = utc_now()
        record = self.repository.set_document_status(
            str(document_id).strip(),
            status="published",
            updated_at=now,
            published_at=now,
        )
        if record is None:
            raise KeyError(f"knowledge document not found: {document_id}")
        self._refresh_cache_fail_open()
        return record

    def archive_document(self, document_id: str) -> KnowledgeDocumentRecord:
        now = utc_now()
        record = self.repository.set_document_status(
            str(document_id).strip(),
            status="archived",
            updated_at=now,
            published_at=None,
        )
        if record is None:
            raise KeyError(f"knowledge document not found: {document_id}")
        self._refresh_cache_fail_open()
        return record

    def get_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        return self.repository.get_document(str(document_id).strip())

    def list_documents(
        self,
        *,
        statuses: tuple[str, ...] | None = None,
    ) -> list[KnowledgeDocumentRecord]:
        return self.repository.list_documents(statuses=statuses)

    def refresh_published_cache(self) -> KnowledgeStatusSnapshot:
        rules = self.repository.list_published_rules(scene_scope="__all__")
        lexicon_entries = self.repository.list_published_lexicon_entries(scene_scope="__all__")
        self._published_rules_cache = list(rules)
        self._published_lexicon_cache = list(lexicon_entries)
        self._published_document_count = self.repository.published_document_count()
        self._active_hard_rule_count = self.repository.active_hard_rule_count()
        self._lexicon_entry_count = self.repository.lexicon_entry_count()
        self._last_refresh_ok = True
        self._degraded_reason = None
        self._cache_ready = True
        return self.status_snapshot()

    def retrieve_for_plan(
        self,
        utterance: str,
        *,
        scene_scope: str | None = None,
        top_k: int | None = None,
    ) -> KnowledgeContext:
        self._ensure_cache()
        hard_rules = self._cached_rules(scene_scope=scene_scope, enforcement="hard")
        soft_rules = self._cached_rules(scene_scope=scene_scope, enforcement="soft")
        lexicon_entries = self._cached_lexicon_entries(scene_scope=scene_scope)
        facts: list[KnowledgeFactChunk] = []
        if self._cache_ready:
            try:
                top_k_value = self.default_fact_top_k if top_k is None else max(1, int(top_k))
                facts = self.repository.search_published_chunks(
                    utterance,
                    scene_scope=scene_scope,
                    top_k=top_k_value,
                )
                if not facts:
                    expanded_query = _expand_query_with_lexicon(utterance, lexicon_entries)
                    if expanded_query != utterance:
                        facts = self.repository.search_published_chunks(
                            expanded_query,
                            scene_scope=scene_scope,
                            top_k=top_k_value,
                        )
            except Exception as exc:  # noqa: BLE001
                self._last_refresh_ok = False
                self._degraded_reason = f"{type(exc).__name__}: {exc}"
                facts = []
        return KnowledgeContext(
            hard_rules=hard_rules,
            soft_rules=soft_rules,
            lexicon_entries=lexicon_entries,
            facts=facts,
            debug={
                "scene_scope": scene_scope,
                "hard_rule_count": len(hard_rules),
                "soft_rule_count": len(soft_rules),
                "lexicon_entry_count": len(lexicon_entries),
                "fact_count": len(facts),
                "cache_ready": self._cache_ready,
                "degraded_reason": self._degraded_reason,
            },
        )

    def evaluate_task_frame(
        self,
        task_frame: dict[str, Any],
        *,
        scene_scope: str | None = None,
        utterance: str | None = None,
    ) -> KnowledgeGuardResult:
        self._ensure_cache()
        hard_rules = self._cached_rules(scene_scope=scene_scope, enforcement="hard")
        result = _apply_hard_rules(task_frame, hard_rules, utterance=utterance)
        self._last_applied_rule_ids = list(result.applied_rule_ids)
        return result

    def audit_rule_application(
        self,
        *,
        rule_ids: list[str],
        phase: str,
        task_id: str | None,
        subgoal_id: str | None,
        payload: dict[str, Any],
    ) -> None:
        if not rule_ids:
            return
        rules_by_id = {row.rule_id: row for row in self._published_rules_cache}
        for rule_id in rule_ids:
            rule = rules_by_id.get(rule_id)
            if rule is None:
                continue
            try:
                self.repository.insert_rule_audit(
                    KnowledgeRuleAuditRecord(
                        audit_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        document_id=rule.document_id,
                        phase=phase,
                        task_id=task_id,
                        subgoal_id=subgoal_id,
                        applied_at=utc_now(),
                        payload=dict(payload),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                self._last_refresh_ok = False
                self._degraded_reason = f"{type(exc).__name__}: {exc}"
                return

    def _ensure_cache(self) -> None:
        if self._cache_ready:
            return
        self._refresh_cache_fail_open()

    def _refresh_cache_fail_open(self) -> None:
        try:
            self.refresh_published_cache()
        except Exception as exc:  # noqa: BLE001
            self._last_refresh_ok = False
            self._degraded_reason = f"{type(exc).__name__}: {exc}"

    def _cached_rules(
        self,
        *,
        scene_scope: str | None,
        enforcement: str,
    ) -> list[KnowledgeRule]:
        rows = [
            row
            for row in self._published_rules_cache
            if row.enforcement == enforcement and _scope_matches(row.scope_kind, row.scope_value, scene_scope)
        ]
        rows.sort(
            key=lambda row: (
                row.priority,
                _rule_specificity(row),
                row.published_at.timestamp() if row.published_at is not None else row.updated_at.timestamp(),
            ),
            reverse=True,
        )
        return rows

    def _cached_lexicon_entries(self, *, scene_scope: str | None) -> list[KnowledgeLexiconEntry]:
        rows = [
            row
            for row in self._published_lexicon_cache
            if _scope_matches(row.scope_kind, row.scope_value, scene_scope)
        ]
        rows.sort(key=lambda row: row.updated_at, reverse=True)
        return rows

    def _normalize_document_input(self, document_input: KnowledgeDocumentInput) -> KnowledgeDocumentInput:
        title = " ".join(str(document_input.title).strip().split())
        if not title:
            raise ValueError("knowledge document title is required")
        if not isinstance(document_input.body_markdown, str):
            raise ValueError("knowledge document body_markdown must be a string")
        scope_kind = normalize_scope_kind(document_input.scope_kind)
        scope_value = normalize_scope_value(document_input.scope_value) if scope_kind == "scene" else None
        if scope_kind == "scene" and scope_value is None:
            raise ValueError("knowledge scene scope requires scope_value")
        return KnowledgeDocumentInput(
            title=title,
            body_markdown=document_input.body_markdown,
            scope_kind=scope_kind,
            scope_value=scope_value,
            publish=bool(document_input.publish),
            metadata=dict(document_input.metadata),
        )

    def _compile_document(
        self,
        document: KnowledgeDocumentRecord,
    ) -> tuple[list[KnowledgeRule], list[KnowledgeLexiconEntry], list[KnowledgeFactChunk]]:
        now = document.updated_at
        parsed = _parse_markdown_blocks(document.body_markdown)
        rules: list[KnowledgeRule] = []
        lexicon_entries: list[KnowledgeLexiconEntry] = []
        for block_index, block in enumerate(parsed["rule_blocks"], start=1):
            payload = _load_json_block(block["body"], block_kind="knowledge-rule")
            specs = payload if isinstance(payload, list) else [payload]
            for item_index, spec in enumerate(specs, start=1):
                rules.append(
                    _normalize_rule_spec(
                        spec,
                        document=document,
                        source_anchor=block["anchor"],
                        created_at=now,
                        updated_at=now,
                        fallback_key=f"rule-{block_index}-{item_index}",
                    )
                )
        for block_index, block in enumerate(parsed["lexicon_blocks"], start=1):
            payload = _load_json_block(block["body"], block_kind="knowledge-lexicon")
            specs = payload if isinstance(payload, list) else [payload]
            for item_index, spec in enumerate(specs, start=1):
                lexicon_entries.append(
                    _normalize_lexicon_spec(
                        spec,
                        document=document,
                        source_anchor=block["anchor"],
                        created_at=now,
                        updated_at=now,
                        fallback_key=f"lexicon-{block_index}-{item_index}",
                    )
                )
        chunks: list[KnowledgeFactChunk] = []
        for chunk_index, prose in enumerate(parsed["prose_chunks"], start=1):
            if not prose["text"].strip():
                continue
            chunks.append(
                KnowledgeFactChunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=document.document_id,
                    chunk_index=chunk_index,
                    text=prose["text"].strip(),
                    scope_kind=document.scope_kind,
                    scope_value=document.scope_value,
                    source_anchor=prose["anchor"],
                    created_at=now,
                    updated_at=now,
                )
            )
        return rules, lexicon_entries, chunks


def _parse_markdown_blocks(markdown: str) -> dict[str, list[dict[str, str]]]:
    current_anchor: str | None = None
    paragraph_lines: list[str] = []
    prose_chunks: list[dict[str, str]] = []
    rule_blocks: list[dict[str, str]] = []
    lexicon_blocks: list[dict[str, str]] = []
    block_kind: str | None = None
    block_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = "\n".join(paragraph_lines).strip()
        if not text:
            paragraph_lines.clear()
            return
        prose_chunks.append(
            {
                "anchor": current_anchor or f"section-{len(prose_chunks) + 1}",
                "text": text,
            }
        )
        paragraph_lines.clear()

    for raw_line in str(markdown).splitlines():
        line = raw_line.rstrip()
        if block_kind is not None:
            if line.strip() == "```":
                target = rule_blocks if block_kind == "knowledge-rule" else lexicon_blocks
                target.append(
                    {
                        "anchor": current_anchor or f"block-{len(target) + 1}",
                        "body": "\n".join(block_lines).strip(),
                    }
                )
                block_kind = None
                block_lines = []
            else:
                block_lines.append(raw_line)
            continue

        fence_match = _FENCE_START_RE.match(line.strip())
        if fence_match is not None:
            flush_paragraph()
            block_kind = str(fence_match.group(1))
            block_lines = []
            continue

        heading_match = _HEADING_RE.match(line.strip())
        if heading_match is not None:
            flush_paragraph()
            current_anchor = _anchor_from_heading(str(heading_match.group(2)))
            continue

        if not line.strip():
            flush_paragraph()
            continue
        paragraph_lines.append(raw_line)

    flush_paragraph()
    return {
        "rule_blocks": rule_blocks,
        "lexicon_blocks": lexicon_blocks,
        "prose_chunks": prose_chunks,
    }


def _load_json_block(raw: str, *, block_kind: str) -> Any:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {block_kind} json: {exc}") from exc
    if not isinstance(payload, (dict, list)):
        raise ValueError(f"{block_kind} block must contain a JSON object or array")
    return payload


def _normalize_rule_spec(
    spec: Any,
    *,
    document: KnowledgeDocumentRecord,
    source_anchor: str | None,
    created_at,
    updated_at,
    fallback_key: str,
) -> KnowledgeRule:
    if not isinstance(spec, dict):
        raise ValueError("knowledge rule entries must be JSON objects")
    action = normalize_rule_action(spec.get("action"))
    enforcement = normalize_rule_enforcement(spec.get("enforcement"))
    conditions = spec.get("conditions")
    if conditions is None:
        conditions = spec.get("when")
    if conditions is None:
        conditions = {}
    if not isinstance(conditions, dict):
        raise ValueError("knowledge rule conditions must be an object")
    params = spec.get("params")
    if params is None:
        params = spec.get("then")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("knowledge rule params must be an object")
    normalized_conditions = _normalize_rule_conditions(conditions or spec)
    normalized_params = _normalize_rule_params(action, params or spec)
    return KnowledgeRule(
        rule_id=str(uuid.uuid4()),
        document_id=document.document_id,
        rule_key=str(spec.get("id") or spec.get("rule_key") or fallback_key),
        scope_kind=document.scope_kind,
        scope_value=document.scope_value,
        enforcement=enforcement,
        action=action,
        conditions=normalized_conditions,
        params=normalized_params,
        priority=int(spec.get("priority") or 0),
        reason=_optional_string(spec.get("reason") or normalized_params.get("reason")),
        source_anchor=source_anchor,
        created_at=created_at,
        updated_at=updated_at,
        published_at=document.published_at,
    )


def _normalize_rule_conditions(raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    intent_values = _normalize_condition_values(raw.get("intent") or raw.get("intents"))
    if intent_values:
        normalized["intent"] = intent_values
    object_values = _normalize_condition_values(
        raw.get("target_object") or raw.get("target_objects") or raw.get("target_class") or raw.get("target_classes")
    )
    if object_values:
        normalized["target_object"] = object_values
    attribute_values = _normalize_condition_values(raw.get("attribute") or raw.get("attributes"))
    if attribute_values:
        normalized["attribute"] = attribute_values
    room_values = _normalize_condition_values(raw.get("room") or raw.get("rooms"))
    if room_values:
        normalized["room"] = room_values
    utterance_values = _normalize_condition_values(raw.get("utterance_contains"))
    if utterance_values:
        normalized["utterance_contains"] = utterance_values
    return normalized


def _normalize_rule_params(action: str, raw: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if action == "force_target_room":
        room = _optional_string(raw.get("room") or raw.get("canonical_room") or raw.get("force_target_room"))
        if room is None:
            raise ValueError("force_target_room requires room")
        normalized["room"] = room
    if action == "restrict_query_attributes":
        allowed_attributes = _normalize_condition_values(raw.get("allowed_attributes") or raw.get("attributes"))
        if not allowed_attributes:
            raise ValueError("restrict_query_attributes requires allowed_attributes")
        normalized["allowed_attributes"] = allowed_attributes
    if action == "require_clarification":
        normalized["question"] = _optional_string(raw.get("question") or raw.get("question_ko"))
    reason = _optional_string(raw.get("reason"))
    if reason is not None:
        normalized["reason"] = reason
    return normalized


def _normalize_lexicon_spec(
    spec: Any,
    *,
    document: KnowledgeDocumentRecord,
    source_anchor: str | None,
    created_at,
    updated_at,
    fallback_key: str,
) -> KnowledgeLexiconEntry:
    del fallback_key
    if not isinstance(spec, dict):
        raise ValueError("knowledge lexicon entries must be JSON objects")

    if spec.get("object_alias") is not None:
        mapping_type = "object"
        alias = spec.get("object_alias")
        canonical = spec.get("canonical_object")
    elif spec.get("attribute_alias") is not None:
        mapping_type = "attribute"
        alias = spec.get("attribute_alias")
        canonical = spec.get("canonical_attribute")
    elif spec.get("room_alias") is not None:
        mapping_type = "room"
        alias = spec.get("room_alias")
        canonical = spec.get("canonical_room")
    else:
        mapping_type = normalize_lexicon_type(spec.get("mapping_type") or spec.get("type"))
        alias = spec.get("alias")
        canonical = spec.get("canonical")

    alias_value = normalize_alias(str(alias or ""))
    canonical_value = normalize_alias(str(canonical or ""))
    if not alias_value or not canonical_value:
        raise ValueError("knowledge lexicon alias and canonical values are required")

    return KnowledgeLexiconEntry(
        entry_id=str(uuid.uuid4()),
        document_id=document.document_id,
        mapping_type=mapping_type,
        alias=alias_value,
        canonical=canonical_value,
        scope_kind=document.scope_kind,
        scope_value=document.scope_value,
        source_anchor=source_anchor,
        created_at=created_at,
        updated_at=updated_at,
    )


def _apply_hard_rules(
    task_frame: dict[str, Any],
    hard_rules: list[KnowledgeRule],
    *,
    utterance: str | None,
) -> KnowledgeGuardResult:
    working = deepcopy(task_frame)
    applied_rule_ids: list[str] = []
    mutated = False

    for rule in hard_rules:
        if not _rule_matches(rule, working, utterance=utterance):
            continue
        applied_rule_ids.append(rule.rule_id)

        if rule.action == "force_target_room":
            target = deepcopy(working.get("target") or {})
            forced_room = _optional_string(rule.params.get("room"))
            if forced_room and target.get("location_hint") != forced_room:
                target["location_hint"] = forced_room
                working["target"] = target
                mutated = True
            continue

        if rule.action == "deny_task":
            return KnowledgeGuardResult(
                allowed=False,
                task_frame=_unsupported_task_frame(working, reason=rule.reason or rule.params.get("reason")),
                applied_rule_ids=applied_rule_ids,
                reason=_optional_string(rule.reason or rule.params.get("reason") or "knowledge_rule_denied"),
                mutated=mutated,
            )

        if rule.action == "require_clarification":
            question = _optional_string(rule.params.get("question") or rule.reason) or "Please clarify the request."
            return KnowledgeGuardResult(
                allowed=False,
                task_frame=_clarification_task_frame(working, question),
                applied_rule_ids=applied_rule_ids,
                reason=question,
                mutated=mutated,
            )

        if rule.action == "restrict_query_attributes":
            allowed_attributes = [str(value) for value in rule.params.get("allowed_attributes", [])]
            current_attribute = _optional_string(_nested_get(working, "query", "attribute"))
            if current_attribute in allowed_attributes:
                continue
            if not allowed_attributes:
                continue
            question = (
                _optional_string(rule.reason)
                or f"Allowed attributes are: {', '.join(allowed_attributes)}."
            )
            return KnowledgeGuardResult(
                allowed=False,
                task_frame=_clarification_task_frame(working, question),
                applied_rule_ids=applied_rule_ids,
                reason=question,
                mutated=mutated,
            )

    return KnowledgeGuardResult(
        allowed=True,
        task_frame=working,
        applied_rule_ids=applied_rule_ids,
        reason=None,
        mutated=mutated,
    )


def _rule_matches(
    rule: KnowledgeRule,
    task_frame: dict[str, Any],
    *,
    utterance: str | None,
) -> bool:
    conditions = rule.conditions
    if not conditions:
        return True
    if not _matches_any(conditions.get("intent"), task_frame.get("intent")):
        return False
    target = task_frame.get("target") if isinstance(task_frame.get("target"), dict) else {}
    query = task_frame.get("query") if isinstance(task_frame.get("query"), dict) else {}
    if not _matches_any(conditions.get("target_object"), target.get("object")):
        return False
    if not _matches_any(conditions.get("attribute"), query.get("attribute")):
        return False
    if not _matches_any(conditions.get("room"), target.get("location_hint")):
        return False
    utterance_contains = conditions.get("utterance_contains")
    if utterance_contains:
        normalized_utterance = normalize_alias(utterance or "")
        if not normalized_utterance:
            return False
        if not any(term in normalized_utterance for term in utterance_contains):
            return False
    return True


def _matches_any(allowed: Any, actual: Any) -> bool:
    if not allowed:
        return True
    if actual is None:
        return False
    normalized_actual = normalize_alias(str(actual))
    return normalized_actual in {normalize_alias(str(value)) for value in allowed}


def _normalize_condition_values(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [normalize_alias(str(item)) for item in raw if str(item).strip()]
    normalized = normalize_alias(str(raw))
    return [normalized] if normalized else []


def _rule_specificity(rule: KnowledgeRule) -> int:
    score = 0
    for key in ("intent", "target_object", "attribute", "room", "utterance_contains"):
        value = rule.conditions.get(key)
        if isinstance(value, list) and value:
            score += 1
    return score


def _anchor_from_heading(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", normalize_alias(value))
    return normalized.strip("-") or "section"


def _scope_matches(scope_kind: str, scope_value: str | None, scene_scope: str | None) -> bool:
    if scope_kind == "global":
        return True
    if scope_kind != "scene":
        return False
    return normalize_alias(scope_value or "") == normalize_alias(scene_scope or "")


def _clarification_task_frame(task_frame: dict[str, Any], question: str) -> dict[str, Any]:
    working = deepcopy(task_frame)
    working["intent"] = "ask_clarification"
    working["constraints"] = {"return_after_check": False, "report_result": True}
    clarification = deepcopy(working.get("clarification") or {})
    clarification["required"] = True
    clarification["question_ko"] = question
    working["clarification"] = clarification
    return working


def _unsupported_task_frame(task_frame: dict[str, Any], *, reason: Any) -> dict[str, Any]:
    working = deepcopy(task_frame)
    working["intent"] = "unsupported"
    working["constraints"] = {"return_after_check": False, "report_result": True}
    clarification = deepcopy(working.get("clarification") or {})
    clarification["required"] = False
    clarification["question_ko"] = _optional_string(reason)
    working["clarification"] = clarification
    return working


def _optional_string(value: Any) -> str | None:
    normalized = " ".join(str(value or "").strip().split())
    return normalized or None


def _expand_query_with_lexicon(query: str, entries: list[KnowledgeLexiconEntry]) -> str:
    normalized_query = str(query)
    lowered = normalized_query.lower()
    expanded_query = normalized_query
    for entry in entries:
        if not entry.alias or entry.alias not in lowered:
            continue
        expanded_query = re.sub(
            rf"\b{re.escape(entry.alias)}\b",
            entry.canonical,
            expanded_query,
            flags=re.IGNORECASE,
        )
    return expanded_query


def _nested_get(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
