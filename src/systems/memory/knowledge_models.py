from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import re
from typing import Any


ALLOWED_KNOWLEDGE_SCOPE_KINDS = ("global", "scene")
ALLOWED_KNOWLEDGE_DOCUMENT_STATUSES = ("draft", "published", "archived")
ALLOWED_KNOWLEDGE_RULE_ACTIONS = (
    "deny_task",
    "require_clarification",
    "force_target_room",
    "restrict_query_attributes",
)
ALLOWED_KNOWLEDGE_RULE_ENFORCEMENTS = ("hard", "soft")
ALLOWED_KNOWLEDGE_LEXICON_TYPES = ("object", "attribute", "room")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_scope_kind(value: str | None) -> str:
    normalized = str(value or "global").strip().lower()
    return normalized if normalized in ALLOWED_KNOWLEDGE_SCOPE_KINDS else "global"


def normalize_scope_value(value: str | None) -> str | None:
    normalized = " ".join(str(value or "").strip().split())
    return normalized or None


def normalize_document_status(value: str | None) -> str:
    normalized = str(value or "draft").strip().lower()
    return normalized if normalized in ALLOWED_KNOWLEDGE_DOCUMENT_STATUSES else "draft"


def normalize_rule_action(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in ALLOWED_KNOWLEDGE_RULE_ACTIONS:
        raise ValueError(f"unsupported knowledge rule action: {value}")
    return normalized


def normalize_rule_enforcement(value: str | None) -> str:
    normalized = str(value or "hard").strip().lower()
    if normalized not in ALLOWED_KNOWLEDGE_RULE_ENFORCEMENTS:
        raise ValueError(f"unsupported knowledge rule enforcement: {value}")
    return normalized


def normalize_lexicon_type(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in ALLOWED_KNOWLEDGE_LEXICON_TYPES:
        raise ValueError(f"unsupported knowledge lexicon type: {value}")
    return normalized


def normalize_alias(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def content_hash(markdown: str) -> str:
    return hashlib.sha256(str(markdown).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentInput:
    title: str
    body_markdown: str
    scope_kind: str = "global"
    scope_value: str | None = None
    publish: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KnowledgeDocumentRecord:
    document_id: str
    title: str
    body_markdown: str
    scope_kind: str
    scope_value: str | None
    status: str
    content_hash: str
    version: int
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeRule:
    rule_id: str
    document_id: str
    rule_key: str
    scope_kind: str
    scope_value: str | None
    enforcement: str
    action: str
    conditions: dict[str, Any]
    params: dict[str, Any]
    priority: int
    reason: str | None
    source_anchor: str | None
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeLexiconEntry:
    entry_id: str
    document_id: str
    mapping_type: str
    alias: str
    canonical: str
    scope_kind: str
    scope_value: str | None
    source_anchor: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class KnowledgeFactChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    scope_kind: str
    scope_value: str | None
    source_anchor: str | None
    created_at: datetime
    updated_at: datetime
    rank: float | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeRuleAuditRecord:
    audit_id: str
    rule_id: str
    document_id: str
    phase: str
    task_id: str | None
    subgoal_id: str | None
    applied_at: datetime
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class KnowledgeContext:
    hard_rules: list[KnowledgeRule]
    soft_rules: list[KnowledgeRule]
    lexicon_entries: list[KnowledgeLexiconEntry]
    facts: list[KnowledgeFactChunk]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class KnowledgeStatusSnapshot:
    enabled: bool
    available: bool
    knowledge_enabled: bool
    published_document_count: int
    active_hard_rule_count: int
    lexicon_entry_count: int
    last_refresh_ok: bool | None
    last_applied_rule_ids: list[str]
    degraded_reason: str | None = None


@dataclass(frozen=True, slots=True)
class KnowledgeGuardResult:
    allowed: bool
    task_frame: dict[str, Any]
    applied_rule_ids: list[str] = field(default_factory=list)
    reason: str | None = None
    mutated: bool = False
