from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .knowledge_models import KnowledgeContext, KnowledgeGuardResult, KnowledgeStatusSnapshot
from .knowledge_repository import PostgresKnowledgeRepository
from .knowledge_service import KnowledgeService


@dataclass(slots=True)
class KnowledgeRuntimeHandle:
    enabled: bool
    scene_scope: str | None = None
    service: KnowledgeService | None = None
    degraded_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.service is not None and self.status_snapshot().available

    def retrieve_for_plan(
        self,
        instruction: str,
        *,
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> KnowledgeContext:
        if self.service is None:
            return KnowledgeContext(
                hard_rules=[],
                soft_rules=[],
                lexicon_entries=[],
                facts=[],
                debug={
                    "enabled": bool(self.enabled),
                    "available": False,
                    "reason": self.degraded_reason,
                },
            )
        try:
            return self.service.retrieve_for_plan(
                instruction,
                scene_scope=_resolve_scene_scope(scene_scope, self.scene_scope),
                top_k=top_k,
            )
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return KnowledgeContext(
                hard_rules=[],
                soft_rules=[],
                lexicon_entries=[],
                facts=[],
                debug={
                    "enabled": bool(self.enabled),
                    "available": False,
                    "reason": self.degraded_reason,
                },
            )

    def evaluate_task_frame(
        self,
        task_frame: dict[str, Any],
        *,
        scene_scope: str | None = None,
        utterance: str | None = None,
    ) -> KnowledgeGuardResult:
        if self.service is None:
            return KnowledgeGuardResult(
                allowed=True,
                task_frame=dict(task_frame),
                reason=None,
            )
        try:
            return self.service.evaluate_task_frame(
                task_frame,
                scene_scope=_resolve_scene_scope(scene_scope, self.scene_scope),
                utterance=utterance,
            )
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return KnowledgeGuardResult(
                allowed=True,
                task_frame=dict(task_frame),
                reason=None,
            )

    def status_snapshot(self) -> KnowledgeStatusSnapshot:
        if self.service is None:
            return KnowledgeStatusSnapshot(
                enabled=bool(self.enabled),
                available=False,
                knowledge_enabled=bool(self.enabled),
                published_document_count=0,
                active_hard_rule_count=0,
                lexicon_entry_count=0,
                last_refresh_ok=None,
                last_applied_rule_ids=[],
                degraded_reason=self.degraded_reason,
            )
        snapshot = self.service.status_snapshot(enabled=self.enabled)
        if self.degraded_reason and snapshot.degraded_reason is None:
            return KnowledgeStatusSnapshot(
                enabled=snapshot.enabled,
                available=False,
                knowledge_enabled=snapshot.knowledge_enabled,
                published_document_count=snapshot.published_document_count,
                active_hard_rule_count=snapshot.active_hard_rule_count,
                lexicon_entry_count=snapshot.lexicon_entry_count,
                last_refresh_ok=snapshot.last_refresh_ok,
                last_applied_rule_ids=snapshot.last_applied_rule_ids,
                degraded_reason=self.degraded_reason,
            )
        return snapshot


def create_knowledge_runtime(
    *,
    dsn: str | None,
    object_memory_dsn: str | None = None,
    scene_scope: str | None = None,
) -> KnowledgeRuntimeHandle:
    normalized_dsn = str(dsn or "").strip() or str(object_memory_dsn or "").strip()
    if not normalized_dsn:
        return KnowledgeRuntimeHandle(
            enabled=False,
            scene_scope=_resolve_scene_scope(scene_scope, None),
        )

    try:
        repository = PostgresKnowledgeRepository(normalized_dsn)
        repository.apply_schema()
        service = KnowledgeService(repository)
        service.refresh_published_cache()
    except Exception as exc:  # noqa: BLE001
        return KnowledgeRuntimeHandle(
            enabled=True,
            scene_scope=_resolve_scene_scope(scene_scope, None),
            degraded_reason=f"{type(exc).__name__}: {exc}",
        )

    return KnowledgeRuntimeHandle(
        enabled=True,
        scene_scope=_resolve_scene_scope(scene_scope, None),
        service=service,
    )


def _resolve_scene_scope(primary: str | None, fallback: str | None) -> str | None:
    normalized = " ".join(str(primary or fallback or "").strip().split())
    return normalized or None
