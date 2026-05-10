from __future__ import annotations

from dataclasses import dataclass, field, replace
from threading import Lock

from systems.reasoning.planner_catalog_errors import PlannerCatalogUnavailableError
from systems.reasoning.planner_catalog_models import (
    PlannerCatalogSnapshot,
    PlannerCatalogStatusSnapshot,
    default_catalog_snapshot,
)
from systems.reasoning.planner_catalog_repository import PostgresPlannerCatalogRepository
from systems.reasoning.planner_catalog_service import PlannerCatalogService


@dataclass(slots=True)
class PlannerCatalogRuntimeHandle:
    enabled: bool
    service: PlannerCatalogService | None = None
    degraded_reason: str | None = None
    _last_snapshot: PlannerCatalogSnapshot = field(default_factory=default_catalog_snapshot)
    _last_refresh_ok: bool | None = None
    _lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def available(self) -> bool:
        return self.service is not None and self._last_snapshot.source == "database" and bool(self._last_refresh_ok)

    @property
    def writable(self) -> bool:
        return self.service is not None and self._last_snapshot.source == "database"

    def snapshot(self) -> PlannerCatalogSnapshot:
        with self._lock:
            if self.service is None:
                return replace(
                    self._last_snapshot,
                    source="default",
                    writable=False,
                    degraded_reason=self.degraded_reason,
                )
            try:
                snapshot = self.service.load_snapshot()
            except Exception as exc:  # noqa: BLE001
                self.degraded_reason = f"{type(exc).__name__}: {exc}"
                self._last_refresh_ok = False
                fallback_source = "last_good" if self._last_snapshot.source == "database" else "default"
                return replace(
                    self._last_snapshot,
                    source=fallback_source,
                    writable=False,
                    degraded_reason=self.degraded_reason,
                )
            self._last_snapshot = replace(snapshot, source="database", writable=True, degraded_reason=None)
            self.degraded_reason = None
            self._last_refresh_ok = True
            return self._last_snapshot

    def snapshot_and_status(self) -> tuple[PlannerCatalogSnapshot, PlannerCatalogStatusSnapshot]:
        snapshot = self.snapshot()
        return snapshot, self.status_snapshot(snapshot)

    def status_snapshot(self, snapshot: PlannerCatalogSnapshot | None = None) -> PlannerCatalogStatusSnapshot:
        resolved_snapshot = snapshot if snapshot is not None else self._last_snapshot
        available = self.service is not None and resolved_snapshot.source == "database" and bool(self._last_refresh_ok)
        writable = self.service is not None and resolved_snapshot.source == "database"
        return PlannerCatalogStatusSnapshot(
            enabled=bool(self.enabled),
            available=available,
            writable=writable,
            source=resolved_snapshot.source,
            degraded_reason=resolved_snapshot.degraded_reason or self.degraded_reason,
            last_refresh_ok=self._last_refresh_ok,
            active_intent_count=len(resolved_snapshot.intents),
            active_subgoal_template_count=resolved_snapshot.active_subgoal_template_count,
        )

    def create_intent(self, intent_key: str) -> PlannerCatalogSnapshot:
        service = self._require_service()
        snapshot = service.create_intent(intent_key)
        return self._accept_mutation_snapshot(snapshot)

    def delete_intent(self, intent_id: str) -> PlannerCatalogSnapshot:
        service = self._require_service()
        snapshot = service.delete_intent(intent_id)
        return self._accept_mutation_snapshot(snapshot)

    def create_subgoal_template(
        self,
        *,
        intent_id: str,
        sequence_no: int,
        subgoal_type: str,
        activation_condition: str,
    ) -> PlannerCatalogSnapshot:
        service = self._require_service()
        snapshot = service.create_subgoal_template(
            intent_id=intent_id,
            sequence_no=sequence_no,
            subgoal_type=subgoal_type,
            activation_condition=activation_condition,
        )
        return self._accept_mutation_snapshot(snapshot)

    def delete_subgoal_template(self, template_id: str) -> PlannerCatalogSnapshot:
        service = self._require_service()
        snapshot = service.delete_subgoal_template(template_id)
        return self._accept_mutation_snapshot(snapshot)

    def _require_service(self) -> PlannerCatalogService:
        if self.service is None:
            raise PlannerCatalogUnavailableError("planner catalog storage is unavailable")
        return self.service

    def _accept_mutation_snapshot(self, snapshot: PlannerCatalogSnapshot) -> PlannerCatalogSnapshot:
        with self._lock:
            self._last_snapshot = replace(snapshot, source="database", writable=True, degraded_reason=None)
            self.degraded_reason = None
            self._last_refresh_ok = True
            return self._last_snapshot


def create_planner_catalog_runtime(
    *,
    dsn: str | None,
    knowledge_dsn: str | None = None,
    object_memory_dsn: str | None = None,
) -> PlannerCatalogRuntimeHandle:
    normalized_dsn = (
        str(dsn or "").strip()
        or str(knowledge_dsn or "").strip()
        or str(object_memory_dsn or "").strip()
    )
    if not normalized_dsn:
        return PlannerCatalogRuntimeHandle(enabled=False)

    try:
        repository = PostgresPlannerCatalogRepository(normalized_dsn)
        repository.apply_schema()
        service = PlannerCatalogService(repository)
        seeded_snapshot = service.ensure_seed_data()
    except Exception as exc:  # noqa: BLE001
        return PlannerCatalogRuntimeHandle(
            enabled=True,
            degraded_reason=f"{type(exc).__name__}: {exc}",
        )

    return PlannerCatalogRuntimeHandle(
        enabled=True,
        service=service,
        degraded_reason=None,
        _last_snapshot=replace(seeded_snapshot, source="database", writable=True, degraded_reason=None),
        _last_refresh_ok=True,
    )
