from __future__ import annotations

from dataclasses import dataclass
import os

from .object_memory_models import MemoryNavigationResolution, ObjectMemoryContext
from .object_memory_repository import PostgresObjectMemoryRepository
from .object_memory_service import ObjectMemoryService


DEFAULT_OBJECT_MEMORY_USER_ID = "local-operator"
TRUE_VALUES = frozenset(("1", "true", "yes", "on"))


@dataclass(slots=True)
class ObjectMemoryRuntimeHandle:
    enabled: bool
    user_id: str
    service: ObjectMemoryService | None = None
    degraded_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.service is not None and self.degraded_reason is None

    def recent_context(
        self,
        *,
        class_name: str | None = None,
        room_id: str | None = None,
        scene_scope: str | None = None,
        top_k: int = 10,
        max_age_sec: int = 86400,
    ) -> ObjectMemoryContext:
        if self.service is None:
            return ObjectMemoryContext(
                user_id=self.user_id,
                entries=[],
                recent_seen=[],
                debug={
                    "enabled": bool(self.enabled),
                    "available": False,
                    "reason": self.degraded_reason,
                },
            )
        try:
            return self.service.query_recent_objects(
                self.user_id,
                class_name=class_name,
                room_id=room_id,
                scene_scope=scene_scope,
                top_k=top_k,
                max_age_sec=max_age_sec,
            )
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return ObjectMemoryContext(
                user_id=self.user_id,
                entries=[],
                recent_seen=[],
                debug={
                    "enabled": bool(self.enabled),
                    "available": False,
                    "reason": self.degraded_reason,
                },
            )

    def resolve_navigation_target(
        self,
        *,
        class_name: str,
        scene_scope: str | None,
        room_hint: str | None = None,
        instance_hint: str | None = None,
        max_pose_age_sec: int = 600,
    ) -> MemoryNavigationResolution:
        if self.service is None:
            return MemoryNavigationResolution(
                status="no_candidate",
                debug={
                    "enabled": bool(self.enabled),
                    "available": False,
                    "reason": self.degraded_reason,
                },
            )
        try:
            return self.service.resolve_memory_navigation_target(
                self.user_id,
                scene_scope=scene_scope,
                class_name=class_name,
                room_hint=room_hint,
                instance_hint=instance_hint,
                max_pose_age_sec=max_pose_age_sec,
            )
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return MemoryNavigationResolution(
                status="no_candidate",
                debug={
                    "enabled": bool(self.enabled),
                    "available": False,
                    "reason": self.degraded_reason,
                },
            )

    def count_objects(self) -> int:
        if self.service is None:
            return 0
        try:
            return self.service.repository.count_object_entries(self.user_id, statuses=("active",))
        except Exception as exc:  # noqa: BLE001
            self.degraded_reason = f"{type(exc).__name__}: {exc}"
            return 0


def normalize_object_memory_user_id(value: str | None) -> str:
    normalized = str(value or "").strip()
    return normalized or DEFAULT_OBJECT_MEMORY_USER_ID


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in TRUE_VALUES


def create_object_memory_runtime(
    *,
    enabled: bool,
    dsn: str | None,
    user_id: str | None,
    auto_migrate: bool | None = None,
) -> ObjectMemoryRuntimeHandle:
    normalized_user_id = normalize_object_memory_user_id(user_id)
    if not enabled:
        return ObjectMemoryRuntimeHandle(enabled=False, user_id=normalized_user_id)

    normalized_dsn = str(dsn or "").strip()
    if not normalized_dsn:
        return ObjectMemoryRuntimeHandle(
            enabled=True,
            user_id=normalized_user_id,
            degraded_reason="object_memory_dsn_missing",
        )

    try:
        repository = PostgresObjectMemoryRepository(normalized_dsn)
        should_auto_migrate = (
            _env_flag("AURA_OBJECT_MEMORY_AUTO_MIGRATE")
            if auto_migrate is None
            else bool(auto_migrate)
        )
        if should_auto_migrate:
            repository.apply_schema()
        repository.verify_schema()
    except Exception as exc:  # noqa: BLE001
        return ObjectMemoryRuntimeHandle(
            enabled=True,
            user_id=normalized_user_id,
            degraded_reason=f"{type(exc).__name__}: {exc}",
        )

    return ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id=normalized_user_id,
        service=ObjectMemoryService(repository),
    )
