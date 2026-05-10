from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


OBJECT_MEMORY_STATUSES = ("active", "stale", "deleted")
OBJECT_INDEX_STATUSES = ("pending", "ready", "failed")
MEMORY_NAVIGATION_RESOLUTION_STATUSES = ("resolved", "ambiguous", "no_candidate", "stale_only")
OBJECT_MEMORY_PERSISTENCE_POLICIES = ("audit_all", "sparse")
OBJECT_MEMORY_LINK_STATUSES = ("created", "linked")

ObjectMemoryStatus = Literal["active", "stale", "deleted"]
ObjectIndexStatus = Literal["pending", "ready", "failed"]
MemoryNavigationResolutionStatus = Literal["resolved", "ambiguous", "no_candidate", "stale_only"]
ObjectMemoryPersistencePolicy = Literal["audit_all", "sparse"]
ObjectMemoryLinkStatus = Literal["created", "linked"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ObjectObservationInput:
    frame_idx: int
    track_id: str
    class_name: str
    detector_conf: float
    bbox_xyxy_norm: tuple[float, float, float, float]
    box_area: float
    aspect_ratio: float
    image_hash: str
    appearance_embedding: list[float] = field(default_factory=list)
    observed_at: datetime = field(default_factory=utc_now)
    mask_area: float | None = None
    room_id: str | None = None
    scene_scope: str | None = None
    world_pose_xyz: tuple[float, float, float] | None = None
    world_pose_observed_at: datetime | None = None
    source_id: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectObservation:
    observation_id: str
    object_id: str
    user_id: str
    session_id: str
    source_id: str | None
    frame_idx: int
    observed_at: datetime
    track_id: str
    class_name: str
    detector_conf: float
    room_id: str | None
    scene_scope: str | None
    bbox_xyxy_norm: tuple[float, float, float, float]
    box_area: float
    aspect_ratio: float
    image_hash: str
    world_pose_xyz: tuple[float, float, float] | None = None
    world_pose_observed_at: datetime | None = None
    appearance_embedding: list[float] = field(default_factory=list)
    appearance_model: str = "unknown"
    mask_area: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectMemoryEntry:
    object_id: str
    user_id: str
    canonical_class: str
    room_id: str | None
    scene_scope: str | None
    status: ObjectMemoryStatus
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int
    last_source_id: str | None
    last_session_id: str
    last_bbox_xyxy_norm: tuple[float, float, float, float]
    last_box_area: float
    last_aspect_ratio: float
    last_detector_conf: float
    appearance_count: int
    dedupe_confidence: float
    world_pose_xyz: tuple[float, float, float] | None = None
    world_pose_observed_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class ObjectMemoryEmbedding:
    object_id: str
    user_id: str
    model_name: str
    embedding: list[float] | None
    index_status: ObjectIndexStatus
    embedded_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class DuplicateMatch:
    object_id: str
    score: float
    appearance_score: float
    spatial_score: float
    temporal_score: float
    class_score: float


@dataclass(frozen=True)
class RetrievedObjectMemory:
    object_id: str
    canonical_class: str
    room_id: str | None
    scene_scope: str | None
    status: ObjectMemoryStatus
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int
    dedupe_confidence: float
    last_detector_conf: float
    world_pose_xyz: tuple[float, float, float] | None
    world_pose_observed_at: datetime | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ObjectMemoryNavigationCandidate:
    object_id: str
    class_name: str
    room_id: str | None
    scene_scope: str | None
    world_pose_xyz: tuple[float, float, float]
    world_pose_observed_at: datetime
    pose_age_sec: int
    last_seen_at: datetime
    dedupe_confidence: float
    last_detector_conf: float
    metadata: dict[str, Any]


@dataclass(frozen=True)
class MemoryNavigationResolution:
    status: MemoryNavigationResolutionStatus
    selected: ObjectMemoryNavigationCandidate | None = None
    candidates: list[ObjectMemoryNavigationCandidate] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectMemoryContext:
    user_id: str
    entries: list[RetrievedObjectMemory]
    recent_seen: list[dict[str, Any]]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObserveResult:
    created_object_ids: list[str]
    updated_object_ids: list[str]
    observation_ids: list[str]
    links: list["ObservedObjectLink"] = field(default_factory=list)
    suppressed_observation_count: int = 0


@dataclass(frozen=True)
class ObservedObjectLink:
    object_id: str
    status: ObjectMemoryLinkStatus
    match_score: float
    persisted: bool
    observation_id: str | None = None


@dataclass(frozen=True)
class ObjectMemoryReindexResult:
    reindexed_count: int
