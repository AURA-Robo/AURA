"""Object-memory ingest helpers for frame metadata."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from systems.memory import (
    DEFAULT_OBJECT_MEMORY_USER_ID,
    ObjectMemoryRuntimeHandle,
    ObjectObservationInput,
    create_object_memory_runtime,
)

from .models import FrameCache


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _timestamp_from_ns(timestamp_ns: int) -> datetime:
    return datetime.fromtimestamp(max(int(timestamp_ns), 0) / 1_000_000_000, tz=timezone.utc)


def _bbox_norm(
    bbox_xyxy: list[object] | tuple[object, ...],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if len(bbox_xyxy) != 4:
        return None
    if width <= 0 or height <= 0:
        return None
    coords: list[float] = []
    for value in bbox_xyxy:
        numeric = _as_float(value)
        if numeric is None:
            return None
        coords.append(numeric)
    x1, y1, x2, y2 = coords
    return (
        min(max(x1 / width, 0.0), 1.0),
        min(max(y1 / height, 0.0), 1.0),
        min(max(x2 / width, 0.0), 1.0),
        min(max(y2 / height, 0.0), 1.0),
    )


def _frame_identity(frame: FrameCache) -> str:
    return f"{frame.frame_header.source}:{frame.frame_header.frame_id}:{frame.frame_header.timestamp_ns}"


def _detection_identity(frame: FrameCache, detection: dict[str, Any], index: int) -> str:
    bbox = detection.get("bbox_xyxy")
    bbox_key = ""
    if isinstance(bbox, list) and len(bbox) == 4:
        bbox_key = ",".join(str(int(_as_float(value) or 0.0)) for value in bbox)
    class_name = str(detection.get("class_name") or "").strip()
    track_id = str(detection.get("track_id") or "").strip()
    return f"{_frame_identity(frame)}:{index}:{class_name}:{track_id}:{bbox_key}"


def _json_number(value: Any) -> int | float | None:
    numeric = _as_float(value)
    if numeric is None:
        return None
    rounded = int(round(numeric))
    if abs(numeric - rounded) <= 1e-6:
        return rounded
    return numeric


def _bbox_payload(value: Any) -> list[int | float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    coords = [_json_number(item) for item in value[:4]]
    if any(item is None for item in coords):
        return None
    return [item for item in coords if item is not None]


def _world_pose_payload(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    coords = [_as_float(item) for item in value[:3]]
    if any(item is None for item in coords):
        return None
    return [float(item) for item in coords if item is not None]


def _object_event_payload(
    frame: FrameCache,
    detection: dict[str, Any],
    *,
    object_id: str,
    status: str,
) -> dict[str, Any]:
    detection_payload: dict[str, Any] = {}
    class_id = _json_number(detection.get("class_id"))
    if class_id is not None:
        detection_payload["class_id"] = int(class_id)
    class_name = str(detection.get("class_name") or "").strip()
    if class_name:
        detection_payload["class_name"] = class_name
    confidence = _as_float(detection.get("confidence"))
    if confidence is not None:
        detection_payload["confidence"] = confidence
    bbox = _bbox_payload(detection.get("bbox_xyxy"))
    if bbox is not None:
        detection_payload["bbox_xyxy"] = bbox
    track_id = str(detection.get("track_id") or "").strip()
    if track_id:
        detection_payload["tracker_id"] = track_id
    world_pose = _world_pose_payload(detection.get("world_pose_xyz"))
    if world_pose is not None:
        detection_payload["world_pose_xyz"] = world_pose

    object_payload: dict[str, Any] = {
        "object_id": str(object_id or ""),
        "status": str(status or detection.get("memory_status") or ""),
    }
    persisted = detection.get("memory_persisted")
    if isinstance(persisted, bool):
        object_payload["persisted"] = persisted
    observation_id = detection.get("memory_observation_id")
    if isinstance(observation_id, str) and observation_id:
        object_payload["observation_id"] = observation_id
    match_score = _as_float(detection.get("memory_match_score"))
    if match_score is not None:
        object_payload["last_match_score"] = match_score
    filter_reason = str(detection.get("memory_filter_reason") or "").strip()
    if filter_reason:
        object_payload["filter_reason"] = filter_reason
    object_payload["last_match_reason"] = filter_reason or object_payload["status"]

    return {
        "frame_index": int(frame.seq),
        "frame_id": int(frame.frame_header.frame_id),
        "timestamp_ms": int(max(int(frame.frame_header.timestamp_ns), 0) // 1_000_000),
        "timestamp_ns": int(frame.frame_header.timestamp_ns),
        "source": str(frame.frame_header.source),
        "detection": detection_payload,
        "object": object_payload,
    }


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = max(left_area + right_area - inter, 1e-6)
    return inter / union


def _bbox_center_delta(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    left_center = ((left[0] + left[2]) * 0.5, (left[1] + left[3]) * 0.5)
    right_center = ((right[0] + right[2]) * 0.5, (right[1] + right[3]) * 0.5)
    return math.dist(left_center, right_center)


def _normalize_class_name(value: object) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _normalize_class_filter(values: tuple[str, ...]) -> frozenset[str]:
    if any(str(value).strip().lower() in {"*", "all", "any"} for value in values):
        return frozenset()
    return frozenset(
        normalized
        for normalized in (_normalize_class_name(value) for value in values)
        if normalized
    )


DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES = (
    "backpack",
    "barrel",
    "bin",
    "bottle",
    "box",
    "cart",
    "chair",
    "cone",
    "crate",
    "cup",
    "desk",
    "dolly",
    "door",
    "drawer",
    "fire extinguisher",
    "forklift",
    "keyboard",
    "laptop",
    "monitor",
    "mouse",
    "pallet",
    "person",
    "plant",
    "purple box",
    "rack",
    "shelf",
    "suitcase",
    "table",
    "toolbox",
    "tote",
    "trash can",
)

DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES = (
    "bus station",
    "ceiling",
    "conference center",
    "court",
    "facility",
    "garage door",
    "glass floor",
    "gym",
    "illuminate",
    "leak",
    "mezzanine",
    "overpass",
    "state school",
    "wall",
)


@dataclass(frozen=True)
class ObjectMemorySinkConfig:
    dsn: str = ""
    object_event_log_path: str = ""
    user_id: str = DEFAULT_OBJECT_MEMORY_USER_ID
    session_id: str = ""
    scene_scope: str = ""
    auto_migrate: bool = False
    max_seen_keys: int = 4096
    persistence_policy: str = "sparse"
    persist_min_interval_sec: int = 30
    persist_position_delta_m: float = 0.25
    persist_confidence_delta: float = 0.15
    min_detector_confidence: float = 0.80
    min_bbox_area_norm: float = 0.0005
    max_bbox_area_norm: float = 0.35
    require_world_pose: bool = True
    allowed_class_names: tuple[str, ...] = DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES
    blocked_class_names: tuple[str, ...] = DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES
    candidate_confirmation_count: int = 2
    candidate_ttl_sec: int = 10
    candidate_pose_radius_m: float = 0.45
    candidate_bbox_iou: float = 0.60
    candidate_center_delta_norm: float = 0.10


@dataclass(frozen=True)
class _PendingObjectCandidate:
    candidate_id: str
    class_name: str
    source_id: str | None
    scene_scope: str | None
    count: int
    first_observed_at: datetime
    last_observed_at: datetime
    last_frame_idx: int
    last_bbox_xyxy_norm: tuple[float, float, float, float]
    last_world_pose_xyz: tuple[float, float, float] | None
    last_observation: ObjectObservationInput


class _ObjectEventJsonlWriter:
    def __init__(self, path: str) -> None:
        self.path = str(path or "").strip()
        self.write_count = 0
        self.last_error: str | None = None

    @property
    def configured(self) -> bool:
        return bool(self.path)

    def write(self, payload: dict[str, Any]) -> None:
        if not self.configured:
            return
        try:
            output_path = Path(self.path).expanduser()
            parent = output_path.parent
            if str(parent):
                parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.write_count += 1
            self.last_error = None
        except Exception as exc:  # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"


class ObjectMemoryFrameSink:
    def __init__(
        self,
        config: ObjectMemorySinkConfig,
        *,
        runtime_handle: ObjectMemoryRuntimeHandle | None = None,
    ) -> None:
        self.config = config
        self.runtime = runtime_handle or create_object_memory_runtime(
            enabled=bool(str(config.dsn).strip()),
            dsn=str(config.dsn),
            user_id=str(config.user_id),
            auto_migrate=bool(config.auto_migrate),
        )
        self._enabled = False
        self._session_id = str(config.session_id).strip() or f"backend-webrtc-{time.time_ns()}"
        self._scene_scope = " ".join(str(config.scene_scope or "").strip().split()) or None
        self._seen_keys: set[str] = set()
        self._seen_order: deque[str] = deque()
        self._max_seen_keys = max(int(config.max_seen_keys), 1)
        self._pending_candidates: dict[str, _PendingObjectCandidate] = {}
        self._pending_seq = 0
        self._event_log = _ObjectEventJsonlWriter(str(config.object_event_log_path))
        self._allowed_class_names = _normalize_class_filter(tuple(config.allowed_class_names))
        self._blocked_class_names = _normalize_class_filter(tuple(config.blocked_class_names))
        self._observation_count = 0
        self._created_count = 0
        self._updated_count = 0
        self._skipped_duplicate_count = 0
        self._linked_count = 0
        self._suppressed_observation_count = 0
        self._pending_suppressed_count = 0
        self._cached_detector_skip_count = 0
        self._filtered_detection_count = 0
        self._filtered_detection_reasons: Counter[str] = Counter()
        self._last_error: str | None = None
        self._last_ingest_latency_ms: float | None = None
        self._last_observed_at: str | None = None
        self._last_success: bool | None = None
        self._object_count = 0

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def set_scene_scope(self, scene_scope: str | None) -> None:
        self._scene_scope = " ".join(str(scene_scope or "").strip().split()) or None

    def observe_frame(self, frame: FrameCache) -> None:
        if not self._enabled:
            return
        service = self.runtime.service
        if service is None:
            self._last_success = False
            self._last_error = self.runtime.degraded_reason
            return

        overlay = frame.viewer_overlay if isinstance(frame.viewer_overlay, dict) else {}
        detections = overlay.get("detections")
        if not isinstance(detections, list) or not detections:
            return

        start = time.perf_counter()
        if bool(overlay.get("detector_cached")):
            self._cached_detector_skip_count += sum(1 for item in detections if isinstance(item, dict))
            self._last_error = None
            self._last_success = True
            self._last_ingest_latency_ms = (time.perf_counter() - start) * 1000.0
            self._object_count = self.runtime.count_objects()
            return

        observations: list[ObjectObservationInput] = []
        seen_keys: list[str] = []
        linked_detections: list[dict[str, Any]] = []
        confirmed_candidate_ids: list[str] = []
        last_observed_at: datetime | None = None
        try:
            for index, item in enumerate(detections):
                if not isinstance(item, dict):
                    continue
                dedupe_key = _detection_identity(frame, item, index)
                if dedupe_key in self._seen_keys:
                    self._skipped_duplicate_count += 1
                    continue

                observation = self._observation_from_detection(frame, item, index, dedupe_key)
                if observation is None:
                    continue

                seen_keys.append(dedupe_key)
                last_observed_at = observation.observed_at
                if service.find_duplicate(self.runtime.user_id, observation) is None:
                    confirmed, candidate_id = self._record_pending_candidate(observation)
                    if not confirmed:
                        item["memory_status"] = "pending"
                        item["memory_match_score"] = 0.0
                        item["memory_persisted"] = False
                        self._write_detection_event(
                            frame,
                            item,
                            object_id=str(candidate_id or ""),
                            status="pending",
                        )
                        continue
                    if candidate_id is not None:
                        confirmed_candidate_ids.append(candidate_id)

                observations.append(observation)
                linked_detections.append(item)

            if not observations:
                for key in seen_keys:
                    self._remember_seen_key(key)
                self._last_error = None
                self._last_success = True
                self._last_ingest_latency_ms = (time.perf_counter() - start) * 1000.0
                self._last_observed_at = last_observed_at.isoformat() if last_observed_at is not None else None
                self._object_count = self.runtime.count_objects()
                return

            result = service.observe_objects(
                self.runtime.user_id,
                self._session_id,
                observations,
                persistence_policy=str(self.config.persistence_policy or "sparse"),
                persist_min_interval_sec=int(self.config.persist_min_interval_sec),
                persist_position_delta_m=float(self.config.persist_position_delta_m),
                persist_confidence_delta=float(self.config.persist_confidence_delta),
            )
        except Exception as exc:  # noqa: BLE001
            self.runtime.degraded_reason = f"{type(exc).__name__}: {exc}"
            self._last_success = False
            self._last_error = f"{type(exc).__name__}: {exc}"
            self._last_ingest_latency_ms = (time.perf_counter() - start) * 1000.0
            return

        self._observation_count += len(result.observation_ids)
        self._created_count += len(result.created_object_ids)
        self._updated_count += len(result.updated_object_ids)
        self._linked_count += sum(1 for link in result.links if link.status == "linked")
        self._suppressed_observation_count += int(result.suppressed_observation_count)
        for candidate_id in confirmed_candidate_ids:
            self._pending_candidates.pop(candidate_id, None)
        for detection, link in zip(linked_detections, result.links):
            detection["memory_object_id"] = str(link.object_id)
            detection["memory_status"] = str(link.status)
            detection["memory_match_score"] = round(float(link.match_score), 4)
            detection["memory_persisted"] = bool(link.persisted)
            if link.observation_id is not None:
                detection["memory_observation_id"] = str(link.observation_id)
            self._write_detection_event(
                frame,
                detection,
                object_id=str(link.object_id),
                status=str(link.status),
            )
        for key in seen_keys:
            self._remember_seen_key(key)
        self._last_error = None
        self._last_success = True
        self._last_ingest_latency_ms = (time.perf_counter() - start) * 1000.0
        self._last_observed_at = observations[-1].observed_at.isoformat()
        self._object_count = self.runtime.count_objects()

    def _observation_from_detection(
        self,
        frame: FrameCache,
        item: dict[str, Any],
        index: int,
        dedupe_key: str,
    ) -> ObjectObservationInput | None:
        del index
        class_name = str(item.get("class_name") or "").strip()
        bbox_xyxy = item.get("bbox_xyxy")
        if not class_name or not isinstance(bbox_xyxy, list):
            return None
        bbox_xyxy_norm = _bbox_norm(
            bbox_xyxy,
            width=int(frame.frame_header.width),
            height=int(frame.frame_header.height),
        )
        if bbox_xyxy_norm is None:
            return None

        confidence = _as_float(item.get("confidence")) or 0.0
        track_id = str(item.get("track_id") or "").strip()
        source_id = str(item.get("source") or frame.frame_header.source)
        image_hash = hashlib.sha1(dedupe_key.encode("utf-8")).hexdigest()
        observed_at = _timestamp_from_ns(int(frame.frame_header.timestamp_ns))
        attributes: dict[str, Any] = {
            "frame_id": int(frame.frame_header.frame_id),
            "frame_seq": int(frame.seq),
            "timestamp_ns": int(frame.frame_header.timestamp_ns),
            "detection_identity": dedupe_key,
        }
        for key in ("depth_m", "approach_yaw_rad"):
            numeric = _as_float(item.get(key))
            if numeric is not None:
                attributes[key] = numeric
        world_pose_xyz = item.get("world_pose_xyz")
        normalized_world_pose_xyz = None
        if isinstance(world_pose_xyz, list) and len(world_pose_xyz) >= 3:
            coords = [_as_float(world_pose_xyz[0]), _as_float(world_pose_xyz[1]), _as_float(world_pose_xyz[2])]
            if all(value is not None for value in coords):
                normalized_world_pose_xyz = tuple(float(value) for value in coords if value is not None)
                attributes["world_pose_xyz"] = list(normalized_world_pose_xyz)

        filter_reason = self._detection_filter_reason(
            class_name=class_name,
            confidence=float(confidence),
            bbox_xyxy_norm=bbox_xyxy_norm,
            world_pose_xyz=normalized_world_pose_xyz,
        )
        if filter_reason is not None:
            self._record_filtered_detection(frame, item, filter_reason)
            return None

        return ObjectObservationInput(
            frame_idx=int(frame.seq),
            track_id=track_id,
            class_name=class_name,
            detector_conf=float(confidence),
            bbox_xyxy_norm=bbox_xyxy_norm,
            box_area=0.0,
            aspect_ratio=0.0,
            image_hash=image_hash,
            observed_at=observed_at,
            room_id=None,
            scene_scope=self._scene_scope,
            world_pose_xyz=normalized_world_pose_xyz,
            world_pose_observed_at=observed_at if normalized_world_pose_xyz is not None else None,
            source_id=source_id,
            attributes=attributes,
        )

    def _detection_filter_reason(
        self,
        *,
        class_name: str,
        confidence: float,
        bbox_xyxy_norm: tuple[float, float, float, float],
        world_pose_xyz: tuple[float, float, float] | None,
    ) -> str | None:
        normalized_class = _normalize_class_name(class_name)
        if self._allowed_class_names and normalized_class not in self._allowed_class_names:
            return "class_not_allowed"
        if normalized_class in self._blocked_class_names:
            return "class_blocked"
        if confidence < max(float(self.config.min_detector_confidence), 0.0):
            return "low_confidence"

        bbox_width = max(0.0, bbox_xyxy_norm[2] - bbox_xyxy_norm[0])
        bbox_height = max(0.0, bbox_xyxy_norm[3] - bbox_xyxy_norm[1])
        bbox_area = bbox_width * bbox_height
        if bbox_area < max(float(self.config.min_bbox_area_norm), 0.0):
            return "bbox_too_small"
        max_bbox_area = max(float(self.config.max_bbox_area_norm), 0.0)
        if max_bbox_area > 0.0 and bbox_area > max_bbox_area:
            return "bbox_too_large"
        if bool(self.config.require_world_pose) and world_pose_xyz is None:
            return "missing_world_pose"
        return None

    def _record_filtered_detection(self, frame: FrameCache, detection: dict[str, Any], reason: str) -> None:
        self._filtered_detection_count += 1
        self._filtered_detection_reasons[str(reason)] += 1
        detection["memory_status"] = "filtered"
        detection["memory_filter_reason"] = str(reason)
        detection["memory_match_score"] = 0.0
        detection["memory_persisted"] = False
        self._write_detection_event(frame, detection, object_id="", status="filtered")

    def _write_detection_event(
        self,
        frame: FrameCache,
        detection: dict[str, Any],
        *,
        object_id: str,
        status: str,
    ) -> None:
        self._event_log.write(_object_event_payload(frame, detection, object_id=object_id, status=status))

    def _record_pending_candidate(self, observation: ObjectObservationInput) -> tuple[bool, str | None]:
        confirmation_count = max(int(self.config.candidate_confirmation_count), 1)
        if confirmation_count <= 1:
            return True, None

        self._purge_expired_pending_candidates(observation.observed_at)
        candidate = self._find_pending_candidate(observation)
        if candidate is None:
            self._pending_seq += 1
            candidate_id = f"pending-{self._pending_seq}"
            self._pending_candidates[candidate_id] = _PendingObjectCandidate(
                candidate_id=candidate_id,
                class_name=observation.class_name,
                source_id=observation.source_id,
                scene_scope=observation.scene_scope,
                count=1,
                first_observed_at=observation.observed_at,
                last_observed_at=observation.observed_at,
                last_frame_idx=observation.frame_idx,
                last_bbox_xyxy_norm=observation.bbox_xyxy_norm,
                last_world_pose_xyz=observation.world_pose_xyz,
                last_observation=observation,
            )
            self._pending_suppressed_count += 1
            return False, candidate_id

        updated = replace(
            candidate,
            count=candidate.count + 1,
            last_observed_at=observation.observed_at,
            last_frame_idx=observation.frame_idx,
            last_bbox_xyxy_norm=observation.bbox_xyxy_norm,
            last_world_pose_xyz=observation.world_pose_xyz,
            last_observation=observation,
        )
        self._pending_candidates[candidate.candidate_id] = updated
        if updated.count >= confirmation_count:
            return True, candidate.candidate_id
        self._pending_suppressed_count += 1
        return False, candidate.candidate_id

    def _find_pending_candidate(self, observation: ObjectObservationInput) -> _PendingObjectCandidate | None:
        self._purge_expired_pending_candidates(observation.observed_at)
        for candidate in self._pending_candidates.values():
            if candidate.last_frame_idx == observation.frame_idx:
                continue
            if candidate.class_name != observation.class_name:
                continue
            if candidate.source_id != observation.source_id:
                continue
            if candidate.scene_scope != observation.scene_scope:
                continue
            if self._pending_pose_matches(candidate, observation):
                return candidate
            if _bbox_iou(candidate.last_bbox_xyxy_norm, observation.bbox_xyxy_norm) >= float(
                self.config.candidate_bbox_iou
            ):
                return candidate
            if _bbox_center_delta(candidate.last_bbox_xyxy_norm, observation.bbox_xyxy_norm) <= float(
                self.config.candidate_center_delta_norm
            ):
                return candidate
        return None

    def _pending_pose_matches(
        self,
        candidate: _PendingObjectCandidate,
        observation: ObjectObservationInput,
    ) -> bool:
        if candidate.last_world_pose_xyz is None or observation.world_pose_xyz is None:
            return False
        distance_m = math.dist(candidate.last_world_pose_xyz, observation.world_pose_xyz)
        return distance_m <= max(float(self.config.candidate_pose_radius_m), 0.0)

    def _purge_expired_pending_candidates(self, observed_at: datetime) -> None:
        ttl_sec = max(int(self.config.candidate_ttl_sec), 0)
        expired_ids = [
            candidate_id
            for candidate_id, candidate in self._pending_candidates.items()
            if max(0.0, (observed_at - candidate.last_observed_at).total_seconds()) > ttl_sec
        ]
        for candidate_id in expired_ids:
            self._pending_candidates.pop(candidate_id, None)

    def health_snapshot(self) -> dict[str, object]:
        return {
            "configured": bool(str(self.config.dsn).strip()),
            "autoMigrate": bool(self.config.auto_migrate),
            "enabled": bool(self._enabled),
            "available": self.runtime.available,
            "userId": self.runtime.user_id,
            "sceneScope": self._scene_scope,
            "objectCount": int(self._object_count),
            "observationCount": int(self._observation_count),
            "createdCount": int(self._created_count),
            "updatedCount": int(self._updated_count),
            "linkedCount": int(self._linked_count),
            "skippedDuplicateCount": int(self._skipped_duplicate_count),
            "suppressedObservationCount": int(self._suppressed_observation_count),
            "pendingCandidateCount": len(self._pending_candidates),
            "pendingSuppressedCount": int(self._pending_suppressed_count),
            "cachedDetectorSkipCount": int(self._cached_detector_skip_count),
            "filteredDetectionCount": int(self._filtered_detection_count),
            "filteredDetectionReasons": dict(self._filtered_detection_reasons),
            "eventLogConfigured": self._event_log.configured,
            "eventLogPath": self._event_log.path,
            "eventLogWriteCount": int(self._event_log.write_count),
            "eventLogLastError": self._event_log.last_error,
            "lastSuccess": self._last_success,
            "lastError": self._last_error,
            "lastIngestLatencyMs": self._last_ingest_latency_ms,
            "lastObservedAt": self._last_observed_at,
            "degradedReason": self.runtime.degraded_reason,
        }

    def _remember_seen_key(self, key: str) -> None:
        self._seen_keys.add(key)
        self._seen_order.append(key)
        while len(self._seen_order) > self._max_seen_keys:
            expired = self._seen_order.popleft()
            self._seen_keys.discard(expired)
