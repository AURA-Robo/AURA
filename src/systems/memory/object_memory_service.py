from __future__ import annotations

import math
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from datetime import timedelta
from typing import Any

from .object_memory_adapter import object_entries_to_recent_seen
from .object_memory_models import (
    DuplicateMatch,
    MemoryNavigationResolution,
    ObjectMemoryContext,
    ObjectMemoryEmbedding,
    ObjectMemoryEntry,
    ObjectMemoryPersistencePolicy,
    ObjectMemoryNavigationCandidate,
    ObjectMemoryReindexResult,
    ObjectObservation,
    ObjectObservationInput,
    ObservedObjectLink,
    ObserveResult,
    RetrievedObjectMemory,
    utc_now,
)
from .object_memory_repository import ObjectMemoryRepository


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0.0:
        return values
    return [value / norm for value in values]


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _normalize_scene_scope(value: str | None) -> str | None:
    normalized = " ".join(str(value or "").strip().split())
    return normalized or None


def _normalize_world_pose_xyz(value: object) -> tuple[float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
    return None


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


class ObjectMemoryService:
    def __init__(
        self,
        repository: ObjectMemoryRepository,
        *,
        appearance_model: str = "CLIP-ViT-B/32",
        recent_window_sec: int = 86400,
        max_candidates: int = 20,
        ema_alpha: float = 0.2,
    ) -> None:
        self.repository = repository
        self.appearance_model = appearance_model
        self.recent_window = timedelta(seconds=recent_window_sec)
        self.max_candidates = max_candidates
        self.ema_alpha = ema_alpha

    def observe_objects(
        self,
        user_id: str,
        session_id: str,
        observations: Sequence[ObjectObservationInput],
        *,
        room_id: str | None = None,
        source_id: str | None = None,
        persistence_policy: ObjectMemoryPersistencePolicy = "audit_all",
        persist_min_interval_sec: int = 30,
        persist_position_delta_m: float = 0.25,
        persist_confidence_delta: float = 0.15,
    ) -> ObserveResult:
        created_object_ids: list[str] = []
        updated_object_ids: list[str] = []
        observation_ids: list[str] = []
        links: list[ObservedObjectLink] = []
        suppressed_observation_count = 0

        for raw_observation in observations:
            observation = self._normalize_input(raw_observation, room_id=room_id, source_id=source_id)
            match_row = self._find_duplicate_row(user_id, observation)
            if match_row is None:
                entry, embedding = self._create_entry(user_id, session_id, observation)
                self.repository.insert_object_entry(entry)
                self.repository.upsert_object_embedding(embedding)
                created_object_ids.append(entry.object_id)
                object_id = entry.object_id
                dedupe_confidence = 0.0
                link_status = "created"
                should_persist = True
                should_update_entry = True
            else:
                existing_entry, match = match_row
                should_update_entry, persistence_reasons = self._should_persist_linked_observation(
                    existing_entry,
                    observation,
                    match=match,
                    persistence_policy=persistence_policy,
                    persist_min_interval_sec=persist_min_interval_sec,
                    persist_position_delta_m=persist_position_delta_m,
                    persist_confidence_delta=persist_confidence_delta,
                )
                object_id = existing_entry.object_id
                dedupe_confidence = match.score
                link_status = "linked"
                should_persist = should_update_entry
                if should_update_entry:
                    entry, embedding = self._merge_entry(
                        existing_entry,
                        session_id,
                        observation,
                        match,
                        persistence_reasons=persistence_reasons,
                    )
                    self.repository.update_object_entry(entry)
                    self.repository.upsert_object_embedding(embedding)
                    updated_object_ids.append(entry.object_id)
                else:
                    suppressed_observation_count += 1

            persisted_observation_id: str | None = None
            if should_persist:
                persisted_observation = ObjectObservation(
                    observation_id=str(uuid.uuid4()),
                    object_id=object_id,
                    user_id=user_id,
                    session_id=session_id,
                    source_id=observation.source_id,
                    frame_idx=observation.frame_idx,
                    observed_at=observation.observed_at,
                    track_id=observation.track_id,
                    class_name=observation.class_name,
                    detector_conf=observation.detector_conf,
                    room_id=observation.room_id,
                    scene_scope=observation.scene_scope,
                    bbox_xyxy_norm=observation.bbox_xyxy_norm,
                    box_area=observation.box_area,
                    aspect_ratio=observation.aspect_ratio,
                    appearance_embedding=list(observation.appearance_embedding),
                    appearance_model=self.appearance_model,
                    image_hash=observation.image_hash,
                    world_pose_xyz=observation.world_pose_xyz,
                    world_pose_observed_at=observation.world_pose_observed_at,
                    mask_area=observation.mask_area,
                    attributes={
                        **dict(observation.attributes),
                        "dedupe_confidence": dedupe_confidence,
                    },
                )
                self.repository.insert_object_observation(persisted_observation)
                persisted_observation_id = persisted_observation.observation_id
                observation_ids.append(persisted_observation.observation_id)

            links.append(
                ObservedObjectLink(
                    object_id=object_id,
                    status=link_status,
                    match_score=dedupe_confidence,
                    persisted=should_persist,
                    observation_id=persisted_observation_id,
                )
            )

        return ObserveResult(
            created_object_ids=created_object_ids,
            updated_object_ids=updated_object_ids,
            observation_ids=observation_ids,
            links=links,
            suppressed_observation_count=suppressed_observation_count,
        )

    def query_recent_objects(
        self,
        user_id: str,
        *,
        class_name: str | None = None,
        room_id: str | None = None,
        scene_scope: str | None = None,
        top_k: int = 10,
        max_age_sec: int = 86400,
    ) -> ObjectMemoryContext:
        updated_since = utc_now() - timedelta(seconds=max_age_sec)
        rows = self.repository.list_object_entries(
            user_id,
            statuses=("active",),
            class_name=class_name,
            room_id=room_id,
            scene_scope=_normalize_scene_scope(scene_scope),
            updated_since=updated_since,
            top_k=top_k,
        )
        entries = [self._to_retrieved(row) for row in rows]
        return ObjectMemoryContext(
            user_id=user_id,
            entries=entries,
            recent_seen=object_entries_to_recent_seen(entries),
            debug={
                "class_name": class_name,
                "room_id": room_id,
                "scene_scope": _normalize_scene_scope(scene_scope),
                "max_age_sec": max_age_sec,
                "retrieved_count": len(entries),
            },
        )

    def resolve_memory_navigation_target(
        self,
        user_id: str,
        *,
        scene_scope: str | None,
        class_name: str,
        room_hint: str | None = None,
        instance_hint: str | None = None,
        max_pose_age_sec: int = 600,
    ) -> MemoryNavigationResolution:
        normalized_scene_scope = _normalize_scene_scope(scene_scope)
        normalized_class_name = str(class_name or "").strip()
        normalized_room_hint = str(room_hint or "").strip() or None
        normalized_instance_hint = str(instance_hint or "").strip() or None
        if not normalized_scene_scope or not normalized_class_name:
            return MemoryNavigationResolution(
                status="no_candidate",
                debug={
                    "reason": "missing_scene_scope_or_class_name",
                    "scene_scope": normalized_scene_scope,
                    "class_name": normalized_class_name,
                },
            )

        rows = self.repository.list_object_entries(
            user_id,
            statuses=("active",),
            class_name=normalized_class_name,
            scene_scope=normalized_scene_scope,
        )
        pose_candidates = [
            row
            for row in rows
            if row.world_pose_xyz is not None
            and row.world_pose_observed_at is not None
        ]
        if normalized_room_hint is not None:
            pose_candidates = [row for row in pose_candidates if row.room_id == normalized_room_hint]
        if not pose_candidates:
            return MemoryNavigationResolution(
                status="no_candidate",
                debug={
                    "scene_scope": normalized_scene_scope,
                    "class_name": normalized_class_name,
                    "room_hint": normalized_room_hint,
                    "instance_hint": normalized_instance_hint,
                    "reason": "no_pose_candidates",
                },
            )

        candidates = [self._to_navigation_candidate(row) for row in pose_candidates]
        fresh_candidates = [
            candidate
            for candidate in candidates
            if candidate.pose_age_sec <= max(0, int(max_pose_age_sec))
        ]
        fresh_candidates.sort(
            key=lambda candidate: (
                candidate.last_seen_at.timestamp(),
                candidate.dedupe_confidence,
                candidate.last_detector_conf,
            ),
            reverse=True,
        )
        if not fresh_candidates:
            return MemoryNavigationResolution(
                status="stale_only",
                candidates=candidates,
                debug={
                    "scene_scope": normalized_scene_scope,
                    "class_name": normalized_class_name,
                    "room_hint": normalized_room_hint,
                    "instance_hint": normalized_instance_hint,
                    "max_pose_age_sec": int(max_pose_age_sec),
                },
            )
        if len(fresh_candidates) > 1:
            return MemoryNavigationResolution(
                status="ambiguous",
                candidates=fresh_candidates,
                debug={
                    "scene_scope": normalized_scene_scope,
                    "class_name": normalized_class_name,
                    "room_hint": normalized_room_hint,
                    "instance_hint": normalized_instance_hint,
                    "max_pose_age_sec": int(max_pose_age_sec),
                },
            )
        return MemoryNavigationResolution(
            status="resolved",
            selected=fresh_candidates[0],
            candidates=fresh_candidates,
            debug={
                "scene_scope": normalized_scene_scope,
                "class_name": normalized_class_name,
                "room_hint": normalized_room_hint,
                "instance_hint": normalized_instance_hint,
                "max_pose_age_sec": int(max_pose_age_sec),
            },
        )

    def find_duplicate(self, user_id: str, observation: ObjectObservationInput) -> DuplicateMatch | None:
        row = self._find_duplicate_row(user_id, self._normalize_input(observation))
        if row is None:
            return None
        return row[1]

    def project_recent_seen(self, user_id: str, *, top_k_per_room: int = 1) -> list[dict[str, Any]]:
        rows = self.repository.list_object_entries(user_id, statuses=("active",))
        entries = [self._to_retrieved(row) for row in rows]
        return object_entries_to_recent_seen(entries, top_k_per_room=top_k_per_room)

    def reindex(
        self,
        *,
        user_id: str | None = None,
        object_ids: Sequence[str] | None = None,
    ) -> ObjectMemoryReindexResult:
        entries = self.repository.list_object_entries(
            user_id,
            statuses=("active",),
            object_ids=object_ids,
        )
        observations = self.repository.list_object_observations(user_id, object_ids=[row.object_id for row in entries])
        observations_by_object: dict[str, list[ObjectObservation]] = defaultdict(list)
        for observation in observations:
            observations_by_object[observation.object_id].append(observation)

        reindexed_count = 0
        for entry in entries:
            object_observations = observations_by_object.get(entry.object_id, [])
            if not object_observations:
                continue
            object_observations.sort(key=lambda row: row.observed_at)
            vectors = [row.appearance_embedding for row in object_observations if row.appearance_embedding]
            if not vectors:
                continue
            mean_vector = [
                sum(values[index] for values in vectors) / len(vectors)
                for index in range(len(vectors[0]))
            ]
            centroid = _normalize_vector(mean_vector)
            latest = object_observations[-1]
            updated_entry = replace(
                entry,
                observation_count=len(object_observations),
                appearance_count=len(vectors),
                scene_scope=latest.scene_scope if latest.scene_scope is not None else entry.scene_scope,
                last_bbox_xyxy_norm=latest.bbox_xyxy_norm,
                last_box_area=latest.box_area,
                last_aspect_ratio=latest.aspect_ratio,
                last_detector_conf=latest.detector_conf,
                last_source_id=latest.source_id,
                last_session_id=latest.session_id,
                last_seen_at=latest.observed_at,
                world_pose_xyz=latest.world_pose_xyz if latest.world_pose_xyz is not None else entry.world_pose_xyz,
                world_pose_observed_at=(
                    latest.world_pose_observed_at
                    if latest.world_pose_observed_at is not None
                    else entry.world_pose_observed_at
                ),
                updated_at=utc_now(),
                metadata={
                    **dict(entry.metadata),
                    "last_image_hash": latest.image_hash,
                    "last_track_id": latest.track_id,
                    "mask_area": latest.mask_area,
                    "attributes": latest.attributes,
                },
            )
            self.repository.update_object_entry(updated_entry)
            self.repository.upsert_object_embedding(
                ObjectMemoryEmbedding(
                    object_id=entry.object_id,
                    user_id=entry.user_id,
                    model_name=self.appearance_model,
                    embedding=centroid,
                    index_status="ready",
                )
            )
            reindexed_count += 1

        return ObjectMemoryReindexResult(reindexed_count=reindexed_count)

    def _normalize_input(
        self,
        observation: ObjectObservationInput,
        *,
        room_id: str | None = None,
        source_id: str | None = None,
    ) -> ObjectObservationInput:
        bbox = tuple(min(1.0, max(0.0, float(value))) for value in observation.bbox_xyxy_norm)
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        box_area = float(observation.box_area) if observation.box_area > 0 else width * height
        aspect_ratio = float(observation.aspect_ratio) if observation.aspect_ratio > 0 else width / max(height, 1e-6)
        embedding = _normalize_vector(list(observation.appearance_embedding))
        raw_world_pose = observation.world_pose_xyz
        if raw_world_pose is None:
            raw_world_pose = observation.attributes.get("world_pose_xyz")
        world_pose_xyz = _normalize_world_pose_xyz(raw_world_pose)
        return ObjectObservationInput(
            frame_idx=int(observation.frame_idx),
            track_id=str(observation.track_id),
            class_name=observation.class_name,
            detector_conf=float(observation.detector_conf),
            bbox_xyxy_norm=bbox,
            box_area=box_area,
            aspect_ratio=aspect_ratio,
            appearance_embedding=embedding,
            image_hash=observation.image_hash,
            observed_at=observation.observed_at,
            mask_area=observation.mask_area,
            room_id=observation.room_id if observation.room_id is not None else room_id,
            scene_scope=_normalize_scene_scope(observation.scene_scope),
            world_pose_xyz=world_pose_xyz,
            world_pose_observed_at=(
                observation.world_pose_observed_at
                if world_pose_xyz is not None and observation.world_pose_observed_at is not None
                else observation.observed_at if world_pose_xyz is not None else None
            ),
            source_id=observation.source_id if observation.source_id is not None else source_id,
            attributes=dict(observation.attributes),
        )

    def _create_entry(
        self,
        user_id: str,
        session_id: str,
        observation: ObjectObservationInput,
    ) -> tuple[ObjectMemoryEntry, ObjectMemoryEmbedding]:
        now = utc_now()
        object_id = str(uuid.uuid4())
        entry = ObjectMemoryEntry(
            object_id=object_id,
            user_id=user_id,
            canonical_class=observation.class_name,
            room_id=observation.room_id,
            scene_scope=observation.scene_scope,
            status="active",
            first_seen_at=observation.observed_at,
            last_seen_at=observation.observed_at,
            observation_count=1,
            last_source_id=observation.source_id,
            last_session_id=session_id,
            last_bbox_xyxy_norm=observation.bbox_xyxy_norm,
            last_box_area=observation.box_area,
            last_aspect_ratio=observation.aspect_ratio,
            last_detector_conf=observation.detector_conf,
            world_pose_xyz=observation.world_pose_xyz,
            world_pose_observed_at=observation.world_pose_observed_at,
            appearance_count=1 if observation.appearance_embedding else 0,
            dedupe_confidence=0.0,
            metadata={
                "last_image_hash": observation.image_hash,
                "last_track_id": observation.track_id,
                "mask_area": observation.mask_area,
                "attributes": observation.attributes,
            },
            created_at=now,
            updated_at=now,
        )
        embedding = ObjectMemoryEmbedding(
            object_id=object_id,
            user_id=user_id,
            model_name=self.appearance_model,
            embedding=list(observation.appearance_embedding) or None,
            index_status="ready" if observation.appearance_embedding else "pending",
        )
        return entry, embedding

    def _merge_entry(
        self,
        entry: ObjectMemoryEntry,
        session_id: str,
        observation: ObjectObservationInput,
        match: DuplicateMatch,
        *,
        persistence_reasons: Sequence[str] | None = None,
    ) -> tuple[ObjectMemoryEntry, ObjectMemoryEmbedding]:
        existing_embedding = self.repository.get_object_embedding(entry.object_id)
        merged_embedding = list(observation.appearance_embedding)
        if existing_embedding is not None and existing_embedding.embedding:
            if observation.appearance_embedding and len(existing_embedding.embedding) == len(observation.appearance_embedding):
                merged_embedding = _normalize_vector(
                    [
                        (1.0 - self.ema_alpha) * old + self.ema_alpha * new
                        for old, new in zip(existing_embedding.embedding, observation.appearance_embedding)
                    ]
                )
            else:
                merged_embedding = list(existing_embedding.embedding)
        now = utc_now()
        updated_entry = replace(
            entry,
            room_id=observation.room_id if observation.room_id is not None else entry.room_id,
            scene_scope=observation.scene_scope if observation.scene_scope is not None else entry.scene_scope,
            last_seen_at=observation.observed_at,
            observation_count=entry.observation_count + 1,
            last_source_id=observation.source_id,
            last_session_id=session_id,
            last_bbox_xyxy_norm=observation.bbox_xyxy_norm,
            last_box_area=observation.box_area,
            last_aspect_ratio=observation.aspect_ratio,
            last_detector_conf=observation.detector_conf,
            world_pose_xyz=observation.world_pose_xyz if observation.world_pose_xyz is not None else entry.world_pose_xyz,
            world_pose_observed_at=(
                observation.world_pose_observed_at
                if observation.world_pose_observed_at is not None
                else entry.world_pose_observed_at
            ),
            appearance_count=entry.appearance_count + (1 if observation.appearance_embedding else 0),
            dedupe_confidence=match.score,
            metadata={
                **dict(entry.metadata),
                "last_image_hash": observation.image_hash,
                "last_track_id": observation.track_id,
                "mask_area": observation.mask_area,
                "attributes": observation.attributes,
                "last_iou": _bbox_iou(entry.last_bbox_xyxy_norm, observation.bbox_xyxy_norm),
                "last_persistence_reasons": list(persistence_reasons or ()),
            },
            updated_at=now,
        )
        embedding = ObjectMemoryEmbedding(
            object_id=entry.object_id,
            user_id=entry.user_id,
            model_name=self.appearance_model,
            embedding=merged_embedding or None,
            index_status="ready" if merged_embedding else "pending",
            embedded_at=now,
        )
        return updated_entry, embedding

    def _find_duplicate_row(
        self,
        user_id: str,
        observation: ObjectObservationInput,
    ) -> tuple[ObjectMemoryEntry, DuplicateMatch] | None:
        candidates = self.repository.list_object_entries(
            user_id,
            statuses=("active",),
            class_name=observation.class_name,
            scene_scope=observation.scene_scope,
            updated_since=observation.observed_at - self.recent_window,
        )
        if observation.room_id is not None:
            candidates.sort(
                key=lambda row: (
                    1 if row.room_id == observation.room_id else 0,
                    row.last_seen_at.timestamp(),
                ),
                reverse=True,
            )
        candidates = candidates[: self.max_candidates]

        track_match = self._find_track_match(observation, candidates)
        if track_match is not None:
            return track_match

        pose_match = self._find_world_pose_match(observation, candidates)
        if pose_match is not None:
            return pose_match

        spatial_match = self._find_short_horizon_spatial_match(observation, candidates)
        if spatial_match is not None:
            return spatial_match

        if not observation.appearance_embedding:
            return None

        ranked: list[tuple[ObjectMemoryEntry, DuplicateMatch]] = []
        for entry in candidates:
            if observation.room_id is not None and entry.room_id is not None and observation.room_id != entry.room_id:
                continue
            if not self._passes_cheap_gate(observation, entry):
                continue
            embedding = self.repository.get_object_embedding(entry.object_id)
            appearance_score = _cosine_similarity(
                observation.appearance_embedding,
                embedding.embedding if embedding is not None and embedding.embedding is not None else [],
            )
            spatial_score = self._spatial_score(observation, entry)
            temporal_score = self._temporal_score(observation, entry)
            class_score = 1.0 if entry.canonical_class == observation.class_name else 0.0
            score = 0.60 * appearance_score + 0.25 * spatial_score + 0.10 * temporal_score + 0.05 * class_score
            ranked.append(
                (
                    entry,
                    DuplicateMatch(
                        object_id=entry.object_id,
                        score=score,
                        appearance_score=appearance_score,
                        spatial_score=spatial_score,
                        temporal_score=temporal_score,
                        class_score=class_score,
                    ),
                )
            )

        if not ranked:
            return None
        ranked.sort(key=lambda row: row[1].score, reverse=True)
        best_entry, best_match = ranked[0]
        second_score = ranked[1][1].score if len(ranked) > 1 else 0.0
        if best_match.score >= 0.85:
            return best_entry, best_match
        if best_match.score >= 0.75 and (best_match.score - second_score) >= 0.05:
            return best_entry, best_match
        return None

    def _find_world_pose_match(
        self,
        observation: ObjectObservationInput,
        candidates: Sequence[ObjectMemoryEntry],
    ) -> tuple[ObjectMemoryEntry, DuplicateMatch] | None:
        if observation.world_pose_xyz is None:
            return None
        ranked: list[tuple[ObjectMemoryEntry, DuplicateMatch]] = []
        for entry in candidates:
            pose_score = self._world_pose_score(observation, entry)
            if pose_score <= 0.0:
                continue
            if observation.room_id is not None and entry.room_id is not None and observation.room_id != entry.room_id:
                continue
            spatial_score = self._spatial_score(observation, entry)
            temporal_score = self._temporal_score(observation, entry)
            class_score = 1.0 if entry.canonical_class == observation.class_name else 0.0
            score = 0.55 * pose_score + 0.20 * spatial_score + 0.20 * temporal_score + 0.05 * class_score
            ranked.append(
                (
                    entry,
                    DuplicateMatch(
                        object_id=entry.object_id,
                        score=score,
                        appearance_score=0.0,
                        spatial_score=spatial_score,
                        temporal_score=temporal_score,
                        class_score=class_score,
                    ),
                )
            )
        return self._select_ranked_match(ranked, min_score=0.72, min_margin=0.05)

    def _find_short_horizon_spatial_match(
        self,
        observation: ObjectObservationInput,
        candidates: Sequence[ObjectMemoryEntry],
    ) -> tuple[ObjectMemoryEntry, DuplicateMatch] | None:
        if not observation.source_id:
            return None
        ranked: list[tuple[ObjectMemoryEntry, DuplicateMatch]] = []
        for entry in candidates:
            if entry.last_source_id != observation.source_id:
                continue
            if observation.room_id is not None and entry.room_id is not None and observation.room_id != entry.room_id:
                continue
            age_sec = max(0.0, (observation.observed_at - entry.last_seen_at).total_seconds())
            iou = _bbox_iou(entry.last_bbox_xyxy_norm, observation.bbox_xyxy_norm)
            if age_sec > 2.5 or iou < 0.70:
                continue
            spatial_score = self._spatial_score(observation, entry)
            temporal_score = self._temporal_score(observation, entry)
            score = 0.60 * iou + 0.25 * temporal_score + 0.15 * spatial_score
            ranked.append(
                (
                    entry,
                    DuplicateMatch(
                        object_id=entry.object_id,
                        score=score,
                        appearance_score=0.0,
                        spatial_score=spatial_score,
                        temporal_score=temporal_score,
                        class_score=1.0,
                    ),
                )
            )
        return self._select_ranked_match(ranked, min_score=0.82, min_margin=0.03)

    def _find_track_match(
        self,
        observation: ObjectObservationInput,
        candidates: Sequence[ObjectMemoryEntry],
    ) -> tuple[ObjectMemoryEntry, DuplicateMatch] | None:
        if not observation.track_id or not observation.source_id:
            return None
        for entry in candidates:
            if entry.last_source_id != observation.source_id:
                continue
            if observation.scene_scope is not None and entry.scene_scope is not None and observation.scene_scope != entry.scene_scope:
                continue
            if observation.room_id is not None and entry.room_id is not None and observation.room_id != entry.room_id:
                continue
            last_track_id = str(entry.metadata.get("last_track_id", "") or "")
            if last_track_id != observation.track_id:
                continue
            spatial_score = self._spatial_score(observation, entry)
            temporal_score = self._temporal_score(observation, entry)
            return (
                entry,
                DuplicateMatch(
                    object_id=entry.object_id,
                    score=1.0,
                    appearance_score=1.0 if observation.appearance_embedding else 0.0,
                    spatial_score=spatial_score,
                    temporal_score=temporal_score,
                    class_score=1.0,
                ),
            )
        return None

    def _select_ranked_match(
        self,
        ranked: Sequence[tuple[ObjectMemoryEntry, DuplicateMatch]],
        *,
        min_score: float,
        min_margin: float,
    ) -> tuple[ObjectMemoryEntry, DuplicateMatch] | None:
        if not ranked:
            return None
        ordered = sorted(ranked, key=lambda row: row[1].score, reverse=True)
        best_entry, best_match = ordered[0]
        second_score = ordered[1][1].score if len(ordered) > 1 else 0.0
        if best_match.score >= min_score and (best_match.score - second_score) >= min_margin:
            return best_entry, best_match
        return None

    def _world_pose_score(self, observation: ObjectObservationInput, entry: ObjectMemoryEntry) -> float:
        if observation.world_pose_xyz is None or entry.world_pose_xyz is None:
            return 0.0
        distance_m = math.dist(observation.world_pose_xyz, entry.world_pose_xyz)
        if distance_m > 0.45:
            return 0.0
        return max(0.0, 1.0 - (distance_m / 0.45))

    def _should_persist_linked_observation(
        self,
        entry: ObjectMemoryEntry,
        observation: ObjectObservationInput,
        *,
        match: DuplicateMatch,
        persistence_policy: ObjectMemoryPersistencePolicy,
        persist_min_interval_sec: int,
        persist_position_delta_m: float,
        persist_confidence_delta: float,
    ) -> tuple[bool, list[str]]:
        if persistence_policy == "audit_all":
            return True, ["audit_all"]

        reasons: list[str] = []
        age_sec = max(0.0, (observation.observed_at - entry.last_seen_at).total_seconds())
        if age_sec >= max(int(persist_min_interval_sec), 1):
            reasons.append("heartbeat")

        last_track_id = str(entry.metadata.get("last_track_id") or "")
        if observation.track_id and observation.track_id != last_track_id:
            reasons.append("track_changed")
        if observation.source_id is not None and observation.source_id != entry.last_source_id:
            reasons.append("source_changed")
        if observation.room_id is not None and observation.room_id != entry.room_id:
            reasons.append("room_changed")
        if observation.scene_scope is not None and observation.scene_scope != entry.scene_scope:
            reasons.append("scene_changed")

        if observation.world_pose_xyz is not None and entry.world_pose_xyz is not None:
            distance_m = math.dist(observation.world_pose_xyz, entry.world_pose_xyz)
            if distance_m >= max(float(persist_position_delta_m), 0.0):
                reasons.append("pose_changed")
        elif observation.world_pose_xyz != entry.world_pose_xyz:
            reasons.append("pose_changed")

        if abs(float(observation.detector_conf) - float(entry.last_detector_conf)) >= max(
            float(persist_confidence_delta),
            0.0,
        ):
            reasons.append("confidence_changed")

        if observation.world_pose_xyz is None:
            if _bbox_iou(entry.last_bbox_xyxy_norm, observation.bbox_xyxy_norm) <= 0.35:
                reasons.append("bbox_changed")

        if match.score < 0.80:
            reasons.append("low_confidence_relink")

        return (len(reasons) > 0), reasons

    def _passes_cheap_gate(self, observation: ObjectObservationInput, entry: ObjectMemoryEntry) -> bool:
        obs_center = _center(observation.bbox_xyxy_norm)
        entry_center = _center(entry.last_bbox_xyxy_norm)
        center_dist = math.dist(obs_center, entry_center)
        if center_dist > 0.20:
            return False
        area_ratio = observation.box_area / max(entry.last_box_area, 1e-6)
        if area_ratio < 0.5 or area_ratio > 2.0:
            return False
        aspect_delta = abs(observation.aspect_ratio - entry.last_aspect_ratio)
        if aspect_delta > 0.35:
            return False
        return True

    def _spatial_score(self, observation: ObjectObservationInput, entry: ObjectMemoryEntry) -> float:
        obs_center = _center(observation.bbox_xyxy_norm)
        entry_center = _center(entry.last_bbox_xyxy_norm)
        center_score = max(0.0, 1.0 - (math.dist(obs_center, entry_center) / 0.20))

        area_ratio = observation.box_area / max(entry.last_box_area, 1e-6)
        area_score = max(0.0, 1.0 - min(abs(math.log(max(area_ratio, 1e-6))) / math.log(2.0), 1.0))

        aspect_delta = abs(observation.aspect_ratio - entry.last_aspect_ratio)
        aspect_score = max(0.0, 1.0 - (aspect_delta / 0.35))
        return 0.5 * center_score + 0.25 * area_score + 0.25 * aspect_score

    def _temporal_score(self, observation: ObjectObservationInput, entry: ObjectMemoryEntry) -> float:
        age_sec = max(0.0, (observation.observed_at - entry.last_seen_at).total_seconds())
        return max(0.0, 1.0 - (age_sec / max(self.recent_window.total_seconds(), 1.0)))

    def _to_retrieved(self, entry: ObjectMemoryEntry) -> RetrievedObjectMemory:
        return RetrievedObjectMemory(
            object_id=entry.object_id,
            canonical_class=entry.canonical_class,
            room_id=entry.room_id,
            scene_scope=entry.scene_scope,
            status=entry.status,
            first_seen_at=entry.first_seen_at,
            last_seen_at=entry.last_seen_at,
            observation_count=entry.observation_count,
            dedupe_confidence=entry.dedupe_confidence,
            last_detector_conf=entry.last_detector_conf,
            world_pose_xyz=entry.world_pose_xyz,
            world_pose_observed_at=entry.world_pose_observed_at,
            metadata=dict(entry.metadata),
        )

    def _to_navigation_candidate(self, entry: ObjectMemoryEntry) -> ObjectMemoryNavigationCandidate:
        if entry.world_pose_xyz is None or entry.world_pose_observed_at is None:
            raise ValueError("entry is missing world pose fields")
        pose_age_sec = max(0, int((utc_now() - entry.world_pose_observed_at).total_seconds()))
        return ObjectMemoryNavigationCandidate(
            object_id=entry.object_id,
            class_name=entry.canonical_class,
            room_id=entry.room_id,
            scene_scope=entry.scene_scope,
            world_pose_xyz=entry.world_pose_xyz,
            world_pose_observed_at=entry.world_pose_observed_at,
            pose_age_sec=pose_age_sec,
            last_seen_at=entry.last_seen_at,
            dedupe_confidence=entry.dedupe_confidence,
            last_detector_conf=entry.last_detector_conf,
            metadata=dict(entry.metadata),
        )
