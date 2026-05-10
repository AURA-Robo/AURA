from __future__ import annotations

from datetime import timedelta

import systems.memory.object_memory_runtime as object_memory_runtime_module
from systems.memory.object_memory_models import ObjectObservationInput, utc_now
from systems.memory.object_memory_repository import InMemoryObjectMemoryRepository
from systems.memory.object_memory_runtime import ObjectMemoryRuntimeHandle, create_object_memory_runtime
from systems.memory.object_memory_service import ObjectMemoryService


def _observation(
    *,
    frame_idx: int,
    track_id: str,
    class_name: str = "chair",
    observed_at=None,
    source_id: str = "camera-0",
    bbox_xyxy_norm: tuple[float, float, float, float] = (0.1, 0.1, 0.4, 0.5),
    world_pose_xyz: tuple[float, float, float] = (1.0, 2.0, 0.0),
) -> ObjectObservationInput:
    return ObjectObservationInput(
        frame_idx=frame_idx,
        track_id=track_id,
        class_name=class_name,
        detector_conf=0.95,
        bbox_xyxy_norm=bbox_xyxy_norm,
        box_area=0.0,
        aspect_ratio=0.0,
        image_hash=f"image-{frame_idx}",
        observed_at=observed_at or utc_now(),
        room_id=None,
        scene_scope="warehouse",
        world_pose_xyz=world_pose_xyz,
        world_pose_observed_at=observed_at or utc_now(),
        source_id=source_id,
        attributes={},
    )


def test_object_memory_roomless_recent_seen_projection_keeps_entries() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)

    service.observe_objects(
        "tester",
        "session-1",
        [_observation(frame_idx=1, track_id="track-1")],
    )

    context = service.query_recent_objects("tester")

    assert len(context.entries) == 1
    assert len(context.recent_seen) == 1
    assert context.recent_seen[0]["class"] == "chair"
    assert context.recent_seen[0]["room"] is None


def test_object_memory_track_match_updates_without_embeddings() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    observed_at = utc_now() - timedelta(seconds=5)

    first = service.observe_objects(
        "tester",
        "session-1",
        [_observation(frame_idx=1, track_id="track-1", observed_at=observed_at)],
    )
    second = service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(
                frame_idx=2,
                track_id="track-1",
                observed_at=observed_at + timedelta(seconds=1),
                bbox_xyxy_norm=(0.12, 0.1, 0.42, 0.5),
            )
        ],
    )

    assert len(first.created_object_ids) == 1
    assert len(second.updated_object_ids) == 1
    assert repository.count_object_entries("tester", statuses=("active",)) == 1
    observations = repository.list_object_observations("tester")
    assert len(observations) == 2
    entry = repository.list_object_entries("tester", statuses=("active",))[0]
    assert entry.observation_count == 2
    assert entry.metadata["last_track_id"] == "track-1"
    assert entry.scene_scope == "warehouse"
    assert entry.world_pose_xyz == (1.0, 2.0, 0.0)


def test_object_memory_resolves_fresh_unique_pose_candidate() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    observed_at = utc_now() - timedelta(seconds=5)
    service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(
                frame_idx=1,
                track_id="track-chair-1",
                observed_at=observed_at,
            )
        ],
    )

    resolution = service.resolve_memory_navigation_target(
        "tester",
        scene_scope="warehouse",
        class_name="chair",
        max_pose_age_sec=600,
    )

    assert resolution.status == "resolved"
    assert resolution.selected is not None
    assert resolution.selected.class_name == "chair"
    assert resolution.selected.scene_scope == "warehouse"
    assert resolution.selected.world_pose_xyz == (1.0, 2.0, 0.0)


def test_object_memory_marks_multiple_fresh_candidates_as_ambiguous() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    now = utc_now()
    service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(frame_idx=1, track_id="track-chair-1", observed_at=now - timedelta(seconds=4)),
            _observation(
                frame_idx=2,
                track_id="track-chair-2",
                observed_at=now - timedelta(seconds=2),
                bbox_xyxy_norm=(0.45, 0.1, 0.7, 0.5),
                world_pose_xyz=(3.0, 4.0, 0.0),
                source_id="camera-1",
            ),
        ],
    )

    resolution = service.resolve_memory_navigation_target(
        "tester",
        scene_scope="warehouse",
        class_name="chair",
        max_pose_age_sec=600,
    )

    assert resolution.status == "ambiguous"
    assert len(resolution.candidates) == 2


def test_object_memory_sparse_policy_suppresses_repeat_track_persistence() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    observed_at = utc_now() - timedelta(seconds=5)

    first = service.observe_objects(
        "tester",
        "session-1",
        [_observation(frame_idx=1, track_id="track-1", observed_at=observed_at)],
        persistence_policy="sparse",
    )
    second = service.observe_objects(
        "tester",
        "session-1",
        [_observation(frame_idx=2, track_id="track-1", observed_at=observed_at + timedelta(seconds=1))],
        persistence_policy="sparse",
    )

    assert len(first.created_object_ids) == 1
    assert len(first.observation_ids) == 1
    assert len(second.updated_object_ids) == 0
    assert len(second.observation_ids) == 0
    assert second.suppressed_observation_count == 1
    assert len(second.links) == 1
    assert second.links[0].status == "linked"
    assert second.links[0].persisted is False
    assert repository.count_object_entries("tester", statuses=("active",)) == 1
    assert len(repository.list_object_observations("tester")) == 1


def test_object_memory_relinks_same_object_by_world_pose_without_embeddings() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    observed_at = utc_now() - timedelta(seconds=5)

    first = service.observe_objects(
        "tester",
        "session-1",
        [_observation(frame_idx=1, track_id="track-1", observed_at=observed_at)],
    )
    second = service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(
                frame_idx=2,
                track_id="track-2",
                observed_at=observed_at + timedelta(seconds=2),
                bbox_xyxy_norm=(0.11, 0.1, 0.41, 0.5),
                world_pose_xyz=(1.08, 2.02, 0.0),
            )
        ],
    )

    assert len(first.created_object_ids) == 1
    assert len(second.updated_object_ids) == 1
    assert second.links[0].status == "linked"
    assert second.links[0].match_score >= 0.72
    assert repository.count_object_entries("tester", statuses=("active",)) == 1


def test_object_memory_filters_out_stale_scene_and_room_mismatched_candidates() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    now = utc_now()
    service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(
                frame_idx=1,
                track_id="track-chair-1",
                observed_at=now - timedelta(seconds=900),
            )
        ],
    )
    service.observe_objects(
        "tester",
        "session-1",
        [
            ObjectObservationInput(
                frame_idx=2,
                track_id="track-chair-2",
                class_name="chair",
                detector_conf=0.95,
                bbox_xyxy_norm=(0.15, 0.1, 0.45, 0.5),
                box_area=0.0,
                aspect_ratio=0.0,
                image_hash="image-2",
                observed_at=now - timedelta(seconds=5),
                room_id="kitchen",
                scene_scope="warehouse",
                world_pose_xyz=(2.0, 3.0, 0.0),
                world_pose_observed_at=now - timedelta(seconds=5),
                source_id="camera-0",
                attributes={},
            )
        ],
    )
    service.observe_objects(
        "tester",
        "session-1",
        [
            ObjectObservationInput(
                frame_idx=3,
                track_id="track-chair-3",
                class_name="chair",
                detector_conf=0.95,
                bbox_xyxy_norm=(0.5, 0.1, 0.8, 0.5),
                box_area=0.0,
                aspect_ratio=0.0,
                image_hash="image-3",
                observed_at=now - timedelta(seconds=5),
                room_id="office",
                scene_scope="interioragent",
                world_pose_xyz=(4.0, 5.0, 0.0),
                world_pose_observed_at=now - timedelta(seconds=5),
                source_id="camera-0",
                attributes={},
            )
        ],
    )

    resolved_resolution = service.resolve_memory_navigation_target(
        "tester",
        scene_scope="warehouse",
        class_name="chair",
        max_pose_age_sec=600,
    )
    room_resolution = service.resolve_memory_navigation_target(
        "tester",
        scene_scope="warehouse",
        class_name="chair",
        room_hint="kitchen",
        max_pose_age_sec=600,
    )
    missing_scene = service.resolve_memory_navigation_target(
        "tester",
        scene_scope="interioragent",
        class_name="table",
        max_pose_age_sec=600,
    )

    assert resolved_resolution.status == "resolved"
    assert resolved_resolution.selected is not None
    assert resolved_resolution.selected.room_id == "kitchen"
    assert room_resolution.status == "resolved"
    assert room_resolution.selected is not None
    assert room_resolution.selected.room_id == "kitchen"
    assert missing_scene.status == "no_candidate"


def test_object_memory_returns_stale_only_when_pose_candidates_are_old() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    observed_at = utc_now() - timedelta(seconds=1200)
    service.observe_objects(
        "tester",
        "session-1",
        [_observation(frame_idx=1, track_id="track-stale", observed_at=observed_at)],
    )

    resolution = service.resolve_memory_navigation_target(
        "tester",
        scene_scope="warehouse",
        class_name="chair",
        max_pose_age_sec=600,
    )

    assert resolution.status == "stale_only"
    assert len(resolution.candidates) == 1


def test_object_memory_runtime_fails_open_without_dsn() -> None:
    runtime = create_object_memory_runtime(enabled=True, dsn="", user_id="")

    assert runtime.enabled is True
    assert runtime.available is False
    assert runtime.user_id == "local-operator"
    assert runtime.degraded_reason == "object_memory_dsn_missing"
    assert runtime.recent_context().recent_seen == []


def test_object_memory_runtime_auto_migrates_and_verifies_postgres_schema(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class _Repository(InMemoryObjectMemoryRepository):
        def __init__(self, dsn: str) -> None:
            calls.append(("init", dsn))
            super().__init__()

        def apply_schema(self) -> None:
            calls.append(("apply_schema", None))

        def verify_schema(self) -> None:
            calls.append(("verify_schema", None))

    monkeypatch.setattr(object_memory_runtime_module, "PostgresObjectMemoryRepository", _Repository)

    runtime = create_object_memory_runtime(
        enabled=True,
        dsn="postgresql://example/object-memory",
        user_id="tester",
        auto_migrate=True,
    )

    assert runtime.available is True
    assert runtime.service is not None
    assert calls == [
        ("init", "postgresql://example/object-memory"),
        ("apply_schema", None),
        ("verify_schema", None),
    ]


def test_object_memory_runtime_degrades_when_postgres_schema_is_missing(monkeypatch) -> None:
    class _Repository(InMemoryObjectMemoryRepository):
        def __init__(self, dsn: str) -> None:
            super().__init__()

        def verify_schema(self) -> None:
            raise RuntimeError("missing tables")

    monkeypatch.setattr(object_memory_runtime_module, "PostgresObjectMemoryRepository", _Repository)

    runtime = create_object_memory_runtime(
        enabled=True,
        dsn="postgresql://example/object-memory",
        user_id="tester",
        auto_migrate=False,
    )

    assert runtime.enabled is True
    assert runtime.available is False
    assert runtime.service is None
    assert runtime.degraded_reason == "RuntimeError: missing tables"


def test_object_memory_runtime_fails_open_on_repository_errors() -> None:
    class _BrokenRepository(InMemoryObjectMemoryRepository):
        def list_object_entries(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("db unavailable")

        def count_object_entries(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("db unavailable")

    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(_BrokenRepository()),
    )

    context = runtime.recent_context()

    assert context.recent_seen == []
    assert runtime.available is False
    assert runtime.count_objects() == 0
    assert runtime.degraded_reason == "RuntimeError: db unavailable"
