from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from backend.webrtc.models import build_frame_meta_message
from backend.webrtc.models import FrameCache
from backend.webrtc.object_memory import ObjectMemoryFrameSink, ObjectMemorySinkConfig
from systems.memory.object_memory_runtime import ObjectMemoryRuntimeHandle
from systems.memory.object_memory_repository import InMemoryObjectMemoryRepository
from systems.memory.object_memory_service import ObjectMemoryService
from systems.transport import FrameHeader


def _frame(
    *,
    frame_id: int,
    timestamp_ns: int,
    class_name: str = "chair",
    class_id: int | None = None,
    track_id: str = "track-1",
    bbox_xyxy: list[int] | None = None,
    confidence: float = 0.92,
    world_pose_xyz: list[float] | None = None,
    detector_cached: bool = False,
) -> FrameCache:
    overlay: dict[str, object] = {
        "detections": [
            {
                "class_name": class_name,
                "track_id": track_id,
                "bbox_xyxy": bbox_xyxy or [10, 20, 110, 160],
                "confidence": confidence,
                "world_pose_xyz": world_pose_xyz or [1.0, 2.0, 0.0],
            }
        ]
    }
    if class_id is not None:
        overlay["detections"][0]["class_id"] = class_id
    if detector_cached:
        overlay["detector_cached"] = True
        overlay["detector_cache_reason"] = "camera_motion_below_threshold"
    return FrameCache(
        seq=frame_id,
        frame_header=FrameHeader(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            source="perception_runtime",
            width=320,
            height=180,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(1.0, 2.0, 3.0),
            robot_yaw_rad=0.25,
            sim_time_s=4.5,
            metadata={},
        ),
        rgb_image=np.zeros((180, 320, 3), dtype=np.uint8),
        depth_image_m=None,
        viewer_overlay=overlay,
        last_frame_monotonic=time.monotonic(),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_object_memory_sink_is_idempotent_per_frame_detection() -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)
    frame = _frame(frame_id=1, timestamp_ns=1_000_000_000)

    sink.observe_frame(frame)
    sink.observe_frame(frame)

    health = sink.health_snapshot()
    assert health["objectCount"] == 0
    assert health["observationCount"] == 0
    assert health["createdCount"] == 0
    assert health["updatedCount"] == 0
    assert health["pendingCandidateCount"] == 1
    assert health["pendingSuppressedCount"] == 1
    assert health["skippedDuplicateCount"] == 1
    assert health["lastError"] is None
    assert frame.viewer_overlay["detections"][0]["memory_status"] == "pending"
    assert frame.viewer_overlay["detections"][0]["memory_persisted"] is False


def test_object_memory_sink_confirms_new_object_after_two_stable_detections() -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    first_frame = _frame(frame_id=1, timestamp_ns=1_000_000_000, track_id="track-7")
    second_frame = _frame(frame_id=2, timestamp_ns=2_000_000_000, track_id="track-7")
    sink.observe_frame(first_frame)
    sink.observe_frame(second_frame)

    health = sink.health_snapshot()
    assert health["objectCount"] == 1
    assert health["observationCount"] == 1
    assert health["createdCount"] == 1
    assert health["pendingCandidateCount"] == 0
    assert health["pendingSuppressedCount"] == 1
    assert first_frame.viewer_overlay["detections"][0]["memory_status"] == "pending"
    assert second_frame.viewer_overlay["detections"][0]["memory_status"] == "created"
    assert second_frame.viewer_overlay["detections"][0]["memory_persisted"] is True


def test_object_memory_sink_filters_non_actionable_scene_labels() -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    sink.observe_frame(
        _frame(
            frame_id=1,
            timestamp_ns=1_000_000_000,
            class_name="court",
            bbox_xyxy=[0, 264, 446, 446],
            confidence=0.86,
        )
    )
    sink.observe_frame(
        _frame(
            frame_id=2,
            timestamp_ns=2_000_000_000,
            class_name="court",
            bbox_xyxy=[0, 263, 446, 446],
            confidence=0.86,
        )
    )

    health = sink.health_snapshot()
    assert health["objectCount"] == 0
    assert health["observationCount"] == 0
    assert health["filteredDetectionCount"] == 2
    assert health["filteredDetectionReasons"]["class_not_allowed"] == 2


def test_object_memory_sink_filters_low_confidence_detections() -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    sink.observe_frame(_frame(frame_id=1, timestamp_ns=1_000_000_000, confidence=0.42))

    health = sink.health_snapshot()
    assert health["objectCount"] == 0
    assert health["pendingCandidateCount"] == 0
    assert health["filteredDetectionCount"] == 1
    assert health["filteredDetectionReasons"]["low_confidence"] == 1


def test_object_memory_sink_updates_existing_track_across_frames() -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    first_frame = _frame(frame_id=1, timestamp_ns=1_000_000_000, track_id="track-7")
    second_frame = _frame(frame_id=2, timestamp_ns=2_000_000_000, track_id="track-7")
    third_frame = _frame(frame_id=3, timestamp_ns=3_000_000_000, track_id="track-7")
    sink.observe_frame(first_frame)
    sink.observe_frame(second_frame)
    sink.observe_frame(third_frame)

    health = sink.health_snapshot()
    assert health["objectCount"] == 1
    assert health["observationCount"] == 1
    assert health["createdCount"] == 1
    assert health["updatedCount"] == 0
    assert health["linkedCount"] == 1
    assert health["suppressedObservationCount"] == 1
    assert health["lastSuccess"] is True
    assert first_frame.viewer_overlay["detections"][0]["memory_status"] == "pending"
    assert second_frame.viewer_overlay["detections"][0]["memory_status"] == "created"
    assert third_frame.viewer_overlay["detections"][0]["memory_status"] == "linked"
    assert third_frame.viewer_overlay["detections"][0]["memory_persisted"] is False
    assert (
        third_frame.viewer_overlay["detections"][0]["memory_object_id"]
        == second_frame.viewer_overlay["detections"][0]["memory_object_id"]
    )
    payload = build_frame_meta_message(third_frame)
    assert payload["detections"][0]["memory_object_id"] == third_frame.viewer_overlay["detections"][0]["memory_object_id"]
    assert payload["detections"][0]["memory_status"] == "linked"


def test_object_memory_sink_records_scene_scope_and_world_pose() -> None:
    repository = InMemoryObjectMemoryRepository()
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(repository),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester", scene_scope="warehouse"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)
    sink.observe_frame(_frame(frame_id=3, timestamp_ns=3_000_000_000, track_id="track-9"))
    sink.observe_frame(_frame(frame_id=4, timestamp_ns=4_000_000_000, track_id="track-9"))

    entries = repository.list_object_entries("tester", statuses=("active",))
    assert len(entries) == 1
    assert entries[0].scene_scope == "warehouse"
    assert entries[0].world_pose_xyz == (1.0, 2.0, 0.0)
    observations = repository.list_object_observations("tester", object_ids=[entries[0].object_id])
    assert len(observations) == 1
    assert observations[0].scene_scope == "warehouse"
    assert observations[0].world_pose_xyz == (1.0, 2.0, 0.0)


def test_object_memory_sink_links_existing_object_without_pending_confirmation() -> None:
    repository = InMemoryObjectMemoryRepository()
    service = ObjectMemoryService(repository)
    runtime = ObjectMemoryRuntimeHandle(enabled=True, user_id="tester", service=service)
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester", scene_scope="warehouse"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)
    sink.observe_frame(_frame(frame_id=1, timestamp_ns=1_000_000_000, track_id="track-1"))
    sink.observe_frame(_frame(frame_id=2, timestamp_ns=2_000_000_000, track_id="track-1"))

    existing_frame = _frame(frame_id=3, timestamp_ns=3_000_000_000, track_id="track-2")
    sink.observe_frame(existing_frame)

    health = sink.health_snapshot()
    assert health["objectCount"] == 1
    assert health["linkedCount"] == 1
    assert existing_frame.viewer_overlay["detections"][0]["memory_status"] == "linked"
    assert existing_frame.viewer_overlay["detections"][0]["memory_persisted"] is True


def test_object_memory_sink_skips_cached_detector_frames() -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    sink.observe_frame(_frame(frame_id=1, timestamp_ns=1_000_000_000, detector_cached=True))

    health = sink.health_snapshot()
    assert health["objectCount"] == 0
    assert health["observationCount"] == 0
    assert health["cachedDetectorSkipCount"] == 1


def test_object_memory_sink_expires_pending_candidate_after_ttl() -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(
            dsn="postgres://configured",
            user_id="tester",
            candidate_ttl_sec=1,
        ),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    first_frame = _frame(frame_id=1, timestamp_ns=1_000_000_000, track_id="track-expire")
    expired_frame = _frame(frame_id=2, timestamp_ns=4_000_000_000, track_id="track-expire")
    confirm_frame = _frame(frame_id=3, timestamp_ns=5_000_000_000, track_id="track-expire")
    sink.observe_frame(first_frame)
    sink.observe_frame(expired_frame)
    sink.observe_frame(confirm_frame)

    health = sink.health_snapshot()
    assert first_frame.viewer_overlay["detections"][0]["memory_status"] == "pending"
    assert expired_frame.viewer_overlay["detections"][0]["memory_status"] == "pending"
    assert confirm_frame.viewer_overlay["detections"][0]["memory_status"] == "created"
    assert health["objectCount"] == 1
    assert health["pendingSuppressedCount"] == 2


def test_object_memory_sink_event_log_is_disabled_by_default(tmp_path: Path) -> None:
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester"),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    sink.observe_frame(_frame(frame_id=1, timestamp_ns=1_000_000_000))
    sink.observe_frame(_frame(frame_id=2, timestamp_ns=2_000_000_000))

    health = sink.health_snapshot()
    assert health["eventLogConfigured"] is False
    assert health["eventLogWriteCount"] == 0
    assert list(tmp_path.iterdir()) == []


def test_object_memory_sink_writes_pending_and_created_events(tmp_path: Path) -> None:
    output_path = tmp_path / "object_events.jsonl"
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(
            dsn="postgres://configured",
            user_id="tester",
            object_event_log_path=str(output_path),
        ),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    first_frame = _frame(
        frame_id=1,
        timestamp_ns=1_000_000_000,
        class_id=821,
        track_id="track-7",
    )
    second_frame = _frame(
        frame_id=2,
        timestamp_ns=2_000_000_000,
        class_id=821,
        track_id="track-7",
    )
    sink.observe_frame(first_frame)
    sink.observe_frame(second_frame)

    events = _read_jsonl(output_path)
    assert [event["object"]["status"] for event in events] == ["pending", "created"]
    assert events[0]["frame_index"] == 1
    assert events[0]["timestamp_ms"] == 1000
    assert events[0]["detection"]["class_id"] == 821
    assert events[0]["detection"]["tracker_id"] == "track-7"
    assert str(events[0]["object"]["object_id"]).startswith("pending-")
    assert events[1]["object"]["object_id"] == second_frame.viewer_overlay["detections"][0]["memory_object_id"]
    assert events[1]["object"]["persisted"] is True
    assert isinstance(events[1]["object"]["observation_id"], str)
    assert sink.health_snapshot()["eventLogWriteCount"] == 2


def test_object_memory_sink_writes_filtered_events(tmp_path: Path) -> None:
    output_path = tmp_path / "object_events.jsonl"
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(
            dsn="postgres://configured",
            user_id="tester",
            object_event_log_path=str(output_path),
        ),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    sink.observe_frame(_frame(frame_id=1, timestamp_ns=1_000_000_000, confidence=0.42))

    events = _read_jsonl(output_path)
    assert len(events) == 1
    assert events[0]["object"]["status"] == "filtered"
    assert events[0]["object"]["filter_reason"] == "low_confidence"
    assert events[0]["object"]["persisted"] is False


def test_object_memory_sink_event_log_failure_does_not_break_ingest(tmp_path: Path) -> None:
    invalid_output_path = tmp_path / "object_events.jsonl"
    invalid_output_path.mkdir()
    runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(InMemoryObjectMemoryRepository()),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(
            dsn="postgres://configured",
            user_id="tester",
            object_event_log_path=str(invalid_output_path),
        ),
        runtime_handle=runtime,
    )
    sink.set_enabled(True)

    sink.observe_frame(_frame(frame_id=1, timestamp_ns=1_000_000_000))
    sink.observe_frame(_frame(frame_id=2, timestamp_ns=2_000_000_000))

    health = sink.health_snapshot()
    assert health["objectCount"] == 1
    assert health["eventLogConfigured"] is True
    assert health["eventLogWriteCount"] == 0
    assert "Error" in str(health["eventLogLastError"])
