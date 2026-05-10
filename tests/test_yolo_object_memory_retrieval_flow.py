from __future__ import annotations

from types import SimpleNamespace
import time

import numpy as np

from backend.webrtc.models import FrameCache
from backend.webrtc.object_memory import ObjectMemoryFrameSink, ObjectMemorySinkConfig
from systems.memory.object_memory_repository import InMemoryObjectMemoryRepository
from systems.memory.object_memory_runtime import ObjectMemoryRuntimeHandle
from systems.memory.object_memory_service import ObjectMemoryService
from systems.perception.detector_runtime import DetectorOutput
from systems.perception.observation import PerceptionObservationService
from systems.reasoning.api.runtime import AuraTaskingAdapter
from systems.shared.contracts.observation import RawObservation
from systems.transport import FrameHeader


class _DetectorStub:
    def detect(self, **_: object) -> DetectorOutput:
        return DetectorOutput(
            detections=[
                {
                    "class_name": "chair",
                    "track_id": "track-chair-1",
                    "bbox_xyxy": [10, 20, 110, 160],
                    "confidence": 0.93,
                    "world_pose_xyz": [1.0, 2.0, 0.0],
                }
            ],
            detector_backend="ultralytics-yoloe",
        )

    def latest_health(self) -> dict[str, object]:
        return {
            "enabled": True,
            "ready": True,
            "backend": "ultralytics-yoloe",
            "lastError": None,
        }

    def close(self) -> None:
        return None


def _raw_observation() -> RawObservation:
    return RawObservation(
        rgb=np.zeros((180, 320, 3), dtype=np.uint8),
        depth=np.ones((180, 320), dtype=np.float32),
        intrinsic=np.asarray(
            ((100.0, 0.0, 160.0), (0.0, 100.0, 90.0), (0.0, 0.0, 1.0)),
            dtype=np.float32,
        ),
        camera_pos_w=np.zeros(3, dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
        robot_state=SimpleNamespace(
            base_pos_w=np.zeros(3, dtype=np.float32),
            base_yaw=0.0,
        ),
        stamp_s=time.monotonic(),
        metadata={},
    )


def _frame_from_detections(
    detections: list[dict[str, object]],
    *,
    frame_id: int,
    timestamp_ns: int,
) -> FrameCache:
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
            robot_pose_xyz=(0.0, 0.0, 0.0),
            robot_yaw_rad=0.0,
            sim_time_s=1.0,
            metadata={},
        ),
        rgb_image=np.zeros((180, 320, 3), dtype=np.uint8),
        depth_image_m=np.ones((180, 320), dtype=np.float32),
        viewer_overlay={"detections": detections},
        last_frame_monotonic=time.monotonic(),
    )


def test_yolo_detection_memory_retrieval_flow_resolves_navigation_target() -> None:
    perception = PerceptionObservationService(
        viewer_publisher=None,
        detector_runtime=_DetectorStub(),
    )

    observation_frame = perception.ingest(_raw_observation())
    detections = observation_frame.metadata["viewer_overlay"]["detections"]

    repository = InMemoryObjectMemoryRepository()
    memory_runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="tester",
        service=ObjectMemoryService(repository),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(
            dsn="postgres://configured",
            user_id="tester",
            scene_scope="warehouse",
        ),
        runtime_handle=memory_runtime,
    )
    sink.set_enabled(True)
    timestamp_ns = time.time_ns()
    sink.observe_frame(_frame_from_detections(detections, frame_id=1, timestamp_ns=timestamp_ns))
    assert detections[0]["memory_status"] == "pending"
    sink.observe_frame(_frame_from_detections(detections, frame_id=2, timestamp_ns=timestamp_ns + 1_000_000_000))

    adapter = AuraTaskingAdapter(completion=None, model="test-model", timeout=1.0)
    task_frame = adapter.plan_task_frame("go to the chair")
    subgoals = adapter.initialize_subgoals(task_frame)
    _resolved_task_frame, resolved_subgoals, resolution = adapter.resolve_memory_navigation(
        task_frame,
        subgoals,
        object_memory_runtime=memory_runtime,
        scene_scope="warehouse",
    )

    entries = repository.list_object_entries("tester", statuses=("active",))
    assert len(entries) == 1
    assert entries[0].canonical_class == "chair"
    assert entries[0].world_pose_xyz == (1.0, 2.0, 0.0)
    assert detections[0]["memory_status"] == "created"
    assert memory_runtime.recent_context(scene_scope="warehouse").recent_seen
    assert resolution is not None
    assert resolution["status"] == "resolved"
    navigation_target = resolved_subgoals[0]["input"]["navigation_target"]
    assert navigation_target["mode"] == "memory_pose"
    assert navigation_target["object_id"] == entries[0].object_id
    assert navigation_target["world_pose_xyz"] == [1.0, 2.0, 0.0]
