from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
from PIL import Image
import pytest

from backend.webrtc.models import FrameCache
from backend.webrtc.object_memory import ObjectMemoryFrameSink, ObjectMemorySinkConfig
from systems.memory.object_memory_repository import InMemoryObjectMemoryRepository
from systems.memory.object_memory_runtime import ObjectMemoryRuntimeHandle
from systems.memory.object_memory_service import ObjectMemoryService
from systems.perception.detector_runtime import InProcessDetectorRuntime
from systems.perception.observation import PerceptionObservationService
from systems.reasoning.api.runtime import AuraTaskingAdapter
from systems.shared.contracts.observation import RawObservation
from systems.transport import FrameHeader


def _live_yolo_asset() -> Path:
    ultralytics = pytest.importorskip("ultralytics")
    asset_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    if not asset_path.is_file():
        pytest.skip(f"ultralytics bus sample not found: {asset_path}")
    return asset_path


def _model_path() -> Path:
    path = Path("artifacts/models/yoloe-26s-seg-pf.pt")
    if not path.is_file():
        pytest.skip(f"YOLO model not found: {path}")
    return path


def _raw_observation_from_image(image_path: Path) -> RawObservation:
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = rgb.shape[:2]
    return RawObservation(
        rgb=rgb,
        depth=np.full((height, width), 2.0, dtype=np.float32),
        intrinsic=np.asarray(
            ((600.0, 0.0, width / 2.0), (0.0, 600.0, height / 2.0), (0.0, 0.0, 1.0)),
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
    raw: RawObservation,
    detections: list[dict[str, object]],
    *,
    frame_id: int,
    timestamp_ns: int,
) -> FrameCache:
    height, width = raw.rgb.shape[:2]
    return FrameCache(
        seq=frame_id,
        frame_header=FrameHeader(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            source="perception_runtime",
            width=int(width),
            height=int(height),
            rgb_encoding="rgb8",
            depth_encoding="depth32f",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(0.0, 0.0, 0.0),
            robot_yaw_rad=0.0,
            sim_time_s=1.0,
            metadata={},
        ),
        rgb_image=raw.rgb,
        depth_image_m=raw.depth,
        viewer_overlay={"detections": detections},
        last_frame_monotonic=time.monotonic(),
    )


def test_live_yolo_detection_memory_retrieval_resolves_bus_navigation_target() -> None:
    raw = _raw_observation_from_image(_live_yolo_asset())
    detector = InProcessDetectorRuntime(
        enabled=True,
        model_path=str(_model_path()),
        max_inference_hz=1000.0,
    )
    perception = PerceptionObservationService(viewer_publisher=None, detector_runtime=detector)

    observation_frame = perception.ingest(raw)
    detections = observation_frame.metadata["viewer_overlay"]["detections"]
    bus_detection = next((item for item in detections if item.get("class_name") == "bus"), None)

    detector_health = detector.latest_health()
    if detector_health["ready"] is not True:
        pytest.skip(f"YOLO detector unavailable: {detector_health.get('lastError')}")
    assert bus_detection is not None
    assert bus_detection.get("world_pose_xyz") is not None

    repository = InMemoryObjectMemoryRepository()
    memory_runtime = ObjectMemoryRuntimeHandle(
        enabled=True,
        user_id="live-e2e",
        service=ObjectMemoryService(repository),
    )
    sink = ObjectMemoryFrameSink(
        ObjectMemorySinkConfig(
            dsn="memory://live-e2e",
            user_id="live-e2e",
            scene_scope="live-yolo-bus-scene",
            allowed_class_names=("bus",),
            max_bbox_area_norm=1.0,
        ),
        runtime_handle=memory_runtime,
    )
    sink.set_enabled(True)
    timestamp_ns = time.time_ns()
    sink.observe_frame(_frame_from_detections(raw, detections, frame_id=1, timestamp_ns=timestamp_ns))
    sink.observe_frame(_frame_from_detections(raw, detections, frame_id=2, timestamp_ns=timestamp_ns + 1_000_000_000))

    adapter = AuraTaskingAdapter(completion=None, model="live-e2e", timeout=1.0)
    task_frame = adapter.plan_task_frame("go to the bus")
    subgoals = adapter.initialize_subgoals(task_frame)
    _task_frame, resolved_subgoals, resolution = adapter.resolve_memory_navigation(
        task_frame,
        subgoals,
        object_memory_runtime=memory_runtime,
        scene_scope="live-yolo-bus-scene",
    )

    bus_entries = [
        entry
        for entry in repository.list_object_entries("live-e2e", statuses=("active",))
        if entry.canonical_class == "bus"
    ]
    navigation_target = resolved_subgoals[0]["input"]["navigation_target"]

    assert bus_detection["memory_status"] == "created"
    assert len(bus_entries) == 1
    assert resolution is not None
    assert resolution["status"] == "resolved"
    assert navigation_target["mode"] == "memory_pose"
    assert navigation_target["class_name"] == "bus"
    assert navigation_target["object_id"] == bus_entries[0].object_id
    assert navigation_target["world_pose_xyz"] == list(bus_entries[0].world_pose_xyz)
