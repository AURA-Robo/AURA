from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import time
import uuid

import numpy as np
from PIL import Image
import pytest

from backend.webrtc.models import FrameCache
from backend.webrtc.object_memory import ObjectMemoryFrameSink, ObjectMemorySinkConfig
from systems.memory.object_memory_runtime import create_object_memory_runtime
from systems.perception.detector_runtime import InProcessDetectorRuntime
from systems.perception.observation import PerceptionObservationService
from systems.reasoning.api.runtime import AuraTaskingAdapter
from systems.shared.contracts.observation import RawObservation
from systems.transport import FrameHeader


def _test_object_memory_dsn() -> str:
    dsn = str(os.environ.get("AURA_TEST_OBJECT_MEMORY_DSN", "")).strip()
    if not dsn:
        pytest.skip("set AURA_TEST_OBJECT_MEMORY_DSN to run Postgres-backed YOLO object-memory E2E")
    return dsn


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


def _cleanup_postgres_user(dsn: str, user_id: str) -> None:
    psycopg = pytest.importorskip("psycopg")
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute("DELETE FROM object_memory_entries WHERE user_id = %s", (user_id,))


def test_live_yolo_detection_persists_to_postgres_and_resolves_from_memory() -> None:
    dsn = _test_object_memory_dsn()
    raw = _raw_observation_from_image(_live_yolo_asset())
    user_id = f"live-postgres-e2e-{uuid.uuid4().hex}"
    scene_scope = f"live-yolo-postgres-{uuid.uuid4().hex}"
    runtime_available = False

    try:
        runtime = create_object_memory_runtime(
            enabled=True,
            dsn=dsn,
            user_id=user_id,
            auto_migrate=True,
        )
        runtime_available = runtime.available
        assert runtime.available is True, runtime.degraded_reason

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

        sink = ObjectMemoryFrameSink(
            ObjectMemorySinkConfig(
                dsn=dsn,
                user_id=user_id,
                scene_scope=scene_scope,
                auto_migrate=True,
                allowed_class_names=("bus",),
                max_bbox_area_norm=1.0,
            ),
            runtime_handle=runtime,
        )
        sink.set_enabled(True)
        timestamp_ns = time.time_ns()
        sink.observe_frame(_frame_from_detections(raw, detections, frame_id=1, timestamp_ns=timestamp_ns))
        sink.observe_frame(_frame_from_detections(raw, detections, frame_id=2, timestamp_ns=timestamp_ns + 1_000_000_000))

        adapter = AuraTaskingAdapter(completion=None, model="live-postgres-e2e", timeout=1.0)
        task_frame = adapter.plan_task_frame("go to the bus")
        subgoals = adapter.initialize_subgoals(task_frame)
        _task_frame, resolved_subgoals, resolution = adapter.resolve_memory_navigation(
            task_frame,
            subgoals,
            object_memory_runtime=runtime,
            scene_scope=scene_scope,
        )

        assert runtime.count_objects() >= 1
        assert resolution is not None
        assert resolution["status"] == "resolved"
        navigation_target = resolved_subgoals[0]["input"]["navigation_target"]
        assert navigation_target["mode"] == "memory_pose"
        assert navigation_target["class_name"] == "bus"
        assert navigation_target["object_id"] == bus_detection["memory_object_id"]
    finally:
        if runtime_available:
            _cleanup_postgres_user(dsn, user_id)
