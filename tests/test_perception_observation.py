from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from systems.perception.detector_runtime import DetectorOutput, InProcessDetectorRuntime, _as_float
from systems.perception.observation import PerceptionObservationService
from systems.shared.contracts.observation import RawObservation


def test_perception_normalizes_raw_observation_without_viewer() -> None:
    service = PerceptionObservationService(viewer_publisher=None)
    raw = RawObservation(
        rgb=np.full((2, 3, 3), 0.5, dtype=np.float32),
        depth=np.asarray([[1.0, np.nan, np.inf]], dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
        camera_pos_w=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
        robot_state=SimpleNamespace(base_pos_w=np.asarray((0.0, 0.0, 0.8), dtype=np.float32), base_yaw=0.25),
        stamp_s=12.5,
    )

    frame = service.ingest(raw)

    assert frame.rgb.dtype == np.uint8
    assert frame.rgb.shape == (2, 3, 3)
    assert int(frame.rgb[0, 0, 0]) == 127
    assert frame.depth.dtype == np.float32
    assert frame.depth.tolist() == [[1.0, 0.0, 0.0]]
    health = service.latest_health()
    assert health["status"] == "running"
    assert health["last_error"] is None


def test_detector_float_parser_accepts_numpy_scalars() -> None:
    assert _as_float(np.float32(2.5)) == 2.5
    assert _as_float(np.asarray(3.0, dtype=np.float32)) == 3.0
    assert _as_float(np.asarray([3.0], dtype=np.float32)) is None
    assert _as_float(np.bool_(True)) is None


def test_perception_merges_detector_metadata_into_viewer_overlay() -> None:
    class _DetectorStub:
        def detect(self, **_: object) -> DetectorOutput:
            return DetectorOutput(
                detections=[
                    {
                        "class_id": 821,
                        "class_name": "chair",
                        "track_id": "7",
                        "bbox_xyxy": (10.2, 12.7, 20.0, 22.0),
                        "confidence": 0.91,
                        "depth_m": 2.4,
                        "world_pose_xyz": (1.0, 2.0, 0.5),
                    }
                ],
                detector_backend="ultralytics-yoloe",
            )

        def latest_health(self) -> dict[str, object]:
            return {"enabled": True, "ready": True, "backend": "ultralytics-yoloe", "lastError": None}

        def close(self) -> None:
            return None

    service = PerceptionObservationService(viewer_publisher=None, detector_runtime=_DetectorStub())
    raw = RawObservation(
        rgb=np.full((32, 32, 3), 255, dtype=np.uint8),
        depth=np.full((32, 32), 2.0, dtype=np.float32),
        intrinsic=np.asarray(((10.0, 0.0, 16.0), (0.0, 10.0, 16.0), (0.0, 0.0, 1.0)), dtype=np.float32),
        camera_pos_w=np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
        robot_state=SimpleNamespace(base_pos_w=np.asarray((0.0, 0.0, 0.8), dtype=np.float32), base_yaw=0.0),
        stamp_s=1.0,
        metadata={"viewer_overlay": {"trajectory_pixels": [[1, 2], [3, 4]]}},
    )

    frame = service.ingest(raw)

    assert frame.metadata["detector_backend"] == "ultralytics-yoloe"
    assert frame.metadata["viewer_overlay"]["trajectory_pixels"] == [[1, 2], [3, 4]]
    assert frame.metadata["viewer_overlay"]["detections"][0]["class_name"] == "chair"
    assert frame.metadata["viewer_overlay"]["detections"][0]["class_id"] == 821
    assert frame.metadata["viewer_overlay"]["detections"][0]["track_id"] == "7"
    assert frame.metadata["viewer_overlay"]["detections"][0]["bbox_xyxy"] == [10, 13, 20, 22]
    assert frame.metadata["viewer_overlay"]["detections"][0]["depth_m"] == 2.4
    assert frame.metadata["viewer_overlay"]["detections"][0]["world_pose_xyz"] == [1.0, 2.0, 0.5]
    health = service.latest_health()
    assert health["detector"]["ready"] is True


def test_perception_marks_cached_detector_output_in_viewer_overlay() -> None:
    class _DetectorStub:
        def detect(self, **_: object) -> DetectorOutput:
            return DetectorOutput(
                detections=[
                    {
                        "class_name": "chair",
                        "track_id": "cached-track",
                        "bbox_xyxy": [10, 12, 20, 22],
                        "confidence": 0.91,
                    }
                ],
                detector_backend="ultralytics-yoloe",
                cached=True,
                cache_reason="camera_motion_below_threshold",
            )

        def latest_health(self) -> dict[str, object]:
            return {"enabled": True, "ready": True, "backend": "ultralytics-yoloe", "lastError": None}

        def close(self) -> None:
            return None

    service = PerceptionObservationService(viewer_publisher=None, detector_runtime=_DetectorStub())
    raw = RawObservation(
        rgb=np.full((32, 32, 3), 255, dtype=np.uint8),
        depth=np.full((32, 32), 2.0, dtype=np.float32),
        intrinsic=np.eye(3, dtype=np.float32),
        camera_pos_w=np.zeros(3, dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
        robot_state=SimpleNamespace(base_pos_w=np.zeros(3, dtype=np.float32), base_yaw=0.0),
        stamp_s=1.0,
        metadata={},
    )

    frame = service.ingest(raw)

    overlay = frame.metadata["viewer_overlay"]
    assert overlay["detector_cached"] is True
    assert overlay["detector_cache_reason"] == "camera_motion_below_threshold"
    assert frame.metadata["detector_cached"] is True


def test_detector_runtime_reuses_cached_output_for_small_camera_motion() -> None:
    class _Model:
        def __init__(self, owner: "_Runtime") -> None:
            self.owner = owner

        def predict(self, **_: object) -> list[object]:
            self.owner.predict_count += 1
            return [object()]

    class _Runtime(InProcessDetectorRuntime):
        def __init__(self) -> None:
            super().__init__(enabled=True, model_path="model.pt", max_inference_hz=1000.0)
            self.predict_count = 0
            self.decode_count = 0

        def _ensure_model(self) -> object:
            return _Model(self)

        def _decode_results(self, **_: object) -> list[dict[str, object]]:
            self.decode_count += 1
            return [{"class_name": "chair", "bbox_xyxy": [1, 2, 3, 4], "confidence": 0.9}]

    runtime = _Runtime()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)

    first = runtime.detect(
        rgb=rgb,
        depth=depth,
        intrinsic=np.eye(3, dtype=np.float32),
        camera_pos_w=np.zeros(3, dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
    )
    second = runtime.detect(
        rgb=rgb,
        depth=depth,
        intrinsic=np.eye(3, dtype=np.float32),
        camera_pos_w=np.asarray((0.05, 0.0, 0.0), dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
    )

    assert first.cached is False
    assert second.cached is True
    assert second.cache_reason == "camera_motion_below_threshold"
    assert runtime.predict_count == 1
    assert runtime.decode_count == 1


def test_detector_runtime_refreshes_cache_after_cache_age_limit() -> None:
    class _Model:
        def __init__(self, owner: "_Runtime") -> None:
            self.owner = owner

        def predict(self, **_: object) -> list[object]:
            self.owner.predict_count += 1
            return [object()]

    class _Runtime(InProcessDetectorRuntime):
        def __init__(self) -> None:
            super().__init__(enabled=True, model_path="model.pt", max_inference_hz=1000.0)
            self.predict_count = 0

        def _ensure_model(self) -> object:
            return _Model(self)

        def _decode_results(self, **_: object) -> list[dict[str, object]]:
            return [{"class_name": "chair", "bbox_xyxy": [1, 2, 3, 4], "confidence": 0.9}]

    runtime = _Runtime()
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.ones((8, 8), dtype=np.float32)
    kwargs = {
        "rgb": rgb,
        "depth": depth,
        "intrinsic": np.eye(3, dtype=np.float32),
        "camera_pos_w": np.zeros(3, dtype=np.float32),
        "camera_rot_w": np.eye(3, dtype=np.float32),
    }

    runtime.detect(**kwargs)
    runtime._last_inference_monotonic -= 3.0
    refreshed = runtime.detect(**kwargs)

    assert refreshed.cached is False
    assert runtime.predict_count == 2
