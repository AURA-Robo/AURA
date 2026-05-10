"""Observation normalization owned by the perception subsystem."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from systems.shared.contracts.observation import ObservationFrame, RawObservation

from .detector_runtime import InProcessDetectorRuntime
from .telemetry import ViewerFramePublisher


def _normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWxC RGB image, got shape {image.shape}.")
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"Expected HxW depth image, got shape {image.shape}.")
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(image)


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
    return None


def _normalize_detection_list(detections: object) -> list[dict[str, object]]:
    if not isinstance(detections, list):
        return []
    normalized_rows: list[dict[str, object]] = []
    for item in detections:
        if not isinstance(item, dict):
            continue
        class_name = str(item.get("class_name") or "").strip()
        if not class_name:
            continue
        row: dict[str, object] = {"class_name": class_name}
        class_id = _as_float(item.get("class_id"))
        if class_id is not None:
            row["class_id"] = int(round(class_id))
        track_id = str(item.get("track_id") or "").strip()
        if track_id:
            row["track_id"] = track_id
        bbox_xyxy = item.get("bbox_xyxy")
        if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
            coords = [_as_float(value) for value in bbox_xyxy[:4]]
            if all(value is not None for value in coords):
                row["bbox_xyxy"] = [int(round(float(value))) for value in coords if value is not None]
        for key in ("confidence", "depth_m"):
            numeric = _as_float(item.get(key))
            if numeric is not None:
                row[key] = numeric
        world_pose_xyz = item.get("world_pose_xyz")
        if isinstance(world_pose_xyz, (list, tuple)) and len(world_pose_xyz) >= 3:
            coords = [_as_float(value) for value in world_pose_xyz[:3]]
            if all(value is not None for value in coords):
                row["world_pose_xyz"] = [float(value) for value in coords if value is not None]
        normalized_rows.append(row)
    return normalized_rows


def _metadata_has_detections(metadata: dict[str, Any]) -> bool:
    overlay = metadata.get("viewer_overlay")
    if isinstance(overlay, dict) and isinstance(overlay.get("detections"), list) and len(overlay.get("detections", [])) > 0:
        return True
    return isinstance(metadata.get("detections"), list) and len(metadata.get("detections", [])) > 0


def _merge_detection_metadata(
    metadata: dict[str, Any],
    *,
    detections: list[dict[str, object]] | None = None,
    detector_backend: str | None = None,
    detector_cached: bool | None = None,
    detector_cache_reason: str | None = None,
) -> dict[str, Any]:
    payload = dict(metadata)
    overlay = payload.get("viewer_overlay")
    overlay_payload = dict(overlay) if isinstance(overlay, dict) else {}

    normalized_detections: list[dict[str, object]] | None = None
    if detections is not None:
        normalized_detections = _normalize_detection_list(detections)
    else:
        overlay_detections = overlay_payload.get("detections")
        if isinstance(overlay_detections, list):
            normalized_detections = _normalize_detection_list(overlay_detections)
        elif isinstance(payload.get("detections"), list):
            normalized_detections = _normalize_detection_list(payload.get("detections"))

    if normalized_detections is not None:
        payload["detections"] = normalized_detections
        overlay_payload["detections"] = list(normalized_detections)

    backend = detector_backend
    if not isinstance(backend, str) or not backend.strip():
        overlay_backend = overlay_payload.get("detector_backend")
        if isinstance(overlay_backend, str) and overlay_backend.strip():
            backend = overlay_backend.strip()
        else:
            payload_backend = payload.get("detector_backend")
            if isinstance(payload_backend, str) and payload_backend.strip():
                backend = payload_backend.strip()
    if isinstance(backend, str) and backend.strip():
        payload["detector_backend"] = backend.strip()
        overlay_payload["detector_backend"] = backend.strip()

    if detector_cached is not None:
        payload["detector_cached"] = bool(detector_cached)
        overlay_payload["detector_cached"] = bool(detector_cached)
        if detector_cache_reason:
            payload["detector_cache_reason"] = str(detector_cache_reason)
            overlay_payload["detector_cache_reason"] = str(detector_cache_reason)

    if overlay_payload:
        payload["viewer_overlay"] = overlay_payload
    return payload


class PerceptionObservationService:
    """Normalize raw Isaac captures and publish viewer-compatible frames."""

    def __init__(
        self,
        *,
        viewer_publisher: ViewerFramePublisher | None = None,
        detector_runtime: InProcessDetectorRuntime | None = None,
    ):
        self._viewer_publisher = viewer_publisher
        self._detector_runtime = detector_runtime
        self._latest_health: dict[str, Any] = {
            "status": "idle",
            "frame_id": None,
            "last_error": None,
            "last_stamp_s": None,
        }

    def ingest(self, raw: RawObservation) -> ObservationFrame:
        metadata = _merge_detection_metadata(dict(raw.metadata))
        frame = ObservationFrame(
            rgb=_normalize_rgb(raw.rgb),
            depth=_normalize_depth(raw.depth),
            intrinsic=np.asarray(raw.intrinsic, dtype=np.float32).copy(),
            camera_pos_w=np.asarray(raw.camera_pos_w, dtype=np.float32).copy(),
            camera_rot_w=np.asarray(raw.camera_rot_w, dtype=np.float32).copy(),
            robot_state=raw.robot_state,
            stamp_s=float(raw.stamp_s),
            metadata=metadata,
        )
        if self._detector_runtime is not None and not _metadata_has_detections(frame.metadata):
            detector_output = self._detector_runtime.detect(
                rgb=frame.rgb,
                depth=frame.depth,
                intrinsic=frame.intrinsic,
                camera_pos_w=frame.camera_pos_w,
                camera_rot_w=frame.camera_rot_w,
            )
            frame.metadata = _merge_detection_metadata(
                frame.metadata,
                detections=detector_output.detections,
                detector_backend=detector_output.detector_backend,
                detector_cached=detector_output.cached,
                detector_cache_reason=detector_output.cache_reason,
            )
        if self._viewer_publisher is not None:
            robot_pose_xyz = np.asarray(raw.robot_state.base_pos_w, dtype=np.float32)
            self._latest_health = {
                "status": "running",
                "last_error": None,
                "last_stamp_s": frame.stamp_s,
                **self._viewer_publisher.publish_frame(
                    rgb=frame.rgb,
                    depth=frame.depth,
                    source="perception_runtime",
                    frame_stamp_s=frame.stamp_s,
                    camera_pos_w=frame.camera_pos_w,
                    camera_rot_w=frame.camera_rot_w,
                    robot_pose_xyz=robot_pose_xyz,
                    robot_yaw_rad=float(raw.robot_state.base_yaw),
                    intrinsic=frame.intrinsic,
                    metadata=frame.metadata,
                ),
            }
        else:
            self._latest_health = {
                "status": "running",
                "frame_id": None,
                "last_error": None,
                "last_stamp_s": frame.stamp_s,
            }
        return frame

    def latest_health(self) -> dict[str, Any]:
        payload = dict(self._latest_health)
        if self._detector_runtime is not None:
            payload["detector"] = self._detector_runtime.latest_health()
        last_stamp_s = payload.get("last_stamp_s")
        if isinstance(last_stamp_s, (int, float)):
            payload["frame_age_ms"] = max(0.0, (time.monotonic() - float(last_stamp_s)) * 1000.0)
        return payload

    def close(self) -> None:
        if self._detector_runtime is not None:
            self._detector_runtime.close()
            self._detector_runtime = None
        if self._viewer_publisher is not None:
            self._viewer_publisher.close()
            self._viewer_publisher = None
