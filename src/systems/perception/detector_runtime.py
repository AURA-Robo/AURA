"""Lazy in-process detector runtime for perception observations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np


def _as_float(value: object) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, np.ndarray):
        if value.shape != ():
            return None
        value = value.item()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if np.isfinite(numeric):
        return numeric
    return None


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            result = tolist()
        except Exception:
            result = None
        if isinstance(result, list):
            return result
    try:
        result = np.asarray(value).tolist()
    except Exception:
        return []
    return result if isinstance(result, list) else []


def _normalize_bbox_xyxy(bbox: object, *, width: int, height: int) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    coords = [_as_float(item) for item in bbox[:4]]
    if any(item is None for item in coords):
        return None
    x1, y1, x2, y2 = (float(item) for item in coords if item is not None)
    if x2 <= x1 or y2 <= y1:
        return None
    max_x = max(width - 1, 0)
    max_y = max(height - 1, 0)
    normalized = [
        int(round(min(max(x1, 0.0), float(max_x)))),
        int(round(min(max(y1, 0.0), float(max_y)))),
        int(round(min(max(x2, 0.0), float(max_x)))),
        int(round(min(max(y2, 0.0), float(max_y)))),
    ]
    if normalized[2] <= normalized[0] or normalized[3] <= normalized[1]:
        return None
    return normalized


def _normalize_track_id(value: object) -> str | None:
    numeric = _as_float(value)
    if numeric is None:
        text = str(value or "").strip()
        return text or None
    rounded = round(numeric)
    if abs(numeric - rounded) <= 1e-6:
        return str(int(rounded))
    return str(numeric)


def _class_name(names: object, class_id: object) -> str:
    index = _as_float(class_id)
    if index is None:
        return ""
    class_index = int(round(index))
    if isinstance(names, dict):
        label = names.get(class_index)
        return str(label).strip() if label is not None else ""
    if isinstance(names, list) and 0 <= class_index < len(names):
        return str(names[class_index]).strip()
    return ""


def _estimate_depth_m(depth: np.ndarray, bbox_xyxy: list[int]) -> float | None:
    depth_image = np.asarray(depth, dtype=np.float32)
    if depth_image.ndim != 2 or depth_image.size == 0:
        return None
    height, width = int(depth_image.shape[0]), int(depth_image.shape[1])
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    region = depth_image[y1 : y2 + 1, x1 : x2 + 1]
    finite = region[np.isfinite(region) & (region > 0.0)]
    if finite.size == 0:
        center_x = int(round((x1 + x2) * 0.5))
        center_y = int(round((y1 + y2) * 0.5))
        cx1 = max(center_x - 2, 0)
        cx2 = min(center_x + 2, width - 1)
        cy1 = max(center_y - 2, 0)
        cy2 = min(center_y + 2, height - 1)
        center_region = depth_image[cy1 : cy2 + 1, cx1 : cx2 + 1]
        finite = center_region[np.isfinite(center_region) & (center_region > 0.0)]
    if finite.size == 0:
        return None
    depth_m = float(np.median(finite))
    return depth_m if np.isfinite(depth_m) and depth_m > 0.0 else None


def _world_pose_xyz(
    *,
    bbox_xyxy: list[int],
    depth_m: float,
    intrinsic: np.ndarray,
    camera_pos_w: np.ndarray,
    camera_rot_w: np.ndarray,
) -> list[float] | None:
    try:
        intrinsic_matrix = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
        camera_xyz = np.asarray(camera_pos_w, dtype=np.float32).reshape(3)
        camera_rot = np.asarray(camera_rot_w, dtype=np.float32).reshape(3, 3)
    except Exception:
        return None
    fx = _as_float(intrinsic_matrix[0, 0])
    fy = _as_float(intrinsic_matrix[1, 1])
    cx = _as_float(intrinsic_matrix[0, 2])
    cy = _as_float(intrinsic_matrix[1, 2])
    if fx is None or fy is None or cx is None or cy is None:
        return None
    if abs(fx) < 1e-6 or abs(fy) < 1e-6:
        return None
    center_x = float(bbox_xyxy[0] + bbox_xyxy[2]) * 0.5
    center_y = float(bbox_xyxy[1] + bbox_xyxy[3]) * 0.5
    camera_frame = np.asarray(
        (
            (center_x - cx) * depth_m / fx,
            (center_y - cy) * depth_m / fy,
            depth_m,
        ),
        dtype=np.float32,
    )
    world_xyz = camera_xyz + (camera_rot @ camera_frame)
    if not np.all(np.isfinite(world_xyz)):
        return None
    return [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])]


def _rotation_angle_delta_rad(left: np.ndarray, right: np.ndarray) -> float:
    try:
        left_rot = np.asarray(left, dtype=np.float32).reshape(3, 3)
        right_rot = np.asarray(right, dtype=np.float32).reshape(3, 3)
    except Exception:
        return float("inf")
    relative = left_rot.T @ right_rot
    cosine = (float(np.trace(relative)) - 1.0) * 0.5
    cosine = min(1.0, max(-1.0, cosine))
    return float(np.arccos(cosine))


@dataclass(slots=True, frozen=True)
class DetectorOutput:
    detections: list[dict[str, object]]
    detector_backend: str | None = None
    cached: bool = False
    cache_reason: str | None = None


class InProcessDetectorRuntime:
    """Run an optional Ultralytics detector inside the perception hot path."""

    BACKEND_NAME = "ultralytics-yoloe"

    def __init__(
        self,
        *,
        enabled: bool,
        model_path: str | None,
        max_inference_hz: float = 5.0,
        cache_pose_delta_m: float = 0.20,
        cache_rotation_delta_rad: float = 0.17,
        cache_max_age_sec: float = 2.0,
    ) -> None:
        self._enabled = bool(enabled)
        self._configured_model_path = str(model_path or "").strip()
        self._max_inference_hz = max(float(max_inference_hz), 0.1)
        self._cache_pose_delta_m = max(float(cache_pose_delta_m), 0.0)
        self._cache_rotation_delta_rad = max(float(cache_rotation_delta_rad), 0.0)
        self._cache_max_age_sec = max(float(cache_max_age_sec), 0.0)
        self._model: Any | None = None
        self._ready = False
        self._load_attempted = False
        self._last_error: str | None = None
        self._last_output = DetectorOutput(
            detections=[],
            detector_backend=self.BACKEND_NAME if self._enabled else None,
        )
        self._last_inference_monotonic = 0.0
        self._last_camera_pos_w: np.ndarray | None = None
        self._last_camera_rot_w: np.ndarray | None = None
        self._resolved_model_path = self._resolve_model_path(self._configured_model_path)

    def detect(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
    ) -> DetectorOutput:
        if not self._enabled:
            return self._last_output
        now = time.monotonic()
        if self._should_use_cached_output(
            now=now,
            camera_pos_w=camera_pos_w,
            camera_rot_w=camera_rot_w,
        ):
            return self._cached_output("camera_motion_below_threshold")
        min_interval = 1.0 / self._max_inference_hz
        if self._last_inference_monotonic > 0.0 and (now - self._last_inference_monotonic) < min_interval:
            return self._cached_output("rate_limited")
        model = self._ensure_model()
        if model is None:
            return self._last_output
        try:
            bgr = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[:, :, ::-1])
            results = model.predict(source=bgr, verbose=False)
            detections = self._decode_results(
                results=results,
                model=model,
                rgb=rgb,
                depth=depth,
                intrinsic=intrinsic,
                camera_pos_w=camera_pos_w,
                camera_rot_w=camera_rot_w,
            )
        except Exception as exc:  # noqa: BLE001
            self._disable(exc)
            return self._last_output
        self._last_inference_monotonic = now
        self._last_camera_pos_w = np.asarray(camera_pos_w, dtype=np.float32).reshape(-1)[:3].copy()
        self._last_camera_rot_w = np.asarray(camera_rot_w, dtype=np.float32).reshape(3, 3).copy()
        self._last_output = DetectorOutput(
            detections=detections,
            detector_backend=self.BACKEND_NAME,
        )
        return self._last_output

    def latest_health(self) -> dict[str, object]:
        return {
            "enabled": bool(self._enabled),
            "ready": bool(self._ready),
            "backend": self.BACKEND_NAME if self._configured_model_path else "inactive",
            "modelPath": str(self._resolved_model_path) if self._resolved_model_path is not None else self._configured_model_path,
            "lastError": self._last_error,
        }

    def close(self) -> None:
        self._model = None

    def _cached_output(self, reason: str) -> DetectorOutput:
        return DetectorOutput(
            detections=[dict(row) for row in self._last_output.detections],
            detector_backend=self._last_output.detector_backend,
            cached=True,
            cache_reason=reason,
        )

    def _should_use_cached_output(
        self,
        *,
        now: float,
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
    ) -> bool:
        if self._last_inference_monotonic <= 0.0:
            return False
        if (now - self._last_inference_monotonic) > self._cache_max_age_sec:
            return False
        if self._last_camera_pos_w is None or self._last_camera_rot_w is None:
            return False
        try:
            current_pos = np.asarray(camera_pos_w, dtype=np.float32).reshape(-1)[:3]
            current_rot = np.asarray(camera_rot_w, dtype=np.float32).reshape(3, 3)
        except Exception:
            return False
        if current_pos.shape[0] < 3:
            return False
        translation_delta = float(np.linalg.norm(current_pos - self._last_camera_pos_w))
        rotation_delta = _rotation_angle_delta_rad(self._last_camera_rot_w, current_rot)
        return (
            translation_delta < self._cache_pose_delta_m
            and rotation_delta < self._cache_rotation_delta_rad
        )

    def _ensure_model(self) -> Any | None:
        if self._model is not None:
            return self._model
        if not self._enabled or self._load_attempted:
            return None
        self._load_attempted = True
        if self._resolved_model_path is None or not self._resolved_model_path.is_file():
            self._disable(FileNotFoundError(f"Detector model not found: {self._configured_model_path or '<empty>'}"))
            return None
        try:
            from ultralytics import YOLO
        except Exception as exc:  # noqa: BLE001
            self._disable(exc)
            return None
        try:
            self._model = YOLO(str(self._resolved_model_path))
        except Exception as exc:  # noqa: BLE001
            self._disable(exc)
            return None
        self._ready = True
        self._last_error = None
        return self._model

    def _decode_results(
        self,
        *,
        results: object,
        model: Any,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
    ) -> list[dict[str, object]]:
        result_rows = _as_list(results)
        if not result_rows:
            return []
        result = result_rows[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        width = int(np.asarray(rgb).shape[1]) if np.asarray(rgb).ndim >= 2 else 0
        height = int(np.asarray(rgb).shape[0]) if np.asarray(rgb).ndim >= 2 else 0
        xyxy_rows = _as_list(getattr(boxes, "xyxy", None))
        conf_rows = _as_list(getattr(boxes, "conf", None))
        cls_rows = _as_list(getattr(boxes, "cls", None))
        track_rows = _as_list(getattr(boxes, "id", None))
        names = getattr(result, "names", None) or getattr(model, "names", None) or {}
        detections: list[dict[str, object]] = []
        for index, bbox in enumerate(xyxy_rows):
            bbox_xyxy = _normalize_bbox_xyxy(bbox, width=width, height=height)
            if bbox_xyxy is None:
                continue
            class_value = cls_rows[index] if index < len(cls_rows) else None
            class_name = _class_name(names, class_value)
            if not class_name:
                continue
            row: dict[str, object] = {
                "class_name": class_name,
                "bbox_xyxy": bbox_xyxy,
            }
            class_id = _as_float(class_value)
            if class_id is not None:
                row["class_id"] = int(round(class_id))
            confidence = _as_float(conf_rows[index] if index < len(conf_rows) else None)
            if confidence is not None:
                row["confidence"] = confidence
            if index < len(track_rows):
                track_id = _normalize_track_id(track_rows[index])
                if track_id:
                    row["track_id"] = track_id
            depth_m = _estimate_depth_m(np.asarray(depth, dtype=np.float32), bbox_xyxy)
            if depth_m is not None:
                row["depth_m"] = depth_m
                pose = _world_pose_xyz(
                    bbox_xyxy=bbox_xyxy,
                    depth_m=depth_m,
                    intrinsic=intrinsic,
                    camera_pos_w=camera_pos_w,
                    camera_rot_w=camera_rot_w,
                )
                if pose is not None:
                    row["world_pose_xyz"] = pose
            detections.append(row)
        return detections

    def _disable(self, exc: Exception) -> None:
        self._enabled = False
        self._ready = False
        self._model = None
        self._last_error = f"{type(exc).__name__}: {exc}"
        self._last_output = DetectorOutput(detections=[], detector_backend=self.BACKEND_NAME)

    @staticmethod
    def _resolve_model_path(model_path: str) -> Path | None:
        normalized = str(model_path or "").strip()
        if not normalized:
            return None
        path = Path(normalized).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        return path


__all__ = ["DetectorOutput", "InProcessDetectorRuntime"]
