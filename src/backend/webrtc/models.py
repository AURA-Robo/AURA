"""Viewer message helpers shared by backend WebRTC components."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from systems.transport import FrameHeader


@dataclass(frozen=True)
class FrameCache:
    seq: int
    frame_header: FrameHeader
    rgb_image: np.ndarray
    depth_image_m: np.ndarray | None
    viewer_overlay: dict[str, object]
    last_frame_monotonic: float


def frame_age_ms(frame: FrameCache | None, *, now: float | None = None) -> float | None:
    if frame is None:
        return None
    current = time.monotonic() if now is None else float(now)
    return max((current - float(frame.last_frame_monotonic)) * 1000.0, 0.0)


def is_frame_stale(frame: FrameCache | None, *, stale_after_sec: float, now: float | None = None) -> bool:
    if frame is None:
        return True
    current = time.monotonic() if now is None else float(now)
    return (current - float(frame.last_frame_monotonic)) > max(float(stale_after_sec), 0.0)


def build_session_ready_message(
    *,
    session_id: str,
    track_roles: list[str],
    channel_labels: tuple[str, str],
) -> dict[str, object]:
    return {
        "type": "session_ready",
        "sessionId": str(session_id),
        "observeOnly": True,
        "trackRoles": list(track_roles),
        "channelLabels": list(channel_labels),
    }


def build_waiting_for_frame_message(*, age_ms: float | None, has_seen_frame: bool) -> dict[str, object]:
    return {
        "type": "waiting_for_frame",
        "age_ms": None if age_ms is None else round(float(age_ms), 3),
        "has_seen_frame": bool(has_seen_frame),
    }


def build_frame_state_message(frame: FrameCache) -> dict[str, object]:
    overlay = frame.viewer_overlay if isinstance(frame.viewer_overlay, dict) else {}
    payload: dict[str, object] = {
        "type": "frame_state",
        "seq": int(frame.seq),
        "frame_id": int(frame.frame_header.frame_id),
        "source": str(frame.frame_header.source),
        "image": {
            "width": int(frame.frame_header.width),
            "height": int(frame.frame_header.height),
            "rgbEncoding": str(frame.frame_header.rgb_encoding),
        },
        "robot_pose_xyz": [float(value) for value in frame.frame_header.robot_pose_xyz[:3]],
        "robot_yaw_rad": float(frame.frame_header.robot_yaw_rad),
        "sim_time_s": float(frame.frame_header.sim_time_s),
        "detector_backend": str(overlay.get("detector_backend", "")),
        "detection_count": len(overlay.get("detections", []))
        if isinstance(overlay.get("detections"), list)
        else 0,
        "has_depth": frame.depth_image_m is not None,
    }
    active_target = overlay.get("active_target", overlay.get("activeTarget", {}))
    if isinstance(active_target, dict) and active_target:
        payload["active_target"] = dict(active_target)
        payload["activeTarget"] = dict(active_target)
    return payload


def build_frame_meta_message(frame: FrameCache) -> dict[str, object]:
    overlay = frame.viewer_overlay if isinstance(frame.viewer_overlay, dict) else {}
    detections = overlay.get("detections", [])
    compact_detections: list[dict[str, object]] = []
    if isinstance(detections, list):
        for item in detections:
            if not isinstance(item, dict):
                continue
            compact: dict[str, object] = {}
            class_id = item.get("class_id")
            if isinstance(class_id, (int, float)) and not isinstance(class_id, bool):
                compact["class_id"] = int(round(float(class_id)))
            for key in ("class_name", "track_id"):
                value = item.get(key)
                if isinstance(value, str) and value != "":
                    compact[key] = value
            bbox = item.get("bbox_xyxy")
            if isinstance(bbox, list) and len(bbox) == 4:
                compact["bbox_xyxy"] = [int(value) for value in bbox]
            for key in ("confidence", "depth_m", "approach_yaw_rad"):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    compact[key] = float(value)
            for key in ("memory_match_score",):
                value = item.get(key)
                if isinstance(value, (int, float)):
                    compact[key] = float(value)
            for key in ("memory_object_id", "memory_status", "memory_observation_id", "memory_filter_reason"):
                value = item.get(key)
                if isinstance(value, str) and value != "":
                    compact[key] = value
            memory_persisted = item.get("memory_persisted")
            if isinstance(memory_persisted, bool):
                compact["memory_persisted"] = memory_persisted
            world_pose = item.get("world_pose_xyz")
            if isinstance(world_pose, list) and len(world_pose) >= 3:
                compact["world_pose_xyz"] = [float(world_pose[0]), float(world_pose[1]), float(world_pose[2])]
            compact_detections.append(compact)

    trajectory_pixels = overlay.get("trajectory_pixels", overlay.get("trajectoryPixels", []))
    compact_trajectory: list[list[int]] = []
    if isinstance(trajectory_pixels, list):
        for point in trajectory_pixels:
            if isinstance(point, list) and len(point) == 2:
                compact_trajectory.append([int(point[0]), int(point[1])])

    system2_pixel_goal = overlay.get("system2_pixel_goal", overlay.get("system2PixelGoal"))
    compact_system2_pixel_goal: list[int] | None = None
    if isinstance(system2_pixel_goal, list) and len(system2_pixel_goal) >= 2:
        compact_system2_pixel_goal = [int(system2_pixel_goal[0]), int(system2_pixel_goal[1])]

    payload: dict[str, object] = {
        "type": "frame_meta",
        "seq": int(frame.seq),
        "frame_id": int(frame.frame_header.frame_id),
        "timestamp_ns": int(frame.frame_header.timestamp_ns),
        "source": str(frame.frame_header.source),
        "robot_pose_xyz": [float(value) for value in frame.frame_header.robot_pose_xyz[:3]],
        "robot_yaw_rad": float(frame.frame_header.robot_yaw_rad),
        "sim_time_s": float(frame.frame_header.sim_time_s),
        "detections": compact_detections,
        "trajectory_pixels": compact_trajectory,
        "trajectoryPixels": compact_trajectory,
    }
    if compact_system2_pixel_goal is not None:
        payload["system2_pixel_goal"] = compact_system2_pixel_goal
        payload["system2PixelGoal"] = compact_system2_pixel_goal
    active_target = overlay.get("active_target", overlay.get("activeTarget", {}))
    if isinstance(active_target, dict) and active_target:
        payload["active_target"] = dict(active_target)
        payload["activeTarget"] = dict(active_target)
    return payload
