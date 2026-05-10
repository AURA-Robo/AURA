"""Publish live control-runtime frames onto the shared viewer transport."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from systems.shared.contracts.viewer_transport import (
    VIEWER_CONTROL_ENDPOINT,
    VIEWER_HEALTH_TOPIC,
    VIEWER_OBSERVATION_TOPIC,
    VIEWER_SHM_CAPACITY,
    VIEWER_SHM_NAME,
    VIEWER_SHM_SLOT_SIZE,
    VIEWER_TELEMETRY_ENDPOINT,
)
from systems.transport import FrameHeader, HealthPing, SharedMemoryRing, ZmqBus, encode_ndarray, ref_to_dict


class ViewerFramePublisher:
    """Bridge camera frames into the backend-owned WebRTC transport."""

    def __init__(self) -> None:
        self._bus = None
        self._shm = None
        try:
            self._bus = ZmqBus(
                control_endpoint=VIEWER_CONTROL_ENDPOINT,
                telemetry_endpoint=VIEWER_TELEMETRY_ENDPOINT,
                role="bridge",
            )
            self._shm = SharedMemoryRing(
                name=VIEWER_SHM_NAME,
                slot_size=VIEWER_SHM_SLOT_SIZE,
                capacity=VIEWER_SHM_CAPACITY,
                create=True,
            )
        except Exception:
            if self._shm is not None:
                self._shm.close(unlink=True)
                self._shm = None
            if self._bus is not None:
                self._bus.close()
                self._bus = None
            raise
        self._sequence = 0
        self._last_health_ns = 0

    def publish_frame(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        source: str,
        frame_stamp_s: float,
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
        robot_pose_xyz: np.ndarray,
        robot_yaw_rad: float,
        intrinsic: np.ndarray | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        self._sequence += 1
        rgb_ref = self._shm.write(encode_ndarray(np.asarray(rgb, dtype=np.uint8)))
        frame_metadata: dict[str, Any] = dict(metadata or {})
        frame_metadata["rgb_ref"] = ref_to_dict(rgb_ref)
        if intrinsic is not None:
            frame_metadata["camera_intrinsic"] = np.asarray(intrinsic, dtype=np.float32).tolist()

        depth_payload = None if depth is None else np.asarray(depth, dtype=np.float32)
        depth_available = depth_payload is not None and depth_payload.size > 0
        if depth_available:
            depth_ref = self._shm.write(encode_ndarray(depth_payload))
            frame_metadata["depth_ref"] = ref_to_dict(depth_ref)

        height = int(rgb.shape[0]) if rgb.ndim >= 2 else 0
        width = int(rgb.shape[1]) if rgb.ndim >= 2 else 0
        header = FrameHeader(
            frame_id=int(self._sequence),
            timestamp_ns=time.time_ns(),
            source=str(source),
            width=width,
            height=height,
            rgb_encoding="rgb8",
            depth_encoding="32FC1" if depth_available else "",
            camera_pose_xyz=tuple(float(value) for value in np.asarray(camera_pos_w, dtype=np.float32)[:3]),
            camera_quat_wxyz=self._camera_quaternion_wxyz(np.asarray(camera_rot_w, dtype=np.float32)),
            robot_pose_xyz=tuple(float(value) for value in np.asarray(robot_pose_xyz, dtype=np.float32)[:3]),
            robot_yaw_rad=float(robot_yaw_rad),
            sim_time_s=float(frame_stamp_s),
            metadata=frame_metadata,
        )
        self._bus.publish(VIEWER_OBSERVATION_TOPIC, header)
        self._maybe_publish_health(
            frame_id=int(self._sequence),
            frame_stamp_s=float(frame_stamp_s),
            width=width,
            height=height,
            depth_available=bool(depth_available),
        )
        return {
            "frameId": int(self._sequence),
            "width": width,
            "height": height,
            "depthAvailable": bool(depth_available),
        }

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close(unlink=True)
            self._shm = None
        if self._bus is not None:
            self._bus.close()
            self._bus = None

    def _maybe_publish_health(self, *, frame_id: int, frame_stamp_s: float, width: int, height: int, depth_available: bool) -> None:
        now_ns = time.time_ns()
        if (now_ns - self._last_health_ns) < 1_000_000_000:
            return
        self._last_health_ns = now_ns
        self._bus.publish(
            VIEWER_HEALTH_TOPIC,
            HealthPing(
                component="aura_runtime",
                status="alive",
                details={
                    "viewer": {
                        "frameId": int(frame_id),
                        "frameStampS": float(frame_stamp_s),
                        "width": int(width),
                        "height": int(height),
                        "depthAvailable": bool(depth_available),
                        "controlEndpoint": VIEWER_CONTROL_ENDPOINT,
                        "telemetryEndpoint": VIEWER_TELEMETRY_ENDPOINT,
                        "shmName": self._shm.name,
                    }
                },
            ),
        )

    @staticmethod
    def _camera_quaternion_wxyz(rotation_matrix: np.ndarray) -> tuple[float, float, float, float]:
        matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (matrix[2, 1] - matrix[1, 2]) * s
            y = (matrix[0, 2] - matrix[2, 0]) * s
            z = (matrix[1, 0] - matrix[0, 1]) * s
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-12))
            w = (matrix[2, 1] - matrix[1, 2]) / s
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
        elif matrix[1, 1] > matrix[2, 2]:
            s = 2.0 * np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-12))
            w = (matrix[0, 2] - matrix[2, 0]) / s
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-12))
            w = (matrix[1, 0] - matrix[0, 1]) / s
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
        return (float(w), float(x), float(y), float(z))


__all__ = ["ViewerFramePublisher"]
