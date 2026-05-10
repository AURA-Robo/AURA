"""HTTP client for the standalone navigation system."""

from __future__ import annotations

from typing import Any

import numpy as np

from systems.shared.contracts.navigation_transport import (
    NAVIGATION_SHM_CAPACITY,
    NAVIGATION_SHM_NAME,
    NAVIGATION_SHM_SLOT_SIZE,
)
from systems.transport import SharedMemoryRing, encode_ndarray, ref_to_dict


def _require_requests():
    try:
        import requests
    except Exception as exc:  # pragma: no cover - runtime dependency validation
        raise RuntimeError("Navigation system client requires requests.") from exc
    return requests


class NavigationSystemClient:
    """Typed client for the navigation-system service."""

    def __init__(self, server_url: str, timeout_s: float = 5.0):
        self.server_url = str(server_url).rstrip("/")
        self.timeout_s = max(0.1, float(timeout_s))
        self._frame_id = 0
        self._shm_owner = False
        try:
            self._shm = SharedMemoryRing(
                name=NAVIGATION_SHM_NAME,
                slot_size=NAVIGATION_SHM_SLOT_SIZE,
                capacity=NAVIGATION_SHM_CAPACITY,
                create=True,
            )
            self._shm_owner = True
        except FileExistsError:
            self._shm = SharedMemoryRing(
                name=NAVIGATION_SHM_NAME,
                slot_size=NAVIGATION_SHM_SLOT_SIZE,
                capacity=NAVIGATION_SHM_CAPACITY,
                create=False,
            )

    def close(self) -> None:
        try:
            self._shm.close(unlink=self._shm_owner)
        except FileNotFoundError:
            return

    def _request(self, method: str, path: str, *, json_payload: dict[str, object] | None = None) -> dict[str, Any]:
        requests = _require_requests()
        response = requests.request(
            method=method,
            url=f"{self.server_url}{path}",
            timeout=self.timeout_s,
            json=json_payload,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            detail = response.text.strip()
            if detail:
                raise RuntimeError(f"Navigation system returned HTTP {response.status_code}: {detail}") from exc
            raise
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected JSON object from navigation system, got {type(payload)!r}.")
        return payload

    def command(self, instruction: str, language: str = "en", *, task_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, object] = {
            "instruction": str(instruction).strip(),
            "language": str(language).strip() or "en",
        }
        if task_id:
            payload["task_id"] = str(task_id)
        return self._request("POST", "/navigation/command", json_payload=payload)

    def command_memory_target(
        self,
        target: dict[str, object],
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, object] = {
            "mode": "memory_pose",
            "target": dict(target),
        }
        if task_id:
            payload["task_id"] = str(task_id)
        return self._request("POST", "/navigation/command", json_payload=payload)

    def command_return_pose(
        self,
        origin_pose: dict[str, object],
        *,
        task_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, object] = {
            "mode": "return_pose",
            "target": dict(origin_pose),
        }
        if task_id:
            payload["task_id"] = str(task_id)
        return self._request("POST", "/navigation/command", json_payload=payload)

    def cancel(self) -> dict[str, Any]:
        return self._request("POST", "/navigation/cancel", json_payload={})

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/navigation/status")

    def trajectory(self) -> dict[str, Any]:
        return self._request("GET", "/navigation/trajectory")

    def update(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
        base_pos_w: np.ndarray,
        base_yaw: float,
        lin_vel_b: np.ndarray,
        yaw_rate: float,
        stamp_s: float,
        detections: list[dict[str, object]] | None = None,
        system2_rgb_history: np.ndarray | None = None,
        navdp_rgb_history: np.ndarray | None = None,
    ) -> dict[str, Any]:
        del system2_rgb_history, navdp_rgb_history
        self._frame_id += 1
        rgb_ref = self._shm.write(encode_ndarray(np.asarray(rgb)))
        depth_ref = self._shm.write(encode_ndarray(np.asarray(depth)))
        payload = {
            "frame_id": self._frame_id,
            "rgb_ref": ref_to_dict(rgb_ref),
            "depth_ref": ref_to_dict(depth_ref),
            "intrinsic": np.asarray(intrinsic, dtype=np.float32).tolist(),
            "camera_pos_w": np.asarray(camera_pos_w, dtype=np.float32).tolist(),
            "camera_rot_w": np.asarray(camera_rot_w, dtype=np.float32).tolist(),
            "robot_state": {
                "base_pos_w": np.asarray(base_pos_w, dtype=np.float32).tolist(),
                "base_yaw": float(base_yaw),
                "lin_vel_b": np.asarray(lin_vel_b, dtype=np.float32).tolist(),
                "yaw_rate": float(yaw_rate),
            },
            "stamp_s": float(stamp_s),
        }
        if detections is not None:
            payload["detections"] = list(detections)
        return self._request("POST", "/navigation/update", json_payload=payload)


def format_navigation_instruction(target: dict[str, Any], *, language: str = "en") -> str:
    object_name = str(target.get("object") or target.get("location_hint") or "target").strip()
    label = object_name.replace("_", " ")
    if str(language).strip().lower().startswith("ko"):
        return f"{label}(으)로 이동하세요."
    return f"Go to the {label}."
