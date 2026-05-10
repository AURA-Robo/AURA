"""Standalone navigation-system service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse

import numpy as np

from systems.inference.api.runtime import (
    ManagedServiceConfig,
    ProcessRegistry,
    REPO_ROOT,
    normalized_uv_to_pixel_xy,
    resolve_goal_world_xy,
)
from systems.navigation.api.runtime import (
    RobotState2D,
    camera_plan_to_world_xy,
    point_goal_body_from_world,
)
from systems.navigation.service_codec import decode_depth_png_base64, decode_rgb_jpeg_base64
from systems.navigation.system1.backends.http_backend import System1HttpBackend as NavDpClient
from systems.navigation.system2.backends.http_backend import System2HttpBackend as InternVlaNavClient
from systems.shared.contracts.navigation_transport import (
    NAVIGATION_SHM_CAPACITY,
    NAVIGATION_SHM_NAME,
    NAVIGATION_SHM_SLOT_SIZE,
)
from systems.transport import SharedMemoryRing, decode_ndarray, ref_from_dict


DIRECT_ACTION_MODES = frozenset(("forward", "yaw_left", "yaw_right"))


@dataclass(slots=True)
class MemoryNavigationTarget:
    object_id: str
    class_name: str
    scene_scope: str | None
    world_pose_xyz: np.ndarray
    pose_age_sec: float
    stop_radius_m: float
    reacquire_radius_m: float
    reacquire_timeout_sec: float


@dataclass(slots=True)
class ReturnPoseTarget:
    world_xy: np.ndarray
    yaw_rad: float | None


@dataclass(slots=True)
class CompactDetection:
    class_name: str
    track_id: str | None
    bbox_xyxy: tuple[float, float, float, float] | None
    confidence: float
    world_pose_xyz: np.ndarray | None


@dataclass(slots=True)
class NavigationCommand:
    instruction: str
    language: str
    task_id: str | None
    session_id: str
    started_at: float
    mode: str = "instruction"
    memory_target: MemoryNavigationTarget | None = None
    return_target: ReturnPoseTarget | None = None


@dataclass(slots=True)
class NavigationObservation:
    frame_id: str
    stamp_s: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    robot_state: RobotState2D
    detections: list[CompactDetection]


def _python_command() -> str:
    return sys.executable or "python"


def _default_backend_log_dir() -> Path:
    return REPO_ROOT / "logs" / "navigation_backends"


def _goal_changed(current: np.ndarray | None, updated: np.ndarray | None) -> bool:
    if current is None and updated is None:
        return False
    if current is None or updated is None:
        return True
    return not np.allclose(np.asarray(current, dtype=np.float32), np.asarray(updated, dtype=np.float32), atol=1.0e-3)


def _direct_action_sequence(result) -> tuple[str, ...]:
    actions = tuple(str(mode) for mode in (result.action_sequence or ()) if str(mode) in DIRECT_ACTION_MODES)
    if actions:
        return actions
    decision_mode = str(result.decision_mode or "").strip()
    if decision_mode in DIRECT_ACTION_MODES:
        return (decision_mode,)
    return ()


def _normalize_scene_scope(value: object) -> str | None:
    normalized = " ".join(str(value or "").strip().split())
    return normalized or None


def _decode_memory_target(payload: object) -> MemoryNavigationTarget:
    if not isinstance(payload, dict):
        raise ValueError("target must be a JSON object")
    object_id = str(payload.get("object_id") or "").strip()
    class_name = str(payload.get("class_name") or "").strip()
    world_pose_xyz = payload.get("world_pose_xyz")
    if not object_id:
        raise ValueError("target.object_id must be a non-empty string")
    if not class_name:
        raise ValueError("target.class_name must be a non-empty string")
    if not isinstance(world_pose_xyz, (list, tuple)) or len(world_pose_xyz) < 3:
        raise ValueError("target.world_pose_xyz must be a 3-element array")
    try:
        xyz = np.asarray(
            (float(world_pose_xyz[0]), float(world_pose_xyz[1]), float(world_pose_xyz[2])),
            dtype=np.float32,
        ).reshape(3)
    except (TypeError, ValueError) as exc:
        raise ValueError("target.world_pose_xyz must contain numeric values") from exc
    return MemoryNavigationTarget(
        object_id=object_id,
        class_name=class_name,
        scene_scope=_normalize_scene_scope(payload.get("scene_scope")),
        world_pose_xyz=xyz,
        pose_age_sec=max(0.0, float(payload.get("pose_age_sec", 0.0) or 0.0)),
        stop_radius_m=max(0.05, float(payload.get("stop_radius_m", 0.8) or 0.8)),
        reacquire_radius_m=max(0.05, float(payload.get("reacquire_radius_m", 1.0) or 1.0)),
        reacquire_timeout_sec=max(0.1, float(payload.get("reacquire_timeout_sec", 4.0) or 4.0)),
    )


def _decode_return_pose_target(payload: object) -> ReturnPoseTarget:
    if not isinstance(payload, dict):
        raise ValueError("target must be a JSON object")
    world_xy = payload.get("world_xy")
    if not isinstance(world_xy, (list, tuple)) or len(world_xy) < 2:
        raise ValueError("target.world_xy must be a 2-element array")
    try:
        xy = np.asarray((float(world_xy[0]), float(world_xy[1])), dtype=np.float32).reshape(2)
    except (TypeError, ValueError) as exc:
        raise ValueError("target.world_xy must contain numeric values") from exc
    yaw_rad = payload.get("yaw_rad")
    if yaw_rad is not None:
        try:
            yaw_value = float(yaw_rad)
        except (TypeError, ValueError) as exc:
            raise ValueError("target.yaw_rad must be numeric when provided") from exc
    else:
        yaw_value = None
    return ReturnPoseTarget(world_xy=xy, yaw_rad=yaw_value)


class NavigationSystem:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._system2 = InternVlaNavClient(server_url=str(args.system2_url), timeout_s=float(args.system2_timeout))
        self._navdp = NavDpClient(
            server_url=str(args.navdp_url),
            timeout_s=float(args.navdp_timeout),
            fallback_mode=str(args.navdp_fallback),
        )
        self._lock = threading.Lock()
        self._command: NavigationCommand | None = None
        self._latest_observation: NavigationObservation | None = None
        self._capture_generation = 0
        self._goal_generation = 0
        self._system2_applied_capture_generation = -1
        self._system1_applied_signature = (-1, -1)
        self._session_reset_required = True
        self._algorithm_name: str | None = None
        self._backend_configs = self._build_backend_configs(args)
        self._backend_registry: ProcessRegistry | None = None
        self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
        self._trajectory_stamp_s = 0.0
        self._last_error: str | None = None
        self._last_system2: dict[str, object] | None = None
        self._system2_stage = self._make_stage_payload("system2", status="inactive")
        self._system1_stage = self._make_stage_payload("navdp", status="inactive")
        self._last_goal_world_xy: np.ndarray | None = None
        self._last_system2_pixel_goal: list[int] | None = None
        self._active_target_summary: dict[str, object] | None = None
        self._action_override_mode: str | None = None
        self._latest_detections: list[CompactDetection] = []
        self._memory_reacquire_state: str | None = None
        self._memory_reacquire_started_at: float | None = None
        self._memory_reacquire_sweep_sent = False
        self._stop_event = threading.Event()
        self._system2_event = threading.Event()
        self._system1_event = threading.Event()
        self._shm_rings: dict[str, SharedMemoryRing] = {}
        if bool(args.backend_autostart):
            self._backend_registry = ProcessRegistry(Path(args.backend_log_dir))
            for config in self._backend_configs.values():
                self._backend_registry.start(config)
        self._system2_thread = threading.Thread(
            target=self._system2_loop,
            name="navigation-system2-loop",
            daemon=True,
        )
        self._system1_thread = threading.Thread(
            target=self._system1_loop,
            name="navigation-system1-loop",
            daemon=True,
        )
        self._system2_thread.start()
        self._system1_thread.start()

    @staticmethod
    def _build_backend_configs(args: argparse.Namespace) -> dict[str, ManagedServiceConfig]:
        python = _python_command()
        return {
            "system2": ManagedServiceConfig(
                name="system2",
                host=str(args.system2_host),
                port=int(args.system2_port),
                health_path="/healthz",
                command=[
                    python,
                    "-m",
                    "systems.inference.system2.server",
                    "--host",
                    str(args.system2_host),
                    "--port",
                    str(int(args.system2_port)),
                    "--llama-url",
                    str(args.system2_llama_url),
                    *(
                        ["--model-path", str(args.system2_model_path)]
                        if str(args.system2_model_path).strip()
                        else []
                    ),
                ],
                required=True,
            ),
            "navdp": ManagedServiceConfig(
                name="navdp",
                host=str(args.navdp_host),
                port=int(args.navdp_port),
                health_path="/healthz",
                command=[
                    python,
                    "-m",
                    "systems.inference.navdp.server",
                    "--port",
                    str(int(args.navdp_port)),
                    "--checkpoint",
                    str(args.navdp_checkpoint),
                    "--device",
                    str(args.navdp_device),
                    "--tensorrt-mode",
                    str(getattr(args, "navdp_tensorrt_mode", "off") or "off"),
                    "--tensorrt-engine-dir",
                    str(getattr(args, "navdp_tensorrt_engine_dir", "") or ""),
                    "--tensorrt-precision",
                    str(getattr(args, "navdp_tensorrt_precision", "fp16") or "fp16"),
                ],
                required=False,
            ),
        }

    def _backend_snapshot(self, name: str) -> dict[str, object] | None:
        registry = self._backend_registry
        if registry is None:
            return None
        for process in registry.snapshot():
            if process.get("name") == name:
                return dict(process)
        return None

    def _make_stage_payload(
        self,
        backend_name: str,
        *,
        status: str,
        latency_ms: float | None = None,
        detail: str | None = None,
        algorithm_name: str | None = None,
        path_points: int | None = None,
    ) -> dict[str, object]:
        config = self._backend_configs.get(backend_name)
        payload: dict[str, object] = {
            "name": backend_name,
            "status": status,
            "mode": "child" if bool(self.args.backend_autostart) else "external",
        }
        if config is not None:
            payload["base_url"] = config.base_url
            payload["health_url"] = config.health_url
        if latency_ms is not None:
            payload["latency_ms"] = float(latency_ms)
        if detail:
            payload["detail"] = detail
        if algorithm_name:
            payload["algorithm_name"] = algorithm_name
        if path_points is not None:
            payload["path_points"] = int(path_points)
        process = self._backend_snapshot(backend_name)
        if process is not None:
            payload["process"] = process
        return payload

    def _clear_runtime_state_locked(self) -> None:
        self._capture_generation = 0
        self._goal_generation = 0
        self._system2_applied_capture_generation = -1
        self._system1_applied_signature = (-1, -1)
        self._session_reset_required = True
        self._algorithm_name = None
        self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
        self._trajectory_stamp_s = 0.0
        self._last_error = None
        self._last_system2 = None
        self._system2_stage = self._make_stage_payload("system2", status="idle")
        self._system1_stage = self._make_stage_payload("navdp", status="idle")
        self._last_goal_world_xy = None
        self._last_system2_pixel_goal = None
        self._active_target_summary = None
        self._action_override_mode = None
        self._latest_detections = []
        self._memory_reacquire_state = None
        self._memory_reacquire_started_at = None
        self._memory_reacquire_sweep_sent = False

    def command(self, instruction: str, language: str, *, task_id: str | None) -> dict[str, object]:
        normalized_instruction = " ".join(str(instruction).strip().split())
        if not normalized_instruction:
            raise ValueError("instruction must be a non-empty string")
        with self._lock:
            self._command = NavigationCommand(
                instruction=normalized_instruction,
                language=str(language).strip() or "en",
                task_id=task_id,
                session_id=f"nav-{time.time_ns()}",
                started_at=time.time(),
                mode="instruction",
            )
            self._clear_runtime_state_locked()
            has_observation = self._latest_observation is not None
        if has_observation:
            self._system2_event.set()
        return self.status_payload()

    def command_memory_target(self, target: MemoryNavigationTarget, *, task_id: str | None) -> dict[str, object]:
        with self._lock:
            self._command = NavigationCommand(
                instruction=f"go to remembered {target.class_name}",
                language="en",
                task_id=task_id,
                session_id=f"nav-{time.time_ns()}",
                started_at=time.time(),
                mode="memory_pose",
                memory_target=target,
            )
            self._clear_runtime_state_locked()
            self._last_system2 = {
                "status": "memory_pose",
                "decision_mode": "memory_pose",
                "text": f"Approach remembered {target.class_name}",
                "latency_ms": None,
                "action_sequence": [],
            }
            self._system2_stage = self._make_stage_payload("system2", status="healthy", detail="memory pose mode")
            self._memory_reacquire_state = "approach_memory_pose"
            self._set_goal_locked(
                np.asarray(target.world_pose_xyz[:2], dtype=np.float32),
                None,
                {
                    "className": target.class_name.replace("_", " "),
                    "source": "object_memory",
                    "object_id": target.object_id,
                    "scene_scope": target.scene_scope,
                    "pose_age_sec": float(target.pose_age_sec),
                    "world_pose_xyz": target.world_pose_xyz.astype(float).tolist(),
                },
            )
            has_observation = self._latest_observation is not None
        if has_observation:
            self._system1_event.set()
        return self.status_payload()

    def command_return_pose(self, target: ReturnPoseTarget, *, task_id: str | None) -> dict[str, object]:
        observation = None
        with self._lock:
            self._command = NavigationCommand(
                instruction="return to origin",
                language="en",
                task_id=task_id,
                session_id=f"nav-{time.time_ns()}",
                started_at=time.time(),
                mode="return_pose",
                return_target=target,
            )
            self._clear_runtime_state_locked()
            self._last_system2 = {
                "status": "return_pose",
                "decision_mode": "return_pose",
                "text": "Returning to origin",
                "latency_ms": None,
                "action_sequence": [],
            }
            self._system2_stage = self._make_stage_payload("system2", status="healthy", detail="return pose mode")
            self._set_goal_locked(
                np.asarray(target.world_xy, dtype=np.float32),
                None,
                {
                    "className": "Start Pose",
                    "source": "return_pose",
                    "world_pose_xyz": [float(target.world_xy[0]), float(target.world_xy[1]), 0.0],
                    "yaw_rad": None if target.yaw_rad is None else float(target.yaw_rad),
                },
            )
            has_observation = self._latest_observation is not None
            observation = self._latest_observation
        if observation is not None:
            self._advance_return_navigation(observation)
        with self._lock:
            has_active_goal = self._last_goal_world_xy is not None
        if has_observation and has_active_goal:
            self._system1_event.set()
        return self.status_payload()

    def cancel(self) -> dict[str, object]:
        with self._lock:
            self._command = None
            self._clear_runtime_state_locked()
        return self.status_payload()

    def update(self, payload: dict[str, object]) -> dict[str, object]:
        observation = self._decode_observation(payload)
        with self._lock:
            self._latest_observation = observation
            self._latest_detections = list(observation.detections)
            self._capture_generation += 1
            command = self._command
            command_mode = None if command is None else command.mode
        if command_mode == "memory_pose":
            self._advance_memory_navigation(observation)
        elif command_mode == "return_pose":
            self._advance_return_navigation(observation)
        with self._lock:
            command_active = self._command is not None
            has_goal = self._last_goal_world_xy is not None
            action_override_mode = self._action_override_mode
        if command_active:
            self._system2_event.set()
            if has_goal and action_override_mode is None:
                self._system1_event.set()
        return self.trajectory_payload()

    def _decode_observation(self, payload: dict[str, object]) -> NavigationObservation:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        rgb = self._decode_array(payload, ref_key="rgb_ref", legacy_key="rgb_jpeg_base64")
        depth = self._decode_array(payload, ref_key="depth_ref", legacy_key="depth_png_base64")
        intrinsic = np.asarray(payload["intrinsic"], dtype=np.float32)
        camera_pos_w = np.asarray(payload["camera_pos_w"], dtype=np.float32)
        camera_rot_w = np.asarray(payload["camera_rot_w"], dtype=np.float32)
        robot_state_payload = payload["robot_state"]
        if not isinstance(robot_state_payload, dict):
            raise ValueError("robot_state must be a JSON object")
        robot_state = RobotState2D(
            base_pos_w=np.asarray(robot_state_payload["base_pos_w"], dtype=np.float32),
            base_yaw=float(robot_state_payload["base_yaw"]),
            lin_vel_b=np.asarray(robot_state_payload.get("lin_vel_b", (0.0, 0.0)), dtype=np.float32),
            yaw_rate=float(robot_state_payload.get("yaw_rate", 0.0)),
        )
        frame_id = str(payload.get("frame_id") or f"frame-{time.time_ns()}")
        stamp_s = float(payload.get("stamp_s", time.monotonic()))
        return NavigationObservation(
            frame_id=frame_id,
            stamp_s=stamp_s,
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            intrinsic=intrinsic,
            camera_pos_w=camera_pos_w,
            camera_rot_w=camera_rot_w,
            robot_state=robot_state,
            detections=self._decode_detections(payload.get("detections")),
        )

    def _decode_array(self, payload: dict[str, object], *, ref_key: str, legacy_key: str) -> np.ndarray:
        ref_payload = payload.get(ref_key)
        if isinstance(ref_payload, dict):
            ring = self._ring_for_name(str(ref_payload.get("name") or NAVIGATION_SHM_NAME))
            return decode_ndarray(ring.read(ref_from_dict(ref_payload)))
        legacy_payload = payload.get(legacy_key)
        if not isinstance(legacy_payload, str) or not legacy_payload:
            raise ValueError(f"{ref_key} or {legacy_key} is required")
        if legacy_key == "rgb_jpeg_base64":
            return decode_rgb_jpeg_base64(legacy_payload)
        return decode_depth_png_base64(legacy_payload)

    def _decode_detections(self, payload: object) -> list[CompactDetection]:
        if not isinstance(payload, list):
            return []
        detections: list[CompactDetection] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            class_name = str(item.get("class_name") or "").strip()
            if not class_name:
                continue
            bbox_xyxy = item.get("bbox_xyxy")
            normalized_bbox = None
            if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
                try:
                    normalized_bbox = (
                        float(bbox_xyxy[0]),
                        float(bbox_xyxy[1]),
                        float(bbox_xyxy[2]),
                        float(bbox_xyxy[3]),
                    )
                except (TypeError, ValueError):
                    normalized_bbox = None
            world_pose_xyz = None
            raw_pose = item.get("world_pose_xyz")
            if isinstance(raw_pose, (list, tuple)) and len(raw_pose) >= 3:
                try:
                    world_pose_xyz = np.asarray(
                        (float(raw_pose[0]), float(raw_pose[1]), float(raw_pose[2])),
                        dtype=np.float32,
                    ).reshape(3)
                except (TypeError, ValueError):
                    world_pose_xyz = None
            detections.append(
                CompactDetection(
                    class_name=class_name,
                    track_id=str(item.get("track_id") or "").strip() or None,
                    bbox_xyxy=normalized_bbox,
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    world_pose_xyz=world_pose_xyz,
                )
            )
        return detections

    def _ring_for_name(self, name: str) -> SharedMemoryRing:
        ring = self._shm_rings.get(name)
        if ring is not None:
            return ring
        ring = SharedMemoryRing(
            name=name,
            slot_size=int(getattr(self.args, "navigation_shm_slot_size", NAVIGATION_SHM_SLOT_SIZE)),
            capacity=int(getattr(self.args, "navigation_shm_capacity", NAVIGATION_SHM_CAPACITY)),
            create=False,
        )
        self._shm_rings[name] = ring
        return ring

    def _system2_loop(self) -> None:
        while not self._stop_event.is_set():
            self._system2_event.wait(timeout=0.1)
            self._system2_event.clear()
            while not self._stop_event.is_set():
                with self._lock:
                    command = self._command
                    observation = self._latest_observation
                    capture_generation = self._capture_generation
                    needs_reset = self._session_reset_required
                    if command is None or observation is None:
                        self._system2_stage = self._make_stage_payload("system2", status="idle")
                        break
                    if command.mode in {"memory_pose", "return_pose"}:
                        self._system2_stage = self._make_stage_payload(
                            "system2",
                            status="healthy",
                            detail="memory pose mode" if command.mode == "memory_pose" else "return pose mode",
                        )
                        break
                    if capture_generation == self._system2_applied_capture_generation and not needs_reset:
                        break
                    self._system2_stage = self._make_stage_payload("system2", status="running")
                try:
                    if needs_reset:
                        self._system2.reset_session(
                            session_id=command.session_id,
                            instruction=command.instruction,
                            language=command.language,
                            image_width=int(observation.rgb.shape[1]),
                            image_height=int(observation.rgb.shape[0]),
                        )
                    started = time.perf_counter()
                    result = self._system2.step_session(
                        session_id=command.session_id,
                        rgb=observation.rgb,
                        depth=observation.depth,
                        stamp_s=observation.stamp_s,
                    )
                    latency_ms = float(getattr(result, "latency_ms", (time.perf_counter() - started) * 1000.0))
                    pixel_goal = self._result_pixel_goal(result, observation)
                    direct_actions = _direct_action_sequence(result)
                    goal_world_xy: np.ndarray | None = None
                    active_target: dict[str, object] | None = None
                    if not direct_actions and result.status == "goal" and result.uv_norm is not None:
                        resolved = resolve_goal_world_xy(
                            uv_norm=result.uv_norm,
                            depth_image=observation.depth,
                            intrinsic=observation.intrinsic,
                            camera_pos_w=observation.camera_pos_w,
                            camera_rot_w=observation.camera_rot_w,
                            window_size=int(self.args.goal_depth_window),
                            depth_min_m=float(self.args.goal_depth_min),
                            depth_max_m=float(self.args.goal_depth_max),
                        )
                        if resolved is None:
                            raise RuntimeError("unable to resolve world goal from grounded pixel")
                        goal_world_xy, resolved_pixel_xy, _depth_m = resolved
                        if pixel_goal is None:
                            pixel_goal = [int(resolved_pixel_xy[0]), int(resolved_pixel_xy[1])]
                        active_target = {
                            "className": "Navigation Goal",
                            "source": "navigation",
                            "nav_goal_pixel": None if pixel_goal is None else list(pixel_goal),
                            "world_pose_xyz": [float(goal_world_xy[0]), float(goal_world_xy[1]), 0.0],
                        }
                    with self._lock:
                        stale = (
                            self._command is None
                            or self._command.session_id != command.session_id
                            or self._latest_observation is None
                            or capture_generation != self._capture_generation
                        )
                        if stale:
                            self._system2_event.set()
                            continue
                        self._session_reset_required = False
                        self._system2_applied_capture_generation = capture_generation
                        self._last_system2 = {
                            "status": result.status,
                            "decision_mode": result.decision_mode,
                            "text": result.text,
                            "latency_ms": latency_ms,
                            "action_sequence": list(direct_actions or (result.action_sequence or ())),
                            "pixel_xy": None if pixel_goal is None else list(pixel_goal),
                            "uv_norm": None
                            if result.uv_norm is None
                            else np.asarray(result.uv_norm, dtype=np.float32).astype(float).tolist(),
                        }
                        self._system2_stage = self._make_stage_payload("system2", status="healthy", latency_ms=latency_ms)
                        self._last_error = None
                        if direct_actions:
                            goal_changed = self._set_goal_locked(None, None, None)
                            self._action_override_mode = direct_actions[0]
                            self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
                            self._trajectory_stamp_s = float(observation.stamp_s)
                            self._system1_stage = self._make_stage_payload(
                                "navdp",
                                status="paused",
                                detail=f"action override: {self._action_override_mode}",
                            )
                            if goal_changed:
                                self._system1_applied_signature = (-1, -1)
                            continue
                        self._action_override_mode = None
                        goal_changed = self._set_goal_locked(goal_world_xy, pixel_goal, active_target)
                        if goal_world_xy is None:
                            self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
                            self._trajectory_stamp_s = float(observation.stamp_s)
                            self._system1_stage = self._make_stage_payload("navdp", status="idle")
                            if goal_changed:
                                self._system1_applied_signature = (-1, -1)
                            continue
                        if goal_changed:
                            self._algorithm_name = None
                            self._system1_applied_signature = (-1, -1)
                        self._system1_event.set()
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        if self._command is None or self._latest_observation is None:
                            continue
                        if capture_generation != self._capture_generation:
                            self._system2_event.set()
                            continue
                        self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
                        self._trajectory_stamp_s = float(observation.stamp_s)
                        self._last_error = f"{type(exc).__name__}: {exc}"
                        self._last_system2 = {
                            "status": "error",
                            "decision_mode": None,
                            "text": str(exc),
                            "latency_ms": None,
                            "action_sequence": [],
                        }
                        self._system2_stage = self._make_stage_payload("system2", status="error", detail=self._last_error)

    def _system1_loop(self) -> None:
        while not self._stop_event.is_set():
            self._system1_event.wait(timeout=0.1)
            self._system1_event.clear()
            while not self._stop_event.is_set():
                with self._lock:
                    command = self._command
                    observation = self._latest_observation
                    goal_world_xy = None if self._last_goal_world_xy is None else np.asarray(self._last_goal_world_xy, dtype=np.float32)
                    capture_generation = self._capture_generation
                    goal_generation = self._goal_generation
                    action_override_mode = self._action_override_mode
                    algorithm_name = self._algorithm_name
                    if command is None or observation is None or goal_world_xy is None:
                        if action_override_mode is None:
                            self._system1_stage = self._make_stage_payload("navdp", status="idle")
                        break
                    if action_override_mode is not None:
                        self._system1_stage = self._make_stage_payload(
                            "navdp",
                            status="paused",
                            detail=f"action override: {action_override_mode}",
                        )
                        break
                    signature = (capture_generation, goal_generation)
                    if signature == self._system1_applied_signature:
                        break
                    robot_state = RobotState2D(
                        base_pos_w=np.asarray(observation.robot_state.base_pos_w, dtype=np.float32),
                        base_yaw=float(observation.robot_state.base_yaw),
                        lin_vel_b=np.asarray(observation.robot_state.lin_vel_b, dtype=np.float32),
                        yaw_rate=float(observation.robot_state.yaw_rate),
                    )
                    self._system1_stage = self._make_stage_payload("navdp", status="running")
                try:
                    if algorithm_name is None:
                        algorithm_name = self._navdp.reset_pointgoal(
                            intrinsic=observation.intrinsic,
                            stop_threshold=float(self.args.navdp_stop_threshold),
                            batch_size=1,
                        )
                    goal_body_xy = point_goal_body_from_world(goal_world_xy, robot_state.base_pos_w, robot_state.base_yaw)
                    started = time.perf_counter()
                    plan = self._navdp.step_pointgoal(goal_body_xy, observation.rgb, observation.depth)
                    latency_ms = float(getattr(plan, "plan_time_s", 0.0)) * 1000.0
                    if latency_ms <= 0.0:
                        latency_ms = (time.perf_counter() - started) * 1000.0
                    world_path = camera_plan_to_world_xy(
                        plan.trajectory_camera,
                        camera_pos_w=observation.camera_pos_w,
                        camera_rot_w=observation.camera_rot_w,
                    )
                    with self._lock:
                        stale = (
                            self._command is None
                            or self._command.session_id != command.session_id
                            or self._latest_observation is None
                            or capture_generation != self._capture_generation
                            or goal_generation != self._goal_generation
                            or self._action_override_mode is not None
                            or self._last_goal_world_xy is None
                        )
                        if stale:
                            self._system1_event.set()
                            continue
                        self._algorithm_name = algorithm_name
                        self._system1_applied_signature = signature
                        self._trajectory_world_xy = np.asarray(world_path, dtype=np.float32).reshape(-1, 2)
                        self._trajectory_stamp_s = float(plan.stamp_s)
                        self._system1_stage = self._make_stage_payload(
                            "navdp",
                            status="healthy",
                            latency_ms=latency_ms,
                            algorithm_name=algorithm_name,
                            path_points=int(self._trajectory_world_xy.shape[0]),
                        )
                        self._last_error = None
                except Exception as exc:  # noqa: BLE001
                    with self._lock:
                        if self._command is None or self._latest_observation is None:
                            continue
                        if capture_generation != self._capture_generation or goal_generation != self._goal_generation:
                            self._system1_event.set()
                            continue
                        self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
                        self._trajectory_stamp_s = float(observation.stamp_s)
                        self._last_error = f"{type(exc).__name__}: {exc}"
                        self._system1_stage = self._make_stage_payload("navdp", status="error", detail=self._last_error)

    @staticmethod
    def _result_pixel_goal(result, observation: NavigationObservation) -> list[int] | None:
        pixel_xy = getattr(result, "pixel_xy", None)
        if pixel_xy is not None:
            pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
            return [int(round(float(pixel[0]))), int(round(float(pixel[1])))]
        uv_norm = getattr(result, "uv_norm", None)
        if uv_norm is None:
            return None
        pixel = normalized_uv_to_pixel_xy(
            np.asarray(uv_norm, dtype=np.float32),
            image_width=int(observation.rgb.shape[1]),
            image_height=int(observation.rgb.shape[0]),
        )
        return [int(pixel[0]), int(pixel[1])]

    def _set_goal_locked(
        self,
        goal_world_xy: np.ndarray | None,
        pixel_goal: list[int] | None,
        active_target: dict[str, object] | None,
    ) -> bool:
        changed = _goal_changed(self._last_goal_world_xy, goal_world_xy)
        self._last_goal_world_xy = None if goal_world_xy is None else np.asarray(goal_world_xy, dtype=np.float32)
        self._last_system2_pixel_goal = None if pixel_goal is None else list(pixel_goal)
        self._active_target_summary = None if active_target is None else dict(active_target)
        if changed:
            self._goal_generation += 1
        return changed

    def _advance_memory_navigation(self, observation: NavigationObservation) -> None:
        with self._lock:
            command = self._command
            if command is None or command.mode != "memory_pose" or command.memory_target is None:
                return
            target = command.memory_target
            robot_xy = np.asarray(observation.robot_state.base_pos_w, dtype=np.float32)[:2]
            target_xy = np.asarray(target.world_pose_xyz[:2], dtype=np.float32)
            distance_m = float(np.linalg.norm(robot_xy - target_xy))
            reacquire_started_at = self._memory_reacquire_started_at
            detections = list(self._latest_detections)

            if reacquire_started_at is None:
                self._memory_reacquire_state = "approach_memory_pose"
                if distance_m > float(target.stop_radius_m):
                    return
                self._memory_reacquire_state = "reacquire_pending"
                self._memory_reacquire_started_at = time.monotonic()
                reacquire_started_at = self._memory_reacquire_started_at
                self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
                self._trajectory_stamp_s = float(observation.stamp_s)
                self._system1_stage = self._make_stage_payload("navdp", status="paused", detail="reacquiring target")
                self._action_override_mode = "reacquire"
                if not self._memory_reacquire_sweep_sent:
                    self._last_system2 = {
                        "status": "reacquire_pending",
                        "decision_mode": "yaw_left",
                        "text": "memory target reacquire sweep",
                        "latency_ms": None,
                        "action_sequence": ["yaw_left", "yaw_right", "yaw_left"],
                    }
                    self._memory_reacquire_sweep_sent = True
                else:
                    self._last_system2 = {
                        "status": "reacquire_pending",
                        "decision_mode": "memory_pose",
                        "text": "waiting for memory target reacquire",
                        "latency_ms": None,
                        "action_sequence": [],
                    }

            matched = self._match_reacquired_detection(target, detections)
            if matched is not None:
                self._memory_reacquire_state = "reacquired"
                self._action_override_mode = None
                self._last_error = None
                self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
                self._trajectory_stamp_s = float(observation.stamp_s)
                self._system1_stage = self._make_stage_payload("navdp", status="idle", detail="memory target reacquired")
                self._last_system2 = {
                    "status": "reacquired",
                    "decision_mode": "stop",
                    "text": "memory target reacquired",
                    "latency_ms": None,
                    "action_sequence": [],
                }
                return

            elapsed = max(0.0, time.monotonic() - reacquire_started_at)
            self._memory_reacquire_state = "reacquire_pending"
            self._last_system2 = {
                "status": "reacquire_pending",
                "decision_mode": "memory_pose",
                "text": "waiting for memory target reacquire",
                "latency_ms": None,
                "action_sequence": [],
            }
            if elapsed < float(target.reacquire_timeout_sec):
                return

            self._memory_reacquire_state = "failed"
            self._action_override_mode = None
            self._last_error = "memory_target_not_reacquired"
            self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
            self._trajectory_stamp_s = float(observation.stamp_s)
            self._system1_stage = self._make_stage_payload("navdp", status="error", detail=self._last_error)
            self._last_system2 = {
                "status": "reacquire_failed",
                "decision_mode": "stop",
                "text": self._last_error,
                "latency_ms": None,
                "action_sequence": [],
            }

    def _advance_return_navigation(self, observation: NavigationObservation) -> None:
        with self._lock:
            command = self._command
            if command is None or command.mode != "return_pose" or command.return_target is None:
                return
            robot_xy = np.asarray(observation.robot_state.base_pos_w, dtype=np.float32)[:2]
            target_xy = np.asarray(command.return_target.world_xy, dtype=np.float32).reshape(2)
            tolerance_m = max(0.05, float(getattr(self.args, "memory_approach_radius_m", 0.8)))
            distance_m = float(np.linalg.norm(robot_xy - target_xy))
            if distance_m > tolerance_m:
                return
            self._action_override_mode = None
            self._last_error = None
            self._trajectory_world_xy = np.zeros((0, 2), dtype=np.float32)
            self._trajectory_stamp_s = float(observation.stamp_s)
            self._set_goal_locked(None, None, None)
            self._system1_stage = self._make_stage_payload("navdp", status="idle", detail="return pose reached")
            self._last_system2 = {
                "status": "stop",
                "decision_mode": "stop",
                "text": "return pose reached",
                "latency_ms": None,
                "action_sequence": [],
            }

    @staticmethod
    def _match_reacquired_detection(
        target: MemoryNavigationTarget,
        detections: list[CompactDetection],
    ) -> CompactDetection | None:
        best: tuple[float, CompactDetection] | None = None
        for detection in detections:
            if detection.world_pose_xyz is None:
                continue
            if detection.class_name != target.class_name:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(detection.world_pose_xyz, dtype=np.float32).reshape(3)
                    - np.asarray(target.world_pose_xyz, dtype=np.float32).reshape(3)
                )
            )
            if distance > float(target.reacquire_radius_m):
                continue
            if best is None or distance < best[0]:
                best = (distance, detection)
        return None if best is None else best[1]

    def status_payload(self) -> dict[str, object]:
        with self._lock:
            command = self._command
            path_points = int(self._trajectory_world_xy.shape[0])
            trajectory_stamp_s = float(self._trajectory_stamp_s)
            last_error = self._last_error
            last_system2 = None if self._last_system2 is None else dict(self._last_system2)
            goal_world_xy = None if self._last_goal_world_xy is None else self._last_goal_world_xy.astype(float).tolist()
            system2_pixel_goal = None if self._last_system2_pixel_goal is None else list(self._last_system2_pixel_goal)
            active_target = None if self._active_target_summary is None else dict(self._active_target_summary)
            system2_stage = dict(self._system2_stage)
            system1_stage = dict(self._system1_stage)
            capture_generation = int(self._capture_generation)
            goal_generation = int(self._goal_generation)
            action_override_mode = self._action_override_mode
            memory_reacquire_state = self._memory_reacquire_state
            memory_target = None if command is None else command.memory_target
            return_target = None if command is None else command.return_target
            observation = self._latest_observation
        now = time.monotonic()
        plan_age_s = None if trajectory_stamp_s <= 0.0 else max(0.0, now - trajectory_stamp_s)
        memory_aware_task_active = command is not None and command.mode == "memory_pose"
        resolved_memory_object_id = None if memory_target is None else memory_target.object_id
        resolved_memory_pose_age_sec = None if memory_target is None else float(memory_target.pose_age_sec)
        current_robot_pose = None
        if observation is not None:
            robot_state = observation.robot_state
            base_pos_w = np.asarray(robot_state.base_pos_w, dtype=np.float32).reshape(-1)
            current_robot_pose = {
                "world_xyz": [
                    float(base_pos_w[0]),
                    float(base_pos_w[1]),
                    float(base_pos_w[2] if base_pos_w.shape[0] >= 3 else 0.0),
                ],
                "world_xy": [float(base_pos_w[0]), float(base_pos_w[1])],
                "yaw_rad": float(robot_state.base_yaw),
            }
        return {
            "ok": True,
            "service": "navigation_system",
            "status": "idle" if command is None else ("error" if last_error else "running"),
            "instruction": None if command is None else command.instruction,
            "language": None if command is None else command.language,
            "task_id": None if command is None else command.task_id,
            "session_id": None if command is None else command.session_id,
            "path_points": path_points,
            "plan_age_s": plan_age_s,
            "last_error": last_error,
            "system2": last_system2,
            "system2_stage": system2_stage,
            "system1": system1_stage,
            "navdp": dict(system1_stage),
            "goal_world_xy": goal_world_xy,
            "capture_generation": capture_generation,
            "goal_generation": goal_generation,
            "action_override_mode": action_override_mode,
            "system2_pixel_goal": system2_pixel_goal,
            "system2PixelGoal": None if system2_pixel_goal is None else list(system2_pixel_goal),
            "active_target": active_target,
            "activeTarget": None if active_target is None else dict(active_target),
            "memoryAwareTaskActive": bool(memory_aware_task_active),
            "memoryNavigationMode": None if not memory_aware_task_active else "memory_pose",
            "resolvedMemoryObjectId": resolved_memory_object_id,
            "resolvedMemoryPoseAgeSec": resolved_memory_pose_age_sec,
            "reacquireState": memory_reacquire_state,
            "reacquire_state": memory_reacquire_state,
            "current_robot_pose": current_robot_pose,
            "currentRobotPose": None if current_robot_pose is None else dict(current_robot_pose),
            "returnTargetWorldXy": None if return_target is None else [float(return_target.world_xy[0]), float(return_target.world_xy[1])],
        }

    def trajectory_payload(self) -> dict[str, object]:
        with self._lock:
            trajectory = self._trajectory_world_xy.astype(float).tolist()
            trajectory_stamp_s = float(self._trajectory_stamp_s)
        payload = self.status_payload()
        payload["trajectory_world_xy"] = trajectory
        payload["stamp_s"] = trajectory_stamp_s
        return payload

    def shutdown(self) -> None:
        self._stop_event.set()
        self._system2_event.set()
        self._system1_event.set()
        if self._system2_thread.is_alive():
            self._system2_thread.join(timeout=2.0)
        if self._system1_thread.is_alive():
            self._system1_thread.join(timeout=2.0)
        for ring in self._shm_rings.values():
            try:
                ring.close()
            except Exception:
                continue
        self._shm_rings.clear()
        if self._backend_registry is not None:
            self._backend_registry.stop_all()


class NavigationSystemServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._service = NavigationSystem(args)
        self._server = ThreadingHTTPServer((str(args.host), int(args.port)), self._build_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="navigation-system-api", daemon=True)

    def _build_handler(self):
        service = self._service

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, object]):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(int(status_code))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _read_json_body(self) -> dict[str, object]:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length > 0 else b""
                if not raw:
                    return {}
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("expected JSON object body")
                return payload

            def do_OPTIONS(self):
                self._send_json(HTTPStatus.NO_CONTENT, {})

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path == "/healthz":
                    self._send_json(HTTPStatus.OK, {"ok": True, "service": "navigation_system"})
                    return
                if path == "/navigation/status":
                    self._send_json(HTTPStatus.OK, service.status_payload())
                    return
                if path == "/navigation/trajectory":
                    self._send_json(HTTPStatus.OK, service.trajectory_payload())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def do_POST(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                try:
                    payload = self._read_json_body()
                except json.JSONDecodeError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid_json: {exc}"})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    return
                try:
                    if path == "/navigation/command":
                        mode = str(payload.get("mode") or "instruction").strip() or "instruction"
                        task_id = payload.get("task_id")
                        if task_id is not None and not isinstance(task_id, str):
                            raise ValueError("task_id must be a string")
                        if mode == "memory_pose":
                            target = _decode_memory_target(payload.get("target"))
                            self._send_json(HTTPStatus.OK, service.command_memory_target(target, task_id=task_id))
                            return
                        if mode == "return_pose":
                            target = _decode_return_pose_target(payload.get("target"))
                            self._send_json(HTTPStatus.OK, service.command_return_pose(target, task_id=task_id))
                            return
                        instruction = payload.get("instruction")
                        language = payload.get("language", "en")
                        if not isinstance(instruction, str) or not instruction.strip():
                            raise ValueError("instruction must be a non-empty string")
                        if not isinstance(language, str):
                            raise ValueError("language must be a string")
                        self._send_json(HTTPStatus.OK, service.command(instruction, language, task_id=task_id))
                        return
                    if path == "/navigation/cancel":
                        self._send_json(HTTPStatus.OK, service.cancel())
                        return
                    if path == "/navigation/update":
                        self._send_json(HTTPStatus.OK, service.update(payload))
                        return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args):
                del format, args

        return Handler

    def start(self) -> None:
        self._thread.start()
        print(f"[INFO] Navigation system API listening on http://{self.args.host}:{self.args.port}")

    def shutdown(self) -> None:
        self._service.shutdown()
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the navigation system service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17882)
    parser.add_argument("--system2-url", default="http://127.0.0.1:15801")
    parser.add_argument("--navdp-url", default="http://127.0.0.1:18888")
    parser.add_argument("--system2-timeout", type=float, default=20.0)
    parser.add_argument("--navdp-timeout", type=float, default=5.0)
    parser.add_argument("--navdp-fallback", choices=("disabled", "heuristic"), default="disabled")
    parser.add_argument("--navdp-stop-threshold", type=float, default=-0.5)
    parser.add_argument("--goal-depth-window", type=int, default=5)
    parser.add_argument("--goal-depth-min", type=float, default=0.25)
    parser.add_argument("--goal-depth-max", type=float, default=6.0)
    parser.add_argument(
        "--memory-approach-radius-m",
        type=float,
        default=float(os.environ.get("AURA_MEMORY_APPROACH_RADIUS_M", "0.8")),
    )
    parser.add_argument(
        "--memory-reacquire-radius-m",
        type=float,
        default=float(os.environ.get("AURA_MEMORY_REACQUIRE_RADIUS_M", "1.0")),
    )
    parser.add_argument(
        "--memory-reacquire-timeout-sec",
        type=float,
        default=float(os.environ.get("AURA_MEMORY_REACQUIRE_TIMEOUT_SEC", "4.0")),
    )
    parser.add_argument("--backend-autostart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend-log-dir", default=str(_default_backend_log_dir()))
    parser.add_argument("--navdp-host", default=os.environ.get("NAVDP_HOST", "127.0.0.1"))
    parser.add_argument("--navdp-port", type=int, default=int(os.environ.get("NAVDP_PORT", "18888")))
    parser.add_argument("--navdp-checkpoint", default=str(os.environ.get("NAVDP_CHECKPOINT", REPO_ROOT / "artifacts" / "models" / "navdp-cross-modal.ckpt")))
    parser.add_argument("--navdp-device", default=os.environ.get("NAVDP_DEVICE", "cuda:0"))
    parser.add_argument("--navdp-tensorrt-mode", choices=("off", "auto", "required", "build"), default=os.environ.get("NAVDP_TENSORRT_MODE", "off"))
    parser.add_argument("--navdp-tensorrt-engine-dir", default=os.environ.get("NAVDP_TENSORRT_ENGINE_DIR", str(REPO_ROOT / "artifacts" / "models" / "navdp_tensorrt")))
    parser.add_argument("--navdp-tensorrt-precision", choices=("fp16", "fp32"), default=os.environ.get("NAVDP_TENSORRT_PRECISION", "fp16"))
    parser.add_argument("--system2-host", default=os.environ.get("SYSTEM2_HOST", "127.0.0.1"))
    parser.add_argument("--system2-port", type=int, default=int(os.environ.get("SYSTEM2_PORT", "15801")))
    parser.add_argument("--system2-llama-url", default=os.environ.get("SYSTEM2_LLAMA_URL", "http://127.0.0.1:15802"))
    parser.add_argument("--system2-model-path", default=os.environ.get("SYSTEM2_MODEL_PATH", ""))
    parser.add_argument("--navigation-shm-name", default=NAVIGATION_SHM_NAME)
    parser.add_argument("--navigation-shm-slot-size", type=int, default=NAVIGATION_SHM_SLOT_SIZE)
    parser.add_argument("--navigation-shm-capacity", type=int, default=NAVIGATION_SHM_CAPACITY)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = NavigationSystemServer(args)
    server.start()
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
