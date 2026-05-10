"""Hot-path runtime coordination for capture, memory, navigation, and control."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from systems.control.runtime_control_api import RuntimeControlApiServer
from systems.memory.api.runtime import ShortTermMemory
from systems.navigation.api.runtime import NavigationSystemClient, RobotState2D, yaw_from_quaternion_wxyz
from systems.perception.api.camera_api import CameraPitchApiServer, G1NavCameraSensor, resolve_camera_control_prim_path
from systems.perception.detector_runtime import InProcessDetectorRuntime
from systems.perception.observation import PerceptionObservationService
from systems.perception.telemetry import ViewerFramePublisher
from systems.shared.contracts.observation import RawObservation


@dataclass(slots=True)
class _NavigationUpdateRequest:
    generation: int
    status_snapshot: dict[str, object]
    update_kwargs: dict[str, object]


@dataclass(slots=True)
class _NavigationWorkerResult:
    generation: int
    payload: dict[str, object] | None = None
    error: str | None = None


class NavigationRuntimeCoordinator:
    """Own the Isaac hot path for navigation-driven runtime control."""

    def __init__(self, args, *, control_handler):
        self._args = args
        self._control_handler = control_handler
        self._navigation = NavigationSystemClient(server_url=args.navigation_url, timeout_s=args.navigation_timeout)
        self._memory = ShortTermMemory()
        self._viewer_publish_enabled = bool(getattr(args, "viewer_publish", True))
        self._viewer_transport_error: str | None = None
        viewer_publisher = self._create_viewer_publisher()
        detector_runtime = InProcessDetectorRuntime(
            enabled=bool(getattr(args, "detection_enabled", False)),
            model_path=getattr(args, "detection_model_path", ""),
        )
        self._perception = PerceptionObservationService(
            viewer_publisher=viewer_publisher,
            detector_runtime=detector_runtime,
        )
        self._controller = None
        self._sensor: G1NavCameraSensor | None = None
        self._camera_api_server: CameraPitchApiServer | None = None
        self._runtime_api_server: RuntimeControlApiServer | None = None
        self._active_navigation_session_id: str | None = None
        self._last_navigation_status: dict[str, object] = {"status": "idle"}
        self._last_navigation_payload: dict[str, object] | None = None
        self._last_error: str | None = None
        self._last_update_time = 0.0
        self._last_status_time = 0.0
        self._update_interval = 1.0 / max(0.1, float(args.navigation_update_hz))
        self._status_interval = self._update_interval
        self._worker_lock = threading.Lock()
        self._worker_event = threading.Event()
        self._worker_stop = False
        self._rpc_generation = 0
        self._pending_status_request = False
        self._status_inflight = False
        self._pending_update_request: _NavigationUpdateRequest | None = None
        self._update_inflight = False
        self._completed_status_result: _NavigationWorkerResult | None = None
        self._completed_update_result: _NavigationWorkerResult | None = None
        self._navigation_worker = threading.Thread(
            target=self._navigation_worker_loop,
            name="navigation-rpc-worker",
            daemon=True,
        )
        self._navigation_worker.start()

    def bind_controller(self, controller):
        if self._controller is not None:
            return
        self._controller = controller
        camera_prim_path = resolve_camera_control_prim_path(controller.robot_prim_path, self._args.camera_prim_path)
        self._sensor = G1NavCameraSensor(
            prim_path=camera_prim_path,
            resolution=(self._args.camera_width, self._args.camera_height),
            translation=tuple(self._args.camera_pos),
            orientation_wxyz=tuple(self._args.camera_quat),
            clipping_range=(self._args.camera_near, self._args.camera_far),
            initial_pitch_deg=self._args.camera_pitch_deg,
            pitch_limits_deg=(self._args.camera_pitch_min_deg, self._args.camera_pitch_max_deg),
            annotator_device="cpu",
        )
        self._sensor.attach()
        if self._args.camera_api_port > 0 and self._camera_api_server is None:
            self._camera_api_server = CameraPitchApiServer(
                host=self._args.camera_api_host,
                port=self._args.camera_api_port,
                camera_sensor=self._sensor,
            )
            self._camera_api_server.start()
        if self._args.runtime_control_api_port > 0 and self._runtime_api_server is None:
            self._runtime_api_server = RuntimeControlApiServer(
                host=self._args.runtime_control_api_host,
                port=self._args.runtime_control_api_port,
                runtime_handler=self,
                camera_sensor=self._sensor,
            )
            self._runtime_api_server.start()

    def reset(self):
        self._memory.reset_epoch(ShortTermMemory.SYSTEM2_CONSUMER)
        self._memory.reset_epoch(ShortTermMemory.NAVDP_CONSUMER)
        self._active_navigation_session_id = None
        self._last_navigation_status = {"status": "idle"}
        self._last_navigation_payload = None
        self._last_error = None
        self._last_update_time = 0.0
        self._last_status_time = 0.0
        with self._worker_lock:
            self._rpc_generation += 1
            self._pending_status_request = False
            self._status_inflight = False
            self._pending_update_request = None
            self._update_inflight = False
            self._completed_status_result = None
            self._completed_update_result = None
        self._worker_event.set()
        self._control_handler.reset()

    def step(self):
        if self._controller is None or self._sensor is None:
            return
        self._consume_navigation_worker_results()
        self._sensor.apply_pending_pitch()

        now = time.monotonic()
        self._request_navigation_status(now)
        frame = self._sensor.capture_frame()
        if frame is None:
            return

        robot_state = self._robot_state()
        navigation_status = self._navigation_snapshot()
        observation = RawObservation(
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            camera_pos_w=frame.camera_pos_w,
            camera_rot_w=frame.camera_rot_w,
            robot_state=robot_state,
            stamp_s=frame.stamp_s,
            metadata=self._viewer_metadata(frame=frame, navigation_status=navigation_status),
        )
        normalized = self._perception.ingest(observation)
        should_update_navigation = (now - self._last_update_time) >= self._update_interval
        if not should_update_navigation:
            return

        self._memory.observe(normalized)
        try:
            normalized_metadata = getattr(normalized, "metadata", {})
            if not isinstance(normalized_metadata, dict):
                normalized_metadata = {}
            self._submit_navigation_update(
                _NavigationUpdateRequest(
                    generation=self._rpc_generation,
                    status_snapshot=dict(navigation_status),
                    update_kwargs={
                        "rgb": normalized.rgb,
                        "depth": normalized.depth,
                        "intrinsic": normalized.intrinsic,
                        "camera_pos_w": normalized.camera_pos_w,
                        "camera_rot_w": normalized.camera_rot_w,
                        "base_pos_w": robot_state.base_pos_w,
                        "base_yaw": robot_state.base_yaw,
                        "lin_vel_b": robot_state.lin_vel_b,
                        "yaw_rate": robot_state.yaw_rate,
                        "stamp_s": normalized.stamp_s,
                        "detections": self._compact_detections(normalized_metadata),
                    },
                ),
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            payload = {
                "status": "error",
                "instruction": navigation_status.get("instruction"),
                "task_id": navigation_status.get("task_id"),
                "session_id": navigation_status.get("session_id"),
                "trajectory_world_xy": [],
                "stamp_s": normalized.stamp_s,
                "last_error": self._last_error,
            }
            self._last_navigation_payload = dict(payload)
            self._control_handler.update_navigation_payload(payload)
        self._last_update_time = now

    def runtime_status(self) -> dict[str, object]:
        control_status = dict(self._control_handler.runtime_status())
        viewer_status = self._perception.latest_health()
        viewer_status["enabled"] = self._viewer_publish_enabled and self._viewer_transport_error is None
        if self._viewer_transport_error is not None:
            viewer_status["status"] = "degraded"
            viewer_status["last_error"] = self._viewer_transport_error
        control_status["viewer"] = viewer_status
        control_status["navigation"] = self._navigation_snapshot()
        control_status["last_error"] = self._last_error or control_status.get("last_error")
        return control_status

    def command_api_status(self) -> dict[str, object]:
        return self.runtime_status()

    def shutdown(self):
        with self._worker_lock:
            self._worker_stop = True
            self._rpc_generation += 1
            self._pending_status_request = False
            self._pending_update_request = None
            self._completed_status_result = None
            self._completed_update_result = None
        self._worker_event.set()
        if self._navigation_worker.is_alive():
            self._navigation_worker.join(timeout=2.0)
        if self._runtime_api_server is not None:
            self._runtime_api_server.shutdown()
            self._runtime_api_server = None
        if self._camera_api_server is not None:
            self._camera_api_server.shutdown()
            self._camera_api_server = None
        if self._sensor is not None:
            self._sensor.shutdown()
            self._sensor = None
        close_navigation = getattr(self._navigation, "close", None)
        if callable(close_navigation):
            close_navigation()
        self._perception.close()

    def _consume_navigation_worker_results(self) -> None:
        with self._worker_lock:
            generation = self._rpc_generation
            status_result = self._completed_status_result
            update_result = self._completed_update_result
            self._completed_status_result = None
            self._completed_update_result = None
        if status_result is not None and status_result.generation == generation:
            self._apply_navigation_status_result(status_result)
        if update_result is not None and update_result.generation == generation:
            self._apply_navigation_update_result(update_result)

    def _create_viewer_publisher(self) -> ViewerFramePublisher | None:
        if not self._viewer_publish_enabled:
            return None
        try:
            return ViewerFramePublisher()
        except Exception as exc:  # noqa: BLE001
            self._viewer_transport_error = f"{type(exc).__name__}: {exc}"
            print(
                "[WARN] Viewer transport disabled; control runtime will continue without dashboard frames. "
                f"Reason: {self._viewer_transport_error}"
            )
            return None

    def _apply_navigation_status_result(self, result: _NavigationWorkerResult) -> None:
        if result.error is not None:
            self._last_error = result.error
            return
        if result.payload is None:
            return
        self._last_error = None
        self._apply_navigation_status(result.payload)
        if str(result.payload.get("status")) == "idle":
            stamp_s = 0.0
            if isinstance(self._last_navigation_payload, dict):
                stamp_s = float(self._last_navigation_payload.get("stamp_s", 0.0))
            elif self._last_update_time > 0.0:
                stamp_s = float(self._last_update_time)
            idle_payload = self._idle_payload(stamp_s)
            self._last_navigation_payload = dict(idle_payload)
            self._control_handler.update_navigation_payload(idle_payload)

    def _apply_navigation_update_result(self, result: _NavigationWorkerResult) -> None:
        payload = result.payload
        if payload is None:
            return
        if str(payload.get("status") or "").strip().lower() == "idle":
            self._last_navigation_status = dict(payload)
            self._last_navigation_payload = dict(payload)
            self._last_error = result.error
            self._control_handler.update_navigation_payload(payload)
            return
        if not self._should_accept_navigation_payload(payload):
            return
        self._last_navigation_payload = dict(payload)
        self._last_error = result.error
        self._control_handler.update_navigation_payload(payload)

    def _apply_navigation_status(self, status: dict[str, object]) -> None:
        self._last_navigation_status = dict(status)
        session_id = status.get("session_id")
        if isinstance(session_id, str) and session_id and session_id != self._active_navigation_session_id:
            self._active_navigation_session_id = session_id
            self._memory.reset_epoch(ShortTermMemory.SYSTEM2_CONSUMER, session_id=session_id, include_latest=True)
            self._memory.reset_epoch(ShortTermMemory.NAVDP_CONSUMER, session_id=session_id, include_latest=True)
        if status.get("status") == "idle":
            self._active_navigation_session_id = None
            self._last_navigation_payload = None

    def _navigation_snapshot(self) -> dict[str, object]:
        snapshot = dict(self._last_navigation_status)
        payload = self._last_navigation_payload
        if not isinstance(payload, dict):
            return snapshot
        if str(snapshot.get("status")) == "idle":
            return snapshot
        payload_session_id = payload.get("session_id")
        snapshot_session_id = snapshot.get("session_id")
        if (
            isinstance(snapshot_session_id, str)
            and snapshot_session_id
            and isinstance(payload_session_id, str)
            and payload_session_id
            and payload_session_id != snapshot_session_id
        ):
            return snapshot
        if str(payload.get("status")) == "idle" and str(snapshot.get("status")) != "idle":
            return snapshot
        merged = dict(snapshot)
        merged.update(payload)
        return merged

    def _should_accept_navigation_payload(self, payload: dict[str, object]) -> bool:
        current_status = self._last_navigation_status
        if str(current_status.get("status")) == "idle":
            return False
        current_session_id = current_status.get("session_id")
        payload_session_id = payload.get("session_id")
        if (
            isinstance(current_session_id, str)
            and current_session_id
            and isinstance(payload_session_id, str)
            and payload_session_id
            and payload_session_id != current_session_id
        ):
            return False
        return True

    def _request_navigation_status(self, now: float) -> None:
        if (now - self._last_status_time) < self._status_interval:
            return
        with self._worker_lock:
            if self._pending_status_request or self._status_inflight:
                return
            self._pending_status_request = True
            self._last_status_time = now
        self._worker_event.set()

    def _submit_navigation_update(self, request: _NavigationUpdateRequest) -> None:
        with self._worker_lock:
            request.generation = self._rpc_generation
            self._pending_update_request = request
        self._worker_event.set()

    def _navigation_worker_loop(self) -> None:
        while True:
            self._worker_event.wait(timeout=0.1)
            while True:
                task_kind: str | None = None
                generation = 0
                update_request: _NavigationUpdateRequest | None = None
                with self._worker_lock:
                    if self._worker_stop:
                        return
                    if self._pending_status_request:
                        self._pending_status_request = False
                        self._status_inflight = True
                        generation = self._rpc_generation
                        task_kind = "status"
                    elif self._pending_update_request is not None:
                        update_request = self._pending_update_request
                        self._pending_update_request = None
                        self._update_inflight = True
                        task_kind = "update"
                    else:
                        self._worker_event.clear()
                        break
                if task_kind == "update" and update_request is not None:
                    result = self._execute_navigation_update(update_request)
                    with self._worker_lock:
                        self._update_inflight = False
                        if result.generation == self._rpc_generation:
                            self._completed_update_result = result
                        if self._pending_status_request or self._pending_update_request is not None:
                            self._worker_event.set()
                    continue
                if task_kind == "status":
                    result = self._execute_navigation_status(generation)
                    with self._worker_lock:
                        self._status_inflight = False
                        if result.generation == self._rpc_generation:
                            self._completed_status_result = result
                        if self._pending_status_request or self._pending_update_request is not None:
                            self._worker_event.set()

    def _execute_navigation_status(self, generation: int) -> _NavigationWorkerResult:
        try:
            payload = self._navigation.status()
        except Exception as exc:  # noqa: BLE001
            return _NavigationWorkerResult(
                generation=generation,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _NavigationWorkerResult(
            generation=generation,
            payload=dict(payload),
            error=None,
        )

    def _execute_navigation_update(self, request: _NavigationUpdateRequest) -> _NavigationWorkerResult:
        try:
            payload = self._navigation.update(**request.update_kwargs)
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            return _NavigationWorkerResult(
                generation=request.generation,
                payload={
                    "status": "error",
                    "instruction": request.status_snapshot.get("instruction"),
                    "task_id": request.status_snapshot.get("task_id"),
                    "session_id": request.status_snapshot.get("session_id"),
                    "trajectory_world_xy": [],
                    "stamp_s": float(request.update_kwargs.get("stamp_s", 0.0)),
                    "last_error": error_text,
                },
                error=error_text,
            )
        return _NavigationWorkerResult(
            generation=request.generation,
            payload=dict(payload),
            error=None,
        )

    def _robot_state(self) -> RobotState2D:
        base_pos_w, base_quat_wxyz = self._controller.robot.get_world_pose()
        lin_vel_w = np.asarray(self._controller.robot.get_linear_velocity(), dtype=np.float32)
        base_yaw = yaw_from_quaternion_wxyz(np.asarray(base_quat_wxyz, dtype=np.float32))
        cos_yaw = float(np.cos(base_yaw))
        sin_yaw = float(np.sin(base_yaw))
        rot_bw = np.asarray(((cos_yaw, sin_yaw), (-sin_yaw, cos_yaw)), dtype=np.float32)
        lin_vel_b_xy = rot_bw @ np.asarray(lin_vel_w[:2], dtype=np.float32)
        yaw_rate = float(np.asarray(self._controller.robot.get_angular_velocity(), dtype=np.float32)[2])
        return RobotState2D(
            base_pos_w=np.asarray(base_pos_w, dtype=np.float32),
            base_yaw=base_yaw,
            lin_vel_b=np.asarray((lin_vel_b_xy[0], lin_vel_b_xy[1]), dtype=np.float32),
            yaw_rate=yaw_rate,
        )

    @staticmethod
    def _idle_payload(stamp_s: float) -> dict[str, object]:
        return {
            "status": "idle",
            "instruction": None,
            "task_id": None,
            "session_id": None,
            "trajectory_world_xy": [],
            "stamp_s": float(stamp_s),
            "last_error": None,
        }

    def _viewer_metadata(self, *, frame, navigation_status: dict[str, object]) -> dict[str, object]:
        overlay: dict[str, object] = {}
        trajectory_pixels = self._trajectory_pixels(
            trajectory_world_xy=navigation_status.get("trajectory_world_xy"),
            camera_pos_w=frame.camera_pos_w,
            camera_rot_w=frame.camera_rot_w,
            intrinsic=frame.intrinsic,
        )
        if trajectory_pixels:
            overlay["trajectory_pixels"] = trajectory_pixels
            overlay["trajectoryPixels"] = trajectory_pixels
        system2_pixel_goal = self._system2_pixel_goal(navigation_status)
        if system2_pixel_goal is not None:
            overlay["system2_pixel_goal"] = system2_pixel_goal
            overlay["system2PixelGoal"] = system2_pixel_goal
        active_target = self._active_target_overlay(
            navigation_status=navigation_status,
            camera_pos_w=frame.camera_pos_w,
            camera_rot_w=frame.camera_rot_w,
            intrinsic=frame.intrinsic,
            system2_pixel_goal=system2_pixel_goal,
        )
        if active_target:
            overlay["active_target"] = active_target
            overlay["activeTarget"] = dict(active_target)
        if not overlay:
            return {}
        return {
            **overlay,
            "viewer_overlay": overlay,
        }

    def _trajectory_pixels(
        self,
        *,
        trajectory_world_xy: object,
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
        intrinsic: np.ndarray,
    ) -> list[list[int]]:
        if not isinstance(trajectory_world_xy, list):
            return []
        pixels: list[list[int]] = []
        for point in trajectory_world_xy:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x_value, y_value = point[0], point[1]
            if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
                continue
            pixel = self._project_world_point(
                world_point=np.asarray((float(x_value), float(y_value), 0.0), dtype=np.float32),
                camera_pos_w=camera_pos_w,
                camera_rot_w=camera_rot_w,
                intrinsic=intrinsic,
            )
            if pixel is not None:
                pixels.append(pixel)
        return pixels

    @staticmethod
    def _system2_pixel_goal(navigation_status: dict[str, object]) -> list[int] | None:
        system2_payload = navigation_status.get("system2")
        if not isinstance(system2_payload, dict):
            system2_payload = {}
        for candidate in (
            navigation_status.get("system2_pixel_goal"),
            navigation_status.get("system2PixelGoal"),
            system2_payload.get("pixel_goal"),
            system2_payload.get("pixel_xy"),
            system2_payload.get("pixel_goal_xy"),
            system2_payload.get("uv_px"),
        ):
            pixel = NavigationRuntimeCoordinator._pixel_pair(candidate)
            if pixel is not None:
                return pixel
        return None

    def _active_target_overlay(
        self,
        *,
        navigation_status: dict[str, object],
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
        intrinsic: np.ndarray,
        system2_pixel_goal: list[int] | None,
    ) -> dict[str, object]:
        active_target = navigation_status.get("active_target")
        if not isinstance(active_target, dict):
            active_target = navigation_status.get("activeTarget")
        summary = dict(active_target) if isinstance(active_target, dict) else {}
        goal_world_xy = navigation_status.get("goal_world_xy")
        goal_world_xyz = None
        if isinstance(goal_world_xy, (list, tuple)) and len(goal_world_xy) >= 2 and all(
            isinstance(value, (int, float)) for value in goal_world_xy[:2]
        ):
            goal_world_xyz = [float(goal_world_xy[0]), float(goal_world_xy[1]), 0.0]
        nav_goal_pixel = self._pixel_pair(summary.get("nav_goal_pixel"))
        if nav_goal_pixel is None and goal_world_xyz is not None:
            nav_goal_pixel = self._project_world_point(
                world_point=np.asarray(goal_world_xyz, dtype=np.float32),
                camera_pos_w=camera_pos_w,
                camera_rot_w=camera_rot_w,
                intrinsic=intrinsic,
            )
        if nav_goal_pixel is None:
            nav_goal_pixel = system2_pixel_goal
        if not summary and nav_goal_pixel is None and goal_world_xyz is None:
            return {}
        summary.setdefault("className", "Navigation Goal")
        summary.setdefault("source", "navigation")
        if nav_goal_pixel is not None:
            summary["nav_goal_pixel"] = nav_goal_pixel
        if goal_world_xyz is not None:
            summary["world_pose_xyz"] = goal_world_xyz
        return summary

    @staticmethod
    def _compact_detections(metadata: dict[str, object]) -> list[dict[str, object]]:
        detections = metadata.get("detections")
        if not isinstance(detections, list):
            return []
        compact_rows: list[dict[str, object]] = []
        for item in detections:
            if not isinstance(item, dict):
                continue
            class_name = item.get("class_name")
            if not isinstance(class_name, str) or not class_name.strip():
                continue
            row: dict[str, object] = {"class_name": class_name.strip()}
            class_id = item.get("class_id")
            if isinstance(class_id, (int, float)) and not isinstance(class_id, bool):
                row["class_id"] = int(round(float(class_id)))
            track_id = item.get("track_id")
            if isinstance(track_id, str) and track_id.strip():
                row["track_id"] = track_id.strip()
            bbox_xyxy = item.get("bbox_xyxy")
            if isinstance(bbox_xyxy, (list, tuple)) and len(bbox_xyxy) >= 4:
                row["bbox_xyxy"] = list(bbox_xyxy[:4])
            for key in ("confidence", "depth_m"):
                numeric = item.get(key)
                if isinstance(numeric, (int, float)) and not isinstance(numeric, bool):
                    row[key] = float(numeric)
            world_pose_xyz = item.get("world_pose_xyz")
            if isinstance(world_pose_xyz, (list, tuple)) and len(world_pose_xyz) >= 3:
                row["world_pose_xyz"] = list(world_pose_xyz[:3])
            compact_rows.append(row)
        return compact_rows

    @staticmethod
    def _pixel_pair(value: object) -> list[int] | None:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            x_value, y_value = value[0], value[1]
            if isinstance(x_value, (int, float)) and isinstance(y_value, (int, float)):
                return [int(round(float(x_value))), int(round(float(y_value)))]
        return None

    @staticmethod
    def _project_world_point(
        *,
        world_point: np.ndarray,
        camera_pos_w: np.ndarray,
        camera_rot_w: np.ndarray,
        intrinsic: np.ndarray,
    ) -> list[int] | None:
        try:
            world_xyz = np.asarray(world_point, dtype=np.float32).reshape(3)
            camera_xyz = np.asarray(camera_pos_w, dtype=np.float32).reshape(3)
            camera_rot = np.asarray(camera_rot_w, dtype=np.float32).reshape(3, 3)
            intrinsic_matrix = np.asarray(intrinsic, dtype=np.float32).reshape(3, 3)
        except Exception:
            return None
        camera_frame = camera_rot.T @ (world_xyz - camera_xyz)
        depth = float(camera_frame[2])
        if not np.isfinite(depth) or depth <= 1e-4:
            return None
        fx = float(intrinsic_matrix[0, 0])
        fy = float(intrinsic_matrix[1, 1])
        cx = float(intrinsic_matrix[0, 2])
        cy = float(intrinsic_matrix[1, 2])
        if not np.isfinite(fx) or not np.isfinite(fy) or abs(fx) < 1e-6 or abs(fy) < 1e-6:
            return None
        pixel_x = fx * float(camera_frame[0]) / depth + cx
        pixel_y = fy * float(camera_frame[1]) / depth + cy
        if not np.isfinite(pixel_x) or not np.isfinite(pixel_y):
            return None
        return [int(round(pixel_x)), int(round(pixel_y))]
