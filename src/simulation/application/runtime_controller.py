"""Asynchronous point-goal command source backed by NavDP."""

from __future__ import annotations

from dataclasses import asdict
import re
import threading
import time
import uuid

import numpy as np

from systems.control.runtime_control_api import RuntimeControlApiServer
from systems.inference.api.runtime import (
    InternVlaNavClient,
    System2Result,
    normalized_uv_to_pixel_xy,
    resolve_goal_world_xy,
    resolve_goal_world_xy_from_pixel,
)
from systems.memory.knowledge_runtime import create_knowledge_runtime
from systems.navigation.api.runtime import (
    FollowerState,
    HolonomicPurePursuitFollower,
    NavDpClient,
    NavDpPlan,
    PointGoalProvider,
    RobotState2D,
    camera_plan_to_world_xy,
    make_follower_state,
    point_goal_body_from_world,
    yaw_from_quaternion_wxyz,
)
from systems.reasoning.api.runtime import AuraTaskingAdapter, make_planner_task_frame_completion
from systems.perception.api.camera_api import CameraPitchApiServer, G1NavCameraSensor, resolve_camera_control_prim_path
from systems.world_state.api.runtime_state import (
    ActionOverrideState,
    CaptureState,
    CommandState,
    GoalState,
    LocomotionState,
    NavDpState,
    NavigationPipelineState,
    PlannerInput,
    StatusState,
    System2State,
    TaskExecutionState,
    goal_current_body_xy,
    goal_has_target,
    goal_is_done,
    goal_pending_pixel_xy,
    goal_pending_world_xy,
    goal_target_mode,
    goal_target_pixel_xy,
    goal_target_world_xy,
    make_navigation_pipeline_state,
)


RUNTIME_STOP_PATTERN = re.compile(r"^\s*(?:stop|halt|hold|freeze)\s*$", re.IGNORECASE)
DIRECT_ACTION_MODES = frozenset(("forward", "yaw_left", "yaw_right"))
SYSTEM2_PIXEL_GOAL_MODE = "pixel_goal"


def _wrap_to_pi(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(float(angle_rad)), np.cos(float(angle_rad))))


def system2_result_signature(*, status: str, uv_norm: np.ndarray | None, text: str) -> tuple[str, tuple[float, ...] | None, str]:
    """Create a stable signature for deduplicating System2 outputs."""

    normalized_status = str(status).strip().lower()
    normalized_text = " ".join(str(text).strip().split())
    if uv_norm is None:
        normalized_uv = None
    else:
        normalized_uv = tuple(float(value) for value in np.round(np.asarray(uv_norm, dtype=np.float32).reshape(2), 4))
    return normalized_status, normalized_uv, normalized_text


def should_latch_goal_update(
    *,
    current_goal_xy: np.ndarray | None,
    candidate_goal_xy: np.ndarray,
    goal_reached: bool,
    same_signature: bool,
    min_update_dist_m: float,
) -> bool:
    """Decide whether a newly projected world goal should replace the latched target."""

    if current_goal_xy is None or goal_reached:
        return True
    if same_signature:
        return False
    if float(min_update_dist_m) <= 0.0:
        return True
    distance = float(
        np.linalg.norm(np.asarray(candidate_goal_xy, dtype=np.float32).reshape(2) - np.asarray(current_goal_xy, dtype=np.float32).reshape(2))
    )
    return distance >= float(min_update_dist_m)


def _zero_command() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _empty_world_path() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


def serialize_system2_result(result: System2Result | None) -> dict[str, object] | None:
    if result is None:
        return None
    payload: dict[str, object] = {
        "status": result.status,
        "uv_norm": None if result.uv_norm is None else [float(result.uv_norm[0]), float(result.uv_norm[1])],
        "pixel_xy": None if result.pixel_xy is None else [int(round(float(result.pixel_xy[0]))), int(round(float(result.pixel_xy[1])))],
        "text": result.text,
        "latency_ms": float(result.latency_ms),
        "stamp_s": float(result.stamp_s),
        "decision_mode": result.decision_mode,
        "action_sequence": None if not result.action_sequence else list(result.action_sequence),
        "needs_requery": bool(result.needs_requery),
    }
    if result.raw_payload is not None:
        payload["raw_payload"] = dict(result.raw_payload)
    return payload


class NavDpPointGoalController:
    """Generate locomotion commands from NavDP point-goal plans."""

    requires_render = True

    def __init__(self, args):
        self.quit_requested = False

        self._args = args
        NavDpClient.validate_runtime_dependencies()
        self._client = NavDpClient(
            server_url=args.navdp_url,
            timeout_s=args.navdp_timeout,
            fallback_mode=args.navdp_fallback,
        )
        self._goal_provider = PointGoalProvider(
            goal_xy=(args.nav_goal_x, args.nav_goal_y),
            goal_frame=args.nav_goal_frame,
            tolerance=args.nav_goal_tolerance,
        )
        self._follower = HolonomicPurePursuitFollower(
            lookahead_distance=args.lookahead_distance,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            smoothing_tau=args.cmd_smoothing_tau,
        )

        self._controller = None
        self._sensor: G1NavCameraSensor | None = None
        self._camera_api_server: CameraPitchApiServer | None = None
        self._algorithm_name: str | None = None

        self._running = True
        self._lock = threading.Lock()
        self._planner_input: PlannerInput | None = None
        self._planner_input_stamp = 0.0
        self._last_planned_input_stamp = 0.0
        self._latest_plan: NavDpPlan | None = None
        self._latest_world_path = np.zeros((0, 2), dtype=np.float32)
        self._latest_plan_time = 0.0
        self._planning_error: str | None = None
        self._last_capture_time = 0.0
        self._last_warning_time = 0.0
        self._last_status_time = 0.0
        self._status_interval = 1.0
        self._goal_done_reported = False
        self._replan_interval = 1.0 / max(0.1, float(args.navdp_replan_hz))
        self._plan_timeout = max(0.1, float(args.navdp_plan_timeout))

        self._planner_thread = threading.Thread(target=self._planner_loop, daemon=True)
        self._planner_thread.start()

    def print_help(self):
        print("[INFO] NavDP point-goal control")
        print(f"[INFO]   server            : {self._args.navdp_url}")
        print(f"[INFO]   fallback          : {self._args.navdp_fallback}")
        print(f"[INFO]   goal frame        : {self._args.nav_goal_frame}")
        print(f"[INFO]   goal             : ({self._args.nav_goal_x:.2f}, {self._args.nav_goal_y:.2f})")
        print(f"[INFO]   replan hz        : {self._args.navdp_replan_hz:.2f}")
        print(f"[INFO]   timeout          : {self._args.navdp_plan_timeout:.2f}s plan age")
        print(f"[INFO]   saturation       : vx={self._args.vx_max:.2f} vy={self._args.vy_max:.2f} wz={self._args.wz_max:.2f}")
        print(
            f"[INFO]   camera pitch     : {self._args.camera_pitch_deg:.1f} deg "
            f"(limits {self._args.camera_pitch_min_deg:.1f} .. {self._args.camera_pitch_max_deg:.1f})"
        )
        if self._args.camera_api_port > 0:
            print(
                "[INFO]   camera pitch api : "
                f"http://{self._args.camera_api_host}:{self._args.camera_api_port}/camera/pitch"
            )

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

        state = self._robot_state()
        self._goal_provider.bind_start_pose(state.base_pos_w, state.base_yaw)
        self._follower.reset()

    def reset(self):
        self._goal_provider.reset()
        self._follower.reset()
        with self._lock:
            self._planner_input = None
            self._planner_input_stamp = 0.0
            self._last_planned_input_stamp = 0.0
            self._latest_plan = None
            self._latest_world_path = np.zeros((0, 2), dtype=np.float32)
            self._latest_plan_time = 0.0
            self._planning_error = None
        self._algorithm_name = None
        self._last_capture_time = 0.0
        self._last_status_time = 0.0
        self._goal_done_reported = False
        if self._controller is not None:
            state = self._robot_state()
            self._goal_provider.bind_start_pose(state.base_pos_w, state.base_yaw)

    def _robot_state(self) -> RobotState2D:
        if self._controller is None:
            raise RuntimeError("NavDP controller must be bound before reading robot state.")
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

    def _maybe_capture_input(self):
        if self._sensor is None or self._controller is None:
            return
        now = time.monotonic()
        if (now - self._last_capture_time) < self._replan_interval:
            return

        frame = self._sensor.capture_frame()
        if frame is None:
            return
        state = self._robot_state()
        if self._goal_provider.is_done(state):
            return

        planner_input = PlannerInput(
            robot_state=state,
            goal_xy_body=self._goal_provider.current_goal_xy(state),
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            camera_pos_w=frame.camera_pos_w,
            camera_rot_w=frame.camera_rot_w,
            stamp_s=frame.stamp_s,
        )
        with self._lock:
            self._planner_input = planner_input
            self._planner_input_stamp = planner_input.stamp_s
        self._last_capture_time = now

    def _planner_loop(self):
        while self._running:
            try:
                if self._controller is None:
                    time.sleep(0.05)
                    continue

                with self._lock:
                    planner_input = self._planner_input
                    planner_input_stamp = self._planner_input_stamp

                if planner_input is None or planner_input_stamp <= self._last_planned_input_stamp:
                    time.sleep(0.01)
                    continue

                if self._algorithm_name is None:
                    self._algorithm_name = self._client.reset_pointgoal(
                        intrinsic=planner_input.intrinsic,
                        stop_threshold=self._args.navdp_stop_threshold,
                        batch_size=1,
                    )
                    status_message = self._client.consume_status_message()
                    if status_message:
                        print(f"[WARN] {status_message}")
                    print(f"[INFO] NavDP planner reset complete with algorithm: {self._algorithm_name}")

                if planner_input.goal_xy_body is None:
                    raise RuntimeError("Planner input is missing a point-goal payload.")
                plan = self._client.step_pointgoal(planner_input.goal_xy_body, planner_input.rgb, planner_input.depth)
                status_message = self._client.consume_status_message()
                if status_message:
                    print(f"[WARN] {status_message}")
                world_path = camera_plan_to_world_xy(
                    plan.trajectory_camera,
                    camera_pos_w=planner_input.camera_pos_w,
                    camera_rot_w=planner_input.camera_rot_w,
                )

                with self._lock:
                    self._latest_plan = plan
                    self._latest_world_path = world_path
                    self._latest_plan_time = time.monotonic()
                    self._planning_error = None
                    self._last_planned_input_stamp = planner_input_stamp
            except Exception as exc:  # pragma: no cover - depends on Isaac Sim and remote server
                with self._lock:
                    self._planning_error = str(exc)
                time.sleep(0.25)

    def _log_status(
        self,
        *,
        cmd: np.ndarray,
        goal_reached: bool,
        path_points: int,
        latest_plan_time: float,
        latest_plan_latency: float | None,
        planning_error: str | None,
    ):
        now = time.monotonic()
        if (now - self._last_status_time) < self._status_interval:
            return

        plan_age_text = "n/a"
        if latest_plan_time > 0.0:
            plan_age_text = f"{max(0.0, now - latest_plan_time):.2f}s"
        plan_latency_text = "n/a" if latest_plan_latency is None else f"{latest_plan_latency:.2f}s"
        error_text = "" if planning_error is None else f" error={planning_error}"
        state_text = "done" if goal_reached else "tracking"
        print(
            "[INFO] NavDP status: "
            f"state={state_text} plan_age={plan_age_text} plan_latency={plan_latency_text} "
            f"path_pts={path_points} cmd=({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}){error_text}"
        )
        self._last_status_time = now

    def command(self) -> np.ndarray:
        if self._controller is None:
            return np.zeros(3, dtype=np.float32)

        if self._sensor is not None:
            self._sensor.apply_pending_pitch()
        self._maybe_capture_input()
        state = self._robot_state()
        goal_reached = self._goal_provider.is_done(state)
        if goal_reached:
            cmd = np.zeros(3, dtype=np.float32)
            if not self._goal_done_reported:
                print("[INFO] NavDP goal reached. Holding zero locomotion command.")
                self._goal_done_reported = True
            self._follower.reset()
            self._log_status(
                cmd=cmd,
                goal_reached=True,
                path_points=0,
                latest_plan_time=0.0,
                latest_plan_latency=None,
                planning_error=None,
            )
            return cmd
        self._goal_done_reported = False

        with self._lock:
            world_path = self._latest_world_path.copy()
            latest_plan = self._latest_plan
            latest_plan_time = self._latest_plan_time
            planning_error = self._planning_error
        latest_plan_latency = None if latest_plan is None else latest_plan.plan_time_s

        if planning_error is not None and len(world_path) == 0:
            now = time.monotonic()
            if (now - self._last_warning_time) > 1.0:
                print(f"[WARN] NavDP planner error: {planning_error}")
                self._last_warning_time = now
            self._follower.reset()
            cmd = np.zeros(3, dtype=np.float32)
            self._log_status(
                cmd=cmd,
                goal_reached=False,
                path_points=0,
                latest_plan_time=latest_plan_time,
                latest_plan_latency=latest_plan_latency,
                planning_error=planning_error,
            )
            return cmd

        if len(world_path) == 0 or (time.monotonic() - latest_plan_time) > self._plan_timeout:
            self._follower.reset()
            cmd = np.zeros(3, dtype=np.float32)
            self._log_status(
                cmd=cmd,
                goal_reached=False,
                path_points=len(world_path),
                latest_plan_time=latest_plan_time,
                latest_plan_latency=latest_plan_latency,
                planning_error=planning_error,
            )
            return cmd

        cmd = self._follower.compute(
            base_pos_w=state.base_pos_w,
            base_yaw=state.base_yaw,
            path_world_xy=world_path,
        )
        self._log_status(
            cmd=cmd,
            goal_reached=False,
            path_points=len(world_path),
            latest_plan_time=latest_plan_time,
            latest_plan_latency=latest_plan_latency,
            planning_error=planning_error,
        )
        return cmd

    def shutdown(self):
        self._running = False
        if self._planner_thread.is_alive():
            self._planner_thread.join(timeout=1.0)
        if self._camera_api_server is not None:
            self._camera_api_server.shutdown()
            self._camera_api_server = None
        if self._sensor is not None:
            self._sensor.shutdown()


class InternVlaNavDpController:
    """Drive NavDP with dynamic point-goals grounded from InternVLA System 2."""

    requires_render = True

    def __init__(self, args):
        self.quit_requested = False

        self._args = args
        NavDpClient.validate_runtime_dependencies()
        InternVlaNavClient.validate_runtime_dependencies()
        self._navdp_client = NavDpClient(
            server_url=args.navdp_url,
            timeout_s=args.navdp_timeout,
            fallback_mode=args.navdp_fallback,
        )
        self._system2_client = InternVlaNavClient(
            server_url=args.internvla_url,
            timeout_s=args.internvla_timeout,
        )
        self._follower = HolonomicPurePursuitFollower(
            lookahead_distance=args.lookahead_distance,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            smoothing_tau=args.cmd_smoothing_tau,
        )

        self._controller = None
        self._sensor: G1NavCameraSensor | None = None
        self._camera_api_server: CameraPitchApiServer | None = None
        self._command_api_server: RuntimeControlApiServer | None = None
        self._session_id = (
            str(args.internvla_session_id).strip()
            if args.internvla_session_id
            else f"g1-internvla-{int(time.time())}"
        )
        planner_base_url = str(getattr(args, "planner_base_url", "") or "").strip()
        planner_completion = (
            make_planner_task_frame_completion(
                planner_base_url,
                slot_id=int(getattr(args, "planner_task_frame_slot_id", 1)),
            )
            if planner_base_url
            else None
        )
        knowledge_runtime = create_knowledge_runtime(
            dsn=str(getattr(args, "knowledge_dsn", "") or ""),
            scene_scope=str(getattr(args, "knowledge_scene_scope", "") or "").strip() or None,
        )
        self._tasking = AuraTaskingAdapter(
            completion=planner_completion,
            model=str(getattr(args, "planner_model", "") or "").strip(),
            timeout=float(getattr(args, "planner_timeout", 120.0)),
            knowledge_runtime=knowledge_runtime,
            scene_scope=str(getattr(args, "knowledge_scene_scope", "") or "").strip() or None,
        )
        self._task_state: TaskExecutionState | None = None
        self._last_task_result: dict[str, object] | None = None
        self._memory_navigation_target: dict[str, object] | None = None
        self._memory_reacquire_state: str | None = None

        self._running = True
        self._lock = threading.Lock()
        self._pipeline_state = make_navigation_pipeline_state(
            instruction=str(args.nav_instruction).strip(),
            language=str(args.nav_instruction_language).strip() or "auto",
            tolerance=args.nav_goal_tolerance,
        )
        self._status_interval = 1.0
        self._replan_interval = 1.0 / max(0.1, float(args.navdp_replan_hz))
        self._plan_timeout = max(0.1, float(args.navdp_plan_timeout))
        self._hold_last_plan_timeout = max(self._plan_timeout, float(args.navdp_hold_last_plan_timeout))
        self._goal_update_min_dist = max(0.0, float(args.internvla_goal_update_min_dist))
        self._pixel_goal_update_min_px = max(6.0, 0.02 * max(float(args.camera_width), float(args.camera_height)))
        self._goal_filter_alpha = float(np.clip(float(args.internvla_goal_filter_alpha), 0.0, 1.0))
        self._goal_confirm_samples = max(1, int(args.internvla_goal_confirm_samples))
        self._goal_min_stable_time = max(0.0, float(args.internvla_goal_min_stable_time))
        self._forward_step_m = max(0.05, float(args.internvla_forward_step_m))
        self._turn_step_rad = float(np.deg2rad(max(1.0, float(args.internvla_turn_step_deg))))
        self._action_timeout_s = max(0.1, float(args.internvla_action_timeout_s))

        self._system2_thread = threading.Thread(target=self._system2_loop, daemon=True, name="internvla-system2-loop")
        self._navdp_thread = threading.Thread(target=self._navdp_loop, daemon=True, name="internvla-navdp-loop")
        self._task_thread = threading.Thread(target=self._task_loop, daemon=True, name="planner-task-loop")
        self._system2_thread.start()
        self._navdp_thread.start()
        self._task_thread.start()
        if str(args.nav_instruction).strip():
            self.apply_runtime_command(args.nav_instruction, args.nav_instruction_language)

    def print_help(self):
        with self._lock:
            command_state = self._pipeline_state.command
        print("[INFO] InternVLA -> NavDP control")
        print(f"[INFO]   instruction       : {command_state.instruction}")
        print(f"[INFO]   language          : {command_state.language}")
        print(f"[INFO]   system2 server    : {self._args.internvla_url}")
        print(f"[INFO]   system2 timeout   : {self._args.internvla_timeout:.2f}s")
        print(f"[INFO]   session id        : {self._session_id}")
        print(f"[INFO]   planner endpoint  : {self._args.planner_base_url or 'deterministic_only'}")
        print(f"[INFO]   planner model     : {self._args.planner_model}")
        print(f"[INFO]   navdp server      : {self._args.navdp_url}")
        print(f"[INFO]   navdp fallback    : {self._args.navdp_fallback}")
        print(f"[INFO]   replan hz         : {self._args.navdp_replan_hz:.2f}")
        print(f"[INFO]   plan timeout      : {self._args.navdp_plan_timeout:.2f}s")
        print(f"[INFO]   stale hold        : {self._hold_last_plan_timeout:.2f}s")
        print(
            f"[INFO]   goal depth window : {self._args.internvla_goal_depth_window} "
            f"(valid {self._args.internvla_goal_depth_min:.2f}m .. {self._args.internvla_goal_depth_max:.2f}m)"
        )
        print(f"[INFO]   goal update min   : {self._goal_update_min_dist:.2f}m")
        print(
            f"[INFO]   goal stabilizer   : alpha={self._goal_filter_alpha:.2f} "
            f"samples={self._goal_confirm_samples} stable_time={self._goal_min_stable_time:.2f}s"
        )
        print(f"[INFO]   pixel goal jitter : {self._pixel_goal_update_min_px:.1f}px min change")
        print(
            f"[INFO]   saturation        : vx={self._args.vx_max:.2f} "
            f"vy={self._args.vy_max:.2f} wz={self._args.wz_max:.2f}"
        )
        print(
            f"[INFO]   direct actions    : forward={self._forward_step_m:.2f}m "
            f"turn={np.rad2deg(self._turn_step_rad):.1f}deg timeout={self._action_timeout_s:.2f}s"
        )
        print(
            f"[INFO]   camera pitch      : {self._args.camera_pitch_deg:.1f} deg "
            f"(limits {self._args.camera_pitch_min_deg:.1f} .. {self._args.camera_pitch_max_deg:.1f})"
        )
        if self._args.camera_api_port > 0:
            print(
                "[INFO]   camera pitch api  : "
                f"http://{self._args.camera_api_host}:{self._args.camera_api_port}/camera/pitch"
            )
        if self._args.runtime_control_api_port > 0:
            print(
                "[INFO]   runtime control   : "
                f"http://{self._args.runtime_control_api_host}:{self._args.runtime_control_api_port}/runtime/status"
            )

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
        if self._args.runtime_control_api_port > 0 and self._command_api_server is None:
            self._command_api_server = RuntimeControlApiServer(
                host=self._args.runtime_control_api_host,
                port=self._args.runtime_control_api_port,
                command_handler=self,
                camera_sensor=self._sensor,
            )
            self._command_api_server.start()

        with self._lock:
            self._pipeline_state.follower = make_follower_state()
            self._set_locomotion_locked(_zero_command(), state_label="waiting", stamp_s=time.monotonic())
            self._pipeline_state.status.goal_done_reported = False
            self._pipeline_state.system2.session_reset_required = True

    def reset(self):
        with self._lock:
            current_state = self._pipeline_state
            generation = current_state.goal.generation + 1
            self._pipeline_state = self._make_reset_pipeline_state(
                instruction=current_state.command.instruction,
                language=current_state.command.language,
                command_revision=current_state.command.command_revision,
                goal_generation=generation,
                clear_reason="reset",
            )

    def _make_reset_pipeline_state(
        self,
        *,
        instruction: str,
        language: str,
        command_revision: int,
        goal_generation: int,
        clear_reason: str,
    ) -> NavigationPipelineState:
        pipeline_state = make_navigation_pipeline_state(
            instruction=instruction,
            language=language,
            tolerance=self._args.nav_goal_tolerance,
        )
        pipeline_state.command.command_revision = int(command_revision)
        pipeline_state.goal.generation = int(goal_generation)
        pipeline_state.goal.last_clear_reason = str(clear_reason)
        return pipeline_state

    def _clear_latest_plan_locked(self):
        self._pipeline_state.navdp.latest_plan = None
        self._pipeline_state.navdp.latest_world_path = _empty_world_path()
        self._pipeline_state.navdp.latest_plan_time = 0.0

    def _clear_navdp_locked(self, *, reset_algorithm: bool, reset_progress: bool):
        self._clear_latest_plan_locked()
        self._pipeline_state.navdp.error = None
        self._pipeline_state.navdp.last_discard_reason = None
        if reset_algorithm:
            self._pipeline_state.navdp.algorithm_name = None
            self._pipeline_state.navdp.last_request_input_stamp = 0.0
            self._pipeline_state.navdp.last_request_goal_generation = -1
            self._pipeline_state.navdp.last_request_started_at_s = 0.0
            self._pipeline_state.navdp.last_committed_goal_generation = -1
            self._pipeline_state.navdp.last_discarded_goal_generation = -1
        if reset_progress:
            self._pipeline_state.navdp.last_input_stamp = 0.0
            self._pipeline_state.navdp.last_goal_generation = -1
            self._pipeline_state.navdp.last_request_input_stamp = 0.0
            self._pipeline_state.navdp.last_request_goal_generation = -1
            self._pipeline_state.navdp.last_request_started_at_s = 0.0
            self._pipeline_state.navdp.last_committed_goal_generation = -1
            self._pipeline_state.navdp.last_discarded_goal_generation = -1

    def _clear_goal_candidate_locked(self):
        goal_state = self._pipeline_state.goal
        goal_state.candidate_kind = "none"
        goal_state.raw_candidate_world_xy = None
        goal_state.filtered_candidate_world_xy = None
        goal_state.raw_candidate_pixel_xy = None
        goal_state.filtered_candidate_pixel_xy = None
        goal_state.candidate_started_at_s = 0.0
        goal_state.candidate_last_stamp_s = 0.0
        goal_state.candidate_sample_count = 0

    def _clear_action_override_locked(self):
        self._pipeline_state.action_override = ActionOverrideState()

    def _current_target_pitch_deg(self) -> float:
        sensor = self._sensor
        if sensor is None:
            return float(self._args.camera_pitch_deg)
        try:
            status = sensor.pitch_status()
        except Exception:
            return float(self._args.camera_pitch_deg)
        return float(status.get("target_pitch_deg", self._args.camera_pitch_deg))

    def _start_action_override_locked(
        self,
        mode: str,
        robot_state: RobotState2D,
        *,
        pending_modes: tuple[str, ...] = (),
    ):
        base_pos_xy = np.asarray(robot_state.base_pos_w, dtype=np.float32)[:2].copy()
        action_state = ActionOverrideState(
            mode=str(mode),
            pending_modes=tuple(str(item) for item in pending_modes),
            started_at_s=time.monotonic(),
            start_pos_xy=base_pos_xy,
            start_yaw=float(robot_state.base_yaw),
            progress=0.0,
        )
        if mode == "forward":
            action_state.target_distance_m = float(self._forward_step_m)
        elif mode in {"yaw_left", "yaw_right"}:
            action_state.target_yaw_rad = float(self._turn_step_rad)
        self._pipeline_state.action_override = action_state

    def _begin_goal_candidate_locked(self, world_xy: np.ndarray, stamp_s: float):
        world = np.asarray(world_xy, dtype=np.float32).reshape(2)
        goal_state = self._pipeline_state.goal
        goal_state.candidate_kind = "point"
        goal_state.raw_candidate_world_xy = world.copy()
        goal_state.filtered_candidate_world_xy = world.copy()
        goal_state.raw_candidate_pixel_xy = None
        goal_state.filtered_candidate_pixel_xy = None
        goal_state.candidate_started_at_s = float(stamp_s)
        goal_state.candidate_last_stamp_s = float(stamp_s)
        goal_state.candidate_sample_count = 1

    def _update_goal_candidate_locked(self, world_xy: np.ndarray, stamp_s: float):
        world = np.asarray(world_xy, dtype=np.float32).reshape(2)
        goal_state = self._pipeline_state.goal
        if goal_state.candidate_kind != "point" or goal_state.filtered_candidate_world_xy is None:
            self._begin_goal_candidate_locked(world, stamp_s)
            return
        filtered = ((1.0 - self._goal_filter_alpha) * goal_state.filtered_candidate_world_xy) + (
            self._goal_filter_alpha * world
        )
        goal_state.raw_candidate_world_xy = world.copy()
        goal_state.filtered_candidate_world_xy = np.asarray(filtered, dtype=np.float32)
        goal_state.candidate_last_stamp_s = float(stamp_s)
        goal_state.candidate_sample_count += 1

    def _begin_pixel_goal_candidate_locked(self, pixel_xy: np.ndarray, stamp_s: float):
        pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
        goal_state = self._pipeline_state.goal
        goal_state.candidate_kind = "pixel"
        goal_state.raw_candidate_world_xy = None
        goal_state.filtered_candidate_world_xy = None
        goal_state.raw_candidate_pixel_xy = pixel.copy()
        goal_state.filtered_candidate_pixel_xy = pixel.copy()
        goal_state.candidate_started_at_s = float(stamp_s)
        goal_state.candidate_last_stamp_s = float(stamp_s)
        goal_state.candidate_sample_count = 1

    def _update_pixel_goal_candidate_locked(self, pixel_xy: np.ndarray, stamp_s: float):
        pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
        goal_state = self._pipeline_state.goal
        if goal_state.candidate_kind != "pixel" or goal_state.filtered_candidate_pixel_xy is None:
            self._begin_pixel_goal_candidate_locked(pixel, stamp_s)
            return
        filtered = ((1.0 - self._goal_filter_alpha) * goal_state.filtered_candidate_pixel_xy) + (
            self._goal_filter_alpha * pixel
        )
        goal_state.raw_candidate_pixel_xy = pixel.copy()
        goal_state.filtered_candidate_pixel_xy = np.asarray(filtered, dtype=np.float32)
        goal_state.candidate_last_stamp_s = float(stamp_s)
        goal_state.candidate_sample_count += 1

    def _goal_candidate_ready_locked(self) -> bool:
        goal_state = self._pipeline_state.goal
        if goal_state.candidate_kind == "point" and goal_state.filtered_candidate_world_xy is None:
            return False
        if goal_state.candidate_kind == "pixel" and goal_state.filtered_candidate_pixel_xy is None:
            return False
        if goal_state.candidate_kind not in {"point", "pixel"}:
            return False
        stable_time = max(0.0, float(goal_state.candidate_last_stamp_s - goal_state.candidate_started_at_s))
        return (
            goal_state.candidate_sample_count >= self._goal_confirm_samples
            or stable_time >= self._goal_min_stable_time
        )

    def _observe_projected_goal_locked(self, world_xy: np.ndarray, stamp_s: float) -> np.ndarray | None:
        world = np.asarray(world_xy, dtype=np.float32).reshape(2)
        goal_state = self._pipeline_state.goal
        active_goal = goal_state.target_world_xy if goal_target_mode(goal_state) == "point" else None

        if active_goal is not None:
            active_distance = float(np.linalg.norm(world - np.asarray(active_goal, dtype=np.float32).reshape(2)))
            if active_distance < self._goal_update_min_dist:
                self._clear_goal_candidate_locked()
                return None

        candidate_goal = goal_state.filtered_candidate_world_xy
        if candidate_goal is None:
            self._begin_goal_candidate_locked(world, stamp_s)
        else:
            candidate_distance = float(np.linalg.norm(world - np.asarray(candidate_goal, dtype=np.float32).reshape(2)))
            if candidate_distance >= self._goal_update_min_dist:
                self._begin_goal_candidate_locked(world, stamp_s)
            else:
                self._update_goal_candidate_locked(world, stamp_s)

        if not self._goal_candidate_ready_locked():
            return None

        stabilized_goal = goal_state.filtered_candidate_world_xy
        if stabilized_goal is None:
            return None
        if active_goal is not None:
            active_distance = float(np.linalg.norm(stabilized_goal - np.asarray(active_goal, dtype=np.float32).reshape(2)))
            if active_distance < self._goal_update_min_dist:
                self._clear_goal_candidate_locked()
                return None

        committed_goal = np.asarray(stabilized_goal, dtype=np.float32).reshape(2).copy()
        self._update_goal(committed_goal, stamp_s)
        return committed_goal

    def _observe_pixel_goal_locked(self, pixel_xy: np.ndarray, stamp_s: float) -> np.ndarray | None:
        pixel = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
        goal_state = self._pipeline_state.goal
        active_goal = goal_state.target_pixel_xy if goal_target_mode(goal_state) == "pixel" else None

        if active_goal is None:
            committed_goal = np.asarray(np.rint(pixel), dtype=np.float32).reshape(2)
            self._update_pixel_goal(committed_goal, stamp_s)
            return committed_goal

        if active_goal is not None:
            active_distance = float(np.linalg.norm(pixel - np.asarray(active_goal, dtype=np.float32).reshape(2)))
            if active_distance < self._pixel_goal_update_min_px:
                self._clear_goal_candidate_locked()
                return None

        candidate_goal = goal_state.filtered_candidate_pixel_xy if goal_state.candidate_kind == "pixel" else None
        if candidate_goal is None:
            self._begin_pixel_goal_candidate_locked(pixel, stamp_s)
        else:
            candidate_distance = float(np.linalg.norm(pixel - np.asarray(candidate_goal, dtype=np.float32).reshape(2)))
            if candidate_distance >= self._pixel_goal_update_min_px:
                self._begin_pixel_goal_candidate_locked(pixel, stamp_s)
            else:
                self._update_pixel_goal_candidate_locked(pixel, stamp_s)

        if not self._goal_candidate_ready_locked():
            return None

        stabilized_goal = goal_state.filtered_candidate_pixel_xy
        if stabilized_goal is None:
            return None
        if active_goal is not None:
            active_distance = float(np.linalg.norm(stabilized_goal - np.asarray(active_goal, dtype=np.float32).reshape(2)))
            if active_distance < self._pixel_goal_update_min_px:
                self._clear_goal_candidate_locked()
                return None

        committed_goal = np.asarray(np.rint(stabilized_goal), dtype=np.float32).reshape(2)
        self._update_pixel_goal(committed_goal, stamp_s)
        return committed_goal

    def _set_locomotion_locked(self, command: np.ndarray, *, state_label: str, stamp_s: float):
        self._pipeline_state.locomotion = LocomotionState(
            command=np.asarray(command, dtype=np.float32).reshape(3).copy(),
            state_label=str(state_label),
            last_command_stamp=float(stamp_s),
        )

    def _combined_error_locked(self, state: NavigationPipelineState | None = None) -> str | None:
        pipeline_state = self._pipeline_state if state is None else state
        errors = [error for error in (pipeline_state.system2.error, pipeline_state.navdp.error) if error]
        if not errors:
            return None
        return " | ".join(errors)

    def _update_goal(self, world_xy: np.ndarray, stamp_s: float):
        self._clear_goal_candidate_locked()
        self._pipeline_state.goal.target_mode = "point"
        self._pipeline_state.goal.target_world_xy = np.asarray(world_xy, dtype=np.float32).reshape(2)
        self._pipeline_state.goal.target_pixel_xy = None
        self._pipeline_state.goal.last_update_stamp_s = float(stamp_s)
        self._pipeline_state.goal.last_clear_reason = ""
        self._pipeline_state.goal.generation += 1

    def _update_pixel_goal(self, pixel_xy: np.ndarray, stamp_s: float):
        self._clear_goal_candidate_locked()
        self._pipeline_state.goal.target_mode = "pixel"
        self._pipeline_state.goal.target_pixel_xy = np.asarray(pixel_xy, dtype=np.float32).reshape(2)
        self._pipeline_state.goal.target_world_xy = None
        self._pipeline_state.goal.last_update_stamp_s = float(stamp_s)
        self._pipeline_state.goal.last_clear_reason = ""
        self._pipeline_state.goal.generation += 1

    def _clear_goal(self, reason: str):
        self._clear_goal_candidate_locked()
        self._pipeline_state.goal.target_mode = "none"
        self._pipeline_state.goal.target_world_xy = None
        self._pipeline_state.goal.target_pixel_xy = None
        self._pipeline_state.goal.last_clear_reason = str(reason)
        self._pipeline_state.goal.generation += 1

    def _robot_state(self) -> RobotState2D:
        if self._controller is None:
            raise RuntimeError("InternVLA/NavDP controller must be bound before reading robot state.")
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

    def _maybe_capture_input(self):
        if self._sensor is None or self._controller is None:
            return
        now = time.monotonic()
        with self._lock:
            last_capture_time = self._pipeline_state.capture.last_capture_time
        if (now - last_capture_time) < self._replan_interval:
            return

        frame = self._sensor.capture_frame()
        if frame is None:
            return

        planner_input = PlannerInput(
            robot_state=self._robot_state(),
            goal_xy_body=None,
            rgb=frame.rgb,
            depth=frame.depth,
            intrinsic=frame.intrinsic,
            camera_pos_w=frame.camera_pos_w,
            camera_rot_w=frame.camera_rot_w,
            stamp_s=frame.stamp_s,
        )
        with self._lock:
            self._pipeline_state.capture.latest_input = planner_input
            self._pipeline_state.capture.last_capture_time = now

    def _reset_remote_session(self):
        with self._lock:
            instruction = self._pipeline_state.command.instruction
            language = self._pipeline_state.command.language
        self._system2_client.reset_session(
            session_id=self._session_id,
            instruction=instruction,
            language=language,
            image_width=self._args.camera_width,
            image_height=self._args.camera_height,
        )
        print(
            "[INFO] InternVLA session reset complete: "
            f"{self._session_id} instruction={instruction!r} language={language}"
        )

    def _capture_origin_pose(self) -> dict[str, object] | None:
        if self._controller is None:
            return None
        state = self._robot_state()
        return {
            "world_xy": [float(state.base_pos_w[0]), float(state.base_pos_w[1])],
            "yaw_rad": float(state.base_yaw),
        }

    def _current_subgoal_locked(self) -> dict[str, object] | None:
        task = self._task_state
        if task is None:
            return None
        for subgoal in task.subgoals:
            if subgoal["status"] in {"pending", "running"}:
                return subgoal
        return None

    def _current_subgoal_type_locked(self) -> str | None:
        subgoal = self._current_subgoal_locked()
        if subgoal is None:
            return None
        return str(subgoal["type"])

    def _serialize_task_state_locked(self, task: TaskExecutionState | None) -> dict[str, object] | None:
        if task is None:
            return None
        payload = asdict(task)
        payload["subgoals"] = [dict(item) for item in task.subgoals]
        return payload

    def _archive_active_task_locked(self) -> None:
        if self._task_state is None:
            return
        self._last_task_result = self._serialize_task_state_locked(self._task_state)
        self._task_state = None

    def _apply_navigation_command_internal(
        self,
        instruction: str,
        language: str,
        *,
        clear_reason: str,
    ) -> dict[str, object]:
        next_instruction = str(instruction).strip()
        if not next_instruction:
            raise ValueError("instruction must be a non-empty string")
        next_language = str(language).strip() or "en"
        with self._lock:
            current_state = self._pipeline_state
            command_revision = current_state.command.command_revision + 1
            goal_generation = current_state.goal.generation + 1
            self._pipeline_state = self._make_reset_pipeline_state(
                instruction=next_instruction,
                language=next_language,
                command_revision=command_revision,
                goal_generation=goal_generation,
                clear_reason=clear_reason,
            )
            self._memory_navigation_target = None
            self._memory_reacquire_state = None
        return {
            "instruction": next_instruction,
            "language": next_language,
            "command_revision": command_revision,
            "session_id": self._session_id,
            "session_reset_required": True,
        }

    def start_navigation_instruction(self, instruction: str, language: str = "en") -> dict[str, object]:
        return self._apply_navigation_command_internal(
            instruction,
            language,
            clear_reason="task_navigate",
        )

    def start_navigation_memory_goal(self, navigation_target: dict[str, object]) -> dict[str, object]:
        world_pose_xyz = np.asarray(navigation_target["world_pose_xyz"], dtype=np.float32).reshape(3)
        now = time.monotonic()
        with self._lock:
            current_state = self._pipeline_state
            command_revision = current_state.command.command_revision + 1
            goal_generation = current_state.goal.generation + 1
            self._pipeline_state = self._make_reset_pipeline_state(
                instruction=f"go to remembered {navigation_target.get('class_name') or 'target'}",
                language="en",
                command_revision=command_revision,
                goal_generation=goal_generation,
                clear_reason="task_memory_navigate",
            )
            self._pipeline_state.system2.session_reset_required = False
            self._pipeline_state.system2.last_signature = None
            self._pipeline_state.system2.last_result = None
            self._pipeline_state.system2.error = None
            self._clear_action_override_locked()
            self._clear_navdp_locked(reset_algorithm=True, reset_progress=False)
            self._pipeline_state.follower = make_follower_state()
            self._set_locomotion_locked(_zero_command(), state_label="waiting", stamp_s=now)
            self._update_goal(np.asarray(world_pose_xyz[:2], dtype=np.float32), now)
            self._memory_navigation_target = dict(navigation_target)
            self._memory_reacquire_state = "approach_memory_pose"
        return {
            "instruction": f"go to remembered {navigation_target.get('class_name') or 'target'}",
            "language": "en",
            "command_revision": command_revision,
            "session_id": self._session_id,
            "session_reset_required": False,
            "goal_world_xy": [float(world_pose_xyz[0]), float(world_pose_xyz[1])],
        }

    def start_return_to_origin(self, origin_pose: dict[str, object]) -> dict[str, object]:
        target_world_xy = np.asarray(origin_pose["world_xy"], dtype=np.float32).reshape(2)
        now = time.monotonic()
        with self._lock:
            current_state = self._pipeline_state
            command_revision = current_state.command.command_revision + 1
            goal_generation = current_state.goal.generation + 1
            self._pipeline_state = self._make_reset_pipeline_state(
                instruction="return to origin",
                language="en",
                command_revision=command_revision,
                goal_generation=goal_generation,
                clear_reason="task_return",
            )
            self._pipeline_state.system2.session_reset_required = False
            self._pipeline_state.system2.last_signature = None
            self._pipeline_state.system2.last_result = None
            self._pipeline_state.system2.error = None
            self._clear_action_override_locked()
            self._clear_navdp_locked(reset_algorithm=True, reset_progress=False)
            self._pipeline_state.follower = make_follower_state()
            self._set_locomotion_locked(_zero_command(), state_label="waiting", stamp_s=now)
            self._update_goal(target_world_xy, now)
            self._memory_navigation_target = None
            self._memory_reacquire_state = None
        return {
            "instruction": "return to origin",
            "language": "en",
            "command_revision": command_revision,
            "session_id": self._session_id,
            "session_reset_required": False,
            "goal_world_xy": [float(target_world_xy[0]), float(target_world_xy[1])],
        }

    def navigation_snapshot(self, *, origin_pose: dict[str, object] | None = None) -> dict[str, object]:
        with self._lock:
            pipeline_state = self._pipeline_state
            goal_mode = goal_target_mode(pipeline_state.goal)
            goal_world_xy = goal_target_world_xy(pipeline_state.goal)
            goal_pixel_xy = goal_target_pixel_xy(pipeline_state.goal)
            last_result = pipeline_state.system2.last_result
            locomotion = pipeline_state.locomotion.command.copy()
            state_label = pipeline_state.locomotion.state_label
            action_override_mode = pipeline_state.action_override.mode
            tolerance = float(pipeline_state.goal.tolerance)
            memory_navigation_target = None if self._memory_navigation_target is None else dict(self._memory_navigation_target)
            memory_reacquire_state = self._memory_reacquire_state
        goal_reached = False
        return_pose_distance = None
        return_pose_reached = False
        robot_state = None if self._controller is None else self._robot_state()
        if robot_state is not None and goal_mode == "point" and goal_world_xy is not None:
            goal_reached = bool(
                float(np.linalg.norm(goal_world_xy - np.asarray(robot_state.base_pos_w, dtype=np.float32)[:2])) <= tolerance
            )
        if origin_pose is not None and robot_state is not None:
            origin_xy = np.asarray(origin_pose["world_xy"], dtype=np.float32).reshape(2)
            return_pose_distance = float(
                np.linalg.norm(origin_xy - np.asarray(robot_state.base_pos_w, dtype=np.float32)[:2])
            )
            return_pose_reached = return_pose_distance <= tolerance
        snapshot: dict[str, object] = {
            "planner_target_mode": goal_mode,
            "has_goal": goal_mode in {"point", "pixel"},
            "goal_world_xy": None if goal_world_xy is None else [float(goal_world_xy[0]), float(goal_world_xy[1])],
            "goal_pixel_xy": None
            if goal_pixel_xy is None
            else [int(round(float(goal_pixel_xy[0]))), int(round(float(goal_pixel_xy[1])))],
            "system2_status": None if last_result is None else last_result.status,
            "system2_decision_mode": None if last_result is None else last_result.decision_mode,
            "system2_text": None if last_result is None else last_result.text,
            "action_override_mode": action_override_mode,
            "locomotion_command": [float(locomotion[0]), float(locomotion[1]), float(locomotion[2])],
            "state_label": state_label,
            "goal_reached": goal_reached,
            "return_pose_distance": return_pose_distance,
            "return_pose_reached": return_pose_reached,
            "memory_navigation_mode": None if memory_navigation_target is None else "memory_pose",
            "resolved_memory_object_id": None if memory_navigation_target is None else memory_navigation_target.get("object_id"),
            "resolved_memory_pose_age_sec": None if memory_navigation_target is None else memory_navigation_target.get("pose_age_sec"),
            "reacquire_state": (
                "reacquire_pending"
                if memory_navigation_target is not None and goal_reached and memory_reacquire_state == "approach_memory_pose"
                else memory_reacquire_state
            ),
        }
        return snapshot

    def check_binary_question(self, question: str) -> str:
        return self._system2_client.check_answer(question)

    def set_last_report(self, message: str) -> None:
        with self._lock:
            if self._task_state is not None:
                self._task_state.last_report = str(message)

    def _step_task_once(self) -> bool:
        with self._lock:
            task = self._task_state
            if task is None or task.status in {"succeeded", "failed", "cancelled"}:
                return False
            if task.origin_pose is None:
                task.origin_pose = self._capture_origin_pose()
            runtime = {
                "controller": self,
                "task_state": task,
                "scene_preset": str(getattr(self._args, "knowledge_scene_scope", "") or "").strip() or None,
            }
        event = self._tasking.step(task.subgoals, runtime)
        if event is None:
            return False
        with self._lock:
            if self._task_state is None or self._task_state.task_id != task.task_id:
                return False
            task = self._task_state
            if event["type"] == "report":
                raw_output = event["raw_output"]
                message = raw_output.get("message")
                if isinstance(message, str) and message.strip():
                    task.last_report = message
            current_index = 0
            for idx, subgoal in enumerate(task.subgoals):
                if subgoal["status"] in {"pending", "running"}:
                    current_index = idx
                    break
            else:
                current_index = max(0, len(task.subgoals) - 1)
            task.current_subgoal_index = current_index

            if any(subgoal["status"] == "failed" for subgoal in task.subgoals):
                task.status = "failed"
                task.failure_reason = next(
                    (subgoal.get("failure_reason") for subgoal in task.subgoals if subgoal["status"] == "failed"),
                    None,
                )
                task.finished_at = time.monotonic()
                self._last_task_result = self._serialize_task_state_locked(task)
            elif all(subgoal["status"] == "succeeded" for subgoal in task.subgoals):
                task.status = "succeeded"
                task.finished_at = time.monotonic()
                self._last_task_result = self._serialize_task_state_locked(task)
            else:
                task.status = "running"
        return True

    def _task_loop(self):
        while self._running:
            try:
                stepped = self._step_task_once()
            except Exception as exc:  # pragma: no cover - depends on live runtime and remote servers
                with self._lock:
                    if self._task_state is not None:
                        self._task_state.status = "failed"
                        self._task_state.failure_reason = f"{type(exc).__name__}: {exc}"
                        self._task_state.finished_at = time.monotonic()
                        self._last_task_result = self._serialize_task_state_locked(self._task_state)
                stepped = False
            time.sleep(0.05 if stepped else 0.1)

    def command_api_status(self) -> dict[str, object]:
        with self._lock:
            pipeline_state = self._pipeline_state
            planner_target_mode = goal_target_mode(pipeline_state.goal)
            goal_world_xy = goal_target_world_xy(pipeline_state.goal)
            goal_pixel_xy = goal_target_pixel_xy(pipeline_state.goal)
            pending_goal_world_xy = goal_pending_world_xy(pipeline_state.goal)
            pending_goal_pixel_xy = goal_pending_pixel_xy(pipeline_state.goal)
            has_goal = goal_world_xy is not None or goal_pixel_xy is not None
            instruction = pipeline_state.command.instruction
            language = pipeline_state.command.language
            command_revision = pipeline_state.command.command_revision
            session_reset_required = pipeline_state.system2.session_reset_required
            planning_error = self._combined_error_locked(pipeline_state)
            last_goal_update_stamp_s = pipeline_state.goal.last_update_stamp_s
            last_goal_clear_reason = pipeline_state.goal.last_clear_reason
            goal_candidate_sample_count = pipeline_state.goal.candidate_sample_count
            last_system2_result = serialize_system2_result(pipeline_state.system2.last_result)
            last_discard_reason = pipeline_state.navdp.last_discard_reason
            system2_decision_mode = None if pipeline_state.system2.last_result is None else pipeline_state.system2.last_result.decision_mode
            action_override_mode = pipeline_state.action_override.mode
            action_progress = pipeline_state.action_override.progress if action_override_mode is not None else None
            pending_action_modes = list(pipeline_state.action_override.pending_modes)
            navdp_supports_pixelgoal = self._navdp_client.supports_pixelgoal
            action_only_suppressed = bool(pipeline_state.status.action_only_suppressed)
            last_action_only_mode = pipeline_state.status.last_action_only_mode
            task_state = self._task_state
            last_task_result = self._last_task_result
            memory_navigation_target = None if self._memory_navigation_target is None else dict(self._memory_navigation_target)
            memory_reacquire_state = self._memory_reacquire_state
        camera_target_pitch_deg = self._current_target_pitch_deg()
        payload: dict[str, object] = {
            "instruction": instruction,
            "language": language,
            "command_revision": command_revision,
            "session_id": self._session_id,
            "session_reset_required": session_reset_required,
            "has_goal": has_goal,
            "last_goal_update_stamp_s": last_goal_update_stamp_s,
            "last_goal_clear_reason": last_goal_clear_reason,
            "planning_error": planning_error,
            "goal_candidate_sample_count": goal_candidate_sample_count,
            "last_discard_reason": last_discard_reason,
            "system2_last_result": last_system2_result,
            "system2_raw_output_text": None if last_system2_result is None else last_system2_result["text"],
            "planner_target_mode": planner_target_mode,
            "system2_decision_mode": system2_decision_mode,
            "action_override_mode": action_override_mode,
            "action_progress": action_progress,
            "pending_action_modes": pending_action_modes,
            "look_down_active": False,
            "camera_target_pitch_deg": camera_target_pitch_deg,
            "pending_force_infer": False,
            "navdp_supports_pixelgoal": navdp_supports_pixelgoal,
            "action_only_suppressed": action_only_suppressed,
            "last_action_only_mode": last_action_only_mode,
            "memoryAwareTaskActive": memory_navigation_target is not None,
            "memoryNavigationMode": None if memory_navigation_target is None else "memory_pose",
            "resolvedMemoryObjectId": None if memory_navigation_target is None else memory_navigation_target.get("object_id"),
            "resolvedMemoryPoseAgeSec": None if memory_navigation_target is None else memory_navigation_target.get("pose_age_sec"),
            "reacquireState": memory_reacquire_state,
        }
        if goal_world_xy is not None:
            payload["goal_world_xy"] = [float(goal_world_xy[0]), float(goal_world_xy[1])]
            payload["active_goal_world_xy"] = [float(goal_world_xy[0]), float(goal_world_xy[1])]
            payload["has_goal"] = True
        if goal_pixel_xy is not None:
            payload["active_pixel_goal_xy"] = [int(round(float(goal_pixel_xy[0]))), int(round(float(goal_pixel_xy[1])))]
            payload["has_goal"] = True
        if pending_goal_world_xy is not None:
            payload["pending_goal_world_xy"] = [float(pending_goal_world_xy[0]), float(pending_goal_world_xy[1])]
        if pending_goal_pixel_xy is not None:
            payload["pending_pixel_goal_xy"] = [
                int(round(float(pending_goal_pixel_xy[0]))),
                int(round(float(pending_goal_pixel_xy[1]))),
            ]
        if task_state is not None:
            current_subgoal = None
            if 0 <= task_state.current_subgoal_index < len(task_state.subgoals):
                current_subgoal = dict(task_state.subgoals[task_state.current_subgoal_index])
            payload.update(
                {
                    "task_id": task_state.task_id,
                    "task_status": task_state.status,
                    "task_frame": dict(task_state.task_frame),
                    "subgoals": [dict(item) for item in task_state.subgoals],
                    "current_subgoal": current_subgoal,
                    "last_report": task_state.last_report,
                }
            )
        if last_task_result is not None:
            payload["last_task_result"] = dict(last_task_result)
        return payload

    def apply_runtime_command(self, instruction: str, language: str | None = None) -> dict[str, object]:
        next_instruction = str(instruction).strip()
        if not next_instruction:
            raise ValueError("instruction must be a non-empty string")
        next_language = str(language).strip() if language is not None and str(language).strip() else None
        immediate_stop = bool(RUNTIME_STOP_PATTERN.fullmatch(next_instruction))
        current_language = next_language or self._pipeline_state.command.language

        if immediate_stop:
            with self._lock:
                if self._task_state is not None:
                    self._task_state.status = "cancelled"
                    self._task_state.failure_reason = "user_stop"
                    self._task_state.finished_at = time.monotonic()
                    self._last_task_result = self._serialize_task_state_locked(self._task_state)
                    self._task_state = None
            response = self._apply_navigation_command_internal("stop", current_language, clear_reason="runtime_stop")
            response["immediate_stop"] = True
            response["task_status"] = "cancelled"
            print("[INFO] Runtime navigation command accepted: STOP")
            return response

        task_frame = self._tasking.plan_task_frame(
            next_instruction,
            planning_context={
                "scene_preset": str(getattr(self._args, "knowledge_scene_scope", "") or "").strip() or None,
            },
        )
        subgoals = self._tasking.initialize_subgoals(task_frame)
        created_at = time.monotonic()
        task_state = TaskExecutionState(
            task_id=f"task-{uuid.uuid4().hex[:8]}",
            raw_instruction=next_instruction,
            language=current_language,
            task_frame=task_frame,
            subgoals=subgoals,
            current_subgoal_index=0,
            origin_pose=self._capture_origin_pose(),
            status="pending",
            started_at=created_at,
        )
        with self._lock:
            self._archive_active_task_locked()
            self._task_state = task_state
        self._step_task_once()
        with self._lock:
            active_task = self._task_state
            command_revision = self._pipeline_state.command.command_revision
            session_reset_required = self._pipeline_state.system2.session_reset_required
            current_subgoal = None
            if active_task is not None and 0 <= active_task.current_subgoal_index < len(active_task.subgoals):
                current_subgoal = dict(active_task.subgoals[active_task.current_subgoal_index])

        print(
            "[INFO] Runtime planner command accepted: "
            f"instruction={next_instruction!r} language={current_language}"
        )
        return {
            "instruction": next_instruction,
            "language": current_language,
            "command_revision": command_revision,
            "immediate_stop": False,
            "session_id": self._session_id,
            "session_reset_required": session_reset_required,
            "task_id": task_state.task_id,
            "task_status": task_state.status if active_task is None else active_task.status,
            "task_frame": dict(task_frame),
            "current_subgoal": current_subgoal,
        }

    @staticmethod
    def _goal_matches_request(current_goal_xy: np.ndarray | None, request_goal_xy: np.ndarray, *, atol: float = 1.0e-3) -> bool:
        if current_goal_xy is None:
            return False
        current = np.asarray(current_goal_xy, dtype=np.float32).reshape(2)
        request = np.asarray(request_goal_xy, dtype=np.float32).reshape(2)
        return bool(np.linalg.norm(current - request) <= float(atol))

    @staticmethod
    def _pixel_goal_matches_request(current_pixel_xy: np.ndarray | None, request_pixel_xy: np.ndarray, *, atol_px: float = 1.0) -> bool:
        if current_pixel_xy is None:
            return False
        current = np.asarray(current_pixel_xy, dtype=np.float32).reshape(2)
        request = np.asarray(request_pixel_xy, dtype=np.float32).reshape(2)
        return bool(np.linalg.norm(current - request) <= float(atol_px))

    def _resolve_pixel_goal_fallback_world_xy(
        self,
        planner_input: PlannerInput,
        pixel_goal_xy: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int], float] | None:
        pixel_tuple = tuple(int(round(float(value))) for value in np.asarray(pixel_goal_xy, dtype=np.float32).reshape(2))
        return resolve_goal_world_xy_from_pixel(
            pixel_tuple,
            depth_image=planner_input.depth,
            intrinsic=planner_input.intrinsic,
            camera_pos_w=planner_input.camera_pos_w,
            camera_rot_w=planner_input.camera_rot_w,
            window_size=self._args.internvla_goal_depth_window,
            depth_min_m=self._args.internvla_goal_depth_min,
            depth_max_m=self._args.internvla_goal_depth_max,
        )

    def _fallback_pixel_goal_to_point_locked(
        self,
        *,
        planner_input: PlannerInput,
        pixel_goal_xy: np.ndarray,
        reason: str,
    ) -> bool:
        projected = self._resolve_pixel_goal_fallback_world_xy(planner_input, pixel_goal_xy)
        if projected is None:
            self._pipeline_state.navdp.error = (
                f"Pixel-goal fallback failed ({reason}): "
                f"no valid depth near pixel={tuple(int(round(float(v))) for v in np.asarray(pixel_goal_xy).reshape(2))}"
            )
            return False
        world_xy, pixel_xy, depth_m = projected
        self._update_goal(world_xy, planner_input.stamp_s)
        self._pipeline_state.navdp.error = None
        print(
            "[INFO] Pixel-goal fallback commit: "
            f"reason={reason} pixel=({pixel_xy[0]}, {pixel_xy[1]}) depth={depth_m:.2f}m "
            f"world=({world_xy[0]:.2f}, {world_xy[1]:.2f}) generation={self._pipeline_state.goal.generation}"
        )
        return True

    def _finalize_navdp_response(
        self,
        *,
        request_input_stamp: float,
        request_goal_generation: int,
        request_goal_mode: str,
        request_goal_world_xy: np.ndarray | None,
        request_goal_pixel_xy: np.ndarray | None,
        request_started_at_s: float,
        plan: NavDpPlan,
        world_path: np.ndarray,
    ) -> str | None:
        with self._lock:
            current_goal_mode = goal_target_mode(self._pipeline_state.goal)
            current_goal_xy = goal_target_world_xy(self._pipeline_state.goal)
            current_pixel_xy = goal_target_pixel_xy(self._pipeline_state.goal)
            current_goal_generation = self._pipeline_state.goal.generation

            discard_reason = None
            if current_goal_generation != request_goal_generation:
                discard_reason = (
                    f"goal_generation_changed:{request_goal_generation}->{current_goal_generation}"
                )
            elif request_goal_mode == "point":
                if current_goal_mode != "point" or request_goal_world_xy is None:
                    discard_reason = "active_goal_changed"
                elif not self._goal_matches_request(current_goal_xy, request_goal_world_xy):
                    discard_reason = "active_goal_changed"
            elif request_goal_mode == "pixel":
                if current_goal_mode != "pixel" or request_goal_pixel_xy is None:
                    discard_reason = "active_goal_changed"
                elif not self._pixel_goal_matches_request(current_pixel_xy, request_goal_pixel_xy):
                    discard_reason = "active_goal_changed"
            else:
                discard_reason = f"unsupported_goal_mode:{request_goal_mode}"

            if discard_reason is not None:
                self._pipeline_state.navdp.last_input_stamp = max(
                    self._pipeline_state.navdp.last_input_stamp,
                    float(request_input_stamp),
                )
                self._pipeline_state.navdp.last_goal_generation = int(request_goal_generation)
                self._pipeline_state.navdp.last_discarded_goal_generation = int(request_goal_generation)
                self._pipeline_state.navdp.last_discard_reason = discard_reason
                self._pipeline_state.navdp.last_request_started_at_s = float(request_started_at_s)
                return discard_reason

            self._pipeline_state.navdp.latest_plan = plan
            self._pipeline_state.navdp.latest_world_path = np.asarray(world_path, dtype=np.float32)
            self._pipeline_state.navdp.latest_plan_time = time.monotonic()
            self._pipeline_state.navdp.error = None
            self._pipeline_state.navdp.last_input_stamp = float(request_input_stamp)
            self._pipeline_state.navdp.last_goal_generation = int(request_goal_generation)
            self._pipeline_state.navdp.last_committed_goal_generation = int(request_goal_generation)
            self._pipeline_state.navdp.last_discard_reason = None
            self._pipeline_state.navdp.last_request_started_at_s = float(request_started_at_s)
        return None

    def _handle_system2_result(self, planner_input: PlannerInput, system2_result) -> None:
        result_signature = system2_result_signature(
            status=system2_result.status,
            uv_norm=system2_result.uv_norm,
            text=system2_result.text,
        )
        with self._lock:
            goal_state = self._pipeline_state.goal
            current_goal_mode = goal_target_mode(goal_state)
            current_goal_xy = goal_target_world_xy(goal_state)
            current_goal_pixel_xy = goal_target_pixel_xy(goal_state)
            pending_goal_kind = goal_state.candidate_kind
            same_signature = result_signature == self._pipeline_state.system2.last_signature
        decision_mode = str(system2_result.decision_mode or "").strip().lower() or "wait"
        if not same_signature and system2_result.text:
            print(f"[INFO] InternVLA raw output: {system2_result.text}")
        if not same_signature:
            print(
                "[INFO] System2 decision: "
                f"mode={decision_mode} needs_requery={bool(system2_result.needs_requery)}"
            )
        goal_reached = (
            current_goal_mode == "point"
            and current_goal_xy is not None
            and float(np.linalg.norm(current_goal_xy - np.asarray(planner_input.robot_state.base_pos_w, dtype=np.float32)[:2]))
            <= goal_state.tolerance
        )

        system2_error = None
        if decision_mode == SYSTEM2_PIXEL_GOAL_MODE:
            with self._lock:
                self._pipeline_state.status.action_only_suppressed = False
                self._pipeline_state.status.last_action_only_mode = None
            pixel_xy = None
            if system2_result.pixel_xy is not None:
                pixel_xy = np.asarray(system2_result.pixel_xy, dtype=np.float32).reshape(2)
            elif system2_result.uv_norm is not None:
                inferred_pixel_xy = normalized_uv_to_pixel_xy(
                    system2_result.uv_norm,
                    image_width=planner_input.rgb.shape[1],
                    image_height=planner_input.rgb.shape[0],
                )
                pixel_xy = np.asarray(inferred_pixel_xy, dtype=np.float32)

            if pixel_xy is None:
                system2_error = "InternVLA pixel goal is missing both pixel_xy and uv_norm."
            else:
                print(
                    "[INFO] InternVLA raw pixel goal: "
                    f"pixel=({int(round(float(pixel_xy[0])))}, {int(round(float(pixel_xy[1])))})"
                )
                supports_pixelgoal = self._navdp_client.supports_pixelgoal
                committed_goal = None
                if supports_pixelgoal is False:
                    projected = self._resolve_pixel_goal_fallback_world_xy(planner_input, pixel_xy)
                    if projected is None:
                        system2_error = (
                            "InternVLA goal projection failed: "
                            f"no valid depth near pixel={pixel_xy.tolist()}"
                        )
                    else:
                        world_xy, projected_pixel_xy, depth_m = projected
                        print(
                            "[INFO] InternVLA raw goal projection: "
                            f"pixel=({projected_pixel_xy[0]}, {projected_pixel_xy[1]}) depth={depth_m:.2f}m "
                            f"world=({world_xy[0]:.2f}, {world_xy[1]:.2f})"
                        )
                        with self._lock:
                            self._clear_action_override_locked()
                            committed_goal = self._observe_projected_goal_locked(world_xy, planner_input.stamp_s)
                        if committed_goal is not None:
                            with self._lock:
                                active_generation = self._pipeline_state.goal.generation
                        print(
                            "[INFO] NavDP pixelgoal fallback: "
                            f"generation={active_generation} "
                            f"world=({committed_goal[0]:.2f}, {committed_goal[1]:.2f})"
                        )
                else:
                    with self._lock:
                        self._clear_action_override_locked()
                        self._pipeline_state.status.action_only_suppressed = False
                        self._pipeline_state.status.last_action_only_mode = None
                        committed_goal = self._observe_pixel_goal_locked(pixel_xy, planner_input.stamp_s)
                    if committed_goal is not None:
                        with self._lock:
                            active_generation = self._pipeline_state.goal.generation
                        print(
                            "[INFO] NavDP pixelgoal accepted: "
                            f"generation={active_generation} "
                            f"pixel=({int(round(float(committed_goal[0])))}, {int(round(float(committed_goal[1])))})"
                        )
        elif decision_mode in DIRECT_ACTION_MODES:
            action_sequence = tuple(
                mode
                for mode in (system2_result.action_sequence or (decision_mode,))
                if mode in DIRECT_ACTION_MODES
            )
            if not action_sequence:
                action_sequence = (decision_mode,)
            active_mode = action_sequence[0]
            pending_modes = action_sequence[1:]
            with self._lock:
                self._clear_goal(f"system2_{decision_mode}")
                self._clear_navdp_locked(reset_algorithm=True, reset_progress=False)
                self._clear_action_override_locked()
                self._start_action_override_locked(
                    active_mode,
                    planner_input.robot_state,
                    pending_modes=pending_modes,
                )
                self._pipeline_state.follower = make_follower_state()
                self._set_locomotion_locked(_zero_command(), state_label="waiting", stamp_s=time.monotonic())
                self._pipeline_state.status.goal_done_reported = False
                self._pipeline_state.status.action_only_suppressed = False
                self._pipeline_state.status.last_action_only_mode = active_mode
            if pending_modes:
                print(
                    "[INFO] Direct action start: "
                    f"mode={active_mode} queued={list(pending_modes)}"
                )
            else:
                print(f"[INFO] Direct action start: mode={active_mode}")
        elif decision_mode == "stop" or system2_result.status == "stop":
            with self._lock:
                self._clear_action_override_locked()
                self._clear_goal("stop")
                self._clear_navdp_locked(reset_algorithm=True, reset_progress=False)
                self._pipeline_state.follower = make_follower_state()
                self._set_locomotion_locked(_zero_command(), state_label="waiting", stamp_s=time.monotonic())
                self._pipeline_state.status.goal_done_reported = False
                self._pipeline_state.status.action_only_suppressed = False
                self._pipeline_state.status.last_action_only_mode = None
            if not same_signature or current_goal_xy is not None:
                print("[INFO] InternVLA issued STOP. Holding zero locomotion command.")
        elif decision_mode in {"wait", "look_down"}:
            with self._lock:
                self._clear_action_override_locked()
                has_active_goal = current_goal_mode in {"point", "pixel"} and (
                    current_goal_xy is not None or current_goal_pixel_xy is not None
                )
                has_pending_goal = pending_goal_kind in {"point", "pixel"}
                if not has_active_goal and not has_pending_goal:
                    self._clear_goal("wait")
                    self._clear_navdp_locked(reset_algorithm=True, reset_progress=False)
                    self._pipeline_state.follower = make_follower_state()
                    self._set_locomotion_locked(_zero_command(), state_label="waiting", stamp_s=time.monotonic())
                    self._pipeline_state.status.goal_done_reported = False
                self._pipeline_state.status.action_only_suppressed = False
                self._pipeline_state.status.last_action_only_mode = None
            if not same_signature or current_goal_xy is not None or current_goal_pixel_xy is not None:
                if current_goal_mode in {"point", "pixel"} or pending_goal_kind in {"point", "pixel"}:
                    print("[INFO] InternVLA issued WAIT. Preserving current goal for NavDP tracking.")
                else:
                    print("[INFO] InternVLA issued WAIT. Holding zero locomotion command.")
        elif system2_result.status == "error":
            system2_error = f"InternVLA server error: {system2_result.text}"
            with self._lock:
                self._pipeline_state.status.action_only_suppressed = False
                self._pipeline_state.status.last_action_only_mode = None

        with self._lock:
            self._pipeline_state.system2.last_signature = result_signature
            self._pipeline_state.system2.last_result = system2_result
            self._pipeline_state.system2.error = system2_error

    def _system2_loop(self):
        while self._running:
            planner_input: PlannerInput | None = None
            planner_input_stamp = 0.0
            try:
                if self._controller is None:
                    time.sleep(0.05)
                    continue

                with self._lock:
                    planner_input = self._pipeline_state.capture.latest_input
                    last_input_stamp = self._pipeline_state.system2.last_input_stamp
                    session_reset_required = self._pipeline_state.system2.session_reset_required
                    action_override_mode = self._pipeline_state.action_override.mode
                    active_subgoal_type = self._current_subgoal_type_locked()
                planner_input_stamp = 0.0 if planner_input is None else planner_input.stamp_s

                if planner_input is None:
                    time.sleep(0.01)
                    continue
                if planner_input_stamp <= last_input_stamp:
                    time.sleep(0.01)
                    continue
                if active_subgoal_type is not None and active_subgoal_type != "navigate":
                    time.sleep(0.01)
                    continue
                if action_override_mode is not None:
                    time.sleep(0.01)
                    continue

                if session_reset_required:
                    self._reset_remote_session()
                    with self._lock:
                        self._pipeline_state.system2.session_reset_required = False
                        self._pipeline_state.system2.last_signature = None
                        self._pipeline_state.system2.error = None

                system2_result = self._system2_client.step_session(
                    session_id=self._session_id,
                    rgb=planner_input.rgb,
                    depth=planner_input.depth,
                    stamp_s=planner_input.stamp_s,
                )
                self._handle_system2_result(planner_input, system2_result)
            except Exception as exc:  # pragma: no cover - depends on Isaac Sim and remote servers
                with self._lock:
                    self._pipeline_state.system2.error = f"InternVLA step failed: {type(exc).__name__}: {exc}"
                time.sleep(0.25)
            finally:
                with self._lock:
                    self._pipeline_state.system2.last_input_stamp = max(
                        self._pipeline_state.system2.last_input_stamp,
                        planner_input_stamp,
                    )
            time.sleep(0.01)

    def _navdp_loop(self):
        while self._running:
            planner_input: PlannerInput | None = None
            planner_input_stamp = 0.0
            goal_generation = -1
            request_started_at_s = 0.0
            try:
                if self._controller is None:
                    time.sleep(0.05)
                    continue

                with self._lock:
                    pipeline_state = self._pipeline_state
                    planner_input = pipeline_state.capture.latest_input
                    goal_generation = pipeline_state.goal.generation
                    goal_tolerance = pipeline_state.goal.tolerance
                    last_input_stamp = pipeline_state.navdp.last_input_stamp
                    last_goal_generation = pipeline_state.navdp.last_goal_generation
                    current_goal_mode = goal_target_mode(pipeline_state.goal)
                    current_goal_xy = goal_target_world_xy(pipeline_state.goal)
                    current_pixel_xy = goal_target_pixel_xy(pipeline_state.goal)
                    algorithm_name = pipeline_state.navdp.algorithm_name
                planner_input_stamp = 0.0 if planner_input is None else planner_input.stamp_s

                if planner_input is None:
                    time.sleep(0.01)
                    continue
                if (
                    planner_input_stamp <= last_input_stamp
                    and goal_generation == last_goal_generation
                ):
                    time.sleep(0.01)
                    continue

                has_goal = current_goal_mode == "point" and current_goal_xy is not None
                has_pixel_goal = current_goal_mode == "pixel" and current_pixel_xy is not None
                goal_reached = has_goal and (
                    float(np.linalg.norm(current_goal_xy - np.asarray(planner_input.robot_state.base_pos_w, dtype=np.float32)[:2]))
                    <= goal_tolerance
                )
                if (not has_goal and not has_pixel_goal) or goal_reached:
                    with self._lock:
                        self._clear_latest_plan_locked()
                        self._pipeline_state.navdp.error = None
                        self._pipeline_state.navdp.last_input_stamp = planner_input_stamp
                        self._pipeline_state.navdp.last_goal_generation = goal_generation
                    time.sleep(0.01)
                    continue

                if algorithm_name is None:
                    algorithm_name = self._navdp_client.reset_pointgoal(
                        intrinsic=planner_input.intrinsic,
                        stop_threshold=self._args.navdp_stop_threshold,
                        batch_size=1,
                    )
                    status_message = self._navdp_client.consume_status_message()
                    if status_message:
                        print(f"[WARN] {status_message}")
                    print(f"[INFO] NavDP planner reset complete with algorithm: {algorithm_name}")

                request_started_at_s = time.monotonic()
                with self._lock:
                    self._pipeline_state.navdp.algorithm_name = algorithm_name
                    self._pipeline_state.navdp.last_request_input_stamp = float(planner_input_stamp)
                    self._pipeline_state.navdp.last_request_goal_generation = int(goal_generation)
                    self._pipeline_state.navdp.last_request_started_at_s = float(request_started_at_s)

                request_goal_mode = current_goal_mode
                request_goal_world_xy = None
                request_goal_pixel_xy = None
                if current_goal_mode == "pixel":
                    request_goal_pixel_xy = np.asarray(current_pixel_xy, dtype=np.float32).reshape(2).copy()
                    if not bool(self._navdp_client.supports_pixelgoal):
                        with self._lock:
                            converted = self._fallback_pixel_goal_to_point_locked(
                                planner_input=planner_input,
                                pixel_goal_xy=request_goal_pixel_xy,
                                reason="pixelgoal_unsupported",
                            )
                        if not converted:
                            with self._lock:
                                self._pipeline_state.navdp.last_input_stamp = planner_input_stamp
                                self._pipeline_state.navdp.last_goal_generation = goal_generation
                            time.sleep(0.01)
                            continue
                        time.sleep(0.01)
                        continue
                    try:
                        plan = self._navdp_client.step_pixelgoal(request_goal_pixel_xy, planner_input.rgb, planner_input.depth)
                    except Exception as exc:
                        print(f"[WARN] NavDP pixel-goal request failed; falling back to point-goal projection: {exc}")
                        with self._lock:
                            converted = self._fallback_pixel_goal_to_point_locked(
                                planner_input=planner_input,
                                pixel_goal_xy=request_goal_pixel_xy,
                                reason="pixelgoal_request_failed",
                            )
                        if not converted:
                            raise
                        time.sleep(0.01)
                        continue
                else:
                    goal_xy_body = point_goal_body_from_world(
                        current_goal_xy,
                        planner_input.robot_state.base_pos_w,
                        planner_input.robot_state.base_yaw,
                    )
                    request_goal_world_xy = np.asarray(current_goal_xy, dtype=np.float32).reshape(2).copy()
                    plan = self._navdp_client.step_pointgoal(goal_xy_body, planner_input.rgb, planner_input.depth)
                status_message = self._navdp_client.consume_status_message()
                if status_message:
                    print(f"[WARN] {status_message}")
                world_path = camera_plan_to_world_xy(
                    plan.trajectory_camera,
                    camera_pos_w=planner_input.camera_pos_w,
                    camera_rot_w=planner_input.camera_rot_w,
                )
                discard_reason = self._finalize_navdp_response(
                    request_input_stamp=planner_input_stamp,
                    request_goal_generation=goal_generation,
                    request_goal_mode=request_goal_mode,
                    request_goal_world_xy=request_goal_world_xy,
                    request_goal_pixel_xy=request_goal_pixel_xy,
                    request_started_at_s=request_started_at_s,
                    plan=plan,
                    world_path=world_path,
                )
                if discard_reason is not None:
                    print(
                        "[INFO] NavDP plan discard: "
                        f"reason={discard_reason} request_generation={goal_generation} "
                        f"request_stamp={planner_input_stamp:.3f}"
                    )
            except Exception as exc:  # pragma: no cover - depends on Isaac Sim and remote servers
                with self._lock:
                    self._pipeline_state.navdp.error = str(exc)
                    self._pipeline_state.navdp.last_input_stamp = max(
                        self._pipeline_state.navdp.last_input_stamp,
                        planner_input_stamp,
                    )
                    self._pipeline_state.navdp.last_goal_generation = goal_generation
                time.sleep(0.25)

    def _log_status(
        self,
        *,
        cmd: np.ndarray,
        state_label: str,
        path_points: int,
        latest_plan_time: float,
        latest_plan_latency: float | None,
        planning_error: str | None,
    ):
        now = time.monotonic()
        with self._lock:
            last_status_time = self._pipeline_state.status.last_status_time
        if (now - last_status_time) < self._status_interval:
            return

        plan_age_text = "n/a"
        if latest_plan_time > 0.0:
            plan_age_text = f"{max(0.0, now - latest_plan_time):.2f}s"
        plan_latency_text = "n/a" if latest_plan_latency is None else f"{latest_plan_latency:.2f}s"
        error_text = "" if planning_error is None else f" error={planning_error}"
        print(
            "[INFO] InternVLA/NavDP status: "
            f"state={state_label} plan_age={plan_age_text} plan_latency={plan_latency_text} "
            f"path_pts={path_points} cmd=({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}){error_text}"
        )
        with self._lock:
            self._pipeline_state.status.last_status_time = now

    def command(self) -> np.ndarray:
        if self._controller is None:
            return np.zeros(3, dtype=np.float32)

        if self._sensor is not None:
            self._sensor.apply_pending_pitch()
        self._maybe_capture_input()
        with self._lock:
            pipeline_state = self._pipeline_state
            goal_mode = goal_target_mode(pipeline_state.goal)
            goal_world_xy = goal_target_world_xy(pipeline_state.goal)
            goal_pixel_xy = goal_target_pixel_xy(pipeline_state.goal)
            goal_tolerance = pipeline_state.goal.tolerance
            world_path = pipeline_state.navdp.latest_world_path.copy()
            latest_plan = pipeline_state.navdp.latest_plan
            latest_plan_time = pipeline_state.navdp.latest_plan_time
            planning_error = self._combined_error_locked(pipeline_state)
            follower_state = FollowerState(
                smoothed_cmd=pipeline_state.follower.smoothed_cmd.copy(),
                last_time=pipeline_state.follower.last_time,
            )
            goal_done_reported = pipeline_state.status.goal_done_reported
            last_warning_time = pipeline_state.status.last_warning_time
            action_override = ActionOverrideState(
                mode=pipeline_state.action_override.mode,
                pending_modes=tuple(pipeline_state.action_override.pending_modes),
                started_at_s=pipeline_state.action_override.started_at_s,
                start_pos_xy=None
                if pipeline_state.action_override.start_pos_xy is None
                else np.asarray(pipeline_state.action_override.start_pos_xy, dtype=np.float32).copy(),
                start_yaw=pipeline_state.action_override.start_yaw,
                target_distance_m=pipeline_state.action_override.target_distance_m,
                target_yaw_rad=pipeline_state.action_override.target_yaw_rad,
                progress=pipeline_state.action_override.progress,
            )
        latest_plan_latency = None if latest_plan is None else latest_plan.plan_time_s
        state = self._robot_state()
        now = time.monotonic()

        if action_override.mode is not None:
            progress = 0.0
            timed_out = (now - action_override.started_at_s) >= self._action_timeout_s
            completed = False
            if action_override.mode == "forward":
                start_pos_xy = np.asarray(
                    state.base_pos_w[:2] if action_override.start_pos_xy is None else action_override.start_pos_xy,
                    dtype=np.float32,
                )
                distance = float(np.linalg.norm(np.asarray(state.base_pos_w, dtype=np.float32)[:2] - start_pos_xy))
                target_distance_m = max(1.0e-6, float(action_override.target_distance_m))
                progress = min(distance / target_distance_m, 1.0)
                completed = distance >= float(action_override.target_distance_m)
                cmd = np.asarray((self._args.vx_max, 0.0, 0.0), dtype=np.float32)
                state_label = "forward-override"
            else:
                yaw_delta = abs(_wrap_to_pi(state.base_yaw - float(action_override.start_yaw)))
                target_yaw_rad = max(1.0e-6, float(action_override.target_yaw_rad))
                progress = min(yaw_delta / target_yaw_rad, 1.0)
                completed = yaw_delta >= float(action_override.target_yaw_rad)
                yaw_sign = 1.0 if action_override.mode == "yaw_left" else -1.0
                cmd = np.asarray((0.0, 0.0, yaw_sign * self._args.wz_max), dtype=np.float32)
                state_label = "yaw-left-override" if action_override.mode == "yaw_left" else "yaw-right-override"

            if completed or timed_out:
                if timed_out and not completed:
                    print(f"[WARN] Direct action timeout: mode={action_override.mode}")
                elif action_override.pending_modes:
                    print(
                        "[INFO] Direct action complete: "
                        f"mode={action_override.mode} progress={progress:.2f} "
                        f"next={action_override.pending_modes[0]}"
                    )
                else:
                    print(f"[INFO] Direct action complete: mode={action_override.mode} progress={progress:.2f}")
                with self._lock:
                    self._pipeline_state.action_override.progress = float(progress)
                    if (not timed_out or completed) and action_override.pending_modes:
                        next_mode = str(action_override.pending_modes[0])
                        remaining_modes = tuple(action_override.pending_modes[1:])
                        self._start_action_override_locked(
                            next_mode,
                            state,
                            pending_modes=remaining_modes,
                        )
                    else:
                        self._clear_action_override_locked()
                    self._pipeline_state.follower = make_follower_state(now=now)
                    self._set_locomotion_locked(_zero_command(), state_label="waiting", stamp_s=now)
                self._log_status(
                    cmd=_zero_command(),
                    state_label="waiting",
                    path_points=0,
                    latest_plan_time=latest_plan_time,
                    latest_plan_latency=latest_plan_latency,
                    planning_error=planning_error,
                )
                return _zero_command()

            with self._lock:
                self._pipeline_state.action_override.progress = float(progress)
                self._pipeline_state.follower = make_follower_state(now=now)
                self._set_locomotion_locked(cmd, state_label=state_label, stamp_s=now)
            self._log_status(
                cmd=cmd,
                state_label=state_label,
                path_points=0,
                latest_plan_time=latest_plan_time,
                latest_plan_latency=latest_plan_latency,
                planning_error=planning_error,
            )
            return cmd

        has_goal = goal_mode == "point" and goal_world_xy is not None
        has_pixel_goal = goal_mode == "pixel" and goal_pixel_xy is not None
        goal_reached = has_goal and (
            float(np.linalg.norm(goal_world_xy - np.asarray(state.base_pos_w, dtype=np.float32)[:2])) <= goal_tolerance
        )

        if not has_goal and not has_pixel_goal:
            cmd = np.zeros(3, dtype=np.float32)
            with self._lock:
                self._pipeline_state.status.goal_done_reported = False
                self._pipeline_state.follower = make_follower_state(now=now)
                self._set_locomotion_locked(cmd, state_label="waiting", stamp_s=now)
            self._log_status(
                cmd=cmd,
                state_label="waiting",
                path_points=len(world_path),
                latest_plan_time=latest_plan_time,
                latest_plan_latency=latest_plan_latency,
                planning_error=planning_error,
            )
            return cmd

        if goal_reached:
            cmd = np.zeros(3, dtype=np.float32)
            if not goal_done_reported:
                print("[INFO] Dynamic goal reached. Waiting for the next System2 goal update.")
            with self._lock:
                self._pipeline_state.status.goal_done_reported = True
                self._pipeline_state.follower = make_follower_state(now=now)
                self._set_locomotion_locked(cmd, state_label="done", stamp_s=now)
            self._log_status(
                cmd=cmd,
                state_label="done",
                path_points=0,
                latest_plan_time=0.0,
                latest_plan_latency=None,
                planning_error=planning_error,
            )
            return cmd
        with self._lock:
            self._pipeline_state.status.goal_done_reported = False

        if planning_error is not None and len(world_path) == 0:
            if (now - last_warning_time) > 1.0:
                print(f"[WARN] InternVLA/NavDP planner error: {planning_error}")
                with self._lock:
                    self._pipeline_state.status.last_warning_time = now
            cmd = np.zeros(3, dtype=np.float32)
            with self._lock:
                self._pipeline_state.follower = make_follower_state(now=now)
                self._set_locomotion_locked(cmd, state_label="tracking", stamp_s=now)
            self._log_status(
                cmd=cmd,
                state_label="tracking",
                path_points=0,
                latest_plan_time=latest_plan_time,
                latest_plan_latency=latest_plan_latency,
                planning_error=planning_error,
            )
            return cmd

        plan_age = max(0.0, time.monotonic() - latest_plan_time) if latest_plan_time > 0.0 else float("inf")
        if len(world_path) == 0 or plan_age > self._hold_last_plan_timeout:
            cmd = np.zeros(3, dtype=np.float32)
            with self._lock:
                self._pipeline_state.follower = make_follower_state(now=now)
                self._set_locomotion_locked(cmd, state_label="tracking", stamp_s=now)
            self._log_status(
                cmd=cmd,
                state_label="tracking",
                path_points=len(world_path),
                latest_plan_time=latest_plan_time,
                latest_plan_latency=latest_plan_latency,
                planning_error=planning_error,
            )
            return cmd

        state_label = "stale-hold" if plan_age > self._plan_timeout else "tracking"
        cmd, next_follower_state = self._follower.compute_with_state(
            base_pos_w=state.base_pos_w,
            base_yaw=state.base_yaw,
            path_world_xy=world_path,
            state=follower_state,
        )
        with self._lock:
            self._pipeline_state.follower = next_follower_state
            self._set_locomotion_locked(cmd, state_label=state_label, stamp_s=now)
        self._log_status(
            cmd=cmd,
            state_label=state_label,
            path_points=len(world_path),
            latest_plan_time=latest_plan_time,
            latest_plan_latency=latest_plan_latency,
            planning_error=planning_error,
        )
        return cmd

    def shutdown(self):
        self._running = False
        if self._task_thread.is_alive():
            self._task_thread.join(timeout=1.0)
        if self._system2_thread.is_alive():
            self._system2_thread.join(timeout=1.0)
        if self._navdp_thread.is_alive():
            self._navdp_thread.join(timeout=1.0)
        if self._command_api_server is not None:
            self._command_api_server.shutdown()
            self._command_api_server = None
        if self._camera_api_server is not None:
            self._camera_api_server.shutdown()
            self._camera_api_server = None
        if self._sensor is not None:
            self._sensor.shutdown()
