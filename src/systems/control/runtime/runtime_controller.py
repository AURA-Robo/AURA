"""Runtime controllers owned by the control subsystem."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from systems.navigation.api.runtime import HolonomicPurePursuitFollower, RobotState2D, yaw_from_quaternion_wxyz


def _zero_command() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


DIRECT_ACTION_MODES = frozenset(("forward", "yaw_left", "yaw_right"))


def _wrap_to_pi(angle_rad: float) -> float:
    return float(np.arctan2(np.sin(float(angle_rad)), np.cos(float(angle_rad))))


class InternVlaNavDpController:
    """Follow trajectories produced elsewhere in the runtime."""

    requires_render = True

    def __init__(self, args):
        self.quit_requested = False
        self._args = args
        self._follower = HolonomicPurePursuitFollower(
            lookahead_distance=args.lookahead_distance,
            vx_max=args.vx_max,
            vy_max=args.vy_max,
            wz_max=args.wz_max,
            smoothing_tau=args.cmd_smoothing_tau,
        )
        self._controller = None
        self._latest_path_world_xy = np.zeros((0, 2), dtype=np.float32)
        self._latest_trajectory_stamp_s = 0.0
        self._last_navigation_status: dict[str, object] = {
            "status": "idle",
            "instruction": None,
            "task_id": None,
            "session_id": None,
            "path_points": 0,
            "last_error": None,
        }
        self._last_error: str | None = None
        self._last_command = _zero_command()
        self._state_label = "waiting"
        self._last_status_time = 0.0
        self._forward_step_m = max(0.05, float(getattr(args, "internvla_forward_step_m", 0.35)))
        self._turn_step_rad = float(np.deg2rad(max(1.0, float(getattr(args, "internvla_turn_step_deg", 25.0)))))
        self._action_timeout_s = max(0.1, float(getattr(args, "internvla_action_timeout_s", 2.0)))
        self._direct_action_mode: str | None = None
        self._pending_action_modes: tuple[str, ...] = ()
        self._direct_action_started_at = 0.0
        self._direct_action_start_pos_xy: np.ndarray | None = None
        self._direct_action_start_yaw = 0.0

    def print_help(self):
        print("[INFO] Navigation trajectory control")
        print(f"[INFO]   navigation url    : {self._args.navigation_url}")
        print(f"[INFO]   update hz         : {self._args.navigation_update_hz:.2f}")
        print(f"[INFO]   trajectory ttl    : {self._args.navigation_trajectory_timeout:.2f}s")
        print(
            f"[INFO]   direct actions    : forward={self._forward_step_m:.2f}m "
            f"turn={np.rad2deg(self._turn_step_rad):.1f}deg timeout={self._action_timeout_s:.2f}s"
        )
        print(
            f"[INFO]   saturation        : vx={self._args.vx_max:.2f} "
            f"vy={self._args.vy_max:.2f} wz={self._args.wz_max:.2f}"
        )

    def bind_controller(self, controller):
        if self._controller is not None:
            return
        self._controller = controller
        self._follower.reset()

    def reset(self):
        self._follower.reset()
        self._latest_path_world_xy = np.zeros((0, 2), dtype=np.float32)
        self._latest_trajectory_stamp_s = 0.0
        self._last_navigation_status = {
            "status": "idle",
            "instruction": None,
            "task_id": None,
            "session_id": None,
            "path_points": 0,
            "last_error": None,
        }
        self._last_error = None
        self._last_command = _zero_command()
        self._state_label = "waiting"
        self._clear_direct_action()

    def update_navigation_payload(self, payload: dict[str, object]) -> None:
        trajectory = np.asarray(payload.get("trajectory_world_xy", []), dtype=np.float32)
        if trajectory.size == 0:
            trajectory = np.zeros((0, 2), dtype=np.float32)
        else:
            trajectory = trajectory.reshape(-1, 2)
        self._latest_path_world_xy = trajectory
        self._latest_trajectory_stamp_s = float(payload.get("stamp_s", 0.0))
        self._last_navigation_status = dict(payload)
        last_error = payload.get("last_error")
        self._last_error = str(last_error) if isinstance(last_error, str) and last_error else None
        if self._last_error is not None:
            self._clear_direct_action()
            return

        direct_action_sequence = self._direct_action_sequence(payload)
        if len(trajectory) > 0:
            self._clear_direct_action()
            return
        if direct_action_sequence:
            if (
                self._direct_action_mode is None
                or self._direct_action_mode != direct_action_sequence[0]
                or self._pending_action_modes != direct_action_sequence[1:]
            ):
                self._start_direct_action(direct_action_sequence)
            return
        self._clear_direct_action()

    def _robot_state(self) -> RobotState2D:
        if self._controller is None:
            raise RuntimeError("controller is not bound")
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

    def _log_status(self):
        now = time.monotonic()
        if (now - self._last_status_time) < 1.0:
            return
        age = None if self._latest_trajectory_stamp_s <= 0.0 else max(0.0, now - self._latest_trajectory_stamp_s)
        age_text = "n/a" if age is None else f"{age:.2f}s"
        print(
            "[INFO] Navigation follower status: "
            f"state={self._state_label} path_pts={len(self._latest_path_world_xy)} "
            f"traj_age={age_text} cmd=({self._last_command[0]:.2f}, {self._last_command[1]:.2f}, {self._last_command[2]:.2f})"
        )
        self._last_status_time = now

    def runtime_status(self) -> dict[str, object]:
        now = time.monotonic()
        trajectory_age_s = None if self._latest_trajectory_stamp_s <= 0.0 else max(0.0, now - self._latest_trajectory_stamp_s)
        status = dict(self._last_navigation_status)
        system2_payload = status.get("system2") if isinstance(status.get("system2"), dict) else {}
        return {
            "executionMode": "NAV",
            "navigation_status": status.get("status", "idle"),
            "instruction": status.get("instruction"),
            "task_id": status.get("task_id"),
            "session_id": status.get("session_id"),
            "memoryAwareTaskActive": bool(status.get("memoryAwareTaskActive")),
            "memoryNavigationMode": status.get("memoryNavigationMode"),
            "resolvedMemoryObjectId": status.get("resolvedMemoryObjectId"),
            "resolvedMemoryPoseAgeSec": status.get("resolvedMemoryPoseAgeSec"),
            "reacquireState": status.get("reacquireState"),
            "trajectory_points": int(len(self._latest_path_world_xy)),
            "trajectory_age_s": trajectory_age_s,
            "trajectory_world_xy": self._latest_path_world_xy.astype(float).tolist(),
            "follower_state": {
                "state_label": self._state_label,
                "smoothed_cmd": self._last_command.astype(float).tolist(),
            },
            "locomotion_command": self._last_command.astype(float).tolist(),
            "state_label": self._state_label,
            "last_error": self._last_error,
            "system2_status": system2_payload.get("status"),
            "system2_decision_mode": system2_payload.get("decision_mode"),
            "action_override_mode": self._direct_action_mode,
            "pending_action_modes": list(self._pending_action_modes),
            "routeState": {
                "pathPoints": int(len(self._latest_path_world_xy)),
                "trajectoryAgeSec": trajectory_age_s,
            },
        }

    def command_api_status(self) -> dict[str, object]:
        return self.runtime_status()

    def command(self) -> np.ndarray:
        if self._controller is None:
            return _zero_command()

        robot_state = self._robot_state()
        path = self._latest_path_world_xy.copy()
        now = time.monotonic()
        path_is_stale = self._latest_trajectory_stamp_s <= 0.0 or (now - self._latest_trajectory_stamp_s) > float(
            self._args.navigation_trajectory_timeout
        )

        if self._last_error is not None:
            self._follower.reset()
            self._clear_direct_action()
            self._last_command = _zero_command()
            self._state_label = "error"
            self._log_status()
            return self._last_command.copy()
        if self._direct_action_mode is not None:
            progress = 0.0
            timed_out = (now - self._direct_action_started_at) >= self._action_timeout_s
            completed = False
            if self._direct_action_mode == "forward":
                start_pos_xy = np.asarray(
                    robot_state.base_pos_w[:2] if self._direct_action_start_pos_xy is None else self._direct_action_start_pos_xy,
                    dtype=np.float32,
                )
                distance = float(np.linalg.norm(np.asarray(robot_state.base_pos_w, dtype=np.float32)[:2] - start_pos_xy))
                progress = min(distance / self._forward_step_m, 1.0)
                completed = distance >= self._forward_step_m
                self._last_command = np.asarray((self._args.vx_max, 0.0, 0.0), dtype=np.float32)
                self._state_label = "forward-override"
            else:
                yaw_delta = abs(_wrap_to_pi(robot_state.base_yaw - self._direct_action_start_yaw))
                progress = min(yaw_delta / self._turn_step_rad, 1.0)
                completed = yaw_delta >= self._turn_step_rad
                yaw_sign = 1.0 if self._direct_action_mode == "yaw_left" else -1.0
                self._last_command = np.asarray((0.0, 0.0, yaw_sign * self._args.wz_max), dtype=np.float32)
                self._state_label = "yaw-left-override" if self._direct_action_mode == "yaw_left" else "yaw-right-override"

            if completed or timed_out:
                next_sequence = self._pending_action_modes
                self._clear_direct_action()
                if next_sequence:
                    self._start_direct_action(next_sequence)
                self._follower.reset()
                self._last_command = _zero_command()
                self._state_label = "waiting"
                self._log_status()
                return self._last_command.copy()

            self._log_status()
            return self._last_command.copy()
        if len(path) == 0 or path_is_stale:
            self._follower.reset()
            self._last_command = _zero_command()
            self._state_label = "waiting" if len(path) == 0 else "stale"
            self._log_status()
            return self._last_command.copy()

        self._last_command = self._follower.compute(
            base_pos_w=robot_state.base_pos_w,
            base_yaw=robot_state.base_yaw,
            path_world_xy=path,
        )
        self._state_label = "tracking"
        self._log_status()
        return self._last_command.copy()

    def shutdown(self):
        return None

    @staticmethod
    def _direct_action_sequence(payload: dict[str, object]) -> tuple[str, ...]:
        system2_payload = payload.get("system2") if isinstance(payload.get("system2"), dict) else {}
        sequence = system2_payload.get("action_sequence")
        if isinstance(sequence, list):
            normalized = tuple(str(mode) for mode in sequence if str(mode) in DIRECT_ACTION_MODES)
            if normalized:
                return normalized
        decision_mode = str(system2_payload.get("decision_mode") or "").strip()
        if decision_mode in DIRECT_ACTION_MODES:
            return (decision_mode,)
        return ()

    def _start_direct_action(self, sequence: tuple[str, ...]) -> None:
        if self._controller is None or not sequence:
            return
        robot_state = self._robot_state()
        self._direct_action_mode = str(sequence[0])
        self._pending_action_modes = tuple(str(mode) for mode in sequence[1:])
        self._direct_action_started_at = time.monotonic()
        self._direct_action_start_pos_xy = np.asarray(robot_state.base_pos_w, dtype=np.float32)[:2].copy()
        self._direct_action_start_yaw = float(robot_state.base_yaw)

    def _clear_direct_action(self) -> None:
        self._direct_action_mode = None
        self._pending_action_modes = ()
        self._direct_action_started_at = 0.0
        self._direct_action_start_pos_xy = None
        self._direct_action_start_yaw = 0.0


NavDpPointGoalController = InternVlaNavDpController

__all__ = ["InternVlaNavDpController", "NavDpPointGoalController"]
