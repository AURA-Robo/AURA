"""Goal providers for NavDP-backed command generation."""

from __future__ import annotations

import threading

import numpy as np

from systems.shared.contracts.navigation import RobotState2D

from .geometry import point_goal_body_from_world, point_goal_world_from_frame


class PointGoalProvider:
    """Maintains a point-goal target and exposes it in body coordinates."""

    def __init__(self, goal_xy: tuple[float, float], goal_frame: str, tolerance: float):
        self.goal_xy = np.asarray(goal_xy, dtype=np.float32)
        self.goal_frame = str(goal_frame)
        self.tolerance = float(tolerance)
        self._target_world_xy: np.ndarray | None = None

    def bind_start_pose(self, base_pos_w: np.ndarray, base_yaw: float):
        if self._target_world_xy is not None:
            return
        self._target_world_xy = point_goal_world_from_frame(self.goal_xy, self.goal_frame, base_pos_w, base_yaw)

    def reset(self):
        self._target_world_xy = None

    @property
    def target_world_xy(self) -> np.ndarray | None:
        return None if self._target_world_xy is None else self._target_world_xy.copy()

    def current_goal_xy(self, state: RobotState2D) -> np.ndarray:
        if self._target_world_xy is None:
            raise RuntimeError("PointGoalProvider must be bound to a start pose before use.")
        return point_goal_body_from_world(self._target_world_xy, state.base_pos_w, state.base_yaw)

    def is_done(self, state: RobotState2D) -> bool:
        if self._target_world_xy is None:
            return False
        distance = float(np.linalg.norm(self._target_world_xy - np.asarray(state.base_pos_w, dtype=np.float32)[:2]))
        return distance <= self.tolerance


class DynamicWorldGoalProvider:
    """Thread-safe world-frame point goal that can be updated by System 2."""

    def __init__(self, tolerance: float):
        self.tolerance = float(tolerance)
        self._target_world_xy: np.ndarray | None = None
        self._last_update_stamp_s = 0.0
        self._last_clear_reason = "init"
        self._lock = threading.Lock()

    def update(self, world_xy: np.ndarray, stamp_s: float):
        with self._lock:
            self._target_world_xy = np.asarray(world_xy, dtype=np.float32).reshape(2)
            self._last_update_stamp_s = float(stamp_s)
            self._last_clear_reason = ""

    def clear(self, reason: str):
        with self._lock:
            self._target_world_xy = None
            self._last_clear_reason = str(reason)

    def reset(self):
        self.clear("reset")

    @property
    def has_goal(self) -> bool:
        with self._lock:
            return self._target_world_xy is not None

    @property
    def target_world_xy(self) -> np.ndarray | None:
        with self._lock:
            if self._target_world_xy is None:
                return None
            return self._target_world_xy.copy()

    @property
    def last_update_stamp_s(self) -> float:
        with self._lock:
            return self._last_update_stamp_s

    @property
    def last_clear_reason(self) -> str:
        with self._lock:
            return self._last_clear_reason

    def current_goal_xy(self, state: RobotState2D) -> np.ndarray:
        target_world_xy = self.target_world_xy
        if target_world_xy is None:
            raise RuntimeError("DynamicWorldGoalProvider does not have an active goal.")
        return point_goal_body_from_world(target_world_xy, state.base_pos_w, state.base_yaw)

    def is_done(self, state: RobotState2D) -> bool:
        target_world_xy = self.target_world_xy
        if target_world_xy is None:
            return False
        distance = float(np.linalg.norm(target_world_xy - np.asarray(state.base_pos_w, dtype=np.float32)[:2]))
        return distance <= self.tolerance
