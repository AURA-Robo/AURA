"""Trajectory follower that converts a path into G1 locomotion commands."""

from __future__ import annotations

import math
import time

import numpy as np

from systems.shared.contracts.navigation import FollowerState, make_follower_state

from .geometry import world_xy_to_body_xy, wrap_to_pi


class HolonomicPurePursuitFollower:
    """Simple holonomic SE(2) follower for the locomotion command interface."""

    def __init__(
        self,
        lookahead_distance: float,
        vx_max: float,
        vy_max: float,
        wz_max: float,
        smoothing_tau: float,
        kx: float = 0.9,
        ky: float = 1.1,
        kyaw: float = 1.0,
        kcross: float = 0.75,
    ):
        self.lookahead_distance = max(0.05, float(lookahead_distance))
        self.vx_max = abs(float(vx_max))
        self.vy_max = abs(float(vy_max))
        self.wz_max = abs(float(wz_max))
        self.smoothing_tau = max(0.0, float(smoothing_tau))
        self.kx = float(kx)
        self.ky = float(ky)
        self.kyaw = float(kyaw)
        self.kcross = float(kcross)

        self._state = make_follower_state()

    def reset(self):
        self._state = make_follower_state()

    def _pick_target(self, path_body_xy: np.ndarray) -> tuple[np.ndarray, float]:
        distances = np.linalg.norm(path_body_xy, axis=1)
        index = next((i for i, distance in enumerate(distances) if distance >= self.lookahead_distance), len(path_body_xy) - 1)
        target = path_body_xy[index]

        if index < len(path_body_xy) - 1:
            segment = path_body_xy[min(index + 1, len(path_body_xy) - 1)] - path_body_xy[index]
            heading = math.atan2(float(segment[1]), float(segment[0])) if np.linalg.norm(segment) > 1e-5 else 0.0
        else:
            heading = math.atan2(float(target[1]), float(target[0])) if distances[index] > 1e-5 else 0.0
        return target, heading

    def _smooth_with_state(self, cmd: np.ndarray, state: FollowerState) -> tuple[np.ndarray, FollowerState]:
        now = time.monotonic()
        previous_cmd = np.asarray(state.smoothed_cmd, dtype=np.float32).reshape(3)
        previous_time = float(state.last_time) if state.last_time > 0.0 else now
        dt = max(1e-3, now - previous_time)
        if self.smoothing_tau <= 0.0:
            smoothed_cmd = cmd.astype(np.float32)
            return smoothed_cmd.copy(), FollowerState(smoothed_cmd=smoothed_cmd.copy(), last_time=now)

        alpha = min(1.0, dt / (self.smoothing_tau + dt))
        smoothed_cmd = ((1.0 - alpha) * previous_cmd) + (alpha * cmd)
        smoothed_cmd = np.asarray(smoothed_cmd, dtype=np.float32)
        return smoothed_cmd.copy(), FollowerState(smoothed_cmd=smoothed_cmd.copy(), last_time=now)

    def _smooth(self, cmd: np.ndarray) -> np.ndarray:
        smoothed_cmd, self._state = self._smooth_with_state(cmd, self._state)
        return smoothed_cmd

    def compute(self, base_pos_w: np.ndarray, base_yaw: float, path_world_xy: np.ndarray) -> np.ndarray:
        cmd, self._state = self.compute_with_state(
            base_pos_w=base_pos_w,
            base_yaw=base_yaw,
            path_world_xy=path_world_xy,
            state=self._state,
        )
        return cmd

    def compute_with_state(
        self,
        *,
        base_pos_w: np.ndarray,
        base_yaw: float,
        path_world_xy: np.ndarray,
        state: FollowerState,
    ) -> tuple[np.ndarray, FollowerState]:
        path_body_xy = world_xy_to_body_xy(path_world_xy, base_pos_w=base_pos_w, base_yaw=base_yaw)
        if len(path_body_xy) == 0:
            return self._smooth_with_state(np.zeros(3, dtype=np.float32), state)

        target, path_heading = self._pick_target(path_body_xy)
        cross_heading = math.atan2(float(target[1]), max(float(target[0]), 1e-4))
        heading_error = wrap_to_pi(path_heading)

        raw = np.asarray(
            (
                np.clip(self.kx * float(target[0]), -self.vx_max, self.vx_max),
                np.clip(self.ky * float(target[1]), -self.vy_max, self.vy_max),
                np.clip(
                    (self.kyaw * heading_error) + (self.kcross * cross_heading),
                    -self.wz_max,
                    self.wz_max,
                ),
            ),
            dtype=np.float32,
        )
        return self._smooth_with_state(raw, state)
