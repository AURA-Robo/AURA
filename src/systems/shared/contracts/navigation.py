from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np


@dataclass(slots=True)
class NavDpPlan:
    """Trajectory plan returned by the navigation subsystem."""

    trajectory_camera: np.ndarray
    all_trajectories_camera: np.ndarray | None
    values: np.ndarray | None
    plan_time_s: float
    stamp_s: float


@dataclass(slots=True)
class RobotState2D:
    """Minimal 2D robot state shared by navigation, memory, and runtime state."""

    base_pos_w: np.ndarray
    base_yaw: float
    lin_vel_b: np.ndarray
    yaw_rate: float


@dataclass(slots=True)
class FollowerState:
    """Externalized follower runtime state for snapshot-based control."""

    smoothed_cmd: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    last_time: float = 0.0


def make_follower_state(*, now: float | None = None) -> FollowerState:
    """Create a follower state initialized to zero command."""

    return FollowerState(last_time=time.monotonic() if now is None else float(now))
