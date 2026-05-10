from __future__ import annotations

import math

import numpy as np

from systems.shared.contracts.navigation import RobotState2D
from systems.world_state.api.runtime_state import GoalState, goal_current_body_xy


def test_goal_current_body_xy_transforms_world_point_into_body_frame() -> None:
    robot_state = RobotState2D(
        base_pos_w=np.asarray([1.0, 2.0, 0.0], dtype=np.float32),
        base_yaw=math.pi / 2.0,
        lin_vel_b=np.zeros(3, dtype=np.float32),
        yaw_rate=0.0,
    )
    goal_state = GoalState(
        tolerance=0.1,
        target_mode="point",
        target_world_xy=np.asarray([1.0, 3.0], dtype=np.float32),
    )

    np.testing.assert_allclose(
        goal_current_body_xy(goal_state, robot_state),
        np.asarray([1.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
