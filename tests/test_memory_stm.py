from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

from systems.memory.api.runtime import ShortTermMemory, decode_rgb_history_npz, encode_rgb_history_npz
from systems.shared.contracts.navigation import RobotState2D


def _planner_input(frame_id: int):
    rgb = np.full((2, 2, 3), frame_id, dtype=np.uint8)
    depth = np.full((2, 2), frame_id / 10.0, dtype=np.float32)
    return SimpleNamespace(
        robot_state=RobotState2D(
            base_pos_w=np.asarray((float(frame_id), float(frame_id) + 0.5), dtype=np.float32),
            base_yaw=float(frame_id) * 0.1,
            lin_vel_b=np.asarray((0.0, 0.0), dtype=np.float32),
            yaw_rate=0.0,
        ),
        goal_xy_body=None,
        rgb=rgb,
        depth=depth,
        intrinsic=np.eye(3, dtype=np.float32),
        camera_pos_w=np.asarray((1.0, 2.0, 3.0), dtype=np.float32),
        camera_rot_w=np.eye(3, dtype=np.float32),
        stamp_s=float(frame_id),
    )


def test_stm_views_exclude_current_frame_and_keep_latest_check_frame() -> None:
    stm = ShortTermMemory()

    stm.observe(_planner_input(1))

    system2_view = stm.build_system2_view(num_history=4, look_down_cap=6)
    navdp_view = stm.build_navdp_view(memory_size=8)

    assert system2_view.rgb_history.shape == (0, 0, 0, 3)
    assert navdp_view.rgb_history.shape == (0, 0, 0, 3)
    assert int(stm.latest_check_frame()[0, 0, 0]) == 1


def test_system2_sparse_sampling_and_recent_history_match_runtime_semantics() -> None:
    stm = ShortTermMemory()
    for frame_id in range(1, 6):
        stm.observe(_planner_input(frame_id))

    view = stm.build_system2_view(num_history=3, look_down_cap=2)

    assert view.rgb_history.shape[0] == 4
    assert view.rgb_history[:, 0, 0, 0].tolist() == [1, 2, 3, 4]
    assert view.sparse_rgb_history[:, 0, 0, 0].tolist() == [1, 2, 4]
    assert view.look_down_rgb_history[:, 0, 0, 0].tolist() == [3, 4]


def test_navdp_history_is_recent_fifo_without_current_frame() -> None:
    stm = ShortTermMemory()
    for frame_id in range(1, 6):
        stm.observe(_planner_input(frame_id))

    view = stm.build_navdp_view(memory_size=3)

    assert view.rgb_history.shape[0] == 2
    assert view.rgb_history[:, 0, 0, 0].tolist() == [3, 4]


def test_stm_keeps_independent_system2_and_navdp_epochs() -> None:
    stm = ShortTermMemory()
    stm.observe(_planner_input(1))
    stm.reset_system2_epoch()
    stm.observe(_planner_input(2))
    stm.observe(_planner_input(3))

    system2_view = stm.build_system2_view(num_history=4, look_down_cap=6)
    navdp_view = stm.build_navdp_view(memory_size=8)

    assert system2_view.rgb_history[:, 0, 0, 0].tolist() == [2]
    assert navdp_view.rgb_history[:, 0, 0, 0].tolist() == [1, 2]

    stm.reset_navdp_epoch()
    stm.observe(_planner_input(4))
    navdp_view_after_reset = stm.build_navdp_view(memory_size=8)

    assert navdp_view_after_reset.rgb_history.shape[0] == 0


def test_rgb_history_npz_round_trip_and_rejects_bad_shape() -> None:
    rgb_history = np.stack(
        [
            np.full((2, 2, 3), 7, dtype=np.uint8),
            np.full((2, 2, 3), 9, dtype=np.uint8),
        ],
        axis=0,
    )

    decoded = decode_rgb_history_npz(encode_rgb_history_npz(rgb_history))
    assert decoded.shape == rgb_history.shape
    assert np.array_equal(decoded, rgb_history)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, rgb_history=np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="rgb_history must have shape"):
        decode_rgb_history_npz(buffer.getvalue())
