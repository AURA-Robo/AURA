"""Short-term frame memory owned by the runtime."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
from systems.shared.contracts.observation import (
    HistoryView,
    ObservationFrame,
    decode_rgb_history_npz,
    encode_rgb_history_npz,
)

if TYPE_CHECKING:
    from systems.shared.contracts.navigation import RobotState2D


def _copy_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWxC RGB image, got shape {image.shape}.")
    if image.shape[2] > 3:
        image = image[:, :, :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size > 0 and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _copy_depth(depth: np.ndarray) -> np.ndarray:
    image = np.asarray(depth, dtype=np.float32)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"Expected HxW depth image, got shape {image.shape}.")
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(image)


def _copy_robot_state(robot_state: Any) -> Any:
    return type(robot_state)(
        base_pos_w=np.asarray(robot_state.base_pos_w, dtype=np.float32).copy(),
        base_yaw=float(robot_state.base_yaw),
        lin_vel_b=np.asarray(robot_state.lin_vel_b, dtype=np.float32).copy(),
        yaw_rate=float(robot_state.yaw_rate),
    )


def _empty_rgb_history() -> np.ndarray:
    return np.zeros((0, 0, 0, 3), dtype=np.uint8)


@dataclass(slots=True)
class StmFrameRecord:
    stamp_s: float
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    robot_state: "RobotState2D"


@dataclass(slots=True)
class System2HistoryView:
    rgb_history: np.ndarray
    sparse_rgb_history: np.ndarray
    look_down_rgb_history: np.ndarray


@dataclass(slots=True)
class NavDpHistoryView:
    rgb_history: np.ndarray


class ShortTermMemory:
    """Canonical frame-history owner for runtime perception consumers."""

    SYSTEM2_CONSUMER = "system2"
    NAVDP_CONSUMER = "navdp"

    def __init__(self):
        self._lock = threading.RLock()
        self._frames: list[StmFrameRecord] = []
        self._consumer_epoch_start = {
            self.SYSTEM2_CONSUMER: 0,
            self.NAVDP_CONSUMER: 0,
        }
        self._consumer_session_id = {
            self.SYSTEM2_CONSUMER: None,
            self.NAVDP_CONSUMER: None,
        }

    def observe(self, planner_input: ObservationFrame | Any) -> StmFrameRecord:
        record = StmFrameRecord(
            stamp_s=float(planner_input.stamp_s),
            rgb=_copy_rgb(planner_input.rgb),
            depth=_copy_depth(planner_input.depth),
            intrinsic=np.asarray(planner_input.intrinsic, dtype=np.float32).copy(),
            camera_pos_w=np.asarray(planner_input.camera_pos_w, dtype=np.float32).copy(),
            camera_rot_w=np.asarray(planner_input.camera_rot_w, dtype=np.float32).copy(),
            robot_state=_copy_robot_state(planner_input.robot_state),
        )
        with self._lock:
            self._frames.append(record)
        return record

    def latest(self) -> StmFrameRecord | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def reset_epoch(self, consumer: str, session_id: str | None = None, *, include_latest: bool = False) -> None:
        consumer_name = str(consumer)
        with self._lock:
            start = len(self._frames)
            if include_latest and start > 0:
                start -= 1
            self._consumer_epoch_start[consumer_name] = start
            self._consumer_session_id[consumer_name] = None if session_id is None else str(session_id)

    def reset_system2_epoch(self) -> None:
        self.reset_epoch(self.SYSTEM2_CONSUMER)

    def reset_navdp_epoch(self) -> None:
        self.reset_epoch(self.NAVDP_CONSUMER)

    def build_history(self, consumer: str) -> HistoryView:
        consumer_name = str(consumer)
        with self._lock:
            start = self._consumer_epoch_start.get(consumer_name, 0)
            frames = self._frames[start:]
            past_frames = frames[:-1] if frames else []
            rgb_history = [frame.rgb.copy() for frame in past_frames]
            session_id = self._consumer_session_id.get(consumer_name)
        history_array = np.stack(rgb_history, axis=0) if rgb_history else _empty_rgb_history()
        return HistoryView(consumer=consumer_name, session_id=session_id, rgb_history=history_array)

    def build_system2_view(self, num_history: int, look_down_cap: int) -> System2HistoryView:
        history_array = self.build_history(self.SYSTEM2_CONSUMER).rgb_history
        sparse_rgb_history = self._sample_sparse_history(history_array, num_history)
        look_down_rgb_history = self._sample_recent_history(history_array, look_down_cap)
        return System2HistoryView(
            rgb_history=history_array,
            sparse_rgb_history=sparse_rgb_history,
            look_down_rgb_history=look_down_rgb_history,
        )

    def build_navdp_view(self, memory_size: int) -> NavDpHistoryView:
        max_frames = max(0, int(memory_size) - 1)
        history_array = self.build_history(self.NAVDP_CONSUMER).rgb_history
        if max_frames > 0:
            history_array = np.ascontiguousarray(history_array[-max_frames:])
        else:
            history_array = _empty_rgb_history()
        return NavDpHistoryView(rgb_history=history_array)

    def latest_check_frame(self) -> np.ndarray | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1].rgb.copy()

    @staticmethod
    def _sample_sparse_history(history_array: np.ndarray, num_history: int) -> np.ndarray:
        if history_array.shape[0] == 0 or int(num_history) <= 0:
            return _empty_rgb_history()
        indices = np.unique(np.linspace(0, history_array.shape[0] - 1, int(num_history), dtype=np.int32)).tolist()
        return np.ascontiguousarray(history_array[indices])

    @staticmethod
    def _sample_recent_history(history_array: np.ndarray, count: int) -> np.ndarray:
        if history_array.shape[0] == 0 or int(count) <= 0:
            return _empty_rgb_history()
        return np.ascontiguousarray(history_array[-int(count) :])
