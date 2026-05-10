from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simulation.application.runtime_coordinator import NavigationRuntimeCoordinator
from systems.perception.api.camera_api import CameraFrame


class _FakeControlHandler:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def update_navigation_payload(self, payload: dict[str, object]) -> None:
        self.payloads.append(dict(payload))

    def runtime_status(self) -> dict[str, object]:
        return {"executionMode": "NAV", "state_label": "running"}

    def reset(self) -> None:
        self.payloads.clear()


class _FakeNavigationClient:
    def __init__(self) -> None:
        self.status_calls = 0
        self.update_calls = 0
        self.last_update_kwargs: dict[str, object] | None = None

    def status(self) -> dict[str, object]:
        self.status_calls += 1
        return {
            "status": "running",
            "instruction": "go to the target",
            "task_id": "task-1",
            "session_id": "session-1",
            "trajectory_world_xy": [[1.0, 2.0], [2.0, 3.0]],
            "goal_world_xy": [3.0, 4.0],
        }

    def update(self, **_kwargs) -> dict[str, object]:
        self.update_calls += 1
        self.last_update_kwargs = dict(_kwargs)
        return {
            "status": "running",
            "instruction": "go to the target",
            "task_id": "task-1",
            "session_id": "session-1",
            "trajectory_world_xy": [[1.0, 2.0], [2.0, 3.0]],
            "goal_world_xy": [3.0, 4.0],
            "stamp_s": 12.5,
        }


class _BlockingNavigationClient(_FakeNavigationClient):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def update(self, **_kwargs) -> dict[str, object]:
        self.update_calls += 1
        self.last_update_kwargs = dict(_kwargs)
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("navigation update release timed out")
        return {
            "status": "running",
            "instruction": "go to the target",
            "task_id": "task-1",
            "session_id": "session-1",
            "trajectory_world_xy": [[1.0, 2.0], [2.0, 3.0]],
            "goal_world_xy": [3.0, 4.0],
            "stamp_s": 12.5,
        }


class _IdleNavigationClient(_FakeNavigationClient):
    def status(self) -> dict[str, object]:
        self.status_calls += 1
        return {"status": "idle", "trajectory_world_xy": []}

    def update(self, **_kwargs) -> dict[str, object]:
        self.update_calls += 1
        self.last_update_kwargs = dict(_kwargs)
        return {
            "status": "idle",
            "instruction": None,
            "task_id": None,
            "session_id": None,
            "trajectory_world_xy": [],
            "stamp_s": 12.5,
            "current_robot_pose": {
                "world_xyz": [0.0, 0.0, 0.8],
                "world_xy": [0.0, 0.0],
                "yaw_rad": 0.0,
            },
        }


class _FakeMemory:
    def __init__(self) -> None:
        self.observe_calls = 0

    def observe(self, _frame) -> None:
        self.observe_calls += 1

    def build_system2_view(self, **_kwargs):
        return SimpleNamespace(rgb_history=np.zeros((0, 0, 0, 3), dtype=np.uint8))

    def build_navdp_view(self, **_kwargs):
        return SimpleNamespace(rgb_history=np.zeros((0, 0, 0, 3), dtype=np.uint8))

    def reset_epoch(self, *_args, **_kwargs) -> None:
        return None


class _FakePerception:
    def __init__(self) -> None:
        self.ingest_calls = 0

    def ingest(self, raw):
        self.ingest_calls += 1
        return SimpleNamespace(
            rgb=raw.rgb,
            depth=raw.depth,
            intrinsic=raw.intrinsic,
            camera_pos_w=raw.camera_pos_w,
            camera_rot_w=raw.camera_rot_w,
            stamp_s=raw.stamp_s,
        )

    def latest_health(self) -> dict[str, object]:
        return {"status": "running"}

    def close(self) -> None:
        return None


class _FakeRobot:
    def get_world_pose(self):
        return np.asarray((0.0, 0.0, 0.8), dtype=np.float32), np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float32)

    def get_linear_velocity(self):
        return np.asarray((0.0, 0.0, 0.0), dtype=np.float32)

    def get_angular_velocity(self):
        return np.asarray((0.0, 0.0, 0.0), dtype=np.float32)


class _FakeController:
    def __init__(self) -> None:
        self.robot = _FakeRobot()


class _FakeSensor:
    def __init__(self) -> None:
        self.capture_calls = 0
        self.apply_pitch_calls = 0
        self._frame = CameraFrame(
            rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            depth=np.ones((32, 32), dtype=np.float32),
            intrinsic=np.asarray(((100.0, 0.0, 16.0), (0.0, 100.0, 16.0), (0.0, 0.0, 1.0)), dtype=np.float32),
            camera_pos_w=np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            camera_rot_w=np.eye(3, dtype=np.float32),
            stamp_s=12.5,
        )

    def apply_pending_pitch(self) -> None:
        self.apply_pitch_calls += 1

    def capture_frame(self):
        self.capture_calls += 1
        return self._frame

    def shutdown(self) -> None:
        return None


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_runtime_coordinator_keeps_capturing_while_navigation_update_runs_in_background() -> None:
    args = SimpleNamespace(
        viewer_publish=False,
        navigation_url="http://127.0.0.1:17882",
        navigation_timeout=1.0,
        navigation_update_hz=1.0,
    )
    control_handler = _FakeControlHandler()
    coordinator = NavigationRuntimeCoordinator(args, control_handler=control_handler)
    coordinator._navigation = _BlockingNavigationClient()
    coordinator._memory = _FakeMemory()
    coordinator._perception = _FakePerception()
    coordinator._controller = _FakeController()
    coordinator._sensor = _FakeSensor()
    coordinator._last_navigation_status = {
        "status": "running",
        "instruction": "go to the target",
        "task_id": "task-1",
        "session_id": "session-1",
    }
    coordinator._active_navigation_session_id = "session-1"
    coordinator._last_status_time = 100.0

    try:
        with patch("simulation.application.runtime_coordinator.time.monotonic", side_effect=[100.0, 100.1]):
            coordinator.step()
            assert coordinator._navigation.started.wait(timeout=0.5)
            coordinator.step()

        assert coordinator._sensor.capture_calls == 2
        assert coordinator._perception.ingest_calls == 2
        assert coordinator._memory.observe_calls == 1
        assert coordinator._navigation.update_calls == 1
        assert coordinator._navigation.last_update_kwargs is not None
        assert "system2_rgb_history" not in coordinator._navigation.last_update_kwargs
        assert "navdp_rgb_history" not in coordinator._navigation.last_update_kwargs
        assert control_handler.payloads == []

        coordinator._navigation.release.set()
        assert _wait_until(lambda: coordinator._completed_update_result is not None)
        coordinator._consume_navigation_worker_results()

        assert len(control_handler.payloads) == 1
        assert control_handler.payloads[0]["status"] == "running"
    finally:
        coordinator._navigation.release.set()
        coordinator.shutdown()


def test_runtime_coordinator_degrades_when_viewer_transport_bind_fails() -> None:
    args = SimpleNamespace(
        viewer_publish=True,
        navigation_url="http://127.0.0.1:17882",
        navigation_timeout=1.0,
        navigation_update_hz=1.0,
    )
    control_handler = _FakeControlHandler()
    with patch(
        "simulation.application.runtime_coordinator.ViewerFramePublisher",
        side_effect=PermissionError("tcp://127.0.0.1:18881"),
    ):
        coordinator = NavigationRuntimeCoordinator(args, control_handler=control_handler)
    try:
        status = coordinator.runtime_status()

        assert status["viewer"]["enabled"] is False
        assert status["viewer"]["status"] == "degraded"
        assert "PermissionError" in status["viewer"]["last_error"]
        assert "18881" in status["viewer"]["last_error"]
    finally:
        coordinator.shutdown()


def test_runtime_coordinator_publishes_idle_pose_to_navigation_status() -> None:
    args = SimpleNamespace(
        viewer_publish=False,
        navigation_url="http://127.0.0.1:17882",
        navigation_timeout=1.0,
        navigation_update_hz=1.0,
    )
    control_handler = _FakeControlHandler()
    coordinator = NavigationRuntimeCoordinator(args, control_handler=control_handler)
    coordinator._navigation = _IdleNavigationClient()
    coordinator._memory = _FakeMemory()
    coordinator._perception = _FakePerception()
    coordinator._controller = _FakeController()
    coordinator._sensor = _FakeSensor()
    coordinator._last_navigation_status = {"status": "idle"}
    coordinator._last_status_time = 100.0

    try:
        with patch("simulation.application.runtime_coordinator.time.monotonic", return_value=100.0):
            coordinator.step()

        assert _wait_until(lambda: coordinator._completed_update_result is not None)
        coordinator._consume_navigation_worker_results()

        assert coordinator._navigation.update_calls == 1
        assert coordinator._navigation.last_update_kwargs is not None
        np.testing.assert_allclose(
            np.asarray(coordinator._navigation.last_update_kwargs["base_pos_w"], dtype=np.float32),
            np.asarray([0.0, 0.0, 0.8], dtype=np.float32),
        )
        assert coordinator._last_navigation_status["current_robot_pose"]["world_xy"] == [0.0, 0.0]
        assert control_handler.payloads[-1]["current_robot_pose"]["world_xy"] == [0.0, 0.0]
    finally:
        coordinator.shutdown()


def test_runtime_coordinator_ignores_stale_navigation_result_after_reset() -> None:
    args = SimpleNamespace(
        viewer_publish=False,
        navigation_url="http://127.0.0.1:17882",
        navigation_timeout=1.0,
        navigation_update_hz=1.0,
    )
    control_handler = _FakeControlHandler()
    coordinator = NavigationRuntimeCoordinator(args, control_handler=control_handler)
    coordinator._navigation = _BlockingNavigationClient()
    coordinator._memory = _FakeMemory()
    coordinator._perception = _FakePerception()
    coordinator._controller = _FakeController()
    coordinator._sensor = _FakeSensor()
    coordinator._last_navigation_status = {
        "status": "running",
        "instruction": "go to the target",
        "task_id": "task-1",
        "session_id": "session-1",
    }
    coordinator._active_navigation_session_id = "session-1"
    coordinator._last_status_time = 100.0

    try:
        with patch("simulation.application.runtime_coordinator.time.monotonic", return_value=100.0):
            coordinator.step()
        assert coordinator._navigation.started.wait(timeout=0.5)

        coordinator.reset()
        coordinator._navigation.release.set()
        time.sleep(0.05)
        coordinator._consume_navigation_worker_results()

        assert control_handler.payloads == []
        assert coordinator._last_navigation_status == {"status": "idle"}
        assert coordinator._last_navigation_payload is None
    finally:
        coordinator._navigation.release.set()
        coordinator.shutdown()
