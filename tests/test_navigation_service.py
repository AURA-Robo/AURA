from __future__ import annotations

from argparse import Namespace
import time

import numpy as np

from systems.navigation import service as navigation_service
from systems.navigation.service_codec import (
    encode_bytes_base64,
    encode_depth_png_base64,
    encode_rgb_jpeg_base64,
)
from systems.shared.contracts.inference import System2Result
from systems.shared.contracts.navigation import NavDpPlan
from systems.shared.contracts.observation import encode_rgb_history_npz
from systems.transport import SharedMemoryRing, encode_ndarray, ref_to_dict


class _GoalSystem2Client:
    def __init__(self, server_url: str, timeout_s: float):
        self.server_url = server_url
        self.timeout_s = timeout_s
        self.reset_calls: list[dict[str, object]] = []
        self.step_calls: list[dict[str, object]] = []

    def reset_session(self, **kwargs):
        self.reset_calls.append(dict(kwargs))
        return {"status": "ok"}

    def step_session(self, **kwargs):
        self.step_calls.append(dict(kwargs))
        return System2Result(
            status="goal",
            uv_norm=np.asarray((0.5, 0.5), dtype=np.float32),
            text="goal",
            latency_ms=1.0,
            stamp_s=float(kwargs["stamp_s"]),
            pixel_xy=np.asarray((2.0, 1.0), dtype=np.float32),
            decision_mode="pixel_goal",
        )


class _ActionSystem2Client(_GoalSystem2Client):
    def step_session(self, **kwargs):
        self.step_calls.append(dict(kwargs))
        return System2Result(
            status="hold",
            uv_norm=None,
            text="turn right",
            latency_ms=1.0,
            stamp_s=float(kwargs["stamp_s"]),
            decision_mode="yaw_right",
            action_sequence=("yaw_right",),
        )


class _NavDpClient:
    def __init__(self, server_url: str, timeout_s: float, *, fallback_mode: str):
        self.server_url = server_url
        self.timeout_s = timeout_s
        self.fallback_mode = fallback_mode
        self.reset_calls: list[dict[str, object]] = []
        self.step_calls: list[dict[str, object]] = []

    def reset_pointgoal(self, **kwargs):
        self.reset_calls.append(dict(kwargs))
        return "navdp"

    def step_pointgoal(self, *args, **kwargs):
        self.step_calls.append({"args": args, "kwargs": dict(kwargs)})
        return NavDpPlan(
            trajectory_camera=np.asarray([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=np.float32),
            all_trajectories_camera=None,
            values=None,
            plan_time_s=0.01,
            stamp_s=123.0,
        )


class _BlockingNavDpClient(_NavDpClient):
    def __init__(self, server_url: str, timeout_s: float, *, fallback_mode: str):
        super().__init__(server_url, timeout_s, fallback_mode=fallback_mode)
        self.started = False
        self.release = False

    def step_pointgoal(self, *args, **kwargs):
        self.started = True
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not self.release:
            time.sleep(0.01)
        return super().step_pointgoal(*args, **kwargs)


def _args() -> Namespace:
    return Namespace(
        system2_url="http://system2",
        system2_timeout=1.0,
        navdp_url="http://navdp",
        navdp_timeout=1.0,
        navdp_fallback="disabled",
        navdp_stop_threshold=-0.5,
        goal_depth_window=5,
        goal_depth_min=0.25,
        goal_depth_max=6.0,
        backend_autostart=False,
        backend_log_dir=".",
        navdp_host="127.0.0.1",
        navdp_port=18888,
        navdp_checkpoint="navdp.ckpt",
        navdp_device="cpu",
        system2_host="127.0.0.1",
        system2_port=15801,
        system2_llama_url="http://127.0.0.1:15802",
        system2_model_path="",
        navigation_shm_name="aura_test_nav_shm",
        navigation_shm_slot_size=8 * 1024 * 1024,
        navigation_shm_capacity=8,
        memory_approach_radius_m=0.8,
        memory_reacquire_radius_m=1.0,
        memory_reacquire_timeout_sec=4.0,
    )


def _update_payload(
    frame_id: int,
    *,
    base_pos_w: tuple[float, float, float] = (0.0, 0.0, 0.0),
    detections: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rgb = np.full((3, 4, 3), frame_id, dtype=np.uint8)
    depth = np.full((3, 4), 1.5, dtype=np.float32)
    payload = {
        "rgb_jpeg_base64": encode_rgb_jpeg_base64(rgb),
        "depth_png_base64": encode_depth_png_base64(depth),
        "intrinsic": np.eye(3, dtype=np.float32).tolist(),
        "camera_pos_w": np.asarray((0.0, 0.0, 1.0), dtype=np.float32).tolist(),
        "camera_rot_w": np.eye(3, dtype=np.float32).tolist(),
        "robot_state": {
            "base_pos_w": np.asarray(base_pos_w, dtype=np.float32).tolist(),
            "base_yaw": 0.0,
            "lin_vel_b": np.asarray((0.0, 0.0), dtype=np.float32).tolist(),
            "yaw_rate": 0.0,
        },
        "stamp_s": float(frame_id),
    }
    if detections is not None:
        payload["detections"] = list(detections)
    return payload


def _wait_until(predicate, timeout_s: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_navigation_service_update_returns_before_system1_finishes(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _BlockingNavDpClient)
    monkeypatch.setattr(
        navigation_service,
        "resolve_goal_world_xy",
        lambda **kwargs: (np.asarray((1.0, 2.0), dtype=np.float32), (1, 1), 1.5),
    )
    monkeypatch.setattr(
        navigation_service,
        "camera_plan_to_world_xy",
        lambda trajectory_camera, camera_pos_w, camera_rot_w: np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
    )

    system = navigation_service.NavigationSystem(_args())
    try:
        system.command("go to purple box", "en", task_id="planner-purple-box")
        started = time.perf_counter()
        response = system.update(_update_payload(1))
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        assert response["status"] == "running"
        assert elapsed_ms < 150.0
        assert _wait_until(lambda: system._navdp.started)

        system._navdp.release = True
        assert _wait_until(lambda: system.status_payload()["path_points"] == 2)
    finally:
        system._navdp.release = True
        system.shutdown()


def test_navigation_service_uses_shm_payload_and_ignores_history_wire(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)
    monkeypatch.setattr(
        navigation_service,
        "resolve_goal_world_xy",
        lambda **kwargs: (np.asarray((1.0, 2.0), dtype=np.float32), (1, 1), 1.5),
    )
    monkeypatch.setattr(
        navigation_service,
        "camera_plan_to_world_xy",
        lambda trajectory_camera, camera_pos_w, camera_rot_w: np.asarray([[1.0, 2.0]], dtype=np.float32),
    )

    shm_name = f"aura_nav_test_{time.time_ns()}"
    ring = SharedMemoryRing(name=shm_name, slot_size=8 * 1024 * 1024, capacity=8, create=True)
    system = navigation_service.NavigationSystem(_args())
    try:
        system.command("find desk", "en", task_id=None)
        rgb = np.full((3, 4, 3), 7, dtype=np.uint8)
        depth = np.full((3, 4), 1.5, dtype=np.float32)
        rgb_ref = ring.write(encode_ndarray(rgb))
        depth_ref = ring.write(encode_ndarray(depth))
        history = np.zeros((1, 3, 4, 3), dtype=np.uint8)
        payload = {
            "frame_id": "frame-7",
            "rgb_ref": ref_to_dict(rgb_ref),
            "depth_ref": ref_to_dict(depth_ref),
            "intrinsic": np.eye(3, dtype=np.float32).tolist(),
            "camera_pos_w": np.asarray((0.0, 0.0, 1.0), dtype=np.float32).tolist(),
            "camera_rot_w": np.eye(3, dtype=np.float32).tolist(),
            "robot_state": {
                "base_pos_w": np.asarray((0.0, 0.0, 0.0), dtype=np.float32).tolist(),
                "base_yaw": 0.0,
                "lin_vel_b": np.asarray((0.0, 0.0), dtype=np.float32).tolist(),
                "yaw_rate": 0.0,
            },
            "stamp_s": 7.0,
            "system2_history_npz_base64": encode_bytes_base64(encode_rgb_history_npz(history)),
            "navdp_history_npz_base64": encode_bytes_base64(encode_rgb_history_npz(history)),
        }
        system.update(payload)

        assert _wait_until(lambda: len(system._system2.step_calls) == 1)
        assert _wait_until(lambda: len(system._navdp.step_calls) == 1)
        assert int(system._system2.step_calls[0]["rgb"][0, 0, 0]) == 7
        assert "rgb_history" not in system._system2.step_calls[0]
        assert "rgb_history" not in system._navdp.step_calls[0]["kwargs"]
    finally:
        system.shutdown()
        ring.close(unlink=True)


def test_navigation_service_pauses_system1_when_system2_requests_direct_action(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _ActionSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)

    system = navigation_service.NavigationSystem(_args())
    try:
        system.command("go to purple box", "en", task_id="planner-purple-box")
        system.update(_update_payload(1))

        assert _wait_until(lambda: len(system._system2.step_calls) == 1)
        status = system.status_payload()
        assert status["action_override_mode"] == "yaw_right"
        assert status["path_points"] == 0
        assert len(system._navdp.step_calls) == 0
    finally:
        system.shutdown()


def test_navigation_service_generates_local_trajectory_for_go_to_purple_box(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)
    monkeypatch.setattr(
        navigation_service,
        "resolve_goal_world_xy",
        lambda **kwargs: (np.asarray((1.0, 2.0), dtype=np.float32), (1, 1), 1.5),
    )
    monkeypatch.setattr(
        navigation_service,
        "camera_plan_to_world_xy",
        lambda trajectory_camera, camera_pos_w, camera_rot_w: np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
    )

    system = navigation_service.NavigationSystem(_args())
    try:
        command_payload = system.command("go to purple box", "en", task_id="planner-purple-box")
        system.update(_update_payload(1))

        assert command_payload["task_id"] == "planner-purple-box"
        assert command_payload["status"] == "running"
        assert _wait_until(lambda: system.status_payload()["path_points"] == 2)
        trajectory_payload = system.trajectory_payload()
        assert len(system._system2.reset_calls) == 1
        assert len(system._system2.step_calls) == 1
        assert len(system._navdp.reset_calls) == 1
        assert len(system._navdp.step_calls) == 1
        assert system._system2.reset_calls[0]["instruction"] == "go to purple box"
        assert trajectory_payload["status"] == "running"
        assert trajectory_payload["task_id"] == "planner-purple-box"
        assert trajectory_payload["system2"]["status"] == "goal"
        assert trajectory_payload["system2"]["decision_mode"] == "pixel_goal"
        assert trajectory_payload["goal_world_xy"] == [1.0, 2.0]
        assert trajectory_payload["trajectory_world_xy"] == [[1.0, 2.0], [2.0, 3.0]]
        assert trajectory_payload["path_points"] == 2
    finally:
        system.shutdown()


def test_navigation_service_memory_pose_bypasses_system2_and_sets_direct_goal(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)
    monkeypatch.setattr(
        navigation_service,
        "camera_plan_to_world_xy",
        lambda trajectory_camera, camera_pos_w, camera_rot_w: np.asarray([[2.0, 3.0], [2.2, 3.0]], dtype=np.float32),
    )

    system = navigation_service.NavigationSystem(_args())
    try:
        command_payload = system.command_memory_target(
            navigation_service.MemoryNavigationTarget(
                object_id="obj-chair-1",
                class_name="chair",
                scene_scope="warehouse",
                world_pose_xyz=np.asarray((2.0, 3.0, 0.0), dtype=np.float32),
                pose_age_sec=5.0,
                stop_radius_m=0.8,
                reacquire_radius_m=1.0,
                reacquire_timeout_sec=4.0,
            ),
            task_id="planner-chair-memory",
        )
        system.update(_update_payload(1))

        assert command_payload["memoryNavigationMode"] == "memory_pose"
        assert _wait_until(lambda: len(system._navdp.step_calls) == 1)
        status = system.status_payload()
        assert len(system._system2.reset_calls) == 0
        assert status["goal_world_xy"] == [2.0, 3.0]
        assert status["resolvedMemoryObjectId"] == "obj-chair-1"
        assert status["reacquireState"] == "approach_memory_pose"
    finally:
        system.shutdown()


def test_navigation_service_memory_pose_reacquires_matching_detection(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)

    system = navigation_service.NavigationSystem(_args())
    try:
        system.command_memory_target(
            navigation_service.MemoryNavigationTarget(
                object_id="obj-chair-2",
                class_name="chair",
                scene_scope="warehouse",
                world_pose_xyz=np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
                pose_age_sec=2.0,
                stop_radius_m=0.8,
                reacquire_radius_m=1.0,
                reacquire_timeout_sec=4.0,
            ),
            task_id="planner-chair-memory",
        )
        status = system.update(
            _update_payload(
                1,
                base_pos_w=(0.0, 0.0, 0.0),
                detections=[
                    {
                        "class_name": "chair",
                        "track_id": "track-chair-2",
                        "bbox_xyxy": [0, 0, 1, 1],
                        "confidence": 0.95,
                        "world_pose_xyz": [0.2, 0.1, 0.0],
                    }
                ],
            )
        )

        assert status["reacquireState"] == "reacquired"
        assert status["last_error"] is None
        assert status["memoryAwareTaskActive"] is True
    finally:
        system.shutdown()


def test_navigation_service_memory_pose_times_out_when_detection_does_not_reappear(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)

    system = navigation_service.NavigationSystem(_args())
    try:
        system.command_memory_target(
            navigation_service.MemoryNavigationTarget(
                object_id="obj-chair-3",
                class_name="chair",
                scene_scope="warehouse",
                world_pose_xyz=np.asarray((0.0, 0.0, 0.0), dtype=np.float32),
                pose_age_sec=2.0,
                stop_radius_m=0.8,
                reacquire_radius_m=1.0,
                reacquire_timeout_sec=0.01,
            ),
            task_id="planner-chair-memory",
        )
        system.update(_update_payload(1, base_pos_w=(0.0, 0.0, 0.0), detections=[]))
        time.sleep(0.02)
        status = system.update(_update_payload(2, base_pos_w=(0.0, 0.0, 0.0), detections=[]))

        assert status["reacquireState"] == "failed"
        assert status["last_error"] == "memory_target_not_reacquired"
    finally:
        system.shutdown()


def test_navigation_service_status_exposes_current_robot_pose(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)

    system = navigation_service.NavigationSystem(_args())
    try:
        status = system.update(_update_payload(1, base_pos_w=(1.5, -0.5, 0.25)))

        assert status["current_robot_pose"]["world_xy"] == [1.5, -0.5]
        assert status["current_robot_pose"]["world_xyz"] == [1.5, -0.5, 0.25]
        assert status["current_robot_pose"]["yaw_rad"] == 0.0
        assert status["currentRobotPose"]["world_xy"] == [1.5, -0.5]
    finally:
        system.shutdown()


def test_navigation_service_return_pose_sets_direct_goal_and_stops_at_origin(monkeypatch) -> None:
    monkeypatch.setattr(navigation_service, "InternVlaNavClient", _GoalSystem2Client)
    monkeypatch.setattr(navigation_service, "NavDpClient", _NavDpClient)
    monkeypatch.setattr(
        navigation_service,
        "camera_plan_to_world_xy",
        lambda trajectory_camera, camera_pos_w, camera_rot_w: np.asarray([[2.0, 3.0], [2.2, 3.0]], dtype=np.float32),
    )

    system = navigation_service.NavigationSystem(_args())
    try:
        command_payload = system.command_return_pose(
            navigation_service.ReturnPoseTarget(
                world_xy=np.asarray((2.0, 3.0), dtype=np.float32),
                yaw_rad=0.0,
            ),
            task_id="planner-return",
        )
        status = system.update(_update_payload(1, base_pos_w=(2.0, 3.0, 0.0)))

        assert command_payload["goal_world_xy"] == [2.0, 3.0]
        assert command_payload["system2"]["status"] == "return_pose"
        assert status["goal_world_xy"] is None
        assert status["path_points"] == 0
        assert status["system2"]["status"] == "stop"
        assert status["system2"]["decision_mode"] == "stop"
        assert status["current_robot_pose"]["world_xy"] == [2.0, 3.0]
        assert status["returnTargetWorldXy"] == [2.0, 3.0]
        assert len(system._system2.reset_calls) == 0
        assert len(system._navdp.step_calls) == 0
    finally:
        system.shutdown()
