"""Runtime state contracts owned by the world-state subsystem."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from systems.shared.contracts.inference import System2Result
from systems.shared.contracts.navigation import FollowerState, NavDpPlan, RobotState2D, make_follower_state


def _zero_command() -> np.ndarray:
    return np.zeros(3, dtype=np.float32)


def _empty_world_path() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.float32)


def _point_goal_body_from_world(goal_world_xy: np.ndarray, base_pos_w: np.ndarray, base_yaw: float) -> np.ndarray:
    goal_xy = np.asarray(goal_world_xy, dtype=np.float64).reshape(2)
    base_xy = np.asarray(base_pos_w, dtype=np.float64).reshape(3)[:2]
    delta_world = goal_xy - base_xy
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    rot_bw = np.asarray(((cos_yaw, sin_yaw), (-sin_yaw, cos_yaw)), dtype=np.float64)
    return (rot_bw @ delta_world).astype(np.float32)


@dataclass(slots=True)
class PlannerInput:
    robot_state: RobotState2D
    goal_xy_body: np.ndarray | None
    rgb: np.ndarray
    depth: np.ndarray
    intrinsic: np.ndarray
    camera_pos_w: np.ndarray
    camera_rot_w: np.ndarray
    stamp_s: float


@dataclass(slots=True)
class CommandState:
    instruction: str
    language: str
    command_revision: int = 0


@dataclass(slots=True)
class CaptureState:
    latest_input: PlannerInput | None = None
    last_capture_time: float = 0.0


@dataclass(slots=True)
class System2State:
    session_reset_required: bool = True
    last_input_stamp: float = 0.0
    last_signature: tuple[str, tuple[float, ...] | None, str] | None = None
    last_result: System2Result | None = None
    error: str | None = None


System2RuntimeState = System2State


@dataclass(slots=True)
class GoalState:
    tolerance: float
    target_mode: str = "none"
    target_world_xy: np.ndarray | None = None
    target_pixel_xy: np.ndarray | None = None
    last_update_stamp_s: float = 0.0
    last_clear_reason: str = "init"
    generation: int = 0
    candidate_kind: str = "none"
    raw_candidate_world_xy: np.ndarray | None = None
    filtered_candidate_world_xy: np.ndarray | None = None
    raw_candidate_pixel_xy: np.ndarray | None = None
    filtered_candidate_pixel_xy: np.ndarray | None = None
    candidate_started_at_s: float = 0.0
    candidate_last_stamp_s: float = 0.0
    candidate_sample_count: int = 0


@dataclass(slots=True)
class NavDpState:
    algorithm_name: str | None = None
    last_input_stamp: float = 0.0
    last_goal_generation: int = -1
    last_request_input_stamp: float = 0.0
    last_request_goal_generation: int = -1
    last_request_started_at_s: float = 0.0
    last_committed_goal_generation: int = -1
    last_discarded_goal_generation: int = -1
    last_discard_reason: str | None = None
    latest_plan: NavDpPlan | None = None
    latest_world_path: np.ndarray = field(default_factory=_empty_world_path)
    latest_plan_time: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class ActionOverrideState:
    mode: str | None = None
    pending_modes: tuple[str, ...] = ()
    started_at_s: float = 0.0
    start_pos_xy: np.ndarray | None = None
    start_yaw: float = 0.0
    target_distance_m: float = 0.0
    target_yaw_rad: float = 0.0
    progress: float = 0.0


@dataclass(slots=True)
class LocomotionState:
    command: np.ndarray = field(default_factory=_zero_command)
    state_label: str = "waiting"
    last_command_stamp: float = 0.0


@dataclass(slots=True)
class StatusState:
    last_warning_time: float = 0.0
    last_status_time: float = 0.0
    goal_done_reported: bool = False
    action_only_suppressed: bool = False
    last_action_only_mode: str | None = None


@dataclass(slots=True)
class NavigationPipelineState:
    command: CommandState
    capture: CaptureState = field(default_factory=CaptureState)
    system2: System2State = field(default_factory=System2State)
    goal: GoalState | None = None
    navdp: NavDpState = field(default_factory=NavDpState)
    action_override: ActionOverrideState = field(default_factory=ActionOverrideState)
    follower: FollowerState = field(default_factory=make_follower_state)
    locomotion: LocomotionState = field(default_factory=LocomotionState)
    status: StatusState = field(default_factory=StatusState)


def make_navigation_pipeline_state(*, instruction: str, language: str, tolerance: float) -> NavigationPipelineState:
    return NavigationPipelineState(
        command=CommandState(instruction=str(instruction), language=str(language)),
        goal=GoalState(tolerance=float(tolerance)),
    )


@dataclass(slots=True)
class TaskExecutionState:
    task_id: str
    raw_instruction: str
    language: str
    task_frame: dict[str, object]
    subgoals: list[dict[str, object]]
    current_subgoal_index: int = 0
    origin_pose: dict[str, object] | None = None
    status: str = "pending"
    last_report: str | None = None
    failure_reason: str | None = None
    started_at: float = 0.0
    finished_at: float | None = None


def goal_has_target(goal_state: GoalState) -> bool:
    return goal_target_mode(goal_state) in {"point", "pixel"}


def goal_target_mode(goal_state: GoalState) -> str:
    if goal_state.target_mode == "point" and goal_state.target_world_xy is not None:
        return "point"
    if goal_state.target_mode == "pixel" and goal_state.target_pixel_xy is not None:
        return "pixel"
    if goal_state.target_world_xy is not None:
        return "point"
    if goal_state.target_pixel_xy is not None:
        return "pixel"
    return "none"


def goal_target_world_xy(goal_state: GoalState) -> np.ndarray | None:
    if goal_target_mode(goal_state) != "point" or goal_state.target_world_xy is None:
        return None
    return goal_state.target_world_xy.copy()


def goal_target_pixel_xy(goal_state: GoalState) -> np.ndarray | None:
    if goal_target_mode(goal_state) != "pixel" or goal_state.target_pixel_xy is None:
        return None
    return goal_state.target_pixel_xy.copy()


def goal_pending_world_xy(goal_state: GoalState) -> np.ndarray | None:
    if goal_state.candidate_kind != "point" or goal_state.filtered_candidate_world_xy is None:
        return None
    return goal_state.filtered_candidate_world_xy.copy()


def goal_pending_pixel_xy(goal_state: GoalState) -> np.ndarray | None:
    if goal_state.candidate_kind != "pixel" or goal_state.filtered_candidate_pixel_xy is None:
        return None
    return goal_state.filtered_candidate_pixel_xy.copy()


def goal_is_done(goal_state: GoalState, robot_state: RobotState2D) -> bool:
    if goal_target_mode(goal_state) != "point":
        return False
    target_world_xy = goal_state.target_world_xy
    if target_world_xy is None:
        return False
    distance = float(np.linalg.norm(target_world_xy - np.asarray(robot_state.base_pos_w, dtype=np.float32)[:2]))
    return distance <= goal_state.tolerance


def goal_current_body_xy(goal_state: GoalState, robot_state: RobotState2D) -> np.ndarray:
    if goal_target_mode(goal_state) != "point" or goal_state.target_world_xy is None:
        raise RuntimeError("GoalState does not have an active goal.")
    return _point_goal_body_from_world(goal_state.target_world_xy, robot_state.base_pos_w, robot_state.base_yaw)
