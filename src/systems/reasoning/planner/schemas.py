from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict


class RobotState(TypedDict):
    current_room: str | None
    holding_object: str | None


class WorldSummary(TypedDict):
    known_rooms: list[str]
    recent_seen: list[dict[str, Any]]


class KnowledgeContextPayload(TypedDict):
    hard_rules: list[dict[str, Any]]
    soft_rules: list[dict[str, Any]]
    lexicon_entries: list[dict[str, Any]]
    facts: list[dict[str, Any]]


class AgentMemoryContextPayload(TypedDict):
    core_blocks: list[dict[str, Any]]
    archival_passages: list[dict[str, Any]]
    conversation_summary: str
    recent_turns: list[dict[str, Any]]
    object_memory: list[dict[str, Any]]
    knowledge_facts: list[dict[str, Any]]
    metadata: dict[str, Any]


class PlanningContext(TypedDict, total=False):
    current_room: str | None
    holding_object: str | None
    known_rooms: list[str]
    recent_seen: list[dict[str, Any]]
    capabilities: dict[str, Any]
    scene_preset: str
    knowledge_context: KnowledgeContextPayload
    agent_memory: AgentMemoryContextPayload


class Capabilities(TypedDict):
    detectable_objects: list[str]
    inspectable_attributes: dict[str, list[str]]
    can_return_home: bool


class PlanRequest(TypedDict):
    utterance_ko: str
    robot_state: RobotState
    world_summary: WorldSummary
    capabilities: Capabilities
    knowledge_context: KnowledgeContextPayload
    agent_memory: AgentMemoryContextPayload


class PlanHints(TypedDict):
    room: str | None
    instance: str | None


class PlanConstraints(TypedDict):
    return_home: bool
    report_result: bool


class PlanClarification(TypedDict):
    required: bool
    question_ko: str | None


class TaskFrameTarget(TypedDict):
    object: str | None
    instance_hint: str | None
    location_hint: str | None


class TaskFrameQuery(TypedDict):
    query_type: str | None
    attribute: str | None
    operator: str | None
    expected_value: str | None


class TaskFrameConstraints(TypedDict):
    return_after_check: bool
    report_result: bool


class TaskFrameClarification(TypedDict):
    required: bool
    question_ko: str | None


class TaskFrame(TypedDict):
    intent: str
    target: TaskFrameTarget
    query: TaskFrameQuery
    constraints: TaskFrameConstraints
    clarification: TaskFrameClarification


class NavigationTargetPayload(TypedDict, total=False):
    mode: str
    object_id: str
    class_name: str
    scene_scope: str | None
    world_pose_xyz: list[float]
    pose_age_sec: int
    stop_radius_m: float
    reacquire_radius_m: float
    reacquire_timeout_sec: float


class PlanResponse(TypedDict):
    intent: str
    plan_template: str
    target: dict[str, Any]
    hints: PlanHints
    constraints: PlanConstraints
    clarification: PlanClarification


class RepairRequest(TypedDict):
    failure_type: str
    target_class: str | None
    searched_rooms: list[str]
    remaining_rooms: list[str]
    recent_seen: dict[str, Any] | None
    retries: int


class RepairClarification(TypedDict):
    required: bool
    question_ko: str | None


class RepairResponse(TypedDict):
    repair_template: str
    clarification: RepairClarification


class Subgoal(TypedDict):
    id: str
    type: str
    status: str
    succeed: bool
    input: dict[str, Any]
    output: dict[str, Any]
    attempts: int
    failure_reason: str | None


@dataclass(frozen=True)
class CompiledNode:
    type: str
    target: str | None = None
    attribute: str | None = None
    room_hint: str | None = None
    instance_hint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompiledMission:
    intent: str
    plan_template: str
    target_class: str | None
    attribute: str | None
    hints: dict[str, str | None]
    constraints: dict[str, bool]
    nodes: list[CompiledNode]


@dataclass(frozen=True)
class CompiledRepair:
    repair_template: str
    target_class: str | None
    nodes: list[CompiledNode]


@dataclass(frozen=True)
class CompletionDecision:
    done: bool
    success: bool
    reason: str | None = None
    retryable: bool = False
