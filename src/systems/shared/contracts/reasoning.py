from __future__ import annotations

from typing import Literal, TypedDict


ReasoningRoute = Literal["task", "dialogue", "clarification", "unsupported", "busy"]


class ReasoningRequest(TypedDict, total=False):
    utterance: str
    language: str
    conversation_id: str
    scene_preset: str | None
    interrupt_current_task: bool


class RouteDecision(TypedDict, total=False):
    route: ReasoningRoute
    reason: str | None
    confidence: float
    normalized_utterance: str
    intent_candidate: str | None


class ReasoningTaskPayload(TypedDict, total=False):
    task_id: str
    task_status: str
    task_frame: dict
    current_subgoal: dict | None
    subgoals: list[dict]


class ReasoningResponse(TypedDict, total=False):
    ok: bool
    route: ReasoningRoute
    request_id: str
    conversation_id: str
    reply_text: str | None
    task: ReasoningTaskPayload | None
    error: str | None
