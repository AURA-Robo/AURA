from __future__ import annotations

from dataclasses import dataclass

from systems.memory.api import ConversationContext
from systems.reasoning.planner.aura_adapter import build_plan_request
from systems.reasoning.planner.normalizer import detect_object_class, normalize_text
from systems.reasoning.planner.planner_service import PlannerService


_TARGET_REFERENCE_TOKENS = (
    "그거",
    "그것",
    "걔",
    "저거",
    "that",
    "it",
)


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: str
    normalized_utterance: str
    interpreted_utterance: str
    intent_candidate: str | None
    reason: str | None = None
    confidence: float = 0.0


class InputInterpreter:
    def __init__(self, planner_service: PlannerService) -> None:
        self._planner_service = planner_service

    def interpret(
        self,
        utterance: str,
        *,
        conversation_context: ConversationContext,
        scene_preset: str | None,
        task_active: bool,
        interrupt_current_task: bool,
    ) -> RouteDecision:
        interpreted_utterance = self._resolve_references(utterance, conversation_context.resolved_slots)
        request = build_plan_request(
            interpreted_utterance,
            planning_context={
                "current_room": conversation_context.resolved_slots.get("last_room_hint"),
                "scene_preset": scene_preset,
            },
        )
        classification = self._planner_service.classify_route(request)

        if classification.route == "task" and task_active and not interrupt_current_task:
            return RouteDecision(
                route="busy",
                normalized_utterance=classification.normalized_utterance,
                interpreted_utterance=interpreted_utterance,
                intent_candidate=classification.intent_candidate,
                reason="active_task_in_progress",
                confidence=0.95,
            )
        return RouteDecision(
            route=classification.route,
            normalized_utterance=classification.normalized_utterance,
            interpreted_utterance=interpreted_utterance,
            intent_candidate=classification.intent_candidate,
            reason=classification.reason,
            confidence=classification.confidence,
        )

    def _resolve_references(self, utterance: str, resolved_slots: dict[str, str]) -> str:
        candidate = " ".join(str(utterance).strip().split())
        if candidate == "":
            return candidate
        if detect_object_class(candidate) is not None:
            return candidate
        normalized = normalize_text(candidate)
        target_object = str(resolved_slots.get("last_target_object") or "").strip()
        if target_object and any(token in normalized for token in _TARGET_REFERENCE_TOKENS):
            replacement = target_object.replace("_", " ")
            for token in _TARGET_REFERENCE_TOKENS:
                candidate = candidate.replace(token, replacement)
        return candidate
