from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from systems.memory.api import (
    AgentMemoryContext,
    KnowledgeContext,
    KnowledgeRuntimeHandle,
    ObjectMemoryRuntimeHandle,
    agent_memory_context_payload,
    inject_knowledge_context_into_plan_request,
)
from systems.reasoning.planner.orchestration import (
    InspectCompletionChecker,
    InspectExecutor,
    NavigateCompletionChecker,
    NavigateExecutor,
    ReportCompletionChecker,
    ReportExecutor,
    SubgoalOrchestrator,
    SubgoalExpander,
    SubgoalRuntimeGuard,
)
from systems.reasoning.planner.planner_service import PlannerService
from systems.reasoning.planner.schemas import PlanningContext
from systems.reasoning.planner.reporting import (
    build_inspection_question,
    build_navigation_instruction,
    observed_value_from_answer,
    render_report_message,
)
from systems.reasoning.planner.validator import validate_task_frame_response
from systems.reasoning.planner_catalog_runtime import PlannerCatalogRuntimeHandle
from systems.inference.api.planner import CompletionFn


DEFAULT_CAPABILITIES = {
    "detectable_objects": [
        "tv",
        "sofa",
        "bed",
        "chair",
        "refrigerator",
        "door",
        "purple_box_cart",
        "bus",
    ],
    "inspectable_attributes": {
        "tv": ["power_state"],
        "door": ["open_state"],
    },
    "can_return_home": True,
}


def build_plan_request(
    instruction: str,
    *,
    planning_context: PlanningContext | None = None,
) -> dict[str, Any]:
    context = planning_context or {}
    request = {
        "utterance_ko": str(instruction),
        "robot_state": {
            "current_room": context.get("current_room"),
            "holding_object": context.get("holding_object"),
        },
        "world_summary": {
            "known_rooms": list(context.get("known_rooms") or []),
            "recent_seen": list(context.get("recent_seen") or []),
        },
        "capabilities": dict(context.get("capabilities") or DEFAULT_CAPABILITIES),
    }
    knowledge_context = context.get("knowledge_context")
    if isinstance(knowledge_context, KnowledgeContext):
        request = inject_knowledge_context_into_plan_request(request, knowledge_context)
    elif isinstance(knowledge_context, dict):
        request["knowledge_context"] = dict(knowledge_context)
    else:
        request["knowledge_context"] = {
            "hard_rules": [],
            "soft_rules": [],
            "lexicon_entries": [],
            "facts": [],
        }
    agent_memory = context.get("agent_memory")
    if isinstance(agent_memory, AgentMemoryContext):
        request["agent_memory"] = agent_memory_context_payload(agent_memory)
    elif isinstance(agent_memory, dict):
        request["agent_memory"] = dict(agent_memory)
    else:
        request["agent_memory"] = {
            "core_blocks": [],
            "archival_passages": [],
            "conversation_summary": "",
            "recent_turns": [],
            "object_memory": [],
            "knowledge_facts": [],
            "metadata": {
                "enabled": False,
                "available": False,
                "degraded_reason": None,
            },
        }
    return request


class AuraSubgoalExpander(SubgoalExpander):
    def __init__(self, planner_catalog_runtime: PlannerCatalogRuntimeHandle | None = None) -> None:
        super().__init__(planner_catalog_runtime=planner_catalog_runtime)

    def expand(self, task_frame: dict[str, Any]) -> list[dict[str, Any]]:
        task_frame = validate_task_frame_response(task_frame, DEFAULT_CAPABILITIES)
        return super().expand(task_frame)


class AuraNavigateExecutor(NavigateExecutor):
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = runtime or {}
        controller = runtime["controller"]
        task_state = runtime.get("task_state")
        previous = subgoal.get("output", {})

        if subgoal["type"] == "return":
            origin_pose = None if task_state is None else task_state.origin_pose
            if origin_pose is None:
                return {"error": "origin_pose_unavailable"}
            if not previous.get("started"):
                start_info = controller.start_return_to_origin(origin_pose)
            else:
                start_info = {}
            snapshot = controller.navigation_snapshot(origin_pose=origin_pose)
            return {"started": True, "mode": "return", "snapshot": snapshot, **start_info}

        navigation_target = subgoal["input"].get("navigation_target")
        if isinstance(navigation_target, dict) and navigation_target.get("mode") == "memory_pose":
            if not previous.get("started") or previous.get("navigation_target") != navigation_target:
                start_info = controller.start_navigation_memory_goal(navigation_target)
            else:
                start_info = {}
            origin_pose = None if task_state is None else task_state.origin_pose
            snapshot = controller.navigation_snapshot(origin_pose=origin_pose)
            return {
                "started": True,
                "mode": "navigate",
                "navigation_target": dict(navigation_target),
                "snapshot": snapshot,
                **start_info,
            }

        prompt = build_navigation_instruction(subgoal["input"]["target"], language="en")
        if not previous.get("started") or previous.get("prompt") != prompt:
            start_info = controller.start_navigation_instruction(prompt, "en")
        else:
            start_info = {}
        origin_pose = None if task_state is None else task_state.origin_pose
        snapshot = controller.navigation_snapshot(origin_pose=origin_pose)
        return {"started": True, "mode": "navigate", "prompt": prompt, "snapshot": snapshot, **start_info}


class AuraInspectExecutor(InspectExecutor):
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = runtime or {}
        controller = runtime["controller"]
        previous = subgoal.get("output", {})
        question = build_inspection_question(subgoal["input"]["target"], subgoal["input"]["query"])
        answer_text = controller.check_binary_question(question)
        if answer_text not in {"true", "false"}:
            return {"error": f"invalid_binary_answer:{answer_text}"}

        answer_is_true = answer_text == "true"
        observations = list(previous.get("observations", []))
        observations.append(answer_is_true)
        confidence = 0.0
        if observations:
            dominant = max(observations.count(True), observations.count(False))
            confidence = float(dominant / len(observations))
        observed_value = observed_value_from_answer(subgoal["input"]["query"], answer_is_true)
        return {
            "question": question,
            "answer": answer_text,
            "target_visible": True,
            "observations": observations,
            "observation_consistent": len(observations) >= 3 and len(set(observations[-3:])) == 1,
            "confidence": confidence,
            "observed_value": observed_value,
            "decision": bool(answer_is_true),
        }


class AuraReportExecutor(ReportExecutor):
    def execute(self, subgoal: dict[str, Any], runtime: dict[str, Any] | None = None) -> dict[str, Any]:
        runtime = runtime or {}
        controller = runtime["controller"]
        task_state = runtime["task_state"]
        message = render_report_message(task_state.task_frame, task_state.subgoals)
        controller.set_last_report(message)
        return {"message": message, "delivered": True}


class AuraNavigateCompletionChecker(NavigateCompletionChecker):
    def check(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ):
        del runtime
        if raw_output.get("error"):
            return super().check(subgoal, raw_output, None)
        snapshot = raw_output.get("snapshot", {})
        reacquire_state = str(snapshot.get("reacquire_state") or "")
        if reacquire_state == "reacquired":
            from systems.reasoning.planner.schemas import CompletionDecision

            return CompletionDecision(done=True, success=True, reason="memory_target_reacquired")
        if reacquire_state in {"failed", "reacquire_failed"}:
            from systems.reasoning.planner.schemas import CompletionDecision

            return CompletionDecision(done=True, success=False, reason="memory_target_not_reacquired")
        locomotion = np.asarray(snapshot.get("locomotion_command", (0.0, 0.0, 0.0)), dtype=np.float32).reshape(3)
        zero_locomotion = float(np.linalg.norm(locomotion)) <= 0.05
        stable_state = str(snapshot.get("state_label") or "") in {"done", "waiting", "tracking", "stale-hold"}

        if subgoal["type"] == "return":
            if bool(snapshot.get("return_pose_reached")) and zero_locomotion and stable_state:
                from systems.reasoning.planner.schemas import CompletionDecision

                return CompletionDecision(done=True, success=True, reason="return_goal_reached")
            from systems.reasoning.planner.schemas import CompletionDecision

            return CompletionDecision(done=False, success=False, reason="return_incomplete", retryable=True)

        system2_status = str(snapshot.get("system2_status") or "").strip().lower()
        system2_mode = str(snapshot.get("system2_decision_mode") or "").strip().lower()
        action_override_mode = snapshot.get("action_override_mode")
        planner_target_mode = str(snapshot.get("planner_target_mode") or "none")
        stop_like = system2_status == "stop" or system2_mode == "stop"
        if stop_like and action_override_mode is None and zero_locomotion and stable_state and planner_target_mode == "none":
            from systems.reasoning.planner.schemas import CompletionDecision

            return CompletionDecision(done=True, success=True, reason="navigate_stopped")
        from systems.reasoning.planner.schemas import CompletionDecision

        return CompletionDecision(done=False, success=False, reason="navigate_incomplete", retryable=True)


class AuraInspectCompletionChecker(InspectCompletionChecker):
    def check(
        self,
        subgoal: dict[str, Any],
        raw_output: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ):
        del subgoal, runtime
        from systems.reasoning.planner.schemas import CompletionDecision

        if raw_output.get("error"):
            return CompletionDecision(done=True, success=False, reason=str(raw_output["error"]))
        observations = list(raw_output.get("observations", []))
        if len(observations) < self.stable_frames:
            return CompletionDecision(done=False, success=False, reason="inspection_incomplete", retryable=True)
        tail = observations[-self.stable_frames :]
        consistent = len(set(tail)) == 1
        confidence = float(raw_output.get("confidence", 0.0))
        if consistent and confidence >= self.min_confidence:
            return CompletionDecision(done=True, success=True, reason="inspection_settled")
        return CompletionDecision(done=False, success=False, reason="inspection_incomplete", retryable=True)


class AuraKnowledgeRuntimeGuard(SubgoalRuntimeGuard):
    def __init__(
        self,
        knowledge_runtime: KnowledgeRuntimeHandle,
        *,
        scene_scope: str | None = None,
    ) -> None:
        self.knowledge_runtime = knowledge_runtime
        self.scene_scope = " ".join(str(scene_scope or "").strip().split()) or None

    def evaluate(
        self,
        subgoal: dict[str, Any],
        runtime: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        runtime = runtime or {}
        task_state = runtime.get("task_state")
        task_frame = getattr(task_state, "task_frame", None)
        if not isinstance(task_frame, dict):
            return None
        scene_scope = str(runtime.get("scene_preset") or self.scene_scope or "").strip() or None
        result = self.knowledge_runtime.evaluate_task_frame(task_frame, scene_scope=scene_scope)
        if result.allowed and not result.mutated:
            return None
        if result.allowed and result.task_frame == task_frame:
            return None
        if result.allowed:
            reason = "knowledge rule requires replanning before execution"
        else:
            reason = str(result.reason or "knowledge rule denied execution")
        if self.knowledge_runtime.service is not None and result.applied_rule_ids:
            self.knowledge_runtime.service.audit_rule_application(
                rule_ids=list(result.applied_rule_ids),
                phase="execution",
                task_id=str(getattr(task_state, "task_id", "") or "") or None,
                subgoal_id=str(subgoal.get("id") or "") or None,
                payload={"task_frame": task_frame, "guard_result": result.task_frame},
            )
        return {
            "reason": reason,
            "message": reason,
            "applied_rule_ids": list(result.applied_rule_ids),
        }


class AuraTaskingAdapter:
    def __init__(
        self,
        completion: CompletionFn | None = None,
        intent_completion: CompletionFn | None = None,
        *,
        model: str,
        timeout: float,
        knowledge_runtime: KnowledgeRuntimeHandle | None = None,
        planner_catalog_runtime: PlannerCatalogRuntimeHandle | None = None,
        scene_scope: str | None = None,
    ) -> None:
        self.knowledge_runtime = knowledge_runtime
        self.planner_catalog_runtime = planner_catalog_runtime
        self.scene_scope = " ".join(str(scene_scope or "").strip().split()) or None
        self.planner_service = PlannerService(
            completion=completion,
            intent_completion=intent_completion,
            planner_catalog_runtime=planner_catalog_runtime,
            model=model,
            timeout=timeout,
        )
        self.orchestrator = SubgoalOrchestrator(
            navigate_executor=AuraNavigateExecutor(),
            inspect_executor=AuraInspectExecutor(),
            report_executor=AuraReportExecutor(),
            navigate_checker=AuraNavigateCompletionChecker(),
            inspect_checker=AuraInspectCompletionChecker(),
            report_checker=ReportCompletionChecker(),
            expander=AuraSubgoalExpander(planner_catalog_runtime=planner_catalog_runtime),
            runtime_guard=(
                AuraKnowledgeRuntimeGuard(knowledge_runtime, scene_scope=self.scene_scope)
                if knowledge_runtime is not None
                else None
            ),
        )

    @property
    def planner_catalog_runtime(self) -> PlannerCatalogRuntimeHandle | None:
        return self._planner_catalog_runtime

    @planner_catalog_runtime.setter
    def planner_catalog_runtime(self, value: PlannerCatalogRuntimeHandle | None) -> None:
        self._planner_catalog_runtime = value
        if hasattr(self, "planner_service"):
            self.planner_service.planner_catalog_runtime = value
        if hasattr(self, "orchestrator") and isinstance(self.orchestrator.expander, SubgoalExpander):
            self.orchestrator.expander.planner_catalog_runtime = value

    def plan_task_frame(
        self,
        instruction: str,
        *,
        planning_context: PlanningContext | None = None,
    ) -> dict[str, Any]:
        context = dict(planning_context or {})
        scene_scope = str(context.get("scene_preset") or self.scene_scope or "").strip() or None
        if self.knowledge_runtime is not None and "knowledge_context" not in context:
            context["knowledge_context"] = self.knowledge_runtime.retrieve_for_plan(
                instruction,
                scene_scope=scene_scope,
            )
        return self.planner_service.plan_task_frame(
            build_plan_request(instruction, planning_context=context)
        )

    def initialize_subgoals(self, task_frame: dict[str, Any]) -> list[dict[str, Any]]:
        return self.orchestrator.initialize(task_frame)

    def resolve_memory_navigation(
        self,
        task_frame: dict[str, Any],
        subgoals: list[dict[str, Any]],
        *,
        object_memory_runtime: ObjectMemoryRuntimeHandle | None,
        scene_scope: str | None,
        max_pose_age_sec: int = 600,
        stop_radius_m: float = 0.8,
        reacquire_radius_m: float = 1.0,
        reacquire_timeout_sec: float = 4.0,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
        if object_memory_runtime is None or not object_memory_runtime.enabled:
            return task_frame, subgoals, None
        if task_frame.get("intent") not in {"navigate_to_object", "find_object", "check_state"}:
            return task_frame, subgoals, None
        target_payload = task_frame.get("target")
        if not isinstance(target_payload, dict):
            return task_frame, subgoals, None
        class_name = str(target_payload.get("object") or "").strip()
        if not class_name:
            return task_frame, subgoals, None

        resolution = object_memory_runtime.resolve_navigation_target(
            class_name=class_name,
            scene_scope=scene_scope,
            room_hint=str(target_payload.get("location_hint") or "").strip() or None,
            instance_hint=str(target_payload.get("instance_hint") or "").strip() or None,
            max_pose_age_sec=max_pose_age_sec,
        )
        resolution_payload = {
            "status": resolution.status,
            "selected": None if resolution.selected is None else {
                "object_id": resolution.selected.object_id,
                "class_name": resolution.selected.class_name,
                "room_id": resolution.selected.room_id,
                "scene_scope": resolution.selected.scene_scope,
                "world_pose_xyz": list(resolution.selected.world_pose_xyz),
                "world_pose_observed_at": resolution.selected.world_pose_observed_at.isoformat(),
                "pose_age_sec": resolution.selected.pose_age_sec,
            },
            "candidates": [
                {
                    "object_id": candidate.object_id,
                    "class_name": candidate.class_name,
                    "room_id": candidate.room_id,
                    "scene_scope": candidate.scene_scope,
                    "world_pose_xyz": list(candidate.world_pose_xyz),
                    "world_pose_observed_at": candidate.world_pose_observed_at.isoformat(),
                    "pose_age_sec": candidate.pose_age_sec,
                }
                for candidate in resolution.candidates
            ],
            "debug": dict(resolution.debug),
        }
        if resolution.status == "ambiguous":
            clarification_frame = self._memory_ambiguity_task_frame(task_frame)
            return clarification_frame, self.initialize_subgoals(clarification_frame), resolution_payload
        if resolution.status != "resolved" or resolution.selected is None:
            return task_frame, subgoals, resolution_payload

        navigation_target = {
            "mode": "memory_pose",
            "object_id": resolution.selected.object_id,
            "class_name": resolution.selected.class_name,
            "scene_scope": resolution.selected.scene_scope,
            "world_pose_xyz": list(resolution.selected.world_pose_xyz),
            "pose_age_sec": resolution.selected.pose_age_sec,
            "stop_radius_m": float(stop_radius_m),
            "reacquire_radius_m": float(reacquire_radius_m),
            "reacquire_timeout_sec": float(reacquire_timeout_sec),
        }
        for subgoal in subgoals:
            if subgoal.get("type") != "navigate":
                continue
            input_payload = subgoal.get("input")
            if not isinstance(input_payload, dict):
                continue
            input_payload["navigation_target"] = dict(navigation_target)
            break
        return task_frame, subgoals, resolution_payload

    def step(self, subgoals: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any] | None:
        return self.orchestrator.step(subgoals, runtime)

    @staticmethod
    def _memory_ambiguity_task_frame(task_frame: dict[str, Any]) -> dict[str, Any]:
        target_payload = task_frame.get("target")
        target_object = None
        if isinstance(target_payload, dict):
            target_object = str(target_payload.get("object") or "").strip() or None
        label = "target" if target_object is None else target_object.replace("_", " ")
        return {
            "intent": "ask_clarification",
            "target": {
                "object": target_object,
                "instance_hint": None,
                "location_hint": None,
            },
            "query": {
                "query_type": None,
                "attribute": None,
                "operator": None,
                "expected_value": None,
            },
            "constraints": {
                "return_after_check": False,
                "report_result": True,
            },
            "clarification": {
                "required": True,
                "question_ko": f"어느 {label}인지 알려주세요. 최근 본 후보가 여러 개 있습니다.",
            },
        }


@dataclass(slots=True)
class PlannerConfig:
    base_url: str
    model: str
    timeout: float
