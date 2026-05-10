from __future__ import annotations

from systems.memory.api import ConversationContext
from systems.reasoning.interpreter import InputInterpreter
from systems.reasoning.planner.aura_adapter import build_plan_request
from systems.reasoning.planner.planner_service import PlannerService


def test_planner_service_classify_route_uses_intent_completion_only() -> None:
    intent_calls: list[list[dict[str, str]]] = []

    def _intent_completion(messages, model, timeout, temperature, max_tokens):  # noqa: ANN001
        del model, timeout, temperature, max_tokens
        intent_calls.append(list(messages))
        return '{"route":"dialogue","intent_candidate":"smalltalk","reason":"casual_chat","confidence":0.91}'

    def _planning_completion(messages, model, timeout, temperature, max_tokens):  # noqa: ANN001
        del messages, model, timeout, temperature, max_tokens
        raise AssertionError("planning completion must not be used for route classification")

    planner = PlannerService(completion=_planning_completion, intent_completion=_intent_completion)

    classification = planner.classify_route(build_plan_request("hello there"))

    assert classification.route == "dialogue"
    assert classification.intent_candidate == "smalltalk"
    assert len(intent_calls) == 1
    assert [message["role"] for message in intent_calls[0]] == ["system", "user"]


def test_planner_service_classify_route_falls_back_to_heuristic_on_invalid_json() -> None:
    planner = PlannerService(intent_completion=lambda *args: "not-json")  # noqa: ARG005

    classification = planner.classify_route(build_plan_request("go to the tv"))

    assert classification.route == "task"
    assert classification.reason == "heuristic_task"


def test_planner_service_route_guard_accepts_detected_zero_shot_target() -> None:
    planner = PlannerService(
        intent_completion=lambda *args: (  # noqa: ARG005
            '{"route":"clarification","intent_candidate":"unsupported","reason":"missing_target","confidence":0.5}'
        )
    )

    classification = planner.classify_route(build_plan_request("보라색 상자를 찾아서 돌아와"))

    assert classification.route == "task"
    assert classification.intent_candidate == "find_object"
    assert classification.reason == "local_target_detected"


def test_input_interpreter_applies_local_busy_override() -> None:
    planner = PlannerService(
        intent_completion=lambda *args: (  # noqa: ARG005
            '{"route":"task","intent_candidate":"navigate_to_object","reason":"move","confidence":0.88}'
        )
    )
    interpreter = InputInterpreter(planner)
    context = ConversationContext(
        conversation_id="conv-1",
        summary="",
        resolved_slots={},
        recent_turns=[],
    )

    decision = interpreter.interpret(
        "go to the tv",
        conversation_context=context,
        scene_preset=None,
        task_active=True,
        interrupt_current_task=False,
    )

    assert decision.route == "busy"
    assert decision.reason == "active_task_in_progress"


def test_planner_service_plan_task_frame_uses_planning_completion_only() -> None:
    def _intent_completion(messages, model, timeout, temperature, max_tokens):  # noqa: ANN001
        del messages, model, timeout, temperature, max_tokens
        raise AssertionError("intent completion must not be used for task-frame planning")

    planning_calls: list[list[dict[str, str]]] = []

    def _planning_completion(messages, model, timeout, temperature, max_tokens):  # noqa: ANN001
        del model, timeout, temperature, max_tokens
        planning_calls.append(list(messages))
        return """
        {
          "intent": "navigate_to_object",
          "target": {"object": "tv", "instance_hint": null, "location_hint": null},
          "query": {"query_type": null, "attribute": null, "operator": null, "expected_value": null},
          "constraints": {"return_after_check": false, "report_result": true},
          "clarification": {"required": false, "question_ko": null}
        }
        """

    planner = PlannerService(completion=_planning_completion, intent_completion=_intent_completion)

    task_frame = planner.plan_task_frame(build_plan_request("go to the tv"))

    assert task_frame["intent"] == "navigate_to_object"
    assert len(planning_calls) == 1
    assert [message["role"] for message in planning_calls[0]] == ["system", "user"]


def test_planner_service_plan_task_frame_accepts_zero_shot_model_object() -> None:
    def _planning_completion(messages, model, timeout, temperature, max_tokens):  # noqa: ANN001
        del messages, model, timeout, temperature, max_tokens
        return """
        {
          "intent": "find_object",
          "target": {"object": "purple_box", "instance_hint": null, "location_hint": null},
          "query": {"query_type": null, "attribute": null, "operator": null, "expected_value": null},
          "constraints": {"return_after_check": true, "report_result": true},
          "clarification": {"required": false, "question_ko": null}
        }
        """

    planner = PlannerService(completion=_planning_completion)

    task_frame = planner.plan_task_frame(build_plan_request("보라색 상자를 찾아서 돌아와"))

    assert task_frame["intent"] == "find_object"
    assert task_frame["target"]["object"] == "purple_box"
    assert task_frame["constraints"]["return_after_check"] is True
