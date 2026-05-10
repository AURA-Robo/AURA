from __future__ import annotations

from typing import Any

from .ontology import (
    ATTRIBUTES,
    FAILURE_TYPES,
    INTENTS,
    PLAN_TEMPLATES,
    QUERY_OPERATORS,
    QUERY_TYPES,
    REPAIR_TEMPLATES,
    SUBGOAL_STATUSES,
    SUBGOAL_TYPES,
    TASK_FRAME_INTENTS,
)


class PlannerValidationError(RuntimeError):
    pass


def _require_dict(payload: Any, name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlannerValidationError(f"{name} must be an object.")
    return payload


def _require_list(payload: Any, name: str) -> list[Any]:
    if not isinstance(payload, list):
        raise PlannerValidationError(f"{name} must be a list.")
    return payload


def _require_str(payload: Any, name: str, allow_null: bool = False) -> str | None:
    if payload is None and allow_null:
        return None
    if not isinstance(payload, str) or not payload.strip():
        raise PlannerValidationError(f"{name} must be a non-empty string.")
    return payload


def _require_target_name(payload: Any, name: str, allow_null: bool = False) -> str | None:
    value = _require_str(payload, name, allow_null=allow_null)
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    if len(normalized) > 80:
        raise PlannerValidationError(f"{name} must be 80 characters or fewer.")
    if any(ord(char) < 32 for char in normalized):
        raise PlannerValidationError(f"{name} must not contain control characters.")
    return normalized


def _require_bool(payload: Any, name: str) -> bool:
    if not isinstance(payload, bool):
        raise PlannerValidationError(f"{name} must be a boolean.")
    return payload


def validate_plan_request(payload: Any) -> dict[str, Any]:
    data = _require_dict(payload, "plan_request")
    _require_str(data.get("utterance_ko"), "plan_request.utterance_ko")
    robot_state = _require_dict(data.get("robot_state"), "plan_request.robot_state")
    if robot_state.get("current_room") is not None:
        _require_str(robot_state.get("current_room"), "plan_request.robot_state.current_room")
    if robot_state.get("holding_object") is not None:
        _require_str(robot_state.get("holding_object"), "plan_request.robot_state.holding_object")
    world_summary = _require_dict(data.get("world_summary"), "plan_request.world_summary")
    known_rooms = _require_list(world_summary.get("known_rooms"), "plan_request.world_summary.known_rooms")
    for idx, room in enumerate(known_rooms):
        _require_str(room, f"plan_request.world_summary.known_rooms[{idx}]")
    recent_seen = _require_list(world_summary.get("recent_seen"), "plan_request.world_summary.recent_seen")
    for idx, item in enumerate(recent_seen):
        row = _require_dict(item, f"plan_request.world_summary.recent_seen[{idx}]")
        _require_str(row.get("class"), f"plan_request.world_summary.recent_seen[{idx}].class")
    capabilities = _require_dict(data.get("capabilities"), "plan_request.capabilities")
    detectable = _require_list(capabilities.get("detectable_objects"), "plan_request.capabilities.detectable_objects")
    for idx, item in enumerate(detectable):
        _require_str(item, f"plan_request.capabilities.detectable_objects[{idx}]")
    inspectable = _require_dict(
        capabilities.get("inspectable_attributes"),
        "plan_request.capabilities.inspectable_attributes",
    )
    for object_class, attrs in inspectable.items():
        _require_str(object_class, "plan_request.capabilities.inspectable_attributes key")
        attr_list = _require_list(attrs, f"plan_request.capabilities.inspectable_attributes.{object_class}")
        for idx, attr in enumerate(attr_list):
            _require_str(attr, f"plan_request.capabilities.inspectable_attributes.{object_class}[{idx}]")
    _require_bool(capabilities.get("can_return_home"), "plan_request.capabilities.can_return_home")
    knowledge_context = data.get("knowledge_context")
    if knowledge_context is not None:
        context_payload = _require_dict(knowledge_context, "plan_request.knowledge_context")
        for field_name in ("hard_rules", "soft_rules", "lexicon_entries", "facts"):
            rows = _require_list(context_payload.get(field_name, []), f"plan_request.knowledge_context.{field_name}")
            for idx, row in enumerate(rows):
                _require_dict(row, f"plan_request.knowledge_context.{field_name}[{idx}]")
    return data


def validate_plan_response(payload: Any, capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    data = _require_dict(payload, "plan_response")
    intent = _require_str(data.get("intent"), "plan_response.intent")
    if intent not in INTENTS:
        raise PlannerValidationError(f"plan_response.intent is not allowed: {intent}")
    plan_template = _require_str(data.get("plan_template"), "plan_response.plan_template")
    if plan_template not in PLAN_TEMPLATES:
        raise PlannerValidationError(f"plan_response.plan_template is not allowed: {plan_template}")
    target = _require_dict(data.get("target"), "plan_response.target")
    target_class = _require_target_name(target.get("class"), "plan_response.target.class", allow_null=True)
    attribute = _require_str(target.get("attribute"), "plan_response.target.attribute", allow_null=True)
    if attribute is not None:
        if target_class is None:
            raise PlannerValidationError("target.attribute requires target.class.")
        if attribute not in ATTRIBUTES.get(target_class, ()):
            raise PlannerValidationError(f"Unsupported attribute {attribute} for {target_class}.")
    hints = _require_dict(data.get("hints"), "plan_response.hints")
    _require_str(hints.get("room"), "plan_response.hints.room", allow_null=True)
    _require_str(hints.get("instance"), "plan_response.hints.instance", allow_null=True)
    constraints = _require_dict(data.get("constraints"), "plan_response.constraints")
    return_home = _require_bool(constraints.get("return_home"), "plan_response.constraints.return_home")
    _require_bool(constraints.get("report_result"), "plan_response.constraints.report_result")
    clarification = _require_dict(data.get("clarification"), "plan_response.clarification")
    clarification_required = _require_bool(
        clarification.get("required"),
        "plan_response.clarification.required",
    )
    question = _require_str(
        clarification.get("question_ko"),
        "plan_response.clarification.question_ko",
        allow_null=True,
    )
    if clarification_required and question is None:
        raise PlannerValidationError("clarification.required=true needs question_ko.")
    if intent == "inspect_attribute":
        if target_class is None or attribute is None:
            raise PlannerValidationError("inspect_attribute requires target.class and target.attribute.")
    if intent in {"find_object", "navigate_to_object"} and target_class is None:
        raise PlannerValidationError(f"{intent} requires target.class.")
    if capabilities is not None and intent not in {"ask_clarification", "unsupported"}:
        validate_capabilities(target_class, attribute, return_home, capabilities)
    return data


def validate_task_frame_response(payload: Any, capabilities: dict[str, Any] | None = None) -> dict[str, Any]:
    data = _require_dict(payload, "task_frame")
    intent = _require_str(data.get("intent"), "task_frame.intent")
    if intent not in TASK_FRAME_INTENTS:
        raise PlannerValidationError(f"task_frame.intent is not allowed: {intent}")

    target = _require_dict(data.get("target"), "task_frame.target")
    target_object = _require_target_name(target.get("object"), "task_frame.target.object", allow_null=True)
    _require_str(target.get("instance_hint"), "task_frame.target.instance_hint", allow_null=True)
    _require_str(target.get("location_hint"), "task_frame.target.location_hint", allow_null=True)

    query = _require_dict(data.get("query"), "task_frame.query")
    query_type = _require_str(query.get("query_type"), "task_frame.query.query_type", allow_null=True)
    attribute = _require_str(query.get("attribute"), "task_frame.query.attribute", allow_null=True)
    operator = _require_str(query.get("operator"), "task_frame.query.operator", allow_null=True)
    expected_value = _require_str(
        query.get("expected_value"),
        "task_frame.query.expected_value",
        allow_null=True,
    )
    if query_type is not None and query_type not in QUERY_TYPES:
        raise PlannerValidationError(f"task_frame.query.query_type is not allowed: {query_type}")
    if operator is not None and operator not in QUERY_OPERATORS:
        raise PlannerValidationError(f"task_frame.query.operator is not allowed: {operator}")
    if attribute is not None:
        if target_object is None:
            raise PlannerValidationError("task_frame.query.attribute requires task_frame.target.object.")
        if attribute not in ATTRIBUTES.get(target_object, ()):
            raise PlannerValidationError(
                f"Unsupported task_frame.query.attribute {attribute} for {target_object}."
            )
    if expected_value is not None and attribute is None:
        raise PlannerValidationError("task_frame.query.expected_value requires task_frame.query.attribute.")

    constraints = _require_dict(data.get("constraints"), "task_frame.constraints")
    return_after_check = _require_bool(
        constraints.get("return_after_check"),
        "task_frame.constraints.return_after_check",
    )
    _require_bool(
        constraints.get("report_result"),
        "task_frame.constraints.report_result",
    )

    clarification = _require_dict(data.get("clarification"), "task_frame.clarification")
    clarification_required = _require_bool(
        clarification.get("required"),
        "task_frame.clarification.required",
    )
    question = _require_str(
        clarification.get("question_ko"),
        "task_frame.clarification.question_ko",
        allow_null=True,
    )
    if clarification_required and question is None:
        raise PlannerValidationError("task_frame clarification requires question_ko.")

    if intent == "check_state":
        if target_object is None:
            raise PlannerValidationError("check_state requires task_frame.target.object.")
        if query_type != "attribute_check":
            raise PlannerValidationError("check_state requires task_frame.query.query_type=attribute_check.")
        if attribute is None:
            raise PlannerValidationError("check_state requires task_frame.query.attribute.")
        if operator != "equals":
            raise PlannerValidationError("check_state requires task_frame.query.operator=equals.")
    elif intent in {"find_object", "navigate_to_object"}:
        if target_object is None:
            raise PlannerValidationError(f"{intent} requires task_frame.target.object.")
        if any(value is not None for value in (query_type, attribute, operator, expected_value)):
            raise PlannerValidationError(f"{intent} does not accept task_frame.query values.")
    elif intent in {"ask_clarification", "unsupported"}:
        if return_after_check:
            raise PlannerValidationError(f"{intent} does not support return_after_check.")

    if capabilities is not None and intent not in {"ask_clarification", "unsupported"}:
        validate_capabilities(target_object, attribute, return_after_check, capabilities)
    return data


def validate_repair_request(payload: Any) -> dict[str, Any]:
    data = _require_dict(payload, "repair_request")
    failure_type = _require_str(data.get("failure_type"), "repair_request.failure_type")
    if failure_type not in FAILURE_TYPES:
        raise PlannerValidationError(f"repair_request.failure_type is not supported: {failure_type}")
    target_class = _require_target_name(data.get("target_class"), "repair_request.target_class", allow_null=True)
    searched_rooms = _require_list(data.get("searched_rooms"), "repair_request.searched_rooms")
    remaining_rooms = _require_list(data.get("remaining_rooms"), "repair_request.remaining_rooms")
    for idx, room in enumerate([*searched_rooms, *remaining_rooms]):
        _require_str(room, f"repair_request.rooms[{idx}]")
    recent_seen = data.get("recent_seen")
    if recent_seen is not None:
        _require_dict(recent_seen, "repair_request.recent_seen")
    retries = data.get("retries")
    if not isinstance(retries, int) or retries < 0:
        raise PlannerValidationError("repair_request.retries must be a non-negative integer.")
    return data


def validate_repair_response(payload: Any) -> dict[str, Any]:
    data = _require_dict(payload, "repair_response")
    repair_template = _require_str(data.get("repair_template"), "repair_response.repair_template")
    if repair_template not in REPAIR_TEMPLATES:
        raise PlannerValidationError(f"repair_response.repair_template is not allowed: {repair_template}")
    clarification = _require_dict(data.get("clarification"), "repair_response.clarification")
    required = _require_bool(clarification.get("required"), "repair_response.clarification.required")
    question = _require_str(
        clarification.get("question_ko"),
        "repair_response.clarification.question_ko",
        allow_null=True,
    )
    if required and question is None:
        raise PlannerValidationError("repair clarification requires a question.")
    return data


def validate_capabilities(
    target_class: str | None,
    attribute: str | None,
    return_home: bool,
    capabilities: dict[str, Any],
) -> None:
    inspectable = capabilities.get("inspectable_attributes", {})
    if attribute is not None:
        allowed = set(inspectable.get(target_class or "", ()))
        if attribute not in allowed:
            raise PlannerValidationError(
                f"Capability check failed: {target_class}.{attribute} is not inspectable."
            )
    if return_home and not capabilities.get("can_return_home", False):
        raise PlannerValidationError("Capability check failed: return_home requested but unsupported.")


def validate_subgoal(payload: Any) -> dict[str, Any]:
    data = _require_dict(payload, "subgoal")
    _require_str(data.get("id"), "subgoal.id")
    subgoal_type = _require_str(data.get("type"), "subgoal.type")
    if subgoal_type not in SUBGOAL_TYPES:
        raise PlannerValidationError(f"subgoal.type is not allowed: {subgoal_type}")
    status = _require_str(data.get("status"), "subgoal.status")
    if status not in SUBGOAL_STATUSES:
        raise PlannerValidationError(f"subgoal.status is not allowed: {status}")
    _require_bool(data.get("succeed"), "subgoal.succeed")
    _require_dict(data.get("input"), "subgoal.input")
    _require_dict(data.get("output"), "subgoal.output")
    attempts = data.get("attempts")
    if not isinstance(attempts, int) or attempts < 0:
        raise PlannerValidationError("subgoal.attempts must be a non-negative integer.")
    _require_str(data.get("failure_reason"), "subgoal.failure_reason", allow_null=True)
    return data
