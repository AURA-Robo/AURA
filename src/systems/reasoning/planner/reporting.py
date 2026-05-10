from __future__ import annotations

from typing import Any

from .ontology import OBJECT_DISPLAY_EN, OBJECT_DISPLAY_KO


OPPOSITE_EXPECTED_VALUES = {
    "off": "on",
    "on": "off",
    "open": "closed",
    "closed": "open",
}


def display_object_ko(name: str | None) -> str:
    if name is None:
        return "target"
    return OBJECT_DISPLAY_KO.get(name, name.replace("_", " "))


def display_object_en(name: str | None) -> str:
    if name is None:
        return "target"
    return OBJECT_DISPLAY_EN.get(name, name.replace("_", " "))


def build_navigation_instruction(target: dict[str, Any], language: str = "en") -> str:
    object_name = target.get("object")
    label = display_object_en(object_name if object_name is not None else target.get("location_hint"))
    if language.lower().startswith("ko"):
        return f"Move toward the {display_object_ko(object_name)}."
    return f"Go to the {label}."


def build_inspection_question(target: dict[str, Any], query: dict[str, Any]) -> str:
    object_label = display_object_en(target.get("object"))
    expected_value = str(query.get("expected_value") or "").strip().lower()
    attribute = str(query.get("attribute") or "").strip().lower()
    if attribute == "power_state":
        if expected_value == "off":
            return f"Is the {object_label} off?"
        if expected_value == "on":
            return f"Is the {object_label} on?"
        return f"What is the power state of the {object_label}?"
    if attribute == "open_state":
        if expected_value == "closed":
            return f"Is the {object_label} closed?"
        if expected_value == "open":
            return f"Is the {object_label} open?"
        return f"What is the open state of the {object_label}?"
    return f"Is this the {object_label}?"


def observed_value_from_answer(query: dict[str, Any], answer_is_true: bool) -> str:
    expected_value = str(query.get("expected_value") or "unknown").strip().lower() or "unknown"
    if expected_value == "unknown":
        return "unknown"
    if answer_is_true:
        return expected_value
    return OPPOSITE_EXPECTED_VALUES.get(expected_value, "unknown")


def render_report_message(task_frame: dict[str, Any], subgoals: list[dict[str, Any]]) -> str:
    intent = task_frame["intent"]
    target = task_frame["target"]
    clarification = task_frame["clarification"]
    label = display_object_en(target.get("object"))
    returned = any(subgoal["type"] == "return" and subgoal["status"] == "succeeded" for subgoal in subgoals)
    blocked_subgoal = next(
        (
            subgoal
            for subgoal in subgoals
            if subgoal["status"] == "failed"
            and isinstance(subgoal.get("output"), dict)
            and subgoal["output"].get("error") == "knowledge_guard_denied"
        ),
        None,
    )

    if blocked_subgoal is not None:
        return str(blocked_subgoal["output"].get("message") or "Execution was blocked by knowledge rules.")

    if intent == "ask_clarification":
        return str(clarification.get("question_ko") or "Please clarify the request.")

    if intent == "unsupported":
        return str(clarification.get("question_ko") or f"The request for {label} is not supported.")

    if intent in {"find_object", "navigate_to_object"}:
        if returned:
            return f"Reached the {label} and returned to the start pose."
        return f"Reached the {label}."

    inspect_subgoal = next((item for item in subgoals if item["type"] == "inspect"), None)
    observed_value = None if inspect_subgoal is None else inspect_subgoal["output"].get("observed_value")
    if observed_value == "off":
        base = f"Confirmed that the {label} is off."
    elif observed_value == "on":
        base = f"Confirmed that the {label} is on."
    elif observed_value == "open":
        base = f"Confirmed that the {label} is open."
    elif observed_value == "closed":
        base = f"Confirmed that the {label} is closed."
    else:
        base = f"Could not confidently determine the state of the {label}."
    if returned:
        return f"{base} Returned to the start pose."
    return base
