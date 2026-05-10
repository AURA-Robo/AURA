from __future__ import annotations

from typing import Any


DESIRED_CHECK_TO_EXPECTED_VALUE = {
    "is_off": "off",
    "is_on": "on",
    "is_closed": "closed",
    "is_open": "open",
}


def expected_value_from_desired_check(desired_check: str) -> str | None:
    return DESIRED_CHECK_TO_EXPECTED_VALUE.get(desired_check)


def task_frame_to_plan(task_frame: dict[str, Any]) -> dict[str, Any]:
    intent = task_frame["intent"]
    target = task_frame["target"]
    query = task_frame["query"]
    constraints = task_frame["constraints"]
    clarification = task_frame["clarification"]

    if intent == "check_state":
        return {
            "intent": "inspect_attribute",
            "plan_template": "inspect_object_and_return",
            "target": {
                "class": target.get("object"),
                "attribute": query.get("attribute"),
            },
            "hints": {
                "room": target.get("location_hint"),
                "instance": target.get("instance_hint"),
            },
            "constraints": {
                "return_home": constraints["return_after_check"],
                "report_result": constraints["report_result"],
            },
            "clarification": clarification,
        }

    if intent == "find_object":
        return {
            "intent": "find_object",
            "plan_template": "find_object_and_wait",
            "target": {"class": target.get("object"), "attribute": None},
            "hints": {
                "room": target.get("location_hint"),
                "instance": target.get("instance_hint"),
            },
            "constraints": {
                "return_home": constraints["return_after_check"],
                "report_result": constraints["report_result"],
            },
            "clarification": clarification,
        }

    if intent == "navigate_to_object":
        return {
            "intent": "navigate_to_object",
            "plan_template": "navigate_to_object",
            "target": {"class": target.get("object"), "attribute": None},
            "hints": {
                "room": target.get("location_hint"),
                "instance": target.get("instance_hint"),
            },
            "constraints": {
                "return_home": constraints["return_after_check"],
                "report_result": constraints["report_result"],
            },
            "clarification": clarification,
        }

    return {
        "intent": intent,
        "plan_template": "return_origin",
        "target": {
            "class": target.get("object"),
            "attribute": query.get("attribute"),
        },
        "hints": {
            "room": target.get("location_hint"),
            "instance": target.get("instance_hint"),
        },
        "constraints": {
            "return_home": False,
            "report_result": constraints["report_result"],
        },
        "clarification": clarification,
    }


def plan_to_task_frame(plan: dict[str, Any]) -> dict[str, Any]:
    intent = plan["intent"]
    target = plan["target"]
    hints = plan["hints"]
    constraints = plan["constraints"]
    clarification = plan["clarification"]

    if intent == "inspect_attribute":
        return {
            "intent": "check_state",
            "target": {
                "object": target.get("class"),
                "instance_hint": hints.get("instance"),
                "location_hint": hints.get("room"),
            },
            "query": {
                "query_type": "attribute_check",
                "attribute": target.get("attribute"),
                "operator": "equals",
                "expected_value": None,
            },
            "constraints": {
                "return_after_check": constraints["return_home"],
                "report_result": constraints["report_result"],
            },
            "clarification": clarification,
        }

    if intent in {"find_object", "navigate_to_object"}:
        return {
            "intent": intent,
            "target": {
                "object": target.get("class"),
                "instance_hint": hints.get("instance"),
                "location_hint": hints.get("room"),
            },
            "query": {
                "query_type": None,
                "attribute": None,
                "operator": None,
                "expected_value": None,
            },
            "constraints": {
                "return_after_check": constraints["return_home"],
                "report_result": constraints["report_result"],
            },
            "clarification": clarification,
        }

    return {
        "intent": intent,
        "target": {
            "object": target.get("class"),
            "instance_hint": hints.get("instance"),
            "location_hint": hints.get("room"),
        },
        "query": {
            "query_type": None,
            "attribute": target.get("attribute"),
            "operator": None,
            "expected_value": None,
        },
        "constraints": {
            "return_after_check": False,
            "report_result": constraints["report_result"],
        },
        "clarification": clarification,
    }
