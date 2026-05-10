from __future__ import annotations

INTENTS = (
    "inspect_attribute",
    "find_object",
    "navigate_to_object",
    "ask_clarification",
    "unsupported",
)

TASK_FRAME_INTENTS = (
    "check_state",
    "find_object",
    "navigate_to_object",
    "ask_clarification",
    "unsupported",
)

QUERY_TYPES = ("attribute_check",)
QUERY_OPERATORS = ("equals",)

SUBGOAL_TYPES = (
    "navigate",
    "inspect",
    "return",
    "report",
)

SUBGOAL_STATUSES = (
    "pending",
    "running",
    "succeeded",
    "failed",
)

PLAN_TEMPLATES = (
    "inspect_object_and_return",
    "find_object_and_wait",
    "navigate_to_object",
    "return_origin",
)

REPAIR_TEMPLATES = (
    "search_next_room",
    "retry_approach",
    "retry_verify_with_reposition",
    "ask_clarification",
    "return_and_report_failure",
)

FAILURE_TYPES = (
    "object_not_found",
    "approach_failed",
    "target_lost",
    "verify_failed",
    "verification_failed",
    "inspection_unknown",
    "ambiguous_target",
    "ambiguous_room",
)

OBJECT_CLASSES = (
    "tv",
    "sofa",
    "bed",
    "chair",
    "refrigerator",
    "door",
    "purple_box_cart",
    "bus",
)

ATTRIBUTES = {
    "tv": ("power_state",),
    "door": ("open_state",),
    "sofa": (),
    "bed": (),
    "chair": (),
    "refrigerator": (),
    "purple_box_cart": (),
    "bus": (),
}

ROOM_CLASSES = (
    "living_room",
    "bedroom",
    "kitchen",
)

OBJECT_DISPLAY_KO = {
    "tv": "TV",
    "sofa": "sofa",
    "bed": "bed",
    "chair": "chair",
    "refrigerator": "refrigerator",
    "door": "door",
    "purple_box_cart": "purple box cart",
    "bus": "bus",
}

OBJECT_DISPLAY_EN = {
    "tv": "TV",
    "sofa": "sofa",
    "bed": "bed",
    "chair": "chair",
    "refrigerator": "refrigerator",
    "door": "door",
    "purple_box_cart": "purple box cart",
    "bus": "bus",
}

ROOM_DISPLAY_KO = {
    "living_room": "living room",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
}
