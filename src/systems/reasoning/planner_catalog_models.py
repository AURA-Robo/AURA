from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


SUPPORTED_ACTIVATION_CONDITIONS = (
    "always",
    "when_return_after_check",
    "when_report_result",
)

EXECUTION_INTENT_KEYS = (
    "check_state",
    "find_object",
    "navigate_to_object",
)

CODE_OWNED_INTENT_KEYS = (
    "ask_clarification",
    "unsupported",
)

SUPPORTED_SUBGOAL_TYPES = (
    "navigate",
    "inspect",
    "return",
    "report",
)

INTENT_LAYOUT_RULES = {
    "check_state": {
        "allowed": frozenset({"navigate", "inspect", "return", "report"}),
        "required": frozenset({"navigate", "inspect"}),
        "max_counts": {"navigate": 1, "inspect": 1, "return": 1, "report": 1},
        "order": ("navigate", "inspect", "return", "report"),
        "activation_conditions": {
            "navigate": "always",
            "inspect": "always",
            "return": "when_return_after_check",
            "report": "when_report_result",
        },
    },
    "find_object": {
        "allowed": frozenset({"navigate", "return", "report"}),
        "required": frozenset({"navigate"}),
        "max_counts": {"navigate": 1, "return": 1, "report": 1},
        "order": ("navigate", "return", "report"),
        "activation_conditions": {
            "navigate": "always",
            "return": "when_return_after_check",
            "report": "when_report_result",
        },
    },
    "navigate_to_object": {
        "allowed": frozenset({"navigate", "return", "report"}),
        "required": frozenset({"navigate"}),
        "max_counts": {"navigate": 1, "return": 1, "report": 1},
        "order": ("navigate", "return", "report"),
        "activation_conditions": {
            "navigate": "always",
            "return": "when_return_after_check",
            "report": "when_report_result",
        },
    },
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class PlannerCatalogIntentSpec:
    intent_key: str
    display_name: str
    description: str
    default_templates: tuple[tuple[int, str, str], ...]


PLANNER_INTENT_SPECS: tuple[PlannerCatalogIntentSpec, ...] = (
    PlannerCatalogIntentSpec(
        intent_key="check_state",
        display_name="Check State",
        description="Navigate to the target, inspect an attribute, optionally return, and report the result.",
        default_templates=(
            (1, "navigate", "always"),
            (2, "inspect", "always"),
            (3, "return", "when_return_after_check"),
            (4, "report", "when_report_result"),
        ),
    ),
    PlannerCatalogIntentSpec(
        intent_key="find_object",
        display_name="Find Object",
        description="Navigate until the target is found, optionally return, and report the result.",
        default_templates=(
            (1, "navigate", "always"),
            (2, "return", "when_return_after_check"),
            (3, "report", "when_report_result"),
        ),
    ),
    PlannerCatalogIntentSpec(
        intent_key="navigate_to_object",
        display_name="Navigate To Object",
        description="Navigate to the target, optionally return, and report the result.",
        default_templates=(
            (1, "navigate", "always"),
            (2, "return", "when_return_after_check"),
            (3, "report", "when_report_result"),
        ),
    ),
)

_INTENT_SPEC_BY_KEY = {spec.intent_key: spec for spec in PLANNER_INTENT_SPECS}
_INTENT_ORDER = {spec.intent_key: index for index, spec in enumerate(PLANNER_INTENT_SPECS)}


def intent_order_key(intent_key: str) -> tuple[int, str]:
    return (_INTENT_ORDER.get(intent_key, len(_INTENT_ORDER)), str(intent_key))


def intent_spec_for_key(intent_key: str) -> PlannerCatalogIntentSpec:
    normalized = str(intent_key or "").strip()
    spec = _INTENT_SPEC_BY_KEY.get(normalized)
    if spec is None:
        raise ValueError(f"unsupported planner catalog intent: {intent_key}")
    return spec


def normalize_intent_key(intent_key: str) -> str:
    return intent_spec_for_key(intent_key).intent_key


def normalize_subgoal_type(subgoal_type: str) -> str:
    normalized = str(subgoal_type or "").strip().lower()
    if normalized not in SUPPORTED_SUBGOAL_TYPES:
        raise ValueError(f"unsupported planner catalog subgoal_type: {subgoal_type}")
    return normalized


def normalize_activation_condition(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in SUPPORTED_ACTIVATION_CONDITIONS:
        raise ValueError(f"unsupported planner catalog activation_condition: {value}")
    return normalized


def normalize_sequence_no(value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError("planner catalog sequence_no must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class PlannerIntentRecord:
    intent_id: str
    intent_key: str
    display_name: str
    description: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PlannerSubgoalTemplateRecord:
    template_id: str
    intent_id: str
    sequence_no: int
    subgoal_type: str
    activation_condition: str
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class PlannerCatalogIntent:
    intent_id: str
    intent_key: str
    display_name: str
    description: str
    created_at: datetime
    updated_at: datetime
    subgoals: tuple[PlannerSubgoalTemplateRecord, ...]


@dataclass(frozen=True, slots=True)
class PlannerCatalogSnapshot:
    intents: tuple[PlannerCatalogIntent, ...]
    source: str = "database"
    writable: bool = True
    degraded_reason: str | None = None

    @property
    def active_intent_keys(self) -> tuple[str, ...]:
        return tuple(intent.intent_key for intent in self.intents)

    @property
    def active_subgoal_template_count(self) -> int:
        return sum(len(intent.subgoals) for intent in self.intents)

    def intent_by_key(self, intent_key: str) -> PlannerCatalogIntent | None:
        normalized = str(intent_key or "").strip()
        for intent in self.intents:
            if intent.intent_key == normalized:
                return intent
        return None

    def intent_by_id(self, intent_id: str) -> PlannerCatalogIntent | None:
        normalized = str(intent_id or "").strip()
        for intent in self.intents:
            if intent.intent_id == normalized:
                return intent
        return None

    def template_by_id(self, template_id: str) -> tuple[PlannerCatalogIntent, PlannerSubgoalTemplateRecord] | None:
        normalized = str(template_id or "").strip()
        for intent in self.intents:
            for template in intent.subgoals:
                if template.template_id == normalized:
                    return intent, template
        return None


@dataclass(frozen=True, slots=True)
class PlannerCatalogStatusSnapshot:
    enabled: bool
    available: bool
    writable: bool
    source: str
    degraded_reason: str | None
    last_refresh_ok: bool | None
    active_intent_count: int
    active_subgoal_template_count: int


def build_catalog_snapshot(
    intents: tuple[PlannerIntentRecord, ...] | list[PlannerIntentRecord],
    subgoal_templates: tuple[PlannerSubgoalTemplateRecord, ...] | list[PlannerSubgoalTemplateRecord],
    *,
    source: str = "database",
    writable: bool = True,
    degraded_reason: str | None = None,
) -> PlannerCatalogSnapshot:
    templates_by_intent: dict[str, list[PlannerSubgoalTemplateRecord]] = {}
    for template in subgoal_templates:
        templates_by_intent.setdefault(template.intent_id, []).append(template)
    grouped: list[PlannerCatalogIntent] = []
    for intent in sorted(tuple(intents), key=lambda item: intent_order_key(item.intent_key)):
        templates = tuple(
            sorted(
                templates_by_intent.get(intent.intent_id, []),
                key=lambda item: (item.sequence_no, item.subgoal_type, item.template_id),
            )
        )
        grouped.append(
            PlannerCatalogIntent(
                intent_id=intent.intent_id,
                intent_key=intent.intent_key,
                display_name=intent.display_name,
                description=intent.description,
                created_at=intent.created_at,
                updated_at=intent.updated_at,
                subgoals=templates,
            )
        )
    return PlannerCatalogSnapshot(
        intents=tuple(grouped),
        source=source,
        writable=writable,
        degraded_reason=degraded_reason,
    )


def default_catalog_snapshot(*, source: str = "default", writable: bool = False, degraded_reason: str | None = None) -> PlannerCatalogSnapshot:
    timestamp = utc_now()
    intents: list[PlannerIntentRecord] = []
    templates: list[PlannerSubgoalTemplateRecord] = []
    for spec in PLANNER_INTENT_SPECS:
        intent_id = f"default-{spec.intent_key}"
        intents.append(
            PlannerIntentRecord(
                intent_id=intent_id,
                intent_key=spec.intent_key,
                display_name=spec.display_name,
                description=spec.description,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )
        for sequence_no, subgoal_type, activation_condition in spec.default_templates:
            templates.append(
                PlannerSubgoalTemplateRecord(
                    template_id=f"{intent_id}-{sequence_no}-{subgoal_type}",
                    intent_id=intent_id,
                    sequence_no=sequence_no,
                    subgoal_type=subgoal_type,
                    activation_condition=activation_condition,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            )
    return build_catalog_snapshot(
        intents,
        templates,
        source=source,
        writable=writable,
        degraded_reason=degraded_reason,
    )
