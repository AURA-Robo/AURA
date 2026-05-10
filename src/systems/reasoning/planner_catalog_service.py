from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable
import uuid

from systems.reasoning.planner_catalog_errors import (
    PlannerCatalogConflictError,
    PlannerCatalogValidationError,
)
from systems.reasoning.planner_catalog_models import (
    EXECUTION_INTENT_KEYS,
    INTENT_LAYOUT_RULES,
    PLANNER_INTENT_SPECS,
    PlannerCatalogIntent,
    PlannerCatalogIntentSpec,
    PlannerCatalogSnapshot,
    PlannerIntentRecord,
    PlannerSubgoalTemplateRecord,
    build_catalog_snapshot,
    intent_order_key,
    intent_spec_for_key,
    normalize_activation_condition,
    normalize_intent_key,
    normalize_sequence_no,
    normalize_subgoal_type,
    utc_now,
)
from systems.reasoning.planner_catalog_repository import PlannerCatalogRepository


class PlannerCatalogService:
    def __init__(self, repository: PlannerCatalogRepository) -> None:
        self.repository = repository

    def ensure_seed_data(self) -> PlannerCatalogSnapshot:
        if self.repository.count_intents(include_deleted=True) > 0:
            return self.load_snapshot()
        for spec in PLANNER_INTENT_SPECS:
            intent, templates = self._records_from_spec(spec)
            self.repository.create_intent_with_templates(intent, templates)
        return self.load_snapshot()

    def load_snapshot(self) -> PlannerCatalogSnapshot:
        intents = self.repository.list_active_intents()
        templates = self.repository.list_active_subgoal_templates()
        snapshot = build_catalog_snapshot(intents, templates)
        for intent in snapshot.intents:
            self._validate_intent_layout(intent.intent_key, intent.subgoals)
        return snapshot

    def create_intent(self, intent_key: str) -> PlannerCatalogSnapshot:
        normalized_key = normalize_intent_key(intent_key)
        snapshot = self.load_snapshot()
        if snapshot.intent_by_key(normalized_key) is not None:
            raise PlannerCatalogConflictError(f"planner intent already active: {normalized_key}")
        intent, templates = self._records_from_spec(intent_spec_for_key(normalized_key))
        self.repository.create_intent_with_templates(intent, templates)
        return self.load_snapshot()

    def delete_intent(self, intent_id: str) -> PlannerCatalogSnapshot:
        snapshot = self.load_snapshot()
        if snapshot.intent_by_id(intent_id) is None:
            raise KeyError(f"planner intent not found: {intent_id}")
        if not self.repository.soft_delete_intent(str(intent_id).strip(), deleted_at=utc_now()):
            raise KeyError(f"planner intent not found: {intent_id}")
        return self.load_snapshot()

    def create_subgoal_template(
        self,
        *,
        intent_id: str,
        sequence_no: int,
        subgoal_type: str,
        activation_condition: str,
    ) -> PlannerCatalogSnapshot:
        normalized_intent_id = str(intent_id or "").strip()
        if not normalized_intent_id:
            raise PlannerCatalogValidationError("planner catalog intent_id is required")
        normalized_sequence = self._normalize_sequence(sequence_no)
        normalized_type = self._normalize_subgoal_type(subgoal_type)
        normalized_condition = self._normalize_activation_condition(activation_condition)

        snapshot = self.load_snapshot()
        intent = snapshot.intent_by_id(normalized_intent_id)
        if intent is None:
            raise KeyError(f"planner intent not found: {intent_id}")

        next_template = PlannerSubgoalTemplateRecord(
            template_id=str(uuid.uuid4()),
            intent_id=intent.intent_id,
            sequence_no=normalized_sequence,
            subgoal_type=normalized_type,
            activation_condition=normalized_condition,
            created_at=utc_now(),
            updated_at=utc_now(),
            deleted_at=None,
        )
        self._validate_intent_layout(intent.intent_key, (*intent.subgoals, next_template))
        self.repository.create_subgoal_template(next_template)
        return self.load_snapshot()

    def delete_subgoal_template(self, template_id: str) -> PlannerCatalogSnapshot:
        normalized_template_id = str(template_id or "").strip()
        if not normalized_template_id:
            raise PlannerCatalogValidationError("planner catalog template_id is required")
        snapshot = self.load_snapshot()
        located = snapshot.template_by_id(normalized_template_id)
        if located is None:
            raise KeyError(f"planner subgoal template not found: {template_id}")
        intent, template = located
        remaining = tuple(item for item in intent.subgoals if item.template_id != template.template_id)
        self._validate_intent_layout(intent.intent_key, remaining)
        if not self.repository.soft_delete_subgoal_template(normalized_template_id, deleted_at=utc_now()):
            raise KeyError(f"planner subgoal template not found: {template_id}")
        return self.load_snapshot()

    @staticmethod
    def supported_intent_specs() -> tuple[PlannerCatalogIntentSpec, ...]:
        return tuple(sorted(PLANNER_INTENT_SPECS, key=lambda spec: intent_order_key(spec.intent_key)))

    @staticmethod
    def _normalize_sequence(sequence_no: int) -> int:
        try:
            return normalize_sequence_no(int(sequence_no))
        except (TypeError, ValueError) as exc:
            raise PlannerCatalogValidationError(str(exc)) from exc

    @staticmethod
    def _normalize_subgoal_type(subgoal_type: str) -> str:
        try:
            return normalize_subgoal_type(subgoal_type)
        except ValueError as exc:
            raise PlannerCatalogValidationError(str(exc)) from exc

    @staticmethod
    def _normalize_activation_condition(value: str) -> str:
        try:
            return normalize_activation_condition(value)
        except ValueError as exc:
            raise PlannerCatalogValidationError(str(exc)) from exc

    def _records_from_spec(
        self,
        spec: PlannerCatalogIntentSpec,
    ) -> tuple[PlannerIntentRecord, tuple[PlannerSubgoalTemplateRecord, ...]]:
        timestamp = utc_now()
        intent_id = str(uuid.uuid4())
        intent = PlannerIntentRecord(
            intent_id=intent_id,
            intent_key=spec.intent_key,
            display_name=spec.display_name,
            description=spec.description,
            created_at=timestamp,
            updated_at=timestamp,
            deleted_at=None,
        )
        templates = tuple(
            PlannerSubgoalTemplateRecord(
                template_id=str(uuid.uuid4()),
                intent_id=intent_id,
                sequence_no=sequence_no,
                subgoal_type=subgoal_type,
                activation_condition=activation_condition,
                created_at=timestamp,
                updated_at=timestamp,
                deleted_at=None,
            )
            for sequence_no, subgoal_type, activation_condition in spec.default_templates
        )
        return intent, templates

    def _validate_intent_layout(
        self,
        intent_key: str,
        templates: Iterable[PlannerSubgoalTemplateRecord],
    ) -> None:
        normalized_intent_key = normalize_intent_key(intent_key)
        if normalized_intent_key not in EXECUTION_INTENT_KEYS:
            raise PlannerCatalogValidationError(f"unsupported execution intent layout: {intent_key}")

        layout = INTENT_LAYOUT_RULES[normalized_intent_key]
        active_templates = tuple(sorted(templates, key=lambda item: (item.sequence_no, item.subgoal_type, item.template_id)))
        counts = Counter(template.subgoal_type for template in active_templates)

        for template in active_templates:
            if template.subgoal_type not in layout["allowed"]:
                raise PlannerCatalogConflictError(
                    f"{normalized_intent_key} does not allow subgoal_type={template.subgoal_type}"
                )
            expected_condition = layout["activation_conditions"].get(template.subgoal_type)
            if expected_condition is not None and template.activation_condition != expected_condition:
                raise PlannerCatalogConflictError(
                    f"{normalized_intent_key} requires {template.subgoal_type} activation_condition={expected_condition}"
                )

        for subgoal_type, max_count in layout["max_counts"].items():
            if counts[subgoal_type] > max_count:
                raise PlannerCatalogConflictError(
                    f"{normalized_intent_key} allows at most {max_count} {subgoal_type} template(s)"
                )

        missing = [subgoal_type for subgoal_type in layout["required"] if counts[subgoal_type] < 1]
        if missing:
            joined = ", ".join(sorted(missing))
            raise PlannerCatalogConflictError(f"{normalized_intent_key} requires subgoal template(s): {joined}")

        order_index = {subgoal_type: index for index, subgoal_type in enumerate(layout["order"])}
        sequence_order = [order_index[template.subgoal_type] for template in active_templates]
        if sequence_order != sorted(sequence_order):
            raise PlannerCatalogConflictError(
                f"{normalized_intent_key} subgoal sequence must follow {', '.join(layout['order'])}"
            )
