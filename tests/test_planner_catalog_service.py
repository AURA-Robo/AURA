from __future__ import annotations

import unittest

from systems.reasoning.planner_catalog_errors import PlannerCatalogConflictError, PlannerCatalogValidationError
from systems.reasoning.planner_catalog_repository import InMemoryPlannerCatalogRepository
from systems.reasoning.planner_catalog_runtime import PlannerCatalogRuntimeHandle
from systems.reasoning.planner_catalog_service import PlannerCatalogService


class PlannerCatalogServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repository = InMemoryPlannerCatalogRepository()
        self.service = PlannerCatalogService(self.repository)

    def test_seed_data_populates_default_catalog(self) -> None:
        snapshot = self.service.ensure_seed_data()

        self.assertEqual(snapshot.active_intent_keys, ("check_state", "find_object", "navigate_to_object"))
        self.assertEqual(snapshot.active_subgoal_template_count, 10)

    def test_duplicate_active_intent_is_rejected(self) -> None:
        self.service.ensure_seed_data()

        with self.assertRaises(PlannerCatalogConflictError):
            self.service.create_intent("check_state")

    def test_delete_intent_cascades_soft_delete_to_subgoals(self) -> None:
        snapshot = self.service.ensure_seed_data()
        target_intent = snapshot.intent_by_key("navigate_to_object")
        assert target_intent is not None

        next_snapshot = self.service.delete_intent(target_intent.intent_id)

        self.assertIsNone(next_snapshot.intent_by_key("navigate_to_object"))
        active_template_ids = {template.template_id for intent in next_snapshot.intents for template in intent.subgoals}
        for template in target_intent.subgoals:
            self.assertNotIn(template.template_id, active_template_ids)

    def test_invalid_activation_condition_is_rejected(self) -> None:
        snapshot = self.service.ensure_seed_data()
        target_intent = snapshot.intent_by_key("check_state")
        assert target_intent is not None

        with self.assertRaises(PlannerCatalogConflictError):
            self.service.create_subgoal_template(
                intent_id=target_intent.intent_id,
                sequence_no=5,
                subgoal_type="report",
                activation_condition="always",
            )

    def test_delete_required_subgoal_is_rejected(self) -> None:
        snapshot = self.service.ensure_seed_data()
        target_intent = snapshot.intent_by_key("check_state")
        assert target_intent is not None
        navigate_template = next(template for template in target_intent.subgoals if template.subgoal_type == "navigate")

        with self.assertRaises(PlannerCatalogConflictError):
            self.service.delete_subgoal_template(navigate_template.template_id)

    def test_runtime_falls_back_to_last_good_snapshot_on_refresh_failure(self) -> None:
        seeded_snapshot = self.service.ensure_seed_data()
        runtime = PlannerCatalogRuntimeHandle(
            enabled=True,
            service=self.service,
            _last_snapshot=seeded_snapshot,
            _last_refresh_ok=True,
        )

        def _raise() -> None:
            raise RuntimeError("database offline")

        self.service.load_snapshot = _raise  # type: ignore[method-assign]

        fallback = runtime.snapshot()
        status = runtime.status_snapshot(fallback)

        self.assertEqual(fallback.source, "last_good")
        self.assertFalse(status.available)
        self.assertFalse(status.writable)
        self.assertIn("database offline", str(status.degraded_reason))

    def test_create_subgoal_requires_positive_integer_sequence(self) -> None:
        snapshot = self.service.ensure_seed_data()
        target_intent = snapshot.intent_by_key("check_state")
        assert target_intent is not None

        with self.assertRaises(PlannerCatalogValidationError):
            self.service.create_subgoal_template(
                intent_id=target_intent.intent_id,
                sequence_no=0,
                subgoal_type="report",
                activation_condition="when_report_result",
            )


if __name__ == "__main__":
    unittest.main()
