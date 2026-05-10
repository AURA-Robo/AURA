from __future__ import annotations

from datetime import timedelta
from types import SimpleNamespace
import unittest

from systems.memory.knowledge_models import (
    KnowledgeContext,
    KnowledgeDocumentInput,
    KnowledgeLexiconEntry,
    KnowledgeRule,
    utc_now,
)
from systems.memory.agent_memory_models import AgentMemoryBlock, AgentMemoryContext, AgentMemoryMetadata
from systems.memory.knowledge_repository import InMemoryKnowledgeRepository
from systems.memory.knowledge_runtime import KnowledgeRuntimeHandle
from systems.memory.knowledge_service import KnowledgeService
from systems.memory.object_memory_runtime import ObjectMemoryRuntimeHandle
from systems.memory.object_memory_repository import InMemoryObjectMemoryRepository
from systems.memory.object_memory_service import ObjectMemoryService
from systems.memory.object_memory_models import ObjectObservationInput
from systems.reasoning.api.runtime import AuraTaskingAdapter
from systems.reasoning.planner_catalog_repository import InMemoryPlannerCatalogRepository
from systems.reasoning.planner_catalog_runtime import PlannerCatalogRuntimeHandle
from systems.reasoning.planner_catalog_service import PlannerCatalogService


class _FakeRuntimeController:
    def __init__(self, snapshot: dict[str, object]) -> None:
        self.snapshot = dict(snapshot)
        self.navigation_calls: list[dict[str, object]] = []
        self.report_messages: list[str] = []

    def start_navigation_instruction(self, instruction: str, language: str = "en") -> dict[str, object]:
        self.navigation_calls.append({"instruction": instruction, "language": language})
        return {
            "instruction": instruction,
            "language": language,
            "command_revision": len(self.navigation_calls),
            "session_id": "test-session",
            "session_reset_required": True,
        }

    def start_navigation_memory_goal(self, navigation_target: dict[str, object]) -> dict[str, object]:
        self.navigation_calls.append({"mode": "memory_pose", "target": dict(navigation_target)})
        return {
            "command_revision": len(self.navigation_calls),
            "session_id": "test-session",
            "session_reset_required": False,
            "goal_world_xy": list(navigation_target["world_pose_xyz"][:2]),
        }

    def navigation_snapshot(self, *, origin_pose: dict[str, object] | None = None) -> dict[str, object]:
        del origin_pose
        return dict(self.snapshot)

    def check_binary_question(self, question: str) -> str:
        del question
        return "true"

    def set_last_report(self, message: str) -> None:
        self.report_messages.append(str(message))


class PlannerTaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = AuraTaskingAdapter(
            completion=None,
            model="test-model",
            timeout=1.0,
        )

    def _planner_catalog_runtime(self) -> PlannerCatalogRuntimeHandle:
        repository = InMemoryPlannerCatalogRepository()
        service = PlannerCatalogService(repository)
        seeded_snapshot = service.ensure_seed_data()
        return PlannerCatalogRuntimeHandle(
            enabled=True,
            service=service,
            _last_snapshot=seeded_snapshot,
            _last_refresh_ok=True,
        )

    def _knowledge_context(
        self,
        *,
        hard_rules: list[KnowledgeRule] | None = None,
        lexicon_entries: list[KnowledgeLexiconEntry] | None = None,
    ) -> KnowledgeContext:
        return KnowledgeContext(
            hard_rules=list(hard_rules or []),
            soft_rules=[],
            lexicon_entries=list(lexicon_entries or []),
            facts=[],
            debug={},
        )

    def test_check_state_command_expands_to_navigate_inspect_return_report(self) -> None:
        task_frame = self.adapter.plan_task_frame("check whether the tv is off and come back")
        self.assertEqual(task_frame["intent"], "check_state")
        self.assertEqual(task_frame["target"]["object"], "tv")
        self.assertEqual(task_frame["query"]["attribute"], "power_state")
        self.assertEqual(task_frame["query"]["expected_value"], "off")
        self.assertTrue(task_frame["constraints"]["return_after_check"])

        subgoals = self.adapter.initialize_subgoals(task_frame)
        self.assertEqual([item["type"] for item in subgoals], ["navigate", "inspect", "return", "report"])

    def test_navigate_and_return_command_expands_to_navigate_return_report(self) -> None:
        task_frame = self.adapter.plan_task_frame("go to the purple box cart and come back")
        self.assertEqual(task_frame["intent"], "navigate_to_object")
        self.assertEqual(task_frame["target"]["object"], "purple_box_cart")
        self.assertTrue(task_frame["constraints"]["return_after_check"])

        subgoals = self.adapter.initialize_subgoals(task_frame)
        self.assertEqual([item["type"] for item in subgoals], ["navigate", "return", "report"])

    def test_zero_shot_korean_object_expands_to_find_return_report(self) -> None:
        task_frame = self.adapter.plan_task_frame("보라색 상자를 찾아서 돌아와")
        self.assertEqual(task_frame["intent"], "find_object")
        self.assertEqual(task_frame["target"]["object"], "보라색_상자")
        self.assertTrue(task_frame["constraints"]["return_after_check"])

        subgoals = self.adapter.initialize_subgoals(task_frame)
        self.assertEqual([item["type"] for item in subgoals], ["navigate", "return", "report"])

    def test_default_catalog_runtime_preserves_current_subgoal_expansion(self) -> None:
        adapter = AuraTaskingAdapter(
            completion=None,
            model="test-model",
            timeout=1.0,
            planner_catalog_runtime=self._planner_catalog_runtime(),
        )

        task_frame = adapter.plan_task_frame("check whether the tv is off")
        subgoals = adapter.initialize_subgoals(task_frame)

        self.assertEqual(task_frame["intent"], "check_state")
        self.assertEqual([item["type"] for item in subgoals], ["navigate", "inspect", "report"])

    def test_deleted_catalog_intent_downgrades_request_to_unsupported(self) -> None:
        runtime = self._planner_catalog_runtime()
        snapshot = runtime.snapshot()
        target_intent = snapshot.intent_by_key("navigate_to_object")
        assert target_intent is not None
        runtime.delete_intent(target_intent.intent_id)
        adapter = AuraTaskingAdapter(
            completion=None,
            model="test-model",
            timeout=1.0,
            planner_catalog_runtime=runtime,
        )

        task_frame = adapter.plan_task_frame("go to the chair")

        self.assertEqual(task_frame["intent"], "unsupported")

    def test_subgoal_template_change_applies_on_next_planning_request_without_restart(self) -> None:
        runtime = self._planner_catalog_runtime()
        adapter = AuraTaskingAdapter(
            completion=None,
            model="test-model",
            timeout=1.0,
            planner_catalog_runtime=runtime,
        )

        before_frame = adapter.plan_task_frame("check whether the tv is off")
        before_subgoals = adapter.initialize_subgoals(before_frame)
        self.assertEqual([item["type"] for item in before_subgoals], ["navigate", "inspect", "report"])

        snapshot = runtime.snapshot()
        check_state_intent = snapshot.intent_by_key("check_state")
        assert check_state_intent is not None
        report_template = next(template for template in check_state_intent.subgoals if template.subgoal_type == "report")
        runtime.delete_subgoal_template(report_template.template_id)

        after_frame = adapter.plan_task_frame("check whether the tv is off")
        after_subgoals = adapter.initialize_subgoals(after_frame)

        self.assertEqual([item["type"] for item in after_subgoals], ["navigate", "inspect"])

    def test_roomless_recent_seen_does_not_create_location_hint(self) -> None:
        task_frame = self.adapter.plan_task_frame(
            "go to the chair",
            planning_context={
                "recent_seen": [
                    {
                        "class": "chair",
                        "room": None,
                        "age_sec": 3.0,
                    }
                ]
            },
        )

        self.assertEqual(task_frame["intent"], "navigate_to_object")
        self.assertEqual(task_frame["target"]["object"], "chair")
        self.assertIsNone(task_frame["target"]["location_hint"])

    def test_agent_memory_context_is_preserved_in_plan_request(self) -> None:
        captured: dict[str, object] = {}

        def _completion(messages, model, timeout, temperature, max_tokens):  # noqa: ANN001
            del model, timeout, temperature, max_tokens
            captured["messages"] = messages
            return """
            {
              "intent": "navigate_to_object",
              "target": {"object": "chair", "instance_hint": null, "location_hint": null},
              "query": {"query_type": null, "attribute": null, "operator": null, "expected_value": null},
              "constraints": {"return_after_check": false, "report_result": true},
              "clarification": {"required": false, "question_ko": null}
            }
            """

        adapter = AuraTaskingAdapter(
            completion=_completion,
            model="test-model",
            timeout=1.0,
        )
        now = utc_now()
        agent_memory = AgentMemoryContext(
            core_blocks=[
                AgentMemoryBlock(
                    label="operator_profile",
                    description="Operator preferences",
                    value="Prefers concise confirmations.",
                    limit=1024,
                    read_only=False,
                    scope="global",
                    version=1,
                    updated_at=now,
                )
            ],
            archival_passages=[],
            conversation_summary="The operator asked about the chair earlier.",
            recent_turns=[],
            object_memory=[],
            knowledge_facts=[],
            metadata=AgentMemoryMetadata(enabled=True, available=True),
        )

        task_frame = adapter.plan_task_frame(
            "go to the chair",
            planning_context={"agent_memory": agent_memory},
        )

        request = captured["messages"][1]["content"]  # type: ignore[index]
        self.assertEqual(task_frame["intent"], "navigate_to_object")
        self.assertIn("agent_memory", request)
        self.assertIn("Prefers concise confirmations.", request)

    def test_knowledge_lexicon_alias_is_applied_to_deterministic_planner(self) -> None:
        now = utc_now()
        task_frame = self.adapter.plan_task_frame(
            "go to the fridge",
            planning_context={
                "knowledge_context": self._knowledge_context(
                    lexicon_entries=[
                        KnowledgeLexiconEntry(
                            entry_id="lex-1",
                            document_id="doc-1",
                            mapping_type="object",
                            alias="fridge",
                            canonical="refrigerator",
                            scope_kind="global",
                            scope_value=None,
                            source_anchor="kitchen",
                            created_at=now,
                            updated_at=now,
                        )
                    ]
                )
            },
        )

        self.assertEqual(task_frame["intent"], "navigate_to_object")
        self.assertEqual(task_frame["target"]["object"], "refrigerator")

    def test_hard_rule_downgrades_navigation_request_to_unsupported(self) -> None:
        now = utc_now()
        task_frame = self.adapter.plan_task_frame(
            "go to the refrigerator",
            planning_context={
                "knowledge_context": self._knowledge_context(
                    hard_rules=[
                        KnowledgeRule(
                            rule_id="rule-1",
                            document_id="doc-1",
                            rule_key="deny-refrigerator-nav",
                            scope_kind="global",
                            scope_value=None,
                            enforcement="hard",
                            action="deny_task",
                            conditions={"intent": ["navigate_to_object"], "target_object": ["refrigerator"]},
                            params={},
                            priority=10,
                            reason="Do not navigate directly to the refrigerator.",
                            source_anchor="warehouse",
                            created_at=now,
                            updated_at=now,
                            published_at=now,
                        )
                    ]
                )
            },
        )

        self.assertEqual(task_frame["intent"], "unsupported")
        self.assertEqual(
            task_frame["clarification"]["question_ko"],
            "Do not navigate directly to the refrigerator.",
        )

    def test_navigation_subgoal_starts_navigation_pipeline(self) -> None:
        task_frame = self.adapter.plan_task_frame("go to the purple box cart and come back")
        subgoals = self.adapter.initialize_subgoals(task_frame)
        controller = _FakeRuntimeController(
            {
                "planner_target_mode": "none",
                "has_goal": False,
                "goal_world_xy": None,
                "goal_pixel_xy": None,
                "system2_status": "move",
                "system2_decision_mode": "move",
                "system2_text": "keep moving",
                "action_override_mode": None,
                "locomotion_command": [0.2, 0.0, 0.0],
                "state_label": "tracking",
                "goal_reached": False,
                "return_pose_distance": None,
                "return_pose_reached": False,
            }
        )
        task_state = SimpleNamespace(task_frame=task_frame, subgoals=subgoals, origin_pose=None)

        event = self.adapter.step(subgoals, {"controller": controller, "task_state": task_state})

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "navigate")
        self.assertEqual(event["status"], "running")
        self.assertEqual(subgoals[0]["status"], "running")
        self.assertEqual(
            controller.navigation_calls,
            [{"instruction": "Go to the purple box cart.", "language": "en"}],
        )

    def test_memory_resolution_injects_memory_pose_navigation_target(self) -> None:
        repository = InMemoryObjectMemoryRepository()
        service = ObjectMemoryService(repository)
        now = utc_now()
        service.observe_objects(
            "tester",
            "session-1",
            [
                ObjectObservationInput(
                    frame_idx=1,
                    track_id="track-chair-1",
                    class_name="chair",
                    detector_conf=0.9,
                    bbox_xyxy_norm=(0.1, 0.1, 0.4, 0.5),
                    box_area=0.0,
                    aspect_ratio=0.0,
                    image_hash="image-1",
                    observed_at=now,
                    room_id=None,
                    scene_scope="warehouse",
                    world_pose_xyz=(1.0, 2.0, 0.0),
                    world_pose_observed_at=now,
                    source_id="camera-0",
                    attributes={},
                )
            ],
        )
        memory_runtime = ObjectMemoryRuntimeHandle(enabled=True, user_id="tester", service=service)
        task_frame = self.adapter.plan_task_frame("go to the chair")
        subgoals = self.adapter.initialize_subgoals(task_frame)

        task_frame, subgoals, resolution = self.adapter.resolve_memory_navigation(
            task_frame,
            subgoals,
            object_memory_runtime=memory_runtime,
            scene_scope="warehouse",
        )

        self.assertEqual(task_frame["intent"], "navigate_to_object")
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["status"], "resolved")
        self.assertEqual(subgoals[0]["input"]["navigation_target"]["mode"], "memory_pose")
        self.assertEqual(subgoals[0]["input"]["navigation_target"]["object_id"], resolution["selected"]["object_id"])

    def test_bus_navigation_can_resolve_from_memory(self) -> None:
        repository = InMemoryObjectMemoryRepository()
        service = ObjectMemoryService(repository)
        now = utc_now()
        service.observe_objects(
            "tester",
            "session-1",
            [
                ObjectObservationInput(
                    frame_idx=1,
                    track_id="track-bus-1",
                    class_name="bus",
                    detector_conf=0.9,
                    bbox_xyxy_norm=(0.1, 0.1, 0.9, 0.8),
                    box_area=0.0,
                    aspect_ratio=0.0,
                    image_hash="image-bus",
                    observed_at=now,
                    room_id=None,
                    scene_scope="street",
                    world_pose_xyz=(1.0, 2.0, 3.0),
                    world_pose_observed_at=now,
                    source_id="camera-0",
                    attributes={},
                )
            ],
        )
        memory_runtime = ObjectMemoryRuntimeHandle(enabled=True, user_id="tester", service=service)
        task_frame = self.adapter.plan_task_frame("go to the bus")
        subgoals = self.adapter.initialize_subgoals(task_frame)

        task_frame, subgoals, resolution = self.adapter.resolve_memory_navigation(
            task_frame,
            subgoals,
            object_memory_runtime=memory_runtime,
            scene_scope="street",
        )

        self.assertEqual(task_frame["intent"], "navigate_to_object")
        self.assertEqual(task_frame["target"]["object"], "bus")
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["status"], "resolved")
        self.assertEqual(subgoals[0]["input"]["navigation_target"]["mode"], "memory_pose")
        self.assertEqual(subgoals[0]["input"]["navigation_target"]["class_name"], "bus")

    def test_memory_resolution_ambiguity_downgrades_to_clarification_report(self) -> None:
        repository = InMemoryObjectMemoryRepository()
        service = ObjectMemoryService(repository)
        now = utc_now()
        service.observe_objects(
            "tester",
            "session-1",
            [
                ObjectObservationInput(
                    frame_idx=1,
                    track_id="track-chair-1",
                    class_name="chair",
                    detector_conf=0.9,
                    bbox_xyxy_norm=(0.1, 0.1, 0.4, 0.5),
                    box_area=0.0,
                    aspect_ratio=0.0,
                    image_hash="image-1",
                    observed_at=now - timedelta(seconds=1),
                    room_id=None,
                    scene_scope="warehouse",
                    world_pose_xyz=(1.0, 2.0, 0.0),
                    world_pose_observed_at=now - timedelta(seconds=1),
                    source_id="camera-0",
                    attributes={},
                ),
                ObjectObservationInput(
                    frame_idx=2,
                    track_id="track-chair-2",
                    class_name="chair",
                    detector_conf=0.9,
                    bbox_xyxy_norm=(0.5, 0.1, 0.8, 0.5),
                    box_area=0.0,
                    aspect_ratio=0.0,
                    image_hash="image-2",
                    observed_at=now,
                    room_id=None,
                    scene_scope="warehouse",
                    world_pose_xyz=(3.0, 4.0, 0.0),
                    world_pose_observed_at=now,
                    source_id="camera-1",
                    attributes={},
                ),
            ],
        )
        memory_runtime = ObjectMemoryRuntimeHandle(enabled=True, user_id="tester", service=service)
        task_frame = self.adapter.plan_task_frame("go to the chair")
        subgoals = self.adapter.initialize_subgoals(task_frame)

        task_frame, subgoals, resolution = self.adapter.resolve_memory_navigation(
            task_frame,
            subgoals,
            object_memory_runtime=memory_runtime,
            scene_scope="warehouse",
        )

        self.assertEqual(task_frame["intent"], "ask_clarification")
        self.assertEqual(task_frame["clarification"]["question_ko"], "어느 chair인지 알려주세요. 최근 본 후보가 여러 개 있습니다.")
        self.assertEqual([item["type"] for item in subgoals], ["report"])
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution["status"], "ambiguous")

    def test_stop_result_advances_from_navigation_to_next_subgoal(self) -> None:
        task_frame = self.adapter.plan_task_frame("check whether the tv is off and come back")
        subgoals = self.adapter.initialize_subgoals(task_frame)
        controller = _FakeRuntimeController(
            {
                "planner_target_mode": "none",
                "has_goal": False,
                "goal_world_xy": None,
                "goal_pixel_xy": None,
                "system2_status": "STOP",
                "system2_decision_mode": "stop",
                "system2_text": "STOP",
                "action_override_mode": None,
                "locomotion_command": [0.0, 0.0, 0.0],
                "state_label": "done",
                "goal_reached": False,
                "return_pose_distance": None,
                "return_pose_reached": False,
            }
        )
        task_state = SimpleNamespace(task_frame=task_frame, subgoals=subgoals, origin_pose=None)

        event = self.adapter.step(subgoals, {"controller": controller, "task_state": task_state})
        current_subgoal = self.adapter.orchestrator.state_machine.current_subgoal(subgoals)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "navigate")
        self.assertEqual(event["status"], "succeeded")
        self.assertEqual(subgoals[0]["status"], "succeeded")
        self.assertIsNotNone(current_subgoal)
        assert current_subgoal is not None
        self.assertEqual(current_subgoal["type"], "inspect")
        self.assertEqual(
            controller.navigation_calls,
            [{"instruction": "Go to the TV.", "language": "en"}],
        )

    def test_wait_result_does_not_finish_navigation_subgoal(self) -> None:
        task_frame = self.adapter.plan_task_frame("check whether the tv is off and come back")
        subgoals = self.adapter.initialize_subgoals(task_frame)
        controller = _FakeRuntimeController(
            {
                "planner_target_mode": "none",
                "has_goal": False,
                "goal_world_xy": None,
                "goal_pixel_xy": None,
                "system2_status": "hold",
                "system2_decision_mode": "wait",
                "system2_text": "WAIT",
                "action_override_mode": None,
                "locomotion_command": [0.0, 0.0, 0.0],
                "state_label": "waiting",
                "goal_reached": False,
                "return_pose_distance": None,
                "return_pose_reached": False,
            }
        )
        task_state = SimpleNamespace(task_frame=task_frame, subgoals=subgoals, origin_pose=None)

        event = self.adapter.step(subgoals, {"controller": controller, "task_state": task_state})
        current_subgoal = self.adapter.orchestrator.state_machine.current_subgoal(subgoals)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["type"], "navigate")
        self.assertEqual(event["status"], "running")
        self.assertEqual(subgoals[0]["status"], "running")
        self.assertIsNotNone(current_subgoal)
        assert current_subgoal is not None
        self.assertEqual(current_subgoal["type"], "navigate")
        self.assertEqual(
            controller.navigation_calls,
            [{"instruction": "Go to the TV.", "language": "en"}],
        )

    def test_runtime_guard_blocks_execution_when_rule_is_published_after_planning(self) -> None:
        repository = InMemoryKnowledgeRepository()
        service = KnowledgeService(repository)
        runtime = KnowledgeRuntimeHandle(enabled=True, service=service)
        adapter = AuraTaskingAdapter(
            completion=None,
            model="test-model",
            timeout=1.0,
            knowledge_runtime=runtime,
        )

        task_frame = adapter.plan_task_frame("go to the tv")
        self.assertEqual(task_frame["intent"], "navigate_to_object")
        subgoals = adapter.initialize_subgoals(task_frame)

        service.register_document(
            KnowledgeDocumentInput(
                title="No TV approach",
                body_markdown="""
```knowledge-rule
{
  "action": "deny_task",
  "enforcement": "hard",
  "conditions": {
    "intent": "navigate_to_object",
    "target_object": "tv"
  },
  "reason": "TV approach is blocked by published knowledge."
}
```
""",
                publish=True,
            )
        )

        controller = _FakeRuntimeController(
            {
                "planner_target_mode": "none",
                "has_goal": False,
                "goal_world_xy": None,
                "goal_pixel_xy": None,
                "system2_status": "move",
                "system2_decision_mode": "move",
                "system2_text": "keep moving",
                "action_override_mode": None,
                "locomotion_command": [0.2, 0.0, 0.0],
                "state_label": "tracking",
                "goal_reached": False,
                "return_pose_distance": None,
                "return_pose_reached": False,
            }
        )
        task_state = SimpleNamespace(task_id="task-1", task_frame=task_frame, subgoals=subgoals, origin_pose=None)

        event = adapter.step(subgoals, {"controller": controller, "task_state": task_state})

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["status"], "failed")
        self.assertEqual(subgoals[0]["status"], "failed")
        self.assertEqual(subgoals[1]["status"], "pending")
        self.assertEqual(controller.navigation_calls, [])


if __name__ == "__main__":
    unittest.main()
