from __future__ import annotations

import unittest

from systems.memory.knowledge_models import KnowledgeDocumentInput
from systems.memory.knowledge_repository import InMemoryKnowledgeRepository
from systems.memory.knowledge_runtime import KnowledgeRuntimeHandle
from systems.memory.knowledge_service import KnowledgeService
from systems.memory.object_memory_models import ObjectObservationInput, utc_now
from systems.memory.object_memory_repository import InMemoryObjectMemoryRepository
from systems.memory.object_memory_runtime import ObjectMemoryRuntimeHandle
from systems.memory.object_memory_service import ObjectMemoryService
from systems.control.api.runtime_args import build_arg_parser as build_control_arg_parser
from systems.control.api.runtime_controller import InternVlaNavDpController
from systems.reasoning.service import ReasoningSystem, build_arg_parser as build_reasoning_arg_parser


def _nav_status(
    *,
    task_id: str | None,
    status: str = "running",
    system2_status: str | None = None,
    decision_mode: str | None = None,
    text: str | None = None,
    goal_world_xy: list[float] | None = None,
    path_points: int = 0,
    action_override_mode: str | None = None,
    last_error: str | None = None,
    current_world_xy: tuple[float, float] | None = (0.0, 0.0),
) -> dict[str, object]:
    payload: dict[str, object] = {
        "ok": True,
        "status": status,
        "task_id": task_id,
        "path_points": path_points,
        "goal_world_xy": goal_world_xy,
        "action_override_mode": action_override_mode,
        "last_error": last_error,
        "system2": {
            "status": system2_status,
            "decision_mode": decision_mode,
            "text": text,
        },
    }
    if current_world_xy is not None:
        payload["current_robot_pose"] = {
            "world_xy": [float(current_world_xy[0]), float(current_world_xy[1])],
            "world_xyz": [float(current_world_xy[0]), float(current_world_xy[1]), 0.0],
            "yaw_rad": 0.0,
        }
    return payload


class _PlannerNavStub:
    def __init__(self, *, status_payload: dict[str, object] | None = None):
        self.cancel_calls = 0
        self.commands: list[dict[str, object]] = []
        self.memory_commands: list[dict[str, object]] = []
        self.return_commands: list[dict[str, object]] = []
        self._status_payload = dict(status_payload or _nav_status(task_id=None, status="idle"))

    def set_status(self, payload: dict[str, object]) -> None:
        self._status_payload = dict(payload)

    def cancel(self):
        self.cancel_calls += 1
        return {"ok": True}

    def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
        self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
        return _nav_status(task_id=task_id, status="running", goal_world_xy=[1.0, 2.0], path_points=1)

    def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None):
        self.memory_commands.append({"target": dict(target), "task_id": task_id})
        return _nav_status(
            task_id=task_id,
            status="running",
            goal_world_xy=list(target["world_pose_xyz"][:2]),
            path_points=1,
            current_world_xy=(0.0, 0.0),
        )

    def command_return_pose(self, origin_pose: dict[str, object], *, task_id: str | None = None):
        self.return_commands.append({"origin_pose": dict(origin_pose), "task_id": task_id})
        return _nav_status(
            task_id=task_id,
            status="running",
            system2_status="return_pose",
            decision_mode="return_pose",
            text="Returning to origin",
            goal_world_xy=list(origin_pose["world_xy"]),
            path_points=1,
            current_world_xy=(1.0, 1.0),
        )

    def status(self):
        return dict(self._status_payload)


class RuntimePlannerStatusTests(unittest.TestCase):
    def test_control_runtime_parser_supports_viewer_publish_toggle(self) -> None:
        parser = build_control_arg_parser()

        default_args = parser.parse_args([])
        disabled_args = parser.parse_args(["--no-viewer-publish"])
        enabled_args = parser.parse_args(["--viewer-publish"])

        self.assertTrue(default_args.viewer_publish)
        self.assertFalse(disabled_args.viewer_publish)
        self.assertTrue(enabled_args.viewer_publish)

    def test_control_runtime_parser_supports_detection_wiring(self) -> None:
        parser = build_control_arg_parser()

        default_args = parser.parse_args([])
        disabled_args = parser.parse_args(["--no-detection-enabled"])
        enabled_args = parser.parse_args(
            [
                "--detection-enabled",
                "--detection-model-path",
                r"C:\models\yoloe-26s-seg-pf.pt",
            ]
        )

        self.assertTrue(default_args.detection_enabled)
        self.assertIsInstance(default_args.detection_model_path, str)
        self.assertFalse(disabled_args.detection_enabled)
        self.assertTrue(enabled_args.detection_enabled)
        self.assertEqual(enabled_args.detection_model_path, r"C:\models\yoloe-26s-seg-pf.pt")

    def test_planner_system_owns_task_and_subgoal_status(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])
        nav = _PlannerNavStub()
        planner = ReasoningSystem(planner_args)
        planner._navigation = nav  # type: ignore[attr-defined]
        response = planner.submit_task("check whether the tv is off and come back", "en", task_id="planner-fixed")

        self.assertEqual(response["task_id"], "planner-fixed")
        self.assertEqual(response["task_frame"]["intent"], "check_state")
        self.assertEqual(response["task_status"], "running")
        self.assertEqual(response["current_subgoal"]["type"], "navigate")
        self.assertEqual([item["type"] for item in response["subgoals"]], ["navigate", "inspect", "return", "report"])
        self.assertEqual(nav.cancel_calls, 0)
        self.assertEqual(nav.commands[0]["instruction"], "Go to the TV.")
        self.assertEqual(nav.commands[0]["language"], "en")
        self.assertEqual(nav.commands[0]["task_id"], "planner-fixed")
        self.assertEqual(len(nav.commands), 1)

    def test_planner_system_advances_navigation_subgoal_when_system2_reports_stop(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])
        nav = _PlannerNavStub(
            status_payload=_nav_status(
                task_id="planner-fixed",
                system2_status="stop",
                decision_mode="stop",
                text="discrete_action:0",
            )
        )
        planner = ReasoningSystem(planner_args)
        planner._navigation = nav  # type: ignore[attr-defined]
        planner.submit_task("check whether the tv is off and come back", "en", task_id="planner-fixed")

        status = planner.status_payload()

        self.assertEqual(status["task_status"], "running")
        self.assertEqual(status["subgoals"][0]["status"], "succeeded")
        self.assertEqual(status["current_subgoal"]["type"], "inspect")
        self.assertEqual(status["current_subgoal"]["status"], "pending")

    def test_planner_system_keeps_navigation_subgoal_running_when_system2_only_waits(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])
        nav = _PlannerNavStub(
            status_payload=_nav_status(
                task_id="planner-wait",
                system2_status="hold",
                decision_mode="wait",
                text="discrete_action:5",
            )
        )
        planner = ReasoningSystem(planner_args)
        planner._navigation = nav  # type: ignore[attr-defined]
        planner.submit_task("check whether the tv is off and come back", "en", task_id="planner-wait")

        status = planner.status_payload()

        self.assertEqual(status["task_status"], "running")
        self.assertEqual(status["subgoals"][0]["status"], "running")
        self.assertEqual(status["current_subgoal"]["type"], "navigate")
        self.assertEqual(status["current_subgoal"]["status"], "running")

    def test_planner_system_ignores_transient_navigation_error_while_goal_is_active(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])

        class _NavStub:
            def __init__(self):
                self.cancel_calls = 0
                self.commands = []
                self.memory_commands = []

            def cancel(self):
                self.cancel_calls += 1
                return {"ok": True}

            def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
                self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
                return {"ok": True}

            def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None):
                self.memory_commands.append({"target": dict(target), "task_id": task_id})
                return {"ok": True}

            def command_return_pose(self, origin_pose: dict[str, object], *, task_id: str | None = None):
                return {"origin_pose": dict(origin_pose), "task_id": task_id}

            def status(self):
                return {
                    "ok": True,
                    "status": "error",
                    "task_id": "planner-transient-nav-error",
                    "path_points": 2,
                    "goal_world_xy": [1.0, 2.0],
                    "action_override_mode": None,
                    "last_error": "RuntimeError: temporary planning miss",
                    "system2": {
                        "status": "goal",
                        "decision_mode": "pixel_goal",
                        "text": "goal",
                    },
                }

        planner = ReasoningSystem(planner_args)
        planner._navigation = _NavStub()  # type: ignore[attr-defined]
        planner.submit_task("go to purple box", "en", task_id="planner-transient-nav-error")

        status = planner.status_payload()

        self.assertEqual(status["task_status"], "running")
        self.assertEqual(status["subgoals"][0]["status"], "running")
        self.assertEqual(status["current_subgoal"]["type"], "navigate")
        self.assertEqual(status["current_subgoal"]["status"], "running")

    def test_planner_system_creates_navigation_task_for_go_to_purple_box(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])

        class _NavStub:
            def __init__(self):
                self.cancel_calls = 0
                self.commands = []
                self.memory_commands = []

            def cancel(self):
                self.cancel_calls += 1
                return {"ok": True}

            def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
                self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
                return {"ok": True}

            def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None):
                self.memory_commands.append({"target": dict(target), "task_id": task_id})
                return {"ok": True}

            def command_return_pose(self, origin_pose: dict[str, object], *, task_id: str | None = None):
                return {"origin_pose": dict(origin_pose), "task_id": task_id}

        planner = ReasoningSystem(planner_args)
        planner._navigation = _NavStub()  # type: ignore[attr-defined]
        response = planner.submit_task("go to purple box", "en", task_id="planner-purple-box")

        self.assertEqual(response["task_id"], "planner-purple-box")
        self.assertEqual(response["task_frame"]["intent"], "navigate_to_object")
        self.assertEqual(response["task_frame"]["target"]["object"], "purple_box_cart")
        self.assertEqual(response["task_status"], "running")
        self.assertEqual(response["current_subgoal"]["type"], "navigate")
        self.assertEqual([item["type"] for item in response["subgoals"]], ["navigate", "report"])
        self.assertEqual(planner._navigation.cancel_calls, 0)  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["instruction"], "Go to the purple box cart.")  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["language"], "en")  # type: ignore[attr-defined]
        self.assertEqual(planner._navigation.commands[0]["task_id"], "planner-purple-box")  # type: ignore[attr-defined]
        self.assertEqual(len(planner._navigation.commands), 1)  # type: ignore[attr-defined]

    def test_planner_system_injects_recent_seen_context_into_plan_request(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])

        class _NavStub:
            def __init__(self):
                self.cancel_calls = 0
                self.commands = []
                self.memory_commands = []

            def cancel(self):
                self.cancel_calls += 1
                return {"ok": True}

            def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
                self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
                return {"ok": True}

            def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None):
                self.memory_commands.append({"target": dict(target), "task_id": task_id})
                return {"ok": True}

            def command_return_pose(self, origin_pose: dict[str, object], *, task_id: str | None = None):
                return {"origin_pose": dict(origin_pose), "task_id": task_id}

        planner = ReasoningSystem(planner_args)
        planner._navigation = _NavStub()  # type: ignore[attr-defined]
        repository = InMemoryObjectMemoryRepository()
        service = ObjectMemoryService(repository)
        service.observe_objects(
            "tester",
            "session-1",
            [
                ObjectObservationInput(
                    frame_idx=1,
                    track_id="track-1",
                    class_name="chair",
                    detector_conf=0.9,
                    bbox_xyxy_norm=(0.1, 0.1, 0.4, 0.5),
                    box_area=0.0,
                    aspect_ratio=0.0,
                    image_hash="image-1",
                    observed_at=utc_now(),
                    room_id="kitchen",
                    source_id="camera-0",
                    attributes={},
                )
            ],
        )
        planner._object_memory = ObjectMemoryRuntimeHandle(  # type: ignore[attr-defined]
            enabled=True,
            user_id="tester",
            service=service,
        )
        response = planner.submit_task("go to the chair", "en", task_id="planner-chair")

        self.assertEqual(response["task_frame"]["intent"], "navigate_to_object")
        self.assertEqual(response["task_frame"]["target"]["location_hint"], "kitchen")
        self.assertEqual(response["object_memory"]["enabled"], True)

    def test_planner_system_dispatches_memory_pose_navigation_when_pose_candidate_is_fresh(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])

        class _NavStub:
            def __init__(self):
                self.cancel_calls = 0
                self.commands = []
                self.memory_commands = []

            def cancel(self):
                self.cancel_calls += 1
                return {"ok": True}

            def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
                self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
                return {"ok": True}

            def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None):
                self.memory_commands.append({"target": dict(target), "task_id": task_id})
                return {"ok": True}

            def command_return_pose(self, origin_pose: dict[str, object], *, task_id: str | None = None):
                return {"origin_pose": dict(origin_pose), "task_id": task_id}

        planner = ReasoningSystem(planner_args)
        planner._navigation = _NavStub()  # type: ignore[attr-defined]
        repository = InMemoryObjectMemoryRepository()
        service = ObjectMemoryService(repository)
        observed_at = utc_now()
        service.observe_objects(
            "tester",
            "session-1",
            [
                ObjectObservationInput(
                    frame_idx=1,
                    track_id="track-chair-memory",
                    class_name="chair",
                    detector_conf=0.9,
                    bbox_xyxy_norm=(0.1, 0.1, 0.4, 0.5),
                    box_area=0.0,
                    aspect_ratio=0.0,
                    image_hash="image-memory",
                    observed_at=observed_at,
                    room_id=None,
                    scene_scope="warehouse",
                    world_pose_xyz=(1.0, 2.0, 0.0),
                    world_pose_observed_at=observed_at,
                    source_id="camera-0",
                    attributes={},
                )
            ],
        )
        planner._object_memory = ObjectMemoryRuntimeHandle(  # type: ignore[attr-defined]
            enabled=True,
            user_id="tester",
            service=service,
        )

        response = planner.submit_task("go to the chair", "en", task_id="planner-memory-chair", scene_preset="warehouse")

        self.assertTrue(response["memoryAwareTaskActive"])
        self.assertEqual(response["memoryNavigationMode"], "memory_pose")
        self.assertEqual(len(planner._navigation.commands), 0)  # type: ignore[attr-defined]
        self.assertEqual(len(planner._navigation.memory_commands), 1)  # type: ignore[attr-defined]
        self.assertEqual(
            planner._navigation.memory_commands[0]["target"]["mode"],  # type: ignore[attr-defined]
            "memory_pose",
        )

    def test_planner_system_includes_knowledge_status_and_scene_scoped_rules(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])

        class _NavStub:
            def cancel(self):
                return {"ok": True}

            def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
                return {"instruction": instruction, "language": language, "task_id": task_id}

            def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None):
                return {"target": dict(target), "task_id": task_id}

            def command_return_pose(self, origin_pose: dict[str, object], *, task_id: str | None = None):
                return {"origin_pose": dict(origin_pose), "task_id": task_id}

        planner = ReasoningSystem(planner_args)
        planner._navigation = _NavStub()  # type: ignore[attr-defined]
        repository = InMemoryKnowledgeRepository()
        service = KnowledgeService(repository)
        service.register_document(
            KnowledgeDocumentInput(
                title="Warehouse chair routing",
                scope_kind="scene",
                scope_value="warehouse",
                body_markdown="""
```knowledge-rule
{
  "action": "force_target_room",
  "enforcement": "hard",
  "conditions": {
    "target_object": "chair"
  },
  "room": "kitchen"
}
```
""",
                publish=True,
            )
        )
        planner._knowledge = KnowledgeRuntimeHandle(enabled=True, scene_scope="warehouse", service=service)  # type: ignore[attr-defined]
        planner._adapter.knowledge_runtime = planner._knowledge  # type: ignore[attr-defined]

        response = planner.submit_task("go to the chair", "en", task_id="planner-knowledge", scene_preset="warehouse")

        self.assertEqual(response["task_frame"]["target"]["location_hint"], "kitchen")
        self.assertTrue(response["knowledge"]["enabled"])
        self.assertEqual(response["knowledge"]["active_hard_rule_count"], 1)
        self.assertEqual(response["knowledge"]["published_document_count"], 1)

    def test_planner_system_dispatches_return_subgoal_and_completes_report(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])
        nav = _PlannerNavStub(status_payload=_nav_status(task_id=None, status="idle", current_world_xy=(0.0, 0.0)))

        planner = ReasoningSystem(planner_args)
        planner._navigation = nav  # type: ignore[attr-defined]
        response = planner.submit_task("go to the purple box cart and come back", "en", task_id="planner-return")

        self.assertEqual(response["current_subgoal"]["type"], "navigate")
        self.assertEqual(response["current_subgoal"]["status"], "running")
        self.assertEqual(len(nav.commands), 1)
        self.assertEqual(len(nav.return_commands), 0)

        nav.set_status(
            _nav_status(
                task_id="planner-return",
                system2_status="stop",
                decision_mode="stop",
                text="STOP",
                current_world_xy=(1.0, 1.0),
            )
        )
        status = planner.status_payload()

        self.assertEqual(status["task_status"], "running")
        self.assertEqual(status["current_subgoal"]["type"], "return")
        self.assertEqual(status["current_subgoal"]["status"], "running")
        self.assertEqual(len(nav.return_commands), 1)
        self.assertEqual(nav.return_commands[0]["origin_pose"]["world_xy"], [0.0, 0.0])

        nav.set_status(
            _nav_status(
                task_id="planner-return",
                system2_status="stop",
                decision_mode="stop",
                text="STOP",
                current_world_xy=(0.0, 0.0),
            )
        )
        completed = planner.status_payload()

        self.assertEqual(completed["task_status"], "completed")
        self.assertEqual([item["status"] for item in completed["subgoals"]], ["succeeded", "succeeded", "succeeded"])
        self.assertEqual(
            completed["subgoals"][2]["output"]["message"],
            "Reached the purple box cart and returned to the start pose.",
        )

        repeated = planner.status_payload()
        self.assertEqual(len(nav.return_commands), 1)
        self.assertEqual(repeated["subgoals"][2]["attempts"], 1)

    def test_planner_system_waits_for_origin_pose_before_dispatching_return_capable_task(self) -> None:
        planner_args = build_reasoning_arg_parser().parse_args(["--planner-model-base-url", ""])
        nav = _PlannerNavStub(status_payload=_nav_status(task_id=None, status="idle", current_world_xy=None))

        planner = ReasoningSystem(planner_args)
        planner._navigation = nav  # type: ignore[attr-defined]
        initial = planner.submit_task("go to the purple box cart and come back", "en", task_id="planner-origin-wait")

        self.assertEqual(initial["task_status"], "running")
        self.assertEqual(initial["current_subgoal"]["type"], "navigate")
        self.assertEqual(initial["current_subgoal"]["status"], "pending")
        self.assertEqual(len(nav.commands), 0)

        nav.set_status(_nav_status(task_id=None, status="idle", current_world_xy=(2.5, -1.0)))
        updated = planner.status_payload()

        self.assertEqual(updated["current_subgoal"]["type"], "navigate")
        self.assertEqual(updated["current_subgoal"]["status"], "running")
        self.assertEqual(len(nav.commands), 1)

    def test_control_runtime_status_is_navigation_only(self) -> None:
        args = build_control_arg_parser().parse_args([])
        controller = InternVlaNavDpController(args)
        try:
            status = controller.runtime_status()
        finally:
            controller.shutdown()

        self.assertEqual(status["executionMode"], "NAV")
        self.assertIn("routeState", status)
        self.assertIn("locomotion_command", status)
        self.assertNotIn("task_status", status)


if __name__ == "__main__":
    unittest.main()
