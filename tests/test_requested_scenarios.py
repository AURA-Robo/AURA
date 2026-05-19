from __future__ import annotations

import pytest

from systems.reasoning.planner_catalog_models import SUPPORTED_SUBGOAL_TYPES
from systems.reasoning.planner_catalog_repository import InMemoryPlannerCatalogRepository
from systems.reasoning.planner_catalog_service import PlannerCatalogService
from systems.reasoning.service import ReasoningCoordinator, build_arg_parser


class _NavigationStub:
    def cancel(self) -> dict[str, object]:
        return {"ok": True}

    def command(self, instruction: str, language: str = "en", *, task_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "instruction": instruction, "language": language, "task_id": task_id}

    def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "target": dict(target), "task_id": task_id}

    def command_return_pose(self, origin_pose: dict[str, object], *, task_id: str | None = None) -> dict[str, object]:
        return {"ok": True, "origin_pose": dict(origin_pose), "task_id": task_id}

    def status(self) -> dict[str, object]:
        return {
            "ok": True,
            "status": "idle",
            "task_id": None,
            "goal_world_xy": None,
            "path_points": 0,
            "action_override_mode": None,
            "last_error": None,
            "current_robot_pose": {"world_xy": [0.0, 0.0], "yaw_rad": 0.0},
            "system2": {"status": "idle", "decision_mode": "idle"},
        }


def _coordinator() -> ReasoningCoordinator:
    args = build_arg_parser().parse_args(
        [
            "--planner-model-base-url",
            "",
            "--dialogue-model-base-url",
            "",
        ]
    )
    coordinator = ReasoningCoordinator(args)
    coordinator._task_coordinator._navigation = _NavigationStub()  # type: ignore[attr-defined]
    return coordinator


def test_requested_warehouse_purple_cart_scenario_currently_needs_clarification() -> None:
    response = _coordinator().respond(
        {
            "utterance": "보라색 박스가 담긴 카트가 있는지 확인한 뒤 복귀해라.",
            "language": "ko",
            "conversation_id": "requested-scenario-warehouse",
            "scene_preset": "warehouse",
        }
    )

    assert response["ok"] is True
    assert response["route"] == "clarification"
    assert response["task"] is None
    assert response["reply_text"] == "어떤 대상을 확인해야 하는지 알려주세요."


def test_requested_interior_agent_tv_off_scenario_currently_unsupported() -> None:
    response = _coordinator().respond(
        {
            "utterance": "거실에 tv가 꺼져있는지 보고올래?",
            "language": "ko",
            "conversation_id": "requested-scenario-interioragent",
            "scene_preset": "interioragent",
        }
    )

    assert response["ok"] is True
    assert response["route"] == "unsupported"
    assert response["task"] is None
    assert response["reply_text"] == "현재는 tv 관련 요청을 지원하지 않습니다."


def test_requested_warehouse2_logistics_box_count_scenario_currently_not_buildable() -> None:
    service = PlannerCatalogService(InMemoryPlannerCatalogRepository())
    supported_intents = {spec.intent_key for spec in service.supported_intent_specs()}

    assert supported_intents == {"check_state", "find_object", "navigate_to_object"}
    assert "count" not in SUPPORTED_SUBGOAL_TYPES
    with pytest.raises(ValueError, match="unsupported planner catalog intent"):
        service.create_intent("count_objects")

    response = _coordinator().respond(
        {
            "utterance": "지정된 구역에 가서 물류 박스가 몇개인지 세고 돌아와서 보고해라",
            "language": "ko",
            "conversation_id": "requested-scenario-warehouse2",
            "scene_preset": "warehouse2",
        }
    )

    assert response["ok"] is True
    assert response["route"] == "dialogue"
    assert response["task"] is None
    assert response["error"] == "dialogue_model_unavailable"
