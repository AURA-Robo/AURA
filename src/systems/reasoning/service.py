"""Standalone reasoning-system service."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import re
import threading
import time
from typing import Any
from urllib.parse import urlparse
import uuid

from systems.inference.api.planner import (
    make_http_completion,
    make_planner_intent_completion,
    make_planner_task_frame_completion,
)
from systems.memory.api import (
    AgentMemoryContext,
    ConversationMemoryRuntimeHandle,
    HumanoidMemoryRuntimeHandle,
    create_conversation_memory_runtime,
    create_humanoid_memory_runtime,
    create_knowledge_runtime,
    create_object_memory_runtime,
)
from systems.navigation.api.runtime import NavigationSystemClient
from systems.reasoning.dialogue import DialogueService
from systems.reasoning.interpreter import InputInterpreter
from systems.reasoning.planner.aura_adapter import AuraTaskingAdapter
from systems.reasoning.planner.reporting import build_navigation_instruction, render_report_message
from systems.reasoning.planner_catalog_runtime import create_planner_catalog_runtime
from systems.reasoning.policy import DialoguePolicy

STOP_REQUEST_PATTERN = re.compile(
    r"^\s*(?:stop|cancel|halt|freeze|멈춰|중지|취소)(?:\s+(?:now|please|task|current\s+task|it))?\s*$",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ActiveTaskState:
    task_id: str
    instruction: str
    language: str
    scene_preset: str | None
    task_frame: dict[str, Any]
    subgoals: list[dict[str, Any]]
    current_subgoal_index: int
    status: str
    started_at: float
    origin_pose: dict[str, Any] | None = None
    last_error: str | None = None
    memory_resolution: dict[str, Any] | None = None

    @property
    def current_subgoal(self) -> dict[str, Any] | None:
        if self.current_subgoal_index < 0 or self.current_subgoal_index >= len(self.subgoals):
            return None
        return self.subgoals[self.current_subgoal_index]


@dataclass(slots=True)
class PlanningResult:
    instruction: str
    interpreted_instruction: str
    scene_preset: str | None
    task_frame: dict[str, Any]
    subgoals: list[dict[str, Any]]
    memory_resolution: dict[str, Any] | None


class PlanningEngine:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._intent_completion = None
        self._task_frame_completion = None
        planner_model_base_url = str(args.planner_model_base_url).strip()
        if planner_model_base_url:
            self._intent_completion = make_planner_intent_completion(
                planner_model_base_url,
                slot_id=int(getattr(args, "planner_intent_slot_id", 0)),
            )
            self._task_frame_completion = make_planner_task_frame_completion(
                planner_model_base_url,
                slot_id=int(getattr(args, "planner_task_frame_slot_id", 1)),
            )
        self._knowledge = create_knowledge_runtime(
            dsn=str(getattr(args, "knowledge_dsn", "") or ""),
            object_memory_dsn=str(getattr(args, "object_memory_dsn", "") or ""),
            scene_scope=str(getattr(args, "knowledge_scene_scope", "") or "").strip() or None,
        )
        self._planner_catalog = create_planner_catalog_runtime(
            dsn=str(getattr(args, "planner_catalog_dsn", "") or ""),
            knowledge_dsn=str(getattr(args, "knowledge_dsn", "") or ""),
            object_memory_dsn=str(getattr(args, "object_memory_dsn", "") or ""),
        )
        self._adapter = AuraTaskingAdapter(
            completion=self._task_frame_completion,
            intent_completion=self._intent_completion,
            model=str(args.planner_model),
            timeout=float(args.planner_timeout),
            knowledge_runtime=self._knowledge,
            planner_catalog_runtime=self._planner_catalog,
            scene_scope=str(getattr(args, "knowledge_scene_scope", "") or "").strip() or None,
        )
        self._object_memory = create_object_memory_runtime(
            enabled=bool(str(getattr(args, "object_memory_dsn", "") or "").strip()),
            dsn=str(getattr(args, "object_memory_dsn", "") or ""),
            user_id=str(getattr(args, "memory_user_id", "") or ""),
            auto_migrate=bool(getattr(args, "object_memory_auto_migrate", False)),
        )

    @property
    def planner_service(self):
        return self._adapter.planner_service

    @property
    def model_available(self) -> bool:
        return self._task_frame_completion is not None

    @property
    def knowledge_runtime(self):
        return self._knowledge

    @property
    def object_memory_runtime(self):
        return self._object_memory

    @property
    def planner_catalog_runtime(self):
        return self._planner_catalog

    def prepare_task(
        self,
        instruction: str,
        *,
        scene_preset: str | None,
        agent_memory_context: AgentMemoryContext | None = None,
    ) -> PlanningResult:
        normalized_instruction = " ".join(str(instruction).strip().split())
        if not normalized_instruction:
            raise ValueError("instruction must be a non-empty string")
        normalized_scene_scope = " ".join(
            str(scene_preset or getattr(self.args, "knowledge_scene_scope", "") or "").strip().split()
        ) or None

        planning_context: dict[str, Any] = {}
        if self._object_memory.enabled:
            recent_context = self._object_memory.recent_context(
                top_k=10,
                scene_scope=normalized_scene_scope,
            )
            planning_context["recent_seen"] = recent_context.recent_seen
        if normalized_scene_scope is not None:
            planning_context["scene_preset"] = normalized_scene_scope
        if self._knowledge.enabled:
            planning_context["knowledge_context"] = self._knowledge.retrieve_for_plan(
                normalized_instruction,
                scene_scope=normalized_scene_scope,
            )
        if agent_memory_context is not None:
            planning_context["agent_memory"] = agent_memory_context

        task_frame = self._adapter.plan_task_frame(
            normalized_instruction,
            planning_context=planning_context,
        )
        knowledge_result = self._knowledge.evaluate_task_frame(
            task_frame,
            scene_scope=normalized_scene_scope,
            utterance=normalized_instruction,
        )
        if knowledge_result.mutated and isinstance(knowledge_result.task_frame, dict):
            task_frame = knowledge_result.task_frame
        subgoals = self._adapter.initialize_subgoals(task_frame)
        task_frame, subgoals, memory_resolution = self._adapter.resolve_memory_navigation(
            task_frame,
            subgoals,
            object_memory_runtime=self._object_memory,
            scene_scope=normalized_scene_scope,
            max_pose_age_sec=int(getattr(self.args, "memory_pose_max_age_sec", 600)),
            stop_radius_m=float(getattr(self.args, "memory_approach_radius_m", 0.8)),
            reacquire_radius_m=float(getattr(self.args, "memory_reacquire_radius_m", 1.0)),
            reacquire_timeout_sec=float(getattr(self.args, "memory_reacquire_timeout_sec", 4.0)),
        )
        return PlanningResult(
            instruction=normalized_instruction,
            interpreted_instruction=normalized_instruction,
            scene_preset=normalized_scene_scope,
            task_frame=task_frame,
            subgoals=subgoals,
            memory_resolution=memory_resolution,
        )


class TaskCoordinator:
    def __init__(
        self,
        args: argparse.Namespace,
        *,
        planning_engine: PlanningEngine,
        navigation_client: NavigationSystemClient | None = None,
    ) -> None:
        self.args = args
        self._planning_engine = planning_engine
        self._navigation = navigation_client or NavigationSystemClient(
            str(args.navigation_url),
            timeout_s=float(args.navigation_timeout),
        )
        self._lock = threading.Lock()
        self._task: ActiveTaskState | None = None

    @property
    def has_active_task(self) -> bool:
        with self._lock:
            task = self._task
            return task is not None and task.status in {"running", "idle"}

    @property
    def active_task(self) -> ActiveTaskState | None:
        with self._lock:
            return self._task

    @staticmethod
    def _update_task_progress_locked(task: ActiveTaskState) -> None:
        current_index = -1
        for idx, subgoal in enumerate(task.subgoals):
            if subgoal.get("status") in {"pending", "running"}:
                current_index = idx
                break
        task.current_subgoal_index = current_index
        if any(subgoal.get("status") == "failed" for subgoal in task.subgoals):
            task.status = "error"
            task.last_error = next(
                (
                    str(subgoal.get("failure_reason") or "subgoal_failed")
                    for subgoal in task.subgoals
                    if subgoal.get("status") == "failed"
                ),
                "subgoal_failed",
            )
            return
        if task.subgoals and all(subgoal.get("status") == "succeeded" for subgoal in task.subgoals):
            task.status = "completed"
            task.last_error = None
            return
        task.status = "running" if current_index >= 0 else "idle"
        if task.status != "error":
            task.last_error = None

    @staticmethod
    def _task_requires_origin_pose(task: ActiveTaskState) -> bool:
        return any(str(subgoal.get("type") or "") == "return" for subgoal in task.subgoals)

    @staticmethod
    def _navigation_status_matches_task(navigation_status: dict[str, Any], task_id: str) -> bool:
        navigation_task_id = str(navigation_status.get("task_id") or "").strip()
        return navigation_task_id == "" or navigation_task_id == task_id

    @staticmethod
    def _origin_pose_from_navigation_status(navigation_status: dict[str, Any] | None) -> dict[str, Any] | None:
        if not isinstance(navigation_status, dict):
            return None
        payload = navigation_status.get("current_robot_pose")
        if not isinstance(payload, dict):
            payload = navigation_status.get("currentRobotPose")
        if not isinstance(payload, dict):
            return None
        world_xy = payload.get("world_xy")
        if not isinstance(world_xy, (list, tuple)) or len(world_xy) < 2:
            world_xyz = payload.get("world_xyz")
            if isinstance(world_xyz, (list, tuple)) and len(world_xyz) >= 2:
                world_xy = [world_xyz[0], world_xyz[1]]
            else:
                return None
        try:
            normalized_xy = [float(world_xy[0]), float(world_xy[1])]
        except (TypeError, ValueError):
            return None
        yaw_rad = payload.get("yaw_rad")
        if yaw_rad is None:
            normalized_yaw = None
        else:
            try:
                normalized_yaw = float(yaw_rad)
            except (TypeError, ValueError):
                normalized_yaw = None
        return {
            "world_xy": normalized_xy,
            "yaw_rad": normalized_yaw,
        }

    @staticmethod
    def _mark_subgoal_running(subgoal: dict[str, Any]) -> None:
        subgoal["status"] = "running"
        subgoal["succeed"] = False
        subgoal["failure_reason"] = None

    @staticmethod
    def _mark_subgoal_succeeded(
        subgoal: dict[str, Any],
        *,
        raw_output: dict[str, Any],
    ) -> None:
        subgoal["attempts"] = max(1, int(subgoal.get("attempts") or 0))
        subgoal["output"] = raw_output
        subgoal["status"] = "succeeded"
        subgoal["succeed"] = True
        subgoal["failure_reason"] = None

    @staticmethod
    def _mark_subgoal_failed(
        subgoal: dict[str, Any],
        *,
        reason: str,
        raw_output: dict[str, Any],
    ) -> None:
        subgoal["attempts"] = max(1, int(subgoal.get("attempts") or 0))
        subgoal["output"] = raw_output
        subgoal["status"] = "failed"
        subgoal["succeed"] = False
        subgoal["failure_reason"] = str(reason or "subgoal_failed")

    @staticmethod
    def _navigation_transition_from_status(navigation_status: dict[str, Any]) -> tuple[bool, str] | None:
        system2 = navigation_status.get("system2")
        system2_payload = system2 if isinstance(system2, dict) else {}
        system2_status = str(system2_payload.get("status") or "").strip().lower()
        system2_mode = str(system2_payload.get("decision_mode") or "").strip().lower()
        reacquire_state = str(
            navigation_status.get("reacquire_state") or navigation_status.get("reacquireState") or ""
        ).strip().lower()
        if reacquire_state == "reacquired":
            return (True, "memory_target_reacquired")
        if reacquire_state in {"failed", "reacquire_failed"}:
            return (False, "memory_target_not_reacquired")
        stop_detected = system2_status == "stop" or system2_mode == "stop"
        goal_world_xy = navigation_status.get("goal_world_xy")
        action_override_mode = navigation_status.get("action_override_mode")
        path_points = int(navigation_status.get("path_points") or 0)
        no_active_navigation = goal_world_xy is None and action_override_mode is None and path_points == 0
        last_error = str(navigation_status.get("last_error") or "").strip()
        if str(navigation_status.get("status") or "").strip().lower() == "error" and no_active_navigation and last_error:
            return (False, last_error)
        if stop_detected and no_active_navigation:
            return (True, "navigate_stopped")
        return None

    def _dispatch_navigation_command(
        self,
        command_kind: str,
        command_payload: object,
        *,
        task_id: str,
    ) -> dict[str, Any]:
        if command_kind == "memory_pose":
            assert isinstance(command_payload, dict)
            return self._navigation.command_memory_target(command_payload, task_id=task_id)
        if command_kind == "return_pose":
            assert isinstance(command_payload, dict)
            return self._navigation.command_return_pose(command_payload, task_id=task_id)
        assert isinstance(command_payload, str)
        return self._navigation.command(command_payload, "en", task_id=task_id)

    def _advance_task(self) -> None:
        try:
            navigation_status = self._navigation.status()
        except Exception:
            navigation_status = None
        if not isinstance(navigation_status, dict) or not bool(navigation_status.get("ok")):
            navigation_status = None

        while True:
            dispatch_kind: str | None = None
            dispatch_payload: object | None = None
            dispatch_subgoal_id: str | None = None
            task_id: str | None = None
            with self._lock:
                task = self._task
                if task is None or task.status not in {"running", "idle"}:
                    return
                self._update_task_progress_locked(task)
                current_subgoal = task.current_subgoal
                if current_subgoal is None:
                    return

                current_type = str(current_subgoal.get("type") or "")
                current_status = str(current_subgoal.get("status") or "")

                if current_type in {"navigate", "return"} and current_status == "running":
                    if navigation_status is None or not self._navigation_status_matches_task(navigation_status, task.task_id):
                        return
                    transition = self._navigation_transition_from_status(navigation_status)
                    if transition is None:
                        return
                    succeeded, reason = transition
                    if succeeded and current_type == "return" and reason == "navigate_stopped":
                        reason = "return_goal_reached"
                    raw_output = {
                        "navigation_status": dict(navigation_status),
                        "reason": reason,
                    }
                    if succeeded:
                        self._mark_subgoal_succeeded(current_subgoal, raw_output=raw_output)
                    else:
                        self._mark_subgoal_failed(current_subgoal, reason=reason, raw_output=raw_output)
                    self._update_task_progress_locked(task)
                    continue

                if current_status == "pending":
                    if self._task_requires_origin_pose(task) and task.origin_pose is None:
                        task.origin_pose = self._origin_pose_from_navigation_status(navigation_status)
                    if current_type == "report":
                        self._mark_subgoal_succeeded(
                            current_subgoal,
                            raw_output={
                                "message": render_report_message(task.task_frame, task.subgoals),
                                "delivered": True,
                            },
                        )
                        self._update_task_progress_locked(task)
                        continue
                    if current_type == "inspect":
                        return
                    if current_type == "navigate":
                        if self._task_requires_origin_pose(task) and task.origin_pose is None:
                            return
                        current_input = current_subgoal.get("input") if isinstance(current_subgoal.get("input"), dict) else {}
                        navigation_target = current_input.get("navigation_target")
                        if isinstance(navigation_target, dict) and navigation_target.get("mode") == "memory_pose":
                            dispatch_kind = "memory_pose"
                            dispatch_payload = dict(navigation_target)
                        else:
                            dispatch_kind = "instruction"
                            dispatch_payload = build_navigation_instruction(current_subgoal["input"]["target"], language="en")
                    elif current_type == "return":
                        if task.origin_pose is None:
                            return
                        dispatch_kind = "return_pose"
                        dispatch_payload = dict(task.origin_pose)
                    else:
                        return
                    self._mark_subgoal_running(current_subgoal)
                    dispatch_subgoal_id = str(current_subgoal.get("id") or "")
                    task_id = task.task_id
                    self._update_task_progress_locked(task)
                else:
                    return

            if dispatch_kind is None or dispatch_subgoal_id is None or task_id is None:
                return

            try:
                response = self._dispatch_navigation_command(dispatch_kind, dispatch_payload, task_id=task_id)
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    task = self._task
                    if task is None or task.task_id != task_id:
                        return
                    failed_subgoal = next(
                        (item for item in task.subgoals if str(item.get("id") or "") == dispatch_subgoal_id),
                        None,
                    )
                    if failed_subgoal is None or failed_subgoal.get("status") != "running":
                        return
                    self._mark_subgoal_failed(
                        failed_subgoal,
                        reason=reason,
                        raw_output={"error": reason},
                    )
                    self._update_task_progress_locked(task)
                return

            navigation_status = response if isinstance(response, dict) else None

    def start_task(
        self,
        planning_result: PlanningResult,
        *,
        language: str,
        task_id: str | None = None,
    ) -> dict[str, object]:
        task = ActiveTaskState(
            task_id=str(task_id or f"task-{time.time_ns()}"),
            instruction=planning_result.instruction,
            language=str(language).strip() or "auto",
            scene_preset=planning_result.scene_preset,
            task_frame=planning_result.task_frame,
            subgoals=planning_result.subgoals,
            current_subgoal_index=0 if planning_result.subgoals else -1,
            status="running" if planning_result.subgoals else "idle",
            started_at=time.time(),
            origin_pose=None,
            memory_resolution=planning_result.memory_resolution,
        )

        with self._lock:
            self._task = task
            self._update_task_progress_locked(task)
        return self.status_payload()

    def submit_task(
        self,
        instruction: str,
        language: str,
        *,
        task_id: str | None = None,
        scene_preset: str | None = None,
    ) -> dict[str, object]:
        planning_result = self._planning_engine.prepare_task(instruction, scene_preset=scene_preset)
        return self.start_task(planning_result, language=language, task_id=task_id)

    def cancel(self) -> dict[str, object]:
        with self._lock:
            task = self._task
            if task is not None:
                task.status = "cancelled"
        try:
            self._navigation.cancel()
        except Exception:
            pass
        return self.status_payload()

    def status_payload(self) -> dict[str, object]:
        self._advance_task()
        with self._lock:
            task = self._task
        knowledge_status = self._planning_engine.knowledge_runtime.status_snapshot()
        object_memory = self._planning_engine.object_memory_runtime
        planner_catalog_status = self._planning_engine.planner_catalog_runtime.status_snapshot()
        if task is None:
            return {
                "task_status": "idle",
                "task_id": None,
                "instruction": "",
                "language": "auto",
                "task_frame": None,
                "current_subgoal": None,
                "subgoals": [],
                "started_at": None,
                "last_error": None,
                "memoryAwareTaskActive": False,
                "memoryNavigationMode": None,
                "resolvedMemoryObjectId": None,
                "resolvedMemoryPoseAgeSec": None,
                "reacquireState": None,
                "memoryResolution": None,
                "object_memory": {
                    "enabled": object_memory.enabled,
                    "available": object_memory.available,
                    "degraded_reason": object_memory.degraded_reason,
                    "user_id": object_memory.user_id,
                },
                "knowledge": {
                    "enabled": knowledge_status.knowledge_enabled,
                    "available": knowledge_status.available,
                    "published_document_count": knowledge_status.published_document_count,
                    "active_hard_rule_count": knowledge_status.active_hard_rule_count,
                    "lexicon_entry_count": knowledge_status.lexicon_entry_count,
                    "last_refresh_ok": knowledge_status.last_refresh_ok,
                    "last_applied_rule_ids": knowledge_status.last_applied_rule_ids,
                    "degraded_reason": knowledge_status.degraded_reason,
                },
                "planner_catalog": {
                    "enabled": planner_catalog_status.enabled,
                    "available": planner_catalog_status.available,
                    "writable": planner_catalog_status.writable,
                    "source": planner_catalog_status.source,
                    "active_intent_count": planner_catalog_status.active_intent_count,
                    "active_subgoal_template_count": planner_catalog_status.active_subgoal_template_count,
                    "last_refresh_ok": planner_catalog_status.last_refresh_ok,
                    "degraded_reason": planner_catalog_status.degraded_reason,
                },
            }
        current_subgoal = task.current_subgoal
        return {
            "task_status": task.status,
            "task_id": task.task_id,
            "instruction": task.instruction,
            "language": task.language,
            "task_frame": task.task_frame,
            "current_subgoal": current_subgoal,
            "subgoals": task.subgoals,
            "started_at": task.started_at,
            "last_error": task.last_error,
            "memoryAwareTaskActive": bool(
                isinstance(current_subgoal, dict)
                and isinstance(current_subgoal.get("input"), dict)
                and isinstance(current_subgoal["input"].get("navigation_target"), dict)
                and current_subgoal["input"]["navigation_target"].get("mode") == "memory_pose"
            ),
            "memoryNavigationMode": (
                "memory_pose"
                if isinstance(current_subgoal, dict)
                and isinstance(current_subgoal.get("input"), dict)
                and isinstance(current_subgoal["input"].get("navigation_target"), dict)
                and current_subgoal["input"]["navigation_target"].get("mode") == "memory_pose"
                else None
            ),
            "resolvedMemoryObjectId": (
                current_subgoal["input"]["navigation_target"].get("object_id")
                if isinstance(current_subgoal, dict)
                and isinstance(current_subgoal.get("input"), dict)
                and isinstance(current_subgoal["input"].get("navigation_target"), dict)
                else None
            ),
            "resolvedMemoryPoseAgeSec": (
                current_subgoal["input"]["navigation_target"].get("pose_age_sec")
                if isinstance(current_subgoal, dict)
                and isinstance(current_subgoal.get("input"), dict)
                and isinstance(current_subgoal["input"].get("navigation_target"), dict)
                else None
            ),
            "reacquireState": None,
            "memoryResolution": None if task.memory_resolution is None else dict(task.memory_resolution),
            "object_memory": {
                "enabled": object_memory.enabled,
                "available": object_memory.available,
                "degraded_reason": object_memory.degraded_reason,
                "user_id": object_memory.user_id,
            },
            "knowledge": {
                "enabled": knowledge_status.knowledge_enabled,
                "available": knowledge_status.available,
                "published_document_count": knowledge_status.published_document_count,
                "active_hard_rule_count": knowledge_status.active_hard_rule_count,
                "lexicon_entry_count": knowledge_status.lexicon_entry_count,
                "last_refresh_ok": knowledge_status.last_refresh_ok,
                "last_applied_rule_ids": knowledge_status.last_applied_rule_ids,
                "degraded_reason": knowledge_status.degraded_reason,
            },
            "planner_catalog": {
                "enabled": planner_catalog_status.enabled,
                "available": planner_catalog_status.available,
                "writable": planner_catalog_status.writable,
                "source": planner_catalog_status.source,
                "active_intent_count": planner_catalog_status.active_intent_count,
                "active_subgoal_template_count": planner_catalog_status.active_subgoal_template_count,
                "last_refresh_ok": planner_catalog_status.last_refresh_ok,
                "degraded_reason": planner_catalog_status.degraded_reason,
            },
        }


class ReasoningSystem(TaskCoordinator):
    def __init__(self, args: argparse.Namespace):
        planning_engine = PlanningEngine(args)
        self._planning_engine_impl = planning_engine
        super().__init__(args, planning_engine=planning_engine)

    @property
    def _adapter(self):
        return self._planning_engine_impl._adapter

    @_adapter.setter
    def _adapter(self, value) -> None:
        self._planning_engine_impl._adapter = value

    @property
    def _knowledge(self):
        return self._planning_engine_impl._knowledge

    @_knowledge.setter
    def _knowledge(self, value) -> None:
        self._planning_engine_impl._knowledge = value
        self._planning_engine_impl._adapter.knowledge_runtime = value

    @property
    def _object_memory(self):
        return self._planning_engine_impl._object_memory

    @_object_memory.setter
    def _object_memory(self, value) -> None:
        self._planning_engine_impl._object_memory = value

    @property
    def _planner_catalog(self):
        return self._planning_engine_impl._planner_catalog

    @_planner_catalog.setter
    def _planner_catalog(self, value) -> None:
        self._planning_engine_impl._planner_catalog = value
        self._planning_engine_impl._adapter.planner_catalog_runtime = value


class ReasoningCoordinator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self._planning_engine = PlanningEngine(args)
        self._task_coordinator = TaskCoordinator(args, planning_engine=self._planning_engine)
        dialogue_completion = None
        dialogue_model_base_url = str(getattr(args, "dialogue_model_base_url", "") or "").strip()
        if dialogue_model_base_url:
            dialogue_completion = make_http_completion(dialogue_model_base_url)
        self._dialogue = DialogueService(
            completion=dialogue_completion,
            model=str(args.dialogue_model),
            timeout=float(args.dialogue_timeout),
        )
        self._conversation_memory = create_conversation_memory_runtime(
            dsn=str(getattr(args, "conversation_memory_dsn", "") or ""),
        )
        self._agent_memory = create_humanoid_memory_runtime(
            dsn=str(getattr(args, "agent_memory_dsn", "") or ""),
            object_memory_dsn=str(getattr(args, "object_memory_dsn", "") or ""),
        )
        self._interpreter = InputInterpreter(self._planning_engine.planner_service)
        self._policy = DialoguePolicy()
        self._last_route: str | None = None
        self._last_error: str | None = None
        self._last_reply_at: float | None = None
        self._last_dialogue_reply_at: float | None = None

    @property
    def conversation_memory(self) -> ConversationMemoryRuntimeHandle:
        return self._conversation_memory

    @property
    def agent_memory(self) -> HumanoidMemoryRuntimeHandle:
        return self._agent_memory

    def respond(self, payload: dict[str, object]) -> dict[str, object]:
        utterance = " ".join(str(payload.get("utterance", "")).strip().split())
        language = str(payload.get("language", "auto")).strip() or "auto"
        conversation_id = str(payload.get("conversation_id", "")).strip()
        scene_preset = payload.get("scene_preset")
        interrupt_current_task = bool(payload.get("interrupt_current_task", False))
        request_id = str(uuid.uuid4())

        if utterance == "":
            raise ValueError("utterance must be a non-empty string")
        if conversation_id == "":
            raise ValueError("conversation_id must be a non-empty string")
        if scene_preset is not None and not isinstance(scene_preset, str):
            raise ValueError("scene_preset must be a string")

        conversation_context = self._conversation_memory.load_context(
            conversation_id,
            max_turns=int(getattr(self.args, "conversation_max_turns", 8)),
        )
        if self._is_stop_request(utterance):
            self._last_route = "task"
            return self._handle_stop_request(
                request_id=request_id,
                conversation_id=conversation_id,
                utterance=utterance,
                resolved_slots=conversation_context.resolved_slots,
            )
        normalized_scene_preset = str(scene_preset or "").strip() or None
        agent_memory_context = self._compile_agent_memory_context(
            utterance,
            conversation_context=conversation_context,
            scene_preset=normalized_scene_preset,
        )
        route_decision = self._interpreter.interpret(
            utterance,
            conversation_context=conversation_context,
            scene_preset=normalized_scene_preset,
            task_active=self._task_coordinator.has_active_task,
            interrupt_current_task=interrupt_current_task,
        )
        self._last_route = route_decision.route

        if route_decision.route == "dialogue":
            return self._handle_dialogue(
                request_id=request_id,
                conversation_id=conversation_id,
                utterance=utterance,
                language=language,
                scene_preset=normalized_scene_preset,
                conversation_context=conversation_context,
                agent_memory_context=agent_memory_context,
            )
        if route_decision.route == "clarification":
            return self._handle_clarification(
                request_id=request_id,
                conversation_id=conversation_id,
                utterance=utterance,
                instruction=route_decision.interpreted_utterance,
                scene_preset=normalized_scene_preset,
                agent_memory_context=agent_memory_context,
            )
        if route_decision.route == "unsupported":
            return self._handle_unsupported(
                request_id=request_id,
                conversation_id=conversation_id,
                utterance=utterance,
                instruction=route_decision.interpreted_utterance,
                scene_preset=normalized_scene_preset,
                agent_memory_context=agent_memory_context,
            )
        if route_decision.route == "busy":
            return self._handle_busy(
                request_id=request_id,
                conversation_id=conversation_id,
                utterance=utterance,
                resolved_slots=conversation_context.resolved_slots,
            )
        return self._handle_task(
            request_id=request_id,
            conversation_id=conversation_id,
            utterance=utterance,
            instruction=route_decision.interpreted_utterance,
            language=language,
            scene_preset=normalized_scene_preset,
            interrupt_current_task=interrupt_current_task,
            agent_memory_context=agent_memory_context,
        )

    def _compile_agent_memory_context(
        self,
        utterance: str,
        *,
        conversation_context,
        scene_preset: str | None,
    ) -> AgentMemoryContext:
        object_memory_context = None
        if self._planning_engine.object_memory_runtime.enabled:
            object_memory_context = self._planning_engine.object_memory_runtime.recent_context(
                top_k=10,
                scene_scope=scene_preset,
            )
        knowledge_context = None
        if self._planning_engine.knowledge_runtime.enabled:
            knowledge_context = self._planning_engine.knowledge_runtime.retrieve_for_plan(
                utterance,
                scene_scope=scene_preset,
            )
        return self._agent_memory.compile_context(
            utterance,
            conversation_context=conversation_context,
            object_memory_context=object_memory_context,
            knowledge_context=knowledge_context,
            scene_scope=scene_preset,
        )

    @staticmethod
    def _is_stop_request(utterance: str) -> bool:
        return bool(STOP_REQUEST_PATTERN.fullmatch(utterance))

    def _handle_stop_request(
        self,
        *,
        request_id: str,
        conversation_id: str,
        utterance: str,
        resolved_slots: dict[str, str],
    ) -> dict[str, object]:
        had_active_task = self._task_coordinator.has_active_task
        cancelled_status = self._task_coordinator.cancel() if had_active_task else self._task_coordinator.status_payload()
        reply_text = "Task cancelled." if had_active_task else "No active task to cancel."
        next_resolved_slots = dict(resolved_slots)
        task_id = str(cancelled_status.get("task_id") or "").strip()
        if task_id:
            next_resolved_slots["last_task_id"] = task_id
        self._last_reply_at = time.time()
        self._last_error = None
        self._record_turns(
            conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route="task",
            resolved_slots=next_resolved_slots,
        )
        task_payload = None
        if cancelled_status.get("task_id") is not None:
            task_payload = {
                "task_id": cancelled_status.get("task_id"),
                "task_status": cancelled_status.get("task_status"),
                "task_frame": cancelled_status.get("task_frame"),
                "current_subgoal": cancelled_status.get("current_subgoal"),
                "subgoals": cancelled_status.get("subgoals"),
            }
        return {
            "ok": True,
            "route": "task",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "reply_text": reply_text,
            "task": task_payload,
            "error": None,
        }

    def _handle_dialogue(
        self,
        *,
        request_id: str,
        conversation_id: str,
        utterance: str,
        language: str,
        scene_preset: str | None,
        conversation_context,
        agent_memory_context: AgentMemoryContext,
    ) -> dict[str, object]:
        dialogue_result = self._dialogue.respond(
            utterance,
            language=language,
            conversation_context=conversation_context,
            scene_preset=scene_preset,
            agent_memory_context=agent_memory_context,
            fallback_text=self._policy.degraded_dialogue_reply(),
        )
        reply_text = dialogue_result.reply_text
        self._last_reply_at = time.time()
        self._last_dialogue_reply_at = self._last_reply_at
        self._last_error = dialogue_result.degraded_reason
        self._record_turns(
            conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route="dialogue",
            resolved_slots=conversation_context.resolved_slots,
            scene_scope=scene_preset,
        )
        return {
            "ok": True,
            "route": "dialogue",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "reply_text": reply_text,
            "task": None,
            "error": dialogue_result.degraded_reason,
        }

    def _handle_clarification(
        self,
        *,
        request_id: str,
        conversation_id: str,
        utterance: str,
        instruction: str,
        scene_preset: str | None,
        agent_memory_context: AgentMemoryContext,
    ) -> dict[str, object]:
        planning_result = self._planning_engine.prepare_task(
            instruction,
            scene_preset=scene_preset,
            agent_memory_context=agent_memory_context,
        )
        reply_text = self._policy.clarification_reply(planning_result.task_frame)
        resolved_slots = self._build_resolved_slots(planning_result.task_frame, None)
        self._last_reply_at = time.time()
        self._last_error = None
        self._record_turns(
            conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route="clarification",
            resolved_slots=resolved_slots,
            scene_scope=scene_preset,
        )
        return {
            "ok": True,
            "route": "clarification",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "reply_text": reply_text,
            "task": None,
            "error": None,
        }

    def _handle_unsupported(
        self,
        *,
        request_id: str,
        conversation_id: str,
        utterance: str,
        instruction: str,
        scene_preset: str | None,
        agent_memory_context: AgentMemoryContext,
    ) -> dict[str, object]:
        planning_result = self._planning_engine.prepare_task(
            instruction,
            scene_preset=scene_preset,
            agent_memory_context=agent_memory_context,
        )
        reply_text = self._policy.unsupported_reply(planning_result.task_frame)
        resolved_slots = self._build_resolved_slots(planning_result.task_frame, None)
        self._last_reply_at = time.time()
        self._last_error = None
        self._record_turns(
            conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route="unsupported",
            resolved_slots=resolved_slots,
            scene_scope=scene_preset,
        )
        return {
            "ok": True,
            "route": "unsupported",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "reply_text": reply_text,
            "task": None,
            "error": None,
        }

    def _handle_busy(
        self,
        *,
        request_id: str,
        conversation_id: str,
        utterance: str,
        resolved_slots: dict[str, str],
    ) -> dict[str, object]:
        reply_text = self._policy.busy_reply()
        self._last_reply_at = time.time()
        self._last_error = None
        self._record_turns(
            conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route="busy",
            resolved_slots=resolved_slots,
        )
        return {
            "ok": True,
            "route": "busy",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "reply_text": reply_text,
            "task": None,
            "error": None,
        }

    def _handle_task(
        self,
        *,
        request_id: str,
        conversation_id: str,
        utterance: str,
        instruction: str,
        language: str,
        scene_preset: str | None,
        interrupt_current_task: bool,
        agent_memory_context: AgentMemoryContext,
    ) -> dict[str, object]:
        if interrupt_current_task and self._task_coordinator.has_active_task:
            self._task_coordinator.cancel()
        try:
            planning_result = self._planning_engine.prepare_task(
                instruction,
                scene_preset=scene_preset,
                agent_memory_context=agent_memory_context,
            )
            intent = str(planning_result.task_frame.get("intent") or "").strip()
            if intent == "ask_clarification":
                return self._handle_clarification(
                    request_id=request_id,
                    conversation_id=conversation_id,
                    utterance=utterance,
                    instruction=instruction,
                    scene_preset=scene_preset,
                    agent_memory_context=agent_memory_context,
                )
            if intent == "unsupported":
                return self._handle_unsupported(
                    request_id=request_id,
                    conversation_id=conversation_id,
                    utterance=utterance,
                    instruction=instruction,
                    scene_preset=scene_preset,
                    agent_memory_context=agent_memory_context,
                )
            task_status = self._task_coordinator.start_task(
                planning_result,
                language=language,
                task_id=f"task-{time.time_ns()}",
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {
                "ok": False,
                "route": "task",
                "request_id": request_id,
                "conversation_id": conversation_id,
                "reply_text": None,
                "task": None,
                "error": self._last_error,
            }

        reply_text = self._policy.accepted_task_reply(planning_result.task_frame)
        resolved_slots = self._build_resolved_slots(planning_result.task_frame, str(task_status.get("task_id") or ""))
        self._last_reply_at = time.time()
        self._last_error = None
        self._record_turns(
            conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route="task",
            resolved_slots=resolved_slots,
            scene_scope=scene_preset,
            task_status=str(task_status.get("task_status") or ""),
        )
        return {
            "ok": True,
            "route": "task",
            "request_id": request_id,
            "conversation_id": conversation_id,
            "reply_text": reply_text,
            "task": {
                "task_id": task_status.get("task_id"),
                "task_status": task_status.get("task_status"),
                "task_frame": task_status.get("task_frame"),
                "current_subgoal": task_status.get("current_subgoal"),
                "subgoals": task_status.get("subgoals"),
            },
            "error": None,
        }

    def cancel(self) -> dict[str, object]:
        self._last_route = "task"
        return {
            "ok": True,
            "cancelled": True,
            "status": self._task_coordinator.cancel(),
        }

    def status_payload(self) -> dict[str, object]:
        task_status = self._task_coordinator.status_payload()
        memory_status = self._conversation_memory.status_snapshot()
        agent_memory_status = self._agent_memory.status_snapshot()
        planner_catalog_status = self._planning_engine.planner_catalog_runtime.status_snapshot()
        return {
            "ok": True,
            "service": "reasoning_system",
            "active_task": None if task_status.get("task_id") is None else {
                "task_id": task_status.get("task_id"),
                "task_status": task_status.get("task_status"),
                "task_frame": task_status.get("task_frame"),
                "current_subgoal": task_status.get("current_subgoal"),
                "subgoals": task_status.get("subgoals"),
            },
            "last_route": self._last_route,
            "dialogue_model_available": self._dialogue.available,
            "planner_model_available": self._planning_engine.model_available,
            "conversation_memory_available": memory_status.available,
            "conversation_memory_degraded_reason": memory_status.degraded_reason,
            "agent_memory_available": agent_memory_status.available,
            "agent_memory_degraded_reason": agent_memory_status.degraded_reason,
            "agent_memory_core_block_count": agent_memory_status.core_block_count,
            "agent_memory_archival_passage_count": agent_memory_status.archival_passage_count,
            "planner_catalog_available": planner_catalog_status.available,
            "planner_catalog_degraded_reason": planner_catalog_status.degraded_reason,
            "last_error": self._last_error or task_status.get("last_error"),
            "last_reply_at": self._last_reply_at,
            "last_dialogue_reply_at": self._last_dialogue_reply_at,
            **task_status,
        }

    def _record_turns(
        self,
        conversation_id: str,
        *,
        utterance: str,
        reply_text: str,
        route: str,
        resolved_slots: dict[str, str],
        scene_scope: str | None = None,
        task_status: str | None = None,
    ) -> None:
        self._conversation_memory.append_turn(
            conversation_id,
            "user",
            utterance,
            {"route": route},
        )
        self._conversation_memory.append_turn(
            conversation_id,
            "assistant",
            reply_text,
            {"route": route},
        )
        self._conversation_memory.update_summary(
            conversation_id,
            self._roll_summary(conversation_id),
            resolved_slots,
        )
        self._agent_memory.record_interaction(
            conversation_id=conversation_id,
            utterance=utterance,
            reply_text=reply_text,
            route=route,
            resolved_slots=resolved_slots,
            scene_scope=scene_scope,
            task_status=task_status,
        )

    def _roll_summary(self, conversation_id: str) -> str:
        context = self._conversation_memory.load_context(conversation_id, max_turns=6)
        parts = [f"{turn.role}: {turn.text}" for turn in context.recent_turns[-4:]]
        return " | ".join(parts)[:512]

    @staticmethod
    def _build_resolved_slots(task_frame: dict[str, Any], task_id: str | None) -> dict[str, str]:
        target_payload = task_frame.get("target") if isinstance(task_frame, dict) else {}
        query_payload = task_frame.get("query") if isinstance(task_frame, dict) else {}
        resolved_slots: dict[str, str] = {}
        if isinstance(target_payload, dict):
            target_object = str(target_payload.get("object") or "").strip()
            location_hint = str(target_payload.get("location_hint") or "").strip()
            if target_object:
                resolved_slots["last_target_object"] = target_object
            if location_hint:
                resolved_slots["last_room_hint"] = location_hint
        if isinstance(query_payload, dict):
            attribute = str(query_payload.get("attribute") or "").strip()
            if attribute:
                resolved_slots["last_attribute"] = attribute
        if task_id:
            resolved_slots["last_task_id"] = str(task_id)
        return resolved_slots


class ReasoningSystemServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._service = ReasoningCoordinator(args)
        self._server = ThreadingHTTPServer((str(args.host), int(args.port)), self._build_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="reasoning-system-api", daemon=True)

    def _build_handler(self):
        service = self._service

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, object]):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(int(status_code))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def _read_json_body(self) -> dict[str, object]:
                content_length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(content_length) if content_length > 0 else b""
                if not raw:
                    return {}
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("expected JSON object body")
                return payload

            def do_OPTIONS(self):
                self._send_json(HTTPStatus.NO_CONTENT, {})

            def do_GET(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path == "/healthz":
                    self._send_json(HTTPStatus.OK, {"ok": True, "service": "reasoning_system"})
                    return
                if path == "/reasoning/status":
                    self._send_json(HTTPStatus.OK, service.status_payload())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def do_POST(self):
                path = urlparse(self.path).path.rstrip("/") or "/"
                try:
                    payload = self._read_json_body()
                except json.JSONDecodeError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": f"invalid_json: {exc}"})
                    return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    return

                try:
                    if path == "/reasoning/respond":
                        response = service.respond(payload)
                        status_code = HTTPStatus.OK if bool(response.get("ok", True)) else HTTPStatus.BAD_GATEWAY
                        self._send_json(status_code, response)
                        return
                    if path == "/reasoning/cancel":
                        self._send_json(HTTPStatus.OK, service.cancel())
                        return
                except ValueError as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args):
                del format, args

        return Handler

    def start(self) -> None:
        self._thread.start()
        print(f"[INFO] Reasoning system API listening on http://{self.args.host}:{self.args.port}")

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the reasoning system service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=17881)
    parser.add_argument("--navigation-url", default="http://127.0.0.1:17882")
    parser.add_argument("--navigation-timeout", type=float, default=5.0)
    parser.add_argument("--planner-model-base-url", default="http://127.0.0.1:8093/v1/chat/completions")
    parser.add_argument("--planner-model", default="Qwen3-1.7B-Q4_K_M-Instruct.gguf")
    parser.add_argument("--planner-timeout", type=float, default=120.0)
    parser.add_argument(
        "--planner-intent-slot-id",
        type=int,
        default=int(os.environ.get("PLANNER_INTENT_SLOT_ID", "0")),
    )
    parser.add_argument(
        "--planner-task-frame-slot-id",
        type=int,
        default=int(os.environ.get("PLANNER_TASK_FRAME_SLOT_ID", "1")),
    )
    parser.add_argument("--dialogue-model-base-url", default="http://127.0.0.1:8094/v1/chat/completions")
    parser.add_argument("--dialogue-model", default="Qwen3-1.7B-Q4_K_M-Instruct.gguf")
    parser.add_argument("--dialogue-timeout", type=float, default=30.0)
    parser.add_argument("--conversation-max-turns", type=int, default=8)
    parser.add_argument("--object-memory-dsn", default=os.environ.get("AURA_OBJECT_MEMORY_DSN", ""))
    parser.add_argument("--memory-user-id", default=os.environ.get("AURA_MEMORY_USER_ID", "local-operator"))
    parser.add_argument(
        "--object-memory-auto-migrate",
        dest="object_memory_auto_migrate",
        action="store_true",
    )
    parser.add_argument(
        "--object-memory-no-auto-migrate",
        dest="object_memory_auto_migrate",
        action="store_false",
    )
    parser.add_argument(
        "--memory-pose-max-age-sec",
        type=int,
        default=int(os.environ.get("AURA_MEMORY_POSE_MAX_AGE_SEC", "600")),
    )
    parser.add_argument(
        "--memory-approach-radius-m",
        type=float,
        default=float(os.environ.get("AURA_MEMORY_APPROACH_RADIUS_M", "0.8")),
    )
    parser.add_argument(
        "--memory-reacquire-radius-m",
        type=float,
        default=float(os.environ.get("AURA_MEMORY_REACQUIRE_RADIUS_M", "1.0")),
    )
    parser.add_argument(
        "--memory-reacquire-timeout-sec",
        type=float,
        default=float(os.environ.get("AURA_MEMORY_REACQUIRE_TIMEOUT_SEC", "4.0")),
    )
    parser.add_argument(
        "--knowledge-dsn",
        default=os.environ.get("AURA_KNOWLEDGE_DSN", os.environ.get("AURA_OBJECT_MEMORY_DSN", "")),
    )
    parser.add_argument(
        "--planner-catalog-dsn",
        default=os.environ.get(
            "AURA_PLANNER_CATALOG_DSN",
            os.environ.get("AURA_KNOWLEDGE_DSN", os.environ.get("AURA_OBJECT_MEMORY_DSN", "")),
        ),
    )
    parser.add_argument(
        "--conversation-memory-dsn",
        default=os.environ.get(
            "AURA_CONVERSATION_MEMORY_DSN",
            os.environ.get("AURA_OBJECT_MEMORY_DSN", ""),
        ),
    )
    parser.add_argument(
        "--agent-memory-dsn",
        default=os.environ.get(
            "AURA_AGENT_MEMORY_DSN",
            os.environ.get("AURA_OBJECT_MEMORY_DSN", ""),
        ),
    )
    parser.add_argument("--knowledge-scene-scope", default=os.environ.get("AURA_SCENE_PRESET", ""))
    parser.set_defaults(
        object_memory_auto_migrate=str(os.environ.get("AURA_OBJECT_MEMORY_AUTO_MIGRATE", "")).strip().lower()
        in {"1", "true", "yes", "on"}
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = ReasoningSystemServer(args)
    server.start()
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
