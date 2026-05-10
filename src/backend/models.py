"""Backend state builders and serializers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import re
from typing import Any

from systems.shared.contracts.dashboard import LogRecord
from systems.shared.contracts.service_endpoints import (
    CONTROL_RUNTIME_ENDPOINT,
    INFERENCE_SYSTEM_ENDPOINT,
    NAVIGATION_SYSTEM_ENDPOINT,
    REASONING_SYSTEM_ENDPOINT,
    RUNTIME_ENDPOINT,
)


DEFAULT_EXECUTION_MODES = ["TALK", "NAV", "MEM_NAV", "EXPLORE", "IDLE"]
DEFAULT_LAUNCH_MODES = ["gui", "headless"]
DEFAULT_SCENE_PRESETS = ["warehouse", "interioragent", "interior agent kujiale 3"]
DEFAULT_LOCOMOTION_CONFIG = {
    "actionScale": 0.5,
    "onnxDevice": "auto",
    "cmdMaxVx": 0.5,
    "cmdMaxVy": 0.3,
    "cmdMaxWz": 0.8,
}
SCENE_LABELS = {
    "warehouse": "Warehouse Floor",
    "interioragent": "Interior Agent Lab",
    "interior agent kujiale 3": "Interior Agent Kujiale 3",
}


def _node(name: str, status: str, summary: str, detail: str, *, required: bool, latency_ms: float | None = None, metrics=None):
    return {
        "name": name,
        "status": status,
        "summary": summary,
        "detail": detail,
        "required": required,
        "latencyMs": latency_ms,
        "metrics": {} if metrics is None else dict(metrics),
    }


def build_bootstrap_data(*, api_base_url: str, dev_origin: str, webrtc_base_path: str) -> dict[str, Any]:
    normalized_base = str(api_base_url).rstrip("/")
    return {
        "executionModes": list(DEFAULT_EXECUTION_MODES),
        "launchModes": list(DEFAULT_LAUNCH_MODES),
        "scenePresets": list(DEFAULT_SCENE_PRESETS),
        "apiBaseUrl": normalized_base,
        "devOrigin": str(dev_origin),
        "webrtcBasePath": str(webrtc_base_path),
        "reasoningRoutes": ["task", "dialogue", "clarification", "unsupported", "busy"],
        "plannerModes": ["interactive", "pointgoal"],
    }


def default_log(source: str, message: str) -> LogRecord:
    return LogRecord(
        source=source,
        stream="event",
        level="info",
        message=message,
        timestampNs=int(time.time() * 1_000_000_000),
    )


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string(value: Any, fallback: str = "") -> str:
    return value if isinstance(value, str) else fallback


def _number(value: Any) -> float | None:
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _coerce_bool(value: Any, *, field: str, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field} must be a boolean")


def _coerce_choice(value: Any, *, field: str, allowed: list[str], default: str) -> str:
    if value is None:
        return str(default)
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = value.strip()
    if normalized == "":
        return str(default)
    lowered = normalized.lower()
    for candidate in allowed:
        if lowered == candidate.lower():
            return candidate
    raise ValueError(f"{field} must be one of {allowed}")


def _coerce_non_negative_number(value: Any, *, field: str, default: float) -> float:
    if value is None:
        return float(default)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return numeric


def _coerce_positive_number(value: Any, *, field: str, default: float) -> float:
    numeric = _coerce_non_negative_number(value, field=field, default=default)
    if numeric <= 0.0:
        raise ValueError(f"{field} must be greater than 0")
    return numeric


def parse_session_config(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("session payload must be a JSON object")
    locomotion_payload = payload.get("locomotionConfig")
    if locomotion_payload is None:
        locomotion_payload = {}
    if not isinstance(locomotion_payload, dict):
        raise ValueError("locomotionConfig must be a JSON object")

    normalized = dict(payload)
    normalized["launchMode"] = _coerce_choice(
        payload.get("launchMode"),
        field="launchMode",
        allowed=DEFAULT_LAUNCH_MODES,
        default="gui",
    )
    scene_preset = payload.get("scenePreset")
    if scene_preset is None:
        normalized["scenePreset"] = DEFAULT_SCENE_PRESETS[0]
    elif isinstance(scene_preset, str) and scene_preset.strip():
        normalized["scenePreset"] = scene_preset.strip()
    else:
        raise ValueError("scenePreset must be a non-empty string")
    normalized["viewerEnabled"] = _coerce_bool(
        payload.get("viewerEnabled"),
        field="viewerEnabled",
        default=True,
    )
    normalized["memoryStore"] = _coerce_bool(
        payload.get("memoryStore"),
        field="memoryStore",
        default=False,
    )
    normalized["detectionEnabled"] = _coerce_bool(
        payload.get("detectionEnabled"),
        field="detectionEnabled",
        default=True,
    )
    normalized["locomotionConfig"] = {
        "actionScale": _coerce_positive_number(
            locomotion_payload.get("actionScale"),
            field="locomotionConfig.actionScale",
            default=float(DEFAULT_LOCOMOTION_CONFIG["actionScale"]),
        ),
        "onnxDevice": _coerce_choice(
            locomotion_payload.get("onnxDevice"),
            field="locomotionConfig.onnxDevice",
            allowed=["auto", "cuda", "cpu"],
            default=str(DEFAULT_LOCOMOTION_CONFIG["onnxDevice"]),
        ),
        "cmdMaxVx": _coerce_non_negative_number(
            locomotion_payload.get("cmdMaxVx"),
            field="locomotionConfig.cmdMaxVx",
            default=float(DEFAULT_LOCOMOTION_CONFIG["cmdMaxVx"]),
        ),
        "cmdMaxVy": _coerce_non_negative_number(
            locomotion_payload.get("cmdMaxVy"),
            field="locomotionConfig.cmdMaxVy",
            default=float(DEFAULT_LOCOMOTION_CONFIG["cmdMaxVy"]),
        ),
        "cmdMaxWz": _coerce_positive_number(
            locomotion_payload.get("cmdMaxWz"),
            field="locomotionConfig.cmdMaxWz",
            default=float(DEFAULT_LOCOMOTION_CONFIG["cmdMaxWz"]),
        ),
    }
    return normalized


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "item"


def _scene_label(scene_preset: str) -> str:
    normalized = scene_preset.strip().lower()
    if normalized in SCENE_LABELS:
        return SCENE_LABELS[normalized]
    words = [part for part in re.split(r"[\s_-]+", scene_preset.strip()) if part]
    return " ".join(word[:1].upper() + word[1:] for word in words) or "Unknown Scene"


def _date_label(timestamp_ns: Any, fallback_s: float) -> str:
    timestamp_s = fallback_s
    if isinstance(timestamp_ns, (int, float)):
        timestamp_s = float(timestamp_ns) / 1_000_000_000
    return time.strftime("%Y-%m-%d", time.localtime(timestamp_s))


def _source_status_from_event(level: str) -> str:
    if level == "error":
        return "error"
    if level in {"warning", "warn"}:
        return "indexing"
    return "indexed"


def _policy_status(enabled: bool) -> str:
    return "active" if enabled else "disabled"


def build_dashboard_catalog(state: dict[str, Any]) -> dict[str, Any]:
    timestamp_s = float(state.get("timestamp") or time.time())
    session = _as_dict(state.get("session"))
    runtime = _as_dict(state.get("runtime"))
    perception = _as_dict(state.get("perception"))
    memory = _as_dict(state.get("memory"))
    transport = _as_dict(state.get("transport"))
    services = _as_dict(state.get("services"))
    route_state = _as_dict(runtime.get("routeState"))
    current_subgoal = _as_dict(runtime.get("currentSubgoal"))
    selected_target = _as_dict(state.get("selectedTargetSummary"))
    session_config = _as_dict(session.get("config"))
    last_event = _as_dict(session.get("lastEvent"))
    system2 = _as_dict(services.get("system2"))
    system2_output = _as_dict(system2.get("output"))
    logs = _as_list(state.get("logs"))

    scene_preset = _string(session_config.get("scenePreset"), "warehouse")
    scene_label = _scene_label(scene_preset)
    launch_mode = _string(session_config.get("launchMode"), "gui")
    active_instruction = _string(runtime.get("activeInstruction"))
    reasoning_status = _string(runtime.get("reasoningTaskStatus"), _string(runtime.get("plannerControlMode"), "idle")) or "idle"
    execution_mode = _string(runtime.get("executionMode"), "IDLE") or "IDLE"
    recovery_state = _string(runtime.get("recoveryState"), "NORMAL") or "NORMAL"
    current_subgoal_type = _string(current_subgoal.get("type"), "observation")
    current_subgoal_label = _string(current_subgoal.get("label")) or _string(current_subgoal.get("target")) or "No active subgoal"
    selected_class = _string(selected_target.get("className"), "Tracked Object")
    selected_source = _string(selected_target.get("source"), "perception")
    last_event_level = _string(last_event.get("level"), "info").lower()
    last_event_message = _string(last_event.get("message"), "No backend events have been emitted yet.")
    last_event_date = _date_label(last_event.get("timestampNs"), timestamp_s)
    route_points = int((_number(route_state.get("pathPoints")) or 0))
    goal_distance = _number(runtime.get("goalDistanceM"))
    detection_count = int((_number(perception.get("detectionCount")) or 0))
    tracked_detection_count = int((_number(perception.get("trackedDetectionCount")) or 0))
    object_count = int((_number(memory.get("objectCount")) or 0))
    place_count = int((_number(memory.get("placeCount")) or 0))
    semantic_rule_count = int((_number(memory.get("semanticRuleCount")) or 0))
    viewer_enabled = bool(session_config.get("viewerEnabled") or transport.get("viewerEnabled"))
    memory_enabled = bool(session_config.get("memoryStore"))
    detection_enabled = bool(session_config.get("detectionEnabled"))
    session_active = bool(session.get("active"))

    sources = [
        {
            "id": "source-session-profile",
            "title": "Session Launch Profile",
            "type": "config",
            "status": "indexed" if session_active else "indexing",
            "domain": "Runtime",
            "tags": [scene_preset, launch_mode],
            "summary": f"Scene {scene_label} is prepared for {launch_mode} launch.",
            "addedAt": last_event_date,
            "linkedEntities": 2,
            "linkedPolicies": 3,
        },
        {
            "id": "source-reasoning-instruction",
            "title": "Reasoning Instruction Feed",
            "type": "task",
            "status": "indexed" if active_instruction else "indexing",
            "domain": "Reasoning",
            "tags": [reasoning_status, execution_mode],
            "summary": active_instruction or "Reasoning is idle and waiting for the next operator instruction.",
            "addedAt": last_event_date,
            "linkedEntities": 2 if active_instruction else 1,
            "linkedPolicies": 1,
        },
        {
            "id": "source-runtime-events",
            "title": "Runtime Event Feed",
            "type": "event",
            "status": _source_status_from_event(last_event_level),
            "domain": "Observability",
            "tags": [_string(last_event.get("source"), "backend"), last_event_level],
            "summary": last_event_message,
            "addedAt": last_event_date,
            "linkedEntities": 1,
            "linkedPolicies": 1,
        },
        {
            "id": "source-service-mesh",
            "title": "Service Mesh Snapshot",
            "type": "service",
            "status": "indexed" if _string(system2.get("status"), "inactive") in {"healthy", "running", "ok"} else "indexing",
            "domain": "Systems",
            "tags": [
                _string(_as_dict(services.get("navdp")).get("status"), "inactive"),
                _string(system2.get("status"), "inactive"),
            ],
            "summary": "Live service health for nav, reasoning, and system2 modules.",
            "addedAt": time.strftime("%Y-%m-%d", time.localtime(timestamp_s)),
            "linkedEntities": 3,
            "linkedPolicies": 0,
        },
    ]
    if route_points > 0 or goal_distance is not None:
        sources.append(
            {
                "id": "source-navigation-route",
                "title": "Navigation Route State",
                "type": "route",
                "status": "indexed",
                "domain": "Navigation",
                "tags": [f"path-{route_points}", recovery_state.lower()],
                "summary": (
                    f"Route planner exposes {route_points} path points"
                    + (f" with {goal_distance:.2f}m to goal." if goal_distance is not None else ".")
                ),
                "addedAt": time.strftime("%Y-%m-%d", time.localtime(timestamp_s)),
                "linkedEntities": 2,
                "linkedPolicies": 1,
            }
        )

    terms = [
        {
            "id": "term-scene",
            "canonical": scene_preset,
            "aliases": [scene_label],
            "category": "place",
            "linkedEntity": scene_label,
            "description": "Active runtime scene preset used by the dashboard session.",
        },
        {
            "id": "term-execution-mode",
            "canonical": execution_mode,
            "aliases": [execution_mode.replace("_", " ").title()],
            "category": "action",
            "linkedEntity": "Execution Mode",
            "description": "Current high-level execution mode reported by the runtime.",
        },
        {
            "id": "term-reasoning-status",
            "canonical": reasoning_status,
            "aliases": [reasoning_status.replace("_", " ")],
            "category": "state",
            "linkedEntity": "Reasoning Task",
            "description": "Reasoning task state used to stage current workflow execution.",
        },
        {
            "id": "term-selected-target",
            "canonical": _slug(selected_class),
            "aliases": [selected_class],
            "category": "object",
            "linkedEntity": selected_class,
            "description": f"Selected target surfaced by {selected_source}.",
        },
    ]

    objects = [
        {
            "id": "object-selected-target",
            "name": selected_class,
            "category": "Perception",
            "aliases": [selected_source],
            "detectable": True,
            "summary": "Current operator-facing target selected from perception output.",
        },
        {
            "id": "object-memory-store",
            "name": "Memory Object Store",
            "category": "Memory",
            "aliases": [f"{object_count} tracked objects"],
            "detectable": False,
            "summary": "Aggregated object inventory tracked by the memory subsystem.",
        },
    ]
    if active_instruction:
        objects.append(
            {
                "id": "object-reasoning-target",
                "name": current_subgoal_label,
                "category": "Reasoning",
                "aliases": [current_subgoal_type],
                "detectable": False,
                "summary": "Current reasoning target or active workflow step.",
            }
        )

    places = [
        {
            "id": "place-scene",
            "name": scene_label,
            "aliases": [scene_preset],
            "zoneType": "Runtime Scene",
            "mapLinked": True,
            "summary": "Primary spatial context used by the current runtime session.",
        },
        {
            "id": "place-route-goal",
            "name": current_subgoal_label,
            "aliases": [current_subgoal_type],
            "zoneType": "Active Goal",
            "mapLinked": route_points > 0,
            "summary": "Current goal context reported by reasoning and navigation services.",
        },
        {
            "id": "place-memory-surface",
            "name": "Spatial Memory Surface",
            "aliases": [f"{place_count} places"],
            "zoneType": "Memory",
            "mapLinked": True,
            "summary": "Places currently tracked by memory and map overlays.",
        },
    ]

    states_catalog = [
        {
            "id": "state-execution",
            "name": "runtime.execution_mode",
            "appliesTo": "Runtime",
            "allowedValues": DEFAULT_EXECUTION_MODES,
            "detectionMethod": "Runtime Telemetry",
            "summary": execution_mode,
        },
        {
            "id": "state-reasoning",
            "name": "reasoning.task_status",
            "appliesTo": "Reasoning",
            "allowedValues": ["idle", "running", "completed", "cancelled", "error"],
            "detectionMethod": "Reasoning API",
            "summary": reasoning_status,
        },
        {
            "id": "state-recovery",
            "name": "runtime.recovery_state",
            "appliesTo": "Recovery Supervisor",
            "allowedValues": ["NORMAL", "RECOVERING", "SAFE_STOP"],
            "detectionMethod": "Runtime Telemetry",
            "summary": recovery_state,
        },
        {
            "id": "state-viewer",
            "name": "transport.viewer_publish",
            "appliesTo": "Telemetry",
            "allowedValues": ["enabled", "disabled"],
            "detectionMethod": "Backend Session Config",
            "summary": "enabled" if viewer_enabled else "disabled",
        },
    ]

    policies = [
        {
            "id": "policy-viewer",
            "name": "Viewer Publish Policy",
            "targetTask": "Observation",
            "priority": "medium",
            "status": _policy_status(viewer_enabled),
            "linkedEntities": ["transport.viewer_publish", scene_label],
            "description": "Controls whether WebRTC viewer streams are published to the dashboard.",
        },
        {
            "id": "policy-memory",
            "name": "Memory Store Policy",
            "targetTask": "Knowledge",
            "priority": "high",
            "status": _policy_status(memory_enabled),
            "linkedEntities": ["Memory Object Store", "Spatial Memory Surface"],
            "description": "Determines whether runtime observations are persisted into long-term memory.",
        },
        {
            "id": "policy-detection",
            "name": "Detection Gate Policy",
            "targetTask": "Perception",
            "priority": "critical",
            "status": _policy_status(detection_enabled),
            "linkedEntities": [selected_class, "runtime.execution_mode"],
            "description": "Enables or disables perception-driven detection output used by reasoning task decisions.",
        },
        {
            "id": "policy-recovery",
            "name": "Recovery Guard",
            "targetTask": "Safety",
            "priority": "critical",
            "status": "active",
            "linkedEntities": ["runtime.recovery_state", "reasoning.task_status"],
            "description": "Promotes safe-stop and retry behavior whenever runtime health falls below nominal state.",
        },
    ]

    recent_items: list[dict[str, Any]] = [
        {
            "id": item["id"],
            "label": item["title"],
            "type": item["type"],
            "status": item["status"],
            "date": item["addedAt"],
        }
        for item in sources[:4]
    ]
    for index, log in enumerate(logs[:4]):
        log_item = _as_dict(log)
        recent_items.append(
            {
                "id": f"log-{index}",
                "label": _string(log_item.get("message"), "log entry"),
                "type": "log",
                "status": _source_status_from_event(_string(log_item.get("level"), "info").lower()),
                "date": _date_label(log_item.get("timestampNs"), timestamp_s),
            }
        )

    return {
        "summary": {
            "sourceCount": len(sources),
            "termCount": len(terms),
            "objectCount": max(len(objects), object_count),
            "placeCount": max(len(places), place_count),
            "stateCount": len(states_catalog),
            "policyCount": max(len(policies), semantic_rule_count),
            "detectionCount": detection_count,
            "trackedDetectionCount": tracked_detection_count,
        },
        "sources": sources,
        "terms": terms,
        "objects": objects,
        "places": places,
        "states": states_catalog,
        "policies": policies,
        "recentItems": recent_items[:6],
        "lastUpdated": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp_s)),
        "highlights": {
            "sceneLabel": scene_label,
            "reasoningStatus": reasoning_status,
            "plannerStatus": reasoning_status,
            "executionMode": execution_mode,
            "currentSubgoal": current_subgoal_label,
            "selectedTarget": selected_class,
            "system2Reason": _string(system2_output.get("reason"), ""),
            "system2Text": _string(system2_output.get("rawText"), ""),
        },
    }


@dataclass(slots=True)
class DashboardStateBuilder:
    api_base_url: str

    def default_state(self) -> dict[str, Any]:
        event = default_log("backend", "backend initialized")
        state = {
            "timestamp": time.time(),
            "session": {
                "active": False,
                "startedAt": None,
                "config": None,
                "conversationId": None,
                "lastEvent": event.to_dict(),
            },
            "processes": [],
            "runtime": {
                "executionMode": "IDLE",
                "reasoningTaskStatus": "idle",
                "plannerControlMode": "idle",
                "reasoningRoute": None,
                "lastDialogueReplyAt": None,
                "agentResponse": {
                    "text": "",
                    "route": None,
                    "requestId": None,
                    "conversationId": None,
                    "error": None,
                    "at": None,
                },
                "activeInstruction": "",
                "routeState": {},
                "memoryNavigationMode": None,
                "resolvedMemoryObjectId": None,
                "resolvedMemoryPoseAgeSec": None,
                "reacquireState": None,
                "lastStatusEvent": {"state": "inactive", "reason": "session not started"},
            },
            "sensors": {
                "rgbAvailable": False,
                "depthAvailable": False,
                "poseAvailable": False,
                "source": "backend",
            },
            "perception": {
                "detectorReady": False,
                "detectorBackend": "inactive",
                "detectionCount": 0,
                "trackedDetectionCount": 0,
                "trajectoryPointCount": 0,
            },
            "memory": {
                "memoryAwareTaskActive": False,
                "memoryNavigationMode": None,
                "resolvedMemoryObjectId": None,
                "resolvedMemoryPoseAgeSec": None,
                "reacquireState": None,
                "objectCount": 0,
                "objectMemoryEnabled": False,
                "objectMemoryAvailable": False,
                "observationCount": 0,
                "lastPersistOk": None,
                "lastObservedAt": None,
                "placeCount": 0,
                "knowledgeEnabled": False,
                "agentMemoryEnabled": False,
                "agentMemoryAvailable": False,
                "agentMemoryCoreBlockCount": 0,
                "agentMemoryArchivalPassageCount": 0,
                "agentMemoryDegradedReason": None,
                "publishedDocumentCount": 0,
                "activeHardRuleCount": 0,
                "lexiconEntryCount": 0,
                "lastKnowledgeRefreshOk": None,
                "lastAppliedRuleIds": [],
                "knowledgeDegradedReason": None,
                "semanticRuleCount": 0,
                "scratchpad": {"taskState": "idle", "instruction": "", "nextPriority": "start session"},
            },
            "architecture": {
                "gateway": _node("Robot Gateway", "inactive", "Session idle", "waiting for session", required=True),
                "mainControlServer": {
                    **_node("Main Control Server", "inactive", "Runtime idle", "waiting for control runtime", required=True),
                    "core": {
                        "worldStateStore": _node("World State Store", "inactive", "No snapshot", "runtime inactive", required=True),
                        "decisionEngine": _node("Decision Engine", "inactive", "No active task", "runtime inactive", required=True),
                        "reasoningCoordinator": _node("Reasoning Coordinator", "inactive", "Reasoning idle", "runtime inactive", required=True),
                        "commandResolver": _node("Command Resolver", "inactive", "No active command", "runtime inactive", required=True),
                        "safetySupervisor": _node("Safety Supervisor", "inactive", "Recovery normal", "runtime inactive", required=True),
                    },
                },
                "modules": {
                    "perception": _node("Perception", "inactive", "Waiting for frames", "runtime inactive", required=True),
                    "memory": _node("Memory", "inactive", "No active memory task", "runtime inactive", required=True),
                    "s2": _node("S2", "inactive", "Session inactive", "inference stack unavailable", required=True),
                    "nav": _node("Nav", "inactive", "No active plan", "inference stack unavailable", required=False),
                    "locomotion": _node("Locomotion", "inactive", "No active command", "runtime inactive", required=False),
                    "telemetry": _node("Telemetry", "inactive", "No active viewers", "viewer unavailable", required=False),
                },
            },
            "services": {
                "backend": {
                    "name": "backend",
                    "status": "healthy",
                    "healthUrl": f"{self.api_base_url}/api/state",
                },
                "runtime": {
                    "name": "runtime",
                    "status": "inactive",
                    "healthUrl": RUNTIME_ENDPOINT.status_url(),
                },
                "controlRuntime": {
                    "name": "control_runtime",
                    "status": "inactive",
                    "healthUrl": CONTROL_RUNTIME_ENDPOINT.status_url(),
                },
                "inferenceSystem": {
                    "name": "inference_system",
                    "status": "inactive",
                    "healthUrl": INFERENCE_SYSTEM_ENDPOINT.status_url(),
                },
                "navigationSystem": {
                    "name": "navigation_system",
                    "status": "inactive",
                    "healthUrl": NAVIGATION_SYSTEM_ENDPOINT.status_url(),
                },
                "reasoningSystem": {
                    "name": "reasoning_system",
                    "status": "inactive",
                    "healthUrl": REASONING_SYSTEM_ENDPOINT.status_url(),
                },
                "navdp": {"name": "navdp", "status": "inactive", "healthUrl": f"{self.api_base_url}/api/state"},
                "system2": {"name": "system2", "status": "inactive", "healthUrl": f"{self.api_base_url}/api/state", "output": None},
            },
            "transport": {
                "viewerEnabled": False,
                "frameAgeMs": None,
                "frameSeq": None,
                "frameAvailable": False,
                "peerActive": False,
                "peerSessionId": None,
                "peerTrackRoles": [],
                "busHealth": {"control_endpoint": "tcp://127.0.0.1:18880", "telemetry_endpoint": "tcp://127.0.0.1:18881"},
            },
            "logs": [event.to_dict()],
            "selectedTargetSummary": None,
            "latencyBreakdown": {
                "frameAgeMs": None,
                "perceptionLatencyMs": None,
                "memoryLatencyMs": None,
                "s2LatencyMs": None,
                "navLatencyMs": None,
                "locomotionLatencyMs": None,
            },
            "cognitionTrace": [],
            "recoveryTransitions": [],
        }
        state["dashboardCatalog"] = build_dashboard_catalog(state)
        return state
