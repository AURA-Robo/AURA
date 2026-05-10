"""aiohttp backend application."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import time
from aiohttp import ClientSession, web
from aiohttp.web_runner import GracefulExit
from collections.abc import Callable

from backend.app_keys import (
    AGENT_MEMORY_RUNTIME,
    API_BASE_URL,
    BROADCAST_TASK,
    CONTROL_RUNTIME_URL,
    DEV_ORIGIN,
    HTTP,
    INFERENCE_SYSTEM_URL,
    KNOWLEDGE_RUNTIME,
    NAVIGATION_SYSTEM_URL,
    PLANNER_CATALOG_RUNTIME,
    REASONING_SYSTEM_URL,
    ROOT_DIR,
    RUNTIME_OWNED,
    RUNTIME_SERVICE,
    RUNTIME_URL,
    RUNTIME_SUBMIT_TASKS,
    SESSION_MANAGER,
    SSE,
    WEBRTC_SERVICE,
    WEBRTC_PROXY_BASE,
)
from backend.occupancy import build_occupancy_payload, handle_image
from backend.session_manager import DashboardSessionManager
from backend.sse import SseBroadcaster
from backend.webrtc import WebRTCService, WebRTCServiceConfig
from backend.webrtc_proxy import get_config, proxy_offer
from backend.models import build_bootstrap_data
from runtime.service import RuntimeService
from systems.memory.agent_memory_models import AgentMemoryBlockInput, AgentMemoryPassageInput
from systems.memory.agent_memory_runtime import HumanoidMemoryRuntimeHandle, create_humanoid_memory_runtime
from systems.memory.knowledge_models import KnowledgeDocumentInput
from systems.memory.knowledge_runtime import KnowledgeRuntimeHandle, create_knowledge_runtime
from systems.memory.object_memory_models import OBJECT_MEMORY_STATUSES
from systems.reasoning.planner_catalog_errors import (
    PlannerCatalogConflictError,
    PlannerCatalogUnavailableError,
    PlannerCatalogValidationError,
)
from systems.reasoning.planner_catalog_models import (
    EXECUTION_INTENT_KEYS,
    PLANNER_INTENT_SPECS,
    SUPPORTED_ACTIVATION_CONDITIONS,
    SUPPORTED_SUBGOAL_TYPES,
)
from systems.reasoning.planner_catalog_runtime import (
    PlannerCatalogRuntimeHandle,
    create_planner_catalog_runtime,
)


async def _json_body(request: web.Request) -> dict[str, object]:
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def _raise_graceful_exit() -> None:
    raise GracefulExit()


def _query_flag(request: web.Request, name: str, *, default: bool) -> bool:
    raw_value = request.query.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _query_limit(request: web.Request, *, default: int = 100, maximum: int = 200) -> int:
    try:
        parsed = int(request.query.get("limit", str(default)))
    except ValueError:
        return default
    return min(max(parsed, 1), maximum)


def _schedule_graceful_exit() -> None:
    asyncio.get_running_loop().call_soon(_raise_graceful_exit)


def _runtime_base_env(
    *,
    object_memory_dsn: str,
    agent_memory_dsn: str,
    knowledge_dsn: str,
    planner_catalog_dsn: str,
    webrtc_config: WebRTCServiceConfig | None,
) -> dict[str, str]:
    env = dict(os.environ)
    normalized_object_memory_dsn = str(object_memory_dsn or "").strip()
    normalized_agent_memory_dsn = str(agent_memory_dsn or "").strip() or normalized_object_memory_dsn
    normalized_knowledge_dsn = str(knowledge_dsn or "").strip()
    normalized_planner_catalog_dsn = str(planner_catalog_dsn or "").strip()

    if normalized_object_memory_dsn:
        env["AURA_OBJECT_MEMORY_DSN"] = normalized_object_memory_dsn
    if normalized_agent_memory_dsn:
        env["AURA_AGENT_MEMORY_DSN"] = normalized_agent_memory_dsn
    if normalized_knowledge_dsn:
        env["AURA_KNOWLEDGE_DSN"] = normalized_knowledge_dsn
    if normalized_planner_catalog_dsn:
        env["AURA_PLANNER_CATALOG_DSN"] = normalized_planner_catalog_dsn

    memory_user_id = ""
    if webrtc_config is not None:
        memory_user_id = str(webrtc_config.object_memory_user_id or "").strip()
    if memory_user_id:
        env["AURA_MEMORY_USER_ID"] = memory_user_id
    if webrtc_config is not None and bool(getattr(webrtc_config, "object_memory_auto_migrate", False)):
        env["AURA_OBJECT_MEMORY_AUTO_MIGRATE"] = "1"

    return env


def create_app(
    *,
    root_dir: str,
    api_base_url: str,
    dev_origin: str,
    inference_system_url: str,
    reasoning_system_url: str,
    navigation_system_url: str,
    control_runtime_url: str,
    runtime_url: str = "",
    webrtc_proxy_base: str = "",
    runtime_service: RuntimeService | None = None,
    webrtc_service: WebRTCService | None = None,
    webrtc_config: WebRTCServiceConfig | None = None,
    knowledge_dsn: str = "",
    planner_catalog_dsn: str = "",
    object_memory_dsn: str = "",
    agent_memory_dsn: str = "",
    knowledge_runtime: KnowledgeRuntimeHandle | None = None,
    agent_memory_runtime: HumanoidMemoryRuntimeHandle | None = None,
    planner_catalog_runtime: PlannerCatalogRuntimeHandle | None = None,
    shutdown_scheduler: Callable[[], None] | None = None,
) -> web.Application:
    app = web.Application()
    resolved_runtime_url = str(runtime_url).rstrip("/")
    runtime_owned = resolved_runtime_url == ""
    app[ROOT_DIR] = str(root_dir)
    app[API_BASE_URL] = str(api_base_url).rstrip("/")
    app[DEV_ORIGIN] = str(dev_origin)
    app[RUNTIME_URL] = resolved_runtime_url
    app[RUNTIME_OWNED] = runtime_owned
    app[RUNTIME_SERVICE] = (
        runtime_service
        if runtime_service is not None
        else (
            RuntimeService(
                Path(root_dir),
                base_env=_runtime_base_env(
                    object_memory_dsn=object_memory_dsn,
                    agent_memory_dsn=agent_memory_dsn,
                    knowledge_dsn=knowledge_dsn,
                    planner_catalog_dsn=planner_catalog_dsn,
                    webrtc_config=webrtc_config,
                ),
            )
            if runtime_owned
            else None
        )
    )
    app[INFERENCE_SYSTEM_URL] = str(inference_system_url).rstrip("/")
    app[REASONING_SYSTEM_URL] = str(reasoning_system_url).rstrip("/")
    app[NAVIGATION_SYSTEM_URL] = str(navigation_system_url).rstrip("/")
    app[CONTROL_RUNTIME_URL] = str(control_runtime_url).rstrip("/")
    app[WEBRTC_PROXY_BASE] = str(webrtc_proxy_base).rstrip("/")
    app[WEBRTC_SERVICE] = (
        webrtc_service
        if webrtc_service is not None
        else (None if app[WEBRTC_PROXY_BASE] else WebRTCService(config=webrtc_config))
    )
    app[AGENT_MEMORY_RUNTIME] = (
        agent_memory_runtime
        if agent_memory_runtime is not None
        else create_humanoid_memory_runtime(
            dsn=str(agent_memory_dsn or "").strip(),
            object_memory_dsn=str(object_memory_dsn or "").strip(),
        )
    )
    app[KNOWLEDGE_RUNTIME] = (
        knowledge_runtime
        if knowledge_runtime is not None
        else create_knowledge_runtime(
            dsn=str(knowledge_dsn or "").strip(),
            object_memory_dsn=str(object_memory_dsn or "").strip(),
        )
    )
    app[PLANNER_CATALOG_RUNTIME] = (
        planner_catalog_runtime
        if planner_catalog_runtime is not None
        else create_planner_catalog_runtime(
            dsn=str(planner_catalog_dsn or "").strip(),
            knowledge_dsn=str(knowledge_dsn or "").strip(),
            object_memory_dsn=str(object_memory_dsn or "").strip(),
        )
    )
    app[SSE] = SseBroadcaster()
    app[HTTP] = None
    app[RUNTIME_SUBMIT_TASKS] = set()
    session_manager = DashboardSessionManager(app)
    app[SESSION_MANAGER] = session_manager
    schedule_shutdown = shutdown_scheduler or _schedule_graceful_exit

    async def broadcast_loop():
        while True:
            state = await session_manager.build_state()
            await app[SSE].publish_state(state)
            await asyncio.sleep(1.0)

    async def on_startup(_app: web.Application):
        app[HTTP] = ClientSession()
        if app[WEBRTC_SERVICE] is not None:
            await app[WEBRTC_SERVICE].start()
        app[BROADCAST_TASK] = asyncio.create_task(broadcast_loop())

    async def on_cleanup(_app: web.Application):
        if app[BROADCAST_TASK] is not None:
            app[BROADCAST_TASK].cancel()
            try:
                await app[BROADCAST_TASK]
            except BaseException:
                pass
        if app[RUNTIME_SUBMIT_TASKS]:
            pending_submits = list(app[RUNTIME_SUBMIT_TASKS])
            for task in pending_submits:
                task.cancel()
            await asyncio.gather(*pending_submits, return_exceptions=True)
        if app[WEBRTC_SERVICE] is not None:
            await app[WEBRTC_SERVICE].close()
        if app[RUNTIME_OWNED] and app[RUNTIME_SERVICE] is not None:
            await asyncio.to_thread(app[RUNTIME_SERVICE].stop_session)
        if app[HTTP] is not None:
            await app[HTTP].close()

    async def submit_reasoning_request(payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        if app[HTTP] is None:
            raise web.HTTPServiceUnavailable(reason="backend http client unavailable")
        try:
            async with app[HTTP].post(f"{app[REASONING_SYSTEM_URL]}/reasoning/respond", json=payload) as response:
                reasoning_payload = await response.json()
        except Exception as exc:  # noqa: BLE001
            raise web.HTTPBadGateway(reason=f"reasoning system unavailable: {type(exc).__name__}: {exc}") from exc
        if not isinstance(reasoning_payload, dict):
            raise web.HTTPBadGateway(reason="reasoning system returned non-object payload")
        return response.status, reasoning_payload

    async def bootstrap(_request: web.Request) -> web.Response:
        payload = build_bootstrap_data(
            api_base_url=app[API_BASE_URL],
            dev_origin=app[DEV_ORIGIN],
            webrtc_base_path=f"{app[API_BASE_URL]}/api/webrtc",
        )
        return web.json_response(payload)

    async def state(_request: web.Request) -> web.Response:
        return web.json_response(await session_manager.build_state())

    async def events(request: web.Request) -> web.StreamResponse:
        return await app[SSE].subscribe(request)

    async def logs(request: web.Request) -> web.Response:
        state_payload = await session_manager.build_state()
        try:
            limit = max(1, int(request.query.get("limit", "80")))
        except ValueError:
            limit = 80
        logs_payload = list(state_payload.get("logs", []))
        return web.json_response({"logs": logs_payload[-limit:]})

    async def runtime_context_summary(request: web.Request) -> web.Response:
        payload = await session_manager.build_runtime_context_summary(
            force_refresh=_query_flag(request, "refresh", default=True),
            persist=_query_flag(request, "persist", default=True),
        )
        return web.json_response(payload)

    async def session_start(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        try:
            state_payload = await session_manager.start_session(payload)
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        status = 200 if bool(state_payload.get("session", {}).get("active")) else 503
        return web.json_response(state_payload, status=status)

    async def session_stop(_request: web.Request) -> web.Response:
        return web.json_response(await session_manager.stop_session())

    async def system_shutdown(_request: web.Request) -> web.Response:
        await session_manager.stop_session()
        schedule_shutdown()
        return web.json_response({"ok": True, "accepted": True})

    async def _proxy_reasoning_payload(payload: dict[str, object]) -> web.Response:
        utterance = " ".join(str(payload.get("utterance", payload.get("instruction", ""))).strip().split())
        if utterance == "":
            raise web.HTTPBadRequest(reason="utterance is required")
        language = str(payload.get("language", "auto")).strip() or "auto"
        interrupt_current_task = bool(payload.get("interrupt_current_task", False))
        scene_preset = (
            str(session_manager.session_config.get("scenePreset"))
            if isinstance(session_manager.session_config, dict) and session_manager.session_config.get("scenePreset")
            else ""
        )
        conversation_id = session_manager.ensure_conversation_id()
        reasoning_payload = {
            "utterance": utterance,
            "language": language,
            "conversation_id": conversation_id,
            "interrupt_current_task": interrupt_current_task,
        }
        if scene_preset:
            reasoning_payload["scene_preset"] = scene_preset
        status_code, response_payload = await submit_reasoning_request(reasoning_payload)
        if bool(response_payload.get("ok")):
            route = str(response_payload.get("route") or "unknown")
            session_manager.record_agent_response(response_payload)
            session_manager.record_event(f"reasoning route={route}: {utterance}")
        else:
            session_manager.record_event(
                f"reasoning request failed: {response_payload.get('error')}",
                level="error",
            )
        return web.json_response(response_payload, status=status_code)

    async def runtime_reason(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        return await _proxy_reasoning_payload(payload)

    async def runtime_cancel(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        async with app[HTTP].post(f"{app[REASONING_SYSTEM_URL]}/reasoning/cancel", json=payload) as response:
            return web.json_response(await response.json(), status=response.status)

    async def occupancy_current(request: web.Request) -> web.Response:
        scene_preset = request.query.get("scenePreset", "warehouse")
        return web.json_response(build_occupancy_payload(scene_preset))

    def _object_memory_runtime_handle():
        webrtc_service = app.get(WEBRTC_SERVICE)
        sink = getattr(webrtc_service, "object_memory_sink", None)
        return getattr(sink, "runtime", None)

    def _object_memory_health_payload() -> dict[str, object]:
        webrtc_service = app.get(WEBRTC_SERVICE)
        sink = getattr(webrtc_service, "object_memory_sink", None)
        if sink is None or not hasattr(sink, "health_snapshot"):
            return {"configured": False, "enabled": False, "available": False}
        snapshot = sink.health_snapshot()
        return snapshot if isinstance(snapshot, dict) else {"configured": False, "enabled": False, "available": False}

    def _json_safe(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if hasattr(value, "isoformat"):
            return value.isoformat()
        return str(value)

    def _iso_or_none(value: object) -> str | None:
        return value.isoformat() if hasattr(value, "isoformat") else None

    def _object_memory_entry_payload(entry) -> dict[str, object]:
        return {
            "objectId": entry.object_id,
            "userId": entry.user_id,
            "canonicalClass": entry.canonical_class,
            "roomId": entry.room_id,
            "sceneScope": entry.scene_scope,
            "status": entry.status,
            "firstSeenAt": entry.first_seen_at.isoformat(),
            "lastSeenAt": entry.last_seen_at.isoformat(),
            "observationCount": entry.observation_count,
            "lastSourceId": entry.last_source_id,
            "lastSessionId": entry.last_session_id,
            "lastBboxXyxyNorm": [float(value) for value in entry.last_bbox_xyxy_norm],
            "lastBoxArea": float(entry.last_box_area),
            "lastAspectRatio": float(entry.last_aspect_ratio),
            "lastDetectorConf": float(entry.last_detector_conf),
            "appearanceCount": int(entry.appearance_count),
            "dedupeConfidence": float(entry.dedupe_confidence),
            "worldPoseXyz": (
                None
                if entry.world_pose_xyz is None
                else [float(value) for value in entry.world_pose_xyz]
            ),
            "worldPoseObservedAt": _iso_or_none(entry.world_pose_observed_at),
            "metadata": _json_safe(entry.metadata),
            "createdAt": entry.created_at.isoformat(),
            "updatedAt": entry.updated_at.isoformat(),
        }

    async def object_memory_list_objects(request: web.Request) -> web.Response:
        runtime_handle = _object_memory_runtime_handle()
        health = _object_memory_health_payload()
        if runtime_handle is None:
            return web.json_response(
                {
                    "ok": True,
                    "enabled": False,
                    "available": False,
                    "configured": bool(health.get("configured", False)),
                    "userId": None,
                    "degradedReason": None,
                    "activeObjectCount": 0,
                    "returnedCount": 0,
                    "limit": _query_limit(request),
                    "filters": {},
                    "health": health,
                    "objects": [],
                }
            )

        if runtime_handle.service is None:
            return web.json_response(
                {
                    "ok": True,
                    "enabled": bool(runtime_handle.enabled),
                    "available": False,
                    "configured": bool(health.get("configured", runtime_handle.enabled)),
                    "userId": runtime_handle.user_id,
                    "degradedReason": runtime_handle.degraded_reason,
                    "activeObjectCount": 0,
                    "returnedCount": 0,
                    "limit": _query_limit(request),
                    "filters": {},
                    "health": health,
                    "objects": [],
                }
            )

        status_filter = request.query.get("status", "active")
        statuses = tuple(part.strip() for part in status_filter.split(",") if part.strip())
        if not statuses:
            statuses = ("active",)
        invalid_statuses = [status for status in statuses if status not in OBJECT_MEMORY_STATUSES]
        if invalid_statuses:
            raise web.HTTPBadRequest(reason=f"unsupported object memory status: {', '.join(invalid_statuses)}")

        class_name = " ".join(str(request.query.get("className", "")).strip().split()) or None
        room_id = " ".join(str(request.query.get("roomId", "")).strip().split()) or None
        scene_scope = " ".join(str(request.query.get("sceneScope", "")).strip().split()) or None
        limit = _query_limit(request)

        try:
            repository = runtime_handle.service.repository
            rows = await asyncio.to_thread(
                repository.list_object_entries,
                runtime_handle.user_id,
                statuses=statuses,
                class_name=class_name,
                room_id=room_id,
                scene_scope=scene_scope,
                top_k=limit,
            )
            active_count = await asyncio.to_thread(
                repository.count_object_entries,
                runtime_handle.user_id,
                statuses=("active",),
            )
        except Exception as exc:  # noqa: BLE001
            runtime_handle.degraded_reason = f"{type(exc).__name__}: {exc}"
            raise web.HTTPServiceUnavailable(reason=f"object memory unavailable: {type(exc).__name__}: {exc}") from exc

        return web.json_response(
            {
                "ok": True,
                "enabled": bool(runtime_handle.enabled),
                "available": bool(runtime_handle.available),
                "configured": bool(health.get("configured", runtime_handle.enabled)),
                "userId": runtime_handle.user_id,
                "degradedReason": runtime_handle.degraded_reason,
                "activeObjectCount": int(active_count),
                "returnedCount": len(rows),
                "limit": limit,
                "filters": {
                    "statuses": list(statuses),
                    "className": class_name,
                    "roomId": room_id,
                    "sceneScope": scene_scope,
                },
                "health": health,
                "objects": [_object_memory_entry_payload(row) for row in rows],
            }
        )

    def _agent_memory_runtime_handle() -> HumanoidMemoryRuntimeHandle:
        runtime_handle = app.get(AGENT_MEMORY_RUNTIME)
        if runtime_handle is None:
            raise web.HTTPServiceUnavailable(reason="agent memory runtime unavailable")
        return runtime_handle

    def _agent_memory_status_payload() -> dict[str, object]:
        runtime_handle = app.get(AGENT_MEMORY_RUNTIME)
        snapshot = None if runtime_handle is None else runtime_handle.status_snapshot()
        if snapshot is None:
            return {
                "ok": True,
                "enabled": False,
                "available": False,
                "coreBlockCount": 0,
                "archivalPassageCount": 0,
                "archivalTags": [],
                "degradedReason": None,
            }
        return {
            "ok": True,
            "enabled": snapshot.enabled,
            "available": snapshot.available,
            "coreBlockCount": snapshot.core_block_count,
            "archivalPassageCount": snapshot.archival_passage_count,
            "archivalTags": list(snapshot.archival_tags),
            "degradedReason": snapshot.degraded_reason,
        }

    def _agent_memory_block_payload(block) -> dict[str, object]:
        return {
            "label": block.label,
            "description": block.description,
            "value": block.value,
            "limit": block.limit,
            "readOnly": block.read_only,
            "scope": block.scope,
            "version": block.version,
            "updatedAt": block.updated_at.isoformat(),
        }

    def _agent_memory_passage_payload(passage) -> dict[str, object]:
        return {
            "passageId": passage.passage_id,
            "content": passage.content,
            "tags": list(passage.tags),
            "sceneScope": passage.scene_scope,
            "metadata": dict(passage.metadata),
            "createdAt": passage.created_at.isoformat(),
            "updatedAt": passage.updated_at.isoformat(),
            "rank": passage.rank,
        }

    def _tags_from_query(request: web.Request) -> tuple[str, ...]:
        values: list[str] = []
        for key in ("tag", "tags"):
            for raw_value in request.query.getall(key, []):
                values.extend(part.strip() for part in str(raw_value).split(",") if part.strip())
        return tuple(values)

    def _tags_from_payload(raw_value: object) -> tuple[str, ...]:
        if isinstance(raw_value, str):
            return tuple(part.strip() for part in raw_value.split(",") if part.strip())
        if isinstance(raw_value, list):
            return tuple(str(part).strip() for part in raw_value if str(part).strip())
        if isinstance(raw_value, tuple):
            return tuple(str(part).strip() for part in raw_value if str(part).strip())
        return ()

    async def agent_memory_status(_request: web.Request) -> web.Response:
        return web.json_response(_agent_memory_status_payload())

    async def agent_memory_list_blocks(_request: web.Request) -> web.Response:
        runtime_handle = _agent_memory_runtime_handle()
        try:
            blocks = await asyncio.to_thread(runtime_handle.list_blocks)
        except RuntimeError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            runtime_handle.degraded_reason = f"{type(exc).__name__}: {exc}"
            raise web.HTTPServiceUnavailable(reason=f"agent memory unavailable: {type(exc).__name__}: {exc}") from exc
        return web.json_response(
            {
                "ok": True,
                "blocks": [_agent_memory_block_payload(block) for block in blocks],
            }
        )

    async def agent_memory_update_block(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        runtime_handle = _agent_memory_runtime_handle()
        try:
            block = await asyncio.to_thread(
                runtime_handle.update_block,
                request.match_info["label"],
                AgentMemoryBlockInput(
                    value=str(payload.get("value", "")),
                    description=payload.get("description") if payload.get("description") is not None else None,
                    limit=payload.get("limit") if payload.get("limit") is not None else None,
                    read_only=payload.get("readOnly", payload.get("read_only")),
                    scope=payload.get("scope") if payload.get("scope") is not None else None,
                ),
            )
        except PermissionError as exc:
            raise web.HTTPForbidden(reason=str(exc)) from exc
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except RuntimeError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        return web.json_response({"ok": True, "block": _agent_memory_block_payload(block)})

    async def agent_memory_list_passages(request: web.Request) -> web.Response:
        runtime_handle = _agent_memory_runtime_handle()
        query = " ".join(str(request.query.get("query", "")).strip().split())
        scene_scope = " ".join(str(request.query.get("sceneScope", request.query.get("scene_scope", ""))).strip().split()) or None
        tags = _tags_from_query(request)
        tag_match_mode = str(request.query.get("tagMatchMode", request.query.get("tag_match_mode", "any"))).strip() or "any"
        limit = _query_limit(request, default=50, maximum=200)
        try:
            if query:
                passages = await asyncio.to_thread(
                    runtime_handle.search_passages,
                    query,
                    tags=tags or None,
                    tag_match_mode=tag_match_mode,
                    scene_scope=scene_scope,
                    top_k=limit,
                )
            else:
                passages = await asyncio.to_thread(
                    runtime_handle.list_passages,
                    tags=tags or None,
                    tag_match_mode=tag_match_mode,
                    scene_scope=scene_scope,
                    top_k=limit,
                )
        except RuntimeError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            runtime_handle.degraded_reason = f"{type(exc).__name__}: {exc}"
            raise web.HTTPServiceUnavailable(reason=f"agent memory unavailable: {type(exc).__name__}: {exc}") from exc
        return web.json_response(
            {
                "ok": True,
                "passages": [_agent_memory_passage_payload(passage) for passage in passages],
                "returnedCount": len(passages),
                "limit": limit,
            }
        )

    async def agent_memory_create_passage(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        metadata = payload.get("metadata", {})
        runtime_handle = _agent_memory_runtime_handle()
        try:
            passage = await asyncio.to_thread(
                runtime_handle.insert_passage,
                AgentMemoryPassageInput(
                    content=str(payload.get("content", "")),
                    tags=_tags_from_payload(payload.get("tags")),
                    scene_scope=payload.get("sceneScope", payload.get("scene_scope")),
                    metadata=metadata if isinstance(metadata, dict) else {},
                ),
            )
        except (TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except RuntimeError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        return web.json_response({"ok": True, "passage": _agent_memory_passage_payload(passage)})

    def _knowledge_service_or_503():
        runtime_handle = app.get(KNOWLEDGE_RUNTIME)
        if runtime_handle is None or runtime_handle.service is None:
            raise web.HTTPServiceUnavailable(reason="knowledge service unavailable")
        return runtime_handle.service

    def _knowledge_status_payload() -> dict[str, object]:
        runtime_handle = app.get(KNOWLEDGE_RUNTIME)
        if runtime_handle is None:
            snapshot = None
        else:
            snapshot = runtime_handle.status_snapshot()
        if snapshot is None:
            return {
                "ok": True,
                "knowledgeEnabled": False,
                "publishedDocumentCount": 0,
                "activeHardRuleCount": 0,
                "lexiconEntryCount": 0,
                "lastKnowledgeRefreshOk": None,
                "lastAppliedRuleIds": [],
                "degradedReason": None,
            }
        return {
            "ok": True,
            "knowledgeEnabled": snapshot.knowledge_enabled,
            "publishedDocumentCount": snapshot.published_document_count,
            "activeHardRuleCount": snapshot.active_hard_rule_count,
            "lexiconEntryCount": snapshot.lexicon_entry_count,
            "lastKnowledgeRefreshOk": snapshot.last_refresh_ok,
            "lastAppliedRuleIds": snapshot.last_applied_rule_ids,
            "degradedReason": snapshot.degraded_reason,
        }

    def _knowledge_document_payload(document) -> dict[str, object]:
        return {
            "documentId": document.document_id,
            "title": document.title,
            "scopeKind": document.scope_kind,
            "scopeValue": document.scope_value,
            "status": document.status,
            "contentHash": document.content_hash,
            "version": document.version,
            "createdAt": document.created_at.isoformat(),
            "updatedAt": document.updated_at.isoformat(),
            "publishedAt": None if document.published_at is None else document.published_at.isoformat(),
        }

    def _planner_catalog_runtime_handle() -> PlannerCatalogRuntimeHandle:
        runtime_handle = app.get(PLANNER_CATALOG_RUNTIME)
        if runtime_handle is None:
            raise web.HTTPServiceUnavailable(reason="planner catalog runtime unavailable")
        return runtime_handle

    def _planner_catalog_response(snapshot, status) -> dict[str, object]:
        return {
            "ok": True,
            "catalog": {
                "intents": [
                    {
                        "intentId": intent.intent_id,
                        "intentKey": intent.intent_key,
                        "displayName": intent.display_name,
                        "description": intent.description,
                        "createdAt": intent.created_at.isoformat(),
                        "updatedAt": intent.updated_at.isoformat(),
                        "subgoals": [
                            {
                                "templateId": template.template_id,
                                "intentId": template.intent_id,
                                "sequenceNo": template.sequence_no,
                                "subgoalType": template.subgoal_type,
                                "activationCondition": template.activation_condition,
                                "createdAt": template.created_at.isoformat(),
                                "updatedAt": template.updated_at.isoformat(),
                            }
                            for template in intent.subgoals
                        ],
                    }
                    for intent in snapshot.intents
                ],
                "supportedIntentKeys": list(EXECUTION_INTENT_KEYS),
                "supportedSubgoalTypes": list(SUPPORTED_SUBGOAL_TYPES),
                "supportedActivationConditions": list(SUPPORTED_ACTIVATION_CONDITIONS),
                "supportedIntents": [
                    {
                        "intentKey": spec.intent_key,
                        "displayName": spec.display_name,
                        "description": spec.description,
                    }
                    for spec in PLANNER_INTENT_SPECS
                ],
            },
            "status": {
                "enabled": status.enabled,
                "available": status.available,
                "writable": status.writable,
                "source": status.source,
                "degradedReason": status.degraded_reason,
                "lastRefreshOk": status.last_refresh_ok,
                "activeIntentCount": status.active_intent_count,
                "activeSubgoalTemplateCount": status.active_subgoal_template_count,
            },
        }

    async def planner_catalog_get(_request: web.Request) -> web.Response:
        runtime_handle = _planner_catalog_runtime_handle()
        snapshot, status = await asyncio.to_thread(runtime_handle.snapshot_and_status)
        return web.json_response(_planner_catalog_response(snapshot, status))

    async def planner_catalog_create_intent(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        runtime_handle = _planner_catalog_runtime_handle()
        intent_key = str(payload.get("intent_key", payload.get("intentKey", ""))).strip()
        try:
            snapshot = await asyncio.to_thread(runtime_handle.create_intent, intent_key)
            status = runtime_handle.status_snapshot(snapshot)
        except PlannerCatalogValidationError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except PlannerCatalogConflictError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except PlannerCatalogUnavailableError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise web.HTTPServiceUnavailable(reason=f"planner catalog unavailable: {type(exc).__name__}: {exc}") from exc
        return web.json_response(_planner_catalog_response(snapshot, status))

    async def planner_catalog_delete_intent(request: web.Request) -> web.Response:
        runtime_handle = _planner_catalog_runtime_handle()
        try:
            snapshot = await asyncio.to_thread(runtime_handle.delete_intent, request.match_info["intent_id"])
            status = runtime_handle.status_snapshot(snapshot)
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        except PlannerCatalogValidationError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except PlannerCatalogConflictError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except PlannerCatalogUnavailableError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise web.HTTPServiceUnavailable(reason=f"planner catalog unavailable: {type(exc).__name__}: {exc}") from exc
        return web.json_response(_planner_catalog_response(snapshot, status))

    async def planner_catalog_create_subgoal(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        runtime_handle = _planner_catalog_runtime_handle()
        intent_id = str(payload.get("intent_id", payload.get("intentId", ""))).strip()
        sequence_no = payload.get("sequence_no", payload.get("sequenceNo"))
        subgoal_type = str(payload.get("subgoal_type", payload.get("subgoalType", ""))).strip()
        activation_condition = str(
            payload.get("activation_condition", payload.get("activationCondition", ""))
        ).strip()
        try:
            snapshot = await asyncio.to_thread(
                runtime_handle.create_subgoal_template,
                intent_id=intent_id,
                sequence_no=int(sequence_no),
                subgoal_type=subgoal_type,
                activation_condition=activation_condition,
            )
            status = runtime_handle.status_snapshot(snapshot)
        except (TypeError, ValueError) as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        except PlannerCatalogValidationError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except PlannerCatalogConflictError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except PlannerCatalogUnavailableError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise web.HTTPServiceUnavailable(reason=f"planner catalog unavailable: {type(exc).__name__}: {exc}") from exc
        return web.json_response(_planner_catalog_response(snapshot, status))

    async def planner_catalog_delete_subgoal(request: web.Request) -> web.Response:
        runtime_handle = _planner_catalog_runtime_handle()
        try:
            snapshot = await asyncio.to_thread(runtime_handle.delete_subgoal_template, request.match_info["template_id"])
            status = runtime_handle.status_snapshot(snapshot)
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        except PlannerCatalogValidationError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except PlannerCatalogConflictError as exc:
            raise web.HTTPConflict(reason=str(exc)) from exc
        except PlannerCatalogUnavailableError as exc:
            raise web.HTTPServiceUnavailable(reason=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise web.HTTPServiceUnavailable(reason=f"planner catalog unavailable: {type(exc).__name__}: {exc}") from exc
        return web.json_response(_planner_catalog_response(snapshot, status))

    async def knowledge_create_document(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        service = _knowledge_service_or_503()
        try:
            document = await asyncio.to_thread(
                service.register_document,
                KnowledgeDocumentInput(
                    title=str(payload.get("title", "")),
                    body_markdown=str(payload.get("body_markdown", "")),
                    scope_kind=str(payload.get("scope_kind", "global")),
                    scope_value=payload.get("scope_value"),
                    publish=bool(payload.get("publish", False)),
                ),
            )
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        return web.json_response({"ok": True, "document": _knowledge_document_payload(document)})

    async def knowledge_update_document(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        service = _knowledge_service_or_503()
        try:
            document = await asyncio.to_thread(
                service.register_document,
                KnowledgeDocumentInput(
                    title=str(payload.get("title", "")),
                    body_markdown=str(payload.get("body_markdown", "")),
                    scope_kind=str(payload.get("scope_kind", "global")),
                    scope_value=payload.get("scope_value"),
                    publish=bool(payload.get("publish", False)),
                ),
                document_id=request.match_info["document_id"],
            )
        except ValueError as exc:
            raise web.HTTPBadRequest(reason=str(exc)) from exc
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        return web.json_response({"ok": True, "document": _knowledge_document_payload(document)})

    async def knowledge_publish_document(request: web.Request) -> web.Response:
        service = _knowledge_service_or_503()
        try:
            document = await asyncio.to_thread(service.publish_document, request.match_info["document_id"])
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        return web.json_response({"ok": True, "document": _knowledge_document_payload(document)})

    async def knowledge_archive_document(request: web.Request) -> web.Response:
        service = _knowledge_service_or_503()
        try:
            document = await asyncio.to_thread(service.archive_document, request.match_info["document_id"])
        except KeyError as exc:
            raise web.HTTPNotFound(reason=str(exc)) from exc
        return web.json_response({"ok": True, "document": _knowledge_document_payload(document)})

    async def knowledge_list_documents(request: web.Request) -> web.Response:
        service = _knowledge_service_or_503()
        status_filter = request.query.get("status", "")
        statuses = tuple(part.strip() for part in status_filter.split(",") if part.strip()) or None
        documents = await asyncio.to_thread(service.list_documents, statuses=statuses)
        return web.json_response(
            {
                "ok": True,
                "documents": [_knowledge_document_payload(document) for document in documents],
            }
        )

    async def knowledge_status(_request: web.Request) -> web.Response:
        return web.json_response(_knowledge_status_payload())

    async def webrtc_config(_request: web.Request) -> web.Response:
        return web.json_response(await get_config(app))

    async def webrtc_offer(request: web.Request) -> web.Response:
        payload = await _json_body(request)
        return await proxy_offer(app, payload)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/api/bootstrap", bootstrap)
    app.router.add_get("/api/state", state)
    app.router.add_get("/api/events", events)
    app.router.add_get("/api/logs", logs)
    app.router.add_get("/api/runtime/context-summary", runtime_context_summary)
    app.router.add_post("/api/session/start", session_start)
    app.router.add_post("/api/session/stop", session_stop)
    app.router.add_post("/api/system/shutdown", system_shutdown)
    app.router.add_post("/api/runtime/reason", runtime_reason)
    app.router.add_post("/api/runtime/cancel", runtime_cancel)
    app.router.add_get("/api/occupancy/current", occupancy_current)
    app.router.add_get("/api/occupancy/image", handle_image)
    app.router.add_post("/api/knowledge/documents", knowledge_create_document)
    app.router.add_put("/api/knowledge/documents/{document_id}", knowledge_update_document)
    app.router.add_post("/api/knowledge/documents/{document_id}/publish", knowledge_publish_document)
    app.router.add_post("/api/knowledge/documents/{document_id}/archive", knowledge_archive_document)
    app.router.add_get("/api/knowledge/documents", knowledge_list_documents)
    app.router.add_get("/api/knowledge/status", knowledge_status)
    app.router.add_get("/api/memory/status", agent_memory_status)
    app.router.add_get("/api/memory/blocks", agent_memory_list_blocks)
    app.router.add_put("/api/memory/blocks/{label}", agent_memory_update_block)
    app.router.add_get("/api/memory/passages", agent_memory_list_passages)
    app.router.add_post("/api/memory/passages", agent_memory_create_passage)
    app.router.add_get("/api/object-memory/objects", object_memory_list_objects)
    app.router.add_get("/api/planner/catalog", planner_catalog_get)
    app.router.add_post("/api/planner/intents", planner_catalog_create_intent)
    app.router.add_delete("/api/planner/intents/{intent_id}", planner_catalog_delete_intent)
    app.router.add_post("/api/planner/subgoals", planner_catalog_create_subgoal)
    app.router.add_delete("/api/planner/subgoals/{template_id}", planner_catalog_delete_subgoal)
    app.router.add_get("/api/webrtc/config", webrtc_config)
    app.router.add_post("/api/webrtc/offer", webrtc_offer)
    return app
