"""Typed aiohttp application keys for backend state."""

from __future__ import annotations

import asyncio
from aiohttp import ClientSession, web

from backend.sse import SseBroadcaster
from backend.webrtc import WebRTCService
from runtime.service import RuntimeService
from systems.memory.agent_memory_runtime import HumanoidMemoryRuntimeHandle
from systems.memory.knowledge_runtime import KnowledgeRuntimeHandle
from systems.reasoning.planner_catalog_runtime import PlannerCatalogRuntimeHandle


ROOT_DIR = web.AppKey("root_dir", str)
API_BASE_URL = web.AppKey("api_base_url", str)
DEV_ORIGIN = web.AppKey("dev_origin", str)
RUNTIME_URL = web.AppKey("runtime_url", str)
RUNTIME_OWNED = web.AppKey("runtime_owned", bool)
RUNTIME_SERVICE = web.AppKey("runtime_service", RuntimeService | None)
INFERENCE_SYSTEM_URL = web.AppKey("inference_system_url", str)
REASONING_SYSTEM_URL = web.AppKey("reasoning_system_url", str)
NAVIGATION_SYSTEM_URL = web.AppKey("navigation_system_url", str)
CONTROL_RUNTIME_URL = web.AppKey("control_runtime_url", str)
WEBRTC_PROXY_BASE = web.AppKey("webrtc_proxy_base", str)
WEBRTC_SERVICE = web.AppKey("webrtc_service", WebRTCService | None)
AGENT_MEMORY_RUNTIME = web.AppKey("agent_memory_runtime", HumanoidMemoryRuntimeHandle | None)
KNOWLEDGE_RUNTIME = web.AppKey("knowledge_runtime", KnowledgeRuntimeHandle | None)
PLANNER_CATALOG_RUNTIME = web.AppKey("planner_catalog_runtime", PlannerCatalogRuntimeHandle | None)
SSE = web.AppKey("sse", SseBroadcaster)
SESSION_CONFIG = web.AppKey("session_config", dict | None)
HTTP = web.AppKey("http", ClientSession | None)
SESSION_MANAGER = web.AppKey("session_manager", object)
BROADCAST_TASK = web.AppKey("broadcast_task", asyncio.Task | None)
RUNTIME_SUBMIT_TASKS = web.AppKey("runtime_submit_tasks", set[asyncio.Task])
