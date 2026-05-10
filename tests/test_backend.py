from __future__ import annotations

import asyncio
import json
from pathlib import Path
import tempfile

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from backend.app_keys import SESSION_MANAGER
from systems.control.runtime_control_api import RuntimeControlApiServer
import backend.app as backend_app
from backend.app import create_app
from backend.webrtc.config import WebRTCServiceConfig
from backend.webrtc.object_memory import ObjectMemoryFrameSink, ObjectMemorySinkConfig
from systems.memory.agent_memory_repository import InMemoryAgentMemoryRepository
from systems.memory.agent_memory_runtime import HumanoidMemoryRuntimeHandle
from systems.memory.agent_memory_service import AgentMemoryService
from systems.memory.knowledge_models import KnowledgeDocumentInput
from systems.memory.knowledge_repository import InMemoryKnowledgeRepository
from systems.memory.knowledge_runtime import KnowledgeRuntimeHandle
from systems.memory.knowledge_service import KnowledgeService
from systems.memory.object_memory_models import ObjectObservationInput
from systems.memory.object_memory_repository import InMemoryObjectMemoryRepository
from systems.memory.object_memory_runtime import ObjectMemoryRuntimeHandle
from systems.memory.object_memory_service import ObjectMemoryService
from systems.reasoning.planner_catalog_repository import InMemoryPlannerCatalogRepository
from systems.reasoning.planner_catalog_runtime import PlannerCatalogRuntimeHandle
from systems.reasoning.planner_catalog_service import PlannerCatalogService


async def _make_client(
    *,
    root_dir: str = "C:/Users/mango/project/AURA/system",
    runtime_url: str = "",
    reasoning_system_url: str = "http://127.0.0.1:17881",
    control_runtime_url: str = "http://127.0.0.1:8892",
    runtime_service=None,
    webrtc_service=None,
    knowledge_runtime=None,
    agent_memory_runtime=None,
    planner_catalog_runtime=None,
    shutdown_scheduler=None,
) -> TestClient:
    app = create_app(
        root_dir=root_dir,
        api_base_url="http://127.0.0.1:18095",
        dev_origin="http://127.0.0.1:5173",
        runtime_url=runtime_url,
        inference_system_url="http://127.0.0.1:15880",
        reasoning_system_url=reasoning_system_url,
        navigation_system_url="http://127.0.0.1:17882",
        control_runtime_url=control_runtime_url,
        runtime_service=runtime_service,
        webrtc_service=webrtc_service,
        knowledge_runtime=knowledge_runtime,
        agent_memory_runtime=agent_memory_runtime,
        planner_catalog_runtime=planner_catalog_runtime,
        shutdown_scheduler=shutdown_scheduler,
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


class FakeWebRTCService:
    def __init__(
        self,
        *,
        frame_state: dict[str, object] | None = None,
        frame_meta: dict[str, object] | None = None,
        health_snapshot: dict[str, object] | None = None,
        object_memory_sink: object | None = None,
    ) -> None:
        self.subscriber = type(
            "_Subscriber",
            (),
            {
                "build_state_snapshot": lambda _self: frame_state,
                "build_frame_meta": lambda _self: frame_meta,
            },
        )()
        self._health_snapshot = health_snapshot
        self.object_memory_sink = object_memory_sink

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    def public_config(self, *, enabled: bool) -> dict[str, object]:
        return {
            "iceServers": [],
            "enableDepthTrack": True,
            "transportMode": "webrtc" if enabled else "disabled",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc" if enabled else "disabled",
            "observeOnly": True,
            "peerModel": "single",
            "channelLabels": ["state", "telemetry"],
            "proxyMode": "internal",
        }

    async def accept_offer(self, offer_payload: dict[str, object]) -> dict[str, object]:
        return {
            "sdp": f"answer-for-{offer_payload.get('type', 'offer')}",
            "type": "answer",
            "sessionId": "peer-test",
        }

    def health_snapshot(self) -> dict[str, object]:
        if self._health_snapshot is not None:
            return dict(self._health_snapshot)
        return {
            "transport": "webrtc",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc",
            "frameAvailable": False,
            "streamStalled": False,
            "frameSeq": None,
            "frameId": None,
            "frameAgeMs": None,
            "lastGoodFrameAgeMs": None,
            "peerActive": False,
            "peerSessionId": None,
            "peerTrackRoles": [],
            "rgbAvailable": False,
            "depthAvailable": False,
            "source": "control_runtime",
            "image": {"width": 0, "height": 0},
            "dropCounters": {"shmOverwrite": 0},
            "transportHealth": {
                "control_endpoint": "tcp://127.0.0.1:18880",
                "telemetry_endpoint": "tcp://127.0.0.1:18881",
                "shm_name": "aura_viewer_shm_01",
                "decodeOk": 0,
                "decodeDrops": 0,
                "shmOverwriteDrops": 0,
                "staleTransitions": 0,
            },
        }


class StaticRuntimeService:
    def __init__(self, *, active: bool = False, config: dict[str, object] | None = None) -> None:
        self._session = {
            "active": active,
            "state": "running" if active else "inactive",
            "startedAt": 123.0 if active else None,
            "config": dict(config) if isinstance(config, dict) else None,
            "lastEvent": {"message": "runtime session running" if active else "runtime initialized"},
        }

    def state_payload(self, *, ok: bool = True):
        return {
            "ok": ok,
            "session": dict(self._session),
            "processes": [],
            "serviceEndpoints": {},
            "lastError": None,
        }

    def start_session(self, config: dict[str, object]):
        self._session = {
            "active": True,
            "state": "running",
            "startedAt": 123.0,
            "config": dict(config),
            "lastEvent": {"message": "runtime session running"},
        }
        return self.state_payload()

    def stop_session(self):
        self._session = {
            "active": False,
            "state": "inactive",
            "startedAt": None,
            "config": None,
            "lastEvent": {"message": "runtime session stopped"},
        }
        return self.state_payload()


def test_backend_bootstrap_and_degraded_state_contracts() -> None:
    async def scenario():
        client = await _make_client(webrtc_service=FakeWebRTCService())
        try:
            bootstrap = await client.get("/api/bootstrap")
            assert bootstrap.status == 200
            bootstrap_payload = await bootstrap.json()
            assert bootstrap_payload["apiBaseUrl"] == "http://127.0.0.1:18095"
            assert "scenePresets" in bootstrap_payload

            state = await client.get("/api/state")
            assert state.status == 200
            state_payload = await state.json()
            assert state_payload["session"]["config"] is None
            assert state_payload["processes"] == []
            assert state_payload["services"]["backend"]["status"] == "healthy"
            assert state_payload["services"]["runtime"]["status"] == "healthy"
            assert state_payload["services"]["runtime"]["health"]["ownedByBackend"] is True
            assert state_payload["services"]["controlRuntime"]["status"] == "inactive"
            assert state_payload["services"]["inferenceSystem"]["status"] == "inactive"
            assert state_payload["services"]["navigationSystem"]["status"] == "inactive"
            assert state_payload["services"]["reasoningSystem"]["status"] == "inactive"

            events = await client.get("/api/events")
            assert events.status == 200
            assert events.headers["Content-Type"].startswith("text/event-stream")

            occupancy = await client.get("/api/occupancy/current?scenePreset=warehouse")
            assert occupancy.status == 200
            occupancy_payload = await occupancy.json()
            assert occupancy_payload["available"] is False

            webrtc = await client.get("/api/webrtc/config")
            assert webrtc.status == 200
            webrtc_payload = await webrtc.json()
            assert "iceServers" in webrtc_payload
            assert webrtc_payload["transportMode"] == "disabled"
            assert webrtc_payload["mediaIngress"] == "zmq+shm"
            assert webrtc_payload["proxyMode"] == "internal"
            assert state_payload["transport"]["busHealth"]["control_endpoint"] == "tcp://127.0.0.1:18880"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_owned_runtime_inherits_memory_environment(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _RuntimeService:
        def __init__(self, repo_root: Path, *, base_env: dict[str, str] | None = None) -> None:
            captured["repo_root"] = Path(repo_root)
            captured["base_env"] = dict(base_env or {})

    monkeypatch.setattr(backend_app, "RuntimeService", _RuntimeService)

    app = create_app(
        root_dir=str(tmp_path),
        api_base_url="http://127.0.0.1:18095",
        dev_origin="http://127.0.0.1:5173",
        inference_system_url="http://127.0.0.1:15880",
        reasoning_system_url="http://127.0.0.1:17881",
        navigation_system_url="http://127.0.0.1:17882",
        control_runtime_url="http://127.0.0.1:8892",
        webrtc_service=FakeWebRTCService(),
        webrtc_config=WebRTCServiceConfig(object_memory_user_id="memory-user", object_memory_auto_migrate=True),
        object_memory_dsn="postgres://object-memory",
        knowledge_dsn="postgres://knowledge",
        planner_catalog_dsn="postgres://catalog",
        knowledge_runtime=KnowledgeRuntimeHandle(enabled=False),
        planner_catalog_runtime=PlannerCatalogRuntimeHandle(enabled=False),
    )

    assert captured["repo_root"] == tmp_path
    base_env = captured["base_env"]
    assert isinstance(base_env, dict)
    assert base_env["AURA_OBJECT_MEMORY_DSN"] == "postgres://object-memory"
    assert base_env["AURA_AGENT_MEMORY_DSN"] == "postgres://object-memory"
    assert base_env["AURA_OBJECT_MEMORY_AUTO_MIGRATE"] == "1"
    assert base_env["AURA_MEMORY_USER_ID"] == "memory-user"
    assert base_env["AURA_KNOWLEDGE_DSN"] == "postgres://knowledge"
    assert base_env["AURA_PLANNER_CATALOG_DSN"] == "postgres://catalog"
    assert app[backend_app.RUNTIME_SERVICE] is not None


def test_backend_runtime_context_summary_endpoint_persists_markdown_snapshot() -> None:
    async def scenario():
        with tempfile.TemporaryDirectory() as temp_dir:
            client = await _make_client(
                root_dir=temp_dir,
                runtime_service=StaticRuntimeService(
                    active=True,
                    config={"launchMode": "headless", "viewerEnabled": True, "memoryStore": True},
                ),
                webrtc_service=FakeWebRTCService(),
            )
            try:
                response = await client.get("/api/runtime/context-summary")
                assert response.status == 200
                payload = await response.json()
                assert payload["ok"] is True
                assert payload["persisted"] is True
                assert payload["persistError"] is None
                assert payload["path"].endswith("logs\\runtime\\runtime-context-summary.md") or payload["path"].endswith(
                    "logs/runtime/runtime-context-summary.md"
                )
                assert "## Session" in payload["summaryText"]
                assert "## Service health" in payload["summaryText"]
                assert "Session active: true" in payload["summaryText"]
                persisted_path = Path(payload["path"])
                assert persisted_path.exists()
                assert persisted_path.read_text(encoding="utf-8") == payload["summaryText"]
            finally:
                await client.close()

    asyncio.run(scenario())


def test_backend_runtime_context_summary_endpoint_survives_persist_failure() -> None:
    async def scenario():
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_path = Path(temp_dir) / "logs"
            logs_path.write_text("not-a-directory", encoding="utf-8")
            client = await _make_client(
                root_dir=temp_dir,
                runtime_service=StaticRuntimeService(),
                webrtc_service=FakeWebRTCService(),
            )
            try:
                response = await client.get("/api/runtime/context-summary")
                assert response.status == 200
                payload = await response.json()
                assert payload["ok"] is True
                assert payload["persisted"] is False
                assert isinstance(payload["persistError"], str)
                assert "## Recent logs" in payload["summaryText"]
                assert "runtime context summary persistence failed" in str(
                    client.app[SESSION_MANAGER].last_event.get("message")
                )

                state = await client.get("/api/state")
                assert state.status == 200
            finally:
                await client.close()

    asyncio.run(scenario())


def test_backend_proxies_runtime_session_routes() -> None:
    async def scenario():
        session_state = {
            "ok": True,
            "session": {
                "active": False,
                "state": "inactive",
                "startedAt": None,
                "config": None,
                "lastEvent": None,
            },
            "processes": [],
            "serviceEndpoints": {},
            "lastError": None,
        }

        async def healthz(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True, "service": "runtime"})

        async def state_route(_request: web.Request) -> web.Response:
            return web.json_response(session_state)

        async def start_route(request: web.Request) -> web.Response:
            payload = await request.json()
            session_state["session"] = {
                "active": True,
                "state": "running",
                "startedAt": 123.0,
                "config": payload,
                "lastEvent": {"message": "runtime session running"},
            }
            session_state["processes"] = [
                {
                    "name": "control_runtime",
                    "state": "running",
                    "required": True,
                    "pid": 1234,
                    "exitCode": None,
                    "startedAt": 123.0,
                    "healthUrl": "http://127.0.0.1:8892/healthz",
                    "stdoutLog": "control.stdout.log",
                    "stderrLog": "control.stderr.log",
                }
            ]
            return web.json_response(session_state)

        async def stop_route(_request: web.Request) -> web.Response:
            session_state["session"] = {
                "active": False,
                "state": "inactive",
                "startedAt": None,
                "config": None,
                "lastEvent": {"message": "runtime session stopped"},
            }
            return web.json_response(session_state)

        supervisor_app = web.Application()
        supervisor_app.router.add_get("/healthz", healthz)
        supervisor_app.router.add_get("/session/state", state_route)
        supervisor_app.router.add_post("/session/start", start_route)
        supervisor_app.router.add_post("/session/stop", stop_route)
        supervisor_server = TestServer(supervisor_app)
        supervisor_client = TestClient(supervisor_server)
        await supervisor_client.start_server()

        client = await _make_client(
            runtime_url=str(supervisor_client.make_url("")).rstrip("/"),
            webrtc_service=FakeWebRTCService(),
        )
        try:
            start = await client.post("/api/session/start", json={"launchMode": "headless", "viewerEnabled": True})
            assert start.status == 200
            start_payload = await start.json()
            assert start_payload["session"]["active"] is True
            assert start_payload["services"]["runtime"]["status"] == "healthy"
            assert start_payload["services"]["backend"]["status"] == "healthy"
            assert start_payload["services"]["controlRuntime"]["status"] == "degraded"
            assert start_payload["processes"][0]["name"] == "control_runtime"

            webrtc = await client.get("/api/webrtc/config")
            assert webrtc.status == 200
            webrtc_payload = await webrtc.json()
            assert webrtc_payload["transportMode"] == "webrtc"

            stop = await client.post("/api/session/stop", json={})
            assert stop.status == 200
            stop_payload = await stop.json()
            assert stop_payload["session"]["active"] is False
        finally:
            await client.close()
            await supervisor_client.close()

    asyncio.run(scenario())


def test_backend_owns_runtime_session_routes_by_default() -> None:
    class FakeRuntimeService:
        def __init__(self) -> None:
            self.last_start_config: dict[str, object] | None = None
            self._state = {
                "ok": True,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime initialized"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }

        def state_payload(self, *, ok: bool = True):
            payload = dict(self._state)
            payload["ok"] = ok
            return payload

        def start_session(self, config: dict[str, object]):
            self.last_start_config = dict(config)
            self._state = {
                "ok": True,
                "session": {
                    "active": True,
                    "state": "running",
                    "startedAt": 123.0,
                    "config": dict(config),
                    "lastEvent": {"message": "runtime session running"},
                },
                "processes": [
                    {
                        "name": "control_runtime",
                        "state": "running",
                        "required": True,
                        "pid": 1234,
                        "exitCode": None,
                        "startedAt": 123.0,
                        "healthUrl": "http://127.0.0.1:8892/healthz",
                        "stdoutLog": "control.stdout.log",
                        "stderrLog": "control.stderr.log",
                    }
                ],
                "serviceEndpoints": {"controlRuntimeUrl": "http://127.0.0.1:8892"},
                "lastError": None,
            }
            return self.state_payload()

        def stop_session(self):
            self._state = {
                "ok": True,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime session stopped"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }
            return self.state_payload()

    async def scenario():
        runtime_service = FakeRuntimeService()
        webrtc_service = FakeWebRTCService(
            frame_meta={
                "type": "frame_meta",
                "frame_id": 3,
                "detections": [],
                "trajectoryPixels": [[12, 34], [56, 78]],
                "activeTarget": {
                    "className": "Navigation Goal",
                    "source": "navigation",
                    "nav_goal_pixel": [220, 140],
                    "world_pose_xyz": [1.0, 2.0, 0.0],
                },
                "system2PixelGoal": [220, 140],
            },
            health_snapshot={
                "transport": "webrtc",
                "mediaIngress": "zmq+shm",
                "mediaEgress": "webrtc",
                "frameAvailable": True,
                "streamStalled": False,
                "frameSeq": 3,
                "frameId": 3,
                "frameAgeMs": 15.0,
                "lastGoodFrameAgeMs": 15.0,
                "peerActive": False,
                "peerSessionId": None,
                "peerTrackRoles": [],
                "rgbAvailable": True,
                "depthAvailable": False,
                "source": "control_runtime",
                "image": {"width": 320, "height": 180},
                "dropCounters": {"shmOverwrite": 0},
                "transportHealth": {
                    "control_endpoint": "tcp://127.0.0.1:18880",
                    "telemetry_endpoint": "tcp://127.0.0.1:18881",
                    "shm_name": "aura_viewer_shm_01",
                    "decodeOk": 5,
                    "decodeDrops": 0,
                    "shmOverwriteDrops": 0,
                    "staleTransitions": 0,
                },
            },
        )
        client = await _make_client(runtime_service=runtime_service, webrtc_service=webrtc_service)
        try:
            start = await client.post(
                "/api/session/start",
                json={
                    "launchMode": "headless",
                    "viewerEnabled": True,
                    "memoryStore": True,
                    "detectionEnabled": False,
                    "locomotionConfig": {
                        "actionScale": 0.6,
                        "onnxDevice": "cpu",
                        "cmdMaxVx": 0.4,
                        "cmdMaxVy": 0.2,
                        "cmdMaxWz": 0.7,
                    },
                },
            )
            assert start.status == 200
            start_payload = await start.json()
            assert start_payload["session"]["active"] is True
            assert start_payload["services"]["runtime"]["status"] == "healthy"
            assert start_payload["services"]["runtime"]["health"]["ownedByBackend"] is True
            assert start_payload["services"]["backend"]["status"] == "healthy"
            assert start_payload["services"]["controlRuntime"]["status"] == "degraded"
            assert start_payload["services"]["reasoningSystem"]["status"] == "degraded"
            assert start_payload["processes"][0]["name"] == "control_runtime"
            assert runtime_service.last_start_config == {
                "launchMode": "headless",
                "scenePreset": "warehouse",
                "viewerEnabled": True,
                "memoryStore": True,
                "detectionEnabled": False,
                "locomotionConfig": {
                    "actionScale": 0.6,
                    "onnxDevice": "cpu",
                    "cmdMaxVx": 0.4,
                    "cmdMaxVy": 0.2,
                    "cmdMaxWz": 0.7,
                },
            }
            assert start_payload["transport"]["frameAvailable"] is True
            assert start_payload["transport"]["streamStalled"] is False
            assert start_payload["transport"]["frameAgeMs"] == 15.0
            assert start_payload["transport"]["lastGoodFrameAgeMs"] == 15.0
            assert start_payload["selectedTargetSummary"]["className"] == "Navigation Goal"
            assert start_payload["selectedTargetSummary"]["navGoalPixel"] == [220, 140]
            assert start_payload["perception"]["trajectoryPointCount"] == 2

            offer = await client.post("/api/webrtc/offer", json={"sdp": "offer", "type": "offer"})
            assert offer.status == 200
            offer_payload = await offer.json()
            assert offer_payload["type"] == "answer"
            assert offer_payload["sessionId"] == "peer-test"

            stop = await client.post("/api/session/stop", json={})
            assert stop.status == 200
            stop_payload = await stop.json()
            assert stop_payload["session"]["active"] is False
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_runtime_logs_start_at_process_offsets(tmp_path: Path) -> None:
    stdout_log = tmp_path / "control.stdout.log"
    stderr_log = tmp_path / "control.stderr.log"
    stdout_log.write_text("old stdout\n", encoding="utf-8")
    stderr_log.write_text("old stderr\n", encoding="utf-8")
    stdout_offset = stdout_log.stat().st_size
    stderr_offset = stderr_log.stat().st_size
    with stdout_log.open("a", encoding="utf-8") as handle:
        handle.write("new stdout\n")
    with stderr_log.open("a", encoding="utf-8") as handle:
        handle.write("new stderr\n")

    class FakeRuntimeService:
        def state_payload(self, *, ok: bool = True):
            return {
                "ok": ok,
                "session": {
                    "active": True,
                    "state": "running",
                    "startedAt": 123.0,
                    "config": {"launchMode": "gui"},
                    "lastEvent": {"message": "runtime session running"},
                },
                "processes": [
                    {
                        "name": "control_runtime",
                        "state": "running",
                        "required": True,
                        "pid": 1234,
                        "exitCode": None,
                        "startedAt": 123.0,
                        "healthUrl": "http://127.0.0.1:8892/healthz",
                        "stdoutLog": str(stdout_log),
                        "stderrLog": str(stderr_log),
                        "stdoutLogOffset": stdout_offset,
                        "stderrLogOffset": stderr_offset,
                    }
                ],
                "serviceEndpoints": {"controlRuntimeUrl": "http://127.0.0.1:8892"},
                "lastError": None,
            }

        def stop_session(self):
            return self.state_payload()

    async def scenario():
        client = await _make_client(
            root_dir=str(tmp_path),
            runtime_service=FakeRuntimeService(),
            webrtc_service=FakeWebRTCService(),
        )
        try:
            response = await client.get("/api/state")
            assert response.status == 200
            payload = await response.json()
            messages = [item["message"] for item in payload["logs"]]
            assert "new stdout" in messages
            assert "new stderr" in messages
            assert "old stdout" not in messages
            assert "old stderr" not in messages
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_system_shutdown_stops_owned_runtime_and_schedules_exit() -> None:
    class FakeRuntimeService:
        def __init__(self) -> None:
            self.stop_calls = 0

        def state_payload(self, *, ok: bool = True):
            return {
                "ok": ok,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime initialized"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }

        def stop_session(self):
            self.stop_calls += 1
            return self.state_payload()

    async def scenario():
        scheduled: list[str] = []
        runtime_service = FakeRuntimeService()
        client = await _make_client(
            runtime_service=runtime_service,
            webrtc_service=FakeWebRTCService(),
            shutdown_scheduler=lambda: scheduled.append("shutdown"),
        )
        try:
            response = await client.post("/api/system/shutdown", json={})
            assert response.status == 200
            payload = await response.json()
            assert payload == {"ok": True, "accepted": True}
            assert runtime_service.stop_calls == 1
            assert scheduled == ["shutdown"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_system_shutdown_proxies_external_runtime_stop_before_exit() -> None:
    async def scenario():
        stop_calls = 0

        async def healthz(_request: web.Request) -> web.Response:
            return web.json_response({"ok": True, "service": "runtime"})

        async def state_route(_request: web.Request) -> web.Response:
            return web.json_response(
                {
                    "ok": True,
                    "session": {
                        "active": False,
                        "state": "inactive",
                        "startedAt": None,
                        "config": None,
                        "lastEvent": {"message": "runtime initialized"},
                    },
                    "processes": [],
                    "serviceEndpoints": {},
                    "lastError": None,
                }
            )

        async def stop_route(_request: web.Request) -> web.Response:
            nonlocal stop_calls
            stop_calls += 1
            return web.json_response(
                {
                    "ok": True,
                    "session": {
                        "active": False,
                        "state": "inactive",
                        "startedAt": None,
                        "config": None,
                        "lastEvent": {"message": "runtime session stopped"},
                    },
                    "processes": [],
                    "serviceEndpoints": {},
                    "lastError": None,
                }
            )

        supervisor_app = web.Application()
        supervisor_app.router.add_get("/healthz", healthz)
        supervisor_app.router.add_get("/session/state", state_route)
        supervisor_app.router.add_post("/session/stop", stop_route)
        supervisor_server = TestServer(supervisor_app)
        supervisor_client = TestClient(supervisor_server)
        await supervisor_client.start_server()

        scheduled: list[str] = []
        client = await _make_client(
            runtime_url=str(supervisor_client.make_url("")).rstrip("/"),
            webrtc_service=FakeWebRTCService(),
            shutdown_scheduler=lambda: scheduled.append("shutdown"),
        )
        try:
            response = await client.post("/api/system/shutdown", json={})
            assert response.status == 200
            payload = await response.json()
            assert payload == {"ok": True, "accepted": True}
        finally:
            await client.close()
            await supervisor_client.close()

        assert stop_calls == 1
        assert scheduled == ["shutdown"]

    asyncio.run(scenario())


def test_backend_webrtc_config_disables_transport_when_viewer_publish_is_disabled() -> None:
    class FakeRuntimeService:
        def __init__(self) -> None:
            self._state = {
                "ok": True,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime initialized"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }

        def state_payload(self, *, ok: bool = True):
            payload = dict(self._state)
            payload["ok"] = ok
            return payload

        def start_session(self, config: dict[str, object]):
            self._state = {
                "ok": True,
                "session": {
                    "active": True,
                    "state": "running",
                    "startedAt": 123.0,
                    "config": dict(config),
                    "lastEvent": {"message": "runtime session running"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }
            return self.state_payload()

        def stop_session(self):
            return self.state_payload()

    async def scenario():
        client = await _make_client(runtime_service=FakeRuntimeService(), webrtc_service=FakeWebRTCService())
        try:
            start = await client.post("/api/session/start", json={"launchMode": "headless", "viewerEnabled": False})
            assert start.status == 200
            start_payload = await start.json()
            assert start_payload["transport"]["viewerEnabled"] is False
            assert start_payload["architecture"]["modules"]["telemetry"]["summary"] == "Viewer publish disabled"

            webrtc = await client.get("/api/webrtc/config")
            assert webrtc.status == 200
            webrtc_payload = await webrtc.json()
            assert webrtc_payload["transportMode"] == "disabled"
            assert webrtc_payload["mediaEgress"] == "disabled"
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_runtime_routes_proxy_to_reasoning_system() -> None:
    async def scenario():
        recorded: list[tuple[str, dict[str, object]]] = []
        reasoning_called = asyncio.Event()

        async def reasoning_respond(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(("respond", payload))
            reasoning_called.set()
            return web.json_response(
                {
                    "ok": True,
                    "route": "task",
                    "request_id": "req-1",
                    "conversation_id": payload["conversation_id"],
                    "reply_text": "작업을 시작했습니다.",
                    "task": {
                        "task_id": "task-1",
                        "task_status": "running",
                        "task_frame": {"intent": "navigate_to_object"},
                        "current_subgoal": {"type": "navigate"},
                        "subgoals": [{"type": "navigate"}],
                    },
                    "error": None,
                }
            )

        async def reasoning_cancel(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(("cancel", payload))
            return web.json_response({"ok": True, "cancelled": True, "status": {"task_status": "cancelled"}})

        reasoning_app = web.Application()
        reasoning_app.router.add_post("/reasoning/respond", reasoning_respond)
        reasoning_app.router.add_post("/reasoning/cancel", reasoning_cancel)
        reasoning_server = TestServer(reasoning_app)
        reasoning_client = TestClient(reasoning_server)
        await reasoning_client.start_server()

        client = await _make_client(reasoning_system_url=str(reasoning_client.make_url("")).rstrip("/"))
        try:
            submit = await client.post("/api/runtime/reason", json={"utterance": "go to the tv", "language": "en"})
            assert submit.status == 200
            submit_payload = await submit.json()
            assert submit_payload["route"] == "task"
            assert submit_payload["task"]["task_status"] == "running"
            await asyncio.wait_for(reasoning_called.wait(), timeout=1.0)

            cancel = await client.post("/api/runtime/cancel", json={})
            assert cancel.status == 200
            cancel_payload = await cancel.json()
            assert cancel_payload["status"]["task_status"] == "cancelled"
        finally:
            await client.close()
            await reasoning_client.close()

        assert recorded[0][0] == "respond"
        assert recorded[0][1]["utterance"] == "go to the tv"
        assert recorded[0][1]["language"] == "en"
        assert recorded[0][1]["conversation_id"]
        assert recorded[1] == ("cancel", {})

    asyncio.run(scenario())


def test_backend_runtime_reason_passes_dialogue_route() -> None:
    async def scenario():
        recorded: list[dict[str, object]] = []

        async def reasoning_respond(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(payload)
            return web.json_response(
                {
                    "ok": True,
                    "route": "dialogue",
                    "request_id": "req-dialogue",
                    "conversation_id": payload["conversation_id"],
                    "reply_text": "안녕하세요.",
                    "task": None,
                    "error": None,
                }
            )

        reasoning_app = web.Application()
        reasoning_app.router.add_post("/reasoning/respond", reasoning_respond)
        reasoning_server = TestServer(reasoning_app)
        reasoning_client = TestClient(reasoning_server)
        await reasoning_client.start_server()

        client = await _make_client(reasoning_system_url=str(reasoning_client.make_url("")).rstrip("/"))
        try:
            submit = await client.post("/api/runtime/reason", json={"utterance": "안녕", "language": "ko"})
            assert submit.status == 200
            submit_payload = await submit.json()
            assert submit_payload["route"] == "dialogue"
            state = await client.get("/api/state")
            assert state.status == 200
            state_payload = await state.json()
            agent_response = state_payload["runtime"]["agentResponse"]
            assert agent_response["text"] == submit_payload["reply_text"]
            assert agent_response["route"] == "dialogue"
            assert agent_response["requestId"] == "req-dialogue"
            assert agent_response["conversationId"] == submit_payload["conversation_id"]
            assert agent_response["error"] is None
            assert submit_payload["reply_text"] == "안녕하세요."
        finally:
            await client.close()
            await reasoning_client.close()

        assert recorded[0]["utterance"] == "안녕"
        assert recorded[0]["language"] == "ko"

    asyncio.run(scenario())


def test_backend_runtime_task_endpoint_is_removed() -> None:
    async def scenario():
        async def reasoning_respond(request: web.Request) -> web.Response:
            payload = await request.json()
            return web.json_response(
                {
                    "ok": True,
                    "route": "task",
                    "request_id": "req-removed",
                    "conversation_id": payload["conversation_id"],
                    "reply_text": "accepted",
                    "task": None,
                    "error": None,
                }
            )

        reasoning_app = web.Application()
        reasoning_app.router.add_post("/reasoning/respond", reasoning_respond)
        reasoning_server = TestServer(reasoning_app)
        reasoning_client = TestClient(reasoning_server)
        await reasoning_client.start_server()

        client = await _make_client(reasoning_system_url=str(reasoning_client.make_url("")).rstrip("/"))
        try:
            submit = await client.post("/api/runtime/task", json={"instruction": "go to the tv", "language": "en"})
            assert submit.status == 404
        finally:
            await client.close()
            await reasoning_client.close()

    asyncio.run(scenario())


def test_backend_runtime_reason_includes_active_scene_preset() -> None:
    class _RuntimeService:
        def __init__(self) -> None:
            self._state = {
                "ok": True,
                "session": {
                    "active": False,
                    "state": "inactive",
                    "startedAt": None,
                    "config": None,
                    "lastEvent": {"message": "runtime initialized"},
                },
                "processes": [],
                "serviceEndpoints": {},
                "lastError": None,
            }

        def state_payload(self):
            return dict(self._state)

        def start_session(self, config: dict[str, object]):
            self._state["session"] = {
                "active": True,
                "state": "running",
                "startedAt": 1.0,
                "config": dict(config),
                "lastEvent": {"message": "runtime session running"},
            }
            return self.state_payload()

        def stop_session(self):
            self._state["session"] = {
                "active": False,
                "state": "inactive",
                "startedAt": None,
                "config": None,
                "lastEvent": {"message": "runtime session stopped"},
            }
            return self.state_payload()

    async def scenario():
        recorded: list[dict[str, object]] = []
        reasoning_called = asyncio.Event()

        async def reasoning_respond(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(payload)
            reasoning_called.set()
            return web.json_response(
                {
                    "ok": True,
                    "route": "task",
                    "request_id": "req-2",
                    "conversation_id": payload["conversation_id"],
                    "reply_text": "작업을 시작했습니다.",
                    "task": {
                        "task_id": "task-2",
                        "task_status": "running",
                        "task_frame": {"intent": "navigate_to_object"},
                        "current_subgoal": {"type": "navigate"},
                        "subgoals": [{"type": "navigate"}],
                    },
                    "error": None,
                }
            )

        reasoning_app = web.Application()
        reasoning_app.router.add_post("/reasoning/respond", reasoning_respond)
        reasoning_server = TestServer(reasoning_app)
        reasoning_client = TestClient(reasoning_server)
        await reasoning_client.start_server()

        client = await _make_client(
            reasoning_system_url=str(reasoning_client.make_url("")).rstrip("/"),
            runtime_service=_RuntimeService(),
            webrtc_service=FakeWebRTCService(),
        )
        try:
            start = await client.post("/api/session/start", json={"scenePreset": "interioragent"})
            assert start.status == 200
            submit = await client.post("/api/runtime/reason", json={"utterance": "go to the tv", "language": "en"})
            assert submit.status == 200
            await asyncio.wait_for(reasoning_called.wait(), timeout=1.0)
        finally:
            await client.close()
            await reasoning_client.close()

        assert recorded[0]["scene_preset"] == "interioragent"

    asyncio.run(scenario())


def test_backend_knowledge_document_api_and_status() -> None:
    async def scenario():
        repository = InMemoryKnowledgeRepository()
        service = KnowledgeService(repository)
        knowledge_runtime = KnowledgeRuntimeHandle(enabled=True, service=service)
        client = await _make_client(webrtc_service=FakeWebRTCService(), knowledge_runtime=knowledge_runtime)
        try:
            create = await client.post(
                "/api/knowledge/documents",
                json={
                    "title": "Warehouse policy",
                    "body_markdown": """
```knowledge-rule
{
  "action": "deny_task",
  "enforcement": "hard",
  "conditions": {
    "intent": "navigate_to_object",
    "target_object": "refrigerator"
  },
  "reason": "Do not navigate directly to the refrigerator."
}
```
""",
                    "publish": False,
                },
            )
            assert create.status == 200
            created_payload = await create.json()
            document_id = created_payload["document"]["documentId"]

            status_before = await client.get("/api/knowledge/status")
            assert status_before.status == 200
            status_before_payload = await status_before.json()
            assert status_before_payload["publishedDocumentCount"] == 0
            assert status_before_payload["activeHardRuleCount"] == 0

            publish = await client.post(f"/api/knowledge/documents/{document_id}/publish", json={})
            assert publish.status == 200

            listed = await client.get("/api/knowledge/documents")
            assert listed.status == 200
            listed_payload = await listed.json()
            assert len(listed_payload["documents"]) == 1
            assert listed_payload["documents"][0]["status"] == "published"

            status_after = await client.get("/api/knowledge/status")
            assert status_after.status == 200
            status_after_payload = await status_after.json()
            assert status_after_payload["knowledgeEnabled"] is True
            assert status_after_payload["publishedDocumentCount"] == 1
            assert status_after_payload["activeHardRuleCount"] == 1
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_agent_memory_admin_api_supports_status_blocks_and_passages() -> None:
    async def scenario():
        repository = InMemoryAgentMemoryRepository()
        service = AgentMemoryService(repository)
        service.ensure_default_blocks()
        runtime = HumanoidMemoryRuntimeHandle(enabled=True, service=service)
        client = await _make_client(
            webrtc_service=FakeWebRTCService(),
            agent_memory_runtime=runtime,
        )
        try:
            status = await client.get("/api/memory/status")
            assert status.status == 200
            status_payload = await status.json()
            assert status_payload["ok"] is True
            assert status_payload["enabled"] is True
            assert status_payload["available"] is True
            assert status_payload["coreBlockCount"] == 6

            blocks = await client.get("/api/memory/blocks")
            assert blocks.status == 200
            blocks_payload = await blocks.json()
            assert {block["label"] for block in blocks_payload["blocks"]} >= {
                "persona",
                "working_memory",
            }

            update = await client.put(
                "/api/memory/blocks/working_memory",
                json={"value": "Current task: inspect the chair.", "limit": 512},
            )
            assert update.status == 200
            update_payload = await update.json()
            assert update_payload["block"]["value"] == "Current task: inspect the chair."
            assert update_payload["block"]["version"] == 2

            read_only = await client.put("/api/memory/blocks/mission_policy", json={"value": "disable rules"})
            assert read_only.status == 403

            create_passage = await client.post(
                "/api/memory/passages",
                json={
                    "content": "The operator corrected that fridge means refrigerator.",
                    "tags": ["correction", "lexicon"],
                    "sceneScope": "warehouse",
                },
            )
            assert create_passage.status == 200

            passages = await client.get("/api/memory/passages?query=fridge&tag=correction&sceneScope=warehouse")
            assert passages.status == 200
            passages_payload = await passages.json()
            assert len(passages_payload["passages"]) == 1
            assert passages_payload["passages"][0]["tags"] == ["correction", "lexicon"]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_object_memory_api_lists_stored_objects() -> None:
    async def scenario():
        repository = InMemoryObjectMemoryRepository()
        service = ObjectMemoryService(repository)
        runtime = ObjectMemoryRuntimeHandle(enabled=True, user_id="tester", service=service)
        sink = ObjectMemoryFrameSink(
            ObjectMemorySinkConfig(dsn="postgres://configured", user_id="tester", scene_scope="warehouse"),
            runtime_handle=runtime,
        )
        service.observe_objects(
            "tester",
            "session-a",
            [
                ObjectObservationInput(
                    frame_idx=7,
                    track_id="track-purple",
                    class_name="purple box",
                    detector_conf=0.91,
                    bbox_xyxy_norm=(0.1, 0.2, 0.4, 0.8),
                    box_area=0.18,
                    aspect_ratio=0.5,
                    image_hash="hash-purple",
                    scene_scope="warehouse",
                    world_pose_xyz=(1.25, 0.0, 4.5),
                    source_id="perception_runtime",
                )
            ],
        )
        client = await _make_client(webrtc_service=FakeWebRTCService(object_memory_sink=sink))
        try:
            response = await client.get("/api/object-memory/objects")
            assert response.status == 200
            payload = await response.json()
            assert payload["ok"] is True
            assert payload["enabled"] is True
            assert payload["available"] is True
            assert payload["activeObjectCount"] == 1
            assert payload["returnedCount"] == 1
            assert payload["userId"] == "tester"
            stored_object = payload["objects"][0]
            assert stored_object["canonicalClass"] == "purple box"
            assert stored_object["sceneScope"] == "warehouse"
            assert stored_object["lastDetectorConf"] == 0.91
            assert stored_object["worldPoseXyz"] == [1.25, 0.0, 4.5]
        finally:
            await client.close()

    asyncio.run(scenario())


def test_backend_planner_catalog_api_supports_get_create_delete_and_validation() -> None:
    async def scenario():
        repository = InMemoryPlannerCatalogRepository()
        service = PlannerCatalogService(repository)
        seeded_snapshot = service.ensure_seed_data()
        planner_catalog_runtime = PlannerCatalogRuntimeHandle(
            enabled=True,
            service=service,
            _last_snapshot=seeded_snapshot,
            _last_refresh_ok=True,
        )
        client = await _make_client(
            webrtc_service=FakeWebRTCService(),
            planner_catalog_runtime=planner_catalog_runtime,
        )
        try:
            catalog = await client.get("/api/planner/catalog")
            assert catalog.status == 200
            catalog_payload = await catalog.json()
            assert catalog_payload["catalog"]["supportedIntentKeys"] == [
                "check_state",
                "find_object",
                "navigate_to_object",
            ]
            assert len(catalog_payload["catalog"]["intents"]) == 3

            duplicate = await client.post("/api/planner/intents", json={"intentKey": "check_state"})
            assert duplicate.status == 409

            delete_report = next(
                template
                for intent in catalog_payload["catalog"]["intents"]
                if intent["intentKey"] == "check_state"
                for template in intent["subgoals"]
                if template["subgoalType"] == "report"
            )
            delete_response = await client.delete(f"/api/planner/subgoals/{delete_report['templateId']}")
            assert delete_response.status == 200
            deleted_payload = await delete_response.json()
            updated_check_state = next(
                intent for intent in deleted_payload["catalog"]["intents"] if intent["intentKey"] == "check_state"
            )
            assert [template["subgoalType"] for template in updated_check_state["subgoals"]] == ["navigate", "inspect", "return"]

            invalid_subgoal = await client.post(
                "/api/planner/subgoals",
                json={
                    "intentId": updated_check_state["intentId"],
                    "subgoalType": "inspect",
                    "sequenceNo": 4,
                    "activationCondition": "when_report_result",
                },
            )
            assert invalid_subgoal.status == 409

            create_intent = await client.post("/api/planner/intents", json={"intentKey": "find_object"})
            assert create_intent.status == 409

            delete_intent = next(
                intent for intent in deleted_payload["catalog"]["intents"] if intent["intentKey"] == "navigate_to_object"
            )
            deleted_intent_response = await client.delete(f"/api/planner/intents/{delete_intent['intentId']}")
            assert deleted_intent_response.status == 200
            deleted_intent_payload = await deleted_intent_response.json()
            assert {
                intent["intentKey"] for intent in deleted_intent_payload["catalog"]["intents"]
            } == {"check_state", "find_object"}
        finally:
            await client.close()

    asyncio.run(scenario())


def test_dashboard_natural_language_route_forwards_go_to_purple_box_utterance() -> None:
    async def scenario():
        recorded: list[dict[str, object]] = []
        reasoning_called = asyncio.Event()

        async def reasoning_respond(request: web.Request) -> web.Response:
            payload = await request.json()
            recorded.append(payload)
            reasoning_called.set()
            return web.json_response(
                {
                    "ok": True,
                    "route": "task",
                    "request_id": "req-3",
                    "conversation_id": payload["conversation_id"],
                    "reply_text": "작업을 시작했습니다.",
                    "task": {
                        "task_id": "task-3",
                        "task_status": "running",
                        "task_frame": {"intent": "navigate_to_object"},
                        "current_subgoal": {"type": "navigate"},
                        "subgoals": [{"type": "navigate"}],
                    },
                    "error": None,
                }
            )

        reasoning_app = web.Application()
        reasoning_app.router.add_post("/reasoning/respond", reasoning_respond)
        reasoning_server = TestServer(reasoning_app)
        reasoning_client = TestClient(reasoning_server)
        await reasoning_client.start_server()

        client = await _make_client(reasoning_system_url=str(reasoning_client.make_url("")).rstrip("/"))
        try:
            submit = await client.post("/api/runtime/reason", json={"utterance": "go to purple box", "language": "en"})
            assert submit.status == 200
            submit_payload = await submit.json()
            assert submit_payload["route"] == "task"
            assert submit_payload["reply_text"] == "작업을 시작했습니다."
            await asyncio.wait_for(reasoning_called.wait(), timeout=1.0)
        finally:
            await client.close()
            await reasoning_client.close()

        assert recorded[0]["utterance"] == "go to purple box"
        assert recorded[0]["language"] == "en"

    asyncio.run(scenario())


def test_backend_rejects_invalid_session_config() -> None:
    async def scenario():
        client = await _make_client(runtime_service=None, webrtc_service=FakeWebRTCService())
        try:
            response = await client.post(
                "/api/session/start",
                json={"viewerEnabled": "maybe", "locomotionConfig": {"actionScale": -1.0}},
            )
            assert response.status == 400
            assert "viewerEnabled must be a boolean" in await response.text()
        finally:
            await client.close()

    asyncio.run(scenario())


def test_runtime_control_api_exposes_runtime_and_camera_routes() -> None:
    class _Handler:
        def runtime_status(self):
            return {
                "executionMode": "NAV",
                "state_label": "waiting",
                "viewer": {
                    "transport": "webrtc",
                    "frameAvailable": True,
                    "frameSeq": 3,
                    "frameId": 3,
                    "frameAgeMs": 15.0,
                    "peerActive": True,
                    "peerSessionId": "peer-test",
                    "peerTrackRoles": ["rgb"],
                    "rgbAvailable": True,
                    "depthAvailable": True,
                    "source": "control_runtime",
                    "image": {"width": 320, "height": 180},
                },
            }

    class _Camera:
        def pitch_status(self):
            return {"ready": True, "target_pitch_deg": 0.0, "applied_pitch_deg": 0.0}

        def set_pitch_deg(self, value: float):
            return value

        def add_pitch_deg(self, value: float):
            return value

    server = RuntimeControlApiServer("127.0.0.1", 0, _Handler(), _Camera())
    try:
        server.start()
        from urllib.request import Request, urlopen

        with urlopen(f"http://127.0.0.1:{server.port}/runtime/status", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["ok"] is True
            assert payload["executionMode"] == "NAV"
            assert payload["viewer"]["frameAvailable"] is True

        request = Request(
            f"http://127.0.0.1:{server.port}/camera/pitch",
            data=json.dumps({"delta_deg": -10.0}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
            assert payload["ok"] is True
            assert payload["updated"] == "relative"
    finally:
        server.shutdown()
