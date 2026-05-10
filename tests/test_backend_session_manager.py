from __future__ import annotations

import asyncio

from backend.app_keys import (
    API_BASE_URL,
    CONTROL_RUNTIME_URL,
    HTTP,
    INFERENCE_SYSTEM_URL,
    NAVIGATION_SYSTEM_URL,
    REASONING_SYSTEM_URL,
    RUNTIME_OWNED,
    RUNTIME_SERVICE,
    RUNTIME_URL,
    WEBRTC_SERVICE,
)
from backend.session_manager import DashboardSessionManager


class _FakeRuntimeService:
    def state_payload(self):
        return {
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


class _FakeRuntimeServiceWithConfig:
    def __init__(self, config: dict[str, object]) -> None:
        self._config = dict(config)

    def state_payload(self):
        return {
            "ok": True,
            "session": {
                "active": True,
                "state": "running",
                "startedAt": "now",
                "config": dict(self._config),
                "lastEvent": {"message": "runtime active"},
            },
            "processes": [],
            "serviceEndpoints": {},
            "lastError": None,
        }


class _FakeWebRTCService:
    def __init__(self) -> None:
        self.enabled_calls: list[bool] = []

    def set_object_memory_enabled(self, enabled: bool) -> None:
        self.enabled_calls.append(bool(enabled))

    def health_snapshot(self):
        return {
            "transport": "webrtc",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc",
            "frameAvailable": True,
            "streamStalled": False,
            "frameSeq": 1,
            "frameId": 10,
            "frameAgeMs": 12.5,
            "lastGoodFrameAgeMs": 12.5,
            "peerActive": False,
            "peerSessionId": None,
            "peerTrackRoles": [],
            "rgbAvailable": True,
            "depthAvailable": False,
            "source": "control_runtime",
            "image": {"width": 320, "height": 180},
            "dropCounters": {"shmOverwrite": 0},
            "transportHealth": {},
            "latestHealth": {},
            "objectMemory": {
                "configured": True,
                "enabled": True,
                "available": True,
                "objectCount": 3,
                "observationCount": 5,
                "lastSuccess": True,
                "lastObservedAt": "2026-04-14T00:00:00+00:00",
                "lastIngestLatencyMs": 4.5,
                "lastError": None,
                "degradedReason": None,
            },
        }


def test_session_manager_caches_probe_results_between_nearby_state_requests(monkeypatch) -> None:
    calls = {
        "runtime": 0,
        "reasoning": 0,
        "navigation": 0,
    }

    def _runtime_status(_base_url: str):
        calls["runtime"] += 1
        return {"ok": False, "error": "offline"}

    def _reasoning_status(_base_url: str):
        calls["reasoning"] += 1
        return {"ok": False, "error": "offline"}

    def _navigation_status(_base_url: str):
        calls["navigation"] += 1
        return {"ok": False, "error": "offline"}

    monkeypatch.setattr("backend.session_manager.fetch_runtime_status", _runtime_status)
    monkeypatch.setattr("backend.session_manager.fetch_reasoning_status", _reasoning_status)
    monkeypatch.setattr("backend.session_manager.fetch_navigation_status", _navigation_status)

    app = {
        API_BASE_URL: "http://127.0.0.1:18095",
        CONTROL_RUNTIME_URL: "http://127.0.0.1:8892",
        HTTP: None,
        INFERENCE_SYSTEM_URL: "http://127.0.0.1:15880",
        NAVIGATION_SYSTEM_URL: "http://127.0.0.1:17882",
        REASONING_SYSTEM_URL: "http://127.0.0.1:17881",
        RUNTIME_OWNED: True,
        RUNTIME_SERVICE: _FakeRuntimeService(),
        RUNTIME_URL: "",
        WEBRTC_SERVICE: None,
    }
    manager = DashboardSessionManager(app)

    async def scenario() -> None:
        await manager.build_state(force_refresh=True)
        await manager.build_state()

    asyncio.run(scenario())

    assert calls == {
        "runtime": 1,
        "reasoning": 1,
        "navigation": 1,
    }


def test_session_manager_syncs_object_memory_policy_and_reports_health(monkeypatch) -> None:
    monkeypatch.setattr("backend.session_manager.fetch_runtime_status", lambda _base_url: {"ok": False, "error": "offline"})
    monkeypatch.setattr("backend.session_manager.fetch_reasoning_status", lambda _base_url: {"ok": False, "error": "offline"})
    monkeypatch.setattr("backend.session_manager.fetch_navigation_status", lambda _base_url: {"ok": False, "error": "offline"})

    webrtc_service = _FakeWebRTCService()
    app = {
        API_BASE_URL: "http://127.0.0.1:18095",
        CONTROL_RUNTIME_URL: "http://127.0.0.1:8892",
        HTTP: None,
        INFERENCE_SYSTEM_URL: "http://127.0.0.1:15880",
        NAVIGATION_SYSTEM_URL: "http://127.0.0.1:17882",
        REASONING_SYSTEM_URL: "http://127.0.0.1:17881",
        RUNTIME_OWNED: True,
        RUNTIME_SERVICE: _FakeRuntimeServiceWithConfig(
            {
                "viewerEnabled": True,
                "memoryStore": True,
                "detectionEnabled": True,
            }
        ),
        RUNTIME_URL: "",
        WEBRTC_SERVICE: webrtc_service,
    }
    manager = DashboardSessionManager(app)

    state = asyncio.run(manager.build_state(force_refresh=True))

    assert webrtc_service.enabled_calls[-1] is True
    assert state["memory"]["objectCount"] == 3
    assert state["memory"]["objectMemoryEnabled"] is True
    assert state["memory"]["objectMemoryAvailable"] is True
    assert state["architecture"]["modules"]["memory"]["status"] == "healthy"
    assert state["latencyBreakdown"]["memoryLatencyMs"] == 4.5


def test_session_manager_promotes_reasoning_status_into_runtime_and_catalog(monkeypatch) -> None:
    monkeypatch.setattr("backend.session_manager.fetch_runtime_status", lambda _base_url: {"ok": False, "error": "offline"})
    monkeypatch.setattr(
        "backend.session_manager.fetch_reasoning_status",
        lambda _base_url: {
            "ok": True,
            "status": {
                "ok": True,
                "task_status": "running",
                "last_route": "task",
                "instruction": "go to the tv",
                "task_id": "task-1",
                "task_frame": {"intent": "navigate_to_object"},
                "current_subgoal": {"type": "navigate", "label": "TV"},
                "subgoals": [{"type": "navigate", "status": "running"}],
                "last_error": None,
            },
        },
    )
    monkeypatch.setattr("backend.session_manager.fetch_navigation_status", lambda _base_url: {"ok": False, "error": "offline"})

    app = {
        API_BASE_URL: "http://127.0.0.1:18095",
        CONTROL_RUNTIME_URL: "http://127.0.0.1:8892",
        HTTP: None,
        INFERENCE_SYSTEM_URL: "http://127.0.0.1:15880",
        NAVIGATION_SYSTEM_URL: "http://127.0.0.1:17882",
        REASONING_SYSTEM_URL: "http://127.0.0.1:17881",
        RUNTIME_OWNED: True,
        RUNTIME_SERVICE: _FakeRuntimeServiceWithConfig(
            {
                "viewerEnabled": True,
                "memoryStore": False,
                "detectionEnabled": True,
            }
        ),
        RUNTIME_URL: "",
        WEBRTC_SERVICE: None,
    }
    manager = DashboardSessionManager(app)

    state = asyncio.run(manager.build_state(force_refresh=True))

    assert state["runtime"]["reasoningTaskStatus"] == "running"
    assert state["runtime"]["plannerControlMode"] == "running"
    assert state["runtime"]["reasoningRoute"] == "task"
    assert state["dashboardCatalog"]["highlights"]["reasoningStatus"] == "running"
    assert state["dashboardCatalog"]["highlights"]["plannerStatus"] == "running"
    assert any(item["id"] == "source-reasoning-instruction" for item in state["dashboardCatalog"]["sources"])
    assert any(item["name"] == "reasoning.task_status" for item in state["dashboardCatalog"]["states"])
