from __future__ import annotations

import asyncio
from types import SimpleNamespace

from backend.webrtc.config import WebRTCServiceConfig
from backend.webrtc.service import WebRTCService
from backend.webrtc.session import WebRTCPeerSession


class _FailingChannel:
    def __init__(self, label: str) -> None:
        self.label = label
        self.readyState = "open"

    def send(self, _payload: str) -> None:
        raise RuntimeError("send failed")


class _SubscriberStub:
    current_frame = None

    @property
    def debug_counters(self) -> dict[str, int]:
        return {
            "decodeOk": 4,
            "decodeDrops": 1,
            "latestFrameDrops": 2,
            "objectMemoryQueued": 3,
            "objectMemoryQueueDrops": 1,
            "objectMemoryQueueDepth": 0,
        }

    @property
    def latest_health(self) -> dict[str, object]:
        return {}

    def last_frame_age_ms(self):  # noqa: ANN001
        return None

    def has_fresh_frame(self) -> bool:
        return False

    def shm_overwrite_drops(self) -> int:
        return 0


class _SessionManagerStub:
    def __init__(self, active_session) -> None:  # noqa: ANN001
        self.active_session = active_session


def test_webrtc_session_records_send_errors_without_raising() -> None:
    async def scenario() -> None:
        session = object.__new__(WebRTCPeerSession)
        session.config = WebRTCServiceConfig()
        session._debug_counters = {  # noqa: SLF001
            "sendErrors": 0,
            "stateSendErrors": 0,
            "telemetrySendErrors": 0,
        }

        await session._send_json(_FailingChannel("state"), {"type": "frame_state"})  # noqa: SLF001
        await session._send_json(_FailingChannel("telemetry"), {"type": "frame_meta"})  # noqa: SLF001

        assert session.debug_snapshot() == {
            "sendErrors": 2,
            "stateSendErrors": 1,
            "telemetrySendErrors": 1,
        }

    asyncio.run(scenario())


def test_webrtc_health_snapshot_exposes_session_send_error_counters() -> None:
    active_session = SimpleNamespace(
        session_id="peer-1",
        track_roles=["rgb"],
        debug_snapshot=lambda: {
            "sendErrors": 3,
            "stateSendErrors": 1,
            "telemetrySendErrors": 2,
        },
    )
    service = WebRTCService(
        WebRTCServiceConfig(),
        subscriber=_SubscriberStub(),
        session_manager=_SessionManagerStub(active_session),
    )

    health = service.health_snapshot()

    assert health["telemetryHz"] == 10.0
    assert health["transportHealth"]["sendErrors"] == 3
    assert health["transportHealth"]["stateSendErrors"] == 1
    assert health["transportHealth"]["telemetrySendErrors"] == 2
    assert health["transportHealth"]["latestFrameDrops"] == 2
    assert health["transportHealth"]["objectMemoryQueued"] == 3
    assert health["transportHealth"]["objectMemoryQueueDrops"] == 1
