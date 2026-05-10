from __future__ import annotations

from backend.api.serve_backend import build_arg_parser
from backend.webrtc import service as webrtc_service_module
from backend.webrtc.config import WebRTCServiceConfig
from backend.webrtc.service import WebRTCService


class _SubscriberStub:
    current_frame = None
    object_memory_sink = None

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _SessionManagerStub:
    active_session = None

    async def close(self) -> None:
        return None


def test_webrtc_defaults_prioritize_direct_rgb_streaming() -> None:
    config = WebRTCServiceConfig()

    assert config.enable_depth_track is False
    assert config.rgb_fps == 30.0
    assert config.depth_fps == 15.0
    assert config.telemetry_hz == 10.0
    assert config.poll_interval_ms == 10
    assert config.latest_frame_drain_batches == 8
    assert config.object_memory_queue_size == 8
    assert config.object_memory_dsn == ""
    assert config.object_memory_event_log_path == ""
    assert config.object_memory_user_id == "local-operator"
    assert config.object_memory_auto_migrate is False
    assert config.object_memory_min_detector_confidence == 0.80
    assert config.object_memory_max_bbox_area_norm == 0.35
    assert config.object_memory_require_world_pose is True
    assert "chair" in config.object_memory_allowed_classes
    assert "court" in config.object_memory_blocked_classes

    public = config.public_config(enabled=True)
    assert public["transportMode"] == "webrtc"
    assert public["enableDepthTrack"] is False
    assert public["rgbFps"] == 30.0
    assert public["depthFps"] == 0.0
    assert public["telemetryHz"] == 10.0


def test_webrtc_public_config_reports_missing_peer_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(webrtc_service_module, "webrtc_dependencies_available", lambda: False, raising=False)
    monkeypatch.setattr(
        webrtc_service_module,
        "webrtc_dependency_error_message",
        lambda: "aiortc is required for WebRTC peer sessions.",
        raising=False,
    )

    service = WebRTCService(
        WebRTCServiceConfig(),
        subscriber=_SubscriberStub(),
        session_manager=_SessionManagerStub(),
    )

    public = service.public_config(enabled=True)

    assert public["transportMode"] == "disabled"
    assert public["mediaEgress"] == "disabled"
    assert public["dependencyMissing"] is True
    assert "aiortc" in public["dependencyError"]


def test_backend_parser_exposes_streaming_optimization_tunables() -> None:
    args = build_arg_parser().parse_args(
        [
            "--webrtc-latest-frame-drain-batches",
            "4",
            "--webrtc-object-memory-queue-size",
            "3",
        ]
    )

    assert args.webrtc_latest_frame_drain_batches == 4
    assert args.webrtc_object_memory_queue_size == 3
