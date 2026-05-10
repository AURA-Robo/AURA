"""Configuration primitives for the backend-owned WebRTC viewer."""

from __future__ import annotations

from dataclasses import dataclass

from systems.memory import DEFAULT_OBJECT_MEMORY_USER_ID

from .object_memory import DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES, DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES
from systems.shared.viewer_transport import (
    VIEWER_CONTROL_ENDPOINT,
    VIEWER_SHM_CAPACITY,
    VIEWER_SHM_NAME,
    VIEWER_SHM_SLOT_SIZE,
    VIEWER_TELEMETRY_ENDPOINT,
)


@dataclass(frozen=True)
class IceServerConfig:
    urls: tuple[str, ...]

    def as_public_dict(self) -> dict[str, object]:
        return {"urls": list(self.urls)}


@dataclass(frozen=True)
class WebRTCServiceConfig:
    control_endpoint: str = VIEWER_CONTROL_ENDPOINT
    telemetry_endpoint: str = VIEWER_TELEMETRY_ENDPOINT
    shm_name: str = VIEWER_SHM_NAME
    shm_slot_size: int = VIEWER_SHM_SLOT_SIZE
    shm_capacity: int = VIEWER_SHM_CAPACITY
    enable_depth_track: bool = False
    rgb_fps: float = 30.0
    depth_fps: float = 15.0
    telemetry_hz: float = 10.0
    state_snapshot_hz: float = 2.0
    poll_interval_ms: int = 10
    latest_frame_drain_batches: int = 8
    object_memory_queue_size: int = 8
    stale_frame_timeout_sec: float = 2.0
    identity: str = "backend_webrtc"
    observe_only: bool = True
    peer_model: str = "single"
    channel_labels: tuple[str, str] = ("state", "telemetry")
    ice_servers: tuple[IceServerConfig, ...] = ()
    object_memory_dsn: str = ""
    object_memory_event_log_path: str = ""
    object_memory_user_id: str = DEFAULT_OBJECT_MEMORY_USER_ID
    object_memory_session_id: str = ""
    object_memory_auto_migrate: bool = False
    object_memory_min_detector_confidence: float = 0.80
    object_memory_min_bbox_area_norm: float = 0.0005
    object_memory_max_bbox_area_norm: float = 0.35
    object_memory_require_world_pose: bool = True
    object_memory_allowed_classes: tuple[str, ...] = DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES
    object_memory_blocked_classes: tuple[str, ...] = DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES
    scene_scope: str = ""

    def public_config(self, *, enabled: bool) -> dict[str, object]:
        return {
            "transportMode": "webrtc" if enabled else "disabled",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc" if enabled else "disabled",
            "observeOnly": bool(self.observe_only),
            "peerModel": str(self.peer_model),
            "channelLabels": list(self.channel_labels),
            "iceServers": [item.as_public_dict() for item in self.ice_servers],
            "enableDepthTrack": bool(self.enable_depth_track),
            "rgbFps": float(self.rgb_fps),
            "depthFps": float(self.depth_fps) if self.enable_depth_track else 0.0,
            "telemetryHz": float(self.telemetry_hz),
            "latestFrameDrainBatches": int(self.latest_frame_drain_batches),
            "objectMemoryQueueSize": int(self.object_memory_queue_size),
            "proxyMode": "internal",
        }
