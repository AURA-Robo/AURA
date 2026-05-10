"""Backend-owned WebRTC gateway service."""

from __future__ import annotations

from typing import Any

from .config import WebRTCServiceConfig
from .object_memory import (
    DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES,
    DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES,
    ObjectMemoryFrameSink,
    ObjectMemorySinkConfig,
)
from .session import PeerSessionManager, webrtc_dependencies_available, webrtc_dependency_error_message
from .subscriber import ObservationSubscriber


class WebRTCService:
    def __init__(
        self,
        config: WebRTCServiceConfig | None = None,
        *,
        subscriber: ObservationSubscriber | None = None,
        session_manager: PeerSessionManager | None = None,
    ) -> None:
        self.config = config or WebRTCServiceConfig()
        self.object_memory_sink = None
        if subscriber is None:
            self.object_memory_sink = ObjectMemoryFrameSink(
                ObjectMemorySinkConfig(
                    dsn=str(self.config.object_memory_dsn),
                    object_event_log_path=str(self.config.object_memory_event_log_path),
                    user_id=str(self.config.object_memory_user_id),
                    session_id=str(self.config.object_memory_session_id),
                    auto_migrate=bool(self.config.object_memory_auto_migrate),
                    scene_scope=str(getattr(self.config, "scene_scope", "") or ""),
                    min_detector_confidence=float(self.config.object_memory_min_detector_confidence),
                    min_bbox_area_norm=float(self.config.object_memory_min_bbox_area_norm),
                    max_bbox_area_norm=float(self.config.object_memory_max_bbox_area_norm),
                    require_world_pose=bool(self.config.object_memory_require_world_pose),
                    allowed_class_names=(
                        tuple(self.config.object_memory_allowed_classes) or DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES
                    ),
                    blocked_class_names=(
                        tuple(self.config.object_memory_blocked_classes) or DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES
                    ),
                )
            )
            self.subscriber = ObservationSubscriber(
                self.config,
                object_memory_sink=self.object_memory_sink,
            )
        else:
            self.subscriber = subscriber
            self.object_memory_sink = getattr(subscriber, "object_memory_sink", None)
        self.session_manager = session_manager or PeerSessionManager(self.config, self.subscriber)

    async def start(self) -> None:
        await self.subscriber.start()

    async def close(self) -> None:
        await self.session_manager.close()
        await self.subscriber.close()

    def set_object_memory_enabled(self, enabled: bool) -> None:
        if self.object_memory_sink is not None:
            self.object_memory_sink.set_enabled(enabled)

    def set_object_memory_scene_scope(self, scene_scope: str | None) -> None:
        if self.object_memory_sink is not None and hasattr(self.object_memory_sink, "set_scene_scope"):
            self.object_memory_sink.set_scene_scope(scene_scope)

    def public_config(self, *, enabled: bool) -> dict[str, object]:
        if self.dependencies_available():
            payload = self.config.public_config(enabled=enabled)
            payload["dependencyMissing"] = False
            return payload
        payload = self.config.public_config(enabled=False)
        payload["dependencyMissing"] = True
        payload["dependencyError"] = self.dependency_error_message()
        payload["installHint"] = (
            "Run scripts\\setup_system_venv_windows.ps1, then restart backend_windows.ps1."
        )
        return payload

    def dependencies_available(self) -> bool:
        return webrtc_dependencies_available()

    def dependency_error_message(self) -> str:
        return webrtc_dependency_error_message()

    async def accept_offer(self, offer_payload: dict[str, object]) -> dict[str, object]:
        if not isinstance(offer_payload, dict):
            raise RuntimeError("offer payload must be a JSON object")
        if str(offer_payload.get("type", "")).strip().lower() != "offer":
            raise RuntimeError("offer payload must have type=offer")
        session, answer = await self.session_manager.accept_offer(offer_payload)
        return {
            "sdp": str(answer.sdp),
            "type": str(answer.type),
            "sessionId": str(session.session_id),
        }

    def health_snapshot(self) -> dict[str, Any]:
        session = self.session_manager.active_session
        frame = self.subscriber.current_frame
        frame_age = self.subscriber.last_frame_age_ms()
        frame_available = self.subscriber.has_fresh_frame()
        stream_stalled = bool(frame is not None and not frame_available)
        drop_counters = {
            "shmOverwrite": int(self.subscriber.shm_overwrite_drops()),
        }
        debug_counters = self.subscriber.debug_counters
        session_debug = (
            {"sendErrors": 0, "stateSendErrors": 0, "telemetrySendErrors": 0}
            if session is None or not hasattr(session, "debug_snapshot")
            else session.debug_snapshot()
        )
        return {
            "transport": "webrtc",
            "mediaIngress": "zmq+shm",
            "mediaEgress": "webrtc",
            "rgbFps": float(self.config.rgb_fps),
            "depthFps": float(self.config.depth_fps) if self.config.enable_depth_track else 0.0,
            "telemetryHz": float(self.config.telemetry_hz),
            "enableDepthTrack": bool(self.config.enable_depth_track),
            "frameAvailable": bool(frame_available),
            "streamStalled": stream_stalled,
            "frameSeq": None if frame is None else int(frame.seq),
            "frameId": None if frame is None else int(frame.frame_header.frame_id),
            "frameAgeMs": frame_age,
            "lastGoodFrameAgeMs": frame_age,
            "peerActive": session is not None,
            "peerSessionId": None if session is None else str(session.session_id),
            "peerTrackRoles": [] if session is None else list(session.track_roles),
            "rgbAvailable": bool(frame_available),
            "depthAvailable": bool(frame_available and frame is not None and frame.depth_image_m is not None),
            "source": "control_runtime" if frame is None else str(frame.frame_header.source),
            "image": {
                "width": 0 if frame is None else int(frame.frame_header.width),
                "height": 0 if frame is None else int(frame.frame_header.height),
            },
            "dropCounters": drop_counters,
            "transportHealth": {
                "control_endpoint": str(self.config.control_endpoint),
                "telemetry_endpoint": str(self.config.telemetry_endpoint),
                "shm_name": str(self.config.shm_name),
                "decodeOk": int(debug_counters.get("decodeOk", 0)),
                "decodeDrops": int(debug_counters.get("decodeDrops", 0)),
                "shmOverwriteDrops": int(debug_counters.get("shmOverwriteDrops", 0)),
                "staleTransitions": int(debug_counters.get("staleTransitions", 0)),
                "latestFrameDrops": int(debug_counters.get("latestFrameDrops", 0)),
                "objectMemoryQueued": int(debug_counters.get("objectMemoryQueued", 0)),
                "objectMemoryQueueDrops": int(debug_counters.get("objectMemoryQueueDrops", 0)),
                "objectMemoryProcessed": int(debug_counters.get("objectMemoryProcessed", 0)),
                "objectMemoryErrors": int(debug_counters.get("objectMemoryErrors", 0)),
                "objectMemoryQueueDepth": int(debug_counters.get("objectMemoryQueueDepth", 0)),
                "sendErrors": int(session_debug.get("sendErrors", 0)),
                "stateSendErrors": int(session_debug.get("stateSendErrors", 0)),
                "telemetrySendErrors": int(session_debug.get("telemetrySendErrors", 0)),
            },
            "latestHealth": self.subscriber.latest_health,
            "objectMemory": (
                {"configured": False, "enabled": False, "available": False}
                if self.object_memory_sink is None
                else self.object_memory_sink.health_snapshot()
            ),
        }
