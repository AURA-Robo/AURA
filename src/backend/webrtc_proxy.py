"""WebRTC signaling proxy endpoints for the backend."""

from __future__ import annotations

from typing import Any

from aiohttp import ClientSession, web

from backend.app_keys import SESSION_MANAGER, WEBRTC_PROXY_BASE, WEBRTC_SERVICE


async def get_config(app: web.Application) -> dict[str, Any]:
    session_manager = app[SESSION_MANAGER]
    session_config = session_manager.session_config or {}
    viewer_enabled = bool(session_config.get("viewerEnabled")) if session_config else False
    if app[WEBRTC_SERVICE] is not None:
        return app[WEBRTC_SERVICE].public_config(enabled=viewer_enabled)
    return {
        "iceServers": [],
        "enableDepthTrack": False,
        "proxyMode": "passthrough" if app[WEBRTC_PROXY_BASE] else "disabled",
        "transportMode": "webrtc" if viewer_enabled and app[WEBRTC_PROXY_BASE] else "disabled",
        "mediaIngress": "external" if app[WEBRTC_PROXY_BASE] else "unavailable",
        "mediaEgress": "webrtc" if viewer_enabled and app[WEBRTC_PROXY_BASE] else "disabled",
        "observeOnly": True,
        "peerModel": "single",
        "channelLabels": ["state", "telemetry"],
    }


async def proxy_offer(app: web.Application, payload: dict[str, Any]) -> web.Response:
    base = str(app[WEBRTC_PROXY_BASE] or "").rstrip("/")
    if not base:
        service = app[WEBRTC_SERVICE]
        if service is None:
            return web.json_response({"error": "webrtc_proxy_unavailable"}, status=503)
        if hasattr(service, "dependencies_available") and not service.dependencies_available():
            message = (
                service.dependency_error_message()
                if hasattr(service, "dependency_error_message")
                else "WebRTC dependencies are unavailable."
            )
            return web.json_response(
                {
                    "error": "webrtc_dependency_missing",
                    "message": message,
                },
                status=503,
            )
        try:
            response = await service.accept_offer(payload)
        except RuntimeError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(response)
    async with ClientSession() as session:
        async with session.post(f"{base}/offer", json=payload) as response:
            return web.json_response(await response.json(), status=response.status)
