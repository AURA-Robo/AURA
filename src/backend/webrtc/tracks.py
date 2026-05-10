"""aiortc video tracks backed by the latest viewer frame."""

from __future__ import annotations

import asyncio
from fractions import Fraction

import numpy as np

from .subscriber import ObservationSubscriber

try:
    from aiortc import VideoStreamTrack
    from av import VideoFrame
except Exception as exc:  # noqa: BLE001
    VideoStreamTrack = object  # type: ignore[assignment]
    VideoFrame = None  # type: ignore[assignment]
    _VIDEO_IMPORT_ERROR = exc
else:
    _VIDEO_IMPORT_ERROR = None


def _require_video_dependencies() -> None:
    if _VIDEO_IMPORT_ERROR is not None:
        raise RuntimeError("aiortc and av are required for WebRTC video tracks.") from _VIDEO_IMPORT_ERROR


def video_dependency_error_message() -> str:
    if _VIDEO_IMPORT_ERROR is None:
        return ""
    return "aiortc and av are required for WebRTC video tracks."


def video_dependencies_available() -> bool:
    return _VIDEO_IMPORT_ERROR is None


def _blank_rgb_frame(height: int, width: int) -> np.ndarray:
    return np.zeros((max(int(height), 1), max(int(width), 1), 3), dtype=np.uint8)


def _render_depth_preview_image(depth_image: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth_image, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        return _blank_rgb_frame(240, 320)
    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        return _blank_rgb_frame(depth.shape[0], depth.shape[1])
    safe_depth = np.where(finite_mask, np.maximum(depth, 0.0), 0.0)
    maximum = float(np.max(safe_depth))
    if maximum <= 0.0:
        normalized = np.zeros_like(safe_depth, dtype=np.float32)
    else:
        normalized = np.clip(safe_depth / maximum, 0.0, 1.0)
    return np.repeat((normalized[..., None] * 255.0).astype(np.uint8), 3, axis=2)


class _LatestFrameTrack(VideoStreamTrack):  # type: ignore[misc]
    def __init__(self, subscriber: ObservationSubscriber, *, fps: float, track_role: str) -> None:
        _require_video_dependencies()
        super().__init__()
        self._subscriber = subscriber
        self._fps = max(float(fps), 1.0)
        self.track_role = str(track_role)
        self._frame_period = 1.0 / self._fps
        self._next_frame_at: float | None = None
        self._pts = 0

    async def recv(self):
        loop = asyncio.get_running_loop()
        now = loop.time()
        if self._next_frame_at is None:
            self._next_frame_at = now + self._frame_period
        else:
            wait_time = self._next_frame_at - now
            if wait_time > 0.0:
                await asyncio.sleep(wait_time)
            self._next_frame_at = max(self._next_frame_at + self._frame_period, loop.time())
        image, pixel_format = self._render_image()
        frame = VideoFrame.from_ndarray(image, format=pixel_format)
        frame.pts = int(self._pts)
        frame.time_base = Fraction(1, 90000)
        self._pts += int(round(90000.0 / self._fps))
        return frame

    def _render_image(self) -> tuple[np.ndarray, str]:
        raise NotImplementedError

    def _frame_shape(self) -> tuple[int, int]:
        current = self._subscriber.current_frame
        if current is not None and current.rgb_image.ndim == 3:
            return int(current.rgb_image.shape[0]), int(current.rgb_image.shape[1])
        return 240, 320


class RgbVideoTrack(_LatestFrameTrack):
    def __init__(self, subscriber: ObservationSubscriber, *, fps: float) -> None:
        super().__init__(subscriber, fps=fps, track_role="rgb")

    def _render_image(self) -> tuple[np.ndarray, str]:
        current = self._subscriber.current_frame
        if current is not None:
            return np.asarray(current.rgb_image, dtype=np.uint8), "rgb24"
        height, width = self._frame_shape()
        return _blank_rgb_frame(height, width), "rgb24"


class DepthPreviewVideoTrack(_LatestFrameTrack):
    def __init__(self, subscriber: ObservationSubscriber, *, fps: float) -> None:
        super().__init__(subscriber, fps=fps, track_role="depth")

    def _render_image(self) -> tuple[np.ndarray, str]:
        current = self._subscriber.current_frame
        if current is not None and current.depth_image_m is not None:
            return _render_depth_preview_image(current.depth_image_m), "rgb24"
        height, width = self._frame_shape()
        return _blank_rgb_frame(height, width), "rgb24"
