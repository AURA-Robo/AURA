from __future__ import annotations

import asyncio
import time
import uuid
from types import SimpleNamespace

import numpy as np

from backend.webrtc.config import WebRTCServiceConfig
from backend.webrtc.models import FrameCache, build_frame_meta_message
from backend.webrtc.subscriber import ObservationSubscriber
from systems.transport import FrameHeader, SharedMemoryRing, encode_ndarray, ref_to_dict


class _NoopBus:
    def poll(self, _topic: str, *, max_items: int = 0):  # noqa: ARG002
        return []

    def close(self) -> None:
        return None


class _SingleRecordBus:
    def __init__(self, record) -> None:  # noqa: ANN001
        self._record = record
        self._sent = False

    def poll(self, topic: str, *, max_items: int = 0):  # noqa: ARG002
        if topic != "isaac.observation" or self._sent:
            return []
        self._sent = True
        return [self._record]

    def close(self) -> None:
        return None


class _BatchRecordBus:
    def __init__(self, records) -> None:  # noqa: ANN001
        self._records = list(records)

    def poll(self, topic: str, *, max_items: int = 0):  # noqa: ARG002
        if topic != "isaac.observation" or not self._records:
            return []
        limit = len(self._records) if max_items <= 0 else min(max_items, len(self._records))
        batch = self._records[:limit]
        self._records = self._records[limit:]
        return batch

    def close(self) -> None:
        return None


class _PayloadShm:
    def read(self, ref):  # noqa: ANN001
        value = int(getattr(ref, "sequence", 1))
        return encode_ndarray(np.full((2, 2, 3), value, dtype=np.uint8))

    def close(self) -> None:
        return None


class _SlowMemorySink:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = float(delay_s)
        self.frames: list[int] = []

    def observe_frame(self, frame: FrameCache) -> None:
        time.sleep(self.delay_s)
        self.frames.append(int(frame.frame_header.frame_id))


class _OverwriteShm:
    def read(self, ref):  # noqa: ANN001, ARG002
        raise RuntimeError("Shared memory slot was overwritten before it was read.")

    def close(self) -> None:
        return None


def _make_frame(*, last_frame_monotonic: float) -> FrameCache:
    metadata = {
        "viewer_overlay": {
            "trajectory_pixels": [[10, 20], [30, 40]],
            "trajectoryPixels": [[10, 20], [30, 40]],
            "system2_pixel_goal": [111, 222],
            "system2PixelGoal": [111, 222],
            "active_target": {
                "className": "Navigation Goal",
                "source": "navigation",
                "nav_goal_pixel": [111, 222],
                "world_pose_xyz": [1.0, 2.0, 0.0],
            },
            "activeTarget": {
                "className": "Navigation Goal",
                "source": "navigation",
                "nav_goal_pixel": [111, 222],
                "world_pose_xyz": [1.0, 2.0, 0.0],
            },
        }
    }
    return FrameCache(
        seq=7,
        frame_header=FrameHeader(
            frame_id=12,
            timestamp_ns=123456789,
            source="perception_runtime",
            width=320,
            height=180,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(1.0, 2.0, 3.0),
            robot_yaw_rad=0.25,
            sim_time_s=4.5,
            metadata=metadata,
        ),
        rgb_image=np.zeros((180, 320, 3), dtype=np.uint8),
        depth_image_m=None,
        viewer_overlay=metadata["viewer_overlay"],
        last_frame_monotonic=last_frame_monotonic,
    )


def test_webrtc_subscriber_decodes_shared_memory_refs_only() -> None:
    shm_name = f"aura_viewer_test_{uuid.uuid4().hex[:12]}"
    writer = SharedMemoryRing(name=shm_name, slot_size=4096, capacity=2, create=True)
    reader = SharedMemoryRing(name=shm_name, slot_size=4096, capacity=2, create=False)
    subscriber = ObservationSubscriber(
        WebRTCServiceConfig(shm_name=shm_name, shm_slot_size=4096, shm_capacity=2),
        bus=_NoopBus(),
        shm_ring=reader,
    )
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.linspace(0.0, 1.0, 4, dtype=np.float32).reshape(2, 2)
    try:
        rgb_ref = writer.write(encode_ndarray(rgb))
        depth_ref = writer.write(encode_ndarray(depth))
        inline_rgb = encode_ndarray(rgb).hex()
        inline_depth = encode_ndarray(depth).hex()

        decoded_rgb = subscriber._decode_rgb({"rgb_ref": ref_to_dict(rgb_ref), "rgb_inline": inline_rgb})
        decoded_depth = subscriber._decode_depth({"depth_ref": ref_to_dict(depth_ref), "depth_inline": inline_depth})

        assert decoded_rgb is not None
        assert decoded_depth is not None
        assert np.array_equal(decoded_rgb, rgb)
        assert np.array_equal(decoded_depth, depth)
        assert subscriber._decode_rgb({"rgb_inline": inline_rgb}) is None
        assert subscriber._decode_depth({"depth_inline": inline_depth}) is None
    finally:
        reader.close()
        writer.close(unlink=True)


def test_webrtc_subscriber_attaches_to_ref_shm_name() -> None:
    config_name = f"aura_viewer_config_{uuid.uuid4().hex[:12]}"
    writer_name = f"aura_viewer_writer_{uuid.uuid4().hex[:12]}"
    writer = SharedMemoryRing(name=writer_name, slot_size=4096, capacity=2, create=True)
    subscriber = ObservationSubscriber(
        WebRTCServiceConfig(shm_name=config_name, shm_slot_size=4096, shm_capacity=2),
        bus=_NoopBus(),
    )
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    try:
        rgb_ref = writer.write(encode_ndarray(rgb))
        decoded_rgb = subscriber._decode_rgb({"rgb_ref": ref_to_dict(rgb_ref)})

        assert decoded_rgb is not None
        assert np.array_equal(decoded_rgb, rgb)
        assert subscriber._shm_ring is not None
        assert subscriber._shm_ring.name == writer.name
    finally:
        asyncio.run(subscriber.close())
        writer.close(unlink=True)


def test_webrtc_frame_meta_preserves_overlay_contract_keys() -> None:
    overlay = {
        "detections": [
            {
                "class_id": 821,
                "class_name": "chair",
                "track_id": "track-1",
                "bbox_xyxy": [10, 20, 110, 160],
                "confidence": 0.93,
            }
        ],
        "trajectory_pixels": [[10, 20], [30, 40]],
        "system2_pixel_goal": [111, 222],
        "active_target": {
            "className": "Navigation Goal",
            "source": "navigation",
            "nav_goal_pixel": [111, 222],
            "world_pose_xyz": [1.0, 2.0, 0.0],
        },
    }
    metadata = {
        **overlay,
        "viewer_overlay": dict(overlay),
    }
    frame = FrameCache(
        seq=7,
        frame_header=FrameHeader(
            frame_id=12,
            timestamp_ns=123456789,
            source="perception_runtime",
            width=320,
            height=180,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(1.0, 2.0, 3.0),
            robot_yaw_rad=0.25,
            sim_time_s=4.5,
            metadata=metadata,
        ),
        rgb_image=np.zeros((180, 320, 3), dtype=np.uint8),
        depth_image_m=None,
        viewer_overlay=metadata["viewer_overlay"],
        last_frame_monotonic=time.monotonic(),
    )

    payload = build_frame_meta_message(frame)

    assert payload["trajectory_pixels"] == [[10, 20], [30, 40]]
    assert payload["trajectoryPixels"] == [[10, 20], [30, 40]]
    assert payload["system2_pixel_goal"] == [111, 222]
    assert payload["system2PixelGoal"] == [111, 222]
    assert payload["active_target"]["nav_goal_pixel"] == [111, 222]
    assert payload["activeTarget"]["world_pose_xyz"] == [1.0, 2.0, 0.0]
    assert payload["detections"][0]["class_id"] == 821


def test_webrtc_subscriber_keeps_last_good_payload_and_marks_stream_stalled() -> None:
    subscriber = ObservationSubscriber(
        WebRTCServiceConfig(stale_frame_timeout_sec=0.05),
        bus=_NoopBus(),
        shm_ring=None,
    )
    now = time.monotonic()
    subscriber._frame = _make_frame(last_frame_monotonic=now)

    fresh_state = subscriber.build_state_snapshot()
    fresh_meta = subscriber.build_frame_meta()

    assert fresh_state["type"] == "frame_state"
    assert fresh_meta is not None
    assert fresh_state["streamStalled"] is False
    assert fresh_meta["streamStalled"] is False

    subscriber._frame = _make_frame(last_frame_monotonic=time.monotonic() - 0.10)

    stale_state = subscriber.build_state_snapshot()
    stale_meta = subscriber.build_frame_meta()

    assert stale_state["type"] == "frame_state"
    assert stale_state["frame_id"] == fresh_state["frame_id"]
    assert stale_state["streamStalled"] is True
    assert stale_state["lastGoodFrameAgeMs"] is not None
    assert stale_meta is not None
    assert stale_meta["frame_id"] == fresh_meta["frame_id"]
    assert stale_meta["streamStalled"] is True
    assert subscriber.debug_counters["staleTransitions"] >= 1


def test_webrtc_subscriber_ignores_transient_shm_overwrite_errors() -> None:
    async def scenario() -> None:
        header = FrameHeader(
            frame_id=1,
            timestamp_ns=123,
            source="runtime",
            width=2,
            height=2,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(0.0, 0.0, 0.0),
            robot_yaw_rad=0.0,
            sim_time_s=0.0,
            metadata={
                "rgb_ref": {
                    "name": "aura_viewer_shm_01",
                    "slot_index": 0,
                    "payload_size": 16,
                    "sequence": 1,
                }
            },
        )
        subscriber = ObservationSubscriber(
            WebRTCServiceConfig(poll_interval_ms=1),
            bus=_SingleRecordBus(SimpleNamespace(message=header)),
            shm_ring=_OverwriteShm(),
        )
        await subscriber.start()
        await asyncio.sleep(0.03)
        assert subscriber._task is not None
        assert subscriber._task.done() is False
        assert subscriber.current_frame is None
        assert subscriber.debug_counters["decodeDrops"] == 1
        assert subscriber.debug_counters["shmOverwriteDrops"] == 1
        await subscriber.close()

    asyncio.run(scenario())


def _record(frame_id: int):
    return SimpleNamespace(
        message=FrameHeader(
            frame_id=frame_id,
            timestamp_ns=frame_id,
            source="runtime",
            width=2,
            height=2,
            rgb_encoding="rgb8",
            depth_encoding="",
            camera_pose_xyz=(0.0, 0.0, 0.0),
            camera_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            robot_pose_xyz=(0.0, 0.0, 0.0),
            robot_yaw_rad=0.0,
            sim_time_s=0.0,
            metadata={
                "rgb_ref": {
                    "name": "aura_viewer_shm_01",
                    "slot_index": 0,
                    "payload_size": 16,
                    "sequence": frame_id,
                }
            },
        )
    )


def test_webrtc_subscriber_decodes_only_latest_observation_batch() -> None:
    async def scenario() -> None:
        subscriber = ObservationSubscriber(
            WebRTCServiceConfig(poll_interval_ms=1),
            bus=_BatchRecordBus([_record(1), _record(2), _record(3)]),
            shm_ring=_PayloadShm(),
        )
        await subscriber.start()
        await asyncio.sleep(0.03)

        assert subscriber.current_frame is not None
        assert subscriber.current_frame.frame_header.frame_id == 3
        assert subscriber.debug_counters["decodeOk"] == 1
        assert subscriber.debug_counters["latestFrameDrops"] == 2

        await subscriber.close()

    asyncio.run(scenario())


def test_object_memory_sink_runs_off_the_subscriber_poll_loop() -> None:
    async def scenario() -> None:
        sink = _SlowMemorySink(delay_s=0.08)
        subscriber = ObservationSubscriber(
            WebRTCServiceConfig(poll_interval_ms=1, object_memory_queue_size=2),
            bus=_BatchRecordBus([_record(1)]),
            shm_ring=_PayloadShm(),
            object_memory_sink=sink,
        )
        started_at = time.perf_counter()
        await subscriber.start()
        await asyncio.sleep(0.02)
        elapsed = time.perf_counter() - started_at

        assert elapsed < 0.06
        assert subscriber.current_frame is not None
        assert subscriber.current_frame.frame_header.frame_id == 1
        assert subscriber.debug_counters["objectMemoryQueued"] == 1

        await subscriber.close()

    asyncio.run(scenario())
