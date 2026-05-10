"""Backend entrypoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from aiohttp import web

from backend.app import create_app
from backend.webrtc import WebRTCServiceConfig
from systems.shared.contracts.service_endpoints import (
    BACKEND_ENDPOINT,
    CONTROL_RUNTIME_ENDPOINT,
    INFERENCE_SYSTEM_ENDPOINT,
    NAVIGATION_SYSTEM_ENDPOINT,
    REASONING_SYSTEM_ENDPOINT,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
TRUE_VALUES = frozenset(("1", "true", "yes", "on"))


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in TRUE_VALUES


def _env_float(name: str, *, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return float(default)
    try:
        return float(str(raw_value).strip())
    except ValueError:
        return float(default)


def _env_int(name: str, *, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return int(default)
    try:
        return int(str(raw_value).strip())
    except ValueError:
        return int(default)


def _csv_values(value: str | None) -> tuple[str, ...]:
    normalized = str(value or "").strip()
    if not normalized:
        return ()
    return tuple(part.strip() for part in normalized.split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the AURA backend.")
    parser.add_argument("--host", default=BACKEND_ENDPOINT.host)
    parser.add_argument("--port", type=int, default=BACKEND_ENDPOINT.port)
    parser.add_argument("--dev-origin", default="http://127.0.0.1:5173")
    parser.add_argument("--api-base-url", default=BACKEND_ENDPOINT.base_url())
    parser.add_argument(
        "--runtime-url",
        "--runtime-supervisor-url",
        dest="runtime_url",
        default="",
        help="Optional external runtime URL. Leave unset to let the backend own runtime lifecycle locally.",
    )
    parser.add_argument("--inference-system-url", default=INFERENCE_SYSTEM_ENDPOINT.base_url())
    parser.add_argument(
        "--reasoning-system-url",
        "--planner-system-url",
        dest="reasoning_system_url",
        default=REASONING_SYSTEM_ENDPOINT.base_url(),
    )
    parser.add_argument("--navigation-system-url", default=NAVIGATION_SYSTEM_ENDPOINT.base_url())
    parser.add_argument("--control-runtime-url", default=CONTROL_RUNTIME_ENDPOINT.base_url())
    parser.add_argument("--webrtc-proxy-base", default="")
    parser.add_argument("--webrtc-rgb-fps", type=float, default=30.0)
    parser.add_argument("--webrtc-depth-fps", type=float, default=15.0)
    parser.add_argument("--webrtc-telemetry-hz", type=float, default=10.0)
    parser.add_argument("--webrtc-poll-interval-ms", type=int, default=10)
    parser.add_argument(
        "--webrtc-latest-frame-drain-batches",
        type=int,
        default=_env_int("AURA_WEBRTC_LATEST_FRAME_DRAIN_BATCHES", default=8),
        help="Maximum ZMQ observation batches to drain while keeping only the newest viewer frame.",
    )
    parser.add_argument(
        "--webrtc-object-memory-queue-size",
        type=int,
        default=_env_int("AURA_WEBRTC_OBJECT_MEMORY_QUEUE_SIZE", default=8),
        help="Bounded async queue size for object-memory ingestion from WebRTC frames.",
    )
    parser.add_argument("--object-memory-dsn", default=os.environ.get("AURA_OBJECT_MEMORY_DSN", ""))
    parser.add_argument(
        "--object-memory-event-log-path",
        default=os.environ.get("AURA_OBJECT_MEMORY_EVENT_LOG_PATH", ""),
        help="Optional JSONL path for YOLO/object-memory detection events.",
    )
    parser.add_argument("--object-memory-user-id", default=os.environ.get("AURA_MEMORY_USER_ID", "local-operator"))
    parser.add_argument(
        "--object-memory-min-confidence",
        type=float,
        default=_env_float("AURA_OBJECT_MEMORY_MIN_CONFIDENCE", default=0.80),
        help="Minimum detector confidence required before a WebRTC detection can enter object memory.",
    )
    parser.add_argument(
        "--object-memory-min-bbox-area",
        type=float,
        default=_env_float("AURA_OBJECT_MEMORY_MIN_BBOX_AREA", default=0.0005),
        help="Minimum normalized bbox area required before a WebRTC detection can enter object memory.",
    )
    parser.add_argument(
        "--object-memory-max-bbox-area",
        type=float,
        default=_env_float("AURA_OBJECT_MEMORY_MAX_BBOX_AREA", default=0.35),
        help="Maximum normalized bbox area allowed before a detection is treated as a scene label.",
    )
    parser.add_argument(
        "--object-memory-allowed-classes",
        default=os.environ.get("AURA_OBJECT_MEMORY_ALLOWED_CLASSES", ""),
        help="Comma-separated class allowlist. Use 'all' to allow every class except blocked classes.",
    )
    parser.add_argument(
        "--object-memory-blocked-classes",
        default=os.environ.get("AURA_OBJECT_MEMORY_BLOCKED_CLASSES", ""),
        help="Comma-separated class blocklist for object-memory ingestion.",
    )
    parser.add_argument(
        "--object-memory-auto-migrate",
        dest="object_memory_auto_migrate",
        action="store_true",
        help="Apply/verify the Postgres object memory schema before enabling DSN-backed memory.",
    )
    parser.add_argument(
        "--object-memory-no-auto-migrate",
        dest="object_memory_auto_migrate",
        action="store_false",
        help="Only verify an existing Postgres object memory schema.",
    )
    parser.add_argument(
        "--knowledge-dsn",
        default=os.environ.get("AURA_KNOWLEDGE_DSN", os.environ.get("AURA_OBJECT_MEMORY_DSN", "")),
    )
    parser.add_argument(
        "--agent-memory-dsn",
        default=os.environ.get("AURA_AGENT_MEMORY_DSN", os.environ.get("AURA_OBJECT_MEMORY_DSN", "")),
    )
    parser.add_argument(
        "--planner-catalog-dsn",
        default=os.environ.get(
            "AURA_PLANNER_CATALOG_DSN",
            os.environ.get("AURA_KNOWLEDGE_DSN", os.environ.get("AURA_OBJECT_MEMORY_DSN", "")),
        ),
    )
    parser.add_argument("--webrtc-enable-depth-track", dest="webrtc_enable_depth_track", action="store_true")
    parser.add_argument("--webrtc-disable-depth-track", dest="webrtc_enable_depth_track", action="store_false")
    parser.add_argument(
        "--object-memory-require-world-pose",
        dest="object_memory_require_world_pose",
        action="store_true",
        help="Require world_pose_xyz before storing WebRTC object-memory detections.",
    )
    parser.add_argument(
        "--object-memory-allow-missing-world-pose",
        dest="object_memory_require_world_pose",
        action="store_false",
        help="Allow detections without world_pose_xyz into object memory.",
    )
    parser.set_defaults(
        object_memory_auto_migrate=_env_flag("AURA_OBJECT_MEMORY_AUTO_MIGRATE"),
        object_memory_require_world_pose=_env_flag("AURA_OBJECT_MEMORY_REQUIRE_WORLD_POSE", default=True),
        webrtc_enable_depth_track=False,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    app = create_app(
        root_dir=str(REPO_ROOT),
        api_base_url=str(args.api_base_url).rstrip("/"),
        dev_origin=str(args.dev_origin),
        runtime_url=str(args.runtime_url).rstrip("/"),
        inference_system_url=str(args.inference_system_url).rstrip("/"),
        reasoning_system_url=str(args.reasoning_system_url).rstrip("/"),
        navigation_system_url=str(args.navigation_system_url).rstrip("/"),
        control_runtime_url=str(args.control_runtime_url).rstrip("/"),
        knowledge_dsn=str(args.knowledge_dsn),
        agent_memory_dsn=str(args.agent_memory_dsn),
        planner_catalog_dsn=str(args.planner_catalog_dsn),
        object_memory_dsn=str(args.object_memory_dsn),
        webrtc_proxy_base=str(args.webrtc_proxy_base).rstrip("/"),
        webrtc_config=WebRTCServiceConfig(
            enable_depth_track=bool(args.webrtc_enable_depth_track),
            rgb_fps=float(args.webrtc_rgb_fps),
            depth_fps=float(args.webrtc_depth_fps),
            telemetry_hz=float(args.webrtc_telemetry_hz),
            poll_interval_ms=int(args.webrtc_poll_interval_ms),
            latest_frame_drain_batches=int(args.webrtc_latest_frame_drain_batches),
            object_memory_queue_size=int(args.webrtc_object_memory_queue_size),
            object_memory_dsn=str(args.object_memory_dsn),
            object_memory_event_log_path=str(args.object_memory_event_log_path),
            object_memory_user_id=str(args.object_memory_user_id),
            object_memory_auto_migrate=bool(args.object_memory_auto_migrate),
            object_memory_min_detector_confidence=float(args.object_memory_min_confidence),
            object_memory_min_bbox_area_norm=float(args.object_memory_min_bbox_area),
            object_memory_max_bbox_area_norm=float(args.object_memory_max_bbox_area),
            object_memory_require_world_pose=bool(args.object_memory_require_world_pose),
            object_memory_allowed_classes=_csv_values(args.object_memory_allowed_classes),
            object_memory_blocked_classes=_csv_values(args.object_memory_blocked_classes),
        ),
    )
    web.run_app(app, host=str(args.host), port=int(args.port))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
