"""Render and persist human-readable runtime context summaries."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any


SUMMARY_RELATIVE_PATH = Path("logs") / "runtime" / "runtime-context-summary.md"
RECENT_LOG_LIMIT = 10


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _format_timestamp(value: Any) -> str:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value)).astimezone().isoformat(timespec="seconds")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "n/a"


def _format_optional(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else "n/a"
    if isinstance(value, dict):
        return "n/a" if not value else _compact_json(value)
    if isinstance(value, list):
        return "n/a" if not value else _compact_json(value)
    return str(value)


def _service_line(name: str, payload: dict[str, Any]) -> str:
    status = _format_optional(payload.get("status"))
    health = _as_dict(payload.get("health"))
    detail = (
        health.get("error")
        or health.get("reason")
        or health.get("state")
        or health.get("message")
    )
    detail_text = _format_optional(detail)
    return f"- {name}: {status}" if detail_text == "n/a" else f"- {name}: {status} ({detail_text})"


def _log_line(payload: dict[str, Any]) -> str:
    level = _format_optional(payload.get("level"))
    source = _format_optional(payload.get("source"))
    message = _format_optional(payload.get("message"))
    timestamp_ns = payload.get("timestampNs")
    if isinstance(timestamp_ns, (int, float)):
        timestamp = _format_timestamp(float(timestamp_ns) / 1_000_000_000.0)
    else:
        timestamp = "n/a"
    return f"- {timestamp} [{level}] {source}: {message}"


def runtime_context_summary_path(root_dir: Path) -> Path:
    return Path(root_dir) / SUMMARY_RELATIVE_PATH


def summary_generated_at(state: dict[str, Any]) -> str:
    return _format_timestamp(state.get("timestamp", time.time()))


def render_runtime_context_summary(state: dict[str, Any]) -> str:
    session = _as_dict(state.get("session"))
    runtime = _as_dict(state.get("runtime"))
    services = _as_dict(state.get("services"))
    transport = _as_dict(state.get("transport"))
    sensors = _as_dict(state.get("sensors"))
    memory = _as_dict(state.get("memory"))
    selected_target = _as_dict(state.get("selectedTargetSummary"))
    last_event = _as_dict(session.get("lastEvent"))
    runtime_health = _as_dict(_as_dict(services.get("runtime")).get("health"))
    logs = _as_list(state.get("logs"))[-RECENT_LOG_LIMIT:]

    lines = [
        "# Runtime Context Summary",
        "",
        f"- Generated at: {summary_generated_at(state)}",
        f"- Session active: {_format_optional(session.get('active'))}",
        f"- Runtime owned by backend: {_format_optional(runtime_health.get('ownedByBackend'))}",
        "",
        "## Session",
        f"- Started at: {_format_optional(session.get('startedAt'))}",
        f"- Config: {_format_optional(session.get('config'))}",
        f"- Last event: {_format_optional(last_event.get('message'))}",
        "",
        "## Runtime execution",
        f"- Execution mode: {_format_optional(runtime.get('executionMode'))}",
        f"- Reasoning task status: {_format_optional(runtime.get('reasoningTaskStatus') or runtime.get('plannerControlMode'))}",
        f"- Reasoning route: {_format_optional(runtime.get('reasoningRoute'))}",
        f"- Active instruction: {_format_optional(runtime.get('activeInstruction'))}",
        f"- Current subgoal: {_format_optional(runtime.get('currentSubgoal'))}",
        f"- Route state: {_format_optional(runtime.get('routeState'))}",
        "",
        "## Service health",
    ]

    for key, label in (
        ("backend", "backend"),
        ("runtime", "runtime"),
        ("controlRuntime", "control_runtime"),
        ("reasoningSystem", "reasoning_system"),
        ("navigationSystem", "navigation_system"),
        ("navdp", "navdp"),
        ("system2", "system2"),
    ):
        service = _as_dict(services.get(key))
        if service:
            lines.append(_service_line(label, service))

    lines.extend(
        [
            "",
            "## Viewer and sensing",
            f"- Viewer enabled: {_format_optional(transport.get('viewerEnabled'))}",
            f"- Transport: {_format_optional(transport.get('transport'))}",
            f"- Frame available: {_format_optional(transport.get('frameAvailable'))}",
            f"- Frame age ms: {_format_optional(transport.get('frameAgeMs'))}",
            f"- RGB available: {_format_optional(sensors.get('rgbAvailable'))}",
            f"- Depth available: {_format_optional(sensors.get('depthAvailable'))}",
            f"- Selected target: {_format_optional(selected_target)}",
            "",
            "## Memory and knowledge",
            f"- Object memory enabled: {_format_optional(memory.get('objectMemoryEnabled'))}",
            f"- Object memory available: {_format_optional(memory.get('objectMemoryAvailable'))}",
            f"- Object count: {_format_optional(memory.get('objectCount'))}",
            f"- Observation count: {_format_optional(memory.get('observationCount'))}",
            f"- Knowledge enabled: {_format_optional(memory.get('knowledgeEnabled'))}",
            f"- Published documents: {_format_optional(memory.get('publishedDocumentCount'))}",
            f"- Active hard rules: {_format_optional(memory.get('activeHardRuleCount'))}",
            f"- Knowledge degraded reason: {_format_optional(memory.get('knowledgeDegradedReason'))}",
            "",
            "## Recent logs",
        ]
    )

    if logs:
        lines.extend(_log_line(_as_dict(item)) for item in logs)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def persist_runtime_context_summary(root_dir: Path, content: str) -> Path:
    target_path = runtime_context_summary_path(root_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target_path.stem}-",
        suffix=target_path.suffix,
        dir=str(target_path.parent),
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_path, target_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return target_path
