from __future__ import annotations

import time
from typing import Any

from .object_memory_models import ObjectMemoryContext, RetrievedObjectMemory


def object_entries_to_recent_seen(
    entries: list[RetrievedObjectMemory],
    *,
    top_k_per_room: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts: dict[tuple[str, str | None], int] = {}
    for entry in sorted(entries, key=lambda row: row.last_seen_at, reverse=True):
        key = (entry.canonical_class, entry.room_id)
        current = counts.get(key, 0)
        if current >= top_k_per_room:
            continue
        counts[key] = current + 1
        rows.append(
            {
                "class": entry.canonical_class,
                "room": entry.room_id,
                "scene_scope": entry.scene_scope,
                "age_sec": max(0, int(time.time() - entry.last_seen_at.timestamp())),
                "object_id": entry.object_id,
                "last_seen_ts": entry.last_seen_at.timestamp(),
                "dedupe_confidence": entry.dedupe_confidence,
                "detector_conf": entry.last_detector_conf,
                "world_pose_xyz": None if entry.world_pose_xyz is None else list(entry.world_pose_xyz),
            }
        )
    return rows


def inject_object_memory_context_into_plan_request(
    request: dict[str, Any],
    context: ObjectMemoryContext,
) -> dict[str, Any]:
    world_summary = dict(request.get("world_summary", {}))
    existing_rows = list(world_summary.get("recent_seen", []))
    merged_rows = [*existing_rows, *context.recent_seen]

    best_rows: dict[tuple[str | None, str | None], dict[str, Any]] = {}
    for row in merged_rows:
        key = (row.get("class"), row.get("room"))
        current = best_rows.get(key)
        if current is None:
            best_rows[key] = row
            continue
        current_age = current.get("age_sec")
        next_age = row.get("age_sec")
        if isinstance(next_age, int) and (not isinstance(current_age, int) or next_age < current_age):
            best_rows[key] = row
            continue
        current_seen = current.get("last_seen_ts")
        next_seen = row.get("last_seen_ts")
        if isinstance(next_seen, (int, float)) and (
            not isinstance(current_seen, (int, float)) or float(next_seen) > float(current_seen)
        ):
            best_rows[key] = row

    world_summary["recent_seen"] = list(best_rows.values())
    return {
        **request,
        "world_summary": world_summary,
    }
