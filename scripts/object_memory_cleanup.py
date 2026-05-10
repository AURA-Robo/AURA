from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover - runtime environment guard.
    raise SystemExit("psycopg is required. Run this with the AURA system virtualenv.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from backend.webrtc.object_memory import (  # noqa: E402
    DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES,
    DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES,
)


def _csv_values(value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    normalized = str(value or "").strip()
    if not normalized:
        return default
    if normalized.lower() in {"*", "all", "any"}:
        return ("*",)
    return tuple(part.strip() for part in normalized.split(",") if part.strip())


def _normalize_classes(values: tuple[str, ...]) -> tuple[str, ...]:
    if any(value.strip().lower() in {"*", "all", "any"} for value in values):
        return ()
    return tuple(" ".join(value.strip().lower().split()) for value in values if value.strip())


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dry-run or stale-mark noisy active object-memory entries.",
    )
    parser.add_argument("--dsn", default=os.environ.get("AURA_OBJECT_MEMORY_DSN", ""))
    parser.add_argument("--user-id", default=os.environ.get("AURA_MEMORY_USER_ID", "local-operator"))
    parser.add_argument("--apply", action="store_true", help="Mark matching active rows as stale.")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--max-observation-count", type=int, default=1)
    parser.add_argument("--created-before-minutes", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-bbox-area", type=float, default=0.0005)
    parser.add_argument("--max-bbox-area", type=float, default=0.35)
    parser.add_argument(
        "--allowed-classes",
        default="",
        help="Comma-separated allowlist. Defaults to the runtime object-memory allowlist. Use 'all' to disable.",
    )
    parser.add_argument(
        "--blocked-classes",
        default="",
        help="Comma-separated blocklist. Defaults to known scene/background labels.",
    )
    parser.add_argument(
        "--allow-missing-world-pose",
        action="store_true",
        help="Do not stale-mark rows solely because world_pose_xyz is missing.",
    )
    return parser


def _candidate_params(args: argparse.Namespace) -> dict[str, Any]:
    allowed_classes = _normalize_classes(
        _csv_values(args.allowed_classes, default=DEFAULT_OBJECT_MEMORY_ALLOWED_CLASSES)
    )
    blocked_classes = _normalize_classes(
        _csv_values(args.blocked_classes, default=DEFAULT_OBJECT_MEMORY_BLOCKED_CLASSES)
    )
    return {
        "user_id": str(args.user_id).strip() or None,
        "limit": max(int(args.limit), 1),
        "max_observation_count": max(int(args.max_observation_count), 1),
        "created_before_minutes": max(int(args.created_before_minutes), 0),
        "min_confidence": max(float(args.min_confidence), 0.0),
        "min_bbox_area": max(float(args.min_bbox_area), 0.0),
        "max_bbox_area": max(float(args.max_bbox_area), 0.0),
        "allowed_classes": list(allowed_classes),
        "blocked_classes": list(blocked_classes),
        "has_allowed_classes": bool(allowed_classes),
        "has_blocked_classes": bool(blocked_classes),
        "require_world_pose": not bool(args.allow_missing_world_pose),
    }


def _candidate_cte() -> str:
    return """
        WITH candidates AS (
            SELECT object_id, canonical_class, last_detector_conf, last_box_area
            FROM object_memory_entries
            WHERE status = 'active'
              AND (%(user_id)s::text IS NULL OR user_id = %(user_id)s::text)
              AND observation_count <= %(max_observation_count)s
              AND appearance_count = 0
              AND COALESCE(metadata->>'last_track_id', '') = ''
              AND created_at <= now() - (%(created_before_minutes)s::integer * INTERVAL '1 minute')
              AND (
                    (%(has_blocked_classes)s AND lower(canonical_class) = ANY(%(blocked_classes)s::text[]))
                 OR (%(has_allowed_classes)s AND NOT (lower(canonical_class) = ANY(%(allowed_classes)s::text[])))
                 OR last_detector_conf < %(min_confidence)s
                 OR last_box_area < %(min_bbox_area)s
                 OR (%(max_bbox_area)s > 0 AND last_box_area > %(max_bbox_area)s)
                 OR (%(require_world_pose)s AND world_pose_xyz IS NULL)
              )
            ORDER BY updated_at DESC
            LIMIT %(limit)s
        )
    """


def _summarize(rows: list[dict[str, Any]], *, applied: bool) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    for row in rows:
        class_name = str(row.get("canonical_class") or "")
        class_counts[class_name] = class_counts.get(class_name, 0) + int(row.get("count", 1))
    matched_count = sum(class_counts.values())
    top_classes = [
        {"className": class_name, "count": count}
        for class_name, count in sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:20]
    ]
    return {
        "ok": True,
        "applied": bool(applied),
        "matchedCount": matched_count,
        "topClasses": top_classes,
    }


def _dry_run(conn, params: dict[str, Any]) -> dict[str, Any]:
    query = (
        _candidate_cte()
        + """
        SELECT canonical_class, COUNT(*) AS count
        FROM candidates
        GROUP BY canonical_class
        ORDER BY count DESC, canonical_class
        """
    )
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = [dict(row) for row in cur.fetchall()]
    return _summarize(rows, applied=False)


def _apply(conn, params: dict[str, Any]) -> dict[str, Any]:
    query = (
        _candidate_cte()
        + """
        UPDATE object_memory_entries AS entry
        SET status = 'stale',
            updated_at = now()
        FROM candidates
        WHERE entry.object_id = candidates.object_id
        RETURNING entry.canonical_class
        """
    )
    with conn.cursor() as cur:
        cur.execute(query, params)
        returned = [dict(row) for row in cur.fetchall()]
    conn.commit()
    return _summarize(returned, applied=True)


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    dsn = str(args.dsn or "").strip()
    if not dsn:
        raise SystemExit("--dsn or AURA_OBJECT_MEMORY_DSN is required.")
    params = _candidate_params(args)
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        summary = _apply(conn, params) if bool(args.apply) else _dry_run(conn, params)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
