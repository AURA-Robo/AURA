from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .object_memory_models import (
    ObjectMemoryEmbedding,
    ObjectMemoryEntry,
    ObjectObservation,
)

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional runtime dependency.
    psycopg = None
    dict_row = None


OBJECT_MEMORY_SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "object_memory_schema.sql"
REQUIRED_OBJECT_MEMORY_TABLES = (
    "object_memory_entries",
    "object_memory_observations",
    "object_memory_entry_embeddings",
)


def _vector_literal(values: list[float] | None) -> str | None:
    if not values:
        return None
    return "[" + ",".join(f"{value:.10f}" for value in values) + "]"


def _vector_values(value: object) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().removeprefix("[").removesuffix("]").strip()
        if normalized == "":
            return []
        return [float(item) for item in normalized.split(",")]
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return None


def _xyz_values(value: object) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        candidates = (value.get("x"), value.get("y"), value.get("z"))
    elif isinstance(value, (list, tuple)):
        if len(value) < 3:
            return None
        candidates = (value[0], value[1], value[2])
    else:
        return None
    try:
        return (float(candidates[0]), float(candidates[1]), float(candidates[2]))
    except (TypeError, ValueError):
        return None


class ObjectMemoryRepository(Protocol):
    def insert_object_entry(self, entry: ObjectMemoryEntry) -> None: ...

    def update_object_entry(self, entry: ObjectMemoryEntry) -> None: ...

    def list_object_entries(
        self,
        user_id: str | None,
        *,
        statuses: Sequence[str] | None = None,
        class_name: str | None = None,
        room_id: str | None = None,
        scene_scope: str | None = None,
        object_ids: Sequence[str] | None = None,
        updated_since: datetime | None = None,
        top_k: int | None = None,
    ) -> list[ObjectMemoryEntry]: ...

    def count_object_entries(
        self,
        user_id: str | None,
        *,
        statuses: Sequence[str] | None = None,
    ) -> int: ...

    def insert_object_observation(self, observation: ObjectObservation) -> None: ...

    def list_object_observations(
        self,
        user_id: str | None,
        *,
        object_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[ObjectObservation]: ...

    def upsert_object_embedding(self, embedding: ObjectMemoryEmbedding) -> None: ...

    def get_object_embedding(self, object_id: str) -> ObjectMemoryEmbedding | None: ...


class InMemoryObjectMemoryRepository:
    def __init__(self) -> None:
        self._entries: dict[str, ObjectMemoryEntry] = {}
        self._observations: dict[str, ObjectObservation] = {}
        self._embeddings: dict[str, ObjectMemoryEmbedding] = {}

    def insert_object_entry(self, entry: ObjectMemoryEntry) -> None:
        self._entries[entry.object_id] = entry

    def update_object_entry(self, entry: ObjectMemoryEntry) -> None:
        self._entries[entry.object_id] = entry

    def list_object_entries(
        self,
        user_id: str | None,
        *,
        statuses: Sequence[str] | None = None,
        class_name: str | None = None,
        room_id: str | None = None,
        scene_scope: str | None = None,
        object_ids: Sequence[str] | None = None,
        updated_since: datetime | None = None,
        top_k: int | None = None,
    ) -> list[ObjectMemoryEntry]:
        status_set = set(statuses) if statuses is not None else None
        object_id_set = set(object_ids) if object_ids is not None else None
        rows = [
            row
            for row in self._entries.values()
            if (user_id is None or row.user_id == user_id)
            and (status_set is None or row.status in status_set)
            and (class_name is None or row.canonical_class == class_name)
            and (room_id is None or row.room_id == room_id)
            and (scene_scope is None or row.scene_scope == scene_scope)
            and (object_id_set is None or row.object_id in object_id_set)
            and (updated_since is None or row.last_seen_at >= updated_since)
        ]
        rows.sort(key=lambda row: row.last_seen_at, reverse=True)
        return rows if top_k is None else rows[:top_k]

    def count_object_entries(
        self,
        user_id: str | None,
        *,
        statuses: Sequence[str] | None = None,
    ) -> int:
        return len(self.list_object_entries(user_id, statuses=statuses))

    def insert_object_observation(self, observation: ObjectObservation) -> None:
        self._observations[observation.observation_id] = observation

    def list_object_observations(
        self,
        user_id: str | None,
        *,
        object_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[ObjectObservation]:
        object_id_set = set(object_ids) if object_ids is not None else None
        rows = [
            row
            for row in self._observations.values()
            if (user_id is None or row.user_id == user_id)
            and (object_id_set is None or row.object_id in object_id_set)
        ]
        rows.sort(key=lambda row: row.observed_at, reverse=True)
        return rows if top_k is None else rows[:top_k]

    def upsert_object_embedding(self, embedding: ObjectMemoryEmbedding) -> None:
        self._embeddings[embedding.object_id] = embedding

    def get_object_embedding(self, object_id: str) -> ObjectMemoryEmbedding | None:
        return self._embeddings.get(object_id)


class PostgresObjectMemoryRepository:
    def __init__(self, dsn: str, *, connect_timeout_s: float = 5.0) -> None:
        normalized_dsn = str(dsn).strip()
        if not normalized_dsn:
            raise RuntimeError("pg-dsn is required for PostgresObjectMemoryRepository.")
        if psycopg is None:
            raise RuntimeError("psycopg is required to use PostgresObjectMemoryRepository.")
        self.dsn = normalized_dsn
        self.connect_timeout_s = max(float(connect_timeout_s), 1.0)

    def _connect(self, *, autocommit: bool = False):
        return psycopg.connect(
            self.dsn,
            row_factory=dict_row,
            autocommit=autocommit,
            connect_timeout=int(round(self.connect_timeout_s)),
        )

    def apply_schema(self, schema_sql: str | None = None) -> None:
        sql_text = schema_sql if schema_sql is not None else OBJECT_MEMORY_SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect(autocommit=True) as conn:
            conn.execute(sql_text)

    def verify_schema(self) -> None:
        status = self.schema_status()
        if not status["vector_extension_installed"]:
            raise RuntimeError("Postgres object memory schema is missing required pgvector extension.")
        missing_tables = list(status["missing_tables"])
        if missing_tables:
            missing = ", ".join(missing_tables)
            raise RuntimeError(f"Postgres object memory schema is incomplete; missing tables: {missing}.")

    def schema_status(self) -> dict[str, object]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS installed")
            vector_row = cur.fetchone()
            vector_installed = bool(vector_row and vector_row["installed"])
            cur.execute(
                """
                SELECT table_name, to_regclass(table_name) IS NOT NULL AS present
                FROM unnest(%s::text[]) AS table_name
                """,
                (list(REQUIRED_OBJECT_MEMORY_TABLES),),
            )
            rows = cur.fetchall()
        table_status = {str(row["table_name"]): bool(row["present"]) for row in rows}
        return {
            "vector_extension_installed": vector_installed,
            "tables": table_status,
            "missing_tables": [name for name in REQUIRED_OBJECT_MEMORY_TABLES if not table_status.get(name, False)],
        }

    def insert_object_entry(self, entry: ObjectMemoryEntry) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO object_memory_entries (
                    object_id, user_id, canonical_class, room_id, status,
                    scene_scope, world_pose_xyz, world_pose_observed_at,
                    first_seen_at, last_seen_at, observation_count, last_source_id,
                    last_session_id, last_bbox_xyxy_norm, last_box_area,
                    last_aspect_ratio, last_detector_conf, appearance_count,
                    dedupe_confidence, metadata, created_at, updated_at
                )
                VALUES (
                    %(object_id)s, %(user_id)s, %(canonical_class)s, %(room_id)s, %(status)s,
                    %(scene_scope)s, %(world_pose_xyz)s::jsonb, %(world_pose_observed_at)s,
                    %(first_seen_at)s, %(last_seen_at)s, %(observation_count)s, %(last_source_id)s,
                    %(last_session_id)s, %(last_bbox_xyxy_norm)s::jsonb, %(last_box_area)s,
                    %(last_aspect_ratio)s, %(last_detector_conf)s, %(appearance_count)s,
                    %(dedupe_confidence)s, %(metadata)s::jsonb, %(created_at)s, %(updated_at)s
                )
                """,
                {
                    **entry.__dict__,
                    "last_bbox_xyxy_norm": json.dumps(entry.last_bbox_xyxy_norm),
                    "world_pose_xyz": json.dumps(entry.world_pose_xyz) if entry.world_pose_xyz is not None else None,
                    "metadata": json.dumps(entry.metadata, ensure_ascii=False),
                },
            )

    def update_object_entry(self, entry: ObjectMemoryEntry) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE object_memory_entries
                SET room_id = %(room_id)s,
                    scene_scope = %(scene_scope)s,
                    status = %(status)s,
                    last_seen_at = %(last_seen_at)s,
                    observation_count = %(observation_count)s,
                    last_source_id = %(last_source_id)s,
                    last_session_id = %(last_session_id)s,
                    last_bbox_xyxy_norm = %(last_bbox_xyxy_norm)s::jsonb,
                    last_box_area = %(last_box_area)s,
                    last_aspect_ratio = %(last_aspect_ratio)s,
                    last_detector_conf = %(last_detector_conf)s,
                    world_pose_xyz = %(world_pose_xyz)s::jsonb,
                    world_pose_observed_at = %(world_pose_observed_at)s,
                    appearance_count = %(appearance_count)s,
                    dedupe_confidence = %(dedupe_confidence)s,
                    metadata = %(metadata)s::jsonb,
                    updated_at = %(updated_at)s
                WHERE object_id = %(object_id)s
                """,
                {
                    **entry.__dict__,
                    "last_bbox_xyxy_norm": json.dumps(entry.last_bbox_xyxy_norm),
                    "world_pose_xyz": json.dumps(entry.world_pose_xyz) if entry.world_pose_xyz is not None else None,
                    "metadata": json.dumps(entry.metadata, ensure_ascii=False),
                },
            )

    def list_object_entries(
        self,
        user_id: str | None,
        *,
        statuses: Sequence[str] | None = None,
        class_name: str | None = None,
        room_id: str | None = None,
        scene_scope: str | None = None,
        object_ids: Sequence[str] | None = None,
        updated_since: datetime | None = None,
        top_k: int | None = None,
    ) -> list[ObjectMemoryEntry]:
        clauses: list[str] = []
        params: dict[str, object] = {}
        if user_id is not None:
            clauses.append("user_id = %(user_id)s")
            params["user_id"] = user_id
        if statuses is not None:
            clauses.append("status = ANY(%(statuses)s)")
            params["statuses"] = list(statuses)
        if class_name is not None:
            clauses.append("canonical_class = %(canonical_class)s")
            params["canonical_class"] = class_name
        if room_id is not None:
            clauses.append("room_id = %(room_id)s")
            params["room_id"] = room_id
        if scene_scope is not None:
            clauses.append("scene_scope = %(scene_scope)s")
            params["scene_scope"] = scene_scope
        if object_ids is not None:
            clauses.append("object_id = ANY(%(object_ids)s)")
            params["object_ids"] = list(object_ids)
        if updated_since is not None:
            clauses.append("last_seen_at >= %(updated_since)s")
            params["updated_since"] = updated_since
        query = f"""
            SELECT *
            FROM object_memory_entries
            {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
            ORDER BY last_seen_at DESC
            {f'LIMIT {int(top_k)}' if top_k is not None else ''}
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [self._entry_from_row(row) for row in rows]

    def count_object_entries(
        self,
        user_id: str | None,
        *,
        statuses: Sequence[str] | None = None,
    ) -> int:
        clauses: list[str] = []
        params: dict[str, object] = {}
        if user_id is not None:
            clauses.append("user_id = %(user_id)s")
            params["user_id"] = user_id
        if statuses is not None:
            clauses.append("status = ANY(%(statuses)s)")
            params["statuses"] = list(statuses)
        query = f"""
            SELECT COUNT(*) AS count
            FROM object_memory_entries
            {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
        return 0 if row is None else int(row["count"])

    def insert_object_observation(self, observation: ObjectObservation) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO object_memory_observations (
                    observation_id, object_id, user_id, session_id, source_id,
                    frame_idx, observed_at, track_id, class_name, detector_conf,
                    room_id, scene_scope, bbox_xyxy_norm, box_area, aspect_ratio, mask_area,
                    world_pose_xyz, world_pose_observed_at,
                    appearance_embedding, appearance_model, image_hash, attributes
                )
                VALUES (
                    %(observation_id)s, %(object_id)s, %(user_id)s, %(session_id)s, %(source_id)s,
                    %(frame_idx)s, %(observed_at)s, %(track_id)s, %(class_name)s, %(detector_conf)s,
                    %(room_id)s, %(scene_scope)s, %(bbox_xyxy_norm)s::jsonb, %(box_area)s, %(aspect_ratio)s, %(mask_area)s,
                    %(world_pose_xyz)s::jsonb, %(world_pose_observed_at)s,
                    %(appearance_embedding)s::vector, %(appearance_model)s, %(image_hash)s, %(attributes)s::jsonb
                )
                """,
                {
                    **observation.__dict__,
                    "bbox_xyxy_norm": json.dumps(observation.bbox_xyxy_norm),
                    "world_pose_xyz": json.dumps(observation.world_pose_xyz) if observation.world_pose_xyz is not None else None,
                    "appearance_embedding": _vector_literal(observation.appearance_embedding),
                    "attributes": json.dumps(observation.attributes, ensure_ascii=False),
                },
            )

    def list_object_observations(
        self,
        user_id: str | None,
        *,
        object_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[ObjectObservation]:
        clauses: list[str] = []
        params: dict[str, object] = {}
        if user_id is not None:
            clauses.append("user_id = %(user_id)s")
            params["user_id"] = user_id
        if object_ids is not None:
            clauses.append("object_id = ANY(%(object_ids)s)")
            params["object_ids"] = list(object_ids)
        query = f"""
            SELECT *
            FROM object_memory_observations
            {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
            ORDER BY observed_at DESC
            {f'LIMIT {int(top_k)}' if top_k is not None else ''}
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [self._observation_from_row(row) for row in rows]

    def upsert_object_embedding(self, embedding: ObjectMemoryEmbedding) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO object_memory_entry_embeddings (
                    object_id, user_id, model_name, embedding, index_status, embedded_at
                )
                VALUES (
                    %(object_id)s, %(user_id)s, %(model_name)s, %(embedding)s::vector, %(index_status)s, %(embedded_at)s
                )
                ON CONFLICT (object_id) DO UPDATE SET
                    user_id = EXCLUDED.user_id,
                    model_name = EXCLUDED.model_name,
                    embedding = EXCLUDED.embedding,
                    index_status = EXCLUDED.index_status,
                    embedded_at = EXCLUDED.embedded_at
                """,
                {
                    **embedding.__dict__,
                    "embedding": _vector_literal(embedding.embedding),
                },
            )

    def get_object_embedding(self, object_id: str) -> ObjectMemoryEmbedding | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM object_memory_entry_embeddings WHERE object_id = %s", (object_id,))
            row = cur.fetchone()
        if row is None:
            return None
        vector = _vector_values(row.get("embedding"))
        return ObjectMemoryEmbedding(
            object_id=str(row["object_id"]),
            user_id=row["user_id"],
            model_name=row["model_name"],
            embedding=vector,
            index_status=row["index_status"],
            embedded_at=row["embedded_at"],
        )

    def _entry_from_row(self, row: dict[str, object]) -> ObjectMemoryEntry:
        return ObjectMemoryEntry(
            object_id=str(row["object_id"]),
            user_id=row["user_id"],
            canonical_class=row["canonical_class"],
            room_id=row["room_id"],
            scene_scope=row.get("scene_scope"),
            status=row["status"],
            first_seen_at=row["first_seen_at"],
            last_seen_at=row["last_seen_at"],
            observation_count=int(row["observation_count"]),
            last_source_id=row["last_source_id"],
            last_session_id=row["last_session_id"],
            last_bbox_xyxy_norm=tuple(float(value) for value in row["last_bbox_xyxy_norm"]),
            last_box_area=float(row["last_box_area"]),
            last_aspect_ratio=float(row["last_aspect_ratio"]),
            last_detector_conf=float(row["last_detector_conf"]),
            world_pose_xyz=_xyz_values(row.get("world_pose_xyz")),
            world_pose_observed_at=row.get("world_pose_observed_at"),
            appearance_count=int(row["appearance_count"]),
            dedupe_confidence=float(row["dedupe_confidence"]),
            metadata=row.get("metadata") or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _observation_from_row(self, row: dict[str, object]) -> ObjectObservation:
        vector = _vector_values(row.get("appearance_embedding"))
        return ObjectObservation(
            observation_id=str(row["observation_id"]),
            object_id=str(row["object_id"]),
            user_id=row["user_id"],
            session_id=row["session_id"],
            source_id=row["source_id"],
            frame_idx=int(row["frame_idx"]),
            observed_at=row["observed_at"],
            track_id=row["track_id"],
            class_name=row["class_name"],
            detector_conf=float(row["detector_conf"]),
            room_id=row["room_id"],
            scene_scope=row.get("scene_scope"),
            bbox_xyxy_norm=tuple(float(value) for value in row["bbox_xyxy_norm"]),
            box_area=float(row["box_area"]),
            aspect_ratio=float(row["aspect_ratio"]),
            image_hash=row["image_hash"],
            world_pose_xyz=_xyz_values(row.get("world_pose_xyz")),
            world_pose_observed_at=row.get("world_pose_observed_at"),
            appearance_embedding=vector or [],
            appearance_model=row["appearance_model"],
            mask_area=float(row["mask_area"]) if row["mask_area"] is not None else None,
            attributes=row.get("attributes") or {},
        )
