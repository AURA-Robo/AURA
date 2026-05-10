from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from systems.memory.object_memory_models import ObjectObservationInput, utc_now
from systems.memory.object_memory_repository import PostgresObjectMemoryRepository
from systems.memory.object_memory_runtime import create_object_memory_runtime
from systems.memory.object_memory_service import ObjectMemoryService


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "systems"
    / "memory"
    / "sql"
    / "object_memory_schema.sql"
)


def _admin_conninfo() -> str:
    return os.environ.get(
        "AURA_TEST_POSTGRES_ADMIN_DSN",
        "host=127.0.0.1 dbname=postgres user=postgres",
    )


def _observation(
    *,
    frame_idx: int,
    track_id: str,
    observed_at,
    bbox_xyxy_norm: tuple[float, float, float, float],
    appearance_embedding: list[float],
) -> ObjectObservationInput:
    return ObjectObservationInput(
        frame_idx=frame_idx,
        track_id=track_id,
        class_name="chair",
        detector_conf=0.93,
        bbox_xyxy_norm=bbox_xyxy_norm,
        box_area=0.0,
        aspect_ratio=0.0,
        image_hash=f"image-{frame_idx}",
        appearance_embedding=appearance_embedding,
        observed_at=observed_at,
        room_id=None,
        scene_scope="warehouse",
        world_pose_xyz=(1.0 + frame_idx, 2.0 + frame_idx, 0.0),
        world_pose_observed_at=observed_at,
        source_id="camera-0",
        attributes={"source": "integration-test"},
    )


def _embedding(*prefix: float) -> list[float]:
    values = list(prefix)
    if len(values) > 512:
        raise ValueError("embedding prefix too long")
    values.extend([0.0] * (512 - len(values)))
    return values


@pytest.fixture()
def postgres_object_memory_dsn() -> str:
    admin_conninfo = _admin_conninfo()
    db_name = f"aura_object_memory_test_{uuid.uuid4().hex[:10]}"
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    try:
        with psycopg.connect(admin_conninfo, autocommit=True) as admin_conn:
            vector_row = admin_conn.execute(
                "select default_version from pg_available_extensions where name = 'vector'"
            ).fetchone()
            if vector_row is None or vector_row[0] is None:
                pytest.skip("pgvector extension is not available on the local PostgreSQL instance")
            admin_conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    except psycopg.Error as exc:
        pytest.skip(f"local PostgreSQL admin connection unavailable: {exc}")

    options = conninfo_to_dict(admin_conninfo)
    test_conninfo = make_conninfo(**{**options, "dbname": db_name})

    try:
        with psycopg.connect(test_conninfo, autocommit=True) as test_conn:
            test_conn.execute(schema_sql)
        yield test_conninfo
    finally:
        with psycopg.connect(admin_conninfo, autocommit=True) as admin_conn:
            admin_conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            admin_conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))


def test_postgres_repository_round_trip_with_real_database(postgres_object_memory_dsn: str) -> None:
    repository = PostgresObjectMemoryRepository(postgres_object_memory_dsn)
    service = ObjectMemoryService(repository)
    observed_at = utc_now() - timedelta(seconds=10)

    first = service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(
                frame_idx=1,
                track_id="track-1",
                observed_at=observed_at,
                bbox_xyxy_norm=(0.10, 0.10, 0.40, 0.50),
                appearance_embedding=_embedding(0.1, 0.2, 0.3),
            )
        ],
    )
    second = service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(
                frame_idx=2,
                track_id="track-1",
                observed_at=observed_at + timedelta(seconds=1),
                bbox_xyxy_norm=(0.11, 0.10, 0.41, 0.50),
                appearance_embedding=_embedding(0.11, 0.19, 0.31),
            )
        ],
    )

    assert len(first.created_object_ids) == 1
    assert len(second.updated_object_ids) == 1

    entries = repository.list_object_entries("tester", statuses=("active",))
    assert len(entries) == 1
    assert isinstance(entries[0].object_id, str)
    assert entries[0].room_id is None
    assert entries[0].scene_scope == "warehouse"
    assert entries[0].observation_count == 2
    assert entries[0].metadata["last_track_id"] == "track-1"
    assert entries[0].world_pose_xyz is not None

    observations = repository.list_object_observations("tester", object_ids=[entries[0].object_id])
    assert len(observations) == 2
    assert isinstance(observations[0].observation_id, str)
    assert isinstance(observations[0].object_id, str)
    assert observations[0].scene_scope == "warehouse"
    assert observations[0].world_pose_xyz is not None

    embedding = repository.get_object_embedding(entries[0].object_id)
    assert embedding is not None
    assert isinstance(embedding.object_id, str)
    assert embedding.index_status == "ready"
    assert embedding.embedding is not None
    assert len(embedding.embedding) == 512

    context = service.query_recent_objects("tester", scene_scope="warehouse")
    assert len(context.recent_seen) == 1
    assert context.recent_seen[0]["class"] == "chair"
    assert context.recent_seen[0]["room"] is None
    assert context.recent_seen[0]["scene_scope"] == "warehouse"

    resolution = service.resolve_memory_navigation_target(
        "tester",
        scene_scope="warehouse",
        class_name="chair",
        max_pose_age_sec=600,
    )
    assert resolution.status == "resolved"
    assert resolution.selected is not None
    assert resolution.selected.scene_scope == "warehouse"


def test_object_memory_runtime_uses_real_postgres(postgres_object_memory_dsn: str) -> None:
    runtime = create_object_memory_runtime(
        enabled=True,
        dsn=postgres_object_memory_dsn,
        user_id="tester",
    )

    assert runtime.available is True
    assert runtime.recent_context().recent_seen == []

    assert runtime.service is not None
    runtime.service.observe_objects(
        "tester",
        "session-1",
        [
            _observation(
                frame_idx=1,
                track_id="track-8",
                observed_at=utc_now(),
                bbox_xyxy_norm=(0.20, 0.20, 0.50, 0.60),
                appearance_embedding=[],
            )
        ],
    )

    context = runtime.recent_context(scene_scope="warehouse")
    assert len(context.recent_seen) == 1
    assert runtime.count_objects() == 1

    resolution = runtime.resolve_navigation_target(
        class_name="chair",
        scene_scope="warehouse",
        max_pose_age_sec=600,
    )
    assert resolution.status == "resolved"
