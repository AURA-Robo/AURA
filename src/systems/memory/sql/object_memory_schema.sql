CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS object_memory_entries (
    object_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    canonical_class TEXT NOT NULL,
    room_id TEXT NULL,
    scene_scope TEXT NULL,
    status TEXT NOT NULL CHECK (status IN ('active', 'stale', 'deleted')),
    first_seen_at TIMESTAMPTZ NOT NULL,
    last_seen_at TIMESTAMPTZ NOT NULL,
    observation_count INTEGER NOT NULL,
    last_source_id TEXT NULL,
    last_session_id TEXT NOT NULL,
    last_bbox_xyxy_norm JSONB NOT NULL,
    last_box_area DOUBLE PRECISION NOT NULL,
    last_aspect_ratio DOUBLE PRECISION NOT NULL,
    last_detector_conf DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    world_pose_xyz JSONB NULL,
    world_pose_observed_at TIMESTAMPTZ NULL,
    appearance_count INTEGER NOT NULL,
    dedupe_confidence DOUBLE PRECISION NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_object_memory_entries_user_status
    ON object_memory_entries (user_id, status, last_seen_at DESC);

CREATE INDEX IF NOT EXISTS idx_object_memory_entries_user_class_room
    ON object_memory_entries (user_id, canonical_class, room_id, last_seen_at DESC);

CREATE TABLE IF NOT EXISTS object_memory_observations (
    observation_id UUID PRIMARY KEY,
    object_id UUID NOT NULL REFERENCES object_memory_entries (object_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    source_id TEXT NULL,
    frame_idx INTEGER NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    track_id TEXT NOT NULL,
    class_name TEXT NOT NULL,
    detector_conf DOUBLE PRECISION NOT NULL,
    room_id TEXT NULL,
    scene_scope TEXT NULL,
    bbox_xyxy_norm JSONB NOT NULL,
    box_area DOUBLE PRECISION NOT NULL,
    aspect_ratio DOUBLE PRECISION NOT NULL,
    mask_area DOUBLE PRECISION NULL,
    world_pose_xyz JSONB NULL,
    world_pose_observed_at TIMESTAMPTZ NULL,
    appearance_embedding VECTOR(512),
    appearance_model TEXT NOT NULL,
    image_hash TEXT NOT NULL,
    attributes JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_object_memory_observations_object_time
    ON object_memory_observations (object_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_object_memory_observations_user_time
    ON object_memory_observations (user_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS object_memory_entry_embeddings (
    object_id UUID PRIMARY KEY REFERENCES object_memory_entries (object_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    embedding VECTOR(512),
    index_status TEXT NOT NULL CHECK (index_status IN ('pending', 'ready', 'failed')),
    embedded_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_object_memory_embeddings_user_status
    ON object_memory_entry_embeddings (user_id, index_status, embedded_at DESC);

CREATE INDEX IF NOT EXISTS idx_object_memory_embeddings_hnsw
    ON object_memory_entry_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WHERE index_status = 'ready' AND embedding IS NOT NULL;

ALTER TABLE IF EXISTS object_memory_entries
    ADD COLUMN IF NOT EXISTS scene_scope TEXT NULL;
ALTER TABLE IF EXISTS object_memory_entries
    ADD COLUMN IF NOT EXISTS world_pose_xyz JSONB NULL;
ALTER TABLE IF EXISTS object_memory_entries
    ADD COLUMN IF NOT EXISTS world_pose_observed_at TIMESTAMPTZ NULL;

ALTER TABLE IF EXISTS object_memory_observations
    ADD COLUMN IF NOT EXISTS scene_scope TEXT NULL;
ALTER TABLE IF EXISTS object_memory_observations
    ADD COLUMN IF NOT EXISTS world_pose_xyz JSONB NULL;
ALTER TABLE IF EXISTS object_memory_observations
    ADD COLUMN IF NOT EXISTS world_pose_observed_at TIMESTAMPTZ NULL;

UPDATE object_memory_observations
SET world_pose_xyz = attributes->'world_pose_xyz',
    world_pose_observed_at = COALESCE(world_pose_observed_at, observed_at)
WHERE world_pose_xyz IS NULL
  AND jsonb_typeof(attributes->'world_pose_xyz') = 'array';

UPDATE object_memory_entries AS entry
SET world_pose_xyz = latest.world_pose_xyz,
    world_pose_observed_at = latest.world_pose_observed_at,
    scene_scope = COALESCE(entry.scene_scope, latest.scene_scope)
FROM (
    SELECT DISTINCT ON (object_id)
        object_id,
        world_pose_xyz,
        COALESCE(world_pose_observed_at, observed_at) AS world_pose_observed_at,
        scene_scope
    FROM object_memory_observations
    WHERE world_pose_xyz IS NOT NULL
    ORDER BY object_id, COALESCE(world_pose_observed_at, observed_at) DESC, observed_at DESC
) AS latest
WHERE entry.object_id = latest.object_id
  AND (
      entry.world_pose_xyz IS NULL
      OR entry.world_pose_observed_at IS NULL
      OR entry.scene_scope IS NULL
  );
