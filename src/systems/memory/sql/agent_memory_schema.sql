CREATE TABLE IF NOT EXISTS agent_memory_blocks (
    label TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    value TEXT NOT NULL,
    limit_chars INTEGER NOT NULL CHECK (limit_chars > 0),
    read_only BOOLEAN NOT NULL,
    scope TEXT NOT NULL DEFAULT 'global',
    version INTEGER NOT NULL CHECK (version > 0),
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_memory_passages (
    passage_id UUID PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    scene_scope TEXT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_agent_memory_passages_tags
    ON agent_memory_passages USING GIN (tags);

CREATE INDEX IF NOT EXISTS idx_agent_memory_passages_search
    ON agent_memory_passages USING GIN (search_vector);

CREATE INDEX IF NOT EXISTS idx_agent_memory_passages_scene_created
    ON agent_memory_passages (scene_scope, created_at DESC);
