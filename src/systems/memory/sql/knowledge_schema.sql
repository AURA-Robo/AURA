CREATE TABLE IF NOT EXISTS knowledge_documents (
    document_id UUID PRIMARY KEY,
    title TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('global', 'scene')),
    scope_value TEXT NULL,
    status TEXT NOT NULL CHECK (status IN ('draft', 'published', 'archived')),
    body_markdown TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    version INTEGER NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_documents_status_scope
    ON knowledge_documents (status, scope_kind, scope_value, updated_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_rules (
    rule_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES knowledge_documents (document_id) ON DELETE CASCADE,
    rule_key TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('global', 'scene')),
    scope_value TEXT NULL,
    enforcement TEXT NOT NULL CHECK (enforcement IN ('hard', 'soft')),
    action TEXT NOT NULL CHECK (
        action IN (
            'deny_task',
            'require_clarification',
            'force_target_room',
            'restrict_query_attributes'
        )
    ),
    conditions JSONB NOT NULL DEFAULT '{}'::jsonb,
    params JSONB NOT NULL DEFAULT '{}'::jsonb,
    priority INTEGER NOT NULL DEFAULT 0,
    reason TEXT NULL,
    source_anchor TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    published_at TIMESTAMPTZ NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_rules_document_enforcement
    ON knowledge_rules (document_id, enforcement, priority DESC, updated_at DESC);

CREATE TABLE IF NOT EXISTS knowledge_lexicon_entries (
    entry_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES knowledge_documents (document_id) ON DELETE CASCADE,
    mapping_type TEXT NOT NULL CHECK (mapping_type IN ('object', 'attribute', 'room')),
    alias TEXT NOT NULL,
    canonical TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('global', 'scene')),
    scope_value TEXT NULL,
    source_anchor TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_lexicon_scope_alias
    ON knowledge_lexicon_entries (scope_kind, scope_value, mapping_type, alias);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    chunk_id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES knowledge_documents (document_id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    scope_kind TEXT NOT NULL CHECK (scope_kind IN ('global', 'scene')),
    scope_value TEXT NULL,
    source_anchor TEXT NULL,
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(text, ''))) STORED,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_document_order
    ON knowledge_chunks (document_id, chunk_index ASC);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_search
    ON knowledge_chunks USING GIN (search_vector);

CREATE TABLE IF NOT EXISTS knowledge_rule_audit (
    audit_id UUID PRIMARY KEY,
    rule_id UUID NOT NULL REFERENCES knowledge_rules (rule_id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES knowledge_documents (document_id) ON DELETE CASCADE,
    phase TEXT NOT NULL,
    task_id TEXT NULL,
    subgoal_id TEXT NULL,
    applied_at TIMESTAMPTZ NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_knowledge_rule_audit_phase_time
    ON knowledge_rule_audit (phase, applied_at DESC);
