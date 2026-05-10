CREATE TABLE IF NOT EXISTS planner_intents (
    intent_id TEXT PRIMARY KEY,
    intent_key TEXT NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS planner_intents_active_intent_key_uidx
    ON planner_intents (intent_key)
    WHERE deleted_at IS NULL;

CREATE TABLE IF NOT EXISTS planner_subgoal_templates (
    template_id TEXT PRIMARY KEY,
    intent_id TEXT NOT NULL REFERENCES planner_intents(intent_id),
    sequence_no INTEGER NOT NULL CHECK (sequence_no > 0),
    subgoal_type TEXT NOT NULL CHECK (
        subgoal_type IN ('navigate', 'inspect', 'return', 'report')
    ),
    activation_condition TEXT NOT NULL CHECK (
        activation_condition IN ('always', 'when_return_after_check', 'when_report_result')
    ),
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    deleted_at TIMESTAMPTZ NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS planner_subgoal_templates_active_sequence_uidx
    ON planner_subgoal_templates (intent_id, sequence_no)
    WHERE deleted_at IS NULL;

CREATE INDEX IF NOT EXISTS planner_subgoal_templates_active_intent_idx
    ON planner_subgoal_templates (intent_id, sequence_no)
    WHERE deleted_at IS NULL;
