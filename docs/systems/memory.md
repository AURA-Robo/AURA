# Memory Subsystem

- Scope: short-term frame memory, humanoid agent memory, object memory, knowledge documents,
  conversation memory, Postgres-backed repositories, and runtime health/status
  adapters.
- Package root: `src/systems/memory`

## Modules

- `api/runtime.py`
- `stm.py`
- `agent_memory_*`
  - models, repository, runtime, service, and adapter
- `object_memory_*`
  - models, repository, runtime, service, and adapter
- `knowledge_*`
  - models, repository, runtime, service, and adapter
- `conversation_memory_*`
  - models, repository, and runtime
- `sql`
  - `object_memory_schema.sql`
  - `agent_memory_schema.sql`
  - `knowledge_schema.sql`
  - `conversation_memory_schema.sql`

## Backend Integration

The backend wires memory through:

- `--object-memory-dsn`
- `--agent-memory-dsn`
- `--object-memory-user-id`
- `--object-memory-auto-migrate`
- `--knowledge-dsn`
- `--planner-catalog-dsn`

Object memory can be fed from WebRTC viewer frames through
`src/backend/webrtc/object_memory.py`. Knowledge documents and planner catalog
data are exposed to the dashboard through backend `/api/knowledge/*` and
`/api/planner/*` endpoints.

Humanoid agent memory is a Letta-inspired structure without a Letta runtime
dependency. It separates always-injected core blocks, searchable archival
passages, recall/conversation context, object-memory summaries, and
knowledge-derived facts. The reasoning service compiles those pieces into a
single `agent_memory` payload for planner requests and a rendered memory section
for dialogue prompts. `AURA_AGENT_MEMORY_DSN` is used first; when unset it falls
back to `AURA_OBJECT_MEMORY_DSN`.

Default core blocks are:

- `persona`
- `operator_profile`
- `mission_policy`
- `environment_baseline`
- `working_memory`
- `capabilities`

`persona`, `mission_policy`, and `capabilities` are read-only by default.
Object memory remains authoritative for spatial/object pose resolution; archival
passages should capture durable events, preferences, corrections, task outcomes,
scene notes, and recovery patterns rather than pose history.

WebRTC object-memory ingest is conservative by default: cached detector output
is skipped, detections must pass confidence/class/bbox/world-pose gates before
they can become pending candidates, and linked repeat observations are persisted
only on meaningful change or heartbeat.

## Dashboard/API Surfaces

- `GET /api/knowledge/status`
- `GET /api/memory/status`
- `GET /api/memory/blocks`
- `PUT /api/memory/blocks/{label}`
- `GET /api/memory/passages`
- `POST /api/memory/passages`
- `GET /api/knowledge/documents`
- `POST /api/knowledge/documents`
- `PUT /api/knowledge/documents/{document_id}`
- `POST /api/knowledge/documents/{document_id}/publish`
- `POST /api/knowledge/documents/{document_id}/archive`
