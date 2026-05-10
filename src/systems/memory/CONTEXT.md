# Memory Subsystem Context

## Scope

`systems.memory` provides in-process/runtime memory components: short-term frame memory, humanoid agent memory, object memory, knowledge documents, conversation memory, repositories, adapters, and SQL schemas. It is not a standalone service daemon in this repository.

## Read First

- `api/runtime.py`
- `stm.py`
- `agent_memory_models.py`
- `agent_memory_repository.py`
- `agent_memory_runtime.py`
- `agent_memory_service.py`
- `agent_memory_adapter.py`
- `object_memory_models.py`
- `object_memory_repository.py`
- `object_memory_runtime.py`
- `object_memory_service.py`
- `knowledge_models.py`
- `knowledge_repository.py`
- `knowledge_runtime.py`
- `knowledge_service.py`
- `conversation_memory_models.py`
- `conversation_memory_repository.py`
- `conversation_memory_runtime.py`
- `sql/*.sql`

## Integration Surfaces

- Backend object-memory ingestion can be fed from WebRTC viewer frames.
- WebRTC object-memory ingestion is sparse: cached detector frames are not persisted, low-confidence/background/oversized detections are filtered before pending candidates, new objects require stable repeated observations, and repeated linked observations only persist on meaningful change or heartbeat.
- Backend WebRTC ingest can optionally append YOLO-style JSONL detection/object events when `AURA_OBJECT_MEMORY_EVENT_LOG_PATH` or `--object-memory-event-log-path` is set; this is an audit log and does not replace object-memory repositories.
- Backend WebRTC object-memory ingest is queued off the live frame subscriber so database latency cannot block viewer frame decode; the queue is bounded and drops older memory-ingest frames under backpressure.
- Backend agent-memory APIs expose core blocks and archival passages for operator inspection and edits.
- Backend knowledge APIs expose knowledge documents to the dashboard.
- Reasoning consumes agent memory, object memory, knowledge runtime, and conversation memory through `systems.memory.api`. Agent memory is compiled into dialogue/planning context on `/reasoning/respond`.
- Navigation can receive memory-resolved targets through reasoning/backend flows.

## Boundary Rules

- Keep persistence behind repository/runtime interfaces.
- Keep object memory authoritative for object identity, pose, and navigation target resolution; do not flatten pose history into archival text passages.
- Keep knowledge documents as curated policy/domain memory; agent memory may surface them but should not make them planner-writable.
- Do not make navigation import memory implementation directly.
- Preserve fail-open/degraded handles unless the task explicitly changes the product contract.
- Validate and normalize externally sourced document/rule/object payloads before persistence.

## State And Side Effects

STM stores copied RGB/depth/intrinsics/camera pose/robot state in memory and maintains epoch boundaries. Agent memory, object memory, and knowledge can persist to Postgres. Conversation memory writes to fallback in-memory storage and attempts configured persistence. Agent memory defaults to `AURA_AGENT_MEMORY_DSN`, falling back to `AURA_OBJECT_MEMORY_DSN` when unset.

## Cautions

- STM history views intentionally exclude the current frame.
- Object memory may return degraded/empty results for missing DSNs or repository failures.
- Agent memory may return degraded metadata with empty core/archival additions when storage is missing or unavailable.
- Agent memory core blocks include read-only system/operator-owned policy surfaces; writable updates must enforce block limits and read-only protection.
- Pending object candidates are backend-process local state; they intentionally disappear on restart because no schema change backs them.
- Object-memory ingest filters are intentionally conservative by default: detections need an allowed actionable class, sufficient confidence, bounded normalized bbox area, and a world pose unless the backend is explicitly configured otherwise.
- Cached detector frames should not be written to object memory or the JSONL event log.
- Under WebRTC object-memory queue backpressure, viewer streaming stays latest-frame-first and object-memory persistence may skip intermediate frames.
- Knowledge runtime can return permissive guard results when degraded.
- Postgres schemas are contract surfaces; update migrations/tests with schema changes.

## Tests

- `tests/test_memory_stm.py`
- `tests/test_agent_memory.py`
- `tests/test_agent_memory_reasoning_integration.py`
- `tests/test_object_memory.py`
- `tests/test_object_memory_postgres_integration.py`
- `tests/test_backend_object_memory_sink.py`
- `tests/test_backend_webrtc_subscriber.py`
- `tests/test_knowledge_service.py`
- `tests/test_knowledge_postgres_integration.py`
- `tests/test_history_clients.py`
