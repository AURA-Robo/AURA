# Backend Service

- Scope: dashboard HTTP API, SSE state broadcasting, session lifecycle control,
  runtime process ownership/proxying, log aggregation, occupancy metadata,
  WebRTC signaling, knowledge APIs, planner catalog APIs, and service health
  aggregation.
- Package root: `src/backend`

## Modules

- `api/serve_backend.py`
- `app.py`
- `session_manager.py`
- `models.py`
- `sse.py`
- `webrtc_proxy.py`
- `webrtc/*`
- `sources/*`

## Entrypoints

- `python -m backend.api.serve_backend`
- `scripts/run_system/backend_windows.ps1`

## Defaults

- Bind host: `127.0.0.1`
- Port: `18095`
- Dashboard dev origin: `http://127.0.0.1:5173`
- API base URL: `http://127.0.0.1:18095`
- Object-memory WebRTC ingest defaults: minimum confidence `0.80`,
  normalized bbox area `0.0005..0.35`, required world pose, and actionable
  class allowlist with known scene/background label blocklist. Override with
  `--object-memory-min-confidence`, `--object-memory-min-bbox-area`,
  `--object-memory-max-bbox-area`, `--object-memory-allowed-classes`,
  `--object-memory-blocked-classes`, or
  `--object-memory-allow-missing-world-pose`.
- Optional YOLO/object-memory JSONL event logging is disabled by default. Set
  `--object-memory-event-log-path` or `AURA_OBJECT_MEMORY_EVENT_LOG_PATH` to
  append detection/object link events without replacing Postgres object memory.
- WebRTC viewer dependencies: install the system venv with the default
  `scripts/setup_system_venv_windows.ps1` path, which includes `.[webrtc]`.
  Without `aiortc`/`av`, `/api/webrtc/config` reports the viewer transport as
  disabled and `/api/webrtc/offer` returns `503 webrtc_dependency_missing`.

## Dashboard API

The backend exposes the dashboard API under `/api`:

- `GET /api/bootstrap`
- `GET /api/state`
- `GET /api/events`
- `GET /api/logs`
- `GET /api/runtime/context-summary`
- `POST /api/session/start`
- `POST /api/session/stop`
- `POST /api/system/shutdown`
- `POST /api/runtime/reason`
- `POST /api/runtime/task`
- `POST /api/runtime/cancel`
- `GET /api/occupancy/current`
- `GET /api/occupancy/image`
- `GET /api/knowledge/status`
- `GET /api/knowledge/documents`
- `POST /api/knowledge/documents`
- `PUT /api/knowledge/documents/{document_id}`
- `POST /api/knowledge/documents/{document_id}/publish`
- `POST /api/knowledge/documents/{document_id}/archive`
- `GET /api/planner/catalog`
- `POST /api/planner/intents`
- `DELETE /api/planner/intents/{intent_id}`
- `POST /api/planner/subgoals`
- `DELETE /api/planner/subgoals/{template_id}`
- `GET /api/webrtc/config`
- `POST /api/webrtc/offer`

## Upstream Services

The backend can own the local runtime service or proxy an external runtime URL.
It also reads/proxies these subsystem services:

- Control runtime: `AURA_CONTROL_RUNTIME_URL`, default `http://127.0.0.1:8892`
- Reasoning system: `AURA_REASONING_SYSTEM_URL`, default
  `http://127.0.0.1:17881`
- Navigation system: `AURA_NAVIGATION_SYSTEM_URL`, default
  `http://127.0.0.1:17882`
- Inference system: `AURA_INFERENCE_SYSTEM_URL`, default
  `http://127.0.0.1:15880`
- Optional external runtime: `AURA_RUNTIME_URL` or
  `AURA_RUNTIME_SUPERVISOR_URL`
