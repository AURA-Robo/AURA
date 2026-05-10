# Reasoning Subsystem

- Scope: natural-language request handling, dialogue/task routing, planner
  orchestration, task-frame validation, planner catalog management, and runtime
  reasoning service APIs.
- Package root: `src/systems/reasoning`

## Modules

- `api`
  - `serve_reasoning_system.py`
  - `runtime.py`
- `planner`
  - `aura_adapter.py`
  - `normalizer.py`
  - `ontology.py`
  - `orchestration.py`
  - `planner_service.py`
  - `reporting.py`
  - `schemas.py`
  - `task_frames.py`
  - `validator.py`
- `planner_catalog_*`
  - catalog errors, models, repository, runtime, schema, and service
- `dialogue.py`
- `interpreter.py`
- `policy.py`
- `service.py`

## Entrypoints

- `python -m systems.reasoning.api.serve_reasoning_system`
- `scripts/run_system/reasoning_system_windows.bat`

## HTTP Surface

The standalone reasoning service defaults to `127.0.0.1:17881` and exposes:

- `GET /healthz`
- `GET /reasoning/status`
- `POST /reasoning/respond`
- `POST /reasoning/cancel`

The dashboard backend proxies dashboard requests to this service through:

- `POST /api/runtime/reason`
- `POST /api/runtime/task`
- `POST /api/runtime/cancel`

The backend reads service health through
`backend.sources.reasoning_system.fetch_reasoning_status`.
