# Reasoning Subsystem Context

## Scope

`systems.reasoning` owns natural-language request routing, dialogue/task decisions, task-frame generation, subgoal orchestration, planner catalog runtime, and the standalone reasoning service.

## Read First

- `service.py`
- `interpreter.py`
- `dialogue.py`
- `policy.py`
- `api/serve_reasoning_system.py`
- `planner/aura_adapter.py`
- `planner/planner_service.py`
- `planner/orchestration.py`
- `planner/schemas.py`
- `planner/validator.py`
- `planner_catalog_runtime.py`
- `planner_catalog_service.py`

## Entrypoints And Surfaces

- `.venv\Scripts\python.exe -m systems.reasoning.api.serve_reasoning_system`
- `scripts/run_system/reasoning_system_windows.bat`
- `GET /healthz`
- `GET /reasoning/status`
- `POST /reasoning/respond`
- `POST /reasoning/cancel`

Default reasoning endpoint is `http://127.0.0.1:17881`. The backend proxies dashboard reason/task/cancel calls to this service.

## Boundary Rules

- Route classification, task-frame planning, and subgoal execution are separate responsibilities; keep their contracts explicit.
- Use inference planner/dialogue clients for model completions.
- Use navigation service clients for execution; do not reach into navigation internals.
- Consume memory through `systems.memory.api.runtime`.

## State And Side Effects

`ReasoningCoordinator` tracks route/reply/error state and active task state. `TaskCoordinator` tracks subgoals, origin pose, navigation progress, cancellation, and errors. Planner catalog runtime can cache last-good/default snapshots and degrade without taking down reasoning.

## Cautions

- Stop/cancel requests are intercepted before normal interpretation.
- Busy-task behavior is part of the contract when interruption is not allowed.
- Knowledge guards can block execution and mark subgoals failed.
- Dialogue/planning paths have local fallback behavior when model calls fail; preserve this unless changing degraded-mode semantics deliberately.

## Tests

- `tests/test_reasoning_intent_routing.py`
- `tests/test_reasoning_runtime_e2e.py`
- `tests/test_reasoning_service_stop.py`
- `tests/test_planner_catalog_service.py`
- `tests/test_planner_tasking.py`
- `tests/test_runtime_planner_status.py`
- `tests/test_service_endpoint_contracts.py`
