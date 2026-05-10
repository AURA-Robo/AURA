# Planner Compatibility Note

The old `src/systems/planner` package has been removed.

Planner behavior is split across the current runtime surfaces:

- `src/systems/reasoning`
  - Owns natural-language intent routing, task-frame generation, planner
    orchestration, and planner catalog service/repository code.
- `src/systems/inference/planner`
  - Owns planner completion model serving and completion clients.
- `src/systems/shared/contracts/planner.py`
  - Keeps shared planner DTOs/contracts.
- `src/backend/app.py`
  - Exposes planner catalog management endpoints for the dashboard.

## Dashboard/API Surfaces

- `GET /api/planner/catalog`
- `POST /api/planner/intents`
- `DELETE /api/planner/intents/{intent_id}`
- `POST /api/planner/subgoals`
- `DELETE /api/planner/subgoals/{template_id}`

## Migration Rule

New planner workflow code should land in `systems.reasoning` unless it is model
serving or completion-client code, which belongs in `systems.inference.planner`.
