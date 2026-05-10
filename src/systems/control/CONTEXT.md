# Control Subsystem Context

## Scope

`systems.control` is the public control/runtime facade. It owns operator input helpers, runtime argument parsing, the lightweight control runtime API, telemetry publishing, and the public Isaac Sim entrypoint. The heavy Isaac Sim execution controller lives in `src/simulation` and is surfaced through control entrypoints.

## Read First

- `runtime_args.py`
- `runtime_control_api.py`
- `runtime/runtime_controller.py`
- `api/play_g1_internvla_navdp.py`
- `api/runtime.py`
- `api/tasking.py`
- `telemetry/viewer_publisher.py`
- `src/simulation/application/runtime_controller.py` only when changing Isaac Sim execution behavior

## Entrypoints And Surfaces

- `%ISAACSIM_PATH%\python.bat -m systems.control.api.play_g1_internvla_navdp`
- `scripts/run_system/control_runtime_windows.bat`
- `GET /healthz`
- `GET /runtime/status`

The dashboard backend reads control runtime health/status through `backend.sources.control_runtime`.

## Boundary Rules

- Do not import `simulation.*` from arbitrary control modules. Keep simulation implementation behind the public entrypoint/facade.
- Cross-subsystem dependencies should use `systems.<subsystem>.api.*` or `systems.shared.contracts.*`.
- Keep control status focused on control/navigation runtime state; planner task state belongs to reasoning/backend aggregation.

## State And Side Effects

The lightweight controller keeps latest trajectory, last command, and direct-action override state. Empty, stale, or errored trajectories intentionally collapse to zero command. Runtime API handlers can mutate camera pitch and runtime command state.

## Cautions

- Direct-action overrides preempt path following until they complete or time out.
- The Isaac Sim path is stateful and concurrency-heavy; prefer small changes with focused tests.
- Be conservative with permissive CORS/API expansion. Validate any new JSON body fields.

## Tests

- `tests/test_control_runtime_controller.py`
- `tests/test_runtime_planner_status.py`
- `tests/test_target_runtime_entrypoints.py`
- `tests/test_service_endpoint_contracts.py`
