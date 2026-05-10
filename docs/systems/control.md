# Control Subsystem

- Scope: operator input handling, runtime task execution, control runtime HTTP
  API, viewer telemetry publishing, and the public Isaac Sim entrypoint.
- Package root: `src/systems/control`

## Modules

- `api`
  - `play_g1_internvla_navdp.py`
  - `runtime.py`
  - `runtime_args.py`
  - `runtime_controller.py`
  - `tasking.py`
- `runtime`
  - `entrypoint.py`
  - `runtime_controller.py`
- `telemetry`
  - `viewer_publisher.py`
- `operator_input.py`
- `runtime_args.py`
- `runtime_control_api.py`

## Entrypoints

- `python -m systems.control.api.play_g1_internvla_navdp`
- `scripts/run_system/control_runtime_windows.bat`

## HTTP Surface

The control runtime status surface is exposed at:

- `GET /healthz`
- `GET /runtime/status`

The dashboard backend reads this service through
`backend.sources.control_runtime.fetch_runtime_status`.
