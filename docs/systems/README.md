# Subsystem Catalog

The subsystem documentation formerly kept under `sub/` now lives in
`docs/systems/`.

AURA separates the dashboard/backend control plane from runtime subsystems under
`src/systems` and the Isaac Sim host runtime under `src/simulation`.

## Import Boundary

Cross-subsystem imports should use only:

- `systems.shared.contracts.*`
- `systems.<subsystem>.api.*`

Subsystem implementation modules should not import another subsystem's private
implementation directly. The test suite enforces this for `src/systems`.

## Runtime Subsystems

- [control.md](./control.md)
  - Operator command ingress, runtime task execution, control runtime HTTP API,
    and Isaac Sim public entrypoint.
- [inference.md](./inference.md)
  - Managed inference stack for NavDP, InternVLA/System2, planner completion,
    dialogue model serving, and health aggregation.
- [navigation.md](./navigation.md)
  - Navigation geometry, goal expansion, follower logic, and standalone
    navigation service APIs.
- [reasoning.md](./reasoning.md)
  - Natural-language intent routing, planner orchestration, task-frame
    generation, and planner catalog runtime.
- [memory.md](./memory.md)
  - Short-term memory, object memory, knowledge documents, conversation memory,
    and Postgres-backed memory repositories.
- [perception.md](./perception.md)
  - Camera control, detector runtime, observation contracts, and viewer
    telemetry publishing.
- [world-state.md](./world-state.md)
  - Runtime state DTOs and helper functions shared by runtime status assembly.
- [transport.md](./transport.md)
  - Runtime messages, buses, codecs, shared-memory frame transport, and
    transport health.
- [planner.md](./planner.md)
  - Legacy planner namespace note. Planner execution now belongs to reasoning
    and inference surfaces.

## Service Surfaces

- [backend.md](./backend.md)
  - Dashboard backend API, SSE state stream, runtime lifecycle proxy, WebRTC
    signaling, knowledge, and planner catalog endpoints.
- [simulation.md](./simulation.md)
  - Isaac Sim runtime assembly, scene setup, policy execution, and observation
    layout.

## Primary Entrypoints

- `python -m backend.api.serve_backend`
- `python -m systems.inference.api.serve_inference_system`
- `python -m systems.reasoning.api.serve_reasoning_system`
- `python -m systems.navigation.api.serve_navigation_system`
- `python -m systems.control.api.play_g1_internvla_navdp`
- `python -m runtime.api.serve_runtime` (optional standalone runtime surface)

## Windows Launchers

- `scripts/run_system/backend_windows.ps1`
- `scripts/run_system/inference_system_windows.bat`
- `scripts/run_system/reasoning_system_windows.bat`
- `scripts/run_system/navigation_system_windows.bat`
- `scripts/run_system/control_runtime_windows.bat`
- `scripts/run_system/runtime_windows.ps1` (optional standalone runtime surface)
- `scripts/run_system/run_dashboard_windows.ps1`
- `scripts/run_system/dashboard_dev_windows.ps1`
