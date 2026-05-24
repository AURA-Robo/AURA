# AURA System

AURA system is the backend and runtime layer for the AURA humanoid AI agent.
It sits behind the dashboard frontend and coordinates natural-language tasking,
reasoning, navigation, control, memory, and simulator-facing runtime services.

The project report frames AURA as an operation-ready humanoid system rather
than a single synchronous perception-to-LLM pipeline. The current service
architecture keeps each runtime domain behind explicit process and package
boundaries so that failures, restarts, model serving, and future hardware
integration can be handled independently.

## Runtime model

AURA converts an operator instruction into a structured task frame, decomposes
that task into subgoals, routes each subgoal to the responsible subsystem, and
streams runtime state back to the dashboard.

```text
Dashboard command
  -> Backend runtime owner
  -> Reasoning interpreter / planner / orchestrator
  -> Navigation target and local trajectory planning
  -> Control runtime and Isaac Sim / Unitree-facing actuation
  -> Memory snapshots, observations, task history, and reports
```

Key design choices from the project report:

- Subsystems communicate through stable contracts instead of one tightly
  coupled loop.
- Scene understanding is VLM-centered, reducing information loss from
  object-list or scene-graph-only text handoffs.
- Reasoning separates instruction interpretation, task-frame generation, and
  subgoal orchestration.
- Navigation follows a System2 VLM plus NavDP-style local trajectory split.
- Control remains isolated from high-level planning so simulator control can
  later be swapped for Unitree SDK hardware control.
- Dashboard APIs provide the operator surface for lifecycle, state, task,
  knowledge, and viewer integration.

## Runtime layout

`src/systems` contains the subsystem packages:

| Path | Responsibility |
| --- | --- |
| `src/systems/reasoning` | Interprets natural-language commands, creates task frames, and decomposes work into executable subgoals such as navigate, inspect, return, and report. |
| `src/systems/navigation` | Consumes frames, sensor context, robot state, and navigation intent to update targets and local trajectory plans. |
| `src/systems/control` | Exposes the public control/runtime facade and bridges local trajectory intent into simulator or robot actuation entrypoints. |
| `src/systems/inference` | Hosts model-facing adapters for planner, dialogue, VLM, and runtime inference surfaces. |
| `src/systems/memory` | Stores state snapshots, observation history, knowledge context, and degraded-mode memory access. |
| `src/systems/perception` | Contains camera and perception-side runtime utilities used by simulator and future hardware inputs. |
| `src/systems/world_state` | Owns world-state representations shared by runtime decisions. |
| `src/systems/transport` | Provides runtime message and transport primitives. |
| `src/systems/shared/contracts` | Defines cross-subsystem contracts, service endpoints, and shared payload shapes. |

Top-level runtime services live directly under `src`:

- `backend`
- `runtime`

`src/simulation` contains the Isaac Sim host runtime:

- entrypoints
- runtime orchestration
- scene and asset loading
- observation layout
- controller binding

## Robot and model assumptions

The report targets a Unitree G1-class humanoid with RGB-D camera and IMU
inputs. This repository currently verifies the runtime through Isaac Sim and
keeps simulator-specific implementation under `src/simulation`.

The intended execution stack is:

- Qwen/VLM-style scene and navigation reasoning for visual-language grounding.
- NavDP-style local trajectory generation for robot-frame motion intent.
- A G1 locomotion/control policy behind the control runtime.
- Quantized or external model-serving backends behind inference adapters.

Model paths, serving URLs, credentials, and hardware-specific settings should
remain environment- or CLI-configured. Do not hardcode local model paths or
secrets in this repository.

## Dashboard operating surface

The backend owns runtime lifecycle for normal dashboard work. It exposes the
HTTP APIs, SSE state stream, WebRTC signaling path, task/planner state, and
knowledge-facing APIs that let an operator see what the robot is doing and send
new instructions without knowing the internal subsystem layout.

The report also identifies Knowledge Studio and Task Builder as expansion
surfaces: domain documents, procedures, inspection checklists, and reusable
task templates should flow through backend/runtime contracts rather than bypass
subsystem boundaries.

## Canonical launchers

- `scripts/run_system/reasoning_system_windows.bat`
- `scripts/run_system/navigation_system_windows.bat`
- `scripts/run_system/control_runtime_windows.bat`
- `scripts/run_system/backend_windows.ps1`
- `scripts/run_system/runtime_windows.ps1` (optional standalone runtime surface)
- `scripts/run_system/dashboard_dev_windows.ps1`

## Python entrypoints

- System services use the repo-local `.venv\Scripts\python.exe` by default.
- Isaac Sim services use `%ISAACSIM_PATH%\python.bat` through `control_runtime_windows.bat`.
- Override the system interpreter with `AURA_PYTHON` or launcher `-Python` only when intentionally testing another environment.

Entrypoints:

- `.venv\Scripts\python.exe -m systems.reasoning.api.serve_reasoning_system`
- `.venv\Scripts\python.exe -m systems.navigation.api.serve_navigation_system`
- `%ISAACSIM_PATH%\python.bat -m systems.control.api.play_g1_internvla_navdp`
- `.venv\Scripts\python.exe -m backend.api.serve_backend`
- `.venv\Scripts\python.exe -m runtime.api.serve_runtime` (optional standalone runtime surface)

## Default local bring-up

For normal dashboard work, start only:

1. `scripts/run_system/backend_windows.ps1`
2. the dashboard frontend from `C:\Users\mango\project\AURA\dashboard`

The backend owns runtime lifecycle by default, so the dashboard Start/Stop controls do not require `runtime_windows.ps1`.

Use `runtime_windows.ps1` only when you explicitly want an external runtime control plane. In that mode, point the backend at it with `AURA_RUNTIME_URL` or `--runtime-url`.

## Ports

- Backend: `18095`
- Runtime: `18096`
- Reasoning system: `17881`
- Navigation system: `17882`
- Control runtime: `8892`

## Install

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_system_venv_windows.ps1
```

The default setup includes the `webrtc` extra because the dashboard viewer
uses the backend-owned WebRTC signaling path. Use `-NoWebRtc` only for a
minimal non-viewer environment.

For test tooling:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_system_venv_windows.ps1 -Extras dev
```

## Test

```bash
pytest tests/test_backend.py ^
  tests/test_runtime.py ^
  tests/test_planner_tasking.py ^
  tests/test_runtime_planner_status.py ^
  tests/test_target_runtime_entrypoints.py ^
  tests/test_target_runtime_paths.py ^
  tests/test_subsystem_architecture.py ^
  tests/scripts/test_windows_fullstack_launcher.py
```
