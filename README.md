# AURA System

AURA system owns the backend and runtime services that sit behind the dashboard frontend.

## Runtime layout

`src/systems` contains the subsystem packages:

- `control`
- `inference`
- `memory`
- `navigation`
- `perception`
- `reasoning`
- `shared/contracts`
- `transport`
- `world_state`

Top-level runtime services live directly under `src`:

- `backend`
- `runtime`

`src/simulation` contains the Isaac Sim host runtime:

- entrypoints
- runtime orchestration
- scene and asset loading
- observation layout
- controller binding

## Canonical launchers

- `scripts/run_system/inference_system_windows.bat`
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

- `.venv\Scripts\python.exe -m systems.inference.api.serve_inference_system`
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
- Inference system: `15880`
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
