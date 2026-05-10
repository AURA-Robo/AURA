# Navigation Subsystem Context

## Scope

`systems.navigation` owns the standalone navigation service, navigation geometry, goal expansion, follower logic, System1/NavDP and System2 backend adapters, and navigation status/trajectory APIs.

## Read First

- `service.py`
- `service_client.py`
- `service_codec.py`
- `follower.py`
- `geometry.py`
- `goals.py`
- `api/serve_navigation_system.py`
- `api/runtime.py`
- `system1/backends/http_backend.py`
- `system2/backends/http_backend.py`

## Entrypoints And Surfaces

- `.venv\Scripts\python.exe -m systems.navigation.api.serve_navigation_system`
- `scripts/run_system/navigation_system_windows.bat`
- `GET /healthz`
- `GET /navigation/status`
- `GET /navigation/trajectory`
- `POST /navigation/command`
- `POST /navigation/cancel`
- `POST /navigation/update`

Default navigation endpoint is `http://127.0.0.1:17882`.

## Boundary Rules

- Navigation consumes inference through clients/adapters; it does not own model process lifecycle.
- Do not import memory implementation modules directly. Memory-aware navigation targets should arrive through API payloads or shared contracts.
- Shared navigation DTOs such as `NavDpPlan`, `RobotState2D`, and `FollowerState` are owned by `systems.shared.contracts.navigation`; navigation modules may re-export them through `api/runtime.py`.
- Keep observation transport compatibility for shared-memory refs and legacy base64 payloads unless deliberately migrating callers.

## State And Side Effects

`NavigationSystem` keeps the active command/session, latest observation, trajectory, backend-stage status, last world goal, direct action state, and memory reacquire state. It runs background System2 and System1 worker threads and can optionally autostart child backend processes.

## Cautions

- `POST /navigation/update` is intentionally non-blocking relative to slow planning work.
- Direct System2 actions such as `forward`, `yaw_left`, and `yaw_right` can pause/skip NavDP path following.
- Memory-pose mode can fail with `memory_target_not_reacquired`.
- Preserve snake_case and camelCase aliases in status payloads when maintaining dashboard compatibility.

## Tests

- `tests/test_navigation_service.py`
- `tests/test_service_endpoint_contracts.py`
- `tests/test_runtime_planner_status.py`
- `tests/test_subsystem_architecture.py`
