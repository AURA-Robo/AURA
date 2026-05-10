# Navigation Subsystem

- Scope: runtime geometry, goal providers, follower logic, NavDP client
  integration, System1/System2 backend adapters, and standalone navigation
  service APIs.
- Package root: `src/systems/navigation`

## Modules

- `api`
  - `geometry.py`
  - `navdp_sensors.py`
  - `runtime.py`
  - `serve_navigation_system.py`
- `system1/backends/http_backend.py`
- `system2/backends/http_backend.py`
- `client.py`
- `follower.py`
- `geometry.py`
- `goals.py`
- `service.py`
- `service_client.py`
- `service_codec.py`

## Entrypoints

- `python -m systems.navigation.api.serve_navigation_system`
- `scripts/run_system/navigation_system_windows.bat`

## HTTP Surface

The standalone navigation service defaults to `127.0.0.1:17882` and exposes:

- `GET /healthz`
- `GET /navigation/status`
- `GET /navigation/trajectory`
- `POST /navigation/command`
- `POST /navigation/cancel`
- `POST /navigation/update`

The dashboard backend reads this service through
`backend.sources.navigation_system.fetch_navigation_status`.

## Notes

Standalone NavDP model serving is owned by the inference subsystem. Navigation
consumes that service through the NavDP client/adapters instead of owning model
process lifecycle directly.

Shared navigation DTOs (`NavDpPlan`, `RobotState2D`, and `FollowerState`) live
in `systems.shared.contracts.navigation`. `systems.navigation.api.runtime`
re-exports them for runtime callers that already depend on the navigation
facade.
