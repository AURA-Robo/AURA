# Shared Subsystem Context

## Scope

`systems.shared` holds cross-subsystem contracts and small shared utilities. It should remain stable, dependency-light, and implementation-neutral.

## Read First

- `contracts/service_endpoints.py`
- `contracts/dashboard.py`
- `contracts/inference.py`
- `contracts/navigation.py`
- `contracts/navigation_transport.py`
- `contracts/observation.py`
- `contracts/planner.py`
- `contracts/reasoning.py`
- `contracts/runtime_state.py`
- `contracts/viewer_transport.py`
- `viewer_transport.py`

## Contract Surfaces

- Default local endpoint constants for backend, runtime, inference, reasoning, navigation, and control.
- DTO/dataclass/TypedDict shapes used across services.
- Dashboard `ProcessRecord` includes optional stdout/stderr log offsets so backend log views can show the current launch segment without replaying stale cumulative launcher logs.
- Navigation runtime DTOs shared across process/state boundaries: `NavDpPlan`, `RobotState2D`, `FollowerState`, and `make_follower_state()`.
- Observation/history encoding helpers.
- Viewer transport constants and payload shapes. Default viewer control/telemetry ports are 18880/18881.

## Boundary Rules

- Shared contracts may be imported by subsystem implementations.
- Shared code must not import private subsystem implementation modules.
- Keep this package free of heavy runtime dependencies and side effects.
- Backward compatibility matters: changes here can break backend, dashboard, launchers, and multiple services at once.

## State And Side Effects

Most code is pure contract/serialization logic. Observation helpers copy or normalize arrays and can raise on malformed tensors.

## Cautions

- Endpoint constants are the repository's default local contract, not runtime discovery.
- Shape validators for observation/history data are intentionally strict.
- Add tests or update downstream assertions for any contract field rename, alias removal, or endpoint change.

## Tests

- `tests/test_service_endpoint_contracts.py`
- `tests/test_subsystem_architecture.py`
- Indirect coverage from backend, runtime, inference, navigation, reasoning, memory, and transport tests.
