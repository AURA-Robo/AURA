# World State Subsystem Context

## Scope

`systems.world_state` defines runtime state DTOs and helper functions used by reasoning/planner flows, navigation, inference, control, simulation, and dashboard status assembly.

## Read First

- `api/runtime_state.py`

## Contract Surfaces

Primary state objects include planner input, command state, capture state, System2 state, goal state, NavDP state, action override state, locomotion state, status state, navigation pipeline state, and task execution state.

Helper functions such as `goal_target_mode()`, `goal_is_done()`, and `goal_current_body_xy()` define canonical read semantics for active goals.

## Boundary Rules

- Keep this package focused on state shapes and helper semantics.
- Do not add simulator, backend, or dashboard side effects here.
- Prefer `systems.shared.contracts.*` for DTO dependencies. Runtime state helpers should not import navigation implementation facades just to compute state read semantics.

## State And Side Effects

This module is mostly pure data/helpers. Actual mutation happens in owning runtime components, often under locks. Several dataclasses contain mutable NumPy arrays. Navigation-facing state uses shared navigation contracts for plan, robot, and follower DTOs.

## Cautions

- State objects are not inherently thread-safe. Callers must lock and copy at mutation/serialization boundaries.
- `goal_current_body_xy()` raises when no active point goal exists; callers should check goal mode first.
- Changes can have broad status-payload impact because many services assemble snapshots from these shapes.

## Tests

- Indirect coverage through `tests/test_runtime_coordinator.py`, `tests/test_runtime_planner_status.py`, and simulator/runtime tests.
- Add focused tests when changing helper semantics or state defaults.
