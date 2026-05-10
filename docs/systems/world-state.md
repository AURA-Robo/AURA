# World State Subsystem

- Scope: runtime state DTOs and snapshot helper functions used by
  reasoning/planner flows, navigation, inference, control, simulation, and
  dashboard status assembly.
- Package root: `src/systems/world_state`

## Modules

- `api/runtime_state.py`

## Responsibilities

- `PlannerInput`
- `CommandState`
- `CaptureState`
- `System2RuntimeState`
- `GoalState`
- `NavDpState`
- `ActionOverrideState`
- `LocomotionState`
- `StatusState`
- `NavigationPipelineState`
- `TaskExecutionState`

Runtime services use these structures to assemble predictable state snapshots
without coupling dashboard code to simulator internals.

Navigation-facing state depends on shared navigation contracts
(`systems.shared.contracts.navigation`) for plan, robot, and follower DTOs
instead of importing navigation implementation modules.
