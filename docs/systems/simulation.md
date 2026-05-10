# Simulation Runtime

- Scope: Isaac Sim runtime assembly, scene spawn, locomotion policy execution,
  asset/path resolution, controller binding, and observation layout.
- Package root: `src/simulation`

## Layers

- `api`
  - `entrypoint.py`
  - `runtime.py`
  - `runtime_controller.py`
  - `paths.py`
- `application`
  - `runtime_controller.py`
  - `runtime_orchestrator.py`
- `domain`
  - `constants.py`
  - `observation_layout.py`
  - `observation_constants.py`
- `infrastructure`
  - `scene.py`
  - `paths.py`
  - `policy_controller.py`
  - `policy_session.py`
  - `training_config.py`

## Notes

`systems.control` remains the public runtime facade. Isaac Sim specific code is
intentionally isolated in `src/simulation` so subsystem packages do not depend
on simulator implementation modules directly.
