# Perception Subsystem

- Scope: camera APIs, camera pitch runtime services, camera prim attachment,
  detector runtime, observation contracts, and viewer telemetry publishing.
- Package root: `src/systems/perception`

## Modules

- `api/camera_api.py`
- `camera_control`
  - `api.py`
  - `math.py`
  - `runtime_service.py`
  - `sensor.py`
  - `targeting.py`
- `telemetry/viewer_publisher.py`
- `camera_runtime.py`
- `detector_runtime.py`
- `observation.py`

## Notes

The runtime controls a child camera prim rather than rotating an
articulation-linked rig root while physics is running. This avoids PhysX
articulation cache warnings caused by in-sim articulation writes.
