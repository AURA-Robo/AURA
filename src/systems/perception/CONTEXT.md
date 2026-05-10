# Perception Subsystem Context

## Scope

`systems.perception` normalizes camera/depth observations, manages camera pitch/control helpers, optionally runs object detection, and publishes viewer telemetry.

## Read First

- `observation.py`
- `detector_runtime.py`
- `camera_runtime.py`
- `api/camera_api.py`
- `camera_control/runtime_service.py`
- `camera_control/sensor.py`
- `camera_control/targeting.py`
- `telemetry/viewer_publisher.py`

## Integration Surfaces

- `PerceptionObservationService.ingest()` accepts `RawObservation` and emits `ObservationFrame`.
- Detector metadata is normalized into frame metadata and viewer overlays, preserving `class_id` when the detector provides one.
- Detector outputs can be marked `detector_cached=true` when camera motion is below the reuse threshold; cached detections are viewer hints and should not be persisted into long-term object memory.
- Viewer telemetry writes RGB/depth into shared memory and publishes frame headers/health over transport.
- Viewer transport startup failures are optional-runtime failures: callers should degrade viewer publishing and keep
  core control/navigation runtime alive.

## Boundary Rules

- Perception should not depend on navigation geometry implementation.
- Use shared observation contracts for data crossing subsystem boundaries.
- Keep detector dependencies optional; `vision` extras are not part of the base install.

## State And Side Effects

Detector runtime caches model/load/error state, rate-limits inference, and reuses recent detections when camera translation/rotation stays below configured thresholds. Viewer publisher owns shared-memory writes and ZMQ frame/health publishing. Camera runtime/control helpers may mutate camera pitch/runtime sensor state.

## Cautions

- Malformed RGB/depth shapes should fail fast.
- Detector failures intentionally disable detector behavior and return empty detections.
- Depth/world-pose enrichment is best effort and can be omitted on invalid intrinsics or depth gaps.

## Tests

- `tests/test_perception_observation.py`
- `tests/test_yolo_object_memory_live_e2e.py`
- `tests/test_yolo_object_memory_postgres_e2e.py`
- `tests/test_yolo_object_memory_retrieval_flow.py`
- `tests/test_subsystem_architecture.py`
