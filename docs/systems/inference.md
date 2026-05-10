# Inference Subsystem

- Scope: managed model services for NavDP, InternVLA/System2, planner
  completion, dialogue inference, child-process supervision, health aggregation,
  and inference clients.
- Package root: `src/systems/inference`

## Modules

- `api`
  - `serve_inference_system.py`
  - `serve_inference_stack.py`
  - `planner.py`
  - `runtime.py`
- `stack`
  - `server.py`
  - `config.py`
  - `process_registry.py`
- `navdp`
  - `server.py`
  - `backend/*`
- `system2`
  - `server.py`
  - `check_session.py`
- `dialogue`
  - `server.py`
- `planner`
  - `completion_client.py`
  - `server.py`
- `client.py`

## Entrypoints

- `python -m systems.inference.api.serve_inference_system`
- `python -m systems.inference.api.serve_inference_stack`
- `python -m systems.inference.system2.check_session`
- `scripts/run_system/inference_system_windows.bat`

## HTTP Surface

The managed stack defaults to `127.0.0.1:15880` and exposes:

- `GET /healthz`
- `GET /models/state`
- `GET /stack/state`
- `GET /services/{service}/health`
- `GET /models/{model}/health`
- `POST /models/start`
- `POST /stack/start`
- `POST /models/stop`
- `POST /stack/stop`

The backend can read stack state through
`backend.sources.inference_stack.fetch_stack_state`, although current dashboard
state assembly primarily reads control, reasoning, navigation, runtime, WebRTC,
memory, and knowledge health.
