# Inference Subsystem Context

## Scope

`systems.inference` owns managed model-serving surfaces and child-process supervision for System2, NavDP, planner completion, and dialogue inference. It also provides clients/helpers consumed by navigation and reasoning.

## Read First

- `api/serve_inference_system.py`
- `stack/server.py`
- `stack/config.py`
- `stack/process_registry.py`
- `client.py`
- `system2/server.py`
- `planner/completion_client.py`
- `planner/server.py`
- `dialogue/server.py`
- `navdp/server.py`
- `navdp/backend/tensorrt_acceleration.py`

## Entrypoints And Surfaces

- `.venv\Scripts\python.exe -m systems.inference.api.serve_inference_system`
- `.venv\Scripts\python.exe -m systems.inference.api.serve_inference_stack`
- `scripts/run_system/inference_system_windows.bat`
- Dashboard-owned runtime sessions launch `inference_system_windows.bat` before navigation, reasoning, and control. When no `DIALOGUE_LORA_ADAPTER_PATH` is configured for that local runtime path, the runtime sets `DIALOGUE_ALLOW_PROMPT_ONLY=1` so dialogue can be tested with the Qwen chat prompt only.
- `GET /healthz`
- `GET /models/state`
- `GET /stack/state`
- `GET /services/{service}/health`
- `GET /models/{model}/health`
- `POST /models/start`
- `POST /stack/start`
- `POST /models/stop`
- `POST /stack/stop`

Default inference system endpoint is `http://127.0.0.1:15880`.

## Boundary Rules

- Keep model process lifecycle in inference. Navigation consumes System2/NavDP through clients and service APIs.
- Shared result shapes should live in `systems.shared.contracts.*`.
- Avoid leaking model-specific internals into backend or dashboard payloads.

## State And Side Effects

`ProcessRegistry` launches long-lived child processes and writes logs. System2 clients and servers maintain request/session state. Planner and dialogue launchers wrap llama.cpp-compatible servers and model paths. InternVLA System2 uses one llama.cpp sidecar with separate slots for navigation and inspect/check; inspect/check can apply a startup-configured LoRA adapter via `INTERNVLA_CHECK_LORA_ADAPTER_PATH` / `--check-lora-adapter-path`, `INTERNVLA_CHECK_LORA_SCALE` / `--check-lora-scale`, and `INTERNVLA_CHECK_SESSION_SYSTEM_PROMPT` / `--check-session-system-prompt`. Dialogue serving defaults to the repo Qwen GGUF (`Qwen3-1.7B-Q4_K_M-Instruct.gguf`) and requires a chat LoRA adapter via `DIALOGUE_LORA_ADAPTER_PATH` or `--lora-adapter-path`; startup should fail instead of silently serving dialogue without that adapter. For prompt-only dialogue testing, explicitly set `DIALOGUE_ALLOW_PROMPT_ONLY=1` or pass `--allow-prompt-only`; this starts the Qwen dialogue server without `--lora` and relies on the reasoning dialogue prompt only.

## Cautions

- Managed stack health is only healthy when required services are healthy.
- Process stop is terminate/kill based; account for partial startup and cleanup.
- Planner completion is single-shot and raises on invalid JSON; add explicit repair behavior only with tests.
- Dialogue server startup validates that the configured chat LoRA adapter exists before launching `llama-server`.
- `system2/server.py` is large and accepts multipart/image/depth inputs; validate every new boundary field.
- NavDP TensorRT acceleration is optional and fail-open in `auto` mode. It accelerates the static DDPM sampler, noise predictor fallback, and critic from prebuilt engines under `artifacts/models/navdp_tensorrt`; use `--navdp-tensorrt-mode build` or `python -m systems.inference.navdp.backend.build_tensorrt` to generate matching engine metadata for the active checkpoint. Windows TensorRT builds use a large-stack worker thread because the unrolled sampler graph can overflow the default builder stack.

## Tests

- `tests/test_inference_stack_server.py`
- `tests/test_inference_dialogue_server.py`
- `tests/test_inference_planner_completion_client.py`
- `tests/test_inference_planner_server.py`
- `tests/test_system2_check_session.py`
- `tests/test_service_endpoint_contracts.py`
- `tests/test_navdp_tensorrt_acceleration.py`
