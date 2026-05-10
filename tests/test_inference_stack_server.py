from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace
import time

from systems.inference.stack.config import build_managed_services
from systems.inference.stack import server as inference_stack_server


def test_inference_service_snapshot_probes_services_in_parallel(monkeypatch) -> None:
    server = inference_stack_server.InferenceSystemServer.__new__(inference_stack_server.InferenceSystemServer)
    server.args = Namespace(health_timeout=1.0)
    server._services = [
        SimpleNamespace(name="navdp", base_url="http://navdp", health_url="http://navdp/healthz"),
        SimpleNamespace(name="system2", base_url="http://system2", health_url="http://system2/healthz"),
        SimpleNamespace(name="planner", base_url="http://planner", health_url="http://planner/healthz"),
    ]

    def fake_json_get(url: str, *, timeout_s: float):
        del timeout_s
        time.sleep(0.2)
        return "healthy", {"url": url}, 200.0, None

    monkeypatch.setattr(inference_stack_server, "_json_get", fake_json_get)

    started = time.perf_counter()
    snapshot = server._service_snapshot()
    elapsed = time.perf_counter() - started

    assert set(snapshot.keys()) == {"navdp", "system2", "planner"}
    assert elapsed < 0.45


def test_inference_models_state_only_requires_required_services(monkeypatch) -> None:
    server = inference_stack_server.InferenceSystemServer.__new__(inference_stack_server.InferenceSystemServer)
    server.args = Namespace(host="127.0.0.1", port=15880, log_dir="logs")
    server._services = [
        SimpleNamespace(name="navdp", base_url="http://navdp", health_url="http://navdp/healthz", required=False),
        SimpleNamespace(name="system2", base_url="http://system2", health_url="http://system2/healthz", required=True),
        SimpleNamespace(name="planner", base_url="http://planner", health_url="http://planner/health", required=False),
    ]
    server._registry = SimpleNamespace(snapshot=lambda: [])

    monkeypatch.setattr(
        server,
        "_service_snapshot",
        lambda service_name=None: {
            "navdp": {"status": "unreachable"},
            "system2": {"status": "healthy"},
            "planner": {"status": "error"},
        },
    )

    state = server.models_state()

    assert state["ok"] is True
    assert state["models"]["system2"]["status"] == "healthy"
    assert state["models"]["navdp"]["status"] == "unreachable"


def test_managed_dialogue_service_receives_lora_adapter_path() -> None:
    args = Namespace(
        navdp_host="127.0.0.1",
        navdp_port=18888,
        navdp_checkpoint="navdp.ckpt",
        navdp_device="cuda:0",
        system2_host="127.0.0.1",
        system2_port=15801,
        system2_llama_url="http://127.0.0.1:15802",
        system2_model_path="",
        system2_check_lora_adapter_path="",
        system2_check_lora_scale=1.0,
        system2_check_session_system_prompt="",
        planner_host="127.0.0.1",
        planner_port=8093,
        planner_model_path="planner.gguf",
        planner_llama_server="llama-server.exe",
        planner_gpu_layers=999,
        planner_ctx_size=1024,
        planner_parallel_slots=2,
        planner_cache_type_k="q8_0",
        planner_cache_type_v="q8_0",
        dialogue_host="127.0.0.1",
        dialogue_port=8094,
        dialogue_model_path="qwen-dialogue.gguf",
        dialogue_llama_server="llama-server.exe",
        dialogue_gpu_layers=999,
        dialogue_ctx_size=1024,
        dialogue_cache_type_k="q8_0",
        dialogue_cache_type_v="q8_0",
        dialogue_lora_adapter_path="qwen-chat-lora.gguf",
        dialogue_allow_prompt_only=False,
    )

    services = build_managed_services(args)
    dialogue = next(service for service in services if service.name == "reasoning_dialogue")

    assert "--lora-adapter-path" in dialogue.command
    assert dialogue.command[dialogue.command.index("--lora-adapter-path") + 1] == "qwen-chat-lora.gguf"


def test_managed_navdp_service_receives_tensorrt_options() -> None:
    args = Namespace(
        navdp_host="127.0.0.1",
        navdp_port=18888,
        navdp_checkpoint="navdp.ckpt",
        navdp_device="cuda:0",
        navdp_tensorrt_mode="auto",
        navdp_tensorrt_engine_dir="artifacts/models/navdp_tensorrt",
        navdp_tensorrt_precision="fp16",
        system2_host="127.0.0.1",
        system2_port=15801,
        system2_llama_url="http://127.0.0.1:15802",
        system2_model_path="",
        system2_check_lora_adapter_path="",
        system2_check_lora_scale=1.0,
        system2_check_session_system_prompt="",
        planner_host="127.0.0.1",
        planner_port=8093,
        planner_model_path="planner.gguf",
        planner_llama_server="llama-server.exe",
        planner_gpu_layers=999,
        planner_ctx_size=1024,
        planner_parallel_slots=2,
        planner_cache_type_k="q8_0",
        planner_cache_type_v="q8_0",
        dialogue_host="127.0.0.1",
        dialogue_port=8094,
        dialogue_model_path="qwen-dialogue.gguf",
        dialogue_llama_server="llama-server.exe",
        dialogue_gpu_layers=999,
        dialogue_ctx_size=1024,
        dialogue_cache_type_k="q8_0",
        dialogue_cache_type_v="q8_0",
        dialogue_lora_adapter_path="qwen-chat-lora.gguf",
        dialogue_allow_prompt_only=False,
    )

    services = build_managed_services(args)
    navdp = next(service for service in services if service.name == "navdp")

    assert "--tensorrt-mode" in navdp.command
    assert navdp.command[navdp.command.index("--tensorrt-mode") + 1] == "auto"
    assert "--tensorrt-engine-dir" in navdp.command
    assert navdp.command[navdp.command.index("--tensorrt-engine-dir") + 1] == "artifacts/models/navdp_tensorrt"
    assert "--tensorrt-precision" in navdp.command
    assert navdp.command[navdp.command.index("--tensorrt-precision") + 1] == "fp16"


def test_managed_dialogue_service_receives_prompt_only_flag() -> None:
    args = Namespace(
        navdp_host="127.0.0.1",
        navdp_port=18888,
        navdp_checkpoint="navdp.ckpt",
        navdp_device="cuda:0",
        system2_host="127.0.0.1",
        system2_port=15801,
        system2_llama_url="http://127.0.0.1:15802",
        system2_model_path="",
        system2_check_lora_adapter_path="",
        system2_check_lora_scale=1.0,
        system2_check_session_system_prompt="",
        planner_host="127.0.0.1",
        planner_port=8093,
        planner_model_path="planner.gguf",
        planner_llama_server="llama-server.exe",
        planner_gpu_layers=999,
        planner_ctx_size=1024,
        planner_parallel_slots=2,
        planner_cache_type_k="q8_0",
        planner_cache_type_v="q8_0",
        dialogue_host="127.0.0.1",
        dialogue_port=8094,
        dialogue_model_path="qwen-dialogue.gguf",
        dialogue_llama_server="llama-server.exe",
        dialogue_gpu_layers=999,
        dialogue_ctx_size=1024,
        dialogue_cache_type_k="q8_0",
        dialogue_cache_type_v="q8_0",
        dialogue_lora_adapter_path="",
        dialogue_allow_prompt_only=True,
    )

    services = build_managed_services(args)
    dialogue = next(service for service in services if service.name == "reasoning_dialogue")

    assert "--allow-prompt-only" in dialogue.command


def test_managed_system2_service_receives_check_lora_and_prompt_options() -> None:
    args = Namespace(
        navdp_host="127.0.0.1",
        navdp_port=18888,
        navdp_checkpoint="navdp.ckpt",
        navdp_device="cuda:0",
        system2_host="127.0.0.1",
        system2_port=15801,
        system2_llama_url="http://127.0.0.1:15802",
        system2_model_path="",
        system2_check_lora_adapter_path="tv-check-lora.gguf",
        system2_check_lora_scale=0.5,
        system2_check_session_system_prompt="Answer true only if the TV is clearly off.",
        planner_host="127.0.0.1",
        planner_port=8093,
        planner_model_path="planner.gguf",
        planner_llama_server="llama-server.exe",
        planner_gpu_layers=999,
        planner_ctx_size=1024,
        planner_parallel_slots=2,
        planner_cache_type_k="q8_0",
        planner_cache_type_v="q8_0",
        dialogue_host="127.0.0.1",
        dialogue_port=8094,
        dialogue_model_path="qwen-dialogue.gguf",
        dialogue_llama_server="llama-server.exe",
        dialogue_gpu_layers=999,
        dialogue_ctx_size=1024,
        dialogue_cache_type_k="q8_0",
        dialogue_cache_type_v="q8_0",
        dialogue_lora_adapter_path="qwen-chat-lora.gguf",
        dialogue_allow_prompt_only=False,
    )

    services = build_managed_services(args)
    system2 = next(service for service in services if service.name == "system2")

    assert "--check-lora-adapter-path" in system2.command
    assert system2.command[system2.command.index("--check-lora-adapter-path") + 1] == "tv-check-lora.gguf"
    assert "--check-lora-scale" in system2.command
    assert system2.command[system2.command.index("--check-lora-scale") + 1] == "0.5"
    assert "--check-session-system-prompt" in system2.command
    assert (
        system2.command[system2.command.index("--check-session-system-prompt") + 1]
        == "Answer true only if the TV is clearly off."
    )
