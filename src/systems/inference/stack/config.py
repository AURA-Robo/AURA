"""Configuration for the managed inference stack."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(slots=True)
class ManagedServiceConfig:
    name: str
    host: str
    port: int
    health_path: str
    command: list[str]
    required: bool = True

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def health_url(self) -> str:
        return f"{self.base_url}{self.health_path}"


def _python_command() -> str:
    return sys.executable or "python"


def build_managed_services(args) -> list[ManagedServiceConfig]:
    python = _python_command()
    navdp_tensorrt_mode = str(getattr(args, "navdp_tensorrt_mode", "off") or "off")
    navdp_tensorrt_engine_dir = str(getattr(args, "navdp_tensorrt_engine_dir", "") or "")
    navdp_tensorrt_precision = str(getattr(args, "navdp_tensorrt_precision", "fp16") or "fp16")
    navdp_command = [
        python,
        "-m",
        "systems.inference.navdp.server",
        "--port",
        str(int(args.navdp_port)),
        "--checkpoint",
        str(args.navdp_checkpoint),
        "--device",
        str(args.navdp_device),
        "--tensorrt-mode",
        navdp_tensorrt_mode,
        "--tensorrt-precision",
        navdp_tensorrt_precision,
    ]
    if navdp_tensorrt_engine_dir:
        navdp_command.extend(["--tensorrt-engine-dir", navdp_tensorrt_engine_dir])
    system2_command = [
        python,
        "-m",
        "systems.inference.system2.server",
        "--host",
        str(args.system2_host),
        "--port",
        str(int(args.system2_port)),
        "--llama-url",
        str(args.system2_llama_url),
    ]
    if str(args.system2_model_path).strip():
        system2_command.extend(["--model-path", str(args.system2_model_path)])
    system2_check_lora_adapter_path = str(
        getattr(args, "system2_check_lora_adapter_path", "") or ""
    ).strip()
    if system2_check_lora_adapter_path:
        system2_command.extend(["--check-lora-adapter-path", system2_check_lora_adapter_path])
    system2_command.extend(
        [
            "--check-lora-scale",
            str(float(getattr(args, "system2_check_lora_scale", 1.0))),
        ]
    )
    system2_check_session_system_prompt = str(
        getattr(args, "system2_check_session_system_prompt", "") or ""
    ).strip()
    if system2_check_session_system_prompt:
        system2_command.extend(["--check-session-system-prompt", system2_check_session_system_prompt])
    planner_command = [
        python,
        "-m",
        "systems.inference.planner.server",
        "--host",
        str(args.planner_host),
        "--port",
        str(int(args.planner_port)),
        "--model",
        str(args.planner_model_path),
        "--llama-server",
        str(args.planner_llama_server),
        "--gpu-layers",
        str(int(args.planner_gpu_layers)),
        "--ctx-size",
        str(int(args.planner_ctx_size)),
        "--parallel-slots",
        str(int(args.planner_parallel_slots)),
        "--cache-type-k",
        str(args.planner_cache_type_k),
        "--cache-type-v",
        str(args.planner_cache_type_v),
    ]
    dialogue_command = [
        python,
        "-m",
        "systems.inference.dialogue.server",
        "--host",
        str(args.dialogue_host),
        "--port",
        str(int(args.dialogue_port)),
        "--model",
        str(args.dialogue_model_path),
        "--llama-server",
        str(args.dialogue_llama_server),
        "--gpu-layers",
        str(int(args.dialogue_gpu_layers)),
        "--ctx-size",
        str(int(args.dialogue_ctx_size)),
        "--cache-type-k",
        str(args.dialogue_cache_type_k),
        "--cache-type-v",
        str(args.dialogue_cache_type_v),
        "--lora-adapter-path",
        str(args.dialogue_lora_adapter_path),
    ]
    if bool(getattr(args, "dialogue_allow_prompt_only", False)):
        dialogue_command.append("--allow-prompt-only")
    return [
        ManagedServiceConfig(
            name="navdp",
            host=str(args.navdp_host),
            port=int(args.navdp_port),
            health_path="/healthz",
            command=navdp_command,
            required=False,
        ),
        ManagedServiceConfig(
            name="system2",
            host=str(args.system2_host),
            port=int(args.system2_port),
            health_path="/healthz",
            command=system2_command,
        ),
        ManagedServiceConfig(
            name="planner",
            host=str(args.planner_host),
            port=int(args.planner_port),
            health_path="/health",
            command=planner_command,
            required=False,
        ),
        ManagedServiceConfig(
            name="reasoning_dialogue",
            host=str(args.dialogue_host),
            port=int(args.dialogue_port),
            health_path="/health",
            command=dialogue_command,
            required=False,
        ),
    ]


def default_log_dir() -> Path:
    return Path(os.environ.get("AURA_INFERENCE_STACK_LOG_DIR", REPO_ROOT / "logs" / "inference_stack"))
