"""Planner LLM server launcher owned by the inference subsystem."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "models" / "Qwen3-1.7B-Q4_K_M-Instruct.gguf"
DEFAULT_LLAMA_HOME = Path(os.environ.get("LLAMA_CPP_HOME", REPO_ROOT / "llama.cpp"))


def _default_llama_exe() -> Path:
    if os.name == "nt":
        return DEFAULT_LLAMA_HOME / "llama-server.exe"
    return DEFAULT_LLAMA_HOME / "llama-server"


def _llama_server_supports_reasoning_flags(llama_server: Path) -> bool:
    try:
        completed = subprocess.run(
            [str(llama_server), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    help_text = f"{completed.stdout}\n{completed.stderr}"
    return "--reasoning" in help_text


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the planner llama.cpp server.")
    parser.add_argument("--host", default=os.environ.get("PLANNER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PLANNER_PORT", "8093") or 8093))
    parser.add_argument("--model", default=os.environ.get("PLANNER_MODEL_PATH", str(DEFAULT_MODEL)))
    parser.add_argument("--llama-server", default=os.environ.get("PLANNER_LLAMA_SERVER", str(_default_llama_exe())))
    parser.add_argument("--gpu-layers", type=int, default=int(os.environ.get("PLANNER_GPU_LAYERS", "999") or 999))
    parser.add_argument("--ctx-size", type=int, default=int(os.environ.get("PLANNER_CTX_SIZE", "1024") or 1024))
    parser.add_argument(
        "--parallel-slots",
        type=int,
        default=int(os.environ.get("PLANNER_PARALLEL_SLOTS", "2") or 2),
    )
    parser.add_argument("--cache-type-k", default=os.environ.get("PLANNER_CACHE_TYPE_K", "q8_0"))
    parser.add_argument("--cache-type-v", default=os.environ.get("PLANNER_CACHE_TYPE_V", "q8_0"))
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    llama_server = Path(args.llama_server)
    parallel_slots = max(1, int(args.parallel_slots))
    total_ctx_size = max(1, int(args.ctx_size)) * parallel_slots
    command = [
        str(llama_server),
        "-m",
        str(Path(args.model)),
        "--jinja",
    ]
    if _llama_server_supports_reasoning_flags(llama_server):
        command.extend(
            [
                "--reasoning",
                "off",
                "--reasoning-budget",
                "0",
                "--reasoning-format",
                "none",
            ]
        )
    command.extend(
        [
            "-ngl",
            str(int(args.gpu_layers)),
            "-c",
            str(total_ctx_size),
            "--parallel",
            str(parallel_slots),
            "-ctk",
            str(args.cache_type_k),
            "-ctv",
            str(args.cache_type_v),
            "--host",
            str(args.host),
            "--port",
            str(int(args.port)),
        ]
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    llama_server = Path(args.llama_server)
    model = Path(args.model)
    if not llama_server.is_file():
        raise SystemExit(f"llama server not found: {llama_server}")
    if not model.is_file():
        raise SystemExit(f"planner model not found: {model}")
    command = build_command(args)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
