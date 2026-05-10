"""Dialogue LLM server launcher owned by the inference subsystem."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL = REPO_ROOT / "artifacts" / "models" / "Qwen3-1.7B-Q4_K_M-Instruct.gguf"
DEFAULT_LLAMA_HOME = Path(os.environ.get("LLAMA_CPP_HOME", REPO_ROOT / "llama.cpp"))
TRUE_VALUES = frozenset(("1", "true", "yes", "on"))


def _default_llama_exe() -> Path:
    if os.name == "nt":
        return DEFAULT_LLAMA_HOME / "llama-server.exe"
    return DEFAULT_LLAMA_HOME / "llama-server"


def _llama_server_help_text(llama_server: Path) -> str | None:
    try:
        completed = subprocess.run(
            [str(llama_server), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return f"{completed.stdout}\n{completed.stderr}"


def _help_has_exact_flag(help_text: str, flag: str) -> bool:
    return re.search(rf"(?m)^\s*{re.escape(flag)}(?:\s|$)", help_text) is not None


def _llama_server_supports_reasoning_flags(llama_server: Path) -> bool:
    help_text = _llama_server_help_text(llama_server)
    return False if help_text is None else _help_has_exact_flag(help_text, "--reasoning")


def _llama_server_supports_no_think_flags(llama_server: Path) -> bool:
    help_text = _llama_server_help_text(llama_server)
    return (
        False
        if help_text is None
        else _help_has_exact_flag(help_text, "--reasoning-budget")
        and _help_has_exact_flag(help_text, "--reasoning-format")
    )


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in TRUE_VALUES


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the dialogue llama.cpp server.")
    parser.add_argument("--host", default=os.environ.get("DIALOGUE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("DIALOGUE_PORT", "8094") or 8094))
    parser.add_argument("--model", default=os.environ.get("DIALOGUE_MODEL_PATH", str(DEFAULT_MODEL)))
    parser.add_argument("--llama-server", default=os.environ.get("DIALOGUE_LLAMA_SERVER", str(_default_llama_exe())))
    parser.add_argument("--gpu-layers", type=int, default=int(os.environ.get("DIALOGUE_GPU_LAYERS", "999") or 999))
    parser.add_argument("--ctx-size", type=int, default=int(os.environ.get("DIALOGUE_CTX_SIZE", "1024") or 1024))
    parser.add_argument("--cache-type-k", default=os.environ.get("DIALOGUE_CACHE_TYPE_K", "q8_0"))
    parser.add_argument("--cache-type-v", default=os.environ.get("DIALOGUE_CACHE_TYPE_V", "q8_0"))
    parser.add_argument("--lora-adapter-path", default=os.environ.get("DIALOGUE_LORA_ADAPTER_PATH", ""))
    parser.add_argument(
        "--allow-prompt-only",
        action="store_true",
        default=_env_flag("DIALOGUE_ALLOW_PROMPT_ONLY"),
        help="Allow test-only dialogue serving without a chat LoRA adapter; reasoning still applies the dialogue prompt.",
    )
    return parser


def build_command(args: argparse.Namespace) -> list[str]:
    llama_server = Path(args.llama_server)
    command = [
        str(llama_server),
        "-m",
        str(Path(args.model)),
        "--jinja",
    ]
    lora_adapter_path = str(getattr(args, "lora_adapter_path", "") or "").strip()
    if lora_adapter_path:
        command.extend(["--lora", str(Path(lora_adapter_path))])
    if _llama_server_supports_reasoning_flags(llama_server):
        command.extend(["--reasoning", "off"])
    if _llama_server_supports_no_think_flags(llama_server):
        command.extend(["--reasoning-budget", "0", "--reasoning-format", "none"])
    command.extend(
        [
            "-ngl",
            str(int(args.gpu_layers)),
            "-c",
            str(int(args.ctx_size)),
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
        raise SystemExit(f"dialogue model not found: {model}")
    lora_adapter_path = str(args.lora_adapter_path or "").strip()
    allow_prompt_only = bool(getattr(args, "allow_prompt_only", False))
    if lora_adapter_path == "" and not allow_prompt_only:
        raise SystemExit("dialogue LoRA adapter path is required")
    if lora_adapter_path != "":
        lora_adapter = Path(lora_adapter_path)
        if not lora_adapter.is_file():
            raise SystemExit(f"dialogue LoRA adapter not found: {lora_adapter}")
    command = build_command(args)
    completed = subprocess.run(command, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
