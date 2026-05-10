#!/usr/bin/env python3
from __future__ import annotations

import atexit
from argparse import ArgumentParser, Namespace
import base64
import cgi
import copy
from dataclasses import dataclass
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlsplit

try:
    from flask import Flask, jsonify, request
except ImportError:
    Flask = None
    jsonify = None
    request = None
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from systems.shared.contracts.observation import decode_rgb_history_npz


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 15801
DEFAULT_BACKEND = "llama_cpp"
DEFAULT_LLAMA_URL = "http://127.0.0.1:15802"
DEFAULT_LLAMA_CTX_SIZE = 8192
DEFAULT_LLAMA_MAX_TOKENS = 16
DEFAULT_LLAMA_FLASH_ATTN = "on"
DEFAULT_LLAMA_GPU_LAYERS = "all"
DEFAULT_LLAMA_MAIN_GPU = 0
DEFAULT_LLAMA_PARALLEL_SLOTS = 1
DEFAULT_LLAMA_NAV_SLOT = 0
DEFAULT_LLAMA_CHECK_SLOT = 1
DEFAULT_LLAMA_CACHE_TYPE_K = "q8_0"
DEFAULT_LLAMA_CACHE_TYPE_V = "q8_0"
DEFAULT_LLAMA_HEALTH_TIMEOUT_S = 300.0
DEFAULT_LLAMA_HEALTH_POLL_TIMEOUT_S = 0.25
DEFAULT_LLAMA_REQUEST_TIMEOUT_S = 30.0
DEFAULT_NAVIGATION_ROUTE = "/navigation"
LEGACY_EVAL_DUAL_ROUTE = "/eval_dual"
DEFAULT_CHAT_SESSION_ID = "chat-default"
DEFAULT_CHAT_SESSION_SYSTEM_PROMPT = """?덈뒗 ?먯뿰?ㅻ읇寃???뷀븯???쒓뎅??議곕젰?먮떎.
?듭떖遺??遺꾨챸?섍쾶, ?ㅼ젣 ??붿쿂??遺?쒕읇寃??듯븳??
?ъ슜?먯쓽 留먰닾? 遺꾩쐞湲곗뿉 留욎텛???깅뵳???덈궡臾몄껜???쇳븳??
紐⑤Ⅴ硫??붿쭅?섍쾶 留먰븳??"""
DEFAULT_CHECK_SESSION_ID = "check-default"
DEFAULT_CHECK_SESSION_SYSTEM_PROMPT = (
    "You are a binary visual verifier attached to the InternVLA navigation server. "
    "Inspect the current image and answer the user's yes-or-no subgoal question. "
    "Respond with exactly one lowercase token: true or false. "
    "If the image does not clearly support true, respond false."
)
DEFAULT_CHECK_LORA_SCALE = 1.0
LLAMA_LOOK_DOWN_IMAGE_TOKEN_BUDGET = 196
LLAMA_LOOK_DOWN_TEXT_TOKEN_RESERVE = 320
AURA_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LLAMA_CPP_ROOT = AURA_ROOT / "llama.cpp"
DEFAULT_LLAMA_MODEL_PATH = (
    AURA_ROOT / "artifacts" / "models" / "InternVLA-N1-System2.Q4_K_M.gguf"
)
DEFAULT_LLAMA_MMPROJ_PATH = (
    AURA_ROOT / "artifacts" / "models" / "InternVLA-N1-System2.mmproj-Q8_0.gguf"
)
DEFAULT_ARTIFACTS_DIR = AURA_ROOT / "artifacts"
DEFAULT_IMAGE_TOKEN = "<image>"
LLAMA_MTMD_MEDIA_MARKER = "<__media__>"
HF_QWEN_IMAGE_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"
HF_QWEN_VIDEO_MARKER = "<|vision_start|><|video_pad|><|vision_end|>"
MISSING_MODEL_SENTINEL = "__missing_internvla_model_path__"
IDENTITY_4X4 = np.eye(4, dtype=np.float32)
PROMPT_TEMPLATE = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
CONJUNCTIONS = (
    "you can see ",
    "in front of you is ",
    "there is ",
    "you can spot ",
    "you are toward the ",
    "ahead of you is ",
    "in your sight is ",
)
ACTION_TO_ID = {
    "STOP": [0],
    "\u2191": [1],
    "\u2190": [2],
    "\u2192": [3],
    "\u2193": [5],
}
ACTION_ID_TO_MODE = {
    0: "stop",
    1: "forward",
    2: "yaw_left",
    3: "yaw_right",
    5: "wait",
}
QUANT_STATE_SUFFIXES = (
    ".absmax",
    ".quant_map",
    ".nested_absmax",
    ".nested_quant_map",
    ".quant_state.bitsandbytes__nf4",
    ".quant_state.bitsandbytes__fp4",
)


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return default
    return str(raw).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    return int(str(raw).strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    return float(str(raw).strip())


def _env_optional_str(name: str) -> str | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def _normalize_cache_type(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() in {"none", "off", "disable", "disabled", "false", "0"}:
        return None
    return normalized


def _flag_is_true(value: Any) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def _normalize_backend(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"", DEFAULT_BACKEND, "llama_cpp"}:
        return "llama_cpp"
    raise SystemExit(f"INTERNVLA_BACKEND only supports llama_cpp. Got {value!r}.")


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Thin InternNav System2 HTTP server exposing /navigation.")
    parser.add_argument("--host", default=_env_str("INTERNVLA_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=_env_int("INTERNVLA_PORT", DEFAULT_PORT))
    parser.add_argument("--backend", default=_env_str("INTERNVLA_BACKEND", DEFAULT_BACKEND))
    parser.add_argument("--skip-system1-trajectory", default=_env_str("INTERNVLA_SKIP_SYSTEM1_TRAJECTORY", "1"))
    parser.add_argument("--inference-only", default=_env_str("INTERNVLA_INFERENCE_ONLY", "1"))
    parser.add_argument("--save-debug-artifacts", default=_env_str("INTERNVLA_SAVE_DEBUG_ARTIFACTS", "0"))
    parser.add_argument("--enable-tf32", default=_env_str("INTERNVLA_ENABLE_TF32", "1"))
    parser.add_argument("--max-new-tokens", type=int, default=_env_int("INTERNVLA_MAX_NEW_TOKENS", DEFAULT_LLAMA_MAX_TOKENS))
    parser.add_argument("--model-path", type=Path, default=Path(_env_str("INTERNVLA_MODEL_PATH", MISSING_MODEL_SENTINEL)))
    parser.add_argument("--resize-w", type=int, default=_env_int("INTERNVLA_RESIZE_W", 384))
    parser.add_argument("--resize-h", type=int, default=_env_int("INTERNVLA_RESIZE_H", 384))
    parser.add_argument("--num-history", type=int, default=_env_int("INTERNVLA_NUM_HISTORY", 4))
    parser.add_argument("--plan-step-gap", type=int, default=_env_int("INTERNVLA_PLAN_STEP_GAP", 4))
    parser.add_argument("--llama-cpp-root", type=Path, default=Path(_env_str("INTERNVLA_LLAMA_CPP_ROOT", str(DEFAULT_LLAMA_CPP_ROOT))))
    parser.add_argument("--llama-model-path", type=Path, default=Path(_env_str("INTERNVLA_LLAMA_MODEL_PATH", str(DEFAULT_LLAMA_MODEL_PATH))))
    parser.add_argument("--llama-mmproj-path", type=Path, default=Path(_env_str("INTERNVLA_LLAMA_MMPROJ_PATH", str(DEFAULT_LLAMA_MMPROJ_PATH))))
    parser.add_argument("--llama-url", default=_env_str("INTERNVLA_LLAMA_URL", DEFAULT_LLAMA_URL))
    parser.add_argument("--llama-ctx-size", type=int, default=_env_int("INTERNVLA_LLAMA_CTX_SIZE", DEFAULT_LLAMA_CTX_SIZE))
    parser.add_argument("--llama-threads", default=_env_optional_str("INTERNVLA_LLAMA_THREADS"))
    parser.add_argument("--llama-gpu-layers", default=_env_str("INTERNVLA_LLAMA_GPU_LAYERS", DEFAULT_LLAMA_GPU_LAYERS))
    parser.add_argument("--llama-main-gpu", type=int, default=_env_int("INTERNVLA_LLAMA_MAIN_GPU", DEFAULT_LLAMA_MAIN_GPU))
    parser.add_argument(
        "--llama-parallel-slots",
        type=int,
        default=_env_int("INTERNVLA_LLAMA_PARALLEL_SLOTS", 0),
    )
    parser.add_argument(
        "--llama-nav-slot",
        type=int,
        default=_env_int("INTERNVLA_LLAMA_NAV_SLOT", DEFAULT_LLAMA_NAV_SLOT),
    )
    parser.add_argument(
        "--llama-check-slot",
        type=int,
        default=_env_int("INTERNVLA_LLAMA_CHECK_SLOT", DEFAULT_LLAMA_CHECK_SLOT),
    )
    parser.add_argument(
        "--llama-cache-type-k",
        default=_env_str("INTERNVLA_LLAMA_CACHE_TYPE_K", DEFAULT_LLAMA_CACHE_TYPE_K),
    )
    parser.add_argument(
        "--llama-cache-type-v",
        default=_env_str("INTERNVLA_LLAMA_CACHE_TYPE_V", DEFAULT_LLAMA_CACHE_TYPE_V),
    )
    parser.add_argument(
        "--check-session-system-prompt",
        "--check-system-prompt",
        dest="check_session_system_prompt",
        default=_env_str(
            "INTERNVLA_CHECK_SESSION_SYSTEM_PROMPT",
            _env_str("INTERNVLA_CHECK_SYSTEM_PROMPT", DEFAULT_CHECK_SESSION_SYSTEM_PROMPT),
        ),
    )
    parser.add_argument(
        "--check-lora-adapter-path",
        dest="check_lora_adapter_path",
        default=_env_str("INTERNVLA_CHECK_LORA_ADAPTER_PATH", ""),
    )
    parser.add_argument(
        "--check-lora-scale",
        dest="check_lora_scale",
        type=float,
        default=_env_float("INTERNVLA_CHECK_LORA_SCALE", DEFAULT_CHECK_LORA_SCALE),
    )
    parser.add_argument(
        "--default-check-session-id",
        dest="default_check_session_id",
        default=_env_str("INTERNVLA_DEFAULT_CHECK_SESSION_ID", DEFAULT_CHECK_SESSION_ID),
    )
    parser.add_argument(
        "--default-check-session-auto-open",
        dest="default_check_session_auto_open",
        default=_env_str("INTERNVLA_DEFAULT_CHECK_SESSION_AUTO_OPEN", "1"),
    )
    return parser.parse_args(argv)


def _require_requests():
    try:
        import requests
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Missing required Python package: requests ({exc})") from exc
    return requests


def _validate_llama_server_executable(llama_cpp_root: Path) -> tuple[Path, Path]:
    resolved_root = Path(llama_cpp_root).expanduser().resolve()
    if not resolved_root.exists():
        raise SystemExit(f"INTERNVLA_LLAMA_CPP_ROOT does not exist: {resolved_root}")
    llama_server = resolved_root / "llama-server.exe"
    if not llama_server.exists():
        raise SystemExit(f"llama-server.exe not found under INTERNVLA_LLAMA_CPP_ROOT: {llama_server}")
    return resolved_root, llama_server


def _validate_llama_file(path: Path, env_name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{env_name} does not exist: {resolved}")
    return resolved


def _normalize_llama_url(value: str) -> str:
    raw = str(value).strip().rstrip("/")
    if not raw:
        raise SystemExit("INTERNVLA_LLAMA_URL is required for llama_cpp backend.")
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise SystemExit(f"INTERNVLA_LLAMA_URL must be an absolute http(s) URL. Got {value!r}.")
    return raw


def _llama_url_host_port(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    host = parsed.hostname or DEFAULT_HOST
    port = parsed.port or 15802
    return host, int(port)


def prepare_runtime_args(args: Namespace) -> Namespace:
    args.backend = _normalize_backend(getattr(args, "backend", DEFAULT_BACKEND))
    _require_requests()
    try:
        from PIL import Image as _Image  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Missing required Python package: Pillow ({exc})") from exc
    args.llama_cpp_root, args.llama_server_path = _validate_llama_server_executable(Path(args.llama_cpp_root))
    args.llama_model_path = _validate_llama_file(Path(args.llama_model_path), "INTERNVLA_LLAMA_MODEL_PATH")
    args.llama_mmproj_path = _validate_llama_file(Path(args.llama_mmproj_path), "INTERNVLA_LLAMA_MMPROJ_PATH")
    args.llama_url = _normalize_llama_url(args.llama_url)
    args.llama_ctx_size = int(getattr(args, "llama_ctx_size", DEFAULT_LLAMA_CTX_SIZE))
    if args.llama_ctx_size <= 0:
        raise SystemExit("INTERNVLA_LLAMA_CTX_SIZE must be a positive integer.")
    args.llama_nav_slot = int(getattr(args, "llama_nav_slot", DEFAULT_LLAMA_NAV_SLOT))
    if args.llama_nav_slot < 0:
        raise SystemExit("INTERNVLA_LLAMA_NAV_SLOT must be a non-negative integer.")
    args.llama_check_slot = int(getattr(args, "llama_check_slot", DEFAULT_LLAMA_CHECK_SLOT))
    if args.llama_check_slot < 0:
        raise SystemExit("INTERNVLA_LLAMA_CHECK_SLOT must be a non-negative integer.")
    args.llama_parallel_slots = int(getattr(args, "llama_parallel_slots", 0))
    if args.llama_parallel_slots <= 0:
        args.llama_parallel_slots = 2
    args.llama_parallel_slots = max(
        int(args.llama_parallel_slots),
        2,
        int(args.llama_nav_slot) + 1,
        int(args.llama_check_slot) + 1,
    )
    args.llama_cache_type_k = _normalize_cache_type(
        getattr(args, "llama_cache_type_k", DEFAULT_LLAMA_CACHE_TYPE_K)
    )
    args.llama_cache_type_v = _normalize_cache_type(
        getattr(args, "llama_cache_type_v", DEFAULT_LLAMA_CACHE_TYPE_V)
    )
    check_lora_adapter_path = str(getattr(args, "check_lora_adapter_path", "") or "").strip()
    args.check_lora_adapter_path = (
        _validate_llama_file(Path(check_lora_adapter_path), "INTERNVLA_CHECK_LORA_ADAPTER_PATH")
        if check_lora_adapter_path
        else None
    )
    try:
        args.check_lora_scale = float(getattr(args, "check_lora_scale", DEFAULT_CHECK_LORA_SCALE))
    except (TypeError, ValueError) as exc:
        raise SystemExit("INTERNVLA_CHECK_LORA_SCALE must be a positive number.") from exc
    if args.check_lora_scale <= 0.0:
        raise SystemExit("INTERNVLA_CHECK_LORA_SCALE must be positive.")
    args.default_check_session_id = (
        str(getattr(args, "default_check_session_id", DEFAULT_CHECK_SESSION_ID)).strip()
        or DEFAULT_CHECK_SESSION_ID
    )
    args.default_check_session_auto_open = _flag_is_true(
        getattr(args, "default_check_session_auto_open", "1")
    )
    args.skip_system1_trajectory = "1" if _flag_is_true(getattr(args, "skip_system1_trajectory", "1")) else "0"
    return args


def split_and_clean(text: str) -> list[str]:
    parts = re.split(r"(<image>)", text)
    results: list[str] = []
    for part in parts:
        if part == DEFAULT_IMAGE_TOKEN:
            results.append(part)
            continue
        clean_part = part.replace("\n", "").strip()
        if clean_part:
            results.append(clean_part)
    return results


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, str):
                texts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    texts.append(text_value)
        return "".join(texts).strip()
    if isinstance(content, dict):
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value.strip()
    return str(content or "").strip()


def _normalize_check_question(content: Any) -> str:
    if not isinstance(content, str):
        raise ValueError("message must be a string.")
    normalized = " ".join(content.strip().split())
    if not normalized:
        raise ValueError("message must not be empty.")
    return normalized


def _data_url_to_base64_payload(url: str) -> str:
    text = str(url).strip()
    if not text:
        raise ValueError("image_url cannot be empty")
    if text.startswith("data:"):
        _, _, payload = text.partition(",")
        if not payload:
            raise ValueError("image_url data URL does not include a base64 payload")
        return payload
    raise ValueError("Only data:image/...;base64 URLs are supported for low-level multimodal prompts")


def _build_hf_chat_template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user") or "user").strip() or "user"
        content = message.get("content", "")
        if isinstance(content, str):
            serialized.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            serialized.append({"role": role, "content": _extract_text_content(content)})
            continue
        parts: list[dict[str, str]] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                parts.append({"type": "image"})
                continue
            text_value = ""
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_value = item.get("text", "")
            elif isinstance(item, str):
                text_value = item
            else:
                text_value = _extract_text_content(item)
            if text_value:
                parts.append({"type": "text", "text": text_value})
        serialized.append({"role": role, "content": parts})
    return serialized


def _serialize_hf_chat_template_prompt(
    processor: Any,
    messages: list[dict[str, Any]],
    *,
    assistant_response: str | None = None,
) -> dict[str, Any]:
    prompt_messages = _build_hf_chat_template_messages(messages)
    prompt_string = str(
        processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    )
    prompt_string = prompt_string.replace(HF_QWEN_IMAGE_MARKER, LLAMA_MTMD_MEDIA_MARKER)
    prompt_string = prompt_string.replace(HF_QWEN_VIDEO_MARKER, LLAMA_MTMD_MEDIA_MARKER)
    if assistant_response is not None:
        prompt_string += str(assistant_response).strip()

    multimodal_data: list[str] = []
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_url = item.get("image_url", {})
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url")
            if not isinstance(url, str) or not url.strip():
                continue
            multimodal_data.append(_data_url_to_base64_payload(url))

    payload: dict[str, Any] = {"prompt_string": prompt_string}
    if multimodal_data:
        payload["multimodal_data"] = multimodal_data
    return payload


def _build_llama_server_command(
    *,
    server_exe: Path,
    host: str,
    port: int,
    model_path: Path,
    mmproj_path: Path,
    ctx_size: int = DEFAULT_LLAMA_CTX_SIZE,
    threads: str | None = None,
    gpu_layers: str = DEFAULT_LLAMA_GPU_LAYERS,
    main_gpu: int = DEFAULT_LLAMA_MAIN_GPU,
    max_tokens: int = DEFAULT_LLAMA_MAX_TOKENS,
    flash_attn: str = DEFAULT_LLAMA_FLASH_ATTN,
    parallel_slots: int = DEFAULT_LLAMA_PARALLEL_SLOTS,
    cache_type_k: str | None = DEFAULT_LLAMA_CACHE_TYPE_K,
    cache_type_v: str | None = DEFAULT_LLAMA_CACHE_TYPE_V,
    chat_lora_path: Path | None = None,
    lora_init_without_apply: bool = True,
) -> list[str]:
    normalized_parallel_slots = max(1, int(parallel_slots))
    command = [
        str(server_exe),
        "--host",
        str(host),
        "--port",
        str(port),
        "--model",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--flash-attn",
        str(flash_attn),
        "--ctx-size",
        str(int(ctx_size) * normalized_parallel_slots),
        "--parallel",
        str(normalized_parallel_slots),
        "--embeddings",
        "--predict",
        str(int(max_tokens)),
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1",
        "--no-warmup",
    ]
    if threads is not None and str(threads).strip():
        command.extend(["--threads", str(threads).strip()])
    if str(gpu_layers).strip():
        command.extend(["--gpu-layers", str(gpu_layers).strip()])
    normalized_cache_type_k = _normalize_cache_type(cache_type_k)
    if normalized_cache_type_k is not None:
        command.extend(["--cache-type-k", normalized_cache_type_k])
    normalized_cache_type_v = _normalize_cache_type(cache_type_v)
    if normalized_cache_type_v is not None:
        command.extend(["--cache-type-v", normalized_cache_type_v])
    if chat_lora_path is not None:
        command.extend(["--lora", str(chat_lora_path)])
        if lora_init_without_apply:
            command.append("--lora-init-without-apply")
    command.extend(["--main-gpu", str(int(main_gpu))])
    return command


@dataclass(slots=True)
class DualSystemOutput:
    output_action: list[int] | None = None
    output_pixel: list[int] | None = None
    output_trajectory: list[list[float]] | None = None


@dataclass(slots=True)
class CheckSessionState:
    session_id: str
    created_at_s: float
    updated_at_s: float


class LlamaCppSidecarManager:
    def __init__(self, args: Namespace, *, auto_start: bool = True):
        self._args = args
        self._requests = _require_requests()
        self.root = Path(args.llama_cpp_root).expanduser().resolve()
        self.server_exe = Path(args.llama_server_path).expanduser().resolve()
        self.model_path = Path(args.llama_model_path).expanduser().resolve()
        self.mmproj_path = Path(args.llama_mmproj_path).expanduser().resolve()
        self.base_url = _normalize_llama_url(str(args.llama_url))
        self.host, self.port = _llama_url_host_port(self.base_url)
        self.model_name = self.model_path.name
        self.log_dir = DEFAULT_ARTIFACTS_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.stdout_log = self.log_dir / "internvla_llama_stdout.log"
        self.stderr_log = self.log_dir / "internvla_llama_stderr.log"
        self.ctx_size = int(getattr(args, "llama_ctx_size", DEFAULT_LLAMA_CTX_SIZE))
        self.parallel_slots = max(1, int(getattr(args, "llama_parallel_slots", DEFAULT_LLAMA_PARALLEL_SLOTS)))
        self.total_ctx_size = self.ctx_size * self.parallel_slots
        self.health_timeout_s = float(
            os.environ.get("INTERNVLA_LLAMA_HEALTH_TIMEOUT_S", DEFAULT_LLAMA_HEALTH_TIMEOUT_S)
        )
        self.embeddings = True
        self.cache_type_k = _normalize_cache_type(getattr(args, "llama_cache_type_k", DEFAULT_LLAMA_CACHE_TYPE_K))
        self.cache_type_v = _normalize_cache_type(getattr(args, "llama_cache_type_v", DEFAULT_LLAMA_CACHE_TYPE_V))
        check_lora_path = str(getattr(args, "check_lora_adapter_path", "") or "").strip()
        self.check_lora_path = Path(check_lora_path).expanduser().resolve() if check_lora_path else None
        self.check_lora_scale = float(getattr(args, "check_lora_scale", DEFAULT_CHECK_LORA_SCALE))
        self.hf_prompt_model_path = self._resolve_hf_prompt_model_path(getattr(args, "model_path", None))
        self._hf_prompt_processor = None
        self._hf_prompt_load_attempted = False
        self._hf_prompt_error: str | None = None
        self._hf_low_level_completions_enabled = True
        self._lora_adapters_cache: list[dict[str, Any]] | None = None
        self._check_lora_id: int | None = None
        self._process: subprocess.Popen[str] | None = None
        self._spawned = False
        self._stdout_handle = None
        self._stderr_handle = None
        self._props_cache: dict[str, Any] | None = None
        self._command = _build_llama_server_command(
            server_exe=self.server_exe,
            host=self.host,
            port=self.port,
            model_path=self.model_path,
            mmproj_path=self.mmproj_path,
            ctx_size=self.ctx_size,
            threads=getattr(args, "llama_threads", None),
            gpu_layers=str(getattr(args, "llama_gpu_layers", DEFAULT_LLAMA_GPU_LAYERS)),
            main_gpu=int(getattr(args, "llama_main_gpu", DEFAULT_LLAMA_MAIN_GPU)),
            max_tokens=int(getattr(args, "max_new_tokens", DEFAULT_LLAMA_MAX_TOKENS)),
            parallel_slots=self.parallel_slots,
            cache_type_k=self.cache_type_k,
            cache_type_v=self.cache_type_v,
            chat_lora_path=self.check_lora_path,
            lora_init_without_apply=True,
        )
        if auto_start:
            self._start_process()

    @staticmethod
    def _resolve_hf_prompt_model_path(model_path: Any) -> Path | None:
        if model_path is None:
            return None
        raw = str(model_path).strip()
        if raw in {"", MISSING_MODEL_SENTINEL}:
            return None
        resolved = Path(model_path).expanduser().resolve()
        if not resolved.exists():
            return None
        if not (resolved / "chat_template.json").exists():
            return None
        return resolved

    def _get_hf_prompt_processor(self):
        if self._hf_prompt_load_attempted:
            return self._hf_prompt_processor
        self._hf_prompt_load_attempted = True
        if self.hf_prompt_model_path is None:
            return None
        try:
            from transformers import AutoProcessor
        except Exception as exc:  # noqa: BLE001
            self._hf_prompt_error = f"{type(exc).__name__}: {exc}"
            return None
        try:
            self._hf_prompt_processor = AutoProcessor.from_pretrained(
                str(self.hf_prompt_model_path),
                local_files_only=True,
                use_fast=False,
            )
        except Exception as exc:  # noqa: BLE001
            self._hf_prompt_error = f"{type(exc).__name__}: {exc}"
            self._hf_prompt_processor = None
        return self._hf_prompt_processor

    def _serialize_prompt_from_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        assistant_response: str | None = None,
    ) -> dict[str, Any] | None:
        processor = self._get_hf_prompt_processor()
        if processor is None:
            return None
        return _serialize_hf_chat_template_prompt(
            processor,
            messages,
            assistant_response=assistant_response,
        )

    @property
    def command(self) -> tuple[str, ...]:
        return tuple(self._command)

    @property
    def pid(self) -> int | None:
        if self._process is None or self._process.poll() is not None:
            return None
        return int(self._process.pid)

    def _cleanup_dead_process(self) -> None:
        if self._process is None or self._process.poll() is None:
            return
        self._process = None
        self._props_cache = None
        self._lora_adapters_cache = None
        self._check_lora_id = None
        if self._stdout_handle is not None:
            self._stdout_handle.close()
            self._stdout_handle = None
        if self._stderr_handle is not None:
            self._stderr_handle.close()
            self._stderr_handle = None

    def healthy(self, *, timeout_s: float = DEFAULT_LLAMA_HEALTH_POLL_TIMEOUT_S) -> bool:
        self._cleanup_dead_process()
        try:
            response = self._requests.get(f"{self.base_url}/health", timeout=float(timeout_s))
            return response.status_code == 200
        except Exception:
            return False

    def _get_json(self, path: str, *, timeout_s: float = DEFAULT_LLAMA_REQUEST_TIMEOUT_S) -> dict[str, Any]:
        response = self._requests.get(f"{self.base_url}{path}", timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"Expected JSON object from llama.cpp {path}, got {type(body)!r}.")
        return body

    def list_lora_adapters(self, *, refresh: bool = False) -> list[dict[str, Any]]:
        if self.check_lora_path is None:
            return []
        if refresh or self._lora_adapters_cache is None:
            self.ensure_running()
            response = self._requests.get(f"{self.base_url}/lora-adapters", timeout=DEFAULT_LLAMA_REQUEST_TIMEOUT_S)
            response.raise_for_status()
            body = response.json()
            if not isinstance(body, list):
                raise RuntimeError(f"Expected JSON list from llama.cpp /lora-adapters, got {type(body)!r}.")
            self._lora_adapters_cache = [dict(item) for item in body if isinstance(item, dict)]
        return copy.deepcopy(self._lora_adapters_cache)

    def resolve_check_lora_id(self, *, refresh: bool = False) -> int | None:
        if self.check_lora_path is None:
            return None
        if self._check_lora_id is not None and not refresh:
            return int(self._check_lora_id)
        target = str(self.check_lora_path).lower()
        target_name = self.check_lora_path.name.lower()
        adapters = self.list_lora_adapters(refresh=refresh)
        for adapter in adapters:
            adapter_id = adapter.get("id")
            adapter_path = str(adapter.get("path", "")).strip()
            if adapter_id is None or not adapter_path:
                continue
            adapter_name = Path(adapter_path).name.lower()
            adapter_resolved = str(Path(adapter_path).expanduser()).lower()
            if adapter_name == target_name or adapter_resolved == target or adapter_path.lower() == target:
                self._check_lora_id = int(adapter_id)
                return int(self._check_lora_id)
        raise RuntimeError(
            "Configured check LoRA adapter was not reported by llama.cpp /lora-adapters: "
            f"path={self.check_lora_path}"
        )

    def check_lora_request(self) -> list[dict[str, Any]]:
        if self.check_lora_path is None:
            return []
        return [{"id": int(self.resolve_check_lora_id()), "scale": float(self.check_lora_scale)}]

    def resolve_chat_lora_id(self, *, refresh: bool = False) -> int | None:
        return self.resolve_check_lora_id(refresh=refresh)

    def chat_lora_request(self) -> list[dict[str, Any]]:
        return self.check_lora_request()

    def props(self, *, refresh: bool = False) -> dict[str, Any]:
        self._cleanup_dead_process()
        if not self.healthy():
            return {}
        if refresh or self._props_cache is None:
            try:
                self._props_cache = self._get_json("/props")
            except Exception:
                self._props_cache = {}
        return dict(self._props_cache)

    def ready(self, *, refresh: bool = False) -> bool:
        del refresh
        return self.healthy()

    def slot_erase(self, slot_id: int) -> dict[str, Any]:
        normalized_slot_id = int(slot_id)
        if normalized_slot_id < 0:
            raise ValueError("slot_id must be a non-negative integer.")
        self.ensure_running()
        response = self._requests.post(
            f"{self.base_url}/slots/{normalized_slot_id}?action=erase",
            json={},
            timeout=DEFAULT_LLAMA_REQUEST_TIMEOUT_S,
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"Expected JSON object from llama.cpp slot erase, got {type(body)!r}.")
        return body

    def ensure_running(self) -> None:
        if self.healthy(timeout_s=0.5):
            return
        self._start_process()
        deadline = time.monotonic() + self.health_timeout_s
        while time.monotonic() < deadline:
            if self.healthy(timeout_s=0.5):
                return
            if self._process is not None and self._process.poll() is not None:
                break
            time.sleep(0.5)
        raise RuntimeError(
            "Timed out waiting for llama.cpp sidecar readiness at "
            f"{self.base_url} using model={self.model_path} mmproj={self.mmproj_path} "
            f"(timeout={self.health_timeout_s:.1f}s, stderr_log={self.stderr_log}, stdout_log={self.stdout_log})"
        )

    def _start_process(self) -> None:
        self._cleanup_dead_process()
        if self._process is not None and self._process.poll() is None:
            return
        self.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        self._props_cache = None
        self._lora_adapters_cache = None
        self._check_lora_id = None
        self._stdout_handle = open(self.stdout_log, "w", encoding="utf-8")
        self._stderr_handle = open(self.stderr_log, "w", encoding="utf-8")
        self._process = subprocess.Popen(
            self._command,
            cwd=str(self.root),
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
        )
        self._spawned = True

    def close(self) -> None:
        try:
            if self._spawned and self._process is not None and self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5.0)
        finally:
            if self._stdout_handle is not None:
                self._stdout_handle.close()
                self._stdout_handle = None
            if self._stderr_handle is not None:
                self._stderr_handle.close()
                self._stderr_handle = None
            self._process = None
            self._props_cache = None
            self._lora_adapters_cache = None
            self._check_lora_id = None

    def health_payload(self) -> dict[str, Any]:
        health_now = self.healthy()
        payload = {
            "llama_url": self.base_url,
            "llama_cpp_root": str(self.root),
            "llama_server_path": str(self.server_exe),
            "llama_model_path": str(self.model_path),
            "llama_mmproj_path": str(self.mmproj_path),
            "llama_sidecar_ready": health_now,
            "llama_sidecar_spawned": bool(self._spawned),
            "llama_sidecar_pid": self.pid,
            "llama_command": list(self.command),
            "llama_stdout_log": str(self.stdout_log),
            "llama_stderr_log": str(self.stderr_log),
            "llama_ctx_size": self.ctx_size,
            "llama_total_ctx_size": self.total_ctx_size,
            "llama_parallel_slots": self.parallel_slots,
            "llama_health_timeout_s": self.health_timeout_s,
            "embeddings": self.embeddings,
            "llama_cache_type_k": self.cache_type_k,
            "llama_cache_type_v": self.cache_type_v,
            "check_lora_configured": self.check_lora_path is not None,
            "check_lora_adapter_path": None if self.check_lora_path is None else str(self.check_lora_path),
            "check_lora_scale": self.check_lora_scale,
            "hf_prompt_serialization": self.hf_prompt_model_path is not None,
            "hf_prompt_model_path": None if self.hf_prompt_model_path is None else str(self.hf_prompt_model_path),
            "hf_low_level_completions_enabled": self._hf_low_level_completions_enabled,
        }
        if self._hf_prompt_error:
            payload["hf_prompt_error"] = self._hf_prompt_error
        return payload

    def chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        max_tokens: int,
        slot_id: int | None = None,
        cache_prompt: bool | None = None,
        lora: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self.ensure_running()
        serialized_prompt = self._serialize_prompt_from_messages(messages)
        if serialized_prompt is None or not self._hf_low_level_completions_enabled:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": int(max_tokens),
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
            }
            if slot_id is not None:
                payload["id_slot"] = int(slot_id)
            if cache_prompt is not None:
                payload["cache_prompt"] = bool(cache_prompt)
            if lora:
                payload["lora"] = copy.deepcopy(lora)
            response = self._requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=DEFAULT_LLAMA_REQUEST_TIMEOUT_S,
            )
        else:
            payload = {
                "prompt": serialized_prompt,
                "n_predict": int(max_tokens),
                "temperature": 0,
                "top_k": 1,
                "top_p": 1,
            }
            if slot_id is not None:
                payload["id_slot"] = int(slot_id)
            if cache_prompt is not None:
                payload["cache_prompt"] = bool(cache_prompt)
            if lora:
                payload["lora"] = copy.deepcopy(lora)
            try:
                response = self._requests.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    timeout=DEFAULT_LLAMA_REQUEST_TIMEOUT_S,
                )
                response.raise_for_status()
            except Exception as exc:
                response_obj = getattr(exc, "response", None)
                if response_obj is None or getattr(response_obj, "status_code", None) not in {400, 404, 405}:
                    raise
                self._hf_low_level_completions_enabled = False
                self._hf_prompt_error = (
                    "HF low-level /completions prompt serialization failed; "
                    "falling back to /v1/chat/completions for this server build."
                )
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": int(max_tokens),
                    "temperature": 0,
                    "top_k": 1,
                    "top_p": 1,
                }
                if slot_id is not None:
                    payload["id_slot"] = int(slot_id)
                if cache_prompt is not None:
                    payload["cache_prompt"] = bool(cache_prompt)
                if lora:
                    payload["lora"] = copy.deepcopy(lora)
                response = self._requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=DEFAULT_LLAMA_REQUEST_TIMEOUT_S,
                )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise RuntimeError(f"Expected JSON object from llama.cpp chat completion, got {type(body)!r}.")
        if "choices" not in body and "content" in body:
            body = {
                "choices": [
                    {
                        "text": _extract_text_content(body.get("content")),
                    }
                ],
                **body,
            }
        return body


class LlamaCppDualVLNRuntime:
    def __init__(self, args: Namespace):
        self._args = args
        self._lock = threading.Lock()
        self._sidecar = LlamaCppSidecarManager(args)
        self._check_session_lock = threading.Lock()
        self._check_slot_lock = threading.Lock()
        self._check_session: CheckSessionState | None = None
        self.resize_w = int(args.resize_w)
        self.resize_h = int(args.resize_h)
        self.num_history = int(args.num_history)
        self.plan_step_gap = int(args.plan_step_gap)
        self.llama_ctx_size = int(getattr(args, "llama_ctx_size", DEFAULT_LLAMA_CTX_SIZE))
        self.llama_parallel_slots = int(
            getattr(args, "llama_parallel_slots", getattr(self._sidecar, "parallel_slots", DEFAULT_LLAMA_PARALLEL_SLOTS))
        )
        self.nav_slot_id = int(getattr(args, "llama_nav_slot", DEFAULT_LLAMA_NAV_SLOT))
        self.check_slot_id = int(getattr(args, "llama_check_slot", DEFAULT_LLAMA_CHECK_SLOT))
        self.max_new_tokens = int(getattr(args, "max_new_tokens", DEFAULT_LLAMA_MAX_TOKENS))
        self.default_check_session_id = str(
            getattr(args, "default_check_session_id", DEFAULT_CHECK_SESSION_ID)
        ).strip() or DEFAULT_CHECK_SESSION_ID
        self.default_check_session_auto_open = bool(
            getattr(args, "default_check_session_auto_open", True)
        )
        max_look_down_images = max(
            2,
            (self.llama_ctx_size - LLAMA_LOOK_DOWN_TEXT_TOKEN_RESERVE) // LLAMA_LOOK_DOWN_IMAGE_TOKEN_BUDGET,
        )
        self.max_look_down_images = max(2, min(self.num_history + 2, max_look_down_images))
        self.actions2idx = copy.deepcopy(ACTION_TO_ID)
        self.conjunctions = tuple(CONJUNCTIONS)
        self.conversation = [{"from": "human", "value": PROMPT_TEMPLATE}, {"from": "gpt", "value": ""}]
        self.check_session_system_prompt = str(
            getattr(args, "check_session_system_prompt", DEFAULT_CHECK_SESSION_SYSTEM_PROMPT)
        ).strip() or DEFAULT_CHECK_SESSION_SYSTEM_PROMPT
        self.reset()
        self._ensure_default_check_session_open()

    def _erase_slot_if_available(self, slot_id: int) -> None:
        if slot_id < 0 or not hasattr(self._sidecar, "slot_erase"):
            return
        try:
            self._sidecar.slot_erase(int(slot_id))
        except Exception:
            return

    def _normalize_check_session_id(self, session_id: Any | None) -> str:
        normalized = str(session_id or "").strip() or self.default_check_session_id
        if normalized != self.default_check_session_id:
            raise ValueError(
                "Only the default check session is supported: "
                f"{self.default_check_session_id}"
            )
        return normalized

    @staticmethod
    def _reject_custom_check_system_prompt(payload: dict[str, Any]) -> None:
        custom_prompt = str(payload.get("system_prompt", "")).strip()
        if custom_prompt:
            raise ValueError("Custom system_prompt is not supported for /check sessions.")

    def _ensure_check_session_locked(self, *, session_id: str) -> tuple[CheckSessionState, bool]:
        existing = self._check_session
        if existing is not None and existing.session_id == session_id:
            return existing, False
        now = time.monotonic()
        session = CheckSessionState(
            session_id=session_id,
            created_at_s=now,
            updated_at_s=now,
        )
        self._check_session = session
        return session, True

    def _ensure_default_check_session_open(self) -> None:
        if not self.default_check_session_auto_open:
            return
        with self._check_session_lock:
            self._ensure_check_session_locked(
                session_id=self.default_check_session_id,
            )

    def _get_latest_check_frame(self) -> Image.Image | None:
        if self._latest_check_image is not None:
            return self._latest_check_image
        if not self.rgb_list:
            return None
        return self.rgb_list[-1]

    def _build_check_messages(self, *, question: str, image: Image.Image) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self.check_session_system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": self._pil_to_data_url(image)}},
                    {"type": "text", "text": f"Question: {question}"},
                ],
            },
        ]

    def reset(self) -> None:
        self.rgb_list: list[Image.Image] = []
        self.depth_list: list[np.ndarray] = []
        self.pose_list: list[np.ndarray] = []
        self.episode_idx = 0
        self.last_s2_idx = -100
        self.conversation_history: list[dict[str, Any]] = []
        self.input_images: list[Image.Image] = []
        self.input_image_keys: list[int | None] = []
        self.llm_output = ""
        self.output_action: list[int] | None = None
        self.output_pixel: list[int] | None = None
        self.pixel_goal_rgb: np.ndarray | None = None
        self.pixel_goal_depth: np.ndarray | None = None
        self.last_decision_mode = "wait"
        self.last_raw_output_text = ""
        self.pixel_goal_count = 0
        self.action_only_count = 0
        self._latest_check_image: Image.Image | None = None
        self._erase_slot_if_available(self.nav_slot_id)
        self._erase_slot_if_available(self.check_slot_id)

    def close(self) -> None:
        with self._check_session_lock:
            self._check_session = None
        self._sidecar.close()

    def upstream_healthy(self) -> bool:
        return self._sidecar.healthy()

    def health_payload(self) -> dict[str, Any]:
        payload = {
            "status": "ok",
            "ready": self.upstream_healthy(),
            "backend": "llama_cpp",
            "resize_w": self.resize_w,
            "resize_h": self.resize_h,
            "num_history": self.num_history,
            "plan_step_gap": self.plan_step_gap,
            "inference_only": "1",
            "save_debug_artifacts": str(getattr(self._args, "save_debug_artifacts", "0")),
            "enable_tf32": str(getattr(self._args, "enable_tf32", "1")),
            "max_new_tokens": self.max_new_tokens,
            "last_decision_mode": self.last_decision_mode,
            "last_raw_output_text": self.last_raw_output_text,
            "pixel_goal_count": self.pixel_goal_count,
            "action_only_count": self.action_only_count,
            "check_session_count": 0 if self._check_session is None else 1,
            "check_session_system_prompt": self.check_session_system_prompt,
            "default_check_session_id": self.default_check_session_id,
            "default_check_session_auto_open": self.default_check_session_auto_open,
            "check_frame_available": self._get_latest_check_frame() is not None,
            "llama_parallel_slots": self.llama_parallel_slots,
            "llama_nav_slot": self.nav_slot_id,
            "llama_check_slot": self.check_slot_id,
            "nav_slot_id": self.nav_slot_id,
            "check_slot_id": self.check_slot_id,
            "llama_cache_type_k": _normalize_cache_type(
                getattr(self._args, "llama_cache_type_k", DEFAULT_LLAMA_CACHE_TYPE_K)
            ),
            "llama_cache_type_v": _normalize_cache_type(
                getattr(self._args, "llama_cache_type_v", DEFAULT_LLAMA_CACHE_TYPE_V)
            ),
        }
        payload.update(self._sidecar.health_payload())
        return payload

    @staticmethod
    def _extract_completion_text(response: dict[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"llama.cpp response does not include choices: {response}")
        first_choice = choices[0]
        if isinstance(first_choice, dict) and "message" in first_choice:
            return _extract_text_content(first_choice.get("message", {}).get("content"))
        if isinstance(first_choice, dict):
            return _extract_text_content(first_choice.get("text"))
        return _extract_text_content(first_choice)

    @classmethod
    def _normalize_check_answer(cls, response: dict[str, Any]) -> str:
        answer = cls._extract_completion_text(response).strip().lower()
        tokens = re.findall(r"[a-z]+", answer)
        if answer in {"true", "false"}:
            return answer
        truth_values = {"true": "true", "yes": "true", "y": "true"}
        false_values = {"false": "false", "no": "false", "n": "false"}
        normalized_values = {
            value
            for token in tokens
            for value in (truth_values.get(token), false_values.get(token))
            if value is not None
        }
        if len(normalized_values) == 1:
            return next(iter(normalized_values))
        raise RuntimeError(f"/check must return a binary true/false answer, got: {answer!r}")

    def open_check_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._reject_custom_check_system_prompt(payload)
        session_id = self._normalize_check_session_id(payload.get("session_id"))
        with self._check_session_lock:
            _, created = self._ensure_check_session_locked(session_id=session_id)
        return {
            "status": "ok",
            "session_id": session_id,
            "created": bool(created),
            "reused": not bool(created),
            "check_slot_id": self.check_slot_id,
            "check_frame_available": self._get_latest_check_frame() is not None,
            "is_default_session": session_id == self.default_check_session_id,
        }

    def check_session_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._reject_custom_check_system_prompt(payload)
        session_id = self._normalize_check_session_id(payload.get("session_id"))
        max_tokens = int(payload.get("max_tokens", min(self.max_new_tokens, 4)))
        if max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        question = _normalize_check_question(payload.get("message"))
        with self._check_session_lock:
            session, created = self._ensure_check_session_locked(session_id=session_id)
        image = self._get_latest_check_frame()
        if image is None:
            session.updated_at_s = time.monotonic()
            return {
                "status": "ok",
                "session_id": session_id,
                "created_session": bool(created),
                "answer": "unknown",
                "check_slot_id": self.check_slot_id,
                "check_frame_available": False,
            }
        request_messages = self._build_check_messages(question=question, image=image)
        with self._check_slot_lock:
            self._erase_slot_if_available(self.check_slot_id)
            response = self._sidecar.chat_completion(
                messages=request_messages,
                max_tokens=max_tokens,
                slot_id=self.check_slot_id,
                cache_prompt=False,
                lora=self._sidecar.check_lora_request(),
            )
        answer = self._normalize_check_answer(response)
        with self._check_session_lock:
            session.updated_at_s = time.monotonic()
        return {
            "status": "ok",
            "session_id": session_id,
            "created_session": bool(created),
            "answer": answer,
            "check_slot_id": self.check_slot_id,
            "check_frame_available": True,
        }

    def close_check_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._reject_custom_check_system_prompt(payload)
        session_id = self._normalize_check_session_id(payload.get("session_id"))
        with self._check_session_lock:
            session = None
            if self._check_session is not None and self._check_session.session_id == session_id:
                session = self._check_session
                self._check_session = None
            reopened = None
            if self.default_check_session_auto_open:
                reopened, _ = self._ensure_check_session_locked(
                    session_id=self.default_check_session_id,
                )
        with self._check_slot_lock:
            self._erase_slot_if_available(self.check_slot_id)
        return {
            "status": "ok",
            "session_id": session_id,
            "closed": session is not None,
            "message_count": 0,
            "reopened_default_session": reopened is not None,
            "default_check_session_id": None if reopened is None else reopened.session_id,
        }

    def _parse_actions(self, output: str) -> list[int]:
        action_patterns = "|".join(re.escape(action) for action in self.actions2idx)
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        return list(value for values in actions for value in values)

    @staticmethod
    def _pil_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_prompt_parts(self, prompt_instruction: str) -> list[str]:
        prompt = self.conjunctions[0] + DEFAULT_IMAGE_TOKEN
        prompt_instruction += f" {prompt}."
        return split_and_clean(prompt_instruction)

    def _build_user_content(self, prompt_instruction: str, images: list[Image.Image], input_img_id: int) -> list[dict[str, Any]]:
        parts = self._build_prompt_parts(prompt_instruction)
        content: list[dict[str, Any]] = []
        for part in parts:
            if part == DEFAULT_IMAGE_TOKEN:
                image = images[input_img_id]
                input_img_id += 1
                content.append({"type": "image_url", "image_url": {"url": self._pil_to_data_url(image)}})
            else:
                content.append({"type": "text", "text": part})
        return content

    def _build_primary_user_turn(
        self,
        *,
        instruction: str,
        history_keys: list[int],
        current_key: int,
    ) -> dict[str, Any]:
        sources = copy.deepcopy(self.conversation)
        sources[0]["value"] = sources[0]["value"].replace("<instruction>.", instruction)
        if history_keys:
            placeholder = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_keys)
            sources[0]["value"] += f" These are your historical observations: {placeholder}."
        self.input_image_keys = history_keys + [current_key]
        self.input_images = [self.rgb_list[i] for i in self.input_image_keys]
        return {"role": "user", "content": self._build_user_content(sources[0]["value"], self.input_images, 0)}

    def _build_look_down_messages(self, image: Image.Image, instruction: str) -> list[dict[str, Any]]:
        if not self.llm_output:
            raise RuntimeError("Last llm_output should not be empty when look down")
        if not self.rgb_list:
            raise RuntimeError("Look-down retry requires at least one previous observation")
        reference_cap = max(1, self.max_look_down_images - 1)
        reference_keys = list(range(max(0, len(self.rgb_list) - reference_cap), len(self.rgb_list)))
        current_key = reference_keys[-1]
        history_keys = reference_keys[:-1]
        self.conversation_history = [self._build_primary_user_turn(instruction=instruction, history_keys=history_keys, current_key=current_key)]
        self.conversation_history.append({"role": "assistant", "content": [{"type": "text", "text": self.llm_output}]})
        self.input_images.append(image)
        self.input_image_keys.append(None)
        self.conversation_history.append({"role": "user", "content": self._build_user_content("", [image], 0)})
        if len(reference_keys) < len(self.rgb_list):
            print(
                "[INFO] llama.cpp look-down context trim: "
                f"kept_recent_images={len(reference_keys)} total_history={len(self.rgb_list)} "
                f"ctx_size={self.llama_ctx_size}"
            )
        return copy.deepcopy(self.conversation_history)

    def _build_messages(self, rgb: np.ndarray, instruction: str, *, look_down: bool) -> list[dict[str, Any]]:
        image = Image.fromarray(rgb).convert("RGB")
        if not look_down:
            image = image.resize((self.resize_w, self.resize_h))
            self.rgb_list.append(image)
            self.depth_list.append(np.zeros((1, 1), dtype=np.float32))
            self.pose_list.append(IDENTITY_4X4.copy())
            self.conversation_history = []
            if self.episode_idx == 0:
                history_id: list[int] = []
            else:
                history_id = np.unique(
                    np.linspace(0, self.episode_idx - 1, self.num_history, dtype=np.int32)
                ).tolist()
            self.episode_idx += 1
            self.conversation_history.append(
                self._build_primary_user_turn(
                    instruction=instruction,
                    history_keys=history_id,
                    current_key=len(self.rgb_list) - 1,
                )
            )
            return copy.deepcopy(self.conversation_history)

        return self._build_look_down_messages(image=image, instruction=instruction)

    def _history_images_from_array(self, rgb_history: np.ndarray) -> list[Image.Image]:
        history = np.asarray(rgb_history)
        if history.size == 0:
            return []
        if history.ndim != 4 or history.shape[-1] != 3:
            raise ValueError(f"history_npz rgb_history must have shape [T,H,W,3], got {history.shape}.")
        return [
            Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB").resize((self.resize_w, self.resize_h))
            for frame in history
        ]

    def _set_external_history(self, rgb_history: np.ndarray | None) -> None:
        if rgb_history is None:
            return
        history_images = self._history_images_from_array(rgb_history)
        self.rgb_list = history_images
        self.depth_list = [np.zeros((1, 1), dtype=np.float32) for _ in history_images]
        self.pose_list = [IDENTITY_4X4.copy() for _ in history_images]
        self.episode_idx = len(history_images)
        self.conversation_history = []
        self.input_images = []
        self.input_image_keys = []

    def step_no_infer(self, rgb: np.ndarray, depth: np.ndarray, pose) -> None:
        del depth
        del pose
        image = Image.fromarray(rgb).convert("RGB").resize((self.resize_w, self.resize_h))
        self.rgb_list.append(image)
        self.depth_list.append(np.zeros((1, 1), dtype=np.float32))
        self.pose_list.append(IDENTITY_4X4.copy())
        self.episode_idx += 1

    def _record_decision(self, *, llm_output: str, output_action: list[int] | None, output_pixel: list[int] | None) -> None:
        self.last_raw_output_text = str(llm_output)
        if output_pixel is not None:
            self.last_decision_mode = "pixel_goal"
            self.pixel_goal_count += 1
            return
        if output_action:
            self.last_decision_mode = ACTION_ID_TO_MODE.get(int(output_action[0]), "wait")
            self.action_only_count += 1
            return
        self.last_decision_mode = "wait"
        self.action_only_count += 1

    def _run_completion(self, messages: list[dict[str, Any]]) -> str:
        start = time.monotonic()
        response = self._sidecar.chat_completion(
            messages=messages,
            max_tokens=self.max_new_tokens,
            slot_id=self.nav_slot_id,
            cache_prompt=True,
        )
        elapsed = time.monotonic() - start
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(f"llama.cpp response does not include choices: {response}")
        first_choice = choices[0]
        if isinstance(first_choice, dict) and "message" in first_choice:
            message = first_choice.get("message", {})
            llm_output = _extract_text_content(message.get("content"))
        else:
            llm_output = _extract_text_content(first_choice.get("text") if isinstance(first_choice, dict) else "")
        self.llm_output = llm_output
        print(f"output {self.episode_idx}  {self.llm_output} cost: {elapsed}s")
        return llm_output

    def step_s2(self, rgb: np.ndarray, depth: np.ndarray, pose, instruction: str, intrinsic, look_down: bool = False) -> DualSystemOutput:
        del depth
        del pose
        del intrinsic
        messages = self._build_messages(rgb, instruction, look_down=look_down)
        llm_output = self._run_completion(messages)
        coord = [int(value) for value in re.findall(r"\d+", llm_output)]
        if len(coord) >= 2:
            output_pixel = [int(coord[1]), int(coord[0])]
            self._record_decision(llm_output=llm_output, output_action=None, output_pixel=output_pixel)
            return DualSystemOutput(output_pixel=output_pixel)
        action_seq = self._parse_actions(llm_output)
        if action_seq:
            self._record_decision(llm_output=llm_output, output_action=action_seq, output_pixel=None)
            return DualSystemOutput(output_action=action_seq)
        fallback_action = [5]
        self._record_decision(llm_output=llm_output, output_action=fallback_action, output_pixel=None)
        return DualSystemOutput(output_action=fallback_action)

    def step(self, rgb: np.ndarray, depth: np.ndarray, pose, instruction: str, intrinsic, look_down: bool = False) -> DualSystemOutput:
        no_output_flag = self.output_action is None and self.output_pixel is None
        if (self.episode_idx - self.last_s2_idx > self.plan_step_gap) or look_down or no_output_flag:
            dual_sys_output = self.step_s2(
                rgb,
                depth,
                pose,
                instruction,
                intrinsic,
                look_down=look_down,
            )
            self.output_action = None if dual_sys_output.output_action is None else list(dual_sys_output.output_action)
            self.output_pixel = None if dual_sys_output.output_pixel is None else list(dual_sys_output.output_pixel)
            self.last_s2_idx = self.episode_idx
            if self.output_pixel is not None:
                self.pixel_goal_rgb = np.asarray(rgb).copy()
                self.pixel_goal_depth = np.asarray(depth).copy()
            else:
                self.pixel_goal_rgb = None
                self.pixel_goal_depth = None
        else:
            self.step_no_infer(rgb, depth, pose)

        if self.output_action is not None:
            output_action = list(self.output_action)
            self.output_action = None
            self.output_pixel = None
            self.pixel_goal_rgb = None
            self.pixel_goal_depth = None
            return DualSystemOutput(output_action=output_action)
        if self.output_pixel is not None:
            return DualSystemOutput(output_pixel=list(self.output_pixel))
        return DualSystemOutput(output_action=[5])

    def eval_dual(
        self,
        image_file,
        depth_file,
        payload: dict[str, Any],
        *,
        history_rgb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        instruction = str(payload.get("instruction", "")).strip()
        language = str(payload.get("language", "auto")).strip() or "auto"
        reset = bool(payload.get("reset", False))
        idx = int(payload.get("idx", 0))
        if not instruction:
            raise ValueError("instruction is required.")

        rgb = InternVlaNavServerRuntime._decode_rgb(image_file)
        depth = InternVlaNavServerRuntime._decode_depth(depth_file)
        camera_pose = IDENTITY_4X4.copy()
        intrinsic = IDENTITY_4X4.copy()

        with self._lock:
            if reset:
                self.reset()
            self._set_external_history(history_rgb)
            self._latest_check_image = Image.fromarray(rgb).convert("RGB").resize((self.resize_w, self.resize_h))
            dual_sys_output = self.step(
                rgb,
                depth,
                camera_pose,
                instruction,
                intrinsic=intrinsic,
                look_down=False,
            )
            if getattr(dual_sys_output, "output_action", None) is not None and list(dual_sys_output.output_action) == [5]:
                dual_sys_output = self.step(
                    rgb,
                    depth,
                    camera_pose,
                    instruction,
                    intrinsic=intrinsic,
                    look_down=True,
                )

        response = InternVlaNavServerRuntime._normalize_output(dual_sys_output)
        print(
            f"[INFO] {DEFAULT_NAVIGATION_ROUTE}: "
            f"idx={idx} reset={reset} language={language} "
            f"response_keys={sorted(response.keys())}"
        )
        return response


LlamaCppSystem2Runtime = LlamaCppDualVLNRuntime


class InternVlaNavServerRuntime:
    def __init__(self, args: Namespace):
        self._args = args
        self._backend = _normalize_backend(getattr(args, "backend", DEFAULT_BACKEND))
        self._runtime = self._load_llama_runtime(args)

    def _load_llama_runtime(self, args: Namespace):
        return LlamaCppDualVLNRuntime(args)

    def close(self) -> None:
        self._runtime.close()

    def upstream_healthy(self) -> bool:
        return self._runtime.upstream_healthy()

    def health_payload(self) -> dict[str, Any]:
        return self._runtime.health_payload()

    @staticmethod
    def _decode_rgb(image_file) -> np.ndarray:
        stream = getattr(image_file, "stream", None)
        if stream is None:
            stream = getattr(image_file, "file", None)
        if stream is None:
            stream = image_file
        image = Image.open(stream).convert("RGB")
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _decode_depth(depth_file) -> np.ndarray:
        stream = getattr(depth_file, "stream", None)
        if stream is None:
            stream = getattr(depth_file, "file", None)
        if stream is None:
            stream = depth_file
        depth = Image.open(stream).convert("I")
        return np.asarray(depth, dtype=np.float32) / 10000.0

    @staticmethod
    def _decode_history(history_file) -> np.ndarray | None:
        if history_file is None:
            return None
        stream = getattr(history_file, "stream", None)
        if stream is None:
            stream = getattr(history_file, "file", None)
        if stream is None:
            stream = history_file
        payload = stream.read()
        return decode_rgb_history_npz(payload)

    @staticmethod
    def _normalize_output(dual_sys_output) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        output_action = getattr(dual_sys_output, "output_action", None)
        output_trajectory = getattr(dual_sys_output, "output_trajectory", None)
        output_pixel = getattr(dual_sys_output, "output_pixel", None)
        if output_action is not None:
            payload["discrete_action"] = [int(value) for value in list(output_action)]
            return payload
        if output_trajectory is not None:
            payload["trajectory"] = np.asarray(output_trajectory).tolist()
        if output_pixel is not None:
            pixel = np.asarray(output_pixel, dtype=np.int32).reshape(2)
            payload["pixel_goal"] = [int(pixel[0]), int(pixel[1])]
        if not payload:
            payload["discrete_action"] = [5]
        return payload

    def eval_dual(self, image_file, depth_file, payload: dict[str, Any], history_file=None) -> dict[str, Any]:
        history_rgb = self._decode_history(history_file)
        return self._runtime.eval_dual(image_file, depth_file, payload, history_rgb=history_rgb)

    def open_check_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self._runtime, "open_check_session"):
            raise RuntimeError("Check sessions are only supported with backend=llama_cpp.")
        return self._runtime.open_check_session(payload)

    def check_session_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self._runtime, "check_session_message"):
            raise RuntimeError("Check sessions are only supported with backend=llama_cpp.")
        return self._runtime.check_session_message(payload)

    def close_check_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not hasattr(self._runtime, "close_check_session"):
            raise RuntimeError("Check sessions are only supported with backend=llama_cpp.")
        return self._runtime.close_check_session(payload)


def create_app(runtime: InternVlaNavServerRuntime) -> Flask:
    if Flask is None:
        raise RuntimeError("Flask is unavailable in the selected Python environment.")
    app = Flask(__name__)

    def _json_request_payload() -> dict[str, Any]:
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValueError("Request body must be a JSON object.")
        if not isinstance(payload, dict):
            raise ValueError("Request body must decode to an object.")
        return payload

    @app.route("/healthz", methods=["GET"])
    def healthz():
        payload = runtime.health_payload()
        if bool(payload.get("ready", False)):
            return jsonify(payload), 200
        error_payload = dict(payload)
        error_payload["status"] = "error"
        error_payload["ready"] = False
        return jsonify(error_payload), 503

    @app.route(DEFAULT_NAVIGATION_ROUTE, methods=["POST"])
    @app.route(LEGACY_EVAL_DUAL_ROUTE, methods=["POST"])
    def navigation():
        image_file = request.files.get("image")
        depth_file = request.files.get("depth")
        history_file = request.files.get("history_npz")
        raw_json = request.form.get("json", "")
        if image_file is None:
            return jsonify({"status": "error", "message": "Missing multipart image file."}), 400
        if depth_file is None:
            return jsonify({"status": "error", "message": "Missing multipart depth file."}), 400
        try:
            payload = json.loads(raw_json) if raw_json else {}
        except json.JSONDecodeError as exc:
            return jsonify({"status": "error", "message": f"Invalid json form field: {exc}"}), 400
        if not isinstance(payload, dict):
            return jsonify({"status": "error", "message": "json form field must decode to an object."}), 400
        try:
            return jsonify(runtime.eval_dual(image_file, depth_file, payload, history_file=history_file))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    @app.route("/check/session/open", methods=["POST"])
    def check_session_open():
        try:
            return jsonify(runtime.open_check_session(_json_request_payload()))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    @app.route("/check/session/message", methods=["POST"])
    def check_session_message():
        try:
            return jsonify(runtime.check_session_message(_json_request_payload()))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    @app.route("/check/session/close", methods=["POST"])
    def check_session_close():
        try:
            return jsonify(runtime.close_check_session(_json_request_payload()))
        except Exception as exc:  # noqa: BLE001
            return jsonify({"status": "error", "message": f"{type(exc).__name__}: {exc}"}), 400

    return app


def _send_stdlib_json(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_stdlib_json_request(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_type = handler.headers.get("Content-Type", "")
    if "application/json" not in content_type:
        raise ValueError("Content-Type must be application/json.")
    content_length = int(handler.headers.get("Content-Length", "0") or 0)
    raw_body = handler.rfile.read(content_length) if content_length > 0 else b"{}"
    payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
    if not isinstance(payload, dict):
        raise ValueError("Request body must decode to an object.")
    return payload


def _create_stdlib_handler(runtime: InternVlaNavServerRuntime):
    class InternVlaStdlibHandler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:  # noqa: A003
            print(f"[HTTP] {self.address_string()} - {format % args}")

        def do_GET(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path != "/healthz":
                _send_stdlib_json(self, {"status": "error", "message": "Not found."}, 404)
                return
            payload = runtime.health_payload()
            if bool(payload.get("ready", False)):
                _send_stdlib_json(self, payload, 200)
                return
            error_payload = dict(payload)
            error_payload["status"] = "error"
            error_payload["ready"] = False
            _send_stdlib_json(self, error_payload, 503)

        def do_POST(self) -> None:  # noqa: N802
            path = urlsplit(self.path).path
            if path == "/check/session/open":
                try:
                    response = runtime.open_check_session(_read_stdlib_json_request(self))
                    _send_stdlib_json(self, response, 200)
                except json.JSONDecodeError as exc:
                    _send_stdlib_json(self, {"status": "error", "message": f"Invalid JSON body: {exc}"}, 400)
                except Exception as exc:  # noqa: BLE001
                    _send_stdlib_json(self, {"status": "error", "message": f"{type(exc).__name__}: {exc}"}, 400)
                return
            if path == "/check/session/message":
                try:
                    response = runtime.check_session_message(_read_stdlib_json_request(self))
                    _send_stdlib_json(self, response, 200)
                except json.JSONDecodeError as exc:
                    _send_stdlib_json(self, {"status": "error", "message": f"Invalid JSON body: {exc}"}, 400)
                except Exception as exc:  # noqa: BLE001
                    _send_stdlib_json(self, {"status": "error", "message": f"{type(exc).__name__}: {exc}"}, 400)
                return
            if path == "/check/session/close":
                try:
                    response = runtime.close_check_session(_read_stdlib_json_request(self))
                    _send_stdlib_json(self, response, 200)
                except json.JSONDecodeError as exc:
                    _send_stdlib_json(self, {"status": "error", "message": f"Invalid JSON body: {exc}"}, 400)
                except Exception as exc:  # noqa: BLE001
                    _send_stdlib_json(self, {"status": "error", "message": f"{type(exc).__name__}: {exc}"}, 400)
                return
            if path not in {DEFAULT_NAVIGATION_ROUTE, LEGACY_EVAL_DUAL_ROUTE}:
                _send_stdlib_json(self, {"status": "error", "message": "Not found."}, 404)
                return

            content_type = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in content_type:
                _send_stdlib_json(
                    self,
                    {"status": "error", "message": "Content-Type must be multipart/form-data."},
                    400,
                )
                return

            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": content_type,
                        "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
                    },
                    keep_blank_values=True,
                )
                image_file = form["image"] if "image" in form else None
                depth_file = form["depth"] if "depth" in form else None
                history_file = form["history_npz"] if "history_npz" in form else None
                raw_json = form.getvalue("json", "")
                if image_file is None:
                    _send_stdlib_json(self, {"status": "error", "message": "Missing multipart image file."}, 400)
                    return
                if depth_file is None:
                    _send_stdlib_json(self, {"status": "error", "message": "Missing multipart depth file."}, 400)
                    return

                payload = json.loads(raw_json) if raw_json else {}
                if not isinstance(payload, dict):
                    _send_stdlib_json(
                        self,
                        {"status": "error", "message": "json form field must decode to an object."},
                        400,
                    )
                    return
                response = runtime.eval_dual(image_file, depth_file, payload, history_file=history_file)
                _send_stdlib_json(self, response, 200)
            except json.JSONDecodeError as exc:
                _send_stdlib_json(self, {"status": "error", "message": f"Invalid json form field: {exc}"}, 400)
            except Exception as exc:  # noqa: BLE001
                _send_stdlib_json(self, {"status": "error", "message": f"{type(exc).__name__}: {exc}"}, 400)

    return InternVlaStdlibHandler


def run_http_server(runtime: InternVlaNavServerRuntime, host: str, port: int) -> None:
    if Flask is not None:
        app = create_app(runtime)
        app.run(host=host, port=port, threaded=True)
        return

    server = ThreadingHTTPServer((host, port), _create_stdlib_handler(runtime))
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    args = prepare_runtime_args(parse_args(argv))
    runtime = InternVlaNavServerRuntime(args)
    atexit.register(runtime.close)

    print("[INFO] InternVLA System2 server ready")
    print(f"[INFO] Host          : {args.host}")
    print(f"[INFO] Port          : {args.port}")
    print(f"[INFO] Backend       : {args.backend}")
    print(f"[INFO] Llama root    : {args.llama_cpp_root}")
    print(f"[INFO] Llama URL     : {args.llama_url}")
    print(f"[INFO] Llama model   : {args.llama_model_path}")
    print(f"[INFO] Llama mmproj  : {args.llama_mmproj_path}")
    print(f"[INFO] Check prompt  : {args.check_session_system_prompt}")
    print(f"[INFO] Check LoRA    : {args.check_lora_adapter_path or 'disabled'}")
    print(f"[INFO] Check LoRA scale      : {args.check_lora_scale:.3f}")
    print(f"[INFO] Default check session     : {args.default_check_session_id}")
    print(f"[INFO] Default check auto-open   : {args.default_check_session_auto_open}")
    print(f"[INFO] Llama ctx     : {args.llama_ctx_size}")
    print(f"[INFO] Llama slots   : {args.llama_parallel_slots}")
    print(f"[INFO] Nav slot      : {args.llama_nav_slot}")
    print(f"[INFO] Check slot    : {args.llama_check_slot}")
    print(f"[INFO] Llama KV K    : {args.llama_cache_type_k or 'disabled'}")
    print(f"[INFO] Llama KV V    : {args.llama_cache_type_v or 'disabled'}")
    print(f"[INFO] Resize        : {args.resize_w}x{args.resize_h}")
    print(f"[INFO] History       : {args.num_history}")
    print(f"[INFO] Plan step gap : {args.plan_step_gap}")
    print(f"[INFO] Max new tokens: {args.max_new_tokens}")
    print(f"[INFO] HTTP backend  : {'flask' if Flask is not None else 'stdlib'}")
    try:
        run_http_server(runtime, str(args.host), int(args.port))
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
