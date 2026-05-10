from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from systems.inference.dialogue import server as dialogue_server


def _args(
    tmp_path: Path,
    *,
    lora_adapter_path: str | None = None,
    allow_prompt_only: bool = False,
) -> Namespace:
    llama_server = tmp_path / "llama-server.exe"
    model = tmp_path / "dialogue.gguf"
    lora = tmp_path / "chat-lora.gguf"
    llama_server.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    lora.write_text("", encoding="utf-8")
    return Namespace(
        host="127.0.0.1",
        port=8094,
        model=str(model),
        llama_server=str(llama_server),
        gpu_layers=999,
        ctx_size=1024,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
        lora_adapter_path=str(lora) if lora_adapter_path is None else lora_adapter_path,
        allow_prompt_only=allow_prompt_only,
    )


def test_default_dialogue_model_uses_existing_qwen_base() -> None:
    assert dialogue_server.DEFAULT_MODEL.name == "Qwen3-1.7B-Q4_K_M-Instruct.gguf"


def test_build_command_applies_chat_lora_adapter(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dialogue_server, "_llama_server_supports_reasoning_flags", lambda _: False)

    command = dialogue_server.build_command(_args(tmp_path))

    assert "--lora" in command
    assert command[command.index("--lora") + 1].endswith("chat-lora.gguf")


def test_reasoning_flag_detection_requires_exact_option(monkeypatch, tmp_path: Path) -> None:
    class _Completed:
        stdout = "description mentions --reasoning off\n--reasoning-format none\n--reasoning-budget N"
        stderr = ""

    monkeypatch.setattr(dialogue_server.subprocess, "run", lambda *args, **kwargs: _Completed())

    assert dialogue_server._llama_server_supports_reasoning_flags(tmp_path / "llama-server.exe") is False


def test_build_command_disables_qwen_thinking_without_unsupported_toggle(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(dialogue_server, "_llama_server_supports_reasoning_flags", lambda _: False)
    monkeypatch.setattr(dialogue_server, "_llama_server_supports_no_think_flags", lambda _: True)

    command = dialogue_server.build_command(_args(tmp_path, lora_adapter_path="", allow_prompt_only=True))

    assert "--reasoning" not in command
    assert command[command.index("--reasoning-budget") + 1] == "0"
    assert command[command.index("--reasoning-format") + 1] == "none"


def test_main_rejects_missing_lora_adapter(tmp_path: Path) -> None:
    args = _args(tmp_path, lora_adapter_path=str(tmp_path / "missing-lora.gguf"))

    with pytest.raises(SystemExit, match="dialogue LoRA adapter not found"):
        dialogue_server.main(
            [
                "--host",
                args.host,
                "--port",
                str(args.port),
                "--model",
                args.model,
                "--llama-server",
                args.llama_server,
                "--lora-adapter-path",
                args.lora_adapter_path,
            ]
        )


def test_main_allows_prompt_only_without_lora_when_explicit(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path, lora_adapter_path="")
    launched: list[list[str]] = []

    class _Completed:
        returncode = 0

    monkeypatch.setattr(dialogue_server, "_llama_server_supports_reasoning_flags", lambda _: False)
    monkeypatch.setattr(
        dialogue_server.subprocess,
        "run",
        lambda command, check=False: launched.append(list(command)) or _Completed(),
    )

    result = dialogue_server.main(
        [
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--model",
            args.model,
            "--llama-server",
            args.llama_server,
            "--allow-prompt-only",
        ]
    )

    assert result == 0
    assert launched
    assert "--lora" not in launched[0]
