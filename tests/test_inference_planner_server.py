from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from systems.inference.planner import server as planner_server


def _args(tmp_path: Path) -> Namespace:
    llama_server = tmp_path / "llama-server.exe"
    model = tmp_path / "planner.gguf"
    llama_server.write_text("", encoding="utf-8")
    model.write_text("", encoding="utf-8")
    return Namespace(
        host="127.0.0.1",
        port=8093,
        model=str(model),
        llama_server=str(llama_server),
        gpu_layers=999,
        ctx_size=1024,
        parallel_slots=2,
        cache_type_k="q8_0",
        cache_type_v="q8_0",
    )


def test_build_command_omits_reasoning_flags_when_llama_server_does_not_support_them(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(planner_server, "_llama_server_supports_reasoning_flags", lambda _: False)

    command = planner_server.build_command(_args(tmp_path))

    assert "--reasoning" not in command
    assert "--reasoning-budget" not in command
    assert "--reasoning-format" not in command


def test_build_command_includes_reasoning_flags_when_llama_server_supports_them(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(planner_server, "_llama_server_supports_reasoning_flags", lambda _: True)

    command = planner_server.build_command(_args(tmp_path))

    assert "--reasoning" in command
    assert "--reasoning-budget" in command
    assert "--reasoning-format" in command


def test_build_command_includes_parallel_slots_and_total_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(planner_server, "_llama_server_supports_reasoning_flags", lambda _: False)

    command = planner_server.build_command(_args(tmp_path))

    assert "--parallel" in command
    assert command[command.index("--parallel") + 1] == "2"
    assert command[command.index("-c") + 1] == "2048"
