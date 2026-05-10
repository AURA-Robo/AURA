from __future__ import annotations

from pathlib import Path

from runtime import service as runtime_service


def _fake_launcher_json(script_path: Path, env):  # noqa: ANN001
    del env
    script_name = Path(script_path).name
    if script_name == "inference_system_windows.bat":
        return {"inference_system_url": "http://127.0.0.1:15880"}
    if script_name == "navigation_system_windows.bat":
        return {"navigation_system_url": "http://127.0.0.1:17882"}
    if script_name == "reasoning_system_windows.bat":
        return {"reasoning_system_url": "http://127.0.0.1:17881"}
    if script_name == "control_runtime_windows.bat":
        return {"runtime_control_api_url": "http://127.0.0.1:8892"}
    raise AssertionError(f"unexpected launcher: {script_name}")


def test_build_default_launchers_resolves_urls_without_config_subprocess(monkeypatch, tmp_path: Path) -> None:
    def fail_launcher_json(script_path: Path, env):  # noqa: ANN001
        del env
        raise AssertionError(f"unexpected config subprocess: {Path(script_path).name}")

    monkeypatch.setattr(runtime_service, "_launcher_json", fail_launcher_json)

    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "headless", "viewerEnabled": True},
        {
            "INFERENCE_SYSTEM_HOST": "127.0.0.5",
            "INFERENCE_SYSTEM_PORT": "25880",
            "NAVIGATION_SYSTEM_HOST": "127.0.0.2",
            "NAVIGATION_SYSTEM_PORT": "27882",
            "REASONING_SYSTEM_HOST": "127.0.0.3",
            "REASONING_SYSTEM_PORT": "27881",
            "RUNTIME_CONTROL_API_HOST": "127.0.0.4",
            "RUNTIME_CONTROL_API_PORT": "9892",
        },
    )

    launchers_by_name = {launcher.name: launcher for launcher in launchers}
    assert launchers_by_name["inference_system"].health_url == "http://127.0.0.5:25880/healthz"
    assert launchers_by_name["navigation_system"].health_url == "http://127.0.0.2:27882/healthz"
    assert launchers_by_name["reasoning_system"].health_url == "http://127.0.0.3:27881/healthz"
    assert launchers_by_name["control_runtime"].health_url == "http://127.0.0.4:9892/healthz"
    assert launchers_by_name["reasoning_system"].env["NAVIGATION_SYSTEM_URL"] == "http://127.0.0.2:27882"
    assert launchers_by_name["control_runtime"].env["NAVIGATION_URL"] == "http://127.0.0.2:27882"


def test_build_default_launchers_starts_inference_stack_for_model_backends(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_service, "_launcher_json", _fake_launcher_json)

    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "headless", "viewerEnabled": True},
        {},
    )

    launchers_by_name = {launcher.name: launcher for launcher in launchers}
    assert list(launchers_by_name) == [
        "inference_system",
        "navigation_system",
        "reasoning_system",
        "control_runtime",
    ]

    inference_env = launchers_by_name["inference_system"].env
    assert launchers_by_name["inference_system"].health_url == "http://127.0.0.1:15880/healthz"
    assert inference_env["PLANNER_MODEL_HOST"] == "127.0.0.1"
    assert inference_env["PLANNER_MODEL_PORT"] == "8093"
    assert inference_env["DIALOGUE_MODEL_HOST"] == "127.0.0.1"
    assert inference_env["DIALOGUE_MODEL_PORT"] == "8094"
    assert inference_env["DIALOGUE_ALLOW_PROMPT_ONLY"] == "1"
    assert launchers_by_name["navigation_system"].env["NAVIGATION_BACKEND_AUTOSTART"] == "0"
    assert launchers_by_name["reasoning_system"].env["DIALOGUE_MODEL_BASE_URL"] == (
        "http://127.0.0.1:8094/v1/chat/completions"
    )


def test_build_default_launchers_keeps_dialogue_lora_strict_when_configured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_service, "_launcher_json", _fake_launcher_json)

    lora_path = r"C:\models\qwen-chat-lora.gguf"
    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "headless"},
        {"DIALOGUE_LORA_ADAPTER_PATH": lora_path},
    )

    inference_env = next(launcher.env for launcher in launchers if launcher.name == "inference_system")
    assert inference_env["DIALOGUE_LORA_ADAPTER_PATH"] == lora_path
    assert inference_env["DIALOGUE_ALLOW_PROMPT_ONLY"] == "0"


def test_build_default_launchers_preserves_explicit_dialogue_prompt_only_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(runtime_service, "_launcher_json", _fake_launcher_json)

    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "headless"},
        {"DIALOGUE_ALLOW_PROMPT_ONLY": "0"},
    )

    inference_env = next(launcher.env for launcher in launchers if launcher.name == "inference_system")
    assert inference_env["DIALOGUE_ALLOW_PROMPT_ONLY"] == "0"


def test_build_default_launchers_uses_longer_timeout_for_gui_mode(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_service, "_launcher_json", _fake_launcher_json)

    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "gui", "viewerEnabled": True},
        {},
    )

    control_launcher = next(launcher for launcher in launchers if launcher.name == "control_runtime")
    assert control_launcher.start_timeout_s == 180.0
    assert control_launcher.env["AURA_LAUNCH_MODE"] == "gui"


def test_build_default_launchers_keeps_headless_timeout_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_service, "_launcher_json", _fake_launcher_json)

    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "headless", "viewerEnabled": False},
        {},
    )

    control_launcher = next(launcher for launcher in launchers if launcher.name == "control_runtime")
    assert control_launcher.start_timeout_s == 60.0
    assert control_launcher.env["AURA_LAUNCH_MODE"] == "headless"


def test_build_default_launchers_respects_control_runtime_timeout_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_service, "_launcher_json", _fake_launcher_json)

    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "gui"},
        {"AURA_CONTROL_RUNTIME_START_TIMEOUT_SECONDS": "240"},
    )

    control_launcher = next(launcher for launcher in launchers if launcher.name == "control_runtime")
    assert control_launcher.start_timeout_s == 240.0


def test_build_default_launchers_preserves_system_python_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runtime_service, "_launcher_json", _fake_launcher_json)

    python_path = r"C:\repo\.venv\Scripts\python.exe"
    launchers = runtime_service.build_default_launchers(
        tmp_path,
        {"launchMode": "headless"},
        {
            "AURA_PYTHON": python_path,
            "VIRTUAL_ENV": r"C:\repo\.venv",
            "CONDA_PREFIX": r"C:\Users\mango\anaconda3",
        },
    )

    launcher_envs = {launcher.name: launcher.env for launcher in launchers}
    assert launcher_envs["inference_system"]["AURA_PYTHON"] == python_path
    assert launcher_envs["navigation_system"]["AURA_PYTHON"] == python_path
    assert launcher_envs["reasoning_system"]["AURA_PYTHON"] == python_path
    assert "AURA_PYTHON" not in launcher_envs["control_runtime"]
    assert "VIRTUAL_ENV" not in launcher_envs["control_runtime"]
    assert "CONDA_PREFIX" not in launcher_envs["control_runtime"]
