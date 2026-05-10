from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher tests require Windows")

ROOT = Path(__file__).resolve().parents[2]


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "AURA_PYTHON",
        "INFERENCE_SYSTEM_HOST",
        "INFERENCE_SYSTEM_PORT",
        "NAVDP_HOST",
        "NAVDP_PORT",
        "SYSTEM2_HOST",
        "SYSTEM2_PORT",
        "PLANNER_MODEL_HOST",
        "PLANNER_MODEL_PORT",
        "REASONING_SYSTEM_HOST",
        "REASONING_SYSTEM_PORT",
        "NAVIGATION_SYSTEM_HOST",
        "NAVIGATION_SYSTEM_PORT",
        "ISAACSIM_PATH",
        "RUNTIME_CONTROL_API_HOST",
        "RUNTIME_CONTROL_API_PORT",
        "NAVIGATION_URL",
        "NAVDP_URL",
        "SYSTEM2_URL",
        "NAVIGATION_SYSTEM_URL",
        "NAVIGATION_NAVDP_FALLBACK",
        "PLANNER_MODEL_BASE_URL",
        "PLANNER_PARALLEL_SLOTS",
        "PLANNER_INTENT_SLOT_ID",
        "PLANNER_TASK_FRAME_SLOT_ID",
        "SYSTEM2_CHECK_LORA_ADAPTER_PATH",
        "SYSTEM2_CHECK_LORA_SCALE",
        "SYSTEM2_CHECK_SESSION_SYSTEM_PROMPT",
        "DIALOGUE_LORA_ADAPTER_PATH",
        "DIALOGUE_ALLOW_PROMPT_ONLY",
        "AURA_LAUNCH_MODE",
        "AURA_VIEWER_ENABLED",
        "AURA_MEMORY_STORE",
        "AURA_DETECTION_ENABLED",
        "AURA_DETECTION_MODEL_PATH",
        "AURA_RUNTIME_URL",
        "AURA_RUNTIME_SUPERVISOR_URL",
        "AURA_INFERENCE_SYSTEM_URL",
        "AURA_REASONING_SYSTEM_URL",
        "AURA_NAVIGATION_SYSTEM_URL",
        "AURA_CONTROL_RUNTIME_URL",
        "AURA_WEBRTC_PROXY_BASE",
        "AURA_WEBRTC_RGB_FPS",
        "AURA_WEBRTC_DEPTH_FPS",
        "AURA_WEBRTC_TELEMETRY_HZ",
        "AURA_WEBRTC_POLL_INTERVAL_MS",
        "AURA_WEBRTC_LATEST_FRAME_DRAIN_BATCHES",
        "AURA_WEBRTC_OBJECT_MEMORY_QUEUE_SIZE",
        "AURA_WEBRTC_ENABLE_DEPTH_TRACK",
        "AURA_OBJECT_MEMORY_DSN",
        "AURA_OBJECT_MEMORY_EVENT_LOG_PATH",
        "AURA_KNOWLEDGE_DSN",
        "AURA_PLANNER_CATALOG_DSN",
        "AURA_OBJECT_MEMORY_AUTO_MIGRATE",
        "AURA_MEMORY_USER_ID",
        "AURA_DASHBOARD_API_BASE_URL",
        "AURA_DASHBOARD_PROXY_TARGET",
        "VITE_AURA_API_BASE",
    ):
        env.pop(key, None)
    return env


def _load_json_from_stdout(stdout: str) -> dict[str, object]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip() != ""]
    assert lines, "launcher stdout was empty"
    return json.loads(lines[-1])


def test_inference_system_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "inference_system_windows.bat"
    env = _base_env()
    env.update(
        {
            "INFERENCE_SYSTEM_PORT": "16880",
            "NAVDP_PORT": "18890",
            "SYSTEM2_PORT": "15813",
            "PLANNER_MODEL_PORT": "8095",
            "PLANNER_PARALLEL_SLOTS": "2",
            "SYSTEM2_CHECK_LORA_ADAPTER_PATH": r"C:\models\tv-check-lora.gguf",
            "SYSTEM2_CHECK_LORA_SCALE": "0.5",
            "SYSTEM2_CHECK_SESSION_SYSTEM_PROMPT": "Answer true only if the TV is clearly off.",
            "DIALOGUE_LORA_ADAPTER_PATH": r"C:\models\qwen-chat-lora.gguf",
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["inference_system_port"] == 16880
    assert payload["navdp_url"] == "http://127.0.0.1:18890"
    assert payload["navdp_tensorrt_mode"] == "auto"
    assert str(payload["navdp_tensorrt_engine_dir"]).endswith(r"artifacts\models\navdp_tensorrt")
    assert payload["navdp_tensorrt_precision"] == "fp16"
    assert payload["system2_url"] == "http://127.0.0.1:15813"
    assert payload["planner_model_url"] == "http://127.0.0.1:8095/v1/chat/completions"
    assert payload["planner_parallel_slots"] == 2
    assert payload["system2_check_lora_adapter_path"] == r"C:\models\tv-check-lora.gguf"
    assert payload["system2_check_lora_configured"] is True
    assert payload["system2_check_lora_scale"] == 0.5
    assert payload["system2_check_session_system_prompt_configured"] is True
    assert payload["dialogue_lora_adapter_path"] == r"C:\models\qwen-chat-lora.gguf"
    assert payload["dialogue_lora_configured"] is True
    assert payload["dialogue_prompt_only"] is False


def test_inference_system_launcher_reports_dialogue_prompt_only_mode() -> None:
    launcher = ROOT / "scripts" / "run_system" / "inference_system_windows.bat"
    env = _base_env()
    env["DIALOGUE_ALLOW_PROMPT_ONLY"] = "1"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["dialogue_lora_adapter_path"] == ""
    assert payload["dialogue_lora_configured"] is False
    assert payload["dialogue_prompt_only"] is True


def test_navigation_system_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "navigation_system_windows.bat"
    env = _base_env()
    env.update(
        {
            "NAVIGATION_SYSTEM_PORT": "17892",
            "SYSTEM2_URL": "http://127.0.0.1:15813",
            "NAVDP_URL": "http://127.0.0.1:18890",
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["navigation_system_port"] == 17892
    assert payload["system2_url"] == "http://127.0.0.1:15813"
    assert payload["navdp_url"] == "http://127.0.0.1:18890"
    assert payload["navdp_fallback"] == "heuristic"
    assert payload["navdp_tensorrt_mode"] == "auto"
    assert str(payload["navdp_tensorrt_engine_dir"]).endswith(r"artifacts\models\navdp_tensorrt")
    assert payload["navdp_tensorrt_precision"] == "fp16"


def test_reasoning_system_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "reasoning_system_windows.bat"
    env = _base_env()
    env.update(
        {
            "REASONING_SYSTEM_PORT": "17891",
            "NAVIGATION_SYSTEM_URL": "http://127.0.0.1:17892",
            "PLANNER_MODEL_BASE_URL": "http://127.0.0.1:8095/v1/chat/completions",
            "PLANNER_INTENT_SLOT_ID": "0",
            "PLANNER_TASK_FRAME_SLOT_ID": "1",
            "DIALOGUE_MODEL_BASE_URL": "http://127.0.0.1:8096/v1/chat/completions",
            "AURA_OBJECT_MEMORY_DSN": "postgresql://example/object_memory",
            "AURA_OBJECT_MEMORY_AUTO_MIGRATE": "1",
            "AURA_MEMORY_USER_ID": "operator-a",
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["reasoning_system_port"] == 17891
    assert payload["navigation_system_url"] == "http://127.0.0.1:17892"
    assert payload["planner_model_base_url"] == "http://127.0.0.1:8095/v1/chat/completions"
    assert payload["planner_intent_slot_id"] == 0
    assert payload["planner_task_frame_slot_id"] == 1
    assert payload["dialogue_model_base_url"] == "http://127.0.0.1:8096/v1/chat/completions"
    assert payload["object_memory_dsn_configured"] is True
    assert payload["object_memory_auto_migrate"] is True
    assert payload["memory_user_id"] == "operator-a"


def test_control_runtime_launcher_reports_active_contract() -> None:
    launcher = ROOT / "scripts" / "run_system" / "control_runtime_windows.bat"
    env = _base_env()
    env.update(
        {
            "RUNTIME_CONTROL_API_PORT": "8898",
            "NAVIGATION_URL": "http://127.0.0.1:17890",
            "AURA_LAUNCH_MODE": "headless",
            "AURA_VIEWER_ENABLED": "0",
            "AURA_MEMORY_STORE": "1",
            "AURA_DETECTION_ENABLED": "0",
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["runtime_control_api_port"] == 8898
    assert payload["navigation_url"] == "http://127.0.0.1:17890"
    assert payload["launch_mode"] == "headless"
    assert payload["viewer_enabled"] is False
    assert payload["viewer_publish"] is False
    assert payload["memory_store"] is True
    assert payload["detection_enabled"] is False
    assert payload["detection_model_path"].endswith(r"artifacts\models\yoloe-26s-seg-pf.pt")
    assert payload["env_url"] == "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
    assert payload["scene_usd"] is None


def test_control_runtime_launcher_reports_detection_model_override() -> None:
    launcher = ROOT / "scripts" / "run_system" / "control_runtime_windows.bat"
    env = _base_env()
    env.update(
        {
            "AURA_DETECTION_ENABLED": "1",
            "AURA_DETECTION_MODEL_PATH": r"C:\models\custom-yoloe.pt",
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["detection_enabled"] is True
    assert payload["detection_model_path"] == r"C:\models\custom-yoloe.pt"


def test_control_runtime_launcher_maps_interior_scene_preset_to_local_usd() -> None:
    launcher = ROOT / "scripts" / "run_system" / "control_runtime_windows.bat"
    env = _base_env()
    env.update(
        {
            "AURA_SCENE_PRESET": "interior agent kujiale 3",
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["scene_preset"] == "interior agent kujiale 3"
    assert payload["scene_usd"].endswith(r"datasets\InteriorAgent\kujiale_0003\kujiale_0003.usda")


def test_control_runtime_launcher_maps_default_interior_agent_scene_to_local_usd() -> None:
    launcher = ROOT / "scripts" / "run_system" / "control_runtime_windows.bat"
    env = _base_env()
    env.update(
        {
            "AURA_SCENE_PRESET": "interioragent",
        }
    )
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", str(launcher), "-PrintConfigJson"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = _load_json_from_stdout(completed.stdout)
    assert payload["scene_preset"] == "interioragent"
    assert payload["scene_usd"].endswith(r"datasets\InteriorAgent\kujiale_0004\kujiale_0004_navila_sanitized.usda")


def test_canonical_launchers_do_not_reference_removed_helpers() -> None:
    control_text = (ROOT / "scripts" / "run_system" / "control_runtime_windows.bat").read_text(encoding="utf-8", errors="ignore")
    inference_text = (ROOT / "scripts" / "run_system" / "inference_system_windows.bat").read_text(encoding="utf-8", errors="ignore")
    backend_text = (ROOT / "scripts" / "run_system" / "backend_windows.ps1").read_text(encoding="utf-8", errors="ignore")
    runtime_text = (ROOT / "scripts" / "run_system" / "runtime_windows.ps1").read_text(encoding="utf-8", errors="ignore")

    assert "send_internvla_nav_command" not in control_text
    assert "src\\systems\\navigation\\bin\\run_navdp_server_windows.bat" not in control_text
    assert "src\\systems\\inference\\bin\\run_internvla_nav_server_windows.bat" not in control_text
    assert "serve_planner_qwen3_nothink.ps1" not in control_text
    assert "systems.inference.api.serve_inference_system" in inference_text
    assert "backend.api.serve_backend" in backend_text
    assert "dashboard\\python" not in backend_text
    assert "runtime.api.serve_runtime" in runtime_text


def test_windows_launchers_separate_system_venv_from_isaac_python() -> None:
    control_text = (ROOT / "scripts" / "run_system" / "control_runtime_windows.bat").read_text(encoding="utf-8", errors="ignore")
    inference_text = (ROOT / "scripts" / "run_system" / "inference_system_windows.bat").read_text(encoding="utf-8", errors="ignore")
    navigation_text = (ROOT / "scripts" / "run_system" / "navigation_system_windows.bat").read_text(encoding="utf-8", errors="ignore")
    reasoning_text = (ROOT / "scripts" / "run_system" / "reasoning_system_windows.bat").read_text(encoding="utf-8", errors="ignore")
    backend_text = (ROOT / "scripts" / "run_system" / "backend_windows.ps1").read_text(encoding="utf-8", errors="ignore")
    runtime_text = (ROOT / "scripts" / "run_system" / "runtime_windows.ps1").read_text(encoding="utf-8", errors="ignore")

    assert r'%REPO_DIR%\.venv\Scripts\python.exe' in inference_text
    assert r'%REPO_DIR%\.venv\Scripts\python.exe' in navigation_text
    assert r'%REPO_DIR%\.venv\Scripts\python.exe' in reasoning_text
    assert 'call "%AURA_PYTHON%"' in inference_text
    assert 'call "%AURA_PYTHON%"' in navigation_text
    assert 'call "%AURA_PYTHON%"' in reasoning_text
    assert ".venv\\Scripts\\python.exe" in backend_text
    assert ".venv\\Scripts\\python.exe" in runtime_text
    assert 'set "PYTHONPATH=%REPO_DIR%\\src"' in inference_text
    assert 'set "PYTHONPATH=%REPO_DIR%\\src"' in navigation_text
    assert 'set "PYTHONPATH=%REPO_DIR%\\src"' in reasoning_text
    assert '$env:PYTHONPATH = "$repoRoot\\src"' in backend_text
    assert '$env:PYTHONPATH = "$repoRoot\\src"' in runtime_text

    assert 'set "ISAACSIM_PYTHON=%ISAACSIM_PATH%\\python.bat"' in control_text
    assert 'call "%ISAACSIM_PYTHON%"' in control_text
    assert 'call "%AURA_PYTHON%"' not in control_text
    assert 'set "PYTHONPATH=%REPO_DIR%\\src"' in control_text


def test_backend_launcher_uses_canonical_environment_names() -> None:
    launcher_text = (ROOT / "scripts" / "run_system" / "backend_windows.ps1").read_text(encoding="utf-8", errors="ignore")

    assert "AURA_RUNTIME_URL" in launcher_text
    assert "AURA_RUNTIME_SUPERVISOR_URL" in launcher_text
    assert "AURA_INFERENCE_SYSTEM_URL" in launcher_text
    assert "AURA_REASONING_SYSTEM_URL" in launcher_text
    assert "AURA_NAVIGATION_SYSTEM_URL" in launcher_text
    assert "AURA_CONTROL_RUNTIME_URL" in launcher_text
    assert "AURA_WEBRTC_RGB_FPS" in launcher_text
    assert "AURA_WEBRTC_ENABLE_DEPTH_TRACK" in launcher_text
    assert "AURA_WEBRTC_LATEST_FRAME_DRAIN_BATCHES" in launcher_text
    assert "AURA_WEBRTC_OBJECT_MEMORY_QUEUE_SIZE" in launcher_text
    assert "AURA_OBJECT_MEMORY_DSN" in launcher_text
    assert "AURA_OBJECT_MEMORY_EVENT_LOG_PATH" in launcher_text
    assert "AURA_OBJECT_MEMORY_AUTO_MIGRATE" in launcher_text
    assert "AURA_MEMORY_USER_ID" in launcher_text


def test_backend_launcher_exports_selected_system_python_to_children(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo AURA_PYTHON=%AURA_PYTHON%",
                "echo %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "backend_windows.ps1"),
            "-Python",
            str(fake_python),
        ],
        cwd=str(ROOT),
        env=_base_env(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert f"AURA_PYTHON={fake_python}" in completed.stdout


def test_backend_launcher_omits_runtime_flag_when_backend_owns_runtime(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "backend_windows.ps1"),
            "-Python",
            str(fake_python),
        ],
        cwd=str(ROOT),
        env=_base_env(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "--runtime-url" not in completed.stdout


def test_backend_launcher_omits_empty_memory_dsn_flags(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "backend_windows.ps1"),
            "-Python",
            str(fake_python),
        ],
        cwd=str(ROOT),
        env=_base_env(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "--object-memory-dsn" not in completed.stdout
    assert "--knowledge-dsn" not in completed.stdout


def test_backend_launcher_includes_memory_dsn_flags_when_configured(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = _base_env()
    env.update(
        {
            "AURA_OBJECT_MEMORY_DSN": "postgresql://example/object-memory",
            "AURA_KNOWLEDGE_DSN": "postgresql://example/knowledge",
        }
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "backend_windows.ps1"),
            "-Python",
            str(fake_python),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "--object-memory-dsn postgresql://example/object-memory" in completed.stdout
    assert "--knowledge-dsn postgresql://example/knowledge" in completed.stdout


def test_backend_launcher_includes_object_memory_event_log_path_when_configured(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    event_log_path = tmp_path / "object_events.jsonl"
    env = _base_env()
    env["AURA_OBJECT_MEMORY_EVENT_LOG_PATH"] = str(event_log_path)

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "backend_windows.ps1"),
            "-Python",
            str(fake_python),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert f"--object-memory-event-log-path {event_log_path}" in completed.stdout


def test_backend_launcher_includes_object_memory_auto_migrate_flag(tmp_path: Path) -> None:
    fake_python = tmp_path / "fake-python.cmd"
    fake_python.write_text(
        "\n".join(
            [
                "@echo off",
                "echo %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = _base_env()
    env["AURA_OBJECT_MEMORY_AUTO_MIGRATE"] = "1"

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "backend_windows.ps1"),
            "-Python",
            str(fake_python),
        ],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "--object-memory-auto-migrate" in completed.stdout
    assert "--object-memory-no-auto-migrate" not in completed.stdout


def test_system_venv_setup_installs_webrtc_extra_by_default() -> None:
    setup_text = (ROOT / "scripts" / "setup_system_venv_windows.ps1").read_text(encoding="utf-8", errors="ignore")

    assert "[switch]$NoWebRtc" in setup_text
    assert '$resolvedExtras += "webrtc"' in setup_text
    assert "Resolve-InstallExtras -RequestedExtras $Extras -IncludeWebRtc:(-not $NoWebRtc.IsPresent)" in setup_text


def test_dashboard_dev_launcher_emits_progress_before_starting_tauri(tmp_path: Path) -> None:
    dashboard_root = tmp_path / "dashboard"
    dashboard_root.mkdir()
    (dashboard_root / "package.json").write_text(
        json.dumps({"name": "dashboard-test", "private": True}),
        encoding="utf-8",
    )
    src_tauri_dir = dashboard_root / "src-tauri"
    src_tauri_dir.mkdir()
    (src_tauri_dir / "tauri.conf.json").write_text("{}", encoding="utf-8")

    fake_npm = tmp_path / "fake-npm.cmd"
    fake_npm.write_text(
        "\n".join(
            [
                "@echo off",
                "echo PROXY_TARGET=%AURA_DASHBOARD_PROXY_TARGET%",
                "echo FAKE_NPM %*",
                "exit /b 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "dashboard_dev_windows.ps1"),
            "-DashboardRoot",
            str(dashboard_root),
            "-Npm",
            str(fake_npm),
            "-BackendUrl",
            "http://127.0.0.1:1",
            "-BackendWaitSeconds",
            "1",
        ],
        cwd=str(ROOT),
        env=_base_env(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    output = "\n".join([completed.stdout, completed.stderr])
    assert completed.returncode == 0, output
    assert "[dashboard] frontend root:" in output
    assert "[dashboard] backend url: http://127.0.0.1:1" in output
    assert "[dashboard] waiting up to 1 seconds for backend bootstrap" in output
    assert "backend bootstrap did not respond within 1 seconds" in output
    assert "[dashboard] starting Tauri dashboard" in output
    assert "PROXY_TARGET=http://127.0.0.1:1" in output
    assert "FAKE_NPM run tauri:dev" in output


def test_run_dashboard_launcher_reuses_existing_backend(tmp_path: Path) -> None:
    class _BootstrapHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path == "/api/bootstrap":
                body = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def log_message(self, format: str, *args):  # noqa: A003
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _BootstrapHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    backend_url = f"http://127.0.0.1:{server.server_address[1]}"

    fake_dashboard = tmp_path / "fake-dashboard.ps1"
    fake_dashboard.write_text(
        "\n".join(
            [
                "param(",
                "    [string]$BackendUrl = '',",
                "    [string]$DashboardRoot = '',",
                "    [string]$Npm = '',",
                "    [int]$BackendWaitSeconds = 0,",
                "    [switch]$SkipBackendWait",
                ")",
                'Write-Output ("FAKE_DASHBOARD backend={0} wait={1} skip={2}" -f $BackendUrl, $BackendWaitSeconds, $SkipBackendWait.IsPresent)',
                'Write-Output ("PROXY_TARGET={0}" -f $env:AURA_DASHBOARD_PROXY_TARGET)',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "run_dashboard_windows.ps1"),
            "-BackendUrl",
            backend_url,
            "-DashboardScriptPath",
            str(fake_dashboard),
            "-BackendWaitSeconds",
            "3",
        ],
        cwd=str(ROOT),
        env=_base_env(),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )

    server.shutdown()
    server.server_close()
    thread.join(timeout=1.0)

    output = "\n".join([completed.stdout, completed.stderr])
    assert completed.returncode == 0, output
    assert f"[run_dashboard] backend already reachable at {backend_url}; reusing existing backend" in output
    assert f"FAKE_DASHBOARD backend={backend_url} wait=3 skip=False" in output
    assert f"PROXY_TARGET={backend_url}" in output


def test_run_dashboard_launcher_cleans_up_owned_backend_without_pid_variable_collision(tmp_path: Path) -> None:
    fake_backend = tmp_path / "fake-backend.ps1"
    fake_backend.write_text(
        "\n".join(
            [
                "param(",
                "    [string]$BindHost = '',",
                "    [string]$Port = ''",
                ")",
                'Write-Output ("FAKE_BACKEND bind={0} port={1}" -f $BindHost, $Port)',
                "Start-Sleep -Seconds 5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fake_dashboard = tmp_path / "fake-dashboard.ps1"
    fake_dashboard.write_text(
        "\n".join(
            [
                "param(",
                "    [string]$BackendUrl = '',",
                "    [string]$DashboardRoot = '',",
                "    [string]$Npm = '',",
                "    [int]$BackendWaitSeconds = 0,",
                "    [switch]$SkipBackendWait",
                ")",
                'Write-Output ("FAKE_DASHBOARD backend={0} skip={1}" -f $BackendUrl, $SkipBackendWait.IsPresent)',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(ROOT / "scripts" / "run_system" / "run_dashboard_windows.ps1"),
            "-BackendPort",
            "9",
            "-BackendScriptPath",
            str(fake_backend),
            "-DashboardScriptPath",
            str(fake_dashboard),
            "-SkipBackendWait",
        ],
        cwd=str(ROOT),
        env=_base_env(),
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )

    output = "\n".join([completed.stdout, completed.stderr])
    assert completed.returncode == 0, output
    assert "[run_dashboard] backend process started" in output
    assert "[run_dashboard] stopping backend process tree" in output
    assert "Pid variable is read-only" not in output
    assert "Pid 변수는 읽기 전용" not in output
    assert "failed to stop backend process tree" not in output
