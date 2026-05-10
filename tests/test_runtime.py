from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap
import time

import pytest

from runtime.service import LauncherSpec, RuntimeService


pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="runtime integration tests require Windows")


def _write_service_script(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            """
            import argparse
            from http import HTTPStatus
            from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

            parser = argparse.ArgumentParser()
            parser.add_argument("--port", type=int, required=True)
            args = parser.parse_args()

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if self.path.rstrip("/") == "/healthz":
                        body = b'{"ok": true}'
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    self.send_response(HTTPStatus.NOT_FOUND)
                    self.end_headers()

                def log_message(self, format, *args):
                    del format, args

            server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
            server.serve_forever()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _write_launcher_script(path: Path, *, service_script: Path | None, port: int, order_file: Path) -> None:
    if service_script is None:
        command = f"\"{sys.executable}\" -c \"import sys; sys.exit(1)\""
    else:
        command = f"\"{sys.executable}\" \"{service_script}\" --port {port}"
    path.write_text(
        "\n".join(
            [
                "@echo off",
                f"echo {path.stem}>> \"{order_file}\"",
                command,
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _fake_launcher_factory(tmp_path: Path, *, fail_name: str | None = None):
    order_file = tmp_path / "start-order.log"
    service_script = tmp_path / "fake_runtime_service.py"
    _write_service_script(service_script)
    names = ["navigation_system", "reasoning_system", "control_runtime"]
    ports = {
        "navigation_system": 18181,
        "reasoning_system": 18182,
        "control_runtime": 18183,
    }
    scripts: dict[str, Path] = {}
    for name in names:
        script_path = tmp_path / f"{name}.cmd"
        _write_launcher_script(
            script_path,
            service_script=None if name == fail_name else service_script,
            port=ports[name],
            order_file=order_file,
        )
        scripts[name] = script_path

    def factory(_repo_root: Path, _session_config: dict[str, object], base_env):
        return [
            LauncherSpec(
                name=name,
                script_path=scripts[name],
                env=dict(base_env),
                health_url=f"http://127.0.0.1:{ports[name]}/healthz",
                endpoints={f"{name}Url": f"http://127.0.0.1:{ports[name]}"},
                start_timeout_s=3.0,
            )
            for name in names
        ]

    return factory, order_file


def _partial_degraded_launcher_factory(tmp_path: Path, *, fail_name: str) -> tuple[object, Path]:
    order_file = tmp_path / "start-order-partial.log"
    service_script = tmp_path / "fake_runtime_service_partial.py"
    _write_service_script(service_script)
    names = ["navigation_system", "reasoning_system", "control_runtime"]
    ports = {
        "navigation_system": 18281,
        "reasoning_system": 18282,
        "control_runtime": 18283,
    }
    scripts: dict[str, Path] = {}
    for name in names:
        script_path = tmp_path / f"{name}.partial.cmd"
        _write_launcher_script(
            script_path,
            service_script=None if name == fail_name else service_script,
            port=ports[name],
            order_file=order_file,
        )
        scripts[name] = script_path

    wait_for_health = {
        "navigation_system": False,
        "reasoning_system": False,
        "control_runtime": True,
    }

    def factory(_repo_root: Path, _session_config: dict[str, object], base_env):
        return [
            LauncherSpec(
                name=name,
                script_path=scripts[name],
                env=dict(base_env),
                health_url=f"http://127.0.0.1:{ports[name]}/healthz",
                endpoints={f"{name}Url": f"http://127.0.0.1:{ports[name]}"},
                wait_for_health=wait_for_health[name],
                start_timeout_s=3.0,
            )
            for name in names
        ]

    return factory, order_file


def test_runtime_uses_start_and_stop_order(tmp_path: Path) -> None:
    class RecordingRegistry:
        def __init__(self):
            self.started: list[str] = []
            self.stop_calls: list[list[str]] = []

        def start(self, spec: LauncherSpec):
            self.started.append(spec.name)
            return None

        def stop_many(self, names: list[str]) -> None:
            self.stop_calls.append(list(names))

        def snapshots(self):
            return []

    names = ["navigation_system", "reasoning_system", "control_runtime"]

    def factory(repo_root: Path, session_config: dict[str, object], base_env):
        del repo_root, session_config, base_env
        return [
            LauncherSpec(
                name=name,
                script_path=tmp_path / f"{name}.cmd",
                env={},
                health_url=f"http://127.0.0.1/{name}/healthz",
                endpoints={},
            )
            for name in names
        ]

    service = RuntimeService(tmp_path, launcher_factory=factory)
    service._registry = RecordingRegistry()
    service._wait_for_health = lambda health_url, timeout_s: None

    payload = service.start_session({"launchMode": "headless"})
    assert payload["ok"] is True
    assert service._registry.started == names

    service.stop_session()
    assert service._registry.stop_calls == [
        ["inference_system", "navigation_system", "reasoning_system", "control_runtime"]
    ]


def test_runtime_rolls_back_started_processes_on_partial_failure(tmp_path: Path) -> None:
    factory, order_file = _fake_launcher_factory(tmp_path, fail_name="reasoning_system")
    service = RuntimeService(tmp_path, launcher_factory=factory, base_env=os.environ.copy())

    payload = service.start_session({"launchMode": "headless"})

    assert payload["ok"] is False
    assert payload["session"]["active"] is False
    assert payload["lastError"] is not None
    started_order = order_file.read_text(encoding="utf-8").splitlines()
    assert started_order[:2] == ["navigation_system", "reasoning_system"]
    assert started_order == ["navigation_system", "reasoning_system"]

    time.sleep(1.0)
    snapshots = {item["name"]: item for item in service.state_payload(ok=False)["processes"]}
    assert snapshots["navigation_system"]["state"] == "exited"


def test_runtime_stop_is_idempotent(tmp_path: Path) -> None:
    factory, _order_file = _fake_launcher_factory(tmp_path)
    service = RuntimeService(tmp_path, launcher_factory=factory, base_env=os.environ.copy())

    payload = service.start_session({"launchMode": "headless"})
    assert payload["ok"] is True
    assert payload["session"]["active"] is True

    stopped = service.stop_session()
    assert stopped["session"]["active"] is False
    stopped_again = service.stop_session()
    assert stopped_again["session"]["active"] is False


def test_runtime_keeps_running_when_noncritical_launcher_never_becomes_healthy(tmp_path: Path) -> None:
    factory, order_file = _partial_degraded_launcher_factory(tmp_path, fail_name="reasoning_system")
    service = RuntimeService(tmp_path, launcher_factory=factory, base_env=os.environ.copy())

    payload = service.start_session({"launchMode": "headless", "viewerEnabled": True})

    assert payload["ok"] is True
    assert payload["session"]["active"] is True
    assert payload["serviceEndpoints"]["control_runtimeUrl"] == "http://127.0.0.1:18283"
    started_order: list[str] = []
    deadline = time.time() + 2.0
    while time.time() < deadline:
        started_order = order_file.read_text(encoding="utf-8").splitlines()
        if len(started_order) >= 3:
            break
        time.sleep(0.1)
    assert started_order == [
        "navigation_system.partial",
        "reasoning_system.partial",
        "control_runtime.partial",
    ]

    snapshots: dict[str, dict[str, object]] = {}
    deadline = time.time() + 2.0
    while time.time() < deadline:
        snapshots = {item["name"]: item for item in service.state_payload()["processes"]}
        if snapshots.get("reasoning_system", {}).get("state") == "exited":
            break
        time.sleep(0.1)
    assert snapshots["control_runtime"]["state"] == "running"
    assert snapshots["reasoning_system"]["state"] == "exited"

    stopped = service.stop_session()
    assert stopped["session"]["active"] is False
