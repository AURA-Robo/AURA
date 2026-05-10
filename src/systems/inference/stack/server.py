"""Managed inference-system HTTP server."""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from systems.inference.stack.config import REPO_ROOT, build_managed_services, default_log_dir
from systems.inference.stack.process_registry import ProcessRegistry

TRUE_VALUES = frozenset(("1", "true", "yes", "on"))


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return str(raw_value).strip().lower() in TRUE_VALUES


def _json_get(url: str, *, timeout_s: float) -> tuple[str, dict[str, Any] | None, float | None, str | None]:
    started = time.perf_counter()
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "healthy", payload if isinstance(payload, dict) else {"payload": payload}, latency_ms, None
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "error", None, latency_ms, f"http_{exc.code}: {detail}"
    except URLError as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "unreachable", None, latency_ms, str(exc)
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - started) * 1000.0
        return "error", None, latency_ms, f"{type(exc).__name__}: {exc}"


class InferenceSystemServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._services = build_managed_services(args)
        self._registry = ProcessRegistry(Path(args.log_dir))
        self._server = ThreadingHTTPServer((str(args.host), int(args.port)), self._build_handler())
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, name="inference-stack-api", daemon=True)
        atexit.register(self.shutdown)

    def start(self) -> None:
        if self.args.autostart:
            for service in self._services:
                self._registry.start(service)
        self._thread.start()
        print(f"[INFO] Inference system API listening on http://{self.args.host}:{self.args.port}")

    def shutdown(self) -> None:
        try:
            self._registry.stop_all()
        finally:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass

    def _service_snapshot(self, service_name: str | None = None) -> dict[str, Any]:
        selected_services = [
            service for service in self._services if service_name is None or service.name == service_name
        ]
        services: dict[str, Any] = {}
        if not selected_services:
            return services

        def probe(service):
            status, payload, latency_ms, error = _json_get(service.health_url, timeout_s=float(self.args.health_timeout))
            return service.name, {
                "name": service.name,
                "base_url": service.base_url,
                "health_url": service.health_url,
                "status": status,
                "latency_ms": latency_ms,
                "health": payload,
                "error": error,
            }

        max_workers = min(len(selected_services), 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for name, payload in executor.map(probe, selected_services):
                services[name] = payload
        return services

    def models_state(self) -> dict[str, Any]:
        services = self._service_snapshot()
        processes = self._registry.snapshot()
        required_service_names = {service.name for service in self._services if bool(service.required)}
        healthy = all(
            services.get(name, {}).get("status") == "healthy"
            for name in required_service_names
        )
        return {
            "ok": healthy,
            "timestamp": time.time(),
            "host": self.args.host,
            "port": self.args.port,
            "log_dir": str(self.args.log_dir),
            "models": services,
            "services": services,
            "processes": processes,
        }

    def _build_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, Any]):
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(int(status_code))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.end_headers()
                self.wfile.write(body)

            def do_OPTIONS(self):
                self._send_json(HTTPStatus.NO_CONTENT, {})

            def do_GET(self):
                path = self.path.rstrip("/") or "/"
                if path == "/healthz":
                    state = server.models_state()
                    status = HTTPStatus.OK if bool(state["ok"]) else HTTPStatus.SERVICE_UNAVAILABLE
                    self._send_json(
                        status,
                        {
                            "ok": state["ok"],
                            "service": "inference_system",
                            "models": state["models"],
                        },
                    )
                    return
                if path in {"/models/state", "/stack/state"}:
                    self._send_json(HTTPStatus.OK, server.models_state())
                    return
                if path.startswith("/services/") and path.endswith("/health"):
                    name = path.split("/")[2]
                    services = server._service_snapshot(name)
                    if name not in services:
                        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown_service"})
                        return
                    service = services[name]
                    status = HTTPStatus.OK if service["status"] == "healthy" else HTTPStatus.SERVICE_UNAVAILABLE
                    self._send_json(status, {"ok": service["status"] == "healthy", **service})
                    return
                if path.startswith("/models/") and path.endswith("/health"):
                    name = path.split("/")[2]
                    services = server._service_snapshot(name)
                    if name not in services:
                        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "unknown_model"})
                        return
                    service = services[name]
                    status = HTTPStatus.OK if service["status"] == "healthy" else HTTPStatus.SERVICE_UNAVAILABLE
                    self._send_json(status, {"ok": service["status"] == "healthy", **service})
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def do_POST(self):
                path = self.path.rstrip("/") or "/"
                if path in {"/models/start", "/stack/start"}:
                    for service in server._services:
                        server._registry.start(service)
                    self._send_json(HTTPStatus.OK, server.models_state())
                    return
                if path in {"/models/stop", "/stack/stop"}:
                    server._registry.stop_all()
                    self._send_json(HTTPStatus.OK, server.models_state())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args):
                del format, args

        return Handler


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and supervise the inference system.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15880)
    parser.add_argument("--autostart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--health-timeout", type=float, default=3.0)
    parser.add_argument("--log-dir", default=str(default_log_dir()))
    parser.add_argument("--navdp-host", default="127.0.0.1")
    parser.add_argument("--navdp-port", type=int, default=18888)
    parser.add_argument("--navdp-checkpoint", default=str(REPO_ROOT / "artifacts" / "models" / "navdp-cross-modal.ckpt"))
    parser.add_argument("--navdp-device", default="cuda:0")
    parser.add_argument(
        "--navdp-tensorrt-mode",
        choices=("off", "auto", "required", "build"),
        default=os.environ.get("NAVDP_TENSORRT_MODE", "off"),
    )
    parser.add_argument(
        "--navdp-tensorrt-engine-dir",
        default=os.environ.get(
            "NAVDP_TENSORRT_ENGINE_DIR",
            str(REPO_ROOT / "artifacts" / "models" / "navdp_tensorrt"),
        ),
    )
    parser.add_argument(
        "--navdp-tensorrt-precision",
        choices=("fp16", "fp32"),
        default=os.environ.get("NAVDP_TENSORRT_PRECISION", "fp16"),
    )
    parser.add_argument("--system2-host", default="127.0.0.1")
    parser.add_argument("--system2-port", type=int, default=15801)
    parser.add_argument("--system2-llama-url", default="http://127.0.0.1:15802")
    parser.add_argument("--system2-model-path", default="")
    parser.add_argument(
        "--system2-check-lora-adapter-path",
        default=os.environ.get(
            "SYSTEM2_CHECK_LORA_ADAPTER_PATH",
            os.environ.get("INTERNVLA_CHECK_LORA_ADAPTER_PATH", ""),
        ),
    )
    parser.add_argument(
        "--system2-check-lora-scale",
        type=float,
        default=float(
            os.environ.get(
                "SYSTEM2_CHECK_LORA_SCALE",
                os.environ.get("INTERNVLA_CHECK_LORA_SCALE", "1.0"),
            )
            or 1.0
        ),
    )
    parser.add_argument(
        "--system2-check-session-system-prompt",
        default=os.environ.get(
            "SYSTEM2_CHECK_SESSION_SYSTEM_PROMPT",
            os.environ.get("INTERNVLA_CHECK_SESSION_SYSTEM_PROMPT", ""),
        ),
    )
    parser.add_argument("--planner-host", default="127.0.0.1")
    parser.add_argument("--planner-port", type=int, default=8093)
    parser.add_argument("--planner-model-path", default=str(REPO_ROOT / "artifacts" / "models" / "Qwen3-1.7B-Q4_K_M-Instruct.gguf"))
    planner_default = REPO_ROOT / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server")
    parser.add_argument("--planner-llama-server", default=str(planner_default))
    parser.add_argument("--planner-gpu-layers", type=int, default=999)
    parser.add_argument("--planner-ctx-size", type=int, default=1024)
    parser.add_argument(
        "--planner-parallel-slots",
        type=int,
        default=int(os.environ.get("PLANNER_PARALLEL_SLOTS", "2") or 2),
    )
    parser.add_argument("--planner-cache-type-k", default="q8_0")
    parser.add_argument("--planner-cache-type-v", default="q8_0")
    parser.add_argument("--dialogue-host", default="127.0.0.1")
    parser.add_argument("--dialogue-port", type=int, default=8094)
    parser.add_argument("--dialogue-model-path", default=str(REPO_ROOT / "artifacts" / "models" / "Qwen3-1.7B-Q4_K_M-Instruct.gguf"))
    parser.add_argument("--dialogue-llama-server", default=str(planner_default))
    parser.add_argument("--dialogue-gpu-layers", type=int, default=999)
    parser.add_argument("--dialogue-ctx-size", type=int, default=1024)
    parser.add_argument("--dialogue-cache-type-k", default="q8_0")
    parser.add_argument("--dialogue-cache-type-v", default="q8_0")
    parser.add_argument("--dialogue-lora-adapter-path", default=os.environ.get("DIALOGUE_LORA_ADAPTER_PATH", ""))
    parser.add_argument(
        "--dialogue-allow-prompt-only",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("DIALOGUE_ALLOW_PROMPT_ONLY"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    server = InferenceSystemServer(args)
    server.start()
    try:
        while True:
            time.sleep(3600.0)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())


InferenceStackServer = InferenceSystemServer
