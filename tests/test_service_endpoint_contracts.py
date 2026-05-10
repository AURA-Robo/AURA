from __future__ import annotations

from pathlib import Path

from backend.api.serve_backend import build_arg_parser as build_backend_arg_parser
from backend.models import DashboardStateBuilder
from runtime.service import build_default_launchers
from systems.control.api.runtime_args import build_arg_parser as build_control_arg_parser
from systems.shared.contracts.service_endpoints import (
    BACKEND_ENDPOINT,
    CONTROL_RUNTIME_ENDPOINT,
    INFERENCE_SYSTEM_ENDPOINT,
    NAVIGATION_SYSTEM_ENDPOINT,
    REASONING_SYSTEM_ENDPOINT,
    RUNTIME_ENDPOINT,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_service_endpoint_contracts_match_documented_ports() -> None:
    assert BACKEND_ENDPOINT.base_url() == "http://127.0.0.1:18095"
    assert RUNTIME_ENDPOINT.base_url() == "http://127.0.0.1:18096"
    assert INFERENCE_SYSTEM_ENDPOINT.base_url() == "http://127.0.0.1:15880"
    assert REASONING_SYSTEM_ENDPOINT.base_url() == "http://127.0.0.1:17881"
    assert NAVIGATION_SYSTEM_ENDPOINT.base_url() == "http://127.0.0.1:17882"
    assert CONTROL_RUNTIME_ENDPOINT.base_url() == "http://127.0.0.1:8892"
    assert CONTROL_RUNTIME_ENDPOINT.status_url() == "http://127.0.0.1:8892/runtime/status"


def test_backend_and_control_arg_parsers_use_endpoint_contract_defaults() -> None:
    backend_args = build_backend_arg_parser().parse_args([])
    control_args = build_control_arg_parser().parse_args([])

    assert backend_args.port == BACKEND_ENDPOINT.port
    assert backend_args.api_base_url == BACKEND_ENDPOINT.base_url()
    assert backend_args.inference_system_url == INFERENCE_SYSTEM_ENDPOINT.base_url()
    assert backend_args.reasoning_system_url == REASONING_SYSTEM_ENDPOINT.base_url()
    assert backend_args.navigation_system_url == NAVIGATION_SYSTEM_ENDPOINT.base_url()
    assert backend_args.control_runtime_url == CONTROL_RUNTIME_ENDPOINT.base_url()
    assert control_args.navigation_url == NAVIGATION_SYSTEM_ENDPOINT.base_url()
    assert control_args.runtime_control_api_host == CONTROL_RUNTIME_ENDPOINT.host
    assert control_args.runtime_control_api_port == CONTROL_RUNTIME_ENDPOINT.port


def test_dashboard_default_state_uses_endpoint_contract_defaults() -> None:
    state = DashboardStateBuilder(api_base_url=BACKEND_ENDPOINT.base_url()).default_state()

    assert state["services"]["backend"]["healthUrl"] == BACKEND_ENDPOINT.status_url()
    assert state["services"]["runtime"]["healthUrl"] == RUNTIME_ENDPOINT.status_url()
    assert state["services"]["controlRuntime"]["healthUrl"] == CONTROL_RUNTIME_ENDPOINT.status_url()
    assert state["services"]["inferenceSystem"]["healthUrl"] == INFERENCE_SYSTEM_ENDPOINT.status_url()
    assert state["services"]["navigationSystem"]["healthUrl"] == NAVIGATION_SYSTEM_ENDPOINT.status_url()
    assert state["services"]["reasoningSystem"]["healthUrl"] == REASONING_SYSTEM_ENDPOINT.status_url()


def test_runtime_launcher_passes_control_runtime_endpoint_contract_to_script(tmp_path: Path) -> None:
    launchers = build_default_launchers(
        tmp_path,
        {"launchMode": "headless", "viewerEnabled": True},
        {},
    )

    control_launcher = next(launcher for launcher in launchers if launcher.name == "control_runtime")

    assert control_launcher.health_url == "http://127.0.0.1:8892/healthz"
    assert control_launcher.endpoints["controlRuntimeUrl"] == CONTROL_RUNTIME_ENDPOINT.base_url()
    assert control_launcher.env["RUNTIME_CONTROL_API_HOST"] == CONTROL_RUNTIME_ENDPOINT.host
    assert control_launcher.env["RUNTIME_CONTROL_API_PORT"] == str(CONTROL_RUNTIME_ENDPOINT.port)


def test_windows_launchers_use_control_runtime_endpoint_contract_defaults() -> None:
    control_text = (REPO_ROOT / "scripts" / "run_system" / "control_runtime_windows.bat").read_text(
        encoding="utf-8",
        errors="ignore",
    )
    backend_text = (REPO_ROOT / "scripts" / "run_system" / "backend_windows.ps1").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert 'set "RUNTIME_CONTROL_API_PORT=8892"' in control_text
    assert '"http://127.0.0.1:8892"' in backend_text
