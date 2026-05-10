from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from systems.control.api.runtime_args import build_arg_parser


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runtime_arg_parser_exposes_runtime_control_flags() -> None:
    args = build_arg_parser().parse_args([])
    assert args.control_mode == "cmd_vel"
    assert hasattr(args, "navigation_url")
    assert hasattr(args, "navigation_timeout")
    assert hasattr(args, "navigation_trajectory_timeout")
    assert hasattr(args, "runtime_control_api_host")
    assert hasattr(args, "runtime_control_api_port")
    assert hasattr(args, "camera_api_host")
    assert hasattr(args, "camera_api_port")
    assert args.camera_width == 448
    assert args.camera_height == 448


def test_root_entrypoints_import_without_old_launcher_layout() -> None:
    play_module = _load_module(
        "play_g1_internvla_navdp_smoke",
        REPO_ROOT / "src" / "systems" / "control" / "api" / "play_g1_internvla_navdp.py",
    )
    inference_module = _load_module(
        "serve_inference_system_smoke",
        REPO_ROOT / "src" / "systems" / "inference" / "api" / "serve_inference_system.py",
    )
    navigation_module = _load_module(
        "serve_navigation_system_smoke",
        REPO_ROOT / "src" / "systems" / "navigation" / "api" / "serve_navigation_system.py",
    )
    reasoning_module = _load_module(
        "serve_reasoning_system_smoke",
        REPO_ROOT / "src" / "systems" / "reasoning" / "api" / "serve_reasoning_system.py",
    )
    backend_module = _load_module(
        "serve_backend_smoke",
        REPO_ROOT / "src" / "backend" / "api" / "serve_backend.py",
    )
    runtime_module = _load_module(
        "serve_runtime_smoke",
        REPO_ROOT / "src" / "runtime" / "api" / "serve_runtime.py",
    )

    assert callable(play_module.main)
    inference_args = inference_module.build_arg_parser().parse_args(["--host", "127.0.0.1", "--port", "15880"])
    assert inference_args.host == "127.0.0.1"
    assert inference_args.port == 15880
    navigation_args = navigation_module.build_arg_parser().parse_args(["--host", "127.0.0.1", "--port", "17882"])
    assert navigation_args.host == "127.0.0.1"
    assert navigation_args.port == 17882
    reasoning_args = reasoning_module.build_arg_parser().parse_args(["--host", "127.0.0.1", "--port", "17881"])
    assert reasoning_args.host == "127.0.0.1"
    assert reasoning_args.port == 17881
    backend_args = backend_module.build_arg_parser().parse_args(["--host", "127.0.0.1", "--port", "18095"])
    assert backend_args.host == "127.0.0.1"
    assert backend_args.port == 18095
    runtime_args = runtime_module.build_arg_parser().parse_args(["--host", "127.0.0.1", "--port", "18096"])
    assert runtime_args.host == "127.0.0.1"
    assert runtime_args.port == 18096
