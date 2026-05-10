from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_ROOT = REPO_ROOT / "src" / "systems"
RUN_SCRIPTS_ROOT = REPO_ROOT / "scripts" / "run_system"
EXTERNAL_DASHBOARD_ROOT = Path(r"C:\Users\mango\project\AURA\dashboard")
SUBSYSTEMS = {
    "control",
    "inference",
    "memory",
    "navigation",
    "perception",
    "reasoning",
    "shared",
    "transport",
    "world_state",
}
TOP_LEVEL_SERVICES = {"backend", "runtime"}
REMOVED_LAYER_DIRS = (
    SYSTEMS_ROOT / "control" / "application",
    SYSTEMS_ROOT / "control" / "domain",
    SYSTEMS_ROOT / "control" / "infrastructure",
    SYSTEMS_ROOT / "control" / "bin",
    SYSTEMS_ROOT / "navigation" / "application",
    SYSTEMS_ROOT / "navigation" / "domain",
    SYSTEMS_ROOT / "navigation" / "infrastructure",
    SYSTEMS_ROOT / "navigation" / "bin",
    SYSTEMS_ROOT / "inference" / "application",
    SYSTEMS_ROOT / "inference" / "domain",
    SYSTEMS_ROOT / "inference" / "infrastructure",
    SYSTEMS_ROOT / "inference" / "bin",
    SYSTEMS_ROOT / "perception" / "application",
    SYSTEMS_ROOT / "perception" / "infrastructure",
    SYSTEMS_ROOT / "world_state" / "application",
    SYSTEMS_ROOT / "world_state" / "domain",
    SYSTEMS_ROOT / "world_state" / "infrastructure",
)


def _module_parts(path: Path) -> tuple[str, ...]:
    rel = path.relative_to(REPO_ROOT / "src")
    return rel.with_suffix("").parts


def _iter_python_files() -> list[Path]:
    return [path for path in SYSTEMS_ROOT.rglob("*.py") if "__pycache__" not in path.parts]


def _import_targets(module: ast.AST) -> list[str]:
    targets: list[str] = []
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            targets.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.append(node.module)
    return targets


def test_runtime_subsystems_and_canonical_launchers_exist() -> None:
    for subsystem in SUBSYSTEMS:
        assert (SYSTEMS_ROOT / subsystem).is_dir()
    for service in TOP_LEVEL_SERVICES:
        assert (REPO_ROOT / "src" / service).is_dir()
    assert EXTERNAL_DASHBOARD_ROOT.is_dir()
    assert (RUN_SCRIPTS_ROOT / "inference_system_windows.bat").is_file()
    assert (RUN_SCRIPTS_ROOT / "reasoning_system_windows.bat").is_file()
    assert (RUN_SCRIPTS_ROOT / "navigation_system_windows.bat").is_file()
    assert (RUN_SCRIPTS_ROOT / "control_runtime_windows.bat").is_file()
    assert (RUN_SCRIPTS_ROOT / "runtime_windows.ps1").is_file()
    assert (RUN_SCRIPTS_ROOT / "backend_windows.ps1").is_file()
    assert (RUN_SCRIPTS_ROOT / "dashboard_dev_windows.ps1").is_file()
    assert (RUN_SCRIPTS_ROOT / "run_dashboard_windows.ps1").is_file()


def test_old_operational_surfaces_are_removed() -> None:
    assert not (SYSTEMS_ROOT / "control" / "api" / "nav_command_api.py").exists()
    assert not (SYSTEMS_ROOT / "dashboard").exists()
    assert not (SYSTEMS_ROOT / "dashboard_backend").exists()
    assert not (SYSTEMS_ROOT / "navigation" / "api" / "navdp_server.py").exists()
    assert not (SYSTEMS_ROOT / "runtime_supervisor").exists()
    assert not (REPO_ROOT / "scripts" / "serve_planner_qwen3_nothink.ps1").exists()
    assert not (SYSTEMS_ROOT / "planner").exists()
    assert not (SYSTEMS_ROOT / "control" / "bin" / "send_internvla_nav_command_windows.bat").exists()
    assert not (RUN_SCRIPTS_ROOT / "inference_stack_windows.bat").exists()
    assert not (RUN_SCRIPTS_ROOT / "dashboard_backend_windows.ps1").exists()
    assert not (RUN_SCRIPTS_ROOT / "runtime_supervisor_windows.ps1").exists()


def test_selected_clean_layer_directories_are_removed() -> None:
    for path in REMOVED_LAYER_DIRS:
        assert not path.exists(), path


def test_cross_subsystem_imports_only_use_api_or_shared_contracts() -> None:
    for path in _iter_python_files():
        parts = _module_parts(path)
        if len(parts) < 3 or parts[0] != "systems":
            continue
        current_subsystem = parts[1]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target in _import_targets(tree):
            if not target.startswith("systems."):
                continue
            target_parts = tuple(target.split("."))
            if len(target_parts) < 3:
                continue
            target_subsystem = target_parts[1]
            target_surface = target_parts[2]
            if target_subsystem == "shared":
                assert target_parts[:3] == ("systems", "shared", "contracts"), (path, target)
                continue
            if target_subsystem == current_subsystem:
                continue
            assert target_surface == "api", (path, target)


def test_navigation_does_not_import_memory_implementation_directly() -> None:
    for path in (SYSTEMS_ROOT / "navigation").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets = _import_targets(tree)
        assert "systems.memory.stm" not in targets, path
        assert "systems.memory.api.runtime" not in targets, path


def test_perception_does_not_depend_on_navigation_geometry() -> None:
    blocked_targets = {
        "systems.navigation.geometry",
        "systems.navigation.api.geometry",
    }
    for path in (SYSTEMS_ROOT / "perception").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets = set(_import_targets(tree))
        assert blocked_targets.isdisjoint(targets), (path, sorted(targets & blocked_targets))


def test_world_state_does_not_import_navigation_facades() -> None:
    for path in (SYSTEMS_ROOT / "world_state").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets = _import_targets(tree)
        for target in targets:
            assert not target.startswith("systems.navigation."), (path, target)


def test_control_subsystem_does_not_import_simulation_modules() -> None:
    for path in (SYSTEMS_ROOT / "control").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets = _import_targets(tree)
        for target in targets:
            assert not target.startswith("simulation."), (path, target)
