from __future__ import annotations

from pathlib import Path

from simulation.api.paths import repo_dir, resolve_default_policy_path, resolve_default_robot_usd_path


def test_repo_dir_and_default_assets_resolve_from_repo_root() -> None:
    repo_root = Path(repo_dir()).resolve()
    expected_root = Path(__file__).resolve().parents[1]

    policy_path = Path(resolve_default_policy_path(str(repo_root)))
    robot_usd_path = Path(resolve_default_robot_usd_path(str(repo_root)))

    assert repo_root == expected_root
    assert policy_path == repo_root / "artifacts" / "models" / "g1_policy_fp16.engine"
    assert policy_path.is_file()
    assert robot_usd_path == repo_root / "robots" / "g1" / "g1_d455.usd"
    assert robot_usd_path.is_file()
