"""Build TensorRT engines for the NavDP backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .policy_agent import NavDP_Agent
from .tensorrt_acceleration import (
    NavDPTensorRtEngineSpec,
    build_navdp_critic_engine,
    build_navdp_noise_engine,
    build_navdp_sampler_engine,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build NavDP TensorRT engine artifacts.")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "artifacts" / "models" / "navdp-cross-modal.ckpt"),
    )
    parser.add_argument(
        "--engine-dir",
        default=str(PROJECT_ROOT / "artifacts" / "models" / "navdp_tensorrt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    intrinsic = np.asarray(
        [
            [426.0, 0.0, 320.0],
            [0.0, 426.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    agent = NavDP_Agent(
        intrinsic,
        image_size=224,
        memory_size=8,
        predict_size=24,
        temporal_depth=16,
        heads=8,
        token_dim=384,
        navi_model=str(args.checkpoint),
        device=str(args.device),
        enable_tf32=True,
        tensorrt_mode="off",
    )
    spec = NavDPTensorRtEngineSpec(precision=str(args.precision))
    sampler_engine_path = build_navdp_sampler_engine(
        agent.navi_former,
        engine_dir=str(args.engine_dir),
        checkpoint_path=str(args.checkpoint),
        spec=spec,
        force=bool(args.force),
    )
    engine_path = build_navdp_noise_engine(
        agent.navi_former,
        engine_dir=str(args.engine_dir),
        checkpoint_path=str(args.checkpoint),
        spec=spec,
        force=bool(args.force),
    )
    critic_engine_path = build_navdp_critic_engine(
        agent.navi_former,
        engine_dir=str(args.engine_dir),
        checkpoint_path=str(args.checkpoint),
        spec=spec,
        force=bool(args.force),
    )
    print(f"[INFO] NavDP TensorRT sampler engine ready: {sampler_engine_path}")
    print(f"[INFO] NavDP TensorRT noise engine ready: {engine_path}")
    print(f"[INFO] NavDP TensorRT critic engine ready: {critic_engine_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
