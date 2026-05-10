from __future__ import annotations

import pytest
import torch

from systems.inference.navdp.backend.tensorrt_acceleration import (
    NavDPTensorRtEngineSpec,
    configure_navdp_tensorrt,
)


class _FakePolicy:
    def __init__(self) -> None:
        self.accelerator = object()

    def _runtime_device(self):
        return torch.device("cuda:0")

    def set_sampler_accelerator(self, accelerator) -> None:
        self.accelerator = accelerator

    def set_noise_accelerator(self, accelerator) -> None:
        self.accelerator = accelerator

    def set_critic_accelerator(self, accelerator) -> None:
        self.accelerator = accelerator


def test_navdp_tensorrt_engine_spec_names_encode_static_shapes() -> None:
    spec = NavDPTensorRtEngineSpec(precision="fp16")

    assert spec.noise_engine_name == "navdp_noise_b1_s16_m8_p24_t384_fp16.engine"
    assert spec.noise_metadata_name == "navdp_noise_b1_s16_m8_p24_t384_fp16.json"
    assert spec.sampler_engine_name == "navdp_sampler_b1_s16_m8_p24_t384_fp16.engine"
    assert spec.sampler_metadata_name == "navdp_sampler_b1_s16_m8_p24_t384_fp16.json"
    assert spec.rgbd_tokens == 128
    assert spec.repeated_batch == 16


def test_navdp_tensorrt_auto_mode_fails_open_when_engine_is_missing(tmp_path) -> None:
    policy = _FakePolicy()

    status = configure_navdp_tensorrt(
        policy,
        mode="auto",
        engine_dir=tmp_path,
        checkpoint_path=tmp_path / "missing.ckpt",
        spec=NavDPTensorRtEngineSpec(),
    )

    assert status["enabled"] is False
    assert status["mode"] == "auto"
    assert "metadata file is missing" in str(status["reason"])
    assert policy.accelerator is None


def test_navdp_tensorrt_required_mode_raises_when_engine_is_missing(tmp_path) -> None:
    policy = _FakePolicy()

    with pytest.raises(RuntimeError, match="metadata file is missing"):
        configure_navdp_tensorrt(
            policy,
            mode="required",
            engine_dir=tmp_path,
            checkpoint_path=tmp_path / "missing.ckpt",
            spec=NavDPTensorRtEngineSpec(),
        )
