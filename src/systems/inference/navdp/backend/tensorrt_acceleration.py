"""Optional TensorRT acceleration for NavDP policy subgraphs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True, slots=True)
class NavDPTensorRtEngineSpec:
    precision: str = "fp16"
    batch_size: int = 1
    sample_num: int = 16
    memory_size: int = 8
    predict_size: int = 24
    token_dim: int = 384
    temporal_depth: int = 16
    heads: int = 8

    @property
    def noise_engine_name(self) -> str:
        return (
            "navdp_noise"
            f"_b{self.batch_size}"
            f"_s{self.sample_num}"
            f"_m{self.memory_size}"
            f"_p{self.predict_size}"
            f"_t{self.token_dim}"
            f"_{self.precision}.engine"
        )

    @property
    def noise_metadata_name(self) -> str:
        return self.noise_engine_name.removesuffix(".engine") + ".json"

    @property
    def critic_engine_name(self) -> str:
        return (
            "navdp_critic"
            f"_b{self.batch_size}"
            f"_s{self.sample_num}"
            f"_m{self.memory_size}"
            f"_p{self.predict_size}"
            f"_t{self.token_dim}"
            f"_{self.precision}.engine"
        )

    @property
    def critic_metadata_name(self) -> str:
        return self.critic_engine_name.removesuffix(".engine") + ".json"

    @property
    def sampler_engine_name(self) -> str:
        return (
            "navdp_sampler"
            f"_b{self.batch_size}"
            f"_s{self.sample_num}"
            f"_m{self.memory_size}"
            f"_p{self.predict_size}"
            f"_t{self.token_dim}"
            f"_{self.precision}.engine"
        )

    @property
    def sampler_metadata_name(self) -> str:
        return self.sampler_engine_name.removesuffix(".engine") + ".json"

    @property
    def rgbd_tokens(self) -> int:
        return self.memory_size * 16

    @property
    def repeated_batch(self) -> int:
        return self.batch_size * self.sample_num


def _checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _read_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metadata_matches(
    metadata: dict[str, Any],
    checkpoint_path: str | Path,
    spec: NavDPTensorRtEngineSpec,
    *,
    engine_kind: str,
) -> bool:
    return (
        metadata.get("engine_kind") == engine_kind
        and metadata.get("spec") == asdict(spec)
        and metadata.get("checkpoint") == _checkpoint_metadata(checkpoint_path)
    )


def _torch_dtype_for_trt(trt_dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if trt_dtype == trt.DataType.FLOAT:
        return torch.float32
    if trt_dtype == trt.DataType.HALF:
        return torch.float16
    if trt_dtype == trt.DataType.INT32:
        return torch.int32
    if trt_dtype == trt.DataType.INT64:
        return torch.int64
    if trt_dtype == trt.DataType.BOOL:
        return torch.bool
    raise RuntimeError(f"Unsupported TensorRT tensor dtype: {trt_dtype!r}")


class TensorRtTorchEngine:
    """Run a static TensorRT engine by binding live CUDA torch tensors."""

    def __init__(
        self,
        engine_path: str | Path,
        *,
        expected_inputs: dict[str, tuple[int, ...]],
        expected_outputs: dict[str, tuple[int, ...]],
    ) -> None:
        import tensorrt as trt

        self._trt = trt
        self.engine_path = Path(engine_path)
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._deserialize_engine(self.engine_path)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TensorRT execution context: {self.engine_path}")

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            mode = self._engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)

        if set(self._input_names) != set(expected_inputs):
            raise RuntimeError(
                f"TensorRT engine inputs mismatch: expected={sorted(expected_inputs)} actual={sorted(self._input_names)}"
            )
        if set(self._output_names) != set(expected_outputs):
            raise RuntimeError(
                f"TensorRT engine outputs mismatch: expected={sorted(expected_outputs)} actual={sorted(self._output_names)}"
            )

        self._input_shapes = self._read_static_shapes(self._input_names)
        self._output_shapes = self._read_static_shapes(self._output_names)
        if self._input_shapes != expected_inputs:
            raise RuntimeError(f"TensorRT engine input shapes mismatch: {self._input_shapes}")
        if self._output_shapes != expected_outputs:
            raise RuntimeError(f"TensorRT engine output shapes mismatch: {self._output_shapes}")

        self._input_dtypes = {
            name: _torch_dtype_for_trt(self._engine.get_tensor_dtype(name)) for name in self._input_names
        }
        self._output_dtypes = {
            name: _torch_dtype_for_trt(self._engine.get_tensor_dtype(name)) for name in self._output_names
        }

    def _deserialize_engine(self, engine_path: Path):
        return self._runtime.deserialize_cuda_engine(engine_path.read_bytes())

    def _read_static_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]:
        shapes: dict[str, tuple[int, ...]] = {}
        for name in names:
            shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise RuntimeError(f"Dynamic TensorRT tensor shapes are not supported: {name}={shape}")
            shapes[name] = shape
        return shapes

    def run(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self._input_names):
            raise RuntimeError(f"TensorRT input map mismatch: expected={self._input_names} actual={sorted(inputs)}")

        bound_inputs: list[torch.Tensor] = []
        device = None
        for name in self._input_names:
            tensor = inputs[name]
            if not tensor.is_cuda:
                raise RuntimeError(f"TensorRT input must be a CUDA tensor: {name}")
            expected_shape = self._input_shapes[name]
            if tuple(int(dim) for dim in tensor.shape) != expected_shape:
                raise RuntimeError(f"TensorRT input shape mismatch for {name}: expected={expected_shape} actual={tuple(tensor.shape)}")
            expected_dtype = self._input_dtypes[name]
            if tensor.dtype != expected_dtype:
                tensor = tensor.to(dtype=expected_dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if device is None:
                device = tensor.device
            elif tensor.device != device:
                raise RuntimeError("All TensorRT inputs must be on the same CUDA device.")
            if not self._context.set_tensor_address(name, int(tensor.data_ptr())):
                raise RuntimeError(f"Failed to bind TensorRT input tensor: {name}")
            bound_inputs.append(tensor)

        outputs: dict[str, torch.Tensor] = {}
        for name in self._output_names:
            output = torch.empty(
                self._output_shapes[name],
                dtype=self._output_dtypes[name],
                device=device,
            )
            if not self._context.set_tensor_address(name, int(output.data_ptr())):
                raise RuntimeError(f"Failed to bind TensorRT output tensor: {name}")
            outputs[name] = output

        stream = torch.cuda.current_stream(device).cuda_stream
        if not self._context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execute_async_v3 failed.")
        return outputs


class NavDPNoiseTensorRtAccelerator:
    def __init__(self, engine_path: str | Path, spec: NavDPTensorRtEngineSpec) -> None:
        batch = spec.repeated_batch
        self.runner = TensorRtTorchEngine(
            engine_path,
            expected_inputs={
                "last_actions": (batch, spec.predict_size, 3),
                "timestep": (1,),
                "goal_embed": (batch, 1, spec.token_dim),
                "rgbd_embed": (batch, spec.rgbd_tokens, spec.token_dim),
            },
            expected_outputs={
                "noise_pred": (batch, spec.predict_size, 3),
            },
        )

    def predict_noise(
        self,
        last_actions: torch.Tensor,
        timestep: torch.Tensor,
        goal_embed: torch.Tensor,
        rgbd_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.runner.run(
            {
                "last_actions": last_actions,
                "timestep": timestep,
                "goal_embed": goal_embed,
                "rgbd_embed": rgbd_embed,
            }
        )["noise_pred"]


class NavDPCriticTensorRtAccelerator:
    def __init__(self, engine_path: str | Path, spec: NavDPTensorRtEngineSpec) -> None:
        batch = spec.repeated_batch
        self.runner = TensorRtTorchEngine(
            engine_path,
            expected_inputs={
                "predict_trajectory": (batch, spec.predict_size, 3),
                "rgbd_embed": (batch, spec.rgbd_tokens, spec.token_dim),
            },
            expected_outputs={
                "critic_values": (batch,),
            },
        )

    def predict_critic(
        self,
        predict_trajectory: torch.Tensor,
        rgbd_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.runner.run(
            {
                "predict_trajectory": predict_trajectory,
                "rgbd_embed": rgbd_embed,
            }
        )["critic_values"]


class NavDPSamplerTensorRtAccelerator:
    def __init__(self, engine_path: str | Path, spec: NavDPTensorRtEngineSpec) -> None:
        batch = spec.repeated_batch
        variance_steps = 9
        self.runner = TensorRtTorchEngine(
            engine_path,
            expected_inputs={
                "noisy_action": (batch, spec.predict_size, 3),
                "variance_noise": (variance_steps, batch, spec.predict_size, 3),
                "goal_embed": (batch, 1, spec.token_dim),
                "rgbd_embed": (batch, spec.rgbd_tokens, spec.token_dim),
            },
            expected_outputs={
                "denoised_action": (batch, spec.predict_size, 3),
            },
        )

    def sample_actions(
        self,
        noisy_action: torch.Tensor,
        variance_noise: torch.Tensor,
        goal_embed: torch.Tensor,
        rgbd_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.runner.run(
            {
                "noisy_action": noisy_action,
                "variance_noise": variance_noise,
                "goal_embed": goal_embed,
                "rgbd_embed": rgbd_embed,
            }
        )["denoised_action"]


class _NavDPNoiseExportWrapper(nn.Module):
    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        last_actions: torch.Tensor,
        timestep: torch.Tensor,
        goal_embed: torch.Tensor,
        rgbd_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy._predict_noise_torch(last_actions, timestep, goal_embed, rgbd_embed)


class _NavDPCriticExportWrapper(nn.Module):
    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        predict_trajectory: torch.Tensor,
        rgbd_embed: torch.Tensor,
    ) -> torch.Tensor:
        return self.policy._predict_critic_torch(predict_trajectory, rgbd_embed)


class _NavDPSamplerExportWrapper(nn.Module):
    def __init__(self, policy) -> None:
        super().__init__()
        self.policy = policy

    def forward(
        self,
        noisy_action: torch.Tensor,
        variance_noise: torch.Tensor,
        goal_embed: torch.Tensor,
        rgbd_embed: torch.Tensor,
    ) -> torch.Tensor:
        naction = noisy_action
        for step_index, timestep in enumerate(self.policy._ddpm_timesteps):
            noise_pred = self.policy._predict_noise_torch(naction, timestep.unsqueeze(0), goal_embed, rgbd_embed)
            noise = variance_noise[step_index] if step_index < variance_noise.shape[0] else None
            naction = self.policy._ddpm_step_with_variance_noise(noise_pred, step_index, naction, noise)
        return naction


def _engine_paths(engine_dir: str | Path, spec: NavDPTensorRtEngineSpec) -> tuple[Path, Path]:
    root = Path(engine_dir)
    return root / spec.noise_engine_name, root / spec.noise_metadata_name


def _critic_engine_paths(engine_dir: str | Path, spec: NavDPTensorRtEngineSpec) -> tuple[Path, Path]:
    root = Path(engine_dir)
    return root / spec.critic_engine_name, root / spec.critic_metadata_name


def _sampler_engine_paths(engine_dir: str | Path, spec: NavDPTensorRtEngineSpec) -> tuple[Path, Path]:
    root = Path(engine_dir)
    return root / spec.sampler_engine_name, root / spec.sampler_metadata_name


def _call_with_large_stack(fn):
    if os.name != "nt":
        return fn()

    result: dict[str, Any] = {}

    def target() -> None:
        try:
            result["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            result["exception"] = exc

    previous_stack_size = threading.stack_size()
    try:
        threading.stack_size(max(previous_stack_size, 128 * 1024 * 1024))
        thread = threading.Thread(target=target, name="navdp-tensorrt-builder")
        thread.start()
        thread.join()
    finally:
        threading.stack_size(previous_stack_size)

    if "exception" in result:
        raise result["exception"]
    return result.get("value")


def _build_serialized_engine(onnx_path: Path, precision: str):
    import tensorrt as trt

    def build():
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"Failed to parse NavDP ONNX for TensorRT:\n{errors}")

        config = builder.create_builder_config()
        normalized_precision = precision.strip().lower()
        if normalized_precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif normalized_precision != "fp32":
            raise ValueError(f"Unsupported NavDP TensorRT precision: {precision!r}")

        started = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT failed to build the NavDP engine.")
        return bytes(serialized), time.perf_counter() - started, trt.__version__

    return _call_with_large_stack(build)


def _write_engine_with_metadata(
    *,
    engine_path: Path,
    metadata_path: Path,
    serialized: bytes,
    build_seconds: float,
    tensorrt_version: str,
    engine_kind: str,
    checkpoint_path: str | Path,
    spec: NavDPTensorRtEngineSpec,
) -> None:
    tmp_engine_path = engine_path.with_suffix(engine_path.suffix + ".tmp")
    tmp_engine_path.write_bytes(serialized)
    tmp_engine_path.replace(engine_path)

    metadata = {
        "engine_kind": engine_kind,
        "spec": asdict(spec),
        "checkpoint": _checkpoint_metadata(checkpoint_path),
        "engine_path": str(engine_path.resolve()),
        "builder": {
            "torch_version": torch.__version__,
            "tensorrt_version": tensorrt_version,
            "build_seconds": build_seconds,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


class _OnnxExportContext:
    def __enter__(self):
        self._mha_fastpath_enabled = None
        mha_backend = getattr(torch.backends, "mha", None)
        if mha_backend is not None and hasattr(mha_backend, "get_fastpath_enabled"):
            self._mha_fastpath_enabled = bool(mha_backend.get_fastpath_enabled())
            mha_backend.set_fastpath_enabled(False)
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback
        mha_backend = getattr(torch.backends, "mha", None)
        if self._mha_fastpath_enabled is not None and mha_backend is not None:
            mha_backend.set_fastpath_enabled(self._mha_fastpath_enabled)
        return False


def build_navdp_noise_engine(
    policy,
    *,
    engine_dir: str | Path,
    checkpoint_path: str | Path,
    spec: NavDPTensorRtEngineSpec,
    force: bool = False,
) -> Path:
    engine_path, metadata_path = _engine_paths(engine_dir, spec)
    if (
        not force
        and engine_path.exists()
        and (metadata := _read_metadata(metadata_path)) is not None
        and _metadata_matches(metadata, checkpoint_path, spec, engine_kind="navdp_noise")
    ):
        return engine_path

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    device = policy._runtime_device()
    wrapper = _NavDPNoiseExportWrapper(policy).eval().to(device)
    batch = spec.repeated_batch
    examples = (
        torch.randn((batch, spec.predict_size, 3), device=device, dtype=torch.float32),
        torch.tensor([9], device=device, dtype=torch.int64),
        torch.randn((batch, 1, spec.token_dim), device=device, dtype=torch.float32),
        torch.randn((batch, spec.rgbd_tokens, spec.token_dim), device=device, dtype=torch.float32),
    )

    with tempfile.TemporaryDirectory(prefix="navdp_trt_") as tmp_dir:
        onnx_path = Path(tmp_dir) / "navdp_noise.onnx"
        with _OnnxExportContext(), torch.inference_mode():
            torch.onnx.export(
                wrapper,
                examples,
                str(onnx_path),
                input_names=["last_actions", "timestep", "goal_embed", "rgbd_embed"],
                output_names=["noise_pred"],
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )

        serialized, build_seconds, tensorrt_version = _build_serialized_engine(onnx_path, spec.precision)

    _write_engine_with_metadata(
        engine_path=engine_path,
        metadata_path=metadata_path,
        serialized=serialized,
        build_seconds=build_seconds,
        tensorrt_version=tensorrt_version,
        engine_kind="navdp_noise",
        checkpoint_path=checkpoint_path,
        spec=spec,
    )
    return engine_path


def build_navdp_critic_engine(
    policy,
    *,
    engine_dir: str | Path,
    checkpoint_path: str | Path,
    spec: NavDPTensorRtEngineSpec,
    force: bool = False,
) -> Path:
    engine_path, metadata_path = _critic_engine_paths(engine_dir, spec)
    if (
        not force
        and engine_path.exists()
        and (metadata := _read_metadata(metadata_path)) is not None
        and _metadata_matches(metadata, checkpoint_path, spec, engine_kind="navdp_critic")
    ):
        return engine_path

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    device = policy._runtime_device()
    wrapper = _NavDPCriticExportWrapper(policy).eval().to(device)
    batch = spec.repeated_batch
    examples = (
        torch.randn((batch, spec.predict_size, 3), device=device, dtype=torch.float32),
        torch.randn((batch, spec.rgbd_tokens, spec.token_dim), device=device, dtype=torch.float32),
    )

    with tempfile.TemporaryDirectory(prefix="navdp_trt_") as tmp_dir:
        onnx_path = Path(tmp_dir) / "navdp_critic.onnx"
        with _OnnxExportContext(), torch.inference_mode():
            torch.onnx.export(
                wrapper,
                examples,
                str(onnx_path),
                input_names=["predict_trajectory", "rgbd_embed"],
                output_names=["critic_values"],
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )
        serialized, build_seconds, tensorrt_version = _build_serialized_engine(onnx_path, spec.precision)

    _write_engine_with_metadata(
        engine_path=engine_path,
        metadata_path=metadata_path,
        serialized=serialized,
        build_seconds=build_seconds,
        tensorrt_version=tensorrt_version,
        engine_kind="navdp_critic",
        checkpoint_path=checkpoint_path,
        spec=spec,
    )
    return engine_path


def build_navdp_sampler_engine(
    policy,
    *,
    engine_dir: str | Path,
    checkpoint_path: str | Path,
    spec: NavDPTensorRtEngineSpec,
    force: bool = False,
) -> Path:
    engine_path, metadata_path = _sampler_engine_paths(engine_dir, spec)
    if (
        not force
        and engine_path.exists()
        and (metadata := _read_metadata(metadata_path)) is not None
        and _metadata_matches(metadata, checkpoint_path, spec, engine_kind="navdp_sampler")
    ):
        return engine_path

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    device = policy._runtime_device()
    wrapper = _NavDPSamplerExportWrapper(policy).eval().to(device)
    batch = spec.repeated_batch
    examples = (
        torch.randn((batch, spec.predict_size, 3), device=device, dtype=torch.float32),
        torch.randn((9, batch, spec.predict_size, 3), device=device, dtype=torch.float32),
        torch.randn((batch, 1, spec.token_dim), device=device, dtype=torch.float32),
        torch.randn((batch, spec.rgbd_tokens, spec.token_dim), device=device, dtype=torch.float32),
    )

    with tempfile.TemporaryDirectory(prefix="navdp_trt_") as tmp_dir:
        onnx_path = Path(tmp_dir) / "navdp_sampler.onnx"
        with _OnnxExportContext(), torch.inference_mode():
            torch.onnx.export(
                wrapper,
                examples,
                str(onnx_path),
                input_names=["noisy_action", "variance_noise", "goal_embed", "rgbd_embed"],
                output_names=["denoised_action"],
                opset_version=18,
                do_constant_folding=True,
                dynamo=False,
            )
        serialized, build_seconds, tensorrt_version = _build_serialized_engine(onnx_path, spec.precision)

    _write_engine_with_metadata(
        engine_path=engine_path,
        metadata_path=metadata_path,
        serialized=serialized,
        build_seconds=build_seconds,
        tensorrt_version=tensorrt_version,
        engine_kind="navdp_sampler",
        checkpoint_path=checkpoint_path,
        spec=spec,
    )
    return engine_path


def configure_navdp_tensorrt(
    policy,
    *,
    mode: str,
    engine_dir: str | Path,
    checkpoint_path: str | Path,
    spec: NavDPTensorRtEngineSpec,
) -> dict[str, Any]:
    normalized_mode = str(mode or "off").strip().lower()
    status: dict[str, Any] = {
        "mode": normalized_mode,
        "enabled": False,
        "sampler_enabled": False,
        "noise_enabled": False,
        "critic_enabled": False,
        "engine_dir": str(Path(engine_dir)),
        "precision": spec.precision,
    }
    policy.set_sampler_accelerator(None)
    policy.set_noise_accelerator(None)
    policy.set_critic_accelerator(None)
    if normalized_mode == "off":
        status["reason"] = "disabled"
        return status
    if normalized_mode not in {"auto", "required", "build"}:
        raise ValueError(f"Unsupported NavDP TensorRT mode: {mode!r}")
    if policy._runtime_device().type != "cuda":
        message = "TensorRT acceleration requires a CUDA NavDP device."
        if normalized_mode == "auto":
            status["reason"] = message
            return status
        raise RuntimeError(message)

    engine_path, metadata_path = _engine_paths(engine_dir, spec)
    try:
        if normalized_mode == "build":
            sampler_engine_path = build_navdp_sampler_engine(
                policy,
                engine_dir=engine_dir,
                checkpoint_path=checkpoint_path,
                spec=spec,
                force=False,
            )
            engine_path = build_navdp_noise_engine(
                policy,
                engine_dir=engine_dir,
                checkpoint_path=checkpoint_path,
                spec=spec,
                force=False,
            )
            critic_engine_path = build_navdp_critic_engine(
                policy,
                engine_dir=engine_dir,
                checkpoint_path=checkpoint_path,
                spec=spec,
                force=False,
            )
        else:
            sampler_engine_path, _ = _sampler_engine_paths(engine_dir, spec)
            critic_engine_path, _ = _critic_engine_paths(engine_dir, spec)
        sampler_metadata_path = _sampler_engine_paths(engine_dir, spec)[1]
        sampler_metadata = _read_metadata(sampler_metadata_path)
        if sampler_metadata is None:
            raise RuntimeError(f"TensorRT metadata file is missing: {sampler_metadata_path}")
        if not _metadata_matches(sampler_metadata, checkpoint_path, spec, engine_kind="navdp_sampler"):
            raise RuntimeError(f"TensorRT metadata does not match the active NavDP checkpoint/spec: {sampler_metadata_path}")
        if not sampler_engine_path.exists():
            raise RuntimeError(f"TensorRT engine file is missing: {sampler_engine_path}")
        policy.set_sampler_accelerator(NavDPSamplerTensorRtAccelerator(sampler_engine_path, spec))

        metadata = _read_metadata(metadata_path)
        if metadata is None:
            raise RuntimeError(f"TensorRT metadata file is missing: {metadata_path}")
        if not _metadata_matches(metadata, checkpoint_path, spec, engine_kind="navdp_noise"):
            raise RuntimeError(f"TensorRT metadata does not match the active NavDP checkpoint/spec: {metadata_path}")
        if not engine_path.exists():
            raise RuntimeError(f"TensorRT engine file is missing: {engine_path}")
        policy.set_noise_accelerator(NavDPNoiseTensorRtAccelerator(engine_path, spec))

        critic_metadata_path = _critic_engine_paths(engine_dir, spec)[1]
        critic_metadata = _read_metadata(critic_metadata_path)
        if critic_metadata is None:
            raise RuntimeError(f"TensorRT metadata file is missing: {critic_metadata_path}")
        if not _metadata_matches(critic_metadata, checkpoint_path, spec, engine_kind="navdp_critic"):
            raise RuntimeError(f"TensorRT metadata does not match the active NavDP checkpoint/spec: {critic_metadata_path}")
        if not critic_engine_path.exists():
            raise RuntimeError(f"TensorRT engine file is missing: {critic_engine_path}")
        policy.set_critic_accelerator(NavDPCriticTensorRtAccelerator(critic_engine_path, spec))

        status.update({
            "enabled": True,
            "sampler_enabled": True,
            "noise_enabled": True,
            "critic_enabled": True,
            "sampler_engine_path": str(sampler_engine_path),
            "noise_engine_path": str(engine_path),
            "critic_engine_path": str(critic_engine_path),
            "reason": "loaded",
        })
        return status
    except Exception as exc:  # noqa: BLE001 - auto mode intentionally fails open
        if normalized_mode == "auto":
            policy.set_sampler_accelerator(None)
            policy.set_noise_accelerator(None)
            policy.set_critic_accelerator(None)
            status["reason"] = f"{type(exc).__name__}: {exc}"
            return status
        raise


__all__ = [
    "NavDPNoiseTensorRtAccelerator",
    "NavDPCriticTensorRtAccelerator",
    "NavDPTensorRtEngineSpec",
    "NavDPSamplerTensorRtAccelerator",
    "build_navdp_critic_engine",
    "build_navdp_noise_engine",
    "build_navdp_sampler_engine",
    "configure_navdp_tensorrt",
]
