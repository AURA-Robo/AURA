from __future__ import annotations

from argparse import Namespace
import threading

from PIL import Image
import pytest

from systems.inference.system2 import server as system2_server
from systems.inference.system2.server import LlamaCppDualVLNRuntime


def _completion(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": text}}]}


@pytest.mark.parametrize(
    ("raw_answer", "expected"),
    [
        ("true", "true"),
        ("false", "false"),
        ("yes", "true"),
        ("'yes'", "true"),
        ("No.", "false"),
    ],
)
def test_check_session_normalizes_common_binary_answers(raw_answer: str, expected: str) -> None:
    assert LlamaCppDualVLNRuntime._normalize_check_answer(_completion(raw_answer)) == expected


def test_check_session_rejects_ambiguous_binary_answer() -> None:
    with pytest.raises(RuntimeError, match="binary true/false"):
        LlamaCppDualVLNRuntime._normalize_check_answer(_completion("yes, but also no"))


class _FakeSidecar:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.slot_erases: list[int] = []

    def slot_erase(self, slot_id: int) -> None:
        self.slot_erases.append(int(slot_id))

    def check_lora_request(self) -> list[dict[str, object]]:
        return [{"id": 7, "scale": 0.75}]

    def chat_completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _completion("true")


def _runtime_for_check(fake_sidecar: _FakeSidecar) -> LlamaCppDualVLNRuntime:
    runtime = LlamaCppDualVLNRuntime.__new__(LlamaCppDualVLNRuntime)
    runtime._sidecar = fake_sidecar
    runtime._check_session_lock = threading.Lock()
    runtime._check_slot_lock = threading.Lock()
    runtime._check_session = None
    runtime.default_check_session_id = "check-default"
    runtime.check_slot_id = 1
    runtime.max_new_tokens = 16
    runtime.check_session_system_prompt = "Answer exactly true or false."
    runtime._latest_check_image = Image.new("RGB", (4, 4), color=(0, 0, 0))
    runtime.rgb_list = []
    return runtime


def test_check_session_message_applies_check_lora_payload() -> None:
    fake_sidecar = _FakeSidecar()
    runtime = _runtime_for_check(fake_sidecar)

    response = runtime.check_session_message({"message": "Is the tv off?"})

    assert response["answer"] == "true"
    assert fake_sidecar.calls[-1]["slot_id"] == 1
    assert fake_sidecar.calls[-1]["cache_prompt"] is False
    assert fake_sidecar.calls[-1]["lora"] == [{"id": 7, "scale": 0.75}]


def test_navigation_completion_does_not_apply_check_lora_payload() -> None:
    fake_sidecar = _FakeSidecar()
    runtime = LlamaCppDualVLNRuntime.__new__(LlamaCppDualVLNRuntime)
    runtime._sidecar = fake_sidecar
    runtime.max_new_tokens = 16
    runtime.nav_slot_id = 0
    runtime.llm_output = ""
    runtime.episode_idx = 0

    output = runtime._run_completion([{"role": "user", "content": "go"}])

    assert output == "true"
    assert fake_sidecar.calls[-1]["slot_id"] == 0
    assert fake_sidecar.calls[-1]["cache_prompt"] is True
    assert "lora" not in fake_sidecar.calls[-1]


def _prepared_args(tmp_path, *, check_lora_adapter_path: str = "", check_lora_scale: object = 1.0) -> Namespace:
    llama_root = tmp_path / "llama.cpp"
    llama_root.mkdir()
    (llama_root / "llama-server.exe").write_text("", encoding="utf-8")
    model = tmp_path / "internvla.gguf"
    mmproj = tmp_path / "internvla-mmproj.gguf"
    model.write_text("", encoding="utf-8")
    mmproj.write_text("", encoding="utf-8")
    return Namespace(
        backend="llama_cpp",
        llama_cpp_root=llama_root,
        llama_model_path=model,
        llama_mmproj_path=mmproj,
        llama_url="http://127.0.0.1:15802",
        llama_ctx_size=8192,
        llama_nav_slot=0,
        llama_check_slot=1,
        llama_parallel_slots=0,
        llama_cache_type_k="q8_0",
        llama_cache_type_v="q8_0",
        default_check_session_id="check-default",
        default_check_session_auto_open="1",
        skip_system1_trajectory="1",
        check_lora_adapter_path=check_lora_adapter_path,
        check_lora_scale=check_lora_scale,
    )


def test_prepare_runtime_args_rejects_missing_check_lora_adapter(tmp_path) -> None:
    args = _prepared_args(tmp_path, check_lora_adapter_path=str(tmp_path / "missing-lora.gguf"))

    with pytest.raises(SystemExit, match="INTERNVLA_CHECK_LORA_ADAPTER_PATH does not exist"):
        system2_server.prepare_runtime_args(args)


def test_prepare_runtime_args_rejects_invalid_check_lora_scale(tmp_path) -> None:
    lora = tmp_path / "check-lora.gguf"
    lora.write_text("", encoding="utf-8")
    args = _prepared_args(tmp_path, check_lora_adapter_path=str(lora), check_lora_scale=0)

    with pytest.raises(SystemExit, match="INTERNVLA_CHECK_LORA_SCALE must be positive"):
        system2_server.prepare_runtime_args(args)


def test_check_lora_adapter_is_loaded_without_default_apply(tmp_path) -> None:
    lora = tmp_path / "check-lora.gguf"
    lora.write_text("", encoding="utf-8")
    args = system2_server.prepare_runtime_args(
        _prepared_args(tmp_path, check_lora_adapter_path=str(lora), check_lora_scale=0.5)
    )

    sidecar = system2_server.LlamaCppSidecarManager(args, auto_start=False)
    command = list(sidecar.command)

    assert "--lora" in command
    assert command[command.index("--lora") + 1] == str(lora.resolve())
    assert "--lora-init-without-apply" in command
