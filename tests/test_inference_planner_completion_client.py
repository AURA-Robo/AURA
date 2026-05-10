from __future__ import annotations

import json

import pytest

from systems.inference.planner import completion_client


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


def test_make_http_completion_sends_slot_and_cache_prompt(monkeypatch) -> None:
    captured_request = {}

    def _fake_urlopen(request, timeout):  # noqa: ANN001
        captured_request["data"] = request.data
        captured_request["timeout"] = timeout
        return _Response({"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(completion_client.urllib.request, "urlopen", _fake_urlopen)

    complete = completion_client.make_http_completion(
        "http://planner.example/v1/chat/completions",
        slot_id=0,
        cache_prompt=True,
    )
    complete(
        [{"role": "system", "content": "router"}, {"role": "user", "content": "hello"}],
        "Qwen",
        12.0,
        0.0,
        64,
    )

    body = json.loads(captured_request["data"].decode("utf-8"))
    assert body["id_slot"] == 0
    assert body["cache_prompt"] is True
    assert body["messages"] == [
        {"role": "system", "content": "router"},
        {"role": "user", "content": "hello"},
    ]


def test_call_json_completion_is_single_shot_without_repair_turns() -> None:
    calls: list[list[dict[str, str]]] = []

    def _completion(messages, model, timeout, temperature, max_tokens):  # noqa: ANN001
        del model, timeout, temperature, max_tokens
        calls.append(list(messages))
        return "not-json"

    with pytest.raises(completion_client.PlannerClientError):
        completion_client.call_json_completion(
            _completion,
            [{"role": "system", "content": "planner"}, {"role": "user", "content": "go to the tv"}],
            "Qwen",
            12.0,
            0.0,
            64,
            lambda payload: payload,
        )

    assert len(calls) == 1
    assert [message["role"] for message in calls[0]] == ["system", "user"]
