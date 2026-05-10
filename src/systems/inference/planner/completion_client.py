from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any


class PlannerClientError(RuntimeError):
    pass


CompletionFn = Callable[[list[dict[str, str]], str, float, float, int], str]
DEFAULT_PLANNER_INTENT_SLOT_ID = 0
DEFAULT_PLANNER_TASK_FRAME_SLOT_ID = 1


def make_http_completion(
    base_url: str,
    *,
    slot_id: int | None = None,
    cache_prompt: bool | None = None,
) -> CompletionFn:
    def complete(
        messages: list[dict[str, str]],
        model: str,
        timeout: float,
        temperature: float,
        max_tokens: int,
    ) -> str:
        body: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if model:
            body["model"] = model
        if slot_id is not None:
            body["id_slot"] = int(slot_id)
        if cache_prompt is not None:
            body["cache_prompt"] = bool(cache_prompt)
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            base_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise PlannerClientError(f"LLM server returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise PlannerClientError(f"Could not reach LLM server at {base_url}. ({exc})") from exc
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise PlannerClientError(f"Unexpected LLM response shape: {payload}") from exc
        if not isinstance(content, str) or not content.strip():
            raise PlannerClientError("LLM response content was empty.")
        return content

    return complete


def make_planner_intent_completion(
    base_url: str,
    *,
    slot_id: int = DEFAULT_PLANNER_INTENT_SLOT_ID,
    cache_prompt: bool = True,
) -> CompletionFn:
    return make_http_completion(base_url, slot_id=slot_id, cache_prompt=cache_prompt)


def make_planner_task_frame_completion(
    base_url: str,
    *,
    slot_id: int = DEFAULT_PLANNER_TASK_FRAME_SLOT_ID,
    cache_prompt: bool = True,
) -> CompletionFn:
    return make_http_completion(base_url, slot_id=slot_id, cache_prompt=cache_prompt)


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = strip_code_fences(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    if start == -1:
        raise PlannerClientError("LLM response did not contain a JSON object.")
    depth = 0
    in_string = False
    escape = False
    end = None
    for idx, char in enumerate(candidate[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break
    if end is None:
        raise PlannerClientError("LLM response contained malformed JSON.")
    try:
        return json.loads(candidate[start:end])
    except json.JSONDecodeError as exc:
        raise PlannerClientError(f"LLM response contained invalid JSON: {exc}") from exc


def call_json_completion(
    completion: CompletionFn,
    messages: list[dict[str, str]],
    model: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    validator,
) -> dict[str, Any]:
    raw_content = completion(list(messages), model, timeout, temperature, max_tokens)
    try:
        parsed = extract_json_object(raw_content)
        return validator(parsed)
    except Exception as exc:  # noqa: BLE001
        raise PlannerClientError(
            f"LLM returned invalid JSON or failed validation. Last response: {raw_content}"
        ) from exc


def call_json_with_retry(
    completion: CompletionFn,
    messages: list[dict[str, str]],
    model: str,
    timeout: float,
    temperature: float,
    max_tokens: int,
    validator,
) -> dict[str, Any]:
    return call_json_completion(
        completion,
        messages,
        model,
        timeout,
        temperature,
        max_tokens,
        validator,
    )
