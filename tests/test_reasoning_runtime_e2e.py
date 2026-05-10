from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import requests

from systems.reasoning.service import ReasoningSystemServer, build_arg_parser


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until(predicate, *, timeout_s: float = 5.0) -> bool:  # noqa: ANN001
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except requests.RequestException:
            pass
        except OSError:
            pass
        time.sleep(0.02)
    return False


def _navigation_status(
    *,
    task_id: str | None = None,
    status: str = "idle",
    current_world_xy: tuple[float, float] = (0.0, 0.0),
    system2_status: str = "idle",
    decision_mode: str = "idle",
    goal_world_xy: list[float] | None = None,
    path_points: int = 0,
) -> dict[str, Any]:
    return {
        "ok": True,
        "status": status,
        "task_id": task_id,
        "goal_world_xy": goal_world_xy,
        "path_points": path_points,
        "action_override_mode": None,
        "last_error": None,
        "current_robot_pose": {
            "world_xy": [float(current_world_xy[0]), float(current_world_xy[1])],
            "world_xyz": [float(current_world_xy[0]), float(current_world_xy[1]), 0.0],
            "yaw_rad": 0.0,
        },
        "system2": {
            "status": system2_status,
            "decision_mode": decision_mode,
            "text": system2_status,
        },
    }


class _FakeChatCompletionServer:
    def __init__(self, responder: Callable[[dict[str, Any]], str | dict[str, Any]]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._responder = responder
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", _free_port()), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}/v1/chat/completions"

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                assert isinstance(payload, dict)
                return payload

            def do_POST(self) -> None:
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path != "/v1/chat/completions":
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                    return
                payload = self._read_json()
                with owner._lock:
                    owner.requests.append(payload)
                response = owner._responder(payload)
                if isinstance(response, dict):
                    self._send_json(HTTPStatus.OK, response)
                    return
                self._send_json(
                    HTTPStatus.OK,
                    {"choices": [{"message": {"content": str(response)}}]},
                )

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        return Handler


class _FakeNavigationServer:
    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []
        self.return_commands: list[dict[str, Any]] = []
        self.cancel_calls = 0
        self.status = _navigation_status()
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", _free_port()), self._handler())
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)

    def set_status(self, status: dict[str, Any]) -> None:
        with self._lock:
            self.status = dict(status)

    def _handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length else b"{}"
                payload = json.loads(raw.decode("utf-8"))
                assert isinstance(payload, dict)
                return payload

            def do_GET(self) -> None:
                path = urlparse(self.path).path.rstrip("/") or "/"
                if path == "/navigation/status":
                    with owner._lock:
                        self._send_json(HTTPStatus.OK, dict(owner.status))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def do_POST(self) -> None:
                path = urlparse(self.path).path.rstrip("/") or "/"
                payload = self._read_json()
                if path == "/navigation/cancel":
                    with owner._lock:
                        owner.cancel_calls += 1
                    self._send_json(HTTPStatus.OK, {"ok": True})
                    return
                if path == "/navigation/command":
                    task_id = str(payload.get("task_id") or "")
                    mode = str(payload.get("mode") or "")
                    with owner._lock:
                        if mode == "return_pose":
                            target = payload.get("target")
                            target_payload = target if isinstance(target, dict) else {}
                            world_xy = target_payload.get("world_xy")
                            owner.return_commands.append({"task_id": task_id, "target": dict(target_payload)})
                            owner.status = _navigation_status(
                                task_id=task_id,
                                status="running",
                                current_world_xy=(4.0, 2.0),
                                system2_status="return_pose",
                                decision_mode="return_pose",
                                goal_world_xy=list(world_xy) if isinstance(world_xy, list) else None,
                                path_points=1,
                            )
                        else:
                            owner.commands.append(dict(payload))
                            owner.status = _navigation_status(
                                task_id=task_id,
                                status="running",
                                current_world_xy=(0.0, 0.0),
                                system2_status="goal",
                                decision_mode="pixel_goal",
                                goal_world_xy=[4.0, 2.0],
                                path_points=2,
                            )
                        self._send_json(HTTPStatus.OK, dict(owner.status))
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not_found"})

            def log_message(self, format: str, *args: Any) -> None:
                del format, args

        return Handler


def test_runtime_hello_routes_to_dialogue_llm_e2e() -> None:
    navigation = _FakeNavigationServer()
    planner = _FakeChatCompletionServer(
        lambda _payload: (
            '{"route":"dialogue","intent_candidate":"smalltalk","reason":"casual_chat","confidence":0.93}'
        )
    )
    dialogue = _FakeChatCompletionServer(lambda _payload: "안녕하세요. 무엇을 도와드릴까요?")
    navigation.start()
    planner.start()
    dialogue.start()
    args = build_arg_parser().parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(_free_port()),
            "--navigation-url",
            navigation.url,
            "--navigation-timeout",
            "2.0",
            "--planner-model-base-url",
            planner.url,
            "--dialogue-model-base-url",
            dialogue.url,
        ]
    )
    reasoning = ReasoningSystemServer(args)
    try:
        reasoning.start()
        base_url = f"http://127.0.0.1:{args.port}"
        assert _wait_until(lambda: requests.get(f"{base_url}/healthz", timeout=1).status_code == 200)

        response = requests.post(
            f"{base_url}/reasoning/respond",
            json={
                "utterance": "안녕",
                "language": "ko",
                "conversation_id": "e2e-dialogue-hello",
                "scene_preset": "warehouse",
            },
            timeout=2,
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["ok"] is True
        assert payload["route"] == "dialogue"
        assert payload["reply_text"] == "안녕하세요. 무엇을 도와드릴까요?"
        assert payload["task"] is None
        assert payload["error"] is None
        assert len(planner.requests) == 1
        assert planner.requests[0]["id_slot"] == 0
        assert len(dialogue.requests) == 1
        assert dialogue.requests[0]["messages"][-1] == {"role": "user", "content": "안녕"}
        assert dialogue.requests[0]["model"] == "Qwen3-1.7B-Q4_K_M-Instruct.gguf"
        assert navigation.commands == []
    finally:
        reasoning.shutdown()
        dialogue.shutdown()
        planner.shutdown()
        navigation.shutdown()


def test_runtime_go_to_purple_box_and_come_back_e2e() -> None:
    navigation = _FakeNavigationServer()
    navigation.start()
    args = build_arg_parser().parse_args(
        [
            "--host",
            "127.0.0.1",
            "--port",
            str(_free_port()),
            "--navigation-url",
            navigation.url,
            "--navigation-timeout",
            "2.0",
            "--planner-model-base-url",
            "",
            "--dialogue-model-base-url",
            "",
        ]
    )
    reasoning = ReasoningSystemServer(args)
    try:
        reasoning.start()
        base_url = f"http://127.0.0.1:{args.port}"
        assert _wait_until(lambda: requests.get(f"{base_url}/healthz", timeout=1).status_code == 200)

        response = requests.post(
            f"{base_url}/reasoning/respond",
            json={
                "utterance": "go to purple box and come back",
                "language": "en",
                "conversation_id": "e2e-purple-return",
            },
            timeout=2,
        )
        assert response.status_code == 200
        accepted = response.json()
        assert accepted["ok"] is True
        assert accepted["route"] == "task"
        assert accepted["task"]["task_frame"]["intent"] == "navigate_to_object"
        assert accepted["task"]["task_frame"]["target"]["object"] == "purple_box_cart"
        assert [item["type"] for item in accepted["task"]["subgoals"]] == ["navigate", "return", "report"]
        assert [item["status"] for item in accepted["task"]["subgoals"]] == ["running", "pending", "pending"]

        assert len(navigation.commands) == 1
        task_id = str(navigation.commands[0]["task_id"])
        assert navigation.commands[0]["instruction"] == "Go to the purple box cart."

        navigation.set_status(
            _navigation_status(
                task_id=task_id,
                status="running",
                current_world_xy=(4.0, 2.0),
                system2_status="stop",
                decision_mode="stop",
            )
        )
        at_box = requests.get(f"{base_url}/reasoning/status", timeout=2).json()

        assert at_box["task_status"] == "running"
        assert [item["status"] for item in at_box["subgoals"]] == ["succeeded", "running", "pending"]
        assert at_box["current_subgoal"]["type"] == "return"
        assert len(navigation.return_commands) == 1
        assert navigation.return_commands[0]["task_id"] == task_id
        assert navigation.return_commands[0]["target"]["world_xy"] == [0.0, 0.0]

        navigation.set_status(
            _navigation_status(
                task_id=task_id,
                status="running",
                current_world_xy=(0.0, 0.0),
                system2_status="stop",
                decision_mode="stop",
            )
        )
        completed = requests.get(f"{base_url}/reasoning/status", timeout=2).json()

        assert completed["task_status"] == "completed"
        assert [item["status"] for item in completed["subgoals"]] == ["succeeded", "succeeded", "succeeded"]
        assert completed["subgoals"][0]["output"]["navigation_status"]["current_robot_pose"]["world_xy"] == [4.0, 2.0]
        assert completed["subgoals"][1]["output"]["navigation_status"]["current_robot_pose"]["world_xy"] == [0.0, 0.0]
        assert (
            completed["subgoals"][2]["output"]["message"]
            == "Reached the purple box cart and returned to the start pose."
        )
    finally:
        try:
            reasoning._service._task_coordinator._navigation.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        reasoning.shutdown()
        navigation.shutdown()
