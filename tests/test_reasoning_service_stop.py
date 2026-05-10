from __future__ import annotations

from systems.reasoning.service import ReasoningCoordinator, build_arg_parser


class _NavStub:
    def __init__(self) -> None:
        self.cancel_calls = 0
        self.commands = []
        self.memory_commands = []

    def cancel(self):
        self.cancel_calls += 1
        return {"ok": True}

    def command(self, instruction: str, language: str = "en", *, task_id: str | None = None):
        self.commands.append({"instruction": instruction, "language": language, "task_id": task_id})
        return {"ok": True}

    def command_memory_target(self, target: dict[str, object], *, task_id: str | None = None):
        self.memory_commands.append({"target": dict(target), "task_id": task_id})
        return {"ok": True}


def test_reasoning_coordinator_handles_stop_locally_before_interpreter() -> None:
    args = build_arg_parser().parse_args(
        [
            "--planner-model-base-url",
            "",
            "--dialogue-model-base-url",
            "",
        ]
    )
    coordinator = ReasoningCoordinator(args)
    nav = _NavStub()
    coordinator._task_coordinator._navigation = nav  # type: ignore[attr-defined]

    start_response = coordinator.respond(
        {
            "utterance": "go to the tv",
            "language": "en",
            "conversation_id": "conv-stop",
            "interrupt_current_task": False,
        }
    )
    assert start_response["route"] == "task"
    assert start_response["task"]["task_status"] == "running"

    coordinator._interpreter.interpret = lambda *args, **kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("stop requests must bypass intent interpretation")
    )

    stop_response = coordinator.respond(
        {
            "utterance": "stop",
            "language": "en",
            "conversation_id": "conv-stop",
            "interrupt_current_task": False,
        }
    )

    assert stop_response["ok"] is True
    assert stop_response["route"] == "task"
    assert stop_response["reply_text"] == "Task cancelled."
    assert stop_response["task"]["task_status"] == "cancelled"
    assert nav.cancel_calls == 1
