from __future__ import annotations

from typing import Any


class DialoguePolicy:
    @staticmethod
    def clarification_reply(task_frame: dict[str, Any] | None = None) -> str:
        clarification = task_frame.get("clarification") if isinstance(task_frame, dict) else {}
        question = clarification.get("question_ko") if isinstance(clarification, dict) else None
        if isinstance(question, str) and question.strip():
            return question.strip()
        return "어떤 작업을 원하는지 조금 더 구체적으로 알려주세요."

    @staticmethod
    def unsupported_reply(task_frame: dict[str, Any] | None = None) -> str:
        target = ""
        attribute = ""
        if isinstance(task_frame, dict):
            target_payload = task_frame.get("target")
            query_payload = task_frame.get("query")
            if isinstance(target_payload, dict):
                target = str(target_payload.get("object") or "").replace("_", " ").strip()
            if isinstance(query_payload, dict):
                attribute = str(query_payload.get("attribute") or "").replace("_", " ").strip()
        if target and attribute:
            return f"현재는 {target}의 {attribute} 확인 요청을 지원하지 않습니다."
        if target:
            return f"현재는 {target} 관련 요청을 지원하지 않습니다."
        return "현재 이 요청은 지원하지 않습니다."

    @staticmethod
    def busy_reply() -> str:
        return "현재 실행 중인 작업이 있습니다. 새 작업으로 바꾸려면 interrupt_current_task=true로 다시 요청하세요."

    @staticmethod
    def degraded_dialogue_reply() -> str:
        return "대화 모델이 현재 사용할 수 없어 일반 대화 응답을 바로 생성하지 못했습니다."

    @staticmethod
    def accepted_task_reply(task_frame: dict[str, Any] | None = None) -> str:
        if not isinstance(task_frame, dict):
            return "작업을 시작했습니다."
        intent = str(task_frame.get("intent") or "").strip()
        target_payload = task_frame.get("target")
        target_object = ""
        if isinstance(target_payload, dict):
            target_object = str(target_payload.get("object") or "").replace("_", " ").strip()
        if target_object and intent == "check_state":
            return f"{target_object} 확인 작업을 시작했습니다."
        if target_object:
            return f"{target_object} 관련 작업을 시작했습니다."
        return "작업을 시작했습니다."
