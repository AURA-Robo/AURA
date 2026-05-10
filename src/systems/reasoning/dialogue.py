from __future__ import annotations

from dataclasses import dataclass

from systems.inference.api.planner import CompletionFn, PlannerClientError
from systems.memory.api import AgentMemoryContext, ConversationContext, render_agent_memory_context


@dataclass(frozen=True, slots=True)
class DialogueResult:
    reply_text: str
    degraded_reason: str | None = None


class DialogueService:
    def __init__(
        self,
        completion: CompletionFn | None = None,
        *,
        model: str,
        timeout: float,
        temperature: float = 0.4,
        max_tokens: int = 256,
    ) -> None:
        self._completion = completion
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def available(self) -> bool:
        return self._completion is not None

    def build_messages(
        self,
        utterance: str,
        *,
        language: str,
        conversation_context: ConversationContext,
        scene_preset: str | None,
        agent_memory_context: AgentMemoryContext | None = None,
    ) -> list[dict[str, str]]:
        summary = conversation_context.summary.strip() or "(empty)"
        resolved_slots = dict(conversation_context.resolved_slots)
        recent_turns = "\n".join(
            f"{turn.role}: {turn.text}"
            for turn in conversation_context.recent_turns
        ) or "(no recent turns)"
        agent_memory = render_agent_memory_context(agent_memory_context)
        system_prompt = (
            "You are the dialogue route for AURA.\n"
            "Respond conversationally and concisely in the user's language.\n"
            "Do not invent robot execution state changes.\n"
            f"Preferred language: {language or 'auto'}\n"
            f"Scene preset: {scene_preset or 'unknown'}\n"
            f"Conversation summary:\n{summary}\n"
            f"Resolved slots: {resolved_slots}\n"
            f"Recent turns:\n{recent_turns}\n"
            f"{agent_memory}"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": utterance},
        ]

    def respond(
        self,
        utterance: str,
        *,
        language: str,
        conversation_context: ConversationContext,
        scene_preset: str | None,
        agent_memory_context: AgentMemoryContext | None = None,
        fallback_text: str,
    ) -> DialogueResult:
        if self._completion is None:
            return DialogueResult(reply_text=fallback_text, degraded_reason="dialogue_model_unavailable")
        messages = self.build_messages(
            utterance,
            language=language,
            conversation_context=conversation_context,
            agent_memory_context=agent_memory_context,
            scene_preset=scene_preset,
        )
        try:
            content = self._completion(
                messages,
                self._model,
                self._timeout,
                self._temperature,
                self._max_tokens,
            )
        except PlannerClientError as exc:
            return DialogueResult(reply_text=fallback_text, degraded_reason=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            return DialogueResult(reply_text=fallback_text, degraded_reason=f"{type(exc).__name__}: {exc}")
        reply_text = " ".join(str(content).strip().split())
        if reply_text == "":
            return DialogueResult(reply_text=fallback_text, degraded_reason="dialogue_model_empty_response")
        return DialogueResult(reply_text=reply_text, degraded_reason=None)
