from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

from systems.memory.api import (
    KnowledgeContext,
    KnowledgeFactChunk,
    KnowledgeLexiconEntry,
    KnowledgeRule,
    apply_knowledge_guards,
    lexicon_alias_maps,
)
from systems.reasoning.planner.normalizer import (
    detect_attribute,
    detect_desired_check,
    detect_instance_hint,
    detect_object_class,
    detect_room_hints,
    infer_intent,
    normalize_text,
)
from systems.reasoning.planner.ontology import QUERY_OPERATORS, QUERY_TYPES, TASK_FRAME_INTENTS
from systems.reasoning.planner.task_frames import expected_value_from_desired_check, task_frame_to_plan
from systems.reasoning.planner.validator import (
    PlannerValidationError,
    validate_plan_request,
    validate_plan_response,
    validate_task_frame_response,
)
from systems.inference.api.planner import CompletionFn, PlannerClientError, call_json_completion
from systems.reasoning.planner_catalog_models import EXECUTION_INTENT_KEYS

if TYPE_CHECKING:
    from systems.reasoning.planner_catalog_runtime import PlannerCatalogRuntimeHandle

DEFAULT_MODEL = "Qwen3-1.7B-Q4_K_M-Instruct.gguf"
ROUTE_VALUES = ("task", "dialogue", "clarification", "unsupported")

RETURN_HOME_KEYWORDS = (
    "\ub3cc\uc544\uc640",
    "\uac14\ub2e4 \uc640",
    "\ud655\uc778\ud558\uace0 \uc640",
    "\ub2e4\ub140\uc640",
    "\ubcf4\uace0 \uc640",
    "come back",
    "return",
    "and come back",
)
HERE_ROOM_KEYWORDS = (
    "\uc5ec\uae30",
    "\uc774 \ubc29",
    "\ud604\uc7ac \ubc29",
    "here",
    "this room",
    "current room",
)
COMMAND_KEYWORDS = (
    "\ud655\uc778",
    "\ubd10\uc918",
    "\ucc3e\uc544",
    "\uc774\ub3d9",
    "\uac00\uc918",
    "\ub2e4\uac00\uac00",
    "\uc0c1\ud0dc",
    "check",
    "inspect",
    "find",
    "navigate",
    "go to",
    "move to",
    "status",
)

CLARIFICATION_MESSAGES = {
    "multiple_explicit_rooms": "\uc5b4\ub290 \ubc29\uc758 \ub300\uc0c1\uc744 \ub9d0\ud558\ub294\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
    "multiple_recent_target_rooms": "\uac19\uc740 \ub300\uc0c1\uc774 \uc5ec\ub7ec \uacf3\uc5d0 \uc788\uc2b5\ub2c8\ub2e4. \uc5b4\ub290 \ubc29\uc758 \ub300\uc0c1\uc778\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
    "missing_target": "\uc5b4\ub5a4 \ub300\uc0c1\uc744 \ud655\uc778\ud574\uc57c \ud558\ub294\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
    "missing_attribute": "\uc5b4\ub5a4 \uc18d\uc131\uc744 \ud655\uc778\ud574\uc57c \ud558\ub294\uc9c0 \uc54c\ub824\uc8fc\uc138\uc694.",
}


def _dump_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


@dataclass(frozen=True)
class PlanSemanticAnalysis:
    normalized_utterance: str
    intent_candidate: str
    target_class_candidate: str | None
    attribute_candidate: str | None
    expected_value_candidate: str | None
    explicit_room_candidates: tuple[str, ...]
    recent_target_rooms: tuple[str, ...]
    preferred_room: str | None
    instance_hint: str | None
    return_home_requested: bool
    clarification_reasons: tuple[str, ...]
    unsupported_reasons: tuple[str, ...]


@dataclass(frozen=True)
class RouteClassification:
    route: str
    normalized_utterance: str
    intent_candidate: str | None
    reason: str | None
    confidence: float


def _semantic_summary(analysis: PlanSemanticAnalysis) -> dict[str, Any]:
    return {
        "intent_candidate": analysis.intent_candidate,
        "target_class_candidate": analysis.target_class_candidate,
        "attribute_candidate": analysis.attribute_candidate,
        "expected_value_candidate": analysis.expected_value_candidate,
        "explicit_room_candidates": list(analysis.explicit_room_candidates),
        "recent_target_rooms": list(analysis.recent_target_rooms),
        "preferred_room": analysis.preferred_room,
        "instance_hint": analysis.instance_hint,
        "return_home_requested": analysis.return_home_requested,
        "clarification_reasons": list(analysis.clarification_reasons),
        "unsupported_reasons": list(analysis.unsupported_reasons),
    }


def build_route_messages(request: dict[str, Any], analysis: PlanSemanticAnalysis) -> list[dict[str, str]]:
    structured_input = {
        "utterance_ko": request.get("utterance_ko"),
        "robot_state": request.get("robot_state"),
        "world_summary": request.get("world_summary"),
        "capabilities": request.get("capabilities"),
        "knowledge_context": request.get("knowledge_context"),
        "semantic_summary": _semantic_summary(analysis),
    }
    system = f"""You are an intent router for a mobile robot reasoning subsystem.

Your job:
- Read one request and classify the route.
- Return JSON only.
- Allowed routes: {", ".join(ROUTE_VALUES)}.
- Never output busy. Runtime handles active-task busy checks locally.
- Stop, cancel, and halt are handled outside this classifier.
- Use dialogue for casual conversation or simple chat.
- Use clarification when execution-critical information is missing or ambiguous.
- Use unsupported when the request is outside robot capabilities.
- Use task when the request is an executable robot command.
- Treat capabilities.detectable_objects as known examples, not as a closed object enum.
- If semantic_summary.target_class_candidate is present and there are no clarification or unsupported reasons, do not classify the target as missing.

Required JSON schema:
{{
  "route": "task | dialogue | clarification | unsupported",
  "intent_candidate": "string or null",
  "reason": "string",
  "confidence": 0.0
}}
"""
    user = "Structured input:\n" + _dump_json(structured_input) + "\n\nReturn JSON only."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_task_frame_messages(request: dict[str, Any], analysis: PlanSemanticAnalysis) -> list[dict[str, str]]:
    semantic_summary = _semantic_summary(analysis)
    system = f"""You are a semantic planner for a mobile robot.

Your job:
- Convert the user's Korean command into a fixed JSON task frame.
- Fill only the allowed enum values.
- Never generate free-form navigation instructions.
- Never invent rooms, attributes, operators, or capabilities.
- target.object is open-world: prefer a known capability name when obvious, otherwise use a concise snake_case object phrase from the user's command.
- For find_object and navigate_to_object, keep every query field null even when the object phrase contains an adjective such as a color.
- Use the semantic summary as advisory normalized context.
- Respect knowledge_context.hard_rules as mandatory constraints.
- Use knowledge_context.lexicon_entries to resolve aliases before deciding intent or target.
- Use knowledge_context.facts only as supporting context with the provided source anchors.
- If the command is ambiguous and affects execution, set clarification.required=true.
- Output JSON only.

Allowed intents:
{", ".join(TASK_FRAME_INTENTS)}

Allowed query types:
{", ".join(QUERY_TYPES)}

Allowed query operators:
{", ".join(QUERY_OPERATORS)}

Required JSON schema:
{{
  "intent": "string",
  "target": {{
    "object": "string or null",
    "instance_hint": "string or null",
    "location_hint": "string or null"
  }},
  "query": {{
    "query_type": "string or null",
    "attribute": "string or null",
    "operator": "string or null",
    "expected_value": "string or null"
  }},
  "constraints": {{
    "return_after_check": false,
    "report_result": true
  }},
  "clarification": {{
    "required": false,
    "question_ko": null
  }}
}}
"""
    user = (
        "Structured input:\n"
        + _dump_json(request)
        + "\n\nSemantic summary:\n"
        + _dump_json(semantic_summary)
        + "\n\nReturn JSON only."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@dataclass
class PlannerService:
    completion: CompletionFn | None = None
    intent_completion: CompletionFn | None = None
    planner_catalog_runtime: PlannerCatalogRuntimeHandle | None = None
    model: str = DEFAULT_MODEL
    timeout: float = 120.0
    temperature: float = 0.1
    max_tokens: int = 192
    intent_temperature: float = 0.0
    intent_max_tokens: int = 96

    def classify_route(self, request: dict[str, Any]) -> RouteClassification:
        request = validate_plan_request(request)
        analysis = self._analyze_request(request)
        if self.intent_completion is None:
            return self._route_from_analysis(request, analysis)
        try:
            payload = call_json_completion(
                self.intent_completion,
                build_route_messages(request, analysis),
                self.model,
                self.timeout,
                self.intent_temperature,
                self.intent_max_tokens,
                self._validate_route_payload,
            )
            payload = self._apply_local_route_guard(payload, analysis)
            return RouteClassification(
                route=str(payload["route"]),
                normalized_utterance=analysis.normalized_utterance,
                intent_candidate=(
                    None
                    if payload.get("intent_candidate") in (None, "")
                    else str(payload.get("intent_candidate"))
                ),
                reason=None if payload.get("reason") in (None, "") else str(payload.get("reason")),
                confidence=float(payload["confidence"]),
            )
        except (PlannerClientError, PlannerValidationError, ValueError, TypeError):
            return self._route_from_analysis(request, analysis)

    def plan_task_frame(self, request: dict[str, Any]) -> dict[str, Any]:
        request = validate_plan_request(request)
        analysis = self._analyze_request(request)
        validator = lambda payload: validate_task_frame_response(payload, request["capabilities"])
        finalize = lambda payload: validator(self._apply_catalog_rules(self._apply_knowledge_rules(payload, request)))
        if self.completion is None:
            return finalize(self._build_task_frame_from_analysis(request, analysis))
        try:
            llm_response = call_json_completion(
                self.completion,
                build_task_frame_messages(request, analysis),
                self.model,
                self.timeout,
                self.temperature,
                self.max_tokens,
                validator,
            )
            return finalize(llm_response)
        except (PlannerClientError, PlannerValidationError):
            return finalize(self._build_task_frame_from_analysis(request, analysis))

    def plan(self, request: dict[str, Any]) -> dict[str, Any]:
        request = validate_plan_request(request)
        task_frame = self.plan_task_frame(request)
        return validate_plan_response(task_frame_to_plan(task_frame), request["capabilities"])

    @staticmethod
    def _validate_route_payload(payload: dict[str, Any]) -> dict[str, Any]:
        route = str(payload.get("route") or "").strip().lower()
        if route not in ROUTE_VALUES:
            raise ValueError(f"unsupported route: {route}")
        confidence_raw = payload.get("confidence", 0.0)
        confidence = float(confidence_raw)
        if confidence < 0.0:
            confidence = 0.0
        if confidence > 1.0:
            confidence = 1.0
        reason = payload.get("reason")
        intent_candidate = payload.get("intent_candidate")
        return {
            "route": route,
            "intent_candidate": None if intent_candidate in (None, "") else str(intent_candidate),
            "reason": None if reason in (None, "") else str(reason),
            "confidence": confidence,
        }

    @staticmethod
    def _apply_local_route_guard(
        payload: dict[str, Any],
        analysis: PlanSemanticAnalysis,
    ) -> dict[str, Any]:
        route = str(payload.get("route") or "").strip().lower()
        reason = str(payload.get("reason") or "").strip().lower()
        if (
            route == "clarification"
            and "missing_target" in reason
            and analysis.target_class_candidate is not None
            and not analysis.clarification_reasons
            and not analysis.unsupported_reasons
        ):
            guarded = dict(payload)
            guarded["route"] = "task"
            guarded["intent_candidate"] = analysis.intent_candidate
            guarded["reason"] = "local_target_detected"
            guarded["confidence"] = max(float(guarded.get("confidence", 0.0)), 0.8)
            return guarded
        return payload

    def _route_from_analysis(
        self,
        request: dict[str, Any],
        analysis: PlanSemanticAnalysis,
    ) -> RouteClassification:
        del request
        if self._is_task_like(analysis):
            if analysis.unsupported_reasons:
                return RouteClassification(
                    route="unsupported",
                    normalized_utterance=analysis.normalized_utterance,
                    intent_candidate=analysis.intent_candidate,
                    reason=analysis.unsupported_reasons[0],
                    confidence=0.9,
                )
            if analysis.clarification_reasons:
                return RouteClassification(
                    route="clarification",
                    normalized_utterance=analysis.normalized_utterance,
                    intent_candidate=analysis.intent_candidate,
                    reason=analysis.clarification_reasons[0],
                    confidence=0.85,
                )
            return RouteClassification(
                route="task",
                normalized_utterance=analysis.normalized_utterance,
                intent_candidate=analysis.intent_candidate,
                reason="heuristic_task",
                confidence=0.8,
            )
        return RouteClassification(
            route="dialogue",
            normalized_utterance=analysis.normalized_utterance,
            intent_candidate=analysis.intent_candidate,
            reason="non_task_dialogue",
            confidence=0.7,
        )

    @staticmethod
    def _is_task_like(analysis: PlanSemanticAnalysis) -> bool:
        if analysis.intent_candidate != "unsupported":
            return True
        if analysis.target_class_candidate is not None:
            return True
        return _contains_any(analysis.normalized_utterance, COMMAND_KEYWORDS)

    def _analyze_request(self, request: dict[str, Any]) -> PlanSemanticAnalysis:
        utterance = request["utterance_ko"]
        normalized_utterance = normalize_text(utterance)
        capabilities = request["capabilities"]
        current_room = request["robot_state"].get("current_room")
        lexicon_maps = lexicon_alias_maps(_knowledge_context_payload(request).get("lexicon_entries", []))
        target_class = detect_object_class(utterance, extra_aliases=lexicon_maps["object"])
        attribute = detect_attribute(utterance, target_class, extra_aliases=lexicon_maps["attribute"])
        desired_check = detect_desired_check(utterance, attribute)
        expected_value = expected_value_from_desired_check(desired_check)
        explicit_room_candidates = tuple(
            detect_room_hints(
                utterance,
                request["world_summary"]["known_rooms"],
                extra_aliases=lexicon_maps["room"],
            )
        )
        recent_target_rooms = tuple(self._collect_recent_target_rooms(request, target_class))
        preferred_room = self._select_preferred_room(
            explicit_room_candidates,
            recent_target_rooms,
            current_room,
            normalized_utterance,
        )
        instance_hint = detect_instance_hint(utterance)
        return_home_requested = _contains_any(normalized_utterance, RETURN_HOME_KEYWORDS)
        intent = infer_intent(utterance, target_class, attribute)

        clarification_reasons: list[str] = []
        unsupported_reasons: list[str] = []

        if len(explicit_room_candidates) > 1:
            clarification_reasons.append("multiple_explicit_rooms")
        elif target_class is not None and not explicit_room_candidates and len(recent_target_rooms) > 1:
            clarification_reasons.append("multiple_recent_target_rooms")

        if target_class is None and _contains_any(normalized_utterance, COMMAND_KEYWORDS):
            clarification_reasons.append("missing_target")

        if intent == "inspect_attribute" and target_class is not None and attribute is None:
            inspectable = capabilities["inspectable_attributes"].get(target_class, [])
            if len(inspectable) > 1:
                clarification_reasons.append("missing_attribute")
            elif len(inspectable) == 1:
                attribute = inspectable[0]

        if attribute is not None:
            inspectable = set(capabilities["inspectable_attributes"].get(target_class or "", ()))
            if attribute not in inspectable:
                unsupported_reasons.append("uninspectable_attribute")

        if return_home_requested and not capabilities["can_return_home"]:
            unsupported_reasons.append("cannot_return_home")

        return PlanSemanticAnalysis(
            normalized_utterance=normalized_utterance,
            intent_candidate=intent,
            target_class_candidate=target_class,
            attribute_candidate=attribute,
            expected_value_candidate=expected_value,
            explicit_room_candidates=explicit_room_candidates,
            recent_target_rooms=recent_target_rooms,
            preferred_room=preferred_room,
            instance_hint=instance_hint,
            return_home_requested=return_home_requested,
            clarification_reasons=tuple(_dedupe(clarification_reasons)),
            unsupported_reasons=tuple(_dedupe(unsupported_reasons)),
        )

    def _collect_recent_target_rooms(self, request: dict[str, Any], target_class: str | None) -> list[str]:
        if target_class is None:
            return []
        recent_seen = request["world_summary"]["recent_seen"]
        matching_rows: list[tuple[int | None, str]] = []
        for row in recent_seen:
            row_class = row.get("class")
            row_room = row.get("room")
            if row_class != target_class or not isinstance(row_room, str):
                continue
            age_value = row.get("age_sec")
            age_sec = age_value if isinstance(age_value, int) else None
            matching_rows.append((age_sec, row_room))
        matching_rows.sort(key=lambda item: item[0] if item[0] is not None else 10**9)
        return _dedupe([room for _, room in matching_rows])

    def _select_preferred_room(
        self,
        explicit_room_candidates: tuple[str, ...],
        recent_target_rooms: tuple[str, ...],
        current_room: str | None,
        normalized_utterance: str,
    ) -> str | None:
        if len(explicit_room_candidates) == 1:
            return explicit_room_candidates[0]
        if current_room and _contains_any(normalized_utterance, HERE_ROOM_KEYWORDS):
            return current_room
        if len(recent_target_rooms) == 1:
            return recent_target_rooms[0]
        return None

    def _build_task_frame_from_analysis(
        self,
        request: dict[str, Any],
        analysis: PlanSemanticAnalysis,
    ) -> dict[str, Any]:
        capabilities = request["capabilities"]
        intent = analysis.intent_candidate
        target_class = analysis.target_class_candidate
        attribute = analysis.attribute_candidate
        expected_value = analysis.expected_value_candidate
        room_hint = analysis.preferred_room
        instance_hint = analysis.instance_hint
        return_after_check = analysis.return_home_requested and bool(capabilities["can_return_home"])

        if analysis.clarification_reasons:
            reason = analysis.clarification_reasons[0]
            clarification_target_class = target_class
            clarification_attribute = attribute
            clarification_expected_value = expected_value
            if reason == "missing_target":
                clarification_target_class = None
                clarification_attribute = None
                clarification_expected_value = None
            return {
                "intent": "ask_clarification",
                "target": {
                    "object": clarification_target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": "attribute_check" if clarification_attribute is not None else None,
                    "attribute": clarification_attribute,
                    "operator": "equals" if clarification_attribute is not None else None,
                    "expected_value": clarification_expected_value,
                },
                "constraints": {"return_after_check": False, "report_result": True},
                "clarification": {
                    "required": True,
                    "question_ko": CLARIFICATION_MESSAGES[reason],
                },
            }

        if analysis.unsupported_reasons:
            query_attribute = attribute if target_class is not None else None
            return {
                "intent": "unsupported",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": "attribute_check" if query_attribute is not None else None,
                    "attribute": query_attribute,
                    "operator": "equals" if query_attribute is not None else None,
                    "expected_value": expected_value if query_attribute is not None else None,
                },
                "constraints": {"return_after_check": False, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        if intent == "inspect_attribute" and target_class is not None and attribute is not None:
            return {
                "intent": "check_state",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": "attribute_check",
                    "attribute": attribute,
                    "operator": "equals",
                    "expected_value": expected_value,
                },
                "constraints": {"return_after_check": return_after_check, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        if intent == "find_object" and target_class is not None:
            return {
                "intent": "find_object",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": None,
                    "attribute": None,
                    "operator": None,
                    "expected_value": None,
                },
                "constraints": {"return_after_check": return_after_check, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        if intent == "navigate_to_object" and target_class is not None:
            return {
                "intent": "navigate_to_object",
                "target": {
                    "object": target_class,
                    "instance_hint": instance_hint,
                    "location_hint": room_hint,
                },
                "query": {
                    "query_type": None,
                    "attribute": None,
                    "operator": None,
                    "expected_value": None,
                },
                "constraints": {"return_after_check": return_after_check, "report_result": True},
                "clarification": {"required": False, "question_ko": None},
            }

        query_attribute = attribute if target_class is not None else None
        return {
            "intent": "unsupported",
            "target": {
                "object": target_class,
                "instance_hint": instance_hint,
                "location_hint": room_hint,
            },
            "query": {
                "query_type": "attribute_check" if query_attribute is not None else None,
                "attribute": query_attribute,
                "operator": "equals" if query_attribute is not None else None,
                "expected_value": expected_value if query_attribute is not None else None,
            },
            "constraints": {"return_after_check": False, "report_result": True},
            "clarification": {"required": False, "question_ko": None},
        }

    def _apply_knowledge_rules(
        self,
        task_frame: dict[str, Any],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        knowledge_context = _knowledge_context_from_payload(_knowledge_context_payload(request))
        if not knowledge_context.hard_rules:
            return task_frame
        result = apply_knowledge_guards(
            task_frame,
            context=knowledge_context,
            utterance=request.get("utterance_ko"),
        )
        return result.task_frame

    def _apply_catalog_rules(
        self,
        task_frame: dict[str, Any],
    ) -> dict[str, Any]:
        intent = str(task_frame.get("intent") or "").strip()
        if intent not in EXECUTION_INTENT_KEYS or self.planner_catalog_runtime is None:
            return task_frame
        snapshot = self.planner_catalog_runtime.snapshot()
        if snapshot.intent_by_key(intent) is not None:
            return task_frame
        downgraded = dict(task_frame)
        downgraded["intent"] = "unsupported"
        downgraded["constraints"] = {"return_after_check": False, "report_result": True}
        downgraded["clarification"] = {"required": False, "question_ko": None}
        return downgraded


def _knowledge_context_payload(request: dict[str, Any]) -> dict[str, Any]:
    payload = request.get("knowledge_context")
    return payload if isinstance(payload, dict) else {}


def _knowledge_context_from_payload(payload: dict[str, Any]) -> KnowledgeContext:
    hard_rules = [_knowledge_rule_from_payload(row) for row in payload.get("hard_rules", []) if isinstance(row, dict)]
    soft_rules = [_knowledge_rule_from_payload(row) for row in payload.get("soft_rules", []) if isinstance(row, dict)]
    lexicon_entries = [
        _knowledge_lexicon_from_payload(row) for row in payload.get("lexicon_entries", []) if isinstance(row, dict)
    ]
    facts = [_knowledge_fact_from_payload(row) for row in payload.get("facts", []) if isinstance(row, dict)]
    return KnowledgeContext(
        hard_rules=hard_rules,
        soft_rules=soft_rules,
        lexicon_entries=lexicon_entries,
        facts=facts,
        debug={},
    )


def _knowledge_rule_from_payload(payload: dict[str, Any]) -> KnowledgeRule:
    from datetime import datetime

    return KnowledgeRule(
        rule_id=str(payload.get("rule_id")),
        document_id=str(payload.get("document_id")),
        rule_key=str(payload.get("rule_key")),
        scope_kind=str(payload.get("scope_kind")),
        scope_value=payload.get("scope_value"),
        enforcement=str(payload.get("enforcement")),
        action=str(payload.get("action")),
        conditions=dict(payload.get("conditions") or {}),
        params=dict(payload.get("params") or {}),
        priority=int(payload.get("priority") or 0),
        reason=payload.get("reason"),
        source_anchor=payload.get("source_anchor"),
        created_at=datetime.fromisoformat(str(payload.get("created_at"))),
        updated_at=datetime.fromisoformat(str(payload.get("updated_at"))),
        published_at=(
            None
            if payload.get("published_at") is None
            else datetime.fromisoformat(str(payload.get("published_at")))
        ),
    )


def _knowledge_lexicon_from_payload(payload: dict[str, Any]) -> KnowledgeLexiconEntry:
    from datetime import datetime

    return KnowledgeLexiconEntry(
        entry_id=str(payload.get("entry_id")),
        document_id=str(payload.get("document_id")),
        mapping_type=str(payload.get("mapping_type")),
        alias=str(payload.get("alias")),
        canonical=str(payload.get("canonical")),
        scope_kind=str(payload.get("scope_kind")),
        scope_value=payload.get("scope_value"),
        source_anchor=payload.get("source_anchor"),
        created_at=datetime.fromisoformat(str(payload.get("created_at"))),
        updated_at=datetime.fromisoformat(str(payload.get("updated_at"))),
    )


def _knowledge_fact_from_payload(payload: dict[str, Any]) -> KnowledgeFactChunk:
    from datetime import datetime

    return KnowledgeFactChunk(
        chunk_id=str(payload.get("chunk_id")),
        document_id=str(payload.get("document_id")),
        chunk_index=int(payload.get("chunk_index") or 0),
        text=str(payload.get("text") or ""),
        scope_kind=str(payload.get("scope_kind")),
        scope_value=payload.get("scope_value"),
        source_anchor=payload.get("source_anchor"),
        created_at=datetime.fromisoformat(str(payload.get("created_at"))),
        updated_at=datetime.fromisoformat(str(payload.get("updated_at"))),
        rank=float(payload["rank"]) if isinstance(payload.get("rank"), (int, float)) else None,
    )
