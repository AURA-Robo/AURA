from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Protocol

from .conversation_memory_models import ConversationContext, ConversationTurn

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional runtime dependency.
    psycopg = None
    dict_row = None


class ConversationMemoryRepository(Protocol):
    def apply_schema(self) -> None: ...

    def load_context(self, conversation_id: str, max_turns: int) -> ConversationContext: ...

    def append_turn(self, turn: ConversationTurn) -> None: ...

    def update_summary(self, conversation_id: str, summary: str, resolved_slots: dict[str, str]) -> None: ...

    def conversation_count(self) -> int: ...

    def turn_count(self) -> int: ...


class InMemoryConversationMemoryRepository:
    def __init__(self) -> None:
        self._turns: dict[str, list[ConversationTurn]] = defaultdict(list)
        self._summaries: dict[str, tuple[str, dict[str, str]]] = {}

    def apply_schema(self) -> None:
        return None

    def load_context(self, conversation_id: str, max_turns: int) -> ConversationContext:
        normalized_id = str(conversation_id).strip()
        summary, resolved_slots = self._summaries.get(normalized_id, ("", {}))
        recent_turns = list(self._turns.get(normalized_id, []))
        if max_turns > 0:
            recent_turns = recent_turns[-max_turns:]
        return ConversationContext(
            conversation_id=normalized_id,
            summary=summary,
            resolved_slots=dict(resolved_slots),
            recent_turns=recent_turns,
        )

    def append_turn(self, turn: ConversationTurn) -> None:
        self._turns[turn.conversation_id].append(turn)

    def update_summary(self, conversation_id: str, summary: str, resolved_slots: dict[str, str]) -> None:
        self._summaries[str(conversation_id).strip()] = (str(summary), dict(resolved_slots))

    def conversation_count(self) -> int:
        return len(set(self._turns) | set(self._summaries))

    def turn_count(self) -> int:
        return sum(len(turns) for turns in self._turns.values())


class PostgresConversationMemoryRepository:
    def __init__(self, dsn: str) -> None:
        normalized_dsn = str(dsn).strip()
        if not normalized_dsn:
            raise RuntimeError("pg-dsn is required for PostgresConversationMemoryRepository.")
        if psycopg is None:
            raise RuntimeError("psycopg is required to use PostgresConversationMemoryRepository.")
        self.dsn = normalized_dsn

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def apply_schema(self) -> None:
        schema_path = Path(__file__).resolve().parent / "sql" / "conversation_memory_schema.sql"
        schema_sql = schema_path.read_text(encoding="utf-8")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(schema_sql)

    def load_context(self, conversation_id: str, max_turns: int) -> ConversationContext:
        normalized_id = str(conversation_id).strip()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT summary, resolved_slots
                FROM conversation_summaries
                WHERE conversation_id = %(conversation_id)s
                """,
                {"conversation_id": normalized_id},
            )
            summary_row = cur.fetchone()
            cur.execute(
                """
                SELECT turn_id, conversation_id, role, text, metadata, created_at
                FROM conversation_turns
                WHERE conversation_id = %(conversation_id)s
                ORDER BY created_at DESC
                LIMIT %(limit)s
                """,
                {"conversation_id": normalized_id, "limit": max(0, int(max_turns))},
            )
            turn_rows = list(reversed(cur.fetchall()))
        summary = ""
        resolved_slots: dict[str, str] = {}
        if isinstance(summary_row, dict):
            summary = str(summary_row.get("summary") or "")
            raw_slots = summary_row.get("resolved_slots")
            if isinstance(raw_slots, dict):
                resolved_slots = {str(key): str(value) for key, value in raw_slots.items()}
        recent_turns = [
            ConversationTurn(
                turn_id=str(row["turn_id"]),
                conversation_id=str(row["conversation_id"]),
                role=str(row["role"]),
                text=str(row["text"]),
                metadata=_metadata_dict(row.get("metadata")),
                created_at=row["created_at"],
            )
            for row in turn_rows
        ]
        return ConversationContext(
            conversation_id=normalized_id,
            summary=summary,
            resolved_slots=resolved_slots,
            recent_turns=recent_turns,
        )

    def append_turn(self, turn: ConversationTurn) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_turns (
                    turn_id,
                    conversation_id,
                    role,
                    text,
                    metadata,
                    created_at
                )
                VALUES (
                    %(turn_id)s,
                    %(conversation_id)s,
                    %(role)s,
                    %(text)s,
                    %(metadata)s::jsonb,
                    %(created_at)s
                )
                """,
                {
                    "turn_id": turn.turn_id,
                    "conversation_id": turn.conversation_id,
                    "role": turn.role,
                    "text": turn.text,
                    "metadata": json.dumps(turn.metadata, ensure_ascii=False),
                    "created_at": turn.created_at,
                },
            )

    def update_summary(self, conversation_id: str, summary: str, resolved_slots: dict[str, str]) -> None:
        normalized_id = str(conversation_id).strip()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_summaries (
                    conversation_id,
                    summary,
                    resolved_slots,
                    updated_at
                )
                VALUES (
                    %(conversation_id)s,
                    %(summary)s,
                    %(resolved_slots)s::jsonb,
                    NOW()
                )
                ON CONFLICT (conversation_id)
                DO UPDATE SET
                    summary = EXCLUDED.summary,
                    resolved_slots = EXCLUDED.resolved_slots,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "conversation_id": normalized_id,
                    "summary": str(summary),
                    "resolved_slots": json.dumps(dict(resolved_slots), ensure_ascii=False),
                },
            )

    def conversation_count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM conversation_summaries")
            row = cur.fetchone()
        return int((row or {}).get("count") or 0)

    def turn_count(self) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS count FROM conversation_turns")
            row = cur.fetchone()
        return int((row or {}).get("count") or 0)


def _metadata_dict(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items()}
    return {}
