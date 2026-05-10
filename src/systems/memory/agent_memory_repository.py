from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Protocol

from .agent_memory_models import (
    AgentMemoryBlock,
    AgentMemoryPassage,
    normalize_agent_memory_label,
    normalize_agent_memory_scope,
    normalize_agent_memory_tags,
)

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional runtime dependency.
    psycopg = None
    dict_row = None


AGENT_MEMORY_SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "agent_memory_schema.sql"
STOP_TERMS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "for",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "where",
    }
)


class AgentMemoryRepository(Protocol):
    def apply_schema(self) -> None: ...

    def get_block(self, label: str) -> AgentMemoryBlock | None: ...

    def list_blocks(self) -> list[AgentMemoryBlock]: ...

    def upsert_block(self, block: AgentMemoryBlock) -> None: ...

    def insert_passage(self, passage: AgentMemoryPassage) -> None: ...

    def search_passages(
        self,
        query: str,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[AgentMemoryPassage]: ...

    def list_passages(
        self,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 50,
    ) -> list[AgentMemoryPassage]: ...

    def block_count(self) -> int: ...

    def passage_count(self) -> int: ...

    def unique_tags(self) -> tuple[str, ...]: ...


class InMemoryAgentMemoryRepository:
    def __init__(self) -> None:
        self._blocks: dict[str, AgentMemoryBlock] = {}
        self._passages: dict[str, AgentMemoryPassage] = {}

    def apply_schema(self) -> None:
        return None

    def get_block(self, label: str) -> AgentMemoryBlock | None:
        return self._blocks.get(normalize_agent_memory_label(label))

    def list_blocks(self) -> list[AgentMemoryBlock]:
        return list(self._blocks.values())

    def upsert_block(self, block: AgentMemoryBlock) -> None:
        label = normalize_agent_memory_label(block.label)
        self._blocks[label] = AgentMemoryBlock(
            label=label,
            description=str(block.description),
            value=str(block.value),
            limit=int(block.limit),
            read_only=bool(block.read_only),
            scope=normalize_agent_memory_scope(block.scope),
            version=int(block.version),
            updated_at=block.updated_at,
        )

    def insert_passage(self, passage: AgentMemoryPassage) -> None:
        self._passages[str(passage.passage_id)] = AgentMemoryPassage(
            passage_id=str(passage.passage_id),
            content=str(passage.content),
            tags=normalize_agent_memory_tags(tuple(passage.tags)),
            scene_scope=_normalize_scene_scope(passage.scene_scope),
            metadata=dict(passage.metadata),
            created_at=passage.created_at,
            updated_at=passage.updated_at,
            rank=passage.rank,
        )

    def search_passages(
        self,
        query: str,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[AgentMemoryPassage]:
        terms = _query_terms(query)
        rows: list[tuple[float, AgentMemoryPassage]] = []
        for passage in self._passages.values():
            if not _tag_matches(passage.tags, tags, tag_match_mode):
                continue
            if not _scene_matches(passage.scene_scope, scene_scope):
                continue
            score = _text_score(passage.content, terms)
            if terms and score <= 0:
                continue
            rows.append((score, passage))
        rows.sort(key=lambda item: (item[0], item[1].created_at.timestamp()), reverse=True)
        return [_with_rank(passage, score) for score, passage in rows[: max(0, int(top_k))]]

    def list_passages(
        self,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 50,
    ) -> list[AgentMemoryPassage]:
        rows = [
            passage
            for passage in self._passages.values()
            if _tag_matches(passage.tags, tags, tag_match_mode)
            and _scene_matches(passage.scene_scope, scene_scope)
        ]
        rows.sort(key=lambda row: row.created_at, reverse=True)
        return rows[: max(0, int(top_k))]

    def block_count(self) -> int:
        return len(self._blocks)

    def passage_count(self) -> int:
        return len(self._passages)

    def unique_tags(self) -> tuple[str, ...]:
        tags: set[str] = set()
        for passage in self._passages.values():
            tags.update(passage.tags)
        return tuple(sorted(tags))


class PostgresAgentMemoryRepository:
    def __init__(self, dsn: str, *, connect_timeout_s: float = 5.0) -> None:
        normalized_dsn = str(dsn or "").strip()
        if not normalized_dsn:
            raise RuntimeError("pg-dsn is required for PostgresAgentMemoryRepository.")
        if psycopg is None:
            raise RuntimeError("psycopg is required to use PostgresAgentMemoryRepository.")
        self.dsn = normalized_dsn
        self.connect_timeout_s = max(float(connect_timeout_s), 1.0)

    def _connect(self, *, autocommit: bool = False):
        return psycopg.connect(
            self.dsn,
            row_factory=dict_row,
            autocommit=autocommit,
            connect_timeout=int(round(self.connect_timeout_s)),
        )

    def apply_schema(self) -> None:
        schema_sql = AGENT_MEMORY_SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect(autocommit=True) as conn:
            conn.execute(schema_sql)

    def get_block(self, label: str) -> AgentMemoryBlock | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM agent_memory_blocks WHERE label = %s", (normalize_agent_memory_label(label),))
            row = cur.fetchone()
        return None if row is None else _block_from_row(row)

    def list_blocks(self) -> list[AgentMemoryBlock]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM agent_memory_blocks ORDER BY label ASC")
            rows = cur.fetchall()
        return [_block_from_row(row) for row in rows]

    def upsert_block(self, block: AgentMemoryBlock) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_memory_blocks (
                    label, description, value, limit_chars, read_only, scope, version, updated_at
                )
                VALUES (
                    %(label)s, %(description)s, %(value)s, %(limit)s, %(read_only)s,
                    %(scope)s, %(version)s, %(updated_at)s
                )
                ON CONFLICT (label) DO UPDATE SET
                    description = EXCLUDED.description,
                    value = EXCLUDED.value,
                    limit_chars = EXCLUDED.limit_chars,
                    read_only = EXCLUDED.read_only,
                    scope = EXCLUDED.scope,
                    version = EXCLUDED.version,
                    updated_at = EXCLUDED.updated_at
                """,
                {
                    "label": normalize_agent_memory_label(block.label),
                    "description": block.description,
                    "value": block.value,
                    "limit": int(block.limit),
                    "read_only": bool(block.read_only),
                    "scope": normalize_agent_memory_scope(block.scope),
                    "version": int(block.version),
                    "updated_at": block.updated_at,
                },
            )

    def insert_passage(self, passage: AgentMemoryPassage) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_memory_passages (
                    passage_id, content, tags, scene_scope, metadata, created_at, updated_at
                )
                VALUES (
                    %(passage_id)s, %(content)s, %(tags)s::text[], %(scene_scope)s,
                    %(metadata)s::jsonb, %(created_at)s, %(updated_at)s
                )
                """,
                {
                    "passage_id": passage.passage_id,
                    "content": passage.content,
                    "tags": list(normalize_agent_memory_tags(tuple(passage.tags))),
                    "scene_scope": _normalize_scene_scope(passage.scene_scope),
                    "metadata": json.dumps(passage.metadata, ensure_ascii=False),
                    "created_at": passage.created_at,
                    "updated_at": passage.updated_at,
                },
            )

    def search_passages(
        self,
        query: str,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[AgentMemoryPassage]:
        terms = _query_terms(query)
        clauses, params = _passage_filter_clauses(tags=tags, tag_match_mode=tag_match_mode, scene_scope=scene_scope)
        params["top_k"] = max(1, int(top_k))
        if terms:
            params["query"] = " | ".join(terms)
            clauses.append("search_vector @@ to_tsquery('simple', %(query)s)")
            rank_sql = "ts_rank_cd(search_vector, to_tsquery('simple', %(query)s)) AS rank"
            order_sql = "rank DESC, created_at DESC"
        else:
            rank_sql = "0.0::real AS rank"
            order_sql = "created_at DESC"
        query_sql = f"""
            SELECT *, {rank_sql}
            FROM agent_memory_passages
            {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
            ORDER BY {order_sql}
            LIMIT %(top_k)s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()
        return [_passage_from_row(row) for row in rows]

    def list_passages(
        self,
        *,
        tags: Sequence[str] | None = None,
        tag_match_mode: str = "any",
        scene_scope: str | None = None,
        top_k: int = 50,
    ) -> list[AgentMemoryPassage]:
        clauses, params = _passage_filter_clauses(tags=tags, tag_match_mode=tag_match_mode, scene_scope=scene_scope)
        params["top_k"] = max(1, int(top_k))
        query_sql = f"""
            SELECT *, NULL::real AS rank
            FROM agent_memory_passages
            {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
            ORDER BY created_at DESC
            LIMIT %(top_k)s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()
        return [_passage_from_row(row) for row in rows]

    def block_count(self) -> int:
        return self._count("SELECT COUNT(*) AS count FROM agent_memory_blocks")

    def passage_count(self) -> int:
        return self._count("SELECT COUNT(*) AS count FROM agent_memory_passages")

    def unique_tags(self) -> tuple[str, ...]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT DISTINCT unnest(tags) AS tag FROM agent_memory_passages ORDER BY tag ASC")
            rows = cur.fetchall()
        return tuple(str(row["tag"]) for row in rows if row.get("tag"))

    def _count(self, query: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
        return 0 if row is None else int(row["count"])


def _query_terms(query: str) -> list[str]:
    terms = [
        term
        for term in re.findall(r"[\w]+", str(query or "").lower(), flags=re.UNICODE)
        if term and term not in STOP_TERMS
    ]
    deduped: list[str] = []
    for term in terms:
        if term not in deduped:
            deduped.append(term)
    return deduped


def _text_score(text: str, terms: Sequence[str]) -> float:
    if not terms:
        return 0.0
    haystack = str(text or "").lower()
    return float(sum(haystack.count(term) for term in terms))


def _tag_matches(candidate_tags: Sequence[str], tags: Sequence[str] | None, tag_match_mode: str) -> bool:
    filter_tags = set(normalize_agent_memory_tags(tuple(tags or ())))
    if not filter_tags:
        return True
    candidate_set = set(normalize_agent_memory_tags(tuple(candidate_tags)))
    if str(tag_match_mode or "any").strip().lower() == "all":
        return filter_tags.issubset(candidate_set)
    return bool(filter_tags.intersection(candidate_set))


def _normalize_scene_scope(scene_scope: str | None) -> str | None:
    normalized = " ".join(str(scene_scope or "").strip().split())
    return normalized or None


def _scene_matches(candidate_scene_scope: str | None, scene_scope: str | None) -> bool:
    normalized_filter = _normalize_scene_scope(scene_scope)
    if normalized_filter is None:
        return True
    normalized_candidate = _normalize_scene_scope(candidate_scene_scope)
    return normalized_candidate is None or normalized_candidate.lower() == normalized_filter.lower()


def _with_rank(passage: AgentMemoryPassage, rank: float) -> AgentMemoryPassage:
    return AgentMemoryPassage(
        passage_id=passage.passage_id,
        content=passage.content,
        tags=tuple(passage.tags),
        scene_scope=passage.scene_scope,
        metadata=dict(passage.metadata),
        created_at=passage.created_at,
        updated_at=passage.updated_at,
        rank=float(rank),
    )


def _block_from_row(row: dict[str, object]) -> AgentMemoryBlock:
    return AgentMemoryBlock(
        label=str(row["label"]),
        description=str(row["description"]),
        value=str(row["value"]),
        limit=int(row["limit_chars"]),
        read_only=bool(row["read_only"]),
        scope=str(row["scope"]),
        version=int(row["version"]),
        updated_at=_ensure_datetime(row["updated_at"]),
    )


def _passage_from_row(row: dict[str, object]) -> AgentMemoryPassage:
    raw_tags = row.get("tags") or ()
    tags = tuple(str(tag) for tag in raw_tags)
    metadata = row.get("metadata") or {}
    rank = row.get("rank")
    return AgentMemoryPassage(
        passage_id=str(row["passage_id"]),
        content=str(row["content"]),
        tags=normalize_agent_memory_tags(tags),
        scene_scope=_normalize_scene_scope(row.get("scene_scope")),
        metadata=metadata if isinstance(metadata, dict) else {},
        created_at=_ensure_datetime(row["created_at"]),
        updated_at=_ensure_datetime(row["updated_at"]),
        rank=float(rank) if isinstance(rank, (int, float)) else None,
    )


def _ensure_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    raise TypeError(f"expected datetime value, got {type(value).__name__}")


def _passage_filter_clauses(
    *,
    tags: Sequence[str] | None,
    tag_match_mode: str,
    scene_scope: str | None,
) -> tuple[list[str], dict[str, object]]:
    clauses: list[str] = []
    params: dict[str, object] = {}
    normalized_tags = normalize_agent_memory_tags(tuple(tags or ()))
    if normalized_tags:
        params["tags"] = list(normalized_tags)
        if str(tag_match_mode or "any").strip().lower() == "all":
            clauses.append("tags @> %(tags)s::text[]")
        else:
            clauses.append("tags && %(tags)s::text[]")
    normalized_scene_scope = _normalize_scene_scope(scene_scope)
    if normalized_scene_scope is not None:
        clauses.append("(scene_scope IS NULL OR lower(scene_scope) = lower(%(scene_scope)s))")
        params["scene_scope"] = normalized_scene_scope
    return clauses, params
