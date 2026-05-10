from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import re
from typing import Any, Protocol

from .knowledge_models import (
    KnowledgeDocumentRecord,
    KnowledgeFactChunk,
    KnowledgeLexiconEntry,
    KnowledgeRule,
    KnowledgeRuleAuditRecord,
)

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional runtime dependency.
    psycopg = None
    dict_row = None


SCHEMA_PATH = Path(__file__).resolve().parent / "sql" / "knowledge_schema.sql"


class KnowledgeRepository(Protocol):
    def upsert_document_bundle(
        self,
        document: KnowledgeDocumentRecord,
        *,
        rules: Sequence[KnowledgeRule],
        lexicon_entries: Sequence[KnowledgeLexiconEntry],
        chunks: Sequence[KnowledgeFactChunk],
    ) -> None: ...

    def get_document(self, document_id: str) -> KnowledgeDocumentRecord | None: ...

    def list_documents(
        self,
        *,
        statuses: Sequence[str] | None = None,
    ) -> list[KnowledgeDocumentRecord]: ...

    def set_document_status(
        self,
        document_id: str,
        *,
        status: str,
        updated_at: datetime,
        published_at: datetime | None = None,
    ) -> KnowledgeDocumentRecord | None: ...

    def list_published_rules(
        self,
        *,
        scene_scope: str | None = None,
        enforcement: str | None = None,
    ) -> list[KnowledgeRule]: ...

    def list_published_lexicon_entries(
        self,
        *,
        scene_scope: str | None = None,
    ) -> list[KnowledgeLexiconEntry]: ...

    def search_published_chunks(
        self,
        query: str,
        *,
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeFactChunk]: ...

    def insert_rule_audit(self, record: KnowledgeRuleAuditRecord) -> None: ...

    def published_document_count(self) -> int: ...

    def active_hard_rule_count(self) -> int: ...

    def lexicon_entry_count(self) -> int: ...


class InMemoryKnowledgeRepository:
    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeDocumentRecord] = {}
        self._rules: dict[str, KnowledgeRule] = {}
        self._lexicon_entries: dict[str, KnowledgeLexiconEntry] = {}
        self._chunks: dict[str, KnowledgeFactChunk] = {}
        self._audits: dict[str, KnowledgeRuleAuditRecord] = {}

    def upsert_document_bundle(
        self,
        document: KnowledgeDocumentRecord,
        *,
        rules: Sequence[KnowledgeRule],
        lexicon_entries: Sequence[KnowledgeLexiconEntry],
        chunks: Sequence[KnowledgeFactChunk],
    ) -> None:
        self._documents[document.document_id] = document
        self._rules = {key: row for key, row in self._rules.items() if row.document_id != document.document_id}
        self._lexicon_entries = {
            key: row for key, row in self._lexicon_entries.items() if row.document_id != document.document_id
        }
        self._chunks = {key: row for key, row in self._chunks.items() if row.document_id != document.document_id}
        for row in rules:
            self._rules[row.rule_id] = row
        for row in lexicon_entries:
            self._lexicon_entries[row.entry_id] = row
        for row in chunks:
            self._chunks[row.chunk_id] = row

    def get_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        return self._documents.get(document_id)

    def list_documents(
        self,
        *,
        statuses: Sequence[str] | None = None,
    ) -> list[KnowledgeDocumentRecord]:
        status_set = set(statuses) if statuses is not None else None
        rows = [
            row
            for row in self._documents.values()
            if status_set is None or row.status in status_set
        ]
        rows.sort(key=lambda row: row.updated_at, reverse=True)
        return rows

    def set_document_status(
        self,
        document_id: str,
        *,
        status: str,
        updated_at: datetime,
        published_at: datetime | None = None,
    ) -> KnowledgeDocumentRecord | None:
        current = self._documents.get(document_id)
        if current is None:
            return None
        next_record = KnowledgeDocumentRecord(
            document_id=current.document_id,
            title=current.title,
            body_markdown=current.body_markdown,
            scope_kind=current.scope_kind,
            scope_value=current.scope_value,
            status=status,
            content_hash=current.content_hash,
            version=current.version,
            metadata=dict(current.metadata),
            created_at=current.created_at,
            updated_at=updated_at,
            published_at=published_at if status == "published" else current.published_at if status == current.status else None,
        )
        self._documents[document_id] = next_record
        for key, row in list(self._rules.items()):
            if row.document_id != document_id:
                continue
            self._rules[key] = KnowledgeRule(
                rule_id=row.rule_id,
                document_id=row.document_id,
                rule_key=row.rule_key,
                scope_kind=row.scope_kind,
                scope_value=row.scope_value,
                enforcement=row.enforcement,
                action=row.action,
                conditions=dict(row.conditions),
                params=dict(row.params),
                priority=row.priority,
                reason=row.reason,
                source_anchor=row.source_anchor,
                created_at=row.created_at,
                updated_at=updated_at,
                published_at=published_at if status == "published" else None,
            )
        return next_record

    def list_published_rules(
        self,
        *,
        scene_scope: str | None = None,
        enforcement: str | None = None,
    ) -> list[KnowledgeRule]:
        rows: list[KnowledgeRule] = []
        for row in self._rules.values():
            document = self._documents.get(row.document_id)
            if document is None or document.status != "published":
                continue
            if enforcement is not None and row.enforcement != enforcement:
                continue
            if not _scope_matches(row.scope_kind, row.scope_value, scene_scope):
                continue
            rows.append(row)
        rows.sort(key=lambda row: (row.priority, row.updated_at.timestamp()), reverse=True)
        return rows

    def list_published_lexicon_entries(
        self,
        *,
        scene_scope: str | None = None,
    ) -> list[KnowledgeLexiconEntry]:
        rows: list[KnowledgeLexiconEntry] = []
        for row in self._lexicon_entries.values():
            document = self._documents.get(row.document_id)
            if document is None or document.status != "published":
                continue
            if not _scope_matches(row.scope_kind, row.scope_value, scene_scope):
                continue
            rows.append(row)
        rows.sort(key=lambda row: row.updated_at, reverse=True)
        return rows

    def search_published_chunks(
        self,
        query: str,
        *,
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeFactChunk]:
        normalized_terms = [term for term in str(query).strip().lower().split() if term]
        rows: list[tuple[int, KnowledgeFactChunk]] = []
        for row in self._chunks.values():
            document = self._documents.get(row.document_id)
            if document is None or document.status != "published":
                continue
            if not _scope_matches(row.scope_kind, row.scope_value, scene_scope):
                continue
            haystack = row.text.lower()
            score = sum(haystack.count(term) for term in normalized_terms)
            if normalized_terms and score <= 0:
                continue
            rows.append((score, row))
        rows.sort(key=lambda item: (item[0], item[1].chunk_index * -1, item[1].updated_at.timestamp()), reverse=True)
        return [
            KnowledgeFactChunk(
                chunk_id=row.chunk_id,
                document_id=row.document_id,
                chunk_index=row.chunk_index,
                text=row.text,
                scope_kind=row.scope_kind,
                scope_value=row.scope_value,
                source_anchor=row.source_anchor,
                created_at=row.created_at,
                updated_at=row.updated_at,
                rank=float(score),
            )
            for score, row in rows[: max(0, int(top_k))]
        ]

    def insert_rule_audit(self, record: KnowledgeRuleAuditRecord) -> None:
        self._audits[record.audit_id] = record

    def published_document_count(self) -> int:
        return len([row for row in self._documents.values() if row.status == "published"])

    def active_hard_rule_count(self) -> int:
        return len(self.list_published_rules(scene_scope="__all__", enforcement="hard"))

    def lexicon_entry_count(self) -> int:
        return len(self.list_published_lexicon_entries(scene_scope="__all__"))


class PostgresKnowledgeRepository:
    def __init__(self, dsn: str) -> None:
        normalized_dsn = str(dsn).strip()
        if not normalized_dsn:
            raise RuntimeError("pg-dsn is required for PostgresKnowledgeRepository.")
        if psycopg is None:
            raise RuntimeError("psycopg is required to use PostgresKnowledgeRepository.")
        self.dsn = normalized_dsn

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def apply_schema(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(schema_sql)

    def upsert_document_bundle(
        self,
        document: KnowledgeDocumentRecord,
        *,
        rules: Sequence[KnowledgeRule],
        lexicon_entries: Sequence[KnowledgeLexiconEntry],
        chunks: Sequence[KnowledgeFactChunk],
    ) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_documents (
                    document_id, title, scope_kind, scope_value, status,
                    body_markdown, content_hash, version, metadata,
                    created_at, updated_at, published_at
                )
                VALUES (
                    %(document_id)s, %(title)s, %(scope_kind)s, %(scope_value)s, %(status)s,
                    %(body_markdown)s, %(content_hash)s, %(version)s, %(metadata)s::jsonb,
                    %(created_at)s, %(updated_at)s, %(published_at)s
                )
                ON CONFLICT (document_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    scope_kind = EXCLUDED.scope_kind,
                    scope_value = EXCLUDED.scope_value,
                    status = EXCLUDED.status,
                    body_markdown = EXCLUDED.body_markdown,
                    content_hash = EXCLUDED.content_hash,
                    version = EXCLUDED.version,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at,
                    published_at = EXCLUDED.published_at
                """,
                {
                    **asdict(document),
                    "metadata": json.dumps(document.metadata, ensure_ascii=False),
                },
            )
            cur.execute("DELETE FROM knowledge_rules WHERE document_id = %s", (document.document_id,))
            cur.execute("DELETE FROM knowledge_lexicon_entries WHERE document_id = %s", (document.document_id,))
            cur.execute("DELETE FROM knowledge_chunks WHERE document_id = %s", (document.document_id,))

            for row in rules:
                cur.execute(
                    """
                    INSERT INTO knowledge_rules (
                        rule_id, document_id, rule_key, scope_kind, scope_value,
                        enforcement, action, conditions, params, priority,
                        reason, source_anchor, created_at, updated_at, published_at
                    )
                    VALUES (
                        %(rule_id)s, %(document_id)s, %(rule_key)s, %(scope_kind)s, %(scope_value)s,
                        %(enforcement)s, %(action)s, %(conditions)s::jsonb, %(params)s::jsonb, %(priority)s,
                        %(reason)s, %(source_anchor)s, %(created_at)s, %(updated_at)s, %(published_at)s
                    )
                    """,
                    {
                        **asdict(row),
                        "conditions": json.dumps(row.conditions, ensure_ascii=False),
                        "params": json.dumps(row.params, ensure_ascii=False),
                    },
                )

            for row in lexicon_entries:
                cur.execute(
                    """
                    INSERT INTO knowledge_lexicon_entries (
                        entry_id, document_id, mapping_type, alias, canonical,
                        scope_kind, scope_value, source_anchor, created_at, updated_at
                    )
                    VALUES (
                        %(entry_id)s, %(document_id)s, %(mapping_type)s, %(alias)s, %(canonical)s,
                        %(scope_kind)s, %(scope_value)s, %(source_anchor)s, %(created_at)s, %(updated_at)s
                    )
                    """,
                    asdict(row),
                )

            for row in chunks:
                cur.execute(
                    """
                    INSERT INTO knowledge_chunks (
                        chunk_id, document_id, chunk_index, text,
                        scope_kind, scope_value, source_anchor, created_at, updated_at
                    )
                    VALUES (
                        %(chunk_id)s, %(document_id)s, %(chunk_index)s, %(text)s,
                        %(scope_kind)s, %(scope_value)s, %(source_anchor)s, %(created_at)s, %(updated_at)s
                    )
                    """,
                    asdict(row),
                )

    def get_document(self, document_id: str) -> KnowledgeDocumentRecord | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT * FROM knowledge_documents WHERE document_id = %s", (document_id,))
            row = cur.fetchone()
        return None if row is None else self._document_from_row(row)

    def list_documents(
        self,
        *,
        statuses: Sequence[str] | None = None,
    ) -> list[KnowledgeDocumentRecord]:
        clauses: list[str] = []
        params: dict[str, object] = {}
        if statuses is not None:
            clauses.append("status = ANY(%(statuses)s)")
            params["statuses"] = list(statuses)
        query = f"""
            SELECT *
            FROM knowledge_documents
            {'WHERE ' + ' AND '.join(clauses) if clauses else ''}
            ORDER BY updated_at DESC
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [self._document_from_row(row) for row in rows]

    def set_document_status(
        self,
        document_id: str,
        *,
        status: str,
        updated_at: datetime,
        published_at: datetime | None = None,
    ) -> KnowledgeDocumentRecord | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE knowledge_documents
                SET status = %(status)s,
                    updated_at = %(updated_at)s,
                    published_at = %(published_at)s
                WHERE document_id = %(document_id)s
                RETURNING *
                """,
                {
                    "document_id": document_id,
                    "status": status,
                    "updated_at": updated_at,
                    "published_at": published_at if status == "published" else None,
                },
            )
            row = cur.fetchone()
            if row is not None:
                cur.execute(
                    """
                    UPDATE knowledge_rules
                    SET published_at = %(published_at)s,
                        updated_at = %(updated_at)s
                    WHERE document_id = %(document_id)s
                    """,
                    {
                        "document_id": document_id,
                        "published_at": published_at if status == "published" else None,
                        "updated_at": updated_at,
                    },
                )
        return None if row is None else self._document_from_row(row)

    def list_published_rules(
        self,
        *,
        scene_scope: str | None = None,
        enforcement: str | None = None,
    ) -> list[KnowledgeRule]:
        clauses, params = _published_scope_clauses(scene_scope)
        if enforcement is not None:
            clauses.append("r.enforcement = %(enforcement)s")
            params["enforcement"] = enforcement
        query = f"""
            SELECT r.*
            FROM knowledge_rules r
            JOIN knowledge_documents d ON d.document_id = r.document_id
            WHERE d.status = 'published'
              AND {' AND '.join(clauses)}
            ORDER BY r.priority DESC, COALESCE(d.published_at, d.updated_at) DESC, r.updated_at DESC
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [self._rule_from_row(row) for row in rows]

    def list_published_lexicon_entries(
        self,
        *,
        scene_scope: str | None = None,
    ) -> list[KnowledgeLexiconEntry]:
        clauses, params = _published_scope_clauses(scene_scope, table_alias="l")
        query = f"""
            SELECT l.*
            FROM knowledge_lexicon_entries l
            JOIN knowledge_documents d ON d.document_id = l.document_id
            WHERE d.status = 'published'
              AND {' AND '.join(clauses)}
            ORDER BY COALESCE(d.published_at, d.updated_at) DESC, l.updated_at DESC
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        return [self._lexicon_from_row(row) for row in rows]

    def search_published_chunks(
        self,
        query: str,
        *,
        scene_scope: str | None = None,
        top_k: int = 5,
    ) -> list[KnowledgeFactChunk]:
        normalized_query = " ".join(str(query).strip().split())
        clauses, params = _published_scope_clauses(scene_scope, table_alias="c")
        params["top_k"] = max(1, int(top_k))
        search_terms = [term for term in re.findall(r"[a-z0-9_]+", normalized_query.lower()) if term]
        if search_terms:
            params["query"] = " | ".join(search_terms)
            clauses.append("c.search_vector @@ to_tsquery('simple', %(query)s)")
            rank_sql = "ts_rank_cd(c.search_vector, to_tsquery('simple', %(query)s)) AS rank"
            order_sql = "rank DESC, COALESCE(d.published_at, d.updated_at) DESC, c.chunk_index ASC"
        else:
            rank_sql = "0.0::real AS rank"
            order_sql = "COALESCE(d.published_at, d.updated_at) DESC, c.chunk_index ASC"
        sql_text = f"""
            SELECT c.*, {rank_sql}
            FROM knowledge_chunks c
            JOIN knowledge_documents d ON d.document_id = c.document_id
            WHERE d.status = 'published'
              AND {' AND '.join(clauses)}
            ORDER BY {order_sql}
            LIMIT %(top_k)s
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql_text, params)
            rows = cur.fetchall()
        return [self._chunk_from_row(row) for row in rows]

    def insert_rule_audit(self, record: KnowledgeRuleAuditRecord) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_rule_audit (
                    audit_id, rule_id, document_id, phase, task_id,
                    subgoal_id, applied_at, payload
                )
                VALUES (
                    %(audit_id)s, %(rule_id)s, %(document_id)s, %(phase)s, %(task_id)s,
                    %(subgoal_id)s, %(applied_at)s, %(payload)s::jsonb
                )
                """,
                {
                    **asdict(record),
                    "payload": json.dumps(record.payload, ensure_ascii=False),
                },
            )

    def published_document_count(self) -> int:
        return self._count(
            """
            SELECT COUNT(*) AS count
            FROM knowledge_documents
            WHERE status = 'published'
            """
        )

    def active_hard_rule_count(self) -> int:
        return self._count(
            """
            SELECT COUNT(*) AS count
            FROM knowledge_rules r
            JOIN knowledge_documents d ON d.document_id = r.document_id
            WHERE d.status = 'published'
              AND r.enforcement = 'hard'
            """
        )

    def lexicon_entry_count(self) -> int:
        return self._count(
            """
            SELECT COUNT(*) AS count
            FROM knowledge_lexicon_entries l
            JOIN knowledge_documents d ON d.document_id = l.document_id
            WHERE d.status = 'published'
            """
        )

    def _count(self, query: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query)
            row = cur.fetchone()
        return 0 if row is None else int(row["count"])

    def _document_from_row(self, row: dict[str, object]) -> KnowledgeDocumentRecord:
        return KnowledgeDocumentRecord(
            document_id=str(row["document_id"]),
            title=str(row["title"]),
            body_markdown=str(row["body_markdown"]),
            scope_kind=str(row["scope_kind"]),
            scope_value=row.get("scope_value"),
            status=str(row["status"]),
            content_hash=str(row["content_hash"]),
            version=int(row["version"]),
            metadata=row.get("metadata") or {},
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            published_at=row.get("published_at"),
        )

    def _rule_from_row(self, row: dict[str, object]) -> KnowledgeRule:
        return KnowledgeRule(
            rule_id=str(row["rule_id"]),
            document_id=str(row["document_id"]),
            rule_key=str(row["rule_key"]),
            scope_kind=str(row["scope_kind"]),
            scope_value=row.get("scope_value"),
            enforcement=str(row["enforcement"]),
            action=str(row["action"]),
            conditions=row.get("conditions") or {},
            params=row.get("params") or {},
            priority=int(row.get("priority") or 0),
            reason=row.get("reason"),
            source_anchor=row.get("source_anchor"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            published_at=row.get("published_at"),
        )

    def _lexicon_from_row(self, row: dict[str, object]) -> KnowledgeLexiconEntry:
        return KnowledgeLexiconEntry(
            entry_id=str(row["entry_id"]),
            document_id=str(row["document_id"]),
            mapping_type=str(row["mapping_type"]),
            alias=str(row["alias"]),
            canonical=str(row["canonical"]),
            scope_kind=str(row["scope_kind"]),
            scope_value=row.get("scope_value"),
            source_anchor=row.get("source_anchor"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _chunk_from_row(self, row: dict[str, object]) -> KnowledgeFactChunk:
        rank = row.get("rank")
        return KnowledgeFactChunk(
            chunk_id=str(row["chunk_id"]),
            document_id=str(row["document_id"]),
            chunk_index=int(row["chunk_index"]),
            text=str(row["text"]),
            scope_kind=str(row["scope_kind"]),
            scope_value=row.get("scope_value"),
            source_anchor=row.get("source_anchor"),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            rank=float(rank) if isinstance(rank, (int, float)) else None,
        )


def _scope_matches(scope_kind: str, scope_value: str | None, scene_scope: str | None) -> bool:
    if str(scene_scope or "").strip() in {"*", "__all__"}:
        return True
    if scope_kind == "global":
        return True
    if scope_kind != "scene":
        return False
    return scene_scope is not None and str(scope_value or "").strip().lower() == str(scene_scope).strip().lower()


def _published_scope_clauses(scene_scope: str | None, *, table_alias: str = "r") -> tuple[list[str], dict[str, object]]:
    normalized_scene = str(scene_scope or "").strip()
    if normalized_scene in {"*", "__all__"}:
        return ["TRUE"], {}
    clauses: list[str] = [f"{table_alias}.scope_kind = 'global'"]
    params: dict[str, object] = {}
    if normalized_scene:
        clauses.append(f"({table_alias}.scope_kind = 'scene' AND {table_alias}.scope_value = %(scene_scope)s)")
        params["scene_scope"] = normalized_scene
    return [f"({' OR '.join(clauses)})"], params
