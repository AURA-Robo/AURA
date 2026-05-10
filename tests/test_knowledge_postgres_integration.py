from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from systems.memory.knowledge_models import KnowledgeDocumentInput
from systems.memory.knowledge_runtime import create_knowledge_runtime
from systems.memory.knowledge_service import KnowledgeService
from systems.memory.knowledge_repository import PostgresKnowledgeRepository


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "systems"
    / "memory"
    / "sql"
    / "knowledge_schema.sql"
)


def _admin_conninfo() -> str:
    return os.environ.get(
        "AURA_TEST_POSTGRES_ADMIN_DSN",
        "host=127.0.0.1 dbname=postgres user=postgres",
    )


@pytest.fixture()
def postgres_knowledge_dsn() -> str:
    admin_conninfo = _admin_conninfo()
    db_name = f"aura_knowledge_test_{uuid.uuid4().hex[:10]}"
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")

    try:
        with psycopg.connect(admin_conninfo, autocommit=True) as admin_conn:
            admin_conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
    except psycopg.Error as exc:
        pytest.skip(f"local PostgreSQL admin connection unavailable: {exc}")

    options = conninfo_to_dict(admin_conninfo)
    test_conninfo = make_conninfo(**{**options, "dbname": db_name})

    try:
        with psycopg.connect(test_conninfo, autocommit=True) as test_conn:
            test_conn.execute(schema_sql)
        yield test_conninfo
    finally:
        with psycopg.connect(admin_conninfo, autocommit=True) as admin_conn:
            admin_conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            admin_conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(db_name)))


def test_postgres_knowledge_round_trip_with_real_database(postgres_knowledge_dsn: str) -> None:
    repository = PostgresKnowledgeRepository(postgres_knowledge_dsn)
    service = KnowledgeService(repository)
    document = service.register_document(
        KnowledgeDocumentInput(
            title="Kitchen facts",
            scope_kind="scene",
            scope_value="warehouse",
            body_markdown="""
# Kitchen
The refrigerator is kept in the kitchen aisle.

```knowledge-rule
{
  "action": "force_target_room",
  "enforcement": "hard",
  "conditions": {
    "target_object": "refrigerator"
  },
  "room": "kitchen"
}
```

```knowledge-lexicon
{
  "object_alias": "fridge",
  "canonical_object": "refrigerator"
}
```
""",
            publish=True,
        )
    )

    assert document.status == "published"

    listed = repository.list_documents()
    assert len(listed) == 1
    assert listed[0].document_id == document.document_id

    context = service.retrieve_for_plan("go to the fridge", scene_scope="warehouse", top_k=3)
    assert len(context.hard_rules) == 1
    assert context.hard_rules[0].action == "force_target_room"
    assert len(context.lexicon_entries) == 1
    assert context.lexicon_entries[0].canonical == "refrigerator"
    assert len(context.facts) == 1
    assert context.facts[0].source_anchor == "kitchen"


def test_knowledge_runtime_uses_real_postgres(postgres_knowledge_dsn: str) -> None:
    runtime = create_knowledge_runtime(dsn=postgres_knowledge_dsn, scene_scope="warehouse")

    assert runtime.available is True
    assert runtime.retrieve_for_plan("go to the fridge").hard_rules == []

    assert runtime.service is not None
    runtime.service.register_document(
        KnowledgeDocumentInput(
            title="No refrigerator approach",
            body_markdown="""
```knowledge-rule
{
  "action": "deny_task",
  "enforcement": "hard",
  "conditions": {
    "intent": "navigate_to_object",
    "target_object": "refrigerator"
  },
  "reason": "Do not navigate directly to the refrigerator."
}
```
""",
            publish=True,
        )
    )

    context = runtime.retrieve_for_plan("go to the refrigerator", scene_scope="warehouse")
    assert len(context.hard_rules) == 1
    status = runtime.status_snapshot()
    assert status.published_document_count == 1
    assert status.active_hard_rule_count == 1
