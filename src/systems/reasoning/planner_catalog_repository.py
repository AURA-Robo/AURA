from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Protocol

from systems.reasoning.planner_catalog_errors import PlannerCatalogConflictError
from systems.reasoning.planner_catalog_models import PlannerIntentRecord, PlannerSubgoalTemplateRecord

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional runtime dependency.
    psycopg = None
    dict_row = None


SCHEMA_PATH = Path(__file__).resolve().with_name("planner_catalog_schema.sql")


class PlannerCatalogRepository(Protocol):
    def apply_schema(self) -> None: ...

    def count_intents(self, *, include_deleted: bool = False) -> int: ...

    def list_active_intents(self) -> list[PlannerIntentRecord]: ...

    def list_active_subgoal_templates(self) -> list[PlannerSubgoalTemplateRecord]: ...

    def create_intent_with_templates(
        self,
        intent: PlannerIntentRecord,
        templates: Sequence[PlannerSubgoalTemplateRecord],
    ) -> None: ...

    def create_subgoal_template(self, template: PlannerSubgoalTemplateRecord) -> None: ...

    def soft_delete_intent(self, intent_id: str, *, deleted_at: datetime) -> bool: ...

    def soft_delete_subgoal_template(self, template_id: str, *, deleted_at: datetime) -> bool: ...


class InMemoryPlannerCatalogRepository:
    def __init__(self) -> None:
        self._intents: dict[str, PlannerIntentRecord] = {}
        self._templates: dict[str, PlannerSubgoalTemplateRecord] = {}

    def apply_schema(self) -> None:
        return None

    def count_intents(self, *, include_deleted: bool = False) -> int:
        if include_deleted:
            return len(self._intents)
        return len([row for row in self._intents.values() if row.deleted_at is None])

    def list_active_intents(self) -> list[PlannerIntentRecord]:
        rows = [row for row in self._intents.values() if row.deleted_at is None]
        rows.sort(key=lambda row: (row.intent_key, row.created_at))
        return rows

    def list_active_subgoal_templates(self) -> list[PlannerSubgoalTemplateRecord]:
        rows = [row for row in self._templates.values() if row.deleted_at is None]
        rows.sort(key=lambda row: (row.intent_id, row.sequence_no, row.template_id))
        return rows

    def create_intent_with_templates(
        self,
        intent: PlannerIntentRecord,
        templates: Sequence[PlannerSubgoalTemplateRecord],
    ) -> None:
        if any(row.deleted_at is None and row.intent_key == intent.intent_key for row in self._intents.values()):
            raise PlannerCatalogConflictError(f"planner intent already active: {intent.intent_key}")
        self._intents[intent.intent_id] = intent
        for template in templates:
            active_conflict = any(
                row.deleted_at is None
                and row.intent_id == template.intent_id
                and row.sequence_no == template.sequence_no
                for row in self._templates.values()
            )
            if active_conflict:
                raise PlannerCatalogConflictError(
                    f"planner subgoal sequence already active for intent {template.intent_id}: {template.sequence_no}"
                )
            self._templates[template.template_id] = template

    def create_subgoal_template(self, template: PlannerSubgoalTemplateRecord) -> None:
        active_conflict = any(
            row.deleted_at is None
            and row.intent_id == template.intent_id
            and row.sequence_no == template.sequence_no
            for row in self._templates.values()
        )
        if active_conflict:
            raise PlannerCatalogConflictError(
                f"planner subgoal sequence already active for intent {template.intent_id}: {template.sequence_no}"
            )
        self._templates[template.template_id] = template

    def soft_delete_intent(self, intent_id: str, *, deleted_at: datetime) -> bool:
        current = self._intents.get(intent_id)
        if current is None or current.deleted_at is not None:
            return False
        self._intents[intent_id] = PlannerIntentRecord(
            intent_id=current.intent_id,
            intent_key=current.intent_key,
            display_name=current.display_name,
            description=current.description,
            created_at=current.created_at,
            updated_at=deleted_at,
            deleted_at=deleted_at,
        )
        for template_id, template in list(self._templates.items()):
            if template.intent_id != intent_id or template.deleted_at is not None:
                continue
            self._templates[template_id] = PlannerSubgoalTemplateRecord(
                template_id=template.template_id,
                intent_id=template.intent_id,
                sequence_no=template.sequence_no,
                subgoal_type=template.subgoal_type,
                activation_condition=template.activation_condition,
                created_at=template.created_at,
                updated_at=deleted_at,
                deleted_at=deleted_at,
            )
        return True

    def soft_delete_subgoal_template(self, template_id: str, *, deleted_at: datetime) -> bool:
        current = self._templates.get(template_id)
        if current is None or current.deleted_at is not None:
            return False
        self._templates[template_id] = PlannerSubgoalTemplateRecord(
            template_id=current.template_id,
            intent_id=current.intent_id,
            sequence_no=current.sequence_no,
            subgoal_type=current.subgoal_type,
            activation_condition=current.activation_condition,
            created_at=current.created_at,
            updated_at=deleted_at,
            deleted_at=deleted_at,
        )
        return True


class PostgresPlannerCatalogRepository:
    def __init__(self, dsn: str) -> None:
        normalized_dsn = str(dsn).strip()
        if not normalized_dsn:
            raise RuntimeError("planner catalog dsn is required for PostgresPlannerCatalogRepository")
        if psycopg is None:
            raise RuntimeError("psycopg is required to use PostgresPlannerCatalogRepository")
        self.dsn = normalized_dsn

    def _connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def apply_schema(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(schema_sql)

    def count_intents(self, *, include_deleted: bool = False) -> int:
        query = "SELECT COUNT(*) AS count FROM planner_intents"
        params: tuple[object, ...] = ()
        if not include_deleted:
            query += " WHERE deleted_at IS NULL"
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(query, params)
            row = cur.fetchone()
        return 0 if row is None else int(row["count"])

    def list_active_intents(self) -> list[PlannerIntentRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM planner_intents
                WHERE deleted_at IS NULL
                ORDER BY intent_key ASC, created_at ASC
                """
            )
            rows = cur.fetchall()
        return [self._intent_from_row(row) for row in rows]

    def list_active_subgoal_templates(self) -> list[PlannerSubgoalTemplateRecord]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM planner_subgoal_templates
                WHERE deleted_at IS NULL
                ORDER BY intent_id ASC, sequence_no ASC, template_id ASC
                """
            )
            rows = cur.fetchall()
        return [self._template_from_row(row) for row in rows]

    def create_intent_with_templates(
        self,
        intent: PlannerIntentRecord,
        templates: Sequence[PlannerSubgoalTemplateRecord],
    ) -> None:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO planner_intents (
                        intent_id, intent_key, display_name, description,
                        created_at, updated_at, deleted_at
                    )
                    VALUES (
                        %(intent_id)s, %(intent_key)s, %(display_name)s, %(description)s,
                        %(created_at)s, %(updated_at)s, %(deleted_at)s
                    )
                    """,
                    asdict(intent),
                )
                for template in templates:
                    cur.execute(
                        """
                        INSERT INTO planner_subgoal_templates (
                            template_id, intent_id, sequence_no, subgoal_type,
                            activation_condition, created_at, updated_at, deleted_at
                        )
                        VALUES (
                            %(template_id)s, %(intent_id)s, %(sequence_no)s, %(subgoal_type)s,
                            %(activation_condition)s, %(created_at)s, %(updated_at)s, %(deleted_at)s
                        )
                        """,
                        asdict(template),
                    )
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "sqlstate", None) == "23505":
                raise PlannerCatalogConflictError(str(exc)) from exc
            raise

    def create_subgoal_template(self, template: PlannerSubgoalTemplateRecord) -> None:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO planner_subgoal_templates (
                        template_id, intent_id, sequence_no, subgoal_type,
                        activation_condition, created_at, updated_at, deleted_at
                    )
                    VALUES (
                        %(template_id)s, %(intent_id)s, %(sequence_no)s, %(subgoal_type)s,
                        %(activation_condition)s, %(created_at)s, %(updated_at)s, %(deleted_at)s
                    )
                    """,
                    asdict(template),
                )
        except Exception as exc:  # noqa: BLE001
            if getattr(exc, "sqlstate", None) == "23505":
                raise PlannerCatalogConflictError(str(exc)) from exc
            raise

    def soft_delete_intent(self, intent_id: str, *, deleted_at: datetime) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE planner_intents
                SET updated_at = %(deleted_at)s,
                    deleted_at = %(deleted_at)s
                WHERE intent_id = %(intent_id)s
                  AND deleted_at IS NULL
                RETURNING intent_id
                """,
                {"intent_id": intent_id, "deleted_at": deleted_at},
            )
            row = cur.fetchone()
            if row is None:
                return False
            cur.execute(
                """
                UPDATE planner_subgoal_templates
                SET updated_at = %(deleted_at)s,
                    deleted_at = %(deleted_at)s
                WHERE intent_id = %(intent_id)s
                  AND deleted_at IS NULL
                """,
                {"intent_id": intent_id, "deleted_at": deleted_at},
            )
        return True

    def soft_delete_subgoal_template(self, template_id: str, *, deleted_at: datetime) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE planner_subgoal_templates
                SET updated_at = %(deleted_at)s,
                    deleted_at = %(deleted_at)s
                WHERE template_id = %(template_id)s
                  AND deleted_at IS NULL
                RETURNING template_id
                """,
                {"template_id": template_id, "deleted_at": deleted_at},
            )
            row = cur.fetchone()
        return row is not None

    @staticmethod
    def _intent_from_row(row: dict[str, object]) -> PlannerIntentRecord:
        return PlannerIntentRecord(
            intent_id=str(row["intent_id"]),
            intent_key=str(row["intent_key"]),
            display_name=str(row["display_name"]),
            description=str(row["description"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted_at=row.get("deleted_at"),
        )

    @staticmethod
    def _template_from_row(row: dict[str, object]) -> PlannerSubgoalTemplateRecord:
        return PlannerSubgoalTemplateRecord(
            template_id=str(row["template_id"]),
            intent_id=str(row["intent_id"]),
            sequence_no=int(row["sequence_no"]),
            subgoal_type=str(row["subgoal_type"]),
            activation_condition=str(row["activation_condition"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            deleted_at=row.get("deleted_at"),
        )
