from __future__ import annotations


class PlannerCatalogError(RuntimeError):
    """Base error for planner catalog operations."""


class PlannerCatalogValidationError(PlannerCatalogError):
    """Raised when a request payload is malformed."""


class PlannerCatalogConflictError(PlannerCatalogError):
    """Raised when a mutation conflicts with the current catalog state."""


class PlannerCatalogUnavailableError(PlannerCatalogError):
    """Raised when the catalog is running in read-only fallback mode."""
