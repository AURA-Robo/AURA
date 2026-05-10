"""Dashboard payload contracts shared across subsystems."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class LogRecord:
    source: str
    stream: str
    message: str
    level: str | None = None
    path: str | None = None
    timestampNs: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.details:
            payload.pop("details", None)
        return payload


@dataclass(slots=True)
class ProcessRecord:
    name: str
    state: str
    required: bool
    pid: int | None
    exitCode: int | None
    startedAt: float | None
    healthUrl: str
    stdoutLog: str
    stderrLog: str
    stdoutLogOffset: int | None = None
    stderrLogOffset: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ServiceSnapshot:
    name: str
    status: str
    healthUrl: str | None = None
    latencyMs: float | None = None
    health: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if not self.health:
            payload.pop("health", None)
        if not self.debug:
            payload.pop("debug", None)
        return payload
