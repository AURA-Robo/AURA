"""Default local service endpoint contracts for AURA processes."""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_LOCAL_HOST = "127.0.0.1"


@dataclass(frozen=True, slots=True)
class ServiceEndpoint:
    """Stable local endpoint for one AURA process boundary."""

    service_id: str
    host: str
    port: int
    status_path: str

    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def status_url(self) -> str:
        return f"{self.base_url()}{self.status_path}"

    def health_url(self) -> str:
        return f"{self.base_url()}/healthz"


BACKEND_ENDPOINT = ServiceEndpoint(
    service_id="backend",
    host=DEFAULT_LOCAL_HOST,
    port=18095,
    status_path="/api/state",
)
RUNTIME_ENDPOINT = ServiceEndpoint(
    service_id="runtime",
    host=DEFAULT_LOCAL_HOST,
    port=18096,
    status_path="/session/state",
)
INFERENCE_SYSTEM_ENDPOINT = ServiceEndpoint(
    service_id="inference_system",
    host=DEFAULT_LOCAL_HOST,
    port=15880,
    status_path="/models/state",
)
REASONING_SYSTEM_ENDPOINT = ServiceEndpoint(
    service_id="reasoning_system",
    host=DEFAULT_LOCAL_HOST,
    port=17881,
    status_path="/reasoning/status",
)
NAVIGATION_SYSTEM_ENDPOINT = ServiceEndpoint(
    service_id="navigation_system",
    host=DEFAULT_LOCAL_HOST,
    port=17882,
    status_path="/navigation/status",
)
CONTROL_RUNTIME_ENDPOINT = ServiceEndpoint(
    service_id="control_runtime",
    host=DEFAULT_LOCAL_HOST,
    port=8892,
    status_path="/runtime/status",
)
