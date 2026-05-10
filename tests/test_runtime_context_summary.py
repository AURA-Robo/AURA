from __future__ import annotations

from backend.models import DashboardStateBuilder
from backend.runtime_context_summary import render_runtime_context_summary


def test_render_runtime_context_summary_renders_stable_sections_and_placeholders() -> None:
    state = DashboardStateBuilder(api_base_url="http://127.0.0.1:18095").default_state()
    state["timestamp"] = 1_710_000_000.0
    state["session"]["active"] = True
    state["session"]["config"] = {"viewerEnabled": True, "memoryStore": False}
    state["runtime"]["executionMode"] = "NAV"
    state["runtime"]["reasoningTaskStatus"] = "running"
    state["runtime"]["plannerControlMode"] = "running"
    state["runtime"]["activeInstruction"] = "go to the tv"
    state["transport"]["viewerEnabled"] = True
    state["transport"]["frameAvailable"] = True
    state["transport"]["frameAgeMs"] = 12.5
    state["services"]["runtime"]["health"] = {"ownedByBackend": True}
    state["selectedTargetSummary"] = {"className": "TV", "source": "navigation"}
    state["logs"] = [{"source": "backend", "level": "info", "message": "session ready", "timestampNs": 1_710_000_000_000_000_000}]

    rendered = render_runtime_context_summary(state)

    assert rendered.startswith("# Runtime Context Summary")
    assert "## Session" in rendered
    assert "## Runtime execution" in rendered
    assert "## Service health" in rendered
    assert "## Viewer and sensing" in rendered
    assert "## Memory and knowledge" in rendered
    assert "## Recent logs" in rendered
    assert "- Session active: true" in rendered
    assert '- Config: {"memoryStore": false, "viewerEnabled": true}' in rendered
    assert "- Reasoning task status: running" in rendered
    assert "- Reasoning route: n/a" in rendered
    assert "- Route state: n/a" in rendered
    assert '- Selected target: {"className": "TV", "source": "navigation"}' in rendered
    assert "[info] backend: session ready" in rendered
