"""HTTP helpers for the reasoning system."""

from __future__ import annotations

import urllib.error
import urllib.request
import json


def fetch_reasoning_status(base_url: str, *, timeout_s: float = 3.0) -> dict[str, object]:
    url = f"{str(base_url).rstrip('/')}/reasoning/status"
    request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"ok": False, "error": f"http_{exc.code}: {detail}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    if not isinstance(payload, dict):
        return {"ok": False, "error": "invalid reasoning payload"}
    return {"ok": bool(payload.get("ok", True)), "status": payload}
