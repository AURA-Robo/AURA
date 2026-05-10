"""Log collection helpers for the backend."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import time
from typing import Iterable

from systems.shared.contracts.dashboard import LogRecord


def tail_log(
    path: str | Path,
    *,
    limit: int = 40,
    source: str,
    stream: str,
    start_offset: int | None = None,
) -> list[dict[str, object]]:
    target = Path(path)
    if not target.is_file():
        return []
    lines = deque(maxlen=max(1, int(limit)))
    file_size = target.stat().st_size
    offset = int(start_offset or 0)
    if offset < 0 or offset > file_size:
        offset = 0
    with target.open("rb") as handle:
        if offset:
            handle.seek(offset)
        for line in handle:
            text = line.decode("utf-8", errors="ignore").rstrip()
            if text:
                lines.append(text)
    now_ns = int(time.time() * 1_000_000_000)
    return [
        LogRecord(source=source, stream=stream, message=text, path=str(target), timestampNs=now_ns).to_dict()
        for text in lines
    ]


def merge_logs(*groups: Iterable[dict[str, object]], limit: int = 100) -> list[dict[str, object]]:
    merged: list[dict[str, object]] = []
    for group in groups:
        merged.extend(group)
    merged.sort(key=lambda item: int(item.get("timestampNs") or 0))
    return merged[-limit:]
