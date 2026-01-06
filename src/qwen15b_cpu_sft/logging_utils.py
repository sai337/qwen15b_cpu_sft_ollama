from __future__ import annotations

import json
import os
import time
from typing import Any, Dict


def atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write JSON atomically to avoid half-written status files."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def format_seconds(secs: float) -> str:
    secs = int(max(secs, 0))
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def now_ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
