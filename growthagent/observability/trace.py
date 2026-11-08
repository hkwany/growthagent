"""Trace writing: persist an agent run as JSONL for later inspection.

No external observability dependency — OpenTelemetry / Langfuse can be
added by emitting the same events, but the JSONL writer keeps the demo
self-contained and greppable.
"""

from __future__ import annotations

import json
import os
from typing import Any, List

from ..schema import TraceEvent, utc_iso


def event_to_jsonl(event: TraceEvent) -> str:
    return json.dumps(event.to_dict(), ensure_ascii=False)


def run_to_jsonl(run: Any) -> List[str]:
    """Serialize a whole run (header + every trace event) as JSONL lines."""
    header = {
        "type": "run_header",
        "task": run.task,
        "model": run.model,
        "terminated": run.terminated,
        "steps": run.steps,
        "usage": run.usage.to_dict(),
        "latency_ms": round(run.latency_ms, 1),
        "started_at": utc_iso(),
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    lines += [event_to_jsonl(e) for e in run.trace]
    return lines


class TraceWriter:
    """Append agent runs to a JSONL file."""

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def write(self, run: Any) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            for line in run_to_jsonl(run):
                f.write(line + "\n")
