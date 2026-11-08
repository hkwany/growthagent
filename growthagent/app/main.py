"""FastAPI entrypoint for the growth agent service.

Endpoints:

* ``GET  /health`` — liveness.
* ``POST /agent/run`` — run a task, return JSON summary.
* ``POST /agent/run/stream`` — run a task, stream trace events as SSE.

Run with::

    pip install "fastapi" "uvicorn[standard]"
    uvicorn growthagent.app.main:app --reload

FastAPI is optional — the service core in ``growthagent.app.service``
runs (and is tested) without it.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator

from pydantic import BaseModel, Field

from ..config import get_settings
from ..schema import TraceEvent
from .service import GrowthAgentService


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=1)
    use_planner: bool = False
    use_reflector: bool = False
    write_trace: bool = True
    write_ledger: bool = True


class TaskResponse(BaseModel):
    task: str
    final_answer: str
    terminated: str
    steps: int
    tool_calls: list[str]
    usage: Dict[str, int]
    latency_ms: float
    model: str


app = None


def build_app():
    global app
    if app is None:
        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse

        app = FastAPI(title="growthagent", version="0.1.0")
        service = GrowthAgentService(get_settings())

        @app.get("/health")
        def health() -> Dict[str, str]:
            return {"status": "ok"}

        @app.post("/agent/run", response_model=TaskResponse)
        def run_task(req: TaskRequest) -> Dict[str, Any]:
            return service.run_summary(
                req.task,
                use_planner=req.use_planner,
                use_reflector=req.use_reflector,
                write_trace=req.write_trace,
                write_ledger=req.write_ledger,
            )

        @app.post("/agent/run/stream")
        def run_task_stream(req: TaskRequest) -> StreamingResponse:
            def gen() -> Iterator[str]:
                def emit(event: TraceEvent) -> None:
                    pass

                events: list = []

                def collect(event: TraceEvent) -> None:
                    events.append(event.to_dict())

                service.run(
                    req.task,
                    on_event=collect,
                    use_planner=req.use_planner,
                    use_reflector=req.use_reflector,
                    write_trace=req.write_trace,
                    write_ledger=req.write_ledger,
                )
                for ev in events:
                    yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

    return app


# module-level app object for `uvicorn growthagent.app.main:app`
app = build_app()
