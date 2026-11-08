"""Tool dispatch over RPC.

* :class:`LocalToolService` — direct in-process dispatch (stdlib only,
  tested), same request/response shape as the gRPC service.
* :func:`serve_grpc` — real gRPC server (lazy-imports ``grpc``; not
  required for the demo).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..tools.base import ToolRegistry


class LocalToolService:
    """RPC-shaped tool dispatcher that runs in-process."""

    def __init__(self, registry: ToolRegistry):
        self.registry = registry

    def call(self, name: str, arguments: Dict[str, Any]) -> str:
        return self.registry.execute(name, arguments)

    def call_json(self, name: str, arguments_json: str) -> str:
        try:
            args = json.loads(arguments_json) if arguments_json else {}
        except json.JSONDecodeError:
            args = {"_raw": arguments_json}
        return self.call(name, args)


def serve_grpc(registry: ToolRegistry, address: str = "[::]:50051") -> None:
    """Run a gRPC ToolService server (requires ``grpcio`` + generated stubs)."""
    try:
        import grpc  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("grpcio is not installed — run `pip install grpcio grpcio-tools`") from e

    raise NotImplementedError(
        "gRPC server needs generated stubs from tools.proto — "
        "see README 'RPC' section. Use LocalToolService for in-process dispatch."
    )
