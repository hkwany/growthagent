"""RPC layer (tool dispatch over the wire)."""

from __future__ import annotations

from .server import LocalToolService, serve_grpc

__all__ = ["LocalToolService", "serve_grpc"]
