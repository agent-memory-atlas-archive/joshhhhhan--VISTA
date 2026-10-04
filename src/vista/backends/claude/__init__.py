"""Shared claude transport and task binding factory."""

from typing import Any

__all__ = ["create_backend"]


def __getattr__(name: str) -> Any:
    if name == "create_backend":
        from .factory import create_backend

        return create_backend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
