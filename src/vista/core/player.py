"""Optional task ports for shared transports; session policy lives in the backends."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class ToolExecutor(Protocol):
    def execute(self, tool: str, arguments: Any, call_id: str) -> Any: ...


@dataclass(frozen=True)
class ToolInterface:
    definitions: Sequence[dict[str, Any]]
    executor: ToolExecutor
    action_tools: frozenset[str]
    visual_readers: frozenset[str]
    predict_images: Callable[[str, object], int]
    acknowledge: Callable[[str | None], None]


@dataclass(frozen=True)
class RecoveryPrompts:
    compact: Callable[[Any], str]
    retry: Callable[[Any], str]
    runtime: Callable[[Any], str]
    fresh_runtime: Callable[[Any], str]
