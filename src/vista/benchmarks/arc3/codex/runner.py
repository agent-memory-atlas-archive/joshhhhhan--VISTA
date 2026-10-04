"""ARC3 defaults for the shared Codex transport."""

from typing import Any

from vista.backends.codex import runner as _runner

from .recovery import (
    compact_recovery_prompt,
    retry_recovery_prompt,
    runtime_recovery_prompt,
)
from .tools import build_tools


class DockerCodexRunner(_runner.DockerCodexRunner):
    def _default_tool_definitions(self) -> tuple[dict[str, Any], ...]:
        return build_tools(self.display_size)

    def _default_recovery_prompt(self, kind: str, recovery: Any) -> str:
        return {
            "compact": compact_recovery_prompt,
            "retry": retry_recovery_prompt,
            "runtime": runtime_recovery_prompt,
        }[kind](recovery)


def __getattr__(name: str) -> Any:
    return getattr(_runner, name)


__all__ = [name for name in dir(_runner) if not name.startswith("_")]
