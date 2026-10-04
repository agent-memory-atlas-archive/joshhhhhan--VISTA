"""Explicit experiment identity and per-session instructions, without presets."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .contracts import ToolSpec
from .json import canonical, freeze, thaw


class ContextMode(StrEnum):
    CONTINUOUS = "continuous"
    FRESH = "fresh"


class InteractionMode(StrEnum):
    TOOLS = "native-tools"
    FINAL = "final-answer"
    STRUCTURED = "structured-reply"


class CompactionMode(StrEnum):
    NATIVE = "native"
    CHECKPOINT = "checkpoint-handoff"


class Phase(StrEnum):
    INITIAL = "initial"
    STEP = "step"
    RETRY = "retry"
    CHECKPOINT = "checkpoint"
    COMPACT = "compact"
    RECOVERY = "recovery"


@dataclass(frozen=True)
class PromptBundle:
    system: str
    initial: str
    step: str | None = None
    retry: str | None = None
    checkpoint: str | None = None
    compact: str | None = None
    recovery: str | None = None

    def __post_init__(self) -> None:
        if not self.system.strip():
            raise ValueError("System instructions must be explicit")
        for phase in Phase:
            text = getattr(self, phase.value)
            if text is not None and (not isinstance(text, str) or not text.strip()):
                raise ValueError("Configured prompt phases must not be empty")
        if self.initial is None:
            raise ValueError("An initial prompt is required")

    def for_phase(self, phase: Phase) -> str:
        text = getattr(self, Phase(phase).value)
        if text is None:
            raise ValueError(f"The profile does not permit the {phase} prompt phase")
        return text

    def manifest(self) -> dict[str, str | None]:
        return {
            "system": self.system,
            **{p.value: getattr(self, p.value) for p in Phase},
        }


@dataclass(frozen=True)
class SessionPolicy:
    context: ContextMode = ContextMode.CONTINUOUS
    interaction: InteractionMode = InteractionMode.TOOLS
    compaction: CompactionMode = CompactionMode.NATIVE
    one_action_per_turn: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", ContextMode(self.context))
        object.__setattr__(self, "interaction", InteractionMode(self.interaction))
        object.__setattr__(self, "compaction", CompactionMode(self.compaction))
        if type(self.one_action_per_turn) is not bool:
            raise ValueError("one_action_per_turn must be boolean")


@dataclass(frozen=True)
class AgentProfile:
    profile_id: str
    tools: tuple[str, ...]
    prompts: PromptBundle
    policy: SessionPolicy = field(default_factory=SessionPolicy)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        if not self.profile_id or len(set(self.tools)) != len(self.tools):
            raise ValueError("A profile needs an identity and unique tool names")

    def validate_tools(self, specs: tuple[ToolSpec, ...]) -> None:
        if tuple(spec.name for spec in specs) != self.tools:
            raise ValueError("Resolved tools must exactly match the profile order")
        if self.policy.interaction != InteractionMode.TOOLS and specs:
            raise ValueError(
                "Final/structured reply profiles cannot expose native tools"
            )

    def manifest(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "tools": list(self.tools),
            "prompts": self.prompts.manifest(),
            "context": self.policy.context.value,
            "interaction": self.policy.interaction.value,
            "compaction": self.policy.compaction.value,
            "one_action_per_turn": self.policy.one_action_per_turn,
        }


@dataclass(frozen=True)
class Experiment:
    benchmark: str
    benchmark_revision: str
    protocol: str
    profile: AgentProfile
    backend: str
    model: str
    effort: str
    presentation: Mapping[str, Any]
    budgets: Mapping[str, int]
    evaluator: str

    def __post_init__(self) -> None:
        if self.backend not in {"codex", "claude"}:
            raise ValueError("Unsupported backend")
        if not all(
            (
                self.benchmark,
                self.benchmark_revision,
                self.protocol,
                self.model,
                self.effort,
                self.evaluator,
            )
        ):
            raise ValueError("Experiment identity must be explicit")
        object.__setattr__(self, "presentation", freeze(self.presentation))
        object.__setattr__(self, "budgets", freeze(self.budgets))
        for name, limit in self.budgets.items():
            if not name or type(limit) is not int or limit < 0:
                raise ValueError("Budgets must be named nonnegative integer limits")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "benchmark": self.benchmark,
            "benchmark_revision": self.benchmark_revision,
            "protocol": self.protocol,
            "profile": self.profile.manifest(),
            "backend": self.backend,
            "model": self.model,
            "effort": self.effort,
            "presentation": thaw(self.presentation),
            "budgets": thaw(self.budgets),
            "evaluator": self.evaluator,
        }

    @property
    def identity(self) -> str:
        return hashlib.sha256(canonical(self.manifest()).encode("ascii")).hexdigest()

    def matched_environment(self, other: Experiment) -> bool:
        first, second = self.manifest(), other.manifest()
        first.pop("profile")
        second.pop("profile")
        return first == second
