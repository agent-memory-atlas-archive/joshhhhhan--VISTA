"""GameWorld actions and role-local observations; VISTA tools live in the core."""

from __future__ import annotations

from typing import Any

from vista.core.contracts import (
    Effect,
    EvidenceKind,
    ObservationBundle,
    PublicTask,
    SubmissionSpec,
    ToolBinding,
    ToolReply,
)

from .engine import GameWorldEngine
from .profiles import action_specs


class RoleSession:
    def __init__(self, engine: GameWorldEngine, role: int):
        self.engine = engine
        self.role = role
        self.closed = False
        self.latest = engine.current
        self.delivered_current: ObservationBundle | None = None
        self.task = PublicTask(
            f"{engine.game.game_id}/{engine.task.task_id}/role-{role}",
            engine.task.task_prompt,
            {
                "game_id": engine.game.game_id,
                "task_id": engine.task.task_id,
                "role": engine.game.game_roles[role].name,
            },
            self.latest,
        )
        self.submission = SubmissionSpec(
            Effect.ENVIRONMENT, max_images=1, costs={"game_decisions": 1}
        )
        self.tools = tuple(
            ToolBinding(
                spec, lambda arguments, name=spec.name: self._act(name, arguments)
            )
            for spec in action_specs(
                engine.upstream, engine.game, role
            )
        )

    def observe(self, bundle: ObservationBundle) -> None:
        self.latest = ObservationBundle(
            tuple(
                item
                for item in bundle.observations
                if item.kind == EvidenceKind.CURRENT
            )
        )

    def _act(self, name: str | None, arguments: Any = None) -> ToolReply:
        transition = self.engine.step(self.role, name, arguments)
        self.observe(transition.after)
        valid = bool(transition.validity["is_valid"])
        text = (
            "Final game screen."
            if valid
            else "No valid action was executed. Final game screen."
        )
        if transition.finished:
            text += " The task has ended."
        if len(transition.after.observations) > 1:
            text += f" Archived {len(transition.after.observations)} visual moments."
        return ToolReply(
            text,
            success=valid,
            observation=transition.after,
            finished=transition.finished,
        )

    def submit(self, answer: str) -> ToolReply:
        return self._act(None)

    def close(self) -> None:
        self.closed = True
