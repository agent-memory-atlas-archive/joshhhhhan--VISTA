"""VISTA computer-use action contracts and per-role session policy."""

from __future__ import annotations

from typing import Any

from vista.core.contracts import Capability, Effect, ToolSpec
from vista.core.profiles import (
    AgentProfile,
    CompactionMode,
    ContextMode,
    Experiment,
    PromptBundle,
    SessionPolicy,
)
from vista.core.prompts import vista_method as vista_method

from .upstream import (
    COMPUTER_USE_PROTOCOL,
    GAMES_REVISION,
    GAMEWORLD_REVISION,
    Upstream,
)

VISTA_TOOLS = (
    "inspect",
    "read_pixels",
    "history",
    "read_guide",
    "write_guide",
    "read_working",
    "write_working",
    "save_compact_checkpoint",
)



def action_specs(upstream: Upstream, game: Any, role: int) -> tuple[ToolSpec, ...]:
    from .computer_use import action_description, action_schema

    return (
        ToolSpec(
            "act",
            action_description(game, role),
            action_schema(game, role),
            Capability.ACTION,
            Effect.ENVIRONMENT,
            max_images=1,
            costs={"game_decisions": 1},
        ),
    )


def make_profile(
    upstream: Upstream,
    game: Any,
    task: Any,
    role: int,
    *,
    backend: str = "codex",
) -> AgentProfile:
    if backend not in {"codex", "claude"}:
        raise ValueError("Unknown backend")
    from .computer_use import objective

    actions = tuple(spec.name for spec in action_specs(upstream, game, role))
    system = objective(game, task, role) + "\n\n" + vista_method(backend)
    vista_tools = tuple(
        name
        for name in VISTA_TOOLS
        if backend == "codex" or name != "save_compact_checkpoint"
    )
    return AgentProfile(
        "gameworld-computer-use/vista",
        (*actions, *vista_tools),
        PromptBundle(
            system,
            "Game screen:\n",
            step="Game screen:\n",
            checkpoint="Save the compact checkpoint before continuing.",
            recovery=(
                "Continue this same role and task from GUIDE, WORKING, and the "
                "supplied current visual. The game has not been reset by this "
                "context handoff. Reconcile the notes with the current screen."
            ),
        ),
        SessionPolicy(
            context=ContextMode.CONTINUOUS,
            compaction=CompactionMode.CHECKPOINT,
            one_action_per_turn=game.role_count > 1,
        ),
    )


def make_experiment(
    profile: AgentProfile,
    *,
    backend: str,
    model: str,
    effort: str,
    width: int,
    height: int,
    seed: int | None,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
) -> Experiment:
    return Experiment(
        "GameWorld",
        GAMEWORLD_REVISION,
        COMPUTER_USE_PROTOCOL,
        profile,
        backend,
        model,
        effort,
        {
            "width": width,
            "height": height,
            "pixels": "unaltered",
            "coordinate_scale": 1.0,
            "games_revision": GAMES_REVISION,
            "seed": seed,
            "role_context": "isolated",
            "cross_task_memory": False,
            "host_action_count": False,
            "host_action_horizon": False,
        },
        {
            "game_decisions": 100,
            **(
                {"backend_segments": backend_segments}
                if backend_segments is not None
                else {}
            ),
            **({"tool_calls": tool_calls} if tool_calls is not None else {}),
            "final_answers": 100,
        },
        f"env.task_evaluator@{GAMEWORLD_REVISION}",
    )
