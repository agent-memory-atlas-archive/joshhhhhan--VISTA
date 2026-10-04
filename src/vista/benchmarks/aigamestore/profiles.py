"""VISTA action contracts and session policy for AI GameStore."""

from __future__ import annotations

from vista.core.contracts import Capability, Effect, ToolSpec
from vista.core.profiles import (
    AgentProfile,
    CompactionMode,
    ContextMode,
    Experiment,
    InteractionMode,
    PromptBundle,
    SessionPolicy,
)
from vista.core.prompts import vista_method

from .environment import (
    HOLD_ACTIONS,
    MAX_DECISIONS,
    MODEL_ACTIONS,
    PROTOCOL,
    SEGMENT_SECONDS,
    SEGMENTS_PER_DECISION,
    TAP_ACTIONS,
    PublicGame,
)

SOURCES = ("live", "local")

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



def action_specs() -> tuple[ToolSpec, ...]:
    taps = ", ".join(sorted(TAP_ACTIONS))
    holds = ", ".join(sorted(HOLD_ACTIONS))
    return (
        ToolSpec(
            "play",
            (
                "Play exactly one game second of this real-time game. Provide "
                f"`segments`: exactly {SEGMENTS_PER_DECISION} chronological "
                f"{SEGMENT_SECONDS}-second segments, each a non-empty array of "
                "simultaneous inputs. Tap inputs press and release the named key "
                f"once at the start of the segment ({taps}); HOLD_ inputs keep the "
                f'key pressed for the whole segment ({holds}); ["NOOP"] is a '
                "segment with no input. NOOP beside other inputs is ignored, and a "
                "key both tapped and held in one segment is held. The game is "
                "frozen between decisions: it runs for exactly the one second you "
                "specify, then pauses and returns the resulting frames. Input on "
                "an ended game does nothing; use restart instead."
            ),
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["segments"],
                "properties": {
                    "segments": {
                        "type": "array",
                        "minItems": SEGMENTS_PER_DECISION,
                        "maxItems": SEGMENTS_PER_DECISION,
                        "description": (
                            f"Exactly {SEGMENTS_PER_DECISION} segments in time order, "
                            f"each covering {SEGMENT_SECONDS} seconds."
                        ),
                        "items": {
                            "type": "array",
                            "minItems": 1,
                            # The host collapses repeated inputs within a segment.
                            "description": (
                                "Simultaneous inputs for one segment; at least one "
                                "entry. A repeated input counts once."
                            ),
                            "items": {"type": "string", "enum": sorted(MODEL_ACTIONS)},
                        },
                    }
                },
            },
            Capability.ACTION,
            Effect.ENVIRONMENT,
            max_images=SEGMENTS_PER_DECISION,
            costs={"game_decisions": 1},
        ),
        ToolSpec(
            "restart",
            (
                "Restart the game from its start screen and pause it at its first "
                "frame. Costs one game second. Use it after a game over or to "
                "abandon the current attempt; the run continues until its game "
                "seconds are spent."
            ),
            {"type": "object", "additionalProperties": False, "properties": {}},
            Capability.ACTION,
            Effect.ENVIRONMENT,
            max_images=1,
            costs={"game_decisions": 1},
        ),
    )


PAPER_ROLE = (
    "You are a professional video game player tasked to win a 2D video game. "
    "You will read the description of the game and controls and provide an "
    "output for the next 5 actions, each covering 0.2 seconds of play."
)


def objective() -> str:
    """The reference harness's opening sentences, then the scoring rule."""
    return (
        f"{PAPER_ROLE} You are scored on the highest score reached at any point "
        "in the run, and you may restart as many times as you want, so keep "
        "playing until the run's game seconds are spent."
    )


def system_prompt(
    game: PublicGame, *, max_decisions: int = MAX_DECISIONS
) -> str:
    """Public game instructions; VISTA appends its method."""
    return "\n\n".join(
        (
            objective(),
            (
                "This is a real-time human video game that is frozen between your "
                f"decisions. Each `play` decision controls exactly one game second as "
                f"{SEGMENTS_PER_DECISION} chronological {SEGMENT_SECONDS}-second input "
                "segments (taps, holds, or NOOP); the game then pauses and shows "
                "you the resulting frames. The world keeps moving during your "
                "second, so plan the whole second. Read the game state from the "
                "frames; the on-screen display is your source for score and "
                f"progress. A run lasts {max_decisions} game seconds in total, and "
                "`restart` also spends one."
            ),
            "Game description:\n" + game.description.strip(),
            (
                "Human control scheme (for reference; you act through the `play` "
                "tool's segment inputs):\n" + game.controls.strip()
            ),
        )
    )


def make_profile(
    game: PublicGame,
    *,
    backend: str = "codex",
    max_decisions: int = MAX_DECISIONS,
) -> AgentProfile:
    if backend not in {"codex", "claude"}:
        raise ValueError("Unknown backend")
    actions = tuple(spec.name for spec in action_specs())
    system = system_prompt(game, max_decisions=max_decisions)
    system += "\n\n" + vista_method(backend)
    vista_tools = tuple(
        name
        for name in VISTA_TOOLS
        if backend == "codex" or name != "save_compact_checkpoint"
    )
    return AgentProfile(
        "aigamestore-segmented-input/vista",
        (*actions, *vista_tools),
        PromptBundle(
            system,
            "Game screen (paused):\n",
            step="Game screen (paused):\n",
            checkpoint="Save the compact checkpoint before continuing.",
            recovery=(
                "Continue this same game from GUIDE, WORKING, and the supplied "
                "current visual. The game has not been restarted or advanced by "
                "this context handoff. Reconcile the notes with the current screen."
            ),
        ),
        SessionPolicy(
            context=ContextMode.CONTINUOUS,
            interaction=InteractionMode.TOOLS,
            compaction=CompactionMode.CHECKPOINT,
        ),
    )


def make_experiment(
    profile: AgentProfile,
    game: PublicGame,
    *,
    backend: str,
    model: str,
    effort: str,
    max_decisions: int = MAX_DECISIONS,
    seed: int | None = None,
    source: str = "live",
    revision: str | None = None,
    backend_segments: int | None = None,
    tool_calls: int | None = None,
) -> Experiment:
    if type(max_decisions) is not int or not 1 <= max_decisions <= MAX_DECISIONS:
        raise ValueError(f"max_decisions must be in 1..{MAX_DECISIONS}")
    if source not in SOURCES:
        raise ValueError("Game source must be live or local")
    return Experiment(
        "AI GameStore",
        revision or game.revision,
        PROTOCOL,
        profile,
        backend,
        model,
        effort,
        {
            "game_number": game.number,
            "game_id": game.game_id,
            "pixels": "unaltered native canvas",
            "coordinate_scale": 1.0,
            "frames_delivered": "five-segment-frames",
            "frames_archived": SEGMENTS_PER_DECISION,
            "win_policy": "play-full-budget",
            "seed": seed,
            "game_source": source,
            "score_visibility": "hidden",
            "host_action_count": False,
        },
        {
            "game_decisions": max_decisions,
            # A final response without a game action is redirected, not counted
            # as a game second; bound those redirections by the same budget.
            "final_answers": max_decisions,
            **(
                {"backend_segments": backend_segments}
                if backend_segments is not None
                else {}
            ),
            **({"tool_calls": tool_calls} if tool_calls is not None else {}),
        },
        "aigamestore.org/getGameState-max-score+public-human-median",
    )
