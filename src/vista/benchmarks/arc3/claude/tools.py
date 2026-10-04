"""Public tool contracts for the visual game player."""

from __future__ import annotations

from typing import Any

from vista.core.tools import (
    DEFAULT_DISPLAY_SIZE as DEFAULT_DISPLAY_SIZE,
    MAX_GUIDE_CHARS as MAX_GUIDE_CHARS,
    MAX_INSPECTION_LABEL_CHARS as MAX_INSPECTION_LABEL_CHARS,
    MAX_INSPECTION_QUESTION_CHARS as MAX_INSPECTION_QUESTION_CHARS,
    MAX_INSPECTION_VIEWS as MAX_INSPECTION_VIEWS,
    MAX_PIXEL_READOUT_SAMPLES as MAX_PIXEL_READOUT_SAMPLES,
    MAX_PIXEL_READOUT_VIEWS as MAX_PIXEL_READOUT_VIEWS,
    MAX_WORKING_CHARS as MAX_WORKING_CHARS,
    READ_GUIDE_TOOL as READ_GUIDE_TOOL,
    SAVE_COMPACT_CHECKPOINT_TOOL as SAVE_COMPACT_CHECKPOINT_TOOL,
    WRITE_GUIDE_TOOL as WRITE_GUIDE_TOOL,
    _visual_region_schema as _visual_region_schema,
    build_inspect_tool as build_inspect_tool,
    build_read_pixels_tool as build_read_pixels_tool,
)

ACTION_NAMES = (
    "RESET",
    "ACTION1",
    "ACTION2",
    "ACTION3",
    "ACTION4",
    "ACTION5",
    "ACTION6",
    "ACTION7",
)


def build_action_schema(
    display_size: int = DEFAULT_DISPLAY_SIZE,
    *,
    reset_requires_retry_state: bool = False,
) -> dict[str, Any]:
    if type(display_size) is not int or display_size < 1:
        raise ValueError("display_size must be a positive integer")
    properties: dict[str, Any] = {
        "action": {"type": "string", "enum": list(ACTION_NAMES)},
        "x": {
            "type": ["integer", "null"],
            "minimum": 0,
            "maximum": display_size - 1,
        },
        "y": {
            "type": ["integer", "null"],
            "minimum": 0,
            "maximum": display_size - 1,
        },
    }
    if reset_requires_retry_state:
        properties["retry_state"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": MAX_WORKING_CHARS,
            "description": (
                "For RESET, the smallest sufficient continuation state for "
                "the next attempt. It replaces WORKING.md."
            ),
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["action"],
        "properties": properties,
    }


def build_play_tool(
    display_size: int = DEFAULT_DISPLAY_SIZE,
    *,
    reset_requires_retry_state: bool = False,
) -> dict[str, Any]:
    reset_handoff = (
        " For RESET, include the smallest sufficient continuation state for the "
        "next attempt in retry_state; it replaces WORKING.md."
        if reset_requires_retry_state
        else ""
    )
    return {
        "name": "play",
        "description": (
            "Execute exactly one game action. Each executed play call counts as one "
            "game action. The result contains the resulting "
            "final visual, the number of archived frames for the action, and the "
            "actions available afterward. The player observes this result before "
            "another action can execute. Use only currently available actions. RESET "
            "initializes or restarts the game or level. ACTION1 through ACTION4 are "
            "game-dependent simple actions usually associated with up, down, left, "
            "and right. ACTION5 is a game-dependent simple action, usually an "
            "interaction such as select, rotate, attach, detach, execute, etc. "
            "ACTION6 is a coordinate action. ACTION7 is Undo. UI bindings: "
            "ACTION1=W/Up Arrow, ACTION2=S/Down Arrow, ACTION3=A/Left Arrow, "
            "ACTION4=D/Right Arrow, ACTION5=Space/F, ACTION6=mouse click, and "
            "ACTION7=Ctrl/Cmd+Z. For ACTION6, always provide "
            f"x and y in the standard {display_size}x{display_size} coordinate system; "
            "x increases right and y increases down."
            f"{reset_handoff}"
        ),
        "inputSchema": build_action_schema(
            display_size,
            reset_requires_retry_state=reset_requires_retry_state,
        ),
        "annotations": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False,
        },
    }


HISTORY_TOOL = {
    "name": "history",
    "description": (
        "Read objective action and environment-result records from this run without "
        "changing the game. `attempts` summarizes RESET-to-GAME_OVER attempts in "
        "the current level. `events` returns exact public action/result records by "
        "turn; use start_turn, end_turn, and limit to select a range."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["view"],
        "properties": {
            "view": {
                "type": "string",
                "enum": ["attempts", "events"],
            },
            "start_turn": {
                "type": "integer",
                "minimum": 0,
                "maximum": 999999,
            },
            "end_turn": {
                "type": "integer",
                "minimum": 0,
                "maximum": 999999,
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 128,
                "default": 64,
            },
        },
    },
    "annotations": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}

READ_WORKING_TOOL = {
    "name": "read_working",
    "description": "Read the agent-authored temporary state for the current level.",
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    },
    "annotations": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}

WRITE_WORKING_TOOL = {
    "name": "write_working",
    "description": (
        "Replace the agent-authored temporary state for the current level. It "
        "persists across RESET and is cleared when level progress advances."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["content"],
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_WORKING_CHARS,
            }
        },
    },
    "annotations": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}


def build_tools(
    display_size: int = DEFAULT_DISPLAY_SIZE,
    *,
    include_compact_checkpoint: bool = False,
    reset_requires_retry_state: bool = False,
) -> tuple[dict[str, Any], ...]:
    tools = (
        build_play_tool(
            display_size,
            reset_requires_retry_state=reset_requires_retry_state,
        ),
        build_inspect_tool(display_size),
        build_read_pixels_tool(display_size),
        HISTORY_TOOL,
        READ_GUIDE_TOOL,
        WRITE_GUIDE_TOOL,
        READ_WORKING_TOOL,
        WRITE_WORKING_TOOL,
    )
    if include_compact_checkpoint:
        return (*tools, SAVE_COMPACT_CHECKPOINT_TOOL)
    return tools
