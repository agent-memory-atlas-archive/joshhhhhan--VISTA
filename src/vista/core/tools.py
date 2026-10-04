"""Shared memory and supplied-visual tool schemas."""

from __future__ import annotations

from typing import Any

MAX_GUIDE_CHARS = 64 * 1024


MAX_WORKING_CHARS = 16 * 1024


DEFAULT_DISPLAY_SIZE = 1024


MAX_INSPECTION_VIEWS = 16


MAX_INSPECTION_QUESTION_CHARS = 1024


MAX_INSPECTION_LABEL_CHARS = 128


MAX_PIXEL_READOUT_VIEWS = 64


MAX_PIXEL_READOUT_SAMPLES = 4096


READ_GUIDE_TOOL = {
    "name": "read_guide",
    "description": "Read `GUIDE.md`.",
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


def _visual_region_schema(display_size: int) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["x", "y", "width", "height"],
        "properties": {
            "x": {"type": "integer", "minimum": 0, "maximum": display_size - 1},
            "y": {"type": "integer", "minimum": 0, "maximum": display_size - 1},
            "width": {"type": "integer", "minimum": 1, "maximum": display_size},
            "height": {"type": "integer", "minimum": 1, "maximum": display_size},
        },
    }


def build_inspect_tool(display_size: int = DEFAULT_DISPLAY_SIZE) -> dict[str, Any]:
    if type(display_size) is not int or display_size < 1:
        raise ValueError("display_size must be a positive integer")
    view_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "turn"],
        "properties": {
            "label": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_INSPECTION_LABEL_CHARS,
            },
            "turn": {
                "type": "integer",
                "minimum": 0,
                "maximum": 999999,
            },
            "frame": {
                "type": "integer",
                "minimum": 0,
                "maximum": 999999,
                "description": "Exact archived frame; omit for the final frame.",
            },
            "region": _visual_region_schema(display_size),
        },
    }
    return {
        "name": "inspect",
        "description": (
            "Inspect one or more supplied visuals without changing the current game "
            "state. Each view selects an archived turn; omit frame to use that turn's "
            "final frame. Omit region to see the full visual, or select a rectangular "
            f"region in the same standard {display_size}x{display_size} coordinates "
            "used by ACTION6. A selected region is cropped exactly from that visual "
            f"and enlarged proportionally to fit within {display_size}x{display_size} "
            "without smoothing. State the visual question these views should answer "
            "and give each view a short label. Include the evidence needed to answer "
            "that question in the same request. The selected views are returned in "
            "request order; your question and labels remain in the inspect call."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["question", "views"],
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_INSPECTION_QUESTION_CHARS,
                },
                "views": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_INSPECTION_VIEWS,
                    "items": view_schema,
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


def build_read_pixels_tool(
    display_size: int = DEFAULT_DISPLAY_SIZE,
) -> dict[str, Any]:
    if type(display_size) is not int or display_size < 1:
        raise ValueError("display_size must be a positive integer")
    view_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "turn", "region", "rows", "columns"],
        "properties": {
            "label": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_INSPECTION_LABEL_CHARS,
            },
            "turn": {
                "type": "integer",
                "minimum": 0,
                "maximum": 999999,
            },
            "frame": {
                "type": "integer",
                "minimum": 0,
                "maximum": 999999,
                "description": "Exact archived frame; omit for the final frame.",
            },
            "region": _visual_region_schema(display_size),
            "rows": {
                "type": "integer",
                "minimum": 1,
                "maximum": display_size,
            },
            "columns": {
                "type": "integer",
                "minimum": 1,
                "maximum": display_size,
            },
        },
    }
    return {
        "name": "read_pixels",
        "description": (
            "Read exact discrete color samples from one or more supplied visuals "
            "without changing the game state. Each view divides its selected "
            "rectangular region into equal rows and columns and samples the center "
            "pixel of every part from the archived supplied PNG. The result contains "
            "one RGB symbol palette and one compact row string per sampled "
            "row for each view in request order. At most "
            f"{MAX_PIXEL_READOUT_VIEWS} views may be selected, with at most "
            f"{MAX_PIXEL_READOUT_SAMPLES} samples total per call. This tool only "
            "measures pixels; it does not align, transform, compare, or interpret "
            "them. State the visual question and give each view a short label; those "
            "remain in this call."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["question", "views"],
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_INSPECTION_QUESTION_CHARS,
                },
                "views": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_PIXEL_READOUT_VIEWS,
                    "items": view_schema,
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


WRITE_GUIDE_TOOL = {
    "name": "write_guide",
    "description": (
        "Write the complete contents of `GUIDE.md`, replacing its previous contents."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["content"],
        "properties": {
            "content": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_GUIDE_CHARS,
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


SAVE_COMPACT_CHECKPOINT_TOOL = {
    "name": "save_compact_checkpoint",
    "description": (
        "Atomically save the pre-compaction game checkpoint. Set guide to a complete "
        "replacement GUIDE.md only when the durable game model materially changed; "
        "otherwise set it to null. Store in working_memory only the current "
        "continuation state that cannot be recovered from GUIDE.md and the current "
        "visual: the active plan, its next visible prediction, unresolved "
        "contradictions, and any necessary turn/frame references."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["guide", "working_memory"],
        "properties": {
            "guide": {
                "type": ["string", "null"],
                "minLength": 1,
                "maxLength": MAX_GUIDE_CHARS,
            },
            "working_memory": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_WORKING_CHARS,
            },
        },
    },
    "annotations": {
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}


READ_WORKING_TOOL = {
    "name": "read_working",
    "description": "Read agent-authored temporary state for the current task.",
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
        "Replace agent-authored temporary state for the current task. It persists until explicitly replaced or cleared by the task lifecycle."
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


HISTORY_TOOL = {
    "name": "history",
    "description": (
        "Read this role's public interaction records and visual references without changing the task. "
        "Event numbers start at 0 and follow this role's record order, including the initial input and supplied observations. "
        "Ranges include start and exclude end. Bounds may be one past the latest event. "
        "A bounded range returns its first limit records; omit both bounds for the most recent records."
    ),
    "inputSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": 128, "default": 64},
        },
    },
    "annotations": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
}


def build_tools(
    display_size: int = DEFAULT_DISPLAY_SIZE,
    *,
    include_compact_checkpoint: bool = True,
) -> tuple[dict[str, Any], ...]:
    tools = (
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
