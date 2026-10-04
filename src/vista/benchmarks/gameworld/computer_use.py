"""The 100-decision computer-use contract: one `act` tool per role.

Physical validation and execution remain owned by the pinned GameWorld runtime.
"""

from __future__ import annotations

from typing import Any

_MAX_DURATION = 2
_KEY_COUNT = 2
_MAX_TEXT_LENGTH = 5
_MIN_DRAG_STEPS = 1
_DEFAULT_DRAG_STEPS = 10
_DEFAULT_TYPE_DURATION = 2


def _bounded_number(description: str, minimum: int, maximum: int) -> dict[str, Any]:
    # Codex 0.145 drops numeric bounds but preserves parameter descriptions.
    return {
        "type": "number",
        "minimum": minimum,
        "maximum": maximum,
        "description": f"{description} Inclusive range: {minimum} to {maximum}.",
    }


def _variant(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def action_schema(game: Any, role_index: int) -> dict[str, Any]:
    controls = game.game_roles[role_index].controls
    keys = sorted(controls.allowed_keys)
    duration = _bounded_number("Duration in seconds.", 0, _MAX_DURATION)
    x = _bounded_number("Horizontal pixels from the left edge.", 0, game.width - 1)
    y = _bounded_number("Vertical pixels from the top edge.", 0, game.height - 1)
    variants = [
        _variant({"action": {"const": "wait"}, "duration": duration}, ["action"])
    ]
    if keys:
        variants.append(
            _variant(
                {
                    "action": {"const": "press_key"},
                    "key": {"type": "string", "enum": keys},
                    "duration": duration,
                },
                ["action", "key"],
            )
        )
        if len(keys) > 1:
            variants.append(
                _variant(
                    {
                        "action": {"const": "press_keys"},
                        "keys": {
                            "type": "array",
                            "minItems": _KEY_COUNT,
                            "maxItems": _KEY_COUNT,
                            "uniqueItems": True,
                            "description": (
                                f"Exactly {_KEY_COUNT} distinct allowed keys, held "
                                "simultaneously. Opposing directions are not allowed."
                            ),
                            "items": {"type": "string", "enum": keys},
                        },
                        "duration": duration,
                    },
                    ["action", "keys"],
                )
            )
        if any(len(key) == 1 for key in keys):
            variants.append(
                _variant(
                    {
                        "action": {"const": "type"},
                        "text": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_TEXT_LENGTH,
                            "description": (
                                f"Text of length 1 to {_MAX_TEXT_LENGTH} characters "
                                "inclusive. Each character must map to an allowed key."
                            ),
                        },
                        "press_enter": {"type": "boolean"},
                        "duration": duration,
                    },
                    ["action", "text"],
                )
            )
    if controls.allow_clicks:
        button = {"type": "string", "enum": ["left", "right", "middle"]}
        variants.extend(
            [
                _variant(
                    {"action": {"const": "click"}, "x": x, "y": y, "button": button},
                    ["action", "x", "y"],
                ),
                _variant(
                    {
                        "action": {"const": "click_hold"},
                        "x": x,
                        "y": y,
                        "button": button,
                        "duration": duration,
                    },
                    ["action", "x", "y"],
                ),
                _variant(
                    {
                        "action": {"const": "mouse_move"},
                        "x": x,
                        "y": y,
                        "from_x": x,
                        "from_y": y,
                        "duration": duration,
                    },
                    ["action", "x", "y"],
                ),
                _variant(
                    {
                        "action": {"const": "drag"},
                        "x1": x,
                        "y1": y,
                        "x2": x,
                        "y2": y,
                        "button": button,
                        "duration": duration,
                        "steps": {
                            "type": "integer",
                            "minimum": _MIN_DRAG_STEPS,
                            "description": (
                                "Drag interpolation steps, an integer at least "
                                f"{_MIN_DRAG_STEPS}; default {_DEFAULT_DRAG_STEPS}."
                            ),
                        },
                    },
                    ["action", "x1", "y1", "x2", "y2"],
                ),
                _variant(
                    {
                        "action": {"const": "scroll"},
                        "delta_x": {"type": "number"},
                        "delta_y": {"type": "number"},
                        "duration": duration,
                    },
                    ["action"],
                ),
            ]
        )
    return {"type": "object", "oneOf": variants}


def action_description(game: Any, role_index: int) -> str:
    role = game.game_roles[role_index]
    controls = role.controls
    keys = ", ".join(sorted(controls.allowed_keys)) or "none"
    lines = [
        "Execute exactly one GameWorld input for the current role, not an action list.",
        f"Role {role_index} ({role.name}): keys [{keys}], "
        f"mouse {'enabled' if controls.allow_clicks else 'disabled'}.",
        f"Coordinates use the {game.width}x{game.height} screenshot, with origin at "
        f"the top-left: x from 0 to {game.width - 1}, y from 0 to {game.height - 1}, "
        "both inclusive. These bounds also apply to from_x/from_y and drag endpoints.",
        f"duration is optional, in seconds from 0 to {_MAX_DURATION} inclusive; "
        "fractions are allowed. An explicit duration overrides the defaults. "
        f"The role default hold/wait is {controls.hold_duration:g}s unless an "
        "action-specific default below applies.",
        "wait: Wait without input for duration.",
    ]
    if controls.allowed_keys:
        lines.append("press_key: Hold one allowed key for duration, then release it.")
        if len(controls.allowed_keys) > 1:
            lines.append(
                f"press_keys: Hold exactly {_KEY_COUNT} distinct allowed keys "
                "simultaneously for duration, then release them."
            )
            opposing = [
                f"{first}+{second}"
                for first, second in (
                    ("ArrowLeft", "ArrowRight"),
                    ("ArrowUp", "ArrowDown"),
                    ("a", "d"),
                    ("w", "s"),
                )
                if {first, second} <= set(controls.allowed_keys)
            ]
            if opposing:
                lines.append(
                    "Forbidden opposing key pairs: " + ", ".join(opposing) + "."
                )
        overrides = getattr(controls, "key_durations", {})
        if overrides:
            values = ", ".join(
                f"{key}={value:g}s" for key, value in sorted(overrides.items())
            )
            lines.append(
                f"When duration is omitted, per-key defaults are {values}. "
                "For press_keys, use the largest configured override among the "
                "selected keys, or the role default if none has an override."
            )
        if any(len(key) == 1 for key in controls.allowed_keys):
            lines.append(
                f"type: Send text of length 1 to {_MAX_TEXT_LENGTH} characters "
                "inclusive; every character must map to an allowed key. "
                "press_enter optionally adds Enter (default false), only if Enter "
                f"is allowed. The total typing duration defaults to {_DEFAULT_TYPE_DURATION}s."
            )
    if controls.allow_clicks:
        lines.extend(
            (
                "Mouse button is left, right, or middle; default left.",
                "click: Click at (x, y), then wait the role default. "
                "This action has no duration parameter.",
                "click_hold: Hold the mouse button at (x, y) for duration, then release it.",
                "mouse_move: Move to (x, y), then wait for duration. "
                "Optionally supply both from_x and from_y to move from that point.",
                "drag: Hold the mouse button from (x1, y1) to (x2, y2). "
                f"steps is an integer at least {_MIN_DRAG_STEPS}, default {_DEFAULT_DRAG_STEPS}; "
                "duration controls the interpolated drag time when steps is greater than 1.",
                "scroll: Scroll by delta_x and delta_y (each defaults to 0), "
                "then wait for duration.",
            )
        )
    return "\n".join(lines)


def objective(game: Any, task: Any, role_index: int) -> str:
    role = game.game_roles[role_index]
    role_text = "\n".join(
        part
        for part in (
            f"Role {role_index}: {role.name}",
            role.prompt.role_section.strip(),
            role.prompt.computer_use_controls_section.strip(),
        )
        if part
    )
    return "\n\n".join(
        part
        for part in (
            game.game_rules.strip(),
            task.task_prompt.strip(),
            role_text,
        )
        if part
    )
