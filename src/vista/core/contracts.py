"""Public evidence and tool contracts. Evaluator state is deliberately absent."""

from __future__ import annotations

import base64
import hashlib
import io
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from html import escape
from typing import Any, Protocol

from .json import freeze, thaw


class Effect(StrEnum):
    NONE = "none"
    ENVIRONMENT = "environment"
    MEMORY = "memory"
    ANNOTATION = "annotation"
    SUBMISSION = "submission"


class Capability(StrEnum):
    ACTION = "action"
    SUBMISSION = "submission"
    INSPECTION = "inspection"
    PIXELS = "pixels"
    HISTORY = "history"
    MEMORY = "memory"
    ANNOTATION = "annotation"
    CHECKPOINT = "checkpoint"


class EvidenceKind(StrEnum):
    CURRENT = "current"
    HISTORICAL = "historical"
    DERIVED = "derived"


@dataclass(frozen=True)
class Visual:
    """Exactly the display bytes approved by an adapter, not a source-file path."""

    view: str
    source_id: str
    data: bytes = field(repr=False)
    width: int
    height: int
    mime_type: str = "image/png"
    transform: Mapping[str, Any] = field(default_factory=dict)
    derived_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.view or not self.source_id:
            raise ValueError("A visual needs a view name and opaque source identity")
        if type(self.data) is not bytes or not self.data:
            raise ValueError("A visual needs immutable image bytes")
        if any(type(size) is not int or size < 1 for size in (self.width, self.height)):
            raise ValueError("Visual dimensions must be positive integers")
        if self.mime_type not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise ValueError("Unsupported visual media type")
        from PIL import Image

        with Image.open(io.BytesIO(self.data)) as image:
            if image.size != (self.width, self.height):
                raise ValueError("Declared visual dimensions do not match the image")
            if Image.MIME.get(image.format) != self.mime_type:
                raise ValueError("Declared media type does not match the image")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(
                    "Animated images must be supplied as explicit observations"
                )
            image.verify()
        object.__setattr__(self, "transform", freeze(self.transform))
        object.__setattr__(self, "derived_from", tuple(self.derived_from))

    def manifest(self) -> dict[str, Any]:
        return {
            "view": self.view,
            "source_id": self.source_id,
            "sha256": hashlib.sha256(self.data).hexdigest(),
            "width": self.width,
            "height": self.height,
            "mime_type": self.mime_type,
            "transform": thaw(self.transform),
            "derived_from": list(self.derived_from),
        }


@dataclass(frozen=True)
class Observation:
    moment: str
    views: tuple[Visual, ...]
    kind: EvidenceKind = EvidenceKind.CURRENT

    def __post_init__(self) -> None:
        object.__setattr__(self, "views", tuple(self.views))
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        if not self.moment or not self.views:
            raise ValueError("An observation needs an identity and at least one view")
        if len({view.view for view in self.views}) != len(self.views):
            raise ValueError("View names must be unique within an observation")
        if self.kind == EvidenceKind.DERIVED and any(
            not view.derived_from for view in self.views
        ):
            raise ValueError("Derived visuals must name their source evidence")
        if self.kind != EvidenceKind.DERIVED and any(
            view.derived_from for view in self.views
        ):
            raise ValueError("Agent-derived evidence must be labeled as derived")


@dataclass(frozen=True)
class ObservationBundle:
    observations: tuple[Observation, ...] = ()
    display: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "observations", tuple(self.observations))
        identities = [observation.moment for observation in self.observations]
        if len(set(identities)) != len(identities):
            raise ValueError("Observation moment identities must be unique")
        if sum(o.kind == EvidenceKind.CURRENT for o in self.observations) > 1:
            raise ValueError("Only one observation can represent the current state")
        if self.display is not None:
            object.__setattr__(self, "display", tuple(self.display))
            if any(
                type(index) is not int or not 0 <= index < len(self.observations)
                for index in self.display
            ) or len(set(self.display)) != len(self.display):
                raise ValueError("Display must select distinct supplied observations")

    @property
    def displayed_observations(self) -> tuple[Observation, ...]:
        if self.display is None:
            return self.observations
        return tuple(self.observations[index] for index in self.display)

    def manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "moment": observation.moment,
                "kind": observation.kind.value,
                "views": [view.manifest() for view in observation.views],
            }
            for observation in self.observations
        ]

    def images(self) -> tuple[dict[str, Any], ...]:
        images = []
        for observation in self.displayed_observations:
            for view in observation.views:
                label = (
                    f'<visual kind="{observation.kind.value}" '
                    f'view="{escape(view.view, quote=True)}" '
                    f'size="{view.width}x{view.height}"></visual>'
                )
                images.append(
                    {
                        "data": base64.b64encode(view.data).decode("ascii"),
                        "mime_type": view.mime_type,
                        "detail": "original",
                        "steer_text": label,
                    }
                )
        return tuple(images)


@dataclass(frozen=True)
class PublicTask:
    task_id: str
    objective: str
    context: Mapping[str, Any] = field(default_factory=dict)
    observation: ObservationBundle = field(default_factory=ObservationBundle)

    def __post_init__(self) -> None:
        if not self.task_id or not self.objective.strip():
            raise ValueError("A public task needs an identity and objective")
        object.__setattr__(self, "context", freeze(self.context))


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    capability: Capability
    effect: Effect = Effect.NONE
    max_images: int = 0
    costs: Mapping[str, int] = field(default_factory=dict)
    dispatch_invalid: bool = False
    annotations: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.name):
            raise ValueError("Invalid tool name")
        if not self.description.strip():
            raise ValueError("A tool needs a description")
        object.__setattr__(self, "capability", Capability(self.capability))
        object.__setattr__(self, "effect", Effect(self.effect))
        object.__setattr__(self, "input_schema", freeze(self.input_schema))
        object.__setattr__(self, "costs", freeze(self.costs))
        if self.annotations is not None:
            object.__setattr__(self, "annotations", freeze(self.annotations))
        if self.input_schema.get("type") != "object":
            raise ValueError("A tool input schema must describe an object")
        if type(self.max_images) is not int or self.max_images < 0:
            raise ValueError("max_images must be a nonnegative integer")
        if type(self.dispatch_invalid) is not bool or (
            self.dispatch_invalid and self.capability != Capability.ACTION
        ):
            raise ValueError("Only action tools may delegate invalid-output handling")
        for name, cost in self.costs.items():
            if (
                not name
                or name in {"tool_calls", "backend_segments", "final_answers"}
                or type(cost) is not int
                or cost < 0
            ):
                raise ValueError("Invalid tool accounting rule")
        allowed_effects = {
            Capability.ACTION: {Effect.ENVIRONMENT},
            Capability.SUBMISSION: {Effect.SUBMISSION},
            Capability.INSPECTION: {Effect.NONE},
            Capability.PIXELS: {Effect.NONE},
            Capability.HISTORY: {Effect.NONE},
            Capability.MEMORY: {Effect.NONE, Effect.MEMORY},
            Capability.ANNOTATION: {Effect.NONE, Effect.ANNOTATION},
            Capability.CHECKPOINT: {Effect.MEMORY},
        }
        if self.effect not in allowed_effects[self.capability]:
            raise ValueError("The tool capability and state effect disagree")

    @property
    def requires_ack(self) -> bool:
        return self.max_images > 0 or self.effect != Effect.NONE

    def public_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": thaw(self.input_schema),
            "annotations": thaw(self.annotations)
            if self.annotations is not None
            else {
                "readOnlyHint": self.effect == Effect.NONE,
                "destructiveHint": False,
                "idempotentHint": self.effect == Effect.NONE,
                "openWorldHint": False,
            },
        }


@dataclass(frozen=True)
class ToolReply:
    text: str
    success: bool = True
    observation: ObservationBundle = field(default_factory=ObservationBundle)
    finished: bool = False
    boundary: str | None = None
    # Persistent environment result, without transient tool-use guidance.
    event_text: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("A public tool reply must contain text")
        if self.event_text is not None and not isinstance(self.event_text, str):
            raise TypeError("Environment event text must be a string")
        if type(self.success) is not bool or type(self.finished) is not bool:
            raise TypeError("Tool reply status fields must be booleans")
        if not isinstance(self.observation, ObservationBundle):
            raise TypeError("A tool reply must use an ObservationBundle")
        if self.boundary not in {None, "decision_complete", "compact_checkpoint_saved"}:
            raise ValueError("Unknown task boundary")
        if self.finished and self.boundary is not None:
            raise ValueError("A terminal reply cannot also request a continuation")


@dataclass(frozen=True)
class ToolBinding:
    spec: ToolSpec
    handler: Callable[[Any], ToolReply] = field(repr=False, compare=False)


@dataclass(frozen=True)
class SubmissionSpec:
    """Host-side handling of a final answer or a structured action response."""

    effect: Effect = Effect.SUBMISSION
    max_images: int = 0
    costs: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "effect", Effect(self.effect))
        object.__setattr__(self, "costs", freeze(self.costs))
        if self.effect not in {Effect.SUBMISSION, Effect.ENVIRONMENT}:
            raise ValueError(
                "A final response must submit an answer or a structured action"
            )
        if type(self.max_images) is not int or self.max_images < 0:
            raise ValueError("max_images must be a nonnegative integer")
        for name, cost in self.costs.items():
            if (
                not name
                or name in {"tool_calls", "backend_segments", "final_answers"}
                or type(cost) is not int
                or cost < 0
            ):
                raise ValueError("Invalid submission accounting rule")

    def manifest(self) -> dict[str, Any]:
        return {
            "effect": self.effect.value,
            "max_images": self.max_images,
            "costs": thaw(self.costs),
        }


@dataclass(frozen=True)
class ToolExecution:
    """Transport-facing reply shared with the ARC runners."""

    text: str
    success: bool
    images: tuple[dict[str, Any], ...] = ()
    interrupt_after: bool = False
    boundary_reason: str | None = None
    images_before_interrupt: bool = False


class TaskSession(Protocol):
    """A host-owned task; only task and selected tool definitions reach the model."""

    @property
    def task(self) -> PublicTask: ...

    @property
    def tools(self) -> tuple[ToolBinding, ...]: ...

    @property
    def submission(self) -> SubmissionSpec: ...

    def submit(self, answer: str) -> ToolReply: ...

    def close(self) -> None: ...
