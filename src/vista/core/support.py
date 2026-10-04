"""Bind public task evidence to VISTA memory and visual operations."""

from __future__ import annotations

import base64
import copy
import hashlib
import threading
from functools import partial
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vista.core.dispatcher import GameToolDispatcher
from vista.core.session import (
    MAX_INSPECTION_QUESTION_CHARS,
    MAX_PIXEL_READOUT_SAMPLES,
    MAX_PIXEL_READOUT_VIEWS,
    PIXEL_READOUT_SYMBOLS,
    SessionController,
    atomic_write_text,
)
from vista.core.tools import build_tools

from .contracts import (
    Capability,
    Effect,
    EvidenceKind,
    Observation,
    ObservationBundle,
    PublicTask,
    ToolBinding,
    ToolReply,
    ToolSpec,
    Visual,
)
from .json import canonical

if TYPE_CHECKING:
    from .runtime import TaskRuntime


def quantize_rgb(color: tuple[int, int, int], bits: int) -> tuple[int, int, int]:
    """Round each channel to `bits` of precision, returning the level's color.

    Exact colors suit a paletted grid; on a compressed or anti-aliased image
    every sample is its own color and a symbol table fills in one call. Four
    bits keep sixteen levels per channel: enough to tell colors apart, few
    enough that JPEG noise collapses into them.
    """
    if bits >= 8:
        return color
    levels = (1 << bits) - 1

    def quantize(channel: int) -> int:
        level = (channel * levels + 127) // 255
        return (level * 255 + levels // 2) // levels

    return (quantize(color[0]), quantize(color[1]), quantize(color[2]))


class TaskContext(SessionController):
    """Public-state adapter for tasks without ARC states or actions."""

    # General benchmarks read screenshots, canvases and photographs, where
    # exact colors can exhaust a 64-symbol table in one call. This path rounds
    # to four bits per channel and uses a palette per reply; ARC3 controllers
    # use the base class's exact-color readout.
    pixel_readout_color_bits = 4
    pixel_readout_palette_scope = "call"

    def __init__(self, task: PublicTask, backend: str) -> None:
        self.task = task
        self.backend = backend
        self.runtime: TaskRuntime | None = None
        self.directory: Path | None = None
        self._lock = threading.RLock()
        self._retry_boundary_step = None
        self.retry_boundary_marker = None
        self.compact_checkpoint_marker = None
        self.compact_checkpoint_ready = None
        self.compact_restore_marker = None
        self.compact_guide_snapshot = None
        self.reset_starts_fresh_session = False
        self.forced_termination_reason = None
        self._pixel_readout_symbol_by_color = {}
        self._sources: dict[str, tuple[int, Visual]] = {}
        self._frame_paths: dict[int, list[Path]] = {}
        self._records: dict[int, dict[str, Any]] = {}
        self._moments: dict[str, tuple[int, int]] = {}
        self._current_ref: tuple[int, int] | None = None
        self._view_size: tuple[int, int] | None = None
        self._dispatcher = None
        self._calls = 0
        self.checkpoints = 0
        self.current = ObservationBundle()
        visuals = [
            view
            for observation in task.observation.observations
            for view in observation.views
        ]
        self.display_size = max(
            (max(view.width, view.height) for view in visuals), default=1024
        )
        self.observation_mode = "vision" if visuals else "none"
        self.tools = self._tools()

    def bind(self, runtime: TaskRuntime, directory: Path) -> None:
        if self.runtime is not None:
            raise RuntimeError("A task context can only be bound once")
        self.runtime = runtime
        self.directory = directory.resolve()
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.frames_dir = self.directory / "visuals"
        self.frames_dir.mkdir(mode=0o700)
        self.guide_path = self.directory / "GUIDE.md"
        self.working_path = self.directory / "WORKING.md"
        self.working_provenance_path = self.directory / "working_memory_provenance.json"
        self.recovery_log_path = self.directory / "compact_recovery.jsonl"
        self.compact_checkpoint_marker = self.directory / "compact_checkpoint.requested"
        self.compact_checkpoint_ready = self.directory / "compact_checkpoint.ready"
        self.compact_restore_marker = self.directory / "compact_restore.pending"
        self.compact_guide_snapshot = self.directory / "compact_guide.snapshot"
        self.retry_boundary_marker = self.directory / "retry_boundary.pending"
        atomic_write_text(self.guide_path, "No reliable model yet.\n", mode=0o600)
        self.record_event(self.task.observation, result=self.task.objective)

    def configure(self, visible: Path, checkpoint_dir: Path) -> None:
        for name in ("guide_path", "working_path"):
            source = getattr(self, name)
            target = visible / source.name
            if source.is_file():
                atomic_write_text(
                    target, source.read_text(encoding="utf-8"), mode=0o600
                )
            setattr(self, name, target)
        self.compact_checkpoint_marker = checkpoint_dir / "compact_checkpoint.requested"
        self.compact_checkpoint_ready = checkpoint_dir / "compact_checkpoint.ready"
        self.compact_restore_marker = checkpoint_dir / "compact_restore.pending"
        self.compact_guide_snapshot = checkpoint_dir / "compact_guide.snapshot"
        self.retry_boundary_marker = checkpoint_dir / "retry_boundary.pending"
        self._dispatcher = None

    @property
    def terminal(self) -> bool:
        return bool(
            self.forced_termination_reason or (self.runtime and self.runtime.terminal)
        )

    @property
    def step_index(self) -> int:
        return self.runtime.state_revision if self.runtime is not None else 0

    @property
    def max_steps(self) -> int:
        if self.runtime is None:
            return 2000
        budgets = self.runtime.experiment.budgets
        bounds = [
            budgets[meter] // cost
            for spec in self.runtime.specs
            if spec.effect == Effect.ENVIRONMENT
            for meter, cost in spec.costs.items()
            if cost and meter in budgets
        ]
        return max(1, min(bounds)) if bounds else budgets.get("backend_segments", 2000)

    @property
    def retry_boundary_pending(self) -> bool:
        return self._retry_boundary_step is not None

    @property
    def retry_boundary_step(self) -> int | None:
        return self._retry_boundary_step

    @property
    def public_events(self) -> list[dict[str, Any]]:
        return [self._public_event(event) for event in self._records]

    @property
    def _events(self) -> list[dict[str, Any]]:
        # The recovery port expects action results, not input deliveries.
        return [
            self._public_event(event)
            for event, record in self._records.items()
            if record["environment"]
        ]

    def _public_event(self, event: int) -> dict[str, Any]:
        record = self._records[event]
        return {
            "event": event,
            "action": copy.deepcopy(record["action"]),
            "success": record["success"],
            "result": record["result"],
            "observations": self._event_observations(event),
        }

    def _event_observations(self, event: int) -> list[dict[str, Any]]:
        return [
            {
                "moment": index,
                "kind": "current"
                if (event, index) == self._current_ref
                else "historical",
                "views": [
                    {"view": view.view, "width": view.width, "height": view.height}
                    for view in observation.views
                ],
            }
            for index, observation in enumerate(
                self._records[event]["observation"].observations
            )
        ]

    def _current_manifest(self) -> list[dict[str, Any]]:
        if self._current_ref is None:
            return []
        event, moment = self._current_ref
        return [{"event": event, **self._event_observations(event)[moment]}]

    def _metadata(self, *, action_applied: bool = False) -> dict[str, Any]:
        return {"observation": self._current_manifest(), "terminal": self.terminal}

    def _attempt_history(self) -> dict[str, Any]:
        if not self._records:
            return {}
        return {
            "first_event": next(iter(self._records)),
            "latest_event": next(reversed(self._records)),
        }

    def _record_working_provenance(self) -> dict[str, Any]:
        provenance = {"observation": self._current_manifest()}
        atomic_write_text(
            self.working_provenance_path, canonical(provenance) + "\n", mode=0o600
        )
        return provenance

    def begin_compact_recovery(self):
        recovery = super().begin_compact_recovery()
        self.checkpoints += 1
        return recovery

    def complete_runtime_recovery(self, recovery, *, delivery="resumed_runtime_thread"):
        super().complete_runtime_recovery(recovery, delivery=delivery)
        self.runtime.acknowledge()

    def complete_compact_recovery(self, recovery, *, delivery="fresh_thread"):
        super().complete_compact_recovery(recovery, delivery=delivery)
        self.runtime.acknowledge()

    def complete_retry_recovery(self, recovery):
        super().complete_retry_recovery(recovery)
        self.runtime.acknowledge()

    def record_event(
        self,
        bundle: ObservationBundle,
        *,
        action: dict[str, Any] | None = None,
        result: str = "",
        success: bool = True,
        update_current: bool = True,
        environment: bool = False,
    ) -> int:
        observations = tuple(
            item for item in bundle.observations if item.kind != EvidenceKind.DERIVED
        )
        event = len(self._records)
        for observation in observations:
            previous_moment = self._moments.get(observation.moment)
            if previous_moment is not None:
                previous_event, index = previous_moment
                previous = self._records[previous_event]["observation"].observations[
                    index
                ]
                if previous.views != observation.views:
                    raise ValueError(
                        "A supplied observation identity cannot change its views"
                    )
            for visual in observation.views:
                previous = self._sources.get(visual.source_id)
                if previous is not None:
                    if previous[1].data != visual.data:
                        raise ValueError(
                            "A supplied visual identity cannot change its pixels"
                        )
                    continue
                index = len(self._sources)
                suffix = {
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                    "image/webp": ".webp",
                    "image/gif": ".gif",
                }[visual.mime_type]
                path = self.frames_dir / f"turn_{index:06d}_frame_0{suffix}"
                path.write_bytes(visual.data)
                path.chmod(0o400)
                self._sources[visual.source_id] = (index, visual)
                self._frame_paths[index] = [path]
        stored = ObservationBundle(observations)
        self._records[event] = {
            "action": copy.deepcopy(action),
            "result": result,
            "success": success,
            "observation": stored,
            "environment": environment,
        }
        for index, observation in enumerate(observations):
            self._moments.setdefault(observation.moment, (event, index))
            if update_current and observation.kind == EvidenceKind.CURRENT:
                self.current = ObservationBundle((observation,))
                self._current_ref = event, index
        if observations:
            self.observation_mode = "vision"
        self.runtime.artifacts.record(
            "evidence_event",
            {
                "reference": event,
                "action": action,
                "result": result,
                "success": success,
                "observation": stored.manifest(),
                "environment": environment,
            },
        )
        return event

    def observe(
        self, bundle: ObservationBundle, *, update_current: bool = True
    ) -> None:
        if self.directory is None:
            return
        unseen = tuple(
            item
            for item in bundle.observations
            if item.kind != EvidenceKind.DERIVED and item.moment not in self._moments
        )
        for observation in bundle.observations:
            if observation.kind == EvidenceKind.DERIVED or observation in unseen:
                continue
            event, index = self._moments[observation.moment]
            previous = self._records[event]["observation"].observations[index]
            if previous.views != observation.views:
                raise ValueError(
                    "A supplied observation identity cannot change its views"
                )
        if unseen:
            self.record_event(ObservationBundle(unseen), update_current=update_current)
        if update_current:
            for observation in bundle.observations:
                if observation.kind == EvidenceKind.CURRENT:
                    if not self.current.observations or (
                        self.current.observations[0].moment != observation.moment
                    ):
                        self._current_ref = self._moments[observation.moment]
                    self.current = ObservationBundle((observation,))

    def _select(self, selection: dict[str, Any]) -> tuple[dict[str, Any], Visual]:
        event = selection["event"]
        if type(event) is not int:
            raise ValueError("Event must be an integer.")
        record = self._records.get(event)
        if record is None:
            raise ValueError("Unknown event.")
        observations = record["observation"].observations
        moment = selection.get("moment", len(observations) - 1)
        if type(moment) is not int or not 0 <= moment < len(observations):
            raise ValueError("The event has no such observation.")
        views = observations[moment].views
        name = selection.get("view")
        if name is None:
            if len(views) != 1:
                raise ValueError("Specify view for a multi-view observation.")
            visual = views[0]
        else:
            visual = next((view for view in views if view.view == name), None)
            if visual is None:
                raise ValueError("The observation has no such view.")
        return {"event": event, "moment": moment, "view": visual.view}, visual

    def _reference(self, observation: Observation, *, event: int | None = None):
        if event is not None:
            moment = next(
                index
                for index, item in enumerate(
                    self._records[event]["observation"].observations
                )
                if item.moment == observation.moment
            )
            return event, moment
        if (
            observation.kind == EvidenceKind.CURRENT
            and self.current.observations
            and self.current.observations[0].moment == observation.moment
        ):
            return self._current_ref
        return self._moments[observation.moment]

    def _origins(self, visual: Visual) -> list[dict[str, Any]]:
        if all(key in visual.transform for key in ("event", "moment", "view")):
            reference, parent = self._select(dict(visual.transform))
            if visual.derived_from != (parent.source_id,):
                raise ValueError("A derived view must reference its supplied original")
            return [reference]
        origins = []
        for source in visual.derived_from:
            match = next(
                (
                    {"event": event, "moment": index, "view": view.view}
                    for event, record in self._records.items()
                    for index, observation in enumerate(
                        record["observation"].observations
                    )
                    for view in observation.views
                    if view.source_id == source
                ),
                None,
            )
            if match is None:
                raise ValueError("Unknown source evidence for a derived visual")
            origins.append(match)
        return origins

    def labels(
        self, bundle: ObservationBundle, *, event: int | None = None
    ) -> tuple[str, ...]:
        labels = []
        for observation in bundle.displayed_observations:
            for visual in observation.views:
                if observation.kind == EvidenceKind.DERIVED:
                    origins = self._origins(visual)
                    attributes = {"kind": "derived"}
                    if len(origins) == 1:
                        attributes.update(origins[0])
                    else:
                        attributes["origins"] = canonical(origins)
                    for key in ("label", "region", "resampling"):
                        if key in visual.transform:
                            value = visual.transform[key]
                            attributes[key] = (
                                canonical(value) if key == "region" else value
                            )
                else:
                    ref = self._reference(observation, event=event)
                    attributes = {
                        "kind": "current"
                        if observation.kind == EvidenceKind.CURRENT
                        and ref == self._current_ref
                        else "historical",
                        "event": ref[0],
                        "moment": ref[1],
                        "view": visual.view,
                    }
                attributes["size"] = f"{visual.width}x{visual.height}"
                label = (
                    "<visual "
                    + " ".join(
                        f'{key}="{escape(str(value), quote=True)}"'
                        for key, value in attributes.items()
                    )
                    + "></visual>"
                )
                labels.append(label)
        return tuple(labels)

    def images(self, bundle: ObservationBundle, *, event: int | None = None):
        return tuple(
            {**image, "steer_text": label}
            for image, label in zip(
                bundle.images(), self.labels(bundle, event=event), strict=True
            )
        )

    def _selected_images(self) -> list[Path]:
        return [
            self._frame_paths[self._sources[view.source_id][0]][-1]
            for observation in self.current.observations
            for view in observation.views
        ]

    def _turn_frame_paths(self, turn: int) -> list[Path]:
        return self._frame_paths.get(turn, [])

    def visual_size(self) -> tuple[int, int]:
        return self._view_size or (self.display_size, self.display_size)

    def _with_view_size(self, selection, index, render):
        turn = selection.get("turn") if isinstance(selection, dict) else None
        match = next(
            (view for number, view in self._sources.values() if number == turn), None
        )
        previous = self._view_size
        self._view_size = (match.width, match.height) if match is not None else None
        try:
            return render(selection, index)
        finally:
            self._view_size = previous

    def _inspection_view(self, selection, index):
        return self._with_view_size(selection, index, super()._inspection_view)

    def _pixel_readout_view(self, selection, index):
        return self._with_view_size(selection, index, super()._pixel_readout_view)

    def _read_pixels(self, arguments: object) -> dict[str, Any]:
        """Read quantized colors with a palette scoped to this call."""
        if self.observation_mode not in {"vision", "both"}:
            return self._inspect_error("Visual observations are unavailable.")
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"question", "views"}
            or not isinstance(arguments.get("question"), str)
            or not arguments["question"].strip()
            or len(arguments["question"]) > MAX_INSPECTION_QUESTION_CHARS
            or not isinstance(arguments.get("views"), list)
            or not 1 <= len(arguments["views"]) <= MAX_PIXEL_READOUT_VIEWS
        ):
            return self._inspect_error("Invalid arguments.")
        sampled_views = []
        total_samples = 0
        bits = self.pixel_readout_color_bits
        for index, view in enumerate(arguments["views"], start=1):
            try:
                metadata, samples = self._pixel_readout_view(view, index)
            except ValueError as exc:
                return self._inspect_error(str(exc))
            total_samples += metadata["rows"] * metadata["columns"]
            if total_samples > MAX_PIXEL_READOUT_SAMPLES:
                return self._inspect_error(
                    f"At most {MAX_PIXEL_READOUT_SAMPLES} samples may be requested."
                )
            if bits is not None:
                samples = [
                    [quantize_rgb(color, bits) for color in row] for row in samples
                ]
            sampled_views.append((metadata, samples))
        table = (
            self._pixel_readout_symbol_by_color
            if self.pixel_readout_palette_scope == "session"
            else {}
        )
        colors = {
            color for _, samples in sampled_views for row in samples for color in row
        }
        unseen_colors = sorted(colors.difference(table))
        if len(table) + len(unseen_colors) > len(PIXEL_READOUT_SYMBOLS):
            return self._inspect_error(
                "The pixel readout color limit has been reached."
            )
        for color in unseen_colors:
            table[color] = PIXEL_READOUT_SYMBOLS[len(table)]
        palette = {
            symbol: f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
            for color, symbol in table.items()
            if color in colors
        }
        metadata_views = []
        for metadata, samples in sampled_views:
            metadata["samples"] = [
                "".join(table[color] for color in row) for row in samples
            ]
            metadata_views.append(metadata)
        return {
            "ok": True,
            "metadata": {
                "visual_kind": "pixel_readout",
                "current_state_unchanged": True,
                "sampling": "equal-bin-centers",
                "color_bits": bits,
                "palette_scope": self.pixel_readout_palette_scope,
                "palette": palette,
                "sample_count": total_samples,
                "view_count": len(metadata_views),
                "views": metadata_views,
            },
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if request["method"] == "save_compact_checkpoint":
                return self._save_compact_checkpoint(request["arguments"])
            raise ValueError("The context port does not execute environment actions")

    def _memory_reply(self, name: str, arguments: Any) -> ToolReply:
        if self._dispatcher is None:
            self._dispatcher = GameToolDispatcher(
                controller=self,
                guide_path=self.guide_path,
                working_path=self.working_path,
                checkpoint_via_working=self.backend == "claude",
            )
        self._calls += 1
        execution = self._dispatcher.execute(name, arguments, f"memory-{self._calls}")
        return ToolReply(
            execution.text,
            success=execution.success,
            boundary=execution.boundary_reason,
        )

    def _visual_reply(self, name: str, arguments: dict[str, Any]) -> ToolReply:
        selections = []
        references = []
        sources = []
        for selection in arguments["views"]:
            try:
                reference, visual = self._select(selection)
            except ValueError as exc:
                return ToolReply(str(exc), success=False)
            number, _ = self._sources[visual.source_id]
            selections.append(
                {
                    key: value
                    for key, value in selection.items()
                    if key not in {"event", "moment", "view"}
                }
                | {"turn": number}
            )
            references.append(reference)
            sources.append(visual.source_id)
        request = {"question": arguments["question"], "views": selections}
        response = (
            self._inspect(request) if name == "inspect" else self._read_pixels(request)
        )
        metadata = response["metadata"]
        if not response.get("ok"):
            return ToolReply(canonical(metadata), success=False)
        observations = []
        for index, view in enumerate(metadata["views"]):
            for key in ("turn", "frame", "available_frames", "available_frame_range"):
                view.pop(key, None)
            view.update(references[index])
            view["label"] = arguments["views"][index]["label"]
            if name == "inspect":
                data = base64.b64decode(
                    response["images"][index]["data"], validate=True
                )
                identity = (
                    "inspect-"
                    + hashlib.sha256(canonical(view).encode() + data).hexdigest()
                )
                visual = Visual(
                    "crop",
                    identity,
                    data,
                    view["image_size"]["width"],
                    view["image_size"]["height"],
                    derived_from=(sources[index],),
                    transform={
                        **references[index],
                        "label": view["label"],
                        "region": view["region"],
                        "resampling": "nearest",
                    },
                )
                observations.append(
                    Observation(identity, (visual,), EvidenceKind.DERIVED)
                )
        return ToolReply(
            canonical(metadata), observation=ObservationBundle(tuple(observations))
        )

    def _history_reply(self, arguments: dict[str, Any]) -> ToolReply:
        events = self.public_events
        start, end = arguments.get("start"), arguments.get("end")
        for name, value in (("start", start), ("end", end)):
            if value is not None and (
                type(value) is not int or not 0 <= value <= len(events)
            ):
                return ToolReply(f"Invalid history {name} event number.", success=False)
        first = start if start is not None else 0
        stop = end if end is not None else len(events)
        if stop < first:
            return ToolReply("History end must not precede its start.", success=False)
        selected = events[first:stop]
        limit = arguments.get("limit", 64)
        if type(limit) is not int:
            return ToolReply("History limit must be an integer.", success=False)
        bounded = start is not None or end is not None
        return ToolReply(
            canonical(
                {
                    "events": selected[:limit] if bounded else selected[-limit:],
                    "truncated": len(selected) > limit,
                }
            )
        )

    def _tools(self) -> tuple[ToolBinding, ...]:
        bindings = []
        for original in build_tools(
            self.display_size, include_compact_checkpoint=self.backend == "codex"
        ):
            definition = copy.deepcopy(original)
            name = definition["name"]
            if name == "play":
                continue
            if name == "read_working":
                definition["description"] = (
                    "Read agent-authored temporary state for the current task."
                )
            elif name == "write_working":
                definition["description"] = (
                    "Replace agent-authored temporary state for the current task. It persists until explicitly replaced or cleared by the task lifecycle."
                )
            elif name == "save_compact_checkpoint":
                definition["description"] = definition["description"].replace(
                    "turn/frame", "event/moment/view"
                )
            schema = definition["inputSchema"]
            if name in {"inspect", "read_pixels"}:
                item = schema["properties"]["views"]["items"]
                item["required"] = [
                    "event" if key == "turn" else key for key in item["required"]
                ]
                item["properties"].pop("turn")
                item["properties"].pop("frame")
                item["properties"].update(
                    {
                        "event": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "This role's event number, starting at 0, shown with a supplied visual or in history.",
                        },
                        "moment": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Observation within the event; omit for the final observation.",
                        },
                        "view": {
                            "type": "string",
                            "minLength": 1,
                            "description": "View name; required only when the observation has multiple views.",
                        },
                    }
                )
                definition["description"] = (
                    "Inspect one or more supplied visuals without changing the task. "
                    "Select an event, optionally an earlier moment and a view name. "
                    "Include a visual question and a label for each view. Regions use original image pixels; "
                    f"crops are enlarged proportionally without smoothing to fit within {self.display_size}x{self.display_size}."
                    if name == "inspect"
                    else "Read RGB samples from one or more supplied visuals without changing the task. "
                    "Select an event, optionally a moment and view name, and an original-image pixel region. The tool divides the region "
                    "into equal rows and columns and samples their center pixels. Colors are rounded to 4 bits "
                    "per channel (16 levels each) so compression noise does not fragment the palette; each reply "
                    "carries its own palette of at most 64 colors as symbols, with compact row strings, without "
                    "interpreting pixels. At most 64 views and 4096 total samples may be requested. Include a "
                    "visual question and labels."
                )
                capability = (
                    Capability.INSPECTION if name == "inspect" else Capability.PIXELS
                )
                handler = partial(self._visual_reply, name)
            elif name == "history":
                capability, handler = Capability.HISTORY, self._history_reply
            else:
                capability = (
                    Capability.CHECKPOINT
                    if name == "save_compact_checkpoint"
                    else Capability.MEMORY
                )
                handler = partial(self._memory_reply, name)
            effect = (
                Effect.MEMORY
                if name.startswith("write_") or name == "save_compact_checkpoint"
                else Effect.NONE
            )
            bindings.append(
                ToolBinding(
                    ToolSpec(
                        name,
                        definition["description"],
                        schema,
                        capability,
                        effect,
                        max_images=16 if name == "inspect" else 0,
                        annotations=definition["annotations"],
                    ),
                    handler,
                )
            )
        return tuple(bindings)
