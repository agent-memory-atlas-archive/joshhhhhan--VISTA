"""Checkpoint and visual-evidence operations shared by task controllers."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

MAX_GUIDE_CHARS = 64 * 1024
MAX_WORKING_CHARS = 16 * 1024
MAX_INSPECTION_VIEWS = 16
MAX_INSPECTION_QUESTION_CHARS = 1024
MAX_INSPECTION_LABEL_CHARS = 128
MAX_PIXEL_READOUT_VIEWS = 64
MAX_PIXEL_READOUT_SAMPLES = 4096
PIXEL_READOUT_SYMBOLS = (
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
)


@dataclass(frozen=True)
class CompactRecovery:
    guide: str
    working_memory: str
    current_observation: dict[str, Any]
    last_action_result: dict[str, Any] | None
    objective_history: dict[str, Any]
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class RetryRecovery:
    boundary_step: int
    guide: str | None
    working_memory: str | None
    working_provenance: dict[str, Any] | None
    current_observation: dict[str, Any]
    last_action_result: dict[str, Any] | None
    objective_history: dict[str, Any]
    image_paths: tuple[Path, ...]


@dataclass(frozen=True)
class RuntimeRecovery:
    boundary_step: int
    current_observation: dict[str, Any]
    last_action_result: dict[str, Any] | None
    image_paths: tuple[Path, ...]
    guide: str | None = None
    working_memory: str | None = None
    working_provenance: dict[str, Any] | None = None
    objective_history: dict[str, Any] = field(default_factory=dict)


class SessionController:
    def _atomic_write_text(self, path: Path, content: str, *, mode: int) -> None:
        atomic_write_text(path, content, mode=mode)

    def visual_size(self) -> tuple[int, int]:
        return self.display_size, self.display_size

    def initial_metadata(self) -> dict[str, Any]:
        return self._metadata(action_applied=False)

    def initial_image_paths(self) -> list[Path]:
        if self.observation_mode not in {"vision", "both"}:
            return []
        return self._selected_images()

    def request_fresh_recovery(self) -> None:
        """Start a fresh player from the current, unchanged game state."""
        with self._lock:
            if self.terminal:
                raise RuntimeError("Cannot recover a terminal game.")
            if self._retry_boundary_step is not None:
                raise RuntimeError("A fresh recovery is already pending.")
            self._retry_boundary_step = self.step_index
            self._mark_retry_boundary()

    def begin_runtime_recovery(self) -> RuntimeRecovery:
        """Snapshot the authoritative game state after a player-runtime failure."""
        with self._lock:
            if self.terminal:
                raise RuntimeError("Cannot recover a terminal game.")
            images = (
                tuple(self._selected_images())
                if self.observation_mode in {"vision", "both"}
                else ()
            )
            if self.observation_mode in {"vision", "both"} and not images:
                raise RuntimeError("The current final visual is missing.")
            guide = None
            if self.guide_path is not None and self.guide_path.is_file():
                guide = self.guide_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            working_memory = None
            if self.working_path is not None and self.working_path.is_file():
                working_memory = self.working_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            return RuntimeRecovery(
                boundary_step=self.step_index,
                guide=guide,
                working_memory=working_memory,
                working_provenance=self._read_working_provenance(),
                current_observation=self._metadata(action_applied=False),
                last_action_result=(dict(self._events[-1]) if self._events else None),
                objective_history=self._attempt_history(),
                image_paths=images,
            )

    def complete_runtime_recovery(
        self,
        recovery: RuntimeRecovery,
        *,
        delivery: str = "resumed_runtime_thread",
    ) -> None:
        with self._lock:
            if self.terminal or recovery.boundary_step != self.step_index:
                raise RuntimeError("Runtime recovery is no longer current.")
            if delivery not in {
                "resumed_runtime_thread",
                "fresh_runtime_thread",
                "fresh_continuation_thread",
                "same_session_native_compact",
            }:
                raise ValueError("Unknown runtime recovery delivery.")
            append_jsonl(
                self.recovery_log_path,
                {
                    "step": self.step_index,
                    "action_applied": False,
                    "delivery": delivery,
                },
            )

    def discard_compact_recovery(self) -> None:
        with self._lock:
            self._clear_compact_recovery_files()
            self._clear_retry_boundary_marker()

    def begin_retry_recovery(self) -> RetryRecovery:
        with self._lock:
            if self._retry_boundary_step is None:
                raise RuntimeError("No retry boundary is pending.")
            self._clear_compact_recovery_files()
            images = (
                tuple(self._selected_images())
                if self.observation_mode in {"vision", "both"}
                else ()
            )
            if self.observation_mode in {"vision", "both"} and not images:
                raise RuntimeError("The reset visual is missing.")
            guide = None
            if self.guide_path is not None and self.guide_path.is_file():
                guide = self.guide_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            working_memory = None
            if self.working_path is not None and self.working_path.is_file():
                working_memory = self.working_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            return RetryRecovery(
                boundary_step=self._retry_boundary_step,
                guide=guide,
                working_memory=working_memory,
                working_provenance=self._read_working_provenance(),
                current_observation=self._metadata(action_applied=False),
                last_action_result=(dict(self._events[-1]) if self._events else None),
                objective_history=self._attempt_history(),
                image_paths=images,
            )

    def complete_retry_recovery(self, recovery: RetryRecovery) -> None:
        with self._lock:
            if (
                self._retry_boundary_step is None
                or recovery.boundary_step != self._retry_boundary_step
                or recovery.boundary_step != self.step_index
            ):
                raise RuntimeError("Retry recovery is no longer pending.")
            current_guide = None
            if self.guide_path is not None and self.guide_path.is_file():
                current_guide = self.guide_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            if current_guide != recovery.guide:
                raise RuntimeError("Retry recovery guide changed before delivery.")
            current_working = None
            if self.working_path is not None and self.working_path.is_file():
                current_working = self.working_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            if (
                current_working != recovery.working_memory
                or self._read_working_provenance() != recovery.working_provenance
            ):
                raise RuntimeError(
                    "Retry recovery working state changed before delivery."
                )
            self._retry_boundary_step = None
            self._clear_compact_recovery_files()
            self._clear_retry_boundary_marker()
            append_jsonl(
                self.recovery_log_path,
                {
                    "step": self.step_index,
                    "action_applied": False,
                    "delivery": "fresh_retry_thread",
                },
            )

    def request_compact_checkpoint(self, *, reason: str) -> None:
        """Request an agent-authored checkpoint without changing game state."""

        with self._lock:
            if self.terminal:
                raise RuntimeError("Cannot checkpoint a terminal game.")
            if (
                self.compact_checkpoint_marker is None
                or self.compact_checkpoint_ready is None
                or self.compact_restore_marker is None
                or self.compact_guide_snapshot is None
            ):
                raise RuntimeError("Compact checkpoint paths are unavailable.")
            if self.compact_restore_marker.exists():
                raise RuntimeError("Compact recovery is already pending.")
            if self.compact_checkpoint_ready.exists():
                raise RuntimeError("A stale compact checkpoint is already present.")
            self.compact_checkpoint_marker.parent.mkdir(parents=True, exist_ok=True)
            self.compact_checkpoint_marker.touch(mode=0o600, exist_ok=True)
            append_jsonl(
                self.recovery_log_path,
                {
                    "step": self.step_index,
                    "action_applied": False,
                    "delivery": "checkpoint_requested",
                    "reason": reason,
                },
            )

    def complete_compact_checkpoint_from_working(self) -> None:
        """Mark an already-written WORKING.md as ready for fresh-thread recovery."""

        with self._lock:
            if (
                self.compact_checkpoint_marker is None
                or not self.compact_checkpoint_marker.is_file()
                or self.compact_checkpoint_ready is None
                or self.working_path is None
                or not self.working_path.is_file()
                or self.guide_path is None
                or not self.guide_path.is_file()
                or not self.working_provenance_path.is_file()
            ):
                raise RuntimeError("Compact checkpoint data is incomplete.")
            if self.compact_checkpoint_ready.exists():
                return
            working = self.working_path.read_text(encoding="utf-8", errors="replace")
            guide = self.guide_path.read_text(encoding="utf-8", errors="replace")
            if not working.strip() or not guide.strip():
                raise RuntimeError("Compact checkpoint data is empty.")
            append_jsonl(
                self.recovery_log_path,
                {
                    "step": self.step_index,
                    "action_applied": False,
                    "delivery": "checkpoint_saved_via_working",
                    "guide_chars": len(guide),
                    "working_chars": len(working),
                },
            )
            self.compact_checkpoint_ready.parent.mkdir(parents=True, exist_ok=True)
            self.compact_checkpoint_ready.touch(mode=0o600)

    def begin_compact_recovery(self) -> CompactRecovery:
        with self._lock:
            try:
                required_paths = (
                    self.compact_checkpoint_marker,
                    self.compact_checkpoint_ready,
                    self.working_path,
                    self.guide_path,
                )
                if any(path is None or not path.is_file() for path in required_paths):
                    raise RuntimeError("Compact checkpoint data is incomplete.")
                if (
                    self.compact_restore_marker is None
                    or self.compact_guide_snapshot is None
                ):
                    raise RuntimeError("Compact recovery paths are unavailable.")

                assert self.guide_path is not None
                assert self.working_path is not None
                guide = self.guide_path.read_text(encoding="utf-8", errors="replace")
                working = self.working_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                if not guide.strip() or not working.strip():
                    raise RuntimeError("Compact checkpoint data is empty.")

                self._atomic_write_text(
                    self.compact_guide_snapshot,
                    guide,
                    mode=0o600,
                )
                self.compact_restore_marker.parent.mkdir(parents=True, exist_ok=True)
                self.compact_restore_marker.touch(mode=0o600)
                images = (
                    tuple(self._selected_images())
                    if self.observation_mode in {"vision", "both"}
                    else ()
                )
                if self.observation_mode in {"vision", "both"} and not images:
                    raise RuntimeError("The current final visual is missing.")
                return CompactRecovery(
                    guide=guide,
                    working_memory=working,
                    current_observation=self._metadata(action_applied=False),
                    last_action_result=(
                        dict(self._events[-1]) if self._events else None
                    ),
                    objective_history=self._attempt_history(),
                    image_paths=images,
                )
            except (OSError, RuntimeError) as exc:
                self.forced_termination_reason = "compact_recovery_failed"
                append_jsonl(
                    self.recovery_log_path,
                    {
                        "step": self.step_index,
                        "action_applied": False,
                        "delivery": "prepare",
                        "error": str(exc),
                    },
                )
                raise RuntimeError("Compact recovery data is unavailable.") from exc

    def complete_compact_recovery(
        self,
        recovery: CompactRecovery,
        *,
        delivery: str = "fresh_thread",
    ) -> None:
        with self._lock:
            try:
                if (
                    self.compact_restore_marker is None
                    or not self.compact_restore_marker.is_file()
                    or self.compact_guide_snapshot is None
                    or not self.compact_guide_snapshot.is_file()
                    or self.working_path is None
                    or not self.working_path.is_file()
                ):
                    raise RuntimeError("Compact recovery is no longer pending.")
                guide = self.compact_guide_snapshot.read_text(
                    encoding="utf-8", errors="replace"
                )
                working = self.working_path.read_text(
                    encoding="utf-8", errors="replace"
                )
                if guide != recovery.guide or working != recovery.working_memory:
                    raise RuntimeError("Compact recovery data changed before delivery.")

                append_jsonl(
                    self.recovery_log_path,
                    {
                        "step": self.step_index,
                        "action_applied": False,
                        "delivery": delivery,
                        "guide_chars": len(guide),
                        "working_chars": len(working),
                    },
                )
                self._clear_compact_recovery_files()
            except (OSError, RuntimeError) as exc:
                self.forced_termination_reason = "compact_recovery_failed"
                raise RuntimeError("Compact recovery could not be completed.") from exc

    def _save_compact_checkpoint(self, arguments: object) -> dict[str, Any]:
        if (
            self.compact_checkpoint_marker is None
            or not self.compact_checkpoint_marker.exists()
            or self.compact_checkpoint_ready is None
            or self.working_path is None
        ):
            return {
                "ok": False,
                "metadata": {"error": "No compact checkpoint is pending."},
            }
        if self.compact_checkpoint_ready.exists():
            if not self.working_path.is_file():
                return {
                    "ok": False,
                    "metadata": {"error": "Compact checkpoint is incomplete."},
                }
            return {
                "ok": True,
                "metadata": {
                    "checkpoint_saved": True,
                    "already_saved": True,
                },
            }
        if not isinstance(arguments, dict) or set(arguments) != {
            "guide",
            "working_memory",
        }:
            return {"ok": False, "metadata": {"error": "Invalid arguments."}}
        guide = arguments.get("guide")
        working_memory = arguments.get("working_memory")
        if guide is not None and (
            not isinstance(guide, str)
            or not guide.strip()
            or len(guide) > MAX_GUIDE_CHARS
            or self.guide_path is None
        ):
            return {
                "ok": False,
                "metadata": {"error": "Invalid GUIDE.md content."},
            }
        if (
            not isinstance(working_memory, str)
            or not working_memory.strip()
            or len(working_memory) > MAX_WORKING_CHARS
        ):
            return {
                "ok": False,
                "metadata": {"error": "Invalid WORKING.md content."},
            }

        try:
            if guide is not None:
                self._atomic_write_text(
                    self.guide_path, guide.strip() + "\n", mode=0o600
                )
            if self.working_provenance_path.exists():
                self.working_provenance_path.unlink()
            self._atomic_write_text(
                self.working_path,
                working_memory.strip() + "\n",
                mode=0o600,
            )
            self._record_working_provenance()
            self.compact_checkpoint_ready.parent.mkdir(parents=True, exist_ok=True)
            self.compact_checkpoint_ready.touch(mode=0o600)
        except OSError:
            return {
                "ok": False,
                "metadata": {"error": "Compact checkpoint could not be saved."},
            }
        return {
            "ok": True,
            "metadata": {
                "checkpoint_saved": True,
                "guide_updated": guide is not None,
                "guide_chars": len(guide.strip()) if guide is not None else 0,
                "working_chars": len(working_memory.strip()),
            },
        }

    def _clear_compact_recovery_files(self) -> None:
        for path in (
            self.compact_guide_snapshot,
            self.compact_restore_marker,
            self.compact_checkpoint_ready,
            self.compact_checkpoint_marker,
        ):
            if path is not None and path.exists():
                path.unlink()

    def _clear_working(self) -> None:
        if self.working_path is not None and self.working_path.exists():
            self.working_path.unlink()
        if self.working_provenance_path.exists():
            self.working_provenance_path.unlink()

    def record_working_write(self) -> dict[str, Any]:
        with self._lock:
            return self._record_working_provenance()

    def _read_working_provenance(self) -> dict[str, Any] | None:
        try:
            value = json.loads(
                self.working_provenance_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def _mark_retry_boundary(self) -> None:
        if self.retry_boundary_marker is None:
            return
        self.retry_boundary_marker.parent.mkdir(parents=True, exist_ok=True)
        self.retry_boundary_marker.touch(mode=0o600)

    def _clear_retry_boundary_marker(self) -> None:
        if (
            self.retry_boundary_marker is not None
            and self.retry_boundary_marker.exists()
        ):
            self.retry_boundary_marker.unlink()

    def _inspect(self, arguments: object) -> dict[str, Any]:
        if self.observation_mode not in {"vision", "both"}:
            return self._inspect_error("Visual observations are unavailable.")
        if (
            not isinstance(arguments, dict)
            or set(arguments) != {"question", "views"}
            or not isinstance(arguments.get("question"), str)
            or not arguments["question"].strip()
            or len(arguments["question"]) > MAX_INSPECTION_QUESTION_CHARS
            or not isinstance(arguments.get("views"), list)
            or not 1 <= len(arguments["views"]) <= MAX_INSPECTION_VIEWS
        ):
            return self._inspect_error("Invalid arguments.")

        metadata_views: list[dict[str, Any]] = []
        images: list[dict[str, Any]] = []
        for index, view in enumerate(arguments["views"], start=1):
            try:
                metadata, image = self._inspection_view(view, index)
            except ValueError as exc:
                return self._inspect_error(str(exc))
            metadata_views.append(metadata)
            images.append(image)

        return {
            "ok": True,
            "metadata": {
                "visual_kind": "inspection",
                "current_state_unchanged": True,
                "view_count": len(metadata_views),
                "views": metadata_views,
            },
            "images": images,
        }

    def _inspection_view(
        self,
        selection: object,
        index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            not isinstance(selection, dict)
            or not isinstance(selection.get("label"), str)
            or not selection["label"].strip()
            or len(selection["label"]) > MAX_INSPECTION_LABEL_CHARS
            or "turn" not in selection
            or set(selection) - {"label", "turn", "frame", "region"}
        ):
            raise ValueError(f"View {index}: invalid selection.")

        turn = selection.get("turn")
        frame = selection.get("frame")
        if type(turn) is not int or not 0 <= turn <= 999999:
            raise ValueError(f"View {index}: invalid turn.")
        if "frame" in selection and (
            type(frame) is not int or not 0 <= frame <= 999999
        ):
            raise ValueError(f"View {index}: invalid frame.")
        if "region" in selection and selection.get("region") is None:
            raise ValueError(f"View {index}: invalid region.")
        try:
            region = self._inspection_region(selection.get("region"))
        except ValueError as exc:
            raise ValueError(f"View {index}: invalid region.") from exc

        paths = self._turn_frame_paths(turn)
        if not paths:
            raise ValueError(f"View {index}: no frames found for turn {turn}.")
        selected_path = paths[-1]
        if frame is not None:
            matching = [path for path in paths if self._frame_number(path) == frame]
            if not matching:
                raise ValueError(
                    f"View {index}: no frame {frame} found for turn {turn}."
                )
            selected_path = matching[0]
        selected_frame = self._frame_number(selected_path)

        try:
            x, y, width, height = region
            with Image.open(selected_path) as archived:
                if archived.size != self.visual_size():
                    raise ValueError("Archived visual has an unexpected size")
                visual = archived.convert("RGB").crop((x, y, x + width, y + height))
            longest_side = max(width, height)
            rendered_width = width * self.display_size // longest_side
            rendered_height = height * self.display_size // longest_side
            if (rendered_width, rendered_height) != visual.size:
                image = visual.resize(
                    (rendered_width, rendered_height),
                    Image.Resampling.NEAREST,
                )
            else:
                image = visual
            output = io.BytesIO()
            image.save(output, format="PNG")
        except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"View {index}: archived visual could not be rendered."
            ) from exc

        metadata = {
            "index": index,
            "turn": turn,
            "frame": selected_frame,
            "available_frames": len(paths),
            "available_frame_range": [
                self._frame_number(paths[0]),
                self._frame_number(paths[-1]),
            ],
            "region": {"x": x, "y": y, "width": width, "height": height},
            "image_size": {"width": image.width, "height": image.height},
        }
        return (
            metadata,
            {
                "mime_type": "image/png",
                "detail": "original",
                "data": base64.b64encode(output.getvalue()).decode("ascii"),
            },
        )

    @staticmethod
    def _inspect_error(message: str) -> dict[str, Any]:
        return {"ok": False, "metadata": {"error": message}}

    def _read_pixels(self, arguments: object) -> dict[str, Any]:
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

        sampled_views: list[
            tuple[dict[str, Any], list[list[tuple[int, int, int]]]]
        ] = []
        total_samples = 0
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
            sampled_views.append((metadata, samples))

        colors = {
            color for _, samples in sampled_views for row in samples for color in row
        }
        unseen_colors = sorted(colors.difference(self._pixel_readout_symbol_by_color))
        if len(self._pixel_readout_symbol_by_color) + len(unseen_colors) > len(
            PIXEL_READOUT_SYMBOLS
        ):
            return self._inspect_error(
                "The pixel readout color limit has been reached."
            )
        for color in unseen_colors:
            self._pixel_readout_symbol_by_color[color] = PIXEL_READOUT_SYMBOLS[
                len(self._pixel_readout_symbol_by_color)
            ]
        palette = {
            symbol: f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
            for color, symbol in self._pixel_readout_symbol_by_color.items()
            if color in colors
        }
        palette_symbols = {
            color: self._pixel_readout_symbol_by_color[color] for color in colors
        }
        metadata_views: list[dict[str, Any]] = []
        for metadata, samples in sampled_views:
            metadata["samples"] = [
                "".join(palette_symbols[color] for color in row) for row in samples
            ]
            metadata_views.append(metadata)

        return {
            "ok": True,
            "metadata": {
                "visual_kind": "pixel_readout",
                "current_state_unchanged": True,
                "sampling": "equal-bin-centers",
                "palette": palette,
                "sample_count": total_samples,
                "view_count": len(metadata_views),
                "views": metadata_views,
            },
        }

    def _pixel_readout_view(
        self,
        selection: object,
        index: int,
    ) -> tuple[dict[str, Any], list[list[tuple[int, int, int]]]]:
        if (
            not isinstance(selection, dict)
            or not isinstance(selection.get("label"), str)
            or not selection["label"].strip()
            or len(selection["label"]) > MAX_INSPECTION_LABEL_CHARS
            or "turn" not in selection
            or "region" not in selection
            or set(selection) - {"label", "turn", "frame", "region", "rows", "columns"}
        ):
            raise ValueError(f"View {index}: invalid selection.")

        turn = selection.get("turn")
        frame = selection.get("frame")
        rows = selection.get("rows")
        columns = selection.get("columns")
        if type(turn) is not int or not 0 <= turn <= 999999:
            raise ValueError(f"View {index}: invalid turn.")
        if "frame" in selection and (
            type(frame) is not int or not 0 <= frame <= 999999
        ):
            raise ValueError(f"View {index}: invalid frame.")
        if (
            type(rows) is not int
            or type(columns) is not int
            or not 1 <= rows <= self.display_size
            or not 1 <= columns <= self.display_size
        ):
            raise ValueError(f"View {index}: invalid sample dimensions.")
        try:
            region = self._inspection_region(selection.get("region"))
        except ValueError as exc:
            raise ValueError(f"View {index}: invalid region.") from exc

        paths = self._turn_frame_paths(turn)
        if not paths:
            raise ValueError(f"View {index}: no frames found for turn {turn}.")
        selected_path = paths[-1]
        if frame is not None:
            matching = [path for path in paths if self._frame_number(path) == frame]
            if not matching:
                raise ValueError(
                    f"View {index}: no frame {frame} found for turn {turn}."
                )
            selected_path = matching[0]
        selected_frame = self._frame_number(selected_path)

        try:
            x, y, width, height = region
            sample_x = [
                x + ((2 * column + 1) * width) // (2 * columns)
                for column in range(columns)
            ]
            sample_y = [
                y + ((2 * row + 1) * height) // (2 * rows) for row in range(rows)
            ]
            with Image.open(selected_path) as archived:
                if archived.size != self.visual_size():
                    raise ValueError("Archived visual has an unexpected size")
                source = archived.convert("RGB")
                pixels = source.load()
                samples = [
                    [pixels[sample_column, sample_row] for sample_column in sample_x]
                    for sample_row in sample_y
                ]
        except (OSError, ValueError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                f"View {index}: archived visual could not be sampled."
            ) from exc

        return (
            {
                "index": index,
                "turn": turn,
                "frame": selected_frame,
                "available_frames": len(paths),
                "available_frame_range": [
                    self._frame_number(paths[0]),
                    self._frame_number(paths[-1]),
                ],
                "region": {"x": x, "y": y, "width": width, "height": height},
                "rows": rows,
                "columns": columns,
            },
            samples,
        )

    def _inspection_region(self, value: object) -> tuple[int, int, int, int]:
        if value is None:
            return 0, 0, *self.visual_size()
        if not isinstance(value, dict) or set(value) != {
            "x",
            "y",
            "width",
            "height",
        }:
            raise ValueError("Invalid region")
        x = value.get("x")
        y = value.get("y")
        width = value.get("width")
        height = value.get("height")
        if any(type(item) is not int for item in (x, y, width, height)):
            raise ValueError("Invalid region")
        assert isinstance(x, int) and isinstance(y, int)
        assert isinstance(width, int) and isinstance(height, int)
        if (
            x < 0
            or y < 0
            or width < 1
            or height < 1
            or x + width > self.visual_size()[0]
            or y + height > self.visual_size()[1]
        ):
            raise ValueError("Invalid region")
        return x, y, width, height

    def _turn_frame_paths(self, turn: int) -> list[Path]:
        return sorted(
            self.frames_dir.glob(f"turn_{turn:03d}_frame_*.png"),
            key=self._frame_number,
        )

    @staticmethod
    def _frame_number(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[-1])


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def append_jsonl(path: Path, data: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, separators=(",", ":"), sort_keys=True))
        handle.write("\n")


def atomic_write_text(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.chmod(mode)
    temporary.replace(path)
