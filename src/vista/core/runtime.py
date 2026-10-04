"""Serialized, allowlisted task execution shared by both CLI transports."""

from __future__ import annotations

import hashlib
import threading
import traceback
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from referencing import Registry

from .artifacts import ArtifactStore
from .contracts import (
    Capability,
    Effect,
    EvidenceKind,
    ObservationBundle,
    TaskSession,
    ToolExecution,
    ToolReply,
    ToolSpec,
)
from .json import canonical, thaw
from .profiles import CompactionMode, Experiment, Phase


class Status(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    BUDGET = "budget_exhausted"
    FAILED = "failed"
    STOPPED = "stopped"


class BudgetExhausted(RuntimeError):
    """A normal stopping condition, not a failed backend call."""


@dataclass(frozen=True)
class TaskInput:
    text: str
    observation: ObservationBundle = field(default_factory=ObservationBundle)
    phase: Phase = Phase.INITIAL
    context_key: str = "task"
    blocks: tuple[str | ObservationBundle, ...] = ()
    image_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "phase", Phase(self.phase))
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or not isinstance(self.context_key, str)
            or not self.context_key
        ):
            raise ValueError("A model input needs text and a context key")
        if not isinstance(self.observation, ObservationBundle):
            raise TypeError("A model input must use an ObservationBundle")
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "image_labels", tuple(self.image_labels))
        if self.image_labels and (
            len(self.image_labels)
            != sum(len(item.views) for item in self.observation.displayed_observations)
            or any(
                not isinstance(label, str) or not label for label in self.image_labels
            )
        ):
            raise ValueError("Image labels must match the supplied visuals in order")
        if self.blocks:
            observations = []
            displayed = []
            for block in self.blocks:
                if isinstance(block, ObservationBundle):
                    observations.extend(block.observations)
                    displayed.extend(block.displayed_observations)
                elif not isinstance(block, str) or not block.strip():
                    raise TypeError(
                        "Ordered input blocks must be nonempty text or observations"
                    )
            if (
                tuple(observations) != self.observation.observations
                or tuple(displayed) != self.observation.displayed_observations
            ):
                raise ValueError(
                    "Ordered blocks must contain the declared observations exactly once"
                )

    def block_manifest(self) -> list[dict[str, Any]]:
        return [
            {"text": block}
            if isinstance(block, str)
            else {"observation": block.manifest()}
            for block in self.blocks
        ]


class TaskRuntime:
    def __init__(
        self,
        session: TaskSession,
        experiment: Experiment,
        *,
        artifact_dir: Path,
    ) -> None:
        self.session = session
        self.experiment = experiment
        self.submission = session.submission
        from .support import TaskContext

        self.context = TaskContext(session.task, experiment.backend)
        support = tuple(
            binding
            for binding in self.context.tools
            if binding.spec.name in experiment.profile.tools
        )
        bindings = (*session.tools, *support)
        all_bindings = {binding.spec.name: binding for binding in bindings}
        if len(all_bindings) != len(bindings):
            raise ValueError("Duplicate task tool names")
        unknown = set(experiment.profile.tools) - all_bindings.keys()
        if unknown:
            raise ValueError(
                f"The profile requested unavailable tools: {sorted(unknown)}"
            )
        self._bindings = {name: all_bindings[name] for name in experiment.profile.tools}
        self.specs = tuple(binding.spec for binding in self._bindings.values())
        experiment.profile.validate_tools(self.specs)
        self._validators = {}
        for spec in self.specs:
            schema = thaw(spec.input_schema)
            Draft202012Validator.check_schema(schema)
            self._validators[spec.name] = Draft202012Validator(
                schema, registry=Registry()
            )
        meters = {"tool_calls", "backend_segments", "final_answers"}
        meters.update(meter for spec in self.specs for meter in spec.costs)
        meters.update(self.submission.costs)
        if set(experiment.budgets) - meters:
            raise ValueError(
                "A budget names a meter not declared by the selected tools"
            )
        self._usage: dict[str, int] = {}
        self._seen_calls: set[str] = set()
        self._pending: str | None = None
        self._lock = threading.RLock()
        self._status = Status.RUNNING
        self._closed = False
        self._failure: str | None = None
        self._boundary: str | None = None
        self._phase = Phase.INITIAL
        self._context_key = "task"
        self._context_images: dict[str, int] = {}
        self._context_tokens: dict[str, int] = {}
        self.state_revision = 0
        self.current_observation = session.task.observation
        self.artifacts = ArtifactStore(artifact_dir)
        configuration = {
            **experiment.manifest(),
            "evidence_protocol": "event-moment-view",
            "pixel_readout_protocol": (
                f"rgb-{self.context.pixel_readout_color_bits}bit-palette-per-"
                f"{self.context.pixel_readout_palette_scope}"
            ),
            **(
                {"recovery_protocol": "notes-current-history-reference"}
                if experiment.profile.policy.compaction == CompactionMode.CHECKPOINT
                else {}
            ),
            "submission_contract": self.submission.manifest(),
            "tool_contracts": [
                {
                    **spec.public_definition(),
                    "capability": spec.capability.value,
                    "effect": spec.effect.value,
                    "max_images": spec.max_images,
                    "costs": thaw(spec.costs),
                    "dispatch_invalid": spec.dispatch_invalid,
                }
                for spec in self.specs
            ],
        }
        self.identity = hashlib.sha256(
            canonical(configuration).encode("ascii")
        ).hexdigest()
        self.artifacts.manifest(
            {
                **configuration,
                "configuration_id": experiment.identity,
                "experiment_id": self.identity,
                "task": {
                    "id": session.task.task_id,
                    "objective": session.task.objective,
                    "context": thaw(session.task.context),
                    "observation": session.task.observation.manifest(),
                },
            }
        )
        self.artifacts.evidence(session.task.observation)
        self.context.bind(self, artifact_dir / "context")

    @property
    def status(self) -> Status:
        return self._status

    @property
    def public_events(self) -> list[dict[str, Any]]:
        return self.context.public_events

    @property
    def terminal(self) -> bool:
        return self._status != Status.RUNNING or self._closed

    @property
    def failure(self) -> str | None:
        return self._failure

    @property
    def usage(self) -> dict[str, int]:
        with self._lock:
            return dict(self._usage)

    @property
    def pending_call(self) -> str | None:
        with self._lock:
            return self._pending

    @property
    def boundary(self) -> str | None:
        return self._boundary

    def observe_context_tokens(self, tokens: int) -> None:
        with self._lock:
            if type(tokens) is int and tokens >= 0:
                self._context_tokens[self._context_key] = tokens

    def fresh_context(
        self, context_key: str | None = None, *, redelivery: bool = False
    ) -> None:
        with self._lock:
            if self._pending is not None and not redelivery:
                raise RuntimeError("Cannot replace a context before result delivery")
            key = context_key or self._context_key
            self._context_images.pop(key, None)
            self._context_tokens.pop(key, None)
            self.artifacts.record("fresh_context", {"context_key": key})

    def tools(self) -> list[dict[str, Any]]:
        return [spec.public_definition() for spec in self.specs]

    def matched_environment(self, other: TaskRuntime) -> bool:
        """Structural comparison guard; semantic adapter validation is still required."""

        def interface(runtime: TaskRuntime) -> list[dict[str, Any]]:
            return [
                {
                    **spec.public_definition(),
                    "effect": spec.effect.value,
                    "max_images": spec.max_images,
                    "costs": thaw(spec.costs),
                    "dispatch_invalid": spec.dispatch_invalid,
                }
                for spec in runtime.specs
                if spec.capability in {Capability.ACTION, Capability.SUBMISSION}
            ]

        return (
            self.experiment.matched_environment(other.experiment)
            and self.session.task == other.session.task
            and self.submission == other.submission
            and interface(self) == interface(other)
        )

    def spec(self, name: str) -> ToolSpec | None:
        binding = self._bindings.get(name)
        return binding.spec if binding is not None else None

    def prepare_input(
        self,
        phase: Phase,
        *,
        public_text: str = "",
        observation: ObservationBundle | None = None,
        context_key: str = "task",
        blocks: tuple[str | ObservationBundle, ...] = (),
    ) -> TaskInput:
        """Compose only the selected phase and explicitly supplied public content."""
        with self._lock:
            if self.terminal or self._pending is not None:
                raise RuntimeError("The session is not ready for another model input")
            phase = Phase(phase)
            prompt = self.experiment.profile.prompts.for_phase(phase)
            if not context_key or not isinstance(public_text, str):
                raise ValueError("Invalid public input")
            if observation is None:
                observation = (
                    self.session.task.observation
                    if phase == Phase.INITIAL
                    else ObservationBundle()
                )
            text = "\n\n".join(part for part in (prompt, public_text) if part)
            return TaskInput(text, observation, phase, context_key, blocks)

    def present_input(self, task_input: TaskInput) -> TaskInput:
        return replace(
            task_input, image_labels=self.context.labels(task_input.observation)
        )

    def begin_turn(self, task_input: TaskInput, *, redelivery: bool = False) -> None:
        with self._lock:
            self.experiment.profile.prompts.for_phase(task_input.phase)
            if self.terminal or (self._pending is not None and not redelivery):
                raise RuntimeError("The session is not ready for a model turn")
            if not self._charge({"backend_segments": 1}):
                raise BudgetExhausted("Backend segment budget exhausted")
            self._boundary = None
            self._phase = task_input.phase
            self._context_key = task_input.context_key
            self._context_images[self._context_key] = self._context_images.get(
                self._context_key, 0
            ) + sum(
                len(item.views)
                for item in task_input.observation.displayed_observations
            )
            self.artifacts.evidence(task_input.observation)
            self._observe_current(task_input.observation)
            self.artifacts.record(
                "model_input",
                {
                    "phase": task_input.phase.value,
                    "context_key": task_input.context_key,
                    "text": task_input.text,
                    "observation": task_input.observation.manifest(),
                    "blocks": task_input.block_manifest(),
                    "image_labels": self.context.labels(task_input.observation),
                },
            )

    def execute(self, tool: str, arguments: Any, call_id: str) -> ToolExecution:
        with self._lock:
            if self.terminal:
                return ToolExecution("The task session has ended.", False)
            if self._boundary is not None:
                return ToolExecution(
                    "This decision has ended; no further tool was executed.",
                    False,
                    interrupt_after=True,
                    boundary_reason=self._boundary,
                )
            if not isinstance(call_id, str) or not call_id:
                return ToolExecution("Invalid tool request identity.", False)
            if call_id in self._seen_calls:
                return ToolExecution(
                    "Duplicate tool request; nothing was executed.", False
                )
            self._seen_calls.add(call_id)
            if not self._charge({"tool_calls": 1}):
                return self._stopped_execution("Tool call budget exhausted.")
            binding = self._bindings.get(tool)
            if binding is None:
                return self._rejected(call_id, tool, "Unknown or disabled tool.")
            if binding.spec.requires_ack and self._pending is not None:
                return self._rejected(
                    call_id, tool, "Observe the previous result before changing state."
                )
            if self._phase == Phase.CHECKPOINT and binding.spec.capability not in {
                Capability.MEMORY,
                Capability.CHECKPOINT,
            }:
                return self._rejected(
                    call_id,
                    tool,
                    "Only memory tools are available during checkpointing.",
                )
            try:
                canonical(arguments)
                valid = isinstance(arguments, dict) and self._validators[tool].is_valid(
                    arguments
                )
            except Exception:
                valid = False
                # Non-JSON input is a counted no-action attempt for adapters
                # that own invalid-output semantics, not an artifact failure.
                if binding.spec.dispatch_invalid:
                    arguments = None
            if not valid and not binding.spec.dispatch_invalid:
                return self._rejected(call_id, tool, "Invalid tool arguments.")
            if not self._charge(dict(binding.spec.costs)):
                return self._stopped_execution("The task budget is exhausted.")
            self.artifacts.record(
                "tool_started",
                {
                    "call_id": call_id,
                    "tool": tool,
                    "arguments": arguments,
                },
            )
            try:
                reply = binding.handler(arguments)
                if not isinstance(reply, ToolReply):
                    raise TypeError("A task handler did not return ToolReply")
                if (
                    binding.spec.capability == Capability.ACTION
                    and self.experiment.profile.policy.one_action_per_turn
                    and not reply.finished
                    and reply.boundary is None
                ):
                    reply = replace(reply, boundary="decision_complete")
                image_count = sum(
                    len(item.views) for item in reply.observation.displayed_observations
                )
                if image_count > binding.spec.max_images:
                    raise ValueError("A handler exceeded its declared image bound")
                if reply.boundary == "compact_checkpoint_saved" and (
                    binding.spec.capability
                    not in {Capability.CHECKPOINT, Capability.MEMORY}
                    or self.experiment.profile.policy.compaction
                    != CompactionMode.CHECKPOINT
                ):
                    raise ValueError(
                        "Checkpoint boundary requires a configured checkpoint tool"
                    )
                if (
                    reply.boundary == "decision_complete"
                    and binding.spec.capability != Capability.ACTION
                ):
                    raise ValueError("Only an action can end an environment decision")
                self.artifacts.evidence(reply.observation)
                event = None
                if binding.spec.effect == Effect.ENVIRONMENT:
                    event = self._record_environment_result(tool, arguments, reply)
                else:
                    self.context.observe(reply.observation, update_current=False)
                images = self.context.images(reply.observation, event=event)
            except Exception:
                self._handler_failed(call_id)
                return self._stopped_execution(
                    "The task handler failed; execution stopped."
                )
            if binding.spec.requires_ack:
                self._pending = call_id
            self._context_images[self._context_key] = self._context_images.get(
                self._context_key, 0
            ) + len(images)
            self._boundary = reply.boundary
            self.artifacts.record(
                "tool_result",
                {
                    "call_id": call_id,
                    "success": reply.success,
                    "text": reply.text,
                    "observation": reply.observation.manifest(),
                    "image_labels": [image["steer_text"] for image in images],
                    "finished": reply.finished,
                    "boundary": reply.boundary,
                },
            )
            if reply.finished:
                self._status = Status.COMPLETED
            return ToolExecution(
                reply.text,
                reply.success,
                images,
                reply.finished or reply.boundary is not None,
                "task_complete" if reply.finished else reply.boundary,
                images_before_interrupt=reply.finished or reply.boundary is not None,
            )

    def acknowledge(self, call_id: str | None = None) -> None:
        with self._lock:
            if self._pending is not None and (
                call_id is None or call_id == self._pending
            ):
                self.artifacts.record("result_delivered", {"call_id": self._pending})
                self._pending = None

    def submit(self, answer: str) -> ToolReply:
        with self._lock:
            if self.terminal or self._pending is not None:
                raise RuntimeError("The session cannot accept a final response now")
            if not isinstance(answer, str):
                raise TypeError("A final response must be text")
            if not self._charge({"final_answers": 1, **dict(self.submission.costs)}):
                return ToolReply("Final response budget exhausted.", success=False)
            self.artifacts.record("final_response", {"text": answer})
            try:
                reply = self.session.submit(answer)
                if not isinstance(reply, ToolReply):
                    raise TypeError("A submission handler did not return ToolReply")
                if (
                    sum(
                        len(item.views)
                        for item in reply.observation.displayed_observations
                    )
                    > self.submission.max_images
                ):
                    raise ValueError(
                        "A submission handler exceeded its declared image bound"
                    )
                self.artifacts.evidence(reply.observation)
                if self.submission.effect == Effect.ENVIRONMENT:
                    self._record_environment_result(None, None, reply)
                else:
                    self.context.observe(reply.observation, update_current=False)
            except Exception:
                self._handler_failed("final_response")
                return ToolReply("The task submission failed.", success=False)
            self.artifacts.record(
                "submission_result",
                {
                    "text": reply.text,
                    "success": reply.success,
                    "finished": reply.finished,
                    "observation": reply.observation.manifest(),
                },
            )
            if reply.finished:
                self._status = Status.COMPLETED
            return reply

    def stop(self, reason: str, *, failed: bool = False) -> None:
        with self._lock:
            if not self._closed and (not self.terminal or failed):
                self._status = Status.FAILED if failed else Status.STOPPED
                self._failure = reason
                self.artifacts.record(
                    "stopped", {"reason": reason, "status": self.status.value}
                )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self.stop("session_closed")
            self._closed = True
            try:
                self.session.close()
            except Exception:
                self._handler_failed("session_close")
                raise
            finally:
                self.artifacts.record(
                    "closed", {"status": self.status.value, "usage": self.usage}
                )

    def _observe_current(self, bundle: ObservationBundle) -> None:
        current = tuple(
            item for item in bundle.observations if item.kind == EvidenceKind.CURRENT
        )
        if current:
            self.current_observation = ObservationBundle(current)
        self.context.observe(bundle)

    def _record_environment_result(
        self, tool: str | None, arguments: Any, reply: ToolReply
    ) -> str:
        event = self.context.record_event(
            reply.observation,
            action={"name": tool, "arguments": arguments} if tool is not None else None,
            success=reply.success,
            result=reply.text if reply.event_text is None else reply.event_text,
            environment=True,
        )
        self.state_revision += 1
        self.current_observation = self.context.current
        return event

    def _charge(self, costs: dict[str, int]) -> bool:
        for meter, cost in costs.items():
            limit = self.experiment.budgets.get(meter)
            if limit is not None and self._usage.get(meter, 0) + cost > limit:
                self._status = Status.BUDGET
                self._failure = meter
                self.artifacts.record("budget_exhausted", {"meter": meter})
                return False
        for meter, cost in costs.items():
            self._usage[meter] = self._usage.get(meter, 0) + cost
        return True

    def _handler_failed(self, call_id: str) -> None:
        self._status = Status.FAILED
        self._failure = "task_handler_failed"
        self.artifacts.record(
            "private_failure",
            {
                "call_id": call_id,
                "traceback": traceback.format_exc(),
            },
        )

    def _rejected(self, call_id: str, tool: str, message: str) -> ToolExecution:
        self.artifacts.record(
            "rejected", {"call_id": call_id, "tool": tool, "reason": message}
        )
        return ToolExecution(message, False)

    def _stopped_execution(self, text: str) -> ToolExecution:
        return ToolExecution(
            text, False, interrupt_after=True, boundary_reason=self.status.value
        )
