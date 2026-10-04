"""Task bindings for shared provider lifecycles."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from time import sleep
from typing import Any, Protocol

from vista.core.contracts import ObservationBundle
from vista.core.profiles import CompactionMode, ContextMode, Phase
from vista.core.runtime import BudgetExhausted, TaskInput, TaskRuntime

CAPACITY_RETRY_DELAYS = (10.0, 20.0, 40.0)


class Runner(Protocol):
    def run_input(
        self, task_input: TaskInput, *, session_id: str | None = None
    ) -> Any: ...


@dataclass(frozen=True)
class BackendResult:
    returncode: int
    final_message: str
    session_id: str | None
    boundary: str | None = None


def preflight(runtime: TaskRuntime, kind: str) -> None:
    if runtime.experiment.backend != kind:
        raise ValueError("The backend does not match the resolved experiment")
    profile = runtime.experiment.profile
    if profile.policy.compaction == CompactionMode.CHECKPOINT:
        required = {"read_guide", "write_guide", "read_working", "write_working"}
        if kind == "codex":
            required.add("save_compact_checkpoint")
        if (
            not required <= set(profile.tools)
            or profile.prompts.checkpoint is None
            or profile.prompts.recovery is None
        ):
            raise ValueError("Checkpoint handoff is not configured")


def prepare_directory(directory: Path) -> tuple[Path, Path, Path]:
    directory = directory.resolve()
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    directory.chmod(0o700)
    paths = tuple(directory / name for name in ("visible", "state", "io"))
    for path in paths:
        path.mkdir(mode=0o700)
    return paths


class BackendSession:
    def __init__(
        self,
        runtime: TaskRuntime,
        runner: Runner,
        *,
        kind: str,
        before_first_run: Callable[[], None] | None = None,
        credential_relay: Callable[[], str] | None = None,
        credential_relay_needed: Callable[[], bool] | None = None,
    ) -> None:
        preflight(runtime, kind)
        self.runtime, self.runner, self.kind = runtime, runner, kind
        self._sessions: dict[str, str] = {}
        self._known_sessions: set[str] = set()
        self._run_lock = threading.Lock()
        self._before_first_run = before_first_run
        self.credential_relay = credential_relay
        self.credential_relay_needed = credential_relay_needed
        self.usage_reports: list[dict[str, Any] | None] = []

    def run(self, task_input: TaskInput) -> BackendResult:
        with self._run_lock:
            if self.runtime.terminal:
                raise RuntimeError("The task session has ended")
            try:
                if self._before_first_run is not None:
                    self._before_first_run()
                    self._before_first_run = None
                return self._run(task_input)
            except BudgetExhausted:
                raise
            except Exception:
                self.runtime.stop("backend_exception", failed=True)
                raise

    def _run(self, task_input: TaskInput) -> BackendResult:
        port = _RunnerPort(self, task_input)
        controller = self.runtime.context
        if self.kind == "codex":
            from vista.backends.codex.lifecycle import run_task_with_compact_checkpoints

            results = run_task_with_compact_checkpoints(
                runner=port,
                controller=controller,
                prompt=task_input.text,
                checkpoint_marker=controller.compact_checkpoint_marker,
                checkpoint_ready=controller.compact_checkpoint_ready,
                working_path=controller.working_path,
                should_yield=port.should_yield,
            )
        else:
            from vista.backends.claude.lifecycle import run_task_with_native_context

            results = run_task_with_native_context(
                runner=port,
                controller=controller,
                prompt=task_input.text,
                results=[],
                should_yield=port.should_yield,
                credential_relay=self.credential_relay,
                credential_relay_needed=self.credential_relay_needed,
            )
        raw = results[-1]
        boundary = getattr(raw, "context_boundary", None) or self.runtime.boundary
        if self.runtime.failure is None:
            if raw.returncode != 0:
                self.runtime.stop("backend_failure", failed=True)
            elif self.runtime.pending_call is not None:
                self.runtime.stop("unconfirmed_result_delivery", failed=True)
            elif controller.forced_termination_reason:
                self.runtime.stop(controller.forced_termination_reason, failed=True)
            elif boundary not in {None, "task_complete", "decision_complete"}:
                self.runtime.stop(boundary)
        return BackendResult(
            raw.returncode, raw.final_message, raw.session_id, boundary
        )


class _RunnerPort:
    """Translate task inputs/results; provider lifecycles own recovery decisions."""

    def __init__(self, owner: BackendSession, initial: TaskInput):
        self.owner, self.initial = owner, initial
        self.runtime = owner.runtime
        self.leaf = owner.runner
        self.context_key = initial.context_key
        self.continuous = (
            self.runtime.experiment.profile.policy.context == ContextMode.CONTINUOUS
        )
        self._yield = False
        self._capacity_retries = 0

    def should_yield(self) -> bool:
        return self._yield

    def _input(self, text: str, phase: Phase, *, images: bool = False) -> TaskInput:
        self.runtime.experiment.profile.prompts.for_phase(phase)
        return TaskInput(
            text,
            self.runtime.current_observation if images else ObservationBundle(),
            phase,
            self.context_key,
        )

    def _invoke(
        self, call, task_input, *, expected: str | None, recovery: bool = False
    ):
        self._yield = False
        if expected is None:
            self.runtime.fresh_context(self.context_key, redelivery=recovery)
        self.runtime.begin_turn(task_input, redelivery=recovery)
        raw = call()
        if getattr(raw, "retryable", None) is False:
            self.runtime.stop("nonretryable_backend_failure", failed=True)
        session_id = raw.session_id
        if session_id:
            if expected is not None and session_id != expected:
                self.runtime.stop("backend_session_changed", failed=True)
            elif expected is None and session_id in self.owner._known_sessions:
                self.runtime.stop("backend_session_reused", failed=True)
            else:
                self.owner._known_sessions.add(session_id)
                if self.continuous:
                    self.owner._sessions[self.context_key] = session_id
        elif raw.returncode == 0:
            self.runtime.stop("missing_backend_session_id", failed=True)
        if raw.returncode == 0 and self.runtime.pending_call is not None:
            self.runtime.stop("unconfirmed_result_delivery", failed=True)
        usage = getattr(raw, "usage", None)
        report = (
            asdict(usage)
            if is_dataclass(usage) and not isinstance(usage, type)
            else None
        )
        self.owner.usage_reports.append(report)
        self.runtime.artifacts.record(
            "backend_result",
            {
                "backend": self.owner.kind,
                "returncode": raw.returncode,
                "context_key": self.context_key,
                "session_id": session_id,
                "boundary": getattr(raw, "context_boundary", None)
                or self.runtime.boundary,
                "resolved_models": list(getattr(raw, "resolved_models", ())),
                "reported_usage": report,
                "retryable": getattr(raw, "retryable", None),
                "failure_kind": getattr(raw, "failure_kind", None),
            },
        )
        if (
            getattr(raw, "failure_kind", None) == "serverOverloaded"
            and getattr(raw, "retryable", None) is True
            and not self.runtime.terminal
        ):
            if self._capacity_retries >= len(CAPACITY_RETRY_DELAYS):
                self.runtime.stop("backend_capacity_exhausted", failed=True)
            else:
                delay = CAPACITY_RETRY_DELAYS[self._capacity_retries]
                self._capacity_retries += 1
                self.runtime.artifacts.record(
                    "backend_retry_backoff",
                    {
                        "failure_kind": "serverOverloaded",
                        "retry": self._capacity_retries,
                        "delay_seconds": delay,
                        "session_id": session_id,
                    },
                )
                sleep(delay)
        controller = self.runtime.context
        checkpoint_pending = controller.compact_checkpoint_marker.is_file()
        boundary = getattr(raw, "context_boundary", None) or self.runtime.boundary
        normal_boundaries = {None, "task_complete", "decision_complete"}
        if self.runtime.experiment.profile.policy.compaction == CompactionMode.NATIVE:
            normal_boundaries.add("provider_image_limit")
        self._yield = (
            raw.returncode == 0
            and boundary in normal_boundaries
            and getattr(raw, "decision_lease_failure", None) is None
            and task_input.phase != Phase.CHECKPOINT
            and not checkpoint_pending
            and not controller.retry_boundary_pending
        )
        return raw

    def run_task(self, prompt):
        expected = (
            self.owner._sessions.get(self.context_key) if self.continuous else None
        )
        return self._invoke(
            lambda: self.leaf.run_input(
                self.runtime.present_input(self.initial), session_id=expected
            ),
            self.initial,
            expected=expected,
        )

    def resume_checkpoint(self, session_id, prompt):
        task_input = self._input(prompt, Phase.CHECKPOINT)
        return self._invoke(
            lambda: self.leaf.resume_checkpoint(session_id, prompt),
            task_input,
            expected=session_id,
        )

    def _recover(self, method, recovery, *, session_id=None, fresh=False):
        mode = self.runtime.experiment.profile.policy.compaction
        if mode == CompactionMode.NATIVE:
            phase = (
                Phase.STEP
                if self.runtime.experiment.profile.prompts.step is not None
                else Phase.INITIAL
            )
            text = self.runtime.experiment.profile.prompts.for_phase(phase)
        else:
            phase = Phase.RECOVERY
            kind = (
                "compact"
                if "compact" in method
                or method in {"run_recovery_task", "resume_recovery_task"}
                else "retry"
                if "retry" in method
                else "fresh_runtime"
                if "fresh_runtime" in method
                else "runtime"
            )
            text = self.leaf._recovery_prompt(kind, recovery)
        task_input = self._input(text, phase, images=True)
        callback = getattr(self.leaf, method)
        arguments = (recovery,) if fresh else (session_id, recovery)
        return self._invoke(
            lambda: callback(*arguments),
            task_input,
            expected=None if fresh else session_id,
            recovery=True,
        )

    def run_recovery_task(self, recovery):
        return self._recover("run_recovery_task", recovery, fresh=True)

    def resume_recovery_task(self, session_id, recovery):
        return self._recover("resume_recovery_task", recovery, session_id=session_id)

    def run_compact_recovery(self, recovery):
        return self._recover("run_compact_recovery", recovery, fresh=True)

    def run_retry_task(self, recovery):
        return self._recover("run_retry_task", recovery, fresh=True)

    def resume_retry_task(self, session_id, recovery):
        return self._recover("resume_retry_task", recovery, session_id=session_id)

    def resume_runtime_recovery(self, session_id, recovery):
        return self._recover("resume_runtime_recovery", recovery, session_id=session_id)

    def run_fresh_runtime_recovery(self, recovery):
        return self._recover("run_fresh_runtime_recovery", recovery, fresh=True)

    def resume_nonterminal_task(self, session_id, prompt):
        task_input = self._input(prompt, Phase.STEP)
        return self._invoke(
            lambda: self.leaf.resume_nonterminal_task(session_id, prompt),
            task_input,
            expected=session_id,
        )

    def replace_oauth_token(self, token):
        self.leaf.replace_oauth_token(token)
