"""Run an explicitly bounded task; never invent a continuation or recovery prompt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .contracts import Effect
from .profiles import Phase
from .runtime import BudgetExhausted, Status, TaskRuntime

if TYPE_CHECKING:
    from vista.backends.session import BackendSession


@dataclass(frozen=True)
class EpisodeResult:
    status: Status
    final_message: str
    usage: dict[str, int]


def run_episode(runtime: TaskRuntime, backend: BackendSession) -> EpisodeResult:
    if backend.runtime is not runtime:
        raise ValueError("A backend cannot be reused for a different task runtime")
    if "backend_segments" not in runtime.experiment.budgets:
        raise ValueError("run_episode requires an explicit backend_segments limit")
    final_message = ""
    try:
        task_input = runtime.prepare_input(Phase.INITIAL)
        while not runtime.terminal:
            before = runtime.state_revision
            result = backend.run(task_input)
            final_message = result.final_message
            if runtime.terminal:
                break
            if result.boundary == "decision_complete" or (
                runtime.state_revision != before
                and runtime.submission.effect == Effect.ENVIRONMENT
            ):
                runtime.stop("benchmark_decision_loop_required")
                break
            reply = runtime.submit(final_message)
            if runtime.terminal:
                break
            if runtime.experiment.profile.prompts.step is None:
                runtime.stop("nonterminal_response_without_step_prompt")
                break
            task_input = runtime.prepare_input(
                Phase.STEP, public_text=reply.text, observation=reply.observation
            )
    except BudgetExhausted:
        pass
    except Exception:
        runtime.stop("episode_exception", failed=True)
        raise
    finally:
        runtime.close()
    return EpisodeResult(runtime.status, final_message, runtime.usage)


def result_manifest(result: EpisodeResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "final_message": result.final_message,
        "usage": result.usage,
    }
