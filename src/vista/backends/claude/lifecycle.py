"""Claude session lifecycle, independent of benchmark setup."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Callable

from vista.backends.claude.runner import ClaudeCodeRunner, ClaudeResult
from vista.core.session import SessionController as GameController

PROVIDER_5XX_BACKOFF_SECONDS = (10, 30, 60, 120)


MAX_DECISION_LEASE_RECOVERIES_PER_STEP = 5


MAX_COMPACT_CYCLES_WITHOUT_ACTION = 3


MAX_RUNTIME_RECOVERIES_PER_STEP = 8


NONTERMINAL_CONTINUATION_PROMPT = (
    "The environment is still active. "
    "Context limits are handled automatically. Continue."
)


def run_task_with_native_context(
    *,
    runner: ClaudeCodeRunner,
    controller: GameController,
    prompt: str,
    results: list[ClaudeResult],
    sleep: Callable[[float], None] = time.sleep,
    credential_relay: Callable[[], str] | None = None,
    credential_relay_needed: Callable[[], bool] | None = None,
    should_yield: Callable[[], bool] | None = None,
) -> list[ClaudeResult]:
    results.append(runner.run_task(prompt))
    session_id = results[-1].session_id
    decision_recovery_step: int | None = None
    decision_recoveries_at_step = 0
    provider_5xx_failures_at_step = 0
    compact_step: int | None = None
    compact_cycles_at_step = 0
    nonterminal_continuation_scope: tuple[str, int | None] | None = None

    while True:
        current = results[-1]
        if current.unconfirmed_play_calls:
            raise RuntimeError(
                "Claude stopped with an unconfirmed play call; "
                "the game state is not safe to relay"
            )
        if controller.terminal:
            discard_compact_recovery = getattr(
                controller,
                "discard_compact_recovery",
                None,
            )
            if callable(discard_compact_recovery):
                discard_compact_recovery()
            return results
        if (
            current.context_boundary != "credential_rate_limit"
            and credential_relay_needed is not None
            and credential_relay_needed()
        ):
            if credential_relay is None:
                raise RuntimeError(
                    "Claude reached a credential rate-limit boundary without a relay"
                )
            runner.replace_oauth_token(credential_relay())
        if current.context_boundary == "credential_rate_limit":
            if credential_relay is None:
                raise RuntimeError(
                    "Claude reached a credential rate-limit boundary without a relay"
                )
            previous_session_id = session_id
            runner.replace_oauth_token(credential_relay())
            recovery = controller.begin_runtime_recovery()
            continued = runner.run_fresh_runtime_recovery(recovery)
            results.append(continued)
            if continued.session_id is None:
                raise RuntimeError("Credential relay created no Claude session")
            if (
                previous_session_id is not None
                and continued.session_id == previous_session_id
            ):
                raise RuntimeError("Credential relay reused the stopped Claude session")
            session_id = continued.session_id
            decision_recovery_step = None
            decision_recoveries_at_step = 0
            provider_5xx_failures_at_step = 0
            continue
        if should_yield is not None and should_yield():
            return results
        if controller.retry_boundary_pending:
            recovery = controller.begin_retry_recovery()
            previous_session_id = session_id
            recovery_session_id: str | None = None
            recovery_failures = 0
            while True:
                if controller.retry_boundary_pending:
                    recovered = runner.run_retry_task(recovery)
                else:
                    if recovery_session_id is None:
                        raise RuntimeError("Retry recovery created no Claude session")
                    recovered = runner.resume_retry_task(
                        recovery_session_id,
                        recovery,
                    )
                results.append(recovered)
                pending_step = controller.retry_boundary_step
                recovery_delivered = not controller.retry_boundary_pending or (
                    type(pending_step) is int and pending_step > recovery.boundary_step
                )
                if recovered.session_id is not None and recovery_delivered:
                    if recovery_session_id is None:
                        if recovered.session_id == previous_session_id:
                            raise RuntimeError(
                                "Retry recovery did not create a fresh Claude session"
                            )
                        recovery_session_id = recovered.session_id
                    elif (
                        not controller.retry_boundary_pending
                        and recovered.session_id != recovery_session_id
                    ):
                        raise RuntimeError(
                            "Retry recovery did not resume the Claude session"
                        )
                if type(pending_step) is int and pending_step > recovery.boundary_step:
                    break
                if recovered.returncode == 0:
                    if controller.retry_boundary_pending:
                        raise RuntimeError("Claude did not enter retry recovery")
                    break
                if (
                    recovered.api_error_status is not None
                    and recovered.api_error_status >= 500
                    and recovery_delivered
                    and recovery_session_id is not None
                ):
                    break
                recovery_failures += 1
                if recovery_failures > MAX_RUNTIME_RECOVERIES_PER_STEP:
                    raise RuntimeError(
                        "Claude retry recovery failed repeatedly at the same game state"
                    )
            if recovery_session_id is None:
                raise RuntimeError("Retry recovery created no Claude session")
            session_id = recovery_session_id
            continue

        current = results[-1]
        if current.context_boundary == "provider_image_limit":
            raise RuntimeError(
                "Claude image boundary ended before WORKING.md was saved"
            )
        elif current.context_boundary == "native_compact":
            raise RuntimeError(
                "Claude compacted before the checkpoint handoff completed"
            )

        checkpoint_marker = getattr(controller, "compact_checkpoint_marker", None)
        checkpoint_ready = getattr(controller, "compact_checkpoint_ready", None)
        if checkpoint_marker is not None and checkpoint_marker.is_file():
            if session_id is None:
                raise RuntimeError("Claude compact checkpoint has no session id")
            if compact_step != controller.step_index:
                compact_step = controller.step_index
                compact_cycles_at_step = 0
            compact_cycles_at_step += 1
            if compact_cycles_at_step > MAX_COMPACT_CYCLES_WITHOUT_ACTION:
                raise RuntimeError(
                    "Claude compact checkpoint loop repeated without an action"
                )

            checkpoint_saved = (
                checkpoint_ready is not None and checkpoint_ready.is_file()
            )
            if not checkpoint_saved:
                raise RuntimeError(
                    "Claude ended before writing WORKING.md for compact handoff"
                )

            previous_session_id = session_id
            recovery = controller.begin_compact_recovery()
            recovered = runner.run_compact_recovery(recovery)
            results.append(recovered)
            session_id = recovered.session_id
            if recovered.returncode != 0:
                raise RuntimeError(
                    f"Claude compact recovery failed; stderr: {recovered.stderr_path}"
                )
            if session_id is None:
                raise RuntimeError("Compact recovery created no Claude session")
            if session_id == previous_session_id:
                raise RuntimeError(
                    "Compact recovery did not create a fresh Claude session"
                )
            restore_marker = controller.compact_restore_marker
            if restore_marker is not None and restore_marker.exists():
                raise RuntimeError("Claude did not enter compact recovery")
            compact_step = None
            compact_cycles_at_step = 0
            decision_recovery_step = None
            decision_recoveries_at_step = 0
            continue

        lease_failure = current.decision_lease_failure
        if current.returncode == 0 and lease_failure is None:
            lease_failure = "nonterminal_completion"
            current = replace(
                current,
                decision_lease_failure=lease_failure,
            )
            results[-1] = current
            level_progress = None
            initial_metadata = getattr(controller, "initial_metadata", None)
            if callable(initial_metadata):
                progress = initial_metadata().get("progress")
                if isinstance(progress, dict):
                    completed = progress.get("completed")
                    if type(completed) is int:
                        level_progress = completed
            continuation_scope = (
                (session_id, level_progress) if session_id is not None else None
            )
            if (
                continuation_scope is not None
                and continuation_scope != nonterminal_continuation_scope
            ):
                nonterminal_continuation_scope = continuation_scope
                continued = runner.resume_nonterminal_task(
                    session_id,
                    NONTERMINAL_CONTINUATION_PROMPT,
                )
                results.append(continued)
                if (
                    continued.session_id is not None
                    and continued.session_id != session_id
                ):
                    raise RuntimeError(
                        "Nonterminal continuation changed Claude session"
                    )
                session_id = continued.session_id or session_id
                continue
        elif lease_failure is None:
            lease_failure = "runtime_exit"

        if decision_recovery_step != controller.step_index:
            decision_recovery_step = controller.step_index
            decision_recoveries_at_step = 0
            provider_5xx_failures_at_step = 0
        decision_recoveries_at_step += 1
        if decision_recoveries_at_step > MAX_DECISION_LEASE_RECOVERIES_PER_STEP:
            return results

        if current.api_error_status is not None and current.api_error_status >= 500:
            provider_5xx_failures_at_step += 1
            delay = PROVIDER_5XX_BACKOFF_SECONDS[
                min(
                    provider_5xx_failures_at_step - 1,
                    len(PROVIDER_5XX_BACKOFF_SECONDS) - 1,
                )
            ]
            sleep(delay)

        previous_session_id = session_id
        recovery = controller.begin_runtime_recovery()
        continued = runner.run_fresh_runtime_recovery(recovery)
        results.append(continued)
        if (
            continued.session_id is not None
            and previous_session_id is not None
            and continued.session_id == previous_session_id
        ):
            raise RuntimeError(
                "Decision lease recovery reused the stopped Claude session"
            )
        session_id = continued.session_id
