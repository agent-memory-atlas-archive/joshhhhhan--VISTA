"""Codex session lifecycle, independent of benchmark setup."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from vista.backends.codex.runner import CodexResult, DockerCodexRunner
from vista.core.session import SessionController as GameController, write_json

CHECKPOINT_PROMPT = (
    '<codex_internal_context source="compact_checkpoint">\n'
    "Call save_compact_checkpoint now as your first and only action, then end "
    "this turn.\n"
    "</codex_internal_context>"
)


MAX_CHECKPOINT_ATTEMPTS = 2


MAX_COMPACT_CYCLES_WITHOUT_ACTION = 3


MAX_RUNTIME_RECOVERIES_PER_STEP = 8


NONTERMINAL_CONTINUATION_PROMPT = (
    "The environment is still active. "
    "Context limits are handled automatically. Continue."
)


def run_task_with_compact_checkpoints(
    *,
    runner: DockerCodexRunner,
    controller: GameController,
    prompt: str,
    checkpoint_marker: Path,
    checkpoint_ready: Path,
    working_path: Path,
    results: list[CodexResult] | None = None,
    should_yield: Callable[[], bool] | None = None,
) -> list[CodexResult]:
    if results is None:
        results = []
    results.append(runner.run_task(prompt))
    session_id = results[-1].session_id
    checkpoints = 0
    retry_boundaries = 0
    last_checkpoint_step: int | None = None
    compact_cycles_without_action = 0
    runtime_recovery_step: int | None = None
    runtime_recoveries_at_step = 0
    nonterminal_continuation_scope: tuple[str, int | None] | None = None

    def record_runtime_failure(result: CodexResult) -> None:
        nonlocal runtime_recovery_step, runtime_recoveries_at_step
        current_step = controller.step_index
        if runtime_recovery_step != current_step:
            runtime_recovery_step = current_step
            runtime_recoveries_at_step = 0
        runtime_recoveries_at_step += 1
        if runtime_recoveries_at_step > MAX_RUNTIME_RECOVERIES_PER_STEP:
            raise RuntimeError(
                "Codex player runtime failed repeatedly at the same game state; "
                f"last stderr: {result.stderr_path}"
            )

    def resume_player_runtime() -> None:
        nonlocal session_id
        failed = results[-1]
        record_runtime_failure(failed)
        if session_id is None:
            raise RuntimeError(
                "Codex player runtime failed without a resumable session id; "
                f"stderr: {failed.stderr_path}"
            )
        expected_session_id = session_id
        recovery = controller.begin_runtime_recovery()
        resumed = runner.resume_runtime_recovery(
            expected_session_id,
            recovery,
        )
        results.append(resumed)
        session_id = resumed.session_id
        if session_id != expected_session_id:
            raise RuntimeError(
                "Runtime recovery did not resume the exact Codex session"
            )

    while True:
        if controller.terminal:
            controller.discard_compact_recovery()
            return results
        if should_yield is not None and should_yield():
            return results

        if getattr(controller, "retry_boundary_pending", False):
            retry_boundaries += 1
            if retry_boundaries > controller.max_steps:
                raise RuntimeError("Retry boundaries made no bounded progress")
            previous_session_id = session_id
            recovery = controller.begin_retry_recovery()
            recovery_session_id: str | None = None

            while getattr(controller, "retry_boundary_pending", False):
                if recovery_session_id is None:
                    recovered = runner.run_retry_task(recovery)
                else:
                    recovered = runner.resume_retry_task(
                        recovery_session_id,
                        recovery,
                    )
                results.append(recovered)

                if recovered.session_id is not None:
                    if recovery_session_id is None:
                        if recovered.session_id == previous_session_id:
                            raise RuntimeError(
                                "Retry recovery did not create its required fresh session"
                            )
                        recovery_session_id = recovered.session_id
                    elif recovered.session_id != recovery_session_id:
                        raise RuntimeError(
                            "Retry recovery did not resume the exact Codex session"
                        )

                boundary_pending = bool(
                    getattr(controller, "retry_boundary_pending", False)
                )
                pending_boundary_step = (
                    getattr(controller, "retry_boundary_step", None)
                    if boundary_pending
                    else None
                )
                if boundary_pending and (
                    type(pending_boundary_step) is not int
                    or pending_boundary_step < recovery.boundary_step
                ):
                    raise RuntimeError("Retry boundary state is inconsistent")
                same_boundary_pending = pending_boundary_step == recovery.boundary_step
                new_boundary_pending = (
                    type(pending_boundary_step) is int
                    and pending_boundary_step > recovery.boundary_step
                )

                if recovered.returncode != 0:
                    record_runtime_failure(recovered)
                    if controller.terminal:
                        break
                    if new_boundary_pending:
                        break
                    continue
                if same_boundary_pending:
                    raise RuntimeError("Codex did not enter retry recovery")
                break

            if controller.terminal:
                continue
            if recovery_session_id is None:
                raise RuntimeError("Retry recovery created no Codex session")
            session_id = recovery_session_id
            continue

        if checkpoint_marker.exists():
            if session_id is None:
                raise RuntimeError(
                    "Codex stopped for compact without a resumable session id"
                )

            current_step = controller.step_index
            if current_step == last_checkpoint_step:
                compact_cycles_without_action += 1
            else:
                last_checkpoint_step = current_step
                compact_cycles_without_action = 1
            if compact_cycles_without_action >= MAX_COMPACT_CYCLES_WITHOUT_ACTION:
                raise RuntimeError(
                    "Compact checkpoint loop repeated without an environment action"
                )

            if not checkpoint_ready.exists() or not working_path.is_file():
                checkpoint_step = controller.step_index
                completed_attempts = 0
                while not checkpoint_ready.exists() or not working_path.is_file():
                    if completed_attempts >= MAX_CHECKPOINT_ATTEMPTS:
                        raise RuntimeError("Codex did not write the compact checkpoint")
                    checkpoint = runner.resume_checkpoint(session_id, CHECKPOINT_PROMPT)
                    results.append(checkpoint)
                    if checkpoint.session_id != session_id:
                        raise RuntimeError(
                            "Compact checkpoint did not resume the exact Codex session"
                        )
                    if controller.terminal:
                        break
                    if controller.step_index != checkpoint_step:
                        raise RuntimeError(
                            "Compact checkpoint turn applied an environment action"
                        )
                    if checkpoint.returncode != 0:
                        record_runtime_failure(checkpoint)
                        continue
                    completed_attempts += 1
                if controller.terminal:
                    continue

            checkpoints += 1
            if checkpoints > controller.max_steps + 1:
                raise RuntimeError("Compact checkpoint loop made no bounded progress")

            previous_session_id = session_id
            recovery = controller.begin_compact_recovery()
            restore_marker = controller.compact_restore_marker
            recovery_session_id: str | None = None

            while restore_marker is not None and restore_marker.exists():
                if recovery_session_id is None:
                    recovered = runner.run_recovery_task(recovery)
                else:
                    recovered = runner.resume_recovery_task(
                        recovery_session_id,
                        recovery,
                    )
                results.append(recovered)

                if recovered.session_id is not None:
                    if recovery_session_id is None:
                        if recovered.session_id == previous_session_id:
                            raise RuntimeError(
                                "Compact recovery did not create its required fresh session"
                            )
                        recovery_session_id = recovered.session_id
                    elif recovered.session_id != recovery_session_id:
                        raise RuntimeError(
                            "Compact recovery did not resume the exact Codex session"
                        )

                if recovered.returncode != 0:
                    record_runtime_failure(recovered)
                    if controller.terminal:
                        break
                    continue
                if restore_marker.exists():
                    raise RuntimeError("Codex did not enter compact recovery")

            if controller.terminal:
                continue
            if recovery_session_id is None:
                raise RuntimeError("Compact recovery created no Codex session")
            session_id = recovery_session_id
            continue

        current_result = results[-1]
        if getattr(current_result, "returncode", 0) != 0:
            if session_id is None:
                if controller.step_index != 0:
                    raise RuntimeError(
                        "Codex runtime failed after game actions without a session id"
                    )
                record_runtime_failure(current_result)
                results.append(runner.run_task(prompt))
                session_id = results[-1].session_id
                continue
            resume_player_runtime()
            continue

        current = results[-1]
        if current.returncode != 0 or session_id is None:
            return results

        progress = controller.initial_metadata().get("progress")
        completed = progress.get("completed") if isinstance(progress, dict) else None
        level_progress = completed if type(completed) is int else None
        continuation_scope = (session_id, level_progress)
        if continuation_scope != nonterminal_continuation_scope:
            nonterminal_continuation_scope = continuation_scope
            continued = runner.resume_nonterminal_task(
                session_id,
                NONTERMINAL_CONTINUATION_PROMPT,
            )
            results.append(continued)
            if continued.session_id is not None and continued.session_id != session_id:
                raise RuntimeError("Nonterminal continuation changed Codex session")
            session_id = continued.session_id or session_id
            continue

        controller.request_fresh_recovery()


def write_compact_restore_hook(visible_dir: Path, codex_home: Path) -> None:
    codex_dir = visible_dir / ".codex"
    codex_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        codex_home / "hooks.json",
        {
            "hooks": {
                "PreCompact": [
                    {
                        "matcher": "auto|manual",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "sh .codex/pre_compact_checkpoint.sh",
                                "statusMessage": "Saving game checkpoint",
                            }
                        ],
                    }
                ]
            }
        },
    )
    pre_script_path = codex_dir / "pre_compact_checkpoint.sh"
    pre_script_path.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "set -eu",
                "umask 077",
                "cat >/dev/null",
                'retry="$CODEX_HOME/retry_boundary.pending"',
                'if [ -f "$retry" ]; then',
                '  printf \'%s\\n\' \'{"continue":false,"stopReason":"Starting a fresh retry context."}\'',
                "  exit 0",
                "fi",
                'request="$CODEX_HOME/compact_checkpoint.requested"',
                ': > "$request"',
                'printf \'%s\\n\' \'{"continue":false,"stopReason":"Save the compact checkpoint before continuing."}\'',
                "",
            ]
        ),
        encoding="utf-8",
    )
    pre_script_path.chmod(0o555)
