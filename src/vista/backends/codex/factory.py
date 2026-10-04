"""Bind general tasks to the app-server transport."""

from __future__ import annotations

from pathlib import Path

from vista.core.profiles import CompactionMode
from vista.core.runtime import TaskRuntime

from ..binding import (
    prepare_player,
    recovery_prompts,
    task_timeout as resolve_timeout,
    tool_interface,
)
from ..session import BackendSession, preflight, prepare_directory


def create_backend(
    runtime: TaskRuntime,
    *,
    directory: Path,
    auth_file: Path,
    binary: Path | None = None,
    ca_bundle: Path | None = None,
    task_timeout: int | None = None,
    image: str | None = None,
) -> BackendSession:
    from vista.backends.codex.lifecycle import write_compact_restore_hook
    from vista.backends.codex.runner import DockerCodexRunner, prepare_codex_home

    preflight(runtime, "codex")
    visible, state, io = prepare_directory(directory)
    home = prepare_codex_home(state)
    prepare_player(runtime, visible, home)
    write_compact_restore_hook(visible, home)
    runner = DockerCodexRunner(
        controller=runtime.context,
        tool_interface=tool_interface(runtime, "codex"),
        recovery_prompts=recovery_prompts(runtime),
        native_compaction=runtime.experiment.profile.policy.compaction
        == CompactionMode.NATIVE,
        visible_dir=visible,
        codex_home=home,
        auth_file=auth_file,
        io_dir=io,
        codex_bin=binary,
        ca_bundle=ca_bundle,
        model=runtime.experiment.model,
        reasoning_effort=runtime.experiment.effort,
        task_timeout=resolve_timeout(runtime, task_timeout),
        **({"image": image} if image is not None else {}),
    )
    return BackendSession(runtime, runner, kind="codex")
