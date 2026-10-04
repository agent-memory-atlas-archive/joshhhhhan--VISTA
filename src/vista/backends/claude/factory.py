"""Bind general tasks to the native-MCP transport."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from vista.core.profiles import CompactionMode
from vista.core.runtime import TaskRuntime

from ..binding import (
    prepare_player,
    recovery_prompts,
    task_timeout as resolve_timeout,
    tool_interface,
)
from ..session import BackendSession, preflight, prepare_directory
from .provider import ANTHROPIC_PROVIDER, ClaudeCodeProvider


def create_backend(
    runtime: TaskRuntime,
    *,
    directory: Path,
    oauth_token: str | None,
    binary: Path | None = None,
    ca_bundle: Path | None = None,
    expected_model_id: str | None = None,
    task_timeout: int | None = None,
    image: str | None = None,
    rate_limit_observer: Callable[[Any], None] | None = None,
    credential_yield_requested: Callable[[], bool] | None = None,
    credential_relay: Callable[[], str] | None = None,
    provider: ClaudeCodeProvider = ANTHROPIC_PROVIDER,
    provider_credential: str | None = None,
    provider_base_url: str | None = None,
    auto_compact_window: int | None = None,
    max_output_tokens: int | None = None,
) -> BackendSession:
    from vista.backends.claude.runner import (
        CLAUDE_AUTO_COMPACT_PERCENT,
        CLAUDE_AUTO_COMPACT_WINDOW,
        ClaudeCodeRunner,
    )

    preflight(runtime, "claude")
    if provider.native and not oauth_token:
        raise ValueError("The Anthropic provider requires an OAuth token")
    if not provider.native and (credential_relay or credential_yield_requested):
        raise ValueError("Credential relay is only defined for the Anthropic provider")
    # Explicit arguments override provider defaults, then transport defaults.
    window = (
        auto_compact_window
        if auto_compact_window is not None
        else provider.auto_compact_window
        if provider.auto_compact_window is not None
        else CLAUDE_AUTO_COMPACT_WINDOW
    )
    visible, state, io = prepare_directory(directory)
    prepare_player(runtime, visible, io)
    runner = ClaudeCodeRunner(
        controller=runtime.context,
        tool_interface=tool_interface(runtime, "claude"),
        recovery_prompts=recovery_prompts(runtime),
        native_compaction=runtime.experiment.profile.policy.compaction
        == CompactionMode.NATIVE,
        visible_dir=visible,
        claude_config_dir=state,
        io_dir=io,
        claude_bin=binary,
        ca_bundle=ca_bundle,
        oauth_token=oauth_token,
        model=runtime.experiment.model,
        effort=runtime.experiment.effort,
        expected_model_id=expected_model_id,
        task_timeout=resolve_timeout(runtime, task_timeout),
        rate_limit_observer=rate_limit_observer,
        credential_yield_requested=credential_yield_requested,
        provider=provider,
        provider_credential=provider_credential,
        provider_base_url=provider_base_url,
        auto_compact_window=window,
        auto_compact_percent=CLAUDE_AUTO_COMPACT_PERCENT,
        max_output_tokens=max_output_tokens,
        **({"image": image} if image is not None else {}),
    )

    def confirm_model() -> None:
        from .preflight import run_runtime_preflight

        result, concrete = run_runtime_preflight(
            visible_dir=visible,
            config_dir=state / "preflight",
            io_dir=io / "preflight",
            claude_bin=runner.claude_bin,
            model=runner.model,
            effort=runner.effort,
            timeout=300,
            tool_interface=runner.tool_interface,
            oauth_token=oauth_token,
            rate_limit_observer=rate_limit_observer,
            ca_bundle=runner.ca_bundle,
            image=runner.image,
            native_compaction=runner.native_compaction,
            provider=provider,
            provider_credential=provider_credential,
            provider_base_url=provider_base_url,
            max_output_tokens=runner.max_output_tokens,
        )
        runner.expected_model_id = concrete
        runtime.artifacts.record(
            "model_preflight",
            {
                "resolved_model_id": concrete,
                "session_id": result.session_id,
            },
        )

    return BackendSession(
        runtime,
        runner,
        kind="claude",
        before_first_run=confirm_model if expected_model_id is None else None,
        credential_relay=credential_relay,
        credential_relay_needed=credential_yield_requested,
    )
