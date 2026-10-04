"""Shared bounds for native-player and environment operations."""

MAX_TASK_TIMEOUT_SECONDS = 2_000_000
ENV_CONNECT_TIMEOUT_SECONDS = 10
ENV_READ_TIMEOUT_SECONDS = 120
ENV_CONNECT_RETRIES = 3


def bounded_task_timeout(tool_timeout: int, maximum_calls: int) -> int:
    return min(tool_timeout * maximum_calls, MAX_TASK_TIMEOUT_SECONDS)
