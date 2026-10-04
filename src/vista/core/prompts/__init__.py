"""Shared VISTA method prompts; benchmarks own their task objectives."""

from importlib.resources import files


def vista_method(backend: str) -> str:
    if backend not in {"codex", "claude"}:
        raise ValueError("Unknown backend")
    return files(__package__).joinpath(f"{backend}.md").read_text(encoding="utf-8")
