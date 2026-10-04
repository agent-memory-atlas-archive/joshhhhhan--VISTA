"""Local runtime provenance checks; these never make model requests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def check_transport(backend: str, binary: Path | None) -> tuple[Path, dict[str, Any]]:
    if backend not in {"codex", "claude"}:
        raise ValueError("Unsupported backend")
    if backend == "codex":
        from vista.backends.codex.runner import (
            PINNED_CODEX_CLI_VERSION,
            PLAYER_IMAGE,
            default_native_codex_bin,
        )

        executable = (binary or default_native_codex_bin()).resolve()
        expected = PINNED_CODEX_CLI_VERSION
    else:
        from vista.backends.claude.runner import (
            PINNED_CLAUDE_VERSION,
            PLAYER_IMAGE,
            default_claude_bin,
        )

        executable = (binary or default_claude_bin()).resolve()
        expected = PINNED_CLAUDE_VERSION
    result = subprocess.run(
        [str(executable), "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    version = result.stdout.strip()
    if (version.split()[0] if backend == "claude" and version else version) != expected:
        raise RuntimeError(f"Expected {expected}, found {version or 'unknown'}")
    image = subprocess.run(
        ["docker", "image", "inspect", PLAYER_IMAGE, "--format", "{{json .Id}}"],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    image_id = json.loads(image.stdout)
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise RuntimeError("Docker returned no immutable image identity")
    with executable.open("rb") as stream:
        binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return executable, {
        "validation": "local-version-and-image-check",
        "version": version,
        "binary_sha256": binary_sha256,
        "container_image": PLAYER_IMAGE,
        "container_image_id": image_id,
    }


def source_fingerprint() -> dict[str, str]:
    source_root = Path(__file__).resolve().parents[2]
    sources = {}
    for package in ("vista",):
        for path in sorted((source_root / package).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md"}:
                with path.open("rb") as stream:
                    sources[str(path.relative_to(source_root))] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()
    return sources
