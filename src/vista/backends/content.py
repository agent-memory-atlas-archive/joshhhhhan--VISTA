"""Preserve benchmark-owned text/image ordering in native CLI inputs."""

from __future__ import annotations

from typing import Any

from vista.core.runtime import TaskInput


def ordered_content(task_input: TaskInput, backend: str) -> list[dict[str, Any]]:
    content = []
    labels = iter(task_input.image_labels)
    for block in task_input.blocks or (task_input.text, task_input.observation):
        if isinstance(block, str):
            content.append({"type": "text", "text": block})
            continue
        for image in block.images():
            content.append({"type": "text", "text": next(labels, image["steer_text"])})
            if backend == "codex":
                content.append(
                    {
                        "type": "image",
                        "url": f"data:{image['mime_type']};base64,{image['data']}",
                        "detail": image["detail"],
                    }
                )
            elif backend == "claude":
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image["mime_type"],
                            "data": image["data"],
                        },
                    }
                )
            else:
                raise ValueError("Unknown native image backend")
    return content
