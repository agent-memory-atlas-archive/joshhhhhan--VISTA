"""VISTA on a static question: one image, one question, one final answer."""

from __future__ import annotations

from typing import Any

from vista.core.profiles import (
    AgentProfile,
    CompactionMode,
    ContextMode,
    Experiment,
    InteractionMode,
    PromptBundle,
    SessionPolicy,
)

from .dataset import Question
from .official import question_text

PROTOCOL = "babyvision-mllm-one-question-final-answer"
EVALUATOR = (
    "last-boxed-answer + normalized-exact-match; official llm-judge as a "
    "separate labeled step"
)


# What the agent is told; the question itself is the first input.
SHARED_INSTRUCTIONS = (
    "You are answering one visual question about one supplied image. The image "
    "is the only evidence; there is no game, no environment action, and nothing "
    "to explore beyond it. Your final response is your answer to the question "
    "and ends the task."
)

VISTA_METHOD_OPENING = (
    "# Visual question task\n\n"
    "Before answering, gather the evidence the question actually needs from "
    "the supplied image."
)
VISTA_METHOD_TOOLS = (
    " Use `inspect` to view enlarged regions where detail matters, and "
    "`read_pixels` to measure exact colors where a judgement depends on "
    "them. Say what each view is meant to settle, then check whether it did."
)
VISTA_METHOD_CLOSING = (
    "Build a short, revisable account of what the image shows and update it as "
    "the evidence changes what is supported. Answer only when the evidence "
    "supports one answer."
)


def vista_method() -> str:
    return VISTA_METHOD_OPENING + VISTA_METHOD_TOOLS + "\n\n" + VISTA_METHOD_CLOSING


def system_prompt() -> str:
    return SHARED_INSTRUCTIONS + "\n\n" + vista_method()


def make_profile(
    question: Question,
    *,
    backend: str = "codex",
) -> AgentProfile:
    if backend not in {"codex", "claude"}:
        raise ValueError("Unknown backend")
    initial = question_text(question)
    return AgentProfile(
        "babyvision-mllm/vista/inspect-read-pixels",
        ("inspect", "read_pixels"),
        PromptBundle(system_prompt(), initial),
        # One question per context; native compaction if it ever runs that long.
        SessionPolicy(
            context=ContextMode.CONTINUOUS,
            interaction=InteractionMode.TOOLS,
            compaction=CompactionMode.NATIVE,
        ),
    )


def make_experiment(
    profile: AgentProfile,
    question: Question,
    *,
    backend: str,
    model: str,
    effort: str,
    revision: str,
    image: dict[str, Any],
    backend_segments: int | None = None,
    tool_calls: int | None = None,
) -> Experiment:
    return Experiment(
        "BabyVision",
        revision,
        PROTOCOL,
        profile,
        backend,
        model,
        effort,
        {
            "track": "mllm",
            "task_id": question.task_id,
            "type": question.category,
            "subtype": question.subtype,
            "answer_type": question.answer_type,
            "pixels": "original image file bytes",
            "coordinate_scale": 1.0,
            "image": dict(image),
            "question_format": "official question, choices and boxed instruction",
        },
        {
            # The first final response is the answer; there is no second.
            "final_answers": 1,
            **(
                {"backend_segments": backend_segments}
                if backend_segments is not None
                else {}
            ),
            **({"tool_calls": tool_calls} if tool_calls is not None else {}),
        },
        EVALUATOR,
    )
