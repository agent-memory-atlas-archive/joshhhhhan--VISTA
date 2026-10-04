"""One BabyVision question: an immutable image and a final answer, nothing else."""

from __future__ import annotations

from typing import Any

from vista.core.contracts import (
    Effect,
    ObservationBundle,
    PublicTask,
    SubmissionSpec,
    ToolBinding,
    ToolReply,
)

from .dataset import Question, question_observation
from .official import question_text


class BabyVisionSession:
    """No environment action exists; the model's final response is its answer.

    The session records the raw text and ends. Extraction and scoring happen
    on the host afterwards, so nothing about the answer key is reachable
    from here.
    """

    def __init__(
        self, question: Question, *, observation: ObservationBundle | None = None
    ) -> None:
        self.question = question
        observation = observation or question_observation(question)
        visual = observation.observations[0].views[0]
        self.task = PublicTask(
            f"babyvision/{question.task_id}",
            question_text(question),
            {
                "task_id": question.task_id,
                "type": question.category,
                "subtype": question.subtype,
                "answer_type": question.answer_type,
                "image": {
                    "width": visual.width,
                    "height": visual.height,
                    "mime_type": visual.mime_type,
                },
            },
            observation,
        )
        self.submission = SubmissionSpec(Effect.SUBMISSION)
        self.tools: tuple[ToolBinding, ...] = ()
        self.raw_output: str | None = None
        self.closed = False

    @property
    def answered(self) -> bool:
        return self.raw_output is not None

    def submit(self, answer: str) -> ToolReply:
        if self.answered:
            return ToolReply("The answer has already been recorded.", success=False)
        self.raw_output = answer
        event = "Answer recorded; the task is complete."
        return ToolReply(event, finished=True, event_text=event)

    def result(self) -> dict[str, Any]:
        return {
            "task_id": self.question.task_id,
            "type": self.question.category,
            "subtype": self.question.subtype,
            "answer_type": self.question.answer_type,
            "question": self.task.objective,
            "answered": self.answered,
            "raw_output": self.raw_output,
        }

    def close(self) -> None:
        self.closed = True
