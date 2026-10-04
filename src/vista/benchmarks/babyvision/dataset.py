"""The pinned official dataset, split into what the player may see and what it may not.

The official metadata carries the answer (`blankAns`, `choiceAns`) and a
worked solution (`coT`) beside every question. Only the question, its choices
and its image are public; the rest never leaves this module except through
the host-side answer key.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from vista.core.contracts import Observation, ObservationBundle, Visual

BABYVISION_REVISION = "7f92fd4b1dc1c68b7b936a9bc09c68b4a944a55a"
DATASET_DIR = Path("data") / "babyvision_data"
METADATA = "meta_data.jsonl"
VIEW = "question_image"

# The fields that may reach the player, by their official names.
PUBLIC_FIELDS = ("taskId", "type", "subtype", "image", "question", "ansType", "options")
ANSWER_TYPES = ("blank", "choice")
IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp"}

# Named subsets; every other selection is explicit task ids or subtypes.
TASK_SETS: dict[str, tuple[str, ...] | None] = {
    "full": None,
    # The whole Visual Tracking type: 83 questions in five subtypes.
    "visual-tracking": (
        "Recognize numbers and letters",
        "Maze",
        "Connect the lines",
        "Metro map",
        "Lines Observation",
    ),
    # Default subset: 48 questions across three Visual Tracking subtypes.
    "tracking48": ("Maze", "Connect the lines", "Lines Observation"),
}
DEFAULT_TASK_SET = "tracking48"


@dataclass(frozen=True)
class Question:
    """One public question: exactly the official inputs, nothing derived."""

    task_id: int
    category: str
    subtype: str
    image: Path
    question: str
    answer_type: str
    options: tuple[str, ...]

    def manifest(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "type": self.category,
            "subtype": self.subtype,
            "image": self.image.name,
            "answer_type": self.answer_type,
            "options": list(self.options),
        }


@dataclass(frozen=True)
class Dataset:
    root: Path
    revision: str
    # sha256 of meta_data.jsonl: two checkouts with equal counts are the same
    # snapshot only if this agrees.
    fingerprint: str
    questions: tuple[Question, ...]
    # Host-only: the official ground truth per task id.
    answers: Mapping[int, str]

    def question(self, task_id: int) -> Question:
        for question in self.questions:
            if question.task_id == task_id:
                return question
        raise KeyError(f"Unknown BabyVision task id {task_id}")

    def select(
        self,
        *,
        task_ids: list[int] | None = None,
        subtypes: list[str] | None = None,
        task_set: str = "full",
    ) -> tuple[Question, ...]:
        """Explicit ids, then explicit subtypes, then the named set, in that order."""
        if task_set not in TASK_SETS:
            raise ValueError(f"Unknown task set {task_set!r}")
        if task_ids:
            known = {question.task_id for question in self.questions}
            missing = sorted(set(task_ids) - known)
            if missing:
                raise ValueError(f"Unknown BabyVision task ids: {missing}")
            wanted = set(task_ids)
            return tuple(q for q in self.questions if q.task_id in wanted)
        selected = subtypes or TASK_SETS[task_set]
        if selected is None:
            return self.questions
        available = {question.subtype for question in self.questions}
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(f"Unknown BabyVision subtypes: {missing}")
        return tuple(q for q in self.questions if q.subtype in set(selected))


def checkout_revision(root: Path) -> str | None:
    """The git revision of the dataset checkout, or None outside a checkout."""
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    revision = completed.stdout.strip()
    if completed.returncode != 0 or not revision:
        return None
    return revision


def load_dataset(root: Path, *, revision: str | None = None) -> Dataset:
    """Load the official metadata from a checkout.

    `revision` names what the caller believes the checkout is; when omitted
    it is read from git and must equal the pin. Equal counts between local,
    GitHub and Hugging Face copies do not make them the same snapshot, so
    the metadata's own hash is recorded beside the revision either way.
    """
    root = root.expanduser().resolve()
    path = root / DATASET_DIR / METADATA
    if not path.is_file():
        raise FileNotFoundError(
            f"BabyVision metadata not found at {path}; unpack data/babyvision_data.zip"
        )
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    if revision is None:
        revision = checkout_revision(root)
        if revision is None:
            raise RuntimeError(f"{root} is not a git checkout of BabyVision")
        if revision != BABYVISION_REVISION:
            raise RuntimeError(
                f"BabyVision must be pinned to {BABYVISION_REVISION}; found {revision}"
            )
    questions: list[Question] = []
    answers: dict[int, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid BabyVision metadata JSON on line {number}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"BabyVision metadata line {number} is not an object")
        question, answer = _parse(path.parent, payload, number)
        if question.task_id in answers:
            raise ValueError(f"Duplicate BabyVision task id {question.task_id}")
        questions.append(question)
        answers[question.task_id] = answer
    if not questions:
        raise ValueError("BabyVision metadata contains no tasks")
    return Dataset(root, revision, fingerprint, tuple(questions), answers)


def _parse(root: Path, payload: Mapping[str, Any], number: int) -> tuple[Question, str]:
    try:
        task_id = int(payload["taskId"])
        category = _text(payload["type"], "type")
        subtype = _text(payload["subtype"], "subtype")
        question = _text(payload["question"], "question")
        answer_type = _text(payload["ansType"], "ansType")
        relative = Path(_text(payload["image"], "image"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid BabyVision metadata on line {number}: {exc}"
        ) from exc
    if answer_type not in ANSWER_TYPES:
        raise ValueError(f"Unsupported BabyVision answer type on line {number}")
    image = (root / relative).resolve()
    if not image.is_relative_to(root) or not image.is_file():
        raise ValueError(f"Invalid BabyVision image on line {number}: {relative}")
    raw_options = payload.get("options", [])
    if not isinstance(raw_options, list) or not all(
        isinstance(option, str) and option.strip() for option in raw_options
    ):
        raise ValueError(f"Invalid BabyVision options on line {number}")
    options = tuple(option.strip() for option in raw_options)
    # The official evaluator's answer key, verbatim: a choice letter or the
    # blank answer text.
    if answer_type == "choice":
        try:
            index = int(payload["choiceAns"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid BabyVision choice answer on line {number}"
            ) from exc
        if not 0 <= index < len(options):
            raise ValueError(f"BabyVision choice answer out of range on line {number}")
        answer = chr(65 + index)
    else:
        if options:
            raise ValueError(f"BabyVision blank task has choices on line {number}")
        answer = _text(payload.get("blankAns"), "blankAns")
    return (
        Question(task_id, category, subtype, image, question, answer_type, options),
        answer,
    )


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def question_visual(question: Question) -> Visual:
    """The official image file's own bytes; the model sees what the API saw."""
    from PIL import Image

    data = question.image.read_bytes()
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            mime_type = Image.MIME.get(image.format or "")
    except OSError as exc:
        raise ValueError(f"Could not read BabyVision image {question.image}") from exc
    if mime_type not in IMAGE_TYPES:
        raise ValueError(f"Unsupported BabyVision image format at {question.image}")
    return Visual(
        VIEW, f"babyvision-{question.task_id}", data, width, height, mime_type
    )


def question_observation(question: Question) -> ObservationBundle:
    return ObservationBundle((Observation("question", (question_visual(question),)),))
