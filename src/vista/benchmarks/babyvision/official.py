"""The official MLLM evaluation recipe, copied from `babyvision_eval` at the pin.

One request per question: the image, then the question text with the boxed
instruction. The last `\\boxed{}` is the answer; an LLM judge compares it with
the ground truth. What is verbatim here is the prompt text, the choice
format, the extraction pattern and the judge prompt; the transport is not.
"""

from __future__ import annotations

from .dataset import Question

ANSWER_INSTRUCTION = (
    "Think about the question and give your final answer in \\boxed{Answer} format."
)

# babyvision_eval/utils.py::LLM_JUDGE_PROMPT, verbatim.
JUDGE_PROMPT = """You are a careful and strict evaluator. You will be given:

1. **Question**
2. **Ground Truth Answer** (correct answer)
3. **Model Output** (answer from another model)

**Your goal:** Determine if the Model Output **accurately matches** the Ground Truth Answer in meaning.

* Matching means: the facts, entities, and key details are equivalent, even if phrasing differs.
* Not matching means: the Model Output is wrong, incomplete, contains extra incorrect facts, or changes the meaning.

**Process (internal reasoning):**

1. Read and understand the Question, Ground Truth Answer, and Model Output.
2. Ignore small wording differences, formatting, or synonyms.
3. If all factual content matches, conclude `1`. Otherwise, conclude `0`.

**Important:**

* Think through your decision step-by-step **internally** before responding.
* In your final output, return **only** True or False, with no extra text or explanation.

**Output format:**

True

or

False

**Input:**

Question: {question},
Ground Truth Answer: {groundtruth},
Model Output: {modeloutput}
"""

def format_choices(options: tuple[str, ...]) -> str:
    """babyvision_eval/utils.py::format_choices."""
    if not options:
        return ""
    formatted = ""
    for index, choice in enumerate(options):
        formatted += f"({chr(65 + index)}) {choice}\n"
    return formatted.strip()


def question_text(question: Question) -> str:
    """The exact text the official script sends beside the image."""
    text = question.question
    if question.answer_type == "choice":
        text = text + "\nChoices:\n" + format_choices(question.options)
    return text + "\n" + ANSWER_INSTRUCTION


def extract_boxed_answer(text: str | None) -> str | None:
    """babyvision_eval/utils.py::extract_boxed_answer, on the same `regex` engine."""
    if text is None:
        return None
    try:
        import regex
    except ImportError as exc:  # pragma: no cover - environment
        raise RuntimeError(
            "The official answer extraction needs the `regex` package: "
            "pip install -e '.[babyvision]'"
        ) from exc
    pattern = r"\\boxed\{((?:[^{}]|{(?:[^{}]|{.*})*})*)\}"
    matches = regex.findall(pattern, text)
    if matches:
        return matches[-1]
    pattern_alt = r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>"
    matches_alt = regex.findall(pattern_alt, text)
    if matches_alt:
        return matches_alt[-1].strip()
    return None


def judge_prompt(question: str, ground_truth: str, extracted: str | None) -> str:
    """The judge sees the extracted answer, as the official script sends it."""
    return JUDGE_PROMPT.format(
        question=question, groundtruth=ground_truth, modeloutput=extracted
    )


def parse_judge(reply: str) -> bool | None:
    """The official reading of a judge reply; None is its unparseable warning."""
    text = str(reply).strip().lower()
    if "true" in text:
        return True
    if "false" in text:
        return False
    return None
