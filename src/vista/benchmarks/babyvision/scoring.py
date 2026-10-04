"""Host-side scoring: a deterministic match now, the official judge separately.

The official score is an LLM judge's verdict on the extracted answer. That
judge is a paid model call with its own identity, so it is a separate step
(`judge`) whose output is labeled as such. The exact match reported with each
run is a strict deterministic reading and is never called the official score.
"""

from __future__ import annotations

import json
import re
import statistics
import unicodedata
from typing import Any, Callable, Mapping

import requests

from .official import judge_prompt, parse_judge

# babyvision_eval/compute_score.py::type_match, for the subtype ordering.
TYPE_ORDER = {
    "Fine-grained Discrimination": "1",
    "Visual Tracking": "2",
    "Spatial Perception": "3",
    "Visual Pattern Recognition": "4",
}


def choice_label(value: str) -> str | None:
    """The letter a choice answer names: `B`, `(B)`, `B.`, `\\text{B}`, `option b`."""
    candidate = value.strip()
    wrapper = re.fullmatch(r"\\(?:text|mathrm|mathbf)\s*\{(.*)\}", candidate, re.DOTALL)
    if wrapper:
        candidate = wrapper.group(1).strip()
    # After the letter: whitespace, punctuation, a TeX spacing/text command
    # (such as `(C)\ \text{...}`), or the end.
    match = re.match(
        r"^(?:option\s*)?[\(\[]?\s*([a-z])\s*[\)\]]?(?:\s|[-:.,\\]|$)",
        candidate,
        flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


TEX_WRAPPERS = (
    r"\\(?:text|textbf|textit|mathrm|mathbf|mathit|operatorname)\s*\{([^{}]*)\}"
)


def normalize(value: str) -> str:
    """Case, width, whitespace and TeX wrapping do not make an answer different.

    For example, `\\text{A}` normalizes to `a`.
    """
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = text.replace("，", ",").replace("–", "-").replace("—", "-")
    previous = None
    while previous != text:
        previous = text
        # Unwrapped with a space on each side so `1\\to\\text{X}` keeps its
        # word boundaries; the list split below drops the spaces again.
        text = re.sub(TEX_WRAPPERS, r" \1 ", text)
    text = re.sub(r"\\(?:[,;:!]|\s|quad\b|qquad\b)", "", text)
    # TeX-escaped punctuation is the punctuation: `\\#3` is `#3`.
    text = re.sub(r"\\([#%&_$])", r"\1", text)
    # `1\\to X`, `1 -> X`, `1 → X` and `1 - X` all pair 1 with X.
    text = re.sub(r"\\(?:to|rightarrow)\b|->|→|—|–", "-", text)
    text = re.sub(r"\s*-\s*", "-", text)
    # Word lists are compared as lists: "better late than never" and
    # "Better,late,than,never" are the same answer, "10,7,7" and "10,4,10" are
    # not, and "(4,7)" keeps its shape on both sides.
    text = ",".join(part for part in re.split(r"[,;\s]+", text) if part)
    text = text.strip("`\"'")
    # One pair of brackets around the whole answer is presentation:
    # `(4,2,6,5)` is `4,2,6,5`. `(4,7)` against `(4,7)` loses both, equally.
    if text and text[:1] in "([{" and text[-1:] == {"(": ")", "[": "]", "{": "}"}[text[0]]:
        inner = text[1:-1]
        depth = 0
        for char in inner:
            depth += char in "([{"
            depth -= char in ")]}"
            if depth < 0:
                break
        else:
            text = inner
    return text


def exact_match(extracted: str | None, expected: str, *, answer_type: str) -> bool:
    if extracted is None:
        return False
    if answer_type == "choice":
        return choice_label(extracted) == expected.upper()
    normalized = normalize(extracted)
    return bool(normalized) and normalized == normalize(expected)


# -- the official judge ---------------------------------------------------------


class JudgeError(RuntimeError):
    pass


def openai_chat(
    base_url: str, api_key: str, model: str, *, timeout: float = 1800.0
) -> Callable[[str], str]:
    """The official script's judge call: one user message, the reply text back."""
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def complete(prompt: str) -> str:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout,
        )
        if response.status_code != 200:
            raise JudgeError(f"Judge returned HTTP {response.status_code}")
        try:
            return str(response.json()["choices"][0]["message"]["content"])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise JudgeError("Judge reply was not a chat completion") from exc

    return complete


def judge_answer(
    complete: Callable[[str], str],
    *,
    question: str,
    ground_truth: str,
    extracted: str | None,
) -> dict[str, Any]:
    """One verdict, with the raw reply kept so a parse can be audited."""
    reply = complete(judge_prompt(question, ground_truth, extracted))
    verdict = parse_judge(reply)
    return {
        "judge_reply": reply,
        # The official script scores an unparseable reply as False.
        "llm_judge_result": bool(verdict),
        "judge_parsed": verdict is not None,
    }


def official_record(result: Mapping[str, Any]) -> dict[str, Any]:
    """One row in the layout `compute_score.py` reads.

    A row without a judge verdict (a failed run, or one not judged yet) is
    False, as the official script scores its own failed attempts; the exact
    match never stands in for the judge here.
    """
    return {
        "Id": result["task_id"],
        "Question": result["question"],
        "ModelResult": result.get("raw_output") or "",
        "GroundTruth": result["ground_truth"],
        "ExtractedAnswer": result.get("extracted_answer") or "",
        "LLMJudgeResult": bool(result.get("judged"))
        and bool(result.get("llm_judge_result")),
        "Type": result["type"],
        "Subtype": result["subtype"],
    }


# -- aggregation -------------------------------------------------------------------


def accuracy(
    rows: list[Mapping[str, Any]], key: str, *, planned: int | None = None
) -> dict[str, Any]:
    """compute_score.py's reading: every planned question counts, errors are wrong.

    A run that never wrote a result (killed, crashed before its record) is
    still a planned question, so `planned` is the denominator when known.
    Category scores are unavailable when missing records leave their denominators
    unknown.
    """
    total = max(len(rows), planned or 0)
    correct = sum(1 for row in rows if row.get(key))
    by_type: dict[str, dict[str, int]] = {}
    by_subtype: dict[str, dict[str, int]] = {}
    for row in rows:
        category = row["type"]
        label = f"{TYPE_ORDER.get(category, '0')}{category}/{row['subtype']}"
        for table, name in ((by_type, category), (by_subtype, label)):
            cell = table.setdefault(name, {"total": 0, "correct": 0})
            cell["total"] += 1
            cell["correct"] += int(bool(row.get(key)))
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "by_type": {
            name: {**cell, "accuracy": cell["correct"] / cell["total"]}
            for name, cell in sorted(by_type.items())
        }
        if len(rows) == total
        else None,
        "by_subtype": {
            name: {**cell, "accuracy": cell["correct"] / cell["total"]}
            for name, cell in sorted(by_subtype.items())
        }
        if len(rows) == total
        else None,
    }


def aggregate(results: list[Mapping[str, Any]], *, planned: int, passes: int) -> dict:
    """Summary over passes; a missing run is a wrong answer, never dropped."""
    if planned < 0 or passes < 1:
        raise ValueError("Planned questions must be nonnegative and passes positive")
    actual = [(r["task_id"], r.get("pass", 1)) for r in results]
    if len(actual) != len(set(actual)):
        raise ValueError("Duplicate question/pass results; score independent runs separately")
    if any(number < 1 or number > passes for _, number in actual):
        raise ValueError("Result pass is outside the planned range")
    if len({task for task, _ in actual}) > planned:
        raise ValueError("More result questions than planned")
    completed = [row for row in results if row.get("completed")]
    key = (
        "llm_judge_result"
        if completed and all(row.get("judged") for row in completed)
        else "exact_match"
    )
    per_pass = []
    for number in range(1, passes + 1):
        rows = [row for row in results if row.get("pass", 1) == number]
        per_pass.append(
            {
                "pass": number,
                "completed": sum(1 for r in rows if r.get("completed")),
                "missing": max(planned - len(rows), 0),
                **accuracy(rows, key, planned=planned),
            }
        )
    values = [p["accuracy"] for p in per_pass if p["accuracy"] is not None]
    return {
        "score_source": "official-llm-judge"
        if key == "llm_judge_result"
        else "normalized-exact-match",
        "planned": planned * passes,
        "completed": sum(1 for r in results if r.get("completed")),
        "answered": sum(1 for r in results if r.get("extracted_answer") is not None),
        "correct": sum(1 for r in results if r.get(key)),
        "mean_accuracy": statistics.fmean(values) if values else None,
        "std_accuracy": statistics.pstdev(values)
        if len(values) > 1
        else (0.0 if values else None),
        "passes": per_pass,
        "incomplete": sorted(
            f"{r['task_id']}/p{r.get('pass', 1)}"
            for r in results
            if not r.get("completed")
        ),
    }


def dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
