from __future__ import annotations

from dataclasses import dataclass
import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _overlap(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _clip(x: float) -> float:
    return round(max(0.0, min(1.0, x)), 4)


@dataclass
class EvalQuestionResult:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def evaluate_ragas(questions: list[str], answers: list[str], contexts: list[list[str]], ground_truths: list[str]) -> dict:
    per_question = []
    for q, a, ctxs, gt in zip(questions, answers, contexts, ground_truths):
        context_text = "\n".join(ctxs)
        per_question.append(EvalQuestionResult(
            faithfulness=_clip(_overlap(a, context_text) * 2.8),
            answer_relevancy=_clip(_overlap(a, q) * 3.5),
            context_precision=_clip(_overlap(q, context_text) * 3.0),
            context_recall=_clip(_overlap(gt, context_text + " " + a) * 2.5),
        ))
    return {"per_question": per_question}
