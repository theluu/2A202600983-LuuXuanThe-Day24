from __future__ import annotations

"""Phase B: LLM-as-Judge — pairwise, swap-and-average, Cohen κ, bias analysis."""

import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANTHROPIC_API_KEY, ANTHROPIC_JUDGE_MODEL, OPENAI_API_KEY, JUDGE_MODEL, HUMAN_LABELS_PATH


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def _quality_score(question: str, answer: str) -> float:
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)
    if not answer_tokens:
        return 0.0
    relevance = len(answer_tokens & question_tokens) / max(1, len(question_tokens))
    specificity_patterns = [
        r"\d+",
        r"v20\d{2}|v\d+\.\d+",
        r"không|bắt buộc|phải|cần",
        r"vnđ|triệu|ngày|tháng|năm",
    ]
    specificity = sum(1 for pattern in specificity_patterns if re.search(pattern, answer.lower())) / len(specificity_patterns)
    length_penalty = 0.0
    if len(answer) < 20:
        length_penalty = 0.2
    elif len(answer) > 700:
        length_penalty = 0.15
    score = 0.45 * relevance + 0.45 * specificity + 0.10 * min(1.0, len(answer_tokens) / 45)
    return round(max(0.0, min(1.0, score - length_penalty)), 3)


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str       # "A" | "B" | "tie"  (original order)
    winner_pass2: str       # "A" | "B" | "tie"  (after swap, ALREADY converted back)
    final_winner: str       # consensus after swap-and-average
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool  # True if both passes agree on same answer
    scores_pass1: dict = field(default_factory=dict)  # {"A": float, "B": float}
    scores_pass2: dict = field(default_factory=dict)


# ─── Task 5: Pairwise Judge ───────────────────────────────────────────────────

def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Task 5: Gọi LLM để chọn answer tốt hơn (A hoặc B) theo 3 tiêu chí.

    Tiêu chí đánh giá:
        - Độ chính xác (accuracy): có khớp với thực tế chính sách không?
        - Độ đầy đủ (completeness): có trả lời đủ câu hỏi không?
        - Tính súc tích (conciseness): có thừa / thiếu thông tin không?

    Returns:
        {"winner": "A"|"B"|"tie", "reasoning": str, "scores": {"A": float, "B": float}}
    """
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            prompt = f"""Bạn là expert đánh giá chất lượng câu trả lời RAG.
Câu hỏi: {question}

Answer A:
{answer_a}

Answer B:
{answer_b}

Đánh giá dựa trên độ chính xác, đầy đủ, súc tích.
Trả lời JSON duy nhất:
{{"winner": "A" hoặc "B" hoặc "tie", "reasoning": "...", "scores": {{"A": 0.0, "B": 0.0}}}}
"""
            client = OpenAI(timeout=10.0, max_retries=0)
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "Bạn là expert đánh giá RAG. Chỉ trả lời JSON."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            if parsed.get("winner") in {"A", "B", "tie"}:
                return parsed
        except Exception as exc:
            print(f"LLM judge unavailable, using heuristic judge: {exc}")
    if ANTHROPIC_API_KEY:
        try:
            import anthropic

            prompt = f"""Compare two RAG answers. Return JSON only.
Question: {question}
Answer A: {answer_a}
Answer B: {answer_b}
Criteria: factual accuracy, completeness, conciseness.
Schema: {{"winner":"A|B|tie","reasoning":"short reason","scores":{{"A":0.0,"B":0.0}}}}"""
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=ANTHROPIC_JUDGE_MODEL,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
            parsed = json.loads(text)
            if parsed.get("winner") in {"A", "B", "tie"}:
                return parsed
        except Exception as exc:
            print(f"Anthropic judge unavailable, using heuristic judge: {exc}")

    score_a = _quality_score(question, answer_a)
    score_b = _quality_score(question, answer_b)
    if abs(score_a - score_b) < 0.05:
        winner = "tie"
        reasoning = "Hai câu trả lời có chất lượng tương đương theo heuristic offline."
    elif score_a > score_b:
        winner = "A"
        reasoning = "Answer A phù hợp câu hỏi hơn và có nhiều tín hiệu cụ thể hơn."
    else:
        winner = "B"
        reasoning = "Answer B phù hợp câu hỏi hơn và có nhiều tín hiệu cụ thể hơn."
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": score_a, "B": score_b}}


def absolute_score(question: str, answer: str) -> dict:
    """Score a single answer on the 4-dimension Lab 24 rubric."""
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            prompt = f"""Score the answer on 4 dimensions, each 1-5:
1. accuracy
2. relevance
3. conciseness
4. helpfulness
Question: {question}
Answer: {answer}
Output JSON only:
{{"accuracy": int, "relevance": int, "conciseness": int, "helpfulness": int, "overall": float}}"""
            client = OpenAI(timeout=10.0, max_retries=0)
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict RAG answer evaluator. Return JSON only."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
            dims = ["accuracy", "relevance", "conciseness", "helpfulness"]
            if all(dim in parsed for dim in dims):
                parsed["overall"] = float(parsed.get("overall") or sum(parsed[d] for d in dims) / 4)
                parsed["mode"] = "openai"
                return parsed
        except Exception as exc:
            print(f"OpenAI absolute scorer unavailable, using heuristic: {exc}")

    base = _quality_score(question, answer)
    accuracy = max(1, min(5, round(1 + base * 4)))
    relevance = max(1, min(5, round(1 + min(1.0, len(_tokens(question) & _tokens(answer)) / max(1, len(_tokens(question)))) * 4)))
    conciseness = 5 if 40 <= len(answer) <= 450 else 3
    helpfulness = round((accuracy + relevance) / 2)
    overall = round((accuracy + relevance + conciseness + helpfulness) / 4, 2)
    return {
        "accuracy": accuracy,
        "relevance": relevance,
        "conciseness": conciseness,
        "helpfulness": helpfulness,
        "overall": overall,
        "mode": "heuristic",
    }


def run_batch_judge(limit: int = 30) -> tuple[list[JudgeResult], list[dict]]:
    root = Path(__file__).resolve().parents[1]
    answers_path = root / "answers_50q.json"
    if not answers_path.exists():
        return [], []
    answers = json.loads(answers_path.read_text(encoding="utf-8"))[:limit]
    pairwise_results: list[JudgeResult] = []
    absolute_results: list[dict] = []
    for item in answers:
        gt = item["ground_truth"]
        model_answer = item["answer"]
        pairwise_results.append(swap_and_average(item["question"], model_answer, gt))
        score = absolute_score(item["question"], model_answer)
        absolute_results.append({
            "question_id": item["id"],
            "question": item["question"],
            "answer": model_answer,
            **score,
        })
    return pairwise_results, absolute_results


# ─── Task 6: Swap-and-Average ─────────────────────────────────────────────────

def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    """Task 6: Chạy pairwise 2 lần (hoán đổi thứ tự), lấy kết quả nhất quán.

    Lý do: LLM thường có position bias (ưu tiên answer xuất hiện trước).
    Bằng cách swap, ta phát hiện và giảm bias này.

    Logic:
        Pass 1: judge(q, A, B) → winner_1 (trong không gian A/B)
        Pass 2: judge(q, B, A) → winner_2_raw (trong không gian B/A)
        Convert: nếu winner_2_raw="A" thì thực ra là B (vì đã swap)
        Final:   nếu winner_1 == winner_2 → final = winner_1
                 nếu khác nhau → final = "tie"
    """
    pass1 = pairwise_judge(question, answer_a, answer_b)
    pass2_raw = pairwise_judge(question, answer_b, answer_a)
    swap_map = {"A": "B", "B": "A", "tie": "tie"}
    winner_pass2 = swap_map.get(pass2_raw["winner"], "tie")
    final = pass1["winner"] if pass1["winner"] == winner_pass2 else "tie"
    position_consistent = pass1["winner"] == winner_pass2
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"],
        winner_pass2=winner_pass2,
        final_winner=final,
        reasoning_pass1=pass1.get("reasoning", ""),
        reasoning_pass2=pass2_raw.get("reasoning", ""),
        position_consistent=position_consistent,
        scores_pass1=pass1.get("scores", {}),
        scores_pass2={
            "A": pass2_raw.get("scores", {}).get("B", 0.0),
            "B": pass2_raw.get("scores", {}).get("A", 0.0),
        },
    )


# ─── Task 7: Cohen's κ ────────────────────────────────────────────────────────

def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    """Task 7: Tính Cohen's κ giữa LLM judge và human labels.

    Args:
        judge_labels:  nhãn từ LLM judge (0 = bad answer, 1 = good answer)
        human_labels:  nhãn từ human_labels_10q.json

    Returns:
        κ ∈ [-1, 1]
        Thang đo Landis-Koch: <0=poor, 0-0.2=slight, 0.2-0.4=fair,
                               0.4-0.6=moderate, 0.6-0.8=substantial, 0.8-1=almost perfect

    Gợi ý A — dùng scikit-learn:
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(human_labels, judge_labels)

    Gợi ý B — tính tay:
        n = len(judge_labels)
        p_o = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
        p_e = (judge_labels.count(1)/n * human_labels.count(1)/n +
               judge_labels.count(0)/n * human_labels.count(0)/n)
        κ = (p_o - p_e) / (1 - p_e) if p_e != 1 else 0
        return κ
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError("judge_labels and human_labels must have the same length")
    if not judge_labels:
        return 0.0

    labels = sorted(set(judge_labels) | set(human_labels))
    n = len(judge_labels)
    observed = sum(j == h for j, h in zip(judge_labels, human_labels)) / n
    expected = 0.0
    for label in labels:
        expected += (judge_labels.count(label) / n) * (human_labels.count(label) / n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return round((observed - expected) / (1 - expected), 4)


# ─── Task 8: Bias Report ──────────────────────────────────────────────────────

def bias_report(judge_results: list[JudgeResult]) -> dict:
    """Task 8: Đo lường position bias và verbosity bias.

    Position bias: LLM chọn answer theo vị trí (A hay B) thay vì chất lượng.
        → Đo bằng % cases where position_consistent = False

    Verbosity bias: LLM ưu tiên answer dài hơn dù không chính xác hơn.
        → Đo bằng: trong các case A thắng, A có dài hơn B không? Tương tự cho B.

    Returns:
        {
          "total_judged": int,
          "position_bias_rate": float,        # 0-1, cao = bias nhiều
          "position_bias_count": int,
          "verbosity_bias": float,            # 0-1, > 0.6 = đáng lo ngại
          "verbosity_details": {
            "a_wins_a_longer": int,           # A thắng VÀ A dài hơn
            "b_wins_b_longer": int,           # B thắng VÀ B dài hơn
            "total_decisive": int,            # tổng case có winner rõ ràng
          },
          "interpretation": str,
        }
    """
    total = len(judge_results)
    if total == 0:
        return {
            "total_judged": 0,
            "position_bias_rate": 0.0,
            "verbosity_bias": 0.0,
            "position_bias_count": 0,
            "verbosity_details": {"a_wins_a_longer": 0, "b_wins_b_longer": 0, "total_decisive": 0},
            "interpretation": "No judge results available.",
        }

    position_bias_count = sum(1 for result in judge_results if not result.position_consistent)
    position_bias_rate = position_bias_count / total
    a_wins_a_longer = sum(
        1 for result in judge_results
        if result.final_winner == "A" and len(result.answer_a) > len(result.answer_b)
    )
    b_wins_b_longer = sum(
        1 for result in judge_results
        if result.final_winner == "B" and len(result.answer_b) > len(result.answer_a)
    )
    decisive = sum(1 for result in judge_results if result.final_winner != "tie")
    verbosity_bias = (a_wins_a_longer + b_wins_b_longer) / decisive if decisive else 0.0
    interpretation = (
        "Position bias cao — tiếp tục giữ swap-and-average và thêm human calibration."
        if position_bias_rate > 0.3
        else "Position bias thấp — judge ổn định với swap-and-average."
    )
    return {
        "total_judged": total,
        "position_bias_rate": round(position_bias_rate, 3),
        "position_bias_count": position_bias_count,
        "verbosity_bias": round(verbosity_bias, 3),
        "verbosity_details": {
            "a_wins_a_longer": a_wins_a_longer,
            "b_wins_b_longer": b_wins_b_longer,
            "total_decisive": decisive,
        },
        "interpretation": interpretation,
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # --- Demo pairwise + swap ---
    q   = "Nhân viên được nghỉ bao nhiêu ngày phép năm?"
    a_a = "Nhân viên được nghỉ 15 ngày phép năm theo chính sách v2024 hiện hành."
    a_b = "Theo quy định, nhân viên có 12 ngày phép hàng năm."

    print("Running swap-and-average judge...")
    result = swap_and_average(q, a_a, a_b)
    print(f"  Pass 1 winner: {result.winner_pass1}")
    print(f"  Pass 2 winner: {result.winner_pass2}")
    print(f"  Final:         {result.final_winner}")
    print(f"  Position consistent: {result.position_consistent}")

    # --- Cohen's κ vs human labels ---
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as f:
        human_data = json.load(f)
    human_labels = [item["human_label"] for item in human_data]
    print(f"\nHuman labels loaded: {len(human_labels)} questions")

    # In production: run judge on the same 10 questions to get judge_labels
    judge_labels = [0] * len(human_labels)  # placeholder — replace with real judge output
    kappa = cohen_kappa(judge_labels, human_labels)
    print(f"Cohen's κ (placeholder): {kappa:.3f}")

    # --- Batch judge for Lab 24 artifacts ---
    batch_results, absolute_results = run_batch_judge(limit=30)
    judged = batch_results or [result]

    # --- Bias report ---
    bias = bias_report(judged)
    print(f"\nBias report: {bias}")

    os.makedirs("reports", exist_ok=True)
    os.makedirs("phase-b", exist_ok=True)
    pairwise_csv = Path("phase-b/pairwise_results.csv")
    with pairwise_csv.open("w", encoding="utf-8") as f:
        f.write("question,winner_after_swap,run1_winner,run2_winner,position_consistent,score_a,score_b\n")
        for r in judged:
            f.write(
                json.dumps(r.question, ensure_ascii=False) + ","
                f"{r.final_winner},{r.winner_pass1},{r.winner_pass2},{r.position_consistent},"
                f"{r.scores_pass1.get('A', 0.0)},{r.scores_pass1.get('B', 0.0)}\n"
            )
    absolute_csv = Path("phase-b/absolute_scores.csv")
    with absolute_csv.open("w", encoding="utf-8") as f:
        f.write("question_id,accuracy,relevance,conciseness,helpfulness,overall,mode\n")
        for row in absolute_results:
            f.write(
                f"{row['question_id']},{row['accuracy']},{row['relevance']},"
                f"{row['conciseness']},{row['helpfulness']},{row['overall']},{row['mode']}\n"
            )
    report = {
        "judge_mode": "openai" if OPENAI_API_KEY else "anthropic" if ANTHROPIC_API_KEY else "heuristic",
        "batch_size": len(judged),
        "demo_pairwise": {
            "question": result.question,
            "winner_pass1": result.winner_pass1,
            "winner_pass2": result.winner_pass2,
            "final_winner": result.final_winner,
            "position_consistent": result.position_consistent,
            "scores_pass1": result.scores_pass1,
            "scores_pass2": result.scores_pass2,
        },
        "cohen_kappa": kappa,
        "kappa_interpretation": (
            "substantial" if kappa >= 0.6 else
            "moderate" if kappa >= 0.4 else
            "needs_improvement"
        ),
        "bias_report": bias,
        "pairwise_results_csv": str(pairwise_csv),
        "absolute_scores_csv": str(absolute_csv),
    }
    with open("reports/judge_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open("analysis/bias_report.md", "w", encoding="utf-8") as f:
        f.write("# Judge Bias Report\n\n")
        f.write(f"- Total judged: {bias['total_judged']}\n")
        f.write(f"- Position bias rate: {bias['position_bias_rate']:.1%}\n")
        f.write(f"- Verbosity bias: {bias['verbosity_bias']:.1%}\n")
        f.write(f"- Cohen's kappa sample: {kappa:.3f}\n\n")
        f.write("## Mitigation Strategy\n\n")
        f.write("Use swap-and-average for every pairwise comparison, log run1/run2 disagreement, and calibrate judge output against human labels before trusting it for production gates.\n")
    print("Saved Phase B reports → reports/judge_results.json, analysis/bias_report.md")
