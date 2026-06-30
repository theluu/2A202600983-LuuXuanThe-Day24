"""Phase B.1 + B.2 — LLM-as-Judge: pairwise (swap-and-average) + absolute scoring.

Compares two RAG variants on the same questions:
    A-system = baseline (dense top-3)      B-system = reranker (dense top-8 -> LLM rerank)

B.1 pairwise: judge with gpt-4o-mini, run twice with swapped order, map back, decide
    winner_after_swap. >= 30 questions.  -> phase-b/pairwise_results.csv
B.2 absolute: score the baseline answer on 4 dimensions (1-5).  -> phase-b/absolute_scores.csv

Also caches phase-b/_judge_cache.json for the calibration step (B.3/B.4).
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import GEN_MODEL, METER, RagPipeline, load_env, openai_client  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASELINE = ROOT / "phase-a" / "answers_baseline.json"
RERANK_CACHE = HERE / "_rerank_answers.json"
PAIRWISE_CSV = HERE / "pairwise_results.csv"
ABSOLUTE_CSV = HERE / "absolute_scores.csv"
JUDGE_CACHE = HERE / "_judge_cache.json"

N = 30


def get_rerank_answers(questions: list[str]) -> dict[str, str]:
    if RERANK_CACHE.exists():
        return json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
    pipe = RagPipeline("reranker")
    out = {}
    for i, q in enumerate(questions):
        ans, _ = pipe.ask(q)
        out[q] = ans
        print(f"  rerank answer {i+1}/{len(questions)}")
    RERANK_CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def pairwise(question: str, ans_a: str, ans_b: str) -> dict:
    """One judge call. Returns {winner: A|B|tie, reason}."""
    client = openai_client()
    resp = client.chat.completions.create(
        model=GEN_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": "Bạn là giám khảo đánh giá câu trả lời RAG về chính sách HR. "
             "So sánh Answer A và Answer B theo độ chính xác, đầy đủ, bám câu hỏi. "
             "Trả về JSON {\"winner\":\"A\"|\"B\"|\"tie\", \"reason\":\"...\"}. Chỉ JSON."},
            {"role": "user", "content": f"Câu hỏi: {question}\n\nAnswer A:\n{ans_a}\n\nAnswer B:\n{ans_b}"},
        ],
    )
    METER.add_chat(resp.usage)
    try:
        obj = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.S).group(0))
        w = obj.get("winner", "tie").strip().upper()
        w = w if w in ("A", "B") else "tie"
        return {"winner": w, "reason": obj.get("reason", "").strip()}
    except Exception:
        return {"winner": "tie", "reason": "parse_error"}


def absolute(question: str, answer: str) -> dict:
    client = openai_client()
    resp = client.chat.completions.create(
        model=GEN_MODEL, temperature=0,
        messages=[
            {"role": "system", "content": "Chấm câu trả lời theo 4 tiêu chí, mỗi tiêu chí thang 1-5: "
             "accuracy, relevance, conciseness, helpfulness. Trả về JSON với 4 khóa số nguyên. Chỉ JSON."},
            {"role": "user", "content": f"Câu hỏi: {question}\n\nCâu trả lời:\n{answer}"},
        ],
    )
    METER.add_chat(resp.usage)
    try:
        obj = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.S).group(0))
        dims = {k: int(obj.get(k, 3)) for k in ["accuracy", "relevance", "conciseness", "helpfulness"]}
        dims = {k: min(5, max(1, v)) for k, v in dims.items()}
        return dims
    except Exception:
        return {"accuracy": 3, "relevance": 3, "conciseness": 3, "helpfulness": 3}


def main() -> None:
    load_env()
    base = json.loads(BASELINE.read_text(encoding="utf-8"))[:N]
    questions = [d["question"] for d in base]
    rerank = get_rerank_answers(questions)

    pairwise_rows = []
    abs_rows = []
    cache = []

    for i, d in enumerate(base):
        q = d["question"]
        a_base, a_rer = d["answer"], rerank[q]
        # run1: A=baseline, B=reranker
        r1 = pairwise(q, a_base, a_rer)
        # run2: swapped -> A=reranker, B=baseline ; map winner back to original A/B space
        r2_raw = pairwise(q, a_rer, a_base)
        r2_mapped = {"A": "B", "B": "A", "tie": "tie"}[r2_raw["winner"]]  # back to (A=baseline,B=reranker)

        # winner_after_swap: agree -> that winner ; disagree -> tie (position-inconsistent)
        if r1["winner"] == r2_mapped:
            final = r1["winner"]
        else:
            final = "tie"
        winner_sys = {"A": "baseline", "B": "reranker", "tie": "tie"}[final]

        pairwise_rows.append({
            "question": q,
            "winner_after_swap": winner_sys,
            "run1_winner": {"A": "baseline", "B": "reranker", "tie": "tie"}[r1["winner"]],
            "run2_winner": {"A": "baseline", "B": "reranker", "tie": "tie"}[r2_mapped],
            "position_consistent": r1["winner"] == r2_mapped,
            "len_baseline": len(a_base),
            "len_reranker": len(a_rer),
            "reason": r1["reason"][:200],
        })

        dims = absolute(q, a_base)
        overall = round(sum(dims.values()) / 4, 2)
        abs_rows.append({"question_id": i + 1, **dims, "overall": overall})

        cache.append({
            "question": q, "ground_truth": d["ground_truth"],
            "answer_baseline": a_base, "answer_reranker": a_rer,
            "run1_winner": pairwise_rows[-1]["run1_winner"],
            "run2_winner": pairwise_rows[-1]["run2_winner"],
            "winner_after_swap": winner_sys,
            "position_consistent": pairwise_rows[-1]["position_consistent"],
        })
        print(f"  judged {i+1}/{N}: swap_winner={winner_sys} consistent={pairwise_rows[-1]['position_consistent']}")

    with PAIRWISE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairwise_rows[0].keys()))
        w.writeheader(); w.writerows(pairwise_rows)
    with ABSOLUTE_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(abs_rows[0].keys()))
        w.writeheader(); w.writerows(abs_rows)
    JUDGE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    print("\npairwise winners:", Counter(r["winner_after_swap"] for r in pairwise_rows))
    print("position-consistent:", sum(r["position_consistent"] for r in pairwise_rows), "/", N)
    print("cost:", METER.summary())


if __name__ == "__main__":
    main()
