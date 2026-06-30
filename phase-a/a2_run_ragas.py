"""Phase A.2 — Real RAGAS evaluation (4 metrics) on the 50q test set.

For each question: run the baseline RAG pipeline (real retrieval + gpt-4o-mini),
then score with the ragas library:
    faithfulness, answer_relevancy, context_precision, context_recall

Outputs:
    phase-a/ragas_results.csv     (per-question scores)
    phase-a/ragas_summary.json    (aggregates + per-evolution + cost)
    phase-a/answers_baseline.json  (cached answers, reused by Phase B)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import EMBED_MODEL, GEN_MODEL, METER, RagPipeline, load_env  # noqa: E402

HERE = Path(__file__).resolve().parent
TESTSET = HERE / "testset_v1.csv"
RESULTS_CSV = HERE / "ragas_results.csv"
SUMMARY_JSON = HERE / "ragas_summary.json"
ANSWERS = HERE / "answers_baseline.json"

METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def build_answers() -> list[dict]:
    if ANSWERS.exists():
        return json.loads(ANSWERS.read_text(encoding="utf-8"))
    pipe = RagPipeline("baseline")
    rows = list(csv.DictReader(TESTSET.open(encoding="utf-8")))
    out = []
    for i, r in enumerate(rows):
        ans, ctxs = pipe.ask(r["question"])
        out.append({
            "question": r["question"],
            "ground_truth": r["ground_truth"],
            "evolution_type": r["evolution_type"],
            "answer": ans,
            "contexts": ctxs,
        })
        print(f"  answered {i+1}/{len(rows)}")
    ANSWERS.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def main() -> None:
    load_env()
    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import (answer_relevancy, context_precision,
                               context_recall, faithfulness)

    data = build_answers()
    ds = Dataset.from_list([{
        "question": d["question"],
        "answer": d["answer"],
        "contexts": d["contexts"],
        "ground_truth": d["ground_truth"],
    } for d in data])

    llm = ChatOpenAI(model=GEN_MODEL, temperature=0)
    emb = OpenAIEmbeddings(model=EMBED_MODEL)

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm, embeddings=emb,
    )
    df = result.to_pandas()

    # per-question CSV
    with RESULTS_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["question", "evolution_type", *METRIC_COLS, "avg_score"])
        for i, d in enumerate(data):
            scores = [float(df.iloc[i][m]) if m in df.columns and df.iloc[i][m] == df.iloc[i][m] else 0.0
                      for m in METRIC_COLS]
            w.writerow([d["question"], d["evolution_type"], *[round(s, 4) for s in scores],
                        round(sum(scores) / len(scores), 4)])

    # aggregates
    import statistics
    rows = list(csv.DictReader(RESULTS_CSV.open(encoding="utf-8")))
    def agg(subset):
        return {m: round(statistics.mean(float(r[m]) for r in subset), 4) for m in METRIC_COLS} if subset else {}
    overall = agg(rows)
    overall["avg_score"] = round(statistics.mean(float(r["avg_score"]) for r in rows), 4)
    per_evo = {}
    for evo in ["simple", "reasoning", "multi_context"]:
        sub = [r for r in rows if r["evolution_type"] == evo]
        a = agg(sub)
        if a:
            a["count"] = len(sub)
            a["avg_score"] = round(statistics.mean(float(r["avg_score"]) for r in sub), 4)
        per_evo[evo] = a

    observations = [f"{m} = {overall[m]} < 0.5 → cần cải thiện" for m in METRIC_COLS if overall[m] < 0.5]

    summary = {
        "total_questions": len(rows),
        "models": {"generation": GEN_MODEL, "embeddings": EMBED_MODEL, "ragas_judge": GEN_MODEL},
        "overall": overall,
        "per_evolution": per_evo,
        "observations_below_0_5": observations,
        "cost": METER.summary(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
