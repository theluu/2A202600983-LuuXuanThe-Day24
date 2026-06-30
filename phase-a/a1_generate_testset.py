"""Phase A.1 — Synthetic test set generation (RAGAS-style evolution).

Generates 50 grounded questions from the HR corpus:
  - 25 simple        (single-hop, one chunk)
  - 13 reasoning     (multi-step / calculation, one chunk)
  - 12 multi_context (needs >=2 chunks)

Output: phase-a/testset_v1.csv  (question, ground_truth, contexts, evolution_type)
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import GEN_MODEL, METER, load_chunks, load_env, openai_client  # noqa: E402

OUT = Path(__file__).resolve().parent / "testset_v1.csv"
RNG = random.Random(24)

PLAN = [("simple", 25), ("reasoning", 13), ("multi_context", 12)]

INSTRUCTION = {
    "simple": "Tạo MỘT câu hỏi đơn giản, single-hop, trả lời trực tiếp từ đoạn văn. "
              "Câu hỏi rõ ràng, ground_truth là câu trả lời ngắn gọn chính xác.",
    "reasoning": "Tạo MỘT câu hỏi cần SUY LUẬN nhiều bước hoặc TÍNH TOÁN từ đoạn văn "
                 "(ví dụ cộng dồn, so sánh điều kiện, suy ra hệ quả). ground_truth phải thể hiện bước suy luận.",
    "multi_context": "Tạo MỘT câu hỏi cần thông tin từ CẢ HAI đoạn văn để trả lời đầy đủ "
                     "(kết hợp/đối chiếu). ground_truth tổng hợp từ cả hai đoạn.",
}


def gen_one(evolution_type: str, contexts: list[str]) -> dict | None:
    client = openai_client()
    ctx = "\n\n---\n\n".join(contexts)
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        temperature=0.4,
        messages=[
            {"role": "system", "content": "Bạn tạo bộ test đánh giá RAG cho tài liệu HR tiếng Việt. "
             "Chỉ trả về JSON {\"question\":..., \"ground_truth\":...}. Không thêm chữ nào khác."},
            {"role": "user", "content": f"{INSTRUCTION[evolution_type]}\n\nĐoạn văn:\n{ctx}"},
        ],
    )
    METER.add_chat(resp.usage)
    try:
        m = re.search(r"\{.*\}", resp.choices[0].message.content, re.S)
        obj = json.loads(m.group(0))
        if obj.get("question") and obj.get("ground_truth"):
            return {
                "question": obj["question"].strip(),
                "ground_truth": obj["ground_truth"].strip(),
                "contexts": " ||| ".join(contexts),
                "evolution_type": evolution_type,
            }
    except Exception:
        return None
    return None


def main() -> None:
    load_env()
    chunks = [c for c in load_chunks() if len(c.text) > 120]
    RNG.shuffle(chunks)
    rows: list[dict] = []

    for evo, n in PLAN:
        made = 0
        attempts = 0
        while made < n and attempts < n * 4:
            attempts += 1
            if evo == "multi_context":
                a, b = RNG.sample(chunks, 2)
                ctxs = [a.text, b.text]
            else:
                ctxs = [RNG.choice(chunks).text]
            row = gen_one(evo, ctxs)
            if row and not any(r["question"] == row["question"] for r in rows):
                rows.append(row)
                made += 1
                print(f"  [{evo}] {made}/{n}: {row['question'][:60]}")

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "ground_truth", "contexts", "evolution_type"])
        w.writeheader()
        w.writerows(rows)

    dist = {evo: sum(1 for r in rows if r["evolution_type"] == evo) for evo, _ in PLAN}
    print(f"\nWrote {len(rows)} rows -> {OUT}")
    print("Distribution:", dist)
    print("Cost so far:", METER.summary())


if __name__ == "__main__":
    main()
