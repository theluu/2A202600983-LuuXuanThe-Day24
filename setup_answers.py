"""
Setup script: chạy Day 18 pipeline trên 50 câu hỏi → lưu answers_50q.json

Chạy TRƯỚC khi bắt đầu Phase A:
    python setup_answers.py

Yêu cầu:
    1. Đã copy src/ từ Day 18 (m1-m5, pipeline.py) vào thư mục này
    2. docker compose up -d  (Qdrant đang chạy trên port 6333)
    3. .env có OPENAI_API_KEY
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_day18_files() -> bool:
    required = [
        "src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
        "src/m4_eval.py",     "src/m5_enrichment.py", "src/pipeline.py",
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n⚠️  Thiếu files từ Day 18, dùng fallback retriever offline cho Lab 24:")
        for f in missing:
            print(f"   missing: {f}")
        return False
    print(f"✓ Day 18 source files: {len(required)}/{len(required)} found")
    return True


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


def load_fallback_docs() -> list[dict]:
    docs = []
    for root, _, files in os.walk("data"):
        for name in sorted(files):
            if not name.endswith(".md"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as f:
                docs.append({"path": path, "text": f.read()})
    if not docs:
        raise FileNotFoundError("No markdown docs found in data/")
    return docs


def fallback_query(question: str, docs: list[dict]) -> tuple[str, list[str]]:
    q_tokens = tokenize(question)
    scored = []
    for doc in docs:
        score = len(q_tokens & tokenize(doc["text"]))
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    contexts = [item[1]["text"][:1800] for item in scored[:3]]
    best_text = contexts[0]
    sentences = re.split(r"(?<=[.!?。])\s+|\n+", best_text)
    best_sentence = max(sentences, key=lambda s: len(q_tokens & tokenize(s)), default=best_text[:500])
    answer = best_sentence.strip() or best_text[:500]
    return answer, contexts


def build_pipeline():
    from src.m1_chunking import load_documents, chunk_hierarchical
    from src.m2_search import HybridSearch
    from src.m3_rerank import CrossEncoderReranker
    from src.m5_enrichment import enrich_chunks
    from config import RERANK_TOP_K

    print("\n[1/3] Chunking + enriching documents...")
    t0 = time.time()
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({
                "text": child.text,
                "metadata": {**child.metadata, "parent_id": child.parent_id},
            })

    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [{"text": e.enriched_text, "metadata": e.auto_metadata} for e in enriched]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)")
    else:
        print(f"  ✓ Using {len(all_chunks)} raw chunks (M5 not implemented or no API key)")

    print("\n[2/3] Indexing (BM25 + Dense)...")
    t0 = time.time()
    search = HybridSearch()
    search.index(all_chunks)
    print(f"  ✓ Indexed {len(all_chunks)} chunks ({time.time()-t0:.1f}s)")

    print("\n[3/3] Loading reranker...")
    t0 = time.time()
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)")

    return search, reranker, RERANK_TOP_K


def run_query(q: str, search, reranker, top_k: int) -> tuple[str, list[str]]:
    from config import OPENAI_API_KEY

    results = search.search(q)
    docs    = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(q, docs, top_k=top_k)
    contexts = [r.text for r in reranked] if reranked else [r.text for r in results[:3]]

    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            ctx = "\n\n".join(contexts)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                    {"role": "user",   "content": f"Context:\n{ctx}\n\nCâu hỏi: {q}"},
                ],
            )
            return resp.choices[0].message.content, contexts
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}")

    return (contexts[0] if contexts else "Không tìm thấy thông tin."), contexts


def main():
    print("=" * 60)
    print("LAB 24 SETUP — Generating answers for 50 questions")
    print("=" * 60)

    has_day18 = check_day18_files()

    with open("test_set_50q.json", encoding="utf-8") as f:
        test_set = json.load(f)
    print(f"✓ Loaded {len(test_set)} questions (factual/multi_hop/adversarial)")

    search = reranker = top_k = None
    fallback_docs = None
    if has_day18:
        try:
            search, reranker, top_k = build_pipeline()
        except Exception as e:
            print(f"\n⚠️  Day 18 pipeline unavailable: {e}")
            print("→ Switching to offline fallback retriever.")
            has_day18 = False
    if not has_day18:
        fallback_docs = load_fallback_docs()
        print(f"✓ Loaded {len(fallback_docs)} Markdown docs for fallback retrieval")

    print(f"\nRunning {len(test_set)} queries...")
    answers = []
    t_start = time.time()

    for i, item in enumerate(test_set):
        if has_day18:
            answer, contexts = run_query(item["question"], search, reranker, top_k)
        else:
            answer, contexts = fallback_query(item["question"], fallback_docs)
        answers.append({
            "id":           item["id"],
            "distribution": item["distribution"],
            "question":     item["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": item["ground_truth"],
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(test_set)}] done ({time.time()-t_start:.0f}s elapsed)")

    with open("answers_50q.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(answers)} answers → answers_50q.json")
    print(f"  Total time: {time.time()-t_start:.1f}s")
    print("\n→ Bây giờ bắt đầu Phase A:")
    print("     python src/phase_a_ragas.py")


if __name__ == "__main__":
    main()
