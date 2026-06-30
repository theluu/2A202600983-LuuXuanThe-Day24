"""Shared infrastructure for Lab 24 — real APIs (OpenAI + Groq).

Provides:
- .env loading
- OpenAI / Groq clients
- Corpus loading + section-level chunking
- Real embedding retrieval (text-embedding-3-small, cached)
- gpt-4o-mini generation
- Two RAG variants: `baseline` (dense top-k) and `reranker` (dense top-n -> LLM rerank)
- A lightweight token + cost meter
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
GEN_MODEL = "gpt-4o-mini"

# OpenAI pricing (USD per 1M tokens) — gpt-4o-mini + text-embedding-3-small
PRICE = {
    "gpt-4o-mini": {"in": 0.15 / 1e6, "out": 0.60 / 1e6},
    "text-embedding-3-small": {"in": 0.02 / 1e6, "out": 0.0},
}


# ─── env ──────────────────────────────────────────────────────────────────────
def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


# ─── cost meter ───────────────────────────────────────────────────────────────
@dataclass
class CostMeter:
    in_tokens: int = 0
    out_tokens: int = 0
    embed_tokens: int = 0
    calls: int = 0

    def add_chat(self, usage) -> None:
        self.calls += 1
        self.in_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.out_tokens += getattr(usage, "completion_tokens", 0) or 0

    def add_embed(self, usage) -> None:
        self.calls += 1
        self.embed_tokens += getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0

    @property
    def usd(self) -> float:
        c = PRICE[GEN_MODEL]
        e = PRICE[EMBED_MODEL]
        return (self.in_tokens * c["in"] + self.out_tokens * c["out"] + self.embed_tokens * e["in"])

    def summary(self) -> dict:
        return {
            "chat_in_tokens": self.in_tokens,
            "chat_out_tokens": self.out_tokens,
            "embed_tokens": self.embed_tokens,
            "api_calls": self.calls,
            "est_cost_usd": round(self.usd, 5),
        }


METER = CostMeter()


# ─── clients ──────────────────────────────────────────────────────────────────
def openai_client():
    from openai import OpenAI
    load_env()
    return OpenAI()


def groq_client():
    from openai import OpenAI
    load_env()
    return OpenAI(api_key=os.environ["GROQ_API_KEY"], base_url="https://api.groq.com/openai/v1")


# ─── corpus + chunking ────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    source: str
    section: str


def load_chunks() -> list[Chunk]:
    """Section-level chunks (split markdown on ## headers)."""
    chunks: list[Chunk] = []
    for path in sorted(DATA_DIR.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        title = ""
        parts = re.split(r"\n(?=## )", raw)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            m = re.match(r"#+\s*(.+)", part)
            section = m.group(1).strip() if m else (title or path.stem)
            if part.startswith("# ") and not part.startswith("## "):
                title = section
            chunks.append(Chunk(text=part, source=path.name, section=section[:80]))
    return chunks


# ─── embeddings (cached) ──────────────────────────────────────────────────────
def _cache_key(texts: list[str]) -> str:
    h = hashlib.sha256(("||".join(texts)).encode("utf-8")).hexdigest()[:16]
    return f"emb_{EMBED_MODEL}_{h}.json"


def embed(texts: list[str], use_cache: bool = True) -> list[list[float]]:
    cache = CACHE_DIR / _cache_key(texts)
    if use_cache and cache.exists():
        return json.loads(cache.read_text())
    client = openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), 256):
        batch = texts[i:i + 256]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        METER.add_embed(resp.usage)
        out.extend([d.embedding for d in resp.data])
    if use_cache:
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(out))
        tmp.replace(cache)  # atomic — safe under concurrency
    return out


def _cos(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb + 1e-9)


# ─── RAG pipeline ─────────────────────────────────────────────────────────────
GEN_SYSTEM = (
    "Bạn là trợ lý chính sách nhân sự nội bộ. CHỈ trả lời dựa trên ngữ cảnh được cung cấp. "
    "Nếu ngữ cảnh không chứa thông tin, trả lời 'Không tìm thấy thông tin trong tài liệu.' "
    "Trả lời ngắn gọn, chính xác, bằng tiếng Việt."
)


@dataclass
class RagPipeline:
    variant: str = "baseline"  # "baseline" | "reranker"
    top_k: int = 3
    candidates: int = 8
    _chunks: list[Chunk] = field(default_factory=list)
    _embs: list[list[float]] = field(default_factory=list)

    def __post_init__(self):
        self._chunks = load_chunks()
        self._embs = embed([c.text for c in self._chunks])

    def retrieve(self, question: str) -> list[Chunk]:
        qemb = embed([question])[0]
        scored = sorted(
            range(len(self._chunks)),
            key=lambda i: _cos(qemb, self._embs[i]),
            reverse=True,
        )
        if self.variant == "baseline":
            idx = scored[: self.top_k]
            return [self._chunks[i] for i in idx]
        # reranker: take more candidates, LLM-rerank to top_k
        cand = [self._chunks[i] for i in scored[: self.candidates]]
        return self._llm_rerank(question, cand)[: self.top_k]

    def _llm_rerank(self, question: str, cand: list[Chunk]) -> list[Chunk]:
        client = openai_client()
        listing = "\n".join(f"[{i}] ({c.source}) {c.text[:300]}" for i, c in enumerate(cand))
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": "Rerank passages by relevance to the question. "
                 "Return ONLY a JSON list of passage indices, most relevant first."},
                {"role": "user", "content": f"Question: {question}\n\nPassages:\n{listing}\n\n"
                 "Return JSON list of indices e.g. [2,0,5]."},
            ],
        )
        METER.add_chat(resp.usage)
        try:
            order = json.loads(re.search(r"\[.*?\]", resp.choices[0].message.content, re.S).group(0))
            ranked = [cand[i] for i in order if isinstance(i, int) and 0 <= i < len(cand)]
            seen = {id(c) for c in ranked}
            ranked += [c for c in cand if id(c) not in seen]
            return ranked
        except Exception:
            return cand

    def generate(self, question: str, contexts: list[str]) -> str:
        client = openai_client()
        ctx = "\n\n---\n\n".join(contexts)
        resp = client.chat.completions.create(
            model=GEN_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": f"Ngữ cảnh:\n{ctx}\n\nCâu hỏi: {question}"},
            ],
        )
        METER.add_chat(resp.usage)
        return resp.choices[0].message.content.strip()

    def ask(self, question: str) -> tuple[str, list[str]]:
        chunks = self.retrieve(question)
        contexts = [c.text for c in chunks]
        answer = self.generate(question, contexts)
        return answer, contexts


if __name__ == "__main__":
    load_env()
    p = RagPipeline("baseline")
    a, c = p.ask("Nhân viên được nghỉ bao nhiêu ngày phép năm?")
    print("ANSWER:", a)
    print("CONTEXTS:", len(c), "| cost:", METER.summary())
