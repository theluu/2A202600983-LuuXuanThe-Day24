from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RerankResult:
    text: str
    score: float
    metadata: dict


class CrossEncoderReranker:
    def rerank(self, query: str, docs: list[dict], top_k: int = 3) -> list[RerankResult]:
        ranked = sorted(docs, key=lambda item: item.get("score", 0.0), reverse=True)
        return [RerankResult(text=d["text"], score=d.get("score", 0.0), metadata=d.get("metadata", {})) for d in ranked[:top_k]]
