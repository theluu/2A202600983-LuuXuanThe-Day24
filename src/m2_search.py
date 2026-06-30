from __future__ import annotations

from dataclasses import dataclass
import re


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE))


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict


class HybridSearch:
    def __init__(self) -> None:
        self.docs: list[dict] = []

    def index(self, chunks: list[dict]) -> None:
        self.docs = chunks

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        q = _tokens(query)
        scored = []
        for chunk in self.docs:
            text = chunk.get("text", "")
            score = len(q & _tokens(text)) / max(1, len(q))
            scored.append(SearchResult(text=text, score=score, metadata=chunk.get("metadata", {})))
        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
