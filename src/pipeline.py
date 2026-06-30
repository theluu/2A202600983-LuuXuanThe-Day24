from __future__ import annotations

from src.m1_chunking import load_documents
from src.m2_search import HybridSearch


class BaselinePipeline:
    def __init__(self) -> None:
        self.search = HybridSearch()
        self.search.index([{"text": doc["text"], "metadata": doc["metadata"]} for doc in load_documents()])

    def ask(self, question: str) -> tuple[str, list[str]]:
        results = self.search.search(question, top_k=3)
        contexts = [result.text for result in results]
        answer = contexts[0][:600] if contexts else "Không tìm thấy thông tin."
        return answer, contexts
