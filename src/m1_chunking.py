from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Chunk:
    text: str
    metadata: dict
    parent_id: str = ""


def load_documents(data_dir: str | None = None) -> list[dict]:
    root = Path(data_dir or Path(__file__).resolve().parents[1] / "data")
    docs = []
    for path in sorted(root.glob("*.md")):
        docs.append({"text": path.read_text(encoding="utf-8"), "metadata": {"source": path.name}})
    return docs


def chunk_hierarchical(text: str, metadata: dict | None = None, size: int = 900) -> tuple[list[Chunk], list[Chunk]]:
    metadata = metadata or {}
    parent = Chunk(text=text, metadata=metadata, parent_id="parent-0")
    children = []
    for idx in range(0, len(text), size):
        children.append(Chunk(text=text[idx:idx + size], metadata={**metadata, "chunk_index": idx // size}, parent_id="parent-0"))
    return [parent], children or [parent]
