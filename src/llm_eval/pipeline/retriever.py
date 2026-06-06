"""Simple local document retriever for the reference RAG pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class LocalRetriever:
    def __init__(self, knowledge_base_path: Path, top_k: int = 5):
        self.top_k = top_k
        self.documents: list[dict[str, str]] = []
        kb_path = Path(knowledge_base_path)
        if kb_path.exists():
            for file_path in kb_path.rglob("*.md"):
                rel_path = str(file_path.relative_to(kb_path)).replace("\\", "/")
                content = file_path.read_text(encoding="utf-8")
                self.documents.append({"path": rel_path, "content": content})
        else:
            logger.warning("Knowledge base path does not exist: %s", kb_path)
        if not self.documents:
            logger.warning("No documents loaded from knowledge base: %s", kb_path)

    def retrieve(self, query: str) -> list[dict[str, str]]:
        if not self.documents:
            return []

        tokens = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, dict[str, str]]] = []

        for doc in self.documents:
            doc_tokens = set(re.findall(r"\w+", doc["content"].lower()))
            path_tokens = set(re.findall(r"\w+", doc["path"].lower()))
            overlap = len(tokens & (doc_tokens | path_tokens))
            if overlap > 0:
                scored.append((overlap / max(len(tokens), 1), doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: self.top_k]]
