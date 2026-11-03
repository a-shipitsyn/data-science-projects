# Semantic search over the indexed documentation
from typing import List, Optional

from llama_index.core import Settings as LlamaSettings

from rag_bot.settings import Settings
from rag_bot.store import DocumentStore, Hit


def format_hits(hits: List[Hit]) -> str:
    """Chunks as prompt-ready text, each tagged with its source file."""
    return "\n\n".join(f'<chunk source="{h.file_name}" relevance="{h.score:.2f}">\n{h.text}\n</chunk>'
                       for h in hits)


def source_files(hits: List[Hit]) -> List[str]:
    """Distinct source files, best match first."""
    return list(dict.fromkeys(h.file_name for h in hits if h.file_name))


class KnowledgeBase:
    def __init__(self, settings: Settings, store: DocumentStore):
        self.s = settings
        self.store = store

    def search(self, question: str, top_k: Optional[int] = None,
               threshold: Optional[float] = None) -> List[Hit]:
        """Top-k chunks with similarity >= threshold (defaults from settings)."""
        top_k = top_k or self.s.top_k
        threshold = self.s.similarity_threshold if threshold is None else threshold
        query = LlamaSettings.embed_model.get_query_embedding(question)
        return [h for h in self.store.search(query, top_k) if h.score >= threshold]
