"""Search over the chunk index: dense (embeddings), BM25 (keywords) or hybrid (both, rank fusion)."""

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .store import KnowledgeBase

MODES = ("hybrid", "dense", "bm25")
RRF_K = 60   # default reciprocal rank fusion constant: damps the weight of top ranks
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "did", "do", "does", "for", "from",
    "has", "have", "how", "in", "is", "it", "its", "of", "on", "or", "our", "that", "the", "their",
    "these", "this", "those", "to", "was", "we", "were", "what", "when", "where", "which", "who",
    "why", "with",
}


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; keeps hyphenated and alphanumeric terms (e.g. 'gpt-4', 'cas9')."""
    return [t for t in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk_id: str
    paper_id: str
    kind: str
    title: str
    url: str
    published: str
    text: str
    score: float


class Retriever:
    def __init__(self, kb: KnowledgeBase, rrf_k: int = RRF_K):
        self.kb = kb
        self.rrf_k = rrf_k
        self._bm25_key = None
        self._bm25 = None
        self._bm25_ids: list[str] = []

    def _candidates(self, topic=None, year_from=None) -> list[str] | None:
        if not (topic or year_from):
            return None
        return [p["paper_id"] for p in self.kb.papers(topic=topic, year_from=year_from)]

    def _bm25_search(self, query: str, n: int, paper_ids: list[str] | None) -> list[str]:
        stats = self.kb.stats()   # cache key changes whenever chunks or notes are added
        key = (tuple(paper_ids) if paper_ids is not None else None, stats["chunks"], stats["notes"])
        if key != self._bm25_key:
            chunks = self.kb.chunks(paper_ids)
            self._bm25_ids = [c["chunk_id"] for c in chunks]
            self._bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks]) if chunks else None
            self._bm25_key = key
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._bm25_ids[i] for i in ranked[:n] if scores[i] > 0]

    def search(self, query: str, k: int, mode: str = "hybrid", topic: str | None = None,
               year_from: int | None = None) -> list[Hit]:
        """Top-k chunks for the query, optionally limited to a saved topic or to recent years."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        paper_ids = self._candidates(topic, year_from)
        pool = max(4 * k, 20)

        rankings = []
        if mode in ("dense", "hybrid"):
            rankings.append(self.kb.dense_search(query, pool, paper_ids))
        if mode in ("bm25", "hybrid"):
            rankings.append(self._bm25_search(query, pool, paper_ids))

        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        best = sorted(scores, key=lambda c: -scores[c])[:k]
        chunks = self.kb.chunks_by_id(best)
        return [Hit(score=round(scores[c], 5), **{f: chunks[c][f] for f in
                    ("chunk_id", "paper_id", "kind", "title", "url", "published", "text")})
                for c in best if c in chunks]
