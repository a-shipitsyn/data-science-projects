"""Shared test helpers: offline embeddings, a scripted LLM and fake article sources."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from chromadb import Documents, EmbeddingFunction, Embeddings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paper_kb.llm import LLMError
from paper_kb.paper import Paper
from paper_kb.settings import Settings
from paper_kb.store import KnowledgeBase

FIXTURES = Path(__file__).parent / "fixtures"


class HashEmbedding(EmbeddingFunction):
    """Bag-of-words hashed into 256 dimensions: deterministic, no model download."""

    def __init__(self):
        pass

    def __call__(self, input: Documents) -> Embeddings:
        vectors = []
        for text in input:
            v = np.zeros(256, dtype=np.float32)
            for word in text.lower().split():
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
            vectors.append(v / (np.linalg.norm(v) or 1))
        return vectors

    @staticmethod
    def name() -> str:
        return "test-hash"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()


class ScriptedLLM:
    """Returns queued replies in order; an Exception in the queue is raised instead."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def complete(self, system, prompt, schema=None):
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)

    def complete_json(self, system, prompt, schema):
        text = self.complete(system, prompt, schema)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError("invalid JSON") from e


def make_paper(source_id, title, text, version="v1", published="2025-01-15"):
    return Paper(source="arxiv", source_id=source_id, version=version, title=title,
                 abstract=text.split("\n\n")[0], url=f"https://example.org/{source_id}",
                 authors=["A. Author", "B. Author"], published=published, text=text, text_kind="full")


class FakeSource:
    def __init__(self, name, papers):
        self.name = name
        self.papers = papers
        self.fetched = []

    def search(self, topic, max_results):
        return [Paper(**vars(p)) for p in self.papers[:max_results]]

    def fetch_full_text(self, paper):
        self.fetched.append(paper.paper_id)
        return paper


PAPERS = [
    make_paper("2501.00001", "Hybrid retrieval for technical search",
               "We combine BM25 with dense embeddings for retrieval.\n\n"
               "Reciprocal rank fusion of keyword and embedding rankings raises recall on queries "
               "that contain rare identifiers."),
    make_paper("2501.00002", "Deep learning for limit order books",
               "A neural model forecasts short-term price moves from order book states.\n\n"
               "The model beats a linear baseline on out-of-sample Sharpe ratio.", published="2022-06-01"),
    make_paper("2501.00003", "Variance estimators for heavy-tailed returns",
               "We study robust variance estimation under heavy tails.\n\n"
               "The trimmed estimator attains the minimax rate for finite fourth moments.",
               published="2024-03-20"),
]


@pytest.fixture
def kb(tmp_path):
    base = KnowledgeBase(tmp_path / "kb", "test-hash", embedding_function=HashEmbedding())
    yield base
    base.close()


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=str(tmp_path / "kb"), retrieval_k=3, max_context_chars=5000,
                    chunk_size=200, chunk_overlap=20)
