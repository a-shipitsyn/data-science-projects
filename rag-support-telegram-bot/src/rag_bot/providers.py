# Embedding and LLM providers.
# "offline" providers need no API keys: a hashing bag-of-words embedding and a
# stand-in LLM that echoes retrieved context. They exist for demos and tests.
import hashlib
import re
from typing import Dict, List, Protocol

import numpy as np
from llama_index.core import Settings as LlamaSettings
from llama_index.core.base.embeddings.base import BaseEmbedding

from rag_bot.settings import Settings

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


Messages = List[Dict[str, str]]  # [{"role": "user" | "assistant", "content": ...}]


class LLM(Protocol):
    def complete(self, system: str, messages: Messages) -> str: ...


class HashEmbedding(BaseEmbedding):
    """Deterministic bag-of-words embedding (hashing trick), L2-normalised."""
    dim: int = 512

    def _embed(self, text: str) -> List[float]:
        vec = np.zeros(self.dim)
        for token in TOKEN_RE.findall(text.lower()):
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return (vec / norm if norm else vec).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, messages: Messages) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=messages,
        )
        if message.stop_reason == "refusal":
            raise RuntimeError("The model declined to answer this request")
        # content may start with thinking blocks: keep text blocks only
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        if not text:  # e.g. max_tokens reached while thinking
            raise RuntimeError(f"Empty reply from the model (stop_reason={message.stop_reason})")
        return text


class OfflineLLM:
    """Answers with the best retrieved chunk of the last message, without calling a model."""

    def complete(self, system: str, messages: Messages) -> str:
        documentation = _tag(messages[-1]["content"], "documentation")
        if not documentation:
            return "I could not find this in the documentation."
        first_chunk = documentation.split("\n\n<chunk")[0]
        return re.sub(r"</?chunk[^>]*>", "", first_chunk).strip()[:400]


def _tag(text: str, name: str) -> str:
    m = re.search(rf"<{name}>\n?(.*?)\n?</{name}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def configure_embeddings(settings: Settings) -> None:
    """Set the global llama-index embedding model (retrieval needs no llama-index LLM)."""
    if settings.embedding_provider == "offline":
        LlamaSettings.embed_model = HashEmbedding()
    elif settings.embedding_provider == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        LlamaSettings.embed_model = OpenAIEmbedding(model=settings.embedding_model)
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def embedding_model_key(settings: Settings) -> str:
    """Identifies the embedding model in the cache: vectors of different models never mix."""
    if settings.embedding_provider == "offline":
        return f"offline:hash{HashEmbedding().dim}"
    return f"{settings.embedding_provider}:{settings.embedding_model}"


def create_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "offline":
        return OfflineLLM()
    if settings.llm_provider == "anthropic":
        return AnthropicLLM(settings.llm_model, settings.max_tokens)
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")
