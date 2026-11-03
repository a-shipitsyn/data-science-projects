"""Decomposition RAG: split the question, retrieve per sub-question, answer from cited fragments."""

import logging
import re
import uuid
from dataclasses import asdict, dataclass, field

from .llm import LLMError, load_prompt
from .retrieval import Hit, Retriever

logger = logging.getLogger(__name__)

NO_DATA = "The indexed papers do not contain information to answer this question."


def decompose_schema(min_n: int, max_n: int) -> dict:
    return {
        "type": "object",
        "properties": {"sub_questions": {"type": "array", "items": {"type": "string"},
                                         "minItems": min_n, "maxItems": max_n}},
        "required": ["sub_questions"],
    }


@dataclass
class Answer:
    request_id: str
    question: str
    status: str                      # "ok", "no_data" or "llm_error"
    answer: str
    sub_questions: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # numbered fragments: n, paper_id, title, url
    context: str = ""
    cited: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def decompose(llm, question: str, min_n: int, max_n: int) -> list[str]:
    """Sub-questions from the LLM; falls back to the question itself on any failure."""
    system = load_prompt("decompose").format(min_n=min_n, max_n=max_n)
    try:
        data = llm.complete_json(system, question, decompose_schema(min_n, max_n))
    except LLMError as e:
        logger.warning("Decomposition failed, searching with the question itself: %s", e)
        return [question]
    subs, seen = [], set()
    for sub in data["sub_questions"]:
        sub = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(sub)).strip()
        if sub and sub.lower() not in seen:
            seen.add(sub.lower())
            subs.append(sub)
    if not subs:
        return [question]
    return subs[:max_n]


def gather(retriever: Retriever, queries: list[str], k: int, max_chars: int, **filters) -> list[Hit]:
    """Union of the top-k hits of every query, without duplicates, within the context budget.

    Hits are taken round-robin (first hit of every query, then second, ...) so that
    each sub-question is represented when the budget runs out.
    """
    per_query = [retriever.search(q, k, **filters) for q in queries]
    chosen, seen, used = [], set(), 0
    for rank in range(k):
        for hits in per_query:
            if rank >= len(hits) or hits[rank].chunk_id in seen:
                continue
            hit = hits[rank]
            if used + len(hit.text) > max_chars:
                logger.info("Context budget reached: %d fragments kept", len(chosen))
                return chosen
            seen.add(hit.chunk_id)
            chosen.append(hit)
            used += len(hit.text)
    return chosen


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{n}] {h.title} ({h.paper_id}, {h.published[:4]}{', note' if h.kind == 'note' else ''})\n{h.text}"
        for n, h in enumerate(hits, 1))


def cited_numbers(text: str, n_sources: int) -> list[int]:
    numbers = {int(n) for group in re.findall(r"\[([\d,\s]+)\]", text) for n in re.findall(r"\d+", group)}
    return sorted(n for n in numbers if 1 <= n <= n_sources)


def answer_question(llm, retriever: Retriever, question: str, settings, **filters) -> Answer:
    request_id = uuid.uuid4().hex[:12]
    logger.info("[%s] question: %s", request_id, question[:120])
    question = question.strip()
    if not question:
        return Answer(request_id, question, "no_data", "Empty question.")

    subs = decompose(llm, question, settings.min_sub_questions, settings.max_sub_questions)
    logger.info("[%s] %d sub-questions: %s", request_id, len(subs), subs)
    hits = gather(retriever, [question] + [s for s in subs if s != question],
                  settings.retrieval_k, settings.max_context_chars, **filters)
    sources = [{"n": n, "paper_id": h.paper_id, "title": h.title, "url": h.url, "kind": h.kind}
               for n, h in enumerate(hits, 1)]
    if not hits:
        logger.warning("[%s] nothing retrieved", request_id)
        return Answer(request_id, question, "no_data", NO_DATA, subs)

    context = format_context(hits)
    prompt = f"QUESTION: {question}\n\nFRAGMENTS:\n\n{context}"
    try:
        text = llm.complete(load_prompt("synthesize"), prompt).strip()
    except LLMError as e:
        logger.error("[%s] synthesis failed: %s", request_id, e)
        return Answer(request_id, question, "llm_error",
                      "The answer could not be generated; the retrieved fragments are listed below.",
                      subs, sources, context)
    status = "no_data" if text.startswith(NO_DATA[:40]) else "ok"
    cited = cited_numbers(text, len(hits))
    if status == "ok" and not cited:
        logger.warning("[%s] answer has no valid citations", request_id)
    logger.info("[%s] %s, %d fragments, cited %s", request_id, status, len(hits), cited)
    return Answer(request_id, question, status, text, subs, sources, context, cited)
