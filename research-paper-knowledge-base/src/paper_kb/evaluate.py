"""Evaluation on a labelled question set.

1. Retrieval (no LLM): for every search mode, the rank of the first chunk from a paper
   marked relevant -> hit rate @k and MRR.
2. Answers (LLM, optional): run the full decomposition RAG flow, then a judge LLM
   - lists the answer's claims and marks each as supported by the retrieved fragments
     -> faithfulness = supported / claims (hallucination check);
   - lists the key facts of the reference answer and marks each as stated in the answer
     -> correctness = stated / key facts (does it actually answer the question).
   Scores are computed in code from the judge's per-item verdicts, not asked for as a number.
   Citation validity (cited fragment numbers exist) is checked without the LLM.

Question file (JSON): each item may carry a `paraphrase`, the same question asked in a
user's own words rather than the papers' vocabulary. Retrieval is scored on both phrasings,
which separates "the index works" from "the index works only when you already know the terms".

{"questions": [{"question": "...", "paraphrase": "...", "reference_answer": "...",
                "relevant_papers": ["arxiv:2401.00001"]}]}
"""

import csv
import json
import logging
import statistics
from pathlib import Path

from .answer import answer_question
from .llm import LLMError, load_prompt
from .retrieval import MODES, Retriever

logger = logging.getLogger(__name__)

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {"claims": {"type": "array", "items": {
        "type": "object",
        "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
        "required": ["claim", "supported"]}}},
    "required": ["claims"],
}
CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"fact": {"type": "string"}, "stated": {"type": "boolean"}},
        "required": ["fact", "stated"]}}},
    "required": ["facts"],
}


def load_questions(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    questions = data["questions"] if isinstance(data, dict) else data
    for i, q in enumerate(questions):
        missing = {"question", "reference_answer", "relevant_papers"} - set(q)
        if missing:
            raise ValueError(f"Question {i} lacks {sorted(missing)}")
    return questions


def first_relevant_rank(paper_ids: list[str], relevant: set[str]) -> int | None:
    for rank, paper_id in enumerate(paper_ids, 1):
        if paper_id in relevant:
            return rank
    return None


def phrasings(q: dict) -> list[tuple[str, str]]:
    """(wording, query) pairs for one question: the original and its paraphrase, if any."""
    pairs = [("original", q["question"])]
    if q.get("paraphrase"):
        pairs.append(("paraphrase", q["paraphrase"]))
    return pairs


def evaluate_retrieval(retriever: Retriever, questions: list[dict], k: int, **filters) -> list[dict]:
    rows = []
    for q in questions:
        relevant = set(q["relevant_papers"])
        for wording, query in phrasings(q):
            for mode in MODES:
                hits = retriever.search(query, k, mode=mode, **filters)
                rank = first_relevant_rank([h.paper_id for h in hits], relevant)
                rows.append({"question": q["question"], "query": query, "wording": wording,
                             "topic": q.get("topic", ""), "mode": mode, "rank": rank,
                             "hit": int(rank is not None), "rr": 1.0 / rank if rank else 0.0})
    return rows


def _scores(rows: list[dict]) -> dict:
    return {"hit_rate": round(statistics.mean(r["hit"] for r in rows), 3),
            "mrr": round(statistics.mean(r["rr"] for r in rows), 3), "n": len(rows)}


def summarize_retrieval(rows: list[dict]) -> dict:
    """Scores per mode, split by wording and by topic so a weak case cannot hide in the mean."""
    summary = {}
    for mode in MODES:
        subset = [r for r in rows if r["mode"] == mode]
        if not subset:
            continue
        entry = _scores(subset)
        for wording in ("original", "paraphrase"):
            group = [r for r in subset if r["wording"] == wording]
            if group:
                entry[wording] = _scores(group)
        topics = {r["topic"] for r in subset if r["topic"]}
        if topics:
            entry["by_topic"] = {t: _scores([r for r in subset if r["topic"] == t])
                                 for t in sorted(topics)}
        summary[mode] = entry
    return summary


def sweep_retrieval(kb, questions: list[dict], ks: list[int], rrf_constants: list[int],
                    **filters) -> list[dict]:
    """Score every (mode, k, fusion constant) combination on the same questions, best MRR first.

    Retrieval-time parameters only, so no re-indexing and no LLM: the knowledge base is read
    as it stands. Chunk size and overlap are swept separately, with `rechunk` between runs.
    """
    rows = []
    for constant in rrf_constants:
        retriever = Retriever(kb, rrf_k=constant)
        for k in ks:
            scored = evaluate_retrieval(retriever, questions, k, **filters)
            for mode, entry in summarize_retrieval(scored).items():
                rows.append({"mode": mode, "k": k, "rrf_k": constant, "hit_rate": entry["hit_rate"],
                             "mrr": entry["mrr"], "n": entry["n"]})
    return sorted(rows, key=lambda r: (-r["mrr"], -r["hit_rate"]))


def judge_faithfulness(llm, answer: str, context: str) -> tuple[float | None, list]:
    data = llm.complete_json(load_prompt("judge_faithfulness"),
                             f"SOURCES:\n{context}\n\nANSWER:\n{answer}", FAITHFULNESS_SCHEMA)
    claims = data["claims"]
    if not claims:
        return None, claims
    return sum(bool(c["supported"]) for c in claims) / len(claims), claims


def judge_correctness(llm, answer: str, reference: str) -> tuple[float | None, list]:
    data = llm.complete_json(load_prompt("judge_correctness"),
                             f"REFERENCE:\n{reference}\n\nANSWER:\n{answer}", CORRECTNESS_SCHEMA)
    facts = data["facts"]
    if not facts:
        return None, facts
    return sum(bool(f["stated"]) for f in facts) / len(facts), facts


def evaluate_answers(llm, judge, retriever: Retriever, questions: list[dict], settings, **filters) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        logger.info("Answer eval %d/%d: %s", i, len(questions), q["question"][:80])
        result = answer_question(llm, retriever, q["question"], settings, **filters)
        row = {"question": q["question"], "status": result.status, "answer": result.answer,
               "n_fragments": len(result.sources), "cited": result.cited,
               "relevant_retrieved": int(bool(set(q["relevant_papers"]) &
                                              {s["paper_id"] for s in result.sources})),
               "faithfulness": None, "correctness": None, "claims": [], "facts": [], "judge_error": ""}
        if result.status == "ok":
            try:
                row["faithfulness"], row["claims"] = judge_faithfulness(judge, result.answer, result.context)
                row["correctness"], row["facts"] = judge_correctness(judge, result.answer, q["reference_answer"])
            except LLMError as e:          # recorded, not scored as 0
                logger.error("Judge failed: %s", e)
                row["judge_error"] = str(e)
        elif result.status == "no_data":
            row["correctness"] = 0.0       # the reference exists, so "no data" is a miss
        rows.append(row)
    return rows


def summarize_answers(rows: list[dict]) -> dict:
    def mean(key):
        values = [r[key] for r in rows if r[key] is not None]
        return round(statistics.mean(values), 3) if values else None
    return {
        "n": len(rows),
        "answered": sum(r["status"] == "ok" for r in rows),
        "with_citations": sum(bool(r["cited"]) for r in rows if r["status"] == "ok"),
        "relevant_paper_in_context": mean("relevant_retrieved"),
        "faithfulness": mean("faithfulness"),
        "correctness": mean("correctness"),
        "judge_errors": sum(bool(r["judge_error"]) for r in rows),
    }


def save(rows: list[dict], summary: dict, output_dir: str | Path, name: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    if not rows:
        return
    flat = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows]
    with open(output_dir / f"{name}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
