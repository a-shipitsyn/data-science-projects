# Retrieval evaluation: does search return the document that answers the question?
# Cases file: JSON list of {"question": ..., "expected_file": ...}.
# Prints hit rate and MRR for several similarity thresholds, to tune the settings.
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from rag_bot.retrieval import KnowledgeBase


def evaluate(kb: KnowledgeBase, cases: List[dict], thresholds: Sequence[float],
             top_k: int) -> List[Dict[str, float]]:
    # search once without a threshold, then filter: scores do not depend on it
    results = [(case["expected_file"], kb.search(case["question"], top_k=top_k, threshold=-1.0))
               for case in cases]
    rows = []
    for threshold in thresholds:
        hits_found, reciprocal_ranks, chunks = 0, 0.0, 0
        for expected, hits in results:
            kept = [h for h in hits if h.score >= threshold]
            chunks += len(kept)
            ranks = [i for i, h in enumerate(kept, 1) if h.file_name == expected]
            if ranks:
                hits_found += 1
                reciprocal_ranks += 1 / ranks[0]
        n = len(results)
        rows.append({"threshold": threshold, "hit_rate": hits_found / n,
                     "mrr": reciprocal_ranks / n, "avg_chunks": chunks / n})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--cases", required=True, help="evaluation questions JSON")
    ap.add_argument("--thresholds", default="0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--top-k", type=int, help="default: top_k from settings")
    args = ap.parse_args()

    import os
    from dotenv import load_dotenv
    from rag_bot.db import connect
    from rag_bot.providers import configure_embeddings
    from rag_bot.settings import load_settings
    from rag_bot.store import DocumentStore
    load_dotenv()
    settings = load_settings(args.settings)
    configure_embeddings(settings)

    cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    top_k = args.top_k or settings.top_k
    with connect(os.environ["DATABASE_URL"]) as pool:
        rows = evaluate(KnowledgeBase(settings, DocumentStore(pool)), cases,
                        [float(t) for t in args.thresholds.split(",")], top_k)
    print(f"{len(cases)} questions, top_k={top_k}")
    print("threshold  hit_rate  MRR   avg_chunks")
    for r in rows:
        print(f"{r['threshold']:>9.2f}  {r['hit_rate']:>8.2f}  {r['mrr']:.2f}  {r['avg_chunks']:>10.1f}")


if __name__ == "__main__":
    main()
