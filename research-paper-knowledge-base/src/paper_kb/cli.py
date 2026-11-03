"""Command-line interface: python -m paper_kb <command> ..."""

import argparse
import json
import logging
import sys
from pathlib import Path

from . import digest, evaluate, ingest, notes
from .answer import answer_question
from .llm import LLMError, OllamaLLM
from .retrieval import MODES, Retriever
from .settings import load_settings
from .sources import make_sources
from .store import KnowledgeBase

logger = logging.getLogger("paper_kb")


def add_filters(parser):
    parser.add_argument("--topic", help="only papers of this saved topic")
    parser.add_argument("--year-from", type=int, help="only papers published in or after this year")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb", description="Research-paper knowledge base")
    parser.add_argument("--config", help="settings JSON (default: config/settings.json)")
    parser.add_argument("--data-dir", help="knowledge-base folder (overrides the config)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="search a topic and add its papers")
    p.add_argument("topic", help='search terms; quote a phrase: "\\"base editing\\""')
    p.add_argument("--max-results", type=int, help="papers per source")
    p.add_argument("--category", default="",
                   help="narrow arXiv to a subject class, e.g. q-fin.TR, cs.IR")
    p.add_argument("--force", action="store_true", help="re-download papers already stored")

    p = sub.add_parser("refresh", help="re-run saved topic searches (for cron)")
    p.add_argument("--topic", help="only this topic")

    sub.add_parser("topics", help="list saved topics")
    p = sub.add_parser("remove-topic", help="delete a saved topic")
    p.add_argument("topic")
    p.add_argument("--delete-papers", action="store_true", help="also delete papers only in this topic")

    p = sub.add_parser("papers", help="list papers in the catalogue")
    add_filters(p)
    p = sub.add_parser("show", help="show one paper with its note")
    p.add_argument("paper_id", help="e.g. arxiv:2401.00001")
    p = sub.add_parser("remove", help="delete a paper")
    p.add_argument("paper_id")

    p = sub.add_parser("search", help="find fragments (no LLM)")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--mode", choices=MODES, default="hybrid")
    add_filters(p)

    p = sub.add_parser("ask", help="answer a complex question with citations")
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    add_filters(p)

    p = sub.add_parser("notes", help="write reading notes for papers without one")
    p.add_argument("--topic")
    p.add_argument("--paper", nargs="+", dest="paper_ids")
    p.add_argument("--force", action="store_true", help="rewrite existing notes")

    p = sub.add_parser("digest", help="export a topic as markdown")
    p.add_argument("topic")
    p.add_argument("--output", required=True, help="markdown file to write")

    p = sub.add_parser("eval", help="evaluate retrieval (and answers) on a question set")
    p.add_argument("--questions", required=True, help="question set JSON")
    p.add_argument("--output", required=True, help="folder for results JSON/CSV")
    p.add_argument("--answers", action="store_true", help="also run and judge full answers (slow, LLM)")
    p.add_argument("-k", type=int, default=5)
    add_filters(p)

    p = sub.add_parser("sweep", help="score retrieval settings on a question set (no LLM)")
    p.add_argument("--questions", required=True)
    p.add_argument("--output", help="folder for results JSON/CSV")
    p.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 10])
    p.add_argument("--rrf-values", type=int, nargs="+", default=[10, 60, 200],
                   help="reciprocal rank fusion constants to try")
    add_filters(p)

    sub.add_parser("reindex", help="rebuild the vector index from the catalogue")
    p = sub.add_parser("rechunk", help="re-split stored papers with the current chunk settings")
    sub.add_parser("stats", help="knowledge-base counts")
    return parser


def filters(args) -> dict:
    return {"topic": args.topic, "year_from": args.year_from}


def make_llm(settings, model: str | None = None, url: str | None = None) -> OllamaLLM:
    llm = OllamaLLM(model or settings.ollama_model, url or settings.ollama_url, settings.ollama_num_ctx)
    llm.check()
    return llm


def run(args, settings) -> int:
    kb = KnowledgeBase(settings.data_dir, settings.embedding_model)
    splitter = ingest.make_splitter(settings.chunk_size, settings.chunk_overlap)
    try:
        if args.command == "ingest":
            sources = make_sources(settings, category=args.category)
            report = ingest.ingest_topic(kb, sources, args.topic,
                                         args.max_results or settings.max_results_per_source,
                                         splitter, force=args.force, category=args.category)
            print(report.summary())
            return 1 if report.found == 0 else 0

        if args.command == "refresh":
            reports = ingest.refresh(kb, lambda names, category: make_sources(settings, names, category),
                                     splitter, args.topic)
            for report in reports:
                print(report.summary())
            return 0

        if args.command == "topics":
            for t in kb.topics():
                print(f"{t['name']}\t{t['papers']} papers\tsources={','.join(t['sources'])}\t"
                      f"category={t['category'] or '-'}\tmax={t['max_results']}\t"
                      f"refreshed={t['refreshed_at'] or '-'}")
            return 0

        if args.command == "remove-topic":
            removed = kb.remove_topic(args.topic, args.delete_papers)
            print(f"Topic removed; {removed} papers deleted")
            return 0

        if args.command == "papers":
            for p in kb.papers(**filters(args)):
                print(f"{p['paper_id']}\t{p['published']}\t{p['text_kind']}\t{p['title'][:90]}")
            return 0

        if args.command == "show":
            paper = kb.paper(args.paper_id)
            if paper is None:
                print(f"Unknown paper {args.paper_id}", file=sys.stderr)
                return 1
            paper["topics"] = kb.paper_topics(args.paper_id)
            paper["note"] = kb.note(args.paper_id)
            print(json.dumps(paper, indent=2, ensure_ascii=False))
            return 0

        if args.command == "remove":
            found = kb.remove_paper(args.paper_id)
            print("Removed" if found else f"Unknown paper {args.paper_id}")
            return 0 if found else 1

        if args.command == "search":
            for n, hit in enumerate(Retriever(kb).search(args.query, args.k, args.mode, **filters(args)), 1):
                print(f"\n[{n}] {hit.title} ({hit.paper_id}, {hit.published[:4]}, {hit.kind}) {hit.url}")
                print("-" * 80)
                print(hit.text)
            return 0

        if args.command == "ask":
            result = answer_question(make_llm(settings), Retriever(kb), args.question, settings,
                                     **filters(args))
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
            else:
                print(result.answer)
                if result.sources:
                    print("\nSources:")
                    for s in result.sources:
                        mark = "*" if s["n"] in result.cited else " "
                        print(f" {mark}[{s['n']}] {s['title']} — {s['url']}")
            return 0 if result.status != "llm_error" else 2

        if args.command == "notes":
            result = notes.write_notes(make_llm(settings), kb, settings.notes_max_chars,
                                       settings.ollama_model, args.topic, args.paper_ids, args.force)
            print({k: len(v) for k, v in result.items()})
            return 1 if result["failed"] else 0

        if args.command == "digest":
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(digest.topic_digest(kb, args.topic))
            print(f"Written {args.output}")
            return 0

        if args.command == "eval":
            questions = evaluate.load_questions(args.questions)
            retriever = Retriever(kb)
            rows = evaluate.evaluate_retrieval(retriever, questions, args.k, **filters(args))
            summary = evaluate.summarize_retrieval(rows)
            evaluate.save(rows, summary, args.output, "retrieval")
            print(json.dumps({"retrieval": summary}, indent=2))
            if args.answers:
                llm = make_llm(settings)
                # a second local model can grade (ollama_judge_model); by default the same one does,
                # which makes the scores self-judged: see README, Results
                same_judge = (settings.judge_model == settings.ollama_model
                              and settings.judge_url == settings.ollama_url)
                judge = llm if same_judge else make_llm(settings, settings.judge_model,
                                                        settings.judge_url)
                logger.info("Answering with %s, judging with %s", llm.model, judge.model)
                rows = evaluate.evaluate_answers(llm, judge, retriever, questions, settings,
                                                 **filters(args))
                summary = evaluate.summarize_answers(rows)
                evaluate.save(rows, summary, args.output, "answers")
                print(json.dumps({"answers": summary}, indent=2))
            return 0

        if args.command == "sweep":
            questions = evaluate.load_questions(args.questions)
            rows = evaluate.sweep_retrieval(kb, questions, args.k_values, args.rrf_values,
                                            **filters(args))
            best = rows[0]
            print(f"{'mode':8} {'k':>3} {'rrf_k':>6} {'hit_rate':>9} {'mrr':>6}")
            for row in rows:
                print(f"{row['mode']:8} {row['k']:>3} {row['rrf_k']:>6} "
                      f"{row['hit_rate']:>9.3f} {row['mrr']:>6.3f}")
            print(f"\nbest: {best['mode']} k={best['k']} rrf_k={best['rrf_k']} "
                  f"mrr={best['mrr']:.3f} hit_rate={best['hit_rate']:.3f}")
            if args.output:
                evaluate.save(rows, {"best": best, "chunk_size": settings.chunk_size,
                                     "chunk_overlap": settings.chunk_overlap}, args.output, "sweep")
            return 0

        if args.command == "reindex":
            print(f"Re-embedded {kb.reindex()} chunks with {settings.embedding_model}")
            return 0

        if args.command == "rechunk":
            n = kb.rechunk(lambda text: ingest.split_text(splitter, text))
            print(f"Re-split into {n} chunks (size {settings.chunk_size}, "
                  f"overlap {settings.chunk_overlap}) and re-embedded")
            return 0

        if args.command == "stats":
            print(json.dumps(kb.stats(), indent=2))
            return 0
    finally:
        kb.close()
    return 1


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        settings = load_settings(args.config, data_dir=args.data_dir)
        return run(args, settings)
    except (LLMError, ValueError, RuntimeError) as e:
        logger.error("%s", e)
        return 2
