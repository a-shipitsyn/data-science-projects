import json

import pytest
from conftest import PAPERS, FakeSource, ScriptedLLM

from paper_kb import evaluate
from paper_kb.digest import topic_digest
from paper_kb.ingest import ingest_topic, make_splitter
from paper_kb.llm import LLMError
from paper_kb.notes import write_notes
from paper_kb.retrieval import Retriever
from paper_kb.settings import Settings, load_settings

NOTE = {"problem": "Keyword search misses paraphrases.", "method": "Reciprocal rank fusion.",
        "data": "not stated", "findings": "Recall improves on rare identifiers.",
        "limitations": "not stated", "keywords": ["BM25", "rank fusion"]}


@pytest.fixture
def filled(kb):
    ingest_topic(kb, {"arxiv": FakeSource("arxiv", PAPERS)}, "retrieval", 10, make_splitter(200, 20))
    return kb


def test_notes_are_stored_indexed_and_skipped_next_time(filled):
    llm = ScriptedLLM(NOTE, LLMError("bad"), NOTE)
    result = write_notes(llm, filled, 10_000, "llama3.2", topic="retrieval")
    assert len(result["written"]) == 2 and len(result["failed"]) == 1
    paper_id = result["written"][0]
    assert filled.note(paper_id)["method"] == "Reciprocal rank fusion."
    assert f"{paper_id}#note" in set(filled.collection.get()["ids"])

    again = write_notes(ScriptedLLM(NOTE), filled, 10_000, "llama3.2", topic="retrieval")
    assert len(again["skipped"]) == 2 and len(again["written"]) == 1

    hits = Retriever(filled).search("Reading note reciprocal rank fusion", 3, mode="bm25")
    assert any(h.kind == "note" for h in hits)


def test_digest_heading_drops_phrase_quotes(kb):
    ingest_topic(kb, {"arxiv": FakeSource("arxiv", PAPERS[:1])}, '"rank fusion"', 10,
                 make_splitter(200, 20))
    assert topic_digest(kb, '"rank fusion"').startswith("# rank fusion\n")


def test_digest(filled):
    write_notes(ScriptedLLM(NOTE), filled, 10_000, "llama3.2", paper_ids=["arxiv:2501.00001"])
    md = topic_digest(filled, "retrieval")
    assert md.startswith("# retrieval\n")
    assert "3 papers, 1 with reading notes" in md
    assert "- **Method:** Reciprocal rank fusion." in md
    assert "**Abstract.** A neural model forecasts" in md
    with pytest.raises(ValueError):
        topic_digest(filled, "missing")


def test_first_relevant_rank():
    assert evaluate.first_relevant_rank(["a", "b", "c"], {"c", "x"}) == 3
    assert evaluate.first_relevant_rank(["a"], {"x"}) is None


def test_retrieval_eval(filled, tmp_path):
    questions = [{"question": "limit order book forecasting", "reference_answer": "-",
                  "relevant_papers": ["arxiv:2501.00002"]},
                 {"question": "unrelated words", "reference_answer": "-", "relevant_papers": ["arxiv:9999.99999"]}]
    rows = evaluate.evaluate_retrieval(Retriever(filled), questions, 3)
    summary = evaluate.summarize_retrieval(rows)
    for mode in ("dense", "bm25", "hybrid"):
        assert summary[mode]["n"] == 2
        assert summary[mode]["hit_rate"] == 0.5 and summary[mode]["mrr"] == 0.5
    evaluate.save(rows, summary, tmp_path / "out", "retrieval")
    assert json.loads((tmp_path / "out" / "retrieval.json").read_text())["summary"] == summary


def test_answer_eval_scores_from_item_verdicts(filled, settings):
    questions = [{"question": "Rank fusion?",
                  "reference_answer": "Fusion merges rankings. It raises recall on rare identifiers.",
                  "relevant_papers": ["arxiv:2501.00001"]}]
    llm = ScriptedLLM({"sub_questions": ["How does fusion merge?", "What does fusion improve?"]},
                      "Fusion merges keyword and dense rankings [1].")
    judge = ScriptedLLM(
        {"claims": [{"claim": "Fusion merges rankings", "supported": True},
                    {"claim": "It is fast", "supported": False}]},
        {"facts": [{"fact": "merges rankings", "stated": True},
                   {"fact": "raises recall", "stated": False}]})
    rows = evaluate.evaluate_answers(llm, judge, Retriever(filled), questions, settings)
    assert rows[0]["faithfulness"] == 0.5 and rows[0]["correctness"] == 0.5
    assert rows[0]["relevant_retrieved"] == 1
    assert evaluate.summarize_answers(rows)["faithfulness"] == 0.5


def test_judge_error_is_not_scored_zero(filled, settings):
    questions = [{"question": "Rank fusion?", "reference_answer": "x", "relevant_papers": []}]
    llm = ScriptedLLM({"sub_questions": ["a?", "b?"]}, "Answer [1].")
    rows = evaluate.evaluate_answers(llm, ScriptedLLM(LLMError("down")), Retriever(filled), questions, settings)
    assert rows[0]["faithfulness"] is None and rows[0]["judge_error"]
    summary = evaluate.summarize_answers(rows)
    assert summary["faithfulness"] is None and summary["judge_errors"] == 1


def test_load_questions_validates(tmp_path):
    path = tmp_path / "q.json"
    path.write_text(json.dumps({"questions": [{"question": "q"}]}))
    with pytest.raises(ValueError, match="relevant_papers"):
        evaluate.load_questions(path)


def test_judge_model_defaults_to_the_generation_model():
    assert Settings(ollama_model="a").judge_model == "a"
    assert Settings(ollama_model="a", ollama_judge_model="b").judge_model == "b"
    assert Settings(ollama_url="u").judge_url == "u"
    assert Settings(ollama_url="u", ollama_judge_url="v").judge_url == "v"


def test_settings_env_and_overrides(tmp_path, monkeypatch):
    config = tmp_path / "s.json"
    config.write_text(json.dumps({"retrieval_k": 7, "sources": ["arxiv"]}))
    monkeypatch.setenv("KB_OLLAMA_MODEL", "qwen2.5")
    monkeypatch.setenv("KB_SOURCES", "arxiv")
    s = load_settings(config, data_dir="x")
    assert (s.retrieval_k, s.ollama_model, s.sources, s.data_dir) == (7, "qwen2.5", ("arxiv",), "x")

    config.write_text(json.dumps({"retrival_k": 7}))
    with pytest.raises(ValueError, match="Unknown settings"):
        load_settings(config)
