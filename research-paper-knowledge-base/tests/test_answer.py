import pytest
from conftest import PAPERS, FakeSource, ScriptedLLM

from paper_kb.answer import NO_DATA, answer_question, cited_numbers, decompose, gather
from paper_kb.ingest import ingest_topic, make_splitter
from paper_kb.llm import LLMError
from paper_kb.retrieval import Retriever, tokenize


@pytest.fixture
def retriever(kb):
    ingest_topic(kb, {"arxiv": FakeSource("arxiv", PAPERS)}, "retrieval", 10, make_splitter(200, 20))
    return Retriever(kb)


def test_tokenize_keeps_technical_terms():
    assert tokenize("What is BM25 and GPT-4?") == ["bm25", "gpt-4"]


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_search_modes_find_the_right_paper(retriever, mode):
    hits = retriever.search("limit order book forecasting", 2, mode=mode)
    assert hits[0].paper_id == "arxiv:2501.00002"


def test_search_filters(retriever):
    recent = retriever.search("retrieval and order books", 5, year_from=2024)
    assert "arxiv:2501.00002" not in {h.paper_id for h in recent}   # published 2022
    assert retriever.search("retrieval", 5, topic="unknown topic") == []


def test_decompose_cleans_and_falls_back():
    llm = ScriptedLLM({"sub_questions": ["1. What is BM25?", "- What is BM25?", "How fast is it?"]})
    assert decompose(llm, "Q", 2, 5) == ["What is BM25?", "How fast is it?"]
    assert decompose(ScriptedLLM(LLMError("down")), "Q", 2, 5) == ["Q"]
    assert decompose(ScriptedLLM("not json"), "Q", 2, 5) == ["Q"]


def test_gather_deduplicates_and_respects_budget(retriever):
    hits = gather(retriever, ["rank fusion", "hybrid retrieval"], 3, max_chars=10_000)
    assert len({h.chunk_id for h in hits}) == len(hits)
    small = gather(retriever, ["rank fusion", "order books"], 3, max_chars=400)
    assert len(small) == 2 and sum(len(h.text) for h in small) <= 400
    assert {h.paper_id for h in small} == {"arxiv:2501.00001", "arxiv:2501.00002"}   # round-robin


def test_cited_numbers():
    assert cited_numbers("A [1]. B [2, 3][9]. C [x]", 3) == [1, 2, 3]


def test_answer_flow(retriever, settings):
    llm = ScriptedLLM({"sub_questions": ["How does rank fusion work?", "How are order books modelled?"]},
                      "Fusion combines keyword and dense rankings [1]. A neural model forecasts moves [2].")
    result = answer_question(llm, retriever, "  Rank fusion and order books?  ", settings)
    assert result.status == "ok"
    assert result.question == "Rank fusion and order books?"
    assert result.cited == [1, 2]
    assert result.sources[0]["n"] == 1 and "[1] " in result.context
    synth_prompt = llm.calls[1]["prompt"]
    assert synth_prompt.startswith("QUESTION: Rank fusion and order books?")


def test_answer_reports_llm_failure_with_sources(retriever, settings):
    llm = ScriptedLLM({"sub_questions": ["Fusion?", "Order books?"]}, LLMError("timeout"))
    result = answer_question(llm, retriever, "Fusion and order books?", settings)
    assert result.status == "llm_error" and result.sources


def test_answer_no_data(kb, settings):
    llm = ScriptedLLM({"sub_questions": ["a?", "b?"]})
    result = answer_question(llm, Retriever(kb), "Anything?", settings)
    assert result.status == "no_data" and result.answer == NO_DATA
