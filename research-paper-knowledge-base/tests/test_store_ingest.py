import pytest
from conftest import PAPERS, FakeSource, HashEmbedding, make_paper

from paper_kb.ingest import ingest_topic, make_splitter, refresh, split_text
from paper_kb.store import KnowledgeBase


@pytest.fixture
def splitter():
    return make_splitter(200, 20)


def sources(papers=PAPERS):
    return {"arxiv": FakeSource("arxiv", papers)}


def vector_ids(kb):
    return set(kb.collection.get()["ids"])


def test_split_merges_short_pieces_and_keeps_sentence_ends(splitter):
    text = "Heading\n" + "A sentence about retrieval. " * 20
    chunks = split_text(splitter, text, min_chars=50)
    assert chunks[0].startswith("Heading\nA sentence")
    assert all(len(c) >= 50 for c in chunks)
    assert not any(c.startswith(".") for c in chunks)


def test_ingest_is_incremental(kb, splitter):
    src = sources()
    first = ingest_topic(kb, src, "retrieval", 10, splitter)
    assert len(first.new) == 3 and first.found == 3

    second = ingest_topic(kb, src, "retrieval", 10, splitter)
    assert len(second.unchanged) == 3 and not second.new
    assert len(src["arxiv"].fetched) == 3      # nothing downloaded again

    chunk_ids = {c["chunk_id"] for c in kb.chunks()}
    assert vector_ids(kb) == chunk_ids         # catalogue and index agree
    assert kb.topics()[0]["papers"] == 3 and kb.topics()[0]["refreshed_at"]


def test_topic_remembers_its_category_for_refresh(kb, splitter):
    ingest_topic(kb, sources(), "order books", 10, splitter, category="q-fin.TR")
    assert kb.topics()[0]["category"] == "q-fin.TR"
    seen = []
    refresh(kb, lambda names, category: (seen.append(category) or sources()), splitter)
    assert seen == ["q-fin.TR"]


def test_new_version_replaces_old_chunks(kb, splitter):
    ingest_topic(kb, sources(), "t", 10, splitter)
    long_text = "Version one text. " * 40
    old = make_paper("2501.00001", "Hybrid retrieval for technical search", long_text)
    ingest_topic(kb, {"arxiv": FakeSource("arxiv", [old])}, "t", 10, splitter, force=True)
    n_old = len(kb.chunks(["arxiv:2501.00001"]))
    assert n_old > 1

    new = make_paper("2501.00001", "Hybrid retrieval for technical search", "Version two, short.", version="v2")
    report = ingest_topic(kb, {"arxiv": FakeSource("arxiv", [new])}, "t", 10, splitter)
    assert report.updated == ["arxiv:2501.00001"]
    chunks = kb.chunks(["arxiv:2501.00001"])
    assert len(chunks) == 1 and chunks[0]["text"] == "Version two, short."
    assert vector_ids(kb) == {c["chunk_id"] for c in kb.chunks()}
    assert kb.paper("arxiv:2501.00001")["version"] == "v2"


def test_failed_paper_does_not_stop_ingest(kb, splitter):
    class Broken(FakeSource):
        def fetch_full_text(self, paper):
            if paper.source_id == "2501.00001":
                raise ConnectionError("network down")
            return paper

    report = ingest_topic(kb, {"arxiv": Broken("arxiv", PAPERS[:2])}, "t", 10, splitter)
    assert report.failed == ["arxiv:2501.00001"] and report.new == ["arxiv:2501.00002"]


def test_search_failure_in_one_source_keeps_others(kb, splitter):
    class Down(FakeSource):
        def search(self, topic, max_results):
            raise TimeoutError("down")

    report = ingest_topic(kb, {"down": Down("down", []), "arxiv": FakeSource("arxiv", PAPERS[2:])},
                          "t", 10, splitter)
    assert report.new == ["arxiv:2501.00003"]


def test_topics_filters_and_removal(kb, splitter):
    ingest_topic(kb, sources(), "a", 10, splitter)
    ingest_topic(kb, {"arxiv": FakeSource("arxiv", PAPERS[:1])}, "b", 10, splitter)
    assert kb.paper_topics("arxiv:2501.00001") == ["a", "b"]
    assert "arxiv:2501.00002" not in {p["paper_id"] for p in kb.papers(year_from=2023)}

    assert kb.remove_topic("b", delete_papers=True) == 0      # its paper is also in "a"
    assert kb.remove_topic("a", delete_papers=True) == 3
    assert kb.stats()["papers"] == 0 and vector_ids(kb) == set()


def test_refresh_reruns_saved_topics(kb, splitter):
    ingest_topic(kb, sources(), "a", 10, splitter)
    reports = refresh(kb, lambda names, category: sources(), splitter)
    assert [len(r.unchanged) for r in reports] == [3]
    with pytest.raises(ValueError):
        refresh(kb, lambda names, category: sources(), splitter, topic="unknown")


def test_embedding_model_mismatch_needs_reindex(tmp_path, splitter):
    kb = KnowledgeBase(tmp_path, "model-a", embedding_function=HashEmbedding())
    ingest_topic(kb, sources(), "t", 10, splitter)
    kb.close()

    other = KnowledgeBase(tmp_path, "model-b", embedding_function=HashEmbedding())
    with pytest.raises(RuntimeError, match="reindex"):
        other.dense_search("lora", 3)
    assert other.reindex() == other.stats()["chunks"]
    assert other.dense_search("hybrid retrieval", 1)
    other.close()
