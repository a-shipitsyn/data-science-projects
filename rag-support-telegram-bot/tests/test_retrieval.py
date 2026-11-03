from rag_bot.retrieval import KnowledgeBase, format_hits, source_files
from rag_bot.store import Hit


def test_search_finds_relevant_chunk(settings, indexed):
    hits = KnowledgeBase(settings, indexed).search("How do I reset the M7 monitor to factory settings?")
    assert hits and "MENU and UNLOCK" in hits[0].text
    assert hits[0].file_name == "indoor-monitor-m7.md"
    assert all(h.score >= settings.similarity_threshold for h in hits)
    assert len(hits) <= settings.top_k
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_threshold_and_top_k_overrides(settings, indexed):
    kb = KnowledgeBase(settings, indexed)
    assert kb.search("reset M7", threshold=0.99) == []
    assert len(kb.search("reset M7", top_k=1, threshold=-1)) == 1
    assert len(kb.search("reset M7", top_k=50, threshold=-1)) == indexed.count_chunks()


def test_format_hits_and_sources():
    hits = [Hit("a.md", "x", 0.9), Hit("b.md", "y", 0.5), Hit("a.md", "z", 0.4)]
    assert format_hits(hits[:1]) == '<chunk source="a.md" relevance="0.90">\nx\n</chunk>'
    assert source_files(hits) == ["a.md", "b.md"]
