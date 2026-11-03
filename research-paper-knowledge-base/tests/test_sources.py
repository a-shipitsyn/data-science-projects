from conftest import FIXTURES

from paper_kb.extract import arxiv_html_to_text
from paper_kb.paper import Paper
from paper_kb.sources.arxiv import ArxivSource, build_query, parse_feed


class FakeResponse:
    def __init__(self, content):
        self.content = content.encode() if isinstance(content, str) else content
        self.text = self.content.decode()

    def json(self):
        import json
        return json.loads(self.text)


class FakeClient:
    def __init__(self, responses):
        self.responses = responses   # url substring -> content or None (404)
        self.urls = []

    def get(self, url, params=None, allow_404=False):
        self.urls.append((url, params))
        for key, content in self.responses.items():
            if key in url:
                return None if content is None else FakeResponse(content)
        raise AssertionError(f"unexpected url {url}")


def test_arxiv_html_keeps_article_prose_only():
    title, abstract, text = arxiv_html_to_text((FIXTURES / "arxiv_paper.html").read_text())
    assert title == "Hybrid retrieval for technical search"
    assert abstract == "We combine BM25 with dense embeddings for retrieval."
    assert text.startswith("Hybrid retrieval for technical search\nAbstract: We combine BM25")
    assert "1 Introduction\nFusion needs no training (Cormack et al., 2009)." in text  # heading kept
    assert r"\mathcal{L}=\sum_{i}x_{i}" in text                                    # formula as LaTeX
    assert "Figure 1: Training curve." in text
    for noise in ("navigation", "footer", "var a", "Ada Lovelace", "References", "funding agency"):
        assert noise not in text


def test_arxiv_html_without_article_returns_empty():
    assert arxiv_html_to_text("<html><body><p>Not found</p></body></html>") == ("", "", "")


def test_build_query():
    assert build_query("limit  order book") == "all:limit AND all:order AND all:book"
    assert build_query('"limit order book"') == 'all:"limit order book"'
    assert build_query('"limit order book"', "q-fin.TR") == 'cat:q-fin.TR AND all:"limit order book"'


def test_arxiv_category_goes_into_the_search_query():
    client = FakeClient({"api/query": (FIXTURES / "arxiv_feed.xml").read_text()})
    ArxivSource(client, category="cs.IR").search("vector search", 5)
    assert client.urls[0][1]["search_query"] == "cat:cs.IR AND all:vector AND all:search"


def test_parse_feed():
    papers = parse_feed((FIXTURES / "arxiv_feed.xml").read_text())
    assert [p.paper_id for p in papers] == ["arxiv:2501.00001", "arxiv:cs/0601001"]
    first = papers[0]
    assert first.version == "v2"
    assert first.title == "Hybrid retrieval for technical search"
    assert first.abstract == "We combine BM25 with dense embeddings for retrieval."
    assert first.authors == ["Ada Lovelace", "Alan Turing"]
    assert first.published == "2025-01-02"
    assert first.year == 2025


def test_arxiv_full_text_and_fallback_to_abstract():
    html = (FIXTURES / "arxiv_paper.html").read_text()
    source = ArxivSource(FakeClient({"2501.00001v2": html, "2501.00009v1": None}))
    paper = source.fetch_full_text(Paper("arxiv", "2501.00001", "T", "abstract", "u", version="v2"))
    assert paper.text_kind == "full" and "1 Introduction" in paper.text

    missing = source.fetch_full_text(Paper("arxiv", "2501.00009", "T", "abstract", "u", version="v1"))
    assert missing.text_kind == "abstract" and missing.text == ""
    assert missing.body() == "T\n\nabstract"
