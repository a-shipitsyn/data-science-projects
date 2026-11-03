"""arXiv: topic search through the arXiv API, full text from the HTML version of the paper."""

import logging
import re
import xml.etree.ElementTree as ET

import requests

from ..extract import arxiv_html_to_text
from ..paper import Paper
from .http import RateLimitedClient

logger = logging.getLogger(__name__)

API_URL = "https://export.arxiv.org/api/query"
HTML_URL = "https://arxiv.org/html/{id}{version}"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
ID_RE = re.compile(r"arxiv\.org/abs/(?P<id>.+?)(?P<version>v\d+)?$")


def build_query(topic: str, category: str = "") -> str:
    """Search all fields; a quoted topic is searched as a phrase, a category narrows the subject."""
    topic = topic.strip()
    if topic.startswith('"') and topic.endswith('"'):
        terms = [f"all:{topic}"]
    else:
        terms = [f"all:{w}" for w in re.split(r"\s+", topic) if w]
    if category:
        terms.insert(0, f"cat:{category}")
    return " AND ".join(terms)


def parse_feed(xml: str | bytes) -> list[Paper]:
    """Parse an arXiv API Atom feed into Paper records (abstract only, no full text yet)."""
    root = ET.fromstring(xml)
    papers = []
    for entry in root.findall("a:entry", ATOM):
        match = ID_RE.search(entry.findtext("a:id", "", ATOM).strip())
        if not match:      # error entries have no paper id
            continue
        papers.append(Paper(
            source="arxiv",
            source_id=match["id"],
            version=match["version"] or "",
            title=re.sub(r"\s+", " ", entry.findtext("a:title", "", ATOM)).strip(),
            abstract=re.sub(r"\s+", " ", entry.findtext("a:summary", "", ATOM)).strip(),
            url=f"https://arxiv.org/abs/{match['id']}",
            authors=[a.findtext("a:name", "", ATOM) for a in entry.findall("a:author", ATOM)],
            published=entry.findtext("a:published", "", ATOM)[:10],
        ))
    return papers


class ArxivSource:
    name = "arxiv"

    def __init__(self, client: RateLimitedClient, category: str = ""):
        self.client = client
        self.category = category

    def search(self, topic: str, max_results: int) -> list[Paper]:
        params = {"search_query": build_query(topic, self.category), "start": 0,
                  "max_results": max_results, "sortBy": "relevance"}
        response = self.client.get(API_URL, params=params)
        papers = parse_feed(response.content)
        logger.info("arXiv: %d papers for %r", len(papers), topic)
        return papers

    def fetch_full_text(self, paper: Paper) -> Paper:
        """Add full text from the HTML version; keep the abstract if there is none."""
        url = HTML_URL.format(id=paper.source_id, version=paper.version)
        try:
            response = self.client.get(url, allow_404=True)
        except requests.RequestException as e:
            logger.warning("arXiv HTML %s not fetched, using abstract: %s", url, e)
            return paper
        if response is None or "ltx_document" not in response.text:
            logger.info("arXiv %s has no HTML version, using abstract", paper.source_id)
            return paper
        _, _, text = arxiv_html_to_text(response.content)
        if len(text) > len(paper.abstract):
            paper.text, paper.text_kind = text, "full"
        return paper
