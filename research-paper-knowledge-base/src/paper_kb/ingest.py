"""Topic ingestion: search sources, fetch full text, split, store. Incremental and idempotent."""

import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from .paper import Paper
from .store import KnowledgeBase

logger = logging.getLogger(__name__)


MIN_CHUNK_CHARS = 200


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    # paragraph breaks first, then lines, sentences, clauses, words;
    # a separator stays at the end of the piece it closes ("... end." not ". Start ...")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, keep_separator="end",
        separators=["\n\n", "\n", ". ", ", ", " ", ""])


def split_text(splitter, text: str, min_chars: int = MIN_CHUNK_CHARS) -> list[str]:
    """Split, then merge pieces shorter than min_chars (lone headings, tails) into the next one."""
    chunks, carry = [], ""
    for piece in splitter.split_text(text):
        piece = piece.strip()
        if not piece:
            continue
        piece = f"{carry}\n{piece}" if carry else piece
        carry = ""
        if len(piece) < min_chars:
            carry = piece
        else:
            chunks.append(piece)
    if carry:
        if chunks:
            chunks[-1] = f"{chunks[-1]}\n{carry}"
        else:
            chunks.append(carry)
    return chunks


@dataclass
class IngestReport:
    topic: str
    found: int = 0
    new: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.topic!r}: found {self.found}, new {len(self.new)}, updated {len(self.updated)}, "
                f"unchanged {len(self.unchanged)}, failed {len(self.failed)}")


def store_paper(kb: KnowledgeBase, paper: Paper, splitter) -> int:
    chunks = split_text(splitter, paper.body())
    kb.save_paper(paper, chunks)
    return len(chunks)


def ingest_topic(kb: KnowledgeBase, sources: dict, topic: str, max_results: int,
                 splitter, force: bool = False, category: str = "") -> IngestReport:
    """Search every source for the topic and add papers that are new or have a new version.

    A paper already stored with the same version is only linked to the topic
    (no download). `force` re-downloads and re-indexes every paper found.
    """
    kb.add_topic(topic, list(sources), max_results, category)
    report = IngestReport(topic)
    seen = set()
    for name, source in sources.items():
        try:
            papers = source.search(topic, max_results)
        except Exception as e:  # noqa: BLE001 - one source down must not stop the others
            logger.error("Search in %s failed for %r: %s", name, topic, e)
            continue
        report.found += len(papers)

        for paper in papers:
            if paper.paper_id in seen:
                continue
            seen.add(paper.paper_id)
            stored = kb.paper(paper.paper_id)
            if stored and stored["version"] == paper.version and not force:
                kb.link_topic(paper.paper_id, topic)
                report.unchanged.append(paper.paper_id)
                continue
            try:
                paper = source.fetch_full_text(paper)
                n_chunks = store_paper(kb, paper, splitter)
            except Exception as e:  # noqa: BLE001 - one bad paper must not stop the topic
                logger.error("Paper %s not stored: %s", paper.paper_id, e)
                report.failed.append(paper.paper_id)
                continue
            kb.link_topic(paper.paper_id, topic)
            (report.updated if stored else report.new).append(paper.paper_id)
            logger.info("Stored %s (%s text, %d chunks): %s", paper.paper_id, paper.text_kind,
                        n_chunks, paper.title[:80])
    kb.mark_refreshed(topic)
    logger.info("Ingest %s", report.summary())
    return report


def refresh(kb: KnowledgeBase, make_sources, splitter, topic: str | None = None) -> list[IngestReport]:
    """Re-run the saved search of every topic (or one) to pick up new papers and versions."""
    topics = [t for t in kb.topics() if topic is None or t["name"] == topic]
    if topic and not topics:
        raise ValueError(f"No saved topic {topic!r}")
    return [ingest_topic(kb, make_sources(t["sources"], t["category"]), t["name"], t["max_results"],
                         splitter, category=t["category"]) for t in topics]
