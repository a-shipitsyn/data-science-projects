"""The paper record shared by sources, the catalogue and the index."""

import hashlib
from dataclasses import dataclass, field


@dataclass
class Paper:
    source: str                  # "arxiv" (the paper_id namespace)
    source_id: str               # arXiv id without the version suffix
    title: str
    abstract: str
    url: str
    version: str = ""            # arXiv version, e.g. "v2"
    authors: list[str] = field(default_factory=list)
    published: str = ""          # ISO date, YYYY-MM-DD when known
    text: str = ""               # full text when available, paragraphs separated by blank lines
    text_kind: str = "abstract"  # "full" or "abstract"

    @property
    def paper_id(self) -> str:
        return f"{self.source}:{self.source_id}"

    @property
    def year(self) -> int:
        return int(self.published[:4]) if self.published[:4].isdigit() else 0

    def body(self) -> str:
        """Text to index: full text if fetched, otherwise title and abstract."""
        if self.text:
            return self.text
        return f"{self.title}\n\n{self.abstract}"

    def content_hash(self) -> str:
        return hashlib.sha256(f"{self.version}\n{self.body()}".encode()).hexdigest()
