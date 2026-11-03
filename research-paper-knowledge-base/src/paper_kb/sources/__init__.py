"""Article sources. Each has search(topic, max_results) and fetch_full_text(paper)."""

from ..settings import Settings
from .arxiv import ArxivSource
from .http import RateLimitedClient


def make_sources(settings: Settings, names=None, category: str = "") -> dict:
    """Build the named sources; category narrows arXiv to a subject class such as q-fin.TR."""
    names = names or settings.sources
    makers = {
        "arxiv": lambda: ArxivSource(RateLimitedClient(
            settings.request_delay_arxiv, settings.http_timeout, settings.user_agent), category),
    }
    unknown = set(names) - set(makers)
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}; use {sorted(makers)}")
    return {name: makers[name]() for name in names}
