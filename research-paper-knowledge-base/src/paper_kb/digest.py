"""Export a topic as a markdown digest: paper table, then one section per paper with its note."""

from datetime import date

from .notes import FIELDS
from .store import KnowledgeBase


def _authors(authors: list[str], limit: int = 4) -> str:
    return ", ".join(authors[:limit]) + (" et al." if len(authors) > limit else "")


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _heading(topic: str) -> str:
    """Topic names keep the quotes that make them a phrase search; drop them for display."""
    return topic.strip('"')


def topic_digest(kb: KnowledgeBase, topic: str) -> str:
    papers = kb.papers(topic=topic)
    if not any(t["name"] == topic for t in kb.topics()):
        raise ValueError(f"No saved topic {topic!r}")
    with_notes = sum(kb.note(p["paper_id"]) is not None for p in papers)
    lines = [
        f"# {_heading(topic)}",
        "",
        (f"Knowledge-base digest, {date.today().isoformat()}: {len(papers)} papers, "
         f"{with_notes} with reading notes."),
        "",
        "| # | Paper | Source | Date | Text |",
        "|---|---|---|---|---|",
    ]
    for n, p in enumerate(papers, 1):
        lines.append(f"| {n} | [{_cell(p['title'])}](#paper-{n}) | [{p['paper_id']}]({p['url']}) | "
                     f"{p['published']} | {p['text_kind']} |")

    for n, p in enumerate(papers, 1):
        lines += ["", f'<a id="paper-{n}"></a>', f"## {n}. {p['title']}", "",
                  f"{_authors(p['authors'])} · {p['published']} · [{p['paper_id']}]({p['url']})"]
        others = [t for t in kb.paper_topics(p["paper_id"]) if t != topic]
        if others:
            lines.append(f"Also in: {', '.join(others)}")
        note = kb.note(p["paper_id"])
        lines.append("")
        if note:
            lines += [f"- **{f.capitalize()}:** {note[f]}" for f in FIELDS]
            lines.append(f"- **Keywords:** {', '.join(note['keywords'])}")
        else:
            lines += ["**Abstract.** " + p["abstract"]]
    return "\n".join(lines) + "\n"
