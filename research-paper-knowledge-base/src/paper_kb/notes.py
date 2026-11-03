"""Per-paper structured notes written by the LLM, stored and indexed as searchable chunks."""

import logging

from .llm import LLMError, load_prompt
from .store import KnowledgeBase

logger = logging.getLogger(__name__)

FIELDS = ["problem", "method", "data", "findings", "limitations"]
NOTE_SCHEMA = {
    "type": "object",
    "properties": {**{f: {"type": "string"} for f in FIELDS},
                   "keywords": {"type": "array", "items": {"type": "string"}}},
    "required": FIELDS + ["keywords"],
}


def note_to_text(title: str, note: dict) -> str:
    lines = [f"Reading note: {title}"]
    lines += [f"{f.capitalize()}: {note[f]}" for f in FIELDS]
    lines.append("Keywords: " + ", ".join(note["keywords"]))
    return "\n".join(lines)


def write_note(llm, kb: KnowledgeBase, paper_id: str, max_chars: int, model_name: str) -> dict:
    paper = kb.paper(paper_id)
    if paper is None:
        raise ValueError(f"Unknown paper {paper_id}")
    text = kb.paper_text(paper_id) or f"{paper['title']}\n\n{paper['abstract']}"
    if len(text) > max_chars:
        logger.info("%s: text cut from %d to %d characters for the note (notes_max_chars)",
                    paper_id, len(text), max_chars)
        text = text[:max_chars]
    prompt = f"TITLE: {paper['title']}\n\nABSTRACT: {paper['abstract']}\n\nTEXT:\n{text}"
    raw = llm.complete_json(load_prompt("note"), prompt, NOTE_SCHEMA)
    note = {f: str(raw[f]).strip() for f in FIELDS}
    note["keywords"] = [str(k).strip() for k in raw["keywords"] if str(k).strip()][:8]
    kb.save_note(paper_id, model_name, note, note_to_text(paper["title"], note))
    return note


def write_notes(llm, kb: KnowledgeBase, max_chars: int, model_name: str, topic: str | None = None,
                paper_ids: list[str] | None = None, force: bool = False) -> dict:
    """Write notes for papers that have none (all of them with force). Failures are logged and skipped."""
    ids = paper_ids or [p["paper_id"] for p in kb.papers(topic=topic)]
    done, skipped, failed = [], [], []
    for paper_id in ids:
        if not force and kb.note(paper_id) is not None:
            skipped.append(paper_id)
            continue
        try:
            write_note(llm, kb, paper_id, max_chars, model_name)
            done.append(paper_id)
            logger.info("Note written for %s (%d/%d)", paper_id, len(done) + len(failed), len(ids))
        except (LLMError, ValueError) as e:
            logger.error("Note for %s failed: %s", paper_id, e)
            failed.append(paper_id)
    return {"written": done, "skipped": skipped, "failed": failed}
