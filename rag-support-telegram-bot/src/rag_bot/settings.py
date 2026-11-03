# Application settings: one flat JSON file. Secrets (API keys, bot token,
# DATABASE_URL) come from the environment (.env)
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

PATH_FIELDS = ("source_dir", "markdown_dir", "prompt_dir")


@dataclass
class Settings:
    # file locations (relative paths are resolved against the settings file);
    # the index and the conversations are in PostgreSQL (DATABASE_URL)
    markdown_dir: Path
    prompt_dir: Path  # system.txt and question.txt
    source_dir: Optional[Path] = None  # raw PDF/DOC/DOCX manuals

    # indexing and retrieval
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 8
    similarity_threshold: float = 0.0  # tune with rag_bot.evaluate

    # conversation
    history_minutes: int = 30
    max_questions_per_day: int = 30

    # models: "anthropic" / "openai", or "offline" for local runs without API keys
    llm_provider: str = "anthropic"
    llm_model: str = "claude-opus-5-5"
    max_tokens: int = 16000
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"


def load_settings(path, **overrides) -> Settings:
    """Read settings JSON; keyword overrides win over the file."""
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        values = json.load(f)
    values.update(overrides)

    known = {f.name for f in fields(Settings)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}")

    for name in PATH_FIELDS:
        if values.get(name) is not None:
            p = Path(values[name])
            values[name] = p if p.is_absolute() else (path.parent / p).resolve()
    return Settings(**values)
