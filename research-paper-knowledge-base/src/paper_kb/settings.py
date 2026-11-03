"""Settings: config/settings.json, overridden by environment variables KB_<NAME>."""

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "settings.json"


@dataclass(frozen=True)
class Settings:
    data_dir: str = "kb_data"
    sources: tuple = ("arxiv",)
    max_results_per_source: int = 10

    chunk_size: int = 1000
    chunk_overlap: int = 100
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    retrieval_k: int = 10            # chunks per query; 10 scored best in the sweep (see README)
    max_context_chars: int = 16000
    min_sub_questions: int = 2
    max_sub_questions: int = 5

    ollama_model: str = "llama3.2"
    ollama_judge_model: str = ""           # evaluation judge; empty means use ollama_model
    ollama_judge_url: str = ""             # a second Ollama instance for the judge; empty means ollama_url
    ollama_url: str = "http://localhost:11434"
    ollama_num_ctx: int = 8192
    notes_max_chars: int = 20000           # longer paper text is cut before note writing (logged)

    request_delay_arxiv: float = 3.0       # arXiv asks for one request every 3 s
    http_timeout: int = 30
    user_agent: str = "research-paper-knowledge-base/0.1"

    @property
    def judge_model(self) -> str:
        """The model that grades answers: a second Ollama model, or the generation model itself."""
        return self.ollama_judge_model or self.ollama_model

    @property
    def judge_url(self) -> str:
        """Where the judge runs: a second Ollama instance, or the same one."""
        return self.ollama_judge_url or self.ollama_url

    @property
    def catalogue_path(self) -> Path:
        return Path(self.data_dir) / "catalogue.sqlite3"

    @property
    def vector_dir(self) -> Path:
        return Path(self.data_dir) / "vectors"


def _convert(value: str, current):
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    if isinstance(current, tuple):
        return tuple(v.strip() for v in value.split(",") if v.strip())
    return value


def load_settings(path: str | Path | None = None, **overrides) -> Settings:
    """Read the JSON config, apply KB_* environment variables, then explicit overrides."""
    path = Path(path or os.getenv("KB_CONFIG", DEFAULT_CONFIG))
    values = {}
    if path.exists():
        values = json.loads(path.read_text())
    known = {f.name: f for f in fields(Settings)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"Unknown settings in {path}: {sorted(unknown)}")
    if "sources" in values:
        values["sources"] = tuple(values["sources"])

    base = Settings(**values)
    for name in known:
        env = os.getenv(f"KB_{name.upper()}")
        if env is not None:
            values[name] = _convert(env, getattr(base, name))
    values.update({k: v for k, v in overrides.items() if v is not None})

    settings = Settings(**values)
    if not 0 <= settings.chunk_overlap < settings.chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    return settings
