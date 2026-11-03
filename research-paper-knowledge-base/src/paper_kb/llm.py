"""Local LLM through the Ollama HTTP API."""

import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


class LLMError(RuntimeError):
    pass


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.txt").read_text()


class OllamaLLM:
    def __init__(self, model: str, url: str, num_ctx: int, timeout: int = 900):
        self.model = model
        self.url = url.rstrip("/")
        self.num_ctx = num_ctx
        self.timeout = timeout

    def check(self):
        """Fail early with a clear message if the server or the model is missing."""
        try:
            tags = requests.get(f"{self.url}/api/tags", timeout=10).json()
        except requests.RequestException as e:
            raise LLMError(f"Ollama is not reachable at {self.url}: start it with `ollama serve`") from e
        names = {m["name"] for m in tags.get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMError(f"Ollama model {self.model!r} not found: run `ollama pull {self.model}`")

    def complete(self, system: str, prompt: str, schema: dict | None = None) -> str:
        """One chat turn at temperature 0. With a JSON schema the reply is constrained to it."""
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }
        if schema is not None:
            body["format"] = schema
        try:
            response = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            raise LLMError(f"Ollama request failed: {e}") from e
        prompt_tokens = data.get("prompt_eval_count", 0)
        if prompt_tokens >= self.num_ctx:
            logger.warning("Prompt filled the whole context (%d tokens): input may have been cut",
                           prompt_tokens)
        return data["message"]["content"]

    def complete_json(self, system: str, prompt: str, schema: dict) -> dict:
        text = self.complete(system, prompt, schema)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM returned invalid JSON: {text[:200]!r}") from e
        missing = [key for key in schema.get("required", []) if key not in data]
        if missing:
            raise LLMError(f"LLM JSON lacks fields {missing}")
        return data
