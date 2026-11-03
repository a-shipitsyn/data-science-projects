import json

import pytest

from rag_bot.settings import load_settings


def test_relative_paths_resolve_against_settings_file(sample_dir):
    s = load_settings(sample_dir / "settings.json")
    assert s.markdown_dir == (sample_dir / "markdown").resolve()
    assert (s.prompt_dir / "system.txt").exists()
    assert s.llm_provider == "offline"


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"markdown_dir": "m", "prompt_dir": "p",
                                "use_router": True}))
    with pytest.raises(ValueError, match="use_router"):
        load_settings(path)


def test_production_settings_template_loads(sample_dir):
    s = load_settings(sample_dir.parent / "config" / "settings.json")
    assert s.llm_provider == "anthropic" and s.embedding_provider == "openai"
