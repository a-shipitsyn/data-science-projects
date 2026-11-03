from types import SimpleNamespace

import pytest

from rag_bot.providers import AnthropicLLM, HashEmbedding


def fake_llm(content, stop_reason="end_turn"):
    llm = AnthropicLLM.__new__(AnthropicLLM)  # no real client, no API key
    llm.model, llm.max_tokens, llm.sent = "model", 100, []

    def create(**kwargs):
        llm.sent.append(kwargs)
        return SimpleNamespace(content=content, stop_reason=stop_reason)
    llm.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return llm


def test_anthropic_skips_thinking_blocks_and_sends_system_prompt():
    llm = fake_llm([SimpleNamespace(type="thinking", thinking=""),
                    SimpleNamespace(type="text", text=" Answer. ")])
    messages = [{"role": "user", "content": "q"}]
    assert llm.complete("rules", messages) == "Answer."
    assert llm.sent[0]["system"] == "rules" and llm.sent[0]["messages"] == messages


@pytest.mark.parametrize("content, stop_reason", [
    ([SimpleNamespace(type="thinking", thinking="")], "max_tokens"),
    ([SimpleNamespace(type="text", text="  ")], "end_turn"),
])
def test_anthropic_empty_reply_is_an_error(content, stop_reason):
    with pytest.raises(RuntimeError, match="Empty reply"):
        fake_llm(content, stop_reason).complete("rules", [{"role": "user", "content": "q"}])


def test_anthropic_refusal_is_an_error():
    with pytest.raises(RuntimeError, match="declined"):
        fake_llm([], "refusal").complete("rules", [{"role": "user", "content": "q"}])


def test_hash_embedding_is_deterministic_and_normalised():
    e = HashEmbedding()
    a, b = e.get_text_embedding("Reset the M7"), e.get_text_embedding("reset the m7")
    assert a == b and abs(sum(x * x for x in a) - 1) < 1e-9
