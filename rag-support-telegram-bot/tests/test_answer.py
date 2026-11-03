import pytest

from rag_bot.answer import QUOTA_MESSAGE, SupportAssistant, build_assistant, format_reply
from rag_bot.retrieval import KnowledgeBase
from rag_bot.store import DocumentStore


class RecordingLLM:
    def __init__(self, reply="ok", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def complete(self, system, messages):
        self.calls.append((system, messages))
        if self.fail:
            raise RuntimeError("API down")
        return self.reply


def make(settings, llm, log):
    return SupportAssistant(settings, KnowledgeBase(settings, DocumentStore(log.pool)), llm, log), log


def test_format_reply_indents_bullets_and_adds_sources():
    assert format_reply("  1\n- a\n12\n", []) == "1\n  - a\n12"
    assert format_reply("ok", ["a.md", "b.md"]) == "ok\n\nSources: a.md, b.md"


def test_system_prompt_history_turns_and_documentation(settings, indexed, log):
    llm = RecordingLLM()
    assistant, log = make(settings, llm, log)
    earlier = log.reserve(7, 7, "Earlier question about DX-10", 10)
    log.complete(earlier, "Earlier answer")

    assistant.answer("How do I change the unlock time on the DX-10?", user_id=7)
    system, messages = llm.calls[0]
    assert "technical support assistant" in system and "<documentation>" not in system
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[:2] == [{"role": "user", "content": "Earlier question about DX-10"},
                            {"role": "assistant", "content": "Earlier answer"}]
    last = messages[-1]["content"]
    assert "UNLOCK TIME trimmer" in last.split("<question>")[0]
    assert last.rstrip().endswith("How do I change the unlock time on the DX-10?\n</question>")


def test_answer_is_logged_with_sources(settings, indexed, log):
    assistant, log = make(settings, RecordingLLM("Turn the trimmer."), log)
    reply = assistant.answer("DX-10 unlock time trimmer", user_id=3, chat_id=30)
    assert reply.startswith("Turn the trimmer.\n\nSources: door-station-dx10.md")
    row = log.recent()[0]
    assert row["answer"] == reply and row["chat_id"] == 30


def test_quota_stops_before_llm_call(settings, indexed, log):
    llm = RecordingLLM()
    assistant, _ = make(settings, llm, log)
    for _ in range(settings.max_questions_per_day):
        assistant.answer("q", user_id=1)
    llm.calls.clear()
    assert assistant.answer("one more", user_id=1) == QUOTA_MESSAGE
    assert llm.calls == []
    assert assistant.answer("other user", user_id=2) != QUOTA_MESSAGE


def test_failed_answer_does_not_use_quota(settings, indexed, log):
    assistant, log = make(settings, RecordingLLM(fail=True), log)
    with pytest.raises(RuntimeError):
        assistant.answer("q", user_id=1)
    assert log.recent() == []


def test_offline_end_to_end(settings, indexed, log):
    assistant = build_assistant(settings, log.pool)
    reply = assistant.answer("What is the maximum cable length for the DX-10?", user_id=5)
    assert log.history(5, 30) == [("What is the maximum cable length for the DX-10?", reply)]
    assert "80 m" in reply and "Sources: door-station-dx10.md" in reply


def test_without_log_no_quota_or_history(settings, indexed):
    assistant = build_assistant(settings, indexed.pool, with_log=False)
    assert assistant.log is None
    assert "Sources:" in assistant.answer("AC2 master code", user_id=1)
