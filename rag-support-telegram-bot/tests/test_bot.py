import asyncio
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram import Chat, Message, Update, User

from rag_bot.bot import ERROR_MESSAGE, WAIT_MESSAGE, build_application, make_question_handler, \
    split_message, utf16_len


def fake_update(text, user_id=11, chat_id=22):
    return SimpleNamespace(
        message=SimpleNamespace(text=text, reply_text=AsyncMock()),
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=chat_id),
    )


class StubAssistant:
    def __init__(self, reply=None):
        self.reply, self.calls = reply, []

    def answer(self, question, user_id=None, chat_id=None):
        self.calls.append((question, user_id, chat_id))
        if self.reply is None:
            raise RuntimeError("secret internal detail")
        return self.reply


def test_split_message():
    text = ("a" * 30 + "\n") * 10
    parts = split_message(text, limit=100)
    assert all(len(p) <= 100 for p in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")
    assert split_message("x" * 250, limit=100) == ["x" * 100, "x" * 100, "x" * 50]
    assert split_message("") == []


def test_handler_answers_and_splits():
    assistant = StubAssistant("line\n" * 2000)
    handler = make_question_handler(assistant)
    update, context = fake_update("question?"), SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    asyncio.run(handler(update, context))

    update.message.reply_text.assert_awaited_once_with(WAIT_MESSAGE)
    calls = context.bot.send_message.await_args_list
    assert len(calls) == 3 and all(c.kwargs["chat_id"] == 22 for c in calls)
    assert assistant.calls == [("question?", 11, 22)]


def test_handler_hides_internal_errors():
    handler = make_question_handler(StubAssistant(None))
    update, context = fake_update("q"), SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
    asyncio.run(handler(update, context))
    assert update.message.reply_text.await_args_list[-1].args == (ERROR_MESSAGE,)
    context.bot.send_message.assert_not_awaited()


def test_split_message_counts_utf16_units():
    text = "😀" * 3000  # 6000 UTF-16 units
    parts = split_message(text, limit=4096)
    assert [utf16_len(p) for p in parts] == [4096, 1904]
    assert "".join(parts) == text


def telegram_update(**kind):
    message = Message(5, datetime.now(), Chat(2, "private"), from_user=User(1, "a", False),
                      text="How do I reset the M7?")
    return Update(10, **{kind["kind"]: message})


class SlowAssistant:
    """Blocks in the worker thread like a real LLM call and records overlap."""
    def __init__(self, seconds=0.2):
        self.seconds, self.active, self.max_active, self.order = seconds, 0, 0, []
        self.lock = threading.Lock()

    def answer(self, question, user_id=None, chat_id=None):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.seconds)
        with self.lock:
            self.active -= 1
            self.order.append(question)
        return question


def run_concurrently(handler, updates):
    context = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))

    async def main():
        start = time.perf_counter()
        await asyncio.gather(*(handler(u, context) for u in updates))
        return time.perf_counter() - start
    return asyncio.run(main()), context


def test_different_users_are_answered_in_parallel():
    assistant = SlowAssistant()
    elapsed, _ = run_concurrently(make_question_handler(assistant),
                                  [fake_update(f"q{i}", user_id=i, chat_id=i) for i in range(4)])
    assert assistant.max_active == 4 and elapsed < 0.5  # not 4 x 0.2 s


def test_one_users_questions_are_answered_in_order():
    assistant = SlowAssistant(0.05)
    _, context = run_concurrently(make_question_handler(assistant),
                                  [fake_update(f"q{i}", user_id=1, chat_id=1) for i in range(3)])
    assert assistant.max_active == 1 and assistant.order == ["q0", "q1", "q2"]
    assert [c.kwargs["text"] for c in context.bot.send_message.await_args_list] == ["q0", "q1", "q2"]


def test_parallel_answers_are_capped():
    assistant = SlowAssistant(0.05)
    run_concurrently(make_question_handler(assistant, max_parallel=2),
                     [fake_update(f"q{i}", user_id=i, chat_id=i) for i in range(6)])
    assert assistant.max_active == 2


def test_application_processes_updates_concurrently():
    app = build_application("123456:TEST-TOKEN-NOT-REAL", StubAssistant("ok"))  # never started
    assert app.update_processor.max_concurrent_updates > 1
    assert app.error_handlers


def test_application_answers_new_messages_only():
    app = build_application("123456:TEST-TOKEN-NOT-REAL", StubAssistant("ok"))  # never started
    start_handler, question_handler = app.handlers[0]
    assert question_handler.check_update(telegram_update(kind="message"))
    assert not question_handler.check_update(telegram_update(kind="edited_message"))
    assert not question_handler.check_update(telegram_update(kind="channel_post"))
