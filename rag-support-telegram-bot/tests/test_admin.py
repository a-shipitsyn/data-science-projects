import asyncio
from unittest.mock import AsyncMock

from telegram.error import RetryAfter

from rag_bot.admin import send_pending, send_text


def test_sends_queued_replies_and_keeps_failed_ones(log):
    ids = [log.reserve(i, 100 + i, f"q{i}", 10) for i in range(3)]
    log.queue_reply(ids[0], "a0")
    log.queue_reply(ids[2], "edited by admin")

    bot = AsyncMock()
    bot.send_message.side_effect = [None, RuntimeError("blocked")]  # second send fails
    assert asyncio.run(send_pending(log, bot, delay=0)) == 1

    sent = [c.kwargs for c in bot.send_message.await_args_list]
    assert sent == [{"chat_id": 100, "text": "a0"}, {"chat_id": 102, "text": "edited by admin"}]
    assert [r["id"] for r in log.pending_replies()] == [ids[2]]  # failed one stays queued


def test_bot_keeps_logging_while_admin_sends(log):
    log.queue_reply(log.reserve(1, 1, "q", 10), "reply")

    async def slow_send(**kwargs):
        log.reserve(2, 2, "arrived during send", 10)  # the bot writes meanwhile
    bot = AsyncMock()
    bot.send_message.side_effect = slow_send
    asyncio.run(send_pending(log, bot, delay=0))
    assert {r["question"] for r in log.recent()} == {"q", "arrived during send"}


def test_send_text_waits_out_flood_control_and_splits():
    bot = AsyncMock()
    bot.send_message.side_effect = [RetryAfter(0), None, None]
    asyncio.run(send_text(bot, 7, "x" * 5000, delay=0))
    texts = [c.kwargs["text"] for c in bot.send_message.await_args_list]
    assert [len(t) for t in texts] == [4096, 4096, 904]  # first part retried once


def test_send_text_gives_up_after_retries():
    bot = AsyncMock()
    bot.send_message.side_effect = RetryAfter(0)
    try:
        asyncio.run(send_text(bot, 7, "x", delay=0))
    except RetryAfter:
        pass
    else:
        raise AssertionError("expected RetryAfter")
