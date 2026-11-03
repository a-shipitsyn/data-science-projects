# Administration of the conversation log:
#   list   show recent questions and answers
#   reply  queue an edited answer for a message
#   send   send queued answers via Telegram
#   purge  delete conversations older than N days
# The database is given by DATABASE_URL.
import argparse
import asyncio
import os

from telegram.error import RetryAfter

from rag_bot.bot import split_message
from rag_bot.history import ConversationLog

SEND_DELAY = 0.05  # seconds between messages, below Telegram's ~30 messages/s limit
MAX_RETRIES = 3


async def send_text(bot, chat_id: int, text: str, delay: float = SEND_DELAY) -> None:
    """Send text in Telegram-sized parts, waiting out flood-control errors."""
    for part in split_message(text):
        for attempt in range(MAX_RETRIES + 1):
            try:
                await bot.send_message(chat_id=chat_id, text=part)
                break
            except RetryAfter as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = e.retry_after
                await asyncio.sleep(wait.total_seconds() if hasattr(wait, "total_seconds") else wait)
        await asyncio.sleep(delay)


async def send_pending(log: ConversationLog, bot, delay: float = SEND_DELAY) -> int:
    sent = 0
    for row in log.pending_replies():
        try:
            await send_text(bot, row["chat_id"], row["admin_answer"], delay)
        except Exception as e:
            print(f"Failed to send message {row['id']} to chat {row['chat_id']}: {e}")
            continue
        log.mark_sent(row["id"])
        sent += 1
    print(f"Sent {sent} message(s)")
    return sent


def run_command(args, log: ConversationLog) -> None:
    if args.command == "list":
        for r in log.recent(args.limit):
            print(f"#{r['id']}  {r['created_at']:%Y-%m-%d %H:%M}  user {r['user_id']}\n"
                  f"  Q: {r['question']}\n  A: {(r['answer'] or '...')[:200]}\n")
    elif args.command == "reply":
        print("Queued" if log.queue_reply(args.id, args.text) else f"No message #{args.id}")
    elif args.command == "send":
        from telegram import Bot

        async def run():
            async with Bot(os.environ["TELEGRAM_BOT_TOKEN"]) as bot:
                await send_pending(log, bot)
        asyncio.run(run())
    elif args.command == "purge":
        print(f"Deleted {log.purge(args.days)} message(s)")


def main():
    ap = argparse.ArgumentParser(description="Conversation log administration")
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("list")
    p.add_argument("--limit", type=int, default=20)
    p = sub.add_parser("reply")
    p.add_argument("id", type=int)
    p.add_argument("text")
    sub.add_parser("send")
    p = sub.add_parser("purge")
    p.add_argument("--days", type=int, required=True)
    args = ap.parse_args()

    from dotenv import load_dotenv
    from rag_bot.db import connect
    load_dotenv()
    with connect(os.environ["DATABASE_URL"]) as pool:
        run_command(args, ConversationLog(pool))


if __name__ == "__main__":
    main()
