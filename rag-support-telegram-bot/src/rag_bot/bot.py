# Telegram bot: answers text messages with the support assistant
import argparse
import asyncio
import logging
import os
from collections import defaultdict
from typing import Dict, List

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from rag_bot.answer import SupportAssistant

logger = logging.getLogger(__name__)
TELEGRAM_LIMIT = 4096
GREETING = "Hello! Ask me any question about our equipment."
WAIT_MESSAGE = "Processing your question, please wait..."
ERROR_MESSAGE = "Sorry, something went wrong. Please try again later."
MAX_PARALLEL_ANSWERS = 8  # worker threads busy with retrieval + LLM; below the DB pool size


def utf16_len(text: str) -> int:
    """Telegram measures message length in UTF-16 code units (an emoji counts as 2)."""
    return len(text.encode("utf-16-le")) // 2


def split_message(text: str, limit: int = TELEGRAM_LIMIT) -> List[str]:
    """Split long text into Telegram-sized parts, preferring line breaks."""
    parts = []
    while utf16_len(text) > limit:
        end, size = 0, 0  # longest prefix that fits
        for ch in text:
            size += 2 if ord(ch) > 0xFFFF else 1
            if size > limit:
                break
            end += 1
        cut = text.rfind("\n", 0, end)
        cut = cut if cut > 0 else end
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts + [text] if text else parts


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(GREETING)


def make_question_handler(assistant: SupportAssistant, max_parallel: int = MAX_PARALLEL_ANSWERS):
    # Different users are answered in parallel (the application must run with
    # concurrent updates); one user's questions are answered one by one, so the
    # answers keep their order and each follow-up sees the previous answer.
    user_locks: Dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
    workers = asyncio.Semaphore(max_parallel)

    async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
        question = update.message.text
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        await update.message.reply_text(WAIT_MESSAGE)
        async with user_locks[user_id]:
            try:
                # quota, retrieval and the LLM call are blocking: run them in a
                # worker thread to keep the event loop free
                async with workers:
                    answer = await asyncio.to_thread(assistant.answer, question, user_id, chat_id)
            except Exception:
                logger.exception("Failed to answer user %s", user_id)
                await update.message.reply_text(ERROR_MESSAGE)
                return
            for part in split_message(answer):
                await context.bot.send_message(chat_id=chat_id, text=part)
    return handle_question


async def log_error(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Error while handling an update", exc_info=context.error)


def build_application(token: str, assistant: SupportAssistant):
    # by default python-telegram-bot handles one update at a time
    app = ApplicationBuilder().token(token).concurrent_updates(True).build()
    app.add_handler(CommandHandler("start", start))
    # new messages only: edited messages and channel posts have no update.message
    app.add_handler(MessageHandler(filters.UpdateType.MESSAGE & filters.TEXT & ~filters.COMMAND,
                                   make_question_handler(assistant)))
    app.add_error_handler(log_error)  # e.g. a failed send_message
    return app


def main():
    ap = argparse.ArgumentParser(description="Run the Telegram support bot")
    ap.add_argument("--settings", required=True)
    args = ap.parse_args()

    from dotenv import load_dotenv
    from rag_bot.answer import build_assistant
    from rag_bot.db import connect
    from rag_bot.settings import load_settings
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    with connect(os.environ["DATABASE_URL"]) as pool:
        assistant = build_assistant(load_settings(args.settings), pool)
        if not assistant.kb.store.document_hashes():
            raise SystemExit("The index is empty: run rag_bot.indexing first")
        # re-indexing while the bot runs is fine: searches read the database directly
        build_application(os.environ["TELEGRAM_BOT_TOKEN"], assistant).run_polling()


if __name__ == "__main__":
    main()
