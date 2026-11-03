# Answer a support question: quota, history, retrieval, LLM call, sources
import argparse
from typing import List, Optional, Tuple

from rag_bot.history import ConversationLog
from rag_bot.providers import LLM, Messages
from rag_bot.retrieval import KnowledgeBase, format_hits, source_files
from rag_bot.settings import Settings
from rag_bot.store import DocumentStore, Hit

QUOTA_MESSAGE = ("You have reached today's question limit. "
                 "Please contact our support team by phone.")


def format_reply(text: str, sources: List[str]) -> str:
    """Tidy LLM output for a chat message and list the documents it was based on."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        lines.append("  " + line if line.startswith(("-", "•")) else line)
    reply = "\n".join(lines)
    return f"{reply}\n\nSources: {', '.join(sources)}" if sources else reply


class SupportAssistant:
    def __init__(self, settings: Settings, kb: KnowledgeBase, llm: LLM,
                 log: Optional[ConversationLog] = None):
        self.s = settings
        self.kb = kb
        self.llm = llm
        self.log = log
        self.system_prompt = (settings.prompt_dir / "system.txt").read_text(encoding="utf-8")
        self.question_template = (settings.prompt_dir / "question.txt").read_text(encoding="utf-8")

    def build_messages(self, question: str,
                       history: List[Tuple[str, str]] = ()) -> Tuple[Messages, List[Hit]]:
        """Earlier turns as real messages, retrieved documentation in the last one."""
        messages = []
        for q, a in history:
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        hits = self.kb.search(question)
        messages.append({"role": "user", "content": self.question_template.format(
            documentation=format_hits(hits), question=question)})
        return messages, hits

    def answer(self, question: str, user_id=None, chat_id=None) -> str:
        message_id, history = None, []
        if self.log is not None and user_id is not None:
            message_id = self.log.reserve(user_id, user_id if chat_id is None else chat_id,
                                          question, self.s.max_questions_per_day)
            if message_id is None:
                return QUOTA_MESSAGE
            history = self.log.history(user_id, self.s.history_minutes)
        try:
            messages, hits = self.build_messages(question, history)
            reply = format_reply(self.llm.complete(self.system_prompt, messages), source_files(hits))
        except Exception:
            if message_id is not None:
                self.log.cancel(message_id)  # a failed answer does not use up the quota
            raise
        if message_id is not None:
            self.log.complete(message_id, reply)
        return reply


def build_assistant(settings: Settings, pool, with_log: bool = True) -> SupportAssistant:
    """Assistant over the index in `pool`; without the log there is no history or quota."""
    from rag_bot.providers import configure_embeddings, create_llm
    configure_embeddings(settings)
    log = ConversationLog(pool) if with_log else None
    return SupportAssistant(settings, KnowledgeBase(settings, DocumentStore(pool)),
                            create_llm(settings), log)


def main():
    ap = argparse.ArgumentParser(description="Ask the assistant one question")
    ap.add_argument("--settings", required=True)
    ap.add_argument("question")
    args = ap.parse_args()

    import os
    from dotenv import load_dotenv
    from rag_bot.db import connect
    from rag_bot.settings import load_settings
    load_dotenv()
    with connect(os.environ["DATABASE_URL"]) as pool:
        assistant = build_assistant(load_settings(args.settings), pool, with_log=False)
        print(assistant.answer(args.question))


if __name__ == "__main__":
    main()
