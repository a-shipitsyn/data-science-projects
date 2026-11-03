# Conversation log in PostgreSQL: dialogue history, daily quota, admin replies.
# The connection pool makes it safe to use from the bot's worker threads, and the
# admin CLI can work on the same database while the bot is running.
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from psycopg_pool import ConnectionPool

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id           BIGSERIAL PRIMARY KEY,
    created_at   TIMESTAMPTZ NOT NULL,
    user_id      BIGINT NOT NULL,
    chat_id      BIGINT NOT NULL,
    question     TEXT NOT NULL,
    answer       TEXT,            -- NULL while the answer is being generated
    admin_answer TEXT,            -- reply written by an administrator
    sent_at      TIMESTAMPTZ      -- when the admin reply was sent
);
CREATE INDEX IF NOT EXISTS messages_user_time ON messages (user_id, created_at);
CREATE INDEX IF NOT EXISTS messages_pending ON messages (id)
    WHERE admin_answer IS NOT NULL AND sent_at IS NULL;
"""


def _now(now: Optional[datetime] = None) -> datetime:
    """Timezone-aware time; naive datetimes are taken as local time."""
    return (now or datetime.now()).astimezone()


class ConversationLog:
    def __init__(self, pool: ConnectionPool):
        self.pool = pool  # from rag_bot.db.connect
        with self.pool.connection() as conn:
            conn.execute(SCHEMA)

    def reserve(self, user_id: int, chat_id: int, question: str, daily_limit: int,
                now: Optional[datetime] = None) -> Optional[int]:
        """Atomically check the daily quota and register the question.
        Returns the message id, or None if the user has reached the limit."""
        now = _now(now)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        with self.pool.connection() as conn:  # one transaction
            # per-user lock until commit: parallel questions of one user queue up here
            conn.execute("SELECT pg_advisory_xact_lock(%s)", (user_id,))
            count = conn.execute(
                "SELECT count(*) AS n FROM messages WHERE user_id = %s AND created_at >= %s",
                (user_id, day_start)).fetchone()["n"]
            if count >= daily_limit:
                return None
            return conn.execute(
                "INSERT INTO messages (created_at, user_id, chat_id, question) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (now, user_id, chat_id, question)).fetchone()["id"]

    def complete(self, message_id: int, answer: str) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE messages SET answer = %s WHERE id = %s", (answer, message_id))

    def cancel(self, message_id: int) -> None:
        """Drop a question that failed, so it does not count towards the quota."""
        with self.pool.connection() as conn:
            conn.execute("DELETE FROM messages WHERE id = %s", (message_id,))

    def history(self, user_id: int, minutes: int,
                now: Optional[datetime] = None) -> List[Tuple[str, str]]:
        """Answered (question, answer) pairs of the user in the last `minutes`."""
        cutoff = _now(now) - timedelta(minutes=minutes)
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT question, answer FROM messages WHERE user_id = %s "
                "AND created_at >= %s AND answer <> '' ORDER BY id",
                (user_id, cutoff)).fetchall()
        return [(r["question"], r["answer"]) for r in rows]

    # --- administration ---

    def recent(self, limit: int = 20) -> List[dict]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT * FROM messages ORDER BY id DESC LIMIT %s",
                                (limit,)).fetchall()

    def queue_reply(self, message_id: int, text: str) -> bool:
        with self.pool.connection() as conn:
            cur = conn.execute("UPDATE messages SET admin_answer = %s, sent_at = NULL WHERE id = %s",
                               (text, message_id))
            return cur.rowcount == 1

    def pending_replies(self) -> List[dict]:
        with self.pool.connection() as conn:
            return conn.execute("SELECT * FROM messages WHERE admin_answer IS NOT NULL "
                                "AND sent_at IS NULL ORDER BY id").fetchall()

    def mark_sent(self, message_id: int, now: Optional[datetime] = None) -> None:
        with self.pool.connection() as conn:
            conn.execute("UPDATE messages SET sent_at = %s WHERE id = %s", (_now(now), message_id))

    def purge(self, days: int, now: Optional[datetime] = None) -> int:
        """Delete conversations older than `days` (data retention)."""
        cutoff = _now(now) - timedelta(days=days)
        with self.pool.connection() as conn:
            return conn.execute("DELETE FROM messages WHERE created_at < %s", (cutoff,)).rowcount
