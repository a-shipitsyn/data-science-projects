from concurrent.futures import ThreadPoolExecutor
from datetime import datetime



def test_quota_is_atomic_under_concurrency(log):
    with ThreadPoolExecutor(max_workers=20) as pool:
        ids = list(pool.map(lambda i: log.reserve(1, 1, f"q{i}", daily_limit=5), range(20)))
    assert sum(i is not None for i in ids) == 5


def test_quota_resets_next_day_and_cancel_frees_a_slot(log):
    day = datetime(2026, 1, 1, 12, 0)
    first = log.reserve(1, 1, "q", 2, now=day)
    log.reserve(1, 1, "q", 2, now=day)
    assert log.reserve(1, 1, "q", 2, now=day) is None
    assert log.reserve(2, 2, "other user", 2, now=day) is not None
    log.cancel(first)
    assert log.reserve(1, 1, "q", 2, now=day) is not None
    assert log.reserve(1, 1, "q", 2, now=datetime(2026, 1, 2, 0, 1)) is not None


def test_history_window_only_answered_messages(log):
    old = log.reserve(1, 1, "old", 10, now=datetime(2026, 1, 1, 9, 0))
    new = log.reserve(1, 1, 'multi\nline, "quoted"', 10, now=datetime(2026, 1, 1, 9, 50))
    log.reserve(1, 1, "pending", 10, now=datetime(2026, 1, 1, 9, 55))
    log.complete(old, "a-old")
    log.complete(new, "a-new")
    assert log.history(1, 30, now=datetime(2026, 1, 1, 10, 0)) == [('multi\nline, "quoted"', "a-new")]
    assert log.history(2, 30, now=datetime(2026, 1, 1, 10, 0)) == []


def test_empty_answers_are_not_replayed(log):
    log.complete(log.reserve(1, 1, "q", 10), "")
    assert log.history(1, 30) == []


def test_admin_replies_and_purge(log):
    a = log.reserve(1, 100, "q1", 10, now=datetime(2026, 1, 1))
    b = log.reserve(2, 200, "q2", 10, now=datetime(2026, 3, 1))
    assert log.queue_reply(a, "edited") and not log.queue_reply(999, "x")
    assert [r["id"] for r in log.pending_replies()] == [a]
    log.mark_sent(a)
    assert log.pending_replies() == []
    assert [r["id"] for r in log.recent()] == [b, a]
    assert log.purge(days=30, now=datetime(2026, 3, 2)) == 1
    assert [r["id"] for r in log.recent()] == [b]
