import asyncio

import pandas as pd

from trading_bot.main_bot import LiveTradingBot
from trading_bot.notifier import Notifier


class FakeNotifier(Notifier):
    def __init__(self):
        super().__init__(token="", chat_id="")
        self.messages = []

    def send(self, text: str) -> None:
        self.messages.append(text)


def make_bot(tmp_path, **kw):
    bot = LiveTradingBot(fast_window=2, slow_window=4, log_path=str(tmp_path / "log.csv"),
                         state_path=str(tmp_path / "state.json"), notifier=FakeNotifier(), **kw)
    idx = pd.date_range("2026-01-01 00:00", periods=6, freq="1min")
    bot.candles = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                                "close": [105, 104, 103, 102, 101, 100.0], "volume": 0.0}, index=idx)
    return bot


async def feed(bot, minutes):
    t0 = pd.Timestamp("2026-01-01 00:06", tz="UTC")
    for m, prices in enumerate(minutes):
        for k, p in enumerate(prices):
            await bot.on_tick({"price": p, "time": (t0 + pd.Timedelta(minutes=m, seconds=5 * k)).isoformat()})


def test_one_log_entry_per_action_and_state_restored(tmp_path):
    bot = make_bot(tmp_path)
    # many ticks per minute: the golden cross must be logged once, on the closed candle
    asyncio.run(feed(bot, [[100, 101], [104, 106, 108], [109, 110, 111], [112]]))
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log["direction"]) == ["BUY"]
    assert bot.position_size > 0

    # restart: the paper position comes back from the state file
    restarted = make_bot(tmp_path)
    assert restarted.position_size == bot.position_size and restarted.capital == bot.capital

    # take-profit on a tick closes it, logged once
    asyncio.run(feed(restarted, [[200, 201, 202]]))
    log = pd.read_csv(tmp_path / "log.csv")
    assert list(log["direction"]) == ["BUY", "SELL"] and log["reason"].iloc[-1] == "Take-Profit Hit"
    assert restarted.position_size == 0


def test_flat_candles_for_minutes_without_ticks_and_volume(tmp_path):
    bot = make_bot(tmp_path)
    t0 = pd.Timestamp("2026-01-01 00:06", tz="UTC")
    for k, (p, size) in enumerate([(100, 0.5), (101, 0.25)]):
        asyncio.run(bot.on_tick({"price": p, "last_size": size, "time": (t0 + pd.Timedelta(seconds=k)).isoformat()}))
    assert bot.current["volume"] == 0.75
    bot.roll(pd.Timestamp("2026-01-01 00:10"))          # what the watchdog does when no ticks arrive
    closed = bot.candles.loc["2026-01-01 00:06":]
    assert list(closed.index.minute) == [6, 7, 8, 9]
    assert (closed["close"].iloc[1:] == 101).all() and (closed["volume"].iloc[1:] == 0).all()


def test_notifications_and_kill_switch(tmp_path):
    bot = make_bot(tmp_path, max_drawdown_pct=0.001)
    asyncio.run(feed(bot, [[100, 101], [104, 106, 108], [109, 110, 111], [112]]))
    assert any(m.startswith("BUY") for m in bot.notifier.messages)
    asyncio.run(feed(bot, [[], [], [], [], [50], [49]]))       # crash: kill switch closes the position
    assert bot.position_size == 0 and bot.risk_manager.halted
    assert any("Kill Switch" in m for m in bot.notifier.messages)
    assert make_bot(tmp_path, max_drawdown_pct=0.001).risk_manager.halted   # survives a restart


def test_notifier_disabled_without_credentials(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert not Notifier().enabled
