import pandas as pd
import pytest
from conftest import candles

from core.models import Signal
from core.risk_manager import RiskManager
from core.strategies.sma_crossover import SMACrossover


def signal(direction, price=100.0):
    return Signal(timestamp=pd.Timestamp("2026-01-01"), asset="BTC-USD", direction=direction, entry_price=price)


def test_buy_sized_by_risk():
    s = RiskManager(risk_per_trade_pct=0.02, sl_pct=0.05, tp_pct=0.10).process_signal(signal("BUY"), 10000)
    # losing 5 % of the position must cost 2 % of capital: 4000 USD position
    assert s.status == "VALIDATED"
    assert s.size == pytest.approx(40.0)
    assert (s.stop_loss, s.take_profit) == (pytest.approx(95.0), pytest.approx(110.0))


def test_buy_capped_so_that_fee_fits():
    s = RiskManager(risk_per_trade_pct=0.10, sl_pct=0.05, fee_rate=0.005).process_signal(signal("BUY"), 10000)
    assert s.size * 100 * 1.005 <= 10000


def test_long_only_rules():
    rm = RiskManager()
    assert rm.process_signal(signal("BUY"), 10000, position_size=1.0).status == "REJECTED"
    assert rm.process_signal(signal("SELL"), 10000, position_size=0.0).status == "REJECTED"
    sell = rm.process_signal(signal("SELL"), 10000, position_size=2.5)
    assert sell.status == "VALIDATED" and sell.size == 2.5


def test_sma_crossover_signals():
    strat = SMACrossover("t", RiskManager(), fast_window=2, slow_window=4)
    assert strat.warmup == 5
    assert strat.generate_raw_signal(candles([5, 4, 3])).direction == "HOLD"           # not enough data
    assert strat.generate_raw_signal(candles([5, 4, 3, 2, 1, 6])).direction == "BUY"   # golden cross
    assert strat.generate_raw_signal(candles([1, 2, 3, 4, 5, 0])).direction == "SELL"  # death cross


def test_kill_switch_daily_loss_resets_next_day():
    rm = RiskManager(max_daily_loss_pct=0.05)
    assert rm.update_equity(pd.Timestamp("2026-01-01 00:00"), 10000) is None
    assert rm.update_equity(pd.Timestamp("2026-01-01 12:00"), 9400).startswith("Kill Switch: daily loss")
    assert rm.halted and rm.process_signal(signal("BUY"), 9400).status == "REJECTED"
    assert rm.process_signal(signal("SELL"), 9400, position_size=1.0).status == "VALIDATED"   # exits still allowed
    rm.update_equity(pd.Timestamp("2026-01-02 00:00"), 9400)
    assert not rm.halted and rm.process_signal(signal("BUY"), 9400).status == "VALIDATED"


def test_kill_switch_drawdown_is_permanent_and_restorable():
    rm = RiskManager(max_drawdown_pct=0.10)
    rm.update_equity(pd.Timestamp("2026-01-01"), 10000)
    rm.update_equity(pd.Timestamp("2026-01-02"), 12000)
    assert rm.update_equity(pd.Timestamp("2026-01-03"), 10700).startswith("Kill Switch: drawdown")
    rm.update_equity(pd.Timestamp("2026-01-10"), 11000)
    assert rm.halted
    restored = RiskManager(max_drawdown_pct=0.10)
    restored.restore(rm.state())
    assert restored.halted and restored.peak_equity == 12000


def test_kill_switch_off_by_default():
    rm = RiskManager()
    rm.update_equity(pd.Timestamp("2026-01-01"), 10000)
    assert rm.update_equity(pd.Timestamp("2026-01-01 01:00"), 1000) is None and not rm.halted
