import numpy as np
import pandas as pd
import pytest
from conftest import candles

from backtester.backtest_engine import BacktestEngine
from backtester.data_quality import find_gaps
from core.risk_manager import RiskManager
from core.strategies.sma_crossover import SMACrossover


def engine(df, fee=0.005, slippage=0.0, sl=0.05, tp=0.10, **kw):
    strat = SMACrossover("t", RiskManager(0.02, sl, tp, fee), fast_window=5, slow_window=20)
    return BacktestEngine(df, strat, fee_rate=fee, slippage=slippage, check_gaps=False, **kw)


def test_pnl_includes_both_fees(wave):
    e = engine(wave)
    m = e.run(verbose=False)
    assert m["trades"] > 2
    assert e.capital == pytest.approx(e.initial_capital + sum(t["pnl_usd"] for t in e.trade_history))
    # without fees and with the same prices every trade earns more
    e0 = engine(wave, fee=0.0)
    e0.run(verbose=False)
    assert e0.capital > e.capital


def test_stop_loss_gap_fills_at_open():
    # falling then rising prices: find the candle with the golden cross
    df = candles(list(np.linspace(120, 90, 30)) + list(np.linspace(91, 100, 10)))
    strat = engine(df).strategy
    i_buy = next(i for i in range(strat.warmup, len(df))
                 if strat.generate_raw_signal(df.iloc[:i + 1]).direction == "BUY")
    entry = df["close"].iloc[i_buy]

    # the next candle opens 20 % lower, far below the 5 % stop-loss
    gap = df.iloc[:i_buy + 1].copy()
    gap.loc[gap.index[-1] + pd.Timedelta(hours=1)] = [entry * 0.80, entry * 0.81, entry * 0.79, entry * 0.80, 1.0]
    e = engine(gap)
    e.run(verbose=False)
    trade = e.trade_history[0]
    assert trade["reason"] == "Stop-Loss Hit"
    assert trade["exit_price"] == pytest.approx(entry * 0.80)   # filled at the open, not at the stop level


def test_slippage_costs_money(wave):
    a, b = engine(wave), engine(wave, slippage=0.001)
    a.run(verbose=False)
    b.run(verbose=False)
    assert b.capital < a.capital


def test_metrics_and_trade_from(wave):
    start = wave.index[300]
    e = engine(wave, trade_from=start)
    m = e.run(verbose=False)
    assert e.equity.index[0] == start and len(e.equity) == 300
    assert m["max_drawdown_pct"] <= 0 and np.isfinite(m["sharpe"])
    assert all(t["exit_time"] >= start for t in e.trade_history)


def test_no_trades_gives_zero_sharpe():
    m = engine(candles(np.full(100, 100.0))).run(verbose=False)
    assert m["trades"] == 0 and m["sharpe"] == 0.0 and m["return_pct"] == 0.0


def test_find_gaps():
    df = candles(np.arange(10.0) + 100).drop(index=pd.date_range("2026-01-01 03:00", periods=2, freq="1h"))
    gaps = find_gaps(df)
    assert len(gaps) == 1 and gaps["missing"].iloc[0] == 2
    assert find_gaps(candles(np.arange(10.0) + 100)).empty


def test_kill_switch_stops_trading():
    # buy after a dip, then a slow decline: the 3 % drawdown limit trips before the 5 % stop-loss,
    # and the golden cross of the later rally must not open a new trade
    close = (list(np.linspace(120, 90, 30)) + list(np.linspace(91, 100, 10)) + list(np.linspace(100, 80, 40))
             + list(np.linspace(80, 110, 40)))
    strat = SMACrossover("t", RiskManager(0.5, 0.05, 0.10, 0.005, max_drawdown_pct=0.03), fast_window=5,
                         slow_window=20)
    e = BacktestEngine(candles(close), strat, check_gaps=False)
    e.run(verbose=False)
    assert e.kill_switch and e.kill_switch.startswith("Kill Switch: drawdown")
    assert [t["reason"] for t in e.trade_history] == ["Kill Switch: drawdown 3% from peak"]
