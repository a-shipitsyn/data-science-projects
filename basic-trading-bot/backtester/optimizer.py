"""Parameter sweep and walk-forward test, to choose parameters without fooling ourselves.

grid search    every (fast, slow) pair from the grid is backtested on a training period and
               ranked by Sharpe ratio
walk-forward   the data after an initial training period is cut into --folds test windows; for
               each window the grid search runs on all candles before it, and the best pair is
               traded on the window only (earlier candles just warm up the moving averages).
               The test windows are never used to choose parameters, so their combined result is
               an honest out-of-sample estimate; it is compared with the in-sample best and with
               buy-and-hold over the same windows.

Usage:
    python -m backtester.optimizer --data data/historical/BTC-USD_1h.csv --folds 4 \
        [--fast 5 10 20] [--slow 30 50 100]
"""
import argparse
from itertools import product

import pandas as pd

from backtester.backtest_engine import BacktestEngine, load_candles
from backtester.data_quality import report_gaps
from core.risk_manager import RiskManager
from core.strategies.sma_crossover import SMACrossover


def backtest(df: pd.DataFrame, fast: int, slow: int, args: argparse.Namespace,
             trade_from: pd.Timestamp | None = None) -> dict:
    risk = RiskManager(risk_per_trade_pct=args.risk, sl_pct=args.sl, tp_pct=args.tp, fee_rate=args.fee)
    strategy = SMACrossover(name=f"SMA_{fast}_{slow}", risk_manager=risk, fast_window=fast, slow_window=slow)
    engine = BacktestEngine(df, strategy, fee_rate=args.fee, slippage=args.slippage, trade_from=trade_from,
                            check_gaps=False)
    return engine.run(verbose=False)


def grid_search(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = [{"fast": f, "slow": s, **backtest(df, f, s, args)}
            for f, s in product(args.fast, args.slow) if f < s]
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def walk_forward(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    n_test = int(len(df) * (1 - args.train_frac) / args.folds)
    first = len(df) - args.folds * n_test
    rows = []
    for k in range(args.folds):
        start, end = first + k * n_test, first + (k + 1) * n_test
        best = grid_search(df.iloc[:start], args).iloc[0]
        test = backtest(df.iloc[:end], int(best["fast"]), int(best["slow"]), args, trade_from=df.index[start])
        rows.append({"fold": k + 1, "test_from": df.index[start], "test_to": df.index[end - 1],
                     "fast": int(best["fast"]), "slow": int(best["slow"]), "train_sharpe": best["sharpe"],
                     "test_sharpe": test["sharpe"], "test_return_pct": test["return_pct"],
                     "test_trades": test["trades"], "test_max_drawdown_pct": test["max_drawdown_pct"],
                     "buy_and_hold_pct": test["buy_and_hold_pct"]})
    return pd.DataFrame(rows)


def compounded(pct: pd.Series) -> float:
    return ((1 + pct / 100).prod() - 1) * 100


def main() -> None:
    ap = argparse.ArgumentParser(description="Grid search and walk-forward test of the SMA crossover")
    ap.add_argument("--data", default="data/historical/BTC-USD_1h.csv", help="CSV from data_downloader")
    ap.add_argument("--fast", type=int, nargs="+", default=[5, 10, 20], help="fast SMA windows")
    ap.add_argument("--slow", type=int, nargs="+", default=[30, 50, 100], help="slow SMA windows")
    ap.add_argument("--folds", type=int, default=4, help="walk-forward test windows")
    ap.add_argument("--train-frac", type=float, default=0.5, help="share of data before the first test window")
    ap.add_argument("--risk", type=float, default=0.02)
    ap.add_argument("--sl", type=float, default=0.05)
    ap.add_argument("--tp", type=float, default=0.10)
    ap.add_argument("--fee", type=float, default=0.005)
    ap.add_argument("--slippage", type=float, default=0.0)
    args = ap.parse_args()

    df = load_candles(args.data)
    report_gaps(df, args.data)

    grid = grid_search(df, args)
    print("Grid search on all data (in-sample, optimistic):")
    print(grid.head(5).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    wf = walk_forward(df, args)
    print(f"\nWalk-forward, {args.folds} out-of-sample windows:")
    print(wf.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nOut-of-sample return, compounded: {compounded(wf['test_return_pct']):.2f}% "
          f"(buy & hold over the same windows: {compounded(wf['buy_and_hold_pct']):.2f}%; "
          f"in-sample best on all data: {grid['return_pct'].iloc[0]:.2f}%)")


if __name__ == "__main__":
    main()
