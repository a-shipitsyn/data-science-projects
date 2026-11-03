import argparse
import math

import pandas as pd

from backtester.data_quality import report_gaps
from core.base_strategy import BaseStrategy
from core.models import Signal
from core.risk_manager import RiskManager
from core.strategies.sma_crossover import SMACrossover


def load_candles(path: str) -> pd.DataFrame:
    """OHLCV CSV from data_downloader, sorted oldest to newest."""
    return pd.read_csv(path, index_col='timestamp', parse_dates=True).sort_index()


class BacktestEngine:
    def __init__(self, data: str | pd.DataFrame, strategy: BaseStrategy, initial_capital: float = 10000.0,
                 fee_rate: float = 0.005, slippage: float = 0.0, trade_from: pd.Timestamp | None = None,
                 check_gaps: bool = True):
        """
        Long-only, event-driven backtest on closed candles.
        A signal on a candle is executed at that candle's close. Stop-loss and take-profit are
        checked from the next candle on; if the candle opens beyond the level (a gap), the fill is
        at the open, otherwise at the level; if both are hit within one candle, the stop-loss is
        assumed to come first (conservative).

        :param data: path to an OHLCV CSV or a DataFrame with a DatetimeIndex
        :param fee_rate: 0.005 represents a 0.5% trading fee per side (Coinbase Advanced tier)
        :param slippage: adverse price move per fill, e.g. 0.0005 = 0.05% worse than the price
        :param trade_from: trade and measure only from this time on; earlier candles only warm up
            the strategy (used for out-of-sample tests)
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage

        # Portfolio State
        self.position_size = 0.0      # Amount of crypto held
        self.position_entry_price = 0.0
        self.position_cost = 0.0      # Entry value plus entry fee
        self.current_sl = None
        self.current_tp = None

        self.trade_history = []       # Store completed trades
        self.equity = pd.Series(dtype=float)
        self.kill_switch = None       # reason, if the risk manager's kill switch tripped

        # Load data, ensure it's sorted oldest to newest
        self.df = load_candles(data) if isinstance(data, str) else data.sort_index()
        if check_gaps:
            report_gaps(self.df, "backtest data")
        self.start = 0 if trade_from is None else int(self.df.index.searchsorted(trade_from))

    def run(self, verbose: bool = True) -> dict:
        if verbose:
            print(f"Starting backtest on {len(self.df) - self.start} candles with ${self.initial_capital:,.2f}")

        warmup = self.strategy.warmup
        equity = []
        for i in range(self.start, len(self.df)):
            candle = self.df.iloc[i]

            # 1. Check Stop-Loss / Take-Profit IF we are in a position
            exited = False
            if self.position_size > 0:
                if self.current_sl and candle['low'] <= self.current_sl:
                    self._close_position(min(candle['open'], self.current_sl), candle.name, reason="Stop-Loss Hit")
                    exited = True
                elif self.current_tp and candle['high'] >= self.current_tp:
                    self._close_position(max(candle['open'], self.current_tp), candle.name, reason="Take-Profit Hit")
                    exited = True

            # Kill switch: equity at the close against the daily loss and drawdown limits
            tripped = self.strategy.risk_manager.update_equity(
                candle.name, self.capital + self.position_size * candle['close'])
            if tripped:
                self.kill_switch = self.kill_switch or f"{tripped} at {candle.name}"
                if self.position_size > 0:
                    self._close_position(candle['close'], candle.name, reason=tripped)
                    exited = True

            # 2. Pass the last closed candles to Strategy & Risk Manager (no peeking at the future!)
            #    No new signal on a candle that already closed a position
            if not exited and i + 1 >= warmup:
                window = self.df.iloc[i + 1 - warmup:i + 1]
                signal = self.strategy.on_data(window, self.capital, self.position_size)

                # 3. Execute Validated Signals
                if signal.status == "VALIDATED":
                    if signal.direction == "BUY":
                        self._execute_buy(signal)
                    else:
                        self._close_position(signal.entry_price, signal.timestamp, reason=signal.reason)

            equity.append(self.capital + self.position_size * candle['close'])

        # End of backtest: Close any open positions at the final price
        if self.position_size > 0:
            self._close_position(self.df['close'].iloc[-1], self.df.index[-1], reason="End of Backtest")
            equity[-1] = self.capital
        self.equity = pd.Series(equity, index=self.df.index[self.start:])

        if verbose:
            self._print_results()
        return self.metrics()

    def _execute_buy(self, signal: Signal) -> None:
        """Simulates buying the asset at the signal price plus slippage, fee on top."""
        price = signal.entry_price * (1 + self.slippage)
        # the Risk Manager sized it to fit at the signal price; slippage may need a slightly smaller size
        size = min(signal.size, int(self.capital / (price * (1 + self.fee_rate)) * 1e6) / 1e6)
        if size <= 0:
            return
        cost = size * price
        fee = cost * self.fee_rate
        self.capital -= cost + fee
        self.position_size = size
        self.position_entry_price = price
        self.position_cost = cost + fee
        self.current_sl = signal.stop_loss
        self.current_tp = signal.take_profit

    def _close_position(self, exit_price: float, timestamp: pd.Timestamp, reason: str) -> None:
        """Simulates selling the asset at the given price minus slippage."""
        price = exit_price * (1 - self.slippage)
        net_revenue = self.position_size * price * (1 - self.fee_rate)
        self.capital += net_revenue

        # Record trade, net of both fees
        pnl = net_revenue - self.position_cost
        self.trade_history.append({
            'exit_time': timestamp,
            'reason': reason,
            'entry_price': self.position_entry_price,
            'exit_price': price,
            'size': self.position_size,
            'pnl_usd': pnl,
            'pnl_pct': pnl / self.position_cost * 100
        })

        # Reset position state
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.position_cost = 0.0
        self.current_sl = None
        self.current_tp = None

    def metrics(self) -> dict:
        """Return, trades, win rate, max drawdown, annualized Sharpe ratio, buy-and-hold return."""
        returns = self.equity.pct_change().dropna()
        candle = self.df.index.to_series().diff().median()
        periods_per_year = pd.Timedelta(days=365) / candle if pd.notna(candle) else 0
        sharpe = returns.mean() / returns.std() * math.sqrt(periods_per_year) if returns.std() > 0 else 0.0
        wins = [t for t in self.trade_history if t['pnl_usd'] > 0]
        close = self.df['close'].iloc[self.start:]
        return {
            'final_capital': self.capital,
            'return_pct': (self.capital / self.initial_capital - 1) * 100,
            'trades': len(self.trade_history),
            'win_rate_pct': len(wins) / len(self.trade_history) * 100 if self.trade_history else 0.0,
            'max_drawdown_pct': self.drawdown().min(),
            'sharpe': sharpe,
            'buy_and_hold_pct': (close.iloc[-1] / close.iloc[0] - 1) * 100,
        }

    def drawdown(self) -> pd.Series:
        return (self.equity / self.equity.cummax() - 1) * 100

    def _print_results(self) -> None:
        m = self.metrics()
        print("\n" + "="*40)
        print("📊 BACKTEST RESULTS")
        print("="*40)
        print(f"Period:           {self.equity.index[0]} – {self.equity.index[-1]}")
        print(f"Initial Capital:  ${self.initial_capital:,.2f}")
        print(f"Final Capital:    ${self.capital:,.2f}")
        print(f"Net Profit:       ${self.capital - self.initial_capital:,.2f} ({m['return_pct']:.2f}%)")
        print(f"Buy & Hold:       {m['buy_and_hold_pct']:.2f}%")
        print(f"Total Trades:     {m['trades']}")
        print(f"Win Rate:         {m['win_rate_pct']:.2f}%")
        print(f"Max Drawdown:     {m['max_drawdown_pct']:.2f}%")
        print(f"Sharpe Ratio:     {m['sharpe']:.2f}")
        if self.kill_switch:
            print(f"Kill switch:      {self.kill_switch}")
        print("="*40)


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest the SMA crossover strategy on OHLCV data")
    ap.add_argument("--data", default="data/historical/BTC-USD_1h.csv", help="CSV from data_downloader")
    ap.add_argument("--fast", type=int, default=10, help="fast SMA window")
    ap.add_argument("--slow", type=int, default=50, help="slow SMA window")
    ap.add_argument("--risk", type=float, default=0.02, help="risk per trade, share of capital")
    ap.add_argument("--sl", type=float, default=0.05, help="stop-loss, share of entry price")
    ap.add_argument("--tp", type=float, default=0.10, help="take-profit, share of entry price")
    ap.add_argument("--capital", type=float, default=10000.0)
    ap.add_argument("--fee", type=float, default=0.005, help="fee per side")
    ap.add_argument("--slippage", type=float, default=0.0, help="adverse price move per fill, e.g. 0.0005")
    ap.add_argument("--max-daily-loss", type=float, default=0.0, help="kill switch: daily loss share, 0 = off")
    ap.add_argument("--max-drawdown", type=float, default=0.0, help="kill switch: drawdown share, 0 = off")
    args = ap.parse_args()

    risk_manager = RiskManager(risk_per_trade_pct=args.risk, sl_pct=args.sl, tp_pct=args.tp, fee_rate=args.fee,
                               max_daily_loss_pct=args.max_daily_loss, max_drawdown_pct=args.max_drawdown)
    strategy = SMACrossover(name=f"SMA_{args.fast}_{args.slow}", risk_manager=risk_manager,
                            fast_window=args.fast, slow_window=args.slow)
    try:
        BacktestEngine(args.data, strategy, initial_capital=args.capital, fee_rate=args.fee,
                       slippage=args.slippage).run()
    except FileNotFoundError:
        print(f"❌ Could not find {args.data}. Please run data_downloader.py first.")


if __name__ == "__main__":
    main()
