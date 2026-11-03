import argparse
import asyncio
import contextlib
import json
import logging
import os
import signal as os_signal

import ccxt
import pandas as pd

from core.models import Signal
from core.risk_manager import RiskManager
from core.strategies.sma_crossover import SMACrossover
from trading_bot.notifier import Notifier
from trading_bot.signal_logger import SignalLogger
from trading_bot.websocket_client import CoinbaseWebsocket

log = logging.getLogger(__name__)
MINUTE = pd.Timedelta(minutes=1)


class LiveTradingBot:
    """Live paper-trading signal bot.

    Ticks from the WebSocket are aggregated into 1-minute candles (open, high, low, close, volume).
    The shared Strategy -> Risk pipeline runs once per closed candle (never on an unfinished one),
    the paper position is tracked with fees, stop-loss and take-profit are checked on every tick,
    and each executed action is appended to the signal log once.

    Operations:
    - kill switch: the risk manager stops new entries after the daily loss limit (until the next
      UTC day) or the maximum drawdown (for good), and the open position is closed;
    - watchdog: candles are closed on a timer, so a minute without ticks still produces a (flat)
      candle; a feed silent for --stale-after seconds triggers a warning and a reconnect;
    - the paper account and the kill switch state are saved to a JSON file after every trade and
      restored on start; notifications go to Telegram when configured (see notifier.py).
    """

    def __init__(self, asset: str = "BTC-USD", fast_window: int = 10, slow_window: int = 50,
                 capital: float = 10000.0, fee_rate: float = 0.005, risk_per_trade_pct: float = 0.02,
                 sl_pct: float = 0.05, tp_pct: float = 0.10, max_daily_loss_pct: float = 0.05,
                 max_drawdown_pct: float = 0.20, log_path: str = "data/live_logs/live_signals.csv",
                 state_path: str = "data/live_logs/paper_state.json", stale_after: float = 60.0,
                 notifier: Notifier | None = None):
        # 1. Initialize Core Shared Components (Same as Backtester!)
        self.asset = asset
        self.fee_rate = fee_rate
        self.state_path = state_path
        self.stale_after = stale_after
        self.risk_manager = RiskManager(risk_per_trade_pct=risk_per_trade_pct, sl_pct=sl_pct, tp_pct=tp_pct,
                                        fee_rate=fee_rate, max_daily_loss_pct=max_daily_loss_pct,
                                        max_drawdown_pct=max_drawdown_pct)

        # We use a 1-minute chart here so you can see signals happen faster during testing
        self.strategy = SMACrossover(name=f"SMA_{fast_window}_{slow_window}_Live", risk_manager=self.risk_manager,
                                     fast_window=fast_window, slow_window=slow_window, asset=asset)

        self.logger = SignalLogger(log_path)
        self.ws_client = CoinbaseWebsocket(product_ids=[asset])
        self.notifier = notifier or Notifier()

        # Simulated paper trading account
        self.capital = capital
        self.position_size = 0.0
        self.stop_loss = None
        self.take_profit = None

        # Closed candles, the candle being built from ticks, time of the last tick
        self.candles = pd.DataFrame()
        self.current = None
        self.last_tick = None
        self._load_state()

    # --- state -------------------------------------------------------------------------------

    def _load_state(self) -> None:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                state = json.load(f)
            if state.get("asset") == self.asset:
                self.capital = state["capital"]
                self.position_size = state["position_size"]
                self.stop_loss, self.take_profit = state["stop_loss"], state["take_profit"]
                self.risk_manager.restore(state.get("risk", {}))
                log.info("Restored paper state: capital $%s, position %s", f"{self.capital:,.2f}", self.position_size)

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump({"asset": self.asset, "capital": self.capital, "position_size": self.position_size,
                       "stop_loss": self.stop_loss, "take_profit": self.take_profit,
                       "risk": self.risk_manager.state()}, f, indent=2)

    def warmup(self) -> None:
        """Fetches recent closed 1-minute candles to calculate the initial Moving Averages."""
        log.info("Warming up data buffer via REST API...")
        exchange = ccxt.coinbase()
        ohlcv = exchange.fetch_ohlcv(self.asset.replace('-', '/'), timeframe='1m', limit=self.strategy.warmup + 10)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')

        # The last candle may still be open: drop it, it will be rebuilt from ticks
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor('1min')
        self.candles = df[df.index < now]
        log.info("Warmup complete. Buffer loaded with %d closed candles.", len(self.candles))

    # --- market data ---------------------------------------------------------------------------

    async def on_tick(self, data: dict) -> None:
        """Callback function triggered every time the WebSocket receives a new price."""
        try:
            price = float(data['price'])
            size = float(data.get('last_size') or 0.0)
            # Coinbase WS returns time in ISO format: '2023-10-25T15:30:00.123456Z'
            tick_time = pd.to_datetime(data['time']).tz_convert(None)
        except (KeyError, ValueError, TypeError) as e:
            log.warning("Error processing tick: %s", e)
            return
        self.last_tick = pd.Timestamp.now(tz="UTC").tz_localize(None)

        # Stop-loss / take-profit of the paper position on every tick
        if self.position_size > 0:
            if price <= self.stop_loss:
                self._sell(price, tick_time, "Stop-Loss Hit")
            elif price >= self.take_profit:
                self._sell(price, tick_time, "Take-Profit Hit")

        minute = tick_time.floor('1min')
        self.roll(minute)
        if self.current is None:
            self.current = {'start': minute, 'open': price, 'high': price, 'low': price, 'close': price, 'volume': 0.0}
        c = self.current
        c['high'], c['low'], c['close'] = max(c['high'], price), min(c['low'], price), price
        c['volume'] += size

    def roll(self, minute: pd.Timestamp) -> None:
        """Close every candle that started before `minute`; the next candle starts at the last close, so
        minutes without ticks become flat candles."""
        while self.current is not None and self.current['start'] < minute:
            self._on_candle_close()
            last = self.current['close']
            start = self.current['start'] + MINUTE
            self.current = {'start': start, 'open': last, 'high': last, 'low': last, 'close': last, 'volume': 0.0}

    def _on_candle_close(self) -> None:
        c = self.current
        row = pd.DataFrame({k: [c[k]] for k in ['open', 'high', 'low', 'close', 'volume']}, index=[c['start']])
        self.candles = pd.concat([self.candles, row]).tail(self.strategy.warmup * 2)
        log.info("Candle %s closed. %s close: $%s", f"{c['start']:%H:%M}", self.asset, f"{c['close']:,.2f}")

        # Kill switch: account equity at the close against the daily loss and drawdown limits
        tripped = self.risk_manager.update_equity(c['start'], self.capital + self.position_size * c['close'])
        if tripped:
            self.notifier.send(f"{self.asset}: {tripped}, no new entries")
            if self.position_size > 0:
                self._sell(c['close'], c['start'] + MINUTE, tripped)
            self._save_state()
            return

        # FEED CLOSED CANDLES TO SHARED CORE PIPELINE
        signal = self.strategy.on_data(self.candles, self.capital, self.position_size)
        if signal.status == "VALIDATED":
            if signal.direction == "BUY":
                self._buy(signal)
            else:
                self._sell(signal.entry_price, signal.timestamp, signal.reason)

    async def watchdog(self, interval: float = 1.0) -> None:
        """Close candles on time and reconnect a silent feed."""
        warned = False
        while True:
            await asyncio.sleep(interval)
            now = pd.Timestamp.now(tz="UTC").tz_localize(None)
            self.roll(now.floor('1min'))
            silent = self.last_tick is not None and (now - self.last_tick).total_seconds() > self.stale_after
            if silent and not warned:
                self.notifier.send(f"{self.asset}: no ticks for {self.stale_after:.0f} s, reconnecting")
                await self.ws_client.reconnect()
                warned = True
            elif not silent:
                warned = False

    # --- paper trading -------------------------------------------------------------------------

    def _buy(self, signal: Signal) -> None:
        self.capital -= signal.size * signal.entry_price * (1 + self.fee_rate)
        self.position_size = signal.size
        self.stop_loss, self.take_profit = signal.stop_loss, signal.take_profit
        self.logger.log_signal(signal)
        self._save_state()
        self.notifier.send(f"BUY {signal.size} {self.asset} @ ${signal.entry_price:,.2f}")

    def _sell(self, price: float, timestamp: pd.Timestamp, reason: str) -> None:
        signal = Signal(timestamp=timestamp, asset=self.asset, direction="SELL", entry_price=price,
                        size=self.position_size, strategy_name=self.strategy.name, status="VALIDATED", reason=reason)
        self.capital += self.position_size * price * (1 - self.fee_rate)
        self.position_size = 0.0
        self.stop_loss = self.take_profit = None
        self.logger.log_signal(signal)
        self._save_state()
        self.notifier.send(f"SELL {signal.size} {self.asset} @ ${price:,.2f} ({reason}), "
                           f"paper capital ${self.capital:,.2f}")

    async def run(self) -> None:
        """Main execution loop: WebSocket feed and watchdog, until cancelled (Ctrl+C or SIGTERM)."""
        self.warmup()
        log.info("Starting Live Paper Trading Bot...")
        tasks = [asyncio.create_task(self.ws_client.start(self.on_tick)), asyncio.create_task(self.watchdog())]
        loop = asyncio.get_running_loop()
        for sig in (os_signal.SIGINT, os_signal.SIGTERM):
            with contextlib.suppress(NotImplementedError):
                loop.add_signal_handler(sig, lambda: [t.cancel() for t in tasks])
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._save_state()
            log.info("Stopped. Paper capital $%s, position %s", f"{self.capital:,.2f}", self.position_size)


def main() -> None:
    ap = argparse.ArgumentParser(description="Live paper-trading signal bot on 1-minute candles")
    ap.add_argument("--asset", default="BTC-USD")
    ap.add_argument("--fast", type=int, default=10, help="fast SMA window")
    ap.add_argument("--slow", type=int, default=50, help="slow SMA window")
    ap.add_argument("--capital", type=float, default=10000.0, help="paper capital at the first start")
    ap.add_argument("--risk", type=float, default=0.02, help="risk per trade, share of capital")
    ap.add_argument("--sl", type=float, default=0.05, help="stop-loss, share of entry price")
    ap.add_argument("--tp", type=float, default=0.10, help="take-profit, share of entry price")
    ap.add_argument("--fee", type=float, default=0.005, help="fee per side")
    ap.add_argument("--max-daily-loss", type=float, default=0.05, help="kill switch: daily loss share, 0 = off")
    ap.add_argument("--max-drawdown", type=float, default=0.20, help="kill switch: drawdown share, 0 = off")
    ap.add_argument("--stale-after", type=float, default=60.0, help="seconds without ticks before reconnecting")
    ap.add_argument("--log", default="data/live_logs/live_signals.csv", help="signal log CSV")
    ap.add_argument("--state", default="data/live_logs/paper_state.json", help="paper account state")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    bot = LiveTradingBot(args.asset, args.fast, args.slow, args.capital, args.fee, args.risk, args.sl, args.tp,
                         args.max_daily_loss, args.max_drawdown, log_path=args.log, state_path=args.state,
                         stale_after=args.stale_after)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
