import pandas as pd

from core.base_strategy import BaseStrategy
from core.models import Signal


class SMACrossover(BaseStrategy):
    def __init__(self, name: str, risk_manager, fast_window: int = 10, slow_window: int = 50,
                 asset: str = "BTC-USD"):
        super().__init__(name, risk_manager, asset)
        self.fast_window = fast_window
        self.slow_window = slow_window

    @property
    def warmup(self) -> int:
        # the slow average for the current and the previous candle
        return self.slow_window + 1

    def generate_raw_signal(self, latest_data: pd.DataFrame) -> Signal:
        """
        Expects a pandas DataFrame with at least a 'close' column and a DatetimeIndex.
        """
        # 1. Check if we have enough data to calculate the slow moving average twice
        if len(latest_data) < self.warmup:
            return Signal(
                timestamp=latest_data.index[-1] if not latest_data.empty else pd.Timestamp.now(),
                asset=self.asset,
                direction="HOLD",
                entry_price=0.0,
                strategy_name=self.name
            )

        # 2. Calculate the Simple Moving Averages on the last candles only
        close = latest_data['close'].iloc[-self.warmup:]
        fast_sma = close.rolling(window=self.fast_window).mean()
        slow_sma = close.rolling(window=self.slow_window).mean()

        # 3. Get the last two values to check for a "crossover"
        current_fast, prev_fast = fast_sma.iloc[-1], fast_sma.iloc[-2]
        current_slow, prev_slow = slow_sma.iloc[-1], slow_sma.iloc[-2]

        # 4. Determine Direction
        direction = "HOLD"

        # Golden Cross: Fast crosses above Slow
        if prev_fast <= prev_slow and current_fast > current_slow:
            direction = "BUY"

        # Death Cross: Fast crosses below Slow
        elif prev_fast >= prev_slow and current_fast < current_slow:
            direction = "SELL"

        # 5. Return the RAW signal
        return Signal(
            timestamp=latest_data.index[-1],
            asset=self.asset,
            direction=direction,
            entry_price=close.iloc[-1],
            strategy_name=self.name
        )
