from abc import ABC, abstractmethod

import pandas as pd

from core.models import Signal
from core.risk_manager import RiskManager


class BaseStrategy(ABC):
    def __init__(self, name: str, risk_manager: RiskManager, asset: str = "BTC-USD"):
        self.name = name
        self.risk_manager = risk_manager
        self.asset = asset

    @property
    @abstractmethod
    def warmup(self) -> int:
        """Number of most recent candles the strategy needs to produce a signal."""

    @abstractmethod
    def generate_raw_signal(self, latest_data: pd.DataFrame) -> Signal:
        """
        This method MUST be overridden by child classes.
        It contains the actual math/logic (e.g., MACD cross).
        It returns a Signal object with status='RAW'.
        """

    def on_data(self, latest_data: pd.DataFrame, current_capital: float, position_size: float = 0.0) -> Signal:
        """
        The main entry point called by the Backtester or Live Bot with closed candles.
        1. Generates the Raw Signal.
        2. Passes it through the Risk Manager.
        3. Returns the Validated (or Rejected) Signal.
        """
        raw_signal = self.generate_raw_signal(latest_data)

        if raw_signal.direction in ["BUY", "SELL"]:
            return self.risk_manager.process_signal(raw_signal, current_capital, position_size)

        return raw_signal
