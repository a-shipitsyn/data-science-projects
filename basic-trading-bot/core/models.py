from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """Standardized format for all trading signals (long-only: BUY opens, SELL closes)."""
    timestamp: datetime
    asset: str                 # e.g., 'BTC-USD'
    direction: str             # 'BUY', 'SELL', or 'HOLD'
    entry_price: float         # The price when the signal was generated

    # These will be populated by the Risk Manager
    size: float = 0.0          # Amount of base asset to buy/sell (e.g., 0.05 BTC)
    stop_loss: float | None = None
    take_profit: float | None = None

    # Metadata for logging
    strategy_name: str = "Unknown"
    status: str = "RAW"        # 'RAW', 'VALIDATED', or 'REJECTED'
    reason: str = ""           # why a position is closed: strategy, stop-loss, take-profit

    def to_dict(self) -> dict:
        """Useful for writing to the live_signals.csv log file."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "asset": self.asset,
            "direction": self.direction,
            "size": self.size,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy_name": self.strategy_name,
            "status": self.status,
            "reason": self.reason,
        }
