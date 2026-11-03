import logging
import os

import pandas as pd

from core.models import Signal

log = logging.getLogger(__name__)


class SignalLogger:
    def __init__(self, filepath: str = "data/live_logs/live_signals.csv"):
        self.filepath = filepath

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)

        # Create the CSV with headers if it doesn't exist
        if not os.path.exists(self.filepath):
            df = pd.DataFrame(columns=[
                "timestamp", "asset", "direction", "size",
                "entry_price", "stop_loss", "take_profit",
                "strategy_name", "status", "reason"
            ])
            df.to_csv(self.filepath, index=False)

    def log_signal(self, signal: Signal) -> None:
        """Appends a new signal to the CSV log file."""
        # Convert signal data class to a dictionary
        signal_dict = signal.to_dict()

        # Convert to a single-row DataFrame
        df_new = pd.DataFrame([signal_dict])

        # Append to CSV
        df_new.to_csv(self.filepath, mode='a', header=False, index=False)

        log.info("LOGGED: %s %s %s @ $%.2f (%s)", signal.direction, signal.size, signal.asset, signal.entry_price,
                 signal.reason or signal.strategy_name)
