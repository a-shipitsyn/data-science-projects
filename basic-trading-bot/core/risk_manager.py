import pandas as pd

from core.models import Signal


class RiskManager:
    def __init__(self, risk_per_trade_pct: float = 0.01, sl_pct: float = 0.05, tp_pct: float = 0.10,
                 fee_rate: float = 0.005, max_daily_loss_pct: float = 0.0, max_drawdown_pct: float = 0.0):
        """
        Long-only: BUY opens a position, SELL closes it.

        :param risk_per_trade_pct: Max % of total capital to risk losing on a single trade.
        :param sl_pct: Stop-Loss percentage drop from entry price.
        :param tp_pct: Take-Profit percentage gain from entry price.
        :param fee_rate: Trading fee per side, so that the position plus its fee fits the capital.
        :param max_daily_loss_pct: Kill switch: after losing this share of the day's starting equity,
            no new entries until the next day (UTC); 0 = off.
        :param max_drawdown_pct: Kill switch: after this drawdown from the equity peak, no new
            entries at all; 0 = off.
        """
        self.risk_per_trade_pct = risk_per_trade_pct
        self.sl_pct = sl_pct
        self.tp_pct = tp_pct
        self.fee_rate = fee_rate
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct

        # Kill switch state
        self.day = None
        self.day_start_equity = None
        self.peak_equity = None
        self.halted_until_day_end = False
        self.halted_permanently = False

    @property
    def halted(self) -> bool:
        return self.halted_until_day_end or self.halted_permanently

    def update_equity(self, timestamp: pd.Timestamp, equity: float) -> str | None:
        """Track the account and trip the kill switch; returns the reason when it trips now."""
        day = pd.Timestamp(timestamp).normalize()
        if day != self.day:
            self.day, self.day_start_equity = day, equity
            self.halted_until_day_end = False
        self.peak_equity = max(self.peak_equity or equity, equity)

        if (self.max_drawdown_pct and not self.halted_permanently
                and equity <= self.peak_equity * (1 - self.max_drawdown_pct)):
            self.halted_permanently = True
            return f"Kill Switch: drawdown {self.max_drawdown_pct:.0%} from peak"
        if (self.max_daily_loss_pct and not self.halted
                and equity <= self.day_start_equity * (1 - self.max_daily_loss_pct)):
            self.halted_until_day_end = True
            return f"Kill Switch: daily loss {self.max_daily_loss_pct:.0%}"
        return None

    def state(self) -> dict:
        return {"day": str(self.day) if self.day is not None else None, "day_start_equity": self.day_start_equity,
                "peak_equity": self.peak_equity, "halted_until_day_end": self.halted_until_day_end,
                "halted_permanently": self.halted_permanently}

    def restore(self, state: dict) -> None:
        self.day = pd.Timestamp(state["day"]) if state.get("day") else None
        self.day_start_equity = state.get("day_start_equity")
        self.peak_equity = state.get("peak_equity")
        self.halted_until_day_end = state.get("halted_until_day_end", False)
        self.halted_permanently = state.get("halted_permanently", False)

    def process_signal(self, signal: Signal, current_capital: float, position_size: float = 0.0) -> Signal:
        """Takes a RAW signal, sizes it, and applies risk parameters."""

        # BUY only when flat and not halted, SELL only when holding; everything else is rejected
        if signal.direction == "BUY" and position_size == 0 and not self.halted:
            return self._size_buy(signal, current_capital)
        if signal.direction == "SELL" and position_size > 0:
            signal.size = position_size
            signal.reason = signal.reason or "Strategy Sell Signal"
            signal.status = "VALIDATED"
            return signal
        signal.status = "REJECTED"
        if self.halted:
            signal.reason = "Kill switch active"
        return signal

    def _size_buy(self, signal: Signal, current_capital: float) -> Signal:
        # Calculate Stop-Loss and Take-Profit prices
        signal.stop_loss = signal.entry_price * (1 - self.sl_pct)
        signal.take_profit = signal.entry_price * (1 + self.tp_pct)

        # Position Sizing based on risk
        # Math: We are willing to lose (capital * risk_pct).
        # We will lose that amount if price drops by sl_pct.
        amount_to_risk = current_capital * self.risk_per_trade_pct

        # Dollar amount to invest so that hitting SL loses exactly 'amount_to_risk',
        # capped so that the position plus its entry fee fits the capital
        position_size_usd = min(amount_to_risk / self.sl_pct, current_capital / (1 + self.fee_rate))

        # Convert USD size to Asset size (e.g., amount of BTC), round down to 6 decimals
        signal.size = int(position_size_usd / signal.entry_price * 1e6) / 1e6
        signal.status = "VALIDATED" if signal.size > 0 else "REJECTED"
        return signal
