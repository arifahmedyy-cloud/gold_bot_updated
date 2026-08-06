"""Risk management with Kelly sizing, drawdown adjustment, and daily guards.

All position sizing and risk validation logic centralized here.
"""

from __future__ import annotations

import math
from datetime import datetime, date
from typing import Optional, Dict, Any, List

from src.logger import get_logger
from src.models import DailyGuardStatus
from src.config import RiskConfig
from src.exceptions import RiskError

log = get_logger(__name__)


class RiskManager:
    """Centralized risk management for all trades."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._consecutive_losses = 0
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._last_reset_date: Optional[date] = None
        self._trade_history: List[Dict[str, Any]] = []
        self._max_daily_pnl = 0.0

    def _reset_daily_if_needed(self) -> None:
        today = date.today()
        if self._last_reset_date != today:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._max_daily_pnl = 0.0
            self._last_reset_date = today
            log.info("Daily risk counters reset")

    def compute_drawdown_adjusted_risk(
        self,
        balance: float,
        peak_balance: float,
        base_risk_pct: float,
    ) -> float:
        """Reduce risk proportionally to current drawdown.

        Args:
            balance: Current account balance.
            peak_balance: Highest balance achieved.
            base_risk_pct: Base risk percentage.

        Returns:
            Adjusted risk percentage.
        """
        if peak_balance <= 0 or balance <= 0:
            return base_risk_pct
        drawdown_pct = (peak_balance - balance) / peak_balance
        if drawdown_pct <= 0:
            return base_risk_pct
        if drawdown_pct >= 0.5:
            # Severe drawdown circuit breaker: cap risk to the floor.
            return max(0.1, base_risk_pct * 0.1)
        reduction = 1 - (drawdown_pct * self.config.drawdown_reduction_factor)
        return max(0.1, base_risk_pct * max(reduction, 0.1))

    def kelly_position_size(
        self,
        balance: float,
        entry: float,
        sl: float,
        base_risk_pct: float,
    ) -> tuple[float, float]:
        """Calculate position size using Kelly Criterion with fractional Kelly.

        Args:
            balance: Account balance.
            entry: Entry price.
            sl: Stop loss price.
            base_risk_pct: Base risk percentage.

        Returns:
            Tuple of (suggested_risk_pct, kelly_pct).
        """
        if len(self._trade_history) < self.config.kelly_min_trades:
            log.info("Kelly: insufficient history (%d < %d), using base risk",
                     len(self._trade_history), self.config.kelly_min_trades)
            return base_risk_pct, 0.0

        wins = [t for t in self._trade_history if t.get("profit_loss", 0) > 0]
        losses = [t for t in self._trade_history if t.get("profit_loss", 0) < 0]
        if not wins or not losses:
            return base_risk_pct, 0.0

        win_rate = len(wins) / len(self._trade_history)
        avg_win = sum(t["profit_loss"] for t in wins) / len(wins)
        avg_loss = abs(sum(t["profit_loss"] for t in losses) / len(losses))
        if avg_loss == 0:
            return base_risk_pct, 0.0

        kelly_pct = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        kelly_pct = max(0, min(kelly_pct, 0.25))
        suggested = round(
            min(max(kelly_pct * self.config.kelly_fraction * 100, 0.25), self.config.max_risk_pct),
            2,
        )
        log.info("Kelly: win_rate=%.2f kelly=%.4f suggested=%.2f%%", win_rate, kelly_pct, suggested)
        return suggested, kelly_pct

    def calculate_lot_size(
        self,
        balance: float,
        entry: float,
        sl: float,
        risk_pct: float,
        leverage: float = 100.0,
    ) -> float:
        """Calculate lot size for XAUUSD.

        Gold: 1.0 lot = 100 oz, so $1 move = $100 per lot.
        Risk amount = balance * risk_pct / 100.
        Lots = risk_amount / (|entry - sl| * 100).

        Args:
            balance: Account balance.
            entry: Entry price.
            sl: Stop loss.
            risk_pct: Risk percentage.
            leverage: Account leverage.

        Returns:
            Lot size (rounded to 2 decimals).
        """
        risk_amount = balance * (risk_pct / 100.0)
        price_distance = abs(entry - sl)
        if price_distance <= 0:
            log.warning("Zero price distance, defaulting to 0.01 lot")
            return 0.01
        raw_lots = risk_amount / (price_distance * 100.0)
        # Margin check
        notional = raw_lots * entry * 100.0
        margin = notional / leverage
        if margin > balance * 0.5:
            raw_lots = (balance * 0.5 * leverage) / (entry * 100.0)
            log.warning("Margin limit hit, reduced to %.2f lots", raw_lots)
        return round(max(0.01, raw_lots), 2)

    def daily_guard(self, balance: float, peak_balance: float) -> DailyGuardStatus:
        """Check if daily limits are breached.

        Args:
            balance: Current balance.
            peak_balance: Peak balance.

        Returns:
            DailyGuardStatus with block/close recommendations.
        """
        self._reset_daily_if_needed()
        should_block = False
        should_close = False
        reason = ""

        daily_pnl_pct = (self._daily_pnl / peak_balance * 100) if peak_balance > 0 else 0

        if daily_pnl_pct <= -self.config.daily_loss_limit_pct:
            should_block = True
            should_close = True
            reason = f"Daily loss limit hit: {daily_pnl_pct:.2f}%"
            log.warning("DAILY GUARD: %s", reason)
        elif self._consecutive_losses >= self.config.max_consecutive_losses:
            should_block = True
            reason = f"Max consecutive losses ({self.config.max_consecutive_losses}) reached"
            log.warning("DAILY GUARD: %s", reason)
        elif self.config.daily_profit_lock_pct and daily_pnl_pct >= self.config.daily_profit_lock_pct:
            should_block = True
            reason = f"Daily profit target reached: {daily_pnl_pct:.2f}%"
            log.info("DAILY GUARD: %s", reason)

        return DailyGuardStatus(
            should_block_new_trades=should_block,
            should_close_all=should_close,
            daily_pnl_pct=daily_pnl_pct,
            reason=reason,
        )

    def record_trade_result(self, profit_loss: float) -> None:
        """Record trade result for consecutive loss tracking."""
        self._reset_daily_if_needed()
        self._trade_history.append({"profit_loss": profit_loss, "time": datetime.now()})
        self._daily_pnl += profit_loss
        self._daily_trades += 1
        if profit_loss < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        if self._daily_pnl > self._max_daily_pnl:
            self._max_daily_pnl = self._daily_pnl
        log.info("Risk record: P/L=$%.2f daily_pnl=$%.2f consecutive_losses=%d",
                 profit_loss, self._daily_pnl, self._consecutive_losses)

    def validate_signal(
        self,
        entry: float,
        sl: float,
        tp: float,
        lot_size: float,
        balance: float,
        ai_score: int,
        ml_confidence: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Validate a trade signal against all risk rules.

        Args:
            entry: Entry price.
            sl: Stop loss.
            tp: Take profit.
            lot_size: Calculated lot size.
            balance: Account balance.
            ai_score: AI confidence score.
            ml_confidence: ML model confidence.

        Returns:
            Tuple of (is_valid, reason).
        """
        if ai_score < self.config.min_ai_score:
            return False, f"AI score {ai_score} below minimum {self.config.min_ai_score}"
        if self.config.enable_ml_filter and ml_confidence is not None:
            if ml_confidence < self.config.min_ml_confidence:
                return False, f"ML confidence {ml_confidence:.1f} below minimum {self.config.min_ml_confidence}"
        if sl <= 0 or tp <= 0:
            return False, "SL and TP must be positive"
        if entry == sl:
            return False, "Entry equals SL (zero risk distance)"
        risk_distance = abs(entry - sl)
        reward_distance = abs(tp - entry)
        if reward_distance < risk_distance:
            return False, f"Reward ({reward_distance:.2f}) < Risk ({risk_distance:.2f})"
        risk_amount = balance * (self.config.base_risk_pct / 100.0)
        calculated_risk = risk_distance * lot_size * 100.0
        if calculated_risk > risk_amount * 1.5:
            return False, f"Calculated risk ${calculated_risk:.2f} exceeds allowed ${risk_amount:.2f}"
        return True, ""

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk status summary."""
        self._reset_daily_if_needed()
        return {
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "total_trades": len(self._trade_history),
            "max_daily_pnl": self._max_daily_pnl,
            "win_rate": len([t for t in self._trade_history if t.get("profit_loss", 0) > 0]) / max(1, len(self._trade_history)),
        }
