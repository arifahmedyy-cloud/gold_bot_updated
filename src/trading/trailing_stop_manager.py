"""Trailing stop + break-even execution.

Uses the R-multiple (how many multiples of the original risk distance the
trade has moved in profit) to decide when to move a position's stop-loss:

- Once profit reaches `breakeven_trigger_r`, move SL to entry + a small
  buffer (locks in a small win, removes downside risk).
- Once profit reaches `trailing_trigger_r`, trail the SL behind price at a
  distance of `trailing_distance_r` R (locks in more profit as price runs).

The stop is only ever moved in the favorable direction — this module never
loosens an existing stop, even if called with stale/out-of-order data.

This module only *computes* the proposed new SL. It never calls the broker
directly — the caller (app.py's trading loop) is the single place that
calls `broker.modify_position_sl_tp()`, consistent with keeping one place
responsible for all broker-side order modification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.config import RiskConfig


@dataclass
class TrailingStopResult:
    new_sl: Optional[float]
    reason: str = ""


class TrailingStopManager:
    def __init__(self, config: RiskConfig) -> None:
        self.config = config

    def compute_new_sl(
        self,
        direction: str,
        entry_price: float,
        initial_sl: float,
        current_price: float,
        current_sl: float,
    ) -> TrailingStopResult:
        """Return a proposed new SL, or None if no change is warranted.

        direction: "BUY" or "SELL"
        entry_price / initial_sl: the position's original values at open
            (fixed reference, from position_risk — never the live SL)
        current_price: latest market price
        current_sl: the SL currently set on the broker (so we never loosen it)
        """
        if not self.config.use_trailing_stop:
            return TrailingStopResult(new_sl=None, reason="trailing stop disabled")

        risk_distance = abs(entry_price - initial_sl)
        if risk_distance <= 0:
            return TrailingStopResult(new_sl=None, reason="no valid risk distance")

        sign = 1 if direction == "BUY" else -1
        profit_r = ((current_price - entry_price) / risk_distance) * sign

        candidate_sl: Optional[float] = None
        reason = ""

        if profit_r >= self.config.trailing_trigger_r:
            offset = self.config.trailing_distance_r * risk_distance
            candidate_sl = current_price - (offset * sign)
            reason = f"trailing at {self.config.trailing_distance_r}R (profit={profit_r:.2f}R)"
        elif profit_r >= self.config.breakeven_trigger_r:
            buffer = self.config.breakeven_buffer_pct * risk_distance
            candidate_sl = entry_price + (buffer * sign)
            reason = f"break-even + buffer (profit={profit_r:.2f}R)"

        if candidate_sl is None:
            return TrailingStopResult(new_sl=None, reason=f"profit {profit_r:.2f}R below trigger")

        # Never loosen the stop: it may only move in the trade's favor.
        if direction == "BUY" and candidate_sl <= current_sl:
            return TrailingStopResult(new_sl=None, reason="candidate SL would loosen the stop")
        if direction == "SELL" and candidate_sl >= current_sl:
            return TrailingStopResult(new_sl=None, reason="candidate SL would loosen the stop")

        return TrailingStopResult(new_sl=candidate_sl, reason=reason)
