"""Guards against taking correlated trades that look diversified but are
actually the same USD bet stacked twice (e.g. BUY gold + BUY EUR/USD are
both effectively "short USD").

This runs as a gate inside risk_manager's pipeline, right before a new
trade is sized — it never touches the broker directly (single-writer rule
from the architecture blueprint: only execution/ may call the broker).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.trading.symbol_manager import SymbolManager


@dataclass
class OpenExposure:
    symbol: str
    direction: str  # "BUY" or "SELL"


def _usd_direction_units(symbol_manager: SymbolManager, symbol: str, direction: str) -> float:
    """Return signed USD exposure: +1 = net long USD, -1 = net short USD, 0 = no USD leg."""
    profile = symbol_manager.profile(symbol)
    sign = 1 if direction == "BUY" else -1

    if profile.category == "gold":
        # Gold conventionally moves opposite USD strength: buying gold is a bet against USD.
        return -1 * sign
    if profile.quote_currency == "USD" and profile.base_currency != "USD":
        # e.g. EURUSD, GBPUSD, AUDUSD: buying the pair = buying base, selling USD.
        return -1 * sign
    if profile.base_currency == "USD" and profile.quote_currency != "USD":
        # e.g. USDJPY: buying the pair = buying USD.
        return 1 * sign
    return 0.0  # no USD leg at all (e.g. a cross pair like EURGBP)


class CorrelationGuard:
    """Checks whether adding a new trade would push net USD exposure too high."""

    def __init__(self, symbol_manager: SymbolManager, max_net_usd_exposure: float = 2.0) -> None:
        self.symbol_manager = symbol_manager
        self.max_net_usd_exposure = max_net_usd_exposure

    def check(
        self, open_positions: List[OpenExposure], new_symbol: str, new_direction: str,
    ) -> "CorrelationCheckResult":
        current_net = sum(
            _usd_direction_units(self.symbol_manager, p.symbol, p.direction) for p in open_positions
        )
        new_leg = _usd_direction_units(self.symbol_manager, new_symbol, new_direction)
        projected_net = current_net + new_leg

        allowed = abs(projected_net) <= self.max_net_usd_exposure
        reason = ""
        if not allowed:
            direction_word = "short" if projected_net < 0 else "long"
            reason = (
                f"Adding {new_direction} {new_symbol} would push net USD exposure to "
                f"{projected_net:+.1f} ({direction_word} USD), exceeding the "
                f"{self.max_net_usd_exposure:.1f} limit — too many correlated positions."
            )
        return CorrelationCheckResult(allowed=allowed, current_net=current_net,
                                       projected_net=projected_net, reason=reason)


@dataclass
class CorrelationCheckResult:
    allowed: bool
    current_net: float
    projected_net: float
    reason: str = ""
