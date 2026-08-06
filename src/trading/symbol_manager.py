"""Manages the set of symbols the bot actively scans and trades, each with
its own risk profile (pip value, typical spread tolerance).

Adding a new pair later is a one-line addition to DEFAULT_SYMBOLS — nothing
else in the trading loop needs to change, since regime_detector, smc, and
decision_engine already operate on a (symbol, dataframe) pair without
knowing how many symbols exist in total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass(frozen=True)
class SymbolProfile:
    """Per-symbol trading parameters."""
    symbol: str                 # broker symbol, e.g. "XAUUSD", "EURUSD"
    display_name: str
    category: str               # "gold" | "forex"
    pip_size: float              # price move that equals 1 pip
    max_spread_pips: float       # skip trading if live spread exceeds this
    quote_currency: str          # currency this symbol is priced in (USD for most)
    base_currency: str           # the "other side" of the pair, for correlation checks


# Vantage-style symbol names. If your broker uses a different suffix
# (e.g. "EURUSDm", "XAUUSD.a"), add a matching profile — the bot tries each
# candidate against symbol_candidates the same way it already does for gold.
DEFAULT_SYMBOLS: List[SymbolProfile] = [
    SymbolProfile("XAUUSD", "Gold / USD", "gold", pip_size=0.1, max_spread_pips=5.0,
                   quote_currency="USD", base_currency="XAU"),
    SymbolProfile("EURUSD", "Euro / USD", "forex", pip_size=0.0001, max_spread_pips=2.0,
                   quote_currency="USD", base_currency="EUR"),
    SymbolProfile("GBPUSD", "Pound / USD", "forex", pip_size=0.0001, max_spread_pips=2.5,
                   quote_currency="USD", base_currency="GBP"),
    SymbolProfile("USDJPY", "USD / Yen", "forex", pip_size=0.01, max_spread_pips=2.0,
                   quote_currency="JPY", base_currency="USD"),
    SymbolProfile("AUDUSD", "Aussie / USD", "forex", pip_size=0.0001, max_spread_pips=2.5,
                   quote_currency="USD", base_currency="AUD"),
]

_BY_SYMBOL: Dict[str, SymbolProfile] = {p.symbol: p for p in DEFAULT_SYMBOLS}


class SymbolManager:
    """Holds the set of symbols currently active for scanning/trading."""

    def __init__(self, active_symbols: Optional[List[str]] = None) -> None:
        self.active_symbols: List[str] = active_symbols or ["XAUUSD"]

    def profile(self, symbol: str) -> SymbolProfile:
        if symbol in _BY_SYMBOL:
            return _BY_SYMBOL[symbol]
        # Fallback for symbols not in DEFAULT_SYMBOLS: derive base/quote from
        # standard 6-character forex naming (e.g. EURGBP -> base=EUR, quote=GBP).
        if len(symbol) == 6 and symbol.isalpha():
            base, quote = symbol[:3].upper(), symbol[3:].upper()
        else:
            base, quote = symbol.upper(), "USD"
        return SymbolProfile(symbol, symbol, "forex", pip_size=0.0001, max_spread_pips=3.0,
                              quote_currency=quote, base_currency=base)

    def is_usd_exposed(self, symbol: str) -> bool:
        p = self.profile(symbol)
        return p.quote_currency == "USD" or p.base_currency == "USD" or p.category == "gold"

    @staticmethod
    def all_known_symbols() -> List[str]:
        return [p.symbol for p in DEFAULT_SYMBOLS]
