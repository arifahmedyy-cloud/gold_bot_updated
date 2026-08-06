"""Tests for symbol_manager.py and correlation_guard.py — multi-symbol
support and USD correlation protection."""

import pytest

from src.trading.symbol_manager import SymbolManager, DEFAULT_SYMBOLS
from src.trading.correlation_guard import CorrelationGuard, OpenExposure


class TestSymbolManager:
    def test_default_active_symbol_is_gold(self):
        mgr = SymbolManager()
        assert mgr.active_symbols == ["XAUUSD"]

    def test_custom_active_symbols(self):
        mgr = SymbolManager(["XAUUSD", "EURUSD"])
        assert "EURUSD" in mgr.active_symbols

    def test_known_profile_lookup(self):
        mgr = SymbolManager()
        profile = mgr.profile("EURUSD")
        assert profile.category == "forex"
        assert profile.quote_currency == "USD"
        assert profile.base_currency == "EUR"

    def test_unknown_symbol_gets_fallback_profile(self):
        mgr = SymbolManager()
        profile = mgr.profile("EURGBP")
        assert profile.symbol == "EURGBP"

    def test_gold_is_usd_exposed(self):
        mgr = SymbolManager()
        assert mgr.is_usd_exposed("XAUUSD") is True

    def test_all_known_symbols_nonempty(self):
        assert len(SymbolManager.all_known_symbols()) == len(DEFAULT_SYMBOLS)


class TestCorrelationGuard:
    def test_no_open_positions_allows_trade(self):
        guard = CorrelationGuard(SymbolManager())
        result = guard.check([], "XAUUSD", "BUY")
        assert result.allowed is True

    def test_stacking_same_usd_direction_gets_blocked(self):
        guard = CorrelationGuard(SymbolManager(), max_net_usd_exposure=1.0)
        # Already short USD via gold; buying EURUSD is another short-USD bet.
        open_positions = [OpenExposure(symbol="XAUUSD", direction="BUY")]
        result = guard.check(open_positions, "EURUSD", "BUY")
        assert result.allowed is False
        assert "correlated" in result.reason.lower()

    def test_opposite_direction_reduces_exposure_and_allowed(self):
        guard = CorrelationGuard(SymbolManager(), max_net_usd_exposure=1.0)
        # Short USD via gold BUY, then SELL EURUSD is long USD — offsets, should be allowed.
        open_positions = [OpenExposure(symbol="XAUUSD", direction="BUY")]
        result = guard.check(open_positions, "EURUSD", "SELL")
        assert result.allowed is True

    def test_usdjpy_direction_is_opposite_convention(self):
        guard = CorrelationGuard(SymbolManager(), max_net_usd_exposure=1.0)
        # BUY USDJPY = long USD. BUY EURUSD = short USD. These offset.
        open_positions = [OpenExposure(symbol="USDJPY", direction="BUY")]
        result = guard.check(open_positions, "EURUSD", "BUY")
        assert result.allowed is True

    def test_cross_pair_with_no_usd_leg_never_blocked(self):
        guard = CorrelationGuard(SymbolManager(), max_net_usd_exposure=0.5)
        result = guard.check([], "EURGBP", "BUY")
        assert result.allowed is True  # EURGBP has no USD leg in our model

    def test_projected_net_reported(self):
        guard = CorrelationGuard(SymbolManager(), max_net_usd_exposure=5.0)
        result = guard.check([], "XAUUSD", "BUY")
        assert result.projected_net == -1.0
