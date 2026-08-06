"""Tests for trailing_stop_manager.py."""

import pytest

from src.trading.trailing_stop_manager import TrailingStopManager
from src.config import RiskConfig


def _config(**overrides):
    base = dict(
        use_trailing_stop=True,
        breakeven_trigger_r=1.0,
        breakeven_buffer_pct=0.05,
        trailing_trigger_r=1.5,
        trailing_distance_r=0.5,
    )
    base.update(overrides)
    return RiskConfig(**base)


class TestTrailingStopManager:
    def test_disabled_returns_none(self):
        mgr = TrailingStopManager(_config(use_trailing_stop=False))
        result = mgr.compute_new_sl("BUY", 2450.0, 2440.0, 2470.0, 2440.0)
        assert result.new_sl is None

    def test_below_breakeven_trigger_no_change(self):
        mgr = TrailingStopManager(_config())
        # risk = 10, profit_r = (2455-2450)/10 = 0.5R, below breakeven trigger of 1.0R
        result = mgr.compute_new_sl("BUY", 2450.0, 2440.0, 2455.0, 2440.0)
        assert result.new_sl is None

    def test_breakeven_trigger_moves_sl_to_entry_plus_buffer(self):
        mgr = TrailingStopManager(_config())
        # risk = 10, profit_r = (2461-2450)/10 = 1.1R, above breakeven trigger, below trailing trigger
        result = mgr.compute_new_sl("BUY", 2450.0, 2440.0, 2461.0, 2440.0)
        assert result.new_sl is not None
        assert result.new_sl == pytest.approx(2450.0 + 0.05 * 10, abs=0.01)

    def test_trailing_trigger_trails_behind_price(self):
        mgr = TrailingStopManager(_config())
        # risk = 10, profit_r = (2470-2450)/10 = 2.0R, above trailing trigger of 1.5R
        result = mgr.compute_new_sl("BUY", 2450.0, 2440.0, 2470.0, 2450.5)
        assert result.new_sl is not None
        expected = 2470.0 - 0.5 * 10
        assert result.new_sl == pytest.approx(expected, abs=0.01)

    def test_never_loosens_stop_buy(self):
        mgr = TrailingStopManager(_config())
        # current_sl already better than what breakeven would produce
        result = mgr.compute_new_sl("BUY", 2450.0, 2440.0, 2461.0, 2455.0)
        assert result.new_sl is None

    def test_sell_direction_mirrors_logic(self):
        mgr = TrailingStopManager(_config())
        # SELL: entry=2450, initial_sl=2460 (risk=10), price drops to 2439 -> profit_r=1.1R
        result = mgr.compute_new_sl("SELL", 2450.0, 2460.0, 2439.0, 2460.0)
        assert result.new_sl is not None
        assert result.new_sl < 2450.0  # moved favorably (down) for a short

    def test_never_loosens_stop_sell(self):
        mgr = TrailingStopManager(_config())
        result = mgr.compute_new_sl("SELL", 2450.0, 2460.0, 2439.0, 2445.0)
        assert result.new_sl is None

    def test_zero_risk_distance_is_safe(self):
        mgr = TrailingStopManager(_config())
        result = mgr.compute_new_sl("BUY", 2450.0, 2450.0, 2470.0, 2450.0)
        assert result.new_sl is None
