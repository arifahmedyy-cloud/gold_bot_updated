"""Tests for risk_manager.py."""

import pytest
from src.trading.risk_manager import RiskManager
from src.config import RiskConfig


class TestRiskManager:
    def test_drawdown_adjustment_no_drawdown(self, risk_config):
        rm = RiskManager(risk_config)
        result = rm.compute_drawdown_adjusted_risk(10000, 10000, 1.0)
        assert result == 1.0

    def test_drawdown_adjustment_with_drawdown(self, risk_config):
        rm = RiskManager(risk_config)
        result = rm.compute_drawdown_adjusted_risk(9000, 10000, 1.0)
        expected = 1.0 * (1 - 0.1 * 0.5)
        assert abs(result - expected) < 0.01

    def test_drawdown_adjustment_max_reduction(self, risk_config):
        rm = RiskManager(risk_config)
        result = rm.compute_drawdown_adjusted_risk(1000, 10000, 1.0)
        assert result == 0.1

    def test_kelly_insufficient_history(self, risk_config):
        rm = RiskManager(risk_config)
        suggested, kelly = rm.kelly_position_size(10000, 2450, 2445, 1.0)
        assert suggested == 1.0
        assert kelly == 0.0

    def test_kelly_with_history(self, risk_config):
        rm = RiskManager(risk_config)
        for _ in range(15): rm.record_trade_result(100.0)
        for _ in range(5): rm.record_trade_result(-50.0)
        suggested, kelly = rm.kelly_position_size(10000, 2450, 2445, 1.0)
        assert suggested > 0
        assert kelly > 0
        assert suggested <= risk_config.max_risk_pct

    def test_lot_size_calculation(self, risk_config):
        rm = RiskManager(risk_config)
        lot = rm.calculate_lot_size(10000, 2450, 2445, 1.0)
        assert lot > 0
        assert lot <= 2.0

    def test_lot_size_zero_distance(self, risk_config):
        rm = RiskManager(risk_config)
        lot = rm.calculate_lot_size(10000, 2450, 2450, 1.0)
        assert lot == 0.01

    def test_daily_guard_no_breach(self, risk_config):
        rm = RiskManager(risk_config)
        guard = rm.daily_guard(10000, 10000)
        assert not guard.should_block_new_trades
        assert not guard.should_close_all

    def test_daily_guard_loss_limit(self, risk_config):
        rm = RiskManager(risk_config)
        for _ in range(10): rm.record_trade_result(-600.0)
        guard = rm.daily_guard(4000, 10000)
        assert guard.should_block_new_trades
        assert guard.should_close_all
        assert "Daily loss limit" in guard.reason

    def test_daily_guard_consecutive_losses(self, risk_config):
        rm = RiskManager(risk_config)
        for _ in range(3): rm.record_trade_result(-100.0)
        guard = rm.daily_guard(10000, 10000)
        assert guard.should_block_new_trades
        assert "consecutive losses" in guard.reason

    def test_validate_signal_valid(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 2445, 2460, 0.1, 10000, 70)
        assert valid
        assert reason == ""

    def test_validate_signal_low_ai_score(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 2445, 2460, 0.1, 10000, 40)
        assert not valid
        assert "AI score" in reason

    def test_validate_signal_poor_rr(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 2445, 2446, 0.1, 10000, 70)
        assert not valid
        assert "Reward" in reason

    def test_validate_signal_zero_sl(self, risk_config):
        rm = RiskManager(risk_config)
        valid, reason = rm.validate_signal(2450, 0, 2460, 0.1, 10000, 70)
        assert not valid
        assert "SL" in reason

    def test_risk_summary(self, risk_config):
        rm = RiskManager(risk_config)
        rm.record_trade_result(100.0)
        rm.record_trade_result(-50.0)
        summary = rm.get_risk_summary()
        assert summary["total_trades"] == 2
        assert summary["consecutive_losses"] == 1
