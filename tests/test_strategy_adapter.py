"""Tests for strategies.py's strategy_signal_to_signal_output adapter —
the bridge that lets live trading use any strategies.py strategy through
the same decision_engine pipeline RegimeDetector already feeds."""

import pandas as pd
import numpy as np

from src.trading.strategies import (
    StrategySignal, strategy_signal_to_signal_output, get_strategy, STRATEGIES,
)
from src.trading.indicators import TechnicalIndicators


class TestStrategyAdapter:
    def test_buy_signal_maps_correctly(self):
        sig = StrategySignal(action="BUY", entry=2450.0, sl=2440.0, tp=2470.0,
                              confidence=70.0, explanation="test buy")
        out = strategy_signal_to_signal_output(sig, "EMA Trend")
        assert out.action == "BUY"
        assert out.confidence == 70
        assert out.entry == 2450.0
        assert out.sl == 2440.0
        assert out.tp == 2470.0
        assert out.strategy == "EMA Trend"
        assert out.regime == "strategy:EMA Trend"

    def test_hold_maps_to_no_trade(self):
        sig = StrategySignal(action="HOLD", entry=2450.0, sl=2450.0, tp=2450.0,
                              confidence=0.0, explanation="no signal")
        out = strategy_signal_to_signal_output(sig, "RSI Mean Reversion")
        assert out.action == "NO_TRADE"

    def test_confidence_clamped_to_valid_range(self):
        sig = StrategySignal(action="BUY", entry=1.0, sl=0.9, tp=1.2,
                              confidence=150.0, explanation="overconfident")
        out = strategy_signal_to_signal_output(sig, "Breakout")
        assert out.confidence == 100

    def test_all_registered_strategies_produce_valid_signal_output(self):
        # Build a small but valid OHLCV frame with enough bars for every
        # strategy's indicators (EMA50, Bollinger, etc.) to compute.
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        rng = np.random.default_rng(42)
        close = 2450 + np.cumsum(rng.normal(0, 2, n))
        df = pd.DataFrame({
            "Open": close, "High": close + 2, "Low": close - 2,
            "Close": close, "Volume": rng.integers(100, 1000, n),
        }, index=dates)
        df_ind = TechnicalIndicators.add_all(df)

        for name in STRATEGIES:
            strategy = get_strategy(name)
            sig = strategy.generate(df_ind)
            out = strategy_signal_to_signal_output(sig, name)
            assert out.action in ("BUY", "SELL", "NO_TRADE")
            assert 0 <= out.confidence <= 100
            assert out.strategy == name
