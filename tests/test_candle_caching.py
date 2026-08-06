"""Tests for candle-based caching in app._run_symbol_cycle — indicators,
regime/strategy, and SMC should only be recomputed when a new candle
appears, not on every 3-second auto-refresh poll."""

import sys
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np
import pytest

# Import app.py as a module without running its __main__ block.
_APP_PATH = Path(__file__).resolve().parent.parent / "app.py"
_spec = importlib.util.spec_from_file_location("gold_bot_app", _APP_PATH)
app = importlib.util.module_from_spec(_spec)
sys.modules["gold_bot_app"] = app
_spec.loader.exec_module(app)


def _make_df(n=60, last_ts="2024-01-01 10:00"):
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    dates = dates[:-1].append(pd.DatetimeIndex([pd.Timestamp(last_ts)]))
    rng = np.random.default_rng(1)
    close = 2450 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({
        "Open": close, "High": close + 1, "Low": close - 1,
        "Close": close, "Volume": 100,
    }, index=dates)


def _mock_deps():
    broker = MagicMock()
    broker.is_connected.return_value = False
    broker.get_price.return_value = {"bid": 2450.0, "ask": 2450.5}
    broker.get_positions.return_value = []

    journal = MagicMock()
    health = MagicMock()
    notifier = MagicMock()

    regime = MagicMock()
    regime.generate_signal.return_value = MagicMock(
        action="NO_TRADE", confidence=50, regime="unclear", entry=2450, sl=2440, tp=2460)

    smc = MagicMock()
    smc.analyze.return_value = MagicMock(bias="neutral", zone="equilibrium")

    decision = MagicMock()
    decision.decide.return_value = MagicMock(
        action="NO_TRADE", entry=2450, sl=2440, tp=2460, ai_score=50,
        explanation="no confluence", confluence_notes=[])

    risk = MagicMock()
    symbol_manager = app.SymbolManager()
    correlation_guard = app.CorrelationGuard(symbol_manager)
    account = MagicMock(balance=10000.0, leverage=100.0)

    config = MagicMock()
    config.broker = "paper"
    config.risk.max_open_trades = 1

    return dict(
        symbol="XAUUSD", config=config, broker=broker, journal=journal,
        notifier=notifier, health=health, regime=regime, smc=smc,
        decision=decision, risk=risk, news_service=None, ai_service=None,
        symbol_manager=symbol_manager, correlation_guard=correlation_guard,
        account=account, live_strategy_name="Regime (default)",
    )


class TestCandleCaching:
    def test_same_candle_reuses_cache(self):
        deps = _mock_deps()
        df = _make_df()
        cache = {}

        with patch.object(app, "DataService") as MockDS:
            MockDS.return_value.fetch_yfinance.return_value = df
            app._run_symbol_cycle(signal_cache=cache, **deps)
            app._run_symbol_cycle(signal_cache=cache, **deps)

        # regime.generate_signal should only be called ONCE — second call
        # reused the cached analysis since the candle didn't change.
        assert deps["regime"].generate_signal.call_count == 1
        assert deps["smc"].analyze.call_count == 1

    def test_new_candle_invalidates_cache(self):
        deps = _mock_deps()
        df1 = _make_df(last_ts="2024-01-01 10:00")
        df2 = _make_df(last_ts="2024-01-01 11:00")  # one hour later -> new candle
        cache = {}

        with patch.object(app, "DataService") as MockDS:
            MockDS.return_value.fetch_yfinance.return_value = df1
            app._run_symbol_cycle(signal_cache=cache, **deps)
            MockDS.return_value.fetch_yfinance.return_value = df2
            app._run_symbol_cycle(signal_cache=cache, **deps)

        assert deps["regime"].generate_signal.call_count == 2
        assert deps["smc"].analyze.call_count == 2

    def test_strategy_switch_invalidates_cache_even_on_same_candle(self):
        deps = _mock_deps()
        df = _make_df()
        cache = {}

        with patch.object(app, "DataService") as MockDS, \
             patch.object(app, "get_strategy") as mock_get_strategy:
            MockDS.return_value.fetch_yfinance.return_value = df
            mock_strategy = MagicMock()
            from src.trading.strategies import StrategySignal
            mock_strategy.generate.return_value = StrategySignal(
                action="BUY", entry=2450, sl=2440, tp=2470, confidence=70, explanation="x")
            mock_get_strategy.return_value = mock_strategy

            deps["live_strategy_name"] = "Regime (default)"
            app._run_symbol_cycle(signal_cache=cache, **deps)

            deps["live_strategy_name"] = "EMA Trend"
            app._run_symbol_cycle(signal_cache=cache, **deps)

        # regime path used once, strategy path used once — different
        # cache_key (candle, strategy_name) forces a fresh computation.
        assert deps["regime"].generate_signal.call_count == 1
        assert mock_strategy.generate.call_count == 1
