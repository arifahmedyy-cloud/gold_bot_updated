"""Pytest fixtures and configuration."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.config import RiskConfig, MT5Config, BotConfig
from src.models import TradeSignal, Position, AccountInfo


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="h")
    base = 2400.0
    prices = base + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "Date": dates,
        "Open": prices + np.random.randn(n) * 0.2,
        "High": prices + abs(np.random.randn(n)) * 0.5 + 0.1,
        "Low": prices - abs(np.random.randn(n)) * 0.5 - 0.1,
        "Close": prices + np.random.randn(n) * 0.1,
        "Volume": np.random.randint(1000, 10000, n),
    })
    df.set_index("Date", inplace=True)
    return df


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        base_risk_pct=1.0, max_risk_pct=2.0, daily_loss_limit_pct=5.0,
        max_consecutive_losses=3, use_kelly_sizing=False,
        use_auto_drawdown_risk=True, min_ai_score=55, max_open_trades=1,
    )


@pytest.fixture
def mt5_config() -> MT5Config:
    return MT5Config(login=12345, password="test_password",
                     server="TestServer-Demo", leverage=100.0)


@pytest.fixture
def bot_config(risk_config, mt5_config) -> BotConfig:
    return BotConfig(broker="paper", mt5=mt5_config, risk=risk_config)


@pytest.fixture
def sample_trade_signal() -> TradeSignal:
    return TradeSignal(
        symbol="XAUUSD", direction="BUY", entry_price=2450.0,
        sl=2445.0, tp=2460.0, lot_size=0.1, strategy="TestStrategy",
    )


@pytest.fixture
def sample_account() -> AccountInfo:
    return AccountInfo(
        balance=10000.0, equity=10000.0, margin=0.0,
        free_margin=10000.0, margin_level=100.0, leverage=100.0,
    )
