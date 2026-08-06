"""Tests for src/trading/mtf_filter.py and its DecisionEngine integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.trading.mtf_filter import MultiTimeframeFilter, MTFConfig, HTFBias, MTFResult
from src.trading.decision_engine import DecisionEngine
from src.models import SignalOutput, SMCResult


def _trending_htf_df(n: int = 120, direction: str = "up") -> pd.DataFrame:
    """Build a synthetic HTF OHLCV frame with a clear, unambiguous trend."""
    dates = pd.date_range(start="2024-01-01", periods=n, freq="4h")
    step = 1.0 if direction == "up" else -1.0
    closes = 2400.0 + np.arange(n) * step
    df = pd.DataFrame({
        "Open": closes - 0.2,
        "High": closes + 0.5,
        "Low": closes - 0.5,
        "Close": closes,
        "Volume": np.full(n, 5000),
    }, index=dates)
    return df


def _flat_htf_df(n: int = 120) -> pd.DataFrame:
    """Build a synthetic HTF OHLCV frame with no real trend (flat/noisy)."""
    np.random.seed(7)
    dates = pd.date_range(start="2024-01-01", periods=n, freq="4h")
    closes = 2400.0 + np.random.randn(n) * 0.05
    df = pd.DataFrame({
        "Open": closes,
        "High": closes + 0.1,
        "Low": closes - 0.1,
        "Close": closes,
        "Volume": np.full(n, 5000),
    }, index=dates)
    return df


def _signal(action: str = "BUY", confidence: int = 70) -> SignalOutput:
    return SignalOutput(
        action=action, confidence=confidence, regime="trend", strategy="TestStrategy",
        expected_pf=1.5, expected_max_dd=0.1, expected_avg_rr=2.0, consistency_score=70,
        sl=2440.0, tp=2460.0, entry=2450.0, lot_size=0.1, explanation="test signal",
        metrics={},
    )


def _neutral_smc() -> SMCResult:
    return SMCResult(bias="neutral", zone="equilibrium")


class TestMTFConfig:
    def test_defaults_are_enabled_and_non_blocking(self):
        cfg = MTFConfig()
        assert cfg.enabled is True
        assert cfg.block_on_conflict is False

    def test_htf_ltf_timeframes_are_configurable(self):
        cfg = MTFConfig(htf_timeframe="D1", ltf_timeframe="M5")
        assert cfg.htf_timeframe == "D1"
        assert cfg.ltf_timeframe == "M5"

    def test_confirmation_strength_is_configurable(self):
        cfg = MTFConfig(slope_threshold=0.5)
        assert cfg.slope_threshold == 0.5

    def test_confidence_bonus_and_penalty_are_configurable(self):
        cfg = MTFConfig(agreement_bonus=20, conflict_penalty=40)
        assert cfg.agreement_bonus == 20
        assert cfg.conflict_penalty == 40


class TestComputeBias:
    def test_insufficient_bars_returns_none(self):
        mtf = MultiTimeframeFilter(MTFConfig(min_htf_bars=60))
        short_df = _trending_htf_df(n=10, direction="up")
        assert mtf.compute_bias(short_df) is None

    def test_none_or_empty_df_returns_none(self):
        mtf = MultiTimeframeFilter()
        assert mtf.compute_bias(None) is None
        assert mtf.compute_bias(pd.DataFrame()) is None

    def test_uptrend_detected_as_bullish(self):
        mtf = MultiTimeframeFilter(MTFConfig(slope_threshold=0.01))
        bias = mtf.compute_bias(_trending_htf_df(direction="up"))
        assert isinstance(bias, HTFBias)
        assert bias.direction == "bullish"

    def test_downtrend_detected_as_bearish(self):
        mtf = MultiTimeframeFilter(MTFConfig(slope_threshold=0.01))
        bias = mtf.compute_bias(_trending_htf_df(direction="down"))
        assert bias.direction == "bearish"

    def test_flat_market_detected_as_neutral(self):
        mtf = MultiTimeframeFilter(MTFConfig(slope_threshold=0.5))
        bias = mtf.compute_bias(_flat_htf_df())
        assert bias.direction == "neutral"


class TestAdjust:
    def test_agreement_gives_bonus(self):
        cfg = MTFConfig(agreement_bonus=10, conflict_penalty=25)
        mtf = MultiTimeframeFilter(cfg)
        bias = HTFBias(direction="bullish", ema20_slope=0.5, price_vs_ema50=0.3)
        result = mtf.adjust("BUY", bias)
        assert result.score_delta == 10
        assert result.hard_block is False
        assert "agrees" in result.note

    def test_conflict_gives_penalty_but_does_not_block_by_default(self):
        cfg = MTFConfig(agreement_bonus=10, conflict_penalty=25)
        mtf = MultiTimeframeFilter(cfg)
        bias = HTFBias(direction="bearish", ema20_slope=-0.5, price_vs_ema50=-0.3)
        result = mtf.adjust("BUY", bias)
        assert result.score_delta == -25
        assert result.hard_block is False  # never blocks unless configured
        assert "conflicts" in result.note

    def test_conflict_hard_blocks_when_explicitly_configured(self):
        cfg = MTFConfig(block_on_conflict=True)
        mtf = MultiTimeframeFilter(cfg)
        bias = HTFBias(direction="bearish", ema20_slope=-0.5, price_vs_ema50=-0.3)
        result = mtf.adjust("BUY", bias)
        assert result.hard_block is True

    def test_neutral_bias_gives_no_adjustment(self):
        mtf = MultiTimeframeFilter()
        bias = HTFBias(direction="neutral", ema20_slope=0.02, price_vs_ema50=0.01)
        result = mtf.adjust("BUY", bias)
        assert result.score_delta == 0
        assert result.hard_block is False

    def test_missing_bias_gives_no_adjustment_and_no_block(self):
        mtf = MultiTimeframeFilter()
        result = mtf.adjust("BUY", None)
        assert result.score_delta == 0
        assert result.hard_block is False

    def test_disabled_filter_is_a_pure_noop_even_with_block_on_conflict(self):
        cfg = MTFConfig(enabled=False, block_on_conflict=True)
        mtf = MultiTimeframeFilter(cfg)
        bias = HTFBias(direction="bearish", ema20_slope=-0.5, price_vs_ema50=-0.3)
        result = mtf.adjust("BUY", bias)
        assert result.score_delta == 0
        assert result.hard_block is False


class TestEvaluate:
    def test_evaluate_combines_bias_and_adjust(self):
        cfg = MTFConfig(slope_threshold=0.01)
        mtf = MultiTimeframeFilter(cfg)
        result = mtf.evaluate("BUY", _trending_htf_df(direction="up"))
        assert isinstance(result, MTFResult)
        assert result.score_delta > 0

    def test_evaluate_disabled_skips_bias_computation_entirely(self):
        mtf = MultiTimeframeFilter(MTFConfig(enabled=False))
        result = mtf.evaluate("BUY", None)  # would crash compute_bias if it ran
        assert result.score_delta == 0
        assert result.hard_block is False


class TestDecisionEngineIntegration:
    def test_decide_ignores_mtf_when_not_provided(self, risk_config):
        engine = DecisionEngine(risk_config)
        decision = engine.decide(_signal(confidence=80), _neutral_smc())
        assert decision.action == "BUY"

    def test_decide_applies_mtf_positive_delta(self, risk_config):
        engine = DecisionEngine(risk_config)
        decision = engine.decide(
            _signal(confidence=50), _neutral_smc(),
            mtf_score_delta=10, mtf_note="HTF agrees (+10)",
        )
        assert "HTF agrees (+10)" in decision.confluence_notes

    def test_decide_applies_mtf_negative_delta_can_drop_below_threshold(self, risk_config):
        engine = DecisionEngine(risk_config)
        decision = engine.decide(
            _signal(confidence=56), _neutral_smc(),
            mtf_score_delta=-25, mtf_note="HTF conflicts (-25)",
        )
        assert decision.action == "NO_TRADE"

    def test_decide_forces_no_trade_on_mtf_hard_block(self, risk_config):
        engine = DecisionEngine(risk_config)
        decision = engine.decide(
            _signal(confidence=95), _neutral_smc(),
            mtf_score_delta=-25, mtf_note="HTF conflicts", mtf_hard_block=True,
        )
        assert decision.action == "NO_TRADE"
        assert "Blocked by MTF filter" in decision.explanation

    def test_decide_never_hard_blocks_by_default(self, risk_config):
        """Default decide() calls (no mtf_hard_block passed) must behave
        exactly as before this migration — regression guard."""
        engine = DecisionEngine(risk_config)
        decision = engine.decide(_signal(confidence=90), _neutral_smc())
        assert decision.action == "BUY"

    def test_full_pipeline_evaluate_then_decide(self, risk_config):
        """End-to-end: MultiTimeframeFilter.evaluate() feeds directly into
        DecisionEngine.decide() the way the live pipeline would use it."""
        mtf = MultiTimeframeFilter(MTFConfig(slope_threshold=0.01))
        mtf_result = mtf.evaluate("BUY", _trending_htf_df(direction="up"))

        engine = DecisionEngine(risk_config)
        decision = engine.decide(
            _signal(action="BUY", confidence=50), _neutral_smc(),
            mtf_score_delta=mtf_result.score_delta,
            mtf_note=mtf_result.note,
            mtf_hard_block=mtf_result.hard_block,
        )
        assert decision.action == "BUY"
        assert mtf_result.note in decision.confluence_notes
