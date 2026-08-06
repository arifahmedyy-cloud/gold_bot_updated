"""Tests for decision_engine.py."""

from src.trading.decision_engine import DecisionEngine
from src.models import SignalOutput, SMCResult
from src.config import RiskConfig


class TestDecisionEngine:
    def test_decide_no_trade_low_score(self):
        config = RiskConfig(min_ai_score=60)
        engine = DecisionEngine(config)
        signal = SignalOutput(
            action="BUY", confidence=50, regime="strong_uptrend",
            strategy="Test", expected_pf=1.5, expected_max_dd=5.0,
            expected_avg_rr=2.0, consistency_score=50,
            sl=2445, tp=2460, entry=2450, lot_size=0.1,
            explanation="Test", metrics={})
        smc = SMCResult(bias="bullish", zone="discount")
        decision = engine.decide(signal, smc)
        assert decision.action == "NO_TRADE"

    def test_decide_buy_bullish_confluence(self):
        config = RiskConfig(min_ai_score=55)
        engine = DecisionEngine(config)
        signal = SignalOutput(
            action="BUY", confidence=80, regime="strong_uptrend",
            strategy="Test", expected_pf=1.5, expected_max_dd=5.0,
            expected_avg_rr=2.0, consistency_score=80,
            sl=2445, tp=2460, entry=2450, lot_size=0.1,
            explanation="Test", metrics={})
        smc = SMCResult(bias="bullish", zone="discount")
        decision = engine.decide(signal, smc)
        assert decision.action == "BUY"

    def test_decide_sell_bearish_confluence(self):
        config = RiskConfig(min_ai_score=55)
        engine = DecisionEngine(config)
        signal = SignalOutput(
            action="SELL", confidence=80, regime="strong_downtrend",
            strategy="Test", expected_pf=1.5, expected_max_dd=5.0,
            expected_avg_rr=2.0, consistency_score=80,
            sl=2455, tp=2440, entry=2450, lot_size=0.1,
            explanation="Test", metrics={})
        smc = SMCResult(bias="bearish", zone="premium")
        decision = engine.decide(signal, smc)
        assert decision.action == "SELL"

    def test_decide_contradiction_reduces_score(self):
        config = RiskConfig(min_ai_score=55)
        engine = DecisionEngine(config)
        signal = SignalOutput(
            action="BUY", confidence=65, regime="weak_uptrend",
            strategy="Test", expected_pf=1.5, expected_max_dd=5.0,
            expected_avg_rr=2.0, consistency_score=65,
            sl=2445, tp=2460, entry=2450, lot_size=0.1,
            explanation="Test", metrics={})
        smc = SMCResult(bias="bearish", zone="premium")
        decision = engine.decide(signal, smc)
        assert decision.action == "NO_TRADE"

    def test_decide_ml_filter_blocks(self):
        config = RiskConfig(min_ai_score=55, enable_ml_filter=True, min_ml_confidence=70)
        engine = DecisionEngine(config)
        signal = SignalOutput(
            action="BUY", confidence=80, regime="strong_uptrend",
            strategy="Test", expected_pf=1.5, expected_max_dd=5.0,
            expected_avg_rr=2.0, consistency_score=80,
            sl=2445, tp=2460, entry=2450, lot_size=0.1,
            explanation="Test", metrics={})
        smc = SMCResult(bias="bullish")
        decision = engine.decide(signal, smc, ml_confidence=50)
        assert decision.action == "NO_TRADE"

    def test_decide_ml_filter_passes(self):
        config = RiskConfig(min_ai_score=55, enable_ml_filter=True, min_ml_confidence=60)
        engine = DecisionEngine(config)
        signal = SignalOutput(
            action="BUY", confidence=70, regime="strong_uptrend",
            strategy="Test", expected_pf=1.5, expected_max_dd=5.0,
            expected_avg_rr=2.0, consistency_score=70,
            sl=2445, tp=2460, entry=2450, lot_size=0.1,
            explanation="Test", metrics={})
        smc = SMCResult(bias="bullish")
        decision = engine.decide(signal, smc, ml_confidence=80)
        assert decision.action == "BUY"
        assert decision.ai_score == 75

    def test_decide_premium_zone_warning(self):
        config = RiskConfig(min_ai_score=55)
        engine = DecisionEngine(config)
        signal = SignalOutput(
            action="BUY", confidence=80, regime="strong_uptrend",
            strategy="Test", expected_pf=1.5, expected_max_dd=5.0,
            expected_avg_rr=2.0, consistency_score=80,
            sl=2445, tp=2460, entry=2450, lot_size=0.1,
            explanation="Test", metrics={})
        smc = SMCResult(bias="bullish", zone="premium")
        decision = engine.decide(signal, smc)
        assert "premium zone" in decision.explanation
        assert decision.ai_score == 70
