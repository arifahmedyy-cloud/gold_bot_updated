"""Tests for regime_detector.py."""

import pandas as pd
import numpy as np
from src.trading.regime_detector import RegimeDetector, RegimeFeatures
from src.trading.indicators import TechnicalIndicators


class TestRegimeDetector:
    def test_extract_features(self, sample_ohlcv):
        detector = RegimeDetector()
        df = TechnicalIndicators.add_all(sample_ohlcv)
        features = detector.extract_features(df)
        assert isinstance(features, RegimeFeatures)
        assert 0 <= features.trend_score <= 100
        assert features.adx >= 0

    def test_classify_regime_strong_uptrend(self):
        detector = RegimeDetector()
        features = RegimeFeatures(
            trend_score=90, adx=50, volatility=0.15, bb_width=0.02,
            rsi=65, macd_hist=0.5, ema_alignment=2, price_vs_ema=1, volume_trend=30)
        regime, confidence = detector.classify_regime(features)
        assert regime == "strong_uptrend"
        assert confidence > 70

    def test_classify_regime_strong_downtrend(self):
        detector = RegimeDetector()
        features = RegimeFeatures(
            trend_score=10, adx=50, volatility=0.15, bb_width=0.02,
            rsi=35, macd_hist=-0.5, ema_alignment=-2, price_vs_ema=-1, volume_trend=30)
        regime, confidence = detector.classify_regime(features)
        assert regime == "strong_downtrend"
        assert confidence > 70

    def test_classify_regime_ranging(self):
        detector = RegimeDetector()
        features = RegimeFeatures(
            trend_score=50, adx=15, volatility=0.05, bb_width=0.02,
            rsi=50, macd_hist=0.0, ema_alignment=0, price_vs_ema=0, volume_trend=0)
        regime, confidence = detector.classify_regime(features)
        assert regime == "ranging"

    def test_generate_signal_buy(self):
        detector = RegimeDetector()
        df = pd.DataFrame({
            "Open": [2400]*50, "High": [2410]*50,
            "Low": [2390]*50, "Close": list(range(2400, 2450)),
            "Volume": [5000]*50})
        df = TechnicalIndicators.add_all(df)
        signal = detector.generate_signal(df)
        assert signal.action in ("BUY", "NO_TRADE")
        assert signal.confidence > 0

    def test_generate_signal_sell(self):
        detector = RegimeDetector()
        df = pd.DataFrame({
            "Open": [2450]*50, "High": [2460]*50,
            "Low": [2440]*50, "Close": list(range(2450, 2400, -1)),
            "Volume": [5000]*50})
        df = TechnicalIndicators.add_all(df)
        signal = detector.generate_signal(df)
        assert signal.action in ("SELL", "NO_TRADE")

    def test_analyze_integration(self, sample_ohlcv):
        detector = RegimeDetector()
        signal = detector.analyze(sample_ohlcv)
        assert isinstance(signal.confidence, int)
        assert 0 <= signal.confidence <= 100
        assert signal.regime != ""
