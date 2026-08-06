"""Tests for smc.py."""

import pytest
import pandas as pd
import numpy as np
from src.trading.smc import SMCAnalyzer, SwingPoint


class TestSMCAnalyzer:
    def test_analyze_empty_data(self):
        analyzer = SMCAnalyzer()
        df = pd.DataFrame({"Open": [], "High": [], "Low": [], "Close": []})
        result = analyzer.analyze(df)
        assert result.bias == "neutral"
        assert result.zone == "equilibrium"

    def test_detect_swing_points(self, sample_ohlcv):
        analyzer = SMCAnalyzer(swing_lookback=3)
        swings = analyzer.detect_swing_points(sample_ohlcv)
        assert len(swings) > 0
        for s in swings:
            assert s.price > 0
            assert s.type in ("high", "low")

    def test_detect_structure_events(self, sample_ohlcv):
        analyzer = SMCAnalyzer()
        swings = analyzer.detect_swing_points(sample_ohlcv)
        if len(swings) >= 4:
            events = analyzer.detect_structure_events(sample_ohlcv, swings)
            for e in events:
                assert e.type in ("BOS", "CHoCH")
                assert e.direction in ("bullish", "bearish")

    def test_detect_order_blocks(self, sample_ohlcv):
        analyzer = SMCAnalyzer()
        swings = analyzer.detect_swing_points(sample_ohlcv)
        obs = analyzer.detect_order_blocks(sample_ohlcv, swings)
        for ob in obs:
            assert ob.type in ("bullish", "bearish")
            assert ob.open_price > 0
            assert ob.high >= ob.low

    def test_detect_fvgs(self, sample_ohlcv):
        analyzer = SMCAnalyzer()
        fvgs = analyzer.detect_fvgs(sample_ohlcv)
        for f in fvgs:
            assert f.type in ("bullish", "bearish")

    def test_detect_liquidity_sweeps(self, sample_ohlcv):
        analyzer = SMCAnalyzer()
        swings = analyzer.detect_swing_points(sample_ohlcv)
        sweeps = analyzer.detect_liquidity_sweeps(sample_ohlcv, swings)
        for s in sweeps:
            assert s.type in ("buy", "sell")
            assert s.price > 0

    def test_determine_bias_bullish(self):
        analyzer = SMCAnalyzer()
        swings = [
            SwingPoint(index=0, price=100, type="low"),
            SwingPoint(index=1, price=110, type="high"),
            SwingPoint(index=2, price=105, type="low"),
            SwingPoint(index=3, price=120, type="high"),
        ]
        df = pd.DataFrame({"Open": [100], "High": [120], "Low": [100], "Close": [120]})
        bias, zone = analyzer._determine_bias_and_zone(df, swings, [], [])
        assert bias == "bullish"

    def test_determine_bias_bearish(self):
        analyzer = SMCAnalyzer()
        swings = [
            SwingPoint(index=0, price=120, type="high"),
            SwingPoint(index=1, price=110, type="low"),
            SwingPoint(index=2, price=115, type="high"),
            SwingPoint(index=3, price=100, type="low"),
        ]
        df = pd.DataFrame({"Open": [120], "High": [120], "Low": [100], "Close": [100]})
        bias, zone = analyzer._determine_bias_and_zone(df, swings, [], [])
        assert bias == "bearish"

    def test_caching(self, sample_ohlcv):
        analyzer = SMCAnalyzer()
        result1 = analyzer.analyze(sample_ohlcv)
        result2 = analyzer.analyze(sample_ohlcv)
        assert result1.bias == result2.bias
