"""Market regime detection using multi-timeframe analysis.

Classifies market state into trend/range/volatility regimes.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.models import SignalOutput
from src.constants import Regime
from src.trading.indicators import TechnicalIndicators

log = get_logger(__name__)


@dataclass
class RegimeFeatures:
    """Features extracted for regime classification."""
    trend_score: float
    adx: float
    volatility: float
    bb_width: float
    rsi: float
    macd_hist: float
    ema_alignment: int
    price_vs_ema: int
    volume_trend: float


class RegimeDetector:
    """Detects market regime from OHLCV data."""

    def __init__(self, symbol: str = "XAUUSD", timeframe: str = "H1") -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self._last_regime: Optional[str] = None
        self._last_confidence: int = 0

    def extract_features(self, df: pd.DataFrame) -> RegimeFeatures:
        """Extract regime classification features."""
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else last

        ema10 = last.get("EMA_10")
        ema20 = last.get("EMA_20")
        ema50 = last.get("EMA_50")
        ema200 = last.get("EMA_200")
        close = float(last["Close"])

        ema_alignment = 0
        if pd.notna(ema10) and pd.notna(ema20) and pd.notna(ema50) and pd.notna(ema200):
            if ema10 > ema20 > ema50 > ema200:
                ema_alignment = 2
            elif ema10 < ema20 < ema50 < ema200:
                ema_alignment = -2
            elif ema20 > ema50:
                ema_alignment = 1
            elif ema20 < ema50:
                ema_alignment = -1

        price_vs_ema = 0
        if pd.notna(ema20) and close > ema20:
            price_vs_ema = 1
        elif pd.notna(ema20) and close < ema20:
            price_vs_ema = -1

        volume_trend = 0.0
        if "Volume" in df.columns and len(df) >= 20:
            vol_now = df["Volume"].iloc[-5:].mean()
            vol_prev = df["Volume"].iloc[-20:-5].mean()
            if vol_prev > 0:
                volume_trend = (vol_now - vol_prev) / vol_prev * 100

        return RegimeFeatures(
            trend_score=float(last.get("Trend_Score", 50)),
            adx=float(last.get("ADX", 20)),
            volatility=float(last.get("Volatility", 0.15)),
            bb_width=float(last.get("BB_Width", 0.02)),
            rsi=float(last.get("RSI", 50)),
            macd_hist=float(last.get("MACD_Hist", 0)),
            ema_alignment=ema_alignment,
            price_vs_ema=price_vs_ema,
            volume_trend=volume_trend,
        )

    def classify_regime(self, features: RegimeFeatures) -> Tuple[str, int]:
        """Classify regime from features.

        Returns:
            Tuple of (regime_name, confidence_0_100).
        """
        regime = Regime.UNCLEAR
        confidence = 50

        if features.adx > 40 and features.ema_alignment in (2, -2):
            regime = Regime.STRONG_UPTREND if features.ema_alignment == 2 else Regime.STRONG_DOWNTREND
            confidence = min(95, 70 + features.adx / 2)
        elif features.adx > 25 and features.ema_alignment in (1, -1):
            regime = Regime.WEAK_UPTREND if features.ema_alignment == 1 else Regime.WEAK_DOWNTREND
            confidence = min(85, 60 + features.adx)
        elif features.adx < 20 and features.bb_width < 0.03:
            regime = Regime.RANGING
            confidence = min(80, 60 + (25 - features.adx))
        elif features.volatility > 0.25 and features.bb_width > 0.05:
            regime = Regime.HIGH_VOLATILITY_BREAKOUT
            confidence = min(85, 65 + features.volatility * 100)
        elif features.volatility < 0.08:
            regime = Regime.LOW_VOLATILITY
            confidence = min(75, 60 + (0.08 - features.volatility) * 500)

        # Adjust confidence by volume confirmation
        if abs(features.volume_trend) > 20:
            confidence = min(95, confidence + 5)

        return regime.value, int(confidence)

    def generate_signal(self, df: pd.DataFrame) -> SignalOutput:
        """Generate regime-based signal.

        Args:
            df: DataFrame with indicators.

        Returns:
            SignalOutput with action, SL, TP, and regime info.
        """
        features = self.extract_features(df)
        regime, confidence = self.classify_regime(features)

        last = df.iloc[-1]
        close = float(last["Close"])
        atr = float(last.get("ATR_14", close * 0.005))

        action = "NO_TRADE"
        sl = close
        tp = close
        entry = close
        explanation = f"Regime: {regime} (confidence={confidence})"

        if regime in (Regime.STRONG_UPTREND.value, Regime.WEAK_UPTREND.value):
            if features.price_vs_ema >= 0 and features.macd_hist >= 0:
                action = "BUY"
                sl = close - atr * 1.5
                tp = close + atr * 3.0
                entry = close
                explanation += " | Pullback buy in uptrend"
        elif regime in (Regime.STRONG_DOWNTREND.value, Regime.WEAK_DOWNTREND.value):
            if features.price_vs_ema <= 0 and features.macd_hist <= 0:
                action = "SELL"
                sl = close + atr * 1.5
                tp = close - atr * 3.0
                entry = close
                explanation += " | Pullback sell in downtrend"
        elif regime == Regime.HIGH_VOLATILITY_BREAKOUT.value:
            if features.macd_hist > 0:
                action = "BUY"
                sl = close - atr * 2.0
                tp = close + atr * 4.0
                explanation += " | Volatility breakout long"
            elif features.macd_hist < 0:
                action = "SELL"
                sl = close + atr * 2.0
                tp = close - atr * 4.0
                explanation += " | Volatility breakout short"

        return SignalOutput(
            action=action,
            confidence=confidence,
            regime=regime,
            strategy="RegimeDetector",
            expected_pf=1.5,
            expected_max_dd=5.0,
            expected_avg_rr=2.0,
            consistency_score=confidence,
            sl=round(sl, 2),
            tp=round(tp, 2),
            entry=round(entry, 2),
            lot_size=0.01,
            explanation=explanation,
            metrics={
                "trend_score": features.trend_score,
                "adx": features.adx,
                "volatility": features.volatility,
                "bb_width": features.bb_width,
                "rsi": features.rsi,
                "macd_hist": features.macd_hist,
                "ema_alignment": features.ema_alignment,
                "volume_trend": features.volume_trend,
            },
        )

    def analyze(self, df: pd.DataFrame) -> SignalOutput:
        """Main entry point — add indicators then detect regime."""
        df_with_indicators = TechnicalIndicators.add_all(df)
        return self.generate_signal(df_with_indicators)
