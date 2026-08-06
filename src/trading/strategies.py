"""Trading strategies for backtesting and signal generation.

Provides the strategy interface expected by legacy dashboard imports.
Each strategy generates signals based on indicator conditions.
"""

from __future__ import annotations

from typing import Dict, Any, List
from dataclasses import dataclass
import pandas as pd

from src.logger import get_logger
from src.models import SignalOutput

log = get_logger(__name__)


@dataclass
class StrategySignal:
    """Output from a strategy."""
    action: str  # BUY, SELL, HOLD
    entry: float
    sl: float
    tp: float
    confidence: float
    explanation: str


class BaseStrategy:
    """Base class for all strategies."""
    name: str = "Base"

    def generate(self, df: pd.DataFrame, idx: int = -1) -> StrategySignal:
        raise NotImplementedError


class EMATrendStrategy(BaseStrategy):
    """EMA crossover trend following."""
    name = "EMA Trend"

    def generate(self, df: pd.DataFrame, idx: int = -1) -> StrategySignal:
        row = df.iloc[idx]
        close = float(row['Close'])
        action = "HOLD"
        explanation = "No clear EMA trend"

        ema20 = row.get('EMA_20')
        ema50 = row.get('EMA_50')
        if pd.notna(ema20) and pd.notna(ema50):
            if ema20 > ema50 and close > ema20:
                action = "BUY"
                explanation = "EMA20 > EMA50 and price above EMA20"
            elif ema20 < ema50 and close < ema20:
                action = "SELL"
                explanation = "EMA20 < EMA50 and price below EMA20"

        atr = row.get('ATR_14', close * 0.005)
        sl = close - atr * 1.5 if action == "BUY" else close + atr * 1.5 if action == "SELL" else close
        tp = close + atr * 3.0 if action == "BUY" else close - atr * 3.0 if action == "SELL" else close

        return StrategySignal(action, close, sl, tp, 70.0, explanation)


class MACDStrategy(BaseStrategy):
    """MACD momentum strategy."""
    name = "MACD Signal"

    def generate(self, df: pd.DataFrame, idx: int = -1) -> StrategySignal:
        row = df.iloc[idx]
        close = float(row['Close'])
        action = "HOLD"
        explanation = "No MACD signal"

        macd = row.get('MACD')
        signal = row.get('MACD_Signal')
        if pd.notna(macd) and pd.notna(signal):
            if macd > signal:
                action = "BUY"
                explanation = "MACD above signal line"
            elif macd < signal:
                action = "SELL"
                explanation = "MACD below signal line"

        atr = row.get('ATR_14', close * 0.005)
        sl = close - atr * 1.5 if action == "BUY" else close + atr * 1.5 if action == "SELL" else close
        tp = close + atr * 3.0 if action == "BUY" else close - atr * 3.0 if action == "SELL" else close
        return StrategySignal(action, close, sl, tp, 65.0, explanation)


class RSIStrategy(BaseStrategy):
    """RSI mean reversion."""
    name = "RSI Mean Reversion"

    def generate(self, df: pd.DataFrame, idx: int = -1) -> StrategySignal:
        row = df.iloc[idx]
        close = float(row['Close'])
        rsi = row.get('RSI')
        action = "HOLD"
        explanation = "RSI neutral"

        if pd.notna(rsi):
            if rsi < 30:
                action = "BUY"
                explanation = f"RSI oversold ({rsi:.1f})"
            elif rsi > 70:
                action = "SELL"
                explanation = f"RSI overbought ({rsi:.1f})"

        atr = row.get('ATR_14', close * 0.005)
        sl = close - atr * 2.0 if action == "BUY" else close + atr * 2.0 if action == "SELL" else close
        tp = close + atr * 2.0 if action == "BUY" else close - atr * 2.0 if action == "SELL" else close
        return StrategySignal(action, close, sl, tp, 60.0, explanation)


class BollingerStrategy(BaseStrategy):
    """Bollinger Bands mean reversion."""
    name = "Bollinger Reversion"

    def generate(self, df: pd.DataFrame, idx: int = -1) -> StrategySignal:
        row = df.iloc[idx]
        close = float(row['Close'])
        action = "HOLD"
        explanation = "Price within bands"

        lower = row.get('BB_Lower')
        upper = row.get('BB_Upper')
        if pd.notna(lower) and pd.notna(upper):
            if close <= lower:
                action = "BUY"
                explanation = "Price at lower Bollinger Band"
            elif close >= upper:
                action = "SELL"
                explanation = "Price at upper Bollinger Band"

        sl = close - (close * 0.01) if action == "BUY" else close + (close * 0.01) if action == "SELL" else close
        tp = close + (close * 0.02) if action == "BUY" else close - (close * 0.02) if action == "SELL" else close
        return StrategySignal(action, close, sl, tp, 62.0, explanation)


class BreakoutStrategy(BaseStrategy):
    """Price breakout strategy."""
    name = "Breakout"

    def generate(self, df: pd.DataFrame, idx: int = -1) -> StrategySignal:
        row = df.iloc[idx]
        close = float(row['Close'])
        action = "HOLD"
        explanation = "No breakout"

        high_20 = df['High'].iloc[max(0, idx-20):idx].max() if idx != -1 else df['High'].iloc[-20:].max()
        low_20 = df['Low'].iloc[max(0, idx-20):idx].min() if idx != -1 else df['Low'].iloc[-20:].min()

        if pd.notna(high_20) and close > high_20:
            action = "BUY"
            explanation = "Price broke above 20-period high"
        elif pd.notna(low_20) and close < low_20:
            action = "SELL"
            explanation = "Price broke below 20-period low"

        atr = row.get('ATR_14', close * 0.005)
        sl = close - atr * 2.0 if action == "BUY" else close + atr * 2.0 if action == "SELL" else close
        tp = close + atr * 4.0 if action == "BUY" else close - atr * 4.0 if action == "SELL" else close
        return StrategySignal(action, close, sl, tp, 68.0, explanation)


class CombinedStrategy(BaseStrategy):
    """EMA + RSI combined confirmation."""
    name = "Combined EMA+RSI"

    def generate(self, df: pd.DataFrame, idx: int = -1) -> StrategySignal:
        row = df.iloc[idx]
        close = float(row['Close'])
        action = "HOLD"
        explanation = "No confluence"

        ema20 = row.get('EMA_20')
        ema50 = row.get('EMA_50')
        rsi = row.get('RSI')

        if pd.notna(ema20) and pd.notna(ema50) and pd.notna(rsi):
            if ema20 > ema50 and close > ema20 and rsi > 50 and rsi < 80:
                action = "BUY"
                explanation = "EMA uptrend + RSI confirmation"
            elif ema20 < ema50 and close < ema20 and rsi < 50 and rsi > 20:
                action = "SELL"
                explanation = "EMA downtrend + RSI confirmation"

        atr = row.get('ATR_14', close * 0.005)
        sl = close - atr * 1.5 if action == "BUY" else close + atr * 1.5 if action == "SELL" else close
        tp = close + atr * 3.0 if action == "BUY" else close - atr * 3.0 if action == "SELL" else close
        return StrategySignal(action, close, sl, tp, 75.0, explanation)


_STRATEGIES: Dict[str, BaseStrategy] = {
    "EMA Trend": EMATrendStrategy(),
    "MACD Signal": MACDStrategy(),
    "RSI Mean Reversion": RSIStrategy(),
    "Bollinger Reversion": BollingerStrategy(),
    "Breakout": BreakoutStrategy(),
    "Combined EMA+RSI": CombinedStrategy(),
}

STRATEGIES: List[str] = list(_STRATEGIES.keys())


def get_strategy(name: str) -> BaseStrategy:
    """Get strategy instance by name."""
    if name not in _STRATEGIES:
        log.warning("Strategy %s not found, using EMA Trend", name)
        return _STRATEGIES["EMA Trend"]
    return _STRATEGIES[name]


def strategy_signal_to_signal_output(sig: StrategySignal, strategy_name: str) -> SignalOutput:
    """Adapt a strategies.py StrategySignal into the SignalOutput shape the
    live decision_engine already expects from RegimeDetector.

    This lets the live trading loop plug any strategy from STRATEGIES in
    place of RegimeDetector without any change to decision_engine.py,
    risk_manager.py, or the rest of the pipeline downstream.
    """
    action = sig.action if sig.action in ("BUY", "SELL") else "NO_TRADE"
    confidence = int(max(0, min(100, sig.confidence)))
    return SignalOutput(
        action=action,
        confidence=confidence,
        regime=f"strategy:{strategy_name}",
        strategy=strategy_name,
        expected_pf=0.0,
        expected_max_dd=0.0,
        expected_avg_rr=0.0,
        consistency_score=confidence,
        sl=sig.sl,
        tp=sig.tp,
        entry=sig.entry,
        lot_size=0.0,  # risk_manager computes the real size downstream
        explanation=sig.explanation,
        metrics={},
    )
