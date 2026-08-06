"""Smart Money Concepts (SMC) analysis with caching.

Detects swing points, order blocks, fair value gaps, liquidity sweeps,
and structural breaks. Optimized with lru_cache for expensive calculations.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import pandas as pd
import numpy as np

from src.logger import get_logger
from src.models import SMCResult

log = get_logger(__name__)


@dataclass
class SwingPoint:
    index: int
    price: float
    type: str  # "high" or "low"
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class StructureEvent:
    index: int
    type: str  # "BOS", "CHoCH"
    direction: str  # "bullish", "bearish"
    level: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class OrderBlock:
    index: int
    type: str  # "bullish", "bearish"
    open_price: float
    high: float
    low: float
    close_price: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class FVG:
    index: int
    type: str  # "bullish", "bearish"
    gap_start: float
    gap_end: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


@dataclass
class LiquiditySweep:
    index: int
    type: str  # "buy", "sell"
    price: float
    date: pd.Timestamp = field(default_factory=pd.Timestamp.now)


class SMCAnalyzer:
    """Smart Money Concepts analyzer."""

    def __init__(self, swing_lookback: int = 5, ob_lookback: int = 5) -> None:
        self.swing_lookback = swing_lookback
        self.ob_lookback = ob_lookback

    @lru_cache(maxsize=32)
    def _cached_analyze(self, data_tuple: Tuple[tuple, ...]) -> SMCResult:
        """Cached version using tuple hash of DataFrame values."""
        df = pd.DataFrame(list(data_tuple), columns=["Open", "High", "Low", "Close"])
        return self._analyze_internal(df)

    def analyze(self, df: pd.DataFrame) -> SMCResult:
        """Analyze DataFrame for SMC patterns.

        Args:
            df: OHLCV DataFrame.

        Returns:
            SMCResult with all detected patterns.
        """
        if len(df) < 20:
            return SMCResult(bias="neutral", zone="equilibrium")
        # Convert to tuple for cacheability
        cols = ["Open", "High", "Low", "Close"]
        data_tuple = tuple(tuple(row) for row in df[cols].values)
        return self._cached_analyze(data_tuple)

    def _analyze_internal(self, df: pd.DataFrame) -> SMCResult:
        swings = self.detect_swing_points(df)
        structure = self.detect_structure_events(df, swings)
        obs = self.detect_order_blocks(df, swings)
        fvgs = self.detect_fvgs(df)
        sweeps = self.detect_liquidity_sweeps(df, swings)
        equal_levels = self.detect_equal_levels(df)
        bias, zone = self._determine_bias_and_zone(df, swings, structure, obs)

        return SMCResult(
            swings=swings,
            structure_events=structure,
            order_blocks=obs,
            fvgs=fvgs,
            liquidity_sweeps=sweeps,
            equal_levels=equal_levels,
            zone=zone,
            bias=bias,
        )

    def detect_swing_points(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Detect swing highs and lows."""
        highs = df["High"].values
        lows = df["Low"].values
        n = len(df)
        lb = self.swing_lookback
        swings = []

        for i in range(lb, n - lb):
            if all(highs[i] > highs[i - j] for j in range(1, lb + 1)) and                all(highs[i] > highs[i + j] for j in range(1, lb + 1)):
                swings.append(SwingPoint(
                    index=i, price=float(highs[i]), type="high",
                    date=pd.Timestamp(df.index[i]) if hasattr(df.index[i], "year") else pd.Timestamp.now()
                ))
            elif all(lows[i] < lows[i - j] for j in range(1, lb + 1)) and                  all(lows[i] < lows[i + j] for j in range(1, lb + 1)):
                swings.append(SwingPoint(
                    index=i, price=float(lows[i]), type="low",
                    date=pd.Timestamp(df.index[i]) if hasattr(df.index[i], "year") else pd.Timestamp.now()
                ))
        return swings

    def detect_structure_events(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[StructureEvent]:
        """Detect Break of Structure (BOS) and Change of Character (CHoCH)."""
        events = []
        if len(swings) < 4:
            return events

        for i in range(3, len(swings)):
            s0, s1, s2, s3 = swings[i-3], swings[i-2], swings[i-1], swings[i]
            if s0.type == "low" and s1.type == "high" and s2.type == "low" and s3.type == "high":
                if s3.price > s1.price:
                    events.append(StructureEvent(
                        index=s3.index, type="BOS", direction="bullish", level=s3.price
                    ))
                elif s3.price < s1.price and s2.price > s0.price:
                    events.append(StructureEvent(
                        index=s3.index, type="CHoCH", direction="bearish", level=s3.price
                    ))
            elif s0.type == "high" and s1.type == "low" and s2.type == "high" and s3.type == "low":
                if s3.price < s1.price:
                    events.append(StructureEvent(
                        index=s3.index, type="BOS", direction="bearish", level=s3.price
                    ))
                elif s3.price > s1.price and s2.price < s0.price:
                    events.append(StructureEvent(
                        index=s3.index, type="CHoCH", direction="bullish", level=s3.price
                    ))
        return events

    def detect_order_blocks(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[OrderBlock]:
        """Detect bullish and bearish order blocks."""
        obs = []
        lb = self.ob_lookback
        for swing in swings:
            idx = swing.index
            if idx < lb:
                continue
            if swing.type == "low":
                # Bullish OB: candle before the low
                ob_idx = idx - 1
                if ob_idx >= 0:
                    obs.append(OrderBlock(
                        index=ob_idx, type="bullish",
                        open_price=float(df.iloc[ob_idx]["Open"]),
                        high=float(df.iloc[ob_idx]["High"]),
                        low=float(df.iloc[ob_idx]["Low"]),
                        close_price=float(df.iloc[ob_idx]["Close"]),
                    ))
            else:
                ob_idx = idx - 1
                if ob_idx >= 0:
                    obs.append(OrderBlock(
                        index=ob_idx, type="bearish",
                        open_price=float(df.iloc[ob_idx]["Open"]),
                        high=float(df.iloc[ob_idx]["High"]),
                        low=float(df.iloc[ob_idx]["Low"]),
                        close_price=float(df.iloc[ob_idx]["Close"]),
                    ))
        return obs

    def detect_fvgs(self, df: pd.DataFrame) -> List[FVG]:
        """Detect Fair Value Gaps."""
        fvgs = []
        for i in range(2, len(df)):
            c1 = df.iloc[i-2]
            c2 = df.iloc[i-1]
            c3 = df.iloc[i]
            if c2["Low"] > c1["High"]:
                fvgs.append(FVG(
                    index=i, type="bullish",
                    gap_start=float(c1["High"]), gap_end=float(c2["Low"]),
                ))
            elif c2["High"] < c1["Low"]:
                fvgs.append(FVG(
                    index=i, type="bearish",
                    gap_start=float(c1["Low"]), gap_end=float(c2["High"]),
                ))
        return fvgs

    def detect_liquidity_sweeps(self, df: pd.DataFrame, swings: List[SwingPoint]) -> List[LiquiditySweep]:
        """Detect liquidity sweeps above/below swing points."""
        sweeps = []
        if not swings:
            return sweeps
        last_close = float(df.iloc[-1]["Close"])
        for swing in swings[-5:]:
            if swing.type == "high" and last_close > swing.price * 1.001:
                sweeps.append(LiquiditySweep(
                    index=swing.index, type="buy", price=swing.price
                ))
            elif swing.type == "low" and last_close < swing.price * 0.999:
                sweeps.append(LiquiditySweep(
                    index=swing.index, type="sell", price=swing.price
                ))
        return sweeps

    def detect_equal_levels(self, df: pd.DataFrame, tolerance: float = 0.001) -> List[Dict[str, Any]]:
        """Detect equal highs/lows (liquidity pools)."""
        levels = []
        highs = df["High"].values
        lows = df["Low"].values
        for i in range(len(df) - 5):
            for j in range(i + 5, len(df)):
                if abs(highs[i] - highs[j]) / highs[i] < tolerance:
                    levels.append({"type": "equal_high", "price": float(highs[i]), "indices": [i, j]})
                if abs(lows[i] - lows[j]) / lows[i] < tolerance:
                    levels.append({"type": "equal_low", "price": float(lows[i]), "indices": [i, j]})
        return levels

    def _determine_bias_and_zone(
        self, df: pd.DataFrame, swings: List[SwingPoint],
        structure: List[StructureEvent], obs: List[OrderBlock]
    ) -> Tuple[str, str]:
        """Determine overall bias and premium/discount zone."""
        if not swings:
            return "neutral", "equilibrium"

        last_close = float(df.iloc[-1]["Close"])
        recent_highs = [s.price for s in swings if s.type == "high"][-3:]
        recent_lows = [s.price for s in swings if s.type == "low"][-3:]
        if not recent_highs or not recent_lows:
            return "neutral", "equilibrium"

        mid = (max(recent_highs) + min(recent_lows)) / 2
        if last_close > mid * 1.01:
            zone = "premium"
        elif last_close < mid * 0.99:
            zone = "discount"
        else:
            zone = "equilibrium"

        bias = "neutral"
        if structure:
            last_event = structure[-1]
            bias = last_event.direction
        elif len(swings) >= 2:
            if swings[-1].type == "high" and swings[-1].price > swings[-2].price:
                bias = "bullish"
            elif swings[-1].type == "low" and swings[-1].price < swings[-2].price:
                bias = "bearish"

        return bias, zone
