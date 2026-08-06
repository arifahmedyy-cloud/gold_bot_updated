"""Multi-Timeframe (MTF) confirmation filter.

Reduces counter-trend false signals by checking the lower-timeframe (LTF)
trade direction against the higher-timeframe (HTF) trend before it reaches
the DecisionEngine.

Logic (deliberately simple and robust — not another full regime model):
    - HTF bias is "bullish" if EMA20 is sloping up AND price is above EMA50.
    - HTF bias is "bearish" if EMA20 is sloping down AND price is below EMA50.
    - Otherwise "neutral" (HTF itself is unclear — don't penalize either way).

If the LTF trade direction agrees with the HTF bias, the decision's
confidence score gets a configurable boost. If it conflicts (e.g. LTF wants
to BUY while HTF is in a clear downtrend), the score takes a configurable
penalty. By default this filter only ever *adjusts* confidence — it does
not veto a trade outright unless explicitly configured to do so via
``MTFConfig.block_on_conflict``.

Usage:
    from src.trading.mtf_filter import MultiTimeframeFilter, MTFConfig

    mtf = MultiTimeframeFilter(MTFConfig(htf_timeframe="H4", ltf_timeframe="M15"))
    result = mtf.evaluate(ltf_direction="BUY", htf_df=htf_ohlcv_df)

    decision = engine.decide(
        regime_signal, smc,
        mtf_score_delta=result.score_delta,
        mtf_note=result.note,
        mtf_hard_block=result.hard_block,
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.logger import get_logger
from src.trading.indicators import TechnicalIndicators

log = get_logger(__name__)


@dataclass
class MTFConfig:
    """Configuration for :class:`MultiTimeframeFilter`.

    Attributes:
        enabled: Master on/off switch. When False, ``evaluate`` is a no-op
            (score_delta=0, hard_block=False) regardless of other settings.
        htf_timeframe: Label for the higher timeframe used for the trend
            bias (e.g. "H4", "D1"). Informational — the caller is
            responsible for actually fetching OHLCV data at this timeframe.
        ltf_timeframe: Label for the lower timeframe whose signal is being
            confirmed (e.g. "M15", "H1"). Informational only.
        slope_threshold: Minimum EMA20 slope (%) required to call the HTF
            trend clearly up/down rather than flat/neutral. This is the
            "confirmation strength" knob — raise it to require a more
            pronounced HTF trend before agreement/conflict is applied.
        agreement_bonus: Confidence points added when the LTF direction
            agrees with a clear HTF bias.
        conflict_penalty: Confidence points subtracted when the LTF
            direction conflicts with a clear HTF bias.
        min_htf_bars: Minimum HTF candles required to trust the bias.
            Indicators need warmup; fewer bars than this just skips the
            check (returns a neutral, non-blocking result).
        block_on_conflict: If True, a conflicting HTF bias causes
            ``evaluate`` to return ``hard_block=True`` (the caller should
            then force NO_TRADE). If False (the default), conflicts only
            reduce confidence via ``conflict_penalty`` — the filter never
            blocks trading outright unless this is explicitly enabled.
    """

    enabled: bool = True
    htf_timeframe: str = "H4"
    ltf_timeframe: str = "M15"
    slope_threshold: float = 0.1
    agreement_bonus: int = 10
    conflict_penalty: int = 25
    min_htf_bars: int = 60
    block_on_conflict: bool = False


@dataclass
class HTFBias:
    """Computed higher-timeframe trend bias."""

    direction: str  # "bullish", "bearish", "neutral"
    ema20_slope: float
    price_vs_ema50: float


@dataclass
class MTFResult:
    """Result of an MTF evaluation, ready to feed into the DecisionEngine."""

    score_delta: int
    note: str
    hard_block: bool
    htf_bias: Optional[HTFBias] = None


class MultiTimeframeFilter:
    """Confirms or penalizes a lower-timeframe signal against the HTF trend."""

    def __init__(self, config: Optional[MTFConfig] = None) -> None:
        """Initialize the filter.

        Args:
            config: MTF configuration. Defaults to ``MTFConfig()`` (enabled,
                non-blocking) if not provided.
        """
        self.config = config or MTFConfig()
        log.debug(
            "MultiTimeframeFilter initialized: enabled=%s htf=%s ltf=%s "
            "slope_threshold=%.3f block_on_conflict=%s",
            self.config.enabled, self.config.htf_timeframe,
            self.config.ltf_timeframe, self.config.slope_threshold,
            self.config.block_on_conflict,
        )

    def compute_bias(self, htf_df: Optional[pd.DataFrame]) -> Optional[HTFBias]:
        """Compute the higher-timeframe trend bias from raw HTF OHLCV data.

        Args:
            htf_df: Raw OHLCV DataFrame on the higher timeframe (e.g. H4 or
                Daily). Must contain a "Close" column; indicators are
                computed internally via :class:`TechnicalIndicators`.

        Returns:
            An :class:`HTFBias`, or None if there isn't enough HTF data to
            trust a bias, or if indicator computation fails.
        """
        if htf_df is None or htf_df.empty or len(htf_df) < self.config.min_htf_bars:
            log.debug(
                "Not enough HTF bars for MTF bias (%s available, need %s)",
                0 if htf_df is None else len(htf_df), self.config.min_htf_bars,
            )
            return None

        try:
            df = TechnicalIndicators.add_all(htf_df)
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last

            ema20 = float(last.get("EMA_20", float("nan")))
            ema20_prev = float(prev.get("EMA_20", ema20))
            ema50 = float(last.get("EMA_50", float("nan")))
            close = float(last["Close"])

            if pd.isna(ema20) or pd.isna(ema50) or ema20_prev == 0:
                log.debug("EMA columns unavailable/invalid for MTF bias computation")
                return None

            slope = ((ema20 - ema20_prev) / abs(ema20_prev)) * 100
            price_vs_ema50 = ((close - ema50) / ema50) * 100 if ema50 else 0.0
        except Exception as exc:  # noqa: BLE001 - never let MTF bias crash the pipeline
            log.error("MTF bias computation failed: %s", exc)
            return None

        threshold = self.config.slope_threshold
        if slope > threshold and price_vs_ema50 > 0:
            direction = "bullish"
        elif slope < -threshold and price_vs_ema50 < 0:
            direction = "bearish"
        else:
            direction = "neutral"

        bias = HTFBias(
            direction=direction,
            ema20_slope=round(slope, 3),
            price_vs_ema50=round(price_vs_ema50, 3),
        )
        log.debug(
            "HTF bias (%s): %s (slope=%.3f, price_vs_ema50=%.3f)",
            self.config.htf_timeframe, direction, slope, price_vs_ema50,
        )
        return bias

    def adjust(self, ltf_direction: str, htf_bias: Optional[HTFBias]) -> MTFResult:
        """Score a lower-timeframe trade direction against an HTF bias.

        Args:
            ltf_direction: "BUY" or "SELL" — the lower-timeframe trade
                direction under consideration.
            htf_bias: Result of :meth:`compute_bias`, or None if unavailable.

        Returns:
            An :class:`MTFResult` with the confidence adjustment, an
            explanatory note, and whether this should hard-block the trade
            (always False unless ``config.block_on_conflict`` is True).
        """
        if not self.config.enabled:
            return MTFResult(score_delta=0, note="MTF filter disabled.", hard_block=False)

        if htf_bias is None:
            return MTFResult(
                score_delta=0,
                note=f"No {self.config.htf_timeframe} data available — MTF check skipped.",
                hard_block=False,
            )

        if htf_bias.direction == "neutral":
            return MTFResult(
                score_delta=0,
                note=f"{self.config.htf_timeframe} trend is neutral/unclear — no MTF adjustment.",
                hard_block=False,
                htf_bias=htf_bias,
            )

        ltf_bullish = ltf_direction == "BUY"
        htf_bullish = htf_bias.direction == "bullish"

        if ltf_bullish == htf_bullish:
            note = (
                f"{self.config.htf_timeframe} trend ({htf_bias.direction}) agrees with "
                f"this {self.config.ltf_timeframe} trade (+{self.config.agreement_bonus})"
            )
            log.debug(note)
            return MTFResult(
                score_delta=self.config.agreement_bonus, note=note,
                hard_block=False, htf_bias=htf_bias,
            )

        note = (
            f"{self.config.htf_timeframe} trend ({htf_bias.direction}) conflicts with "
            f"this {self.config.ltf_timeframe} trade (-{self.config.conflict_penalty})"
        )
        hard_block = self.config.block_on_conflict
        if hard_block:
            note += " — hard-blocked (block_on_conflict enabled)"
            log.info(note)
        else:
            log.debug(note)

        return MTFResult(
            score_delta=-self.config.conflict_penalty, note=note,
            hard_block=hard_block, htf_bias=htf_bias,
        )

    def evaluate(self, ltf_direction: str, htf_df: Optional[pd.DataFrame]) -> MTFResult:
        """Convenience wrapper: compute HTF bias then score it in one call.

        Args:
            ltf_direction: "BUY" or "SELL" — the lower-timeframe trade
                direction under consideration.
            htf_df: Raw OHLCV DataFrame on the higher timeframe.

        Returns:
            An :class:`MTFResult` ready to pass into
            ``DecisionEngine.decide(mtf_score_delta=..., mtf_note=...,
            mtf_hard_block=...)``.
        """
        if not self.config.enabled:
            return MTFResult(score_delta=0, note="MTF filter disabled.", hard_block=False)

        bias = self.compute_bias(htf_df)
        return self.adjust(ltf_direction, bias)
