"""Decision engine that gates signals through confluence checks.

Combines regime signal, SMC bias, and optional ML filter into a final decision.
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from src.logger import get_logger
from src.models import Decision, SignalOutput, SMCResult
from src.config import RiskConfig

log = get_logger(__name__)


class DecisionEngine:
    """Gates trading signals through multi-layer confluence."""

    def __init__(self, config: RiskConfig) -> None:
        self.config = config
        self._last_decision: Optional[Decision] = None

    def decide(
        self,
        regime_signal: SignalOutput,
        smc: SMCResult,
        ml_confidence: Optional[float] = None,
        current_price: float = 0.0,
        mtf_score_delta: Optional[int] = None,
        mtf_note: Optional[str] = None,
        mtf_hard_block: bool = False,
    ) -> Decision:
        """Make final trading decision.

        Args:
            regime_signal: Signal from regime detector.
            smc: SMC analysis result.
            ml_confidence: Optional ML model confidence (0-100).
            current_price: Current market price.
            mtf_score_delta: Optional confidence adjustment from
                ``MultiTimeframeFilter.evaluate(...).score_delta`` (positive
                when the HTF trend agrees with the trade, negative when it
                conflicts). None means the MTF filter wasn't run.
            mtf_note: Optional explanatory note from the MTF filter,
                appended to the confluence notes for traceability.
            mtf_hard_block: If True, the MTF filter's ``block_on_conflict``
                setting was enabled and triggered — the decision is forced
                to NO_TRADE. Defaults to False, so the MTF filter never
                blocks a trade unless explicitly configured to.

        Returns:
            Decision object with action and explanation.
        """
        notes: List[str] = []
        action = "NO_TRADE"
        ai_score = regime_signal.confidence
        entry = regime_signal.entry
        sl = regime_signal.sl
        tp = regime_signal.tp
        lot_size = regime_signal.lot_size

        # MTF (multi-timeframe) confirmation filter
        if mtf_note is not None:
            notes.append(mtf_note)
        if mtf_hard_block:
            return Decision(
                action="NO_TRADE", ai_score=ai_score, regime_signal=regime_signal,
                smc=smc, confluence_notes=notes,
                explanation="Blocked by MTF filter (higher-timeframe conflict)",
            )
        if mtf_score_delta:
            ai_score = max(0, min(100, ai_score + mtf_score_delta))

        # ML filter
        if self.config.enable_ml_filter and ml_confidence is not None:
            if ml_confidence < self.config.min_ml_confidence:
                notes.append(f"ML filter blocked: confidence {ml_confidence:.1f} < {self.config.min_ml_confidence}")
                return Decision(
                    action="NO_TRADE", ai_score=ai_score, regime_signal=regime_signal,
                    smc=smc, confluence_notes=notes,
                    explanation="ML confidence too low",
                )
            notes.append(f"ML confidence: {ml_confidence:.1f}")
            ai_score = int((ai_score + ml_confidence) / 2)

        # Minimum AI score
        if ai_score < self.config.min_ai_score:
            notes.append(f"AI score {ai_score} below minimum {self.config.min_ai_score}")
            return Decision(
                action="NO_TRADE", ai_score=ai_score, regime_signal=regime_signal,
                smc=smc, confluence_notes=notes,
                explanation="AI score below threshold",
            )

        # SMC confluence
        if regime_signal.action == "BUY" and smc.bias == "bullish":
            notes.append("SMC confirms bullish bias")
            action = "BUY"
        elif regime_signal.action == "SELL" and smc.bias == "bearish":
            notes.append("SMC confirms bearish bias")
            action = "SELL"
        elif regime_signal.action == "BUY" and smc.bias == "bearish":
            notes.append("SMC contradicts bullish signal — reduced confidence")
            ai_score = max(0, ai_score - 15)
            if ai_score >= self.config.min_ai_score:
                action = "BUY"
        elif regime_signal.action == "SELL" and smc.bias == "bullish":
            notes.append("SMC contradicts bearish signal — reduced confidence")
            ai_score = max(0, ai_score - 15)
            if ai_score >= self.config.min_ai_score:
                action = "SELL"
        elif smc.bias == "neutral":
            notes.append("SMC neutral — using regime signal only")
            action = regime_signal.action if regime_signal.action in ("BUY", "SELL") else "NO_TRADE"
        else:
            notes.append(f"No confluence: regime={regime_signal.action}, SMC={smc.bias}")

        # Zone check
        if action in ("BUY", "SELL"):
            if action == "BUY" and smc.zone == "premium":
                notes.append("Warning: buying in premium zone")
                ai_score = max(0, ai_score - 10)
            elif action == "SELL" and smc.zone == "discount":
                notes.append("Warning: selling in discount zone")
                ai_score = max(0, ai_score - 10)

        # Final validation
        if action in ("BUY", "SELL") and ai_score < self.config.min_ai_score:
            notes.append(f"Final score {ai_score} below threshold after adjustments")
            action = "NO_TRADE"

        explanation = f"Decision: {action} | Score: {ai_score}/100 | " + " | ".join(notes)
        log.info(explanation)

        self._last_decision = Decision(
            action=action, ai_score=ai_score, regime_signal=regime_signal,
            smc=smc, confluence_notes=notes, explanation=explanation,
            sl=sl, tp=tp, entry=entry, lot_size=lot_size,
        )
        return self._last_decision
