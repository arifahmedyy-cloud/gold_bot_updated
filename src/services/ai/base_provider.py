"""Shared interface for AI trade-review providers.

Every provider (Claude, Gemini, GPT) implements the same `analyze()`
contract and returns the same `AIAnalysis` shape, so `ai_service.py` and
everything above it (decision_engine, app.py) never needs to know which
provider is active. Adding a new provider later means adding one new file
that implements `AIProvider` — nothing else in the codebase changes.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional

from src.models import SignalOutput, SMCResult, NewsSentiment, AIAnalysis

SYSTEM_PROMPT = (
    "You are a risk-averse trade reviewer for a XAU/USD (gold) algorithmic "
    "trading system. You are given the system's regime signal, Smart Money "
    "Concepts (SMC) analysis, and recent news sentiment. Your job is to give "
    "an independent second opinion — not to repeat the inputs back. "
    "Respond with ONLY a JSON object, no other text, in this exact shape: "
    '{"confidence": <integer 0-100>, "reasoning": "<max 40 words>"}. '
    "confidence reflects how much you'd trust this specific setup right now; "
    "be willing to score low when signals conflict, news is uncertain, or "
    "the setup looks weak."
)


def build_user_prompt(
    regime_signal: SignalOutput, smc: SMCResult, news: Optional[NewsSentiment], current_price: float,
) -> str:
    news = news or NewsSentiment()
    return (
        f"Price: {current_price}\n"
        f"Regime: {regime_signal.regime} | System action: {regime_signal.action} "
        f"| System confidence: {regime_signal.confidence}\n"
        f"SMC bias: {smc.bias} | Zone: {smc.zone}\n"
        f"News sentiment: {news.label} (score={news.score}, "
        f"{news.article_count} articles, source={news.source})\n"
        "Give your independent confidence in this setup."
    )


def parse_ai_response(text: str) -> AIAnalysis:
    """Parse a provider's raw text response into AIAnalysis. Never raises."""
    try:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(cleaned)
        confidence = float(parsed["confidence"])
        confidence = max(0.0, min(100.0, confidence))
        reasoning = str(parsed.get("reasoning", ""))[:300]
        return AIAnalysis(confidence=confidence, reasoning=reasoning, available=True)
    except Exception as exc:
        return AIAnalysis(available=False, error=f"Could not parse AI response: {exc}")


class AIProvider(ABC):
    """Base class every AI reviewer provider must implement."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None

    @property
    def is_available(self) -> bool:
        return self._client is not None

    @abstractmethod
    def analyze(
        self, regime_signal: SignalOutput, smc: SMCResult,
        news: Optional[NewsSentiment], current_price: float,
    ) -> AIAnalysis:
        """Return an independent confidence score. Must never raise —
        catch all provider-specific errors and return AIAnalysis(available=False, error=...)."""
        raise NotImplementedError
