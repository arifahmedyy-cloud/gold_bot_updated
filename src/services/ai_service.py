"""AI trade reviewer facade — picks the configured provider (Claude, Gemini,
or GPT) and exposes one stable interface to the rest of the app.

decision_engine.py and app.py only ever talk to AIService; they never know
or care which underlying provider answered. Adding a 4th provider later
means adding one new file under src/services/ai/ that implements
AIProvider — this file only needs a one-line addition to _PROVIDERS.
"""

from __future__ import annotations

from typing import Optional

from src.logger import get_logger
from src.config import AIConfig
from src.models import SignalOutput, SMCResult, NewsSentiment, AIAnalysis
from src.services.ai.claude_provider import ClaudeProvider
from src.services.ai.gemini_provider import GeminiProvider
from src.services.ai.gpt_provider import GPTProvider

log = get_logger(__name__)

_PROVIDERS = {
    "claude": lambda cfg: ClaudeProvider(cfg.anthropic_api_key, cfg.model or "claude-sonnet-5"),
    "gemini": lambda cfg: GeminiProvider(cfg.google_api_key, cfg.gemini_model or "gemini-2.0-flash"),
    "gpt": lambda cfg: GPTProvider(cfg.openai_api_key, cfg.gpt_model or "gpt-4o-mini"),
}


class AIService:
    """Wraps the active AI provider to produce an independent trade confidence score."""

    def __init__(self, config: AIConfig) -> None:
        self.config = config
        self._provider = None
        if self.config.enabled:
            factory = _PROVIDERS.get(self.config.provider)
            if factory is None:
                log.warning("Unknown AI provider '%s', AI reviewer disabled", self.config.provider)
            else:
                self._provider = factory(self.config)

    @property
    def is_available(self) -> bool:
        return self._provider is not None and self._provider.is_available

    @property
    def active_provider_name(self) -> str:
        return self.config.provider if self.is_available else "none"

    def analyze(
        self,
        regime_signal: SignalOutput,
        smc: SMCResult,
        news: Optional[NewsSentiment],
        current_price: float,
    ) -> AIAnalysis:
        """Ask the active provider for an independent confidence score on the current setup.

        Never raises. Returns AIAnalysis(available=False, error=...) if no
        provider is configured or the call fails.
        """
        if not self.is_available:
            return AIAnalysis(available=False, error="AI reviewer not configured")
        return self._provider.analyze(regime_signal, smc, news, current_price)
