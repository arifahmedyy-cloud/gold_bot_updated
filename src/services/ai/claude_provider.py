"""Claude (Anthropic) AI reviewer provider."""

from __future__ import annotations

from typing import Optional

from src.logger import get_logger
from src.models import SignalOutput, SMCResult, NewsSentiment, AIAnalysis
from src.services.ai.base_provider import AIProvider, SYSTEM_PROMPT, build_user_prompt, parse_ai_response

log = get_logger(__name__)


class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "claude-sonnet-5") -> None:
        super().__init__(api_key, model)
        if self.api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as exc:
                log.warning("Claude client init failed: %s", exc)
                self._client = None

    def analyze(
        self, regime_signal: SignalOutput, smc: SMCResult,
        news: Optional[NewsSentiment], current_price: float,
    ) -> AIAnalysis:
        if not self.is_available:
            return AIAnalysis(available=False, error="Claude provider not configured")
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(regime_signal, smc, news, current_price)}],
            )
            text = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            return parse_ai_response(text)
        except Exception as exc:
            log.warning("Claude analyze() call failed: %s", exc)
            return AIAnalysis(available=False, error=str(exc))
