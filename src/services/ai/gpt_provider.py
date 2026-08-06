"""OpenAI GPT AI reviewer provider."""

from __future__ import annotations

from typing import Optional

from src.logger import get_logger
from src.models import SignalOutput, SMCResult, NewsSentiment, AIAnalysis
from src.services.ai.base_provider import AIProvider, SYSTEM_PROMPT, build_user_prompt, parse_ai_response

log = get_logger(__name__)


class GPTProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        super().__init__(api_key, model)
        if self.api_key:
            try:
                import openai
                self._client = openai.OpenAI(api_key=self.api_key)
            except Exception as exc:
                log.warning("GPT client init failed: %s", exc)
                self._client = None

    def analyze(
        self, regime_signal: SignalOutput, smc: SMCResult,
        news: Optional[NewsSentiment], current_price: float,
    ) -> AIAnalysis:
        if not self.is_available:
            return AIAnalysis(available=False, error="GPT provider not configured")
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(regime_signal, smc, news, current_price)},
                ],
            )
            text = (response.choices[0].message.content or "").strip()
            return parse_ai_response(text)
        except Exception as exc:
            log.warning("GPT analyze() call failed: %s", exc)
            return AIAnalysis(available=False, error=str(exc))
