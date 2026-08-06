"""Google Gemini AI reviewer provider (uses the current google-genai SDK,
the successor to the deprecated google-generativeai package)."""

from __future__ import annotations

from typing import Optional

from src.logger import get_logger
from src.models import SignalOutput, SMCResult, NewsSentiment, AIAnalysis
from src.services.ai.base_provider import AIProvider, SYSTEM_PROMPT, build_user_prompt, parse_ai_response

log = get_logger(__name__)


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        super().__init__(api_key, model)
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                log.warning("Gemini client init failed: %s", exc)
                self._client = None

    def analyze(
        self, regime_signal: SignalOutput, smc: SMCResult,
        news: Optional[NewsSentiment], current_price: float,
    ) -> AIAnalysis:
        if not self.is_available:
            return AIAnalysis(available=False, error="Gemini provider not configured")
        try:
            from google.genai import types
            response = self._client.models.generate_content(
                model=self.model,
                contents=build_user_prompt(regime_signal, smc, news, current_price),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=200,
                    temperature=0.3,
                ),
            )
            text = (response.text or "").strip()
            return parse_ai_response(text)
        except Exception as exc:
            log.warning("Gemini analyze() call failed: %s", exc)
            return AIAnalysis(available=False, error=str(exc))
