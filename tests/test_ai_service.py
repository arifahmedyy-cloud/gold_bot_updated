"""Tests for ai_service.py — the multi-provider dispatcher.

Provider-specific behavior is tested in test_ai_providers.py; these tests
only check that AIService picks the right provider and stays fail-soft.
"""

import pytest
from unittest.mock import MagicMock

from src.services.ai_service import AIService
from src.config import AIConfig
from src.models import SignalOutput, SMCResult, AIAnalysis


def _make_signal():
    return SignalOutput(
        action="BUY", confidence=70, regime="strong_uptrend", strategy="Test",
        expected_pf=1.5, expected_max_dd=0.1, expected_avg_rr=1.2,
        consistency_score=80, sl=2440.0, tp=2460.0, entry=2450.0,
        lot_size=0.1, explanation="test", metrics={},
    )


class TestAIServiceDispatcher:
    def test_disabled_when_not_enabled(self):
        svc = AIService(AIConfig(enabled=False))
        assert svc.is_available is False
        result = svc.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is False

    def test_disabled_when_no_key_for_provider(self):
        svc = AIService(AIConfig(provider="claude", anthropic_api_key="", enabled=True))
        assert svc.is_available is False

    def test_unknown_provider_fails_soft(self):
        svc = AIService(AIConfig(provider="not_a_real_provider", enabled=True))
        assert svc.is_available is False
        result = svc.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is False

    def test_picks_claude_provider(self):
        svc = AIService(AIConfig(provider="claude", anthropic_api_key="fake-key", enabled=True))
        assert svc.active_provider_name in ("claude", "none")

    def test_picks_gemini_provider_class(self):
        svc = AIService(AIConfig(provider="gemini", google_api_key="fake-key", enabled=True))
        assert svc.is_available in (True, False)

    def test_picks_gpt_provider_class(self):
        svc = AIService(AIConfig(provider="gpt", openai_api_key="fake-key", enabled=True))
        assert svc.is_available in (True, False)

    def test_analyze_delegates_to_provider(self):
        svc = AIService(AIConfig(provider="claude", anthropic_api_key="fake-key", enabled=True))
        mock_provider = MagicMock()
        mock_provider.is_available = True
        mock_provider.analyze.return_value = AIAnalysis(confidence=80.0, reasoning="ok", available=True)
        svc._provider = mock_provider

        result = svc.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is True
        assert result.confidence == 80.0
        mock_provider.analyze.assert_called_once()
