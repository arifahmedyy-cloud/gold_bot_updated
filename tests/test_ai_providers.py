"""Tests for each AI provider (Claude, Gemini, GPT) — parsing, clamping,
and fail-soft behavior, using mocked SDK clients so no real API calls happen."""

from unittest.mock import MagicMock

from src.services.ai.claude_provider import ClaudeProvider
from src.services.ai.gemini_provider import GeminiProvider
from src.services.ai.gpt_provider import GPTProvider
from src.models import SignalOutput, SMCResult


def _make_signal():
    return SignalOutput(
        action="BUY", confidence=70, regime="strong_uptrend", strategy="Test",
        expected_pf=1.5, expected_max_dd=0.1, expected_avg_rr=1.2,
        consistency_score=80, sl=2440.0, tp=2460.0, entry=2450.0,
        lot_size=0.1, explanation="test", metrics={},
    )


class TestClaudeProvider:
    def test_parses_valid_response(self):
        provider = ClaudeProvider(api_key="fake")
        mock_block = MagicMock()
        mock_block.type = "text"
        mock_block.text = '{"confidence": 65, "reasoning": "moderate setup"}'
        mock_response = MagicMock()
        mock_response.content = [mock_block]
        provider._client = MagicMock()
        provider._client.messages.create.return_value = mock_response

        result = provider.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is True
        assert result.confidence == 65.0

    def test_no_key_unavailable(self):
        provider = ClaudeProvider(api_key="")
        assert provider.is_available is False


class TestGeminiProvider:
    def test_parses_valid_response(self):
        provider = GeminiProvider(api_key="fake")
        mock_response = MagicMock()
        mock_response.text = '{"confidence": 55, "reasoning": "mixed signals"}'
        provider._client = MagicMock()
        provider._client.models.generate_content.return_value = mock_response

        result = provider.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is True
        assert result.confidence == 55.0

    def test_no_key_unavailable(self):
        provider = GeminiProvider(api_key="")
        assert provider.is_available is False

    def test_call_failure_fails_soft(self):
        provider = GeminiProvider(api_key="fake")
        provider._client = MagicMock()
        provider._client.models.generate_content.side_effect = Exception("network error")

        result = provider.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is False


class TestGPTProvider:
    def test_parses_valid_response(self):
        provider = GPTProvider(api_key="fake")
        mock_message = MagicMock()
        mock_message.content = '{"confidence": 90, "reasoning": "strong confluence"}'
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        provider._client = MagicMock()
        provider._client.chat.completions.create.return_value = mock_response

        result = provider.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is True
        assert result.confidence == 90.0

    def test_no_key_unavailable(self):
        provider = GPTProvider(api_key="")
        assert provider.is_available is False

    def test_call_failure_fails_soft(self):
        provider = GPTProvider(api_key="fake")
        provider._client = MagicMock()
        provider._client.chat.completions.create.side_effect = Exception("timeout")

        result = provider.analyze(_make_signal(), SMCResult(), None, 2450.0)
        assert result.available is False
