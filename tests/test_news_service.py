"""Tests for news_service.py — must never raise, always fail soft."""

import pytest
from unittest.mock import patch, MagicMock

from src.services.news_service import NewsService
from src.config import NewsConfig


class TestNewsService:
    def test_disabled_returns_neutral(self):
        svc = NewsService(NewsConfig(alpha_vantage_api_key="", enabled=False))
        result = svc.fetch_news_sentiment("XAUUSD")
        assert result.label == "neutral"
        assert result.source == "none"

    def test_enabled_no_key_returns_neutral(self):
        svc = NewsService(NewsConfig(alpha_vantage_api_key="", enabled=True))
        result = svc.fetch_news_sentiment("XAUUSD")
        assert result.source == "none"

    def test_api_error_fails_soft(self):
        svc = NewsService(NewsConfig(alpha_vantage_api_key="fake_key", enabled=True))
        with patch("src.services.news_service.requests.get", side_effect=Exception("network down")):
            result = svc.fetch_news_sentiment("XAUUSD")
        assert result.label == "neutral"
        assert result.source == "none"

    def test_successful_fetch_parses_sentiment(self):
        svc = NewsService(NewsConfig(alpha_vantage_api_key="fake_key", enabled=True))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "feed": [
                {"title": "Gold rallies on rate cut hopes", "overall_sentiment_score": 0.4},
                {"title": "Gold steady ahead of data", "overall_sentiment_score": 0.1},
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        with patch("src.services.news_service.requests.get", return_value=mock_resp):
            result = svc.fetch_news_sentiment("XAUUSD")
        assert result.source == "alpha_vantage"
        assert result.article_count == 2
        assert result.label == "bullish"

    def test_caches_within_ttl(self):
        svc = NewsService(NewsConfig(alpha_vantage_api_key="fake_key", enabled=True))
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"feed": []}
        mock_resp.raise_for_status = MagicMock()
        with patch("src.services.news_service.requests.get", return_value=mock_resp) as mock_get:
            svc.fetch_news_sentiment("XAUUSD")
            svc.fetch_news_sentiment("XAUUSD")
        assert mock_get.call_count == 1
