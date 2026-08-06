"""News sentiment service using Alpha Vantage NEWS_SENTIMENT.

Fetches recent news for a symbol and aggregates a bullish/bearish score.
Fails soft: any error or missing API key returns a neutral, zero-confidence
result rather than raising, so the trading loop never blocks on this.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

import requests

from src.logger import get_logger
from src.config import NewsConfig
from src.models import NewsSentiment

log = get_logger(__name__)

_ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
_CACHE_TTL_SECONDS = 900  # 15 minutes — respects free-tier rate limits


class NewsService:
    """Fetches and caches news sentiment for a trading symbol."""

    def __init__(self, config: NewsConfig) -> None:
        self.config = config
        self._cache: Dict[str, Any] = {}
        self._cache_time: Dict[str, float] = {}

    def fetch_news_sentiment(self, symbol: str = "XAUUSD", topics: str = "financial_markets") -> NewsSentiment:
        """Return aggregated news sentiment, using a 15-minute cache.

        Never raises — returns a neutral NewsSentiment on any failure so the
        trading loop can proceed without news data.
        """
        if not self.config.enabled or not self.config.alpha_vantage_api_key:
            return NewsSentiment(score=0.0, label="neutral", article_count=0, headlines=[], source="none")

        now = time.time()
        cached = self._cache.get(symbol)
        if cached is not None and (now - self._cache_time.get(symbol, 0)) < _CACHE_TTL_SECONDS:
            return cached

        try:
            result = self._fetch_from_api(symbol, topics)
            self._cache[symbol] = result
            self._cache_time[symbol] = now
            return result
        except Exception as exc:
            log.warning("News sentiment fetch failed, falling back to neutral: %s", exc)
            # Serve stale cache if we have it, otherwise neutral fallback
            if cached is not None:
                return cached
            return NewsSentiment(score=0.0, label="neutral", article_count=0, headlines=[], source="none")

    def _fetch_from_api(self, symbol: str, topics: str) -> NewsSentiment:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "topics": topics,
            "apikey": self.config.alpha_vantage_api_key,
            "limit": 20,
        }
        resp = requests.get(_ALPHA_VANTAGE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        if "Note" in data or "Information" in data:
            # Rate-limited or invalid key — treat as unavailable, not an error
            log.warning("Alpha Vantage rate-limited or unavailable: %s", data.get("Note") or data.get("Information"))
            return NewsSentiment(score=0.0, label="neutral", article_count=0, headlines=[], source="none")

        feed = data.get("feed", [])
        if not feed:
            return NewsSentiment(score=0.0, label="neutral", article_count=0, headlines=[], source="alpha_vantage")

        scores = []
        headlines = []
        for item in feed[:20]:
            try:
                scores.append(float(item.get("overall_sentiment_score", 0.0)))
            except (TypeError, ValueError):
                continue
            title = item.get("title")
            if title and len(headlines) < 5:
                headlines.append(title)

        avg_score = sum(scores) / len(scores) if scores else 0.0
        label = self._score_to_label(avg_score)

        return NewsSentiment(
            score=round(avg_score, 4),
            label=label,
            article_count=len(feed),
            headlines=headlines,
            source="alpha_vantage",
        )

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.15:
            return "bullish"
        if score <= -0.15:
            return "bearish"
        return "neutral"
