"""
Sentiment agent that interprets FinBERT-enhanced news analytics for trading decisions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger

from services.crypto_news_analyzer import (
    CryptoNewsAnalyzer,
    CryptoNewsSentiment,
)


@dataclass
class SentimentAgentResult:
    """Structured output from the sentiment agent."""

    symbol: str
    overall_sentiment: str
    sentiment_score: float
    average_confidence: float
    trend: str
    bias: str
    news_count: int
    high_impact_count: int
    insights: List[str]
    supporting_news: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "overall_sentiment": self.overall_sentiment,
            "sentiment_score": self.sentiment_score,
            "average_confidence": self.average_confidence,
            "trend": self.trend,
            "bias": self.bias,
            "news_count": self.news_count,
            "high_impact_count": self.high_impact_count,
            "insights": self.insights,
            "supporting_news": self.supporting_news,
        }


class SentimentAgent:
    """
    Sentiment-focused agent that summarizes the latest news and FinBERT analysis.

    The agent can operate on top of a pre-computed `CryptoNewsSentiment` object
    so callers can share API responses across agents.
    """

    def __init__(
        self,
        llm_analyzer: Optional[Any] = None,
        news_analyzer: Optional[CryptoNewsAnalyzer] = None,
        include_social: bool = False,
    ) -> None:
        self.llm_analyzer = llm_analyzer
        self.news_analyzer = news_analyzer or CryptoNewsAnalyzer(use_finbert=True)
        self.include_social = include_social

    async def analyze(
        self,
        symbol: str,
        *,
        news_sentiment: Optional[CryptoNewsSentiment] = None,
        hours: int = 2,
    ) -> SentimentAgentResult:
        """
        Produce a structured sentiment analysis for the specified symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC").
            news_sentiment: Optional pre-computed sentiment payload.
            hours: Lookback window for gathering news if a payload needs to be fetched.
        """
        sentiment_data = news_sentiment

        if sentiment_data is None:
            logger.debug(f"SentimentAgent fetching sentiment data for {symbol} ({hours}h window)")
            try:
                sentiment_data = await self.news_analyzer.analyze_comprehensive_sentiment(
                    symbol,
                    include_social=self.include_social,
                    hours=hours,
                )
            except Exception as exc:  # pragma: no cover - network edge cases
                logger.error(f"SentimentAgent failed to fetch news for {symbol}: {exc}", exc_info=True)
                sentiment_data = CryptoNewsSentiment(
                    symbol=symbol,
                    news_count=0,
                    recent_news=[],
                    overall_sentiment="NEUTRAL",
                    sentiment_score=0.0,
                    bullish_articles=0,
                    bearish_articles=0,
                    neutral_articles=0,
                    major_catalysts=[],
                    average_confidence=0.0,
                    overall_sentiment_score=50.0,
                    high_impact_news_count=0,
                    sentiment_trend="STABLE",
                )

        insights = self._build_insights(sentiment_data)
        bias = self._determine_bias(sentiment_data)

        supporting_news = [
            {
                "title": article.title,
                "sentiment": article.sentiment,
                "confidence": article.sentiment_confidence,
                "impact": article.market_impact,
                "published_at": article.published_at,
            }
            for article in sentiment_data.recent_news
        ]

        return SentimentAgentResult(
            symbol=sentiment_data.symbol,
            overall_sentiment=sentiment_data.overall_sentiment,
            sentiment_score=sentiment_data.overall_sentiment_score,
            average_confidence=round(sentiment_data.average_confidence * 100, 1),
            trend=sentiment_data.sentiment_trend,
            bias=bias,
            news_count=sentiment_data.news_count,
            high_impact_count=sentiment_data.high_impact_news_count,
            insights=insights,
            supporting_news=supporting_news,
        )

    def _determine_bias(self, sentiment_data: CryptoNewsSentiment) -> str:
        """Map raw scores to a trading bias."""
        score = sentiment_data.overall_sentiment_score
        trend = sentiment_data.sentiment_trend.upper() if sentiment_data.sentiment_trend else "STABLE"

        if score >= 70:
            return "AGGRESSIVE_BULLISH" if trend == "IMPROVING" else "BULLISH"
        if score >= 60:
            return "BULLISH"
        if score <= 30:
            return "AGGRESSIVE_BEARISH" if trend == "DETERIORATING" else "BEARISH"
        if score <= 40:
            return "BEARISH"
        return "NEUTRAL"

    def _build_insights(self, sentiment_data: CryptoNewsSentiment) -> List[str]:
        """Generate concise human-readable insights for the coordinator."""
        insights: List[str] = []

        if sentiment_data.high_impact_news_count > 0:
            insights.append(
                f"{sentiment_data.high_impact_news_count} high-impact catalyst(s) detected in the last window."
            )

        if sentiment_data.sentiment_trend == "IMPROVING":
            insights.append("Sentiment is improving compared to the prior articles.")
        elif sentiment_data.sentiment_trend == "DETERIORATING":
            insights.append("Sentiment momentum is deteriorating versus earlier headlines.")

        if sentiment_data.overall_sentiment == "BULLISH":
            insights.append("FinBERT-weighted sentiment is bullish.")
        elif sentiment_data.overall_sentiment == "BEARISH":
            insights.append("FinBERT-weighted sentiment is bearish.")
        else:
            insights.append("FinBERT-weighted sentiment is neutral.")

        top_catalysts = sentiment_data.major_catalysts[:3]
        for catalyst in top_catalysts:
            insights.append(f"Catalyst: {catalyst}")

        return insights

