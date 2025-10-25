"""Market condition enumerations."""

from enum import Enum


class MarketCondition(Enum):
    """Market condition indicators for trading decisions."""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    OVERSOLD = "oversold"
    OVERBOUGHT = "overbought"
