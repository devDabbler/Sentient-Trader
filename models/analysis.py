"""Data models for stock analysis and strategy recommendations."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class StrategyRecommendation:
    """Recommendation for a trading strategy"""
    strategy_name: str
    action: str
    confidence: float
    reasoning: str
    risk_level: str
    max_loss: str
    max_gain: str
    best_conditions: List[str]
    experience_level: str
    examples: Optional[List[str]] = None
    notes: Optional[str] = None
    example_trade: Optional[Dict] = None


@dataclass
class StockAnalysis:
    """Complete stock analysis with technicals, news, and catalysts"""
    ticker: str
    price: float
    change_pct: float
    volume: int
    avg_volume: int
    rsi: float
    macd_signal: str
    trend: str
    support: float
    resistance: float
    iv_rank: float
    iv_percentile: float
    earnings_date: Optional[str]
    earnings_days_away: Optional[int]
    recent_news: List[Dict]
    catalysts: List[Dict]
    sentiment_score: float
    sentiment_signals: List[str]
    confidence_score: float
    recommendation: str
    # Optional advanced indicator context (additive)
    ema8: Optional[float] = None
    ema21: Optional[float] = None
    demarker: Optional[float] = None
    fib_targets: Optional[Dict[str, float]] = None
    ema_power_zone: Optional[bool] = None
    ema_reclaim: Optional[bool] = None
    # Multi-timeframe and sector analysis (additive)
    timeframe_alignment: Optional[Dict] = None
    sector_rs: Optional[Dict] = None
