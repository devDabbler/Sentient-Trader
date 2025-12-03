"""
Macro Market Filter - Multi-Factor Market Health Assessment

Provides comprehensive macro and micro market analysis to enhance trading decisions:

MACRO FACTORS:
1. SPY/QQQ Trend Filter - Major index direction (above/below key SMAs)
2. VIX Fear Gauge - Volatility regime detection
3. 10Y Treasury Yield - Interest rate environment
4. Dollar Strength (DXY) - Currency impact on multinationals
5. Sector Rotation - Defensive vs Growth allocation
6. Market Breadth - % stocks above 200 SMA, A/D ratio
7. Economic Calendar Integration - Fed events, CPI, NFP proximity

MICRO FACTORS:
1. Pre/Post Market Gaps
2. Intraday Momentum
3. Time-of-Day filters (avoid choppy lunch hours)
4. Options Expiration impact (OpEx weeks)

Score Integration:
- Returns a macro adjustment (-30 to +30 points) to composite scores
- Provides risk-adjusted position sizing multiplier (0.25x to 1.25x)
- Blocks trades during high-risk macro conditions

Author: Sentient Trader
Last Updated: 2024-12
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
import threading

from loguru import logger
import yfinance as yf
import pandas as pd
import numpy as np

# Try to import economic calendar for integration
try:
    from services.event_detectors.economic_detector import EconomicDetector
    HAS_ECONOMIC_DETECTOR = True
except ImportError:
    HAS_ECONOMIC_DETECTOR = False
    logger.debug("EconomicDetector not available")


class MacroRegime(Enum):
    """Market regime classification"""
    RISK_ON = "risk_on"           # Favorable for longs
    NEUTRAL = "neutral"           # Mixed signals
    RISK_OFF = "risk_off"         # Caution, reduce exposure
    CRISIS = "crisis"             # High volatility, avoid new positions


class TrendDirection(Enum):
    """Trend direction for indices"""
    STRONG_UP = "strong_uptrend"
    UP = "uptrend"
    NEUTRAL = "neutral"
    DOWN = "downtrend"
    STRONG_DOWN = "strong_downtrend"


@dataclass
class IndexHealth:
    """Health metrics for a market index"""
    symbol: str
    current_price: float = 0.0
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    above_sma_20: bool = False
    above_sma_50: bool = False
    above_sma_200: bool = False
    trend: TrendDirection = TrendDirection.NEUTRAL
    momentum_5d: float = 0.0
    score: float = 50.0  # 0-100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "current_price": round(self.current_price, 2),
            "sma_20": round(self.sma_20, 2),
            "sma_50": round(self.sma_50, 2),
            "sma_200": round(self.sma_200, 2),
            "above_sma_20": self.above_sma_20,
            "above_sma_50": self.above_sma_50,
            "above_sma_200": self.above_sma_200,
            "trend": self.trend.value,
            "momentum_5d": round(self.momentum_5d, 2),
            "score": round(self.score, 1)
        }


@dataclass
class VIXAnalysis:
    """VIX volatility analysis"""
    current_level: float = 0.0
    sma_20: float = 0.0
    percentile_rank: float = 50.0  # Where VIX sits in 1-year range
    regime: str = "NORMAL"  # LOW, NORMAL, ELEVATED, HIGH, EXTREME
    contango_backwardation: str = "UNKNOWN"  # CONTANGO, BACKWARDATION
    score: float = 50.0  # 0-100 (higher = more fearful = less favorable)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_level": round(self.current_level, 2),
            "sma_20": round(self.sma_20, 2),
            "percentile_rank": round(self.percentile_rank, 1),
            "regime": self.regime,
            "contango_backwardation": self.contango_backwardation,
            "score": round(self.score, 1)
        }


@dataclass
class TreasuryAnalysis:
    """10Y Treasury yield analysis"""
    current_yield: float = 0.0
    yield_1m_ago: float = 0.0
    yield_change: float = 0.0
    trend: str = "STABLE"  # RISING, STABLE, FALLING
    impact: str = "NEUTRAL"  # POSITIVE, NEUTRAL, NEGATIVE for growth stocks
    score: float = 50.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_yield": round(self.current_yield, 2),
            "yield_1m_ago": round(self.yield_1m_ago, 2),
            "yield_change_bps": round(self.yield_change * 100, 0),
            "trend": self.trend,
            "impact": self.impact,
            "score": round(self.score, 1)
        }


@dataclass
class DollarAnalysis:
    """Dollar strength (DXY) analysis"""
    current_level: float = 0.0
    sma_50: float = 0.0
    trend: str = "STABLE"  # STRENGTHENING, STABLE, WEAKENING
    impact: str = "NEUTRAL"  # Impact on multinationals
    score: float = 50.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_level": round(self.current_level, 2),
            "sma_50": round(self.sma_50, 2),
            "trend": self.trend,
            "impact": self.impact,
            "score": round(self.score, 1)
        }


@dataclass
class BreadthAnalysis:
    """Market breadth indicators"""
    pct_above_sma200: float = 50.0
    pct_above_sma50: float = 50.0
    advance_decline_ratio: float = 1.0
    new_highs_vs_lows: float = 0.0
    breadth_thrust: bool = False
    score: float = 50.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pct_above_sma200": round(self.pct_above_sma200, 1),
            "pct_above_sma50": round(self.pct_above_sma50, 1),
            "advance_decline_ratio": round(self.advance_decline_ratio, 2),
            "new_highs_vs_lows": round(self.new_highs_vs_lows, 0),
            "breadth_thrust": self.breadth_thrust,
            "score": round(self.score, 1)
        }


@dataclass
class SectorRotation:
    """Sector rotation analysis"""
    leading_sectors: List[str] = field(default_factory=list)
    lagging_sectors: List[str] = field(default_factory=list)
    rotation_signal: str = "NEUTRAL"  # RISK_ON (growth), RISK_OFF (defensive)
    sector_momentum: Dict[str, float] = field(default_factory=dict)
    score: float = 50.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leading_sectors": self.leading_sectors,
            "lagging_sectors": self.lagging_sectors,
            "rotation_signal": self.rotation_signal,
            "sector_momentum": {k: round(v, 2) for k, v in self.sector_momentum.items()},
            "score": round(self.score, 1)
        }


@dataclass
class EconomicCalendarImpact:
    """Economic calendar analysis"""
    has_fed_event_48h: bool = False
    has_cpi_48h: bool = False
    has_nfp_48h: bool = False
    upcoming_events: List[str] = field(default_factory=list)
    event_risk_level: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    score_adjustment: float = 0.0  # Points to add/subtract
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_fed_event_48h": self.has_fed_event_48h,
            "has_cpi_48h": self.has_cpi_48h,
            "has_nfp_48h": self.has_nfp_48h,
            "upcoming_events": self.upcoming_events,
            "event_risk_level": self.event_risk_level,
            "score_adjustment": round(self.score_adjustment, 1)
        }


@dataclass
class MicroFactors:
    """Micro/intraday factors"""
    market_open: bool = False
    first_hour: bool = False
    last_hour: bool = False
    lunch_hour_avoid: bool = False
    opex_week: bool = False
    monday_effect: bool = False
    friday_effect: bool = False
    score_adjustment: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "market_open": self.market_open,
            "first_hour": self.first_hour,
            "last_hour": self.last_hour,
            "lunch_hour_avoid": self.lunch_hour_avoid,
            "opex_week": self.opex_week,
            "monday_effect": self.monday_effect,
            "friday_effect": self.friday_effect,
            "score_adjustment": round(self.score_adjustment, 1)
        }


@dataclass
class MacroMarketHealth:
    """Complete macro market health assessment"""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Individual analyses
    spy_health: IndexHealth = field(default_factory=lambda: IndexHealth(symbol="SPY"))
    qqq_health: IndexHealth = field(default_factory=lambda: IndexHealth(symbol="QQQ"))
    iwm_health: IndexHealth = field(default_factory=lambda: IndexHealth(symbol="IWM"))  # Small caps
    vix_analysis: VIXAnalysis = field(default_factory=VIXAnalysis)
    treasury_analysis: TreasuryAnalysis = field(default_factory=TreasuryAnalysis)
    dollar_analysis: DollarAnalysis = field(default_factory=DollarAnalysis)
    breadth_analysis: BreadthAnalysis = field(default_factory=BreadthAnalysis)
    sector_rotation: SectorRotation = field(default_factory=SectorRotation)
    economic_calendar: EconomicCalendarImpact = field(default_factory=EconomicCalendarImpact)
    micro_factors: MicroFactors = field(default_factory=MicroFactors)
    
    # Composite scores
    macro_score: float = 50.0  # 0-100 (higher = more favorable)
    regime: MacroRegime = MacroRegime.NEUTRAL
    score_adjustment: float = 0.0  # Points to add to opportunity scores (-30 to +30)
    position_size_multiplier: float = 1.0  # 0.25 to 1.25
    
    # Trading guidance
    should_trade: bool = True
    caution_reasons: List[str] = field(default_factory=list)
    favorable_reasons: List[str] = field(default_factory=list)
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "spy_health": self.spy_health.to_dict(),
            "qqq_health": self.qqq_health.to_dict(),
            "iwm_health": self.iwm_health.to_dict(),
            "vix_analysis": self.vix_analysis.to_dict(),
            "treasury_analysis": self.treasury_analysis.to_dict(),
            "dollar_analysis": self.dollar_analysis.to_dict(),
            "breadth_analysis": self.breadth_analysis.to_dict(),
            "sector_rotation": self.sector_rotation.to_dict(),
            "economic_calendar": self.economic_calendar.to_dict(),
            "micro_factors": self.micro_factors.to_dict(),
            "macro_score": round(self.macro_score, 1),
            "regime": self.regime.value,
            "score_adjustment": round(self.score_adjustment, 1),
            "position_size_multiplier": round(self.position_size_multiplier, 2),
            "should_trade": self.should_trade,
            "caution_reasons": self.caution_reasons,
            "favorable_reasons": self.favorable_reasons,
            "recommendation": self.recommendation
        }


class MacroMarketFilter:
    """
    Comprehensive macro market filter for trading decisions
    
    Analyzes multiple macro factors to provide:
    - Score adjustments for opportunity detection
    - Position sizing recommendations
    - Trade blocking during high-risk conditions
    """
    
    # Sector ETF mapping for rotation analysis
    SECTOR_ETFS = {
        "Technology": "XLK",
        "Healthcare": "XLV",
        "Financials": "XLF",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Energy": "XLE",
        "Utilities": "XLU",
        "Real Estate": "XLRE",
        "Materials": "XLB",
        "Industrials": "XLI",
        "Communications": "XLC"
    }
    
    # Defensive vs Growth sectors
    DEFENSIVE_SECTORS = ["Consumer Staples", "Utilities", "Healthcare"]
    GROWTH_SECTORS = ["Technology", "Consumer Discretionary", "Communications"]
    
    # VIX thresholds
    VIX_LOW = 12
    VIX_NORMAL = 18
    VIX_ELEVATED = 25
    VIX_HIGH = 35
    VIX_EXTREME = 50
    
    def __init__(self, cache_ttl_minutes: int = 15, enable_micro: bool = True):
        """
        Initialize macro market filter
        
        Args:
            cache_ttl_minutes: Cache duration for macro data
            enable_micro: Enable micro (intraday) factor analysis
        """
        self.cache_ttl_minutes = cache_ttl_minutes
        self.enable_micro = enable_micro
        
        # Cache
        self._cache: Optional[MacroMarketHealth] = None
        self._cache_time: Optional[datetime] = None
        self._cache_lock = threading.Lock()
        
        # Economic detector integration
        self.economic_detector = None
        if HAS_ECONOMIC_DETECTOR:
            try:
                from services.alert_system import get_alert_system
                self.economic_detector = EconomicDetector(get_alert_system())
            except Exception as e:
                logger.debug(f"Could not initialize EconomicDetector: {e}")
        
        logger.info("✅ MacroMarketFilter initialized")
        logger.info(f"   Cache TTL: {cache_ttl_minutes}min | Micro factors: {'ON' if enable_micro else 'OFF'}")
    
    # ================================================================
    # MAIN API
    # ================================================================
    
    def get_market_health(self, force_refresh: bool = False) -> MacroMarketHealth:
        """
        Get comprehensive macro market health assessment
        
        Args:
            force_refresh: Bypass cache and fetch fresh data
            
        Returns:
            MacroMarketHealth with all analyses
        """
        with self._cache_lock:
            # Check cache
            if not force_refresh and self._cache and self._cache_time:
                age = (datetime.now() - self._cache_time).total_seconds() / 60
                if age < self.cache_ttl_minutes:
                    logger.debug(f"Using cached macro health (age: {age:.1f}min)")
                    return self._cache
            
            # Perform fresh analysis
            logger.info("🌐 Analyzing macro market conditions...")
            start_time = time.time()
            
            health = MacroMarketHealth()
            
            try:
                # 1. Index Health (SPY, QQQ, IWM)
                health.spy_health = self._analyze_index("SPY")
                health.qqq_health = self._analyze_index("QQQ")
                health.iwm_health = self._analyze_index("IWM")
                
                # 2. VIX Analysis
                health.vix_analysis = self._analyze_vix()
                
                # 3. Treasury Yields
                health.treasury_analysis = self._analyze_treasury()
                
                # 4. Dollar Strength
                health.dollar_analysis = self._analyze_dollar()
                
                # 5. Sector Rotation
                health.sector_rotation = self._analyze_sector_rotation()
                
                # 6. Market Breadth (using proxy ETFs)
                health.breadth_analysis = self._analyze_breadth()
                
                # 7. Economic Calendar
                health.economic_calendar = self._analyze_economic_calendar()
                
                # 8. Micro Factors (if enabled)
                if self.enable_micro:
                    health.micro_factors = self._analyze_micro_factors()
                
                # Calculate composite score and regime
                self._calculate_composite_health(health)
                
                # Update cache
                self._cache = health
                self._cache_time = datetime.now()
                
                duration = time.time() - start_time
                logger.info(f"✅ Macro analysis complete in {duration:.1f}s")
                logger.info(f"   Score: {health.macro_score:.0f}/100 | Regime: {health.regime.value}")
                logger.info(f"   Adjustment: {health.score_adjustment:+.0f} pts | Size: {health.position_size_multiplier:.2f}x")
                
            except Exception as e:
                logger.error(f"Error in macro analysis: {e}", exc_info=True)
                # Return neutral health on error
                health.recommendation = f"⚠️ Macro analysis error: {e}"
            
            return health
    
    def get_score_adjustment(self, ticker: Optional[str] = None) -> Tuple[float, str]:
        """
        Get score adjustment for a ticker based on macro conditions
        
        Args:
            ticker: Optional ticker for sector-specific adjustments
            
        Returns:
            Tuple of (score_adjustment, reason)
        """
        health = self.get_market_health()
        
        # Base adjustment from macro score
        adjustment = health.score_adjustment
        reasons = []
        
        # Ticker-specific adjustments (if provided)
        if ticker:
            # Could add sector-specific logic here
            # e.g., if ticker is in tech sector and treasury yields rising, extra penalty
            pass
        
        # Compile reason
        if adjustment > 0:
            reason = f"Macro favorable ({adjustment:+.0f}): {', '.join(health.favorable_reasons[:2])}"
        elif adjustment < 0:
            reason = f"Macro caution ({adjustment:+.0f}): {', '.join(health.caution_reasons[:2])}"
        else:
            reason = "Macro neutral"
        
        return adjustment, reason
    
    def get_position_size_multiplier(self) -> Tuple[float, str]:
        """
        Get position size multiplier based on macro conditions
        
        Returns:
            Tuple of (multiplier, reason)
        """
        health = self.get_market_health()
        
        multiplier = health.position_size_multiplier
        
        if multiplier < 0.5:
            reason = f"Reduce size to {multiplier:.0%}: High macro risk"
        elif multiplier > 1.0:
            reason = f"Full size ({multiplier:.0%}): Favorable conditions"
        else:
            reason = f"Standard size ({multiplier:.0%})"
        
        return multiplier, reason
    
    def should_trade(self) -> Tuple[bool, List[str]]:
        """
        Check if trading should proceed based on macro conditions
        
        Returns:
            Tuple of (should_trade, blocking_reasons)
        """
        health = self.get_market_health()
        
        blocking_reasons = []
        
        # Check for blocking conditions
        if health.vix_analysis.current_level >= self.VIX_EXTREME:
            blocking_reasons.append(f"VIX at extreme level ({health.vix_analysis.current_level:.0f})")
        
        if health.economic_calendar.has_fed_event_48h:
            blocking_reasons.append("FOMC event within 48 hours")
        
        if health.regime == MacroRegime.CRISIS:
            blocking_reasons.append("Market in CRISIS regime")
        
        should_trade = len(blocking_reasons) == 0 and health.should_trade
        
        return should_trade, blocking_reasons
    
    def get_quick_summary(self) -> str:
        """Get a one-line summary of macro conditions"""
        health = self.get_market_health()
        
        # Emojis for quick visual
        regime_emoji = {
            MacroRegime.RISK_ON: "🟢",
            MacroRegime.NEUTRAL: "🟡",
            MacroRegime.RISK_OFF: "🟠",
            MacroRegime.CRISIS: "🔴"
        }
        
        vix_level = health.vix_analysis.current_level
        spy_trend = health.spy_health.trend.value.replace("_", " ").title()
        
        emoji = regime_emoji.get(health.regime, "⚪")
        
        summary = (
            f"{emoji} {health.regime.value.upper()} | "
            f"Score: {health.macro_score:.0f} | "
            f"SPY: {spy_trend} | "
            f"VIX: {vix_level:.1f} ({health.vix_analysis.regime}) | "
            f"Size: {health.position_size_multiplier:.0%}"
        )
        
        return summary
    
    # ================================================================
    # INDIVIDUAL ANALYSES
    # ================================================================
    
    def _analyze_index(self, symbol: str) -> IndexHealth:
        """Analyze major index health"""
        health = IndexHealth(symbol=symbol)
        
        try:
            # Fetch data
            data = yf.download(symbol, period="1y", progress=False, threads=False, auto_adjust=False)
            
            if data is None or data.empty:
                logger.debug(f"No data for {symbol}")
                return health
            
            # Handle multi-index columns
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            close = data['Close']
            current_price = float(close.iloc[-1])
            
            # Calculate SMAs
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            sma_200 = float(close.rolling(200).mean().iloc[-1])
            
            # 5-day momentum
            momentum_5d = ((current_price / float(close.iloc[-6])) - 1) * 100 if len(close) > 5 else 0
            
            # Determine trend
            above_20 = current_price > sma_20
            above_50 = current_price > sma_50
            above_200 = current_price > sma_200
            
            if above_20 and above_50 and above_200 and sma_50 > sma_200:
                trend = TrendDirection.STRONG_UP
                score = 90
            elif above_50 and above_200:
                trend = TrendDirection.UP
                score = 75
            elif above_200:
                trend = TrendDirection.NEUTRAL
                score = 55
            elif not above_50 and not above_200:
                if sma_50 < sma_200:
                    trend = TrendDirection.STRONG_DOWN
                    score = 15
                else:
                    trend = TrendDirection.DOWN
                    score = 30
            else:
                trend = TrendDirection.NEUTRAL
                score = 50
            
            health.current_price = current_price
            health.sma_20 = sma_20
            health.sma_50 = sma_50
            health.sma_200 = sma_200
            health.above_sma_20 = above_20
            health.above_sma_50 = above_50
            health.above_sma_200 = above_200
            health.trend = trend
            health.momentum_5d = momentum_5d
            health.score = score
            
            logger.debug(f"   {symbol}: {trend.value} (score: {score})")
            
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
        
        return health
    
    def _analyze_vix(self) -> VIXAnalysis:
        """Analyze VIX volatility"""
        analysis = VIXAnalysis()
        
        try:
            # Fetch VIX data
            data = yf.download("^VIX", period="1y", progress=False, threads=False, auto_adjust=False)
            
            if data is None or data.empty:
                logger.debug("No VIX data")
                return analysis
            
            # Handle multi-index
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            close = data['Close']
            current = float(close.iloc[-1])
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            
            # Calculate percentile rank (where VIX sits in 1-year range)
            min_val = float(close.min())
            max_val = float(close.max())
            percentile = ((current - min_val) / (max_val - min_val)) * 100 if max_val > min_val else 50
            
            # Determine regime
            if current < self.VIX_LOW:
                regime = "LOW"
                score = 20  # Low VIX = complacency, moderate score
            elif current < self.VIX_NORMAL:
                regime = "NORMAL"
                score = 30
            elif current < self.VIX_ELEVATED:
                regime = "ELEVATED"
                score = 55
            elif current < self.VIX_HIGH:
                regime = "HIGH"
                score = 75
            else:
                regime = "EXTREME"
                score = 95
            
            analysis.current_level = current
            analysis.sma_20 = sma_20
            analysis.percentile_rank = percentile
            analysis.regime = regime
            analysis.score = score
            
            logger.debug(f"   VIX: {current:.1f} ({regime})")
            
        except Exception as e:
            logger.debug(f"Error analyzing VIX: {e}")
        
        return analysis
    
    def _analyze_treasury(self) -> TreasuryAnalysis:
        """Analyze 10Y Treasury yields"""
        analysis = TreasuryAnalysis()
        
        try:
            # Use ^TNX for 10Y Treasury yield
            data = yf.download("^TNX", period="3mo", progress=False, threads=False, auto_adjust=False)
            
            if data is None or data.empty:
                logger.debug("No Treasury data")
                return analysis
            
            # Handle multi-index
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            close = data['Close']
            current = float(close.iloc[-1])
            month_ago = float(close.iloc[-22]) if len(close) > 22 else current
            
            change = current - month_ago
            
            # Determine trend
            if change > 0.25:  # 25+ bps increase
                trend = "RISING"
                impact = "NEGATIVE"  # Rising yields hurt growth stocks
                score = 65
            elif change < -0.25:  # 25+ bps decrease
                trend = "FALLING"
                impact = "POSITIVE"  # Falling yields help growth stocks
                score = 35
            else:
                trend = "STABLE"
                impact = "NEUTRAL"
                score = 50
            
            analysis.current_yield = current
            analysis.yield_1m_ago = month_ago
            analysis.yield_change = change
            analysis.trend = trend
            analysis.impact = impact
            analysis.score = score
            
            logger.debug(f"   10Y Treasury: {current:.2f}% ({trend})")
            
        except Exception as e:
            logger.debug(f"Error analyzing Treasury: {e}")
        
        return analysis
    
    def _analyze_dollar(self) -> DollarAnalysis:
        """Analyze dollar strength using UUP as DXY proxy"""
        analysis = DollarAnalysis()
        
        try:
            # Use UUP (Dollar Bull ETF) as DXY proxy
            data = yf.download("UUP", period="3mo", progress=False, threads=False, auto_adjust=False)
            
            if data is None or data.empty:
                logger.debug("No dollar data")
                return analysis
            
            # Handle multi-index
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            close = data['Close']
            current = float(close.iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            
            # Trend based on SMA
            if current > sma_50 * 1.02:  # 2% above
                trend = "STRENGTHENING"
                impact = "NEGATIVE"  # Strong dollar hurts multinationals
                score = 60
            elif current < sma_50 * 0.98:  # 2% below
                trend = "WEAKENING"
                impact = "POSITIVE"
                score = 40
            else:
                trend = "STABLE"
                impact = "NEUTRAL"
                score = 50
            
            analysis.current_level = current
            analysis.sma_50 = sma_50
            analysis.trend = trend
            analysis.impact = impact
            analysis.score = score
            
            logger.debug(f"   Dollar (UUP): {current:.2f} ({trend})")
            
        except Exception as e:
            logger.debug(f"Error analyzing dollar: {e}")
        
        return analysis
    
    def _analyze_sector_rotation(self) -> SectorRotation:
        """Analyze sector rotation patterns"""
        rotation = SectorRotation()
        
        try:
            sector_returns = {}
            
            # Get 1-month returns for each sector
            for sector, etf in self.SECTOR_ETFS.items():
                try:
                    data = yf.download(etf, period="1mo", progress=False, threads=False, auto_adjust=False)
                    if data is not None and not data.empty:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                        ret = ((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100
                        sector_returns[sector] = float(ret)
                except:
                    continue
            
            if not sector_returns:
                return rotation
            
            # Sort by performance
            sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
            
            rotation.leading_sectors = [s[0] for s in sorted_sectors[:3]]
            rotation.lagging_sectors = [s[0] for s in sorted_sectors[-3:]]
            rotation.sector_momentum = sector_returns
            
            # Determine rotation signal
            # If defensive sectors leading = RISK_OFF
            # If growth sectors leading = RISK_ON
            leading_set = set(rotation.leading_sectors)
            
            defensive_leading = len(leading_set & set(self.DEFENSIVE_SECTORS))
            growth_leading = len(leading_set & set(self.GROWTH_SECTORS))
            
            if growth_leading >= 2:
                rotation.rotation_signal = "RISK_ON"
                rotation.score = 70
            elif defensive_leading >= 2:
                rotation.rotation_signal = "RISK_OFF"
                rotation.score = 35
            else:
                rotation.rotation_signal = "NEUTRAL"
                rotation.score = 50
            
            logger.debug(f"   Sector Rotation: {rotation.rotation_signal}")
            
        except Exception as e:
            logger.debug(f"Error analyzing sector rotation: {e}")
        
        return rotation
    
    def _analyze_breadth(self) -> BreadthAnalysis:
        """Analyze market breadth using proxy ETFs"""
        analysis = BreadthAnalysis()
        
        try:
            # Use RSP (equal-weight S&P) vs SPY for breadth proxy
            rsp = yf.download("RSP", period="1mo", progress=False, threads=False, auto_adjust=False)
            spy = yf.download("SPY", period="1mo", progress=False, threads=False, auto_adjust=False)
            
            if rsp is not None and spy is not None and not rsp.empty and not spy.empty:
                # Handle multi-index
                if isinstance(rsp.columns, pd.MultiIndex):
                    rsp.columns = rsp.columns.get_level_values(0)
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                
                rsp_ret = ((rsp['Close'].iloc[-1] / rsp['Close'].iloc[0]) - 1) * 100
                spy_ret = ((spy['Close'].iloc[-1] / spy['Close'].iloc[0]) - 1) * 100
                
                # If RSP > SPY, breadth is good (smaller stocks participating)
                diff = float(rsp_ret - spy_ret)
                
                if diff > 1:  # RSP outperforming by 1%+
                    analysis.advance_decline_ratio = 1.5
                    analysis.pct_above_sma200 = 65
                    analysis.score = 70
                    analysis.breadth_thrust = True
                elif diff < -1:  # SPY outperforming (narrow breadth)
                    analysis.advance_decline_ratio = 0.7
                    analysis.pct_above_sma200 = 40
                    analysis.score = 35
                else:
                    analysis.advance_decline_ratio = 1.0
                    analysis.pct_above_sma200 = 50
                    analysis.score = 50
                
                logger.debug(f"   Breadth (RSP-SPY): {diff:.1f}%")
            
        except Exception as e:
            logger.debug(f"Error analyzing breadth: {e}")
        
        return analysis
    
    def _analyze_economic_calendar(self) -> EconomicCalendarImpact:
        """Analyze upcoming economic events"""
        impact = EconomicCalendarImpact()
        
        try:
            # Check for key events in next 48 hours
            now = datetime.now()
            
            # Check day of week for typical event patterns
            # FOMC: Usually Tues-Wed (8 times/year)
            # CPI: Usually mid-month (10th-15th)
            # NFP: First Friday of month
            
            day_of_week = now.weekday()
            day_of_month = now.day
            
            # Estimate FOMC (rough approximation)
            # Real implementation would use economic calendar API
            if day_of_week in [1, 2]:  # Tues/Wed
                # Check if it's a typical FOMC week (every ~6 weeks)
                week_of_year = now.isocalendar()[1]
                if week_of_year in [3, 11, 19, 25, 32, 38, 45, 50]:  # Approximate FOMC weeks
                    impact.has_fed_event_48h = True
                    impact.upcoming_events.append("FOMC Meeting (estimated)")
                    impact.score_adjustment = -15
            
            # CPI estimate (mid-month)
            if 10 <= day_of_month <= 14 and day_of_week <= 3:
                impact.has_cpi_48h = True
                impact.upcoming_events.append("CPI Release (estimated)")
                impact.score_adjustment -= 10
            
            # NFP (first Friday)
            if day_of_month <= 7 and day_of_week == 4:  # Friday
                impact.has_nfp_48h = True
                impact.upcoming_events.append("Non-Farm Payrolls (estimated)")
                impact.score_adjustment -= 8
            
            # Use economic detector if available
            if self.economic_detector:
                try:
                    events = self.economic_detector.get_finnhub_calendar(days_ahead=2)
                    for event in events:
                        event_name = event.get('event', '')
                        if 'FOMC' in event_name or 'Fed' in event_name:
                            impact.has_fed_event_48h = True
                            impact.upcoming_events.append(event_name)
                        elif 'CPI' in event_name:
                            impact.has_cpi_48h = True
                            impact.upcoming_events.append(event_name)
                        elif 'Payroll' in event_name or 'NFP' in event_name:
                            impact.has_nfp_48h = True
                            impact.upcoming_events.append(event_name)
                except:
                    pass
            
            # Determine risk level
            if impact.has_fed_event_48h:
                impact.event_risk_level = "CRITICAL"
            elif impact.has_cpi_48h or impact.has_nfp_48h:
                impact.event_risk_level = "HIGH"
            elif impact.upcoming_events:
                impact.event_risk_level = "MEDIUM"
            else:
                impact.event_risk_level = "LOW"
            
            logger.debug(f"   Economic Calendar: {impact.event_risk_level}")
            
        except Exception as e:
            logger.debug(f"Error analyzing economic calendar: {e}")
        
        return impact
    
    def _analyze_micro_factors(self) -> MicroFactors:
        """Analyze intraday/micro timing factors"""
        factors = MicroFactors()
        
        try:
            now = datetime.now()
            
            # Market hours (Eastern Time - approximate)
            # TODO: Add proper timezone handling
            hour = now.hour
            minute = now.minute
            day_of_week = now.weekday()
            day_of_month = now.day
            
            # Market open check (9:30 AM - 4:00 PM ET)
            factors.market_open = 9 <= hour < 16
            
            # First hour (9:30-10:30 AM) - high volatility
            factors.first_hour = 9 <= hour < 11
            
            # Last hour (3:00-4:00 PM) - increased volume
            factors.last_hour = hour == 15
            
            # Lunch hour (11:30 AM - 2:00 PM) - typically choppy
            factors.lunch_hour_avoid = 11 <= hour < 14
            
            # OpEx week (3rd Friday of month, and days around it)
            # 3rd Friday is between 15th and 21st
            if 14 <= day_of_month <= 21 and day_of_week >= 3:
                factors.opex_week = True
            
            # Monday effect (often lower returns historically)
            factors.monday_effect = day_of_week == 0
            
            # Friday effect (often profit-taking before weekend)
            factors.friday_effect = day_of_week == 4
            
            # Calculate adjustment
            adjustment = 0.0
            
            if factors.first_hour:
                adjustment += 3  # Good momentum opportunity
            if factors.lunch_hour_avoid:
                adjustment -= 5  # Avoid choppy period
            if factors.last_hour:
                adjustment += 2  # Good for momentum
            if factors.opex_week:
                adjustment -= 5  # Higher volatility/manipulation
            if factors.monday_effect:
                adjustment -= 2  # Slight caution
            if factors.friday_effect:
                adjustment -= 3  # Profit-taking risk
            
            factors.score_adjustment = adjustment
            
        except Exception as e:
            logger.debug(f"Error analyzing micro factors: {e}")
        
        return factors
    
    # ================================================================
    # COMPOSITE CALCULATION
    # ================================================================
    
    def _calculate_composite_health(self, health: MacroMarketHealth):
        """Calculate composite macro score and regime"""
        
        caution_reasons = []
        favorable_reasons = []
        
        # Weight individual scores
        weights = {
            'spy': 0.20,
            'qqq': 0.15,
            'iwm': 0.10,
            'vix': 0.20,
            'treasury': 0.10,
            'dollar': 0.05,
            'breadth': 0.10,
            'rotation': 0.10
        }
        
        # Invert VIX score (high VIX = low score for trading)
        vix_trading_score = 100 - health.vix_analysis.score
        
        # Calculate weighted score
        composite = (
            health.spy_health.score * weights['spy'] +
            health.qqq_health.score * weights['qqq'] +
            health.iwm_health.score * weights['iwm'] +
            vix_trading_score * weights['vix'] +
            (100 - health.treasury_analysis.score) * weights['treasury'] +
            (100 - health.dollar_analysis.score) * weights['dollar'] +
            health.breadth_analysis.score * weights['breadth'] +
            health.sector_rotation.score * weights['rotation']
        )
        
        # Add economic calendar adjustment
        composite += health.economic_calendar.score_adjustment
        
        # Add micro factor adjustment
        composite += health.micro_factors.score_adjustment
        
        # Clamp to valid range
        composite = max(0, min(100, composite))
        health.macro_score = composite
        
        # Determine regime
        if composite >= 70:
            health.regime = MacroRegime.RISK_ON
        elif composite >= 50:
            health.regime = MacroRegime.NEUTRAL
        elif composite >= 30:
            health.regime = MacroRegime.RISK_OFF
        else:
            health.regime = MacroRegime.CRISIS
        
        # Calculate score adjustment (-30 to +30)
        # 50 = neutral, above adds points, below subtracts
        health.score_adjustment = (composite - 50) * 0.6
        health.score_adjustment = max(-30, min(30, health.score_adjustment))
        
        # Calculate position size multiplier (0.25 to 1.25)
        if health.regime == MacroRegime.CRISIS:
            health.position_size_multiplier = 0.25
        elif health.regime == MacroRegime.RISK_OFF:
            health.position_size_multiplier = 0.50
        elif health.regime == MacroRegime.NEUTRAL:
            health.position_size_multiplier = 0.80
        else:
            # Scale between 1.0 and 1.25 based on how far above 70
            bonus = min(0.25, (composite - 70) * 0.01)
            health.position_size_multiplier = 1.0 + bonus
        
        # Collect reasons
        # VIX
        if health.vix_analysis.current_level >= self.VIX_HIGH:
            caution_reasons.append(f"VIX elevated at {health.vix_analysis.current_level:.0f}")
        elif health.vix_analysis.current_level <= self.VIX_LOW:
            favorable_reasons.append("VIX low - favorable risk environment")
        
        # SPY trend
        if health.spy_health.trend in [TrendDirection.STRONG_UP, TrendDirection.UP]:
            favorable_reasons.append(f"SPY in {health.spy_health.trend.value}")
        elif health.spy_health.trend in [TrendDirection.STRONG_DOWN, TrendDirection.DOWN]:
            caution_reasons.append(f"SPY in {health.spy_health.trend.value}")
        
        # Treasury
        if health.treasury_analysis.trend == "RISING":
            caution_reasons.append("10Y yields rising - pressure on growth")
        elif health.treasury_analysis.trend == "FALLING":
            favorable_reasons.append("10Y yields falling - tailwind for growth")
        
        # Breadth
        if health.breadth_analysis.breadth_thrust:
            favorable_reasons.append("Strong market breadth")
        elif health.breadth_analysis.score < 40:
            caution_reasons.append("Narrow market breadth")
        
        # Sector rotation
        if health.sector_rotation.rotation_signal == "RISK_ON":
            favorable_reasons.append("Sector rotation favors growth")
        elif health.sector_rotation.rotation_signal == "RISK_OFF":
            caution_reasons.append("Defensive rotation in progress")
        
        # Economic events
        if health.economic_calendar.has_fed_event_48h:
            caution_reasons.append("FOMC event within 48h")
        if health.economic_calendar.has_cpi_48h:
            caution_reasons.append("CPI release within 48h")
        
        health.caution_reasons = caution_reasons
        health.favorable_reasons = favorable_reasons
        
        # Determine if should trade
        health.should_trade = (
            health.regime != MacroRegime.CRISIS and
            health.vix_analysis.current_level < self.VIX_EXTREME and
            not health.economic_calendar.has_fed_event_48h
        )
        
        # Generate recommendation
        if health.regime == MacroRegime.RISK_ON:
            health.recommendation = "✅ FAVORABLE - Full position sizes, normal trading"
        elif health.regime == MacroRegime.NEUTRAL:
            health.recommendation = "🟡 NEUTRAL - Standard sizes, be selective"
        elif health.regime == MacroRegime.RISK_OFF:
            health.recommendation = "🟠 CAUTION - Reduce sizes, focus on highest conviction"
        else:
            health.recommendation = "🔴 AVOID - High-risk environment, wait for clarity"


# Singleton instance
_macro_filter_instance: Optional[MacroMarketFilter] = None


def get_macro_market_filter(
    cache_ttl_minutes: int = 15,
    enable_micro: bool = True
) -> MacroMarketFilter:
    """
    Get singleton instance of MacroMarketFilter
    
    Args:
        cache_ttl_minutes: Cache duration
        enable_micro: Enable micro timing factors
        
    Returns:
        MacroMarketFilter instance
    """
    global _macro_filter_instance
    
    if _macro_filter_instance is None:
        _macro_filter_instance = MacroMarketFilter(
            cache_ttl_minutes=cache_ttl_minutes,
            enable_micro=enable_micro
        )
    
    return _macro_filter_instance


if __name__ == "__main__":
    # Test the macro filter
    print("=" * 80)
    print("MACRO MARKET FILTER TEST")
    print("=" * 80)
    
    filter = get_macro_market_filter()
    
    # Get full health analysis
    health = filter.get_market_health(force_refresh=True)
    
    print(f"\n{filter.get_quick_summary()}")
    print(f"\nRecommendation: {health.recommendation}")
    
    print("\n--- Details ---")
    print(f"SPY: {health.spy_health.trend.value} (${health.spy_health.current_price:.2f})")
    print(f"QQQ: {health.qqq_health.trend.value} (${health.qqq_health.current_price:.2f})")
    print(f"VIX: {health.vix_analysis.current_level:.1f} ({health.vix_analysis.regime})")
    print(f"10Y: {health.treasury_analysis.current_yield:.2f}% ({health.treasury_analysis.trend})")
    print(f"Dollar: {health.dollar_analysis.trend}")
    print(f"Sector Rotation: {health.sector_rotation.rotation_signal}")
    
    if health.caution_reasons:
        print(f"\n⚠️ Caution: {', '.join(health.caution_reasons)}")
    if health.favorable_reasons:
        print(f"✅ Favorable: {', '.join(health.favorable_reasons)}")
    
    # Test trading check
    should_trade, blocking = filter.should_trade()
    print(f"\nShould Trade: {should_trade}")
    if blocking:
        print(f"Blocking: {blocking}")
    
    # Test score adjustment
    adjustment, reason = filter.get_score_adjustment()
    print(f"\nScore Adjustment: {adjustment:+.0f} pts")
    print(f"Reason: {reason}")

