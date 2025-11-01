"""
Dynamic Strategy Selector
Analyzes ticker characteristics and recommends the best trading strategy
"""

import logging
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

from models.analysis import StockAnalysis

logger = logging.getLogger(__name__)


class TradingStrategy(Enum):
    """Available trading strategies"""
    SCALPING = "SCALPING"           # 2% profit, 1% stop, minutes-hours
    WARRIOR_SCALPING = "WARRIOR_SCALPING"  # Gap & Go, 2% profit, 1% stop, 9:30-10:00 AM
    SWING = "SWING"                 # 10% profit, 4% stop, 1-5 days
    MEAN_REVERSION = "MEAN_REVERSION"  # 5-10% profit, 3% stop, 1-3 days
    BREAKOUT = "BREAKOUT"           # 10-50% profit, 8% stop, 1-7 days
    OPTIONS_WHEEL = "OPTIONS_WHEEL"  # 2-4% monthly, 30-45 days
    BUY_HOLD = "BUY_HOLD"           # Long-term position


@dataclass
class StrategyRecommendation:
    """Strategy recommendation with reasoning"""
    ticker: str
    strategy: TradingStrategy
    confidence: float  # 0-100
    reasoning: str
    
    # Trade parameters
    position_size_pct: float  # % of capital for this trade
    profit_target_pct: float
    stop_loss_pct: float
    hold_time_estimate: str  # "Minutes-Hours", "1-5 Days", etc.
    
    # Supporting data
    analysis_score: float
    key_indicators: Dict[str, float]


class DynamicStrategySelector:
    """
    Analyzes stock characteristics and recommends optimal trading strategy
    """
    
    @staticmethod
    def select_strategy(analysis: StockAnalysis) -> StrategyRecommendation:
        """
        Select the best strategy based on stock analysis
        
        Args:
            analysis: StockAnalysis object with all technical/fundamental data
            
        Returns:
            StrategyRecommendation with strategy and parameters
        """
        ticker = analysis.ticker
        
        # Calculate strategy scores
        scores = {
            TradingStrategy.SCALPING: DynamicStrategySelector._score_scalping(analysis),
            TradingStrategy.WARRIOR_SCALPING: DynamicStrategySelector._score_warrior_scalping(analysis),
            TradingStrategy.SWING: DynamicStrategySelector._score_swing(analysis),
            TradingStrategy.MEAN_REVERSION: DynamicStrategySelector._score_mean_reversion(analysis),
            TradingStrategy.BREAKOUT: DynamicStrategySelector._score_breakout(analysis),
            TradingStrategy.OPTIONS_WHEEL: DynamicStrategySelector._score_options(analysis),
            TradingStrategy.BUY_HOLD: DynamicStrategySelector._score_buy_hold(analysis),
        }
        
        # Get best strategy
        best_strategy = max(scores.items(), key=lambda x: x[1]['score'])
        strategy, strategy_data = best_strategy
        
        # Build recommendation
        recommendation = StrategyRecommendation(
            ticker=ticker,
            strategy=strategy,
            confidence=strategy_data['score'],
            reasoning=strategy_data['reasoning'],
            position_size_pct=strategy_data['position_size'],
            profit_target_pct=strategy_data['profit_target'],
            stop_loss_pct=strategy_data['stop_loss'],
            hold_time_estimate=strategy_data['hold_time'],
            analysis_score=analysis.confidence_score,
            key_indicators=strategy_data['indicators']
        )
        
        logger.info(f"{ticker}: Recommended {strategy.value} (confidence: {strategy_data['score']:.1f}%)")
        logger.debug(f"  Reasoning: {strategy_data['reasoning']}")
        
        return recommendation
    
    @staticmethod
    def _score_scalping(analysis: StockAnalysis) -> Dict:
        """Score suitability for scalping"""
        score = 40  # Base score boost for active strategy
        reasoning = []
        
        # High volatility (good for quick moves)
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        if volume_ratio > 2.0:
            score += 35  # Increased from 30
            reasoning.append(f"High volume {volume_ratio:.1f}x avg")
        elif volume_ratio > 1.5:
            score += 25  # Increased from 20
            reasoning.append(f"Elevated volume {volume_ratio:.1f}x")
        
        # Price movement (need volatility)
        if abs(analysis.change_pct) > 5:
            score += 30  # New tier for very high movement
            reasoning.append(f"Very strong move {analysis.change_pct:+.1f}%")
        elif abs(analysis.change_pct) > 3:
            score += 25
            reasoning.append(f"Strong intraday move {analysis.change_pct:+.1f}%")
        elif abs(analysis.change_pct) > 1.5:
            score += 15
            reasoning.append(f"Moderate move {analysis.change_pct:+.1f}%")
        
        # Avoid slow movers
        if abs(analysis.change_pct) < 0.5:
            score -= 30  # Increased penalty
            reasoning.append("Low volatility - not ideal for scalping")
        
        # Liquidity (need tight spreads)
        if analysis.volume > 5_000_000:
            score += 15
            reasoning.append("High liquidity")
        elif analysis.volume < 1_000_000:
            score -= 15
            reasoning.append("Low liquidity")
        
        # Trend doesn't matter for scalping
        if "STRONG" in analysis.trend:
            score += 10
            reasoning.append("Strong trend helps")
        
        # Entropy (prefer structured markets for scalping)
        if analysis.entropy is not None and analysis.entropy < 50:
            score += 10
            reasoning.append("Low entropy - structured market")
        elif analysis.entropy is not None and analysis.entropy > 70:
            score -= 10
            reasoning.append("High entropy - noisy market")
        
        return {
            'score': max(0, min(100, score)),
            'reasoning': " | ".join(reasoning) if reasoning else "Standard scalping conditions",
            'position_size': 3.0,  # 3% per scalp
            'profit_target': 2.0,
            'stop_loss': 1.0,
            'hold_time': "Minutes-Hours",
            'indicators': {
                'volume_ratio': volume_ratio,
                'change_pct': analysis.change_pct,
                'entropy': analysis.entropy or 0
            }
        }
    
    @staticmethod
    def _score_warrior_scalping(analysis: StockAnalysis) -> Dict:
        """Score suitability for Warrior Trading Gap & Go scalping"""
        score = 35  # Base score
        reasoning = []
        
        # Price filter: $2-$20 (Warrior Trading requirement)
        if 2.0 <= analysis.price <= 20.0:
            score += 20
            reasoning.append(f"Price ${analysis.price:.2f} in range")
        else:
            score -= 30
            reasoning.append(f"Price ${analysis.price:.2f} outside $2-$20 range")
        
        # Gap requirement: Need 4-10% premarket gap (use change_pct as proxy)
        # In real implementation, would check premarket gap specifically
        if 4.0 <= abs(analysis.change_pct) <= 10.0:
            score += 35
            reasoning.append(f"Gap {analysis.change_pct:+.1f}% in 4-10% range")
        elif abs(analysis.change_pct) > 10.0:
            score -= 10
            reasoning.append(f"Gap {analysis.change_pct:+.1f}% too large (>10%)")
        elif abs(analysis.change_pct) < 4.0:
            score -= 15
            reasoning.append(f"Gap {analysis.change_pct:+.1f}% too small (<4%)")
        
        # Volume filter: 2-3x average volume (critical for Warrior Trading)
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        if 2.0 <= volume_ratio <= 3.0:
            score += 30
            reasoning.append(f"Volume {volume_ratio:.1f}x (ideal range)")
        elif volume_ratio > 3.0:
            score += 20  # Still good, just higher
            reasoning.append(f"Volume {volume_ratio:.1f}x (above ideal)")
        elif volume_ratio < 2.0:
            score -= 25
            reasoning.append(f"Volume {volume_ratio:.1f}x (insufficient)")
        
        # Momentum requirement: Strong intraday move
        if abs(analysis.change_pct) > 5.0:
            score += 15
            reasoning.append("Strong momentum")
        
        # Liquidity: Need high volume for tight spreads
        if analysis.volume > 1_000_000:
            score += 10
            reasoning.append("High liquidity")
        elif analysis.volume < 500_000:
            score -= 15
            reasoning.append("Low liquidity")
        
        # Trading window: Best during 9:30-10:00 AM (would check time in real implementation)
        # For now, just check if we're in morning (using current hour as proxy)
        from datetime import datetime
        current_hour = datetime.now().hour
        if 9 <= current_hour < 10:
            score += 10
            reasoning.append("Morning momentum window")
        elif current_hour >= 10:
            score -= 5
            reasoning.append("Past optimal window (10 AM)")
        
        return {
            'score': max(0, min(100, score)),
            'reasoning': " | ".join(reasoning) if reasoning else "Warrior Trading conditions",
            'position_size': 3.0,  # 3% per trade
            'profit_target': 2.0,  # 2% profit target
            'stop_loss': 1.0,  # 1% stop loss
            'hold_time': "9:30-10:00 AM",
            'indicators': {
                'gap_pct': analysis.change_pct,
                'volume_ratio': volume_ratio,
                'price': analysis.price
            }
        }
    
    @staticmethod
    def _score_swing(analysis: StockAnalysis) -> Dict:
        """Score suitability for swing trading"""
        score = 30  # Base score boost for active strategy
        reasoning = []
        
        # Strong trend (essential for swings)
        if "STRONG UPTREND" in analysis.trend:
            score += 40  # Increased from 35
            reasoning.append("Strong uptrend - excellent for swings")
        elif "UPTREND" in analysis.trend:
            score += 30  # Increased from 25
            reasoning.append("Uptrend detected")
        elif "SIDEWAYS" in analysis.trend:
            score -= 15  # Increased penalty
            reasoning.append("Sideways trend - not ideal")
        elif "DOWNTREND" in analysis.trend:
            score -= 30  # Increased penalty
            reasoning.append("Downtrend - avoid swings")
        
        # EMA reclaim (key swing entry signal)
        if analysis.ema_reclaim:
            score += 25
            reasoning.append("EMA reclaim - quality entry")
        
        # Power zone (strong setup)
        if analysis.ema_power_zone:
            score += 20
            reasoning.append("Power zone active")
        
        # RSI not overbought (room to run)
        if 40 <= analysis.rsi <= 65:
            score += 15
            reasoning.append(f"RSI {analysis.rsi:.0f} - healthy range")
        elif analysis.rsi > 70:
            score -= 15
            reasoning.append(f"RSI {analysis.rsi:.0f} - overbought")
        
        # Volume confirmation
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        if volume_ratio > 1.5:
            score += 10
            reasoning.append("Volume confirms move")
        
        # Timeframe alignment (multi-timeframe confirmation)
        if analysis.timeframe_alignment and analysis.timeframe_alignment.get('aligned'):
            score += 15
            reasoning.append("Timeframes aligned")
        
        return {
            'score': max(0, min(100, score)),
            'reasoning': " | ".join(reasoning) if reasoning else "Standard swing conditions",
            'position_size': 5.0,  # 5% per swing
            'profit_target': 10.0,
            'stop_loss': 4.0,
            'hold_time': "1-5 Days",
            'indicators': {
                'trend': analysis.trend,
                'rsi': analysis.rsi,
                'ema_reclaim': analysis.ema_reclaim,
                'power_zone': analysis.ema_power_zone
            }
        }
    
    @staticmethod
    def _score_mean_reversion(analysis: StockAnalysis) -> Dict:
        """Score suitability for mean reversion"""
        score = 20  # Base score for active strategy
        reasoning = []
        
        # Oversold (key mean reversion signal)
        if analysis.rsi < 25:
            score += 50  # Increased from 40
            reasoning.append(f"Extreme oversold RSI {analysis.rsi:.0f}")
        elif analysis.rsi < 30:
            score += 40  # Increased from 30
            reasoning.append(f"Oversold RSI {analysis.rsi:.0f}")
        elif analysis.rsi > 75:  # More extreme threshold
            score += 30  # Increased from 20
            reasoning.append(f"Extreme overbought RSI {analysis.rsi:.0f}")
        elif analysis.rsi > 70:
            score += 20
            reasoning.append(f"Overbought RSI {analysis.rsi:.0f}")
        else:
            score -= 30  # Increased penalty
            reasoning.append(f"RSI {analysis.rsi:.0f} - not extreme")
        
        # Volume spike (confirms panic/euphoria)
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        if volume_ratio > 2.5:
            score += 25
            reasoning.append(f"Volume spike {volume_ratio:.1f}x")
        elif volume_ratio > 1.5:
            score += 15
        
        # Price down (for oversold bounce)
        if analysis.change_pct < -5 and analysis.rsi < 30:
            score += 20
            reasoning.append(f"Down {analysis.change_pct:.1f}% + oversold")
        elif analysis.change_pct < -3:
            score += 10
        
        # Historical mean-reverting behavior (would need to calculate)
        # For now, assume certain tickers are known mean-reverters
        mean_reverting_tickers = ['AMC', 'BB', 'TLRY', 'NOK', 'SNDL']
        if analysis.ticker in mean_reverting_tickers:
            score += 15
            reasoning.append("Known mean-reverting ticker")
        
        return {
            'score': max(0, min(100, score)),
            'reasoning': " | ".join(reasoning) if reasoning else "Mean reversion setup",
            'position_size': 4.0,  # 4% per trade
            'profit_target': 7.0,
            'stop_loss': 3.0,
            'hold_time': "1-3 Days",
            'indicators': {
                'rsi': analysis.rsi,
                'volume_ratio': volume_ratio,
                'change_pct': analysis.change_pct
            }
        }
    
    @staticmethod
    def _score_breakout(analysis: StockAnalysis) -> Dict:
        """Score suitability for breakout trading"""
        score = 25  # Base score for active strategy
        reasoning = []
        
        # Volume surge (key breakout signal)
        volume_ratio = analysis.volume / analysis.avg_volume if analysis.avg_volume > 0 else 0
        if volume_ratio > 5.0:
            score += 45  # Increased from 35
            reasoning.append(f"Massive volume {volume_ratio:.1f}x")
        elif volume_ratio > 3.0:
            score += 30  # Increased from 25
            reasoning.append(f"Strong volume {volume_ratio:.1f}x")
        elif volume_ratio < 1.5:
            score -= 30  # Increased penalty
            reasoning.append("Insufficient volume for breakout")
        
        # Strong price movement
        if analysis.change_pct > 10:
            score += 35  # Increased from 30
            reasoning.append(f"Explosive move +{analysis.change_pct:.1f}%")
        elif analysis.change_pct > 5:
            score += 25  # Increased from 20
            reasoning.append(f"Strong move +{analysis.change_pct:.1f}%")
        elif analysis.change_pct < 2:
            score -= 25  # Increased penalty
            reasoning.append("Weak price action")
        
        # Catalyst (news, earnings)
        if analysis.catalysts and len(analysis.catalysts) > 0:
            score += 20
            reasoning.append(f"Catalyst detected: {analysis.catalysts[0]}")
        
        # Sentiment (positive news helps breakouts)
        if analysis.sentiment_score > 0.3:
            score += 15
            reasoning.append("Positive sentiment")
        
        # EMA reclaim (confirmation)
        if analysis.ema_reclaim:
            score += 10
            reasoning.append("EMA reclaim confirms")
        
        return {
            'score': max(0, min(100, score)),
            'reasoning': " | ".join(reasoning) if reasoning else "Breakout setup",
            'position_size': 3.0,  # 3% per breakout (higher risk)
            'profit_target': 15.0,
            'stop_loss': 6.0,
            'hold_time': "1-7 Days",
            'indicators': {
                'volume_ratio': volume_ratio,
                'change_pct': analysis.change_pct,
                'sentiment': analysis.sentiment_score
            }
        }
    
    @staticmethod
    def _score_options(analysis: StockAnalysis) -> Dict:
        """Score suitability for options wheel strategy"""
        score = 0
        reasoning = []
        
        # High IV rank (essential for premium selling)
        # BUT: If IV rank is suspiciously high (>95%), likely bad data - heavily penalize
        if analysis.iv_rank is not None:
            if analysis.iv_rank > 95:
                score -= 50  # Likely bad/stale IV data
                reasoning.append(f"IV rank {analysis.iv_rank:.0f}% suspicious - likely bad data")
            elif analysis.iv_rank > 60:
                score += 35
                reasoning.append(f"High IV rank {analysis.iv_rank:.0f}%")
            elif analysis.iv_rank > 40:
                score += 20
                reasoning.append(f"Good IV rank {analysis.iv_rank:.0f}%")
            elif analysis.iv_rank < 30:
                score -= 20
                reasoning.append(f"Low IV rank {analysis.iv_rank:.0f}%")
        else:
            score -= 50  # Heavy penalty for no IV data
            reasoning.append("No IV data - can't trade options")
        
        # Stable/uptrending (want to own the stock)
        if "UPTREND" in analysis.trend or "SIDEWAYS" in analysis.trend:
            score += 20
            reasoning.append(f"Good trend for ownership: {analysis.trend}")
        elif "DOWNTREND" in analysis.trend:
            score -= 15
            reasoning.append("Downtrend - risky for wheel")
        
        # Price range (affordability for cash-secured puts)
        if 10 <= analysis.price <= 100:
            score += 15
            reasoning.append(f"Good price range ${analysis.price:.2f}")
        elif analysis.price > 200:
            score -= 10
            reasoning.append(f"Expensive ${analysis.price:.2f}")
        elif analysis.price < 5:
            score -= 10
            reasoning.append(f"Too cheap ${analysis.price:.2f}")
        
        # Liquidity (need good option volume)
        if analysis.volume > 2_000_000:
            score += 15
            reasoning.append("High liquidity - good option spreads")
        
        # Quality tickers (would you want to own it?)
        quality_tickers = ['AAPL', 'MSFT', 'AMD', 'SOFI', 'PLTR', 'WFC', 'JPM', 'SPY', 'QQQ']
        if analysis.ticker in quality_tickers:
            score += 15
            reasoning.append("Quality ticker for wheel")
        
        return {
            'score': max(0, min(100, score)),
            'reasoning': " | ".join(reasoning) if reasoning else "Options wheel candidate",
            'position_size': 8.0,  # 8% per option position
            'profit_target': 3.0,  # 3% monthly
            'stop_loss': 5.0,
            'hold_time': "30-45 Days",
            'indicators': {
                'iv_rank': analysis.iv_rank or 0,
                'price': analysis.price,
                'trend': analysis.trend
            }
        }
    
    @staticmethod
    def _score_buy_hold(analysis: StockAnalysis) -> Dict:
        """Score suitability for buy and hold"""
        score = 0
        reasoning = []
        
        # Strong fundamentals (would need to add fundamental data)
        # For now, use trend and quality as proxy
        
        # Very strong trend
        if "STRONG UPTREND" in analysis.trend:
            score += 30
            reasoning.append("Strong uptrend - quality hold")
        elif "UPTREND" in analysis.trend:
            score += 20
        else:
            score -= 20
            reasoning.append("Poor trend for holding")
        
        # High confidence score
        if analysis.confidence_score > 80:
            score += 25
            reasoning.append(f"High confidence {analysis.confidence_score:.0f}")
        elif analysis.confidence_score > 70:
            score += 15
        
        # Sector relative strength (outperforming)
        if analysis.sector_rs and analysis.sector_rs.get('rs_score', 0) > 60:
            score += 20
            reasoning.append("Outperforming sector")
        
        # Quality tickers
        buy_hold_tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL']  # Reduced list - only true long-term holds
        if analysis.ticker in buy_hold_tickers:
            score += 15
            reasoning.append("Quality growth stock")
        
        # Positive sentiment
        if analysis.sentiment_score > 0.5:
            score += 10
            reasoning.append("Strong positive sentiment")
        
        # BUT - buy & hold usually has lower score than active strategies
        # Only recommend if all conditions are exceptional
        score = score * 0.6  # 40% penalty vs active strategies (was 20%)
        
        return {
            'score': max(0, min(100, score)),
            'reasoning': " | ".join(reasoning) if reasoning else "Buy & hold candidate",
            'position_size': 10.0,  # 10% for longer holds
            'profit_target': 30.0,  # 30% long-term target
            'stop_loss': 12.0,  # Wider stop
            'hold_time': "Weeks-Months",
            'indicators': {
                'trend': analysis.trend,
                'confidence': analysis.confidence_score,
                'sentiment': analysis.sentiment_score
            }
        }


# Quick helper for auto-trader integration
def get_strategy_for_ticker(ticker: str, trading_style: str = "AUTO") -> Optional[StrategyRecommendation]:
    """
    Convenience function to analyze a ticker and get strategy recommendation
    
    Args:
        ticker: Stock ticker symbol
        trading_style: Trading style preference (AUTO lets analyzer decide)
        
    Returns:
        StrategyRecommendation or None if analysis fails
    """
    from analyzers.comprehensive import ComprehensiveAnalyzer
    
    try:
        # Get full analysis
        analysis = ComprehensiveAnalyzer.analyze_stock(ticker, trading_style)
        if not analysis:
            return None
        
        # Get strategy recommendation
        selector = DynamicStrategySelector()
        recommendation = selector.select_strategy(analysis)
        
        return recommendation
        
    except Exception as e:
        logger.error(f"Error getting strategy for {ticker}: {e}")
        return None
