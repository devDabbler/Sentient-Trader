"""
Market State Analyzer

Provides entropy-based market state analysis with clear explanations.
Helps determine if market conditions are favorable for trading.
"""

import logging
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np
from math import log2

logger = logging.getLogger(__name__)


class MarketStateAnalyzer:
    """Analyze market state using entropy and other metrics"""
    
    # Entropy thresholds
    ENTROPY_THRESHOLDS = {
        'HIGHLY_STRUCTURED': 25,
        'STRUCTURED': 40,
        'MIXED': 60,
        'CHAOTIC': 75,
    }
    
    @staticmethod
    def calculate_shannon_entropy(prices: pd.Series, bins: int = 10, window: int = 20) -> float:
        """
        Calculate Shannon entropy of price returns.
        
        Lower entropy = more predictable/structured
        Higher entropy = more random/chaotic
        
        Args:
            prices: Price series
            bins: Number of bins for histogram
            window: Lookback window
            
        Returns:
            Entropy score (0-100)
        """
        try:
            # Get recent returns
            returns = prices.pct_change().dropna()
            recent_returns = returns.tail(window)
            
            if len(recent_returns) < window // 2:
                return 50.0  # Default if insufficient data
            
            # Create histogram
            counts, _ = np.histogram(recent_returns, bins=bins)
            
            # Calculate probabilities
            probabilities = counts / counts.sum()
            probabilities = probabilities[probabilities > 0]  # Remove zeros
            
            # Shannon entropy: -sum(p * log2(p))
            entropy = -np.sum(probabilities * np.log2(probabilities))
            
            # Normalize to 0-100 scale (max entropy for 10 bins ≈ 3.32)
            max_entropy = log2(bins)
            normalized_entropy = (entropy / max_entropy) * 100
            
            return min(100, max(0, normalized_entropy))
            
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 50.0
    
    @staticmethod
    def interpret_entropy(entropy_score: float, window: int = 20) -> Dict[str, any]:
        """
        Interpret entropy score and provide trading guidance.
        
        Args:
            entropy_score: Entropy score (0-100)
            window: Window used for calculation
            
        Returns:
            Dict with interpretation
        """
        if entropy_score < MarketStateAnalyzer.ENTROPY_THRESHOLDS['HIGHLY_STRUCTURED']:
            state = "HIGHLY_STRUCTURED"
            emoji = "✅"
            interpretation = "Very predictable price patterns"
            trading_edge = "EXCELLENT"
            recommendation = "✅ TRADE FAVORABLE - Strong patterns suggest systematic strategies work well"
            details = (
                f"Price movements over the last {window} days show low randomness. "
                "This indicates trending or mean-reverting patterns that technical analysis can exploit."
            )
        
        elif entropy_score < MarketStateAnalyzer.ENTROPY_THRESHOLDS['STRUCTURED']:
            state = "STRUCTURED"
            emoji = "✅"
            interpretation = "Moderately predictable patterns"
            trading_edge = "GOOD"
            recommendation = "✅ TRADE FAVORABLE - Patterns present, systematic entry/exit likely to work"
            details = (
                f"Price movements over the last {window} days show some structure. "
                "Technical patterns and momentum strategies should work with proper risk management."
            )
        
        elif entropy_score < MarketStateAnalyzer.ENTROPY_THRESHOLDS['MIXED']:
            state = "MIXED"
            emoji = "⚠️"
            interpretation = "Mixed signals, some randomness"
            trading_edge = "NEUTRAL"
            recommendation = "⚠️ NEUTRAL CONDITIONS - Be selective, use tight stops"
            details = (
                f"Price movements over the last {window} days show moderate randomness. "
                "Patterns are less reliable. Focus on high-conviction setups only."
            )
        
        elif entropy_score < MarketStateAnalyzer.ENTROPY_THRESHOLDS['CHAOTIC']:
            state = "CHAOTIC"
            emoji = "⚠️"
            interpretation = "High randomness, unpredictable"
            trading_edge = "POOR"
            recommendation = "⚠️ UNFAVORABLE - High randomness, reduce position sizing"
            details = (
                f"Price movements over the last {window} days show high randomness. "
                "Technical patterns are unreliable. Consider sitting out or trading very small."
            )
        
        else:
            state = "EXTREME_CHAOS"
            emoji = "❌"
            interpretation = "Extremely random, no discernible pattern"
            trading_edge = "NONE"
            recommendation = "❌ AVOID TRADING - Market behavior is too random for systematic strategies"
            details = (
                f"Price movements over the last {window} days are nearly random. "
                "Avoid trading until clearer patterns emerge."
            )
        
        return {
            'entropy_score': round(entropy_score, 1),
            'state': state,
            'emoji': emoji,
            'interpretation': interpretation,
            'trading_edge': trading_edge,
            'recommendation': recommendation,
            'details': details,
            'window': window,
            'data_source': 'Daily price returns',
            'calculation': 'Shannon entropy of price return distribution'
        }
    
    @staticmethod
    def calculate_market_regime(prices: pd.Series, volume: pd.Series = None) -> Dict[str, any]:
        """
        Determine market regime (trending, ranging, volatile).
        
        Args:
            prices: Price series
            volume: Volume series (optional)
            
        Returns:
            Dict with regime analysis
        """
        try:
            # Calculate indicators
            returns = prices.pct_change()
            
            # Trend strength (ADX-like)
            window = 20
            rolling_max = prices.rolling(window).max()
            rolling_min = prices.rolling(window).min()
            price_range = rolling_max - rolling_min
            
            current_price = prices.iloc[-1]
            range_position = ((current_price - rolling_min.iloc[-1]) / price_range.iloc[-1]) if price_range.iloc[-1] > 0 else 0.5
            
            # Volatility
            volatility = returns.rolling(window).std().iloc[-1] * np.sqrt(252)
            
            # Determine regime
            if range_position > 0.7:
                regime = "UPTREND"
                emoji = "📈"
                bias = "BULLISH"
            elif range_position < 0.3:
                regime = "DOWNTREND"
                emoji = "📉"
                bias = "BEARISH"
            else:
                regime = "RANGING"
                emoji = "↔️"
                bias = "NEUTRAL"
            
            # Volatility state
            if volatility > 0.50:  # 50%+ annualized
                vol_state = "HIGH_VOLATILITY"
            elif volatility > 0.30:
                vol_state = "ELEVATED_VOLATILITY"
            elif volatility > 0.20:
                vol_state = "NORMAL_VOLATILITY"
            else:
                vol_state = "LOW_VOLATILITY"
            
            return {
                'regime': regime,
                'emoji': emoji,
                'bias': bias,
                'volatility': round(volatility * 100, 1),
                'volatility_state': vol_state,
                'range_position': round(range_position * 100, 1),
                'summary': f"{emoji} {regime} with {vol_state.replace('_', ' ').title()}"
            }
            
        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return {
                'regime': 'UNKNOWN',
                'emoji': '❓',
                'bias': 'NEUTRAL',
                'volatility': 0,
                'volatility_state': 'UNKNOWN',
                'summary': 'Unable to determine market regime'
            }
    
    @staticmethod
    def analyze_market_state(ticker: str, prices: pd.Series, volume: pd.Series = None) -> Dict[str, any]:
        """
        Complete market state analysis.
        
        Args:
            ticker: Stock ticker
            prices: Price series
            volume: Volume series (optional)
            
        Returns:
            Dict with complete market state analysis
        """
        # Calculate entropy
        entropy_score = MarketStateAnalyzer.calculate_shannon_entropy(prices)
        entropy_interpretation = MarketStateAnalyzer.interpret_entropy(entropy_score)
        
        # Calculate regime
        regime = MarketStateAnalyzer.calculate_market_regime(prices, volume)
        
        # Combined recommendation
        if entropy_interpretation['trading_edge'] in ['EXCELLENT', 'GOOD'] and regime['bias'] != 'NEUTRAL':
            combined_rec = f"✅ FAVORABLE CONDITIONS - {regime['regime']} with structured patterns"
        elif entropy_interpretation['trading_edge'] == 'NEUTRAL':
            combined_rec = f"⚠️ MIXED CONDITIONS - {regime['regime']} but patterns less reliable"
        else:
            combined_rec = f"❌ UNFAVORABLE CONDITIONS - High randomness, avoid or reduce size"
        
        return {
            'ticker': ticker,
            'entropy': entropy_interpretation,
            'regime': regime,
            'combined_recommendation': combined_rec,
            'favorable_for_trading': entropy_interpretation['trading_edge'] in ['EXCELLENT', 'GOOD']
        }


class TradingStyleConfig:
    """Standardized stop/target configurations for different trading styles"""
    
    CONFIGS = {
        'WARRIOR_SCALPING': {
            'name': 'Warrior Trading Gap & Go',
            'stop_method': 'PERCENTAGE',
            'stop_pct': 1.0,  # 1% stop (low of breakout candle)
            'target_pct': 2.0,  # 2% target (scale out)
            'max_holding_time': '30 minutes',
            'time_window': '9:30-10:00 AM ET',
            'position_size': '2-3% of capital',
            'risk_per_trade': 1.0,
            'description': 'Tight stops for premarket gappers, quick 2% scalps'
        },
        'PENNY_STOCK_SCALP': {
            'name': 'Penny Stock Scalp',
            'stop_method': 'ATR',
            'atr_multiplier': 1.0,
            'stop_pct_max': 8.0,  # Cap at 8%
            'target_rr': 2.0,  # 2:1 R/R
            'max_holding_time': '1-3 days',
            'position_size': '1-2% of capital',
            'risk_per_trade': 1.0,
            'description': 'ATR-based stops capped at 8%, tight risk management'
        },
        'PENNY_STOCK_SWING': {
            'name': 'Penny Stock Swing',
            'stop_method': 'ATR',
            'atr_multiplier': 1.5,
            'stop_pct_max': 10.0,  # Cap at 10%
            'target_rr': 2.5,  # 2.5:1 R/R
            'max_holding_time': '5-15 days',
            'position_size': '2-3% of capital',
            'risk_per_trade': 1.5,
            'description': 'Medium ATR stops for multi-day swings'
        },
        'PENNY_STOCK_POSITION': {
            'name': 'Penny Stock Position',
            'stop_method': 'ATR',
            'atr_multiplier': 2.0,
            'stop_pct_max': 12.0,  # Cap at 12%
            'target_rr': 3.0,  # 3:1 R/R
            'max_holding_time': '15-30 days',
            'position_size': '3-5% of capital',
            'risk_per_trade': 2.0,
            'description': 'Wider ATR stops for position trades, still capped for penny stocks'
        },
        'OPTIONS_PREMIUM_SELL': {
            'name': 'Options Premium Selling',
            'stop_method': 'PERCENTAGE_OF_PREMIUM',
            'stop_pct': 200.0,  # 2x premium (close at 200% loss)
            'target_pct': 50.0,  # Close at 50% profit
            'max_holding_time': 'Until 21 DTE or target',
            'position_size': '5-10% of capital per trade',
            'risk_per_trade': 2.0,
            'requirements': ['IV Rank > 50', 'OI > 100', 'Bid-Ask < 10%'],
            'description': 'Premium selling requires high IV and liquid options'
        }
    }
    
    @classmethod
    def get_config(cls, strategy: str) -> Dict:
        """Get configuration for a strategy"""
        return cls.CONFIGS.get(strategy.upper(), cls.CONFIGS['PENNY_STOCK_SWING'])
    
    @classmethod
    def display_all_configs(cls) -> str:
        """Display all trading style configurations"""
        output = "\n=== STANDARDIZED TRADING STYLES & STOP LOGIC ===\n\n"
        
        for strategy, config in cls.CONFIGS.items():
            output += f"📊 {config['name']}\n"
            output += f"   Strategy: {strategy}\n"
            
            if config['stop_method'] == 'ATR':
                output += f"   Stop: {config['atr_multiplier']}x ATR (Max: {config['stop_pct_max']}%)\n"
                output += f"   Target: {config['target_rr']}:1 Risk/Reward\n"
            elif config['stop_method'] == 'PERCENTAGE':
                output += f"   Stop: {config['stop_pct']}%\n"
                output += f"   Target: {config['target_pct']}%\n"
            else:
                output += f"   Stop: {config.get('stop_pct', 'N/A')}% of premium\n"
                output += f"   Target: {config.get('target_pct', 'N/A')}% profit\n"
            
            output += f"   Holding Time: {config['max_holding_time']}\n"
            output += f"   Position Size: {config['position_size']}\n"
            output += f"   Description: {config['description']}\n"
            
            if 'requirements' in config:
                output += f"   Requirements: {', '.join(config['requirements'])}\n"
            
            output += "\n"
        
        return output

