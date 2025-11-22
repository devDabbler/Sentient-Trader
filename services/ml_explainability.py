"""
ML Explainability Module

Provides feature importance and SHAP-like explanations for ML predictions
to help understand why ML scores diverge from technical scores.
"""

from loguru import logger
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np



class MLExplainer:
    """Explains ML predictions and feature importance"""
    
    # Feature importance weights (tuned based on alpha factor significance)
    FEATURE_WEIGHTS = {
        # Momentum features (high impact)
        'momentum_20d': 0.15,
        'momentum_60d': 0.12,
        'momentum_5d': 0.08,
        
        # Volatility features (high impact)
        'volatility_20d': 0.10,
        'annual_volatility': 0.08,
        
        # Technical features
        'rsi_14': 0.08,
        'macd_histogram': 0.07,
        'price_position_20d': 0.06,
        
        # Volume features
        'volume_ratio': 0.07,
        'volume_trend': 0.05,
        
        # Trend features
        'ma_crossover': 0.06,
        'trend_strength': 0.05,
        
        # Other
        'leverage_ratio': 0.03,
    }
    
    @staticmethod
    def calculate_feature_importance(features: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """
        Calculate feature importance scores and impact direction.
        
        Args:
            features: Dict of feature name -> value
            
        Returns:
            List of (feature_name, importance_score, direction, explanation)
        """
        importance_list = []
        
        for feature_name, value in features.items():
            if feature_name in MLExplainer.FEATURE_WEIGHTS:
                base_importance = MLExplainer.FEATURE_WEIGHTS[feature_name]
                
                # Determine direction and impact
                direction, explanation = MLExplainer._interpret_feature(feature_name, value)
                
                # Scale importance by magnitude
                magnitude = abs(value) if isinstance(value, (int, float)) else 1.0
                scaled_importance = base_importance * min(magnitude, 2.0)  # Cap at 2x
                
                importance_list.append((
                    feature_name,
                    round(scaled_importance * 100, 2),
                    direction,
                    explanation
                ))
        
        # Sort by importance (descending)
        importance_list.sort(key=lambda x: x[1], reverse=True)
        
        return importance_list
    
    @staticmethod
    def _interpret_feature(feature_name: str, value: float) -> Tuple[str, str]:
        """
        Interpret feature value and provide human-readable explanation.
        
        Returns:
            (direction, explanation)
            direction: 'POSITIVE', 'NEGATIVE', 'NEUTRAL'
        """
        if 'momentum' in feature_name:
            if value > 0.10:  # 10%+ momentum
                return ('POSITIVE', f'Strong positive momentum ({value*100:.1f}%)')
            elif value < -0.10:
                return ('NEGATIVE', f'Negative momentum ({value*100:.1f}%)')
            else:
                return ('NEUTRAL', f'Weak momentum ({value*100:.1f}%)')
        
        elif 'volatility' in feature_name:
            if value > 0.40:  # High volatility
                return ('NEGATIVE', f'High volatility ({value*100:.1f}%) = risk')
            elif value < 0.15:
                return ('POSITIVE', f'Low volatility ({value*100:.1f}%) = stable')
            else:
                return ('NEUTRAL', f'Moderate volatility ({value*100:.1f}%)')
        
        elif feature_name == 'rsi_14':
            if value < 30:
                return ('POSITIVE', f'Oversold RSI ({value:.1f}) = potential bounce')
            elif value > 70:
                return ('NEGATIVE', f'Overbought RSI ({value:.1f}) = potential pullback')
            else:
                return ('NEUTRAL', f'Neutral RSI ({value:.1f})')
        
        elif 'volume' in feature_name:
            if value > 2.0:
                return ('POSITIVE', f'Strong volume ({value:.1f}x average)')
            elif value < 0.5:
                return ('NEGATIVE', f'Weak volume ({value:.1f}x average)')
            else:
                return ('NEUTRAL', f'Normal volume ({value:.1f}x average)')
        
        elif 'leverage' in feature_name or 'debt' in feature_name:
            if value > 2.0:
                return ('NEGATIVE', f'High leverage ({value:.1f}x) = financial risk')
            elif value < 0.5:
                return ('POSITIVE', f'Low leverage ({value:.1f}x) = strong balance sheet')
            else:
                return ('NEUTRAL', f'Moderate leverage ({value:.1f}x)')
        
        elif 'macd' in feature_name:
            if value > 0:
                return ('POSITIVE', f'Bullish MACD ({value:.2f})')
            elif value < 0:
                return ('NEGATIVE', f'Bearish MACD ({value:.2f})')
            else:
                return ('NEUTRAL', 'Neutral MACD')
        
        else:
            # Generic interpretation
            if value > 0:
                return ('POSITIVE', f'{feature_name}: {value:.2f}')
            elif value < 0:
                return ('NEGATIVE', f'{feature_name}: {value:.2f}')
            else:
                return ('NEUTRAL', f'{feature_name}: {value:.2f}')
    
    @staticmethod
    def explain_score_divergence(technical_score: float, ml_score: float, 
                                 features: Dict[str, float]) -> Dict[str, any]:
        """
        Explain why technical and ML scores diverge.
        
        Args:
            technical_score: Technical analysis score (0-100)
            ml_score: ML prediction score (0-100)
            features: Feature dictionary
            
        Returns:
            Dict with divergence explanation
        """
        divergence = abs(technical_score - ml_score)
        
        if divergence < 15:
            agreement = "STRONG"
            explanation = "Technical and ML signals are aligned"
        elif divergence < 30:
            agreement = "MODERATE"
            explanation = "Some disagreement between technical and ML signals"
        else:
            agreement = "WEAK"
            explanation = "Significant disagreement between technical and ML signals"
        
        # Get top features driving ML score
        feature_importance = MLExplainer.calculate_feature_importance(features)
        top_features = feature_importance[:5]  # Top 5
        
        # Identify negative drivers if ML score is low
        negative_drivers = [
            f for f in top_features 
            if f[2] == 'NEGATIVE'
        ]
        
        positive_drivers = [
            f for f in top_features 
            if f[2] == 'POSITIVE'
        ]
        
        # Build explanation
        reasons = []
        
        if ml_score < technical_score:
            reasons.append(f"ML score ({ml_score:.0f}) is lower than technical ({technical_score:.0f})")
            if negative_drivers:
                reasons.append("ML detected negative factors:")
                for feature_name, importance, direction, explanation in negative_drivers[:3]:
                    reasons.append(f"  • {explanation} (importance: {importance:.1f}%)")
        else:
            reasons.append(f"ML score ({ml_score:.0f}) is higher than technical ({technical_score:.0f})")
            if positive_drivers:
                reasons.append("ML detected positive factors:")
                for feature_name, importance, direction, explanation in positive_drivers[:3]:
                    reasons.append(f"  • {explanation} (importance: {importance:.1f}%)")
        
        return {
            'divergence': round(divergence, 1),
            'agreement': agreement,
            'summary': explanation,
            'reasons': reasons,
            'top_features': top_features,
            'negative_drivers': negative_drivers,
            'positive_drivers': positive_drivers
        }
    
    @staticmethod
    def generate_trade_recommendation(technical_score: float, ml_score: float,
                                     sentiment_score: float, divergence_info: Dict) -> str:
        """
        Generate trade recommendation based on score agreement.
        
        Args:
            technical_score: Technical score (0-100)
            ml_score: ML score (0-100)
            sentiment_score: Sentiment score (0-100)
            divergence_info: Output from explain_score_divergence
            
        Returns:
            Trade recommendation string
        """
        agreement = divergence_info['agreement']
        avg_score = (technical_score + ml_score + sentiment_score) / 3
        
        if agreement == "STRONG":
            if avg_score >= 75:
                return "✅ HIGH CONFIDENCE - All signals aligned (bullish)"
            elif avg_score <= 35:
                return "❌ AVOID - All signals aligned (bearish)"
            else:
                return "⚠️ WAIT - Signals aligned but not strong enough"
        
        elif agreement == "MODERATE":
            if ml_score < 40 and technical_score > 70:
                return "⚠️ CAUTION - Technical bullish but ML sees long-term risk. Consider small position or short-term scalp only."
            elif ml_score > 70 and technical_score < 40:
                return "⚠️ WAIT - ML bullish but technicals weak. Wait for better entry."
            else:
                return "⚠️ NEUTRAL - Mixed signals, prefer high conviction setups"
        
        else:  # WEAK agreement
            return "❌ AVOID - Significant disagreement between signals. High uncertainty."


class OptionsLiquidityChecker:
    """Check options liquidity before recommending premium selling"""
    
    # Liquidity thresholds
    MIN_OPEN_INTEREST = 100  # Minimum OI per strike
    MIN_DAILY_VOLUME = 50  # Minimum daily volume per strike
    MAX_BID_ASK_SPREAD_PCT = 10.0  # Max 10% spread
    MIN_STRIKES_WITH_LIQUIDITY = 3  # Need at least 3 liquid strikes
    
    @staticmethod
    def check_options_liquidity(options_data: pd.DataFrame, option_type: str = 'call') -> Dict[str, any]:
        """
        Check options chain liquidity.
        
        Args:
            options_data: DataFrame with columns: strike, bid, ask, volume, openInterest
            option_type: 'call' or 'put'
            
        Returns:
            Dict with liquidity assessment
        """
        try:
            if options_data.empty:
                return {
                    'is_liquid': False,
                    'reason': 'No options data available',
                    'liquid_strikes': [],
                    'recommendation': 'AVOID - No options liquidity'
                }
            
            liquid_strikes = []
            
            for idx, row in options_data.iterrows():
                strike = row.get('strike', 0)
                bid = row.get('bid', 0)
                ask = row.get('ask', 0)
                volume = row.get('volume', 0)
                oi = row.get('openInterest', 0)
                
                # Calculate bid-ask spread
                mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                spread_pct = ((ask - bid) / mid * 100) if mid > 0 else 999
                
                # Check liquidity criteria
                is_liquid = (
                    oi >= OptionsLiquidityChecker.MIN_OPEN_INTEREST and
                    volume >= OptionsLiquidityChecker.MIN_DAILY_VOLUME and
                    spread_pct <= OptionsLiquidityChecker.MAX_BID_ASK_SPREAD_PCT
                )
                
                if is_liquid:
                    liquid_strikes.append({
                        'strike': strike,
                        'bid': bid,
                        'ask': ask,
                        'mid': round(mid, 2),
                        'spread_pct': round(spread_pct, 2),
                        'volume': int(volume),
                        'open_interest': int(oi)
                    })
            
            is_liquid = len(liquid_strikes) >= OptionsLiquidityChecker.MIN_STRIKES_WITH_LIQUIDITY
            
            if is_liquid:
                recommendation = f"✅ LIQUID - {len(liquid_strikes)} strikes meet liquidity criteria"
                reason = f"Sufficient liquidity with {len(liquid_strikes)} liquid strikes"
            else:
                recommendation = f"❌ ILLIQUID - Only {len(liquid_strikes)} liquid strikes (need {OptionsLiquidityChecker.MIN_STRIKES_WITH_LIQUIDITY}+)"
                reason = "Insufficient options liquidity"
            
            return {
                'is_liquid': is_liquid,
                'reason': reason,
                'liquid_strikes': liquid_strikes,
                'liquid_strike_count': len(liquid_strikes),
                'recommendation': recommendation,
                'option_type': option_type
            }
            
        except Exception as e:
            logger.error(f"Error checking options liquidity: {e}")
            return {
                'is_liquid': False,
                'reason': f'Error checking liquidity: {str(e)}',
                'liquid_strikes': [],
                'recommendation': 'AVOID - Unable to verify liquidity'
            }
    
    @staticmethod
    def recommend_premium_strategy(iv_rank: float, liquidity_check: Dict) -> Dict[str, str]:
        """
        Recommend premium selling strategy based on IV Rank and liquidity.
        
        Args:
            iv_rank: IV Rank (0-100)
            liquidity_check: Output from check_options_liquidity
            
        Returns:
            Dict with strategy recommendation
        """
        if not liquidity_check['is_liquid']:
            return {
                'strategy': 'NONE',
                'recommendation': '❌ AVOID OPTIONS - Illiquid options chain',
                'reason': liquidity_check['reason']
            }
        
        if iv_rank >= 75:
            return {
                'strategy': 'SELL_PREMIUM',
                'recommendation': '✅ SELL PREMIUM - High IV Rank + Liquid Options',
                'reason': f'IV Rank {iv_rank}% is very high, great for selling premium',
                'suggested_strategies': [
                    'Cash-Secured Puts (if bullish)',
                    'Covered Calls (if holding stock)',
                    'Credit Spreads (defined risk)',
                    'Iron Condors (if expecting range-bound)'
                ]
            }
        elif iv_rank >= 50:
            return {
                'strategy': 'SELECTIVE_SELL',
                'recommendation': '⚠️ SELECTIVE PREMIUM SELLING - Moderate IV',
                'reason': f'IV Rank {iv_rank}% is elevated but not extreme',
                'suggested_strategies': [
                    'Credit Spreads (better risk/reward)',
                    'Covered Calls if holding'
                ]
            }
        else:
            return {
                'strategy': 'BUY_OPTIONS',
                'recommendation': '✅ BUY OPTIONS - Low IV (cheap premium)',
                'reason': f'IV Rank {iv_rank}% is low, options are cheap',
                'suggested_strategies': [
                    'Long Calls (if bullish)',
                    'Long Puts (if bearish)',
                    'Debit Spreads (defined risk)'
                ]
            }

